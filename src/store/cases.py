"""
Case persistence — SQLite, with the queryable parts kept out of the blob.

Cases are stored twice over: once as a JSON document holding the complete state,
and once as a handful of indexed columns holding the fields the hospital
dashboard sorts and filters on. That is a deliberate duplication. The dashboard
needs to answer "show me everything awaiting review, most urgent first" without
deserialising every case in the table, and the case itself is a nested structure
that would be miserable to normalise into rows for no benefit.

The audit trail is separate and append-only. Every decision, every escalation,
every clinician override is written there and never updated, because the
question a reviewer asks afterwards is not "what does this case say now" but
"what did the system do, and when". A record that can be edited cannot answer
that.

Nothing here is on the request path for triage. If persistence fails the intake
still works and the failure is logged - losing the record of a case is bad, but
refusing to triage the patient in front of you is worse.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import RUNTIME_DIR
from ..core.case import Case, CaseStatus, Turn
from ..core.schema import Complaint, Fact, FactSource, Urgency

logger = logging.getLogger(__name__)

DEFAULT_DB = RUNTIME_DIR / "vita.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id       TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    status        TEXT NOT NULL,
    complaint     TEXT NOT NULL,
    urgency       TEXT NOT NULL DEFAULT '',
    urgency_rank  INTEGER NOT NULL DEFAULT -1,
    department    TEXT NOT NULL DEFAULT '',
    review        INTEGER NOT NULL DEFAULT 0,
    language      TEXT NOT NULL DEFAULT 'en',
    document      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cases_queue
    ON cases (status, urgency_rank DESC, created_at ASC);

CREATE TABLE IF NOT EXISTS audit (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id   TEXT NOT NULL,
    at        TEXT NOT NULL,
    action    TEXT NOT NULL,
    actor     TEXT NOT NULL DEFAULT 'system',
    detail    TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_audit_case ON audit (case_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CaseStore:
    """Reads and writes cases. Safe to share across request threads."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("case store ready at %s", self.path)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- writing ---------------------------------------------------------

    def save(self, case: Case) -> None:
        document = json.dumps(case.as_dict(full=True), default=str)
        urgency = case.effective_urgency
        rank = Urgency(urgency).rank if urgency else -1

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO cases (case_id, created_at, updated_at, status, complaint,
                                   urgency, urgency_rank, department, review, language, document)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(case_id) DO UPDATE SET
                    updated_at=excluded.updated_at, status=excluded.status,
                    complaint=excluded.complaint, urgency=excluded.urgency,
                    urgency_rank=excluded.urgency_rank, department=excluded.department,
                    review=excluded.review, language=excluded.language,
                    document=excluded.document
                """,
                (
                    case.case_id,
                    case.created_at,
                    case.updated_at,
                    case.status.value,
                    case.complaint.value,
                    urgency,
                    rank,
                    case.decision.department if case.decision else "",
                    int(bool(case.decision and case.decision.requires_human_review)),
                    case.language,
                    document,
                ),
            )
            self._conn.commit()

    def audit(self, case_id: str, action: str, *, actor: str = "system", detail: str = "") -> None:
        """Append to the trail. Never updates, never deletes."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit (case_id, at, action, actor, detail) VALUES (?,?,?,?,?)",
                (case_id, _now(), action, actor, detail),
            )
            self._conn.commit()

    # -- reading ---------------------------------------------------------

    def get(self, case_id: str) -> Case | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT document FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        if row is None:
            return None
        try:
            return _case_from_document(json.loads(row["document"]))
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error("case %s could not be read back: %s", case_id, exc)
            return None

    def queue(self, *, status: str = "", limit: int = 200) -> list[dict[str, Any]]:
        """The dashboard queue: most urgent first, then longest waiting.

        Ordering is by urgency and then by arrival, never by capacity. What a
        department can currently take is shown to the clinician alongside this
        list; it is not allowed to reorder it.
        """
        sql = (
            "SELECT case_id, created_at, updated_at, status, complaint, urgency, "
            "department, review, language FROM cases "
        )
        params: list[Any] = []
        if status:
            sql += "WHERE status = ? "
            params.append(status)
        sql += "ORDER BY urgency_rank DESC, created_at ASC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def trail(self, case_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT at, action, actor, detail FROM audit WHERE case_id = ? ORDER BY id",
                (case_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT urgency, COUNT(*) AS n FROM cases WHERE urgency <> '' GROUP BY urgency"
            ).fetchall()
            awaiting = self._conn.execute(
                "SELECT COUNT(*) AS n FROM cases WHERE status = ?",
                (CaseStatus.AWAITING_REVIEW.value,),
            ).fetchone()
            total = self._conn.execute("SELECT COUNT(*) AS n FROM cases").fetchone()

        counts = {r["urgency"]: r["n"] for r in rows}
        counts["awaiting_review"] = awaiting["n"] if awaiting else 0
        counts["total"] = total["n"] if total else 0
        return counts

    def find_prior_visit(self, *, complaint: str, within_hours: int, exclude: str) -> dict[str, Any] | None:
        """Look for an earlier visit with the same complaint.

        This is what makes rule GEN-07 possible: a patient returning inside 72
        hours with an unresolved complaint is a recognised marker of a missed or
        deteriorating condition, and it can only be spotted by something with a
        memory across visits.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT case_id, created_at, urgency, department FROM cases
                WHERE complaint = ? AND case_id <> ?
                ORDER BY created_at DESC LIMIT 5
                """,
                (complaint, exclude),
            ).fetchall()

        now = datetime.now(timezone.utc)
        for row in rows:
            try:
                seen = datetime.fromisoformat(row["created_at"])
            except ValueError:
                continue
            if seen.tzinfo is None:
                seen = seen.replace(tzinfo=timezone.utc)
            hours = (now - seen).total_seconds() / 3600.0
            if 0 <= hours <= within_hours:
                return {**dict(row), "hours_ago": round(hours, 1)}
        return None


# ---------------------------------------------------------------------------
# Rehydration
# ---------------------------------------------------------------------------


def _fact_from_dict(raw: dict[str, Any]) -> Fact:
    return Fact(
        key=raw["key"],
        value=raw.get("value"),
        source=FactSource(raw.get("source", FactSource.PATIENT_VERBATIM.value)),
        turn=int(raw.get("turn", 0)),
        verbatim=raw.get("verbatim", ""),
        language=raw.get("language", "en"),
        confidence=float(raw.get("confidence", 1.0)),
        agent=raw.get("agent", ""),
        recorded_at=raw.get("recorded_at", _now()),
    )


def _case_from_document(raw: dict[str, Any]) -> Case:
    """Rebuild a case from its stored document.

    The decision is deliberately not rehydrated. It is derived state - the rule
    engine can always produce it again from the facts - and reconstructing an
    object graph of rule evaluations from JSON would create a second code path
    that could disagree with the engine. The stored copy of the decision is kept
    in the document for display; anything that needs to act on it re-evaluates.
    """
    case = Case(
        case_id=raw["case_id"],
        created_at=raw.get("created_at", _now()),
        updated_at=raw.get("updated_at", _now()),
        language=raw.get("language", "en"),
        complaint=Complaint(raw.get("complaint", Complaint.UNDETERMINED.value)),
        status=CaseStatus(raw.get("status", CaseStatus.INTAKE.value)),
    )
    case.facts = {k: _fact_from_dict(v) for k, v in (raw.get("facts") or {}).items()}
    case.history = [_fact_from_dict(f) for f in (raw.get("history") or [])]
    case.turns = [
        Turn(
            role=t.get("role", "patient"),
            text=t.get("text", ""),
            language=t.get("language", "en"),
            at=t.get("at", _now()),
            asked_about=t.get("asked_about", ""),
            driven_by_rule=t.get("driven_by_rule", ""),
        )
        for t in (raw.get("turns") or [])
    ]
    case.pending_fact = raw.get("pending_fact", "")
    case.pending_rule = raw.get("pending_rule", "")
    case.red_flags = list(raw.get("red_flags") or [])
    case.out_of_scope = bool(raw.get("out_of_scope", False))

    override = raw.get("override") or {}
    case.override_urgency = override.get("urgency", "")
    case.override_reason = override.get("reason", "")
    case.override_by = override.get("by", "")
    case.override_at = override.get("at", "")

    return case
