"""
Persistence for the approval queue and for admissions.

Requests are append-and-amend: a row is written when the planner asks for
something and updated once when a human decides. Nothing else touches it. The
decision, who made it and when are kept alongside the original reasoning, so the
record shows both what was proposed and what was done about it - including the
proposals that were turned down, which are the interesting ones when anybody
asks later how much the system was trusted.

Admissions are separate rows created only when an admission request is approved.
A room is never assigned by VITA; the approving clinician chooses one from the
beds this store reports as free.
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
from ..core.requests import Request, RequestKind, RequestStatus

logger = logging.getLogger(__name__)

DEFAULT_DB = RUNTIME_DIR / "vita.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id    TEXT PRIMARY KEY,
    case_id       TEXT NOT NULL,
    patient_id    TEXT NOT NULL DEFAULT '',
    kind          TEXT NOT NULL,
    summary       TEXT NOT NULL DEFAULT '',
    reasoning     TEXT NOT NULL DEFAULT '',
    payload       TEXT NOT NULL DEFAULT '{}',
    evidence      TEXT NOT NULL DEFAULT '[]',
    status        TEXT NOT NULL DEFAULT 'pending',
    created_at    TEXT NOT NULL,
    decided_at    TEXT NOT NULL DEFAULT '',
    decided_by    TEXT NOT NULL DEFAULT '',
    decision_note TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_requests_pending
    ON requests (status, created_at ASC);

CREATE TABLE IF NOT EXISTS admissions (
    admission_id TEXT PRIMARY KEY,
    case_id      TEXT NOT NULL,
    patient_id   TEXT NOT NULL DEFAULT '',
    patient_name TEXT NOT NULL DEFAULT '',
    room_id      TEXT NOT NULL,
    department   TEXT NOT NULL DEFAULT '',
    reason       TEXT NOT NULL DEFAULT '',
    admitted_by  TEXT NOT NULL DEFAULT '',
    admitted_at  TEXT NOT NULL,
    discharged_at TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_admissions_room ON admissions (room_id, discharged_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RequestStore:
    """The approval queue, and the admissions that come out of it."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("request store ready at %s", self.path)

    # -- requests --------------------------------------------------------

    def add(self, request: Request) -> Request:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO requests (request_id, case_id, patient_id, kind, summary,
                                      reasoning, payload, evidence, status, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    request.request_id,
                    request.case_id,
                    request.patient_id,
                    request.kind.value,
                    request.summary,
                    request.reasoning,
                    json.dumps(request.payload, default=str),
                    json.dumps(request.evidence),
                    request.status.value,
                    request.created_at,
                ),
            )
            self._conn.commit()
        logger.info(
            "request %s raised: %s for case %s", request.request_id, request.kind.value, request.case_id
        )
        return request

    def decide(self, request_id: str, *, approved: bool, by: str, note: str = "") -> Request | None:
        request = self.get(request_id)
        if request is None or not request.pending:
            return None

        if approved:
            request.approve(by, note)
        else:
            request.reject(by, note)

        with self._lock:
            self._conn.execute(
                """
                UPDATE requests SET status=?, decided_at=?, decided_by=?, decision_note=?
                WHERE request_id=?
                """,
                (request.status.value, request.decided_at, request.decided_by,
                 request.decision_note, request.request_id),
            )
            self._conn.commit()
        return request

    def get(self, request_id: str) -> Request | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM requests WHERE request_id = ?", (request_id,)
            ).fetchone()
        return _request_from_row(row) if row else None

    def list(self, *, status: str = "", case_id: str = "", limit: int = 200) -> list[Request]:
        sql = "SELECT * FROM requests"
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if case_id:
            clauses.append("case_id = ?")
            params.append(case_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        # Pending first, then newest: the queue is a to-do list, not a log.
        sql += " ORDER BY (status = 'pending') DESC, created_at DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_request_from_row(r) for r in rows]

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM requests GROUP BY status"
            ).fetchall()
        counts = {r["status"]: r["n"] for r in rows}
        counts.setdefault("pending", 0)
        return counts

    # -- admissions ------------------------------------------------------

    def admit(
        self,
        *,
        admission_id: str,
        case_id: str,
        patient_id: str,
        patient_name: str,
        room_id: str,
        department: str,
        reason: str,
        admitted_by: str,
    ) -> dict[str, Any]:
        record = {
            "admission_id": admission_id,
            "case_id": case_id,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "room_id": room_id,
            "department": department,
            "reason": reason,
            "admitted_by": admitted_by,
            "admitted_at": _now(),
            "discharged_at": "",
        }
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO admissions (admission_id, case_id, patient_id, patient_name,
                                        room_id, department, reason, admitted_by, admitted_at, discharged_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                tuple(record.values()),
            )
            self._conn.commit()
        logger.info("admitted %s to room %s (case %s)", patient_name or patient_id, room_id, case_id)
        return record

    def admissions(self, *, active_only: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT * FROM admissions"
        if active_only:
            sql += " WHERE discharged_at = ''"
        sql += " ORDER BY admitted_at DESC"
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def occupied_rooms(self) -> set[str]:
        return {a["room_id"] for a in self.admissions(active_only=True)}

    def discharge(self, admission_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE admissions SET discharged_at=? WHERE admission_id=? AND discharged_at=''",
                (_now(), admission_id),
            )
            self._conn.commit()
        return cursor.rowcount > 0


def _request_from_row(row: sqlite3.Row) -> Request:
    return Request(
        request_id=row["request_id"],
        case_id=row["case_id"],
        patient_id=row["patient_id"],
        kind=RequestKind(row["kind"]),
        summary=row["summary"],
        reasoning=row["reasoning"],
        payload=json.loads(row["payload"] or "{}"),
        evidence=json.loads(row["evidence"] or "[]"),
        status=RequestStatus(row["status"]),
        created_at=row["created_at"],
        decided_at=row["decided_at"],
        decided_by=row["decided_by"],
        decision_note=row["decision_note"],
    )
