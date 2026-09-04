"""
Application wiring — one object that owns everything with a lifetime.

Built once at startup and handed to the request handlers. The endpoints stay
thin because all the sequencing lives here and in the orchestrator, which means
the interesting behaviour is testable without going through HTTP.

The startup ordering matters and is deliberate. The knowledge base loads first
and is allowed to fail loudly: a VITA with no rules cannot triage anybody, and
coming up anyway to serve wrong answers would be worse than not coming up. The
Gemini client is built next and is *not* allowed to fail - a missing key
produces a client that reports OFFLINE, because the port has to open either way.
"""

from __future__ import annotations

import logging
from typing import Any

from ..agents.base import AgentRegistry
from ..agents.complaint import ComplaintAgent
from ..agents.history import HistoryAgent
from ..agents.red_flag import RedFlagAgent, verify_coverage
from ..agents.symptoms import SymptomAgent
from ..agents.timeline import TimelineAgent
from ..agents.vitals import VitalsAgent
from ..config import Settings, SystemMode, load_settings
from ..core.case import Case, CaseStatus
from ..core.knowledge import KnowledgeBase, load_knowledge_base
from ..core.note import build_note, render_text
from ..core.schema import Complaint, Fact, FactSource, Tri, Urgency
from ..llm.gemini import GeminiClient
from ..llm.phrasing import Phraser
from ..orchestrator.intake import IntakeOrchestrator, TurnResult
from ..rag.retriever import Retriever
from ..store.cases import CaseStore
from ..store.hospital import HospitalDirectory
from ..tools import Tier
from .ambulance import AmbulanceService
from .notify import Notifier

logger = logging.getLogger(__name__)

#: Window for the re-presentation rule. A patient back inside this with the same
#: unresolved complaint is reviewed by a clinician whatever else the rules say.
RETURN_WINDOW_HOURS = 72

#: Urgency at or above which a clinician is notified directly rather than only
#: through the dashboard.
NOTIFY_AT = Urgency.HIGH


class VitaServices:
    """Everything the application needs, assembled once."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

        # Fails loudly on purpose. No rules means no triage.
        self.kb: KnowledgeBase = load_knowledge_base()

        self.registry = AgentRegistry()
        red_flag = RedFlagAgent()
        for agent in (
            ComplaintAgent(),
            red_flag,
            SymptomAgent(),
            TimelineAgent(),
            VitalsAgent(),
            HistoryAgent(),
        ):
            self.registry.register(agent)
        verify_coverage(red_flag, self.kb.red_flags)

        # Never fails. A missing key yields a client that reports OFFLINE.
        self.llm = GeminiClient(self.settings)
        self.phraser = Phraser(self.llm)
        self.retriever = Retriever(self.llm)
        self.orchestrator = IntakeOrchestrator(
            self.kb, self.registry, self.llm, self.phraser, self.retriever
        )

        self.cases = CaseStore()
        self.hospital = HospitalDirectory()
        self.notifier = Notifier()
        self.ambulance = AmbulanceService()

        # Built last: the tool layer reaches back into everything above it.
        from ..tools import ToolLayer

        self.tools = ToolLayer(self)

        self._live: dict[str, Case] = {}

    # -- status ----------------------------------------------------------

    @property
    def mode(self) -> SystemMode:
        return self.llm.mode

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "llm": self.llm.status(),
            "knowledge": self.kb.summary(),
            "retrieval": self.retriever.status(),
            "agents": self.registry.describe(),
            "queue": self.cases.counts(),
            "notifications": {
                "dry_run": not self.settings.notify_enabled,
                "sent": len(self.notifier.outbox),
            },
            "tools": {
                "retrieval": self.tools.names(tier=Tier.RETRIEVAL),
                "decision": self.tools.names(tier=Tier.DECISION),
                "note": (
                    "Decision tools are reachable over MCP by a deliberate client "
                    "but are never advertised to the conversation model."
                ),
            },
            "ambulance": {"requests": len(self.ambulance.requests)},
        }

    # -- intake ----------------------------------------------------------

    def start_case(self, language: str = "en", *, synthetic: bool = False) -> Case:
        case = Case(language=language, synthetic=synthetic)
        self._live[case.case_id] = case
        self.cases.save(case)
        self.cases.audit(case.case_id, "case_opened", detail=f"language={language}")
        logger.info("case %s opened", case.case_id)
        return case

    def get_case(self, case_id: str) -> Case | None:
        return self._live.get(case_id) or self.cases.get(case_id)

    def message(self, case_id: str, text: str) -> TurnResult | None:
        """Run one conversational turn and persist the result."""
        case = self.get_case(case_id)
        if case is None:
            return None
        self._live[case.case_id] = case

        self._recall_prior_visit(case)
        result = self.orchestrator.handle(case, text)

        self.cases.save(case)
        if result.red_flags:
            self.cases.audit(
                case.case_id, "red_flags_matched", detail=", ".join(result.red_flags)
            )
        if result.finished:
            self._on_finished(case)

        return result

    def _recall_prior_visit(self, case: Case) -> None:
        """Check the record for a recent visit with the same complaint.

        This is the one place VITA remembers anything across cases, and it earns
        its place by feeding a rule rather than by decorating the note. A patient
        back within 72 hours with an unresolved complaint is a recognised marker
        of a missed or deteriorating condition, and no amount of skill at reading
        this conversation would surface it.
        """
        if case.complaint in (Complaint.UNDETERMINED, Complaint.OUT_OF_SCOPE):
            return
        if "prior_visit_72h_same_complaint" in case.facts:
            return

        prior = self.cases.find_prior_visit(
            complaint=case.complaint.value,
            within_hours=RETURN_WINDOW_HOURS,
            exclude=case.case_id,
        )
        if prior is None:
            return

        logger.info(
            "case %s: prior visit %s with same complaint %.1fh ago",
            case.case_id,
            prior["case_id"],
            prior["hours_ago"],
        )
        case.record(
            Fact(
                key="prior_visit_72h_same_complaint",
                value=Tri.TRUE,
                source=FactSource.MEMORY_RECALL,
                turn=case.turn_number,
                verbatim=(
                    f"case {prior['case_id']}, {prior['hours_ago']} hours ago, "
                    f"same complaint, triaged {prior['urgency'] or 'unrecorded'}"
                ),
                language=case.language,
                confidence=1.0,
                agent="memory:case_store",
            )
        )
        self.cases.audit(
            case.case_id,
            "prior_visit_recalled",
            detail=f"{prior['case_id']} {prior['hours_ago']}h ago",
        )

    # -- closing ---------------------------------------------------------

    def _on_finished(self, case: Case) -> None:
        decision = case.decision
        if decision is None:
            return

        self.cases.audit(
            case.case_id,
            "triage_decided",
            detail=(
                f"{decision.urgency.value} / {decision.department} / "
                f"rules={','.join(decision.cited_rules) or 'none'} / "
                f"mode={case.decided_in_mode.value}"
            ),
        )

        if decision.requires_human_review:
            self.cases.audit(
                case.case_id,
                "escalated",
                detail=", ".join(r.value for r in decision.escalation_reasons),
            )

        if decision.urgency.rank >= NOTIFY_AT.rank or decision.requires_human_review:
            self._notify(case)

    def _notify(self, case: Case) -> None:
        """Notify the on-call clinician for the routed department.

        The recipient comes from the hospital directory. It is not a parameter,
        not a model output, and not anything the patient typed.
        """
        decision = case.decision
        if decision is None:
            return

        doctor = self.hospital.on_call_for(decision.department)
        if doctor is None:
            logger.error("no on-call doctor available for %s", decision.department)
            self.cases.audit(case.case_id, "notification_skipped", detail="no on-call doctor")
            return

        note = build_note(case, self.kb)
        notification = self.notifier.notify_clinician(
            case_id=case.case_id,
            urgency=decision.urgency.value,
            department=decision.department,
            note_text=render_text(note),
            recipient=doctor.email,
            recipient_name=doctor.name,
            cited_rules=decision.cited_rules,
            unknowns=decision.unknowns,
        )
        self.cases.audit(
            case.case_id, "clinician_notified", detail=notification.describe()
        )

    # -- clinician actions -----------------------------------------------

    def override(self, case_id: str, urgency: str, reason: str, by: str) -> Case | None:
        """Record a clinician's override of the system's recommendation.

        Both values are kept. The dashboard shows what VITA recommended and what
        the clinician decided, side by side, and the audit trail keeps the
        reason. An override that replaced the original would make the system
        look like it had agreed all along.
        """
        case = self.get_case(case_id)
        if case is None:
            return None
        try:
            level = Urgency(urgency.upper())
        except ValueError:
            return None

        from datetime import datetime, timezone

        case.override_urgency = level.value
        case.override_reason = reason
        case.override_by = by
        case.override_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        case.status = CaseStatus.REVIEWED

        self._live[case.case_id] = case
        self.cases.save(case)
        self.cases.audit(
            case.case_id,
            "clinician_override",
            actor=by,
            detail=(
                f"{case.decision.urgency.value if case.decision else '?'} -> "
                f"{level.value}: {reason}"
            ),
        )
        return case

    def mark_reviewed(self, case_id: str, by: str) -> Case | None:
        case = self.get_case(case_id)
        if case is None:
            return None
        case.status = CaseStatus.REVIEWED
        self._live[case.case_id] = case
        self.cases.save(case)
        self.cases.audit(case.case_id, "marked_reviewed", actor=by)
        return case

    # -- views -----------------------------------------------------------

    def note(self, case_id: str) -> dict[str, Any] | None:
        case = self.get_case(case_id)
        if case is None:
            return None
        note = build_note(case, self.kb)
        note["text"] = render_text(note)
        note["routing"] = self.hospital.routing_note(
            case.decision.department if case.decision else ""
        )
        note["notifications"] = [n.as_dict() for n in self.notifier.for_case(case_id)]
        note["audit"] = self.cases.trail(case_id)
        return note
