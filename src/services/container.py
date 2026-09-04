"""
Application wiring — one object owning everything with a lifetime.

Built once at startup and handed to the request handlers, so sequencing lives
here and in the planner rather than in HTTP endpoints.

Startup order is deliberate. The knowledge base loads first and is allowed to
fail loudly: a VITA with no rules cannot triage anybody, and coming up anyway to
serve wrong answers is worse than not coming up. Everything after it fails
quietly and reports itself - a missing Gemini key, an unopenable memory store,
an MCP session that will not start. Each costs a capability; none is a reason to
leave port 8000 closed.

The MCP session opens last, because the tool layer it publishes reaches back
into everything above it.
"""

from __future__ import annotations

import logging
from typing import Any

from ..agents.red_flag import RedFlagAgent, verify_coverage
from ..config import Settings, SystemMode, load_settings
from ..core.case import Case, CaseStatus
from ..core.knowledge import KnowledgeBase, load_knowledge_base
from ..core.note import build_note, render_text
from ..core.patient import Patient
from ..core.requests import Request, RequestKind
from ..core.schema import Urgency
from ..llm.gemini import GeminiClient
from ..llm.phrasing import Phraser
from ..mcp_bridge import McpBridge
from ..memory.palace import KIND_FACT, KIND_OUTCOME, KIND_VISIT, PatientMemory
from ..orchestrator.planner import PlannerTurn, TriagePlanner
from ..rag.retriever import Retriever
from ..store.cases import CaseStore
from ..store.hospital import HospitalDirectory
from ..store.requests import RequestStore
from ..tools import ToolLayer
from .ambulance import AmbulanceError, AmbulanceService
from .medication import MedicationReader
from .notify import Notifier

logger = logging.getLogger(__name__)

#: Urgency at or above which a notification request is raised automatically,
#: without the planner having to think of it. Below that, notifying is a
#: judgement, and judgements go through approval.
NOTIFY_AT = Urgency.HIGH


class VitaServices:
    """Everything the application needs, assembled once."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

        # Fails loudly on purpose. No rules means no triage.
        self.kb: KnowledgeBase = load_knowledge_base()

        # Never fails. A missing key yields a client reporting OFFLINE.
        self.llm = GeminiClient(self.settings)
        self.phraser = Phraser(self.llm)
        self.retriever = Retriever(self.llm)
        self.memory = PatientMemory(self.llm)

        self.cases = CaseStore()
        self.requests = RequestStore()
        self.hospital = HospitalDirectory()
        self.notifier = Notifier()
        self.ambulance = AmbulanceService()
        self.medications = MedicationReader(self.llm)

        self.red_flag_agent = RedFlagAgent()
        verify_coverage(self.red_flag_agent, self.kb.red_flags)

        # The tool layer, then an MCP session over it. The planner reaches every
        # capability through the protocol, never by calling these objects.
        self.tools = ToolLayer(self)
        self.mcp = McpBridge(self._build_mcp_server)
        self.planner = TriagePlanner(
            self.kb, self.llm, self.mcp, self.red_flag_agent, self.phraser
        )

        self._live: dict[str, Case] = {}

    def _build_mcp_server(self) -> Any:
        from ..mcp_server import build_server

        return build_server(self.tools)

    def close(self) -> None:
        self.mcp.close()

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
            "memory": self.memory.status(),
            "mcp": self.mcp.status(),
            "tools": self.tools.names(),
            "queue": self.cases.counts(),
            "requests": self.requests.counts(),
            "notifications": {
                "dry_run": not self.settings.notify_enabled,
                "sent": len(self.notifier.outbox),
            },
        }

    # -- intake ----------------------------------------------------------

    def start_case(self, name: str = "", language: str = "en", *, synthetic: bool = False) -> Case:
        """Open a case for a named patient.

        The name is the whole of identity, and it is what links this visit to
        the last one - which is what makes memory and history worth having.
        """
        patient = Patient.from_name(name)
        case = Case(
            language=language,
            synthetic=synthetic,
            patient_id=patient.patient_id,
            patient_name=patient.name,
        )
        self._live[case.case_id] = case
        self.cases.save(case)
        self.cases.audit(
            case.case_id,
            "case_opened",
            detail=f"patient={patient.name or 'anonymous'} language={language}",
        )
        logger.info("case %s opened for %s", case.case_id, patient.name or "anonymous")
        return case

    def get_case(self, case_id: str) -> Case | None:
        return self._live.get(case_id) or self.cases.get(case_id)

    def message(self, case_id: str, text: str) -> PlannerTurn | None:
        case = self.get_case(case_id)
        if case is None:
            return None
        self._live[case.case_id] = case

        result = self.planner.handle(case, text)
        self.cases.save(case)

        if result.red_flags:
            self.cases.audit(case.case_id, "red_flags_matched", detail=", ".join(result.red_flags))
        if result.finished:
            self._on_finished(case)

        return result

    def read_medication_photo(self, case_id: str, image: bytes, mime_type: str) -> dict[str, Any]:
        """Read a photo of the patient's medication and record what it establishes.

        The facts go in through the same tool the planner uses, so a drug read
        off a packet is recorded exactly like a drug the patient named out loud -
        same provenance, same validation, same rules.
        """
        case = self.get_case(case_id)
        if case is None:
            return {"error": f"no case {case_id!r}"}

        reading = self.medications.read(image, mime_type)
        if reading.error:
            return {"error": reading.error, "reading": reading.as_dict()}

        if reading.facts:
            self.tools.call(
                "record_facts",
                {
                    "case_id": case_id,
                    "facts": [
                        {
                            "key": key,
                            "value": value,
                            "evidence": f"read from a photo of {reading.attribution.get(key, 'the packet')}",
                        }
                        for key, value in reading.facts.items()
                    ],
                },
            )

        case.medication_photos.append(reading.as_dict())
        self.cases.save(case)
        self.cases.audit(case_id, "medication_photo_read", detail=reading.summary())

        for entry in reading.names:
            printed = entry.get("name_as_printed", "")
            if printed and case.patient_id:
                self.memory.remember(
                    patient_id=case.patient_id,
                    case_id=case_id,
                    kind=KIND_FACT,
                    text=f"{case.patient_name or 'Patient'} takes {printed}"
                         + (f" {entry.get('strength')}" if entry.get("strength") else "") + ".",
                )

        return {"case_id": case_id, "reading": reading.as_dict(), "summary": reading.summary()}

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
                case.case_id, "escalated",
                detail=", ".join(r.value for r in decision.escalation_reasons),
            )

        self._remember(case)

        if decision.urgency.rank >= NOTIFY_AT.rank:
            already = [
                r for r in self.requests.list(case_id=case.case_id)
                if r.kind is RequestKind.NOTIFY_DOCTOR
            ]
            if not already:
                doctor = self.hospital.on_call_for(decision.department)
                self.requests.add(
                    Request.create(
                        case_id=case.case_id,
                        patient_id=case.patient_id,
                        kind=RequestKind.NOTIFY_DOCTOR,
                        summary=(
                            f"Notify {doctor.name if doctor else 'on-call clinician'} "
                            f"({decision.department})"
                        ),
                        reasoning=(
                            f"Triage graded this {decision.urgency.value} on "
                            f"{', '.join(decision.cited_rules) or 'no matched rule'}. "
                            "Policy POL-09 requires the on-call clinician to be told "
                            "directly at this urgency."
                        ),
                        payload={
                            "department": decision.department,
                            "doctor": doctor.name if doctor else "",
                        },
                        evidence=[f"rule {r}" for r in decision.cited_rules],
                    )
                )

    def _remember(self, case: Case) -> None:
        """Write the few things worth recalling at this patient's next visit."""
        if not case.patient_id or case.synthetic or case.decision is None:
            return

        rules = case.decision.cited_rules
        self.memory.remember(
            patient_id=case.patient_id,
            case_id=case.case_id,
            kind=KIND_VISIT,
            text=(
                f"Visit {case.created_at[:10]}: {case.complaint.value.replace('_', ' ')}, "
                f"triaged {case.decision.urgency.value}, routed to {case.decision.department}."
                + (f" Rules: {', '.join(rules)}." if rules else "")
            ),
        )

        if case.clinical_impression:
            self.memory.remember(
                patient_id=case.patient_id,
                case_id=case.case_id,
                kind=KIND_OUTCOME,
                text=f"Impression at {case.created_at[:10]} visit: {case.clinical_impression}",
            )

        # Durable facts outlive the visit. A patient who told us once that they
        # take an anticoagulant should not have to remember to say so next time.
        for key in ("on_anticoagulants", "immunocompromised", "known_asthma", "known_cardiac_history"):
            fact = case.facts.get(key)
            if fact is not None and fact.is_known and fact.tri.value == "true":
                self.memory.remember(
                    patient_id=case.patient_id,
                    case_id=case.case_id,
                    kind=KIND_FACT,
                    text=f"{case.patient_name or 'Patient'}: {key.replace('_', ' ')} confirmed.",
                )

    # -- approvals -------------------------------------------------------

    def decide_request(
        self, request_id: str, *, approved: bool, by: str, note: str = "", room_id: str = ""
    ) -> dict[str, Any]:
        """Approve or reject a planner request, and carry out what was approved.

        The only path by which anything the planner wanted actually happens.
        Rejections are recorded rather than discarded - the proposals a hospital
        turned down are the interesting ones when anybody later asks how far the
        system was trusted.
        """
        request = self.requests.decide(request_id, approved=approved, by=by, note=note)
        if request is None:
            return {"error": "unknown request, or it has already been decided"}

        self.cases.audit(
            request.case_id,
            "request_approved" if approved else "request_rejected",
            actor=by,
            detail=f"{request.kind.value}: {note or '(no note)'}",
        )
        if not approved:
            return {"request": request.as_dict(), "carried_out": None}

        return {
            "request": request.as_dict(),
            "carried_out": self._carry_out(request, by=by, room_id=room_id),
        }

    def _carry_out(self, request: Request, *, by: str, room_id: str = "") -> dict[str, Any]:
        case = self.get_case(request.case_id)
        if case is None:
            return {"error": f"no case {request.case_id}"}

        if request.kind is RequestKind.NOTIFY_DOCTOR:
            return self._notify(case, request.payload.get("department", ""))

        if request.kind is RequestKind.ADMIT_PATIENT:
            return self._admit(case, request, by=by, room_id=room_id)

        if request.kind is RequestKind.REQUEST_AMBULANCE:
            try:
                created = self.ambulance.create(
                    case_id=case.case_id,
                    urgency=case.effective_urgency,
                    pickup_location=request.payload.get("pickup_location") or "confirmed at the desk",
                    confirmed_by=by,
                )
            except AmbulanceError as exc:
                return {"error": str(exc)}
            return {"ambulance": created.as_dict()}

        if request.kind is RequestKind.RAISE_URGENCY:
            level = request.payload.get("urgency", "")
            case.override_urgency = level
            case.override_reason = f"planner request approved: {request.reasoning[:160]}"
            case.override_by = by
            case.override_at = request.decided_at
            self.cases.save(case)
            return {"urgency": level}

        if request.kind is RequestKind.REFER_DEPARTMENT:
            department = request.payload.get("department", "")
            self.cases.audit(case.case_id, "referred", actor=by, detail=department)
            return self._notify(case, department)

        return {"error": f"no handler for {request.kind.value}"}

    def _admit(self, case: Case, request: Request, *, by: str, room_id: str) -> dict[str, Any]:
        """Admit a patient to a room the approving clinician chose."""
        if not room_id:
            return {"error": "a room must be chosen before admitting; VITA does not pick one"}

        room = self.hospital.room(room_id)
        if room is None:
            return {"error": f"no room {room_id!r}"}
        if room.room_id in self.requests.occupied_rooms():
            return {"error": f"room {room.room_id} is already occupied"}

        record = self.requests.admit(
            admission_id=f"ADM-{request.request_id[4:]}",
            case_id=case.case_id,
            patient_id=case.patient_id,
            patient_name=case.patient_name,
            room_id=room.room_id,
            department=room.department,
            reason=request.reasoning,
            admitted_by=by,
        )
        self.cases.audit(case.case_id, "admitted", actor=by, detail=f"room {room.room_id}")
        self.memory.remember(
            patient_id=case.patient_id,
            case_id=case.case_id,
            kind=KIND_OUTCOME,
            text=f"Admitted to room {room.room_id} ({room.department}) on {record['admitted_at'][:10]}.",
        )
        return {"admission": record, "notification": self._notify(case, room.department, admission=record)}

    def _notify(self, case: Case, department: str, admission: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send the triage note to the on-call clinician for a department.

        The recipient comes from the roster. Not a parameter, not model output,
        and not anything a patient typed.
        """
        target = department or (case.decision.department if case.decision else "")
        doctor = self.hospital.on_call_for(target)
        if doctor is None:
            self.cases.audit(case.case_id, "notification_skipped", detail="no on-call clinician")
            return {"error": "no on-call clinician for that department"}

        note = build_note(case, self.kb)
        body = render_text(note)
        if case.clinical_impression:
            body += (
                "\n\nAI IMPRESSION - not a diagnosis. Generated by the intake assistant\n"
                "for your consideration, and overridable from the dashboard.\n"
                f"  {case.clinical_impression}\n"
            )
        if admission:
            body += (
                "\n\nADMISSION\n"
                f"  Room:        {admission['room_id']} ({admission['department']})\n"
                f"  Admitted by: {admission['admitted_by']}\n"
                f"  Reason:      {admission['reason']}\n"
            )

        notification = self.notifier.notify_clinician(
            case_id=case.case_id,
            urgency=case.effective_urgency or "UNKNOWN",
            department=target,
            note_text=body,
            recipient=doctor.address,
            recipient_name=doctor.name,
            cited_rules=case.decision.cited_rules if case.decision else [],
            unknowns=case.decision.unknowns if case.decision else [],
        )
        self.cases.audit(case.case_id, "clinician_notified", detail=notification.describe())
        return {"notification": notification.as_dict()}

    # -- clinician actions -----------------------------------------------

    def override(self, case_id: str, urgency: str, reason: str, by: str) -> Case | None:
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
            case.case_id, "clinician_override", actor=by,
            detail=f"{case.decision.urgency.value if case.decision else '?'} -> {level.value}: {reason}",
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
        note["clinical_impression"] = case.clinical_impression
        note["routing"] = self.hospital.routing_note(
            case.decision.department if case.decision else ""
        )
        note["conversation"] = [t.as_dict() for t in case.turns]
        note["notifications"] = [n.as_dict() for n in self.notifier.for_case(case_id)]
        note["requests"] = [r.as_dict() for r in self.requests.list(case_id=case_id)]
        note["audit"] = self.cases.trail(case_id)
        note["patient"] = {"patient_id": case.patient_id, "name": case.patient_name}
        note["memory"] = [
            m.as_dict() for m in self.memory.recall(patient_id=case.patient_id, limit=6)
        ]
        return note
