"""
The requests queue — everything the planner wants to do that a human must approve.

This is the mechanism that lets the planner be given real freedom. It can reason
about a case however it likes, and it can want anything: notify a doctor, admit
the patient, call an ambulance, raise the urgency. What it cannot do is act. Every
action becomes a request with its reasoning attached, and a person on the hospital
dashboard approves or rejects it.

That inversion is what makes an LLM-driven planner safe here. The usual worry
about giving a model tools is that a bad reading of a sentence becomes a real
action - an ambulance dispatched, a patient admitted, an email sent to the wrong
clinician. With approval in front of every action, a bad reading becomes a request
a human declines in two seconds. The model's judgement is a proposal; the human's
is the decision.

Each request carries the planner's own reasoning, because a queue of bare actions
is unreviewable. "Admit this patient" is not something anybody can approve
sensibly. "Admit this patient: third presentation in eleven days with the same
unresolved abdominal pain, each time discharged, pain now rated higher than on
either previous visit" is.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RequestKind(str, Enum):
    """What the planner is asking for.

    Kept to a closed set. A planner that could invent new kinds of action would
    produce requests nobody has written an approval path for, and an approval
    button that does not know what it is approving is worse than no button.
    """

    NOTIFY_DOCTOR = "notify_doctor"
    ADMIT_PATIENT = "admit_patient"
    REQUEST_AMBULANCE = "request_ambulance"
    RAISE_URGENCY = "raise_urgency"
    REFER_DEPARTMENT = "refer_department"
    TALK_TO_CLINICIAN = "talk_to_clinician"
    PREPARE_TEAM = "prepare_team"


class RequestStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


#: How each kind is labelled and what approving it will actually do. Shown on
#: the dashboard so the person approving knows the consequence before they click.
KIND_DETAIL: dict[RequestKind, dict[str, str]] = {
    RequestKind.NOTIFY_DOCTOR: {
        "label": "Notify clinician",
        "consequence": "Sends the triage note to the on-call clinician for the department.",
    },
    RequestKind.ADMIT_PATIENT: {
        "label": "Admit patient",
        "consequence": "Reserves a bed. You choose the room; VITA does not pick one.",
    },
    RequestKind.REQUEST_AMBULANCE: {
        "label": "Emergency transport",
        "consequence": "Raises a dispatch request with emergency operations.",
    },
    RequestKind.RAISE_URGENCY: {
        "label": "Raise urgency",
        "consequence": "Increases the triage urgency above what the rules produced.",
    },
    RequestKind.REFER_DEPARTMENT: {
        "label": "Refer to department",
        "consequence": "Routes the case to a department other than the one triage chose.",
    },
    RequestKind.TALK_TO_CLINICIAN: {
        "label": "Patient asked to speak to someone",
        "consequence": "Opens the chat so you can reply to them directly.",
    },
    RequestKind.PREPARE_TEAM: {
        "label": "Prepare the team",
        "consequence": "Alerts the receiving department to get ready before the patient reaches them.",
    },
}


@dataclass
class Request:
    """One thing the planner wants done, and why."""

    request_id: str
    case_id: str
    patient_id: str
    kind: RequestKind
    summary: str
    reasoning: str

    #: Kind-specific detail: which department, which urgency, which address.
    payload: dict[str, Any] = field(default_factory=dict)

    #: What the planner leaned on. Rule ids, recalled memories, retrieved policy -
    #: so a reviewer can check the reasoning rather than take it on trust.
    evidence: list[str] = field(default_factory=list)

    status: RequestStatus = RequestStatus.PENDING
    created_at: str = field(default_factory=_now)
    decided_at: str = ""
    decided_by: str = ""
    decision_note: str = ""

    @classmethod
    def create(
        cls,
        *,
        case_id: str,
        patient_id: str,
        kind: RequestKind,
        summary: str,
        reasoning: str,
        payload: dict[str, Any] | None = None,
        evidence: list[str] | None = None,
    ) -> "Request":
        return cls(
            request_id=f"REQ-{uuid.uuid4().hex[:6].upper()}",
            case_id=case_id,
            patient_id=patient_id,
            kind=kind,
            summary=summary.strip(),
            reasoning=reasoning.strip(),
            payload=dict(payload or {}),
            evidence=list(evidence or []),
        )

    @property
    def pending(self) -> bool:
        return self.status is RequestStatus.PENDING

    @property
    def label(self) -> str:
        return KIND_DETAIL.get(self.kind, {}).get("label", self.kind.value)

    @property
    def consequence(self) -> str:
        return KIND_DETAIL.get(self.kind, {}).get("consequence", "")

    def approve(self, by: str, note: str = "") -> None:
        self.status = RequestStatus.APPROVED
        self.decided_by = by
        self.decision_note = note
        self.decided_at = _now()

    def reject(self, by: str, note: str = "") -> None:
        self.status = RequestStatus.REJECTED
        self.decided_by = by
        self.decision_note = note
        self.decided_at = _now()

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "case_id": self.case_id,
            "patient_id": self.patient_id,
            "kind": self.kind.value,
            "label": self.label,
            "consequence": self.consequence,
            "summary": self.summary,
            "reasoning": self.reasoning,
            "payload": self.payload,
            "evidence": self.evidence,
            "status": self.status.value,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
            "decided_by": self.decided_by,
            "decision_note": self.decision_note,
        }
