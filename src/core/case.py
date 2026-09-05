"""
The patient case — the structured state a conversation accumulates.

This is the memory the intake actually runs on, and the shape of it is a
deliberate answer to something the problem statement asks for: the triage note
has to separate what the patient reported from what the follow-ups established
from what remains unknown. That is not a formatting job done at the end. It is a
property of every fact, recorded when the fact is created, and the note simply
reads it back.

The store is append-only underneath. A fact that supersedes another does not
erase it - `history` keeps every version, so a clinician can see that the
patient said one thing on turn two and another on turn six, and the
contradiction machinery can notice when those two things cannot both be true.

Nothing here decides anything either. It holds what is known, what was said, and
what is still open, and hands all three to the rule engine.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from ..config import SystemMode
from . import contradictions as contradiction_check
from .schema import (
    Complaint,
    Contradiction,
    Fact,
    FactSource,
    TriageDecision,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


#: Set by the case store at startup so ids continue from what is already saved.
_counter = itertools.count(1)


def reset_case_numbering(start: int) -> None:
    """Continue numbering from an existing store rather than restarting at 1."""
    global _counter
    _counter = itertools.count(max(1, start))


def new_case_id() -> str:
    """A short, sayable id.

    Hex ids like VITA-AB2B23 are unambiguous and unreadable. Somebody reading a
    queue aloud says "case seven", so that is what the id should be.
    """
    return f"C-{next(_counter)}"


class CaseStatus(str, Enum):
    """Where a case has got to."""

    INTAKE = "intake"
    AWAITING_REVIEW = "awaiting_review"
    REVIEWED = "reviewed"
    CLOSED = "closed"


@dataclass
class Turn:
    """One exchange, kept in the language it happened in.

    `role` is "patient", "vita", or "staff" - a real person at the hospital
    typing to the patient. Staff messages sit in the same thread as everything
    else so the patient has one conversation rather than two, and so the record
    shows who said what.
    """

    role: str  # "patient", "vita" or "staff"
    text: str
    language: str = "en"
    at: str = field(default_factory=_now)

    #: For a VITA turn, the fact the question was trying to establish, and the
    #: rule that wanted it. This is what lets the interface answer "why are you
    #: asking me this?" with a rule id rather than a paraphrase.
    asked_about: str = ""
    driven_by_rule: str = ""

    #: For a staff message, who sent it.
    author: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "author": self.author,
            "text": self.text,
            "language": self.language,
            "at": self.at,
            "asked_about": self.asked_about,
            "driven_by_rule": self.driven_by_rule,
        }


@dataclass
class Case:
    """Everything known about one intake."""

    case_id: str = field(default_factory=new_case_id)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    #: Who this case is about. A name is the whole of identity here - see
    #: core/patient.py for why that is enough and where it falls short.
    patient_id: str = ""
    patient_name: str = ""

    #: Collected at the desk before the conversation starts, the way a real
    #: intake form works. Age becomes a triage fact directly; the rest is
    #: context a clinician wants and the rules do not use.
    patient_age: str = ""
    patient_gender: str = ""
    past_history: str = ""
    takes_medication: str = ""

    #: What they said they take, and for how long. Free text - a patient writes
    #: "the small white one for BP" as readily as a drug name, and both are
    #: worth a clinician seeing even when the reference table recognises neither.
    medications_declared: str = ""
    medication_duration: str = ""

    language: str = "en"
    complaint: Complaint = Complaint.UNDETERMINED
    status: CaseStatus = CaseStatus.INTAKE

    #: Current belief, one fact per key.
    facts: dict[str, Fact] = field(default_factory=dict)
    #: Every fact ever recorded, superseded ones included.
    history: list[Fact] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)

    turns: list[Turn] = field(default_factory=list)

    #: The fact the outstanding question is about, and the rule that wants it.
    pending_fact: str = ""
    pending_rule: str = ""

    #: How many times each fact has been asked about. A patient who cannot
    #: answer should not be asked a third time, and the count is what lets the
    #: conversation give up - which is what lets the case close and escalate.
    asked_counts: dict[str, int] = field(default_factory=dict)

    #: Ids of red flags that fired at any point during the intake.
    red_flags: list[str] = field(default_factory=list)
    #: True when a red flag or the scope classifier placed the case outside the
    #: covered rule set.
    out_of_scope: bool = False

    #: The scope classifier's verdict and its evidence, kept so a clinician can
    #: see why VITA declined to triage rather than only that it did.
    scope_verdict: dict[str, Any] = field(default_factory=dict)

    #: One fingerprint of the triage state per turn, used to notice when the
    #: conversation has stopped changing the outcome. Convergence, not a
    #: question count, is what ends an intake.
    state_history: list[str] = field(default_factory=list)

    #: True once the patient has been invited to add anything else. Asked once,
    #: when the picture has stabilised - never as a reflex after every question.
    asked_anything_else: bool = False

    #: The planner's reading of the case, for the clinician only. Never shown to
    #: the patient, and clearly an AI impression rather than a diagnosis.
    clinical_impression: str = ""

    #: The same thing, but written from the first turn and rewritten as the
    #: picture changes. A clinician watching the queue wants to know what this
    #: might be before the intake finishes, not after - by then they could have
    #: read the notes themselves.
    working_impression: str = ""
    working_impression_turn: int = 0

    #: True once the patient has asked to speak to a person.
    asked_for_clinician: bool = False

    #: One entry per turn: what the patient said, what VITA asked, what it
    #: took from the answer, and where the triage stood afterwards. This is the
    #: record that lets somebody walk a decision backwards - a triage note says
    #: what was concluded, and this says how it got there.
    reasoning_trace: list[dict[str, Any]] = field(default_factory=list)

    #: Photographs of the injury, and what could be seen in each.
    injury_photos: list[dict[str, Any]] = field(default_factory=list)

    #: Clips the patient spoke, and what was heard in each. Kept so a
    #: clinician can see what was heard rather than only what was understood.
    voice_clips: list[dict[str, Any]] = field(default_factory=list)

    #: Medication photographs the patient sent, and what was read from each.
    medication_photos: list[dict[str, Any]] = field(default_factory=list)

    #: Created by the evaluation harness rather than by a patient. Excluded
    #: from cross-visit recall, because otherwise every eval run leaves fever
    #: cases behind that make the next run's fever scenario trip GEN-01 - the
    #: rule working correctly on data that should never have been there.
    synthetic: bool = False

    #: The description sat near the boundary between covered and uncovered.
    #: Distinct from out_of_scope: the case is still triaged, because the facts
    #: may be perfectly clear even when the classification is not, but a human
    #: sees it either way.
    scope_uncertain: bool = False

    decision: TriageDecision | None = None
    #: The mode the system was in when the decision was produced.
    decided_in_mode: SystemMode = SystemMode.FULL

    #: A clinician's override, if one has been made.
    override_urgency: str = ""
    override_reason: str = ""
    override_by: str = ""
    override_at: str = ""

    # -- turns -----------------------------------------------------------

    @property
    def turn_number(self) -> int:
        return sum(1 for t in self.turns if t.role == "patient")

    def add_patient_turn(self, text: str, language: str = "") -> Turn:
        turn = Turn(role="patient", text=text, language=language or self.language)
        self.turns.append(turn)
        self.updated_at = _now()
        return turn

    def add_staff_turn(self, text: str, author: str) -> Turn:
        """A message from a real person at the hospital to the patient."""
        turn = Turn(role="staff", text=text, language=self.language, author=author)
        self.turns.append(turn)
        self.updated_at = _now()
        return turn

    def add_vita_turn(self, text: str, *, asked_about: str = "", rule: str = "") -> Turn:
        turn = Turn(
            role="vita",
            text=text,
            language=self.language,
            asked_about=asked_about,
            driven_by_rule=rule,
        )
        self.turns.append(turn)
        self.pending_fact = asked_about
        self.pending_rule = rule
        self.updated_at = _now()
        return turn

    # -- facts -----------------------------------------------------------

    def record(self, incoming: Fact) -> Contradiction | None:
        """Record a fact, keeping any earlier version and noting conflicts.

        A red-flag assertion never loses to a later extraction. The patient who
        wrote "I can't breathe" has said something the pattern matcher read
        correctly and a model might soften, and the safer of two readings is the
        one that survives.
        """
        existing = self.facts.get(incoming.key)
        conflict = contradiction_check.detect(existing, incoming)
        if conflict is not None:
            self.contradictions.append(conflict)

        self.history.append(incoming)

        if existing is not None and self._outranks(existing, incoming):
            self.updated_at = _now()
            return conflict

        self.facts[incoming.key] = incoming
        self.updated_at = _now()
        return conflict

    def record_all(self, facts: list[Fact]) -> list[Contradiction]:
        return [c for c in (self.record(f) for f in facts) if c is not None]

    @staticmethod
    def _outranks(existing: Fact, incoming: Fact) -> bool:
        """Should the existing fact be kept in preference to the new one?"""
        if existing.source is FactSource.RED_FLAG_MATCH and incoming.source is not FactSource.RED_FLAG_MATCH:
            return True
        # A drug named on the registration form is not displaced by an answer
        # given in conversation. Somebody who wrote "Warfarin" on the form and
        # then said no to "any blood-thinning medicine?" has not stopped taking
        # warfarin - they have not recognised the category, which is ordinary
        # and common. Letting the second answer win would delete the fact that
        # makes IN-03 apply to them. The disagreement is still recorded as a
        # contradiction, so the clinician sees both halves of it.
        if existing.source is FactSource.REGISTRATION and incoming.source in (
            FactSource.PATIENT_VERBATIM,
            FactSource.FOLLOWUP_ANSWER,
            FactSource.DEGRADED_EXTRACTION,
        ):
            return True
        # A confident earlier answer is not displaced by a much less confident
        # restatement of the same thing.
        if existing.is_known and not incoming.is_known:
            return True
        return False

    def known(self) -> dict[str, Fact]:
        return {k: f for k, f in self.facts.items() if f.is_known}

    # -- the three views the triage note requires -------------------------

    def reported(self) -> list[Fact]:
        """What the patient gave without being asked for it.

        The form counts. A patient who wrote their age and their medication at
        the desk reported those as surely as the one who typed them into the
        chat, and the note's distinction is against what a follow-up *had* to
        draw out - not against paper.
        """
        return [
            f
            for f in self.facts.values()
            if f.source in (FactSource.PATIENT_VERBATIM, FactSource.RED_FLAG_MATCH,
                            FactSource.REGISTRATION)
            and f.is_known
            and f.key not in ("complaint", "language")
        ]

    def established(self) -> list[Fact]:
        """What a follow-up question settled."""
        return [
            f
            for f in self.facts.values()
            if f.source is FactSource.FOLLOWUP_ANSWER and f.is_known
        ]

    def recalled(self) -> list[Fact]:
        """What came from a previous visit rather than from this conversation."""
        return [f for f in self.facts.values() if f.source is FactSource.MEMORY_RECALL]

    def unknowns(self) -> list[str]:
        """Facts the current decision is still waiting on."""
        return list(self.decision.unknowns) if self.decision else []

    # -- serialisation ---------------------------------------------------

    @property
    def effective_urgency(self) -> str:
        """The urgency in force, which is the clinician's if they overrode it."""
        if self.override_urgency:
            return self.override_urgency
        return self.decision.urgency.value if self.decision else ""

    def as_dict(self, *, full: bool = True) -> dict[str, Any]:
        summary = {
            "case_id": self.case_id,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "patient_age": self.patient_age,
            "patient_gender": self.patient_gender,
            "past_history": self.past_history,
            "takes_medication": self.takes_medication,
            "medications_declared": self.medications_declared,
            "medication_duration": self.medication_duration,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "language": self.language,
            "complaint": self.complaint.value,
            "status": self.status.value,
            "urgency": self.effective_urgency,
            "system_urgency": self.decision.urgency.value if self.decision else "",
            "department": self.decision.department if self.decision else "",
            "matched_rules": self.decision.cited_rules if self.decision else [],
            "requires_human_review": self.decision.requires_human_review if self.decision else False,
            "red_flags": self.red_flags,
            "out_of_scope": self.out_of_scope,
            "scope_verdict": self.scope_verdict,
            "scope_uncertain": self.scope_uncertain,
            "clinical_impression": self.clinical_impression,
            "working_impression": self.working_impression,
            "working_impression_turn": self.working_impression_turn,
            "asked_for_clinician": self.asked_for_clinician,
            "medication_photos": self.medication_photos,
            "injury_photos": self.injury_photos,
            "voice_clips": self.voice_clips,
            "reasoning_trace": self.reasoning_trace,
            "asked_anything_else": self.asked_anything_else,
            "state_history": self.state_history,
            "synthetic": self.synthetic,
            "decided_in_mode": self.decided_in_mode.value,
            "overridden": bool(self.override_urgency),
            "turn_count": self.turn_number,
        }
        if not full:
            return summary

        summary.update(
            {
                "facts": {k: f.as_dict() for k, f in self.facts.items()},
                "history": [f.as_dict() for f in self.history],
                "reported": [f.as_dict() for f in self.reported()],
                "established": [f.as_dict() for f in self.established()],
                "recalled": [f.as_dict() for f in self.recalled()],
                "unknowns": self.unknowns(),
                "contradictions": [c.as_dict() for c in self.contradictions],
                "turns": [t.as_dict() for t in self.turns],
                "pending_fact": self.pending_fact,
                "pending_rule": self.pending_rule,
                "decision": self.decision.as_dict() if self.decision else None,
                "override": {
                    "urgency": self.override_urgency,
                    "reason": self.override_reason,
                    "by": self.override_by,
                    "at": self.override_at,
                },
            }
        )
        return summary
