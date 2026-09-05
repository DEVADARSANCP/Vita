"""
The frozen contract — every structure the rest of VITA agrees on.

Two ideas carry most of the weight here.

**Facts are three-valued.** A symptom is not a boolean. It is TRUE, FALSE, or
UNKNOWN, and the third state is the one that matters: a patient who has not been
asked about breathing difficulty is not a patient without breathing difficulty.
Collapsing UNKNOWN into FALSE is the single most dangerous thing a triage system
can do, because it silently turns "we do not know" into "we checked and it is
fine". Every structure below keeps the distinction.

**Every fact carries where it came from.** The triage note has to separate what
the patient volunteered from what a follow-up established from what is still
unknown. That is a property of the data, not a rendering decision made at the
end, so `Fact` records its source, the turn it arrived on, and the patient's own
words in their own language.

Nothing in this module reasons. It only defines what the parts may say to each
other, so that the extraction agents and the rule engine can be developed and
changed independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


class Tri(str, Enum):
    """A three-valued answer. UNKNOWN is a real answer, not a missing one."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: Any) -> "Tri":
        """Best-effort conversion, defaulting to UNKNOWN rather than guessing.

        Anything we cannot read confidently becomes UNKNOWN, which routes the
        case towards a follow-up question or a human — never towards a
        recommendation built on an assumption.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.TRUE if value else cls.FALSE
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"true", "yes", "y", "present", "positive"}:
                return cls.TRUE
            if token in {"false", "no", "n", "absent", "negative", "denies"}:
                return cls.FALSE
        return cls.UNKNOWN


class Urgency(str, Enum):
    """Triage urgency, ordered. Comparison is by `rank`, never by name."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return _URGENCY_RANK[self]

    @classmethod
    def max(cls, *values: "Urgency") -> "Urgency":
        present = [v for v in values if v is not None]
        if not present:
            return cls.LOW
        return max(present, key=lambda u: u.rank)


_URGENCY_RANK = {
    Urgency.LOW: 0,
    Urgency.MODERATE: 1,
    Urgency.HIGH: 2,
    Urgency.CRITICAL: 3,
}


class Complaint(str, Enum):
    """The complaints the rule set covers.

    OUT_OF_SCOPE is deliberately one of them. A patient describing an obstetric
    or psychiatric emergency has a real complaint that this rule set cannot
    triage, and the correct behaviour is to say so and hand over — not to force
    the description into whichever of the five it resembles most.
    """

    FEVER = "fever"
    INJURY = "injury"
    CHEST_PAIN = "chest_pain"
    BREATHING_DIFFICULTY = "breathing_difficulty"
    ABDOMINAL_PAIN = "abdominal_pain"

    #: Rules that apply whatever the presenting complaint is - a return visit
    #: within 72 hours, an age modifier, pregnancy. These are always evaluated
    #: alongside the complaint-specific set.
    GENERAL = "general"

    OUT_OF_SCOPE = "out_of_scope"
    UNDETERMINED = "undetermined"


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


class FactSource(str, Enum):
    """How a fact came to be known.

    This is what lets the triage note say "the patient reported X, the
    follow-up established Y" rather than presenting a flat list that hides
    which is which.
    """

    #: Volunteered in the patient's own opening description.
    PATIENT_VERBATIM = "patient_verbatim"
    #: Written on the registration form before the conversation began. A
    #: different kind of evidence from anything said afterwards: the patient
    #: read it off the box rather than recalled it under questioning.
    REGISTRATION = "registration"
    #: Established by asking, because a rule needed it.
    FOLLOWUP_ANSWER = "followup_answer"
    #: Recalled from a previous visit by the same patient.
    MEMORY_RECALL = "memory_recall"
    #: Matched by a deterministic pattern, with no model involved.
    RED_FLAG_MATCH = "red_flag_match"
    #: Produced by the keyword fallback while the model was unavailable.
    DEGRADED_EXTRACTION = "degraded_extraction"


@dataclass
class Fact:
    """One thing VITA believes, and its complete provenance.

    `verbatim` holds the patient's own words in their own language. It is kept
    even when the value is canonicalised to English, so a clinician can always
    audit what was actually said against what the system recorded.
    """

    key: str
    value: Any
    source: FactSource
    turn: int = 0
    verbatim: str = ""
    language: str = "en"
    confidence: float = 1.0
    agent: str = ""
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def tri(self) -> Tri:
        """The value read as three-valued, for boolean conditions."""
        return Tri.coerce(self.value)

    @property
    def is_known(self) -> bool:
        if self.value is None:
            return False
        if isinstance(self.value, Tri):
            return self.value is not Tri.UNKNOWN
        if isinstance(self.value, str):
            return self.value.strip().lower() not in {"", "unknown"}
        return True

    @property
    def established_by_followup(self) -> bool:
        return self.source is FactSource.FOLLOWUP_ANSWER

    def as_dict(self) -> dict[str, Any]:
        value = self.value.value if isinstance(self.value, Tri) else self.value
        return {
            "key": self.key,
            "value": value,
            "source": self.source.value,
            "turn": self.turn,
            "verbatim": self.verbatim,
            "language": self.language,
            "confidence": self.confidence,
            "agent": self.agent,
            "recorded_at": self.recorded_at,
        }


@dataclass
class Contradiction:
    """Two incompatible answers about the same fact.

    Never resolved automatically. The system does not get to decide which of
    the patient's two statements was the real one, so both are kept and the
    case goes to a human.
    """

    key: str
    earlier: Fact
    later: Fact

    def describe(self) -> str:
        return (
            f"{self.key}: reported as {_render(self.earlier.value)} on turn "
            f"{self.earlier.turn}, then as {_render(self.later.value)} on turn "
            f"{self.later.turn}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "description": self.describe(),
            "earlier": self.earlier.as_dict(),
            "later": self.later.as_dict(),
        }


def _render(value: Any) -> str:
    return value.value if isinstance(value, Tri) else str(value)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


class Operator(str, Enum):
    """The comparisons a rule condition may make.

    Deliberately small. A rule an author cannot read at a glance is a rule
    nobody will audit, and auditability is the entire reason the decision lives
    in rules rather than in a model.
    """

    IS = "is"
    IS_NOT = "is_not"
    GTE = "gte"
    LTE = "lte"
    GT = "gt"
    LT = "lt"
    IN = "in"


@dataclass
class Condition:
    """One testable clause of a rule."""

    fact: str
    op: Operator
    value: Any

    def describe(self) -> str:
        words = {
            Operator.IS: "is",
            Operator.IS_NOT: "is not",
            Operator.GTE: "is at least",
            Operator.LTE: "is at most",
            Operator.GT: "is more than",
            Operator.LT: "is less than",
            Operator.IN: "is one of",
        }
        return f"{self.fact.replace('_', ' ')} {words[self.op]} {_render(self.value)}"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Condition":
        return cls(fact=raw["fact"], op=Operator(raw["op"]), value=raw["value"])

    def as_dict(self) -> dict[str, Any]:
        return {"fact": self.fact, "op": self.op.value, "value": _render(self.value)}


@dataclass
class Rule:
    """A triage rule. All conditions must hold for the rule to match.

    There is no OR inside a rule on purpose: an alternative is a second rule
    with its own id, so that whatever fires can always be cited precisely.
    "CP-03 matched" is an auditable statement; "CP-03's second branch matched"
    is not.

    `rationale` is the sentence a clinician reads. `source` names the public
    framework the rule was adapted from, so the recommendation is traceable
    past VITA itself.
    """

    rule_id: str
    complaint: Complaint
    conditions: list[Condition]
    urgency: Urgency
    department: str
    requires_human_review: bool = False
    rationale: str = ""
    source: str = ""
    version: str = "1.0"

    @property
    def required_facts(self) -> list[str]:
        return [c.fact for c in self.conditions]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Rule":
        return cls(
            rule_id=raw["rule_id"],
            complaint=Complaint(raw["complaint"]),
            conditions=[Condition.from_dict(c) for c in raw.get("conditions", [])],
            urgency=Urgency(raw["urgency"]),
            department=raw["department"],
            requires_human_review=bool(raw.get("requires_human_review", False)),
            rationale=raw.get("rationale", ""),
            source=raw.get("source", ""),
            version=str(raw.get("version", "1.0")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "complaint": self.complaint.value,
            "conditions": [c.as_dict() for c in self.conditions],
            "urgency": self.urgency.value,
            "department": self.department,
            "requires_human_review": self.requires_human_review,
            "rationale": self.rationale,
            "source": self.source,
            "version": self.version,
        }


class MatchState(str, Enum):
    """The outcome of evaluating a rule against what is currently known.

    POTENTIAL is the state the whole system is built around. It means: nothing
    known contradicts this rule, and something it needs is still unknown. A
    potentially-matched HIGH rule is the reason to ask another question, and if
    the question cannot be answered, the reason to call a human.
    """

    MATCHED = "matched"
    NOT_MATCHED = "not_matched"
    POTENTIAL = "potential"


@dataclass
class RuleEvaluation:
    """Why a rule did or did not fire, in terms an auditor can follow."""

    rule: Rule
    state: MatchState
    satisfied: list[Condition] = field(default_factory=list)
    failed: list[Condition] = field(default_factory=list)
    blocking: list[Condition] = field(default_factory=list)

    @property
    def blocking_facts(self) -> list[str]:
        """Fact keys that, if established, could decide this rule."""
        return [c.fact for c in self.blocking]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule.rule_id,
            "state": self.state.value,
            "urgency": self.rule.urgency.value,
            "department": self.rule.department,
            "rationale": self.rule.rationale,
            "source": self.rule.source,
            "satisfied": [c.describe() for c in self.satisfied],
            "failed": [c.describe() for c in self.failed],
            "blocking": [c.describe() for c in self.blocking],
            "blocking_facts": self.blocking_facts,
        }


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


class EscalationReason(str, Enum):
    """Why a case needs a human. Always stated; never left implicit."""

    NONE = "none"
    RULE_REQUIRES_REVIEW = "rule_requires_review"
    UNRESOLVED_UNKNOWN = "unresolved_unknown"
    CONTRADICTORY_REPORT = "contradictory_report"
    OUT_OF_SCOPE = "out_of_scope"
    SCOPE_UNCERTAIN = "scope_uncertain"
    DEGRADED_MODE = "degraded_mode"
    RED_FLAG = "red_flag"
    NO_RULE_MATCHED = "no_rule_matched"


@dataclass
class TriageDecision:
    """The output of the rule engine. Produced by code, never by a model."""

    urgency: Urgency
    department: str
    matched: list[RuleEvaluation] = field(default_factory=list)
    potential: list[RuleEvaluation] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    requires_human_review: bool = False
    escalation_reasons: list[EscalationReason] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def cited_rules(self) -> list[str]:
        return [e.rule.rule_id for e in self.matched]

    def as_dict(self) -> dict[str, Any]:
        return {
            "urgency": self.urgency.value,
            "department": self.department,
            "matched_rules": self.cited_rules,
            "matched": [e.as_dict() for e in self.matched],
            "potential": [e.as_dict() for e in self.potential],
            "unknowns": self.unknowns,
            "requires_human_review": self.requires_human_review,
            "escalation_reasons": [r.value for r in self.escalation_reasons],
            "notes": self.notes,
        }
