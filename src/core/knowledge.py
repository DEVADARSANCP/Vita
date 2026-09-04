"""
Loading the clinical knowledge base, and refusing to run on a broken one.

The rules, the follow-up questions and the red-flag patterns are data, not code,
so they can be reviewed by someone who does not read Python. That only works if
the data is checked: a rule requiring a fact no question can establish is a rule
that can never be resolved, and a patient would sit in a loop being asked
nothing while a HIGH rule stayed open forever.

So the knowledge base validates itself on load. Structural problems raise;
survivable ones are logged and the affected rule is dropped rather than allowed
to poison the rule set. Loading happens once at startup, well inside the
90-second budget, because it is a few JSON files and no network.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import DATA_DIR
from .schema import Complaint, Rule, Urgency

logger = logging.getLogger(__name__)

CLINICAL_DIR = DATA_DIR / "clinical"

#: Facts that are never asked about directly because they are established from
#: records or from a red-flag match rather than from a question.
_DERIVED_FACTS = {"prior_visit_72h_same_complaint"}


class KnowledgeError(RuntimeError):
    """The knowledge base is unusable. Raised at startup, never mid-request."""


@dataclass
class Question:
    """The wording used to establish one fact."""

    fact: str
    text: str
    type: str = "boolean"
    askable: bool = True
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"fact": self.fact, "text": self.text, "type": self.type}


@dataclass
class RedFlag:
    """A deterministic pattern that fires before any model call."""

    id: str
    label: str
    patterns: list[str]
    facts: dict[str, Any]
    urgency: Urgency
    department: str
    rationale: str
    out_of_scope: bool = False

    def matches(self, text: str) -> bool:
        lowered = text.lower()
        return any(pattern.lower() in lowered for pattern in self.patterns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "urgency": self.urgency.value,
            "department": self.department,
            "rationale": self.rationale,
            "out_of_scope": self.out_of_scope,
        }


@dataclass
class KnowledgeBase:
    """Everything the deterministic side of VITA reasons over."""

    rules: list[Rule] = field(default_factory=list)
    questions: dict[str, Question] = field(default_factory=dict)
    red_flags: list[RedFlag] = field(default_factory=list)
    disclaimer: str = ""
    version: str = "0"

    # -- lookups ---------------------------------------------------------

    def rule(self, rule_id: str) -> Rule | None:
        return next((r for r in self.rules if r.rule_id == rule_id), None)

    def rules_for(self, complaint: Complaint) -> list[Rule]:
        """Every rule that could apply to this complaint.

        Selection is a deterministic lookup, never a retrieval. Embedding
        search is used elsewhere in VITA, but not here: a similarity miss that
        drops a HIGH rule would under-triage a patient silently, and no
        retrieval quality is worth that. General rules always come along,
        because a re-presentation or an age modifier applies whatever the
        complaint is.
        """
        general = [r for r in self.rules if r.complaint.value == "general"]
        if complaint in (Complaint.UNDETERMINED, Complaint.OUT_OF_SCOPE):
            return general
        specific = [r for r in self.rules if r.complaint is complaint]
        return specific + general

    def question(self, fact: str) -> Question | None:
        return self.questions.get(fact)

    def match_red_flags(self, text: str) -> list[RedFlag]:
        return [flag for flag in self.red_flags if flag.matches(text)]

    def summary(self) -> dict[str, Any]:
        by_complaint: dict[str, int] = {}
        for rule in self.rules:
            by_complaint[rule.complaint.value] = by_complaint.get(rule.complaint.value, 0) + 1
        return {
            "version": self.version,
            "rules": len(self.rules),
            "rules_by_complaint": by_complaint,
            "questions": len(self.questions),
            "red_flags": len(self.red_flags),
            "disclaimer": self.disclaimer,
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise KnowledgeError(f"missing knowledge file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise KnowledgeError(f"{path.name} is not valid JSON: {exc}") from exc


def load_knowledge_base(directory: Path | None = None) -> KnowledgeBase:
    """Load and validate the knowledge base. Raises if it cannot be trusted."""
    directory = directory or CLINICAL_DIR

    rules_raw = _read_json(directory / "rules.json")
    questions_raw = _read_json(directory / "questions.json")
    flags_raw = _read_json(directory / "red_flags.json")

    questions = {
        fact: Question(
            fact=fact,
            text=spec["text"],
            type=spec.get("type", "boolean"),
            askable=bool(spec.get("askable", True)),
            note=spec.get("note", ""),
        )
        for fact, spec in questions_raw.get("questions", {}).items()
    }

    red_flags = [
        RedFlag(
            id=spec["id"],
            label=spec["label"],
            patterns=list(spec.get("patterns", [])),
            facts=dict(spec.get("facts", {})),
            urgency=Urgency(spec["urgency"]),
            department=spec["department"],
            rationale=spec.get("rationale", ""),
            out_of_scope=bool(spec.get("out_of_scope", False)),
        )
        for spec in flags_raw.get("flags", [])
    ]

    rules = _load_rules(rules_raw, questions)

    kb = KnowledgeBase(
        rules=rules,
        questions=questions,
        red_flags=red_flags,
        disclaimer=rules_raw.get("disclaimer", ""),
        version=str(rules_raw.get("version", "0")),
    )

    logger.info(
        "knowledge base loaded: %d rules, %d questions, %d red flags",
        len(kb.rules),
        len(kb.questions),
        len(kb.red_flags),
    )
    return kb


def _load_rules(raw: dict[str, Any], questions: dict[str, Question]) -> list[Rule]:
    """Parse rules, dropping any that could never be resolved.

    A rule that depends on a fact with no way to establish it is worse than a
    missing rule: it sits permanently in POTENTIAL, holds the conversation open,
    and eventually escalates every case that touches it. Dropping it loudly is
    the honest failure.
    """
    rules: list[Rule] = []
    seen: set[str] = set()

    for spec in raw.get("rules", []):
        rule_id = spec.get("rule_id", "<unnamed>")
        try:
            rule = Rule.from_dict(spec)
        except (KeyError, ValueError) as exc:
            logger.error("rule %s is malformed and was dropped: %s", rule_id, exc)
            continue

        if rule.rule_id in seen:
            raise KnowledgeError(
                f"duplicate rule id {rule.rule_id!r} - rule ids are cited in triage "
                "notes and must be unique"
            )
        seen.add(rule.rule_id)

        unresolvable = [
            fact
            for fact in rule.required_facts
            if fact not in questions and fact not in _DERIVED_FACTS
        ]
        if unresolvable:
            logger.error(
                "rule %s was dropped: no follow-up question can establish %s",
                rule.rule_id,
                ", ".join(unresolvable),
            )
            continue

        if not rule.rationale:
            logger.warning(
                "rule %s has no rationale; it will be cited without an explanation",
                rule.rule_id,
            )

        rules.append(rule)

    if not rules:
        raise KnowledgeError("no usable rules were loaded - VITA cannot triage anything")

    return rules
