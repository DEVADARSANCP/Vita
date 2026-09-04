"""
Patient history — age, pregnancy, conditions, and what is in the medicine box.

The medication half is the part worth explaining. A patient does not say "I am
anticoagulated". They say "I take Acitrom for my heart", and rule IN-03 - head
injury while anticoagulated, which is HIGH even when the patient looks
completely well - never fires unless something makes that connection.

So the work is split by what each side is good at. The model does one thing:
pull medication names out of a sentence, in whatever language, however
misspelled. **The mapping from name to drug class is a lookup table**, in
`data/clinical/medications.json`, and the mapping from class to triage fact is
in the same file. Neither is a judgement the model gets to make - a model that
decides for itself whether a drug is a blood thinner is a model that can be
wrong about warfarin.

VITA does not recommend, adjust, or comment on any medication. It reads the name
as a fact about the patient, in the same way it reads their age.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..core.knowledge import CLINICAL_DIR
from ..core.schema import Fact, FactSource, Tri
from .base import AgentContext, ExtractionAgent
from .fields import make_fact, number_property, read_number, read_tri, tri_property, with_evidence

logger = logging.getLogger(__name__)

#: Oldest age accepted. Beyond this the number is a misread, not a patient.
MAX_AGE_YEARS = 120


def _load_medications() -> tuple[dict[str, str], dict[str, str]]:
    """Load the name-to-class and class-to-fact tables.

    A missing or broken table is survivable: medication facts simply go
    unestablished, which leaves them UNKNOWN and routes affected cases to a
    human. That is the correct failure - the alternative would be asserting a
    patient is not anticoagulated because a JSON file would not parse.
    """
    path = CLINICAL_DIR / "medications.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("medication table unavailable (%s); medication facts will stay unknown", exc)
        return {}, {}

    names = {name.lower(): cls for name, cls in raw.get("medications", {}).items()}
    class_facts = dict(raw.get("class_facts", {}))
    return names, class_facts


class HistoryAgent(ExtractionAgent):
    """Establishes demographics, known conditions and medication-derived facts."""

    name = "history"
    provides = {
        "age_years",
        "pregnancy",
        "on_anticoagulants",
        "immunocompromised",
        "known_asthma",
        "known_cardiac_history",
    }

    def __init__(self) -> None:
        self._medications, self._class_facts = _load_medications()
        logger.info("medication table: %d names, %d classes", len(self._medications), len(self._class_facts))

    # -- request ---------------------------------------------------------

    def schema_fragment(self, ctx: AgentContext) -> dict[str, Any]:
        targets = sorted(self.provides & (ctx.wanted or self.provides))
        if not targets:
            return {}

        properties: dict[str, Any] = {}
        if "age_years" in targets:
            properties["age_years"] = number_property("The patient's age in years.")
        if "pregnancy" in targets:
            properties["pregnancy"] = tri_property("Is the patient pregnant, or possibly pregnant?")
        for fact in ("known_asthma", "known_cardiac_history", "immunocompromised"):
            if fact in targets:
                question = ctx.kb.question(fact)
                properties[fact] = tri_property(
                    question.text if question else fact.replace("_", " ")
                )

        # Always collected when this agent runs at all: a medication named in
        # passing establishes facts no direct question was asked about.
        properties["medications_named"] = {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Every medication the patient names, exactly as they wrote it, in "
                "any language or spelling. Do not translate, expand, correct or "
                "classify them - copy the names out. Empty array if none."
            ),
        }
        return with_evidence(properties, targets)

    def prompt_hint(self, ctx: AgentContext) -> str:
        return (
            "Copy out medication names exactly as the patient wrote them, without "
            "deciding what they are for. Report age only if the patient states it; "
            "never estimate it from how they write."
        )

    # -- response --------------------------------------------------------

    def build_facts(self, payload: dict[str, Any], ctx: AgentContext) -> list[Fact]:
        facts: list[Fact] = []

        age = read_number(payload, "age_years")
        if age is not None:
            if 0 < age <= MAX_AGE_YEARS:
                facts.append(make_fact("age_years", age, payload, ctx, self.name))
            else:
                logger.info("discarding implausible age_years=%s", age)

        for key in ("pregnancy", "known_asthma", "known_cardiac_history", "immunocompromised"):
            value = read_tri(payload, key)
            if value is not Tri.UNKNOWN:
                facts.append(make_fact(key, value, payload, ctx, self.name))

        facts.extend(self._facts_from_medications(payload, ctx))
        return facts

    def _facts_from_medications(self, payload: dict[str, Any], ctx: AgentContext) -> list[Fact]:
        """Turn named medications into triage facts, by table lookup only."""
        named = payload.get("medications_named") or []
        if not isinstance(named, list):
            return []

        established: dict[str, str] = {}
        for raw_name in named:
            drug_class = self._classify(str(raw_name))
            if not drug_class:
                continue
            fact = self._class_facts.get(drug_class)
            if fact:
                established.setdefault(fact, str(raw_name).strip())

        return [
            Fact(
                key=fact,
                value=Tri.TRUE,
                # Derived from a lookup table, not from the conversation. Marking
                # it as a follow-up answer would overstate what the patient was
                # actually asked.
                source=FactSource.PATIENT_VERBATIM,
                turn=ctx.turn,
                verbatim=f"patient reported taking {drug}",
                language=ctx.language,
                confidence=1.0,
                agent=f"{self.name}:medication_table",
            )
            for fact, drug in established.items()
        ]

    def _classify(self, name: str) -> str | None:
        """Look up a drug class. Exact match first, then a contained match.

        The contained match handles "Ecosprin 75" and "T. Warfarin 5mg", which
        is how a patient actually writes it. It is still a table lookup - the
        name has to be in the table to match anything.
        """
        token = name.strip().lower()
        if not token:
            return None
        if token in self._medications:
            return self._medications[token]
        for known, drug_class in self._medications.items():
            if known in token:
                return drug_class
        return None
