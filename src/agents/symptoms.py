"""
Symptom extraction — reading yes/no clinical features out of ordinary language.

This is the agent that turns "I've had this tightness and I get puffed just
walking to the door" into `chest_pain = true, breathing_difficulty = true`. It
handles the bulk of the fact keys, and it is the one most exposed to the failure
this whole architecture exists to prevent: a model that fills in a plausible
answer for something the patient never mentioned.

Three defences.

**The schema is narrowed to what is actually wanted.** Only the facts the rule
engine is currently blocked on, plus those relevant to the presenting complaint,
appear in the request. A model not asked about neck stiffness cannot invent it.

**"unknown" is always available.** Forcing a binary choice on an unmentioned
symptom guarantees a fabricated one.

**Nothing arrives without a quotation.** A value whose evidence field is empty
is discarded rather than recorded, because a fact with no words behind it is
exactly what a hallucination looks like.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.schema import Complaint, Fact, Tri
from .base import AgentContext, ExtractionAgent
from .fields import EVIDENCE_SUFFIX, make_fact, read_tri, tri_property, with_evidence

logger = logging.getLogger(__name__)

#: Boolean clinical features this agent owns. Facts about the patient's history,
#: their timeline and their measurements belong to other agents, which read them
#: differently.
SYMPTOM_FACTS = {
    "chest_pain",
    "breathing_difficulty",
    "pain_radiating",
    "sweating",
    "nausea",
    "fainting",
    "pain_on_breathing",
    "pain_reproducible_on_pressure",
    "fever",
    "neck_stiffness",
    "rash_non_blanching",
    "confusion",
    "speaking_full_sentences",
    "lips_blue",
    "wheezing",
    "injury",
    "bleeding_uncontrolled",
    "head_injury",
    "loss_of_consciousness",
    "mechanism_high_energy",
    "deformity_visible",
    "can_bear_weight",
    "abdominal_pain",
    "rigid_abdomen",
    "vomiting_blood",
    "black_stool",
    "testicular_pain",
    "prior_visit_72h_same_complaint",
}

#: The symptom that anchors each complaint's rule set. Requested whenever the
#: complaint is still unsettled, so the patient's opening description is read
#: for what it plainly says before any question is put to them.
ANCHOR_FACTS = [
    "chest_pain",
    "breathing_difficulty",
    "abdominal_pain",
    "injury",
    "fever",
]

# High-signal facts also requested on the opening turn, before a complaint is
# settled. Anchors alone are not enough: "I banged my head and I take warfarin"
# establishes `injury` and nothing else, so IN-03 - head injury on
# anticoagulants, the presentation most likely to be under-triaged because the
# patient looks well - can never match. The opening message is the richest one
# a patient sends and it is worth reading properly.
OPENING_FACTS = [
    "head_injury",
    "loss_of_consciousness",
    "bleeding_uncontrolled",
    "pain_radiating",
    "sweating",
    "neck_stiffness",
    "rash_non_blanching",
    "confusion",
    "speaking_full_sentences",
    "lips_blue",
    "vomiting_blood",
    "rigid_abdomen",
]

#: Cap on how many symptoms are requested at once. Beyond this the model starts
#: answering the list rather than reading the message, and accuracy on the facts
#: that matter falls.
MAX_FIELDS_PER_TURN = 12

# Raised for the opening turn only, where the patient is describing their
# situation rather than answering one question.
MAX_FIELDS_OPENING = 18


class SymptomAgent(ExtractionAgent):
    """Extracts three-valued clinical features from the patient's own words."""

    name = "symptom"
    provides = SYMPTOM_FACTS

    def schema_fragment(self, ctx: AgentContext) -> dict[str, Any]:
        targets = self._targets(ctx)
        if not targets:
            return {}

        properties: dict[str, Any] = {}
        for fact in targets:
            question = ctx.kb.question(fact)
            description = question.text if question else fact.replace("_", " ")
            properties[fact] = tri_property(f"Does the patient report this? ({description})")
        return with_evidence(properties, targets)

    def prompt_hint(self, ctx: AgentContext) -> str:
        return (
            "For each symptom field, answer 'true' only if the patient's message "
            "states or clearly implies it, 'false' only if the patient denies it, "
            "and 'unknown' in every other case - including when the message simply "
            "does not mention it. Do not reason from one symptom to another: a "
            "patient reporting chest pain has not told you anything about their "
            "breathing."
        )

    def build_facts(self, payload: dict[str, Any], ctx: AgentContext) -> list[Fact]:
        facts: list[Fact] = []
        for fact in self._targets(ctx):
            value = read_tri(payload, fact)
            if value is Tri.UNKNOWN:
                continue

            evidence = str(payload.get(f"{fact}{EVIDENCE_SUFFIX}", "") or "").strip()
            if not evidence and not ctx.is_followup:
                # An asserted symptom with nothing quoted behind it is the
                # signature of a fabrication. During a follow-up the answer is
                # the whole message ("no"), so the requirement is relaxed there.
                logger.info(
                    "discarding %s=%s from turn %d: no supporting quotation",
                    fact,
                    value.value,
                    ctx.turn,
                )
                continue

            facts.append(make_fact(fact, value, payload, ctx, self.name))
        return facts

    def _targets(self, ctx: AgentContext) -> list[str]:
        """The symptoms worth asking the model about this turn.

        Wanted facts come first - those are what the rule engine is blocked on -
        followed by anything else this complaint's rules could use, so a patient
        who volunteers something unprompted is still heard.

        Before a complaint is settled there is no rule set to draw from, and
        asking for nothing means the opening description goes unmined: a patient
        who wrote "I've had a fever since yesterday" gets asked, as their first
        question, whether they have a fever. So while the complaint is
        undetermined the anchor symptoms are always requested.
        """
        wanted = [f for f in self.provides if f in ctx.wanted]

        relevant: list[str] = []
        if ctx.complaint in (Complaint.UNDETERMINED, Complaint.OUT_OF_SCOPE):
            candidates = ANCHOR_FACTS + OPENING_FACTS
            relevant = [f for f in candidates if f not in wanted]
            return (wanted + relevant)[:MAX_FIELDS_OPENING]
        else:
            for rule in ctx.kb.rules_for(ctx.complaint):
                for fact in rule.required_facts:
                    if fact in self.provides and fact not in wanted and fact not in relevant:
                        relevant.append(fact)

        ordered = wanted + relevant
        return ordered[:MAX_FIELDS_PER_TURN]
