"""
Complaint classification — deciding which rule set applies, or that none does.

Everything downstream depends on this. The complaint selects the rules, the
rules select the questions, the questions drive the conversation. Get it wrong
and a correct rule engine produces a confident answer to the wrong question.

The classification that matters most is the one that refuses. VITA covers five
walk-in complaints. A patient describing a stroke, an obstetric emergency, a
mental health crisis or a paediatric illness has a real problem this rule set
cannot triage, and the honest output is to say so and hand over - not to file it
under whichever of the five it superficially resembles. `OUT_OF_SCOPE` is
therefore a first-class answer with the same standing as the other five, and
`UNDETERMINED` is available for a message too vague to place at all.

The agent also reports the language the patient wrote in, so replies come back
in the same one and the verbatim record keeps their own words.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.schema import Complaint, Fact, FactSource
from .base import AgentContext, ExtractionAgent
from .fields import EVIDENCE_SUFFIX

logger = logging.getLogger(__name__)

#: Confidence below which a classification is not acted on. A low-confidence
#: guess between two rule sets is worse than admitting the complaint is
#: undetermined and asking the patient to say more.
MIN_CONFIDENCE = 0.55

#: The pseudo-fact key under which the classification is recorded, so it flows
#: through the same provenance machinery as every other fact.
COMPLAINT_KEY = "complaint"

_CHOICES = [
    "fever",
    "injury",
    "chest_pain",
    "breathing_difficulty",
    "abdominal_pain",
    "out_of_scope",
    "undetermined",
]


class ComplaintAgent(ExtractionAgent):
    """Places the patient's description into a covered rule set, or refuses to."""

    name = "complaint"
    provides = {COMPLAINT_KEY, "language"}

    def applies(self, ctx: AgentContext) -> bool:
        # Runs while the complaint is unsettled, and on the first turn always.
        return ctx.complaint in (Complaint.UNDETERMINED, Complaint.OUT_OF_SCOPE) or ctx.turn <= 1

    def schema_fragment(self, ctx: AgentContext) -> dict[str, Any]:
        return {
            COMPLAINT_KEY: {
                "type": "string",
                "enum": _CHOICES,
                "description": (
                    "Which of the covered complaints the patient is describing. "
                    "Use 'out_of_scope' when the patient clearly has a medical "
                    "problem that is none of the five - a suspected stroke, a "
                    "pregnancy complication, a mental health crisis, a child's "
                    "illness, an eye or dental problem. Use 'undetermined' when "
                    "the message is too vague to place at all."
                ),
            },
            "complaint_confidence": {
                "type": "number",
                "description": "How confident this classification is, from 0 to 1.",
            },
            f"{COMPLAINT_KEY}{EVIDENCE_SUFFIX}": {
                "type": "string",
                "description": "The words in the message that decided the classification.",
            },
            "language": {
                "type": "string",
                "description": (
                    "BCP-47 code for the language the patient wrote in - 'en', "
                    "'ml' for Malayalam, 'hi' for Hindi, and so on."
                ),
            },
        }

    def prompt_hint(self, ctx: AgentContext) -> str:
        return (
            "Classify the complaint, and prefer 'out_of_scope' over a poor fit. "
            "Forcing a stroke or an obstetric emergency into one of the five "
            "covered complaints is a worse error than declining to classify it."
        )

    def build_facts(self, payload: dict[str, Any], ctx: AgentContext) -> list[Fact]:
        raw = str(payload.get(COMPLAINT_KEY, "")).strip().lower()
        if raw not in _CHOICES:
            return []

        confidence = _confidence(payload.get("complaint_confidence"))
        complaint = Complaint(raw)

        # A hesitant guess between rule sets is not usable. Undetermined keeps
        # the conversation open instead of committing to the wrong questions.
        if complaint not in (Complaint.OUT_OF_SCOPE, Complaint.UNDETERMINED) and confidence < MIN_CONFIDENCE:
            logger.info(
                "complaint %s reported at confidence %.2f, below %.2f - treating as undetermined",
                raw,
                confidence,
                MIN_CONFIDENCE,
            )
            complaint = Complaint.UNDETERMINED

        evidence = str(payload.get(f"{COMPLAINT_KEY}{EVIDENCE_SUFFIX}", "") or "").strip()
        language = str(payload.get("language", "") or ctx.language).strip() or "en"

        return [
            Fact(
                key=COMPLAINT_KEY,
                value=complaint.value,
                source=FactSource.PATIENT_VERBATIM,
                turn=ctx.turn,
                verbatim=evidence or ctx.message,
                language=language,
                confidence=confidence,
                agent=self.name,
            ),
            Fact(
                key="language",
                value=language,
                source=FactSource.PATIENT_VERBATIM,
                turn=ctx.turn,
                verbatim="",
                language=language,
                confidence=confidence,
                agent=self.name,
            ),
        ]


def _confidence(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))
