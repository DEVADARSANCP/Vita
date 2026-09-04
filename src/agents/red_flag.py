"""
Red flags — the deterministic pass that runs before anything else.

Two jobs, and each would justify the module on its own.

**Safety.** A patient who writes "crushing chest pain going down my arm" should
not then be asked five follow-up questions. The pattern is recognised, the facts
are set, and the case escalates on the first turn.

**Availability.** This runs in plain Python over a lowercased string. It works
when Gemini is slow, rate-limited, out of quota or absent entirely, which means
the presentations that matter most are still caught in DEGRADED and OFFLINE
mode. A safety net that only works while the network does is not a safety net.

Patterns are listed in every language the interface offers. A red flag that only
fires in English is no use to the patient who most needs it.

Matching is deliberately dumb - lowercased substring, no stemming, no fuzzy
distance. A red flag that fires when it should not costs a clinician a glance. A
clever matcher that misses is a different kind of mistake, and this module is
built to make the first kind.
"""

from __future__ import annotations

import logging

from ..core.knowledge import RedFlag
from ..core.schema import Fact, FactSource, Tri
from .base import AgentContext, DeterministicAgent

logger = logging.getLogger(__name__)


class RedFlagAgent(DeterministicAgent):
    """Matches critical phrases and asserts the facts they imply."""

    name = "red_flag"

    #: Everything any red flag can set. Kept in sync with data/clinical/red_flags.json
    #: by `verify_coverage`, which is called at startup rather than trusted.
    provides = {
        "breathing_difficulty",
        "speaking_full_sentences",
        "chest_pain",
        "pain_radiating",
        "fainting",
        "loss_of_consciousness",
        "injury",
        "bleeding_uncontrolled",
        "head_injury",
        "on_anticoagulants",
        "lips_blue",
        "fever",
        "rash_non_blanching",
        "neck_stiffness",
        "abdominal_pain",
        "vomiting_blood",
        "rigid_abdomen",
        "pregnancy",
    }

    def run(self, ctx: AgentContext) -> list[Fact]:
        matches = ctx.kb.match_red_flags(ctx.message)
        if not matches:
            return []

        facts: list[Fact] = []
        for flag in matches:
            logger.info("red flag %s matched on turn %d", flag.id, ctx.turn)
            facts.extend(self._facts_from(flag, ctx))
        return facts

    def _facts_from(self, flag: RedFlag, ctx: AgentContext) -> list[Fact]:
        return [
            Fact(
                key=key,
                value=Tri.coerce(value),
                source=FactSource.RED_FLAG_MATCH,
                turn=ctx.turn,
                verbatim=ctx.message,
                language=ctx.language,
                confidence=1.0,
                agent=f"{self.name}:{flag.id}",
            )
            for key, value in flag.facts.items()
        ]

    @staticmethod
    def matched_flags(ctx: AgentContext) -> list[RedFlag]:
        """The flags themselves, for the caller that needs urgency and routing.

        Facts are only half of what a red flag carries. The other half - the
        urgency floor, the department, whether the presentation is out of scope
        entirely - belongs to the orchestrator, not to the fact stream.
        """
        return ctx.kb.match_red_flags(ctx.message)


def verify_coverage(agent: RedFlagAgent, flags: list[RedFlag]) -> list[str]:
    """Report facts the data can set that the agent does not declare.

    `provides` is what the orchestrator consults when working out which agent
    can answer an open question, so a fact the data sets but the agent does not
    declare is invisible to that lookup. Checking at startup turns a silent
    routing gap into a log line.
    """
    from_data = {key for flag in flags for key in flag.facts}
    undeclared = sorted(from_data - agent.provides)
    if undeclared:
        logger.warning(
            "red flag data sets facts not declared by the agent: %s",
            ", ".join(undeclared),
        )
    return undeclared
