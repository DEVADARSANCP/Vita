"""
Timeline extraction — turning "since this morning" into hours.

Onset is a rule input, not colour: CP-07 asks whether new chest pain began
within the last twelve hours, and FV-07 asks whether a fever has run five days
or more. Both need a number, and patients give neither. They say "since I woke
up", "a couple of days", "on and off for weeks".

So the model does the language half - resolving a phrase against the current
time - and this module does the arithmetic half, checks the result is a duration
a human could actually have, and refuses anything it cannot verify. An onset of
"about 3" from a model that misread the question would silently satisfy a
twelve-hour cardiac window, so implausible values are dropped rather than
clamped into range.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.schema import Fact, Tri
from .base import AgentContext, ExtractionAgent
from .fields import make_fact, number_property, read_number, read_tri, tri_property, with_evidence

logger = logging.getLogger(__name__)

#: Longest onset accepted, in hours - five years. Anything beyond this is a
#: misparse, not a patient. Rejecting it keeps a garbled number from satisfying
#: a "more than a week" condition by accident.
MAX_ONSET_HOURS = 24 * 365 * 5

#: Longest duration accepted, in days.
MAX_DURATION_DAYS = 365 * 5


class TimelineAgent(ExtractionAgent):
    """Establishes when the problem started and how quickly it came on."""

    name = "timeline"
    provides = {"onset_hours", "duration_days", "sudden_onset", "sudden_severe_onset"}

    def schema_fragment(self, ctx: AgentContext) -> dict[str, Any]:
        targets = sorted(self.provides & (ctx.wanted or self.provides))
        if not targets:
            return {}

        properties: dict[str, Any] = {}
        if "onset_hours" in targets:
            properties["onset_hours"] = number_property(
                "How many hours ago did the problem begin? Convert the patient's "
                "own phrasing: 'this morning' is roughly 6, 'yesterday' 24, "
                "'last week' 168."
            )
        if "duration_days" in targets:
            properties["duration_days"] = number_property(
                "How many days has the patient had this for?"
            )
        if "sudden_onset" in targets:
            properties["sudden_onset"] = tri_property(
                "Did it begin abruptly, over seconds or minutes, rather than building gradually?"
            )
        if "sudden_severe_onset" in targets:
            properties["sudden_severe_onset"] = tri_property(
                "Did it begin abruptly AND at severe intensity straight away?"
            )
        return with_evidence(properties, targets)

    def prompt_hint(self, ctx: AgentContext) -> str:
        return (
            "For timing, convert the patient's own words into the unit asked for. "
            "If they gave a range, take the earlier end - a problem that started "
            "'two or three days ago' started two days ago. If they gave nothing, "
            "answer 'unknown'. Never guess a duration from the severity of the "
            "complaint."
        )

    def build_facts(self, payload: dict[str, Any], ctx: AgentContext) -> list[Fact]:
        facts: list[Fact] = []

        onset = read_number(payload, "onset_hours")
        if onset is not None:
            if 0 <= onset <= MAX_ONSET_HOURS:
                facts.append(make_fact("onset_hours", onset, payload, ctx, self.name))
            else:
                logger.info("discarding implausible onset_hours=%s", onset)

        duration = read_number(payload, "duration_days")
        if duration is not None:
            if 0 <= duration <= MAX_DURATION_DAYS:
                facts.append(make_fact("duration_days", duration, payload, ctx, self.name))
            else:
                logger.info("discarding implausible duration_days=%s", duration)

        for key in ("sudden_onset", "sudden_severe_onset"):
            value = read_tri(payload, key)
            if value is not Tri.UNKNOWN:
                facts.append(make_fact(key, value, payload, ctx, self.name))

        return facts
