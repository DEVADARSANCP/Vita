"""
Measurements — temperature and pain severity.

Small agent, real work. Both facts are numeric rule inputs with thresholds
attached (FV-04 fires at 40 degrees Celsius, CP-05 at severity 7), and both
arrive from patients in whatever form they happen to use.

Temperature is the interesting one. "101" and "38.5" are the same fever in
different units, and a patient will say either without naming which. Treating
101 as Celsius would clear the hyperpyrexia threshold by sixty degrees and fire
a CRITICAL rule on a routine fever. So the model reports the number and the unit
it believes was meant, and **this module does the conversion** - deterministic
arithmetic on a validated range, not a model asked to do sums.

Anything outside a range a living person can present with is discarded. A
misread number that lands in a plausible range is a wrong triage; one that lands
outside it should never reach the rule engine at all.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.schema import Fact
from .base import AgentContext, ExtractionAgent
from .fields import make_fact, number_property, read_number, with_evidence

logger = logging.getLogger(__name__)

#: Survivable body temperature in Celsius. Outside this, the reading is wrong.
MIN_TEMP_C, MAX_TEMP_C = 25.0, 45.0

#: The range a Fahrenheit reading occupies. Used only as a sanity check after
#: the model has already told us which unit it meant.
MIN_TEMP_F, MAX_TEMP_F = 77.0, 113.0


class VitalsAgent(ExtractionAgent):
    """Extracts numeric measurements and normalises their units."""

    name = "vitals"
    provides = {"temperature_c", "severity"}

    def schema_fragment(self, ctx: AgentContext) -> dict[str, Any]:
        targets = sorted(self.provides & (ctx.wanted or self.provides))
        if not targets:
            return {}

        properties: dict[str, Any] = {}
        if "temperature_c" in targets:
            properties["temperature_value"] = number_property(
                "The temperature reading the patient gave, as they gave it, "
                "without converting it."
            )
            properties["temperature_unit"] = {
                "type": "string",
                "enum": ["celsius", "fahrenheit", "unknown"],
                "description": (
                    "Which unit the reading is in. If the patient did not say, "
                    "infer from the number: readings near 37-40 are celsius, "
                    "readings near 98-104 are fahrenheit. Answer 'unknown' if "
                    "there is no reading at all."
                ),
            }
        if "severity" in targets:
            properties["severity"] = number_property(
                "Pain severity from 0 to 10, but only if the patient gave a "
                "number or an unmistakable equivalent such as 'the worst pain of "
                "my life' (10). Do not infer a score from how serious the "
                "complaint sounds."
            )
        return with_evidence(properties, ["temperature_c", "severity"])

    def build_facts(self, payload: dict[str, Any], ctx: AgentContext) -> list[Fact]:
        facts: list[Fact] = []

        celsius = self._temperature_celsius(payload)
        if celsius is not None:
            facts.append(make_fact("temperature_c", celsius, payload, ctx, self.name))

        severity = read_number(payload, "severity")
        if severity is not None:
            if 0 <= severity <= 10:
                facts.append(make_fact("severity", severity, payload, ctx, self.name))
            else:
                logger.info("discarding out-of-range severity=%s", severity)

        return facts

    def _temperature_celsius(self, payload: dict[str, Any]) -> float | None:
        """Convert the reported reading to Celsius, or discard it.

        The conversion is done here rather than by the model because it is
        arithmetic with a patient-safety threshold on the far side, and
        arithmetic is the one thing a deterministic system is unambiguously
        better at.
        """
        value = read_number(payload, "temperature_value")
        if value is None:
            return None

        unit = str(payload.get("temperature_unit", "unknown")).strip().lower()

        if unit == "fahrenheit":
            if not MIN_TEMP_F <= value <= MAX_TEMP_F:
                logger.info("discarding implausible fahrenheit reading %s", value)
                return None
            celsius = (value - 32.0) * 5.0 / 9.0
        elif unit == "celsius":
            celsius = value
        else:
            # No unit and no way to ask. Rather than guess, fall back on the
            # ranges - they do not overlap for any temperature a patient walks
            # in with, so this is a reading of the number, not of the patient.
            if MIN_TEMP_C <= value <= MAX_TEMP_C:
                celsius = value
            elif MIN_TEMP_F <= value <= MAX_TEMP_F:
                celsius = (value - 32.0) * 5.0 / 9.0
            else:
                logger.info("discarding temperature %s: no plausible unit", value)
                return None

        if not MIN_TEMP_C <= celsius <= MAX_TEMP_C:
            logger.info("discarding temperature %.1fC: outside survivable range", celsius)
            return None

        return round(celsius, 1)
