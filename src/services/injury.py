"""
Looking at a photograph of an injury.

A patient can hold up their arm faster than they can describe what is wrong with
it, and "there is a rod in my leg" is a sentence some people will not think to
say because it seems obvious to them. A photograph carries it in one step.

The line this module walks carefully: **it reports what is visibly present, and
nothing else.** Not what caused it, not how bad it is, not what should be done
about it. "There is a metal rod through the lower leg" is an observation. "This
is an open tibial fracture" is a diagnosis, and VITA does not make those.

So the model is asked for a short list of things that can be seen, each mapped
to a triage fact the rules already understand. The rules then do what they
always do. A photograph is a faster way of establishing `foreign_object_embedded`
than asking about it - it is not a second opinion.

Some things are deliberately not asked for. No estimate of blood loss, no burn
percentage, no fracture, no wound depth. Each of those is a clinical judgement
made in person, and a number invented from a photograph would be recorded with
the same weight as one somebody measured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..llm.gemini import GeminiClient

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 6 * 1024 * 1024
ACCEPTED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic"}

#: What can be seen, and the triage fact it establishes. Every one of these is
#: an observation a person could make from across the room - which is the test
#: for whether it belongs here at all.
_OBSERVATIONS: dict[str, str] = {
    "object_embedded_in_wound": "foreign_object_embedded",
    "part_severed_or_crushed": "amputation_or_degloving",
    "actively_bleeding": "bleeding_uncontrolled",
    "limb_visibly_deformed": "deformity_visible",
}

_SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {
            "type": "boolean",
            "description": "False if the photo is too dark, blurred or unclear to tell anything from.",
        },
        "description": {
            "type": "string",
            "description": (
                "One or two plain sentences describing only what is visible, for a "
                "clinician to read. Say what you can see, e.g. 'a metal rod "
                "passing through the left calf, some blood on the skin around it'. "
                "Do not name an injury, do not say how serious it is, and do not "
                "say what should be done."
            ),
        },
        "object_embedded_in_wound": {
            "type": "string", "enum": ["true", "false", "unclear"],
            "description": "Is a foreign object - metal, glass, wood, a blade - visibly still in the wound?",
        },
        "part_severed_or_crushed": {
            "type": "string", "enum": ["true", "false", "unclear"],
            "description": "Is a body part visibly severed, crushed, or has skin been torn away?",
        },
        "actively_bleeding": {
            "type": "string", "enum": ["true", "false", "unclear"],
            "description": "Is blood visibly flowing right now, as opposed to dried or on a dressing?",
        },
        "limb_visibly_deformed": {
            "type": "string", "enum": ["true", "false", "unclear"],
            "description": "Is a limb bent, angled or out of shape in a way it should not be?",
        },
        "body_part": {
            "type": "string",
            "description": "Which part of the body this is, if it can be told. Empty if not.",
        },
    },
}

_PROMPT = (
    "This is a photograph a patient has taken of their own injury at a hospital "
    "intake desk. A triage nurse will read what you write.\n\n"
    "Describe only what is visible. Answer each observation with 'true' only if "
    "you can actually see it, 'false' only if you can see that it is not the "
    "case, and 'unclear' whenever the photograph does not settle it - which will "
    "often be the answer, and is a better one than a guess.\n\n"
    "Do not name an injury or a condition. Do not say how serious it is. Do not "
    "say what should be done about it. A rule engine decides urgency from what "
    "you report, so a detail you invent becomes a clinical decision nobody "
    "checked, and there is nothing further down that can tell it was invented."
)


@dataclass
class InjuryReading:
    """What could be seen in one photograph."""

    usable: bool = False
    description: str = ""
    body_part: str = ""
    #: Fact keys the visible findings establish.
    facts: dict[str, str] = field(default_factory=dict)
    #: Observations the photograph could not settle either way.
    unclear: list[str] = field(default_factory=list)
    error: str = ""
    elapsed_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "usable": self.usable,
            "description": self.description,
            "body_part": self.body_part,
            "facts": self.facts,
            "unclear": self.unclear,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
        }

    def summary(self) -> str:
        if self.error:
            return f"Could not read the photo: {self.error}"
        if not self.usable:
            return "Photo too unclear to tell anything from."
        established = ", ".join(self.facts) or "nothing the triage rules use"
        return f"{self.description[:120]} Establishes: {established}."


class InjuryReader:
    """Reads injury photographs into triage facts."""

    def __init__(self, llm: GeminiClient) -> None:
        self.llm = llm

    @property
    def available(self) -> bool:
        return self.llm.available

    def read(self, image: bytes, mime_type: str) -> InjuryReading:
        if not image:
            return InjuryReading(error="no image supplied")
        if len(image) > MAX_IMAGE_BYTES:
            return InjuryReading(error="image is too large; please send a smaller photo")

        base = mime_type.split(";")[0].strip().lower()
        if base not in ACCEPTED_TYPES:
            return InjuryReading(error=f"unsupported image type {base!r}")
        if not self.available:
            return InjuryReading(error="photo reading is unavailable; please describe it instead")

        outcome = self.llm.read_media_json(image, base, _PROMPT, _SCHEMA)
        if not outcome.ok:
            return InjuryReading(error=outcome.error, elapsed_ms=outcome.elapsed_ms)

        data = outcome.data if isinstance(outcome.data, dict) else {}
        reading = InjuryReading(
            usable=bool(data.get("usable", False)),
            description=str(data.get("description", "")).strip(),
            body_part=str(data.get("body_part", "")).strip(),
            elapsed_ms=outcome.elapsed_ms,
        )

        if not reading.usable:
            return reading

        # An injury photograph always establishes that there is an injury.
        reading.facts["injury"] = "true"

        for observation, fact in _OBSERVATIONS.items():
            answer = str(data.get(observation, "unclear")).strip().lower()
            if answer == "true":
                reading.facts[fact] = "true"
            elif answer == "false":
                # A negative from a photograph is worth recording: it is how a
                # low-urgency rule becomes reachable at all, since those require
                # their red flags to be explicitly excluded rather than merely
                # unmentioned.
                reading.facts[fact] = "false"
            else:
                reading.unclear.append(fact)

        logger.info("injury photo read in %dms: %s", reading.elapsed_ms, reading.summary())
        return reading
