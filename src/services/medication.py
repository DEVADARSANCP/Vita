"""
Reading a photograph of a patient's medication.

A patient who takes Acitrom does not say "I am anticoagulated". Often they do
not say the drug name either - they say "some tablet for my heart", or they say
a name they have half-remembered and spelled wrong. But they will happily
photograph the packet, and rule IN-03 (head injury while anticoagulated, HIGH
even when the patient looks completely well) depends on somebody making that
connection.

The work is split by what each side is actually good at.

**Gemini reads the packet.** It is multimodal, so no OCR engine is needed - no
model weights, no second network dependency, nothing to download on a machine we
do not control. Measured on a mock Acitrom and Ecosprin strip: about three
seconds on `gemini-flash-lite-latest`, well inside the request budget.

**A lookup table decides what the drug is.** `data/clinical/medications.json`
maps name to class and class to triage fact. Gemini is asked only to copy the
printed text; it is never asked whether something is a blood thinner, because a
model that answers that question for itself is a model that can be wrong about
warfarin.

Anything the table does not recognise is reported to the clinician as a name we
read but could not classify, rather than dropped. A drug we do not know about is
information a human may still want.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.knowledge import CLINICAL_DIR
from ..llm.gemini import GeminiClient

logger = logging.getLogger(__name__)

#: Largest photograph accepted. Phone cameras produce far larger files than this
#: needs; anything bigger is resized by the browser before upload.
MAX_IMAGE_BYTES = 6 * 1024 * 1024

ACCEPTED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic"}

_SCHEMA = {
    "type": "object",
    "properties": {
        "readable": {
            "type": "boolean",
            "description": "False if the photo is too blurred, dark or cropped to read.",
        },
        "medications": {
            "type": "array",
            "description": "Every medication name printed on the packet.",
            "items": {
                "type": "object",
                "properties": {
                    "name_as_printed": {
                        "type": "string",
                        "description": "The brand or trade name exactly as printed. Do not correct spelling.",
                    },
                    "generic_name": {
                        "type": "string",
                        "description": "The generic name if it is also printed. Empty if it is not - do not supply one from your own knowledge.",
                    },
                    "strength": {"type": "string", "description": "Strength as printed, e.g. '75mg'. Empty if absent."},
                },
            },
        },
        "note": {
            "type": "string",
            "description": "Anything that would help a person reading this, e.g. 'label partly obscured'.",
        },
    },
}

_PROMPT = (
    "This is a photograph of medication a patient has brought to a hospital "
    "intake desk. Read every medication name printed on it.\n\n"
    "Copy the names exactly as printed, including spelling. Include the generic "
    "name and strength only if they are also printed on the packet. Do not add a "
    "generic name from your own knowledge, do not guess at a blurred word, and do "
    "not say what any of it is for - that is decided elsewhere from a reference "
    "table. If you cannot read the packet, say so."
)


@dataclass
class MedicationReading:
    """What was read from one photograph, and what it established."""

    readable: bool = False
    names: list[dict[str, str]] = field(default_factory=list)
    #: Fact keys the recognised drugs establish, e.g. on_anticoagulants.
    facts: dict[str, str] = field(default_factory=dict)
    #: Names read but not found in the reference table.
    unrecognised: list[str] = field(default_factory=list)
    #: Which name established each fact, for the clinician to check.
    attribution: dict[str, str] = field(default_factory=dict)
    note: str = ""
    error: str = ""
    elapsed_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "readable": self.readable,
            "medications": self.names,
            "facts": self.facts,
            "unrecognised": self.unrecognised,
            "attribution": self.attribution,
            "note": self.note,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
        }

    def summary(self) -> str:
        if self.error:
            return f"Could not read the photo: {self.error}"
        if not self.readable or not self.names:
            return "Nothing readable on the photo."
        printed = ", ".join(m.get("name_as_printed", "") for m in self.names if m.get("name_as_printed"))
        established = ", ".join(self.facts) or "nothing the triage rules use"
        return f"Read: {printed}. Establishes: {established}."


class MedicationReader:
    """Reads medication photographs and maps what it finds to triage facts."""

    def __init__(self, llm: GeminiClient) -> None:
        self.llm = llm
        self._names, self._class_facts = self._load_table()

    @staticmethod
    def _load_table() -> tuple[dict[str, str], dict[str, str]]:
        path = CLINICAL_DIR / "medications.json"
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("medication table unavailable (%s); photos will read but not classify", exc)
            return {}, {}
        return (
            {name.lower(): cls for name, cls in raw.get("medications", {}).items()},
            dict(raw.get("class_facts", {})),
        )

    def read(self, image: bytes, mime_type: str) -> MedicationReading:
        """Read a photograph and work out what it establishes."""
        if not image:
            return MedicationReading(error="no image supplied")
        if len(image) > MAX_IMAGE_BYTES:
            return MedicationReading(error="image is too large; please send a smaller photo")
        if mime_type not in ACCEPTED_TYPES:
            return MedicationReading(error=f"unsupported image type {mime_type!r}")

        outcome = self.llm.read_media_json(image, mime_type, _PROMPT, _SCHEMA)
        if not outcome.ok:
            return MedicationReading(error=outcome.error, elapsed_ms=outcome.elapsed_ms)

        data = outcome.data if isinstance(outcome.data, dict) else {}
        names = [m for m in (data.get("medications") or []) if isinstance(m, dict)]

        reading = MedicationReading(
            readable=bool(data.get("readable", bool(names))),
            names=names,
            note=str(data.get("note", "")),
            elapsed_ms=outcome.elapsed_ms,
        )

        for entry in names:
            printed = str(entry.get("name_as_printed", "")).strip()
            generic = str(entry.get("generic_name", "")).strip()
            drug_class = self._classify(printed) or self._classify(generic)
            if not drug_class:
                if printed:
                    reading.unrecognised.append(printed)
                continue
            fact = self._class_facts.get(drug_class)
            if fact and fact not in reading.facts:
                reading.facts[fact] = "true"
                reading.attribution[fact] = printed or generic

        logger.info(
            "medication photo read in %dms: %s", reading.elapsed_ms, reading.summary()
        )
        return reading

    def _classify(self, name: str) -> str | None:
        """Look up a drug class. Exact match first, then a contained match.

        The contained match handles "Ecosprin 75" and "T. Warfarin 5mg", which is
        how a packet is actually printed. It is still a table lookup - the name
        has to be in the table to match anything.
        """
        token = (name or "").strip().lower()
        if not token:
            return None
        if token in self._names:
            return self._names[token]
        return next((cls for known, cls in self._names.items() if known in token), None)
