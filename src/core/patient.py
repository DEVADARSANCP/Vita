"""
Patient identity.

A name, and nothing else. No password, no date of birth, no record number.
That is enough to link a person's visits together, which is what the history
and chronic-presentation agents need, and it is the least identifying thing
that does the job.

Names are matched on a normalised form - case folded, punctuation dropped,
whitespace collapsed - so "Priya Nair", "priya nair" and "Priya  Nair" are one
patient. The display name keeps whatever the patient actually typed.

This is a demonstration identity model and it is deliberately weak. Two real
people sharing a name would share a record, which is exactly why a real system
uses a record number. Saying so plainly is better than implying a rigour that
is not here.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalise(name: str) -> str:
    """The form two spellings of one name have to agree on."""
    folded = _PUNCTUATION.sub(" ", (name or "").strip().casefold())
    return _WHITESPACE.sub(" ", folded).strip()


def patient_id(name: str) -> str:
    """A stable id for a name.

    Derived rather than assigned, so the same name reaching the system twice -
    from the intake page, from a tool call, from a seeded record - resolves to
    one patient without a lookup table to keep in sync.
    """
    key = normalise(name)
    if not key:
        return ""
    return "P-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10].upper()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Patient:
    """A person, and the visits they have made."""

    patient_id: str
    name: str
    normalised: str
    created_at: str = field(default_factory=_now)
    last_seen_at: str = field(default_factory=_now)
    visit_count: int = 0

    @classmethod
    def from_name(cls, name: str) -> "Patient":
        cleaned = (name or "").strip()
        return cls(
            patient_id=patient_id(cleaned),
            name=cleaned,
            normalised=normalise(cleaned),
        )

    @property
    def valid(self) -> bool:
        return bool(self.patient_id and self.name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
            "visit_count": self.visit_count,
        }
