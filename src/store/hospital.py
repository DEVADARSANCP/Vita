"""
Hospital operations — departments, doctors on call, and how busy each is.

The load-bearing rule in this module is what it is *not* allowed to do.

Triage decides urgency and department from clinical facts alone. This data then
says who is on call there and how full it is. It never flows backwards. A HIGH
chest pain is still HIGH when Emergency is at capacity; what changes is the line
the clinician reads next to it - "Emergency at capacity, 9 of 12" - not the
patient's grading.

That separation is the whole point of the queue being ordered by urgency and
arrival rather than by anything in here. Allowing scarcity to lower an acuity
would be an error of a different kind from a mistuned rule: a rule can be wrong,
but a system that quietly downgrades sick patients when it is busy is wrong by
design, and the busiest moment is exactly when that behaviour causes harm.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import DATA_DIR

logger = logging.getLogger(__name__)

HOSPITAL_FILE = DATA_DIR / "hospital" / "hospital.json"


@dataclass
class Department:
    id: str
    name: str
    location: str
    open_hours: str
    capacity: int
    current_load: int

    @property
    def at_capacity(self) -> bool:
        return self.current_load >= self.capacity

    @property
    def free(self) -> int:
        return max(0, self.capacity - self.current_load)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "location": self.location,
            "open_hours": self.open_hours,
            "capacity": self.capacity,
            "current_load": self.current_load,
            "free": self.free,
            "at_capacity": self.at_capacity,
        }


@dataclass
class Doctor:
    id: str
    name: str
    specialty: str
    department: str
    on_call: bool
    email: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "specialty": self.specialty,
            "department": self.department,
            "on_call": self.on_call,
            "email": self.email,
        }


class HospitalDirectory:
    """Read-only view of the hospital's operational state."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or HOSPITAL_FILE
        self.facility: dict[str, Any] = {}
        self.departments: list[Department] = []
        self.doctors: list[Doctor] = []
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Survivable. Triage does not need this data; only routing detail
            # and the notification recipient do, and both degrade to "unknown"
            # rather than taking the intake down.
            logger.error("hospital directory unavailable (%s); routing detail will be limited", exc)
            return

        self.facility = raw.get("facility", {})
        self.departments = [Department(**d) for d in raw.get("departments", [])]
        self.doctors = [Doctor(**d) for d in raw.get("doctors", [])]
        logger.info(
            "hospital directory: %d departments, %d doctors",
            len(self.departments),
            len(self.doctors),
        )

    # -- lookups ---------------------------------------------------------

    def department(self, name: str) -> Department | None:
        return next((d for d in self.departments if d.name.lower() == name.lower()), None)

    def on_call_for(self, department: str) -> Doctor | None:
        """The doctor to notify for a department.

        Falls back to any on-call doctor rather than returning nobody: a
        high-urgency case with no recipient is a notification that silently goes
        nowhere, which is worse than one that reaches the wrong desk.
        """
        exact = [d for d in self.doctors if d.department.lower() == department.lower() and d.on_call]
        if exact:
            return exact[0]
        anyone = [d for d in self.doctors if d.on_call]
        if anyone:
            logger.info("no on-call doctor for %s; falling back to %s", department, anyone[0].name)
            return anyone[0]
        return None

    def routing_note(self, department: str) -> str:
        """One line of operational context for the clinician reading the case."""
        dept = self.department(department)
        if dept is None:
            return ""
        if dept.at_capacity:
            alternatives = [d.name for d in self.departments if not d.at_capacity and d.id != "TRI"]
            suggestion = f" Nearest with space: {alternatives[0]}." if alternatives else ""
            return (
                f"{dept.name} is at capacity ({dept.current_load} of {dept.capacity})."
                f"{suggestion} Urgency is unchanged."
            )
        return f"{dept.name}: {dept.free} of {dept.capacity} free. {dept.location}."

    def as_dict(self) -> dict[str, Any]:
        return {
            "facility": self.facility,
            "departments": [d.as_dict() for d in self.departments],
            "doctors": [d.as_dict() for d in self.doctors],
        }
