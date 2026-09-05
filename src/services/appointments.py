"""
Booking a patient in to see somebody.

Being told which department to go to is not the same as knowing when you will be
seen. A patient who is not an emergency wants a time, a name and a number they
can hold on to, and a hospital wants that patient arriving when there is
somebody free rather than joining a queue at the door.

Two things make an appointment here honest rather than decorative.

**Times come from the clinician's actual hours.** Each doctor has a clinic
window and a slot length in the hospital directory, and a booking is the next
free slot inside that window. A patient told 14:20 is told a time somebody is
present, not a number the system produced to look helpful.

**Nobody urgent gets one.** HIGH and CRITICAL patients are not booked in - they
are sent through now, and offering them a slot at half past two would be an
invitation to sit down and wait. Booking is for the patients for whom waiting is
the right answer.

Token numbers run per clinician per day, because that is what a patient reads
off a screen in a waiting room.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from ..config import RUNTIME_DIR
from ..core.schema import Urgency

logger = logging.getLogger(__name__)

DEFAULT_DB = RUNTIME_DIR / "vita.db"

#: At or above this the patient is going straight through, so there is nothing
#: to book. Anything below it waits, and waiting is better with a time on it.
WALK_THROUGH_AT = Urgency.HIGH

#: How soon the first offered slot can be. Somebody has to get to the
#: department, and a slot four minutes from now is a slot they will miss.
LEAD_MINUTES = 20

_SCHEMA = """
CREATE TABLE IF NOT EXISTS appointments (
    appointment_id TEXT PRIMARY KEY,
    case_id        TEXT NOT NULL,
    patient_id     TEXT NOT NULL DEFAULT '',
    patient_name   TEXT NOT NULL DEFAULT '',
    doctor_id      TEXT NOT NULL,
    doctor_name    TEXT NOT NULL DEFAULT '',
    department     TEXT NOT NULL DEFAULT '',
    slot_date      TEXT NOT NULL,
    slot_time      TEXT NOT NULL,
    token          INTEGER NOT NULL,
    status         TEXT NOT NULL DEFAULT 'booked',
    created_at     TEXT NOT NULL,
    note           TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_appt_slot ON appointments (doctor_id, slot_date, slot_time);
CREATE INDEX IF NOT EXISTS idx_appt_case ON appointments (case_id);
"""


def _now() -> datetime:
    return datetime.now()


def _parse(hhmm: str) -> time:
    try:
        hours, minutes = hhmm.split(":")
        return time(int(hours), int(minutes))
    except (ValueError, AttributeError):
        return time(9, 0)


@dataclass
class Appointment:
    appointment_id: str
    case_id: str
    patient_id: str
    patient_name: str
    doctor_id: str
    doctor_name: str
    department: str
    slot_date: str
    slot_time: str
    token: int
    status: str = "booked"
    created_at: str = ""
    note: str = ""

    @property
    def when(self) -> str:
        """How a person would say it."""
        try:
            when = datetime.strptime(f"{self.slot_date} {self.slot_time}", "%Y-%m-%d %H:%M")
        except ValueError:
            return f"{self.slot_date} at {self.slot_time}"
        today = date.today()
        if when.date() == today:
            return f"today at {when.strftime('%H:%M')}"
        if when.date() == today + timedelta(days=1):
            return f"tomorrow at {when.strftime('%H:%M')}"
        return when.strftime("%A %d %B at %H:%M")

    def as_dict(self) -> dict[str, Any]:
        return {
            "appointment_id": self.appointment_id,
            "case_id": self.case_id,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "doctor_id": self.doctor_id,
            "doctor_name": self.doctor_name,
            "department": self.department,
            "slot_date": self.slot_date,
            "slot_time": self.slot_time,
            "when": self.when,
            "token": self.token,
            "status": self.status,
            "created_at": self.created_at,
            "note": self.note,
        }


class AppointmentBook:
    """Finds a free slot in a clinician's day and holds it."""

    def __init__(self, hospital: Any, path: Path | None = None) -> None:
        self.hospital = hospital
        self.path = path or DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("appointment book ready at %s", self.path)

    # -- eligibility -----------------------------------------------------

    @staticmethod
    def should_book(urgency: str | Urgency | None) -> bool:
        """Is this a patient who should be given a time rather than sent through?

        Anyone HIGH or above is walking through now. Offering them a slot would
        read as an instruction to sit down and wait, which is the opposite of
        what their grading means.
        """
        if urgency is None:
            return False
        try:
            level = urgency if isinstance(urgency, Urgency) else Urgency(str(urgency).upper())
        except ValueError:
            return False
        return level.rank < WALK_THROUGH_AT.rank

    # -- booking ---------------------------------------------------------

    def book(
        self,
        *,
        case_id: str,
        patient_id: str,
        patient_name: str,
        department: str,
        urgency: str,
    ) -> Appointment | None:
        """Take the next free slot with the on-call clinician for a department."""
        if not self.should_book(urgency):
            return None

        doctor = self.hospital.on_call_for(department)
        if doctor is None:
            logger.info("no clinician for %s; nothing to book", department)
            return None

        existing = self.for_case(case_id)
        if existing:
            return existing[0]

        slot = self._next_free_slot(doctor)
        if slot is None:
            logger.info("no free slot for %s", doctor.name)
            return None

        slot_date, slot_time = slot
        appointment = Appointment(
            appointment_id=f"APT-{abs(hash(case_id + slot_time)) % 100000:05d}",
            case_id=case_id,
            patient_id=patient_id,
            patient_name=patient_name,
            doctor_id=doctor.id,
            doctor_name=doctor.name,
            department=doctor.department,
            slot_date=slot_date,
            slot_time=slot_time,
            token=self._next_token(doctor.id, slot_date),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO appointments (appointment_id, case_id, patient_id, patient_name,
                                          doctor_id, doctor_name, department, slot_date,
                                          slot_time, token, status, created_at, note)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (appointment.appointment_id, case_id, patient_id, patient_name,
                 doctor.id, doctor.name, doctor.department, slot_date, slot_time,
                 appointment.token, appointment.status, appointment.created_at, ""),
            )
            self._conn.commit()

        logger.info(
            "booked %s with %s %s, token %d",
            case_id, doctor.name, appointment.when, appointment.token,
        )
        return appointment

    def _next_free_slot(self, doctor: Any) -> tuple[str, str] | None:
        """The first slot in the clinician's day that nobody else has.

        Looks at today first and rolls into following days if the clinic is
        finished. A patient told "tomorrow at 09:20" has a real answer; one told
        nothing has to ask at the desk.
        """
        start = _parse(getattr(doctor, "clinic_start", "09:00"))
        end = _parse(getattr(doctor, "clinic_end", "17:00"))
        step = max(5, int(getattr(doctor, "slot_minutes", 15)))

        earliest = _now() + timedelta(minutes=LEAD_MINUTES)

        for day_offset in range(0, 7):
            day = date.today() + timedelta(days=day_offset)
            taken = self._taken(doctor.id, day.isoformat())

            cursor = datetime.combine(day, start)
            closing = datetime.combine(day, end)

            while cursor <= closing:
                if cursor >= earliest:
                    label = cursor.strftime("%H:%M")
                    if label not in taken:
                        return day.isoformat(), label
                cursor += timedelta(minutes=step)

        return None

    def _taken(self, doctor_id: str, day: str) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT slot_time FROM appointments "
                "WHERE doctor_id = ? AND slot_date = ? AND status <> 'cancelled'",
                (doctor_id, day),
            ).fetchall()
        return {r["slot_time"] for r in rows}

    def _next_token(self, doctor_id: str, day: str) -> int:
        """Tokens run per clinician per day - what a patient reads off a screen."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(token), 0) AS highest FROM appointments "
                "WHERE doctor_id = ? AND slot_date = ?",
                (doctor_id, day),
            ).fetchone()
        return int(row["highest"]) + 1 if row else 1

    # -- reading and changing --------------------------------------------

    def for_case(self, case_id: str) -> list[Appointment]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM appointments WHERE case_id = ? AND status <> 'cancelled' "
                "ORDER BY slot_date, slot_time",
                (case_id,),
            ).fetchall()
        return [_from_row(r) for r in rows]

    def cancel(self, appointment_id: str, note: str = "") -> bool:
        """Release a slot. Used when a clinician sends the patient through instead."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE appointments SET status='cancelled', note=? WHERE appointment_id=?",
                (note, appointment_id),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def cancel_for_case(self, case_id: str, note: str = "") -> int:
        released = 0
        for appointment in self.for_case(case_id):
            if self.cancel(appointment.appointment_id, note):
                released += 1
        return released


def _from_row(row: sqlite3.Row) -> Appointment:
    return Appointment(
        appointment_id=row["appointment_id"],
        case_id=row["case_id"],
        patient_id=row["patient_id"],
        patient_name=row["patient_name"],
        doctor_id=row["doctor_id"],
        doctor_name=row["doctor_name"],
        department=row["department"],
        slot_date=row["slot_date"],
        slot_time=row["slot_time"],
        token=int(row["token"]),
        status=row["status"],
        created_at=row["created_at"],
        note=row["note"],
    )
