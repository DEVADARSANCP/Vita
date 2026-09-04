"""
Emergency transport requests — offered by the system, raised only by a person.

The whole module exists to enforce one distinction. VITA may *offer* transport
when triage has graded a case HIGH or CRITICAL. It may not *request* it. A
request is created only when a person explicitly confirms, and only once a
pickup location has been given.

That is not caution for its own sake. An automated system that can summon an
ambulance from its own reading of a sentence is a system where a
misclassification dispatches a vehicle, and where anyone who can type into the
intake box can dispatch one deliberately. The confirmation step is what stands
between those two failure modes and the road.

The second distinction, kept in the naming throughout: a request is a request
for dispatch, not a dispatch. Allocation is made by emergency operations, who
see the request the moment it is raised. VITA records what was asked for and by
whom, and stops there.

For this build the dispatch side is simulated - there is no integration with a
real service, and there should not be. What is real is the workflow: the
eligibility gate, the explicit confirmation, the location requirement, and the
record.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..core.schema import Urgency

logger = logging.getLogger(__name__)

#: Urgency at or above which transport may be offered. Below this the option is
#: not shown, because offering it implies a judgement the rules did not make.
OFFER_AT = Urgency.HIGH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AmbulanceError(RuntimeError):
    """A request was attempted that the workflow does not permit."""


@dataclass
class AmbulanceRequest:
    """One request for emergency transport."""

    request_id: str
    case_id: str
    priority: str
    pickup_location: str
    confirmed_by: str
    destination: str = "Emergency Department"
    status: str = "dispatch_requested"
    at: str = field(default_factory=_now)
    simulated: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "case_id": self.case_id,
            "priority": self.priority,
            "pickup_location": self.pickup_location,
            "destination": self.destination,
            "confirmed_by": self.confirmed_by,
            "status": self.status,
            "at": self.at,
            "simulated": self.simulated,
        }


class AmbulanceService:
    """Records transport requests. Never raises one on its own initiative."""

    def __init__(self) -> None:
        self._requests: list[AmbulanceRequest] = []
        self._lock = threading.Lock()

    # -- eligibility -----------------------------------------------------

    @staticmethod
    def may_offer(urgency: str | Urgency | None) -> bool:
        """Is this case urgent enough for transport to be offered at all?

        Note what this does not do: it reads the urgency the rule engine already
        produced. It does not form a view of its own about how unwell the
        patient is, because that view would be a second, unaudited triage
        decision sitting beside the real one.
        """
        if urgency is None:
            return False
        try:
            level = urgency if isinstance(urgency, Urgency) else Urgency(str(urgency).upper())
        except ValueError:
            return False
        return level.rank >= OFFER_AT.rank

    # -- requests --------------------------------------------------------

    def create(
        self,
        *,
        case_id: str,
        urgency: str,
        pickup_location: str,
        confirmed_by: str,
    ) -> AmbulanceRequest:
        """Raise a transport request. Every argument here is a precondition.

        Raises rather than returning an error object: a caller that reaches this
        without a confirmation or a location has a bug, and failing quietly
        would leave a patient believing transport was on its way.
        """
        location = (pickup_location or "").strip()
        confirmer = (confirmed_by or "").strip()

        if not self.may_offer(urgency):
            raise AmbulanceError(
                f"transport may not be requested at urgency {urgency!r}; "
                f"the threshold is {OFFER_AT.value}"
            )
        if not location:
            raise AmbulanceError("a pickup location must be confirmed before requesting transport")
        if not confirmer:
            raise AmbulanceError(
                "a transport request requires explicit confirmation by a person; "
                "VITA does not raise one on its own"
            )

        request = AmbulanceRequest(
            request_id=f"AMB-{uuid.uuid4().hex[:5].upper()}",
            case_id=case_id,
            priority=str(urgency).upper(),
            pickup_location=location,
            confirmed_by=confirmer,
        )

        with self._lock:
            self._requests.append(request)

        logger.info(
            "transport requested: %s for case %s, priority %s, confirmed by %s",
            request.request_id,
            case_id,
            request.priority,
            confirmer,
        )
        return request

    # -- reading ---------------------------------------------------------

    @property
    def requests(self) -> list[AmbulanceRequest]:
        with self._lock:
            return list(reversed(self._requests))

    def for_case(self, case_id: str) -> list[AmbulanceRequest]:
        return [r for r in self.requests if r.case_id == case_id]

    def get(self, request_id: str) -> AmbulanceRequest | None:
        return next((r for r in self.requests if r.request_id == request_id), None)
