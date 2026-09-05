"""
The agent contract — the one kind of fact production that is not the planner.

An earlier design had a registry of agents, each owning a slice of a shared
model call: one for symptoms, one for durations, one for vitals. The MCP
rebuild replaced all of it. The planner makes the call, chooses the questions
and records what it hears through `record_facts`, and the registry, the
extraction agents and the schema-fragment machinery went with it.

What survives is the half that never needed a model. A deterministic agent runs
in plain Python - no call, no network, no key - which is why the most dangerous
presentations are still caught in OFFLINE mode and why the red-flag pass runs
before the planner gets a turn. Somebody who has just written that they cannot
breathe should not wait on a network round trip.

`RedFlagAgent` is the only implementation, and one is enough for the contract to
earn its place: it keeps that pass behind an interface the planner cannot reach
around, and it keeps `provides` declared where the data-integrity check in
`red_flag.py` can verify the flags file against it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..core.knowledge import KnowledgeBase
from ..core.schema import Complaint, Fact

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Everything an agent may look at when producing facts."""

    message: str
    turn: int
    kb: KnowledgeBase
    language: str = "en"
    complaint: Complaint = Complaint.UNDETERMINED
    known: dict[str, Fact] = field(default_factory=dict)


class Agent(ABC):
    """Base for everything that can produce facts."""

    #: Stable identifier, recorded on every fact the agent produces.
    name: str = "agent"

    #: Fact keys this agent knows how to establish.
    provides: set[str] = set()

    #: True if the agent works without a model call.
    deterministic: bool = False


class DeterministicAgent(Agent):
    """An agent that produces facts with no model call."""

    deterministic = True

    @abstractmethod
    def run(self, ctx: AgentContext) -> list[Fact]:
        """Produce facts from the context alone."""
