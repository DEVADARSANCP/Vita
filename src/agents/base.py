"""
The agent contract — one file per kind of fact VITA can establish.

ACPIA's agents varied by evidence modality: an image agent, a video agent, an
OCR agent, each returning the same `Claim`. VITA's vary by **fact type**: one
knows how to read symptoms out of a sentence, one knows that "since this
morning" is a duration, one knows that 101 is Fahrenheit and 38.5 is Celsius,
one knows that warfarin is an anticoagulant. They all return the same `Fact`,
so the rule engine can consume everything without knowing how any of it was
produced.

Two kinds, and the split is the point:

* **Deterministic agents** run in plain Python. No model, no network. They keep
  working when Gemini is unreachable, which is why the most dangerous
  presentations are caught even in OFFLINE mode.
* **Extraction agents** need language understanding. Rather than each making its
  own model call - ten agents, ten round trips, and the 60-second request budget
  gone - every active one contributes a **schema fragment** and a **prompt
  hint**. The orchestrator merges them into a single structured request, makes
  one call, and hands each agent its own slice of the response to validate and
  convert.

Each agent declares the facts it `provides`. That is what lets the orchestrator
work backwards from the rule engine: the engine names the fact blocking the most
urgent unresolved rule, and the registry says which agent can establish it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

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

    #: The facts the rule engine is currently waiting on. Extraction agents
    #: narrow their schema to these so the model is asked a focused question
    #: rather than invited to volunteer everything it can imagine.
    wanted: set[str] = field(default_factory=set)

    #: True when the patient is answering a specific follow-up rather than
    #: describing their situation for the first time. It decides whether a fact
    #: is recorded as PATIENT_VERBATIM or FOLLOWUP_ANSWER.
    is_followup: bool = False

    #: The fact the pending follow-up question was asked about, if any.
    asked_about: str = ""


class Agent(ABC):
    """Base for everything that can produce facts."""

    #: Stable identifier, recorded on every fact the agent produces.
    name: str = "agent"

    #: Fact keys this agent knows how to establish.
    provides: set[str] = set()

    #: True if the agent works without a model call.
    deterministic: bool = False

    def applies(self, ctx: AgentContext) -> bool:
        """Should this agent run for this turn?

        The default asks whether anything the agent provides is currently
        wanted. Deterministic agents usually override this to always run, since
        they are cheap and a red flag is worth checking on every message.
        """
        return bool(self.provides & ctx.wanted)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "deterministic": self.deterministic,
            "provides": sorted(self.provides),
        }


class DeterministicAgent(Agent):
    """An agent that produces facts with no model call."""

    deterministic = True

    def applies(self, ctx: AgentContext) -> bool:
        return True

    @abstractmethod
    def run(self, ctx: AgentContext) -> list[Fact]:
        """Produce facts from the context alone."""


class ExtractionAgent(Agent):
    """An agent that contributes to the shared structured model request.

    The agent never calls the model itself. It says what it wants to know
    (`schema_fragment`), how to ask for it (`prompt_hint`), and how to read the
    answer (`build_facts`). The orchestrator owns the call.
    """

    deterministic = False

    @abstractmethod
    def schema_fragment(self, ctx: AgentContext) -> dict[str, Any]:
        """JSON-schema properties this agent wants filled in.

        Return an empty dict to sit this turn out.
        """

    def prompt_hint(self, ctx: AgentContext) -> str:
        """Extra instruction appended to the shared prompt. Optional."""
        return ""

    @abstractmethod
    def build_facts(self, payload: dict[str, Any], ctx: AgentContext) -> list[Fact]:
        """Convert this agent's slice of the model response into facts.

        Must ignore anything it does not recognise and must never invent a fact
        the payload did not contain. A key the model omitted stays unknown.
        """


class AgentRegistry:
    """The set of agents available, and which of them can answer a question."""

    def __init__(self, agents: Iterable[Agent] | None = None) -> None:
        self._agents: list[Agent] = list(agents or [])

    def register(self, agent: Agent) -> None:
        # A deterministic agent and an extraction agent claiming the same fact
        # is the intended arrangement, not a clash: the red flag agent asserts
        # `chest_pain` from a matched phrase, and the symptom agent establishes
        # it by asking. Two agents of the *same* kind claiming one fact is the
        # real problem, because one will silently overwrite the other.
        clashes = {
            fact
            for fact in agent.provides
            for existing in self._agents
            if fact in existing.provides and existing.deterministic == agent.deterministic
        }
        if clashes:
            logger.warning(
                "agent %s claims %s, already claimed by another %s agent",
                agent.name,
                ", ".join(sorted(clashes)),
                "deterministic" if agent.deterministic else "extraction",
            )
        self._agents.append(agent)

    def all(self) -> list[Agent]:
        return list(self._agents)

    def deterministic(self) -> list[DeterministicAgent]:
        return [a for a in self._agents if isinstance(a, DeterministicAgent)]

    def extraction(self) -> list[ExtractionAgent]:
        return [a for a in self._agents if isinstance(a, ExtractionAgent)]

    def active_extraction(self, ctx: AgentContext) -> list[ExtractionAgent]:
        return [a for a in self.extraction() if a.applies(ctx)]

    def provider_of(self, fact: str) -> Agent | None:
        """Which agent can establish this fact, if any."""
        return next((a for a in self._agents if fact in a.provides), None)

    def asker_of(self, fact: str) -> ExtractionAgent | None:
        """Which agent can establish this fact *by asking about it*.

        Distinct from `provider_of`, and the distinction is load-bearing. The
        red flag agent can assert `chest_pain` when it matches a phrase, but it
        cannot elicit it - there is no question it can put to the patient. When
        the rule engine reports a blocking fact, the conversation needs the
        agent that can ask, so routing a follow-up through `provider_of` would
        pick an agent with nothing to say.
        """
        return next(
            (a for a in self.extraction() if fact in a.provides),
            None,
        )

    def describe(self) -> list[dict[str, Any]]:
        return [a.describe() for a in self._agents]
