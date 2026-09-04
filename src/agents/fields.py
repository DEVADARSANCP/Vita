"""
Shared field shapes for the structured extraction request.

Every extraction agent describes what it wants as JSON schema, and they are all
built from the same handful of shapes defined here. Two properties matter more
than the rest.

**"unknown" is always an allowed answer.** Every enum includes it and every
numeric field is nullable. A model forced to choose between "true" and "false"
about a symptom nobody mentioned will choose one, and that invented answer then
looks identical to an established fact by the time it reaches the rule engine.
Giving the model somewhere honest to put "the patient did not say" is the
cheapest hallucination control available.

**Every field carries the words it came from.** The `_evidence` companion field
holds the fragment of the patient's own message that justified the value, in the
patient's own language. It is what the hospital dashboard shows beside the
canonical English fact, so a clinician can audit the extraction rather than
trust it.
"""

from __future__ import annotations

from typing import Any

from ..core.schema import Fact, FactSource, Tri

#: Suffix for the companion field holding the supporting quotation.
EVIDENCE_SUFFIX = "_evidence"

_TRI_VALUES = ["true", "false", "unknown"]


def tri_property(description: str) -> dict[str, Any]:
    """A three-valued field. "unknown" is a first-class answer, not a failure."""
    return {
        "type": "string",
        "enum": _TRI_VALUES,
        "description": f"{description} Answer 'unknown' unless the patient's own words settle it.",
    }


def number_property(description: str) -> dict[str, Any]:
    """A numeric field that may legitimately be absent."""
    return {
        "type": "string",
        "description": (
            f"{description} Give digits only, or the exact string 'unknown' if the "
            "patient's words do not establish it. Never estimate."
        ),
    }


def evidence_property(fact: str) -> dict[str, Any]:
    return {
        "type": "string",
        "description": (
            f"The exact fragment of the patient's message that established "
            f"'{fact}', quoted verbatim in their own language. Empty string if "
            "nothing in the message established it."
        ),
    }


def with_evidence(properties: dict[str, Any], facts: list[str]) -> dict[str, Any]:
    """Add an evidence companion for each fact field."""
    for fact in facts:
        properties[f"{fact}{EVIDENCE_SUFFIX}"] = evidence_property(fact)
    return properties


# ---------------------------------------------------------------------------
# Reading the response
# ---------------------------------------------------------------------------


def read_number(payload: dict[str, Any], key: str) -> float | None:
    """Pull a number out of the response, or None if it is not one.

    The model is asked for digits or the literal 'unknown'. Anything else -
    "about 3", "a few", an empty string - is treated as not established. Parsing
    "about 3" into 3 would be inventing a precision the patient never gave.
    """
    raw = payload.get(key)
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in {"", "unknown", "null", "none", "n/a"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_tri(payload: dict[str, Any], key: str) -> Tri:
    """Pull a three-valued answer out of the response, defaulting to UNKNOWN."""
    if key not in payload:
        return Tri.UNKNOWN
    return Tri.coerce(payload.get(key))


def make_fact(
    key: str,
    value: Any,
    payload: dict[str, Any],
    ctx: Any,
    agent: str,
    *,
    confidence: float = 0.9,
) -> Fact:
    """Build a fact, attributing it to the turn and the words it came from.

    The source distinguishes a fact the patient volunteered from one a
    follow-up established. That distinction is a requirement of the triage note,
    so it is recorded at the moment the fact is created rather than reconstructed
    later from conversation history.
    """
    source = (
        FactSource.FOLLOWUP_ANSWER
        if ctx.is_followup and (ctx.asked_about == key or not ctx.asked_about)
        else FactSource.PATIENT_VERBATIM
    )
    evidence = str(payload.get(f"{key}{EVIDENCE_SUFFIX}", "") or "").strip()
    return Fact(
        key=key,
        value=value,
        source=source,
        turn=ctx.turn,
        verbatim=evidence or ctx.message,
        language=ctx.language,
        confidence=confidence,
        agent=agent,
    )
