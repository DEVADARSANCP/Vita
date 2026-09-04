"""
Contradiction detection — noticing when the patient has said two things.

A patient says "no, my breathing is fine" on turn two, and on turn six says "I
get short of breath just walking to the door". Both are real answers. The system
does not get to decide which one the patient meant.

The tempting behaviour is to overwrite: the later answer wins, the fact becomes
true, the case carries on. That quietly discards evidence a clinician would want
to see, and it is wrong in both directions - the patient may have misunderstood
the first question, or the second, or their condition may have changed in the
four minutes between them. All three possibilities matter and none of them is
VITA's to resolve.

So both facts are kept, the conflict is recorded, and the case goes to a human.
This runs over the fact store rather than over the patient's message, which is
why it lives in core alongside the rule engine rather than among the agents.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from .schema import Contradiction, Fact, Tri

logger = logging.getLogger(__name__)

#: How much a numeric answer may move before it counts as a contradiction rather
#: than a patient refining an estimate. "About three hours" then "maybe four" is
#: the same story; three hours then three days is not.
_NUMERIC_TOLERANCE = 0.35

#: Facts that legitimately change during an intake conversation and must not be
#: reported as contradictions.
_VOLATILE = {"complaint", "language", "severity"}


def detect(existing: Fact | None, incoming: Fact) -> Contradiction | None:
    """Compare a new fact against the one it would replace.

    Returns a contradiction only when the two are genuinely incompatible. A
    fact merely being restated, refined, or upgraded from unknown is not a
    conflict.
    """
    if existing is None or incoming.key in _VOLATILE:
        return None
    if not existing.is_known or not incoming.is_known:
        return None
    if existing.turn == incoming.turn:
        return None

    if _conflicts(existing.value, incoming.value):
        logger.info(
            "contradiction on %s: turn %d said %s, turn %d said %s",
            incoming.key,
            existing.turn,
            existing.value,
            incoming.turn,
            incoming.value,
        )
        return Contradiction(key=incoming.key, earlier=existing, later=incoming)

    return None


def _conflicts(earlier: Any, later: Any) -> bool:
    earlier_tri, later_tri = Tri.coerce(earlier), Tri.coerce(later)
    if earlier_tri is not Tri.UNKNOWN and later_tri is not Tri.UNKNOWN:
        return earlier_tri is not later_tri

    earlier_num, later_num = _number(earlier), _number(later)
    if earlier_num is not None and later_num is not None:
        if earlier_num == 0 and later_num == 0:
            return False
        scale = max(abs(earlier_num), abs(later_num), 1.0)
        return abs(earlier_num - later_num) / scale > _NUMERIC_TOLERANCE

    return str(earlier).strip().lower() != str(later).strip().lower()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or isinstance(value, Tri):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def scan(known: dict[str, Fact], incoming: Iterable[Fact]) -> list[Contradiction]:
    """Check a batch of new facts against what is already established."""
    found: list[Contradiction] = []
    for fact in incoming:
        conflict = detect(known.get(fact.key), fact)
        if conflict is not None:
            found.append(conflict)
    return found
