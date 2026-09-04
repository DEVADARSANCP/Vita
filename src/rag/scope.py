"""
Scope classification — deciding whether VITA should be triaging this at all.

The first attempt at this was an absolute similarity threshold: if a patient's
description matched nothing in the corpus above some floor, call it out of
scope. Measured against the real corpus, that does not work. Gemini embeddings
put "my cat scratched my laptop screen" at 0.554 and a genuine self-harm
disclosure at 0.594, both against a corpus about neither. There is no cut-off
that separates them, because the floor is a property of the embedding space, not
of relevance.

Nearest-class does work. Hold labelled examples of what VITA covers and labelled
examples of what it does not, and ask which side a description lands nearer. The
question stops being "how relevant is this in the abstract", which embeddings
answer badly, and becomes "is this more like a chest pain or more like a stroke",
which is exactly what they answer well.

Two safeguards on top:

* **A margin.** When the two sides are close, the answer is not trusted. A
  description that is nearly as much like one as the other is escalated rather
  than classified.
* **This is a second net, not the first.** Stroke, self-harm and obstetric
  presentations are already caught by deterministic red-flag phrases, which need
  no model and no network. This layer catches what those phrases miss.

Nothing here assigns an urgency. A true verdict routes the case to a human; it
does not conclude anything clinical.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import DATA_DIR

logger = logging.getLogger(__name__)

EXEMPLAR_FILE = DATA_DIR / "knowledge" / "scope_exemplars.json"

#: How much closer the winning side must be before the verdict is trusted.
#: Below this the description sits between the two, and an uncertain scope
#: decision is escalated rather than guessed.
DECISION_MARGIN = 0.02


@dataclass
class Exemplar:
    """One labelled example phrasing."""

    text: str
    label: str
    covered: bool

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "label": self.label, "covered": self.covered}


@dataclass
class ScopeVerdict:
    """The outcome of a scope check, with the evidence behind it."""

    in_scope: bool
    confident: bool
    label: str = ""
    score: float = 0.0
    margin: float = 0.0
    nearest_covered: str = ""
    nearest_out: str = ""
    method: str = "exemplars"

    @property
    def refuses(self) -> bool:
        """Should VITA decline to triage this at all?

        Only a *confident* negative verdict refuses. An uncertain one must not,
        and conflating the two was a real defect: a patient writing "I banged my
        head, I feel alright, but I take warfarin" lands 0.001 from the boundary
        because the message is about both an injury and a heart medication, and
        VITA refused to triage a case whose facts had already been extracted
        correctly and which rule IN-03 covers exactly.

        Refusing and escalating are different actions. Uncertainty warrants the
        second, never the first.
        """
        return not self.in_scope and self.confident

    @property
    def needs_review(self) -> bool:
        """Should a human look at this, whatever the rules go on to decide?"""
        return not self.in_scope or not self.confident

    def explain(self) -> str:
        if not self.in_scope:
            return (
                f"The description most closely matches '{self.label}', which is "
                f"outside the five complaints VITA covers."
            )
        if not self.confident:
            return (
                f"The description sits between covered and uncovered presentations "
                f"(margin {self.margin:.3f}); VITA has not attempted to triage it."
            )
        return f"The description matches the covered complaint '{self.label}'."

    def as_dict(self) -> dict[str, Any]:
        return {
            "in_scope": self.in_scope,
            "confident": self.confident,
            "label": self.label,
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "nearest_covered": self.nearest_covered,
            "nearest_out_of_scope": self.nearest_out,
            "method": self.method,
            "explanation": self.explain(),
        }


def load_exemplars(path: Path | None = None) -> list[Exemplar]:
    """Load labelled exemplars in a stable order.

    Order matters for the same reason it does in the corpus: the embedding rows
    are aligned to this list by position.
    """
    path = path or EXEMPLAR_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("scope exemplars unavailable (%s); scope checking disabled", exc)
        return []

    exemplars: list[Exemplar] = []
    for label, phrases in sorted((raw.get("covered") or {}).items()):
        for text in phrases:
            exemplars.append(Exemplar(text=text, label=label, covered=True))
    for label, phrases in sorted((raw.get("out_of_scope") or {}).items()):
        for text in phrases:
            exemplars.append(Exemplar(text=text, label=label, covered=False))

    return exemplars


class ScopeClassifier:
    """Nearest-class scope check over labelled exemplars."""

    def __init__(self, matrix: Any = None, exemplars: list[Exemplar] | None = None) -> None:
        self.exemplars = exemplars if exemplars is not None else load_exemplars()
        self.matrix = matrix

    @property
    def ready(self) -> bool:
        return self.matrix is not None and len(self.exemplars) == getattr(
            self.matrix, "shape", [0]
        )[0]

    def classify(self, vector: list[float] | None) -> ScopeVerdict:
        """Which side of the boundary does this description fall on?"""
        if not self.ready or vector is None:
            # No embeddings means no opinion. Saying nothing is correct here:
            # the deterministic red flags still run, and a case with no scope
            # verdict is simply triaged normally.
            return ScopeVerdict(in_scope=True, confident=True, method="unavailable")

        import numpy as np

        q = np.asarray(vector, dtype="float32")
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return ScopeVerdict(in_scope=True, confident=True, method="unavailable")

        scores = self.matrix @ (q / norm)

        best_covered = -1.0
        best_out = -1.0
        covered_label = ""
        out_label = ""

        for exemplar, score in zip(self.exemplars, scores):
            value = float(score)
            if exemplar.covered:
                if value > best_covered:
                    best_covered, covered_label = value, exemplar.label
            elif value > best_out:
                best_out, out_label = value, exemplar.label

        in_scope = best_covered >= best_out
        margin = abs(best_covered - best_out)

        return ScopeVerdict(
            in_scope=in_scope,
            confident=margin >= DECISION_MARGIN,
            label=covered_label if in_scope else out_label,
            score=best_covered if in_scope else best_out,
            margin=margin,
            nearest_covered=covered_label,
            nearest_out=out_label,
        )
