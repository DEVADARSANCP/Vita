"""
The retrieval corpus — what VITA is allowed to ground an answer in.

Two kinds of document, kept distinct because they answer different questions and
carry different consequences when retrieval gets them wrong.

**Clinical guidance** explains why the triage rules look the way they do. It is
cited alongside a recommendation so a clinician can read the reasoning in prose
rather than as a list of matched conditions.

**Hospital policy** covers what varies between hospitals: after-hours routing,
who must be notified, what happens when a department is full, which
presentations are out of scope. This is the layer a legal-RAG pattern fits -
retrieve the governing clause, cite it, and where no clause covers the
situation, escalate rather than invent a route.

Neither is ever used to choose which clinical rules to evaluate. That selection
is a deterministic lookup by complaint, because a similarity miss that dropped a
HIGH rule would under-triage a patient silently. The principle, stated once:
**retrieval where a miss costs routing quality, determinism where a miss costs
patient safety.**

Documents are short enough to embed whole. Chunking would add a boundary problem
for no benefit at this size.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import DATA_DIR

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = DATA_DIR / "knowledge"


@dataclass
class Document:
    """One retrievable document."""

    id: str
    title: str
    text: str
    kind: str  # "policy" or "guidance"
    category: str = ""
    applies_to: list[str] = field(default_factory=list)

    @property
    def embedding_text(self) -> str:
        """What gets embedded.

        The title is included because it carries the terms a patient's own
        phrasing is most likely to echo - "chest pain", "blood thinners" - and
        omitting it measurably weakens matching on short queries.
        """
        return f"{self.title}\n\n{self.text}"

    def as_manifest_entry(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "category": self.category,
            "applies_to": self.applies_to,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.as_manifest_entry(), "text": self.text}


def load_corpus(directory: Path | None = None) -> list[Document]:
    """Load every document, in a stable order.

    Order matters: the embedding matrix is stored as rows aligned to this list,
    so a change in ordering would silently mis-attribute every result. Sorting
    by id makes the alignment reproducible rather than dependent on how a JSON
    file happened to be written.
    """
    directory = directory or KNOWLEDGE_DIR
    documents: list[Document] = []

    for filename, kind in (("policies.json", "policy"), ("guidance.json", "guidance")):
        path = directory / filename
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("could not read %s: %s", path.name, exc)
            continue

        for spec in raw.get("documents", []):
            documents.append(
                Document(
                    id=spec["id"],
                    title=spec["title"],
                    text=spec["text"],
                    kind=kind,
                    category=spec.get("category", spec.get("complaint", "")),
                    applies_to=list(spec.get("applies_to", []))
                    or ([spec["complaint"]] if spec.get("complaint") else []),
                )
            )

    documents.sort(key=lambda d: d.id)
    return documents


def corpus_fingerprint(documents: list[Document]) -> str:
    """A hash of the corpus, so a stale index can announce itself.

    An index built against a corpus that has since changed will return
    confidently wrong citations - the right similarity score attached to the
    wrong document. Comparing fingerprints at startup turns that into a warning
    rather than a mystery.
    """
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.id.encode("utf-8"))
        digest.update(document.embedding_text.encode("utf-8"))
    return digest.hexdigest()[:16]
