"""
Retrieval — cosine similarity over a committed matrix, and a refusal signal.

The search itself is four lines: normalise the query vector, dot it against a
pre-normalised matrix, take the top k. There is no vector database because at
this corpus size there is nothing for one to do.

The interesting part is what retrieval is used *for*, which is narrower than it
usually is:

* **Citing policy.** When a case is routed, the governing clause is retrieved
  and cited. A legal-RAG pattern: find the clause, quote it, and where nothing
  covers the situation, say so rather than inventing a route.
* **Explaining a recommendation.** The guidance prose behind a matched rule,
  for a clinician who wants the reasoning in sentences.
* **Refusing.** This is the one that earns its place. A description that is
  nearer to examples of what VITA does not cover than to examples of what it
  does is evidence the case should go to a human - a refusal made on the
  strength of the system's own labelled data rather than by asking a model
  whether it feels confident. The mechanism lives in `scope.py`, and it is
  nearest-class rather than a similarity threshold, for reasons recorded there.

Retrieval is never used to select clinical rules. Those come from a
deterministic lookup by complaint. A similarity miss here costs a citation; a
similarity miss there would drop a HIGH rule and under-triage a patient with
nothing in the output to show it happened.

Degrades to keyword overlap when embeddings are unavailable. Worse search, not
a wrong decision - nothing downstream of this module makes a triage call.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import DATA_DIR
from ..llm.gemini import GeminiClient
from .corpus import Document, corpus_fingerprint, load_corpus
from .scope import ScopeClassifier, ScopeVerdict

logger = logging.getLogger(__name__)

INDEX_DIR = DATA_DIR / "index"

#: How many documents a query returns by default.
TOP_K = 3

_WORD = re.compile(r"[a-z]{3,}")

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "any", "are", "was", "have",
    "has", "not", "you", "your", "from", "but", "can", "all", "were", "been",
}


@dataclass
class Hit:
    """One retrieved document and how well it matched."""

    document: Document
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.document.id,
            "title": self.document.title,
            "kind": self.document.kind,
            "category": self.document.category,
            "text": self.document.text,
            "score": round(self.score, 4),
        }


class Retriever:
    """Searches the corpus. Never decides anything."""

    def __init__(self, llm: GeminiClient, index_dir: Path | None = None) -> None:
        self.llm = llm
        self.index_dir = index_dir or INDEX_DIR
        self.documents: list[Document] = load_corpus()
        self.matrix = None
        self.manifest: dict[str, Any] = {}
        self._query_cache: dict[str, list[float]] = {}
        self._load_index()
        self.scope = ScopeClassifier(self._load_exemplar_matrix())

    # -- index -----------------------------------------------------------

    def _load_index(self) -> None:
        """Load the committed vectors, and check they still match the corpus."""
        vectors_path = self.index_dir / "embeddings.npy"
        manifest_path = self.index_dir / "manifest.json"

        if not vectors_path.exists() or not manifest_path.exists():
            logger.warning(
                "no embedding index at %s - retrieval will fall back to keyword "
                "overlap. Run: python scripts/build_index.py",
                self.index_dir,
            )
            return

        try:
            import numpy as np

            self.matrix = np.load(vectors_path)
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a bad index must not stop startup
            logger.error("embedding index could not be loaded (%s); using keyword fallback", exc)
            self.matrix = None
            return

        if self.matrix.shape[0] != len(self.documents):
            logger.error(
                "index has %d vectors but the corpus has %d documents - "
                "rebuild with scripts/build_index.py; using keyword fallback",
                self.matrix.shape[0],
                len(self.documents),
            )
            self.matrix = None
            return

        expected = corpus_fingerprint(self.documents)
        if self.manifest.get("fingerprint") != expected:
            # Same document count, different content. The index would return the
            # right score attached to the wrong text, which is worse than no
            # index at all because it looks like it worked.
            logger.error(
                "index fingerprint %s does not match corpus %s - the corpus changed "
                "since the index was built; rebuild it. Using keyword fallback.",
                self.manifest.get("fingerprint"),
                expected,
            )
            self.matrix = None
            return

        logger.info(
            "retrieval index: %d documents, %d dimensions, built %s",
            self.matrix.shape[0],
            self.matrix.shape[1],
            self.manifest.get("built_at", "unknown"),
        )

    def _load_exemplar_matrix(self) -> Any:
        """Load the scope exemplar vectors, if they were built."""
        path = self.index_dir / "exemplars.npy"
        if not path.exists():
            logger.warning(
                "no scope exemplars at %s - scope checking disabled. "
                "Run: python scripts/build_index.py",
                path,
            )
            return None
        try:
            import numpy as np

            return np.load(path)
        except Exception as exc:  # noqa: BLE001
            logger.error("scope exemplars could not be loaded (%s)", exc)
            return None

    @property
    def ready(self) -> bool:
        return self.matrix is not None

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "documents": len(self.documents),
            "model": self.manifest.get("model", ""),
            "dimensions": self.manifest.get("dimensions", 0),
            "built_at": self.manifest.get("built_at", ""),
            "mode": "embeddings" if self.ready else "keyword_fallback",
            "scope_check": self.scope.ready,
            "scope_exemplars": len(self.scope.exemplars),
        }

    # -- search ----------------------------------------------------------

    def search(self, query: str, *, k: int = TOP_K, kind: str = "") -> list[Hit]:
        """Retrieve the documents most like the query."""
        query = (query or "").strip()
        if not query:
            return []

        hits = self._semantic(query) if self.ready else []
        if not hits:
            hits = self._keyword(query)

        if kind:
            hits = [h for h in hits if h.document.kind == kind]
        return hits[:k]

    def _semantic(self, query: str) -> list[Hit]:
        vector = self._embed_query(query)
        if vector is None:
            return []

        import numpy as np

        q = np.asarray(vector, dtype="float32")
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return []
        q = q / norm

        # The matrix was normalised at build time, so this dot product is
        # already cosine similarity.
        scores = self.matrix @ q
        order = np.argsort(-scores)
        return [Hit(document=self.documents[i], score=float(scores[i])) for i in order]

    def _embed_query(self, query: str) -> list[float] | None:
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached

        vectors = self.llm.embed([query], task="RETRIEVAL_QUERY")
        if not vectors:
            return None

        # Bounded: the same handful of complaints recur constantly, and an
        # unbounded cache on user-supplied text is a slow memory leak.
        if len(self._query_cache) < 256:
            self._query_cache[query] = vectors[0]
        return vectors[0]

    def _keyword(self, query: str) -> list[Hit]:
        """Word overlap, used when embeddings are unavailable.

        Scores are scaled into roughly the same range as cosine similarity so
        that RELEVANCE_FLOOR means something comparable in both modes. It is a
        cruder signal, and the out-of-scope check is correspondingly less
        confident - which is why that check escalates rather than concludes.
        """
        terms = self._terms(query)
        if not terms:
            return []

        hits: list[Hit] = []
        for document in self.documents:
            document_terms = self._terms(document.embedding_text)
            if not document_terms:
                continue
            overlap = len(terms & document_terms) / len(terms)
            if overlap > 0:
                hits.append(Hit(document=document, score=round(overlap, 4)))

        hits.sort(key=lambda h: -h.score)
        return hits

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}

    # -- the uses ---------------------------------------------------------

    def policy_for(self, query: str, *, k: int = 2) -> list[Hit]:
        """The governing policy clauses for a situation."""
        return self.search(query, k=k, kind="policy")

    def guidance_for(self, complaint: str, *, k: int = 2) -> list[Hit]:
        """Clinical guidance prose behind a complaint's rules."""
        hits = self.search(complaint.replace("_", " "), k=k * 3, kind="guidance")
        preferred = [h for h in hits if complaint in h.document.applies_to]
        return (preferred or hits)[:k]

    def check_scope(self, description: str) -> ScopeVerdict:
        """Is this description something VITA covers?

        Nearest-class against labelled exemplars, not a similarity threshold.
        The first version of this used a threshold and did not work: measured
        against this corpus, "my cat scratched my laptop screen" scored 0.554 and
        a self-harm disclosure 0.594, so no cut-off separated them. Asking which
        side a description falls nearer is a question embeddings answer well.

        The caller escalates on a negative verdict. Nothing here concludes
        anything clinical, and nothing here changes an urgency.
        """
        if not self.scope.ready:
            return ScopeVerdict(in_scope=True, confident=True, method="unavailable")
        return self.scope.classify(self._embed_query(description))
