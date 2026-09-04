"""
Build the retrieval index. Run once, commit the output, never run at startup.

    python scripts/build_index.py

The submission rules give the application 90 seconds to start answering, and
embedding a corpus over the network is exactly the kind of thing that fits
comfortably on a developer's machine and then does not fit on a judge's. So the
vectors are computed here, written to `data/index/`, and committed. Startup
loads two numpy arrays from disk and is done in milliseconds.

There is no vector database. At this size - a few dozen short documents - an
index costs more to build and load than a cosine similarity over a matrix costs
to run, and it is one more dependency that can fail on a machine we do not
control. `numpy.dot` on a normalised matrix is the whole search engine.

Two matrices are written. `embeddings.npy` holds the document corpus, for citing
policy and guidance. `exemplars.npy` holds labelled example phrasings, for
deciding whether a description is something VITA covers at all - a nearest-class
question rather than a similarity-threshold one, for reasons recorded in
src/rag/scope.py.

Re-run this whenever the corpus changes. The manifest records a hash of the
input, and the retriever warns at startup if the two have drifted apart, so a
stale index announces itself rather than quietly returning yesterday's answers.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.config import DATA_DIR, load_settings  # noqa: E402
from src.llm.gemini import GeminiClient  # noqa: E402
from src.rag.corpus import corpus_fingerprint, load_corpus  # noqa: E402
from src.rag.scope import load_exemplars  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("build_index")

INDEX_DIR = DATA_DIR / "index"

#: Documents per embedding request. Small enough to stay well inside free-tier
#: limits, large enough that the whole corpus takes a handful of calls.
BATCH_SIZE = 8

#: Seconds between batches. Free-tier keys are rate limited per minute, and this
#: script is not in a hurry - it runs once, offline, before anything is
#: committed.
BATCH_PAUSE = 2.0


def embed_all(client: GeminiClient, texts: list[str], label: str) -> list[list[float]] | None:
    """Embed a list of texts in batches, pausing between them."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        embedded = client.embed(batch, task="RETRIEVAL_DOCUMENT")
        if embedded is None or len(embedded) != len(batch):
            logger.error("%s: embedding failed at item %d - index not written", label, start)
            return None
        vectors.extend(embedded)
        logger.info("%s: embedded %d/%d", label, len(vectors), len(texts))
        if start + BATCH_SIZE < len(texts):
            time.sleep(BATCH_PAUSE)
    return vectors


def normalise(vectors: list[list[float]]) -> Any:
    """Unit-normalise once at build time, so search is a plain dot product."""
    import numpy as np

    matrix = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def main() -> int:
    load_dotenv(ROOT / ".env")
    settings = load_settings()

    if not settings.has_key:
        logger.error("GEMINI_API_KEY is not set - cannot build the index")
        return 1

    client = GeminiClient(settings)
    if not client.available:
        logger.error("Gemini client unavailable")
        return 1

    documents = load_corpus()
    exemplars = load_exemplars()
    logger.info("corpus: %d documents, %d scope exemplars", len(documents), len(exemplars))

    vectors = embed_all(client, [d.embedding_text for d in documents], "corpus")
    if vectors is None:
        return 1

    exemplar_vectors = embed_all(client, [e.text for e in exemplars], "exemplars")
    if exemplar_vectors is None:
        return 1

    import numpy as np

    matrix = normalise(vectors)
    exemplar_matrix = normalise(exemplar_vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", matrix)
    np.save(INDEX_DIR / "exemplars.npy", exemplar_matrix)

    manifest = {
        "model": settings.embedding_model,
        "dimensions": int(matrix.shape[1]),
        "count": int(matrix.shape[0]),
        "fingerprint": corpus_fingerprint(documents),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "documents": [d.as_manifest_entry() for d in documents],
        "exemplar_count": int(exemplar_matrix.shape[0]),
        "exemplars": [e.as_dict() for e in exemplars],
    }
    (INDEX_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        "wrote embeddings.npy (%d x %d) and exemplars.npy (%d x %d)",
        matrix.shape[0],
        matrix.shape[1],
        exemplar_matrix.shape[0],
        exemplar_matrix.shape[1],
    )
    logger.info("commit data/index/ - the judge's clone must not rebuild this")
    return 0


if __name__ == "__main__":
    sys.exit(main())
