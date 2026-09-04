"""
Patient memory, backed by MemPalace.

What VITA remembers between visits: who a patient is, what they came in with
last time, whether it resolved, what they take, what a clinician concluded. The
planner reads it at the start of every intake, so a patient who has been in
three times this fortnight is recognised as such rather than treated as a
stranger.

Two configuration choices make this work inside the submission rules.

**Backend is `sqlite_exact`.** MemPalace defaults to Chroma, which is fine, but
`sqlite_exact` is an explicit-vector backend needing only sqlite3 and numpy -
no ONNX runtime, no vector service, one file on disk.

**Embeddings come from Gemini.** MemPalace's built-in embedders download ONNX
weights from HuggingFace on first use, which would be a second network
dependency and a 300 MB download on a machine we do not control. Injecting
`gemini-embedding-001` - the model the rules require anyway - means the only
thing that ever leaves the process is a Gemini call.

Memory is written sparingly and deliberately. Every stored line costs an
embedding call, and a store full of conversational noise retrieves worse than a
store holding a dozen clinically meaningful sentences. So VITA records visit
outcomes, durable patient facts and clinician conclusions - not transcripts.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..config import RUNTIME_DIR
from ..llm.gemini import GeminiClient

logger = logging.getLogger(__name__)

PALACE_DIR = RUNTIME_DIR / "palace"

#: What a stored memory is about. Kept small so retrieval can be filtered by
#: kind when the planner wants one sort of thing specifically.
KIND_VISIT = "visit"
KIND_FACT = "fact"
KIND_MEDICATION = "medication"
KIND_OUTCOME = "outcome"
KIND_CLINICIAN = "clinician_note"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class GeminiEmbeddingFunction:
    """A Chroma-shaped embedding function that calls Gemini.

    MemPalace asks its embedding layer for one thing - a callable taking
    `input` and returning vectors - so satisfying that interface is enough to
    replace the local ONNX embedder entirely. `name()` is reported to the
    backend and persisted alongside the collection, so it must stay stable:
    changing it later would make MemPalace refuse to read its own store.
    """

    def __init__(self, llm: GeminiClient) -> None:
        self.llm = llm

    @staticmethod
    def name() -> str:
        return "gemini_embedding_001"

    def __call__(self, input: Any) -> list[list[float]]:  # noqa: A002 - the library's parameter name
        texts = [input] if isinstance(input, str) else list(input)
        vectors = self.llm.embed(texts, task="RETRIEVAL_DOCUMENT")
        if vectors is None:
            # Raising rather than returning zeros: a zero vector would be stored
            # and would then match everything equally, which is worse than the
            # write failing where the caller can see it.
            raise RuntimeError("Gemini embedding unavailable - memory write cannot proceed")
        return vectors


@dataclass
class Memory:
    """One remembered thing about one patient."""

    memory_id: str
    patient_id: str
    kind: str
    text: str
    at: str = field(default_factory=_now)
    case_id: str = ""
    score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "patient_id": self.patient_id,
            "kind": self.kind,
            "text": self.text,
            "at": self.at,
            "case_id": self.case_id,
            "score": round(self.score, 4),
        }


class PatientMemory:
    """Reads and writes what VITA remembers about patients.

    Constructed once at startup and never allowed to fail: a memory layer that
    will not open is a lost feature, not a reason for the application to refuse
    to triage anybody. `available` reports the truth and every method degrades
    to doing nothing.
    """

    def __init__(self, llm: GeminiClient, path: Any = None) -> None:
        self.llm = llm
        self.path = str(path or PALACE_DIR)
        self._collection: Any = None
        self._unavailable_reason = ""
        self._open()

    def _open(self) -> None:
        if not self.llm.available:
            self._unavailable_reason = "Gemini unavailable, so memory cannot be embedded"
            logger.warning("patient memory disabled: %s", self._unavailable_reason)
            return

        try:
            os.makedirs(self.path, exist_ok=True)
            # Selected by environment because that is the documented way to
            # choose a MemPalace backend, and it keeps the choice visible to
            # anyone inspecting the process rather than buried in a call.
            os.environ.setdefault("MEMPALACE_BACKEND", "sqlite_exact")

            import mempalace.embedding as mp_embedding

            embedder = GeminiEmbeddingFunction(self.llm)
            mp_embedding.get_embedding_function = lambda *args, **kwargs: embedder

            import mempalace.palace as palace

            self._collection = palace.get_collection(self.path, create=True)
            logger.info(
                "patient memory ready: backend=%s path=%s embedder=%s",
                palace.resolve_backend_name(self.path),
                self.path,
                embedder.name(),
            )
        except ImportError as exc:
            self._unavailable_reason = f"mempalace is not installed: {exc}"
            logger.error(self._unavailable_reason)
        except Exception as exc:  # noqa: BLE001 - startup must not fail here
            self._unavailable_reason = f"could not open patient memory: {exc}"
            logger.error(self._unavailable_reason)

    @property
    def available(self) -> bool:
        return self._collection is not None

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "backend": os.environ.get("MEMPALACE_BACKEND", "sqlite_exact"),
            "embedder": GeminiEmbeddingFunction.name(),
            "path": self.path,
            "reason": self._unavailable_reason,
            "stored": self.count(),
        }

    def count(self) -> int:
        if not self.available:
            return 0
        try:
            return int(self._collection.count())
        except Exception:  # noqa: BLE001
            return 0

    # -- writing ---------------------------------------------------------

    def remember(
        self,
        *,
        patient_id: str,
        kind: str,
        text: str,
        case_id: str = "",
    ) -> Memory | None:
        """Store one memory. Returns None if memory is unavailable.

        Deliberately one sentence at a time, and deliberately not called for
        every conversational turn. What earns a place here is what a clinician
        would want surfaced at the patient's next visit.
        """
        if not self.available or not patient_id or not text.strip():
            return None

        memory_id = f"{patient_id}:{kind}:{_now()}:{abs(hash(text)) % 10000:04d}"
        record = Memory(
            memory_id=memory_id,
            patient_id=patient_id,
            kind=kind,
            text=text.strip(),
            case_id=case_id,
        )

        try:
            self._collection.add(
                ids=[record.memory_id],
                documents=[record.text],
                metadatas=[
                    {
                        "patient_id": patient_id,
                        "kind": kind,
                        "case_id": case_id,
                        "at": record.at,
                    }
                ],
            )
            logger.info("remembered [%s] for %s: %s", kind, patient_id, text[:60])
            return record
        except Exception as exc:  # noqa: BLE001 - a failed write must not end an intake
            logger.error("could not store memory for %s: %s", patient_id, exc)
            return None

    # -- reading ---------------------------------------------------------

    def recall(
        self,
        *,
        patient_id: str,
        query: str = "",
        kind: str = "",
        limit: int = 5,
    ) -> list[Memory]:
        """Retrieve what is remembered about a patient.

        With a query, this is semantic search. Without one, it returns the
        patient's memories generally - which is what the planner wants at the
        start of an intake, before it knows what to look for.
        """
        if not self.available or not patient_id:
            return []

        where: dict[str, Any] = {"patient_id": patient_id}
        if kind:
            where["kind"] = kind

        try:
            result = self._collection.query(
                query_texts=[query or "previous visits, conditions and medications"],
                n_results=max(1, limit),
                where=where,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("memory recall failed for %s: %s", patient_id, exc)
            return []

        return self._to_memories(result)

    def _to_memories(self, result: dict[str, Any]) -> list[Memory]:
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0] or [0.0] * len(documents)

        memories: list[Memory] = []
        for index, text in enumerate(documents):
            meta = metadatas[index] if index < len(metadatas) else {}
            memories.append(
                Memory(
                    memory_id=ids[index] if index < len(ids) else "",
                    patient_id=str(meta.get("patient_id", "")),
                    kind=str(meta.get("kind", "")),
                    text=text,
                    at=str(meta.get("at", "")),
                    case_id=str(meta.get("case_id", "")),
                    # Distance is a cosine distance; report it as similarity so
                    # a bigger number means a better match everywhere in VITA.
                    score=1.0 - float(distances[index]) if index < len(distances) else 0.0,
                )
            )
        return memories
