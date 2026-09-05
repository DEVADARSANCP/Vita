"""
The Gemini client — the only thing in VITA that touches the network.

Everything the model does is language work: reading what a patient wrote,
turning it into structured facts, and phrasing a question back. It never
decides an urgency, so every failure mode here costs comprehension, not safety.
That is what makes the defensive posture below affordable.

Three rules this module keeps.

**It never raises at the caller.** A failed call returns ``None`` and flips the
system into DEGRADED. Callers handle "no answer", which they must anyway,
rather than an exception that could take a request down mid-triage.

**It never returns unvalidated output.** Responses are requested as JSON against
an explicit schema and parsed before being handed back. Text that does not parse
is a failure, not a value, because a half-understood answer is more dangerous
here than no answer at all.

**It reports its own state.** Failures are counted and the mode is published, so
the interface can say "language understanding is unavailable" rather than
quietly getting worse.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings, SystemMode

logger = logging.getLogger(__name__)

#: Per-call ceiling. A single request gets 60 seconds end to end, and a triage
#: turn may make more than one call, so no individual call may approach it.
CALL_TIMEOUT_SECONDS = 20.0

#: One retry, then degrade. A second failure is a signal, not bad luck, and
#: spending the request budget on a third attempt helps nobody.
MAX_ATTEMPTS = 2

#: Consecutive failures before the system stops describing itself as FULL.
FAILURES_BEFORE_DEGRADED = 2

#: Tried in order if the configured model is unavailable to the running key.
MODEL_FALLBACKS = ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-2.5-flash"]

#: Longest we will wait out a rate limit before giving up on a call. A single
#: request has 60 seconds, and the call itself needs some of them.
MAX_RATE_LIMIT_WAIT = 22.0

#: Rate limiting is not failure. On a free-tier key the quota is a handful of
#: requests per minute, and an intake conversation is one call per patient
#: message - so a judge clicking through a demo will meet it. The server says
#: how long to wait; waiting and retrying keeps the conversation intact, where
#: degrading immediately would drop the patient into keyword extraction over a
#: delay measured in seconds.
_RATE_LIMIT_TOKENS = ("429", "resource_exhausted", "rate limit", "quota")


def _rate_limit_delay(exc: Exception) -> float | None:
    """How long to wait for a rate limit, or None if this is not one.

    Gemini returns the delay it wants in the error body ("retryDelay": "17s").
    Honouring it is better than a fixed backoff: too short and the retry is
    wasted, too long and the request budget is gone.
    """
    message = str(exc).lower()
    if not any(token in message for token in _RATE_LIMIT_TOKENS):
        return None

    match = re.search(r"'retrydelay':\s*'(\d+(?:\.\d+)?)s'", message)
    if match:
        return min(float(match.group(1)) + 1.0, MAX_RATE_LIMIT_WAIT)
    return min(10.0, MAX_RATE_LIMIT_WAIT)


@dataclass
class LLMResult:
    """The outcome of one model call, including how it failed if it did."""

    ok: bool
    data: Any = None
    text: str = ""
    error: str = ""
    attempts: int = 0
    elapsed_ms: int = 0

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class _Health:
    """Rolling account of whether the model is answering."""

    consecutive_failures: int = 0
    total_calls: int = 0
    total_failures: int = 0
    last_error: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, ok: bool, error: str = "") -> None:
        with self.lock:
            self.total_calls += 1
            if ok:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
                self.total_failures += 1
                self.last_error = error


class GeminiClient:
    """Structured generation and embeddings, with failure treated as expected.

    Construction never fails and never blocks. A missing key is a state the
    object reports, not an error it raises, because the application has to come
    up on port 8000 either way.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.gemini_model
        self.embedding_model = settings.embedding_model
        self._health = _Health()
        self._client: Any = None
        self._types: Any = None
        self._unavailable_reason = ""

        if not settings.has_key:
            self._unavailable_reason = "GEMINI_API_KEY is not set"
            logger.warning("Gemini client created without a key - running OFFLINE")
            return

        try:
            from google import genai
            from google.genai import types

            self._client = genai.Client(api_key=settings.gemini_api_key)
            self._types = types
            logger.info("Gemini client ready (model=%s)", self.model)
        except ImportError as exc:
            self._unavailable_reason = f"google-genai is not installed: {exc}"
            logger.error(self._unavailable_reason)
        except Exception as exc:  # noqa: BLE001 - startup must not fail here
            self._unavailable_reason = f"could not initialise Gemini client: {exc}"
            logger.error(self._unavailable_reason)

    # -- state -----------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def mode(self) -> SystemMode:
        """What the system can currently do, as one value the UI can render."""
        if not self.available:
            return SystemMode.OFFLINE
        if self._health.consecutive_failures >= FAILURES_BEFORE_DEGRADED:
            return SystemMode.DEGRADED
        return SystemMode.FULL

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "available": self.available,
            "reason": self._unavailable_reason,
            "calls": self._health.total_calls,
            "failures": self._health.total_failures,
            "consecutive_failures": self._health.consecutive_failures,
            "last_error": self._health.last_error,
        }

    # -- generation ------------------------------------------------------

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system_instruction: str = "",
        temperature: float = 0.0,
        media: tuple[bytes, str] | None = None,
    ) -> LLMResult:
        """Ask for JSON matching `schema`, and return it only if it parses.

        `media` attaches an image or an audio clip to the same request. That is
        what lets a spoken turn cost one round trip rather than two: the model
        hears the patient and answers in a single call, instead of transcribing
        first and being asked again with the text.

        Temperature defaults to zero. Extraction is not a creative task, and two
        different readings of the same sentence would make the triage note
        irreproducible.
        """
        if not self.available:
            return LLMResult(ok=False, error=self._unavailable_reason or "model unavailable")

        started = time.monotonic()
        last_error = ""

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                config = self._types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature,
                    system_instruction=system_instruction or None,
                    http_options=self._types.HttpOptions(
                        timeout=int(CALL_TIMEOUT_SECONDS * 1000)
                    ),
                )
                contents: Any = prompt
                if media is not None:
                    payload, mime_type = media
                    contents = [
                        self._types.Part.from_bytes(data=payload, mime_type=mime_type),
                        prompt,
                    ]

                response = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                text = (response.text or "").strip()
                if not text:
                    raise ValueError("model returned an empty response")

                data = json.loads(text)
                self._health.record(True)
                return LLMResult(
                    ok=True,
                    data=data,
                    text=text,
                    attempts=attempt,
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                )

            except json.JSONDecodeError as exc:
                # Unparseable output is a failure, never a partial value. Acting
                # on half-understood extraction is worse than admitting we did
                # not understand.
                last_error = f"response was not valid JSON: {exc}"
                logger.warning("Gemini attempt %d: %s", attempt, last_error)

            except Exception as exc:  # noqa: BLE001 - any failure degrades
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Gemini attempt %d failed: %s", attempt, last_error)

                wait = _rate_limit_delay(exc)
                if wait is not None and attempt < MAX_ATTEMPTS:
                    logger.info("rate limited; waiting %.1fs before retrying", wait)
                    time.sleep(wait)
                    continue

                if attempt == 1 and self._try_fallback_model(exc):
                    continue

        self._health.record(False, last_error)
        return LLMResult(
            ok=False,
            error=last_error,
            attempts=MAX_ATTEMPTS,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    def _try_fallback_model(self, exc: Exception) -> bool:
        """Move to the next known model id if the current one was rejected.

        A judge supplies their own key, and key entitlements differ. Being wrong
        about a model name should cost one retry, not the whole feature.
        """
        message = str(exc).lower()
        if not any(token in message for token in ("not found", "404", "not supported")):
            return False
        remaining = [m for m in MODEL_FALLBACKS if m != self.model]
        if not remaining:
            return False
        logger.warning("model %s was rejected; trying %s", self.model, remaining[0])
        self.model = remaining[0]
        return True

    def read_media_json(
        self,
        media: bytes,
        mime_type: str,
        prompt: str,
        schema: dict[str, Any],
        *,
        system_instruction: str = "",
    ) -> LLMResult:
        """Read an image or an audio clip and return structured JSON about it.

        Used for medication photographs. Gemini is multimodal, so a picture of a
        blister pack needs no OCR engine, no model weights and no second network
        dependency - which is the difference between this working on a judge's
        clean machine and not.

        What comes back is only ever the text printed on the packet. Deciding
        what a drug is *for* stays a table lookup, because a model that decides
        for itself whether something is a blood thinner is a model that can be
        wrong about warfarin.
        """
        if not self.available:
            return LLMResult(ok=False, error=self._unavailable_reason or "model unavailable")

        started = time.monotonic()
        try:
            config = self._types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0,
                system_instruction=system_instruction or None,
                http_options=self._types.HttpOptions(timeout=int(CALL_TIMEOUT_SECONDS * 1000)),
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=[
                    self._types.Part.from_bytes(data=media, mime_type=mime_type),
                    prompt,
                ],
                config=config,
            )
            data = json.loads((response.text or "").strip())
            self._health.record(True)
            return LLMResult(ok=True, data=data, attempts=1,
                             elapsed_ms=int((time.monotonic() - started) * 1000))
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("media read failed: %s", error)
            self._health.record(False, error)
            return LLMResult(ok=False, error=error,
                             elapsed_ms=int((time.monotonic() - started) * 1000))

    #: The old name, from when this only took photographs.
    read_image_json = read_media_json

    # -- embeddings ------------------------------------------------------

    def embed(self, texts: list[str], *, task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]] | None:
        """Embed a batch of texts. Returns None if the call fails.

        Used to build the retrieval index offline and to embed one query at a
        time at request time. Retrieval degrades to keyword matching without it,
        which is a worse search, not a wrong triage decision.
        """
        if not self.available or not texts:
            return None

        try:
            config = self._types.EmbedContentConfig(task_type=task)
            response = self._client.models.embed_content(
                model=self.embedding_model, contents=texts, config=config
            )
            vectors = [list(e.values) for e in response.embeddings]
            self._health.record(True)
            return vectors
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("embedding call failed: %s", error)
            self._health.record(False, error)
            return None
