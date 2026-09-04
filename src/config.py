"""
Runtime configuration and the system's own account of what it can currently do.

The single most important property here: **the application starts whether or not
a Gemini key is present.** A judge runs `python app.py` on a clean machine; if a
missing or rejected key takes the process down, nothing is served on port 8000
and there is nothing to evaluate. So the key is read, its absence is recorded,
and the app comes up anyway in a reduced mode that says so.

Three modes, and the system reports which one it is in rather than failing
quietly:

* ``FULL``     — Gemini reachable. Language understanding and extraction active.
* ``DEGRADED`` — the model failed, timed out, or returned unusable output.
  Falls back to deterministic keyword extraction and the scripted follow-up
  questions. **Every case triaged in this mode is forced to human review.**
* ``OFFLINE``  — no key configured at all. The red-flag fast path and the rule
  engine still work, because neither has ever depended on the model.

The rule engine is unaffected in all three. That is the point of keeping the
triage decision out of the model: losing the LLM costs extraction quality, not
patient safety.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

APP_NAME = "VITA"
APP_VERSION = "0.1.0"

#: Fixed by the submission rules: the app must answer on this port.
PORT = 8000
HOST = "0.0.0.0"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNTIME_DIR = ROOT / "runtime"


class SystemMode(str, Enum):
    """How much of the system is currently available."""

    FULL = "FULL"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


@dataclass
class Settings:
    """Configuration resolved once at startup.

    Every field has a working default. Nothing here raises, because a
    configuration error must not be the reason the port stays closed.
    """

    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-lite-latest"
    embedding_model: str = "gemini-embedding-001"

    #: Outbound notification is dry-run unless this is explicitly turned on.
    #: On a judge's machine it never will be, so the full message is composed,
    #: recorded and shown in the dashboard without anything leaving the process.
    notify_enabled: bool = False

    @property
    def has_key(self) -> bool:
        return bool(self.gemini_api_key.strip())

    def initial_mode(self) -> SystemMode:
        """The mode implied by configuration alone, before any model call."""
        return SystemMode.FULL if self.has_key else SystemMode.OFFLINE


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


def load_settings() -> Settings:
    """Read configuration from the environment. Never raises."""
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("VITA_GEMINI_MODEL", "gemini-flash-lite-latest"),
        embedding_model=os.getenv("VITA_EMBEDDING_MODEL", "gemini-embedding-001"),
        notify_enabled=_flag("VITA_NOTIFY_ENABLED"),
    )
