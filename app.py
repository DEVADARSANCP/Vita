"""
VITA — AI-assisted patient intake and triage.

Entry point. `python app.py` starts everything — API and both interfaces — on
port 8000, with no build step and no second command.

Startup is deliberately defensive. Configuration problems are reported on the
console and surfaced through `/api/health`, but they never prevent the port from
opening: an application that exits because a key is missing is indistinguishable,
to whoever is trying to run it, from one that does not work at all.
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from src.config import APP_NAME, APP_VERSION, HOST, PORT, load_settings
from src.web.server import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vita")


def banner(settings) -> None:
    mode = settings.initial_mode()
    print()
    print(f"  {APP_NAME} {APP_VERSION} - patient intake and triage")
    print(f"  mode      {mode.value}")
    print(f"  listening http://localhost:{PORT}")

    if not settings.has_key:
        print(
            "\n  GEMINI_API_KEY is not set. The app is running, but language\n"
            "  understanding is unavailable: every case will be handled by the\n"
            "  deterministic path and escalated for human review.\n"
        )
    print()


def main() -> int:
    load_dotenv()
    settings = load_settings()
    banner(settings)

    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn is not installed — run: pip install -r requirements.txt")
        return 1

    uvicorn.run(create_app(settings), host=HOST, port=PORT, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
