"""
The HTTP surface — one FastAPI application serving both interfaces.

The submission rules allow exactly one command, so there is no second server and
no build step. The patient interface and the hospital dashboard are static files
served from `src/web/static`; everything else is JSON under `/api`.

The handlers are deliberately thin. Sequencing lives in the orchestrator and the
service container, so the behaviour that matters can be exercised without going
through HTTP, and an endpoint is never the place a clinical decision gets made.

Two conventions worth knowing:

* **Failures are values.** A model that will not answer produces a degraded turn
  and a 200 describing it, not a 500. The patient is mid-conversation; an error
  page tells them nothing and loses the case.
* **Citations resolve.** Every rule id in a response can be fetched from
  `/api/rules/{rule_id}` and read in full. A citation nobody can follow is
  decoration.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import APP_NAME, APP_VERSION, Settings, load_settings
from ..services.container import VitaServices

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class StartCaseRequest(BaseModel):
    language: str = Field(default="en", max_length=16)


class MessageRequest(BaseModel):
    text: str = Field(default="", max_length=4000)


class OverrideRequest(BaseModel):
    urgency: str = Field(max_length=16)
    reason: str = Field(default="", max_length=500)
    by: str = Field(default="clinician", max_length=120)


class ReviewRequest(BaseModel):
    by: str = Field(default="clinician", max_length=120)


def create_app(settings: Settings | None = None, services: VitaServices | None = None) -> FastAPI:
    settings = settings or load_settings()
    services = services or VitaServices(settings)

    app = FastAPI(
        title=f"{APP_NAME} - Patient Intake & Triage",
        version=APP_VERSION,
        docs_url="/api/docs",
    )
    app.state.settings = settings
    app.state.services = services

    # -- system ----------------------------------------------------------

    @app.get("/api/health")
    def health() -> JSONResponse:
        """Liveness plus an honest account of what the system can currently do."""
        return JSONResponse(
            {
                "status": "ok",
                "app": APP_NAME,
                "version": APP_VERSION,
                "mode": services.mode.value,
                "gemini_key_present": settings.has_key,
                "notify_enabled": settings.notify_enabled,
            }
        )

    @app.get("/api/status")
    def status() -> JSONResponse:
        return JSONResponse(services.status())

    @app.get("/api/hospital")
    def hospital() -> JSONResponse:
        return JSONResponse(services.hospital.as_dict())

    # -- knowledge -------------------------------------------------------

    @app.get("/api/rules")
    def rules() -> JSONResponse:
        return JSONResponse(
            {
                "version": services.kb.version,
                "disclaimer": services.kb.disclaimer,
                "rules": [r.as_dict() for r in services.kb.rules],
                "red_flags": [f.as_dict() for f in services.kb.red_flags],
            }
        )

    @app.get("/api/rules/{rule_id}")
    def rule(rule_id: str) -> JSONResponse:
        """Resolve a cited rule. Every citation in the system reaches this."""
        found = services.kb.rule(rule_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no rule {rule_id}")
        return JSONResponse(found.as_dict())

    # -- intake ----------------------------------------------------------

    @app.post("/api/cases")
    def start_case(body: StartCaseRequest) -> JSONResponse:
        case = services.start_case(language=body.language)
        return JSONResponse(
            {
                "case_id": case.case_id,
                "language": case.language,
                "mode": services.mode.value,
                "opening": services.phraser.say("opening", case.language),
            }
        )

    @app.post("/api/cases/{case_id}/messages")
    def send_message(case_id: str, body: MessageRequest) -> JSONResponse:
        result = services.message(case_id, body.text)
        if result is None:
            raise HTTPException(status_code=404, detail=f"no case {case_id}")

        payload = {
            "case_id": case_id,
            "reply": result.reply,
            "finished": result.finished,
            "asked_about": result.asked_about,
            "driven_by_rule": result.driven_by_rule,
            "mode": result.mode.value,
            "red_flags": result.red_flags,
            "notes": result.notes,
            "facts_added": sorted(set(result.facts_added)),
        }
        if result.finished:
            payload["note"] = services.note(case_id)
        return JSONResponse(payload)

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str) -> JSONResponse:
        case = services.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail=f"no case {case_id}")
        return JSONResponse(case.as_dict(full=True))

    @app.get("/api/cases/{case_id}/note")
    def get_note(case_id: str) -> JSONResponse:
        note = services.note(case_id)
        if note is None:
            raise HTTPException(status_code=404, detail=f"no case {case_id}")
        return JSONResponse(note)

    # -- hospital dashboard ----------------------------------------------

    @app.get("/api/queue")
    def queue(status: str = "") -> JSONResponse:
        """The triage queue, ordered by urgency then arrival.

        Capacity is reported alongside each case but never reorders the list.
        Letting a full department push a sick patient down the queue would be a
        different class of mistake from a mistuned rule.
        """
        rows = services.cases.queue(status=status)
        for row in rows:
            row["routing"] = services.hospital.routing_note(row.get("department", ""))
        return JSONResponse({"counts": services.cases.counts(), "cases": rows})

    @app.post("/api/cases/{case_id}/override")
    def override(case_id: str, body: OverrideRequest) -> JSONResponse:
        case = services.override(case_id, body.urgency, body.reason, body.by)
        if case is None:
            raise HTTPException(status_code=400, detail="unknown case or urgency")
        return JSONResponse(case.as_dict(full=False))

    @app.post("/api/cases/{case_id}/review")
    def review(case_id: str, body: ReviewRequest) -> JSONResponse:
        case = services.mark_reviewed(case_id, body.by)
        if case is None:
            raise HTTPException(status_code=404, detail=f"no case {case_id}")
        return JSONResponse(case.as_dict(full=False))

    @app.get("/api/notifications")
    def notifications() -> JSONResponse:
        return JSONResponse(
            {
                "dry_run": not settings.notify_enabled,
                "messages": [n.as_dict() for n in services.notifier.outbox],
            }
        )

    # -- interfaces ------------------------------------------------------

    @app.get("/")
    def patient_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/dashboard")
    def dashboard_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "dashboard.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
