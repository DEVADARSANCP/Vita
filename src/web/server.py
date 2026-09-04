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
    name: str = Field(default="", max_length=120)
    language: str = Field(default="en", max_length=16)


class MessageRequest(BaseModel):
    text: str = Field(default="", max_length=4000)


class OverrideRequest(BaseModel):
    urgency: str = Field(max_length=16)
    reason: str = Field(default="", max_length=500)
    by: str = Field(default="clinician", max_length=120)


class ReviewRequest(BaseModel):
    by: str = Field(default="clinician", max_length=120)


class DecideRequestBody(BaseModel):
    approved: bool = True
    by: str = Field(default="clinician", max_length=120)
    note: str = Field(default="", max_length=500)
    room_id: str = Field(default="", max_length=32)


class AmbulanceRequestBody(BaseModel):
    pickup_location: str = Field(default="", max_length=300)
    confirmed_by: str = Field(default="", max_length=120)


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
        case = services.start_case(name=body.name, language=body.language)
        return JSONResponse(
            {
                "case_id": case.case_id,
                "patient_id": case.patient_id,
                "patient_name": case.patient_name,
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
            "mode": result.mode.value,
            "thinking": result.thinking,
            "tools_called": list(dict.fromkeys(result.tools_called)),
            "facts_recorded": sorted(set(result.facts_recorded)),
            "red_flags": result.red_flags,
            "converged": result.converged,
            "notes": result.notes,
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

    # -- approval queue ----------------------------------------------------

    @app.get("/api/requests")
    def requests_queue(status: str = "", case_id: str = "") -> JSONResponse:
        """What the planner has asked for and nobody has decided yet."""
        items = services.requests.list(status=status, case_id=case_id)
        return JSONResponse(
            {
                "counts": services.requests.counts(),
                "requests": [r.as_dict() for r in items],
            }
        )

    @app.post("/api/requests/{request_id}/decide")
    def decide_request(request_id: str, body: DecideRequestBody) -> JSONResponse:
        """Approve or reject. Approving is the only thing that makes it happen."""
        result = services.decide_request(
            request_id,
            approved=body.approved,
            by=body.by,
            note=body.note,
            room_id=body.room_id,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return JSONResponse(result)

    # -- admissions and rooms ----------------------------------------------

    @app.get("/api/rooms")
    def rooms(department: str = "") -> JSONResponse:
        occupied = services.requests.occupied_rooms()
        return JSONResponse(
            {
                "rooms": [
                    r.as_dict(occupied=r.room_id in occupied)
                    for r in services.hospital.rooms
                    if not department or r.department.lower() == department.lower()
                ],
                "admissions": services.requests.admissions(active_only=True),
            }
        )

    @app.get("/api/doctors")
    def doctors(department: str = "") -> JSONResponse:
        listing = services.hospital.doctors
        if department:
            listing = [d for d in listing if d.department.lower() == department.lower()]
        return JSONResponse({"doctors": [d.as_dict() for d in listing]})

    # -- emergency transport ---------------------------------------------

    @app.post("/api/cases/{case_id}/ambulance")
    def request_ambulance(case_id: str, body: AmbulanceRequestBody) -> JSONResponse:
        """Raise a transport request on explicit confirmation.

        Every precondition is enforced in the service, not here: the case must
        already be graded HIGH or CRITICAL by the rule engine, a pickup location
        must be given, and a person must be named as confirming. VITA offers;
        it does not request.
        """
        result = services.tools.call(
            "create_ambulance_request",
            {
                "case_id": case_id,
                "pickup_location": body.pickup_location,
                "confirmed_by": body.confirmed_by,
            },
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return JSONResponse(result)

    @app.get("/api/ambulance")
    def ambulance(case_id: str = "") -> JSONResponse:
        return JSONResponse(services.tools.call("get_ambulance_status", {"case_id": case_id}))

    # -- evaluation --------------------------------------------------------

    @app.get("/api/eval")
    def evaluate(conversation: bool = False, limit: int = 0) -> JSONResponse:
        """Run the evaluation suite and report what happened.

        The deterministic tier always runs: no model, no network, milliseconds,
        and it must pass at 100%. The conversational tier is opt-in because each
        turn is a model call and a free-tier key allows a handful per minute -
        exhausting a judge's quota to prove a point is a poor trade.

        The headline is under-triage count, not accuracy. Over-triage costs a
        clinician's time; under-triage sends home someone who should not have
        gone, and averaging the two into one percentage hides the only failure
        that hurts a patient.
        """
        from ..eval.runner import run

        report = run(services, include_conversation=conversation, limit=limit)
        return JSONResponse(report.as_dict())

    # -- tool surface ------------------------------------------------------

    @app.get("/api/tools")
    def tools() -> JSONResponse:
        """The tool surface, split by who may call what.

        Published because the split is a claim worth being able to check: the
        conversation model is advertised the retrieval tier only and has no tool
        that assigns an urgency.
        """
        return JSONResponse(
            {
                "transport": services.mcp.status(),
                "tools": services.mcp.tools() or services.tools.list_tools(),
            }
        )

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

    @app.get("/evaluation")
    def evaluation_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "evaluation.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
