"""
The tool surface — everything the planner can reach, as flat MCP tools.

This is a plain tool layer, not a set of agents. There is no per-capability
wrapper class, no registry of things that decide for themselves when to run:
each tool is a named function with a schema, the planner calls it by name over
MCP, and the result comes back as JSON. That is the whole idea of the protocol
and it is deliberately different from ACPIA, where agents were auto-wrapped into
tools and the planner chose between capabilities that each held their own logic.

The tools fall into three groups, and the grouping is the safety model.

**Read tools** return what the system knows: the current triage state, what the
rules are still waiting on, what VITA remembers about this patient, which
clinicians are on call, which rooms are free. The planner calls these as freely
as it likes. Reading the rule engine's output is not the same as deciding it -
the planner may know the urgency is HIGH, and may not make it HIGH.

**`record_facts`** is the only tool that changes clinical state, and it is
tightly bounded: fact keys must already exist in the knowledge base, values are
coerced to the three-valued type, and anything unrecognised is rejected rather
than stored. The planner cannot invent a fact key, and it cannot write an
urgency.

**Request tools** do not act. They create an entry in the approval queue with
the planner's reasoning attached, and a human on the dashboard decides. This is
what lets the planner be trusted with judgement: its conclusions are proposals,
and a bad one costs somebody a two-second rejection rather than an ambulance.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .core.note import build_note, render_text
from .core.requests import Request, RequestKind
from .core.rules import decide, evaluate_all, next_unknown
from .core.schema import Complaint, Fact, FactSource, MatchState, Tri, Urgency

logger = logging.getLogger(__name__)


def _string(description: str, *, required: bool = False) -> dict[str, Any]:
    spec: dict[str, Any] = {"type": "string", "description": description}
    if required:
        spec["required"] = True
    return spec


class ToolLayer:
    """Dispatches named tool calls against a running VITA."""

    def __init__(self, services: Any) -> None:
        self.services = services
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._specs: list[dict[str, Any]] = []
        self._register_all()

    # -- registry --------------------------------------------------------

    def _add(self, name: str, description: str, parameters: dict[str, Any], handler: Callable[..., Any]) -> None:
        self._handlers[name] = handler
        self._specs.append(
            {
                "name": name,
                "description": description,
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        key: {k: v for k, v in spec.items() if k != "required"}
                        for key, spec in parameters.items()
                    },
                    "required": [k for k, s in parameters.items() if s.get("required")],
                },
            }
        )

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self._specs)

    def names(self) -> list[str]:
        return [s["name"] for s in self._specs]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            return {"error": f"no tool named {name!r}", "available": sorted(self._handlers)}
        try:
            return handler(**(arguments or {}))
        except TypeError as exc:
            return {"error": f"bad arguments for {name}: {exc}"}
        except Exception as exc:  # noqa: BLE001 - a failing tool must not end a turn
            logger.exception("tool %s failed", name)
            return {"error": f"{type(exc).__name__}: {exc}"}

    # -- registration ----------------------------------------------------

    def _register_all(self) -> None:
        # ---------------- reading the case ----------------

        self._add(
            "get_triage_state",
            "The current triage state for a case: urgency and department produced by "
            "the deterministic rules, which rules matched, which could not be ruled "
            "out, and what is still unknown. Read this to decide whether more "
            "questions would change anything. You cannot set the urgency; only the "
            "rules produce it.",
            {"case_id": _string("Case identifier", required=True)},
            self._get_triage_state,
        )
        self._add(
            "get_open_questions",
            "The facts the rules are still waiting on, most urgent first, with the "
            "rule that wants each one and a suggested wording. Use these to steer "
            "your questions - but ask them in your own words, in the patient's "
            "language, and follow whatever they actually say.",
            {"case_id": _string("Case identifier", required=True)},
            self._get_open_questions,
        )
        self._add(
            "get_case_facts",
            "Everything established about this case so far, with where each fact "
            "came from and the patient's own words behind it.",
            {"case_id": _string("Case identifier", required=True)},
            self._get_case_facts,
        )

        # ---------------- memory and history ----------------

        self._add(
            "recall_patient_memory",
            "What VITA remembers about this patient from previous visits: what they "
            "presented with, whether it resolved, what they take, what a clinician "
            "concluded. Search it with a question. Always worth calling at the start "
            "of an intake.",
            {
                "patient_id": _string("Patient identifier", required=True),
                "query": _string("What you want to know about them"),
            },
            self._recall_patient_memory,
        )
        self._add(
            "get_patient_history",
            "This patient's previous cases: when, what complaint, what urgency, how "
            "it was dispositioned. Use it to see whether they keep coming back with "
            "the same unresolved problem.",
            {"patient_id": _string("Patient identifier", required=True)},
            self._get_patient_history,
        )

        # ---------------- grounding ----------------

        self._add(
            "get_triage_rules",
            "Every triage rule for a complaint, with conditions, urgency, department "
            "and the framework it was adapted from. Selection is a deterministic "
            "lookup - all rules for the complaint, so none can be missed.",
            {"complaint": _string("fever, injury, chest_pain, breathing_difficulty, abdominal_pain or general", required=True)},
            self._get_triage_rules,
        )
        self._add(
            "retrieve_policy",
            "Hospital policy governing a situation - after-hours routing, escalation, "
            "capacity, scope, notification, transport, admission. If nothing covers "
            "it, say so rather than inventing a rule.",
            {"query": _string("What the policy question is about", required=True)},
            self._retrieve_policy,
        )
        self._add(
            "retrieve_guidance",
            "Clinical guidance explaining why a complaint's rules look the way they "
            "do. Useful when writing reasoning for a clinician.",
            {"complaint": _string("Complaint name", required=True)},
            self._retrieve_guidance,
        )

        # ---------------- hospital ----------------

        self._add(
            "list_doctors",
            "Clinicians and who is on call, optionally for one department.",
            {"department": _string("Department name. Omit for all.")},
            self._list_doctors,
        )
        self._add(
            "list_free_rooms",
            "Rooms with nobody in them. Read this before requesting an admission so "
            "your reasoning reflects what is actually available - but do not name a "
            "room. The approving clinician chooses.",
            {"department": _string("Department name. Omit for all.")},
            self._list_free_rooms,
        )
        self._add(
            "get_department_status",
            "A department's location, hours and occupancy. Occupancy is context for a "
            "human; it never changes a patient's urgency.",
            {"department": _string("Department name. Omit for all.")},
            self._get_department_status,
        )

        # ---------------- writing facts ----------------

        self._add(
            "record_facts",
            "Record what the patient has told you, as structured facts. Only fact "
            "keys the knowledge base already defines are accepted. Use 'true', "
            "'false' or 'unknown' for symptoms, and a number for measurements. Quote "
            "the patient's own words as evidence for each one. Never record a fact "
            "the patient did not establish.",
            {
                "case_id": _string("Case identifier", required=True),
                "facts": {
                    "type": "array",
                    "description": "The facts to record.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "Fact key, e.g. breathing_difficulty"},
                            "value": {"type": "string", "description": "'true', 'false', 'unknown', or a number"},
                            "evidence": {"type": "string", "description": "The patient's own words that established it"},
                        },
                    },
                    "required": True,
                },
            },
            self._record_facts,
        )

        self._add(
            "set_complaint",
            "Tell the system which of the five covered complaints this is, so the "
            "right rules apply and the right questions become relevant. Use "
            "'out_of_scope' when the patient clearly has a medical problem that is "
            "none of them - a stroke, a pregnancy complication, a mental health "
            "crisis, a child, an eye or dental problem. Call this as soon as you "
            "know; until you do, only the general rules apply and the questions you "
            "are offered will not fit the patient.",
            {
                "case_id": _string("Case identifier", required=True),
                "complaint": _string(
                    "fever, injury, chest_pain, breathing_difficulty, abdominal_pain, "
                    "out_of_scope or undetermined",
                    required=True,
                ),
            },
            self._set_complaint,
        )

        # ---------------- requests: proposals, never actions ----------------

        self._add(
            "request_notify_doctor",
            "Ask for the on-call clinician to be sent this case. Creates a request "
            "for human approval; it does not send anything.",
            {
                "case_id": _string("Case identifier", required=True),
                "reasoning": _string("Why this clinician should see this case now", required=True),
                "department": _string("Department whose on-call clinician should be notified"),
            },
            self._request_notify_doctor,
        )
        self._add(
            "request_admission",
            "Ask for the patient to be admitted. Creates a request for human "
            "approval; the approving clinician assigns the room. State plainly why "
            "sending them home would be wrong - repeated unresolved presentations, "
            "deterioration between visits, a condition needing observation.",
            {
                "case_id": _string("Case identifier", required=True),
                "reasoning": _string("Why admission rather than discharge", required=True),
                "department": _string("Department the admission would be under"),
            },
            self._request_admission,
        )
        self._add(
            "request_ambulance",
            "Ask for emergency transport. Creates a request for human approval. Only "
            "for cases the rules have already graded HIGH or CRITICAL.",
            {
                "case_id": _string("Case identifier", required=True),
                "reasoning": _string("Why transport is needed", required=True),
                "pickup_location": _string("Pickup location, if the patient gave one"),
            },
            self._request_ambulance,
        )
        self._add(
            "request_raise_urgency",
            "Ask for the urgency to be raised above what the rules produced, when you "
            "can see something the rules cannot. You can only raise it. Creates a "
            "request for human approval.",
            {
                "case_id": _string("Case identifier", required=True),
                "urgency": _string("LOW, MODERATE, HIGH or CRITICAL", required=True),
                "reasoning": _string("What you can see that the rules cannot", required=True),
            },
            self._request_raise_urgency,
        )
        self._add(
            "request_referral",
            "Ask for the case to be routed to a different department, for example a "
            "complaint outside the five this rule set covers. Creates a request for "
            "human approval.",
            {
                "case_id": _string("Case identifier", required=True),
                "department": _string("Department to refer to", required=True),
                "reasoning": _string("Why this department", required=True),
            },
            self._request_referral,
        )

    # -- read handlers ---------------------------------------------------

    def _case_or_error(self, case_id: str) -> Any:
        case = self.services.get_case(case_id)
        return case if case is not None else None

    def _get_triage_state(self, case_id: str) -> dict[str, Any]:
        case = self._case_or_error(case_id)
        if case is None:
            return {"error": f"no case {case_id!r}"}

        decision = decide(
            self.services.kb.rules_for(case.complaint),
            case.facts,
            complaint=case.complaint,
            contradictions=case.contradictions,
            final=False,
        )
        return {
            "case_id": case_id,
            "complaint": case.complaint.value,
            "urgency": decision.urgency.value,
            "department": decision.department,
            "matched_rules": decision.cited_rules,
            "could_not_rule_out": [
                {"rule_id": e.rule.rule_id, "urgency": e.rule.urgency.value,
                 "waiting_on": [c.fact for c in e.blocking]}
                for e in decision.potential
                if e.rule.urgency.rank >= Urgency.HIGH.rank
            ],
            "unknowns": decision.unknowns,
            "requires_human_review": decision.requires_human_review,
            "note": "Urgency is produced by the rules. You may read it; you cannot set it.",
        }

    def _get_open_questions(self, case_id: str) -> dict[str, Any]:
        case = self._case_or_error(case_id)
        if case is None:
            return {"error": f"no case {case_id!r}"}

        evaluations = evaluate_all(self.services.kb.rules_for(case.complaint), case.facts)
        questions: list[dict[str, Any]] = []
        seen: set[str] = set()

        # Complaint-specific rules first, general modifiers last. Both are
        # evaluated, but a patient who came in breathless should be asked about
        # their breathing before being asked their age.
        def priority(evaluation: Any) -> tuple[int, int]:
            general = evaluation.rule.complaint is Complaint.GENERAL
            return (int(general), -evaluation.rule.urgency.rank)

        for evaluation in sorted(
            (e for e in evaluations if e.state is MatchState.POTENTIAL), key=priority
        ):
            for condition in evaluation.blocking:
                if condition.fact in seen:
                    continue
                seen.add(condition.fact)
                question = self.services.kb.question(condition.fact)
                questions.append(
                    {
                        "fact": condition.fact,
                        "wanted_by_rule": evaluation.rule.rule_id,
                        "rule_urgency": evaluation.rule.urgency.value,
                        "suggested_wording": question.text if question else "",
                        "already_asked": case.asked_counts.get(condition.fact, 0),
                    }
                )

        return {"case_id": case_id, "open_questions": questions[:12]}

    def _get_case_facts(self, case_id: str) -> dict[str, Any]:
        case = self._case_or_error(case_id)
        if case is None:
            return {"error": f"no case {case_id!r}"}
        return {
            "case_id": case_id,
            "complaint": case.complaint.value,
            "language": case.language,
            "facts": {k: f.as_dict() for k, f in case.facts.items() if f.is_known},
            "contradictions": [c.describe() for c in case.contradictions],
        }

    def _recall_patient_memory(self, patient_id: str, query: str = "") -> dict[str, Any]:
        memories = self.services.memory.recall(patient_id=patient_id, query=query, limit=6)
        return {
            "patient_id": patient_id,
            "available": self.services.memory.available,
            "memories": [m.as_dict() for m in memories],
        }

    def _get_patient_history(self, patient_id: str) -> dict[str, Any]:
        visits = self.services.cases.patient_visits(patient_id)
        return {
            "patient_id": patient_id,
            "visit_count": len(visits),
            "visits": visits,
        }

    def _get_triage_rules(self, complaint: str) -> dict[str, Any]:
        try:
            parsed = Complaint(complaint.strip().lower())
        except ValueError:
            return {"error": f"unknown complaint {complaint!r}", "valid": [c.value for c in Complaint]}
        rules = self.services.kb.rules_for(parsed)
        return {"complaint": parsed.value, "count": len(rules), "rules": [r.as_dict() for r in rules]}

    def _retrieve_policy(self, query: str) -> dict[str, Any]:
        hits = self.services.retriever.policy_for(query)
        return {
            "query": query,
            "hits": [h.as_dict() for h in hits],
            "note": "No governing clause means the situation is not covered. Escalate rather than infer one.",
        }

    def _retrieve_guidance(self, complaint: str) -> dict[str, Any]:
        return {"complaint": complaint,
                "hits": [h.as_dict() for h in self.services.retriever.guidance_for(complaint)]}

    def _list_doctors(self, department: str = "") -> dict[str, Any]:
        doctors = self.services.hospital.doctors
        if department:
            doctors = [d for d in doctors if d.department.lower() == department.lower()]
        return {"department": department or "all", "doctors": [d.as_dict() for d in doctors]}

    def _list_free_rooms(self, department: str = "") -> dict[str, Any]:
        occupied = self.services.requests.occupied_rooms()
        rooms = self.services.hospital.free_rooms(occupied, department)
        return {
            "department": department or "all",
            "free": [r.as_dict() for r in rooms],
            "note": "Do not name a room in an admission request. The approving clinician chooses.",
        }

    def _get_department_status(self, department: str = "") -> dict[str, Any]:
        directory = self.services.hospital
        if department:
            found = directory.department(department)
            if found is None:
                return {"error": f"no department {department!r}"}
            return {"department": found.as_dict(), "routing_note": directory.routing_note(department)}
        return {"departments": [d.as_dict() for d in directory.departments]}

    # -- fact recording --------------------------------------------------

    def _record_facts(self, case_id: str, facts: list[dict[str, Any]]) -> dict[str, Any]:
        """Record facts, rejecting anything the knowledge base does not define.

        The bound is what makes this tool safe to expose. A planner that could
        write arbitrary keys could write `urgency`, and a planner that could
        write arbitrary values could write a symptom nobody reported.
        """
        case = self._case_or_error(case_id)
        if case is None:
            return {"error": f"no case {case_id!r}"}
        if not isinstance(facts, list):
            return {"error": "facts must be a list of {key, value, evidence} objects"}

        recorded: list[str] = []
        rejected: list[dict[str, str]] = []

        for entry in facts:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("key", "")).strip()
            raw = str(entry.get("value", "")).strip()
            evidence = str(entry.get("evidence", "")).strip()

            if not key or self.services.kb.question(key) is None:
                rejected.append({"key": key, "why": "not a fact this knowledge base defines"})
                continue

            value: Any = Tri.coerce(raw)
            if value is Tri.UNKNOWN:
                try:
                    value = float(raw)
                except ValueError:
                    rejected.append({"key": key, "why": f"value {raw!r} is neither three-valued nor numeric"})
                    continue

            case.record(
                Fact(
                    key=key,
                    value=value,
                    source=FactSource.FOLLOWUP_ANSWER if case.turn_number > 1 else FactSource.PATIENT_VERBATIM,
                    turn=case.turn_number,
                    verbatim=evidence,
                    language=case.language,
                    confidence=0.9,
                    agent="planner",
                )
            )
            recorded.append(key)

        # A recorded symptom often settles the complaint on its own, and a case
        # with no complaint is evaluated against the general modifiers only -
        # which is how a breathing problem ends up being asked about pregnancy.
        if case.complaint is Complaint.UNDETERMINED:
            from .core.rules import infer_complaint

            inferred = infer_complaint(case.facts)
            if inferred is not Complaint.UNDETERMINED:
                case.complaint = inferred
                logger.info("case %s complaint inferred as %s", case_id, inferred.value)

        self.services.cases.save(case)
        return {
            "case_id": case_id,
            "recorded": recorded,
            "rejected": rejected,
            "complaint": case.complaint.value,
        }

    def _set_complaint(self, case_id: str, complaint: str) -> dict[str, Any]:
        case = self._case_or_error(case_id)
        if case is None:
            return {"error": f"no case {case_id!r}"}
        try:
            parsed = Complaint(complaint.strip().lower())
        except ValueError:
            return {"error": f"unknown complaint {complaint!r}",
                    "valid": [c.value for c in Complaint]}

        case.complaint = parsed
        if parsed is Complaint.OUT_OF_SCOPE:
            case.out_of_scope = True
        self.services.cases.save(case)
        logger.info("case %s complaint set to %s by the planner", case_id, parsed.value)
        return {
            "case_id": case_id,
            "complaint": parsed.value,
            "rules_now_in_play": len(self.services.kb.rules_for(parsed)),
        }

    # -- request handlers ------------------------------------------------

    def _raise(self, case_id: str, kind: RequestKind, summary: str, reasoning: str,
               payload: dict[str, Any] | None = None) -> dict[str, Any]:
        case = self._case_or_error(case_id)
        if case is None:
            return {"error": f"no case {case_id!r}"}
        if not reasoning.strip():
            return {"error": "a request without reasoning cannot be reviewed; say why"}

        evidence: list[str] = []
        if case.decision:
            evidence.extend(f"rule {r}" for r in case.decision.cited_rules)
        evidence.extend(f"red flag {f}" for f in case.red_flags)

        request = Request.create(
            case_id=case_id,
            patient_id=case.patient_id,
            kind=kind,
            summary=summary,
            reasoning=reasoning,
            payload=payload or {},
            evidence=evidence,
        )
        self.services.requests.add(request)
        self.services.cases.audit(case_id, f"request_raised:{kind.value}", actor="planner", detail=summary)
        return {
            "request_id": request.request_id,
            "status": request.status.value,
            "note": "Created for human approval. Nothing has happened yet.",
        }

    def _request_notify_doctor(self, case_id: str, reasoning: str, department: str = "") -> dict[str, Any]:
        case = self._case_or_error(case_id)
        target = department or (case.decision.department if case and case.decision else "")
        doctor = self.services.hospital.on_call_for(target) if target else None
        return self._raise(
            case_id, RequestKind.NOTIFY_DOCTOR,
            f"Notify {doctor.name if doctor else 'on-call clinician'} ({target or 'department unset'})",
            reasoning, {"department": target, "doctor": doctor.name if doctor else ""},
        )

    def _request_admission(self, case_id: str, reasoning: str, department: str = "") -> dict[str, Any]:
        case = self._case_or_error(case_id)
        target = department or (case.decision.department if case and case.decision else "")
        return self._raise(
            case_id, RequestKind.ADMIT_PATIENT,
            f"Admit patient under {target or 'a department to be chosen'}",
            reasoning, {"department": target},
        )

    def _request_ambulance(self, case_id: str, reasoning: str, pickup_location: str = "") -> dict[str, Any]:
        case = self._case_or_error(case_id)
        if case and case.decision and case.decision.urgency.rank < Urgency.HIGH.rank:
            return {
                "error": (
                    f"transport may not be requested at urgency "
                    f"{case.decision.urgency.value}; the threshold is HIGH"
                )
            }
        return self._raise(
            case_id, RequestKind.REQUEST_AMBULANCE, "Emergency transport",
            reasoning, {"pickup_location": pickup_location},
        )

    def _request_raise_urgency(self, case_id: str, urgency: str, reasoning: str) -> dict[str, Any]:
        try:
            level = Urgency(urgency.strip().upper())
        except ValueError:
            return {"error": f"unknown urgency {urgency!r}", "valid": [u.value for u in Urgency]}

        case = self._case_or_error(case_id)
        if case and case.decision and level.rank <= case.decision.urgency.rank:
            return {
                "error": (
                    f"the rules already produced {case.decision.urgency.value}. "
                    "You may only ask to raise the urgency, never to lower it."
                )
            }
        return self._raise(
            case_id, RequestKind.RAISE_URGENCY, f"Raise urgency to {level.value}",
            reasoning, {"urgency": level.value},
        )

    def _request_referral(self, case_id: str, department: str, reasoning: str) -> dict[str, Any]:
        return self._raise(
            case_id, RequestKind.REFER_DEPARTMENT, f"Refer to {department}",
            reasoning, {"department": department},
        )
