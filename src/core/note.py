"""
The triage note — rendered from the decision, never written by the model.

Every sentence in the output below is assembled from a rule id, a rule's stored
rationale, and facts that carry their own provenance. Gemini is not asked to
explain the recommendation, and this is not a stylistic preference. A model
asked to justify a decision it did not make will produce a fluent justification
whether or not it matches the actual reasoning, and a plausible explanation of
the wrong reasoning is worse than no explanation at all.

The note answers the four questions the problem statement asks for, and each
comes from a different place in the case:

* the recommendation, from the rule engine
* the reasoning, from the matched rule's own rationale and its cited framework
* what the patient reported against what the follow-ups established, from the
  source recorded on every fact
* what remains unknown, from the conditions still blocking an unresolved rule

Anything the system did not establish is printed as unknown. There is no path
through this module that turns silence into a negative finding.
"""

from __future__ import annotations

from typing import Any

from ..config import SystemMode
from .case import Case
from .knowledge import KnowledgeBase
from .schema import EscalationReason, Fact, Tri

#: How an escalation reason is explained to a human, in one line.
_ESCALATION_TEXT = {
    EscalationReason.RULE_REQUIRES_REVIEW: (
        "A matched rule requires clinical review before the patient is routed."
    ),
    EscalationReason.UNRESOLVED_UNKNOWN: (
        "A high-urgency rule could not be ruled out because required information "
        "could not be established."
    ),
    EscalationReason.CONTRADICTORY_REPORT: (
        "The patient gave incompatible answers about the same finding. VITA has "
        "not chosen between them."
    ),
    EscalationReason.OUT_OF_SCOPE: (
        "The complaint falls outside the five conditions this rule set covers."
    ),
    EscalationReason.SCOPE_UNCERTAIN: (
        "The description sat close to the boundary of what this rule set covers. "
        "It has been triaged, but the complaint itself should be confirmed."
    ),
    EscalationReason.DEGRADED_MODE: (
        "Language understanding was unavailable, so the facts below were extracted "
        "by keyword fallback and are not conversationally confirmed."
    ),
    EscalationReason.RED_FLAG: (
        "A critical presentation was recognised directly from the patient's own words."
    ),
    EscalationReason.NO_RULE_MATCHED: (
        "No triage rule matched the established facts."
    ),
}


def _render_value(value: Any) -> str:
    if isinstance(value, Tri):
        return {"true": "Yes", "false": "No", "unknown": "Unknown"}[value.value]
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _label(fact: Fact) -> str:
    return f"{fact.key.replace('_', ' ').capitalize()}: {_render_value(fact.value)}"


def build_note(case: Case, kb: KnowledgeBase) -> dict[str, Any]:
    """Assemble the triage note as structured data.

    Returned as a dictionary rather than a string so the dashboard can render
    each section on its own and the notification path can quote from it without
    parsing prose back apart.
    """
    decision = case.decision
    if decision is None:
        return {"case_id": case.case_id, "status": "incomplete", "sections": {}}

    reasoning = [
        {
            "rule_id": e.rule.rule_id,
            "urgency": e.rule.urgency.value,
            "department": e.rule.department,
            "rationale": e.rule.rationale,
            "source": e.rule.source,
            "conditions_met": [c.describe() for c in e.satisfied],
        }
        for e in decision.matched
    ]

    unresolved = [
        {
            "rule_id": e.rule.rule_id,
            "urgency": e.rule.urgency.value,
            "rationale": e.rule.rationale,
            "waiting_on": [c.describe() for c in e.blocking],
        }
        for e in decision.potential
        if e.rule.urgency.rank >= 2
    ]

    return {
        "case_id": case.case_id,
        "generated_at": case.updated_at,
        "complaint": case.complaint.value,
        "language": case.language,
        "recommendation": {
            "urgency": decision.urgency.value,
            "department": decision.department,
            "requires_human_review": decision.requires_human_review,
            "cited_rules": decision.cited_rules,
        },
        "override": (
            {
                "urgency": case.override_urgency,
                "reason": case.override_reason,
                "by": case.override_by,
                "at": case.override_at,
            }
            if case.override_urgency
            else None
        ),
        "reasoning": reasoning,
        "patient_reported": [_label(f) for f in sorted(case.reported(), key=lambda f: f.key)],
        "established_by_followup": [
            _label(f) for f in sorted(case.established(), key=lambda f: f.key)
        ],
        "recalled_from_records": [_label(f) for f in sorted(case.recalled(), key=lambda f: f.key)],
        "unknown": [u.replace("_", " ") for u in decision.unknowns],
        "unresolved_rules": unresolved,
        "contradictions": [c.describe() for c in case.contradictions],
        "red_flags": [
            f.as_dict() for f in kb.red_flags if f.id in case.red_flags
        ],
        "escalation": {
            "required": decision.requires_human_review,
            "reasons": [
                {"code": r.value, "explanation": _ESCALATION_TEXT.get(r, r.value)}
                for r in decision.escalation_reasons
            ],
            "notes": decision.notes,
        },
        "system_mode": case.decided_in_mode.value,
        "disclaimer": (
            "VITA is a triage assistant. It does not diagnose. Every recommendation "
            "above is produced by a deterministic rule engine and cites the rule "
            "that produced it. " + kb.disclaimer
        ),
    }


def render_text(note: dict[str, Any]) -> str:
    """Render the note as plain text, for the notification and for reading.

    Deliberately the same content as the structured form. A summary that says
    less than the record it summarises is how information goes missing between a
    system and the clinician relying on it.
    """
    if not note.get("recommendation"):
        return f"Case {note.get('case_id', '?')}: intake incomplete."

    rec = note["recommendation"]
    lines: list[str] = [
        "VITA TRIAGE NOTE",
        "",
        f"Case ID:            {note['case_id']}",
        f"Primary complaint:  {note['complaint'].replace('_', ' ')}",
        f"Recommended urgency:    {rec['urgency']}",
        f"Recommended department: {rec['department']}",
    ]

    if note.get("override"):
        o = note["override"]
        lines += [
            "",
            f"CLINICIAN OVERRIDE: {o['urgency']} - {o['reason']} ({o['by']}, {o['at']})",
        ]

    lines += ["", "RULE APPLIED"]
    if note["reasoning"]:
        for r in note["reasoning"]:
            lines.append(f"  {r['rule_id']} - {r['urgency']}, {r['department']}")
            lines.append(f"    {r['rationale']}")
            if r["source"]:
                lines.append(f"    Source: {r['source']}")
            for condition in r["conditions_met"]:
                lines.append(f"    - {condition}")
    else:
        lines.append("  None. No rule matched the established facts.")

    lines += ["", "PATIENT REPORTED"]
    lines += [f"  - {item}" for item in note["patient_reported"]] or [
        "  - Nothing recorded from the opening description."
    ]

    lines += ["", "ESTABLISHED THROUGH FOLLOW-UP"]
    lines += [f"  - {item}" for item in note["established_by_followup"]] or [
        "  - No follow-up answers recorded."
    ]

    if note["recalled_from_records"]:
        lines += ["", "FROM PREVIOUS VISITS"]
        lines += [f"  - {item}" for item in note["recalled_from_records"]]

    lines += ["", "COULD NOT BE ESTABLISHED"]
    lines += [f"  - {item}" for item in note["unknown"]] or [
        "  - Everything the rules needed was established."
    ]

    if note["contradictions"]:
        lines += ["", "CONFLICTING ANSWERS - NOT RESOLVED BY VITA"]
        lines += [f"  - {item}" for item in note["contradictions"]]

    if note["unresolved_rules"]:
        lines += ["", "COULD NOT BE RULED OUT"]
        for r in note["unresolved_rules"]:
            lines.append(f"  {r['rule_id']} ({r['urgency']}) - waiting on:")
            lines += [f"    - {w}" for w in r["waiting_on"]]

    lines += ["", "ACTION"]
    if note["escalation"]["required"]:
        lines.append("  Human clinical assessment required.")
        for reason in note["escalation"]["reasons"]:
            lines.append(f"  - {reason['explanation']}")
    else:
        lines.append(f"  Route to {rec['department']} on the standard pathway.")

    if note["system_mode"] != SystemMode.FULL.value:
        lines += ["", f"SYSTEM MODE AT DECISION: {note['system_mode']}"]

    lines += ["", note["disclaimer"]]
    return "\n".join(lines)
