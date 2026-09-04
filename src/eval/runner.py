
"""
The evaluation harness — evidence that the hard cases work, not a claim that they do.

A judge watching a three-minute video sees two patients. They cannot check that
the other thirty-eight behave, and "we handle edge cases" is exactly the sort of
assertion every submission makes. So the system carries its own exam paper and
marks itself: press the button, watch forty scenarios run through the real code,
read the result.

Two tiers, because they have different costs and different standards.

**Rule cases** put established facts straight into the rule engine. No model, no
network, milliseconds, and they **must** pass at 100%. The engine is
deterministic, so a failure here is a defect and never noise. This is where the
coverage lives: incomplete information, contradictions, degraded mode, out of
scope, and a set of under-triage guards that check a rule does *not* fire when
it should not.

**Conversation scenarios** drive the whole stack, model included, by scripting a
patient. These are slower, cost quota, and are graded more forgivingly, because
extraction is probabilistic and a suite that demands identical wording every run
would fail for reasons that do not matter.

The headline number is deliberately not accuracy. It is **under-triage count**,
and it must be zero. Over-triage costs a clinician's time; under-triage sends
someone home who should not have gone, and in a record it looks exactly like a
case that was genuinely low risk. Reporting the two together as one percentage
would hide the only failure that hurts a patient.

Expected outcomes are authored by hand. If a model wrote both the question and
the answer key, the suite would grade the model against itself and pass
everything.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import DATA_DIR
from ..core.rules import decide
from ..core.schema import (
    Complaint,
    Contradiction,
    Fact,
    FactSource,
    Tri,
    TriageDecision,
    Urgency,
)

logger = logging.getLogger(__name__)

EVAL_DIR = DATA_DIR / "eval"


@dataclass
class CaseResult:
    """How one scenario went."""

    id: str
    name: str
    category: str
    passed: bool
    tier: str = "rules"
    failures: list[str] = field(default_factory=list)
    expected_urgency: str = ""
    actual_urgency: str = ""
    actual_rules: list[str] = field(default_factory=list)
    escalated: bool = False
    elapsed_ms: int = 0

    @property
    def under_triaged(self) -> bool:
        """Did the system grade this *below* what was expected?

        The only failure in the suite that could hurt somebody. Tracked
        separately from pass/fail because a case can fail on the department or
        the cited rule while getting the urgency right, and those are not the
        same kind of wrong.
        """
        if not self.expected_urgency or not self.actual_urgency:
            return False
        try:
            return Urgency(self.actual_urgency).rank < Urgency(self.expected_urgency).rank
        except ValueError:
            return False

    @property
    def over_triaged(self) -> bool:
        if not self.expected_urgency or not self.actual_urgency:
            return False
        try:
            return Urgency(self.actual_urgency).rank > Urgency(self.expected_urgency).rank
        except ValueError:
            return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "tier": self.tier,
            "passed": self.passed,
            "failures": self.failures,
            "expected_urgency": self.expected_urgency,
            "actual_urgency": self.actual_urgency,
            "actual_rules": self.actual_rules,
            "escalated": self.escalated,
            "under_triaged": self.under_triaged,
            "over_triaged": self.over_triaged,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class Report:
    """The suite's result, led by the number that matters."""

    results: list[CaseResult] = field(default_factory=list)
    started_at: str = ""
    elapsed_ms: int = 0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def under_triaged(self) -> int:
        return sum(1 for r in self.results if r.under_triaged)

    @property
    def over_triaged(self) -> int:
        return sum(1 for r in self.results if r.over_triaged)

    @property
    def escalated(self) -> int:
        """Correct escalations. A feature of the system, not a failure of it."""
        return sum(1 for r in self.results if r.escalated)

    def by_category(self) -> dict[str, dict[str, int]]:
        buckets: dict[str, dict[str, int]] = {}
        for result in self.results:
            bucket = buckets.setdefault(result.category, {"total": 0, "passed": 0})
            bucket["total"] += 1
            bucket["passed"] += int(result.passed)
        return buckets

    def as_dict(self) -> dict[str, Any]:
        return {
            "headline": {
                "under_triaged": self.under_triaged,
                "passed": self.passed,
                "total": self.total,
                "over_triaged": self.over_triaged,
                "escalated": self.escalated,
            },
            "note": (
                "Under-triage is the number that matters: a case graded below what "
                "it should have been. Over-triage costs a clinician's time; "
                "under-triage sends home someone who should not have gone. "
                "Escalations are correct behaviour, not failures."
            ),
            "started_at": self.started_at,
            "elapsed_ms": self.elapsed_ms,
            "by_category": self.by_category(),
            "results": [r.as_dict() for r in self.results],
            "failures": [r.as_dict() for r in self.results if not r.passed],
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fact(key: str, raw: Any) -> Fact:
    """Build a fact from a fixture value, preserving 'unknown' as unknown."""
    if isinstance(raw, str) and raw.strip().lower() in {"true", "false", "unknown"}:
        value: Any = Tri.coerce(raw)
    else:
        value = raw
    return Fact(
        key=key,
        value=value,
        source=FactSource.FOLLOWUP_ANSWER,
        turn=1,
        verbatim="(evaluation fixture)",
        agent="eval",
    )


def load_rule_cases(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or EVAL_DIR / "rule_cases.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("cases", [])
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("could not load rule cases: %s", exc)
        return []


def load_scenarios(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or EVAL_DIR / "scenarios.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("scenarios", [])
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("could not load scenarios: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


def check(decision: TriageDecision, expect: dict[str, Any]) -> list[str]:
    """Compare a decision against an expectation. Returns what went wrong.

    Every assertion is optional, so a case can pin only what it is about. A
    fixture testing that a rule does *not* fire says nothing about the
    department, and demanding otherwise would make it fail for a reason it was
    never checking.
    """
    failures: list[str] = []

    if "urgency" in expect and decision.urgency.value != expect["urgency"]:
        failures.append(f"urgency: expected {expect['urgency']}, got {decision.urgency.value}")

    if "urgency_at_least" in expect:
        floor = Urgency(expect["urgency_at_least"])
        if decision.urgency.rank < floor.rank:
            failures.append(f"urgency: expected at least {floor.value}, got {decision.urgency.value}")

    if "urgency_at_most" in expect:
        ceiling = Urgency(expect["urgency_at_most"])
        if decision.urgency.rank > ceiling.rank:
            failures.append(f"urgency: expected at most {ceiling.value}, got {decision.urgency.value}")

    if "department" in expect and decision.department != expect["department"]:
        failures.append(f"department: expected {expect['department']}, got {decision.department}")

    cited = set(decision.cited_rules)
    for rule_id in expect.get("rules_include", []):
        if rule_id not in cited:
            failures.append(f"rule {rule_id} did not fire (cited: {sorted(cited) or 'none'})")
    for rule_id in expect.get("rules_exclude", []):
        if rule_id in cited:
            failures.append(f"rule {rule_id} fired but should not have")
    if expect.get("rules_empty") and cited:
        failures.append(f"expected no rules to fire, got {sorted(cited)}")

    if "human_review" in expect and decision.requires_human_review != expect["human_review"]:
        failures.append(
            f"human_review: expected {expect['human_review']}, got {decision.requires_human_review}"
        )

    reasons = {r.value for r in decision.escalation_reasons}
    for reason in expect.get("escalation_includes", []):
        if reason not in reasons:
            failures.append(f"escalation reason {reason!r} missing (got {sorted(reasons) or 'none'})")

    unknowns = set(decision.unknowns)
    for fact in expect.get("unknowns_include", []):
        if fact not in unknowns:
            failures.append(f"expected {fact!r} among unknowns")

    return failures


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_rule_cases(kb: Any, cases: list[dict[str, Any]] | None = None) -> list[CaseResult]:
    """Run the deterministic tier. No model, no network, must pass at 100%."""
    cases = cases if cases is not None else load_rule_cases()
    results: list[CaseResult] = []

    for spec in cases:
        started = time.monotonic()
        expect = spec.get("expect", {})

        try:
            complaint = Complaint(spec.get("complaint", "undetermined"))
        except ValueError:
            results.append(
                CaseResult(
                    id=spec.get("id", "?"),
                    name=spec.get("name", ""),
                    category=spec.get("category", "unknown"),
                    passed=False,
                    failures=[f"fixture names an unknown complaint: {spec.get('complaint')!r}"],
                )
            )
            continue

        facts = {key: _fact(key, value) for key, value in (spec.get("facts") or {}).items()}
        contradictions = [
            Contradiction(
                key=c["key"],
                earlier=_fact(c["key"], c["earlier"]),
                later=_fact(c["key"], c["later"]),
            )
            for c in (spec.get("contradictions") or [])
        ]
        # Contradiction fixtures give both readings the same turn by default,
        # which the detector treats as a restatement. Separate them.
        for contradiction in contradictions:
            contradiction.later.turn = 2

        decision = decide(
            kb.rules_for(complaint),
            facts,
            complaint=complaint,
            contradictions=contradictions,
            degraded=bool(spec.get("degraded", False)),
            final=bool(spec.get("final", True)),
        )

        failures = check(decision, expect)
        results.append(
            CaseResult(
                id=spec.get("id", "?"),
                name=spec.get("name", ""),
                category=spec.get("category", "unknown"),
                passed=not failures,
                tier="rules",
                failures=failures,
                expected_urgency=expect.get("urgency", ""),
                actual_urgency=decision.urgency.value,
                actual_rules=decision.cited_rules,
                escalated=decision.requires_human_review,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
        )

    return results


def run_scenarios(services: Any, scenarios: list[dict[str, Any]] | None = None, *, limit: int = 0) -> list[CaseResult]:
    """Run the conversational tier through the whole stack, model included.

    Slower and quota-hungry, so `limit` exists for a demonstration that should
    not spend a judge's rate budget. Graded on the outcome rather than on
    wording: extraction is probabilistic, and a suite demanding identical
    phrasing every run would fail for reasons that do not matter.
    """
    scenarios = scenarios if scenarios is not None else load_scenarios()
    if limit:
        scenarios = scenarios[:limit]

    results: list[CaseResult] = []

    for spec in scenarios:
        started = time.monotonic()
        expect = spec.get("expect", {})
        case = services.start_case(language=spec.get("language", "en"))

        answers = spec.get("answers", {})
        default = answers.get("*")
        messages = [spec["opening"]]
        result = None

        for _ in range(spec.get("max_turns", 10)):
            if not messages:
                break
            turn = services.message(case.case_id, messages.pop(0))
            if turn is None:
                break
            result = turn
            if turn.finished:
                break
            # Answer whatever was asked, falling back to the wildcard.
            reply = answers.get(turn.asked_about, default)
            if reply is None:
                break
            messages.append(reply)

        decision = case.decision
        if decision is None:
            results.append(
                CaseResult(
                    id=spec.get("id", "?"),
                    name=spec.get("name", ""),
                    category=spec.get("category", "unknown"),
                    passed=False,
                    tier="conversation",
                    failures=["intake did not reach a decision"],
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                )
            )
            continue

        failures = check(decision, expect)
        if "complaint" in expect and case.complaint.value != expect["complaint"]:
            failures.append(f"complaint: expected {expect['complaint']}, got {case.complaint.value}")
        if expect.get("out_of_scope") and not case.out_of_scope:
            failures.append("expected the case to be flagged out of scope")

        results.append(
            CaseResult(
                id=spec.get("id", "?"),
                name=spec.get("name", ""),
                category=spec.get("category", "unknown"),
                passed=not failures,
                tier="conversation",
                failures=failures,
                expected_urgency=expect.get("urgency", ""),
                actual_urgency=decision.urgency.value,
                actual_rules=decision.cited_rules,
                escalated=decision.requires_human_review,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
        )

    return results


def run(services: Any, *, include_conversation: bool = False, limit: int = 0) -> Report:
    """Run the suite. The deterministic tier always; the model tier on request."""
    started = time.monotonic()
    report = Report(started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    report.results.extend(run_rule_cases(services.kb))
    if include_conversation:
        report.results.extend(run_scenarios(services, limit=limit))

    report.elapsed_ms = int((time.monotonic() - started) * 1000)
    logger.info(
        "evaluation: %d/%d passed, %d under-triaged, %dms",
        report.passed,
        report.total,
        report.under_triaged,
        report.elapsed_ms,
    )
    return report
