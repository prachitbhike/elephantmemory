from __future__ import annotations

import statistics
import traceback
from collections import defaultdict

from .adapters.base import MemoryAdapter
from .judge import score_probe
from .types import (
    AdapterStats,
    ForgetOp,
    Probe,
    ProbeOutcome,
    Scenario,
    ScenarioResult,
    Session,
)


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    if len(xs) < 2:
        return xs[0]
    return statistics.quantiles(xs, n=100, method="inclusive")[int(p) - 1]


def _users_in_scenario(scenario: Scenario) -> set[str]:
    users: set[str] = set()
    for ev in scenario.events:
        users.add(ev.payload.user_id)
    return users


def run_scenario(adapter: MemoryAdapter, scenario: Scenario) -> ScenarioResult:
    write_latencies: list[float] = []
    write_cost = 0.0
    outcomes: list[ProbeOutcome] = []
    final_stats = AdapterStats()
    error: str | None = None

    try:
        for u in _users_in_scenario(scenario):
            adapter.reset_user(u)

        for ev in scenario.events:
            payload = ev.payload
            if isinstance(payload, Session):
                wr = adapter.record_session(payload)
                write_latencies.append(wr.latency_ms)
                write_cost += wr.cost_usd
            elif isinstance(payload, Probe):
                qr = adapter.query(payload.user_id, payload.prompt)
                score, passed, reason, judge_cost = score_probe(payload, qr.response)
                outcomes.append(
                    ProbeOutcome(
                        probe_id=payload.probe_id,
                        category=payload.category,
                        score=score,
                        passed=passed,
                        response=qr.response,
                        expected=payload.expected,
                        judge_reason=reason,
                        latency_ms=qr.latency_ms,
                        cost_usd=qr.cost_usd + judge_cost,
                    )
                )
            elif isinstance(payload, ForgetOp):
                fr = adapter.forget(payload.user_id, payload.predicate)
                write_cost += fr.cost_usd

        for u in _users_in_scenario(scenario):
            s = adapter.stats(u)
            final_stats = AdapterStats(
                facts_stored=final_stats.facts_stored + s.facts_stored,
                bytes_stored=final_stats.bytes_stored + s.bytes_stored,
            )
    except Exception as e:
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        adapter=adapter.name,
        outcomes=outcomes,
        write_latency_ms_p50=_percentile(write_latencies, 50) if write_latencies else 0.0,
        write_latency_ms_p95=_percentile(write_latencies, 95) if write_latencies else 0.0,
        write_cost_usd=write_cost,
        final_stats=final_stats,
        error=error,
    )


def aggregate_by_category(results: list[ScenarioResult]) -> dict[str, dict[str, float]]:
    by_cat: dict[str, list[ProbeOutcome]] = defaultdict(list)
    for r in results:
        for o in r.outcomes:
            by_cat[o.category].append(o)
    summary: dict[str, dict[str, float]] = {}
    for cat, outs in by_cat.items():
        n = len(outs)
        passed = sum(1 for o in outs if o.passed)
        avg_score = sum(o.score for o in outs) / n if n else 0.0
        summary[cat] = {
            "n": n,
            "passed": passed,
            "pass_rate": passed / n if n else 0.0,
            "avg_score": avg_score,
            "p50_query_ms": _percentile([o.latency_ms for o in outs], 50),
            "p95_query_ms": _percentile([o.latency_ms for o in outs], 95),
        }
    return summary
