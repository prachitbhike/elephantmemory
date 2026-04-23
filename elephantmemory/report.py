from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from tabulate import tabulate

from .types import ScenarioResult


def write_run(run_dir: Path, results: list[ScenarioResult]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps([asdict(r) for r in results], default=str, indent=2)
    )


def render_report(run_dir: Path) -> str:
    raw = json.loads((run_dir / "results.json").read_text())

    by_adapter_cat: dict[tuple[str, str], list[dict]] = defaultdict(list)
    write_stats: dict[str, dict] = defaultdict(lambda: {"latencies": [], "cost": 0.0, "facts": 0})

    for r in raw:
        adapter = r["adapter"]
        write_stats[adapter]["latencies"].append(r["write_latency_ms_p95"])
        write_stats[adapter]["cost"] += r["write_cost_usd"]
        write_stats[adapter]["facts"] += r["final_stats"]["facts_stored"]
        for o in r["outcomes"]:
            by_adapter_cat[(adapter, o["category"])].append(o)

    accuracy_rows = []
    categories = sorted({c for _, c in by_adapter_cat})
    adapters = sorted({a for a, _ in by_adapter_cat})

    header = ["category"] + adapters
    for cat in categories:
        row = [cat]
        for ad in adapters:
            outs = by_adapter_cat.get((ad, cat), [])
            if not outs:
                row.append("—")
            else:
                pr = sum(1 for o in outs if o["passed"]) / len(outs)
                row.append(f"{pr:.0%} ({len(outs)})")
        accuracy_rows.append(row)

    perf_rows = []
    for ad in adapters:
        s = write_stats[ad]
        avg_p95 = sum(s["latencies"]) / len(s["latencies"]) if s["latencies"] else 0.0
        perf_rows.append(
            [ad, f"{avg_p95:.0f}", f"${s['cost']:.4f}", s["facts"]]
        )

    out = ["# Run report\n", "## Pass rate by category\n",
           tabulate(accuracy_rows, headers=header, tablefmt="github"), "\n",
           "## Write performance & cost\n",
           tabulate(perf_rows,
                    headers=["adapter", "avg p95 write ms", "total write $", "facts stored"],
                    tablefmt="github"), "\n"]
    return "\n".join(out)
