"""Run the elephantmemory benchmark on Modal.

Three functions, one per "infra profile" — each only boots the services
its adapters need and gets sized accordingly:

  - run_lite:  pgvector_diy, claude_memory, mem0  (postgres,            4GB)
  - run_zep:   zep                                 (neo4j + JVM,         8GB)
  - run_letta: letta                               (letta server + sqlite, 4GB)

Each function returns the run's results.json text; the local entrypoint
writes it back to results/runs/modal_<run_id>/.

Setup:
    pip install modal
    modal token new                                              # one-time
    modal secret create elephantmemory \\
        ANTHROPIC_API_KEY=... OPENAI_API_KEY=...                 # one-time

Usage:
    modal run modal_app.py --adapters pgvector_diy,mem0
    modal run modal_app.py --adapters zep
    modal run modal_app.py --adapters letta
    modal run modal_app.py --adapters pgvector_diy,mem0,claude_memory,zep,letta  # fans out
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent

PYTHON_DEPS_BASE = [
    "anthropic>=0.40",
    "openai>=1.50",
    "psycopg[binary,pool]>=3.2",
    "pgvector>=0.3",
    "pyyaml>=6.0",
    "click>=8.1",
    "tabulate>=0.9",
    "python-dotenv>=1.0",
    "httpx>=0.27",
]
IGNORE = ["results", "__pycache__", ".venv", ".git", "*.pyc", ".pytest_cache",
          ".ruff_cache", ".mypy_cache", "uv.lock", ".env"]

NEO4J_VERSION = "5.20.0"
NEO4J_PASSWORD = "elephantmemory"

app = modal.App("elephantmemory")
secret = modal.Secret.from_name(
    "elephantmemory", required_keys=["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
)
results_volume = modal.Volume.from_name("elephantmemory-results", create_if_missing=True)


# ─── images ───────────────────────────────────────────────────────────────────

lite_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "postgresql", "postgresql-contrib", "postgresql-server-dev-all",
        "build-essential", "git",
    )
    .run_commands(
        "git clone --depth 1 --branch v0.8.0 "
        "https://github.com/pgvector/pgvector.git /tmp/pgvector",
        "cd /tmp/pgvector && make && make install",
    )
    .pip_install(*PYTHON_DEPS_BASE, "mem0ai>=0.1.30")
    .add_local_dir(str(REPO_ROOT), "/repo", ignore=IGNORE)
)

zep_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("openjdk-17-jre-headless", "wget", "curl", "git")
    .run_commands(
        f"wget -q https://dist.neo4j.org/neo4j-community-{NEO4J_VERSION}-unix.tar.gz "
        f"-O /tmp/neo4j.tgz",
        "tar xzf /tmp/neo4j.tgz -C /opt && "
        f"ln -s /opt/neo4j-community-{NEO4J_VERSION} /opt/neo4j",
        f"/opt/neo4j/bin/neo4j-admin dbms set-initial-password {NEO4J_PASSWORD}",
        "rm /tmp/neo4j.tgz",
    )
    .pip_install(*PYTHON_DEPS_BASE, "graphiti-core>=0.3")
    .add_local_dir(str(REPO_ROOT), "/repo", ignore=IGNORE)
)

letta_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    # asyncpg is imported by letta.orm at module load even in sqlite mode.
    .pip_install(*PYTHON_DEPS_BASE, "letta-client>=0.1", "letta>=0.6", "asyncpg>=0.29")
    .add_local_dir(str(REPO_ROOT), "/repo", ignore=IGNORE)
)


# ─── service helpers ──────────────────────────────────────────────────────────

def _wait_for_port(host: str, port: int, *, timeout: float = 60.0, label: str = "") -> None:
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"{label or host + ':' + str(port)} did not come up in {timeout}s")


def _boot_postgres() -> None:
    import subprocess
    subprocess.run(["service", "postgresql", "start"], check=True)
    subprocess.run(
        ["su", "-", "postgres", "-c",
         "psql -c \"CREATE USER elephant WITH PASSWORD 'elephant' SUPERUSER;\""],
        check=False,
    )
    subprocess.run(
        ["su", "-", "postgres", "-c",
         "psql -c 'CREATE DATABASE elephant OWNER elephant;'"],
        check=False,
    )
    subprocess.run(
        ["su", "-", "postgres", "-c",
         "psql -d elephant -c 'CREATE EXTENSION IF NOT EXISTS vector;'"],
        check=True,
    )


def _boot_neo4j() -> None:
    import subprocess
    subprocess.Popen(
        ["/opt/neo4j/bin/neo4j", "console"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    _wait_for_port("localhost", 7687, timeout=120, label="neo4j bolt")


def _boot_letta_server() -> None:
    import os
    import subprocess
    env = os.environ.copy()
    env.setdefault("LETTA_SERVER_PASSWORD", "elephantmemory")
    subprocess.Popen(["letta", "server"], env=env)
    _wait_for_port("localhost", 8283, timeout=300, label="letta server")


def _run_cli(
    adapters: list[str], scenarios_path: str, *, extra_env: dict[str, str] | None = None
) -> tuple[str, str]:
    import os
    import subprocess
    if extra_env:
        os.environ.update(extra_env)
    cmd = ["python", "-m", "elephantmemory.cli", "run",
           "--scenarios", scenarios_path,
           "--out", "/results/runs"]
    for a in adapters:
        cmd.extend(["--adapter", a])
    subprocess.run(cmd, check=True, cwd="/repo")
    runs = sorted(Path("/results/runs").iterdir())
    latest = runs[-1]
    results_volume.commit()
    return latest.name, (latest / "results.json").read_text()


# ─── functions ────────────────────────────────────────────────────────────────

@app.function(
    image=lite_image, secrets=[secret],
    volumes={"/results": results_volume},
    timeout=3600, memory=4096, cpu=2.0,
)
def run_lite(adapters: list[str], scenarios_path: str = "scenarios") -> dict:
    valid = {"pgvector_diy", "claude_memory", "mem0"}
    bad = [a for a in adapters if a not in valid]
    if bad:
        raise ValueError(f"adapters {bad} not in lite set {sorted(valid)}")
    _boot_postgres()
    run_id, results_json = _run_cli(
        adapters, scenarios_path,
        extra_env={"POSTGRES_DSN": "postgresql://elephant:elephant@localhost:5432/elephant"},
    )
    return {"run_id": run_id, "results_json": results_json, "profile": "lite"}


@app.function(
    image=zep_image, secrets=[secret],
    volumes={"/results": results_volume},
    timeout=3600, memory=8192, cpu=2.0,
)
def run_zep(scenarios_path: str = "scenarios") -> dict:
    _boot_neo4j()
    run_id, results_json = _run_cli(
        ["zep"], scenarios_path,
        extra_env={
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": NEO4J_PASSWORD,
        },
    )
    return {"run_id": run_id, "results_json": results_json, "profile": "zep"}


@app.function(
    image=letta_image, secrets=[secret],
    volumes={"/results": results_volume},
    timeout=3600, memory=4096, cpu=2.0,
)
def run_letta(scenarios_path: str = "scenarios") -> dict:
    _boot_letta_server()
    run_id, results_json = _run_cli(
        ["letta"], scenarios_path,
        extra_env={"LETTA_BASE_URL": "http://localhost:8283"},
    )
    return {"run_id": run_id, "results_json": results_json, "profile": "letta"}


# ─── scheduled regression + aggregate report ──────────────────────────────────

REPORT_IMAGE = (
    modal.Image.debian_slim(python_version="3.11").pip_install("tabulate>=0.9")
)


@app.function(
    image=REPORT_IMAGE,
    schedule=modal.Cron("0 6 * * 0"),  # weekly Sunday 06:00 UTC
    timeout=4200,
)
def weekly_regression() -> dict:
    """Run the full 5-adapter grid; results land in the volume for trend tracking.

    Activate by deploying once: `modal deploy modal_app.py`
    """
    pending = [
        ("lite", run_lite.spawn(sorted(LITE), "scenarios")),
        ("zep", run_zep.spawn("scenarios")),
        ("letta", run_letta.spawn("scenarios")),
    ]
    summary: dict[str, str] = {}
    for label, call in pending:
        try:
            out = call.get()
            summary[label] = out["run_id"]
        except Exception as e:
            summary[label] = f"FAILED: {type(e).__name__}: {e}"
    return summary


@app.function(
    image=REPORT_IMAGE,
    volumes={"/results": results_volume},
    timeout=600,
)
def aggregate_report() -> str:
    """Walk every results.json in the volume; return a markdown trend report."""
    import json
    from collections import defaultdict
    from pathlib import Path

    from tabulate import tabulate

    runs = sorted(p for p in Path("/results/runs").iterdir() if p.is_dir())
    if not runs:
        return "no runs in volume yet"

    history: dict[tuple[str, str], list[tuple[str, float, int]]] = defaultdict(list)
    for run_dir in runs:
        rj = run_dir / "results.json"
        if not rj.exists():
            continue
        data = json.loads(rj.read_text())
        per_cell: dict[tuple[str, str], list[bool]] = defaultdict(list)
        for r in data:
            for o in r["outcomes"]:
                per_cell[(r["adapter"], o["category"])].append(o["passed"])
        for (a, c), passes in per_cell.items():
            n = len(passes)
            history[(a, c)].append((run_dir.name, sum(passes) / n, n))

    adapters = sorted({a for a, _ in history})
    categories = sorted({c for _, c in history})

    latest_run = runs[-1].name
    rows = []
    for cat in categories:
        row = [cat]
        for ad in adapters:
            cell = next(((p, n) for r, p, n in reversed(history.get((ad, cat), []))
                         if r == latest_run), None)
            row.append(f"{cell[0]:.0%} ({cell[1]})" if cell else "—")
        rows.append(row)

    out = ["# elephantmemory regression report\n",
           f"## Latest run: `{latest_run}`\n",
           tabulate(rows, headers=["category"] + adapters, tablefmt="github"), "\n",
           f"Total runs in history: {len(runs)}\n",
           "## Trend (last 5 runs, % passing per cell)\n"]
    for (ad, cat) in sorted(history):
        recent = history[(ad, cat)][-5:]
        if len(recent) < 2:
            continue
        trend = " → ".join(f"{p:.0%}" for _, p, _ in recent)
        delta = recent[-1][1] - recent[0][1]
        flag = " ⚠️" if delta < -0.10 else ""
        out.append(f"- **{ad}** / {cat}: {trend}{flag}")

    diffs = _diff_latest_two(runs)
    if diffs:
        out.append("\n## Probes that flipped vs. previous run\n")
        out.extend(diffs)

    return "\n".join(out)


def _diff_latest_two(runs: list) -> list[str]:
    import json
    if len(runs) < 2:
        return []
    prev, curr = runs[-2], runs[-1]
    p_data = json.loads((prev / "results.json").read_text()) if (prev / "results.json").exists() else []
    c_data = json.loads((curr / "results.json").read_text()) if (curr / "results.json").exists() else []
    p_idx = {(r["adapter"], o["probe_id"]): o["passed"] for r in p_data for o in r["outcomes"]}
    c_idx = {(r["adapter"], o["probe_id"]): o["passed"] for r in c_data for o in r["outcomes"]}
    flips = []
    for k, c_passed in c_idx.items():
        if k in p_idx and p_idx[k] != c_passed:
            arrow = "✅→❌" if p_idx[k] and not c_passed else "❌→✅"
            flips.append(f"- {arrow} **{k[0]}** / `{k[1]}`")
    return flips


# ─── local entrypoint ─────────────────────────────────────────────────────────

LITE = {"pgvector_diy", "claude_memory", "mem0"}


def _summarize(label: str, results_json: str) -> tuple[int, int]:
    import json
    parsed = json.loads(results_json)
    n_pass = sum(sum(1 for o in r["outcomes"] if o["passed"]) for r in parsed)
    n_total = sum(len(r["outcomes"]) for r in parsed)
    print(f"  {label}: {n_pass}/{n_total} probes passed")
    return n_pass, n_total


@app.local_entrypoint()
def main(adapters: str = "pgvector_diy", scenarios: str = "scenarios") -> None:
    requested = [a.strip() for a in adapters.split(",") if a.strip()]
    lite_set = [a for a in requested if a in LITE]
    has_zep = "zep" in requested
    has_letta = "letta" in requested
    unknown = [a for a in requested if a not in LITE | {"zep", "letta"}]
    if unknown:
        raise SystemExit(f"unknown adapters: {unknown}")

    print(f"running on Modal: lite={lite_set} zep={has_zep} letta={has_letta}")

    pending: list = []
    if lite_set:
        pending.append(("lite", run_lite.spawn(lite_set, scenarios)))
    if has_zep:
        pending.append(("zep", run_zep.spawn(scenarios)))
    if has_letta:
        pending.append(("letta", run_letta.spawn(scenarios)))

    print(f"\nspawned {len(pending)} parallel function(s); waiting...")
    for label, call in pending:
        out = call.get()
        run_id = out["run_id"]
        local_dir = REPO_ROOT / "results" / "runs" / f"modal_{label}_{run_id}"
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "results.json").write_text(out["results_json"])
        _summarize(label, out["results_json"])
        print(f"    → {local_dir}")


@app.local_entrypoint()
def report() -> None:
    """Pull the markdown trend report off the volume to results/trend_report.md."""
    md = aggregate_report.remote()
    out = REPO_ROOT / "results" / "trend_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"\nreport → {out}")


