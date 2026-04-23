"""Run the elephantmemory benchmark on Modal.

One function per "infra profile" so each one only boots the services it
needs and we can request appropriate RAM:

  - run_lite:  pgvector_diy, claude_memory, mem0  (postgres only, 4GB)
  - run_zep:   zep                                 (postgres + neo4j, 8GB)  TODO
  - run_letta: letta                               (letta server, 4GB)      TODO

Each function returns the run's results.json text; the local entrypoint
writes it back to results/runs/modal_<run_id>/.

Setup:
    pip install modal
    modal token new                                              # one-time
    modal secret create elephantmemory \\
        ANTHROPIC_API_KEY=... OPENAI_API_KEY=...                 # one-time

Usage:
    modal run modal_app.py --adapters pgvector_diy,mem0
    modal run modal_app.py --adapters claude_memory --scenarios scenarios
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent
PYTHON_DEPS = [
    "anthropic>=0.40",
    "openai>=1.50",
    "psycopg[binary,pool]>=3.2",
    "pgvector>=0.3",
    "pyyaml>=6.0",
    "click>=8.1",
    "tabulate>=0.9",
    "python-dotenv>=1.0",
    "httpx>=0.27",
    "mem0ai>=0.1.30",
]
IGNORE = ["results", "__pycache__", ".venv", ".git", "*.pyc", ".pytest_cache",
          ".ruff_cache", ".mypy_cache", "uv.lock", ".env"]

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
    .pip_install(*PYTHON_DEPS)
    .add_local_dir(str(REPO_ROOT), "/repo", ignore=IGNORE)
)

app = modal.App("elephantmemory")
secret = modal.Secret.from_name("elephantmemory", required_keys=["ANTHROPIC_API_KEY", "OPENAI_API_KEY"])
results_volume = modal.Volume.from_name("elephantmemory-results", create_if_missing=True)


def _boot_postgres() -> None:
    import subprocess
    subprocess.run(["service", "postgresql", "start"], check=True)
    # Idempotent — these may already exist on a snapshotted container
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


def _run_cli(adapters: list[str], scenarios_path: str) -> tuple[str, str]:
    import os
    import subprocess
    os.environ["POSTGRES_DSN"] = "postgresql://elephant:elephant@localhost:5432/elephant"
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


@app.function(
    image=lite_image,
    secrets=[secret],
    volumes={"/results": results_volume},
    timeout=3600,
    memory=4096,
    cpu=2.0,
)
def run_lite(adapters: list[str], scenarios_path: str = "scenarios") -> dict:
    valid = {"pgvector_diy", "claude_memory", "mem0"}
    bad = [a for a in adapters if a not in valid]
    if bad:
        raise ValueError(f"adapters {bad} not in lite set {sorted(valid)}")
    _boot_postgres()
    run_id, results_json = _run_cli(adapters, scenarios_path)
    return {"run_id": run_id, "results_json": results_json, "profile": "lite"}


@app.local_entrypoint()
def main(adapters: str = "pgvector_diy", scenarios: str = "scenarios") -> None:
    import json
    adapter_list = [a.strip() for a in adapters.split(",") if a.strip()]
    print(f"running on Modal: adapters={adapter_list}")

    out = run_lite.remote(adapter_list, scenarios)
    run_id = out["run_id"]

    local_dir = REPO_ROOT / "results" / "runs" / f"modal_{run_id}"
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "results.json").write_text(out["results_json"])

    parsed = json.loads(out["results_json"])
    n_pass = sum(sum(1 for o in r["outcomes"] if o["passed"]) for r in parsed)
    n_total = sum(len(r["outcomes"]) for r in parsed)
    print(f"results → {local_dir}")
    print(f"summary: {n_pass}/{n_total} probes passed")
