#!/usr/bin/env python3
"""Run the elephantmemory benchmark inside a fresh e2b sandbox.

Spins up an e2b sandbox, apt-installs Postgres + pgvector, uploads the
local working tree, installs Python deps, runs the benchmark for the
requested adapters, and downloads results back to results/runs/e2b_<id>/.

Lite adapter set (works in default e2b template):
  pgvector_diy, claude_memory, mem0

Heavy adapter set (needs a custom template with Neo4j / Letta installed):
  zep, letta

For the heavy set, build a template via the e2b CLI ahead of time and
pass --template <id>; this script just connects to it.

Requires:
  - E2B_API_KEY in environment or .env
  - ANTHROPIC_API_KEY + OPENAI_API_KEY in environment or .env
  - the repo to be a git checkout (uses `git archive` for the upload)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from e2b import Sandbox

REPO_ROOT = Path(__file__).resolve().parent.parent
LITE_ADAPTERS = {"pgvector_diy", "claude_memory", "mem0"}
HEAVY_ADAPTERS = {"zep", "letta"}
ALL_ADAPTERS = LITE_ADAPTERS | HEAVY_ADAPTERS


def bundle_repo() -> bytes:
    proc = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        check=True, capture_output=True, cwd=REPO_ROOT,
    )
    return proc.stdout


def run_in_sandbox(sbx: Sandbox, cmd: str, *, timeout: int = 120) -> str:
    print(f"$ {cmd}")
    result = sbx.commands.run(cmd, timeout=timeout)
    if result.exit_code != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed (exit {result.exit_code}): {cmd}")
    return result.stdout


def install_postgres(sbx: Sandbox) -> None:
    run_in_sandbox(sbx, "sudo apt-get update -qq", timeout=120)
    run_in_sandbox(
        sbx,
        "sudo apt-get install -y -qq postgresql postgresql-contrib "
        "postgresql-server-dev-all build-essential git",
        timeout=300,
    )
    # Build pgvector from source — most portable across e2b base images
    run_in_sandbox(
        sbx,
        "cd /tmp && git clone --depth 1 --branch v0.8.0 "
        "https://github.com/pgvector/pgvector.git && "
        "cd pgvector && sudo make && sudo make install",
        timeout=300,
    )
    run_in_sandbox(sbx, "sudo service postgresql start", timeout=60)
    run_in_sandbox(
        sbx,
        "sudo -u postgres psql -c \"CREATE USER elephant WITH PASSWORD 'elephant' SUPERUSER;\"",
        timeout=30,
    )
    run_in_sandbox(
        sbx, "sudo -u postgres psql -c 'CREATE DATABASE elephant OWNER elephant;'",
        timeout=30,
    )
    run_in_sandbox(
        sbx, "sudo -u postgres psql -d elephant -c 'CREATE EXTENSION vector;'",
        timeout=30,
    )


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", action="append", required=True)
    parser.add_argument("--scenarios", default="scenarios/")
    parser.add_argument("--template", default=None,
                        help="e2b template id; default uses base template")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--keep-alive", action="store_true",
                        help="don't kill the sandbox at end (debug)")
    args = parser.parse_args()

    for a in args.adapter:
        if a not in ALL_ADAPTERS:
            sys.exit(f"unknown adapter: {a}")
        if a in HEAVY_ADAPTERS and args.template is None:
            print(
                f"warning: {a} needs Neo4j/Letta which the default e2b "
                f"template doesn't include — pass --template <id> with a "
                f"custom template, or stick to the lite set", file=sys.stderr,
            )

    for var in ("E2B_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if not os.getenv(var):
            sys.exit(f"{var} not set; export it or add to .env")

    sbx = Sandbox.create(
        template=args.template,
        timeout=args.timeout,
        envs={
            "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "POSTGRES_DSN": "postgresql://elephant:elephant@localhost:5432/elephant",
        },
    )
    print(f"sandbox: {sbx.sandbox_id}  (host: {sbx.get_host(80)})")

    try:
        install_postgres(sbx)

        run_in_sandbox(sbx, "mkdir -p /home/user/repo")
        sbx.files.write("/tmp/repo.tar", bundle_repo())
        run_in_sandbox(sbx, "tar -xf /tmp/repo.tar -C /home/user/repo")

        extras = sorted(set(args.adapter) & {"mem0", "zep", "letta"})
        extras_str = "[" + ",".join(extras) + "]" if extras else ""
        run_in_sandbox(
            sbx, f"cd /home/user/repo && pip install -q -e '.{extras_str}'",
            timeout=600,
        )

        adapter_args = " ".join(f"--adapter {a}" for a in args.adapter)
        out = run_in_sandbox(
            sbx,
            f"cd /home/user/repo && elephantmemory run {adapter_args} "
            f"--scenarios {args.scenarios}",
            timeout=args.timeout,
        )
        print(out)

        run_id = run_in_sandbox(
            sbx, "ls -t /home/user/repo/results/runs/ | head -1"
        ).strip()
        local_dir = REPO_ROOT / "results" / "runs" / f"e2b_{run_id}"
        local_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("results.json",):
            content = sbx.files.read(f"/home/user/repo/results/runs/{run_id}/{fname}")
            (local_dir / fname).write_text(content)
        print(f"results → {local_dir}")
        return 0

    finally:
        if not args.keep_alive:
            sbx.kill()


if __name__ == "__main__":
    raise SystemExit(main())
