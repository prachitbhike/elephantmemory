# elephantmemory

A benchmark harness comparing how LLM memory frameworks behave on the failure modes developers actually hit in production: stale-fact handling, multi-tenant isolation, deletion compliance, write cost, and concurrency — not just the read-only QA accuracy that existing benchmarks (LoCoMo, LongMemEval) measure.

## What it tests

A simulated personal assistant ("Atlas") and a second user ("Brooke") talk to a memory-backed agent across many sessions covering travel, health, work, and personal life. Each scenario probes one of eight failure modes:

| Category | What it probes |
|---|---|
| recall | Sanity baseline — facts stated once, queried later |
| temporal_supersede | New fact replaces old fact; query must return the new one |
| implicit_update | Fact derived from behavior, not stated |
| cross_session_reasoning | Answer joins facts from sessions far apart |
| abstention | Was something *never* said? Hallucination check |
| forget | Erasure request; future queries must fail |
| isolation | User B asks about User A's facts. Must fail. |
| concurrency | Parallel writers to same fact. Final state predictable? |

## Frameworks (Tier 1)

| Adapter | Status | Memory model |
|---|---|---|
| `pgvector_diy` | ✅ functional | Vanilla embeddings + LLM-extracted facts in Postgres. The "no framework" baseline. |
| `claude_memory` | ✅ functional | Anthropic memory tool — agent curates a filesystem via `BetaLocalFilesystemMemoryTool` |
| `mem0` | ✅ functional | Vector-extracted facts; configured to share the pgvector backend with the DIY baseline |
| `zep` | ✅ functional | Bitemporal knowledge graph via graphiti-core OSS against Neo4j |
| `letta` | ✅ functional | Agent self-edits OS-style memory blocks; one agent per user against self-hosted Letta server |

Tier 2 (LangMem, LlamaIndex, Cognee, Mastra, AWS AgentCore) deferred until Tier 1 results are stable.

## Decisions

- **Assistant + judge model**: Claude Sonnet 4.6 for both. Same family avoids provider home-field bias.
- **Embeddings**: `text-embedding-3-small` (OpenAI). Most adapters default to OpenAI embeddings; using one model across all adapters keeps retrieval comparable.
- **Hosting**: self-host every framework. No Mem0 Platform / Zep Cloud — managed perf is uneven and paywalled.
- **LLM calls**: real API calls for the assistant turn (honest latency/cost). Judge calls cached by `(probe_id, response_hash)`.
- **Scenarios**: hand-authored to fit the Atlas application. Not seeded from LongMemEval — the gaps in existing benchmarks are exactly what we're trying to fill.
- **Two users**: Atlas + Brooke. Required for isolation tests.

## Quickstart

```bash
cp .env.example .env  # fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
docker compose up -d postgres
pip install -e ".[dev]"
elephantmemory run --adapter pgvector_diy --scenarios scenarios/
elephantmemory report results/runs/<run_id>
```

To enable additional adapters:
```bash
pip install -e ".[mem0]"     # mem0
pip install -e ".[zep]"      # graphiti-core
pip install -e ".[letta]"    # letta-client; also: docker compose up -d letta
```

## Layout

```
elephantmemory/
  types.py            # Scenario, Session, Probe, WriteResult, QueryResult
  llm.py              # Anthropic + OpenAI clients with cost tracking
  cost.py             # Token → $ pricing tables
  scoring.py          # exact / contains / must_not_contain
  judge.py            # LLM-as-judge with sqlite cache
  runner.py           # Replays scenario events, captures metrics
  scenarios.py        # YAML loader
  cli.py              # `elephantmemory run|report`
  adapters/
    base.py           # MemoryAdapter Protocol
    pgvector_diy.py
    claude_memory.py  # skeleton
    mem0_adapter.py   # skeleton
    zep_adapter.py    # skeleton
    letta_adapter.py  # skeleton
scenarios/            # YAML scenarios (one per file)
results/
  runs/               # raw per-run output (gitignored)
  snapshots/          # canonical results checked in
```

## Adding a new adapter

Implement `MemoryAdapter` from [`elephantmemory/adapters/base.py`](elephantmemory/adapters/base.py). The runner calls `setup` once, then for each scenario event in timestamp order calls one of `record_session`, `query`, or `forget`, then `stats` at the end.

Keep adapter code thin — push framework-specific config into `__init__` kwargs so the runner can stay framework-agnostic.

## Running in an e2b sandbox (no local docker)

If you don't want to run Postgres locally, you can run the lite adapter
set (`pgvector_diy`, `claude_memory`, `mem0`) inside a fresh
[e2b](https://e2b.dev) sandbox:

```bash
pip install -e ".[mem0,e2b]"
# add E2B_API_KEY + ANTHROPIC_API_KEY + OPENAI_API_KEY to .env
python scripts/run_in_e2b.py --adapter pgvector_diy --adapter mem0
```

The script `git archive`s the working tree, uploads it to a fresh
sandbox, apt-installs Postgres, builds pgvector from source, runs the
benchmark, and downloads `results.json` back to
`results/runs/e2b_<run_id>/`.

`zep` (needs Neo4j) and `letta` (needs the Letta server) require a
custom e2b template — build one with the e2b CLI and pass `--template
<id>`. The default sandbox doesn't have the RAM headroom for both.

## Running multiple adapters at once

```bash
docker compose up -d postgres neo4j letta
elephantmemory run \
  --adapter pgvector_diy \
  --adapter claude_memory \
  --adapter mem0 \
  --adapter zep \
  --adapter letta \
  --scenarios scenarios/
```

Estimated cost for the 8-scenario starter set across all 5 adapters at
Sonnet 4.6 prices: ~$1–3 per full run, depending on how chatty Letta gets
internally and how many extraction calls mem0/Zep make. Add the dependencies
you actually want with `pip install -e ".[mem0,zep,letta]"`.

## Status

v0.1 — all five Tier 1 adapters functional. Awaiting first canonical
results snapshot (pending postgres + neo4j + letta containers running with
real API keys); will land under `results/snapshots/` and be referenced
from the report.
