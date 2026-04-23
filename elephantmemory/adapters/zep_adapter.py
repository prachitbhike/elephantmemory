"""Zep / Graphiti adapter — bitemporal knowledge graph.

Uses graphiti-core OSS against Neo4j. Zep Cloud is excluded per the self-host
fairness policy. Group_id = user_id provides per-user isolation; the
bitemporal model means contradicting facts get marked invalid_at rather
than overwritten — relevant to the temporal_supersede scenario.

Graphiti is async; we wrap each call in a per-instance event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass

from graphiti_core import Graphiti
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType

from .. import llm

logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# Graphiti's extraction pipeline depends on reliable structured-output / function-
# calling, which is significantly more stable on OpenAI than on Anthropic in
# graphiti-core. Using AnthropicClient produced ~0 entities per episode in
# benchmark runs (writes succeed, no nodes/edges are created → searches return
# empty → assistant says "I have no info"). We use a small OpenAI model just for
# graphiti's extraction; the final answer turn in `query()` still uses Claude
# via `llm.chat` to keep parity with the other adapters.
GRAPHITI_LLM_MODEL = os.getenv("GRAPHITI_LLM_MODEL", "gpt-4o-mini")
from ..types import (
    AdapterStats,
    ForgetResult,
    QueryResult,
    RetrievedContext,
    Session,
    WriteResult,
)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "elephantmemory")

ANSWER_SYSTEM = """You are a helpful assistant with access to a temporal knowledge graph
about the user. Each retrieved edge is a (subject, predicate, object) fact
with optional valid_at / invalid_at timestamps.

Use only the facts in the system context. Prefer facts that are currently
valid (no invalid_at, or invalid_at in the future). If the answer is not
present, say so plainly. Be concise."""


def _fact_line(edge) -> str:
    parts = [edge.fact or edge.name]
    if edge.valid_at:
        parts.append(f"(valid from {edge.valid_at.date()})")
    if edge.invalid_at:
        parts.append(f"(invalid from {edge.invalid_at.date()})")
    return " ".join(parts)


@dataclass
class ZepAdapter:
    name: str = "zep"

    def __post_init__(self) -> None:
        self._g: Graphiti | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._initialized = False

    def _l(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _graphiti(self) -> Graphiti:
        if self._g is None:
            llm_client = OpenAIClient(
                config=LLMConfig(model=GRAPHITI_LLM_MODEL, max_tokens=2048, temperature=0.0)
            )
            embedder = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(embedding_model=llm.EMBED_MODEL, embedding_dim=1536)
            )
            self._g = Graphiti(
                uri=NEO4J_URI,
                user=NEO4J_USER,
                password=NEO4J_PASSWORD,
                llm_client=llm_client,
                embedder=embedder,
            )
        return self._g

    def setup(self) -> None:
        g = self._graphiti()
        if not self._initialized:
            self._l().run_until_complete(g.build_indices_and_constraints())
            # Smoke-test the search path against a sentinel group_id so a broken
            # graphiti API surfaces here instead of producing a run of empty
            # retrievals.
            self._l().run_until_complete(
                g.search(query="ping", group_ids=["__elephantmemory_smoke__"], num_results=1)
            )
            self._initialized = True

    def teardown(self) -> None:
        if self._g is not None and self._loop is not None:
            try:
                self._loop.run_until_complete(self._g.close())
            except Exception:
                pass
            self._loop.close()

    def reset_user(self, user_id: str) -> None:
        async def _reset() -> None:
            driver = self._graphiti().driver
            cypher = (
                "MATCH (n {group_id: $gid}) DETACH DELETE n"
            )
            try:
                await driver.execute_query(cypher, gid=user_id)
            except Exception:
                pass

        self._l().run_until_complete(_reset())

    def record_session(self, session: Session) -> WriteResult:
        t0 = time.perf_counter()
        body = "\n".join(f"{t.role}: {t.content}" for t in session.turns)
        try:
            result = self._l().run_until_complete(
                self._graphiti().add_episode(
                    name=session.session_id,
                    episode_body=body,
                    source=EpisodeType.message,
                    source_description="conversation transcript",
                    reference_time=session.timestamp,
                    group_id=session.user_id,
                )
            )
        except Exception as e:
            return WriteResult(False, (time.perf_counter() - t0) * 1000, note=f"error: {e}")
        # Surface the silent-extraction failure mode: a successful add_episode
        # that produced no nodes/edges means the LLM extraction collapsed.
        n_nodes = len(getattr(result, "nodes", []) or [])
        n_edges = len(getattr(result, "edges", []) or [])
        return WriteResult(
            success=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=0.0,
            note=f"graphiti extracted nodes={n_nodes} edges={n_edges}",
        )

    def query(self, user_id: str, prompt: str) -> QueryResult:
        t0 = time.perf_counter()
        try:
            edges = self._l().run_until_complete(
                self._graphiti().search(query=prompt, group_ids=[user_id], num_results=12)
            )
        except Exception as e:
            return QueryResult(
                response=f"[adapter error: {e}]",
                context=RetrievedContext(text=""),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        facts = [_fact_line(e) for e in edges]
        ctx_text = (
            "Known facts about the user:\n" + "\n".join(f"- {f}" for f in facts)
            if facts
            else "No facts known."
        )
        ans = llm.chat(
            system=ANSWER_SYSTEM + "\n\n" + ctx_text,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return QueryResult(
            response=ans.text,
            context=RetrievedContext(text=ctx_text, facts=facts),
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=ans.cost_usd,
            tokens_in=ans.tokens_in,
            tokens_out=ans.tokens_out,
        )

    def forget(self, user_id: str, predicate: str) -> ForgetResult:
        t0 = time.perf_counter()
        try:
            edges = self._l().run_until_complete(
                self._graphiti().search(query=predicate, group_ids=[user_id], num_results=50)
            )
        except Exception:
            return ForgetResult(0, (time.perf_counter() - t0) * 1000)

        episode_uuids: set[str] = set()
        for e in edges:
            for ep in (e.episodes or []):
                episode_uuids.add(ep)

        removed = 0
        for ep_uuid in episode_uuids:
            try:
                self._l().run_until_complete(self._graphiti().remove_episode(ep_uuid))
                removed += 1
            except Exception:
                pass
        return ForgetResult(
            items_removed=removed,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def stats(self, user_id: str) -> AdapterStats:
        async def _count() -> tuple[int, int]:
            driver = self._graphiti().driver
            try:
                result = await driver.execute_query(
                    "MATCH (n:Entity {group_id: $gid}) RETURN count(n) AS c",
                    gid=user_id,
                )
                rows = result[0] if isinstance(result, tuple) else result.records
                n = rows[0]["c"] if rows else 0
                return int(n), 0
            except Exception:
                return 0, 0

        try:
            n, b = self._l().run_until_complete(_count())
        except Exception:
            n, b = 0, 0
        return AdapterStats(facts_stored=n, bytes_stored=b)
