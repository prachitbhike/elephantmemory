"""DIY baseline: messages + LLM-extracted facts in Postgres with pgvector.

Represents the "no framework" path most teams ship first. Embedding-based
retrieval over extracted facts; the assistant turn at query time is just
Claude with retrieved facts injected as system context.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

from .. import llm
from ..cost import cost_usd
from ..types import (
    AdapterStats,
    ForgetResult,
    QueryResult,
    RetrievedContext,
    Session,
    WriteResult,
)

DSN = os.getenv("POSTGRES_DSN", "postgresql://elephant:elephant@localhost:5433/elephant")

EXTRACT_SYSTEM = """You extract atomic, durable facts from a conversation.

Return JSON: {"facts": ["<fact 1>", "<fact 2>", ...]}

Rules:
- One claim per fact. No compound sentences.
- Include the subject (e.g., "Atlas's dog is named Pepper" not "her dog is named Pepper").
- Skip pleasantries, questions, and ephemeral state ("user is currently typing").
- Skip facts already obvious from common knowledge.
- If nothing durable, return {"facts": []}.
"""

ANSWER_SYSTEM = """You are a helpful assistant with access to remembered facts about the user.

Use only the facts provided in the system context to answer. If the facts do
not contain the answer, say so plainly — do not guess. Be concise."""

FORGET_SYSTEM = """You decide which stored facts match a deletion request.

Return JSON: {"ids": [<fact id 1>, ...]} listing every fact that should be
deleted to honor the request. Be thorough — derived/related facts count too.
If none match, return {"ids": []}."""


@dataclass
class PgVectorDIY:
    name: str = "pgvector_diy"
    dsn: str = DSN

    def __post_init__(self) -> None:
        self._conn: psycopg.Connection | None = None

    def _c(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, autocommit=True)
            register_vector(self._conn)
        return self._conn

    def setup(self) -> None:
        with self._c().cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id BIGSERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    embedding vector(1536) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS facts_user_idx ON facts (user_id)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS facts_embed_idx ON facts "
                "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            )

    def teardown(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    def reset_user(self, user_id: str) -> None:
        with self._c().cursor() as cur:
            cur.execute("DELETE FROM facts WHERE user_id=%s", (user_id,))

    def record_session(self, session: Session) -> WriteResult:
        t0 = time.perf_counter()
        transcript = "\n".join(f"{t.role}: {t.content}" for t in session.turns)
        ext = llm.chat(
            system=EXTRACT_SYSTEM,
            messages=[{"role": "user", "content": transcript}],
            max_tokens=1024,
        )
        try:
            facts = json.loads(ext.text).get("facts", [])
        except json.JSONDecodeError:
            facts = []
        cost = ext.cost_usd
        if facts:
            embeddings, embed_cost = llm.embed(facts)
            cost += embed_cost
            with self._c().cursor() as cur:
                cur.executemany(
                    "INSERT INTO facts (user_id, fact, embedding) VALUES (%s, %s, %s)",
                    [(session.user_id, f, e) for f, e in zip(facts, embeddings)],
                )
        return WriteResult(
            success=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=cost,
            tokens_in=ext.tokens_in,
            tokens_out=ext.tokens_out,
            note=f"extracted {len(facts)} facts",
        )

    def query(self, user_id: str, prompt: str) -> QueryResult:
        t0 = time.perf_counter()
        embeddings, embed_cost = llm.embed([prompt])
        with self._c().cursor() as cur:
            cur.execute(
                "SELECT fact FROM facts WHERE user_id=%s "
                "ORDER BY embedding <=> %s::vector LIMIT 8",
                (user_id, embeddings[0]),
            )
            facts = [r[0] for r in cur.fetchall()]
        ctx_text = "Known facts about the user:\n" + "\n".join(f"- {f}" for f in facts) if facts else "No facts known."
        ans = llm.chat(
            system=ANSWER_SYSTEM + "\n\n" + ctx_text,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return QueryResult(
            response=ans.text,
            context=RetrievedContext(text=ctx_text, facts=facts),
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=embed_cost + ans.cost_usd,
            tokens_in=ans.tokens_in,
            tokens_out=ans.tokens_out,
        )

    def forget(self, user_id: str, predicate: str) -> ForgetResult:
        t0 = time.perf_counter()
        with self._c().cursor() as cur:
            cur.execute("SELECT id, fact FROM facts WHERE user_id=%s", (user_id,))
            rows = cur.fetchall()
        if not rows:
            return ForgetResult(0, (time.perf_counter() - t0) * 1000, 0.0)

        listing = "\n".join(f"{rid}: {f}" for rid, f in rows)
        out = llm.chat(
            system=FORGET_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": f"Deletion request: {predicate}\n\nStored facts:\n{listing}",
                }
            ],
            max_tokens=512,
        )
        try:
            ids = json.loads(out.text).get("ids", [])
        except json.JSONDecodeError:
            ids = []
        if ids:
            with self._c().cursor() as cur:
                cur.execute("DELETE FROM facts WHERE user_id=%s AND id = ANY(%s)", (user_id, ids))
        return ForgetResult(
            items_removed=len(ids),
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=cost_usd(out.model, out.tokens_in, out.tokens_out),
        )

    def stats(self, user_id: str) -> AdapterStats:
        with self._c().cursor() as cur:
            cur.execute(
                "SELECT count(*), coalesce(sum(length(fact)), 0) "
                "FROM facts WHERE user_id=%s",
                (user_id,),
            )
            n, b = cur.fetchone()
        return AdapterStats(facts_stored=int(n), bytes_stored=int(b))
