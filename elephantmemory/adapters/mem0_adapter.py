"""mem0 adapter — vector-extracted facts.

Configured to share the same Postgres+pgvector backend as pgvector_diy and the
same OpenAI embedding model. The variables under test are mem0's own LLM-based
extraction, dedup, and update pipeline — not the storage layer.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from mem0 import Memory

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


def _pg_kwargs() -> dict:
    import urllib.parse as up
    p = up.urlparse(DSN)
    return {
        "dbname": (p.path or "/elephant").lstrip("/"),
        "user": p.username,
        "password": p.password,
        "host": p.hostname,
        "port": p.port or 5432,
    }


def _make_config() -> dict:
    pg = _pg_kwargs()
    return {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                **pg,
                "collection_name": "mem0_elephant",
                "embedding_model_dims": 1536,
                "diskann": False,
                "hnsw": True,
            },
        },
        "llm": {
            "provider": "anthropic",
            "config": {"model": llm.ASSISTANT_MODEL, "max_tokens": 1024},
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": llm.EMBED_MODEL},
        },
    }


ANSWER_SYSTEM = """You are a helpful assistant with access to remembered facts about the user.

Use only the facts provided in the system context to answer. If the facts do
not contain the answer, say so plainly — do not guess. Be concise."""


@dataclass
class Mem0Adapter:
    name: str = "mem0"

    def __post_init__(self) -> None:
        self._mem: Memory | None = None

    def _m(self) -> Memory:
        if self._mem is None:
            self._mem = Memory.from_config(_make_config())
        return self._mem

    def setup(self) -> None:
        self._m()

    def teardown(self) -> None:
        if self._mem is not None:
            try:
                self._mem.close()
            except Exception:
                pass

    def reset_user(self, user_id: str) -> None:
        try:
            self._m().delete_all(user_id=user_id)
        except Exception:
            pass

    def record_session(self, session: Session) -> WriteResult:
        t0 = time.perf_counter()
        msgs = [{"role": t.role, "content": t.content} for t in session.turns]
        try:
            res = self._m().add(msgs, user_id=session.user_id, infer=True)
        except Exception as e:
            return WriteResult(False, (time.perf_counter() - t0) * 1000, note=f"error: {e}")
        n_results = len(res.get("results", [])) if isinstance(res, dict) else 0
        return WriteResult(
            success=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=0.0,
            note=f"mem0 returned {n_results} ops",
        )

    def query(self, user_id: str, prompt: str) -> QueryResult:
        t0 = time.perf_counter()
        try:
            search_res = self._m().search(prompt, user_id=user_id, limit=8)
        except Exception as e:
            return QueryResult(
                response=f"[adapter error: {e}]",
                context=RetrievedContext(text=""),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        items = search_res.get("results", []) if isinstance(search_res, dict) else []
        facts = [it.get("memory", "") for it in items]
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
            res = self._m().search(predicate, user_id=user_id, limit=50)
        except Exception:
            return ForgetResult(0, (time.perf_counter() - t0) * 1000)
        items = res.get("results", []) if isinstance(res, dict) else []
        removed = 0
        for it in items:
            try:
                self._m().delete(it["id"])
                removed += 1
            except Exception:
                pass
        return ForgetResult(
            items_removed=removed,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=0.0,
        )

    def stats(self, user_id: str) -> AdapterStats:
        try:
            res = self._m().get_all(user_id=user_id, limit=1000)
        except Exception:
            return AdapterStats()
        items = res.get("results", []) if isinstance(res, dict) else []
        size = sum(len(it.get("memory", "")) for it in items)
        return AdapterStats(facts_stored=len(items), bytes_stored=size)
