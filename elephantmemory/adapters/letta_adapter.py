"""Letta adapter — agent self-edits OS-style memory blocks.

Per-user agents (Atlas, Brooke). The agent autonomously decides when to
update its core memory blocks and when to push to archival. We do not
inject our own extraction step — testing the agent's own curation is the
whole point of this framework.

Latency will likely be the worst of any adapter (Letta makes multiple
internal LLM calls per turn). That is the honest measurement.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from letta_client import Letta

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

LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
LETTA_LLM_MODEL = os.getenv("LETTA_LLM_MODEL", f"anthropic/{llm.ASSISTANT_MODEL}")
LETTA_EMBED_MODEL = os.getenv("LETTA_EMBED_MODEL", f"openai/{llm.EMBED_MODEL}")
AGENT_NAME_PREFIX = "elephantmemory_"


def _agent_name(user_id: str) -> str:
    safe = "".join(c for c in user_id if c.isalnum() or c in "-_")
    return f"{AGENT_NAME_PREFIX}{safe}"


def _extract_assistant_text(response) -> str:
    parts: list[str] = []
    for m in response.messages:
        kind = getattr(m, "message_type", None)
        if kind == "assistant_message":
            content = getattr(m, "content", None)
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for blk in content:
                    text = getattr(blk, "text", None)
                    if text:
                        parts.append(text)
    return "\n".join(parts).strip()


def _usage_tokens(response) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    tin = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
    tout = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
    return int(tin), int(tout)


@dataclass
class LettaAdapter:
    name: str = "letta"

    def __post_init__(self) -> None:
        self._client: Letta | None = None
        self._agent_ids: dict[str, str] = {}

    def _c(self) -> Letta:
        if self._client is None:
            self._client = Letta(base_url=LETTA_BASE_URL)
        return self._client

    def setup(self) -> None:
        self._c()

    def teardown(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass

    def _get_or_create_agent(self, user_id: str) -> str:
        if user_id in self._agent_ids:
            return self._agent_ids[user_id]
        name = _agent_name(user_id)
        existing = self._c().agents.list(name=name)
        if existing:
            agent_id = existing[0].id
        else:
            agent = self._c().agents.create(
                name=name,
                model=LETTA_LLM_MODEL,
                embedding=LETTA_EMBED_MODEL,
                memory_blocks=[
                    {"label": "human", "value": f"User id: {user_id}. Details unknown — learn from conversation."},
                    {"label": "persona", "value": "You are a helpful assistant who remembers context across sessions."},
                ],
                include_base_tools=True,
            )
            agent_id = agent.id
        self._agent_ids[user_id] = agent_id
        return agent_id

    def reset_user(self, user_id: str) -> None:
        try:
            existing = self._c().agents.list(name=_agent_name(user_id))
            for a in existing:
                self._c().agents.delete(a.id)
        except Exception:
            pass
        self._agent_ids.pop(user_id, None)

    def record_session(self, session: Session) -> WriteResult:
        t0 = time.perf_counter()
        try:
            agent_id = self._get_or_create_agent(session.user_id)
            messages = [{"role": t.role, "content": t.content} for t in session.turns]
            r = self._c().agents.messages.create(agent_id=agent_id, messages=messages)
        except Exception as e:
            return WriteResult(False, (time.perf_counter() - t0) * 1000, note=f"error: {e}")
        tin, tout = _usage_tokens(r)
        return WriteResult(
            success=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=cost_usd(llm.ASSISTANT_MODEL, tin, tout),
            tokens_in=tin,
            tokens_out=tout,
        )

    def query(self, user_id: str, prompt: str) -> QueryResult:
        t0 = time.perf_counter()
        try:
            agent_id = self._get_or_create_agent(user_id)
            r = self._c().agents.messages.create(
                agent_id=agent_id,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            return QueryResult(
                response=f"[adapter error: {e}]",
                context=RetrievedContext(text=""),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        text = _extract_assistant_text(r) or "[no assistant message returned]"
        tin, tout = _usage_tokens(r)
        return QueryResult(
            response=text,
            context=RetrievedContext(text="(letta — memory managed in-agent)"),
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=cost_usd(llm.ASSISTANT_MODEL, tin, tout),
            tokens_in=tin,
            tokens_out=tout,
        )

    def forget(self, user_id: str, predicate: str) -> ForgetResult:
        t0 = time.perf_counter()
        try:
            agent_id = self._get_or_create_agent(user_id)
            matches = self._c().agents.passages.search(agent_id=agent_id, query=predicate, top_k=50)
        except Exception:
            return ForgetResult(0, (time.perf_counter() - t0) * 1000)
        items = matches if isinstance(matches, list) else getattr(matches, "results", [])
        removed = 0
        for it in items:
            pid = getattr(it, "id", None) or (it.get("id") if isinstance(it, dict) else None)
            if pid is None:
                continue
            try:
                self._c().agents.passages.delete(agent_id=agent_id, memory_id=pid)
                removed += 1
            except Exception:
                pass
        return ForgetResult(
            items_removed=removed,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def stats(self, user_id: str) -> AdapterStats:
        try:
            agent_id = self._get_or_create_agent(user_id)
            passages = self._c().agents.passages.list(agent_id=agent_id, limit=1000)
        except Exception:
            return AdapterStats()
        items = passages if isinstance(passages, list) else getattr(passages, "data", [])
        size = sum(len(getattr(p, "text", "") or "") for p in items)
        return AdapterStats(facts_stored=len(items), bytes_stored=size)
