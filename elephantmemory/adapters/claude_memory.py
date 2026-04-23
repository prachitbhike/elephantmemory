"""Anthropic memory tool adapter — agent curates a per-user filesystem.

Uses the SDK's `BetaLocalFilesystemMemoryTool` scoped per user. Recording a
session = run a tool-loop with the transcript so Claude can update its
/memories. Querying = run another tool-loop with the probe; Claude views
memory then answers. Forget = same loop, asking Claude to delete files
matching the predicate (this is honest — there's no schema, the model is
the deletion engine).
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from anthropic import Anthropic
from anthropic.tools import BetaLocalFilesystemMemoryTool

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

MEMORY_ROOT = Path(os.getenv("CLAUDE_MEMORY_ROOT", "/tmp/elephantmemory/claude_memory"))

RECORD_SYSTEM = """You are recording context from a conversation that just took place.

Read your memory directory first. Then update it with any new durable facts,
preferences, or context from this conversation. Reorganize, rename, or
delete files as needed to keep memory coherent. Skip ephemera (greetings,
in-progress tasks). Do not write down sensitive financial figures or
medical details unless the user has clearly asked you to remember them.

Reply with a brief summary of what you updated. Do not echo the conversation."""

QUERY_SYSTEM = """You are answering a question using your persistent memory.

Always view your memory directory first. Read whatever files look relevant.
Then answer concisely from what you find — if memory does not contain the
answer, say so plainly. Do not guess or invent facts."""

FORGET_SYSTEM = """You have received a deletion request. View your memory
directory, identify every file (or section within a file) that matches the
request, and remove it. Use delete for whole files and str_replace to
strip out sections of files that contain mixed content. Be thorough —
derived or related facts must go too. Reply with a one-line summary of
what you removed."""


def _safe_user_dir(user_id: str) -> Path:
    safe = "".join(c for c in user_id if c.isalnum() or c in "-_")
    return MEMORY_ROOT / safe


@dataclass
class ClaudeMemoryAdapter:
    name: str = "claude_memory"

    def __post_init__(self) -> None:
        self._client = Anthropic()

    def setup(self) -> None:
        MEMORY_ROOT.mkdir(parents=True, exist_ok=True)

    def teardown(self) -> None:
        pass

    def reset_user(self, user_id: str) -> None:
        d = _safe_user_dir(user_id)
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    def _tool(self, user_id: str) -> BetaLocalFilesystemMemoryTool:
        d = _safe_user_dir(user_id)
        d.mkdir(parents=True, exist_ok=True)
        return BetaLocalFilesystemMemoryTool(base_path=str(d))

    def _run_loop(
        self, user_id: str, system: str, user_msg: str, *, max_tokens: int = 1024
    ) -> tuple[str, int, int]:
        runner = self._client.beta.messages.tool_runner(
            model=llm.ASSISTANT_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
            tools=[self._tool(user_id)],
        )
        tin = tout = 0
        text_parts: list[str] = []
        last_text: list[str] = []
        for msg in runner:
            tin += msg.usage.input_tokens
            tout += msg.usage.output_tokens
            this_text: list[str] = []
            for block in msg.content:
                if block.type == "text":
                    this_text.append(block.text)
            if this_text:
                last_text = this_text
                text_parts.extend(this_text)
        response = "\n".join(last_text) if last_text else "\n".join(text_parts)
        return response, tin, tout

    def record_session(self, session: Session) -> WriteResult:
        t0 = time.perf_counter()
        transcript = "\n".join(f"{t.role}: {t.content}" for t in session.turns)
        msg = (
            f"Session timestamp: {session.timestamp.isoformat()}\n"
            f"User: {session.user_id}\n\n"
            f"Conversation transcript:\n{transcript}"
        )
        try:
            _, tin, tout = self._run_loop(session.user_id, RECORD_SYSTEM, msg, max_tokens=1024)
        except Exception as e:
            return WriteResult(False, (time.perf_counter() - t0) * 1000, note=f"error: {e}")
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
            response, tin, tout = self._run_loop(
                user_id, QUERY_SYSTEM, prompt, max_tokens=1024
            )
        except Exception as e:
            return QueryResult(
                response=f"[adapter error: {e}]",
                context=RetrievedContext(text=""),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        return QueryResult(
            response=response,
            context=RetrievedContext(text="(memory tool — context inlined via tool calls)"),
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=cost_usd(llm.ASSISTANT_MODEL, tin, tout),
            tokens_in=tin,
            tokens_out=tout,
        )

    def forget(self, user_id: str, predicate: str) -> ForgetResult:
        t0 = time.perf_counter()
        before = self.stats(user_id).facts_stored
        msg = f"Deletion request: {predicate}"
        try:
            _, tin, tout = self._run_loop(user_id, FORGET_SYSTEM, msg, max_tokens=1024)
        except Exception:
            tin = tout = 0
        after = self.stats(user_id).facts_stored
        return ForgetResult(
            items_removed=max(0, before - after),
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=cost_usd(llm.ASSISTANT_MODEL, tin, tout),
        )

    def stats(self, user_id: str) -> AdapterStats:
        d = _safe_user_dir(user_id)
        if not d.exists():
            return AdapterStats()
        files = [f for f in d.rglob("*") if f.is_file()]
        size = sum(f.stat().st_size for f in files)
        return AdapterStats(facts_stored=len(files), bytes_stored=size)
