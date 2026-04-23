"""GPT counterpart to the Anthropic memory tool adapter.

Mirrors claude_memory.py so the paradigm — agent-curated filesystem — is
the same; only the provider + tool-loop mechanics change. The tool schema
below matches Anthropic's beta memory tool (view / create / str_replace /
insert / delete, paths rooted at /memories) so the model-visible contract
is identical across adapters.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

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

MEMORY_ROOT = Path(os.getenv("GPT_MEMORY_ROOT", "/tmp/elephantmemory/gpt_memory"))
GPT_MODEL = os.getenv("ELEPHANT_OPENAI_ASSISTANT_MODEL", "gpt-5")

RECORD_SYSTEM = """You are recording context from a conversation that just took place.

Read your memory directory first (it lives at /memories). Then update it
with any new durable facts, preferences, or context from this conversation.
Reorganize, rename, or delete files as needed to keep memory coherent. Skip
ephemera (greetings, in-progress tasks). Do not write down sensitive
financial figures or medical details unless the user has clearly asked you
to remember them.

Reply with a brief summary of what you updated. Do not echo the conversation."""

QUERY_SYSTEM = """You are answering a question using your persistent memory at /memories.

Always view /memories first. Read whatever files look relevant. Then answer
concisely from what you find — if memory does not contain the answer, say
so plainly. Do not guess or invent facts."""

FORGET_SYSTEM = """You have received a deletion request. View /memories,
identify every file (or section within a file) that matches the request,
and remove it. Use the delete tool for whole files and str_replace to strip
out sections of files that contain mixed content. Be thorough — derived or
related facts must go too. Reply with a one-line summary of what you removed."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "view",
            "description": "View a file's contents, or list a directory. Path must start with /memories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional [start, end] 1-indexed line range.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create",
            "description": "Create a new file at `path` with the given text. Overwrites if it exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "file_text": {"type": "string"},
                },
                "required": ["path", "file_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace",
            "description": "Replace the first occurrence of `old_str` with `new_str` inside file at `path`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert",
            "description": "Insert `insert_text` at 1-indexed line `insert_line` in file at `path`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "insert_line": {"type": "integer"},
                    "insert_text": {"type": "string"},
                },
                "required": ["path", "insert_line", "insert_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete",
            "description": "Delete a file (or an empty directory) at `path`.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


def _safe_user_dir(user_id: str) -> Path:
    safe = "".join(c for c in user_id if c.isalnum() or c in "-_")
    return MEMORY_ROOT / safe


def _resolve(base: Path, path: str) -> Path:
    if not path.startswith("/memories"):
        raise ValueError("path must start with /memories")
    rel = path[len("/memories"):].lstrip("/")
    full = (base / rel).resolve()
    base_resolved = base.resolve()
    if base_resolved != full and base_resolved not in full.parents:
        raise ValueError("path escapes /memories")
    return full


def _run_tool(base: Path, name: str, args: dict) -> str:
    try:
        if name == "view":
            target = _resolve(base, args["path"])
            if target.is_dir():
                entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
                return f"Directory {args['path']}:\n" + "\n".join(entries) if entries else f"Directory {args['path']} is empty."
            if not target.exists():
                return f"File {args['path']} does not exist."
            text = target.read_text()
            vr = args.get("view_range")
            if vr and len(vr) == 2:
                lines = text.splitlines()
                start, end = max(1, vr[0]) - 1, min(len(lines), vr[1])
                return "\n".join(lines[start:end])
            return text
        if name == "create":
            target = _resolve(base, args["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(args["file_text"])
            return f"Wrote {args['path']}."
        if name == "str_replace":
            target = _resolve(base, args["path"])
            text = target.read_text()
            old = args["old_str"]
            if old not in text:
                return f"str_replace: `old_str` not found in {args['path']}."
            target.write_text(text.replace(old, args["new_str"], 1))
            return f"Replaced 1 occurrence in {args['path']}."
        if name == "insert":
            target = _resolve(base, args["path"])
            lines = target.read_text().splitlines()
            idx = max(0, min(len(lines), args["insert_line"] - 1))
            lines.insert(idx, args["insert_text"])
            target.write_text("\n".join(lines))
            return f"Inserted into {args['path']} at line {args['insert_line']}."
        if name == "delete":
            target = _resolve(base, args["path"])
            if target.is_dir():
                target.rmdir()
            else:
                target.unlink(missing_ok=True)
            return f"Deleted {args['path']}."
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error ({name}): {e}"


@dataclass
class GPTMemoryAdapter:
    name: str = "gpt_memory"

    def __post_init__(self) -> None:
        self._client = llm.openai_client()

    def setup(self) -> None:
        MEMORY_ROOT.mkdir(parents=True, exist_ok=True)

    def teardown(self) -> None:
        pass

    def reset_user(self, user_id: str) -> None:
        d = _safe_user_dir(user_id)
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    def _run_loop(
        self, user_id: str, system: str, user_msg: str, *, max_iters: int = 20
    ) -> tuple[str, int, int]:
        base = _safe_user_dir(user_id)
        base.mkdir(parents=True, exist_ok=True)
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        tin = tout = 0
        for _ in range(max_iters):
            resp = self._client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                tools=TOOLS,
            )
            usage = resp.usage
            tin += usage.prompt_tokens
            tout += usage.completion_tokens
            msg = resp.choices[0].message
            tool_calls = msg.tool_calls or []
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ] if tool_calls else None,
            })
            if not tool_calls:
                return (msg.content or ""), tin, tout
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = _run_tool(base, tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        return "[gpt_memory: exceeded tool-loop iterations]", tin, tout

    def record_session(self, session: Session) -> WriteResult:
        t0 = time.perf_counter()
        transcript = "\n".join(f"{t.role}: {t.content}" for t in session.turns)
        msg = (
            f"Session timestamp: {session.timestamp.isoformat()}\n"
            f"User: {session.user_id}\n\n"
            f"Conversation transcript:\n{transcript}"
        )
        try:
            _, tin, tout = self._run_loop(session.user_id, RECORD_SYSTEM, msg)
        except Exception as e:
            return WriteResult(False, (time.perf_counter() - t0) * 1000, note=f"error: {e}")
        return WriteResult(
            success=True,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=cost_usd(GPT_MODEL, tin, tout),
            tokens_in=tin,
            tokens_out=tout,
        )

    def query(self, user_id: str, prompt: str) -> QueryResult:
        t0 = time.perf_counter()
        try:
            response, tin, tout = self._run_loop(user_id, QUERY_SYSTEM, prompt)
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
            cost_usd=cost_usd(GPT_MODEL, tin, tout),
            tokens_in=tin,
            tokens_out=tout,
        )

    def forget(self, user_id: str, predicate: str) -> ForgetResult:
        t0 = time.perf_counter()
        before = self.stats(user_id).facts_stored
        msg = f"Deletion request: {predicate}"
        try:
            _, tin, tout = self._run_loop(user_id, FORGET_SYSTEM, msg)
        except Exception:
            tin = tout = 0
        after = self.stats(user_id).facts_stored
        return ForgetResult(
            items_removed=max(0, before - after),
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=cost_usd(GPT_MODEL, tin, tout),
        )

    def stats(self, user_id: str) -> AdapterStats:
        d = _safe_user_dir(user_id)
        if not d.exists():
            return AdapterStats()
        files = [f for f in d.rglob("*") if f.is_file()]
        size = sum(f.stat().st_size for f in files)
        return AdapterStats(facts_stored=len(files), bytes_stored=size)
