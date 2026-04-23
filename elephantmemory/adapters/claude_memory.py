"""Anthropic memory tool adapter — agent curates a per-user filesystem.

Uses tool name `memory_20250818`. The adapter implements the BetaAbstractMemoryTool
contract (view/create/str_replace/insert/delete/rename) backed by a local
directory at MEMORY_ROOT/<user_id>/. Recording a session = run an agent loop
over the conversation so Claude can choose what to write to /memories.
Querying = run a separate agent loop that reads /memories before answering.

STATUS: skeleton. Wire up in a follow-up commit.

Implementation notes:
- Use anthropic.Anthropic().beta.messages.create with tool=[{"type":"memory_20250818"}]
- Implement BetaAbstractMemoryTool.{view,create,str_replace,insert,delete,rename}
  against pathlib under MEMORY_ROOT/<user_id>/.
- record_session: run loop with the session transcript as a single user
  message, instructing Claude to update /memories as it sees fit.
- query: separate loop, system prompt nudges Claude to view /memories first.
- forget: scripted file deletion (no LLM needed for the predicate match — this
  framework relies on the agent for curation, so a programmatic deletion
  matches what a developer would actually do under a GDPR request).
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from ..types import (
    AdapterStats,
    ForgetResult,
    QueryResult,
    RetrievedContext,
    Session,
    WriteResult,
)

MEMORY_ROOT = Path(os.getenv("CLAUDE_MEMORY_ROOT", "/tmp/elephantmemory/claude_memory"))


@dataclass
class ClaudeMemoryAdapter:
    name: str = "claude_memory"

    def setup(self) -> None:
        MEMORY_ROOT.mkdir(parents=True, exist_ok=True)

    def teardown(self) -> None:
        pass

    def reset_user(self, user_id: str) -> None:
        path = MEMORY_ROOT / user_id
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    def record_session(self, session: Session) -> WriteResult:
        t0 = time.perf_counter()
        raise NotImplementedError("claude_memory adapter pending — see module docstring")

    def query(self, user_id: str, prompt: str) -> QueryResult:
        raise NotImplementedError("claude_memory adapter pending — see module docstring")

    def forget(self, user_id: str, predicate: str) -> ForgetResult:
        raise NotImplementedError("claude_memory adapter pending — see module docstring")

    def stats(self, user_id: str) -> AdapterStats:
        path = MEMORY_ROOT / user_id
        if not path.exists():
            return AdapterStats()
        files = list(path.rglob("*"))
        size = sum(f.stat().st_size for f in files if f.is_file())
        return AdapterStats(facts_stored=len([f for f in files if f.is_file()]), bytes_stored=size)
