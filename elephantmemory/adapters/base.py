from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import (
    AdapterStats,
    ForgetResult,
    QueryResult,
    Session,
    WriteResult,
)


@runtime_checkable
class MemoryAdapter(Protocol):
    name: str

    def setup(self) -> None: ...
    def teardown(self) -> None: ...
    def reset_user(self, user_id: str) -> None: ...
    def record_session(self, session: Session) -> WriteResult: ...
    def query(self, user_id: str, prompt: str) -> QueryResult: ...
    def forget(self, user_id: str, predicate: str) -> ForgetResult: ...
    def stats(self, user_id: str) -> AdapterStats: ...


def build_adapter(name: str) -> MemoryAdapter:
    if name == "pgvector_diy":
        from .pgvector_diy import PgVectorDIY
        return PgVectorDIY()
    if name == "claude_memory":
        from .claude_memory import ClaudeMemoryAdapter
        return ClaudeMemoryAdapter()
    if name == "gpt_memory":
        from .gpt_memory import GPTMemoryAdapter
        return GPTMemoryAdapter()
    if name == "mem0":
        from .mem0_adapter import Mem0Adapter
        return Mem0Adapter()
    if name == "zep":
        from .zep_adapter import ZepAdapter
        return ZepAdapter()
    if name == "letta":
        from .letta_adapter import LettaAdapter
        return LettaAdapter()
    raise ValueError(f"unknown adapter: {name}")
