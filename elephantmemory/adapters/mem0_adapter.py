"""mem0 adapter — vector-extracted facts (optionally with graph layer).

STATUS: skeleton. Wire up in a follow-up commit.

Implementation notes:
- `from mem0 import Memory` then `Memory.from_config({...})`.
- Configure with same OpenAI embedding model + Postgres/pgvector backend as
  the DIY adapter so the only variable is mem0's extraction + dedup pipeline.
- record_session → m.add(messages, user_id=...).
- query → m.search(query, user_id=..., limit=8); inject results into Claude
  call (same ANSWER_SYSTEM as pgvector_diy for fairness).
- forget → m.search(predicate) → m.delete(memory_id) for each match.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..types import (
    AdapterStats,
    ForgetResult,
    QueryResult,
    Session,
    WriteResult,
)


@dataclass
class Mem0Adapter:
    name: str = "mem0"

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def reset_user(self, user_id: str) -> None:
        pass

    def record_session(self, session: Session) -> WriteResult:
        raise NotImplementedError("mem0 adapter pending — see module docstring")

    def query(self, user_id: str, prompt: str) -> QueryResult:
        raise NotImplementedError("mem0 adapter pending — see module docstring")

    def forget(self, user_id: str, predicate: str) -> ForgetResult:
        raise NotImplementedError("mem0 adapter pending — see module docstring")

    def stats(self, user_id: str) -> AdapterStats:
        return AdapterStats()
