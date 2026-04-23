"""Letta (formerly MemGPT) adapter — agent self-edits OS-style memory blocks.

STATUS: skeleton. Wire up in a follow-up commit.

Implementation notes:
- Use letta-client against the self-hosted server (docker-compose service `letta`).
- One agent per user (Atlas, Brooke). Provision in setup with
  client.agents.create(name=user_id, model="claude-sonnet-4-6", ...).
- record_session → for each turn in the session, send to the agent and let it
  call core_memory_replace / archival_memory_insert as it sees fit. We do NOT
  inject our own extraction step — testing the agent's own curation is the
  whole point of this framework.
- query → client.agents.messages.send(agent_id, prompt). Capture the assistant
  reply directly; Letta's response is the answer.
- forget → script archival_memory_search + delete; or use a "you must delete
  facts about X" instruction and verify behaviorally. Both worth testing.
- Cross-user isolation should be trivially correct here (separate agents).
- Latency will likely be the worst here — Letta makes multiple internal LLM
  calls per turn. That's the honest measurement.
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
class LettaAdapter:
    name: str = "letta"

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def reset_user(self, user_id: str) -> None:
        pass

    def record_session(self, session: Session) -> WriteResult:
        raise NotImplementedError("letta adapter pending — see module docstring")

    def query(self, user_id: str, prompt: str) -> QueryResult:
        raise NotImplementedError("letta adapter pending — see module docstring")

    def forget(self, user_id: str, predicate: str) -> ForgetResult:
        raise NotImplementedError("letta adapter pending — see module docstring")

    def stats(self, user_id: str) -> AdapterStats:
        return AdapterStats()
