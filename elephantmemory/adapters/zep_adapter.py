"""Zep / Graphiti adapter — bitemporal knowledge graph.

STATUS: skeleton. Wire up in a follow-up commit.

Implementation notes:
- Use graphiti-core (OSS) against Neo4j (docker-compose service `neo4j`).
  Zep Cloud is excluded from v1 per the self-host fairness policy.
- record_session → graphiti.add_episode(name=session_id, episode_body=transcript,
  source=EpisodeType.message, reference_time=session.timestamp,
  group_id=user_id). Graphiti runs LLM extraction internally.
- query → graphiti.search(query, group_ids=[user_id]); inject edges/nodes into
  the answering Claude call.
- forget → graphiti has node/edge delete APIs; for predicate-based, do a
  semantic search first then delete matching nodes/edges. Test whether
  bitemporal invalidation suffices for "forget" semantics or if we need hard
  delete for compliance.
- This is the framework where the temporal_supersede category should win — pay
  attention to whether Graphiti correctly invalidates the prior fact.
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
class ZepAdapter:
    name: str = "zep"

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def reset_user(self, user_id: str) -> None:
        pass

    def record_session(self, session: Session) -> WriteResult:
        raise NotImplementedError("zep adapter pending — see module docstring")

    def query(self, user_id: str, prompt: str) -> QueryResult:
        raise NotImplementedError("zep adapter pending — see module docstring")

    def forget(self, user_id: str, predicate: str) -> ForgetResult:
        raise NotImplementedError("zep adapter pending — see module docstring")

    def stats(self, user_id: str) -> AdapterStats:
        return AdapterStats()
