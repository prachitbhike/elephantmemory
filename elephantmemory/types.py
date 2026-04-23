from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

Role = Literal["user", "assistant"]
ScoreMethod = Literal["exact", "contains", "must_not_contain", "llm_judge"]
EventKind = Literal["session", "probe", "forget"]


@dataclass
class Turn:
    role: Role
    content: str


@dataclass
class Session:
    session_id: str
    user_id: str
    timestamp: datetime
    turns: list[Turn]


@dataclass
class Probe:
    probe_id: str
    user_id: str
    timestamp: datetime
    prompt: str
    expected: str | None = None
    must_not_contain: list[str] = field(default_factory=list)
    category: str = "recall"
    score_method: ScoreMethod = "llm_judge"
    rubric: str | None = None


@dataclass
class ForgetOp:
    user_id: str
    timestamp: datetime
    predicate: str


@dataclass
class Event:
    timestamp: datetime
    kind: EventKind
    payload: Session | Probe | ForgetOp


@dataclass
class Scenario:
    scenario_id: str
    description: str
    category: str
    events: list[Event]


@dataclass
class RetrievedContext:
    text: str
    facts: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class WriteResult:
    success: bool
    latency_ms: float
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    note: str = ""


@dataclass
class QueryResult:
    response: str
    context: RetrievedContext
    latency_ms: float
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class ForgetResult:
    items_removed: int
    latency_ms: float
    cost_usd: float = 0.0


@dataclass
class AdapterStats:
    facts_stored: int = 0
    bytes_stored: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeOutcome:
    probe_id: str
    category: str
    score: float
    passed: bool
    response: str
    expected: str | None
    judge_reason: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass
class ScenarioResult:
    scenario_id: str
    adapter: str
    outcomes: list[ProbeOutcome]
    write_latency_ms_p50: float
    write_latency_ms_p95: float
    write_cost_usd: float
    final_stats: AdapterStats
    error: str | None = None
