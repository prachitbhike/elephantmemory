from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from .types import Event, ForgetOp, Probe, Scenario, Session, Turn


def _ts(s: str | datetime) -> datetime:
    if isinstance(s, datetime):
        return s
    return datetime.fromisoformat(s)


def _session(d: dict) -> Session:
    return Session(
        session_id=d["session_id"],
        user_id=d["user_id"],
        timestamp=_ts(d["timestamp"]),
        turns=[Turn(role=t["role"], content=t["content"]) for t in d["turns"]],
    )


def _probe(d: dict) -> Probe:
    return Probe(
        probe_id=d["probe_id"],
        user_id=d["user_id"],
        timestamp=_ts(d["timestamp"]),
        prompt=d["prompt"],
        expected=d.get("expected"),
        must_not_contain=d.get("must_not_contain", []),
        category=d.get("category", "recall"),
        score_method=d.get("score_method", "llm_judge"),
        rubric=d.get("rubric"),
    )


def _forget(d: dict) -> ForgetOp:
    return ForgetOp(
        user_id=d["user_id"],
        timestamp=_ts(d["timestamp"]),
        predicate=d["predicate"],
    )


def load_scenario(path: Path) -> Scenario:
    raw = yaml.safe_load(path.read_text())
    events: list[Event] = []
    for e in raw["events"]:
        kind = e["kind"]
        payload_dict = e["payload"]
        if kind == "session":
            payload = _session(payload_dict)
        elif kind == "probe":
            payload = _probe(payload_dict)
        elif kind == "forget":
            payload = _forget(payload_dict)
        else:
            raise ValueError(f"unknown event kind: {kind}")
        events.append(Event(timestamp=payload.timestamp, kind=kind, payload=payload))
    events.sort(key=lambda x: x.timestamp)
    return Scenario(
        scenario_id=raw["scenario_id"],
        description=raw["description"],
        category=raw["category"],
        events=events,
    )


def load_all(dir_path: Path) -> list[Scenario]:
    paths = sorted(dir_path.glob("*.yaml")) + sorted(dir_path.glob("*.yml"))
    return [load_scenario(p) for p in paths]
