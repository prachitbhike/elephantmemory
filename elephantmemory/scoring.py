from __future__ import annotations

import re

from .types import Probe


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def exact_match(probe: Probe, response: str) -> tuple[float, str]:
    if probe.expected is None:
        return 0.0, "no expected value"
    matched = normalize(probe.expected) == normalize(response)
    return (1.0 if matched else 0.0), ""


def contains_match(probe: Probe, response: str) -> tuple[float, str]:
    if probe.expected is None:
        return 0.0, "no expected value"
    matched = normalize(probe.expected) in normalize(response)
    return (1.0 if matched else 0.0), ""


def must_not_contain(probe: Probe, response: str) -> tuple[float, str]:
    n = normalize(response)
    leaks = [t for t in probe.must_not_contain if normalize(t) in n]
    if leaks:
        return 0.0, f"leaked tokens: {leaks}"
    return 1.0, ""
