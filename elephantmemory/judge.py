"""LLM-as-judge with on-disk cache keyed by (probe_id, response_hash).

Cached so re-runs of the report don't re-bill the judge. The cache is keyed on
the response text, so any change to the adapter's output invalidates it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from . import llm
from .cost import cost_usd
from .scoring import contains_match, exact_match, must_not_contain
from .types import Probe

CACHE_PATH = Path("results/cache/judge.sqlite")

JUDGE_SYSTEM = """You are grading a memory-system response.

You receive: the user's original prompt, the system's response, the expected
answer (if any), and a rubric. Reply ONLY with a single JSON object:
{"score": 0.0-1.0, "passed": true|false, "reason": "<one short sentence>"}

Be strict. A response that hedges or refuses when the expected answer was
clearly known counts as a failure. A response that confidently answers when
the expected outcome was abstention counts as a failure."""


def _conn() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(CACHE_PATH)
    c.execute(
        "CREATE TABLE IF NOT EXISTS judge_cache ("
        "key TEXT PRIMARY KEY, score REAL, passed INTEGER, reason TEXT)"
    )
    return c


def _key(probe: Probe, response: str) -> str:
    h = hashlib.sha256(response.encode("utf-8")).hexdigest()[:16]
    return f"{probe.probe_id}:{h}"


def llm_judge(probe: Probe, response: str) -> tuple[float, bool, str, float]:
    key = _key(probe, response)
    with _conn() as c:
        row = c.execute(
            "SELECT score, passed, reason FROM judge_cache WHERE key=?", (key,)
        ).fetchone()
        if row is not None:
            return row[0], bool(row[1]), row[2], 0.0

    user_msg = json.dumps(
        {
            "prompt": probe.prompt,
            "expected": probe.expected,
            "rubric": probe.rubric or "Award full credit only if the response matches the expected answer in substance.",
            "response": response,
        }
    )
    out = llm.chat(
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
        model=llm.JUDGE_MODEL,
        max_tokens=256,
    )
    try:
        parsed = json.loads(out.text.strip())
        score = float(parsed.get("score", 0.0))
        passed = bool(parsed.get("passed", score >= 0.75))
        reason = str(parsed.get("reason", ""))
    except (json.JSONDecodeError, ValueError, KeyError):
        score, passed, reason = 0.0, False, f"judge parse error: {out.text[:120]}"

    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO judge_cache VALUES (?,?,?,?)",
            (key, score, int(passed), reason),
        )
    return score, passed, reason, cost_usd(llm.JUDGE_MODEL, out.tokens_in, out.tokens_out)


def score_probe(probe: Probe, response: str) -> tuple[float, bool, str, float]:
    if probe.score_method == "exact":
        s, r = exact_match(probe, response)
        return s, s >= 1.0, r, 0.0
    if probe.score_method == "contains":
        s, r = contains_match(probe, response)
        return s, s >= 1.0, r, 0.0
    if probe.score_method == "must_not_contain":
        s, r = must_not_contain(probe, response)
        return s, s >= 1.0, r, 0.0
    return llm_judge(probe, response)
