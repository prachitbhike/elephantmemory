"""Token → USD pricing.

Pinned to the models we use for assistant + judge + embeddings. Update if you
swap models. Numbers are per 1M tokens.
"""

PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "claude-opus-4-7": {"in": 15.00, "out": 75.00},
    "claude-haiku-4-5-20251001": {"in": 1.00, "out": 5.00},
    "text-embedding-3-small": {"in": 0.02, "out": 0.0},
}


def cost_usd(model: str, tokens_in: int, tokens_out: int = 0) -> float:
    p = PRICING.get(model)
    if p is None:
        return 0.0
    return (tokens_in / 1_000_000) * p["in"] + (tokens_out / 1_000_000) * p["out"]
