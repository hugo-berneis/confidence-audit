"""USD pricing for Anthropic models, per million tokens.

Hand-maintained -- there's no pricing API. Source: Anthropic's published
per-model pricing, checked 2026-10-04. Update alongside ANTHROPIC_MODEL in
.env.example if the model changes.
"""

from __future__ import annotations

# (input $ / 1M tokens, output $ / 1M tokens), keyed by model-ID prefix --
# dated snapshots (e.g. "claude-haiku-4-5-20251001") share their alias's price.
_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    for prefix, (input_price, output_price) in _PRICING_PER_MILLION.items():
        if model.startswith(prefix):
            return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    raise ValueError(f"no pricing on file for model {model!r} -- add it to _PRICING_PER_MILLION")
