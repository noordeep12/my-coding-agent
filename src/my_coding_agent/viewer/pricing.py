"""Model pricing table and cost helpers.

Prices are per 1 000 000 tokens (prompt and completion separately).
Local / open-source models are listed with zero cost — they have no per-token
API fee.  Unknown models also return zero cost so the viewer always renders
rather than crashing on an unrecognised model name.
"""

from __future__ import annotations

# (prompt_usd_per_1M, completion_usd_per_1M)
PRICING: dict[str, dict[str, float]] = {
    # Anthropic Claude
    "claude-opus-4-8": {"prompt": 15.0, "completion": 75.0},
    "claude-sonnet-4-6": {"prompt": 3.0, "completion": 15.0},
    "claude-haiku-4-5": {"prompt": 0.8, "completion": 4.0},
    # OpenAI
    "gpt-4o": {"prompt": 2.5, "completion": 10.0},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
    "o3": {"prompt": 10.0, "completion": 40.0},
    "o4-mini": {"prompt": 1.1, "completion": 4.4},
    # Local / open-source (zero API cost)
    "Qwen3.6-35B-A3B-6bit": {"prompt": 0.0, "completion": 0.0},
    "llama-3.3-70b": {"prompt": 0.0, "completion": 0.0},
    "mistral-small": {"prompt": 0.0, "completion": 0.0},
    "deepseek-r1": {"prompt": 0.0, "completion": 0.0},
}

_UNKNOWN: dict[str, float] = {"prompt": 0.0, "completion": 0.0}


def get_price(model: str) -> dict[str, float]:
    """Return the ``{prompt, completion}`` price dict for *model*.

    Returns zero-cost dict for unknown or local models.

    Args:
        model: Model identifier string (e.g. ``"gpt-4o"``).

    Returns:
        Dict with keys ``"prompt"`` and ``"completion"``, values in USD per 1M tokens.
    """
    return PRICING.get(model, _UNKNOWN)


def project_costs(
    prompt_tokens: int | None, completion_tokens: int | None
) -> dict[str, float]:
    """Return projected USD cost on every hosted model for the given tokens.

    Args:
        prompt_tokens: Recorded prompt tokens; ``None`` is treated as ``0``.
        completion_tokens: Recorded completion tokens; ``None`` is treated as ``0``.

    Returns:
        Dict mapping model name to projected USD cost, one entry per
        non-zero-priced row in ``PRICING``. Zero-priced (local) and unknown
        models are excluded.
    """
    prompt = prompt_tokens or 0
    completion = completion_tokens or 0
    return {
        model: (prompt * price["prompt"] + completion * price["completion"])
        / 1_000_000
        for model, price in PRICING.items()
        if price["prompt"] or price["completion"]
    }


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return the estimated cost in USD for one LLM call.

    Args:
        model: Model identifier string.
        prompt_tokens: Number of prompt tokens billed.
        completion_tokens: Number of completion tokens billed.

    Returns:
        Estimated cost in USD; ``0.0`` for unknown or local models.
    """
    price = get_price(model)
    return (
        prompt_tokens * price["prompt"] + completion_tokens * price["completion"]
    ) / 1_000_000
