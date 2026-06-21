"""Token pricing — edit ``PRICES`` in Python to value hosted models.

Local models (MLX Server, Ollama) are free, so the default table is empty and
every cost resolves to ``$0.00``. To estimate spend for a hosted model, add an
entry mapping its id to ``(input_usd_per_1k, output_usd_per_1k)``.
"""

# model id -> (USD per 1k prompt tokens, USD per 1k completion tokens)
PRICES: dict[str, tuple[float, float]] = {
    # "claude-opus-4-8": (15.0 / 1000, 75.0 / 1000),  # example; uncomment + adjust
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return the estimated USD cost for a model's token usage.

    Args:
        model: Model id to look up in ``PRICES``.
        prompt_tokens: Number of prompt (input) tokens.
        completion_tokens: Number of completion (output) tokens.

    Returns:
        Estimated cost in USD, or ``0.0`` when the model is not priced.
    """
    rate = PRICES.get(model)
    if not rate:
        return 0.0
    in_rate, out_rate = rate
    return prompt_tokens / 1000 * in_rate + completion_tokens / 1000 * out_rate
