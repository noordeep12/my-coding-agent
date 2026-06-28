"""Tests for viewer/pricing.py."""

from my_coding_agent.viewer.pricing import compute_cost, get_price


class TestGetPrice:
    def test_known_model_returns_price(self):
        p = get_price("gpt-4o")
        assert p["prompt"] == 2.5
        assert p["completion"] == 10.0

    def test_unknown_model_returns_zero(self):
        p = get_price("some-unknown-model-xyz")
        assert p["prompt"] == 0.0
        assert p["completion"] == 0.0

    def test_local_model_returns_zero(self):
        p = get_price("Qwen3.6-35B-A3B-4bit")
        assert p["prompt"] == 0.0
        assert p["completion"] == 0.0


class TestComputeCost:
    def test_known_model_positive_cost(self):
        # gpt-4o: $2.5/1M prompt, $10/1M completion
        cost = compute_cost("gpt-4o", 1_000_000, 1_000_000)
        assert abs(cost - 12.5) < 1e-9

    def test_zero_tokens_is_zero_cost(self):
        assert compute_cost("gpt-4o", 0, 0) == 0.0

    def test_unknown_model_is_zero(self):
        assert compute_cost("nonexistent-model", 100_000, 50_000) == 0.0

    def test_local_model_is_zero(self):
        assert compute_cost("Qwen3.6-35B-A3B-4bit", 500_000, 200_000) == 0.0

    def test_partial_million(self):
        # claude-sonnet-4-6: $3/1M prompt → 500k tokens = $1.50
        cost = compute_cost("claude-sonnet-4-6", 500_000, 0)
        assert abs(cost - 1.5) < 1e-9
