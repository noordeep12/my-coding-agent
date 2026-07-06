"""Tests for viewer/pricing.py."""

from my_coding_agent.viewer.pricing import (
    PRICING,
    compute_cost,
    get_price,
    project_costs,
)


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
        p = get_price("Qwen3.6-35B-A3B-6bit")
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
        assert compute_cost("Qwen3.6-35B-A3B-6bit", 500_000, 200_000) == 0.0

    def test_partial_million(self):
        # claude-sonnet-4-6: $3/1M prompt → 500k tokens = $1.50
        cost = compute_cost("claude-sonnet-4-6", 500_000, 0)
        assert abs(cost - 1.5) < 1e-9


class TestProjectCosts:
    def test_known_tokens_match_compute_cost_per_model(self):
        projected = project_costs(1_000_000, 1_000_000)
        for model, price in PRICING.items():
            if price["prompt"] or price["completion"]:
                assert (
                    abs(projected[model] - compute_cost(model, 1_000_000, 1_000_000))
                    < 1e-9
                )

    def test_excludes_zero_priced_and_unknown_models(self):
        projected = project_costs(1_000_000, 1_000_000)
        assert "Qwen3.6-35B-A3B-6bit" not in projected
        assert "llama-3.3-70b" not in projected
        assert "some-unknown-model-xyz" not in projected

    def test_includes_only_hosted_models(self):
        projected = project_costs(100, 100)
        hosted = {m for m, p in PRICING.items() if p["prompt"] or p["completion"]}
        assert set(projected.keys()) == hosted

    def test_missing_token_counts_project_to_zero(self):
        projected = project_costs(None, None)
        assert all(v == 0.0 for v in projected.values())

    def test_partial_missing_token_counts(self):
        projected = project_costs(None, 1_000_000)
        assert abs(projected["gpt-4o"] - 10.0) < 1e-9

    def test_per_call_sum_equals_whole_tree_projection(self):
        calls = [(1_000_000, 0), (0, 1_000_000), (500_000, 500_000)]
        total_prompt = sum(p for p, _ in calls)
        total_completion = sum(c for _, c in calls)
        session_projection = project_costs(total_prompt, total_completion)
        summed = {}
        for prompt, completion in calls:
            for model, cost in project_costs(prompt, completion).items():
                summed[model] = summed.get(model, 0.0) + cost
        for model, cost in session_projection.items():
            assert abs(summed[model] - cost) < 1e-9
