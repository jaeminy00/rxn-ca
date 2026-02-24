"""Tests for ObjectiveFunction and MockObjectiveFunction."""

import pytest

from rxn_ca.optimization.objective import (
    ObjectiveConfig,
    ObjectiveFunction,
    MockObjectiveFunction,
    ScorerType,
)
from rxn_ca.optimization.base import OptimizationResult


class TestObjectiveConfig:
    """Tests for ObjectiveConfig."""

    def test_default_config(self):
        """Config has sensible defaults."""
        config = ObjectiveConfig(target_phase="BaTiO3")

        assert config.target_phase == "BaTiO3"
        assert config.scorer_type == ScorerType.FINAL
        assert config.simulation_size == 10
        assert config.num_realizations == 3
        assert config.cache_results is True

    def test_custom_config(self):
        """Can customize config."""
        config = ObjectiveConfig(
            target_phase="YMnO3",
            scorer_type=ScorerType.MAXIMUM,
            simulation_size=15,
            num_realizations=5,
            cache_results=False,
        )

        assert config.target_phase == "YMnO3"
        assert config.scorer_type == ScorerType.MAXIMUM
        assert config.simulation_size == 15


class TestMockObjectiveFunction:
    """Tests for MockObjectiveFunction."""

    def test_mock_evaluation(self):
        """Mock objective uses provided scoring function."""
        config = ObjectiveConfig(target_phase="Test")

        def score_fn(params):
            return params.get("x", 0) * 0.1

        objective = MockObjectiveFunction(config=config, score_fn=score_fn)

        result = objective.evaluate({"x": 5})

        assert isinstance(result, OptimizationResult)
        assert result.score == 0.5
        assert result.parameters == {"x": 5}

    def test_mock_caching(self):
        """Mock objective caches results."""
        config = ObjectiveConfig(target_phase="Test", cache_results=True)

        call_count = 0

        def score_fn(params):
            nonlocal call_count
            call_count += 1
            return params.get("x", 0)

        objective = MockObjectiveFunction(config=config, score_fn=score_fn)

        # First call
        result1 = objective.evaluate({"x": 5})
        assert call_count == 1

        # Same params - should be cached
        result2 = objective.evaluate({"x": 5})
        assert call_count == 1  # Not incremented
        assert result1.score == result2.score

        # Different params - not cached
        result3 = objective.evaluate({"x": 10})
        assert call_count == 2

    def test_mock_cache_disabled(self):
        """Caching can be disabled."""
        config = ObjectiveConfig(target_phase="Test", cache_results=False)

        call_count = 0

        def score_fn(params):
            nonlocal call_count
            call_count += 1
            return 1.0

        objective = MockObjectiveFunction(config=config, score_fn=score_fn)

        objective.evaluate({"x": 5})
        objective.evaluate({"x": 5})

        assert call_count == 2  # Called twice, not cached

    def test_mock_with_optimization_style_params(self):
        """Mock works with realistic optimization params."""
        config = ObjectiveConfig(target_phase="BaTiO3")

        def score_fn(params):
            # Optimal at temp=1100
            temp = params.get("hold_temp", 1000)
            return 1.0 - abs(temp - 1100) / 500

        objective = MockObjectiveFunction(config=config, score_fn=score_fn)

        result1 = objective.evaluate({"hold_temp": 1100})
        result2 = objective.evaluate({"hold_temp": 900})

        assert result1.score > result2.score


class TestObjectiveFunctionCaching:
    """Tests for ObjectiveFunction caching (without running real simulations)."""

    def test_cumulative_lib_starts_none(self):
        """Cumulative library starts as None."""
        config = ObjectiveConfig(target_phase="Test")
        objective = ObjectiveFunction(config=config)

        assert objective._cumulative_lib is None
        assert objective.cached_temps == []

    def test_cache_size_property(self):
        """Can check result cache size."""
        config = ObjectiveConfig(target_phase="Test")

        def score_fn(params):
            return 0.5

        objective = MockObjectiveFunction(config=config, score_fn=score_fn)

        assert objective.cache_size == 0

        objective.evaluate({"x": 1})
        assert objective.cache_size == 1

        objective.evaluate({"x": 2})
        assert objective.cache_size == 2

    def test_clear_cache(self):
        """Can clear result cache."""
        config = ObjectiveConfig(target_phase="Test")

        def score_fn(params):
            return 0.5

        objective = MockObjectiveFunction(config=config, score_fn=score_fn)

        objective.evaluate({"x": 1})
        objective.evaluate({"x": 2})
        assert objective.cache_size == 2

        objective.clear_cache()
        assert objective.cache_size == 0


class TestEvaluateBatch:
    """Tests for batch evaluation."""

    def test_evaluate_batch(self):
        """Can evaluate multiple params at once."""
        config = ObjectiveConfig(target_phase="Test")

        def score_fn(params):
            return params.get("x", 0) * 0.1

        objective = MockObjectiveFunction(config=config, score_fn=score_fn)

        params_list = [{"x": 1}, {"x": 2}, {"x": 3}]
        results = objective.evaluate_batch(params_list)

        assert len(results) == 3
        assert results[0].score == pytest.approx(0.1)
        assert results[1].score == pytest.approx(0.2)
        assert results[2].score == pytest.approx(0.3)

    def test_evaluate_batch_empty(self):
        """Empty batch returns empty list."""
        config = ObjectiveConfig(target_phase="Test")

        def score_fn(params):
            return 0.5

        objective = MockObjectiveFunction(config=config, score_fn=score_fn)

        results = objective.evaluate_batch([])
        assert results == []
