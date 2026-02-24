"""Tests for BayesianOptimizer and GeneticAlgorithmOptimizer."""

import pytest

from rxn_ca.optimization import (
    SearchSpace,
    ObjectiveConfig,
    MockObjectiveFunction,
    BayesianOptimizer,
    GeneticAlgorithmOptimizer,
)


# Skip tests if optional dependencies not installed
try:
    import baybe
    BAYBE_AVAILABLE = True
except ImportError:
    BAYBE_AVAILABLE = False

try:
    import deap
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False


@pytest.fixture
def simple_search_space():
    """A simple search space for testing."""
    return (
        SearchSpace()
        .add_temperature_range(800, 1200, step=100)
        .add_hold_time_range(1, 10)
    )


@pytest.fixture
def mock_objective():
    """A mock objective function with known optimum."""
    config = ObjectiveConfig(target_phase="Test")

    def score_fn(params):
        # Optimum at temp=1000, time=5
        temp = params.get("hold_temp", 1000)
        time = params.get("hold_time", 5)
        temp_penalty = ((temp - 1000) / 200) ** 2
        time_penalty = ((time - 5) / 5) ** 2
        return max(0, 1.0 - temp_penalty - time_penalty)

    return MockObjectiveFunction(config=config, score_fn=score_fn)


@pytest.mark.skipif(not BAYBE_AVAILABLE, reason="BayBE not installed")
class TestBayesianOptimizer:
    """Tests for BayesianOptimizer."""

    def test_create_optimizer(self, simple_search_space, mock_objective):
        """Can create a Bayesian optimizer."""
        optimizer = BayesianOptimizer(
            simple_search_space,
            mock_objective,
            n_initial=2,
            n_iterations=3,
        )

        assert optimizer.n_initial == 2
        assert optimizer.n_iterations == 3

    def test_suggest(self, simple_search_space, mock_objective):
        """Optimizer can suggest parameter configurations."""
        optimizer = BayesianOptimizer(
            simple_search_space,
            mock_objective,
            n_initial=2,
            n_iterations=2,
        )

        suggestions = optimizer.suggest(n_suggestions=1)

        assert len(suggestions) == 1
        assert "hold_temp" in suggestions[0]
        assert "hold_time" in suggestions[0]

    def test_suggest_multiple(self, simple_search_space, mock_objective):
        """Can request multiple suggestions at once."""
        optimizer = BayesianOptimizer(
            simple_search_space,
            mock_objective,
            n_initial=3,
            n_iterations=2,
        )

        suggestions = optimizer.suggest(n_suggestions=3)

        assert len(suggestions) == 3

    def test_tell(self, simple_search_space, mock_objective):
        """Optimizer accepts evaluation results."""
        optimizer = BayesianOptimizer(
            simple_search_space,
            mock_objective,
            n_initial=2,
            n_iterations=2,
        )

        params = {"hold_temp": 1000, "hold_time": 5}
        optimizer.tell(params, score=0.8)

        # Should have one measurement recorded
        assert len(optimizer.measurements) == 1

    def test_optimize_loop(self, simple_search_space, mock_objective):
        """Full optimization loop works."""
        optimizer = BayesianOptimizer(
            simple_search_space,
            mock_objective,
            n_initial=2,
            n_iterations=2,
        )

        history = optimizer.optimize(verbose=False)

        assert len(history) == 4  # 2 initial + 2 optimization
        assert history.best_result is not None
        assert history.best_result.score > 0

    def test_optimize_improves(self, simple_search_space, mock_objective):
        """Optimization tends to improve over iterations."""
        optimizer = BayesianOptimizer(
            simple_search_space,
            mock_objective,
            n_initial=3,
            n_iterations=5,
        )

        history = optimizer.optimize(verbose=False)

        # Best score should be reasonably good
        assert history.best_result.score > 0.5

    def test_with_precursor_slot(self, mock_objective):
        """Works with precursor slot parameters."""
        search_space = (
            SearchSpace()
            .add_temperature_range(900, 1100, step=100)
            .add_precursor_slot("Ba_source", ["BaO", "BaO2"])
        )

        optimizer = BayesianOptimizer(
            search_space,
            mock_objective,
            n_initial=2,
            n_iterations=1,
        )

        suggestions = optimizer.suggest(n_suggestions=1)

        assert "Ba_source" in suggestions[0]
        assert suggestions[0]["Ba_source"] in ["BaO", "BaO2"]


@pytest.mark.skipif(not DEAP_AVAILABLE, reason="DEAP not installed")
class TestGeneticAlgorithmOptimizer:
    """Tests for GeneticAlgorithmOptimizer."""

    def test_create_optimizer(self, simple_search_space, mock_objective):
        """Can create a GA optimizer."""
        optimizer = GeneticAlgorithmOptimizer(
            simple_search_space,
            mock_objective,
            population_size=5,
            n_generations=2,
        )

        assert optimizer.population_size == 5
        assert optimizer.n_generations == 2

    def test_suggest(self, simple_search_space, mock_objective):
        """Optimizer can suggest parameter configurations."""
        optimizer = GeneticAlgorithmOptimizer(
            simple_search_space,
            mock_objective,
            population_size=5,
            n_generations=2,
        )

        suggestions = optimizer.suggest(n_suggestions=1)

        assert len(suggestions) == 1
        assert "hold_temp" in suggestions[0]
        assert "hold_time" in suggestions[0]

    def test_tell(self, simple_search_space, mock_objective):
        """Optimizer accepts evaluation results."""
        optimizer = GeneticAlgorithmOptimizer(
            simple_search_space,
            mock_objective,
            population_size=5,
            n_generations=2,
        )

        params = {"hold_temp": 1000, "hold_time": 5}
        # GA uses tell to update fitness
        optimizer.tell(params, score=0.8)

    def test_optimize_loop(self, simple_search_space, mock_objective):
        """Full optimization loop works."""
        optimizer = GeneticAlgorithmOptimizer(
            simple_search_space,
            mock_objective,
            population_size=5,
            n_generations=2,
        )

        history = optimizer.optimize(verbose=False)

        assert len(history) > 0
        assert history.best_result is not None

    def test_with_categorical(self, mock_objective):
        """Works with categorical parameters."""
        search_space = (
            SearchSpace()
            .add_temperature_range(900, 1100, step=100)
            .add_precursor_slot("Ba_source", ["BaO", "BaO2"])
        )

        optimizer = GeneticAlgorithmOptimizer(
            search_space,
            mock_objective,
            population_size=4,
            n_generations=2,
        )

        history = optimizer.optimize(verbose=False)

        # Should complete without error
        assert len(history) > 0


class TestOptimizerBatchMode:
    """Tests for batch optimization mode."""

    @pytest.mark.skipif(not BAYBE_AVAILABLE, reason="BayBE not installed")
    def test_optimize_batch(self, simple_search_space, mock_objective):
        """Batch optimization works."""
        optimizer = BayesianOptimizer(
            simple_search_space,
            mock_objective,
            n_initial=4,
            n_iterations=0,
        )

        history = optimizer.optimize_batch(batch_size=2, verbose=False)

        assert len(history) == 4

    @pytest.mark.skipif(not BAYBE_AVAILABLE, reason="BayBE not installed")
    def test_tell_batch(self, simple_search_space, mock_objective):
        """Can report batch results."""
        from rxn_ca.optimization.base import OptimizationResult

        optimizer = BayesianOptimizer(
            simple_search_space,
            mock_objective,
            n_initial=4,
            n_iterations=0,
        )

        results = [
            OptimizationResult(parameters={"hold_temp": 900, "hold_time": 3}, score=0.5),
            OptimizationResult(parameters={"hold_temp": 1000, "hold_time": 5}, score=0.9),
        ]

        optimizer.tell_batch(results)

        assert len(optimizer.measurements) == 2
