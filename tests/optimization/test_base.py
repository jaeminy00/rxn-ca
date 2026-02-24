"""Tests for base optimization classes."""

import pytest
import pandas as pd

from rxn_ca.optimization.base import (
    OptimizationResult,
    OptimizationHistory,
)


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_create_result(self):
        """Can create an optimization result."""
        result = OptimizationResult(
            parameters={"hold_temp": 1000, "hold_time": 5},
            score=0.75,
        )

        assert result.parameters["hold_temp"] == 1000
        assert result.score == 0.75
        assert result.recipe is None
        assert result.result_doc is None

    def test_result_with_metadata(self):
        """Result can include metadata."""
        result = OptimizationResult(
            parameters={"hold_temp": 1000},
            score=0.5,
            metadata={"iteration": 5, "phase": "optimization"},
        )

        assert result.metadata["iteration"] == 5

    def test_result_repr(self):
        """Result has readable string representation."""
        result = OptimizationResult(
            parameters={"hold_temp": 1000, "hold_time": 5},
            score=0.75,
        )

        s = repr(result)
        assert "0.75" in s
        assert "hold_temp" in s


class TestOptimizationHistory:
    """Tests for OptimizationHistory."""

    def test_empty_history(self):
        """Empty history has no results."""
        history = OptimizationHistory()

        assert len(history) == 0
        assert history.best_result is None
        assert history.scores == []

    def test_add_result(self):
        """Can add results to history."""
        history = OptimizationHistory()
        result = OptimizationResult(
            parameters={"hold_temp": 1000},
            score=0.5,
        )

        history.add(result)

        assert len(history) == 1
        assert history[0] == result

    def test_add_batch(self):
        """Can add multiple results at once."""
        history = OptimizationHistory()
        results = [
            OptimizationResult(parameters={"x": 1}, score=0.1),
            OptimizationResult(parameters={"x": 2}, score=0.2),
            OptimizationResult(parameters={"x": 3}, score=0.3),
        ]

        history.add_batch(results)

        assert len(history) == 3

    def test_best_result(self):
        """Best result returns highest score."""
        history = OptimizationHistory()
        history.add(OptimizationResult(parameters={"x": 1}, score=0.1))
        history.add(OptimizationResult(parameters={"x": 2}, score=0.9))
        history.add(OptimizationResult(parameters={"x": 3}, score=0.5))

        best = history.best_result
        assert best.score == 0.9
        assert best.parameters["x"] == 2

    def test_scores(self):
        """Can get list of all scores."""
        history = OptimizationHistory()
        history.add(OptimizationResult(parameters={}, score=0.1))
        history.add(OptimizationResult(parameters={}, score=0.5))
        history.add(OptimizationResult(parameters={}, score=0.3))

        assert history.scores == [0.1, 0.5, 0.3]

    def test_iteration(self):
        """History is iterable."""
        history = OptimizationHistory()
        history.add(OptimizationResult(parameters={}, score=0.1))
        history.add(OptimizationResult(parameters={}, score=0.2))

        scores = [r.score for r in history]
        assert scores == [0.1, 0.2]

    def test_to_dataframe(self):
        """Can convert history to DataFrame."""
        history = OptimizationHistory()
        history.add(OptimizationResult(parameters={"x": 1, "y": 2}, score=0.5))
        history.add(OptimizationResult(parameters={"x": 3, "y": 4}, score=0.8))

        df = history.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "score" in df.columns
        assert "x" in df.columns
        assert "y" in df.columns
        assert "iteration" in df.columns

    def test_get_best_n(self):
        """Can get top N results."""
        history = OptimizationHistory()
        history.add(OptimizationResult(parameters={"x": 1}, score=0.1))
        history.add(OptimizationResult(parameters={"x": 2}, score=0.9))
        history.add(OptimizationResult(parameters={"x": 3}, score=0.5))
        history.add(OptimizationResult(parameters={"x": 4}, score=0.7))

        top2 = history.get_best_n(2)

        assert len(top2) == 2
        assert top2[0].score == 0.9
        assert top2[1].score == 0.7
