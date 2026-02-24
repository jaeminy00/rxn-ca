"""Base classes and data structures for optimization module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class ParameterType(str, Enum):
    """Types of optimization parameters."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    PRECURSOR_SLOT = "precursor_slot"


@dataclass
class Parameter:
    """Base class for optimization parameters."""
    name: str
    param_type: ParameterType

    def validate(self, value: Any) -> bool:
        """Validate that a value is valid for this parameter."""
        raise NotImplementedError


@dataclass
class ContinuousParameter(Parameter):
    """A continuous parameter with bounds."""
    low: float
    high: float
    param_type: ParameterType = field(default=ParameterType.CONTINUOUS, init=False)

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be less than high ({self.high})")

    def validate(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass
class DiscreteParameter(Parameter):
    """A discrete parameter with step size."""
    low: float
    high: float
    step: float
    param_type: ParameterType = field(default=ParameterType.DISCRETE, init=False)

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be less than high ({self.high})")
        if self.step <= 0:
            raise ValueError(f"step ({self.step}) must be positive")

    @property
    def values(self) -> List[float]:
        """Get all valid discrete values."""
        import numpy as np
        return list(np.arange(self.low, self.high + self.step / 2, self.step))

    def validate(self, value: float) -> bool:
        return value in self.values


@dataclass
class CategoricalParameter(Parameter):
    """A categorical parameter with discrete choices."""
    choices: List[str]
    param_type: ParameterType = field(default=ParameterType.CATEGORICAL, init=False)

    def __post_init__(self):
        if not self.choices:
            raise ValueError("choices must not be empty")

    def validate(self, value: str) -> bool:
        return value in self.choices


@dataclass
class PrecursorSlotParameter(Parameter):
    """A parameter for selecting precursors with chemical encodings.

    This is a special categorical parameter that uses chemical formulas
    as choices, enabling chemically-informed optimization via molecular
    fingerprints (e.g., MORDRED descriptors in BayBE).
    """
    candidates: List[str]  # Chemical formulas like "BaCO3", "BaO"
    param_type: ParameterType = field(default=ParameterType.PRECURSOR_SLOT, init=False)

    def __post_init__(self):
        if not self.candidates:
            raise ValueError("candidates must not be empty")

    @property
    def choices(self) -> List[str]:
        """Alias for candidates to match CategoricalParameter interface."""
        return self.candidates

    def validate(self, value: str) -> bool:
        return value in self.candidates


@dataclass
class OptimizationResult:
    """Result from a single optimization evaluation."""
    parameters: Dict[str, Any]
    score: float
    recipe: Any = None  # OptimizableRecipe
    result_doc: Any = None  # RxnCAResultDoc
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"OptimizationResult(score={self.score:.4f}, {param_str})"


class OptimizationHistory:
    """History of optimization results with utility methods."""

    def __init__(self):
        self._results: List[OptimizationResult] = []

    def add(self, result: OptimizationResult) -> None:
        """Add a result to the history."""
        self._results.append(result)

    def add_batch(self, results: List[OptimizationResult]) -> None:
        """Add multiple results to the history."""
        self._results.extend(results)

    @property
    def results(self) -> List[OptimizationResult]:
        """Get all results."""
        return self._results

    @property
    def best_result(self) -> Optional[OptimizationResult]:
        """Get the result with the highest score."""
        if not self._results:
            return None
        return max(self._results, key=lambda r: r.score)

    @property
    def scores(self) -> List[float]:
        """Get all scores."""
        return [r.score for r in self._results]

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self):
        return iter(self._results)

    def __getitem__(self, idx: int) -> OptimizationResult:
        return self._results[idx]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to a pandas DataFrame."""
        if not self._results:
            return pd.DataFrame()

        rows = []
        for i, result in enumerate(self._results):
            row = {"iteration": i, "score": result.score}
            row.update(result.parameters)
            rows.append(row)

        return pd.DataFrame(rows)

    def get_best_n(self, n: int) -> List[OptimizationResult]:
        """Get the top N results by score."""
        return sorted(self._results, key=lambda r: r.score, reverse=True)[:n]


class BaseOptimizer(ABC):
    """Abstract base class for optimizers.

    All optimizers implement a common interface with suggest/tell pattern
    and a convenience optimize() method for running the full loop.
    """

    def __init__(
        self,
        search_space: "SearchSpace",
        objective: "ObjectiveFunction",
        n_initial: int = 5,
        n_iterations: int = 20,
    ):
        """Initialize the optimizer.

        Args:
            search_space: The parameter space to search
            objective: The objective function to optimize
            n_initial: Number of initial random samples
            n_iterations: Number of optimization iterations after initial sampling
        """
        self.search_space = search_space
        self.objective = objective
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.history = OptimizationHistory()

    @abstractmethod
    def suggest(self, n_suggestions: int = 1) -> List[Dict[str, Any]]:
        """Suggest parameter configurations to evaluate.

        Args:
            n_suggestions: Number of suggestions to return

        Returns:
            List of parameter dictionaries
        """
        pass

    @abstractmethod
    def tell(self, parameters: Dict[str, Any], score: float) -> None:
        """Report the result of an evaluation to the optimizer.

        Args:
            parameters: The evaluated parameter configuration
            score: The resulting score (higher is better)
        """
        pass

    def tell_batch(self, results: List[OptimizationResult]) -> None:
        """Report multiple results to the optimizer.

        Args:
            results: List of OptimizationResult objects
        """
        for result in results:
            self.tell(result.parameters, result.score)

    def optimize(self, verbose: bool = True) -> OptimizationHistory:
        """Run the full optimization loop.

        Args:
            verbose: Whether to print progress

        Returns:
            OptimizationHistory containing all results
        """
        total_iterations = self.n_initial + self.n_iterations

        for i in range(total_iterations):
            if verbose:
                phase = "initial" if i < self.n_initial else "optimization"
                print(f"Iteration {i + 1}/{total_iterations} ({phase})")

            # Get suggestion
            suggestions = self.suggest(n_suggestions=1)
            params = suggestions[0]

            # Evaluate
            result = self.objective.evaluate(params)
            self.history.add(result)

            # Update optimizer
            self.tell(params, result.score)

            if verbose:
                print(f"  Score: {result.score:.4f}")
                if self.history.best_result:
                    print(f"  Best so far: {self.history.best_result.score:.4f}")

        return self.history

    def optimize_batch(
        self, batch_size: int = 1, verbose: bool = True
    ) -> OptimizationHistory:
        """Run optimization with batch evaluation.

        Args:
            batch_size: Number of parallel evaluations per iteration
            verbose: Whether to print progress

        Returns:
            OptimizationHistory containing all results
        """
        total_iterations = self.n_initial + self.n_iterations
        iterations = (total_iterations + batch_size - 1) // batch_size

        for i in range(iterations):
            remaining = total_iterations - len(self.history)
            current_batch = min(batch_size, remaining)

            if current_batch <= 0:
                break

            if verbose:
                print(f"Batch {i + 1}: evaluating {current_batch} configurations")

            # Get suggestions
            suggestions = self.suggest(n_suggestions=current_batch)

            # Evaluate batch
            results = self.objective.evaluate_batch(suggestions)
            self.history.add_batch(results)

            # Update optimizer
            self.tell_batch(results)

            if verbose:
                best_in_batch = max(results, key=lambda r: r.score)
                print(f"  Best in batch: {best_in_batch.score:.4f}")
                if self.history.best_result:
                    print(f"  Best overall: {self.history.best_result.score:.4f}")

        return self.history
