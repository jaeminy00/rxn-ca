"""Optimization module for rxn-ca.

This module provides tools for optimizing reaction recipes to maximize
target product yield. It supports both Bayesian optimization (via BayBE)
and genetic algorithms (via DEAP).

Example:
    from rxn_ca.optimization import (
        SearchSpace, ObjectiveConfig, ObjectiveFunction,
        BayesianOptimizer, GeneticAlgorithmOptimizer
    )

    # Define search space
    search_space = (SearchSpace()
        .add_temperature_range(800, 1400, step=50)
        .add_hold_time_range(1, 15)
        .add_ramp_rate_range(5.0, 20.0)
        .add_precursor_slot("Ba_source", ["BaCO3", "BaO", "Ba(OH)2"])
        .add_precursor_ratio("Ba_source", 0.4, 0.6)
    )

    # Define objective
    objective = ObjectiveFunction(
        config=ObjectiveConfig(target_phase="BaTiO3", scorer_type="final"),
        base_reactions=rxn_set,
        phase_set=phase_set
    )

    # Run optimization
    optimizer = BayesianOptimizer(search_space, objective, n_initial=5, n_iterations=20)
    history = optimizer.optimize()

    print(f"Best: {history.best_result.score}, params: {history.best_result.parameters}")
"""

# Base classes and data structures
from .base import (
    Parameter,
    ContinuousParameter,
    DiscreteParameter,
    CategoricalParameter,
    PrecursorSlotParameter,
    ParameterType,
    OptimizationResult,
    OptimizationHistory,
    BaseOptimizer,
)

# Search space
from .search_space import SearchSpace

# Objective function
from .objective import (
    ObjectiveConfig,
    ObjectiveFunction,
    MockObjectiveFunction,
    ScorerType,
)

# Recipe
from .optimizable_recipe import OptimizableRecipe

# Utilities (scorers)
from .utilities import (
    get_result_analysis,
    AnalyzedResult,
    MaximumProductScorer,
    FinalProductScorer,
)

# Optimizers
from .bayesian import BayesianOptimizer, RecipeBayesianOptimizer
from .genetic import GeneticAlgorithmOptimizer


__all__ = [
    # Base classes
    "Parameter",
    "ContinuousParameter",
    "DiscreteParameter",
    "CategoricalParameter",
    "PrecursorSlotParameter",
    "ParameterType",
    "OptimizationResult",
    "OptimizationHistory",
    "BaseOptimizer",
    # Search space
    "SearchSpace",
    # Objective
    "ObjectiveConfig",
    "ObjectiveFunction",
    "MockObjectiveFunction",
    "ScorerType",
    # Recipe
    "OptimizableRecipe",
    # Utilities
    "get_result_analysis",
    "AnalyzedResult",
    "MaximumProductScorer",
    "FinalProductScorer",
    # Optimizers
    "BayesianOptimizer",
    "RecipeBayesianOptimizer",
    "GeneticAlgorithmOptimizer",
]
