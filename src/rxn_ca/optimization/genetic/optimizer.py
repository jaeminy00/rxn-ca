"""Genetic algorithm optimization using DEAP."""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base import (
    BaseOptimizer,
    ContinuousParameter,
    DiscreteParameter,
    CategoricalParameter,
    PrecursorSlotParameter,
    OptimizationResult,
    OptimizationHistory,
)
from ..search_space import SearchSpace
from ..objective import ObjectiveFunction

# DEAP imports
try:
    from deap import base, creator, tools, algorithms

    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """Genetic algorithm optimizer using the DEAP library.

    Implements a standard genetic algorithm with:
    - Tournament selection
    - Uniform crossover for mixed parameter types
    - Gaussian mutation for continuous parameters
    - Random choice mutation for discrete/categorical parameters

    Example:
        optimizer = GeneticAlgorithmOptimizer(
            search_space=search_space,
            objective=objective,
            population_size=20,
            n_generations=10,
        )
        history = optimizer.optimize()
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective: ObjectiveFunction,
        population_size: int = 20,
        n_generations: int = 10,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        tournament_size: int = 3,
        elite_size: int = 2,
        mutation_sigma: float = 0.1,
        n_initial: int = 0,
        n_iterations: int = 0,
    ):
        """Initialize the genetic algorithm optimizer.

        Args:
            search_space: The parameter space to search
            objective: The objective function to optimize
            population_size: Number of individuals in the population
            n_generations: Number of generations to evolve
            crossover_prob: Probability of crossover between two individuals
            mutation_prob: Probability of mutating an individual
            tournament_size: Number of individuals in tournament selection
            elite_size: Number of best individuals to preserve each generation
            mutation_sigma: Standard deviation for Gaussian mutation (as fraction of range)
            n_initial: Not used (kept for interface compatibility)
            n_iterations: Not used (kept for interface compatibility)
        """
        if not DEAP_AVAILABLE:
            raise ImportError(
                "DEAP is not installed. Install it with: pip install deap"
            )

        # GA uses population_size * n_generations as total evaluations
        super().__init__(
            search_space,
            objective,
            n_initial=population_size,  # First generation is like initial sampling
            n_iterations=population_size * (n_generations - 1),
        )

        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.mutation_sigma = mutation_sigma

        # DEAP setup
        self._setup_deap()

        # Current population and generation tracking
        self._population: List[Any] = []
        self._generation = 0
        self._evaluated_count = 0
        self._pending_evaluations: List[Dict[str, Any]] = []

    def _setup_deap(self) -> None:
        """Set up DEAP creator and toolbox."""
        # Create fitness class (maximize)
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        # Create individual class
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        # Create toolbox
        self._toolbox = base.Toolbox()

        # Register gene generators for each parameter
        for i, param in enumerate(self.search_space.parameters):
            if isinstance(param, ContinuousParameter):
                self._toolbox.register(
                    f"attr_{i}",
                    random.uniform,
                    param.low,
                    param.high,
                )
            elif isinstance(param, DiscreteParameter):
                self._toolbox.register(
                    f"attr_{i}",
                    random.choice,
                    param.values,
                )
            elif isinstance(param, (CategoricalParameter, PrecursorSlotParameter)):
                self._toolbox.register(
                    f"attr_{i}",
                    random.choice,
                    param.choices,
                )

        # Register individual and population creators
        n_params = len(self.search_space.parameters)

        def create_individual():
            ind = []
            for i in range(n_params):
                gene = getattr(self._toolbox, f"attr_{i}")()
                ind.append(gene)
            return creator.Individual(ind)

        self._toolbox.register("individual", create_individual)
        self._toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self._toolbox.individual,
        )

        # Register genetic operators
        self._toolbox.register(
            "select",
            tools.selTournament,
            tournsize=self.tournament_size,
        )
        self._toolbox.register("mate", self._crossover)
        self._toolbox.register("mutate", self._mutate)

    def _individual_to_params(self, individual: List[Any]) -> Dict[str, Any]:
        """Convert a DEAP individual to a parameter dictionary."""
        params = {}
        for i, param in enumerate(self.search_space.parameters):
            value = individual[i]
            # Handle numpy types
            if hasattr(value, "item"):
                value = value.item()
            params[param.name] = value
        return params

    def _params_to_individual(self, params: Dict[str, Any]) -> List[Any]:
        """Convert a parameter dictionary to a DEAP individual."""
        individual = []
        for param in self.search_space.parameters:
            individual.append(params[param.name])
        return creator.Individual(individual)

    def _crossover(
        self, ind1: List[Any], ind2: List[Any]
    ) -> Tuple[List[Any], List[Any]]:
        """Uniform crossover for mixed parameter types."""
        for i, param in enumerate(self.search_space.parameters):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2

    def _mutate(self, individual: List[Any]) -> Tuple[List[Any]]:
        """Mutate an individual.

        Uses Gaussian mutation for continuous parameters and
        random choice for discrete/categorical parameters.
        """
        for i, param in enumerate(self.search_space.parameters):
            if random.random() < self.mutation_prob:
                if isinstance(param, ContinuousParameter):
                    # Gaussian mutation
                    sigma = self.mutation_sigma * (param.high - param.low)
                    individual[i] += random.gauss(0, sigma)
                    # Clip to bounds
                    individual[i] = max(param.low, min(param.high, individual[i]))

                elif isinstance(param, DiscreteParameter):
                    # Random choice from discrete values
                    individual[i] = random.choice(param.values)

                elif isinstance(param, (CategoricalParameter, PrecursorSlotParameter)):
                    # Random choice from categories
                    individual[i] = random.choice(param.choices)

        return (individual,)

    def suggest(self, n_suggestions: int = 1) -> List[Dict[str, Any]]:
        """Suggest parameter configurations to evaluate.

        For the genetic algorithm, this manages the population evolution.

        Args:
            n_suggestions: Number of suggestions to return

        Returns:
            List of parameter dictionaries
        """
        suggestions = []

        for _ in range(n_suggestions):
            if not self._population:
                # Initialize population on first call
                self._population = self._toolbox.population(n=self.population_size)
                self._generation = 0

            # Find next unevaluated individual
            unevaluated = [
                ind for ind in self._population if not ind.fitness.valid
            ]

            if unevaluated:
                # Return next unevaluated individual
                ind = unevaluated[0]
                params = self._individual_to_params(ind)
                self._pending_evaluations.append(params)
                suggestions.append(params)
            else:
                # All individuals evaluated, evolve to next generation
                self._evolve_population()

                # Get first individual from new generation
                unevaluated = [
                    ind for ind in self._population if not ind.fitness.valid
                ]
                if unevaluated:
                    ind = unevaluated[0]
                    params = self._individual_to_params(ind)
                    self._pending_evaluations.append(params)
                    suggestions.append(params)

        return suggestions

    def _evolve_population(self) -> None:
        """Evolve the population to the next generation."""
        self._generation += 1

        # Select elite individuals
        elite = tools.selBest(self._population, self.elite_size)

        # Select parents for next generation
        offspring = self._toolbox.select(
            self._population,
            len(self._population) - self.elite_size,
        )
        offspring = list(map(self._toolbox.clone, offspring))

        # Apply crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < self.crossover_prob:
                self._toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Apply mutation
        for mutant in offspring:
            if random.random() < self.mutation_prob:
                self._toolbox.mutate(mutant)
                del mutant.fitness.values

        # Create new population: elite + offspring
        self._population = [self._toolbox.clone(e) for e in elite] + offspring

        # Elite individuals keep their fitness
        for i, e in enumerate(elite):
            self._population[i].fitness = e.fitness

    def tell(self, parameters: Dict[str, Any], score: float) -> None:
        """Report the result of an evaluation to the optimizer.

        Args:
            parameters: The evaluated parameter configuration
            score: The resulting score (higher is better)
        """
        # Find the individual that matches these parameters
        for ind in self._population:
            if not ind.fitness.valid:
                ind_params = self._individual_to_params(ind)
                if self._params_match(ind_params, parameters):
                    ind.fitness.values = (score,)
                    self._evaluated_count += 1
                    break

        # Remove from pending
        if parameters in self._pending_evaluations:
            self._pending_evaluations.remove(parameters)

    def _params_match(
        self, params1: Dict[str, Any], params2: Dict[str, Any]
    ) -> bool:
        """Check if two parameter dictionaries match."""
        for key in params1:
            v1, v2 = params1[key], params2.get(key)
            if isinstance(v1, float) and isinstance(v2, float):
                if abs(v1 - v2) > 1e-9:
                    return False
            elif v1 != v2:
                return False
        return True

    def optimize(self, verbose: bool = True) -> OptimizationHistory:
        """Run the full genetic algorithm optimization.

        Args:
            verbose: Whether to print progress

        Returns:
            OptimizationHistory containing all results
        """
        # Initialize population
        self._population = self._toolbox.population(n=self.population_size)
        self._generation = 0

        for gen in range(self.n_generations):
            if verbose:
                print(f"Generation {gen + 1}/{self.n_generations}")

            # Evaluate all unevaluated individuals
            unevaluated = [
                ind for ind in self._population if not ind.fitness.valid
            ]

            for ind in unevaluated:
                params = self._individual_to_params(ind)
                result = self.objective.evaluate(params)
                ind.fitness.values = (result.score,)
                self.history.add(result)

            # Get statistics
            fits = [ind.fitness.values[0] for ind in self._population]
            best = max(fits)
            avg = sum(fits) / len(fits)

            if verbose:
                print(f"  Best: {best:.4f}, Avg: {avg:.4f}")

            # Evolve (except for last generation)
            if gen < self.n_generations - 1:
                self._evolve_population()

        return self.history

    @property
    def best_individual(self) -> Optional[Dict[str, Any]]:
        """Get the best individual from the current population."""
        if not self._population:
            return None

        valid = [ind for ind in self._population if ind.fitness.valid]
        if not valid:
            return None

        best = max(valid, key=lambda x: x.fitness.values[0])
        return self._individual_to_params(best)

    @property
    def current_generation(self) -> int:
        """Get the current generation number."""
        return self._generation
