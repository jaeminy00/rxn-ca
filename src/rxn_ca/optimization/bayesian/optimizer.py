"""Bayesian optimization using BayBE."""

from typing import Any, Dict, List, Optional

import pandas as pd

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

# BayBE imports
try:
    from baybe import Campaign
    from baybe.parameters import (
        NumericalContinuousParameter as BayBEContinuous,
        NumericalDiscreteParameter as BayBEDiscrete,
        CategoricalParameter as BayBECategorical,
        SubstanceParameter as BayBESubstance,
    )
    from baybe.searchspace import SearchSpace as BayBESearchSpace
    from baybe.targets import NumericalTarget
    from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
    from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
    from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender

    BAYBE_AVAILABLE = True
except ImportError:
    BAYBE_AVAILABLE = False


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimizer using the BayBE library.

    Uses Gaussian Process regression with Bayesian optimization to efficiently
    explore the parameter space. Supports chemical encoding for precursor
    selection via SMILES strings.

    Example:
        optimizer = BayesianOptimizer(
            search_space=search_space,
            objective=objective,
            n_initial=5,
            n_iterations=20,
        )
        history = optimizer.optimize()
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective: ObjectiveFunction,
        n_initial: int = 5,
        n_iterations: int = 20,
        target_name: str = "yield",
        precursor_smiles: Optional[Dict[str, Dict[str, str]]] = None,
        acquisition_function: Optional[str] = None,
    ):
        """Initialize the Bayesian optimizer.

        Args:
            search_space: The parameter space to search
            objective: The objective function to optimize
            n_initial: Number of initial random samples before Bayesian optimization
            n_iterations: Number of Bayesian optimization iterations
            target_name: Name of the target variable (default "yield")
            precursor_smiles: Optional mapping from precursor slot names to
                dictionaries of {precursor_formula: SMILES_string}. If provided,
                enables chemical encoding via MORDRED fingerprints.
                Example: {"Ba_source": {"BaCO3": "C(=O)([O-])[O-].[Ba+2]"}}
            acquisition_function: Optional acquisition function name (not yet implemented)
        """
        if not BAYBE_AVAILABLE:
            raise ImportError(
                "BayBE is not installed. Install it with: pip install baybe"
            )

        super().__init__(search_space, objective, n_initial, n_iterations)

        self.target_name = target_name
        self.precursor_smiles = precursor_smiles or {}
        self.acquisition_function = acquisition_function

        # Build BayBE campaign
        self._campaign = self._build_campaign()

    def _convert_parameter(self, param) -> Any:
        """Convert a SearchSpace parameter to a BayBE parameter."""
        if isinstance(param, ContinuousParameter):
            return BayBEContinuous(
                name=param.name,
                bounds=(param.low, param.high),
            )

        elif isinstance(param, DiscreteParameter):
            return BayBEDiscrete(
                name=param.name,
                values=param.values,
            )

        elif isinstance(param, PrecursorSlotParameter):
            # Check if SMILES are provided for this slot
            if param.name in self.precursor_smiles:
                smiles_map = self.precursor_smiles[param.name]
                # Verify all candidates have SMILES
                missing = set(param.candidates) - set(smiles_map.keys())
                if missing:
                    raise ValueError(
                        f"Missing SMILES for precursors in slot '{param.name}': {missing}"
                    )
                return BayBESubstance(
                    name=param.name,
                    data=smiles_map,
                )
            else:
                # Fall back to categorical parameter
                return BayBECategorical(
                    name=param.name,
                    values=param.candidates,
                )

        elif isinstance(param, CategoricalParameter):
            return BayBECategorical(
                name=param.name,
                values=param.choices,
            )

        else:
            raise TypeError(f"Unknown parameter type: {type(param)}")

    def _build_campaign(self) -> "Campaign":
        """Build a BayBE Campaign from the search space."""
        # Convert parameters
        baybe_params = [
            self._convert_parameter(p) for p in self.search_space.parameters
        ]

        # Create BayBE SearchSpace from parameters
        baybe_searchspace = BayBESearchSpace.from_product(baybe_params)

        # Create target (maximize yield)
        target = NumericalTarget(name=self.target_name, minimize=False)

        # Create recommender with two phases:
        # 1. Random exploration for n_initial samples
        # 2. Bayesian optimization after that
        recommender = TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(),
            switch_after=self.n_initial,
        )

        # Create campaign
        campaign = Campaign(
            searchspace=baybe_searchspace,
            objective=target,
            recommender=recommender,
        )

        return campaign

    def suggest(self, n_suggestions: int = 1) -> List[Dict[str, Any]]:
        """Suggest parameter configurations to evaluate.

        Args:
            n_suggestions: Number of suggestions to return

        Returns:
            List of parameter dictionaries
        """
        # Get recommendations from BayBE
        recommendations = self._campaign.recommend(batch_size=n_suggestions)

        # Convert DataFrame to list of dicts
        suggestions = []
        for _, row in recommendations.iterrows():
            config = {}
            for param in self.search_space.parameters:
                value = row[param.name]
                # Handle numpy types
                if hasattr(value, "item"):
                    value = value.item()
                config[param.name] = value
            suggestions.append(config)

        return suggestions

    def tell(self, parameters: Dict[str, Any], score: float) -> None:
        """Report the result of an evaluation to the optimizer.

        Args:
            parameters: The evaluated parameter configuration
            score: The resulting score (higher is better)
        """
        # Create a DataFrame with the measurement
        data = {**parameters, self.target_name: score}
        df = pd.DataFrame([data])

        # Add measurement to campaign
        self._campaign.add_measurements(df)

    def tell_batch(self, results: List[OptimizationResult]) -> None:
        """Report multiple results to the optimizer.

        Args:
            results: List of OptimizationResult objects
        """
        if not results:
            return

        # Create DataFrame from all results
        rows = []
        for result in results:
            row = {**result.parameters, self.target_name: result.score}
            rows.append(row)
        df = pd.DataFrame(rows)

        # Add all measurements at once
        self._campaign.add_measurements(df)

    @property
    def campaign(self) -> "Campaign":
        """Access the underlying BayBE Campaign."""
        return self._campaign

    @property
    def measurements(self) -> pd.DataFrame:
        """Get all measurements recorded by the campaign."""
        return self._campaign.measurements


class RecipeBayesianOptimizer(BayesianOptimizer):
    """Convenience class for optimizing reaction recipes.

    This is an alias for BayesianOptimizer with recipe-specific defaults.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective: ObjectiveFunction,
        n_initial: int = 5,
        n_iterations: int = 20,
        **kwargs,
    ):
        """Initialize the recipe optimizer.

        Args:
            search_space: The parameter space to search
            objective: The objective function to optimize
            n_initial: Number of initial random samples
            n_iterations: Number of optimization iterations
            **kwargs: Additional arguments passed to BayesianOptimizer
        """
        super().__init__(
            search_space=search_space,
            objective=objective,
            n_initial=n_initial,
            n_iterations=n_iterations,
            target_name="yield",
            **kwargs,
        )
