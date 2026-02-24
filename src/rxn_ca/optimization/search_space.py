"""Search space definition for optimization."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    Parameter,
    ContinuousParameter,
    DiscreteParameter,
    CategoricalParameter,
    PrecursorSlotParameter,
    ParameterType,
)


@dataclass
class SearchSpace:
    """Defines the parameter space for optimization with a fluent builder API.

    Example:
        search_space = (SearchSpace()
            .add_temperature_range(800, 1400, step=50)
            .add_hold_time_range(1, 15)
            .add_ramp_rate_range(5.0, 20.0)
            .add_precursor_slot("Ba_source", ["BaCO3", "BaO", "Ba(OH)2"])
            .add_precursor_ratio("Ba_source", 0.4, 0.6)
        )
    """

    parameters: List[Parameter] = field(default_factory=list)
    _param_names: set = field(default_factory=set, repr=False)

    def _add_parameter(self, param: Parameter) -> "SearchSpace":
        """Add a parameter, ensuring unique names."""
        if param.name in self._param_names:
            raise ValueError(f"Parameter '{param.name}' already exists in search space")
        self.parameters.append(param)
        self._param_names.add(param.name)
        return self

    def add_temperature_range(
        self,
        low: float,
        high: float,
        step: float = 50,
        name: str = "hold_temp",
    ) -> "SearchSpace":
        """Add a hold temperature parameter.

        Args:
            low: Minimum temperature (K)
            high: Maximum temperature (K)
            step: Temperature step size (default 50K)
            name: Parameter name (default "hold_temp")

        Returns:
            self for method chaining
        """
        param = DiscreteParameter(name=name, low=low, high=high, step=step)
        return self._add_parameter(param)

    def add_hold_time_range(
        self,
        low: float,
        high: float,
        name: str = "hold_time",
    ) -> "SearchSpace":
        """Add a hold time parameter.

        Args:
            low: Minimum hold time (simulation steps)
            high: Maximum hold time (simulation steps)
            name: Parameter name (default "hold_time")

        Returns:
            self for method chaining
        """
        param = ContinuousParameter(name=name, low=low, high=high)
        return self._add_parameter(param)

    def add_ramp_rate_range(
        self,
        low: float,
        high: float,
        name: str = "ramp_rate",
    ) -> "SearchSpace":
        """Add a ramp rate parameter (heating rate in K/step).

        Args:
            low: Minimum ramp rate (K/step)
            high: Maximum ramp rate (K/step)
            name: Parameter name (default "ramp_rate")

        Returns:
            self for method chaining
        """
        param = ContinuousParameter(name=name, low=low, high=high)
        return self._add_parameter(param)

    def add_precursor_slot(
        self,
        name: str,
        candidates: List[str],
    ) -> "SearchSpace":
        """Add a precursor selection parameter.

        Uses chemical encoding (e.g., MORDRED fingerprints) for
        chemically-informed optimization.

        Args:
            name: Parameter name (e.g., "Ba_source")
            candidates: List of chemical formulas (e.g., ["BaCO3", "BaO"])

        Returns:
            self for method chaining
        """
        param = PrecursorSlotParameter(name=name, candidates=candidates)
        return self._add_parameter(param)

    def add_precursor_ratio(
        self,
        name: str,
        low: float,
        high: float,
    ) -> "SearchSpace":
        """Add a precursor ratio parameter.

        Args:
            name: Parameter name (should match a precursor slot name + "_ratio")
            low: Minimum ratio (typically 0-1)
            high: Maximum ratio (typically 0-1)

        Returns:
            self for method chaining
        """
        ratio_name = f"{name}_ratio" if not name.endswith("_ratio") else name
        param = ContinuousParameter(name=ratio_name, low=low, high=high)
        return self._add_parameter(param)

    def add_continuous(
        self,
        name: str,
        low: float,
        high: float,
    ) -> "SearchSpace":
        """Add a generic continuous parameter.

        Args:
            name: Parameter name
            low: Minimum value
            high: Maximum value

        Returns:
            self for method chaining
        """
        param = ContinuousParameter(name=name, low=low, high=high)
        return self._add_parameter(param)

    def add_discrete(
        self,
        name: str,
        low: float,
        high: float,
        step: float,
    ) -> "SearchSpace":
        """Add a generic discrete parameter.

        Args:
            name: Parameter name
            low: Minimum value
            high: Maximum value
            step: Step size

        Returns:
            self for method chaining
        """
        param = DiscreteParameter(name=name, low=low, high=high, step=step)
        return self._add_parameter(param)

    def add_categorical(
        self,
        name: str,
        choices: List[str],
    ) -> "SearchSpace":
        """Add a generic categorical parameter.

        Args:
            name: Parameter name
            choices: List of valid choices

        Returns:
            self for method chaining
        """
        param = CategoricalParameter(name=name, choices=choices)
        return self._add_parameter(param)

    def get_parameter(self, name: str) -> Optional[Parameter]:
        """Get a parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None

    def get_parameters_by_type(self, param_type: ParameterType) -> List[Parameter]:
        """Get all parameters of a specific type."""
        return [p for p in self.parameters if p.param_type == param_type]

    @property
    def continuous_parameters(self) -> List[ContinuousParameter]:
        """Get all continuous parameters."""
        return self.get_parameters_by_type(ParameterType.CONTINUOUS)

    @property
    def discrete_parameters(self) -> List[DiscreteParameter]:
        """Get all discrete parameters."""
        return self.get_parameters_by_type(ParameterType.DISCRETE)

    @property
    def categorical_parameters(self) -> List[CategoricalParameter]:
        """Get all categorical parameters."""
        return self.get_parameters_by_type(ParameterType.CATEGORICAL)

    @property
    def precursor_parameters(self) -> List[PrecursorSlotParameter]:
        """Get all precursor slot parameters."""
        return self.get_parameters_by_type(ParameterType.PRECURSOR_SLOT)

    @property
    def param_names(self) -> List[str]:
        """Get all parameter names in order."""
        return [p.name for p in self.parameters]

    def validate(self, params: Dict[str, Any]) -> bool:
        """Validate a parameter configuration.

        Args:
            params: Dictionary of parameter values

        Returns:
            True if valid, raises ValueError otherwise
        """
        for param in self.parameters:
            if param.name not in params:
                raise ValueError(f"Missing parameter: {param.name}")
            if not param.validate(params[param.name]):
                raise ValueError(
                    f"Invalid value {params[param.name]} for parameter {param.name}"
                )
        return True

    def sample_random(self, n: int = 1) -> List[Dict[str, Any]]:
        """Sample random parameter configurations.

        Args:
            n: Number of configurations to sample

        Returns:
            List of parameter dictionaries
        """
        import numpy as np
        import random

        samples = []
        for _ in range(n):
            config = {}
            for param in self.parameters:
                if isinstance(param, ContinuousParameter):
                    config[param.name] = np.random.uniform(param.low, param.high)
                elif isinstance(param, DiscreteParameter):
                    config[param.name] = random.choice(param.values)
                elif isinstance(param, (CategoricalParameter, PrecursorSlotParameter)):
                    config[param.name] = random.choice(param.choices)
            samples.append(config)
        return samples

    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get bounds for continuous and discrete parameters.

        Returns:
            Dictionary mapping parameter names to (low, high) tuples
        """
        bounds = {}
        for param in self.parameters:
            if isinstance(param, (ContinuousParameter, DiscreteParameter)):
                bounds[param.name] = (param.low, param.high)
        return bounds

    def __len__(self) -> int:
        return len(self.parameters)

    @property
    def parameter_names(self) -> List[str]:
        """Get list of all parameter names."""
        return [p.name for p in self.parameters]

    def __repr__(self) -> str:
        param_strs = []
        for p in self.parameters:
            if isinstance(p, ContinuousParameter):
                param_strs.append(f"{p.name}: [{p.low}, {p.high}]")
            elif isinstance(p, DiscreteParameter):
                param_strs.append(f"{p.name}: [{p.low}, {p.high}, step={p.step}]")
            elif isinstance(p, (CategoricalParameter, PrecursorSlotParameter)):
                param_strs.append(f"{p.name}: {p.choices}")
        return f"SearchSpace({', '.join(param_strs)})"
