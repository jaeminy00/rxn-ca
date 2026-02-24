from typing import Dict, Optional, Union

from ..core.recipe import ReactionRecipe
from ..core.heating import HeatingSchedule, HeatingStep


class OptimizableRecipe:
    """A recipe that can be parameterized for optimization.

    This class wraps the ReactionRecipe with parameters that are suitable
    for optimization: hold temperature, hold time, ramp rate, and precursor
    amounts.

    The ramp_rate parameter controls heating rate in K/step. Internally this
    is converted to sweep_step_size (simulation steps per temperature step).
    """

    TEMP_STEP_SIZE = 50  # Temperature increment per heating step (K)
    DEFAULT_RAMP_RATE = 10.0  # Default heating rate (K/step)

    def __init__(
        self,
        precursors: Dict[str, float],
        hold_temp: int,
        hold_time: int,
        ramp_rate: Optional[float] = None,
        sweep_step_size: Optional[int] = None,
        simulation_size: int = 10,
        num_simulations: int = 3,
    ):
        """Initialize an optimizable recipe.

        Args:
            precursors: Dictionary mapping precursor formulas to amounts
            hold_temp: Hold temperature in K (must be divisible by 50)
            hold_time: Duration at hold temperature (simulation steps)
            ramp_rate: Heating rate in K/step. If provided, sweep_step_size
                is calculated as TEMP_STEP_SIZE / ramp_rate
            sweep_step_size: Simulation steps per temperature step. Deprecated,
                use ramp_rate instead. If both provided, ramp_rate takes precedence.
            simulation_size: Size of the simulation grid
            num_simulations: Number of simulation realizations to run
        """
        assert hold_temp % 50 == 0, "Optimizable recipes use hold temperatures divisible by 50"

        self.hold_temp = hold_temp
        self.precursors = precursors
        self.hold_time = hold_time
        self.simulation_size = simulation_size
        self.num_simulations = num_simulations

        # Handle ramp_rate vs sweep_step_size (backward compatibility)
        if ramp_rate is not None:
            self.ramp_rate = ramp_rate
            self.sweep_step_size = max(1, int(self.TEMP_STEP_SIZE / ramp_rate))
        elif sweep_step_size is not None:
            self.sweep_step_size = sweep_step_size
            self.ramp_rate = self.TEMP_STEP_SIZE / sweep_step_size
        else:
            # Default behavior
            self.ramp_rate = self.DEFAULT_RAMP_RATE
            self.sweep_step_size = max(1, int(self.TEMP_STEP_SIZE / self.ramp_rate))

    @classmethod
    def from_params(
        cls,
        params: Dict[str, Union[float, str]],
        precursor_slots: Optional[Dict[str, str]] = None,
        simulation_size: int = 10,
        num_simulations: int = 3,
    ) -> "OptimizableRecipe":
        """Create an OptimizableRecipe from optimizer parameters.

        Args:
            params: Dictionary of parameter values from optimizer, containing:
                - hold_temp: Hold temperature
                - hold_time: Hold time
                - ramp_rate (optional): Heating rate
                - *_ratio: Precursor ratios (e.g., Ba_source_ratio)
            precursor_slots: Dictionary mapping slot names to selected precursors
                (e.g., {"Ba_source": "BaCO3", "Ti_source": "TiO2"})
            simulation_size: Simulation grid size
            num_simulations: Number of realizations

        Returns:
            Configured OptimizableRecipe
        """
        # Extract temperature parameters
        hold_temp = int(params.get("hold_temp", 1000))
        hold_time = int(params.get("hold_time", 5))
        ramp_rate = params.get("ramp_rate", cls.DEFAULT_RAMP_RATE)

        # Build precursors dict from slots and ratios
        precursors = {}
        if precursor_slots:
            for slot_name, precursor in precursor_slots.items():
                ratio_key = f"{slot_name}_ratio"
                ratio = params.get(ratio_key, 0.5)
                precursors[precursor] = ratio

        # Also check for direct precursor entries in params
        for key, value in params.items():
            if key.endswith("_ratio"):
                continue
            if key in ["hold_temp", "hold_time", "ramp_rate"]:
                continue
            # Could be a precursor from a slot that's directly in params
            if precursor_slots and key in precursor_slots.values():
                continue
            # Check if it's a precursor slot selection
            if precursor_slots and key in precursor_slots:
                # This is the selected precursor for a slot
                selected = value
                ratio_key = f"{key}_ratio"
                ratio = params.get(ratio_key, 0.5)
                precursors[selected] = ratio

        return cls(
            precursors=precursors,
            hold_temp=hold_temp,
            hold_time=hold_time,
            ramp_rate=ramp_rate,
            simulation_size=simulation_size,
            num_simulations=num_simulations,
        )

    def to_recipe(self) -> ReactionRecipe:
        """Convert to a ReactionRecipe for simulation."""
        # Use HeatingSchedule.build() because sweep() and hold() return lists
        heating_schedule = HeatingSchedule.build(
            HeatingStep.sweep(
                300,
                self.hold_temp,
                stage_length=self.sweep_step_size,
                temp_step_size=OptimizableRecipe.TEMP_STEP_SIZE,
            ),
            HeatingStep.hold(self.hold_temp, self.hold_time),
        )
        return ReactionRecipe(
            heating_schedule=heating_schedule,
            reactant_amounts=self.precursors,
            simulation_size=self.simulation_size,
            num_realizations=self.num_simulations,
        )

    def __repr__(self) -> str:
        return (
            f"OptimizableRecipe(hold_temp={self.hold_temp}, "
            f"hold_time={self.hold_time}, ramp_rate={self.ramp_rate:.1f}, "
            f"precursors={self.precursors})"
        )

