from typing import Dict, Optional, Union

from ..core.recipe import ReactionRecipe
from ..core.heating import HeatingSchedule, HeatingStep


class OptimizableRecipe:
    """A recipe that can be parameterized for optimization.

    This class wraps the ReactionRecipe with parameters suitable for optimization.

    The heating profile is always: ramp from 300K to hold_temp in 100K steps,
    then hold at hold_temp for hold_time steps.

    Parameters:
        - hold_temp: Target plateau temperature (K), must be divisible by 100
        - hold_time: Duration at plateau (simulation steps)
        - ramp_step_time: Duration at each 100K step during ramp (simulation steps)
        - precursors: Dictionary of precursor formulas to amounts
    """

    TEMP_STEP_SIZE = 100  # Temperature increment per heating step (K)
    DEFAULT_RAMP_STEP_TIME = 1  # Default time at each ramp step

    def __init__(
        self,
        precursors: Dict[str, float],
        hold_temp: int,
        hold_time: int,
        ramp_step_time: int = None,
        simulation_size: int = 10,
        num_simulations: int = 3,
    ):
        """Initialize an optimizable recipe.

        Args:
            precursors: Dictionary mapping precursor formulas to amounts
            hold_temp: Target plateau temperature in K (must be divisible by 100)
            hold_time: Duration at plateau temperature (simulation steps)
            ramp_step_time: Duration at each 100K step during ramp (simulation steps)
            simulation_size: Size of the simulation grid
            num_simulations: Number of simulation realizations to run
        """
        assert hold_temp % 100 == 0, "hold_temp must be divisible by 100"

        self.hold_temp = hold_temp
        self.precursors = precursors
        self.hold_time = hold_time
        self.simulation_size = simulation_size
        self.num_simulations = num_simulations
        self.ramp_step_time = ramp_step_time if ramp_step_time is not None else self.DEFAULT_RAMP_STEP_TIME

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
                - hold_temp: Target plateau temperature (K)
                - hold_time: Duration at plateau (simulation steps)
                - ramp_step_time (optional): Duration at each ramp step
                - *_ratio: Precursor ratios (e.g., Ba_source_ratio)
            precursor_slots: Dictionary mapping slot names to selected precursors
                (e.g., {"Ba_source": "BaCO3", "Ti_source": "TiO2"})
            simulation_size: Simulation grid size
            num_simulations: Number of realizations

        Returns:
            Configured OptimizableRecipe
        """
        # Extract heating parameters
        hold_temp = int(params.get("hold_temp", 1000))
        hold_time = int(params.get("hold_time", 5))
        ramp_step_time = int(params.get("ramp_step_time", cls.DEFAULT_RAMP_STEP_TIME))

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
            if key in ["hold_temp", "hold_time", "ramp_step_time"]:
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
            ramp_step_time=ramp_step_time,
            simulation_size=simulation_size,
            num_simulations=num_simulations,
        )

    def to_recipe(self) -> ReactionRecipe:
        """Convert to a ReactionRecipe for simulation."""
        # Use HeatingSchedule.build() because sweep() returns a list
        # For hold, create a single step with the full duration
        heating_schedule = HeatingSchedule.build(
            HeatingStep.sweep(
                300,
                self.hold_temp,
                stage_length=self.ramp_step_time,
                temp_step_size=OptimizableRecipe.TEMP_STEP_SIZE,
            ),
            HeatingStep(self.hold_time, self.hold_temp),  # Single hold step
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
            f"hold_time={self.hold_time}, ramp_step_time={self.ramp_step_time}, "
            f"precursors={self.precursors})"
        )

