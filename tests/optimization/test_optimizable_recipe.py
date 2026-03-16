"""Tests for OptimizableRecipe."""

import pytest

from rxn_ca.optimization.optimizable_recipe import OptimizableRecipe
from rxn_ca.core.recipe import ReactionRecipe
from rxn_ca.core.heating import HeatingSchedule


class TestOptimizableRecipe:
    """Tests for OptimizableRecipe creation and conversion."""

    def test_create_recipe(self):
        """Can create a basic optimizable recipe."""
        recipe = OptimizableRecipe(
            precursors={"BaO": 0.5, "TiO2": 0.5},
            hold_temp=1000,
            hold_time=5,
        )

        assert recipe.hold_temp == 1000
        assert recipe.hold_time == 5
        assert recipe.precursors["BaO"] == 0.5
        assert recipe.precursors["TiO2"] == 0.5

    def test_hold_temp_must_be_divisible_by_50(self):
        """Hold temperature must be divisible by 50."""
        with pytest.raises(AssertionError):
            OptimizableRecipe(
                precursors={"BaO": 0.5},
                hold_temp=1025,  # Not divisible by 50
                hold_time=5,
            )

    def test_ramp_rate_conversion(self):
        """Ramp rate is converted to sweep_step_size."""
        recipe = OptimizableRecipe(
            precursors={"BaO": 0.5},
            hold_temp=1000,
            hold_time=5,
            ramp_rate=10.0,  # 10 K/step
        )

        # TEMP_STEP_SIZE = 50, so sweep_step_size = 50/10 = 5
        assert recipe.ramp_rate == 10.0
        assert recipe.sweep_step_size == 5

    def test_sweep_step_size_backward_compat(self):
        """Can still use sweep_step_size directly."""
        recipe = OptimizableRecipe(
            precursors={"BaO": 0.5},
            hold_temp=1000,
            hold_time=5,
            sweep_step_size=5,
        )

        # ramp_rate = 50/5 = 10
        assert recipe.sweep_step_size == 5
        assert recipe.ramp_rate == 10.0

    def test_to_recipe(self):
        """Can convert to ReactionRecipe."""
        opt_recipe = OptimizableRecipe(
            precursors={"BaO": 0.5, "TiO2": 0.5},
            hold_temp=1000,
            hold_time=5,
            simulation_size=10,
            num_simulations=3,
        )

        recipe = opt_recipe.to_recipe()

        assert isinstance(recipe, ReactionRecipe)
        assert recipe.simulation_size == 10
        assert recipe.num_realizations == 3
        assert recipe.reactant_amounts == {"BaO": 0.5, "TiO2": 0.5}

    def test_to_recipe_heating_schedule(self):
        """Converted recipe has correct heating schedule."""
        opt_recipe = OptimizableRecipe(
            precursors={"BaO": 0.5},
            hold_temp=1000,
            hold_time=5,
        )

        recipe = opt_recipe.to_recipe()
        heating_schedule = recipe.heating_schedule

        assert isinstance(heating_schedule, HeatingSchedule)
        # Should have sweep temps (300 to 1000) + hold steps
        temps = heating_schedule.all_temps
        assert 300 in temps  # Start temp
        assert 1000 in temps  # Hold temp

    def test_from_params(self):
        """Can create recipe from optimizer params dict."""
        params = {
            "hold_temp": 1000,
            "hold_time": 5,
            "ramp_rate": 10.0,
            "Ba_source_ratio": 0.5,
            "Ti_source_ratio": 0.5,
        }
        precursor_slots = {
            "Ba_source": "BaO",
            "Ti_source": "TiO2",
        }

        recipe = OptimizableRecipe.from_params(
            params=params,
            precursor_slots=precursor_slots,
            simulation_size=8,
            num_simulations=2,
        )

        assert recipe.hold_temp == 1000
        assert recipe.hold_time == 5
        assert recipe.ramp_rate == 10.0
        assert recipe.precursors["BaO"] == 0.5
        assert recipe.precursors["TiO2"] == 0.5
        assert recipe.simulation_size == 8
        assert recipe.num_simulations == 2

    def test_from_params_defaults(self):
        """from_params uses reasonable defaults."""
        params = {"hold_temp": 1000}
        precursor_slots = {"Ba_source": "BaO"}

        recipe = OptimizableRecipe.from_params(
            params=params,
            precursor_slots=precursor_slots,
        )

        assert recipe.hold_temp == 1000
        assert recipe.hold_time == 5  # Default from params.get
        assert recipe.precursors["BaO"] == 0.5  # Default ratio

    def test_repr(self):
        """Recipe has readable string representation."""
        recipe = OptimizableRecipe(
            precursors={"BaO": 0.5},
            hold_temp=1000,
            hold_time=5,
        )

        s = repr(recipe)
        assert "1000" in s
        assert "BaO" in s
