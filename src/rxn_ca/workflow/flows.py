"""Pre-built flows for common rxn-ca simulation workflows."""

from typing import Any, Dict, List, Optional, Union

from jobflow import Flow

from .jobs import setup_reaction_library, run_simulation
from .schemas import SimulationOutput


def create_simulation_flow(
    recipe: Union[Dict[str, Any], "ReactionRecipe"],
    chemical_system: str,
    ensure_phases: List[str] = None,
    temperatures: List[float] = None,
    metastability_cutoff: float = 0.1,
    exclude_theoretical: bool = True,
    save_to_file: bool = True,
    metadata: Dict[str, Any] = None,
    name: str = "rxn_ca_simulation",
) -> Flow:
    """Create a flow that sets up a reaction library and runs a simulation.

    This is a convenience function that links setup_reaction_library and
    run_simulation jobs together. Use this when you have a single simulation
    to run. For multiple simulations in the same chemical system, create
    the jobs manually to share the reaction library setup.

    Args:
        recipe: ReactionRecipe or its dict representation
        chemical_system: Chemical system string (e.g., "Ba-Ti-O")
        ensure_phases: Phases to ensure are in the reaction library
        temperatures: Temperatures to score reactions at. If None, uses
            the temperatures from the recipe's heating schedule.
        metastability_cutoff: Energy above hull cutoff for phases
        exclude_theoretical: Whether to exclude theoretical phases
        save_to_file: If True, save reaction library and results to files
        metadata: Optional metadata to attach to the simulation output
        name: Name for the flow

    Returns:
        Flow with setup and simulation jobs, outputting SimulationOutput

    Example:
        >>> from rxn_ca.core.recipe import ReactionRecipe
        >>> from rxn_ca.core.heating import HeatingSchedule, HeatingStep
        >>> from rxn_ca.workflow import create_simulation_flow
        >>>
        >>> # Create a recipe
        >>> heating_steps = HeatingStep.sweep(t0=298, tf=1273, stage_length=1, temp_step_size=50)
        >>> heating_sched = HeatingSchedule.build(heating_steps)
        >>> recipe = ReactionRecipe(
        ...     heating_schedule=heating_sched,
        ...     reactant_amounts={"BaCO3": 1.0, "TiO2": 1.0},
        ...     simulation_size=15,
        ... )
        >>>
        >>> # Create and run the flow
        >>> flow = create_simulation_flow(
        ...     recipe=recipe,
        ...     chemical_system="Ba-C-O-Ti",
        ...     ensure_phases=["BaTiO3", "BaCO3", "TiO2"],
        ... )
    """
    # Convert recipe to dict if needed
    if hasattr(recipe, "as_dict"):
        recipe_dict = recipe.as_dict()
        heating_schedule = recipe.heating_schedule
    else:
        recipe_dict = recipe
        # Need to reconstruct heating schedule to get temperatures
        from rxn_ca.core.recipe import ReactionRecipe

        heating_schedule = ReactionRecipe.from_dict(recipe_dict).heating_schedule

    # Get temperatures from recipe if not provided
    if temperatures is None:
        temperatures = heating_schedule.all_temps

    # Create setup job
    setup_job = setup_reaction_library(
        chemical_system=chemical_system,
        temperatures=temperatures,
        ensure_phases=ensure_phases,
        metastability_cutoff=metastability_cutoff,
        exclude_theoretical=exclude_theoretical,
        save_to_file=save_to_file,
    )
    setup_job.name = f"setup_{chemical_system}"

    # Create simulation job
    sim_job = run_simulation(
        recipe_dict=recipe_dict,
        reaction_library_data=setup_job.output,
        save_to_file=save_to_file,
        metadata=metadata,
    )
    sim_job.name = f"simulate_{chemical_system}"

    return Flow([setup_job, sim_job], output=sim_job.output, name=name)


def create_multi_simulation_flow(
    recipes: List[Union[Dict[str, Any], "ReactionRecipe"]],
    chemical_system: str,
    ensure_phases: List[str] = None,
    temperatures: List[float] = None,
    metastability_cutoff: float = 0.1,
    exclude_theoretical: bool = True,
    save_to_file: bool = True,
    metadata_list: List[Dict[str, Any]] = None,
    name: str = "rxn_ca_multi_simulation",
) -> Flow:
    """Create a flow for multiple simulations sharing a reaction library.

    This is more efficient than creating separate flows when running multiple
    simulations in the same chemical system, as the expensive reaction library
    setup is only done once.

    Args:
        recipes: List of ReactionRecipes or their dict representations
        chemical_system: Chemical system string (e.g., "Ba-Ti-O")
        ensure_phases: Phases to ensure are in the reaction library
        temperatures: Temperatures to score reactions at. If None, collects
            all temperatures needed by all recipes.
        metastability_cutoff: Energy above hull cutoff for phases
        exclude_theoretical: Whether to exclude theoretical phases
        save_to_file: If True, save reaction library and results to files
        metadata_list: Optional list of metadata dicts, one per recipe
        name: Name for the flow

    Returns:
        Flow with one setup job and multiple simulation jobs,
        outputting list of SimulationOutput
    """
    from rxn_ca.core.recipe import ReactionRecipe

    # Convert recipes to dicts and collect temperatures
    recipe_dicts = []
    all_temps = set()

    for recipe in recipes:
        if hasattr(recipe, "as_dict"):
            recipe_dicts.append(recipe.as_dict())
            all_temps.update(recipe.heating_schedule.all_temps)
        else:
            recipe_dicts.append(recipe)
            r = ReactionRecipe.from_dict(recipe)
            all_temps.update(r.heating_schedule.all_temps)

    # Use provided temperatures or collected ones
    if temperatures is None:
        temperatures = sorted(all_temps)

    # Create setup job (shared across all simulations)
    setup_job = setup_reaction_library(
        chemical_system=chemical_system,
        temperatures=temperatures,
        ensure_phases=ensure_phases,
        metastability_cutoff=metastability_cutoff,
        exclude_theoretical=exclude_theoretical,
        save_to_file=save_to_file,
    )
    setup_job.name = f"setup_{chemical_system}"

    jobs = [setup_job]
    outputs = []

    # Create simulation jobs
    for i, recipe_dict in enumerate(recipe_dicts):
        metadata = metadata_list[i] if metadata_list else None

        sim_job = run_simulation(
            recipe_dict=recipe_dict,
            reaction_library_data=setup_job.output,
            save_to_file=save_to_file,
            metadata=metadata,
        )
        sim_job.name = f"simulate_{i}"
        jobs.append(sim_job)
        outputs.append(sim_job.output)

    return Flow(jobs, output=outputs, name=name)
