from ..core.recipe import ReactionRecipe

from ..reactions import ReactionLibrary
from ..phases import SolidPhaseSet
from ..computing.schemas.ca_result_schema import RxnCAResultDoc, MultiRxnCAResultDoc, compress_doc, get_metadata_from_results
from ..core.reaction_result import ReactionResult

from rxn_network.reactions.reaction_set import ReactionSet
from pylattica.core import Simulation

import multiprocessing as mp
import ray
from jobflow import job

from .single_sim import run_single_sim
from .get_scored_rxns import get_scored_rxns

_reaction_lib = "reaction_lib"
_recipe = "recipe"
_initial_simulation = "initial_simulation"

_JOBSTORE_OBJECTS = [ReactionLibrary, ReactionResult, RxnCAResultDoc] 
#have to include RxnCAResultDoc because it contains lists of ReactionResult objects, 
#and jobflow DOES NOT automatically recursively search inside each list item.
#solution is to use a MultiRxnCAResultDoc as the output schema and 
#store the RxnCAResultDoc objects in the data objects list.

def _get_result(_):

    result: RxnCAResultDoc = run_single_sim(
        mp_globals[_recipe],
        reaction_lib=mp_globals.get(_reaction_lib),
        initial_simulation=mp_globals.get(_initial_simulation)
    )
    return result.results[0]


def run_sim_parallel(recipe: ReactionRecipe,
                     base_reactions: ReactionSet = None,
                     reaction_lib: ReactionLibrary = None,
                     initial_simulation: Simulation = None,
                     phase_set: SolidPhaseSet = None):

    print("================= RETRIEVING AND SCORING REACTIONS =================")

    if base_reactions is None and reaction_lib is None:
        raise ValueError("Must provide either base_reactions or reaction_lib")

    if reaction_lib is None:
        reaction_lib: ReactionLibrary = get_scored_rxns(
            base_reactions,
            heating_sched=recipe.heating_schedule,
            exclude_phases=recipe.exclude_phases,
            exclude_theoretical=recipe.exclude_theoretical,
            scorer_class=recipe.get_score_class(),
            phase_set=phase_set
        )

    print()
    print()
    print()

    print(f'================= RUNNING SIMULATION w/ {recipe.num_realizations} REALIZATIONS =================')


    global mp_globals

    mp_globals = {
        _reaction_lib: reaction_lib,
        _recipe: recipe,
        _initial_simulation: initial_simulation
    }

    with mp.get_context("fork").Pool(recipe.num_realizations) as pool:
        results = pool.map(_get_result, [_ for _ in range(recipe.num_realizations)])

    good_results = [res for res in results if res is not None]
    print(f'{len(good_results)} results achieved out of {len(results)}')

    result_doc = RxnCAResultDoc(
        recipe=recipe,
        results=good_results,
        reaction_library=reaction_lib,
        phases=reaction_lib.phases
    )

    return result_doc


@ray.remote
def _run_single_realization(
    recipe: ReactionRecipe,
    recipe_index: int,
    realization_id: int,
    reaction_lib: ReactionLibrary,
    initial_sim: Simulation = None,
    compress: bool = True,
    num_steps: int = 500,
    **sim_kwargs
) -> tuple[int, int, any]:
    """
    Run a single realization of a recipe using Ray.
    
    Args:
        recipe: The ReactionRecipe to run
        recipe_index: Index of the recipe in the original list (for grouping results)
        realization_id: ID of this realization (0 to num_realizations-1)
        reaction_lib: The ReactionLibrary to use
        initial_sim: Optional initial Simulation
        **sim_kwargs: Additional kwargs to pass to run_single_sim
    
    Returns:
        Tuple of (recipe_index, realization_id, result)
    """
    result_doc = run_single_sim(
        recipe, 
        reaction_lib=reaction_lib,
        initial_simulation=initial_sim,
        **sim_kwargs
    )
    
    result_doc.metadata = get_metadata_from_results(result_doc.results)
    
    if compress:
        result_doc = compress_doc(result_doc, num_steps)
    
    return (recipe_index, realization_id, result_doc.results[0])


def run_multi_recipe_parallel_ray(
    recipes: list[ReactionRecipe],
    reaction_libraries: list[ReactionLibrary],
    initial_simulations: list[Simulation] = None,
    num_workers: int = None,
    memory_per_task_gb: float = None,
    cpus_per_task: int = 1,
    **kwargs
) -> list[RxnCAResultDoc]:
    """
    Run multiple recipes in parallel using Ray, with all realizations for each recipe
    distributed across available cores. 
    For each recipe[i], uses reaction_libraries[i] and optionally initial_simulations[i]
    to run all realizations for that recipe. 
    
    Args:
        recipes: List of ReactionRecipe objects
        reaction_libraries: List of ReactionLibrary objects (one per recipe)
        initial_simulations: Optional list of Simulation objects (one per recipe)
        num_workers: Number of CPU cores to use (default: all available on the current node)
        memory_per_task_gb: Memory requirement per task in GB (optional, for memory-aware scheduling)
        cpus_per_task: Number of CPUs per task (default: 1, since simulations typically don't benefit from multiple CPUs)
        **kwargs: Additional kwargs to pass to run_single_sim (e.g., phase_set)
    
    Returns:
        List of RxnCAResultDoc objects, one per recipe
    """
    MAX_CPUS_PER_TASK = 4 # Any more does not seem to improve performance
    
    # Validate inputs
    if len(reaction_libraries) != len(recipes):
        raise ValueError(
            f"Number of reaction_libraries ({len(reaction_libraries)}) must match "
            f"number of recipes ({len(recipes)})"
        )
    
    if initial_simulations is not None and len(initial_simulations) != len(recipes):
        raise ValueError(
            f"Number of initial_simulations ({len(initial_simulations)}) must match "
            f"number of recipes ({len(recipes)})"
        )
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        init_kwargs = {}
        if num_workers:
            init_kwargs['num_cpus'] = num_workers
        ray.init(**init_kwargs)
    
    # Calculate total realizations
    total_realizations = sum(recipe.num_realizations for recipe in recipes)
    
    # Determine CPUs per task
    # Default to 1 CPU per task since simulations typically don't benefit from multiple CPUs
    # and the overhead can actually slow things down
    cpus_per_task = max(1, min(cpus_per_task, MAX_CPUS_PER_TASK))
    
    available_cpus = int(ray.cluster_resources().get('CPU', 1))
    print(f"Using {cpus_per_task} CPU(s) per task (total: {total_realizations} realizations, {available_cpus} CPUs available)")
    
    # Generate all tasks: for each recipe, create tasks for all its realizations
    task_refs = []
    
    for recipe_idx, recipe in enumerate(recipes):
        num_realizations = recipe.num_realizations
        
        task_options = {"num_cpus": cpus_per_task}
        if memory_per_task_gb:
            task_options["memory"] = int(memory_per_task_gb * 1_000_000_000)  # Convert GB to bytes
        
        run_task = _run_single_realization.options(**task_options)
        
        # Launch all realizations for this recipe
        for realization_id in range(num_realizations):
            task_ref = run_task.remote(
                recipe=recipe,
                recipe_index=recipe_idx,
                realization_id=realization_id,
                reaction_lib=reaction_libraries[recipe_idx],
                initial_sim=initial_simulations[recipe_idx] if initial_simulations else None,
                **kwargs
            )
            task_refs.append(task_ref)
    
    print(f"================= RUNNING {total_realizations} REALIZATIONS ACROSS {len(recipes)} RECIPES ON {ray.cluster_resources()['CPU']} CORES =================")
    print(f"Distributing tasks across available cores...")
    
    # Execute all tasks (Ray handles scheduling and load balancing)
    results = ray.get(task_refs)
    
    # Group results by recipe
    recipe_results = {}
    for recipe_idx, realization_id, result in results:
        if recipe_idx not in recipe_results:
            recipe_results[recipe_idx] = []
        recipe_results[recipe_idx].append(result)
    
    result_docs = []
    for recipe_idx, recipe in enumerate(recipes):
        good_results = [res for res in recipe_results.get(recipe_idx, []) if res is not None]
        num_good = len(good_results)
        num_total = recipe.num_realizations
        
        print(f"Recipe {recipe_idx}: {num_good}/{num_total} successful realizations")
        
        result_doc = RxnCAResultDoc(
            recipe=recipe,
            results=good_results,
            reaction_library=reaction_libraries[recipe_idx],
            phases=reaction_libraries[recipe_idx].phases
        )
        result_docs.append(result_doc)
    
    return result_docs

@job(data=_JOBSTORE_OBJECTS)
def run_multi_recipe_job(recipes: list[ReactionRecipe],
                                      reaction_libraries: list[ReactionLibrary],
                                      initial_simulations: list[Simulation] = None,
                                      metadata: dict = None,
                                      **kwargs) -> MultiRxnCAResultDoc:
    result_docs = run_multi_recipe_parallel_ray(recipes, reaction_libraries, initial_simulations, **kwargs)
    return MultiRxnCAResultDoc(recipes=recipes, results=result_docs, metadata=metadata)