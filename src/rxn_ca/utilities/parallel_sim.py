from ..core.recipe import ReactionRecipe

from ..reactions import ReactionLibrary
from ..phases import SolidPhaseSet
from ..computing.schemas.ca_result_schema import RxnCAResultDoc

from rxn_network.reactions.reaction_set import ReactionSet
from pylattica.core import Simulation

import multiprocessing as mp

from .single_sim import run_single_sim
from .get_scored_rxns import get_scored_rxns

_reaction_lib = "reaction_lib"
_recipe = "recipe"
_initial_simulation = "initial_simulation"
_compress_freq = "compress_freq"
_live_compress = "live_compress"


def _get_result(_):

    result: RxnCAResultDoc = run_single_sim(
        mp_globals[_recipe],
        reaction_lib=mp_globals.get(_reaction_lib),
        initial_simulation=mp_globals.get(_initial_simulation),
        compress_freq=mp_globals.get(_compress_freq, 1),
        live_compress=mp_globals.get(_live_compress, False),
    )
    return result.results[0]


def run_sim_parallel(recipe: ReactionRecipe,
                     base_reactions: ReactionSet = None,
                     reaction_lib: ReactionLibrary = None,
                     initial_simulation: Simulation = None,
                     phase_set: SolidPhaseSet = None,
                     existing_lib: ReactionLibrary = None,
                     compress_freq: int = 1,
                     live_compress: bool = False):
    """Run simulation with multiple realizations in parallel.

    Args:
        recipe: The reaction recipe to simulate
        base_reactions: Base reactions for scoring (required if reaction_lib not provided)
        reaction_lib: Pre-computed complete reaction library (skips all scoring)
        initial_simulation: Initial simulation state (optional)
        phase_set: Phase set for the system
        existing_lib: Existing library with some temps already scored.
            New temps will be added to this library incrementally.
        compress_freq: Interval for storing frames when live_compress is True.
        live_compress: If True, store full state snapshots at compress_freq
            intervals instead of diffs. Avoids slow reconstruction during analysis.

    Returns:
        RxnCAResultDoc with averaged results from all realizations
    """
    if base_reactions is None and reaction_lib is None:
        raise ValueError("Must provide either base_reactions or reaction_lib")

    if reaction_lib is None:
        # Get required temperatures
        required_temps = recipe.heating_schedule.all_temps
        cached_temps = existing_lib.temps if existing_lib else []
        new_temps = [t for t in required_temps if t not in cached_temps]

        if new_temps:
            print("================= RETRIEVING AND SCORING REACTIONS =================")
            if existing_lib:
                print(f"    (Reusing {len(cached_temps)} cached temps, scoring {len(new_temps)} new temps)")

        reaction_lib: ReactionLibrary = get_scored_rxns(
            base_reactions,
            heating_sched=recipe.heating_schedule,
            scorer_class=recipe.get_score_class(),
            phase_set=phase_set,
            existing_lib=existing_lib,
        )

    print()
    print()
    print()

    print(f'================= RUNNING SIMULATION w/ {recipe.num_realizations} REALIZATIONS =================')


    global mp_globals

    mp_globals = {
        _reaction_lib: reaction_lib,
        _recipe: recipe,
        _initial_simulation: initial_simulation,
        _compress_freq: compress_freq,
        _live_compress: live_compress,
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

