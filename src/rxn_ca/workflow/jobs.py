"""Jobflow jobs for rxn-ca simulations.

These jobs can be used standalone or composed into flows for running
rxn-ca simulations in workflow frameworks.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from jobflow import Response, job
from monty.json import MontyEncoder

from .schemas import ReactionLibraryData, SimulationOutput
from ..utilities.bayesian_helpers import build_recipe_from_params, _score_result


def _build_reaction_library(
    chemical_system: str,
    temperatures: List[float],
    ensure_phases: List[str] = None,
    metastability_cutoff: float = 0.1,
    exclude_theoretical: bool = True,
    **entry_kwargs
) -> tuple:
    """Build phase set and reaction library for a chemical system.

    Args:
        chemical_system: Chemical system string (e.g., "Ba-Ti-O")
        temperatures: List of temperatures (K) to score reactions at
        ensure_phases: List of phase formulas that MUST be included
        metastability_cutoff: Energy above hull cutoff for phases
        exclude_theoretical: Whether to exclude theoretical phases

    Returns:
        Tuple of (phase_set, reaction_lib)
    """
    from rxn_ca.utilities.get_entries import get_entries
    from rxn_ca.phases import SolidPhaseSet
    from rxn_ca.utilities.get_scored_rxns import get_scored_rxns
    from rxn_network.enumerators.basic import BasicEnumerator
    from rxn_network.enumerators.minimize import MinimizeGibbsEnumerator
    from rxn_network.enumerators.utils import run_enumerators

    entries = get_entries(
        chem_sys=chemical_system,
        metastability_cutoff=metastability_cutoff,
        ensure_phases=ensure_phases or [],
        exclude_theoretical_phases=exclude_theoretical,
        **entry_kwargs
    )
    print(f"Got {len(entries)} entries for {chemical_system}")
    if ensure_phases:
        print(f"  (ensured inclusion of: {ensure_phases})")

    phase_set = SolidPhaseSet.from_entry_set(entries)

    enumerators = [MinimizeGibbsEnumerator(), BasicEnumerator()]
    rxn_set = run_enumerators(enumerators, entries)
    print(f"Enumerated {len(rxn_set)} reactions")

    temp_rxn_mapping = rxn_set.compute_at_temperatures(temperatures)

    reaction_lib = get_scored_rxns(
        rxn_set,
        temps=temperatures,
        phase_set=phase_set,
        rxns_at_temps=temp_rxn_mapping,
        parallel=True,
    )

    return phase_set, reaction_lib


@job
def setup_reaction_library(
    chemical_system: str,
    temperatures: List[float],
    ensure_phases: List[str] = None,
    metastability_cutoff: float = 0.1,
    exclude_theoretical: bool = True,
    save_to_file: bool = True,
    **entry_kwargs
) -> ReactionLibraryData:
    """Set up phase set and reaction library for a chemical system.

    This job fetches thermodynamic data from Materials Project, enumerates
    possible reactions, and scores them at each temperature. This is typically
    the most expensive step and can be shared across multiple simulations
    in the same chemical system.

    Args:
        chemical_system: Chemical system string (e.g., "Ba-Ti-O")
        temperatures: List of temperatures (K) to score reactions at
        ensure_phases: List of phase formulas that MUST be included
            even if they would otherwise be filtered out (e.g., phases
            known to exist from experimental observations)
        metastability_cutoff: Energy above hull cutoff for phases
        exclude_theoretical: Whether to exclude theoretical phases
        save_to_file: If True, save reaction library to a JSON file

    Returns:
        ReactionLibraryData with phase set, reaction library, and metadata
    """
    phase_set, reaction_lib = _build_reaction_library(
        chemical_system,
        temperatures,
        ensure_phases,
        metastability_cutoff,
        exclude_theoretical,
        **entry_kwargs
    )

    reaction_library_path = None
    if save_to_file:
        filename = f"reaction_library_{chemical_system.replace('-', '_')}.json"
        reaction_library_path = str(Path.cwd() / filename)
        with open(reaction_library_path, "w") as f:
            json.dump(reaction_lib.as_dict(), f, cls=MontyEncoder)
        print(f"Saved reaction library to {reaction_library_path}")

    return ReactionLibraryData(
        phase_set_dict=phase_set.as_dict(),
        reaction_library_dict=reaction_lib.as_dict(),
        chemical_system=chemical_system,
        temperatures=temperatures,
        phases_available=list(phase_set.phases),
        reaction_library_path=reaction_library_path,
    )


@job
def run_simulation(
    recipe: "ReactionRecipe",
    reaction_library_data: ReactionLibraryData = None,
    chemical_system: str = None,
    ensure_phases: List[str] = None,
    metastability_cutoff: float = 0.1,
    save_to_file: bool = True,
    metadata: Dict[str, Any] = None,
    live_compress: bool = True,
    compress_freq: int = 100,
) -> SimulationOutput:
    """Run an rxn-ca simulation.

    Args:
        recipe: ReactionRecipe specifying reactants, heating schedule, etc.
        reaction_library_data: Pre-computed reaction library from setup_reaction_library.
            If not provided, will build one from scratch (requires chemical_system).
        chemical_system: Chemical system string, required if reaction_library_data
            not provided
        ensure_phases: Phases to ensure are included, used if building library
            from scratch
        metastability_cutoff: Energy above hull cutoff, used if building library
            from scratch
        save_to_file: If True, save the full result doc to a JSON file
        metadata: Optional user-provided metadata for tagging/provenance
        live_compress: If True, store full state snapshots at compress_freq
            intervals instead of diffs. Avoids slow reconstruction during analysis.
        compress_freq: Interval for storing frames when live_compress is True.

    Returns:
        SimulationOutput with analyzed results and file references
    """
    from rxn_ca.phases import SolidPhaseSet
    from rxn_ca.core.recipe import ReactionRecipe
    from rxn_ca.utilities.parallel_sim import run_sim_parallel
    from rxn_ca.utilities.single_sim import run_single_sim
    from rxn_ca.analysis import BulkReactionAnalyzer
    from rxn_ca.reactions import ReactionLibrary

    recipe_dict = recipe.as_dict()

    if reaction_library_data is not None:
        phase_set = SolidPhaseSet.from_dict(reaction_library_data.phase_set_dict)
        reaction_lib = ReactionLibrary.from_dict(reaction_library_data.reaction_library_dict)
        chem_sys = reaction_library_data.chemical_system
        reaction_library_path = reaction_library_data.reaction_library_path
    else:
        if chemical_system is None:
            raise ValueError("chemical_system required when reaction_library_data not provided")
        all_temps = recipe.heating_schedule.all_temps
        phase_set, reaction_lib = _build_reaction_library(
            chemical_system, all_temps, ensure_phases, metastability_cutoff
        )
        chem_sys = chemical_system
        reaction_library_path = None

    if recipe.num_realizations > 1:
        result_doc = run_sim_parallel(
            recipe=recipe,
            reaction_lib=reaction_lib,
            phase_set=phase_set,
            live_compress=live_compress,
            compress_freq=compress_freq,
        )
    else:
        result_doc = run_single_sim(
            recipe=recipe,
            reaction_lib=reaction_lib,
            phase_set=phase_set,
            live_compress=live_compress,
            compress_freq=compress_freq,
        )

    result_doc_path = None
    if save_to_file:
        filename = f"result_doc_{chem_sys.replace('-', '_')}.json"
        result_doc_path = str(Path.cwd() / filename)
        with open(result_doc_path, "w") as f:
            json.dump(result_doc.as_dict(), f, cls=MontyEncoder)
        print(f"Saved result doc to {result_doc_path}")

    analyzer = BulkReactionAnalyzer.from_result_doc(result_doc)

    final_molar_amounts = analyzer.get_all_absolute_molar_amounts(
        analyzer.last_loaded_step_idx
    )

    molar_trajectory: Dict[str, List[float]] = {}
    temp_trajectory: List[float] = []
    step_indices: List[int] = list(analyzer.loaded_step_idxs)

    for step_idx in step_indices:
        amounts = analyzer.get_all_absolute_molar_amounts(step_idx)
        for phase, amount in amounts.items():
            if phase not in molar_trajectory:
                molar_trajectory[phase] = []
            molar_trajectory[phase].append(amount)
        temp_trajectory.append(recipe.heating_schedule.temp_at(step_idx))

    return SimulationOutput(
        final_molar_amounts=final_molar_amounts,
        molar_amounts_trajectory=molar_trajectory,
        temperature_trajectory=temp_trajectory,
        step_indices=step_indices,
        phase_set_dict=phase_set.as_dict(),
        reaction_library_path=reaction_library_path,
        result_doc_path=result_doc_path,
        recipe_dict=recipe_dict,
        chemical_system=chem_sys,
        num_realizations=recipe.num_realizations,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Bayesian Optimization Jobs
# ---------------------------------------------------------------------------

@job
def init_bo_campaign(
    search_space_config: dict,
    n_initial: int,
    n_iterations: int,
    target_name: str = "yield",
    # precursor_smiles: Optional[Dict[str, Dict[str, str]]] = None,
) -> dict:
    """Initialize a BayBE Campaign from a serialized SearchSpace.

    Builds the BayBE Campaign (search space, recommender, target) and returns
    it as a JSON string so it can be passed between stateless HPC jobs without
    losing the GP posterior.

    Args:
        search_space_config: Serialized SearchSpace from SearchSpace.as_dict().
        n_initial: Number of random exploration trials before BO guidance.
        n_iterations: Number of BO-guided trials after the initial phase.
        target_name: Name of the optimization target (default "yield").

    Returns:
        {"campaign_json": <BayBE Campaign serialized to JSON string>}
    """
    from rxn_ca.optimization import BayesianOptimizer, SearchSpace
    from rxn_ca.optimization.objective import MockObjectiveFunction, ObjectiveConfig, ScorerType

    search_space = SearchSpace.from_dict(search_space_config)

    # MockObjectiveFunction is a stub — no simulations run here.
    # BayesianOptimizer only needs it to satisfy the constructor signature;
    # _build_campaign() does not call objective.evaluate().
    stub_objective = MockObjectiveFunction(
        config=ObjectiveConfig(
            target_phase="__stub__",
            scorer_type=ScorerType.FINAL,
        ),
        score_fn=lambda p: 0.0,
    )

    optimizer = BayesianOptimizer(
        search_space=search_space,
        objective=stub_objective,
        n_initial=n_initial,
        n_iterations=n_iterations,
        target_name=target_name,
        # precursor_smiles=precursor_smiles or {},
    )

    print(f"Initialized BayBE Campaign: {search_space}")
    print(f"  n_initial={n_initial}, n_iterations={n_iterations}, target='{target_name}'")

    return {"campaign_json": optimizer._campaign.to_json()}


@job
def bo_trial_step(
    iteration: int,
    total_iterations: int,
    campaign_json: str,
    reaction_library_data: ReactionLibraryData,
    precursor_slot_names: List[str],
    fixed_precursors: Optional[Dict[str, float]],
    objective_config: dict,
    output_dir: str,
) -> Response:
    """Run one Bayesian optimization trial and chain the next.

    Each call:
    1. Restores the BayBE Campaign from JSON (GP posterior preserved).
    2. Calls campaign.recommend() for the next parameter configuration.
    3. Builds an OptimizableRecipe from those parameters.
    4. Runs the CA simulation using the pre-built reaction library.
    5. Scores the result and adds it to the Campaign.
    6. Saves per-trial JSON and appends to history.csv.
    7. Returns Response(addition=<next trial>) or a summary on the final trial.

    Args:
        iteration: Current 0-based trial index.
        total_iterations: Total trials (n_initial + n_iterations).
        campaign_json: Serialized BayBE Campaign from init_bo_campaign or the
            previous trial. Contains the full GP posterior.
        reaction_library_data: Output from setup_reaction_library. Passed
            through every trial — MP enumeration is done once.
        precursor_slot_names: Names of PrecursorSlotParameters in the search
            space (e.g. ["li_source", "si_source"]). Used to extract the
            selected precursor from BayBE's recommendation.
        fixed_precursors: Formula → amount map when precursors are not
            optimized. Set to None when using precursor slots.
        objective_config: Dict with keys:
            target_phase, scorer_type ("final"|"maximum"),
            simulation_size, num_realizations,
            live_compress (bool), compress_freq (int).
        output_dir: Shared filesystem path for per-trial JSON, history.csv,
            and best_result.json. Must be accessible from all worker nodes.

    Returns:
        Response(addition=<next bo_trial_step>) if more trials remain,
        otherwise a summary dict with best score and parameters.
    """
    from baybe import Campaign
    from rxn_ca.phases import SolidPhaseSet
    from rxn_ca.reactions import ReactionLibrary
    from rxn_ca.utilities.parallel_sim import run_sim_parallel
    from rxn_ca.utilities.single_sim import run_single_sim

    target_phase = objective_config["target_phase"]
    scorer_type = objective_config.get("scorer_type", "final")
    simulation_size = objective_config["simulation_size"]
    num_realizations = objective_config["num_realizations"]
    live_compress = objective_config.get("live_compress", True)
    compress_freq = objective_config.get("compress_freq", 50)
    target_name = objective_config.get("target_name", "yield")

    output_path = Path(output_dir)
    sim_dir = output_path / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== BO Trial {iteration + 1}/{total_iterations} ===")

    # --- Step 1: Restore BayBE Campaign (GP posterior preserved) ---
    campaign = Campaign.from_json(campaign_json)

    # --- Step 2: Get next recommendation ---
    recommendation = campaign.recommend(batch_size=1)
    params = {
        col: (
            recommendation.iloc[0][col].item()
            if hasattr(recommendation.iloc[0][col], "item")
            else recommendation.iloc[0][col]
        )
        for col in recommendation.columns
    }
    print(f"Recommended params: {params}")

    # --- Step 3: Build recipe ---
    opt_recipe = build_recipe_from_params(
        params=params,
        precursor_slot_names=precursor_slot_names,
        fixed_precursors=fixed_precursors,
        simulation_size=simulation_size,
        num_realizations=num_realizations,
    )
    recipe = opt_recipe.to_recipe()
    print(f"Recipe: {opt_recipe}")

    # --- Step 4: Load reaction library ---
    phase_set = SolidPhaseSet.from_dict(reaction_library_data.phase_set_dict)
    rxn_lib = ReactionLibrary.from_dict(reaction_library_data.reaction_library_dict)

    # --- Step 5: Run simulation ---
    print("Running simulation...")
    if num_realizations > 1:
        result_doc = run_sim_parallel(
            recipe,
            reaction_lib=rxn_lib,
            phase_set=phase_set,
            live_compress=live_compress,
            compress_freq=compress_freq,
        )
    else:
        result_doc = run_single_sim(
            recipe,
            reaction_lib=rxn_lib,
            phase_set=phase_set,
            live_compress=live_compress,
            compress_freq=compress_freq,
        )

    # --- Step 6: Score ---
    score = _score_result(result_doc, target_phase, scorer_type)
    print(f"Score ({target_phase}, {scorer_type}): {score:.4f}")

    # --- Step 7: Tell Campaign ---
    campaign.add_measurements(pd.DataFrame([{**params, target_name: score}]))

    # --- Step 8: Save per-trial outputs ---
    trial_result = {
        "iteration": iteration,
        "params": params,
        "score": score,
        "timestamp": datetime.utcnow().isoformat(),
    }
    trial_path = sim_dir / f"trial_{iteration:03d}.json"
    trial_path.write_text(json.dumps(trial_result, indent=2))

    history_path = output_path / "history.csv"
    write_header = not history_path.exists()
    fieldnames = ["iteration", "score", *sorted(params.keys())]
    with history_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({"iteration": iteration, "score": score, **params})

    if result_doc is not None and hasattr(result_doc, "to_file"):
        try:
            result_doc.to_file(str(sim_dir / f"trial_{iteration:03d}_full.json"))
        except Exception as e:
            print(f"Warning: could not save full result doc: {e}")

    # --- Step 9: Chain next trial or finalize ---
    new_campaign_json = campaign.to_json()

    if iteration + 1 < total_iterations:
        next_job = bo_trial_step(
            iteration=iteration + 1,
            total_iterations=total_iterations,
            campaign_json=new_campaign_json,
            reaction_library_data=reaction_library_data,
            precursor_slot_names=precursor_slot_names,
            fixed_precursors=fixed_precursors,
            objective_config=objective_config,
            output_dir=output_dir,
        )
        return Response(addition=next_job)

    # Final iteration: collect all trial results and write summary
    history_rows: List[dict] = []
    for i in range(total_iterations):
        trial_file = sim_dir / f"trial_{i:03d}.json"
        if trial_file.exists():
            history_rows.append(json.loads(trial_file.read_text()))

    if history_rows:
        best = max(history_rows, key=lambda r: r["score"])
        summary = {
            "target_phase": target_phase,
            "fixed_precursors": fixed_precursors,
            "best_score": best["score"],
            "best_params": best["params"],
            "best_iteration": best["iteration"],
            "total_evaluations": total_iterations,
        }
        (output_path / "best_result.json").write_text(json.dumps(summary, indent=2))
        print(f"\nOptimization complete.")
        print(f"Best score: {best['score']:.4f} at iteration {best['iteration']}")
        print(f"Best params: {best['params']}")
        return summary

    return {"status": "complete", "iterations": total_iterations}
