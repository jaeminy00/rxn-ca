"""Bayesian Optimization jobs for rxn-ca simulations."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from jobflow import Response, job

from ..schemas import ReactionLibraryData


# ---------------------------------------------------------------------------
# Helpers (not @job — called inside bo_trial_step)
# ---------------------------------------------------------------------------

def build_recipe_from_params(
    params: Dict[str, Any],
    precursor_slot_names: List[str],
    fixed_precursors: Optional[Dict[str, float]],
    simulation_size: int,
    num_realizations: int,
) -> "OptimizableRecipe":
    """Convert a BayBE parameter recommendation into an OptimizableRecipe.

    Args:
        params: Parameter dict recommended by BayBE (may contain numpy scalars).
        precursor_slot_names: Names of precursor slot parameters in the search
            space (e.g. ["li_source", "si_source"]). Used to extract the
            selected precursor formula for each slot from params.
        fixed_precursors: If provided, use these formula→amount pairs directly
            (precursor selection is not optimized).
        simulation_size: CA grid size (NxN).
        num_realizations: Number of simulation realizations to average.

    Returns:
        Configured OptimizableRecipe ready for to_recipe().
    """
    from rxn_ca.optimization import OptimizableRecipe

    # Normalize numpy scalar types that BayBE returns in its DataFrame
    clean_params = {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in params.items()
    }

    if fixed_precursors is not None:
        return OptimizableRecipe(
            precursors=fixed_precursors,
            hold_temp=int(clean_params["hold_temp"]),
            hold_time=int(clean_params["hold_time"]),
            ramp_step_time=int(clean_params.get("ramp_step_time", 1)),
            simulation_size=simulation_size,
            num_simulations=num_realizations,
        )

    precursor_slots = {name: clean_params[name] for name in precursor_slot_names}

    return OptimizableRecipe.from_params(
        params=clean_params,
        precursor_slots=precursor_slots,
        simulation_size=simulation_size,
        num_simulations=num_realizations,
    )


def _score_result(result_doc: Any, target_phase: str, scorer_type: str) -> float:
    """Score a simulation result document.

    Args:
        result_doc: RxnCAResultDoc from the simulation.
        target_phase: Chemical formula of the target product.
        scorer_type: "final" (yield at end) or "maximum" (peak yield).

    Returns:
        Float score in [0, 1], or 0.0 if scoring fails.
    """
    from rxn_ca.optimization import AnalyzedResult, FinalProductScorer, MaximumProductScorer, get_result_analysis

    try:
        traces = get_result_analysis(result_doc)
        analyzed = AnalyzedResult(traces)
        scorer_cls = MaximumProductScorer if scorer_type == "maximum" else FinalProductScorer
        return scorer_cls(target_phase).score(analyzed)
    except Exception as e:
        print(f"Warning: scoring failed ({type(e).__name__}: {e})")
        return 0.0


# ---------------------------------------------------------------------------
# Jobs
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
            space (e.g. ["li_source", "si_source"]).
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
