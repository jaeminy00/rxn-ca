"""Helper functions for Bayesian Optimization jobs.

These are plain functions (not @job decorated) called from within
bo_trial_step in rxn_ca.workflow.jobs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rxn_ca.optimization import (
    AnalyzedResult,
    FinalProductScorer,
    MaximumProductScorer,
    OptimizableRecipe,
    get_result_analysis,
)


def build_recipe_from_params(
    params: Dict[str, Any],
    precursor_slot_names: List[str],
    fixed_precursors: Optional[Dict[str, float]],
    simulation_size: int,
    num_realizations: int,
) -> OptimizableRecipe:
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

    # Variable precursors: extract the selected formula for each slot from params.
    # We pass actual string values (not None) so OptimizableRecipe.from_params
    # populates precursors correctly via its first loop.
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
    try:
        traces = get_result_analysis(result_doc)
        analyzed = AnalyzedResult(traces)
        scorer_cls = MaximumProductScorer if scorer_type == "maximum" else FinalProductScorer
        return scorer_cls(target_phase).score(analyzed)
    except Exception as e:
        print(f"Warning: scoring failed ({type(e).__name__}: {e})")
        return 0.0
