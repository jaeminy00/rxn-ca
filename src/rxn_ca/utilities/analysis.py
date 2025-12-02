from typing import Dict
import numpy as np

from rxn_ca.phases.gasses import DEFAULT_GASES
from rxn_ca.analysis.bulk_reaction_analyzer import BulkReactionAnalyzer

def has_simulation_converged(analyzer: BulkReactionAnalyzer, convergence_criteria : float = 0.001) -> bool:
    """Check if a simulation has converged based on the phase data"""
    
    converged = True
    phase_amounts = [analyzer.get_all_absolute_molar_amounts(i) for i in range(analyzer.result_length-10, analyzer.result_length)]
    for phase in phase_amounts[0].keys():
        if phase in [*DEFAULT_GASES, "FREE_SPACE"]:
            continue
        if np.std([p[phase] for p in phase_amounts]) / np.mean([p[phase] for p in phase_amounts]) > convergence_criteria:
            converged = False
            break
    return converged