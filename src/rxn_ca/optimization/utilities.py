from typing import Dict, List
from ..analysis.bulk_reaction_analyzer import BulkReactionAnalyzer
from ..analysis.visualization import PhaseTraceCalculator, PhaseTraceConfig
from ..computing.schemas.ca_result_schema import RxnCAResultDoc


def get_result_analysis(reaction_result: RxnCAResultDoc) -> Dict[str, List[float]]:
    """Extract phase traces from a reaction result document.

    Args:
        reaction_result: The result document from a simulation

    Returns:
        Dictionary mapping phase names to lists of mass fractions over time
    """
    analyzer = BulkReactionAnalyzer.from_result_doc(reaction_result)
    trace_config = PhaseTraceConfig(minimum_required_prevalence=0.00)
    phase_trace_calc = PhaseTraceCalculator(analyzer.loaded_step_groups, analyzer.step_analyzer)
    phase_traces = phase_trace_calc.get_mass_fraction_traces(trace_config)

    # Convert List[PhaseTrace] to dict for easier access
    traces_dict = {trace.name: list(trace.ys) for trace in phase_traces}
    return traces_dict


class AnalyzedResult():

    def __init__(self, traces: Dict[str, List[float]]):
        self.traces = traces

    def get_trace(self, phase: str) -> List[float]:
        return self.traces.get(phase, [])


class MaximumProductScorer():

    def __init__(self,
                 target_product: str):
        self.target = target_product

    def score(self, res: AnalyzedResult) -> float:
        trace = res.get_trace(self.target)
        if not trace:
            return 0.0
        return max(trace)


class FinalProductScorer():

    def __init__(self,
                 target_product: str):
        self.target = target_product

    def score(self, res: AnalyzedResult) -> float:
        trace = res.get_trace(self.target)
        if not trace:
            return 0.0
        return trace[-1]

