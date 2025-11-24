from typing import List, Optional

from ...core.recipe import ReactionRecipe
from ...computing.schemas.ca_result_schema import RxnCAResultDoc
from ...analysis.visualization.reaction_plotter import ReactionPlotter
from ...analysis.bulk_reaction_analyzer import BulkReactionAnalyzer
from ...analysis.visualization.reaction_graph import ReactionGraph
from pylattica.core import Simulation

from .base_schema import BaseSchema
from dataclasses import dataclass

@dataclass
class MultiRxnCAResultDoc(BaseSchema):
    recipes: List[ReactionRecipe]
    result_docs: Optional[List[RxnCAResultDoc]]
    mass_fraction_plots: Optional[List[dict]]
    molar_fraction_plots: Optional[List[dict]]
    molar_amount_plots: Optional[List[dict]]
    phase_volume_plots: Optional[List[dict]]
    phase_mass_plots: Optional[List[dict]]
    elemental_amount_plots: Optional[List[dict]]
    elemental_fraction_plots: Optional[List[dict]]
    final_simulations: Optional[List[Simulation]]
    metadata: dict = None
    
    @classmethod
    def from_multiple_jobs(cls, result_docs : List[RxnCAResultDoc], 
                           metadata : Optional[dict] = None, 
                           save_results_to_store : bool = False,
                           **kwargs):

        if "reaction_plotter_kwargs" in kwargs:
            reaction_plotter_kwargs = kwargs["reaction_plotter_kwargs"]
        else:
            reaction_plotter_kwargs = {}
        
        recipes = [rd.recipe for rd in result_docs]
        analyzers = [BulkReactionAnalyzer.from_result_doc(rd) for rd in result_docs]
        plotter_objects = [ReactionPlotter(analyzer, **reaction_plotter_kwargs) for analyzer in analyzers]
        
        mass_fraction_plots = [p.plot_mass_fractions().to_json() for p in plotter_objects]
        molar_fraction_plots = [p.plot_molar_phase_fractions().to_json() for p in plotter_objects]
        molar_amount_plots = [p.plot_molar_phase_amounts().to_json() for p in plotter_objects]
        phase_volume_plots = [p.plot_phase_volumes().to_json() for p in plotter_objects]
        phase_mass_plots = [p.plot_phase_masses().to_json() for p in plotter_objects]
        elemental_amount_plots = [p.plot_elemental_amounts().to_json() for p in plotter_objects]
        elemental_fraction_plots = [p.plot_elemental_fractions().to_json() for p in plotter_objects]
        
        final_simulations = [rd.final_simulation for rd in result_docs]
        
        return cls(recipes=recipes,
                   result_docs=result_docs if save_results_to_store else None,
                   mass_fraction_plots=mass_fraction_plots,
                   molar_fraction_plots=molar_fraction_plots,
                   molar_amount_plots=molar_amount_plots,
                   phase_volume_plots=phase_volume_plots,
                   phase_mass_plots=phase_mass_plots,
                   elemental_amount_plots=elemental_amount_plots,
                   elemental_fraction_plots=elemental_fraction_plots,
                   final_simulations=final_simulations,
                   metadata=metadata)