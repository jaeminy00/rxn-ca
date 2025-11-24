from ..core.recipe import ReactionRecipe
from ..reactions import ReactionLibrary
from ..computing.schemas.multi_ca_result_schema import MultiRxnCAResultDoc
from ..computing.schemas.ca_result_schema import RxnCAResultDoc
from ..utilities.parallel_sim import run_multi_recipe_parallel_ray
from pylattica.core import Simulation
from jobflow import job, Maker
from dataclasses import dataclass, field
from typing import List

_JOBSTORE_OBJECTS = [ReactionLibrary, RxnCAResultDoc] 
#have to include RxnCAResultDoc because it contains lists of ReactionResult objects, 
#and jobflow DOES NOT automatically recursively search inside each list item.
#solution is to use a MultiRxnCAResultDoc as the output schema.

@dataclass
class MultiRxnCAMaker(Maker):
    name: str = "MultiRxnCA Maker"
    metadata: dict = None
    save_results_to_store: bool = False
    num_workers: int = 1
    memory_per_task_gb: float = 1.0
    cpus_per_task: int = 1
    compress: bool = True
    plotting_kwargs: dict = None
    reaction_plotter_kwargs: dict = field(default_factory=lambda: {"include_heating_trace": True})

    @job(results="rxn_docs", data=_JOBSTORE_OBJECTS)
    def make(self, recipes: List[ReactionRecipe], 
             reaction_libraries: List[ReactionLibrary], 
             initial_simulations: List[Simulation | str] = None,
             **kwargs) -> MultiRxnCAResultDoc:

        if initial_simulations is None:
            initial_simulations = [None] * len(recipes)
        for i,initial_simulation in enumerate(initial_simulations):
            if isinstance(initial_simulation, str):
                initial_simulations[i] = Simulation.from_file(initial_simulation)
        
        result_docs = run_multi_recipe_parallel_ray(recipes, 
                                                    reaction_libraries, 
                                                    initial_simulations, 
                                                    num_workers=self.num_workers, 
                                                    memory_per_task_gb=self.memory_per_task_gb, 
                                                    cpus_per_task=self.cpus_per_task, 
                                                    compress=self.compress,
                                                    **kwargs)
        
        print("============ SAVING RESULT DOCS =============")
        
        for i,result_doc in enumerate(result_docs):
            print(f"Saving result doc {i} to file...")
            result_doc.to_file(f"result_doc_{i}.json")

        print("============ CREATING MULTI RXN CA RESULT DOCUMENT =============")
        multi_rxn_ca_result_doc = MultiRxnCAResultDoc.from_multiple_jobs(result_docs, 
                                                                        metadata=self.metadata, 
                                                                        save_results_to_store=self.save_results_to_store,
                                                                        plotting_kwargs=self.plotting_kwargs,
                                                                        reaction_plotter_kwargs=self.reaction_plotter_kwargs)
        multi_rxn_ca_result_doc.to_file("multi_rxn_ca_result.json")
        return multi_rxn_ca_result_doc

