from ..core.recipe import ReactionRecipe
from ..reactions import ReactionLibrary
from ..computing.schemas.multi_ca_result_schema import MultiRxnCAResultDoc
from ..computing.schemas.ca_result_schema import RxnCAResultDoc
from ..utilities.parallel_sim import run_multi_recipe_parallel_ray
from pylattica.core import Simulation
from jobflow import job, Maker
from dataclasses import dataclass, field
from typing import List
import os
from enum import Enum

_JOBSTORE_OBJECTS = [ReactionLibrary, RxnCAResultDoc] 
#have to include RxnCAResultDoc because it contains lists of ReactionResult objects, 
#and jobflow DOES NOT automatically recursively search inside each list item.
#solution is to use a MultiRxnCAResultDoc as the output schema.

class MultiRxnType(Enum):
    MULTI_LIB = "multi_lib"
    MULTI_RECIPE = "multi_recipe"
    MULTI_LIB_AND_RECIPE = "multi_lib_and_recipe"


@dataclass
class MultiRxnCAMaker(Maker):
    """
    Maker for running multiple recipes in parallel using Ray. Can run one of the following:
    - MULTI_LIB: Run multiple recipes in parallel using one reaction library.
    - MULTI_RECIPE: Run multiple recipes in parallel using multiple reaction libraries.
    - MULTI_LIB_AND_RECIPE: Run multiple recipes in parallel using multiple reaction libraries and multiple initial simulations.
    Default is MULTI_LIB_AND_RECIPE. 

    Parameters
    ----------
    multi_rxn_type (Enum): The type of multi-recipe run to perform.
        Instance of rxn_ca.computing.jobs.MultiRxnType.
    metadata (dict): Metadata to save with the result doc.
        Default is None. Helps with tagging the result doc.
    save_results_to_store (bool): Whether to save the result docs to the store.
        Default is False. The result docs are saved to the launch directory by default, 
        and not the JobStore becuase these files can be extremely large. 
    num_workers (int): Number of workers to use for the parallel run.
        Default is None, which uses all available cores on the current node.
    memory_per_task_gb (float): Memory per task in GB.
        Default is None, which uses the default memory per task.
    cpus_per_task (int): Number of CPUs per task.
        Default is 1. Any more does not usually improve performance.
        Consider increasing if you are running into memory issues.
    compress (bool): Whether to compress the result docs.
        Default is True. Compresing the output result docs is highly recommended.
    reaction_plotter_kwargs (dict): Keyword arguments for the reaction plotter.
        Default is {"include_heating_trace": True}. 
        See rxn_ca.analysis.visualization.reaction_plotter for more details.
    """
    
    name: str = "MultiRxnCA Maker"
    multi_rxn_type: MultiRxnType = field(default=MultiRxnType.MULTI_LIB_AND_RECIPE)
    metadata: dict = field(default=None)
    save_results_to_store: bool = field(default=False)
    num_workers: int = None
    memory_per_task_gb: float = field(default=None)
    cpus_per_task: int = field(default=1)
    compress: bool = field(default=True)
    reaction_plotter_kwargs: dict = field(default_factory=lambda: {"include_heating_trace": True})
    
    @job(data=_JOBSTORE_OBJECTS)
    def make(self, recipes: List[ReactionRecipe] | ReactionRecipe, 
             reaction_libraries: List[ReactionLibrary] | ReactionLibrary, 
             initial_simulations: List[Simulation] | Simulation | str = None,
             **kwargs) -> MultiRxnCAResultDoc:
        
        if self.multi_rxn_type == MultiRxnType.MULTI_LIB:
            if len(recipes) != 1:
                raise ValueError("MultiRxnType.MULTI_LIB requires exactly one recipe")
            
            print(f"Running {len(reaction_libraries)} library realizations for given recipe")
            if isinstance(recipes, ReactionRecipe):
                recipes = [recipes] * len(reaction_libraries)
            else:
                recipes = [recipes[0]] * len(reaction_libraries)
            
        if self.multi_rxn_type == MultiRxnType.MULTI_RECIPE:
            if len(reaction_libraries) != 1:
                raise ValueError("MultiRxnType.MULTI_RECIPE requires exactly one reaction library")
            
            print(f"Running {len(recipes)} recipe realizations with given reaction library")
            if isinstance(reaction_libraries, ReactionLibrary):
                reaction_libraries = [reaction_libraries] * len(recipes)
            else:
                reaction_libraries = [reaction_libraries[0]] * len(recipes)
            
        if self.multi_rxn_type == MultiRxnType.MULTI_LIB_AND_RECIPE:
            if len(recipes) != len(reaction_libraries):
                raise ValueError("MultiRxnType.MULTI_LIB_AND_RECIPE requires the same number of recipes and reaction libraries")
            
        if initial_simulations is None:
            initial_simulations = [None] * len(recipes)
        if initial_simulations:
            if not isinstance(initial_simulations, list):
                if isinstance(initial_simulations, Simulation):
                    initial_simulations = [initial_simulations] * len(recipes)
                elif isinstance(initial_simulations, str):
                    initial_simulations = [Simulation.from_file(initial_simulations)] * len(recipes)
                else:
                    raise ValueError("If initial_simulations is provided, it must be a list of Simulation objects, a single Simulation object, or a string path to a Simulation object")
            elif len(initial_simulations) != len(recipes):
                raise ValueError("If initial_simulations is provided, it must be the same length as the number of recipes/libraries")
            
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
                                                                        reaction_plotter_kwargs=self.reaction_plotter_kwargs,
                                                                        run_dir=os.getcwd())
        multi_rxn_ca_result_doc.to_file("multi_rxn_ca_result.json")
        return multi_rxn_ca_result_doc

