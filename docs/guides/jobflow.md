# Using Jobflow for Parallel Simulations

While the [Basic Usage](basic-usage.md) guide works well for small-scale studies and debugging, most applications require running many simulations in parallel. This guide shows how to use [Jobflow](https://materialsproject.github.io/jobflow/) for high-throughput simulations on HPC infrastructure.

!!! note "When to Use Jobflow"
    This approach is most useful when running hundreds or more simulations. For smaller numbers, the basic approach may be sufficient.

## Setup

We'll use the same MgAl₂O₄ spinel example. First, prepare the recipe and reaction library:

```python
from rxn_ca.core.recipe import ReactionRecipe
from rxn_ca.core.heating import HeatingSchedule, HeatingStep
from rxn_ca.utilities.get_entries import get_entries
from rxn_ca.utilities.get_scored_rxns import get_scored_rxns
from rxn_ca.phases import SolidPhaseSet
from rxn_network.enumerators.utils import run_enumerators
from rxn_network.enumerators.basic import BasicEnumerator
from rxn_network.enumerators.minimize import MinimizeGibbsEnumerator

reactants = {"MgO": 1, "Al2O3": 1}

heating_schedule = HeatingSchedule.build(
    HeatingStep.sweep(500, 1600, stage_length=1, temp_step_size=50),
    HeatingStep.hold(1600, stage_length=20)
)

recipe = ReactionRecipe(
    reactant_amounts=reactants,
    heating_schedule=heating_schedule
)

# Get entries and enumerate reactions
all_entries = get_entries(
    chem_sys="Mg-Al-O",
    metastability_cutoff=0.03,
    exclude_theoretical_phases=True
)

rxn_set = run_enumerators(
    [MinimizeGibbsEnumerator(), BasicEnumerator()],
    all_entries
)

all_temps = recipe.heating_schedule.all_temps
temp_rxn_mapping = rxn_set.compute_at_temperatures(all_temps)

solid_phase_set = SolidPhaseSet.from_entry_set(all_entries)
reaction_library = get_scored_rxns(
    rxn_set,
    temps=all_temps,
    phase_set=solid_phase_set,
    rxns_at_temps=temp_rxn_mapping,
    parallel=True
)
```

## Creating Jobs with MultiRxnCAMaker

Jobflow allows you to create similar "jobs" and orchestrate complex workflows. The `MultiRxnCAMaker` handles parallel simulation jobs.

```python
from rxn_ca.computing.jobs import MultiRxnCAMaker, MultiRxnType
```

### Simulation Types

There are three types of parallel simulations:

| Type | Description | Use Case |
|------|-------------|----------|
| `MULTI_RECIPE` | One reaction library, many recipes | Testing different heating profiles |
| `MULTI_LIB` | One recipe, many reaction libraries | Testing different scoring functions |
| `MULTI_LIB_AND_RECIPE` | Many recipes and many libraries | Full parameter sweeps |

### Creating Jobs

```python
num_replications = 3

# Case 1: Multiple recipes, single library
maker = MultiRxnCAMaker(multi_rxn_type=MultiRxnType.MULTI_RECIPE)
job = maker.make(
    recipes=[recipe] * num_replications,
    reaction_libraries=reaction_library
)

# Case 2: Single recipe, multiple libraries
maker = MultiRxnCAMaker(multi_rxn_type=MultiRxnType.MULTI_LIB)
job = maker.make(
    recipes=recipe,
    reaction_libraries=[reaction_library] * num_replications
)

# Case 3: Multiple recipes and libraries (default)
maker = MultiRxnCAMaker(multi_rxn_type=MultiRxnType.MULTI_LIB_AND_RECIPE)
job = maker.make(
    recipes=[recipe] * num_replications,
    reaction_libraries=[reaction_library] * num_replications
)
```

## Running Jobs Locally

For testing or small runs:

```python
from jobflow import run_locally

output = run_locally(job, create_folders=True)
```

Each simulation runs 3 replicates by default to better sample the system. With `num_replications=3`, this executes 9 total simulations.

!!! tip "CPU Usage"
    By default, each simulation uses 1 CPU. Run at least as many simulations as available cores for optimal performance. If you encounter memory errors, increase `cpus_per_task` to 2 or 4.

## Accessing Results

Results are stored in the job output:

```python
job_output = output[job.uuid][1].output
```

### Viewing Plots

Analysis plots are stored in serialized format. Deserialize with Plotly:

```python
from plotly.io import from_json

for i, serial_plot in enumerate(job_output.molar_fraction_plots):
    print(f"Plotting job {i}...")
    fig = from_json(serial_plot['figure_json'])
    fig.show()
```

### Accessing Raw Data

```python
for i, serial_plot in enumerate(job_output.molar_fraction_plots):
    for phase, data in serial_plot['data'].items():
        print(f"{phase}: {data['y']}")
```

### Checking Convergence

```python
print(job_output.have_simulations_converged)
# [False, False, False]  # Short simulations typically don't converge
```

If simulations haven't converged, consider increasing the simulation time. For detailed analysis, read the full result documents from `job_output.run_dir`.

## Running on HPC with FireWorks

For large-scale runs, use [FireWorks](https://materialsproject.github.io/fireworks/) to manage workflows:

```python
from jobflow.manager.fireworks import flow_to_workflow
from fireworks.core.launchpad import LaunchPad

lp = LaunchPad.from_file("fw_config.yaml")
wf = flow_to_workflow(job)
lp.add_wf(wf)
```

Then on your HPC cluster:

```bash
# Single workflow
qlaunch singleshot -f <fw_id>
# or
rlaunch singleshot -f <fw_id>

# Many workflows in parallel
qlaunch rapidfire --nlaunches <num_launches>
```

Refer to the [Jobflow documentation](https://materialsproject.github.io/jobflow/) and [FireWorks documentation](https://materialsproject.github.io/fireworks/) for detailed setup instructions.
