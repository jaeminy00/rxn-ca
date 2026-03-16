# Basic Usage

This guide walks through running a complete solid-state reaction simulation using `rxn-ca`. We'll simulate the spinel reaction:

$$MgO + Al_{2}O_{3} \rightarrow MgAl_{2}O_{4}$$

## 1. Define the Synthesis Recipe

The first step is to define a `ReactionRecipe` that specifies the reactants and heating profile.

### Specify Reactants

Express reactants as a dictionary of mole ratios:

```python
reactants = {
    "MgO": 1,
    "Al2O3": 1,
}
```

### Create Heating Schedule

Heating profiles are defined using the `HeatingSchedule` class. Because reaction energies must be computed for every temperature, heating profiles consist of discrete steps rather than continuous sweeps.

```python
from rxn_ca.core.recipe import ReactionRecipe
from rxn_ca.core.heating import HeatingSchedule, HeatingStep

heating_schedule = HeatingSchedule.build(
    HeatingStep.sweep(500, 1600, stage_length=1, temp_step_size=50),
    HeatingStep.hold(1600, stage_length=20)
)

recipe = ReactionRecipe(
    reactant_amounts=reactants,
    heating_schedule=heating_schedule
)
```

!!! note "Stage Length"
    The `stage_length` parameter determines how long each temperature step is held. A reasonable starting point is integers between 1 and 10.

Save the recipe for later use:

```python
recipe.to_file("spinel_recipe.json")
```

## 2. Collect Entries from Materials Project

After defining the recipe, download phase entries from the Materials Project database. Each entry corresponds to a phase that might appear in the reaction pathway.

### Filtering Parameters

- **Chemical System**: The set of elements present in the reaction (precursors + atmosphere)
- **Metastability Cutoff**: Maximum energy above the convex hull (in eV/atom). Phases with higher metastability are excluded.
- **Exclude Theoretical**: Whether to exclude phases without ICSD observations

```python
from rxn_ca.utilities.get_entries import get_entries

chem_sys = "Mg-Al-O"
metastability_cutoff = 0.03
exclude_theoretical = True

all_entries = get_entries(
    chem_sys=chem_sys,
    metastability_cutoff=metastability_cutoff,
    exclude_theoretical_phases=exclude_theoretical
)

print(f"Considering {len(all_entries)} phases")
```

## 3. Enumerate Reactions

Enumerate the reactions possible between the phases. This uses the [reaction-network](https://github.com/materialsproject/reaction-network) library.

```python
from rxn_network.enumerators.utils import run_enumerators
from rxn_network.enumerators.basic import BasicEnumerator
from rxn_network.enumerators.minimize import MinimizeGibbsEnumerator

gibbs_enums = [MinimizeGibbsEnumerator(), BasicEnumerator()]
rxn_set = run_enumerators(gibbs_enums, all_entries)
```

## 4. Compute Temperature-Dependent Reaction Energies

The Gibbs formation energy of each phase depends on temperature. These values are computed using the [SISSO descriptor](https://www.nature.com/articles/s41467-018-06682-4) which captures vibrational contributions to free energy.

```python
all_temps = recipe.heating_schedule.all_temps
print(f"All temperatures required: {all_temps}")

temp_rxn_mapping = rxn_set.compute_at_temperatures(all_temps)
```

## 5. Score Reactions

Create a reaction library with scored reactions at each temperature:

```python
from rxn_ca.utilities.get_scored_rxns import get_scored_rxns
from rxn_ca.phases import SolidPhaseSet
from monty.serialization import dumpfn

solid_phase_set = SolidPhaseSet.from_entry_set(all_entries)

reaction_library = get_scored_rxns(
    rxn_set,
    temps=all_temps,
    phase_set=solid_phase_set,
    rxns_at_temps=temp_rxn_mapping,
    parallel=True
)

dumpfn(reaction_library, "reaction_library.json")
```

## 6. Run the Simulation

With the recipe and reaction library files ready, run the simulation using the `react` CLI:

```bash
react spinel_recipe.json -l reaction_library.json -o output.json
```

!!! note "Live Compression"
    Compression is enabled by default to keep output file sizes manageable. The output will be saved as `output_compressed.json`. Use `--no-compress` if you need all frames (e.g., for `--count-reactions`).

## 7. Analyze Results

Visualize the evolution of phase amounts over the course of the reaction:

```python
from rxn_ca.analysis.bulk_reaction_analyzer import BulkReactionAnalyzer
from rxn_ca.analysis.visualization.reaction_plotter import ReactionPlotter
from rxn_ca.analysis.visualization import PhaseTraceConfig

output_filename = "output_compressed.json"
analyzer = BulkReactionAnalyzer.from_result_doc_file(output_filename)

trace_config = PhaseTraceConfig(minimum_required_prevalence=0.00)
plotter = ReactionPlotter(analyzer, include_heating_trace=True)

plotter.plot_molar_phase_amounts()
```

The plot shows molar amounts of each phase over simulation time. The absolute numbers are not meaningful—only the relative amounts matter.

## Complete Script

Here's the complete workflow in a single script:

```python
from rxn_ca.core.recipe import ReactionRecipe
from rxn_ca.core.heating import HeatingSchedule, HeatingStep
from rxn_ca.utilities.get_entries import get_entries
from rxn_ca.utilities.get_scored_rxns import get_scored_rxns
from rxn_ca.phases import SolidPhaseSet
from rxn_network.enumerators.utils import run_enumerators
from rxn_network.enumerators.basic import BasicEnumerator
from rxn_network.enumerators.minimize import MinimizeGibbsEnumerator
from monty.serialization import dumpfn

# 1. Define recipe
reactants = {"MgO": 1, "Al2O3": 1}
heating_schedule = HeatingSchedule.build(
    HeatingStep.sweep(500, 1600, stage_length=1, temp_step_size=50),
    HeatingStep.hold(1600, stage_length=20)
)
recipe = ReactionRecipe(reactant_amounts=reactants, heating_schedule=heating_schedule)

# 2. Get entries
all_entries = get_entries(
    chem_sys="Mg-Al-O",
    metastability_cutoff=0.03,
    exclude_theoretical_phases=True
)

# 3. Enumerate reactions
rxn_set = run_enumerators(
    [MinimizeGibbsEnumerator(), BasicEnumerator()],
    all_entries
)

# 4. Compute at temperatures
all_temps = recipe.heating_schedule.all_temps
temp_rxn_mapping = rxn_set.compute_at_temperatures(all_temps)

# 5. Score reactions
solid_phase_set = SolidPhaseSet.from_entry_set(all_entries)
reaction_library = get_scored_rxns(
    rxn_set,
    temps=all_temps,
    phase_set=solid_phase_set,
    rxns_at_temps=temp_rxn_mapping,
    parallel=True
)

# Save outputs
recipe.to_file("spinel_recipe.json")
dumpfn(reaction_library, "reaction_library.json")

# Run: react spinel_recipe.json -l reaction_library.json -o output.json
```
