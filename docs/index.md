# rxn-ca

A lattice model for simulating solid state reactions.

## Overview

`rxn-ca` is a Python library for predicting the outcome of solid-state synthesis reactions using a cellular automaton approach. It leverages thermodynamic data from the [Materials Project](https://next-gen.materialsproject.org/) to enumerate possible reactions and simulate phase evolution during synthesis.

## Key Features

- **Reaction Enumeration**: Automatically enumerate possible reactions between precursor phases using thermodynamic data
- **Temperature-Dependent Kinetics**: Compute reaction energies at any temperature using the SISSO descriptor
- **Cellular Automaton Simulation**: Model spatial evolution of phases on a lattice grid
- **Heating Profile Support**: Define complex heating schedules with temperature sweeps and holds
- **HPC Integration**: Run parallel simulations using [Jobflow](https://materialsproject.github.io/jobflow/) for high-throughput studies
- **Analysis Tools**: Visualize phase evolution and extract quantitative metrics

## Quick Example

```python
from rxn_ca.core.recipe import ReactionRecipe
from rxn_ca.core.heating import HeatingSchedule, HeatingStep

# Define reactants (mole ratios)
reactants = {"MgO": 1, "Al2O3": 1}

# Create heating schedule
heating_schedule = HeatingSchedule.build(
    HeatingStep.sweep(500, 1600, stage_length=1, temp_step_size=50),
    HeatingStep.hold(1600, stage_length=20)
)

# Create recipe
recipe = ReactionRecipe(
    reactant_amounts=reactants,
    heating_schedule=heating_schedule
)
```

## Next Steps

- [Getting Started](getting-started.md) - Installation and setup
- [Basic Usage](guides/basic-usage.md) - Complete walkthrough of running a simulation
- [Using Jobflow](guides/jobflow.md) - Running parallel simulations on HPC
