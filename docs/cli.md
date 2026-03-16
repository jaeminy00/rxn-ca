# CLI Reference

`rxn-ca` provides several command-line tools for running simulations and preparing inputs.

## react

Run a cellular automaton reaction simulation.

```bash
react <recipe_file> -l <library_file> -o <output_file> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `recipe_file` | Path to the reaction recipe JSON file |

### Options

| Option | Description |
|--------|-------------|
| `-l, --reaction-library-file` | Path to the reaction library JSON file (required) |
| `-o, --output-file` | Output file path |
| `-d, --input-dir` | Input directory for batch processing |
| `-p, --output-dir` | Output directory |
| `-s, --single` | Run single simulation (no parallelization) |
| `-i, --initial-simulation-file` | Path to initial simulation state |
| `--no-compress` | Disable live compression (keeps all frames) |
| `--count-reactions` | Assemble reaction choice metadata (requires `--no-compress`) |
| `--store-lib / --no-store-lib` | Include reaction library in output |

### Example

```bash
react spinel_recipe.json -l reaction_library.json -o output.json
```

Live compression is enabled by default, saving output as `*_compressed.json`.

## enumerate

Enumerate reactions for a chemical system.

```bash
enumerate <chem_sys> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `chem_sys` | Chemical system (e.g., "Mg-Al-O") |

## build-library

Build a reaction library from enumerated reactions.

```bash
build-library [options]
```

## suggest-precursors

Generate suggested precursor sets for a target phase.

```bash
suggest-precursors <target> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `target` | Target phase formula (e.g., "BaTiO3") |

### Example

```bash
suggest-precursors BaTiO3
```

This generates practical precursor suggestions including oxides, carbonates, hydroxides, and nitrates based on the target composition.
