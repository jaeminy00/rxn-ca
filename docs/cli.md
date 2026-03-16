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
| `-l, --library` | Path to the reaction library JSON file |
| `-o, --output` | Output file path |
| `--compress` | Compress output (recommended) |

### Example

```bash
react spinel_recipe.json -l reaction_library.json -o output.json --compress
```

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
