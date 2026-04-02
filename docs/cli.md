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

### Options

| Option | Description |
|--------|-------------|
| `-n, --n-precursors` | Number of precursors per template (default: 2) |
| `-m, --max-templates` | Maximum templates to show (default: 10) |
| `-e, --energy-cutoff` | Metastability cutoff in eV/atom (default: 0.1) |
| `--anions` | Comma-separated anion formulas (default: O,CO3,OH,NO3) |
| `--metathesis` | Comma-separated metathesis anions (e.g., Cl,Br) |
| `--counter-cations` | Comma-separated counter-cations (e.g., Na,K) |
| `--json` | Output as JSON |
| `-l, --literature-data` | Path to synthesis dataset for literature-based ranking |

### Anion Reference

| Formula | Name | Elements Added |
|---------|------|----------------|
| `O` | oxide | O |
| `CO3` | carbonate | C, O |
| `NO3` | nitrate | N, O |
| `OH` | hydroxide | O, H |
| `Cl` | chloride | Cl |
| `Br` | bromide | Br |
| `SO4` | sulfate | S, O |

### Examples

Basic usage with 3 precursors:
```bash
suggest-precursors BaTiO3 -n 3
```

With chloride metathesis and sodium/potassium sources:
```bash
suggest-precursors Ba2ZrTiO6 -n 3 --metathesis Cl --counter-cations Na,K
```

Minimal element set (oxides and carbonates only):
```bash
suggest-precursors Ba2ZrTiO6 --anions O,CO3
```
