# Getting Started

## Installation

### From Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/mcgalcode/rxn-ca.git
cd rxn-ca
pip install -e .
```

### Optional Dependencies

For optimization features:

```bash
pip install -e ".[optimization]"
```

For workflow integration with Jobflow:

```bash
pip install -e ".[workflow]"
```

For visualization tools:

```bash
pip install -e ".[vis]"
```

## Materials Project API Key

`rxn-ca` retrieves thermodynamic data from the Materials Project. You'll need an API key:

1. Create an account at [Materials Project](https://next-gen.materialsproject.org/)
2. Navigate to your [API settings](https://next-gen.materialsproject.org/api#api-key)
3. Set the environment variable:

```bash
export MP_API_KEY="your_api_key_here"
```

Or add it to your shell configuration file (`.bashrc`, `.zshrc`, etc.).

## Verifying Installation

Test that everything is working:

```python
from rxn_ca.utilities.get_entries import get_entries

# This should download phase data without errors
entries = get_entries(chem_sys="Mg-O", metastability_cutoff=0.03)
print(f"Found {len(entries)} phases")
```

## Next Steps

Once installation is complete, head to the [Basic Usage](guides/basic-usage.md) guide to run your first simulation.
