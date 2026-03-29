# Repository Guidelines

## Architecture
`jaxpt/` contains the importable package. Keep numerical kernels, adapters, and public APIs separate:

- `jaxpt/config.py`: typed parameter and settings containers.
- `jaxpt/cosmology.py`: linear-input containers and `classy` helpers.
- `jaxpt/reference/`: adapters to external reference implementations.
- `jaxpt/bias.py`: pure assembly formulas from named basis terms.
- `jaxpt/api.py`: stable public entry points.

Do not hide physics combinations inside mutable global objects. Prefer immutable dataclasses and explicit function inputs.

## Development
Use `pytest` for validation. For numerical changes, add or update tests that compare against the installed `CLASS-PT` reference backend when possible.
After every major repository update, create a git commit that captures the completed state.

## Style
Prefer small, named helpers over integer-index-heavy logic. If a `CLASS-PT` basis term is imported by index, map it once into a descriptive name and use the name everywhere else.
