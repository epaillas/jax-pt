# Repository Guidelines

## Architecture
`jaxpt/` contains the importable package. Keep theory interfaces, numerical kernels, assembly helpers, and reference adapters separate:

- `jaxpt/theories/`: canonical observable-level entrypoints. `PowerSpectrumTemplate` resolves cosmology inputs, and `GalaxyPowerSpectrumMultipolesTheory` or `ClassPTGalaxyPowerSpectrumMultipolesTheory` drive predictions.
- `jaxpt/cosmology.py`: linear-input containers plus provider/resolution helpers for `classy` and `cosmoprimo` backed templates.
- `jaxpt/config.py`: immutable `PTSettings` for backend, loop-order, FFTLog, and output configuration.
- `jaxpt/parameter.py`: named scalar parameter metadata via `Parameter` and `ParameterCollection`.
- `jaxpt/basis.py`: basis construction and lower-level prediction helpers such as `compute_basis`.
- `jaxpt/kernels/`: analytic tree, loop, spectral, and RSD kernel implementations used by basis construction.
- `jaxpt/bias.py`: assembly formulas that map named basis terms into matter, real-space galaxy, and multipole observables.
- `jaxpt/reference/`: adapters and parity utilities for the installed `CLASS-PT` reference backend.
- `jaxpt/emulators/`: Taylor-emulator building and evaluation on top of theory objects.

The public import surface is re-exported from `jaxpt/__init__.py`; do not add a parallel API layer unless there is a clear stability reason.

Do not hide physics combinations inside mutable global objects. Prefer explicit data flow through immutable dataclasses, named parameters, and function inputs.

## Development
Use `pytest` for validation.

- Run `pytest -q` for the main suite.
- For numerical changes, add or update tests under `tests/`.
- Prefer parity checks against the installed `CLASS-PT` backend when the code path has a direct reference implementation.
- Keep examples in `examples/` and CLI scripts in `scripts/` aligned with the current theory API when changing entrypoints or parameter names.

After every major repository update, create a git commit that captures the completed state.

## Style
Prefer small, named helpers over integer-index-heavy logic.

- If a `CLASS-PT` basis term is imported by index, map it once into a descriptive name and use the name everywhere else.
- Keep observable assembly in `bias.py` and theory orchestration in `jaxpt/theories/`; avoid pushing high-level policy into kernel modules.
- Prefer extending `ParameterCollection` and YAML-backed defaults over introducing ad hoc parameter dictionaries.
- Preserve the current backend split: `jaxpt` code should remain usable without silently depending on the `CLASS-PT` path except for explicit reference checks.
