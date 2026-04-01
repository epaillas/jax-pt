# jax-pt

`jaxpt` is a JAX implementation of perturbation-theory building blocks for galaxy-clustering predictions, with direct validation against [CLASS-PT](https://github.com/Michalychforever/CLASS-PT).

The package is organized around observable-specific theory interfaces under
`jaxpt.theories`. Power-spectrum theory classes live there today, with
`emulators` and `inference` reserved for future emulator and sampling tooling.

## Installation

```bash
pip install -e ".[dev]"
```

Reference checks and examples that compare against CLASS-PT require a local `CLASS-PT`-enabled `classy` installation.

## Quick Start

```python
import numpy as np
from classy import Class

from jaxpt import PTSettings
from jaxpt.theories import GalaxyPowerSpectrumMultipolesTheory, PowerSpectrumTemplate

z = 0.5
support_k = np.logspace(-5.0, 1.0, 256)
eval_k = np.linspace(0.01, 0.2, 64)

cosmo = Class()
cosmo.set(
    {
        "A_s": 2.089e-9,
        "n_s": 0.9649,
        "tau_reio": 0.052,
        "omega_b": 0.02237,
        "omega_cdm": 0.12,
        "h": 0.6736,
        "YHe": 0.2425,
        "N_ur": 2.0328,
        "N_ncdm": 1,
        "m_ncdm": 0.06,
        "z_pk": z,
        "output": "mPk",
        "non linear": "PT",
        "IR resummation": "No",
        "Bias tracers": "Yes",
        "cb": "Yes",
        "RSD": "Yes",
    }
)
cosmo.compute()

template = PowerSpectrumTemplate(cosmo, z=z, k=support_k, settings=PTSettings(ir_resummation=False))
theory = GalaxyPowerSpectrumMultipolesTheory(template=template, k=eval_k)
prediction = theory()
```

The returned `prediction` is a `MultipolePrediction` with `k`, `p0`, `p2`, and
`p4` arrays plus backend/theory metadata.

## Main API Surface

- `PTSettings`
  Controls backend choice, loop order, FFTLog support, and output options.
- `PowerSpectrumTemplate`
  Resolves cosmology inputs into a `LinearPowerInput`. The source can be a live
  `classy.Class`, a live `cosmoprimo.Cosmology`, a fiducial parameter mapping,
  or a precomputed `LinearPowerInput`.
- `GalaxyPowerSpectrumMultipolesTheory`
  High-level observable wrapper that accepts a flat query of nuisance and
  cosmology parameters and returns `MultipolePrediction`.
- `compute_basis` and `galaxy_multipoles`
  Lower-level basis and assembly entry points for users who want to work below
  the theory layer.
- `TaylorEmulator` and `build_multipole_emulator`
  Tools for building hashed Taylor emulators around a fiducial multipole
  theory.

## Current Scope

- Tree-level Kaiser basis generation from `LinearPowerInput`
- One-loop real-space matter and bias terms from analytic FFTLog kernels generated in-repo
- Assembly of `P_0(k)`, `P_2(k)`, and `P_4(k)`
- Taylor emulation of native multipole predictions over non-fixed,
  non-marginalized theory parameters
- Direct parity tests against installed `CLASS-PT`

Current limitations:

- IR resummation is not implemented in the `jaxpt` backend
- AP effects are not implemented in the `jaxpt` backend

## Validation

Run the test suite with:

```bash
pytest -q
```

The suite includes direct comparisons against the installed `CLASS-PT` backend where available.

## Apple Metal Benchmarking

This repository includes a separate Apple Silicon benchmark environment at
[`envs/apple-metal-jax-benchmark.yml`](/Users/epaillas/code/jax-pt/envs/apple-metal-jax-benchmark.yml).
It is intentionally isolated from the main project environment because Apple
Metal support uses a separate JAX plugin stack and may require older pinned
`jax`/`jaxlib` versions than the rest of the repo.

Create the benchmark environment with:

```bash
conda env create -f envs/apple-metal-jax-benchmark.yml
conda activate jaxpt-apple-metal-benchmark
```

Then verify the active backend and benchmark the native JAX kernels with:

```bash
python scripts/benchmark_jax_backend.py --output-json scripts/benchmark_outputs/jax_backend_cpu.json
JAX_PLATFORMS=METAL python scripts/benchmark_jax_backend.py --require-gpu --output-json scripts/benchmark_outputs/jax_backend_metal.json
```

The benchmark targets the native JAX core only:

- `build_realspace_predictor(...)`
- `GalaxyPowerSpectrumMultipolesTheory._predict_multipoles(...)`

Both timings synchronize with `.block_until_ready()` so the reported latencies
reflect actual device execution rather than queued dispatch.

`jaxpt` defaults to 64-bit mode on CPU, but when `JAX_PLATFORMS=METAL` it
automatically leaves JAX in 32-bit mode because Apple Metal does not support
the `float64`/`complex128` traces used by the native kernels.

At the moment, tree-level native predictions run on Metal, but the one-loop
FFTLog path still fails on `jax-metal==0.1.1` because the backend does not
legalize the `mhlo.fft` operation emitted by `jnp.fft.fft`.

## Emulator Training

Build a hashed Taylor emulator for the multipole theory with:

```bash
python scripts/build_taylor_emulator.py --output-dir scripts/emulator_outputs
```

The script prints the resolved emulated parameters, held fixed or marginalized
parameters, theory settings, and a build progress bar. The saved filename uses a
configuration hash so different theory grids, parameter choices, or Taylor
settings map to different emulator files.

## Documentation

- Flow chart: [docs/flow.md](/Users/epaillas/code/jax-pt/docs/flow.md)
- Package overview: [docs/index.md](/Users/epaillas/code/jax-pt/docs/index.md)
- CLASS-PT tree-level parity notes: [docs/classpt-tree-parity.md](/Users/epaillas/code/jax-pt/docs/classpt-tree-parity.md)
- FFTLog matrix provenance: [docs/matrices.md](/Users/epaillas/code/jax-pt/docs/matrices.md)
- MCMC bottleneck notes: [docs/mcmc-bottleneck-analysis.md](/Users/epaillas/code/jax-pt/docs/mcmc-bottleneck-analysis.md)

## Authors and References

`jaxpt` builds on the public CLASS-PT codebase by Mikhail Ivanov, Anton Chudaykin, Marko Simonovic, Oliver Philcox, and Giovanni Cabass.

- CLASS-PT repository: https://github.com/Michalychforever/CLASS-PT
- Anton Chudaykin, Mikhail M. Ivanov, Oliver H. E. Philcox, Marko Simonovic, "Non-linear perturbation theory extension of the Boltzmann code CLASS," Phys. Rev. D 102, 063533 (2020): https://arxiv.org/abs/2004.10607
