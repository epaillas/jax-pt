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

from jaxpt import EFTBiasParams, PTSettings, build_linear_input_from_classy, compute_basis, galaxy_multipoles

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

linear_input = build_linear_input_from_classy(cosmo, z=z, k=support_k)
basis = compute_basis(linear_input, settings=PTSettings(ir_resummation=False), k=eval_k)
params = EFTBiasParams(
    b1=2.0,
    b2=-1.0,
    bG2=0.1,
    bGamma3=-0.1,
    cs0=0.0,
    cs2=30.0,
    cs4=0.0,
    Pshot=3000.0,
    b4=10.0,
)
prediction = galaxy_multipoles(basis, params)
```

## Current Scope

- Native tree-level Kaiser basis generation from `LinearPowerInput`
- Native one-loop real-space matter and bias terms from analytic FFTLog kernels generated in-repo
- Native assembly of `P_0(k)`, `P_2(k)`, and `P_4(k)`
- Direct parity tests against installed `CLASS-PT`

Current limitations:

- IR resummation is not implemented in the native backend
- AP effects are not implemented in the native backend

## Validation

Run the test suite with:

```bash
pytest -q
```

The suite includes direct comparisons against the installed `CLASS-PT` backend where available.

## Authors and References

`jaxpt` builds on the public CLASS-PT codebase by Mikhail Ivanov, Anton Chudaykin, Marko Simonovic, Oliver Philcox, and Giovanni Cabass.

- CLASS-PT repository: https://github.com/Michalychforever/CLASS-PT
- Anton Chudaykin, Mikhail M. Ivanov, Oliver H. E. Philcox, Marko Simonovic, "Non-linear perturbation theory extension of the Boltzmann code CLASS," Phys. Rev. D 102, 063533 (2020): https://arxiv.org/abs/2004.10607
