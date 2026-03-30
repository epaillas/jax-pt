# jaxpt

`jaxpt` is structured around explicit basis containers and pure assembly functions.

Current public workflow:

1. Build a `CLASS-PT` cosmology with the local `classy` installation.
2. Convert linear theory to `LinearPowerInput` with `build_linear_input_from_classy`.
3. Build basis spectra with `compute_basis`.
4. Assemble `P_0`, `P_2`, and `P_4` with `galaxy_multipoles`.

Current backends:

1. `compute_basis(...)` uses the `jaxpt` backend and supports explicit theory order selection: `loop_order="tree"` for the tree-level Kaiser basis and `loop_order="one_loop"` for the current non-resummed one-loop basis.
2. `CLASS-PT` is used as a direct validation oracle in tests and benchmarks, not as a basis backend inside the public API.

The one-loop basis now fills both real-space and RSD loop components behind the same `BasisSpectra` contract.

The current execution flow for prediction calls is documented in
[docs/flow.md](/Users/epaillas/code/jax-pt/docs/flow.md).

The current `jaxpt`/reference plotting example is:

```bash
python examples/native_realspace_vs_classpt.py
```

The example compares the analytic FFTLog `jaxpt` real-space backend against
`CLASS-PT` itself on `0.01 <= k <= 0.2 1/Mpc`.

The original `CLASS-PT` FFTLog matrix assets and their provenance are documented in
[docs/matrices.md](/Users/epaillas/code/jax-pt/docs/matrices.md).

The tree-level parity mismatch against direct `CLASS-PT` and the dedicated
parity-only linear-spectrum helper are documented in
[docs/classpt-tree-parity.md](/Users/epaillas/code/jax-pt/docs/classpt-tree-parity.md).
