# MCMC Bottleneck Analysis

This note records a profiling pass over the current cosmology-varying
prediction workflow in `jaxpt`, with the goal of identifying whether the main
MCMC bottleneck lives in `classy` or in the `jaxpt` one-loop calculations.

## Setup

The timings below were collected on the local development machine using the
same workflow as the example script:

- redshift: `z = 0.5`
- linear support grid: `256` points
- evaluation grid: `96` points
- settings: `PTSettings(ir_resummation=False)`
- observable: `P0`, `P2`, `P4`

The profiled call path was:

1. build a fresh `classy.Class`
2. call `cosmo.compute()`
3. convert the result with `build_linear_input_from_classy(...)`
4. run `compute_basis(...)`
5. assemble multipoles with `galaxy_multipoles(...)`

## Main Result

For cosmology-varying sampling, the dominant cost is `cosmo.compute()`.

Warm steady-state timing for the one-loop path:

- `classy` init + `set()`: effectively zero
- `cosmo.compute()`: about `3.34 s`
- `build_linear_input_from_classy(...)`: about `0.00038 s`
- `compute_basis(..., loop_order="one_loop")`: about `0.105 s`
- `galaxy_multipoles(...)`: about `0.00035 s`
- total: about `3.45 s`

On this machine, `cosmo.compute()` is about `97%` of the steady-state
cosmology-varying wall time.

The practical conclusion is that repeated CLASS solves dominate the current
MCMC cost. The `jaxpt` one-loop basis construction is visible, but secondary.

## Cold Versus Warm Behavior

The first one-loop evaluation is substantially slower than later evaluations.
That extra cost is not representative of steady-state MCMC usage, but it does
explain why the first sample can look much worse than subsequent samples.

Measured cold and warm timings:

- jaxpt real-space loops:
  - cold: about `1.26 s`
  - warm: about `0.02 s`
- jaxpt RSD loops:
  - cold: about `0.66 s`
  - warm: about `0.08 s`
- full jaxpt basis:
  - cold: about `0.108 s` after caches are populated at the stage level
  - warm: about `0.105 s`

The cold-start penalty is driven by first-use setup in the JAX loop
path and by transfer-kernel compilation in the RSD code. Once warm, the jaxpt
one-loop path is much smaller than the CLASS solve.

## Breakdown Inside `jaxpt`

The current jaxpt prediction pipeline is wired through
[jaxpt/kernels/linear.py](/Users/epaillas/code/jax-pt/jaxpt/kernels/linear.py),
where `compute_tree_level_basis(...)` calls:

- `prepare_fftlog_input(...)`
- `compute_real_loop_terms(...)`
- `compute_rsd_loop_terms(...)`

The measured warm costs show:

- `prepare_fftlog_input(...)` in
  [jaxpt/cosmology.py](/Users/epaillas/code/jax-pt/jaxpt/cosmology.py): about
  `4e-5 s`
- real-space loop terms in
  [jaxpt/kernels/spectral.py](/Users/epaillas/code/jax-pt/jaxpt/kernels/spectral.py):
  about `0.02 s`
- RSD loop terms in
  [jaxpt/kernels/rsd_spectral.py](/Users/epaillas/code/jax-pt/jaxpt/kernels/rsd_spectral.py):
  about `0.08 s`

So the dominant jaxpt cost is the RSD one-loop path, not the real-space path.

`galaxy_multipoles(...)` in
[jaxpt/bias.py](/Users/epaillas/code/jax-pt/jaxpt/bias.py)
is not a bottleneck. It is a fast algebraic assembly over the precomputed basis
and runs in well under a millisecond once the basis exists.

## `classy` Versus `CLASS-PT` Flags

To separate generic CLASS cost from the extra PT configuration cost, the local
profiling compared two fresh `compute()` calls:

- plain linear CLASS with `output = "mPk"`: about `3.05 s`
- current CLASS-PT-style configuration:
  - `non linear = "PT"`
  - `IR resummation = "No"`
  - `Bias tracers = "Yes"`
  - `cb = "Yes"`
  - `RSD = "Yes"`
  - timing: about `3.38 s`

This means the PT-related flags do add cost, but only at the
few-hundred-millisecond level on this setup. Even the plain CLASS solve is
already around `3 s`, so the main bottleneck is not specific to the
`jaxpt` loop code.

## Why `build_linear_input_from_classy(...)` Is Not The Problem

`build_linear_input_from_classy(...)` samples `pk_lin` in a Python loop:

- implementation:
  [jaxpt/cosmology.py:161](/Users/epaillas/code/jax-pt/jaxpt/cosmology.py:161)
- measured time for `256` support-grid points: about `0.00026` to `0.00029 s`

That is negligible compared with both `cosmo.compute()` and the jaxpt one-loop
path.

## Profiling Notes

A `cProfile` pass over warm `compute_basis(...)` showed the hottest jaxpt
region inside the RSD loop builder:

- `compute_rsd_loop_terms(...)` in
  [jaxpt/kernels/loops.py:78](/Users/epaillas/code/jax-pt/jaxpt/kernels/loops.py:78)
- `compute_fftlog_rsd_terms(...)` in
  [jaxpt/kernels/rsd_spectral.py:124](/Users/epaillas/code/jax-pt/jaxpt/kernels/rsd_spectral.py:124)

Within that path, repeated projection to the output grid through
`_interpolate_to_output_jax(...)` in
[jaxpt/kernels/spectral.py:159](/Users/epaillas/code/jax-pt/jaxpt/kernels/spectral.py:159)
shows up prominently in cumulative time, along with the surrounding matrix-stack
evaluations.

## Takeaways

- The immediate blocker for cosmology-varying MCMC is repeated
  `cosmo.compute()`.
- Jaxpt one-loop work matters, but it is about `0.1 s`, not multiple seconds.
- Inside the jaxpt one-loop path, RSD terms dominate the cost.
- If the goal is a large speedup for cosmology-varying sampling, reducing or
  avoiding the per-sample CLASS solve is the highest-leverage target.
