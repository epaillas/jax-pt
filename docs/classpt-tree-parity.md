# CLASS-PT Tree-Level Parity

`jaxpt` defaults to the direct linear spectrum from `classy`:

- `LinearPowerInput.pk_linear <- cosmo.pk_lin(k, z)`

This remains the default because it is the clean public linear-theory input and
matches the intended native tree-level Kaiser interpretation.

## Tree-Level Mismatch Origin

During parity checks against direct `CLASS-PT`, the tree-level basis term
`pk_mult[14]` was found to differ smoothly from `cosmo.pk_lin(k, z)` by roughly
`0.4%-1.0%` over the standard `0.01 <= k <= 0.2 1/Mpc` range.

The key result is:

- `cosmo.pk(k, z)[14] == cosmo.get_pk_mult(k, z, len(k))[14]`
- `cosmo.pk(k, z)[14] != cosmo.pk_lin(k, z)`

This is not an indexing bug in `jaxpt`. It comes from how `CLASS-PT` builds its
tree basis internally.

## What CLASS-PT Does

In `CLASS-PT/source/nonlinear_pt.c`, the tree basis used by `pk_mult[14]` is
not built from direct public `pk_lin(k, z)` calls.

Instead, `CLASS-PT`:

1. reconstructs an internal linear spectrum table `lnpk_l(log k)` from the
   perturbation source arrays,
2. natural-spline interpolates that table onto its internal FFTLog grid
   `kdisc` to build `Pdisc` / `Pbin`,
3. forms the tree multipoles from `Pbin`,
4. natural-spline interpolates those tree multipoles back to the requested
   output `k`.

So the exported tree basis term is effectively a backend-specific reconstructed
linear spectrum, not the raw public `pk_lin(k, z)` value.

## Policy In jaxpt

`jaxpt` keeps the direct `pk_lin(k, z)` path as the default production input.
That is the cleaner and more explicit native-theory input.

For strict backend parity checks, `jaxpt` also provides a separate helper:

- `jaxpt.cosmology.build_classpt_parity_linear_input_from_classy(...)`

This parity-only helper fills `LinearPowerInput.pk_linear` from
`cosmo.pk(k, z)[14]` so tests can compare against `CLASS-PT` using the same
effective tree input that `CLASS-PT` uses internally.

This helper should be treated as a validation compatibility path, not as the
default physical definition of the linear spectrum.
