from __future__ import annotations

import jax.numpy as jnp

from ..basis import make_basis
from ..cosmology import LinearPowerInput, prepare_fftlog_input
from ..reference.classpt import BasisSpectra
from .loops import compute_real_loop_terms, compute_rsd_loop_terms
from .rsd import compute_counterterm_multipoles, compute_linear_rsd_terms
from .tree import compute_counterterm_shape, compute_real_tree_matter


def compute_tree_level_basis(linear_input: LinearPowerInput, settings, output_k: jnp.ndarray | None = None) -> BasisSpectra:
    """Construct the staged jaxpt basis from linear theory.

    The current implementation wires together the kernel stages and returns the
    linear Kaiser contribution plus the tree-level counterterm shapes. One-loop
    physics terms still enter as zeros through the dedicated loop-stage
    functions so future work can replace them without changing the public
    `BasisSpectra` contract.
    """

    k = jnp.asarray(linear_input.k) if output_k is None else output_k
    h = float(linear_input.h)
    f = float(linear_input.growth_rate)

    real_tree_matter = compute_real_tree_matter(linear_input, output_k=output_k)
    real_counterterm_shape = compute_counterterm_shape(linear_input, output_k=output_k)
    fftlog_input = None
    if settings.loop_order == "one_loop":
        fftlog_input = prepare_fftlog_input(linear_input, settings)

    components = {
        "real_tree_matter": real_tree_matter,
        "real_counterterm_shape": real_counterterm_shape,
        "k_over_h_squared": (k / h) ** 2,
    }
    components.update(compute_real_loop_terms(linear_input, settings, output_k=k, fftlog_input=fftlog_input))
    components.update(compute_linear_rsd_terms(real_tree_matter, f))
    components.update(compute_counterterm_multipoles(real_counterterm_shape, f))
    components.update(compute_rsd_loop_terms(linear_input, settings, output_k=k, fftlog_input=fftlog_input))

    metadata = {
        "backend": settings.backend,
        "approximation": "linear_kaiser" if settings.loop_order == "tree" else "analytic_fftlog_realspace",
        "kernel_source": "tree" if settings.loop_order == "tree" else settings.kernel_source,
        "spectral_method": "none" if settings.loop_order == "tree" else "fftlog_jax",
        "resummed": False,
        "ir_resummation": False,
        "cb": bool(linear_input.metadata.get("field", "cb") == "cb"),
        "field": linear_input.metadata.get("field", "cb"),
        "k_units": linear_input.metadata.get("k_units", "1/Mpc"),
        "pk_units": linear_input.metadata.get("pk_units", "Mpc^3"),
        "rsd": True,
        "realspace_matter": settings.loop_order != "tree",
        "realspace_bias": settings.loop_order != "tree",
        "rsd_matter": True,
        "rsd_bias": True,
        "pk_mult_shape": None,
    }
    if fftlog_input is not None:
        metadata["fftlog_grid_source"] = fftlog_input.metadata.get("fftlog_grid_source", "fftlog_kdisc")
        metadata["fftlog_input_aligned"] = fftlog_input.metadata.get("fftlog_input_aligned")
    return make_basis(
        k=k,
        z=float(linear_input.z),
        h=h,
        growth_rate=f,
        components=components,
        metadata=metadata,
    )
