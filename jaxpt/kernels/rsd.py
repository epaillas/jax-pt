from __future__ import annotations

import jax.numpy as jnp


def compute_linear_rsd_terms(real_tree_matter: jnp.ndarray, growth_rate: float) -> dict[str, jnp.ndarray]:
    """Build the tree-level RSD multipole basis terms from `P_lin(k)`."""
    f = float(growth_rate)
    return {
        "rsd_l0_mm_00": (f**2 / 5.0) * real_tree_matter,
        "rsd_l0_mm_01": (2.0 * f / 3.0) * real_tree_matter,
        "rsd_l0_mm_11": real_tree_matter,
        "rsd_l2_mm_00": (4.0 * f**2 / 7.0) * real_tree_matter,
        "rsd_l2_mm_01": (4.0 * f / 3.0) * real_tree_matter,
        "rsd_l2_11": jnp.zeros_like(real_tree_matter),
        "rsd_l4_mm_00": (8.0 * f**2 / 35.0) * real_tree_matter,
    }


def compute_counterterm_multipoles(real_counterterm_shape: jnp.ndarray, growth_rate: float) -> dict[str, jnp.ndarray]:
    """Build the linear counterterm multipole shapes used in the assembly layer."""
    f = float(growth_rate)
    return {
        "rsd_l0_counterterm_shape": real_counterterm_shape,
        "rsd_l2_counterterm_shape": (2.0 * f / 3.0) * real_counterterm_shape,
        "rsd_l4_counterterm_shape": (8.0 * f**2 / 35.0) * real_counterterm_shape,
    }
