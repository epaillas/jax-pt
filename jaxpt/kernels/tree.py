from __future__ import annotations

import jax.numpy as jnp

from ..cosmology import LinearPowerInput


def _interpolate_on_logk(source_k: jnp.ndarray, values: jnp.ndarray, output_k: jnp.ndarray) -> jnp.ndarray:
    log_source_k = jnp.log(source_k)
    log_output_k = jnp.log(output_k)
    slopes = jnp.empty_like(values)
    slopes = slopes.at[0].set((values[1] - values[0]) / (log_source_k[1] - log_source_k[0]))
    slopes = slopes.at[-1].set((values[-1] - values[-2]) / (log_source_k[-1] - log_source_k[-2]))
    slopes = slopes.at[1:-1].set((values[2:] - values[:-2]) / (log_source_k[2:] - log_source_k[:-2]))

    interval_index = jnp.clip(jnp.searchsorted(log_source_k, log_output_k, side="right") - 1, 0, log_source_k.size - 2)
    x0 = log_source_k[interval_index]
    x1 = log_source_k[interval_index + 1]
    y0 = values[interval_index]
    y1 = values[interval_index + 1]
    m0 = slopes[interval_index]
    m1 = slopes[interval_index + 1]
    dx = x1 - x0
    t = (log_output_k - x0) / dx
    t2 = t * t
    t3 = t2 * t

    hermite = (
        (2.0 * t3 - 3.0 * t2 + 1.0) * y0
        + (t3 - 2.0 * t2 + t) * dx * m0
        + (-2.0 * t3 + 3.0 * t2) * y1
        + (t3 - t2) * dx * m1
    )

    hermite = jnp.where(
        log_output_k < log_source_k[0],
        values[0] + slopes[0] * (log_output_k - log_source_k[0]),
        hermite,
    )
    hermite = jnp.where(
        log_output_k > log_source_k[-1],
        values[-1] + slopes[-1] * (log_output_k - log_source_k[-1]),
        hermite,
    )
    return hermite


def compute_real_tree_matter(linear_input: LinearPowerInput, output_k: jnp.ndarray | None = None) -> jnp.ndarray:
    """Return the linear real-space matter spectrum on the requested grid.

    Parameters
    ----------
    linear_input
        Linear-theory input sampled on its support grid.
    output_k
        Optional output grid in ``1/Mpc``. If omitted, return the stored
        support-grid values directly.
    """
    source_k = jnp.asarray(linear_input.k)
    values = jnp.asarray(linear_input.pk_linear)
    if output_k is None:
        return values
    return _interpolate_on_logk(source_k, values, output_k)


def compute_counterterm_shape(linear_input: LinearPowerInput, output_k: jnp.ndarray | None = None) -> jnp.ndarray:
    """Return the standard real-space counterterm shape ``-k^2 P_lin(k)``."""
    k = jnp.asarray(linear_input.k) if output_k is None else output_k
    return -(k**2) * compute_real_tree_matter(linear_input, output_k=output_k)
