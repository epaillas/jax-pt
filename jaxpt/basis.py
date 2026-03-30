from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .config import PTSettings
from .cosmology import LinearPowerInput
from .reference.classpt import BASIS_COMPONENT_NAMES, BasisSpectra


def empty_components(k: jnp.ndarray) -> dict[str, jnp.ndarray]:
    zeros = jnp.zeros_like(k)
    return {name: zeros for name in BASIS_COMPONENT_NAMES}


def make_basis(
    *,
    k: jnp.ndarray,
    z: float,
    h: float,
    growth_rate: float,
    components: dict[str, jnp.ndarray],
    metadata: dict[str, Any] | None = None,
) -> BasisSpectra:
    payload = empty_components(k)
    payload.update(components)
    return BasisSpectra(
        k=k,
        z=float(z),
        h=float(h),
        growth_rate=float(growth_rate),
        components=payload,
        metadata=dict(metadata or {}),
    )


def compute_basis(
    linear_input: LinearPowerInput,
    settings: PTSettings | None = None,
    k: np.ndarray | None = None,
) -> BasisSpectra:
    """Compute a `jaxpt` basis from linear inputs.

    The `jaxpt` backend supports explicit theory-order selection through
    ``settings.loop_order``. Tree-level and one-loop real-space predictions
    share the same basis-construction interface so the public `BasisSpectra`
    contract stays stable as more kernels are added.
    """
    if settings is None:
        settings = PTSettings(ir_resummation=False)

    if settings.backend != "jaxpt":
        raise ValueError("compute_basis only supports settings.backend == 'jaxpt'.")
    if settings.loop_order not in {"tree", "one_loop"}:
        raise ValueError("compute_basis only supports settings.loop_order in {'tree', 'one_loop'}.")

    if settings.ap_effect:
        raise NotImplementedError("AP support is not implemented in the jaxpt backend yet.")
    if settings.ir_resummation:
        if settings.require_nowiggle and linear_input.pk_nowiggle is None:
            raise ValueError("pk_nowiggle is required when require_nowiggle=True.")
        raise NotImplementedError("IR-resummed one-loop kernels are not implemented yet.")
    if not settings.rsd:
        raise NotImplementedError("The jaxpt backend currently targets redshift-space outputs.")

    from .kernels import compute_tree_level_basis

    output_k = None if k is None else np.asarray(k, dtype=float)
    basis = compute_tree_level_basis(linear_input, settings, output_k=output_k)

    if settings.kmin is None and settings.kmax is None:
        return basis

    kmin = settings.kmin if settings.kmin is not None else float(np.asarray(basis.k)[0])
    kmax = settings.kmax if settings.kmax is not None else float(np.asarray(basis.k)[-1])
    mask = (np.asarray(basis.k) >= kmin) & (np.asarray(basis.k) <= kmax)
    components = {name: values[mask] for name, values in basis.components.items()}
    return BasisSpectra(
        k=basis.k[mask],
        z=basis.z,
        h=basis.h,
        growth_rate=basis.growth_rate,
        components=components,
        metadata=basis.metadata,
    )


def build_realspace_predictor(
    linear_input: LinearPowerInput,
    settings: PTSettings | None = None,
    k: np.ndarray | None = None,
):
    """Build a compiled real-space galaxy-spectrum predictor."""
    if settings is None:
        settings = PTSettings(ir_resummation=False)

    if settings.backend != "jaxpt":
        raise ValueError("build_realspace_predictor only supports settings.backend == 'jaxpt'.")
    if settings.loop_order not in {"tree", "one_loop"}:
        raise ValueError("build_realspace_predictor only supports settings.loop_order in {'tree', 'one_loop'}.")
    if settings.ir_resummation:
        raise NotImplementedError("IR-resummed one-loop kernels are not implemented yet.")
    if settings.ap_effect:
        raise NotImplementedError("AP support is not implemented in the jaxpt backend yet.")

    support_k = jnp.asarray(np.asarray(linear_input.k, dtype=float))
    pk_linear = jnp.asarray(np.asarray(linear_input.pk_linear, dtype=float))
    output_k = support_k if k is None else jnp.asarray(np.asarray(k, dtype=float))
    h = float(linear_input.h)
    from .kernels.spectral import _loglog_interpolate_jax, compute_fftlog_realspace_terms_from_arrays

    @jax.jit
    def _predict(
        b1: float,
        b2: float,
        bG2: float,
        bGamma3: float,
        cs: float,
        cs0: float,
        Pshot: float,
    ) -> jnp.ndarray:
        real_tree_matter = _loglog_interpolate_jax(output_k, support_k, pk_linear)
        real_counterterm_shape = -(output_k**2) * real_tree_matter
        if settings.loop_order == "tree":
            zeros = jnp.zeros_like(output_k)
            loop_terms = {
                "real_loop_matter": zeros,
                "real_loop_b2_b2": zeros,
                "real_cross_b1_b2": zeros,
                "real_cross_b1_bG2": zeros,
                "real_loop_b2_bG2": zeros,
                "real_loop_bG2_bG2": zeros,
                "real_gamma3": zeros,
            }
        else:
            loop_terms = compute_fftlog_realspace_terms_from_arrays(
                support_k=support_k,
                pk_linear=pk_linear,
                output_k=output_k,
                h=h,
                settings=settings,
            )
        return (
            b1**2 * loop_terms["real_loop_matter"]
            + b1**2 * real_tree_matter
            + 2.0 * (cs * b1**2 + cs0 * b1) * real_counterterm_shape / h**2
            + b1 * b2 * loop_terms["real_cross_b1_b2"]
            + 0.25 * b2**2 * loop_terms["real_loop_b2_b2"]
            + 2.0 * b1 * bG2 * loop_terms["real_cross_b1_bG2"]
            + b1 * (2.0 * bG2 + 0.8 * bGamma3) * loop_terms["real_gamma3"]
            + bG2**2 * loop_terms["real_loop_bG2_bG2"]
            + b2 * bG2 * loop_terms["real_loop_b2_bG2"]
        ) * h**3 + Pshot

    return _predict
