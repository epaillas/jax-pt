from __future__ import annotations

import jax.numpy as jnp

from ..config import PTSettings
from ..cosmology import LinearPowerInput, NativeFFTLogInput
from .rsd_spectral import compute_native_rsd_terms
from .spectral import compute_native_realspace_terms


_RSD_LOOP_COMPONENT_NAMES = (
    "rsd_l0_loop_00",
    "rsd_l0_loop_01",
    "rsd_l0_loop_11",
    "rsd_l2_loop_00",
    "rsd_l2_loop_01",
    "rsd_l2_11",
    "rsd_l4_loop_00",
    "rsd_l4_loop_01",
    "rsd_l4_loop_11",
    "rsd_l0_b1_b2",
    "rsd_l0_b2",
    "rsd_l0_b1_bG2",
    "rsd_l0_bG2",
    "rsd_l2_b1_b2",
    "rsd_l2_b2",
    "rsd_l2_b1_bG2",
    "rsd_l2_bG2",
    "rsd_l4_b2",
    "rsd_l4_bG2",
    "rsd_l0_gamma3_b1",
    "rsd_l0_gamma3_bias",
    "rsd_l2_gamma3",
)

_REAL_LOOP_COMPONENT_NAMES = (
    "real_loop_matter",
    "real_loop_b2_b2",
    "real_cross_b1_b2",
    "real_cross_b1_bG2",
    "real_loop_b2_bG2",
    "real_loop_bG2_bG2",
    "real_gamma3",
)


def _zero_real_loop_terms(k: jnp.ndarray) -> dict[str, jnp.ndarray]:
    zeros = jnp.zeros_like(k)
    return {name: zeros for name in _REAL_LOOP_COMPONENT_NAMES}


def _zero_rsd_loop_terms(k: jnp.ndarray) -> dict[str, jnp.ndarray]:
    zeros = jnp.zeros_like(k)
    return {name: zeros for name in _RSD_LOOP_COMPONENT_NAMES}


def compute_real_loop_terms(
    linear_input: LinearPowerInput,
    settings: PTSettings,
    output_k: jnp.ndarray | None = None,
    fftlog_input: NativeFFTLogInput | None = None,
) -> dict[str, jnp.ndarray]:
    k = jnp.asarray(linear_input.k) if output_k is None else output_k
    if settings.loop_order == "tree":
        return _zero_real_loop_terms(k)
    if settings.loop_order != "one_loop":
        raise ValueError("Native compute_real_loop_terms only supports settings.loop_order in {'tree', 'one_loop'}.")

    if fftlog_input is None:
        native = compute_native_realspace_terms(linear_input, settings, output_k=output_k)
    else:
        from .spectral import compute_native_realspace_terms_from_preprocessed

        native = compute_native_realspace_terms_from_preprocessed(fftlog_input, settings, output_k=output_k)
    return {name: jnp.asarray(values) for name, values in native.items()}


def compute_rsd_loop_terms(
    linear_input: LinearPowerInput,
    settings: PTSettings,
    output_k: jnp.ndarray | None = None,
    fftlog_input: NativeFFTLogInput | None = None,
) -> dict[str, jnp.ndarray]:
    if settings.loop_order == "tree":
        k = jnp.asarray(linear_input.k) if output_k is None else output_k
        return _zero_rsd_loop_terms(k)
    if settings.loop_order != "one_loop":
        raise ValueError("Native compute_rsd_loop_terms only supports settings.loop_order in {'tree', 'one_loop'}.")
    native = compute_native_rsd_terms(linear_input, settings, output_k=output_k, fftlog_input=fftlog_input)
    return {name: jnp.asarray(native[name]) for name in _RSD_LOOP_COMPONENT_NAMES}
