from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .bias import galaxy_multipoles as _galaxy_multipoles
from .bias import galaxy_real_spectrum as _galaxy_real_spectrum
from .bias import matter_real_spectrum as _matter_real_spectrum
from .config import EFTBiasParams, PTSettings
from .cosmology import LinearPowerInput
from .kernels.spectral import _loglog_interpolate_jax, compute_native_realspace_terms_from_arrays
from .native import compute_basis as _compute_basis
from .reference.classpt import BasisSpectra
from .theory import GalaxyPowerSpectrumMultipolesTheory, PowerSpectrumTemplate


def compute_basis(linear_input: LinearPowerInput, settings: PTSettings | None = None, k: np.ndarray | None = None) -> BasisSpectra:
    return _compute_basis(linear_input=linear_input, settings=settings, k=k)


def galaxy_multipoles(
    basis: BasisSpectra,
    params: EFTBiasParams,
    *,
    return_components: bool = False,
):
    return _galaxy_multipoles(basis=basis, params=params, return_components=return_components)


def matter_real_spectrum(basis: BasisSpectra, cs: float):
    return _matter_real_spectrum(basis=basis, cs=cs)


def galaxy_real_spectrum(
    basis: BasisSpectra,
    b1: float,
    b2: float,
    bG2: float,
    bGamma3: float,
    cs: float,
    cs0: float,
    Pshot: float,
):
    return _galaxy_real_spectrum(
        basis=basis,
        b1=b1,
        b2=b2,
        bG2=bG2,
        bGamma3=bGamma3,
        cs=cs,
        cs0=cs0,
        Pshot=Pshot,
    )


def build_native_realspace_predictor(
    linear_input: LinearPowerInput,
    settings: PTSettings | None = None,
    k: np.ndarray | None = None,
):
    """Build a compiled native real-space galaxy-spectrum predictor.

    The returned callable is intended for repeated evaluations on a fixed
    linear-theory support grid. It keeps the FFTLog loop kernel in JAX and
    avoids rebuilding the basis object on each call.
    """
    if settings is None:
        settings = PTSettings(ir_resummation=False)

    if settings.backend != "native":
        raise ValueError("build_native_realspace_predictor only supports settings.backend == 'native'.")
    if settings.loop_order not in {"tree", "one_loop"}:
        raise ValueError("build_native_realspace_predictor only supports settings.loop_order in {'tree', 'one_loop'}.")
    if settings.ir_resummation:
        raise NotImplementedError("Native IR-resummed one-loop kernels are not implemented yet.")
    if settings.ap_effect:
        raise NotImplementedError("AP support is not implemented in the native backend yet.")

    support_k = jnp.asarray(np.asarray(linear_input.k, dtype=float))
    pk_linear = jnp.asarray(np.asarray(linear_input.pk_linear, dtype=float))
    output_k = support_k if k is None else jnp.asarray(np.asarray(k, dtype=float))
    h = float(linear_input.h)

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
            loop_terms = compute_native_realspace_terms_from_arrays(
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


def predict_galaxy_multipoles(
    source,
    *args,
    params: EFTBiasParams | None = None,
    settings: PTSettings | None = None,
    return_components: bool = False,
):
    if isinstance(source, GalaxyPowerSpectrumMultipolesTheory):
        if params is None:
            if not args:
                raise TypeError("predict_galaxy_multipoles requires params when called with a GalaxyPowerSpectrumMultipolesTheory source.")
            params = args[0]
        return source(params, return_components=return_components)

    if isinstance(source, BasisSpectra):
        if params is None:
            if not args:
                raise TypeError("predict_galaxy_multipoles requires params when called with a BasisSpectra source.")
            params = args[0]
        basis = source
    elif isinstance(source, LinearPowerInput):
        if params is None:
            if not args:
                raise TypeError("predict_galaxy_multipoles requires params when called with a LinearPowerInput source.")
            params = args[0]
        theory = GalaxyPowerSpectrumMultipolesTheory(
            template=PowerSpectrumTemplate.from_linear_input(source, settings=settings),
            k=np.asarray(source.k, dtype=float),
            return_components=return_components,
        )
        return theory(params, return_components=return_components)
    else:
        raise TypeError(
            "predict_galaxy_multipoles only accepts GalaxyPowerSpectrumMultipolesTheory, BasisSpectra, or LinearPowerInput. "
            "Use build_linear_input_from_classy(...) before calling it."
        )

    return galaxy_multipoles(basis, params, return_components=return_components)
