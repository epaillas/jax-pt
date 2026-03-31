from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from .reference.classpt import BasisSpectra, MultipolePrediction


def _c(basis: BasisSpectra, name: str) -> jnp.ndarray:
    return basis.components[name]


def _p(params: Mapping[str, float], name: str) -> float:
    return float(params[name])


def matter_real_spectrum(basis: BasisSpectra, cs: float) -> jnp.ndarray:
    """Assemble the real-space matter power spectrum from named basis terms.

    Parameters
    ----------
    basis
        Basis container with ``real_tree_matter``, ``real_loop_matter``, and
        ``real_counterterm_shape`` components.
    cs
        EFT counterterm coefficient multiplying the matter counterterm shape.
    """
    h = basis.h
    return (_c(basis, "real_tree_matter") + _c(basis, "real_loop_matter") + 2.0 * cs * _c(basis, "real_counterterm_shape") / h**2) * h**3


def galaxy_real_spectrum(
    basis: BasisSpectra,
    b1: float,
    b2: float,
    bG2: float,
    bGamma3: float,
    cs: float,
    cs0: float,
    Pshot: float,
) -> jnp.ndarray:
    """Assemble the real-space galaxy power spectrum from a basis and biases.

    Parameters
    ----------
    basis
        Basis container containing the named real-space matter and bias terms.
    b1, b2, bG2, bGamma3
        Eulerian bias parameters.
    cs
        Matter-like counterterm coefficient.
    cs0
        Galaxy counterterm coefficient for the real-space monopole-like shape.
    Pshot
        Constant shot-noise contribution added after the physical terms.
    """
    h = basis.h
    return (
        b1**2 * _c(basis, "real_loop_matter")
        + b1**2 * _c(basis, "real_tree_matter")
        + 2.0 * (cs * b1**2 + cs0 * b1) * _c(basis, "real_counterterm_shape") / h**2
        + b1 * b2 * _c(basis, "real_cross_b1_b2")
        + 0.25 * b2**2 * _c(basis, "real_loop_b2_b2")
        + 2.0 * b1 * bG2 * _c(basis, "real_cross_b1_bG2")
        + b1 * (2.0 * bG2 + 0.8 * bGamma3) * _c(basis, "real_gamma3")
        + bG2**2 * _c(basis, "real_loop_bG2_bG2")
        + b2 * bG2 * _c(basis, "real_loop_b2_bG2")
    ) * h**3 + Pshot


def galaxy_multipoles(
    basis: BasisSpectra,
    params: Mapping[str, float],
    return_components: bool = False,
) -> MultipolePrediction:
    """Assemble `P0`, `P2`, and `P4` from a basis and nuisance parameters.

    Parameters
    ----------
    basis
        Basis container with the named real-space, RSD, and counterterm
        components expected by the CLASS-PT-style assembly formulas.
    params
        Flat nuisance-parameter mapping. Required names are ``b1``, ``b2``,
        ``bG2``, ``bGamma3``, ``cs0``, ``cs2``, ``cs4``, ``Pshot``, and
        ``b4``.
    return_components
        If ``True``, include a compact decomposition of each multipole into its
        core and ``b4`` contributions.
    """
    h = basis.h
    f = basis.growth_rate
    b1 = _p(params, "b1")
    b2 = _p(params, "b2")
    bG2 = _p(params, "bG2")
    bGamma3 = _p(params, "bGamma3")

    p0_core = (
        _c(basis, "rsd_l0_mm_00")
        + _c(basis, "rsd_l0_loop_00")
        + b1 * _c(basis, "rsd_l0_mm_01")
        + b1 * _c(basis, "rsd_l0_loop_01")
        + b1**2 * _c(basis, "rsd_l0_mm_11")
        + b1**2 * _c(basis, "rsd_l0_loop_11")
        + 0.25 * b2**2 * _c(basis, "real_loop_b2_b2")
        + b1 * b2 * _c(basis, "rsd_l0_b1_b2")
        + b2 * _c(basis, "rsd_l0_b2")
        + b1 * bG2 * _c(basis, "rsd_l0_b1_bG2")
        + bG2 * _c(basis, "rsd_l0_bG2")
        + b2 * bG2 * _c(basis, "real_loop_b2_bG2")
        + bG2**2 * _c(basis, "real_loop_bG2_bG2")
        + 2.0 * _p(params, "cs0") * _c(basis, "rsd_l0_counterterm_shape") / h**2
        + (2.0 * bG2 + 0.8 * bGamma3) * (b1 * _c(basis, "rsd_l0_gamma3_b1") + _c(basis, "rsd_l0_gamma3_bias"))
    ) * h**3

    p2_core = (
        _c(basis, "rsd_l2_mm_00")
        + _c(basis, "rsd_l2_loop_00")
        + b1 * _c(basis, "rsd_l2_loop_01")
        + b1 * _c(basis, "rsd_l2_mm_01")
        + b1**2 * _c(basis, "rsd_l2_11")
        + b1 * b2 * _c(basis, "rsd_l2_b1_b2")
        + b2 * _c(basis, "rsd_l2_b2")
        + b1 * bG2 * _c(basis, "rsd_l2_b1_bG2")
        + bG2 * _c(basis, "rsd_l2_bG2")
        + 2.0 * _p(params, "cs2") * _c(basis, "rsd_l2_counterterm_shape") / h**2
        + (2.0 * bG2 + 0.8 * bGamma3) * _c(basis, "rsd_l2_gamma3")
    ) * h**3

    p4_core = (
        _c(basis, "rsd_l4_mm_00")
        + _c(basis, "rsd_l4_loop_00")
        + b1 * _c(basis, "rsd_l4_loop_01")
        + b1**2 * _c(basis, "rsd_l4_loop_11")
        + b2 * _c(basis, "rsd_l4_b2")
        + bG2 * _c(basis, "rsd_l4_bG2")
        + 2.0 * _p(params, "cs4") * _c(basis, "rsd_l4_counterterm_shape") / h**2
    ) * h**3

    b4_shape = (35.0 / 8.0) * _c(basis, "rsd_l4_counterterm_shape") * h
    p0_b4 = f**2 * _p(params, "b4") * _c(basis, "k_over_h_squared") * (f**2 / 9.0 + 2.0 * f * b1 / 7.0 + b1**2 / 5.0) * b4_shape
    p2_b4 = f**2 * _p(params, "b4") * _c(basis, "k_over_h_squared") * ((f**2 * 70.0 + 165.0 * f * b1 + 99.0 * b1**2) * 4.0 / 693.0) * b4_shape
    p4_b4 = f**2 * _p(params, "b4") * _c(basis, "k_over_h_squared") * ((f**2 * 210.0 + 390.0 * f * b1 + 143.0 * b1**2) * 8.0 / 5005.0) * b4_shape

    p0 = p0_core + _p(params, "Pshot") + p0_b4
    p2 = p2_core + p2_b4
    p4 = p4_core + p4_b4

    components = None
    if return_components:
        components = {
            "p0_core": p0_core,
            "p0_b4": p0_b4,
            "p2_core": p2_core,
            "p2_b4": p2_b4,
            "p4_core": p4_core,
            "p4_b4": p4_b4,
        }

    return MultipolePrediction(
        k=basis.k,
        p0=p0,
        p2=p2,
        p4=p4,
        components=components,
        metadata={"backend": basis.metadata.get("backend", "unknown"), "z": basis.z},
    )


def galaxy_multipole_templates(
    basis: BasisSpectra,
    params: Mapping[str, float],
    names: tuple[str, ...] | None = None,
) -> dict[str, MultipolePrediction]:
    """Return linear multipole templates for nuisance parameters.

    Templates are defined around the current parameter baseline, so the full
    prediction can be written as ``P(fiducial) + A (x - x_fiducial)`` for the
    returned nuisance subset.
    """
    requested = ("cs0", "cs2", "cs4", "Pshot", "b4") if names is None else tuple(str(name) for name in names)
    unknown = sorted(set(requested) - {"cs0", "cs2", "cs4", "Pshot", "b4"})
    if unknown:
        raise ValueError(f"Unsupported multipole template parameters: {', '.join(unknown)}.")

    h = basis.h
    f = basis.growth_rate
    b1 = _p(params, "b1")
    b4_shape = (35.0 / 8.0) * _c(basis, "rsd_l4_counterterm_shape") * h
    b4_common = f**2 * _c(basis, "k_over_h_squared") * b4_shape

    zero = jnp.zeros_like(_c(basis, "rsd_l0_mm_00"))
    one = jnp.ones_like(zero)

    templates: dict[str, MultipolePrediction] = {}
    for name in requested:
        if name == "cs0":
            p0 = 2.0 * _c(basis, "rsd_l0_counterterm_shape") * h
            p2 = zero
            p4 = zero
        elif name == "cs2":
            p0 = zero
            p2 = 2.0 * _c(basis, "rsd_l2_counterterm_shape") * h
            p4 = zero
        elif name == "cs4":
            p0 = zero
            p2 = zero
            p4 = 2.0 * _c(basis, "rsd_l4_counterterm_shape") * h
        elif name == "Pshot":
            p0 = one
            p2 = zero
            p4 = zero
        else:
            p0 = b4_common * (f**2 / 9.0 + 2.0 * f * b1 / 7.0 + b1**2 / 5.0)
            p2 = b4_common * ((f**2 * 70.0 + 165.0 * f * b1 + 99.0 * b1**2) * 4.0 / 693.0)
            p4 = b4_common * ((f**2 * 210.0 + 390.0 * f * b1 + 143.0 * b1**2) * 8.0 / 5005.0)

        templates[name] = MultipolePrediction(
            k=basis.k,
            p0=p0,
            p2=p2,
            p4=p4,
            metadata={"backend": basis.metadata.get("backend", "unknown"), "z": basis.z},
        )

    return templates
