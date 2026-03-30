from __future__ import annotations

from functools import lru_cache
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gamma

from ..config import PTSettings
from ..cosmology import FFTLogInput, LinearPowerInput, prepare_fftlog_input

_EPS = 1.0e-6
_CUTOFF_OVER_H = 3.0


def _j_np(nu: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return (
        gamma(1.5 - nu)
        * gamma(1.5 - mu)
        * gamma(nu + mu - 1.5)
        / (8.0 * np.pi**1.5 * gamma(nu) * gamma(mu) * gamma(3.0 - nu - mu))
    )


def _m13_np(nu: np.ndarray) -> np.ndarray:
    return (1.0 + 9.0 * nu) * np.tan(np.pi * nu) / (
        112.0 * np.pi * (nu + 1.0) * nu * (nu - 1.0) * (nu - 2.0) * (nu - 3.0)
    )


def _ifg2_np(nu: np.ndarray) -> np.ndarray:
    return -15.0 * np.tan(np.pi * nu) / (
        28.0 * np.pi * (nu + 1.0) * nu * (nu - 1.0) * (nu - 2.0) * (nu - 3.0)
    )


def _m22_np(nu: np.ndarray, mu: np.ndarray, jmat: np.ndarray) -> np.ndarray:
    numerator = (1.5 - nu - mu) * (0.5 - nu - mu)
    poly = nu * mu * (98.0 * (nu + mu) ** 2 - 14.0 * (nu + mu) + 36.0)
    poly -= 91.0 * (nu + mu) ** 2
    poly += 3.0 * (nu + mu) + 58.0
    denominator = 196.0 * nu * (1.0 + nu) * (0.5 - nu) * mu * (1.0 + mu) * (0.5 - mu)
    return numerator * poly * jmat / denominator


def _build_bias_matrices(etam2: np.ndarray, m22basic: np.ndarray) -> dict[str, np.ndarray]:
    eta_i = etam2[:, None]
    eta_l = etam2[None, :]
    denom = (-0.5 * eta_l) * (-0.5 * eta_i)

    m_id2 = m22basic * ((3.0 + eta_i + eta_l) * (4.0 + 3.5 * (eta_i + eta_l)) / (14.0 * denom))
    m_ig2 = m22basic * (
        -(3.0 + eta_i + eta_l)
        * (1.0 + eta_i + eta_l)
        * (6.0 - 3.5 * (eta_i + eta_l))
        / (28.0 * (1.0 - 0.5 * eta_i) * (1.0 - 0.5 * eta_l) * denom)
    )
    m_id2g2 = m22basic * ((3.0 + eta_i + eta_l) / denom)
    m_ig2g2 = m22basic * (
        (3.0 + eta_i + eta_l)
        * (1.0 + eta_i + eta_l)
        / ((1.0 - 0.5 * eta_i) * (1.0 - 0.5 * eta_l) * denom)
    )
    return {
        "id2": m_id2,
        "ig2": m_ig2,
        "id2g2": m_id2g2,
        "ig2g2": m_ig2g2,
    }


@lru_cache(maxsize=16)
def _analytic_realspace_kernel_registry(
    nmax: int,
    k0_over_h: float,
    kmax_over_h: float,
    bias_matter: float,
    bias_bias: float,
) -> dict[str, np.ndarray]:
    delta = np.log(kmax_over_h / k0_over_h) / (nmax - 1.0)
    js = np.arange(nmax + 1, dtype=float) - nmax / 2.0
    etam = bias_matter + 2.0j * np.pi * js / (nmax * delta)
    etam2 = bias_bias + 2.0j * np.pi * js / (nmax * delta)
    nu = -0.5 * etam
    nu2 = -0.5 * etam2

    nu_row = nu[:, None]
    nu_col = nu[None, :]
    nu2_row = nu2[:, None]
    nu2_col = nu2[None, :]

    m22basic = _j_np(nu2_row, nu2_col)
    bias_matrices = _build_bias_matrices(etam2, m22basic)

    return {
        "delta_logk": np.asarray(delta, dtype=float),
        "etam": etam,
        "etam2": etam2,
        "m13": _m13_np(nu),
        "ifg2": _ifg2_np(nu2),
        "m22": _m22_np(nu_row, nu_col, _j_np(nu_row, nu_col)),
        "m22basic": m22basic,
        "bias_id2": bias_matrices["id2"],
        "bias_ig2": bias_matrices["ig2"],
        "bias_id2g2": bias_matrices["id2g2"],
        "bias_ig2g2": bias_matrices["ig2g2"],
    }


def _build_kdisc(settings: PTSettings, h: float) -> np.ndarray:
    nmax = settings.fftlog_n
    delta = np.log(settings.fftlog_kmax_over_h / settings.fftlog_k0_over_h) / (nmax - 1.0)
    return settings.fftlog_k0_over_h * h * np.exp(np.arange(nmax, dtype=float) * delta)


def _loglog_interpolate_jax(x: jnp.ndarray, xp: jnp.ndarray, fp: jnp.ndarray) -> jnp.ndarray:
    lx = jnp.log(x)
    lxp = jnp.log(xp)
    lfp = jnp.log(fp)
    left_slope = (lfp[1] - lfp[0]) / (lxp[1] - lxp[0])
    right_slope = (lfp[-1] - lfp[-2]) / (lxp[-1] - lxp[-2])
    ly = jnp.interp(lx, lxp, lfp)
    ly = jnp.where(lx < lxp[0], lfp[0] + left_slope * (lx - lxp[0]), ly)
    ly = jnp.where(lx > lxp[-1], lfp[-1] + right_slope * (lx - lxp[-1]), ly)
    return jnp.exp(ly)


@jax.jit
def _fftlog_coefficients_jax(
    support_k: jnp.ndarray,
    values: jnp.ndarray,
    kdisc: jnp.ndarray,
    etam: jnp.ndarray,
    bias: float,
) -> jnp.ndarray:
    nmax = kdisc.shape[0]
    sampled = _loglog_interpolate_jax(kdisc, support_k, values)
    input_real = sampled * jnp.exp(-jnp.arange(nmax, dtype=kdisc.dtype) * bias * jnp.log(kdisc[1] / kdisc[0]))
    output = jnp.fft.fft(input_real)

    indices = jnp.arange(nmax + 1)
    half = nmax // 2
    output_index = jnp.where(indices < half, half - indices, indices - half)
    base = jnp.where(indices < half, jnp.conj(output[output_index]), output[output_index])
    cmsym = (kdisc[0] ** (-etam)) * base / float(nmax)
    cmsym = cmsym.at[0].set(cmsym[0] * 0.5)
    cmsym = cmsym.at[-1].set(cmsym[-1] * 0.5)
    return cmsym


@jax.jit
def _quadratic_form_columns(x: jnp.ndarray, matrix: jnp.ndarray) -> jnp.ndarray:
    y = matrix @ x
    return jnp.sum(x * y, axis=0)


def _interpolate_to_output_jax(kdisc: jnp.ndarray, values: jnp.ndarray, output_k: jnp.ndarray) -> jnp.ndarray:
    log_kdisc = jnp.log(kdisc)
    log_output_k = jnp.log(output_k)
    slopes = jnp.empty_like(values)
    slopes = slopes.at[0].set((values[1] - values[0]) / (log_kdisc[1] - log_kdisc[0]))
    slopes = slopes.at[-1].set((values[-1] - values[-2]) / (log_kdisc[-1] - log_kdisc[-2]))
    centered = (values[2:] - values[:-2]) / (log_kdisc[2:] - log_kdisc[:-2])
    slopes = slopes.at[1:-1].set(centered)

    interval_index = jnp.clip(jnp.searchsorted(log_kdisc, log_output_k, side="right") - 1, 0, log_kdisc.size - 2)
    x0 = log_kdisc[interval_index]
    x1 = log_kdisc[interval_index + 1]
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

    left_slope = slopes[0]
    right_slope = slopes[-1]
    hermite = jnp.where(
        log_output_k < log_kdisc[0],
        values[0] + left_slope * (log_output_k - log_kdisc[0]),
        hermite,
    )
    hermite = jnp.where(
        log_output_k > log_kdisc[-1],
        values[-1] + right_slope * (log_output_k - log_kdisc[-1]),
        hermite,
    )
    return hermite


@partial(jax.jit, static_argnames=("settings",))
def compute_fftlog_realspace_terms_from_arrays(
    support_k: jnp.ndarray,
    pk_linear: jnp.ndarray,
    output_k: jnp.ndarray,
    h: float,
    settings: PTSettings,
) -> dict[str, jnp.ndarray]:
    """Compute analytic FFTLog real-space loop terms from raw arrays.

    Parameters
    ----------
    support_k
        Input support grid in ``1/Mpc``.
    pk_linear
        Linear power spectrum on ``support_k``.
    output_k
        Output grid in ``1/Mpc`` where the loop terms should be returned.
    h
        Reduced Hubble parameter.
    settings
        `PTSettings` describing the FFTLog discretization. The current code
        requires ``kernel_source="analytic"``.
    """
    if settings.kernel_source != "analytic":
        raise NotImplementedError("FFTLog real-space loops only support kernel_source='analytic'.")

    nmax = settings.fftlog_n
    kernels = _analytic_realspace_kernel_registry(
        nmax,
        settings.fftlog_k0_over_h,
        settings.fftlog_kmax_over_h,
        settings.fftlog_bias_matter + settings.fftlog_bias,
        settings.fftlog_bias_bias + settings.fftlog_bias,
    )

    delta = np.log(settings.fftlog_kmax_over_h / settings.fftlog_k0_over_h) / (nmax - 1.0)
    kdisc = settings.fftlog_k0_over_h * h * jnp.exp(jnp.arange(nmax, dtype=pk_linear.dtype) * delta)
    etam = jnp.asarray(kernels["etam"])
    etam2 = jnp.asarray(kernels["etam2"])
    cmsym = _fftlog_coefficients_jax(
        support_k,
        pk_linear,
        kdisc,
        etam,
        settings.fftlog_bias_matter + settings.fftlog_bias,
    )
    cmsym2 = _fftlog_coefficients_jax(
        support_k,
        pk_linear,
        kdisc,
        etam2,
        settings.fftlog_bias_bias + settings.fftlog_bias,
    )

    logk = jnp.log(kdisc)
    x = cmsym[:, None] * jnp.exp(etam[:, None] * logk[None, :])
    x2 = cmsym2[:, None] * jnp.exp(etam2[:, None] * logk[None, :])

    pbin = _loglog_interpolate_jax(kdisc, support_k, pk_linear)
    sigmav = jnp.trapezoid(support_k * pk_linear, jnp.log(support_k)) / (6.0 * jnp.pi**2)
    cutoff = _CUTOFF_OVER_H * h
    damping = jnp.exp(-jnp.power(kdisc / cutoff, 6.0))

    m13 = jnp.asarray(kernels["m13"])
    ifg2 = jnp.asarray(kernels["ifg2"])
    m22 = jnp.asarray(kernels["m22"])
    m22basic = jnp.asarray(kernels["m22basic"])
    bias_id2 = jnp.asarray(kernels["bias_id2"])
    bias_ig2 = jnp.asarray(kernels["bias_ig2"])
    bias_id2g2 = jnp.asarray(kernels["bias_id2g2"])
    bias_ig2g2 = jnp.asarray(kernels["bias_ig2g2"])

    f13 = jnp.einsum("nk,n->k", x, m13)
    p13_uv = -61.0 * pbin * kdisc**2 * sigmav / 105.0
    p13 = (jnp.real(kdisc**3 * f13 * pbin) + p13_uv) * damping
    p22 = jnp.real(kdisc**3 * _quadratic_form_columns(x, m22)) * damping

    raw_id2d2 = jnp.real(kdisc**3 * (2.0 * _quadratic_form_columns(x2, m22basic)))
    p_id2d2 = jnp.abs(raw_id2d2 - raw_id2d2[0] + _EPS)
    p_id2 = jnp.real(kdisc**3 * _quadratic_form_columns(x2, bias_id2))
    p_ig2 = jnp.abs(jnp.real(kdisc**3 * _quadratic_form_columns(x2, bias_ig2)))
    p_id2g2 = jnp.abs(jnp.real(kdisc**3 * _quadratic_form_columns(x2, bias_id2g2)))
    p_ig2g2 = jnp.abs(jnp.real(kdisc**3 * _quadratic_form_columns(x2, bias_ig2g2)))
    p_ifg2 = jnp.abs(jnp.real(kdisc**3 * jnp.einsum("nk,n->k", x2, ifg2) * pbin))

    return {
        "real_loop_matter": _interpolate_to_output_jax(kdisc, p13 + p22, output_k),
        "real_loop_b2_b2": _interpolate_to_output_jax(kdisc, -p_id2d2, output_k),
        "real_cross_b1_b2": _interpolate_to_output_jax(kdisc, p_id2, output_k),
        "real_cross_b1_bG2": _interpolate_to_output_jax(kdisc, -p_ig2, output_k),
        "real_loop_b2_bG2": _interpolate_to_output_jax(kdisc, -p_id2g2, output_k),
        "real_loop_bG2_bG2": _interpolate_to_output_jax(kdisc, p_ig2g2, output_k),
        "real_gamma3": _interpolate_to_output_jax(kdisc, -p_ifg2, output_k),
    }


def compute_fftlog_realspace_terms_from_preprocessed(
    fftlog_input: FFTLogInput,
    settings: PTSettings,
    output_k: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Compute analytic FFTLog real-space loop terms from `FFTLogInput`.

    Parameters
    ----------
    fftlog_input
        Linear spectra already aligned to the internal FFTLog support grid.
    settings
        `PTSettings` describing the FFTLog discretization. The current code
        requires ``kernel_source="analytic"``.
    output_k
        Optional output grid in ``1/Mpc``. If omitted, the terms are returned
        on the FFTLog support grid.
    """
    if settings.kernel_source != "analytic":
        raise NotImplementedError("FFTLog real-space loops only support kernel_source='analytic'.")

    nmax = settings.fftlog_n
    kernels = _analytic_realspace_kernel_registry(
        nmax,
        settings.fftlog_k0_over_h,
        settings.fftlog_kmax_over_h,
        settings.fftlog_bias_matter + settings.fftlog_bias,
        settings.fftlog_bias_bias + settings.fftlog_bias,
    )

    kdisc = jnp.asarray(np.asarray(fftlog_input.kdisc, dtype=float))
    pdisc = jnp.asarray(np.asarray(fftlog_input.pdisc, dtype=float))
    if output_k is None:
        output_k = kdisc

    etam = jnp.asarray(kernels["etam"])
    etam2 = jnp.asarray(kernels["etam2"])
    cmsym = _fftlog_coefficients_jax(kdisc, pdisc, kdisc, etam, settings.fftlog_bias_matter + settings.fftlog_bias)
    cmsym2 = _fftlog_coefficients_jax(kdisc, pdisc, kdisc, etam2, settings.fftlog_bias_bias + settings.fftlog_bias)

    logk = jnp.log(kdisc)
    x = cmsym[:, None] * jnp.exp(etam[:, None] * logk[None, :])
    x2 = cmsym2[:, None] * jnp.exp(etam2[:, None] * logk[None, :])

    h = float(fftlog_input.h)
    sigmav = jnp.trapezoid(kdisc * pdisc, jnp.log(kdisc)) / (6.0 * jnp.pi**2)
    cutoff = _CUTOFF_OVER_H * h
    damping = jnp.exp(-jnp.power(kdisc / cutoff, 6.0))

    m13 = jnp.asarray(kernels["m13"])
    ifg2 = jnp.asarray(kernels["ifg2"])
    m22 = jnp.asarray(kernels["m22"])
    m22basic = jnp.asarray(kernels["m22basic"])
    bias_id2 = jnp.asarray(kernels["bias_id2"])
    bias_ig2 = jnp.asarray(kernels["bias_ig2"])
    bias_id2g2 = jnp.asarray(kernels["bias_id2g2"])
    bias_ig2g2 = jnp.asarray(kernels["bias_ig2g2"])

    f13 = jnp.einsum("nk,n->k", x, m13)
    p13_uv = -61.0 * pdisc * kdisc**2 * sigmav / 105.0
    p13 = (jnp.real(kdisc**3 * f13 * pdisc) + p13_uv) * damping
    p22 = jnp.real(kdisc**3 * _quadratic_form_columns(x, m22)) * damping

    raw_id2d2 = jnp.real(kdisc**3 * (2.0 * _quadratic_form_columns(x2, m22basic)))
    p_id2d2 = jnp.abs(raw_id2d2 - raw_id2d2[0] + _EPS)
    p_id2 = jnp.real(kdisc**3 * _quadratic_form_columns(x2, bias_id2))
    p_ig2 = jnp.abs(jnp.real(kdisc**3 * _quadratic_form_columns(x2, bias_ig2)))
    p_id2g2 = jnp.abs(jnp.real(kdisc**3 * _quadratic_form_columns(x2, bias_id2g2)))
    p_ig2g2 = jnp.abs(jnp.real(kdisc**3 * _quadratic_form_columns(x2, bias_ig2g2)))
    p_ifg2 = jnp.abs(jnp.real(kdisc**3 * jnp.einsum("nk,n->k", x2, ifg2) * pdisc))

    return {
        "real_loop_matter": _interpolate_to_output_jax(kdisc, p13 + p22, output_k),
        "real_loop_b2_b2": _interpolate_to_output_jax(kdisc, -p_id2d2, output_k),
        "real_cross_b1_b2": _interpolate_to_output_jax(kdisc, p_id2, output_k),
        "real_cross_b1_bG2": _interpolate_to_output_jax(kdisc, -p_ig2, output_k),
        "real_loop_b2_bG2": _interpolate_to_output_jax(kdisc, -p_id2g2, output_k),
        "real_loop_bG2_bG2": _interpolate_to_output_jax(kdisc, p_ig2g2, output_k),
        "real_gamma3": _interpolate_to_output_jax(kdisc, -p_ifg2, output_k),
    }


def compute_fftlog_realspace_terms(
    linear_input: LinearPowerInput,
    settings: PTSettings,
    output_k: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Compute analytic FFTLog real-space loop terms from `LinearPowerInput`.

    Parameters
    ----------
    linear_input
        Linear-theory input sampled on an arbitrary support grid.
    settings
        `PTSettings` describing the FFTLog discretization. The current code
        requires ``kernel_source="analytic"``.
    output_k
        Optional output grid in ``1/Mpc``. If omitted, the original support
        grid from ``linear_input`` is used.
    """
    fftlog_input = prepare_fftlog_input(linear_input, settings)
    if output_k is None:
        output_k = jnp.asarray(np.asarray(linear_input.k, dtype=float))
    return compute_fftlog_realspace_terms_from_preprocessed(fftlog_input, settings, output_k=output_k)
