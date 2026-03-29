from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import PTSettings


@dataclass(frozen=True, slots=True)
class LinearPowerInput:
    k: np.ndarray
    pk_linear: np.ndarray
    z: float
    growth_factor: float
    growth_rate: float
    h: float
    transfer_linear: np.ndarray | None = None
    pk_nowiggle: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        k = np.asarray(self.k, dtype=float)
        pk_linear = np.asarray(self.pk_linear, dtype=float)

        if k.ndim != 1:
            raise ValueError("LinearPowerInput.k must be a one-dimensional array.")
        if pk_linear.shape != k.shape:
            raise ValueError("LinearPowerInput.pk_linear must have the same shape as k.")
        if k.size == 0:
            raise ValueError("LinearPowerInput.k must not be empty.")
        if not np.all(np.isfinite(k)) or not np.all(np.isfinite(pk_linear)):
            raise ValueError("LinearPowerInput arrays must contain only finite values.")
        if np.any(np.diff(k) <= 0.0):
            raise ValueError("LinearPowerInput.k must be strictly increasing.")
        if self.h <= 0.0:
            raise ValueError("LinearPowerInput.h must be positive.")
        if not np.isfinite(self.growth_rate):
            raise ValueError("LinearPowerInput.growth_rate must be finite.")
        if self.metadata:
            if "field" in self.metadata and self.metadata["field"] not in {"cb", "matter"}:
                raise ValueError("LinearPowerInput.metadata['field'] must be either 'cb' or 'matter'.")
            if "k_units" in self.metadata and self.metadata["k_units"] not in {"1/Mpc", "h/Mpc"}:
                raise ValueError("LinearPowerInput.metadata['k_units'] must be '1/Mpc' or 'h/Mpc'.")

        object.__setattr__(self, "k", k)
        object.__setattr__(self, "pk_linear", pk_linear)

        if self.transfer_linear is not None:
            transfer_linear = np.asarray(self.transfer_linear, dtype=float)
            if transfer_linear.shape != k.shape:
                raise ValueError("LinearPowerInput.transfer_linear must have the same shape as k.")
            if not np.all(np.isfinite(transfer_linear)):
                raise ValueError("LinearPowerInput.transfer_linear must contain only finite values.")
            object.__setattr__(self, "transfer_linear", transfer_linear)

        if self.pk_nowiggle is not None:
            pk_nowiggle = np.asarray(self.pk_nowiggle, dtype=float)
            if pk_nowiggle.shape != k.shape:
                raise ValueError("LinearPowerInput.pk_nowiggle must have the same shape as k.")
            if not np.all(np.isfinite(pk_nowiggle)):
                raise ValueError("LinearPowerInput.pk_nowiggle must contain only finite values.")
            object.__setattr__(self, "pk_nowiggle", pk_nowiggle)


@dataclass(frozen=True, slots=True)
class NativeFFTLogInput:
    kdisc: np.ndarray
    pdisc: np.ndarray
    tdisc: np.ndarray
    z: float
    growth_factor: float
    growth_rate: float
    h: float
    pnw: np.ndarray | None = None
    tnw: np.ndarray | None = None
    pw: np.ndarray | None = None
    tw: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kdisc = np.asarray(self.kdisc, dtype=float)
        pdisc = np.asarray(self.pdisc, dtype=float)
        tdisc = np.asarray(self.tdisc, dtype=float)
        if kdisc.ndim != 1:
            raise ValueError("NativeFFTLogInput.kdisc must be one-dimensional.")
        if pdisc.shape != kdisc.shape or tdisc.shape != kdisc.shape:
            raise ValueError("NativeFFTLogInput spectra must have the same shape as kdisc.")
        if np.any(np.diff(kdisc) <= 0.0):
            raise ValueError("NativeFFTLogInput.kdisc must be strictly increasing.")
        object.__setattr__(self, "kdisc", kdisc)
        object.__setattr__(self, "pdisc", pdisc)
        object.__setattr__(self, "tdisc", tdisc)

        for name in ("pnw", "tnw", "pw", "tw"):
            values = getattr(self, name)
            if values is None:
                continue
            array = np.asarray(values, dtype=float)
            if array.shape != kdisc.shape:
                raise ValueError(f"NativeFFTLogInput.{name} must have the same shape as kdisc.")
            object.__setattr__(self, name, array)


def _primordial_power_spectrum_from_classy_params(k: np.ndarray, params: dict[str, Any]) -> np.ndarray | None:
    n_s = float(params.get("n_s", 1.0))
    alpha_s = float(params.get("alpha_s", 0.0))
    k_pivot = float(params.get("k_pivot", 0.05))
    log_ratio = np.log(np.asarray(k, dtype=float) / k_pivot)
    primordial_over_As = np.exp((n_s - 1.0) * log_ratio + 0.5 * alpha_s * log_ratio * log_ratio)
    return 2.0 * np.pi**2 * np.asarray(k, dtype=float) * primordial_over_As


def _build_linear_input_from_classy(
    cosmo: Any,
    z: float,
    k: np.ndarray,
    *,
    pk_linear: np.ndarray,
    linear_pk_source: str,
    transfer_linear: np.ndarray | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> LinearPowerInput:
    k = np.asarray(k, dtype=float)
    pk_linear = np.asarray(pk_linear, dtype=float)
    growth_rate = float(cosmo.scale_independent_growth_factor_f(float(z)))
    if transfer_linear is None:
        primordial_pk = _primordial_power_spectrum_from_classy_params(k, getattr(cosmo, "pars", {}))
        transfer_linear = None if primordial_pk is None else (5.0 / 3.0) * np.sqrt(pk_linear / primordial_pk)
    else:
        transfer_linear = np.asarray(transfer_linear, dtype=float)

    if hasattr(cosmo, "scale_independent_growth_factor"):
        growth_factor = float(cosmo.scale_independent_growth_factor(float(z)))
    else:
        growth_factor = np.nan

    metadata = {
        "source": "classy",
        "field": "cb",
        "k_units": "1/Mpc",
        "pk_units": "Mpc^3",
        "linear_pk_source": linear_pk_source,
        "_classpt_cosmo": cosmo,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return LinearPowerInput(
        k=k,
        pk_linear=pk_linear,
        transfer_linear=transfer_linear,
        z=float(z),
        growth_factor=growth_factor,
        growth_rate=growth_rate,
        h=float(cosmo.h()),
        metadata=metadata,
    )


def build_linear_input_from_classy(cosmo: Any, z: float, k: np.ndarray) -> LinearPowerInput:
    """Build a linear-input container from a `classy.Class` instance."""
    k = np.asarray(k, dtype=float)
    pk_linear = np.asarray([cosmo.pk_lin(float(ki), float(z)) for ki in k], dtype=float)
    return _build_linear_input_from_classy(
        cosmo,
        z,
        k,
        pk_linear=pk_linear,
        linear_pk_source="classpt_pk_lin",
    )


def build_linear_input_from_cosmoprimo(cosmo: Any, z: float, k: np.ndarray) -> LinearPowerInput:
    """Build a linear-input container from a `cosmoprimo.Cosmology` instance."""
    engine = getattr(cosmo, "engine", None)
    if engine is None:
        raise ValueError("build_linear_input_from_cosmoprimo requires a cosmoprimo cosmology with an attached engine.")

    k = np.asarray(k, dtype=float)
    h = float(cosmo["h"])
    fourier = cosmo.get_fourier()
    background = cosmo.get_background()
    pk_interpolator = fourier.pk_interpolator(non_linear=False, of="delta_cb").to_1d(z=float(z))
    pk_linear = np.asarray(pk_interpolator(k / h), dtype=float) / h**3

    primordial_pk = _primordial_power_spectrum_from_classy_params(k, cosmo.get_params())
    transfer_linear = None if primordial_pk is None else (5.0 / 3.0) * np.sqrt(pk_linear / primordial_pk)
    return LinearPowerInput(
        k=k,
        pk_linear=pk_linear,
        transfer_linear=transfer_linear,
        z=float(z),
        growth_factor=float(background.growth_factor(float(z))),
        growth_rate=float(background.growth_rate(float(z))),
        h=h,
        metadata={
            "source": "cosmoprimo",
            "engine": getattr(engine, "name", engine.__class__.__name__),
            "field": "cb",
            "k_units": "1/Mpc",
            "pk_units": "Mpc^3",
            "linear_pk_source": "fourier_delta_cb",
        },
    )


def build_classpt_parity_linear_input_from_classy(cosmo: Any, z: float, k: np.ndarray) -> LinearPowerInput:
    """Build a parity-only linear input using the CLASS-PT internal tree basis term."""
    k = np.asarray(k, dtype=float)
    pk_linear = np.asarray([np.asarray(cosmo.pk(float(ki), float(z)), dtype=float)[14] for ki in k], dtype=float)
    return _build_linear_input_from_classy(
        cosmo,
        z,
        k,
        pk_linear=pk_linear,
        linear_pk_source="classpt_internal_tree",
    )


def _native_fftlog_support_k(settings: PTSettings, h: float) -> np.ndarray:
    delta = np.log(settings.fftlog_kmax_over_h / settings.fftlog_k0_over_h) / (settings.fftlog_n - 1.0)
    return settings.fftlog_k0_over_h * h * np.exp(np.arange(settings.fftlog_n, dtype=float) * delta)


def _loglog_interp_numpy(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    lx = np.log(np.asarray(x, dtype=float))
    lxp = np.log(np.asarray(xp, dtype=float))
    lfp = np.log(np.asarray(fp, dtype=float))
    left_slope = (lfp[1] - lfp[0]) / (lxp[1] - lxp[0])
    right_slope = (lfp[-1] - lfp[-2]) / (lxp[-1] - lxp[-2])
    ly = np.interp(lx, lxp, lfp)
    ly = np.where(lx < lxp[0], lfp[0] + left_slope * (lx - lxp[0]), ly)
    ly = np.where(lx > lxp[-1], lfp[-1] + right_slope * (lx - lxp[-1]), ly)
    return np.exp(ly)


def _classy_phi_transfer(cosmo: Any, z: float, k: np.ndarray, reference_transfer: np.ndarray | None) -> np.ndarray | None:
    if not hasattr(cosmo, "get_transfer"):
        return reference_transfer

    transfer = cosmo.get_transfer(float(z))
    if not transfer or "k (h/Mpc)" not in transfer or "phi" not in transfer:
        return reference_transfer

    source_k = np.asarray(transfer["k (h/Mpc)"], dtype=float) * float(cosmo.h())
    phi = np.asarray(transfer["phi"], dtype=float)
    if source_k.ndim != 1 or phi.shape != source_k.shape or source_k.size < 2:
        return reference_transfer

    interpolated_phi = np.interp(np.log(np.asarray(k, dtype=float)), np.log(source_k), phi)
    if reference_transfer is None:
        return interpolated_phi

    norm = np.dot(interpolated_phi, interpolated_phi)
    if norm == 0.0:
        return reference_transfer

    scale = np.dot(np.asarray(reference_transfer, dtype=float), interpolated_phi) / norm
    return scale * interpolated_phi


def build_classpt_native_grid_parity_linear_input_from_classy(
    cosmo: Any,
    z: float,
    settings: PTSettings | None = None,
) -> LinearPowerInput:
    """Build a parity-only linear input on the native FFTLog support grid."""
    settings = PTSettings() if settings is None else settings
    k = _native_fftlog_support_k(settings, float(cosmo.h()))
    pk_linear = np.asarray([np.asarray(cosmo.pk(float(ki), float(z)), dtype=float)[14] for ki in k], dtype=float)
    primordial_pk = _primordial_power_spectrum_from_classy_params(k, getattr(cosmo, "pars", {}))
    derived_transfer = None if primordial_pk is None else (5.0 / 3.0) * np.sqrt(pk_linear / primordial_pk)
    transfer_linear = _classy_phi_transfer(cosmo, z, k, derived_transfer)
    transfer_source = "classy_phi_scaled" if transfer_linear is not derived_transfer else "pk_over_primordial"
    return _build_linear_input_from_classy(
        cosmo,
        z,
        k,
        pk_linear=pk_linear,
        linear_pk_source="classpt_internal_tree",
        transfer_linear=transfer_linear,
        extra_metadata={
            "support_grid": "native_fftlog_kdisc",
            "transfer_source": transfer_source,
        },
    )


def prepare_native_fftlog_input(
    linear_input: LinearPowerInput,
    settings: PTSettings,
) -> NativeFFTLogInput:
    """Preprocess linear inputs onto the native FFTLog grid."""
    kdisc = _native_fftlog_support_k(settings, float(linear_input.h))
    support_k = np.asarray(linear_input.k, dtype=float)

    aligned_support = support_k.shape == kdisc.shape and np.allclose(support_k, kdisc, rtol=0.0, atol=0.0)
    if aligned_support:
        pdisc = np.asarray(linear_input.pk_linear, dtype=float)
        if linear_input.transfer_linear is None:
            tdisc = pdisc
        else:
            tdisc = np.asarray(linear_input.transfer_linear, dtype=float)
        pnw = None if linear_input.pk_nowiggle is None else np.asarray(linear_input.pk_nowiggle, dtype=float)
    else:
        pdisc = _loglog_interp_numpy(kdisc, support_k, np.asarray(linear_input.pk_linear, dtype=float))
        source_transfer = (
            np.asarray(linear_input.transfer_linear, dtype=float)
            if linear_input.transfer_linear is not None
            else np.asarray(linear_input.pk_linear, dtype=float)
        )
        tdisc = _loglog_interp_numpy(kdisc, support_k, source_transfer)
        pnw = None
        if linear_input.pk_nowiggle is not None:
            pnw = _loglog_interp_numpy(kdisc, support_k, np.asarray(linear_input.pk_nowiggle, dtype=float))

    if pnw is None:
        pnw = pdisc
    tnw = tdisc
    pw = pdisc - pnw
    tw = tdisc - tnw

    metadata = dict(linear_input.metadata)
    metadata.update(
        {
            "fftlog_grid_source": "native_kdisc",
            "fftlog_input_aligned": aligned_support,
            "fftlog_input_mode": "no_ir" if not settings.ir_resummation else "ir_requested",
        }
    )
    return NativeFFTLogInput(
        kdisc=kdisc,
        pdisc=pdisc,
        tdisc=tdisc,
        pnw=pnw,
        tnw=tnw,
        pw=pw,
        tw=tw,
        z=linear_input.z,
        growth_factor=linear_input.growth_factor,
        growth_rate=linear_input.growth_rate,
        h=linear_input.h,
        metadata=metadata,
    )
