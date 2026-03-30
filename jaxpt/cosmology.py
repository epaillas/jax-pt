from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import PTSettings


@dataclass(frozen=True, slots=True)
class LinearPowerInput:
    """Linear-theory input sampled on a single support grid.

    Parameters
    ----------
    k
        One-dimensional support grid in ``1/Mpc``.
    pk_linear
        Linear power spectrum on ``k``, typically in ``Mpc^3``.
    z
        Redshift at which the input is defined.
    growth_factor
        Linear growth factor at ``z``.
    growth_rate
        Linear growth rate ``f(z)``.
    h
        Reduced Hubble parameter.
    transfer_linear
        Optional linear transfer-like quantity on the same grid. In CLASS-backed
        parity workflows this stores the transfer used by the RSD FFTLog terms.
    pk_nowiggle
        Optional smooth linear spectrum on the same grid for future
        IR-resummation support.
    metadata
        Free-form provenance metadata. Common entries include ``source``,
        ``field``, ``k_units``, and ``pk_units``.
    """

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
class FFTLogInput:
    """Linear input resampled to the internal FFTLog support grid.

    Parameters
    ----------
    kdisc
        Internal FFTLog support grid in ``1/Mpc``.
    pdisc
        Linear power spectrum sampled on ``kdisc``.
    tdisc
        Transfer-like quantity sampled on ``kdisc``.
    z
        Redshift at which the spectra are defined.
    growth_factor
        Linear growth factor at ``z``.
    growth_rate
        Linear growth rate ``f(z)``.
    h
        Reduced Hubble parameter.
    pnw, tnw
        Optional nowiggle power and transfer spectra on ``kdisc``.
    pw, tw
        Optional wiggle residuals, usually ``pdisc - pnw`` and ``tdisc - tnw``.
    metadata
        Free-form provenance metadata for the preprocessing step.
    """

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
            raise ValueError("FFTLogInput.kdisc must be one-dimensional.")
        if pdisc.shape != kdisc.shape or tdisc.shape != kdisc.shape:
            raise ValueError("FFTLogInput spectra must have the same shape as kdisc.")
        if np.any(np.diff(kdisc) <= 0.0):
            raise ValueError("FFTLogInput.kdisc must be strictly increasing.")
        object.__setattr__(self, "kdisc", kdisc)
        object.__setattr__(self, "pdisc", pdisc)
        object.__setattr__(self, "tdisc", tdisc)

        for name in ("pnw", "tnw", "pw", "tw"):
            values = getattr(self, name)
            if values is None:
                continue
            array = np.asarray(values, dtype=float)
            if array.shape != kdisc.shape:
                raise ValueError(f"FFTLogInput.{name} must have the same shape as kdisc.")
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
    """Build `LinearPowerInput` from a live `classy.Class` cosmology.

    Parameters
    ----------
    cosmo
        Live `classy.Class` instance with a computed linear spectrum.
    z
        Redshift at which to sample the linear spectrum.
    k
        One-dimensional support grid in ``1/Mpc``.
    """
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
    """Build `LinearPowerInput` from a live `cosmoprimo.Cosmology`.

    Parameters
    ----------
    cosmo
        `cosmoprimo.Cosmology` instance with an attached engine.
    z
        Redshift at which to sample the linear spectrum.
    k
        One-dimensional support grid in ``1/Mpc``.
    """
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
    """Build parity-oriented linear input from the CLASS-PT internal tree term.

    Parameters
    ----------
    cosmo
        Live `classy.Class` PT cosmology.
    z
        Redshift at which to sample the internal tree term.
    k
        One-dimensional support grid in ``1/Mpc``.
    """
    k = np.asarray(k, dtype=float)
    pk_linear = np.asarray([np.asarray(cosmo.pk(float(ki), float(z)), dtype=float)[14] for ki in k], dtype=float)
    return _build_linear_input_from_classy(
        cosmo,
        z,
        k,
        pk_linear=pk_linear,
        linear_pk_source="classpt_internal_tree",
    )


def _fftlog_support_k(settings: PTSettings, h: float) -> np.ndarray:
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


def build_classpt_fftlog_grid_parity_linear_input_from_classy(
    cosmo: Any,
    z: float,
    settings: PTSettings | None = None,
) -> LinearPowerInput:
    """Build CLASS-PT parity input directly on the internal FFTLog grid.

    Parameters
    ----------
    cosmo
        Live `classy.Class` PT cosmology.
    z
        Redshift at which to sample the internal tree term.
    settings
        Optional `PTSettings` controlling the FFTLog support grid through
        ``fftlog_n``, ``fftlog_k0_over_h``, and ``fftlog_kmax_over_h``.
    """
    settings = PTSettings() if settings is None else settings
    k = _fftlog_support_k(settings, float(cosmo.h()))
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
            "support_grid": "fftlog_kdisc",
            "transfer_source": transfer_source,
        },
    )


def prepare_fftlog_input(
    linear_input: LinearPowerInput,
    settings: PTSettings,
) -> FFTLogInput:
    """Preprocess linear input onto the internal FFTLog grid.

    Parameters
    ----------
    linear_input
        Linear spectrum sampled on an arbitrary support grid.
    settings
        `PTSettings` that define the FFTLog support grid and IR-resummation
        request flags.
    """
    kdisc = _fftlog_support_k(settings, float(linear_input.h))
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
            "fftlog_grid_source": "fftlog_kdisc",
            "fftlog_input_aligned": aligned_support,
            "fftlog_input_mode": "no_ir" if not settings.ir_resummation else "ir_requested",
        }
    )
    return FFTLogInput(
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


_CLASSY_FIXED_PARAM_NAMES = frozenset(
    {
        "z_pk",
        "output",
        "non linear",
        "IR resummation",
        "Bias tracers",
        "cb",
        "RSD",
    }
)
_COSMOPRIMO_FIXED_PARAM_NAMES = frozenset({"z_pk", "kmax_pk", "ellmax_cl", "non_linear", "modes", "lensing"})
_COSMOLOGY_ALIAS_GROUPS = (
    ("logA", "A_s", "ln10^10A_s"),
    ("h", "H0"),
)


def _a_s_to_log_a(value: Any) -> float:
    return float(np.log(1.0e10 * float(value)))


def _log_a_to_a_s(value: Any) -> float:
    return float(np.exp(float(value)) / 1.0e10)


def _to_public_cosmology_value(canonical: str, name: str, value: Any) -> Any:
    if canonical == "logA":
        if name == "A_s":
            return _a_s_to_log_a(value)
        return float(value)
    if canonical == "h" and name == "H0":
        return float(value) / 100.0
    return value


def _convert_public_cosmology_params_to_backend(params: Mapping[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for name, value in params.items():
        if name == "logA":
            converted["A_s"] = _log_a_to_a_s(value)
        else:
            converted[str(name)] = value
    return converted


def _convert_backend_cosmology_params_to_public(params: Mapping[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for name, value in params.items():
        if name == "A_s":
            converted["logA"] = _a_s_to_log_a(value)
        elif name == "H0":
            converted["h"] = float(value) / 100.0
        else:
            converted[str(name)] = value
    return converted


def _freeze_cache_value(value: Any) -> Any:
    if isinstance(value, (str, bytes, bool)):
        return value
    array = np.asarray(value)
    if array.dtype.kind in {"U", "S", "O"}:
        if array.ndim == 0:
            return array.item()
        return tuple(array.reshape(-1).tolist())
    if array.ndim == 0:
        return float(array)
    return tuple(float(item) for item in array.reshape(-1))


def _cache_key_from_params(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple((name, _freeze_cache_value(value)) for name, value in sorted(params.items()))


def _split_fixed_and_query_params(
    params: dict[str, Any],
    *,
    fixed_names: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixed, query = {}, {}
    for name, value in params.items():
        if name in fixed_names:
            fixed[name] = value
        else:
            query[name] = value
    return fixed, query


def _normalize_cosmology_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(overrides)
    for canonical, *aliases in _COSMOLOGY_ALIAS_GROUPS:
        values = [(name, normalized[name]) for name in (canonical, *aliases) if name in normalized]
        if len(values) > 1:
            first_value = _to_public_cosmology_value(canonical, values[0][0], values[0][1])
            for name, value in values[1:]:
                public_value = _to_public_cosmology_value(canonical, name, value)
                if not np.allclose(np.asarray(public_value), np.asarray(first_value), rtol=1.0e-12, atol=1.0e-12):
                    names = ", ".join(name for name, _ in values)
                    raise ValueError(f"Conflicting cosmology aliases provided for {canonical}: {names}.")
        if not values:
            continue
        name, value = values[-1]
        for name, _ in values:
            del normalized[name]
        normalized[canonical] = _to_public_cosmology_value(canonical, name, value)
    return normalized


def _extract_classy_fiducial_params(cosmo: Any) -> dict[str, Any]:
    return _convert_backend_cosmology_params_to_public(dict(getattr(cosmo, "pars", {})))


def _extract_cosmoprimo_fiducial_params(cosmo: Any) -> dict[str, Any]:
    params = {
        "h": float(cosmo["h"]),
        "n_s": float(cosmo["n_s"]),
        "logA": _a_s_to_log_a(float(cosmo["A_s"])),
    }
    for name in ["omega_b", "omega_cdm", "tau_reio", "YHe", "N_ur", "Omega_k", "w0_fld", "wa_fld"]:
        try:
            value = cosmo[name]
        except Exception:
            continue
        params[name] = np.asarray(value).item() if np.asarray(value).ndim == 0 else np.asarray(value)
    try:
        value = cosmo["m_ncdm"]
    except Exception:
        value = None
    if value is not None:
        array = np.asarray(value)
        params["m_ncdm"] = array.item() if array.ndim == 0 or array.size == 1 else array
    engine_name = getattr(getattr(cosmo, "engine", None), "name", None)
    full_params = dict(cosmo.get_params())
    for name in _COSMOPRIMO_FIXED_PARAM_NAMES:
        if name in full_params:
            params[name] = full_params[name]
    if engine_name is not None:
        params["engine"] = engine_name
    return params


def _default_classy_engine_settings(
    *,
    z: float,
    settings: PTSettings,
    input_recipe: str | None,
) -> dict[str, Any]:
    recipe = "linear_pk" if input_recipe is None else input_recipe
    params: dict[str, Any] = {"z_pk": float(z)}
    if recipe == "linear_pk":
        params["output"] = "mPk"
        return params
    if recipe in {"classpt_parity", "classpt_fftlog_grid_parity"}:
        params.update(
            {
                "output": "mTk,mPk",
                "non linear": "PT",
                "IR resummation": "Yes" if settings.ir_resummation else "No",
                "Bias tracers": "Yes",
                "cb": "Yes" if settings.cb else "No",
                "RSD": "Yes" if settings.rsd else "No",
            }
        )
        return params
    raise ValueError(
        "Unsupported CLASS query recipe. Expected one of {'linear_pk', 'classpt_parity', 'classpt_fftlog_grid_parity'}."
    )


@dataclass(frozen=True, slots=True)
class ResolvedCosmologyState:
    """Concrete cosmology/template resolution result.

    Attributes
    ----------
    cosmology
        Resolved live cosmology object, or ``None`` for fixed linear-input
        templates.
    linear_input
        Linear-theory input derived from the cosmology state.
    cosmology_params
        Full cosmology parameter mapping used to build ``cosmology``.
    query_key
        Immutable cache key representing the resolved cosmology state.
    """

    cosmology: Any
    linear_input: LinearPowerInput
    cosmology_params: dict[str, Any]
    query_key: tuple[tuple[str, Any], ...]


class BaseCosmologyProvider:
    """Abstract provider that turns cosmology parameters into linear input.

    Parameters
    ----------
    fiducial_params
        Mapping of queryable cosmology parameters and their fiducial values.
    """

    name = "base"

    def __init__(self, fiducial_params: dict[str, Any]) -> None:
        self.fiducial_params = dict(fiducial_params)
        self.query_param_names = set(self.fiducial_params)

    def accepts_param(self, name: str) -> bool:
        """Return whether the provider accepts the supplied cosmology name or alias."""
        return name in self.query_param_names or any(name in aliases for aliases in _COSMOLOGY_ALIAS_GROUPS)

    def resolve(self, *, overrides: dict[str, Any], z: float, k: np.ndarray | None, settings: PTSettings, input_recipe: str | None) -> ResolvedCosmologyState:
        """Resolve cosmology overrides into a live cosmology and linear input.

        Parameters
        ----------
        overrides
            Query-time cosmology overrides.
        z
            Redshift at which the linear input should be built.
        k
            Optional support grid in ``1/Mpc``. Some recipes ignore this and
            derive the grid internally.
        settings
            `PTSettings` controlling backend- and FFTLog-related options.
        input_recipe
            Optional recipe string. Supported values depend on the concrete
            provider and are documented on `PowerSpectrumTemplate`.
        """
        normalized = _normalize_cosmology_overrides(overrides)
        invalid = sorted(name for name in normalized if name not in self.query_param_names)
        if invalid:
            raise ValueError(f"Unexpected cosmology parameters: {', '.join(invalid)}")
        cosmology_params = dict(self.fiducial_params)
        cosmology_params.update(normalized)
        cosmology = self.build_cosmology(_convert_public_cosmology_params_to_backend(cosmology_params))
        linear_input = self.build_linear_input(cosmology=cosmology, z=z, k=k, settings=settings, input_recipe=input_recipe)
        return ResolvedCosmologyState(
            cosmology=cosmology,
            linear_input=linear_input,
            cosmology_params=cosmology_params,
            query_key=_cache_key_from_params(cosmology_params),
        )

    def build_cosmology(self, cosmology_params: dict[str, Any]) -> Any:
        """Construct a live cosmology object from the supplied parameter mapping."""
        raise NotImplementedError

    def build_linear_input(self, *, cosmology: Any, z: float, k: np.ndarray | None, settings: PTSettings, input_recipe: str | None) -> LinearPowerInput:
        """Build `LinearPowerInput` for a resolved cosmology and recipe."""
        raise NotImplementedError


class ClassyCosmologyProvider(BaseCosmologyProvider):
    """Cosmology provider backed by live `classy.Class` instances.

    Parameters
    ----------
    fiducial_params
        Queryable cosmology parameters and fiducial values.
    fixed_params
        Engine/configuration parameters held fixed for every query, such as
        ``z_pk`` or CLASS-PT control flags.
    """

    name = "classy"

    def __init__(self, fiducial_params: dict[str, Any], *, fixed_params: dict[str, Any]) -> None:
        super().__init__(fiducial_params)
        self.fixed_params = dict(fixed_params)
        self.query_param_names = set(fiducial_params)

    @classmethod
    def from_cosmology(cls, cosmo: Any) -> "ClassyCosmologyProvider":
        """Build a provider from an existing `classy.Class` cosmology."""
        params = _extract_classy_fiducial_params(cosmo)
        fixed_params, fiducial_params = _split_fixed_and_query_params(params, fixed_names=set(_CLASSY_FIXED_PARAM_NAMES))
        return cls(fiducial_params=fiducial_params, fixed_params=fixed_params)

    @classmethod
    def from_mapping(
        cls,
        params: dict[str, Any],
        *,
        z: float,
        settings: PTSettings,
        input_recipe: str | None,
    ) -> "ClassyCosmologyProvider":
        """Build a provider from a fiducial cosmology parameter mapping."""
        fixed_params = _default_classy_engine_settings(z=z, settings=settings, input_recipe=input_recipe)
        query_params = dict(params)
        for name in list(query_params):
            if name in _CLASSY_FIXED_PARAM_NAMES or name in fixed_params:
                fixed_params[name] = query_params.pop(name)
        return cls(fiducial_params=query_params, fixed_params=fixed_params)

    def build_cosmology(self, cosmology_params: dict[str, Any]) -> Any:
        """Instantiate and compute a new `classy.Class` cosmology."""
        from classy import Class

        cosmo = Class()
        cosmo.set({**self.fixed_params, **cosmology_params})
        cosmo.compute()
        return cosmo

    def build_linear_input(self, *, cosmology: Any, z: float, k: np.ndarray | None, settings: PTSettings, input_recipe: str | None) -> LinearPowerInput:
        """Build `LinearPowerInput` from a CLASS-backed cosmology and recipe."""
        recipe = "linear_pk" if input_recipe is None else input_recipe
        if recipe == "linear_pk":
            support_k = _default_support_k(settings) if k is None else np.asarray(k, dtype=float)
            return build_linear_input_from_classy(cosmology, z=float(z), k=support_k)
        if recipe == "classpt_parity":
            support_k = _default_support_k(settings) if k is None else np.asarray(k, dtype=float)
            return build_classpt_parity_linear_input_from_classy(cosmology, z=float(z), k=support_k)
        if recipe == "classpt_fftlog_grid_parity":
            if k is not None:
                raise ValueError("CLASS FFTLog-grid parity recipes derive the support grid internally and do not accept k.")
            return build_classpt_fftlog_grid_parity_linear_input_from_classy(cosmology, z=float(z), settings=settings)
        raise ValueError(
            "Unsupported CLASS query recipe. Expected one of {'linear_pk', 'classpt_parity', 'classpt_fftlog_grid_parity'}."
        )


class CosmoprimoCosmologyProvider(BaseCosmologyProvider):
    """Cosmology provider backed by `cosmoprimo`.

    Parameters
    ----------
    fiducial_params
        Queryable cosmology parameters and fiducial values.
    fixed_params
        Fixed engine/configuration parameters, including the engine name.
    base_cosmology
        Optional live cosmoprimo object used as the cloning base for queries.
    """

    name = "cosmoprimo"

    def __init__(self, fiducial_params: dict[str, Any], *, fixed_params: dict[str, Any], base_cosmology: Any | None = None) -> None:
        super().__init__(fiducial_params)
        self.fixed_params = dict(fixed_params)
        self.base_cosmology = base_cosmology
        self.query_param_names = set(fiducial_params)

    @classmethod
    def from_cosmology(cls, cosmo: Any) -> "CosmoprimoCosmologyProvider":
        """Build a provider from an existing `cosmoprimo.Cosmology`."""
        params = _extract_cosmoprimo_fiducial_params(cosmo)
        fixed_params = {name: params.pop(name) for name in list(params) if name in _COSMOPRIMO_FIXED_PARAM_NAMES or name == "engine"}
        return cls(fiducial_params=params, fixed_params=fixed_params, base_cosmology=cosmo)

    @classmethod
    def from_mapping(cls, params: dict[str, Any], *, engine: str = "class") -> "CosmoprimoCosmologyProvider":
        """Build a provider from a fiducial cosmology parameter mapping."""
        fixed_params = {"engine": engine}
        query_params = dict(params)
        for name in list(query_params):
            if name in _COSMOPRIMO_FIXED_PARAM_NAMES:
                fixed_params[name] = query_params.pop(name)
        return cls(fiducial_params=query_params, fixed_params=fixed_params)

    def build_cosmology(self, cosmology_params: dict[str, Any]) -> Any:
        """Instantiate or clone a `cosmoprimo.Cosmology` object."""
        from cosmoprimo import Cosmology

        params = {**self.fixed_params, **cosmology_params}
        engine = params.pop("engine", "class")
        if self.base_cosmology is not None:
            return self.base_cosmology.clone(base="input", **cosmology_params)
        return Cosmology(engine=engine, **params)

    def build_linear_input(self, *, cosmology: Any, z: float, k: np.ndarray | None, settings: PTSettings, input_recipe: str | None) -> LinearPowerInput:
        """Build `LinearPowerInput` from a cosmoprimo-backed cosmology."""
        if input_recipe not in {None, "linear_pk"}:
            raise ValueError("Cosmoprimo queryable templates only support input_recipe='linear_pk'.")
        support_k = _default_support_k(settings) if k is None else np.asarray(k, dtype=float)
        return build_linear_input_from_cosmoprimo(cosmology, z=float(z), k=support_k)


def _default_support_k(settings: PTSettings) -> np.ndarray:
    return np.logspace(-5.0, 1.0, int(settings.integration_nk))
