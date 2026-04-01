from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ..parameter import ParameterCollection
from ..reference.classpt import MultipolePrediction, multipole_prediction_to_array
from .base import BasePowerSpectrumTheory, finalize_multipole_prediction, normalize_flat_query
from .defaults import load_density_split_galaxy_power_spectrum_multipoles_parameters


_SUPPORTED_SMOOTHING_KERNELS = {"gaussian", "tophat"}
_QUANTILES = (1, 2, 3, 4, 5)
_DENSITY_SPLIT_PARAMETERS = load_density_split_galaxy_power_spectrum_multipoles_parameters()


def default_density_split_nuisance_parameters() -> ParameterCollection:
    """Return a fresh copy of the packaged density-split nuisance parameters."""
    return _DENSITY_SPLIT_PARAMETERS.copy()


def _normalize_quantiles(quantiles: tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if quantiles is None:
        return _QUANTILES
    resolved = tuple(int(quantile) for quantile in quantiles)
    invalid = [quantile for quantile in resolved if quantile not in _QUANTILES]
    if invalid:
        raise ValueError(f"Unsupported quantiles requested: {', '.join(str(value) for value in invalid)}.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Requested quantiles must be unique.")
    return resolved


def _smoothing_argument(k: np.ndarray, radius_hmpc: float, h: float) -> np.ndarray:
    """Return the dimensionless smoothing argument for k in 1/Mpc and R in Mpc/h."""
    return np.asarray(k, dtype=float) * float(radius_hmpc) / float(h)


def smoothing_kernel(k: np.ndarray, *, radius_hmpc: float, h: float, kind: str = "gaussian") -> np.ndarray:
    """Evaluate the isotropic density-split smoothing kernel."""
    kernel = str(kind).lower()
    if kernel not in _SUPPORTED_SMOOTHING_KERNELS:
        raise ValueError(f"Unsupported smoothing kernel '{kind}'. Expected one of {_SUPPORTED_SMOOTHING_KERNELS}.")

    x = _smoothing_argument(k, radius_hmpc=radius_hmpc, h=h)
    if kernel == "gaussian":
        return np.exp(-0.5 * x**2)

    result = np.ones_like(x)
    nonzero = x != 0.0
    xnz = x[nonzero]
    result[nonzero] = 3.0 * (np.sin(xnz) - xnz * np.cos(xnz)) / xnz**3
    return result


def _interpolate_linear_power(k_eval: np.ndarray, k_input: np.ndarray, pk_input: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(k_eval, dtype=float), np.asarray(k_input, dtype=float), np.asarray(pk_input, dtype=float))


def _tree_multipoles(
    *,
    pk_linear: np.ndarray,
    smoothing: np.ndarray,
    growth_rate: float,
    b1: float,
    bq: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    common = np.asarray(pk_linear, dtype=float) * np.asarray(smoothing, dtype=float)
    f = float(growth_rate)

    density_density = common * (bq * b1)
    mixed_rsd = common * (f * (bq + beta * b1))
    rsd_rsd = common * (beta * f**2)

    p0 = density_density + mixed_rsd / 3.0 + rsd_rsd / 5.0
    p2 = 2.0 * mixed_rsd / 3.0 + 4.0 * rsd_rsd / 7.0
    p4 = 8.0 * rsd_rsd / 35.0
    components = {
        "density_density": density_density,
        "mixed_rsd": mixed_rsd,
        "rsd_rsd": rsd_rsd,
        "smoothing_kernel": np.asarray(smoothing, dtype=float),
        "linear_power": np.asarray(pk_linear, dtype=float),
    }
    return p0, p2, p4, components


@dataclass(slots=True)
class QuantileGalaxyPowerSpectrumMultipolesTheory(BasePowerSpectrumTheory):
    """Tree-level density-split quantile-galaxy cross-power multipoles from Eq. 81.

    The theory evaluates all requested quantiles independently using

    ``P_qg(k, mu) = W_R(k) (b_q + beta_q f mu^2) (b1 + f mu^2) P_L(k)``

    with a configurable isotropic smoothing kernel. The smoothing radius is
    interpreted in ``Mpc/h`` while the theory wavenumbers are in ``1/Mpc``.
    """

    nuisance_parameters: ParameterCollection = field(default_factory=default_density_split_nuisance_parameters, repr=False)
    smoothing_radius_hmpc: float = 10.0
    smoothing_kernel_kind: str = "gaussian"

    def __post_init__(self) -> None:
        BasePowerSpectrumTheory.__post_init__(self)
        if float(self.smoothing_radius_hmpc) <= 0.0:
            raise ValueError("QuantileGalaxyPowerSpectrumMultipolesTheory.smoothing_radius_hmpc must be positive.")
        kernel = str(self.smoothing_kernel_kind).lower()
        if kernel not in _SUPPORTED_SMOOTHING_KERNELS:
            raise ValueError(
                f"Unsupported smoothing kernel '{self.smoothing_kernel_kind}'. Expected one of {_SUPPORTED_SMOOTHING_KERNELS}."
            )
        self.smoothing_radius_hmpc = float(self.smoothing_radius_hmpc)
        self.smoothing_kernel_kind = kernel

    @property
    def quantiles(self) -> tuple[int, ...]:
        """Return the available quantile labels."""
        return _QUANTILES

    def _predict_quantile(
        self,
        quantile: int,
        nuisance_params: Mapping[str, float],
        *,
        return_components: bool,
        cosmology_params: Mapping[str, float] | None = None,
    ) -> MultipolePrediction:
        state = self.template.resolve(cosmology_params)
        linear_input = state.linear_input
        pk_linear = _interpolate_linear_power(self.k, linear_input.k, linear_input.pk_linear)
        window = smoothing_kernel(
            self.k,
            radius_hmpc=self.smoothing_radius_hmpc,
            h=linear_input.h,
            kind=self.smoothing_kernel_kind,
        )
        p0, p2, p4, components = _tree_multipoles(
            pk_linear=pk_linear,
            smoothing=window,
            growth_rate=linear_input.growth_rate,
            b1=float(nuisance_params["b1"]),
            bq=float(nuisance_params[f"bq{quantile}"]),
            beta=float(nuisance_params[f"beta{quantile}"]),
        )
        prediction = MultipolePrediction(
            k=np.asarray(self.k, dtype=float),
            p0=p0,
            p2=p2,
            p4=p4,
            components=components if return_components else None,
            metadata={
                "backend": "tree",
                "ells": [0, 2, 4],
                "z": linear_input.z,
                "quantile": int(quantile),
                "smoothing_kernel": self.smoothing_kernel_kind,
                "smoothing_radius_hmpc": self.smoothing_radius_hmpc,
            },
        )
        return finalize_multipole_prediction(
            prediction,
            theory_name=self.__class__.__name__,
            template_name=self.template.__class__.__name__,
        )

    def predict_quantiles(
        self,
        parameters: Mapping[str, float] | None = None,
        *,
        quantiles: tuple[int, ...] | list[int] | None = None,
        return_components: bool | None = None,
        **kwargs: float,
    ) -> dict[int, MultipolePrediction]:
        """Return a mapping of quantile label to multipole prediction."""
        query = normalize_flat_query(parameters, kwargs)
        nuisance_params, cosmology_params = self._split_query(query)
        requested_components = self.return_components if return_components is None else return_components
        resolved_quantiles = _normalize_quantiles(quantiles)
        return {
            quantile: self._predict_quantile(
                quantile,
                nuisance_params,
                return_components=requested_components,
                cosmology_params=cosmology_params,
            )
            for quantile in resolved_quantiles
        }

    def __call__(
        self,
        parameters: Mapping[str, float] | None = None,
        *,
        ells: tuple[int, ...] | list[int] | None = None,
        quantiles: tuple[int, ...] | list[int] | None = None,
        return_components: bool | None = None,
        **kwargs: float,
    ) -> np.ndarray:
        """Evaluate the requested quantile multipoles as ``(n_quantiles, n_ells, n_k)``."""
        predictions = self.predict_quantiles(
            parameters,
            quantiles=quantiles,
            return_components=return_components,
            **kwargs,
        )
        return np.stack(
            [multipole_prediction_to_array(predictions[quantile], ells=ells) for quantile in predictions],
            axis=0,
        )
