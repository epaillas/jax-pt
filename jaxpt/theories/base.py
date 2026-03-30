from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import PTSettings
from ..cosmology import LinearPowerInput
from ..reference.classpt import MultipolePrediction
from .defaults import load_galaxy_power_spectrum_multipoles_defaults, load_power_spectrum_template_defaults


_NUISANCE_DEFAULTS = load_galaxy_power_spectrum_multipoles_defaults()
_NUISANCE_PARAM_NAMES = tuple(_NUISANCE_DEFAULTS)
_TEMPLATE_DEFAULTS = load_power_spectrum_template_defaults()
_COSMOLOGY_ALIAS_NAMES = {"A_s", "logA", "ln10^10A_s", "h", "H0"}
_COMMON_COSMOLOGY_PARAM_NAMES = set(_TEMPLATE_DEFAULTS) | _COSMOLOGY_ALIAS_NAMES


def normalize_flat_query(
    parameters: Mapping[str, float] | None,
    kwargs: Mapping[str, float],
) -> dict[str, float]:
    if parameters is None:
        base: dict[str, float] = {}
    elif isinstance(parameters, Mapping):
        base = {str(name): float(value) for name, value in parameters.items()}
    else:
        raise TypeError("Theory parameters must be provided as a mapping or flat keyword arguments.")

    overlap = sorted(set(base) & set(kwargs))
    if overlap:
        raise ValueError(f"Duplicate query parameters provided both positionally and by keyword: {', '.join(overlap)}.")
    for name, value in kwargs.items():
        base[str(name)] = float(value)
    return base


def normalize_nuisance_params(
    parameters: Mapping[str, float],
    *,
    defaults: Mapping[str, float] | None = None,
) -> dict[str, float]:
    allowed = set(_NUISANCE_PARAM_NAMES)
    extra = sorted(set(parameters) - allowed)
    merged = dict(_NUISANCE_DEFAULTS if defaults is None else defaults)
    if extra:
        raise ValueError(f"unexpected parameters: {', '.join(extra)}")
    for name, value in parameters.items():
        merged[name] = float(value)
    missing = [name for name in _NUISANCE_PARAM_NAMES if name not in merged]
    if missing:
        raise ValueError(f"missing required parameters: {', '.join(missing)}")
    return {name: float(merged[name]) for name in _NUISANCE_PARAM_NAMES}


def finalize_multipole_prediction(
    prediction: MultipolePrediction,
    *,
    theory_name: str,
    template_name: str,
) -> MultipolePrediction:
    metadata = dict(prediction.metadata)
    metadata.update({"theory": theory_name, "template": template_name})
    return MultipolePrediction(
        k=prediction.k,
        p0=prediction.p0,
        p2=prediction.p2,
        p4=prediction.p4,
        components=prediction.components,
        metadata=metadata,
    )


class CosmologyQueryMixin:
    template: Any
    nuisance_defaults: Mapping[str, float]

    def _split_query(self, query: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        nuisance, cosmology, unknown = {}, {}, []
        for name, value in query.items():
            if name in self.nuisance_defaults:
                nuisance[name] = value
            elif self.template.is_queryable and name in self.template.cosmology_param_names:
                cosmology[name] = value
            elif self.template.is_queryable and name in _COSMOLOGY_ALIAS_NAMES:
                cosmology[name] = value
            elif (not self.template.is_queryable) and name in _COMMON_COSMOLOGY_PARAM_NAMES:
                cosmology[name] = value
            else:
                unknown.append(name)
        if unknown:
            raise ValueError(f"unexpected parameters: {', '.join(sorted(unknown))}")
        return normalize_nuisance_params(nuisance, defaults=self.nuisance_defaults), cosmology


@dataclass(slots=True)
class BasePowerSpectrumTheory(CosmologyQueryMixin):
    template: Any
    k: np.ndarray
    return_components: bool = False

    def __post_init__(self) -> None:
        self.k = np.asarray(self.k, dtype=float)
        if self.k.ndim != 1:
            raise ValueError(f"{self.__class__.__name__}.k must be a one-dimensional array.")

    @property
    def settings(self) -> PTSettings:
        return self.template.settings

    @property
    def linear_input(self) -> LinearPowerInput:
        return self.template.linear_input

    @property
    def z(self) -> float:
        return float(self.template.z)
