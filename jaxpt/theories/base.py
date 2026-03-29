from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from ..config import EFTBiasParams, PTSettings
from ..cosmology import LinearPowerInput
from ..reference.classpt import MultipolePrediction


_EFT_PARAM_NAMES = tuple(param.name for param in fields(EFTBiasParams))
_COSMOLOGY_ALIAS_NAMES = {"A_s", "logA", "ln10^10A_s", "h", "H0"}
_COMMON_COSMOLOGY_PARAM_NAMES = {
    "A_s",
    "logA",
    "ln10^10A_s",
    "h",
    "H0",
    "n_s",
    "tau_reio",
    "omega_b",
    "omega_cdm",
    "N_ur",
    "N_ncdm",
    "m_ncdm",
    "YHe",
    "Omega_k",
    "w0_fld",
    "wa_fld",
}


def _eft_params_as_dict(params: EFTBiasParams) -> dict[str, float]:
    return {name: float(getattr(params, name)) for name in _EFT_PARAM_NAMES}


def normalize_flat_query(
    parameters: EFTBiasParams | Mapping[str, float] | None,
    kwargs: Mapping[str, float],
) -> dict[str, float]:
    if parameters is None:
        base: dict[str, float] = {}
    elif isinstance(parameters, EFTBiasParams):
        base = _eft_params_as_dict(parameters)
    elif isinstance(parameters, Mapping):
        base = {str(name): float(value) for name, value in parameters.items()}
    else:
        raise TypeError(
            "Theory parameters must be provided as an EFTBiasParams instance, a mapping, or flat keyword arguments."
        )

    overlap = sorted(set(base) & set(kwargs))
    if overlap:
        raise ValueError(f"Duplicate query parameters provided both positionally and by keyword: {', '.join(overlap)}.")
    for name, value in kwargs.items():
        base[str(name)] = float(value)
    return base


def normalize_nuisance_params(parameters: Mapping[str, float]) -> EFTBiasParams:
    extra = sorted(set(parameters) - set(_EFT_PARAM_NAMES))
    missing = [name for name in _EFT_PARAM_NAMES if name not in parameters]
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing required parameters: {', '.join(missing)}")
        if extra:
            parts.append(f"unexpected parameters: {', '.join(extra)}")
        raise ValueError("; ".join(parts))
    return EFTBiasParams(**{name: float(parameters[name]) for name in _EFT_PARAM_NAMES})


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

    def _split_query(self, query: Mapping[str, float]) -> tuple[EFTBiasParams, dict[str, float]]:
        nuisance, cosmology, unknown = {}, {}, []
        for name, value in query.items():
            if name in _EFT_PARAM_NAMES:
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
        return normalize_nuisance_params(nuisance), cosmology


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
