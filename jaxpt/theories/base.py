from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import PTSettings
from ..cosmology import LinearPowerInput
from ..parameter import ParameterCollection
from ..reference.classpt import MultipolePrediction
from .defaults import (
    load_galaxy_power_spectrum_multipoles_parameters,
    load_power_spectrum_template_parameters,
)


_NUISANCE_DEFAULTS = load_galaxy_power_spectrum_multipoles_parameters().defaults_dict()
_NUISANCE_PARAM_NAMES = tuple(_NUISANCE_DEFAULTS)
_NUISANCE_PARAMETERS = load_galaxy_power_spectrum_multipoles_parameters()
_TEMPLATE_DEFAULTS = load_power_spectrum_template_parameters().defaults_dict()
_COSMOLOGY_ALIAS_NAMES = {"A_s", "logA", "ln10^10A_s", "h", "H0"}
_COMMON_COSMOLOGY_PARAM_NAMES = set(_TEMPLATE_DEFAULTS) | _COSMOLOGY_ALIAS_NAMES


def normalize_flat_query(
    parameters: Mapping[str, float] | None,
    kwargs: Mapping[str, float],
) -> dict[str, float]:
    """Normalize a flat parameter query from mapping and keyword forms.

    Parameters
    ----------
    parameters
        Optional mapping of parameter values supplied positionally.
    kwargs
        Flat keyword arguments to merge into ``parameters``.

    Returns
    -------
    dict
        Normalized ``{name: value}`` mapping with all values cast to ``float``.
    """
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
    """Validate and fill nuisance parameters against the theory defaults.

    Parameters
    ----------
    parameters
        Partial nuisance-parameter mapping. Allowed names are those defined in
        `galaxy_power_spectrum_multipoles.yaml`.
    defaults
        Optional full default mapping to merge into. If omitted, the package
        nuisance defaults are used.
    """
    merged = dict(_NUISANCE_DEFAULTS if defaults is None else defaults)
    allowed = set(merged)
    names = tuple(_NUISANCE_PARAM_NAMES if defaults is None else merged)
    extra = sorted(set(parameters) - allowed)
    if extra:
        raise ValueError(f"unexpected parameters: {', '.join(extra)}")
    for name, value in parameters.items():
        merged[name] = float(value)
    missing = [name for name in names if name not in merged]
    if missing:
        raise ValueError(f"missing required parameters: {', '.join(missing)}")
    return {name: float(merged[name]) for name in names}


def finalize_multipole_prediction(
    prediction: MultipolePrediction,
    *,
    theory_name: str,
    template_name: str,
) -> MultipolePrediction:
    """Attach theory/template provenance metadata to a prediction object."""
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
    """Mixin that splits flat theory queries into nuisance and cosmology parts."""
    template: Any
    nuisance_parameters: ParameterCollection

    @property
    def nuisance_defaults(self) -> dict[str, float]:
        """Return the default nuisance-parameter values for the theory."""
        return self.nuisance_parameters.defaults_dict()

    def _split_query(self, query: Mapping[str, float]) -> tuple[dict[str, float], dict[str, float]]:
        """Split a flat query into nuisance and cosmology parameter groups."""
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
    """Shared state for power-spectrum theory wrappers.

    Parameters
    ----------
    template
        `PowerSpectrumTemplate` describing how cosmology-dependent linear input
        should be resolved.
    k
        One-dimensional evaluation grid, in ``1/Mpc``.
    return_components
        Whether calls should request decomposed observable components when the
        concrete backend supports them.
    """

    template: Any
    k: np.ndarray
    return_components: bool = False

    def __post_init__(self) -> None:
        self.k = np.asarray(self.k, dtype=float)
        if self.k.ndim != 1:
            raise ValueError(f"{self.__class__.__name__}.k must be a one-dimensional array.")

    @property
    def settings(self) -> PTSettings:
        """Shortcut to `template.settings`."""
        return self.template.settings

    @property
    def linear_input(self) -> LinearPowerInput:
        """Shortcut to the currently resolved linear input."""
        return self.template.linear_input

    @property
    def z(self) -> float:
        """Redshift associated with the underlying template."""
        return float(self.template.z)

    @property
    def params(self) -> ParameterCollection:
        """Merged cosmology and nuisance parameters exposed by the theory."""
        return ParameterCollection.combine(self.template.params, self.nuisance_parameters)


def default_nuisance_parameters() -> ParameterCollection:
    """Return a fresh copy of the packaged nuisance-parameter defaults."""
    return _NUISANCE_PARAMETERS.copy()
