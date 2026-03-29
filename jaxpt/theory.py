from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from .bias import galaxy_multipoles
from .config import EFTBiasParams, PTSettings
from .cosmology import (
    LinearPowerInput,
    build_classpt_native_grid_parity_linear_input_from_classy,
    build_classpt_parity_linear_input_from_classy,
    build_linear_input_from_classy,
    build_linear_input_from_cosmoprimo,
)
from .native import compute_basis
from .reference.classpt import BasisSpectra, MultipolePrediction, predict_classpt_multipoles


_EFT_PARAM_NAMES = tuple(param.name for param in fields(EFTBiasParams))


def _normalize_eft_params(parameters: EFTBiasParams | Mapping[str, float]) -> EFTBiasParams:
    if isinstance(parameters, EFTBiasParams):
        return parameters
    if not isinstance(parameters, Mapping):
        raise TypeError("Theory parameters must be provided as an EFTBiasParams instance or a mapping of parameter names to floats.")

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


def _default_support_k(settings: PTSettings) -> np.ndarray:
    return np.logspace(-5.0, 1.0, int(settings.integration_nk))


def _is_cosmoprimo_cosmology(source: Any) -> bool:
    return hasattr(source, "get_fourier") and hasattr(source, "get_background")


def _is_classy_cosmology(source: Any) -> bool:
    return hasattr(source, "pk_lin") and hasattr(source, "h") and hasattr(source, "scale_independent_growth_factor_f")


def _build_template_linear_input(
    source: Any,
    *,
    z: float | None,
    k: np.ndarray | None,
    settings: PTSettings,
    input_recipe: str | None,
) -> LinearPowerInput:
    if isinstance(source, LinearPowerInput):
        if z is not None and not np.isclose(float(z), float(source.z)):
            raise ValueError("PowerSpectrumTemplate z must match LinearPowerInput.z when both are provided.")
        if k is not None:
            raise ValueError("PowerSpectrumTemplate.k is only supported when constructing from a cosmology object.")
        if input_recipe is not None:
            raise ValueError("PowerSpectrumTemplate.input_recipe is only supported when constructing from a cosmology object.")
        return source

    if z is None:
        raise ValueError("PowerSpectrumTemplate requires z when constructed from a cosmology object.")

    if _is_cosmoprimo_cosmology(source):
        if input_recipe not in {None, "linear_pk"}:
            raise ValueError("Cosmoprimo-backed PowerSpectrumTemplate only supports input_recipe='linear_pk'.")
        support_k = _default_support_k(settings) if k is None else np.asarray(k, dtype=float)
        return build_linear_input_from_cosmoprimo(source, z=float(z), k=support_k)

    if _is_classy_cosmology(source):
        recipe = "linear_pk" if input_recipe is None else input_recipe
        if recipe == "linear_pk":
            support_k = _default_support_k(settings) if k is None else np.asarray(k, dtype=float)
            return build_linear_input_from_classy(source, z=float(z), k=support_k)
        if recipe == "classpt_parity":
            support_k = _default_support_k(settings) if k is None else np.asarray(k, dtype=float)
            return build_classpt_parity_linear_input_from_classy(source, z=float(z), k=support_k)
        if recipe == "classpt_native_grid_parity":
            if k is not None:
                raise ValueError(
                    "PowerSpectrumTemplate with input_recipe='classpt_native_grid_parity' derives its support grid from settings and does not accept k."
                )
            return build_classpt_native_grid_parity_linear_input_from_classy(source, z=float(z), settings=settings)
        raise ValueError(
            "Unsupported PowerSpectrumTemplate input_recipe for classy cosmology. "
            "Expected one of {'linear_pk', 'classpt_parity', 'classpt_native_grid_parity'}."
        )

    raise TypeError(
        "PowerSpectrumTemplate only accepts LinearPowerInput or a supported cosmology object. "
        "Supported cosmology sources are classy.Class-like and cosmoprimo.Cosmology-like objects."
    )


def _finalize_multipole_prediction(
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


def _require_classpt_cosmo(template: "PowerSpectrumTemplate", theory_name: str):
    cosmo = template.linear_input.metadata.get("_classpt_cosmo")
    if cosmo is None:
        raise ValueError(
            f"{theory_name} requires a template built from a live classy.Class cosmology."
        )
    return cosmo


@dataclass(frozen=True, slots=True, init=False)
class PowerSpectrumTemplate:
    linear_input: LinearPowerInput
    settings: PTSettings = field(default_factory=PTSettings)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        source: LinearPowerInput | Any,
        *,
        z: float | None = None,
        k: np.ndarray | None = None,
        settings: PTSettings | None = None,
        input_recipe: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_settings = PTSettings() if settings is None else settings
        linear_input = _build_template_linear_input(
            source,
            z=z,
            k=k,
            settings=normalized_settings,
            input_recipe=input_recipe,
        )
        object.__setattr__(self, "linear_input", linear_input)
        object.__setattr__(self, "settings", normalized_settings)
        object.__setattr__(self, "metadata", {} if metadata is None else dict(metadata))

    @classmethod
    def from_linear_input(
        cls,
        linear_input: LinearPowerInput,
        *,
        settings: PTSettings | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PowerSpectrumTemplate:
        return cls(linear_input, settings=settings, metadata=metadata)

    @classmethod
    def from_classy(
        cls,
        cosmo: Any,
        *,
        z: float,
        k: np.ndarray | None = None,
        settings: PTSettings | None = None,
        input_recipe: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PowerSpectrumTemplate:
        return cls(cosmo, z=z, k=k, settings=settings, input_recipe=input_recipe, metadata=metadata)

    @classmethod
    def from_cosmoprimo(
        cls,
        cosmo: Any,
        *,
        z: float,
        k: np.ndarray | None = None,
        settings: PTSettings | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PowerSpectrumTemplate:
        return cls(cosmo, z=z, k=k, settings=settings, metadata=metadata)

    @property
    def z(self) -> float:
        return float(self.linear_input.z)


@dataclass(frozen=True, slots=True)
class GalaxyPowerSpectrumMultipolesTheory:
    template: PowerSpectrumTemplate
    k: np.ndarray
    return_components: bool = False
    basis: BasisSpectra | None = field(init=False)

    def __post_init__(self) -> None:
        eval_k = np.asarray(self.k, dtype=float)
        if eval_k.ndim != 1:
            raise ValueError("GalaxyPowerSpectrumMultipolesTheory.k must be a one-dimensional array.")
        object.__setattr__(self, "k", eval_k)
        if self.template.settings.backend == "native":
            basis = compute_basis(
                self.template.linear_input,
                settings=self.template.settings,
                k=eval_k,
            )
        elif self.template.settings.backend == "classpt":
            basis = None
            self._require_classpt_cosmo()
        else:
            raise ValueError(f"Unsupported multipole backend '{self.template.settings.backend}'.")
        object.__setattr__(self, "basis", basis)

    def __call__(
        self,
        parameters: EFTBiasParams | Mapping[str, float],
        *,
        return_components: bool | None = None,
    ) -> MultipolePrediction:
        params = _normalize_eft_params(parameters)
        requested_components = self.return_components if return_components is None else return_components
        if self.template.settings.backend == "classpt":
            if requested_components:
                raise NotImplementedError("CLASS-PT reference predictions do not expose decomposed multipole components through GalaxyPowerSpectrumMultipolesTheory.")
            prediction = predict_classpt_multipoles(
                _require_classpt_cosmo(self.template, self.__class__.__name__),
                self.k,
                self.z,
                params,
            )
        else:
            prediction = galaxy_multipoles(
                self.basis,
                params,
                return_components=requested_components,
            )
        return _finalize_multipole_prediction(
            prediction,
            theory_name=self.__class__.__name__,
            template_name=self.template.__class__.__name__,
        )

    @property
    def settings(self) -> PTSettings:
        return self.template.settings

    @property
    def linear_input(self) -> LinearPowerInput:
        return self.template.linear_input

    @property
    def z(self) -> float:
        return self.template.z

    def _require_classpt_cosmo(self):
        return _require_classpt_cosmo(self.template, self.__class__.__name__)


@dataclass(frozen=True, slots=True)
class ClassPTGalaxyPowerSpectrumMultipolesTheory:
    template: PowerSpectrumTemplate
    k: np.ndarray
    return_components: bool = False

    def __post_init__(self) -> None:
        eval_k = np.asarray(self.k, dtype=float)
        if eval_k.ndim != 1:
            raise ValueError("ClassPTGalaxyPowerSpectrumMultipolesTheory.k must be a one-dimensional array.")
        object.__setattr__(self, "k", eval_k)
        self._require_classpt_cosmo()

    def __call__(
        self,
        parameters: EFTBiasParams | Mapping[str, float],
        *,
        return_components: bool | None = None,
    ) -> MultipolePrediction:
        params = _normalize_eft_params(parameters)
        requested_components = self.return_components if return_components is None else return_components
        if requested_components:
            raise NotImplementedError("CLASS-PT reference predictions do not expose decomposed multipole components.")
        prediction = predict_classpt_multipoles(
            self._require_classpt_cosmo(),
            self.k,
            self.z,
            params,
        )
        return _finalize_multipole_prediction(
            prediction,
            theory_name=self.__class__.__name__,
            template_name=self.template.__class__.__name__,
        )

    @property
    def settings(self) -> PTSettings:
        return self.template.settings

    @property
    def linear_input(self) -> LinearPowerInput:
        return self.template.linear_input

    @property
    def z(self) -> float:
        return self.template.z

    def _require_classpt_cosmo(self):
        return _require_classpt_cosmo(self.template, self.__class__.__name__)
