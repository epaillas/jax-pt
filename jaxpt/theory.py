from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from .bias import galaxy_multipoles
from .config import EFTBiasParams, PTSettings
from .cosmology import (
    BaseCosmologyProvider,
    ClassyCosmologyProvider,
    CosmoprimoCosmologyProvider,
    LinearPowerInput,
    ResolvedCosmologyState,
)
from .native import compute_basis
from .reference.classpt import BasisSpectra, MultipolePrediction, predict_classpt_multipoles


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


def _normalize_flat_query(
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


def _normalize_nuisance_params(parameters: Mapping[str, float]) -> EFTBiasParams:
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


def _is_cosmoprimo_cosmology(source: Any) -> bool:
    return hasattr(source, "get_fourier") and hasattr(source, "get_background")


def _is_classy_cosmology(source: Any) -> bool:
    return hasattr(source, "pk_lin") and hasattr(source, "h") and hasattr(source, "scale_independent_growth_factor_f")


def _build_provider(
    source: Any,
    *,
    z: float,
    settings: PTSettings,
    input_recipe: str | None,
    provider: str | None,
) -> BaseCosmologyProvider:
    if provider is not None:
        provider = provider.lower()

    if _is_classy_cosmology(source):
        return ClassyCosmologyProvider.from_cosmology(source)
    if _is_cosmoprimo_cosmology(source):
        return CosmoprimoCosmologyProvider.from_cosmology(source)
    if isinstance(source, Mapping):
        if provider is None:
            provider = "classy"
        if provider == "classy":
            return ClassyCosmologyProvider.from_mapping(dict(source), z=float(z), settings=settings, input_recipe=input_recipe)
        if provider == "cosmoprimo":
            return CosmoprimoCosmologyProvider.from_mapping(dict(source))
        raise ValueError("Unsupported PowerSpectrumTemplate provider. Expected one of {'classy', 'cosmoprimo'}.")
    raise TypeError(
        "PowerSpectrumTemplate only accepts LinearPowerInput, a supported cosmology object, or a fiducial cosmology mapping."
    )


@dataclass(slots=True)
class PowerSpectrumTemplate:
    """Template holding fixed setup plus fiducial cosmology defaults.

    Templates built from a cosmology object or a fiducial cosmology mapping are
    query-aware: they can rebuild their cosmology-dependent state when the user
    provides cosmology overrides at theory evaluation time. Templates built from
    a `LinearPowerInput` are fixed-cosmology only.
    """

    source: Any
    z: float | None = None
    k: np.ndarray | None = None
    settings: PTSettings = field(default_factory=PTSettings)
    input_recipe: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provider: str | None = None
    _fixed_linear_input: LinearPowerInput | None = field(init=False, default=None, repr=False)
    _cosmology_provider: BaseCosmologyProvider | None = field(init=False, default=None, repr=False)
    _resolved_state: ResolvedCosmologyState | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.k is not None:
            self.k = np.asarray(self.k, dtype=float)
        self.metadata = dict(self.metadata)
        if isinstance(self.source, LinearPowerInput):
            if self.z is not None and not np.isclose(float(self.z), float(self.source.z)):
                raise ValueError("PowerSpectrumTemplate z must match LinearPowerInput.z when both are provided.")
            if self.input_recipe is not None:
                raise ValueError("PowerSpectrumTemplate.input_recipe is only supported for cosmology-backed templates.")
            self._fixed_linear_input = self.source
            self.z = float(self.source.z)
            return

        if self.z is None:
            raise ValueError("PowerSpectrumTemplate requires z when constructed from a cosmology object or fiducial cosmology mapping.")
        self.z = float(self.z)
        self._cosmology_provider = _build_provider(
            self.source,
            z=self.z,
            settings=self.settings,
            input_recipe=self.input_recipe,
            provider=self.provider,
        )

    @classmethod
    def from_linear_input(
        cls,
        linear_input: LinearPowerInput,
        *,
        settings: PTSettings | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PowerSpectrumTemplate:
        return cls(linear_input, settings=PTSettings() if settings is None else settings, metadata={} if metadata is None else dict(metadata))

    @property
    def linear_input(self) -> LinearPowerInput:
        return self.resolve({}).linear_input

    @property
    def cosmology_param_names(self) -> set[str]:
        if self._cosmology_provider is None:
            return set()
        return set(self._cosmology_provider.query_param_names)

    @property
    def is_queryable(self) -> bool:
        return self._cosmology_provider is not None

    def resolve(self, cosmology_overrides: Mapping[str, float] | None = None) -> ResolvedCosmologyState:
        overrides = {} if cosmology_overrides is None else {str(name): float(value) for name, value in cosmology_overrides.items()}
        if self._fixed_linear_input is not None:
            if overrides:
                names = ", ".join(sorted(overrides))
                raise ValueError(
                    "This PowerSpectrumTemplate was built from a fixed LinearPowerInput and cannot accept cosmology overrides. "
                    f"Received: {names}."
                )
            if self._resolved_state is None:
                self._resolved_state = ResolvedCosmologyState(
                    cosmology=None,
                    linear_input=self._fixed_linear_input,
                    cosmology_params={},
                    query_key=(),
                )
            return self._resolved_state

        assert self._cosmology_provider is not None
        state = self._cosmology_provider.resolve(
            overrides=overrides,
            z=float(self.z),
            k=self.k,
            settings=self.settings,
            input_recipe=self.input_recipe,
        )
        if self._resolved_state is None or self._resolved_state.query_key != state.query_key:
            self._resolved_state = state
        return self._resolved_state


@dataclass(slots=True)
class GalaxyPowerSpectrumMultipolesTheory:
    template: PowerSpectrumTemplate
    k: np.ndarray
    return_components: bool = False
    _basis: BasisSpectra | None = field(init=False, default=None, repr=False)
    _basis_query_key: tuple[tuple[str, Any], ...] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.k = np.asarray(self.k, dtype=float)
        if self.k.ndim != 1:
            raise ValueError("GalaxyPowerSpectrumMultipolesTheory.k must be a one-dimensional array.")
        if self.template.settings.backend not in {"native", "classpt"}:
            raise ValueError(f"Unsupported multipole backend '{self.template.settings.backend}'.")
        if self.template.settings.backend == "classpt" and not self.template.is_queryable:
            if self.template.linear_input.metadata.get("_classpt_cosmo") is None:
                raise ValueError("GalaxyPowerSpectrumMultipolesTheory requires a template built from a live classy.Class cosmology.")

    @property
    def settings(self) -> PTSettings:
        return self.template.settings

    @property
    def linear_input(self) -> LinearPowerInput:
        return self.template.linear_input

    @property
    def z(self) -> float:
        return float(self.template.z)

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
        return _normalize_nuisance_params(nuisance), cosmology

    def __call__(
        self,
        parameters: EFTBiasParams | Mapping[str, float] | None = None,
        *,
        return_components: bool | None = None,
        **kwargs: float,
    ) -> MultipolePrediction:
        query = _normalize_flat_query(parameters, kwargs)
        nuisance_params, cosmology_params = self._split_query(query)
        requested_components = self.return_components if return_components is None else return_components
        state = self.template.resolve(cosmology_params)

        if self.template.settings.backend == "classpt":
            if requested_components:
                raise NotImplementedError(
                    "CLASS-PT reference predictions do not expose decomposed multipole components through GalaxyPowerSpectrumMultipolesTheory."
                )
            cosmo = state.cosmology
            if cosmo is None:
                cosmo = state.linear_input.metadata.get("_classpt_cosmo")
            if cosmo is None:
                raise ValueError("CLASS-PT predictions require a queryable cosmology-backed template.")
            prediction = predict_classpt_multipoles(
                cosmo,
                self.k,
                self.z,
                nuisance_params,
            )
        else:
            if self._basis is None or self._basis_query_key != state.query_key:
                self._basis = compute_basis(state.linear_input, settings=self.template.settings, k=self.k)
                self._basis_query_key = state.query_key
            prediction = galaxy_multipoles(
                self._basis,
                nuisance_params,
                return_components=requested_components,
            )

        return _finalize_multipole_prediction(
            prediction,
            theory_name=self.__class__.__name__,
            template_name=self.template.__class__.__name__,
        )


@dataclass(slots=True)
class ClassPTGalaxyPowerSpectrumMultipolesTheory:
    template: PowerSpectrumTemplate
    k: np.ndarray
    return_components: bool = False

    def __post_init__(self) -> None:
        self.k = np.asarray(self.k, dtype=float)
        if self.k.ndim != 1:
            raise ValueError("ClassPTGalaxyPowerSpectrumMultipolesTheory.k must be a one-dimensional array.")
        if not self.template.is_queryable and self.template.linear_input.metadata.get("_classpt_cosmo") is None:
            raise ValueError("ClassPTGalaxyPowerSpectrumMultipolesTheory requires a template built from a live classy.Class cosmology.")

    @property
    def settings(self) -> PTSettings:
        return self.template.settings

    @property
    def linear_input(self) -> LinearPowerInput:
        return self.template.linear_input

    @property
    def z(self) -> float:
        return float(self.template.z)

    def __call__(
        self,
        parameters: EFTBiasParams | Mapping[str, float] | None = None,
        *,
        return_components: bool | None = None,
        **kwargs: float,
    ) -> MultipolePrediction:
        query = _normalize_flat_query(parameters, kwargs)
        nuisance, cosmology = GalaxyPowerSpectrumMultipolesTheory._split_query(self, query)
        requested_components = self.return_components if return_components is None else return_components
        if requested_components:
            raise NotImplementedError("CLASS-PT reference predictions do not expose decomposed multipole components.")
        state = self.template.resolve(cosmology)
        cosmo = state.cosmology
        if cosmo is None:
            cosmo = state.linear_input.metadata.get("_classpt_cosmo")
        if cosmo is None:
            raise ValueError("CLASS-PT predictions require a queryable cosmology-backed template.")
        prediction = predict_classpt_multipoles(
            cosmo,
            self.k,
            self.z,
            nuisance,
        )
        return _finalize_multipole_prediction(
            prediction,
            theory_name=self.__class__.__name__,
            template_name=self.template.__class__.__name__,
        )
