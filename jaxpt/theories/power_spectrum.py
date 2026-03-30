from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..bias import galaxy_multipoles
from ..config import PTSettings
from ..cosmology import (
    BaseCosmologyProvider,
    ClassyCosmologyProvider,
    CosmoprimoCosmologyProvider,
    LinearPowerInput,
    ResolvedCosmologyState,
)
from ..native import compute_basis
from ..reference.classpt import BasisSpectra, MultipolePrediction, predict_classpt_multipoles
from .base import BasePowerSpectrumTheory, finalize_multipole_prediction, normalize_flat_query
from .defaults import load_galaxy_power_spectrum_multipoles_defaults, load_power_spectrum_template_defaults


_TEMPLATE_DEFAULTS = load_power_spectrum_template_defaults()


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
        source = {**_TEMPLATE_DEFAULTS, **{str(name): float(value) for name, value in source.items()}}
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
    """Template holding fixed setup plus fiducial cosmology defaults."""

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
    def default_cosmology(self) -> dict[str, float]:
        return dict(_TEMPLATE_DEFAULTS)

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
class GalaxyPowerSpectrumMultipolesTheory(BasePowerSpectrumTheory):
    nuisance_defaults: dict[str, float] = field(default_factory=load_galaxy_power_spectrum_multipoles_defaults)
    _basis: BasisSpectra | None = field(init=False, default=None, repr=False)
    _basis_query_key: tuple[tuple[str, Any], ...] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        BasePowerSpectrumTheory.__post_init__(self)
        if self.template.settings.backend not in {"native", "classpt"}:
            raise ValueError(f"Unsupported multipole backend '{self.template.settings.backend}'.")
        if self.template.settings.backend == "classpt" and not self.template.is_queryable:
            if self.template.linear_input.metadata.get("_classpt_cosmo") is None:
                raise ValueError("GalaxyPowerSpectrumMultipolesTheory requires a template built from a live classy.Class cosmology.")

    def __call__(
        self,
        parameters: Mapping[str, float] | None = None,
        *,
        return_components: bool | None = None,
        **kwargs: float,
    ) -> MultipolePrediction:
        query = normalize_flat_query(parameters, kwargs)
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

        return finalize_multipole_prediction(
            prediction,
            theory_name=self.__class__.__name__,
            template_name=self.template.__class__.__name__,
        )


@dataclass(slots=True)
class ClassPTGalaxyPowerSpectrumMultipolesTheory(BasePowerSpectrumTheory):
    nuisance_defaults: dict[str, float] = field(default_factory=load_galaxy_power_spectrum_multipoles_defaults)

    def __post_init__(self) -> None:
        BasePowerSpectrumTheory.__post_init__(self)
        if not self.template.is_queryable and self.template.linear_input.metadata.get("_classpt_cosmo") is None:
            raise ValueError("ClassPTGalaxyPowerSpectrumMultipolesTheory requires a template built from a live classy.Class cosmology.")

    def __call__(
        self,
        parameters: Mapping[str, float] | None = None,
        *,
        return_components: bool | None = None,
        **kwargs: float,
    ) -> MultipolePrediction:
        query = normalize_flat_query(parameters, kwargs)
        nuisance, cosmology = self._split_query(query)
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
        return finalize_multipole_prediction(
            prediction,
            theory_name=self.__class__.__name__,
            template_name=self.template.__class__.__name__,
        )


def predict_galaxy_multipoles(
    source,
    *args,
    params: Mapping[str, float] | None = None,
    settings: PTSettings | None = None,
    return_components: bool = False,
    **kwargs,
):
    if isinstance(source, (GalaxyPowerSpectrumMultipolesTheory, ClassPTGalaxyPowerSpectrumMultipolesTheory)):
        if params is None:
            params = args[0] if args else None
        return source(params, return_components=return_components, **kwargs)

    if isinstance(source, PowerSpectrumTemplate):
        if params is None:
            params = args[0] if args else None
        theory_cls = (
            ClassPTGalaxyPowerSpectrumMultipolesTheory
            if source.settings.backend == "classpt"
            else GalaxyPowerSpectrumMultipolesTheory
        )
        eval_k = np.asarray(source.linear_input.k if source.k is None else source.k, dtype=float)
        theory = theory_cls(template=source, k=eval_k, return_components=return_components)
        return theory(params, return_components=return_components, **kwargs)

    if isinstance(source, BasisSpectra):
        if params is None:
            if not args:
                raise TypeError("predict_galaxy_multipoles requires params when called with a BasisSpectra source.")
            params = args[0]
        basis = source
    elif isinstance(source, LinearPowerInput):
        if params is None:
            if not args:
                raise TypeError("predict_galaxy_multipoles requires params when called with a LinearPowerInput source.")
            params = args[0]
        theory_cls = ClassPTGalaxyPowerSpectrumMultipolesTheory if settings is not None and settings.backend == "classpt" else GalaxyPowerSpectrumMultipolesTheory
        theory = theory_cls(template=PowerSpectrumTemplate(source, settings=settings), k=np.asarray(source.k, dtype=float), return_components=return_components)
        return theory(params, return_components=return_components)
    else:
        raise TypeError(
            "predict_galaxy_multipoles only accepts GalaxyPowerSpectrumMultipolesTheory, PowerSpectrumTemplate, BasisSpectra, or LinearPowerInput. "
            "Build a LinearPowerInput or PowerSpectrumTemplate before calling it."
        )

    return galaxy_multipoles(basis, params, return_components=return_components)
