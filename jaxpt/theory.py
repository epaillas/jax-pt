from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from .bias import galaxy_multipoles
from .config import EFTBiasParams, PTSettings
from .cosmology import LinearPowerInput, build_linear_input_from_classy, build_linear_input_from_cosmoprimo
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


@dataclass(frozen=True, slots=True)
class PowerSpectrumTemplate:
    linear_input: LinearPowerInput
    settings: PTSettings = field(default_factory=PTSettings)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_linear_input(
        cls,
        linear_input: LinearPowerInput,
        *,
        settings: PTSettings | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PowerSpectrumTemplate:
        return cls(
            linear_input=linear_input,
            settings=PTSettings() if settings is None else settings,
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def from_classy(
        cls,
        cosmo: Any,
        *,
        z: float,
        k: np.ndarray,
        settings: PTSettings | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PowerSpectrumTemplate:
        linear_input = build_linear_input_from_classy(cosmo, z=z, k=np.asarray(k, dtype=float))
        return cls.from_linear_input(
            linear_input,
            settings=settings,
            metadata=metadata,
        )

    @classmethod
    def from_cosmoprimo(
        cls,
        cosmo: Any,
        *,
        z: float,
        k: np.ndarray,
        settings: PTSettings | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PowerSpectrumTemplate:
        linear_input = build_linear_input_from_cosmoprimo(cosmo, z=z, k=np.asarray(k, dtype=float))
        return cls.from_linear_input(
            linear_input,
            settings=settings,
            metadata=metadata,
        )

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
                self._require_classpt_cosmo(),
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
        metadata = dict(prediction.metadata)
        metadata.update(
            {
                "theory": self.__class__.__name__,
                "template": self.template.__class__.__name__,
            }
        )
        return MultipolePrediction(
            k=prediction.k,
            p0=prediction.p0,
            p2=prediction.p2,
            p4=prediction.p4,
            components=prediction.components,
            metadata=metadata,
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
        cosmo = self.template.linear_input.metadata.get("_classpt_cosmo")
        if cosmo is None:
            raise ValueError(
                "GalaxyPowerSpectrumMultipolesTheory with settings.backend == 'classpt' requires a template built from a live classy.Class cosmology."
            )
        return cosmo
