from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..parameter import ParameterCollection
from ..reference.classpt import MultipolePrediction


def _as_parameter_collection(priors: ParameterCollection | Mapping[str, Any] | None) -> ParameterCollection | None:
    if priors is None:
        return None
    if isinstance(priors, ParameterCollection):
        return priors.copy()
    return ParameterCollection(priors)


def _resolve_model_params(model: Any) -> ParameterCollection | None:
    params = getattr(model, "params", None)
    if isinstance(params, ParameterCollection):
        return params.copy()
    return None


def _resolve_default_parameter_names(model: Any, params: ParameterCollection | None) -> tuple[str, ...]:
    if hasattr(model, "param_names"):
        return tuple(str(name) for name in getattr(model, "param_names"))
    if params is not None and hasattr(params, "emulated_names"):
        names = tuple(str(name) for name in params.emulated_names())
        if names:
            return names
    raise ValueError("parameter_names must be provided when the model does not expose a default emulated parameter subset.")


def flatten_prediction(prediction: Any) -> np.ndarray:
    if isinstance(prediction, MultipolePrediction):
        return np.concatenate(
            [
                np.asarray(prediction.p0, dtype=float).reshape(-1),
                np.asarray(prediction.p2, dtype=float).reshape(-1),
                np.asarray(prediction.p4, dtype=float).reshape(-1),
            ]
        )
    array = np.asarray(prediction, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.reshape(-1)


@dataclass(slots=True)
class BaseSampler:
    """Shared Gaussian-likelihood plumbing for inference backends."""

    data: np.ndarray
    model: Any
    covariance: np.ndarray
    priors: ParameterCollection | Mapping[str, Any] | None = None
    parameter_names: Sequence[str] | None = None
    model_params: ParameterCollection | None = field(init=False, repr=False, default=None)
    _covariance_inv: np.ndarray = field(init=False, repr=False)
    _loglike_norm: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float).reshape(-1)
        self.covariance = np.asarray(self.covariance, dtype=float)
        if self.covariance.ndim != 2 or self.covariance.shape[0] != self.covariance.shape[1]:
            raise ValueError("covariance must be a square matrix.")
        if self.covariance.shape[0] != self.data.size:
            raise ValueError("data and covariance dimensions do not match.")

        self.model_params = _resolve_model_params(self.model)
        explicit_priors = _as_parameter_collection(self.priors)
        self.priors = self._merge_priors(explicit_priors)

        if self.parameter_names is None:
            self.parameter_names = _resolve_default_parameter_names(self.model, self.priors)
        else:
            self.parameter_names = tuple(str(name) for name in self.parameter_names)

        missing = [name for name in self.parameter_names if name not in self.priors]
        if missing:
            raise ValueError(f"Missing prior metadata for parameters: {', '.join(missing)}.")

        self._covariance_inv = np.linalg.inv(self.covariance)
        sign, logdet = np.linalg.slogdet(self.covariance)
        if sign <= 0:
            raise ValueError("covariance must be positive definite.")
        self._loglike_norm = -0.5 * (self.data.size * np.log(2.0 * np.pi) + logdet)

    def _merge_priors(self, priors: ParameterCollection | None) -> ParameterCollection:
        if self.model_params is None and priors is None:
            raise ValueError("The model must expose a ParameterCollection via `.params`, or priors must be provided explicitly.")
        if self.model_params is None:
            assert priors is not None
            return priors
        if priors is None:
            return self.model_params
        return ParameterCollection.combine(self.model_params, priors)

    @property
    def ndim(self) -> int:
        return len(self.parameter_names)

    def vector_to_params(self, theta: Sequence[float] | np.ndarray) -> dict[str, float]:
        values = np.asarray(theta, dtype=float).reshape(-1)
        if values.size != self.ndim:
            raise ValueError(f"Expected {self.ndim} sampled parameters, got {values.size}.")
        return {name: float(value) for name, value in zip(self.parameter_names, values, strict=True)}

    def predict_vector(self, theta: Sequence[float] | np.ndarray) -> np.ndarray:
        params = self.vector_to_params(theta)
        prediction = self.model.predict(params) if hasattr(self.model, "predict") else self.model(params)
        vector = flatten_prediction(prediction)
        if vector.size != self.data.size:
            raise ValueError(f"Model prediction size {vector.size} does not match data size {self.data.size}.")
        return vector

    def residual(self, theta: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.data - self.predict_vector(theta)

    def log_likelihood(self, theta: Sequence[float] | np.ndarray) -> float:
        delta = self.residual(theta)
        chi2 = float(delta @ self._covariance_inv @ delta)
        return float(self._loglike_norm - 0.5 * chi2)
