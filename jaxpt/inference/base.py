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


def _as_design_matrix(matrix: Any, *, n_data: int, names: tuple[str, ...]) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim == 1:
        if len(names) != 1:
            raise ValueError("Marginalized design matrix must be two-dimensional when multiple parameters are marginalized.")
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("Marginalized design matrix must be a two-dimensional array.")
    if array.shape[0] != n_data:
        raise ValueError(f"Marginalized design matrix has {array.shape[0]} rows, expected {n_data}.")
    if array.shape[1] != len(names):
        raise ValueError(f"Marginalized design matrix has {array.shape[1]} columns, expected {len(names)}.")
    return array


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
    _sampled_parameter_names: tuple[str, ...] = field(init=False, repr=False, default=())
    _marginalized_parameter_names: tuple[str, ...] = field(init=False, repr=False, default=())
    _marginalized_baseline: np.ndarray = field(init=False, repr=False, default_factory=lambda: np.zeros(0, dtype=float))
    _marginalized_prior_mean_shift: np.ndarray = field(init=False, repr=False, default_factory=lambda: np.zeros(0, dtype=float))
    _marginalized_prior_precision: np.ndarray = field(init=False, repr=False, default_factory=lambda: np.zeros((0, 0), dtype=float))
    _marginalized_prior_logdet_precision: float = field(init=False, repr=False, default=0.0)
    _marginalized_has_gaussian_prior: bool = field(init=False, repr=False, default=False)

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

        invalid_sampled = [name for name in self.parameter_names if self.priors[name].fixed or self.priors[name].marginalized]
        if invalid_sampled:
            raise ValueError(
                "Sampled parameters must be non-fixed and non-marginalized. Invalid: " + ", ".join(invalid_sampled)
            )

        self._covariance_inv = np.linalg.inv(self.covariance)
        sign, logdet = np.linalg.slogdet(self.covariance)
        if sign <= 0:
            raise ValueError("covariance must be positive definite.")
        self._loglike_norm = -0.5 * (self.data.size * np.log(2.0 * np.pi) + logdet)
        self._sampled_parameter_names = tuple(self.parameter_names)
        self._initialize_marginalized_state()

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

    @property
    def sampled_parameter_names(self) -> tuple[str, ...]:
        return self._sampled_parameter_names

    @property
    def marginalized_parameter_names(self) -> tuple[str, ...]:
        return self._marginalized_parameter_names

    def _initialize_marginalized_state(self) -> None:
        names = tuple(name for name, parameter in self.priors.items() if parameter.marginalized)
        self._marginalized_parameter_names = names
        if not names:
            self._marginalized_baseline = np.zeros(0, dtype=float)
            self._marginalized_prior_mean_shift = np.zeros(0, dtype=float)
            self._marginalized_prior_precision = np.zeros((0, 0), dtype=float)
            self._marginalized_prior_logdet_precision = 0.0
            self._marginalized_has_gaussian_prior = False
            return

        baseline = np.asarray([float(self.priors[name].value) for name in names], dtype=float)
        mean_shift = np.zeros(len(names), dtype=float)
        precision = np.zeros((len(names), len(names)), dtype=float)
        has_gaussian_prior = False
        for index, name in enumerate(names):
            prior = self.priors[name].prior
            if prior is None:
                continue
            prior_type = str(prior.get("type", "flat")).lower()
            if prior_type == "gaussian":
                sigma = prior.get("sigma")
                if sigma is None:
                    raise ValueError(f"Marginalized Gaussian prior for '{name}' requires 'sigma'.")
                variance = float(sigma) ** 2
                if variance <= 0.0:
                    raise ValueError(f"Marginalized Gaussian prior for '{name}' must have positive sigma.")
                mean = float(prior.get("mean", baseline[index]))
                precision[index, index] = 1.0 / variance
                mean_shift[index] = mean - baseline[index]
                has_gaussian_prior = True
            elif prior_type == "flat":
                continue
            else:
                raise ValueError(f"Unsupported marginalized prior type '{prior_type}' for parameter '{name}'.")

        self._marginalized_baseline = baseline
        self._marginalized_prior_mean_shift = mean_shift
        self._marginalized_prior_precision = precision
        self._marginalized_has_gaussian_prior = has_gaussian_prior
        if has_gaussian_prior:
            positive = np.diag(precision) > 0.0
            precision_sign, precision_logdet = np.linalg.slogdet(precision[np.ix_(positive, positive)])
            if precision_sign <= 0:
                raise ValueError("Marginalized Gaussian prior precision must be positive definite.")
            self._marginalized_prior_logdet_precision = float(precision_logdet)
        else:
            self._marginalized_prior_logdet_precision = 0.0

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

    def _marginalized_design_matrix(self, theta: Sequence[float] | np.ndarray) -> np.ndarray:
        if not self._marginalized_parameter_names:
            return np.zeros((self.data.size, 0), dtype=float)
        params = self.vector_to_params(theta)
        if not hasattr(self.model, "marginalized_design_matrix"):
            raise ValueError(
                "Model parameters are marked marginalized, but the model does not expose "
                "`marginalized_design_matrix(parameters, parameter_names=...)`."
            )
        matrix = self.model.marginalized_design_matrix(
            params,
            parameter_names=self._marginalized_parameter_names,
        )
        return _as_design_matrix(matrix, n_data=self.data.size, names=self._marginalized_parameter_names)

    def log_likelihood(self, theta: Sequence[float] | np.ndarray) -> float:
        delta = self.residual(theta)
        if not self._marginalized_parameter_names:
            chi2 = float(delta @ self._covariance_inv @ delta)
            return float(self._loglike_norm - 0.5 * chi2)

        design = self._marginalized_design_matrix(theta)
        fisher = design.T @ self._covariance_inv @ design
        system = fisher + self._marginalized_prior_precision
        rhs = design.T @ self._covariance_inv @ delta + self._marginalized_prior_precision @ self._marginalized_prior_mean_shift
        quadratic = float(
            delta @ self._covariance_inv @ delta
            + self._marginalized_prior_mean_shift @ self._marginalized_prior_precision @ self._marginalized_prior_mean_shift
        )

        sign, logdet = np.linalg.slogdet(system)
        if sign <= 0:
            raise ValueError("Marginalized design matrix leads to a non-positive-definite normal matrix.")
        solved = np.linalg.solve(system, rhs)
        if self._marginalized_has_gaussian_prior:
            loglike = self._loglike_norm + 0.5 * self._marginalized_prior_logdet_precision
        else:
            loglike = self._loglike_norm + 0.5 * len(self._marginalized_parameter_names) * np.log(2.0 * np.pi)
        loglike += -0.5 * logdet - 0.5 * (quadratic - float(rhs @ solved))
        return float(loglike)
