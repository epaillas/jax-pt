from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..parameter import ParameterCollection
from .base import BaseSampler


def _import_pocomc():
    try:
        from pocomc import Prior
        from pocomc import Sampler
    except ImportError as exc:
        raise ImportError("PocoMCSampler requires the optional 'inference' dependencies: install `pocomc` and `scipy`.") from exc
    return Prior, Sampler


def _import_scipy_stats():
    try:
        from scipy.stats import norm, uniform
    except ImportError as exc:
        raise ImportError("PocoMCSampler requires the optional 'inference' dependencies: install `pocomc` and `scipy`.") from exc
    return norm, uniform


def _build_distribution(prior: Mapping[str, Any]) -> Any:
    norm, uniform = _import_scipy_stats()
    prior_type = str(prior.get("type", "")).lower()
    if prior_type == "gaussian":
        mean = prior.get("mean")
        sigma = prior.get("sigma")
        if mean is None or sigma is None:
            raise ValueError("Gaussian priors require 'mean' and 'sigma'.")
        return norm(loc=float(mean), scale=float(sigma))
    if prior_type == "flat":
        lower = prior.get("min")
        upper = prior.get("max")
        if lower is None or upper is None:
            raise ValueError("PocoMC requires bounded flat priors; both 'min' and 'max' must be set.")
        return uniform(loc=float(lower), scale=float(upper) - float(lower))
    raise ValueError(f"Unsupported prior type '{prior_type}'.")


@dataclass(slots=True)
class PocoMCSampler(BaseSampler):
    """PocoMC-backed sampler for Gaussian likelihoods."""

    sampler_kwargs: Mapping[str, Any] | None = None
    run_kwargs: Mapping[str, Any] | None = None
    vectorize: bool = False
    _prior: Any = field(init=False, repr=False, default=None)
    _sampler: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        BaseSampler.__post_init__(self)
        self.sampler_kwargs = {} if self.sampler_kwargs is None else dict(self.sampler_kwargs)
        self.run_kwargs = {} if self.run_kwargs is None else dict(self.run_kwargs)
        self._prior = self._build_prior()

    def _build_prior(self) -> Any:
        Prior, _ = _import_pocomc()
        distributions = []
        for name in self.parameter_names:
            parameter = self.priors[name]
            if parameter.prior is None:
                raise ValueError(f"Parameter '{name}' is missing prior metadata.")
            distributions.append(_build_distribution(parameter.prior))
        return Prior(distributions)

    def build_sampler(self) -> Any:
        _, Sampler = _import_pocomc()
        kwargs = dict(self.sampler_kwargs)
        kwargs.setdefault("vectorize", self.vectorize)
        self._sampler = Sampler(prior=self._prior, likelihood=self._log_likelihood_impl, **kwargs)
        return self._sampler

    def _log_likelihood_impl(self, theta: np.ndarray) -> np.ndarray | float:
        values = np.asarray(theta, dtype=float)
        if self.vectorize:
            if values.ndim != 2:
                raise ValueError("Vectorized PocoMC likelihood expects a two-dimensional parameter array.")
            return np.asarray([self.log_likelihood(row) for row in values], dtype=float)
        return float(self.log_likelihood(values))

    @property
    def prior(self) -> Any:
        return self._prior

    @property
    def sampler(self) -> Any:
        if self._sampler is None:
            return self.build_sampler()
        return self._sampler

    def run(self, **kwargs: Any) -> Any:
        sampler = self.sampler
        run_kwargs = dict(self.run_kwargs)
        run_kwargs.update(kwargs)
        sampler.run(**run_kwargs)
        return sampler

    def posterior(self, **kwargs: Any) -> dict[str, Any]:
        samples, weights, logl, logp = self.sampler.posterior(**kwargs)
        return {
            "samples": np.asarray(samples, dtype=float),
            "weights": np.asarray(weights, dtype=float),
            "logl": np.asarray(logl, dtype=float),
            "logp": np.asarray(logp, dtype=float),
            "parameter_names": tuple(self.parameter_names),
        }
