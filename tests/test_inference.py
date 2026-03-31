from __future__ import annotations

import numpy as np
import pytest

from jaxpt import BaseSampler, PocoMCSampler, TaylorEmulator
from jaxpt.parameter import ParameterCollection
from jaxpt.reference.classpt import MultipolePrediction


def test_base_sampler_analytic_marginalization_matches_manual_gaussian_result() -> None:
    class ToyLinearModel:
        def __init__(self) -> None:
            self.params = ParameterCollection(
                {
                    "x": {"value": 0.0, "prior": {"type": "gaussian", "mean": 0.0, "sigma": 1.0}},
                    "a": {"value": 0.0, "marginalized": True, "prior": {"type": "gaussian", "mean": 1.0, "sigma": 2.0}},
                }
            )

        def predict(self, params):
            return np.asarray([2.0 * params["x"]], dtype=float)

        def marginalized_design_matrix(self, params, *, parameter_names=None):
            del params
            assert parameter_names == ("a",)
            return np.asarray([[3.0]], dtype=float)

    sampler = BaseSampler(
        data=np.asarray([5.0]),
        model=ToyLinearModel(),
        covariance=np.asarray([[4.0]]),
    )

    sigma_y2 = 4.0
    sigma_a2 = 4.0
    expected_mean = 2.0 * 0.5 + 3.0 * 1.0
    expected_variance = sigma_y2 + 9.0 * sigma_a2
    expected = -0.5 * (np.log(2.0 * np.pi * expected_variance) + (5.0 - expected_mean) ** 2 / expected_variance)
    assert np.isclose(sampler.log_likelihood(np.asarray([0.5])), expected)


def test_base_sampler_analytic_marginalization_supports_flat_prior() -> None:
    class ToyLinearModel:
        def __init__(self) -> None:
            self.params = ParameterCollection(
                {
                    "x": {"value": 0.0, "prior": {"type": "gaussian", "mean": 0.0, "sigma": 1.0}},
                    "a": {"value": 0.0, "marginalized": True, "prior": {"type": "flat", "min": 0.0, "max": 10.0}},
                }
            )

        def predict(self, params):
            return np.asarray([params["x"]], dtype=float)

        def marginalized_design_matrix(self, params, *, parameter_names=None):
            del params
            assert parameter_names == ("a",)
            return np.asarray([[2.0]], dtype=float)

    sampler = BaseSampler(
        data=np.asarray([1.0]),
        model=ToyLinearModel(),
        covariance=np.asarray([[9.0]]),
    )

    expected = -np.log(2.0)
    assert np.isclose(sampler.log_likelihood(np.asarray([1.0])), expected)


def test_base_sampler_rejects_missing_design_matrix_for_marginalized_parameters() -> None:
    class ToyModel:
        def __init__(self) -> None:
            self.params = ParameterCollection(
                {
                    "x": {"value": 0.0, "prior": {"type": "gaussian", "mean": 0.0, "sigma": 1.0}},
                    "a": {"value": 0.0, "marginalized": True},
                }
            )

        def predict(self, params):
            return np.asarray([params["x"]], dtype=float)

    sampler = BaseSampler(
        data=np.asarray([0.0]),
        model=ToyModel(),
        covariance=np.eye(1),
    )

    with pytest.raises(ValueError, match="marginalized_design_matrix"):
        sampler.log_likelihood(np.asarray([0.0]))


def test_base_sampler_default_parameter_names_exclude_marginalized_parameters() -> None:
    class ToyModel:
        def __init__(self) -> None:
            self.params = ParameterCollection(
                {
                    "x": {"value": 0.0, "prior": {"type": "gaussian", "mean": 0.0, "sigma": 1.0}},
                    "a": {"value": 0.0, "marginalized": True},
                }
            )

        def predict(self, params):
            return np.asarray([params["x"]], dtype=float)

        def marginalized_design_matrix(self, params, *, parameter_names=None):
            del params
            assert parameter_names == ("a",)
            return np.asarray([[1.0]], dtype=float)

    sampler = BaseSampler(
        data=np.asarray([0.0]),
        model=ToyModel(),
        covariance=np.eye(1),
    )

    assert sampler.parameter_names == ("x",)
    assert sampler.sampled_parameter_names == ("x",)
    assert sampler.marginalized_parameter_names == ("a",)


def test_base_sampler_gaussian_loglikelihood_matches_manual_result() -> None:
    class ToyModel:
        def __init__(self) -> None:
            self.params = ParameterCollection(
                {"x": {"value": 1.0, "prior": {"type": "gaussian", "mean": 1.0, "sigma": 0.5}}}
            )

        def predict(self, params):
            return np.asarray([params["x"]], dtype=float)

    sampler = BaseSampler(
        data=np.asarray([1.5]),
        model=ToyModel(),
        covariance=np.asarray([[4.0]]),
    )

    expected = -0.5 * (np.log(2.0 * np.pi * 4.0) + (1.5 - 1.0) ** 2 / 4.0)
    assert np.isclose(sampler.log_likelihood(np.asarray([1.0])), expected)


def test_base_sampler_flattens_multipole_prediction_in_ell_major_order() -> None:
    class ToyMultipoleModel:
        def __init__(self) -> None:
            self.params = ParameterCollection(
                {"alpha": {"value": 0.0, "prior": {"type": "gaussian", "mean": 0.0, "sigma": 1.0}}}
            )
            self.param_names = ["alpha"]

        def predict(self, params):
            alpha = params["alpha"]
            return MultipolePrediction(
                k=np.asarray([0.1, 0.2]),
                p0=np.asarray([1.0 + alpha, 2.0 + alpha]),
                p2=np.asarray([3.0 + alpha, 4.0 + alpha]),
                p4=np.asarray([5.0 + alpha, 6.0 + alpha]),
            )

    sampler = BaseSampler(
        data=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        model=ToyMultipoleModel(),
        covariance=np.eye(6),
    )

    np.testing.assert_allclose(
        sampler.predict_vector(np.asarray([0.0])),
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_pocomc_sampler_rejects_unbounded_flat_prior() -> None:
    class ToyModel:
        def __init__(self) -> None:
            self.params = ParameterCollection({"x": {"value": 0.0, "prior": {"type": "flat", "min": None, "max": None}}})
            self.param_names = ["x"]

        def predict(self, params):
            return np.asarray([params["x"]], dtype=float)

    with pytest.raises(ValueError, match="bounded flat priors"):
        PocoMCSampler(data=np.asarray([0.0]), model=ToyModel(), covariance=np.eye(1))


def test_pocomc_sampler_explicit_priors_override_model_priors() -> None:
    pocomc = pytest.importorskip("pocomc")
    assert pocomc is not None

    class ToyModel:
        def __init__(self) -> None:
            self.params = ParameterCollection(
                {"x": {"value": 0.0, "prior": {"type": "gaussian", "mean": 0.0, "sigma": 1.0}}}
            )
            self.param_names = ["x"]

        def predict(self, params):
            return np.asarray([params["x"]], dtype=float)

    sampler = PocoMCSampler(
        data=np.asarray([0.0]),
        model=ToyModel(),
        covariance=np.eye(1),
        priors={"x": {"value": 0.0, "prior": {"type": "gaussian", "mean": 0.0, "sigma": 2.0}}},
    )

    logp = sampler.prior.logpdf(np.asarray([[1.0]]))
    expected = -0.5 * np.log(2.0 * np.pi * 4.0) - 0.5 * (1.0**2) / 4.0
    np.testing.assert_allclose(logp, np.asarray([expected]), rtol=1e-12, atol=1e-12)


def test_pocomc_sampler_runs_tiny_smoke_test() -> None:
    pytest.importorskip("pocomc")

    class ToyModel:
        def __init__(self) -> None:
            self.params = ParameterCollection(
                {"x": {"value": 0.0, "prior": {"type": "gaussian", "mean": 0.0, "sigma": 1.0}}}
            )
            self.param_names = ["x"]

        def predict(self, params):
            return np.asarray([params["x"]], dtype=float)

    sampler = PocoMCSampler(
        data=np.asarray([0.0]),
        model=ToyModel(),
        covariance=np.eye(1),
        sampler_kwargs={"random_state": 0, "train_config": {"epochs": 1}, "n_active": 8, "n_effective": 16},
        run_kwargs={"n_total": 16, "n_evidence": 0, "progress": False},
    )
    sampler.run()
    posterior = sampler.posterior()

    assert posterior["samples"].ndim == 2
    assert posterior["samples"].shape[1] == 1
    assert posterior["parameter_names"] == ("x",)


def test_taylor_emulator_accepts_loga_alias_for_old_as_parameter() -> None:
    class ToyTheory:
        def __call__(self, params):
            return np.asarray([params["A_s"]], dtype=float)

    emulator = TaylorEmulator(
        theory_fn=ToyTheory(),
        fiducial={"A_s": 2.089e-9},
        order=1,
        step_sizes={"A_s": 1.0e-11},
        param_names=["A_s"],
        valid_param_names=["A_s"],
    ).build()

    predicted = emulator.predict({"logA": np.log(1.0e10 * 2.2e-9)})
    np.testing.assert_allclose(predicted, np.asarray([2.2e-9]), rtol=0.0, atol=1e-15)
