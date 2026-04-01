from __future__ import annotations

import numpy as np
import pytest

from jaxpt import LinearPowerInput, PTSettings
from jaxpt.theories import (
    PowerSpectrumTemplate,
    QuantileGalaxyPowerSpectrumMultipolesTheory,
    load_density_split_galaxy_power_spectrum_multipoles_parameters,
)


def _linear_input() -> LinearPowerInput:
    return LinearPowerInput(
        k=np.logspace(-3.0, 0.0, 128),
        pk_linear=np.linspace(2.0e4, 5.0e2, 128),
        z=0.5,
        growth_factor=0.76,
        growth_rate=0.81,
        h=0.67,
    )


def test_density_split_parameter_loader_preserves_defaults_and_priors() -> None:
    params = load_density_split_galaxy_power_spectrum_multipoles_parameters()

    assert params["b1"].value == 2.0
    assert params["bq1"].value == -1.6
    assert params["bq5"].value == 1.6
    assert params["beta1"].value == -0.8
    assert params["beta5"].value == 0.8
    assert params["bq3"].prior == {"type": "flat", "min": -4.0, "max": 4.0}
    assert params["beta3"].prior == {"type": "flat", "min": -3.0, "max": 3.0}
    assert params["bq1"].marginalized is False
    assert params["beta5"].marginalized is False


def test_density_split_theory_without_arguments_uses_default_parameter_configuration() -> None:
    theory = QuantileGalaxyPowerSpectrumMultipolesTheory(
        template=PowerSpectrumTemplate.from_linear_input(_linear_input(), settings=PTSettings(ir_resummation=False)),
        k=np.linspace(0.02, 0.18, 12),
    )

    implicit = theory()
    explicit = theory(theory.nuisance_parameters.defaults_dict())

    np.testing.assert_allclose(implicit, explicit, rtol=1e-12, atol=1e-12)


def test_density_split_theory_returns_quantile_ell_k_stack() -> None:
    theory = QuantileGalaxyPowerSpectrumMultipolesTheory(
        template=PowerSpectrumTemplate.from_linear_input(_linear_input(), settings=PTSettings(ir_resummation=False)),
        k=np.linspace(0.02, 0.18, 7),
    )

    values = theory(ells=(0, 2), quantiles=(1, 3, 5))

    assert values.shape == (3, 2, 7)
    assert not np.allclose(values[0], values[1])
    assert not np.allclose(values[1], values[2])


def test_density_split_smoothing_suppresses_high_k_predictions() -> None:
    template = PowerSpectrumTemplate.from_linear_input(_linear_input(), settings=PTSettings(ir_resummation=False))
    theory_no_smoothing = QuantileGalaxyPowerSpectrumMultipolesTheory(
        template=template,
        k=np.linspace(0.02, 0.18, 16),
        smoothing_radius_hmpc=1.0e-6,
    )
    theory_smoothed = QuantileGalaxyPowerSpectrumMultipolesTheory(
        template=template,
        k=np.linspace(0.02, 0.18, 16),
        smoothing_radius_hmpc=10.0,
    )

    unsmoothed = theory_no_smoothing(quantiles=(5,))
    smoothed = theory_smoothed(quantiles=(5,))

    assert np.all(np.abs(smoothed[0, :, -1]) < np.abs(unsmoothed[0, :, -1]))


def test_density_split_predict_quantiles_can_return_components() -> None:
    theory = QuantileGalaxyPowerSpectrumMultipolesTheory(
        template=PowerSpectrumTemplate.from_linear_input(_linear_input(), settings=PTSettings(ir_resummation=False)),
        k=np.linspace(0.02, 0.18, 5),
        return_components=True,
    )

    predictions = theory.predict_quantiles(quantiles=(2,), return_components=True)
    prediction = predictions[2]

    assert prediction.metadata["quantile"] == 2
    assert prediction.components is not None
    assert set(prediction.components) == {"density_density", "mixed_rsd", "rsd_rsd", "smoothing_kernel", "linear_power"}
def test_density_split_theory_rejects_unknown_parameters() -> None:
    theory = QuantileGalaxyPowerSpectrumMultipolesTheory(
        template=PowerSpectrumTemplate.from_linear_input(_linear_input(), settings=PTSettings(ir_resummation=False)),
        k=np.linspace(0.02, 0.18, 5),
    )

    with pytest.raises(ValueError, match="unexpected parameters"):
        theory({"not_a_parameter": 1.0})
