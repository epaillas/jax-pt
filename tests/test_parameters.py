from __future__ import annotations

import numpy as np

from jaxpt import LinearPowerInput, PTSettings, ParameterCollection, TaylorEmulator
from jaxpt.theories import (
    GalaxyPowerSpectrumMultipolesTheory,
    PowerSpectrumTemplate,
    load_galaxy_power_spectrum_multipoles_parameters,
    load_power_spectrum_template_parameters,
)


def test_parameter_yaml_loaders_preserve_values_and_statuses() -> None:
    template_params = load_power_spectrum_template_parameters()
    nuisance_params = load_galaxy_power_spectrum_multipoles_parameters()

    assert template_params["logA"].value == 3.0392705755684744
    assert template_params["omega_cdm"].value == 0.12
    assert template_params["Omega_k"].fixed is True
    assert template_params["w0_fld"].fixed is True
    assert nuisance_params["b1"].value == 2.0
    assert nuisance_params["cs0"].marginalized is True
    assert nuisance_params["cs2"].marginalized is True
    assert nuisance_params["cs4"].marginalized is True
    assert nuisance_params["Pshot"].marginalized is True
    assert template_params["omega_b"].prior == {"type": "gaussian", "mean": 0.02233, "sigma": 0.00036}
    assert template_params["logA"].prior == {"type": "flat", "min": 2.0, "max": 4.0}
    assert template_params["omega_cdm"].prior == {"type": "flat", "min": 0.05, "max": 0.2}
    assert template_params["h"].prior == {"type": "flat", "min": 0.4, "max": 1.0}
    assert template_params["m_ncdm"].prior == {"type": "flat", "min": 0.06, "max": 0.18}
    assert nuisance_params["cs0"].prior == {"type": "flat", "min": None, "max": None}
    assert nuisance_params["cs2"].prior == {"type": "flat", "min": None, "max": None}
    assert nuisance_params["cs4"].prior == {"type": "flat", "min": None, "max": None}
    assert nuisance_params["Pshot"].prior == {"type": "flat", "min": 0.0, "max": 10000.0}


def test_parameter_update_changes_emulation_status() -> None:
    params = load_power_spectrum_template_parameters()

    assert params["omega_cdm"].emulated is True
    params["omega_cdm"].update(marginalized=True)
    assert params["omega_cdm"].marginalized is True
    assert params["omega_cdm"].emulated is False


def test_theory_exposes_merged_parameter_collection() -> None:
    template = PowerSpectrumTemplate({"omega_cdm": 0.13}, z=0.5, settings=PTSettings())
    theory = GalaxyPowerSpectrumMultipolesTheory(template=template, k=np.linspace(0.02, 0.1, 4))

    assert "omega_cdm" in theory.params
    assert "logA" in theory.params
    assert "b1" in theory.params
    assert theory.params["omega_cdm"].value == 0.13
    assert theory.params["Omega_k"].fixed is True

    theory.params["omega_cdm"].update(marginalized=True)
    assert template.params["omega_cdm"].marginalized is True


def test_template_normalizes_amplitude_aliases_to_public_loga() -> None:
    template_from_as = PowerSpectrumTemplate({"A_s": 2.089e-9}, z=0.5, settings=PTSettings())
    template_from_loga = PowerSpectrumTemplate({"logA": np.log(1.0e10 * 2.089e-9)}, z=0.5, settings=PTSettings())

    assert "logA" in template_from_as.params
    assert "A_s" not in template_from_as.params
    np.testing.assert_allclose(template_from_as.params["logA"].value, template_from_loga.params["logA"].value, rtol=0.0, atol=1e-12)


def test_theory_without_arguments_uses_default_parameter_configuration() -> None:
    linear_input = LinearPowerInput(
        k=np.logspace(-3.0, 0.0, 64),
        pk_linear=np.linspace(2.0e4, 5.0e2, 64),
        z=0.5,
        growth_factor=0.76,
        growth_rate=0.81,
        h=0.67,
    )
    theory = GalaxyPowerSpectrumMultipolesTheory(
        template=PowerSpectrumTemplate.from_linear_input(linear_input, settings=PTSettings(ir_resummation=False)),
        k=np.linspace(0.02, 0.18, 12),
    )

    implicit = theory()
    explicit = theory(theory.nuisance_parameters.defaults_dict())

    np.testing.assert_allclose(implicit.p0, explicit.p0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(implicit.p2, explicit.p2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(implicit.p4, explicit.p4, rtol=1e-12, atol=1e-12)


def test_taylor_emulator_defaults_to_emulated_parameter_subset() -> None:
    class ToyTheory:
        def __init__(self) -> None:
            self.params = ParameterCollection(
                {
                    "x": {"value": 1.0},
                    "y": {"value": 2.0, "marginalized": True},
                    "z": {"value": 3.0, "fixed": True},
                }
            )

        def __call__(self, params):
            return np.asarray([params["x"] + params["y"] + params["z"]], dtype=float)

    theory = ToyTheory()
    emulator = TaylorEmulator(
        theory,
        fiducial=theory.params.defaults_dict(),
        order=1,
        step_sizes={"x": 0.1, "y": 0.1, "z": 0.1},
    )

    assert emulator.param_names == ["x"]
