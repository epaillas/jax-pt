import numpy as np
import pytest
from classy import Class
from cosmoprimo import Cosmology
from unittest.mock import patch

from jaxpt import (
    ClassPTGalaxyPowerSpectrumMultipolesTheory,
    EFTBiasParams,
    GalaxyPowerSpectrumMultipolesTheory,
    LinearPowerInput,
    PTSettings,
    PowerSpectrumTemplate,
    build_linear_input_from_classy,
    build_linear_input_from_cosmoprimo,
    compute_basis,
    galaxy_multipoles,
    predict_galaxy_multipoles,
)
from jaxpt.cosmology import (
    BaseCosmologyProvider,
    ResolvedCosmologyState,
    build_classpt_native_grid_parity_linear_input_from_classy,
    build_classpt_parity_linear_input_from_classy,
    prepare_native_fftlog_input,
)


def test_build_linear_input_from_classy() -> None:
    z = 0.5
    k = np.logspace(-3, -1, 8)

    cosmo = Class()
    cosmo.set(
        {
            "A_s": 2.089e-9,
            "n_s": 0.9649,
            "tau_reio": 0.052,
            "omega_b": 0.02237,
            "omega_cdm": 0.12,
            "h": 0.6736,
            "YHe": 0.2425,
            "N_ur": 2.0328,
            "N_ncdm": 1,
            "m_ncdm": 0.06,
            "z_pk": z,
            "output": "mPk",
        }
    )
    cosmo.compute()

    linear_input = build_linear_input_from_classy(cosmo, z=z, k=k)

    assert linear_input.k.shape == k.shape
    assert linear_input.pk_linear.shape == k.shape
    assert linear_input.transfer_linear is not None
    assert linear_input.transfer_linear.shape == k.shape
    assert np.all(np.isfinite(linear_input.transfer_linear))
    assert linear_input.h > 0.0
    assert np.isfinite(linear_input.growth_rate)
    assert linear_input.metadata["linear_pk_source"] == "classpt_pk_lin"
    assert "classy_cosmo" not in linear_input.metadata


def test_build_linear_input_from_cosmoprimo_matches_classy_builder() -> None:
    z = 0.5
    k = np.logspace(-3, -1, 8)

    classy_cosmo = Class()
    classy_cosmo.set(
        {
            "A_s": 2.089e-9,
            "n_s": 0.9649,
            "tau_reio": 0.052,
            "omega_b": 0.02237,
            "omega_cdm": 0.12,
            "h": 0.6736,
            "YHe": 0.2425,
            "N_ur": 2.0328,
            "N_ncdm": 1,
            "m_ncdm": 0.06,
            "z_pk": z,
            "output": "mPk",
        }
    )
    classy_cosmo.compute()

    cosmoprimo_cosmo = Cosmology(
        engine="class",
        h=0.6736,
        omega_b=0.02237,
        omega_cdm=0.12,
        n_s=0.9649,
        tau_reio=0.052,
        m_ncdm=0.06,
        YHe=0.2425,
        N_ur=2.0328,
        logA=np.log(1.0e10 * 2.089e-9),
        z_pk=[z],
        kmax_pk=10.0,
    )

    expected = build_linear_input_from_classy(classy_cosmo, z=z, k=k)
    linear_input = build_linear_input_from_cosmoprimo(cosmoprimo_cosmo, z=z, k=k)

    np.testing.assert_allclose(linear_input.k, expected.k)
    np.testing.assert_allclose(linear_input.pk_linear, expected.pk_linear, rtol=1.0e-2, atol=0.0)
    np.testing.assert_allclose(linear_input.growth_factor, expected.growth_factor, rtol=5.0e-4, atol=0.0)
    np.testing.assert_allclose(linear_input.growth_rate, expected.growth_rate, rtol=5.0e-4, atol=0.0)
    assert linear_input.transfer_linear is not None
    assert np.all(np.isfinite(linear_input.transfer_linear))
    assert linear_input.metadata["source"] == "cosmoprimo"
    assert linear_input.metadata["engine"] == "class"
    assert linear_input.metadata["linear_pk_source"] == "fourier_delta_cb"


def test_build_linear_input_from_cosmoprimo_requires_engine() -> None:
    cosmo = Cosmology(h=0.6736, omega_b=0.02237, omega_cdm=0.12, n_s=0.9649, logA=3.044)

    with pytest.raises(ValueError, match="attached engine"):
        build_linear_input_from_cosmoprimo(cosmo, z=0.5, k=np.logspace(-3, -1, 8))


def test_build_classpt_parity_linear_input_from_classy() -> None:
    z = 0.5
    k = np.logspace(-3, -1, 8)

    cosmo = Class()
    cosmo.set(
        {
            "A_s": 2.089e-9,
            "n_s": 0.9649,
            "tau_reio": 0.052,
            "omega_b": 0.02237,
            "omega_cdm": 0.12,
            "h": 0.6736,
            "YHe": 0.2425,
            "N_ur": 2.0328,
            "N_ncdm": 1,
            "m_ncdm": 0.06,
            "z_pk": z,
            "output": "mPk",
            "non linear": "PT",
            "IR resummation": "No",
            "Bias tracers": "Yes",
            "cb": "Yes",
            "RSD": "Yes",
        }
    )
    cosmo.compute()

    linear_input = build_classpt_parity_linear_input_from_classy(cosmo, z=z, k=k)
    expected_pk = np.asarray([np.asarray(cosmo.pk(float(ki), z), dtype=float)[14] for ki in k], dtype=float)

    np.testing.assert_allclose(linear_input.pk_linear, expected_pk)
    assert linear_input.transfer_linear is not None
    assert linear_input.metadata["linear_pk_source"] == "classpt_internal_tree"


def test_build_classpt_native_grid_parity_linear_input_from_classy() -> None:
    z = 0.5
    settings = PTSettings(ir_resummation=False)

    cosmo = Class()
    cosmo.set(
        {
            "A_s": 2.089e-9,
            "n_s": 0.9649,
            "tau_reio": 0.052,
            "omega_b": 0.02237,
            "omega_cdm": 0.12,
            "h": 0.6736,
            "YHe": 0.2425,
            "N_ur": 2.0328,
            "N_ncdm": 1,
            "m_ncdm": 0.06,
            "z_pk": z,
            "output": "mTk,mPk",
            "non linear": "PT",
            "IR resummation": "No",
            "Bias tracers": "Yes",
            "cb": "Yes",
            "RSD": "Yes",
        }
    )
    cosmo.compute()

    linear_input = build_classpt_native_grid_parity_linear_input_from_classy(cosmo, z=z, settings=settings)

    assert linear_input.k.shape == (settings.fftlog_n,)
    assert linear_input.transfer_linear is not None
    assert linear_input.metadata["linear_pk_source"] == "classpt_internal_tree"
    assert linear_input.metadata["support_grid"] == "native_fftlog_kdisc"
    assert linear_input.metadata["transfer_source"] == "classy_phi_scaled"


def test_prepare_native_fftlog_input_aligns_to_internal_grid() -> None:
    k = np.logspace(-4, 0, 64)
    linear_input = LinearPowerInput(
        k=k,
        pk_linear=2.0e4 * (k / 0.1) ** -1.1 * np.exp(-k / 0.5),
        transfer_linear=3.0e2 * (k / 0.1) ** -0.55 * np.exp(-k / 0.8),
        pk_nowiggle=1.9e4 * (k / 0.1) ** -1.1 * np.exp(-k / 0.5),
        z=0.5,
        growth_factor=0.75,
        growth_rate=0.8,
        h=0.7,
    )

    fftlog_input = prepare_native_fftlog_input(linear_input, PTSettings(ir_resummation=False))

    assert fftlog_input.kdisc.shape == (PTSettings().fftlog_n,)
    assert fftlog_input.pdisc.shape == fftlog_input.kdisc.shape
    assert fftlog_input.tdisc.shape == fftlog_input.kdisc.shape
    assert fftlog_input.pnw is not None
    assert fftlog_input.tnw is not None
    assert fftlog_input.pw is not None
    assert fftlog_input.tw is not None
    assert fftlog_input.metadata["fftlog_grid_source"] == "native_kdisc"
    assert fftlog_input.metadata["fftlog_input_aligned"] is False


def test_linear_power_input_validates_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        LinearPowerInput(
            k=np.array([0.1, 0.2]),
            pk_linear=np.array([1.0]),
            z=0.5,
            growth_factor=1.0,
            growth_rate=0.8,
            h=0.7,
        )

    with pytest.raises(ValueError, match="metadata\\['field'\\]"):
        LinearPowerInput(
            k=np.array([0.1, 0.2]),
            pk_linear=np.array([1.0, 1.0]),
            z=0.5,
            growth_factor=1.0,
            growth_rate=0.8,
            h=0.7,
            metadata={"field": "baryons"},
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        LinearPowerInput(
            k=np.array([0.2, 0.1]),
            pk_linear=np.array([1.0, 1.0]),
            z=0.5,
            growth_factor=1.0,
            growth_rate=0.8,
            h=0.7,
        )


def test_power_spectrum_template_from_linear_input_preserves_state() -> None:
    linear_input = LinearPowerInput(
        k=np.logspace(-3, -1, 8),
        pk_linear=np.linspace(100.0, 200.0, 8),
        z=0.5,
        growth_factor=0.75,
        growth_rate=0.8,
        h=0.7,
    )
    settings = PTSettings(loop_order="tree", ir_resummation=False)

    template = PowerSpectrumTemplate.from_linear_input(
        linear_input,
        settings=settings,
        metadata={"name": "test-template"},
    )

    assert template.linear_input is linear_input
    assert template.settings is settings
    assert template.z == linear_input.z
    assert template.metadata["name"] == "test-template"


def test_power_spectrum_template_constructor_from_classy_matches_existing_builder() -> None:
    z = 0.5
    k = np.logspace(-3, -1, 8)
    settings = PTSettings(ir_resummation=False)

    cosmo = Class()
    cosmo.set(
        {
            "A_s": 2.089e-9,
            "n_s": 0.9649,
            "tau_reio": 0.052,
            "omega_b": 0.02237,
            "omega_cdm": 0.12,
            "h": 0.6736,
            "YHe": 0.2425,
            "N_ur": 2.0328,
            "N_ncdm": 1,
            "m_ncdm": 0.06,
            "z_pk": z,
            "output": "mPk",
        }
    )
    cosmo.compute()

    expected = build_linear_input_from_classy(cosmo, z=z, k=k)
    template = PowerSpectrumTemplate(cosmo, z=z, k=k, settings=settings)

    np.testing.assert_allclose(template.linear_input.k, expected.k)
    np.testing.assert_allclose(template.linear_input.pk_linear, expected.pk_linear)
    np.testing.assert_allclose(template.linear_input.transfer_linear, expected.transfer_linear)
    assert template.settings is settings


def test_power_spectrum_template_constructor_from_cosmoprimo_matches_builder() -> None:
    z = 0.5
    k = np.logspace(-3, -1, 8)
    settings = PTSettings(ir_resummation=False)
    cosmo = Cosmology(
        engine="class",
        h=0.6736,
        omega_b=0.02237,
        omega_cdm=0.12,
        n_s=0.9649,
        tau_reio=0.052,
        m_ncdm=0.06,
        YHe=0.2425,
        N_ur=2.0328,
        logA=np.log(1.0e10 * 2.089e-9),
        z_pk=[z],
        kmax_pk=10.0,
    )

    expected = build_linear_input_from_cosmoprimo(cosmo, z=z, k=k)
    template = PowerSpectrumTemplate(cosmo, z=z, k=k, settings=settings)

    np.testing.assert_allclose(template.linear_input.k, expected.k)
    np.testing.assert_allclose(template.linear_input.pk_linear, expected.pk_linear)
    np.testing.assert_allclose(template.linear_input.transfer_linear, expected.transfer_linear)
    assert template.settings is settings


def test_power_spectrum_template_constructor_uses_default_support_grid_for_cosmology() -> None:
    z = 0.5
    settings = PTSettings(ir_resummation=False, integration_nk=64)

    cosmo = Class()
    cosmo.set(
        {
            "A_s": 2.089e-9,
            "n_s": 0.9649,
            "tau_reio": 0.052,
            "omega_b": 0.02237,
            "omega_cdm": 0.12,
            "h": 0.6736,
            "YHe": 0.2425,
            "N_ur": 2.0328,
            "N_ncdm": 1,
            "m_ncdm": 0.06,
            "z_pk": z,
            "output": "mPk",
        }
    )
    cosmo.compute()

    template = PowerSpectrumTemplate(cosmo, z=z, settings=settings)

    assert template.linear_input.k.shape == (64,)
    np.testing.assert_allclose(template.linear_input.k[[0, -1]], np.array([1.0e-5, 1.0e1]))


def test_power_spectrum_template_constructor_supports_classpt_native_grid_parity_recipe() -> None:
    z = 0.5
    settings = PTSettings(ir_resummation=False)

    cosmo = Class()
    cosmo.set(
        {
            "A_s": 2.089e-9,
            "n_s": 0.9649,
            "tau_reio": 0.052,
            "omega_b": 0.02237,
            "omega_cdm": 0.12,
            "h": 0.6736,
            "YHe": 0.2425,
            "N_ur": 2.0328,
            "N_ncdm": 1,
            "m_ncdm": 0.06,
            "z_pk": z,
            "output": "mTk,mPk",
            "non linear": "PT",
            "IR resummation": "No",
            "Bias tracers": "Yes",
            "cb": "Yes",
            "RSD": "Yes",
        }
    )
    cosmo.compute()

    expected = build_classpt_native_grid_parity_linear_input_from_classy(cosmo, z=z, settings=settings)
    template = PowerSpectrumTemplate(cosmo, z=z, settings=settings, input_recipe="classpt_native_grid_parity")

    np.testing.assert_allclose(template.linear_input.k, expected.k)
    np.testing.assert_allclose(template.linear_input.pk_linear, expected.pk_linear)
    assert template.linear_input.metadata["support_grid"] == "native_fftlog_kdisc"


def test_power_spectrum_template_rejects_missing_z_for_cosmology_source() -> None:
    cosmo = Cosmology(engine="class", h=0.6736, omega_b=0.02237, omega_cdm=0.12, n_s=0.9649, logA=3.044, z_pk=[0.5], kmax_pk=10.0)

    with pytest.raises(ValueError, match="requires z"):
        PowerSpectrumTemplate(cosmo)


def test_power_spectrum_template_rejects_unsupported_source() -> None:
    with pytest.raises(TypeError, match="supported cosmology object"):
        PowerSpectrumTemplate(object(), z=0.5)


def test_power_spectrum_template_rejects_inconsistent_linear_input_z() -> None:
    linear_input = LinearPowerInput(
        k=np.logspace(-3, -1, 8),
        pk_linear=np.linspace(100.0, 200.0, 8),
        z=0.5,
        growth_factor=0.75,
        growth_rate=0.8,
        h=0.7,
    )

    with pytest.raises(ValueError, match="must match LinearPowerInput.z"):
        PowerSpectrumTemplate(linear_input, z=1.0)


def test_galaxy_power_spectrum_multipoles_theory_matches_functional_path() -> None:
    k = np.logspace(-3, -1, 16)
    eval_k = np.linspace(0.01, 0.08, 9)
    linear_input = LinearPowerInput(
        k=k,
        pk_linear=np.linspace(100.0, 200.0, k.size),
        z=0.5,
        growth_factor=0.75,
        growth_rate=0.8,
        h=0.7,
    )
    settings = PTSettings(ir_resummation=False)
    params = {
        "b1": 2.0,
        "b2": -1.0,
        "bG2": 0.1,
        "bGamma3": -0.1,
        "cs0": 0.0,
        "cs2": 30.0,
        "cs4": 0.0,
        "Pshot": 3000.0,
        "b4": 10.0,
    }

    theory = GalaxyPowerSpectrumMultipolesTheory(
        template=PowerSpectrumTemplate.from_linear_input(linear_input, settings=settings),
        k=eval_k,
    )
    expected = galaxy_multipoles(compute_basis(linear_input, settings=settings, k=eval_k), EFTBiasParams(**params))
    prediction = theory(params)

    np.testing.assert_allclose(np.asarray(prediction.k), np.asarray(expected.k))
    np.testing.assert_allclose(np.asarray(prediction.p0), np.asarray(expected.p0))
    np.testing.assert_allclose(np.asarray(prediction.p2), np.asarray(expected.p2))
    np.testing.assert_allclose(np.asarray(prediction.p4), np.asarray(expected.p4))
    assert prediction.metadata["theory"] == "GalaxyPowerSpectrumMultipolesTheory"
    assert prediction.metadata["template"] == "PowerSpectrumTemplate"


def test_galaxy_power_spectrum_multipoles_theory_rejects_invalid_mapping() -> None:
    linear_input = LinearPowerInput(
        k=np.logspace(-3, -1, 8),
        pk_linear=np.linspace(100.0, 200.0, 8),
        z=0.5,
        growth_factor=0.75,
        growth_rate=0.8,
        h=0.7,
    )
    theory = GalaxyPowerSpectrumMultipolesTheory(
        template=PowerSpectrumTemplate.from_linear_input(linear_input, settings=PTSettings(ir_resummation=False)),
        k=np.linspace(0.01, 0.08, 6),
    )

    with pytest.raises(ValueError, match="missing required parameters"):
        theory({"b1": 2.0})

    with pytest.raises(ValueError, match="unexpected parameters"):
        theory(
            {
                "b1": 2.0,
                "b2": -1.0,
                "bG2": 0.1,
                "bGamma3": -0.1,
                "cs0": 0.0,
                "cs2": 30.0,
                "cs4": 0.0,
                "Pshot": 3000.0,
                "b4": 10.0,
                "extra": 1.0,
            }
        )


def test_fixed_linear_input_template_rejects_cosmology_overrides() -> None:
    linear_input = LinearPowerInput(
        k=np.logspace(-3, -1, 8),
        pk_linear=np.linspace(100.0, 200.0, 8),
        z=0.5,
        growth_factor=0.75,
        growth_rate=0.8,
        h=0.7,
    )
    theory = GalaxyPowerSpectrumMultipolesTheory(
        template=PowerSpectrumTemplate.from_linear_input(linear_input, settings=PTSettings(ir_resummation=False)),
        k=np.linspace(0.01, 0.08, 6),
    )

    with pytest.raises(ValueError, match="cannot accept cosmology overrides"):
        theory(
            b1=2.0,
            b2=-1.0,
            bG2=0.1,
            bGamma3=-0.1,
            cs0=0.0,
            cs2=30.0,
            cs4=0.0,
            Pshot=3000.0,
            b4=10.0,
            omega_cdm=0.13,
        )


def test_queryable_template_accepts_flat_nuisance_and_cosmology_kwargs() -> None:
    class FakeProvider(BaseCosmologyProvider):
        def __init__(self) -> None:
            super().__init__({"omega_cdm": 0.12})

        def build_cosmology(self, cosmology_params):
            raise NotImplementedError

        def build_linear_input(self, *, cosmology, z, k, settings, input_recipe):
            raise NotImplementedError

        def resolve(self, *, overrides, z, k, settings, input_recipe):
            omega_cdm = float(overrides.get("omega_cdm", self.fiducial_params["omega_cdm"]))
            support_k = np.logspace(-3, -1, 16) if k is None else np.asarray(k, dtype=float)
            linear_input = LinearPowerInput(
                k=support_k,
                pk_linear=(omega_cdm / 0.12) * np.linspace(100.0, 200.0, support_k.size),
                z=float(z),
                growth_factor=0.75,
                growth_rate=0.8,
                h=0.7,
            )
            cosmology_params = {"omega_cdm": omega_cdm}
            return ResolvedCosmologyState(
                cosmology={"omega_cdm": omega_cdm},
                linear_input=linear_input,
                cosmology_params=cosmology_params,
                query_key=(("omega_cdm", omega_cdm),),
            )

    template = PowerSpectrumTemplate({"omega_cdm": 0.12}, z=0.5, k=np.logspace(-3, -1, 16), settings=PTSettings(loop_order="tree", ir_resummation=False))
    template._cosmology_provider = FakeProvider()
    theory = GalaxyPowerSpectrumMultipolesTheory(template=template, k=np.linspace(0.01, 0.08, 9))

    prediction = theory(
        omega_cdm=0.13,
        b1=2.0,
        b2=-1.0,
        bG2=0.1,
        bGamma3=-0.1,
        cs0=0.0,
        cs2=30.0,
        cs4=0.0,
        Pshot=3000.0,
        b4=10.0,
    )

    assert prediction.p0.shape == theory.k.shape
    assert prediction.metadata["theory"] == "GalaxyPowerSpectrumMultipolesTheory"


def test_queryable_template_reuses_basis_for_nuisance_only_updates(monkeypatch) -> None:
    class FakeProvider(BaseCosmologyProvider):
        def __init__(self) -> None:
            super().__init__({"omega_cdm": 0.12})

        def build_cosmology(self, cosmology_params):
            raise NotImplementedError

        def build_linear_input(self, *, cosmology, z, k, settings, input_recipe):
            raise NotImplementedError

        def resolve(self, *, overrides, z, k, settings, input_recipe):
            omega_cdm = float(overrides.get("omega_cdm", self.fiducial_params["omega_cdm"]))
            support_k = np.logspace(-3, -1, 16)
            linear_input = LinearPowerInput(
                k=support_k,
                pk_linear=(omega_cdm / 0.12) * np.linspace(100.0, 200.0, support_k.size),
                z=float(z),
                growth_factor=0.75,
                growth_rate=0.8,
                h=0.7,
            )
            return ResolvedCosmologyState(
                cosmology={"omega_cdm": omega_cdm},
                linear_input=linear_input,
                cosmology_params={"omega_cdm": omega_cdm},
                query_key=(("omega_cdm", omega_cdm),),
            )

    template = PowerSpectrumTemplate({"omega_cdm": 0.12}, z=0.5, settings=PTSettings(loop_order="tree", ir_resummation=False))
    template._cosmology_provider = FakeProvider()
    theory = GalaxyPowerSpectrumMultipolesTheory(template=template, k=np.linspace(0.01, 0.08, 9))

    original_compute_basis = compute_basis
    calls = []

    def wrapped_compute_basis(linear_input, settings, k=None):
        calls.append(float(linear_input.pk_linear[0]))
        return original_compute_basis(linear_input, settings=settings, k=k)

    monkeypatch.setattr("jaxpt.theories.power_spectrum.compute_basis", wrapped_compute_basis)

    base_params = dict(b1=2.0, b2=-1.0, bG2=0.1, bGamma3=-0.1, cs0=0.0, cs2=30.0, cs4=0.0, Pshot=3000.0, b4=10.0)
    theory(omega_cdm=0.12, **base_params)
    theory(omega_cdm=0.12, **{**base_params, "b1": 2.2})
    theory(omega_cdm=0.13, **base_params)

    assert calls == [100.0, 100.0 * 0.13 / 0.12]


def test_native_tree_loop_order_matches_kaiser_prediction() -> None:
    k = np.logspace(-3, -1, 8)
    pk_linear = np.linspace(100.0, 200.0, k.size)
    linear_input = LinearPowerInput(
        k=k,
        pk_linear=pk_linear,
        z=0.5,
        growth_factor=0.75,
        growth_rate=0.8,
        h=0.7,
    )

    basis = compute_basis(linear_input, settings=PTSettings(loop_order="tree", ir_resummation=False))
    params = EFTBiasParams(b1=2.0, b2=0.0, bG2=0.0, bGamma3=0.0, cs0=0.0, cs2=0.0, cs4=0.0, Pshot=0.0, b4=0.0)
    prediction = galaxy_multipoles(basis, params)

    expected_p0 = (params.b1**2 + 2.0 * params.b1 * linear_input.growth_rate / 3.0 + linear_input.growth_rate**2 / 5.0) * pk_linear * linear_input.h**3
    expected_p2 = (4.0 * params.b1 * linear_input.growth_rate / 3.0 + 4.0 * linear_input.growth_rate**2 / 7.0) * pk_linear * linear_input.h**3
    expected_p4 = (8.0 * linear_input.growth_rate**2 / 35.0) * pk_linear * linear_input.h**3

    np.testing.assert_allclose(np.asarray(prediction.p0), expected_p0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(prediction.p2), expected_p2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(prediction.p4), expected_p4, rtol=1e-12, atol=1e-12)
    assert basis.metadata["approximation"] == "linear_kaiser"
    assert basis.metadata["kernel_source"] == "tree"
    assert basis.metadata["realspace_matter"] is False
    assert basis.metadata["realspace_bias"] is False


def test_native_backend_rejects_unknown_backend() -> None:
    linear_input = LinearPowerInput(
        k=np.logspace(-3, -2, 4),
        pk_linear=np.ones(4),
        z=0.5,
        growth_factor=1.0,
        growth_rate=0.8,
        h=0.7,
    )

    with pytest.raises(ValueError, match="settings.backend == 'native'"):
        compute_basis(linear_input, settings=PTSettings(backend="legacy", ir_resummation=False))


def test_native_backend_rejects_resummation_without_support() -> None:
    linear_input = LinearPowerInput(
        k=np.logspace(-3, -2, 4),
        pk_linear=np.ones(4),
        z=0.5,
        growth_factor=1.0,
        growth_rate=0.8,
        h=0.7,
    )

    with pytest.raises(NotImplementedError, match="IR-resummed"):
        compute_basis(linear_input, settings=PTSettings(ir_resummation=True))


def test_native_backend_applies_k_window() -> None:
    k = np.logspace(-3, -1, 8)
    linear_input = LinearPowerInput(
        k=k,
        pk_linear=np.linspace(100.0, 200.0, k.size),
        z=0.5,
        growth_factor=1.0,
        growth_rate=0.8,
        h=0.7,
    )

    basis = compute_basis(linear_input, settings=PTSettings(ir_resummation=False, kmin=1.0e-2, kmax=5.0e-2))

    assert np.all(np.asarray(basis.k) >= 1.0e-2)
    assert np.all(np.asarray(basis.k) <= 5.0e-2)


def test_native_backend_does_not_load_importlib_matrix_assets() -> None:
    linear_input = LinearPowerInput(
        k=np.logspace(-3, -1, 16),
        pk_linear=np.linspace(100.0, 200.0, 16),
        z=0.5,
        growth_factor=1.0,
        growth_rate=0.8,
        h=0.7,
    )

    with patch("importlib.resources.files", side_effect=AssertionError("matrix assets should not be loaded through importlib.resources")):
        basis = compute_basis(linear_input, settings=PTSettings(ir_resummation=False))

    assert np.all(np.isfinite(np.asarray(basis.components["real_loop_matter"])))


def test_compute_basis_projects_to_requested_k_without_resampling_input() -> None:
    linear_input = LinearPowerInput(
        k=np.logspace(-3, -1, 32),
        pk_linear=np.linspace(100.0, 200.0, 32),
        z=0.5,
        growth_factor=1.0,
        growth_rate=0.8,
        h=0.7,
    )
    output_k = np.linspace(0.01, 0.08, 11)

    basis = compute_basis(linear_input, settings=PTSettings(ir_resummation=False), k=output_k)

    np.testing.assert_allclose(np.asarray(basis.k), output_k)
    assert np.asarray(basis.components["real_tree_matter"]).shape == output_k.shape
    np.testing.assert_allclose(
        np.asarray(basis.components["real_tree_matter"]),
        np.interp(np.log(output_k), np.log(linear_input.k), linear_input.pk_linear),
    )
    np.testing.assert_allclose(
        np.asarray(basis.components["rsd_l2_mm_00"]),
        (4.0 * linear_input.growth_rate**2 / 7.0) * np.asarray(basis.components["real_tree_matter"]),
    )


def test_predict_galaxy_multipoles_rejects_cosmology_object_source() -> None:
    z = 0.5
    cosmo = Class()
    cosmo.set(
        {
            "A_s": 2.089e-9,
            "n_s": 0.9649,
            "tau_reio": 0.052,
            "omega_b": 0.02237,
            "omega_cdm": 0.12,
            "h": 0.6736,
            "YHe": 0.2425,
            "N_ur": 2.0328,
            "N_ncdm": 1,
            "m_ncdm": 0.06,
            "z_pk": z,
            "output": "mPk",
        }
    )
    cosmo.compute()

    params = EFTBiasParams(b1=2.0, b2=0.0, bG2=0.0, bGamma3=0.0, cs0=0.0, cs2=0.0, cs4=0.0, Pshot=0.0, b4=0.0)
    with pytest.raises(TypeError, match="Build a LinearPowerInput or PowerSpectrumTemplate"):
        predict_galaxy_multipoles(cosmo, np.logspace(-3, -1, 8), z, params)


def test_predict_galaxy_multipoles_accepts_theory_source() -> None:
    linear_input = LinearPowerInput(
        k=np.logspace(-3, -1, 16),
        pk_linear=np.linspace(100.0, 200.0, 16),
        z=0.5,
        growth_factor=0.75,
        growth_rate=0.8,
        h=0.7,
    )
    params = EFTBiasParams(b1=2.0, b2=0.0, bG2=0.0, bGamma3=0.0, cs0=0.0, cs2=0.0, cs4=0.0, Pshot=0.0, b4=0.0)
    theory = GalaxyPowerSpectrumMultipolesTheory(
        template=PowerSpectrumTemplate.from_linear_input(linear_input, settings=PTSettings(ir_resummation=False)),
        k=np.linspace(0.01, 0.08, 8),
    )

    prediction = predict_galaxy_multipoles(theory, params)

    assert prediction.p0.shape == theory.k.shape
    assert prediction.metadata["theory"] == "GalaxyPowerSpectrumMultipolesTheory"


def test_classpt_reference_theory_uses_template_without_backend_flag() -> None:
    z = 0.5
    k = np.logspace(-3, -1, 8)
    params = EFTBiasParams(b1=2.0, b2=0.0, bG2=0.0, bGamma3=0.0, cs0=0.0, cs2=0.0, cs4=0.0, Pshot=0.0, b4=0.0)

    cosmo = Class()
    cosmo.set(
        {
            "A_s": 2.089e-9,
            "n_s": 0.9649,
            "tau_reio": 0.052,
            "omega_b": 0.02237,
            "omega_cdm": 0.12,
            "h": 0.6736,
            "YHe": 0.2425,
            "N_ur": 2.0328,
            "N_ncdm": 1,
            "m_ncdm": 0.06,
            "z_pk": z,
            "output": "mTk,mPk",
            "non linear": "PT",
            "IR resummation": "No",
            "Bias tracers": "Yes",
            "cb": "Yes",
            "RSD": "Yes",
        }
    )
    cosmo.compute()

    theory = ClassPTGalaxyPowerSpectrumMultipolesTheory(template=PowerSpectrumTemplate(cosmo, z=z), k=k)
    prediction = theory(params)

    assert prediction.k.shape == k.shape
    assert prediction.metadata["backend"] == "classpt"
    assert prediction.metadata["theory"] == "ClassPTGalaxyPowerSpectrumMultipolesTheory"
