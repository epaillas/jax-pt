import numpy as np
import pytest
from classy import Class
from unittest.mock import patch

from jaxpt import EFTBiasParams, LinearPowerInput, PTSettings, build_linear_input_from_classy, compute_basis, galaxy_multipoles, predict_galaxy_multipoles
from jaxpt.cosmology import (
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
    with pytest.raises(TypeError, match="build_linear_input_from_classy"):
        predict_galaxy_multipoles(cosmo, np.logspace(-3, -1, 8), z, params)
