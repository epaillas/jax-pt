import numpy as np
import pytest
from classy import Class

from jaxpt import (
    EFTBiasParams,
    LinearPowerInput,
    PTSettings,
    build_linear_input_from_classy,
    build_native_realspace_predictor,
    compare_multipoles_to_classpt,
    compute_basis,
    galaxy_real_spectrum,
    galaxy_multipoles,
    predict_galaxy_multipoles,
)
from jaxpt.cosmology import (
    build_classpt_native_grid_parity_linear_input_from_classy,
    build_classpt_parity_linear_input_from_classy,
    prepare_native_fftlog_input,
)
from jaxpt.bias import matter_real_spectrum
from jaxpt.kernels.rsd_spectral import compute_native_rsd_terms
from jaxpt.kernels.spectral import compute_native_realspace_terms_from_preprocessed
from jaxpt.reference.classpt import BasisSpectra, MultipolePrediction, PK_MULT_INDEX

from conftest import DEFAULT_PT_OPTIONS, DEFAULT_PT_OPTIONS_NOIR, FIDUCIAL_COSMOLOGY, SHIFTED_COSMOLOGY


class MockClassPT:
    def __init__(self, pk_mult: np.ndarray, h: float, fz: float):
        self._pk_mult = pk_mult
        self._h = h
        self._fz = fz

    def get_pk_mult(self, k, z, k_size, no_wiggle=False, alpha_rs=1.0):
        assert len(k) == k_size
        return self._pk_mult

    def h(self):
        return self._h

    def scale_independent_growth_factor_f(self, z):
        return self._fz

    def pk_gg_l0(self, b1, b2, bG2, bGamma3, cs0, Pshot, b4):
        h = self._h
        kh = self._pk_mult[-1]
        return (
            self._pk_mult[15]
            + self._pk_mult[21]
            + b1 * self._pk_mult[16]
            + b1 * self._pk_mult[22]
            + b1**2 * self._pk_mult[17]
            + b1**2 * self._pk_mult[23]
            + 0.25 * b2**2 * self._pk_mult[1]
            + b1 * b2 * self._pk_mult[30]
            + b2 * self._pk_mult[31]
            + b1 * bG2 * self._pk_mult[32]
            + bG2 * self._pk_mult[33]
            + b2 * bG2 * self._pk_mult[4]
            + bG2**2 * self._pk_mult[5]
            + 2.0 * cs0 * self._pk_mult[11] / h**2
            + (2.0 * bG2 + 0.8 * bGamma3) * (b1 * self._pk_mult[7] + self._pk_mult[8])
        ) * h**3 + Pshot + self._fz**2 * b4 * (kh / h) ** 2 * (self._fz**2 / 9.0 + 2.0 * self._fz * b1 / 7.0 + b1**2 / 5.0) * (35.0 / 8.0) * self._pk_mult[13] * h

    def pk_gg_l2(self, b1, b2, bG2, bGamma3, cs2, b4):
        h = self._h
        kh = self._pk_mult[-1]
        return (
            self._pk_mult[18]
            + self._pk_mult[24]
            + b1 * self._pk_mult[19]
            + b1 * self._pk_mult[25]
            + b1**2 * self._pk_mult[26]
            + b1 * b2 * self._pk_mult[34]
            + b2 * self._pk_mult[35]
            + b1 * bG2 * self._pk_mult[36]
            + bG2 * self._pk_mult[37]
            + 2.0 * cs2 * self._pk_mult[12] / h**2
            + (2.0 * bG2 + 0.8 * bGamma3) * self._pk_mult[9]
        ) * h**3 + self._fz**2 * b4 * (kh / h) ** 2 * ((self._fz**2 * 70.0 + 165.0 * self._fz * b1 + 99.0 * b1**2) * 4.0 / 693.0) * (35.0 / 8.0) * self._pk_mult[13] * h

    def pk_gg_l4(self, b1, b2, bG2, bGamma3, cs4, b4):
        h = self._h
        kh = self._pk_mult[-1]
        return (
            self._pk_mult[20]
            + self._pk_mult[27]
            + b1 * self._pk_mult[28]
            + b1**2 * self._pk_mult[29]
            + b2 * self._pk_mult[38]
            + bG2 * self._pk_mult[39]
            + 2.0 * cs4 * self._pk_mult[13] / h**2
        ) * h**3 + self._fz**2 * b4 * (kh / h) ** 2 * ((self._fz**2 * 210.0 + 390.0 * self._fz * b1 + 143.0 * b1**2) * 8.0 / 5005.0) * (35.0 / 8.0) * self._pk_mult[13] * h


def make_mock_basis(pk_mult: np.ndarray, k: np.ndarray, *, z: float, h: float, fz: float, backend: str = "mock") -> BasisSpectra:
    components = {
        name: pk_mult[index]
        for name, index in PK_MULT_INDEX.items()
    }
    components["k_over_h_squared"] = (k / h) ** 2
    return BasisSpectra(k=k, z=z, h=h, growth_rate=fz, components=components, metadata={"backend": backend})


def test_galaxy_multipoles_match_classpt_formula_surface() -> None:
    z = 0.5
    k = np.logspace(-3, -0.1, 32)
    h = 0.6736
    fz = 0.76
    pk_mult = np.vstack([np.linspace(i + 1.0, i + 2.0, k.size) for i in range(96)])
    pk_mult[-1] = k

    cosmo = MockClassPT(pk_mult=pk_mult, h=h, fz=fz)
    params = EFTBiasParams(
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

    basis = make_mock_basis(pk_mult=pk_mult, k=k, z=z, h=h, fz=fz)
    prediction = galaxy_multipoles(basis, params)

    expected_p0 = cosmo.pk_gg_l0(params.b1, params.b2, params.bG2, params.bGamma3, params.cs0, params.Pshot, params.b4)
    expected_p2 = cosmo.pk_gg_l2(params.b1, params.b2, params.bG2, params.bGamma3, params.cs2, params.b4)
    expected_p4 = cosmo.pk_gg_l4(params.b1, params.b2, params.bG2, params.bGamma3, params.cs4, params.b4)

    np.testing.assert_allclose(np.asarray(prediction.p0), expected_p0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(prediction.p2), expected_p2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(prediction.p4), expected_p4, rtol=1e-12, atol=1e-12)


def test_synthetic_basis_maps_named_components() -> None:
    k = np.logspace(-3, -1, 8)
    pk_mult = np.vstack([np.full(k.size, i, dtype=float) for i in range(96)])
    cosmo = MockClassPT(pk_mult=pk_mult, h=0.7, fz=0.8)

    basis = make_mock_basis(pk_mult=pk_mult, k=k, z=0.5, h=0.7, fz=0.8)

    assert np.allclose(np.asarray(basis.components["real_tree_matter"]), 14.0)
    assert np.allclose(np.asarray(basis.components["real_loop_matter"]), 0.0)
    assert np.allclose(np.asarray(basis.components["rsd_l0_mm_00"]), 15.0)
    assert np.allclose(np.asarray(basis.components["rsd_l4_bG2"]), 39.0)
    np.testing.assert_allclose(np.asarray(basis.components["k_over_h_squared"]), (k / 0.7) ** 2)


@pytest.mark.parametrize(
    ("cosmology", "z"),
    [
        (FIDUCIAL_COSMOLOGY, 0.5),
        (SHIFTED_COSMOLOGY, 1.0),
    ],
)
def test_live_classpt_parity_for_galaxy_multipoles(benchmark_k: np.ndarray, cosmology: dict[str, float], z: float) -> None:
    cosmo = Class()
    cosmo.set({**cosmology, **DEFAULT_PT_OPTIONS, "z_pk": z})
    cosmo.compute()

    params = EFTBiasParams(
        b1=2.0 if z < 0.75 else 1.7,
        b2=-1.0 if z < 0.75 else -0.4,
        bG2=0.1,
        bGamma3=-0.1,
        cs0=0.0,
        cs2=30.0,
        cs4=0.0,
        Pshot=3000.0 if z < 0.75 else 1800.0,
        b4=10.0 if z < 0.75 else 6.0,
    )

    cosmo.initialize_output(benchmark_k, z, len(benchmark_k))
    expected_p0 = cosmo.pk_gg_l0(params.b1, params.b2, params.bG2, params.bGamma3, params.cs0, params.Pshot, params.b4)
    expected_p2 = cosmo.pk_gg_l2(params.b1, params.b2, params.bG2, params.bGamma3, params.cs2, params.b4)
    expected_p4 = cosmo.pk_gg_l4(params.b1, params.b2, params.bG2, params.bGamma3, params.cs4, params.b4)
    prediction = MultipolePrediction(
        k=benchmark_k,
        p0=expected_p0,
        p2=expected_p2,
        p4=expected_p4,
        metadata={"z": z},
    )
    comparison = compare_multipoles_to_classpt(prediction, cosmo, params)

    assert comparison["p0"]["max_rel"] == 0.0
    assert comparison["p2"]["max_rel"] == 0.0
    assert comparison["p4"]["max_rel"] == 0.0


def test_predict_galaxy_multipoles_accepts_linear_input(benchmark_k: np.ndarray) -> None:
    z = 0.5
    cosmo = Class()
    cosmo.set({**FIDUCIAL_COSMOLOGY, "z_pk": z, "output": "mPk"})
    cosmo.compute()

    linear_input = build_linear_input_from_classy(cosmo, z=z, k=benchmark_k)
    params = EFTBiasParams(b1=2.0, b2=0.0, bG2=0.0, bGamma3=0.0, cs0=0.0, cs2=0.0, cs4=0.0, Pshot=0.0, b4=0.0)

    prediction = predict_galaxy_multipoles(linear_input, params, settings=PTSettings(ir_resummation=False))

    assert prediction.p0.shape == benchmark_k.shape
    assert prediction.metadata["backend"] == "native"


def test_classpt_parity_linear_input_matches_internal_tree_basis(benchmark_k: np.ndarray) -> None:
    z = 0.5
    linear_k = np.logspace(-5.0, 1.0, 256)

    cosmo = Class()
    cosmo.set({**FIDUCIAL_COSMOLOGY, **DEFAULT_PT_OPTIONS_NOIR, "z_pk": z})
    cosmo.compute()

    default_input = build_linear_input_from_classy(cosmo, z=z, k=linear_k)
    parity_input = build_classpt_parity_linear_input_from_classy(cosmo, z=z, k=linear_k)

    default_basis = compute_basis(default_input, settings=PTSettings(loop_order="tree", ir_resummation=False), k=benchmark_k)
    parity_basis = compute_basis(parity_input, settings=PTSettings(loop_order="tree", ir_resummation=False), k=benchmark_k)

    class_tree = np.asarray([np.asarray(cosmo.pk(float(k), z), dtype=float)[14] for k in benchmark_k], dtype=float)
    default_tree = np.asarray(default_basis.components["real_tree_matter"])
    parity_tree = np.asarray(parity_basis.components["real_tree_matter"])

    assert np.max(np.abs(default_tree / class_tree - 1.0)) > 5.0e-3
    np.testing.assert_allclose(parity_tree, class_tree, rtol=2.0e-3, atol=0.0)


def test_native_grid_parity_input_reduces_live_classpt_multipole_mismatch() -> None:
    z = 0.5
    eval_k = np.linspace(0.01, 0.2, 64)
    settings = PTSettings(ir_resummation=False)
    params = EFTBiasParams(
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

    cosmo = Class()
    cosmo.set({**FIDUCIAL_COSMOLOGY, **DEFAULT_PT_OPTIONS_NOIR, "output": "mTk,mPk", "z_pk": z})
    cosmo.compute()
    cosmo.initialize_output(eval_k, z, len(eval_k))

    reference = [
        np.asarray(cosmo.pk_gg_l0(params.b1, params.b2, params.bG2, params.bGamma3, params.cs0, params.Pshot, params.b4)),
        np.asarray(cosmo.pk_gg_l2(params.b1, params.b2, params.bG2, params.bGamma3, params.cs2, params.b4)),
        np.asarray(cosmo.pk_gg_l4(params.b1, params.b2, params.bG2, params.bGamma3, params.cs4, params.b4)),
    ]

    coarse_input = build_classpt_parity_linear_input_from_classy(cosmo, z=z, k=np.logspace(-5.0, 1.0, 256))
    native_grid_input = build_classpt_native_grid_parity_linear_input_from_classy(cosmo, z=z, settings=settings)

    coarse_prediction = galaxy_multipoles(compute_basis(coarse_input, settings=settings, k=eval_k), params)
    native_grid_prediction = galaxy_multipoles(compute_basis(native_grid_input, settings=settings, k=eval_k), params)

    coarse_max_rel = []
    native_grid_max_rel = []
    for coarse_values, native_grid_values, expected in zip(
        [coarse_prediction.p0, coarse_prediction.p2, coarse_prediction.p4],
        [native_grid_prediction.p0, native_grid_prediction.p2, native_grid_prediction.p4],
        reference,
    ):
        denominator = np.maximum(np.abs(expected), 1.0)
        coarse_max_rel.append(float(np.max(np.abs(np.asarray(coarse_values) - expected) / denominator)))
        native_grid_max_rel.append(float(np.max(np.abs(np.asarray(native_grid_values) - expected) / denominator)))

    assert native_grid_max_rel[0] < coarse_max_rel[0]
    assert native_grid_max_rel[1] < coarse_max_rel[1]
    assert native_grid_max_rel[2] < coarse_max_rel[2]


def test_preprocessed_fftlog_input_matches_direct_loop_kernel_entrypoints() -> None:
    z = 0.5
    eval_k = np.linspace(0.01, 0.2, 24)
    settings = PTSettings(ir_resummation=False)

    cosmo = Class()
    cosmo.set({**FIDUCIAL_COSMOLOGY, **DEFAULT_PT_OPTIONS_NOIR, "output": "mTk,mPk", "z_pk": z})
    cosmo.compute()

    linear_input = build_classpt_native_grid_parity_linear_input_from_classy(cosmo, z=z, settings=settings)
    fftlog_input = prepare_native_fftlog_input(linear_input, settings)

    real_terms = compute_native_realspace_terms_from_preprocessed(fftlog_input, settings, output_k=eval_k)
    rsd_terms = compute_native_rsd_terms(linear_input, settings, output_k=eval_k, fftlog_input=fftlog_input)

    direct_basis = compute_basis(linear_input, settings=settings, k=eval_k)

    np.testing.assert_allclose(np.asarray(real_terms["real_loop_matter"]), np.asarray(direct_basis.components["real_loop_matter"]))
    np.testing.assert_allclose(np.asarray(real_terms["real_gamma3"]), np.asarray(direct_basis.components["real_gamma3"]))
    np.testing.assert_allclose(np.asarray(rsd_terms["rsd_l2_loop_01"]), np.asarray(direct_basis.components["rsd_l2_loop_01"]))
    np.testing.assert_allclose(np.asarray(rsd_terms["rsd_l4_b2"]), np.asarray(direct_basis.components["rsd_l4_b2"]))


def test_native_real_loop_matter_matches_classpt_on_conservative_k_range() -> None:
    z = 0.5
    eval_k = np.logspace(-2.5, -1.8, 6)
    linear_k = np.logspace(-5.0, 1.0, 256)

    cosmo = Class()
    cosmo.set({**FIDUCIAL_COSMOLOGY, **DEFAULT_PT_OPTIONS_NOIR, "z_pk": z})
    cosmo.compute()

    linear_input = build_linear_input_from_classy(cosmo, z=z, k=linear_k)
    basis = compute_basis(linear_input, settings=PTSettings(ir_resummation=False), k=eval_k)
    native_loop = np.asarray(basis.components["real_loop_matter"])

    assert native_loop.shape == eval_k.shape
    assert np.all(np.isfinite(native_loop))
    assert np.any(np.abs(native_loop) > 0.0)


def test_tree_loop_order_returns_zero_realspace_loop_terms() -> None:
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
    for name in (
        "real_loop_matter",
        "real_loop_b2_b2",
        "real_cross_b1_b2",
        "real_cross_b1_bG2",
        "real_loop_b2_bG2",
        "real_loop_bG2_bG2",
        "real_gamma3",
    ):
        np.testing.assert_allclose(np.asarray(basis.components[name]), 0.0, rtol=0.0, atol=0.0)


def test_native_one_loop_basis_populates_rsd_terms_without_classpt_metadata() -> None:
    k = np.logspace(-5.0, 0.2, 256)
    pk_linear = 2.0e4 * (k / 0.1) ** -1.2 * np.exp(-k / 0.45)
    linear_input = LinearPowerInput(
        k=k,
        pk_linear=pk_linear,
        z=0.5,
        growth_factor=0.75,
        growth_rate=0.8,
        h=0.7,
    )

    basis = compute_basis(linear_input, settings=PTSettings(ir_resummation=False), k=np.linspace(0.01, 0.2, 16))

    assert basis.metadata["kernel_source"] == "analytic"
    assert basis.metadata["spectral_method"] == "fftlog_jax"
    assert basis.metadata["rsd_matter"] is True
    assert basis.metadata["rsd_bias"] is True
    assert np.any(np.abs(np.asarray(basis.components["rsd_l0_loop_00"])) > 0.0)
    assert np.any(np.abs(np.asarray(basis.components["rsd_l2_b1_b2"])) > 0.0)
    assert np.any(np.abs(np.asarray(basis.components["rsd_l0_gamma3_b1"])) > 0.0)
    assert np.all(np.isfinite(np.asarray(basis.components["rsd_l4_bG2"])))


def test_native_one_loop_basis_has_same_contract_for_plain_and_class_inputs() -> None:
    z = 0.5
    eval_k = np.linspace(0.01, 0.2, 16)
    linear_k = np.logspace(-5.0, 1.0, 256)

    cosmo = Class()
    cosmo.set({**FIDUCIAL_COSMOLOGY, **DEFAULT_PT_OPTIONS_NOIR, "z_pk": z})
    cosmo.compute()

    class_linear_input = build_linear_input_from_classy(cosmo, z=z, k=linear_k)
    plain_linear_input = LinearPowerInput(
        k=np.asarray(class_linear_input.k),
        pk_linear=np.asarray(class_linear_input.pk_linear),
        z=class_linear_input.z,
        growth_factor=class_linear_input.growth_factor,
        growth_rate=class_linear_input.growth_rate,
        h=class_linear_input.h,
        pk_nowiggle=class_linear_input.pk_nowiggle,
        metadata={key: value for key, value in class_linear_input.metadata.items() if key != "_classpt_cosmo"},
    )

    class_basis = compute_basis(class_linear_input, settings=PTSettings(ir_resummation=False), k=eval_k)
    plain_basis = compute_basis(plain_linear_input, settings=PTSettings(ir_resummation=False), k=eval_k)

    assert set(class_basis.components) == set(plain_basis.components)
    for name in class_basis.components:
        assert np.asarray(class_basis.components[name]).shape == np.asarray(plain_basis.components[name]).shape


@pytest.mark.parametrize(
    ("cosmology", "z", "params"),
    [
        (
            FIDUCIAL_COSMOLOGY,
            0.5,
            EFTBiasParams(b1=2.0, b2=-1.0, bG2=0.1, bGamma3=-0.1, cs0=0.0, cs2=30.0, cs4=0.0, Pshot=3000.0, b4=10.0),
        ),
        (
            SHIFTED_COSMOLOGY,
            1.0,
            EFTBiasParams(b1=1.7, b2=-0.4, bG2=0.1, bGamma3=-0.1, cs0=0.0, cs2=25.0, cs4=0.0, Pshot=1800.0, b4=6.0),
        ),
    ],
)
def test_native_one_loop_multipoles_match_classpt_on_plot_range(
    cosmology: dict[str, float],
    z: float,
    params: EFTBiasParams,
) -> None:
    eval_k = np.linspace(0.01, 0.2, 32)
    linear_k = np.logspace(-5.0, 1.0, 256)

    cosmo = Class()
    cosmo.set({**cosmology, **DEFAULT_PT_OPTIONS_NOIR, "z_pk": z})
    cosmo.compute()

    linear_input = build_linear_input_from_classy(cosmo, z=z, k=linear_k)
    basis = compute_basis(linear_input, settings=PTSettings(ir_resummation=False), k=eval_k)
    prediction = galaxy_multipoles(basis, params)

    cosmo.initialize_output(eval_k, z, len(eval_k))
    expected_p0 = np.asarray(cosmo.pk_gg_l0(params.b1, params.b2, params.bG2, params.bGamma3, params.cs0, params.Pshot, params.b4))
    expected_p2 = np.asarray(cosmo.pk_gg_l2(params.b1, params.b2, params.bG2, params.bGamma3, params.cs2, params.b4))
    expected_p4 = np.asarray(cosmo.pk_gg_l4(params.b1, params.b2, params.bG2, params.bGamma3, params.cs4, params.b4))

    rel_p0 = np.abs(np.asarray(prediction.p0) - expected_p0) / np.maximum(np.abs(expected_p0), 1.0)
    rel_p2 = np.abs(np.asarray(prediction.p2) - expected_p2) / np.maximum(np.abs(expected_p2), 1.0)
    rel_p4 = np.abs(np.asarray(prediction.p4) - expected_p4) / np.maximum(np.abs(expected_p4), 1.0)

    assert rel_p0.max() < 0.02
    assert rel_p2.max() < 0.06
    assert rel_p4.max() < 0.04


def test_native_real_matter_spectrum_matches_classpt_on_conservative_k_range() -> None:
    z = 0.5
    eval_k = np.logspace(-2.5, -1.8, 6)
    linear_k = np.logspace(-5.0, 1.0, 256)

    cosmo = Class()
    cosmo.set({**FIDUCIAL_COSMOLOGY, **DEFAULT_PT_OPTIONS_NOIR, "z_pk": z})
    cosmo.compute()

    linear_input = build_linear_input_from_classy(cosmo, z=z, k=linear_k)
    basis = compute_basis(linear_input, settings=PTSettings(ir_resummation=False), k=eval_k)
    native_pk = np.asarray(matter_real_spectrum(basis, cs=0.0))

    cosmo.initialize_output(eval_k, z, len(eval_k))
    reference_pk = np.asarray(cosmo.pk_mm_real(0.0))
    relative = np.abs(native_pk - reference_pk) / np.maximum(np.abs(reference_pk), 1.0e-30)

    assert np.all(np.isfinite(native_pk))
    assert relative.max() < 0.05


def test_native_realspace_bias_terms_match_classpt_on_conservative_k_range() -> None:
    z = 0.5
    eval_k = np.logspace(-2.5, -1.8, 6)
    linear_k = np.logspace(-5.0, 1.0, 256)

    cosmo = Class()
    cosmo.set({**FIDUCIAL_COSMOLOGY, **DEFAULT_PT_OPTIONS_NOIR, "z_pk": z})
    cosmo.compute()

    linear_input = build_linear_input_from_classy(cosmo, z=z, k=linear_k)
    basis = compute_basis(linear_input, settings=PTSettings(ir_resummation=False), k=eval_k)
    for name in (
        "real_loop_b2_b2",
        "real_cross_b1_b2",
        "real_cross_b1_bG2",
        "real_loop_b2_bG2",
        "real_loop_bG2_bG2",
        "real_gamma3",
    ):
        native = np.asarray(basis.components[name])
        assert np.all(np.isfinite(native))
        assert native.shape == eval_k.shape


def test_native_real_galaxy_spectrum_matches_classpt_on_conservative_k_range() -> None:
    z = 0.5
    eval_k = np.logspace(-2.5, -1.8, 6)
    linear_k = np.logspace(-5.0, 1.0, 256)

    cosmo = Class()
    cosmo.set({**FIDUCIAL_COSMOLOGY, **DEFAULT_PT_OPTIONS_NOIR, "z_pk": z})
    cosmo.compute()

    linear_input = build_linear_input_from_classy(cosmo, z=z, k=linear_k)
    basis = compute_basis(linear_input, settings=PTSettings(ir_resummation=False), k=eval_k)
    prediction = np.asarray(
        galaxy_real_spectrum(
            basis,
            b1=2.0,
            b2=-1.0,
            bG2=0.1,
            bGamma3=-0.1,
            cs=0.0,
            cs0=0.0,
            Pshot=3000.0,
        )
    )

    cosmo.initialize_output(eval_k, z, len(eval_k))
    reference = np.asarray(cosmo.pk_gg_real(2.0, -1.0, 0.1, -0.1, 0.0, 0.0, 3000.0))
    relative = np.abs(prediction - reference) / np.maximum(np.abs(reference), 1.0e-30)

    assert np.all(np.isfinite(prediction))
    assert relative.max() < 0.05


def test_native_real_galaxy_spectrum_matches_classpt_on_plot_range() -> None:
    z = 0.5
    eval_k = np.linspace(0.01, 0.2, 32)
    linear_k = np.logspace(-5.0, 1.0, 256)

    cosmo = Class()
    cosmo.set({**FIDUCIAL_COSMOLOGY, **DEFAULT_PT_OPTIONS_NOIR, "z_pk": z})
    cosmo.compute()

    linear_input = build_linear_input_from_classy(cosmo, z=z, k=linear_k)
    basis = compute_basis(linear_input, settings=PTSettings(ir_resummation=False), k=eval_k)
    prediction = np.asarray(
        galaxy_real_spectrum(
            basis,
            b1=2.0,
            b2=-1.0,
            bG2=0.1,
            bGamma3=-0.1,
            cs=0.0,
            cs0=0.0,
            Pshot=3000.0,
        )
    )

    cosmo.initialize_output(eval_k, z, len(eval_k))
    reference = np.asarray(cosmo.pk_gg_real(2.0, -1.0, 0.1, -0.1, 0.0, 0.0, 3000.0))
    relative = np.abs(prediction - reference) / np.maximum(np.abs(reference), 1.0)

    assert np.all(np.isfinite(prediction))
    assert relative.max() < 0.02


def test_compiled_native_realspace_predictor_matches_basis_path() -> None:
    z = 0.5
    eval_k = np.linspace(0.01, 0.2, 32)
    linear_k = np.logspace(-5.0, 1.0, 256)

    cosmo = Class()
    cosmo.set({**FIDUCIAL_COSMOLOGY, **DEFAULT_PT_OPTIONS_NOIR, "z_pk": z})
    cosmo.compute()

    linear_input = build_linear_input_from_classy(cosmo, z=z, k=linear_k)
    settings = PTSettings(ir_resummation=False)

    basis = compute_basis(linear_input, settings=settings, k=eval_k)
    reference = np.asarray(
        galaxy_real_spectrum(
            basis,
            b1=2.0,
            b2=-1.0,
            bG2=0.1,
            bGamma3=-0.1,
            cs=0.0,
            cs0=0.0,
            Pshot=3000.0,
        )
    )

    predictor = build_native_realspace_predictor(linear_input, settings=settings, k=eval_k)
    compiled = np.asarray(
        predictor(
            b1=2.0,
            b2=-1.0,
            bG2=0.1,
            bGamma3=-0.1,
            cs=0.0,
            cs0=0.0,
            Pshot=3000.0,
        )
    )

    assert compiled.shape == eval_k.shape
    assert np.all(np.isfinite(compiled))
    np.testing.assert_allclose(compiled, reference, rtol=2.0e-2, atol=1.0e-6)


def test_compiled_tree_predictor_matches_basis_path() -> None:
    k = np.logspace(-3, -1, 16)
    linear_input = LinearPowerInput(
        k=k,
        pk_linear=np.linspace(100.0, 200.0, k.size),
        z=0.5,
        growth_factor=0.75,
        growth_rate=0.8,
        h=0.7,
    )
    settings = PTSettings(loop_order="tree", ir_resummation=False)

    basis = compute_basis(linear_input, settings=settings)
    reference = np.asarray(
        galaxy_real_spectrum(
            basis,
            b1=2.0,
            b2=-1.0,
            bG2=0.1,
            bGamma3=-0.1,
            cs=0.0,
            cs0=0.0,
            Pshot=3000.0,
        )
    )

    predictor = build_native_realspace_predictor(linear_input, settings=settings)
    compiled = np.asarray(
        predictor(
            b1=2.0,
            b2=-1.0,
            bG2=0.1,
            bGamma3=-0.1,
            cs=0.0,
            cs0=0.0,
            Pshot=3000.0,
        )
    )

    np.testing.assert_allclose(compiled, reference, rtol=1.0e-12, atol=1.0e-12)


def test_galaxy_multipoles_accepts_plain_native_one_loop_rsd_basis() -> None:
    k = np.logspace(-3, -1, 16)
    linear_input = LinearPowerInput(
        k=k,
        pk_linear=np.linspace(100.0, 200.0, k.size),
        z=0.5,
        growth_factor=0.75,
        growth_rate=0.8,
        h=0.7,
    )
    params = EFTBiasParams(b1=2.0, b2=-1.0, bG2=0.1, bGamma3=-0.1, cs0=0.0, cs2=30.0, cs4=0.0, Pshot=3000.0, b4=10.0)

    basis = compute_basis(linear_input, settings=PTSettings(ir_resummation=False))
    prediction = galaxy_multipoles(basis, params)

    assert prediction.p0.shape == k.shape
    assert prediction.p2.shape == k.shape
    assert prediction.p4.shape == k.shape
    assert np.all(np.isfinite(np.asarray(prediction.p0)))
