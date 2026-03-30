from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import json

import numpy as np
import pytest

from jaxpt import LinearPowerInput, PTSettings, TaylorEmulator, build_multipole_emulator
from jaxpt.parameter import ParameterCollection
from jaxpt.reference.classpt import MultipolePrediction
from jaxpt.theories import GalaxyPowerSpectrumMultipolesTheory, PowerSpectrumTemplate

from conftest import make_bias_params


def test_taylor_emulator_reproduces_polynomial_exactly() -> None:
    def theory(params: dict[str, float]) -> np.ndarray:
        x = params["x"]
        y = params["y"]
        return np.asarray(
            [
                1.0 + 2.0 * x - 3.0 * y + 4.0 * x * y + 5.0 * x**2,
                -2.0 + 0.5 * x**2 * y + 7.0 * y**3,
            ],
            dtype=float,
        )

    emulator = TaylorEmulator(
        theory,
        fiducial={"x": 0.0, "y": 0.0},
        order=3,
        step_sizes={"x": 0.1, "y": 0.1},
    ).build()

    query = {"x": 0.07, "y": -0.04}
    np.testing.assert_allclose(emulator.predict(query), theory(query), rtol=1e-11, atol=1e-11)


def test_taylor_emulator_handles_subset_parameters_and_query_validation() -> None:
    def theory(params: dict[str, float]) -> np.ndarray:
        x = params["x"]
        y = params["y"]
        z = params["z"]
        return np.asarray([x**2 + 2.0 * y + 3.0 * z], dtype=float)

    emulator = TaylorEmulator(
        theory,
        fiducial={"x": 1.0, "y": -2.0, "z": 0.5},
        order=2,
        step_sizes=0.1,
        param_names=["x", "y"],
        valid_param_names=["x", "y", "z"],
    ).build()

    np.testing.assert_allclose(
        emulator.predict({"x": 1.2, "y": -1.7}),
        theory({"x": 1.2, "y": -1.7, "z": 0.5}),
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        emulator.predict({"x": 1.1}),
        theory({"x": 1.1, "y": -2.0, "z": 0.5}),
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        emulator.predict({"x": 1.1, "z": 0.5}),
        theory({"x": 1.1, "y": -2.0, "z": 0.5}),
        rtol=1e-11,
        atol=1e-11,
    )

    with pytest.raises(ValueError, match="were not emulated"):
        emulator.predict({"z": 0.6})

    with pytest.raises(ValueError, match="Unexpected emulator parameters"):
        emulator.predict({"w": 1.0})


def test_taylor_emulator_reconstructs_multipole_predictions() -> None:
    k = np.array([0.05, 0.1, 0.15], dtype=float)

    def theory(params: dict[str, float]) -> MultipolePrediction:
        alpha = params["alpha"]
        beta = params["beta"]
        return MultipolePrediction(
            k=k,
            p0=1.0 + alpha + beta * k,
            p2=-0.5 + alpha * beta * k**2,
            p4=2.0 + beta**2 * k,
            metadata={"backend": "synthetic"},
        )

    emulator = TaylorEmulator(
        theory,
        fiducial={"alpha": 0.0, "beta": 0.0},
        order=2,
        step_sizes={"alpha": 0.1, "beta": 0.2},
    ).build()

    query = {"alpha": 0.03, "beta": -0.08}
    prediction = emulator.predict(query)
    expected = theory(query)

    assert isinstance(prediction, MultipolePrediction)
    np.testing.assert_allclose(prediction.k, expected.k, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(prediction.p0, expected.p0, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(prediction.p2, expected.p2, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(prediction.p4, expected.p4, rtol=1e-11, atol=1e-11)
    assert prediction.metadata == expected.metadata


def test_taylor_emulator_save_load_and_hash_cache(tmp_path) -> None:
    def theory(params: dict[str, float]) -> np.ndarray:
        x = params["x"]
        y = params["y"]
        return np.asarray([1.0 + x + y**2], dtype=float)

    emulator = TaylorEmulator(
        theory,
        fiducial={"x": 0.0, "y": 0.0, "z": 1.5},
        order=2,
        step_sizes={"x": 0.1, "y": 0.2},
        param_names=["x", "y"],
        cache_dir=tmp_path,
        metadata={"theory": "toy"},
        valid_param_names=["x", "y", "z"],
    ).build()

    assert emulator.cache_path is not None
    assert emulator.cache_path.exists()

    emulator.save(emulator.cache_path)

    loaded = TaylorEmulator.load(emulator.cache_path)
    query = {"x": 0.03, "y": -0.04}
    np.testing.assert_allclose(loaded.predict(query), theory(query), rtol=1e-11, atol=1e-11)
    assert loaded.params["x"].value == 0.0
    assert loaded.params["z"].value == 1.5
    with pytest.raises(ValueError, match="were not emulated"):
        loaded.predict({"z": 0.0})

    other = TaylorEmulator(
        theory,
        fiducial={"x": 0.0, "y": 0.0},
        order=3,
        step_sizes={"x": 0.1, "y": 0.2},
        param_names=["x", "y"],
        cache_dir=tmp_path,
        metadata={"theory": "toy"},
    )
    assert other._resolve_cache_path() != emulator.cache_path


def test_taylor_emulator_preserves_parameter_collection_metadata(tmp_path) -> None:
    params = ParameterCollection(
        {
            "x": {"value": 1.0, "prior": {"type": "gaussian", "mean": 1.0, "sigma": 0.2}},
            "y": {"value": 2.0, "fixed": True},
        }
    )

    emulator = TaylorEmulator(
        lambda query: np.asarray([query["x"] + query["y"]], dtype=float),
        fiducial=params.defaults_dict(),
        order=1,
        step_sizes={"x": 0.1},
        param_names=["x"],
        valid_param_names=["x", "y"],
        params=params,
    ).build()

    path = tmp_path / "toy_emulator.npz"
    emulator.save(path)
    loaded = TaylorEmulator.load(path)

    assert loaded.params["x"].prior == {"type": "gaussian", "mean": 1.0, "sigma": 0.2}
    assert loaded.params["y"].fixed is True
    assert loaded.params["y"].value == 2.0


def test_taylor_emulator_loads_legacy_file_without_parameter_metadata(tmp_path) -> None:
    emulator = TaylorEmulator(
        lambda query: np.asarray([query["x"] ** 2], dtype=float),
        fiducial={"x": 1.0},
        order=2,
        step_sizes={"x": 0.1},
    ).build()
    path = tmp_path / "legacy_emulator.npz"
    emulator.save(path)

    with np.load(path, allow_pickle=False) as data:
        config = json.loads(str(data["config_json"]))
        config.pop("parameter_collection", None)
        payload = {name: data[name] for name in data.files if name != "config_json"}
    np.savez(path, **payload, config_json=json.dumps(config, sort_keys=True))

    loaded = TaylorEmulator.load(path)
    np.testing.assert_allclose(loaded.predict({"x": 1.2}), np.asarray([1.2**2], dtype=float), rtol=1e-11, atol=1e-11)
    assert loaded.params["x"].prior is None


def test_taylor_emulator_build_reports_progress_and_skips_on_cache_hit(tmp_path) -> None:
    calls: list[tuple[int, int]] = []

    def theory(params: dict[str, float]) -> np.ndarray:
        return np.asarray([params["x"] ** 2 + params["y"]], dtype=float)

    emulator = TaylorEmulator(
        theory,
        fiducial={"x": 0.0, "y": 0.0},
        order=2,
        step_sizes={"x": 0.1, "y": 0.1},
        cache_dir=tmp_path,
    ).build(progress_callback=lambda completed, total: calls.append((completed, total)))

    assert calls
    assert calls[-1] == (emulator.n_terms, emulator.n_terms)
    assert calls == sorted(calls)

    cached_calls: list[tuple[int, int]] = []
    TaylorEmulator(
        theory,
        fiducial={"x": 0.0, "y": 0.0},
        order=2,
        step_sizes={"x": 0.1, "y": 0.1},
        cache_dir=tmp_path,
    ).build(progress_callback=lambda completed, total: cached_calls.append((completed, total)))

    assert cached_calls == []


def test_taylor_emulator_tracks_nearby_jaxpt_multipole_theory() -> None:
    linear_input = LinearPowerInput(
        k=np.logspace(-3.0, 0.0, 64),
        pk_linear=np.linspace(2.0e4, 5.0e2, 64),
        z=0.5,
        growth_factor=0.76,
        growth_rate=0.81,
        h=0.67,
    )
    eval_k = np.linspace(0.02, 0.18, 12)
    theory = GalaxyPowerSpectrumMultipolesTheory(
        template=PowerSpectrumTemplate.from_linear_input(linear_input, settings=PTSettings(ir_resummation=False)),
        k=eval_k,
    )
    fiducial = make_bias_params()

    emulator = TaylorEmulator(
        theory,
        fiducial=fiducial,
        param_names=["b1", "b2", "bG2", "cs2"],
        order=2,
        step_sizes={"b1": 0.05, "b2": 0.05, "bG2": 0.02, "cs2": 1.0},
    ).build()

    center = emulator.predict(fiducial)
    expected_center = theory(fiducial)
    np.testing.assert_allclose(center.p0, expected_center.p0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(center.p2, expected_center.p2, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(center.p4, expected_center.p4, rtol=1e-10, atol=1e-10)

    query = {**fiducial, "b1": fiducial["b1"] + 0.03, "b2": fiducial["b2"] - 0.02, "bG2": fiducial["bG2"] + 0.01, "cs2": fiducial["cs2"] + 0.5}
    predicted = emulator.predict(query)
    expected = theory(query)

    np.testing.assert_allclose(predicted.p0, expected.p0, rtol=3e-3, atol=1e-3)
    np.testing.assert_allclose(predicted.p2, expected.p2, rtol=3e-3, atol=1e-3)
    np.testing.assert_allclose(predicted.p4, expected.p4, rtol=3e-3, atol=1e-3)


def test_build_multipole_emulator_uses_non_fixed_non_marginalized_params(tmp_path) -> None:
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
        k=np.linspace(0.02, 0.18, 8),
    )

    emulator = build_multipole_emulator(
        theory,
        order=1,
        step_sizes=0.05,
        cache_dir=tmp_path,
    )

    assert emulator.param_names == ["b1", "b2", "bG2", "bGamma3", "b4"]
    assert emulator.cache_path is not None
    assert emulator.cache_path.name.startswith("taylor_")
    assert emulator.cache_path.suffix == ".npz"

    with pytest.raises(ValueError, match="were not emulated"):
        emulator.predict({"cs2": make_bias_params()["cs2"] + 1.0})


def test_build_multipole_emulator_rejects_marginalized_parameter_request() -> None:
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
        k=np.linspace(0.02, 0.18, 8),
    )

    with pytest.raises(ValueError, match="non-fixed and non-marginalized"):
        build_multipole_emulator(
            theory,
            order=1,
            step_sizes=0.05,
            param_names=["b1", "cs2"],
        )


def test_build_multipole_emulator_script_smoke(tmp_path) -> None:
    pytest.importorskip("classy")
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_taylor_emulator.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--order",
            "1",
            "--nk",
            "4",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    outputs = sorted(tmp_path.glob("taylor_*.npz"))
    assert len(outputs) == 1
    assert "Saved emulator:" in result.stdout

    loaded = TaylorEmulator.load(outputs[0])
    with pytest.raises(ValueError, match="were not emulated"):
        loaded.predict({"n_s": 0.97})
