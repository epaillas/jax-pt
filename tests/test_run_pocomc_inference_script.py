from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from jaxpt import PTSettings, build_multipole_emulator, load_pgg_data_vector
from jaxpt.theories import GalaxyPowerSpectrumMultipolesTheory, PowerSpectrumTemplate


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_VECTOR_PATH = REPO_ROOT / "data" / "data_vector" / "mesh2_spectrum_poles_c000_hod006.h5"
MOCK_DIR = REPO_ROOT / "data" / "for_covariance"


def _build_small_mock_directory(tmp_path: Path, n_mocks: int = 4) -> Path:
    directory = tmp_path / "mocks"
    directory.mkdir()
    for path in sorted(MOCK_DIR.glob("mesh2_spectrum_poles_ph*.h5"))[:n_mocks]:
        shutil.copy2(path, directory / path.name)
    return directory


def test_run_pocomc_inference_script_smoke(tmp_path) -> None:
    pytest.importorskip("pocomc")

    k_data, _ = load_pgg_data_vector(DATA_VECTOR_PATH, ells=(0, 2, 4), rebin=13, kmin=0.01, kmax=0.2)
    cosmology = {
        "omega_b": 0.02237,
        "omega_cdm": 0.12,
        "h": 0.6736,
        "n_s": 0.9649,
        "A_s": 2.089e-9,
        "tau_reio": 0.052,
        "YHe": 0.2425,
        "N_ur": 2.0328,
        "N_ncdm": 1.0,
        "m_ncdm": 0.06,
        "Omega_k": 0.0,
        "w0_fld": -1.0,
        "wa_fld": 0.0,
    }
    template = PowerSpectrumTemplate(
        cosmology,
        z=0.5,
        settings=PTSettings(backend="jaxpt", ir_resummation=False),
        provider="cosmoprimo",
    )
    train_k = np.linspace(0.01, 0.2, len(k_data) + 9)
    theory = GalaxyPowerSpectrumMultipolesTheory(template=template, k=train_k)
    for name in theory.template.params.names():
        theory.params[name].update(fixed=True)
    theory.params["b1"].update(value=2.0)

    emulator = build_multipole_emulator(
        theory,
        order=1,
        param_names=["b1"],
        step_sizes={"b1": 0.05},
        cache_dir=tmp_path,
        metadata={"script": "test_run_pocomc_inference_script"},
    )
    assert emulator.cache_path is not None

    mock_dir = _build_small_mock_directory(tmp_path)
    output = tmp_path / "posterior.npz"
    script = REPO_ROOT / "scripts" / "run_pocomc_inference.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(emulator.cache_path),
            "--data",
            str(DATA_VECTOR_PATH),
            "--mocks",
            str(mock_dir),
            "--param",
            "b1",
            "--flat-prior",
            "b1=1.0,3.0",
            "--n-active",
            "8",
            "--n-effective",
            "16",
            "--n-total",
            "16",
            "--n-evidence",
            "0",
            "--epochs",
            "1",
            "--no-precondition",
            "--covariance-jitter",
            "1e-3",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.exists()
    with np.load(output, allow_pickle=False) as data:
        assert data["samples"].ndim == 2
        assert data["samples"].shape[1] == 1
        assert data["weights"].ndim == 1
        assert data["parameter_names"].tolist() == ["b1"]
        assert str(data["meta_observable"]) == "pgg"
        assert str(data["meta_model_kind"]) == "taylor_emulator_pgg"
        assert str(data["meta_bestfit_rule"]) == "max_logl"
        np.testing.assert_allclose(np.asarray(data["meta_k"], dtype=float), k_data)
        assert tuple(np.asarray(data["meta_ells"], dtype=int)) == (0, 2, 4)
        assert np.asarray(data["meta_data_vector"], dtype=float).shape == (len(k_data) * 3,)
        assert np.asarray(data["meta_covariance"], dtype=float).shape == (len(k_data) * 3, len(k_data) * 3)
        assert np.asarray(data["meta_errors"], dtype=float).shape == (len(k_data) * 3,)
        assert np.asarray(data["meta_baseline_param_names"], dtype=str).size == np.asarray(data["meta_baseline_param_values"], dtype=float).size
    assert "PocoMC inference" in result.stdout
    assert "sampled parameters (1): b1" in result.stdout
    assert "covariance_cache:" in result.stdout
    assert "posterior_samples:" in result.stdout
