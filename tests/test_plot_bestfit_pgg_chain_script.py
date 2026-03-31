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


def _build_chain(tmp_path: Path) -> Path:
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
    theory = GalaxyPowerSpectrumMultipolesTheory(template=template, k=k_data)
    for name in theory.template.params.names():
        theory.params[name].update(fixed=True)
    theory.params["b1"].update(value=2.0)

    emulator = build_multipole_emulator(
        theory,
        order=1,
        param_names=["b1"],
        step_sizes={"b1": 0.05},
        cache_dir=tmp_path,
        metadata={"script": "test_plot_bestfit_pgg_chain_script"},
    )
    assert emulator.cache_path is not None

    mock_dir = _build_small_mock_directory(tmp_path)
    output = tmp_path / "posterior.npz"
    script = REPO_ROOT / "scripts" / "run_pocomc_inference.py"
    subprocess.run(
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
    return output


def test_plot_bestfit_pgg_chain_script_smoke(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    output = tmp_path / "bestfit.png"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "plot_bestfit_pgg_chain.py"),
            str(chain),
            "--output",
            str(output),
            "--n-fine",
            "64",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert "Best-fit Pgg plot" in result.stdout
    assert "bestfit_index:" in result.stdout
    assert "n_k_theory: 64" in result.stdout


def test_plot_bestfit_pgg_chain_script_requires_self_describing_metadata(tmp_path: Path) -> None:
    legacy_chain = tmp_path / "legacy_chain.npz"
    np.savez(
        legacy_chain,
        samples=np.asarray([[1.0]], dtype=float),
        logl=np.asarray([0.0], dtype=float),
        parameter_names=np.asarray(["b1"], dtype=str),
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "plot_bestfit_pgg_chain.py"),
            str(legacy_chain),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "missing required plotting metadata keys" in result.stderr
