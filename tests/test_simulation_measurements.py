from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from jaxpt.utils import cached_sample_covariance, covariance_errors, flatten_pgg_measurements, load_pgg_data_vector, load_pgg_mock_matrix, sample_covariance


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_VECTOR_PATH = REPO_ROOT / "data" / "data_vector" / "mesh2_spectrum_poles_c000_hod006.h5"
MOCK_DIR = REPO_ROOT / "data" / "for_covariance"


def _build_small_mock_directory(tmp_path: Path, n_mocks: int = 4) -> Path:
    directory = tmp_path / "mocks"
    directory.mkdir()
    for path in sorted(MOCK_DIR.glob("mesh2_spectrum_poles_ph*.h5"))[:n_mocks]:
        shutil.copy2(path, directory / path.name)
    return directory


def test_load_pgg_data_vector_returns_requested_multipoles() -> None:
    k, poles = load_pgg_data_vector(DATA_VECTOR_PATH, ells=(0, 2, 4), rebin=13, kmin=0.01, kmax=0.2)

    assert k.ndim == 1
    assert np.all(np.isfinite(k))
    assert np.all((k >= 0.01) & (k <= 0.2))
    assert set(poles) == {0, 2, 4}
    for values in poles.values():
        assert values.shape == k.shape
        assert np.all(np.isfinite(values))


def test_load_pgg_mock_matrix_matches_flattened_layout(tmp_path: Path) -> None:
    k_data, poles = load_pgg_data_vector(DATA_VECTOR_PATH, ells=(0, 2, 4), rebin=13, kmin=0.01, kmax=0.2)
    mock_dir = _build_small_mock_directory(tmp_path)

    k_mock, matrix = load_pgg_mock_matrix(mock_dir, ells=(0, 2, 4), rebin=13, k_data=k_data, kmin=0.01, kmax=0.2)

    assert np.allclose(k_mock, k_data)
    assert matrix.shape == (4, len(k_data) * 3)
    assert np.all(np.isfinite(matrix))
    assert flatten_pgg_measurements(poles, ells=(0, 2, 4)).shape == (len(k_data) * 3,)


def test_sample_covariance_and_errors_have_expected_shapes(tmp_path: Path) -> None:
    k_data, _ = load_pgg_data_vector(DATA_VECTOR_PATH, ells=(0, 2, 4), rebin=13, kmin=0.01, kmax=0.2)
    mock_dir = _build_small_mock_directory(tmp_path)
    _, matrix = load_pgg_mock_matrix(mock_dir, ells=(0, 2, 4), rebin=13, k_data=k_data, kmin=0.01, kmax=0.2)

    covariance = sample_covariance(matrix)
    errors = covariance_errors(covariance)

    n_data = len(k_data) * 3
    assert covariance.shape == (n_data, n_data)
    assert errors.shape == (n_data,)
    assert np.all(np.isfinite(errors))
    assert np.all(errors >= 0.0)


def test_cached_sample_covariance_reuses_hashed_artifact(tmp_path: Path) -> None:
    k_data, _ = load_pgg_data_vector(DATA_VECTOR_PATH, ells=(0, 2, 4), rebin=13, kmin=0.01, kmax=0.2)
    mock_dir = _build_small_mock_directory(tmp_path)
    cache_dir = tmp_path / "cache"

    k_first, covariance_first, cache_path = cached_sample_covariance(
        mock_dir,
        ells=(0, 2, 4),
        rebin=13,
        k_data=k_data,
        kmin=0.01,
        kmax=0.2,
        cache_dir=cache_dir,
    )
    assert cache_path.exists()
    mtime_first = cache_path.stat().st_mtime_ns

    k_second, covariance_second, cache_path_second = cached_sample_covariance(
        mock_dir,
        ells=(0, 2, 4),
        rebin=13,
        k_data=k_data,
        kmin=0.01,
        kmax=0.2,
        cache_dir=cache_dir,
    )

    assert cache_path_second == cache_path
    assert cache_path_second.stat().st_mtime_ns == mtime_first
    np.testing.assert_allclose(k_second, k_first, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(covariance_second, covariance_first, rtol=0.0, atol=0.0)


def test_plot_simulation_script_writes_png(tmp_path: Path) -> None:
    mock_dir = _build_small_mock_directory(tmp_path)
    output_path = tmp_path / "measurements.png"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "plot_simulation_pgg_measurements.py"),
            "--data",
            str(DATA_VECTOR_PATH),
            "--mocks",
            str(mock_dir),
            "--kmin",
            "0.01",
            "--kmax",
            "0.2",
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "covariance_shape" in result.stdout
    assert "covariance_cache:" in result.stdout
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_simulation_script_writes_png_with_default_k_range(tmp_path: Path) -> None:
    mock_dir = _build_small_mock_directory(tmp_path)
    output_path = tmp_path / "measurements_default.png"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "plot_simulation_pgg_measurements.py"),
            "--data",
            str(DATA_VECTOR_PATH),
            "--mocks",
            str(mock_dir),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "n_k" in result.stdout
    assert "covariance_cache:" in result.stdout
    assert output_path.exists()
    assert output_path.stat().st_size > 0
