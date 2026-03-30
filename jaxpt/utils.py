from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_cache_dir(subdir: str | Path) -> Path:
    """Return a subdirectory under the repository-root `.cache` directory."""
    path = _REPO_ROOT / ".cache" / Path(subdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _freeze_cache_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(name): _freeze_cache_value(item) for name, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_freeze_cache_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).reshape(-1).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        try:
            return str(value.resolve())
        except OSError:
            return str(value)
    return value


def covariance_cache_key(
    *,
    directory: str | Path,
    ells: tuple[int, ...] = (0, 2, 4),
    rebin: int = 13,
    k_data: np.ndarray | None = None,
    kmin: float = 0.0,
    kmax: float = np.inf,
    pattern: str = "mesh2_spectrum_poles_ph*.h5",
) -> str:
    """Return a deterministic hash key for a mock-matrix covariance configuration."""
    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise ValueError(f"No mock files matched pattern '{pattern}' in {directory}.")
    payload = {
        "directory": Path(directory),
        "pattern": pattern,
        "ells": [int(ell) for ell in ells],
        "rebin": int(rebin),
        "k_data": None if k_data is None else np.asarray(k_data, dtype=float),
        "kmin": float(kmin),
        "kmax": float(kmax),
        "files": [
            {
                "path": path,
                "mtime_ns": path.stat().st_mtime_ns,
                "size": path.stat().st_size,
            }
            for path in paths
        ],
    }
    encoded = json.dumps(_freeze_cache_value(payload), sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def cached_sample_covariance(
    directory: str | Path,
    *,
    ells: tuple[int, ...] = (0, 2, 4),
    rebin: int = 13,
    k_data: np.ndarray | None = None,
    kmin: float = 0.0,
    kmax: float = np.inf,
    pattern: str = "mesh2_spectrum_poles_ph*.h5",
    cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, Path]:
    """Load or build a hashed covariance artifact from the mock HDF5 inputs."""
    resolved_cache_dir = repo_cache_dir("covariances") if cache_dir is None else Path(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = covariance_cache_key(
        directory=directory,
        ells=ells,
        rebin=rebin,
        k_data=k_data,
        kmin=kmin,
        kmax=kmax,
        pattern=pattern,
    )
    cache_path = resolved_cache_dir / f"covariance_{cache_key}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            return (
                np.asarray(data["k"], dtype=float),
                np.asarray(data["covariance"], dtype=float),
                cache_path,
            )

    k_resolved, mock_matrix = load_pgg_mock_matrix(
        directory,
        ells=ells,
        rebin=rebin,
        k_data=k_data,
        kmin=kmin,
        kmax=kmax,
        pattern=pattern,
    )
    covariance = sample_covariance(mock_matrix)
    np.savez(
        cache_path,
        k=np.asarray(k_resolved, dtype=float),
        covariance=np.asarray(covariance, dtype=float),
    )
    return np.asarray(k_resolved, dtype=float), np.asarray(covariance, dtype=float), cache_path


def load_pgg_data_vector(
    path: str | Path,
    ells: tuple[int, ...] = (0, 2, 4),
    rebin: int = 13,
    kmin: float = 0.0,
    kmax: float = np.inf,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Load galaxy auto-power spectrum multipoles from a jaxpower HDF5 file."""
    from jaxpower import read

    data = read(str(path)).select(k=slice(0, None, rebin))
    if kmin > 0.0 or kmax < np.inf:
        data = data.select(k=(kmin, kmax))

    k = None
    poles: dict[int, np.ndarray] = {}
    for ell in ells:
        leaf = data.get(ell)
        if k is None:
            k = np.asarray(leaf.coords("k"), dtype=float)
        poles[int(ell)] = np.asarray(leaf.value(), dtype=float)

    assert k is not None
    return np.asarray(k, dtype=float), poles


def flatten_pgg_measurements(
    poles: dict[int, np.ndarray],
    ells: tuple[int, ...] = (0, 2, 4),
) -> np.ndarray:
    """Flatten multipoles in ell-major order."""
    return np.concatenate([np.asarray(poles[ell], dtype=float) for ell in ells])


def load_pgg_mock_matrix(
    directory: str | Path,
    ells: tuple[int, ...] = (0, 2, 4),
    rebin: int = 13,
    k_data: np.ndarray | None = None,
    kmin: float = 0.0,
    kmax: float = np.inf,
    pattern: str = "mesh2_spectrum_poles_ph*.h5",
) -> tuple[np.ndarray, np.ndarray]:
    """Load mock realizations into a row-stacked ell-major matrix."""
    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise ValueError(f"No mock files matched pattern '{pattern}' in {directory}.")

    target_k = None if k_data is None else np.asarray(k_data, dtype=float)
    rows: list[np.ndarray] = []
    for path in paths:
        k_mock, poles = load_pgg_data_vector(path, ells=ells, rebin=rebin, kmin=kmin, kmax=kmax)
        if target_k is None:
            target_k = np.asarray(k_mock, dtype=float)
        else:
            same_grid = k_mock.shape == target_k.shape and np.allclose(k_mock, target_k, rtol=1.0e-10, atol=1.0e-12)
            if same_grid:
                rows.append(flatten_pgg_measurements(poles, ells=ells))
                continue
            poles = {
                ell: np.interp(target_k, np.asarray(k_mock, dtype=float), np.asarray(values, dtype=float))
                for ell, values in poles.items()
            }
        rows.append(flatten_pgg_measurements(poles, ells=ells))

    assert target_k is not None
    return np.asarray(target_k, dtype=float), np.asarray(rows, dtype=float)


def sample_covariance(mock_matrix: np.ndarray) -> np.ndarray:
    """Return the sample covariance matrix for a mock data matrix."""
    matrix = np.asarray(mock_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("mock_matrix must be a two-dimensional array.")
    return np.cov(matrix.T)


def covariance_errors(covariance: np.ndarray) -> np.ndarray:
    """Return standard deviations from the covariance diagonal."""
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    return np.sqrt(np.diag(cov))
