from __future__ import annotations

from pathlib import Path

import numpy as np


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
        elif not np.allclose(k_mock, target_k, rtol=1.0e-10, atol=1.0e-12):
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
