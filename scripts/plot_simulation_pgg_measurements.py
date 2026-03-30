from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from jaxpt.utils import covariance_errors, flatten_pgg_measurements, load_pgg_data_vector, load_pgg_mock_matrix, sample_covariance


DEFAULT_DATA_PATH = ROOT / "data" / "data_vector" / "mesh2_spectrum_poles_c000_hod006.h5"
DEFAULT_MOCK_DIR = ROOT / "data" / "for_covariance"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load simulation Pgg measurements, build a mock covariance, and plot the data vector with diagonal errors."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the mean Pgg measurement HDF5 file.")
    parser.add_argument("--mocks", type=Path, default=DEFAULT_MOCK_DIR, help="Directory containing mock realization HDF5 files.")
    parser.add_argument(
        "--ells",
        type=int,
        nargs="+",
        default=[0, 2, 4],
        help="Multipoles to load and plot in ell-major order.",
    )
    parser.add_argument("--rebin", type=int, default=13, help="Rebinning factor passed to the measurement readers.")
    parser.add_argument("--kmin", type=float, default=0.0, help="Minimum k included in the measurement.")
    parser.add_argument("--kmax", type=float, default=float("inf"), help="Maximum k included in the measurement.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "scripts" / "simulation_pgg_measurements.png",
        help="Path to the output PNG.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot interactively after writing the PNG.")
    return parser


def _split_errors(errors: np.ndarray, nk: int, ells: tuple[int, ...]) -> dict[int, np.ndarray]:
    return {
        ell: np.asarray(errors[index * nk : (index + 1) * nk], dtype=float)
        for index, ell in enumerate(ells)
    }


def _plot_measurements(
    output_path: Path,
    *,
    k: np.ndarray,
    poles: dict[int, np.ndarray],
    pole_errors: dict[int, np.ndarray],
    ells: tuple[int, ...],
    show: bool,
) -> None:
    fig, axes = plt.subplots(
        len(ells),
        1,
        figsize=(8.0, 2.8 * len(ells)),
        sharex=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)

    for ax, ell in zip(axes_array, ells):
        ax.errorbar(
            k,
            np.asarray(poles[ell], dtype=float),
            yerr=np.asarray(pole_errors[ell], dtype=float),
            fmt="o",
            ms=3.5,
            lw=1.2,
            capsize=2.0,
        )
        ax.set_title(f"P{ell}")
        ax.set_ylabel(rf"$P_{{{ell}}}(k)$")
        ax.grid(True, alpha=0.25)

    axes_array[-1].set_xlabel("k [1/Mpc]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(f"plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    ells = tuple(int(ell) for ell in args.ells)

    k, poles = load_pgg_data_vector(args.data, ells=ells, rebin=args.rebin, kmin=args.kmin, kmax=args.kmax)
    _, mock_matrix = load_pgg_mock_matrix(
        args.mocks,
        ells=ells,
        rebin=args.rebin,
        k_data=k,
        kmin=args.kmin,
        kmax=args.kmax,
    )
    covariance = sample_covariance(mock_matrix)
    errors = covariance_errors(covariance)
    flattened = flatten_pgg_measurements(poles, ells=ells)
    pole_errors = _split_errors(errors, nk=len(k), ells=ells)

    print("Simulation Pgg measurements")
    print(f"data: {args.data}")
    print(f"mocks: {args.mocks}")
    print(f"ells: {', '.join(str(ell) for ell in ells)}")
    print(f"n_mocks: {mock_matrix.shape[0]}")
    print(f"n_k: {len(k)}")
    print(f"data_vector_length: {flattened.size}")
    print(f"covariance_shape: {covariance.shape}")

    _plot_measurements(args.output, k=k, poles=poles, pole_errors=pole_errors, ells=ells, show=args.show)


if __name__ == "__main__":
    main()
