from __future__ import annotations

import argparse
import math
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

from compare_taylor_emulator import _build_theory, _fractional_residual, _load_config
from jaxpt import TaylorEmulator


DEFAULT_STEP_MULTIPLE = 3.0
DEFAULT_N_SAMPLES = 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan one-at-a-time Taylor emulator errors against direct native jaxpt multipole theory."
    )
    parser.add_argument("emulator", type=Path, help="Path to a serialized Taylor emulator `.npz` file.")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME",
        help="Restrict the scan to this emulated parameter. Repeat for multiple parameters.",
    )
    parser.add_argument(
        "--step-multiple",
        type=float,
        default=DEFAULT_STEP_MULTIPLE,
        help="Largest scan offset in units of the serialized emulator step size.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Total number of nonzero scan samples per parameter. Must be an even integer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the output PNG. Defaults to `<emulator_stem>_error_scan.png` next to the emulator file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after writing the PNG.",
    )
    return parser


def _default_output_path(emulator_path: Path) -> Path:
    return emulator_path.with_name(f"{emulator_path.stem}_error_scan.png")


def _resolve_scan_parameters(emulator: TaylorEmulator, requested: list[str]) -> list[str]:
    available = list(emulator.param_names)
    if not requested:
        return available
    invalid = sorted(set(requested) - set(available))
    if invalid:
        raise ValueError("Requested scan parameters are not emulated: " + ", ".join(invalid))
    return [name for name in available if name in set(requested)]


def _scan_multiples(step_multiple: float, n_samples: int) -> np.ndarray:
    if step_multiple <= 0.0:
        raise ValueError("--step-multiple must be positive.")
    if n_samples < 2 or n_samples % 2 != 0:
        raise ValueError("--n-samples must be an even integer greater than or equal to 2.")
    grid = np.linspace(-float(step_multiple), float(step_multiple), n_samples + 1, dtype=float)
    return grid[~np.isclose(grid, 0.0, rtol=0.0, atol=0.0)]


def _envelope_curves(
    emulator: TaylorEmulator,
    theory,
    *,
    parameter: str,
    scan_values: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    query = dict(emulator.fiducial)
    envelopes = {
        "P0": None,
        "P2": None,
        "P4": None,
    }

    for value in scan_values:
        query[parameter] = float(value)
        emulator_prediction = emulator.predict(query)
        direct_prediction = theory(query)
        residuals = {
            "P0": np.abs(_fractional_residual(np.asarray(emulator_prediction.p0), np.asarray(direct_prediction.p0))),
            "P2": np.abs(_fractional_residual(np.asarray(emulator_prediction.p2), np.asarray(direct_prediction.p2))),
            "P4": np.abs(_fractional_residual(np.asarray(emulator_prediction.p4), np.asarray(direct_prediction.p4))),
        }
        for label, residual in residuals.items():
            current = envelopes[label]
            envelopes[label] = residual if current is None else np.maximum(current, residual)

    worst_case = {label: float(np.max(curve)) for label, curve in envelopes.items() if curve is not None}
    return np.asarray(theory.k, dtype=float), {label: np.asarray(curve) for label, curve in envelopes.items()}, worst_case


def _plot_error_scan(
    output_path: Path,
    *,
    k: np.ndarray,
    results: list[tuple[str, dict[str, np.ndarray]]],
    show: bool,
) -> None:
    n_panels = len(results)
    ncols = min(3, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 3.4 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(-1)
    colors = {"P0": "C0", "P2": "C1", "P4": "C2"}

    for ax, (parameter, curves) in zip(axes_array, results, strict=True):
        for label in ("P0", "P2", "P4"):
            ax.plot(k, curves[label], lw=2.0, color=colors[label], label=label)
        ax.set_title(parameter)
        ax.set_xlabel("k [1/Mpc]")
        ax.set_ylabel(r"$\max |\Delta P_\ell / P_\ell|$")
        ax.grid(True, alpha=0.25)

    for ax in axes_array[n_panels:]:
        ax.set_visible(False)

    axes_array[0].legend(loc="best")
    fig.suptitle("Taylor Emulator Relative Error Envelopes by Parameter")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(f"plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    config = _load_config(args.emulator)
    metadata = dict(config.get("metadata", {}))

    base_emulator = TaylorEmulator.load(args.emulator)
    theory = _build_theory(base_emulator, metadata)
    emulator = TaylorEmulator.load(args.emulator, theory_fn=theory)

    scan_params = _resolve_scan_parameters(emulator, [str(name) for name in args.param])
    scan_multiples = _scan_multiples(args.step_multiple, args.n_samples)

    k = None
    panel_results: list[tuple[str, dict[str, np.ndarray]]] = []
    summary_rows: list[str] = []

    print("Taylor emulator error scan")
    print(f"emulator: {args.emulator}")
    print("parameters: " + ", ".join(scan_params))
    print("scan multiples: " + ", ".join(f"{value:g}" for value in scan_multiples))

    for parameter in scan_params:
        step_size = float(emulator._step_sizes[parameter])
        scan_values = emulator.fiducial[parameter] + scan_multiples * step_size
        parameter_k, curves, worst_case = _envelope_curves(
            emulator,
            theory,
            parameter=parameter,
            scan_values=scan_values,
        )
        if k is None:
            k = parameter_k
        panel_results.append((parameter, curves))
        summary_rows.append(
            f"{parameter}: "
            f"step={step_size:.6e}, "
            f"max|dP0/P0|={worst_case['P0']:.6e}, "
            f"max|dP2/P2|={worst_case['P2']:.6e}, "
            f"max|dP4/P4|={worst_case['P4']:.6e}"
        )

    for row in summary_rows:
        print(row)

    assert k is not None
    output_path = args.output if args.output is not None else _default_output_path(args.emulator)
    _plot_error_scan(output_path, k=k, results=panel_results, show=args.show)


if __name__ == "__main__":
    main()
