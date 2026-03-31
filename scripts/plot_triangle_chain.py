from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import numpy as np


_LATEX = {
    "logA": r"\log(10^{10} A_s)",
    "A_s": r"A_s",
    "omega_cdm": r"\omega_{\rm cdm}",
    "omega_b": r"\omega_b",
    "h": r"h",
    "n_s": r"n_s",
    "b1": r"b_1",
    "b2": r"b_2",
    "bG2": r"b_{G_2}",
    "bGamma3": r"b_{\Gamma_3}",
    "cs0": r"c_{s,0}",
    "cs2": r"c_{s,2}",
    "cs4": r"c_{s,4}",
    "Pshot": r"P_{\rm shot}",
    "b4": r"b_4",
}


def _labels_from_names(param_names: list[str]) -> list[str]:
    return [_LATEX.get(name, name) for name in param_names]


def _default_output_path(chain_path: Path) -> Path:
    return chain_path.with_name(f"{chain_path.stem}_triangle.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Triangle plot from a jaxpt PocoMC posterior chain.")
    parser.add_argument("chain", type=Path, help="Path to a self-describing posterior `.npz` file.")
    parser.add_argument("--params", nargs="+", default=None, metavar="NAME", help="Subset of parameter names to plot.")
    parser.add_argument("--output", type=Path, default=None, help="Path to the output PNG.")
    parser.add_argument("--dpi", type=int, default=150, help="Output figure DPI.")
    return parser


def main() -> None:
    try:
        from getdist import MCSamples, plots
    except ImportError as exc:
        raise ImportError("plot_triangle_chain.py requires the optional 'getdist' package.") from exc

    args = build_parser().parse_args()
    output_path = args.output if args.output is not None else _default_output_path(args.chain)

    with np.load(args.chain, allow_pickle=False) as data:
        for key in ("samples", "weights", "parameter_names"):
            if key not in data:
                raise ValueError(f"Chain file is missing required key '{key}'.")
        samples = np.asarray(data["samples"], dtype=float)
        weights = np.asarray(data["weights"], dtype=float)
        param_names = [str(name) for name in np.asarray(data["parameter_names"], dtype=str)]

    if args.params is not None:
        indices = [param_names.index(name) for name in args.params]
        param_names = [param_names[index] for index in indices]
        samples = samples[:, indices]

    labels = _labels_from_names(param_names)
    mcs = MCSamples(samples=samples, weights=weights, names=param_names, labels=labels, label="PocoMC posterior")

    plotter = plots.get_subplot_plotter()
    plotter.settings.axes_fontsize = 10
    plotter.settings.legend_fontsize = 11
    plotter.triangle_plot([mcs], filled=True, title_limit=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"plot: {output_path}")


if __name__ == "__main__":
    main()
