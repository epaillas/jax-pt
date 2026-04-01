from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from classy import Class

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxpt import PTSettings
from jaxpt.theories import PowerSpectrumTemplate, QuantileGalaxyPowerSpectrumMultipolesTheory


FIDUCIAL_COSMOLOGY = {
    "A_s": 2.089e-9,
    "n_s": 0.9649,
    "tau_reio": 0.052,
    "omega_b": 0.02237,
    "omega_cdm": 0.12,
    "h": 0.6736,
    "YHe": 0.2425,
    "N_ur": 2.0328,
    "N_ncdm": 1,
    "m_ncdm": 0.06,
}
PT_OPTIONS_NOIR = {
    "output": "mPk",
    "non linear": "PT",
    "IR resummation": "No",
    "Bias tracers": "Yes",
    "cb": "Yes",
    "RSD": "Yes",
}

ELL_LABELS = (0, 2, 4)
QUANTILE_COLORS = ("#274c77", "#6096ba", "#a3cef1", "#8b8c89", "#bc4749")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot default tree-level density-split quantile-galaxy multipoles.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("density_split_multipoles.png"),
        help="Path to the output PNG file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after writing the PNG.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    k = np.linspace(0.01, 0.2, 128)
    z = 0.5
    cosmo = Class()
    cosmo.set({**FIDUCIAL_COSMOLOGY, **PT_OPTIONS_NOIR, "z_pk": z})
    cosmo.compute()

    template = PowerSpectrumTemplate(
        cosmo,
        z=z,
        settings=PTSettings(loop_order="tree", ir_resummation=False),
    )
    theory = QuantileGalaxyPowerSpectrumMultipolesTheory(template=template, k=k)
    poles = theory()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, constrained_layout=True)
    for ell_index, (ax, ell) in enumerate(zip(axes, ELL_LABELS, strict=True)):
        for quantile_index, color in enumerate(QUANTILE_COLORS):
            ax.plot(
                k,
                poles[quantile_index, ell_index, :],
                lw=2.0,
                color=color,
                label=f"Q{quantile_index + 1}",
            )
        ax.set_title(rf"$\ell = {ell}$")
        ax.set_xlabel(r"$k$ [1/Mpc]")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$P^{qg}_{\ell}(k)$ [Mpc$^3$]")
    axes[-1].legend(frameon=False, loc="best")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(args.output)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
