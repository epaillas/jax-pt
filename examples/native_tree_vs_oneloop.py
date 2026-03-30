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

from jaxpt.bias import galaxy_real_spectrum
from jaxpt.config import PTSettings
from jaxpt.cosmology import build_linear_input_from_classy
from jaxpt.native import compute_basis


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

VALIDATED_EVAL_K = np.linspace(0.01, 0.2, 128)
LINEAR_SUPPORT_K = np.logspace(-5.0, 1.0, 256)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the native tree-level and one-loop jaxpt real-space galaxy power spectra."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("native_tree_vs_oneloop.png"),
        help="Path to the output PNG file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after writing the PNG.",
    )
    return parser


def _evaluate_realspace_spectrum(linear_input, eval_k: np.ndarray, settings: PTSettings) -> np.ndarray:
    basis = compute_basis(linear_input, settings=settings, k=eval_k)
    return np.asarray(
        galaxy_real_spectrum(
            basis,
            b1=2.0,
            b2=-1.0,
            bG2=0.1,
            bGamma3=-0.1,
            cs=0.0,
            cs0=0.0,
            Pshot=3000.0,
        )
    )


def main() -> None:
    args = build_parser().parse_args()

    z = 0.5
    eval_k = VALIDATED_EVAL_K
    linear_support_k = LINEAR_SUPPORT_K

    cosmo = Class()
    cosmo.set({**FIDUCIAL_COSMOLOGY, **PT_OPTIONS_NOIR, "z_pk": z})
    cosmo.compute()

    linear_input = build_linear_input_from_classy(cosmo, z=z, k=linear_support_k)
    tree_pk = _evaluate_realspace_spectrum(
        linear_input,
        eval_k,
        PTSettings(loop_order="tree", ir_resummation=False),
    )
    one_loop_pk = _evaluate_realspace_spectrum(
        linear_input,
        eval_k,
        PTSettings(loop_order="one_loop", ir_resummation=False),
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(eval_k, eval_k * tree_pk, label="jaxpt native tree", lw=2.0)
    ax.plot(eval_k, eval_k * one_loop_pk, label="jaxpt native one-loop", lw=2.0, ls="--")
    ax.set_xlabel("k [1/Mpc]")
    ax.set_ylabel(r"$k\,P_{gg}(k)$ [Mpc$^2$]")
    ax.set_title("Native Tree vs One-Loop Real-Space Galaxy Power Spectrum, z = 0.5")
    ax.grid(True, alpha=0.25)
    ax.legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(args.output)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
