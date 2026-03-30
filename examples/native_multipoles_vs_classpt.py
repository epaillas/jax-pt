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

from jaxpt.config import PTSettings
from jaxpt.theories import (
    ClassPTGalaxyPowerSpectrumMultipolesTheory,
    GalaxyPowerSpectrumMultipolesTheory,
    PowerSpectrumTemplate,
)


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
    "output": "mTk,mPk",
    "non linear": "PT",
    "IR resummation": "No",
    "Bias tracers": "Yes",
    "cb": "Yes",
    "RSD": "Yes",
}

EVAL_K = np.linspace(0.01, 0.2, 128)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare native jaxpt one-loop galaxy multipoles against direct CLASS-PT multipoles."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("native_multipoles_vs_classpt.png"),
        help="Path to the output PNG file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after writing the PNG.",
    )
    return parser


def _fractional_residual(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (prediction - reference) / np.maximum(np.abs(reference), 1.0)


def main() -> None:
    args = build_parser().parse_args()

    z = 0.5

    cosmo = Class()
    cosmo.set({**FIDUCIAL_COSMOLOGY, **PT_OPTIONS_NOIR, "z_pk": z})
    cosmo.compute()

    # Use the CLASS-PT internal tree-basis spectrum here so this example targets
    # backend parity rather than the default native `pk_lin` input convention.
    native_settings = PTSettings(ir_resummation=False)
    native_template = PowerSpectrumTemplate(cosmo, z=z, settings=native_settings, input_recipe="classpt_fftlog_grid_parity")
    native_theory = GalaxyPowerSpectrumMultipolesTheory(template=native_template, k=EVAL_K)
    native = native_theory()

    classpt_template = PowerSpectrumTemplate(cosmo, z=z, settings=PTSettings(ir_resummation=False))
    classpt_theory = ClassPTGalaxyPowerSpectrumMultipolesTheory(template=classpt_template, k=EVAL_K)
    classpt = classpt_theory()

    spectra = [
        ("P0", np.asarray(native.p0), np.asarray(classpt.p0)),
        ("P2", np.asarray(native.p2), np.asarray(classpt.p2)),
        ("P4", np.asarray(native.p4), np.asarray(classpt.p4)),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.5, 6.5),
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.2]},
    )

    for column, (label, native_values, classpt_values) in enumerate(spectra):
        ax_top = axes[0, column]
        ax_bottom = axes[1, column]

        ax_top.plot(EVAL_K, EVAL_K * native_values, lw=2.0, label="jaxpt")
        ax_top.plot(EVAL_K, EVAL_K * classpt_values, lw=2.0, ls="--", label="CLASS-PT")
        ax_top.set_title(label)
        ax_top.set_ylabel(r"$k\,P_\ell(k)$")
        ax_top.grid(True, alpha=0.25)

        residual = _fractional_residual(native_values, classpt_values)
        ax_bottom.axhline(0.0, color="black", lw=1.0, alpha=0.6)
        ax_bottom.plot(EVAL_K, residual, lw=1.8)
        ax_bottom.set_xlabel("k [1/Mpc]")
        ax_bottom.set_ylabel(r"$\Delta / P$")
        ax_bottom.grid(True, alpha=0.25)

        max_abs = float(np.max(np.abs(residual)))
        limit = max(1.0e-5, 1.15 * max_abs)
        ax_bottom.set_ylim(-limit, limit)

    axes[0, 0].legend(loc="best")
    fig.suptitle("Native One-Loop Galaxy Multipoles vs Direct CLASS-PT, z = 0.5")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(args.output)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
