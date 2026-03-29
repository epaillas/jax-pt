from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from classy import Class

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxpt import EFTBiasParams, GalaxyPowerSpectrumMultipolesTheory, PTSettings, PowerSpectrumTemplate


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

COSMOLOGY_SAMPLES = [
    ("fiducial", {}),
    ("high-omega_cdm", {"omega_cdm": 0.126}),
    ("tilted-ns", {"n_s": 0.9749, "A_s": 2.03e-9}),
]

NUISANCE_SAMPLES = [
    (
        "fiducial",
        EFTBiasParams(
            b1=2.0,
            b2=-1.0,
            bG2=0.1,
            bGamma3=-0.1,
            cs0=0.0,
            cs2=30.0,
            cs4=0.0,
            Pshot=3000.0,
            b4=10.0,
        ),
    ),
    (
        "high-bias",
        EFTBiasParams(
            b1=2.2,
            b2=-0.7,
            bG2=0.15,
            bGamma3=-0.05,
            cs0=5.0,
            cs2=36.0,
            cs4=4.0,
            Pshot=3000.0,
            b4=12.0,
        ),
    ),
    (
        "lower-shot-noise",
        EFTBiasParams(
            b1=1.85,
            b2=-1.2,
            bG2=0.05,
            bGamma3=-0.15,
            cs0=-5.0,
            cs2=24.0,
            cs4=-3.0,
            Pshot=2200.0,
            b4=8.0,
        ),
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show the object-oriented jaxpt API flow for MCMC-style multipole predictions with varying cosmology and nuisance parameters."
    )
    parser.add_argument("--z", type=float, default=0.5, help="Redshift at which to evaluate the predictions.")
    parser.add_argument("--kmin", type=float, default=0.01, help="Minimum evaluation k in 1/Mpc.")
    parser.add_argument("--kmax", type=float, default=0.2, help="Maximum evaluation k in 1/Mpc.")
    parser.add_argument("--nk", type=int, default=96, help="Number of evaluation-grid points.")
    parser.add_argument(
        "--support-nk",
        type=int,
        default=256,
        help="Number of support-grid points used when building LinearPowerInput.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("mcmc_multipole_parameter_sweep.png"),
        help="Path to the output PNG file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after writing the PNG.",
    )
    return parser


def build_classy(cosmology: dict[str, float], *, z: float) -> Class:
    cosmo = Class()
    cosmo.set({**cosmology, **PT_OPTIONS_NOIR, "z_pk": z})
    cosmo.compute()
    return cosmo


def predict_with_theory_api(
    cosmology: dict[str, float],
    nuisance: EFTBiasParams,
    *,
    z: float,
    support_k: np.ndarray,
    eval_k: np.ndarray,
    settings: PTSettings,
):
    # Sampler-facing workflow after the OOP redesign: build a cosmology-side
    # template once, feed it into a theory object, and evaluate that theory for
    # each nuisance-parameter sample.
    cosmo = build_classy(cosmology, z=z)
    template = PowerSpectrumTemplate.from_classy(cosmo, z=z, k=support_k, settings=settings)
    theory = GalaxyPowerSpectrumMultipolesTheory(template=template, k=eval_k)
    prediction = theory(nuisance)
    return prediction, template, theory


def fractional_delta(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (values - reference) / np.maximum(np.abs(reference), 1.0)


def format_summary_row(label: str, elapsed_s: float, prediction) -> str:
    return (
        f"{label:>18s}  "
        f"time={elapsed_s * 1.0e3:8.2f} ms  "
        f"P0[0]={float(np.asarray(prediction.p0)[0]): .6e}  "
        f"P2[0]={float(np.asarray(prediction.p2)[0]): .6e}  "
        f"P4[0]={float(np.asarray(prediction.p4)[0]): .6e}"
    )


def main() -> None:
    args = build_parser().parse_args()

    settings = PTSettings(ir_resummation=False)
    support_k = np.logspace(-5.0, 1.0, args.support_nk)
    eval_k = np.linspace(args.kmin, args.kmax, args.nk)

    fiducial_nuisance = NUISANCE_SAMPLES[0][1]

    cosmology_predictions: list[tuple[str, object]] = []
    cosmology_timings: list[tuple[str, float]] = []

    for label, overrides in COSMOLOGY_SAMPLES:
        t0 = time.perf_counter()
        prediction, _, _ = predict_with_theory_api(
            {**FIDUCIAL_COSMOLOGY, **overrides},
            fiducial_nuisance,
            z=args.z,
            support_k=support_k,
            eval_k=eval_k,
            settings=settings,
        )
        cosmology_timings.append((label, time.perf_counter() - t0))
        cosmology_predictions.append((label, prediction))

    # When cosmology is fixed, the expensive state is encapsulated in the
    # template/theory pair and only nuisance parameters vary between calls.
    fiducial_prediction, _, fiducial_theory = predict_with_theory_api(
        FIDUCIAL_COSMOLOGY,
        fiducial_nuisance,
        z=args.z,
        support_k=support_k,
        eval_k=eval_k,
        settings=settings,
    )
    nuisance_predictions: list[tuple[str, object]] = [("fiducial", fiducial_prediction)]
    nuisance_timings: list[tuple[str, float]] = [("fiducial", 0.0)]

    for label, params in NUISANCE_SAMPLES[1:]:
        t0 = time.perf_counter()
        prediction = fiducial_theory(params)
        nuisance_timings.append((label, time.perf_counter() - t0))
        nuisance_predictions.append((label, prediction))

    print("Cosmology-varying loop: rebuild classy, PowerSpectrumTemplate, and theory each sample")
    for (label, elapsed), (_, prediction) in zip(cosmology_timings, cosmology_predictions, strict=True):
        print(format_summary_row(label, elapsed, prediction))

    print()
    print("Nuisance-only loop: reuse fixed GalaxyPowerSpectrumMultipolesTheory and vary only nuisance parameters")
    for (label, elapsed), (_, prediction) in zip(nuisance_timings, nuisance_predictions, strict=True):
        print(format_summary_row(label, elapsed, prediction))

    fiducial_cosmology_prediction = cosmology_predictions[0][1]
    fiducial_nuisance_prediction = nuisance_predictions[0][1]

    figure, axes = plt.subplots(2, 3, figsize=(13.0, 7.0), sharex=True, constrained_layout=True)
    cosmology_series = [
        ("P0", np.asarray(fiducial_cosmology_prediction.p0), [np.asarray(pred.p0) for _, pred in cosmology_predictions]),
        ("P2", np.asarray(fiducial_cosmology_prediction.p2), [np.asarray(pred.p2) for _, pred in cosmology_predictions]),
        ("P4", np.asarray(fiducial_cosmology_prediction.p4), [np.asarray(pred.p4) for _, pred in cosmology_predictions]),
    ]
    nuisance_series = [
        ("P0", np.asarray(fiducial_nuisance_prediction.p0), [np.asarray(pred.p0) for _, pred in nuisance_predictions]),
        ("P2", np.asarray(fiducial_nuisance_prediction.p2), [np.asarray(pred.p2) for _, pred in nuisance_predictions]),
        ("P4", np.asarray(fiducial_nuisance_prediction.p4), [np.asarray(pred.p4) for _, pred in nuisance_predictions]),
    ]

    for column, (label, reference, values_list) in enumerate(cosmology_series):
        ax = axes[0, column]
        ax.axhline(0.0, color="black", lw=1.0, alpha=0.6)
        for (sample_label, _), values in zip(cosmology_predictions, values_list, strict=True):
            ax.plot(eval_k, fractional_delta(values, reference), lw=2.0, label=sample_label)
        ax.set_title(f"{label}: cosmology sweep")
        ax.set_ylabel(r"$\Delta P_\ell / P_{\ell,\mathrm{fid}}$")
        ax.grid(True, alpha=0.25)

    for column, (label, reference, values_list) in enumerate(nuisance_series):
        ax = axes[1, column]
        ax.axhline(0.0, color="black", lw=1.0, alpha=0.6)
        for (sample_label, _), values in zip(nuisance_predictions, values_list, strict=True):
            ax.plot(eval_k, fractional_delta(values, reference), lw=2.0, label=sample_label)
        ax.set_title(f"{label}: nuisance sweep")
        ax.set_xlabel("k [1/Mpc]")
        ax.set_ylabel(r"$\Delta P_\ell / P_{\ell,\mathrm{fid}}$")
        ax.grid(True, alpha=0.25)

    axes[0, 0].legend(loc="best")
    figure.suptitle(
        "Object-Oriented jaxpt MCMC Call Pattern\n"
        "Top: full cosmology updates, Bottom: nuisance-only updates with theory reuse"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print()
    print(args.output)

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
