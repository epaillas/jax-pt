from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from classy import Class

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxpt import PTSettings, build_linear_input_from_classy, build_native_realspace_predictor


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

DEFAULT_PARAMS = {
    "b1": 2.0,
    "b2": -1.0,
    "bG2": 0.1,
    "bGamma3": -0.1,
    "cs": 0.0,
    "cs0": 0.0,
    "Pshot": 3000.0,
}


@dataclass(frozen=True, slots=True)
class TimingSummary:
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    p95_ms: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark real-space galaxy power-spectrum prediction latency for jaxpt and CLASS-PT."
    )
    parser.add_argument("--z", type=float, default=0.5, help="Redshift to benchmark.")
    parser.add_argument("--kmin", type=float, default=0.01, help="Minimum evaluation k in 1/Mpc.")
    parser.add_argument("--kmax", type=float, default=0.2, help="Maximum evaluation k in 1/Mpc.")
    parser.add_argument("--nk", type=int, default=128, help="Default number of output k samples.")
    parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="Evaluation-grid sizes for the scaling benchmark.",
    )
    parser.add_argument("--support-nk", type=int, default=256, help="Number of linear-theory support-grid points.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations before steady-state timing.")
    parser.add_argument("--repeat", type=int, default=30, help="Measured iterations per benchmark point.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("benchmark_outputs"),
        help="Directory where figures and JSON will be written.",
    )
    return parser


def summarize(samples: list[float]) -> TimingSummary:
    arr = np.asarray(samples, dtype=float) * 1.0e3
    return TimingSummary(
        mean_ms=float(arr.mean()),
        median_ms=float(np.median(arr)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        std_ms=float(arr.std(ddof=0)),
        p95_ms=float(np.percentile(arr, 95.0)),
    )


def time_callable(fn, *, warmup: int, repeat: int) -> tuple[float, TimingSummary]:
    t0 = time.perf_counter()
    fn()
    cold_ms = (time.perf_counter() - t0) * 1.0e3

    for _ in range(warmup):
        fn()

    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return cold_ms, summarize(samples)


def make_latency_figure(output_path: Path, cold: dict[str, float], steady: dict[str, TimingSummary]) -> None:
    labels = ["jaxpt native", "CLASS-PT direct"]
    means = [steady["native"].mean_ms, steady["classpt_direct"].mean_ms]
    stds = [steady["native"].std_ms, steady["classpt_direct"].std_ms]
    cold_values = [cold["native"], cold["classpt_direct"]]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.bar(x - width / 2.0, means, width=width, yerr=stds, capsize=4, label="steady-state mean ± std")
    ax.bar(x + width / 2.0, cold_values, width=width, alpha=0.7, label="first call")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Latency [ms]")
    ax.set_title("Real-Space Galaxy Power Spectrum Prediction Latency")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_scaling_figure(output_path: Path, scaling_rows: list[dict[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)

    for key, label in (
        ("native_mean_ms", "jaxpt native"),
        ("classpt_direct_mean_ms", "CLASS-PT direct"),
    ):
        ax.plot(
            [row["nk"] for row in scaling_rows],
            [row[key] for row in scaling_rows],
            marker="o",
            lw=2.0,
            label=label,
        )

    ax.set_xlabel("Number of evaluation k samples")
    ax.set_ylabel("Steady-state mean latency [ms]")
    ax.set_title("Prediction Latency vs Output Grid Size")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def benchmark_point(
    cosmo: Class,
    linear_input,
    *,
    z: float,
    eval_k: np.ndarray,
    settings: PTSettings,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    native_predict = build_native_realspace_predictor(linear_input, settings=settings, k=eval_k)
    native_fn = lambda: np.asarray(
        native_predict(
            b1=DEFAULT_PARAMS["b1"],
            b2=DEFAULT_PARAMS["b2"],
            bG2=DEFAULT_PARAMS["bG2"],
            bGamma3=DEFAULT_PARAMS["bGamma3"],
            cs=DEFAULT_PARAMS["cs"],
            cs0=DEFAULT_PARAMS["cs0"],
            Pshot=DEFAULT_PARAMS["Pshot"],
        )
    )
    classpt_direct_fn = lambda: (
        cosmo.initialize_output(eval_k, z, len(eval_k)),
        np.asarray(
            cosmo.pk_gg_real(
                DEFAULT_PARAMS["b1"],
                DEFAULT_PARAMS["b2"],
                DEFAULT_PARAMS["bG2"],
                DEFAULT_PARAMS["bGamma3"],
                DEFAULT_PARAMS["cs"],
                DEFAULT_PARAMS["cs0"],
                DEFAULT_PARAMS["Pshot"],
            )
        ),
    )
    native_cold_ms, native_steady = time_callable(native_fn, warmup=warmup, repeat=repeat)
    classpt_direct_cold_ms, classpt_direct_steady = time_callable(classpt_direct_fn, warmup=warmup, repeat=repeat)

    return {
        "cold_ms": {
            "native": native_cold_ms,
            "classpt_direct": classpt_direct_cold_ms,
        },
        "steady": {
            "native": native_steady,
            "classpt_direct": classpt_direct_steady,
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = PTSettings(ir_resummation=False)

    cosmo = Class()
    cosmo.set({**FIDUCIAL_COSMOLOGY, **PT_OPTIONS_NOIR, "z_pk": args.z})
    cosmo.compute()

    support_k = np.logspace(-5.0, 1.0, args.support_nk)
    linear_input = build_linear_input_from_classy(cosmo, z=args.z, k=support_k)
    eval_k = np.linspace(args.kmin, args.kmax, args.nk)

    benchmark = benchmark_point(
        cosmo,
        linear_input,
        z=args.z,
        eval_k=eval_k,
        settings=settings,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    scaling_rows: list[dict[str, float]] = []
    for nk in args.grid_sizes:
        grid = np.linspace(args.kmin, args.kmax, nk)
        result = benchmark_point(
            cosmo,
            linear_input,
            z=args.z,
            eval_k=grid,
            settings=settings,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        scaling_rows.append(
            {
                "nk": nk,
                "native_mean_ms": result["steady"]["native"].mean_ms,
                "classpt_direct_mean_ms": result["steady"]["classpt_direct"].mean_ms,
            }
        )

    summary_payload = {
        "config": {
            "z": args.z,
            "kmin": args.kmin,
            "kmax": args.kmax,
            "nk": args.nk,
            "support_nk": args.support_nk,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "grid_sizes": args.grid_sizes,
            "params": DEFAULT_PARAMS,
        },
        "benchmark_point": {
            "cold_ms": benchmark["cold_ms"],
            "steady": {name: asdict(stats) for name, stats in benchmark["steady"].items()},
        },
        "scaling": scaling_rows,
    }

    summary_path = output_dir / "realspace_prediction_benchmark.json"
    latency_path = output_dir / "realspace_prediction_latency.png"
    scaling_path = output_dir / "realspace_prediction_scaling.png"

    summary_path.write_text(json.dumps(summary_payload, indent=2))
    make_latency_figure(latency_path, benchmark["cold_ms"], benchmark["steady"])
    make_scaling_figure(scaling_path, scaling_rows)

    print(json.dumps(summary_payload["benchmark_point"], indent=2))
    print(summary_path)
    print(latency_path)
    print(scaling_path)


if __name__ == "__main__":
    main()
