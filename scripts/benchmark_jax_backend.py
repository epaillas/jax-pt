from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import importlib.metadata as md
import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxpt import LinearPowerInput, PTSettings, build_realspace_predictor
from jaxpt.theories import GalaxyPowerSpectrumMultipolesTheory, PowerSpectrumTemplate


DEFAULT_NUISANCE = {
    "b1": 2.0,
    "b2": -1.0,
    "bG2": 0.1,
    "bGamma3": -0.1,
    "cs": 0.0,
    "cs0": 0.0,
    "cs2": 0.0,
    "cs4": 0.0,
    "Pshot": 3000.0,
    "b4": 0.0,
}


@dataclass(frozen=True, slots=True)
class TimingSummary:
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark native jaxpt prediction kernels on the active JAX backend.")
    parser.add_argument("--nk", type=int, default=64, help="Number of output k samples.")
    parser.add_argument("--support-nk", type=int, default=256, help="Number of support k samples in the synthetic linear input.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations after the first compile call.")
    parser.add_argument("--repeat", type=int, default=30, help="Measured iterations per benchmark target.")
    parser.add_argument("--z", type=float, default=0.5, help="Redshift used in the synthetic linear input.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path for machine-readable benchmark output.")
    parser.add_argument("--require-gpu", action="store_true", help="Exit with a non-zero code unless a non-CPU backend is visible.")
    return parser


def make_linear_input(*, z: float, support_nk: int) -> LinearPowerInput:
    support_k = np.logspace(-4.0, 0.5, support_nk)
    pk_linear = 2.5e4 * (support_k / 0.05) ** -1.2 * np.exp(-(support_k / 0.7) ** 1.4)
    return LinearPowerInput(
        k=support_k,
        pk_linear=pk_linear,
        z=z,
        growth_factor=0.76,
        growth_rate=0.81,
        h=0.6736,
    )


def block_tree(value: Any) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, dict):
        for item in value.values():
            block_tree(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            block_tree(item)
        return
    for attr in ("p0", "p2", "p4"):
        if hasattr(value, attr):
            block_tree(getattr(value, attr))


def summarize(samples_ms: list[float]) -> TimingSummary:
    return TimingSummary(
        mean_ms=float(statistics.fmean(samples_ms)),
        median_ms=float(statistics.median(samples_ms)),
        min_ms=float(min(samples_ms)),
        max_ms=float(max(samples_ms)),
        std_ms=float(statistics.pstdev(samples_ms)),
    )


def time_callable(fn, *, warmup: int, repeat: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    first = fn()
    block_tree(first)
    compile_ms = (time.perf_counter() - t0) * 1.0e3

    for _ in range(warmup):
        value = fn()
        block_tree(value)

    samples_ms: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        value = fn()
        block_tree(value)
        samples_ms.append((time.perf_counter() - start) * 1.0e3)

    return {
        "compile_ms": compile_ms,
        "steady_state": asdict(summarize(samples_ms)),
    }


def package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def main() -> int:
    args = build_parser().parse_args()

    settings = PTSettings(ir_resummation=False)
    eval_k = np.linspace(0.02, 0.18, args.nk)
    linear_input = make_linear_input(z=args.z, support_nk=args.support_nk)

    realspace_predictor = build_realspace_predictor(linear_input, settings=settings, k=eval_k)
    multipole_theory = GalaxyPowerSpectrumMultipolesTheory(
        template=PowerSpectrumTemplate.from_linear_input(linear_input, settings=settings),
        k=eval_k,
    )

    realspace_args = {
        "b1": DEFAULT_NUISANCE["b1"],
        "b2": DEFAULT_NUISANCE["b2"],
        "bG2": DEFAULT_NUISANCE["bG2"],
        "bGamma3": DEFAULT_NUISANCE["bGamma3"],
        "cs": DEFAULT_NUISANCE["cs"],
        "cs0": DEFAULT_NUISANCE["cs0"],
        "Pshot": DEFAULT_NUISANCE["Pshot"],
    }
    multipole_args = {
        "b1": DEFAULT_NUISANCE["b1"],
        "b2": DEFAULT_NUISANCE["b2"],
        "bG2": DEFAULT_NUISANCE["bG2"],
        "bGamma3": DEFAULT_NUISANCE["bGamma3"],
        "cs0": DEFAULT_NUISANCE["cs0"],
        "cs2": DEFAULT_NUISANCE["cs2"],
        "cs4": DEFAULT_NUISANCE["cs4"],
        "Pshot": DEFAULT_NUISANCE["Pshot"],
        "b4": DEFAULT_NUISANCE["b4"],
    }

    devices = jax.devices()
    accelerator_devices = [device for device in devices if getattr(device, "platform", "cpu") != "cpu"]
    result = {
        "environment": {
            "python": sys.version.split()[0],
            "jax": package_version("jax"),
            "jaxlib": package_version("jaxlib"),
            "jax_metal": package_version("jax-metal"),
            "default_backend": jax.default_backend(),
            "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
            "devices": [str(device) for device in devices],
            "accelerator_visible": bool(accelerator_devices),
        },
        "config": {
            "nk": args.nk,
            "support_nk": args.support_nk,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "z": args.z,
        },
        "benchmarks": {
            "realspace_native": time_callable(lambda: realspace_predictor(**realspace_args), warmup=args.warmup, repeat=args.repeat),
            "multipoles_internal": time_callable(
                lambda: multipole_theory._predict_multipoles(multipole_args),
                warmup=args.warmup,
                repeat=args.repeat,
            ),
        },
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="ascii")

    if args.require_gpu and not accelerator_devices:
        print("No non-CPU JAX device is visible.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
