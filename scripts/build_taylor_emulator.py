from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from jaxpt import PTSettings, build_multipole_emulator
from jaxpt.theories import (
    GalaxyPowerSpectrumMultipolesTheory,
    PowerSpectrumTemplate,
    load_galaxy_power_spectrum_multipoles_parameters,
    load_power_spectrum_template_parameters,
)


DEFAULT_FREE_COSMOLOGY = ("A_s", "omega_cdm")


def _parse_assignment(text: str) -> tuple[str, float]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"Expected NAME=VALUE assignment, received '{text}'.")
    name, value = text.split("=", 1)
    try:
        return name.strip(), float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Assignment '{text}' must have a numeric VALUE.") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a hashed Taylor emulator for jaxpt power-spectrum multipoles.")
    parser.add_argument("--z", type=float, default=0.5, help="Redshift at which to build the emulator.")
    parser.add_argument("--kmin", type=float, default=0.01, help="Minimum evaluation k in 1/Mpc.")
    parser.add_argument("--kmax", type=float, default=0.2, help="Maximum evaluation k in 1/Mpc.")
    parser.add_argument("--nk", type=int, default=64, help="Number of evaluation-grid points.")
    parser.add_argument("--order", type=int, default=4, help="Taylor expansion order.")
    parser.add_argument("--finite-difference-accuracy", type=int, default=2, help="Finite-difference stencil accuracy.")
    parser.add_argument("--step-size-scale", type=float, default=0.01, help="Relative default step size for emulated parameters.")
    parser.add_argument("--step-size", action="append", default=[], type=_parse_assignment, metavar="NAME=VALUE", help="Per-parameter step-size override.")
    parser.add_argument("--param", action="append", default=[], help="Explicit emulator parameter to include. Repeat for multiple parameters.")
    parser.add_argument(
        "--fix-param",
        action="append",
        default=[],
        help="Additional valid theory parameters to force fixed before emulator construction. Repeat for multiple parameters.",
    )
    parser.add_argument(
        "--cosmo",
        action="append",
        default=[],
        type=_parse_assignment,
        metavar="NAME=VALUE",
        help="Override fiducial cosmology defaults before building.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("emulator_outputs"),
        help="Directory where the hashed emulator artifact will be saved.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if the hashed emulator file already exists.")
    return parser


def _render_progress(completed: int, total: int) -> None:
    width = 32
    fraction = 1.0 if total == 0 else completed / total
    filled = min(width, int(round(width * fraction)))
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if completed >= total else "\r"
    print(f"[{bar}] {completed:4d}/{total:4d} {100.0 * fraction:6.2f}%", end=end, flush=True)


def _set_fixed_status(theory: GalaxyPowerSpectrumMultipolesTheory, free_cosmology: set[str], extra_fixed: set[str]) -> None:
    cosmology_names = set(theory.template.params.names())
    valid_names = set(theory.params.names())
    invalid = sorted((free_cosmology | extra_fixed) - valid_names)
    if invalid:
        raise ValueError(f"Unknown parameters in fixed/free overrides: {', '.join(invalid)}.")

    for name in cosmology_names:
        theory.params[name].update(fixed=name not in free_cosmology)
    for name in extra_fixed:
        theory.params[name].update(fixed=True)


def _resolve_step_sizes(default_scale: float, explicit: list[tuple[str, float]]) -> float | dict[str, float]:
    if not explicit:
        return default_scale
    return {name: value for name, value in explicit}


def _format_param_values(theory: GalaxyPowerSpectrumMultipolesTheory, names: list[str]) -> str:
    if not names:
        return "(none)"
    return ", ".join(f"{name}={theory.params[name].value:g}" for name in names)


def main() -> None:
    args = build_parser().parse_args()

    cosmology_defaults = load_power_spectrum_template_parameters().defaults_dict()
    nuisance_defaults = load_galaxy_power_spectrum_multipoles_parameters().defaults_dict()
    cosmology_defaults.update(dict(args.cosmo))

    settings = PTSettings(backend="jaxpt", ir_resummation=False)
    eval_k = np.linspace(args.kmin, args.kmax, args.nk)

    template = PowerSpectrumTemplate(cosmology_defaults, z=args.z, settings=settings, provider="cosmoprimo")
    theory = GalaxyPowerSpectrumMultipolesTheory(template=template, k=eval_k)
    for name, value in nuisance_defaults.items():
        theory.params[name].update(value=value)

    _set_fixed_status(
        theory,
        free_cosmology=set(DEFAULT_FREE_COSMOLOGY),
        extra_fixed=set(args.fix_param),
    )

    explicit_params = [str(name) for name in args.param]
    step_sizes = _resolve_step_sizes(args.step_size_scale, args.step_size)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    emulated_params = explicit_params if explicit_params else list(theory.params.emulated_names())
    held_params = [name for name in theory.params.names() if name not in emulated_params]

    print("Taylor emulator build")
    print(f"  backend: {settings.backend}")
    print(f"  provider: cosmoprimo")
    print(f"  redshift: {args.z:g}")
    print(f"  evaluation grid: nk={args.nk}, kmin={args.kmin:g}, kmax={args.kmax:g}")
    print(f"  order: {args.order}")
    print(f"  finite-difference accuracy: {args.finite_difference_accuracy}")
    print(f"  output dir: {output_dir}")
    print(f"  force rebuild: {args.force}")
    print(f"  emulated parameters ({len(emulated_params)}): {_format_param_values(theory, emulated_params)}")
    print(f"  held fixed or marginalized ({len(held_params)}): {_format_param_values(theory, held_params)}")
    if isinstance(step_sizes, dict):
        step_text = ", ".join(f"{name}={value:g}" for name, value in step_sizes.items())
        print(f"  step sizes: {step_text}")
    else:
        print(f"  step-size scale: {step_sizes:g}")
    print()

    emulator = build_multipole_emulator(
        theory,
        order=args.order,
        step_sizes=step_sizes,
        param_names=explicit_params if explicit_params else None,
        cache_dir=output_dir,
        finite_difference_accuracy=args.finite_difference_accuracy,
        metadata={"script": "build_multipole_emulator"},
        progress_callback=_render_progress,
        force=args.force,
    )

    print(f"Saved emulator: {emulator.cache_path}")
    print(f"Emulated parameters: {', '.join(emulator.param_names)}")
    print(f"Held fixed or marginalized: {', '.join(held_params)}")


if __name__ == "__main__":
    main()
