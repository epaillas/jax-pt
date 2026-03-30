from __future__ import annotations

import argparse
import json
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

from jaxpt import PTSettings, TaylorEmulator
from jaxpt.theories import (
    GalaxyPowerSpectrumMultipolesTheory,
    PowerSpectrumTemplate,
    load_galaxy_power_spectrum_multipoles_parameters,
    load_power_spectrum_template_parameters,
)


DEFAULT_Z = 0.5
DEFAULT_PROVIDER = "cosmoprimo"
DEFAULT_IR_RESUMMATION = False


def _parse_assignment(text: str) -> tuple[str, float]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"Expected NAME=VALUE assignment, received '{text}'.")
    name, value = text.split("=", 1)
    try:
        return name.strip(), float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Assignment '{text}' must have a numeric VALUE.") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a serialized Taylor emulator against direct native jaxpt multipole theory."
    )
    parser.add_argument("emulator", type=Path, help="Path to a serialized Taylor emulator `.npz` file.")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        type=_parse_assignment,
        metavar="NAME=VALUE",
        help="Parameter override applied on top of the emulator fiducial point. Repeat for multiple parameters.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the output PNG. Defaults to `<emulator_stem>_comparison.png` next to the emulator file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after writing the PNG.",
    )
    return parser


def _load_config(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["config_json"]))


def _resolve_settings(metadata: dict[str, object]) -> PTSettings:
    raw = metadata.get("settings")
    if isinstance(raw, dict):
        settings = dict(raw)
        settings["backend"] = "jaxpt"
        return PTSettings(**settings)
    return PTSettings(backend="jaxpt", ir_resummation=DEFAULT_IR_RESUMMATION)


def _resolve_z(metadata: dict[str, object]) -> float:
    value = metadata.get("z")
    if value is None:
        return DEFAULT_Z
    return float(value)


def _resolve_provider(metadata: dict[str, object]) -> str:
    value = metadata.get("provider")
    if value is None:
        return DEFAULT_PROVIDER
    return str(value)


def _resolve_k(metadata: dict[str, object], emulator: TaylorEmulator) -> np.ndarray:
    value = metadata.get("k")
    if value is not None:
        return np.asarray(value, dtype=float)
    return np.asarray(emulator.predict({}).k, dtype=float)


def _set_parameter_status(theory: GalaxyPowerSpectrumMultipolesTheory, metadata: dict[str, object]) -> None:
    raw = metadata.get("parameter_status")
    if not isinstance(raw, dict):
        return
    for name, status in raw.items():
        if name not in theory.params or not isinstance(status, dict):
            continue
        updates: dict[str, bool] = {}
        if "fixed" in status:
            updates["fixed"] = bool(status["fixed"])
        if "marginalized" in status:
            updates["marginalized"] = bool(status["marginalized"])
        if updates:
            theory.params[str(name)].update(**updates)


def _build_theory(emulator: TaylorEmulator, metadata: dict[str, object]) -> GalaxyPowerSpectrumMultipolesTheory:
    cosmology_defaults = load_power_spectrum_template_parameters().defaults_dict()
    nuisance_defaults = load_galaxy_power_spectrum_multipoles_parameters().defaults_dict()
    full_fiducial = dict(cosmology_defaults)
    full_fiducial.update(nuisance_defaults)
    full_fiducial.update(emulator.fiducial)

    settings = _resolve_settings(metadata)
    z = _resolve_z(metadata)
    k = _resolve_k(metadata, emulator)
    provider = _resolve_provider(metadata)

    cosmology_fiducial = {name: full_fiducial[name] for name in cosmology_defaults}
    template = PowerSpectrumTemplate(cosmology_fiducial, z=z, settings=settings, provider=provider)
    theory = GalaxyPowerSpectrumMultipolesTheory(template=template, k=k)
    for name, value in full_fiducial.items():
        if name in theory.params:
            theory.params[name].update(value=float(value))
    _set_parameter_status(theory, metadata)
    return theory


def _resolve_query(emulator: TaylorEmulator, overrides: list[tuple[str, float]]) -> dict[str, float]:
    query = dict(emulator.fiducial)
    for name, value in overrides:
        if name not in emulator.fiducial:
            raise ValueError(f"Unknown parameter override '{name}'.")
        query[name] = float(value)
    return query


def _fractional_residual(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (prediction - reference) / np.maximum(np.abs(reference), 1.0)


def _format_metrics(label: str, predicted: np.ndarray, reference: np.ndarray) -> list[str]:
    absolute = np.asarray(predicted, dtype=float) - np.asarray(reference, dtype=float)
    fractional = _fractional_residual(predicted, reference)
    return [
        (
            f"{label}: "
            f"max|dP|={np.max(np.abs(absolute)):.6e}, "
            f"rms|dP|={np.sqrt(np.mean(absolute**2)):.6e}, "
            f"max|dP/P|={np.max(np.abs(fractional)):.6e}, "
            f"rms|dP/P|={np.sqrt(np.mean(fractional**2)):.6e}"
        )
    ]


def _default_output_path(emulator_path: Path) -> Path:
    return emulator_path.with_name(f"{emulator_path.stem}_comparison.png")


def _plot_comparison(
    output_path: Path,
    *,
    k: np.ndarray,
    emulator_prediction,
    direct_prediction,
    show: bool,
) -> None:
    spectra = [
        ("P0", np.asarray(emulator_prediction.p0), np.asarray(direct_prediction.p0)),
        ("P2", np.asarray(emulator_prediction.p2), np.asarray(direct_prediction.p2)),
        ("P4", np.asarray(emulator_prediction.p4), np.asarray(direct_prediction.p4)),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.5, 6.5),
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.2]},
    )

    for column, (label, emulator_values, direct_values) in enumerate(spectra):
        ax_top = axes[0, column]
        ax_bottom = axes[1, column]

        ax_top.plot(k, k * emulator_values, lw=2.0, label="emulator")
        ax_top.plot(k, k * direct_values, lw=2.0, ls="--", label="direct jaxpt")
        ax_top.set_title(label)
        ax_top.set_ylabel(r"$k\,P_\ell(k)$")
        ax_top.grid(True, alpha=0.25)

        residual = _fractional_residual(emulator_values, direct_values)
        ax_bottom.axhline(0.0, color="black", lw=1.0, alpha=0.6)
        ax_bottom.plot(k, residual, lw=1.8)
        ax_bottom.set_xlabel("k [1/Mpc]")
        ax_bottom.set_ylabel(r"$\Delta / P$")
        ax_bottom.grid(True, alpha=0.25)

        limit = max(1.0e-5, 1.15 * float(np.max(np.abs(residual))))
        ax_bottom.set_ylim(-limit, limit)

    axes[0, 0].legend(loc="best")
    fig.suptitle("Taylor Emulator vs Direct jaxpt Multipoles")

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

    theory = _build_theory(TaylorEmulator.load(args.emulator), metadata)
    emulator = TaylorEmulator.load(args.emulator, theory_fn=theory)

    query = _resolve_query(emulator, args.param)
    emulator_prediction = emulator.predict(query)
    direct_prediction = theory(query)
    k = np.asarray(direct_prediction.k, dtype=float)

    output_path = args.output if args.output is not None else _default_output_path(args.emulator)

    print("Taylor emulator comparison")
    print(f"emulator: {args.emulator}")
    print(f"redshift: {_resolve_z(metadata):g}")
    print(f"k-grid: nk={k.size}, kmin={k.min():g}, kmax={k.max():g}")
    if metadata.get("z") is None or metadata.get("settings") is None or metadata.get("provider") is None:
        print(
            "metadata fallback: "
            f"z={metadata.get('z', DEFAULT_Z)}, "
            f"provider={metadata.get('provider', DEFAULT_PROVIDER)}, "
            f"settings={'serialized' if metadata.get('settings') is not None else 'default_jaxpt'}"
        )
    print("query: " + ", ".join(f"{name}={value:g}" for name, value in query.items()))
    for line in _format_metrics("P0", emulator_prediction.p0, direct_prediction.p0):
        print(line)
    for line in _format_metrics("P2", emulator_prediction.p2, direct_prediction.p2):
        print(line)
    for line in _format_metrics("P4", emulator_prediction.p4, direct_prediction.p4):
        print(line)

    _plot_comparison(
        output_path,
        k=k,
        emulator_prediction=emulator_prediction,
        direct_prediction=direct_prediction,
        show=args.show,
    )


if __name__ == "__main__":
    main()
