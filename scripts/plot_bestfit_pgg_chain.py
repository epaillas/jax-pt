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

from jaxpt import PTSettings
from jaxpt.cosmology import _normalize_cosmology_overrides
from jaxpt.inference.base import flatten_prediction
from jaxpt.theories import (
    GalaxyPowerSpectrumMultipolesTheory,
    PowerSpectrumTemplate,
    load_galaxy_power_spectrum_multipoles_parameters,
    load_power_spectrum_template_parameters,
)


REQUIRED_KEYS = ("samples", "logl", "parameter_names")


def _scalar(entry):
    value = np.asarray(entry)
    if value.shape == ():
        return value.item()
    return value


def _require_keys(data, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(
            "Chain file is missing required plotting metadata keys: "
            + ", ".join(missing)
            + ". Re-run scripts/run_pocomc_inference.py with the updated metadata writer."
        )


def _load_legacy_chain_metadata(data) -> dict[str, object]:
    legacy_keys = (
        "meta_observable",
        "meta_model_kind",
        "meta_bestfit_rule",
        "meta_ells",
        "meta_k",
        "meta_kmin",
        "meta_kmax",
        "meta_data_vector",
        "meta_covariance",
        "meta_errors",
        "meta_z",
        "meta_provider",
        "meta_settings_json",
        "meta_baseline_param_names",
        "meta_baseline_param_values",
    )
    _require_keys(data, legacy_keys)
    baseline_names = [str(name) for name in np.asarray(data["meta_baseline_param_names"], dtype=str)]
    baseline_values = np.asarray(data["meta_baseline_param_values"], dtype=float)
    if len(baseline_names) != baseline_values.size:
        raise ValueError("Baseline parameter metadata is inconsistent.")
    return {
        "observable": str(_scalar(data["meta_observable"])),
        "model_kind": str(_scalar(data["meta_model_kind"])),
        "bestfit_rule": str(_scalar(data["meta_bestfit_rule"])),
        "ells": tuple(int(value) for value in np.asarray(data["meta_ells"], dtype=int)),
        "k_units": str(_scalar(data["meta_k_units"])) if "meta_k_units" in data else "h/Mpc",
        "kmin": float(_scalar(data["meta_kmin"])),
        "kmax": float(_scalar(data["meta_kmax"])),
        "z": float(_scalar(data["meta_z"])),
        "provider": str(_scalar(data["meta_provider"])),
        "settings": json.loads(str(_scalar(data["meta_settings_json"]))),
        "baseline": {name: float(value) for name, value in zip(baseline_names, baseline_values, strict=True)},
        "k": np.asarray(data["meta_k"], dtype=float),
        "data_vector": np.asarray(data["meta_data_vector"], dtype=float),
        "covariance": np.asarray(data["meta_covariance"], dtype=float),
        "errors": np.asarray(data["meta_errors"], dtype=float),
    }


def _default_output_path(chain_path: Path) -> Path:
    return chain_path.with_name(f"{chain_path.stem}_bestfit.png")


def _split_vector_by_ell(vector: np.ndarray, *, nk: int, ells: tuple[int, ...]) -> dict[int, np.ndarray]:
    return {
        ell: np.asarray(vector[index * nk : (index + 1) * nk], dtype=float)
        for index, ell in enumerate(ells)
    }


def _load_chain_metadata(data) -> dict[str, object]:
    _require_keys(data, REQUIRED_KEYS)
    if "metadata_json" not in data:
        return _load_legacy_chain_metadata(data)

    metadata = json.loads(str(_scalar(data["metadata_json"])))
    required_fields = {
        "observable",
        "model_kind",
        "bestfit_rule",
        "ells",
        "k_units",
        "kmin",
        "kmax",
        "z",
        "provider",
        "settings",
        "baseline",
    }
    missing = sorted(required_fields - set(metadata))
    if missing:
        raise ValueError("Chain metadata_json is missing required fields: " + ", ".join(missing) + ".")
    for key in ("k", "data_vector", "covariance", "errors"):
        if key not in data:
            raise ValueError(f"Chain file is missing required array key '{key}'.")
    metadata = dict(metadata)
    metadata["ells"] = tuple(int(ell) for ell in metadata["ells"])
    metadata["k"] = np.asarray(data["k"], dtype=float)
    metadata["data_vector"] = np.asarray(data["data_vector"], dtype=float)
    metadata["covariance"] = np.asarray(data["covariance"], dtype=float)
    metadata["errors"] = np.asarray(data["errors"], dtype=float)
    metadata["baseline"] = {str(name): float(value) for name, value in dict(metadata["baseline"]).items()}
    return metadata


def _build_theory(metadata: dict[str, object], k: np.ndarray) -> GalaxyPowerSpectrumMultipolesTheory:
    cosmology_defaults = load_power_spectrum_template_parameters().defaults_dict()
    nuisance_defaults = load_galaxy_power_spectrum_multipoles_parameters().defaults_dict()
    baseline = dict(cosmology_defaults)
    baseline.update(nuisance_defaults)
    baseline.update(_normalize_cosmology_overrides(dict(metadata["baseline"])))

    settings_payload = dict(metadata["settings"])
    settings_payload["backend"] = "jaxpt"
    settings = PTSettings(**settings_payload)
    z = float(metadata["z"])
    provider = str(metadata["provider"])

    cosmology = {name: baseline[name] for name in cosmology_defaults}
    template = PowerSpectrumTemplate(cosmology, z=z, settings=settings, provider=provider)
    return GalaxyPowerSpectrumMultipolesTheory(template=template, k=np.asarray(k, dtype=float))


def _resolve_h_value(query: dict[str, float], baseline: dict[str, float]) -> float:
    if "h" in query:
        return float(query["h"])
    if "h" in baseline:
        return float(baseline["h"])
    if "H0" in query:
        return float(query["H0"]) / 100.0
    if "H0" in baseline:
        return float(baseline["H0"]) / 100.0
    raise ValueError("Could not resolve h for k-unit conversion.")


def _bestfit_query(data, metadata: dict[str, object]) -> tuple[int, dict[str, float]]:
    samples = np.asarray(data["samples"], dtype=float)
    logl = np.asarray(data["logl"], dtype=float)
    param_names = [str(name) for name in np.asarray(data["parameter_names"], dtype=str)]
    if samples.ndim != 2 or samples.shape[0] != logl.size or samples.shape[1] != len(param_names):
        raise ValueError("Chain samples/logl metadata are inconsistent.")
    best_index = int(np.argmax(logl))
    query = dict(metadata["baseline"])
    for name, value in zip(param_names, samples[best_index], strict=True):
        query[name] = float(value)
    return best_index, query


def _solve_conditional_marginalized_query(
    theory: GalaxyPowerSpectrumMultipolesTheory,
    *,
    sampled_query: dict[str, float],
    data_vector: np.ndarray,
    covariance: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    marginalized_names = tuple(str(name) for name in theory.params.marginalized_names())
    if not marginalized_names:
        return dict(sampled_query), {}

    sampled_prediction = theory(sampled_query)
    design = np.asarray(theory.marginalized_design_matrix(sampled_query, parameter_names=marginalized_names), dtype=float)
    baseline = np.asarray([float(theory.params[name].value) for name in marginalized_names], dtype=float)
    covariance_inv = np.linalg.inv(np.asarray(covariance, dtype=float))
    residual = np.asarray(data_vector, dtype=float) - flatten_prediction(sampled_prediction)

    precision = np.zeros((len(marginalized_names), len(marginalized_names)), dtype=float)
    rhs_prior = np.zeros(len(marginalized_names), dtype=float)
    for index, name in enumerate(marginalized_names):
        prior = theory.params[name].prior
        if prior is None:
            continue
        prior_type = str(prior.get("type", "flat")).lower()
        if prior_type == "gaussian":
            sigma = prior.get("sigma")
            if sigma is None:
                raise ValueError(f"Marginalized Gaussian prior for '{name}' requires 'sigma'.")
            variance = float(sigma) ** 2
            if variance <= 0.0:
                raise ValueError(f"Marginalized Gaussian prior for '{name}' must have positive sigma.")
            mean = float(prior.get("mean", baseline[index]))
            precision[index, index] = 1.0 / variance
            rhs_prior[index] = precision[index, index] * (mean - baseline[index])
        elif prior_type == "flat":
            continue
        else:
            raise ValueError(f"Unsupported marginalized prior type '{prior_type}' for parameter '{name}'.")

    fisher = design.T @ covariance_inv @ design + precision
    rhs = design.T @ covariance_inv @ residual + rhs_prior
    shift = np.linalg.solve(fisher, rhs)
    conditional_values = baseline + shift

    query = dict(sampled_query)
    conditional = {}
    for name, value in zip(marginalized_names, conditional_values, strict=True):
        query[name] = float(value)
        conditional[name] = float(value)
    return query, conditional


def _plot_bestfit(
    output_path: Path,
    *,
    measured_k: np.ndarray,
    measured: dict[int, np.ndarray],
    errors: dict[int, np.ndarray],
    theory_k: np.ndarray,
    theory: dict[int, np.ndarray],
    bestfit: dict[str, float],
    sampled_names: list[str],
    marginalized_names: list[str],
    z: float,
    show: bool,
) -> None:
    fig, axes = plt.subplots(
        len(measured),
        1,
        figsize=(8.0, 3.0 * len(measured)),
        sharex=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)

    for ax, ell in zip(axes_array, measured):
        ax.errorbar(
            measured_k,
            measured_k * measured[ell],
            yerr=measured_k * errors[ell],
            fmt="o",
            ms=3.5,
            lw=1.2,
            capsize=2.0,
            label="measurement",
        )
        ax.plot(theory_k, theory_k * theory[ell], lw=2.0, label="best fit")
        ax.set_title(f"P{ell}")
        ax.set_ylabel(rf"$ k P_{{{ell}}}(k) $ [$(\mathrm{{Mpc}}/h)^2$]")
        ax.grid(True, alpha=0.25)

    axes_array[-1].set_xlabel("k [h/Mpc]")
    axes_array[0].legend()

    display_names = sampled_names + marginalized_names
    param_text = ", ".join(f"{name}={bestfit[name]:.3f}" for name in display_names)
    fig.suptitle(f"Best-fit Pgg multipoles at z={z:g}\n{param_text}", fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(f"plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot measured Pgg multipoles against the best-fit direct model from a self-describing chain.")
    parser.add_argument("chain", type=Path, help="Path to a self-describing PocoMC chain `.npz` file.")
    parser.add_argument("--output", type=Path, default=None, help="Path to the output PNG.")
    parser.add_argument("--n-fine", type=int, default=300, help="Number of points in the smooth theory k grid.")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively after writing the PNG.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = args.output if args.output is not None else _default_output_path(args.chain)

    with np.load(args.chain, allow_pickle=False) as data:
        metadata = _load_chain_metadata(data)
        if metadata["observable"] != "pgg":
            raise ValueError(f"Unsupported observable {metadata['observable']!r}; expected 'pgg'.")
        if metadata["model_kind"] != "taylor_emulator_pgg":
            raise ValueError(f"Unsupported chain model kind {metadata['model_kind']!r}.")
        if metadata["bestfit_rule"] != "max_logl":
            raise ValueError(f"Unsupported best-fit rule {metadata['bestfit_rule']!r}.")
        if metadata["k_units"] != "h/Mpc":
            raise ValueError(f"Unsupported k units {metadata['k_units']!r}; expected 'h/Mpc'.")

        best_index, bestfit = _bestfit_query(data, metadata)
        sampled_names = [str(name) for name in np.asarray(data["parameter_names"], dtype=str)]
        measured_k = np.asarray(metadata["k"], dtype=float)
        ells = tuple(int(ell) for ell in metadata["ells"])
        nk = len(measured_k)
        measured = _split_vector_by_ell(np.asarray(metadata["data_vector"], dtype=float), nk=nk, ells=ells)
        errors = _split_vector_by_ell(np.asarray(metadata["errors"], dtype=float), nk=nk, ells=ells)

    h_for_data = _resolve_h_value(bestfit, dict(metadata["baseline"]))
    measured_theory = _build_theory(metadata, h_for_data * measured_k)
    bestfit, conditional_marginalized = _solve_conditional_marginalized_query(
        measured_theory,
        sampled_query=bestfit,
        data_vector=np.asarray(metadata["data_vector"], dtype=float),
        covariance=np.asarray(metadata["covariance"], dtype=float),
    )

    theory_k = np.linspace(float(metadata["kmin"]), float(metadata["kmax"]), int(args.n_fine))
    h_bestfit = _resolve_h_value(bestfit, dict(metadata["baseline"]))
    theory = _build_theory(metadata, h_bestfit * theory_k)
    prediction = theory(bestfit)
    theory_by_ell = {
        0: np.asarray(prediction.p0, dtype=float),
        2: np.asarray(prediction.p2, dtype=float),
        4: np.asarray(prediction.p4, dtype=float),
    }

    print("Best-fit Pgg plot")
    print(f"chain: {args.chain}")
    print(f"bestfit_rule: {metadata['bestfit_rule']}")
    print(f"bestfit_index: {best_index}")
    print(f"ells: {', '.join(str(ell) for ell in ells)}")
    print(f"n_k_data: {nk}")
    print(f"n_k_theory: {len(theory_k)}")
    for name in sampled_names:
        print(f"{name}: {bestfit[name]:.6g}")
    for name, value in conditional_marginalized.items():
        print(f"{name} (conditional): {value:.6g}")

    _plot_bestfit(
        output_path,
        measured_k=measured_k,
        measured=measured,
        errors=errors,
        theory_k=theory_k,
        theory=theory_by_ell,
        bestfit=bestfit,
        sampled_names=sampled_names,
        marginalized_names=list(conditional_marginalized),
        z=float(metadata["z"]),
        show=args.show,
    )


if __name__ == "__main__":
    main()
