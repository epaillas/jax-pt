from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from jaxpt import PocoMCSampler, TaylorEmulator, cached_sample_covariance, repo_cache_dir
from jaxpt.parameter import ParameterCollection
from jaxpt.reference.classpt import MultipolePrediction
from jaxpt.utils import flatten_pgg_measurements, load_pgg_data_vector


DEFAULT_DATA_PATH = ROOT / "data" / "data_vector" / "mesh2_spectrum_poles_c000_hod006.h5"
DEFAULT_MOCK_DIR = ROOT / "data" / "for_covariance"


def _parse_pair_assignment(text: str, value_names: tuple[str, str]) -> tuple[str, float, float]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"Expected NAME=VALUE1,VALUE2 assignment, received '{text}'.")
    name, raw_values = text.split("=", 1)
    parts = [item.strip() for item in raw_values.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected NAME={value_names[0]},{value_names[1]} assignment, received '{text}'."
        )
    try:
        first = float(parts[0])
        second = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Assignment '{text}' must contain numeric prior values.") from exc
    return name.strip(), first, second


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PocoMC on a serialized jaxpt Taylor emulator using a Gaussian likelihood from HDF5 multipole measurements. Input k is interpreted as h/Mpc."
    )
    parser.add_argument("emulator", type=Path, help="Path to a serialized Taylor emulator `.npz` file.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to the mean Pgg measurement HDF5 file.")
    parser.add_argument("--mocks", type=Path, default=DEFAULT_MOCK_DIR, help="Directory containing mock realization HDF5 files.")
    parser.add_argument("--ells", type=int, nargs="+", default=[0, 2, 4], help="Multipoles to load in ell-major order.")
    parser.add_argument("--rebin", type=int, default=13, help="Rebinning factor passed to the HDF5 readers.")
    parser.add_argument("--kmin", type=float, default=0.01, help="Minimum k included in the likelihood.")
    parser.add_argument("--kmax", type=float, default=0.2, help="Maximum k included in the likelihood.")
    parser.add_argument(
        "--covariance-jitter",
        type=float,
        default=0.0,
        help="Diagonal covariance regularization added after the sample covariance is built.",
    )
    parser.add_argument(
        "--covariance-cache-dir",
        type=Path,
        default=repo_cache_dir("covariances"),
        help="Directory where hashed covariance artifacts are stored.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Explicit sampled parameter name. Repeat to sample a subset of the emulator parameters.",
    )
    parser.add_argument(
        "--flat-prior",
        action="append",
        default=[],
        metavar="NAME=MIN,MAX",
        help="Override or define a bounded flat prior for a sampled parameter. Repeat as needed.",
    )
    parser.add_argument(
        "--gaussian-prior",
        action="append",
        default=[],
        metavar="NAME=MEAN,SIGMA",
        help="Override or define a Gaussian prior for a sampled parameter. Repeat as needed.",
    )
    parser.add_argument("--n-active", type=int, default=32, help="PocoMC active particle count.")
    parser.add_argument("--n-effective", type=int, default=64, help="PocoMC effective particle count.")
    parser.add_argument("--n-total", type=int, default=128, help="Target effective posterior sample count.")
    parser.add_argument("--n-evidence", type=int, default=0, help="Importance samples used for evidence estimation.")
    parser.add_argument("--epochs", type=int, default=50, help="Normalizing-flow training epochs per PocoMC update.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed forwarded to PocoMC.")
    parser.add_argument("--no-precondition", action="store_true", help="Disable PocoMC flow preconditioning for tiny or fragile runs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to an output `.npz` file. Defaults to `<emulator_stem>_pocomc_posterior.npz` next to the emulator.",
    )
    parser.add_argument("--progress", action="store_true", help="Show PocoMC progress output.")
    return parser


def _default_output_path(emulator_path: Path) -> Path:
    return emulator_path.with_name(f"{emulator_path.stem}_pocomc_posterior.npz")


def _build_prior_overrides(
    emulator: TaylorEmulator,
    flat_specs: list[str],
    gaussian_specs: list[str],
) -> ParameterCollection:
    overrides = ParameterCollection()
    for item in flat_specs:
        name, lower, upper = _parse_pair_assignment(item, ("MIN", "MAX"))
        fiducial = emulator.fiducial.get(name)
        if fiducial is None:
            raise ValueError(f"Unknown flat-prior parameter '{name}'.")
        overrides.update(
            {
                name: {
                    "value": fiducial,
                    "prior": {"type": "flat", "min": lower, "max": upper},
                }
            }
        )
    for item in gaussian_specs:
        name, mean, sigma = _parse_pair_assignment(item, ("MEAN", "SIGMA"))
        fiducial = emulator.fiducial.get(name)
        if fiducial is None:
            raise ValueError(f"Unknown gaussian-prior parameter '{name}'.")
        overrides.update(
            {
                name: {
                    "value": fiducial,
                    "prior": {"type": "gaussian", "mean": mean, "sigma": sigma},
                }
            }
        )
    return overrides


def _format_summary(name: str, values: np.ndarray, weights: np.ndarray) -> str:
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return f"{name}: mean={mean:.6g}, std={np.sqrt(max(variance, 0.0)):.6g}"


def _load_emulator_config(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["config_json"]))


def _baseline_parameter_arrays(emulator: TaylorEmulator) -> tuple[np.ndarray, np.ndarray]:
    names = np.asarray([str(name) for name in emulator.fiducial], dtype=str)
    values = np.asarray([float(emulator.fiducial[str(name)]) for name in names], dtype=float)
    return names, values


def _resolve_h_value(parameters: dict[str, float], fiducial: dict[str, float]) -> float:
    if "h" in parameters:
        return float(parameters["h"])
    if "h" in fiducial:
        return float(fiducial["h"])
    if "H0" in parameters:
        return float(parameters["H0"]) / 100.0
    if "H0" in fiducial:
        return float(fiducial["H0"]) / 100.0
    raise ValueError("Could not resolve h for k-unit conversion.")


def _build_metadata_payload(
    *,
    emulator: TaylorEmulator,
    emulator_metadata: dict[str, object],
    emulator_path: Path,
    data_path: Path,
    mocks_path: Path,
    ells: tuple[int, ...],
    rebin: int,
    kmin: float,
    kmax: float,
    covariance_cache_path: Path,
    covariance_jitter: float,
) -> dict[str, object]:
    baseline_names, baseline_values = _baseline_parameter_arrays(emulator)
    return {
        "observable": "pgg",
        "model_kind": "taylor_emulator_pgg",
        "bestfit_rule": "max_logl",
        "emulator_path": str(emulator_path.resolve()),
        "data_path": str(data_path.resolve()),
        "mocks_path": str(mocks_path.resolve()),
        "ells": [int(ell) for ell in ells],
        "k_units": "h/Mpc",
        "rebin": int(rebin),
        "kmin": float(kmin),
        "kmax": float(kmax),
        "covariance_cache": str(covariance_cache_path.resolve()),
        "covariance_jitter": float(covariance_jitter),
        "z": float(emulator_metadata.get("z", 0.0)),
        "provider": str(emulator_metadata.get("provider", "cosmoprimo")),
        "settings": emulator_metadata.get("settings", {}),
        "baseline": {str(name): float(value) for name, value in zip(baseline_names, baseline_values, strict=True)},
        "emulator_param_names": [str(name) for name in emulator.param_names],
    }


@dataclass(slots=True)
class _FixedKModel:
    """Bind an emulator to a fixed likelihood k-grid in h/Mpc."""

    model: TaylorEmulator
    k_target: np.ndarray

    @property
    def params(self) -> ParameterCollection:
        return self.model.params

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.model.param_names)

    def predict(self, parameters: dict[str, float]) -> MultipolePrediction | np.ndarray:
        h = _resolve_h_value(parameters, self.model.fiducial)
        return self.model.predict(parameters, k=h * np.asarray(self.k_target, dtype=float))

    def marginalized_design_matrix(
        self,
        parameters: dict[str, float],
        *,
        parameter_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        h = _resolve_h_value(parameters, self.model.fiducial)
        return np.asarray(
            self.model.marginalized_design_matrix(
                parameters,
                parameter_names=parameter_names,
                k=h * np.asarray(self.k_target, dtype=float),
            ),
            dtype=float,
        )


def main() -> None:
    args = build_parser().parse_args()
    ells = tuple(int(ell) for ell in args.ells)

    emulator = TaylorEmulator.load(args.emulator)
    emulator_config = _load_emulator_config(args.emulator)
    emulator_metadata = dict(emulator_config.get("metadata", {}))
    sampled_parameters = tuple(str(name) for name in (args.param if args.param else emulator.param_names))
    prior_overrides = _build_prior_overrides(emulator, args.flat_prior, args.gaussian_prior)

    k_data, poles = load_pgg_data_vector(args.data, ells=ells, rebin=args.rebin, kmin=args.kmin, kmax=args.kmax)
    _, covariance, covariance_cache_path = cached_sample_covariance(
        args.mocks,
        ells=ells,
        rebin=args.rebin,
        k_data=k_data,
        kmin=args.kmin,
        kmax=args.kmax,
        cache_dir=args.covariance_cache_dir,
    )
    covariance /= 64
    data_vector = flatten_pgg_measurements(poles, ells=ells)
    if args.covariance_jitter > 0.0:
        covariance = covariance + float(args.covariance_jitter) * np.eye(covariance.shape[0], dtype=float)

    matched_model = _FixedKModel(emulator, k_target=k_data)

    sampler = PocoMCSampler(
        data=data_vector,
        model=matched_model,
        covariance=covariance,
        priors=prior_overrides if len(prior_overrides) else None,
        parameter_names=sampled_parameters,
        sampler_kwargs={
            "random_state": args.seed,
            "n_active": args.n_active,
            "n_effective": args.n_effective,
            "train_config": {"epochs": args.epochs},
            "precondition": not args.no_precondition,
        },
        run_kwargs={
            "n_total": args.n_total,
            "n_evidence": args.n_evidence,
            "progress": args.progress,
        },
    )

    sampler.run()
    posterior = sampler.posterior()
    output_path = args.output if args.output is not None else _default_output_path(args.emulator)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = np.sqrt(np.diag(covariance))
    metadata = _build_metadata_payload(
        emulator=emulator,
        emulator_metadata=emulator_metadata,
        emulator_path=args.emulator,
        data_path=args.data,
        mocks_path=args.mocks,
        ells=ells,
        rebin=args.rebin,
        kmin=args.kmin,
        kmax=args.kmax,
        covariance_cache_path=covariance_cache_path,
        covariance_jitter=args.covariance_jitter,
    )
    np.savez(
        output_path,
        samples=posterior["samples"],
        weights=posterior["weights"],
        logl=posterior["logl"],
        logp=posterior["logp"],
        parameter_names=np.asarray(posterior["parameter_names"], dtype=str),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        k=np.asarray(k_data, dtype=float),
        data_vector=np.asarray(data_vector, dtype=float),
        covariance=np.asarray(covariance, dtype=float),
        errors=np.asarray(errors, dtype=float),
    )

    logz, dlogz = sampler.sampler.evidence()

    print("PocoMC inference")
    print(f"emulator: {args.emulator}")
    print(f"data: {args.data}")
    print(f"mocks: {args.mocks}")
    print(f"ells: {', '.join(str(ell) for ell in ells)}")
    print(f"n_k: {len(k_data)}")
    print(f"data_vector_length: {data_vector.size}")
    print(f"sampled parameters ({len(sampled_parameters)}): {', '.join(sampled_parameters)}")
    print(f"covariance_shape: {covariance.shape}")
    print(f"covariance_cache: {covariance_cache_path}")
    print(f"covariance_jitter: {args.covariance_jitter:g}")
    print(f"precondition: {not args.no_precondition}")
    if logz is not None:
        if dlogz is None:
            print(f"log_evidence: {float(logz):.6g}")
        else:
            print(f"log_evidence: {float(logz):.6g} +/- {float(dlogz):.6g}")
    print(f"posterior_samples: {posterior['samples'].shape[0]}")
    print(f"output: {output_path}")
    for index, name in enumerate(posterior["parameter_names"]):
        print(_format_summary(str(name), posterior["samples"][:, index], posterior["weights"]))


if __name__ == "__main__":
    main()
