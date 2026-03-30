from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..theories import GalaxyPowerSpectrumMultipolesTheory
from .taylor import TaylorEmulator


def build_native_multipole_taylor_emulator(
    theory: GalaxyPowerSpectrumMultipolesTheory,
    *,
    order: int = 4,
    step_sizes: float | Mapping[str, float] = 0.01,
    param_names: list[str] | tuple[str, ...] | None = None,
    cache_dir: str | Path | None = None,
    cache_key: str | None = None,
    finite_difference_accuracy: int = 2,
    metadata: Mapping[str, Any] | None = None,
    progress_callback=None,
    force: bool = False,
) -> TaylorEmulator:
    if theory.template.settings.backend != "native":
        raise ValueError("build_native_multipole_taylor_emulator requires a native GalaxyPowerSpectrumMultipolesTheory.")

    if theory.template.is_queryable:
        params = theory.params
    else:
        params = theory.nuisance_parameters

    fiducial = params.defaults_dict()
    valid_param_names = list(params.names())
    emulated_names = list(params.emulated_names()) if param_names is None else [str(name) for name in param_names]

    invalid = [name for name in emulated_names if name not in valid_param_names]
    if invalid:
        raise ValueError(f"Requested emulator parameters are not valid for the theory: {', '.join(invalid)}.")

    disallowed = [name for name in emulated_names if not params[name].emulated]
    if disallowed:
        raise ValueError(
            "Requested emulator parameters must be non-fixed and non-marginalized. Invalid: " + ", ".join(disallowed)
        )

    status = {
        name: {
            "fixed": bool(params[name].fixed),
            "marginalized": bool(params[name].marginalized),
        }
        for name in valid_param_names
    }
    build_metadata = {
        "emulator_kind": "native_multipole_taylor",
        "theory": theory.__class__.__name__,
        "backend": theory.template.settings.backend,
        "z": float(theory.z),
        "k": np.asarray(theory.k, dtype=float).tolist(),
        "settings": asdict(theory.template.settings),
        "parameter_status": status,
    }
    if metadata is not None:
        build_metadata.update(dict(metadata))

    emulator = TaylorEmulator(
        theory_fn=theory,
        fiducial=fiducial,
        order=order,
        step_sizes=step_sizes,
        param_names=emulated_names,
        cache_dir=cache_dir,
        cache_key=cache_key,
        finite_difference_accuracy=finite_difference_accuracy,
        metadata=build_metadata,
        valid_param_names=valid_param_names,
    )
    return emulator.build(force=force, progress_callback=progress_callback)
