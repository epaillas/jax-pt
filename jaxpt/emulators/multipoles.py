from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..theories import GalaxyPowerSpectrumMultipolesTheory, QuantileGalaxyPowerSpectrumMultipolesTheory
from .taylor import TaylorEmulator


MultipoleEmulatorTheory = GalaxyPowerSpectrumMultipolesTheory | QuantileGalaxyPowerSpectrumMultipolesTheory


def _cache_settings_payload(settings: Any) -> dict[str, Any]:
    return {
        "loop_order": settings.loop_order,
        "ir_resummation": settings.ir_resummation,
        "cb": settings.cb,
        "rsd": settings.rsd,
        "ap_effect": settings.ap_effect,
    }


def build_multipole_emulator(
    theory: MultipoleEmulatorTheory,
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
    """Build a Taylor emulator for multipole-like theory outputs.

    Parameters
    ----------
    theory
        Supported theory instance to emulate. The template backend must be
        ``"jaxpt"``.
    order
        Maximum total Taylor order.
    step_sizes
        Either a scalar relative step scale or an explicit per-parameter
        mapping.
    param_names
        Optional explicit parameter subset to emulate. If omitted, the theory's
        non-fixed, non-marginalized parameters are used.
    cache_dir
        Optional directory where hashed emulator files should be written.
    cache_key
        Optional explicit cache key. If omitted, a deterministic hash is built
        from the emulator configuration and theory metadata.
    finite_difference_accuracy
        Central-stencil accuracy order used for derivative estimates.
    metadata
        Optional extra metadata merged into the build record.
    progress_callback
        Optional callback receiving ``(completed_terms, total_terms)`` during
        training.
    force
        If ``True``, rebuild even if a matching hashed file already exists.
    """
    if theory.template.settings.backend != "jaxpt":
        raise ValueError("build_multipole_emulator requires a jaxpt multipole theory.")

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
        "emulator_kind": "multipole_taylor",
        "theory": theory.__class__.__name__,
        "backend": theory.template.settings.backend,
        "provider": getattr(theory.template, "provider", "cosmoprimo"),
        "z": float(theory.z),
        "k": np.asarray(theory.k, dtype=float).tolist(),
        "settings": asdict(theory.template.settings),
        "parameter_status": status,
    }
    if metadata is not None:
        build_metadata.update(dict(metadata))

    cache_identity = {
        "emulator_kind": "multipole_taylor",
        "z": float(theory.z),
        "k": np.asarray(theory.k, dtype=float).tolist(),
        "settings": _cache_settings_payload(theory.template.settings),
        "parameter_status": status,
    }

    emulator = TaylorEmulator(
        theory_fn=theory,
        fiducial=fiducial,
        order=order,
        step_sizes=step_sizes,
        param_names=emulated_names,
        cache_dir=cache_dir,
        cache_key=cache_key,
        finite_difference_accuracy=finite_difference_accuracy,
        cache_identity=cache_identity,
        metadata=build_metadata,
        valid_param_names=valid_param_names,
    )
    emulator = emulator.build(force=force, progress_callback=progress_callback)

    marginalized_names = list(params.marginalized_names())
    if marginalized_names and hasattr(theory, "marginalized_design_matrix"):
        template_emulator = TaylorEmulator(
            theory_fn=lambda query: np.asarray(theory.marginalized_design_matrix(query, parameter_names=marginalized_names), dtype=float),
            fiducial=fiducial,
            order=order,
            step_sizes=step_sizes,
            param_names=emulated_names,
            finite_difference_accuracy=finite_difference_accuracy,
            valid_param_names=valid_param_names,
            params=params,
        ).build()
        assert template_emulator._coefficients is not None
        assert template_emulator._output_state is not None
        emulator.attach_marginalized_design(
            marginalized_names,
            template_emulator._coefficients,
            template_emulator._output_state,
        )
        if emulator.cache_path is not None:
            emulator.save(emulator.cache_path)

    return emulator
