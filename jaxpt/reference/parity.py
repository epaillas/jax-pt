from __future__ import annotations

from typing import Any

import numpy as np

from ..config import EFTBiasParams
from .classpt import BasisSpectra, MultipolePrediction


def compare_predictions(lhs: MultipolePrediction, rhs: MultipolePrediction) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    for name in ("p0", "p2", "p4"):
        lhs_values = np.asarray(getattr(lhs, name), dtype=float)
        rhs_values = np.asarray(getattr(rhs, name), dtype=float)
        diff = np.abs(lhs_values - rhs_values)
        scale = np.maximum(np.abs(rhs_values), 1.0e-30)
        comparison[name] = {
            "max_abs": float(diff.max(initial=0.0)),
            "max_rel": float((diff / scale).max(initial=0.0)),
        }
    return comparison


def compare_multipoles_to_classpt(
    prediction: MultipolePrediction,
    cosmo: Any,
    params: EFTBiasParams,
) -> dict[str, dict[str, float]]:
    k = np.asarray(prediction.k, dtype=float)
    z = float(prediction.metadata["z"])
    cosmo.initialize_output(k, z, len(k))
    reference = MultipolePrediction(
        k=prediction.k,
        p0=np.asarray(cosmo.pk_gg_l0(params.b1, params.b2, params.bG2, params.bGamma3, params.cs0, params.Pshot, params.b4)),
        p2=np.asarray(cosmo.pk_gg_l2(params.b1, params.b2, params.bG2, params.bGamma3, params.cs2, params.b4)),
        p4=np.asarray(cosmo.pk_gg_l4(params.b1, params.b2, params.bG2, params.bGamma3, params.cs4, params.b4)),
        metadata={"backend": "classpt_reference"},
    )
    return compare_predictions(prediction, reference)
