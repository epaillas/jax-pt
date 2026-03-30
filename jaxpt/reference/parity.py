from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from .classpt import BasisSpectra, MultipolePrediction, predict_classpt_multipoles


def compare_predictions(lhs: MultipolePrediction, rhs: MultipolePrediction) -> dict[str, dict[str, float]]:
    """Compare two multipole predictions with max-absolute and max-relative errors."""
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
    params: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Compare a prediction against a direct CLASS-PT evaluation on the same grid.

    Parameters
    ----------
    prediction
        Prediction to validate. Its metadata must include ``z``.
    cosmo
        Live `classy.Class` object configured for CLASS-PT.
    params
        Nuisance parameters forwarded to `predict_classpt_multipoles`.
    """
    k = np.asarray(prediction.k, dtype=float)
    z = float(prediction.metadata["z"])
    reference = predict_classpt_multipoles(
        cosmo,
        k,
        z,
        params,
        metadata={"backend": "classpt_reference"},
    )
    return compare_predictions(prediction, reference)
