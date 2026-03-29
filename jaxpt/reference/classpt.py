from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..config import EFTBiasParams


@dataclass(frozen=True, slots=True)
class BasisSpectra:
    k: jnp.ndarray
    z: float
    h: float
    growth_rate: float
    components: dict[str, jnp.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MultipolePrediction:
    k: jnp.ndarray
    p0: jnp.ndarray
    p2: jnp.ndarray
    p4: jnp.ndarray
    components: dict[str, jnp.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


PK_MULT_INDEX = {
    "real_tree_matter": 14,
    "real_loop_b2_b2": 1,
    "real_cross_b1_b2": 2,
    "real_cross_b1_bG2": 3,
    "real_loop_b2_bG2": 4,
    "real_loop_bG2_bG2": 5,
    "real_gamma3": 6,
    "rsd_l0_gamma3_b1": 7,
    "rsd_l0_gamma3_bias": 8,
    "rsd_l2_gamma3": 9,
    "real_counterterm_shape": 10,
    "rsd_l0_counterterm_shape": 11,
    "rsd_l2_counterterm_shape": 12,
    "rsd_l4_counterterm_shape": 13,
    "real_loop_matter": 0,
    "rsd_l0_mm_00": 15,
    "rsd_l0_mm_01": 16,
    "rsd_l0_mm_11": 17,
    "rsd_l2_mm_00": 18,
    "rsd_l2_mm_01": 19,
    "rsd_l4_mm_00": 20,
    "rsd_l0_loop_00": 21,
    "rsd_l0_loop_01": 22,
    "rsd_l0_loop_11": 23,
    "rsd_l2_loop_00": 24,
    "rsd_l2_loop_01": 25,
    "rsd_l2_11": 26,
    "rsd_l4_loop_00": 27,
    "rsd_l4_loop_01": 28,
    "rsd_l4_loop_11": 29,
    "rsd_l0_b1_b2": 30,
    "rsd_l0_b2": 31,
    "rsd_l0_b1_bG2": 32,
    "rsd_l0_bG2": 33,
    "rsd_l2_b1_b2": 34,
    "rsd_l2_b2": 35,
    "rsd_l2_b1_bG2": 36,
    "rsd_l2_bG2": 37,
    "rsd_l4_b2": 38,
    "rsd_l4_bG2": 39,
}

BASIS_COMPONENT_NAMES = tuple(PK_MULT_INDEX.keys()) + ("k_over_h_squared",)


def predict_classpt_multipoles(
    cosmo: Any,
    k: np.ndarray,
    z: float,
    params: EFTBiasParams,
    *,
    metadata: dict[str, Any] | None = None,
) -> MultipolePrediction:
    eval_k = np.asarray(k, dtype=float)
    cosmo.initialize_output(eval_k, float(z), len(eval_k))

    prediction_metadata = {"backend": "classpt", "z": float(z)}
    if metadata:
        prediction_metadata.update(metadata)

    return MultipolePrediction(
        k=eval_k,
        p0=np.asarray(cosmo.pk_gg_l0(params.b1, params.b2, params.bG2, params.bGamma3, params.cs0, params.Pshot, params.b4)),
        p2=np.asarray(cosmo.pk_gg_l2(params.b1, params.b2, params.bG2, params.bGamma3, params.cs2, params.b4)),
        p4=np.asarray(cosmo.pk_gg_l4(params.b1, params.b2, params.bG2, params.bGamma3, params.cs4, params.b4)),
        metadata=prediction_metadata,
    )
