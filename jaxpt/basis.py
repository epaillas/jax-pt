from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .reference.classpt import BASIS_COMPONENT_NAMES, BasisSpectra


def empty_components(k: jnp.ndarray) -> dict[str, jnp.ndarray]:
    zeros = jnp.zeros_like(k)
    return {name: zeros for name in BASIS_COMPONENT_NAMES}


def make_basis(
    *,
    k: jnp.ndarray,
    z: float,
    h: float,
    growth_rate: float,
    components: dict[str, jnp.ndarray],
    metadata: dict[str, Any] | None = None,
) -> BasisSpectra:
    payload = empty_components(k)
    payload.update(components)
    return BasisSpectra(
        k=k,
        z=float(z),
        h=float(h),
        growth_rate=float(growth_rate),
        components=payload,
        metadata=dict(metadata or {}),
    )
