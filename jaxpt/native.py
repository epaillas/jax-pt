from __future__ import annotations

import numpy as np

from .config import PTSettings
from .cosmology import LinearPowerInput
from .kernels import compute_tree_level_basis
from .reference.classpt import BasisSpectra


def compute_basis(
    linear_input: LinearPowerInput,
    settings: PTSettings | None = None,
    k: np.ndarray | None = None,
) -> BasisSpectra:
    """Compute a native `jaxpt` basis from linear inputs.

    The native backend supports explicit theory-order selection through
    ``settings.loop_order``. Tree-level and one-loop real-space predictions
    share the same basis-construction interface so the public `BasisSpectra`
    contract stays stable as more native kernels are added.
    """
    if settings is None:
        settings = PTSettings(ir_resummation=False)

    if settings.backend != "native":
        raise ValueError("Native compute_basis only supports settings.backend == 'native'.")
    if settings.loop_order not in {"tree", "one_loop"}:
        raise ValueError("Native compute_basis only supports settings.loop_order in {'tree', 'one_loop'}.")

    if settings.ap_effect:
        raise NotImplementedError("AP support is not implemented in the native backend yet.")
    if settings.ir_resummation:
        if settings.require_nowiggle and linear_input.pk_nowiggle is None:
            raise ValueError("pk_nowiggle is required when require_nowiggle=True.")
        raise NotImplementedError("Native IR-resummed one-loop kernels are not implemented yet.")
    if not settings.rsd:
        raise NotImplementedError("The native backend currently targets redshift-space outputs.")

    output_k = None if k is None else np.asarray(k, dtype=float)
    basis = compute_tree_level_basis(linear_input, settings, output_k=output_k)

    if settings.kmin is None and settings.kmax is None:
        return basis

    kmin = settings.kmin if settings.kmin is not None else float(np.asarray(basis.k)[0])
    kmax = settings.kmax if settings.kmax is not None else float(np.asarray(basis.k)[-1])
    mask = (np.asarray(basis.k) >= kmin) & (np.asarray(basis.k) <= kmax)
    components = {name: values[mask] for name, values in basis.components.items()}
    return BasisSpectra(
        k=basis.k[mask],
        z=basis.z,
        h=basis.h,
        growth_rate=basis.growth_rate,
        components=components,
        metadata=basis.metadata,
    )
