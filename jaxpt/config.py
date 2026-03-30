from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PTSettings:
    backend: str = "native"
    loop_order: str = "one_loop"
    ir_resummation: bool = True
    cb: bool = True
    rsd: bool = True
    ap_effect: bool = False
    return_components: bool = False
    validate_units: bool = True
    kmin: float | None = None
    kmax: float | None = None
    integration_nk: int = 256
    fftlog_n: int = 256
    fftlog_bias: float = 0.0
    fftlog_k0_over_h: float = 5.0e-5
    fftlog_kmax_over_h: float = 1.0e2
    fftlog_bias_matter: float = -0.3
    fftlog_bias_bias: float = -1.6000001
    native_kernel_cache: bool = True
    native_kernel_source: str = "analytic"
    require_nowiggle: bool = False
