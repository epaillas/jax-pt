from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PTSettings:
    """Configuration for `jaxpt` and reference power-spectrum calculations.

    Parameters
    ----------
    backend
        Theory backend to use. Allowed values are ``"jaxpt"`` for the in-repo
        implementation and ``"classpt"`` for direct CLASS-PT predictions.
    loop_order
        Perturbative order for the `jaxpt` backend. Allowed values are
        ``"tree"`` and ``"one_loop"``.
    ir_resummation
        Whether to request IR-resummed calculations. The flag is accepted for
        API compatibility, but the current `jaxpt` backend raises
        ``NotImplementedError`` when it is enabled.
    cb
        Whether the calculation targets the CDM+baryon field. This is mainly
        propagated to CLASS-backed providers and metadata.
    rsd
        Whether to compute redshift-space distortions. The current `jaxpt`
        backend expects ``True``.
    ap_effect
        Whether to enable Alcock-Paczynski distortions. The current `jaxpt`
        backend does not implement this yet.
    return_components
        Whether high-level theory calls should expose decomposed multipole
        components when the backend supports them.
    validate_units
        Whether helper constructors should validate recorded unit metadata.
    kmin, kmax
        Optional post-processing cuts on the output grid, in ``1/Mpc``.
    integration_nk
        Number of support-grid samples used when a cosmology provider builds
        its default linear power spectrum grid.
    fftlog_n
        Number of FFTLog samples used by the analytic one-loop kernels.
    fftlog_bias
        Extra additive shift applied to the FFTLog bias exponents.
    fftlog_k0_over_h, fftlog_kmax_over_h
        Minimum and maximum FFTLog support wavenumbers in units of ``h/Mpc``.
    fftlog_bias_matter, fftlog_bias_bias
        Base FFTLog bias exponents for matter-like and bias-like loop kernels.
    kernel_cache
        Whether cached analytic kernel registries may be reused.
    kernel_source
        Kernel implementation to use. The current code supports
        ``"analytic"``.
    require_nowiggle
        Whether IR-resummed code paths should require a nowiggle spectrum in
        `LinearPowerInput.pk_nowiggle`.
    """

    backend: str = "jaxpt"
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
    kernel_cache: bool = True
    kernel_source: str = "analytic"
    require_nowiggle: bool = False
