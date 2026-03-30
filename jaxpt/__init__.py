from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .bias import galaxy_multipoles, galaxy_real_spectrum, matter_real_spectrum
from .config import PTSettings
from .cosmology import LinearPowerInput, build_linear_input_from_classy, build_linear_input_from_cosmoprimo
from .emulators import TaylorEmulator, build_native_multipole_taylor_emulator
from .native import build_native_realspace_predictor, compute_basis
from .parameter import Parameter, ParameterCollection
from .reference.classpt import BasisSpectra, MultipolePrediction
from .reference.parity import compare_multipoles_to_classpt, compare_predictions
from .theories import (
    ClassPTGalaxyPowerSpectrumMultipolesTheory,
    GalaxyPowerSpectrumMultipolesTheory,
    PowerSpectrumTemplate,
    load_galaxy_power_spectrum_multipoles_parameters,
    load_power_spectrum_template_parameters,
)

__all__ = [
    "BasisSpectra",
    "ClassPTGalaxyPowerSpectrumMultipolesTheory",
    "GalaxyPowerSpectrumMultipolesTheory",
    "LinearPowerInput",
    "MultipolePrediction",
    "Parameter",
    "ParameterCollection",
    "PTSettings",
    "PowerSpectrumTemplate",
    "TaylorEmulator",
    "build_native_multipole_taylor_emulator",
    "build_linear_input_from_classy",
    "build_linear_input_from_cosmoprimo",
    "build_native_realspace_predictor",
    "compare_multipoles_to_classpt",
    "compare_predictions",
    "compute_basis",
    "galaxy_multipoles",
    "galaxy_real_spectrum",
    "load_galaxy_power_spectrum_multipoles_parameters",
    "load_power_spectrum_template_parameters",
    "matter_real_spectrum",
]
