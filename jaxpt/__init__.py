from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .bias import galaxy_multipoles, galaxy_real_spectrum, matter_real_spectrum
from .basis import build_realspace_predictor, compute_basis
from .config import PTSettings
from .cosmology import LinearPowerInput, build_linear_input_from_classy, build_linear_input_from_cosmoprimo
from .emulators import TaylorEmulator, build_multipole_emulator
from .parameter import Parameter, ParameterCollection
from .reference.classpt import BasisSpectra, MultipolePrediction
from .reference.parity import compare_multipoles_to_classpt, compare_predictions
from .utils import covariance_errors, flatten_pgg_measurements, load_pgg_data_vector, load_pgg_mock_matrix, sample_covariance
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
    "build_multipole_emulator",
    "build_linear_input_from_classy",
    "build_linear_input_from_cosmoprimo",
    "build_realspace_predictor",
    "compare_multipoles_to_classpt",
    "compare_predictions",
    "covariance_errors",
    "compute_basis",
    "flatten_pgg_measurements",
    "galaxy_multipoles",
    "galaxy_real_spectrum",
    "load_pgg_data_vector",
    "load_pgg_mock_matrix",
    "load_galaxy_power_spectrum_multipoles_parameters",
    "load_power_spectrum_template_parameters",
    "matter_real_spectrum",
    "sample_covariance",
]
