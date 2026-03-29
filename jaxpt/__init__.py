from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .api import build_native_realspace_predictor, compute_basis, galaxy_multipoles, galaxy_real_spectrum, matter_real_spectrum, predict_galaxy_multipoles
from .config import EFTBiasParams, PTSettings
from .cosmology import LinearPowerInput, build_linear_input_from_classy, build_linear_input_from_cosmoprimo
from .reference.classpt import BasisSpectra, MultipolePrediction
from .reference.parity import compare_multipoles_to_classpt, compare_predictions
from .theory import GalaxyPowerSpectrumMultipolesTheory, PowerSpectrumTemplate

__all__ = [
    "BasisSpectra",
    "EFTBiasParams",
    "GalaxyPowerSpectrumMultipolesTheory",
    "LinearPowerInput",
    "MultipolePrediction",
    "PTSettings",
    "PowerSpectrumTemplate",
    "build_linear_input_from_classy",
    "build_linear_input_from_cosmoprimo",
    "build_native_realspace_predictor",
    "compare_multipoles_to_classpt",
    "compare_predictions",
    "compute_basis",
    "galaxy_multipoles",
    "galaxy_real_spectrum",
    "matter_real_spectrum",
    "predict_galaxy_multipoles",
]
