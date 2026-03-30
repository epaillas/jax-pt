"""Theory interfaces grouped by observable."""

from .defaults import load_galaxy_power_spectrum_multipoles_defaults, load_power_spectrum_template_defaults
from .power_spectrum import (
    ClassPTGalaxyPowerSpectrumMultipolesTheory,
    GalaxyPowerSpectrumMultipolesTheory,
    PowerSpectrumTemplate,
    predict_galaxy_multipoles,
)

__all__ = [
    "ClassPTGalaxyPowerSpectrumMultipolesTheory",
    "GalaxyPowerSpectrumMultipolesTheory",
    "PowerSpectrumTemplate",
    "load_galaxy_power_spectrum_multipoles_defaults",
    "load_power_spectrum_template_defaults",
    "predict_galaxy_multipoles",
]
