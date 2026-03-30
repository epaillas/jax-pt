"""Theory interfaces grouped by observable."""

from .defaults import (
    load_galaxy_power_spectrum_multipoles_parameters,
    load_power_spectrum_template_parameters,
)
from .power_spectrum import (
    ClassPTGalaxyPowerSpectrumMultipolesTheory,
    GalaxyPowerSpectrumMultipolesTheory,
    PowerSpectrumTemplate,
)

__all__ = [
    "ClassPTGalaxyPowerSpectrumMultipolesTheory",
    "GalaxyPowerSpectrumMultipolesTheory",
    "PowerSpectrumTemplate",
    "load_galaxy_power_spectrum_multipoles_parameters",
    "load_power_spectrum_template_parameters",
]
