"""Theory interfaces grouped by observable."""

from .defaults import (
    load_density_split_galaxy_power_spectrum_multipoles_parameters,
    load_galaxy_power_spectrum_multipoles_parameters,
    load_power_spectrum_template_parameters,
)
from .density_split import QuantileGalaxyPowerSpectrumMultipolesTheory
from .power_spectrum import (
    ClassPTGalaxyPowerSpectrumMultipolesTheory,
    GalaxyPowerSpectrumMultipolesTheory,
    PowerSpectrumTemplate,
)

__all__ = [
    "ClassPTGalaxyPowerSpectrumMultipolesTheory",
    "GalaxyPowerSpectrumMultipolesTheory",
    "PowerSpectrumTemplate",
    "QuantileGalaxyPowerSpectrumMultipolesTheory",
    "load_density_split_galaxy_power_spectrum_multipoles_parameters",
    "load_galaxy_power_spectrum_multipoles_parameters",
    "load_power_spectrum_template_parameters",
]
