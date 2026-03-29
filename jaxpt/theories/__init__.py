"""Theory interfaces grouped by observable."""

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
    "predict_galaxy_multipoles",
]
