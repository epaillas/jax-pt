import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jaxpt import EFTBiasParams


FIDUCIAL_COSMOLOGY = {
    "A_s": 2.089e-9,
    "n_s": 0.9649,
    "tau_reio": 0.052,
    "omega_b": 0.02237,
    "omega_cdm": 0.12,
    "h": 0.6736,
    "YHe": 0.2425,
    "N_ur": 2.0328,
    "N_ncdm": 1,
    "m_ncdm": 0.06,
}

SHIFTED_COSMOLOGY = {
    **FIDUCIAL_COSMOLOGY,
    "omega_cdm": 0.118,
    "n_s": 0.972,
    "h": 0.68,
}

DEFAULT_PT_OPTIONS = {
    "output": "mPk",
    "non linear": "PT",
    "IR resummation": "Yes",
    "Bias tracers": "Yes",
    "cb": "Yes",
    "RSD": "Yes",
}

DEFAULT_PT_OPTIONS_NOIR = {
    **DEFAULT_PT_OPTIONS,
    "IR resummation": "No",
}


@pytest.fixture
def benchmark_k() -> np.ndarray:
    return np.logspace(-3, -0.1, 48)


@pytest.fixture(params=[0.5, 1.0], ids=["z0p5", "z1p0"])
def benchmark_redshift(request) -> float:
    return float(request.param)


@pytest.fixture(params=["fiducial", "shifted"])
def benchmark_cosmology(request) -> dict[str, float]:
    if request.param == "fiducial":
        return FIDUCIAL_COSMOLOGY
    return SHIFTED_COSMOLOGY


@pytest.fixture(params=["low_bias", "high_bias"])
def benchmark_params(request) -> EFTBiasParams:
    if request.param == "low_bias":
        return EFTBiasParams(b1=1.7, b2=-0.4, bG2=0.1, bGamma3=-0.1, cs0=0.0, cs2=25.0, cs4=0.0, Pshot=1800.0, b4=6.0)
    return EFTBiasParams(b1=2.2, b2=-1.2, bG2=0.2, bGamma3=-0.15, cs0=0.0, cs2=35.0, cs4=0.0, Pshot=3200.0, b4=10.0)
