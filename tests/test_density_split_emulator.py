from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from jaxpt import LinearPowerInput, PTSettings, TaylorEmulator, build_multipole_emulator
from jaxpt.theories import PowerSpectrumTemplate, QuantileGalaxyPowerSpectrumMultipolesTheory


def _linear_input() -> LinearPowerInput:
    return LinearPowerInput(
        k=np.logspace(-3.0, 0.0, 64),
        pk_linear=np.linspace(2.0e4, 5.0e2, 64),
        z=0.5,
        growth_factor=0.76,
        growth_rate=0.81,
        h=0.67,
    )


def test_build_multipole_emulator_supports_density_split_theory(tmp_path) -> None:
    theory = QuantileGalaxyPowerSpectrumMultipolesTheory(
        template=PowerSpectrumTemplate.from_linear_input(_linear_input(), settings=PTSettings(ir_resummation=False)),
        k=np.linspace(0.02, 0.18, 8),
    )
    for name in theory.template.params.names():
        theory.params[name].update(fixed=True)

    emulator = build_multipole_emulator(
        theory,
        order=1,
        step_sizes=0.05,
        cache_dir=tmp_path,
        param_names=["bq1", "beta1"],
    )

    prediction = emulator.predict({"bq1": theory.params["bq1"].value, "beta1": theory.params["beta1"].value})
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (5, 3, 8)
    assert emulator.cache_path is not None


def test_build_taylor_emulator_script_supports_density_split(tmp_path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_taylor_emulator.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--observable",
            "density_split",
            "--order",
            "1",
            "--nk",
            "4",
            "--param",
            "bq1",
            "--param",
            "beta1",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    outputs = sorted(tmp_path.glob("taylor_*.npz"))
    assert len(outputs) == 1
    assert "observable: density_split" in result.stdout
    assert "QuantileGalaxyPowerSpectrumMultipolesTheory" in result.stdout

    loaded = TaylorEmulator.load(outputs[0])
    prediction = loaded.predict({"bq1": -1.6, "beta1": -0.8})
    assert prediction.shape == (5, 3, 4)
