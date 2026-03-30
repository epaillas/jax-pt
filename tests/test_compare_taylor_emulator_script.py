from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from jaxpt import PTSettings, build_multipole_emulator
from jaxpt.theories import (
    GalaxyPowerSpectrumMultipolesTheory,
    PowerSpectrumTemplate,
    load_galaxy_power_spectrum_multipoles_parameters,
    load_power_spectrum_template_parameters,
)


def test_compare_taylor_emulator_script_smoke(tmp_path) -> None:
    eval_k = np.linspace(0.02, 0.18, 6)

    cosmology_defaults = load_power_spectrum_template_parameters().defaults_dict()
    nuisance_defaults = load_galaxy_power_spectrum_multipoles_parameters().defaults_dict()
    template = PowerSpectrumTemplate(
        cosmology_defaults,
        z=0.5,
        settings=PTSettings(backend="jaxpt", ir_resummation=False),
        provider="cosmoprimo",
    )
    theory = GalaxyPowerSpectrumMultipolesTheory(template=template, k=eval_k)
    for name, value in nuisance_defaults.items():
        theory.params[name].update(value=value)
    for name in theory.template.params.names():
        theory.params[name].update(fixed=name not in {"A_s", "omega_cdm"})

    emulator = build_multipole_emulator(
        theory,
        order=1,
        step_sizes=0.01,
        cache_dir=tmp_path,
        metadata={"script": "test_compare_taylor_emulator_script"},
    )
    assert emulator.cache_path is not None

    script = Path(__file__).resolve().parents[1] / "scripts" / "compare_taylor_emulator.py"
    output = tmp_path / "comparison.png"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(emulator.cache_path),
            "--param",
            "A_s=2.05e-9",
            "--param",
            "omega_cdm=0.121",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.exists()
    assert "Taylor emulator comparison" in result.stdout
    assert "P0:" in result.stdout
    assert "P2:" in result.stdout
    assert "P4:" in result.stdout
