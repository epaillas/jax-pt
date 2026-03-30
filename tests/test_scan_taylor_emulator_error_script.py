from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from jaxpt import PTSettings, build_multipole_emulator
from jaxpt.theories import (
    GalaxyPowerSpectrumMultipolesTheory,
    PowerSpectrumTemplate,
    load_galaxy_power_spectrum_multipoles_parameters,
    load_power_spectrum_template_parameters,
)


def _build_small_emulator(tmp_path: Path):
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
        metadata={"script": "test_scan_taylor_emulator_error_script"},
    )
    assert emulator.cache_path is not None
    return emulator


def test_scan_taylor_emulator_error_script_smoke(tmp_path) -> None:
    emulator = _build_small_emulator(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "scan_taylor_emulator_error.py"
    output = tmp_path / "error_scan.png"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(emulator.cache_path),
            "--param",
            "A_s",
            "--param",
            "omega_cdm",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.exists()
    assert "Taylor emulator error scan" in result.stdout
    assert "parameters: A_s, omega_cdm" in result.stdout
    assert "A_s: step=" in result.stdout
    assert "omega_cdm: step=" in result.stdout


def test_scan_taylor_emulator_error_script_rejects_non_emulated_parameter(tmp_path) -> None:
    emulator = _build_small_emulator(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "scan_taylor_emulator_error.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(emulator.cache_path),
            "--param",
            "n_s",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "not emulated" in result.stderr


@pytest.mark.parametrize(
    ("n_samples", "expected"),
    [
        (2, np.array([-3.0, 3.0])),
        (6, np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])),
    ],
)
def test_scan_script_sampling_grid(n_samples, expected) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "scan_taylor_emulator_error.py"
    sys.path.insert(0, str(script.parent))
    try:
        spec = importlib.util.spec_from_file_location("scan_taylor_emulator_error", script)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    grid = module._scan_multiples(3.0, n_samples)
    np.testing.assert_allclose(grid, expected, rtol=0.0, atol=0.0)
