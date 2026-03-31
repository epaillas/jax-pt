from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_plot_triangle_chain_script_smoke(tmp_path: Path) -> None:
    pytest.importorskip("getdist")

    chain = tmp_path / "chain.npz"
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=[1.0, 2.0], scale=[0.05, 0.08], size=(256, 2))
    weights = np.full(samples.shape[0], 1.0 / samples.shape[0], dtype=float)
    np.savez(
        chain,
        samples=samples,
        weights=weights,
        logl=np.linspace(-2.0, -1.0, samples.shape[0], dtype=float),
        logp=np.linspace(-2.5, -1.5, samples.shape[0], dtype=float),
        parameter_names=np.asarray(["b1", "b2"], dtype=str),
        metadata_json=np.asarray('{"observable":"pgg"}'),
    )

    output = tmp_path / "triangle.png"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "plot_triangle_chain.py"),
            str(chain),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert "plot:" in result.stdout
