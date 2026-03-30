from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import files

import yaml


def _load_yaml_defaults(filename: str, *, kind_key: str) -> dict[str, float]:
    resource = files("jaxpt.theories").joinpath(filename)
    payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{filename} must define a top-level mapping.")
    if kind_key not in payload:
        raise ValueError(f"{filename} must define '{kind_key}'.")
    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError(f"{filename} must define a non-empty 'parameters' mapping.")

    normalized: dict[str, float] = {}
    for name, value in parameters.items():
        if not isinstance(name, str):
            raise ValueError(f"{filename} contains a non-string parameter name.")
        try:
            normalized[name] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{filename} parameter '{name}' must have a numeric default value.") from exc
    return normalized


def load_power_spectrum_template_defaults() -> dict[str, float]:
    return _load_yaml_defaults("power_spectrum_template.yaml", kind_key="template")


def load_galaxy_power_spectrum_multipoles_defaults() -> dict[str, float]:
    return _load_yaml_defaults("galaxy_power_spectrum_multipoles.yaml", kind_key="theory")
