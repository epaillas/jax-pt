from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import files

import yaml

from ..parameter import ParameterCollection


def _load_yaml_parameters(filename: str, *, kind_key: str) -> ParameterCollection:
    resource = files("jaxpt.theories").joinpath(filename)
    payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{filename} must define a top-level mapping.")
    if kind_key not in payload:
        raise ValueError(f"{filename} must define '{kind_key}'.")
    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError(f"{filename} must define a non-empty 'parameters' mapping.")
    normalized: dict[str, dict[str, object]] = {}
    for name, value in parameters.items():
        if not isinstance(name, str):
            raise ValueError(f"{filename} contains a non-string parameter name.")
        if isinstance(value, Mapping):
            entry = dict(value)
            if "value" not in entry:
                raise ValueError(f"{filename} parameter '{name}' must define a 'value' field.")
        else:
            entry = {"value": value}
        try:
            entry["value"] = float(entry["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{filename} parameter '{name}' must have a numeric default value.") from exc
        normalized[name] = entry
    return ParameterCollection(normalized)


def load_power_spectrum_template_parameters() -> ParameterCollection:
    return _load_yaml_parameters("power_spectrum_template.yaml", kind_key="template")


def load_galaxy_power_spectrum_multipoles_parameters() -> ParameterCollection:
    return _load_yaml_parameters("galaxy_power_spectrum_multipoles.yaml", kind_key="theory")
