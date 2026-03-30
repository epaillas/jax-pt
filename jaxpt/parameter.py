from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Parameter:
    name: str
    value: float
    fixed: bool = False
    marginalized: bool = False
    prior: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.value = float(self.value)
        self.fixed = bool(self.fixed)
        self.marginalized = bool(self.marginalized)
        self.prior = None if self.prior is None else dict(self.prior)
        self.metadata = dict(self.metadata)

    @property
    def varied(self) -> bool:
        return not self.fixed

    @property
    def emulated(self) -> bool:
        return (not self.fixed) and (not self.marginalized)

    def update(self, **kwargs: Any) -> Parameter:
        for name, value in kwargs.items():
            if name == "name":
                self.name = str(value)
            elif name == "value":
                self.value = float(value)
            elif name == "fixed":
                self.fixed = bool(value)
            elif name == "marginalized":
                self.marginalized = bool(value)
            elif name == "prior":
                self.prior = None if value is None else dict(value)
            elif name == "metadata":
                self.metadata = dict(value)
            else:
                self.metadata[str(name)] = value
        return self

    def copy(self) -> Parameter:
        return Parameter(
            name=self.name,
            value=self.value,
            fixed=self.fixed,
            marginalized=self.marginalized,
            prior=None if self.prior is None else dict(self.prior),
            metadata=dict(self.metadata),
        )


class ParameterCollection:
    def __init__(self, parameters: Mapping[str, Any] | Iterable[Any] | None = None) -> None:
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        if parameters is None:
            return
        if isinstance(parameters, Mapping):
            iterable = parameters.items()
        else:
            iterable = parameters

        for item in iterable:
            if isinstance(item, Parameter):
                parameter = item
            elif isinstance(item, tuple) and len(item) == 2:
                name, payload = item
                parameter = self._coerce_parameter(name, payload)
            else:
                parameter = self._coerce_parameter(None, item)
            self.set(parameter)

    @staticmethod
    def _coerce_parameter(name: str | None, payload: Any) -> Parameter:
        if isinstance(payload, Parameter):
            return payload
        if isinstance(payload, Mapping):
            data = dict(payload)
            parameter_name = str(data.pop("name", name))
            if "value" not in data:
                raise ValueError(f"Parameter '{parameter_name}' must define a 'value'.")
            return Parameter(name=parameter_name, value=data.pop("value"), **data)
        if name is None:
            raise TypeError("ParameterCollection requires explicit parameter names for non-mapping payloads.")
        return Parameter(name=str(name), value=float(payload))

    @classmethod
    def combine(cls, *collections: ParameterCollection) -> ParameterCollection:
        merged = cls()
        for collection in collections:
            for parameter in collection:
                merged.set(parameter)
        return merged

    def copy(self, *, shared: bool = False) -> ParameterCollection:
        copied = self.__class__()
        for parameter in self:
            copied.set(parameter if shared else parameter.copy())
        return copied

    def set(self, parameter: Parameter) -> None:
        self._parameters[str(parameter.name)] = parameter

    def update(self, parameters: Mapping[str, Any] | Iterable[Any] | ParameterCollection) -> ParameterCollection:
        other = parameters if isinstance(parameters, ParameterCollection) else ParameterCollection(parameters)
        for parameter in other:
            self.set(parameter)
        return self

    def defaults_dict(self) -> dict[str, float]:
        return {name: parameter.value for name, parameter in self._parameters.items()}

    def names(self) -> tuple[str, ...]:
        return tuple(self._parameters)

    def values(self) -> tuple[Parameter, ...]:
        return tuple(self._parameters.values())

    def items(self) -> tuple[tuple[str, Parameter], ...]:
        return tuple(self._parameters.items())

    def fixed_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self._parameters.items() if parameter.fixed)

    def varied_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self._parameters.items() if parameter.varied)

    def marginalized_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self._parameters.items() if parameter.marginalized)

    def emulated_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self._parameters.items() if parameter.emulated)

    def __getitem__(self, name: str) -> Parameter:
        return self._parameters[str(name)]

    def __contains__(self, name: object) -> bool:
        return str(name) in self._parameters if isinstance(name, str) else False

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self._parameters.values())

    def __len__(self) -> int:
        return len(self._parameters)

    def __repr__(self) -> str:
        names = ", ".join(self._parameters)
        return f"{self.__class__.__name__}([{names}])"
