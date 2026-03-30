from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Parameter:
    """Single scalar theory parameter with emulation and prior metadata.

    Parameters
    ----------
    name
        Parameter name used in flat theory and emulator queries, for example
        ``"omega_cdm"`` or ``"b1"``.
    value
        Fiducial/default value.
    fixed
        If ``True``, the parameter is treated as fixed and is excluded from
        default emulator parameter sets.
    marginalized
        If ``True``, the parameter is considered analytically marginalized and
        is also excluded from default emulator parameter sets.
    prior
        Optional prior description copied verbatim from YAML or user input.
    metadata
        Free-form metadata that accompanies the parameter definition.
    """

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
        """Whether the parameter is not fixed."""
        return not self.fixed

    @property
    def emulated(self) -> bool:
        """Whether the parameter should enter default emulator expansions."""
        return (not self.fixed) and (not self.marginalized)

    def update(self, **kwargs: Any) -> Parameter:
        """Update the parameter in place and return `self`.

        Recognized keyword arguments are ``name``, ``value``, ``fixed``,
        ``marginalized``, ``prior``, and ``metadata``. Any other keyword is
        stored inside ``metadata`` under the provided name.
        """
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
        """Return a detached copy of the parameter."""
        return Parameter(
            name=self.name,
            value=self.value,
            fixed=self.fixed,
            marginalized=self.marginalized,
            prior=None if self.prior is None else dict(self.prior),
            metadata=dict(self.metadata),
        )


class ParameterCollection:
    """Ordered collection of named `Parameter` objects.

    Parameters
    ----------
    parameters
        Optional source used to populate the collection. Accepted forms are:

        - ``None`` for an empty collection;
        - a mapping from parameter names to `Parameter` instances, scalar
          values, or mapping payloads containing at least ``value``;
        - an iterable of `Parameter` instances or ``(name, payload)`` pairs.
    """

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
        """Merge multiple collections, with later entries overriding earlier ones."""
        merged = cls()
        for collection in collections:
            for parameter in collection:
                merged.set(parameter)
        return merged

    def copy(self, *, shared: bool = False) -> ParameterCollection:
        """Copy the collection.

        Parameters
        ----------
        shared
            If ``True``, reuse the same `Parameter` objects. If ``False``,
            clone each parameter.
        """
        copied = self.__class__()
        for parameter in self:
            copied.set(parameter if shared else parameter.copy())
        return copied

    def set(self, parameter: Parameter) -> None:
        """Insert or replace a parameter by name."""
        self._parameters[str(parameter.name)] = parameter

    def update(self, parameters: Mapping[str, Any] | Iterable[Any] | ParameterCollection) -> ParameterCollection:
        """Merge another parameter source into the collection."""
        other = parameters if isinstance(parameters, ParameterCollection) else ParameterCollection(parameters)
        for parameter in other:
            self.set(parameter)
        return self

    def defaults_dict(self) -> dict[str, float]:
        """Return the current parameter values as a plain mapping."""
        return {name: parameter.value for name, parameter in self._parameters.items()}

    def names(self) -> tuple[str, ...]:
        """Return parameter names in insertion order."""
        return tuple(self._parameters)

    def values(self) -> tuple[Parameter, ...]:
        """Return parameters in insertion order."""
        return tuple(self._parameters.values())

    def items(self) -> tuple[tuple[str, Parameter], ...]:
        """Return ``(name, parameter)`` pairs in insertion order."""
        return tuple(self._parameters.items())

    def fixed_names(self) -> tuple[str, ...]:
        """Return the names of parameters marked fixed."""
        return tuple(name for name, parameter in self._parameters.items() if parameter.fixed)

    def varied_names(self) -> tuple[str, ...]:
        """Return the names of parameters that are not fixed."""
        return tuple(name for name, parameter in self._parameters.items() if parameter.varied)

    def marginalized_names(self) -> tuple[str, ...]:
        """Return the names of parameters flagged for marginalization."""
        return tuple(name for name, parameter in self._parameters.items() if parameter.marginalized)

    def emulated_names(self) -> tuple[str, ...]:
        """Return the default parameter subset for emulator expansion."""
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
