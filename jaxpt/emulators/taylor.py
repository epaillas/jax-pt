from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import factorial
from pathlib import Path
from typing import Any

import numpy as np

from ..reference.classpt import MultipolePrediction


_SERIALIZATION_VERSION = 1


ProgressCallback = Callable[[int, int], None]


def _normalize_query(
    parameters: Mapping[str, float] | None,
    kwargs: Mapping[str, float],
) -> dict[str, float]:
    if parameters is None:
        query: dict[str, float] = {}
    elif isinstance(parameters, Mapping):
        query = {str(name): float(value) for name, value in parameters.items()}
    else:
        raise TypeError("Emulator parameters must be provided as a mapping or flat keyword arguments.")

    overlap = sorted(set(query) & set(kwargs))
    if overlap:
        raise ValueError(f"Duplicate query parameters provided both positionally and by keyword: {', '.join(overlap)}.")

    for name, value in kwargs.items():
        query[str(name)] = float(value)
    return query


def _freeze_cache_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((str(name), _freeze_cache_value(item)) for name, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value)
        return tuple(float(item) for item in value.reshape(-1))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return repr(value)


def _factorial_product(powers: np.ndarray) -> int:
    result = 1
    for power in powers:
        result *= factorial(int(power))
    return result


def _enumerate_multi_indices(n_params: int, order: int) -> np.ndarray:
    result: list[tuple[int, ...]] = []

    def recurse(depth: int, remaining: int, current: list[int]) -> None:
        if depth == n_params:
            result.append(tuple(current))
            return
        for power in range(remaining + 1):
            current.append(power)
            recurse(depth + 1, remaining - power, current)
            current.pop()

    recurse(0, int(order), [])
    return np.asarray(result, dtype=int)


def _fornberg_weights(center: float, stencil_points: np.ndarray, max_deriv: int) -> np.ndarray:
    n_points = len(stencil_points)
    weights = np.zeros((max_deriv + 1, n_points), dtype=float)
    weights[0, 0] = 1.0
    c1 = 1.0
    for i in range(1, n_points):
        c2 = 1.0
        for j in range(i):
            c3 = float(stencil_points[i] - stencil_points[j])
            c2 *= c3
            for deriv in range(min(i, max_deriv), 0, -1):
                weights[deriv, i] = c1 * (deriv * weights[deriv - 1, i - 1] - (stencil_points[i - 1] - center) * weights[deriv, i - 1]) / c2
            weights[0, i] = -c1 * (stencil_points[i - 1] - center) * weights[0, i - 1] / c2
            for deriv in range(min(i, max_deriv), 0, -1):
                weights[deriv, j] = ((stencil_points[i] - center) * weights[deriv, j] - deriv * weights[deriv - 1, j]) / c3
            weights[0, j] = (stencil_points[i] - center) * weights[0, j] / c3
        c1 = c2
    return weights


def _central_stencil(derivative_order: int, accuracy: int) -> tuple[np.ndarray, np.ndarray]:
    half_width = max((derivative_order + 1) // 2 + 1, (derivative_order + accuracy - 1) // 2 + (derivative_order + accuracy) % 2)
    offsets = np.arange(-half_width, half_width + 1, dtype=int)
    weights = _fornberg_weights(0.0, offsets.astype(float), derivative_order)[derivative_order]
    return offsets, weights


@dataclass(frozen=True, slots=True)
class _OutputAdapter:
    tag: str
    flatten: Callable[[Any], tuple[np.ndarray, dict[str, Any]]]
    reconstruct: Callable[[np.ndarray, dict[str, Any]], Any]


def _flatten_array_output(prediction: Any) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.asarray(prediction, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.reshape(-1), {"shape": list(array.shape)}


def _reconstruct_array_output(vector: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    return np.asarray(vector, dtype=float).reshape(tuple(int(item) for item in state["shape"]))


def _flatten_multipole_prediction(prediction: MultipolePrediction) -> tuple[np.ndarray, dict[str, Any]]:
    if prediction.components is not None:
        raise ValueError("TaylorEmulator does not support MultipolePrediction.components in v1. Pass a custom output_adapter if component emulation is required.")

    k = np.asarray(prediction.k, dtype=float)
    p0 = np.asarray(prediction.p0, dtype=float)
    p2 = np.asarray(prediction.p2, dtype=float)
    p4 = np.asarray(prediction.p4, dtype=float)
    if p0.shape != k.shape or p2.shape != k.shape or p4.shape != k.shape:
        raise ValueError("MultipolePrediction arrays must all share the same shape.")

    vector = np.concatenate([p0.reshape(-1), p2.reshape(-1), p4.reshape(-1)])
    state = {
        "shape": list(k.shape),
        "k": k.reshape(-1).tolist(),
        "metadata": dict(prediction.metadata),
    }
    return vector, state


def _reconstruct_multipole_prediction(vector: np.ndarray, state: dict[str, Any]) -> MultipolePrediction:
    shape = tuple(int(item) for item in state["shape"])
    size = int(np.prod(shape))
    values = np.asarray(vector, dtype=float).reshape(3, size)
    k = np.asarray(state["k"], dtype=float).reshape(shape)
    return MultipolePrediction(
        k=k,
        p0=values[0].reshape(shape),
        p2=values[1].reshape(shape),
        p4=values[2].reshape(shape),
        metadata=dict(state.get("metadata", {})),
    )


_ARRAY_OUTPUT_ADAPTER = _OutputAdapter(
    tag="array",
    flatten=_flatten_array_output,
    reconstruct=_reconstruct_array_output,
)

_MULTIPOLE_OUTPUT_ADAPTER = _OutputAdapter(
    tag="multipole_prediction",
    flatten=_flatten_multipole_prediction,
    reconstruct=_reconstruct_multipole_prediction,
)

_BUILTIN_OUTPUT_ADAPTERS = {
    _ARRAY_OUTPUT_ADAPTER.tag: _ARRAY_OUTPUT_ADAPTER,
    _MULTIPOLE_OUTPUT_ADAPTER.tag: _MULTIPOLE_OUTPUT_ADAPTER,
}


def _resolve_output_adapter(output_adapter: _OutputAdapter | None, sample_output: Any | None = None) -> _OutputAdapter:
    if output_adapter is not None:
        return output_adapter
    if isinstance(sample_output, MultipolePrediction):
        return _MULTIPOLE_OUTPUT_ADAPTER
    return _ARRAY_OUTPUT_ADAPTER


@dataclass(slots=True)
class TaylorEmulator:
    """Multivariate Taylor emulator around a fiducial theory point.

    Parameters
    ----------
    theory_fn
        Callable that accepts a flat parameter mapping and returns either a
        NumPy-like array or a `MultipolePrediction`.
    fiducial
        Full fiducial parameter mapping used as the expansion point.
    order
        Maximum total Taylor order. Must be a non-negative integer.
    step_sizes
        Either a scalar relative step scale or an explicit mapping from each
        emulated parameter name to its finite-difference step size.
    param_names
        Ordered subset of parameters to emulate. If omitted, the emulator uses
        the theory's default emulated subset when available.
    output_adapter
        Optional custom adapter for flattening and reconstructing theory
        outputs.
    cache_dir
        Optional directory where hashed emulator files should be written.
    cache_key
        Optional explicit hash key. If omitted, a deterministic key is derived
        from the emulator configuration.
    finite_difference_accuracy
        Central-stencil accuracy order used for derivative estimates. Must be a
        positive integer.
    metadata
        Free-form build metadata persisted in the serialized emulator.
    valid_param_names
        Full set of valid theory parameters. Parameters outside this set are
        rejected at evaluation time.
    """

    theory_fn: Callable[[Mapping[str, float]], Any] | None = None
    fiducial: Mapping[str, float] = field(default_factory=dict)
    order: int = 4
    step_sizes: float | Mapping[str, float] = 0.01
    param_names: list[str] | tuple[str, ...] | None = None
    output_adapter: _OutputAdapter | None = None
    cache_dir: str | Path | None = None
    cache_key: str | None = None
    finite_difference_accuracy: int = 2
    metadata: Mapping[str, Any] | None = None
    valid_param_names: list[str] | tuple[str, ...] | None = None
    _step_sizes: dict[str, float] = field(init=False, repr=False)
    _powers: np.ndarray = field(init=False, repr=False)
    _stencils: dict[int, tuple[np.ndarray, np.ndarray]] = field(init=False, repr=False)
    _coefficients: np.ndarray | None = field(init=False, default=None, repr=False)
    _output_state: dict[str, Any] | None = field(init=False, default=None, repr=False)
    _output_adapter: _OutputAdapter | None = field(init=False, default=None, repr=False)
    _eval_cache: dict[tuple[int, ...], np.ndarray] = field(init=False, default_factory=dict, repr=False)
    _cache_path: Path | None = field(init=False, default=None, repr=False)
    _valid_param_names: tuple[str, ...] = field(init=False, repr=False)
    _constrained_param_names: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.fiducial = {str(name): float(value) for name, value in dict(self.fiducial).items()}
        self.metadata = {} if self.metadata is None else dict(self.metadata)
        self.order = int(self.order)
        self.finite_difference_accuracy = int(self.finite_difference_accuracy)
        if self.order < 0:
            raise ValueError("TaylorEmulator.order must be non-negative.")
        if self.finite_difference_accuracy < 1:
            raise ValueError("TaylorEmulator.finite_difference_accuracy must be positive.")

        if self.param_names is None:
            if self.theory_fn is not None and hasattr(self.theory_fn, "params") and hasattr(self.theory_fn.params, "emulated_names"):
                emulated_names = [str(name) for name in self.theory_fn.params.emulated_names() if str(name) in self.fiducial]
                self.param_names = emulated_names if emulated_names else list(self.fiducial)
            else:
                self.param_names = list(self.fiducial)
        else:
            self.param_names = [str(name) for name in self.param_names]

        missing = [name for name in self.param_names if name not in self.fiducial]
        if missing:
            raise ValueError(f"TaylorEmulator.param_names are missing from fiducial: {', '.join(missing)}.")
        if len(set(self.param_names)) != len(self.param_names):
            raise ValueError("TaylorEmulator.param_names must be unique.")

        if self.valid_param_names is None:
            self._valid_param_names = tuple(self.fiducial)
        else:
            self._valid_param_names = tuple(str(name) for name in self.valid_param_names)
        if len(set(self._valid_param_names)) != len(self._valid_param_names):
            raise ValueError("TaylorEmulator.valid_param_names must be unique.")

        missing_valid = [name for name in self._valid_param_names if name not in self.fiducial]
        if missing_valid:
            raise ValueError(f"TaylorEmulator.valid_param_names are missing from fiducial: {', '.join(missing_valid)}.")
        unknown_emulated = [name for name in self.param_names if name not in self._valid_param_names]
        if unknown_emulated:
            raise ValueError(f"TaylorEmulator.param_names must be a subset of valid_param_names. Invalid: {', '.join(unknown_emulated)}.")
        self._constrained_param_names = tuple(name for name in self._valid_param_names if name not in set(self.param_names))

        self._step_sizes = self._resolve_step_sizes(self.step_sizes)
        self._powers = _enumerate_multi_indices(len(self.param_names), self.order)
        self._stencils = {
            derivative_order: _central_stencil(derivative_order, self.finite_difference_accuracy)
            for derivative_order in range(1, self.order + 1)
        }
        self._coefficients: np.ndarray | None = None
        self._output_state: dict[str, Any] | None = None
        self._output_adapter = self.output_adapter
        self._eval_cache: dict[tuple[int, ...], np.ndarray] = {}
        self._cache_path: Path | None = None

    def _resolve_step_sizes(self, step_sizes: float | Mapping[str, float]) -> dict[str, float]:
        resolved: dict[str, float] = {}
        if isinstance(step_sizes, Mapping):
            for name in self.param_names:
                if name not in step_sizes:
                    raise ValueError(f"Missing TaylorEmulator step size for parameter '{name}'.")
                resolved[name] = float(step_sizes[name])
        else:
            scale = float(step_sizes)
            for name in self.param_names:
                fiducial_value = float(self.fiducial[name])
                resolved[name] = scale if fiducial_value == 0.0 else scale * abs(fiducial_value)

        invalid = [name for name, value in resolved.items() if not np.isfinite(value) or value == 0.0]
        if invalid:
            raise ValueError(f"TaylorEmulator step sizes must be finite and non-zero. Invalid: {', '.join(invalid)}.")
        return resolved

    @property
    def is_built(self) -> bool:
        """Whether the emulator coefficients and output metadata are available."""
        return self._coefficients is not None and self._output_state is not None and self._output_adapter is not None

    @property
    def n_terms(self) -> int:
        """Number of Taylor monomials included in the expansion."""
        return int(self._powers.shape[0])

    @property
    def n_evals(self) -> int:
        """Number of unique theory evaluations cached during the current build."""
        return len(self._eval_cache)

    @property
    def cache_path(self) -> Path | None:
        """Resolved path to the serialized emulator, if caching is enabled."""
        return self._cache_path

    def _config_payload(self) -> dict[str, Any]:
        return {
            "version": _SERIALIZATION_VERSION,
            "order": self.order,
            "finite_difference_accuracy": self.finite_difference_accuracy,
            "param_names": list(self.param_names),
            "fiducial": {name: self.fiducial[name] for name in self.param_names},
            "step_sizes": {name: self._step_sizes[name] for name in self.param_names},
            "valid_param_names": list(self._valid_param_names),
            "constrained_fiducial": {name: self.fiducial[name] for name in self._constrained_param_names},
            "adapter_tag": None if self._output_adapter is None else self._output_adapter.tag,
            "metadata": dict(self.metadata),
        }

    def _default_cache_key(self) -> str:
        frozen = _freeze_cache_value(self._config_payload())
        payload = json.dumps(frozen, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()

    def _resolve_cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        cache_dir = Path(self.cache_dir)
        key = self.cache_key or self._default_cache_key()
        return cache_dir / f"taylor_{key}.npz"

    def _evaluate_theory(self, offset: tuple[int, ...]) -> np.ndarray:
        if self.theory_fn is None:
            raise ValueError("TaylorEmulator requires theory_fn to build coefficients.")
        if offset in self._eval_cache:
            return self._eval_cache[offset]

        params = dict(self.fiducial)
        for index, name in enumerate(self.param_names):
            params[name] = self.fiducial[name] + offset[index] * self._step_sizes[name]

        raw_output = self.theory_fn(params)
        if self._output_adapter is None:
            self._output_adapter = _resolve_output_adapter(self.output_adapter, raw_output)
        vector, state = self._output_adapter.flatten(raw_output)
        if self._output_state is None:
            self._output_state = state
        elif state != self._output_state:
            raise ValueError("TaylorEmulator theory output metadata changed during build.")

        array = np.asarray(vector, dtype=float).reshape(-1)
        if self._coefficients is not None and array.size != self._coefficients.shape[1]:
            raise ValueError("TaylorEmulator theory output size changed during build.")
        self._eval_cache[offset] = array
        return array

    def _compute_derivative(self, powers: np.ndarray) -> np.ndarray:
        active = [(index, int(power)) for index, power in enumerate(powers) if int(power) > 0]
        if not active:
            return self._evaluate_theory(tuple(0 for _ in self.param_names))

        weighted_offsets: dict[tuple[int, ...], float] = {tuple(0 for _ in self.param_names): 1.0}
        for axis, derivative_order in active:
            offsets_1d, weights_1d = self._stencils[derivative_order]
            scale = self._step_sizes[self.param_names[axis]] ** derivative_order
            updated: dict[tuple[int, ...], float] = {}
            for base_offset, base_weight in weighted_offsets.items():
                for stencil_offset, stencil_weight in zip(offsets_1d, weights_1d, strict=True):
                    if stencil_weight == 0.0:
                        continue
                    values = list(base_offset)
                    values[axis] += int(stencil_offset)
                    key = tuple(values)
                    updated[key] = updated.get(key, 0.0) + base_weight * float(stencil_weight) / scale
            weighted_offsets = updated

        derivative = np.zeros_like(self._evaluate_theory(tuple(0 for _ in self.param_names)))
        for offset, weight in weighted_offsets.items():
            derivative = derivative + weight * self._evaluate_theory(offset)
        return derivative

    def build(self, force: bool = False, progress_callback: ProgressCallback | None = None) -> TaylorEmulator:
        """Build the emulator coefficients, optionally reusing a cached file.

        Parameters
        ----------
        force
            If ``True``, rebuild even when a matching hashed file already
            exists on disk.
        progress_callback
            Optional callable receiving ``(completed_terms, total_terms)``
            after each Taylor term has been constructed.
        """
        self._cache_path = self._resolve_cache_path()
        if not force and self._cache_path is not None and self._cache_path.exists():
            loaded = self.load(self._cache_path, theory_fn=self.theory_fn, output_adapter=self.output_adapter)
            self._coefficients = loaded._coefficients
            self._output_state = loaded._output_state
            self._output_adapter = loaded._output_adapter
            self._cache_path = loaded._cache_path
            self._valid_param_names = loaded._valid_param_names
            self._constrained_param_names = loaded._constrained_param_names
            return self

        self._eval_cache = {}
        fiducial_value = self._evaluate_theory(tuple(0 for _ in self.param_names))
        self._coefficients = np.zeros((self.n_terms, fiducial_value.size), dtype=float)

        for index, powers in enumerate(self._powers):
            derivative = self._compute_derivative(powers)
            self._coefficients[index] = derivative / _factorial_product(powers)
            if progress_callback is not None:
                progress_callback(index + 1, self.n_terms)

        if self._cache_path is not None:
            self.save(self._cache_path)
        return self

    def _predict_vector(self, params: Mapping[str, float]) -> np.ndarray:
        if not self.is_built or self._coefficients is None or self._output_state is None or self._output_adapter is None:
            raise ValueError("TaylorEmulator must be built or loaded before predict().")

        unknown = sorted(set(params) - set(self._valid_param_names))
        if unknown:
            raise ValueError(f"Unexpected emulator parameters: {', '.join(unknown)}.")
        varied_constrained = [
            name
            for name in self._constrained_param_names
            if name in params and not np.isclose(float(params[name]), float(self.fiducial[name]), rtol=0.0, atol=0.0)
        ]
        if varied_constrained:
            raise ValueError(
                "Parameters were provided that are valid for the theory but were not emulated: "
                + ", ".join(varied_constrained)
            )

        deltas = np.asarray([float(params.get(name, self.fiducial[name])) - self.fiducial[name] for name in self.param_names], dtype=float)
        monomials = np.prod(np.power(deltas[None, :], self._powers, dtype=float), axis=1)
        return monomials @ self._coefficients

    def predict(self, parameters: Mapping[str, float] | None = None, **kwargs: float) -> Any:
        """Evaluate the emulator for a flat parameter query.

        Parameters
        ----------
        parameters
            Optional mapping of query values. Names may include any parameter in
            ``valid_param_names``.
        **kwargs
            Keyword-argument form of ``parameters``. This is merged with the
            mapping and must not repeat names.

        Notes
        -----
        Parameters that are valid for the underlying theory but were not
        emulated must remain at their fiducial values. Varying them raises
        ``ValueError``.
        """
        query = _normalize_query(parameters, kwargs)
        vector = self._predict_vector(query)
        assert self._output_adapter is not None
        assert self._output_state is not None
        return self._output_adapter.reconstruct(vector, self._output_state)

    def _serialized_state(self) -> dict[str, Any]:
        if not self.is_built or self._coefficients is None or self._output_state is None or self._output_adapter is None:
            raise ValueError("TaylorEmulator must be built before serialization.")

        payload = self._config_payload()
        payload["adapter_tag"] = self._output_adapter.tag
        return {
            "coefficients": self._coefficients,
            "powers": self._powers,
            "param_names": np.asarray(self.param_names, dtype=str),
            "fiducial_values": np.asarray([self.fiducial[name] for name in self.param_names], dtype=float),
            "step_sizes": np.asarray([self._step_sizes[name] for name in self.param_names], dtype=float),
            "config_json": json.dumps(payload, sort_keys=True),
            "output_state_json": json.dumps(self._output_state, sort_keys=True),
        }

    def save(self, path: str | Path) -> Path:
        """Serialize the built emulator to ``.npz`` format."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(destination, **self._serialized_state())
        self._cache_path = destination
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        theory_fn: Callable[[Mapping[str, float]], Any] | None = None,
        output_adapter: _OutputAdapter | None = None,
    ) -> TaylorEmulator:
        """Load a serialized emulator from disk.

        Parameters
        ----------
        path
            Path to a file written by `TaylorEmulator.save`.
        theory_fn
            Optional theory callable attached to the loaded emulator. This is
            not required for prediction, but it is useful if the loaded
            instance will be rebuilt.
        output_adapter
            Optional output adapter used when the serialized adapter tag is not
            one of the built-in adapters.
        """
        source = Path(path)
        with np.load(source, allow_pickle=False) as data:
            config = json.loads(str(data["config_json"]))
            adapter_tag = config["adapter_tag"]
            adapter = output_adapter or _BUILTIN_OUTPUT_ADAPTERS.get(adapter_tag)
            if adapter is None:
                raise ValueError(f"Unknown TaylorEmulator output adapter '{adapter_tag}'. Pass output_adapter explicitly when loading.")

            fiducial = dict(zip(data["param_names"].tolist(), data["fiducial_values"].tolist(), strict=True))
            fiducial.update({str(name): float(value) for name, value in config.get("constrained_fiducial", {}).items()})

            emulator = cls(
                theory_fn=theory_fn,
                fiducial=fiducial,
                order=int(config["order"]),
                step_sizes=dict(zip(data["param_names"].tolist(), data["step_sizes"].tolist(), strict=True)),
                param_names=data["param_names"].tolist(),
                output_adapter=adapter,
                cache_dir=source.parent,
                cache_key=source.stem.removeprefix("taylor_"),
                finite_difference_accuracy=int(config["finite_difference_accuracy"]),
                metadata=config.get("metadata", {}),
                valid_param_names=config.get("valid_param_names"),
            )
            emulator._valid_param_names = tuple(str(name) for name in config.get("valid_param_names", data["param_names"].tolist()))
            emulator._constrained_param_names = tuple(
                name for name in emulator._valid_param_names if name not in set(emulator.param_names)
            )
            emulator._powers = np.asarray(data["powers"], dtype=int)
            emulator._coefficients = np.asarray(data["coefficients"], dtype=float)
            emulator._output_state = json.loads(str(data["output_state_json"]))
            emulator._output_adapter = adapter
            emulator._cache_path = source
            emulator._eval_cache = {}
            return emulator
