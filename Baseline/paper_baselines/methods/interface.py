"""Common interface of the paper *methods* (PAPER-METHODS): safety filters and sampling guidance.

Two families share one observation type (:class:`SceneObservation`):

* :class:`SafetyFilter` -- runs **after** the policy and edits the action chunk just before execution
  (KNOWS, VLSA/AEGIS, multi-link CBF, ...). Returns a :class:`FilterResult` with actions of the **same shape and
  convention** as the input (LIBERO: 7D = EE delta xyz + axis-angle delta + gripper).
* :class:`SamplingGuidance` -- runs **inside** flow-matching denoising through the openpi hook
  ``pi0.sample_actions(..., guidance_fn=...)`` (``third_party/openpi/src/openpi/models/pi0.py:226-283``).
  ``guidance_fn(x_next, t_next)`` receives the post-Euler state in the **normalized** action space and must return a
  corrected state of the same shape / dtype, written in pure JAX (it runs under ``jax.jit``). ``norm`` carries the
  statistics to (un)normalize (see :func:`unnormalize_actions` / :func:`normalize_actions`). ``guidance_fn=None`` is
  bit-identical to the baseline; openpi itself is not modified.

Registry: :func:`get_method` (``name`` -> instance). Built-ins are resolved lazily (no torch / jax / cvxpy import at
registry import). Lookup order: explicit :func:`register_method` registrations, :data:`BUILTIN_METHODS`, then the
**convention** ``Baseline.paper_baselines.methods.<name>`` (or its ``.adapter`` submodule): importing it may register
the method with ``@register_method(name)``; otherwise a module-level ``make_method(**kwargs)`` factory is used.
"""
from __future__ import annotations

import dataclasses
import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

#: keys of ``SceneObservation.extras`` that carry privileged (simulator / GT) information start with this prefix,
#: e.g. ``"privileged/obstacle_names"``, ``"privileged/gt_seg"``. A method that reads such a key must say so in
#: its ``provenance()`` (``"privileged_inputs": [...]``).
PRIVILEGED_PREFIX = "privileged/"


@dataclass
class SceneObservation:  # one policy call's observation
    rgb: dict[str, np.ndarray]  # camera -> (H, W, 3) uint8, OpenCV pixel frame
    depth: dict[str, np.ndarray]  # camera -> (H, W) float metres, same pixel frame as rgb
    intrinsics: dict[str, np.ndarray]  # camera -> (3, 3)
    T_base_cam: dict[str, np.ndarray]  # camera -> (4, 4) OpenCV camera frame -> robot base frame
    q: np.ndarray  # joint position (Panda 7)
    ee_pose: np.ndarray  # (4, 4) EE pose in the base frame
    gripper: float
    instruction: str
    target_phrase: str | None = None
    extras: dict = field(default_factory=dict)  # GT seg, attention grid, obstacle names ... (privileged -> PRIVILEGED_PREFIX)

    def privileged(self, key: str, default: Any = None) -> Any:
        """``extras[PRIVILEGED_PREFIX + key]`` (helper so readers of privileged data are easy to grep)."""
        return self.extras.get(PRIVILEGED_PREFIX + key, default)


@dataclass
class FilterResult:
    actions: np.ndarray  # same shape / convention as the input (LIBERO: 7D EE delta + gripper)
    intervened: np.ndarray  # (T,) bool -- True where the action was changed
    info: dict  # QP status, barrier values, detected objects ...

    def summary(self) -> dict:
        return {"T": int(len(self.intervened)), "n_intervened": int(np.count_nonzero(self.intervened)),
                "intervened": np.asarray(self.intervened, bool).tolist(), "info": jsonable(self.info)}


@runtime_checkable
class SafetyFilter(Protocol):  # edits the action chunk just before execution (KNOWS, AEGIS, multi-link)
    name: str

    def provenance(self) -> dict: ...

    def reset(self) -> None: ...

    def filter(self, obs: SceneObservation, actions: np.ndarray) -> FilterResult: ...


@runtime_checkable
class SamplingGuidance(Protocol):  # acts inside flow-matching denoising (OmniGuide, CFM, Lan-o3dp adapted, NUDGE)
    name: str

    def provenance(self) -> dict: ...

    def make_guidance_fn(self, obs: SceneObservation, norm: dict) -> Callable:  # openpi pi0.sample_actions guidance_fn(x_next, t_next)
        ...


# --- helpers ---------------------------------------------------------------------------------------------------
def validate_filter_result(res: FilterResult, actions_in: np.ndarray) -> None:
    """Raise if a filter result breaks the contract (shape, finite, intervened length / dtype)."""
    a_in = np.asarray(actions_in)
    a = np.asarray(res.actions)
    if a.shape != a_in.shape:
        raise ValueError(f"actions shape changed: {a_in.shape} -> {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("filtered actions contain non-finite values")
    iv = np.asarray(res.intervened)
    T = a_in.shape[0] if a_in.ndim >= 2 else 1
    if iv.shape != (T,) or iv.dtype != bool:
        raise ValueError(f"intervened must be bool ({T},), got {iv.dtype} {iv.shape}")
    if not isinstance(res.info, dict):
        raise ValueError("info must be a dict")


def _norm_arrays(norm: dict, d: int, xp):
    if "q01" in norm and "q99" in norm:
        return "quantile", xp.asarray(norm["q01"])[..., :d], xp.asarray(norm["q99"])[..., :d]
    if "mean" in norm and "std" in norm:
        return "z", xp.asarray(norm["mean"])[..., :d], xp.asarray(norm["std"])[..., :d]
    raise KeyError("norm must contain ('q01', 'q99') or ('mean', 'std')")


def unnormalize_actions(x, norm: dict, xp=np):
    """Normalized -> raw actions, same formulas as ``openpi.transforms.Unnormalize`` (quantile if q01/q99 given).

    ``xp`` = ``numpy`` or ``jax.numpy`` (pure, jit-safe). Only the first ``len(stats)`` dims are mapped; extra
    padded dims (pi0 action_dim 32 > 7) are returned unchanged.
    """
    d = int(np.asarray(norm.get("q01", norm.get("mean"))).shape[-1])
    d = min(d, x.shape[-1])
    kind, a, b = _norm_arrays(norm, d, xp)
    head = x[..., :d]
    head = (head + 1.0) / 2.0 * (b - a + 1e-6) + a if kind == "quantile" else head * (b + 1e-6) + a
    return xp.concatenate([head, x[..., d:]], axis=-1) if x.shape[-1] > d else head


def normalize_actions(x, norm: dict, xp=np):
    """Raw -> normalized actions (inverse of :func:`unnormalize_actions`)."""
    d = int(np.asarray(norm.get("q01", norm.get("mean"))).shape[-1])
    d = min(d, x.shape[-1])
    kind, a, b = _norm_arrays(norm, d, xp)
    head = x[..., :d]
    head = (head - a) / (b - a + 1e-6) * 2.0 - 1.0 if kind == "quantile" else (head - a) / (b + 1e-6)
    return xp.concatenate([head, x[..., d:]], axis=-1) if x.shape[-1] > d else head


def jsonable(x: Any):
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return jsonable(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# --- registry --------------------------------------------------------------------------------------------------
#: Built-in methods, imported lazily (``"module:attr"``; attr is a class or factory taking ``**kwargs``).
BUILTIN_METHODS: dict[str, str] = {
    # Implementer C -- safety filters
    "knows": "Baseline.paper_baselines.methods.knows.adapter:KnowsSafetyFilter",
    "aegis": "Baseline.paper_baselines.methods.aegis.adapter:AegisSafetyFilter",
    "multilink_cbf": "Baseline.paper_baselines.methods.multilink_cbf.adapter:MultiLinkCBFFilter",
    # Implementer D -- sampling guidance: resolved by the package convention below (or register_method).
}
_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_method(name: str):
    """Decorator: ``@register_method("my_method")`` on a class / factory."""
    def deco(factory):
        _REGISTRY[name] = factory
        return factory
    return deco


def _resolve_by_convention(name: str):
    pkg = f"{__name__.rsplit('.', 1)[0]}.{name}"
    for mod_name in (pkg, f"{pkg}.adapter"):
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as e:
            if e.name is not None and (mod_name == e.name or mod_name.startswith(e.name + ".")):
                continue  # the method module itself does not exist -> try the next candidate
            raise  # a dependency inside the method is missing: surface it
        if name in _REGISTRY:
            return _REGISTRY[name]
        if hasattr(mod, "make_method"):
            return mod.make_method
    return None


def available_methods() -> list[str]:
    return sorted(set(_REGISTRY) | set(BUILTIN_METHODS))


def get_method(name: str, **kwargs):
    """Instantiate a method (``SafetyFilter`` or ``SamplingGuidance``) by name."""
    if name not in _REGISTRY:
        if name in BUILTIN_METHODS:
            module, attr = BUILTIN_METHODS[name].split(":")
            _REGISTRY[name] = getattr(importlib.import_module(module), attr)
        else:
            factory = _resolve_by_convention(name)
            if factory is None:
                raise KeyError(f"unknown method {name!r}; available: {available_methods()}")
            _REGISTRY[name] = factory
    return _REGISTRY[name](**kwargs)


__all__ = ["BUILTIN_METHODS", "PRIVILEGED_PREFIX", "FilterResult", "SafetyFilter", "SamplingGuidance", "SceneObservation",
           "available_methods", "get_method", "jsonable", "normalize_actions", "register_method", "unnormalize_actions",
           "validate_filter_result"]
