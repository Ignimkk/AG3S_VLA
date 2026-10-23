"""Common interface of the paper baselines (PAPER-BASELINES): one target-mask contract for every provider.

A provider turns RGB (+ instruction) into a per-camera **target mask** in the same pixel frame as the RGB it was
given (and therefore the same frame as the depth that ``lift.py`` back-projects). Everything after the mask --
lifting, voxelisation, metrics -- is shared, so providers differ only in how they produce the mask.

Contract (checked by :func:`validate_results` and ``tests/test_interface.py``):

* ``mask`` is ``bool`` with shape ``(H, W)`` of the input RGB of that camera;
* ``found=False`` -> the mask is all ``False`` (nothing is invented when detection fails; same rule as ATS);
* several candidates -> the provider keeps the **highest-score** one (:func:`select_top1`); GT is never used for
  selection (only ``gt_oracle`` reads GT, by definition). The rule is part of ``provenance()``;
* ``provenance()`` lists everything needed to reproduce the run: repo + SHA, weights id + sha256, thresholds.

Dataset-independent: LIBERO-specific code lives only in ``eval/`` and ``providers/gt_oracle.py``.
"""
from __future__ import annotations

import dataclasses
import importlib
import time
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

import numpy as np

PROMPT_MODES = ("phrase", "instruction")
#: Rule every detector-based provider uses when several candidates pass the thresholds.
SELECTION_RULE = "top1_score: keep the single candidate with the highest detector score (ties -> lowest index); GT not used"


@dataclasses.dataclass(frozen=True)
class TargetQuery:
    instruction: str  # original task instruction
    target_phrase: str  # e.g. "black bowl" -- from BDDL / GT target name or a noun phrase
    prompt_mode: str = "phrase"  # "phrase" | "instruction": which text the detector receives
    episode_id: str | None = None
    frame_index: int | None = None

    def __post_init__(self):
        if self.prompt_mode not in PROMPT_MODES:
            raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}, got {self.prompt_mode!r}")

    @property
    def prompt(self) -> str:
        """The text given to a text-prompted detector."""
        return self.target_phrase if self.prompt_mode == "phrase" else self.instruction


@dataclasses.dataclass
class TargetMaskResult:
    camera: str
    mask: np.ndarray  # bool (H, W), same pixel frame as the input RGB / depth
    found: bool  # False -> empty mask
    score: float | None
    boxes: np.ndarray | None  # (N, 4) xyxy, all candidates
    phrases: list[str]
    latency_ms: float
    extras: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def not_found(cls, camera: str, shape: tuple[int, int], latency_ms: float, *, boxes: np.ndarray | None = None,
                  phrases: Iterable[str] = (), reason: str = "no_candidate", extras: dict | None = None
                  ) -> "TargetMaskResult":
        ex = dict(extras or {})
        ex.setdefault("not_found_reason", reason)
        return cls(camera=camera, mask=np.zeros(tuple(shape[:2]), bool), found=False, score=None,
                   boxes=None if boxes is None else np.asarray(boxes, np.float64).reshape(-1, 4),
                   phrases=list(phrases), latency_ms=float(latency_ms), extras=ex)

    def summary(self) -> dict:
        """JSON-able record (mask reduced to its pixel count)."""
        return {"camera": self.camera, "found": bool(self.found), "score": None if self.score is None else float(self.score),
                "mask_pixels": int(np.count_nonzero(self.mask)), "n_candidates": 0 if self.boxes is None else int(len(self.boxes)),
                "boxes": None if self.boxes is None else np.asarray(self.boxes, float).round(2).tolist(),
                "phrases": list(self.phrases), "latency_ms": float(self.latency_ms),
                "extras": _jsonable(self.extras)}


@runtime_checkable
class TargetMaskProvider(Protocol):
    name: str

    def provenance(self) -> dict: ...

    def reset(self) -> None: ...

    def predict(self, rgb: dict[str, np.ndarray], query: TargetQuery, *, aux: dict | None = None
                ) -> dict[str, TargetMaskResult]: ...


# --- helpers shared by providers -------------------------------------------------------------------------------
def select_top1(scores: np.ndarray) -> int | None:
    """Index of the highest score (ties -> lowest index); ``None`` for no candidate."""
    s = np.asarray(scores, np.float64).reshape(-1)
    if s.size == 0:
        return None
    return int(np.flatnonzero(s == s.max())[0])


def check_rgb(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8 (H, W, 3), got {a.dtype} {a.shape}")
    return np.ascontiguousarray(a)


def validate_results(results: dict[str, TargetMaskResult], rgb: dict[str, np.ndarray]) -> None:
    """Raise if a result breaks the contract (camera set, mask dtype / shape, not-found -> empty)."""
    if set(results) != set(rgb):
        raise ValueError(f"cameras differ: results {sorted(results)} vs rgb {sorted(rgb)}")
    for cam, r in results.items():
        if r.camera != cam:
            raise ValueError(f"result camera {r.camera!r} stored under {cam!r}")
        if r.mask.dtype != bool or r.mask.shape != np.asarray(rgb[cam]).shape[:2]:
            raise ValueError(f"{cam}: mask must be bool {np.asarray(rgb[cam]).shape[:2]}, got {r.mask.dtype} {r.mask.shape}")
        if not r.found and r.mask.any():
            raise ValueError(f"{cam}: found=False with a non-empty mask")


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.t0) * 1000.0
        return False


def sha256_file(path, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _jsonable(x: Any):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# --- registry -------------------------------------------------------------------------------------------------
#: Built-in providers, imported lazily so that the registry itself never imports torch.
BUILTIN_PROVIDERS: dict[str, str] = {
    "gt_oracle": "Baseline.paper_baselines.providers.gt_oracle:GTOracleProvider",
    "grounded_sam": "Baseline.paper_baselines.providers.grounded_sam:GroundedSAMProvider",
    "grounded_sam2": "Baseline.paper_baselines.providers.grounded_sam2:GroundedSAM2Provider",
    "yoloe": "Baseline.paper_baselines.providers.yoloe:YOLOEProvider",
    # Implementer B
    "localization_heads": "Baseline.paper_baselines.providers.localization_heads:LocalizationHeadsProvider",
    "tokencut": "Baseline.paper_baselines.providers.tokencut:TokenCutProvider",
}
_REGISTRY: dict[str, Callable[..., TargetMaskProvider]] = {}


def register_provider(name: str):
    """Decorator: ``@register_provider("my_provider")`` on a class / factory."""
    def deco(factory):
        _REGISTRY[name] = factory
        return factory
    return deco


def available_providers() -> list[str]:
    return sorted(set(_REGISTRY) | set(BUILTIN_PROVIDERS))


def get_provider(name: str, **kwargs) -> TargetMaskProvider:
    if name not in _REGISTRY:
        if name not in BUILTIN_PROVIDERS:
            raise KeyError(f"unknown provider {name!r}; available: {available_providers()}")
        module, attr = BUILTIN_PROVIDERS[name].split(":")
        _REGISTRY[name] = getattr(importlib.import_module(module), attr)
    return _REGISTRY[name](**kwargs)


__all__ = ["BUILTIN_PROVIDERS", "PROMPT_MODES", "SELECTION_RULE", "TargetMaskProvider", "TargetMaskResult", "TargetQuery",
           "Timer", "available_providers", "check_rgb", "get_provider", "register_provider", "select_top1", "sha256_file",
           "validate_results"]
