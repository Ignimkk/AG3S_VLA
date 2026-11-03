"""T90 — the T86 hidden image-patch probe as the ATS target signal (`--attention-source hidden_probe[:NPZ]`).

**What it is.** T86 (`experiments/studies/t86_probe.py`, offline) trained, per camera, a logistic regression on the
VLM hidden state of one image token (one 2048-d vector of the prefix pass) to say "this patch is the target object
the prompt names". On held-out scenes it picked the right object more often than the noun-token attention
(`noun_target:L10H7`) that feeds ATS today (head scenes 0.849 → 0.942, left-wrist and right-wrist more). T90 §1 feeds
the probe's per-patch score into the place where the attention map goes today (`result["attention"]`, three 16 × 16
grids); everything after it (ATS seed top-k, graph, region, surface extraction) is unchanged.

**What is read per request.** The *same* prefix forward the noun source / `--hsr5` already run
(`attention_source.JaxPrefixBackend`, gemma ``return_hidden`` on for that pass only): the hidden stack
``[depth + 1, B, T, D]`` (HF numbering, h_0 = input embedding, h_l = output of block l before the final norm — the
numbering of `pi05_hidden` / T86's ``layers``), the image tokens of each camera (`CAMERA_BINDINGS` ranges: cam_high
0–255, left wrist 256–511, right wrist 512–767) at **that camera's layer**, standardised with the asset's mean / scale
and projected on ``coef[1] − coef[0]`` (+ intercept difference) = the logit of ``p(target)`` of the two-class model.
No extra forward. The map fed to ATS is ``p = sigmoid(logit)`` (positive, bounded, comparable between cameras: ATS takes
the top-5 patches over all 768, weighted by score — a negative logit could not be a weight).

**What the probe does not do.** It never reads the prompt text. The prompt reaches it only through the hidden state
(the model's own reading of "pick up the apple"). The noun lookup is kept for one purpose: **a prompt without exactly one
target fruit word gives "no target" exactly as the noun source does** (empty attention, reason recorded); the
asset is never asked then.

**Asset** (`AG3S/ag3s/asset/target_probe/t86_hidden_probe.npz`, written by ``t86_probe --export``): per camera
``{cam}__layer`` ``{cam}__mean`` ``{cam}__scale`` ``{cam}__coef`` (2, D) ``{cam}__intercept`` (2,) ``{cam}__C`` and the
training protocol keys; a sibling ``.json`` carries the source run, the md5 and the settings. Class order (0 = other,
1 = target) is checked on load.

Pure numpy / duck-typed ``xp`` (numpy or jax.numpy): no jax at import.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import pathlib
from typing import Any, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "ASSET_NAME", "CAMERAS", "GRID", "HiddenProbe", "N_TOKENS", "SPEC_NAME", "camera_probabilities", "default_asset_path",
    "describe_asset", "load_hidden_probe", "logits_of_stack", "probe_map", "save_asset", "score_logits", "sigmoid",
]

SPEC_NAME = "hidden_probe"
ASSET_NAME = "t86_hidden_probe"
GRID = 16
N_TOKENS = GRID * GRID
#: policy camera order = the order of the three 256-token blocks of the attention map (`policy_record`).
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
HIDDEN_DIM = 2048


def default_asset_path() -> pathlib.Path:
    """`AG3S/ag3s/asset/target_probe/t86_hidden_probe.npz`, located from this file (no symlinks)."""
    return pathlib.Path(__file__).resolve().parents[1] / "ag3s" / "asset" / "target_probe" / f"{ASSET_NAME}.npz"


def file_md5(path) -> str:
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ asset


@dataclasses.dataclass(frozen=True, eq=False)
class HiddenProbe:
    """The per-camera two-class logistic models. ``layers`` / ``mean`` / ``scale`` / ``coef`` / ``intercept`` are dicts
    keyed by camera name (`CAMERAS` order). ``w`` and ``b`` are the logit form of the 2-class softmax:
    ``logit = ((h − mean) / scale) @ w + b``, ``w = coef[1] − coef[0]``, ``b = intercept[1] − intercept[0]`` — and
    ``p(target) = sigmoid(logit) = softmax(...)[1]``."""

    layers: Mapping[str, int]
    mean: Mapping[str, np.ndarray]
    scale: Mapping[str, np.ndarray]
    coef: Mapping[str, np.ndarray]
    intercept: Mapping[str, np.ndarray]
    path: str
    md5: str
    meta: Mapping[str, Any]

    def w(self, cam: str) -> np.ndarray:
        c = np.asarray(self.coef[cam], np.float64)
        return c[1] - c[0]

    def b(self, cam: str) -> float:
        i = np.asarray(self.intercept[cam], np.float64)
        return float(i[1] - i[0])

    @property
    def layer_set(self) -> tuple[int, ...]:
        return tuple(sorted({int(v) for v in self.layers.values()}))

    def record(self) -> dict[str, Any]:
        return {"asset": self.path, "asset_md5": self.md5, "layers": {c: int(self.layers[c]) for c in CAMERAS},
                "variant": str(self.meta.get("variant", "raw")), "score": "p(target) = sigmoid(logit)"}


def save_asset(path, models: Mapping[str, Any], layers: Mapping[str, int], *, variant: str = "raw",
               meta: Optional[Mapping[str, Any]] = None) -> pathlib.Path:
    """Write the asset: ``models[cam]`` is a `t50_core.LogReg` (StandardScaler + 2-class logistic regression),
    ``layers[cam]`` the hidden layer it reads. Atomic. Returns the path."""
    out: dict[str, Any] = {"cameras": np.asarray(CAMERAS), "variant": np.asarray(variant),
                           "class_order": np.asarray(["other", "target"])}
    for cam in CAMERAS:
        m = models[cam]
        coef = np.asarray(m.coef, np.float64)
        if coef.shape != (2, HIDDEN_DIM):
            raise ValueError(f"{cam}: coef shape {coef.shape}, expected (2, {HIDDEN_DIM}) two-class")
        out[f"{cam}__layer"] = np.asarray(int(layers[cam]))
        out[f"{cam}__mean"] = np.asarray(m.mean, np.float64)
        out[f"{cam}__scale"] = np.asarray(m.scale, np.float64)
        out[f"{cam}__coef"] = coef
        out[f"{cam}__intercept"] = np.asarray(m.intercept, np.float64)
        out[f"{cam}__C"] = np.asarray(float(m.C))
    for k, v in (meta or {}).items():
        out[f"meta__{k}"] = np.asarray(v)
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(p)
    return p


def load_hidden_probe(path: Optional[str | pathlib.Path] = None) -> HiddenProbe:
    """Load the asset (`default_asset_path` for None / ``"default"``). Refuses a missing file, a missing camera, a wrong
    class order, a wrong dimension."""
    p = default_asset_path() if path in (None, "default", "") else pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"hidden probe asset {p} does not exist")
    layers, mean, scale, coef, icpt = {}, {}, {}, {}, {}
    with np.load(p, allow_pickle=False) as z:
        if "class_order" in z and [str(x) for x in z["class_order"]] != ["other", "target"]:
            raise ValueError(f"hidden probe {p}: class order {list(z['class_order'])} != ['other', 'target']")
        for cam in CAMERAS:
            if f"{cam}__coef" not in z:
                raise ValueError(f"hidden probe {p}: no model for camera {cam}")
            layers[cam] = int(z[f"{cam}__layer"])
            mean[cam] = np.asarray(z[f"{cam}__mean"], np.float64)
            scale[cam] = np.asarray(z[f"{cam}__scale"], np.float64)
            coef[cam] = np.asarray(z[f"{cam}__coef"], np.float64)
            icpt[cam] = np.asarray(z[f"{cam}__intercept"], np.float64)
            if coef[cam].shape != (2, HIDDEN_DIM) or mean[cam].shape != (HIDDEN_DIM,) or scale[cam].shape != (HIDDEN_DIM,):
                raise ValueError(f"hidden probe {p}: {cam} has shapes coef {coef[cam].shape} mean {mean[cam].shape} "
                                 f"scale {scale[cam].shape}, expected (2, {HIDDEN_DIM})")
            if not (0 <= layers[cam] <= 18):
                raise ValueError(f"hidden probe {p}: {cam} layer {layers[cam]} is outside h_0 … h_18")
        meta = {"variant": str(z["variant"]) if "variant" in z else "raw"}
        meta.update({k[len("meta__"):]: z[k].tolist() for k in z.files if k.startswith("meta__")})
    return HiddenProbe(layers=layers, mean=mean, scale=scale, coef=coef, intercept=icpt, path=str(p),
                       md5=file_md5(p), meta=meta)


@functools.lru_cache(maxsize=8)
def _describe(path_key: str, mtime_ns: int) -> str:
    return json.dumps(load_hidden_probe(path_key).record())


def describe_asset(path: Optional[str | pathlib.Path] = None) -> dict[str, Any]:
    """The record of an asset without keeping it (`HiddenProbeSource.record`, called every request: cached by path and
    modification time, so the file is read once)."""
    p = default_asset_path() if path in (None, "default", "") else pathlib.Path(path)
    return json.loads(_describe(str(p), p.stat().st_mtime_ns if p.is_file() else -1))


# ------------------------------------------------------------------------------------------------ scoring (numpy / jax)


def score_logits(xp, h, mean, scale, w, b):
    """Logit of p(target) of every token: ``h`` ``[N, D]`` (any float dtype) → ``[N]`` float32 / float64 as the inputs.
    The one definition serving (jax, float32, on the device) and the tests (numpy, float64) share."""
    return ((h - mean) / scale) @ w + b


def sigmoid(x) -> np.ndarray:
    """Host-side ``1 / (1 + exp(−x))`` in float64 (a float32 sigmoid saturates to exactly 1.0 above ~17 and would tie
    patches that ATS ranks by score)."""
    x = np.asarray(x, np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


def logits_of_stack(xp, hidden, probe_arrays: Mapping[str, Mapping[str, Any]], layers: Mapping[str, int],
                    ranges: Mapping[str, Sequence[int]]):
    """``hidden`` ``[depth + 1, B, T, D]`` → ``[3, 256]`` logits (batch 0), cameras in `CAMERAS` order.
    ``probe_arrays[cam]`` = ``{"mean", "scale", "w", "b"}`` (xp arrays), ``layers[cam]`` ints, ``ranges[cam]`` =
    ``(lo, hi)`` token range of the camera. Numpy and jax.numpy share this body."""
    rows = []
    for cam in CAMERAS:
        lo, hi = ranges[cam]
        a = probe_arrays[cam]
        h = hidden[int(layers[cam]), 0, int(lo):int(hi)].astype(xp.float32)
        rows.append(score_logits(xp, h, a["mean"], a["scale"], a["w"], a["b"]))
    return xp.stack(rows)


def camera_probabilities(logits: np.ndarray) -> np.ndarray:
    """``[3, 256]`` logits → ``[768]`` float32 probabilities in the attention map's column order (cam_high, left
    wrist, right wrist; row-major 16 × 16 inside a camera) — what `attention_source.cell_maps` splits again."""
    return sigmoid(np.asarray(logits, np.float64).reshape(len(CAMERAS), N_TOKENS)).astype(np.float32).reshape(-1)


def probe_map(logits: np.ndarray) -> dict[str, np.ndarray]:
    """``[3, 256]`` logits → ``{camera: [16, 16] float32 p(target)}``."""
    p = camera_probabilities(logits).reshape(len(CAMERAS), GRID, GRID)
    return {c: p[i] for i, c in enumerate(CAMERAS)}
