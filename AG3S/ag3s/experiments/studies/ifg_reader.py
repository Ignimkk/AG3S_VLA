"""Internal-feature grounding (IFG) — reader for the 2026-10-15 phase-sampled VLA dump.

Protocol: `doc/reviews/vla_dump_20261015/EXPERIMENT_PROTOCOL_KO.md` v1.1 (S0–S3). This module only
*reads*: the dump produced by `phase_dump/vla_phase_dump.py` (another session's collector), the source
rollout records it was computed from (T53 / T54c `runs/C1a/R/.../run_0000`) and the evaluator-only GT.

Dump layout (one directory per frame, complete when `done.json` exists):
  inputs.npz   image_<cam> (224,224,3) float32 in [-1,1], raw_state, token_ids/pieces/mask, prompt,
               query_rows / query_row_names (target_noun, destination_noun, verb), image_order
  embed.npz    image_tokens (768,2048) f16 (= projector out), siglip_encoded_<cam> (256,1152) f16
  prefix.npz   resid_out (18,968,2048) f16 (axis0 = layer, 0-based), attn_rows, final_hidden, ...
  suffix.npz   attn_probs (10,18,8,50,1018) f16 (denoise, layer, head, action query, key)
  derived.npz  serving_L8H2_last_d0 (768,), noun_target_L10H7 (768,), kv_L4, hsr_p

Algorithm inputs are `FrameInputs` (features, maps, images, state). GT is returned by a separate
function (`load_gt`) and must only be passed to evaluators.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import pathlib
import zipfile
from typing import Optional

import numpy as np

STORAGE = os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage")
DUMP_ROOT = pathlib.Path(
    f"{STORAGE}/results/development/20261015T150325+0900_ddc7a00060c4_d857a08dcb63/phase_dump")
SOURCE_ROOTS = {
    "T53": pathlib.Path(f"{STORAGE}/results/development/20261013T100210+0900_4a540f1a18d7_cb037c425d2f/verify/T53"),
    "T54c": pathlib.Path(f"{STORAGE}/results/development/20261014T133115+0900_cf38c9dbca7b_5697bbdb5190/verify/T54c"),
}
#: dump camera order (inputs.npz image_order) -> record policy key -> MuJoCo camera.
DUMP_CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
RECORD_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
N_IMAGE = 768

#: Fixed feature set (protocol §5), 0-based prefix layers.
FEATURES = ("siglip_encoded", "projector", "prefix_L4", "prefix_L8", "prefix_L12")
#: Fixed map sources (protocol §5): A, N, P and the T61 control A61.
MAP_SOURCES = ("A", "N", "P", "A61")


@dataclasses.dataclass(frozen=True)
class FrameRef:
    ds: str
    key: str
    frame: int
    moment: str
    kind: str
    path: pathlib.Path
    done: dict

    @property
    def frame_id(self) -> str:
        return f"{self.ds}_{self.key}_f{self.frame:03d}"

    @property
    def episode_id(self) -> str:
        return f"{self.ds}_{self.key}"


@dataclasses.dataclass
class FrameInputs:
    ref: FrameRef
    features: dict  # name -> (768, D) float32 (unnormalised)
    maps: dict  # source -> (768,) float64, raw
    images: np.ndarray  # (3, 224, 224, 3) float32 in [-1, 1]
    raw_state: np.ndarray
    prompt: str
    query_rows: dict  # role -> prefix row index


def list_frames(root: pathlib.Path = DUMP_ROOT) -> list[FrameRef]:
    """Completed frames only (`done.json` written last by the collector)."""
    out = []
    for d in sorted(root.iterdir()):
        dj = d / "done.json"
        if not d.is_dir() or not dj.exists():
            continue
        done = json.loads(dj.read_text())
        out.append(FrameRef(done["ds"], done["key"], int(done["frame"]), done["moment"], done["kind"], d, done))
    return out


def _npz_member(path: pathlib.Path, name: str) -> np.ndarray:
    with np.load(path) as z:
        return z[name]


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def load_frame(ref: FrameRef, *, features: tuple = FEATURES) -> FrameInputs:
    p = ref.path
    with np.load(p / "inputs.npz") as z:
        images = np.stack([z[f"image_{c}"] for c in DUMP_CAMERAS])
        order = [str(c) for c in z["image_order"]]
        if tuple(order) != DUMP_CAMERAS:
            raise ValueError(f"{ref.frame_id}: image_order {order}")
        raw_state = np.asarray(z["raw_state"], np.float64)
        prompt = str(z["prompt"])
        qrows = dict(zip([str(s) for s in z["query_row_names"]], [int(r) for r in z["query_rows"]]))
    feats = {}
    with np.load(p / "embed.npz") as z:
        if "siglip_encoded" in features:
            feats["siglip_encoded"] = np.concatenate(
                [z[f"siglip_encoded_{c}"] for c in DUMP_CAMERAS]).astype(np.float32)
        if "projector" in features:
            feats["projector"] = z["image_tokens"].astype(np.float32)
    layers = [int(f.split("_L")[1]) for f in features if f.startswith("prefix_L")]
    if layers:
        ro = _npz_member(p / "prefix.npz", "resid_out")  # (18, 968, 2048)
        for l in layers:
            feats[f"prefix_L{l}"] = ro[l, :N_IMAGE].astype(np.float32)
        del ro
    with np.load(p / "derived.npz") as z:
        a = z["serving_L8H2_last_d0"].astype(np.float64).reshape(-1)
        n = z["noun_target_L10H7"].astype(np.float64).reshape(-1)
    probs = _npz_member(p / "suffix.npz", "attn_probs")  # (10, 18, 8, 50, 1018)
    a61 = probs[0, 10, 7, :, :N_IMAGE].astype(np.float64).mean(0)
    a_check = probs[0, 8, 2, -1, :N_IMAGE].astype(np.float64)
    first8 = probs[0, 8, 2, :8, :N_IMAGE].astype(np.float64).mean(0)
    del probs
    maps = {"A": a, "N": n, "P": normalize_max(a) * normalize_max(n), "A61": a61, "A_first8": first8,
            "_A_from_probs": a_check}
    return FrameInputs(ref, feats, maps, images, raw_state, prompt, qrows)


def normalize_max(m: np.ndarray) -> np.ndarray:
    """`normalize(A)` of protocol §5 for P = normalize(A) * normalize(N): divide by the map max
    (scale only; the product's ranking is invariant to the choice of positive scale)."""
    m = np.asarray(m, np.float64)
    mx = float(m.max())
    return m / mx if mx > 0 else np.zeros_like(m)


# ------------------------------------------------------------------- record / GT (evaluator only)


def record_dir(ref: FrameRef) -> pathlib.Path:
    hits = sorted(glob.glob(str(SOURCE_ROOTS[ref.ds] / f"runs/C1a/R/C1a_{ref.key}_s*/ag3s/run_0000")))
    if len(hits) != 1:
        raise FileNotFoundError(f"{ref.frame_id}: {len(hits)} records")
    return pathlib.Path(hits[0])


def gt_paths(ref: FrameRef) -> tuple[pathlib.Path, pathlib.Path]:
    root = SOURCE_ROOTS[ref.ds] / "gt"
    return root / f"{ref.key}.npz", root / f"{ref.key}.grasp.npz"


def load_gt(ref: FrameRef, t_step: int) -> dict:
    """Evaluator-only: T47-format patch coverage + small label images + grasp labels at `t_step`."""
    gp, gg = gt_paths(ref)
    with np.load(gp, allow_pickle=True) as z:
        ix = np.flatnonzero(z["t_step"] == t_step)
        if ix.size != 1:
            raise KeyError(f"{ref.frame_id}: t_step {t_step} not in GT")
        i = int(ix[0])
        out = dict(classes=[str(c) for c in z["classes"]], class_cov=np.asarray(z["class_cov"][i], np.float64),
                   label_small=np.asarray(z["label_small"][i]),
                   fingerprint=str(z["source_record_fingerprint"]), index=i)
    with np.load(gg, allow_pickle=True) as z:
        out["grasp"] = {k: z[k][out["index"]].tolist() for k in ("grasp_state", "phys_state", "finger_contact_lr")
                        if k in z.files}
        out["target"] = str(z["target"])
        out["grasp_fingerprint"] = str(z["source_record_fingerprint"])
    return out


def npz_nonfinite(path: pathlib.Path, names: Optional[list] = None) -> dict:
    """Per member: number of non-finite values and of |x| == float16 max (saturated)."""
    out = {}
    with zipfile.ZipFile(path) as zz:
        members = [i.filename[:-4] for i in zz.infolist()]
    with np.load(path) as z:
        for k in members:
            if names is not None and k not in names:
                continue
            a = z[k]
            if a.dtype.kind != "f":
                continue
            nf = int((~np.isfinite(a)).sum())
            sat = int((np.abs(a) == np.finfo(np.float16).max).sum()) if a.dtype == np.float16 else 0
            if nf or sat:
                out[k] = {"nonfinite": nf, "saturated": sat, "size": int(a.size), "dtype": str(a.dtype)}
    return out
