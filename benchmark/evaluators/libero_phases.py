"""Manipulation phase labels of LIBERO policy calls (SL3 / SL4 definition, promoted from the SL3 verifier).

The definition is the one the SL3 verifier fixed (``outputs/verify/SL3/scripts/score_attention.py``
``phases_of``) and SL4 reused; it is restated here once so SL4b (and later steps) compute it in code:

* ``z0``  = target body (``<target>_main``) world z at the **first policy-call frame**;
* ``first_contact`` = first frame with a MuJoCo contact between a ``gripper0_*`` geom and a target geom;
* ``lift`` = first frame ``>= first_contact`` whose target z exceeds ``z0 + LIFT_M`` (10 mm);
* ``last_contact`` = last frame with gripper-target contact;
* per call frame ``f``: ``approach`` if ``f < first_contact``; ``grasp`` if ``first_contact <= f < lift``
  (or no lift); ``transport`` if ``lift <= f <= last_contact``; ``post_release`` after that.
  Episodes without any contact: every call is ``approach``.

States are the recorded **image-instant** simulator states (``sim_state_image``, SL2), all recorded frames.
Inputs (one file per episode, the same keys either way):

* ``phase_<stem>.npz`` -- ``benchmark/adapters/libero_panda/phase_export.py`` (py3.8; SL4b, init 1-4);
* ``gt_<stem>.npz`` -- the SL3 verifier export (``export_gt_dev.py``; init 0). Same contact rule, same
  ``segP/<cam>`` render (``phase_export`` reproduces the SL3 file bit for bit on t0 / t4, init 0).

``segP/<cam>`` (:func:`segmentation_at`) is the **image-instant** geom-id segmentation of each policy-call frame
(row = index in ``call_frames``) -- what the recorded RGB / depth / attention show. The rollout's own
``segmentation_geom/<cam>`` is rendered at the return-time state (100-300 px per frame differ, SL3).
"""
from __future__ import annotations

import hashlib
import pathlib
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

LIFT_M = 0.01
PHASES = ("approach", "grasp", "transport", "post_release")
FILE_PATTERNS = ("phase_{stem}.npz", "gt_{stem}.npz")
SEG_CAMERAS = ("head", "left_wrist")
DEFINITION = {
    "z0": "target body <target>_main world z at the first policy-call frame (image-instant state)",
    "first_contact": "first frame with a MuJoCo contact between a gripper0_* geom and a target geom",
    "lift": f"first frame >= first_contact with target z > z0 + {LIFT_M} m",
    "last_contact": "last frame with gripper-target contact",
    "labels": "approach: f < first_contact; grasp: first_contact <= f < lift (or no lift); transport: "
              "lift <= f <= last_contact; post_release: f > last_contact; no contact at all: approach",
    "source": "SL3 verifier score_attention.phases_of (SL3 / SL4 phase labels)",
}


def phase_labels(track: Mapping[str, np.ndarray], frames: Sequence[int]) -> tuple[dict, np.ndarray]:
    """``(info, labels)`` for the call frames ``frames`` of one episode.

    ``track``: ``target_xpos (N, 3)``, ``contact_grip_target (N,)``, ``call_frames`` and optionally
    ``frames (N,)`` (the recorded frame index of each row; default ``arange(N)``, the SL3 layout).
    """
    z = np.asarray(track["target_xpos"], np.float64)[:, 2]
    contact = np.asarray(track["contact_grip_target"]).astype(bool)
    rows = np.asarray(track["frames"], np.int64) if "frames" in track else np.arange(len(z), dtype=np.int64)
    if len(rows) != len(z) or len(contact) != len(z):
        raise ValueError("phase track arrays differ in length")
    row_of = {int(f): i for i, f in enumerate(rows)}
    call_frames = sorted(int(x) for x in np.asarray(track["call_frames"]).reshape(-1))
    z0 = float(z[row_of[call_frames[0]]])
    hit = rows[contact]
    info = {"z0": z0, "first_contact": None, "lift": None, "last_contact": None}
    labels = np.array(["approach"] * len(frames), dtype=object)
    if len(hit):
        fc, last = int(hit[0]), int(hit[-1])
        info["first_contact"], info["last_contact"] = fc, last
        up = rows[(rows >= fc) & (z > z0 + LIFT_M)]
        lift = int(up[0]) if len(up) else None
        info["lift"] = lift
        for j, f in enumerate(int(x) for x in frames):
            if f < fc:
                labels[j] = "approach"
            elif lift is None or f < lift:
                labels[j] = "grasp"
            elif f <= last:
                labels[j] = "transport"
            else:
                labels[j] = "post_release"
    info["z_max_minus_z0"] = float(z.max() - z0)
    return info, labels


def find_track(dirs: Iterable[str | pathlib.Path], stem: str) -> pathlib.Path:
    """The phase track file of ``stem`` in ``dirs`` (exactly one match over all dirs and patterns)."""
    hits = [pathlib.Path(d) / pat.format(stem=stem) for d in dirs for pat in FILE_PATTERNS
            if (pathlib.Path(d) / pat.format(stem=stem)).exists()]
    if not hits:
        raise FileNotFoundError(f"no phase track for {stem} in {list(map(str, dirs))}")
    if len(hits) > 1:
        raise ValueError(f"more than one phase track for {stem}: {hits}")
    return hits[0]


def load_track(path: str | pathlib.Path, *, geom_npz_sha256: Optional[str] = None,
               chunk_t: Optional[Sequence[int]] = None, segmentation: bool = False) -> dict:
    """Arrays of a phase track file + its SHA-256.

    ``geom_npz_sha256``: checked against the file's provenance when it records one (``phase_export.py``);
    ``chunk_t``: the episode's policy-call frames, checked against the file's (both file kinds);
    ``segmentation``: also load ``segP/<cam>`` (raises when the file has none)."""
    path = pathlib.Path(path)
    keys = ["target_xpos", "contact_grip_target", "call_frames", "frames", "chunk_t", "provenance_json"]
    with np.load(path) as z:
        if segmentation:
            missing = [f"segP/{c}" for c in SEG_CAMERAS if f"segP/{c}" not in z.files]
            if missing:
                raise ValueError(f"{path.name} has no {missing} (phase_export.py --no-segmentation?)")
            keys += [f"segP/{c}" for c in SEG_CAMERAS]
        out = {k: z[k] for k in z.files if k in keys}
    if geom_npz_sha256 is not None and "provenance_json" in out:
        import json

        prov = json.loads(str(out["provenance_json"]))
        if prov.get("geom_npz_sha256") not in (None, geom_npz_sha256):
            raise ValueError(f"{path.name} was exported from a different geometry npz")
    if chunk_t is not None and "chunk_t" in out:
        if not np.array_equal(np.asarray(out["chunk_t"]).astype(np.int64).reshape(-1),
                              np.asarray(chunk_t, np.int64).reshape(-1)):
            raise ValueError(f"{path.name}: chunk_t differs from the episode's")
    out["path"] = str(path)
    out["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def segmentation_at(track: Mapping[str, np.ndarray], frame: int, camera: str) -> np.ndarray:
    """Image-instant geom-id segmentation (OpenCV rows, -1 = not a geom) of policy-call ``frame``."""
    call_frames = [int(x) for x in np.asarray(track["call_frames"]).reshape(-1)]
    if int(frame) not in call_frames:
        raise KeyError(f"frame {frame} is not a policy-call frame of {track.get('path')}")
    return np.asarray(track[f"segP/{camera}"][call_frames.index(int(frame))])


__all__ = ["DEFINITION", "FILE_PATTERNS", "LIFT_M", "PHASES", "SEG_CAMERAS", "find_track", "load_track", "phase_labels",
           "segmentation_at"]
