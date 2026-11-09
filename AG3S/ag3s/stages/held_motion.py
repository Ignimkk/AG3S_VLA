"""SL-PE8 (dev-4a, 2026-11-09 03:xx KST, decision log G162-) — the held object = the points that move with the hand.

Why: while held, the lifted bowl stayed in the scene obstacle-object set (`pipeline._scene_obstacle_points`), and the hand /
link7 / held-query rows next to it refused the pass-through 8-15 calls per episode (LIBERO t9 s7 · s8 · s9 failures: carrying
and placing). SL-PE3 drops components within 15 mm of the attach snapshot moved with the hand, but the snapshot is what the
head camera saw *before* the lift (inside and rim of the bowl); after the lift it sees the outer wall / bottom, 26-73 mm from
every snapshot point (offline, `outputs/2026-11-09_집기놓기_장애물취급/`), so PE3 dropped nothing.

Rule (no GT, deterministic). Every request gives the non-support cloud `P` (base frame), the hold-link pose `T` (FK at q) and
the fingertip midpoint `tip`. In the hand frame `H = T^-1 P`:
  * pre-held: the manipulated geometry (cut to `extent + reach_pad` around the fingertips) is the *reference* snapshot;
  * held: a point *moved with the hand* when it lies within `match` of the reference (snapshot, verified points, or the
    previous held request's hand-frame points), a point at rest there would have drifted >= `drift` in the hand frame since
    the previous request, it is NOT within `match` of the previous request's base-frame cloud (that would be explained by
    "at rest" -- a sliding plane), and it is within `reach` of the fingertips. Such points accumulate (hand frame) as the
    verified held body;
  * removed from the obstacle-object set: points within `match` of the verified body, within `reach`, and not explained by
    "at rest" while the hand moved. The raw snapshot itself is never removed (it can contain a neighbour; SafeLIBERO t3e2:
    the book).
Offline (pe12345 records, GT scoring only): 64-88 % of the observed held-bowl points removed, SafeLIBERO obstacle points
removed 0 (reach_pad 20 mm; 30 mm -> 1 book point, 40 mm -> 6).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.spatial import cKDTree


def _to_frame(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    Ti = np.linalg.inv(np.asarray(T, np.float64))
    return np.asarray(P, np.float64).reshape(-1, 3) @ Ti[:3, :3].T + Ti[:3, 3]


def _near(ref: np.ndarray, P: np.ndarray, r: float) -> np.ndarray:
    if not len(ref) or not len(P):
        return np.zeros(len(P), bool)
    return cKDTree(ref).query(P)[0] <= r


def _dedup(P: np.ndarray, v: float) -> np.ndarray:
    return np.unique(np.round(P / v), axis=0) * v if len(P) else P


class HeldMotionTracker:
    """Per-episode state of SL-PE8. `update(...)` once per request, in request order; `reset()` on episode reset."""

    def __init__(self, match_m: float = 0.010, drift_m: float = 0.025, reach_pad_m: float = 0.02,
                 reach_cap_m: float = 0.15, voxel_m: float = 0.004, max_points: int = 20000):
        self.match, self.drift, self.reach_pad, self.reach_cap = float(match_m), float(drift_m), float(reach_pad_m), \
            float(reach_cap_m)
        self.voxel, self.max_points = float(voxel_m), int(max_points)
        self.reset()

    def reset(self) -> None:
        self._snap = np.zeros((0, 3))       # hand frame, reference only
        self._verified = np.zeros((0, 3))   # hand frame, removed
        self._reach = self.reach_cap
        self._prev: Optional[tuple] = None  # (H, T, P, held)
        self.last: dict[str, Any] = {}

    def update(self, points: np.ndarray, T_base_hand: np.ndarray, tip: Optional[np.ndarray], held: bool,
               preheld_geometry: Optional[np.ndarray] = None) -> np.ndarray:
        """Bool mask over `points` (base frame, non-support cloud): True = the held object (drop from the obstacle set)."""
        P = np.asarray(points, np.float64).reshape(-1, 3)
        T = np.asarray(T_base_hand, np.float64).reshape(4, 4)
        H = _to_frame(T, P)
        tip = None if tip is None else np.asarray(tip, np.float64).reshape(3)
        out = np.zeros(len(P), bool)
        rec: dict[str, Any] = {"held": bool(held), "n_points": int(len(P))}
        if not held:
            if preheld_geometry is not None and len(preheld_geometry):
                g = np.asarray(preheld_geometry, np.float64).reshape(-1, 3)
                self._reach = min(float(np.ptp(g, axis=0).max()) + self.reach_pad, self.reach_cap)
                if tip is not None:
                    g = g[np.linalg.norm(g - tip, axis=1) <= self._reach]
                self._snap = _to_frame(T, g)
            self._verified = np.zeros((0, 3))
            self._prev = (H, T, P, False)
            rec.update(n_snapshot=int(len(self._snap)), reach_mm=round(self._reach * 1e3, 1))
            self.last = rec
            return out
        prev = self._prev
        inside = (np.linalg.norm(P - tip, axis=1) <= self._reach) if tip is not None else np.zeros(len(P), bool)
        moved_test = np.zeros(len(P), bool)
        if prev is not None and len(P):
            pH, pT, pP, p_held = prev
            drift = np.linalg.norm(H - _to_frame(pT, P), axis=1)
            at_rest = _near(pP, P, self.match)
            ref = np.concatenate([self._snap, self._verified] + ([pH] if p_held else []))
            new = _near(ref, H, self.match) & (drift >= self.drift) & ~at_rest & inside
            if new.any():
                self._verified = _dedup(np.concatenate([self._verified, H[new]]), self.voxel)[-self.max_points:]
            moved_test = at_rest & (drift >= self.drift)
            rec.update(n_new=int(new.sum()), median_drift_mm=round(float(np.median(drift)) * 1e3, 1))
        out = _near(self._verified, H, self.match) & ~moved_test & inside
        rec.update(n_verified=int(len(self._verified)), n_removed=int(out.sum()), reach_mm=round(self._reach * 1e3, 1))
        self._prev = (H, T, P, True)
        self.last = rec
        return out


def hand_pose_and_tip(model, q) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """(T_base_hold_link 4x4, fingertip midpoint (3,), hold link) from the constraint robot model at `q` -- the first gripper
    pair (`gripper_joint_pairs`); fingertip = per finger the constraint sphere farthest along the approach axis (the same
    definition as `grasp.tip_alignment`). (None, None, None) when the model cannot say."""
    pairs_fn = getattr(model, "gripper_joint_pairs", None)
    pose_fn = getattr(model, "link_pose", None)
    if model is None or q is None or not callable(pairs_fn) or not callable(pose_fn):
        return None, None, None
    try:
        pairs = list(pairs_fn())
        if not pairs:
            return None, None, None
        pair = pairs[0]
        hold = str(pair.get("parent_link"))
        kids = [str(c) for c in pair.get("children", ())]
        q = np.asarray(q, np.float64).reshape(-1)
        T = np.asarray(pose_fn(q, hold), np.float64)
        names = [str(n) for n in getattr(model, "sphere_link_names", ()) or ()]
        centres = np.asarray(model.sphere_centers_numeric(q)[0], np.float64)
        by = []
        for k in kids:
            idx = [i for i, n in enumerate(names) if n == k]
            by.append(centres[idx] if idx else np.asarray(pose_fn(q, k), np.float64)[:3, 3].reshape(1, 3))
        if not by:
            return T, None, hold
        origin = T[:3, 3]
        axis = np.concatenate(by).mean(axis=0) - origin
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        tip = np.mean([v[int(np.argmax((v - origin) @ axis))] for v in by], axis=0)
        return T, tip, hold
    except Exception:  # noqa: BLE001 -- no pose: SL-PE8 removes nothing this request
        return None, None, None
