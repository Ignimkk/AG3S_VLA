"""T81 — is the manipulated object's old place empty? (`clustering.target_vacancy_release`)

`TargetConfirm` keeps a manipulated object it does not see: hidden by the hand (T32b), latched (T43 T 3(a)),
frozen while a grasp closes (T26 §4). All three read "unseen" as "covered". On regression 211 (T79) the
fingers pushed the apple ~10 cm before the grasp; the target stayed at the old place to the end of the run
while the depth images looked straight through it to the table. Unseen and *empty* are different
observations, and a depth image can tell them apart: project the old points into a camera and compare the
depth the camera measured along each ray with the point's own depth.

  * **empty**    measured depth > point depth + δ  — the ray passed through where the object was;
  * **occluded** measured depth < point depth − δ, or the pixel is the robot  — something is in front;
  * **present**  |measured − point| ≤ δ.

Only in-image points with a valid depth count. Per camera the counts are kept; the camera with the most
judgeable (non-robot) points decides the request (`judge`). Fewer than `min_points` of them = undecidable.
The fractions are taken over all valid points of that camera, **robot pixels counted as occluded** — a
hand in front of the old place lowers the empty share, never raises it.

This module is pure (numpy only): `TargetConfirm` calls it with the views the pipeline gives it
(`TargetConfirm.note_views`), and the offline measurement (`experiments/studies/t81_vacancy.py`) calls the
same function on recorded requests.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

import numpy as np

#: `judge()["verdict"]`
EMPTY, OCCLUDED_V, PRESENT_V, UNDECIDABLE, NO_VIEWS = "empty", "occluded", "present", "undecidable", "no_views"


@dataclasses.dataclass(frozen=True)
class VacancyView:
    """One camera of this request: depth in metres (H, W), K (3, 3), camera → base (4, 4), robot pixels."""

    name: str
    depth_m: np.ndarray
    K: np.ndarray
    T_base_cam: np.ndarray
    #: (H, W) True where the pixel is the robot (the self-filter's depth mask). None = unknown (none marked).
    robot_mask: Optional[np.ndarray] = None


def sample_points(points: np.ndarray, n_max: int) -> np.ndarray:
    """At most `n_max` of `points`, evenly spaced over the array (deterministic, order preserved)."""
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if n_max <= 0 or len(pts) <= n_max:
        return pts
    idx = np.unique(np.linspace(0, len(pts) - 1, int(n_max)).round().astype(np.int64))
    return pts[idx]


def judge_camera(points: np.ndarray, view: VacancyView, delta: float) -> dict[str, Any]:
    """Counts of one camera for `points` (base frame): `n_points` given, `n_valid` (in image ∧ depth > 0),
    `n_robot` (valid and a robot pixel), `n_judged` = `n_valid − n_robot`, and `n_empty` / `n_front` /
    `n_present` over the judged ones (`n_occluded` = `n_front + n_robot`)."""
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    T = np.asarray(view.T_base_cam, np.float64).reshape(4, 4)
    K = np.asarray(view.K, np.float64).reshape(3, 3)
    d = np.asarray(view.depth_m, np.float64)
    h, w = d.shape[:2]
    pc = (pts - T[:3, 3][None, :]) @ T[:3, :3]          # base → camera
    z = pc[:, 2]
    front = z > 1e-6
    u = np.full(len(pts), -1, np.int64)
    v = np.full(len(pts), -1, np.int64)
    zf = np.where(front, z, 1.0)
    u[front] = np.rint(K[0, 0] * pc[front, 0] / zf[front] + K[0, 2]).astype(np.int64)
    v[front] = np.rint(K[1, 1] * pc[front, 1] / zf[front] + K[1, 2]).astype(np.int64)
    inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    obs = np.zeros(len(pts), np.float64)
    obs[inside] = d[v[inside], u[inside]]
    valid = inside & np.isfinite(obs) & (obs > 0.0)
    robot = np.zeros(len(pts), bool)
    if view.robot_mask is not None:
        mk = np.asarray(view.robot_mask, bool)
        if mk.shape[:2] == (h, w):
            robot[valid] = mk[v[valid], u[valid]]
    judged = valid & ~robot
    diff = obs - z
    n_empty = int((judged & (diff > delta)).sum())
    n_front = int((judged & (diff < -delta)).sum())
    n_judged = int(judged.sum())
    return {"camera": str(view.name), "n_points": int(len(pts)), "n_inside": int(inside.sum()),
            "n_valid": int(valid.sum()), "n_robot": int((valid & robot).sum()), "n_judged": n_judged,
            "n_empty": n_empty, "n_front": n_front, "n_present": n_judged - n_empty - n_front}


def judge(points: np.ndarray, views: Sequence[VacancyView], *, delta: float, min_points: int) -> dict[str, Any]:
    """This request's vacancy evidence for the old `points` (see the module doc).

    Returns `{camera, n_judged, n_valid, n_robot, empty, occluded, present, decidable, per_camera}`;
    `empty` / `occluded` / `present` are fractions of the chosen camera's valid points (robot = occluded),
    None when undecidable. `camera` = the one with the most judged points (first on a tie)."""
    per = [judge_camera(points, v, delta) for v in (views or ())]
    out: dict[str, Any] = {"camera": None, "n_judged": 0, "n_valid": 0, "n_robot": 0, "empty": None,
                           "occluded": None, "present": None, "decidable": False,
                           "per_camera": [{k: c[k] for k in ("camera", "n_judged", "n_robot", "n_empty",
                                                               "n_front", "n_present")} for c in per]}
    if not per:
        out["why"] = NO_VIEWS
        return out
    best = max(range(len(per)), key=lambda i: (per[i]["n_judged"], -i))
    c = per[best]
    out.update({"camera": c["camera"], "n_judged": c["n_judged"], "n_valid": c["n_valid"],
                "n_robot": c["n_robot"]})
    if c["n_judged"] < int(min_points) or c["n_valid"] <= 0:
        out["why"] = UNDECIDABLE
        return out
    nv = float(c["n_valid"])
    out.update({"empty": round(c["n_empty"] / nv, 4), "occluded": round((c["n_front"] + c["n_robot"]) / nv, 4),
                "present": round(c["n_present"] / nv, 4), "decidable": True})
    return out


def vacant(evidence: dict[str, Any], *, p_empty: float, p_present: float) -> bool:
    """§1.4: decidable ∧ empty ≥ p_empty ∧ present ≤ p_present."""
    return bool(evidence.get("decidable")) and float(evidence["empty"]) >= float(p_empty) \
        and float(evidence["present"]) <= float(p_present)


__all__ = ["VacancyView", "judge", "judge_camera", "sample_points", "vacant"]
