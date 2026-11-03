"""T83 — the grasp corridor: the space between the two finger pads of the grasping hand, extended past the finger tips.

Task `AG3S/ag3s/docs/handoff/T83.task.md` §1 (definition) · §1b (use: carve the corridor's obstacle seeds out of the
fine target-free tier), both fixed before results.

All boxes are in the **hold-link frame** of `AG3S/ag3s/grasp/gripper.py` (RB-Y1 ``ee_left`` / ``ee_right``: ``+x``
closing axis, ``-z`` palm -> finger tips = approach direction):

    lo + m_in <= x <= hi - m_in          (lo / hi: the two pads' inner faces on that pose)
    |y| <= pad_half_y + m_y
    z_tip - ahead <= z <= z_palm

`lo` / `hi` come from the constraint robot model's finger spheres at that pose (the finger joints of that pose: the
measured opening at `q_now`, the TrajOpt finger envelope on the planned rows) — the same finger geometry the TrajOpt
rows read. Finger sides are told apart by position (the left fingers' MJCF / URDF names are swapped, T43 OC).

Pure numpy: `SafePolicy` builds the request (`GraspCorridorRequest`), AG3S (`runtime/pipeline.py`) decides whether it
applies (target points in the `q_now` corridor) and `fields/curobo_builder.py` carves (`corridor_voxels`).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

import numpy as np

#: task §1 / §1b values (`TrajOptConfig.collision.grasp_corridor_*`, `AG3S_RULES_REGISTRY.md`).
DEFAULT_AHEAD = 0.060
DEFAULT_MARGIN_IN = 0.003
DEFAULT_MARGIN_Y = 0.005
DEFAULT_SUPPORT_BAND = 0.003
DEFAULT_MIN_TARGET_POINTS = 5


def hand_corridor_spec(robot_model, hold_link: str, finger_prefix: str) -> tuple[Optional[dict], str]:
    """`(spec, reason)` — the two finger links of the grasping hand and their collision spheres (link frame), or
    `(None, why)`. The self-filter cover spheres are not the gripper's geometry (`UrdfSphereChain.gripper_openings`)."""
    try:
        from AG3S.ag3s.robot_models.urdf_sphere_chain import COVER_ROLE
    except ImportError:  # pragma: no cover - model module without roles
        COVER_ROLE = "self_filter_cover"
    names = [str(n) for n in getattr(robot_model, "sphere_link_names", ())]
    roles = [str(r) for r in (getattr(robot_model, "sphere_roles", ()) or ("collision",) * len(names))]
    if len(roles) != len(names):
        return None, "sphere roles do not match the spheres"
    if not callable(getattr(robot_model, "local_spheres", None)) or not callable(getattr(robot_model, "link_pose",
                                                                                         None)):
        return None, "robot model has no local_spheres / link_pose"
    fingers = sorted({n for n in names if n.startswith(str(finger_prefix)) and n != str(hold_link)})
    if len(fingers) != 2:
        return None, f"expected 2 finger links {finger_prefix}*, got {fingers}"
    out = {"hold_link": str(hold_link), "fingers": fingers, "local": [], "radii": []}
    for f in fingers:
        c, r = robot_model.local_spheres(f)
        keep = np.asarray([ro != COVER_ROLE for n, ro in zip(names, roles) if n == f], bool)
        c = np.asarray(c, np.float64).reshape(-1, 3)
        r = np.asarray(r, np.float64).reshape(-1)
        if keep.shape[0] != c.shape[0] or not keep.any():
            return None, f"finger link {f} has no collision sphere"
        out["local"].append(c[keep])
        out["radii"].append(r[keep])
    return out, ""


def corridor_pose(robot_model, q: np.ndarray, params: Optional[np.ndarray], spec: dict
                  ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """`(R, t, lo, hi)` at `q` (finger joints `params`, None = the model's current values): the hold-link pose
    (`p_base = R p_local + t`) and the two pads' inner faces along its `x` — `max(x + r)` of the finger on the
    `-x` side, `min(x - r)` of the one on `+x` (sides by the fingers' mean `x`)."""
    q = np.asarray(q, np.float64).reshape(-1)
    kw = {} if params is None else {"params": np.asarray(params, np.float64).reshape(-1)}
    T = np.asarray(robot_model.link_pose(q, spec["hold_link"], **kw), np.float64)
    R, t = T[:3, :3], T[:3, 3]
    xs = []
    for f, loc, rad in zip(spec["fingers"], spec["local"], spec["radii"]):
        Tf = np.asarray(robot_model.link_pose(q, f, **kw), np.float64)
        pts = loc @ Tf[:3, :3].T + Tf[:3, 3]
        xs.append(((pts - t) @ R)[:, 0])
    (xa, ra), (xb, rb) = zip(xs, spec["radii"])
    if xa.mean() <= xb.mean():
        lo, hi = float((xa + ra).max()), float((xb - rb).min())
    else:
        lo, hi = float((xb + rb).max()), float((xa - ra).min())
    return R, t, lo, hi


@dataclasses.dataclass(frozen=True)
class GraspCorridorRequest:
    """One request's corridors: pose 0 = `q_now`, poses 1.. = the planned rows of the VLA reference chunk.

    `R (n, 3, 3)`, `t (n, 3)`, `lo (n,)`, `hi (n,)` (`corridor_pose`), the rule numbers (m) and the hand geometry
    (`gripper.ParallelGripperSpec`: `pad_half_y`, `z_tip = pad_z[0]`, `z_palm = palm.lo[2]`). `info` is recorded."""

    R: np.ndarray
    t: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    pad_half_y: float
    z_tip: float
    z_palm: float
    ahead: float = DEFAULT_AHEAD
    margin_in: float = DEFAULT_MARGIN_IN
    margin_y: float = DEFAULT_MARGIN_Y
    support_band: float = DEFAULT_SUPPORT_BAND
    min_target_points: int = DEFAULT_MIN_TARGET_POINTS
    info: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        n = np.asarray(self.R).reshape(-1, 3, 3).shape[0]
        if n < 1 or np.asarray(self.t).reshape(-1, 3).shape[0] != n or np.asarray(self.lo).reshape(-1).shape[0] != n \
                or np.asarray(self.hi).reshape(-1).shape[0] != n:
            raise ValueError("GraspCorridorRequest: R, t, lo, hi must describe the same number (>= 1) of poses")

    @property
    def n_poses(self) -> int:
        return int(np.asarray(self.R).reshape(-1, 3, 3).shape[0])

    def box_local(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """`(lo, hi)` of pose `i`'s corridor in its hold-link frame (may be empty: lo > hi on x when the pads close)."""
        lo = np.array([float(self.lo[i]) + self.margin_in, -(self.pad_half_y + self.margin_y),
                       self.z_tip - self.ahead])
        hi = np.array([float(self.hi[i]) - self.margin_in, self.pad_half_y + self.margin_y, self.z_palm])
        return lo, hi

    def contains(self, points: np.ndarray, poses: Optional[Sequence[int]] = None) -> np.ndarray:
        """`(N,)` bool: points (base frame) inside the corridor of any of `poses` (default: all)."""
        p = np.asarray(points, np.float64).reshape(-1, 3)
        out = np.zeros(len(p), bool)
        R = np.asarray(self.R, np.float64).reshape(-1, 3, 3)
        t = np.asarray(self.t, np.float64).reshape(-1, 3)
        for i in (range(self.n_poses) if poses is None else poses):
            lo, hi = self.box_local(i)
            loc = (p - t[i]) @ R[i]
            out |= np.all((loc >= lo) & (loc <= hi), axis=1)
        return out

    def summary(self) -> dict:
        return {"n_poses": self.n_poses, "ahead_mm": round(self.ahead * 1e3, 3),
                "margin_in_mm": round(self.margin_in * 1e3, 3), "margin_y_mm": round(self.margin_y * 1e3, 3),
                "support_band_mm": round(self.support_band * 1e3, 3), "min_target_points": int(self.min_target_points),
                "opening_mm": [round(float(h - l) * 1e3, 2) for l, h in zip(np.reshape(self.lo, -1),
                                                                           np.reshape(self.hi, -1))],
                **dict(self.info)}


@dataclasses.dataclass(frozen=True)
class CorridorCarve:
    """What the builder carves: the request and the support planes `(K, 4)` `[n, offset]` (free side `n.p >= offset`)."""

    request: GraspCorridorRequest
    planes: np.ndarray


def support_planes(support_surfaces: Sequence[Any]) -> np.ndarray:
    """`(K, 4)` `[n_x, n_y, n_z, offset]` of AG3S support surfaces (`SupportSurface.normal` · `offset`)."""
    rows = []
    for s in support_surfaces or ():
        n = np.asarray(getattr(s, "normal", ()), np.float64).reshape(-1)
        if n.size == 3:
            rows.append([n[0], n[1], n[2], float(getattr(s, "offset", 0.0))])
    return np.asarray(rows, np.float64).reshape(-1, 4)


def near_or_below_support(points: np.ndarray, planes: np.ndarray, band: float) -> np.ndarray:
    """`(N,)` bool: within `band` of a support plane on its free side, or below it (`n.p - offset <= band`)."""
    p = np.asarray(points, np.float64).reshape(-1, 3)
    pl = np.asarray(planes, np.float64).reshape(-1, 4)
    if not pl.shape[0] or not len(p):
        return np.zeros(len(p), bool)
    s = p @ pl[:, :3].T - pl[None, :, 3]
    return (s <= float(band)).any(axis=1)


def corridor_voxels(request: GraspCorridorRequest, origin: np.ndarray, voxel_size: float,
                    shape: Sequence[int]) -> np.ndarray:
    """`(K, 3)` int64 — voxel indices of the grid (`centre = origin + ijk * voxel_size`, the `_ball_candidates`
    convention) whose centre lies in the union of the request's corridors, unique, sorted by linear index."""
    vs = float(voxel_size)
    origin = np.asarray(origin, np.float64).reshape(3)
    dims = np.asarray([int(v) for v in shape], np.int64)
    R = np.asarray(request.R, np.float64).reshape(-1, 3, 3)
    t = np.asarray(request.t, np.float64).reshape(-1, 3)
    found = []
    for i in range(request.n_poses):
        blo, bhi = request.box_local(i)
        if np.any(bhi < blo):
            continue
        corners = np.array([[x, y, z] for x in (blo[0], bhi[0]) for y in (blo[1], bhi[1]) for z in (blo[2], bhi[2])])
        w = corners @ R[i].T + t[i]
        lo = np.maximum(np.floor((w.min(axis=0) - origin) / vs).astype(np.int64), 0)
        hi = np.minimum(np.ceil((w.max(axis=0) - origin) / vs).astype(np.int64) + 1, dims)
        if np.any(hi <= lo):
            continue
        ax = [np.arange(int(lo[k]), int(hi[k]), dtype=np.int64) for k in range(3)]
        grid = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
        loc = (origin[None, :] + grid.astype(np.float64) * vs - t[i]) @ R[i]
        inside = np.all((loc >= blo) & (loc <= bhi), axis=1)
        if inside.any():
            found.append(grid[inside])
    if not found:
        return np.zeros((0, 3), np.int64)
    allv = np.vstack(found)
    lin = (allv[:, 0] * dims[1] + allv[:, 1]) * dims[2] + allv[:, 2]
    _, first = np.unique(lin, return_index=True)
    return allv[first]


__all__ = ["DEFAULT_AHEAD", "DEFAULT_MARGIN_IN", "DEFAULT_MARGIN_Y", "DEFAULT_SUPPORT_BAND",
           "DEFAULT_MIN_TARGET_POINTS", "hand_corridor_spec", "corridor_pose", "GraspCorridorRequest",
           "CorridorCarve", "support_planes", "near_or_below_support", "corridor_voxels"]
