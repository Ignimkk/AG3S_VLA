"""AG3S-GEO-GRASP — at most eight 6-D grasp candidates from the observed target points.

Pipeline (one call, no learning, no global planner):

1. **Surface** (`surface.estimate_surface`): downsampled target points, outward PCA normals, a sphere fit of the target.
2. **Contact pairs:** two observed points ``p1, p2`` the two pads can press from opposite sides — the closing axis
   ``u = (p2 - p1) / |p2 - p1|``; each outward normal lies in the friction cone of the pad pushing on it
   (``angle(n1, -u)``, ``angle(n2, u)`` <= ``atan(mu)``); the width fits the opening with clearance; both contacts
   have `min_support` neighbours with agreeing normals within `support_radius` (a pad-sized observed patch). Pairs
   are thinned greedily (best alignment first) so that no two kept pairs share a centre and an axis.
   **Only observed points are contacts** — the hidden back of the object is never assumed to be a contact surface.
3. **Poses:** for each pair, `n_roll` approach directions around the closing axis and `depths` pad depths. Hold-link
   frame (`gripper.py`): ``x = u``, ``z = -a`` (``a`` = approach direction), origin ``o = c - depth * a`` (the pad
   centre at the contact midpoint ``c``). Pre-grasp: ``o - pregrasp_offset * a``, same rotation.
4. **Checks** (all recorded per candidate): non-target points (obstacles, support surface) inside the palm / finger
   boxes at the open width grown by `scene_margin`; the target **completed by its sphere fit** inside the palm or
   fingers (the unseen side is not taken as free); the same along the approach segment (pre-grasp -> grasp);
   IK of grasp and pre-grasp (`ik_fn`, optional). A candidate without IK is not returned when `ik_fn` is given.
5. **Order:** ``selection = "vla"`` — closest to the policy's planned close pose (position error + `vla_rot_weight` *
   rotation error, the 180 deg finger swap counted as the same grasp); ``"geometric"`` — best contact alignment first,
   ties within `geometric_tie` broken by the VLA distance. At most `max_candidates` are returned; none is a valid answer.

Simulator ground truth is never read here; evaluation code compares afterwards.
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from AG3S.ag3s.grasp.arm_shape import field_hits_many, shapes_hits_many
from AG3S.ag3s.grasp.gripper import ParallelGripperSpec
from AG3S.ag3s.grasp.surface import SurfaceEstimate, estimate_surface, voxel_downsample


@dataclasses.dataclass(frozen=True)
class GeoGraspConfig:
    """Thresholds of the candidate generator and the task state. Fixed in the run manifest before closed-loop."""

    voxel: float = 0.004
    #: obstacle / support points are thinned to this voxel (m) before the collision checks.
    scene_voxel: float = 0.005
    normal_radius: float = 0.012
    min_neighbours: int = 6
    #: planarity ratio above which a point's normal is not trusted (0 = plane, 1/3 = isotropic).
    max_curvature: float = 0.06
    #: assumed Coulomb friction coefficient; cone half-angle = atan(mu).
    friction_mu: float = 0.8   # user ruling 2026-10-26 ~20:55 KST (0.5 -> 0.8; sim objects 1.2)
    normal_margin_deg: float = 5.0
    min_width: float = 0.010
    #: total pad-to-surface gap (both sides) when the gripper arrives open: open width = width + clearance.
    opening_clearance: float = 0.010
    support_radius: float = 0.008
    min_support: int = 4
    support_normal_deg: float = 30.0
    #: pair thinning: a pair is a duplicate of a kept one when centres are closer than this and axes within this angle.
    pair_centre_sep: float = 0.006
    pair_axis_sep_deg: float = 15.0
    max_pairs: int = 16
    n_roll: int = 8
    #: approach samples added around the policy's own approach direction (0 = none): `vla_roll_samples` directions
    #: `vla_roll_step` apart, centred on it. The RB-Y1 wrist reaches only ~+-15-30 deg around the policy's approach
    #: at the table (offline 121 request 9), so the uniform 45 deg grid alone mostly misses reachable approaches.
    vla_roll_samples: int = 3
    vla_roll_step: float = math.pi / 8
    depths: tuple[float, ...] = (0.098, 0.115)
    pregrasp_offset: float = 0.06
    approach_samples: int = 4
    scene_margin: float = 0.005
    target_margin: float = -0.001
    target_shell_points: int = 400
    max_ik: int = 8
    #: stop the IK pass once this many candidates passed (the task state uses the best and compares with it; IK is
    #: ~70 % of a full generation: S1b 121 median 250 of 362 ms).
    ik_accept: int = 3
    #: close gate (`close_pose_check`): observed target points needed under each pad face.
    close_min_support: int = 3
    #: wall-clock budget (ms) of one full generation: pose checks and IK stop once it is spent (the candidates found
    #: so far are returned, `stats["budget_hit"]`). S2c replay: full generation reached 834 ms (IK) > the 533 ms
    #: execution window. 0 = no budget. User ruling 2026-10-26 ~23:00 KST.
    time_budget_ms: float = 150.0
    max_candidates: int = 8
    selection: str = "vla"
    #: m per rad when position and rotation errors are summed (1 rad ~ 50 mm).
    vla_rot_weight: float = 0.05
    geometric_tie: float = 0.03
    # --- task state (task_state.py) ---
    #: a tracked grasp is re-matched to a fresh candidate within these.
    match_pos: float = 0.012
    match_rot_deg: float = 20.0
    #: switch to a different valid candidate only when its key is better by this (m-equivalent).
    switch_margin: float = 0.015
    #: target centre jump (m) between observations above which the tracked grasp is dropped (target moved / swapped).
    max_centre_jump: float = 0.04
    #: full candidate generation at most every `refresh_every` requests while a tracked grasp stays valid (the
    #: tracked grasp is re-checked on every request in between).
    refresh_every: int = 4
    #: re-selections allowed after a grasp is lost while closing (plan §3: once).
    closing_reselect_budget: int = 1
    # --- obstacle-aware checks (task §3g-1, all off by default) ---
    #: check the wrist parts rigid with the hand (`gripper.wrist`) wherever the hand boxes are checked.
    wrist_check: bool = False
    #: check the arm links 0..5 at each candidate's IK solutions (`arm_fn` of `generate_candidates`).
    arm_check: bool = False
    #: obstacles = the scene points accumulated over the episode's pre-grasp requests (`ObstacleMemory`), not only
    #: this request's.
    obstacle_memory: bool = False
    obstacle_voxel: float = 0.005
    obstacle_max_points: int = 60000
    #: where the hand / wrist / arm checks read the obstacles (task §3j): "points" (the scene points, default) or
    #: "field" (the trajectory optimizer's distance field without the target, `field_fn` of `generate_candidates`;
    #: shape boundary points closer than `scene_margin` are hits). The target itself is always the completed points.
    obstacle_source: str = "points"

    @property
    def cone_cos(self) -> float:
        """cos of the friction cone half-angle atan(mu) — the physical limit (ground-truth evaluation uses this)."""
        return math.cos(math.atan(float(self.friction_mu)))

    @property
    def generation_cone_cos(self) -> float:
        """The cone the generator accepts: `normal_margin_deg` narrower than the friction cone, for the error of
        normals estimated from observed points (offline 121: an accepted pair at 24-25 deg observed was 28.4 deg on
        the true surface, outside the 26.6 deg cone)."""
        return math.cos(max(math.atan(float(self.friction_mu)) - math.radians(self.normal_margin_deg), 0.0))

    def record(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(self).items()}


@dataclasses.dataclass
class GraspCandidate:
    """One 6-D grasp. `T` / `T_pre`: hold-link pose (4x4, robot base frame) at the grasp and the pre-grasp."""

    index: int
    contacts: np.ndarray            # (2, 3): finger-1 (-x) and finger-2 (+x) contact
    normals: np.ndarray             # (2, 3) outward normals at the contacts
    width: float                    # contact distance (m)
    open_width: float               # pad distance on arrival (m)
    depth: float
    roll: float                     # approach angle around the closing axis (rad, pair-local)
    T: np.ndarray
    T_pre: np.ndarray
    alignment: float                # min cos of the two normal-to-axis angles (1 = perfect antipodal)
    support: tuple[int, int]
    pair: int = -1                  # index of the contact pair (one candidate per pair and approach direction)
    checks: dict[str, Any] = dataclasses.field(default_factory=dict)
    q_grasp: Optional[np.ndarray] = None
    q_pre: Optional[np.ndarray] = None
    key: float = float("inf")
    vla_pos_err: Optional[float] = None
    vla_rot_err: Optional[float] = None

    @property
    def approach(self) -> np.ndarray:
        """Unit approach direction (world): palm -> fingertips."""
        return -self.T[:3, 2]

    @property
    def valid(self) -> bool:
        return bool(self.checks.get("valid", False))

    def record(self) -> dict[str, Any]:
        r = lambda a, k=5: [round(float(v), k) for v in np.asarray(a).reshape(-1)]  # noqa: E731
        rec = {"index": self.index, "width_mm": round(self.width * 1000, 2), "open_width_mm": round(self.open_width * 1000, 2),
               "depth_mm": round(self.depth * 1000, 1), "roll_deg": round(math.degrees(self.roll), 1),
               "alignment": round(self.alignment, 4), "support": list(self.support),
               "position": r(self.T[:3, 3]), "rotation": r(self.T[:3, :3]), "pre_position": r(self.T_pre[:3, 3]),
               "contacts": r(self.contacts), "key": None if not np.isfinite(self.key) else round(float(self.key), 5),
               "checks": self.checks}
        if self.vla_pos_err is not None:
            rec["vla_pos_err_mm"] = round(self.vla_pos_err * 1000, 2)
            rec["vla_rot_err_deg"] = round(math.degrees(self.vla_rot_err), 2)
        return rec


@dataclasses.dataclass
class CandidateSet:
    """Result of one `generate_candidates` call."""

    candidates: list[GraspCandidate]
    surface: Optional[SurfaceEstimate]
    stats: dict[str, Any]
    #: the scene points near the target and the completed target the checks used (for re-checking a tracked grasp).
    scene: Optional[np.ndarray] = None
    shell: Optional[np.ndarray] = None
    #: every scene point (not only those near the target), for the arm check (`arm_fn` given only).
    scene_arm: Optional[np.ndarray] = None
    #: the obstacle distance function the checks used (`cfg.obstacle_source == "field"`), for the re-check.
    field_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    @property
    def best(self) -> Optional[GraspCandidate]:
        return self.candidates[0] if self.candidates else None


# --- rotation helpers -------------------------------------------------------------------------------------------
def rot_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle (rad) between two rotations."""
    c = (float(np.trace(np.asarray(R1).T @ np.asarray(R2))) - 1.0) * 0.5
    return math.acos(max(-1.0, min(1.0, c)))


_RZ_PI = np.diag([-1.0, -1.0, 1.0])


def grasp_rot_angle(R_hand: np.ndarray, R_goal: np.ndarray) -> float:
    """Rotation error of a parallel grasp: the 180 deg turn about the approach axis (fingers swapped) is the same grasp."""
    return min(rot_angle(R_hand, R_goal), rot_angle(R_hand, np.asarray(R_goal) @ _RZ_PI))


def nearest_symmetric(R_goal: np.ndarray, R_hand: np.ndarray) -> np.ndarray:
    """`R_goal` or `R_goal Rz(pi)`, whichever is closer to `R_hand`."""
    R_goal = np.asarray(R_goal, np.float64)
    alt = R_goal @ _RZ_PI
    return R_goal if rot_angle(R_hand, R_goal) <= rot_angle(R_hand, alt) else alt


def _perp_basis(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to `u` and each other — `e1` is the world-z component removed from `u`'s
    complement when possible (so roll 0 is 'from above' for a horizontal axis), deterministic."""
    up = np.array([0.0, 0.0, -1.0])           # roll 0: approach downward (top grasp) when the axis is not vertical
    e1 = up - np.dot(up, u) * u
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([1.0, 0.0, 0.0]) - u[0] * u
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return e1, e2


def grasp_frame(c: np.ndarray, u: np.ndarray, a: np.ndarray, depth: float) -> np.ndarray:
    """Hold-link pose: x = closing axis `u`, z = -approach `a`, origin so the pad centre sits at `c`."""
    x = np.asarray(u, np.float64) / np.linalg.norm(u)
    z = -np.asarray(a, np.float64)
    z = z - np.dot(z, x) * x
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = np.asarray(c, np.float64) - float(depth) * (-z)
    return T


def _in_hand(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (np.asarray(points) - T[:3, 3]) @ T[:3, :3]


def _sphere_shell(centre: np.ndarray, radius: float, n: int) -> np.ndarray:
    """`n` points on a sphere (Fibonacci lattice)."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    th = math.pi * (1.0 + math.sqrt(5.0)) * k
    d = np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])
    return np.asarray(centre, np.float64) + float(radius) * d


# --- pairs ------------------------------------------------------------------------------------------------------
def antipodal_pairs(surf: SurfaceEstimate, gripper: ParallelGripperSpec, cfg: GeoGraspConfig,
                    prefer_axis: Optional[np.ndarray] = None) -> tuple[list[tuple[int, int, float]], dict[str, int]]:
    """Kept contact pairs ``(i, j, alignment)`` (i = finger-1 contact) and counts per stage. Pairs are thinned in
    order of alignment, or — with `prefer_axis` (the policy's closing axis, rule ``"vla"``) — of the angle between
    their closing axis and it (sign-agnostic), so the kept pairs are those the policy's hand can close on."""
    from scipy.spatial import cKDTree

    P, N = surf.points, surf.normals
    ok = (np.linalg.norm(N, axis=1) > 0.5) & (surf.curvature <= cfg.max_curvature)
    counts = {"n_points": int(len(P)), "n_usable_normals": int(ok.sum())}
    idx = np.flatnonzero(ok)
    if idx.size < 2:
        counts.update(n_pairs_geometric=0, n_pairs_supported=0, n_pairs_kept=0)
        return [], counts
    # support: neighbours within the pad patch whose normals agree (the point itself included)
    cos_s = math.cos(math.radians(cfg.support_normal_deg))
    pr = cKDTree(P).query_pairs(r=cfg.support_radius, output_type="ndarray")
    support = ok.astype(int)
    if len(pr):
        a, b = pr[:, 0], pr[:, 1]
        agree = ok[a] & ok[b] & (np.einsum("ij,ij->i", N[a], N[b]) >= cos_s)
        support += np.bincount(a[agree], minlength=len(P)) + np.bincount(b[agree], minlength=len(P))
    Pi, Ni = P[idx], N[idx]
    D = Pi[None, :, :] - Pi[:, None, :]                      # D[a, b] = P_b - P_a
    W = np.linalg.norm(D, axis=2)
    max_w = gripper.max_opening - cfg.opening_clearance
    with np.errstate(invalid="ignore", divide="ignore"):
        U = D / W[:, :, None]
    c1 = -np.einsum("ak,abk->ab", Ni, U)                    # cos(n1, -u)
    c2 = np.einsum("bk,abk->ab", Ni, U)                     # cos(n2, u)
    geo = (W >= cfg.min_width) & (W <= max_w) & (c1 >= cfg.generation_cone_cos) & (c2 >= cfg.generation_cone_cos)
    np.fill_diagonal(geo, False)
    counts["n_pairs_geometric"] = int(geo.sum())
    sup = support[idx]
    geo &= (sup[:, None] >= cfg.min_support) & (sup[None, :] >= cfg.min_support)
    counts["n_pairs_supported"] = int(geo.sum())
    a_idx, b_idx = np.nonzero(geo)
    align = np.minimum(c1[a_idx, b_idx], c2[a_idx, b_idx])
    if prefer_axis is None:
        order = np.argsort(-align, kind="stable")
    else:
        ax = np.asarray(prefer_axis, np.float64) / np.linalg.norm(prefer_axis)
        Uab = U[a_idx, b_idx]
        order = np.lexsort((-align, -np.round(np.abs(Uab @ ax), 3)))
    kept: list[tuple[int, int, float]] = []
    kept_c, kept_u = [], []
    cos_ax = math.cos(math.radians(cfg.pair_axis_sep_deg))
    for o in order:
        a, b = int(idx[a_idx[o]]), int(idx[b_idx[o]])
        c = 0.5 * (P[a] + P[b])
        u = (P[b] - P[a]) / np.linalg.norm(P[b] - P[a])
        dup = any(np.linalg.norm(c - kc) < cfg.pair_centre_sep and abs(float(u @ ku)) >= cos_ax
                  for kc, ku in zip(kept_c, kept_u))
        if dup:
            continue
        kept.append((a, b, float(align[o])))
        kept_c.append(c)
        kept_u.append(u)
        if len(kept) >= cfg.max_pairs:
            break
    counts["n_pairs_kept"] = len(kept)
    counts["support"] = support   # type: ignore[assignment]  (popped by the caller)
    return kept, counts


# --- main -------------------------------------------------------------------------------------------------------
def generate_candidates(target_points: np.ndarray, scene_points: Optional[np.ndarray], gripper: ParallelGripperSpec,
                        cfg: GeoGraspConfig = GeoGraspConfig(), *,
                        vla_pose: Optional[np.ndarray] = None,
                        ik_fn: Optional[Callable[[np.ndarray, Optional[np.ndarray]], tuple[Optional[np.ndarray], dict]]] = None,
                        ik_seed: Optional[np.ndarray] = None,
                        selection: Optional[str] = None, recheck_only: bool = False,
                        arm_fn: Optional[Callable[[np.ndarray, np.ndarray], dict]] = None,
                        field_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> CandidateSet:
    """Candidates for grasping the object observed as `target_points` (base frame, m) among `scene_points`
    (every other observed point: obstacles, support surface — base frame). `vla_pose`: the policy's planned hold-link
    pose at its close row (4x4) — the preference of rule ``"vla"`` and the tie-break of ``"geometric"``.
    `ik_fn(T, seed) -> (q or None, info)`: inverse kinematics of the hold link (None = not checked).
    `recheck_only`: only the surface, the nearby scene and the completed target (what a tracked grasp is re-checked
    against) — no pairs, no candidates.
    `arm_fn(q, points) -> {"hits": n, ...}` (with `cfg.arm_check`): obstacle points inside the arm links at the joint
    configuration `q` of the hand's arm; a candidate whose grasp or pre-grasp IK solution has hits is not returned.
    With `cfg.wrist_check` the wrist parts of `gripper` are checked with the hand boxes.
    `field_fn(points) -> (N,)` (with `cfg.obstacle_source == "field"`): the obstacles are this distance function, not
    `scene_points` (task §3j).
    """
    t0 = time.perf_counter()
    sel = str(selection or cfg.selection)
    stats: dict[str, Any] = {"selection": sel}
    tp = voxel_downsample(np.asarray(target_points, np.float64).reshape(-1, 3), cfg.voxel)
    stats["n_target_points_raw"] = int(np.asarray(target_points).reshape(-1, 3).shape[0])
    if len(tp) < 2 * cfg.min_neighbours:
        stats.update(reason="too few target points", n_target_points=int(len(tp)), compute_ms=_ms(t0))
        return CandidateSet([], None, stats)
    surf = estimate_surface(tp, radius=cfg.normal_radius, min_neighbours=cfg.min_neighbours,
                            max_object_radius=0.5 * gripper.max_opening + 0.01)
    stats.update(n_target_points=int(len(tp)), centre=[round(float(v), 5) for v in surf.centre],
                 centre_method=surf.centre_method, sphere_radius_mm=None if surf.radius is None else round(surf.radius * 1000, 2))
    arm_fn = arm_fn if cfg.arm_check else None
    wrist = tuple(gripper.wrist) if cfg.wrist_check else ()
    field = field_fn if cfg.obstacle_source == "field" else None
    stats["obstacle_source"] = "field" if field is not None else "points"
    scene_arm = None
    if arm_fn is not None and field is None:
        scene_arm = (np.zeros((0, 3)) if scene_points is None
                     else voxel_downsample(np.asarray(scene_points, np.float64).reshape(-1, 3), cfg.scene_voxel))
        stats["n_scene_points_arm"] = int(len(scene_arm))
    if recheck_only:
        scene = np.zeros((0, 3)) if scene_points is None else np.asarray(scene_points, np.float64).reshape(-1, 3)
        reach = 0.5 * gripper.max_opening + gripper.pad_centre_depth + cfg.pregrasp_offset + 0.08
        if len(scene):
            scene = voxel_downsample(scene[np.linalg.norm(scene - surf.centre, axis=1) <= reach], cfg.scene_voxel)
        stats.update(recheck_only=True, n_scene_points_near=int(len(scene)), compute_ms=_ms(t0))
        return CandidateSet([], surf, stats, scene, target_shell(surf, cfg), scene_arm, field)
    pairs, counts = antipodal_pairs(surf, gripper, cfg,
                                    prefer_axis=None if (vla_pose is None or sel != "vla") else np.asarray(vla_pose)[:3, 0])
    support = counts.pop("support", np.zeros(len(surf.points), int))
    stats.update(counts)
    t_pairs = _ms(t0)
    scene = np.zeros((0, 3)) if scene_points is None else np.asarray(scene_points, np.float64).reshape(-1, 3)
    # only scene points that can matter (within reach of any gripper pose around the target)
    reach = 0.5 * gripper.max_opening + gripper.pad_centre_depth + cfg.pregrasp_offset + 0.08 + _wrist_reach(wrist)
    if len(scene):
        scene = voxel_downsample(scene[np.linalg.norm(scene - surf.centre, axis=1) <= reach], cfg.scene_voxel)
    stats["n_scene_points_near"] = int(len(scene))
    shell = target_shell(surf, cfg)
    stats["target_completion"] = "sphere" if surf.radius is not None else "observed only"

    raw: list[GraspCandidate] = []
    n_poses = 0
    rej = {"scene_collision": 0, "target_collision": 0, "approach_collision": 0}
    if wrist:
        rej["wrist_collision"] = 0
    surf_cache: dict[float, np.ndarray] = {}
    rolls = 2.0 * math.pi * np.arange(cfg.n_roll) / max(cfg.n_roll, 1)
    budget_hit = None
    over = lambda: cfg.time_budget_ms > 0 and _ms(t0) > cfg.time_budget_ms  # noqa: E731
    for pid, (i, j, align) in enumerate(pairs):
        if over():
            budget_hit = f"pose checks ({pid} of {len(pairs)} pairs)"
            break
        p1, p2 = surf.points[i], surf.points[j]
        u = (p2 - p1) / np.linalg.norm(p2 - p1)
        c = 0.5 * (p1 + p2)
        w = float(np.linalg.norm(p2 - p1))
        ow = min(w + cfg.opening_clearance, gripper.max_opening)
        e1, e2 = _perp_basis(u)
        boxes = gripper.body_boxes(ow)
        pr = rolls
        if vla_pose is not None and cfg.vla_roll_samples:
            # the policy's approach direction projected around this closing axis, and +-`vla_roll_step` from it
            av = -np.asarray(vla_pose, np.float64)[:3, 2]
            if np.linalg.norm(av - (av @ u) * u) > 0.2:
                r0 = math.atan2(float(av @ e2), float(av @ e1))
                extra = [r0 + k * cfg.vla_roll_step for k in range(-(cfg.vla_roll_samples // 2), cfg.vla_roll_samples // 2 + 1)]
                pr = np.concatenate([rolls, np.mod(extra, 2.0 * math.pi)])
        combos = [(roll, depth) for roll in pr for depth in cfg.depths]
        A = np.array([math.cos(r) * e1 + math.sin(r) * e2 for r, _ in combos])
        Ts = np.stack([grasp_frame(c, u, a, d) for a, (_, d) in zip(A, combos)])
        Tp = Ts.copy()
        Tp[:, :3, 3] -= cfg.pregrasp_offset * A
        n_poses += len(Ts)
        if field is not None:
            if ow not in surf_cache:
                surf_cache[ow] = gripper.surface_points(ow)
            hs = field_hits_many(field, Ts, surf_cache[ow], cfg.scene_margin)
        else:
            hs = _hits_many(scene, Ts, boxes, cfg.scene_margin)
        ht = _hits_many(shell, Ts, boxes, cfg.target_margin)
        rej["scene_collision"] += int((hs > 0).sum())
        rej["target_collision"] += int(((hs == 0) & (ht > 0)).sum())
        if wrist and field is not None:
            ok0 = np.flatnonzero((hs == 0) & (ht == 0))
            hw = np.zeros(len(Ts), int)
            hw[ok0] = field_hits_many(field, Ts[ok0], gripper.wrist_surface, cfg.scene_margin)
            rej["wrist_collision"] += int((hw > 0).sum())
            hs = hs + hw
        elif wrist:
            ok0 = np.flatnonzero((hs == 0) & (ht == 0))
            hw = np.zeros(len(Ts), int)
            hw[ok0] = shapes_hits_many(scene, Ts[ok0], wrist)
            rej["wrist_collision"] += int((hw > 0).sum())
            hs = hs + hw
        alive = np.flatnonzero((hs == 0) & (ht == 0))
        sweep = np.zeros(len(alive), int)
        if alive.size:
            for sfrac in np.linspace(0.0, 1.0, cfg.approach_samples + 1)[:-1]:
                Ss = Ts[alive].copy()
                Ss[:, :3, 3] = Tp[alive, :3, 3] + sfrac * (Ts[alive, :3, 3] - Tp[alive, :3, 3])
                sweep += _hits_many(shell, Ss, boxes, cfg.target_margin)
                if field is not None:
                    sweep += field_hits_many(field, Ss, surf_cache[ow], cfg.scene_margin)
                    if wrist:
                        sweep += field_hits_many(field, Ss, gripper.wrist_surface, cfg.scene_margin)
                else:
                    sweep += _hits_many(scene, Ss, boxes, cfg.scene_margin)
                    if wrist:
                        sweep += shapes_hits_many(scene, Ss, wrist)
        rej["approach_collision"] += int((sweep > 0).sum())
        for k, sw in zip(alive, sweep):
            if sw:
                continue
            roll, depth = combos[k]
            raw.append(GraspCandidate(index=-1, contacts=np.stack([p1, p2]), normals=np.stack([surf.normals[i], surf.normals[j]]),
                                      width=w, open_width=ow, depth=float(depth), roll=float(roll), T=Ts[k], T_pre=Tp[k],
                                      alignment=float(align), support=(int(support[i]), int(support[j])),
                                      checks={"scene_hits": 0, "target_hits": 0, "approach_hits": 0}, pair=pid))
    t_checks = _ms(t0)
    stats.update(n_poses=int(n_poses), n_collision_free=len(raw), rejected=rej,
                 pairs_ms=round(t_pairs, 2))
    # preference keys; the finger-swapped twin (Rz(pi) about the approach axis passes the same checks) closest to the
    # policy's hand is the one kept, so IK starts near the policy's wrist roll
    for g in raw:
        if vla_pose is not None:
            vp = np.asarray(vla_pose, np.float64)
            R = nearest_symmetric(g.T[:3, :3], vp[:3, :3])
            if not np.allclose(R, g.T[:3, :3]):
                g.T = g.T.copy()
                g.T_pre = g.T_pre.copy()
                g.T[:3, :3] = R
                g.T_pre[:3, :3] = R
                g.contacts = g.contacts[::-1].copy()
                g.normals = g.normals[::-1].copy()
                g.support = g.support[::-1]
            g.vla_pos_err = float(np.linalg.norm(g.T[:3, 3] - vp[:3, 3]))
            g.vla_rot_err = grasp_rot_angle(vp[:3, :3], g.T[:3, :3])
        g.key = preference_key(g.T, g.alignment, vla_pose, cfg, sel)
    raw.sort(key=lambda g: (g.key, -g.alignment, g.depth))
    # IK on the best-ordered candidates until max_candidates pass (at most max_ik tried)
    out: list[GraspCandidate] = []
    n_ik, n_ik_fail, n_arm = 0, 0, 0
    ik_log: list[dict[str, Any]] = []
    seen: set[tuple[int, float]] = set()
    for g in raw:
        if len(out) >= min(cfg.max_candidates, cfg.ik_accept if ik_fn is not None else cfg.max_candidates):
            break
        if (g.pair, round(g.roll, 6)) in seen:      # the same grasp at another pad depth
            continue
        if ik_fn is not None and over() and budget_hit is None:
            budget_hit = f"IK ({n_ik} tried)"
        if budget_hit is not None and ik_fn is not None:
            break
        if ik_fn is not None:
            if n_ik >= cfg.max_ik:
                break
            n_ik += 1
            q, info = ik_fn(g.T, ik_seed)
            g.checks["ik"] = info
            ik_log.append({"pair": g.pair, "roll_deg": round(math.degrees(g.roll), 1), "key": round(float(g.key), 4),
                           "vla_rot_err_deg": None if g.vla_rot_err is None else round(math.degrees(g.vla_rot_err), 1),
                           "vla_pos_err_mm": None if g.vla_pos_err is None else round(g.vla_pos_err * 1000, 1),
                           "converged": q is not None, "pos_err_mm": info.get("pos_err_mm"),
                           "rot_err_deg": info.get("rot_err_deg")})
            if q is None:
                n_ik_fail += 1
                continue
            qp, info_p = ik_fn(g.T_pre, q)
            g.checks["ik_pre"] = info_p
            if qp is None:
                n_ik_fail += 1
                continue
            if arm_fn is not None:
                obs = field if field is not None else scene_arm
                a, ap = arm_fn(q, obs), arm_fn(qp, obs)
                g.checks["arm"] = {"grasp": a, "pre": ap}
                ik_log[-1]["arm_hits"] = [int(a["hits"]), int(ap["hits"])]
                if a["hits"] or ap["hits"]:
                    n_arm += 1
                    continue
            g.q_grasp, g.q_pre = q, qp
        seen.add((g.pair, round(g.roll, 6)))
        g.checks["valid"] = True
        g.index = len(out)
        out.append(g)
    stats.update(budget_hit=budget_hit, checks_ms=round(t_checks - t_pairs, 2), ik_ms=round(_ms(t0) - t_checks, 2),
                 n_ik_tried=n_ik, n_ik_failed=n_ik_fail, ik_log=ik_log, n_candidates=len(out), compute_ms=_ms(t0))
    if arm_fn is not None:
        stats["n_arm_rejected"] = n_arm
    if not out:
        stats["reason"] = ("no antipodal pair" if not pairs else f"time budget spent in {budget_hit}" if budget_hit
                           else "every pose collides" if not raw
                           else "every IK solution puts the arm in an obstacle" if n_arm and n_arm + n_ik_fail >= n_ik
                           else "no IK solution")
    return CandidateSet(out, surf, stats, scene, shell, scene_arm, field)


def preference_key(T: np.ndarray, alignment: float, vla_pose: Optional[np.ndarray], cfg: GeoGraspConfig,
                   selection: str) -> float:
    """Ordering key (smaller first). ``"vla"``: position error + `vla_rot_weight` * rotation error (finger swap
    counted as the same grasp) to the policy's planned close pose. ``"geometric"``: alignment bins of `geometric_tie`
    (better first), the VLA distance only inside a bin."""
    vd = 0.0
    if vla_pose is not None:
        vp = np.asarray(vla_pose, np.float64)
        vd = float(np.linalg.norm(T[:3, 3] - vp[:3, 3])) + cfg.vla_rot_weight * grasp_rot_angle(vp[:3, :3], T[:3, :3])
    if selection == "vla":
        return vd
    if selection == "geometric":
        return -math.floor(alignment / cfg.geometric_tie) + 1e-3 * vd
    raise ValueError(f"unknown selection {selection!r}")


def close_pose_check(T: np.ndarray, gripper: ParallelGripperSpec, cfg: GeoGraspConfig,
                     surf: Optional[SurfaceEstimate], scene: Optional[np.ndarray], shell: Optional[np.ndarray],
                     open_width: Optional[float] = None, field_fn=None) -> tuple[Optional[bool], dict[str, Any]]:
    """Is a planned close pose `T` (hold link, base frame) itself a valid grasp on the observed target?

    `(None, record)` when there is nothing to judge with (no observed surface) — the caller does not block on that.
    Otherwise `(ok, record)`: under each pad face (hand frame: |y| <= pad half-width, z in the pad span, between the
    open fingers) at least `close_min_support` observed target points with trusted normals on its side; the first
    points each pad meets as it closes (the outermost on each side) form the contact pair; its width fits
    `[min_width, max_opening]`; the mean outward normal of each contact patch (`support_radius`) lies in the
    generation cone around the pad's pushing direction; and the open hand (palm + fingers at `open_width`, default the
    maximum) overlaps neither the scene points nor the completed target (`pose_checks` at `T`). The hidden side of the
    object is never assumed: a pad with no observed surface under it fails.
    """
    rec: dict[str, Any] = {}
    if surf is None or len(surf.points) == 0:
        rec["reason"] = "no observed target surface"
        return None, rec
    ow = float(gripper.max_opening if open_width is None else open_width)
    R, o = np.asarray(T, np.float64)[:3, :3], np.asarray(T, np.float64)[:3, 3]
    ph = (surf.points - o) @ R
    trusted = (np.linalg.norm(surf.normals, axis=1) > 0.5) & (surf.curvature <= cfg.max_curvature)
    under = (np.abs(ph[:, 1]) <= gripper.pad_half_y) & (ph[:, 2] >= gripper.pad_z[0]) & (ph[:, 2] <= gripper.pad_z[1]) \
        & (np.abs(ph[:, 0]) <= 0.5 * ow) & trusted
    side1, side2 = under & (ph[:, 0] < 0.0), under & (ph[:, 0] > 0.0)
    rec.update(n_under_pad1=int(side1.sum()), n_under_pad2=int(side2.sum()))
    if side1.sum() < cfg.close_min_support or side2.sum() < cfg.close_min_support:
        rec["reason"] = "no observed surface under " + ("both pads" if side1.sum() < cfg.close_min_support and
                                                        side2.sum() < cfg.close_min_support
                                                        else "pad 1" if side1.sum() < cfg.close_min_support else "pad 2")
        return False, rec
    i1 = np.flatnonzero(side1)[np.argmin(ph[side1, 0])]
    i2 = np.flatnonzero(side2)[np.argmax(ph[side2, 0])]
    width = float(ph[i2, 0] - ph[i1, 0])

    def patch_normal(i, mask):
        near = mask & (np.linalg.norm(surf.points - surf.points[i], axis=1) <= cfg.support_radius)
        n = surf.normals[near].mean(axis=0)
        return n / max(float(np.linalg.norm(n)), 1e-12)

    x = R[:, 0]
    c1 = float(-patch_normal(i1, side1) @ x)        # pad 1 pushes along +x: the surface normal there faces -x
    c2 = float(patch_normal(i2, side2) @ x)
    rec.update(width_mm=round(width * 1000, 1),
               normal_angle_deg=[round(math.degrees(math.acos(max(-1.0, min(1.0, c)))), 1) for c in (c1, c2)])
    if not (cfg.min_width <= width <= gripper.max_opening):
        rec["reason"] = "contact width outside the opening"
        return False, rec
    if min(c1, c2) < cfg.generation_cone_cos:
        rec["reason"] = "contacts not antipodal within the friction cone"
        return False, rec
    sc = np.zeros((0, 3)) if scene is None else scene
    sh = surf.points if shell is None else shell
    ok, checks, why = pose_checks(T, T, gripper.body_boxes(ow), sc, sh, cfg,
                                  wrist=tuple(gripper.wrist) if cfg.wrist_check else (),
                                  field_fn=field_fn if cfg.obstacle_source == "field" else None, gripper=gripper,
                                  open_width=ow)
    rec["checks"] = checks
    if not ok:
        rec["reason"] = why
        return False, rec
    rec["reason"] = "valid"
    return True, rec


def pose_checks(T: np.ndarray, T_pre: np.ndarray, boxes: Sequence, scene: np.ndarray, shell: np.ndarray,
                cfg: GeoGraspConfig, wrist: Sequence = (), field_fn=None, gripper: Optional[ParallelGripperSpec] = None,
                open_width: Optional[float] = None) -> tuple[bool, dict[str, Any], Optional[str]]:
    """Collision checks of one grasp pose: `(ok, checks, first failing check)`. Scene points (obstacles, support)
    inside the hand boxes grown by `scene_margin` (and inside the `wrist` shapes, already grown), the completed target
    inside the boxes (`target_margin`), and both along the approach segment `T_pre` -> `T` (`approach_samples` poses,
    the grasp pose itself excluded). With `field_fn` (and the `gripper` at `open_width`) the obstacles are that distance
    function instead of `scene` (task §3j)."""
    if field_fn is not None and gripper is not None:
        return _pose_checks_field(T, T_pre, boxes, shell, cfg, wrist, field_fn, gripper, open_width)
    checks: dict[str, Any] = {"scene_hits": _hits(scene, T, boxes, cfg.scene_margin),
                              "target_hits": _hits(shell, T, boxes, cfg.target_margin)}
    if wrist:
        checks["wrist_hits"] = int(shapes_hits_many(scene, np.asarray(T)[None], wrist)[0])
        if checks["wrist_hits"]:
            return False, checks, "wrist_collision"
    if checks["scene_hits"]:
        return False, checks, "scene_collision"
    if checks["target_hits"]:
        return False, checks, "target_collision"
    sweep = 0
    for s in np.linspace(0.0, 1.0, cfg.approach_samples + 1)[:-1]:
        Ts = T.copy()
        Ts[:3, 3] = T_pre[:3, 3] + s * (T[:3, 3] - T_pre[:3, 3])
        sweep += _hits(scene, Ts, boxes, cfg.scene_margin) + _hits(shell, Ts, boxes, cfg.target_margin)
        if wrist:
            sweep += int(shapes_hits_many(scene, Ts[None], wrist)[0])
    checks["approach_hits"] = sweep
    if sweep:
        return False, checks, "approach_collision"
    return True, checks, None


def _pose_checks_field(T, T_pre, boxes, shell, cfg, wrist, field_fn, gripper, open_width):
    """`pose_checks` against a distance field: the hand (and wrist) boundary points, then the same approach samples."""
    ow = float(gripper.max_opening if open_width is None else open_width)
    S = gripper.surface_points(ow, wrist=bool(wrist))
    T = np.asarray(T, np.float64)
    checks: dict[str, Any] = {"scene_hits": int(field_hits_many(field_fn, T[None], S, cfg.scene_margin)[0]),
                              "target_hits": _hits(shell, T, boxes, cfg.target_margin), "obstacle_source": "field"}
    if checks["scene_hits"]:
        return False, checks, "scene_collision"
    if checks["target_hits"]:
        return False, checks, "target_collision"
    Ss = []
    for s in np.linspace(0.0, 1.0, cfg.approach_samples + 1)[:-1]:
        Tk = T.copy()
        Tk[:3, 3] = T_pre[:3, 3] + s * (T[:3, 3] - T_pre[:3, 3])
        Ss.append(Tk)
    sweep = int(field_hits_many(field_fn, np.stack(Ss), S, cfg.scene_margin).sum()) + sum(
        _hits(shell, Tk, boxes, cfg.target_margin) for Tk in Ss)
    checks["approach_hits"] = sweep
    if sweep:
        return False, checks, "approach_collision"
    return True, checks, None


def target_shell(surf: SurfaceEstimate, cfg: GeoGraspConfig) -> np.ndarray:
    """The observed target points, plus the sphere-fit surface when the fit was accepted (the unseen side is not free)."""
    if surf.radius is None:
        return surf.points
    return np.vstack([surf.points, _sphere_shell(surf.centre, surf.radius, cfg.target_shell_points)])


def _wrist_reach(wrist: Sequence) -> float:
    """How far (m) the wrist shapes reach from the hold-link origin beyond the hand (0 without them)."""
    return max((float(np.linalg.norm(c)) + r for c, r in (s.bound for s in wrist)), default=0.0)


def _hits_many(points: np.ndarray, Ts: np.ndarray, boxes: Sequence, margin: float) -> np.ndarray:
    """`(K,)` number of `points` inside any of `boxes` (hand frame) at each pose `Ts[K]`, grown by `margin`.
    Points farther from the pose origin than the farthest box corner are skipped before the frame change."""
    out = np.zeros(len(Ts), int)
    if len(points) == 0 or len(Ts) == 0:
        return out
    lo_b = np.array([b.lo for b in boxes]) - margin
    hi_b = np.array([b.hi for b in boxes]) + margin
    rad2 = float(np.sum(np.maximum(np.abs(lo_b.min(axis=0)), np.abs(hi_b.max(axis=0))) ** 2))
    pts = np.asarray(points, np.float64)
    for k, T in enumerate(Ts):
        d = pts - T[:3, 3]
        q = d[np.einsum("ij,ij->i", d, d) <= rad2] @ T[:3, :3]
        if len(q):
            out[k] = int(((q[:, None, :] >= lo_b[None]) & (q[:, None, :] <= hi_b[None])).all(axis=2).any(axis=1).sum())
    return out


def _hits(points: np.ndarray, T: np.ndarray, boxes: Sequence, margin: float) -> int:
    """`_hits_many` for one pose."""
    return int(_hits_many(points, np.asarray(T)[None], boxes, margin)[0])


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 2)


__all__ = ["GeoGraspConfig", "GraspCandidate", "CandidateSet", "generate_candidates", "antipodal_pairs",
           "grasp_frame", "preference_key", "close_pose_check", "pose_checks", "target_shell", "rot_angle", "grasp_rot_angle", "nearest_symmetric"]
