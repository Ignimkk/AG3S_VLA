"""Offline evaluation of the comparison *methods* on every recorded policy call of a LIBERO run (BASELINE-EVAL F).

Input: an SL3-style run recorded with ``--capture-geometry --capture-segmentation`` (default SL3 dev
``20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0``, 10 episodes, 221 policy calls) and its scene GT
(``scene_export``; default ``outputs/verify/SL4/raw/scene``). For every call the **recorded pi0.5 action chunk** is the
nominal chunk; each method modifies it exactly as in its CPU smoke (PAPER-METHODS C / D), and the result is scored.
The episode is replayed **open loop**: the next call starts from the recorded state, not from the modified action
(methods that keep state across calls -- KNOWS / AEGIS / multi-link obstacle fits, OmniGuide freeze zone -- are
reset at every episode start and then called in time order).

Two families, two interpreters (they cannot share one process: cvxpy / torch vs JAX):

* ``filters``  (``.venv-baselines``): ``knows``, ``aegis`` (GT segmentation obstacles), ``aegis:detector_names``
  (Grounding DINO tiny + privileged LIBERO obstacle names, the VLM replacement), ``aegis:detector_all`` (generic
  prompt minus the target box), ``multilink_cbf`` -- ``SafetyFilter.filter(obs, chunk)``;
* ``guidance`` (``.venv-openpi-live``, ``JAX_PLATFORMS=cpu``): ``omniguide``, ``constrained_flow_matching``,
  ``lan_o3dp_guidance_adapted``, ``nudge`` -- ``make_guidance_fn`` inside D's **surrogate** flow sampler
  (``guidance_common.core.surrogate_sample``: straight-flow velocity towards the recorded chunk replaces the frozen
  network; ``tracking`` and ``constant`` bracket how much of a correction survives later denoising steps). This is
  a limitation recorded in every output: the real pi0.5 velocity is not used (GPU smoke = ``gpu_smoke.py``).
  Obstacle variants: ``scene`` (depth lift minus privileged target / robot / support / destination masks) and
  ``injected`` (a privileged 3 cm sphere 1 cm beside the recorded path at row 6, D's synthetic repeller check).

Metrics per (call, arm) -- ``calls.jsonl``; aggregated in ``summary.json`` (definitions in ``docs/metrics.md``):

* intervention: rows with ``max |delta a[:6]| > tol`` (filters: the method's ``intervened``; guidance: ``|delta
  a_xyz| > 1e-4`` raw, D's rule); ``intervention_rate`` = intervened rows / rows, ``call_intervened`` = any row;
* ``|delta a|``: L2 of the xyz (and rotation) part per row, LIBERO raw action units (mean over all rows, over
  intervened rows, max);
* own barrier / cost (method's own definition, not comparable across methods): filters ``h_min`` per step (KNOWS
  min ellipsoid barrier, AEGIS min over obstacles, multi-link min over (body, obstacle)) and ``h_neg_steps`` = steps
  with h < 0 (the constraint is violated at that predicted state); guidance: OmniGuide cost (maximised), NUDGE
  ``C_SDF``, CFM ``min g`` (discrete CBF rows, < 0 violated) and violated rows, Lan-o3dp waypoints inside a circle,
  before and after;
* QP: status per step, ``qp_fail`` = infeasible / solver error / fallback-to-nominal steps, KNOWS emergency stops;
* time: wall ms of ``filter`` (whole chunk) or of ``make_guidance_fn`` + jitted sampling per surrogate (CPU);
* **scene-GT clearance (same for every method)**: the chunk is decoded to an EE path with the OSC response model
  (``guidance_common.kinematics.EEPathModel``, orientation held, actions clipped to [-1, 1]); joint positions along
  the path by damped least-squares IK from the recorded ``q`` (FD Jacobian, adapter URDF); the 61 cuRobo
  ``franka.yml`` spheres at each row are measured against the **scene GT** surfaces (``libero_scene_gt`` exact
  point-triangle distance with KD-tree pruning, :meth:`GTClearance.clearance_distance`; sign from collision
  primitives): ``clearance = distance - radius``, capped at :data:`CLEAR_CAP_M` (0.05 m). Groups: ``obstacle`` = every instance except the BDDL target, destination and support
  (table); ``destination``; ``support``. Spheres: ``ee`` = panda_hand + fingers, ``arm`` = panda_link1-7. Reported for
  the nominal and the modified chunk: min over predicted rows 1..T and the count of rows with clearance < 0.

``--dump`` writes per (arm, episode) npz files for the core-function figures (``dump/<family>/<arm>/<stem>.npz``):
nominal / modified chunks and EE paths, h traces, obstacle / guarded-body ellipsoids, per-row GT clearances,
per-denoising-step corrections (guidance), and for one featured call per (arm, variant, episode) -- the call with the
largest ``|delta a|`` -- xy / xz slices of the method's field (SDF, OmniGuide energy, NUDGE cost, CFM barrier,
Lan-o3dp push).

Run (CPU, one worker; V1)::

    source Baseline/paper_baselines/setup/paper_baselines_env.sh
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=$PWD \\
      .venv-baselines/bin/python -m Baseline.paper_baselines.methods.evaluate_offline filters --run-id baseline-eval-methods-sl3dev --dump
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=$PWD \\
      .venv-openpi-live/bin/python -m Baseline.paper_baselines.methods.evaluate_offline guidance --into <run dir> --dump
    PYTHONPATH=$PWD .venv-baselines/bin/python -m Baseline.paper_baselines.methods.evaluate_offline summarize --into <run dir>
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
SL3_DEV_RUN = "20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0"
SL3_SCENE_GT = "outputs/verify/SL4/raw/scene"
CLEAR_CAP_M = 0.05
SMALL_TRI_M = 0.02  # triangles with circumradius below this go through the centroid KD-tree
EE_LINKS = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
SUPPORT_NAMES = ("main_table", "kitchen_table", "study_table", "living_room_table", "coffee_table", "floor", "table")
FILTER_ARMS: dict[str, tuple[str, dict]] = {
    "knows": ("knows", {}),
    "aegis": ("aegis", {}),
    "aegis:detector_names": ("aegis", {"obstacle_source": "detector", "detector_prompt": "obstacle_names"}),
    "aegis:detector_all": ("aegis", {"obstacle_source": "detector", "detector_prompt": "all_objects"}),
    "multilink_cbf": ("multilink_cbf", {}),
}
GUIDANCE_ARMS = ("omniguide", "constrained_flow_matching", "lan_o3dp_guidance_adapted", "nudge")
VARIANTS = ("scene", "injected")
SURROGATES = ("tracking", "constant")
GUIDE_TOL_RAW = 1e-4  # D's intervened-row rule (scripts/smoke_guidance_methods.py)
DEFAULT_NORM = "checkpoints/panda/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
LIMITATION_GUIDANCE = ("guidance is evaluated with D's surrogate flow sampler (straight-flow velocity towards the "
                       "recorded chunk, 'tracking' and 'constant' bracket), not the pi0.5 network; open-loop replay of "
                       "recorded calls (the modified chunk is not executed)")
LIMITATION_FILTERS = ("open-loop replay of recorded calls: each filter edits the recorded chunk; later actions of the "
                      "chunk use the method's own EE prediction (measured gain 0.0115 m/unit); the modified chunk is "
                      "not executed")


# =============================================================================================================
# shared: episode arrays, EE path, scene-GT clearance
# =============================================================================================================
def resolve_run(arg) -> pathlib.Path:
    p = pathlib.Path(arg)
    if p.is_dir():
        return p.resolve()
    from common.run_artifacts import results_root

    q = results_root() / str(arg)
    if not q.is_dir():
        raise FileNotFoundError(f"run not found: {arg}")
    return q.resolve()


def episode_stems(source: pathlib.Path) -> list[dict]:
    rows = [json.loads(l) for l in (source / "episodes.jsonl").read_text().splitlines() if l.strip()]
    out = []
    for r in rows:
        stem = f"{r['suite']}_t{int(r['task_id'])}_e{int(r['episode_idx'])}"
        if (source / "episodes" / f"{stem}.npz").exists() and (source / "episodes" / f"{stem}_geom.npz").exists():
            out.append({**r, "stem": stem})
    return out


class EpisodeArrays:
    """Small per-episode arrays (no images): chunk times, EE positions, robot state, T_world_base."""

    def __init__(self, source: pathlib.Path, stem: str):
        with np.load(source / "episodes" / f"{stem}.npz", allow_pickle=False) as z:
            self.chunk_t = z["chunk_t"].astype(np.int64)
            self.eef_pos = z["obs_robot0_eef_pos"].astype(np.float64)
        with np.load(source / "episodes" / f"{stem}_geom.npz", allow_pickle=False) as g:
            self.obs_index = g["obs_index"].astype(np.int64)
            self.T_world_base = g["T_world_base"].astype(np.float64)
            self.robot_state = g["robot_state/head"].astype(np.float64)

    def row(self, t: int) -> int:
        return int(np.flatnonzero(self.obs_index == int(t))[0])

    def path_inputs(self, t: int) -> dict:
        r = self.row(t)
        return {"R_bw": self.T_world_base[r][:3, :3].T, "y_prev": self.eef_pos[t] - self.eef_pos[max(t - 1, 0)],
                "robot_state": self.robot_state[r]}


def osc_path(ee_pos: np.ndarray, R_bw: np.ndarray, y_prev: np.ndarray, raw_chunk: np.ndarray) -> np.ndarray:
    """``(T + 1, 3)`` EE positions (base): current + after each row (OSC response model, D's ``path_metrics``)."""
    from Baseline.paper_baselines.methods.guidance_common.kinematics import EEPathModel

    raw = np.asarray(raw_chunk, np.float64)
    pm = EEPathModel(H=raw.shape[0], p0=np.asarray(ee_pos, np.float64), R_bw=R_bw, y_prev=y_prev)
    return np.vstack([np.asarray(ee_pos, np.float64)[None], pm.numpy(np.clip(raw[:, :3], -1, 1))])


class GTClearance:
    """Scene-GT clearance of the Panda collision spheres along a predicted EE path (see module docstring)."""

    def __init__(self, scene_path: pathlib.Path, *, cap: float = CLEAR_CAP_M, ik_iters: int = 2, damping: float = 1e-3):
        from AG3S.adapters.panda import model as panda
        from benchmark.evaluators.libero_scene_gt import SceneGT

        self.scene = SceneGT(scene_path)
        self.panda = panda
        self.model = panda.build_panda_bare_model()
        self.links = np.array(self.model.sphere_link_names)
        self.is_ee = np.isin(self.links, EE_LINKS)
        self.is_arm = ~self.is_ee & (self.links != "panda_link0")
        self.cap, self.ik_iters, self.damping = float(cap), int(ik_iters), float(damping)
        inst = self.scene.instances()
        self.groups = {"obstacle": [i for i, x in enumerate(inst) if x.role in ("object", "fixture") and x.name not in SUPPORT_NAMES],
                       "destination": [i for i, x in enumerate(inst) if x.role == "destination"],
                       "support": [i for i, x in enumerate(inst) if x.name in SUPPORT_NAMES]}
        self.names = [x.name for x in inst]
        self.used = sorted({i for ix in self.groups.values() for i in ix})  # target / mount are never measured

    def describe(self) -> dict:
        return {"scene": str(self.scene.path), "scene_sha256": self.scene.sha256, "cap_m": self.cap,
                "groups": {g: [self.names[i] for i in ix] for g, ix in self.groups.items()},
                "ee_links": list(EE_LINKS), "n_spheres": int(len(self.links)), "ik": f"DLS, {self.ik_iters} it/row, lambda {self.damping}"}

    # --- kinematics -------------------------------------------------------------------------------------------------
    def _fk(self, q):
        return self.model.link_pose(q, "panda_grip_site")

    def _jac(self, q, eps=1e-6):
        from Baseline.paper_baselines.methods.safety_common import mat_to_axisangle

        J = np.zeros((6, 7))
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = eps
            Tp, Tm = self._fk(q + dq), self._fk(q - dq)
            J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps)
            J[3:, i] = mat_to_axisangle(Tp[:3, :3] @ Tm[:3, :3].T) / (2 * eps)
        return J

    def ik_path(self, robot_state: np.ndarray, path: np.ndarray) -> np.ndarray:
        """``(T + 1, 7)`` joint positions following ``path - path[0]`` from the FK grip site, orientation held."""
        from Baseline.paper_baselines.methods.safety_common import mat_to_axisangle

        q = self.panda.set_finger_parameters(self.model, robot_state).astype(np.float64)
        T0 = self._fk(q)
        out = [q.copy()]
        for k in range(1, len(path)):
            target = T0[:3, 3] + (path[k] - path[0])
            for _ in range(self.ik_iters):
                T = self._fk(q)
                err = np.concatenate([target - T[:3, 3], mat_to_axisangle(T0[:3, :3] @ T[:3, :3].T)])
                J = self._jac(q)
                q = q + J.T @ np.linalg.solve(J @ J.T + self.damping ** 2 * np.eye(6), err)
            out.append(q.copy())
        return np.asarray(out)

    # --- clearance ---------------------------------------------------------------------------------------------------
    def signed_distance(self, pts: np.ndarray, row: int, max_distance: float) -> np.ndarray:
        """``(N, I)`` distance to each instance's visual surface, negative inside its collision primitives (``inf``
        beyond ``max_distance``; a point deeper than ``max_distance`` inside an instance gets ``-0.0``).
        Reference implementation (``SceneGT.exact_distance_near``); :meth:`clearance_distance` is the fast one."""
        D = self.scene.exact_distance_near(pts, row, max_distance=max_distance)
        inside = self.scene.inside(pts, row)
        D = np.where(inside & np.isfinite(D), -D, D)
        return np.where(inside & ~np.isfinite(D), -0.0, D)

    def _instance_geometry(self, row: int) -> list[dict | None]:
        """Per instance at frame ``row`` (cached per row): triangles split into large / small, small-triangle
        centroid KD-tree."""
        from scipy.spatial import cKDTree

        if getattr(self, "_geo_row", None) != row:
            geo = []
            for k, x in enumerate(self.scene.instances()):
                if not x.visual_geoms or k not in self.used:
                    geo.append(None)
                    continue
                tri = np.concatenate([self.scene.triangles_base(row, g) for g in x.visual_geoms])
                c = tri.mean(1)
                rt = np.sqrt(((tri - c[:, None]) ** 2).sum(-1)).max(1)
                big = rt >= SMALL_TRI_M
                g = {"big_c": c[big], "big_r": rt[big], "big_tri": tri[big], "small_tri": tri[~big]}
                if (~big).any():
                    g.update(tree=cKDTree(c[~big]), rmax=float(rt[~big].max()))
                geo.append(g)
            self._geo, self._geo_row = geo, row
        return self._geo

    def clearance_distance(self, pts: np.ndarray, radii: np.ndarray, row: int) -> np.ndarray:
        """``(N, I)`` exact signed surface distance of each sphere centre where ``distance - radius < cap``; ``inf``
        where it is provably ``>= cap + radius``. Pruning: a triangle centroid lies on the triangle, so the nearest
        centroid distance ``d_up`` bounds the exact distance from above, and ``d_up - r_t,max`` from below; only
        triangles with ``|p - c_t| - r_t <= min(d_up, cap + r)`` can hold the closest point. Triangles larger than
        :data:`SMALL_TRI_M` (tables, boxes) use the bounding-sphere bound pair by pair. Equal to
        :meth:`signed_distance` below the cap (``tests/test_evaluate_offline.py``)."""
        from benchmark.evaluators.libero_scene_gt import closest_points_pairwise

        P = np.asarray(pts, np.float64).reshape(-1, 3)
        r = np.asarray(radii, np.float64).reshape(-1)
        need = self.cap + r  # exact distance is only needed below this
        geo = self._instance_geometry(row)
        D = np.full((len(P), len(geo)), np.inf)
        for i, g in enumerate(geo):
            if g is None:
                continue
            d = np.full(len(P), np.inf)
            if len(g["big_c"]):  # few large triangles: pairs whose bounding-sphere lower bound is < need
                lb = np.linalg.norm(P[:, None, :] - g["big_c"][None], axis=2) - g["big_r"][None]
                pi, ti = np.nonzero(lb < need[:, None])
                if len(pi):
                    tb = g["big_tri"]
                    q = closest_points_pairwise(P[pi], tb[ti, 0], tb[ti, 1], tb[ti, 2])
                    np.minimum.at(d, pi, np.sqrt(((q - P[pi]) ** 2).sum(-1)))
            if "tree" in g:
                d_up, _ = g["tree"].query(P)
                todo = d_up - g["rmax"] < need  # else every small triangle is >= need away
                if todo.any():
                    idx = np.flatnonzero(todo)
                    lists = g["tree"].query_ball_point(P[idx], np.minimum(d_up[idx], need[idx]) + g["rmax"])
                    cnt = np.fromiter((len(x) for x in lists), np.int64, count=len(lists))
                    if cnt.sum():
                        pi = np.repeat(idx, cnt)
                        ti = np.concatenate([np.asarray(x, np.int64) for x in lists if len(x)])
                        ts = g["small_tri"]
                        q = closest_points_pairwise(P[pi], ts[ti, 0], ts[ti, 1], ts[ti, 2])
                        np.minimum.at(d, pi, np.sqrt(((q - P[pi]) ** 2).sum(-1)))
            d[d > need] = np.inf
            D[:, i] = d
        inside = self.scene.inside(P, row)
        D = np.where(inside & np.isfinite(D), -D, D)
        return np.where(inside & ~np.isfinite(D), -0.0, D)

    def sphere_clearance(self, q_path: np.ndarray, t: int) -> dict:
        """Per group: ``(T + 1, S)`` sphere clearance (m, capped) to that group's instances."""
        row = self.scene.frame_row(int(t))
        C, R = [], None
        for q in q_path:
            c, r = self.model.sphere_centers_numeric(q)
            C.append(np.asarray(c, np.float64))
            R = np.asarray(r, np.float64)
        C = np.asarray(C)  # (T+1, S, 3)
        D = self.clearance_distance(C.reshape(-1, 3), np.tile(R, len(C)), row)  # (N, I)
        out = {}
        for g, ix in self.groups.items():
            if not ix:
                out[g] = np.full(C.shape[:2], self.cap)
                continue
            d = D[:, ix].min(1).reshape(C.shape[:2]) - R[None]
            out[g] = np.minimum(np.where(np.isfinite(d), d, self.cap), self.cap)
        return out

    def path_metrics(self, robot_state: np.ndarray, path: np.ndarray, t: int) -> dict:
        q_path = self.ik_path(robot_state, path)
        cl = self.sphere_clearance(q_path, t)
        T = len(path) - 1
        m: dict[str, Any] = {}
        for g, a in cl.items():
            ee, arm = a[:, self.is_ee].min(1), a[:, self.is_arm].min(1)
            m[f"ee_{g}_by_row"] = ee.round(5).tolist()
            m[f"arm_{g}_by_row"] = arm.round(5).tolist()
            m[f"ee_{g}_min"] = float(ee[1:].min()) if T else None
            m[f"arm_{g}_min"] = float(arm[1:].min()) if T else None
            m[f"ee_{g}_neg_rows"] = int(np.count_nonzero(ee[1:] < 0))
            m[f"arm_{g}_neg_rows"] = int(np.count_nonzero(arm[1:] < 0))
        ob = cl["obstacle"]
        per_link = {}
        for link in np.unique(self.links[self.is_arm | self.is_ee]):
            per_link[str(link)] = float(ob[1:, self.links == link].min()) if T else None
        m["obstacle_min_by_link"] = per_link
        ee_fk = np.array([self._fk(q)[:3, 3] for q in q_path])
        m["ik_pos_err_max_m"] = float(np.max(np.linalg.norm((ee_fk - ee_fk[0]) - (path - path[0]), axis=1)))
        return m


def delta_stats(A: np.ndarray, B: np.ndarray, intervened: np.ndarray) -> dict:
    d = np.asarray(B, np.float64) - np.asarray(A, np.float64)
    nx = np.linalg.norm(d[:, :3], axis=1)
    nr = np.linalg.norm(d[:, 3:6], axis=1) if d.shape[1] >= 6 else np.zeros(len(d))
    iv = np.asarray(intervened, bool)
    return {"T": int(len(d)), "n_intervened": int(iv.sum()), "intervention_rate": float(iv.mean()) if len(iv) else None,
            "call_intervened": bool(iv.any()), "da_xyz_l2_rows": nx.round(6).tolist(),
            "da_xyz_l2_mean": float(nx.mean()), "da_xyz_l2_mean_intervened": float(nx[iv].mean()) if iv.any() else None,
            "da_xyz_l2_max": float(nx.max()), "da_rot_l2_mean": float(nr.mean()), "da_rot_l2_max": float(nr.max()),
            "max_abs_da_xyz": float(np.abs(d[:, :3]).max()),
            "gripper_unchanged": bool(np.array_equal(np.asarray(A)[:, 6], np.asarray(B)[:, 6])) if d.shape[1] > 6 else None}


def make_output(args, family: str, settings: dict) -> pathlib.Path:
    """``<run>/<family>/``: ``--into`` an existing run dir, ``--out-dir`` (below outputs/), or ``create_run``."""
    tool = "Baseline/paper_baselines/methods/evaluate_offline.py"
    if args.into:
        root = pathlib.Path(args.into).resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
    elif args.out_dir:
        root = pathlib.Path(args.out_dir).resolve()
        if not root.is_relative_to((PROJECT_ROOT / "outputs").resolve()):
            raise ValueError("--out-dir must be below outputs/ (benchmark results go through create_run)")
        root.mkdir(parents=True, exist_ok=True)
    else:
        from common.run_artifacts import create_run

        root = create_run(run_id=args.run_id, settings={"tool": tool, "family": family, **settings})
    out = root / family
    out.mkdir(exist_ok=False)
    (out / "settings.json").write_text(json.dumps({"tool": tool, "family": family, **settings}, indent=1, default=str) + "\n")
    return out


def _jsonable(x):
    from Baseline.paper_baselines.methods.interface import jsonable

    def fin(v):
        if isinstance(v, float) and not np.isfinite(v):
            return None
        if isinstance(v, dict):
            return {k: fin(w) for k, w in v.items()}
        if isinstance(v, list):
            return [fin(w) for w in v]
        return v

    return fin(jsonable(x))


def select_episodes(source, args) -> list[dict]:
    eps = episode_stems(source)
    if args.episodes:
        eps = [e for e in eps if e["stem"] in set(args.episodes)]
    if args.limit_episodes:
        eps = eps[: args.limit_episodes]
    return eps


# =============================================================================================================
# filters
# =============================================================================================================
def filter_step_info(name: str, info: Mapping) -> dict:
    """Per-step own barrier / QP status of one filter result (KNOWS / AEGIS / multi-link info layouts)."""
    steps = info.get("steps", [])
    h, status, solve_ms, h_mat, h_body = [], [], [], [], []
    estop = 0
    for s in steps:
        if s.get("min_h") is not None:
            h.append(float(s["min_h"]))
        elif s.get("h"):
            h.append(float(min(s["h"])))
            h_mat.append([float(v) for v in s["h"]])
        else:
            h.append(float("nan"))
        if "h_by_body" in s:
            h_body.append(s["h_by_body"])
        status.append(str(s.get("status")))
        if s.get("solve_ms") is not None:
            solve_ms.append(float(s["solve_ms"]))
        elif s.get("ms") is not None:
            solve_ms.append(float(s["ms"]))
        estop += int(bool(s.get("emergency_stop", False)))
    fail = sum(1 for s, st in zip(steps, status) if (s.get("feasible") is False) or st.startswith("fallback")
               or st.startswith("error") or "infeasible" in st.lower())
    h_arr = np.asarray(h, np.float64)
    return {"h_steps": h, "h_min": float(np.nanmin(h_arr)) if np.isfinite(h_arr).any() else None,
            "h_neg_steps": int(np.count_nonzero(h_arr < 0)), "qp_status": status, "qp_fail": int(fail),
            "emergency_stops": int(estop), "solve_ms_steps": solve_ms, "h_matrix": h_mat, "h_by_body": h_body,
            "min_h_pair": [s.get("min_h_pair") for s in steps if "min_h_pair" in s],
            "binding": [s.get("binding") for s in steps if "binding" in s]}


def obstacle_ellipsoids(info: Mapping) -> list[dict]:
    """``[{name, center, R, semi_axes}]`` of a filter's obstacle set (KNOWS ``{name: {c, Q}}`` or AEGIS list)."""
    obs = info.get("obstacles", [])
    out = []
    if isinstance(obs, dict):
        for k, v in obs.items():
            Q = np.asarray(v["Q"], np.float64)
            w, V = np.linalg.eigh(0.5 * (Q + Q.T))
            out.append({"name": k, "center": np.asarray(v["c"], float).tolist(), "R": V.tolist(),
                        "semi_axes": np.sqrt(np.clip(w, 0, None)).tolist()})
    else:
        for v in obs:
            out.append({"name": v["name"], "center": v["center"], "R": v.get("R", np.eye(3).tolist()), "semi_axes": v["semi_axes"]})
    return out


def run_filters(args) -> pathlib.Path:
    from Baseline.paper_baselines.methods.interface import get_method, validate_filter_result
    from Baseline.paper_baselines.methods.safety_common import POS_GAIN_M, load_recorded_call, rollout_positions

    source = resolve_run(args.source_run)
    scene_dir = _scene_dir(args)
    arms = list(args.arms or FILTER_ARMS)
    eps = select_episodes(source, args)
    settings = {"source_run": str(source), "scene_gt": str(scene_dir), "arms": arms, "episodes": [e["stem"] for e in eps],
                "limit_calls": args.limit_calls, "argv": sys.argv, "limitation": LIMITATION_FILTERS,
                "clearance_cap_m": CLEAR_CAP_M}
    out = make_output(args, "filters", settings)
    detector = None
    if any(FILTER_ARMS[a][1].get("obstacle_source") == "detector" for a in arms):
        from Baseline.paper_baselines.providers.grounding_dino import GroundingDinoDetector

        detector = GroundingDinoDetector("grounding-dino-tiny", device="cpu")
    filters = {}
    for a in arms:
        name, kw = FILTER_ARMS[a]
        if kw.get("obstacle_source") == "detector":
            kw = {**kw, "detector": detector}
        filters[a] = get_method(name, **kw)
    calls_f = (out / "calls.jsonl").open("x")
    rows: list[dict] = []
    t_start = time.time()
    prov = {}
    for ep in eps:
        stem = ep["stem"]
        EA = EpisodeArrays(source, stem)
        gtc = GTClearance(scene_dir / f"scene_{stem}.npz")
        for f in filters.values():
            f.reset()
        n_calls = len(EA.chunk_t) if not args.limit_calls else min(args.limit_calls, len(EA.chunk_t))
        dumps: dict[str, dict] = {a: {} for a in arms}
        for k in range(n_calls):
            obs, A, meta = load_recorded_call(source, stem, k)
            t = int(meta["obs_index"])
            pin = EA.path_inputs(t)
            p_nom = osc_path(obs.ee_pose[:3, 3], pin["R_bw"], pin["y_prev"], A)
            gt_nom = gtc.path_metrics(pin["robot_state"], p_nom, t)
            for a in arms:
                f = filters[a]
                t0 = time.perf_counter()
                res = f.filter(obs, A)
                ms = (time.perf_counter() - t0) * 1e3
                validate_filter_result(res, A)
                B = np.asarray(res.actions, np.float64)
                p_mod = osc_path(obs.ee_pose[:3, 3], pin["R_bw"], pin["y_prev"], B)
                gt_mod = gtc.path_metrics(pin["robot_state"], p_mod, t) if not np.array_equal(A, B) else gt_nom
                st = filter_step_info(a, res.info)
                row = {"family": "filters", "arm": a, "method": FILTER_ARMS[a][0], "episode_id": stem, "call_k": k,
                       "obs_index": t, "ms_total": ms, **delta_stats(A, B, res.intervened),
                       **{k2: v for k2, v in st.items() if k2 not in ("h_matrix", "h_by_body")},
                       "n_obstacles": len(obstacle_ellipsoids(res.info)),
                       "obstacle_names": [o["name"] for o in obstacle_ellipsoids(res.info)],
                       "perception": res.info.get("perception", {}),
                       "gt_nominal": _gt_summary(gt_nom), "gt_modified": _gt_summary(gt_mod)}
                calls_f.write(json.dumps(_jsonable(row)) + "\n")
                rows.append(_slim_row(row))
                if args.dump:
                    dumps[a][k] = {"actions_in": A, "actions_out": B, "intervened": np.asarray(res.intervened),
                                   "path_nominal_osc": p_nom, "path_modified_osc": p_mod,
                                   "path_nominal_gain": rollout_positions(obs.ee_pose, A, POS_GAIN_M),
                                   "path_modified_gain": rollout_positions(obs.ee_pose, B, POS_GAIN_M),
                                   "h_steps": np.asarray(st["h_steps"], np.float64),
                                   "h_matrix": np.asarray(st["h_matrix"], np.float64) if st["h_matrix"] else np.zeros((0, 0)),
                                   "gt_nominal_ee_obstacle": np.asarray(gt_nom["ee_obstacle_by_row"]),
                                   "gt_modified_ee_obstacle": np.asarray(gt_mod["ee_obstacle_by_row"]),
                                   "gt_nominal_arm_obstacle": np.asarray(gt_nom["arm_obstacle_by_row"]),
                                   "gt_modified_arm_obstacle": np.asarray(gt_mod["arm_obstacle_by_row"]),
                                   "ee_pose": np.asarray(obs.ee_pose), "json": json.dumps(_jsonable({
                                       "obstacles": obstacle_ellipsoids(res.info), "h_by_body": st["h_by_body"],
                                       "bodies": _guarded_bodies(f, obs), "qp_status": st["qp_status"],
                                       "robot_ellipsoid": _robot_ellipsoid(a, f, res.info), "obs_index": t}))}
                if a not in prov:
                    prov[a] = _jsonable(f.provenance())
            print(f"[filters] {stem} call {k + 1}/{n_calls} ({time.time() - t_start:.0f}s)", flush=True)
        if args.dump:
            for a, d in dumps.items():
                write_call_dump(out / "dump" / a.replace(":", "-") / f"{stem}.npz", d)
        prov[f"gt_clearance/{stem}"] = gtc.describe()
    calls_f.close()
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=str) + "\n")
    summary = summarize_family(rows, "filters")
    summary["limitation"] = LIMITATION_FILTERS
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1) + "\n")
    print(f"[filters] wrote {out}")
    return out


def _scene_dir(args) -> pathlib.Path:
    p = pathlib.Path(args.scene_gt)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _gt_summary(m: Mapping) -> dict:
    return {k: v for k, v in m.items() if not k.endswith("_by_row")}


def _slim_row(row: Mapping) -> dict:
    keep = dict(row)
    for k in ("da_xyz_l2_rows", "h_steps", "qp_status", "solve_ms_steps", "binding", "perception", "min_h_pair_rows"):
        keep.pop(k, None)
    keep["solve_ms_mean"] = float(np.mean(row["solve_ms_steps"])) if row.get("solve_ms_steps") else None
    keep["min_h_pair_set"] = sorted({p for p in row.get("min_h_pair", []) if p}) if row.get("min_h_pair") else []
    return keep


def _guarded_bodies(f, obs) -> list | None:
    if getattr(f, "name", "") != "multilink_cbf":
        return None
    q = np.asarray(obs.q, np.float64).reshape(-1)[:7]
    return [{"name": b.name, "center": b.center.tolist(), "R": b.R.tolist(), "semi_axes": b.semi_axes.tolist()}
            for b in f.robot.bodies(q, refit=False)]


def _robot_ellipsoid(arm: str, f, info: Mapping) -> dict | None:
    if arm.startswith("aegis"):
        from Baseline.paper_baselines.methods.aegis.adapter import EE_OFFSET_LOCAL, Q1_DEFAULT

        return {"kind": "aegis_ee", "semi_axes": Q1_DEFAULT.tolist(), "offset_local": EE_OFFSET_LOCAL.tolist(),
                "p1_steps": [s.get("p1") for s in info.get("steps", [])]}
    if arm == "knows":
        return {"kind": "knows_ee", "semi_axes": np.sqrt(np.diag(f.Q_local)).tolist(), "frame": "EE (grip site)"}
    return None


def write_call_dump(path: pathlib.Path, d: Mapping[int, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = {}
    for k, rec in d.items():
        for key, v in rec.items():
            arr[f"c{k:03d}/{key}"] = np.asarray(v) if not isinstance(v, str) else np.array(v)
    np.savez_compressed(path, **arr)


# =============================================================================================================
# guidance
# =============================================================================================================
def surrogate_sample_traced(x_target, noise, *, num_steps: int = 10, guidance_fn=None, kind: str = "tracking"):
    """``core.surrogate_sample`` with the per-step guidance correction recorded: returns ``(x0, corr (num_steps, H, D))``
    where ``corr[i] = guidance_fn(x_next, t_next) - x_next`` of denoising step ``i`` (normalized units). Same
    ``while_loop``, time arithmetic and velocity as ``surrogate_sample`` (bit-equality checked at run time)."""
    import jax
    import jax.numpy as jnp

    if kind not in SURROGATES:
        raise ValueError(f"kind must be one of {SURROGATES}")
    dt = -1.0 / num_steps
    x_target = jnp.asarray(x_target, jnp.float32)
    eps = jnp.asarray(noise, jnp.float32)
    corr0 = jnp.zeros((num_steps,) + x_target.shape[1:], jnp.float32)

    def step(carry):
        x_t, time, i, corr = carry
        v_t = (x_t - x_target) / time if kind == "tracking" else eps - x_target
        x_next = x_t + dt * v_t
        time_next = time + dt
        if guidance_fn is not None:
            g = guidance_fn(x_next, time_next)
            corr = corr.at[i].set((g - x_next)[0])
            x_next = g
        return x_next, time_next, i + 1, corr

    def cond(carry):
        return carry[1] >= -dt / 2

    x0, _, _, corr = jax.lax.while_loop(cond, step, (eps, jnp.float32(1.0), jnp.int32(0), corr0))
    return x0, corr


def own_costs(arm: str, m, X: np.ndarray, ctx, scene) -> list[dict]:
    """The method's own objective on normalized chunks ``X (n, H, D)`` (one jitted vmap; not comparable across
    methods): OmniGuide cost (maximised), NUDGE ``C_SDF``, CFM ``min g`` + violated rows, Lan-o3dp rows inside a circle."""
    import jax
    import jax.numpy as jnp

    n = len(X)
    if ctx is None or scene is None:
        return [{} for _ in range(n)]
    path = m.path_for(X.shape[1], scene)
    if arm in ("omniguide", "nudge"):
        f = lambda x: m.cost(x, ctx, path)  # noqa: E731
    elif arm == "constrained_flow_matching":
        f = lambda x: m.constraints(x, jnp.zeros((x.shape[0], 3), x.dtype), ctx, path)  # noqa: E731
    elif arm == "lan_o3dp_guidance_adapted":
        f = lambda x: jnp.linalg.norm(m.field(path(m.raw_xyz(x, ctx["norm"])), ctx), axis=-1)  # noqa: E731
    else:
        return [{} for _ in range(n)]
    try:
        vals = np.asarray(jax.jit(jax.vmap(f))(jnp.asarray(X, jnp.float32)), np.float64)
    except Exception as e:  # recorded, not fatal
        return [{"own_cost_error": f"{type(e).__name__}: {e}"} for _ in range(n)]
    out = []
    for v in vals:
        if arm == "omniguide":
            out.append({"omniguide_cost": float(v)})
        elif arm == "nudge":
            out.append({"nudge_c_sdf": float(v)})
        elif arm == "constrained_flow_matching":
            out.append({"cfm_min_g": float(v.min()), "cfm_violated_rows": int(np.count_nonzero(v < 0))})
        else:
            out.append({"lan_rows_inside_circle": int(np.count_nonzero(v > 0)), "lan_n_circles": int(ctx.get("n_circles", 0))})
    return out


_SD_JIT = None


def scene_sdf_metrics(raws: Sequence[np.ndarray], scene, H: int, pad_points: int = 1024) -> list[dict]:
    """D's ``evaluation.path_metrics`` (EE and hand-sphere min signed distance to the scene obstacles along the OSC
    path) for several raw chunks in one jitted call; obstacles padded to a fixed size so the call compiles once."""
    import jax
    import jax.numpy as jnp

    from Baseline.paper_baselines.methods.guidance_common.geometry import signed_distance
    from Baseline.paper_baselines.methods.guidance_common.kinematics import EEPathModel

    global _SD_JIT
    o = scene.obstacles
    if o.empty:
        return [{"ee_min_sdf_m": None, "hand_min_sdf_m": None} for _ in raws]
    if _SD_JIT is None:
        _SD_JIT = jax.jit(signed_distance)
    pm = EEPathModel(H=H, p0=scene.ee_pos, R_bw=scene.R_bw, y_prev=scene.y_prev)
    P = np.stack([pm.numpy(np.clip(np.asarray(r)[:, :3], -1, 1)) for r in raws])  # (n, H, 3)
    hs = P[:, :, None, :] + (scene.hand_c - scene.ee_pos[None])[None, None]  # (n, H, S, 3)
    pts = np.concatenate([P.reshape(-1, 3), hs.reshape(-1, 3)])
    obs = o.padded(n_points=max(pad_points, len(o.points))).as_jax(jnp.float32)
    sd = np.asarray(_SD_JIT(jnp.asarray(pts, jnp.float32), obs), np.float64)
    n = len(raws)
    ee = sd[: n * H].reshape(n, H)
    hand = sd[n * H:].reshape(n, H, -1) - np.asarray(scene.hand_r)[None, None]
    return [{"ee_min_sdf_m": float(ee[i].min()), "hand_min_sdf_m": float(hand[i].min())} for i in range(n)]


def field_slices(arm: str, m, ctx, scene, centre: np.ndarray, half: float = 0.15, n: int = 61) -> dict:
    """xy (z = centre z) and xz (y = centre y) slices of the method's field around ``centre`` (base frame)."""
    import jax
    import jax.numpy as jnp

    from Baseline.paper_baselines.methods.guidance_common.geometry import all_distances_and_closest, signed_distance

    lin = np.linspace(-half, half, n)
    out: dict[str, np.ndarray] = {"axis": lin, "centre": np.asarray(centre)}
    planes = {"xy": lambda a, b: np.stack([centre[0] + a, centre[1] + b, np.full_like(a, centre[2])], -1),
              "xz": lambda a, b: np.stack([centre[0] + a, np.full_like(a, centre[1]), centre[2] + b], -1)}
    P = m.params

    def chunked(fn, pts, chunk: int = 512):
        """Evaluate ``fn`` on grid points in chunks (the NUDGE slice is grid x spheres x obstacles)."""
        return np.concatenate([np.asarray(fn(pts[i:i + chunk]), np.float64) for i in range(0, len(pts), chunk)])

    for pl, mk in planes.items():
        A_, B_ = np.meshgrid(lin, lin, indexing="xy")
        pts_np = mk(A_, B_).reshape(-1, 3)
        pts = jnp.asarray(pts_np, jnp.float32)
        obst = ctx.get("obstacles") if ctx else None
        if obst is None:
            obst = scene.obstacles.padded().as_jax()
        sdf = chunked(lambda q: signed_distance(q, obst), pts)
        out[f"{pl}/sdf"] = sdf.reshape(n, n)
        if arm == "omniguide":
            e = P["ee_cost_weight"] * (sdf <= P["ee_sdf_threshold"]) * np.log(np.maximum(sdf, 1e-5))
            out[f"{pl}/energy_collision_ee"] = e.reshape(n, n)
            if ctx.get("attract"):
                c = np.asarray(ctx["centroid"], np.float64)
                out[f"{pl}/energy_attract"] = (-np.sum((pts_np - c) ** 2, 1)).reshape(n, n)
        elif arm == "nudge":
            off = jnp.asarray(ctx["sphere_offsets"], jnp.float32)
            r = jnp.asarray(ctx["sphere_r"], jnp.float32)

            def cost(q):
                phi = signed_distance(q[:, None, :] + off[None], obst)
                return jnp.sum(jnp.maximum(P["d_safe"] - phi + r[None], 0.0) ** 2, axis=1)

            out[f"{pl}/nudge_cost"] = chunked(cost, pts, 64).reshape(n, n)
        elif arm == "constrained_flow_matching":
            from Baseline.paper_baselines.methods.constrained_flow_matching.guidance import ellipsoid_clearance

            radii = jnp.asarray(ctx["radii"], jnp.float32)

            def h(c):
                _, cp = all_distances_and_closest(c, obst)
                return jnp.min(ellipsoid_clearance(c, cp, radii))

            out[f"{pl}/cfm_h"] = (chunked(jax.vmap(h), pts) - P["safety_margin"]).reshape(n, n)
        elif arm == "lan_o3dp_guidance_adapted":
            u = np.asarray(m.field(pts, ctx), np.float64)
            out[f"{pl}/lan_push_norm"] = np.linalg.norm(u, axis=1).reshape(n, n)
    if arm == "lan_o3dp_guidance_adapted":
        out["lan_circle_centres"] = np.asarray(ctx["centres"])[: ctx["n_circles"]]
        out["lan_circle_radii"] = np.asarray(ctx["radii"])[: ctx["n_circles"]]
    if scene is not None and len(scene.obstacles.points):
        sel = np.linalg.norm(scene.obstacles.points - centre[None], axis=1) < half * 1.5
        out["obstacle_points"] = scene.obstacles.points[sel]
    if scene is not None and scene.target_points is not None and len(scene.target_points):
        out["target_points"] = scene.target_points[:: max(1, len(scene.target_points) // 400)]
    return out


def run_guidance(args) -> pathlib.Path:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    from Baseline.paper_baselines.methods.guidance_common.core import (chunk_to_model_space, load_norm_stats,
                                                                       model_space_to_chunk, raw_to_norm_scale,
                                                                       surrogate_sample)
    from Baseline.paper_baselines.methods.guidance_common.evaluation import SCENE_KW, injected_obs
    from Baseline.paper_baselines.methods.guidance_common.recorded import load_recorded_call
    from Baseline.paper_baselines.methods.guidance_common.scene import build_scene
    from Baseline.paper_baselines.methods.interface import get_method

    source = resolve_run(args.source_run)
    scene_dir = _scene_dir(args)
    arms = list(args.arms or GUIDANCE_ARMS)
    variants = list(args.variants)
    kinds = list(args.surrogates)
    storage = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", PROJECT_ROOT.parent / "AG3S_VLA-storage"))
    norm_path = pathlib.Path(args.norm_stats) if args.norm_stats else storage / DEFAULT_NORM
    norm = load_norm_stats(str(norm_path))
    scale = raw_to_norm_scale(norm)[:3]
    eps = select_episodes(source, args)
    settings = {"source_run": str(source), "scene_gt": str(scene_dir), "arms": arms, "variants": variants,
                "surrogates": kinds, "norm_stats": str(norm_path), "scene_kwargs": SCENE_KW, "seed": args.seed,
                "episodes": [e["stem"] for e in eps], "limit_calls": args.limit_calls, "argv": sys.argv,
                "limitation": LIMITATION_GUIDANCE, "clearance_cap_m": CLEAR_CAP_M}
    out = make_output(args, "guidance", settings)
    calls_f = (out / "calls.jsonl").open("x")
    rows: list[dict] = []
    prov: dict[str, Any] = {"trace_matches_surrogate_sample": {}}
    t_start = time.time()
    for ei, ep in enumerate(eps):
        stem = ep["stem"]
        EA = EpisodeArrays(source, stem)
        gtc = GTClearance(scene_dir / f"scene_{stem}.npz")
        methods = {(a, v): get_method(a, scene_kwargs=SCENE_KW) for a in arms for v in variants}
        for mm in methods.values():
            mm.reset()
        n_calls = len(EA.chunk_t) if not args.limit_calls else min(args.limit_calls, len(EA.chunk_t))
        dumps: dict[tuple, dict] = {(a, v): {} for a in arms for v in variants}
        keep_ctx: dict[tuple, dict] = {}
        for k in range(n_calls):
            obs, chunk, meta = load_recorded_call(source, stem, k)
            t = int(meta["obs_index"])
            H = chunk.shape[0]
            pin = EA.path_inputs(t)
            x = chunk_to_model_space(chunk, norm)
            noise = np.random.default_rng([args.seed, ei, k]).standard_normal(x.shape).astype(np.float32)
            base = np.asarray(jax.jit(lambda: surrogate_sample(x, noise))())
            raw_base = model_space_to_chunk(base, norm)
            p_nom = osc_path(obs.ee_pose[:3, 3], pin["R_bw"], pin["y_prev"], raw_base)
            gt_nom = gtc.path_metrics(pin["robot_state"], p_nom, t)
            scene0 = build_scene(obs, **SCENE_KW)
            for v in variants:
                if v == "scene":
                    o, inj = obs, None
                else:
                    o, c_inj, k_inj = injected_obs(obs, scene0, chunk)
                    inj = {"centre": c_inj.tolist(), "radius": 0.03, "near_row": int(k_inj)}
                for a in arms:
                    m = methods[(a, v)]
                    t0 = time.perf_counter()
                    fn = m.make_guidance_fn(o, norm, noise=noise)
                    scene, ctx = m.last_scene, m.last_ctx
                    build_ms = (time.perf_counter() - t0) * 1e3
                    t1 = time.perf_counter()
                    traced = jax.jit(lambda: tuple(surrogate_sample_traced(x, noise, guidance_fn=fn, kind=kd) for kd in kinds))()
                    run_ms = (time.perf_counter() - t1) * 1e3
                    outs = {kd: (np.asarray(traced[i][0]), np.asarray(traced[i][1])) for i, kd in enumerate(kinds)}
                    for kd in kinds:
                        key = f"{a}|{v}|{kd}"
                        if key not in prov["trace_matches_surrogate_sample"]:
                            ref = np.asarray(jax.jit(lambda: surrogate_sample(x, noise, guidance_fn=fn, kind=kd))())
                            prov["trace_matches_surrogate_sample"][key] = {"bit_equal": bool(np.array_equal(ref, outs[kd][0])),
                                                                            "max_abs": float(np.abs(ref - outs[kd][0]).max())}
                    raws = {kd: model_space_to_chunk(outs[kd][0], norm) for kd in kinds}
                    sdm = scene_sdf_metrics([raw_base] + [raws[kd] for kd in kinds], scene, H)
                    costs = own_costs(a, m, np.stack([base[0]] + [outs[kd][0][0] for kd in kinds]), ctx, scene)
                    for i, kind in enumerate(kinds):
                        out_x, corr = outs[kind]
                        raw_g = raws[kind]
                        d = np.abs(raw_g - raw_base)[:, :3]
                        iv = d.max(1) > GUIDE_TOL_RAW
                        p_mod = osc_path(obs.ee_pose[:3, 3], pin["R_bw"], pin["y_prev"], raw_g)
                        gt_mod = gtc.path_metrics(pin["robot_state"], p_mod, t) if iv.any() else gt_nom
                        corr_xyz_raw = corr[:, :, :3] / scale[None, None]
                        row = {"family": "guidance", "arm": f"{a}|{v}|{kind}", "method": a, "variant": v, "surrogate": kind,
                               "episode_id": stem, "call_k": k, "obs_index": t, "ms_build": build_ms,
                               "ms_jit_and_sample": run_ms, **delta_stats(raw_base, raw_g, iv),
                               "finite": bool(np.all(np.isfinite(out_x))),
                               "skip": bool((ctx or {}).get("skip", False)), "frozen": bool((ctx or {}).get("frozen", False)),
                               "scene_ee_min_sdf_before": sdm[0]["ee_min_sdf_m"], "scene_ee_min_sdf_after": sdm[1 + i]["ee_min_sdf_m"],
                               "scene_hand_min_sdf_before": sdm[0]["hand_min_sdf_m"],
                               "scene_hand_min_sdf_after": sdm[1 + i]["hand_min_sdf_m"],
                               "own_before": costs[0], "own_after": costs[1 + i],
                               "step_correction_raw_max_by_step": np.abs(corr_xyz_raw).max(axis=(1, 2)).round(6).tolist(),
                               "scene_info": {"obstacle_source": scene.info.get("obstacle_source"),
                                              "obstacle_counts": scene.info.get("obstacle_counts"),
                                              "privileged_inputs": scene.info.get("privileged_inputs")} if scene else None,
                               "injected": inj, "gt_nominal": _gt_summary(gt_nom), "gt_modified": _gt_summary(gt_mod)}
                        calls_f.write(json.dumps(_jsonable(row)) + "\n")
                        rows.append(_slim_row({**row, "solve_ms_steps": None}))
                        if args.dump:
                            dumps[(a, v)].setdefault(k, {}).update({
                                "actions_nominal": raw_base, f"{kind}/actions_guided": raw_g,
                                f"{kind}/step_correction_raw_xyz": corr_xyz_raw.astype(np.float32),
                                "path_nominal_osc": p_nom, f"{kind}/path_guided_osc": p_mod,
                                "gt_nominal_ee_obstacle": np.asarray(gt_nom["ee_obstacle_by_row"]),
                                f"{kind}/gt_guided_ee_obstacle": np.asarray(gt_mod["ee_obstacle_by_row"]),
                                "gt_nominal_arm_obstacle": np.asarray(gt_nom["arm_obstacle_by_row"]),
                                f"{kind}/gt_guided_arm_obstacle": np.asarray(gt_mod["arm_obstacle_by_row"]),
                                "ee_pose": np.asarray(obs.ee_pose), "injected_json": json.dumps(inj)})
                    if args.dump:
                        keep_ctx[(a, v, k)] = (scene, ctx, p_nom)
                    if f"{a}|{v}" not in prov:
                        prov[f"{a}|{v}"] = _jsonable(m.provenance())
            if args.clear_jax_caches:
                jax.clear_caches()  # each call compiles new closures; keeps the jit cache (and RSS) bounded
            print(f"[guidance] {stem} call {k + 1}/{n_calls} ({time.time() - t_start:.0f}s)", flush=True)
        if args.dump:
            for (a, v), d in dumps.items():
                feat = featured_call([r for r in rows if r["episode_id"] == stem and r["method"] == a and r["variant"] == v])
                if feat is not None and (a, v, feat) in keep_ctx:
                    scene, ctx, p_nom = keep_ctx[(a, v, feat)]
                    centre = p_nom.mean(0)
                    try:
                        sl = field_slices(a, methods[(a, v)], ctx, scene, centre)
                        d.setdefault(feat, {}).update({f"slice/{kk}": vv for kk, vv in sl.items()})
                        d[feat]["featured"] = np.array(True)
                    except Exception as e:  # recorded; the dump still has the light data
                        d.setdefault(feat, {})["slice_error"] = f"{type(e).__name__}: {e}"
                write_call_dump(out / "dump" / f"{a}__{v}" / f"{stem}.npz", d)
            keep_ctx.clear()
        prov[f"gt_clearance/{stem}"] = gtc.describe()
    calls_f.close()
    (out / "provenance.json").write_text(json.dumps(_jsonable(prov), indent=1, default=str) + "\n")
    summary = summarize_family(rows, "guidance")
    summary["limitation"] = LIMITATION_GUIDANCE
    summary["trace_matches_surrogate_sample"] = prov["trace_matches_surrogate_sample"]
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1) + "\n")
    print(f"[guidance] wrote {out}")
    return out


def featured_call(rows: Sequence[Mapping]) -> int | None:
    """Call with the largest ``|delta a_xyz|`` (tracking surrogate first, else any); ties -> earliest."""
    if not rows:
        return None
    pref = [r for r in rows if r.get("surrogate") == "tracking" and r["da_xyz_l2_max"] > 0] or \
           [r for r in rows if r["da_xyz_l2_max"] > 0] or list(rows)
    best = max(pref, key=lambda r: (r["da_xyz_l2_max"], -r["call_k"]))
    return int(best["call_k"])


# =============================================================================================================
# summaries
# =============================================================================================================
def _mean(v):
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def _min(v):
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    return float(np.min(v)) if v else None


def summarize_arm(rs: Sequence[Mapping]) -> dict:
    n_rows = sum(r["T"] for r in rs)
    s: dict[str, Any] = {"n_calls": len(rs), "n_episodes": len({r["episode_id"] for r in rs}), "n_rows": n_rows,
                         "intervention_rate_rows": sum(r["n_intervened"] for r in rs) / n_rows if n_rows else None,
                         "intervention_rate_calls": _mean([float(r["call_intervened"]) for r in rs]),
                         "da_xyz_l2_mean": _mean([r["da_xyz_l2_mean"] for r in rs]),
                         "da_xyz_l2_mean_intervened": _mean([r["da_xyz_l2_mean_intervened"] for r in rs]),
                         "da_xyz_l2_max": max((r["da_xyz_l2_max"] for r in rs), default=None),
                         "da_rot_l2_mean": _mean([r["da_rot_l2_mean"] for r in rs]),
                         "gripper_unchanged_all": all(r.get("gripper_unchanged") in (True, None) for r in rs)}
    if "ms_total" in rs[0]:
        ms = np.array([r["ms_total"] for r in rs])
        s["ms_per_call"] = {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95)), "max": float(ms.max())}
        s["solve_ms_step_mean"] = _mean([r.get("solve_ms_mean") for r in rs])
        s["h_min"] = _min([r.get("h_min") for r in rs])
        s["h_min_call_median"] = float(np.median([r["h_min"] for r in rs if r.get("h_min") is not None])) \
            if any(r.get("h_min") is not None for r in rs) else None
        s["h_neg_steps"] = int(sum(r.get("h_neg_steps", 0) for r in rs))
        s["qp_fail_steps"] = int(sum(r.get("qp_fail", 0) for r in rs))
        s["emergency_stops"] = int(sum(r.get("emergency_stops", 0) for r in rs))
    else:
        for key in ("ms_build", "ms_jit_and_sample"):
            ms = np.array([r[key] for r in rs])
            s[key] = {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95))}
        s["skipped_calls"] = int(sum(bool(r.get("skip")) for r in rs))
        s["frozen_calls"] = int(sum(bool(r.get("frozen")) for r in rs))
        s["finite_all"] = all(r.get("finite", True) for r in rs)
        for key in ("scene_ee_min_sdf_before", "scene_ee_min_sdf_after", "scene_hand_min_sdf_before", "scene_hand_min_sdf_after"):
            s[f"{key}_min"] = _min([r.get(key) for r in rs])
        own = {}
        for when in ("own_before", "own_after"):
            keys = sorted({kk for r in rs for kk in (r.get(when) or {}) if kk != "own_cost_error"})
            own[when] = {kk: _mean([(r.get(when) or {}).get(kk) for r in rs]) for kk in keys}
        s["own_cost_mean"] = own
        s["cfm_violated_rows_before"] = int(sum((r.get("own_before") or {}).get("cfm_violated_rows", 0) for r in rs))
        s["cfm_violated_rows_after"] = int(sum((r.get("own_after") or {}).get("cfm_violated_rows", 0) for r in rs))
    for which in ("gt_nominal", "gt_modified"):
        g = {}
        for grp in ("obstacle", "destination", "support"):
            for part in ("ee", "arm"):
                g[f"{part}_{grp}_min"] = _min([r[which].get(f"{part}_{grp}_min") for r in rs])
                g[f"{part}_{grp}_neg_rows"] = int(sum(r[which].get(f"{part}_{grp}_neg_rows", 0) for r in rs))
                g[f"{part}_{grp}_neg_calls"] = int(sum(r[which].get(f"{part}_{grp}_neg_rows", 0) > 0 for r in rs))
        g["ik_pos_err_max_m"] = max((r[which].get("ik_pos_err_max_m", 0.0) for r in rs), default=None)
        s[which] = g
    s["gt_clearance_change_ee_obstacle_mean"] = _mean([
        (r["gt_modified"].get("ee_obstacle_min") - r["gt_nominal"].get("ee_obstacle_min"))
        if r["gt_modified"].get("ee_obstacle_min") is not None and r["gt_nominal"].get("ee_obstacle_min") is not None else None
        for r in rs if r["call_intervened"]])
    return s


def summarize_family(rows: Sequence[Mapping], family: str) -> dict:
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    out: dict[str, Any] = {"family": family, "arms": {a: summarize_arm(rs) for a, rs in sorted(by.items())}}
    if family == "filters" and "aegis" in by and "multilink_cbf" in by:
        ae = {(r["episode_id"], r["call_k"]): r for r in by["aegis"]}
        cases = []
        for r in by["multilink_cbf"]:
            a = ae.get((r["episode_id"], r["call_k"]))
            if a is None or not r["call_intervened"] or a["call_intervened"]:
                continue
            bodies = sorted({p.split("|")[0] for p in r.get("min_h_pair_set", [])})
            cases.append({"episode_id": r["episode_id"], "call_k": r["call_k"], "obs_index": r["obs_index"],
                          "multilink_binding_bodies": bodies, "non_gripper": any(not b.startswith("gripper") for b in bodies)})
        out["multilink_intervenes_aegis_not"] = {"n_calls": len(cases), "n_non_gripper_binding": sum(c["non_gripper"] for c in cases),
                                                 "calls": cases,
                                                 "definition": "calls where multilink_cbf changed the chunk and EE-only aegis "
                                                               "(same GT-seg obstacles) did not; binding bodies = argmin-h body per step"}
    return out


def run_summarize(args) -> pathlib.Path:
    root = pathlib.Path(args.into).resolve()
    out: dict[str, Any] = {"run": str(root), "families": {}}
    for fam in ("filters", "guidance"):
        p = root / fam / "summary.json"
        if p.exists():
            out["families"][fam] = json.loads(p.read_text())
    table = []
    for fam, d in out["families"].items():
        for arm, s in d["arms"].items():
            table.append({"family": fam, "arm": arm, "n_calls": s["n_calls"], "intervention_rate_rows": s["intervention_rate_rows"],
                          "intervention_rate_calls": s["intervention_rate_calls"], "da_xyz_l2_mean": s["da_xyz_l2_mean"],
                          "da_xyz_l2_max": s["da_xyz_l2_max"], "h_min": s.get("h_min"), "h_neg_steps": s.get("h_neg_steps"),
                          "qp_fail_steps": s.get("qp_fail_steps"),
                          "ms_median": (s.get("ms_per_call") or s.get("ms_jit_and_sample") or {}).get("median"),
                          "gt_ee_obstacle_min_nominal": s["gt_nominal"]["ee_obstacle_min"],
                          "gt_ee_obstacle_min_modified": s["gt_modified"]["ee_obstacle_min"],
                          "gt_arm_obstacle_min_nominal": s["gt_nominal"]["arm_obstacle_min"],
                          "gt_arm_obstacle_min_modified": s["gt_modified"]["arm_obstacle_min"],
                          "gt_ee_obstacle_neg_rows_nominal": s["gt_nominal"]["ee_obstacle_neg_rows"],
                          "gt_ee_obstacle_neg_rows_modified": s["gt_modified"]["ee_obstacle_neg_rows"],
                          "gt_arm_obstacle_neg_rows_nominal": s["gt_nominal"]["arm_obstacle_neg_rows"],
                          "gt_arm_obstacle_neg_rows_modified": s["gt_modified"]["arm_obstacle_neg_rows"]})
    out["table"] = table
    (root / "summary_all.json").write_text(json.dumps(_jsonable(out), indent=1) + "\n")
    print(f"[summarize] wrote {root / 'summary_all.json'} ({len(table)} arms)")
    return root


# =============================================================================================================
# CLI
# =============================================================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="family", required=True)
    for fam in ("filters", "guidance"):
        p = sub.add_parser(fam)
        p.add_argument("--source-run", default=SL3_DEV_RUN)
        p.add_argument("--scene-gt", default=SL3_SCENE_GT)
        p.add_argument("--arms", nargs="+", choices=list(FILTER_ARMS) if fam == "filters" else list(GUIDANCE_ARMS))
        p.add_argument("--episodes", nargs="*")
        p.add_argument("--limit-episodes", type=int)
        p.add_argument("--limit-calls", type=int)
        p.add_argument("--dump", action="store_true", help="figure dumps (dump/<family>/<arm>/<stem>.npz)")
        p.add_argument("--run-id")
        p.add_argument("--into", help="existing run dir: write <run>/<family>/")
        p.add_argument("--out-dir", help="below outputs/ (smoke)")
        if fam == "guidance":
            p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
            p.add_argument("--surrogates", nargs="+", default=list(SURROGATES), choices=list(SURROGATES))
            p.add_argument("--norm-stats")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--no-clear-jax-caches", dest="clear_jax_caches", action="store_false",
                           help="keep JAX compile caches across calls (faster, RSS grows)")
    p = sub.add_parser("summarize")
    p.add_argument("--into", required=True)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    {"filters": run_filters, "guidance": run_guidance, "summarize": run_summarize}[args.family](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
