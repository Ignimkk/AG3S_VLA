"""KNOWS [17] CBF-QP (``cbf/filter.py``) behind the common :class:`~Baseline.paper_baselines.methods.interface.SafetyFilter`.

Nothing in ``cbf/`` changes; this adapter only maps one policy call onto it, following ``offline_filter.py``:

* robot = one ellipsoid on the EE (``offline_filter.DEFAULT_GRIPPER_SEMI_AXES`` = Franka hand, gripper frame),
  centre = ``ee_pose`` translation, shape rotated with the EE (``R Q_local R^T``);
* obstacles = one MVEE per object point cloud (KNOWS ``perception.ellipsoid_fit.mvee``, Khachiyan), shapes frozen at
  the first call after ``reset`` and centroids re-measured on later calls (``ObjectTracker`` rule, Sec. 3.2);
* target exclusion (paper Eq. 4: attention top-1 object) -- by default the BDDL target and destination from
  privileged GT (``ObstacleCloudSource(exclude="task_objects")``); ``exclude="phase_target"`` is the
  ``offline_filter --target gt`` rule (target before grasp, destination after); an upstream attention / grounding
  module can instead hand over target-free clouds with ``obstacle_source="given"``;
* units: ``delta_c = a[:3] * 0.05 m``, ``delta_theta = a[3:6] * 0.5 rad`` (robosuite OSC ``output_max``, Appendix 7.1
  "scaled to physical units") and back; emergency stop -> zero motion (Appendix 7.1); gripper untouched;
* chunk (T > 1): the EE pose before action k is predicted with :func:`propagate_ee` (measured gain); T = 1 per env
  step reproduces the per-step filter.

``CbfParams()`` defaults reproduce the paper (Eq. 12 as written); pass ``cbf_params={...}`` for the D1-D7 variants.

Track identity (CLOSED-LOOP-FIX, ``associate``): the paper keeps one track per object (shape frozen at t=0, centroid
re-measured, **held at the last position while occluded**, identity swaps re-associated by an HSV histogram match;
§3.2, ``perception/ellipsoid_fit.ObjectTracker``). Source names are stable for ``gt_seg`` (instance names) but not for
``detector`` (``object#j`` = j-th box by score, re-ranked every call) or ``given`` (``depth#j`` = j-th largest
component). ``associate="centroid"`` (default for those sources) matches each frozen track to the nearest current
cloud centroid (Hungarian, gate ``assoc_gate_m``; centroid distance replaces the colour histogram, which the box
crops cannot provide) and holds unmatched tracks at their last centroid. ``associate="name"`` (default for
``gt_seg`` / ``ellipsoids``) is the previous behaviour (match by name, unmatched -> t=0 centroid).

Per-step diagnostics (``info["steps"][k]``, ``diagnostics=True``; nothing here feeds back into the QP):

* ``min_h`` -- Eq. (6) ``h(n)`` at the **stored** separating normal (the QP's own row value, before the step's
  ``delta_n``), at the EE pose before row ``k`` along the **filtered** chunk (row 0 = measured pose, rows k > 0 =
  ``propagate_ee`` of the filtered rows 0..k-1). This is the quantity the closed-loop ``h_min`` / ``h_neg_steps`` count;
* ``h_star`` / ``h_star_next`` -- the true separation ``max_n h(n)`` (``optimal_normals_batch``) before the row and
  after the QP's step ``(delta_c, delta_theta)``; ``h_star_next_nominal`` -- after the nominal step instead;
* ``cbf_ok`` -- docs/16 §1 condition on the true barrier, every obstacle: ``h*_next >= (1 - gamma_h) h*``;
* ``relax_m`` / ``need_m`` (binding row = min ``h(n)``): ``eps * ||grad_n h||_1`` (what tilting the normal inside the
  Eq. 12 box can supply, ``cbf.filter.effective_margin``) vs ``gamma_h * max(-h, 0)`` (what Eq. 7 demands of an
  overlapping pair). ``relax_m >= need_m`` = the row can be met by ``delta_n`` alone (docs/11 §4).
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from Baseline.paper_baselines.methods.interface import FilterResult, SceneObservation, register_method
from Baseline.paper_baselines.methods.knows.cbf.ellipsoid import (Ellipsoid, barrier, grad_normal, initial_normal,
                                                                  optimal_normals_batch, rotate_shape)
from Baseline.paper_baselines.methods.knows.cbf.filter import CbfParams, SafetyFilter as KnowsCbfFilter
from Baseline.paper_baselines.methods.knows.perception.ellipsoid_fit import mvee
from Baseline.paper_baselines.methods.safety_common import (OSC_POS_SCALE, OSC_ROT_SCALE, POS_GAIN_M, ROT_GAIN_RAD,
                                                             ObstacleCloudSource, check_actions, propagate_ee)

#: ``offline_filter.DEFAULT_GRIPPER_SEMI_AXES`` (not imported: offline_filter pulls the LIBERO camera model).
GRIPPER_SEMI_AXES = (0.04, 0.04, 0.07)


class KnowsSafetyFilter:
    name = "knows"

    ASSOCIATE = ("auto", "name", "centroid")

    def __init__(self, obstacle_source: str = "gt_seg", *, cbf_params: dict | CbfParams | None = None,
                 gripper_semi_axes=GRIPPER_SEMI_AXES, exclude: str = "task_objects", track_centroids: bool = True,
                 pos_gain: float = POS_GAIN_M, rot_gain: float = ROT_GAIN_RAD, intervene_tol: float = 1e-6,
                 associate: str = "auto", assoc_gate_m: float = 0.10, diagnostics: bool = True,
                 **source_kwargs):
        if associate not in self.ASSOCIATE:
            raise ValueError(f"associate must be one of {self.ASSOCIATE}")
        if associate == "auto":
            associate = "name" if obstacle_source in ("gt_seg", "ellipsoids") else "centroid"
        self.associate, self.assoc_gate_m, self.diagnostics = associate, float(assoc_gate_m), bool(diagnostics)
        if isinstance(cbf_params, CbfParams):
            self.params = cbf_params
        else:
            self.params = CbfParams(**(cbf_params or {}))
        self.obstacle_source = obstacle_source
        self.source = (None if obstacle_source == "ellipsoids"
                       else ObstacleCloudSource(obstacle_source, exclude=exclude, **source_kwargs))
        self.Q_local = np.diag(np.asarray(gripper_semi_axes, np.float64) ** 2)
        self.track_centroids, self.pos_gain, self.rot_gain = bool(track_centroids), float(pos_gain), float(rot_gain)
        self.intervene_tol = float(intervene_tol)
        self.cbf = KnowsCbfFilter(self.params)
        self.reset()

    def provenance(self) -> dict:
        return {"method": self.name,
                "cite": "[17] Your Model Already Knows: Attention-Guided Safety Filter for VLA Models, arXiv:2606.09749",
                "official_code": "none (paper-only reimplementation, Baseline/paper_baselines/methods/knows, moved from "
                                 "Baseline/knows_vla)",
                "cbf_params": dataclasses.asdict(self.params) | {"joint_weights": None},
                "robot": {"semi_axes": np.sqrt(np.diag(self.Q_local)).tolist(), "frame": "EE (grip site)"},
                "obstacle": (self.source.provenance() if self.source is not None
                             else {"obstacle_source": "ellipsoids", "privileged": False}),
                "obstacle_fit": "KNOWS mvee (Khachiyan), shapes frozen at first call, centroids tracked",
                "track_identity": {"associate": self.associate, "gate_m": self.assoc_gate_m,
                                   "unmatched_track": "hold last centroid" if self.associate == "centroid"
                                   else "t=0 centroid"},
                "diagnostics": self.diagnostics,
                "units": {"pos_scale": OSC_POS_SCALE, "rot_scale": OSC_ROT_SCALE},
                "chunk": {"pos_gain_m": self.pos_gain, "rot_gain_rad": self.rot_gain}}

    def reset(self) -> None:
        self.cbf.reset()
        self.shapes: dict[str, Ellipsoid] | None = None
        self.mean0: dict[str, np.ndarray] = {}
        self.last_c: dict[str, np.ndarray] = {}
        self.perception_info: dict = {}

    def _match(self, clouds: dict[str, np.ndarray]) -> tuple[dict[str, str], dict]:
        """Track -> current cloud name: Hungarian on centroid distance (last known centroid), gated."""
        from scipy.optimize import linear_sum_assignment

        tracks = list(self.shapes)
        names = list(clouds)
        if not tracks or not names:
            return {}, {"matched": 0, "unmatched_tracks": len(tracks), "unmatched_clouds": len(names)}
        ref = np.stack([self.last_c[t] for t in tracks])
        cur = np.stack([clouds[n].mean(0) for n in names])
        D = np.linalg.norm(ref[:, None, :] - cur[None, :, :], axis=-1)
        r, c = linear_sum_assignment(D)
        m = {tracks[i]: names[j] for i, j in zip(r, c) if D[i, j] <= self.assoc_gate_m}
        return m, {"matched": len(m), "unmatched_tracks": len(tracks) - len(m), "unmatched_clouds": len(names) - len(m),
                   "pairs": {t: [n, round(float(D[tracks.index(t), names.index(n)]), 4)] for t, n in m.items()}}

    # --- obstacles ---------------------------------------------------------------------------------------------------
    def _obstacles(self, obs: SceneObservation) -> dict[str, Ellipsoid]:
        if self.obstacle_source == "ellipsoids":
            out = {}
            for d in obs.extras.get("obstacle_ellipsoids", []):
                R = np.asarray(d.get("R", np.eye(3)), float)
                out[str(d["name"])] = Ellipsoid.from_semi_axes(d["center"], d["semi_axes"], R)
            return out
        t0 = time.perf_counter()
        clouds = self.source(obs)
        first = self.shapes is None
        if first:
            self.shapes = {}
            for name, pts in clouds.points.items():
                try:
                    c, Q = mvee(pts)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                self.shapes[name] = Ellipsoid(c, Q)
                self.mean0[name] = pts.mean(0)
                self.last_c[name] = pts.mean(0)
        cur = {}
        assoc = None
        if self.associate == "centroid":
            m, assoc = (({n: n for n in self.shapes}, {"matched": len(self.shapes), "first_call": True}) if first
                        else self._match(clouds.points))
            for name, E in self.shapes.items():
                if self.track_centroids and name in m:  # ObjectTracker: shape frozen, centroid moves
                    self.last_c[name] = clouds.points[m[name]].mean(0)
                # unmatched (occluded / not detected): hold the last known centroid (paper §3.2)
                cur[name] = Ellipsoid(E.c + (self.last_c[name] - self.mean0[name]), E.Q)
        else:
            for name, E in self.shapes.items():
                c = E.c
                if self.track_centroids and name in clouds.points:  # ObjectTracker: shape frozen, centroid moves
                    c = E.c + (clouds.points[name].mean(0) - self.mean0[name])
                cur[name] = Ellipsoid(c, E.Q)
        self.perception_info = {"source": clouds.source, "privileged": clouds.privileged, "excluded": clouds.excluded,
                                "n_points": {k: int(len(v)) for k, v in clouds.points.items()},
                                "associate": self.associate, "association": assoc,
                                "ms": (time.perf_counter() - t0) * 1e3}
        return cur

    def _diagnose(self, robot: Ellipsoid, obstacles: dict[str, Ellipsoid], res, dc_nom, dth_nom, stored: dict) -> dict:
        """True-barrier view of one step (see module docstring); read-only w.r.t. the filter state."""
        p = self.params
        keys = sorted(obstacles, key=repr)
        if not keys:
            return {}
        obs_l = [obstacles[k] for k in keys]

        def h_star(R):
            N = optimal_normals_batch([R] * len(keys), obs_l, inits=[None] * len(keys), iters=400)
            return np.array([barrier(n, R, o) for n, o in zip(N, obs_l)])

        def moved(dc, dth):
            return Ellipsoid(robot.c + np.asarray(dc, np.float64), rotate_shape(robot.Q, np.asarray(dth, np.float64)))

        hb = h_star(robot)
        ha = h_star(moved(res.delta_c, res.delta_theta))
        hn = h_star(moved(dc_nom, dth_nom))
        ok = ha >= (1.0 - p.gamma_h) * hb - 1e-9
        out = {"h_star": float(hb.min()), "h_star_next": float(ha.min()), "h_star_next_nominal": float(hn.min()),
               "cbf_ok": bool(ok.all()), "n_cbf_violated": int(np.count_nonzero(~ok))}
        if res.h.size:  # binding row = min h(n): what delta_n alone could supply vs what Eq. 7 asks
            i = int(np.argmin(res.h))
            key = res.rows[i] if i < len(res.rows) else None
            okey = key[1] if key is not None else None
            n = stored.get(key) if key is not None else None
            if okey in obstacles and n is None:
                n = initial_normal(robot, obstacles[okey])  # first sight: the row used the centre-to-centre normal
            if okey in obstacles and n is not None:
                g = grad_normal(n, robot, obstacles[okey])
                out.update(relax_m=float(p.eps_normal * np.abs(g).sum()) if p.normals == "relaxed" else 0.0,
                           need_m=float(p.gamma_h * max(-float(res.h[i]), 0.0)), binding_obstacle=str(okey))
        return out

    # --- filter --------------------------------------------------------------------------------------------------------
    def filter(self, obs: SceneObservation, actions: np.ndarray) -> FilterResult:
        A = check_actions(actions)
        out = A.copy()
        intervened = np.zeros(len(A), bool)
        obstacles = self._obstacles(obs)
        info = {"obstacles": {k: {"c": E.c.round(4).tolist(), "Q": E.Q.round(7).tolist(),
                                  "semi_axes": np.sqrt(np.clip(np.linalg.eigvalsh(E.Q), 0, None)).round(4).tolist()}
                              for k, E in obstacles.items()}, "perception": self.perception_info, "steps": []}
        T_ee = np.asarray(obs.ee_pose, np.float64)
        for k, a in enumerate(A):
            R = T_ee[:3, :3]
            robot = Ellipsoid(T_ee[:3, 3], R @ self.Q_local @ R.T)
            dc_nom, dth_nom = a[:3] * OSC_POS_SCALE, a[3:6] * OSC_ROT_SCALE
            stored = self.cbf.normals() if self.diagnostics else None  # normals the QP rows use (before delta_n)
            res = self.cbf(robot, obstacles, dc_nom, dth_nom)
            new = a.copy()
            if res.status != "no_obstacles":
                new[:3], new[3:6] = res.delta_c / OSC_POS_SCALE, res.delta_theta / OSC_ROT_SCALE
            diff = float(np.max(np.abs(new[:6] - a[:6])))
            if diff <= self.intervene_tol:
                new[:6] = a[:6]  # solver round-off on an inactive QP is not an intervention
            out[k] = new
            intervened[k] = diff > self.intervene_tol
            step = {"status": res.status, "feasible": bool(res.feasible), "emergency_stop": bool(res.emergency_stop),
                    "min_h": float(res.h.min()) if res.h.size else None,
                    "binding": [str(r[1]) for r, h in zip(res.rows, res.h) if h < 0.05][:4]}
            if self.diagnostics and res.status != "no_obstacles":
                step.update(self._diagnose(robot, obstacles, res, dc_nom, dth_nom, stored))
            info["steps"].append(step)
            T_ee = propagate_ee(T_ee, new, self.pos_gain, self.rot_gain)
        return FilterResult(out, intervened, info)


register_method("knows")(KnowsSafetyFilter)


def make_method(**kwargs) -> KnowsSafetyFilter:
    return KnowsSafetyFilter(**kwargs)


__all__ = ["GRIPPER_SEMI_AXES", "KnowsSafetyFilter", "make_method"]
