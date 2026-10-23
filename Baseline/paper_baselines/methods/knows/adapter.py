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
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from Baseline.paper_baselines.methods.interface import FilterResult, SceneObservation, register_method
from Baseline.paper_baselines.methods.knows.cbf.ellipsoid import Ellipsoid
from Baseline.paper_baselines.methods.knows.cbf.filter import CbfParams, SafetyFilter as KnowsCbfFilter
from Baseline.paper_baselines.methods.knows.perception.ellipsoid_fit import mvee
from Baseline.paper_baselines.methods.safety_common import (OSC_POS_SCALE, OSC_ROT_SCALE, POS_GAIN_M, ROT_GAIN_RAD,
                                                             ObstacleCloudSource, check_actions, propagate_ee)

#: ``offline_filter.DEFAULT_GRIPPER_SEMI_AXES`` (not imported: offline_filter pulls the LIBERO camera model).
GRIPPER_SEMI_AXES = (0.04, 0.04, 0.07)


class KnowsSafetyFilter:
    name = "knows"

    def __init__(self, obstacle_source: str = "gt_seg", *, cbf_params: dict | CbfParams | None = None,
                 gripper_semi_axes=GRIPPER_SEMI_AXES, exclude: str = "task_objects", track_centroids: bool = True,
                 pos_gain: float = POS_GAIN_M, rot_gain: float = ROT_GAIN_RAD, intervene_tol: float = 1e-6,
                 **source_kwargs):
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
                "units": {"pos_scale": OSC_POS_SCALE, "rot_scale": OSC_ROT_SCALE},
                "chunk": {"pos_gain_m": self.pos_gain, "rot_gain_rad": self.rot_gain}}

    def reset(self) -> None:
        self.cbf.reset()
        self.shapes: dict[str, Ellipsoid] | None = None
        self.mean0: dict[str, np.ndarray] = {}
        self.perception_info: dict = {}

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
        if self.shapes is None:
            self.shapes = {}
            for name, pts in clouds.points.items():
                try:
                    c, Q = mvee(pts)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                self.shapes[name] = Ellipsoid(c, Q)
                self.mean0[name] = pts.mean(0)
        cur = {}
        for name, E in self.shapes.items():
            c = E.c
            if self.track_centroids and name in clouds.points:  # ObjectTracker: shape frozen, centroid moves
                c = E.c + (clouds.points[name].mean(0) - self.mean0[name])
            cur[name] = Ellipsoid(c, E.Q)
        self.perception_info = {"source": clouds.source, "privileged": clouds.privileged, "excluded": clouds.excluded,
                                "n_points": {k: int(len(v)) for k, v in clouds.points.items()},
                                "ms": (time.perf_counter() - t0) * 1e3}
        return cur

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
            res = self.cbf(robot, obstacles, dc_nom, dth_nom)
            new = a.copy()
            if res.status != "no_obstacles":
                new[:3], new[3:6] = res.delta_c / OSC_POS_SCALE, res.delta_theta / OSC_ROT_SCALE
            diff = float(np.max(np.abs(new[:6] - a[:6])))
            if diff <= self.intervene_tol:
                new[:6] = a[:6]  # solver round-off on an inactive QP is not an intervention
            out[k] = new
            intervened[k] = diff > self.intervene_tol
            info["steps"].append({"status": res.status, "feasible": bool(res.feasible), "emergency_stop": bool(res.emergency_stop),
                                  "min_h": float(res.h.min()) if res.h.size else None,
                                  "binding": [str(r[1]) for r, h in zip(res.rows, res.h) if h < 0.05][:4]})
            T_ee = propagate_ee(T_ee, new, self.pos_gain, self.rot_gain)
        return FilterResult(out, intervened, info)


register_method("knows")(KnowsSafetyFilter)


def make_method(**kwargs) -> KnowsSafetyFilter:
    return KnowsSafetyFilter(**kwargs)


__all__ = ["GRIPPER_SEMI_AXES", "KnowsSafetyFilter", "make_method"]
