"""Multi-link safety filter [1] (Agarwal & Raghunathan, "Multi-Link Safety Filtering for VLA Policies Around Moving
Hazards", arXiv:2609.40007) as a :class:`~Baseline.paper_baselines.methods.interface.SafetyFilter`.

No public code; implemented from the paper (arXiv HTML v1). The paper extends the AEGIS layer [10] from one EE
ellipsoid to several guarded link ellipsoids (Eq. 2):

    min_{u, zdot}  ||u - u_nom||^2_W + ||zdot - zdot_nom||^2_Wz
    s.t.  a^u_k G_k(x) u + a^z_k zdot_k + alpha h_k(x, z_k) >= 0   for every guarded body k (x every hazard),
    G_k = J_k J_ee^+ ,  W = I_6 / 25,  W_z = I_3,  alpha = 10,  OSQP via CVXPY,

``u`` = EE twist, ``h_k`` / ``a^u_k`` / ``a^z_k`` = the AEGIS ellipsoid-ellipsoid barrier and its derivatives
(official ``compute_h_coeffs_3d``; here mapped from the body frame to the base frame, checked by finite differences
in the tests), ``zdot_nom`` "inherited from AEGIS" (= 10 dh/dz), z integrated over the step and renormalised.
Infeasible QP -> nominal action, logged (paper Sec. "current implementation executes and logs the nominal action").

Panda bodies: :mod:`.robot` (gripper, wrist, three overlapping forearm pieces from the adapter sphere model).
Hazards: AEGIS geometry (:func:`aegis.perception.perceive_obstacles`) fitted at the first call after ``reset``.
Moving-hazard transport (sparse LK optical flow on the external camera, centre only, every 5 steps) is **not**
implemented; ``track_centers=True`` instead re-measures each obstacle's point-cloud centroid at every call and moves
the frozen ellipsoid by the centroid shift (same "centre only" model, measured by re-perception instead of flow).
"""
from __future__ import annotations

import time

import numpy as np

from Baseline.paper_baselines.methods.aegis import official
from Baseline.paper_baselines.methods.aegis.adapter import solve_cbf_qp
from Baseline.paper_baselines.methods.aegis.perception import ObstacleEllipsoid, perceive_obstacles
from Baseline.paper_baselines.methods.interface import FilterResult, SceneObservation, register_method
from Baseline.paper_baselines.methods.multilink_cbf.robot import PandaGuardedBodies, damped_pinv
from Baseline.paper_baselines.methods.safety_common import (POS_GAIN_M, ROT_GAIN_RAD, ObstacleCloudSource, check_actions)

ALPHA = 10.0  # paper Sec. IV
W_POSE = 1.0 / 25.0  # W = I_6 / 25
W_Z = 1.0  # W_z = I_3
U_SCALE = 5.0  # AEGIS action <-> twist scale (main_aegis.py:305-307, 352-353); the paper: "a fixed scale factor"
Z_GAIN = 10.0  # zdot_nom = 10 dh/dz (AEGIS main_aegis.py:315)
Z_DT = 0.05  # 20 Hz step


def barrier_rows_world(U, body_center, body_semi, body_R, obst: ObstacleEllipsoid, z):
    """AEGIS barrier of (guarded body, hazard) with base-frame pose coefficients.

    Official ``compute_h_coeffs_3d`` returns coefficients on the body-frame velocity ``R^T v`` and angular velocity;
    ``@ R^T`` maps them to base-frame ``v`` / ``omega`` (finite-difference check: ``tests/test_multilink_filter.py``).
    Returns ``(a_u (6,), a_z (3,), h, mu (3,))``.
    """
    a_v, a_w, a_uz, h, mu = U.compute_h_coeffs_3d(body_center, body_semi, body_R, obst.center, obst.semi_axes, obst.R, z)
    a_u = np.concatenate([np.asarray(a_v).ravel() @ body_R.T, np.asarray(a_w).ravel() @ body_R.T])
    return a_u, np.asarray(a_uz, float).ravel(), float(h), np.asarray(mu, float).ravel()


class MultiLinkCBFFilter:
    name = "multilink_cbf"

    def __init__(self, obstacle_source: str = "gt_seg", *, groups: dict | None = None, alpha: float = ALPHA,
                 detector=None, detector_prompt: str = "obstacle_names", cameras: tuple[str, ...] | None = None,
                 exclude: str = "task_objects", max_obstacles: int = 8, track_centers: bool = False,
                 solver: str = "OSQP", skip_inactive: bool = True, pinv_damping: float = 1e-3,
                 pos_gain: float = POS_GAIN_M, rot_gain: float = ROT_GAIN_RAD, intervene_tol: float = 1e-6,
                 **source_kwargs):
        self.obstacle_source = obstacle_source
        self.source = (None if obstacle_source == "ellipsoids" else
                       ObstacleCloudSource(obstacle_source, detector=detector, detector_prompt=detector_prompt,
                                           cameras=cameras, exclude=exclude, max_obstacles=max_obstacles,
                                           **source_kwargs))
        self.robot = PandaGuardedBodies(groups)
        self.alpha, self.max_obstacles, self.track_centers = float(alpha), int(max_obstacles), bool(track_centers)
        self.solver, self.skip_inactive, self.pinv_damping = solver, bool(skip_inactive), float(pinv_damping)
        self.pos_gain, self.rot_gain, self.intervene_tol = float(pos_gain), float(rot_gain), float(intervene_tol)
        self.reset()

    def provenance(self) -> dict:
        return {"method": self.name,
                "cite": "[1] Agarwal & Raghunathan, Multi-link safety filtering for VLA policies around moving hazards, "
                        "arXiv:2609.40007 (2026)",
                "official_code": "none (project page only); implemented from the arXiv HTML v1",
                "barrier": {"source": "AEGIS official compute_h_coeffs_3d", **official.provenance()},
                "robot": self.robot.provenance(),
                "obstacle": (self.source.provenance() if self.source is not None
                             else {"obstacle_source": "ellipsoids", "privileged": False}),
                "params": {"alpha": self.alpha, "W": W_POSE, "W_z": W_Z, "u_scale": U_SCALE, "z_gain": Z_GAIN,
                           "z_dt": Z_DT, "solver": self.solver, "skip_inactive": self.skip_inactive,
                           "pinv_damping": self.pinv_damping, "max_obstacles": self.max_obstacles,
                           "track_centers": self.track_centers, "chunk_pos_gain_m": self.pos_gain,
                           "chunk_rot_gain_rad": self.rot_gain},
                "omitted": ["moving-hazard optical-flow transport (track_centers re-measures centroids instead)",
                            "VLM hazard naming (Qwen3-VL) -> privileged names / GT seg / detector",
                            "edge-hardware policy trimming"]}

    def reset(self) -> None:
        self.obstacles: list[ObstacleEllipsoid] | None = None
        self.z: dict[tuple[str, str], np.ndarray] = {}
        self.centroid0: dict[str, np.ndarray] = {}
        self.base_centers: dict[str, np.ndarray] = {}
        self.perception_info: dict = {}

    def _robot_state(self, obs: SceneObservation, q: np.ndarray, refit: bool):
        if refit:
            self.robot.set_fingers(obs.extras.get("robot_state"), obs.gripper)
        return self.robot.bodies(q, refit=refit), self.robot.ee_jacobian(q)

    def _update_obstacles(self, obs: SceneObservation) -> None:
        if self.obstacles is None:
            ref = np.asarray(obs.ee_pose)[:3, 3]
            self.obstacles, self.perception_info = perceive_obstacles(self.source, obs, ref_point=ref,
                                                                      max_obstacles=self.max_obstacles)
            self.centroid0 = {k: np.asarray(v) for k, v in self.perception_info.get("centroids", {}).items()}
            self.base_centers = {E.name: E.center.copy() for E in self.obstacles}
        elif self.track_centers and self.source is not None:
            clouds = self.source(obs)
            for E in self.obstacles:
                if E.name in clouds.points and E.name in self.centroid0:
                    E.center = self.base_centers[E.name] + clouds.points[E.name].mean(0) - self.centroid0[E.name]

    def filter(self, obs: SceneObservation, actions: np.ndarray) -> FilterResult:
        A = check_actions(actions)
        out = A.copy()
        T = len(A)
        intervened = np.zeros(T, bool)
        self._update_obstacles(obs)
        info = {"obstacles": [E.summary() for E in self.obstacles], "perception": self.perception_info, "steps": []}
        if not self.obstacles:
            return FilterResult(out, intervened, info)
        U = official.load_official_utils()
        q = np.asarray(obs.q, np.float64).reshape(-1)[:7].copy()
        for k in range(T):
            a = A[k]
            t0 = time.perf_counter()
            bodies, J_ee = self._robot_state(obs, q, refit=(k == 0))
            J_pinv = damped_pinv(J_ee, self.pinv_damping)
            if k == 0:
                info["bodies"] = [b.summary() for b in bodies]
            pairs = [(b, E) for b in bodies for E in self.obstacles]
            P = len(pairs)
            u_ref = np.concatenate([U_SCALE * a[:6], np.zeros(3 * P)])
            rows, rhs, hs = np.zeros((P, 6 + 3 * P)), np.zeros(P), np.zeros(P)
            for i, (b, E) in enumerate(pairs):
                key = (b.name, E.name)
                if key not in self.z:  # AEGIS init: unit vector body -> hazard
                    d = E.center - b.center
                    self.z[key] = d / max(np.linalg.norm(d), 1e-12)
                a_u, a_z, h, mu = barrier_rows_world(U, b.center, b.semi_axes, b.R, E, self.z[key])
                G = b.J @ J_pinv  # (6, 6): EE twist -> body twist
                rows[i, :6] = 0.2 * (a_u @ G)
                rows[i, 6 + 3 * i: 9 + 3 * i] = a_z
                rhs[i] = self.alpha * h
                hs[i] = h
                u_ref[6 + 3 * i: 9 + 3 * i] = Z_GAIN * mu
            w = np.concatenate([np.full(6, W_POSE), np.full(3 * P, W_Z)])
            nominal_ok = bool(np.all(rows @ u_ref + rhs >= 0.0))
            if nominal_ok and self.skip_inactive:
                u, status = u_ref.copy(), "inactive"
            else:
                u, status = solve_cbf_qp(u_ref, w, rows, rhs, self.solver)
                if u is None:
                    u, status = u_ref.copy(), f"fallback_nominal({status})"
            for i, (b, E) in enumerate(pairs):
                z = self.z[(b.name, E.name)]
                z = z + (np.eye(3) - np.outer(z, z)) @ u[6 + 3 * i: 9 + 3 * i] * Z_DT
                self.z[(b.name, E.name)] = z / max(np.linalg.norm(z), 1e-12)
            new = a.copy()
            if status != "inactive" and not status.startswith("fallback"):
                new[:6] = u[:6] / U_SCALE
            out[k] = new
            intervened[k] = bool(np.max(np.abs(new[:6] - a[:6])) > self.intervene_tol)
            worst = int(np.argmin(hs))
            info["steps"].append({"status": status, "nominal_satisfies": nominal_ok, "min_h": float(hs[worst]),
                                  "min_h_pair": f"{pairs[worst][0].name}|{pairs[worst][1].name}",
                                  "h_by_body": {b.name: float(min(hs[i] for i, (bb, _) in enumerate(pairs) if bb is b))
                                                for b in bodies},
                                  "ms": round((time.perf_counter() - t0) * 1e3, 2)})
            # predicted joint motion for the next action of the chunk: q += J_ee^+ (gain-scaled twist)
            q = q + J_pinv @ np.concatenate([self.pos_gain * new[:3], self.rot_gain * new[3:6]])
        return FilterResult(out, intervened, info)


register_method("multilink_cbf")(MultiLinkCBFFilter)


def make_method(**kwargs) -> MultiLinkCBFFilter:
    return MultiLinkCBFFilter(**kwargs)


__all__ = ["ALPHA", "MultiLinkCBFFilter", "barrier_rows_world", "make_method"]
