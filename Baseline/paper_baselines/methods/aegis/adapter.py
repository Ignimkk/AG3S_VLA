"""VLSA / AEGIS [10] safety constraint layer as a :class:`~Baseline.paper_baselines.methods.interface.SafetyFilter`.

Official code: ``benchmark/SafeLIBERO/main/main_aegis.py`` + ``main/utils.py`` (THU-RCSCT/vlsa-aegis @ 2457feed, MIT;
submodule not modified). ``main_aegis.py`` is a script (the layer is inline in ``eval_libero``), so its per-step logic
is ported here with the original line numbers; the math helpers (``compute_h_coeffs_3d``, ``fit_ellipse`` ...) are
imported unchanged from ``utils.py`` (:mod:`.official`).

Per episode (first ``filter`` call after ``reset``; main_aegis.py:184-233):
  obstacle point cloud (:class:`ObstacleCloudSource`; VLM replaced, see README) -> ``filtering_points`` -> MVEE
  ellipsoid (p2, R2, Q2) -> ``z = (p2 - p1) / |p2 - p1|``.
Per action (main_aegis.py:301-354, 378-388):
  EE ellipsoid p1 = eef + R1 [0, 0, -0.08], Q1 = (0.06, 0.12, 0.11) (0.2 for tall objects, :149-164);
  u_ref = [5 R1^T a_xyz, 5 a_rot, 10 mu]; QP  min (u - u_ref)^T W (u - u_ref),  W = diag(1/25 x6, 1 x3)
  s.t. 0.2 a_v u_v + 0.2 a_w u_w + a_z u_z + 10 h >= 0  (cvxpy, OSQP); a_out = [0.2 R1 u_v, 0.2 u_w, a_grip];
  z <- normalize(z + (I - z z^T) u_z dt), dt = 0.05.

Differences from the official layer (also in README.md): obstacle naming without the GLM-4.5V API; several
obstacles -> one row + one ``z`` per obstacle (official: exactly one); a chunk (T > 1) is filtered with the EE pose
predicted by :func:`propagate_ee` (official: measured pose every env step -- call with T = 1 for that); QP failure ->
nominal action + log (official: ``print`` then a ``NameError`` that ends the episode, main_aegis.py:337-339); when the
nominal action already satisfies every row, it is returned exactly (it is the QP optimum) instead of OSQP's
approximate solution (``skip_inactive``).
"""
from __future__ import annotations

import time

import numpy as np

from Baseline.paper_baselines.methods.aegis import official
from Baseline.paper_baselines.methods.aegis.perception import ObstacleEllipsoid, perceive_obstacles
from Baseline.paper_baselines.methods.interface import FilterResult, SceneObservation, register_method
from Baseline.paper_baselines.methods.safety_common import (POS_GAIN_M, ROT_GAIN_RAD, ObstacleCloudSource, check_actions,
                                                             propagate_ee)

#: main_aegis.py:155 -- EE ellipsoid centre offset in the EE frame.
EE_OFFSET_LOCAL = np.array([0.0, 0.0, -0.08])
#: main_aegis.py:161-164 -- EE ellipsoid semi-axes; taller for tasks with tall objects.
Q1_DEFAULT = np.array([0.06, 0.12, 0.11])
Q1_TALL = np.array([0.06, 0.12, 0.2])
TALL_WORDS = ("orange juice", "milk", "alphabet soup")
#: main_aegis.py:304-319, 324, 344-345 -- scalings, weights, CBF gain, z step.
U_SCALE = 5.0
W_DIAG_POSE = 1.0 / 25.0
CBF_GAIN = 10.0
Z_GAIN = 10.0
Z_DT = 0.05


def ee_ellipsoid(ee_pose: np.ndarray, instruction: str = "") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(p1, R1, Q1_diag) of the AEGIS end-effector ellipsoid (main_aegis.py:150-164, 378-388)."""
    T = np.asarray(ee_pose, np.float64)
    R1 = T[:3, :3]
    p1 = T[:3, 3] + R1 @ EE_OFFSET_LOCAL
    q1 = Q1_TALL if any(w in str(instruction) for w in TALL_WORDS) else Q1_DEFAULT
    return p1, R1, q1.copy()


def solve_cbf_qp(u_ref: np.ndarray, w_diag: np.ndarray, rows: np.ndarray, rhs: np.ndarray, solver: str = "OSQP"
                 ) -> tuple[np.ndarray | None, str]:
    """``min (u-u_ref)^T diag(w) (u-u_ref)  s.t.  rows @ u + rhs >= 0`` with cvxpy (main_aegis.py:316-328)."""
    import cvxpy as cp

    u = cp.Variable(len(u_ref))
    prob = cp.Problem(cp.Minimize(cp.quad_form(u - u_ref, np.diag(w_diag))), [rows @ u + rhs >= 0])
    try:
        prob.solve(solver=getattr(cp, solver))
    except Exception as e:  # solver error -> caller falls back to the nominal action
        return None, f"error:{type(e).__name__}"
    return (None if u.value is None else np.asarray(u.value, np.float64)), str(prob.status)


class AegisSafetyFilter:
    """AEGIS CBF-QP safety layer. ``obstacle_source``: ``gt_seg`` | ``detector`` | ``given`` | ``ellipsoids``.

    ``ellipsoids``: ``extras["obstacle_ellipsoids"] = [{"name", "center", "R", "semi_axes"}, ...]`` (base frame),
    used as given (tests, or an upstream perception). Other sources build them as in the official layer.
    """

    name = "aegis"

    def __init__(self, obstacle_source: str = "gt_seg", *, detector=None, detector_prompt: str = "obstacle_names",
                 cameras: tuple[str, ...] | None = None, exclude: str = "task_objects", max_obstacles: int = 8,
                 refit_every_call: bool = False, cbf_gain: float = CBF_GAIN, solver: str = "OSQP",
                 skip_inactive: bool = True, pos_gain: float = POS_GAIN_M, rot_gain: float = ROT_GAIN_RAD,
                 intervene_tol: float = 1e-6, **source_kwargs):
        self.obstacle_source = obstacle_source
        self.source = None
        if obstacle_source != "ellipsoids":
            self.source = ObstacleCloudSource(obstacle_source, detector=detector, detector_prompt=detector_prompt,
                                              cameras=cameras, exclude=exclude, max_obstacles=max_obstacles,
                                              **source_kwargs)
        self.max_obstacles = int(max_obstacles)
        self.refit_every_call, self.cbf_gain, self.solver = bool(refit_every_call), float(cbf_gain), solver
        self.skip_inactive, self.pos_gain, self.rot_gain = bool(skip_inactive), float(pos_gain), float(rot_gain)
        self.intervene_tol = float(intervene_tol)
        self.reset()

    # --- protocol -------------------------------------------------------------------------------------------------
    def provenance(self) -> dict:
        return {"method": self.name, "cite": "[10] VLSA: Vision-Language-Action Models with Plug-and-Play Safety "
                "Constraint Layer (AEGIS), arXiv:2512.11891", "official": official.provenance(),
                "ported_lines": {"main_aegis.py": ["149-164 EE ellipsoid", "184-233 per-episode obstacle fit",
                                                   "301-354 per-step CBF-QP", "378-388 EE update"],
                                 "utils.py": ["257-316 filtering_points (scipy DBSCAN)", "155-255 get_point_cloud (box crop)"]},
                "obstacle": (self.source.provenance() if self.source is not None
                             else {"obstacle_source": "ellipsoids", "privileged": False}),
                "vlm": "not used (official obstacle_detection = GLM-4.5V API, utils.py:447-499; external API forbidden)",
                "params": {"cbf_gain": self.cbf_gain, "u_scale": U_SCALE, "W": [W_DIAG_POSE] * 6 + [1.0] * 3,
                           "z_gain": Z_GAIN, "z_dt": Z_DT, "Q1": Q1_DEFAULT.tolist(), "Q1_tall": Q1_TALL.tolist(),
                           "ee_offset": EE_OFFSET_LOCAL.tolist(), "solver": self.solver,
                           "skip_inactive": self.skip_inactive, "refit_every_call": self.refit_every_call,
                           "max_obstacles": self.max_obstacles, "chunk_pos_gain_m": self.pos_gain,
                           "chunk_rot_gain_rad": self.rot_gain}}

    def reset(self) -> None:
        self.obstacles: list[ObstacleEllipsoid] | None = None
        self.z: list[np.ndarray] = []
        self._names: list[str] = []
        self.perception_info: dict = {}

    # --- perception (once per episode) ----------------------------------------------------------------------------
    def perceive(self, obs: SceneObservation) -> list[ObstacleEllipsoid]:
        obstacles, self.perception_info = perceive_obstacles(self.source, obs, ref_point=ee_ellipsoid(obs.ee_pose)[0],
                                                             max_obstacles=self.max_obstacles)
        return obstacles

    # --- filter -----------------------------------------------------------------------------------------------------
    def filter(self, obs: SceneObservation, actions: np.ndarray) -> FilterResult:
        A = check_actions(actions)
        out = A.copy()
        T = len(A)
        intervened = np.zeros(T, bool)
        if self.obstacles is None or self.refit_every_call:
            self.obstacles = self.perceive(obs)
            p1 = ee_ellipsoid(obs.ee_pose)[0]
            old = {}
            if self.refit_every_call and self.z:
                old = dict(zip(self._names, self.z))
            self.z = [old.get(E.name, (E.center - p1) / max(np.linalg.norm(E.center - p1), 1e-12))
                      for E in self.obstacles]  # main_aegis.py:229-230
            self._names = [E.name for E in self.obstacles]
        info = {"flag_safety_control": bool(self.obstacles), "obstacles": [E.summary() for E in self.obstacles],
                "perception": self.perception_info, "steps": []}
        if not self.obstacles:  # main_aegis.py:219-221, 359-362: no obstacle -> nominal actions
            return FilterResult(out, intervened, info)
        U = official.load_official_utils()
        M = len(self.obstacles)
        T_ee = np.asarray(obs.ee_pose, np.float64)
        for k in range(T):
            a = A[k]
            p1, R1, q1 = ee_ellipsoid(T_ee, obs.instruction)
            u_ref = np.concatenate([U_SCALE * (R1.T @ a[:3]), U_SCALE * a[3:6], np.zeros(3 * M)])  # :304-307
            rows = np.zeros((M, 6 + 3 * M))
            rhs = np.zeros(M)
            hs = np.zeros(M)
            for j, E in enumerate(self.obstacles):
                a_v, a_w, a_uz, h, mu = U.compute_h_coeffs_3d(p1, q1, R1, E.center, E.semi_axes, E.R, self.z[j])  # :310
                u_ref[6 + 3 * j: 9 + 3 * j] = Z_GAIN * np.asarray(mu).ravel()  # :315
                rows[j, 0:3], rows[j, 3:6] = 0.2 * np.asarray(a_v).ravel(), 0.2 * np.asarray(a_w).ravel()  # :311-312
                rows[j, 6 + 3 * j: 9 + 3 * j] = np.asarray(a_uz).ravel()
                rhs[j] = self.cbf_gain * float(h)  # :324
                hs[j] = float(h)
            w = np.concatenate([np.full(6, W_DIAG_POSE), np.ones(3 * M)])  # :319
            nominal_ok = bool(np.all(rows @ u_ref + rhs >= 0.0))
            t0 = time.perf_counter()
            if nominal_ok and self.skip_inactive:
                u, status = u_ref.copy(), "inactive"
            else:
                u, status = solve_cbf_qp(u_ref, w, rows, rhs, self.solver)  # :327-328
                if u is None:  # official: print + NameError (:337-339); here: nominal action, logged
                    u, status = u_ref.copy(), f"fallback_nominal({status})"
            solve_ms = (time.perf_counter() - t0) * 1e3
            for j in range(M):  # :343-347
                z = self.z[j]
                uz = u[6 + 3 * j: 9 + 3 * j]
                z = z + (np.eye(3) - np.outer(z, z)) @ uz * Z_DT
                self.z[j] = z / max(np.linalg.norm(z), 1e-12)
            new = a.copy()
            new[:3] = 0.2 * R1 @ u[0:3]  # :351-354
            new[3:6] = 0.2 * u[3:6]
            if status == "inactive" or status.startswith("fallback"):
                new[:6] = a[:6]  # exact identity (0.2 * 5 round-off removed)
            out[k] = new
            intervened[k] = bool(np.max(np.abs(new[:6] - a[:6])) > self.intervene_tol)
            info["steps"].append({"h": hs.round(5).tolist(), "status": status, "nominal_satisfies": nominal_ok,
                                  "solve_ms": round(solve_ms, 3), "p1": p1.round(4).tolist()})
            T_ee = propagate_ee(T_ee, new, self.pos_gain, self.rot_gain)
        return FilterResult(out, intervened, info)


register_method("aegis")(AegisSafetyFilter)


def make_method(**kwargs) -> AegisSafetyFilter:
    return AegisSafetyFilter(**kwargs)


__all__ = ["AegisSafetyFilter", "CBF_GAIN", "EE_OFFSET_LOCAL", "Q1_DEFAULT", "Q1_TALL", "ee_ellipsoid", "make_method",
           "solve_cbf_qp"]
