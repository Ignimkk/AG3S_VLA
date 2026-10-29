"""IFG-CL1 — the **rendered-robot depth mask** on the server (`esdf.robot_mask = "render"`, CLI `--robot-mask render`).

The offline reference is `experiments/studies/ifg_render_mask.py` (IFG R6, locked): render only the robot's geoms at
the recorded pose, and call a pixel *robot* iff the robot render is valid there and the observed depth agrees with it
within `TOL` (10 mm). The server has no snapshot scene, only `--model-xml` and the request — so this module

| step | offline R6 | here |
|---|---|---|
| pose | the recorded full `qpos` (`pose_scene`) | the request's 20-joint `robot_state` (`DEFAULT_RBY1_JOINTS`) + the measured finger opening (`SafePolicy` → `AG3S.set_finger_joints`, the client's convention `gripper_state`) + the MJCF joint equalities (second finger); every other joint (head, wheels, base) from the `keyframe` (`teleop`, the pose `TransportScene.reset` starts the simulation in) |
| camera | the MJCF camera (`zed_left` · `wrist_cam_l` · `wrist_cam_r`) | the request's `K` and `T_base_cam`: one MJCF camera's pose (`data.cam_xpos/xmat`) and `fovy` are overwritten before each render — the client derived both from that same camera, so the frustum is the same |
| robot geoms | bodies below the gripper roots or `is_robot_body`, mocap excluded; others moved to group 5; groups 0–2 drawn | the same rule (`t47_replay.gripper_body_ids`, `mujoco_source.is_robot_body`) on this module's own model copy |
| decision | `valid_r = rd < 0.99 · extent · zfar`, `|depth − rd| ≤ 10 mm` | the same expression (`TOL` imported from the locked module) |

Only a centred, square-pixel `K` (what the simulated client sends: `TransportScene.intrinsics`) is rendered; any other
camera gets no mask (`None`) and the reason is recorded — the caller keeps its sphere mask for that camera.

The renderer is created lazily (one per image size) in the thread that renders; `probe()` renders once at start-up
so a missing GL backend (`MUJOCO_GL`) stops the server before the first request instead of failing every frame.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies.ifg_render_mask import TOL

#: The keyframe the simulated episode starts in (`TransportScene.reset(keyframe="teleop")`): head tilt 0.7 rad.
DEFAULT_KEYFRAME = "teleop"
#: The principal point must be within this many pixels of the image centre, fx = fy within this relative error.
CENTRE_TOL_PX = 0.5
SQUARE_TOL = 1e-6


class RobotRenderMask:
    """Renders the robot alone and compares it with the observed depth (see the module docstring). Not thread-safe."""

    def __init__(self, model_xml: str, *, keyframe: Optional[str] = DEFAULT_KEYFRAME, tol: float = TOL,
                 convention: Any = "client"):
        import mujoco

        from AG3S.ag3s.experiments.sources.mujoco_source import is_robot_body
        from AG3S.ag3s.experiments.studies.t47_replay import gripper_body_ids
        from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS

        self.mujoco = mujoco
        self.model_xml = str(pathlib.Path(model_xml).resolve())
        m = mujoco.MjModel.from_xml_path(self.model_xml)
        self.model = m
        self.data = mujoco.MjData(m)
        self.tol = float(tol)
        if m.ncam < 1:
            raise ValueError(f"{self.model_xml}: no camera to render through")
        self._cid = 0
        # the pose of everything the request does not carry
        self.keyframe = None
        base = np.asarray(m.qpos0, np.float64).copy()
        if keyframe:
            k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if k >= 0:
                base = np.asarray(m.key_qpos[k], np.float64).copy()
                self.keyframe = str(keyframe)
        self._q_base = base
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "" for j in range(m.njnt)]
        self._jid = {n: j for j, n in enumerate(names) if n}
        missing = [j for j in DEFAULT_RBY1_JOINTS if j not in self._jid]
        if missing:
            raise ValueError(f"{self.model_xml}: no joint {missing} — robot_state cannot be posed")
        self._q_adr = np.array([m.jnt_qposadr[self._jid[j]] for j in DEFAULT_RBY1_JOINTS], np.int64)
        # joint equalities (the second finger mirrors the first: polycoef "0 -1")
        self._joint_eq = []
        for i in range(m.neq):
            if int(m.eq_type[i]) == int(mujoco.mjtEq.mjEQ_JOINT) and bool(m.eq_active0[i]) and m.eq_obj2id[i] >= 0:
                self._joint_eq.append((int(m.eq_obj1id[i]), int(m.eq_obj2id[i]),
                                       np.asarray(m.eq_data[i][:5], np.float64).copy()))
        # robot geoms only (the offline rule), on this private copy of the model
        bnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
        grip = gripper_body_ids(m)
        robot_body = np.array([(i in grip or is_robot_body(bnames[i])) and m.body_mocapid[i] < 0
                               for i in range(m.nbody)])
        robot_geom = robot_body[m.geom_bodyid]
        m.geom_group[:] = np.where(robot_geom, m.geom_group, 5)
        self.n_robot_geoms = int(robot_geom.sum())
        self._opt = mujoco.MjvOption()
        self._opt.geomgroup[:] = 0
        self._opt.geomgroup[:3] = 1
        self._base_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        if self._base_bid < 0:
            raise ValueError(f"{self.model_xml}: no body 'base' (T_base_cam is in that frame)")
        self._far = 0.99 * float(m.stat.extent) * float(m.vis.map.zfar)
        self._renderers: dict[tuple[int, int], Any] = {}
        # the client's gripper convention (driven joint names, open value) — None: fingers stay at the keyframe
        self.convention = None
        self.convention_error = None
        if convention == "client":
            try:
                from AG3S.ag3s.robot_models.gripper_state import load_client_convention

                self.convention = load_client_convention()
            except (OSError, ValueError) as exc:     # the client script is not in this checkout
                self.convention_error = f"{type(exc).__name__}: {exc}"
        elif convention is not None:
            self.convention = convention

    # ------------------------------------------------------------------------------------------------
    def record(self) -> dict[str, Any]:
        return {"model_xml": self.model_xml, "keyframe": self.keyframe, "tol_m": self.tol,
                "n_robot_geoms": self.n_robot_geoms,
                "finger_convention": None if self.convention is None else self.convention.record(),
                **({"finger_convention_error": self.convention_error} if self.convention_error else {})}

    def pose(self, robot_state, openings: Optional[Mapping[str, Optional[float]]] = None) -> dict[str, Any]:
        """Put the model at `robot_state` (20,) with the hands' normalised openings (`{hand: norm}`)."""
        mj, m, d = self.mujoco, self.model, self.data
        q = self._q_base.copy()
        rs = np.asarray(robot_state, np.float64).reshape(-1)
        if rs.size != len(self._q_adr):
            raise ValueError(f"robot_state has {rs.size} values, the render pose needs {len(self._q_adr)}")
        q[self._q_adr] = rs
        driven = {}
        if openings and self.convention is not None:
            for hand, norm in openings.items():
                joint = self.convention.driven_joints.get(hand)
                if norm is None or joint not in self._jid or not np.isfinite(float(norm)):
                    continue
                j = self._jid[joint]
                v = self.convention.driven_value(float(norm))
                if m.jnt_limited[j]:
                    v = float(np.clip(v, *m.jnt_range[j]))
                q[m.jnt_qposadr[j]] = v
                driven[j] = v
        for j1, j2, c in self._joint_eq:
            a1, a2 = m.jnt_qposadr[j1], m.jnt_qposadr[j2]
            if j1 in driven and j2 not in driven and c[1] != 0.0 and not np.any(c[2:]):
                q[a2] = m.qpos0[a2] + (q[a1] - m.qpos0[a1] - c[0]) / c[1]
            elif j2 in driven or j1 not in driven:
                x = q[a2] - m.qpos0[a2]
                q[a1] = m.qpos0[a1] + c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3 + c[4] * x ** 4
        d.qpos[:] = q
        mj.mj_kinematics(m, d)
        return {"fingers": "measured" if driven else ("keyframe" if self.keyframe else "qpos0"),
                "finger_q": {mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j): round(float(v), 5)
                             for j, v in driven.items()}}

    def _renderer(self, h: int, w: int):
        key = (int(h), int(w))
        r = self._renderers.get(key)
        if r is None:
            vis = self.model.vis.global_
            vis.offwidth = max(int(vis.offwidth), key[1])
            vis.offheight = max(int(vis.offheight), key[0])
            r = self.mujoco.Renderer(self.model, key[0], key[1])
            r.enable_depth_rendering()
            self._renderers[key] = r
        return r

    @staticmethod
    def intrinsics_ok(K, hw) -> Optional[str]:
        """None when `K` is a centred square-pixel pinhole of an `hw` image, else why not."""
        K = np.asarray(K, np.float64)
        h, w = int(hw[0]), int(hw[1])
        if not (K[0, 0] > 0 and K[1, 1] > 0):
            return "non-positive focal length"
        if abs(K[0, 0] - K[1, 1]) > SQUARE_TOL * K[1, 1]:
            return f"fx {K[0, 0]:.3f} != fy {K[1, 1]:.3f}"
        if abs(K[0, 2] - w / 2.0) > CENTRE_TOL_PX or abs(K[1, 2] - h / 2.0) > CENTRE_TOL_PX:
            return f"principal point ({K[0, 2]:.2f}, {K[1, 2]:.2f}) is not the image centre ({w / 2}, {h / 2})"
        return None

    def render_depth(self, K, T_base_cam, hw) -> np.ndarray:
        """(H, W) metric depth of the posed robot alone through the camera (`K`, `T_base_cam` OpenCV, base frame)."""
        m, d = self.model, self.data
        h, w = int(hw[0]), int(hw[1])
        K = np.asarray(K, np.float64)
        T_wb = np.eye(4)
        T_wb[:3, :3] = d.xmat[self._base_bid].reshape(3, 3)
        T_wb[:3, 3] = d.xpos[self._base_bid]
        T_wc = T_wb @ np.asarray(T_base_cam, np.float64)
        # OpenCV (+z forward, +y down) → MuJoCo camera (−z forward, +y up)
        d.cam_xpos[self._cid] = T_wc[:3, 3]
        d.cam_xmat[self._cid] = (T_wc[:3, :3] @ np.diag([1.0, -1.0, -1.0])).reshape(-1)
        m.cam_fovy[self._cid] = float(np.degrees(2.0 * np.arctan((h / 2.0) / K[1, 1])))
        r = self._renderer(h, w)
        r.update_scene(d, camera=self._cid, scene_option=self._opt)
        return np.asarray(r.render(), np.float64)

    def masks(self, views: Sequence[Mapping[str, Any]], openings=None) -> tuple[list, dict[str, Any]]:
        """Per view `{"depth_m", "K", "T_base_cam", "robot_state", "name"}`: `([(H, W) bool | None], record)`."""
        t0 = time.perf_counter()
        out, per = [], {}
        posed = None
        for v in views:
            name = str(v.get("name", len(per)))
            depth = np.asarray(v["depth_m"], np.float64)
            why = self.intrinsics_ok(v["K"], depth.shape)
            if why is not None:
                out.append(None)
                per[name] = {"mask": None, "why": f"unsupported_intrinsics: {why}"}
                continue
            rs = np.asarray(v["robot_state"], np.float64)
            key = rs.tobytes()
            if posed is None or posed[0] != key:
                posed = (key, self.pose(rs, openings))
            t = time.perf_counter()
            rd = self.render_depth(v["K"], v["T_base_cam"], depth.shape)
            valid_r = rd < self._far
            mk = valid_r & (np.abs(depth - rd) <= self.tol)
            out.append(mk)
            per[name] = {"n_robot_px": int(mk.sum()), "n_rendered_px": int(valid_r.sum()),
                         "render_ms": round((time.perf_counter() - t) * 1e3, 3)}
        rec = {"cameras": per, "pose": None if posed is None else posed[1],
               "ms": round((time.perf_counter() - t0) * 1e3, 3)}
        return out, rec

    def probe(self, hw=(480, 640)) -> dict[str, Any]:
        """Render once at the keyframe pose through camera 0 (start-up check of the GL backend)."""
        mj, m, d = self.mujoco, self.model, self.data
        d.qpos[:] = self._q_base
        mj.mj_kinematics(m, d)
        f = (hw[0] / 2.0) / np.tan(np.deg2rad(45.0) / 2.0)
        K = np.array([[f, 0, hw[1] / 2.0], [0, f, hw[0] / 2.0], [0, 0, 1.0]])
        # 1.5 m in front of the base, looking back at it (OpenCV axes: x = +y_base, y = −z_base, z = −x_base)
        T_bc = np.eye(4)
        T_bc[:3, :3] = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        T_bc[:3, 3] = [1.5, 0.0, 0.8]
        t = time.perf_counter()
        rd = self.render_depth(K, T_bc, hw)
        return {"n_robot_px": int((rd < self._far).sum()), "ms": round((time.perf_counter() - t) * 1e3, 3)}

    def close(self) -> None:
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001 — closing a GL context at exit must not raise
                pass
        self._renderers = {}


__all__ = ["DEFAULT_KEYFRAME", "RobotRenderMask", "TOL"]
