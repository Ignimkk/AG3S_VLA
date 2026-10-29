"""T63 — cuRobo v2 robot model and voxel world for the served RB-Y1 constraint model.

Needs `torch` + `curobo` (`.venv-curobo`) and a CUDA device (cuRobo's kinematics loader refuses a
CPU device). The pure mapping rules are in `spec.py` / `world.py` and are unit-tested on CPU.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.types.robot import RobotCfg
from curobo.kinematics import Kinematics
from curobo.scene import VoxelGrid
from curobo.types import DeviceCfg, JointState

from AG3S.trajopt.curobo_v2 import spec as S
from AG3S.trajopt.curobo_v2 import world as W

#: the three recorded layers, in the order they are loaded into one voxel cache
LAYER_NAMES = ("coarse", "main", "free")


def device_cfg(device: str = "cuda:0") -> DeviceCfg:
    return DeviceCfg(device=torch.device(device), dtype=torch.float32)


def build_robot_cfg(spec: Mapping[str, Any], dev: DeviceCfg) -> RobotCfg:
    """`RobotCfg` with the served spheres and the TO's joint limits (overridden in memory)."""
    rc = RobotCfg.create(S.robot_cfg_dict(spec), dev)
    apply_limits(rc, spec)
    return rc


def apply_limits(rc: RobotCfg, spec: Mapping[str, Any]) -> None:
    """Overwrite the URDF position / velocity / acceleration limits with the TO's (in place)."""
    jl = rc.kinematics.kinematics_config.joint_limits
    names = list(jl.joint_names)
    want = list(spec["arm_joint_names"])
    if names != want:
        raise ValueError(f"cuRobo joint order {names} != c-space order {want}")
    lim = S.limits_override(spec)
    for key in ("position", "velocity", "acceleration"):
        t = getattr(jl, key)
        t.copy_(torch.as_tensor(lim[key], dtype=t.dtype, device=t.device))


def joint_limits_now(rc: RobotCfg) -> dict[str, list]:
    jl = rc.kinematics.kinematics_config.joint_limits
    return {k: getattr(jl, k).detach().cpu().numpy().tolist()
            for k in ("position", "velocity", "acceleration", "jerk")}


def kinematics_params_in(objects: Iterable[Any]) -> list[Any]:
    """Every distinct `KinematicsParams` reachable from kinematics / rollout objects (dedup by the
    identity of the `link_spheres` tensor). Finger updates must reach all of them, because each
    rollout may hold its own copy of the robot model."""
    out, seen = [], set()

    def visit(kin):
        kp = getattr(kin, "kinematics_config", None)
        if kp is None or getattr(kp, "link_spheres", None) is None:
            return
        key = kp.link_spheres.data_ptr()
        if key not in seen:
            seen.add(key)
            out.append(kp)

    for obj in objects:
        if obj is None:
            continue
        if isinstance(obj, Kinematics):
            visit(obj)
            continue
        for attr in ("transition_model", "metrics_transition_model"):
            tm = getattr(obj, attr, None)
            if tm is not None:
                visit(getattr(tm, "robot_model", None))
        visit(getattr(obj, "kinematics", None))
    return out


def set_finger_openings(kparams: Iterable[Any], spec: Mapping[str, Any],
                        values: Mapping[str, float]) -> None:
    """Shift the finger spheres to joint values `values` in every `KinematicsParams` (in place,
    so a captured CUDA graph sees the new spheres)."""
    table = S.finger_sphere_table(spec, values)
    for kp in kparams:
        for link, arr in table.items():
            t = torch.as_tensor(arr, dtype=kp.link_spheres.dtype, device=kp.link_spheres.device)
            kp.update_link_spheres(link, t)


class RobotModel:
    """cuRobo FK for the served sphere model, sphere output in **our** sphere order."""

    def __init__(self, spec: Mapping[str, Any], dev: Optional[DeviceCfg] = None):
        self.spec = spec
        self.dev = dev or device_cfg()
        self.robot_cfg = build_robot_cfg(spec, self.dev)
        self.kin = Kinematics(self.robot_cfg.kinematics)
        self.perm = S.sphere_permutation(spec)  # perm[i_curobo] = i_ours
        self.inv_perm = np.argsort(self.perm)
        if self.kin.joint_names != list(spec["arm_joint_names"]):
            raise ValueError(f"cuRobo joints {self.kin.joint_names}")

    def set_fingers(self, values: Mapping[str, float]) -> None:
        set_finger_openings(kinematics_params_in([self.kin]), self.spec, values)

    def spheres(self, q: np.ndarray) -> np.ndarray:
        """`(T, 14)` arm joints -> `(T, S, 4)` [x, y, z, r] in the `link_torso_5` frame, our order."""
        q_t = torch.as_tensor(np.ascontiguousarray(q), dtype=self.dev.dtype,
                              device=self.dev.device).contiguous()
        js = JointState.from_position(q_t, joint_names=self.kin.joint_names)
        state = self.kin.compute_kinematics(js)
        arr = state.robot_spheres.detach().cpu().numpy().reshape(q_t.shape[0], -1, 4)
        return arr[:, self.inv_perm]


class VoxelWorld:
    """One cuRobo `SceneCollision` holding the three recorded layers (coarse / main / free).

    `load(z, T_base_frame)` copies a record's grids (float16, bit-for-bit) and poses them in the
    solver frame. `signed_distance(...)` queries **one layer at a time** through cuRobo's own
    collision kernel and inverts its activation (`world.sdf_from_activation_cost`).
    """

    def __init__(self, dev: Optional[DeviceCfg] = None, *, shape=(128, 128, 128),
                 voxel_size: float = 0.005, layers: int = len(LAYER_NAMES),
                 max_esdf_distance: float = 10000.0):
        self.dev = dev or device_cfg()
        cache = {"voxel": {"layers": int(layers),
                           "dims": [float(n) * voxel_size for n in shape],
                           "voxel_size": float(voxel_size)}}
        self.scene = SceneCollision.from_config(
            SceneCollisionCfg(device_cfg=self.dev, scene_model=None, cache=cache))
        self.scene.data.voxels.max_esdf_distance = float(max_esdf_distance)
        self.capacity = int(np.prod(shape))
        self.loaded: dict[str, dict[str, Any]] = {}

    @staticmethod
    def voxel_grid(name: str, layer: Mapping[str, Any], T_base_frame: np.ndarray,
                   dev: DeviceCfg) -> VoxelGrid:
        d = np.asarray(layer["distance"])
        vs = float(layer["voxel_size"])
        centre = W.grid_centre(layer["origin"], d.shape, vs)
        pose = W.grid_pose_in_frame(T_base_frame, centre)
        feat = torch.as_tensor(np.ascontiguousarray(d.astype(np.float16)).reshape(-1, 1),
                               device=dev.device)
        return VoxelGrid(name=name, pose=pose, dims=W.grid_dims(d.shape, vs).tolist(),
                         voxel_size=vs, feature_tensor=feat, feature_dtype=torch.float16)

    def load(self, z: Mapping[str, np.ndarray], T_base_frame: np.ndarray,
             names: Iterable[str] = LAYER_NAMES) -> None:
        grids, info = [], {}
        for name in names:
            if f"{W.LAYER_PREFIXES[name]}_distance" not in z:
                continue  # e.g. no target-excluded layer in this chunk
            layer = W.layer_arrays(z, name)
            if int(np.prod(layer["distance"].shape)) > self.capacity:
                raise ValueError(f"layer {name} larger than the voxel cache")
            grids.append(self.voxel_grid(name, layer, T_base_frame, self.dev))
            info[name] = {"origin": layer["origin"], "shape": layer["distance"].shape,
                          "voxel_size": layer["voxel_size"]}
        self.scene.data.voxels.load_batch(grids, env_idx=0)
        self.loaded = info

    def enable_only(self, name: Optional[str]) -> None:
        for n in self.loaded:
            self.scene.enable_obstacle(n, enable=(name is None or n == name))

    def signed_distance(self, spheres_frame: np.ndarray, layer: str, *,
                        eta: float = 20.0) -> np.ndarray:
        """`(N, 4)` spheres in the solver frame -> `(N,)` signed distance of the **centre** to the
        surface of `layer`, from cuRobo's kernel (`+inf` = farther than `r + eta` / outside)."""
        sph = np.asarray(spheres_frame, np.float64).reshape(-1, 4)
        q = torch.as_tensor(sph, dtype=self.dev.dtype, device=self.dev.device).view(1, 1, -1, 4)
        buf = CollisionBuffer.from_shape(q.shape, self.dev)
        w = torch.ones(1, dtype=self.dev.dtype, device=self.dev.device)
        act = torch.full((1,), float(eta), dtype=self.dev.dtype, device=self.dev.device)
        self.enable_only(layer)
        try:
            cost = self.scene.get_sphere_distance_raw(q, buf, w, act)
        finally:
            self.enable_only(None)
        return W.sdf_from_activation_cost(cost.detach().cpu().numpy().reshape(-1),
                                          sph[:, 3], eta)

