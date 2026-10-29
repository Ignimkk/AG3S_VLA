"""T63 — the served constraint sphere model as a cuRobo v2 robot config (pure Python / numpy).

Input is `robot_spec.json` written by `export_inputs.py` from the **baseline snapshot** (the same
`build_constraint_robot_model` call and flags as the C1aN server). No cuRobo sphere fit is used:
every sphere is copied from the served `UrdfSphereChain` (link, link-frame centre, radius).

How the robot is mapped onto cuRobo (and why):

| our model | cuRobo config | why |
|---|---|---|
| FK root `base`, 6 torso joints read from `q_now`, not optimized | `base_link = link_torso_5`; the world (ESDF grids) is posed in the `link_torso_5` frame per chunk (`world.grid_pose_in_frame`) | the torso varies by up to 0.02 rad between chunks (about 20 mm at the hand); cuRobo bakes locked joints into fixed transforms at build time, so locking the torso would freeze one chunk's torso. Moving the world instead is an exact rigid transform |
| 14 optimized arm joints | c-space `left_arm_0..6, right_arm_0..6` (chunk-column order) | same decision variables as the TO (`ChunkLayout.rby1`) |
| 4 finger joints = FK **parameters** (measured opening, per-step envelope) | `lock_joints` at 0.0; the finger spheres are shifted along the prismatic axis by the chunk's opening at run time (`finger_sphere_table`, `kinematics_params.update_link_spheres`) | cuRobo has no per-step FK parameter; a locked prismatic joint at 0 plus a shifted sphere is exactly the joint at value `v` (`T(v) p = T(0) (p + axis v)`). One value per chunk (step 0 = measured); the drift of the server's envelope over the plan window is reported by the exporter |
| 270 spheres, gripper links only | `collision_spheres` per link, `collision_link_names` in first-appearance order | `sphere_permutation` maps our index to cuRobo's and back |
| position limits from model_xml ± 0.02 rad, velocity / acceleration = URDF x 0.9 | URDF limits are **overridden in memory** after load (`limits_override`) | same robot limits as the TO; recorded as a changed setting |
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Mapping, Optional

import numpy as np

#: cuRobo's default jerk limit (`CSpaceParams.max_jerk`); our TO has no jerk limit.
CUROBO_DEFAULT_MAX_JERK = 500.0


def load_spec(path: str | pathlib.Path) -> dict[str, Any]:
    spec = json.loads(pathlib.Path(path).read_text())
    if int(spec["n_spheres"]) != len(spec["spheres"]):
        raise ValueError("robot_spec: n_spheres disagrees with the sphere list")
    return spec


def collision_link_order(spec: Mapping[str, Any]) -> list[str]:
    """Links in the order they first appear in our sphere list."""
    seen: list[str] = []
    for s in spec["spheres"]:
        if s["link"] not in seen:
            seen.append(s["link"])
    return seen


def sphere_permutation(spec: Mapping[str, Any]) -> np.ndarray:
    """`perm[i_curobo] = i_ours`. cuRobo orders spheres by `collision_link_names`, then by the
    per-link list order (`kinematics_loader._build_collision_model`)."""
    out: list[int] = []
    for link in collision_link_order(spec):
        out.extend(int(s["index"]) for s in spec["spheres"] if s["link"] == link)
    perm = np.asarray(out, np.int64)
    if sorted(perm.tolist()) != list(range(len(spec["spheres"]))):
        raise ValueError("sphere permutation is not a permutation")
    return perm


def finger_lock_joints(spec: Mapping[str, Any]) -> dict[str, float]:
    return {name: 0.0 for name in spec["finger_joints"]}


def robot_cfg_dict(spec: Mapping[str, Any], *, collision_sphere_buffer: float = 0.0
                   ) -> dict[str, Any]:
    """cuRobo v2 `robot_cfg` dict. Spheres are at finger value 0 (`finger_sphere_table` shifts them)."""
    links = collision_link_order(spec)
    spheres: dict[str, list[dict[str, Any]]] = {link: [] for link in links}
    for s in spec["spheres"]:
        spheres[s["link"]].append({"center": [float(v) for v in s["center"]],
                                   "radius": float(s["radius"])})
    arm = list(spec["arm_joint_names"])
    lim = spec["limits"]
    if list(lim["joint_names"]) != arm:
        raise ValueError("limits are not in the c-space order")
    kin = {
        "format_version": 2.0,
        "urdf_path": str(spec["urdf_path"]),
        "asset_root_path": str(pathlib.Path(spec["urdf_path"]).parent),
        "base_link": str(spec["curobo_base_link"]),
        "tool_frames": [link for link in ("ee_left", "ee_right") if link in links] or [links[0]],
        "collision_link_names": links,
        "collision_spheres": spheres,
        "collision_sphere_buffer": float(collision_sphere_buffer),
        "self_collision_ignore": {},
        "self_collision_buffer": {},
        "lock_joints": finger_lock_joints(spec),
        "load_meshes": False,
        "cspace": {
            "joint_names": arm,
            "default_joint_position": [0.0] * len(arm),
            "cspace_distance_weight": [1.0] * len(arm),
            "null_space_weight": [1.0] * len(arm),
            "max_acceleration": [float(v) for v in lim["acceleration"]],
            "max_jerk": [CUROBO_DEFAULT_MAX_JERK] * len(arm),
        },
    }
    return {"robot_cfg": {"kinematics": kin}}


def limits_override(spec: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The TO's effective limits in c-space order — position (2, n), velocity (2, n), acceleration."""
    lim = spec["limits"]
    lo, hi = np.asarray(lim["lower"], np.float64), np.asarray(lim["upper"], np.float64)
    v = np.asarray(lim["velocity"], np.float64)
    a = np.asarray(lim["acceleration"], np.float64)
    return {"position": np.stack([lo, hi]), "velocity": np.stack([-v, v]),
            "acceleration": np.stack([-a, a])}


def finger_values(spec: Mapping[str, Any], path: np.ndarray, step: int = 0) -> dict[str, float]:
    """`{finger joint: value}` from an exporter finger path `(4, H)` (order = `finger_joints` keys)."""
    names = list(spec["finger_joints"])
    P = np.asarray(path, np.float64)
    if P.shape[0] != len(names):
        raise ValueError(f"finger path has {P.shape[0]} rows for {len(names)} finger joints")
    return {n: float(P[i, step]) for i, n in enumerate(names)}


def finger_sphere_table(spec: Mapping[str, Any], values: Mapping[str, float]
                        ) -> dict[str, np.ndarray]:
    """`{finger link: (n, 4) [x, y, z, r]}` in the finger link frame **at joint value 0**, for a
    prismatic joint at `values[joint]`. `T_parent_child(v) = T_origin · Trans(axis v)`, so a point
    `p` of the child at `v` is the point `p + axis v` of the child frame at 0."""
    out: dict[str, np.ndarray] = {}
    for joint, info in spec["finger_joints"].items():
        if info["type"] != "prismatic":
            raise ValueError(f"{joint}: only prismatic fingers are handled, got {info['type']}")
        v = float(values.get(joint, 0.0))
        lo, hi = info.get("lower"), info.get("upper")
        if lo is not None and hi is not None and not (lo - 1e-9 <= v <= hi + 1e-9):
            raise ValueError(f"{joint}={v} outside [{lo}, {hi}]")
        axis = np.asarray(info["axis"], np.float64)
        rows = [np.r_[np.asarray(s["center"], np.float64) + axis * v, float(s["radius"])]
                for s in spec["spheres"] if s["link"] == info["child"]]
        out[str(info["child"])] = np.asarray(rows, np.float64).reshape(-1, 4)
    return out


def chunk_to_cspace(chunk: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    """`(T, 16)` policy chunk -> `(T, 14)` arm joints in c-space order."""
    cols = [*range(0, 7), *range(8, 15)]
    c = np.asarray(chunk, np.float64)
    if c.ndim != 2 or c.shape[1] < 16:
        raise ValueError(f"chunk must be (T, 16), got {c.shape}")
    return c[:, cols]


def cspace_to_chunk(q: np.ndarray, gripper_from: np.ndarray) -> np.ndarray:
    """`(T, 14)` arm joints + the gripper columns of `gripper_from (T, 16)` -> `(T, 16)` chunk."""
    q = np.asarray(q, np.float64)
    out = np.array(gripper_from, np.float64, copy=True)
    out[:, 0:7] = q[:, 0:7]
    out[:, 8:15] = q[:, 7:14]
    return out


def q_now_cspace(q_now20: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    """`DEFAULT_RBY1_JOINTS` (20) -> 14 arm joints in c-space order."""
    names = list(spec["model_joint_names"])
    idx = [names.index(n) for n in spec["arm_joint_names"]]
    return np.asarray(q_now20, np.float64)[idx]



def knot_seed(q_now: np.ndarray, ref_cspace: np.ndarray, n_knots: int) -> np.ndarray:
    """Reference path (q_now at t=0, ref[k-1] at t=k/15) sampled at cuRobo's linear-seed knot
    fractions `k / (n_knots - 1)` of the span -> `(n_knots, dof)`."""
    path = np.vstack([np.asarray(q_now, np.float64)[None], np.asarray(ref_cspace, np.float64)])
    t_path = np.arange(path.shape[0]) / (path.shape[0] - 1.0)
    t_knot = np.arange(n_knots) / (n_knots - 1.0)
    return np.stack([np.interp(t_knot, t_path, path[:, j]) for j in range(path.shape[1])], axis=1)


def sample_chunk_grid(pos: np.ndarray, state_dt: float, chunk_dt: float,
                      steps: int) -> Optional[np.ndarray]:
    """`(H, dof)` states at `i * state_dt` (state 0 = start) -> `(steps, dof)` at `k * chunk_dt`,
    k = 1..steps. Exact pick when the grids align, linear interpolation otherwise."""
    t_state = np.arange(pos.shape[0]) * state_dt
    t_want = np.arange(1, steps + 1) * chunk_dt
    if t_want[-1] > t_state[-1] + 1e-6:
        return None
    ratio = chunk_dt / state_dt
    if abs(ratio - round(ratio)) < 1e-6:
        idx = np.rint(t_want / state_dt).astype(int)
        return pos[idx]
    return np.stack([np.interp(t_want, t_state, pos[:, j]) for j in range(pos.shape[1])], axis=1)


def chunk_row_weights(horizon: int, rows: int, chunk_dt: float, span_s: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """`(W (rows, horizon), owner (rows,))` — chunk row `k` (time `(k+1)·chunk_dt`) as a linear
    combination of optimizer states `i` (time `i·span_s/(horizon-1)`, state 0 = start).

    `W @ pos` equals `sample_chunk_grid(pos, span_s/(horizon-1), chunk_dt, rows)` — an exact pick
    when the grids align (timing `exact`: one weight 1 per row), linear interpolation otherwise
    (`resample16`). `owner[k]` is the state the row's cost is reported at (the nearest state; the
    total cost does not depend on it)."""
    state_dt = float(span_s) / (int(horizon) - 1)
    t = np.arange(1, int(rows) + 1) * float(chunk_dt)
    if t[-1] > float(span_s) + 1e-9:
        raise ValueError(f"{rows} rows at {chunk_dt} s exceed the {span_s} s state span")
    W = np.zeros((int(rows), int(horizon)))
    x = t / state_dt
    lo = np.floor(x + 1e-9).astype(int)
    frac = x - lo
    for k in range(int(rows)):
        if frac[k] < 1e-9 or lo[k] >= horizon - 1:
            W[k, min(lo[k], horizon - 1)] = 1.0
        else:
            W[k, lo[k]] = 1.0 - frac[k]
            W[k, lo[k] + 1] = frac[k]
    owner = np.clip(np.rint(x).astype(int), 0, horizon - 1)
    return W, owner


__all__ = [
    "CUROBO_DEFAULT_MAX_JERK", "chunk_row_weights", "chunk_to_cspace", "collision_link_order",
    "cspace_to_chunk",
    "finger_lock_joints", "finger_sphere_table", "finger_values", "knot_seed", "limits_override",
    "load_spec",
    "q_now_cspace", "robot_cfg_dict", "sample_chunk_grid", "sphere_permutation",
]
