"""T80 scene operations for `pi05_infer.py`: a second apple (`apple_2`) and a one-shot target move.

Both are client options that are **off by default** (`--extra-apple`, `--move-target-when-palm-mm`). With them off,
`pi05_infer.py` never imports this module and loads the shipped XML exactly as before.

* `build_model_with_extra_apple(model_xml)` — the shipped MJCF through `mujoco.MjSpec`, plus `apple_2`: a copy of the
  `apple` body (same free joint settings, inertial, every geom with the same mesh / material / friction / contact
  parameters / colour). The shipped XML is not edited. `apple_2` is the last child of the world body, so every id
  and `qpos` / `qvel` address of the original model is unchanged and `apple_2_free` is appended at the end of `qpos`.
  A keyframe that was complete for the shipped model holds `apple_2` at `PARK_POSE`.
* `palm_target_distance(m, d, arm, body)` — distance between the grasping hand's `<arm>_ee` site (the point between
  the finger bases, the "palm site" of the T75 module metrics) and the body's frame origin (the apple centre).
* `TargetMover` — the one-shot move of `--move-target-when-palm-mm D --move-target-dxy DX DY`: at the first control
  step after whose physics the palm–target distance is below D, the target's free joint is moved by (DX, DY) in the
  world xy plane (z and orientation unchanged) and its velocity is set to 0.

The names are fixed here so the evaluation (`AG3S.ag3s.experiments.studies.t80_eval`) builds the same model.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

SOURCE_BODY = "apple"
EXTRA_BODY = "apple_2"
EXTRA_JOINT = "apple_2_free"
#: Where `apple_2` sits in the keyframes (and before a scene reset places it): on the floor, out of every camera —
#: the convention of `--remove-objects` (`[-1.5, -2.5 - 0.3 k, 0.1]`), one row further.
PARK_POSE = (-1.5, -3.7, 0.1, 1.0, 0.0, 0.0, 0.0)

_GEOM_ATTRS = ("type", "size", "pos", "quat", "meshname", "material", "rgba", "friction", "contype", "conaffinity",
               "condim", "priority", "solmix", "solref", "solimp", "margin", "gap", "group", "mass", "density",
               "typeinertia", "fluid_ellipsoid", "fluid_coefs", "fitscale", "fromto", "hfieldname")
_JOINT_ATTRS = ("damping", "armature", "frictionloss", "stiffness", "margin", "group", "solref_limit", "solimp_limit",
                "solref_friction", "solimp_friction", "actgravcomp")
_BODY_ATTRS = ("mass", "ipos", "iquat", "inertia", "fullinertia", "explicitinertial", "gravcomp")


def _qpos_sizes():
    import mujoco

    return {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}


def _copy(dst, src, attrs):
    for a in attrs:
        if hasattr(src, a):
            v = getattr(src, a)
            setattr(dst, a, v.copy() if isinstance(v, np.ndarray) else v)


def build_spec_with_extra_apple(model_xml: str):
    """`MjSpec` of `model_xml` with `apple_2` appended (not compiled)."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(model_xml))
    src = spec.body(SOURCE_BODY)
    if src is None:
        raise ValueError(f"{model_xml} has no body {SOURCE_BODY!r}")
    if spec.body(EXTRA_BODY) is not None:
        raise ValueError(f"{model_xml} already has a body {EXTRA_BODY!r}")
    sizes = _qpos_sizes()
    nq_shipped = sum(sizes.get(j.type, 1) for j in spec.joints)
    body = spec.worldbody.add_body()
    body.name = EXTRA_BODY
    body.pos = np.asarray(PARK_POSE[:3], float)
    body.quat = np.asarray(PARK_POSE[3:], float)
    _copy(body, src, _BODY_ATTRS)
    joints = list(src.joints)
    if len(joints) != 1 or joints[0].type != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"{SOURCE_BODY!r} must have exactly one free joint")
    joint = body.add_freejoint()
    joint.name = EXTRA_JOINT
    _copy(joint, joints[0], _JOINT_ATTRS)
    for g in src.geoms:
        ng = body.add_geom()
        _copy(ng, g, _GEOM_ATTRS)
        ng.name = f"{g.name}_2" if g.name else ""
    # MuJoCo pads a short keyframe with zeros and a unit quaternion (not `qpos0`): `apple_2` would sit at the world
    # origin, inside the robot base. A keyframe that was complete for the shipped model gets `PARK_POSE`; a keyframe
    # that was already short (`stand`, robot only) stays short, like the shipped free bodies it already leaves at 0.
    for key in spec.keys:
        q = list(key.qpos)
        if len(q) == nq_shipped:
            key.qpos = q + list(PARK_POSE)
    return spec


def build_model_with_extra_apple(model_xml: str):
    """Compiled `MjModel` of `model_xml` plus `apple_2` (see the module docstring)."""
    return build_spec_with_extra_apple(model_xml).compile()


def free_qpos_adr(m, joint_name: str) -> tuple[int, int]:
    import mujoco

    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"joint {joint_name!r} missing from the model")
    return int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])


def place_free_body(m, d, joint_name: str, pose: Sequence[float]) -> None:
    """Free joint `joint_name` to `pose` (x y z qw qx qy qz), velocity 0. Caller runs `mj_forward` if needed."""
    qa, va = free_qpos_adr(m, joint_name)
    pose = np.asarray(pose, np.float64)
    if pose.shape != (7,):
        raise ValueError(f"pose must have 7 values, got {pose.shape}")
    d.qpos[qa:qa + 7] = pose
    d.qvel[va:va + 6] = 0.0


def extra_apple_pose(record: dict, xy: Sequence[float]) -> list[float]:
    """`apple_2` initial pose for a scene record: world `xy`, the recorded apple's z and orientation."""
    ap = [float(v) for v in record["initial_fruit_poses"][SOURCE_BODY]]
    return [float(xy[0]), float(xy[1]), ap[2], *ap[3:]]


def palm_target_distance(m, d, arm: str, body: str) -> float:
    """|`<arm>_ee` site − body origin| in metres (world frame, current `d.site_xpos` / `d.xpos`)."""
    import mujoco

    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_ee")
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    if site < 0 or bid < 0:
        raise ValueError(f"site {arm}_ee or body {body!r} missing")
    return float(np.linalg.norm(d.site_xpos[site] - d.xpos[bid]))


class TargetMover:
    """One-shot move of the target's free joint when the grasping palm first comes within `threshold_m`.

    `step(m, d, t_step)` is called once per control step **after** that step's physics. It returns the move record
    the first time it fires (and `None` otherwise); `record` keeps it (`None` until the move).
    """

    def __init__(self, *, arm: str, body: str, joint: str, threshold_m: float, dxy: Sequence[float]):
        if arm not in ("left", "right"):
            raise ValueError(f"arm must be left or right, got {arm!r}")
        if not (np.isfinite(threshold_m) and threshold_m > 0):
            raise ValueError(f"threshold must be > 0, got {threshold_m}")
        self.arm, self.body, self.joint = arm, body, joint
        self.threshold_m = float(threshold_m)
        self.dxy = (float(dxy[0]), float(dxy[1]))
        self.record: Optional[dict] = None
        self.min_distance_m = float("inf")

    @property
    def moved(self) -> bool:
        return self.record is not None

    def step(self, m, d, t_step: int) -> Optional[dict]:
        if self.record is not None:
            return None
        import mujoco

        dist = palm_target_distance(m, d, self.arm, self.body)
        self.min_distance_m = min(self.min_distance_m, dist)
        if dist >= self.threshold_m:
            return None
        qa, _ = free_qpos_adr(m, self.joint)
        before = [float(v) for v in d.qpos[qa:qa + 7]]
        after = list(before)
        after[0] += self.dxy[0]
        after[1] += self.dxy[1]
        place_free_body(m, d, self.joint, after)
        mujoco.mj_forward(m, d)
        self.record = {"t_step": int(t_step), "palm_distance_m": dist, "threshold_m": self.threshold_m,
                       "dxy_m": list(self.dxy), "arm": self.arm, "body": self.body,
                       "qpos_before": before, "qpos_after": after,
                       "sim_time_s": float(d.time)}
        return self.record
