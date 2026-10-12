"""T48b — MuJoCo side of the HSR counterfactual (b): policy-image re-render, state, predicates, edits.

Everything that touches a MuJoCo model lives here; the CLI is `t48b_hsr`, the pure-numpy
metrics are `t48b_metrics`. `mujoco` and `PIL` are imported lazily (set ``MUJOCO_GL=osmesa``
before the first call on a headless machine).

Policy image convention — copied, not imported
----------------------------------------------
`integration/rby1/bringup/pi05_infer.py` imports `mujoco.viewer` and the manipulation package at
module level (`pi05_infer.py:16-58`), so it is not imported here. The pieces used are copied:

* `render_cam(..., match_rby1_dataset=True)` (`pi05_infer.py:217-225`): ``update_scene(data,
  camera)``, reflection flag off (`POLICY_RENDER_REFLECTIONS = False`,
  `rby1_manipulation/data/recording.py:35`), ``render()``, then `resize_policy_image`
  (`recording.py:47-54`, PIL bilinear 299x224 -> 224x224, a pure horizontal squash);
* the renderer size `POLICY_SOURCE_HEIGHT x POLICY_SOURCE_WIDTH` = 224 x 299
  (`recording.py:28-31`, `pi05_infer.py:1293-1300`);
* camera binding cam_high/zed_left, cam_left_wrist/wrist_cam_l, cam_right_wrist/wrist_cam_r
  (`pi05_infer.py:477-479`, `policy_record.CAMERA_BINDINGS`);
* the 16D state of `build_obs("rby1_16d")` (`pi05_infer.py:466-497`): ``[left arm_0..6,
  |q(gripper_finger_l1)|/0.045, right arm_0..6, |q(gripper_finger_r1)|/0.045]``
  (`RBY1_GRIPPER_OPEN = -0.045`, `pi05_infer.py:128`).

**The render pose is one physics step behind the recorded qpos.** The recorder stores `d.qpos`
right after the last `mj_step`, but MuJoCo's `mj_step` computes the kinematics (`xpos`, `xmat`,
`geom_xpos` — what `update_scene` reads) *before* it integrates. The scene the camera saw is
therefore the kinematics of ``q_prev = q - dt * qvel`` (semi-implicit Euler, `opt.integrator` 0,
`dt` 0.002 s); `render_qpos` recovers it with `mj_integratePos(q, qvel, -dt)`. The exception is
the first record of a ``--start-delay 0`` run, which follows the reset's `mj_forward` and no step
(`observation_follows_step`). With that pose the re-render is bit-identical to the recorded RGB
(CPU smoke in `T48b.impl.md`); without it, moving links differ at their edges (MAD up to 0.4,
max 111 gray levels). The **state** is built from the recorded `qpos` itself (that is what
`build_obs` read), so it is reconstructed exactly.

Physical predicates
-------------------
From the contacts of `mj_forward` on the render pose (MJCF body names throughout; the T43 left
finger name swap between MJCF and URDF does not arise because no URDF name is used):

* ``held``      the fruit touches **both** finger bodies of one hand and neither table nor crate;
* ``grasping``  both fingers, but still touching table or crate (closing / setting down);
* ``in_crate``  not held, fruit centre inside the crate's inner volume (crate frame);
* ``on_table``  touches the table (one finger may touch it too);
* ``other``     none of the above (e.g. falling).

`PENETRATION_MAX_M` (2.0 mm) is the largest contact penetration an edit may create — see its
comment for the basis.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Optional, Sequence

import numpy as np

# ----------------------------------------------------------------------------- conventions

#: Policy image key -> MuJoCo camera (`pi05_infer.py:477-479`; `policy_record.CAMERA_BINDINGS`).
POLICY_CAMERAS = {"cam_high": "zed_left", "cam_left_wrist": "wrist_cam_l",
                  "cam_right_wrist": "wrist_cam_r"}
CAMS = tuple(POLICY_CAMERAS)
#: `recording.py:28-31`: render 4:3 at 224 rows, squash to the 224x224 policy tensor.
POLICY_IMAGE_SIZE = 224
POLICY_SOURCE_HEIGHT = POLICY_IMAGE_SIZE
POLICY_SOURCE_WIDTH = round(POLICY_IMAGE_SIZE * 4 / 3)  # 299
#: `recording.py:35`.
POLICY_RENDER_REFLECTIONS = False
#: `pi05_infer.py:128` — the dataset's open gripper ctrl; state gripper = |q(finger_x1)| / 0.045.
RBY1_GRIPPER_OPEN = -0.045
#: `pi05_infer.py:115-122`.
LEFT_ARM_JOINTS = tuple(f"left_arm_{i}" for i in range(7))
RIGHT_ARM_JOINTS = tuple(f"right_arm_{i}" for i in range(7))
#: Hand -> (finger joint read by the state, its coupled twin). `model_transport.xml` equality
#: ``x2 = -x1`` (eq_data polycoef [0, -1]); x1 range [-0.05, 0], x2 [0, 0.05].
FINGER_JOINTS = {"left": ("gripper_finger_l1", "gripper_finger_l2"),
                 "right": ("gripper_finger_r1", "gripper_finger_r2")}
#: Hand -> finger bodies (MJCF names) and the wrist body both fingers hang from.
FINGER_BODIES = {"left": ("ee_finger_l1", "ee_finger_l2"), "right": ("ee_finger_r1", "ee_finger_r2")}
HAND_BODY = {"left": "link_left_arm_6", "right": "link_right_arm_6"}
HANDS = ("left", "right")
#: State gripper dims (`aloha_policy.py:160` layout; `t48_hsr_core.GRIPPER_INDEX`).
GRIPPER_INDEX = {"left": 7, "right": 15}
FRUITS = ("apple", "banana", "orange", "pear")
FRUIT_JOINT = {f: f"{f}_free" for f in FRUITS}
CRATE_JOINT = "crate_free"
#: Free-joint qpos starts reported by the investigation (task) and SUBTASK-b `extract.py:60, 67`;
#: `SceneIndex.from_model` re-reads them from the model and refuses a mismatch.
EXPECTED_FREE_QPOS = {"crate": 31, "apple": 38, "banana": 45, "orange": 52, "pear": 59}
TABLE_BODY = "table"
CRATE_BODY = "crate"
#: Crate inner volume from its collision boxes (`model_transport.xml` crate_floor / crate_wall_*).
CRATE_FLOOR_GEOM = "crate_floor"
CRATE_WALL_GEOMS = ("crate_wall_px", "crate_wall_nx", "crate_wall_py", "crate_wall_ny")

#: Largest contact penetration an edited scene may contain. Basis: over every unedited recorded
#: frame of the seven 16D runs (260 frames, contacts touching any fruit or finger body, render
#: pose) MuJoCo itself produced at most 1.83 mm (ep1807 record 19, the closing frame), p99 0.40 mm.
#: An edit is kept only when it is no less physical than that: <= 2.0 mm (`T48b.impl.md`).
PENETRATION_MAX_M = 0.002
#: Used-gripper value at or above which the hand is open (SUBTASK-b rule threshold 0.9,
#: `t48_hsr_core.RULE_CLOSED_BELOW`).
OPEN_AT_LEAST = 0.9
#: `grasp_latch.LatchConfig.settle_epsilon` (0.02): a closed gripper counts as settled when its
#: value moved less than this since the previous record.
SETTLE_EPS = 0.02


# ----------------------------------------------------------------------------- model / index


def load_record_model(meta: Mapping, *, model_xml: Optional[str] = None):
    """The MuJoCo model of a recorded run, checked like `policy_record.replay_scene`: same `nq` and
    the same joint name order, else `ValueError` (a positional qpos would silently mean another
    scene)."""
    import mujoco

    from AG3S.ag3s.runtime.asset_path import resolve_asset

    path = resolve_asset(model_xml or meta["model_xml"], what="transport model")
    model = mujoco.MjModel.from_xml_path(str(path))
    if int(model.nq) != int(meta["nq"]):
        raise ValueError(f"record nq={meta['nq']} but {path} has nq={model.nq}")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(model.njnt)]
    if list(meta["joint_names"]) != names:
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(meta["joint_names"], names)) if a != b]
        raise ValueError(f"joint order differs between record and {path}: {diff[:5]}")
    return model, path


@dataclasses.dataclass(frozen=True)
class SceneIndex:
    """qpos addresses and body ids, read from the model (never assumed)."""

    left_q: tuple
    right_q: tuple
    finger_q: dict          # hand -> (adr x1, adr x2)
    free_q: dict            # "crate" / fruit -> qpos start of its free joint
    finger_body: dict       # hand -> (body x1, body x2)
    hand_body: dict         # hand -> wrist body id
    fruit_body: dict        # fruit -> body id
    table_body: int
    crate_body: int
    crate_volume: tuple     # (half_x, half_y, z_lo, z_hi) in the crate frame
    timestep: float

    @classmethod
    def from_model(cls, model, *, check_expected: bool = True) -> "SceneIndex":
        import mujoco

        def jadr(name):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise ValueError(f"joint {name!r} missing from the model")
            return int(model.jnt_qposadr[j])

        def bid(name):
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if b < 0:
                raise ValueError(f"body {name!r} missing from the model")
            return int(b)

        free = {"crate": jadr(CRATE_JOINT), **{f: jadr(FRUIT_JOINT[f]) for f in FRUITS}}
        for name in (CRATE_JOINT, *FRUIT_JOINT.values()):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE):
                raise ValueError(f"{name} is not a free joint")
        if check_expected and free != EXPECTED_FREE_QPOS:
            raise ValueError(f"free-joint qpos starts {free} != expected {EXPECTED_FREE_QPOS}")
        return cls(
            left_q=tuple(jadr(j) for j in LEFT_ARM_JOINTS),
            right_q=tuple(jadr(j) for j in RIGHT_ARM_JOINTS),
            finger_q={h: (jadr(a), jadr(b)) for h, (a, b) in FINGER_JOINTS.items()},
            free_q=free,
            finger_body={h: (bid(a), bid(b)) for h, (a, b) in FINGER_BODIES.items()},
            hand_body={h: bid(n) for h, n in HAND_BODY.items()},
            fruit_body={f: bid(f) for f in FRUITS},
            table_body=bid(TABLE_BODY), crate_body=bid(CRATE_BODY),
            crate_volume=crate_inner_volume(model),
            timestep=float(model.opt.timestep),
        )


def crate_inner_volume(model) -> tuple:
    """(half_x, half_y, z_lo, z_hi) of the crate's inside, crate body frame: xy = the floor box's
    half extents minus the wall thickness, z from the floor top to the wall top."""
    import mujoco

    def geom(name):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g < 0:
            raise ValueError(f"geom {name!r} missing")
        return np.asarray(model.geom_pos[g], float), np.asarray(model.geom_size[g], float)

    fp, fs = geom(CRATE_FLOOR_GEOM)
    walls = [geom(n) for n in CRATE_WALL_GEOMS]
    wall_top = max(p[2] + s[2] for p, s in walls)
    thick_x = min(s[0] for p, s in walls if abs(p[0]) > 1e-6)
    thick_y = min(s[1] for p, s in walls if abs(p[1]) > 1e-6)
    return (float(fs[0] - 2 * thick_x), float(fs[1] - 2 * thick_y), float(fp[2] + fs[2]), float(wall_top))


# ----------------------------------------------------------------------------- state / render


def reconstruct_state(idx: SceneIndex, qpos: np.ndarray) -> np.ndarray:
    """`build_obs("rby1_16d")` state (`pi05_infer.py:466-497`) from a qpos vector."""
    q = np.asarray(qpos, np.float64)
    lg = abs(q[idx.finger_q["left"][0]]) / abs(RBY1_GRIPPER_OPEN)
    rg = abs(q[idx.finger_q["right"][0]]) / abs(RBY1_GRIPPER_OPEN)
    s = np.concatenate([q[list(idx.left_q)], [lg], q[list(idx.right_q)], [rg]]).astype(np.float64)
    if s.shape != (16,):
        raise ValueError(f"rby1_16d state must be (16,), got {s.shape}")
    return s


def render_qpos(model, qpos: np.ndarray, qvel: np.ndarray, *, after_step: bool = True) -> np.ndarray:
    """The pose whose kinematics the recorded camera saw (module docstring): one `mj_step` back
    when the observation followed an `mj_step` (`after_step`), else `qpos` itself."""
    import mujoco

    q = np.array(qpos, np.float64, copy=True)
    if after_step:
        mujoco.mj_integratePos(model, q, np.asarray(qvel, np.float64), -float(model.opt.timestep))
    return q


def _argv_value(argv: Sequence[str], flag: str):
    argv = list(argv)
    return argv[argv.index(flag) + 1] if flag in argv else None


def observation_follows_step(meta: Mapping, record_index: int) -> bool:
    """Did an `mj_step` run between the last `mj_forward` and this observation?

    `pi05_infer.py:1235-1237` calls `mj_forward` after the reset. Before the first observation
    `wait_before_inference` (`:1672-1700`) steps only with `--warmup-steps N > 0` or
    `--start-delay > 0` (default 2.0, `:842`). Every later observation follows the chunk's
    `mj_step` (`:2057`). So record 0 of a `--start-delay 0` run (all seven T48b runs) shows `qpos`
    itself; every other record shows the pose one step back."""
    if int(record_index) > 0:
        return True
    argv = meta.get("argv") or []
    warm = _argv_value(argv, "--warmup-steps")
    if warm is not None:
        return int(warm) > 0
    delay = _argv_value(argv, "--start-delay")
    return float(2.0 if delay is None else delay) > 0


def resize_policy_image(image: np.ndarray) -> np.ndarray:
    """`recording.py:47-54`: native 4:3 frame -> 224x224 policy tensor (PIL bilinear)."""
    from PIL import Image

    return np.asarray(Image.fromarray(image).resize((POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE),
                                                    resample=Image.Resampling.BILINEAR))


class PolicyRenderer:
    """`render_cam(..., match_rby1_dataset=True)` for the three policy cameras."""

    def __init__(self, model):
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=POLICY_SOURCE_HEIGHT, width=POLICY_SOURCE_WIDTH)

    def pose(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

    def render(self, qpos: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
        """{policy key: (224, 224, 3) uint8 HWC}. Poses the scene first when `qpos` is given."""
        if qpos is not None:
            self.pose(qpos)
        out = {}
        for key, cam in POLICY_CAMERAS.items():
            self.renderer.update_scene(self.data, camera=cam)
            if not POLICY_RENDER_REFLECTIONS:
                self.renderer.scene.flags[self._mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            out[key] = resize_policy_image(self.renderer.render())
        return out

    def close(self) -> None:
        self.renderer.close()


def image_parity(recorded: Mapping[str, np.ndarray], rendered: Mapping[str, np.ndarray]) -> dict:
    """Per camera: mean absolute difference (gray levels, over H*W*3), max, fraction of differing
    values."""
    out = {}
    for c in CAMS:
        d = np.abs(np.asarray(recorded[c], np.int16) - np.asarray(rendered[c], np.int16))
        out[c] = {"mad": float(d.mean()), "max": int(d.max()), "frac_nonzero": float((d > 0).mean())}
    return out


# ----------------------------------------------------------------------------- contacts / predicates


def body_contacts(model, data, body: int) -> dict[int, float]:
    """{other body id: min signed distance (m, < 0 = penetration)} over the contacts of `body`."""
    out: dict[int, float] = {}
    gb = model.geom_bodyid
    for i in range(int(data.ncon)):
        c = data.contact[i]
        b1, b2 = int(gb[c.geom1]), int(gb[c.geom2])
        if body not in (b1, b2):
            continue
        other = b2 if b1 == body else b1
        out[other] = min(out.get(other, np.inf), float(c.dist))
    return out


def max_penetration(model, data, bodies: Sequence[int]) -> float:
    """Largest penetration depth (m, >= 0) over contacts with at least one geom on `bodies`."""
    bs = set(int(b) for b in bodies)
    gb = model.geom_bodyid
    p = 0.0
    for i in range(int(data.ncon)):
        c = data.contact[i]
        if int(gb[c.geom1]) in bs or int(gb[c.geom2]) in bs:
            p = max(p, -float(c.dist))
    return p


def classify_contacts(touching: set, finger_body: Mapping[str, tuple], table_body: int,
                      crate_body: int, in_crate_volume: bool) -> tuple[str, Optional[str]]:
    """Physical state of a fruit from the set of bodies it touches (module docstring).
    Returns ``(state, hand)``; `hand` is set for ``held`` / ``grasping``."""
    support = (table_body in touching) or (crate_body in touching)
    for h in HANDS:
        if all(b in touching for b in finger_body[h]):
            return ("grasping" if support else "held"), h
    if in_crate_volume:
        return "in_crate", None
    if table_body in touching:
        return "on_table", None
    return "other", None


def point_in_crate(p_world: np.ndarray, crate_pos: np.ndarray, crate_xmat: np.ndarray,
                   volume: tuple) -> bool:
    hx, hy, zlo, zhi = volume
    local = np.asarray(crate_xmat, float).reshape(3, 3).T @ (np.asarray(p_world, float) - np.asarray(crate_pos, float))
    return bool(abs(local[0]) <= hx and abs(local[1]) <= hy and zlo <= local[2] <= zhi)


@dataclasses.dataclass(frozen=True)
class PhysState:
    state: str
    hand: Optional[str]
    touching: tuple            # body names, sorted
    penetration: float         # m, over contacts of the fruit and of every finger body
    fruit_pos: tuple
    finger_contact: dict       # hand -> number of that hand's fingers touching the fruit


def classify(model, data, idx: SceneIndex, fruit: str) -> PhysState:
    """Physical state of `fruit` in the posed `data` (call `mj_forward` first)."""
    import mujoco

    fb = idx.fruit_body[fruit]
    c = body_contacts(model, data, fb)
    touching = set(c)
    inside = point_in_crate(data.xpos[fb], data.xpos[idx.crate_body], data.xmat[idx.crate_body],
                            idx.crate_volume)
    st, hand = classify_contacts(touching, idx.finger_body, idx.table_body, idx.crate_body, inside)
    fingers = [b for h in HANDS for b in idx.finger_body[h]]
    pen = max_penetration(model, data, [fb, *fingers])
    names = tuple(sorted(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or str(b) for b in touching))
    return PhysState(state=st, hand=hand, touching=names, penetration=float(pen),
                     fruit_pos=tuple(float(x) for x in data.xpos[fb]),
                     finger_contact={h: int(sum(b in touching for b in idx.finger_body[h])) for h in HANDS})


#: Physical state -> probe class (pick 0 / place 1 / home 2); ambiguous states -> -1.
PHYS_CLASS = {"on_table": 0, "held": 1, "in_crate": 2, "grasping": -1, "other": -1}


# ----------------------------------------------------------------------------- host frames


def held_rows(phys: Sequence[PhysState], hand: str) -> np.ndarray:
    """Records where the target is `held` by the used hand."""
    return np.asarray([i for i, p in enumerate(phys) if p.state == "held" and p.hand == hand], np.int64)


def reach_rows(phys: Sequence[PhysState], g_used: np.ndarray) -> np.ndarray:
    """Pre-grasp approach records: before the first record whose used-hand gripper is closed
    (< `OPEN_AT_LEAST`, the first grasp attempt), fruit `on_table`, gripper open."""
    g = np.asarray(g_used, float)
    closed = np.flatnonzero(g < OPEN_AT_LEAST)
    first = int(closed[0]) if closed.size else len(g)
    return np.asarray([i for i in range(first) if phys[i].state == "on_table" and g[i] >= OPEN_AT_LEAST],
                      np.int64)


def empty_closure_values(phys: Sequence[PhysState], g_used: np.ndarray, hand: str) -> np.ndarray:
    """Used-hand gripper values of records where the hand is closed on nothing: gripper closed
    (< `OPEN_AT_LEAST`) and settled (|g_t - g_{t-1}| < `SETTLE_EPS`), no finger of that hand on the
    target, target `on_table` (it was not picked up)."""
    g = np.asarray(g_used, float)
    out = []
    for i in range(1, len(g)):
        p = phys[i]
        if (g[i] < OPEN_AT_LEAST and abs(g[i] - g[i - 1]) < SETTLE_EPS
                and p.finger_contact[hand] == 0 and p.state == "on_table"):
            out.append(g[i])
    return np.asarray(out, float)


# ----------------------------------------------------------------------------- edits


def set_fingers(q: np.ndarray, idx: SceneIndex, hand: str, g: float) -> np.ndarray:
    """A copy of `q` with the hand's two finger joints at state value `g` (x1 = -g * 0.045,
    x2 = -x1, the model's equality)."""
    out = np.array(q, np.float64, copy=True)
    a, b = idx.finger_q[hand]
    out[a] = -float(g) * abs(RBY1_GRIPPER_OPEN)
    out[b] = float(g) * abs(RBY1_GRIPPER_OPEN)
    return out


def set_free(q: np.ndarray, idx: SceneIndex, obj: str, pose7: np.ndarray) -> np.ndarray:
    out = np.array(q, np.float64, copy=True)
    a = idx.free_q[obj]
    pose7 = np.asarray(pose7, np.float64)
    if pose7.shape != (7,):
        raise ValueError(f"free pose must be 7 values, got {pose7.shape}")
    out[a:a + 7] = pose7
    out[a + 3:a + 7] /= np.linalg.norm(out[a + 3:a + 7])
    return out


def free_pose(q: np.ndarray, idx: SceneIndex, obj: str) -> np.ndarray:
    a = idx.free_q[obj]
    return np.array(q[a:a + 7], np.float64, copy=True)


def quat_mul(a, b):
    import mujoco

    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def quat_conj(a):
    a = np.asarray(a, float)
    return np.array([a[0], -a[1], -a[2], -a[3]])


def quat_rot(qt, v):
    import mujoco

    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, float), np.asarray(qt, float))
    return out


def relative_pose(hand_pos, hand_quat, obj_pose7) -> np.ndarray:
    """Object pose in the hand frame, 7 values [p, q]."""
    p = quat_rot(quat_conj(hand_quat), np.asarray(obj_pose7[:3]) - np.asarray(hand_pos))
    return np.concatenate([p, quat_mul(quat_conj(hand_quat), obj_pose7[3:])])


def compose_pose(hand_pos, hand_quat, rel7) -> np.ndarray:
    p = np.asarray(hand_pos) + quat_rot(hand_quat, rel7[:3])
    qt = quat_mul(hand_quat, rel7[3:])
    return np.concatenate([p, qt / np.linalg.norm(qt)])


@dataclasses.dataclass
class EditContext:
    """Everything an edit of one run needs besides the frame itself."""

    idx: SceneIndex
    fruit: str
    hand: str
    initial_pose: np.ndarray            # fruit 7D pose at record 0 (settled on the table)
    g_empty: float                      # used-hand value of a hand closed on nothing (recorded)
    g_hold: float                       # used-hand value holding this fruit (dataset train median)
    g_limit: float = 0.0                # joint-limit closure (state 0)
    template_rel: Optional[np.ndarray] = None   # fruit pose in the hand frame from a real held record
    template_src: str = ""
    search_mm: tuple = (-4.0, -2.0, 0.0, 2.0, 4.0)
    pen_max: float = PENETRATION_MAX_M


#: Edit family -> intended physical state of the target after the edit, and its probe class.
INTENDED = {"E1": ("on_table", 0), "E1s": ("on_table", 0), "E1z": ("on_table", 0),
            "E2": ("on_table", 0), "E3": ("held", 1)}
EDIT_FAMILIES = tuple(INTENDED)


@dataclasses.dataclass
class EditResult:
    family: str
    q_render: np.ndarray        # edited render pose (what is rendered and judged)
    q_state: np.ndarray         # edited recorded qpos (what the state is built from)
    state: np.ndarray           # 16D state fed to the model
    phys: Optional[PhysState]
    included: bool
    reason: str                 # "" when included
    info: dict


def _judge(model, data, ctx: EditContext, family: str, q_render, q_state, info) -> EditResult:
    import mujoco

    data.qpos[:] = q_render
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    ph = classify(model, data, ctx.idx, ctx.fruit)
    want, _ = INTENDED[family]
    reason = ""
    if ph.penetration > ctx.pen_max:
        reason = "penetration"
    elif ph.state != want or (want == "held" and ph.hand != ctx.hand):
        reason = f"state_{ph.state}"
    elif want == "on_table" and ph.finger_contact[ctx.hand] > 0:
        reason = "finger_touch"
    return EditResult(family, q_render, q_state, reconstruct_state(ctx.idx, q_state), ph,
                      included=not reason, reason=reason, info=info)


def edit_frame(model, data, ctx: EditContext, family: str, q_rec: np.ndarray,
               q_ren: np.ndarray) -> EditResult:
    """One edit of one host frame. `q_rec` = recorded qpos (state source), `q_ren` = its render
    pose. Only the target fruit's free joint and (E1 / E1z / E2 / E3) the used hand's two finger
    joints change; `mj_forward` only, no simulation.

    * E1   fruit -> its initial table pose; fingers -> `g_empty`
    * E1s  fruit -> its initial table pose; fingers and state untouched
    * E1z  fruit -> its initial table pose; fingers -> joint-limit closure `g_limit` (state 0)
    * E2   fruit -> straight below its held position: current xy, initial (resting) z and
           orientation; fingers -> `g_empty`
    * E3   fruit -> into the hand, fingers closed onto it (`place_in_hand`); state gripper = the
           fitted finger value
    """
    idx, f, h = ctx.idx, ctx.fruit, ctx.hand
    info: dict = {}
    if family in ("E1", "E1s", "E1z"):
        qr = set_free(q_ren, idx, f, ctx.initial_pose)
        qs = np.array(q_rec, np.float64, copy=True)
        if family != "E1s":
            g = ctx.g_empty if family == "E1" else ctx.g_limit
            qr, qs = set_fingers(qr, idx, h, g), set_fingers(qs, idx, h, g)
            info["g"] = float(g)
        return _judge(model, data, ctx, family, qr, qs, info)
    if family == "E2":
        cur = free_pose(q_ren, idx, f)
        pose = np.concatenate([[cur[0], cur[1], ctx.initial_pose[2]], ctx.initial_pose[3:]])
        qr = set_fingers(set_free(q_ren, idx, f, pose), idx, h, ctx.g_empty)
        qs = set_fingers(q_rec, idx, h, ctx.g_empty)
        info.update(g=float(ctx.g_empty), drop_m=float(cur[2] - ctx.initial_pose[2]))
        return _judge(model, data, ctx, family, qr, qs, info)
    if family == "E3":
        qr, g_fit, pinfo = place_in_hand(model, data, ctx, set_fingers(q_ren, idx, h, ctx.g_hold))
        qs = set_fingers(q_rec, idx, h, g_fit)
        info.update(g=float(g_fit), g_hold_reference=float(ctx.g_hold), **pinfo)
        return _judge(model, data, ctx, family, qr, qs, info)
    raise ValueError(f"unknown edit family {family!r}")


def pad_midpoint(model, data, idx: SceneIndex, hand: str) -> np.ndarray:
    """World midpoint of the two fingers' largest collision geoms (the pads)."""
    pts = []
    for b in idx.finger_body[hand]:
        gs = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == b
              and (model.geom_contype[g] or model.geom_conaffinity[g])]
        g = max(gs, key=lambda k: float(np.prod(model.geom_size[k])))
        pts.append(np.asarray(data.geom_xpos[g], float))
    return 0.5 * (pts[0] + pts[1])


def fruit_center_local(model, idx: SceneIndex, fruit: str) -> np.ndarray:
    b = idx.fruit_body[fruit]
    gs = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == b
          and (model.geom_contype[g] or model.geom_conaffinity[g])]
    return np.mean([model.geom_pos[g] for g in gs], axis=0)


def touches_both(model, data, idx: SceneIndex, fruit: str, hand: str, q: np.ndarray) -> bool:
    import mujoco

    data.qpos[:] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    c = body_contacts(model, data, idx.fruit_body[fruit])
    return all(b in c for b in idx.finger_body[hand])


def fit_fingers(model, data, idx: SceneIndex, fruit: str, hand: str, q: np.ndarray, lo: float, hi: float,
                iters: int = 12) -> Optional[float]:
    """Largest finger value g in [lo, hi] at which both fingers of `hand` touch the fruit (the
    fingers closing onto it and stopping at contact). Both-finger contact only grows as g falls
    (the fingers only move inward), so a bisection finds it. None when the fruit is not reached
    at `lo` or is already squeezed at `hi`."""
    if not touches_both(model, data, idx, fruit, hand, set_fingers(q, idx, hand, lo)):
        return None
    if touches_both(model, data, idx, fruit, hand, set_fingers(q, idx, hand, hi)):
        return None
    a, b = float(lo), float(hi)          # P(a) true, P(b) false
    for _ in range(iters):
        m = 0.5 * (a + b)
        if touches_both(model, data, idx, fruit, hand, set_fingers(q, idx, hand, m)):
            a = m
        else:
            b = m
    return a


#: Finger values searched around the fruit's hold value (state units; 0.01 = 0.45 mm per finger).
FIT_BELOW, FIT_ABOVE = 0.15, 0.10
#: Candidates whose fitted value is within this of the best are equally centred (0.005 = 0.23 mm
#: per finger); among them the smallest offset from the start pose wins.
FIT_CENTRED_TOL = 0.005


#: Largest shift (mm, 1 mm steps) along the wrist -> pad axis used to free a fruit from the palm.
PUSH_OUT_MAX_MM = 30


def push_out_of_palm(model, data, idx: SceneIndex, fruit: str, hand: str, q: np.ndarray, pose: np.ndarray,
                     axis: np.ndarray) -> tuple[np.ndarray, float]:
    """Shift `pose` along `axis` (wrist -> pad midpoint, i.e. towards the fingertips) in 1 mm steps
    until the fruit touches no robot body other than the hand's two fingers (fingers opened for the
    test). Returns the shifted pose and the shift in mm (`PUSH_OUT_MAX_MM` when never freed)."""
    import mujoco

    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    scene = {idx.table_body, idx.crate_body, *idx.fruit_body.values(), 0}
    allowed = set(idx.finger_body[hand]) | scene
    qo = set_fingers(q, idx, hand, 1.0)
    for k in range(PUSH_OUT_MAX_MM + 1):
        cand = np.concatenate([pose[:3] + axis * k * 1e-3, pose[3:]])
        data.qpos[:] = set_free(qo, idx, fruit, cand)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if set(body_contacts(model, data, idx.fruit_body[fruit])) <= allowed:
            return cand, float(k)
    return np.concatenate([pose[:3] + axis * PUSH_OUT_MAX_MM * 1e-3, pose[3:]]), float(PUSH_OUT_MAX_MM)


def place_in_hand(model, data, ctx: EditContext, q: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """E3: put the fruit between the used hand's fingers and close the fingers onto it.

    Start pose: the template (fruit pose relative to the wrist in a real `held` record of the same
    fruit and hand) or, without one, the fruit's collision centre at the finger-pad midpoint with its
    current orientation, pushed towards the fingertips until it is clear of the palm
    (`push_out_of_palm`). Around it, offsets `search_mm` along the wrist axes; for each, the fingers
    are closed onto the fruit (`fit_fingers` in [g_hold - 0.15, g_hold + 0.10]). Kept: the fruit is
    `held` by the used hand and penetration <= `pen_max`. Chosen: among the candidates whose fitted
    value is within `FIT_CENTRED_TOL` of the largest (best-centred: both pads reach the fruit at its
    full width) the smallest offset, then the smallest penetration. Returns `(q, g, info)`; without a valid candidate the
    start pose with `g_hold` (then `_judge` excludes the frame)."""
    import mujoco

    idx, f, h = ctx.idx, ctx.fruit, ctx.hand
    data.qpos[:] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    hp = np.array(data.xpos[idx.hand_body[h]], float)
    hq = np.array(data.xquat[idx.hand_body[h]], float)
    rot = np.asarray(data.xmat[idx.hand_body[h]], float).reshape(3, 3)
    if ctx.template_rel is not None:
        start = compose_pose(hp, hq, ctx.template_rel)
        src = f"template:{ctx.template_src}"
    else:
        cur = free_pose(q, idx, f)
        target = pad_midpoint(model, data, idx, h)
        start = np.concatenate([target - quat_rot(cur[3:], fruit_center_local(model, idx, f)), cur[3:]])
        start, push = push_out_of_palm(model, data, idx, f, h, q, start, target - hp)
        src = f"pad_midpoint+{push:.0f}mm"
    lo, hi = max(0.0, ctx.g_hold - FIT_BELOW), ctx.g_hold + FIT_ABOVE
    valid = []
    steps = np.asarray(ctx.search_mm, float) * 1e-3
    for dx in steps:
        for dy in steps:
            for dz in steps:
                pose = np.concatenate([start[:3] + rot @ np.array([dx, dy, dz]), start[3:]])
                qq = set_free(q, idx, f, pose)
                g = fit_fingers(model, data, idx, f, h, qq, lo, hi)
                if g is None:
                    continue
                qf = set_fingers(qq, idx, h, g)
                data.qpos[:] = qf
                mujoco.mj_forward(model, data)
                ph = classify(model, data, idx, f)
                if ph.state != "held" or ph.hand != h or ph.penetration > ctx.pen_max:
                    continue
                valid.append((qf, float(g), float(ph.penetration), [dx * 1e3, dy * 1e3, dz * 1e3]))
    if not valid:
        return (set_fingers(set_free(q, idx, f, start), idx, h, ctx.g_hold), float(ctx.g_hold),
                {"placement": src, "offset_mm": None, "n_valid_candidates": 0})
    g_max = max(v[1] for v in valid)
    near = [v for v in valid if v[1] >= g_max - FIT_CENTRED_TOL]
    qf, g, pen, off = min(near, key=lambda v: (float(np.linalg.norm(v[3])), v[2]))
    return qf, g, {"placement": src, "offset_mm": off, "n_valid_candidates": len(valid), "g_max_candidates": g_max}
