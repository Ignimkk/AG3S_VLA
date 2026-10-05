"""T42 — real-time "avoidance arrow" overlay (visual only).

What the arrow means
--------------------
At each control step, the arrow starts at the active arm's TCP (site ``left_ee`` / ``right_ee``,
100 mm below ``EE_BODY_*``, between the fingers) and points along

    v = TCP(q_to) - TCP(q_reference)

* ``q_reference``: the **policy reference** action row of this step (``actions_reference``).
* ``q_to``: the **refined** row of this step (``actions``) when the chunk executes, and the
  applied HOLD target (``q_hold``) when it holds.

Both TCPs come from forward kinematics on the **same** ``MjModel``. The commanded arm joint
targets go on top of the current ``qpos`` (torso, base, free bodies), in a private scratch
``MjData``. ``v`` is therefore "how far, and in which direction, the TO pushed the gripper
away from what the policy wanted", in world metres.

Drawn length = ``scale * |v|`` (default ``scale`` = 8, so a 10 mm deflection is an 80 mm
arrow), capped at ``MAX_DRAWN_LENGTH_M`` = 0.40 m (the caption says "clipped"). Nothing is drawn
when ``|v| <= threshold`` (default 5 mm). In the T41a E3b hurdle runs the per-step magnitude
has a median of 2–5 mm, a p90 of 10–23 mm and a maximum of 91 mm.

Colours (gate state of the step):

| state | drawn |
|---|---|
| execute | green arrow (``v`` = refined - reference) |
| HOLD, reference row known | red arrow (``v`` = q_hold - reference), red sphere marker if ``|v| <= threshold`` |
| HOLD, no reference (comms: timeout / error / no response) | red sphere marker only |
| no refined/reference pair (E0 ``--no-safe``, in-process ``--trajopt``) | nothing |

Optional (off by default): a blue arrow along the obstacle normal at the closest point of the
active gripper. It uses MuJoCo's own ``mj_geomDistance`` between the active obstacle collision
geoms and the gripper collision geoms, on a scratch copy of the state. A witness pair whose
length differs from the distance by more than 0.1 mm is skipped (the T40 H false-zero case), and
so is any pair in contact. This is **simulator ground truth**, not what the TO saw.

Why it is purely visual
-----------------------
* FK and distance queries run on scratch ``MjData`` objects owned by this module. The run's ``d``
  is only **read** (``qpos``, ``mocap_*``, ``site_xpos``).
* Geoms go into an ``MjvScene`` after ``update_scene`` (recording) or into the viewer's
  ``user_scn``. Neither is read by physics, by the records, or by the server request.
* No RNG is used. Nothing is sent anywhere.

``tests/sim/test_t42_avoidance_arrow.py`` pins these properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import mujoco
import numpy as np

#: execute — the refined chunk runs.
GREEN = (0.10, 0.85, 0.20, 1.0)
#: HOLD — the chunk does not run.
RED = (0.95, 0.10, 0.10, 1.0)
#: optional obstacle normal at the closest gripper point.
BLUE = (0.15, 0.45, 1.00, 1.0)

DEFAULT_THRESHOLD_M = 0.005
DEFAULT_SCALE = 8.0
#: drawn length cap (m). Longer arrows are shortened to this along the same direction; the
#: caption then says "clipped". 0.40 m = a 50 mm deflection at the default scale.
MAX_DRAWN_LENGTH_M = 0.40
ARROW_WIDTH_M = 0.012
MARKER_RADIUS_M = 0.015
NORMAL_RANGE_M = 0.10
NORMAL_LENGTH_M = 0.08
#: witness-length vs distance tolerance for the obstacle normal (T40 H false zero).
NORMAL_WITNESS_TOL_M = 1e-4

ARM_JOINTS = {
    "left": tuple(f"left_arm_{i}" for i in range(7)),
    "right": tuple(f"right_arm_{i}" for i in range(7)),
}
TCP_SITES = {"left": "left_ee", "right": "right_ee"}
GRIPPER_BODIES = {
    "left": ("EE_BODY_L", "ee_finger_l1", "ee_finger_l2"),
    "right": ("EE_BODY_R", "ee_finger_r1", "ee_finger_r2"),
}
SIDES = ("left", "right")


def arm_targets(action_format: str, row, *, arm6_hold: Optional[dict] = None) -> dict:
    """Commanded arm joint targets of one action row, ``{"left": (7,), "right": (7,)}``.

    The mapping is the one ``pi05_infer.apply_action`` uses for ``d.ctrl``:

    * ``rby1_16d`` — ``[L 7, L grip, R 7, R grip]``;
    * ``rby1`` (14-D) — ``[L 6, L grip, R 6, R grip]`` + the held ``arm_6`` target (``arm6_hold``).
    """
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    if action_format == "rby1_16d":
        if row.shape[0] < 16:
            raise ValueError(f"rby1_16d row needs 16 values, got {row.shape[0]}")
        return {"left": row[0:7].copy(), "right": row[8:15].copy()}
    if action_format == "rby1":
        if row.shape[0] < 14:
            raise ValueError(f"rby1 row needs 14 values, got {row.shape[0]}")
        if arm6_hold is None:
            raise ValueError("rby1 (14-D) rows need arm6_hold {'left': ., 'right': .}")
        return {"left": np.r_[row[0:6], float(arm6_hold["left"])],
                "right": np.r_[row[7:13], float(arm6_hold["right"])]}
    raise ValueError(f"avoidance arrow supports rby1 / rby1_16d actions, not {action_format!r}")


class TcpKinematics:
    """TCP position for a commanded arm target, by FK on a private scratch ``MjData``."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self._scratch = mujoco.MjData(model)
        self.qpos_adr = {}
        self.site_id = {}
        for side in SIDES:
            adr = []
            for name in ARM_JOINTS[side]:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    raise ValueError(f"joint {name!r} missing from the model")
                adr.append(int(model.jnt_qposadr[jid]))
            self.qpos_adr[side] = np.asarray(adr, dtype=np.int64)
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITES[side])
            if sid < 0:
                raise ValueError(f"site {TCP_SITES[side]!r} missing from the model")
            self.site_id[side] = int(sid)

    def tcp(self, base_qpos, side: str, q_arm) -> np.ndarray:
        """World TCP of ``side`` with its 7 arm joints at ``q_arm`` and every other DoF at ``base_qpos``."""
        s = self._scratch
        s.qpos[:] = base_qpos
        s.qpos[self.qpos_adr[side]] = np.asarray(q_arm, dtype=np.float64)[:7]
        mujoco.mj_kinematics(self.model, s)
        return s.site_xpos[self.site_id[side]].copy()

    def displacement(self, base_qpos, side: str, q_from, q_to) -> np.ndarray:
        """``TCP(q_to) - TCP(q_from)``, both at the same ``base_qpos``."""
        return self.tcp(base_qpos, side, q_to) - self.tcp(base_qpos, side, q_from)


@dataclass(frozen=True)
class DrawItem:
    """One overlay geom. ``kind`` is ``arrow`` or ``marker``."""

    kind: str
    start: np.ndarray
    end: np.ndarray
    rgba: tuple
    width: float = ARROW_WIDTH_M
    role: str = "avoidance"          # avoidance | obstacle_normal


@dataclass
class StepOverlay:
    """What one control step draws, plus the numbers behind it (for captions and tests)."""

    state: str                                   # execute | hold | none
    side: Optional[str] = None
    vector_m: Optional[np.ndarray] = None        # TCP(q_to) - TCP(q_reference), world
    magnitude_m: Optional[float] = None
    drawn: bool = False                          # arrow above the threshold
    clipped: bool = False                        # drawn length hit max_length_m
    items: list = field(default_factory=list)
    normal: Optional[dict] = None                # {point, normal, distance_m} when computed

    def caption(self, scale: float, threshold_m: float) -> list[str]:
        if self.state == "none":
            return []
        side = f"{self.side} " if self.side else ""
        head = f"TO deflection {side}TCP"
        if self.magnitude_m is None:
            body = "HOLD (no reference row) - marker"
        else:
            what = "refined - reference" if self.state == "execute" else "q_hold - reference"
            mm = self.magnitude_m * 1000.0
            shown = ("" if self.drawn else f"  (<= {threshold_m * 1000:.0f} mm: not drawn)")
            shown += "  (arrow clipped)" if self.clipped else ""
            body = f"{self.state.upper()}  |{what}| = {mm:5.1f} mm{shown}"
        lines = [f"{head}: {body}", f"arrow length = {scale:g} x |v|"]
        if self.normal is not None:
            lines.append(f"obstacle normal (blue): {self.normal['distance_m'] * 1000:.1f} mm")
        return lines


class AvoidanceArrows:
    """Per-step overlay builder. Reads ``data`` only; owns its scratch ``MjData``."""

    def __init__(self, model: mujoco.MjModel, *, action_format: str,
                 arms: Sequence[str] = SIDES, threshold_m: float = DEFAULT_THRESHOLD_M,
                 scale: float = DEFAULT_SCALE, arm6_hold: Optional[dict] = None,
                 obstacle_geom_ids: Optional[Iterable[int]] = None,
                 obstacle_normal: bool = False, normal_range_m: float = NORMAL_RANGE_M,
                 normal_length_m: float = NORMAL_LENGTH_M,
                 max_length_m: float = MAX_DRAWN_LENGTH_M,
                 arrow_width_m: float = ARROW_WIDTH_M, marker_radius_m: float = MARKER_RADIUS_M):
        arms = tuple(arms)
        if not arms or any(a not in SIDES for a in arms):
            raise ValueError(f"arms must be a non-empty subset of {SIDES}, got {arms}")
        if threshold_m < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold_m}")
        if not scale > 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        self.model = model
        self.action_format = action_format
        self.arms = arms
        self.threshold_m = float(threshold_m)
        self.scale = float(scale)
        self.max_length_m = float(max_length_m)
        self.arm6_hold = arm6_hold
        self.arrow_width_m = float(arrow_width_m)
        self.marker_radius_m = float(marker_radius_m)
        self.kin = TcpKinematics(model)
        self.obstacle_normal = bool(obstacle_normal)
        self.normal_range_m = float(normal_range_m)
        self.normal_length_m = float(normal_length_m)
        self.obstacle_geom_ids = tuple(sorted(int(g) for g in (obstacle_geom_ids or ())))
        self.gripper_geom_ids = {side: self._gripper_geoms(side) for side in SIDES}
        self._normal_scratch = mujoco.MjData(model) if self.obstacle_normal else None

    # ------------------------------------------------------------------ geometry
    def _gripper_geoms(self, side: str) -> tuple:
        bodies = set()
        for name in GRIPPER_BODIES[side]:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                bodies.add(bid)
        m = self.model
        return tuple(g for g in range(m.ngeom)
                     if int(m.geom_bodyid[g]) in bodies
                     and (m.geom_contype[g] or m.geom_conaffinity[g]))

    def _targets(self, row) -> dict:
        return arm_targets(self.action_format, row, arm6_hold=self.arm6_hold)

    def anchor(self, data, side: str) -> np.ndarray:
        """Arrow start: the TCP as rendered (``data.site_xpos``, read only)."""
        return np.asarray(data.site_xpos[self.kin.site_id[side]], dtype=np.float64).copy()

    def closest_obstacle_normal(self, data, side: str) -> Optional[dict]:
        """Obstacle normal at the gripper's closest point, or ``None`` (see the module docstring)."""
        if not self.obstacle_geom_ids or self._normal_scratch is None:
            return None
        s = self._normal_scratch
        s.qpos[:] = data.qpos
        if self.model.nmocap:
            s.mocap_pos[:] = data.mocap_pos
            s.mocap_quat[:] = data.mocap_quat
        mujoco.mj_kinematics(self.model, s)
        best = None
        fromto = np.zeros(6, dtype=np.float64)
        for go in self.obstacle_geom_ids:
            for gg in self.gripper_geom_ids[side]:
                fromto[:] = 0.0
                dist = float(mujoco.mj_geomDistance(self.model, s, go, gg,
                                                    self.normal_range_m, fromto))
                if not (0.0 < dist < self.normal_range_m):
                    continue
                p_obs, p_grip = fromto[:3].copy(), fromto[3:].copy()
                gap = p_grip - p_obs
                if abs(float(np.linalg.norm(gap)) - dist) > NORMAL_WITNESS_TOL_M:
                    continue
                if best is None or dist < best["distance_m"]:
                    best = {"point": p_grip, "normal": gap / np.linalg.norm(gap),
                            "distance_m": dist, "obstacle_geom": go, "gripper_geom": gg}
        return best

    # ------------------------------------------------------------------ per step
    def step(self, data, *, executed: bool, refined_row=None, reference_row=None,
             applied_row=None, applied_arm=None) -> StepOverlay:
        """Overlay for one control step.

        ``executed``: the gate let this step's chunk run (``SafeRemoteClient.should_execute``).
        ``refined_row`` / ``reference_row``: this step's row of ``actions`` / ``actions_reference``
        (``None`` = not available). ``applied_row``: the action row given to ``apply_action``
        (on HOLD, the hold target). ``applied_arm`` (alternative to ``applied_row``):
        ``{"left": (7,), "right": (7,)}`` joint targets already split by arm (replay from
        ``applied_ctrl``).
        """
        state = "execute" if executed else "hold"
        base = np.asarray(data.qpos, dtype=np.float64)
        if executed:
            to_targets = None if refined_row is None else self._targets(refined_row)
        elif applied_arm is not None:
            to_targets = {k: np.asarray(v, dtype=np.float64) for k, v in applied_arm.items()}
        else:
            to_targets = None if applied_row is None else self._targets(applied_row)
        ref_targets = None if reference_row is None else self._targets(reference_row)

        if to_targets is None or ref_targets is None:
            if executed:
                # E0-like: no pair to compare. Nothing to draw.
                return StepOverlay(state="none")
            side = self.arms[0]
            out = StepOverlay(state="hold", side=side)
            out.items.append(self._marker(self.anchor(data, side)))
            self._add_normal(out, data, side)
            return out

        vectors = {side: self.kin.displacement(base, side, ref_targets[side], to_targets[side])
                   for side in self.arms}
        side = max(self.arms, key=lambda a: float(np.linalg.norm(vectors[a])))
        v = vectors[side]
        mag = float(np.linalg.norm(v))
        out = StepOverlay(state=state, side=side, vector_m=v, magnitude_m=mag)
        start = self.anchor(data, side)
        if mag > self.threshold_m:
            out.drawn = True
            drawn = self.scale * v
            length = float(np.linalg.norm(drawn))
            if length > self.max_length_m:
                drawn = drawn * (self.max_length_m / length)
                out.clipped = True
            out.items.append(DrawItem("arrow", start, start + drawn,
                                      GREEN if executed else RED, self.arrow_width_m))
        elif not executed:
            out.items.append(self._marker(start))
        self._add_normal(out, data, side)
        return out

    def _marker(self, at) -> DrawItem:
        at = np.asarray(at, dtype=np.float64)
        return DrawItem("marker", at, at.copy(), RED, self.marker_radius_m)

    def _add_normal(self, out: StepOverlay, data, side: str) -> None:
        if not self.obstacle_normal:
            return
        n = self.closest_obstacle_normal(data, side)
        out.normal = n
        if n is not None:
            out.items.append(DrawItem("arrow", n["point"],
                                      n["point"] + self.normal_length_m * n["normal"],
                                      BLUE, self.arrow_width_m * 0.75, role="obstacle_normal"))


def draw(scene, items: Sequence[DrawItem]) -> int:
    """Append ``items`` to an ``MjvScene`` as decor geoms; returns how many were added.

    Call after ``update_scene`` (``mujoco.Renderer``) or on a viewer's ``user_scn`` (which the
    caller resets to ``ngeom = 0`` first). Geoms beyond ``maxgeom`` are dropped silently, as
    MuJoCo itself does.
    """
    added = 0
    eye = np.eye(3).ravel()
    for item in items:
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        rgba = np.asarray(item.rgba, dtype=np.float32)
        if item.kind == "arrow":
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), eye, rgba)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, float(item.width),
                                 np.asarray(item.start, dtype=np.float64),
                                 np.asarray(item.end, dtype=np.float64))
        elif item.kind == "marker":
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([item.width, 0.0, 0.0]),
                                np.asarray(item.start, dtype=np.float64), eye, rgba)
        else:
            raise ValueError(f"unknown draw item kind {item.kind!r}")
        g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        g.emission = 0.6
        scene.ngeom += 1
        added += 1
    return added


def draw_user_scene(viewer_handle, items: Sequence[DrawItem]) -> None:
    """Replace the passive viewer's ``user_scn`` overlay with ``items``."""
    with viewer_handle.lock():
        viewer_handle.user_scn.ngeom = 0
        draw(viewer_handle.user_scn, items)


def caption(frame: np.ndarray, lines: Sequence[str], rgba=None) -> np.ndarray:
    """A copy of ``frame`` with ``lines`` written top-left (PIL). ``frame`` is not modified."""
    if not lines:
        return frame
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.asarray(frame))
    drw = ImageDraw.Draw(img)
    colour = tuple(int(255 * c) for c in (rgba or (1, 1, 1, 1))[:3])
    y = 6
    for i, line in enumerate(lines):
        drw.rectangle([4, y - 1, 8 + 6 * len(line), y + 11], fill=(0, 0, 0))
        drw.text((6, y), line, fill=colour if i == 0 else (235, 235, 235))
        y += 14
    return np.asarray(img)


def overlay_colour(overlay: StepOverlay):
    return GREEN if overlay.state == "execute" else RED
