"""Static obstacle profiles and inference-time safety for pick-and-place.

Adding and removing obstacles (T40 E)
-------------------------------------
No obstacle is ``--obstacle-profile clear`` (the default).  ``pi05_infer.py`` then
loads the plain scene XML (``model.xml`` / ``model_transport.xml``) and never
constructs a :class:`PickPlaceObstacleManager`, so the run is exactly the scene
without obstacle slots.

Adding obstacles is ``--obstacle-config <json> --obstacle-profile <name>``; the
client then loads the ``*_pick_place_obstacles.xml`` variant, which carries every
slot parked far away (mocap bodies: no qpos/qvel/ctrl, same qpos addresses), and
:meth:`PickPlaceObstacleManager.activate` moves the profile's slots onto the table.
A per-run config is a JSON object::

    {
      "version": 1,
      "table_surface_z": 0.82,
      "profiles": {
        "clear": {"scene": "any", "obstacles": []},
        "ep1807_hurdle": {
          "scene": "fruit",
          "obstacles": [
            {"slot": "hurdle_0", "position": [0.55, 0.22, 0.82], "yaw": 0.0,
             "bar_height": 0.26}
          ]
        }
      }
    }

* ``position`` is the prop's footprint centre on the table, ``[x, y, table_z]``
  (world frame, x in [0.38, 0.85], y in [-0.35, 0.35], z within 0.02 of
  ``table_surface_z``).
* ``yaw`` (rad) rotates the prop about world z.  For ``hurdle_0`` the bar lies
  along the prop's local x axis, so ``yaw = 0`` puts the bar along world x.
* ``bar_height`` (``hurdle_*`` slots only, metres, default 0.26) is the height
  of the bar *centre* above ``position[2]``, in [0.20, 0.32].  Any other slot
  rejects the key.

Slots: ``bollard_0``/``bollard_1`` (cylinder posts), ``divider_0`` (boxes;
MuJoCo 3.11 ``mj_geomDistance`` is wrong for some box pairs, T40 A) and
``hurdle_0`` (capsule bar on two telescoping posts, capsules/cylinders only).
The hurdle is two mocap bodies, ``pick_obstacle_hurdle_0`` (feet + sleeves on
the table) and ``pick_obstacle_hurdle_0_bar`` (bar + sliding posts, placed at
``position + [0, 0, bar_height]``): the height is set with mocap positions only,
so no compiled model quantity (geom size, bounding sphere, AABB, BVH) changes.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from rby1_manipulation.paths import PICK_PLACE_OBSTACLE_CONFIG
from rby1_manipulation.simulation.convex_distance import GeomSupport, certify, gjk_distance
DEFAULT_CONFIG = PICK_PLACE_OBSTACLE_CONFIG
SLOTS = ("bollard_0", "bollard_1", "divider_0", "hurdle_0")
HURDLE_SLOTS = ("hurdle_0",)
HURDLE_BAR_HEIGHT_RANGE = (0.20, 0.32)
HURDLE_DEFAULT_BAR_HEIGHT = 0.26
# Mocap bodies per slot.  The first body sits at the profile position; a hurdle's
# second body (the bar) sits bar_height above it.
SLOT_BODIES = {
    slot: (f"pick_obstacle_{slot}",) + ((f"pick_obstacle_{slot}_bar",) if slot in HURDLE_SLOTS else ())
    for slot in SLOTS
}
MOVABLE_OBJECT_BODIES = (
    "red_block", "green_block", "blue_block",
    "apple", "banana", "orange", "pear", "crate",
)
PARKING_POSITIONS = {
    "bollard_0": np.array([8.0, 8.0, 2.0]),
    "bollard_1": np.array([9.0, 8.0, 2.0]),
    "divider_0": np.array([10.0, 8.0, 2.0]),
    "hurdle_0": np.array([11.0, 8.0, 2.0]),
}


def validate_pick_place_obstacle_config(raw: dict) -> dict:
    """Normalize a profile config and reject unsafe or ambiguous placements."""
    if not isinstance(raw, dict):
        raise ValueError("pick-place obstacle config must be a JSON object")
    config = copy.deepcopy(raw)
    if int(config.get("version", 1)) != 1:
        raise ValueError(f"unsupported obstacle config version {config.get('version')!r}")
    table_z = float(config.get("table_surface_z", 0.82))
    if not np.isfinite(table_z) or not 0.75 <= table_z <= 0.90:
        raise ValueError("table_surface_z must be finite and in [0.75, 0.90]")
    config["table_surface_z"] = table_z

    profiles = config.get("profiles")
    if not isinstance(profiles, dict) or "clear" not in profiles:
        raise ValueError("profiles must be a mapping containing 'clear'")
    if profiles["clear"].get("obstacles", []) != []:
        raise ValueError("the 'clear' profile must not contain obstacles")

    for profile_name, profile in profiles.items():
        scene = profile.get("scene", "any" if profile_name == "clear" else None)
        if scene not in ("any", "block", "fruit"):
            raise ValueError(
                f"profiles.{profile_name}.scene must be any, block, or fruit"
            )
        profile["scene"] = scene
        obstacles = profile.get("obstacles", [])
        if not isinstance(obstacles, list):
            raise ValueError(f"profiles.{profile_name}.obstacles must be a list")
        used: set[str] = set()
        for index, obstacle in enumerate(obstacles):
            label = f"profiles.{profile_name}.obstacles[{index}]"
            slot = obstacle.get("slot")
            if slot not in SLOTS:
                raise ValueError(f"{label}.slot={slot!r}; known slots: {SLOTS}")
            if slot in used:
                raise ValueError(f"{label}: slot {slot!r} is used twice")
            used.add(slot)
            position = np.asarray(obstacle.get("position"), dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError(f"{label}.position must contain 3 finite values")
            if not 0.38 <= position[0] <= 0.85 or not -0.35 <= position[1] <= 0.35:
                raise ValueError(f"{label}.position lies outside the tabletop workspace")
            if abs(float(position[2]) - table_z) > 0.02:
                raise ValueError(f"{label}.position z must be the table surface {table_z:.3f}")
            obstacle["position"] = position.tolist()
            yaw = float(obstacle.get("yaw", 0.0))
            if not np.isfinite(yaw):
                raise ValueError(f"{label}.yaw must be finite")
            obstacle["yaw"] = yaw
            if slot in HURDLE_SLOTS:
                bar_height = float(obstacle.get("bar_height", HURDLE_DEFAULT_BAR_HEIGHT))
                low, high = HURDLE_BAR_HEIGHT_RANGE
                if not np.isfinite(bar_height) or not low <= bar_height <= high:
                    raise ValueError(
                        f"{label}.bar_height must be finite and in [{low:.2f}, {high:.2f}] m"
                    )
                obstacle["bar_height"] = bar_height
            elif "bar_height" in obstacle:
                raise ValueError(f"{label}.bar_height is only valid for slots {HURDLE_SLOTS}")
    return config


def load_pick_place_obstacle_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return validate_pick_place_obstacle_config(json.load(stream))


def profile_names(config: dict | None = None) -> tuple[str, ...]:
    if config is None:
        config = load_pick_place_obstacle_config()
    return tuple(config["profiles"])


_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
_CAPSULE_BOX = {
    (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_BOX)),
    (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CAPSULE)),
}


def capsule_box_distance(
    model: mujoco.MjModel, data: mujoco.MjData, capsule_geom: int, box_geom: int
) -> float:
    """Separation of a capsule and a box geom; <= 0 when they overlap.

    MuJoCo 3.11 ``mj_geomDistance`` reports a *positive* distance (up to 34 mm in
    T40 E's survey) for about 5 % of overlapping capsule-box pairs, so the
    obstacle checks use this for those pairs.  A capsule is its axis segment
    grown by the radius, so the separation is ``min_t sdf_box(p(t)) - radius``
    over the segment ``p(t)``.  ``sdf_box`` is convex and ``p`` affine, so the
    minimum over ``t`` is found by golden-section search.  Positive values are
    the exact Euclidean distance; negative values mark overlap (their magnitude is
    a depth measure, not MuJoCo's penetration depth).
    """
    radius, half_length = (float(v) for v in model.geom_size[capsule_geom][:2])
    box_rot = data.geom_xmat[box_geom].reshape(3, 3)
    half = model.geom_size[box_geom]
    centre = box_rot.T @ (data.geom_xpos[capsule_geom] - data.geom_xpos[box_geom])
    axis = box_rot.T @ data.geom_xmat[capsule_geom].reshape(3, 3)[:, 2]

    def sdf(t: float) -> float:
        q = np.abs(centre + t * axis) - half
        return float(np.linalg.norm(np.maximum(q, 0.0)) + min(float(q.max()), 0.0))

    low, high = -half_length, half_length
    x1 = high - _GOLDEN * (high - low)
    x2 = low + _GOLDEN * (high - low)
    f1, f2 = sdf(x1), sdf(x2)
    for _ in range(60):
        if f1 <= f2:
            high, x2, f2 = x2, x1, f1
            x1 = high - _GOLDEN * (high - low)
            f1 = sdf(x1)
        else:
            low, x1, f1 = x1, x2, f2
            x2 = low + _GOLDEN * (high - low)
            f2 = sdf(x2)
    best = min(f1, f2, sdf(-half_length), sdf(half_length))
    return best - radius


def _is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    while body_id > 0:
        if body_id == ancestor_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom#{geom_id}"


class PickPlaceObstacleManager:
    """Place fixed obstacle props and measure clearance from the RBY1 body."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: dict):
        self.model = model
        self.data = data
        self.config = validate_pick_place_obstacle_config(config)
        # slot -> mocap id of its first (table-level) body; _slot_mocap_ids has all of them.
        self._mocap_ids: dict[str, int] = {}
        self._slot_mocap_ids: dict[str, tuple[int, ...]] = {}
        self._collision_geoms: dict[str, tuple[int, ...]] = {}
        for slot in SLOTS:
            mocap_ids: list[int] = []
            geom_ids: list[int] = []
            for body_name in SLOT_BODIES[slot]:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                body_geoms = [
                    geom_id
                    for geom_id in range(model.ngeom)
                    if int(model.geom_bodyid[geom_id]) == body_id
                    and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
                ]
                if body_id < 0 or not body_geoms:
                    raise RuntimeError(
                        "model is missing static pick-place obstacle slots; load "
                        "model_pick_place_obstacles.xml"
                    )
                mocap_id = int(model.body_mocapid[body_id])
                if mocap_id < 0:
                    raise RuntimeError(f"pick obstacle body {body_name!r} is not a mocap body")
                mocap_ids.append(mocap_id)
                geom_ids.extend(body_geoms)
            self._mocap_ids[slot] = mocap_ids[0]
            self._slot_mocap_ids[slot] = tuple(mocap_ids)
            self._collision_geoms[slot] = tuple(geom_ids)

        arm_root_ids = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_left_arm_0"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_right_arm_0"),
        )
        self.robot_geom_ids = tuple(
            geom_id for geom_id in range(model.ngeom)
            if any(
                _is_descendant(model, int(model.geom_bodyid[geom_id]), root_id)
                for root_id in arm_root_ids
            )
            and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
        )
        payload_body_ids = {
            body_id
            for body_name in MOVABLE_OBJECT_BODIES
            if (body_id := mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )) >= 0
        }
        self.payload_geom_ids = tuple(
            geom_id for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in payload_body_ids
            and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
        )
        self._support = GeomSupport(model, data)
        self.active_slots: tuple[str, ...] = ()
        self.profile_name = "clear"
        self.reset_metrics()
        self.park_all()

    def reset_metrics(self) -> None:
        self.contact_steps = 0
        self.robot_collision = False
        self.payload_collision = False
        self.min_robot_clearance = float("inf")
        self.contact_pairs: set[tuple[str, str]] = set()

    def park_all(self) -> None:
        self.active_slots = ()
        for slot, position in PARKING_POSITIONS.items():
            self._place(slot, position, 0.0, HURDLE_DEFAULT_BAR_HEIGHT)
        mujoco.mj_forward(self.model, self.data)

    def _place(self, slot: str, position, yaw: float, bar_height: float | None) -> None:
        quat = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        base = np.asarray(position, dtype=float)
        for index, mocap_id in enumerate(self._slot_mocap_ids[slot]):
            # index 0: the prop at the table; index 1 (hurdles only): the bar body.
            offset = np.array([0.0, 0.0, bar_height if index == 1 else 0.0])
            self.data.mocap_pos[mocap_id] = base + offset
            self.data.mocap_quat[mocap_id] = quat

    def activate(self, profile_name: str) -> None:
        profiles = self.config["profiles"]
        if profile_name not in profiles:
            raise ValueError(f"unknown obstacle profile {profile_name!r}; known: {tuple(profiles)}")
        self.park_all()
        self.reset_metrics()
        active: list[str] = []
        for obstacle in profiles[profile_name].get("obstacles", []):
            slot = obstacle["slot"]
            self._place(
                slot,
                obstacle["position"],
                float(obstacle["yaw"]),
                obstacle.get("bar_height", HURDLE_DEFAULT_BAR_HEIGHT),
            )
            active.append(slot)
        self.active_slots = tuple(active)
        self.profile_name = profile_name
        mujoco.mj_forward(self.model, self.data)

    @property
    def collision_geom_ids(self) -> set[int]:
        return {
            geom_id
            for slot in self.active_slots
            for geom_id in self._collision_geoms[slot]
        }

    def positions(self) -> dict[str, np.ndarray]:
        return {
            slot: self.data.mocap_pos[self._mocap_ids[slot]].copy()
            for slot in self.active_slots
        }

    def _minimum_distance(self, other_geom_ids: Iterable[int], max_distance: float) -> float:
        """Smallest :meth:`pair_distance` over (active obstacle geom, other geom), capped.

        Pairs go in order of their bounding-sphere gap (a lower bound on their distance), each
        with ``distmax`` = the smallest checked distance so far, and the sweep stops at the first
        pair whose sphere gap exceeds it.  Every value that lowers the minimum has passed
        :meth:`_checked_distance` (T40 H), so a false 0.0 can no longer become the minimum.
        """
        if not self.active_slots:
            return float("inf")
        best = float(max_distance)
        obstacles = np.fromiter(sorted(self.collision_geom_ids), dtype=np.int64)
        others = np.fromiter(other_geom_ids, dtype=np.int64)
        if not len(others):
            return best
        xpos, rbound = self.data.geom_xpos, self.model.geom_rbound
        gap = (
            np.linalg.norm(xpos[others][None, :, :] - xpos[obstacles][:, None, :], axis=2)
            - rbound[others][None, :] - rbound[obstacles][:, None]
        )
        gap[:, rbound[others] <= 0.0] = -np.inf  # rbound 0 = plane: no bound
        near = np.flatnonzero(gap <= best)
        from_to = np.zeros(6, dtype=np.float64)
        for index in near[np.argsort(gap.flat[near], kind="stable")]:
            if gap.flat[index] > best:
                break
            row, col = divmod(int(index), len(others))
            # distmax stays > 0: mj_geomDistance only measures depth with a positive cut-off
            distance = self.pair_distance(
                int(obstacles[row]), int(others[col]), max(best, 1e-6), from_to
            )
            best = min(best, distance)
        return best

    def _exact_pair(self, geom1: int, geom2: int) -> bool:
        types = (int(self.model.geom_type[geom1]), int(self.model.geom_type[geom2]))
        return types in _CAPSULE_BOX

    def _raw_distance(
        self, geom1: int, geom2: int, max_distance: float, from_to: np.ndarray | None
    ) -> float:
        """capsule-box: exact (:func:`capsule_box_distance`); else ``mj_geomDistance``."""
        if self._exact_pair(geom1, geom2):
            types = (int(self.model.geom_type[geom1]), int(self.model.geom_type[geom2]))
            capsule, box = (geom1, geom2) if types[0] == mujoco.mjtGeom.mjGEOM_CAPSULE else (geom2, geom1)
            centre_gap = float(np.linalg.norm(self.data.geom_xpos[capsule] - self.data.geom_xpos[box]))
            if centre_gap - self.model.geom_rbound[capsule] - self.model.geom_rbound[box] > max_distance:
                return float(max_distance)
            return min(float(max_distance), capsule_box_distance(self.model, self.data, capsule, box))
        if from_to is None:
            from_to = np.zeros(6, dtype=np.float64)
        return float(mujoco.mj_geomDistance(
            self.model, self.data, geom1, geom2, max_distance, from_to
        ))

    def _checked_distance(
        self, geom1: int, geom2: int, raw: float, max_distance: float, from_to: np.ndarray
    ) -> float:
        """T40 H: keep a raw ``mj_geomDistance`` value only if it is consistent.

        ``>= max_distance``: MuJoCo's cut-off exit (a separating plane) -- kept.  Positive and
        :func:`certify` passes: kept.  Otherwise (a false 0.0, a claimed overlap, an
        inconsistent witness) the exact :func:`gjk_distance` decides: separated -> that
        distance, overlapping -> ``min(raw, 0)`` (MuJoCo's depth when it has one).
        Geom types without a support function keep the raw value (old behaviour).
        """
        if raw >= max_distance:
            return float(max_distance)
        support = self._support
        if not (support.supported(geom1) and support.supported(geom2)):
            return raw
        if certify(support, geom1, geom2, raw, from_to):
            return raw
        exact = gjk_distance(support, geom1, geom2)
        if exact <= 0.0:
            return min(raw, 0.0)
        return min(float(max_distance), exact)

    def pair_distance(
        self, geom1: int, geom2: int, max_distance: float, from_to: np.ndarray | None = None
    ) -> float:
        """Distance of two geoms, capped at max_distance like mj_geomDistance; <= 0 on overlap.

        Capsule-box pairs use :func:`capsule_box_distance` (mj_geomDistance is wrong for some
        overlapping capsule-box pairs in MuJoCo 3.11, T40 E).  Every other pair is
        mj_geomDistance checked by :meth:`_checked_distance` (MuJoCo 3.11 returns a false 0.0
        for some separated cylinder-mesh / box pairs, T40 H).
        """
        if self._exact_pair(geom1, geom2):
            return self._raw_distance(geom1, geom2, max_distance, None)
        if from_to is None:
            from_to = np.zeros(6, dtype=np.float64)
        raw = self._raw_distance(geom1, geom2, max_distance, from_to)
        return self._checked_distance(geom1, geom2, raw, max_distance, from_to)

    def robot_clearance(self, max_distance: float = 0.25) -> float:
        clearance = self._minimum_distance(self.robot_geom_ids, max_distance)
        self.min_robot_clearance = min(self.min_robot_clearance, clearance)
        return clearance

    def object_clearance(self, max_distance: float = 0.25) -> float:
        return self._minimum_distance(self.payload_geom_ids, max_distance)

    def observe_contacts(self) -> bool:
        obstacle_geoms = self.collision_geom_ids
        if not obstacle_geoms:
            return False
        robot_geoms = set(self.robot_geom_ids)
        payload_geoms = set(self.payload_geom_ids)
        contacted = False
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not pair & obstacle_geoms:
                continue
            if pair & robot_geoms:
                self.robot_collision = True
            elif pair & payload_geoms:
                self.payload_collision = True
            else:
                continue
            contacted = True
            self.contact_pairs.add(tuple(sorted(_geom_name(self.model, geom) for geom in pair)))
        if contacted:
            self.contact_steps += 1
        return contacted

    def unsafe(self, stop_distance: float) -> bool:
        self.observe_contacts()
        clearance = self.robot_clearance(max(0.25, stop_distance + 0.05))
        return self.robot_collision or clearance <= stop_distance

    def hold_robot(self) -> None:
        """Freeze every joint-position actuator at its measured joint position."""
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id >= 0:
                self.data.ctrl[actuator_id] = self.data.qpos[self.model.jnt_qposadr[joint_id]]

    def summary(self) -> dict:
        return {
            "profile": self.profile_name,
            "active_slots": list(self.active_slots),
            "robot_collision": self.robot_collision,
            "payload_collision": self.payload_collision,
            "contact_steps": self.contact_steps,
            "min_robot_clearance_m": self.min_robot_clearance,
            "contact_pairs": [list(pair) for pair in sorted(self.contact_pairs)],
        }
