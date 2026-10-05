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
* ``bar_height`` (``hurdle_0``, metres, default 0.26) is the height of the bar
  *centre* above ``position[2]``, in [0.20, 0.32].  ``fence_0`` / ``overbar_0``
  take it too, with their own ranges (below); any other slot rejects the key.

Slots: ``bollard_0``/``bollard_1`` (cylinder posts), ``divider_0`` (boxes;
MuJoCo 3.11 ``mj_geomDistance`` is wrong for some box pairs, T40 A) and
``hurdle_0`` (capsule bar on two telescoping posts, capsules/cylinders only).
The hurdle is two mocap bodies, ``pick_obstacle_hurdle_0`` (feet + sleeves on
the table) and ``pick_obstacle_hurdle_0_bar`` (bar + sliding posts, placed at
``position + [0, 0, bar_height]``): the height is set with mocap positions only,
so no compiled model quantity (geom size, bounding sphere, AABB, BVH) changes.

Black obstacles for the pick and the place phase (T42-2)
--------------------------------------------------------
Every obstacle's visual geoms are black by default (``BLACK_RGBA``; the scene
XML colour).  Any obstacle entry may set ``"rgba": [r, g, b, a]`` (each in
[0, 1]); it recolours that slot's visual geoms only (``model.geom_rgba``), never
a collision geom, and ``park_all`` / the next ``activate`` restore black.

New slots, capsules and cylinders only (``SLOT_PARAMS`` lists ranges/defaults):

==============  =====  ====================================================  =============================
slot            phase  shape                                                 size keys (m)
==============  =====  ====================================================  =============================
``post_0``      pick   vertical cylinder beside the fruit                    ``height``, ``radius``
``fence_0``     pick   low capsule rail on two short legs (r 7 mm)          ``bar_height``, ``length``,
                                                                             ``radius`` (rail)
``overbar_0``   pick   capsule bar above the fruit on two legs (r 12 mm)    ``bar_height``, ``length``,
                                                                             ``radius`` (bar)
``hurdle_0``    place  T40 E hurdle                                          ``bar_height``
``crate_post_0`` place vertical cylinder beside the crate rim               ``height``, ``radius``
==============  =====  ====================================================  =============================

``position`` is the footprint centre on the table for all of them; ``yaw``
turns the bar (local x) of ``fence_0`` / ``overbar_0``, whose legs stand at
``position +- R(yaw) [length/2, 0, 0]``.  ``bar_height`` is the bar *centre*
above the table, ``height`` the post top above the table, ``length`` the
distance between the legs (= the bar's axis segment; the capsule adds its
radius at each end).  A size key on a slot that does not take it is an error.

Each of these slots is one mocap body per collision geom (+ a visual twin of the
same size), with an explicit inertial frame, so a size only changes
``geom_size`` and the quantities MuJoCo derives from it: ``geom_rbound``,
``geom_aabb`` and the body's one-node BVH (``bvh_aabb``).  :meth:`activate`
writes all four; ``park_all`` restores the compiled values (= the defaults).
``tests/sim/test_t42_obstacles.py`` checks the result equals a model compiled
with the same sizes.  Several slots may be active in one profile (e.g. a pick
obstacle and a place obstacle); every active slot's collision geoms go through
:meth:`PickPlaceObstacleManager.pair_distance` (T40 H) in the reset clearance
and the run-time stop checks.  Example (one pick + one place obstacle)::

    "ep1828_post_and_crate_post": {"scene": "fruit", "obstacles": [
      {"slot": "post_0", "position": [0.625, 0.30, 0.82], "height": 0.20, "radius": 0.02},
      {"slot": "crate_post_0", "position": [0.64, 0.17, 0.82], "height": 0.26,
       "rgba": [0.02, 0.02, 0.02, 1.0]}]}
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
SLOTS = (
    "bollard_0", "bollard_1", "divider_0", "hurdle_0",
    "post_0", "fence_0", "overbar_0", "crate_post_0",
)
HURDLE_SLOTS = ("hurdle_0",)
HURDLE_BAR_HEIGHT_RANGE = (0.20, 0.32)
HURDLE_DEFAULT_BAR_HEIGHT = 0.26
#: T42-2 slots: one cylinder, one mocap body
POST_SLOTS = ("post_0", "crate_post_0")
#: T42-2 slots: a capsule bar (local x) on two cylinder legs, three mocap bodies
FRAME_SLOTS = ("fence_0", "overbar_0")
FRAME_LEG_RADIUS = {"fence_0": 0.007, "overbar_0": 0.012}
#: which task phase a slot is meant for (documentation / reports only; placement is the config's)
SLOT_PHASE = {
    "bollard_0": "any", "bollard_1": "any", "divider_0": "any",
    "post_0": "pick", "fence_0": "pick", "overbar_0": "pick",
    "hurdle_0": "place", "crate_post_0": "place",
}
SIZE_KEYS = ("bar_height", "height", "radius", "length")
#: slot -> size key -> (low, high, default), metres.  The defaults are the scene XML's compiled sizes.
SLOT_PARAMS: dict[str, dict[str, tuple[float, float, float]]] = {
    "hurdle_0": {"bar_height": (*HURDLE_BAR_HEIGHT_RANGE, HURDLE_DEFAULT_BAR_HEIGHT)},
    "post_0": {"height": (0.05, 0.40, 0.22), "radius": (0.010, 0.045, 0.020)},
    "crate_post_0": {"height": (0.05, 0.40, 0.26), "radius": (0.010, 0.045, 0.020)},
    "fence_0": {"bar_height": (0.03, 0.15, 0.07), "length": (0.08, 0.30, 0.18),
                "radius": (0.005, 0.020, 0.008)},
    "overbar_0": {"bar_height": (0.12, 0.45, 0.24), "length": (0.16, 0.50, 0.34),
                  "radius": (0.008, 0.030, 0.012)},
}
BLACK_RGBA = (0.02, 0.02, 0.02, 1.0)


def _slot_bodies(slot: str) -> tuple[str, ...]:
    base = f"pick_obstacle_{slot}"
    if slot in HURDLE_SLOTS:
        return (base, f"{base}_bar")
    if slot in FRAME_SLOTS:
        return (base, f"{base}_leg_left", f"{base}_leg_right")
    return (base,)


# Mocap bodies per slot.  The first body is the slot's main body; a hurdle's
# second body (the bar) sits bar_height above it; a frame's legs follow its bar.
SLOT_BODIES = {slot: _slot_bodies(slot) for slot in SLOTS}
MOVABLE_OBJECT_BODIES = (
    "red_block", "green_block", "blue_block",
    "apple", "banana", "orange", "pear", "crate",
)
PARKING_POSITIONS = {
    "bollard_0": np.array([8.0, 8.0, 2.0]),
    "bollard_1": np.array([9.0, 8.0, 2.0]),
    "divider_0": np.array([10.0, 8.0, 2.0]),
    "hurdle_0": np.array([11.0, 8.0, 2.0]),
    "post_0": np.array([12.0, 8.0, 2.0]),
    "fence_0": np.array([13.0, 8.0, 2.0]),
    "overbar_0": np.array([14.0, 8.0, 2.0]),
    "crate_post_0": np.array([15.0, 8.0, 2.0]),
}


def slot_sizes(slot: str, sizes: dict | None = None) -> dict[str, float]:
    """The slot's size keys: the defaults of ``SLOT_PARAMS`` updated with ``sizes``."""
    out = {key: spec[2] for key, spec in SLOT_PARAMS.get(slot, {}).items()}
    out.update({key: float(value) for key, value in (sizes or {}).items() if key in out})
    return out


def slot_layout(slot: str, sizes: dict | None = None) -> tuple[tuple[np.ndarray, tuple[float, float] | None], ...]:
    """Per mocap body of ``slot`` (``SLOT_BODIES`` order): its offset from the footprint in the
    slot frame (before yaw) and, for resizable bodies, the (radius, half-length) of its geoms."""
    p = slot_sizes(slot, sizes)
    zero = np.zeros(3)
    if slot in HURDLE_SLOTS:
        return ((zero, None), (np.array([0.0, 0.0, p["bar_height"]]), None))
    if slot in POST_SLOTS:
        half = p["height"] / 2.0
        return ((np.array([0.0, 0.0, half]), (p["radius"], half)),)
    if slot in FRAME_SLOTS:
        bar_height, half_length = p["bar_height"], p["length"] / 2.0
        leg = (FRAME_LEG_RADIUS[slot], bar_height / 2.0)
        return (
            (np.array([0.0, 0.0, bar_height]), (p["radius"], half_length)),
            (np.array([-half_length, 0.0, bar_height / 2.0]), leg),
            (np.array([half_length, 0.0, bar_height / 2.0]), leg),
        )
    return ((zero, None),)


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
            params = SLOT_PARAMS.get(slot, {})
            for key in SIZE_KEYS:
                if key in params:
                    low, high, default = params[key]
                    value = float(obstacle.get(key, default))
                    if not np.isfinite(value) or not low <= value <= high:
                        raise ValueError(
                            f"{label}.{key} must be finite and in [{low:.3f}, {high:.3f}] m"
                        )
                    obstacle[key] = value
                elif key in obstacle:
                    takes = tuple(name for name in SLOTS if key in SLOT_PARAMS.get(name, {}))
                    raise ValueError(f"{label}.{key} is only valid for slots {takes}")
            if "rgba" in obstacle:
                rgba = np.asarray(obstacle["rgba"], dtype=float)
                if rgba.shape != (4,) or not np.isfinite(rgba).all() or (rgba < 0).any() or (rgba > 1).any():
                    raise ValueError(f"{label}.rgba must be 4 values in [0, 1]")
                obstacle["rgba"] = rgba.tolist()
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


def _rotated_extent(quat: np.ndarray, extent: np.ndarray) -> np.ndarray:
    """Half extents of the axis-aligned box around a centred box of half extents ``extent`` rotated by
    ``quat``.  Corners rotated with ``mju_rotVecQuat``, then half of max - min: the order of operations
    that reproduces the MuJoCo 3.11 compiler's ``bvh_aabb`` bit for bit (``|R| @ extent`` is 1-3 ulp off)."""
    corners = np.empty((8, 3))
    for index in range(8):
        signs = np.array([1.0 if index & bit else -1.0 for bit in (1, 2, 4)])
        mujoco.mju_rotVecQuat(corners[index], signs * extent, quat)
    return 0.5 * (corners.max(axis=0) - corners.min(axis=0))


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
        # slot -> mocap id of its first (main) body; _slot_mocap_ids has all of them.
        self._mocap_ids: dict[str, int] = {}
        self._slot_mocap_ids: dict[str, tuple[int, ...]] = {}
        self._slot_body_ids: dict[str, tuple[int, ...]] = {}
        self._collision_geoms: dict[str, tuple[int, ...]] = {}
        self._visual_geoms: dict[str, tuple[int, ...]] = {}
        for slot in SLOTS:
            mocap_ids: list[int] = []
            body_ids: list[int] = []
            geom_ids: list[int] = []
            visual_ids: list[int] = []
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
                body_ids.append(body_id)
                geom_ids.extend(body_geoms)
                visual_ids.extend(
                    geom_id for geom_id in range(model.ngeom)
                    if int(model.geom_bodyid[geom_id]) == body_id
                    and not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
                )
            self._mocap_ids[slot] = mocap_ids[0]
            self._slot_mocap_ids[slot] = tuple(mocap_ids)
            self._slot_body_ids[slot] = tuple(body_ids)
            self._collision_geoms[slot] = tuple(geom_ids)
            self._visual_geoms[slot] = tuple(visual_ids)
        # colour: the scene XML's (black) is the default; an entry's "rgba" overrides it
        self._default_rgba = {
            slot: model.geom_rgba[list(geoms)].copy() for slot, geoms in self._visual_geoms.items()
        }
        self._init_resizable()
        self._footprints: dict[str, np.ndarray] = {}

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
            self.model.geom_rgba[list(self._visual_geoms[slot])] = self._default_rgba[slot]
        self._restore_compiled_sizes()
        self._footprints = {}
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ resizable slots (T42-2)

    _RESIZED_FIELDS = ("geom_size", "geom_rbound", "geom_aabb")

    def _init_resizable(self) -> None:
        """Snapshot the compiled sizes of every resizable body and check the layout matches them.

        A resizable body (``POST_SLOTS`` / ``FRAME_SLOTS``) holds exactly one collision geom,
        centred on the body origin, plus visual twins of the same type and size, and has an
        explicit inertial frame equal to the body frame (so its BVH is one node in the body frame
        and no inertial quantity depends on the size).  The scene XML's sizes must be the
        ``SLOT_PARAMS`` defaults: then :meth:`_resize_body` with the defaults reproduces the
        compiled arrays bit for bit, which is checked here.
        """
        m = self.model
        self._resizable: dict[int, tuple[tuple[int, ...], int]] = {}  # body -> (geoms, bvh node)
        for slot in (*POST_SLOTS, *FRAME_SLOTS):
            for body_id, (_, size) in zip(self._slot_body_ids[slot], slot_layout(slot)):
                geoms = tuple(g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == body_id)
                colliding = [g for g in geoms if m.geom_contype[g] or m.geom_conaffinity[g]]
                kinds = {int(m.geom_type[g]) for g in geoms}
                if (
                    len(colliding) != 1 or int(m.body_bvhnum[body_id]) != 1
                    or not kinds <= {int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)}
                    or len(kinds) != 1
                    or any(np.any(m.geom_pos[g] != 0.0) for g in geoms)
                    or np.any(m.body_ipos[body_id] != 0.0)
                    or np.any(m.body_iquat[body_id] != (1.0, 0.0, 0.0, 0.0))
                ):
                    raise RuntimeError(
                        f"obstacle body {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id)!r} "
                        "is not a resizable slot body (one centred capsule/cylinder collision geom, "
                        "explicit inertial at the body origin)"
                    )
                self._resizable[body_id] = (geoms, int(m.body_bvhadr[body_id]))
                if not np.array_equal(m.geom_size[geoms[0], :2], np.asarray(size)):
                    raise RuntimeError(
                        f"geom size of {SLOT_BODIES[slot]} is not the SLOT_PARAMS default: the scene "
                        "XML changed, or another PickPlaceObstacleManager left this model resized "
                        "(call its park_all() first)"
                    )
        self._compiled_sizes = {
            body: (
                {field: getattr(m, field)[list(geoms)].copy() for field in self._RESIZED_FIELDS},
                m.bvh_aabb[node].copy(),
            )
            for body, (geoms, node) in self._resizable.items()
        }
        for body in self._resizable:  # the derived formulas reproduce the compiler
            geoms, node = self._resizable[body]
            fields, bvh = self._compiled_sizes[body]
            self._resize_body(body, *m.geom_size[geoms[0], :2])
            if not (all(np.array_equal(getattr(m, f)[list(geoms)], v) for f, v in fields.items())
                    and np.array_equal(m.bvh_aabb[node], bvh)):
                self._restore_compiled_sizes()
                raise RuntimeError("geom bounds recomputed from geom_size differ from the compiled model")

    def _restore_compiled_sizes(self) -> None:
        m = self.model
        for body, (fields, bvh) in self._compiled_sizes.items():
            geoms, node = self._resizable[body]
            for field, value in fields.items():
                getattr(m, field)[list(geoms)] = value
            m.bvh_aabb[node] = bvh

    def _resize_body(self, body_id: int, radius: float, half: float) -> None:
        """Set (radius, half-length) of every geom of a resizable body and what MuJoCo derives from it.

        Same formulas as the MuJoCo 3.11 compiler: ``geom_rbound`` = r + h (capsule) /
        sqrt(r^2 + h^2) (cylinder); ``geom_aabb`` = centre 0, half extents (r, r, h + r) / (r, r, h)
        in the geom frame; the body's single BVH node = that box in the body frame (geom at the
        origin, rotated by ``geom_quat``).
        """
        m = self.model
        geoms, node = self._resizable[body_id]
        radius, half = float(radius), float(half)
        capsule = int(m.geom_type[geoms[0]]) == mujoco.mjtGeom.mjGEOM_CAPSULE
        extent = np.array([radius, radius, half + radius if capsule else half])
        rbound = radius + half if capsule else math.sqrt(radius * radius + half * half)
        for geom in geoms:
            m.geom_size[geom, 0] = radius
            m.geom_size[geom, 1] = half
            m.geom_rbound[geom] = rbound
            m.geom_aabb[geom, :3] = 0.0
            m.geom_aabb[geom, 3:] = extent
        m.bvh_aabb[node, :3] = 0.0
        m.bvh_aabb[node, 3:] = _rotated_extent(np.array(m.geom_quat[geoms[0]]), extent)

    def _place(
        self, slot: str, position, yaw: float, bar_height: float | None = None,
        sizes: dict | None = None,
    ) -> None:
        """Put ``slot`` at footprint ``position`` / ``yaw`` with the given sizes (defaults otherwise).

        ``bar_height`` is the T40 E positional argument and applies to ``HURDLE_SLOTS`` only;
        every other size goes in ``sizes`` (keys of ``SLOT_PARAMS[slot]``).
        """
        sizes = dict(sizes or {})
        if bar_height is not None and slot in HURDLE_SLOTS:
            sizes.setdefault("bar_height", bar_height)
        quat = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        base = np.asarray(position, dtype=float)
        layout = slot_layout(slot, sizes)
        for mocap_id, body_id, (offset, size) in zip(
            self._slot_mocap_ids[slot], self._slot_body_ids[slot], layout
        ):
            if offset[0] == 0.0 and offset[1] == 0.0:  # on the vertical axis (T40 E bodies: exactly as before)
                self.data.mocap_pos[mocap_id] = base + np.array([0.0, 0.0, offset[2]])
            else:
                self.data.mocap_pos[mocap_id] = base + np.array([
                    cos_yaw * offset[0] - sin_yaw * offset[1],
                    sin_yaw * offset[0] + cos_yaw * offset[1],
                    offset[2],
                ])
            self.data.mocap_quat[mocap_id] = quat
            if size is not None:
                self._resize_body(body_id, *size)
        self._footprints[slot] = base.copy()

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
                sizes={key: obstacle[key] for key in SLOT_PARAMS.get(slot, {}) if key in obstacle},
            )
            if obstacle.get("rgba") is not None:
                self.model.geom_rgba[list(self._visual_geoms[slot])] = obstacle["rgba"]
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
        """Footprint position of each active slot (the config ``position``)."""
        return {slot: self._footprints[slot].copy() for slot in self.active_slots}

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
