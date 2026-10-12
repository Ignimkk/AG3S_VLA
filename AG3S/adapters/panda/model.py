"""Panda (LIBERO) robot model for AG3S: the same structure as V11AB's RB-Y1 model (SL2).

V11AB builds two robot models from one URDF sphere chain (`UrdfSphereChain`):

* **self-filter model** -- the whole robot plus *covering* capsules fitted where the collision
  spheres leave the simulator's visible surface uncovered (`grounding_report.build_robot_model`,
  `mujoco_source.self_filter_covering_capsules`). It deletes the robot from the depth cloud.
* **constraint model** -- the collision spheres of the links the optimizer can move, no covers
  (`grounding_report.build_constraint_robot_model`, `link_filter`).

This module builds the Panda equivalents:

* kinematics -- ``robots/panda/description/panda_libero.urdf``: the cuRobo ``franka_panda.urdf`` with
  the robosuite hand/finger offsets (provenance in that file's header; FK measured against the LIBERO
  MuJoCo model by ``tests/benchmark/test_panda_sl2.py``);
* constraint spheres -- cuRobo ``franka.yml`` ``collision_spheres`` (61 on 11 links), each a
  zero-length `UrdfCapsule` in ``extra_capsules`` (the Panda URDF has mesh collisions only, which
  `parse_urdf` does not read: 0 URDF capsules);
* self-filter covers -- ``robots/panda/description/libero_panda_self_filter_cover.json``, fitted by
  V11AB's own `self_filter_covering_capsules` on the robosuite Panda meshes (``cover.py``), with role
  `COVER_ROLE`; never in the constraint model;
* finger opening -- the two finger joints are `UrdfSphereChain` *parameter* joints (T29), set from
  the adapter ``robot_state`` (:func:`set_finger_parameters`).

Link sets and their RB-Y1 roles (``grounding_report`` / ``config.DEFAULT_CONTACT_LINKS``)::

    role (RB-Y1 name)                RB-Y1 links                               Panda links
    PALM_LINKS                       ee_left, ee_right                         panda_hand
    FINGER_LINKS                     ee_finger_{l1,l2,r1,r2}                   panda_leftfinger, panda_rightfinger
    GRIPPER_LINKS (= --links gripper, V11AB default)
                                     PALM + FINGER                             PALM + FINGER
    ARM_LINKS (= --links arms)       link_{left,right}_arm_0..6 + GRIPPER      panda_link1..7 + GRIPPER
    not in ARM_LINKS (no decision    base, wheels, torso, head                 panda_link0 (fixed base) and the
      variable moves them)                                                       mount (cover only)
    DEFAULT_CONTACT_LINKS[side]      left: ee_left, ee_finger_l1, l2           left: panda_hand, panda_leftfinger,
                                     right: ee_right, ee_finger_r1, r2               panda_rightfinger

The Panda is one arm; it is the **left** manipulator (:data:`MANIPULATOR_SIDE`): its wrist camera is
the ``left_wrist`` wire camera (SL1, pi05_libero token slot), so camera and contact side agree, and
it occupies the L slot of V11AB's 16-D layout ``[L7, Lg, R7, Rg]`` (SL-5).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
from typing import Any, Mapping, Optional, Sequence

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
DESCRIPTION_DIR = PROJECT_ROOT / "robots/panda/description"
#: Adapter URDF (kinematics only).
PANDA_URDF = DESCRIPTION_DIR / "panda_libero.urdf"
#: Self-filter covering capsules, fitted by ``cover.py`` (generated; provenance inside).
PANDA_SELF_FILTER_COVER = DESCRIPTION_DIR / "libero_panda_self_filter_cover.json"
#: cuRobo robot config whose ``collision_spheres`` are the constraint geometry (submodule, read only).
CUROBO_FRANKA_YML = PROJECT_ROOT / "third_party/curobo/curobo/content/configs/robot/franka.yml"
#: cuRobo URDF the adapter URDF was derived from (only its arm joints are reused verbatim).
CUROBO_FRANKA_URDF = PROJECT_ROOT / "third_party/curobo/curobo/content/assets/robot/franka_description/franka_panda.urdf"

#: Decision-variable joints, in the order of the adapter ``robot_state[:7]``.
PANDA_ARM_JOINTS: tuple[str, ...] = tuple(f"panda_joint{i}" for i in range(1, 8))
PANDA_FINGER_JOINTS: tuple[str, ...] = ("panda_finger_joint1", "panda_finger_joint2")

#: Adapter ``robot_state`` (MuJoCo qpos, ``camera.ROBOT_STATE_LAYOUT``) -> URDF joint and sign.
#: ``gripper0_finger_joint2`` slides along +y with range [-0.04, 0]; the URDF joint slides along -y
#: with range [0, 0.04], so its value is negated.
JOINT_MAP: tuple[tuple[str, str, float], ...] = (
    *((f"robot0_joint{i}", f"panda_joint{i}", 1.0) for i in range(1, 8)),
    ("gripper0_finger_joint1", "panda_finger_joint1", 1.0),
    ("gripper0_finger_joint2", "panda_finger_joint2", -1.0),
)

#: MuJoCo body (LIBERO robosuite model) -> URDF link with the identical frame (FK test pairs).
BODY_TO_LINK: dict[str, str] = {
    "robot0_base": "panda_link0",
    "robot0_link0": "panda_link0",
    **{f"robot0_link{i}": f"panda_link{i}" for i in range(1, 8)},
    "robot0_right_hand": "panda_hand",
    "gripper0_leftfinger": "panda_leftfinger",
    "gripper0_rightfinger": "panda_rightfinger",
}
#: MuJoCo site / camera -> URDF link (fixed frames added by the adapter URDF).
SITE_TO_LINK: dict[str, str] = {"gripper0_grip_site": "panda_grip_site"}
CAMERA_TO_LINK: dict[str, str] = {"robot0_eye_in_hand": "panda_eye_in_hand"}  # OpenCV axes

# --- link sets (RB-Y1 roles: module docstring) ------------------------------------------------------
PANDA_PALM_LINKS: tuple[str, ...] = ("panda_hand",)
PANDA_FINGER_LINKS: tuple[str, ...] = ("panda_leftfinger", "panda_rightfinger")
#: Palm + fingers. The Panda analogue of ``GRIPPER_LINKS`` (``--links gripper``, V11AB default).
PANDA_GRIPPER_LINKS: tuple[str, ...] = PANDA_PALM_LINKS + PANDA_FINGER_LINKS
#: Links moved by the 7 arm joints + gripper. ``panda_link0`` is the fixed base: like RB-Y1's
#: base/wheels/torso, no decision variable moves it, so it is not a constraint link.
PANDA_ARM_LINKS: tuple[str, ...] = tuple(f"panda_link{i}" for i in range(1, 8)) + PANDA_GRIPPER_LINKS
#: Every link that carries geometry (self-filter scope).
PANDA_ALL_LINKS: tuple[str, ...] = ("panda_link0",) + PANDA_ARM_LINKS

#: The single Panda arm is the V11AB ``left`` manipulator (module docstring).
MANIPULATOR_SIDE = "left"
#: Panda analogue of ``config.DEFAULT_CONTACT_LINKS``: links that may touch the target, per side.
PANDA_CONTACT_LINKS: dict[str, tuple[str, ...]] = {MANIPULATOR_SIDE: PANDA_GRIPPER_LINKS}

#: `serve_safe --links` groups for the Panda (same names as ``serve_safe.LINK_GROUPS``).
PANDA_LINK_GROUPS: dict[str, Optional[tuple[str, ...]]] = {
    "gripper": PANDA_GRIPPER_LINKS, "arms": PANDA_ARM_LINKS, "all": None}

#: RB-Y1 set name -> Panda set (the table above, as data; manifest and tests read it).
LINK_SET_TABLE: tuple[dict, ...] = (
    {"role": "PALM_LINKS", "rby1": ("ee_left", "ee_right"), "panda": PANDA_PALM_LINKS},
    {"role": "FINGER_LINKS", "rby1": tuple(f"ee_finger_{s}{i}" for s in ("l", "r") for i in (1, 2)),
     "panda": PANDA_FINGER_LINKS},
    {"role": "GRIPPER_LINKS (--links gripper, V11AB)", "rby1": "PALM_LINKS + FINGER_LINKS", "panda": PANDA_GRIPPER_LINKS},
    {"role": "ARM_LINKS (--links arms)", "rby1": "link_{left,right}_arm_0..6 + GRIPPER_LINKS", "panda": PANDA_ARM_LINKS},
    {"role": "fixed links (filter only)", "rby1": "base, wheels, torso, head", "panda": ("panda_link0", "mount (cover)")},
    {"role": "DEFAULT_CONTACT_LINKS", "rby1": {"left": ("ee_left", "ee_finger_l1", "ee_finger_l2"),
                                               "right": ("ee_right", "ee_finger_r1", "ee_finger_r2")},
     "panda": dict(PANDA_CONTACT_LINKS)},
)


# --- sources ----------------------------------------------------------------------------------------
def file_sha256(path: str | pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def load_panda_urdf(path: str | pathlib.Path = PANDA_URDF):
    from AG3S.ag3s.robot_models.urdf_sphere_chain import parse_urdf

    return parse_urdf(pathlib.Path(path).resolve())


def franka_collision_spheres(path: str | pathlib.Path = CUROBO_FRANKA_YML) -> dict[str, list[tuple[np.ndarray, float]]]:
    """``{link: [(centre (3,), radius), ...]}`` from cuRobo ``franka.yml``, file order."""
    import yaml

    data = yaml.safe_load(pathlib.Path(path).read_text())
    raw = data["robot_cfg"]["kinematics"]["collision_spheres"]
    out: dict[str, list[tuple[np.ndarray, float]]] = {}
    for link, spheres in raw.items():
        out[str(link)] = [(np.asarray(s["center"], np.float64).reshape(3), float(s["radius"])) for s in spheres]
    return out


def franka_sphere_capsules(urdf_links: Sequence[str], path: str | pathlib.Path = CUROBO_FRANKA_YML) -> list:
    """The ``franka.yml`` spheres as zero-length `UrdfCapsule` (role ``collision``), file order.

    Every sphere link must be a link of the adapter URDF and every radius > 0 -- a sphere that does
    not attach, or attaches with no volume, is a hole nobody sees, so this raises instead.
    """
    from AG3S.ag3s.robot_models.urdf_sphere_chain import UrdfCapsule

    known = set(urdf_links)
    capsules = []
    for link, spheres in franka_collision_spheres(path).items():
        if link not in known:
            raise ValueError(f"franka.yml sphere link {link!r} is not a link of {PANDA_URDF.name} ({sorted(known)})")
        for centre, radius in spheres:
            if not radius > 0.0:
                raise ValueError(f"franka.yml sphere on {link} has radius {radius}")
            origin = np.eye(4)
            origin[:3, 3] = centre
            capsules.append(UrdfCapsule(link=link, origin=origin, radius=radius, length=0.0))
    return capsules


def load_self_filter_cover(path: str | pathlib.Path = PANDA_SELF_FILTER_COVER) -> tuple[list, dict]:
    """``(capsules, document)`` of the committed cover file (role `COVER_ROLE`)."""
    from AG3S.adapters.panda.cover import read_cover_json

    return read_cover_json(path)


# --- link sets --------------------------------------------------------------------------------------
def check_link_set(links: Optional[Sequence[str]], *, sphere_links: Sequence[str], what: str = "link set") -> None:
    """Every name in ``links`` must be a link that carries a sphere. A typo raises.

    `UrdfSphereChain(link_filter=...)` keeps capsules whose link is in the set and silently ignores a
    name that matches nothing -- the constraint set would quietly lose that link. ``None`` = all.
    """
    if links is None:
        return
    links = tuple(links)
    if not links:
        raise ValueError(f"{what} is empty")
    unknown = sorted(set(links) - set(sphere_links))
    if unknown:
        raise ValueError(f"{what} names link(s) with no Panda sphere: {unknown}; known: {sorted(set(sphere_links))}")


def panda_constraint_link_filter(links: str) -> Optional[tuple[str, ...]]:
    """`serve_safe.constraint_link_filter` for the Panda: ``gripper`` | ``arms`` | ``all`` (None)."""
    if links not in PANDA_LINK_GROUPS:
        raise ValueError(f"links must be one of {sorted(PANDA_LINK_GROUPS)}, got {links!r}")
    return PANDA_LINK_GROUPS[links]


# --- builders ---------------------------------------------------------------------------------------
def build_panda_bare_model(urdf=None, **sphere_options):
    """URDF + ``franka.yml`` spheres on every link, no covers. The model the cover is fitted against
    (the RB-Y1 ``bare`` of ``build_robot_model``)."""
    from AG3S.ag3s.robot_models.urdf_sphere_chain import UrdfSphereChain

    urdf = urdf or load_panda_urdf()
    return UrdfSphereChain(urdf, PANDA_ARM_JOINTS, extra_capsules=franka_sphere_capsules(urdf.links),
                           **sphere_options)


def build_panda_self_filter_model(cover_path: str | pathlib.Path = PANDA_SELF_FILTER_COVER, urdf=None):
    """Self-filter model: every link's ``franka.yml`` spheres + the committed covering capsules.

    No sphere options -- like RB-Y1 ``build_robot_model``, a thinner filter leaks robot points as
    obstacles. Finger joints are parameters (:func:`set_finger_parameters`).
    """
    from AG3S.ag3s.robot_models.urdf_sphere_chain import UrdfSphereChain

    urdf = urdf or load_panda_urdf()
    cover, document = load_self_filter_cover(cover_path)
    model = UrdfSphereChain(urdf, PANDA_ARM_JOINTS,
                            extra_capsules=franka_sphere_capsules(urdf.links) + list(cover))
    model.panda_cover_document = document
    return model


def build_panda_constraint_model(link_filter: Optional[Sequence[str]] = PANDA_ARM_LINKS, *,
                                 sphere_options: Optional[Mapping[str, Any]] = None, urdf=None):
    """Constraint model: ``franka.yml`` spheres of ``link_filter`` links, **no covers**.

    Default ``PANDA_ARM_LINKS`` (RB-Y1 ``build_constraint_robot_model`` default ``ARM_LINKS``);
    V11AB serves ``gripper`` -> ``panda_constraint_link_filter("gripper")``. ``None`` = whole robot.
    ``sphere_options`` go to `UrdfSphereChain` unchanged (note: the spheres are zero-length capsules
    with role ``collision``, so ``capsule_radius_scale`` scales their radius).
    """
    from AG3S.ag3s.robot_models.urdf_sphere_chain import UrdfSphereChain

    urdf = urdf or load_panda_urdf()
    spheres = franka_sphere_capsules(urdf.links)
    check_link_set(link_filter, sphere_links=[c.link for c in spheres], what="constraint link_filter")
    return UrdfSphereChain(urdf, PANDA_ARM_JOINTS, extra_capsules=spheres,
                           link_filter=None if link_filter is None else tuple(link_filter),
                           **dict(sphere_options or {}))


# --- robot_state -> model -----------------------------------------------------------------------------
def split_robot_state(robot_state: Sequence[float]) -> tuple[np.ndarray, dict[str, float]]:
    """Adapter ``robot_state`` (9: 7 arm qpos + 2 MuJoCo finger qpos) -> ``(q7, finger parameters)``."""
    rs = np.asarray(robot_state, np.float64).reshape(-1)
    if rs.shape[0] != len(JOINT_MAP):
        raise ValueError(f"Panda robot_state must have {len(JOINT_MAP)} entries, got {rs.shape[0]}")
    q = np.array([rs[i] * sign for i, (_, joint, sign) in enumerate(JOINT_MAP) if joint in PANDA_ARM_JOINTS])
    fingers = {joint: float(rs[i] * sign) for i, (_, joint, sign) in enumerate(JOINT_MAP)
               if joint in PANDA_FINGER_JOINTS}
    return q, fingers


def set_finger_parameters(model, robot_state: Sequence[float]) -> np.ndarray:
    """Put the finger opening of ``robot_state`` into ``model``; return the 7 arm joint values."""
    q, fingers = split_robot_state(robot_state)
    model.set_joint_parameters(fingers)
    return q


def panda_depth_robot_mask(depth_m: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray,
                           robot_state: Sequence[float], model, config=None, **kwargs):
    """One camera's self-filter with the Panda model: V11AB `backproject` + `depth_robot_mask`.

    The only Panda-specific step is :func:`set_finger_parameters` (9-D ``robot_state`` -> 7 arm
    joints + finger parameters) -- the job `serve_safe` does for RB-Y1 from the 16-D state. The model
    keeps the finger values afterwards (as `set_finger_joints_from_scene` does).
    """
    from AG3S.ag3s.config import PointCloudConfig
    from AG3S.ag3s.stages.reconstruction import backproject
    from AG3S.ag3s.stages.robot_filter import depth_robot_mask

    config = config or PointCloudConfig()
    depth = np.asarray(depth_m, np.float64)
    cloud = backproject(depth, K, T_base_cam, config)
    q = set_finger_parameters(model, robot_state)
    return depth_robot_mask(cloud, depth.shape, model, q, config, **kwargs)


# --- reporting --------------------------------------------------------------------------------------
def sphere_counts(model) -> dict[str, dict[str, int]]:
    """``{link: {role: count}}`` of a built model."""
    out: dict[str, dict[str, int]] = {}
    for link, role in zip(model.sphere_link_names, model.sphere_roles):
        out.setdefault(link, {}).setdefault(role, 0)
        out[link][role] += 1
    return out


def manifest_entry() -> dict:
    """What the run manifest records about the Panda robot model (JSON-able)."""
    cover = json.loads(PANDA_SELF_FILTER_COVER.read_text()) if PANDA_SELF_FILTER_COVER.exists() else None
    spheres = franka_collision_spheres()
    return {
        "urdf": {"path": str(PANDA_URDF.relative_to(PROJECT_ROOT)), "sha256": file_sha256(PANDA_URDF),
                 "derived_from": str(CUROBO_FRANKA_URDF.relative_to(PROJECT_ROOT))},
        "constraint_spheres": {"source": str(CUROBO_FRANKA_YML.relative_to(PROJECT_ROOT)),
                               "sha256": file_sha256(CUROBO_FRANKA_YML),
                               "count": sum(len(v) for v in spheres.values()),
                               "per_link": {k: len(v) for k, v in spheres.items()}},
        "self_filter_cover": None if cover is None else {
            "path": str(PANDA_SELF_FILTER_COVER.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(PANDA_SELF_FILTER_COVER), "count": len(cover["capsules"]),
            "geometry_digest": cover["provenance"]["geometry_digest"]},
        "joint_map": [{"robot_state": a, "urdf": b, "sign": s} for a, b, s in JOINT_MAP],
        "manipulator_side": MANIPULATOR_SIDE,
        "link_sets": {"palm": list(PANDA_PALM_LINKS), "finger": list(PANDA_FINGER_LINKS),
                      "gripper": list(PANDA_GRIPPER_LINKS), "arms": list(PANDA_ARM_LINKS),
                      "contact": {k: list(v) for k, v in PANDA_CONTACT_LINKS.items()}},
    }


@dataclasses.dataclass(frozen=True)
class PandaModels:
    """The two models V11AB builds, for the Panda."""

    self_filter: Any
    constraint: Any


def build_panda_models(constraint_links: str = "gripper") -> PandaModels:
    return PandaModels(self_filter=build_panda_self_filter_model(),
                       constraint=build_panda_constraint_model(panda_constraint_link_filter(constraint_links)))


__all__ = [
    "BODY_TO_LINK", "CAMERA_TO_LINK", "CUROBO_FRANKA_YML", "JOINT_MAP", "LINK_SET_TABLE", "MANIPULATOR_SIDE",
    "PANDA_ALL_LINKS", "PANDA_ARM_JOINTS", "PANDA_ARM_LINKS", "PANDA_CONTACT_LINKS", "PANDA_FINGER_JOINTS",
    "PANDA_FINGER_LINKS", "PANDA_GRIPPER_LINKS", "PANDA_LINK_GROUPS", "PANDA_PALM_LINKS", "PANDA_SELF_FILTER_COVER",
    "PANDA_URDF", "PandaModels", "SITE_TO_LINK", "build_panda_bare_model", "build_panda_constraint_model",
    "build_panda_models", "build_panda_self_filter_model", "check_link_set", "franka_collision_spheres",
    "franka_sphere_capsules", "manifest_entry", "panda_constraint_link_filter", "panda_depth_robot_mask",
    "set_finger_parameters", "split_robot_state", "sphere_counts",
]
