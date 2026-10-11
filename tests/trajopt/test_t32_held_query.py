"""T32 H2 on the optimizer side — the held object is queried as **spheres**, not radius-0 points.

* `scene_from_constraint_set` hands the fitted held spheres (centre + radius, parent frame) to the
  linearizer; an attachment built the pre-T32 way (no `held_object` role) keeps its point queries.
* The held rows stay alive against the environment (a crate wall, with the destination margin);
  they measure `d(centre) − r − margin`.
* The finger/robot rows do not see the held object: it is not in the field (H1) and adding it as
  query spheres changes no robot row.
* The SQP loop calls `set_attached(points, link)` without radii — the radii travel on the scene.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.ag3s.robot_models.held_object import HELD_ROLE
from AG3S.ag3s.types import AttachedCollisionGeometry, Primitive, PrimitiveType
from AG3S.trajopt.config import TrajOptConfig
from AG3S.trajopt.linearize import CollisionLinearizer, SceneSnapshot, scene_from_constraint_set

WALL_X = 0.60
DEST = "destination"
DEST_MARGIN = 0.020
ESDF_MARGIN = 0.050


class _CrateWall:
    """A crate wall at x = WALL_X (obstacle for x > WALL_X), labelled as the destination."""

    has_labels = True

    def __init__(self):
        self.grid = types.SimpleNamespace(voxel_size=0.005)

    def distance(self, pts):
        return WALL_X - np.asarray(pts, np.float64).reshape(-1, 3)[:, 0]

    def gradient(self, pts):
        g = np.zeros((len(np.asarray(pts).reshape(-1, 3)), 3))
        g[:, 0] = -1.0
        return g

    def is_label(self, pts, label):
        return np.full(len(np.asarray(pts).reshape(-1, 3)), label == DEST)


def _held(role=HELD_ROLE):
    prims = [Primitive(type=PrimitiveType.SPHERE, center=np.array([0.0, 0.0, 0.10]),
                       dimensions=np.full(3, 0.039), semantic_role=role),
             Primitive(type=PrimitiveType.SPHERE, center=np.array([0.0, 0.0, 0.145]),
                       dimensions=np.full(3, 0.008), semantic_role=role)]
    pts = np.array([[0.0, 0.0, 0.066], [0.03, 0.0, 0.10], [0.0, 0.03, 0.10]])
    return AttachedCollisionGeometry(parent_link="ee_left", T_parent_object=np.eye(4),
                                     primitives=prims, points=pts,
                                     allowed_contact_links=frozenset({"ee_left"}))


def _constraint_set(attached, field):
    return types.SimpleNamespace(constraints=None, esdf=field, attached=attached, manipulated=None,
                                 manipulated_link_margin=None, target_field_exclude=None,
                                 destination_label=DEST, destination_margin=DEST_MARGIN,
                                 support_surfaces=[])


def _cfg():
    return TrajOptConfig.from_dict({"collision": {"backend": "esdf", "esdf_margin": ESDF_MARGIN,
                                                  "use_support_planes": False}})


def test_scene_carries_the_held_spheres_not_the_points():
    field = _CrateWall()
    scene = scene_from_constraint_set(_constraint_set(_held(), field), np.array([0.01]), _cfg())
    assert scene.attached_parent_link == "ee_left"
    assert np.allclose(scene.attached_points, [[0, 0, 0.10], [0, 0, 0.145]])
    assert np.allclose(scene.attached_radii, [0.039, 0.008])


def test_a_pre_t32_attachment_keeps_its_point_queries():
    field = _CrateWall()
    old = _held(role="target")
    scene = scene_from_constraint_set(_constraint_set(old, field), np.array([0.01]), _cfg())
    assert scene.attached_radii is None
    assert np.allclose(scene.attached_points, old.points)


def _linearizer(robot_radii):
    lin = object.__new__(CollisionLinearizer)
    lin.robot_radii = np.asarray(robot_radii, np.float64)
    lin.query_radii = lin.robot_radii
    lin.n_spheres = len(robot_radii)
    lin._attached_local = None
    lin._attached_link = None
    return lin


def _snapshot(field, radii, attached_radii=None):
    return SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
        candidate_active=np.zeros(0, bool), d_safe=np.zeros((len(radii), 0)),
        plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=np.asarray(radii), candidate_ids=np.zeros(0, np.int64),
        esdf=field, esdf_margin=ESDF_MARGIN, destination_label=DEST,
        destination_margin=DEST_MARGIN, attached_radii=attached_radii)


def test_held_sphere_rows_against_the_crate_wall_stay_alive():
    """The held sphere approaching the wall: `d − r − destination margin`, negative once it is in."""
    field = _CrateWall()
    finger_r = 0.0033
    lin = _linearizer([finger_r])
    held_r = np.array([0.039, 0.008])
    lin.set_attached(np.zeros((2, 3)), "ee_left")              # points only, as the SQP loop does
    scene = _snapshot(field, [finger_r], attached_radii=held_r)
    finger = np.array([0.40, 0.0, 0.9])
    for x, sign in ((0.45, +1), (0.55, -1)):
        centres = np.array([[finger, [x, 0.0, 0.9], [x, 0.0, 0.95]]])      # (1, 3, 3)
        c = lin._esdf_clearance(centres, scene)[0, :, 0]                   # noqa: SLF001
        assert c[1] == pytest.approx(WALL_X - x - 0.039 - DEST_MARGIN)
        assert np.sign(c[1]) == sign
        assert c[2] == pytest.approx(WALL_X - x - 0.008 - DEST_MARGIN)
    # the radii reached the linearizer's own table (every reader agrees)
    assert np.allclose(lin.query_radii, [finger_r, 0.039, 0.008])


def test_finger_rows_do_not_see_the_held_object():
    """The held object is not an obstacle for the fingers: it is not in the field (H1 removes its
    pixels, the seed exclusion its old surface) and its query spheres change no robot row."""
    field = _CrateWall()
    finger_r = 0.0033
    finger = np.array([[0.52, 0.0, 0.9], [0.50, 0.03, 0.9]])
    bare = _linearizer([finger_r, finger_r])
    c0 = bare._esdf_clearance(finger[None], _snapshot(field, [finger_r] * 2))[0, :, 0]  # noqa: SLF001
    held = _linearizer([finger_r, finger_r])
    held.set_attached(np.zeros((1, 3)), "ee_left", radii=np.array([0.039]))
    # the held sphere sits right between the fingers
    centres = np.vstack([finger, [[0.51, 0.015, 0.9]]])[None]
    c1 = held._esdf_clearance(centres, _snapshot(field, [finger_r] * 2,          # noqa: SLF001
                                                  attached_radii=np.array([0.039])))[0, :, 0]
    assert np.allclose(c1[:2], c0)
    # the finger rows get the full margin (the wall is not the destination for them)
    assert np.allclose(c0, WALL_X - finger[:, 0] - finger_r - ESDF_MARGIN)


def test_set_attached_rejects_a_radius_count_mismatch():
    lin = _linearizer([0.01])
    with pytest.raises(ValueError, match="radii"):
        lin.set_attached(np.zeros((2, 3)), "ee_left", radii=np.array([0.03]))
