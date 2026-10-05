"""T43 T fix 2 — `AG3S.held_free_observed_pad`: on every held frame the TSDF voxels within the pad of
the manipulated object's last pre-grasp observed points, **at their observed place** (base frame, not
riding the hand), are freed too — above the support plane + one fine voxel, never near the destination.
Off (None, default): `_held_free_points` returns exactly the pre-T43 array. CPU only (the point set is
what the cuRobo builder receives; the builder is not run)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from benchmark.ag3s.robot_models.held_object import points_near
from benchmark.ag3s.types import SupportSurface

tests_r2 = pytest.importorskip("tests.ag3s.test_t43r2_held_body_cover")


def _plane(normal, offset, sid=1):
    return SupportSurface(id=sid, normal=np.asarray(normal, float), offset=float(offset),
                          point_indices=np.zeros(0, np.int64), point_count=1000, rms_error=0.001)


@pytest.fixture(scope="module")
def rby1():
    from benchmark.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf

    return UrdfSphereChain(parse_urdf(RBY1_URDF),
                           link_filter=["ee_left", "ee_right", "ee_finger_l1", "ee_finger_l2",
                                        "ee_finger_r1", "ee_finger_r2"])


def _held(rby1, pad):
    """A pear-like body on a table: the fitted cap is observed, the bulge below it too (pre-grasp)."""
    q = np.zeros(rby1.nq)
    T1, T2 = rby1.link_pose(q, "ee_finger_l1"), rby1.link_pose(q, "ee_finger_l2")
    c = 0.5 * (T1[:3, 3] + T2[:3, 3])
    table = c[2] - 0.045
    body = tests_r2._ellipsoid_surface(c + [0, 0, -0.005], np.array([0.040, 0.040, 0.045]), n=1500)
    cap = tests_r2._cap(body, c, [1.0, 0.0, 0.6], 0.3)            # what the closing frame shows
    ag = tests_r2._ag3s(rby1, False)
    ag.held_free_observed_pad = pad
    ag._last_support_surfaces = [_plane([0, 0, 1], table)]                     # noqa: SLF001
    ag._last_esdf_field = tests_r2._TableField(table, 0.0088)                  # noqa: SLF001
    ag._last_object_points = None if pad is None else body.copy()             # noqa: SLF001
    held = ag.attach(tests_r2._target(cap), robot_state=q, parent_link="ee_finger_l1")
    return ag, held, q, table, body


def test_points_near_is_a_shared_lattice_within_the_distance():
    p = np.array([[0.0, 0.0, 0.0], [0.004, 0.0, 0.0]])
    g = points_near(p, 0.010, 0.0025)
    d = np.min(np.linalg.norm(g[:, None] - p[None], axis=2), axis=1)
    assert d.max() <= 0.010 + 1e-9
    assert np.allclose(np.round(g / 0.0025), g / 0.0025)                # one lattice, no duplicates
    assert len(np.unique(np.round(g / 0.0025).astype(int), axis=0)) == len(g)
    assert points_near(np.zeros((0, 3)), 0.01, 0.0025).shape == (0, 3)
    assert points_near(p, 0.0, 0.0025).shape == (0, 3)


def test_off_is_the_pre_t43_array(rby1):
    ag, held, q, table, _ = _held(rby1, None)
    assert ag._held_capture_points is None                                     # noqa: SLF001
    assert "capture_free" not in ag.held_record
    surfaces = [_plane([0, 0, 1], table)]
    a = ag._held_free_points(q, surfaces, None)                                # noqa: SLF001
    ag.held_free_observed_pad = None
    b = ag._held_free_points(q, surfaces, None)                                # noqa: SLF001
    assert a is not None and b is not None


def test_off_on_differ_only_by_the_observation_band(rby1):
    off, _, q, table, body = _held(rby1, None)
    on, _, _, _, _ = _held(rby1, 0.015)
    surfaces = [_plane([0, 0, 1], table)]
    a = off._held_free_points(q, surfaces, None)                               # noqa: SLF001
    on._held_first_free_frames = off._held_first_free_frames                   # noqa: SLF001
    b = on._held_free_points(q, surfaces, None)                                # noqa: SLF001
    assert len(b) > len(a)
    assert np.array_equal(b[:len(a)], a)                                       # the old set first
    extra = b[len(a):]
    d, _ = cKDTree(body).query(extra)
    assert d.max() <= 0.015 + 1e-9
    assert extra[:, 2].min() >= table + 0.005 - 1e-9                           # never into the table
    assert on.held_record["capture_free"] == {"pad_mm": 15.0, "n_points": len(body)}


def test_the_observation_is_freed_at_its_place_not_riding_the_hand(rby1):
    on, _, q, table, body = _held(rby1, 0.015)
    surfaces = [_plane([0, 0, 1], table)]
    on._held_first_free_frames = 0                                             # noqa: SLF001

    def extra(qq):
        b = on._held_free_points(qq, surfaces, None)                           # noqa: SLF001
        on.held_free_observed_pad = None
        a = on._held_free_points(qq, surfaces, None)                           # noqa: SLF001
        on.held_free_observed_pad = 0.015
        assert np.array_equal(b[:len(a)], a)
        return b[len(a):]

    q2 = q.copy()
    q2[on.constraint_robot_model.joint_names.index("left_arm_1")] += 0.3
    e0, e1 = extra(q), extra(q2)
    assert len(e0) and np.array_equal(e0, e1)          # the hand moved, the freed place did not


def test_the_destination_is_never_freed(rby1):
    on, _, q, table, body = _held(rby1, 0.015)
    surfaces = [_plane([0, 0, 1], table)]
    dest = body[body[:, 0] > np.median(body[:, 0])] + [0.0, 0.0, 0.0]
    b = on._held_free_points(q, surfaces, dest)                                # noqa: SLF001
    d, _ = cKDTree(dest).query(b)
    assert d.min() > 2.0 * 0.005 - 1e-9


def test_detach_and_reset_clear_the_capture(rby1):
    on, _, q, table, _ = _held(rby1, 0.015)
    assert on._held_capture_points is not None                                # noqa: SLF001
    on.detach()
    assert on._held_capture_points is None                                     # noqa: SLF001
    on, _, _, _, _ = _held(rby1, 0.015)
    on.reset()
    assert on._held_capture_points is None and on._last_object_points is None  # noqa: SLF001
