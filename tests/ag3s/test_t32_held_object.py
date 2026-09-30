"""T32a — 쥔 물체를 로봇의 일부로 (H1 · H2) · target ball 구 맞춤 (S1) · 미세 TSDF truncation (H4-fix).

고정하는 것:

* **H2** 한쪽만 보인 구 표면에서 중심을 찾는다 (centroid 는 보이는 면 쪽으로 ~20 mm 치우친다) ·
  덮개 구가 모든 관측 점을 담는다 · 꼭지처럼 튀어나온 점은 작은 구 하나 · 평평한 조각은 반지름 상한 ·
  구가 아닌 표면은 centroid 로 물러난다 · RB-Y1 손가락 → 손바닥 · `AG3S.attach` 가 손바닥에 붙이고
  (슬롯이 예약됐을 때) reach 는 손가락에서 잰다.
* **H1** 쥔 구 안의 점은 self-filter 가 지운다 · detach 하면 안 지운다 · inflation 은 쥔 구에 0 ·
  **움직이는 쥔 물체가 TSDF 에 흔적을 남기지 않는다** (끈 대조군은 남긴다) · 손가락 구는 쥔 물체를
  장애물로 보지 않는다.
* **S1** `_target_ball` 이 한쪽만 보인 구의 뒷면까지 담고, 105 mm 떨어진 이웃 중심은 밖에 둔다.
* **H4-fix** 미세 TSDF truncation 키와 하한.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from benchmark.ag3s.config import AG3SConfig, EsdfConfig
from benchmark.ag3s.robot_models.held_object import (
    HELD_ROLE,
    HeldSphereFilterModel,
    covering_spheres,
    enclosing_ball,
    extend_inflation,
    fit_surface_sphere,
    held_spheres_of,
    palm_link_for,
    sphere_volume_points,
)

R_APPLE = 0.034
MAX_OPENING = 0.07146
#: RB-Y1 `DEFAULT_RBY1_JOINTS` indices of left_arm_0..3 (torso 0–5, right arm 6–12, left arm 13–19).
DEFAULT_LEFT_ARM = [13, 14, 15, 16]


def _cap(centre, radius=R_APPLE, *, view=(1.0, 0.0, 0.0), min_cos=0.2, n=4000, noise=0.0007,
         seed=0):
    """`view` 쪽에서 본 구 표면 (cos ≥ `min_cos`) + depth 잡음."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    v = np.asarray(view, np.float64) / np.linalg.norm(view)
    d = d[d @ v > min_cos]
    return np.asarray(centre, np.float64) + radius * d + rng.normal(0.0, noise, (len(d), 3))


# =============================================================================================
# H2 — 구 맞춤
# =============================================================================================


def test_fit_recovers_the_centre_of_a_one_sided_sphere():
    c = np.array([0.55, 0.10, 0.85])
    pts = _cap(c)
    fit = fit_surface_sphere(pts, max_radius=MAX_OPENING / 2)
    assert fit.accepted and not fit.radius_capped
    assert np.linalg.norm(fit.centre - c) < 0.002                      # centroid: ~20 mm off
    assert np.linalg.norm(pts.mean(axis=0) - c) > 0.015
    assert fit.radius == pytest.approx(R_APPLE, abs=0.002)
    assert fit.shift_from_centroid > 0.015


def test_a_flat_patch_is_capped_at_half_the_max_opening():
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(-0.03, 0.03, 500), rng.uniform(-0.03, 0.03, 500),
                           rng.normal(0.0, 0.0003, 500)])
    fit = fit_surface_sphere(pts, max_radius=MAX_OPENING / 2)
    assert fit.radius <= MAX_OPENING / 2 + 1e-9
    if fit.accepted:
        assert fit.radius_capped


def test_a_non_sphere_or_a_tiny_cluster_falls_back_to_the_centroid():
    rng = np.random.default_rng(2)
    blob = rng.normal(0.0, 0.02, (300, 3))                             # not a surface
    fit = fit_surface_sphere(blob, max_radius=MAX_OPENING / 2)
    assert fit.method == "centroid" and fit.reason
    assert np.allclose(fit.centre, blob.mean(axis=0))
    few = fit_surface_sphere(_cap([0, 0, 0])[:5])
    assert few.method == "centroid" and "too_few" in few.reason


def test_covering_spheres_hold_every_observed_point_and_give_the_stem_its_own_sphere():
    c = np.array([0.5, 0.0, 0.8])
    body = _cap(c, view=(0.3, 0.0, 1.0))
    stem = c + np.array([0.0, 0.0, 1.0]) * np.linspace(0.036, 0.048, 8)[:, None]
    pts = np.vstack([body, stem])
    fit = fit_surface_sphere(pts, max_radius=MAX_OPENING / 2)
    spheres, rec = covering_spheres(pts, fit, pad=0.005)
    assert rec["mode"] == "fit" and rec["n_cover"] == 1 and len(spheres) == 2
    assert spheres[0][1] == pytest.approx(fit.radius + 0.005)
    assert spheres[1][1] < 0.015                                         # small, not a 57 mm ball
    C = np.array([s for s, _ in spheres])
    R = np.array([r for _, r in spheres])
    gap = (np.linalg.norm(pts[:, None] - C[None], axis=2) - R[None]).min(axis=1)
    assert (gap <= 1e-12).all()
    # the whole apple (not only the seen side) is inside the primary sphere
    full = c + R_APPLE * _cap([0, 0, 0], 1.0, min_cos=-1.0, noise=0.0)
    assert (np.linalg.norm(full - spheres[0][0], axis=1) <= spheres[0][1]).all()


def test_too_many_protrusions_merge_into_one_enclosing_sphere():
    c = np.zeros(3)
    pts = np.vstack([_cap(c, view=(0, 0, 1))] + [c + 0.05 * np.eye(3)[i % 3] * (1 + 0.3 * i)
                                                  for i in range(5)])
    fit = fit_surface_sphere(pts, max_radius=MAX_OPENING / 2)
    spheres, rec = covering_spheres(pts, fit, pad=0.005, max_spheres=3)
    assert len(spheres) == 1 and rec["merged"]
    assert (np.linalg.norm(pts - spheres[0][0], axis=1) <= spheres[0][1] + 1e-12).all()


def test_palm_of_a_finger_is_found_structurally():
    from benchmark.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf
    from tests.ag3s.fixtures import PlanarTwoLinkArm

    model = UrdfSphereChain(parse_urdf(RBY1_URDF), link_filter=["ee_left", "ee_right"])
    assert palm_link_for(model, "ee_finger_l1") == "ee_left"
    assert palm_link_for(model, "ee_finger_r2") == "ee_right"
    assert palm_link_for(model, "ee_left") == "ee_left"
    assert palm_link_for(model, "link_left_arm_3") is None
    assert palm_link_for(PlanarTwoLinkArm(), "mock_fingertip") is None


@pytest.fixture(scope="module")
def rby1():
    from benchmark.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf

    return UrdfSphereChain(parse_urdf(RBY1_URDF),
                           link_filter=["ee_left", "ee_right", "ee_finger_l1", "ee_finger_l2",
                                        "ee_finger_r1", "ee_finger_r2"])


def _target(points, tid=3):
    from benchmark.ag3s.stages.geometry import fit_sphere
    from benchmark.ag3s.types import TargetGeometry

    pts = np.asarray(points, np.float64)
    return TargetGeometry(id=tid, points=pts, point_indices=np.arange(len(pts)),
                          centroid=pts.mean(axis=0), bounding_geometry=fit_sphere(pts),
                          attention_score=0.9, confidence=0.9)


def _rby1_ag3s(model, parents):
    from benchmark.ag3s.runtime.pipeline import AG3S

    cfg = AG3SConfig().with_overrides({"constraint": {"horizon": 2, "max_attached_primitives": 4},
                                       "clustering": {"gripper_max_opening": MAX_OPENING}})
    return AG3S(cfg, robot_model=model, constraint_robot_model=model,
                attached_parent_links=parents)


def _apple_between_fingers(model, q):
    T_l1 = model.link_pose(q, "ee_finger_l1")
    T_l2 = model.link_pose(q, "ee_finger_l2")
    return 0.5 * (T_l1[:3, 3] + T_l2[:3, 3])


def test_attach_fits_spheres_on_the_palm_and_checks_reach_on_the_finger(rby1):
    q = np.zeros(rby1.nq)
    c = _apple_between_fingers(rby1, q)
    ag3s = _rby1_ag3s(rby1, ("ee_left", "ee_right"))
    held = ag3s.attach(_target(_cap(c, view=(1.0, 0.0, 0.5))), robot_state=q,
                       parent_link="ee_finger_l1", allowed_contact_links=("ee_finger_l1",),
                       reach=0.12)
    assert held.parent_link == "ee_left"
    assert "ee_left" in held.allowed_contact_links and "ee_finger_l1" in held.allowed_contact_links
    assert all(p.semantic_role == HELD_ROLE for p in held.primitives)
    spheres = held_spheres_of(held)
    T = rby1.link_pose(q, "ee_left")
    centre_base = T[:3, :3] @ spheres[0][0] + T[:3, 3]
    assert np.linalg.norm(centre_base - c) < 0.003
    rec = ag3s.held_record
    assert rec["frame_link"] == "ee_left" and rec["grasp_link"] == "ee_finger_l1"
    assert rec["fit"]["method"] == "surface_fit"
    # the snapshot does not move with the finger opening (palm, not finger)
    rby1.set_joint_parameters({"gripper_finger_l1": -0.03, "gripper_finger_l2": 0.03})
    try:
        T2 = rby1.link_pose(q, "ee_left")
        assert np.allclose(T2, T)
    finally:
        rby1.set_joint_parameters(None)
    ag3s.detach()
    assert ag3s.held_record is None


def test_attach_keeps_the_finger_frame_when_the_palm_has_no_slots(rby1):
    q = np.zeros(rby1.nq)
    c = _apple_between_fingers(rby1, q)
    ag3s = _rby1_ag3s(rby1, ("ee_finger_l1", "ee_finger_r1"))
    held = ag3s.attach(_target(_cap(c)), robot_state=q, parent_link="ee_finger_l1")
    assert held.parent_link == "ee_finger_l1"
    with pytest.raises(ValueError, match="reserved"):
        ag3s.detach()
        ag3s.attach(_target(_cap(c)), robot_state=q, parent_link="ee_finger_l1",
                    hold_link="ee_left")


# =============================================================================================
# H1 — self-filter
# =============================================================================================


@pytest.fixture()
def mock_ag3s():
    from benchmark.ag3s.runtime.pipeline import AG3S
    from tests.ag3s.fixtures import FINGERTIP_LINK, PlanarTwoLinkArm

    arm = PlanarTwoLinkArm()
    cfg = AG3SConfig().with_overrides({"constraint": {"horizon": 2, "max_attached_primitives": 4},
                                       "collision_backend": "esdf",
                                       "esdf": {"voxel_size": 0.02, "fine_voxel_size": None}})
    return AG3S(cfg, robot_model=arm, attached_parent_links=(FINGERTIP_LINK, "link_1")), arm


def _held(arm, q, offset=(0.12, 0.0, 0.0), radius=0.04, link=None):
    """A prebuilt held sphere `offset` from the mock fingertip frame (as `AG3S.attach` would make)."""
    from benchmark.ag3s.types import AttachedCollisionGeometry, Primitive, PrimitiveType
    from tests.ag3s.fixtures import FINGERTIP_LINK

    link = link or FINGERTIP_LINK
    local = np.asarray(offset, np.float64)
    prim = Primitive(type=PrimitiveType.SPHERE, center=local, dimensions=np.full(3, radius),
                     semantic_role=HELD_ROLE)
    pts = local + radius * 0.8 * np.vstack([np.eye(3), -np.eye(3)])
    return AttachedCollisionGeometry(parent_link=link, T_parent_object=np.eye(4),
                                     primitives=[prim], points=pts,
                                     allowed_contact_links=frozenset({link}))


def _held_centre(arm, q, held):
    T = arm.link_pose_numeric(q, held.parent_link)
    return T[:3, :3] @ held.primitives[0].center + T[:3, 3], float(held.primitives[0].dimensions[0])


def test_points_inside_a_held_sphere_are_removed_by_the_self_filter(mock_ag3s):
    from benchmark.ag3s.stages.robot_filter import filter_robot_points
    from benchmark.ag3s.types import PointCloud
    from tests.ag3s.fixtures import DEFAULT_ARM_Q

    ag3s, arm = mock_ag3s
    ag3s.held_filter_slip_m = 0.0          # exactly the query spheres (the slip test is below)
    q = DEFAULT_ARM_Q
    held = _held(arm, q)
    ag3s.attach(held)
    c, r = _held_centre(arm, q, held)
    rng = np.random.default_rng(3)
    u = rng.normal(size=(500, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    inside = c + (r - 0.001) * u * rng.uniform(0.0, 1.0, (500, 1)) ** (1.0 / 3.0)
    outside = c + np.array([r + 0.02, 0.0, 0.0]) + rng.normal(0.0, 0.002, (50, 3))
    cloud = PointCloud(np.vstack([inside, outside]))
    ag3s._prepare_frame_filter()                                                  # noqa: SLF001
    kept, stats = filter_robot_points(cloud, ag3s._frame_filter_model, q,         # noqa: SLF001
                                      ag3s.config.pointcloud,
                                      inflation=ag3s._frame_filter_inflation)     # noqa: SLF001
    assert stats["n_removed"] >= len(inside)
    assert len(kept) == len(outside)
    assert ag3s._frame_filter_model.n_held == 1                                   # noqa: SLF001
    # detach → the object's pixels are scene again
    ag3s.detach()
    ag3s._prepare_frame_filter()                                                  # noqa: SLF001
    kept, _ = filter_robot_points(cloud, ag3s._frame_filter_model, q,             # noqa: SLF001
                                  ag3s.config.pointcloud,
                                  inflation=ag3s._frame_filter_inflation)         # noqa: SLF001
    assert len(kept) == len(cloud)
    assert ag3s._frame_filter_model is arm                                        # noqa: SLF001


def test_held_spheres_get_zero_inflation():
    assert extend_inflation(0.0, 10, 2) == 0.0
    ext = extend_inflation(0.05, 3, 2)
    assert np.allclose(ext, [0.05, 0.05, 0.05, 0.0, 0.0])
    ext = extend_inflation(np.array([0.01, 0.02]), 2, 1)
    assert np.allclose(ext, [0.01, 0.02, 0.0])
    assert extend_inflation(0.05, 3, 0) == 0.05


def test_filter_model_delegates_everything_but_the_spheres():
    from tests.ag3s.fixtures import DEFAULT_ARM_Q, PlanarTwoLinkArm

    arm = PlanarTwoLinkArm()
    model = HeldSphereFilterModel(arm, [(np.zeros(3), 0.03)], "link_1")
    c, r = model.sphere_centers_numeric(DEFAULT_ARM_Q)
    c0, r0 = arm.sphere_centers_numeric(DEFAULT_ARM_Q)
    assert len(r) == len(r0) + 1 and r[-1] == pytest.approx(0.03)
    assert np.allclose(c[-1], arm.link_pose_numeric(DEFAULT_ARM_Q, "link_1")[:3, 3])
    assert model.nq == arm.nq and model.n_spheres == arm.n_spheres + 1
    assert model.sphere_link_names[-1] == "held:link_1"


def _render(eye, look, K, hw, spheres, table_z):
    """Depth (H, W) of a table rectangle + spheres, OpenCV camera."""
    from tests.ag3s.fixtures import _hit_plane_rect, _hit_sphere, _look_at

    T = _look_at(eye, look)
    h, w = hw
    rows, cols = np.mgrid[0:h, 0:w]
    d_cam = np.stack([(cols.ravel() - K[0, 2]) / K[0, 0], (rows.ravel() - K[1, 2]) / K[1, 1],
                      np.ones(h * w)], axis=1)
    d = d_cam @ T[:3, :3].T
    o = T[:3, 3]
    t = _hit_plane_rect(o, d, table_z, ((0.2, 1.3), (-0.6, 0.6)))
    for c, r in spheres:
        t = np.minimum(t, _hit_sphere(o, d, np.asarray(c, np.float64), float(r)))
    z = np.where(np.isfinite(t), t * (d @ T[:3, 2]), 0.0)
    return z.reshape(h, w), T


@pytest.mark.parametrize("held_in_filter", [True, False])
def test_a_moving_held_object_leaves_no_tsdf_trace(mock_ag3s, held_in_filter):
    """H1: the held apple rides the hand through three places; with its spheres in the self-filter
    the TSDF never integrates it, so no old place is an obstacle. The control (switch off) does
    leave the trail — so the test can tell the two apart."""
    from benchmark.ag3s.fields.esdf import CameraDepth, EsdfBuilder

    ag3s, arm = mock_ag3s
    ag3s.self_filter_held_object = held_in_filter
    true_r = 0.034
    held = _held(arm, np.array([-0.6, -0.6]), radius=true_r + 0.005)
    ag3s.attach(held)
    K = np.array([[200.0, 0, 80.0], [0, 200.0, 60.0], [0, 0, 1.0]])
    cfg = EsdfConfig(voxel_size=0.02)
    builder = EsdfBuilder(cfg, bounds=((0.2, -0.6, 0.6), (1.3, 0.6, 1.6)))
    path = [np.array([-0.60, -0.60]), np.array([-0.45, -0.70]), np.array([-0.30, -0.80])]
    all_c = np.array([_held_centre(arm, q, held)[0] for q in path])
    mid = all_c.mean(axis=0)
    # One fixed camera looking level at the path: nothing behind the object, so no later ray can
    # clear an old place (a masked or missing ray is unobserved, not free) — exactly the case where
    # only not-integrating helps.
    eye = mid - np.array([0.7, 0.0, 0.0])
    centres = []
    field = None
    for q in path:
        c, _ = _held_centre(arm, q, held)
        centres.append(c)
        depth, T = _render(eye, mid, K, (120, 160), [(c, true_r)], table_z=0.30)
        ag3s._prepare_frame_filter()                                               # noqa: SLF001
        mask = ag3s._robot_mask_for(depth, K, T, q)                                # noqa: SLF001
        field = builder.update([CameraDepth("cam", depth, K, T, robot_mask=mask)])
    old = np.asarray(centres[:-1])
    # the surface of an apple at an old place, seen from the camera side
    probe = old + np.array([-1.0, 0.0, 0.0]) * true_r
    d = np.asarray(field.distance(probe))
    if held_in_filter:
        assert (d > 0.02).all(), d                      # nothing there: free / far
        assert sum(ag3s._mask_held_px) > 0              # noqa: SLF001 — pixels were masked
    else:
        assert (d < 0.02).any(), d                      # the trail the control leaves


def test_the_seed_exclusion_uses_the_held_volume_above_the_table():
    pts = sphere_volume_points([(np.array([0.0, 0.0, 0.034]), 0.039)], 0.005, z_min=0.005)
    assert len(pts) > 1000
    assert pts[:, 2].min() >= 0.005 - 1e-12
    assert (np.linalg.norm(pts - [0, 0, 0.034], axis=1) <= 0.039 + 1e-12).all()


def test_held_seed_points_come_from_the_spheres_at_the_current_pose(mock_ag3s):
    from tests.ag3s.fixtures import DEFAULT_ARM_Q

    ag3s, arm = mock_ag3s
    held = _held(arm, DEFAULT_ARM_Q)
    ag3s.attach(held)
    q = np.array([-0.3, -0.8])
    pts = ag3s._held_seed_points(q, ())                                           # noqa: SLF001
    c, r = _held_centre(arm, q, held)
    # spacing = one fine voxel (here the single 20 mm tier)
    assert len(pts) > 20 and (np.linalg.norm(pts - c, axis=1) <= r + 1e-9).all()
    # a points-only (pre-T32) attachment keeps the point path
    ag3s.detach()
    from benchmark.ag3s.types import AttachedCollisionGeometry
    from tests.ag3s.fixtures import FINGERTIP_LINK

    ag3s.attach(AttachedCollisionGeometry(parent_link=FINGERTIP_LINK, T_parent_object=np.eye(4),
                                          primitives=[], points=np.zeros((4, 3)) + 0.01 * np.eye(4, 3)))
    assert ag3s._held_seed_points(q, ()) is None                                  # noqa: SLF001


def test_exclusion_record_names_the_self_filter_while_held(mock_ag3s):
    from tests.ag3s.fixtures import DEFAULT_ARM_Q

    ag3s, arm = mock_ag3s
    ag3s.attach(_held(arm, DEFAULT_ARM_Q))
    ag3s._prepare_frame_filter()                                                  # noqa: SLF001
    rec = ag3s._exclusion_record(types.SimpleNamespace(manipulated=None, esdf=None),  # noqa: SLF001
                                 None, None, DEFAULT_ARM_Q)
    assert rec["source"] == "attached" and "self_filter" in rec["mechanism"]


# =============================================================================================
# S1 — target ball
# =============================================================================================


def test_target_ball_covers_the_back_of_a_one_sided_sphere_and_not_a_neighbour_105mm_away():
    from benchmark.ag3s.runtime.pipeline import AG3S

    cfg = AG3SConfig().with_overrides({"clustering": {"gripper_max_opening": MAX_OPENING}})
    ag3s = AG3S(cfg)
    table_z = 0.75
    apple = np.array([0.6, 0.0, table_z + R_APPLE])
    neighbour = apple + np.array([0.0, 0.105, 0.0])
    # seen from the head camera side: front-top, as the recordings are
    pts = _cap(apple, view=(-1.0, 0.0, 1.0), min_cos=0.1)
    table = types.SimpleNamespace(normal=np.array([0.0, 0.0, 1.0]), offset=table_z)
    ball = ag3s._target_ball(pts, [table], voxel=0.005)                          # noqa: SLF001
    assert ball is not None and ball.provenance["centre_method"] == "surface_fit"
    assert np.linalg.norm(apple - ball.centre) + R_APPLE <= ball.radius            # whole apple
    assert np.linalg.norm(neighbour - ball.centre) > ball.radius                   # neighbour out
    # the old rule (centroid + max distance + pad) missed the back
    c_old = pts.mean(axis=0)
    r_old = np.linalg.norm(pts - c_old, axis=1).max() + 0.005
    assert np.linalg.norm(apple - c_old) + R_APPLE > r_old
    # every observed point is still inside
    assert (np.linalg.norm(pts - ball.centre, axis=1) <= ball.radius + 1e-12).all()


def test_enclosing_ball_holds_observed_points_and_the_fitted_sphere():
    c = np.array([0.1, 0.2, 0.3])
    pts = _cap(c)
    fit = fit_surface_sphere(pts, max_radius=MAX_OPENING / 2)
    centre, radius = enclosing_ball(pts, fit, pad=0.005)
    assert radius >= fit.radius + 0.005
    assert (np.linalg.norm(pts - centre, axis=1) <= radius).all()


# =============================================================================================
# H4-fix — fine TSDF truncation
# =============================================================================================


def test_fine_truncation_is_its_own_key_with_a_floor():
    server = EsdfConfig(voxel_size=0.02, fine_voxel_size=0.005, tsdf_voxel_size=0.005)
    assert server.truncation == pytest.approx(0.060)                  # coarse unchanged
    assert server.fine_truncation == pytest.approx(0.030)             # default
    assert server.fine_truncation_floor == pytest.approx(0.020)       # 4 × 5 mm
    low = EsdfConfig(voxel_size=0.02, fine_voxel_size=0.005, tsdf_voxel_size=0.005,
                     fine_truncation_m=0.010)
    assert low.fine_truncation == pytest.approx(0.020)                # clamped up, never below
    off = EsdfConfig(voxel_size=0.02, fine_voxel_size=0.005, tsdf_voxel_size=0.005,
                     fine_truncation_m=None)
    assert off.fine_truncation == pytest.approx(off.truncation)       # pre-T32 build
    single = EsdfConfig(voxel_size=0.02)
    assert single.fine_truncation == pytest.approx(single.truncation)
    with pytest.raises(Exception, match="fine_truncation_m"):
        AG3SConfig.from_dict({"esdf": {"voxel_size": 0.02, "fine_voxel_size": 0.005,
                                       "fine_truncation_m": -0.01}})
    via = AG3SConfig.from_dict({"esdf": {"voxel_size": 0.02, "fine_voxel_size": 0.005,
                                         "tsdf_voxel_size": 0.005, "fine_truncation_m": 0.025}})
    assert via.esdf.fine_truncation == pytest.approx(0.025)


def test_the_builder_splits_the_tsdf_only_when_the_truncations_differ():
    from benchmark.ag3s.fields.curobo_builder import CuroboFieldBuilder

    b = CuroboFieldBuilder(EsdfConfig(voxel_size=0.02, fine_voxel_size=0.005,
                                      tsdf_voxel_size=0.005))
    assert b.separate_fine_tsdf and b.fine_truncation == pytest.approx(0.030)
    assert b.coarse_truncation == pytest.approx(0.060)
    same = CuroboFieldBuilder(EsdfConfig(voxel_size=0.02, fine_voxel_size=0.005,
                                         tsdf_voxel_size=0.005, fine_truncation_m=None))
    assert not same.separate_fine_tsdf
    assert not CuroboFieldBuilder(EsdfConfig(voxel_size=0.02)).separate_fine_tsdf


def test_the_snapshot_uses_the_state_the_geometry_was_captured_at(rby1):
    """The attach request comes a request after the geometry was seen; the hand has moved since (the
    lift). Snapshotting at the request's q put the held points 12–46 mm off (T31-diag D3)."""
    from benchmark.ag3s.stages.target_grounding import ManipulatedIdentity

    q_seen = np.zeros(rby1.nq)
    q_now = q_seen.copy()
    q_now[DEFAULT_LEFT_ARM] += 0.15                     # the arm moved between capture and attach
    c = _apple_between_fingers(rby1, q_seen)
    ag3s = _rby1_ag3s(rby1, ("ee_left", "ee_right"))
    geom = _target(_cap(c, view=(1.0, 0.0, 0.5)))
    ag3s._manipulated = ManipulatedIdentity(                                  # noqa: SLF001
        id=3, state="visible", geometry=geom, centroid=geom.centroid, age_frames=0,
        last_seen_frame=7, last_seen_time=1.0, switched_from=None, frame=7)
    ag3s._state_by_frame[7] = q_seen                                           # noqa: SLF001
    held = ag3s.attach(geom, robot_state=q_now, parent_link="ee_finger_l1")
    rec = ag3s.held_record
    assert rec["snapshot_state"].startswith("frame 7")
    assert rec["hand_shift_since_capture_mm"] > 10.0
    local = held_spheres_of(held)[0][0]
    T_seen = rby1.link_pose(q_seen, "ee_left")
    assert np.linalg.norm(T_seen[:3, :3] @ local + T_seen[:3, 3] - c) < 0.003   # rides the hand
    # an explicit state wins; without either the request's q is used (pre-T32 behaviour)
    ag3s.detach()
    ag3s._manipulated = None                                                   # noqa: SLF001
    ag3s.attach(geom, robot_state=q_now, parent_link="ee_finger_l1")
    assert ag3s.held_record["snapshot_state"] == "attach_request"
    ag3s.detach()
    ag3s.attach(geom, robot_state=q_now, parent_link="ee_finger_l1", geometry_state=q_seen)
    assert ag3s.held_record["snapshot_state"] == "given"


def test_the_filter_spheres_carry_the_slip_allowance_and_the_query_spheres_do_not(mock_ag3s):
    """The held apple slides in the hand (up to 24 mm over a transport, T32a §2): the self-filter
    spheres are the query spheres + `HELD_FILTER_SLIP_M`; the constraint side is untouched."""
    from benchmark.ag3s.robot_models.held_object import HELD_FILTER_SLIP_M
    from benchmark.ag3s.stages.robot_filter import filter_robot_points
    from benchmark.ag3s.types import PointCloud
    from benchmark.trajopt.config import TrajOptConfig
    from benchmark.trajopt.linearize import scene_from_constraint_set
    from tests.ag3s.fixtures import DEFAULT_ARM_Q

    ag3s, arm = mock_ag3s
    assert ag3s.held_filter_slip_m == HELD_FILTER_SLIP_M == pytest.approx(0.020)
    held = _held(arm, DEFAULT_ARM_Q)
    ag3s.attach(held)
    c, r = _held_centre(arm, DEFAULT_ARM_Q, held)
    slipped = c + np.array([0.0, 0.0, -(r + 0.015)])          # 15 mm out of the query sphere
    beyond = c + np.array([0.0, 0.0, -(r + 0.035)])
    ag3s._prepare_frame_filter()                                                   # noqa: SLF001
    kept, _ = filter_robot_points(PointCloud(np.vstack([slipped, beyond])),
                                  ag3s._frame_filter_model, DEFAULT_ARM_Q,          # noqa: SLF001
                                  ag3s.config.pointcloud, inflation=ag3s._frame_filter_inflation)  # noqa: SLF001
    assert len(kept) == 1 and np.allclose(kept.points[0], beyond)
    cs = types.SimpleNamespace(constraints=None, esdf=object(), attached=held, manipulated=None,
                               manipulated_link_margin=None, target_field_exclude=None,
                               destination_label=None, destination_margin=None, support_surfaces=[])
    scene = scene_from_constraint_set(
        cs, np.asarray(arm.radii), TrajOptConfig.from_dict({"collision": {"backend": "esdf"}}))
    assert np.allclose(scene.attached_radii, [r])              # the query sphere is not inflated


# =============================================================================================
# H3 — the held object's old traces are set free in the TSDF (pipeline side; the TSDF write is
# tested on the GPU in `test_t32_fine_truncation_curobo.py`)
# =============================================================================================


def _hand_ag3s(monkeypatch):
    from tests.ag3s import test_fine_window as FW

    class _HandWithPose(FW._HandRobot):
        def link_pose_numeric(self, q, link):
            q = np.asarray(q, np.float64).reshape(-1)
            T = np.eye(4)
            i = self.sphere_link_names.index(link)
            T[:3, 3] = self.BASE[i] + np.array([q[self.JOINT[i]] if self.JOINT[i] >= 0 else 0, 0, 0])
            return T

    ag3s = FW._ag3s(monkeypatch)
    ag3s.robot_model = ag3s.constraint_robot_model = _HandWithPose()
    return ag3s, FW


def test_held_frames_free_the_held_volume_instead_of_excluding_seeds(monkeypatch):
    from benchmark.ag3s.types import AttachedCollisionGeometry, Primitive, PrimitiveType

    ag3s, FW = _hand_ag3s(monkeypatch)
    # a pre-grasp frame: the object's surface fit is noted (its place on the table)
    FW._build(ag3s)
    assert ag3s._last_object_fit is not None                                       # noqa: SLF001
    assert np.linalg.norm(ag3s._last_object_fit[0] - FW.APPLE) < 1e-3               # noqa: SLF001
    assert "held_free_points" not in ag3s._esdf_builder.calls[-1]
    # held: a sphere riding ee_left, reaching below the table top and next to a "crate" point
    local = FW.APPLE - FW._HandRobot.BASE[1]
    prim = Primitive(type=PrimitiveType.SPHERE, center=local, dimensions=np.full(3, 0.039),
                     semantic_role=HELD_ROLE)
    # (this toy robot has no gripper, so `attach` would refuse on admissibility — set it directly)
    ag3s._attached = AttachedCollisionGeometry(                                    # noqa: SLF001
        parent_link="ee_left", T_parent_object=np.eye(4), primitives=[prim],
        points=local + 0.03 * np.eye(3))
    crate = FW.APPLE + np.array([0.039, 0.0, 0.0])
    ag3s._build_esdf([object()], None, support_surfaces=[FW.TABLE], robot_state=np.zeros(2),  # noqa: SLF001
                     contact_context=FW.LEFT, destination_points=crate[None])
    call = ag3s._esdf_builder.calls[-1]
    pts = call["held_free_points"]
    assert call["attached_points"] is None                       # seed exclusion retired (H3)
    assert len(pts) > 100                                   # half a (20 mm) TSDF voxel spacing here
    assert pts[:, 2].min() >= FW.TABLE_Z + ag3s._guard_pad() - 1e-12              # noqa: SLF001
    assert np.linalg.norm(pts - crate, axis=1).min() >= 2 * ag3s._guard_pad() - 1e-12  # noqa: SLF001
    assert (np.linalg.norm(pts - FW.APPLE, axis=1) <= 0.039 + 1e-9).all()
    # switch off → the pre-T32 seed exclusion path again
    ag3s.free_held_traces = False
    ag3s._build_esdf([object()], None, support_surfaces=[FW.TABLE], robot_state=np.zeros(2),  # noqa: SLF001
                     contact_context=FW.LEFT)
    call = ag3s._esdf_builder.calls[-1]
    assert "held_free_points" not in call and call["attached_points"] is not None


def test_attach_arms_the_first_frames_with_the_pregrasp_place_and_the_capture_pose(rby1):
    q = np.zeros(rby1.nq)
    c = _apple_between_fingers(rby1, q)
    ag3s = _rby1_ag3s(rby1, ("ee_left", "ee_right"))
    ag3s._last_object_fit = (c + np.array([0.0, 0.0, -0.012]), 0.033)              # noqa: SLF001
    ag3s.attach(_target(_cap(c, view=(1.0, 0.0, 0.5))), robot_state=q, parent_link="ee_finger_l1")
    extra = ag3s._held_first_free                                                  # noqa: SLF001
    assert ag3s.held_record["first_free_spheres"] == len(extra) >= 2
    assert np.allclose(extra[-1][0], c + [0.0, 0.0, -0.012]) and extra[-1][1] == pytest.approx(
        0.033 + ag3s._guard_pad())                                                  # noqa: SLF001
    n = [len(ag3s._held_free_points(q, (), None)) for _ in range(3)]               # noqa: SLF001
    assert n[0] == n[1] > n[2] > 0                                  # two frames with the extras
    ag3s.detach()
    assert ag3s._held_first_free == []                                             # noqa: SLF001


# =============================================================================================
# T32b S3 wired into AG3S — the hands reach `TargetConfirm.note_hand` on every frame
# =============================================================================================


@pytest.fixture(scope="module")
def hand_model():
    """RB-Y1 palms + fingers with a capsule each (the bare URDF has none on the gripper; the server
    adds them from the MuJoCo meshes) — enough for `hand_spheres` and for the finger joints to move."""
    from benchmark.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf
    from benchmark.ag3s.robot_models.urdf_sphere_chain import UrdfCapsule

    links = ["ee_left", "ee_right"] + [f"ee_finger_{s}{i}" for s in ("l", "r") for i in (1, 2)]
    caps = [UrdfCapsule(link=l, origin=np.eye(4), radius=0.012 if l.startswith("ee_f") else 0.03,
                        length=0.04) for l in links]
    return UrdfSphereChain(parse_urdf(RBY1_URDF), extra_capsules=caps, link_filter=links)


def _s3_ag3s(rby1):
    from benchmark.ag3s.runtime.pipeline import AG3S

    cfg = AG3SConfig().with_overrides({"collision_backend": "esdf",
                                       "constraint": {"horizon": 2, "max_attached_primitives": 4},
                                       "clustering": {"gripper_max_opening": MAX_OPENING}})
    return AG3S(cfg, robot_model=rby1, constraint_robot_model=rby1,
                attached_parent_links=("ee_left", "ee_right"))


def _scripted_grounding(monkeypatch, frames, seen):
    """AG3S grounds its own cloud; here the frames are scripted (the T32b S3 scene) while
    everything else — freeze, note_hand, the confirm object — is AG3S's own."""
    import benchmark.ag3s.runtime.pipeline as P
    from benchmark.ag3s.stages.target_grounding import ground_target as real

    it = iter(frames)

    def spy(attention_cloud, cfg, **kw):
        seen.append(kw["confirm"]._hand)                                          # noqa: SLF001
        return real(next(it), cfg, seed_threshold=0.3, confirm=kw["confirm"])

    monkeypatch.setattr(P, "ground_target", spy)


def _s3_frames(apple_c):
    from benchmark.ag3s.types import AttentionPointCloud, PointCloud

    def sphere(c, r, seed):
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(400, 3))
        return np.asarray(c) + r * v / np.linalg.norm(v, axis=1, keepdims=True)

    def apc(pts):
        att = np.full(len(pts), 0.9, np.float32) + np.linspace(0, 1e-4, len(pts), dtype=np.float32)
        return AttentionPointCloud(PointCloud(pts), att, att)

    apple = sphere(apple_c, 0.031, 0)
    pear = sphere(apple_c + np.array([0.0, -0.150, 0.0]), 0.030, 2)
    return [apc(apple)] + [apc(pear)] * 5


def test_ag3s_hands_the_constraint_hand_with_the_measured_fingers_to_grounding(hand_model, monkeypatch):
    rby1 = hand_model
    from benchmark.ag3s.stages.admissibility import HandSpheres, hand_links

    ag3s = _s3_ag3s(rby1)
    seen = []
    q = np.zeros(rby1.nq)
    _scripted_grounding(monkeypatch, _s3_frames(_apple_between_fingers(rby1, q)), seen)
    cloud = np.zeros((10, 3)) + [2.0, 2.0, 2.0]
    ag3s.set_finger_joints({"gripper_finger_l1": -0.045, "gripper_finger_l2": 0.045})   # open
    ag3s.process(pointcloud=cloud, robot_state=q)
    hand = seen[-1]
    assert isinstance(hand, HandSpheres) and set(hand.links) <= set(hand_links(rby1))
    centres, radii = rby1.sphere_centers_numeric(q)
    keep = np.array([n in set(hand_links(rby1)) for n in rby1.sphere_link_names])
    assert np.allclose(hand.centres, centres[keep]) and np.allclose(hand.radii, radii[keep])
    open_c = hand.centres.copy()
    ag3s.set_finger_joints({"gripper_finger_l1": -0.010, "gripper_finger_l2": 0.010})   # closed
    ag3s.process(pointcloud=cloud, robot_state=q)
    assert not np.allclose(seen[-1].centres, open_c)          # the fingers moved with the joints
    # a held object's filter spheres are not the hand
    ag3s.attach(_target(_cap(_apple_between_fingers(rby1, q))), robot_state=q,
                parent_link="ee_finger_l1")
    ag3s.process(pointcloud=cloud, robot_state=q)
    assert not any(str(n).startswith("held:") for n in seen[-1].links)


@pytest.mark.parametrize("wired", [True, False])
def test_s3_is_active_in_the_ag3s_path(hand_model, monkeypatch, wired):
    """T32b S3 through AG3S: the manipulated apple sits between the fingers and disappears; a pear
    150 mm away leads. With the wiring the hand blocks the count (no switch in 5 frames); with the
    hand withheld the pear takes over on frame 3 (the T31b behaviour)."""
    rby1 = hand_model
    ag3s = _s3_ag3s(rby1)
    q = np.zeros(rby1.nq)
    apple_c = _apple_between_fingers(rby1, q)
    seen = []
    _scripted_grounding(monkeypatch, _s3_frames(apple_c), seen)
    if not wired:
        monkeypatch.setattr(ag3s, "_hand_for_grounding", lambda state: None)
    cloud = np.zeros((10, 3)) + [2.0, 2.0, 2.0]
    ag3s.process(pointcloud=cloud, robot_state=q)                   # adopt the apple
    first = ag3s.manipulated.id
    modes = []
    for _ in range(5):
        ag3s.process(pointcloud=cloud, robot_state=q)
        modes.append(ag3s._target_confirm.last.mode)                               # noqa: SLF001
    if wired:
        assert ag3s.manipulated.id == first and "switch" not in modes
        assert ag3s.manipulated.record()["hand"]["occluded_by_hand"] is True
    else:
        assert modes[2] == "switch" and ag3s.manipulated.id != first
