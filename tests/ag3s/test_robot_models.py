"""Robot-model adapter tests, including the self-filter.

The single most important assertion here is that `sphere_centers_numeric` and
`sphere_centers_symbolic` describe the *same robot*. They are used by different stages — the
self-filter deletes points with the numeric one, the constraint builder writes inequalities with the
symbolic one — so a disagreement would delete points as "robot" that the constraints then treat as
free space, and nothing would notice until something hit something.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.ag3s.config import DEFAULT_CONTACT_LINKS, PointCloudConfig
from benchmark.ag3s.stages.reconstruction import backproject, reconstruct
from benchmark.ag3s.stages.robot_filter import filter_robot_points, robot_sphere_mask
from benchmark.ag3s.robot_models import (
    RBY1_URDF,
    UrdfCapsule,
    UrdfSphereChain,
    load_rby1,
    parse_urdf,
)
from benchmark.ag3s.types import PointCloud, RobotCollisionModel
from tests.ag3s.fixtures import (
    DEFAULT_ARM_Q,
    FINGERTIP_LINK,
    LABEL_BOX,
    LABEL_CYLINDER,
    LABEL_ROBOT,
    LABEL_SPHERE,
    LABEL_TABLE,
    PlanarTwoLinkArm,
    make_scene,
)

casadi = pytest.importorskip("casadi", reason="constraint side needs CasADi")

rby1_available = pytest.mark.skipif(
    not RBY1_URDF.exists(), reason=f"RB-Y1 URDF not found at {RBY1_URDF} (run from the workspace root)"
)


# ---------------------------------------------------------------------------- analytic mock


def test_mock_satisfies_the_protocol():
    assert isinstance(PlanarTwoLinkArm(), RobotCollisionModel)


def test_mock_fk_is_analytically_correct():
    """Straight-out arm along +x: the last sphere sits at base + total length."""
    arm = PlanarTwoLinkArm(base=(0.0, 0.0, 0.0), link_lengths=(0.3, 0.2), spheres_per_link=1)
    centres, radii = arm.sphere_centers_numeric(np.zeros(2))
    assert np.allclose(centres[0], [0.3, 0.0, 0.0])
    assert np.allclose(centres[1], [0.5, 0.0, 0.0])
    assert np.allclose(radii, arm.sphere_radius)

    # Second joint folded 90 degrees: the tip turns upward by link 1's length.
    centres, _ = arm.sphere_centers_numeric(np.array([0.0, np.pi / 2]))
    assert np.allclose(centres[1], [0.3, 0.0, 0.2], atol=1e-12)


def test_mock_numeric_and_symbolic_fk_agree():
    arm = PlanarTwoLinkArm()
    q = casadi.SX.sym("q", arm.nq)
    f = casadi.Function("fk", [q], [casadi.horzcat(*[c for c, _ in arm.sphere_centers_symbolic(q)])])
    rng = np.random.default_rng(0)
    for _ in range(20):
        q_val = rng.uniform(-np.pi, np.pi, arm.nq)
        num, _ = arm.sphere_centers_numeric(q_val)
        assert np.abs(num - np.asarray(f(q_val)).T).max() < 1e-12


def test_mock_rejects_wrong_sized_q():
    with pytest.raises(ValueError, match="2 entries"):
        PlanarTwoLinkArm().sphere_centers_numeric(np.zeros(3))


# --------------------------------------------------------------------------- RB-Y1 from URDF


@rby1_available
def test_urdf_parses_links_joints_and_commented_out_capsules():
    """RB-Y1 ships its collision capsules inside XML comments; they must still be read."""
    model = parse_urdf(RBY1_URDF)
    assert model.root == "base"
    assert len(model.links) == 34
    assert len(model.capsules) == 17
    assert {"torso_0", "right_arm_0", "left_arm_6"} <= set(model.movable_joint_names())
    for cap in model.capsules:
        assert cap.radius > 0.0 and cap.length >= 0.0
        assert cap.origin.shape == (4, 4)


@rby1_available
def test_rby1_sphere_chain_is_physically_plausible():
    robot = load_rby1()
    assert robot.nq == 20  # 6 torso + 7 + 7 arm
    centres, radii = robot.sphere_centers_numeric(np.zeros(robot.nq))
    assert centres.shape == (robot.n_spheres, 3)
    # RB-Y1 is a ~1.4 m mobile manipulator on a wheeled base.
    assert 0.2 < centres[:, 2].min() < 0.5
    assert 1.2 < centres[:, 2].max() < 1.6
    assert 0.03 < radii.min() and radii.max() < 0.20


@rby1_available
def test_rby1_numeric_and_symbolic_fk_agree_to_machine_precision():
    """The property the whole two-backend design exists to guarantee."""
    robot = load_rby1()
    q = casadi.SX.sym("q", robot.nq)
    f = casadi.Function("fk", [q], [casadi.horzcat(*[c for c, _ in robot.sphere_centers_symbolic(q)])])
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(10):
        q_val = rng.uniform(-1.0, 1.0, robot.nq)
        num, _ = robot.sphere_centers_numeric(q_val)
        worst = max(worst, float(np.abs(num - np.asarray(f(q_val)).T).max()))
    assert worst < 1e-12, f"numeric and symbolic FK disagree by {worst:.3e} m"


# --------------------------------------------------------------- sphere <-> link metadata


def test_mock_link_names_line_up_with_its_spheres():
    arm = PlanarTwoLinkArm()
    centres, radii = arm.sphere_centers_numeric(DEFAULT_ARM_Q)
    names = arm.sphere_link_names
    assert len(names) == centres.shape[0] == radii.shape[0]
    assert names[-1] == FINGERTIP_LINK, "the distal sphere is the one contact tests authorize"
    assert names.count(FINGERTIP_LINK) == 1


@rby1_available
def test_rby1_link_names_align_with_both_fk_paths():
    """§26.3's core: one name per sphere, in the order **both** FK implementations emit.

    Length alone is not enough — a list of the right length in the wrong order would give the
    forearm the fingertip's clearance, and every distance would still be computed correctly. So this
    checks the alignment through geometry: each sphere must actually sit on the link its name
    claims, verified by transforming the sphere back into that link's frame with `link_pose` and
    finding a bounded offset. A shuffled list puts spheres metres away from their supposed link.
    """
    robot = load_rby1()
    names = robot.sphere_link_names
    q = np.zeros(robot.nq)
    centres, radii = robot.sphere_centers_numeric(q)

    assert len(names) == robot.n_spheres == centres.shape[0] == radii.shape[0]
    assert set(names) <= set(robot.model.links), f"unknown link names: {set(names) - set(robot.model.links)}"

    for i, link in enumerate(names):
        T = robot.link_pose(q, link)
        local = T[:3, :3].T @ (centres[i] - T[:3, 3])
        # A collision capsule lives on its own link, so its spheres sit within the link's own extent.
        # Half a metre is generous for RB-Y1's longest link and still far tighter than the metres a
        # mis-ordered list would produce.
        assert np.linalg.norm(local) < 0.5, (
            f"sphere {i} is named {link!r} but sits {np.linalg.norm(local):.3f} m from that frame"
        )

    # The symbolic path must emit the same count in the same order, since the constraint rows index
    # the two together.
    q_sym = casadi.SX.sym("q", robot.nq)
    assert len(robot.sphere_centers_symbolic(q_sym)) == len(names)


@rby1_available
def test_the_shipped_contact_links_exist_on_the_robot():
    """A typo in the allowlist is a silent loss of the ability to grasp, not an error.

    `ClearancePolicy` matches link names by string, and a name no sphere carries simply never
    matches — the target stays at full clearance and the grasp never closes. Checking the shipped
    defaults against the URDF is the only place that mistake surfaces.
    """
    robot = load_rby1()
    for manipulator, links in DEFAULT_CONTACT_LINKS.items():
        for link in links:
            assert link in robot.model.links, f"{manipulator} allowlist names a missing link {link!r}"


@rby1_available
def test_the_arm_only_urdf_model_carries_no_gripper_spheres():
    """A real limitation, recorded rather than papered over.

    RB-Y1's URDF ships collision capsules for the torso and arm links 0-5 and nothing else, so a
    constraint model built from it alone has **no sphere on any contact-authorized link**. The
    clearance policy then has nothing to relax and grasping is inexpressible — correctly, because
    there is no fingertip geometry to relax. The fix is `extra_capsules` (see
    `experiments/mujoco_source.gap_filling_capsules`), and this test pins the fact so the fix cannot
    be quietly dropped.
    """
    robot = load_rby1()
    authorized = {link for links in DEFAULT_CONTACT_LINKS.values() for link in links}
    assert not (set(robot.sphere_link_names) & authorized), (
        "the URDF gained gripper capsules; the extra_capsules workaround may no longer be needed"
    )


@rby1_available
def test_extra_capsules_make_the_gripper_addressable():
    """With fingertip capsules injected, the allowlist finds spheres and contact becomes expressible."""
    model = parse_urdf(RBY1_URDF)
    extra = [
        UrdfCapsule(link="ee_finger_r1", origin=np.eye(4), radius=0.02, length=0.04),
        UrdfCapsule(link="ee_finger_r2", origin=np.eye(4), radius=0.02, length=0.04),
    ]
    robot = UrdfSphereChain(model, extra_capsules=extra)
    names = set(robot.sphere_link_names)
    assert {"ee_finger_r1", "ee_finger_r2"} <= names
    assert len(robot.sphere_link_names) == robot.n_spheres


@rby1_available
def test_rby1_symbolic_link_pose_matches_the_numeric_one():
    """Attached-object geometry rides on this; a mismatch would move the held object off the hand."""
    robot = load_rby1()
    q_sym = casadi.SX.sym("q", robot.nq)
    rng = np.random.default_rng(3)
    for link in ("link_right_arm_6", "link_left_arm_6", "link_head_2"):
        f = casadi.Function("T", [q_sym], [robot.link_pose_symbolic(q_sym, link)])
        for _ in range(3):
            q_val = rng.uniform(-1.0, 1.0, robot.nq)
            assert np.abs(robot.link_pose(q_val, link) - np.asarray(f(q_val))).max() < 1e-12


@rby1_available
def test_rby1_symbolic_fk_is_differentiable():
    """A constant Jacobian would mean the chain lost its dependence on q."""
    robot = load_rby1()
    q = casadi.SX.sym("q", robot.nq)
    spheres = robot.sphere_centers_symbolic(q)
    J = casadi.jacobian(spheres[-1][0], q)
    assert J.shape == (3, robot.nq)
    assert J.nnz() > 0


@rby1_available
def test_rby1_joints_actually_move_the_geometry():
    robot = load_rby1()
    base, _ = robot.sphere_centers_numeric(np.zeros(robot.nq))
    q = np.zeros(robot.nq)
    q[6] = 0.8  # right_arm_0
    moved, _ = robot.sphere_centers_numeric(q)
    assert np.abs(moved - base).max() > 0.1


@rby1_available
def test_link_filter_restricts_the_sphere_set():
    full = load_rby1()
    arms_only = UrdfSphereChain(
        parse_urdf(RBY1_URDF), link_filter=[f"link_right_arm_{i}" for i in range(7)]
    )
    assert 0 < arms_only.n_spheres < full.n_spheres


@rby1_available
def test_the_sphere_chain_actually_covers_each_capsule():
    """Coverage, not spacing.

    Spacing alone does not imply coverage: at centre spacing `s`, the capsule-surface point midway
    between two spheres is sqrt(r^2 + (s/2)^2) from the nearest centre, so even `s = r` leaves 11.8%
    of the radius outside. The chain inflates its spheres to close that, and this samples the
    capsule surface densely to confirm it.
    """
    robot = load_rby1()
    for cap in robot.capsules:
        centres, radius, required = robot._capsule_sphere_centres(cap)  # noqa: SLF001
        assert radius >= cap.radius - 1e-12
        # 기본 설정에서는 '필요한 반지름' 과 '쓴 반지름' 이 같다 — 덮개를 잃는 설정이 아니다.
        assert radius == pytest.approx(required)

        # Sample the capsule's cylindrical surface in the capsule frame, then map to the link frame.
        t = np.linspace(-cap.length / 2.0, cap.length / 2.0, 40)
        theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        tt, th = np.meshgrid(t, theta, indexing="ij")
        local = np.stack(
            [cap.radius * np.cos(th).ravel(), cap.radius * np.sin(th).ravel(), tt.ravel()], axis=1
        )
        surface = local @ cap.origin[:3, :3].T + cap.origin[:3, 3]

        covered = np.zeros(surface.shape[0], bool)
        for centre in centres:
            covered |= np.linalg.norm(surface - centre, axis=1) <= radius + 1e-9
        assert covered.all(), (
            f"{cap.link}: {int((~covered).sum())}/{covered.size} capsule surface samples uncovered"
        )


# ------------------------------------------------------------------------------ self-filter


def test_sphere_mask_is_exact_on_a_known_configuration():
    centres = np.array([[0.0, 0.0, 0.0]])
    radii = np.array([0.1])
    pts = np.array([[0.0, 0.0, 0.0], [0.09, 0.0, 0.0], [0.11, 0.0, 0.0], [1.0, 1.0, 1.0]])
    mask = robot_sphere_mask(pts, centres, radii, inflation=0.0)
    assert mask.tolist() == [True, True, False, False]


def test_sphere_mask_inflation_widens_the_rejection():
    centres = np.array([[0.0, 0.0, 0.0]])
    radii = np.array([0.1])
    pts = np.array([[0.11, 0.0, 0.0]])
    assert not robot_sphere_mask(pts, centres, radii, 0.0)[0]
    assert robot_sphere_mask(pts, centres, radii, 0.02)[0]


def test_sphere_mask_handles_differing_radii():
    """A single-radius ball query would be wrong here; the small sphere must not over-reject."""
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    radii = np.array([0.5, 0.02])
    pts = np.array([[0.4, 0.0, 0.0], [0.9, 0.0, 0.0]])
    assert robot_sphere_mask(pts, centres, radii, 0.0).tolist() == [True, False]


def test_self_filter_removes_the_robot_and_keeps_the_scene():
    """The measurable claim: robot pixels go, object pixels stay.

    The fixture is built so the arm occludes nothing, so any object point lost here was lost by the
    filter and not by the renderer.
    """
    arm = PlanarTwoLinkArm()
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    cloud = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam)
    u, v = cloud.uv[:, 0], cloud.uv[:, 1]
    labels_before = scene.labels[v, u]

    cfg = PointCloudConfig(self_filter=True, self_filter_inflation=0.005)
    out, stats = filter_robot_points(cloud, arm, DEFAULT_ARM_Q, cfg)
    assert stats["enabled"] is True

    lu, lv = out.uv[:, 0], out.uv[:, 1]
    labels_after = scene.labels[lv, lu]

    robot_before = int((labels_before == LABEL_ROBOT).sum())
    robot_after = int((labels_after == LABEL_ROBOT).sum())
    assert robot_before > 5000, "fixture should place a substantial robot in view"
    assert robot_after / robot_before < 0.02, f"{robot_after}/{robot_before} robot points survived"

    for label in (LABEL_TABLE, LABEL_SPHERE, LABEL_BOX, LABEL_CYLINDER):
        before = int((labels_before == label).sum())
        after = int((labels_after == label).sum())
        assert after / before > 0.98, f"self-filter ate {before - after}/{before} of label {label}"


def test_self_filter_off_is_a_no_op():
    """`pointcloud.self_filter: false` is an ablation axis and must change nothing else."""
    arm = PlanarTwoLinkArm()
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    cloud, _ = reconstruct(
        depth=scene.depth,
        camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam,
        config=PointCloudConfig(voxel_size=0.01),
    )
    out, stats = filter_robot_points(cloud, arm, DEFAULT_ARM_Q, PointCloudConfig(self_filter=False))
    assert out is cloud and stats["enabled"] is False and stats["n_removed"] == 0


def test_self_filter_without_a_robot_model_is_a_no_op():
    """AG3S must stay usable with no robot model injected."""
    cloud = PointCloud(np.zeros((5, 3)))
    out, stats = filter_robot_points(cloud, None, np.zeros(2), PointCloudConfig(self_filter=True))
    assert out is cloud and stats["enabled"] is False


def test_self_filter_rejects_a_mismatched_robot_state():
    """Silently truncating `q` would filter against a different pose than the constraints use."""
    arm = PlanarTwoLinkArm()
    cloud = PointCloud(np.zeros((5, 3)))
    with pytest.raises(ValueError, match="nq=2"):
        filter_robot_points(cloud, arm, np.zeros(7), PointCloudConfig(self_filter=True))


def test_self_filter_preserves_uv():
    arm = PlanarTwoLinkArm()
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    cloud = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam)
    out, _ = filter_robot_points(cloud, arm, DEFAULT_ARM_Q, PointCloudConfig(self_filter=True))
    assert out.uv is not None and out.uv.shape == (len(out), 2)
