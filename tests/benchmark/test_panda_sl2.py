"""SL2: Panda geometry -- adapter URDF FK vs the LIBERO MuJoCo model, sphere models, link sets (CPU)."""
import ast
import json
import math
import subprocess

import numpy as np
import pytest

from AG3S.adapters.panda import cover
from AG3S.adapters.panda import model as pm
from benchmark.runners import launch_libero_pi05_only as launcher
from common.workspace import PROJECT_ROOT

LIBERO_PYTHON = PROJECT_ROOT / ".venv-libero/bin/python"
ROBOT_GEOMETRY = PROJECT_ROOT / "benchmark/adapters/libero_panda/robot_geometry.py"
FRANKA_COUNTS = {"panda_link0": 2, "panda_link1": 4, "panda_link2": 4, "panda_link3": 4, "panda_link4": 4,
                 "panda_link5": 13, "panda_link6": 3, "panda_link7": 5, "panda_hand": 18, "panda_leftfinger": 2,
                 "panda_rightfinger": 2}


def _rot_err_rad(Ra, Rb):
    """Rotation angle of ``Ra^T Rb`` (atan2 form: acos near 1 loses half the digits)."""
    D = Ra.T @ Rb
    skew = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]])
    return math.atan2(float(np.linalg.norm(skew)), float(np.trace(D) - 1.0))


# --- .venv-libero fixture: the robot as robosuite builds it (FK at 4 q, geometry export) -----------------
@pytest.fixture(scope="module")
def mujoco_robot(tmp_path_factory):
    if not LIBERO_PYTHON.exists():
        pytest.skip(".venv-libero not installed")
    out = tmp_path_factory.mktemp("sl2_py38")
    env = launcher.client_environment()
    arrays = {}
    for what in ("fk", "export"):
        path = out / f"{what}.npz"
        done = subprocess.run([str(LIBERO_PYTHON), str(ROBOT_GEOMETRY), what, "--source", "robosuite", "--out", str(path)],
                              cwd=str(out), env=env, capture_output=True, text=True, timeout=300)
        assert done.returncode == 0 and path.exists(), done.stdout[-2000:] + done.stderr[-3000:]
        arrays[what] = cover.load_geometry(path)
    return arrays


def test_fk_matches_libero_mujoco_bodies_sites_camera(mujoco_robot):
    fk = mujoco_robot["fk"]
    chain = pm.build_panda_constraint_model(None)
    bodies = fk["body_names"].tolist()
    assert fk["q_layout"].tolist() == [a for a, _, _ in pm.JOINT_MAP]
    worst_p, worst_r = 0.0, 0.0
    pairs = [("T_base_body", bodies.index(b), link) for b, link in pm.BODY_TO_LINK.items()]
    pairs += [("T_base_site", fk["site_names"].tolist().index(s), link) for s, link in pm.SITE_TO_LINK.items()]
    pairs += [("T_base_camera_opencv", fk["camera_names"].tolist().index(c), link) for c, link in pm.CAMERA_TO_LINK.items()]
    assert len(pairs) == 14
    for k, q9 in enumerate(fk["q"]):
        q7, fingers = pm.split_robot_state(q9)
        params = chain.param_vector(fingers)
        for key, index, link in pairs:
            T_mj = fk[key][k][index]
            T_urdf = chain.link_pose(q7, link, params=params)
            worst_p = max(worst_p, float(np.linalg.norm(T_mj[:3, 3] - T_urdf[:3, 3])))
            worst_r = max(worst_r, _rot_err_rad(T_mj[:3, :3], T_urdf[:3, :3]))
    # Same kinematics, not an approximation: float round-off only.
    assert worst_p < 1e-9 and worst_r < 1e-8, (worst_p, worst_r)


def test_curobo_urdf_differs_only_by_the_documented_hand_offsets(mujoco_robot):
    from AG3S.ag3s.robot_models.urdf_sphere_chain import UrdfSphereChain, parse_urdf

    ours, theirs = pm.load_panda_urdf(), parse_urdf(pm.CUROBO_FRANKA_URDF)
    for name in pm.PANDA_ARM_JOINTS:
        a, b = ours.joint_by_name(name), theirs.joint_by_name(name)
        assert (a.parent, a.child, a.type, a.lower, a.upper, a.velocity) == (b.parent, b.child, b.type, b.lower, b.upper,
                                                                            b.velocity), name
        np.testing.assert_array_equal(a.origin, b.origin)
        np.testing.assert_array_equal(a.axis, b.axis)
    assert theirs.joint_by_name("panda_joint8").origin[2, 3] == 0.107
    assert ours.joint_by_name("panda_joint8").origin[2, 3] == 0.1065
    for name in pm.PANDA_FINGER_JOINTS:
        assert theirs.joint_by_name(name).origin[2, 3] == 0.0584 and ours.joint_by_name(name).origin[2, 3] == 0.0524
    # Why the adapter URDF exists: the unmodified cuRobo URDF is off at the hand (0.5 mm) and fingers (6.5 mm).
    fk = mujoco_robot["fk"]
    plain = UrdfSphereChain(theirs, pm.PANDA_ARM_JOINTS)
    bodies = fk["body_names"].tolist()
    errors = {}
    for body, link in (("robot0_link7", "panda_link7"), ("robot0_right_hand", "panda_hand"),
                       ("gripper0_leftfinger", "panda_leftfinger")):
        q7, fingers = pm.split_robot_state(fk["q"][0])
        T = plain.link_pose(q7, link, params=plain.param_vector(fingers))
        errors[link] = float(np.linalg.norm(fk["T_base_body"][0][bodies.index(body)][:3, 3] - T[:3, 3]))
    assert errors["panda_link7"] < 1e-9
    assert errors["panda_hand"] == pytest.approx(0.0005, abs=1e-6)
    assert errors["panda_leftfinger"] == pytest.approx(0.0065, abs=1e-5)


def test_hand_yaw_is_the_robosuite_quaternion():
    R = pm.load_panda_urdf().joint_by_name("panda_hand_joint").origin[:3, :3]
    w, z = 0.924 / math.hypot(0.924, 0.383), -0.383 / math.hypot(0.924, 0.383)  # robot.xml quat "0.924 0 0 -0.383"
    expected = np.array([[1 - 2 * z * z, -2 * z * w, 0.0], [2 * z * w, 1 - 2 * z * z, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(R, expected, atol=1e-15)


def test_geometry_export_is_the_one_the_committed_cover_was_fitted_from(mujoco_robot):
    from benchmark.adapters.libero_panda import robot_geometry  # py3.8-compatible module, numpy only at import

    export = mujoco_robot["export"]
    _, document = pm.load_self_filter_cover()
    prov = document["provenance"]
    assert prov["geometry_export"]["source"] == "libero" and prov["surfaces"] == list(cover.DEFAULT_SURFACES)
    # robosuite-only robot == the LIBERO env's robot (kinematics and every mesh), base placement aside.
    assert str(export["geometry_digest"]) == prov["geometry_digest"]
    assert str(export["kinematic_digest"]) == prov["kinematic_digest"]
    assert robot_geometry.tables_digest(export, robot_geometry.GEOMETRY_KEYS) == prov["geometry_digest"]
    assert prov["urdf_sha256"] == pm.file_sha256(pm.PANDA_URDF)
    assert prov["franka_yml_sha256"] == pm.file_sha256(pm.CUROBO_FRANKA_YML)


def test_cover_refits_to_the_committed_capsules(mujoco_robot):
    capsules, per_body = cover.fit_self_filter_cover(mujoco_robot["export"], pm.build_panda_bare_model())
    committed, document = pm.load_self_filter_cover()
    assert per_body == document["per_body"]
    assert len(capsules) == len(committed) == 51
    for a, b in zip(capsules, committed):
        assert (a.link, a.role) == (b.link, b.role)
        np.testing.assert_allclose(a.origin, b.origin, atol=1e-12)
        assert a.radius == pytest.approx(b.radius, abs=1e-12) and a.length == pytest.approx(b.length, abs=1e-12)


def test_cover_contains_every_collision_and_visual_surface_sample(mujoco_robot):
    """The self-filter model (spheres + cover) contains the sampled robosuite surfaces of every body,
    collision and visual (4 mm sampling, link frames; what the V11AB fitter guarantees, re-checked)."""
    import mujoco

    from AG3S.ag3s.experiments.sources import mujoco_source as ms

    model = pm.build_panda_self_filter_model()
    for surfaces in (("collision",), ("visual",)):
        mj, geoms = cover.build_mujoco_model(mujoco_robot["export"], surfaces)
        for body in geoms:
            link, T = ms.static_urdf_ancestor(mj, body, model.model.links)
            pts = ms._body_surface_points(mj, mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, body))
            pts = pts @ T[:3, :3].T + T[:3, 3]
            c, r = model.local_spheres(link)
            worst = -np.inf
            for k in range(0, len(pts), 20000):
                d = np.linalg.norm(pts[k:k + 20000, None, :] - c[None], axis=2) - r[None]
                worst = max(worst, float(d.min(axis=1).max()))
            assert worst <= 1e-9, (surfaces, body, worst)


# --- sphere models -------------------------------------------------------------------------------------
def test_franka_spheres_source_counts_and_radii():
    spheres = pm.franka_collision_spheres()
    assert {k: len(v) for k, v in spheres.items()} == FRANKA_COUNTS
    assert sum(FRANKA_COUNTS.values()) == 61
    assert all(r > 0.0 for v in spheres.values() for _, r in v)
    capsules = pm.franka_sphere_capsules(pm.load_panda_urdf().links)
    assert len(capsules) == 61 and all(c.length == 0.0 and c.role == "collision" for c in capsules)
    with pytest.raises(ValueError, match="not a link"):
        pm.franka_sphere_capsules(["panda_link1"])


def test_constraint_and_self_filter_models_have_the_v11ab_structure():
    from AG3S.ag3s.robot_models.urdf_sphere_chain import COVER_ROLE

    filt = pm.build_panda_self_filter_model()
    arms = pm.build_panda_constraint_model()
    grip = pm.build_panda_constraint_model(pm.panda_constraint_link_filter("gripper"))
    whole = pm.build_panda_constraint_model(pm.panda_constraint_link_filter("all"))
    assert whole.n_spheres == 61 and arms.n_spheres == 61 - 2 and grip.n_spheres == 18 + 2 + 2
    assert set(arms.sphere_link_names) == set(pm.PANDA_ARM_LINKS)
    assert set(grip.sphere_link_names) == set(pm.PANDA_GRIPPER_LINKS)
    for m in (arms, grip, whole):
        assert COVER_ROLE not in m.sphere_roles and np.all(m.radii > 0.0)
    counts = pm.sphere_counts(filt)
    assert filt.n_spheres == 61 + sum(counts[link].get(COVER_ROLE, 0) for link in counts)
    assert sum(1 for role in filt.sphere_roles if role == COVER_ROLE) > 0 and np.all(filt.radii > 0.0)
    assert set(filt.sphere_link_names) == set(pm.PANDA_ALL_LINKS)
    # The constraint spheres are the filter's non-cover spheres, at any q and finger opening.
    rng = np.random.default_rng(3)
    lo, hi = whole.joint_limits()
    for _ in range(3):
        q, fingers = rng.uniform(lo, hi), {"panda_finger_joint1": 0.03, "panda_finger_joint2": 0.01}
        for m in (filt, whole):
            m.set_joint_parameters(fingers)
        cf, rf = filt.sphere_centers_numeric(q)
        keep = np.array([role != COVER_ROLE for role in filt.sphere_roles])
        cw, rw = whole.sphere_centers_numeric(q)
        np.testing.assert_array_equal(cf[keep], cw)
        np.testing.assert_array_equal(rf[keep], rw)
    # Finger joints are T29 parameter joints of one parallel gripper on the palm.
    pairs = whole.gripper_joint_pairs()
    assert [p["parent_link"] for p in pairs] == ["panda_hand"] and set(pairs[0]["joints"]) == set(pm.PANDA_FINGER_JOINTS)
    assert whole.param_joint_names == pm.PANDA_FINGER_JOINTS and whole.nq == 7


def test_robot_state_maps_to_q_and_finger_parameters():
    rs = np.array([0.1, 0.2, 0.3, -1.4, 0.5, 1.6, 0.7, 0.031, -0.029])
    q, fingers = pm.split_robot_state(rs)
    np.testing.assert_array_equal(q, rs[:7])
    assert fingers == {"panda_finger_joint1": 0.031, "panda_finger_joint2": 0.029}
    with pytest.raises(ValueError):
        pm.split_robot_state(rs[:8])
    model = pm.build_panda_constraint_model(pm.PANDA_FINGER_LINKS)
    q7 = pm.set_finger_parameters(model, rs)
    np.testing.assert_array_equal(q7, rs[:7])
    left, right = (model.link_pose(q7, link) for link in pm.PANDA_FINGER_LINKS)
    hand = model.link_pose(q7, "panda_hand")
    gap = (np.linalg.inv(hand) @ left)[:3, 3] - (np.linalg.inv(hand) @ right)[:3, 3]
    np.testing.assert_allclose(gap, [0.0, 0.031 + 0.029, 0.0], atol=1e-12)  # open = +y left, -y right


# --- link sets -----------------------------------------------------------------------------------------
def test_link_sets_are_consistent_and_typos_raise():
    from AG3S.ag3s.experiments.reports import grounding_report as gr

    links = set(pm.load_panda_urdf().links)
    sphere_links = set(pm.franka_collision_spheres())
    for group in (pm.PANDA_PALM_LINKS, pm.PANDA_FINGER_LINKS, pm.PANDA_GRIPPER_LINKS, pm.PANDA_ARM_LINKS,
                  pm.PANDA_ALL_LINKS, *pm.PANDA_CONTACT_LINKS.values()):
        assert set(group) <= links and set(group) <= sphere_links and len(set(group)) == len(group)
    assert pm.PANDA_GRIPPER_LINKS == pm.PANDA_PALM_LINKS + pm.PANDA_FINGER_LINKS
    contact = set(pm.PANDA_CONTACT_LINKS[pm.MANIPULATOR_SIDE])
    assert contact <= set(pm.PANDA_GRIPPER_LINKS) <= set(pm.PANDA_ARM_LINKS) < set(pm.PANDA_ALL_LINKS)
    assert "panda_link0" not in pm.PANDA_ARM_LINKS and set(pm.PANDA_ALL_LINKS) == sphere_links
    assert list(pm.PANDA_CONTACT_LINKS) == ["left"] == [pm.MANIPULATOR_SIDE]
    # Every arm joint moves exactly one ARM link (its child), the base link0 is moved by none.
    urdf = pm.load_panda_urdf()
    assert [urdf.joint_by_name(j).child for j in pm.PANDA_ARM_JOINTS] == [f"panda_link{i}" for i in range(1, 8)]
    # The RB-Y1 column of the role table is the V11AB definition, not a copy that can drift.
    table = {row["role"].split()[0]: row for row in pm.LINK_SET_TABLE}
    assert table["PALM_LINKS"]["rby1"] == gr.PALM_LINKS and table["FINGER_LINKS"]["rby1"] == gr.FINGER_LINKS
    from AG3S.ag3s.config import DEFAULT_CONTACT_LINKS

    assert table["DEFAULT_CONTACT_LINKS"]["rby1"] == DEFAULT_CONTACT_LINKS
    assert set(pm.PANDA_LINK_GROUPS) == {"gripper", "arms", "all"}
    with pytest.raises(ValueError, match="no Panda sphere"):
        pm.build_panda_constraint_model(("panda_hand", "panda_lefftfinger"))
    with pytest.raises(ValueError, match="empty"):
        pm.build_panda_constraint_model(())
    with pytest.raises(ValueError, match="links must be one of"):
        pm.panda_constraint_link_filter("gripers")


def test_panda_depth_robot_mask_removes_robot_points_only():
    from AG3S.ag3s.config import PointCloudConfig
    from benchmark.adapters.libero_panda import camera

    model = pm.build_panda_self_filter_model()
    rs = np.array([0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, 0.7853981633974483, 0.04, -0.04])
    q7 = pm.set_finger_parameters(model, rs)
    hand = model.link_pose(q7, "panda_hand")
    # A camera 0.6 m in front of the hand looking back at it; robot depth = the hand origin, rest = far wall.
    H = W = 64
    K = camera.pixel_centre_intrinsic_matrix(60.0, H, W)
    eye = hand[:3, 3] + np.array([0.6, 0.0, 0.0])
    z = (hand[:3, 3] - eye) / np.linalg.norm(hand[:3, 3] - eye)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    T_base_cam = camera.pose(eye, np.column_stack([x, np.cross(z, x), z]))
    depth = np.full((H, W), 2.5)
    depth[30:34, 30:34] = 0.6
    mask = pm.panda_depth_robot_mask(depth, K, T_base_cam, rs, model, PointCloudConfig())
    assert mask.mask.shape == (H, W) and mask.mask[30:34, 30:34].all() and mask.mask.sum() == 16
    assert model.joint_parameters == {"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04}


def test_manifest_entry_and_camera_conventions_agree():
    from benchmark.adapters.libero_panda import camera

    entry = pm.manifest_entry()
    json.dumps(entry)
    assert entry["manipulator_side"] == camera.MANIPULATOR_SIDE == "left"
    assert [row["robot_state"] for row in entry["joint_map"]] == list(camera.ROBOT_STATE_LAYOUT)
    assert entry["constraint_spheres"]["count"] == 61 and entry["self_filter_cover"]["count"] == 51
    assert camera.conventions()["manipulator_side"]["panda"] == pm.MANIPULATOR_SIDE


def test_py38_sources_parse_as_python38():
    for path in ("benchmark/adapters/libero_panda/robot_geometry.py", "benchmark/adapters/libero_panda/camera.py",
                 "tests/benchmark/libero_py38_checks.py"):
        ast.parse((PROJECT_ROOT / path).read_text(), feature_version=(3, 8))
