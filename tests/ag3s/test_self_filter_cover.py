"""T29 — self-filter 구 모델이 로봇을 덮는다 (inflation 이 아니라 구로).

T28 P1-3b: self-filter inflation 0 에서 로봇 GT px 가 36,037 (T14) / 54,106 (T17) 새었다
(`link_left_arm_2` · wrist bracket · D435i …). 그리고 T17 에서 사과가 통째로 지워진 17 프레임은
`link_left_arm_5` 의 URDF capsule (r 75 mm) 이 손가락 사이를 삼킨 탓이었다. 이 파일이 박는 것:

* 덮개 목록의 **모든 body 의 mesh 표면**(삼각형 안쪽까지)이 자기 필터 모델의 구 안에 있다 —
  inflation 0 으로. 렌더링 없이, link frame 에서.
* `link_*_arm_5` 의 URDF capsule 은 자기 필터 모델에 없다. 제약 모델은 **그대로**다 (구 수 ·
  역할 · arm_5 capsule).
* T26 max opening (71.46 mm) 은 덮개에 영향받지 않는다.
* 실제 렌더 한 장: 손을 연 teleop 자세에서 세 카메라의 로봇 누수가 0 에 가깝다.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

MODEL = pathlib.Path(
    "/mnt/dev/work/pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml")


@pytest.fixture(scope="module")
def mj():
    mujoco = pytest.importorskip("mujoco")
    if not MODEL.exists():  # pragma: no cover - asset-dependent
        pytest.skip(f"model not present: {MODEL}")
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    return mujoco, model


@pytest.fixture(scope="module")
def filter_model(mj):
    from benchmark.ag3s.experiments.reports.grounding_report import build_robot_model
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene

    mujoco, model = mj
    return build_robot_model(TransportScene.attach(model, mujoco.MjData(model)))


@pytest.fixture(scope="module")
def urdf_links():
    from benchmark.ag3s.robot_models import RBY1_URDF, parse_urdf

    return parse_urdf(RBY1_URDF).links


def test_static_ancestor_finds_the_urdf_link(mj, urdf_links):
    from benchmark.ag3s.experiments.sources.mujoco_source import static_urdf_ancestor

    _, model = mj
    assert static_urdf_ancestor(model, "link_left_arm_2", urdf_links)[0] == "link_left_arm_2"
    np.testing.assert_array_equal(static_urdf_ancestor(model, "link_left_arm_2", urdf_links)[1],
                                  np.eye(4))
    assert static_urdf_ancestor(model, "wrist_bracket_l", urdf_links)[0] == "link_left_arm_6"
    assert static_urdf_ancestor(model, "d435i_body_r", urdf_links)[0] == "link_right_arm_6"
    assert static_urdf_ancestor(model, "EE_BODY_L", urdf_links)[0] == "ee_left"
    # the left-finger name swap (T11): MJCF ee_finger_l1 is URDF ee_finger_l2
    assert static_urdf_ancestor(model, "ee_finger_l1", urdf_links)[0] == "ee_finger_l2"
    T = static_urdf_ancestor(model, "wrist_bracket_l", urdf_links)[1]
    np.testing.assert_allclose(T[:3, 3], [0.0, 0.0, -0.1548], atol=1e-9)
    with pytest.raises(ValueError, match="joint"):
        static_urdf_ancestor(model, "apple", urdf_links)      # free joint before any URDF link


def test_every_covered_body_surface_is_inside_the_self_filter_spheres(mj, filter_model, urdf_links):
    """inflation 0. 표면 = 정점 + 삼각형 안쪽 (4 mm 격자) — 카메라가 보는 것."""
    from benchmark.ag3s.experiments.sources.mujoco_source import (
        SELF_FILTER_COVER_BODIES,
        _body_surface_points,
        static_urdf_ancestor,
    )

    mujoco, model = mj
    worst = {}
    for body in SELF_FILTER_COVER_BODIES:
        link, T = static_urdf_ancestor(model, body, urdf_links)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        pts = _body_surface_points(model, bid) @ T[:3, :3].T + T[:3, 3]
        c, r = filter_model.local_spheres(link)
        d = (np.linalg.norm(pts[:, None, :] - c[None], axis=2) - r[None]).min(axis=1)
        worst[body] = float(d.max())
    assert max(worst.values()) <= 1e-9, {k: round(v * 1000, 3) for k, v in worst.items() if v > 0}


def test_arm5_urdf_capsule_is_replaced_in_the_filter_model_only(mj, filter_model):
    from benchmark.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from benchmark.ag3s.experiments.sources.mujoco_source import (
        SELF_FILTER_MESH_LINKS, TransportScene, gap_filling_capsules)
    from benchmark.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf
    from benchmark.ag3s.robot_models.urdf_sphere_chain import COVER_ROLE

    mujoco, model = mj
    for link in SELF_FILTER_MESH_LINKS:
        roles = {c.role for c in filter_model.capsules if c.link == link}
        assert roles == {COVER_ROLE}
    scene = TransportScene.attach(model, mujoco.MjData(model))
    cons = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    assert COVER_ROLE not in cons.sphere_roles
    assert any(c.link == "link_left_arm_5" and c.radius == pytest.approx(0.075)
               for c in cons.capsules)
    pre_t29 = UrdfSphereChain(parse_urdf(RBY1_URDF), extra_capsules=gap_filling_capsules(model),
                              link_filter=ARM_LINKS)
    assert cons.n_spheres == pre_t29.n_spheres
    np.testing.assert_array_equal(cons.radii, pre_t29.radii)


def test_the_cover_does_not_change_the_t26_max_opening(filter_model):
    from benchmark.ag3s.stages.admissibility import max_opening_from_models

    got = max_opening_from_models(filter_model)
    assert got.value_m * 1000.0 == pytest.approx(71.46, abs=0.01)
    assert len(filter_model.local_spheres("ee_finger_l1")[1]) > 11   # the cover *is* there


def test_rendered_leak_is_near_zero_with_open_hands(mj, filter_model):
    """teleop 자세 + 손을 데이터셋 open(−0.045)으로. inflation 0 에서 GT 로봇 px 가 거의 안 샌다."""
    from benchmark.ag3s.config import PointCloudConfig
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene
    from benchmark.ag3s.robot_models.gripper_state import FingerJointMap, openings_from_mujoco
    from benchmark.ag3s.stages.robot_filter import filter_robot_points
    from benchmark.ag3s.types import PointCloud

    mujoco, model = mj
    data = mujoco.MjData(model)
    scene = TransportScene.attach(model, data)
    scene.reset()
    for name, v in (("gripper_finger_l1", -0.045), ("gripper_finger_l2", 0.045),
                    ("gripper_finger_r1", -0.045), ("gripper_finger_r2", 0.045)):
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = v
    mujoco.mj_forward(model, data)
    filter_model.set_joint_parameters(
        FingerJointMap(filter_model).joint_values(openings_from_mujoco(model, data)))
    try:
        root = model.body_rootid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")]
        robot = np.array([b for b in range(model.nbody) if model.body_rootid[b] == root])
        q = scene.robot_state()
        leak = {}
        for cam in ("zed_left", "wrist_cam_l", "wrist_cam_r"):
            f = scene.capture(cam)
            vs, us = np.nonzero((f.body_ids >= 0) & np.isfinite(f.depth) & (f.depth > 0))
            K, z = f.camera_intrinsics, f.depth[vs, us]
            pc = np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], 1)
            pts = pc @ f.T_base_cam[:3, :3].T + f.T_base_cam[:3, 3]
            is_robot = np.isin(f.body_ids[vs, us], robot)
            kept, _ = filter_robot_points(PointCloud(points=pts[is_robot]), filter_model, q,
                                          PointCloudConfig(self_filter_inflation=0.0))
            leak[cam] = (len(kept), int(is_robot.sum()))
        assert all(n <= 50 for n, _ in leak.values()), leak
    finally:
        filter_model.set_joint_parameters(None)
