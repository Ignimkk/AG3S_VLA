"""T29 — 손가락 관절을 실제 개도로.

`UrdfSphereChain` 은 `q` 에 없는 관절을 `fixed_joint_values.get(name, 0.0)` 으로 고정했고 서버는 head
관절만 넘겼으므로 **손가락은 늘 0 = 닫힘**이었다 (T28 P1-3a: 모델 ↔ 실제 51 mm). 이 파일이 박는 것:

* 규약은 **클라이언트 파일에서 읽는다** — `RBY1_GRIPPER_OPEN` · `GRIPPER_L/R_JOINT` 와, 두 식
  (`|q| / |open|`, `clip(a, 0, 1) * open`) 이 그 파일에 그대로 있는지.
* 역함수가 **정확하다** — 클라이언트 식으로 정규화한 MuJoCo 개도를 되돌려 URDF FK 에 넣으면 네
  손가락이 MuJoCo body 와 0.01 mm 안에서 겹친다 (닫힘 · 중간 · 데이터셋 open · 한계).
* finger 관절은 parameter 다 — 기본값은 예전 모델 그대로, 모르는 이름은 거절, numeric 과 symbolic 이 같다.
* TO 창 모델 (`opening_envelope`) — 스텝 0 = 측정, 닫는 중·쥔 채는 측정값, 여는 중은 명령.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from benchmark.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf
from benchmark.ag3s.robot_models.gripper_state import (
    FingerJointMap,
    find_client_script,
    load_client_convention,
    opening_envelope,
    openings_from_mujoco,
)

MODEL = pathlib.Path(
    "/mnt/dev/work/pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml")
FINGERS = ("ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2")
FINGER_JOINTS = ("gripper_finger_l1", "gripper_finger_l2", "gripper_finger_r1", "gripper_finger_r2")


@pytest.fixture(scope="module")
def urdf():
    return parse_urdf(RBY1_URDF)


@pytest.fixture(scope="module")
def chain(urdf):
    return UrdfSphereChain(urdf)


@pytest.fixture(scope="module")
def client():
    try:
        return find_client_script()
    except FileNotFoundError:  # pragma: no cover - asset-dependent
        pytest.skip("client script not present")


# ------------------------------------------------------------------ 1. the client's convention


def test_the_convention_is_read_from_the_client_file(client):
    conv = load_client_convention(client)
    assert conv.open_command_m == pytest.approx(-0.045)
    assert dict(conv.driven_joints) == {"left": "gripper_finger_l1", "right": "gripper_finger_r1"}
    assert "pi05_infer.py" in conv.source and "RBY1_GRIPPER_OPEN@" in conv.source


def _function_source(tree, text, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError(f"no function {name!r} in the client")


def test_the_client_still_uses_the_two_formulas_this_module_inverts(client):
    """식이 바뀌면 역함수를 다시 유도해야 한다 — 이 테스트가 그날 실패한다."""
    text = client.read_text()
    tree = ast.parse(text)
    state = _function_source(tree, text, "rby1_state")
    assert 'abs(d.qpos[idx["left_grip_q"]]) / abs(RBY1_GRIPPER_OPEN)' in state
    assert 'abs(d.qpos[idx["right_grip_q"]]) / abs(RBY1_GRIPPER_OPEN)' in state
    apply = _function_source(tree, text, "apply_action")
    branch = apply[apply.index('action_format == "rby1_16d"'):]
    branch = branch[:branch.index("return")]
    assert "left_grip = float(np.clip(action[7], 0.0, 1.0))" in branch
    assert "right_grip = float(np.clip(action[15], 0.0, 1.0))" in branch
    assert 'd.ctrl[act["left_grip_a"]] = left_grip * RBY1_GRIPPER_OPEN' in branch


def test_driven_value_inverts_both_directions(client):
    conv = load_client_convention(client)
    assert conv.driven_value(1.0) == pytest.approx(-0.045)
    assert conv.driven_value(0.0) == 0.0
    assert conv.driven_value(1.084) == pytest.approx(-0.04878)          # measured may exceed 1
    assert conv.driven_value(1.3, command=True) == pytest.approx(-0.045)  # command is clipped
    assert conv.driven_value(-0.2, command=True) == 0.0
    for q in (0.0, -0.01, -0.045, -0.05):
        assert conv.driven_value(conv.normalize(q)) == pytest.approx(q)


# --------------------------------------------------------------------- 2. finger parameters


def test_finger_joints_are_parameters_with_the_old_default(urdf, chain):
    assert chain.param_joint_names == FINGER_JOINTS
    assert chain.joint_parameters == {n: 0.0 for n in FINGER_JOINTS}
    assert chain.joint_parameters_are_default
    # the same model built without the parameter machinery in mind: pinned at 0 → identical FK
    q = np.linspace(-0.4, 0.4, chain.nq)
    before = chain.sphere_centers_numeric(q)[0].copy()
    chain.set_joint_parameters({"gripper_finger_l1": -0.03})
    moved = chain.sphere_centers_numeric(q)[0]
    chain.set_joint_parameters(None)
    np.testing.assert_array_equal(chain.sphere_centers_numeric(q)[0], before)
    assert chain.joint_parameters_are_default
    # the pairs are structural — the chain has no finger capsules, but link poses move
    T0 = chain.link_pose(q, "ee_finger_l1")
    T1 = chain.link_pose(q, "ee_finger_l1", params=np.array([-0.03, 0.03, 0.0, 0.0]))
    assert np.linalg.norm(T1[:3, 3] - T0[:3, 3]) == pytest.approx(0.03)
    np.testing.assert_array_equal(moved, before)  # no finger spheres in the bare URDF


def test_unknown_parameter_names_are_refused(chain):
    with pytest.raises(ValueError, match="not parameter joints"):
        chain.set_joint_parameters({"gripper_finger_x1": -0.02})
    with pytest.raises(ValueError, match="not finite"):
        chain.set_joint_parameters({"gripper_finger_l1": float("nan")})
    with pytest.raises(ValueError, match="params must have 4"):
        chain.sphere_centers_numeric(np.zeros(chain.nq), params=np.zeros(3))
    chain.set_joint_parameters(None)


def test_gripper_pairs_open_and_closed_come_from_the_limits(chain):
    pairs = {p["parent_link"]: p for p in chain.gripper_joint_pairs()}
    assert set(pairs) == {"ee_left", "ee_right"}
    left = pairs["ee_left"]
    assert left["open"] == {"gripper_finger_l1": -0.05, "gripper_finger_l2": 0.05}
    assert left["closed"] == {"gripper_finger_l1": 0.0, "gripper_finger_l2": 0.0}


def test_symbolic_fk_with_params_matches_numeric(urdf):
    ca = pytest.importorskip("casadi")
    from benchmark.ag3s.robot_models.urdf_sphere_chain import UrdfCapsule

    finger = UrdfCapsule(link="ee_finger_l1", origin=np.eye(4), radius=0.01, length=0.04)
    model = UrdfSphereChain(urdf, extra_capsules=[finger])
    q = ca.SX.sym("q", model.nq)
    p = ca.SX.sym("p", model.n_params)
    fn = ca.Function("f", [q, p], [ca.horzcat(*[c for c, _ in model.sphere_centers_symbolic(q, params=p)])])
    qv = np.linspace(-0.3, 0.3, model.nq)
    pv = np.array([-0.04, 0.04, -0.01, 0.01])
    got = np.asarray(fn(qv, pv)).T
    np.testing.assert_allclose(got, model.sphere_centers_numeric(qv, params=pv)[0], atol=1e-12)
    # without params the current values are baked in (the pre-T29 path)
    model.set_joint_parameters({"gripper_finger_l1": -0.02})
    baked = ca.Function("g", [q], [ca.horzcat(*[c for c, _ in model.sphere_centers_symbolic(q)])])
    np.testing.assert_allclose(np.asarray(baked(qv)).T, model.sphere_centers_numeric(qv)[0],
                               atol=1e-12)


# -------------------------------------------------------- 3. inverse is exact (vs MuJoCo)


@pytest.fixture(scope="module")
def mj():
    mujoco = pytest.importorskip("mujoco")
    if not MODEL.exists():  # pragma: no cover - asset-dependent
        pytest.skip(f"model not present: {MODEL}")
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    scene = TransportScene.attach(model, data)
    scene.reset()
    return mujoco, model, data, scene


@pytest.mark.parametrize("q_open", [0.0, -0.02, -0.045, -0.05])
def test_client_norm_inverted_puts_urdf_fingers_on_the_mujoco_fingers(mj, chain, client, q_open):
    """T28 P1-3a 는 −0.045 에서 51.0 mm, 한계에서 56.0 mm 였다. 이제 0.01 mm 안이다."""
    from benchmark.ag3s.experiments.sources.mujoco_source import MJCF_BODY_ALIASES

    mujoco, model, data, scene = mj
    for name in FINGER_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        # MJCF equality `polycoef="0 -1"`: finger 2 = −finger 1
        data.qpos[model.jnt_qposadr[jid]] = q_open if name.endswith("1") else -q_open
    mujoco.mj_forward(model, data)
    openings = openings_from_mujoco(model, data)            # the client's `rby1_state()` formula
    assert openings["left"] == pytest.approx(abs(q_open) / 0.045)
    chain.set_joint_parameters(FingerJointMap(chain).joint_values(openings))
    try:
        q = scene.robot_state()
        base_inv = np.linalg.inv(scene.body_pose("base"))
        for link in FINGERS:
            mj_pose = base_inv @ scene.body_pose(MJCF_BODY_ALIASES.get(link, link))
            err = np.linalg.norm(chain.link_pose(q, link)[:3, 3] - mj_pose[:3, 3]) * 1000.0
            assert err < 0.01, (link, q_open, err)
    finally:
        chain.set_joint_parameters(None)


def test_the_map_refuses_a_client_that_opens_the_other_way(chain, client):
    import dataclasses

    conv = load_client_convention(client)
    flipped = dataclasses.replace(conv, open_command_m=+0.045)
    with pytest.raises(ValueError, match="other way"):
        FingerJointMap(chain, flipped)


def test_joint_values_clamp_and_mirror(chain, client):
    fmap = FingerJointMap(chain, load_client_convention(client))
    v = fmap.joint_values({"left": 1.0, "right": 0.5})
    assert v == pytest.approx({"gripper_finger_l1": -0.045, "gripper_finger_l2": 0.045,
                               "gripper_finger_r1": -0.0225, "gripper_finger_r2": 0.0225})
    assert fmap.joint_values({"left": 1.5})["gripper_finger_l1"] == pytest.approx(-0.05)
    assert fmap.joint_values({"left": None, "right": float("nan")}) == {}


# ------------------------------------------------------------------ 4. TO window model


def test_step_zero_is_the_measured_opening():
    assert opening_envelope(0.4, [1.0] * 5, 5)[0] == 0.4


def test_closing_keeps_the_measured_opening_so_fingers_stay_out_of_the_object():
    """T17: 명령 0.0 인데 개도 1.013 → 0.904 → 0.712. 명령을 그대로 쓰면 모델 손가락이 사과 안이다."""
    np.testing.assert_allclose(opening_envelope(1.013, [0.0] * 8, 8), [1.013] * 8)


def test_holding_uses_the_blocked_opening_not_the_command():
    """T17 유지 구간: 개도 0.715, 명령 ≈ 0.0–0.57."""
    np.testing.assert_allclose(opening_envelope(0.715, [0.0, 0.57, 0.3] + [0.0] * 5, 8),
                               [0.715] * 8)


def test_release_opens_with_the_command_and_stays_open():
    got = opening_envelope(0.715, [0.715, 0.715, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2], 8)
    np.testing.assert_allclose(got, [0.715, 0.715, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def test_commands_are_clipped_and_padded_like_the_client():
    got = opening_envelope(0.2, [1.4, -0.3], 4)
    np.testing.assert_allclose(got, [0.2, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(opening_envelope(0.5, [], 3), [0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        opening_envelope(0.5, [0.5], 0)


def test_path_is_per_step_joint_values(chain, client):
    fmap = FingerJointMap(chain, load_client_convention(client))
    P, openings = fmap.path({"left": 0.715}, {"left": np.array([0.0, 0.0, 1.0, 1.0])}, 4,
                            default=np.zeros(4))
    assert P.shape == (4, 4)
    np.testing.assert_allclose(openings["left"], [0.715, 0.715, 1.0, 1.0])
    np.testing.assert_allclose(P[0], -0.045 * np.array([0.715, 0.715, 1.0, 1.0]))
    np.testing.assert_allclose(P[1], -P[0])
    np.testing.assert_array_equal(P[2:], 0.0)       # right hand not measured → default


# ------------------------------------------------------------------ 5. AG3S and the scene


def test_ag3s_sets_both_models_and_records_the_source(urdf):
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.runtime.pipeline import AG3S

    filt, cons = UrdfSphereChain(urdf), UrdfSphereChain(urdf)
    ag3s = AG3S(AG3SConfig.from_dict({"clustering": {"gripper_max_opening": 0.07}}),
                robot_model=filt, constraint_robot_model=cons)
    assert ag3s.finger_record["source"] == "model_default"
    values = {"gripper_finger_l1": -0.045, "gripper_finger_l2": 0.045}
    rec = ag3s.set_finger_joints(values, record={"source": "state[7],state[15]"})
    assert rec["source"] == "state[7],state[15]"
    assert filt.joint_parameters["gripper_finger_l1"] == -0.045
    assert cons.joint_parameters["gripper_finger_l2"] == 0.045
    assert ag3s.finger_record["applied"]["constraint"]["gripper_finger_l1"] == -0.045
    ag3s.set_finger_joints(None)
    assert filt.joint_parameters_are_default and cons.joint_parameters_are_default
    # set on the model directly (offline: camera_observation) → `model_state`
    filt.set_joint_parameters({"gripper_finger_r1": -0.01})
    ag3s._finger_record = {"source": "model_default"}                         # noqa: SLF001
    assert ag3s.finger_record["source"] == "model_state"
    filt.set_joint_parameters(None)


def test_every_frame_carries_the_finger_record():
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.runtime.pipeline import AG3S
    from tests.ag3s.fixtures import DEFAULT_ARM_Q, PlanarTwoLinkArm, make_scene

    arm = PlanarTwoLinkArm()
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    ag3s = AG3S(AG3SConfig(), robot_model=arm)
    cs = ag3s.process(attention_map=scene.attention_grid, depth=scene.depth,
                      camera_intrinsics=scene.camera_intrinsics, T_base_cam=scene.T_base_cam,
                      image_hw=scene.hw, robot_state=DEFAULT_ARM_Q)
    assert cs.metrics["finger_joints"]["applied"] == {}      # a model without fingers


def test_camera_observation_puts_the_scene_opening_into_the_model(mj, chain):
    from benchmark.ag3s.experiments.sources.mujoco_source import set_finger_joints_from_scene

    mujoco, model, data, scene = mj
    for name in FINGER_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = -0.03 if name.endswith("1") else 0.03
    mujoco.mj_forward(model, data)
    try:
        values = set_finger_joints_from_scene(scene, chain)
        assert values["gripper_finger_l1"] == pytest.approx(-0.03)
        assert chain.joint_parameters["gripper_finger_r2"] == pytest.approx(0.03)
    finally:
        chain.set_joint_parameters(None)
