"""T29 — TO 가 보는 손가락: linearizer 의 per-step parameter 와 SafePolicy 배선.

* 손가락 관절은 **결정 변수가 아니다** — `CollisionLinearizer` 의 두 번째 입력 `p` 이고 Jacobian 은
  팔 관절(`nq_opt`)만이다. 스텝별 구 중심은 그 스텝의 손가락 값으로 푼 numeric FK 와 같다.
* path 를 안 주면 모델의 현재 값 — 아무도 안 건드린 모델이면 예전(닫힘)과 비트 단위로 같다.
* `SafePolicy` 는 요청의 16D `state` 열 7·15 (촬영 시점 측정 개도) 로 두 AG3S 모델을 놓고,
  TO 창에는 스텝 0 = 측정, 스텝 k ≥ 1 = `max(측정, 명령의 누적 최대)` 를 준다.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf
from AG3S.ag3s.robot_models.urdf_sphere_chain import DEFAULT_RBY1_JOINTS, UrdfCapsule
from AG3S.trajopt import wire
from AG3S.trajopt.linearize import CollisionLinearizer
from AG3S.trajopt.types import ChunkLayout


def _finger_model():
    """Arms + a small capsule on each finger — enough for the finger joints to move spheres."""
    caps = [UrdfCapsule(link=f"ee_finger_{s}{i}", origin=np.eye(4), radius=0.01, length=0.04)
            for s in ("l", "r") for i in (1, 2)]
    return UrdfSphereChain(parse_urdf(RBY1_URDF), extra_capsules=caps,
                           link_filter=[f"link_{s}_arm_{i}" for s in ("left", "right")
                                        for i in range(6)]
                           + [f"ee_finger_{s}{i}" for s in ("l", "r") for i in (1, 2)])


@pytest.fixture(scope="module")
def lin():
    model = _finger_model()
    return CollisionLinearizer(model, ChunkLayout.rby1(DEFAULT_RBY1_JOINTS), 4)


def test_the_linearizer_takes_finger_parameters_not_decision_variables(lin):
    assert lin.param_names == ("gripper_finger_l1", "gripper_finger_l2",
                               "gripper_finger_r1", "gripper_finger_r2")
    rng = np.random.default_rng(0)
    traj = rng.normal(0, 0.2, (lin.nq_opt, 4))
    q_now = np.zeros(lin.robot_model.nq)
    P = np.array([[-0.0, -0.01, -0.03, -0.045],
                  [0.0, 0.01, 0.03, 0.045],
                  [-0.02] * 4, [0.02] * 4])
    lin.set_joint_parameter_path(P)
    try:
        centres, jac = lin.sphere_states(traj, q_now)
        assert jac.shape == (4, lin.n_spheres, 3, lin.nq_opt)      # arms only
        full = lin.layout.full_q(traj, q_now)
        for k in range(4):
            want = lin.robot_model.sphere_centers_numeric(full[:, k], params=P[:, k])[0]
            np.testing.assert_allclose(centres[k], want, atol=1e-12)
        # Jacobian vs finite differences in the arm joints, fingers held at the step's values
        eps = 1e-6
        j = 3
        bumped = traj.copy()
        bumped[j, 2] += eps
        c2, _ = lin.sphere_states(bumped, q_now)
        np.testing.assert_allclose((c2[2] - centres[2]) / eps, jac[2, :, :, j], atol=1e-5)
    finally:
        lin.set_joint_parameter_path(None)


def test_without_a_path_the_model_current_values_are_used(lin):
    model = lin.robot_model
    traj = np.zeros((lin.nq_opt, 4))
    q_now = np.zeros(model.nq)
    closed, _ = lin.sphere_states(traj, q_now)
    ref = model.sphere_centers_numeric(q_now)[0]
    np.testing.assert_allclose(closed[0], ref, atol=1e-12)        # default = pinned at 0
    model.set_joint_parameters({"gripper_finger_l1": -0.045, "gripper_finger_l2": 0.045})
    try:
        opened, _ = lin.sphere_states(traj, q_now)
        np.testing.assert_allclose(opened[1], model.sphere_centers_numeric(q_now)[0], atol=1e-12)
        assert np.abs(opened - closed).max() == pytest.approx(0.045)
    finally:
        model.set_joint_parameters(None)


def test_bad_paths_are_refused(lin):
    with pytest.raises(ValueError, match=r"\(4, 4\)"):
        lin.set_joint_parameter_path(np.zeros((4, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        lin.set_joint_parameter_path(np.full((4, 4), np.nan))
    lin.set_joint_parameter_path(np.full(4, -0.01))              # one row → every step
    np.testing.assert_array_equal(lin.param_path(), np.full((4, 4), -0.01))
    lin.set_joint_parameter_path(None)


def test_the_held_object_rides_the_finger_parameter(lin):
    """`ee_finger_l1` 이 attach parent 다 — 쥔 물체의 질의점도 스텝별 손가락 값을 따른다."""
    lin.set_attached(np.array([[0.0, 0.0, 0.0]]), "ee_finger_l1")
    try:
        traj = np.zeros((lin.nq_opt, 4))
        q_now = np.zeros(lin.robot_model.nq)
        P = np.zeros((4, 4))
        P[0] = [0.0, -0.01, -0.02, -0.03]
        lin.set_joint_parameter_path(P)
        pos, jac = lin.attached_states(lin.layout.full_q(traj, q_now))
        for k in range(4):
            want = lin.robot_model.link_pose(q_now, "ee_finger_l1", params=P[:, k])[:3, 3]
            np.testing.assert_allclose(pos[k, 0], want, atol=1e-12)
        assert jac.shape == (4, 1, 3, lin.nq_opt)
    finally:
        lin.set_attached(None, None)
        lin.set_joint_parameter_path(None)


def test_a_model_without_fingers_keeps_the_old_signature():
    from tests.ag3s.fixtures import PlanarTwoLinkArm

    arm = PlanarTwoLinkArm()
    layout = ChunkLayout(action_dim=arm.nq, n_valid=arm.nq,
                         action_to_q=tuple((i, i) for i in range(arm.nq)), fixed_q=(),
                         passthrough=(), nq_model=arm.nq)
    lin = CollisionLinearizer(arm, layout, 3)
    assert lin.param_names == ()
    lin.set_joint_parameter_path(None)
    with pytest.raises(ValueError, match="no finger parameters"):
        lin.set_joint_parameter_path(np.zeros((2, 3)))


# ------------------------------------------------------------------ SafePolicy wiring


def _policy():
    from tests.trajopt.test_safe_policy import _StubPolicy, _safe_policy

    return _safe_policy(_StubPolicy())


def _spy(sp, monkeypatch, q):
    obs = [types.SimpleNamespace(timestamp=1.0, robot_state=q, camera_id="zed_left")]
    monkeypatch.setattr(wire, "unpack_camera_observations", lambda scene, **kw: obs)
    seen = {}

    class _Stop(Exception):
        pass

    def spy(*args, **kwargs):
        seen["filter"] = sp.ag3s.robot_model.joint_parameters
        seen["constraint"] = sp.ag3s.constraint_robot_model.joint_parameters
        seen["record"] = sp.ag3s.finger_record
        raise _Stop

    monkeypatch.setattr(sp.ag3s, "process_multi_debug", spy)
    return seen, _Stop


def test_safe_policy_places_the_measured_opening_and_the_plan(monkeypatch):
    sp = _policy()
    assert sp._finger_map is not None                                          # noqa: SLF001
    q = np.zeros(sp.constraint_robot_model.nq)
    seen, stop = _spy(sp, monkeypatch, q)
    left, right = sp.gripper_columns
    state = np.zeros(wire.ACTION_WIDTH)
    state[left], state[right] = 1.0, 0.5
    chunk = np.zeros((16, wire.ACTION_WIDTH))
    chunk[:, left] = [1.0, 1.0, 0.0, 0.0] + [0.0] * 12      # closing command
    chunk[:, right] = [0.5, 1.0] + [1.0] * 14                # opening command
    sp._pending = {"scene": {}, "chunk": chunk, "state": state,               # noqa: SLF001
                   "exec_feedback": wire.unpack_exec_feedback({})}
    with pytest.raises(stop):
        sp._scene_fn(None)                                                     # noqa: SLF001
    want = {"gripper_finger_l1": -0.045, "gripper_finger_l2": 0.045,
            "gripper_finger_r1": -0.0225, "gripper_finger_r2": 0.0225}
    assert seen["constraint"] == pytest.approx(want)
    assert seen["filter"] == pytest.approx(want)
    assert seen["record"]["source"] == f"state[{left}],state[{right}]"
    for lin in sp._finger_linearizers():                                       # noqa: SLF001
        P = lin.param_path()
        H = lin.horizon
        np.testing.assert_allclose(P[0], -0.045)                    # closing → stays at measured
        np.testing.assert_allclose(P[2], -0.045 * np.r_[0.5, np.ones(H - 1)])  # opening → command
    rec = sp._exec_summary()["finger_joints"]                                  # noqa: SLF001
    assert rec["opening_norm"] == {"left": 1.0, "right": 0.5}
    assert rec["plan"]["opening_norm"]["left"][0] == 1.0


def test_safe_policy_falls_back_to_feedback_then_to_the_default(monkeypatch):
    sp = _policy()
    q = np.zeros(sp.constraint_robot_model.nq)
    seen, stop = _spy(sp, monkeypatch, q)
    fb = wire.unpack_exec_feedback({})
    fb = {**fb, "available": True, "seq": 3, "measured_gripper": np.array([0.8, 1.0])}
    sp._pending = {"scene": {}, "chunk": np.zeros((16, wire.ACTION_WIDTH)),   # noqa: SLF001
                   "exec_feedback": fb}
    with pytest.raises(stop):
        sp._scene_fn(None)                                                     # noqa: SLF001
    assert seen["constraint"]["gripper_finger_l1"] == pytest.approx(-0.036)
    assert "exec_feedback" in seen["record"]["source"]

    sp._pending = {"scene": {}, "exec_feedback": wire.unpack_exec_feedback({})}  # noqa: SLF001
    with pytest.raises(stop):
        sp._scene_fn(None)                                                     # noqa: SLF001
    assert seen["constraint"] == {n: 0.0 for n in seen["constraint"]}
    assert seen["record"]["source"] == "model_default"
    for lin in sp._finger_linearizers():                                       # noqa: SLF001
        assert lin._param_path is None                                         # noqa: SLF001


def test_reset_clears_the_plan():
    sp = _policy()
    for lin in sp._finger_linearizers():                                       # noqa: SLF001
        lin.set_joint_parameter_path(np.full(len(lin.param_names), -0.02))
    sp.reset()
    for lin in sp._finger_linearizers():                                       # noqa: SLF001
        assert lin._param_path is None                                         # noqa: SLF001
