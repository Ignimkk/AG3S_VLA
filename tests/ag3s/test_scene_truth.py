"""MuJoCo 참값 probe — 실제 body 자세를 읽고, 없는 body 로 죽지 않는다.

`T5c` 가 *"과제가 실제로 완결됐나"* 를 못 쟀다. 기록에 물체 자세가 없었기 때문이고, 그 값은
추정할 것이 아니라 시뮬레이터가 이미 아는 것이다. 여기서 고정하는 것은 두 가지다 — **읽은
값이 `data.xpos`/`data.xquat` 그대로인가**, 그리고 **없는 body 가 예외가 아니라 `missing`
으로 남는가**.

씬 XML 을 인라인으로 짓는다. 실제 `model_transport.xml` 을 읽으면 이 테스트가 자산 경로에
매이고, 그러면 probe 의 계약이 아니라 자산의 존재를 재게 된다.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from AG3S.ag3s.runtime.scene_truth import (
    DEFAULT_OBJECT_BODIES, ObjectPoseProbe, joint_positions)

#: `apple` 과 `crate` 만 있는 씬. 나머지 셋은 일부러 없다.
XML = """
<mujoco>
  <worldbody>
    <body name="apple" pos="0.5 0.1 0.85">
      <freejoint name="apple_free"/>
      <geom type="sphere" size="0.04"/>
    </body>
    <body name="crate" pos="0.7 -0.2 0.83" quat="0.92388 0 0 0.38268">
      <freejoint name="crate_free"/>
      <geom type="box" size="0.09 0.15 0.06"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_reads_the_bodies_that_exist(scene):
    model, data = scene
    probe = ObjectPoseProbe(model)

    assert probe.present == ("apple", "crate")
    poses = probe.poses(data)
    assert set(poses) == {"apple", "crate"}
    for name in ("apple", "crate"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        np.testing.assert_allclose(poses[name]["pos"], data.xpos[bid])
        np.testing.assert_allclose(poses[name]["quat"], data.xquat[bid])
    # crate 는 돌려 세웠다 — 자세까지 실리지 않으면 "들어갔나" 와 "넘어졌나" 가 구별되지 않는다.
    assert abs(poses["crate"]["quat"][3]) > 0.3


def test_missing_bodies_are_recorded_not_raised(scene):
    model, _ = scene
    probe = ObjectPoseProbe(model)

    assert set(probe.missing) == {"banana", "orange", "pear"}
    described = probe.describe()
    assert described["bodies"] == ["apple", "crate"]
    assert set(described["missing"]) == {"banana", "orange", "pear"}
    assert set(described["body_ids"]) == {"apple", "crate"}


def test_default_bodies_are_the_four_fruits_and_the_crate():
    """이름 다섯 개가 기본값이다. 조용히 자라면 옛 기록과 키 집합이 갈라진다."""
    assert DEFAULT_OBJECT_BODIES == ("apple", "banana", "orange", "pear", "crate")


def test_explicit_body_list_is_honoured(scene):
    model, data = scene
    probe = ObjectPoseProbe(model, bodies=("crate",))
    assert probe.present == ("crate",)
    assert probe.missing == ()
    assert set(probe.poses(data)) == {"crate"}


def test_poses_are_plain_floats(scene):
    """`ndarray` 를 그대로 넘기면 기록기의 `default=str` 가 문자열 repr 을 적는다."""
    model, data = scene
    poses = ObjectPoseProbe(model).poses(data)
    flat = poses["apple"]["pos"] + poses["apple"]["quat"]
    assert isinstance(poses["apple"]["pos"], list)
    assert all(type(v) is float for v in flat)


def test_joint_positions_is_the_whole_qpos(scene):
    model, data = scene
    data.qpos[:] = np.linspace(-1.0, 1.0, model.nq)
    q = joint_positions(data)

    assert isinstance(q, list)
    assert len(q) == model.nq
    assert all(type(v) is float for v in q)
    np.testing.assert_allclose(np.asarray(q), data.qpos)


def test_the_probe_does_not_step_the_simulation(scene):
    """기록이 물리를 건드리면 기록할 때와 안 할 때의 실행이 갈라진다."""
    model, data = scene
    before = (data.qpos.copy(), data.qvel.copy(), data.time)
    ObjectPoseProbe(model).poses(data)
    np.testing.assert_array_equal(data.qpos, before[0])
    np.testing.assert_array_equal(data.qvel, before[1])
    assert data.time == before[2]
