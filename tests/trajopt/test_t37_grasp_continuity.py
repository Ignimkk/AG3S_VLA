"""T37 (K1) — 닫기(`closing`)·쥐기(`held`) 청크에서는 continuity 항을 끈다. 그 청크만.

T35.diag `C1_to_ablation`: E3b ep1808 r1 t=120 · r2 t=112 (둘 다 latch `closing`) 에서 refined −
reference 손끝 편차 19.6 / 34.7 mm 가 continuity 만 끄면 0.0 mm 였다 — 연속성 기준(직전 청크의 꼬리)이
닫는 동안 손을 파지 자리에서 끌어낸다. 이 파일이 고정하는 것 (기록 없이 도는 합성 부분):

1. `CostConfig.grasp_continuity_off` — 기본 False (O2: T37.verify 에서 이득 없음), dict 로 켜고 끌 수
   있고, bool 이 아니면 거절.
2. `grasp_continuity_off_reason` — `closing` · `held` 에서만 사유를 낸다 (문자열 상태도 받는다).
3. `TrajOptChunkRefiner` — `context["continuity_off"]` 가 있으면 그 청크의 목적함수가 `w_continuity=0`
   과 **같다** (같은 궤적). 없으면 예전 그대로. `metrics.continuity_{active,reason,off_by}` 가 실린다.
4. `SafePolicy._scene_fn` — 이번 청크의 최종 latch 상태로 키를 쓰고 지운다. 설정이 False(기본) 면
   쓰지 않는다 — gate 를 보는 테스트는 설정을 명시적으로 켠다.
   `_to_block` 에 `continuity_active` · `continuity_reason`.

기록 재생 (T34 E3b 1808 청크) 은 `test_t37_grasp_continuity_replay.py`.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt import safe_policy as SP
from AG3S.trajopt import wire
from AG3S.trajopt.config import CostConfig, TrajOptConfig, TrajOptConfigError
from AG3S.trajopt.grasp_latch import GraspPhase
from AG3S.trajopt.refiner import TrajOptChunkRefiner
from tests.trajopt.fixtures import empty_scene, make_scenario

pytest.importorskip("casadi")
pytest.importorskip("osqp")


# --- 1. config --------------------------------------------------------------------------


def test_the_key_defaults_off_and_can_be_turned_on_and_off():
    # O2 (2026-10-01): T37.verify 폐루프에서 성공 이득 없음 + place 요동 증가 → 기본 off.
    assert CostConfig().grasp_continuity_off is False
    assert TrajOptConfig.from_dict({}).cost.grasp_continuity_off is False
    on = TrajOptConfig.from_dict({"cost": {"grasp_continuity_off": True}})
    assert on.cost.grasp_continuity_off is True
    off = TrajOptConfig.from_dict({"cost": {"grasp_continuity_off": False}})
    assert off.cost.grasp_continuity_off is False
    # 나머지 비용은 그대로다.
    assert on.cost.w_continuity == off.cost.w_continuity == CostConfig().w_continuity
    assert TrajOptConfig().with_overrides(
        {"cost": {"grasp_continuity_off": True}}).cost.grasp_continuity_off is True
    assert TrajOptConfig().with_overrides(
        {"cost": {"grasp_continuity_off": False}}).cost.grasp_continuity_off is False


def test_a_non_bool_value_is_refused():
    with pytest.raises(TrajOptConfigError, match="grasp_continuity_off"):
        TrajOptConfig.from_dict({"cost": {"grasp_continuity_off": "yes"}})


# --- 2. the gate --------------------------------------------------------------------------


@pytest.mark.parametrize("phase", [GraspPhase.CLOSING, GraspPhase.HELD, "closing", "held"])
def test_closing_and_held_turn_the_term_off(phase):
    reason = SP.grasp_continuity_off_reason(phase, True)
    assert reason is not None and GraspPhase(phase).value in reason
    assert "grasp_continuity_off" in reason


@pytest.mark.parametrize("phase", [GraspPhase.SEARCHING, GraspPhase.LATCHED, GraspPhase.PLACED,
                                   "latched", "searching", "placed", None, "", "nonsense"])
def test_every_other_state_keeps_it(phase):
    assert SP.grasp_continuity_off_reason(phase, True) is None


@pytest.mark.parametrize("phase", [GraspPhase.CLOSING, GraspPhase.HELD])
def test_the_config_switch_keeps_it_on_in_every_state(phase):
    assert SP.grasp_continuity_off_reason(phase, False) is None


def test_the_phase_set_is_exactly_closing_and_held():
    assert set(SP.GRASP_CONTINUITY_OFF_PHASES) == {GraspPhase.CLOSING, GraspPhase.HELD}


# --- 3. refiner ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scenario():
    return make_scenario(horizon=12, penetration=0.03)


def _refiner(scenario, **overrides):
    config = scenario.config.with_overrides(overrides) if overrides else scenario.config
    return TrajOptChunkRefiner(scenario.robot, scenario.layout,
                               lambda ctx: (scenario.scene, scenario.q_now, True), config)


def _previous(scenario, offset=0.15):
    """직전 청크: 같은 청크에 arm 열만 `offset` rad 어긋남 — continuity 가 끌어당길 거리가 있다."""
    prev = np.array(scenario.chunk, np.float64, copy=True)
    prev[:, scenario.layout.action_columns] += offset
    return prev


def test_the_context_key_is_the_same_objective_as_a_zero_weight(scenario):
    prev = _previous(scenario)
    gated = _refiner(scenario)
    out = gated.refine(scenario.chunk, {"previous_physical_chunk": prev,
                                        "continuity_off": "grasp latch closing (test)"})
    zero = _refiner(scenario, cost={"w_continuity": 0.0})
    ref = zero.refine(scenario.chunk, {"previous_physical_chunk": prev})
    np.testing.assert_array_equal(gated.last_result.trajectory, zero.last_result.trajectory)
    np.testing.assert_array_equal(out, ref)

    m = gated.last_result.metrics
    assert m["continuity_active"] is False
    assert m["continuity_off_by"] == "grasp latch closing (test)"
    assert m["continuity_reason"] == "disabled for this chunk: grasp latch closing (test)"
    assert gated.last_continuity["used"] is False


def test_without_the_key_nothing_changes_and_the_term_is_active(scenario):
    prev = _previous(scenario)
    plain = _refiner(scenario)
    out = plain.refine(scenario.chunk, {"previous_physical_chunk": prev})
    empty_key = _refiner(scenario)
    out2 = empty_key.refine(scenario.chunk, {"previous_physical_chunk": prev,
                                             "continuity_off": None})
    zero = _refiner(scenario, cost={"w_continuity": 0.0})
    zero.refine(scenario.chunk, {"previous_physical_chunk": prev})

    np.testing.assert_array_equal(out, out2)
    m = plain.last_result.metrics
    assert m["continuity_active"] is True
    assert m["continuity_reason"] == "assumed"
    assert m["continuity_off_by"] is None
    # 항이 실제로 무언가를 끌어당기는 판이다 — 끄면 궤적이 다르다.
    assert np.abs(plain.last_result.trajectory - zero.last_result.trajectory).max() > 1e-4


def test_the_key_is_per_call_not_sticky(scenario):
    prev = _previous(scenario)
    r = _refiner(scenario)
    r.refine(scenario.chunk, {"previous_physical_chunk": prev, "continuity_off": "held"})
    assert r.last_result.metrics["continuity_active"] is False
    r.refine(scenario.chunk, {"previous_physical_chunk": prev})
    assert r.last_result.metrics["continuity_active"] is True
    assert r.last_result.metrics["continuity_off_by"] is None


def test_other_reasons_for_no_term_are_reported_as_before(scenario):
    r = _refiner(scenario)
    r.refine(scenario.chunk, {})
    assert r.last_result.metrics["continuity_active"] is False
    assert r.last_result.metrics["continuity_reason"] == "no previous chunk"
    assert r.last_result.metrics["continuity_off_by"] is None
    z = _refiner(scenario, cost={"w_continuity": 0.0})
    z.refine(scenario.chunk, {"previous_physical_chunk": _previous(scenario),
                              "continuity_off": "held"})
    # 가중치가 이미 0 이면 그 사유가 먼저다.
    assert z.last_result.metrics["continuity_reason"] == "disabled (w_continuity <= 0)"


# --- 4. SafePolicy ------------------------------------------------------------------------


def _to_config(switch: bool | None):
    """`SafePolicy` 의 기본 `to_config` 와 같은 설정 + `cost.grasp_continuity_off`.

    `switch=None` 이면 키를 넣지 않는다 — 기본값을 그대로 쓴다.
    """
    d = {"collision": {"backend": "esdf", "esdf_margin": 0.05, "use_support_planes": False}}
    if switch is not None:
        d["cost"] = {"grasp_continuity_off": switch}
    return TrajOptConfig.from_dict(d)


def _policy(**kw):
    from tests.trajopt.test_safe_policy import _StubPolicy, _safe_policy

    return _safe_policy(_StubPolicy(), **kw)


def _gated_policy():
    """gate 를 **명시적으로 켠** 정책 (O2 이후 기본은 off)."""
    return _policy(to_config=_to_config(True))


def _wire_scene_fn(sp, monkeypatch):
    """`_scene_fn` 을 카메라·AG3S·거리장 없이 끝까지 돌게: 관측 하나, 빈 제약, 빈 씬."""
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    obs = [types.SimpleNamespace(timestamp=1.0, robot_state=q, camera_id="zed_left",
                                 attention_map=None)]
    monkeypatch.setattr(wire, "unpack_camera_observations", lambda scene, **kw: obs)
    cs = types.SimpleNamespace(status=types.SimpleNamespace(value="ok"), attached=None,
                               target=None, esdf=None)
    monkeypatch.setattr(sp.ag3s, "process_multi_debug", lambda *a, **k: (cs, {}))
    empty = empty_scene(sp.linearizer)
    monkeypatch.setattr(SP, "scene_from_constraint_set", lambda *a, **k: empty)
    return q


def _chunk_pair(sp):
    """정책 청크(0) 와 직전 청크(arm 열 +0.15 rad)."""
    from tests.trajopt.test_safe_policy import DOF, H

    chunk = np.zeros((H, DOF))
    prev = chunk.copy()
    prev[:, sp.layout.action_columns] += 0.15
    return chunk, prev


def _step(sp, fb, chunk, prev):
    sp._pending = {"scene": {}, "exec_feedback": fb, "chunk": chunk,           # noqa: SLF001
                   "state": None, "timing": {}}
    ctx = {"t_step": int(fb.get("seq") or 0), "previous_physical_chunk": prev}
    out = sp.refiner.refine(chunk, ctx)
    return ctx, out, sp.refiner.last_result.metrics


def test_scene_fn_turns_the_term_off_while_closing_and_held_only(monkeypatch, capsys):
    from tests.trajopt.test_grasp_evidence import _fb, _finger_origin, _lock, _manipulated

    sp = _gated_policy()
    q = _wire_scene_fn(sp, monkeypatch)
    sp.ag3s._manipulated = _manipulated(                                       # noqa: SLF001
        _finger_origin(sp, q) + np.array([0.05, 0.0, 0.0]))
    _lock(sp)
    chunk, prev = _chunk_pair(sp)

    seen = []
    for fb in (_fb(1, [1.0] * 8, 1.0),           # 열림 — latched
               _fb(2, [0.0] * 8, 0.715),          # 닫힘 명령 적용, 손가락 움직이는 중 — closing
               _fb(3, [0.0] * 8, 0.715),          # 멈춤 — held (attach)
               _fb(4, [0.72] * 8, 0.72),          # 쥔 채로
               _fb(5, [1.0] * 8, 1.0),            # 열림 1
               _fb(6, [1.0] * 8, 1.0)):           # 열림 2 — placed (detach)
        ctx, out, m = _step(sp, fb, chunk, prev)
        seen.append((sp._latch.phase, ctx.get("continuity_off"), m["continuity_active"],  # noqa: SLF001
                     m["continuity_off_by"]))

    phases = [p for p, *_ in seen]
    assert phases == [GraspPhase.LATCHED, GraspPhase.CLOSING, GraspPhase.HELD, GraspPhase.HELD,
                      GraspPhase.HELD, GraspPhase.PLACED]
    # (빈 씬이라 최적화기는 reference 를 그대로 돌려준다 — 궤적 효과는 위 refiner 테스트와
    # 기록 재생 테스트가 본다. 여기서는 배선: 상태 → context 키 → 항이 들어갔나.)
    for phase, key, active, off_by in seen:
        if phase in (GraspPhase.CLOSING, GraspPhase.HELD):
            assert key == f"grasp latch {phase.value} (cost.grasp_continuity_off)"
            assert active is False and off_by == key
        else:
            assert key is None and active is True and off_by is None
    # exec_continuity 기록에도 사유가 실린다 (refiner.last_continuity 를 합친다).
    assert sp.refiner.last_continuity["used"] is True
    out = capsys.readouterr().out
    assert "continuity term OFF — grasp latch closing" in out
    assert "continuity term back ON (grasp latch placed)" in out
    assert out.count("continuity term") == 2        # 켜짐/꺼짐이 바뀔 때만


def test_scene_fn_leaves_the_term_on_when_the_switch_is_off(monkeypatch):
    from tests.trajopt.test_grasp_evidence import _fb, _finger_origin, _lock, _manipulated

    sp = _policy(to_config=_to_config(False))
    q = _wire_scene_fn(sp, monkeypatch)
    sp.ag3s._manipulated = _manipulated(                                       # noqa: SLF001
        _finger_origin(sp, q) + np.array([0.05, 0.0, 0.0]))
    _lock(sp)
    chunk, prev = _chunk_pair(sp)
    for fb in (_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.715), _fb(3, [0.0] * 8, 0.715)):
        ctx, _, m = _step(sp, fb, chunk, prev)
        assert "continuity_off" not in ctx and m["continuity_active"] is True
    assert sp._latch.phase is GraspPhase.HELD                                   # noqa: SLF001


def test_to_block_and_reset_carry_the_gate(monkeypatch):
    from tests.trajopt.test_grasp_evidence import _fb, _finger_origin, _lock, _manipulated

    sp = _gated_policy()
    q = _wire_scene_fn(sp, monkeypatch)
    sp.ag3s._manipulated = _manipulated(                                       # noqa: SLF001
        _finger_origin(sp, q) + np.array([0.05, 0.0, 0.0]))
    _lock(sp)
    chunk, prev = _chunk_pair(sp)
    _step(sp, _fb(1, [1.0] * 8, 1.0), chunk, prev)
    block = sp._to_block()                                                     # noqa: SLF001
    assert block["continuity_active"] is True and block["continuity_reason"] == "assumed"
    _step(sp, _fb(2, [0.0] * 8, 0.715), chunk, prev)
    block = sp._to_block()                                                     # noqa: SLF001
    assert block["continuity_active"] is False
    assert "grasp latch closing" in block["continuity_reason"]
    assert sp._continuity_gate_on is True                                      # noqa: SLF001
    sp.reset()
    assert sp._continuity_gate_on is False                                     # noqa: SLF001


@pytest.mark.parametrize("to_config", [None, "no_key"], ids=["SafePolicy_default", "from_dict_no_key"])
def test_by_default_the_term_stays_on_while_closing_and_held(monkeypatch, capsys, to_config):
    """O2: 기본 설정이면 closing · held 청크에서도 continuity 가 켜져 있다 (T37 이전과 같다).

    `None` 은 `SafePolicy` 가 스스로 짓는 기본 `to_config`, `"no_key"` 는 키 없이 지은 설정
    (`serve_safe.trajopt_config_from_args` 가 이 경로다).
    """
    from tests.trajopt.test_grasp_evidence import _fb, _finger_origin, _lock, _manipulated

    assert CostConfig().grasp_continuity_off is False
    sp = _policy() if to_config is None else _policy(to_config=_to_config(None))
    assert sp.to_config.cost.grasp_continuity_off is False
    q = _wire_scene_fn(sp, monkeypatch)
    sp.ag3s._manipulated = _manipulated(                                       # noqa: SLF001
        _finger_origin(sp, q) + np.array([0.05, 0.0, 0.0]))
    _lock(sp)
    chunk, prev = _chunk_pair(sp)

    phases = []
    for fb in (_fb(1, [1.0] * 8, 1.0),           # latched
               _fb(2, [0.0] * 8, 0.715),          # closing
               _fb(3, [0.0] * 8, 0.715),          # held
               _fb(4, [0.72] * 8, 0.72)):         # held
        ctx, _, m = _step(sp, fb, chunk, prev)
        phases.append(sp._latch.phase)                                          # noqa: SLF001
        assert "continuity_off" not in ctx
        assert m["continuity_active"] is True and m["continuity_off_by"] is None
        block = sp._to_block()                                                 # noqa: SLF001
        assert block["continuity_active"] is True and block["continuity_reason"] == "assumed"
    # gate 가 볼 상태를 실제로 지났다 — 그런데도 켜져 있다.
    assert GraspPhase.CLOSING in phases and GraspPhase.HELD in phases
    assert sp._continuity_gate_on is False                                      # noqa: SLF001
    assert "continuity term" not in capsys.readouterr().out
