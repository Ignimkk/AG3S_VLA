"""T9 — closed loop 도 정책 **원본** 청크를 싣고, 목적함수 가중치를 CLI 로 뺀다.

## 왜

충돌 제약이 **하나도 활성이 아닌** 판에서도 TO 가 청크를 고친다 (실행되는 8 step 안에서 중앙값
2.6°, 최대 8.8°, 손이 다가갈수록 커진다). 그러면 남은 변형은 전부 **목적함수**가 만든 것이다
(`w_track` · `w_smooth` · `w_continuity` · `w_slack`). 그런데 그 크기를 closed loop 에서 잴 수가
없었다 — 원본 청크가 shadow 응답에만 실렸기 때문이다. 같은 결핍에 다섯 번 막혔다.

## 무엇이 걸려 있었나 — 한 키가 두 뜻을 겸했다

`actions_reference` 의 **있음/없음**이 *"서버가 shadow 모드인가"* 의 유일한 신호였다. 그래서
closed loop 이 그 키를 실으면 로컬이 *"서버가 shadow 인데 나는 아니다"* 로 읽고 **즉시 죽는다.**
겸직을 끊었다: 모드는 `shadow` 키가 **명시**하고, `actions_reference` 는 데이터일 뿐이다.

**그 거절 자체는 한 줄도 느슨해지지 않았다.** 이 파일의 절반이 그것을 지킨다 — 네 조합 전부,
그리고 `shadow` 키를 모르는 **옛 서버**에 대한 물러남까지.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from AG3S.trajopt import serve_safe, wire
from AG3S.trajopt.client import SafeRemoteClient
from AG3S.trajopt.config import CostConfig, TrajOptConfig

H, DOF = 5, wire.ACTION_WIDTH
BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def _verdict(safe=True):
    return wire.SafetyVerdict(ag3s_status="ok", geometry_certified=True,
                              trajopt_status="optimal", max_violation_m=0.0, safe=safe)


def _response(*, shadow, with_reference=True, refined=2.0, reference=1.0, seq=1):
    return wire.pack_response(
        np.full((H, DOF), refined), _verdict(), seq=seq, timing_ms={},
        actions_reference=(np.full((H, DOF), reference) if with_reference else None),
        shadow=shadow)


# --- 1. 와이어 — 모드와 데이터가 갈렸다 -------------------------------------------------


@pytest.mark.parametrize("shadow", [False, True])
def test_the_mode_is_always_declared(shadow):
    """**`False` 여도 실린다.** 없음이 "closed loop" 이면 옛 서버와 구별할 수 없다."""
    out = _response(shadow=shadow)
    assert out[wire.SHADOW] is shadow
    assert wire.unpack_shadow(out) is shadow


def test_a_missing_mode_key_means_an_old_server_not_closed_loop():
    """세 값이어야 한다 — `True` · `False` · **말하지 않았다**."""
    assert wire.unpack_shadow({}) is None
    assert wire.unpack_shadow({wire.SHADOW: False}) is False


@pytest.mark.parametrize("shadow", [False, True])
def test_the_reference_rides_in_both_modes_and_actions_stay_refined(shadow):
    out = _response(shadow=shadow)
    np.testing.assert_allclose(out["actions"], 2.0), "서버는 자기가 계산한 것을 말한다"
    np.testing.assert_allclose(wire.unpack_actions_reference(out), 1.0)
    # 전송은 msgpack_numpy 다 — `actions` 와 같은 float32 ndarray 여야 그대로 건너간다.
    assert out[wire.ACTIONS_REFERENCE].dtype == np.float32


def test_a_reference_of_a_different_shape_is_still_refused():
    """**한 칸씩 밀린 청크가 실행되는 것**이 가장 위험한 실패다. 이 검사는 그대로 남는다."""
    with pytest.raises(ValueError, match="shifted chunk"):
        wire.pack_response(np.zeros((H, DOF)), _verdict(), seq=1, timing_ms={},
                           actions_reference=np.zeros((H + 1, DOF)))


def test_a_frame_with_no_reference_still_declares_its_mode():
    """원본을 못 낸 프레임(hold 등)이 있어도 모드는 말한다 — 없음은 데이터의 부재다."""
    out = _response(shadow=False, with_reference=False)
    assert wire.unpack_actions_reference(out) is None
    assert wire.unpack_shadow(out) is False


# --- 2. 클라이언트 — 거절이 느슨해지지 않았다 -------------------------------------------


class _FakePolicy:
    """응답을 그대로 돌려주는 서버. `metadata` 로 접속 시점 검사도 흉내 낸다."""

    def __init__(self, response, meta=None):
        self.response = response
        self._meta = meta

    def infer(self, request):
        out = dict(self.response)
        out["seq"] = int(request.get(wire.PREFIX + "seq", 1))
        return out

    def get_server_metadata(self):
        return dict(self._meta or {})


_MATCH = object()


def _client(response, *, local_shadow, meta=_MATCH):
    """`meta=_MATCH` 면 메타데이터가 로컬 모드와 **맞다** — 접속 시점 검사를 통과시켜 프레임 쪽
    검사를 따로 시험할 수 있게 한다. 메타데이터 경로 자체를 시험할 때는 값을 명시한다."""
    from tests.trajopt.test_safe_client import _Scene

    if meta is _MATCH:
        meta = {"shadow": True} if local_shadow else {}
    return SafeRemoteClient(policy=_FakePolicy(response, meta), scene=_Scene(),
                            cameras=("zed_left",), shadow=local_shadow)


def _infer(client):
    return client.infer({"state": np.zeros(DOF)})


@pytest.mark.parametrize("mode", [False, True])
def test_a_matched_pair_runs(mode):
    client = _client(_response(shadow=mode), local_shadow=mode)
    out = _infer(client)
    assert client.last_actions_reference is not None
    assert client.last_executed_chunk == ("reference" if mode else "refined")
    # 닫힌 고리에서 실행되는 것은 **refined** 다. 원본이 실렸다고 실행이 바뀌지 않는다.
    np.testing.assert_allclose(out["actions"], 1.0 if mode else 2.0)


def test_closed_loop_with_a_reference_does_not_raise_anymore():
    """**이 한 줄이 T9 의 회귀다.** 예전 배선에서는 여기서 즉시 죽었다."""
    client = _client(_response(shadow=False), local_shadow=False)
    _infer(client)
    assert client.last_actions_refined is not None
    np.testing.assert_allclose(client.last_actions_reference, 1.0)
    np.testing.assert_allclose(client.last_actions_refined, 2.0)
    assert client.last_executed_chunk == "refined"


def test_the_local_side_still_refuses_a_shadow_server_when_it_is_not_shadow():
    # (a) 메타데이터에서 — 접속 즉시, 카메라 한 번 안 찍고 죽는다.
    with pytest.raises(RuntimeError, match="--safe-shadow is off"):
        _client(_response(shadow=True), local_shadow=False, meta={"shadow": True})
    # (b) 메타데이터를 안 주는 전송에서도 **첫 프레임**에서 죽는다.
    client = _client(_response(shadow=True), local_shadow=False, meta={})
    with pytest.raises(RuntimeError, match="--safe-shadow is off"):
        _infer(client)


def test_the_local_side_still_refuses_a_closed_loop_server_when_it_asks_for_shadow():
    # (a) 메타데이터에서.
    with pytest.raises(RuntimeError, match="not running with --shadow"):
        _client(_response(shadow=False), local_shadow=True, meta={})
    # (b) 메타데이터가 shadow 라고 말했는데 응답이 아니라고 말하는 경우 — 응답이 이긴다.
    client = _client(_response(shadow=False), local_shadow=True, meta={"shadow": True})
    with pytest.raises(RuntimeError, match="not running with --shadow"):
        _infer(client)


def test_a_shadow_server_that_sends_no_reference_is_a_server_fault():
    """모드 불일치가 아니라 서버 결함이다. 그래도 실행할 것이 없으므로 죽는다."""
    client = _client(_response(shadow=True, with_reference=False), local_shadow=True,
                     meta={"shadow": True})
    with pytest.raises(RuntimeError, match="server-side fault"):
        _infer(client)


@pytest.mark.parametrize("local_shadow", [False, True])
def test_an_old_server_without_the_mode_key_falls_back_to_the_old_rule(local_shadow):
    """`shadow` 키를 모르는 서버 — 그때는 `actions_reference` 의 있음/없음이 여전히 신호다."""
    old = _response(shadow=False, with_reference=local_shadow)
    old.pop(wire.SHADOW)
    client = _client(old, local_shadow=local_shadow)
    _infer(client)  # 짝이 맞으므로 통과한다
    assert client.last_executed_chunk == ("reference" if local_shadow else "refined")

    crossed = _response(shadow=False, with_reference=not local_shadow)
    crossed.pop(wire.SHADOW)
    bad = _client(crossed, local_shadow=local_shadow)
    with pytest.raises(RuntimeError):
        _infer(bad)


def test_the_pairing_message_does_not_claim_the_key_is_the_evidence():
    """메시지가 세 경로(메타데이터·`shadow` 키·옛 서버 추론)에서 다 쓰이므로, 한 경로의 근거를
    적으면 나머지에서 거짓말이 된다. 예전 문장은 *"it carries 'actions_reference'"* 였다."""
    client = _client(_response(shadow=False), local_shadow=False)
    says_shadow = client._pairing_message(server_shadow=True)
    assert "--shadow" in says_shadow
    assert f"carries {wire.ACTIONS_REFERENCE!r}" not in says_shadow


# --- 3. 서버 — shadow 와 "기록에 싣는 것" 을 갈라 적는다 --------------------------------


@pytest.mark.parametrize("mode", [False, True])
def test_the_safe_policy_declares_its_mode_and_always_carries_the_reference(mode):
    from tests.trajopt.test_safe_policy import _shadow_policy

    rng = np.random.default_rng(1)
    chunk = rng.normal(size=(H, DOF)) * 0.1
    sp = _shadow_policy(chunk, shadow=mode)
    out = sp.infer({"state": np.zeros(DOF), wire.PREFIX + "cameras": []})

    assert wire.unpack_shadow(out) is mode
    np.testing.assert_allclose(wire.unpack_actions_reference(out), chunk, rtol=0, atol=1e-6)
    # `actions` 는 **어느 모드에서도 refined** 다 — 서버는 자기가 계산한 것을 말한다.
    for col in range(DOF):
        if col in wire.GRIPPER_COLUMNS:
            continue
        np.testing.assert_allclose(out["actions"][:, col], chunk[:, col] + 0.5,
                                   rtol=0, atol=1e-6)


def test_the_metadata_key_is_untouched():
    """접속 시점 검사의 근거는 예전 그대로다 — 켜졌을 때만 키가 있다."""
    from tests.trajopt.test_safe_policy import _safe_policy

    assert "shadow" not in _safe_policy().metadata


def _captured(fn, *args):
    """`serve_safe` logger 로 나간 메시지 전부. `caplog` 을 쓰지 않는 이유는 이 파일이 두 함수의
    **출력이 서로 모순되지 않는가** 를 보므로 둘을 따로 잡아야 하기 때문이다."""
    import logging as _logging

    records: list = []
    handler = _logging.Handler()
    handler.emit = records.append
    log = _logging.getLogger(serve_safe.__name__)
    log.addHandler(handler)
    previous = log.level
    log.setLevel(_logging.INFO)
    try:
        fn(*args)
    finally:
        log.removeHandler(handler)
        log.setLevel(previous)
    return "\n".join(r.getMessage() for r in records)


def test_the_data_line_names_the_key_and_claims_no_mode():
    """데이터를 설명하는 줄이 모드를 단정하면 모드 줄과 반대를 말할 수 있다 — 실제로 그랬다."""
    text = _captured(serve_safe.announce_reference_payload)
    assert wire.ACTIONS_REFERENCE in text, "이 줄은 그 키에 **대한** 줄이다"
    assert "BOTH modes" in text
    # 모드를 단정하지 않는다. 처음 문구는 여기서 "the robot still executes the refined chunk"
    # 라고 적었고, `--shadow` 서버에서 그 문장은 거짓이다.
    assert "executes the refined" not in text
    assert "NOT shadow mode" not in text


def test_the_startup_mode_line_does_not_name_the_data_key():
    """**키 이름이 모드를 설명하는 자리에 나오면 겸직이 로그에서 되살아난다.**

    T9 직후 이 자리가 실제로 거짓이었다: closed loop 분기가
    *"the response carries the refined chunk only (no 'actions_reference' key)"* 라고 적었는데
    T9 부터 원본은 두 모드 모두 실린다 — 같은 시작 로그 안에서 두 줄이 반대를 말했다.

    문자열 전체를 박지 않는다 (문구는 바뀔 수 있다). 박는 것은 **세 사실이 갈려 있는가** 와
    **모드 설명에 데이터 키 이름이 없는가** 둘이다.
    """
    for shadow, executes, other in ((False, "REFINED", "POLICY"), (True, "POLICY", "REFINED")):
        text = _captured(serve_safe.announce_execution_mode, shadow)
        # 1. 로봇이 무엇을 실행하는가.
        assert f"executes the {executes}" in text
        assert f"executes the {other}" not in text
        # 2. 모드는 `shadow` 로만 말한다 — 데이터 키 이름은 여기 없다.
        assert f"{wire.SHADOW}={shadow}" in text
        assert wire.ACTIONS_REFERENCE not in text, "키 이름이 모드를 설명하면 안 된다"
        # 3. 짝 검사는 두 모드 모두에서 언급된다.
        assert "--safe-shadow" in text
        # 그리고 **원본이 실리는 것이 모드의 특징이 아니라는 것**을 말한다.
        assert "either mode" in text or "every response" in text


def test_main_uses_that_function_rather_than_inlining_the_strings():
    """문구가 두 곳에 있으면 한 곳만 고쳐지고, 그 한 곳이 T9 에서 거짓이 된 자리다."""
    import inspect

    src = inspect.getsource(serve_safe.main)
    assert "announce_execution_mode(" in src
    assert "announce_reference_payload(" in src
    assert "closed-loop mode" not in src, "문구는 함수 안에만 있어야 한다"
    assert "SHADOW mode" not in src
    assert "response carries" not in src


# --- 4. 목적함수 가중치 CLI ------------------------------------------------------------


def test_no_weight_flag_changes_nothing():
    args = serve_safe.build_parser().parse_args(BASE_ARGV)
    assert serve_safe.cost_overrides(args) == {}
    for name in serve_safe.COST_WEIGHTS:
        assert getattr(args, name) is None
    base, plain = CostConfig(), TrajOptConfig.from_dict({}).cost
    for name in serve_safe.COST_WEIGHTS:
        assert getattr(plain, name) == getattr(base, name)


def test_the_four_weights_are_the_ones_the_objective_reads():
    """목록이 두 곳이면 flag 를 더할 때 갈라진다."""
    assert set(serve_safe.COST_WEIGHTS) == {
        f for f in CostConfig().__dataclass_fields__ if f.startswith("w_")}


@pytest.mark.parametrize("flag,field,value", [
    ("--w-track", "w_track", 0.2),
    ("--w-smooth", "w_smooth", 0.0),
    ("--w-continuity", "w_continuity", 0.0),
    ("--w-slack", "w_slack", 5000.0),
])
def test_each_flag_reaches_the_config(flag, field, value):
    args = serve_safe.build_parser().parse_args(BASE_ARGV + [flag, str(value)])
    overrides = serve_safe.cost_overrides(args)
    assert overrides == {field: value}
    cfg = TrajOptConfig.from_dict({"cost": overrides})
    assert getattr(cfg.cost, field) == value
    # 나머지 셋은 기본값 그대로다.
    for other in set(serve_safe.COST_WEIGHTS) - {field}:
        assert getattr(cfg.cost, other) == getattr(CostConfig(), other)


def test_a_weight_that_the_config_forbids_is_still_refused():
    """`w_slack` 이 `w_track` 보다 작으면 위반을 사는 편이 싸다. config 가 그것을 막는다."""
    from AG3S.trajopt.config import TrajOptConfigError

    with pytest.raises(TrajOptConfigError):
        TrajOptConfig.from_dict({"cost": {"w_track": 10.0, "w_slack": 1.0}})
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig.from_dict({"cost": {"w_track": 0.0}})


def test_a_non_default_objective_is_shouted_at_startup(caplog):
    with caplog.at_level(logging.WARNING, logger=serve_safe.__name__):
        serve_safe.announce_cost_weights({"w_track": 0.2})
    assert "THE OBJECTIVE IS NOT THE DEFAULT ONE" in caplog.text
    assert "1 → 0.2" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=serve_safe.__name__):
        serve_safe.announce_cost_weights({})
    assert caplog.text == ""


def test_the_weights_are_refused_with_no_safe():
    ap = serve_safe.build_parser()
    args = ap.parse_args(BASE_ARGV + ["--no-safe", "--w-track", "0.2"])
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)


def test_the_default_flag_set_still_starts():
    ap = serve_safe.build_parser()
    serve_safe.reject_bad_flag_combinations(ap, ap.parse_args(BASE_ARGV))


# --- 5. 가중치가 실제로 궤적을 바꾼다 (이 축을 훑는 것이 다음 단계다) -------------------


def test_lowering_w_track_lets_the_optimizer_leave_the_chunk():
    """**제약이 없는데도** 가중치가 청크를 바꾼다는 것을 QP 수준에서 본다.

    충돌 행이 0 개인 판을 쓰는 이유는 `test_problem` 과 같다 — 그때 정답이 독립적으로 알려져
    있다 (추종 항만 남으면 최소점은 reference 다). 그 판에서 `w_smooth` 를 올리면 해가 reference
    에서 **멀어지고**, 그 거리가 곧 목적함수가 만든 변형이다.
    """
    osqp = pytest.importorskip("osqp")
    import scipy.sparse as sp

    from AG3S.trajopt.problem import build_problem
    from AG3S.trajopt.types import JointLimits

    nq, horizon = 4, 8
    limits = JointLimits(lower=np.full(nq, -3.0), upper=np.full(nq, 3.0),
                         max_step=np.full(nq, 0.2), max_step_change=np.full(nq, 0.08),
                         dt=1.0 / 15.0)
    base = TrajOptConfig.from_dict(
        {"horizon": {"horizon": horizon, "execution_length": 4, "plan_horizon": None}})
    # 2 차 차분이 0 이 아닌 reference — 매끄러움 항이 붙잡을 것이 있어야 한다.
    reference = 0.05 * np.sin(np.arange(horizon))[None, :] * np.ones((nq, 1))

    def solve(cfg):
        problem = build_problem(reference, reference, limits, cfg, n_slack=0)
        solver = osqp.OSQP()
        solver.setup(P=sp.triu(problem.P, format="csc"), q=problem.q, A=problem.A,
                     l=np.clip(problem.l, -1e20, 1e20), u=np.clip(problem.u, -1e20, 1e20),
                     verbose=False, eps_abs=1e-9, eps_rel=1e-9, polish=True, max_iter=20000)
        return problem.split(solver.solve().x)[0]

    plain = solve(base.with_overrides({"cost": {"w_smooth": 0.0, "w_continuity": 0.0}}))
    smooth = solve(base.with_overrides({"cost": {"w_smooth": 5.0, "w_continuity": 0.0}}))
    assert np.abs(plain - reference).max() < 1e-6, "추종 항만 남으면 답은 reference 다"
    assert np.abs(smooth - reference).max() > 1e-3, "매끄러움 항이 청크를 바꾼다"
