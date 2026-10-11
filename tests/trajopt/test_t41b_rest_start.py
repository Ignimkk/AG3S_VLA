"""T41 b — HOLD 뒤의 TO 계획은 **정지 상태에서** 출발한다 (`limits.rest_start`, `--rest-start`).

T40 I (C1807 seq 28): HOLD 세 청크 뒤 계획의 첫 스텝이 q_now 에서 10.8° 였다. 그 값은 첫 스텝 anchor
``|Q0 − q_now| ≤ v_max·dt`` 의 경계 (세 관절이 정확히 경계) 이고, 정지한 로봇에게는 가속도 한계
``a_max·dt² = 2.29°`` 의 4.7 배다. TO 의 시작 자세 자체는 틀리지 않았다 — q_now 는 촬영 순간의 측정
자세이고 시뮬레이션에서는 적용 순간과 비트 동일하다 (T40 G 641/641 청크). 빠진 것은 **속도** 다.

`rest_start` 를 켜면 직전 청크가 통째로 HOLD 였던 청크만 (`exec_feedback.n_exec == 0`) 모든 스텝을
``|Q_k − q_now| ≤ e_k = Σ_{i≤k} min(e0 + i·a, v)`` 로 묶는다 (`limits.rest_start_envelope`).

이 파일이 고정하는 것:

1. 범위 계산 (`rest_start_envelope`) 의 값.
2. **기본 off 는 비트 동일** — `build_problem` · `project_to_limits` · `solve` 의 기본 호출, 그리고 기록된
   E3b 청크 6 개의 replay 가 pre-T41b 코드 (`b7d5849`, `git archive`) 와 같다.
3. 켜면: 합성 post-HOLD 상태에서 step 0 이 범위 안 (`e0 = 0` 이면 `Q0 = q_now`), 범위로 충돌을 못 풀면
   VIOLATED (= gate 가 HOLD 로 거절).
4. C1807 seq 28 replay — off 는 10.8° 를 재현, on 은 2.29° 에서 시작 (feasible), `e0 = 0` 은 거절.
5. 배선 — `SafePolicy` 가 피드백으로 정지를 판단, refiner 가 그때만 인자를 넘긴다, config · CLI.
"""

from __future__ import annotations
from scripts.setup.reference_layout import expose_legacy_benchmark
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import os
import pathlib
import subprocess
import sys
import types

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError
from AG3S.trajopt.limits import project_to_limits, rest_start_envelope
from AG3S.trajopt.problem import build_problem
from tests.trajopt import _t41b_replay as RP
from tests.trajopt.fixtures import make_scenario

DEG = np.pi / 180.0
#: OSQP (eps 1e-3, polish 없음) 가 box 행을 지키는 정밀도보다 넉넉하게.
QP_TOL = 1e-4


def _limits(v=10.8 * DEG, a=2.292 * DEG, nq=3):
    return types.SimpleNamespace(max_step=np.full(nq, v), max_step_change=np.full(nq, a), nq=nq,
                                 lower=np.full(nq, -10.0), upper=np.full(nq, 10.0))


# =============================================================================================
# 1. 범위
# =============================================================================================


def test_default_envelope_is_the_reach_from_rest_under_the_acceleration_limit():
    a, v = 2.292 * DEG, 10.8 * DEG
    env = rest_start_envelope(_limits(v, a), 8)
    # 속도 i·a 가 v 에 닿기 전까지는 a·(k+1)(k+2)/2.
    np.testing.assert_allclose(env[0, :4], a * np.array([1, 3, 6, 10]), rtol=0, atol=1e-15)
    speeds = np.diff(np.concatenate([[0.0], env[0]]))
    assert np.all(speeds <= v + 1e-15)
    np.testing.assert_allclose(speeds[5:], v)                  # 6·a > v 부터 속도 한계


def test_tolerance_sets_step_zero_and_is_capped_by_the_old_anchor():
    a, v = 2.292 * DEG, 10.8 * DEG
    zero = rest_start_envelope(_limits(v, a), 4, 0.0)
    np.testing.assert_allclose(zero[0], [0.0, a, 3 * a, 6 * a], atol=1e-15)
    wide = rest_start_envelope(_limits(v, a), 3, 1.0)
    np.testing.assert_allclose(wide[0], [v, 2 * v, 3 * v])     # 예전 anchor 보다 넓어지지 않는다
    no_acc = rest_start_envelope(_limits(v, np.inf), 3)        # 가속도 한계가 없으면 예전 anchor
    np.testing.assert_allclose(no_acc[0], [v, 2 * v, 3 * v])
    assert np.all(np.isinf(rest_start_envelope(_limits(np.inf, np.inf), 3)))


# =============================================================================================
# 2. 기본 off 는 비트 동일
# =============================================================================================


def _axis(sc) -> int:
    """`right_arm_0` 의 축소(최적화) 인덱스 — fixture 의 sweep 관절."""
    names = list(sc.layout.joint_names)
    return [names[q] for _, q in sc.layout.action_to_q].index("right_arm_0")


def _problem_args():
    sc = make_scenario(horizon=6)
    ref = sc.reference
    q = ref[:, 0].copy()
    q[_axis(sc)] -= 0.15                                       # right_arm_0 이 reference 보다 0.15 rad 뒤
    return sc, ref, q


def test_build_problem_without_an_envelope_is_unchanged():
    sc, ref, q = _problem_args()
    a = build_problem(ref, ref, sc.limits, sc.config, q_now=q, n_slack=4)
    b = build_problem(ref, ref, sc.limits, sc.config, q_now=q, n_slack=4, anchor_envelope=None)
    for name in ("P", "A"):
        assert (getattr(a, name) != getattr(b, name)).nnz == 0
    for name in ("q", "l", "u"):
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes()


def test_build_problem_holds_every_step_inside_the_envelope_even_past_the_trust_region():
    sc, ref, q = _problem_args()
    env = rest_start_envelope(sc.limits, ref.shape[1])
    far = ref + 0.4                                            # trust region (0.15) 이 범위에 안 닿는다
    p = build_problem(ref, far, sc.limits, sc.config, q_now=q, n_slack=4, anchor_envelope=env)
    lo, hi = p.row_blocks["box"]
    nq, H = ref.shape
    centre = np.tile(q, H)
    width = env.T.reshape(-1)
    assert np.all(p.l[lo:hi] >= centre - width - 1e-12)
    assert np.all(p.u[lo:hi] <= centre + width + 1e-12)
    with pytest.raises(ValueError):
        build_problem(ref, ref, sc.limits, sc.config, anchor_envelope=env)


def test_project_to_limits_default_is_unchanged_and_the_envelope_clips_every_step():
    sc, ref, q = _problem_args()
    base = project_to_limits(ref, sc.limits, q)
    assert base.tobytes() == project_to_limits(ref, sc.limits, q, envelope=None).tobytes()
    env = rest_start_envelope(sc.limits, ref.shape[1])
    clipped = project_to_limits(ref, sc.limits, q, envelope=env)
    assert np.all(np.abs(clipped - q[:, None]) <= env + 1e-12)


def _solve(sc, q_now, *, limits=None, **kw):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    cfg = sc.config if not limits else sc.config.with_overrides({"limits": limits})
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    return opt.solve(sc.reference, q_now, sc.scene, template=sc.chunk, **kw)


def _q_behind(sc, rad):
    """정지한 로봇: reference 의 step 0 보다 `rad` 뒤 (sweep 하는 `right_arm_0`)."""
    q = sc.q_now.copy()
    q[sc.layout.q_indices[_axis(sc)]] -= rad
    return q


def test_solve_default_and_explicit_false_are_bit_identical():
    sc = make_scenario(horizon=6)
    q = _q_behind(sc, 0.15)
    a = _solve(sc, q)
    b = _solve(sc, q, start_at_rest=False)
    assert a.chunk.tobytes() == b.chunk.tobytes()
    assert a.status == b.status and "rest_start" not in a.metrics and "rest_start" not in b.metrics


# =============================================================================================
# 3. 켜면 — 합성 post-HOLD 상태
# =============================================================================================


def test_post_hold_step_zero_is_within_the_envelope_and_pinned_when_the_tolerance_is_zero():
    sc = make_scenario(horizon=6)
    q = _q_behind(sc, 0.15)                                    # reference step 0 이 q_now 에서 8.6°
    qi = sc.layout.q_indices
    off = _solve(sc, q)
    assert np.abs(off.trajectory[:, 0] - q[qi]).max() > 5.0 * DEG   # off: reference 를 따라 멀리 출발

    on = _solve(sc, q, limits={"rest_start": True}, start_at_rest=True)
    env = rest_start_envelope(sc.limits, sc.horizon)
    off0 = np.abs(on.trajectory - q[qi][:, None])
    assert off0[:, 0].max() <= sc.limits.max_step_change.max() + QP_TOL
    assert np.all(off0 <= env + QP_TOL)
    rec = on.metrics["rest_start"]
    assert rec["applied"] is True and rec["envelope_overshoot_rad"] <= QP_TOL
    assert rec["reference_step0_offset_rad"] == pytest.approx(0.15, abs=1e-9)

    pinned = _solve(sc, q, limits={"rest_start": True, "rest_start_tolerance": 0.0},
                    start_at_rest=True)
    assert np.abs(pinned.trajectory[:, 0] - q[qi]).max() <= QP_TOL    # step 0 = q_now
    assert pinned.metrics["rest_start"]["tolerance_rad"] == 0.0


def test_a_post_hold_chunk_that_cannot_clear_inside_the_envelope_is_rejected():
    """q_now 에서 이미 10 mm 관통 (step 0 의 구). `e0 = 0` 이면 Q0 = q_now 라 못 푼다 → VIOLATED."""
    import dataclasses

    sc = make_scenario(horizon=6)
    lin, ref, q = sc.linearizer, sc.reference, sc.q_now
    centres, _ = lin.sphere_states(ref, q)
    s = int(np.argmax(np.linalg.norm(centres[1] - centres[0], axis=1)))
    d = centres[1, s] - centres[0, s]
    perp = np.cross(d / np.linalg.norm(d), [0.0, 0.0, 1.0])
    perp /= np.linalg.norm(perp)
    radius, d_safe = 0.10, 0.05
    obstacle = centres[0, s] + perp * (lin.robot_radii[s] + radius + d_safe - 0.010)
    n = sc.scene.candidate_pos.shape[0]
    sc.scene = dataclasses.replace(sc.scene, candidate_pos=np.tile(obstacle, (n, 1)))
    off = _solve(sc, q)
    pinned = _solve(sc, q, limits={"rest_start": True, "rest_start_tolerance": 0.0},
                    start_at_rest=True)
    assert off.status.safe                                     # off 는 첫 스텝에서 비켜 간다
    assert not pinned.status.safe                              # 정지 출발로는 못 비킨다 → HOLD
    assert pinned.metrics["clearance_m"] < -0.005


# =============================================================================================
# 4. 배선
# =============================================================================================


class _SpyOptimizer:
    def __init__(self, inner):
        self.inner, self.kwargs = inner, []
        self.linearizer = inner.linearizer

    def solve(self, *args, **kwargs):
        self.kwargs.append(dict(kwargs))
        return self.inner.solve(*args, **kwargs)

    def reset(self):
        self.inner.reset()


def test_the_refiner_passes_start_at_rest_only_when_the_context_says_so():
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    sc = make_scenario(horizon=6)
    ref = TrajOptChunkRefiner(sc.robot, sc.layout, lambda _c: (sc.scene, sc.q_now, True),
                              sc.config.with_overrides({"limits": {"rest_start": True}}))
    spy = ref.optimizer = _SpyOptimizer(ref.optimizer)
    ref.refine(sc.chunk, {"t_step": 1})
    ref.refine(sc.chunk, {"t_step": 2, "start_at_rest": True})
    assert "start_at_rest" not in spy.kwargs[0]
    assert spy.kwargs[1]["start_at_rest"] is True


def _fb(n_exec, n=8, seq=27):
    return {"available": True, "seq": seq, "n_exec": n_exec, "n_steps": n,
            "executed": [True] * n_exec + [False] * (n - n_exec), "hold_kind": "unsafe"}


def _policy(rest_start: bool, tol=None):
    from AG3S.trajopt.safe_policy import SafePolicy

    sp = object.__new__(SafePolicy)
    limits = {"rest_start": True, **({} if tol is None else {"rest_start_tolerance": tol})}
    sp.to_config = TrajOptConfig.from_dict({"limits": limits} if rest_start else {})
    return sp


def test_safe_policy_decides_rest_from_the_feedback_only_when_on():
    assert _policy(False)._rest_start(_fb(0)) == {}
    sp = _policy(True, 0.01)
    held = sp._rest_start(_fb(0))
    assert held["applied"] is True and held["tolerance_rad"] == 0.01
    for fb in (_fb(8), _fb(3), {"available": False, "reason": "first chunk"}):
        out = sp._rest_start(fb)
        assert out["enabled"] is True and out["applied"] is False and out["reason"]


def test_safe_policy_infer_puts_the_key_in_the_context_only_after_a_full_hold():
    from tests.trajopt.test_exec_feedback import _feedback, _obs, _safe_policy, _spy

    H, DOF = 50, 16
    on = _safe_policy(np.full((H, DOF), 0.2),
                      to_config=TrajOptConfig.from_dict({"limits": {"rest_start": True}}))
    seen = _spy(on)
    on.infer(_obs(1))
    on.infer(_obs(2, _feedback(seq=1)))                                       # 8/8 HOLD
    on.infer(_obs(3, _feedback(seq=2, executed=(True,) * 8, executed_chunk="refined", ipc="ok",
                               hold_kind=None, hold_reason=None)))
    assert [c.get("start_at_rest") for c in seen] == [None, True, None]
    assert on._exec_summary()["rest_start"]["applied"] is False

    off = _safe_policy(np.full((H, DOF), 0.2))
    seen = _spy(off)
    off.infer(_obs(1))
    out = off.infer(_obs(2, _feedback(seq=1)))
    assert "start_at_rest" not in seen[1]
    assert "rest_start" not in off._exec_summary()
    assert not any(n.startswith("rest start") for n in out["notes"])


def test_config_defaults_and_validation():
    lim = TrajOptConfig().limits
    assert lim.rest_start is False and lim.rest_start_tolerance is None
    for bad in ({"rest_start": True, "rest_start_tolerance": -0.01},
                {"rest_start": True, "rest_start_tolerance": float("nan")},
                {"rest_start_tolerance": 0.01}):
        with pytest.raises(TrajOptConfigError):
            TrajOptConfig.from_dict({"limits": bad})


BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def test_cli_default_adds_no_key_and_the_flags_set_the_config():
    from AG3S.trajopt import serve_safe

    args = serve_safe.build_parser().parse_args(BASE_ARGV)
    assert serve_safe.rest_start_overrides(args) == {}
    cfg = serve_safe.trajopt_config_from_args(args)
    assert cfg.limits.rest_start is False
    assert "rest_start" not in serve_safe.joint_limits_meta(cfg.limits)
    args = serve_safe.build_parser().parse_args(BASE_ARGV + ["--rest-start"])
    cfg = serve_safe.trajopt_config_from_args(args)
    assert cfg.limits.rest_start is True and cfg.limits.rest_start_tolerance is None
    assert serve_safe.joint_limits_meta(cfg.limits)["rest_start"] == {"tolerance_rad": None}
    args = serve_safe.build_parser().parse_args(
        BASE_ARGV + ["--rest-start", "--rest-start-tolerance", "0"])
    assert serve_safe.trajopt_config_from_args(args).limits.rest_start_tolerance == 0.0


@pytest.mark.parametrize("extra", [["--rest-start-tolerance", "0.01"],
                                   ["--rest-start", "--rest-start-tolerance", "-0.01"]])
def test_cli_rejects_meaningless_combinations(extra):
    from AG3S.trajopt import serve_safe

    args = serve_safe.build_parser().parse_args(BASE_ARGV + extra)
    with pytest.raises(SystemExit):
        serve_safe.trajopt_config_from_args(args)


# =============================================================================================
# 5. 기록 — T40 G E3b (있을 때만)
# =============================================================================================

records = pytest.mark.skipif(not RP.available(), reason="T40 G E3b 기록이 없다")
#: pre-T41b benchmark (T41 c 까지). 이 커밋의 trajopt 와 비트 동일해야 한다.
BASELINE = "b7d5849"


def _has_commit(rev):
    return subprocess.run(["git", "-C", _ag3s_workspace_path('legacy/benchmark'), "cat-file", "-e", rev],
                          capture_output=True).returncode == 0


@records
@pytest.mark.skipif(not _has_commit(BASELINE), reason=f"benchmark {BASELINE} 가 없다")
def test_off_replay_of_recorded_chunks_is_bit_identical_to_the_pre_t41b_code(tmp_path):
    root = tmp_path / "head"
    (root / "benchmark").mkdir(parents=True)
    archive = subprocess.run(["git", "-C", _ag3s_workspace_path('legacy/benchmark'), "archive", BASELINE, "--", ".",
                              ":(exclude)ag3s/docs"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(root / "benchmark")], input=archive, check=True)
    expose_legacy_benchmark(root)
    out = tmp_path / "head.npz"
    env = {**os.environ, "AG3S_ASSET_ROOT": _ag3s_workspace_path("legacy/pi05_TO_hybrid"), "AG3S_CLIENT_SCRIPT": _ag3s_workspace_path("integration/rby1/bringup/pi05_infer.py"), "PYTHONPATH": f"{root}:{_ag3s_workspace_path('')}"}
    proc = subprocess.run([sys.executable, str(pathlib.Path(RP.__file__)), str(out)], env=env,
                          cwd=_ag3s_workspace_path(''), capture_output=True, text=True, timeout=1200)
    assert proc.returncode == 0, proc.stderr[-3000:]
    head = np.load(out)
    meta = json.loads(str(head["meta"]))
    assert meta.pop("benchmark_file").startswith(str(root))      # 정말로 옛 코드가 돌았다
    for name, seq in RP.IDENTITY:
        r = RP.replay(name, seq)
        assert r["chunk"].tobytes() == head[f"{name}_{seq}_chunk"].tobytes(), (name, seq)
        assert meta[f"{name}_{seq}"]["status"] == r["status"]
        assert meta[f"{name}_{seq}"]["clearance_m"] == r["clearance_m"]
        # 켜도 정지가 아니면 (SafePolicy 가 키를 안 넣는다) 같은 해다.
        on = RP.replay(name, seq, limits={"rest_start": True})
        assert on["chunk"].tobytes() == r["chunk"].tobytes(), (name, seq)
        assert "rest_start" not in on["metrics"]


@records
def test_c1807_seq28_off_reproduces_the_10_8_degree_start_and_on_starts_from_rest():
    off = RP.replay("C1807", 28)
    a = float(off["max_step_change"].max())
    # 기록과 replay 모두 step 0 이 anchor 경계 v_max·dt (10.8°) 에 붙어 있다.
    assert off["recorded_step0_offset_rad"].max() == pytest.approx(10.8 * DEG, abs=0.01 * DEG)
    assert off["step0_offset_rad"].max() == pytest.approx(10.8 * DEG, abs=0.01 * DEG)
    assert int(np.sum(np.abs(off["recorded_step0_offset_rad"] - off["max_step"]) < 1e-4)) == 3
    assert off["feedback"]["n_exec"] == 0                        # 직전 seq 27 은 통째로 HOLD

    on = RP.replay("C1807", 28, limits={"rest_start": True}, at_rest=True)
    assert on["step0_offset_rad"].max() <= a + QP_TOL            # 2.29° 에서 시작
    assert on["metrics"]["rest_start"]["envelope_overshoot_rad"] <= QP_TOL
    assert on["safe"] and on["clearance_m"] > 0.0               # 범위 안에서 충돌을 푼다

    pinned = RP.replay("C1807", 28, limits={"rest_start": True, "rest_start_tolerance": 0.0},
                       at_rest=True)
    assert pinned["step0_offset_rad"].max() <= QP_TOL            # step 0 = q_now ...
    assert not pinned["safe"]                                    # ... 이면 못 풀어 거절된다


@records
def test_safe_policy_applies_rest_start_to_the_recorded_post_hold_chunk_only():
    sp = _policy(True)
    assert sp._rest_start(RP.summary("C1807", 28)["exec_feedback"])["applied"] is True   # seq 27 HOLD
    assert sp._rest_start(RP.summary("C1807", 25)["exec_feedback"])["applied"] is False  # seq 24 실행
