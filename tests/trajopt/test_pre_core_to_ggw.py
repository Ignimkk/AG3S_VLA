"""AG3S-PRE-CORE-TO-GGW — 쥐기 목표 항 목표 행 넓히기 (`--grasp-goal-window widen`) + 연속성 항은 쥔 뒤에만
(`--continuity-after-grasp`). 둘 다 기본 꺼짐.

Task: `AG3S/ag3s/docs/handoff/AG3S-PRE-CORE-TO.task.md` §6 "쥐기 목표 항 켜는 조건 넓히기 + 연속성 · 꺾임 항은 운반 · 놓기
단계에만" (사용자 판정 2026-10-28 01:07 KST). 보고: `AG3S/ag3s/docs/handoff/AG3S-PRE-CORE-TO-GGW.impl.md`.

이 파일이 고정하는 것:

1. **Config** — 기본 `close` · False, 값 검사, `grasp_continuity_off` (T37) 와 배타, 기본값이면 `to_dict` 에 키 없음 (기록 키 그대로).
2. **`GraspAnchor.goal_mask`** — `close` 는 예전 규칙 그대로; `widen` 은 close_row None · ≥ H → 마지막 G 행, 0 → 처음 G 행,
   0 < c < H → 그대로 (H = 8 · 16 둘 다). 목표가 없으면 행 없음.
3. **SQP (CPU)** — `close` 와 기본 anchor 의 해가 bit 동일; `widen` 의 c 안쪽은 `close` 와 bit 동일; `widen` 의 c 없음은
   마지막 행을 목표로 당기고 기록에 규칙 · 적용 행이 남는다; look-ahead 가 닫혀 실행 창만 풀 때 기록의 적용 행은 그 창의 행.
4. **E2 / E3-b GPU seed** — `_goal_probe` 와 `gpu_seeds.anchor_terms_for` 가 같은 넓힌 행을 쓴다.
5. **Anchor gate** — `widen` 이어도 켜짐 · 꺼짐 판정과 고정 목표가 `close` 와 같고, 기록에 horizon 별 목표 행이 더해진다.
6. **연속성 gate** — searching · latched · closing 에서 사유 (`before_grasp: …`), held · placed 에서 None; flag 꺼짐이면
   모든 단계에서 T37 그대로; refiner 에 가면 held 에서 항이 들어가고 쥐기 전에는 빠진다 (`continuity_off_by`).
7. **CLI** — flag → cost key, 꺼짐 = 키 없음, 거절 (widen 인데 anchor 없음 · w_continuity 0 · --no-safe), E3 + 껍질 구 +
   꼭지 덮개 + 쥐기 목표 항 + 꺾임 항과 함께 parse, `--help`, 시작 로그.
"""

from __future__ import annotations

import hashlib
import logging
import shlex
import sys
import types

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError
from AG3S.trajopt.grasp_anchor import GraspAnchor, anchor_record, anchor_targets
from AG3S.trajopt.grasp_latch import GraspPhase

pytest.importorskip("casadi")
pytest.importorskip("osqp")

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
G = 3


def _sha(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ 1. config
def test_config_defaults_check_exclusive_and_record_keys():
    c = TrajOptConfig()
    assert c.cost.grasp_goal_window == "close" and c.cost.continuity_after_grasp is False
    keys = set(c.to_dict()["cost"])
    assert "grasp_goal_window" not in keys and "continuity_after_grasp" not in keys   # off: the dump as before
    on = TrajOptConfig.from_dict({"cost": {"grasp_goal_window": "widen", "continuity_after_grasp": True}})
    d = on.to_dict()["cost"]
    assert d["grasp_goal_window"] == "widen" and d["continuity_after_grasp"] is True
    back = TrajOptConfig.from_dict(on.to_dict())
    assert back.cost.grasp_goal_window == "widen" and back.cost.continuity_after_grasp is True
    assert TrajOptConfig.from_dict(c.to_dict()).cost.grasp_goal_window == "close"
    for bad in ({"grasp_goal_window": "wide"}, {"continuity_after_grasp": "yes"},
                {"continuity_after_grasp": True, "grasp_continuity_off": True}):
        with pytest.raises(TrajOptConfigError):
            TrajOptConfig.from_dict({"cost": bad})


# ------------------------------------------------------------------------------------------------ 2. goal_mask
def _rows(anchor, H):
    return [int(k) for k in np.flatnonzero(anchor.goal_mask(H))]


@pytest.mark.parametrize("H", [8, 16])
def test_goal_mask_close_is_the_old_rule_and_widen_fills_the_gaps(H):
    g = np.zeros(3)
    for c in (None, 0, 1, H - 2, H - 1, H, H + 5):
        close = GraspAnchor("ee_left", g, c, 0.0, 2000.0)
        old = [] if c is None or not 0 < c < H else list(range(c, min(c + G, H)))
        assert _rows(close, H) == old, (H, c)
        assert _rows(GraspAnchor("ee_left", g, c, 0.0, 2000.0, goal_window="close"), H) == old
        widen = GraspAnchor("ee_left", g, c, 0.0, 2000.0, goal_window="widen")
        want = (list(range(H - G, H)) if c is None or c >= H else list(range(G)) if c == 0 else old)
        assert _rows(widen, H) == want, (H, c)
        assert _rows(GraspAnchor("ee_left", None, c, 0.0, 2000.0, goal_window="widen"), H) == []   # no goal: no rows
    # the per-row weights follow the mask (what the SQP, E2 and the GPU seeds all read)
    w, _ = anchor_targets(GraspAnchor("ee_left", g, None, 0.0, 2000.0, goal_window="widen"), np.zeros((H, 3)))
    assert [int(k) for k in np.flatnonzero(w[:, 1])] == list(range(H - G, H)) and not w[:, 0].any()


def test_records_say_the_rule_only_under_widen():
    p = np.zeros((8, 3))
    g = np.array([0.0, 0.0, 0.01])
    close = anchor_record(GraspAnchor("ee_left", g, 9, 0.0, 2000.0), p, p)
    assert not {"goal_window", "goal_window_rule", "horizon"} & set(close)
    rec = anchor_record(GraspAnchor("ee_left", g, 9, 0.0, 2000.0, goal_window="widen"), p, p)
    assert rec["goal_window"] == "widen" and rec["horizon"] == 8 and rec["goal_rows"] == [5, 6, 7]
    assert rec["goal_window_rule"] == "close row beyond this horizon: last rows"
    a0 = GraspAnchor("ee_left", g, 0, 0.0, 2000.0, goal_window="widen")
    assert a0.goal_window_rule(8) == "starts closed (close row 0): first rows"
    assert GraspAnchor("ee_left", g, None, 0.0, 1.0, goal_window="widen").goal_window_rule(8) == \
        "no close planned: last rows"
    assert GraspAnchor("ee_left", g, 3, 0.0, 1.0, goal_window="widen").goal_window_rule(8).startswith("close row inside")


# ------------------------------------------------------------------------------------------------ 3. SQP (CPU)
def _solve(sc, anchor=None, opt=None, **kw):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = opt or TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config)
    if anchor is not None:
        kw["grasp_anchor"] = anchor
    return opt.solve(sc.reference, sc.q_now, sc.scene, **kw)


def _p_ref(sc, link, H):
    from AG3S.trajopt.grasp_anchor import LinkPointFK

    return LinkPointFK(sc.robot, sc.layout, H, link)(sc.reference, sc.q_now)[0]


def test_sqp_close_is_bit_identical_and_widen_inside_equals_close():
    from tests.trajopt.fixtures import make_scenario
    from tests.trajopt.test_e1_grasp_goal import HEAD_SHA

    sc = make_scenario(horizon=8, penetration=0.03)
    goal = _p_ref(sc, "link_right_arm_5", 8)[-1] + np.array([0.0, 0.0, 0.01])
    # the E1 golden (HEAD 35b909cf) with the field given explicitly as "close"
    r = _solve(sc, GraspAnchor("link_right_arm_5", goal, 6, 0.0, 5000.0, goal_window="close"))
    assert _sha(r.trajectory) == HEAD_SHA["toa_goal"]
    assert not {"goal_window", "applied_horizon", "goal_rows_applied"} & set(r.metrics["grasp_anchor"])
    # widen, close row inside the horizon: the same solve (case c unchanged), plus the rule in the record
    w = _solve(sc, GraspAnchor("link_right_arm_5", goal, 6, 0.0, 5000.0, goal_window="widen"))
    assert _sha(w.trajectory) == HEAD_SHA["toa_goal"]
    rec = w.metrics["grasp_anchor"]
    assert rec["goal_window"] == "widen" and rec["goal_rows"] == [6, 7] and rec["goal_rows_applied"] == [6, 7]
    assert rec["applied_horizon"] == 8 and rec["goal_window_rule"].startswith("close row inside")


def test_sqp_widen_without_a_close_row_pulls_the_last_rows():
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)
    goal = _p_ref(sc, "link_right_arm_5", 8)[-1] + np.array([0.0, 0.0, 0.01])
    off = _solve(sc, GraspAnchor("link_right_arm_5", goal, None, 0.0, 5000.0))
    assert off.metrics["grasp_anchor"]["applied"] is False                          # close: no rows (as before)
    assert off.metrics["grasp_anchor"]["reason"] == "no close planned"
    on = _solve(sc, GraspAnchor("link_right_arm_5", goal, None, 0.0, 5000.0, goal_window="widen"))
    rec = on.metrics["grasp_anchor"]
    assert rec["goal_rows_applied"] == [5, 6, 7] and rec["n_goal_rows"] == 3
    assert rec["goal_window_rule"] == "no close planned: last rows"
    assert rec["goal_err_result_mm"] < rec["goal_err_reference_mm"]
    assert not np.array_equal(on.trajectory, off.trajectory)


def test_lookahead_closed_gate_records_the_window_rows():
    """Look-ahead (plan 10, execution 4) with a closed gate: only the window (H = 4) is solved — the record's applied rows
    are the window's last 3 rows, the outer record's own rows are those of H = 10."""
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario
    from tests.trajopt.test_e2_multi_seed import _head_on

    sc = make_scenario(horizon=10, penetration=0.03, sqp={"max_iterations": 1, "min_iterations": 1})
    scene = _head_on(sc, step=1)
    cfg = sc.config.with_overrides({"horizon": {"execution_length": 4, "lookahead": "obstacles", "lookahead_gate": 0.0},
                                    "collision": {"backend": "esdf"}})
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    goal = _p_ref(sc, "ee_right", 10)[3] + np.array([0.0, 0.0, 0.01])
    res = opt.solve(sc.reference, sc.q_now, scene,
                    grasp_anchor=GraspAnchor("ee_right", goal, 12, 0.0, 2000.0, goal_window="widen"))
    assert res.metrics["lookahead"]["gate"] == "closed"
    rec = res.metrics["grasp_anchor"]
    assert rec["horizon"] == 10 and rec["goal_rows"] == [7, 8, 9]
    assert rec["applied_horizon"] == 4 and rec["window"]["horizon"] == 4 and rec["window"]["goal_rows"] == [1, 2, 3]
    assert rec["window"]["goal_window_rule"] == "close row beyond this horizon: last rows"
    # the rows that shaped the returned trajectory: none when the window's safety-first plain re-solve won (this
    # stuck fixture: anchored 37.7 mm vs plain 13.4 mm left), else the window's goal rows
    won_plain = bool(rec["window"].get("fallback_used"))
    assert rec["goal_rows_applied"] == ([] if won_plain else [1, 2, 3])
    # a resolvable window (the fixture's own scene, gate always closed): the anchored window solve wins
    sc2 = make_scenario(horizon=10, penetration=0.03)                            # the default 6 SQP iterations
    cfg2 = sc2.config.with_overrides({"horizon": {"execution_length": 4, "lookahead": "obstacles",
                                                  "lookahead_gate": -1.0}, "collision": {"backend": "esdf"}})
    opt2 = TrajectoryOptimizer(sc2.robot, sc2.layout, sc2.limits, cfg2)
    r2 = opt2.solve(sc2.reference, sc2.q_now, sc2.scene,
                    grasp_anchor=GraspAnchor("ee_right", goal, 12, 0.0, 2000.0, goal_window="widen"))
    assert r2.metrics["lookahead"]["gate"] == "closed"
    w2 = r2.metrics["grasp_anchor"]
    assert w2["window"]["fallback_used"] is False and w2["goal_rows_applied"] == [1, 2, 3]
    assert w2["window"]["goal_err_result_mm"] < w2["window"]["goal_err_reference_mm"]
    # the same chunk under close: the window has no goal rows (the E3G "close row beyond this horizon" case)
    res0 = opt.solve(sc.reference, sc.q_now, scene, grasp_anchor=GraspAnchor("ee_right", goal, 12, 0.0, 2000.0))
    assert res0.metrics["grasp_anchor"]["applied"] is False
    assert res0.metrics["grasp_anchor"]["reason"] == "close row beyond this horizon"


# ------------------------------------------------------------------------------------------------ 4. E2 / E3-b GPU seed
@pytest.mark.parametrize("c", [None, 0, 12])
def test_e2_goal_probe_and_gpu_seed_terms_use_the_widened_rows(c):
    from AG3S.trajopt import gpu_seeds as GS
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config)
    goal_p = _p_ref(sc, "ee_right", 8)[-1]
    anchor = GraspAnchor("ee_right", goal_p, c, 0.0, 2000.0, goal_window="widen")
    want = [0, 1, 2] if c == 0 else [5, 6, 7]
    probe = opt._goal_probe(anchor, 8)
    assert [int(k) for k in probe["rows"]] == want
    terms = GS.anchor_terms_for(opt, anchor, sc.reference, sc.q_now, probe)
    assert [int(k) for k in np.flatnonzero(terms["weights"][:, 1])] == want
    np.testing.assert_allclose(terms["targets"][want, 1], np.tile(goal_p, (3, 1)))
    # close: no goal rows on this horizon -> no probe, no GPU goal term (as before)
    assert opt._goal_probe(GraspAnchor("ee_right", goal_p, c, 0.0, 2000.0), 8) is None
    # a chosen seed's record (E2 / E3-b) carries the same rows as applied
    seed_rec = opt._seed_anchor_record(anchor, sc.reference, sc.q_now, sc.reference, {})
    assert seed_rec["goal_rows"] == seed_rec["goal_rows_applied"] == want and seed_rec["applied_horizon"] == 8


# ------------------------------------------------------------------------------------------------ 5. anchor gate
def _gate(window):
    from tests.trajopt.test_ifg_toa_grasp_anchor import _gate_policy

    p, model = _gate_policy()
    if window is not None:
        p.to_config = TrajOptConfig.from_dict({"cost": {"w_grasp_tcp": 0.0, "w_grasp_goal": 2000.0,
                                                        "grasp_goal_window": window},
                                               "horizon": {"plan_horizon": 16, "lookahead": "obstacles"},
                                               "collision": {"backend": "esdf"}})
    return p, model


def test_anchor_gate_decisions_are_unchanged_under_widen():
    from AG3S.ag3s.constraints.attached import _link_pose_numeric
    from tests.trajopt.test_ifg_toa_grasp_anchor import _chunk

    seen = {}
    for window in ("close", "widen"):
        p, model = _gate(window)
        q = np.zeros(len(model.joint_names))
        palm = np.asarray(_link_pose_numeric(model, q, "ee_left")[:3, 3])
        p.ag3s.manipulated = types.SimpleNamespace(id=0, centroid=palm + np.array([0.0, 0.0, -0.03]))
        far = q.copy()
        far[[i for i, n in enumerate(model.joint_names) if n == "left_arm_1"][0]] = 0.8
        out = []
        for phase, chunk, qq in ((GraspPhase.LATCHED, _chunk(p), q),                 # no close yet: no goal
                                 (GraspPhase.LATCHED, _chunk(p, close_row=20), q),   # frozen, close row 20
                                 (GraspPhase.CLOSING, _chunk(p), q),                 # goal kept, no close row
                                 (GraspPhase.CLOSING, _chunk(p, close_row=0), q),    # starts closed
                                 (GraspPhase.CLOSING, _chunk(p, close_row=0), far),  # palm outside the radius
                                 (GraspPhase.HELD, _chunk(p, close_row=0), q)):      # held: cleared
            p._latch.phase = phase
            p._pending = {"chunk": chunk}
            a = p._grasp_anchor_gate(qq, ["left"])
            rec = dict(p._last_grasp_anchor)
            out.append((a is not None, None if a is None else a.close_row,
                        None if a is None else a.goal.tolist(), rec.get("reason")))
            if window == "close":
                assert not {"goal_window", "goal_rows_by_horizon"} & set(rec)
            elif a is not None:
                assert a.goal_window == "widen" and rec["goal_window"] == "widen"
                assert set(rec["goal_rows_by_horizon"]) == {"8", "16"}
                if a.close_row is None:
                    assert rec["goal_rows_by_horizon"] == {"8": [5, 6, 7], "16": [13, 14, 15]}
                elif a.close_row == 0:
                    assert rec["goal_rows_by_horizon"] == {"8": [0, 1, 2], "16": [0, 1, 2]}
                else:   # close row 20: beyond both horizons
                    assert rec["goal_rows_by_horizon"] == {"8": [5, 6, 7], "16": [13, 14, 15]}
        seen[window] = out
    assert seen["close"] == seen["widen"]
    assert [x[0] for x in seen["close"]] == [False, True, True, True, False, False]


# ------------------------------------------------------------------------------------------------ 6. continuity gate
def _cont_policy(flag, phase, w=0.5, t37=False):
    from AG3S.trajopt.safe_policy import SafePolicy

    p = object.__new__(SafePolicy)
    p.to_config = TrajOptConfig.from_dict({"cost": {"w_continuity": w, "continuity_after_grasp": flag,
                                                    "grasp_continuity_off": t37}})
    p._latch = types.SimpleNamespace(phase=phase)
    p._continuity_gate_on = False
    return p


def test_continuity_gate_phases_and_off_is_t37_as_before():
    from AG3S.trajopt import safe_policy as SP

    for ph in (GraspPhase.SEARCHING, GraspPhase.LATCHED, GraspPhase.CLOSING):
        why = _cont_policy(True, ph)._continuity_off_reason()
        assert why == f"before_grasp: grasp latch {ph.value} (cost.continuity_after_grasp)"
        assert SP.continuity_after_grasp_off_reason(ph.value, True) == why          # replay form (recorded state)
    for ph in (GraspPhase.HELD, GraspPhase.PLACED):
        assert _cont_policy(True, ph)._continuity_off_reason() is None
    assert _cont_policy(True, GraspPhase.LATCHED, w=0.0)._continuity_off_reason() is None
    # flag off: exactly T37's gate on every phase (off by default -> None; on -> its own reason)
    for ph in GraspPhase:
        for t37 in (False, True):
            p = _cont_policy(False, ph, t37=t37)
            assert p._continuity_off_reason() == _cont_policy(False, ph, t37=t37)._grasp_continuity_gate() \
                == SP.grasp_continuity_off_reason(ph, t37)


@pytest.fixture(scope="module")
def scenario():
    from tests.trajopt.fixtures import make_scenario

    return make_scenario(horizon=12, penetration=0.03)


def _refine(scenario, phase, flag):
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    prev = np.array(scenario.chunk, np.float64, copy=True)
    prev[:, scenario.layout.action_columns] += 0.15
    ref = TrajOptChunkRefiner(scenario.robot, scenario.layout, lambda ctx: (scenario.scene, scenario.q_now, True),
                              scenario.config)
    ctx = {"previous_physical_chunk": prev}
    off = _cont_policy(flag, phase)._continuity_off_reason()      # what `_scene_fn` writes
    if off:
        ctx["continuity_off"] = off
    ref.refine(scenario.chunk, ctx)
    return ref.last_result


def test_continuity_term_in_held_and_out_before_grasp(scenario):
    held_on, held_off = _refine(scenario, GraspPhase.HELD, True), _refine(scenario, GraspPhase.HELD, False)
    np.testing.assert_array_equal(held_on.trajectory, held_off.trajectory)
    assert held_on.metrics["continuity_active"] is True and held_on.metrics["continuity_off_by"] is None
    for ph in (GraspPhase.SEARCHING, GraspPhase.LATCHED, GraspPhase.CLOSING):
        on, off = _refine(scenario, ph, True), _refine(scenario, ph, False)
        assert off.metrics["continuity_active"] is True and off.metrics["continuity_off_by"] is None   # flag off
        np.testing.assert_array_equal(off.trajectory, held_off.trajectory)
        assert on.metrics["continuity_active"] is False
        assert on.metrics["continuity_off_by"] == f"before_grasp: grasp latch {ph.value} (cost.continuity_after_grasp)"
        assert np.abs(on.trajectory - off.trajectory).max() > 1e-4                 # the term did pull


# ------------------------------------------------------------------------------------------------ 7. CLI
def test_cli_maps_off_has_no_keys_and_refusals(monkeypatch, capsys):
    from AG3S.trajopt import serve_safe

    parse = serve_safe.build_parser().parse_args
    a = parse([*BASE_ARGV, "--grasp-anchor", "2000", "--grasp-anchor-path-weight", "0",
               "--grasp-goal-window", "widen", "--continuity-after-grasp"])
    assert serve_safe.cost_overrides(a) == {"w_grasp_tcp": 0.0, "w_grasp_goal": 2000.0, "grasp_goal_window": "widen",
                                            "continuity_after_grasp": True}
    base = serve_safe.cost_overrides(parse([*BASE_ARGV, "--grasp-anchor", "2000"]))
    assert serve_safe.cost_overrides(parse([*BASE_ARGV, "--grasp-anchor", "2000", "--grasp-goal-window", "close"])) == base
    assert serve_safe.cost_overrides(parse(BASE_ARGV)) == {}
    assert serve_safe.cost_overrides(parse([*BASE_ARGV, "--grasp-goal-box", "10", "--grasp-goal-window", "widen"])) \
        == {"grasp_goal_box": 0.01, "grasp_goal_window": "widen"}
    for extra, needle in ((["--grasp-goal-window", "widen"], "--grasp-goal-window"),
                          (["--continuity-after-grasp", "--w-continuity", "0"], "--continuity-after-grasp"),
                          (["--no-safe", "--continuity-after-grasp"], "--w-*"),
                          (["--no-safe", "--grasp-anchor", "2000", "--grasp-goal-window", "widen"], "--w-*")):
        monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, *extra])
        with pytest.raises(SystemExit) as e:
            serve_safe.main()
        assert e.value.code == 2
        assert needle in capsys.readouterr().err
    with pytest.raises(SystemExit):
        parse([*BASE_ARGV, "--grasp-goal-window", "wide"])


def test_e3_shell_stem_cap_goal_and_both_after_grasp_flags_parse_together(monkeypatch):
    from AG3S.trajopt import gpu_seeds, serve_safe
    from tests.trajopt.test_pre_core_to_port import E3_SAFE, SHELL, STEM_CAP, _build_kwargs

    monkeypatch.setattr(gpu_seeds, "available", lambda: (True, ""))
    goal = ["--grasp-anchor", "2000", "--grasp-anchor-path-weight", "0", "--deflection-rate-after-grasp"]
    new = ["--grasp-goal-window", "widen", "--continuity-after-grasp"]
    p = serve_safe.build_parser()
    a = p.parse_args(BASE_ARGV + shlex.split(E3_SAFE) + STEM_CAP + SHELL + goal + new)
    serve_safe.reject_bad_flag_combinations(p, a)
    cfg = serve_safe.trajopt_config_from_args(a)
    c = cfg.cost
    assert (c.grasp_goal_window, c.continuity_after_grasp, c.deflection_rate_after_grasp) == ("widen", True, True)
    assert (c.w_grasp_tcp, c.w_grasp_goal, c.w_deflection_rate) == (0.0, 2000.0, 20.0)
    assert cfg.sqp.gpu_seeds == "curobo" and cfg.sqp.multi_seed == 16 and cfg.collision.palm_shell_margin == 0.005
    assert cfg.horizon.planned == 16 and cfg.horizon.lookahead_from == 8
    kw = _build_kwargs(monkeypatch, a)
    assert kw["palm_shell"]["hands"] == "both" and kw["target_exclusion"]["target_stem_cap"] == "in_ball"
    # without the two new flags: the same config except the two keys (and the dump has no new keys)
    b = p.parse_args(BASE_ARGV + shlex.split(E3_SAFE) + STEM_CAP + SHELL + goal)
    serve_safe.reject_bad_flag_combinations(p, b)
    db, da = serve_safe.trajopt_config_from_args(b).to_dict(), cfg.to_dict()
    assert {k: v for k, v in da["cost"].items() if k not in ("grasp_goal_window", "continuity_after_grasp")} == db["cost"]
    assert {k: v for k, v in da.items() if k != "cost"} == {k: v for k, v in db.items() if k != "cost"}


def test_help_and_start_log(caplog):
    from AG3S.trajopt import serve_safe

    text = serve_safe.build_parser().format_help()
    assert "--grasp-goal-window" in text and "--continuity-after-grasp" in text
    cfg = TrajOptConfig.from_dict({"cost": {"w_grasp_goal": 2000.0, "grasp_goal_window": "widen",
                                            "continuity_after_grasp": True}})
    with caplog.at_level(logging.WARNING):
        serve_safe.announce_ta(cfg)
        serve_safe.announce_cost_weights({"grasp_goal_window": "widen", "continuity_after_grasp": True})
    log = caplog.text
    assert "AG3S-PRE-CORE-TO-GGW" in log and "widen" in log and "continuity after grasp" in log
    assert "grasp_goal_window: close → widen" in log and "continuity_after_grasp: 0 → 1" in log
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        serve_safe.announce_ta(TrajOptConfig.from_dict({"cost": {"w_grasp_goal": 2000.0}}))
    assert "AG3S-PRE-CORE-TO-GGW" not in caplog.text
