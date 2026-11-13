"""SL-GTTO11 F9a (lead G54, 2026-11-07): the scaled form of the F7 action QP (`--ee-action-qp-scaled`)."""
from __future__ import annotations

import numpy as np
import pytest

from AG3S.adapters.panda import profile as pf
from AG3S.adapters.panda import server as srv
from tests.trajopt.test_sl_gtto9_ee_action_qp import _Lin, _opt  # noqa: F401

V = "AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-qs"


def test_scaled_qp_gives_the_same_min_norm_correction():
    opt = _opt()
    opt.ee_action_scaled = True
    res = opt._solve_ee_action(np.zeros((2, 3)), np.zeros(2), None, None, 0.01, None, [], 0.0, True, None)
    da = opt.last_ee_action["delta_action"]
    assert np.allclose(da[:, 0], 0.01 / 3, atol=1e-4) and np.allclose(da[:, 1], 0.0, atol=1e-6)
    rec = res.metrics["ee_action_qp"]
    assert rec["slack_max_m"] <= 1e-5 and rec["n_rows_unreachable"][0] == 0 and rec["osqp_iter"][0] < 20000


def test_scaled_qp_keeps_the_box_and_reports_slack_in_metres():
    opt = _opt()
    opt.ee_action_scaled = True
    opt.ee_action_map["a_ref"] = np.full((3, 2), 0.999)
    res = opt._solve_ee_action(np.zeros((2, 3)), np.zeros(2), None, None, 0.01, None, [], 0.0, True, None)
    da = opt.last_ee_action["delta_action"]
    assert np.all(opt.ee_action_map["a_ref"] + da <= 1.0 + 1e-9)
    assert res.metrics["ee_action_qp"]["slack_max_m"] == pytest.approx(0.01 - 0.003, abs=2e-4)


def test_unreachable_rows_leave_the_qp():
    opt = _opt()
    opt.ee_action_scaled = True
    opt.ee_action_map["M"][:] = 0.0          # no action reaches any step: the row is unreachable
    res = opt._solve_ee_action(np.zeros((2, 3)), np.zeros(2), None, None, 0.01, None, [], 0.0, True, None)
    rec = res.metrics["ee_action_qp"]
    assert rec["n_rows_unreachable"] == [1] and "no reachable row" in rec["status"]
    assert np.allclose(opt.last_ee_action["delta_action"], 0.0)


def test_default_is_the_f7_form():
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    assert TrajectoryOptimizer.ee_action_scaled is False
    opt = _opt()
    res = opt._solve_ee_action(np.zeros((2, 3)), np.zeros(2), None, None, 0.01, None, [], 0.0, True, None)
    assert "n_rows_unreachable" not in res.metrics["ee_action_qp"]


def test_version_and_flag():
    assert pf.GT11_VERSIONS == (V,)
    assert list(pf.AG3S_VERSIONS[V]) == list(pf.AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp"]) + ["--ee-action-qp-scaled", "--grasp-opening-rule-moved", "0.05"]
    assert "ee_action_qp_scaled" in srv.PORTED_FINAL_CONFIG_DESTS and pf.is_gtto_version(V)
    args = pf.parse_v11ab_args(argv=pf.version_server_argv(V) + list(pf.AG3S_VERSIONS[V]))
    assert args.ee_action_qp_scaled is True and pf.version_flag_values(args)["ee_action_qp_scaled"] is True
    for name, flags in pf.AG3S_VERSIONS.items():
        if name != V and name not in pf.SOTA_SCALED_VERSIONS + pf.SOTA_PEAK_VERSIONS + pf.SOTA_ROTREL_VERSIONS + pf.SOTA_REACHSTICKY_VERSIONS + pf.SOTA_ZONE_VERSIONS + pf.SOTA_ZONEPALM_VERSIONS + pf.SOTA_ZONEPALM5_VERSIONS + pf.SOTA_DEADCAM_VERSIONS + pf.SOTA_HELDBAND_VERSIONS + pf.SOTA_LSNATIVE_VERSIONS + pf.SOTA_HANDSIZED_VERSIONS + pf.SOTA_HANDSIZED_ATT_VERSIONS + pf.SOTA_GTO_VERSIONS + pf.SOTA_SPLITWIDE_VERSIONS + pf.SOTA_ATT_VERSIONS + pf.SOTA_AUDIT_VERSIONS:   # SL-SOTA6 (2026-11-11): the ``-qs`` SOTA versions
            assert "--ee-action-qp-scaled" not in flags, name


# --------------------------------------------------------------------------- (c') the opening rule's `moved` check
from tests.benchmark.test_sl14_opening_rule import _grasp, _latch, _sig  # noqa: E402
from AG3S.trajopt.grasp_latch import GraspPhase  # noqa: E402


def test_moved_off_is_sl14():
    latch = _latch()
    ev = _grasp(latch, opening=0.04)
    assert ev.attach and "moved" not in ev.evidence["checks"] and "open_ref" not in ev.evidence["opening_rule"]


def test_first_close_call_with_open_fingers_does_not_attach():
    """F8 LS t0e3: open 0.998, then the first closed-command call reads 0.989 (settled: diff < 0.02)."""
    for moved, expect in ((None, True), (0.05, False)):
        latch = _latch(**({} if moved is None else {"opening_rule_moved_min": moved}))
        latch.observe_grasp(_sig(1.0, 0.998), reach_m=0.11)
        ev = latch.observe_grasp(_sig(0.0034, 0.989), reach_m=0.11)
        assert ev.attach is expect


def test_moved_fingers_attach():
    latch = _latch(opening_rule_moved_min=0.05)
    ev = _grasp(latch, opening=0.06)
    assert ev.attach and ev.evidence["checks"]["moved"] is True and ev.evidence["opening_rule"]["open_ref"] == 1.0
    assert latch.phase is GraspPhase.HELD


def test_moved_flag_and_latch_config():
    from AG3S.adapters.panda.server import panda_latch_config
    from AG3S.trajopt import serve_safe as ss
    assert panda_latch_config().opening_rule_moved_min is None
    assert panda_latch_config(opening_rule=True, opening_rule_moved=0.05).opening_rule_moved_min == 0.05
    act = ss.build_parser()._option_string_actions["--grasp-opening-rule-moved"]
    assert act.default is None and "grasp_opening_rule_moved" in srv.PORTED_FINAL_CONFIG_DESTS
    args = pf.parse_v11ab_args(argv=pf.version_server_argv(V) + list(pf.AG3S_VERSIONS[V]))
    assert args.grasp_opening_rule_moved == 0.05 and pf.version_flag_values(args)["grasp_opening_rule_moved"] == 0.05
