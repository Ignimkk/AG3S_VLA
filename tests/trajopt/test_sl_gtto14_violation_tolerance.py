"""SL-GTTO14 (lead G69, 2026-11-07) — `serve_safe --violation-tolerance-mm TAU`: the accepted model violation.

User ruling 2026-11-07 19:1x: a LIBERO episode the VLA alone succeeds must succeed with the module. On the VLA's own
collision-free LIBERO path (GT12 observe) the sphere model + 5 mm field read violations of up to ~25 mm. TAU sets
`sqp.passthrough_tolerance` (the pass-through accepts a reference violation <= TAU, T88's test is > 0) and
`safety.violation_tolerance` (the final check / verdict accept the same). Pinned here:
1. off (flag absent): no new config key, the T88 pass-through test is `> 0` as before;
2. on: a reference that violates by less than TAU goes out unchanged (no optimization step) with status safe; one that
   violates by more than TAU is optimized as before;
3. the CLI checks (needs --passthrough-when-clear, TAU > 0) and the Panda server ports the dest.
"""
from __future__ import annotations

import pytest

from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError

pytest.importorskip("casadi")
pytest.importorskip("osqp")

from tests.trajopt.test_t88_passthrough_when_clear import BASE, _inside, _no_qp, _opt  # noqa: E402


def _violating(pen):
    from tests.trajopt.fixtures import make_scenario

    return _inside(make_scenario(horizon=8), pen)


def test_default_has_no_new_key():
    c = TrajOptConfig()
    assert c.sqp.passthrough_tolerance == 0.0 and c.safety.violation_tolerance == pytest.approx(1e-4)
    assert "passthrough_tolerance" not in c.to_dict()["sqp"]
    on = c.with_overrides({"sqp": {"passthrough_tolerance": 0.02}})
    assert TrajOptConfig.from_dict(on.to_dict()) == on
    with pytest.raises(TrajOptConfigError, match="passthrough_tolerance"):
        TrajOptConfig.from_dict({"sqp": {"passthrough_tolerance": -0.001}})


def test_cli_sets_both_tolerances_and_absent_changes_nothing():
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    plain = ss.trajopt_config_from_args(ap.parse_args(BASE + ["--passthrough-when-clear"]))
    a = ap.parse_args(BASE + ["--passthrough-when-clear", "--violation-tolerance-mm", "25"])
    on = ss.trajopt_config_from_args(a)
    assert on.sqp.passthrough_tolerance == pytest.approx(0.025) and on.safety.violation_tolerance == pytest.approx(0.025)
    assert on == plain.with_overrides({"sqp": {"passthrough_tolerance": 0.025}, "safety": {"violation_tolerance": 0.025}})
    assert plain.sqp.passthrough_tolerance == 0.0
    for bad in (["--violation-tolerance-mm", "25"], ["--passthrough-when-clear", "--violation-tolerance-mm", "0"]):
        with pytest.raises(SystemExit):
            ss.reject_bad_flag_combinations(ap, ap.parse_args(BASE + bad))


def test_reference_within_tau_passes_unchanged(monkeypatch):
    sc = _violating(0.01)                       # the obstacle 10 mm inside the floor: the reference violates by ~33 mm
    assert 0.0 < sc.reference_violation < 0.04
    tau = {"sqp": {"passthrough_tolerance": 0.04}, "safety": {"violation_tolerance": 0.04}}
    _no_qp(monkeypatch)
    out = _opt(sc, on=True, **tau).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    rec = out.metrics["passthrough_clear"]
    assert rec["applied"] is True and rec["reason"] == "clear" and rec["tolerance_m"] == pytest.approx(0.04)
    assert out.trajectory.tobytes() == sc.reference.astype(out.trajectory.dtype).tobytes()
    assert out.status.safe


def test_reference_beyond_tau_is_optimized_as_without_it():
    sc = _violating(0.03)
    assert sc.reference_violation > 0.01
    off = _opt(sc, on=True).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    on = _opt(sc, on=True, sqp={"passthrough_tolerance": 0.005}).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    assert on.metrics["passthrough_clear"]["applied"] is False and on.metrics["passthrough_clear"]["reason"] == "collision"
    assert on.trajectory.tobytes() == off.trajectory.tobytes()


def test_panda_server_ports_the_dest():
    from AG3S.adapters.panda import server as srv

    assert "violation_tolerance_mm" in srv.PORTED_FINAL_CONFIG_DESTS


# ---- SL-GTTO15 (G70): --passthrough-limit-tolerance · --oracle-robot-mask --------------------------------------------
def _limit_case(sc, opt):
    """A reference that overshoots only a limit (no collision row): one joint shifted at every step until the step-0
    anchor overshoots while the chunk stays clear. `None` when the fixture has no such joint."""
    import numpy as np

    for j in range(sc.reference.shape[0]):
        for eps in (0.05, 0.1, 0.2, 0.3, 0.5, -0.05, -0.1, -0.2, -0.3, -0.5):
            ref = sc.reference.copy()
            ref[j, :] += eps
            if max(opt._limit_overshoot(ref, sc.q_now).values()) > 1e-6:
                out = opt.solve(ref, sc.q_now, sc.scene, template=sc.chunk)
                if out.metrics["passthrough_clear"]["reason"] == "limit":
                    return ref
                break
    return None


def test_small_limit_overshoot_passes_with_the_tolerance_only(monkeypatch):
    from tests.trajopt.test_t88_passthrough_when_clear import _clear

    sc = _clear()
    ref = _limit_case(sc, _opt(sc, on=True))
    assert ref is not None, "no limit-only reference in the fixture"
    over = max(_opt(sc, on=True)._limit_overshoot(ref, sc.q_now).values())
    _no_qp(monkeypatch)
    loose = _opt(sc, on=True, sqp={"passthrough_limit_tolerance": 2 * over}).solve(ref, sc.q_now, sc.scene, template=sc.chunk)
    rec = loose.metrics["passthrough_clear"]
    assert rec["applied"] is True and rec["limit_tolerance"] == pytest.approx(2 * over) and rec["limit_overshoot"]


def test_limit_tolerance_cli_and_default():
    from AG3S.trajopt import serve_safe as ss

    assert TrajOptConfig().sqp.passthrough_limit_tolerance == 0.0
    assert "passthrough_limit_tolerance" not in TrajOptConfig().to_dict()["sqp"]
    ap = ss.build_parser()
    a = ap.parse_args(BASE + ["--passthrough-when-clear", "--passthrough-limit-tolerance", "0.01"])
    assert ss.trajopt_config_from_args(a).sqp.passthrough_limit_tolerance == pytest.approx(0.01)
    with pytest.raises(SystemExit):
        ss.reject_bad_flag_combinations(ap, ap.parse_args(BASE + ["--passthrough-limit-tolerance", "0.01"]))


def test_robot_mask_adds_only_code0_pixels_with_depth():
    import dataclasses

    import numpy as np

    from AG3S.ag3s.runtime.pipeline import AG3S

    @dataclasses.dataclass
    class Cam:
        name: str
        depth: np.ndarray
        robot_mask: object = None
        device_depth: object = "gpu"

    lab = np.array([[0, 0, 1], [2, 0, 3]], np.uint8)
    depth = np.array([[0.5, np.inf, 0.5], [0.5, 0.0, 0.5]], np.float32)   # (0,1) no hit, (1,1) no depth
    p = AG3S.__new__(AG3S)
    p._grounding_frame = {}
    p._oracle_pending = {"labels": {"head": lab}}
    out = p._oracle_robot_mask([Cam("head", depth)])
    assert out[0].robot_mask.tolist() == [[True, False, False], [False, False, False]]
    assert out[0].device_depth is None and p._grounding_frame["robot_mask"]["n_px_new"]["head"] == 1
    old = np.array([[True, False, False], [False, False, False]])
    same = p._oracle_robot_mask([Cam("head", depth, robot_mask=old)])[0]
    assert same.device_depth == "gpu"                                     # nothing new: the camera is untouched


def test_robot_mask_needs_the_oracle_grounding():
    from AG3S.ag3s.config import AG3SConfig, AG3SConfigError

    with pytest.raises(AG3SConfigError, match="oracle_robot_mask"):
        AG3SConfig.from_dict({"clustering": {"oracle_robot_mask": True}})
    from AG3S.adapters.panda import server as srv

    assert {"oracle_robot_mask", "passthrough_limit_tolerance"} <= set(srv.PORTED_FINAL_CONFIG_DESTS)
