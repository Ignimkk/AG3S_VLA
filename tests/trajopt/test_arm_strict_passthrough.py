"""ARMSTRICT (2026-11-15) -- `serve_safe --passthrough-arm-strict` / `sqp.passthrough_arm_strict` (default False).

In the T88 pass-through test (`TrajectoryOptimizer._clear_passthrough`) the rows of the robot's ARM links (Panda
`panda_link1..7`; not the hand, fingers or held-object rows) are held to `passthrough_obstacle_tolerance` whatever their
nearest surface is; before, an arm row next to a surface not classified as an obstacle object was accepted down to
`passthrough_tolerance` (100 mm in the frozen version). Measured 2026-11-14 with L12H3: link5 / link7 grazed the obstacle
beside the target in chunks passed as `clear`.

Pinned here:
1. off (default): no config key, no record key, the same result and record as before (the new code is never reached);
2. on: an arm row below its floor by more than the tolerance refuses with reason `arm` and the call goes on exactly as
   for any refused chunk (bit-identical to the solve without pass-through); a hand / finger row of the same depth is
   still accepted; obstacle-object rows are decided as before (that test runs first);
3. the REAL linearizer and rows (Panda constraint model, a real `EsdfField`, nothing of the linearizer replaced) --
   the lesson of SL-AUD1, which passed on stubs of the wrong shape and never fired on the server;
4. what the action QP does after an `arm` refusal (recorded, not changed);
5. the CLI, the Panda port and the versions of 2026-11-15 (Tab. 2 `-lsnative`, `-gh156a-a12h3`, `-pas`).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError

pytest.importorskip("casadi")
pytest.importorskip("osqp")

from tests.trajopt.test_t88_passthrough_when_clear import BASE, _clear, _inside, _opt  # noqa: E402

TAU = {"sqp": {"passthrough_tolerance": 0.04, "passthrough_obstacle_tolerance": 0.0},
       "safety": {"violation_tolerance": 0.04}}
STEM = ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-"
        "pg4-hr150-hb-qs-rs")
X = STEM + "-gh156a-tro"
XF = STEM + "-gh156a-fwd-tro"


def _violating(pen):
    from tests.trajopt.fixtures import make_scenario

    return _inside(make_scenario(horizon=8), pen)


def _tau(arm: bool) -> dict:
    sqp = dict(TAU["sqp"], **({"passthrough_arm_strict": True} if arm else {}))
    return {"sqp": sqp, "safety": TAU["safety"]}


def _solve(sc, *, on=True, **over):
    return _opt(sc, on=on, **over).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)


# ------------------------------------------------------------------------------------------------ 1. config / off
def test_default_is_off_and_the_dump_has_no_new_key():
    c = TrajOptConfig()
    assert c.sqp.passthrough_arm_strict is False and "passthrough_arm_strict" not in c.to_dict()["sqp"]
    on = c.with_overrides({"sqp": {"passthrough_when_clear": True, "passthrough_obstacle_tolerance": 0.0,
                                   "passthrough_arm_strict": True}})
    assert on.to_dict()["sqp"]["passthrough_arm_strict"] is True and TrajOptConfig.from_dict(on.to_dict()) == on
    for bad in ({"passthrough_arm_strict": 1, "passthrough_when_clear": True, "passthrough_obstacle_tolerance": 0.0},
                {"passthrough_arm_strict": True, "passthrough_obstacle_tolerance": 0.0},      # no pass-through test
                {"passthrough_arm_strict": True, "passthrough_when_clear": True}):            # no tolerance to hold to
        with pytest.raises(TrajOptConfigError, match="passthrough_arm_strict"):
            TrajOptConfig.from_dict({"sqp": bad})


def test_arm_link_names():
    from AG3S.trajopt.sqp import arm_link_row_mask

    names = ["panda_link1", "panda_link4", "panda_link5", "panda_link7", "panda_hand", "panda_leftfinger",
             "panda_rightfinger", "panda_link0", "link_right_arm_3", "ee_right", "ee_finger_r1", "link_torso_2"]
    assert arm_link_row_mask(names, len(names)).tolist() == [True, True, True, True, False, False, False, False, True,
                                                             False, False, False]
    with pytest.raises(ValueError, match="arm rows"):
        arm_link_row_mask(names, len(names) + 1)


@pytest.mark.parametrize("pen", [-0.05, 0.01])
def test_off_never_reaches_the_new_code_and_nothing_moves(monkeypatch, pen):
    from AG3S.trajopt import sqp as sqp_mod

    sc = _violating(pen)
    before = _solve(sc, **_tau(False))

    def boom(*a, **k):
        raise AssertionError("ARMSTRICT code reached with the switch off")

    monkeypatch.setattr(sqp_mod, "arm_link_row_mask", boom)
    after = _solve(sc, **_tau(False))
    explicit = _solve(sc, sqp={**TAU["sqp"], "passthrough_arm_strict": False}, safety=TAU["safety"])
    for out in (after, explicit):
        assert out.trajectory.tobytes() == before.trajectory.tobytes() and out.chunk.tobytes() == before.chunk.tobytes()
        assert out.metrics["passthrough_clear"] == before.metrics["passthrough_clear"]
        assert "arm_rows" not in out.metrics["passthrough_clear"] and set(out.metrics) == set(before.metrics)


# ------------------------------------------------------------------------------------------------ 2. on (RB-Y1 fixture)
def test_on_a_clear_chunk_passes_with_one_more_record_key():
    sc = _clear()
    off = _solve(sc, **_tau(False))
    on = _solve(sc, **_tau(True))
    r_off, r_on = off.metrics["passthrough_clear"], on.metrics["passthrough_clear"]
    assert r_on["applied"] is True and r_on["reason"] == "clear"
    assert set(r_on) - set(r_off) == {"arm_rows"} and {k: v for k, v in r_on.items() if k != "arm_rows"} == r_off
    arm = r_on["arm_rows"]
    assert arm["n_violating"] == 0 and arm["worst_m"] > 0.0 and arm["link"].startswith("link_") and "_arm_" in arm["link"]
    assert on.trajectory.tobytes() == off.trajectory.tobytes()


def test_on_an_arm_row_inside_tau_is_refused_like_any_refused_chunk():
    """The fixture's obstacle (a candidate slot: no field, so no obstacle-object row) clips an ARM sphere (33 mm after the
    fixture's 80 % sweep): tau (40 mm) passes it, the arm rows' tolerance (0) does not. The refused call is the solve without pass-through, bit for bit."""
    sc = _violating(0.01)
    _, pair = sc.linearizer.worst_row(sc.reference, sc.q_now, sc.scene)
    assert "_arm_" in pair["link"] and 0.0 < sc.reference_violation < 0.04    # an arm sphere, inside tau
    off = _solve(sc, **_tau(False))
    assert off.metrics["passthrough_clear"]["applied"] is True                    # before: accepted within tau
    on = _solve(sc, **_tau(True))
    rec = on.metrics["passthrough_clear"]
    assert rec["applied"] is False and rec["reason"] == "arm"
    assert rec["arm_rows"]["n_violating"] >= 1 and rec["arm_rows"]["worst_m"] == pytest.approx(-sc.reference_violation)
    assert "_arm_" in rec["arm_rows"]["link"] and rec["arm_rows"]["tolerance_m"] == 0.0
    plain = _solve(sc, on=False, **_tau(False))                              # no pass-through at all
    assert on.trajectory.tobytes() == plain.trajectory.tobytes() and on.iterations == plain.iterations > 0


def test_obstacle_rows_are_decided_first_and_as_before(monkeypatch):
    from AG3S.trajopt.linearize import CollisionLinearizer as Linearizer

    pair = {"step": 1, "link": "held_0", "obstacle": None, "clearance_m": -0.02, "point_m": [0, 0, 0],
            "n_obstacle_rows": 3, "n_rows": 10}
    monkeypatch.setattr(Linearizer, "obstacle_object_rows", lambda self, *a, **k: (-0.02, pair))
    sc = _violating(0.01)
    off, on = _solve(sc, **_tau(False)), _solve(sc, **_tau(True))
    assert on.metrics["passthrough_clear"] == off.metrics["passthrough_clear"]
    assert on.metrics["passthrough_clear"]["reason"] == "obstacle" and "arm_rows" not in on.metrics["passthrough_clear"]
    assert on.trajectory.tobytes() == off.trajectory.tobytes()


def test_unreadable_link_names_fail_closed(monkeypatch):
    sc = _violating(-0.05)
    opt = _opt(sc, on=True, **_tau(True))
    monkeypatch.setattr(type(opt.linearizer.robot_model), "sphere_link_names", property(lambda self: ("x",)), raising=False)
    out = opt.solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    rec = out.metrics["passthrough_clear"]
    assert rec["reason"] == "arm" and rec["applied"] is False and "error" in rec["arm_rows"]


# ------------------------------------------------------------------------------------------------ 3. the real Panda rows
H = 8
Q0 = np.array([0.0, 0.17, 0.0, -2.45, 0.0, 2.6, 0.79])


@pytest.fixture(scope="module")
def panda():
    """The Panda constraint model of the final versions (`--links arms`: panda_link1..7 + hand + fingers), its layout and
    the real collision linearizer; a short reference (joint 1 turns 0.02 rad)."""
    pytest.importorskip("mujoco")
    from types import SimpleNamespace

    from AG3S.adapters.panda import action as pa
    from AG3S.adapters.panda import model as pm
    from AG3S.trajopt.linearize import CollisionLinearizer

    robot = pm.build_panda_constraint_model(pm.panda_constraint_link_filter("arms"))
    layout = pa.panda_chunk_layout(robot.joint_names)
    lin = CollisionLinearizer(robot, layout, H)
    ref = np.tile(Q0[:, None], (1, H))
    ref[0] += np.linspace(0.0, 0.02, H)
    chunk = np.zeros((H, layout.action_dim))
    chunk[:, layout.action_columns] = ref.T
    return SimpleNamespace(robot=robot, layout=layout, lin=lin, ref=ref, chunk=chunk,
                           names=list(robot.sphere_link_names),
                           centres=np.asarray(lin.sphere_states(ref, Q0)[0]))


def _field_scene(w, link, direction, depth=0.015, radius=0.01):
    """A real `EsdfField` (an analytic 10 mm sphere) placed beside the first sphere of `link`, moved until the worst row
    of the reference is `depth` inside its floor; its surface is NOT an obstacle object (the set's one point is far)."""
    from tests.trajopt.fixtures import empty_scene
    from tests.trajopt.test_esdf_backend import analytic_field, scene_with_field

    k = w.names.index(link)
    u = np.asarray(direction, float) / np.linalg.norm(direction)
    off = float(w.lin.robot_radii[k]) + radius - depth
    for _ in range(4):
        field = analytic_field(w.centres[0, k] + u * off, radius, voxel=0.005, half_extent=0.35)
        scene = dataclasses.replace(scene_with_field(empty_scene(w.lin), field, 0.0),
                                    obstacle_object_points=np.array([[9.0, 9.0, 9.0]]))
        worst, pair = w.lin.worst_row(w.ref, Q0, scene)
        off += -float(worst) - depth
    assert worst == pytest.approx(-depth, abs=1e-3) and pair["link"] == link
    return scene


def _panda_opt(w, arm, action_map=False, **sqp):
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt.limits import build_limits
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import small_config

    cfg = small_config(H, sqp={"passthrough_when_clear": True, "passthrough_tolerance": 0.1,
                               "passthrough_obstacle_tolerance": 0.0, "passthrough_obstacle_full_horizon": True,
                               **({"passthrough_arm_strict": True} if arm else {}), **sqp},
                       safety={"violation_tolerance": 0.1},
                       limits={"source": "urdf", "model_xml": None, "default_acceleration": srv.PANDA_DEFAULT_ACCELERATION})
    opt = TrajectoryOptimizer(w.robot, w.layout, build_limits(w.robot, w.layout, dt=cfg.horizon.dt, config=cfg.limits), cfg)
    if action_map:   # the action QP of the final versions (one action = 0.05 rad of one joint), scaled form
        nq = w.ref.shape[0]
        opt.ee_action_map = {"M": 0.05 * np.eye(nq * H), "a_ref": np.zeros((H, nq))}
        opt.ee_action_scaled = True
    return opt


def test_real_rows_the_arm_link_is_refused(panda):
    w = panda
    assert sum(n.startswith("panda_link") for n in w.names) == 37 and "panda_link0" not in w.names
    scene = _field_scene(w, "panda_link5", (0.0, 1.0, 0.0))
    off = _panda_opt(w, False).solve(w.ref, Q0, scene, template=w.chunk)
    assert off.metrics["passthrough_clear"]["reason"] == "clear"           # before: accepted within tau (100 mm)
    assert off.metrics["passthrough_clear"]["obstacle_rows"]["violation_m"] == 0.0
    on = _panda_opt(w, True).solve(w.ref, Q0, scene, template=w.chunk)
    rec = on.metrics["passthrough_clear"]
    assert rec["applied"] is False and rec["reason"] == "arm"
    arm = rec["arm_rows"]
    assert set(arm) >= {"n_violating", "worst_m", "link"} and arm["link"] == "panda_link5"
    assert arm["n_violating"] >= H and arm["worst_m"] == pytest.approx(-0.015, abs=1e-3)
    assert arm["n_arm_spheres"] == 37 and "error" not in arm


@pytest.mark.parametrize("link,direction", [("panda_leftfinger", (0.0, 0.0, -1.0)), ("panda_hand", (0.0, 1.0, 0.0))])
def test_real_rows_a_hand_or_finger_row_of_the_same_depth_still_passes(panda, link, direction):
    w = panda
    scene = _field_scene(w, link, direction)
    out = _panda_opt(w, True).solve(w.ref, Q0, scene, template=w.chunk)
    rec = out.metrics["passthrough_clear"]
    assert rec["applied"] is True and rec["reason"] == "clear" and out.trajectory.tobytes() == w.ref.tobytes()
    assert rec["arm_rows"]["n_violating"] == 0 and rec["arm_rows"]["worst_m"] > 0.0
    assert rec["reference_violation_m"] == pytest.approx(0.015, abs=1e-3)


def test_real_rows_the_action_qp_after_an_arm_refusal(panda):
    """Recorded, not changed: after `arm` the action QP holds a non-obstacle-class arm row at +backoff (the row is
    repaired); with SL-AUD1 `-rf` it is held at min(backoff, its clearance on the policy chunk) -- not repaired."""
    w = panda
    scene = _field_scene(w, "panda_link5", (0.0, 1.0, 0.0))
    for extra, repaired in (({}, True), ({"obstacle_row_margin": 0.015, "obstacle_row_priority": 10.0}, True),
                            ({"ee_action_reference_floor": True}, False)):
        out = _panda_opt(w, True, action_map=True, **extra).solve(w.ref, Q0, scene, template=w.chunk)
        assert out.metrics["passthrough_clear"]["reason"] == "arm" and "ee_action_qp" in out.metrics
        worst, _ = w.lin.worst_row(out.trajectory, Q0, scene)
        if repaired:
            assert worst > 0.0 and out.metrics["ee_action_qp"]["max_abs_delta_action"] > 0.1
        else:
            assert worst == pytest.approx(-0.015, abs=1e-3) and out.metrics["ee_action_qp"]["max_abs_delta_action"] < 1e-6


# ------------------------------------------------------------------------------------------------ 5. CLI, port, versions
def test_cli_and_port():
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    plain = ap.parse_args(BASE + ["--passthrough-when-clear", "--passthrough-obstacle-tolerance-mm", "0"])
    assert "passthrough_arm_strict" not in ss.passthrough_overrides(plain)
    a = ap.parse_args(BASE + ["--passthrough-when-clear", "--passthrough-obstacle-tolerance-mm", "0", "--passthrough-arm-strict"])
    ss.reject_bad_flag_combinations(ap, a)
    cfg = ss.trajopt_config_from_args(a)
    assert cfg.sqp.passthrough_arm_strict is True
    assert cfg == ss.trajopt_config_from_args(plain).with_overrides({"sqp": {"passthrough_arm_strict": True}})
    for bad in (["--passthrough-arm-strict"], ["--passthrough-when-clear", "--passthrough-arm-strict"],
                ["--passthrough-obstacle-tolerance-mm", "0", "--passthrough-arm-strict"]):
        with pytest.raises(SystemExit):
            ss.reject_bad_flag_combinations(ap, ap.parse_args(BASE + bad))
    assert "passthrough_arm_strict" in srv.PORTED_FINAL_CONFIG_DESTS


def test_versions_of_2026_11_15():
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss

    a12, a12f = STEM + "-gh156a-a12h3-tro", STEM + "-gh156a-a12h3-fwd-tro"
    assert (pf.SOTA_FINAL_X, pf.SOTA_FINAL_X_FWD) == (X, XF)
    assert pf.SOTA_TAB2_LSNATIVE_VERSIONS == (X + "-lsnative", XF + "-lsnative")
    assert pf.SOTA_FINAL_ATT_VERSIONS == (a12, a12f)
    pas = (STEM + "-gh156a-pas-tro", STEM + "-gh156a-a12h3-pas-tro", STEM + "-gh156a-fwd-pas-tro",
           STEM + "-gh156a-a12h3-fwd-pas-tro")
    assert pf.SOTA_ARMSTRICT_VERSIONS == pas
    assert pf.SOTA_FINAL_NEXT_VERSIONS == pf.SOTA_TAB2_LSNATIVE_VERSIONS + pf.SOTA_FINAL_ATT_VERSIONS + pas
    assert set(pf.SOTA_FINAL_NEXT_VERSIONS) <= set(pf.GT15_VERSIONS) and not set(pf.SOTA_FINAL_NEXT_VERSIONS) & set(pf.SOTA_VERSIONS)
    assert len(set(pf.SOTA_FINAL_NEXT_VERSIONS)) == 8
    V = pf.AG3S_VERSIONS
    # each = its base argv + the segment's flags only
    assert V[X + "-lsnative"] == tuple(V[X]) + pf.LSNATIVE_FLAGS and V[XF + "-lsnative"] == tuple(V[XF]) + pf.LSNATIVE_FLAGS
    assert V[a12] == tuple(V[X]) + ("--attention-source", "action:last:L12H3")
    assert V[a12f] == tuple(V[a12]) + ("--attention-inpass",)
    # the L12H3 segment is the one SL-ATT1 put on the frozen version, and the FWDATT one the one of -a12h3-fwd-tro
    frozen = pf.FROZEN_20261112
    assert tuple(V[a12])[len(V[X]):] == tuple(V[STEM + "-a12h3-tro"])[len(V[frozen]):]
    assert sorted(V[a12f]) == sorted(tuple(V[STEM + "-a12h3-fwd-tro"]) + ("--graspable-hand-sized-mm", "156",
                                                                         "--graspable-hand-sized-attention-top"))
    for base, name in zip((X, a12, XF, a12f), pas):
        assert V[name] == tuple(V[base]) + ("--passthrough-arm-strict",)
    ap = ss.build_parser()
    for name in pf.SOTA_FINAL_NEXT_VERSIONS:
        assert pf.is_gtto_version(name) and not pf.version_requires_oracle_segmentation(name)
        a = ap.parse_args(pf.version_server_argv(name) + list(pf.version_server_flags(name)))
        ss.reject_bad_flag_combinations(ap, a)
        srv.check_unported_flags(a, None)
        srv.check_inpass_flags(a, None)                                   # FWDATT: no refusal (no-op without the flag)
        cfg = ss.trajopt_config_from_args(a)
        cfg.validate()
        assert cfg.sqp.passthrough_arm_strict is (name in pas)
        assert a.attention_inpass is ("-fwd-" in name)
        assert (a.attention_source == "action:last:L12H3") is ("-a12h3-" in name)
        assert (getattr(a, "response_model", None) == "libero_safety_fork_native") is name.endswith("-lsnative")
    # the earlier versions keep their argv: no other version carries the new flag
    assert [n for n, f in V.items() if "--passthrough-arm-strict" in f and n not in pf.SOTA_ARMSTRICT_W_VERSIONS] == list(pas)   # (-pas6: test_pas6_versions)


def test_tab2_versions_pass_the_libero_safety_launcher_rules(monkeypatch):
    import pathlib

    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).resolve().parents[2] / "benchmark/runners"))
    import launch_libero_safety_paired as lls

    for name in (X + "-lsnative", XF + "-lsnative"):
        assert lls.native_fork_version(name)
        ns = lls.build_parser().parse_args(["--task-suite-name", "affordance", "--level", "2", "--task-ids", "0",
                                            "--episode-ids", "19", "--run-id", "x", "--data-split", "test",
                                            "--policy-checkpoint", "pi05_libero_safety", "--conversion", "C",
                                            "--ag3s-version", name])
        lls.check(ns, ["vla_v11ab"])
        assert ns.action_adapter == "fork_native_unit"
    assert not lls.native_fork_version(X) and not lls.native_fork_version(XF)


# --- ARMSTRICT-W (2026-11-15 15:4x): --passthrough-arm-strict-no-wrist leaves panda_link7 out -------------------------
def test_no_wrist_mask_and_config_rules():
    from AG3S.trajopt import sqp as sq

    names = ["panda_link1", "panda_link6", "panda_link7", "panda_hand", "link_right_arm_6"]
    assert sq.arm_link_row_mask(names, 5).tolist() == [True, True, True, False, True]
    assert sq.arm_link_row_mask(names, 5, no_wrist=True).tolist() == [True, True, False, False, True]
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig.from_dict({"sqp": {"passthrough_arm_strict_no_wrist": True}})
    assert "passthrough_arm_strict_no_wrist" not in TrajOptConfig().to_dict()["sqp"]


def test_real_rows_link7_refused_by_strict_and_passed_without_the_wrist(panda):
    w = panda
    scene = _field_scene(w, "panda_link7", (0.0, 1.0, 0.0))
    on = _panda_opt(w, True).solve(w.ref, Q0, scene, template=w.chunk)
    assert on.metrics["passthrough_clear"]["reason"] == "arm" and on.metrics["passthrough_clear"]["arm_rows"]["link"] == "panda_link7"
    nw = _panda_opt(w, True, passthrough_arm_strict_no_wrist=True).solve(w.ref, Q0, scene, template=w.chunk)
    rec = nw.metrics["passthrough_clear"]
    assert rec["applied"] is True and rec["reason"] == "clear" and rec["arm_rows"]["no_wrist"] is True
    assert rec["arm_rows"]["n_arm_spheres"] < 37 and rec["arm_rows"]["n_violating"] == 0
    # link5 is still an arm row without the wrist
    scene5 = _field_scene(w, "panda_link5", (0.0, 1.0, 0.0))
    r5 = _panda_opt(w, True, passthrough_arm_strict_no_wrist=True).solve(w.ref, Q0, scene5, template=w.chunk)
    assert r5.metrics["passthrough_clear"]["reason"] == "arm"


def test_pas6_versions():
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss
    v6 = (STEM + "-gh156a-a12h3-fwd-pas6-tro", STEM + "-gh156a-fwd-pas6-tro")
    assert pf.SOTA_ARMSTRICT_W_VERSIONS == v6 and set(v6) <= set(pf.GT15_VERSIONS)
    assert pf.AG3S_VERSIONS[v6[0]] == tuple(pf.AG3S_VERSIONS[STEM + "-gh156a-a12h3-fwd-tro"]) + pf.ARMSTRICT_W_FLAGS
    assert pf.AG3S_VERSIONS[v6[1]] == tuple(pf.AG3S_VERSIONS[XF]) + pf.ARMSTRICT_W_FLAGS
    ap = ss.build_parser()
    for v in v6:
        a = ap.parse_args(pf.version_server_argv(v) + list(pf.version_server_flags(v)))
        ss.reject_bad_flag_combinations(ap, a); srv.check_unported_flags(a, None)
        c = ss.trajopt_config_from_args(a); c.validate()
        assert c.sqp.passthrough_arm_strict and c.sqp.passthrough_arm_strict_no_wrist

