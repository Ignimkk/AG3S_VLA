"""SL10 §2.2 -- the Panda controller's action limit as TrajOpt QP rows (`serve_safe --to-action-box`, task SL10 §2.2).

What this file pins:
1. Off (the default) is bit-identical to HEAD 62c0076f: golden SHA-1 of refiner results on synthetic Panda chunks measured
   on a `git archive 62c0076f` copy (same script), no `action_box` key, no new record key, the SafePolicy arguments and the
   QP budget of the earlier versions unchanged.
2. On, on a synthetic chunk: without the rows the conversion C output saturates on every executed row, with them it does
   not (exact prediction and the real conversion agree), `to.action_box` has its four keys, a chunk nothing touches passes
   its bits, and what the rows cannot avoid stays a collision violation (the existing rules decide).
3. The linearisation (first order, exact at the iterate, fixed sparsity pattern) and the QP layout.
4. The RB-Y1 server refuses the flags; the argument checks.
5. The two versions are their bases plus this one flag; they build; the name is read back.
"""
from __future__ import annotations

import hashlib
import types

import numpy as np
import pytest

from tests.benchmark.test_sl6a_panda_final import _Stub, gpu_ok  # noqa: F401 (fixture)

pytest.importorskip("casadi")
pytest.importorskip("osqp")

PC = "AG3S-final-panda-v3-e0to-pc"
PCS = "AG3S-final-panda-v3-e0to-pc-surface"
VERSIONS = {f"{PC}-abox": PC, f"{PCS}-abox": PCS}
H = 10
K = 5
Q0 = np.array([0.0, 0.17, 0.0, -2.45, 0.0, 2.6, 0.79])
#: SHA-1[:16] of the refined float32 chunk (`TrajOptChunkRefiner.refine` of `[q7, g]`), scenarios below, measured
#: 2026-11-04 on `git archive 62c0076f` with .venv-ag3s (AG3S_PANDA_CONVERSION=C, the `-pc` version's TrajOpt config).
HEAD_SHA = {"graze": "f7af4b8cf14c926d", "deep": "1ca07ed0744e6e01", "clear": "aca7094e993b474d"}
HEAD_STATUS = {"graze": "feasible", "deep": "feasible", "clear": "unconstrained"}
HEAD_N_METRICS = {"graze": 33, "deep": 33, "clear": 25}
SCENARIOS = {"graze": (0.5, 0.004), "deep": (0.3, 0.03), "clear": (0.0, None)}   # (policy position action, penetration)


def _sha(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()[:16]


def _args(version, extra=()):
    from AG3S.adapters.panda import profile as pf

    argv = pf.version_server_argv(version)
    argv[argv.index("--config") + 1] = "pi05_libero"
    return pf.parse_v11ab_args(argv=argv + ["--robot", "panda"] + pf.version_server_flags(version) + list(extra))


@pytest.fixture(scope="module")
def world():
    """Panda constraint model, layout, kinematics, collision linearizer (once per module: CasADi graphs are slow)."""
    from AG3S.adapters.panda import action as pa
    from AG3S.adapters.panda import model as pm
    from AG3S.trajopt.linearize import CollisionLinearizer

    robot = pm.build_panda_constraint_model(pm.panda_constraint_link_filter("gripper"))
    layout = pa.panda_chunk_layout(robot.joint_names)
    return types.SimpleNamespace(robot=robot, layout=layout, kin=pa.PandaKinematics(robot),
                                 lin=CollisionLinearizer(robot, layout, H))


def _config(version):
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import serve_safe as ss

    cfg = ps._replace_horizon_limits(ss.trajopt_config_from_args(_args(version)))
    return ps.action_box_qp_config(cfg) if getattr(_args(version), "to_action_box", False) else cfg


def _scenario(world, name):
    from AG3S.adapters.panda import action as pa
    from tests.trajopt.fixtures import empty_scene, near_miss_scene

    a_val, pen = SCENARIOS[name]
    A = np.zeros((H, 7))
    A[:, :3] = a_val
    conv = pa.delta_chunk_to_joint_chunk(world.kin, A.astype(np.float32), Q0, np.eye(4))
    ref = conv.joint_chunk[:, :7].T.copy()
    scene = (empty_scene(world.lin) if pen is None
             else near_miss_scene(world.lin, ref, Q0, penetration=pen, radius=0.05, d_safe=0.01))
    return conv, scene


def _refine(world, version, name, *, box=False, monkeypatch=None):
    from AG3S.adapters.panda import action as pa
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    conv, scene = _scenario(world, name)
    refiner = TrajOptChunkRefiner(world.robot, world.layout, lambda ctx: (scene, Q0, True), _config(version))
    context = ({"action_box": pa.action_box_from_conversion(conv, margin=0.05, n_rows=K, mode="C")} if box else None)
    out = refiner.refine(conv.joint_chunk.astype(np.float32), context)
    return conv, out, refiner.last_result


# --- 1. off is HEAD ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_off_is_bit_identical_to_head(world, name, monkeypatch):
    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    _conv, out, res = _refine(world, PC, name)
    assert _sha(out) == HEAD_SHA[name]
    assert res.status.value == HEAD_STATUS[name]
    assert "action_box" not in res.metrics and len(res.metrics) == HEAD_N_METRICS[name]


def test_off_changes_nothing_else(world, gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.safe_policy import SafePolicy

    args = _args(PC)
    assert args.to_action_box is False and args.to_action_box_margin is None
    cfg = _config(PC)
    assert cfg.qp.max_iter == ss.trajopt_config_from_args(args).qp.max_iter != ps.ACTION_BOX_QP_MAX_ITER
    assert ps.action_box_kwargs(args, cfg, types.SimpleNamespace(last=None), "C") == {}      # SafePolicy arguments as before
    assert SafePolicy.action_box_fn is None
    assert "to_action_box" not in pf.version_flag_values(args)                              # recorded flag keys as before


# --- 2. on -----------------------------------------------------------------------------------------------------------
def _sat_exec_rows(world, conv, out):
    from AG3S.adapters.panda import action as pa

    _actions, rec = pa.joint_chunk_to_delta_chunk(world.kin, conv, out[:, :7 + 1].astype(np.float32),
                                                  conv.joint_chunk.astype(np.float32), mode="C")
    return [k for k in rec["saturated_rows"] if k < K], _actions


def test_on_removes_the_saturation_conversion_c_would_cause(world, monkeypatch):
    from AG3S.adapters.panda import action as pa

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    conv, off_out, off_res = _refine(world, PC, "graze")
    sat_off, act_off = _sat_exec_rows(world, conv, off_out)
    assert sat_off == list(range(K))                       # without the rows every executed row is clipped
    assert float(np.abs(act_off[:K, :6]).max()) == pytest.approx(1.0)
    conv, on_out, on_res = _refine(world, f"{PC}-abox", "graze", box=True)
    sat_on, act_on = _sat_exec_rows(world, conv, on_out)
    assert sat_on == []                                    # with them none is, on any of the 6 components
    rec = on_res.metrics["action_box"]
    assert {"on", "n_rows", "active_rows", "max_pred_abs_action"} <= set(rec)
    assert rec["on"] is True and rec["n_rows"] == 6 * K and rec["active_rows"] > 0
    assert rec["max_pred_abs_action"] <= 1.0 and rec["max_excess"] < 0.05
    # the record is the exact prediction of what conversion C then sent
    assert rec["max_pred_abs_action"] == pytest.approx(float(np.abs(act_on[:K, :6]).max()), abs=2e-3)
    # the rows changed the plan (they are what they ask for), and the chunk's grippers and shape are untouched
    assert not np.array_equal(on_out, off_out) and on_out.shape == off_out.shape
    assert np.array_equal(on_out[:, 7], off_out[:, 7])


def test_on_leaves_a_clear_chunk_alone(world, monkeypatch):
    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    _c, off_out, _ = _refine(world, PC, "clear")
    conv, on_out, res = _refine(world, f"{PC}-abox", "clear", box=True)
    assert np.array_equal(on_out, off_out)                 # nothing to avoid: the policy chunk, bit for bit
    rec = res.metrics["action_box"]
    assert rec["active_rows"] == 0 and rec["max_pred_abs_action"] == 0.0
    _sat, actions = _sat_exec_rows(world, conv, on_out)
    assert np.array_equal(actions[:, :6], np.asarray(conv.delta)[:, :6])


def test_what_the_rows_cannot_avoid_stays_a_violation(world, monkeypatch):
    """A 30 mm penetration needs more than the controller can deliver in 5 rows: the plan stays inside the box and the
    remaining collision is the optimizer's own `violated` (the existing HOLD / violation rules read it)."""
    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    conv, out, res = _refine(world, f"{PC}-abox", "deep", box=True)
    assert res.status.value == "violated" and res.max_violation > 0.005
    assert _sat_exec_rows(world, conv, out)[0] == []
    assert res.metrics["action_box"]["max_pred_abs_action"] <= 1.0
    _c, _o, off = _refine(world, PC, "deep")
    assert off.status.value == "feasible"                  # (the unconstrained plan "avoids" it with actions the robot clips)


def test_the_box_is_off_for_another_q_now(world, monkeypatch):
    """A box built from another state than the solver's is not applied (the rows would be about another chunk)."""
    import dataclasses

    from AG3S.adapters.panda import action as pa
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    conv, scene = _scenario(world, "graze")
    box = dataclasses.replace(pa.action_box_from_conversion(conv, margin=0.05, n_rows=K, mode="C"), q_now=Q0 + 0.01)
    refiner = TrajOptChunkRefiner(world.robot, world.layout, lambda ctx: (scene, Q0, True), _config(f"{PC}-abox"))
    out = refiner.refine(conv.joint_chunk.astype(np.float32), {"action_box": box})
    rec = refiner.last_result.metrics["action_box"]
    assert rec["on"] is False and "q_now" in rec["reason"] and rec["n_rows"] == 0
    assert _sha(out) == HEAD_SHA["graze"]                  # = the solve without rows


def test_to_block_carries_the_record(world):
    from AG3S.trajopt.safe_policy import SafePolicy

    safe = object.__new__(SafePolicy)
    rec = {"on": True, "n_rows": 30, "active_rows": 3, "max_pred_abs_action": 0.9}
    res = types.SimpleNamespace(status=types.SimpleNamespace(value="feasible"), iterations=1, solve_time_ms=1.0,
                                reference_deviation=0.0, metrics={"action_box": rec})
    safe.refiner = types.SimpleNamespace(last_result=res)
    safe.held_hand_exempt = safe.closing_hand_exempt = False
    safe.grasp_zone_allowance = None
    safe.latch_phase = False
    assert safe._to_block()["action_box"] == rec
    res.metrics = {}
    assert "action_box" not in safe._to_block()


# --- 3. the linearisation and the QP layout --------------------------------------------------------------------------
def test_linearisation_is_exact_at_the_iterate_and_first_order(world, monkeypatch):
    from AG3S.adapters.panda import action as pa
    from AG3S.trajopt import action_box as AB
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import empty_scene
    from AG3S.trajopt.limits import build_limits

    cfg = _config(f"{PC}-abox")
    conv, _scene = _scenario(world, "graze")
    ref = conv.joint_chunk[:, :7].T.copy()
    box = pa.action_box_from_conversion(conv, margin=0.05, n_rows=K, mode="C")
    opt = TrajectoryOptimizer(world.robot, world.layout, build_limits(world.robot, world.layout, dt=cfg.horizon.dt,
                                                                      config=cfg.limits), cfg)
    st = opt._abox_setup(box, ref, Q0)
    rng = np.random.default_rng(0)
    errs = []
    for scale in (1e-4, 1e-3):
        X = ref + rng.normal(0, scale, ref.shape)
        Q = X + rng.normal(0, scale, ref.shape)
        pose = st["fk"].pose(X, Q0)
        G, e = AB.linear_rows(box, X, pose, st["now"], st["ref_steps"], K)
        pq, _, Rq, _ = st["fk"].pose(Q, Q0)
        exact_X = AB.exact_correction(box, pose[0], pose[2], st["now"][0], st["now"][1], st["ref_steps"], K).reshape(-1)
        exact_Q = AB.exact_correction(box, pq, Rq, st["now"][0], st["now"][1], st["ref_steps"], K).reshape(-1)
        assert np.abs(G @ X.T.reshape(-1) + e - exact_X).max() < 1e-9             # exact at the iterate
        errs.append(float(np.abs(G @ Q.T.reshape(-1) + e - exact_Q).max()))
    assert errs[1] < 1e-2 and errs[1] / errs[0] < 200                              # second order: x10 step -> ~x100 error
    # the sparsity pattern is the same at any iterate (the QP backend updates A.data in place)
    X2 = ref + rng.normal(0, 0.05, ref.shape)
    G2, _ = AB.linear_rows(box, X2, st["fk"].pose(X2, Q0), st["now"], st["ref_steps"], K)
    assert np.array_equal(G.indptr, G2.indptr) and np.array_equal(G.indices, G2.indices)


def test_qp_layout_with_the_rows(world, monkeypatch):
    from AG3S.trajopt import action_box as AB
    from AG3S.trajopt.problem import build_problem
    from AG3S.trajopt.limits import build_limits

    cfg = _config(PC)
    limits = build_limits(world.robot, world.layout, dt=cfg.horizon.dt, config=cfg.limits)
    ref = np.tile(Q0[:, None], (1, H))
    pb = build_problem(ref, ref, limits, cfg, q_now=Q0, n_slack=4)
    assert pb.n_action_slack == 0 and pb.n_var == pb.n_q + pb.n_slack + pb.n_kinematic_slack
    m = 6 * K
    G = AB.sp.csr_matrix(np.zeros((m, pb.n_q)))
    lo, hi = -np.ones(m), np.ones(m)
    pb2 = AB.add_action_box(pb, G, np.zeros(m), lo, hi, 123.0)
    assert pb2.n_action_slack == m and pb2.n_var == pb.n_var + m
    assert pb2.A.shape == (pb.A.shape[0] + 3 * m, pb.n_var + m) and pb2.P.shape == (pb.n_var + m,) * 2
    assert np.all(pb2.q[-m:] == 123.0) and np.array_equal(pb2.q[:-m], pb.q)
    z = pb2.flatten(ref)
    assert z.shape == (pb2.n_var,) and pb2.split(z)[0].shape == ref.shape
    for name in ("action_box", "action_box_lower", "action_box_slack"):
        a, b = pb2.row_blocks[name]
        assert b - a == m


def test_the_bound_never_asks_a_change_of_the_reference(world):
    """`hi = max(1 - margin, a)`: a policy component already beyond the margin is allowed to stay (Dy = 0 is inside)."""
    from AG3S.adapters.panda import action as pa

    A = np.zeros((H, 7))
    A[:, 0], A[:, 3], A[:, 1] = 0.99, -1.0, -0.2
    conv = pa.delta_chunk_to_joint_chunk(world.kin, A.astype(np.float32), Q0, np.eye(4))
    box = pa.action_box_from_conversion(conv, margin=0.05, n_rows=K, mode="C")
    lo, hi = box.allowed(K)
    assert np.all(hi[:, 0] == pytest.approx(0.99)) and np.all(lo[:, 3] == -1.0)
    assert np.all(hi[:, 1] == pytest.approx(0.95)) and np.all(lo[:, 1] == pytest.approx(-0.95))
    assert np.all(lo <= box.actions[:K, :6]) and np.all(box.actions[:K, :6] <= hi)
    # constants are the adapter's own
    assert tuple(box.alpha[:4]) == (pa.RESPONSE["pos"][0],) * 3 + (pa.RESPONSE["rot"][0],)
    assert tuple(box.beta[2:4]) == (pa.RESPONSE["pos"][1], pa.RESPONSE["rot"][1])
    assert tuple(box.scale[[0, 3]]) == (pa.OSC_POS_SCALE, pa.OSC_ORI_SCALE)
    b = pa.action_box_from_conversion(conv, margin=0.05, n_rows=K, mode="B")
    assert np.all(b.alpha == 0.0) and np.all(b.beta == 1.0)


# --- 4. RB-Y1 refuses, argument checks --------------------------------------------------------------------------------
RBY1_BASE = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def _parse(extra):
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    return ap, ap.parse_args(RBY1_BASE + list(extra))


def test_the_rby1_server_refuses_the_flags(monkeypatch):
    import sys

    from AG3S.trajopt import serve_safe as ss

    for extra in (["--to-action-box"], ["--to-action-box", "--to-action-box-margin", "0.1"]):
        ap, args = _parse(extra)
        assert args.robot == "rby1"
        with pytest.raises(SystemExit):
            ss.reject_panda_only_flags(ap, args)
        ap, args = _parse(extra + ["--robot", "panda"])
        ss.reject_bad_flag_combinations(ap, args)
        ss.reject_panda_only_flags(ap, args)                                      # the Panda server takes it
    ap, args = _parse([])
    ss.reject_panda_only_flags(ap, args)                                          # off: nothing refused
    # and it is `main` that refuses, before anything is loaded
    monkeypatch.setattr(sys, "argv", ["serve_safe"] + RBY1_BASE + ["--to-action-box"])
    with pytest.raises(SystemExit):
        ss.main()


def test_argument_checks():
    from AG3S.trajopt import serve_safe as ss

    def refused(extra):
        ap, args = _parse(extra)
        with pytest.raises(SystemExit):
            ss.reject_bad_flag_combinations(ap, args)

    refused(["--to-action-box-margin", "0.1"])                                      # a margin without the box
    refused(["--to-action-box", "--to-action-box-margin", "1.0"])
    refused(["--to-action-box", "--to-action-box-margin", "-0.1"])
    refused(["--to-action-box", "--no-safe"])
    refused(["--to-action-box", "--to-multi-seed", "4"])
    ap, args = _parse(["--to-action-box", "--to-action-box-margin", "0.0"])
    ss.reject_bad_flag_combinations(ap, args)
    assert args.to_action_box is True and args.to_action_box_margin == 0.0


# --- 5. the two versions ----------------------------------------------------------------------------------------------
def test_versions_are_their_bases_plus_the_flag(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    assert set(VERSIONS) == set(pf.SL10_VERSIONS)
    for v, base in VERSIONS.items():
        assert tuple(pf.AG3S_VERSIONS[v]) == tuple(pf.AG3S_VERSIONS[base]) + ("--to-action-box",)
        assert pf.is_final_version(v) and v in pf.PC_LATCH_VERSIONS
        assert pf.version_server_argv(v) == pf.version_server_argv(base)
        assert pf.version_server_flags(v)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]
        assert pf.version_server_flags(v)[:-2] == list(pf.AG3S_VERSIONS[base]) + ["--to-action-box"]
    # the bases did not move
    assert PC in pf.FINAL_CONFIG_VERSIONS and "--to-action-box" not in pf.AG3S_VERSIONS[PC]
    assert pf.AG3S_VERSIONS[PCS] == pf.PC_SURFACE_FLAGS


def test_args_and_names_read_back(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    for v, base in VERSIONS.items():
        a, b = _args(v), _args(base)
        assert a.to_action_box is True and a.to_action_box_margin is None and b.to_action_box is False
        assert pf.match_ag3s_version(a) == v and pf.match_ag3s_version(b) == base
        va, vb = pf.version_flag_values(a), pf.version_flag_values(b)
        assert va["to_action_box"] is True and "to_action_box" not in vb          # earlier records keep their keys
        assert set(vb) <= set(va) and all(va[k] == vb[k] for k in vb)               # (the SL9 pattern: dests join)
    # only these two flag values differ between a version and its base
    a, b = vars(_args(f"{PC}-abox")), vars(_args(PC))
    assert {k for k in a if a[k] != b[k]} == {"to_action_box"}


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_versions_build_and_hand_the_box_to_safe_policy(version, gpu_ok, monkeypatch):
    from AG3S.adapters.panda import action as pa
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as ps

    if version.startswith(PCS) and not __import__("pathlib").Path(pf.PANDA_HSR6_PROBE).exists():
        pytest.skip("Panda HSR6 probe not in this storage root")
    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    args = _args(version)
    ps.check_unported_flags(args, None)
    served, _info = ps.build_safe_server(_Stub(), args, gpu_warm_up=False)
    safe = served._policy
    assert safe.action_box_fn is not None and safe.to_config.qp.max_iter == ps.ACTION_BOX_QP_MAX_ITER
    assert safe.refiner.config.qp.max_iter == ps.ACTION_BOX_QP_MAX_ITER
    assert safe.action_box_fn() is None                                       # no chunk converted yet
    conv = pa.delta_chunk_to_joint_chunk(served.chunk_policy.kin, np.zeros((H, 7), np.float32), Q0, np.eye(4))
    served.chunk_policy.last = conv
    box = safe.action_box_fn()
    assert box.margin == 0.05 and box.n_rows == ps.PANDA_HORIZON["execution_length"] and box.mode == "C"
    assert box.q_now.shape == (7,) and np.array_equal(box.q_now, Q0)
    # the base version's SafePolicy has none, and the old QP budget
    base_served, _ = ps.build_safe_server(_Stub(), _args(VERSIONS[version]), gpu_warm_up=False)
    assert base_served._policy.action_box_fn is None
    assert base_served._policy.to_config.qp.max_iter != ps.ACTION_BOX_QP_MAX_ITER
