"""SL11 (pass part) -- the three pc-based versions that fix the no-obstacle failures at their cause
(task `AG3S/ag3s/docs/handoff/SL11.task.md` §1). CPU.

What this file pins:

(a) Registry. `-pc-mi10` = pc + `--minimal-intervention-mm 10`; `-pc-mi10-sdev` = mi10 + `--smooth-on-deviation`;
    `-pc-observe` = pc + `--observe-only`. Every other version is unchanged (sha over the registry minus the new keys),
    the parsed flags differ only in the new dests, the record keys of the old versions gain nothing, the name is read back,
    the three launchers accept the names.
(b) `--smooth-on-deviation` (`cost.smooth_on_deviation`): off is bit-identical (golden hashes measured on main 66b2c14f with
    the unchanged code; the QP with the explicit False equals the default QP; the config dump has no new key). On: value and
    gradient are zero at `Q == Q_ref`, the QP encodes `w ||D2 (Q - Q_ref)||^2` exactly (objective agreement), a free-space QP
    returns `Q_ref` while the off QP bends a curved chunk; on the Panda, a free-space curved chunk comes back from the SQP
    bit for bit (the pc version changes it).
(c) `--minimal-intervention-mm 10` end to end on the Panda: the gate returns the reference (bit for bit through the refiner's
    float32 chunk), and the reverse conversion C then returns the policy action bit for bit on every row, where the pc
    version's TO (smoothness bending in free space) makes it correct and saturate rows.
(d) `--observe-only`: the response is the policy's own chunk (bit for bit) with `safe` True even when the verdict is a HOLD,
    the would-be actions and verdict are recorded, off the response is as before; the client row records it; the RB-Y1 server
    refuses the flag.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import types

import numpy as np
import pytest

from tests.benchmark.test_sl6a_panda_final import ROOT, _Stub, gpu_ok  # noqa: F401 (fixture)

pytest.importorskip("casadi")
pytest.importorskip("osqp")

PC = "AG3S-final-panda-v3-e0to-pc"
MI = f"{PC}-mi10"
SDEV = f"{PC}-mi10-sdev"
OBS = f"{PC}-observe"
NEW = (MI, SDEV, OBS)

#: sha256 of ``json.dumps({name: list(flags)}, sort_keys=True)`` over AG3S_VERSIONS before SL11 (51 versions), measured on
#: main 66b2c14f with the registry code of that commit.
OLD_REGISTRY_SHA256 = "c5f2473207811c4d71292450cf611451a02f94b2f1b8b3875cbfb9378d73479e"
OLD_REGISTRY_SIZE = 51

#: (sha1[:16] of the refined trajectory, of the refined chunk, status, iterations, number of metrics keys) of
#: ``TrajectoryOptimizer.solve`` on ``make_scenario(horizon=10, penetration=p)`` with its default config, measured on main
#: 66b2c14f (unchanged code).
HEAD_SOLVE = {0.03: ("43d91c6ee077f179", "4391ad9c203d7e41", "feasible", 6, 26),
              -0.05: ("d02a4dd326a967ac", "e3b1b756726a444b", "feasible", 6, 26)}


def _sha(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()[:16]


def _probe_or_skip():
    from AG3S.adapters.panda import profile as pf

    if not pathlib.Path(pf.PANDA_HSR6_PROBE).exists():
        pytest.skip("Panda HSR6 probe not in this storage root")


def _args(v, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    argv = pf.version_server_argv(v)
    argv[argv.index("--config") + 1] = "pi05_libero"
    return pf.parse_v11ab_args(argv=argv + ["--robot", "panda"] + pf.version_server_flags(v))


def _diff(a, b):
    return {k for k in vars(a) if vars(a)[k] != vars(b)[k]}


# ------------------------------------------------------------------------------------------------ (a) registry
def test_each_new_version_is_the_base_plus_exactly_its_flags():
    from AG3S.adapters.panda import profile as pf

    assert pf.SL11_VERSIONS == NEW
    reg = pf.AG3S_VERSIONS
    assert list(reg[MI]) == list(reg[PC]) + ["--minimal-intervention-mm", "10"]
    assert list(reg[SDEV]) == list(reg[MI]) + ["--smooth-on-deviation"]
    assert list(reg[OBS]) == list(reg[PC]) + ["--observe-only"]
    for v in NEW:
        assert pf.is_final_version(v) and v in pf.PC_LATCH_VERSIONS
        assert pf.version_server_argv(v) == pf.version_server_argv(PC)
        assert pf.version_server_flags(v)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]     # takes the HSR6 probe like pc
    # the new flags belong to no other version
    for flag in ("--smooth-on-deviation", "--observe-only"):
        assert [n for n, f in reg.items() if flag in f and not n.endswith("-lsfork")
                and n not in getattr(pf, "GT12_VERSIONS", ()) + getattr(pf, "GT14_VERSIONS", ()) + getattr(pf, "GT15_VERSIONS", ())] == ([SDEV] if flag == "--smooth-on-deviation" else [OBS])   # SL-GTTO12 observes F8
    assert "--minimal-intervention-mm" not in reg[PC]


def test_every_other_version_is_unchanged():
    from AG3S.adapters.panda import profile as pf

    old = {k: list(v) for k, v in pf.AG3S_VERSIONS.items()
           if k not in NEW and not k.endswith("-lsfork") and k not in pf.SL11_GAP_VERSIONS
           and k not in pf.SL12_VERSIONS and k not in pf.SL13_VERSIONS and k not in pf.SL14_VERSIONS and k not in pf.SL15_NTM_VERSIONS and k not in pf.SL15_SFHAND_VERSIONS and k not in pf.SL16_HSC_VERSIONS and k not in pf.SL16_HVC_VERSIONS and not pf.is_gtto_version(k)}   # SL11 fork merge adds -lsfork; §5 the -gap pair; SL12 the -oraclegt pair   # 2026-11-06 merge: SL-GTTO versions came later
    assert len(old) == OLD_REGISTRY_SIZE
    assert hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest() == OLD_REGISTRY_SHA256
    assert list(pf.PC_LATCH_VERSIONS[-3 - len(pf.SL11_GAP_VERSIONS):-len(pf.SL11_GAP_VERSIONS)]) == list(NEW)   # the others keep their place


def test_parsed_flags_differ_only_in_the_new_dests(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import serve_safe as ss

    _probe_or_skip()
    base = _args(PC, monkeypatch)
    mi, sd, ob = (_args(v, monkeypatch) for v in NEW)
    assert _diff(mi, base) == {"minimal_intervention_mm"} and mi.minimal_intervention_mm == 10.0
    assert _diff(sd, base) == {"minimal_intervention_mm", "smooth_on_deviation"} and sd.smooth_on_deviation is True
    assert _diff(sd, mi) == {"smooth_on_deviation"}
    assert _diff(ob, base) == {"observe_only"} and ob.observe_only is True
    assert base.minimal_intervention_mm is None and base.smooth_on_deviation is False and base.observe_only is False
    for a in (mi, sd, ob):
        ps.check_unported_flags(a, None)                                                     # the Panda server accepts them
        ss.reject_hsr6_flags(ss.build_parser(), a)
        ss.reject_panda_only_flags(ss.build_parser(), a)
    # the name is read back; record keys: the two new dests join only when set, minimal_intervention_mm is a family dest
    for v, a in zip(NEW, (mi, sd, ob)):
        assert pf.match_ag3s_version(a) == v
    assert pf.match_ag3s_version(base) == PC
    vb, vm, vs, vo = (pf.version_flag_values(a) for a in (base, mi, sd, ob))
    assert vm["minimal_intervention_mm"] == 10.0 and vb["minimal_intervention_mm"] is None
    assert set(vm) == set(vb)                                                                # mi10 adds no key
    # the SL11 group joins as a group (the SL9 / SL10 pattern): setting one of its dests records both
    group = {"smooth_on_deviation", "observe_only"}
    assert vs["smooth_on_deviation"] is True and vs["observe_only"] is False and not group & (set(vb) | set(vm))
    assert vo["observe_only"] is True and vo["smooth_on_deviation"] is False
    # (as for the SL9 / SL10 groups the record also gains the earlier groups' dests, at their parser default)
    from AG3S.trajopt import serve_safe as ss

    defaults = ss.build_parser()
    for va in (vs, vo):
        assert set(vb) <= set(va) and group <= set(va)
        assert all(va[k] == vb[k] for k in vb if k != "minimal_intervention_mm")
        assert all(va[k] == pf._jsonable(defaults.get_default(k)) for k in set(va) - set(vb) - group)


def test_old_versions_record_keys_do_not_gain_the_dests(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    _probe_or_skip()
    for v in (PC, f"{PC}-surface", f"{PC}-plift", f"{PC}-hfq", f"{PC}-uncwaive", "AG3S-final-panda-v3-e0to", "V11AB-pb7"):
        vals = pf.version_flag_values(_args(v, monkeypatch))
        assert "smooth_on_deviation" not in vals and "observe_only" not in vals, v


def test_launchers_accept_the_versions(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from benchmark.runners import launch_libero_paired as lp
    from benchmark.runners import launch_libero_safety_paired as lsp

    flag = {MI: "--minimal-intervention-mm", SDEV: "--smooth-on-deviation", OBS: "--observe-only"}
    for v in NEW:
        got = lp.server_argv("vla_v11ab", "/ck", 9, v, hsr6_probe="/p.npz")
        want = lp.server_argv("vla_v11ab", "/ck", 9, PC, hsr6_probe="/p.npz")
        assert got[:len(want) - 2] == want[:-2] and got[-2:] == ["--hsr6-probe", "/p.npz"] and flag[v] in got
        a = lp.build_parser().parse_args(["--task-ids", "1", "--episode-ids", "1", "--run-id", "x", "--conversion", "C",
                                          "--ag3s-version", v])
        assert a.ag3s_version == v and pf.version_server_flags(v, **lp.version_options(a))
        wrapped = lsp.make_server_argv("pi05_libero_safety")("vla_v11ab", "/ck", 9, v)
        assert flag[v] in wrapped and wrapped[wrapped.index("--robot") + 1] == "panda"
    assert pf.version_server_flags(OBS)         # eval_matrix.build_jobs validates the name with exactly this call


# ------------------------------------------------------------------------------------------------ refusals + CLI
BASE = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]


def test_rby1_server_refuses_observe_only_and_the_panda_server_does_not():
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    ss.reject_panda_only_flags(ap, ap.parse_args(BASE))                                     # off: nothing
    rby1 = ap.parse_args(BASE + ["--observe-only"])
    assert rby1.robot == "rby1" and rby1.observe_only is True
    with pytest.raises(SystemExit):
        ss.reject_panda_only_flags(ap, rby1)
    ss.reject_panda_only_flags(ap, ap.parse_args(BASE + ["--observe-only", "--robot", "panda"]))


def test_smooth_on_deviation_is_refused_with_multi_seed_and_without_the_optimizer():
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    ss.reject_bad_flag_combinations(ap, ap.parse_args(BASE + ["--smooth-on-deviation"]))     # alone: accepted
    for extra in (["--smooth-on-deviation", "--to-multi-seed", "3"], ["--smooth-on-deviation", "--no-safe"],
                  ["--observe-only", "--no-safe", "--robot", "panda"]):
        with pytest.raises(SystemExit):
            ss.reject_bad_flag_combinations(ap, ap.parse_args(BASE + extra))


def test_cli_reaches_the_config_and_absent_flag_changes_nothing(caplog):
    import logging

    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError

    off = ss.trajopt_config_from_args(ss.build_parser().parse_args(BASE))
    on = ss.trajopt_config_from_args(ss.build_parser().parse_args(BASE + ["--smooth-on-deviation"]))
    assert ss.cost_overrides(ss.build_parser().parse_args(BASE)) == {}
    assert off.cost.smooth_on_deviation is False and on.cost.smooth_on_deviation is True
    assert on == off.with_overrides({"cost": {"smooth_on_deviation": True}})
    assert "smooth_on_deviation" not in off.to_dict()["cost"] and on.to_dict()["cost"]["smooth_on_deviation"] is True
    assert TrajOptConfig.from_dict(on.to_dict()) == on and TrajOptConfig.from_dict(off.to_dict()) == off
    with pytest.raises(TrajOptConfigError, match="smooth_on_deviation"):
        TrajOptConfig.from_dict({"cost": {"smooth_on_deviation": 1}})
    with caplog.at_level(logging.WARNING):
        ss.announce_ta(off)
        assert "smoothness on the deviation" not in caplog.text
        ss.announce_ta(on)
    assert "smoothness on the deviation (SL11)" in caplog.text


# ------------------------------------------------------------------------------------------------ (b) smoothness on D2(Q - Q_ref)
NQ, H = 4, 8


@pytest.fixture
def limits():
    from AG3S.trajopt.types import JointLimits

    return JointLimits(lower=np.full(NQ, -3.0), upper=np.full(NQ, 3.0), max_step=np.full(NQ, 0.2),
                       max_step_change=np.full(NQ, 0.08), dt=1.0 / 15.0)


def _cfg(sdev=None, **over):
    from AG3S.trajopt.config import TrajOptConfig

    base = {"horizon": {"horizon": H, "execution_length": 4, "plan_horizon": None}}
    if sdev is not None:
        base["cost"] = {"smooth_on_deviation": sdev}
    for k, v in over.items():
        base[k] = {**base.get(k, {}), **v}
    return TrajOptConfig.from_dict(base)


def _curved(limits):
    t = np.linspace(0.0, np.pi, H)
    return 0.01 * np.arange(H)[None, :] + 0.03 * np.stack([np.sin(t), np.cos(2 * t), np.sin(3 * t), np.cos(t)])


def _solve(problem, eps=1e-10):
    import osqp
    import scipy.sparse as sp

    solver = osqp.OSQP()
    solver.setup(P=sp.triu(problem.P, format="csc"), q=problem.q, A=problem.A, l=np.clip(problem.l, -1e20, 1e20),
                 u=np.clip(problem.u, -1e20, 1e20), verbose=False, eps_abs=eps, eps_rel=eps, polish=True, max_iter=40000)
    res = solver.solve()
    assert res.info.status in ("solved", "solved inaccurate"), res.info.status
    return problem.split(res.x)[0], res.x


def test_off_qp_is_the_default_qp_bit_for_bit(limits):
    from AG3S.trajopt.problem import build_problem

    ref = _curved(limits)
    a = build_problem(ref, ref + 0.01, limits, _cfg(), q_now=ref[:, 0], n_slack=3)
    b = build_problem(ref, ref + 0.01, limits, _cfg(sdev=False), q_now=ref[:, 0], n_slack=3)
    for name in ("P", "A"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.shape == y.shape and (x != y).nnz == 0 and x.indices.tobytes() == y.indices.tobytes(), name
    for name in ("q", "l", "u"):
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), name


def test_value_and_gradient_are_zero_at_the_reference(limits):
    from AG3S.trajopt.problem import build_problem, objective

    ref = _curved(limits)
    on, off = _cfg(sdev=True), _cfg(sdev=False)
    assert objective(ref, ref, on) == 0.0 and objective(ref, ref, off) > 0.0
    p_on = build_problem(ref, ref, limits, on, q_now=ref[:, 0])
    p_off = build_problem(ref, ref, limits, off, q_now=ref[:, 0])
    z = p_on.flatten(ref)
    grad_on = p_on.P @ z + p_on.q
    grad_off = p_off.P @ z + p_off.q
    n = NQ * H
    assert np.abs(grad_on[:n]).max() < 1e-12                       # no force on the planned path at Q == Q_ref
    assert np.abs(grad_off[:n]).max() > 1e-3                       # the off cost pulls a curved chunk straight


def test_the_qp_encodes_the_deviation_smoothness_exactly(limits):
    from AG3S.trajopt.problem import build_problem, objective

    rng = np.random.default_rng(5)
    ref = _curved(limits)
    Q = ref + rng.normal(0, 0.04, ref.shape)
    for cfg in (_cfg(sdev=True), _cfg(sdev=True, cost={"w_smooth": 3.0})):
        p = build_problem(ref, Q, limits, cfg)
        z, z0 = p.flatten(Q), p.flatten(ref)
        quad = lambda v: 0.5 * v @ (p.P @ v) + p.q @ v   # noqa: E731
        assert quad(z) - quad(z0) == pytest.approx(objective(Q, ref, cfg) - objective(ref, ref, cfg), rel=1e-9)
    # the value is w * sum ||D2 (Q - Q_ref)||^2 and nothing of the path itself
    cfg = _cfg(sdev=True)
    d = Q - ref
    track = cfg.cost.w_track * np.sum(d ** 2)
    assert objective(Q, ref, cfg) == pytest.approx(track + cfg.cost.w_smooth * np.sum(np.diff(d, n=2, axis=1) ** 2), rel=1e-12)


def test_free_space_qp_returns_the_reference_on_and_bends_it_off(limits):
    from AG3S.trajopt.problem import build_problem

    ref = _curved(limits)
    on, _ = _solve(build_problem(ref, ref, limits, _cfg(sdev=True), q_now=ref[:, 0]))
    off, _ = _solve(build_problem(ref, ref, limits, _cfg(sdev=False), q_now=ref[:, 0]))
    assert np.abs(on - ref).max() < 1e-7
    assert np.abs(off - ref).max() > 1e-3


def test_off_solve_is_bit_identical_to_head():
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    import AG3S.trajopt.sqp as sqp_module

    assert pathlib.Path(sqp_module.__file__).resolve().is_relative_to(ROOT)       # the code under test is this checkout's
    for pen, want in HEAD_SOLVE.items():
        sc = make_scenario(horizon=10, penetration=pen)
        r = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config).solve(sc.reference, sc.q_now, sc.scene,
                                                                                template=sc.chunk)
        assert (_sha(r.trajectory), _sha(r.chunk), r.status.value, r.iterations, len(r.metrics)) == want
        explicit = sc.config.with_overrides({"cost": {"smooth_on_deviation": False}})
        r2 = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, explicit).solve(sc.reference, sc.q_now, sc.scene,
                                                                                template=sc.chunk)
        assert r2.trajectory.tobytes() == r.trajectory.tobytes() and r2.chunk.tobytes() == r.chunk.tobytes()


# ------------------------------------------------------------------------------------------------ (b2) + (c) Panda
from tests.benchmark.test_sl10_action_box import H as PH, Q0, _config  # noqa: E402


@pytest.fixture(scope="module")
def world():
    from AG3S.adapters.panda import action as pa
    from AG3S.adapters.panda import model as pm
    from AG3S.trajopt.linearize import CollisionLinearizer

    robot = pm.build_panda_constraint_model(pm.panda_constraint_link_filter("gripper"))
    layout = pa.panda_chunk_layout(robot.joint_names)
    return types.SimpleNamespace(robot=robot, layout=layout, kin=pa.PandaKinematics(robot),
                                 lin=CollisionLinearizer(robot, layout, PH))


def _policy_chunk(amp):
    """A smooth, curved EE-delta chunk (pos + rot, a gripper ramp) as float32, the dtype the policy sends."""
    t = np.linspace(0.0, 2.0 * np.pi, PH)
    A = np.zeros((PH, 7))
    A[:, :3] = amp * np.stack([np.sin(t), np.cos(2 * t), np.sin(3 * t)], 1)
    A[:, 3:6] = 0.3 * amp * np.stack([np.cos(t), np.sin(2 * t), np.cos(3 * t)], 1)
    A[:, 6] = np.linspace(-1.0, 1.0, PH)
    return A.astype(np.float32)


def _refine(world, cfg, amp, penetration=-0.3):
    from AG3S.adapters.panda import action as pa
    from AG3S.trajopt.refiner import TrajOptChunkRefiner
    from tests.trajopt.fixtures import near_miss_scene

    A = _policy_chunk(amp)
    conv = pa.delta_chunk_to_joint_chunk(world.kin, A, Q0, np.eye(4))
    ref = conv.joint_chunk[:, :7].T.copy()
    scene = near_miss_scene(world.lin, ref, Q0, penetration=penetration, radius=0.05, d_safe=0.01)   # far from the chunk
    refiner = TrajOptChunkRefiner(world.robot, world.layout, lambda ctx: (scene, Q0, True), cfg)
    refined = refiner.refine(conv.joint_chunk.astype(np.float32), None)
    qr = conv.joint_chunk.astype(np.float32)
    actions, rec = pa.joint_chunk_to_delta_chunk(world.kin, conv, np.asarray(refined, np.float32), qr, mode="C")
    return A, conv, np.asarray(refined, np.float32), qr, actions, rec, refiner.last_result


@pytest.mark.parametrize("amp", [0.4, 0.8])
def test_mi10_gate_passes_the_policy_action_bit_for_bit(world, monkeypatch, amp):
    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    A, conv, refined, ref32, actions, rec, res = _refine(world, _config(MI), amp)
    mi = res.metrics["minimal_intervention"]
    assert mi["applied"] is True and mi["threshold_m"] == 0.01 and mi["reference_clearance_m"] > 0.01 and res.iterations == 0
    assert refined.tobytes() == ref32.tobytes()                       # the gate returns the reference ...
    assert actions.dtype == A.dtype and actions.tobytes() == A.tobytes()   # ... so the conversion returns the policy's bits
    assert rec["corrected_rows"] == [] and rec["saturated_rows"] == [] and rec["passthrough_rows"] == list(range(PH))
    # the pc version's TO bends the same free-space chunk (smoothness on D2(Q)): a correction on every executed row
    _A, _c, refined_pc, _r, actions_pc, rec_pc, res_pc = _refine(world, _config(PC), amp)
    assert "minimal_intervention" not in res_pc.metrics
    assert refined_pc.tobytes() != ref32.tobytes() and actions_pc.tobytes() != A.tobytes() and rec_pc["corrected_rows"]


def test_mi10_gate_does_not_apply_to_a_close_chunk(world, monkeypatch):
    """The gate keeps the pb2 meaning: a chunk within 10 mm of a row is still optimized."""
    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    _A, _c, refined, ref32, _a, _r, res = _refine(world, _config(MI), 0.4, penetration=0.004)
    assert res.metrics["minimal_intervention"]["applied"] is False and res.iterations >= 1
    assert refined.tobytes() != ref32.tobytes()


@pytest.mark.parametrize("amp", [0.4, 0.8])
def test_sdev_free_space_solve_returns_the_policy_chunk_bit_for_bit(world, monkeypatch, amp):
    """The gate off, a far obstacle, a curved chunk: with the smoothness on D2(Q - Q_ref) the SQP leaves the chunk (and the
    conversion the action) alone; the pc version bends it."""
    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    over = {"cost": {"smooth_on_deviation": True}}
    A, _c, refined, ref32, actions, rec, res = _refine(world, _config(PC).with_overrides(over), amp)
    assert "minimal_intervention" not in res.metrics and res.iterations >= 1            # it did solve
    assert refined.tobytes() == ref32.tobytes() and actions.tobytes() == A.tobytes() and rec["corrected_rows"] == []
    _A, _c, refined_pc, _r, _a, _rec, _res = _refine(world, _config(PC), amp)
    assert refined_pc.tobytes() != ref32.tobytes()


def test_the_sdev_version_gate_and_cost_are_both_in_its_config(world, monkeypatch):
    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    c = _config(SDEV)
    assert c.sqp.minimal_intervention == pytest.approx(0.01) and c.cost.smooth_on_deviation is True
    assert _config(MI).cost.smooth_on_deviation is False and _config(MI).sqp.minimal_intervention == pytest.approx(0.01)
    assert _config(PC).sqp.minimal_intervention is None and _config(OBS).cost.smooth_on_deviation is False


# ------------------------------------------------------------------------------------------------ (d) observe-only
class _SafeStub:
    """What ``PandaSafeServer`` wraps: answers like ``SafePolicy`` -- a refined chunk that differs from the reference, a verdict."""

    def __init__(self, refined, reference, safe):
        self.refined, self.reference, self.safe = refined, reference, safe

    def reset(self):
        pass

    def infer(self, obs, **kwargs):
        return {"actions": np.array(self.refined, np.float32), "actions_reference": np.array(self.reference, np.float32),
                "safe": self.safe, "seq": 3, "trajopt_status": "violated" if not self.safe else "optimal",
                "max_violation_m": 0.02 if not self.safe else 0.0,
                "verdict_reasons": [] if self.safe else [{"kind": "collision", "action": "hold", "detail": "x"}],
                "notes": ["stub"]}


def _server(world, safe, observe, monkeypatch):
    from AG3S.adapters.panda import action as pa
    from AG3S.adapters.panda import server as ps

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    A = _policy_chunk(0.4)
    conv = pa.delta_chunk_to_joint_chunk(world.kin, A, Q0, np.eye(4))
    reference = conv.joint_chunk.astype(np.float64)
    refined = reference.copy()
    refined[:, :7] += 0.02 * np.sin(np.arange(PH))[:, None]                     # a TO deflection on every row
    chunk_policy = ps.ChunkToJointPolicy(_Stub(), world.kin)
    chunk_policy.last = conv
    kw = {"observe_only": True} if observe else {}
    return ps.PandaSafeServer(_SafeStub(refined, reference, safe), chunk_policy, mode="C", **kw), A


def test_observe_only_sends_the_policy_action_and_never_holds(world, monkeypatch):
    for safe in (True, False):                     # an execute verdict and a HOLD verdict
        plain, A = _server(world, safe, False, monkeypatch)
        obs, _ = _server(world, safe, True, monkeypatch)
        r_plain, r_obs = plain.infer({}), obs.infer({})
        # without the flag: the converted (corrected) chunk, and the server's own verdict
        assert r_plain["actions"].tobytes() != A.tobytes() and r_plain["safe"] is safe
        assert "panda/observe" not in r_plain and "panda/would_be_actions" not in r_plain
        # with it: the policy's bits, execute -- whatever the verdict was
        assert r_obs["actions"].dtype == A.dtype and r_obs["actions"].tobytes() == A.tobytes()
        assert r_obs["actions"].tobytes() == np.asarray(r_obs["panda/actions_policy"]).tobytes()
        assert r_obs["safe"] is True
        rec = r_obs["panda/observe"]
        assert rec["observe_only"] is True and rec["would_be_safe"] is safe and rec["would_be_hold"] is (not safe)
        assert rec["would_be_max_abs_action_change"] > 0.0 and rec["would_be_rows_changed"]
        # what the server would have sent is recorded, bit for bit the plain server's actions
        assert np.asarray(r_obs["panda/would_be_actions"]).tobytes() == r_plain["actions"].tobytes()
        # the verdict keys are the server's own (the analysis reads them)
        for k in ("trajopt_status", "max_violation_m", "verdict_reasons", "notes", "panda/conversion", "panda/joint_refined"):
            assert json.dumps(r_obs[k], default=lambda o: np.asarray(o).tolist()) == \
                json.dumps(r_plain[k], default=lambda o: np.asarray(o).tolist()), k
        json.dumps(rec)                                                         # JSON-able as the call record needs


def test_observe_only_server_is_built_only_with_the_flag(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import server as ps

    _probe_or_skip()
    off, info_off = ps.build_safe_server(_Stub(), _args(PC, monkeypatch), gpu_warm_up=False)
    on, info_on = ps.build_safe_server(_Stub(), _args(OBS, monkeypatch), gpu_warm_up=False)
    assert off.observe_only is False and "observe_only" not in info_off
    assert on.observe_only is True and info_on["observe_only"] is True
    mi, _ = ps.build_safe_server(_Stub(), _args(SDEV, monkeypatch), gpu_warm_up=False)
    assert mi.observe_only is False
    assert mi._policy.to_config.cost.smooth_on_deviation is True and mi._policy.to_config.sqp.minimal_intervention == pytest.approx(0.01)  # noqa: SLF001
    assert off._policy.to_config.cost.smooth_on_deviation is False                    # noqa: SLF001


def _libero_python():
    for base in (ROOT, *ROOT.parents):
        cand = base / ".venv-libero" / "bin" / "python"
        if cand.exists():
            return cand
    return None


def test_client_row_records_the_observe_only_response():
    """The py3.8 client: policy action == executed action; the row gets `observe` and the npz `would_be_actions` only when
    the server sends `panda/observe`; a response without it gives the row keys of before."""
    py = _libero_python()
    if py is None:
        pytest.skip("no .venv-libero in this checkout or above it")
    code = r'''
import sys, json, numpy as np
sys.path.insert(0, "benchmark/runners")
import libero_safe_client as c

POL = np.linspace(-0.5, 0.5, 70).reshape(10, 7).astype(np.float32)

class WS:
    def __init__(self, extra):
        self.extra = extra
    def infer(self, req):
        out = {"actions": POL.copy(), "safe": True, "panda/mode": "vla_v11ab", "panda/actions_policy": POL.copy(),
               "panda/joint_refined": np.zeros((10, 8)), "panda/joint_reference": np.zeros((10, 8)),
               "panda/predicted_ee_base": np.zeros((10, 3))}
        out.update(self.extra)
        return out

res = []
for extra in ({}, {"panda/observe": {"observe_only": True, "would_be_safe": False, "would_be_hold": True},
                   "panda/would_be_actions": POL + np.float32(0.1)}):
    cl = c.SafeClient(WS(extra), 7, 0, 0, 5)
    cl.eef = [None]
    cl._request = lambda element: ({"panda/T_world_base": np.eye(4)}, np.zeros(9))
    out = cl.infer({})
    res.append({"keys": sorted(cl.calls[-1]), "observe": cl.calls[-1].get("observe"), "hold": cl.calls[-1]["hold"],
                "same": bool(np.asarray(out["actions"])[:5].tobytes() == POL[:5].tobytes()),
                "change": cl.calls[-1]["max_abs_action_change_executed"], "arrays": sorted(cl.arrays)})
print(json.dumps(res))
'''
    r = subprocess.run([str(py), "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0 and "No module named" in r.stderr:
        pytest.skip(r.stderr.splitlines()[-1])
    assert r.returncode == 0, r.stderr
    off, on = json.loads(r.stdout.strip().splitlines()[-1])
    assert set(on["keys"]) - set(off["keys"]) == {"observe"} and off["observe"] is None
    assert on["observe"]["would_be_hold"] is True and on["hold"] is False and on["same"] and on["change"] == 0.0
    assert set(on["arrays"]) - set(off["arrays"]) == {"would_be_actions"}
