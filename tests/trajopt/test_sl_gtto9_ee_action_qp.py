"""SL-GTTO9 (F7, lead self-ruling G47, 2026-11-05): the TO solved in the policy's OSC action space (`--ee-action-qp`)."""
from __future__ import annotations

import types

import numpy as np
import pytest


@pytest.fixture(scope="module")
def kin():
    from AG3S.adapters.panda import action as pa
    from AG3S.adapters.panda import model as pm

    return pa.PandaKinematics(pm.build_panda_constraint_model(pm.panda_constraint_link_filter("gripper")))


def test_map_matches_the_forward_conversion(kin):
    """M Da predicts the joint chunk the forward conversion gives for a + Da (first order)."""
    from AG3S.adapters.panda import action as pa

    q0 = np.array([0.0, 0.17, 0.0, -2.45, 0.0, 2.6, 0.79])
    rng = np.random.default_rng(3)
    A = np.clip(rng.normal(0, 0.3, (10, 7)), -0.8, 0.8)
    T = np.eye(4)
    T[:3, :3] = pa.rotvec_to_matrix(np.array([0.0, 0.0, 0.3]))
    prev = (np.full(3, 0.002), np.zeros(3))
    conv = pa.delta_chunk_to_joint_chunk(kin, A, q0, T, prev_step_world=prev)
    amap = pa.ee_action_map(kin, conv, 10)
    assert amap["M"].shape == (70, 60) and np.allclose(amap["a_ref"], A[:, :6])
    da = rng.normal(0, 0.02, (10, 6))
    A2 = A.copy()
    A2[:, :6] += da
    conv2 = pa.delta_chunk_to_joint_chunk(kin, A2, q0, T, prev_step_world=prev)
    true = (conv2.joint_chunk[:, :7] - conv.joint_chunk[:, :7]).reshape(-1)
    pred = amap["M"] @ da.reshape(-1)
    assert np.linalg.norm(pred - true) <= 0.05 * np.linalg.norm(true)
    # causal: a change of row j moves no earlier row
    assert np.allclose(amap["M"][:7, 6:], 0.0)


class _Lin:
    """One collision row on step 2 of a 3-step chunk: value -0.01 m, gradient on joint 0."""

    def __init__(self, nq, H):
        self.nq, self.H = nq, H

    def linearize(self, traj, q_now, scene, cfg):
        value = np.full((self.H, 1), np.inf)
        grad = np.zeros((self.H, 1, self.nq))
        used = np.zeros((self.H, 1), bool)
        # the row is linear in joint 0 of step 2: clearance = -0.01 + (traj[0, 2] - 0)
        value[2, 0] = -0.01 + traj[0, 2]
        grad[2, 0, 0] = 1.0
        used[2, 0] = True
        return types.SimpleNamespace(value=value, gradient=grad, used=used)

    def full_violation(self, traj, q_now, scene):
        return max(0.0, 0.01 - traj[0, 2])


def _opt(nq=2, H=3, m=2):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = TrajectoryOptimizer.__new__(TrajectoryOptimizer)
    opt.config = types.SimpleNamespace(reduction=types.SimpleNamespace(linearization_backoff=0.0),
                                       safety=types.SimpleNamespace(violation_tolerance=1e-6))
    opt.linearizer = _Lin(nq, H)
    # Dq_k(joint 0) = sum_{j<=k} Da_j(0) (an integrator), joint 1 untouched
    M = np.zeros((H * nq, H * m))
    for k in range(H):
        for j in range(k + 1):
            M[k * nq + 0, j * m + 0] = 1.0
    opt.ee_action_map = {"M": M, "a_ref": np.zeros((H, m))}
    opt._finish = lambda traj, *a, **k: types.SimpleNamespace(trajectory=traj, metrics={})
    return opt


def test_action_qp_is_the_min_norm_correction():
    opt = _opt()
    reference = np.zeros((2, 3))
    res = opt._solve_ee_action(reference, np.zeros(2), None, None, 0.01, None, [], 0.0, True, None)
    da = opt.last_ee_action["delta_action"]
    # min ||Da||^2 s.t. Da_0 + Da_1 + Da_2 >= 0.01 on component 0 -> 0.01 / 3 each, nothing on component 1
    assert np.allclose(da[:, 0], 0.01 / 3, atol=1e-5) and np.allclose(da[:, 1], 0.0, atol=1e-6)
    assert res.trajectory[0, 2] == pytest.approx(0.01, abs=1e-5)
    assert res.metrics["ee_action_qp"]["slack_max_m"] <= 1e-6


def test_action_box_is_hard_and_slack_takes_the_rest():
    opt = _opt()
    opt.ee_action_map["a_ref"] = np.full((3, 2), 0.999)   # only 0.001 of room per row on component 0
    res = opt._solve_ee_action(np.zeros((2, 3)), np.zeros(2), None, None, 0.01, None, [], 0.0, True, None)
    da = opt.last_ee_action["delta_action"]
    assert np.all(opt.ee_action_map["a_ref"] + da <= 1.0 + 1e-9)
    assert res.metrics["ee_action_qp"]["slack_max_m"] > 0.0


def test_server_sends_qp_actions_only_for_the_returned_trajectory():
    from AG3S.adapters.panda import server as srv

    H = 10
    traj = np.zeros((7, H))
    traj[0] = 0.01
    da = np.zeros((H, 6))
    da[:, 2] = 0.2
    opt = types.SimpleNamespace(last_ee_action={"trajectory": traj, "delta_action": da})
    conv = types.SimpleNamespace(delta=np.full((H, 7), 0.9, np.float32))
    actions = np.zeros((H, 7), np.float32)
    refined = np.zeros((H, 8), np.float32)
    refined[:, 0] = 0.01
    out, rec = srv.ee_action_actions(opt, conv, refined, actions)
    assert rec["used"] and np.allclose(out[:, 2], 1.0) and np.allclose(out[:, 0], 0.9) and np.all(out[:, 6] == 0)
    refined[3, 0] = 0.02   # SafePolicy returned something else (e.g. a HOLD): conversion C's actions stay
    out2, rec2 = srv.ee_action_actions(opt, conv, refined, actions)
    assert not rec2["used"] and np.array_equal(out2, actions)
    out3, rec3 = srv.ee_action_actions(types.SimpleNamespace(last_ee_action=None), conv, refined, actions)
    assert not rec3["used"] and out3 is actions


def test_flag_and_version(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss
    from tests.benchmark.test_sl_gtto_gt_perception import _args

    act = ss.build_parser()._option_string_actions["--ee-action-qp"]
    assert act.dest == "ee_action_qp" and act.default is False
    assert "ee_action_qp" in srv.PORTED_FINAL_CONFIG_DESTS
    v = "AG3S-commonto-gt2-termsoff-pass-zr-ee"
    assert pf.GT9_VERSIONS == (v,)
    assert list(pf.AG3S_VERSIONS[v]) == list(pf.AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr"]) + ["--ee-action-qp"]
    args = _args(v)
    assert pf.match_ag3s_version(args) == v and pf.version_flag_values(args)["ee_action_qp"] is True
    assert "ee_action_qp" not in pf.version_flag_values(_args("AG3S-commonto-gt2-termsoff-pass-zr"))
    for name, flags in pf.AG3S_VERSIONS.items():
        if name != v and name not in getattr(pf, "GT10_VERSIONS", ()) + getattr(pf, "GT11_VERSIONS", ()) + getattr(pf, "GT12_VERSIONS", ()) + getattr(pf, "GT13_VERSIONS", ()) + getattr(pf, "GT14_VERSIONS", ()) + getattr(pf, "GT15_VERSIONS", ()):   # SL-GTTO10 / 11 build on F7
            assert "--ee-action-qp" not in flags, name


def test_sqp_default_is_unchanged():
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    assert TrajectoryOptimizer.ee_action_map is None and TrajectoryOptimizer.last_ee_action is None


from tests.benchmark.test_sl6a_panda_final import gpu_ok  # noqa: E402,F401 (fixture)


def test_server_wires_the_optimizer(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import server as ps
    from tests.benchmark.test_sl_gtto_gt_perception import _args

    class _Stub:
        def infer(self, obs, **kw):  # pragma: no cover
            raise AssertionError

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    served, info = ps.build_safe_server(_Stub(), _args("AG3S-commonto-gt2-termsoff-pass-zr-ee"), gpu_warm_up=False)
    assert served.chunk_policy.ee_optimizer is served._policy.refiner.optimizer
    assert info["grasp_contact"]["ee_action_qp"] is True
    served0, info0 = ps.build_safe_server(_Stub(), _args("AG3S-commonto-gt2-termsoff-pass-zr"), gpu_warm_up=False)
    assert served0.chunk_policy.ee_optimizer is None and "ee_action_qp" not in info0["grasp_contact"]
