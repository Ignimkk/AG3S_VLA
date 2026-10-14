"""SL5a — Panda server injection + EE-delta <-> joint conversion (CPU)."""
from __future__ import annotations

import json
import pathlib
import subprocess

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROBOSUITE_OSC_JSON = ROOT / ".venv-libero/lib/python3.8/site-packages/robosuite/controllers/config/osc_pose.json"


@pytest.fixture(scope="module")
def kin():
    from AG3S.adapters.panda import action as pa
    from AG3S.adapters.panda import model as pm

    return pa.PandaKinematics(pm.build_panda_constraint_model(pm.panda_constraint_link_filter("gripper")))


def test_osc_constants_match_robosuite():
    from AG3S.adapters.panda import action as pa

    if not ROBOSUITE_OSC_JSON.exists():
        pytest.skip("robosuite (.venv-libero) not installed")
    cfg = json.loads(ROBOSUITE_OSC_JSON.read_text())
    assert cfg["output_max"] == [pa.OSC_POS_SCALE] * 3 + [pa.OSC_ORI_SCALE] * 3
    assert cfg["input_max"] == pa.OSC_INPUT_CLIP and cfg["input_min"] == -pa.OSC_INPUT_CLIP
    assert cfg["control_delta"] is True


def test_rotvec_roundtrip():
    from AG3S.adapters.panda import action as pa

    rng = np.random.default_rng(0)
    for _ in range(50):
        v = rng.normal(size=3)
        v *= rng.uniform(0, 3.0) / np.linalg.norm(v)
        assert np.allclose(pa.matrix_to_rotvec(pa.rotvec_to_matrix(v)), v, atol=1e-9)


def test_response_model_file():
    from AG3S.adapters.panda import action as pa

    fit = json.loads(pa.OSC_RESPONSE_FIT.read_text())
    assert fit["pos"]["r2"] > 0.95 and fit["rot"]["r2"] > 0.9
    assert "init 0-4" in fit["provenance"]["data_split"]
    assert pa.RESPONSE["pos"] == (fit["pos"]["alpha"], fit["pos"]["beta"])


def test_forward_ik_and_passthrough_bits(kin):
    from AG3S.adapters.panda import action as pa

    q0 = np.array([0.0, 0.17, 0.0, -2.45, 0.0, 2.6, 0.79])
    A = np.clip(np.random.default_rng(1).normal(0, 0.5, (10, 7)), -1, 1).astype(np.float32)
    conv = pa.delta_chunk_to_joint_chunk(kin, A, q0, np.eye(4), prev_step_world=(np.full(3, 0.002), np.zeros(3)))
    assert conv.record()["ik_pos_err_m_max"] < 1e-4
    ref = conv.joint_chunk.astype(np.float32)
    out, rec = pa.joint_chunk_to_delta_chunk(kin, conv, ref.copy(), ref)
    assert out.dtype == A.dtype and np.array_equal(out, A)  # unchanged chunk -> original bits
    assert rec["passthrough_rows"] == list(range(10)) and rec["corrected_rows"] == []


def test_reverse_only_touches_changed_rows(kin):
    from AG3S.adapters.panda import action as pa

    q0 = np.array([0.0, 0.17, 0.0, -2.45, 0.0, 2.6, 0.79])
    A = (np.zeros((10, 7)) + [0.2, 0, -0.2, 0, 0, 0, -1]).astype(np.float32)
    conv = pa.delta_chunk_to_joint_chunk(kin, A, q0, np.eye(4))
    ref = conv.joint_chunk.astype(np.float32)
    mod = ref.copy()
    mod[6:, 2] += 0.01
    out, rec = pa.joint_chunk_to_delta_chunk(kin, conv, mod, ref)
    assert all(np.array_equal(out[k], A[k]) for k in range(6))
    assert rec["corrected_rows"][0] == 6
    assert np.array_equal(out[:, 6], A[:, 6])  # gripper always the original


def test_panda_layout():
    from AG3S.adapters.panda import action as pa

    lay = pa.panda_chunk_layout([f"panda_joint{i}" for i in range(1, 8)])
    assert lay.nq_opt == 7 and lay.passthrough == (7,) and lay.action_dim == 8
    chunk = np.arange(80, dtype=float).reshape(10, 8)
    assert np.array_equal(lay.chunk_to_trajectory(chunk), chunk[:, :7].T)


def test_adapt_request_and_vla_only_strip():
    from AG3S.adapters.panda import server as ps

    rs9 = np.array([0.1, 0.2, 0.3, -2.0, 0.4, 2.5, 0.7, 0.03, -0.03])
    req = {"observation/state": np.zeros(8), "ag3s/robot_state/head": rs9, "ag3s/robot_state/left_wrist": rs9,
           "panda/robot_state": rs9, "panda/T_world_base": np.eye(4), "ag3s/policy_seed": 5}
    out = ps.adapt_request(req)
    assert out["ag3s/robot_state/head"].shape == (7,)
    assert np.allclose(out["state"][:7], rs9[:7]) and np.isclose(out["state"][7], 0.03 / 0.04)

    seen = {}

    class Inner:
        def infer(self, obs, **kw):
            seen.update(obs)
            return {"actions": np.zeros((10, 7))}

    ps.PandaVLAOnly(Inner()).infer(req)
    assert set(seen) == {"observation/state", "policy_seed"} and seen["policy_seed"] == 5


def test_rby1_golden_identity():
    """The SL5a core changes keep the RB-Y1 SafePolicy / serve_safe path bit-identical (fixture captured before)."""
    for flag, fixture in (("--check", "sl5a_rby1_golden.npz"), ("--check-serve-safe", "sl5a_serve_safe_golden.json")):
        out = subprocess.run([str(ROOT / ".venv-ag3s/bin/python"), "-m", "tests.trajopt._sl5a_golden", flag,
                              str(ROOT / "tests/trajopt/data" / fixture)], cwd=str(ROOT), capture_output=True,
                             text=True, env={**__import__("os").environ, "MUJOCO_GL": "osmesa", "CUDA_VISIBLE_DEVICES": ""})
        last = [l for l in out.stdout.splitlines() if l.startswith("{")][-1]
        assert json.loads(last)["identical"] is True, out.stdout[-2000:] + out.stderr[-2000:]


def test_mode_b_passthrough_and_realisation(kin):
    from AG3S.adapters.panda import action as pa

    q0 = np.array([0.0, 0.17, 0.0, -2.45, 0.0, 2.6, 0.79])
    A = (np.zeros((10, 7)) + [0.3, 0.1, -0.2, 0, 0, 0.05, -1]).astype(np.float32)
    conv = pa.delta_chunk_to_joint_chunk(kin, A, q0, np.eye(4))
    ref = conv.joint_chunk.astype(np.float32)
    out, rec = pa.joint_chunk_to_delta_chunk(kin, conv, ref.copy(), ref, mode="B")
    assert np.array_equal(out, A) and rec["mode"] == "B" and rec["passthrough_rows"] == list(range(10))
    mod = ref.copy()
    mod[2:, 1] += 0.01
    outA, _ = pa.joint_chunk_to_delta_chunk(kin, conv, mod, ref, mode="A")
    outB, _ = pa.joint_chunk_to_delta_chunk(kin, conv, mod, ref, mode="B")
    rA = pa.realisation_record(kin, conv, outA, mod.astype(np.float64), 5)
    rB = pa.realisation_record(kin, conv, outB, mod.astype(np.float64), 5)
    # A inverts the response model, so it realises the refined path more closely than B (unsaturated case)
    assert rA["predicted_vs_refined_mm_max"] <= rB["predicted_vs_refined_mm_max"] + 1e-9
    assert np.abs(outB.astype(float) - A).max() < np.abs(outA.astype(float) - A).max()
    with pytest.raises(ValueError):
        pa.joint_chunk_to_delta_chunk(kin, conv, mod, ref, mode="C")


def test_panda_latch_calibration():
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt.grasp_latch import LatchConfig

    cal = ps.gripper_calibration()
    cfg = ps.panda_latch_config()
    e, g = cal["measured"]["empty_closure"]["norm"], cal["measured"]["grasped_lifted"]["norm_min"]
    assert e < cfg.empty_below < g  # separates the measured empty closure from the grasped openings
    d = LatchConfig()
    for f in d.__dataclass_fields__:
        if f != "empty_below":
            assert getattr(cfg, f) == getattr(d, f), f  # every other V11AB default unchanged


def _feedback(seq, n, safe=True):
    return {"available": True, "seq": seq, "t_step_start": 10, "n_steps": n, "n_exec": n if safe else 0,
            "n_hold": 0 if safe else n, "executed": [safe] * n, "executed_chunk": "refined" if safe else "none"}


def test_applied_command_rows_contract(kin):
    """applied_arm = the applied OSC rows in the server's joint command convention (same map, same start state)."""
    from types import SimpleNamespace

    from AG3S.adapters.panda import action as pa
    from AG3S.adapters.panda import server as ps

    q0 = np.array([0.0, 0.17, 0.0, -2.45, 0.0, 2.6, 0.79])
    A = np.clip(np.random.default_rng(3).normal(0, 0.5, (10, 7)), -1, 1).astype(np.float32)
    conv = pa.delta_chunk_to_joint_chunk(kin, A, q0, np.eye(4), prev_step_world=(np.full(3, 0.002), np.zeros(3)))
    ref = conv.joint_chunk.astype(np.float32)
    rows = pa.applied_command_rows(kin, conv, A[:5])
    assert np.array_equal(rows, conv.joint_chunk[:5, :7])  # unintervened rows -> the reference rows, bit for bit
    hold = np.zeros((5, 7)); hold[:, 6] = A[4, 6]
    assert not np.allclose(pa.applied_command_rows(kin, conv, hold), conv.joint_chunk[:5, :7])

    srv = ps.PandaSafeServer(None, SimpleNamespace(kin=kin), mode="B")
    srv._sent[3] = (conv, ref, ref)
    obs = {"ag3s/exec_feedback": _feedback(3, 5), "panda/exec_applied_osc": A[:5].astype(np.float64)}
    rec = srv._feedback(obs)
    assert rec["used"] and rec["applied_equals_reference_rows"] == 5 and "panda/exec_applied_osc" not in obs
    assert obs["ag3s/exec_feedback"]["applied_arm"].shape == (5, 7)
    # a partial chunk (episode ended / fewer steps ran): rows must match n_steps
    obs = {"ag3s/exec_feedback": _feedback(3, 2), "panda/exec_applied_osc": A[:2].astype(np.float64)}
    assert srv._feedback(obs)["n_steps"] == 2
    for bad in ({"ag3s/exec_feedback": _feedback(9, 5), "panda/exec_applied_osc": A[:5]},      # unknown seq
                {"ag3s/exec_feedback": _feedback(3, 5)},                                       # no OSC rows
                {"ag3s/exec_feedback": _feedback(3, 4), "panda/exec_applied_osc": A[:5]}):     # row count mismatch
        rec = srv._feedback(bad)
        assert rec["used"] is False and bad["ag3s/exec_feedback"]["available"] is False


def test_client_confirms_feedback_after_execution():
    """The py3.8 client counts the steps that actually ran and measures the gripper at this request."""
    code = r'''
import sys, json, numpy as np
sys.path.insert(0, "benchmark/runners")
import libero_safe_client as c
cl = c.SafeClient(None, 7, 0, 0, 5)
cl.eef = [None] * 8
cl.sent = {"seq": 2, "t_call": 7, "safe": True, "applied": np.arange(35.0).reshape(5, 7) / 100,
           "planned": np.zeros((5, 7))}
out = {}
cl.eef += [None] * 3                                   # only 3 of the 5 rows ran
fb, osc = cl._confirmed_feedback(np.r_[np.zeros(7), 0.012, -0.012])
out["partial"] = [fb["n_steps"], fb["n_exec"], osc.shape[0], round(float(fb["measured_gripper"][0]), 4)]
cl.eef += [None] * 2
cl.sent["safe"] = False
fb, osc = cl._confirmed_feedback(np.r_[np.zeros(7), 0.04, -0.04])
out["hold"] = [fb["n_steps"], fb["n_exec"], fb["n_hold"], fb["executed_chunk"], osc.shape[0]]
print(json.dumps(out))
'''
    r = subprocess.run([str(ROOT / ".venv-libero/bin/python"), "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0 and "No module named" in r.stderr:
        pytest.skip(r.stderr.splitlines()[-1])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["partial"] == [3, 3, 3, 0.3]
    assert out["hold"] == [5, 0, 5, "none", 5]
