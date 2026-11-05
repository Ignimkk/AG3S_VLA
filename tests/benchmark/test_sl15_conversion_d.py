"""SL15 -- reverse conversion D (``lag_steady``) and E (``lag_steady_deadband``) (CPU; no server, no GPU).

(a) D gain: ``Dx = Dy (1 - alpha) / beta`` per row, position and rotation, action correction ``Dx / scale``;
(b) the clip keeps the correction direction (an axis-wise clip would not), ``s < 1`` lists the row in ``saturated_rows``;
(c) unchanged rows pass the policy's bits;
(d) E: a tiny change passes through, a larger or a cumulated one is corrected;
(e) the same 3 mm change: C saturates, D does not;
(f) A / B / C (outputs, records, action boxes) are bit-identical to the code before SL15 (``frozen_action_2d3a736c.py``);
(g) the mode reaches the server's reverse step and ``panda/conversion.mode`` from the launchers' ``--conversion``.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from AG3S.adapters.panda import action as pa

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark/runners"))
(AP, BP), (AR, BR) = pa.RESPONSE["pos"], pa.RESPONSE["rot"]
GP, GR = (1 - AP) / BP, (1 - AR) / BR
H = 6


# ------------------------------------------------------------------------------------------------ synthetic world
class _FakeKin:
    """FK = (q[:3], exp(q[3:6])): a joint change maps one to one to an EE change, so ``Dy`` is exact."""

    def fk(self, q):
        q = np.asarray(q, np.float64)
        return q[:3].copy(), pa.rotvec_to_matrix(q[3:6])


def _conv(delta):
    return pa.Conversion(delta=np.asarray(delta, np.float32), q_now=np.zeros(7), R_world_base=np.eye(3),
                         joint_chunk=np.zeros((len(delta), 8)), ik=[], targets_base=[])


def _chunks(dpos=None, drot=None):
    """``(reference, refined)`` joint chunks (H, 8): refined column ``i`` = ``d*[i]`` per row (absolute, so the chained
    step of row k is ``d[k] - d[k-1]``)."""
    ref = np.zeros((H, 8), np.float32)
    new = ref.copy()
    for cols, d in ((slice(0, 3), dpos), (slice(3, 6), drot)):
        if d is not None:
            new[:, cols] = np.asarray(d, np.float32)
    return ref, new


def _run(mode, delta, dpos=None, drot=None):
    ref, new = _chunks(dpos, drot)
    return pa.joint_chunk_to_delta_chunk(_FakeKin(), _conv(delta), new, ref, mode=mode)


def _x(v):  # a position path displaced along x by v[k] (absolute)
    d = np.zeros((H, 3))
    d[:, 0] = v
    return d


def _rz(v):
    d = np.zeros((H, 3))
    d[:, 2] = v
    return d


# ------------------------------------------------------------------------------------------------ (a) gain
def test_modes_are_registered():
    assert pa.CONVERSION_MODES["D"] == "lag_steady" and pa.CONVERSION_MODES["E"] == "lag_steady_deadband"
    assert [pa.CONVERSION_MODES[m] for m in "ABC"] == ["lag_inverse", "command_offset", "lag_inverse_tracking"]
    assert GP == pytest.approx(4.01, abs=0.02) and GR == pytest.approx(4.03, abs=0.02)


def test_d_gain_position_1mm():
    out, rec = _run("D", np.zeros((H, 7)), dpos=_x([0.001] * H))
    assert out[0, 0] == pytest.approx(GP * 0.001 / pa.OSC_POS_SCALE, rel=1e-6)   # (1 - alpha)/beta * 0.001 / 0.05
    assert out[0, 0] == pytest.approx(0.0802, abs=1e-3) and np.all(out[0, 1:6] == 0)
    assert np.all(out[1:] == 0)                                                   # the change happens once: no carry-over
    assert rec["mode"] == "D" and rec["mode_name"] == "lag_steady" and rec["corrected_rows"][0] == 0
    assert rec["gain_pos"] == pytest.approx(GP) and rec["gain_rot"] == pytest.approx(GR) and rec["min_scale_s"] == 1.0
    assert rec["saturated_rows"] == []


def test_d_gain_rotation_and_direction_per_axis():
    out, rec = _run("D", np.zeros((H, 7)), drot=_rz([0.01] * H))                  # 0.01 rad about z
    assert out[0, 5] == pytest.approx(GR * 0.01 / pa.OSC_ORI_SCALE, rel=1e-5)
    assert np.all(out[0, :5] == 0) and np.all(out[1:] == 0)
    out, _ = _run("D", np.zeros((H, 7)), dpos=_x([-0.002] * H))
    assert out[0, 0] == pytest.approx(-GP * 0.002 / pa.OSC_POS_SCALE, rel=1e-6)


def test_d_adds_to_the_policy_action_and_keeps_the_gripper():
    delta = np.zeros((H, 7))
    delta[0, :3] = [0.3, -0.2, 0.1]
    delta[:, 6] = np.linspace(-1, 1, H)
    out, _ = _run("D", delta, dpos=_x([0.001] * H))
    assert out[0, 0] == pytest.approx(np.float32(0.3) + GP * 0.001 / 0.05, rel=1e-6)
    assert out[0, 1] == np.float32(-0.2) and out[0, 2] == np.float32(0.1)
    assert np.array_equal(out[:, 6], np.asarray(delta, np.float32)[:, 6])


# ------------------------------------------------------------------------------------------------ (b) direction-preserving clip
def test_d_clip_preserves_the_correction_direction():
    delta = np.zeros((H, 7))
    delta[0, :2] = [0.5, 0.1]
    dpos = np.zeros((H, 3))
    dpos[:, 0], dpos[:, 1] = 0.02, 0.004                    # corr = (1.604, 0.321, 0, ...)
    out, rec = _run("D", delta, dpos=dpos)
    corr = np.array([GP * 0.02 / 0.05, GP * 0.004 / 0.05, 0.0])
    s = (1.0 - 0.5) / corr[0]
    assert s < 1 and rec["saturated_rows"] == [0] and rec["min_scale_s"] == pytest.approx(s, rel=1e-5)
    assert out[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert out[0, 1] == pytest.approx(0.1 + s * corr[1], rel=1e-5)                # an axis-wise clip would give 0.1 + 0.321
    d_out = np.asarray(out[0, :3], np.float64) - np.array([0.5, 0.1, 0.0])
    assert np.allclose(d_out / np.linalg.norm(d_out), corr / np.linalg.norm(corr), atol=1e-5)
    axiswise = np.clip(np.array([0.5, 0.1, 0.0]) + corr, -1, 1)
    assert abs(axiswise[1] - out[0, 1]) > 0.1                                      # the two differ


def test_d_clip_negative_side_and_already_saturated_policy():
    delta = np.zeros((H, 7))
    delta[0, 0] = -0.8
    out, rec = _run("D", delta, dpos=_x([-0.02] * H))                             # corr x = -1.604 -> s = 0.2 / 1.604
    assert out[0, 0] == pytest.approx(-1.0, abs=1e-6) and rec["saturated_rows"] == [0]
    delta[0, 0] = 1.0                                                              # policy at the bound, correction outward
    out, rec = _run("D", delta, dpos=_x([0.02] * H))
    assert out[0, 0] == 1.0 and rec["min_scale_s"] == 0.0 and rec["saturated_rows"] == [0]
    delta[0, 0] = 1.4                                                              # an out-of-range policy action is clipped first
    out, _ = _run("D", delta, dpos=_x([0.001] * H))
    assert out[0, 0] == 1.0


# ------------------------------------------------------------------------------------------------ (c) unchanged rows
@pytest.mark.parametrize("mode", ["D", "E"])
def test_unchanged_rows_pass_the_policy_bits(mode):
    delta = np.random.default_rng(1).uniform(-1.3, 1.3, (H, 7)).astype(np.float32)   # incl. out-of-range: bits stay
    ref, new = _chunks(dpos=_x([0, 0, 0, 0.02, 0.02, 0.02]))
    out, rec = pa.joint_chunk_to_delta_chunk(_FakeKin(), _conv(delta), new, ref, mode=mode)
    assert rec["passthrough_rows"] == [0, 1, 2]
    assert out[:3].tobytes() == delta[:3].tobytes()
    assert 3 in rec["corrected_rows"] and out[3, 0] != delta[3, 0]
    out, rec = pa.joint_chunk_to_delta_chunk(_FakeKin(), _conv(delta), ref.copy(), ref, mode=mode)   # nothing changed at all
    assert out.tobytes() == delta.tobytes() and rec["passthrough_rows"] == list(range(H)) and rec["corrected_rows"] == []
    assert rec["min_scale_s"] == 1.0 and rec["saturated_rows"] == []


# ------------------------------------------------------------------------------------------------ (d) E deadband
def test_e_deadband_passes_a_tiny_change():
    delta = np.zeros((H, 7), np.float32)
    delta[:, :3] = 0.25
    out, rec = _run("E", delta, dpos=_x([0.0003] * H))                             # 0.3 mm < 0.5 mm, cumulative 0.3 mm < 1 mm
    assert rec["mode"] == "E" and rec["deadband_rows"][0] == 0 and 0 in rec["passthrough_rows"]
    assert out[0].tobytes() == delta[0].tobytes() and rec["corrected_rows"][:1] != [0]
    out_d, rec_d = _run("D", delta, dpos=_x([0.0003] * H))                         # D corrects the same row
    assert out_d[0, 0] != delta[0, 0] and rec_d["corrected_rows"][0] == 0 and "deadband_rows" not in rec_d


def test_e_deadband_tiny_rotation_and_larger_changes():
    delta = np.zeros((H, 7), np.float32)
    ang = np.radians(0.2)
    out, rec = _run("E", delta, drot=_rz([ang] * H))                               # 0.2 deg < 0.3 deg and < 0.5 deg
    assert 0 in rec["deadband_rows"] and out[0].tobytes() == delta[0].tobytes()
    ang = np.radians(0.4)                                                          # 0.4 deg > 0.3 deg: corrected
    out, rec = _run("E", delta, drot=_rz([ang] * H))
    assert 0 not in rec["deadband_rows"] and out[0, 5] == pytest.approx(GR * ang / pa.OSC_ORI_SCALE, rel=1e-5)
    out, rec = _run("E", delta, dpos=_x([0.0006] * H))                             # 0.6 mm > 0.5 mm: corrected
    assert 0 not in rec["deadband_rows"] and out[0, 0] == pytest.approx(GP * 0.0006 / 0.05, rel=1e-5)
    # a change just under the per-row limit is passed; the limits are module constants
    assert (pa.DEADBAND_POS_M, pa.DEADBAND_CUM_POS_M) == (0.0005, 0.001)
    assert (pa.DEADBAND_ROT_RAD, pa.DEADBAND_CUM_ROT_RAD) == (np.radians(0.3), np.radians(0.5))
    out, rec = _run("E", delta, dpos=_x([0.00049] * H))
    assert 0 in rec["deadband_rows"]


def test_e_cumulative_change_leaves_the_deadband():
    delta = np.zeros((H, 7), np.float32)
    # 0.4 mm per row for three rows: cumulative 0.4, 0.8, 1.2 mm -> rows 0, 1 pass, row 2 (cum >= 1 mm) is corrected
    out, rec = _run("E", delta, dpos=_x([0.0004, 0.0008, 0.0012, 0.0012, 0.0012, 0.0012]))
    assert rec["deadband_rows"] == [0, 1] and out[0].tobytes() == delta[0].tobytes() and out[1].tobytes() == delta[1].tobytes()
    assert 2 in rec["corrected_rows"] and out[2, 0] == pytest.approx(GP * 0.0004 / 0.05, rel=1e-5)
    # rows after the change: the refined path stays shifted, the row's own change is 0 but the cumulative is not -> not a deadband row
    assert 3 not in rec["deadband_rows"] and 3 in rec["corrected_rows"]
    # rotation: 0.2 deg per row, cumulative 0.2, 0.4, 0.6 deg -> row 2 leaves the deadband
    a = np.radians(0.2)
    out, rec = _run("E", delta, drot=_rz([a, 2 * a, 3 * a, 3 * a, 3 * a, 3 * a]))
    assert rec["deadband_rows"] == [0, 1] and 2 in rec["corrected_rows"]


def test_e_equals_d_outside_the_deadband():
    rng = np.random.default_rng(4)
    delta = rng.uniform(-0.6, 0.6, (H, 7)).astype(np.float32)
    dpos = np.cumsum(rng.uniform(0.002, 0.004, (H, 3)), axis=0)                    # every step > 2 mm
    od, rd = _run("D", delta, dpos=dpos)
    oe, re = _run("E", delta, dpos=dpos)
    assert od.tobytes() == oe.tobytes() and re["deadband_rows"] == []
    assert {k: v for k, v in rd.items() if k not in ("mode", "mode_name")} == \
        {k: v for k, v in re.items() if k not in ("mode", "mode_name", "deadband_rows")}


# ------------------------------------------------------------------------------------------------ (e) C vs D at 3 mm
def test_c_saturates_where_d_does_not_for_a_3mm_change():
    delta = np.zeros((H, 7))
    delta[0, 0] = 0.3                                                              # a moving policy action
    dpos = _x([0.003] * H)
    _, rec_c = _run("C", delta, dpos=dpos)
    out_d, rec_d = _run("D", delta, dpos=dpos)
    assert 0 in rec_c["saturated_rows"]                                            # 0.3 + 0.003/(0.0706 * 0.05) = 1.15 > 1
    assert rec_d["saturated_rows"] == [] and rec_d["min_scale_s"] == 1.0
    assert out_d[0, 0] == pytest.approx(0.3 + GP * 0.003 / 0.05, rel=1e-6) and 0.3 < out_d[0, 0] < 1.0


# ------------------------------------------------------------------------------------------------ (f) A / B / C unchanged
@pytest.fixture(scope="module")
def kin_pair():
    from AG3S.adapters.panda import model as pm
    from tests.benchmark import frozen_action_2d3a736c as old

    robot = pm.build_panda_constraint_model(pm.panda_constraint_link_filter("gripper"))
    return pa.PandaKinematics(robot), old.PandaKinematics(robot), old


Q0 = np.array([0.0, 0.17, 0.0, -2.45, 0.0, 2.6, 0.79])


@pytest.mark.parametrize("bump", [0.004, 0.05])                                   # small and saturating joint changes
@pytest.mark.parametrize("seed", [3, 11])
def test_abc_are_bit_identical_to_the_code_before_sl15(kin_pair, bump, seed):
    kin_new, kin_old, old = kin_pair
    A = np.clip(np.random.default_rng(seed).normal(0, 0.5, (10, 7)), -1, 1).astype(np.float32)
    prev = (np.full(3, 0.002), np.full(3, 0.01))
    c_new = pa.delta_chunk_to_joint_chunk(kin_new, A, Q0, np.eye(4), prev_step_world=prev)
    c_old = old.delta_chunk_to_joint_chunk(kin_old, A, Q0, np.eye(4), prev_step_world=prev)
    assert np.array_equal(c_new.joint_chunk, c_old.joint_chunk)
    ref = c_new.joint_chunk.astype(np.float32)
    mod = ref.copy()
    mod[3:, 1] += bump
    mod[6:, 4] -= 2 * bump
    for mode in ("A", "B", "C"):
        o_new, r_new = pa.joint_chunk_to_delta_chunk(kin_new, c_new, mod, ref, mode=mode)
        o_old, r_old = old.joint_chunk_to_delta_chunk(kin_old, c_old, mod, ref, mode=mode)
        assert o_new.dtype == o_old.dtype and o_new.tobytes() == o_old.tobytes(), mode
        assert json.dumps(r_new, sort_keys=True) == json.dumps(r_old, sort_keys=True), mode   # same record keys and values
        b_new = pa.action_box_from_conversion(c_new, margin=0.05, n_rows=5, mode=mode)
        b_old = old.action_box_from_conversion(c_old, margin=0.05, n_rows=5, mode=mode)
        for f in ("actions", "alpha", "beta", "scale", "R_world_base", "q_now"):
            assert np.array_equal(getattr(b_new, f), getattr(b_old, f)), (mode, f)
        assert (b_new.mode, b_new.margin, b_new.n_rows) == (b_old.mode, b_old.margin, b_old.n_rows)
    assert json.dumps(pa.conventions(), sort_keys=True, default=str) == json.dumps(old.conventions(), sort_keys=True, default=str)


def test_action_box_for_d_and_e_is_the_steady_gain(kin_pair):
    kin_new, _, _ = kin_pair
    A = np.clip(np.random.default_rng(3).normal(0, 0.5, (10, 7)), -1, 1).astype(np.float32)
    conv = pa.delta_chunk_to_joint_chunk(kin_new, A, Q0, np.eye(4))
    for mode in ("D", "E"):
        box = pa.action_box_from_conversion(conv, margin=0.05, n_rows=5, mode=mode)
        assert box.mode == mode and np.all(box.alpha == 0)
        assert box.gain[:3] == pytest.approx(GP / pa.OSC_POS_SCALE) and box.gain[3:] == pytest.approx(GR / pa.OSC_ORI_SCALE)
    with pytest.raises(ValueError):
        pa.action_box_from_conversion(conv, margin=0.05, n_rows=5, mode="F")
    with pytest.raises(ValueError):
        pa.joint_chunk_to_delta_chunk(_FakeKin(), _conv(np.zeros((H, 7))), *reversed(_chunks()), mode="F")


# ------------------------------------------------------------------------------------------------ (g) the mode reaches the server
@pytest.mark.parametrize("mode", ["A", "B", "C", "D", "E"])
def test_server_env_accepts_the_mode(monkeypatch, mode):
    from AG3S.adapters.panda import server as ps

    monkeypatch.setenv(ps.ENV_CONVERSION, mode.lower())
    assert ps.conversion_mode() == mode
    monkeypatch.setenv(ps.ENV_CONVERSION, "F")
    with pytest.raises(ValueError, match="expected one of"):
        ps.conversion_mode()


class _SafeStub:
    def __init__(self, refined, reference):
        self.refined, self.reference = refined, reference

    def reset(self):
        pass

    def infer(self, obs, **kwargs):
        return {"actions": np.array(self.refined, np.float32), "actions_reference": np.array(self.reference, np.float32),
                "safe": True, "seq": 3, "trajopt_status": "optimal", "max_violation_m": 0.0, "verdict_reasons": [],
                "notes": []}


def test_d_reaches_the_servers_reverse_step_and_the_call_record(kin_pair, monkeypatch):
    """env ``AG3S_PANDA_CONVERSION=D`` -> ``conversion_mode()`` -> ``PandaSafeServer(mode=)`` -> ``joint_chunk_to_delta_chunk``
    -> ``response["panda/conversion"]["mode"]`` (what ``libero_safe_client`` writes as the call's ``conversion``)."""
    from AG3S.adapters.panda import server as ps

    kin, _, _ = kin_pair
    A = np.clip(np.random.default_rng(3).normal(0, 0.4, (10, 7)), -1, 1).astype(np.float32)
    conv = pa.delta_chunk_to_joint_chunk(kin, A, Q0, np.eye(4))
    reference = conv.joint_chunk.astype(np.float32)
    refined = reference.copy()
    refined[:, :7] += 0.003 * np.sin(np.arange(10))[:, None]
    seen = {}
    for mode in ("B", "C", "D", "E"):
        monkeypatch.setenv(ps.ENV_CONVERSION, mode)
        chunk_policy = ps.ChunkToJointPolicy(SimpleNamespace(), kin)
        chunk_policy.last = conv
        srv = ps.PandaSafeServer(_SafeStub(refined, reference), chunk_policy, mode=ps.conversion_mode())
        resp = srv.infer({})
        rec = resp["panda/conversion"]
        assert rec["mode"] == mode and rec["mode_name"] == pa.CONVERSION_MODES[mode]
        json.dumps(rec)
        seen[mode] = (np.asarray(resp["actions"]), rec)
    d_act, d_rec = seen["D"]
    assert d_rec["gain_pos"] == pytest.approx(GP) and d_rec["min_scale_s"] <= 1.0 and d_rec["corrected_rows"]
    assert d_act.tobytes() != seen["B"][0].tobytes() and d_act.tobytes() != seen["C"][0].tobytes()
    assert np.array_equal(d_act[:, 6], A[:, 6])                                    # gripper: the policy's
    assert "deadband_rows" in seen["E"][1] and "deadband_rows" not in d_rec


SAFETY_BASE = ["--task-suite-name", "obstacle_avoidance", "--level", "1", "--policy-checkpoint", "pi05_libero"]


def _launcher_modules():
    from benchmark.runners import eval_matrix as em
    from benchmark.runners import launch_libero_paired as lp
    from benchmark.runners import launch_libero_safety_paired as ls

    return em, lp, ls


@pytest.mark.parametrize("mode", ["A", "B", "C", "D", "E"])
def test_launchers_accept_the_mode(mode):
    em, lp, ls = _launcher_modules()
    base = ["--task-ids", "0", "--episode-ids", "0", "--run-id", "x"]
    assert lp.build_parser().parse_args(base + ["--conversion", mode]).conversion == mode
    assert ls.build_parser().parse_args(base + SAFETY_BASE + ["--conversion", mode]).conversion == mode
    jobs = em.build_jobs(suites=[("libero_goal", None)], version="V11AB-pb7", conversion=mode, split="dev", episode_ids=[0],
                         run_prefix="t", step="S")
    assert jobs and all(j["conversion"] == mode and j["argv"][j["argv"].index("--conversion") + 1] == mode for j in jobs)


@pytest.mark.parametrize("mode", ["B", "D", "E"])
def test_eval_matrix_plan_cli_writes_the_mode_into_every_job_line(tmp_path, monkeypatch, mode):
    em, _, _ = _launcher_modules()
    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(ROOT))
    assert em.main(["plan", "--version", "V11AB-pb7", "--conversion", mode, "--suites", "libero_goal", "--split", "dev",
                    "--tasks", "0", "--episodes", "0", "--vla-only", "skip", "--run-prefix", "t",
                    "--out-dir", str(tmp_path)]) == 0
    lines = [l for l in (tmp_path / "jobs.txt").read_text().splitlines() if l and not l.startswith("#")]
    assert lines and all(f"--conversion {mode}" in l for l in lines)
    assert json.loads((tmp_path / "plan.json").read_text())["meta"]["conversion"] == mode


@pytest.mark.parametrize("bad", ["F", "d", ""])
def test_launchers_still_refuse_other_values(bad, capsys):
    em, lp, ls = _launcher_modules()
    base = ["--task-ids", "0", "--episode-ids", "0", "--run-id", "x"]
    for mod in (lp, ls):
        with pytest.raises(SystemExit):
            mod.build_parser().parse_args(base + (SAFETY_BASE if mod is ls else []) + ["--conversion", bad])
    with pytest.raises(ValueError, match="conversion must be"):
        em.build_jobs(suites=[("libero_goal", None)], version="V11AB-pb7", conversion=bad, split="dev", episode_ids=[0],
                      run_prefix="t", step="S")
    capsys.readouterr()


def test_launcher_hands_d_to_the_server_environment_and_the_manifest(monkeypatch, tmp_path):
    """``--conversion D`` -> the server process gets ``AG3S_PANDA_CONVERSION=D`` (the only channel) and the manifest says D."""
    import common.run_artifacts as ra
    from benchmark.runners import launch_libero_paired as lp

    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    seen = {"env": None, "create": None}

    def popen(argv, **kw):
        seen["env"] = kw.get("env")
        return SimpleNamespace(pid=4242)

    monkeypatch.setattr(ra, "create_run", lambda **kw: seen.update(create=kw) or run_dir)
    monkeypatch.setattr(ra, "record_execution", lambda run, *, settings, checkpoint=None: None)
    for name, val in (("choose_port", lambda requested=None: 23456), ("openpi_python", lambda: "/fake/openpi/python"),
                      ("libero_python", lambda: "/fake/libero/python"), ("wait_ready", lambda s, p, t: 1.0),
                      ("client_environment", lambda: {}), ("stop_own_process", lambda s: 0)):
        monkeypatch.setattr(lp.l0, name, val)
    monkeypatch.setattr(lp, "protocol_settings", lambda a: {"p": 1})
    monkeypatch.setattr(lp, "runtime_schedule", lambda a: {"r": 1})
    monkeypatch.setattr(lp, "gpu_compute_apps", lambda: [])
    monkeypatch.setattr(lp, "GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.delenv("AG3S_GPU_LOCK_HELD", raising=False)
    monkeypatch.setattr(lp, "subprocess", SimpleNamespace(STDOUT=subprocess.STDOUT, Popen=popen,
                                                          run=lambda *a, **k: SimpleNamespace(returncode=0)))
    args = lp.build_parser().parse_args(["--task-ids", "0", "--episode-ids", "0", "--run-id", "x", "--conversion", "D",
                                         "--ag3s-version", "V11AB-pb7"])
    for name, value in (("resize_size", 224), ("replan_steps", 5), ("num_steps_wait", 10)):
        setattr(args, name, getattr(args, name, value))
    lp.run_mode("vla_v11ab", args, {"path": "/ckpt"}, {"versions": {"v": 1}})
    assert seen["env"]["AG3S_PANDA_CONVERSION"] == "D"
    assert seen["create"]["settings"]["panda_conversion_mode"] == "D"
