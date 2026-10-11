"""T43 Q in MuJoCo — the servo model against simulated servo steps, and the executed outcome.

From **bit-exact** replays of recorded runs (`_hold_replay.replay_to` asserts every recorded `qpos`):

1. **The model predicts MuJoCo.** The server's observer is fed only what the server received
   (capture `qpos` + each chunk's `exec_feedback.applied_arm`); its prediction of every sim sub-step of
   the next rows is compared with MuJoCo's — at rest after a HOLD (C1807 t=216) and moving (B1968 t=112).
2. **The reachable set from rest is the servo's.** The first row's reach after a 10.8° and a 21.6° command
   (MuJoCo vs model): 1.9–2.7° and 3.9–5.1° (MuJoCo, left arm 1–3), i.e. the 21.6° escape is reachable —
   the rest-start envelope (2.29°) clipped it.
3. **The re-solved contact chunks do not touch.** The recorded plan of T40 C1807 seq 28 (contact t=218) and
   T41 a C1807 seq 35 (contact t=276) is executed through the client's `apply_action` and touches the bar;
   the plan re-solved with `--servo-model --sweep-check` (same recorded inputs) does not.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _t43q_exec as E  # noqa: E402

from tests.trajopt import _t43q_replay as RP  # noqa: E402

DEG = np.pi / 180.0
records = pytest.mark.skipif(not RP.available(), reason="T40 G / T41 a E3b 기록이 없다")
BOTH = {"limits": {"servo_model": True}, "reduction": {"sweep_check": True}}


def _state(name, seq):
    """What the server's observer would hold at `seq` (recorded history + this chunk's feedback)."""
    from AG3S.trajopt import wire
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    sh = RP._setup(name)
    cfg = sh["T"].to_config(sh["st"], collision=True, cost=dict(sh["st"].cost), sqp={})
    opt = TrajOptChunkRefiner(sh["robot"], sh["layout"], lambda _c: None,
                              cfg.with_overrides({"limits": {"servo_model": True}})).optimizer
    obs = RP.history_state(name, seq, opt.servo)
    fb = wire.unpack_exec_feedback({"exec_feedback": RP.summary(name, seq).get("exec_feedback")})
    return opt.servo, obs.update(RP.q_now(name, seq)[sh["layout"].q_indices], fb, seq), sh["layout"]


def _commands(layout, chunk):
    return layout.chunk_to_trajectory(np.asarray(chunk, np.float64))[:, :8]


@pytest.fixture(scope="module")
def c1807_t216():
    name, seq = "T40C1807", 28
    t0 = int(RP._setup(name)["rows"][seq]["t_step"])
    assert t0 == 216
    return E.load(RP.RUNS[name][1], t0)


@records
def test_the_model_predicts_mujoco_from_rest_after_a_hold(c1807_t216):
    client, local, _ = c1807_t216
    servo, st, layout = _state("T40C1807", 28)
    assert st.source.startswith("propagated") and np.abs(st.v).max() < 1e-3   # at rest after HOLD
    S = servo.substeps
    reach = {}
    for step in (10.8, 21.6):
        U = np.tile(st.q[:, None], (1, 8))
        U[1:4, :] += step * DEG                         # left arm 1–3, away from the bar side
        chunk = np.zeros((8, 16))
        chunk[:, :7], chunk[:, 8:15] = U[:7].T, U[7:].T
        ex = E.execute(client, local, chunk, n=2, every=33)
        sim = ex["joint_trace"][1:].reshape(2, S, 14)                         # (row, sub-step, joint)
        pred = servo.rollout(st.q, st.v, U[:, :2], st.tau, samples=list(range(1, S + 1)))
        err = np.abs(np.transpose(pred["q_samples"], (1, 2, 0)) - sim)
        assert np.degrees(err.max()) < 0.25, step                             # every sub-step, 2 rows
        reach[step] = np.degrees(sim[0, -1, 1:4] - st.q[1:4])
    assert np.all((reach[10.8] > 1.9) & (reach[10.8] < 2.7))                  # T41 b measured 2.0–2.6°
    assert np.all(reach[21.6] > 1.5 * 2.292)                                  # past the 2.29° envelope


@records
def test_the_model_predicts_mujoco_while_moving():
    name, seq = "T40C1807", 20                     # the fastest capture of the run (0.77 rad/s)
    t0 = int(RP._setup(name)["rows"][seq]["t_step"])
    client, local, _ = E.load(RP.RUNS[name][1], t0)
    servo, st, layout = _state(name, seq)
    assert np.abs(st.v).max() > 0.5                                           # the arm is moving
    chunk = np.asarray(RP.npz(name, seq)["refined_chunk"], np.float64)
    ex = E.execute(client, local, chunk, n=8, every=33)
    S = servo.substeps
    sim = ex["joint_trace"][1:].reshape(8, S, 14)
    pred = servo.rollout(st.q, st.v, _commands(layout, chunk), st.tau, samples=list(range(1, S + 1)))
    err = np.abs(np.transpose(pred["q_samples"], (1, 2, 0)) - sim)
    assert np.degrees(err.max()) < 0.3
    rest = servo.rollout(st.q, np.zeros(14), _commands(layout, chunk), np.zeros(14),
                         samples=list(range(1, S + 1)))
    err0 = np.abs(np.transpose(rest["q_samples"], (1, 2, 0)) - sim)
    assert err0.max() > 2.0 * err.max()                       # the velocity / torque estimate matters


@records
@pytest.mark.parametrize("name, seq, margin", [("T40C1807", 28, 0.0), ("T41aC1807", 35, None)])
def test_the_resolved_contact_chunk_does_not_touch_the_bar(name, seq, margin):
    t0 = int(RP._setup(name)["rows"][seq]["t_step"])
    client, local, _ = E.load(RP.RUNS[name][1], t0)
    recorded = np.asarray(RP.npz(name, seq)["refined_chunk"], np.float64)
    before = E.execute(client, local, recorded)
    assert before["contact"] is True and before["true_mm_min"] == 0.0           # what happened
    on = RP.replay(name, seq, overrides=BOTH, obstacle_margin=margin)
    assert on["safe"]
    after = E.execute(client, local, on["chunk"])
    assert after["contact"] is False
    assert after["true_mm_min"] >= min(5.0, after["true_mm_q_now"] - 0.5)     # no closer than it starts
