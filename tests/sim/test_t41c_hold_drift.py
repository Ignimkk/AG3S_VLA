"""T41 c — HOLD in MuJoCo: `measured` stands where the arm is; how far does gravity move it?

The state is the run's own: ``tests/sim/_hold_replay.py`` replays a T40 G E3b record through
``pi05_infer.main()`` reset + the recorded ``applied_ctrl``, bit for bit up to row ``t0`` (asserted at
every row), then runs the client's HOLD branch (``HoldController`` -> ``apply_action`` -> 33 x
``mj_step``) for 64 control steps (4.3 s).

What this file pins:

1. **Mid-carry, apple held** (B ep1807, row 200, apple 0.22 m above the table, gripper command 0.57):
   ``measured`` commands the latched measured arm on every step and the last gripper command; the
   arm settles once by the position servo's gravity sag (no gravity compensation in the model) and
   stops there — 0.56° after 8 steps, 0.58° after 64 (EE 6.2 mm) — instead of accumulating; the
   apple is still in the hand. ``fixed`` from the same state moves the full 3.9° to the last command
   (EE 32 mm).
2. **C1982** (T40 I item 3, row 176 = the HOLD that touched the bar): ``fixed`` touches the hurdle bar
   in HOLD step 2 (the recorded contact row 178); ``measured`` has no contact in 64 steps.

Needs the T40 G records (``outputs/verify/T40/G/runs/E3b``); skipped without them.
CPU only: ``JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa``.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")
mujoco = pytest.importorskip("mujoco")

from tests.sim import _hold_replay as hr  # noqa: E402

N = hr.N
STEPS = 64
B1807 = hr.RUNS / "B" / "E3b_ep1807_s18071"
C1982 = hr.RUNS / "C" / "E3b_ep1982_s19822"


@pytest.fixture(scope="module")
def client():
    if not hr.PI05_INFER.exists():  # pragma: no cover
        pytest.skip("pi05_TO_hybrid not present")
    try:
        import PIL  # noqa: F401
        return hr.load_client()
    except ImportError as exc:  # pragma: no cover - needs .venv-openpi-live
        pytest.skip(f"pi05_infer.py dependencies missing: {exc}")


def _branch(client, run_dir, t0):
    if not (run_dir / "frames.jsonl").exists():
        pytest.skip(f"T40 G record not present: {run_dir}")
    run = hr.load_run(run_dir)
    local = hr.reset(client, run)
    hr.replay_to(client, local, run["control"], t0)
    return {mode: hr.hold(client, local, mode, STEPS) for mode in ("fixed", "measured")}


@pytest.fixture(scope="module")
def carry(client):
    return _branch(client, B1807, 200)


@pytest.fixture(scope="module")
def c1982(client):
    return _branch(client, C1982, 176)


def _deg(rad):
    return math.degrees(rad)


def test_measured_commands_the_latched_measured_pose_and_the_last_gripper_command(carry):
    res = carry["measured"]
    entry = res["arm_entry"]
    target = res["ctrl_rows"]
    np.testing.assert_array_equal(target[:, :N], np.broadcast_to(entry[:N], (STEPS, N)))
    np.testing.assert_array_equal(target[:, N + 1:2 * N + 1], np.broadcast_to(entry[N:], (STEPS, N)))
    last = res["last_command_entry"]["gripper"]
    assert np.all(target[:, N] == last[0]) and np.all(target[:, 2 * N + 1] == last[1])
    assert last[0] == pytest.approx(0.572, abs=0.01)           # the policy's carry command, not open
    rec = res["records"][0]
    assert rec["mode"] == "measured" and rec["source"] == "measured" and rec["gripper_source"] == "command"
    np.testing.assert_array_equal(rec["q_hold"], entry)
    # the entry is the measured state of row 200, and the command was 3.9° ahead of it
    assert _deg(rec["entry_command_dev_rad"]) == pytest.approx(3.88, abs=0.05)


def test_measured_settles_by_the_gravity_sag_once_and_does_not_accumulate(carry):
    rows = carry["measured"]["rows"]
    dev = [_deg(r["arm_dev_from_entry_rad"]) for r in rows]
    ee = [r["ee_disp_mm"] for r in rows]
    print(f"\n[T41c] measured HOLD B1807 t0=200: arm dev step 1/8/64 = {dev[0]:.3f}/{dev[7]:.3f}/{dev[63]:.3f} deg, "
          f"EE {ee[0]:.1f}/{ee[7]:.1f}/{ee[63]:.1f} mm, max {max(dev):.3f} deg / {max(ee):.1f} mm")
    assert max(dev) < 0.8 and max(ee) < 10.0          # sag 0.56-0.60 deg, EE 6-8 mm (4 carry states)
    assert abs(dev[63] - dev[7]) < 0.05               # settled after one chunk: no per-step creep
    assert abs(ee[63] - ee[7]) < 1.0


def test_fixed_from_the_same_state_moves_to_the_last_command(carry):
    fixed, measured = carry["fixed"]["rows"], carry["measured"]["rows"]
    gap = _deg(np.max(np.abs(carry["fixed"]["last_command_entry"]["arm"] - carry["fixed"]["arm_entry"])))
    assert _deg(fixed[63]["arm_dev_from_entry_rad"]) == pytest.approx(gap, rel=0.05)
    assert fixed[63]["ee_disp_mm"] > 4 * measured[63]["ee_disp_mm"]


def test_the_apple_stays_in_the_hand_and_the_gripper_does_not_open(carry):
    rows = carry["measured"]["rows"]
    assert all(r["fruit_z_m"] > 1.0 for r in rows)                       # table top 0.82
    assert max(r["fruit_in_hand_shift_mm"] for r in rows) < 15.0
    g0 = rows[0]["grip_meas_norm"]
    assert all(r["grip_meas_norm"] <= g0 + 0.005 for r in rows)          # fingers not opening


def test_c1982_fixed_touches_the_bar_measured_does_not(c1982):
    fixed, measured = c1982["fixed"], c1982["measured"]
    print(f"\n[T41c] C1982 t0=176: fixed contact={fixed['first_contact_step_substep']} "
          f"{fixed['contact_pairs']}; measured contact={measured['collided']} "
          f"min clearance {measured['min_robot_clearance_m'] * 1e3:.1f} mm")
    assert _deg(np.max(np.abs(fixed["last_command_entry"]["arm"] - fixed["arm_entry"]))) == pytest.approx(
        7.27, abs=0.05)                                                   # T40 I: 7.274 deg
    assert fixed["collided"] and fixed["first_contact_step_substep"][0] == 2   # row 178 as recorded
    assert any("hurdle_0_collision_bar" in name for pair in fixed["contact_pairs"] for name in pair)
    assert not measured["collided"] and measured["min_robot_clearance_m"] > 0.0
