"""T41 c — `--safe-hold-mode measured`: HOLD stops where the arm **is**, not where it was last told to go.

T40 I C1982: a HOLD started with the arm 7.3° behind its last command; `fixed` latched `q_hold` = that
command, the arm kept moving toward it during the HOLD and touched the hurdle bar (row 178). `measured`
latches `q_hold` = the measured arm joints at the instant the HOLD starts.

What this file pins (unit level; the MuJoCo drift is `tests/sim/test_t41c_hold_drift.py`):

1. `fixed` (default) and `legacy` are **bit-identical** to the pre-T41c `HoldController` (benchmark
   `e13292d`) over a scripted sequence of chunks, HOLD steps, releases and records.
2. `measured` latches the measured arm once and never moves toward the last command, however the
   command and the measurement change during the HOLD.
3. The gripper is the last command (never the measured opening — with a held apple the fingers rest
   at the apple's width, so commanding the measured opening removes the squeeze); measured only when
   no command is known (first step of an episode).
4. Consecutive HOLD chunks keep the first latch; it re-latches only after an executed chunk.
5. The control-frame record: mode, `q_hold`, `source="measured"`, `gripper_source`,
   `entry_command_dev_rad`; `fixed` records carry exactly the old keys.
6. `pi05_infer.py --safe-hold-mode` offers every `HOLD_MODES` value, default `fixed`.
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path
from common.workspace import bind_historical_client_inputs as _bind_legacy_inputs

import ast
import importlib.util
import json
import pathlib
import subprocess

import numpy as np
import pytest

from AG3S.trajopt import wire
from AG3S.trajopt.client import HOLD_MODES, HoldController, action_row

N = wire.ARM_JOINT_DIM
BASE_COMMIT = "e13292d"  # benchmark HEAD before T41 c
BENCHMARK = pathlib.Path(_ag3s_workspace_path('legacy/benchmark'))
PI05_INFER = pathlib.Path(_ag3s_workspace_path('integration/rby1/bringup/pi05_infer.py'))


def _measured(arm, grip):
    return action_row(arm, grip)


@pytest.fixture(scope="module")
def old_client(tmp_path_factory):
    try:
        src = subprocess.run(["git", "-C", str(BENCHMARK), "show", f"{BASE_COMMIT}:trajopt/client.py"],
                             check=True, capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover
        pytest.skip(f"benchmark {BASE_COMMIT} not available: {exc}")
    path = tmp_path_factory.mktemp("t41c") / "client_e13292d.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("_client_e13292d", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _bind_legacy_inputs(mod)
    assert mod.HOLD_MODES == ("fixed", "legacy")
    return mod


def _script(seed: int):
    """A random sequence of chunk decisions and HOLD steps with drifting commands/measurements."""
    rng = np.random.default_rng(seed)
    ops = []
    for _ in range(60):
        executed = bool(rng.random() < 0.35)
        ops.append(("begin", executed, f"r{rng.integers(9)}", [str(k) for k in rng.choice(
            ["collision", "uncertified", "comms"], size=rng.integers(0, 3))]))
        for _ in range(int(rng.integers(1, 9))):
            if executed:
                ops.append(("release",))
                continue
            cmd = (None if rng.random() < 0.1 else
                   {"arm": rng.normal(size=2 * N), "gripper": rng.uniform(0, 1, size=2)})
            meas = rng.normal(size=2 * N + 2)
            ops.append(("hold", cmd, meas))
    return ops


def _run(cls, mode, ops, max_hold_chunks):
    hold = cls(mode=mode, max_hold_chunks=max_hold_chunks)
    out = []
    for op in ops:
        if op[0] == "begin":
            out.append(("begin", json.dumps(hold.begin_chunk(executed=op[1], reason=op[2], kinds=op[3]),
                                            sort_keys=True)))
        elif op[0] == "release":
            hold.release()
            out.append(("release", hold.holding))
        else:
            row = hold.hold_action(last_command=op[1], measured=op[2])
            out.append(("row", row.tobytes(), row.dtype.str, row.shape))
            out.append(("rec", json.dumps(hold.record(measured=op[2]), sort_keys=True),
                        json.dumps(hold.record(), sort_keys=True)))
    out.append(("stats", json.dumps(hold.stats, sort_keys=True), hold.source, hold.age, hold.consecutive))
    return out


# =============================================================================================
# 1. fixed / legacy are bit-identical to the pre-T41c controller
# =============================================================================================


@pytest.mark.parametrize("mode", ["fixed", "legacy"])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("max_hold_chunks", [None, 3])
def test_fixed_and_legacy_are_bit_identical_to_e13292d(old_client, mode, seed, max_hold_chunks):
    ops = _script(seed)
    assert sum(op[0] == "hold" for op in ops) > 50
    assert _run(HoldController, mode, ops, max_hold_chunks) == _run(old_client.HoldController, mode, ops,
                                                                     max_hold_chunks)


def test_the_default_mode_is_still_fixed():
    assert HOLD_MODES[0] == "fixed" and HoldController().mode == "fixed"
    assert set(HOLD_MODES) == {"fixed", "legacy", "measured"}


# =============================================================================================
# 2. measured latches the measurement, not the command
# =============================================================================================


def test_measured_latches_the_measured_arm_and_never_moves_toward_the_command():
    hold = HoldController(mode="measured")
    hold.begin_chunk(executed=False, reason="TO says violated", kinds=["collision"])
    q_meas = np.linspace(-0.4, 0.4, 2 * N)
    command = {"arm": q_meas + np.radians(7.3), "gripper": np.array([0.43, 1.0])}   # C1982-like lag
    first = hold.hold_action(last_command=command, measured=_measured(q_meas, [0.6, 1.0]))

    np.testing.assert_array_equal(first, action_row(q_meas, [0.43, 1.0]))
    assert hold.source == "measured" and hold.gripper_source == "command"
    assert hold.entry_command_dev_rad == pytest.approx(np.radians(7.3))
    # later steps: the "last command" is the HOLD itself or anything else, the arm sags — same target
    for k in range(40):
        drifted = _measured(q_meas - 0.01 * (k + 1), [0.55, 1.0])
        row = hold.hold_action(last_command={"arm": q_meas + 1.0, "gripper": np.zeros(2)}, measured=drifted)
        np.testing.assert_array_equal(row, first)
    assert hold.age == 40
    # the target is the measured pose, not between it and the command
    target_arm = np.concatenate([first[:N], first[N + 1:2 * N + 1]])
    assert np.max(np.abs(target_arm - q_meas)) == 0.0
    assert np.min(np.abs(target_arm - command["arm"])) == pytest.approx(np.radians(7.3))


def test_fixed_on_the_same_input_takes_the_command():
    """The contrast `measured` removes: `fixed` would drive the arm 7.3° on during the HOLD."""
    q_meas = np.zeros(2 * N)
    command = {"arm": q_meas + np.radians(7.3), "gripper": np.array([0.43, 1.0])}
    rows = {}
    for mode in ("fixed", "measured"):
        hold = HoldController(mode=mode)
        hold.begin_chunk(executed=False)
        rows[mode] = hold.hold_action(last_command=command, measured=_measured(q_meas, [0.6, 1.0]))
    np.testing.assert_array_equal(rows["fixed"], action_row(command["arm"], [0.43, 1.0]))
    np.testing.assert_array_equal(rows["measured"], action_row(q_meas, [0.43, 1.0]))


# =============================================================================================
# 3. the gripper does not open
# =============================================================================================


@pytest.mark.parametrize("commanded", [0.0, 0.43, 0.57, 1.0])
def test_measured_gripper_is_the_last_command_not_the_measured_opening(commanded):
    """A held apple keeps the fingers at its width (measured ~0.72 for a 0.57 command in B1807 t=200);
    commanding the measured opening would remove the squeeze — in MuJoCo the apple fell within 2
    control steps that way (impl note T41c). The last command is kept, for both hands."""
    hold = HoldController(mode="measured")
    hold.begin_chunk(executed=False)
    for k in range(20):
        meas_open = 0.72 + 0.01 * k                               # measured opening drifting open
        row = hold.hold_action(last_command={"arm": np.zeros(2 * N),
                                             "gripper": np.array([commanded, 0.9 - 0.1 * (k > 0)])},
                               measured=_measured(np.zeros(2 * N), [meas_open, 1.0]))
        cols = wire.gripper_columns()
        assert row[cols[0]] == commanded and row[cols[1]] == 0.9
    assert hold.record()["gripper_hold"] == [commanded, 0.9]


def test_without_a_known_command_measured_takes_the_measured_opening_once():
    hold = HoldController(mode="measured")
    hold.begin_chunk(executed=False)
    m0 = _measured(np.full(2 * N, 0.4), [1.0, 1.0])
    a = hold.hold_action(last_command=None, measured=m0)
    b = hold.hold_action(last_command=None, measured=_measured(np.zeros(2 * N), [0.2, 0.2]))
    np.testing.assert_array_equal(a, m0)
    np.testing.assert_array_equal(b, m0)
    assert hold.gripper_source == "measured" and hold.entry_command_dev_rad is None


# =============================================================================================
# 4. consecutive HOLDs keep the latch; an executed chunk releases it
# =============================================================================================


def test_consecutive_hold_chunks_keep_the_first_latch_and_re_latch_after_execution():
    hold = HoldController(mode="measured")
    hold.begin_chunk(executed=False, reason="r1", kinds=["collision"])
    a = hold.hold_action(last_command={"arm": np.full(2 * N, 0.3), "gripper": np.array([0.5, 1.0])},
                         measured=_measured(np.full(2 * N, 0.1), [0.7, 1.0]))
    for chunk in range(3):                                       # three more HOLD chunks
        hold.begin_chunk(executed=False, reason=f"r{chunk + 2}", kinds=["uncertified"])
        for _ in range(8):
            b = hold.hold_action(last_command={"arm": np.full(2 * N, 9.0), "gripper": np.zeros(2)},
                                 measured=_measured(np.full(2 * N, -0.2), [0.1, 0.1]))
            np.testing.assert_array_equal(a, b)
    assert hold.consecutive == 4 and hold.stats["holds"] == 1
    rec = hold.record()
    assert rec["entry_reason"] == "r1" and rec["reason"] == "r4" and rec["hold_chunks"] == 4

    hold.begin_chunk(executed=True)
    hold.release()
    assert not hold.holding and hold.record() is None
    assert hold.gripper_source is None and hold.entry_command_dev_rad is None
    hold.begin_chunk(executed=False)
    c = hold.hold_action(last_command={"arm": np.full(2 * N, 0.7), "gripper": np.array([0.2, 1.0])},
                         measured=_measured(np.full(2 * N, 0.65), [0.4, 1.0]))
    np.testing.assert_array_equal(c, action_row(np.full(2 * N, 0.65), [0.2, 1.0]))
    assert hold.stats["holds"] == 2


# =============================================================================================
# 5. the record
# =============================================================================================


def test_the_measured_record_and_the_control_frame(tmp_path):
    from AG3S.ag3s.runtime.frame_record import FrameRecorder

    hold = HoldController(mode="measured")
    hold.begin_chunk(executed=False, reason="TO says violated", kinds=["collision"])
    hold.hold_action(last_command={"arm": np.full(2 * N, 0.25), "gripper": np.array([0.57, 1.0])},
                     measured=action_row(np.full(2 * N, 0.2), [0.72, 1.0]))
    record = hold.record(measured=action_row(np.full(2 * N, 0.19), [0.72, 1.0]))
    assert record["mode"] == "measured" and record["source"] == "measured"
    assert record["gripper_source"] == "command"
    assert record["entry_command_dev_rad"] == pytest.approx(0.05)
    assert record["max_abs_dev_rad"] == pytest.approx(0.01)
    with FrameRecorder(str(tmp_path), manifest={"timing": {}}) as rec:
        rec.control(t_step=1, chunk_seq=1, step_in_chunk=0, now=0.0, field=None,
                    executed=False, qpos=[0.0], hold=record)
    rows = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    control = [r for r in rows if r["kind"] == "control"][0]
    assert control["hold"]["q_hold"] == pytest.approx([0.2] * (2 * N))
    assert control["hold"]["gripper_hold"] == [0.57, 1.0]
    assert control["hold"]["mode"] == "measured"


def test_the_fixed_record_keys_are_unchanged():
    hold = HoldController()
    hold.begin_chunk(executed=False)
    hold.hold_action(last_command={"arm": np.zeros(2 * N), "gripper": np.ones(2)},
                     measured=action_row(np.zeros(2 * N), [1.0, 1.0]))
    assert set(hold.record(measured=action_row(np.zeros(2 * N), [1.0, 1.0]))) == {
        "mode", "source", "entry_reason", "entry_kinds", "reason", "kinds", "q_hold", "gripper_hold",
        "age", "hold_chunks", "max_abs_dev_rad"}


# =============================================================================================
# 6. the client flag
# =============================================================================================


def test_pi05_infer_offers_every_hold_mode_with_fixed_default():
    if not PI05_INFER.exists():  # pragma: no cover
        pytest.skip("pi05_TO_hybrid not present")
    tree = ast.parse(PI05_INFER.read_text())
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--safe-hold-mode"):
            kw = {k.arg: k.value for k in node.keywords}
            found.append((ast.literal_eval(kw["choices"]), ast.literal_eval(kw["default"])))
    assert found == [(HOLD_MODES, "fixed")]
