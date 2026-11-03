"""T91c — the T91b phase machine as the source of the 3-stage gate label, behind `serve_safe --subtask-from-phase-machine`
(default off).

Task: `AG3S/ag3s/docs/handoff/T91c.task.md`. Report: `T91c.impl.md`.

What this file fixes:

1. off = as before: `PhaseGate` is never built (it is replaced by something that raises), the gate input is the probe's
   probabilities, no `subtask_source` / `phase_machine` key in the summary, no kwarg to `SafePolicy`, no key in the server
   record, no other parsed argument moves.
2. the 5 -> 3 table (approach · grasp -> pick, carry · place -> place, retreat -> home).
3. `PhaseGate`: an empty hand that closes never gives place; a closed hand with a surface between the fingertips does;
   a missing wrist depth / opening is handled; `reset()` between runs.
4. `SafePolicy` on: the gate input is the one-hot of the machine, built from the request (opening · hand site · previous
   request's ATS target / destination · chunk gripper column · wrist depth), stepped once per request; an error is a missing
   label, not a dead request; summary keys; `reset()` clears it.
5. `serve_safe`: parse, the four refusals (--subtask-from-latch · --hsr5 · --hsr-gt-stage · --no-subtask-gate) and the
   no-label-path refusals, args -> SafePolicy kwarg and record meta, the P0 argv + the flag is accepted.
"""
from __future__ import annotations

import inspect
import json
import pathlib
import shlex
import types

import numpy as np
import pytest

from AG3S.trajopt import phase_gate as PG
from AG3S.trajopt import safe_policy as SP
from AG3S.trajopt import serve_safe
from AG3S.trajopt.hsr5 import subtask3_probabilities
from AG3S.trajopt.phase_gate import PhaseGate, finger_gap_from_depth_mm, phase_config
from AG3S.trajopt.phase_machine import PhaseConfig
from AG3S.trajopt.safe_policy import SafePolicy, _subtask_probabilities
from tests.trajopt.test_grasp_evidence import _policy

BASE = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
PROBE_SAYS_HOME = {"p": {"pick": 0.0, "place": 0.0, "home": 1.0}, "argmax": "home", "probe": "kv_L4_v1"}

TABLE = [("approach", "pick"), ("grasp", "pick"), ("carry", "place"), ("place", "place"), ("retreat", "home")]

TABLE_MM = 500          # a far surface (the table / the room): outside the finger reach window
FINGER_MM = 60          # a surface inside the window [30, 90] mm between the fingertips


def _onehot(label):
    return {c: (1.0 if c == label else 0.0) for c in ("pick", "place", "home")}


def _label(p):
    return None if p is None else max(p, key=p.get)


def _depth(mm):
    return np.full((480, 640), mm, np.uint16)


def _parse(argv):
    ap = serve_safe.build_parser()
    return ap, ap.parse_args(BASE + list(argv))


def _check(argv):
    ap, a = _parse(argv)
    serve_safe.reject_bad_flag_combinations(ap, a)
    return a


TARGET = np.array([0.50, 0.20, 0.90])
FAR = np.array([0.30, -0.30, 1.10])


def _run(gate, steps):
    """`steps` = list of dicts of `PhaseGate.step` kwargs; returns the list of `(label, stage)`."""
    out = []
    for kw in steps:
        p, rec = gate.step(**kw)
        out.append((_label(p), rec.get("stage")))
    return out


def _empty_hand_close(depth_mm):
    """Open x3, then the hand closes on nothing far from any target (the VLA keeps asking to close) and moves on."""
    steps = []
    for _ in range(3):
        steps.append(dict(hand="left", gripper_raw=1.0, ee_pos=FAR, target_pos=TARGET, target_fresh=True,
                          chunk_gripper=np.ones(16), wrist_depth_mm=depth_mm))
    for k, g in enumerate((0.7, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3)):
        steps.append(dict(hand="left", gripper_raw=g, ee_pos=FAR + np.array([0.01 * k, 0.0, 0.0]), target_pos=TARGET,
                          target_fresh=True, chunk_gripper=np.zeros(16), wrist_depth_mm=depth_mm))
    return steps


def _grasp_and_lift(depth_mm):
    """Open x3, approach the target, close on it (gap evidence from `depth_mm`), lift the hand 40 mm."""
    steps = []
    for _ in range(3):
        steps.append(dict(hand="left", gripper_raw=1.0, ee_pos=TARGET + np.array([0, 0, 0.15]), target_pos=TARGET,
                          target_fresh=True, chunk_gripper=np.ones(16), wrist_depth_mm=None))
    steps.append(dict(hand="left", gripper_raw=1.0, ee_pos=TARGET + np.array([0, 0, 0.02]), target_pos=TARGET,
                      target_fresh=True, chunk_gripper=np.zeros(16), wrist_depth_mm=None))
    for k, g in enumerate((0.6, 0.35, 0.35, 0.35, 0.35, 0.35)):
        steps.append(dict(hand="left", gripper_raw=g, ee_pos=TARGET + np.array([0, 0, 0.02 + 0.01 * k]),
                          target_pos=TARGET + np.array([0, 0, 0.01 * k]), target_fresh=True,
                          chunk_gripper=np.zeros(16), wrist_depth_mm=depth_mm))
    return steps


# ------------------------------------------------------------------------------------------------ 1. off = before
def test_off_never_builds_the_gate_and_reads_the_probe(monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("PhaseGate was built with the flag off")

    monkeypatch.setattr(PG, "PhaseGate", Boom)
    sp = _policy()
    assert sp.subtask_from_phase_machine is False and sp._phase_gate is None
    monkeypatch.setattr(SafePolicy, "_phase_machine_step", lambda *_: pytest.fail("called with the flag off"))
    for sub in (PROBE_SAYS_HOME, {"p": {"pick": 0.7, "place": 0.2, "home": 0.1}}, None):
        sp._pending = {"subtask": sub}
        assert sp._subtask_gate_input() == _subtask_probabilities(sub)
    summary = sp._exec_summary()
    assert "subtask_source" not in summary and "phase_machine" not in summary
    assert sp._last_phase_machine == {} and sp._phase_gate is None


def test_off_summary_is_the_same_with_the_argument_written_out():
    """Passing `subtask_from_phase_machine=False` explicitly changes nothing: same summary keys as the default call."""
    a, b = _policy(), _policy(subtask_from_phase_machine=False)
    assert list(a._exec_summary()) == list(b._exec_summary())
    assert json.dumps(a._exec_summary(), default=str, sort_keys=True) == json.dumps(b._exec_summary(), default=str, sort_keys=True)


def test_off_does_not_touch_the_t92_record_either():
    sp = _policy(subtask_from_latch=True)
    assert sp.subtask_from_phase_machine is False
    summary = sp._exec_summary()
    assert summary["subtask_source"] == "latch" and "phase_machine" not in summary


def test_off_double_built_without_init_reads_off(monkeypatch):
    """A test double made with `object.__new__` (no `__init__`) reads the class default = off."""
    p = object.__new__(SafePolicy)
    monkeypatch.setattr(SafePolicy, "_phase_machine_step", lambda *_: pytest.fail("called"))
    p._pending = {"subtask": PROBE_SAYS_HOME}
    assert p.subtask_from_phase_machine is False
    assert p._subtask_gate_input() == _subtask_probabilities(PROBE_SAYS_HOME)


def test_off_server_side_has_no_kwarg_and_no_record_key():
    _, a = _parse([])
    assert a.subtask_from_phase_machine is False
    assert serve_safe.subtask_from_phase_machine_kwargs(a) == {}
    assert serve_safe.subtask_from_phase_machine_record_meta(a) == {}
    src = inspect.getsource(serve_safe.main)
    assert "**subtask_from_phase_machine_kwargs(args)" in src             # the only SafePolicy call site passes it
    assert inspect.signature(SafePolicy.__init__).parameters["subtask_from_phase_machine"].default is False


def test_off_flag_default_changes_no_other_argument():
    _, base = _parse([])
    _, on = _parse(["--subtask-from-phase-machine"])
    assert {k for k in vars(on) if vars(on)[k] != vars(base).get(k)} == {"subtask_from_phase_machine"}


# ------------------------------------------------------------------------------------------------ 2. the table
@pytest.mark.parametrize("stage,label", TABLE)
def test_five_to_three_mapping(stage, label):
    assert subtask3_probabilities(stage) == _onehot(label)


def test_gate_gives_the_mapped_one_hot_for_every_stage(monkeypatch):
    gate = PhaseGate()
    for stage, label in TABLE:
        gate.reset()
        gate.machine = None
        monkeypatch.setattr(PG, "PhaseMachine", lambda cfg, s=stage: types.SimpleNamespace(step=lambda obs: (s, {"open_norm": 1.0})))
        p, rec = gate.step(hand="left", gripper_raw=1.0)
        assert p == _onehot(label) and rec["stage"] == stage and rec["label"] == label


# ------------------------------------------------------------------------------------------------ 3. PhaseGate
def test_config_is_the_t91b_depth_arm():
    c = phase_config("right", 15)
    assert c.chunk_gripper_column == 15 and c.finger_gap_mode == "on" and c.finger_gap_source == "depth"
    assert (c.gap_tau, c.gap_m) == (0.3, 1)
    base = PhaseConfig()
    assert c.open_thr == base.open_thr and c.d_near == base.d_near and c.bootstrap_hand_mm == base.bootstrap_hand_mm
    with pytest.raises(ValueError):
        phase_config("middle", 7)


def test_t91_thresholds_file_matches_the_defaults():
    p = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development")
    files = sorted(p.glob("*/verify/T91/thresholds.json"))
    if not files:
        pytest.skip("thresholds.json not on this machine")
    th = json.loads(files[-1].read_text())
    params = th["params"]
    cfg = phase_config("left", 7)                                             # what the server's gate runs with
    assert len(params) >= 8
    for k, v in params.items():
        assert getattr(cfg, k) == v, k


def test_wrist_depth_surface_fraction():
    assert finger_gap_from_depth_mm(_depth(FINGER_MM)) == pytest.approx(1.0)
    assert finger_gap_from_depth_mm(_depth(TABLE_MM)) == pytest.approx(0.0)
    assert finger_gap_from_depth_mm(_depth(0)) == pytest.approx(0.0)           # invalid pixels are not a surface
    assert finger_gap_from_depth_mm(None) is None
    assert finger_gap_from_depth_mm(np.zeros((0, 0), np.uint16)) is None
    assert finger_gap_from_depth_mm(np.zeros((4, 4, 3), np.uint16)) is None


@pytest.mark.parametrize("depth_mm", [_depth(TABLE_MM), None])
def test_empty_hand_close_never_gives_place(depth_mm):
    """A hand that closes on nothing (far from the target, the table / nothing between the fingertips) is not carrying."""
    out = _run(PhaseGate(), _empty_hand_close(depth_mm))
    stages = [s for _, s in out]
    assert not set(stages) & {"carry", "place"}, stages
    assert "place" not in [lab for lab, _ in out]
    assert out[-1][0] == "pick"


def test_empty_hand_close_even_with_the_hand_moving_a_long_way():
    steps = _empty_hand_close(_depth(TABLE_MM))
    for k, s in enumerate(steps[3:]):
        s["ee_pos"] = FAR + np.array([0.05 * k, 0.0, 0.0])                    # 0.35 m of travel after closing
    out = _run(PhaseGate(), steps)
    assert "place" not in [lab for lab, _ in out]


def test_surface_between_the_fingertips_and_closed_gives_place():
    out = _run(PhaseGate(), _grasp_and_lift(_depth(FINGER_MM)))
    labels = [lab for lab, _ in out]
    assert labels[:4] == ["pick"] * 4
    assert labels[-1] == "place" and out[-1][1] == "carry"


def test_same_closing_without_the_surface_stays_pick():
    out = _run(PhaseGate(), _grasp_and_lift(_depth(TABLE_MM)))
    assert "place" not in [lab for lab, _ in out]


def test_surface_but_open_hand_is_not_carry():
    steps = _grasp_and_lift(_depth(FINGER_MM))
    for s in steps:
        s["gripper_raw"] = 1.0                                                # the hand never closes
    out = _run(PhaseGate(), steps)
    assert "place" not in [lab for lab, _ in out]


def test_record_carries_the_evidence_and_the_inputs():
    gate = PhaseGate()
    p, rec = gate.step(hand="left", gripper_raw=1.0, ee_pos=FAR, wrist_depth_mm=_depth(FINGER_MM))
    assert rec["source"] == "phase_machine" and rec["hand"] == "left" and rec["n"] == 0
    assert rec["inputs"] == {"ee": True, "target": False, "destination": False, "chunk": False, "wrist_depth": True}
    assert rec["stage"] == "approach" and rec["label"] == "pick"
    ev = rec["evidence"]
    assert ev["gap_depth"] == pytest.approx(1.0) and ev["open_norm"] == pytest.approx(1.0) and "comotion" in ev
    json.dumps(rec)                                                           # JSON-able as it is


def test_no_opening_means_no_label_and_the_machine_is_not_stepped():
    gate = PhaseGate()
    for bad in (None, float("nan"), "x"):
        p, rec = gate.step(hand="left", gripper_raw=bad, ee_pos=FAR)
        assert p is None and rec["label"] is None and rec["stage"] is None and "reason" in rec
    assert gate.machine.n == 0
    p, rec = gate.step(hand="left", gripper_raw=1.0)
    assert _label(p) == "pick" and gate.machine.n == 1


def test_unknown_hand_is_an_error_not_a_label():
    with pytest.raises(ValueError):
        PhaseGate().step(hand="middle", gripper_raw=1.0)


def test_reset_between_runs():
    gate = PhaseGate()
    _run(gate, _grasp_and_lift(_depth(FINGER_MM)))
    assert gate.n > 0 and gate.machine.ever_carry
    gate.reset()
    assert gate.machine is None and gate.n == 0 and gate.hand is None
    # a new run starts from "approach" and re-learns the opening reference from its own first requests
    p, rec = gate.step(hand="left", gripper_raw=0.5)
    assert rec["n"] == 0 and rec["stage"] == "approach" and rec["evidence"]["open_norm"] == pytest.approx(1.0)
    # without the reset the old reference (1.0) would have read 0.5
    other = PhaseGate()
    other.step(hand="left", gripper_raw=1.0)
    _, rec2 = other.step(hand="left", gripper_raw=0.5)
    assert rec2["evidence"]["open_norm"] == pytest.approx(0.5)


def test_reset_gives_the_same_labels_as_a_fresh_gate():
    steps = _grasp_and_lift(_depth(FINGER_MM))
    fresh = _run(PhaseGate(), steps)
    gate = PhaseGate()
    _run(gate, _empty_hand_close(_depth(TABLE_MM)))
    gate.reset()
    assert _run(gate, steps) == fresh


def test_switching_the_grasping_hand_starts_a_new_machine():
    gate = PhaseGate()
    gate.step(hand="left", gripper_raw=1.0)
    first = gate.machine
    gate.step(hand="right", gripper_raw=1.0)
    assert gate.machine is not first and gate.machine.cfg.chunk_gripper_column == 15


# ------------------------------------------------------------------------------------------------ 4. SafePolicy on
def _ready(monkeypatch, ee=FAR, manip=None, dest=None):
    """A minimal `SafePolicy` double with the flag on and the pieces `_phase_machine_step` reads."""
    p = object.__new__(SafePolicy)
    p.subtask_from_phase_machine = True
    p._phase_gate = None
    p._last_phase_machine = {}
    p.gripper_columns = (7, 15)
    p.hold_links = {"left": "ee_left", "right": "ee_right"}
    p.grasp_links = {"left": ("ee_finger_l1", ()), "right": ("ee_finger_r1", ())}
    p.constraint_robot_model = object()
    p.ag3s = types.SimpleNamespace(manipulated=manip, destination=dest)
    seen = []

    def fake_site(model, q, link):
        seen.append((np.asarray(q).copy(), link))
        return np.asarray(ee, np.float64)

    monkeypatch.setattr(SP, "hold_site_position", fake_site)
    p._sites = seen
    return p


def _request(p, *, left=1.0, right=1.0, chunk_gripper=1.0, depth=None, manipulators=("left",)):
    state = np.zeros(16)
    state[7], state[15] = left, right
    chunk = np.zeros((50, 16))
    chunk[:, 7] = chunk[:, 15] = chunk_gripper
    scene = {"active_manipulators": list(manipulators), "cameras": ["zed_left", "wrist_cam_l"],
             "robot_state/zed_left": np.arange(12.0), "stamp/zed_left": 1.0,
             "robot_state/wrist_cam_l": np.arange(12.0) + 100.0, "stamp/wrist_cam_l": 2.0}
    if depth is not None:
        scene["depth/wrist_cam_l"] = depth
    p._pending = {"scene": scene, "state": state, "chunk": chunk}


def _manip(centroid, state="visible"):
    return types.SimpleNamespace(centroid=np.asarray(centroid, np.float64), state=state)


def test_on_builds_the_gate_input_from_the_request(monkeypatch):
    p = _ready(monkeypatch, manip=_manip(TARGET), dest=types.SimpleNamespace(centroid=np.array([0.0, 0.6, 0.9])))
    _request(p, depth=_depth(FINGER_MM))
    out = p._subtask_gate_input()
    assert out == _onehot("pick")
    rec = p._last_phase_machine
    assert rec["source"] == "phase_machine" and rec["hand"] == "left" and rec["stage"] == "approach" and rec["label"] == "pick"
    assert rec["inputs"] == {"ee": True, "target": True, "destination": True, "chunk": True, "wrist_depth": True}
    assert rec["evidence"]["gap_depth"] == pytest.approx(1.0) and rec["evidence"]["fresh"] is True
    # the hand site is read at the NEWEST camera's robot state, on the hold link of the grasping hand
    (q, link), = p._sites
    assert link == "ee_left" and np.array_equal(q, np.arange(12.0) + 100.0)
    assert p._pending["t91c"][0] == out


def test_on_steps_once_per_request(monkeypatch):
    p = _ready(monkeypatch)
    _request(p)
    first = p._subtask_gate_input()
    n = p._phase_gate.machine.n
    assert p._subtask_gate_input() == first and p._subtask_gate_input() == first     # `_scene_fn` may run again
    assert p._phase_gate.machine.n == n == 1
    _request(p)                                                                      # a new request
    p._subtask_gate_input()
    assert p._phase_gate.machine.n == 2


def test_on_right_hand_reads_its_own_columns_and_wrist_camera(monkeypatch):
    p = _ready(monkeypatch)
    _request(p, left=0.2, right=1.0, manipulators=("right",))
    p._subtask_gate_input()
    ev = p._last_phase_machine
    assert ev["hand"] == "right" and ev["evidence"]["open_norm"] == pytest.approx(1.0)
    assert ev["inputs"]["wrist_depth"] is False                                      # only wrist_cam_l was sent
    assert p._sites[0][1] == "ee_right"


def test_on_target_not_visible_is_not_fresh_and_missing_inputs_are_none(monkeypatch):
    p = _ready(monkeypatch, manip=_manip(TARGET, state="occluded"))
    _request(p)
    p._pending["scene"].pop("cameras")
    p._subtask_gate_input()
    rec = p._last_phase_machine
    assert rec["evidence"]["fresh"] is False and rec["inputs"]["ee"] is False and rec["inputs"]["target"] is True
    assert p._sites == []                                                            # no camera -> no FK


def test_on_chunk_without_the_gripper_column_gives_no_intent(monkeypatch):
    p = _ready(monkeypatch)
    _request(p)
    p._pending["chunk"] = np.zeros((50, 7))
    p._subtask_gate_input()
    assert p._last_phase_machine["inputs"]["chunk"] is False


def test_on_error_is_a_missing_label_not_a_dead_request(monkeypatch):
    p = _ready(monkeypatch)
    _request(p)

    def boom(*a, **k):
        raise RuntimeError("fk exploded")

    monkeypatch.setattr(SP, "hold_site_position", boom)
    assert p._subtask_gate_input() is None
    rec = p._last_phase_machine
    assert rec["label"] is None and rec["stage"] is None and "RuntimeError" in rec["error"]
    assert p._pending["t91c"][0] is None                                             # cached: not retried within the request


def test_on_no_opening_gives_no_label(monkeypatch):
    p = _ready(monkeypatch)
    _request(p)
    p._pending["state"] = None
    assert p._subtask_gate_input() is None
    assert p._last_phase_machine["label"] is None and "reason" in p._last_phase_machine


def test_on_empty_hand_close_does_not_give_place_through_the_policy(monkeypatch):
    """The whole path: request dicts -> `_subtask_gate_input` -> label, over a run that closes on nothing."""
    p = _ready(monkeypatch)
    labels = []
    for k, st in enumerate(_empty_hand_close(_depth(TABLE_MM))):
        monkeypatch.setattr(SP, "hold_site_position", lambda m, q, l, ee=st["ee_pos"]: ee)
        p.ag3s.manipulated = _manip(TARGET)
        _request(p, left=st["gripper_raw"], chunk_gripper=float(st["chunk_gripper"][0]), depth=_depth(TABLE_MM))
        labels.append(_label(p._subtask_gate_input()))
    assert "place" not in labels and labels[-1] == "pick"


def test_on_grasp_and_lift_gives_place_through_the_policy(monkeypatch):
    p = _ready(monkeypatch)
    labels = []
    for st in _grasp_and_lift(_depth(FINGER_MM)):
        monkeypatch.setattr(SP, "hold_site_position", lambda m, q, l, ee=st["ee_pos"]: ee)
        p.ag3s.manipulated = _manip(st["target_pos"])
        _request(p, left=st["gripper_raw"], chunk_gripper=float(st["chunk_gripper"][0]), depth=st["wrist_depth_mm"])
        labels.append(_label(p._subtask_gate_input()))
    assert labels[-1] == "place" and labels[0] == "pick"


def test_on_matches_the_standalone_gate_on_the_same_inputs(monkeypatch):
    """The server path and the replay path (`PhaseGate` called directly) build the same observation."""
    steps = _grasp_and_lift(_depth(FINGER_MM))
    direct = _run(PhaseGate(), steps)
    p = _ready(monkeypatch)
    via = []
    for st in steps:
        monkeypatch.setattr(SP, "hold_site_position", lambda m, q, l, ee=st["ee_pos"]: ee)
        p.ag3s.manipulated = _manip(st["target_pos"])
        _request(p, left=st["gripper_raw"], chunk_gripper=float(st["chunk_gripper"][0]), depth=st["wrist_depth_mm"])
        lab = _label(p._subtask_gate_input())
        via.append((lab, p._last_phase_machine["stage"]))
    assert via == direct


def test_real_policy_summary_keys_and_reset():
    sp = _policy(subtask_from_phase_machine=True)
    assert sp.subtask_from_phase_machine is True and sp._phase_gate is None
    sp._last_phase_machine = {"source": "phase_machine", "stage": "carry", "label": "place"}
    summary = sp._exec_summary()
    assert summary["subtask_source"] == "phase_machine"
    assert summary["phase_machine"] == {"source": "phase_machine", "stage": "carry", "label": "place"}
    sp._phase_gate = PhaseGate()
    sp._phase_gate.step(hand="left", gripper_raw=1.0)
    sp._phase_gate.step(hand="left", gripper_raw=1.0)
    sp.reset()
    assert sp._phase_gate.machine is None and sp._phase_gate.n == 0 and sp._last_phase_machine == {}


def test_real_policy_refuses_two_label_sources():
    with pytest.raises(ValueError, match="give one"):
        _policy(subtask_from_phase_machine=True, subtask_from_latch=True)


def test_gate_input_is_handed_to_ag3s_in_scene_fn_order():
    src = inspect.getsource(SafePolicy._scene_fn)
    assert "set_subtask(self._subtask_gate_input())" in src
    assert src.index("set_subtask(") < src.index("set_placed(")


def test_ag3s_debounce_confirms_the_gate_label_as_before():
    from AG3S.ag3s.stages.target_grounding import SubtaskDebounce

    deb = SubtaskDebounce()
    for _ in range(10):
        deb.update(subtask3_probabilities("retreat"))
    assert deb.label == "home"


# ------------------------------------------------------------------------------------------------ 5. serve_safe
def test_flag_parses_and_propagates_to_policy_kwargs_and_record():
    a = _check(["--subtask-from-phase-machine"])
    assert a.subtask_from_phase_machine is True
    assert serve_safe.subtask_from_phase_machine_kwargs(a) == {"subtask_from_phase_machine": True}
    assert serve_safe.subtask_from_phase_machine_record_meta(a) == {"source": "phase_machine"}
    serve_safe.resolve_subtask_gate(a)
    assert a.subtask_gate is True                                                    # the gate stays the default on


def test_server_args_reach_a_real_safe_policy():
    a = _check(["--subtask-from-phase-machine"])
    sp = _policy(**serve_safe.subtask_from_phase_machine_kwargs(a))
    assert sp.subtask_from_phase_machine is True


def test_start_log_line(caplog):
    _, a = _parse(["--subtask-from-phase-machine"])
    with caplog.at_level("WARNING"):
        serve_safe.announce_subtask_from_phase_machine(a)
    assert any("(T91c): ON" in r.getMessage() for r in caplog.records)
    caplog.clear()
    _, off = _parse([])
    with caplog.at_level("WARNING"):
        serve_safe.announce_subtask_from_phase_machine(off)
    assert not caplog.records


@pytest.mark.parametrize("extra,named", [
    (["--subtask-from-latch"], "--subtask-from-latch"),
    (["--hsr5"], "--hsr5"),
    (["--hsr5", "/x/probe.npz"], "--hsr5"),
    (["--hsr-gt-stage"], "--hsr5"),
    (["--no-subtask-gate"], "--no-subtask-gate"),
    (["--no-safe"], "--subtask-from-phase-machine"),
    (["--no-attention"], "--subtask-from-phase-machine"),
    (["--no-perception"], "--subtask-from-phase-machine"),
])
def test_conflicting_combinations_are_refused(extra, named, capsys):
    ap, a = _parse(["--subtask-from-phase-machine"] + extra)
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, a)
    err = capsys.readouterr().err
    assert "--subtask-from-phase-machine" in err and named in err


def test_the_four_source_refusals_come_from_the_new_check(capsys):
    for extra in (["--subtask-from-latch"], ["--hsr5"], ["--hsr-gt-stage"], ["--no-subtask-gate"]):
        ap, a = _parse(["--subtask-from-phase-machine"] + extra)
        with pytest.raises(SystemExit):
            serve_safe.reject_subtask_from_phase_machine(ap, a)
        assert "--subtask-from-phase-machine" in capsys.readouterr().err


def test_explicit_subtask_gate_is_compatible():
    a = _check(["--subtask-from-phase-machine", "--subtask-gate"])
    assert a.subtask_gate is True and a.subtask_from_phase_machine is True


P0_FLAGS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261103T002419+0900_8ab3c7ef7757_82e2790bab11/"
                        "verify/T88/kit/server_flags_p0.txt")


@pytest.mark.skipif(not P0_FLAGS.is_file(), reason="P0 argv file not on this machine")
def test_p0_server_argv_plus_the_flag_is_accepted():
    p0 = shlex.split(P0_FLAGS.read_text())
    off = _check(p0)
    on = _check(p0 + ["--subtask-from-phase-machine"])
    assert off.subtask_from_phase_machine is False and on.subtask_from_phase_machine is True
    assert {k for k in vars(on) if vars(on)[k] != vars(off).get(k)} == {"subtask_from_phase_machine"}


@pytest.mark.skipif(not P0_FLAGS.is_file(), reason="P0 argv file not on this machine")
def test_p0m_argv_with_the_latch_swapped_for_the_phase_machine_is_accepted():
    """P0m = P0 + --subtask-from-latch (T92). T91c's closed-loop replaces that one flag; the latch flag together is refused."""
    p0 = shlex.split(P0_FLAGS.read_text())
    p0m = _check(p0 + ["--subtask-from-latch"])
    t91c = _check(p0 + ["--subtask-from-phase-machine"])
    assert p0m.subtask_from_latch and not p0m.subtask_from_phase_machine
    assert t91c.subtask_from_phase_machine and not t91c.subtask_from_latch
    ap, a = _parse(p0 + ["--subtask-from-latch", "--subtask-from-phase-machine"])
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, a)


P0M_FLAGS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261103T133225+0900_ed0145a692ad_6e6100ee51f4/"
                         "verify/T92/kit/server_flags_p0m.txt")


@pytest.mark.skipif(not P0M_FLAGS.is_file(), reason="T92 P0m argv file not on this machine")
def test_the_recorded_t92_p0m_argv_with_the_flag_swapped_is_accepted():
    """The exact T92 server argv (`... --subtask-from-latch`) with that one flag replaced by T91c's."""
    toks = shlex.split(P0M_FLAGS.read_text())
    assert toks.count("--subtask-from-latch") == 1
    p0m = _check(toks)
    swapped = _check([t for t in toks if t != "--subtask-from-latch"] + ["--subtask-from-phase-machine"])
    assert p0m.subtask_from_latch and swapped.subtask_from_phase_machine and not swapped.subtask_from_latch
    assert {k for k in vars(swapped) if vars(swapped)[k] != vars(p0m).get(k)} == {"subtask_from_latch",
                                                                                   "subtask_from_phase_machine"}
