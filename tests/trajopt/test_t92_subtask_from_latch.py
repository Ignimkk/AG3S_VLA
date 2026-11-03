"""T92 — the 3-stage gate label from the grasp latch phase, behind `serve_safe --subtask-from-latch` (default off).

Task: `AG3S/ag3s/docs/handoff/T92.task.md`. Report: `T92.impl.md`.

What this file fixes:

1. off = as before: the new function is never called (it is replaced by one that raises), the gate input is the
   probe's probabilities, no `subtask_source` key in the summary, no kwarg to `SafePolicy`, no key in the server record.
2. the phase → label table (SEARCHING · LATCHED · CLOSING → pick, HELD → place, PLACED → home; no / unknown phase → None).
3. on: `_subtask_gate_input` = the one-hot of this request's latch phase, the probe record is not read, summary keys.
4. `_scene_fn` still hands AG3S `_subtask_gate_input()` before `set_placed` (same request's phase).
5. `serve_safe`: parse, refusals (--no-subtask-gate · --hsr5 · --hsr-gt-stage · no label path), args → SafePolicy kwarg
   and record meta, the P0 server argv + the flag is accepted.
"""
from __future__ import annotations

import inspect
import pathlib
import shlex
import types

import pytest

from AG3S.trajopt import safe_policy as SP
from AG3S.trajopt import serve_safe
from AG3S.trajopt.grasp_latch import GraspPhase
from AG3S.trajopt.safe_policy import SafePolicy, _subtask_probabilities, latch_subtask_probabilities
from tests.trajopt.test_grasp_evidence import _policy

BASE = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
PROBE_SAYS_HOME = {"p": {"pick": 0.0, "place": 0.0, "home": 1.0}, "argmax": "home", "probe": "kv_L4_v1"}

TABLE = [
    (GraspPhase.SEARCHING, "pick"), (GraspPhase.LATCHED, "pick"), (GraspPhase.CLOSING, "pick"),
    (GraspPhase.HELD, "place"), (GraspPhase.PLACED, "home"),
]


def _onehot(label):
    return {c: (1.0 if c == label else 0.0) for c in ("pick", "place", "home")}


def _parse(argv):
    ap = serve_safe.build_parser()
    return ap, ap.parse_args(BASE + list(argv))


def _check(argv):
    ap, a = _parse(argv)
    serve_safe.reject_bad_flag_combinations(ap, a)
    return a


# ------------------------------------------------------------------------------------------------ 1. off = before
def test_off_never_calls_the_new_function(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("latch_subtask_probabilities was called with the flag off")

    monkeypatch.setattr(SP, "latch_subtask_probabilities", boom)
    sp = _policy()
    assert sp.subtask_from_latch is False
    for phase in GraspPhase:
        sp._latch.phase = phase
        for sub in (PROBE_SAYS_HOME, {"p": {"pick": 0.7, "place": 0.2, "home": 0.1}}, None):
            sp._pending = {"subtask": sub}
            assert sp._subtask_gate_input() == _subtask_probabilities(sub)
    summary = sp._exec_summary()
    assert "subtask_source" not in summary and "subtask_latch" not in summary
    assert sp._last_subtask_latch == {}


def test_off_double_built_without_init_reads_off(monkeypatch):
    """A test double made with `object.__new__` (no `__init__`) reads the class default = off."""
    monkeypatch.setattr(SP, "latch_subtask_probabilities", lambda *_: pytest.fail("called"))
    p = object.__new__(SafePolicy)
    p._latch = types.SimpleNamespace(phase=GraspPhase.HELD)
    p._pending = {"subtask": PROBE_SAYS_HOME}
    assert p._subtask_gate_input() == _subtask_probabilities(PROBE_SAYS_HOME)


def test_off_server_side_has_no_kwarg_and_no_record_key():
    _, a = _parse([])
    assert a.subtask_from_latch is False
    assert serve_safe.subtask_from_latch_kwargs(a) == {}
    assert serve_safe.subtask_from_latch_record_meta(a) == {}
    src = inspect.getsource(serve_safe.main)
    assert "**subtask_from_latch_kwargs(args)" in src               # the only SafePolicy call site passes it
    sig = inspect.signature(SafePolicy.__init__)
    assert sig.parameters["subtask_from_latch"].default is False


def test_off_flag_default_changes_no_other_argument():
    """Adding the flag moves no other parsed value: the namespace differs by exactly the new key."""
    _, base = _parse([])
    _, on = _parse(["--subtask-from-latch"])
    assert {k for k in vars(on) if vars(on)[k] != vars(base).get(k)} == {"subtask_from_latch"}


# ------------------------------------------------------------------------------------------------ 2. the table
@pytest.mark.parametrize("phase,label", TABLE)
def test_phase_label_table(phase, label):
    assert latch_subtask_probabilities(phase) == _onehot(label)
    assert latch_subtask_probabilities(phase.value) == _onehot(label)      # the value string (a record) reads the same


def test_table_covers_every_latch_phase():
    assert {p for p, _ in TABLE} == set(GraspPhase) == set(SP.LATCH_SUBTASK)


@pytest.mark.parametrize("phase", [None, "", "bogus", 7])
def test_no_or_unknown_phase_has_no_label(phase):
    assert latch_subtask_probabilities(phase) is None


# ------------------------------------------------------------------------------------------------ 3. on
@pytest.mark.parametrize("phase,label", TABLE)
def test_on_gate_input_is_the_latch_phase_not_the_probe(phase, label):
    sp = _policy(subtask_from_latch=True)
    assert sp.subtask_from_latch is True
    sp._latch.phase = phase
    for sub in (PROBE_SAYS_HOME, {"p": {"pick": 1.0, "place": 0.0, "home": 0.0}}, None):
        sp._pending = {"subtask": sub}                                    # the probe record is not read
        assert sp._subtask_gate_input() == _onehot(label)
    assert sp._last_subtask_latch == {"source": "latch", "phase": phase.value, "label": label}
    summary = sp._exec_summary()
    assert summary["subtask_source"] == "latch"
    assert summary["subtask_latch"] == {"source": "latch", "phase": phase.value, "label": label}


def test_on_tracks_the_phase_per_request_and_clears_on_reset():
    sp = _policy(subtask_from_latch=True)
    seen = []
    for phase in (GraspPhase.SEARCHING, GraspPhase.LATCHED, GraspPhase.CLOSING, GraspPhase.HELD, GraspPhase.PLACED):
        sp._latch.phase = phase
        seen.append(max(sp._subtask_gate_input().items(), key=lambda kv: kv[1])[0])
    assert seen == ["pick", "pick", "pick", "place", "home"]
    assert sp._last_subtask_latch["label"] == "home"
    sp._pending = {}
    sp._last_subtask_latch = {}                                           # what the per-request reset does
    assert sp._exec_summary()["subtask_latch"] == {}


def test_on_without_latch_phase_gives_no_label():
    p = object.__new__(SafePolicy)
    p.subtask_from_latch = True
    p._latch = types.SimpleNamespace(phase=None)
    p._pending = {"subtask": PROBE_SAYS_HOME}
    assert p._subtask_gate_input() is None
    assert p._last_subtask_latch == {"source": "latch", "phase": None, "label": None}


def test_on_hands_ag3s_the_one_hot_through_scene_fn_order():
    """`_scene_fn` gives AG3S `_subtask_gate_input()` after the latch advanced and before `set_placed`."""
    src = inspect.getsource(SafePolicy._scene_fn)
    assert "set_subtask(self._subtask_gate_input())" in src
    assert src.index("_advance_grasp(") < src.index("set_subtask(") < src.index("set_placed(")


def test_ag3s_debounce_confirms_the_latch_label_as_before():
    """The same `SubtaskDebounce` that confirms the probe label confirms the one-hot (nothing is bypassed)."""
    from AG3S.ag3s.stages.target_grounding import SubtaskDebounce

    deb = SubtaskDebounce()
    for _ in range(10):
        deb.update(latch_subtask_probabilities(GraspPhase.PLACED))
    assert deb.label == "home"


# ------------------------------------------------------------------------------------------------ 5. serve_safe
def test_flag_parses_and_propagates_to_policy_kwargs_and_record():
    a = _check(["--subtask-from-latch"])
    assert a.subtask_from_latch is True
    assert serve_safe.subtask_from_latch_kwargs(a) == {"subtask_from_latch": True}
    assert serve_safe.subtask_from_latch_record_meta(a) == {"source": "latch"}
    serve_safe.resolve_subtask_gate(a)
    assert a.subtask_gate is True                                         # the gate stays the default on


def test_server_args_reach_a_real_safe_policy():
    a = _check(["--subtask-from-latch"])
    sp = _policy(**serve_safe.subtask_from_latch_kwargs(a))
    assert sp.subtask_from_latch is True
    sp._latch.phase = GraspPhase.HELD
    assert sp._subtask_gate_input() == _onehot("place")


def test_start_log_line(caplog):
    _, a = _parse(["--subtask-from-latch"])
    with caplog.at_level("WARNING"):
        serve_safe.announce_subtask_from_latch(a)
    assert any("subtask gate label from grasp latch (T92): ON" in r.getMessage() for r in caplog.records)
    caplog.clear()
    _, off = _parse([])
    with caplog.at_level("WARNING"):
        serve_safe.announce_subtask_from_latch(off)
    assert not caplog.records


@pytest.mark.parametrize("extra", [
    ["--no-subtask-gate"],
    ["--hsr5"],
    ["--hsr5", "/x/probe.npz"],
    ["--hsr-gt-stage"],
    ["--no-safe"],
    ["--no-attention"],
    ["--no-perception"],
])
def test_conflicting_combinations_are_refused(extra, capsys):
    ap, a = _parse(["--subtask-from-latch"] + extra)
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, a)
    assert "--subtask-from-latch" in capsys.readouterr().err


def test_no_subtask_gate_message_names_the_flag(capsys):
    ap, a = _parse(["--subtask-from-latch", "--no-subtask-gate"])
    with pytest.raises(SystemExit):
        serve_safe.reject_subtask_from_latch(ap, a)
    assert "--subtask-from-latch" in capsys.readouterr().err


def test_hsr5_message_names_the_flag(capsys):
    ap, a = _parse(["--subtask-from-latch", "--hsr5"])
    with pytest.raises(SystemExit):
        serve_safe.reject_subtask_from_latch(ap, a)
    assert "--hsr5" in capsys.readouterr().err


def test_explicit_subtask_gate_is_compatible():
    a = _check(["--subtask-from-latch", "--subtask-gate"])
    assert a.subtask_gate is True and a.subtask_from_latch is True


P0_FLAGS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261103T002419+0900_8ab3c7ef7757_82e2790bab11/"
                        "verify/T88/kit/server_flags_p0.txt")


@pytest.mark.skipif(not P0_FLAGS.is_file(), reason="P0 argv file not on this machine")
def test_p0_server_argv_plus_the_flag_is_accepted():
    p0 = shlex.split(P0_FLAGS.read_text())
    off = _check(p0)
    on = _check(p0 + ["--subtask-from-latch"])
    assert off.subtask_from_latch is False and on.subtask_from_latch is True
    assert {k for k in vars(on) if vars(on)[k] != vars(off).get(k)} == {"subtask_from_latch"}
