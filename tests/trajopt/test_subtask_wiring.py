"""SUBTASK-c Part B — the subtask label reaches AG3S, and the server flag reaches the config.

1. `SafePolicy.infer` keeps `result["subtask"]` for `_scene_fn`, which hands its `p` to
   `ag3s.set_subtask` (None when absent / malformed) next to `set_grasp_active`.
2. `set_placed` is True exactly in latch phase PLACED — not after a revoked false attach (T34
   leaves the latch LATCHED), not while closing / held (B3).
3. `serve_safe --subtask-gate` / `--subtask-probe` → `clustering.subtask_gate` /
   `subtask_probe_path`; impossible combinations are refused. SUBTASK-g: the gate is on by default
   and `main` always writes it into the config (on or off); `--no-subtask-gate` turns it off
   (`tests/trajopt/test_subtask_gate_default.py`).
4. the constraint record carries `metrics["subtask"]`.
"""

from __future__ import annotations

import inspect
import types

import numpy as np
import pytest

from AG3S.trajopt import serve_safe, wire
from AG3S.trajopt.grasp_latch import GraspPhase
from AG3S.trajopt.safe_policy import _subtask_probabilities

P = {"pick": 0.1, "place": 0.85, "home": 0.05}
SUBTASK = {"p": P, "argmax": "place", "probe": "kv_L4_v1"}


def _policy(chunk_extra=None):
    from tests.trajopt.test_safe_policy import _StubPolicy, _safe_policy

    class _WithSubtask(_StubPolicy):
        def infer(self, obs, **kw):
            out = super().infer(obs, **kw)
            if chunk_extra is not None:
                out["subtask"] = chunk_extra
            return out

    return _safe_policy(_WithSubtask())


def _spy(sp, monkeypatch):
    q = np.zeros(sp.constraint_robot_model.nq)
    obs = [types.SimpleNamespace(timestamp=1.0, robot_state=q, camera_id="zed_left")]
    monkeypatch.setattr(wire, "unpack_camera_observations", lambda scene, **kw: obs)
    calls = {"subtask": [], "placed": []}
    real_subtask, real_placed = sp.ag3s.set_subtask, sp.ag3s.set_placed
    monkeypatch.setattr(sp.ag3s, "set_subtask",
                        lambda p: (calls["subtask"].append(p), real_subtask(p))[1])
    monkeypatch.setattr(sp.ag3s, "set_placed",
                        lambda v: (calls["placed"].append(v), real_placed(v))[1])

    class _Stop(Exception):
        pass

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(sp.ag3s, "process_multi_debug", stop)
    monkeypatch.setattr(sp, "_advance_grasp", lambda *a, **k: None)
    return calls, _Stop


def test_subtask_probabilities():
    assert _subtask_probabilities(SUBTASK) == P
    for bad in (None, {}, {"p": None}, {"p": {}}, {"p": {"pick": "x"}}, "place", {"argmax": "x"}):
        assert _subtask_probabilities(bad) is None


@pytest.mark.parametrize("subtask,expected", [(SUBTASK, P), (None, None)])
def test_infer_keeps_the_label_and_scene_fn_hands_it_to_ag3s(monkeypatch, subtask, expected):
    sp = _policy(subtask)
    sp.infer({"state": np.zeros(wire.ACTION_WIDTH), wire.PREFIX + "cameras": []})
    assert sp._pending["subtask"] == subtask                                   # noqa: SLF001
    calls, stop = _spy(sp, monkeypatch)
    with pytest.raises(stop):
        sp._scene_fn(None)                                                     # noqa: SLF001
    assert calls["subtask"] == [expected]
    assert sp.ag3s._subtask_p == expected                                      # noqa: SLF001


@pytest.mark.parametrize("phase,placed", [(GraspPhase.SEARCHING, False), (GraspPhase.LATCHED, False),
                                          (GraspPhase.CLOSING, False), (GraspPhase.HELD, False),
                                          (GraspPhase.PLACED, True)])
def test_set_placed_is_true_only_in_placed(monkeypatch, phase, placed):
    sp = _policy()
    calls, stop = _spy(sp, monkeypatch)
    sp._latch.phase = phase                                                    # noqa: SLF001
    sp._pending = {"scene": {}, "exec_feedback": wire.unpack_exec_feedback({})}  # noqa: SLF001
    with pytest.raises(stop):
        sp._scene_fn(None)                                                     # noqa: SLF001
    assert calls["placed"] == [placed]
    assert sp.ag3s._target_confirm.placed_seen is placed                       # noqa: SLF001


def test_a_revoked_attach_is_not_placed(monkeypatch):
    """T34 J2: the revoke detaches but returns the latch to LATCHED — B3 must not fire."""
    from AG3S.trajopt.grasp_latch import GraspLatch, LatchConfig

    latch = GraspLatch(LatchConfig())
    for _ in range(3):
        latch.update(label="obj0", score=0.9, runner_up=0.1, gripper=1.0)
    latch.update(label="obj0", score=0.9, runner_up=0.1, gripper=0.5)
    assert latch.phase is GraspPhase.HELD
    src = inspect.getsource(GraspLatch)
    assert "attach_revoked" in src or "revoke" in src
    # the revoke branch sets LATCHED, never PLACED (grasp_latch.py T34 J2)
    revoke = src[src.index("거짓 attach 회수"):]
    assert "GraspPhase.LATCHED" in revoke.split("GraspPhase.PLACED")[0]
    sp = _policy()
    calls, stop = _spy(sp, monkeypatch)
    sp._latch.phase = GraspPhase.LATCHED                                       # noqa: SLF001
    sp._pending = {"scene": {}, "exec_feedback": wire.unpack_exec_feedback({})}  # noqa: SLF001
    with pytest.raises(stop):
        sp._scene_fn(None)                                                     # noqa: SLF001
    assert calls["placed"] == [False]


def test_reset_clears_placed_through_safe_policy(monkeypatch):
    sp = _policy()
    sp.ag3s.set_placed(True)
    assert sp.ag3s._target_confirm.placed_seen                                 # noqa: SLF001
    sp.reset()
    assert not sp.ag3s._target_confirm.placed_seen                             # noqa: SLF001


# ------------------------------------------------------------------------------ serve_safe

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def _args(*extra):
    ap = serve_safe.build_parser()
    return ap, ap.parse_args(BASE_ARGV + list(extra))


def test_flags_default():
    """SUBTASK-g: no flag parses to None (= "not given"), which `resolve_subtask_gate` makes on.
    (SUBTASK-c: `False` — the default was off.)"""
    _, args = _args()
    assert args.subtask_gate is None and args.subtask_probe is None
    assert serve_safe.resolve_subtask_gate(args) is None and args.subtask_gate is True


@pytest.mark.parametrize("extra", [["--subtask-gate", "--no-safe"],
                                   ["--subtask-gate", "--no-attention"],
                                   ["--subtask-gate", "--no-perception"],
                                   ["--subtask-probe", "p.npz", "--no-attention"]])
def test_impossible_combinations_are_refused(extra):
    ap, args = _args(*extra)
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)


def test_gate_flag_is_accepted():
    ap, args = _args("--subtask-gate", "--subtask-probe", "p.npz")
    serve_safe.reject_bad_flag_combinations(ap, args)


def test_build_ag3s_puts_the_gate_in_the_config():
    """SUBTASK-g: the fragment comes from `subtask_config_section` (the gate always, when `main`
    gives a bool); SUBTASK-c inlined it and wrote the gate only when on."""
    src = inspect.getsource(serve_safe.build_ag3s)
    assert "**subtask_config_section(subtask_gate, subtask_probe_path)" in src
    main = inspect.getsource(serve_safe.main)
    assert "subtask_gate=args.subtask_gate" in main
    assert 'subtask_probe=args.subtask_probe or "default"' in main
    assert '"subtask": {"gate": bool(args.subtask_gate)' in main


def test_the_config_shape_build_ag3s_produces():
    """The dict fragment `build_ag3s` adds, loaded. SUBTASK-g: a bool gate is always written (off
    too — SUBTASK-c wrote nothing for off, the then default); None = no key = the config default,
    which is now on (SUBTASK-c: off)."""
    from AG3S.ag3s.config import AG3SConfig

    fragment = serve_safe.subtask_config_section
    assert fragment(None, None) == {}
    assert fragment(False, None) == {"clustering": {"subtask_gate": False}}
    assert fragment(True, None) == {"clustering": {"subtask_gate": True}}
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", **fragment(True, "p.npz")})
    assert cfg.clustering.subtask_gate is True and cfg.clustering.subtask_probe_path == "p.npz"
    off = AG3SConfig.from_dict({"collision_backend": "esdf", **fragment(False, None)})
    assert off.clustering.subtask_gate is False and off.clustering.subtask_probe_path is None
    assert AG3SConfig.from_dict({"collision_backend": "esdf"}).clustering.subtask_gate is True


def test_constraint_record_carries_the_subtask_metrics():
    from AG3S.ag3s.experiments.sources import constraint_record

    src = inspect.getsource(constraint_record)
    assert '"gripper_max_opening", "finger_joints", "subtask"' in src
