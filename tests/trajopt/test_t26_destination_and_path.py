"""T26 on the SafePolicy side — destination from AG3S, grasp freeze, execution path.

1. `GraspLatch(external_destination=True)` does not learn the destination from grounding labels
   (T26: the target stays the apple after the grasp, so "the label after the lock" would lock the
   apple as its own destination); it takes it from `set_destination`. The default mode is unchanged.
2. `SafePolicy._run_latch` reads `ag3s.destination` (registered or injected), hands its label to the
   latch and mirrors its points; `process_multi_debug` is not given `destination_points` any more
   (AG3S owns the destination — passing it would be an injection).
3. `set_grasp_active` is True exactly while the latch is `closing` / `held`.
4. `execution_path` (T21b): the reference chunk's execution window, K = `execution_length` rows, as
   full robot-state vectors (`ChunkLayout.full_q`), is passed to `process_multi_debug`.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from benchmark.trajopt import wire
from benchmark.trajopt.grasp_latch import GraspLatch, GraspPhase, LatchConfig


# ----------------------------------------------------------------------------- 1. the latch


def _lock_and_hold(latch):
    for _ in range(3):
        latch.update(label="obj0", score=0.9, runner_up=0.1, gripper=1.0)
    ev = latch.update(label="obj0", score=0.9, runner_up=0.1, gripper=0.5)
    assert ev.attach and latch.phase is GraspPhase.HELD


def test_external_destination_is_not_learned_from_labels():
    latch = GraspLatch(LatchConfig(), external_destination=True)
    _lock_and_hold(latch)
    for _ in range(6):   # the apple's label keeps coming (T26: the target stays the apple)
        ev = latch.update(label="obj0", score=0.9, runner_up=0.1, gripper=0.5)
    assert ev.destination is None and latch.destination is None
    latch.set_destination("destination0")
    ev = latch.update(label="obj0", score=0.9, runner_up=0.1, gripper=0.5)
    assert ev.destination == "destination0" == latch.destination
    latch.reset()
    assert latch.destination is None


def test_the_default_latch_still_learns_from_labels_and_refuses_set_destination():
    latch = GraspLatch(LatchConfig())
    _lock_and_hold(latch)
    for _ in range(3):
        ev = latch.update(label="obj1", score=0.9, runner_up=0.1, gripper=0.5)
    assert ev.destination == "obj1"
    with pytest.raises(RuntimeError):
        latch.set_destination("x")


# ----------------------------------------------------------------------------- 2–4. SafePolicy


def _policy(**kw):
    from tests.trajopt.test_safe_policy import _StubPolicy, _safe_policy

    return _safe_policy(_StubPolicy(), **kw)


def test_safe_policy_latch_takes_the_ag3s_destination():
    sp = _policy()
    assert sp._latch.external_destination                                      # noqa: SLF001
    crate = np.random.default_rng(0).uniform([0.4, 0.2, 0.75], [0.6, 0.6, 0.9], (500, 3))
    sp.ag3s.reset(destination_points=crate)
    q = np.zeros(sp.constraint_robot_model.nq)
    obs = [types.SimpleNamespace(timestamp=1.0, robot_state=q)]
    sp._pending = {"scene": {}, "chunk": np.zeros((4, wire.ACTION_WIDTH))}    # noqa: SLF001
    sp._run_latch(types.SimpleNamespace(target=None), obs, ["left"])           # noqa: SLF001
    assert sp._latch.destination == sp.ag3s.destination.label == "destination0"  # noqa: SLF001
    np.testing.assert_array_equal(sp._destination_points, crate)              # noqa: SLF001


def _scene_fn_spy(sp, monkeypatch, q):
    obs = [types.SimpleNamespace(timestamp=1.0, robot_state=q, camera_id="zed_left")]
    monkeypatch.setattr(wire, "unpack_camera_observations", lambda scene, **kw: obs)
    calls = []

    class _Stop(Exception):
        pass

    def spy(*args, **kwargs):
        calls.append((dict(kwargs), sp.ag3s._grasp_active))                   # noqa: SLF001
        raise _Stop

    monkeypatch.setattr(sp.ag3s, "process_multi_debug", spy)
    return calls, _Stop


def test_scene_fn_passes_the_execution_window_and_no_destination(monkeypatch):
    sp = _policy()
    q = np.linspace(-0.3, 0.3, sp.constraint_robot_model.nq)
    rng = np.random.default_rng(1)
    chunk = rng.normal(0.0, 0.2, (16, wire.ACTION_WIDTH))
    calls, stop = _scene_fn_spy(sp, monkeypatch, q)
    sp._pending = {"scene": {}, "chunk": chunk,                                # noqa: SLF001
                   "exec_feedback": wire.unpack_exec_feedback({})}
    with pytest.raises(stop):
        sp._scene_fn(None)                                                     # noqa: SLF001
    kwargs, _ = calls[-1]
    assert "destination_points" not in kwargs
    K = sp.to_config.horizon.execution_length
    path = kwargs["execution_path"]
    assert path.shape == (K, sp.constraint_robot_model.nq)
    expected = sp.layout.full_q(sp.layout.chunk_to_trajectory(chunk)[:, :K], q).T
    np.testing.assert_array_equal(path, expected)
    # joints the chunk does not drive stay at q_now (torso etc.)
    free = set(int(i) for i in sp.layout.q_indices)
    fixed = [i for i in range(len(q)) if i not in free]
    np.testing.assert_array_equal(path[:, fixed], np.tile(q[fixed], (K, 1)))
    assert sp._exec_summary()["execution_path"]["rows"] == K                   # noqa: SLF001


def test_scene_fn_without_a_chunk_passes_no_path(monkeypatch):
    sp = _policy()
    q = np.zeros(sp.constraint_robot_model.nq)
    calls, stop = _scene_fn_spy(sp, monkeypatch, q)
    sp._pending = {"scene": {}, "exec_feedback": wire.unpack_exec_feedback({})}  # noqa: SLF001
    with pytest.raises(stop):
        sp._scene_fn(None)                                                     # noqa: SLF001
    assert "execution_path" not in calls[-1][0]
    assert sp._last_execution_path["note"] == "no reference chunk"            # noqa: SLF001


@pytest.mark.parametrize("phase,active", [(GraspPhase.SEARCHING, False), (GraspPhase.LATCHED, False),
                                          (GraspPhase.CLOSING, True), (GraspPhase.HELD, True),
                                          (GraspPhase.PLACED, False)])
def test_grasp_active_follows_the_latch_phase(monkeypatch, phase, active):
    sp = _policy()
    q = np.zeros(sp.constraint_robot_model.nq)
    calls, stop = _scene_fn_spy(sp, monkeypatch, q)
    monkeypatch.setattr(sp, "_advance_grasp", lambda *a, **k: None)
    sp._latch.phase = phase                                                    # noqa: SLF001
    sp._pending = {"scene": {}, "exec_feedback": wire.unpack_exec_feedback({})}  # noqa: SLF001
    with pytest.raises(stop):
        sp._scene_fn(None)                                                     # noqa: SLF001
    assert calls[-1][1] is active
