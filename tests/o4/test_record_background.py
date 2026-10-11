"""T38 B5 — the constraint recorder writes off the response path, and the record is the same.

`SafePolicy._record` cost ≈ 245 ms per chunk inside the request, mostly `zipfile.write`
(T38 Phase A, py-spy). `ConstraintRecordWriter(background=True)` keeps *what* is recorded on the
caller's thread (payload, `summary_json`, array copies) and moves only `np.savez_compressed` to a
worker. These tests pin: same bytes, same order, snapshot isolation, flush on close/exit, and failures
counted and logged rather than dropped.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import pathlib
import subprocess
import sys
import textwrap
import threading
import time
import types

import numpy as np
import pytest

CR = pytest.importorskip("AG3S.ag3s.experiments.sources.constraint_record")
if "background" not in inspect.signature(CR.ConstraintRecordWriter.__init__).parameters:
    pytest.skip("ConstraintRecordWriter has no background mode — not the O4 worktree",
                allow_module_level=True)

import benchmark  # noqa: E402
from AG3S.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays  # noqa: E402
from AG3S.ag3s.fields.esdf import VoxelGrid  # noqa: E402

Writer = CR.ConstraintRecordWriter
WORKTREE = str(pathlib.Path(benchmark.__file__).parents[1])


# ----------------------------------------------------------------------------- fixtures
def _layer(lower, upper, voxel, fn):
    grid = VoxelGrid.from_bounds(np.asarray(lower, float), np.asarray(upper, float), voxel)
    return layer_from_arrays(fn(grid.centres()).astype(np.float32), grid.origin, voxel)


def _field(k):
    coarse = _layer((0.0, -0.3, 0.0), (1.0, 0.3, 0.5), 0.02, lambda p: p[..., 2] + 0.001 * k)
    fine = _layer((0.4, -0.1, -0.05), (0.6, 0.1, 0.15), 0.005, lambda p: p[..., 0] - 0.001 * k)
    free = _layer((0.4, -0.1, -0.05), (0.6, 0.1, 0.15), 0.005, lambda p: p[..., 1] + 0.01)
    field = CuroboEsdfField((coarse, fine), outside_distance=0.5, target_free_layers=(free,))
    field.stats = {"backend": "curobo", "n_occupied": 100 + k}
    return field


def _inputs(k):
    """One chunk's worth of record() arguments; every array dtype already matches what the writer
    stores, so `np.asarray` inside record() returns the caller's own buffer (the aliasing case)."""
    rng = np.random.default_rng(k)
    target = types.SimpleNamespace(points=rng.normal(size=(50, 3)).astype(np.float32),
                                   point_indices=np.arange(50, dtype=np.int32),
                                   centroid=np.zeros(3), attention_score=0.5, confidence=0.9)
    cs = types.SimpleNamespace(
        esdf=_field(k), target=target, candidates=[1, 2], support_surfaces=[], status=None,
        validity=None, notes=[f"note {k}"], metrics={"manipulated": {"state": "held", "k": k}},
        profile={"esdf": 1.5 * k}, has_target=True)
    debug = {"raw_cloud": types.SimpleNamespace(points=rng.normal(size=(400, 3)).astype(np.float32)),
             "seed_indices": np.arange(10, dtype=np.int32)}
    return dict(
        t_step=k, chunk_index=k + 1, constraint_set=cs, debug=debug,
        reference_chunk=rng.normal(size=(32, 16)).astype(np.float32),
        refined_chunk=rng.normal(size=(32, 16)).astype(np.float32),
        sphere_centres=rng.normal(size=(20, 3)).astype(np.float32),
        sphere_radii=np.full(20, 0.02, np.float32), sphere_link_names=[f"l{i}" for i in range(20)],
        clearance=rng.normal(size=(8, 20)).astype(np.float32),
        attention_maps={"zed_left": rng.random((16, 16)).astype(np.float16)},
        summary_extra={"grasp": {"state": "held"}, "verdict": {"reasons": [{"kind": "budget_only"}]}})


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _assert_same_npz(a, b):
    da, db = _load(a), _load(b)
    assert sorted(da) == sorted(db)
    for k in da:
        assert da[k].dtype == db[k].dtype and da[k].shape == db[k].shape, k
        assert da[k].tobytes() == db[k].tobytes(), k


@pytest.fixture
def gate(monkeypatch):
    """Hold the background worker before each write until released (and record which thread wrote)."""
    ev = threading.Event()
    threads = []
    real = np.savez_compressed

    def gated(file, *a, **k):
        threads.append(threading.current_thread().name)
        ev.wait(30)
        return real(file, *a, **k)

    monkeypatch.setattr(np, "savez_compressed", gated)
    ev.threads = threads
    return ev


# ----------------------------------------------------------------------------- tests
def test_sync_is_still_the_default(tmp_path):
    w = Writer(tmp_path)
    assert w.background is False and w.pending == 0
    w.record(**_inputs(0))
    assert (w.run_dir / "chunk_00000.npz").exists()  # written before record() returned
    w.close()


def test_background_writes_the_same_bytes_as_sync(tmp_path):
    sync = Writer(tmp_path / "s", meta={"m": 1})
    bg = Writer(tmp_path / "b", meta={"m": 1}, background=True)
    outs_s = [sync.record(**_inputs(k)) for k in range(5)]
    outs_b = [bg.record(**_inputs(k)) for k in range(5)]
    sync.close()
    bg.close()
    assert [p.name for p in outs_s] == [p.name for p in outs_b]
    for a, b in zip(outs_s, outs_b):
        _assert_same_npz(a, b)
    meta_b = json.loads((bg.run_dir / "meta.json").read_text())
    assert meta_b["n_chunks"] == 5 and meta_b["writer"] == {
        "mode": "background", "n_failed": 0, "failed_paths": []}
    # loaders see the same run
    _, cs = CR.load_constraint_run(sync.run_dir)
    _, cb = CR.load_constraint_run(bg.run_dir)
    assert [c["summary"] for c in cs] == [c["summary"] for c in cb]


def test_the_write_happens_on_the_worker_not_the_caller(tmp_path, gate):
    w = Writer(tmp_path, background=True)
    t = time.perf_counter()
    out = w.record(**_inputs(0))
    assert time.perf_counter() - t < 5.0  # did not wait for the gated write
    assert w.pending == 1 and not out.exists()
    gate.set()
    w.flush()
    assert out.exists() and w.pending == 0
    assert gate.threads == ["constraint-record-writer"]
    w.close()


def test_later_mutation_of_the_callers_arrays_does_not_reach_the_record(tmp_path, gate):
    ref = Writer(tmp_path / "ref")
    expected = ref.record(**_inputs(3)) if gate.set() is None else None  # sync, ungated
    ref.close()
    gate.clear()
    bg = Writer(tmp_path / "bg", background=True)
    inp = _inputs(3)
    out = bg.record(**inp)
    # The caller (next chunk) reuses its buffers before the worker has written anything.
    for arr in (inp["reference_chunk"], inp["refined_chunk"], inp["clearance"],
                inp["sphere_centres"], inp["attention_maps"]["zed_left"],
                inp["debug"]["raw_cloud"].points, inp["debug"]["seed_indices"],
                inp["constraint_set"].target.points):
        arr[...] = 0
    inp["constraint_set"].esdf.layers[-1].distance_grid[...] = -1.0
    inp["constraint_set"].metrics["manipulated"]["state"] = "MUTATED"
    inp["summary_extra"]["grasp"]["state"] = "MUTATED"
    gate.set()
    bg.close()
    _assert_same_npz(expected, out)


def test_order_is_the_call_order(tmp_path):
    w = Writer(tmp_path, background=True, queue_size=2)
    outs = [w.record(**_inputs(k)) for k in range(12)]
    w.close()
    assert [p.name for p in outs] == [f"chunk_{k:05d}.npz" for k in range(12)]
    steps = [json.loads(str(_load(p)["summary_json"]))["t_step"] for p in outs]
    assert steps == list(range(12))


def test_a_full_queue_waits_instead_of_dropping(tmp_path, gate):
    w = Writer(tmp_path, background=True, queue_size=1)
    done = []

    def producer():
        for k in range(4):
            w.record(**_inputs(k))
            done.append(k)

    th = threading.Thread(target=producer)
    th.start()
    time.sleep(0.5)
    assert len(done) < 4  # blocked on the full queue (1 in the worker + 1 queued)
    gate.set()
    th.join(30)
    w.close()
    assert done == [0, 1, 2, 3]
    assert sorted(p.name for p in w.run_dir.glob("chunk_*.npz")) == [
        f"chunk_{k:05d}.npz" for k in range(4)]


def test_close_flushes_everything_queued(tmp_path, gate):
    w = Writer(tmp_path, background=True)
    for k in range(5):
        w.record(**_inputs(k))
    threading.Timer(0.3, gate.set).start()
    w.close()  # must wait for the gated writes
    assert len(list(w.run_dir.glob("chunk_*.npz"))) == 5
    assert json.loads((w.run_dir / "meta.json").read_text())["n_chunks"] == 5
    # after close the writer is synchronous again
    assert w.background is False
    out = w.record(**_inputs(9))
    assert out.exists()


def test_a_failed_write_is_counted_and_logged_not_dropped_silently(tmp_path, monkeypatch, caplog):
    real = np.savez_compressed

    def flaky(file, *a, **k):
        if str(file).endswith("chunk_00001.npz") or str(file).endswith("chunk_00003.npz"):
            raise OSError("disk full (simulated)")
        return real(file, *a, **k)

    monkeypatch.setattr(np, "savez_compressed", flaky)
    w = Writer(tmp_path, background=True)
    with caplog.at_level(logging.ERROR):
        for k in range(5):
            w.record(**_inputs(k))
        w.flush()
    assert w.n_failed == 2 and w.failed_paths == ["chunk_00001.npz", "chunk_00003.npz"]
    assert "CONSTRAINT RECORD WRITE FAILED" in caplog.text and "2 번째" in caplog.text
    # the worker survived and kept writing
    assert sorted(p.name for p in w.run_dir.glob("chunk_*.npz")) == [
        "chunk_00000.npz", "chunk_00002.npz", "chunk_00004.npz"]
    w.close()
    meta = json.loads((w.run_dir / "meta.json").read_text())
    assert meta["n_chunks"] == 3 and meta["writer"]["n_failed"] == 2


def test_a_dead_worker_is_an_error_not_a_hang(tmp_path):
    w = Writer(tmp_path, background=True)
    w._queue.put(None)  # stop the worker behind the writer's back
    w._worker.join(10)
    with pytest.raises(RuntimeError, match="not running"):
        w.record(**_inputs(0))
    w.flush()  # returns
    w.close()


_CHILD = textwrap.dedent('''
    import os, signal, sys, time
    import numpy as np
    sys.path.insert(0, {tests!r})
    from o4.test_record_background import _inputs
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter
    real = np.savez_compressed
    def slow(f, *a, **k):
        time.sleep(0.4)
        return real(f, *a, **k)
    np.savez_compressed = slow
    {setup}
    w = ConstraintRecordWriter({root!r}, background=True)
    for k in range(4):
        w.record(**_inputs(k))
    print("ENQUEUED", w.pending, flush=True)
    {finish}
''')


def _child(tmp_path, setup="", finish=""):
    code = _CHILD.format(tests=str(pathlib.Path(__file__).parents[1]), root=str(tmp_path),
                         setup=setup, finish=finish)
    env = dict(os.environ, PYTHONPATH=WORKTREE)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=120, cwd=WORKTREE)


def test_interpreter_exit_flushes_the_queue(tmp_path):
    r = _child(tmp_path)  # no close(): the process just ends
    assert r.returncode == 0, r.stderr
    assert "ENQUEUED" in r.stdout and "flushed at exit" in r.stdout
    assert len(list(tmp_path.glob("run_0000/chunk_*.npz"))) == 4


def test_sigterm_flushes_the_queue_when_the_server_handler_is_installed(tmp_path):
    r = _child(tmp_path,
               setup="from AG3S.trajopt.serve_safe import exit_cleanly_on_sigterm\n"
                     "exit_cleanly_on_sigterm()",
               finish="os.kill(os.getpid(), signal.SIGTERM)\ntime.sleep(30)\nprint('NOT REACHED')")
    assert r.returncode == 128 + 15, (r.returncode, r.stderr)
    assert "NOT REACHED" not in r.stdout and "flushed at exit" in r.stdout
    assert len(list(tmp_path.glob("run_0000/chunk_*.npz"))) == 4


def test_the_server_builds_its_recorder_in_background_mode():
    src = inspect.getsource(sys.modules.get("AG3S.trajopt.serve_safe")
                            or __import__("AG3S.trajopt.serve_safe", fromlist=["main"]))
    i = src.index("recorder = ConstraintRecordWriter(")
    assert "background=True" in src[i:i + 400]
    assert "exit_cleanly_on_sigterm()" in src[src.index("def main"):]
