"""SL5h-launcher: AG3S_GPU_LOCK_HELD=1 lock handling of launch_libero_paired.py and run_parallel_gpu.sh (CPU, no GPU).

Every test uses a lock file under tmp_path, never the shared /tmp/ag3s-vla-gpu-validation.lock.
"""
from __future__ import annotations

import fcntl
import os
import pathlib
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmark/runners/run_parallel_gpu.sh"


@pytest.fixture
def lp(monkeypatch, tmp_path):
    from benchmark.runners import launch_libero_paired as lp

    monkeypatch.setattr(lp, "GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.delenv("AG3S_GPU_LOCK_HELD", raising=False)
    return lp


def _lock_is_free(path) -> bool:
    with open(path, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(f, fcntl.LOCK_UN)
        return True


class _FlockSpy:
    """Stands in for the ``fcntl`` module inside launch_libero_paired; records every flock(op)."""

    def __init__(self):
        self.ops = []
        for name in ("LOCK_EX", "LOCK_NB", "LOCK_UN", "LOCK_SH"):
            setattr(self, name, getattr(fcntl, name))

    def flock(self, f, op):
        self.ops.append(op)
        return fcntl.flock(f, op)


def _mock_run_mode_world(monkeypatch, lp, tmp_path, *, gpu_apps):
    """Replace run/server/client side effects of run_mode with in-memory fakes; return the recorded settings."""
    import common.run_artifacts as ra

    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    recorded = []
    monkeypatch.setattr(ra, "create_run", lambda **kw: run_dir)
    monkeypatch.setattr(ra, "record_execution", lambda run, *, settings, checkpoint=None: recorded.append(dict(settings)))
    monkeypatch.setattr(lp.l0, "choose_port", lambda requested=None: 23456)
    monkeypatch.setattr(lp.l0, "openpi_python", lambda: "/fake/python")
    monkeypatch.setattr(lp.l0, "wait_ready", lambda server, port, timeout: 1.0)
    monkeypatch.setattr(lp.l0, "client_environment", lambda: {})
    monkeypatch.setattr(lp.l0, "stop_own_process", lambda server: 0)
    monkeypatch.setattr(lp, "server_argv", lambda *a, **k: ["--port", "23456"])
    monkeypatch.setattr(lp, "client_argv", lambda *a, **k: ["client"])
    monkeypatch.setattr(lp, "protocol_settings", lambda args: {})
    monkeypatch.setattr(lp, "runtime_schedule", lambda args: {})
    monkeypatch.setattr(lp, "gpu_compute_apps", gpu_apps)
    fake_sub = SimpleNamespace(STDOUT=subprocess.STDOUT,
                               Popen=lambda *a, **k: SimpleNamespace(pid=4242),
                               run=lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(lp, "subprocess", fake_sub)
    spy = _FlockSpy()
    monkeypatch.setattr(lp, "fcntl", spy)
    args = SimpleNamespace(step="SL5h", run_id="t", conversion="B", ag3s_version="V11AB", task_ids=[0], episode_ids=[0],
                           seed=7, safety_level=None, data_split="dev", busy_gpu_wait=0.0, server_timeout=1.0)
    return args, recorded, spy


def test_env_unset_takes_own_lock_and_checks_gpu(monkeypatch, lp, tmp_path):
    calls = []
    args, recorded, spy = _mock_run_mode_world(monkeypatch, lp, tmp_path, gpu_apps=lambda: calls.append(1) or [])
    lp.run_mode("vla_v11ab", args, {"path": "<ckpt>"}, {})
    assert spy.ops == [fcntl.LOCK_EX, fcntl.LOCK_UN]          # blocking acquire, then release: the original path
    assert calls == [1]                                        # busy-GPU check ran
    result = recorded[-1]
    assert list(result)[:5] == ["role", "mode", "port", "gpu_lock_wait_s", "gpu_lock_acquired_utc"]
    assert "gpu_lock_released_utc" in result and "gpu_lock_mode" not in result
    assert _lock_is_free(lp.GPU_LOCK)


def test_env_value_must_be_1_or_unset(monkeypatch, lp):
    assert lp.gpu_lock_held_by_caller() is False
    monkeypatch.setenv("AG3S_GPU_LOCK_HELD", "0")
    assert lp.gpu_lock_held_by_caller() is False
    monkeypatch.setenv("AG3S_GPU_LOCK_HELD", "1")
    assert lp.gpu_lock_held_by_caller() is True
    monkeypatch.setenv("AG3S_GPU_LOCK_HELD", "yes")
    with pytest.raises(ValueError):
        lp.gpu_lock_held_by_caller()


def test_held_env_refuses_free_lock_and_leaves_it_free(lp):
    with pytest.raises(lp.GpuLockNotHeld, match="not locked"):
        lp.check_caller_holds_gpu_lock()
    assert _lock_is_free(lp.GPU_LOCK)


def test_held_env_refuses_lock_held_by_unrelated_process(lp):
    holder = subprocess.Popen(["flock", lp.GPU_LOCK, "sleep", "30"])   # this process has no fd on the lock
    try:
        for _ in range(100):
            if os.path.exists(lp.GPU_LOCK) and not _lock_is_free(lp.GPU_LOCK):
                break
            time.sleep(0.05)
        with pytest.raises(lp.GpuLockNotHeld, match="not through a descriptor inherited"):
            lp.check_caller_holds_gpu_lock()
    finally:
        holder.kill()
        holder.wait()


def test_held_env_main_refuses_before_allocating_a_run(monkeypatch, lp, capsys):
    import common.run_artifacts as ra

    monkeypatch.setenv("AG3S_GPU_LOCK_HELD", "1")
    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(lp.PROJECT_ROOT))
    monkeypatch.setattr(lp.l0, "checkpoint_identity", lambda path, require=True: {"path": str(path)})
    monkeypatch.setattr(lp, "versions", lambda: pytest.fail("versions() must not run after a refusal"))
    monkeypatch.setattr(ra, "create_run", lambda **kw: pytest.fail("no run may be allocated after a refusal"))
    rc = lp.main(["--task-ids", "0", "--episode-ids", "0", "--run-id", "x", "--conversion", "B"])
    assert rc == 2 and "refused" in capsys.readouterr().err


def test_held_env_with_caller_lock_skips_flock_and_gpu_wait(monkeypatch, lp, tmp_path):
    caller = open(lp.GPU_LOCK, "a")                       # plays the inherited fd 9 of run_parallel_gpu.sh
    fcntl.flock(caller, fcntl.LOCK_EX)
    try:
        monkeypatch.setenv("AG3S_GPU_LOCK_HELD", "1")
        args, recorded, spy = _mock_run_mode_world(
            monkeypatch, lp, tmp_path, gpu_apps=lambda: pytest.fail("no busy-GPU wait in held_by_caller mode"))
        lp.run_mode("vla_v11ab", args, {"path": "<ckpt>"}, {})
        assert all(op & fcntl.LOCK_NB for op in spy.ops) and fcntl.LOCK_UN not in spy.ops   # only non-blocking probes
        result = recorded[-1]
        assert result["gpu_lock_mode"] == "held_by_caller" and result["gpu_lock_caller_pid"] == os.getppid()
        assert result["gpu_lock_fd"] == caller.fileno()
        assert not {"gpu_lock_wait_s", "gpu_lock_acquired_utc", "gpu_lock_released_utc"} & set(result)
        assert not _lock_is_free(lp.GPU_LOCK)             # the caller still holds it
    finally:
        fcntl.flock(caller, fcntl.LOCK_UN)
        caller.close()


# ---- run_parallel_gpu.sh with a fake launcher / nvidia-smi / cgroup -------------------------------------------

FAKE_LAUNCHER = textwrap.dedent("""
    import json, os, sys, time
    from benchmark.runners import launch_libero_paired as lp
    lock = os.environ["AG3S_PARALLEL_TEST_LOCK"]
    info = lp.check_caller_holds_gpu_lock(lock) if lp.gpu_lock_held_by_caller() else {"gpu_lock_mode": "own"}
    t0 = time.time(); time.sleep(float(os.environ.get("FAKE_SLEEP", "1.5")))
    print(json.dumps({"argv": sys.argv[1:], "t0": t0, "t1": time.time(), **info}))
    sys.exit(int(dict(zip(sys.argv[1::2], sys.argv[2::2])).get("--rc", "0")))
""")


@pytest.fixture
def script_env(tmp_path):
    (tmp_path / "fake_launcher.py").write_text(FAKE_LAUNCHER)
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/sh\ncat \"$(dirname \"$0\")/apps.txt\"\n")
    smi.chmod(0o755)
    (tmp_path / "apps.txt").write_text("")
    cg = tmp_path / "cgroup"
    cg.mkdir()
    (cg / "memory.current").write_text(str(30 << 30) + "\n")
    (cg / "memory.stat").write_text(f"anon 1\ninactive_file {10 << 30}\nactive_file 5\n")
    env = {**os.environ, "AG3S_PROJECT_ROOT": str(ROOT), "PYTHONPATH": str(ROOT),
           "AG3S_PARALLEL_TEST_LOCK": str(tmp_path / "gpu.lock"),
           "AG3S_PARALLEL_TEST_LAUNCHER": f"{ROOT / '.venv-ag3s/bin/python'} {tmp_path / 'fake_launcher.py'}",
           "AG3S_PARALLEL_TEST_NVIDIA_SMI": str(smi), "AG3S_PARALLEL_TEST_CGROUP_DIR": str(cg)}
    env.pop("AG3S_GPU_LOCK_HELD", None)
    return tmp_path, env


def _run_script(tmp_path, env, jobs, *extra):
    (tmp_path / "jobs.txt").write_text(jobs)
    log = tmp_path / "batch.log"
    p = subprocess.run(["bash", str(SCRIPT), "--jobs", str(tmp_path / "jobs.txt"), "--stagger-s", "0",
                        "--log", str(log), *extra], env=env, capture_output=True, text=True, timeout=120)
    return p, (log.read_text() if log.exists() else "")


def test_script_runs_at_most_two_jobs_under_one_held_lock(script_env):
    import json

    tmp_path, env = script_env
    jobs = "# comment\n--run-id a --x 'two words'\n\n--run-id b\n--run-id c --rc 3\n"
    p, log = _run_script(tmp_path, env, jobs, "--est-gib-per-run", "12")
    assert p.returncode == 1, p.stderr                          # job c exits 3 -> batch rc 1
    assert log.count("event=start") == 3 and log.count("event=end") == 3
    assert "rc=3" in log and log.count("rc=0") == 2 and "preflight ok" in log
    outs = [json.loads((tmp_path / f"batch.job{i}.out").read_text().strip().splitlines()[-1]) for i in range(3)]
    assert outs[0]["argv"] == ["--run-id", "a", "--x", "two words"]   # shlex parsing, quoting kept
    assert all(o["gpu_lock_mode"] == "held_by_caller" for o in outs)
    spans = [(o["t0"], o["t1"]) for o in outs]
    peak = max(sum(a <= t < b for a, b in spans) for t, _ in spans)
    assert peak == 2                                              # parallel, never more than 2
    assert _lock_is_free(env["AG3S_PARALLEL_TEST_LOCK"])          # released after the batch


@pytest.mark.parametrize("case", ["busy_gpu", "memory", "lock_taken", "max_parallel"])
def test_script_refusals_start_no_job(script_env, case):
    tmp_path, env = script_env
    extra, holder = [], None
    if case == "busy_gpu":
        (tmp_path / "apps.txt").write_text("12345, 9000 MiB\n")
    elif case == "memory":
        extra = ["--est-gib-per-run", "30"]                       # 20 GiB working set + 30 x 2 > 64
    elif case == "max_parallel":
        extra = ["--max-parallel", "3"]
    else:
        holder = subprocess.Popen(["flock", env["AG3S_PARALLEL_TEST_LOCK"], "sleep", "30"])
        time.sleep(0.3)
        extra = ["--lock-wait", "1"]
    try:
        p, log = _run_script(tmp_path, env, "--run-id a\n--run-id b\n", *extra)
    finally:
        if holder is not None:
            holder.kill()
            holder.wait()
    assert p.returncode == 2 and "REFUSED" in log and "event=start" not in log, (p.stderr, log)
