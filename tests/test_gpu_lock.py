"""scripts/setup/gpu_lock.py: shared / exclusive GPU lock (lock files in a temp dir, nvidia-smi replaced by a fixed value)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "scripts" / "setup" / "gpu_lock.py"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="flock semantics")


def env(tmp_path, used_mib=1024):
    return dict(os.environ, AG3S_GPU_LOCK_DIR=str(tmp_path), AG3S_GPU_LOCK_USED_MIB_CMD=f"echo {used_mib}")


def job(tmp_path, name, mode, sleep_s, est=10, extra=(), used_mib=1024):
    """start a locked child that records its start / end time in <name>.json."""
    out = tmp_path / f"{name}.json"
    code = (f"import json,time,os; t0=time.time(); time.sleep({sleep_s}); "
            f"json.dump({{'start': t0, 'end': time.time(), 'mode': os.environ['AG3S_GPU_LOCK_MODE']}}, open({str(out)!r}, 'w'))")
    return subprocess.Popen([sys.executable, str(TOOL), mode, "--est-gib", str(est), "--poll-s", "0.05", "--tag", name, *extra,
                             "--", sys.executable, "-c", code], env=env(tmp_path, used_mib),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def times(tmp_path, name):
    return json.loads((tmp_path / f"{name}.json").read_text())


def test_shared_jobs_overlap(tmp_path):
    a, b = job(tmp_path, "a", "shared", 1.0), job(tmp_path, "b", "shared", 1.0)
    assert a.wait() == 0 and b.wait() == 0
    ta, tb = times(tmp_path, "a"), times(tmp_path, "b")
    assert ta["mode"] == "shared" and max(ta["start"], tb["start"]) < min(ta["end"], tb["end"])


def test_exclusive_waits_for_shared_and_blocks_later_shared(tmp_path):
    s1 = job(tmp_path, "s1", "shared", 1.0); time.sleep(0.3)
    x = job(tmp_path, "x", "exclusive", 0.5); time.sleep(0.3)
    s2 = job(tmp_path, "s2", "shared", 0.2)          # queued behind the exclusive request (gate)
    assert s1.wait() == 0 and x.wait() == 0 and s2.wait() == 0
    t1, tx, t2 = times(tmp_path, "s1"), times(tmp_path, "x"), times(tmp_path, "s2")
    assert tx["start"] >= t1["end"] - 1e-3            # exclusive starts after the running shared job ends
    assert t2["start"] >= tx["end"] - 1e-3            # the later shared job did not jump the queue


def test_memory_admission_waits_and_times_out(tmp_path):
    p = job(tmp_path, "big", "shared", 0.1, est=30, extra=("--budget-gib", "40", "--wait-max-s", "0.5"), used_mib=20 * 1024)
    assert p.wait() == 3 and not (tmp_path / "big.json").exists()
    assert "waiting for GPU memory" in p.stderr.read()


def test_ledger_books_running_estimates(tmp_path):
    a = job(tmp_path, "a", "shared", 1.0, est=25, extra=("--budget-gib", "40")); time.sleep(0.4)
    b = job(tmp_path, "b", "shared", 0.1, est=20, extra=("--budget-gib", "40"))   # 25 + 20 > 40 -> waits for a
    assert a.wait() == 0 and b.wait() == 0
    assert times(tmp_path, "b")["start"] >= times(tmp_path, "a")["end"] - 1e-3
    assert json.loads((tmp_path / "ag3s-vla-gpu-ledger.json").read_text()) == []


def test_legacy_flock_holder_blocks_shared(tmp_path):
    legacy = subprocess.Popen(["flock", str(tmp_path / "ag3s-vla-gpu-validation.lock"), "sleep", "1"])
    time.sleep(0.2)
    p = job(tmp_path, "s", "shared", 0.1, extra=("--wait-max-s", "0.4"))
    assert p.wait() == 3
    legacy.wait()


def test_child_exit_code_and_status(tmp_path):
    p = subprocess.run([sys.executable, str(TOOL), "exclusive", "--est-gib", "5", "--", sys.executable, "-c", "raise SystemExit(7)"],
                       env=env(tmp_path), capture_output=True, text=True)
    assert p.returncode == 7
    s = subprocess.run([sys.executable, str(TOOL), "status"], env=env(tmp_path), capture_output=True, text=True, check=True)
    assert json.loads(s.stdout)["lock"] == "free"
