"""Two-level GPU lock shared by every checkout on this pod (user decision 2026-10-25).

Modes
-----
``exclusive``  the result depends on wall-clock time, or the job measures time:
               closed-loop runs where TrajOpt refines the chunk (serve_safe safe mode: ``SqpConfig.time_budget_ms`` is a hard
               cap, so a slower GPU changes the SQP iterations and the closed loop), latency / runtime measurements.
               Nothing else holds the lock meanwhile.
``shared``     the result does not depend on wall-clock time: VLA alone (``serve_safe --no-safe``), offline extraction /
               replay, smoke runs that do not report time. Several shared jobs run together while memory admission passes.

Files (all under ``AG3S_GPU_LOCK_DIR``, default ``/tmp``)
---------------------------------------------------------
``ag3s-vla-gpu-validation.lock``  the lock everybody already uses. shared = ``flock -s``, exclusive = ``flock -x``, so a
                                  plain ``flock /tmp/ag3s-vla-gpu-validation.lock CMD`` (legacy) is an exclusive holder and
                                  stays correct against both modes.
``ag3s-vla-gpu-gate.lock``        turnstile: every request passes it before taking the lock, and an exclusive request keeps
                                  it while it waits, so a stream of shared jobs cannot starve a waiting exclusive job.
``ag3s-vla-gpu-ledger.json``      live holders (pid, mode, estimate, tag), guarded by flock on ``.ledger.lock``.

Admission (both modes, checked while holding the gate): ``nvidia-smi memory.used + estimate <= budget`` and
``sum(estimates of live ledger holders) + estimate <= budget``; budget defaults to 120 GiB (pod limit 130 GB minus headroom
for allocation spikes of jobs that are still starting). The request waits (re-checking every ``--poll-s``) until both hold.

usage:
  python3 scripts/setup/gpu_lock.py shared    --est-gib 20 --tag occ-vla -- CMD ARGS...
  python3 scripts/setup/gpu_lock.py exclusive --est-gib 30 --tag occ-ag3s -- CMD ARGS...
  python3 scripts/setup/gpu_lock.py status
The child gets ``AG3S_GPU_LOCK_MODE`` and ``AG3S_GPU_LOCK_HELD=1`` and inherits the lock descriptor (the lock lives as long as
the child does, even if this wrapper is killed). Exit code = the child's; 3 = ``--wait-max-s`` exceeded; 2 = usage error.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DIR = Path(os.environ.get("AG3S_GPU_LOCK_DIR", "/tmp"))
LOCK = DIR / "ag3s-vla-gpu-validation.lock"
GATE = DIR / "ag3s-vla-gpu-gate.lock"
LEDGER = DIR / "ag3s-vla-gpu-ledger.json"
LEDGER_LOCK = DIR / "ag3s-vla-gpu-ledger.json.lock"
DEFAULT_BUDGET_GIB = 120.0


def log(msg: str) -> None:
    print(f"[gpu_lock {time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _open(path: Path) -> int:
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        os.chmod(path, 0o666)
    except PermissionError:
        pass
    return fd


def _acquire(fd: int, op: int, deadline: float | None, what: str, poll_s: float) -> bool:
    """flock with a deadline (polling LOCK_NB so the wait can end); logs once when it has to wait."""
    try:
        fcntl.flock(fd, op | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        log(f"waiting for {what}")
    while deadline is None or time.time() < deadline:
        time.sleep(poll_s)
        try:
            fcntl.flock(fd, op | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            continue
    return False


def gpu_used_mib() -> float:
    cmd = os.environ.get("AG3S_GPU_LOCK_USED_MIB_CMD")   # tests: a shell command printing MiB
    if cmd:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout
    else:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    return float(sum(float(x) for x in out.split()))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Ledger:
    def __enter__(self):
        self.fd = _open(LEDGER_LOCK); fcntl.flock(self.fd, fcntl.LOCK_EX)
        try:
            rows = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
        except json.JSONDecodeError:
            rows = []
        self.rows = [r for r in rows if _alive(int(r["pid"])) or _alive(int(r.get("child", 0) or 0))]
        return self

    def __exit__(self, *exc):
        tmp = LEDGER.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.rows, indent=1)); os.replace(tmp, LEDGER)
        try:
            os.chmod(LEDGER, 0o666)
        except PermissionError:
            pass
        fcntl.flock(self.fd, fcntl.LOCK_UN); os.close(self.fd)


def admitted(est_gib: float, budget_gib: float) -> tuple[bool, dict]:
    used = gpu_used_mib() / 1024.0
    with Ledger() as L:
        booked = sum(float(r["est_gib"]) for r in L.rows)
    info = {"gpu_used_gib": round(used, 1), "booked_gib": round(booked, 1), "est_gib": est_gib, "budget_gib": budget_gib}
    return (used + est_gib <= budget_gib and booked + est_gib <= budget_gib), info


def run(mode: str, est_gib: float, tag: str, budget_gib: float, wait_max_s: float, poll_s: float, cmd: list[str]) -> int:
    deadline = time.time() + wait_max_s if wait_max_s > 0 else None
    gate, lock = _open(GATE), _open(LOCK)
    op = fcntl.LOCK_EX if mode == "exclusive" else fcntl.LOCK_SH
    if not _acquire(gate, fcntl.LOCK_EX, deadline, "the gate (an exclusive request is queued or admitting)", poll_s):
        log("wait limit reached at the gate"); return 3
    try:
        if not _acquire(lock, op, deadline, f"the GPU lock ({mode})", poll_s):
            log("wait limit reached at the GPU lock"); return 3
        first = True
        while True:
            ok, info = admitted(est_gib, budget_gib)
            if ok:
                break
            if first:
                log(f"waiting for GPU memory: {info}"); first = False
            if deadline is not None and time.time() >= deadline:
                log(f"wait limit reached for GPU memory: {info}"); fcntl.flock(lock, fcntl.LOCK_UN); return 3
            time.sleep(poll_s)
        log(f"acquired {mode} ({tag}): {info}")
        env = dict(os.environ, AG3S_GPU_LOCK_MODE=mode, AG3S_GPU_LOCK_HELD="1")
        child = subprocess.Popen(cmd, env=env, pass_fds=(lock,))
        with Ledger() as L:
            L.rows.append({"pid": os.getpid(), "child": child.pid, "mode": mode, "est_gib": est_gib, "tag": tag,
                           "since": time.strftime("%Y-%m-%dT%H:%M:%S"), "cmd": " ".join(cmd)[:200]})
    finally:
        fcntl.flock(gate, fcntl.LOCK_UN); os.close(gate)
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, lambda sig, _f: child.send_signal(sig))
    rc = child.wait()
    with Ledger() as L:
        L.rows = [r for r in L.rows if int(r["pid"]) != os.getpid()]
    fcntl.flock(lock, fcntl.LOCK_UN); os.close(lock)
    log(f"released {mode} ({tag}) rc={rc}")
    return rc


def status() -> int:
    fd = _open(LOCK)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); state = "free"; fcntl.flock(fd, fcntl.LOCK_UN)
    except BlockingIOError:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB); state = "shared"; fcntl.flock(fd, fcntl.LOCK_UN)
        except BlockingIOError:
            state = "exclusive (or a legacy flock holder)"
    gfd = _open(GATE)
    try:
        fcntl.flock(gfd, fcntl.LOCK_EX | fcntl.LOCK_NB); gate = "open"; fcntl.flock(gfd, fcntl.LOCK_UN)
    except BlockingIOError:
        gate = "held (a request is queued or admitting)"
    with Ledger() as L:
        rows = list(L.rows)
    try:
        used = round(gpu_used_mib() / 1024.0, 1)
    except Exception as e:  # noqa: BLE001
        used = f"unavailable ({e})"
    print(json.dumps({"lock": state, "gate": gate, "gpu_used_gib": used, "holders": rows}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["status"]:
        return status()
    cmd = []
    if "--" in argv:
        i = argv.index("--"); cmd = argv[i + 1:]; argv = argv[:i]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("shared", "exclusive"))
    ap.add_argument("--est-gib", type=float, required=True, help="peak GPU memory this job may use (GiB)")
    ap.add_argument("--tag", default="", help="who / what, shown by `status`")
    ap.add_argument("--budget-gib", type=float, default=float(os.environ.get("AG3S_GPU_BUDGET_GIB", DEFAULT_BUDGET_GIB)))
    ap.add_argument("--wait-max-s", type=float, default=0.0, help="0 = wait forever")
    ap.add_argument("--poll-s", type=float, default=5.0)
    a = ap.parse_args(argv)
    if not cmd:
        ap.error("missing command after --")
    if a.est_gib <= 0 or a.est_gib > a.budget_gib:
        ap.error(f"--est-gib must be in (0, {a.budget_gib}]")
    return run(a.mode, a.est_gib, a.tag or os.path.basename(cmd[0]), a.budget_gib, a.wait_max_s, a.poll_s, cmd)


if __name__ == "__main__":
    sys.exit(main())
