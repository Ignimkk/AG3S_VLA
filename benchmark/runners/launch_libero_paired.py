"""Paired LIBERO runs: VLA-only and VLA+V11AB on the same tasks / init states / seeds (SL5a/SL5b, ``.venv-ag3s``).

For each mode a new run is allocated (``common.run_artifacts.create_run``), the Panda server
(``AG3S.trajopt.serve_safe --robot panda``, ``--no-safe`` for VLA-only) is started on a free port, the same
client (``benchmark/runners/libero_safe_client.py``, ``.venv-libero``) runs the episodes, and the server this
launcher started is stopped. Each mode holds the shared GPU lock (``/tmp/ag3s-vla-gpu-validation.lock``,
``fcntl.flock``) for its whole server+client lifetime and refuses to start while other GPU compute processes run.

With ``AG3S_GPU_LOCK_HELD=1`` (``benchmark/runners/run_parallel_gpu.sh``) the caller holds that lock for several
launchers of this checkout at once: the launcher then takes no lock and does not wait for busy GPUs, but refuses
to start unless the lock is held through a descriptor it inherited from the caller (``check_caller_holds_gpu_lock``).

Both runs get the same protocol, task list, episode list, ``--seed`` and per-episode ``policy_seed``; the per-task
env reset history is the same because both modes run the same episode order. After both modes,
``paired_summary.json`` is written into the VLA+V11AB run (and referenced from both manifests): per episode the
official success of each mode and the candidate list (VLA-only failure, VLA+V11AB success).

``--record-constraints {none,occupancy,full}`` (SL5j, default ``none``) appends
``--record-constraints <run_dir>/server_constraints --record-constraints-esdf <value>`` to the VLA+V11AB server
argv (``serve_safe``'s per-call constraint record, ``AG3S/ag3s/experiments/sources/constraint_record.py``). The run
directory exists only after ``create_run``, so the manifest ``settings.server_argv`` carries the ``<run_dir>`` token
and ``settings.record_constraints`` says so; the resolved argv is the ``role=server`` execution entry. After the
server stops the launcher counts the chunk files (``launcher_result.record_constraints``). Refused together with
the ``vla_only`` mode (``--no-safe`` server). With ``none`` the server argv and the settings are unchanged.

BM-SETUP-A: every suite of ``benchmark/configs/suites.json`` (LIBERO spatial / object / goal / 10, SafeLIBERO spatial /
object / goal / long x Level I / II) runs through the same launcher; task / episode ids and registry split labels are
checked against the suite before anything starts (``suite_registry.check_selection``). ``--safelibero-step-budget
main_aegis`` (opt-in) runs the official main_aegis.py step budget; the default keeps the SL5c protocol and leaves the
client argv and the manifest settings unchanged. Batches of jobs: ``benchmark/runners/eval_matrix.py``.

    source scripts/setup/env.sh
    .venv-ag3s/bin/python benchmark/runners/launch_libero_paired.py --task-ids 0 --episode-ids 0 --run-id sl5b-t0e0
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.runners import launch_libero_pi05_only as l0  # noqa: E402

GPU_LOCK = "/tmp/ag3s-vla-gpu-validation.lock"
GPU_LOCK_HELD_ENV = "AG3S_GPU_LOCK_HELD"
DEFAULT_SPLIT = "dev (LIBERO init 0-4, plan §6.0)"
MODES = ("vla_only", "vla_v11ab")
CLIENT_SCRIPT = PROJECT_ROOT / "benchmark/runners/libero_safe_client.py"
RECORD_CONSTRAINTS_CHOICES = ("none", "occupancy", "full")
RECORD_CONSTRAINTS_DIRNAME = "server_constraints"
RUN_DIR_TOKEN = "<run_dir>"  # manifest settings are written before the run directory exists
SAFELIBERO_STEP_BUDGETS = ("total_incl_wait", "main_aegis")  # = libero_safe_client.SAFELIBERO_STEP_BUDGETS
SAFELIBERO_DEFAULT_STEP_BUDGET = "total_incl_wait"


def server_argv(mode: str, checkpoint: str, port: int, version: str = "V11AB", **version_options) -> list[str]:
    """``version_options`` (SL6-A, final-configuration versions only): ``hsr6_probe`` (required) and
    ``attention_source`` -> ``profile.version_server_flags``. A final version's base argv carries no
    ``V11AB_PRE_MERGE_DEFAULTS`` (``profile.version_server_argv``); the V11AB family is unchanged."""
    from AG3S.adapters.panda import profile

    if mode == "vla_only":
        rby1_xml = profile.v11ab_server_argv()[profile.v11ab_server_argv().index("--model-xml") + 1]
        return ["--config", "pi05_libero", "--checkpoint", checkpoint, "--model-xml", rby1_xml, "--port", str(port),
                "--robot", "panda", "--no-safe"]
    argv = list(profile.version_server_argv(version))
    argv[argv.index("--config") + 1] = "pi05_libero"
    argv[argv.index("--checkpoint") + 1] = checkpoint
    argv[argv.index("--port") + 1] = str(port)
    return argv + ["--robot", "panda"] + profile.version_server_flags(version, **version_options)


def version_options(args) -> dict:
    """SL6-A — ``--hsr6-probe`` / ``--attention-source`` as ``server_argv`` keywords; ``{}`` when neither is given (the
    call is then the pre-SL6 one, so a launcher that replaces ``server_argv`` keeps working for the V11AB family)."""
    out = {}
    if getattr(args, "hsr6_probe", None):
        out["hsr6_probe"] = str(args.hsr6_probe)
    if getattr(args, "attention_source", None):
        out["attention_source"] = str(args.attention_source)
    return out


def record_constraints_argv(mode: str, run_dir: str | Path, record: str) -> list[str]:
    """serve_safe flags of ``--record-constraints <record>``; ``[]`` for ``none`` (argv unchanged)."""
    if record not in RECORD_CONSTRAINTS_CHOICES:
        raise ValueError(f"--record-constraints must be one of {RECORD_CONSTRAINTS_CHOICES}, got {record!r}")
    if record == "none":
        return []
    if mode != "vla_v11ab":
        raise ValueError(f"--record-constraints {record} applies to the vla_v11ab server only "
                         f"(the {mode} server runs --no-safe and builds no constraints)")
    return ["--record-constraints", f"{run_dir}/{RECORD_CONSTRAINTS_DIRNAME}", "--record-constraints-esdf", record]


def record_constraints_summary(run_dir: Path, record: str) -> dict:
    """What the server left under ``<run_dir>/server_constraints`` (counted after it stopped)."""
    root = Path(run_dir) / RECORD_CONSTRAINTS_DIRNAME
    chunks = sorted(root.glob("run_*/chunk_*.npz"))
    return {"esdf": record, "dir": str(root), "writer_runs": sorted(p.name for p in root.glob("run_*")),
            "n_chunk_files": len(chunks), "bytes": sum(p.stat().st_size for p in chunks),
            "meta_json": [p.parent.name for p in sorted(root.glob("run_*/meta.json"))]}


def step_budget(args) -> str:
    """``--safelibero-step-budget`` (BM-SETUP-A); namespaces built before it existed get the default."""
    return getattr(args, "safelibero_step_budget", SAFELIBERO_DEFAULT_STEP_BUDGET)


def client_argv(run_dir: str, port: int, args) -> list[str]:
    budget = step_budget(args)  # appended only when not the default: default argv unchanged
    return [str(l0.libero_python()), str(CLIENT_SCRIPT), "--run-dir", str(run_dir), "--host", "127.0.0.1",
            "--port", str(port), "--task-suite-name", args.task_suite_name,
            "--task-ids", ",".join(map(str, args.task_ids)), "--episode-ids", ",".join(map(str, args.episode_ids)),
            "--seed", str(args.seed)] + (["--capture-segmentation"] if args.capture_segmentation else []) + (
                ["--safety-level", args.safety_level] if args.safety_level else []) + (
                ["--safelibero-step-budget", budget] if budget != SAFELIBERO_DEFAULT_STEP_BUDGET else [])


def oracle_client_argv(mode: str, args) -> list[str]:
    """SL12: ``--oracle-segmentation`` for the vla_v11ab client of an ``-oraclegt`` version (the server runs ``--target-extent
    oracle-visible`` and needs the GT code image); ``[]`` for every other version and for the vla_only client."""
    from AG3S.adapters.panda import profile

    out = []
    if mode == "vla_v11ab" and profile.needs_oracle_segmentation(getattr(args, "ag3s_version", "V11AB")):
        from benchmark.adapters.libero_panda import oracle_codes
        out.append(oracle_codes.CLIENT_FLAG)
    if mode == "vla_v11ab" and profile.needs_oracle_grasp(getattr(args, "ag3s_version", "V11AB")):
        # SL14: ``--oracle-grasp`` for the vla_v11ab client of an ``--oracle-latch`` version (the server needs ag3s/oracle_grasp)
        from benchmark.adapters.libero_panda import oracle_codes
        out.append(oracle_codes.CLIENT_GRASP_FLAG)
    return out


def oracle_settings(mode: str, args) -> dict:
    """SL12: the manifest entry that says what the oracle is and how its target is chosen ({} when the client has no oracle).
    SL14: the ground-truth grasp entry for an ``--oracle-grasp`` client."""
    argv = oracle_client_argv(mode, args)
    if not argv:
        return {}
    from benchmark.adapters.libero_panda import oracle_codes
    out = {}
    if oracle_codes.CLIENT_FLAG in argv:
        out.update(oracle_codes.settings_record())
    if oracle_codes.CLIENT_GRASP_FLAG in argv:
        out.update(oracle_codes.grasp_settings_record())
    return out


def _is_safelibero(args) -> bool:
    return args.task_suite_name.startswith("safelibero")


def protocol_settings(args) -> dict:
    """``l0.protocol_settings`` for LIBERO; for SafeLIBERO the same keys with the official main_aegis.py step
    protocol (``libero_safe_client.safelibero_protocol``) — what the client actually runs."""
    if not _is_safelibero(args):
        return l0.protocol_settings(args)
    from benchmark.runners import libero_safe_client as sc

    out = l0.protocol_settings(argparse.Namespace(**{**vars(args), "task_suite_name": "libero_spatial"}))
    total = sc.SAFELIBERO_TOTAL_STEPS[args.task_suite_name]
    out.update({"suite": args.task_suite_name, "safety_level": args.safety_level,
                "max_steps": total - sc.SAFELIBERO_NUM_STEPS_WAIT, "num_steps_wait": sc.SAFELIBERO_NUM_STEPS_WAIT,
                "loop": f"t < max_steps + num_steps_wait = {total} (wait steps counted)",
                "safelibero": sc.safelibero_protocol()})
    budget = step_budget(args)
    if budget != SAFELIBERO_DEFAULT_STEP_BUDGET:  # default: the keys above, unchanged
        rec = sc.safelibero_step_budget_record(args.task_suite_name, budget)
        out.update({"max_steps": rec["policy_steps"],
                    "loop": f"t < max_steps + num_steps_wait = {rec['policy_steps'] + rec['num_steps_wait']} "
                            "(wait steps counted; main_aegis budget: wait on top of the total)",
                    "safelibero_step_budget": rec})
    return out


def runtime_schedule(args) -> dict:
    if not _is_safelibero(args):
        return l0.runtime_schedule(args)
    from benchmark.runners import libero_safe_client as sc

    path = PROJECT_ROOT / "benchmark/SafeLIBERO/safelibero/libero/libero/envs/env_wrapper.py"
    out = l0.runtime_schedule(args)
    hz = l0.libero_control_freq(path)
    out.update({"control_freq_hz": hz, "control_freq_source": f"{path.relative_to(PROJECT_ROOT)} ControlEnv.__init__ default",
                "control_period_s": 1.0 / hz, "policy_call_period_sim_s": out["replan_steps"] / hz,
                "num_steps_wait": sc.SAFELIBERO_NUM_STEPS_WAIT, "first_policy_call_env_step": sc.SAFELIBERO_NUM_STEPS_WAIT})
    return out


def gpu_compute_apps() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


class GpuLockNotHeld(RuntimeError):
    """``AG3S_GPU_LOCK_HELD=1`` was set but the caller does not hold the shared GPU lock for this launcher."""


def gpu_lock_held_by_caller() -> bool:
    """True when ``AG3S_GPU_LOCK_HELD=1``; unset (or ``0``) keeps the launcher's own lock handling."""
    value = os.environ.get(GPU_LOCK_HELD_ENV)
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"{GPU_LOCK_HELD_ENV} must be 1 (caller holds {GPU_LOCK}) or unset, got {value!r}")


def _same_file_fds(st: os.stat_result) -> list[int]:
    out = []
    for name in os.listdir("/proc/self/fd"):
        try:
            fst = os.fstat(int(name))
        except (OSError, ValueError):
            continue
        if (fst.st_dev, fst.st_ino) == (st.st_dev, st.st_ino):
            out.append(int(name))
    return sorted(out)


def check_caller_holds_gpu_lock(path: str | None = None) -> dict:
    """Confirm, without taking it, that the shared GPU lock is held by the caller of this launcher.

    1. A fresh open file description of ``path`` must fail ``LOCK_EX | LOCK_NB`` (somebody holds the lock);
       if it succeeds the lock was free: it is released at once and the launcher refuses.
    2. One descriptor this process inherited from the caller must refer to the same file and be the holder
       (``LOCK_EX | LOCK_NB`` on the holder's own description is a no-op that succeeds). This rejects a lock held
       by an unrelated process (another checkout), which step 1 alone cannot tell apart. Inherited descriptors are
       never unlocked here.
    """
    path = path or GPU_LOCK
    with open(path, "a") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise GpuLockNotHeld(f"{GPU_LOCK_HELD_ENV}=1 but {path} is not locked: the caller must hold it "
                                 "(benchmark/runners/run_parallel_gpu.sh)")
        st = os.fstat(probe.fileno())
        probe_fd = probe.fileno()
        candidates = [fd for fd in _same_file_fds(st) if fd != probe_fd]
    for fd in candidates:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            continue
        return {"gpu_lock_mode": "held_by_caller", "gpu_lock_caller_pid": os.getppid(), "gpu_lock_fd": fd}
    raise GpuLockNotHeld(f"{GPU_LOCK_HELD_ENV}=1 and {path} is locked, but not through a descriptor inherited by "
                         f"this launcher (inherited fds on that file: {candidates}); another process holds it")


def versions() -> dict:
    from AG3S.adapters.panda import action as pa
    from AG3S.adapters.panda import profile

    return {"panda_profile": profile.manifest_entry(), "conversion": pa.conventions(),
            "libero_python": l0.interpreter_info(l0.libero_python(), l0.LIBERO_PACKAGES),
            "openpi_python": l0.interpreter_info(l0.openpi_python(), l0.OPENPI_PACKAGES)}


def run_tag(mode: str, args) -> str:
    """Suffix of the run id of ``mode``: ``<run-id>-<tag>`` (the run directory ends with it)."""
    tag = mode
    if mode == "vla_v11ab":
        tag = f"{mode}-{args.conversion}" if args.ag3s_version == "V11AB" else f"{mode}-{args.ag3s_version}-{args.conversion}"
    if args.safety_level:
        tag = f"L{args.safety_level}-{tag}"
    return tag


def run_mode(mode: str, args, identity: dict, shared: dict) -> Path:
    from common.run_artifacts import create_run, record_execution

    port = l0.choose_port(None)
    s_argv = [str(l0.openpi_python()), "-m", "AG3S.trajopt.serve_safe"] + server_argv(mode, identity["path"], port,
                                                                                       args.ag3s_version,
                                                                                       **version_options(args))
    record = getattr(args, "record_constraints", "none")
    rec_settings = {}
    if record != "none":
        rec_settings = {"record_constraints": {
            "esdf": record, "dir": f"{RUN_DIR_TOKEN}/{RECORD_CONSTRAINTS_DIRNAME}",
            "argv_note": f"settings.server_argv keeps the {RUN_DIR_TOKEN} token (the run directory does not exist "
                         "yet); the resolved argv is the role=server execution entry"}}
    settings = {"step": args.step, "condition": mode, "paired_group": args.run_id, "port": port,
                "panda_conversion_mode": args.conversion if mode == "vla_v11ab" else None,
                "ag3s_version": args.ag3s_version if mode == "vla_v11ab" else None,
                "protocol": protocol_settings(args), "runtime_schedule": runtime_schedule(args),
                "task_ids": args.task_ids, "episode_ids": args.episode_ids, "seed": args.seed,
                "safety_level": args.safety_level,
                "policy_seed_rule": "seed*100000 + task_id*1000 + episode_idx (libero_safe_client.episode_seed)",
                "checkpoint": identity, "server_argv": s_argv + record_constraints_argv(mode, RUN_DIR_TOKEN, record),
                "data_split": args.data_split, **rec_settings, **oracle_settings(mode, args), **shared}
    run_dir = create_run(run_id=f"{args.run_id}-{run_tag(mode, args)}", settings=settings, checkpoint=identity["path"])
    print(f"[{mode}] {run_dir}", flush=True)
    s_argv = s_argv + record_constraints_argv(mode, run_dir, record)
    c_argv = client_argv(str(run_dir), port, args) + oracle_client_argv(mode, args)   # SL12 (else unchanged)
    logs = run_dir / "logs"
    result = {"role": "launcher_result", "mode": mode, "port": port}
    server = None
    held = gpu_lock_held_by_caller()
    lock = None if held else open(GPU_LOCK, "a")
    try:
        if held:
            # caller (run_parallel_gpu.sh) holds the lock and checked the GPU once before starting its jobs
            result.update(check_caller_holds_gpu_lock())
        else:
            t_wait = time.monotonic()
            fcntl.flock(lock, fcntl.LOCK_EX)
            result["gpu_lock_wait_s"] = time.monotonic() - t_wait
            result["gpu_lock_acquired_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            deadline = time.monotonic() + args.busy_gpu_wait
            while gpu_compute_apps():
                if time.monotonic() > deadline:
                    raise RuntimeError(f"other GPU compute processes still running: {gpu_compute_apps()}")
                time.sleep(15)
        env = dict(os.environ, AG3S_RUN_DIR=str(run_dir), AG3S_PANDA_CONVERSION=args.conversion)
        with open(logs / "server_console.log", "x") as server_log:
            server = subprocess.Popen(s_argv, cwd=str(PROJECT_ROOT), env=env, stdout=server_log, stderr=subprocess.STDOUT)
        record_execution(run_dir, checkpoint=identity["path"], settings={"role": "server", "argv": s_argv, "pid": server.pid})
        result["server_ready_seconds"] = l0.wait_ready(server, port, args.server_timeout)
        print(f"[{mode}] server ready on {port} after {result['server_ready_seconds']:.0f}s", flush=True)
        cenv = l0.client_environment()
        record_execution(run_dir, checkpoint=identity["path"], settings={"role": "client", "argv": c_argv})
        started = time.monotonic()
        with open(logs / "libero_client.log", "x") as client_log:
            client = subprocess.run(c_argv, cwd=str(run_dir), env=cenv, stdout=client_log, stderr=subprocess.STDOUT)
        result["client_seconds"] = time.monotonic() - started
        result["client_returncode"] = client.returncode
    finally:
        if server is not None:
            result["server_returncode"] = l0.stop_own_process(server)
        if record != "none":
            try:  # after the server stopped: its background writer has flushed (atexit)
                result["record_constraints"] = record_constraints_summary(run_dir, record)
            except OSError as exc:
                result["record_constraints"] = {"esdf": record, "error": repr(exc)}
            if not result["record_constraints"].get("n_chunk_files"):
                print(f"[{mode}] !!! --record-constraints {record}: no chunk file under "
                      f"{run_dir / RECORD_CONSTRAINTS_DIRNAME} !!!", flush=True)
        if lock is not None:
            result["gpu_lock_released_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        record_execution(run_dir, checkpoint=identity["path"], settings=result)
        print(json.dumps(result), flush=True)
    return run_dir


def episodes(run_dir: Path) -> dict:
    rows = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    return {(r["task_id"], r["episode_idx"]): r for r in rows}


def paired_summary(runs: dict) -> dict:
    a, b = episodes(runs["vla_only"]), episodes(runs["vla_v11ab"])
    rows, candidates = [], []
    for key in sorted(set(a) | set(b)):
        ra, rb = a.get(key, {}), b.get(key, {})
        row = {"task_id": key[0], "episode_idx": key[1], "task": ra.get("task_description") or rb.get("task_description"),
               "vla_only": {**{k: ra.get(k) for k in ("official_success", "status", "env_steps_excluding_wait", "video")},
                            **_safety(ra)},
               "vla_v11ab": {**{k: rb.get(k) for k in ("official_success", "status", "env_steps_excluding_wait", "video")},
                             **_safety(rb)},
               "v11ab_hold_calls": (rb.get("safe_client") or {}).get("hold_calls"),
               "v11ab_max_abs_action_change": (rb.get("safe_client") or {}).get("max_abs_action_change_executed")}
        rows.append(row)
        ok = "safe_success" if "safe_success" in row["vla_only"] else "official_success"
        if row["vla_only"].get(ok) is False and row["vla_v11ab"].get(ok) is True:
            candidates.append(row)
    out = {"runs": {m: str(p) for m, p in runs.items()}, "episodes": rows,
           "n": len(rows), "vla_only_success": sum(bool(r["vla_only"]["official_success"]) for r in rows),
           "vla_v11ab_success": sum(bool(r["vla_v11ab"]["official_success"]) for r in rows),
           "candidate_rule": "VLA-only fails and VLA+V11AB succeeds on {} (same task, init state, policy_seed)".format(
               "safe_success (official success and no obstacle collision)" if rows and "safe_success" in rows[0]["vla_only"]
               else "official success"),
           "candidates_vla_fail_v11ab_success": candidates}
    if rows and "safe_success" in rows[0]["vla_only"]:
        for m in ("vla_only", "vla_v11ab"):
            out[m + "_collided"] = sum(bool(r[m].get("collided")) for r in rows)
            out[m + "_safe_success"] = sum(bool(r[m].get("safe_success")) for r in rows)
    return out


def _safety(row: dict) -> dict:
    sl = row.get("safelibero")
    if not sl:
        return {}
    return {"collided": sl.get("collided"), "collide_t": sl.get("collide_t"), "safe_success": sl.get("safe_success"),
            "obstacle": sl.get("obstacle")}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task-suite-name", default="libero_spatial")
    ap.add_argument("--task-ids", type=lambda s: [int(x) for x in s.split(",")], required=True)
    ap.add_argument("--episode-ids", type=lambda s: [int(x) for x in s.split(",")], required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--step", default="SL5b")
    ap.add_argument("--data-split", default=DEFAULT_SPLIT)
    ap.add_argument("--checkpoint-dir", default=str(l0.default_checkpoint()))
    ap.add_argument("--capture-segmentation", action="store_true")
    ap.add_argument("--server-timeout", type=float, default=1800.0)
    ap.add_argument("--busy-gpu-wait", type=float, default=1800.0)
    ap.add_argument("--safety-level", choices=("I", "II"), default=None, help="SafeLIBERO suites only")
    ap.add_argument("--ag3s-version", default="V11AB",
                    help="AG3S version served in vla_v11ab mode (AG3S/adapters/panda/profile.py AG3S_VERSIONS)")
    ap.add_argument("--hsr6-probe", default=None, metavar="NPZ",
                    help="SL6-A — the Panda HSR6 probe for the final-configuration versions (default profile.PANDA_HSR6_PROBE, "
                         "SL6-B B4); refused by the other versions")
    ap.add_argument("--attention-source", default=None, metavar="SPEC",
                    help="SL6-A — replace the final-configuration version's --attention-source cell (SL6-B head "
                         "re-selection); final versions only")
    ap.add_argument("--conversion", choices=("A", "B", "C", "D", "E"), required=True,
                    help="Panda reverse conversion for VLA+V11AB (AG3S/adapters/panda/action.py CONVERSION_MODES)")
    ap.add_argument("--record-constraints", choices=RECORD_CONSTRAINTS_CHOICES, default="none",
                    help="vla_v11ab server only: serve_safe --record-constraints <run_dir>/server_constraints "
                         "--record-constraints-esdf <value> (per-call constraint npz). none = argv unchanged")
    ap.add_argument("--safelibero-step-budget", choices=SAFELIBERO_STEP_BUDGETS, default=SAFELIBERO_DEFAULT_STEP_BUDGET,
                    help="SafeLIBERO only: total_incl_wait (SL5c protocol, default; argv and settings unchanged) or "
                         "main_aegis (main_aegis.py as written: the total is the policy-step budget after the wait)")
    ap.add_argument("--dry-run", action="store_true")
    # protocol constants read by l0.protocol_settings / runtime_schedule
    ap.add_argument("--host", default="127.0.0.1")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for name, value in (("resize_size", 224), ("replan_steps", 5), ("num_steps_wait", 10)):
        setattr(args, name, getattr(args, name, value))
    if os.environ.get("AG3S_PROJECT_ROOT") != str(PROJECT_ROOT):
        raise RuntimeError("source scripts/setup/env.sh of this checkout first")
    if args.task_suite_name.startswith("safelibero") and args.data_split == DEFAULT_SPLIT:
        raise ValueError("SafeLIBERO has no fixed dev/test split yet (plan §6.0: decided before P3); "
                         "pass --data-split explicitly")
    if args.task_suite_name.startswith("safelibero") != (args.safety_level is not None):
        raise ValueError("--safety-level is required for SafeLIBERO suites and only for them")
    if args.data_split == DEFAULT_SPLIT and any(e > 4 for e in args.episode_ids):
        raise ValueError("dev split is LIBERO init states 0-4 (plan §6.0); pass --data-split for anything else")
    from AG3S.adapters.panda import profile as _profile   # SL6-A: refuse a bad version / probe before any GPU work

    _profile.version_server_flags(args.ag3s_version, **version_options(args))
    if args.safelibero_step_budget != SAFELIBERO_DEFAULT_STEP_BUDGET and not _is_safelibero(args):
        raise ValueError("--safelibero-step-budget applies to SafeLIBERO suites only")
    from benchmark.runners import suite_registry  # BM-SETUP-A: refuse ids outside the suite before any GPU work

    suite_registry.check_selection(args.task_suite_name, args.safety_level, args.task_ids, args.episode_ids,
                                   args.data_split)
    modes = [m for m in args.modes.split(",") if m]
    if args.record_constraints != "none" and "vla_only" in modes:
        raise ValueError(f"--record-constraints {args.record_constraints} cannot run with the vla_only mode "
                         "(--no-safe server, no constraints); pass --modes vla_v11ab")
    if _is_safelibero(args) and oracle_client_argv("vla_v11ab", args):
        raise ValueError("SL12: the oracle versions run on LIBERO and LIBERO-Safety only (the SafeLIBERO target choice is not "
                         "checked); remove this refusal after checking oracle_codes.choose_target on the SafeLIBERO BDDL")
    identity = l0.checkpoint_identity(Path(args.checkpoint_dir), require=True)
    if args.dry_run:
        print(json.dumps({m: server_argv(m, identity["path"], 0, args.ag3s_version, **version_options(args))
                          + record_constraints_argv(m, RUN_DIR_TOKEN, args.record_constraints) for m in modes},
                         indent=1))
        print(json.dumps(client_argv("<run>", 0, args) + oracle_client_argv("vla_v11ab", args)))
        print(json.dumps({"protocol": protocol_settings(args), "runtime_schedule": runtime_schedule(args)}, default=str))
        return 0
    if gpu_lock_held_by_caller():
        try:  # refuse before any run directory is allocated
            check_caller_holds_gpu_lock()
        except GpuLockNotHeld as exc:
            print(f"refused: {exc}", file=sys.stderr, flush=True)
            return 2
    shared = {"versions": versions()}
    runs = {m: run_mode(m, args, identity, shared) for m in modes}
    if set(runs) == set(MODES):
        summary = paired_summary(runs)
        with open(runs["vla_v11ab"] / "paired_summary.json", "x") as f:
            json.dump(summary, f, indent=1)
        print(json.dumps({k: summary[k] for k in ("n", "vla_only_success", "vla_v11ab_success")}), flush=True)
        print("candidates:", json.dumps(summary["candidates_vla_fail_v11ab_success"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
