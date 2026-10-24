"""Launch one closed-loop method arm on LIBERO / SafeLIBERO (``.venv-ag3s``): run directory, method server, the
existing client, metrics.

Same structure as ``benchmark/runners/launch_libero_paired.py:run_mode`` and built from its helpers (called by
keyword, arguments taken from its own parser so later options keep their defaults):

1. ``common.run_artifacts.create_run`` -> ``results/benchmark/<date>_<commit>_<run-id>-L<level>-<arm>/`` with the arm
   record in the manifest (method, variant, privileged inputs, deviations from the paper);
2. the shared GPU lock (``fcntl.flock`` on ``/tmp/ag3s-vla-gpu-validation.lock``; or ``AG3S_GPU_LOCK_HELD=1`` with an
   inherited descriptor, like ``run_parallel_gpu.sh``) and no other GPU compute process -- skipped with ``--cpu-smoke``;
3. ``serve_method`` (``.venv-openpi-live``) on a free port, ``AG3S_RUN_DIR`` set (it logs ``method_calls/``);
4. the **unchanged** client ``benchmark/runners/libero_safe_client.py`` (``.venv-libero``) with
   ``launch_libero_paired.client_argv`` -> same tasks / init states / seed / per-episode ``policy_seed`` as SL5c;
5. server stopped (SIGINT), ``closed_loop_metrics.json`` (``metrics.summarize``, optionally paired with ``--ref-run``).

    source scripts/setup/env.sh
    .venv-ag3s/bin/python Baseline/paper_baselines/closed_loop/launch_method.py --method aegis \
        --task-suite-name safelibero_spatial --safety-level I --task-ids 0,1,2,3 --episode-ids 0,1,2,3,4 \
        --run-id clm-pilot --sidecar-device cuda

``--cpu-smoke`` (no GPU, no lock, no checkpoint): ``--fake-policy constant`` or ``--fake-policy recorded
--fake-source-run <run>``; JAX on CPU, sidecar on CPU, one thread per process.
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Baseline.paper_baselines.closed_loop import arms as A  # noqa: E402
from benchmark.runners import launch_libero_paired as lp  # noqa: E402
from benchmark.runners import launch_libero_pi05_only as l0  # noqa: E402

SERVER_MODULE = "Baseline.paper_baselines.closed_loop.serve_method"


def paired_namespace(args) -> argparse.Namespace:
    """``launch_libero_paired`` arguments through its own parser (defaults of any later option included)."""
    argv = ["--task-suite-name", args.task_suite_name, "--task-ids", ",".join(map(str, args.task_ids)),
            "--episode-ids", ",".join(map(str, args.episode_ids)), "--seed", str(args.seed), "--run-id", args.run_id,
            "--conversion", "A", "--data-split", args.data_split, "--step", args.step,
            "--safelibero-step-budget", args.safelibero_step_budget]
    if args.safety_level:
        argv += ["--safety-level", args.safety_level]
    ns = lp.build_parser().parse_args(argv)
    for name, value in (("resize_size", 224), ("replan_steps", 5), ("num_steps_wait", 10)):  # = lp.main
        setattr(ns, name, getattr(ns, name, value))
    return ns


def check_selection(args) -> None:
    from benchmark.runners import suite_registry

    safe = args.task_suite_name.startswith("safelibero")
    if safe != (args.safety_level is not None):
        raise ValueError("--safety-level is required for SafeLIBERO suites and only for them")
    if args.safelibero_step_budget != lp.SAFELIBERO_DEFAULT_STEP_BUDGET and not safe:
        raise ValueError("--safelibero-step-budget applies to SafeLIBERO suites only")
    if args.data_split == lp.DEFAULT_SPLIT and any(e > 4 for e in args.episode_ids):
        raise ValueError("dev split is init states 0-4; pass --data-split for anything else")
    suite_registry.check_selection(args.task_suite_name, args.safety_level, args.task_ids, args.episode_ids,
                                   args.data_split)


def server_argv(args, port: int, checkpoint: str | None) -> list[str]:
    argv = [str(l0.openpi_python()), "-m", SERVER_MODULE, "--method", args.method, "--port", str(port),
            "--config", args.config, "--guidance-exec", args.guidance_exec, "--sidecar-device", args.sidecar_device,
            "--detector-weight", args.detector_weight]
    if checkpoint:
        argv += ["--checkpoint", checkpoint]
    for flag, value in (("--obstacles", args.obstacles), ("--target", args.target),
                        ("--method-kwargs", args.method_kwargs), ("--scene-kwargs", args.scene_kwargs)):
        if value is not None:
            argv += [flag, value]
    if args.fake_policy != "none":
        argv += ["--fake-policy", args.fake_policy]
        if args.fake_source_run:
            from Baseline.paper_baselines.closed_loop.replay import resolve_run

            argv += ["--fake-source-run", str(resolve_run(args.fake_source_run))]
    if args.no_record_nominal:
        argv.append("--no-record-nominal")
    return argv


def server_env(args, run_dir: Path) -> dict:
    env = dict(os.environ, AG3S_RUN_DIR=str(run_dir))
    if args.cpu_smoke:
        env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", XLA_FLAGS="--xla_force_host_platform_device_count=1")
    else:
        env.setdefault("OPENPI_DATA_HOME", str(PROJECT_ROOT / ".cache/openpi"))
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        env.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return env


def run_tag(args) -> str:
    tag = args.method if not args.tag else f"{args.method}-{args.tag}"
    return f"L{args.safety_level}-{tag}" if args.safety_level else tag


def acquire_gpu(args, result: dict):
    if args.cpu_smoke:
        result["gpu_lock_mode"] = "none (cpu smoke)"
        return None
    if lp.gpu_lock_held_by_caller():
        result.update(lp.check_caller_holds_gpu_lock())
        return None
    lock = open(lp.GPU_LOCK, "a")
    t = time.monotonic()
    fcntl.flock(lock, fcntl.LOCK_EX)
    result["gpu_lock_wait_s"] = time.monotonic() - t
    result["gpu_lock_acquired_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    deadline = time.monotonic() + args.busy_gpu_wait
    while lp.gpu_compute_apps():
        if time.monotonic() > deadline:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
            raise RuntimeError(f"other GPU compute processes still running: {lp.gpu_compute_apps()}")
        time.sleep(15)
    return lock


def run(args) -> Path:
    from Baseline.paper_baselines.closed_loop import metrics
    from common.run_artifacts import create_run, record_execution

    arm = A.resolve(args.method, args.obstacles, args.target)
    pargs = paired_namespace(args)
    identity = l0.checkpoint_identity(Path(args.checkpoint_dir), require=not args.cpu_smoke)
    ckpt = identity.get("path") if identity else None
    port = l0.choose_port(None)
    s_argv = server_argv(args, port, ckpt)  # cpu smoke: norm stats only
    settings = {"step": args.step, "condition": f"method:{arm['variant']}", "arm": arm, "port": port,
                "protocol": lp.protocol_settings(pargs), "runtime_schedule": lp.runtime_schedule(pargs),
                "task_ids": args.task_ids, "episode_ids": args.episode_ids, "seed": args.seed,
                "safety_level": args.safety_level, "data_split": args.data_split,
                "policy_seed_rule": "seed*100000 + task_id*1000 + episode_idx (libero_safe_client.episode_seed)",
                "checkpoint": identity, "server_argv": s_argv, "cpu_smoke": bool(args.cpu_smoke),
                "fake_policy": args.fake_policy, "fake_source_run": args.fake_source_run,
                "action_convention": "pi05_libero EE-delta chunk H=10, client executes rows 0-4 (replan 5) as VLA-only "
                                     "/ V11AB", "versions": lp.versions()}
    run_dir = create_run(run_id=f"{args.run_id}-{run_tag(args)}", settings=settings,
                         checkpoint=None if args.cpu_smoke else ckpt)
    print(f"[{arm['variant']}] {run_dir}", flush=True)
    c_argv = lp.client_argv(run_dir=str(run_dir), port=port, args=pargs)
    logs = run_dir / "logs"
    result = {"role": "launcher_result", "arm": arm["variant"], "port": port}
    server, lock = None, None
    try:
        lock = acquire_gpu(args, result)
        with open(logs / "server_console.log", "x") as server_log:
            server = subprocess.Popen(s_argv, cwd=str(PROJECT_ROOT), env=server_env(args, run_dir), stdout=server_log,
                                      stderr=subprocess.STDOUT)
        record_execution(run_dir, checkpoint=ckpt, settings={"role": "server", "argv": s_argv, "pid": server.pid})
        result["server_ready_seconds"] = l0.wait_ready(server, port, args.server_timeout)
        print(f"[{arm['variant']}] server ready on {port} after {result['server_ready_seconds']:.0f}s", flush=True)
        cenv = l0.client_environment()
        if args.cpu_smoke:
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "LP_NUM_THREADS"):
                cenv[k] = "1"
            cenv["CUDA_VISIBLE_DEVICES"] = ""
        record_execution(run_dir, checkpoint=ckpt, settings={"role": "client", "argv": c_argv})
        started = time.monotonic()
        with open(logs / "libero_client.log", "x") as client_log:
            client = subprocess.run(c_argv, cwd=str(run_dir), env=cenv, stdout=client_log, stderr=subprocess.STDOUT)
        result["client_seconds"] = time.monotonic() - started
        result["client_returncode"] = client.returncode
    finally:
        if server is not None:
            result["server_returncode"] = l0.stop_own_process(server)
        if lock is not None:
            result["gpu_lock_released_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        record_execution(run_dir, checkpoint=ckpt, settings=result)
        print(json.dumps(result), flush=True)
    if (run_dir / "episodes.jsonl").exists():
        summ = metrics.summarize(run_dir, args.ref_run)
        with open(run_dir / "closed_loop_metrics.json", "x") as f:
            json.dump(summ, f, indent=1)
        brief = {k: summ[k] for k in ("safelibero_official", "sl5c") if k in summ}
        print(json.dumps({k: {kk: v.get(kk) for kk in ("n", "TSR", "CAR", "ETS", "official_success", "collided",
                                                         "safe_success") if kk in v} for k, v in brief.items()}))
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", required=True, choices=sorted(A.ARMS))
    ap.add_argument("--obstacles", choices=A.OBSTACLE_SOURCES, default=None)
    ap.add_argument("--target", choices=A.TARGET_SOURCES, default=None)
    ap.add_argument("--tag", default=None, help="extra run-id suffix (e.g. obst-depth)")
    ap.add_argument("--task-suite-name", default="safelibero_spatial")
    ap.add_argument("--task-ids", type=lambda s: [int(x) for x in s.split(",")], required=True)
    ap.add_argument("--episode-ids", type=lambda s: [int(x) for x in s.split(",")], required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--safety-level", choices=("I", "II"), default=None)
    ap.add_argument("--data-split", default=None, help="default: suite_registry dev label of the suite / level")
    ap.add_argument("--safelibero-step-budget", choices=lp.SAFELIBERO_STEP_BUDGETS,
                    default=lp.SAFELIBERO_DEFAULT_STEP_BUDGET)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--step", default="CLOSED-LOOP-METHODS")
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--checkpoint-dir", default=str(l0.default_checkpoint()))
    ap.add_argument("--guidance-exec", choices=("callback", "inline"), default="callback")
    ap.add_argument("--sidecar-device", default="cuda", help="Grounding DINO device in the sidecar (cpu with --cpu-smoke)")
    ap.add_argument("--detector-weight", default="grounding-dino-tiny")
    ap.add_argument("--method-kwargs", default=None)
    ap.add_argument("--scene-kwargs", default=None)
    ap.add_argument("--no-record-nominal", action="store_true")
    ap.add_argument("--ref-run", default=None, help="reference run (same tasks / episodes) for paired metrics")
    ap.add_argument("--cpu-smoke", action="store_true")
    ap.add_argument("--fake-policy", choices=("none", "constant", "recorded"), default="none")
    ap.add_argument("--fake-source-run", default=None)
    ap.add_argument("--server-timeout", type=float, default=1800.0)
    ap.add_argument("--busy-gpu-wait", type=float, default=1800.0)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if os.environ.get("AG3S_PROJECT_ROOT") != str(PROJECT_ROOT):
        raise RuntimeError("source scripts/setup/env.sh of this checkout first")
    if args.data_split is None:
        from benchmark.runners import suite_registry

        args.data_split = (suite_registry.split_label(args.task_suite_name, args.safety_level, "dev")
                           if args.task_suite_name in suite_registry.suite_names() else lp.DEFAULT_SPLIT)
    if args.cpu_smoke:
        if args.fake_policy == "none":
            raise ValueError("--cpu-smoke needs --fake-policy constant|recorded (no checkpoint on CPU)")
        args.sidecar_device = "cpu"
    elif args.fake_policy != "none":
        raise ValueError("--fake-policy is a CPU smoke option (add --cpu-smoke)")
    check_selection(args)
    A.resolve(args.method, args.obstacles, args.target)
    if args.dry_run:
        pargs = paired_namespace(args)
        print(json.dumps({"server": server_argv(args, 0, "<checkpoint>"),
                          "client": lp.client_argv(run_dir="<run>", port=0, args=pargs),
                          "arm": A.resolve(args.method, args.obstacles, args.target),
                          "protocol": lp.protocol_settings(pargs)}, indent=1, default=str))
        return 0
    if not args.cpu_smoke and lp.gpu_lock_held_by_caller():
        lp.check_caller_holds_gpu_lock()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
