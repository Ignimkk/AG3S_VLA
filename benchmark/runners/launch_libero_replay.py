"""SL1 replay launcher: re-run recorded LIBERO episodes without a policy (run under .venv-ag3s).

Allocates a new run with ``common.run_artifacts.create_run`` (the launcher's path), records the
replay source (run id, path, manifest/npz SHA-256) in the manifest, and runs
``benchmark/runners/libero_replay.py`` under ``.venv-libero`` with the same sanitized environment
as the SL0 client. No GPU, no server, no port. ``--dry-run`` prints the command and settings only.

Example (SL1 geometry replay of the SL0 episode)::

    .venv-ag3s/bin/python benchmark/runners/launch_libero_replay.py \\
        --source-run 20261012T045129_6ba81024d243_sl0-pi05only-t0e0 --task-ids 0 --episode-ids 0 \\
        --capture-geometry --run-id sl1-replay-geom-t0e0
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common.workspace import PROJECT_ROOT, shared_storage_root
from benchmark.runners import launch_libero_pi05_only as launcher
from benchmark.runners import libero_pi05_client as client_mod

REPLAY_SCRIPT = PROJECT_ROOT / "benchmark/runners/libero_replay.py"
CONDITION = "replay (no policy): executed actions of the source run"


def resolve_source_run(value: str) -> Path:
    """A run directory below results/benchmark (absolute path or bare run name), read-only."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and len(candidate.parts) == 1:
        candidate = shared_storage_root() / "results/benchmark" / candidate
    return client_mod.validate_run_dir(candidate)


def source_record(source: Path, task_ids, episode_ids) -> dict:
    manifest = json.loads((source / "run_manifest.json").read_text())
    rows = {}
    for line in (source / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(int(row["task_id"]), int(row["episode_idx"]))] = row
    episodes = []
    for task_id in task_ids:
        for episode_idx in episode_ids:
            row = rows.get((task_id, episode_idx))
            if row is None:
                raise ValueError(f"source run has no episode task {task_id} / episode {episode_idx}: {source}")
            npz = (source / row["npz"]).resolve()
            episodes.append({"task_id": task_id, "episode_idx": episode_idx, "npz": row["npz"],
                             "npz_sha256": launcher.sha256_file(npz), "status": row.get("status"),
                             "policy_calls": row.get("policy_calls"),
                             "env_steps_including_wait": row.get("env_steps_including_wait")})
    protocol = manifest.get("settings", {}).get("protocol", {})
    return {"run_dir": str(source), "run_id": source.name, "commit": manifest.get("commit"),
            "branch": manifest.get("branch"), "condition": manifest.get("settings", {}).get("condition"),
            "manifest_sha256": launcher.sha256_file(source / "run_manifest.json"),
            "episodes_jsonl_sha256": launcher.sha256_file(source / "episodes.jsonl"),
            "protocol": {key: protocol.get(key) for key in ("suite", "seed", "num_steps_wait", "replan_steps",
                                                             "resize_size", "render_resolution", "max_steps")},
            "episodes": episodes}


def replay_command(python: Path, run_dir: str, source: Path, args) -> list[str]:
    return [str(python), str(REPLAY_SCRIPT), "--run-dir", str(run_dir), "--source-run-dir", str(source),
            "--task-suite-name", args.task_suite_name,
            "--task-ids", ",".join(map(str, args.task_ids)),
            "--episode-ids", ",".join(map(str, args.episode_ids)),
            "--seed", str(args.seed)] + launcher.geometry_flags(args)


def build_settings(args, source: dict, replay_cmd, *, probe: bool = True) -> dict:
    protocol = source["protocol"]
    mismatch = {key: (getattr(args, key if key != "suite" else "task_suite_name"), protocol[key])
                for key in ("suite", "seed") if protocol.get(key) is not None
                and protocol[key] != getattr(args, key if key != "suite" else "task_suite_name")}
    if mismatch:
        raise ValueError(f"replay protocol differs from the source run: {mismatch}")
    return {
        "step": "SL1",
        "condition": CONDITION,
        "ag3s": {"enabled": False, "path": "no policy server; recorded action chunks are executed"},
        "replay_source": source,
        "protocol": launcher.protocol_settings(args),
        "privileged_inputs": "none (actions are the source run's executed actions)",
        "camera": {"names": ["agentview", "robot0_eye_in_hand"], "height": client_mod.LIBERO_ENV_RESOLUTION,
                   "width": client_mod.LIBERO_ENV_RESOLUTION},
        "robot": {"name": "Panda", "controller": "OSC_POSE (LIBERO ControlEnv default)"},
        "runtime_schedule": launcher.runtime_schedule(args),
        "not_applicable": launcher.not_applicable_items(),
        "geometry_capture": launcher.geometry_settings(args),
        "interpreters": {
            "libero": (launcher.interpreter_info(launcher.libero_python(), launcher.LIBERO_PACKAGES) if probe
                       else str(launcher.libero_python())),
            "launcher": sys.executable,
        },
        "environment": {key: launcher.client_environment().get(key) for key in
                        ["MUJOCO_GL", "LIBERO_CONFIG_PATH", "LD_LIBRARY_PATH", *launcher.THREAD_LIMIT_KEYS]}
                       | {"AG3S_STORAGE_ROOT": os.environ.get("AG3S_STORAGE_ROOT")},
        "commands": {"replay": replay_cmd},
    }


def run(args) -> int:
    from common.run_artifacts import create_run, record_execution

    if os.environ.get("AG3S_PROJECT_ROOT") != str(PROJECT_ROOT):
        raise RuntimeError("source scripts/setup/env.sh of this checkout first")
    if not launcher.libero_python().exists():
        raise FileNotFoundError(f"interpreter missing: {launcher.libero_python()}")
    source_dir = resolve_source_run(args.source_run)
    source = source_record(source_dir, args.task_ids, args.episode_ids)
    settings = build_settings(args, source, replay_command(launcher.libero_python(), "<run>", source_dir, args))
    run_dir = create_run(run_id=args.run_id, settings=settings, checkpoint=None)
    cmd = replay_command(launcher.libero_python(), str(run_dir), source_dir, args)
    print(run_dir, flush=True)

    environment = launcher.client_environment()
    record_execution(run_dir, settings={
        "role": "replay_client", "argv": cmd, "cwd": str(run_dir),
        "environment": {key: environment.get(key) for key in
                        ["MUJOCO_GL", "LIBERO_CONFIG_PATH", "PYTHONPATH", "PYOPENGL_PLATFORM", "LD_LIBRARY_PATH",
                         *launcher.THREAD_LIMIT_KEYS]},
        "original_LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH")})
    result = {"role": "launcher_result"}
    started = time.monotonic()
    try:
        with open(run_dir / "logs" / "libero_replay.log", "x") as log:
            completed = subprocess.run(cmd, cwd=str(run_dir), env=environment, stdout=log, stderr=subprocess.STDOUT)
        result["replay_returncode"] = completed.returncode
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["replay_seconds"] = time.monotonic() - started
        record_execution(run_dir, settings=result)
        print(json.dumps(result), flush=True)
    return 0 if result.get("replay_returncode") == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-run", required=True, help="run name below results/benchmark or its absolute path")
    parser.add_argument("--task-suite-name", default=client_mod.DEFAULT_TASK_SUITE, choices=sorted(client_mod.MAX_STEPS))
    parser.add_argument("--task-ids", type=client_mod._int_list, default=[0])
    parser.add_argument("--episode-ids", type=client_mod._int_list, default=[0])
    parser.add_argument("--seed", type=int, default=client_mod.DEFAULT_SEED)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true", help="print command and settings; allocate nothing")
    client_mod.add_geometry_arguments(parser)
    return parser


def dry_run(args) -> dict:
    source_dir = resolve_source_run(args.source_run)
    source = source_record(source_dir, args.task_ids, args.episode_ids)
    placeholder = str(shared_storage_root() / "results/benchmark/<KST_commit_run-id>")
    return build_settings(args, source, replay_command(launcher.libero_python(), placeholder, source_dir, args))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    client_mod.check_geometry_arguments(args)
    if args.dry_run:
        print(json.dumps(dry_run(args), indent=2, ensure_ascii=False))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
