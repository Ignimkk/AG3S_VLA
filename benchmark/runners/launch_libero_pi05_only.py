"""SL0 launcher: pi0.5-only LIBERO episodes with a storage checkpoint (run under .venv-ag3s).

Allocates a new run with ``common.run_artifacts.create_run``, picks a free port (never 8123),
starts the OpenPI policy server from ``.venv-openpi-live`` with ``AG3S_RUN_DIR=<run>``, waits for
``/healthz``, runs ``benchmark/runners/libero_pi05_client.py`` under ``.venv-libero`` and finally
stops only the server process it started. ``--dry-run`` prints both commands and the settings
without allocating a run or starting anything.

The policy path is ``serve_policy.py policy:checkpoint`` -- plain OpenPI inference; no AG3S
safety layer / trajectory-optimisation wrapper is involved (condition ``pi0.5-only``).
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time

from common.workspace import PROJECT_ROOT, shared_storage_root
from benchmark.runners import libero_pi05_client as client_mod

CONDITION = "pi0.5-only"
POLICY_CONFIG = "pi05_libero"
FORBIDDEN_PORTS = frozenset({8123})
NORM_STATS_RELATIVE = "assets/physical-intelligence/libero/norm_stats.json"
CLIENT_SCRIPT = PROJECT_ROOT / "benchmark/runners/libero_pi05_client.py"
SERVER_SCRIPT = PROJECT_ROOT / "third_party/openpi/scripts/serve_policy.py"
LIBERO_PACKAGES = ("robosuite", "mujoco", "numpy", "torch", "imageio", "imageio-ffmpeg", "opencv-python",
                   "openpi-client", "libero", "websockets", "bddl")
OPENPI_PACKAGES = ("jax", "jaxlib", "flax", "openpi", "openpi-client", "numpy", "websockets", "orbax-checkpoint")
CLIENT_THREADS = 4
THREAD_LIMIT_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS",
                     "LP_NUM_THREADS")
OBSERVATION_KEYS = ("observation/image", "observation/wrist_image", "observation/state", "prompt")


def default_checkpoint() -> Path:
    return shared_storage_root() / "checkpoints/panda/pi05_libero"


def libero_python() -> Path:
    return PROJECT_ROOT / ".venv-libero/bin/python"


def openpi_python() -> Path:
    return Path(os.environ.get("AG3S_OPENPI_PYTHON", PROJECT_ROOT / ".venv-openpi-live/bin/python"))


# --- Port ----------------------------------------------------------------------------------------
def port_is_free(port: int, host: str = "0.0.0.0") -> bool:
    """True if ``port`` can be bound now (the server binds 0.0.0.0)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _random_candidates(count: int = 200):
    rng = random.SystemRandom()
    for _ in range(count):
        yield rng.randint(20000, 29999)


def choose_port(requested: int | None = None, candidates=None) -> int:
    """Return a bindable port that is not forbidden. A requested port is checked, never replaced."""
    if requested is not None:
        if requested in FORBIDDEN_PORTS:
            raise ValueError(f"port {requested} is reserved and must not be used")
        if not port_is_free(requested):
            raise ValueError(f"port {requested} is already in use")
        return requested
    for port in (candidates if candidates is not None else _random_candidates()):
        if port in FORBIDDEN_PORTS or not 1024 <= port <= 65535:
            continue
        if port_is_free(port):
            return port
    raise RuntimeError("no free port found")


# --- Checkpoint identity -------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_records(checksums: dict) -> dict[str, dict]:
    """CHECKSUMS.json ``records`` list ({path, bytes, sha256, ...}) as {relative path: record}."""
    return {item["path"]: item for item in checksums.get("records", [])}


def aggregate_sha256(records: dict[str, dict]) -> str:
    """Aggregate of CHECKSUMS.json: sha256 over '<sha256>  <path>\\n' lines ordered by *path*.

    (Records are sorted by path, not by line text; this reproduces the recorded value.)
    """
    lines = (f"{records[path]['sha256']}  {path}\n" for path in sorted(records))
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def checkpoint_identity(directory: Path, *, require: bool = True, rehash: bool = False) -> dict:
    """Read-only identity from CHECKSUMS.json.

    Always: aggregate recomputed from the recorded per-file SHA-256 (internal consistency), listed
    files present with recorded byte size, file count, norm_stats SHA-256 hashed from disk.
    ``rehash``: additionally hash every file (12 GB; the verifier's identity check does this).
    """
    directory = Path(directory).expanduser().resolve()
    info = {"path": str(directory), "exists": directory.is_dir(),
            "partial_sibling_exists": directory.with_name(directory.name + ".partial").exists()}
    checksums_path = directory / "CHECKSUMS.json"
    norm_stats = directory / NORM_STATS_RELATIVE
    if not checksums_path.is_file() or not norm_stats.is_file():
        info["error"] = f"missing {'CHECKSUMS.json' if not checksums_path.is_file() else NORM_STATS_RELATIVE}"
        if require:
            raise FileNotFoundError(f"checkpoint not ready: {directory} ({info['error']})")
        return info
    checksums = json.loads(checksums_path.read_text())
    records = _file_records(checksums)
    bad = []
    for name, record in records.items():
        target = directory / name
        if not target.is_file() or ("bytes" in record and target.stat().st_size != int(record["bytes"])):
            bad.append(name)
        elif rehash and sha256_file(target) != record.get("sha256"):
            bad.append(name)
    norm_sha = sha256_file(norm_stats)
    recorded_norm = records.get(NORM_STATS_RELATIVE, {}).get("sha256")
    recomputed = aggregate_sha256(records) if records else None
    info.update({
        "source": checksums.get("source"),
        "checksums_json": str(checksums_path),
        "checksums_json_sha256": sha256_file(checksums_path),
        "aggregate_sha256": checksums.get("aggregate_sha256"),
        "aggregate_rule": checksums.get("aggregate_rule"),
        "aggregate_recomputed_from_records": recomputed,
        "file_count": len(records),
        "file_count_declared": checksums.get("files"),
        "total_bytes": checksums.get("total_bytes"),
        "files_rehashed": rehash,
        "size_or_hash_mismatch_or_missing": bad,
        "norm_stats": str(norm_stats),
        "norm_stats_sha256": norm_sha,
        "norm_stats_matches_checksums": None if recorded_norm is None else recorded_norm == norm_sha,
    })
    failed = (bad or recorded_norm is None or recorded_norm != norm_sha or not info["aggregate_sha256"]
              or recomputed != info["aggregate_sha256"]
              or (info["file_count_declared"] is not None and info["file_count_declared"] != len(records)))
    if require and failed:
        raise ValueError(f"checkpoint identity check failed: {info}")
    if require and info["partial_sibling_exists"]:
        raise ValueError(f"download still in progress ({directory.name}.partial exists)")
    return info


# --- Interpreters --------------------------------------------------------------------------------
_PROBE = (
    "import importlib.metadata as m, json, os, platform, sys\n"
    "out = {'executable': sys.executable, 'realpath': os.path.realpath(sys.executable),"
    " 'base_prefix': os.path.realpath(sys.base_prefix), 'version': platform.python_version(), 'packages': {}}\n"
    "for name in sys.argv[1:]:\n"
    "    try: out['packages'][name] = m.version(name)\n"
    "    except m.PackageNotFoundError: out['packages'][name] = None\n"
    "print(json.dumps(out))\n"
)


def interpreter_info(python: Path, packages) -> dict:
    python = Path(python)
    if not python.exists():
        return {"executable": str(python), "exists": False}
    try:
        result = subprocess.run([str(python), "-I", "-c", _PROBE, *packages], capture_output=True, text=True,
                                timeout=120, check=True)
        info = json.loads(result.stdout.strip().splitlines()[-1])
    except (subprocess.SubprocessError, ValueError, IndexError) as error:
        return {"executable": str(python), "exists": True, "probe_error": repr(error)}
    info["exists"] = True
    return info


# --- Commands and settings -----------------------------------------------------------------------
def server_command(python: Path, port: int, checkpoint: Path) -> list[str]:
    return [str(python), str(SERVER_SCRIPT), "--port", str(port), "--record",
            "policy:checkpoint", "--policy.config", POLICY_CONFIG, "--policy.dir", str(checkpoint)]


def client_command(python: Path, run_dir: str, port: int, args) -> list[str]:
    return [str(python), str(CLIENT_SCRIPT), "--run-dir", str(run_dir), "--host", args.host, "--port", str(port),
            "--task-suite-name", args.task_suite_name,
            "--task-ids", ",".join(map(str, args.task_ids)),
            "--episode-ids", ",".join(map(str, args.episode_ids)),
            "--seed", str(args.seed)]


def sanitize_library_path(value: str | None) -> str:
    """Drop LD_LIBRARY_PATH entries inside a foreign Python's site/dist-packages.

    The pod exports e.g. /usr/local/lib/python3.10/dist-packages/torch/lib, which makes the venv's
    torch 1.11 load a foreign libtorch_python.so (undefined symbol). Driver/CUDA entries are kept.
    """
    kept = []
    for entry in (value or "").split(":"):
        if not entry:
            continue
        foreign = ("/site-packages/" in entry + "/" or "/dist-packages/" in entry + "/") and not (
            entry + "/").startswith(str(PROJECT_ROOT) + "/")
        if not foreign:
            kept.append(entry)
    return ":".join(kept)


def client_environment(base: dict | None = None) -> dict:
    env = dict(os.environ if base is None else base)
    # The py3.8 client relies on its own editable installs; the checkout PYTHONPATH (py3.11 sources) is dropped.
    env.pop("PYTHONPATH", None)
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = sanitize_library_path(env["LD_LIBRARY_PATH"])
    env["MUJOCO_GL"] = env.get("MUJOCO_GL") or "osmesa"
    # Shared CPU pod: cap BLAS/torch/numba/llvmpipe(OSMesa) worker threads unless already set.
    for key in THREAD_LIMIT_KEYS:
        env.setdefault(key, str(CLIENT_THREADS))
    env.setdefault("LIBERO_CONFIG_PATH", str(PROJECT_ROOT / ".runtime/libero"))
    return env


def protocol_settings(args) -> dict:
    return {
        "suite": args.task_suite_name,
        "task_ids": list(args.task_ids),
        "episode_ids": list(args.episode_ids),
        "init_state_index": "episode_idx",
        "seed": args.seed,
        "num_steps_wait": client_mod.DEFAULT_NUM_STEPS_WAIT,
        "replan_steps": client_mod.DEFAULT_REPLAN_STEPS,
        "resize_size": client_mod.DEFAULT_RESIZE_SIZE,
        "render_resolution": client_mod.LIBERO_ENV_RESOLUTION,
        "max_steps": client_mod.MAX_STEPS[args.task_suite_name],
        "dummy_action": list(client_mod.LIBERO_DUMMY_ACTION),
        "video_fps": client_mod.VIDEO_FPS,
        "image_preprocessing": "rotate 180 deg ([::-1, ::-1]), resize_with_pad to resize_size, uint8",
        "state": "eef_pos(3) + quat2axisangle(eef_quat)(3) + gripper_qpos(2)",
        "action": "EE delta 6 + gripper 1 (OSC_POSE), first replan_steps of each chunk",
        "metric": {"official_success": "done returned by env.step() (official example)",
                   "status": list(client_mod.STATUS_VALUES)},
        "reference": "third_party/openpi/examples/libero/main.py",
    }


def build_settings(args, port: int, checkpoint: dict, server_cmd, client_cmd, *, probe: bool = True) -> dict:
    return {
        "step": "SL0",
        "condition": CONDITION,
        "ag3s": {"enabled": False,
                 "path": "serve_policy.py policy:checkpoint (plain OpenPI policy; no AG3S safety layer, no TO wrapper)"},
        "policy_config": POLICY_CONFIG,
        "protocol": protocol_settings(args),
        "checkpoint": checkpoint,
        "port": {"selected": port, "forbidden": sorted(FORBIDDEN_PORTS), "host": args.host},
        "privileged_inputs": "none",
        "observation_keys": list(OBSERVATION_KEYS),
        "camera": {"names": ["agentview", "robot0_eye_in_hand"], "height": client_mod.LIBERO_ENV_RESOLUTION,
                   "width": client_mod.LIBERO_ENV_RESOLUTION},
        "robot": {"name": "Panda", "controller": "OSC_POSE (LIBERO ControlEnv default)"},
        "interpreters": {
            "libero": interpreter_info(libero_python(), LIBERO_PACKAGES) if probe else str(libero_python()),
            "openpi_live": interpreter_info(openpi_python(), OPENPI_PACKAGES) if probe else str(openpi_python()),
            "launcher": sys.executable,
        },
        "environment": {**{key: client_environment().get(key) for key in
                           ["MUJOCO_GL", "LIBERO_CONFIG_PATH", "LD_LIBRARY_PATH", *THREAD_LIMIT_KEYS]},
                        **{key: os.environ.get(key) for key in
                           ["XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "OPENPI_DATA_HOME", "AG3S_STORAGE_ROOT"]}},
        "commands": {"server": server_cmd, "client": client_cmd},
    }


# --- Process control -----------------------------------------------------------------------------
def healthz(port: int, timeout: float = 2.0) -> bool:
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        connection.request("GET", "/healthz")
        ok = connection.getresponse().status == 200
        connection.close()
        return ok
    except (OSError, http.client.HTTPException):
        return False


def wait_ready(process: subprocess.Popen, port: int, timeout: float) -> float:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"policy server exited early with code {process.returncode}")
        if healthz(port):
            return time.monotonic() - start
        time.sleep(2.0)
    raise TimeoutError(f"policy server not ready after {timeout:.0f}s")


def stop_own_process(process: subprocess.Popen) -> int | None:
    """Stop exactly the process this launcher started (SIGINT -> SIGTERM -> SIGKILL)."""
    if process.poll() is not None:
        return process.returncode
    for sig, wait in ((signal.SIGINT, 60), (signal.SIGTERM, 30), (signal.SIGKILL, 30)):
        process.send_signal(sig)
        try:
            return process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
    return process.poll()


def run(args) -> int:
    from common.run_artifacts import create_run, record_execution

    if os.environ.get("AG3S_PROJECT_ROOT") != str(PROJECT_ROOT):
        raise RuntimeError("source scripts/setup/env.sh of this checkout first")
    for python in (libero_python(), openpi_python()):
        if not python.exists():
            raise FileNotFoundError(f"interpreter missing: {python}")
    identity = checkpoint_identity(Path(args.checkpoint_dir), require=True, rehash=args.verify_checksums)
    port = choose_port(args.port)
    ckpt = identity["path"]
    server_cmd = server_command(openpi_python(), port, Path(ckpt))
    settings = build_settings(args, port, identity, server_cmd, client_command(libero_python(), "<run>", port, args))
    run_dir = create_run(run_id=args.run_id, settings=settings, checkpoint=ckpt)
    client_cmd = client_command(libero_python(), str(run_dir), port, args)
    print(run_dir, flush=True)

    server_env = dict(os.environ, AG3S_RUN_DIR=str(run_dir))
    logs = run_dir / "logs"
    result = {"role": "launcher_result", "port": port}
    server = None
    try:
        with open(logs / "server_console.log", "x") as server_log:
            server = subprocess.Popen(server_cmd, cwd=str(PROJECT_ROOT), env=server_env,
                                      stdout=server_log, stderr=subprocess.STDOUT)
        record_execution(run_dir, checkpoint=ckpt, settings={
            "role": "server", "argv": server_cmd, "pid": server.pid, "cwd": str(PROJECT_ROOT),
            "environment": {key: server_env.get(key) for key in
                            ["AG3S_RUN_DIR", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "OPENPI_DATA_HOME",
                             "PYTHONPATH", "CUDA_VISIBLE_DEVICES"]}})
        result["server_ready_seconds"] = wait_ready(server, port, args.server_timeout)
        print(f"server ready on port {port} after {result['server_ready_seconds']:.1f}s", flush=True)

        environment = client_environment()
        record_execution(run_dir, checkpoint=ckpt, settings={
            "role": "client", "argv": client_cmd, "cwd": str(run_dir),
            "environment": {key: environment.get(key) for key in
                            ["MUJOCO_GL", "LIBERO_CONFIG_PATH", "PYTHONPATH", "PYOPENGL_PLATFORM", "LD_LIBRARY_PATH",
                             *THREAD_LIMIT_KEYS]},
            "original_LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH")})
        started = time.monotonic()
        with open(logs / "libero_client.log", "x") as client_log:
            client = subprocess.run(client_cmd, cwd=str(run_dir), env=environment,
                                    stdout=client_log, stderr=subprocess.STDOUT)
        result["client_seconds"] = time.monotonic() - started
        result["client_returncode"] = client.returncode
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        if server is not None:
            result["server_pid"] = server.pid
            result["server_returncode"] = stop_own_process(server)
        record_execution(run_dir, checkpoint=ckpt, settings=result)
        print(json.dumps(result), flush=True)
    return 0 if result.get("client_returncode") == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint-dir", default=str(default_checkpoint()))
    parser.add_argument("--task-suite-name", default=client_mod.DEFAULT_TASK_SUITE, choices=sorted(client_mod.MAX_STEPS))
    parser.add_argument("--task-ids", type=client_mod._int_list, default=[0])
    parser.add_argument("--episode-ids", type=client_mod._int_list, default=[0])
    parser.add_argument("--seed", type=int, default=client_mod.DEFAULT_SEED)
    parser.add_argument("--host", default="127.0.0.1", help="host the client connects to")
    parser.add_argument("--port", type=int, default=None, help="default: pick a free port (never 8123)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--server-timeout", type=float, default=1800.0, help="seconds to wait for /healthz")
    parser.add_argument("--verify-checksums", action="store_true",
                        help="re-hash every checkpoint file before launching (slow, ~12 GB)")
    parser.add_argument("--dry-run", action="store_true", help="print commands and settings; allocate nothing")
    return parser


def dry_run(args) -> dict:
    identity = checkpoint_identity(Path(args.checkpoint_dir), require=False)
    port = choose_port(args.port)
    run_placeholder = str(shared_storage_root() / "results/benchmark/<KST_commit_run-id>")
    server_cmd = server_command(openpi_python(), port, Path(identity["path"]))
    client_cmd = client_command(libero_python(), run_placeholder, port, args)
    return build_settings(args, port, identity, server_cmd, client_cmd)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run(args), indent=2, ensure_ascii=False))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
