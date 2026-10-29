"""Allocate new run outputs without writing to historical shared inputs.

Set AG3S_RUN_DIR once in the launching shell to group a server, client and logs.
Relative output arguments are always interpreted inside that run directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import uuid

from common.workspace import PROJECT_ROOT, shared_storage_root


def development_results_root(storage_root: str | Path | None = None) -> Path:
    storage = Path(storage_root).expanduser().resolve() if storage_root is not None else shared_storage_root()
    root = storage / "results" / "development"
    # Resolving a symlink in results/development must not redirect output elsewhere.
    if root.resolve() != root:
        raise ValueError(f"Development results root must not redirect through a symlink: {root}")
    return root


def _contained(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"Output must be inside {root}: {path}")
    return resolved


def _check_existing_links(path: Path, root: Path) -> None:
    """Reject escaping links under a reused output directory before its writers run."""
    if not path.is_dir():
        return
    for directory, directories, files in os.walk(path, followlinks=False):
        for name in directories + files:
            child = Path(directory) / name
            if child.is_symlink():
                _contained(child, root)


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short=12", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def allocate_run_dir(
    run_dir: str | Path | None = None, *, storage_root: str | Path | None = None,
) -> Path:
    """Create a KST-date/commit/UUID run or reuse an explicitly inherited run.

    Every process without AG3S_RUN_DIR gets a fresh run. An explicit run must be
    a direct child of results/development; inherited server/client processes can
    reuse it. Existing input paths and symlink escapes fail before any writes.
    """
    root = development_results_root(storage_root)
    selected = run_dir if run_dir is not None else os.environ.get("AG3S_RUN_DIR")
    # Fixed KST offset avoids a tzdata dependency in standalone CPU Python.
    created_at = datetime.now(timezone(timedelta(hours=9), name="KST"))
    commit = _commit()
    if selected is None:
        name = f"{created_at:%Y%m%dT%H%M%S%z}_{commit}_{uuid.uuid4().hex[:12]}"
        target = root / name
    else:
        target = Path(selected).expanduser()
        if not target.is_absolute():
            target = root / target
    target = _contained(target, root)
    if target.parent != root:
        raise ValueError(f"Run directory must be a direct child of {root}: {target}")
    _check_existing_links(target, target)
    root.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=selected is not None)
    _contained(target, root)
    manifest = target / "run_manifest.json"
    _contained(manifest, target)
    try:
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump({"created_at": created_at.isoformat(), "commit": commit,
                       "checkout": str(PROJECT_ROOT), "run_dir": str(target)}, stream, indent=2)
            stream.write("\n")
    except FileExistsError:
        pass
    os.environ["AG3S_RUN_DIR"] = str(target)
    return target


def resolve_output_path(
    path: str | Path | None, *, run_dir: str | Path, default_name: str,
) -> Path:
    """Resolve a requested file/directory within the selected run, rejecting escapes."""
    root = Path(run_dir).expanduser().resolve()
    # The run itself is also checked against the configured storage boundary.
    results = development_results_root()
    _contained(root, results)
    if root.parent != results:
        raise ValueError(f"Invalid run directory: {root}")
    target = Path(default_name if path is None else path).expanduser()
    if not target.is_absolute():
        target = root / target
    target = _contained(target, root)
    _check_existing_links(target, root)
    return target


def configure_run_logging(name: str) -> Path:
    """Keep console logging and add a unique per-process log inside the active run."""
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in name):
        raise ValueError(f"Invalid log name: {name!r}")
    run = allocate_run_dir()
    logs = resolve_output_path("logs", run_dir=run, default_name="logs")
    logs.mkdir(parents=True, exist_ok=True)
    wandb_dir = resolve_output_path("logs/wandb", run_dir=run, default_name="logs/wandb")
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    log_path = logs / f"{name}_{uuid.uuid4().hex[:12]}.log"
    # Reserve with O_EXCL so a restart never reopens a preexisting log.
    with log_path.open("x", encoding="utf-8"):
        pass
    logger = logging.getLogger()
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.info("run outputs: %s; log: %s", run, log_path)
    return log_path


def add_run_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", default=None, metavar="DIR",
                        help="run under shared results/development (default: AG3S_RUN_DIR or a new KST-date_commit_UUID run); relative output paths are inside this run")


def prepare_run_outputs(args: argparse.Namespace, output_names: dict[str, str]) -> Path:
    """Route enabled argparse outputs while keeping optional recording disabled."""
    run = allocate_run_dir(args.run_dir)
    args.run_dir = run
    for attribute, default_name in output_names.items():
        value = getattr(args, attribute)
        if value is not None:
            resolved = resolve_output_path(value, run_dir=run, default_name=default_name)
            # Existing runners use str.replace for video file fallbacks.
            setattr(args, attribute, resolved if isinstance(value, Path) else str(resolved))
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_dir_argument(parser)
    print(allocate_run_dir(parser.parse_args().run_dir))
