"""Drop the page cache of finished run outputs and large inputs; print one summary line.

Usage: python3 scripts/setup/reclaim_cache.py "$AG3S_RUN_DIR" <large input dir/file ...>

Files that any live process has open (another session's run, a checkpoint loaded by a
server) are skipped, so shared checkpoints are only dropped when nobody holds them.
Never writes /proc/sys/vm/drop_caches or memory.reclaim.
"""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

CGROUP = Path("/sys/fs/cgroup")
GIB = 1024 ** 3


def working_set():
    try:
        current = int((CGROUP / "memory.current").read_text())
        stat = dict(line.split() for line in (CGROUP / "memory.stat").read_text().splitlines())
        return current - int(stat["inactive_file"])
    except (OSError, KeyError, ValueError):
        return None


def iter_files(paths):
    for root in paths:
        if root.is_file():
            yield root
        elif root.is_dir():
            for dirpath, _, names in os.walk(root):
                for name in names:
                    p = Path(dirpath, name)
                    if p.is_file() and not p.is_symlink():
                        yield p


def open_files(roots):
    """Map open file path -> pids, limited to files under the given roots."""
    prefixes = tuple(str(r) for r in roots)
    held = {}
    for fd_dir in Path("/proc").glob("[0-9]*/fd"):
        pid = int(fd_dir.parent.name)
        if pid == os.getpid():
            continue
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith(prefixes):
                held.setdefault(target, set()).add(pid)
    return held


def cached_bytes(files):
    if not files or not shutil.which("fincore"):
        return None
    total = 0
    for i in range(0, len(files), 500):
        out = subprocess.run(["fincore", "-b", "-n", "-o", "RES", *map(str, files[i:i + 500])],
                             capture_output=True, text=True).stdout
        total += sum(int(v) for v in out.split() if v.isdigit())
    return total


def drop(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fdatasync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def gib(v):
    return "?" if v is None else f"{v / GIB:.1f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="measure and list skips only")
    args = parser.parse_args()

    roots = [p.resolve() for p in args.paths if p.exists()]
    missing = [str(p) for p in args.paths if not p.exists()]
    ws_before = working_set()
    held = open_files(roots)
    files = [p for p in iter_files(roots) if str(p) not in held]
    cache_before = cached_bytes(files)

    errors = 0
    if not args.dry_run:
        for p in files:
            try:
                drop(p)
            except OSError:
                errors += 1

    cache_after = cached_bytes(files)
    ws_after = working_set()
    pids = sorted({pid for s in held.values() for pid in s})
    line = (f"reclaim: ws {gib(ws_before)}->{gib(ws_after)} GiB | cache {gib(cache_before)}->"
            f"{gib(cache_after)} GiB, {len(files)} files, {len(roots)} paths")
    if held:
        line += f" | skipped {len(held)} open files (pid {','.join(map(str, pids[:8]))})"
    if errors:
        line += f" | {errors} errors"
    if missing:
        line += f" | missing {' '.join(missing)}"
    if args.dry_run:
        line += " | dry-run"
    print(line)
    return 1 if errors or missing else 0


if __name__ == "__main__":
    sys.exit(main())
