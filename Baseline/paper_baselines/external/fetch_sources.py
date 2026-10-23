#!/usr/bin/env python3
"""Fetch the pinned external sources listed in sources.json into external/<name>/.

Each source is cloned without checkout, the exact pinned SHA is fetched (shallow) and checked out
detached. The clone bodies are untracked (external/.gitignore). A fetched tree whose HEAD differs from
the pinned SHA is reported and, with --force, re-checked-out. No git submodule is created, so the
checkout index is never touched.

Usage:
    python Baseline/paper_baselines/external/fetch_sources.py            # fetch all missing
    python Baseline/paper_baselines/external/fetch_sources.py --only TokenCut
    python Baseline/paper_baselines/external/fetch_sources.py --check    # verify SHA + license file only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "sources.json"


def _git(args: list[str], cwd: Path | None = None) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def load_sources() -> list[dict]:
    return json.loads(MANIFEST.read_text())["sources"]


def source_dir(name: str) -> Path:
    return HERE / name


def head_sha(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        return _git(["rev-parse", "HEAD"], cwd=path)
    except subprocess.CalledProcessError:
        return None


def fetch_one(src: dict, force: bool = False) -> dict:
    path = source_dir(src["name"])
    sha = src["sha"]
    current = head_sha(path)
    if current == sha:
        return {"name": src["name"], "status": "present", "sha": sha}
    if current is not None and not force:
        return {"name": src["name"], "status": "sha_mismatch", "sha": current, "expected": sha}
    path.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").exists():
        _git(["init", "-q"], cwd=path)
        _git(["remote", "add", "origin", src["repo"]], cwd=path)
    _git(["fetch", "-q", "--depth", "1", "origin", sha], cwd=path)
    _git(["-c", "advice.detachedHead=false", "checkout", "-q", "--detach", sha], cwd=path)
    got = head_sha(path)
    if got != sha:
        raise RuntimeError(f"{src['name']}: checked out {got}, expected {sha}")
    return {"name": src["name"], "status": "fetched", "sha": got}


def license_info(src: dict) -> dict:
    lic = source_dir(src["name"]) / src.get("license_file", "LICENSE")
    if not lic.exists():
        return {"license_file_present": False}
    data = lic.read_bytes()
    first = data.decode("utf-8", "replace").strip().splitlines()[:3]
    return {"license_file_present": True, "license_sha256": hashlib.sha256(data).hexdigest(),
            "license_head": " | ".join(s.strip() for s in first if s.strip())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None, help="source names to fetch (default: all)")
    ap.add_argument("--check", action="store_true", help="only verify SHAs and license files")
    ap.add_argument("--force", action="store_true", help="re-checkout a tree whose HEAD differs from the pin")
    args = ap.parse_args(argv)
    rc = 0
    for src in load_sources():
        if args.only and src["name"] not in args.only:
            continue
        if args.check:
            cur = head_sha(source_dir(src["name"]))
            rec = {"name": src["name"], "status": "ok" if cur == src["sha"] else "missing_or_mismatch",
                   "sha": cur, "expected": src["sha"]}
        else:
            try:
                rec = fetch_one(src, force=args.force)
            except (subprocess.CalledProcessError, RuntimeError) as exc:
                rec = {"name": src["name"], "status": "error", "error": str(exc)}
        rec.update(license_info(src))
        rec["license_declared"] = src.get("license")
        if rec["status"] not in ("present", "fetched", "ok"):
            rc = 1
        print(json.dumps(rec))
    return rc


if __name__ == "__main__":
    sys.exit(main())
