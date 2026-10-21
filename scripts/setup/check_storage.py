"""Check independent storage, artifact links, interpreter and local fixture references."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.workspace import shared_storage_root, legacy_storage_root, workspace_path

OLD = Path("/mnt/dev/work")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash-artifacts", action="store_true")
    args = parser.parse_args()
    storage = shared_storage_root()
    assert storage.is_dir(), f"Provision shared inputs at {storage}; see doc/active/SHARED_STORAGE_AND_PYTHON.md"
    assert not storage.is_relative_to(OLD), storage
    manifest = json.loads((ROOT / "experiments/manifests/shared-storage.json").read_text())
    for entry in manifest["inputs"]:
        target = storage / entry["storage_relative_path"]
        assert target.exists() or target.is_symlink(), target
        alias = legacy_storage_root() / entry["legacy_relative_path"]
        assert alias.resolve() == target.resolve(), (alias, target)
    for entry in manifest.get("checkout_links", []):
        p = ROOT / entry["target"]
        assert p.is_symlink() and p.resolve() == (storage / entry["storage_relative_path"]).resolve(), p
        assert not str(os.readlink(p)).startswith(str(OLD) + "/"), p
    linked = []
    artifacts = json.loads((ROOT / "experiments/manifests/legacy-artifacts.json").read_text())["files"]
    for entry in artifacts:
        p = ROOT / entry["target"]
        assert p.is_symlink() and p.is_file(), p
        resolved = p.resolve()
        assert resolved.is_relative_to(storage), (p, resolved)
        assert not str(os.readlink(p)).startswith(str(OLD) + "/"), p
        assert resolved.stat().st_size == entry["size"], resolved
        if args.hash_artifacts:
            h = hashlib.sha256()
            with resolved.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    h.update(block)
            assert h.hexdigest() == entry["sha256"], resolved
        linked.append(str(resolved))
    assert Path(sys.executable).resolve().is_relative_to(ROOT), sys.executable
    assert Path(sys.prefix).resolve().is_relative_to(ROOT), sys.prefix
    forbidden = []
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
        for name in dirs + files:
            p = Path(base) / name
            if p.is_symlink() and p.resolve().is_relative_to(OLD):
                forbidden.append(str(p.relative_to(ROOT)))
    assert not forbidden, forbidden
    for name in ["pi05", "benchmark"]:
        p = ROOT / f".runtime/legacy-{name}/.git/objects/info/alternates"
        if p.exists():
            for line in p.read_text().splitlines():
                assert Path(line).resolve().is_relative_to(ROOT), (p, line)
    print(json.dumps({"project":str(ROOT), "storage":str(storage), "legacy_namespace":str(legacy_storage_root()),
                      "input_roots":len(manifest["inputs"]), "linked_artifacts":len(linked),
                      "interpreter":sys.executable, "python":sys.version.split()[0],
                      "old_workspace_symlinks":forbidden, "artifact_sha256_checked":args.hash_artifacts}, indent=2))


if __name__ == "__main__":
    main()
