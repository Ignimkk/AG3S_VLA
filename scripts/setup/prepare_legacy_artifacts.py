"""Verify and link declared immutable inputs from separate storage."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.workspace import legacy_storage_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", type=Path, default=legacy_storage_root(), help="Compatibility namespace under AG3S_STORAGE_ROOT/legacy-inputs")
    parser.add_argument("--relink", action="store_true", help="Replace existing symlinks after checking the new input's size and SHA-256")
    args = parser.parse_args()
    entries = json.loads((ROOT / "experiments/manifests/legacy-artifacts.json").read_text())
    if isinstance(entries, dict):
        entries = entries["files"]
    for item in entries:
        src = (args.storage / item["legacy_relative_path"]).resolve()
        dst = ROOT / item["target"]
        assert dst.is_relative_to(ROOT) and ".." not in Path(item["target"]).parts
        if not src.is_file():
            raise FileNotFoundError(src)
        h = hashlib.sha256()
        with src.open("rb") as file:
            for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
                h.update(block)
        assert src.stat().st_size == item["size"] and h.hexdigest() == item["sha256"], src
        if dst.is_symlink():
            # Compare link text too: old backlinks must not conceal an old lexical dependency.
            desired = os.path.relpath(src, dst.parent)
            if os.readlink(dst) != desired:
                if not args.relink:
                    raise RuntimeError(f"Run with --relink to update {dst}")
                dst.unlink()
                dst.symlink_to(desired)
        else:
            assert not dst.exists(), dst
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(os.path.relpath(src, dst.parent))
        print(item["target"])


    storage_root = args.storage.resolve().parent
    manifest = json.loads((ROOT / "experiments/manifests/shared-storage.json").read_text())
    for entry in manifest.get("checkout_links", []):
        src = (storage_root / entry["storage_relative_path"]).resolve()
        dst = ROOT / entry["target"]
        assert src.is_dir() and dst.is_relative_to(ROOT), (src, dst)
        desired = os.path.relpath(src, dst.parent)
        if dst.is_symlink():
            if os.readlink(dst) != desired:
                if not args.relink:
                    raise RuntimeError(f"Run with --relink to update {dst}")
                dst.unlink()
                dst.symlink_to(desired, target_is_directory=True)
        else:
            assert not dst.exists(), dst
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(desired, target_is_directory=True)
        print(entry["target"])


if __name__ == "__main__":
    main()
