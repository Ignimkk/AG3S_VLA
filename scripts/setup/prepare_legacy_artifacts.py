"""Link declared immutable inputs into this checkout after SHA-256 verification."""
from pathlib import Path
import argparse, hashlib, json, os
ROOT = Path(__file__).resolve().parents[2]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", type=Path, default=Path(os.environ.get("AG3S_LEGACY_STORAGE", "/mnt/dev/work")))
    args = parser.parse_args()
    entries = json.loads((ROOT / "experiments/manifests/legacy-artifacts.json").read_text())
    if isinstance(entries, dict): entries = entries["files"]
    for item in entries:
        src = args.storage / item["legacy_relative_path"]
        dst = ROOT / item["target"]
        assert dst.is_relative_to(ROOT) and ".." not in Path(item["target"]).parts
        if not src.is_file(): raise FileNotFoundError(src)
        h = hashlib.sha256()
        with src.open("rb") as file:
            for block in iter(lambda: file.read(4 * 1024 * 1024), b""): h.update(block)
        assert src.stat().st_size == item["size"] and h.hexdigest() == item["sha256"], src
        if dst.is_symlink():
            assert dst.resolve() == src.resolve(), dst
        else:
            assert not dst.exists(), dst
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)
        print(item["target"])
if __name__ == "__main__": main()
