"""Frozen pi05 and benchmark snapshots for historical comparison tests, local to this checkout."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "doc/migration/source-manifest.json").read_text())
PINS = {"pi05": MANIFEST["original_heads"]["pi05_TO_hybrid"],
        "benchmark": MANIFEST["sources"]["benchmark"]}

for name, pin in PINS.items():
    dest = ROOT / (".runtime/legacy-" + name)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--shared", "--no-checkout", str(ROOT), str(dest)], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", pin], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-C", str(dest), "remote", "set-url", "origin", "git@github.com:Ignimkk/AG3S_VLA.git"], check=True)
    actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    assert actual == pin, (actual, pin)
    print(dest)
