"""Materialize the original pi05/v2 for tests that compare historical Git snapshots.

Only the ignored .runtime directory is written. The object store is shared with
this same checkout, never with the other development checkout or /mnt/dev/work.
This frozen fixture is not the current integration client under test.
"""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / ".runtime/legacy-pi05"
MANIFEST = json.loads((ROOT / "doc/migration/source-manifest.json").read_text())
PIN = MANIFEST["original_heads"]["pi05_TO_hybrid"]

def run(*args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)

if not DEST.exists():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    run("git", "clone", "--shared", "--no-checkout", str(ROOT), str(DEST))
    run("git", "-C", str(DEST), "checkout", "--detach", PIN)
    run("git", "-C", str(DEST), "remote", "set-url", "origin", "git@github.com:Ignimkk/AG3S_VLA.git")
else:
    actual = subprocess.check_output(["git", "-C", str(DEST), "rev-parse", "HEAD"], text=True).strip()
    assert actual == PIN, (actual, PIN)
print(DEST)
