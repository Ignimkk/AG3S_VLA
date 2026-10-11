"""Paths for a checkout; legacy storage is input data, never another source tree."""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def workspace_path(relative: str) -> str:
    if relative in {"legacy/pi05_TO_hybrid", "legacy/benchmark"}:
        name = "pi05" if relative.endswith("pi05_TO_hybrid") else "benchmark"
        fixture = PROJECT_ROOT / (".runtime/legacy-" + name)
        if not fixture.is_dir():
            raise FileNotFoundError("Run python scripts/setup/prepare_legacy_test_fixture.py for historical comparison tests.")
        return str(fixture)
    if relative.startswith("legacy/"):
        storage = Path(os.environ.get("AG3S_LEGACY_STORAGE", "/mnt/dev/work"))
        return str(storage / relative.removeprefix("legacy/"))
    return str(PROJECT_ROOT / relative)
