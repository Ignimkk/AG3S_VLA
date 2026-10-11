"""Paths for a checkout; legacy storage is input data, never another source tree."""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def workspace_path(relative: str) -> str:
    if relative.startswith("legacy/"):
        storage = Path(os.environ.get("AG3S_LEGACY_STORAGE", "/mnt/dev/work"))
        return str(storage / relative.removeprefix("legacy/"))
    return str(PROJECT_ROOT / relative)
