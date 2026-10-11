"""Paths for a checkout; legacy storage is input data, never another source tree."""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _path_string(path: Path, relative: str) -> str:
    value = str(path)
    return value + "/" if relative.endswith("/") and not value.endswith("/") else value

def workspace_path(relative: str = "") -> str:
    if relative in {"legacy/pi05_TO_hybrid", "legacy/benchmark"}:
        name = "pi05" if relative.endswith("pi05_TO_hybrid") else "benchmark"
        fixture = PROJECT_ROOT / (".runtime/legacy-" + name)
        if not fixture.is_dir():
            raise FileNotFoundError("Run python scripts/setup/prepare_legacy_test_fixture.py for historical comparison tests.")
        return str(fixture)
    if relative.startswith("legacy/"):
        storage = Path(os.environ.get("AG3S_LEGACY_STORAGE", "/mnt/dev/work"))
        return _path_string(storage / relative.removeprefix("legacy/"), relative)
    return _path_string(PROJECT_ROOT / relative, relative)


def legacy_input_path(relative: str) -> Path:
    """Read-only recording/dataset input selected from the legacy storage root."""
    return Path(workspace_path("legacy/" + relative))
