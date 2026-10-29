"""Paths for this checkout and its separate shared input storage."""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def shared_storage_root() -> Path:
    return Path(os.environ.get("AG3S_STORAGE_ROOT", PROJECT_ROOT.parent / "AG3S_VLA-storage")).expanduser().resolve()


def legacy_storage_root() -> Path:
    """Compatibility namespace for recordings; contains no active source checkout."""
    return shared_storage_root() / "legacy-inputs"


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
        return _path_string(legacy_storage_root() / relative.removeprefix("legacy/"), relative)
    return _path_string(PROJECT_ROOT / relative, relative)


def legacy_input_path(relative: str) -> Path:
    """Read-only recording/dataset input in the separate storage root."""
    return Path(workspace_path("legacy/" + relative))


#: Absolute root of the retired checkout; recorded JSON/logs still name inputs under it.
OLD_CHECKOUT_PREFIX = "/mnt/dev/work/"


def rebase_legacy_path(value: str) -> str:
    """Map one recorded `/mnt/dev/work/<rel>` path onto the shared legacy input root; others unchanged."""
    if value.startswith(OLD_CHECKOUT_PREFIX):
        return str(legacy_input_path(value.removeprefix(OLD_CHECKOUT_PREFIX)))
    return value


def rebase_legacy_paths(obj):
    """Return a copy of loaded recorded JSON with old-checkout paths rebased in dict keys and string values."""
    if isinstance(obj, str):
        return rebase_legacy_path(obj)
    if isinstance(obj, dict):
        return {rebase_legacy_paths(k) if isinstance(k, str) else k: rebase_legacy_paths(v)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [rebase_legacy_paths(v) for v in obj]
    return obj


def bind_historical_client_inputs(client):
    """Rebase input locations after loading a frozen client, preserving its source/algorithm."""
    if hasattr(client, "RANDOMIZED_16D_DATASET"):
        client.RANDOMIZED_16D_DATASET = legacy_input_path("pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1")
    for model in getattr(client, "MODELS", {}).values():
        checkpoint = model.get("checkpoint")
        if isinstance(checkpoint, str):
            model["checkpoint"] = rebase_legacy_path(checkpoint)
    return client
