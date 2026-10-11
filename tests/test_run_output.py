"""Run isolation is a storage contract, independent of VLA or GPU execution."""
import argparse
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from common.run_output import allocate_run_dir, configure_run_logging, prepare_run_outputs, resolve_output_path


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setenv("AG3S_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.delenv("AG3S_RUN_DIR", raising=False)
    return tmp_path / "storage"


def test_fresh_runs_are_unique_and_manifest_records_own_checkout(storage, monkeypatch):
    first = allocate_run_dir()
    monkeypatch.delenv("AG3S_RUN_DIR")
    second = allocate_run_dir()
    assert first != second
    assert first.parent == storage / "results" / "development"
    assert re.fullmatch(r"\d{8}T\d{6}\+0900_[a-f0-9]{12}_[a-f0-9]{12}", first.name)
    manifest = json.loads((first / "run_manifest.json").read_text())
    assert manifest["checkout"] == str(Path(__file__).resolve().parents[1])
    assert manifest["created_at"].endswith("+09:00")


def test_inherited_run_groups_child_process_and_preserves_manifest(storage):
    run = allocate_run_dir()
    before = (run / "run_manifest.json").read_bytes()
    output = subprocess.check_output([sys.executable, "-m", "common.run_output"], text=True)
    assert Path(output.strip()) == run
    assert (run / "run_manifest.json").read_bytes() == before


def test_record_video_log_checkpoint_and_eval_targets_are_inside_run(storage):
    args = argparse.Namespace(run_dir=None, record="videos/rollout.mp4", record_inputs="policy_inputs",
                              trace="logs/trace", checkpoint=Path("checkpoints"),
                              atomic_output_dir=Path("atomic_eval"), unused=None)
    run = prepare_run_outputs(args, {"record": "video.mp4", "record_inputs": "inputs",
                                    "trace": "logs", "checkpoint": "checkpoints",
                                    "atomic_output_dir": "eval", "unused": "unused"})
    for attribute in ("record", "record_inputs", "trace", "checkpoint", "atomic_output_dir"):
        assert Path(getattr(args, attribute)).is_relative_to(run)
    assert isinstance(args.record, str)
    assert isinstance(args.checkpoint, Path)
    assert args.unused is None
    assert resolve_output_path(None, run_dir=run, default_name="policy_records") == run / "policy_records"


@pytest.mark.parametrize("relative", ["../other-run/video.mp4", "../../recordings/step_0.npy"])
def test_relative_output_escape_is_rejected(storage, relative):
    run = allocate_run_dir()
    with pytest.raises(ValueError, match="Output must be inside"):
        resolve_output_path(relative, run_dir=run, default_name="unused")


def test_existing_recording_remains_unchanged(storage):
    historic = storage / "recordings" / "policy_records"
    historic.mkdir(parents=True)
    record = historic / "step_0.npy"
    record.write_bytes(b"historical evidence")
    run = allocate_run_dir()
    with pytest.raises(ValueError):
        resolve_output_path(record, run_dir=run, default_name="unused")
    with pytest.raises(ValueError):
        allocate_run_dir(historic)
    assert record.read_bytes() == b"historical evidence"


def test_symlink_output_escape_is_rejected(storage):
    run = allocate_run_dir()
    historic = storage / "recordings"
    historic.mkdir()
    (run / "records").symlink_to(historic, target_is_directory=True)
    with pytest.raises(ValueError):
        resolve_output_path("records/step_0.npy", run_dir=run, default_name="unused")
    (run.parent / "escaped").symlink_to(historic, target_is_directory=True)
    with pytest.raises(ValueError):
        allocate_run_dir("escaped")
    assert list(historic.iterdir()) == []


def test_symlink_results_root_is_rejected_before_write(storage, tmp_path):
    (storage / "results").mkdir(parents=True)
    historic = tmp_path / "historical"
    historic.mkdir()
    (storage / "results" / "development").symlink_to(historic, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        allocate_run_dir()
    assert list(historic.iterdir()) == []


def test_nested_or_root_run_is_rejected(storage):
    for path in (storage / "results" / "development", "parent/nested"):
        with pytest.raises(ValueError):
            allocate_run_dir(path)


def test_reused_output_subtree_cannot_contain_escaping_symlink(storage):
    run = allocate_run_dir()
    output = run / "grid_eval"
    output.mkdir()
    historic = storage / "recordings"
    historic.mkdir()
    (output / "videos").symlink_to(historic, target_is_directory=True)
    with pytest.raises(ValueError):
        resolve_output_path(output, run_dir=run, default_name="unused")
    with pytest.raises(ValueError):
        allocate_run_dir(run)
    assert list(historic.iterdir()) == []


def test_unique_logs_keep_console_and_override_inherited_wandb(storage, monkeypatch):
    logger = logging.getLogger()
    before = tuple(logger.handlers)
    monkeypatch.setenv("WANDB_DIR", str(storage / "historical-wandb"))
    first = configure_run_logging("server")
    second = configure_run_logging("server")
    try:
        assert first != second
        assert first.parent == second.parent
        assert first.exists() and second.exists()
        assert all(handler in logger.handlers for handler in before)
        assert Path(os.environ["WANDB_DIR"]) == first.parent / "wandb"
        logger.warning("persisted routing verification")
        assert "persisted routing verification" in first.read_text()
        assert "persisted routing verification" in second.read_text()
    finally:
        for handler in tuple(logger.handlers):
            if handler not in before:
                logger.removeHandler(handler)
                handler.close()


def test_invalid_log_name_cannot_escape(storage):
    with pytest.raises(ValueError):
        configure_run_logging("../old-input")
