"""SL6 B5 — ``attention_audit.py --noun-rows`` (CPU parts + the tiny pi05_libero model).

Pinned: the flag is off by default (the SL3 audit path unchanged); `NounRowReader` looks the target noun up with the
B1 libero rule and keeps the target row of the requested layers; `audit_noun_episode` joins every policy record to the
episode (bit-exact, as the SL3 audit) and writes ``noun_rows/<stem>_noun.npz``; the dry run reports the B5 settings;
on the tiny real pi05_libero model the extraction equals what a served ``noun_target:L<l>H<h>`` source returns.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from common.workspace import PROJECT_ROOT
from tests.trajopt.test_sl6_b1_libero_noun import BOS, IDS, StubSP, tokenize

PROMPT = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def _has(*modules):
    return all(importlib.util.find_spec(m) is not None for m in modules)


@pytest.fixture
def source_run(tmp_path, monkeypatch):
    """Two libero_spatial episodes laid out like launch_libero_pi05_only's run, below a temporary storage root."""
    from openpi_client import image_tools

    root = tmp_path / "storage"
    run = root / "results/benchmark/20261029T000000_000000000000_fake"
    (run / "episodes").mkdir(parents=True)
    (run / "policy_records").mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"commit": "0" * 40, "settings": {
        "condition": "pi0.5-only", "policy_config": "pi05_libero", "checkpoint": {"path": "/nowhere"}}}))
    rng = np.random.default_rng(1)
    rows, step = [], 0
    for task_id, calls in ((0, 2), (3, 3)):
        n_obs = 10 + 5 * calls + 1
        agent = rng.integers(0, 256, (n_obs, 256, 256, 3), dtype=np.uint8)
        wrist = rng.integers(0, 256, (n_obs, 256, 256, 3), dtype=np.uint8)
        chunk_t = np.array([10 + 5 * k for k in range(calls)])
        state = rng.normal(size=(calls, 8))
        stem = f"libero_spatial_t{task_id}_e1"
        np.savez_compressed(run / "episodes" / f"{stem}.npz", obs_agentview_image=agent,
                            obs_robot0_eye_in_hand_image=wrist, chunk_t=chunk_t, policy_state=state)
        for k in range(calls):
            t = int(chunk_t[k])
            img = image_tools.resize_with_pad(np.ascontiguousarray(agent[t][::-1, ::-1]), 224, 224)
            wimg = image_tools.resize_with_pad(np.ascontiguousarray(wrist[t][::-1, ::-1]), 224, 224)
            record = {"inputs/observation/image": img, "inputs/observation/wrist_image": wimg,
                      "inputs/observation/state": state[k].copy(), "inputs/prompt": PROMPT,
                      "outputs/actions": np.zeros((10, 7))}
            np.save(run / "policy_records" / f"step_{step}.npy", np.asarray(record))
            step += 1
        rows.append({"suite": "libero_spatial", "task_id": task_id, "episode_idx": 1, "init_state_index": 1,
                     "task_description": PROMPT, "status": "success", "official_success": True,
                     "policy_calls": calls, "npz": f"episodes/{stem}.npz"})
    (run / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setenv("AG3S_STORAGE_ROOT", str(root))
    return run


class _Policy:
    def __init__(self, sp):
        self.sp = sp

    def _input_transform(self, request):
        return tokenize(self.sp, request["prompt"])


class _Extractor:
    """``rows`` -> slice_probs-like dict with image[l, g, r, :] = 1000 l + 10 g + r + token / 1e4."""

    def __init__(self):
        self.calls = []

    def rows(self, obs, rows, timed=True):
        self.calls.append(np.asarray(rows).tolist())
        L, G, R = 18, 8, len(rows)
        base = (1000.0 * np.arange(L)[:, None, None] + 10.0 * np.arange(G)[None, :, None]
                + np.arange(R)[None, None, :])
        image = base[..., None] + np.arange(768)[None, None, None, :] / 1e4
        z = np.zeros((L, G, R))
        return {"image": image.astype(np.float32), "image_mass": z + 1, "text_mass": z, "bos": z, "self": z}


def _reader(**kw):
    from AG3S.trajopt.attention_source import NounTokens
    from benchmark.runners import attention_audit as audit

    sp = StubSP()
    ex = _Extractor()
    reader = audit.NounRowReader(_Policy(sp), tokens=NounTokens(ids=IDS, bos=BOS, rule="libero", sp=sp),
                                 extractor=ex, observation_fn=lambda inputs: inputs, **kw)
    return reader, ex, sp


def test_flag_default_off_and_layer_range():
    from benchmark.runners import attention_audit as audit

    args = audit.build_parser().parse_args(["--source-run", "x"])
    assert args.noun_rows is False and args.noun_layers == "2-17"
    assert audit._layer_range("2-17") == tuple(range(2, 18)) and audit._layer_range("2,5,9") == (2, 5, 9)
    assert audit.NOUN_LAYERS == tuple(range(2, 18))


def test_reader_keeps_the_target_row_of_the_layers():
    reader, ex, sp = _reader()
    out = reader.read({"prompt": PROMPT})
    assert out["ok"] and out["lookup"]["noun"] == "bowl" and out["lookup"]["noun_rule"] == "libero"
    pieces = [sp.id_to_piece(int(i)) for i in tokenize(sp, PROMPT)["tokenized_prompt"][:20]]
    assert pieces[out["rows"][0] - 768] == "▁bowl" and ex.calls == [out["rows"].tolist()]
    assert out["image"].shape == (16, 8, 768)
    # layer 2, head 3, row 0 (target)
    assert np.allclose(out["image"][0, 3], 2000 + 30 + np.arange(768) / 1e4)
    assert np.allclose(out["image"][15, 7], 17000 + 70 + np.arange(768) / 1e4)
    assert out["mass"]["image_mass"].shape == (16, 8, 3)
    with pytest.raises(ValueError, match="libero"):
        from AG3S.trajopt.attention_source import NounTokens
        from benchmark.runners import attention_audit as audit
        audit.NounRowReader(_Policy(sp), tokens=NounTokens(ids=IDS, bos=BOS), extractor=ex,
                            observation_fn=lambda i: i)
    bad = reader.read({"prompt": "put the bowl on the plate"})
    assert not bad["ok"] and bad["image"] is None and "libero rule" in bad["lookup"]["noun_error"]


def test_episode_writer_joins_and_writes(source_run, tmp_path):
    from benchmark.runners import attention_audit as audit

    reader, ex, _ = _reader(layers=(2, 10))
    src = audit.index_source_run(audit.resolve_source_run(source_run.name))
    out_dir = tmp_path / "out"
    (out_dir / "noun_rows").mkdir(parents=True)
    args = argparse.Namespace(limit_calls=None)
    res = audit.audit_noun_episode(reader, src, src["episodes"][1], out_dir, args)
    assert res["calls"] == 3 and res["noun_ok"] == 3 and res["shape_noun_image"] == [3, 2, 8, 768]
    assert all(j["image"] and j["wrist_image"] and j["state"] and j["prompt"] for j in res["joins"])
    with np.load(out_dir / res["noun_npz"]) as z:
        assert set(audit.NOUN_EPISODE_KEYS) <= set(z.files)
        assert z["layers"].tolist() == [2, 10] and z["obs_index"].tolist() == [10, 15, 20]
        assert z["global_step"].tolist() == [2, 3, 4] and z["noun_ok"].all()
        assert np.allclose(z["noun_image"][1, 1, 4], 10000 + 40 + np.arange(768) / 1e4)
        assert z["noun_image"].dtype == np.float32 and len(z["record_sha256"][0]) == 64
    with pytest.raises(FileExistsError):
        audit.audit_noun_episode(reader, src, src["episodes"][1], out_dir, args)


def test_dry_run_settings(source_run, capsys):
    from benchmark.runners import attention_audit as audit

    assert audit.main(["--source-run", source_run.name, "--noun-rows", "--dry-run", "--task-ids", "3"]) == 0
    out = json.loads(capsys.readouterr().out)
    s = out["settings"]
    assert out["calls"] == 3 and s["step"] == "SL6-B5" and s["noun_rows"]["layers"] == list(range(2, 18))
    assert s["attention"]["noun_rows"] is True
    # without the flag: the SL3 audit settings
    assert audit.main(["--source-run", source_run.name, "--dry-run", "--task-ids", "3"]) == 0
    s3 = json.loads(capsys.readouterr().out)["settings"]
    assert s3["step"] == "SL3" and "noun_rows" not in s3


@pytest.mark.skipif(not _has("jax", "flax", "openpi", "sentencepiece"), reason="needs .venv-openpi-live")
def test_noun_rows_equal_the_served_noun_cell_on_a_tiny_model():
    env = dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    done = subprocess.run([sys.executable, str(PROJECT_ROOT / "tests/benchmark/sl6_tiny_noun_rows.py")],
                          cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-4000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["ok"] and out["lookup"]["noun"] == "bowl" and out["shape"] == [8, 8, 768]
    assert out["bit_equal"], out["max_abs_diff_vs_served"]
    assert out["right_wrist_mass_max"] == 0.0 and out["image_mass_head_block_max"] > 0.0
