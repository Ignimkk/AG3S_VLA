"""T55 feature jobs (`t55_features`) — CPU only (no jax, no checkpoint).

Pinned:
  - instruction prompts: the noun swapped to every other fruit of the scene, template kept, the original first;
    a prompt whose fruit is not the target is refused;
  - jobs: every re-sim gets its own prompt; only E0 gets the instruction prompts; the val jobs are dev only;
  - finished files are recognised by their stamp, another stamp is a conflict (never overwritten silently);
  - the extract stage refuses unless the shared lock is held by another holder (before any GPU work);
  - the T53 parity check reads both feature files and reports max |Δ|.
"""

from __future__ import annotations

import fcntl
import json
import os
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t55_features as tf
from AG3S.ag3s.experiments.studies import t55_resim as rs

META = {1804: {"non_target_fruits": ["banana", "orange", "pear"]}}


def test_instruction_prompts():
    out = tf.instruction_prompts("Put the apple in the basket.", "apple", ["pear", "banana", "orange", "apple"])
    assert out[0] == ("orig", "Put the apple in the basket.", "apple")
    assert [o[0] for o in out[1:]] == ["instr_banana", "instr_orange", "instr_pear"]
    assert out[3][1] == "Put the pear in the basket." and out[3][2] == "pear"
    with pytest.raises(ValueError):
        tf.instruction_prompts("Put the apple in the basket.", "pear", ["banana"])


def _resim(path: pathlib.Path, **kw):
    np.savez(path, prompt=np.asarray("Put the apple in the basket."), target=np.asarray("apple"),
             episode=np.asarray(1804), **kw)


def test_resim_jobs(tmp_path):
    rd, fd = tmp_path / "resim", tmp_path / "feat"
    rd.mkdir()
    _resim(rd / "ep1804__E0.npz")
    _resim(rd / "ep1804__empty_L0O0_park.npz")
    (rd / "ep1804__E0.partial.npz").write_bytes(b"x")
    jobs = tf.resim_jobs(rd, fd, set_name="test", meta_ds=META)
    tags = sorted((j["tag"], j["prompt_tag"]) for j in jobs)
    assert tags == [("E0", "instr_banana"), ("E0", "instr_orange"), ("E0", "instr_pear"), ("E0", "orig"),
                    ("empty_L0O0_park", "orig")]
    assert all(j["out"].startswith(str(fd)) for j in jobs)
    assert len(tf.resim_jobs(rd, fd, set_name="test", meta_ds=META, instruction=False)) == 2
    assert len(tf.resim_jobs(rd, fd, set_name="test", meta_ds=META, tags=["E0"], keys=["ep1804"])) == 4
    v = tf.val_jobs(fd, [1600, 1604])
    assert [j["key"] for j in v] == ["ep_001600", "ep_001604"] and v[0]["out_dir"] == str(fd / "val")


def test_job_status(tmp_path):
    job = {"kind": "resim", "key": "k", "tag": "E0", "prompt_tag": "orig", "prompt": "p", "noun": "apple",
           "src": "s", "out": str(tmp_path / "k__E0__orig.npz")}
    st = tf.t55_stamp(job, src_md5="abc", layers=[0, 4], checkpoint="ck")
    assert tf.job_status(job, st) == "todo"
    np.savez(job["out"], t55_stamp=np.asarray(json.dumps(st, sort_keys=True)))
    assert tf.job_status(job, st) == "done"
    assert tf.job_status(job, {**st, "source_md5": "other"}) == "conflict"


def test_extract_refuses_without_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "out_root", lambda sub="", forbid=(): tmp_path / sub)
    (tmp_path / "resim" / "dev").mkdir(parents=True)
    lock = tmp_path / "gpu.lock"
    lock.write_text("")
    with pytest.raises(SystemExit, match="shared GPU lock"):
        tf.main(["--stage", "extract", "--set", "dev", "--gpu-lock", str(lock)])
    with pytest.raises(SystemExit, match="shared GPU lock"):
        tf.main(["--stage", "extract", "--set", "dev", "--gpu-lock", str(tmp_path / "absent.lock")])
    # held by someone else -> passes the probe (then stops at the GPU guard, mocked here)
    fd = os.open(lock, os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        monkeypatch.setattr(tf, "session", lambda args: (_ for _ in ()).throw(SystemExit("gpu guard reached")))
        with pytest.raises(SystemExit, match="gpu guard reached"):
            tf.main(["--stage", "extract", "--set", "dev", "--gpu-lock", str(lock)])
    finally:
        os.close(fd)
    with pytest.raises(SystemExit, match="dev data only"):
        tf.main(["--stage", "plan", "--set", "test", "--val"])


def test_plan_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "out_root", lambda sub="", forbid=(): tmp_path / sub)
    from AG3S.ag3s.experiments.studies import t53_records as rc

    monkeypatch.setattr(rc, "load_dataset_meta", lambda *a, **k: META)
    rd = tmp_path / "resim" / "test"
    rd.mkdir(parents=True)
    _resim(rd / "ep1804__E0.npz")
    info = tf.main(["--stage", "plan", "--set", "test"])
    assert info["n_resim_jobs"] == 4 and (tmp_path / "feat" / "test" / "plan.json").is_file()


def test_check_against_t53(tmp_path):
    fd, run = tmp_path / "feat", tmp_path / "run"
    (run / "verify" / "T53" / "feat").mkdir(parents=True)
    fd.mkdir()
    rng = np.random.default_rng(0)
    kv, nv, h = rng.normal(size=(3, 512)), rng.normal(size=(3, 3, 2, 4)), rng.normal(size=(3, 3, 6))
    base = dict(kv_l4=kv.astype(np.float32), noun_v=nv.astype(np.float32), noun_hidden_final=h.astype(np.float32),
                noun_layers=np.asarray([0, 4]), noun_ok=np.ones(3, bool))
    np.savez(fd / "ep1804__E0__orig.npz", **base)
    np.savez(run / "verify/T53/feat/ep1804.npz", **{**base, "kv_l4": (kv + 0.5).astype(np.float32)})
    out = tf.check_against_t53(fd, run, ["ep1804", "ep1809"])
    assert out["ep1804"]["kv_l4_max_abs"] == pytest.approx(0.5, abs=1e-6)
    assert out["ep1804"]["noun_v_max_abs"] == 0.0 and out["ep1804"]["noun_ok_equal"]
    assert "missing" in out["ep1809"]
