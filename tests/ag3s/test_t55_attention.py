"""T55 serving-attention extraction (`t55_attention`) — no GPU, no model.

Pinned (T55.task.md 추가 사전 고정 2):
  - the cell is the serving one (`attention_policy`: denoise 0, agg last, layer 8, head 2; 10 Euler steps, seed 0);
  - each frame goes through the serving `AttentionPolicy._attention_and_feature` with the client's observation
    (CHW images, float64 state, the job's prompt — the changed prompt for an instruction job) and the probe's
    ``predict`` gives the ``subtask`` the serving SafePolicy received;
  - files: shape checks, stamp (re-sim md5, prompt, cell) -> done / conflict;
  - the extract stage refuses without the shared GPU lock before any model load;
  - cell comparison numbers (argmax share, |Δ|, Pearson).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t55_attention as ta
from AG3S.ag3s.experiments.studies import t55_resim as rs


def test_serving_cell():
    c = ta.serving_cell()
    assert c == {"denoise_step": 0, "agg": "last", "layer": 8, "head": 2, "num_denoise": 10, "noise_seed": 0}


class _FakeProbe:
    name = "fake"

    def predict(self, x):
        return {"p": {"pick": 0.2, "place": 0.7, "home": 0.1}, "argmax": "place", "probe": self.name}


class _FakeAP:
    """Stands in for the serving `AttentionPolicy`: records what it was given."""

    def __init__(self):
        self.seen = []
        self.subtask_probe = _FakeProbe()

    def _attention_and_feature(self, obs):
        self.seen.append(obs)
        k = len(self.seen)
        att = {c: np.full((16, 16), k + i, np.float32) for i, c in enumerate(ta.CAMS)}
        return att, np.arange(512, dtype=np.float32)


def test_frames_attention_uses_the_client_observation():
    F = 3
    images = {c: np.random.default_rng(i).integers(0, 255, (F, 224, 224, 3), dtype=np.uint8)
              for i, c in enumerate(ta.CAMS)}
    states = np.random.default_rng(9).normal(size=(F, 16))
    ap = _FakeAP()
    att, sub, kv = ta.frames_attention(ap, images, states, "Put the pear in the basket.")
    assert att.shape == (F, 3, 16, 16) and att.dtype == np.float32 and kv.shape == (F, 512)
    assert att[1, 2, 0, 0] == 2 + 2 and sub[0]["argmax"] == "place"
    o = ap.seen[0]
    assert o["prompt"] == "Put the pear in the basket." and o["state"].dtype == np.float64
    assert o["images"]["cam_high"].shape == (3, 224, 224)
    assert np.array_equal(o["images"]["cam_left_wrist"], np.transpose(images["cam_left_wrist"][0], (2, 0, 1)))


def test_write_and_stamp(tmp_path):
    job = {"key": "k", "tag": "E0", "prompt_tag": "instr_pear", "prompt": "Put the pear in the basket.",
           "noun": "pear", "src": "/x/k__E0.npz", "out": str(tmp_path / "k__E0__instr_pear.npz")}
    st = ta.attn_stamp(job, src_md5="abc", checkpoint="ck", config="cfg", probe=None)
    assert st["cell"] == ta.serving_cell() and st["prompt"] == job["prompt"]
    assert ta.job_status(job, st) == "todo"
    att = np.zeros((4, 3, 16, 16), np.float32)
    ta.write_attention(pathlib.Path(job["out"]), attention=att, subtask=[None] * 4, kv_l4=None,
                       t_step=np.arange(4) * 8, stamp=st)
    assert ta.job_status(job, st) == "done"
    assert ta.job_status(job, {**st, "source_md5": "other"}) == "conflict"
    with pytest.raises(ValueError):
        ta.write_attention(tmp_path / "bad.npz", attention=np.zeros((4, 2, 16, 16)), subtask=[None] * 4, kv_l4=None,
                           t_step=np.arange(4), stamp=st)
    from AG3S.ag3s.experiments.studies import t55_perception as tp

    a = tp.t55_attention_inputs(pathlib.Path(job["out"]))
    assert a["attention"].shape == (4, 3, 16, 16) and a["subtask"] == [None] * 4 and a["stamp"]["prompt"] == job["prompt"]


def test_compare_cells():
    rng = np.random.default_rng(0)
    a = rng.random((5, 3, 16, 16))
    r = ta.compare_cells(a, a)
    assert r["argmax_patch_same_share"] == 1.0 and r["max_abs"] == 0.0 and r["pearson_r"] == pytest.approx(1.0)
    b = a.copy()
    b[0, 0] = np.nan                                   # a camera missing in the record is skipped
    assert ta.compare_cells(a, b)["n_maps"] == 14
    assert "shape" in ta.compare_cells(a, a[:4])


def test_extract_refuses_without_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "out_root", lambda sub="", forbid=(): tmp_path / sub if sub else tmp_path)
    (tmp_path / "resim" / "dev").mkdir(parents=True)
    from AG3S.ag3s.experiments.studies import t53_extract as ex

    monkeypatch.setattr(ex, "lock_is_held", lambda path=None: False)
    with pytest.raises(SystemExit, match="shared GPU lock"):
        ta.main(["--stage", "extract", "--set", "dev"])
    out = ta.main(["--stage", "plan", "--set", "dev"])
    assert out["n_jobs"] == 0 and (tmp_path / "attention" / "dev" / "plan.json").is_file()
