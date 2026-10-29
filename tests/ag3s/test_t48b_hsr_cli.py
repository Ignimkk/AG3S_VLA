"""T48b — metrics by hand, CLI plumbing, and render -> extract -> score end to end with a fake
extractor (no GPU, no checkpoint load).

  - `t48b_metrics`: parity agreement, target rate, changed-vs-E0, prediction counts, exclusions,
    pixel summary — against hand-computed small tables;
  - CLI: argument validation, the AG3S_RUN_DIR refusal, run selection / groups, membership;
  - end to end on the first records of ep1807 (storage; skipped when absent): the render caches are
    written by the real `render_e0` / `render_edits` with a stand-in renderer (no OpenGL), extract
    runs with a fake `FeatureExtractor` (features = f(state)), score writes parity / metrics / CSV;
    a disagreeing fake probe stops extract at the parity gate before any edit is extracted.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

from AG3S.ag3s.experiments.studies import t48b_hsr as H  # noqa: E402
from AG3S.ag3s.experiments.studies import t48b_metrics as mx  # noqa: E402

GROUPS = {"main": ["a", "b"], "ep1807": ["a"], "main_minus_ep1807": ["b"]}


# ------------------------------------------------------------------------------- metrics


def test_parity_agreement_hand_computed():
    run = np.array(["a"] * 6 + ["b"] * 4)
    fam = np.array(["E0rec", "E0rec", "E0rec", "E0", "E0", "E0", "E0rec", "E0rec", "E0", "E0"])
    host = np.array([0, 1, 2, 0, 1, 2, 0, 1, 0, 1])
    pred = np.array([0, 1, 2, 0, 1, 0, 1, 1, 1, 2])
    r = mx.parity_agreement(run, fam, host, pred, GROUPS)
    assert (r["main"]["k"], r["main"]["n"]) == (3, 5)              # a: 2/3, b: 1/2
    assert r["main"]["per_run"]["a"]["ratio"] == pytest.approx(2 / 3)
    assert r["main"]["episode_median"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert (r["ep1807"]["k"], r["ep1807"]["n"]) == (2, 3)
    assert (r["main_minus_ep1807"]["k"], r["main_minus_ep1807"]["n"]) == (1, 2)


def test_method_metrics_hand_computed():
    run = np.array(["a"] * 7)
    fam = np.array(["E0", "E0", "E1", "E1", "E3", "E3", "E1s"])
    host = np.array([10, 11, 10, 11, 3, 4, 10])
    gt = np.array([1, 1, 0, 0, 1, 1, 0])
    gtp = np.array([1, -1, 0, 0, 1, 1, 0])
    pred = np.array([1, 1, 1, 0, 0, 1, -1])
    m = mx.method_metrics(run, fam, host, gt, gtp, pred, GROUPS)
    f = m["families"]
    assert m["n_missing"] == 1 and "E1s" not in f                  # pred -1 -> missing
    assert f["E0"]["target_rate"]["main"]["pooled"] == 1.0
    assert (f["E0"]["phys_rate"]["main"]["k"], f["E0"]["phys_rate"]["main"]["n"]) == (1, 1)
    assert (f["E1"]["target_rate"]["main"]["k"], f["E1"]["target_rate"]["main"]["n"]) == (1, 2)
    assert f["E1"]["pred_counts"]["main"] == [1, 1, 0]
    assert (f["E1"]["changed_vs_E0"]["main"]["k"], f["E1"]["changed_vs_E0"]["main"]["n"]) == (1, 2)
    assert f["E1"]["E0_pred_counts_same_hosts"] == [0, 2, 0]
    assert "changed_vs_E0" not in f["E3"]                          # hosts 3, 4 have no E0 row here
    assert f["E3"]["confusion_all"] == [[0, 0, 0], [1, 1, 0], [0, 0, 0]]


def test_exclusion_table_and_pixels():
    run = np.array(["a", "a", "a", "b"])
    fam = np.array(["E1", "E1", "E3", "E1"])
    inc = np.array([True, False, False, True])
    why = np.array(["", "penetration", "state_other", ""])
    t = mx.exclusion_table(run, fam, inc, why)
    assert t["E1"] == {"planned": 3, "included": 2, "excluded": {"penetration": 1},
                       "per_run": {"a": [1, 2], "b": [1, 1]}}
    assert t["E3"]["excluded"] == {"state_other": 1}
    mad = np.array([[0.0, 0.1, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.3]])
    mxa = np.array([[0, 3, 0], [5, 0, 0], [0, 0, 7]])
    p = mx.parity_pixels(np.array(["a", "a", "b"]), mad, mxa, ("c0", "c1", "c2"))
    assert p["all"]["c0"]["mad_max"] == 0.2 and p["all"]["c2"]["max_abs"] == 7
    assert p["per_run"]["a"]["c1"]["mad_median"] == pytest.approx(0.05)


def test_proba_change():
    run = np.array(["a"] * 4)
    fam = np.array(["E0rec", "E0", "E0rec", "E0"])
    host = np.array([0, 0, 1, 1])
    p = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
    r = mx.proba_change(run, fam, host, p)
    assert r["n"] == 2 and r["max"] == pytest.approx(0.1)


# ------------------------------------------------------------------------------- CLI


def test_parse_args_validation():
    a = H.parse_args(["--stage", "render"])
    assert a.families == list(mx.EDIT_FAMILIES) and a.extra_runs == []
    for bad in (["--stage", "render", "--max-frames", "1"], ["--stage", "extract", "--batch", "0"],
                ["--stage", "render", "--g-empty", "0.95"], ["--stage", "render", "--parity-only"],
                ["--stage", "render", "--pen-max-mm", "0"], ["--stage", "render", "--max-hosts", "0"],
                ["--stage", "render", "--runs", "nope"]):
        with pytest.raises(SystemExit):
            H.parse_args(bad)


def test_refuses_without_run_dir(monkeypatch):
    monkeypatch.delenv("AG3S_RUN_DIR", raising=False)
    with pytest.raises(SystemExit, match="AG3S_RUN_DIR"):
        H.main(["--stage", "score"])


def test_run_selection_and_groups():
    a = H.parse_args(["--stage", "score"])
    keys = [s.key for s in H.selected_runs(a)]
    assert keys == ["ep1807", "long", "t2", "headfix", "ep1800", "ep1850", "ep1900"]
    g = H.run_groups(H.selected_runs(a))
    assert g["ep1807"] == ["ep1807"] and "ep1807" not in g["main_minus_ep1807"]
    assert g["main_scene_ep1800"] == ["long", "t2", "headfix", "ep1800"] and "train_range" not in g
    a = H.parse_args(["--stage", "score", "--extra-runs", "train_range"])
    g = H.run_groups(H.selected_runs(a))
    assert g["train_range"] == ["ep0", "ep500", "ep1200"] and g["main"] == keys


def _rd(key, ep):
    return H.RunData(spec=H.ALL_RUNS[key], path=pathlib.Path("/x"), meta={}, files=[], t_step=np.zeros(0),
                     qpos=np.zeros((0, 66)), qvel=np.zeros((0, 61)), state=np.zeros((0, 16)), episode=ep,
                     prompt="", fruit="apple", hand="left", distractors=(), fingerprint="")


def test_membership():
    splits = {"train": {"start_episode_index": 0, "end_episode_index_exclusive": 1600},
              "validation": {"start_episode_index": 1600, "end_episode_index_exclusive": 1800},
              "test": {"start_episode_index": 1800, "end_episode_index_exclusive": 2000}}
    eps = {"long": 1800, "t2": 1800, "ep1807": 1807, "ep0": 0, "ep500": 500}
    m = H.membership(_rd("long", 1800), splits, eps)
    assert m["dataset_split"] == "test" and not m["vla_finetune_episode"] and not m["probe_train_episode"]
    assert m["same_scene_runs"] == ["t2"] and m["t47_head_selection_role"] == "selection"
    assert H.membership(_rd("ep1807", 1807), splits, eps)["subtask_b_closed_loop_eval"]
    m0 = H.membership(_rd("ep0", 0), splits, eps)
    assert m0["vla_finetune_episode"] and m0["probe_train_episode"] and m0["dataset_split"] == "train"
    m5 = H.membership(_rd("ep500", 500), splits, eps)
    assert m5["vla_finetune_episode"] and not m5["probe_train_episode"]      # 500 % 8 != 0


def test_episode_of():
    assert H.episode_of({"argv": ["x", "--episode-index", "1850", "--headless"]}) == 1850
    with pytest.raises(ValueError):
        H.episode_of({"argv": ["x"]})


# ------------------------------------------------------------------------------- end to end (fake GPU)

STORAGE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage"))
EP1807 = STORAGE / H.ALL_RUNS["ep1807"].rel
DATASET = STORAGE / "datasets/rby1/rby1_randomized_pick_place_16d_v1"
needs_storage = pytest.mark.skipif(not ((EP1807 / "meta.json").exists() and (DATASET / "meta").exists()),
                                   reason="shared storage not mounted")


class _StandInRenderer:
    """Returns the recorded images for the poses it is asked about (no OpenGL in this test)."""

    def __init__(self, images):
        self.images = images

    def render(self, q):
        return {c: v.copy() for c, v in self.images.items()}


class _FakeExtractor:
    def __init__(self, config, ckpt, batch=1):
        self.batch, self.load_s, self.t_prefix, self.n_per_call = batch, 0.0, [], []

    def inputs(self, state, images, prompt):
        return {"state": np.asarray(state, np.float32), "tokenized_prompt_mask": np.ones(5, bool)}

    def features(self, xs, timed=True):
        f = np.zeros((len(xs), 512), np.float32)
        for i, x in enumerate(xs):
            f[i, :16] = x["state"]
        self.t_prefix.append(0.001)
        self.n_per_call.append(len(xs))
        return f

    def decoded_prompt(self, x):
        return "Task: x, State: 1;\nAction: "

    def warmup(self, s, n=3):
        pass

    def batch_parity(self, xs):
        return None

    def memory(self):
        return {}

    def timing(self):
        return {"n_calls": len(self.t_prefix)}


@pytest.fixture()
def e2e(tmp_path, monkeypatch):
    import mujoco

    from AG3S.ag3s.experiments.studies import t48_hsr_io as io
    from AG3S.ag3s.experiments.studies import t48b_scene as sc

    root = tmp_path / "T48b"
    (root / "plan").mkdir(parents=True)
    # cached train statistics (the real ones read 200 parquet files)
    H.write_json(root / "plan" / "train_stats.json", {
        "dataset": str(DATASET), "frame_index_majority": {"0": 0, "120": 1, "300": 2},
        "gripper_closed_median_by_fruit": {"apple": 0.7231, "banana": 0.495, "orange": 0.667, "pear": 0.673}})
    ep_meta = io.episode_meta(DATASET)
    spec = H.ALL_RUNS["ep1807"]
    rd = H.load_run_data(spec, STORAGE, ep_meta, max_frames=27)
    model, _ = sc.load_record_model(rd.meta)
    idx = sc.SceneIndex.from_model(model)
    data = mujoco.MjData(model)
    an = H.analyse_run(rd, model, idx, data)
    imgs = H.record_images(rd, [0])[0]
    rdir = root / "render"
    rdir.mkdir()
    info0 = H.render_e0(an, _StandInRenderer(imgs), idx, rdir)
    ctx = sc.EditContext(idx=idx, fruit="apple", hand="left", initial_pose=sc.free_pose(an.q_ren[0], idx, "apple"),
                         g_empty=0.457, g_hold=0.7231)
    info_e = H.render_edits(an, ctx, model, data, _StandInRenderer(imgs), rdir, ["E1", "E3"], max_hosts=2)
    monkeypatch.setattr("AG3S.ag3s.experiments.studies.t48_hsr_features.FeatureExtractor", _FakeExtractor)
    monkeypatch.setattr("AG3S.ag3s.experiments.studies.t48_hsr_counterfactual.gpu_compute_processes", lambda: [])
    args = H.parse_args(["--stage", "extract", "--runs", "ep1807", "--max-frames", "27", "--families", "E1", "E3",
                         "--checkpoint", str(tmp_path), "--storage", str(STORAGE), "--batch", "2"])
    return root, args, an, info0, info_e


@needs_storage
def test_end_to_end_fake_gpu(e2e):
    root, args, an, info0, info_e = e2e
    assert info0["n"] == 27 and info0["state_max_abs"] == 0.0
    assert info_e["counts"]["E1"]["planned"] == 2 and info_e["counts"]["E3"]["planned"] == 2
    assert H.stage_extract(args, root) == 0
    gate = json.loads((root / "extract" / "parity_gate.json").read_text())
    assert gate["passed"] and gate["n"] == 27
    assert (root / "feat" / "ep1807__E0.npz").exists() and (root / "feat" / "ep1807__edits.npz").exists()
    assert H.stage_extract(args, root) == 0                     # resumable: cached files are reused
    sargs = H.parse_args(["--stage", "score", "--runs", "ep1807", "--max-frames", "27", "--families", "E1", "E3",
                          "--storage", str(STORAGE)])
    assert H.stage_score(sargs, root) == 0
    met = json.loads((root / "score" / "metrics.json").read_text())
    par = json.loads((root / "score" / "parity.json").read_text())
    assert par["passed"] is True and par["probe_agreement"]["main"]["pooled"] == 1.0
    fams = met["methods"]["probe"]["families"]
    assert fams["E0rec"]["n"] == fams["E0"]["n"] == 27
    assert set(fams) >= {"E0rec", "E0"}
    n_inc = sum(info_e["counts"][f]["included"] for f in ("E1", "E3"))
    assert sum(fams[f]["n"] for f in ("E1", "E3") if f in fams) == n_inc
    rows = (root / "score" / "predictions.csv").read_text().splitlines()
    assert len(rows) == 1 + 54 + n_inc
    # rule on E1 (closed empty hand, state 0.457 < 0.9) predicts place
    if "E1" in fams:
        assert met["methods"]["rule"]["families"]["E1"]["pred_counts"]["main"][1] == fams["E1"]["n"]


@needs_storage
def test_parity_gate_stops_extract(e2e, monkeypatch):
    root, args, *_ = e2e

    class _DisagreeingProbe:
        @classmethod
        def load(cls, path=None):
            return cls()

        def proba(self, x):
            # E0rec / E0 states are identical; make the class depend on a coin from the row order
            n = len(x)
            p = np.zeros((n, 3))
            p[np.arange(n), np.arange(n) % 2] = 1.0     # rows alternate E0rec, E0 -> always disagree
            return p

    monkeypatch.setattr("AG3S.ag3s.stages.subtask_probe.SubtaskProbe", _DisagreeingProbe)
    assert H.stage_extract(args, root) == 3
    gate = json.loads((root / "extract" / "parity_gate.json").read_text())
    assert gate["passed"] is False and gate["k"] == 0
    assert not (root / "feat" / "ep1807__edits.npz").exists()
