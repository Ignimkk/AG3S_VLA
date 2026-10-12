"""T50 — sensitivity (①), counterfactual probe (②), controls (②′) and the T50b extract plumbing.

No GPU, no checkpoint weights. Pinned on synthetic data / synthetic T48-format caches:
  - GroupKFold replica (sklearn 1.5 assignment) and the numpy/scipy logistic regression (gradient,
    objective, C choice);
  - relative-L2 distances, r = d(C0,C1)/d(C0,C1p) per part, adjacent-frame baseline, episode-then-pooled
    aggregation — against hand computation;
  - training rows: original : C1 = 1 : 1, C1 -> pick, one C1 per host, seeded and per-unit stable;
    standard = C0 only; shuffled = random C1 labels;
  - evaluation (C0 accuracy, C1 instruction accuracy, rule check), nested CV end to end;
  - cache loader (digest check, family filter), CLI validation / AG3S_RUN_DIR refusal, stage outputs;
  - the extraction loop with a fake extractor (T48 cache format, resume, digest refusal);
  - checkpoint metadata parsing (nnx "value" suffix).
Shared storage (skipped when absent): LogReg parity with `kv_L4_v1` on SUBTASK-b features; the T48
LoRA cache loads with consistent digests. openpi (skipped without flax): base / LoRA parameter trees
fit their models and the transform chain is identical.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t48_hsr_core as t48core
from AG3S.ag3s.experiments.studies import t48_hsr_io as io
from AG3S.ag3s.experiments.studies import t50_core as core
from AG3S.ag3s.experiments.studies import t50_extract as ex
from AG3S.ag3s.experiments.studies import t50_hsr_rep as cli

TEMPLATES = sorted(["Pick up the {fruit} and place it in the basket.", "Put the {fruit} in the basket.",
                    "Place the {fruit} into the basket."])
REPS = t48core.GripperReps(open=1.0, closed=0.67)
Y = [0, 0, 0, 1, 1, 1, 2, 2, 2]


def make_unit(y=Y, uid="ep_000001", prompt="Put the apple in the basket.", target="apple") -> t48core.Unit:
    y = np.asarray(y, np.int64)
    n = len(y)
    seq = np.zeros((n, 16))
    seq[:, 7] = seq[:, 15] = 1.0
    seq[:, :7] = np.arange(n)[:, None] * 0.01
    seq[y == t48core.PLACE, 7] = 0.7
    return t48core.Unit(unit_id=uid, set_name="val", prompt=prompt, target=target,
                        distractors=("banana", "orange", "pear"), gripper_index=7, seq_states=seq,
                        rows=np.arange(n), clock=np.arange(n) * 6, y=y, proxy_cadence=1,
                        seed_key=io.stable_key(uid), extra={"episode": 1})


def synth_features(table: t48core.VariantTable, y: np.ndarray, rng, *, noise=0.05) -> np.ndarray:
    """text half: [is-C1, one-hot(host class) small]; image half: one-hot of the image-source class.
    A linear probe can read 'phase' from the image half and 'prompt names a distractor' from text."""
    f = np.zeros((len(table), 512))
    for blk in (slice(0, 8), slice(256, 264)):          # noise on a few dims only (n << 512)
        f[:, blk] += rng.normal(0, noise, (len(table), 8))
    f[:, 0] += (table.variant == "C1") * 3.0
    f[:, 1] += 1.0
    for c in range(3):
        f[:, 256 + c] += 3.0 * (y[table.image_row] == c)
    f[:, 300] += 2.0
    return f.astype(np.float32)


def write_cache(d: pathlib.Path, n_units=10, *, seed=0, families=t48core.VARIANTS) -> list[str]:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = []
    for k in range(n_units):
        uid = f"ep_{1600 + 4 * k:06d}"
        u = make_unit(uid=uid)
        t = t48core.build_variants(u, TEMPLATES, REPS, seed=0, families=families)
        arrs = t.as_arrays("plan__")
        arrs.update(feature=synth_features(t, u.y, rng), plan_digest=np.asarray(t.digest()),
                    unit_id=np.asarray(uid), set_name=np.asarray("val"), clock=u.clock, y=u.y, rows=u.rows)
        np.savez(d / f"{uid}.npz", **arrs)
        ids.append(uid)
    return ids


# ------------------------------------------------------------------------------- GroupKFold / LogReg


def test_group_kfold_matches_sklearn_assignment_by_hand():
    # group sizes: a=5, b=3, c=3, d=1 -> order (largest first; ties as np.argsort[::-1]) and lightest fold
    groups = np.array(list("aaaaabbbcccd"))
    folds = core.group_kfold(groups, 3)
    test_groups = [sorted(set(groups[te])) for _, te in folds]
    sizes = np.bincount(np.unique(groups, return_inverse=True)[1])
    order = np.argsort(sizes)[::-1]          # the same call sklearn makes
    uniq = np.unique(groups)
    expect, load = {0: [], 1: [], 2: []}, np.zeros(3)
    for gi in order:
        f = int(np.argmin(load))
        load[f] += sizes[gi]
        expect[f].append(uniq[gi])
    assert test_groups == [sorted(expect[f]) for f in range(3)]
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te]) and len(tr) + len(te) == len(groups)
    with pytest.raises(ValueError):
        core.group_kfold(groups, 5)


def test_logreg_gradient_and_objective():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = rng.integers(0, 3, 60)
    m = core.fit_logreg(X, y, C=0.5)
    assert m.converged and m.coef.shape == (3, 4) and m.intercept.shape == (3,)
    # stationarity of the sklearn objective at the solution: grad = (P - Y)^T Xs / n + coef / (C n)
    Xs = (X - m.mean) / m.scale
    P = m.proba(X)
    Yh = np.eye(3)[y]
    g = (P - Yh).T @ Xs / len(X) + m.coef / (0.5 * len(X))
    assert np.abs(g).max() < 1e-4
    assert np.abs((P - Yh).sum(0)).max() / len(X) < 1e-4
    np.testing.assert_allclose(m.mean, X.mean(0))
    np.testing.assert_allclose(m.scale, X.std(0))


def test_logreg_separable_and_constant_feature():
    X = np.zeros((30, 3))
    y = np.repeat([0, 1, 2], 10)
    X[np.arange(30), y] = 1.0
    Xc = np.c_[X, np.ones(30)]                 # constant column -> scale 1 (StandardScaler)
    m = core.fit_logreg(Xc, y, C=1.0)
    assert m.scale[-1] == 1.0
    assert (m.predict(Xc) == y).all()
    with pytest.raises(ValueError):
        core.fit_logreg(Xc, np.r_[y[:-1], 3], C=1.0)


def test_choose_c_takes_first_best():
    X = np.zeros((20, 2))
    y = np.r_[np.zeros(10, int), np.ones(10, int)]
    X[:, 0] = y
    groups = np.repeat(np.arange(10), 2)
    C, scores = core.choose_C(X, y, groups, grid=(1e-3, 1.0), n_splits=5)
    assert scores[1e-3] == scores[1.0] == 1.0 and C == 1e-3


# ------------------------------------------------------------------------------- cache loader


def test_load_feature_dir_rows_digest_and_filters(tmp_path):
    ids = write_cache(tmp_path / "val", 3)
    fs = core.load_feature_dir(tmp_path / "val")
    assert fs.units == ids and len(fs) == 3 * 9 * 9          # 9 frames x (1 + 3 + 2 + 1 + 1 + 1)
    assert set(fs.digests) == set(ids)
    c1 = fs.variant == "C1"
    assert (fs.gt[c1] == core.PICK).all() and (fs.y_host[fs.variant == "C0"] == np.tile(Y, 3)).all()
    assert (fs.clock == fs.host * 6).all()
    f2 = core.load_feature_dir(tmp_path / "val", max_units=2, families=("C0", "C1"))
    assert f2.units == ids[:2] and set(f2.variant) == {"C0", "C1"} and len(f2) == 2 * 9 * 4
    # tampered rows -> refused
    p = tmp_path / "val" / f"{ids[0]}.npz"
    z = dict(np.load(p))
    z["plan__gt"] = np.zeros_like(z["plan__gt"])
    np.savez(p, **z)
    with pytest.raises(ValueError, match="plan_digest"):
        core.load_feature_dir(tmp_path / "val")
    with pytest.raises(FileNotFoundError):
        core.load_feature_dir(tmp_path / "empty")


# ------------------------------------------------------------------------------- ① sensitivity


def _tiny_set() -> core.FeatureSet:
    """One unit, two hosts; hand-made features (dims 0-255 text, 256-511 image)."""
    rows = []

    def f(t, i):
        v = np.zeros(512)
        v[0], v[256] = t, i
        return v

    # host 0 (pick): C0 text 1 image 1; C1 x3 text 2,2,4 image 1; C1p x2 text 1.5,1.5 image 1.5
    # host 1 (place): C0 text 2 image 2; C1 x3 text 3 image 2; C1p x2 text 3 image 3
    spec = [(0, 0, "C0", "", f(1, 1)), (0, 0, "C1", "a", f(2, 1)), (0, 0, "C1", "b", f(2, 1)),
            (0, 0, "C1", "c", f(4, 1)), (0, 0, "C1p", "T0", f(1.5, 1.5)), (0, 0, "C1p", "T1", f(1.5, 1.5)),
            (1, 1, "C0", "", f(2, 2)), (1, 1, "C1", "a", f(3, 2)), (1, 1, "C1", "b", f(3, 2)),
            (1, 1, "C1", "c", f(3, 2)), (1, 1, "C1p", "T0", f(3, 3)), (1, 1, "C1p", "T1", f(3, 3))]
    for host, y, var, sub, x in spec:
        rows.append((host, y, var, sub, x))
    n = len(rows)
    return core.FeatureSet(unit=np.full(n, "u"), variant=np.array([r[2] for r in rows]),
                           sub=np.array([r[3] for r in rows]), host=np.array([r[0] for r in rows]),
                           clock=np.array([r[0] * 6 for r in rows]), y_host=np.array([r[1] for r in rows]),
                           gt=np.array([0 if r[2] == "C1" else r[1] for r in rows]),
                           y_image=np.array([r[1] for r in rows]), y_state=np.array([r[1] for r in rows]),
                           X=np.stack([r[4] for r in rows]).astype(np.float32), digests={"u": "x"})


def test_sensitivity_hand_computed():
    fr = core.sensitivity_frames(_tiny_set())
    assert fr["host"].tolist() == [0, 1] and fr["y"].tolist() == [0, 1]
    # host 0 text: d_c1 = mean(1, 1, 3) / 1 = 5/3; d_c1p = 0.5; r = 10/3; image: d_c1 = 0, d_c1p = 0.5
    assert fr["text__d_c1"][0] == pytest.approx(5 / 3)
    assert fr["text__d_c1p"][0] == pytest.approx(0.5)
    assert fr["text__r"][0] == pytest.approx(10 / 3)
    assert fr["image__d_c1"][0] == pytest.approx(0.0) and fr["image__r"][0] == pytest.approx(0.0)
    # full: ||(1,0)|| / ||(1,1)|| etc.
    d1 = np.mean([1, 1, 3]) / np.sqrt(2)
    d1p = np.sqrt(0.5) / np.sqrt(2)
    assert fr["full__r"][0] == pytest.approx(d1 / d1p)
    # adjacent: host 0 -> host 1 C0 (2,2) vs (1,1): text 1/1, image 1/1, full sqrt2/sqrt2
    assert fr["text__d_adj"][0] == pytest.approx(1.0) and fr["full__d_adj"][0] == pytest.approx(1.0)
    assert np.isnan(fr["text__d_adj"][1])                     # last frame has no next
    # host 1 text: d_c1 = 1/2, d_c1p = 1/2, r = 1
    assert fr["text__r"][1] == pytest.approx(1.0)


def test_episode_then_pooled_hand_computed():
    units = np.array(["a", "a", "a", "b", "b"])
    v = np.array([1.0, 2.0, 9.0, 4.0, np.nan])
    s = core.episode_then_pooled(units, v)
    # episode medians: a 2, b 4 -> median 3, q25 2.5, q75 3.5
    assert s["episode"] == {"n_episodes": 2, "median": 3.0, "q25": 2.5, "q75": 3.5}
    assert s["pooled_frames"]["n"] == 4 and s["pooled_frames"]["median"] == pytest.approx(3.0)
    summ = core.sensitivity_summary(core.sensitivity_frames(_tiny_set()))
    assert summ["text"]["r"]["pick"]["episode"]["median"] == pytest.approx(10 / 3)
    assert summ["text"]["r"]["home"]["episode"]["n_episodes"] == 0


# ------------------------------------------------------------------------------- ② training rows / eval


def test_training_rows_counterfactual_standard_shuffled(tmp_path):
    write_cache(tmp_path / "val", 4)
    fs = core.load_feature_dir(tmp_path / "val")
    units = fs.units[:3]
    idx, lab = core.training_rows(fs, units, "counterfactual", seed=0)
    c0 = idx[fs.variant[idx] == "C0"]
    c1 = idx[fs.variant[idx] == "C1"]
    assert len(c0) == len(c1) == 3 * 9                         # 1 : 1
    assert (lab[fs.variant[idx] == "C1"] == core.PICK).all()
    assert (lab[fs.variant[idx] == "C0"] == fs.gt[c0]).all()   # original answers kept
    # one C1 per (unit, host)
    keys = [(fs.unit[i], int(fs.host[i])) for i in c1]
    assert len(set(keys)) == len(keys) == len({(fs.unit[i], int(fs.host[i])) for i in c0})
    assert set(fs.unit[idx]) == set(units)
    # reproducible, per-unit stable (adding a unit does not change the others' choice), seed matters
    idx2, _ = core.training_rows(fs, fs.units, "counterfactual", seed=0)
    assert set(c1) <= set(idx2)
    idx3, _ = core.training_rows(fs, units, "counterfactual", seed=0)
    assert np.array_equal(idx, idx3)
    assert any(not np.array_equal(core.training_rows(fs, units, "counterfactual", seed=s)[0], idx)
               for s in range(1, 4))
    assert len(set(fs.sub[c1].tolist())) == 3                  # all three distractors get drawn
    # standard
    i_s, l_s = core.training_rows(fs, units, "standard", seed=0)
    assert set(fs.variant[i_s]) == {"C0"} and len(i_s) == 27
    # shuffled: same rows, C1 labels random in {0,1,2}, seeded
    i_h, l_h = core.training_rows(fs, units, "shuffled", seed=0)
    assert np.array_equal(i_h, idx)
    m1 = fs.variant[i_h] == "C1"
    assert set(l_h[m1]) <= {0, 1, 2} and len(set(l_h[m1])) > 1
    assert np.array_equal(l_h[~m1], lab[~m1])
    assert np.array_equal(core.training_rows(fs, units, "shuffled", seed=0)[1], l_h)
    with pytest.raises(ValueError):
        core.training_rows(fs, units, "bogus", seed=0)


def test_evaluate_and_rule_check_hand_computed():
    fs = _tiny_set()
    # predict: C0 host0 pick (right), C0 host1 place (right); C1 host0 all pick; C1 host1 pick, place, pick
    pred = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1])
    ev = core.evaluate(fs, pred)
    assert ev["accuracy"]["C0"]["pooled"] == 1.0
    assert ev["C1_instr"]["k"] == 2 and ev["C1_instr"]["n"] == 3      # host 1 is place
    assert ev["C1_to_pick"]["pick"]["pooled"] == 1.0
    assert ev["accuracy"]["C1p"]["pooled"] == 1.0                    # 0,0 on host 0; 1,1 on host 1
    rc = core.rule_check(ev)
    assert rc["pooled"]["pass"] is False and rc["pooled"]["C1_instr"] == pytest.approx(2 / 3)
    pred2 = pred.copy()
    pred2[8] = 0
    rc2 = core.rule_check(core.evaluate(fs, pred2))
    assert rc2["pooled"]["pass"] is True and rc2["episode_median"]["pass"] is True
    ev3 = core.evaluate(fs, np.full(len(fs), -1))
    assert ev3["n_evaluated"] == 0 and core.rule_check(ev3)["pooled"]["pass"] is None


def test_valcv_end_to_end_counterfactual_reads_the_prompt(tmp_path):
    write_cache(tmp_path / "val", 10)
    fs = core.load_feature_dir(tmp_path / "val")
    folds = core.outer_folds(fs)
    assert sorted(u for f in folds for u in f) == fs.units and all(len(f) == 2 for f in folds)
    r = core.run_valcv(fs, "full", "counterfactual", seed=0, folds=folds, grid=(0.1, 1.0))
    ev = core.evaluate(fs, r["pred"])
    assert (r["pred"] >= 0).all() and len(r["folds"]) == 5
    assert ev["accuracy"]["C0"]["pooled"] > 0.95 and ev["C1_instr"]["pooled"] > 0.95
    for f in r["folds"]:
        assert f["train_label_counts"][0] == f["n_train_rows"] - f["train_label_counts"][1] - f["train_label_counts"][2]
        assert not set(f["held_out_units"]) & set(fs.unit[core.training_rows(fs, [u for u in fs.units if u not in f["held_out_units"]], "counterfactual", 0)[0]])
    std = core.evaluate(fs, core.run_valcv(fs, "full", "standard", folds=folds, grid=(0.1,))["pred"])
    assert std["C1_instr"]["pooled"] < 0.2                     # trained on C0 only: follows the image
    img = core.evaluate(fs, core.run_valcv(fs, "image", "counterfactual", folds=folds, grid=(0.1,))["pred"])
    assert img["C1_instr"]["pooled"] < 0.8 or img["accuracy"]["C0"]["pooled"] < 0.95  # image half cannot see C1


def test_trainval_end_to_end(tmp_path):
    write_cache(tmp_path / "train", 6, seed=1, families=("C0", "C1"))
    write_cache(tmp_path / "val", 2, seed=2)
    tr = core.load_feature_dir(tmp_path / "train")
    va = core.load_feature_dir(tmp_path / "val")
    r = core.run_trainval(tr, va, "full", "counterfactual", grid=(0.1,))
    ev = core.evaluate(va, r["pred"])
    assert ev["C1_instr"]["pooled"] > 0.95 and r["folds"][0]["n_train_rows"] == 2 * 6 * 9


# ------------------------------------------------------------------------------- CLI


def test_cli_argument_validation():
    a = cli.parse_args(["--stage", "probe"])
    assert a.model == "lora" and a.mode == "valcv" and a.parts == ["full", "text", "image"] and a.batch == 4
    assert a.sets == ["val", "train"] and a.seed == 0 and cli.cpu_tag(a) == "full"
    assert cli.cpu_tag(cli.parse_args(["--stage", "sensitivity", "--max-units", "3"])) == "smoke_first3"
    for bad in (["--stage", "bogus"], ["--stage", "probe", "--max-units", "4"], ["--stage", "extract", "--batch", "0"],
                ["--stage", "probe", "--mode", "x"], ["--stage", "probe", "--parts", "half"],
                ["--stage", "extract", "--sets", "ep1807"], ["--stage", "probe", "--model", "pi0"]):
        with pytest.raises(SystemExit):
            cli.parse_args(bad)


def test_cli_refuses_without_run_dir(monkeypatch):
    monkeypatch.delenv("AG3S_RUN_DIR", raising=False)
    with pytest.raises(SystemExit, match="AG3S_RUN_DIR"):
        cli.out_root()
    with pytest.raises(SystemExit, match="AG3S_RUN_DIR"):
        cli.main(["--stage", "sensitivity"])


def test_default_feature_dirs(tmp_path):
    a = cli.parse_args(["--stage", "probe", "--storage", str(tmp_path)])
    assert cli.val_feat_dir(a, tmp_path / "o") == tmp_path / cli.T48_LORA_VAL_FEAT_REL
    b = cli.parse_args(["--stage", "probe", "--model", "base"])
    assert cli.val_feat_dir(b, tmp_path / "o") == tmp_path / "o" / "base" / "feat" / "full" / "val"
    assert cli.train_feat_dir(b, tmp_path / "o") == tmp_path / "o" / "base" / "feat" / "full" / "train"


def test_stages_write_outputs_and_compare(tmp_path):
    write_cache(tmp_path / "val", 10)
    write_cache(tmp_path / "train", 5, seed=3, families=("C0", "C1"))
    root = tmp_path / "out"
    for model in ("lora", "base"):
        a = cli.parse_args(["--stage", "sensitivity", "--model", model, "--val-feat", str(tmp_path / "val")])
        s = cli.stage_sensitivity(a, root)
        assert s["n_frames"] == 90 and s["n_frames_skipped"] == 0
        a = cli.parse_args(["--stage", "probe", "--model", model, "--val-feat", str(tmp_path / "val"),
                            "--train-feat", str(tmp_path / "train"), "--mode", "trainval",
                            "--parts", "full", "--mixes", "counterfactual", "standard"])
        p = cli.stage_probe(a, root)
        assert set(p["cells"]) == {"full__counterfactual", "full__standard"}
        assert p["cells"]["full__counterfactual"]["role"].startswith("primary")
    out = root / "lora" / "full"
    assert (out / "sensitivity" / "frames.csv").exists() and (out / "probe" / "predictions_trainval.csv").exists()
    lines = (out / "probe" / "predictions_trainval.csv").read_text().splitlines()
    assert len(lines) == 1 + 10 * 81
    c = cli.stage_compare(cli.parse_args(["--stage", "compare"]), root)
    assert len(c["rows"]) == 4 and c["val_plan_digests_equal"]["n_equal"] == 10
    assert (root / "compare" / "compare_full.csv").exists()


# ------------------------------------------------------------------------------- extract plumbing (no GPU)


class FakeExtractor:
    """Stands in for FeatureExtractor: inputs -> dict, features -> deterministic hash of the input."""

    def __init__(self, q01, q99):
        self.t_prefix, self.n_per_call, self.q01, self.q99 = [], [], q01, q99
        self.load_s = 0.0

    def inputs(self, state, images, prompt):
        bins = t48core.discretize_state(state, self.q01, self.q99)
        return {"prompt": prompt, "bins": bins, "img": float(np.asarray(images["cam_high"]).mean()),
                "tokenized_prompt_mask": np.ones(10 + len(prompt.split()), bool)}

    def decoded_prompt(self, x):
        return f"Task: {x['prompt']}, State: {' '.join(map(str, x['bins']))};\nAction: "

    def features(self, xs, timed=True):
        f = np.stack([np.full(512, x["img"] + len(x["prompt"]), np.float32) for x in xs])
        if timed:
            self.t_prefix.append(0.001)
            self.n_per_call.append(len(xs))
        return f

    def warmup(self, sample, n=3):
        pass

    def batch_parity(self, samples):
        return {"n": len(samples), "rel_l2_max": 0.0}

    def memory(self):
        return {}


def test_run_extraction_writes_t48_cache_and_resumes(tmp_path):
    q01, q99 = np.zeros(16), np.ones(16)
    units = []
    for k in range(2):
        u = make_unit(uid=f"ep_{k:06d}")
        units.append((u, t48core.build_variants(u, TEMPLATES, REPS, families=ex.TRAIN_FAMILIES)))

    def images_fn(u, rows):
        return {int(r): {c: np.full((4, 4, 3), int(r), np.uint8) for c in io.CAMS} for r in rows}

    fx = FakeExtractor(q01, q99)
    meta = {}
    ex.run_extraction(fx, units, tmp_path / "feat", images_fn=images_fn, batch=4, warmup=1, q01=q01, q99=q99,
                      set_name="train", meta=meta)
    assert set(meta["units"]) == {"train/ep_000000", "train/ep_000001"} and meta["batch_parity"]["n"] == 4
    fs = core.load_feature_dir(tmp_path / "feat")
    assert set(fs.variant) == {"C0", "C1"} and len(fs) == 2 * 9 * 4
    with np.load(tmp_path / "feat" / "ep_000000.npz") as z:
        assert int(z["state_bins_replica_mismatch"]) == 0 and str(z["set_name"]) == "train"
        np.testing.assert_allclose(z["feature"][:, 0], z["plan__image_row"] + np.char.str_len(z["plan__prompt"]))
    # resume: same digest -> skipped (no new prefix calls)
    n = len(fx.t_prefix)
    ex.run_extraction(fx, units, tmp_path / "feat", images_fn=images_fn, batch=4, warmup=1, q01=q01, q99=q99,
                      set_name="train", meta={})
    assert len(fx.t_prefix) == n
    # different plan for the same unit id -> refused, file untouched
    u0 = make_unit(uid="ep_000000")
    t_other = t48core.build_variants(u0, TEMPLATES, REPS, families=("C0",))
    with pytest.raises(SystemExit, match="different plan digest"):
        ex.run_extraction(fx, [(u0, t_other)], tmp_path / "feat", images_fn=images_fn, batch=4, warmup=1,
                          q01=q01, q99=q99, set_name="train", meta={})


def test_checkpoint_param_shapes_strips_value_suffix(tmp_path):
    def meta(keys):
        return {"tree_metadata": {str(k): {"value_metadata": {"write_shape": s}} for k, s in keys}}

    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "_METADATA").write_text(json.dumps(meta([(("params", "x", "kernel", "value"), [2, 3]),
                                                               (("params", "y", "value"), [4])])))
    assert ex.checkpoint_param_shapes(tmp_path / "a") == {("x", "kernel"): (2, 3), ("y",): (4,)}
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "_METADATA").write_text(json.dumps(meta([(("params", "x", "kernel"), [2, 3])])))
    assert ex.checkpoint_param_shapes(tmp_path / "b") == {("x", "kernel"): (2, 3)}
    r = ex.compare_param_trees({("x",): (2,), ("z",): (1,)}, {("x",): (3,), ("w",): (1,)})
    assert not r["loadable"] and r["missing_in_checkpoint"] == [["z"]] and r["n_extra"] == 1
    assert r["shape_mismatch"][0]["key"] == ["x"]


# ------------------------------------------------------------------------------- shared storage


PATHS = io.Paths.from_storage()
SUBTASK_B = PATHS.subtask_b_feat
have_subtask_b = pytest.mark.skipif(not (SUBTASK_B / "train").exists(), reason="SUBTASK-b feat not on this machine")
T48_LORA = cli.storage_root(cli.parse_args(["--stage", "probe"])) / cli.T48_LORA_VAL_FEAT_REL
have_t48 = pytest.mark.skipif(not T48_LORA.exists(), reason="T48 LoRA cache not on this machine")


@have_subtask_b
def test_logreg_reproduces_kv_l4_v1_on_subtask_b_features():
    """The numpy/scipy fit with C = 0.01 (`kv_L4_v1.json` model.C) on SUBTASK-b train reproduces the
    exported sklearn probe: val accuracy 0.98078 and probabilities within 1e-3."""
    import glob

    from AG3S.ag3s.stages.subtask_probe import SubtaskProbe

    def load(split):
        X, y = [], []
        for f in sorted(glob.glob(str(SUBTASK_B / split / "*.npz"))):
            with np.load(f) as z:
                X.append(z["normal__kv_L4"])
                y.append(t48core.phase_to_class(z["phase_index"]))
        X, y = np.concatenate(X), np.concatenate(y)
        return X[y >= 0], y[y >= 0]

    Xt, yt = load("train")
    Xv, yv = load("val")
    assert len(Xt) == 10401 and len(Xv) == 2601
    m = core.fit_logreg(Xt, yt, 0.01)
    ref = SubtaskProbe.load()
    p, pr = m.proba(Xv), ref.proba(Xv)
    assert (p.argmax(1) == yv).mean() == pytest.approx(0.9807766243752403, abs=1e-12)
    assert np.abs(p - pr).max() < 1e-3
    assert np.abs(m.coef - ref.coef).max() < 1e-4


@have_t48
def test_t48_lora_cache_loads_with_consistent_digests():
    fs = core.load_feature_dir(T48_LORA, max_units=2)
    assert fs.units == ["ep_001600", "ep_001604"]
    assert sorted(set(fs.variant)) == sorted(t48core.VARIANTS)
    n0 = int((fs.variant == "C0").sum())
    assert int((fs.variant == "C1").sum()) == 3 * n0 and int((fs.variant == "C1p").sum()) == 2 * n0
    fr = core.sensitivity_frames(fs)
    assert len(fr["unit"]) == n0 and np.isfinite(fr["full__r"]).all()


# ------------------------------------------------------------------------------- openpi (CPU, abstract)


@pytest.mark.skipif(not PATHS.checkpoint.exists() or not ex.base_checkpoint().exists(),
                    reason="checkpoints not on this machine")
def test_param_trees_fit_and_transforms_equal():
    pytest.importorskip("flax")
    pytest.importorskip("openpi")
    res = ex.check_params(io.CONFIG_NAME, PATHS.checkpoint, ex.base_checkpoint())
    assert res["lora"]["loadable"] and res["base"]["loadable"] and res["transforms_equal"]
    assert res["base"]["model_config"]["paligemma_variant"] == "gemma_2b"
    assert res["base"]["model_config"]["action_dim"] == res["lora"]["model_config"]["action_dim"] == "32"
