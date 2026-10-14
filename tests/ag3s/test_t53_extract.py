"""T53 extraction CLI (`t53_extract`) — synthetic data, no GPU, no checkpoint; one MuJoCo smoke on a
stored T47 record when storage is present.

Pinned:
  - grasp-state sequence (pre / held / placed / post / held_other_hand) from T48b physical states;
  - the noun-token feature: `noun_token_impl` gathers V[layer, batch 0, 768 + text index, kv head 0] and
    the final normed prefix output at the same rows (stand-in model, numpy), refuses a prefix whose image
    token count is not 768; `feature_frame` asks for T49's rows, reorders to (Q, nL, H), and a frame whose
    noun is missing gets zeros with the learned shapes; `assemble_feat` + `feat_problems` round trip;
  - the noun record check (stamp / t_step / failed frames) and the array format check;
  - the GPU-lock probe and every CLI refusal before any GPU work (no lock, bad layers / config / parity
    record / run dir / records);
  - ``--stage layout`` and ``--stage check`` on a synthetic rollout;
  - real record (skipped without storage): `grasp_predicates` on ep1807 (T48b predicate, 75 frames).
"""

from __future__ import annotations

import fcntl
import json
import os
import pathlib
import types

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t49_noun_query_core as core
from AG3S.ag3s.experiments.studies import t51_records as rc51
from AG3S.ag3s.experiments.studies import t53_extract as ex
from AG3S.ag3s.experiments.studies import t53_records as rc

from tests.ag3s.test_t53_records import KEY, PROMPT, _dataset_rows, make_rollout, write_attention

IDS = {"apple": 15491, "banana": 31985, "basket": 12220, "orange": 10436, "pear": 21338}
BOS = 2


@pytest.fixture(autouse=True)
def _isolated(tmp_path):
    rc.configure(t53_dir=tmp_path / "run" / "verify" / "T53")
    yield
    rc.configure()


# ------------------------------------------------------------------------------ grasp state


def test_grasp_state_sequence():
    st = ["on_table", "grasping", "held", "held", "grasping", "in_crate", "in_crate"]
    hd = [None, "right", "right", "right", "right", None, None]
    assert ex.grasp_state_sequence(st, hd, "right") == ["pre", "pre", "held", "held", "post", "placed",
                                                        "placed"]
    # held by the other hand never starts "post"; a drop after holding is "post"
    st2 = ["on_table", "held", "other", "on_table", "held", "on_table"]
    hd2 = [None, "left", None, None, "right", None]
    assert ex.grasp_state_sequence(st2, hd2, "right") == ["pre", "held_other_hand", "pre", "pre", "held",
                                                          "post"]
    assert ex.state_counts(["pre", "held", "pre"]) == {"pre": 2, "held": 1, "placed": 0, "post": 0,
                                                        "held_other_hand": 0}


# ------------------------------------------------------------------------------ noun-token feature

N_L, H, D, T = 4, 5, 3, 12


class FakeLLM:
    """V[l, 0, s, 0, h] = 1000 l + s + 0.01 h ; normed = 2 * tokens."""

    def __call__(self, embedded, mask, positions):
        tok = np.asarray(embedded[0], np.float32)
        b, s, _ = tok.shape
        v = np.zeros((N_L, b, s, 1, H), np.float32)
        for l in range(N_L):
            v[l, :, :, 0, :] = 1000 * l + np.arange(s)[None, :, None] + 0.01 * np.arange(H)
        return [2 * tok, None], (np.zeros_like(v), v), "probs"


class FakeModel:
    def __init__(self, n_image=core.N_IMAGE_TOKENS):
        self.n_image = n_image
        self.PaliGemma = types.SimpleNamespace(llm=FakeLLM())

    def embed_prefix(self, obs):
        s = self.n_image + obs.tokenized_prompt.shape[1]
        tokens = np.broadcast_to(np.arange(s, dtype=np.float32)[None, :, None], (1, s, D)).copy()
        return tokens, np.ones((1, s), bool), np.zeros((1, s), bool)


def test_noun_token_impl_gathers_rows_and_layers():
    obs = types.SimpleNamespace(tokenized_prompt=np.zeros((1, T), np.int32))
    rows = np.asarray([770, 773, 779])
    layers = np.asarray([1, 3])
    out = ex.noun_token_impl(FakeModel(), obs, rows, layers, make_attn_mask=lambda m, a: None, xp=np)
    assert out["v"].shape == (2, 3, H) and out["v"].dtype == np.float32
    np.testing.assert_allclose(out["v"][1, 2], 3000 + 779 + 0.01 * np.arange(H), rtol=0, atol=1e-3)
    np.testing.assert_allclose(out["v"][0, 0, 0], 1000 + 770)
    assert out["hidden_final"].shape == (3, D)
    np.testing.assert_allclose(out["hidden_final"][:, 0], 2 * rows)
    with pytest.raises(ValueError, match="image tokens"):
        ex.noun_token_impl(FakeModel(n_image=700), obs, rows, layers,
                           make_attn_mask=lambda m, a: None, xp=np)


def tokens_for(*pieces, pad_to=16):
    ids = [BOS] + [IDS.get(p, 900 + i) for i, p in enumerate(pieces)]
    toks = np.zeros(pad_to, np.int64)
    toks[: len(ids)] = ids
    mask = np.zeros(pad_to, bool)
    mask[: len(ids)] = True
    return {"tokenized_prompt": toks, "tokenized_prompt_mask": mask}


def fake_fns(n_layers=N_L):
    calls = []

    def kv_fn(inputs):
        return np.arange(512, dtype=np.float32)

    def obs_fn(inputs):
        return "OBS"

    def noun_fn(obs, rows):
        calls.append(np.asarray(rows).tolist())
        r = len(rows)
        v = np.zeros((n_layers, r, H), np.float32) + np.asarray(rows, np.float32)[None, :, None]
        return {"v": v, "hidden_final": np.ones((r, D), np.float32)}

    return kv_fn, obs_fn, noun_fn, calls


def test_feature_frame_rows_and_layout():
    kv_fn, obs_fn, noun_fn, calls = fake_fns()
    inputs = tokens_for("Pick", "up", "the", "banana", "and", "place", "it", "in", "the", "basket", ".")
    f = ex.feature_frame(inputs, kv_fn=kv_fn, obs_fn=obs_fn, noun_fn=noun_fn, noun="banana", ids=IDS,
                         bos=BOS, n_layers=N_L)
    assert f["noun_ok"] and f["noun_error"] == ""
    assert f["noun_token_index"].tolist() == [4, 10, 11]  # banana, basket, last valid token
    assert calls == [[768 + 4, 768 + 10, 768 + 11]]
    assert f["noun_prefix_row"].tolist() == calls[0]
    assert f["noun_token_id"].tolist() == [IDS["banana"], IDS["basket"], 900 + 10]
    assert f["noun_v"].shape == (3, N_L, H)  # (Q, nL, H)
    assert f["noun_v"][1, 2, 0] == 768 + 10
    assert f["kv_l4"].shape == (512,) and f["n_text_valid"] == 12


def test_feature_frame_missing_noun_gets_zeros():
    kv_fn, obs_fn, noun_fn, calls = fake_fns()
    bad = tokens_for("Put", "the", "fruit", "in", "the", "basket", ".")
    state = {}
    f = ex.feature_frame(bad, kv_fn=kv_fn, obs_fn=obs_fn, noun_fn=noun_fn, noun="pear", ids=IDS, bos=BOS,
                         n_layers=N_L, state=state)
    assert not f["noun_ok"] and "target" in f["noun_error"]
    assert calls == [[768, 768, 768]]  # one shape-learning pass on BOS rows
    assert f["noun_v"].shape == (3, N_L, H) and not f["noun_v"].any()
    assert f["noun_token_index"].tolist() == [-1, -1, -1]
    f2 = ex.feature_frame(bad, kv_fn=kv_fn, obs_fn=obs_fn, noun_fn=noun_fn, noun="pear", ids=IDS,
                          bos=BOS, n_layers=N_L, state=state)
    assert len(calls) == 1 and f2["noun_hidden_final"].shape == (3, D)  # shapes reused


def test_feature_frame_refuses_wrong_layer_count():
    kv_fn, obs_fn, noun_fn, _ = fake_fns(n_layers=2)
    inputs = tokens_for("Put", "the", "pear", "in", "the", "basket", ".")
    with pytest.raises(ValueError, match="noun V shape"):
        ex.feature_frame(inputs, kv_fn=kv_fn, obs_fn=obs_fn, noun_fn=noun_fn, noun="pear", ids=IDS,
                         bos=BOS, n_layers=3)


def make_feat(key=KEY, n=4, layers=(0, 4, 17), bad_frame=None, checkpoint=None):
    kv_fn, obs_fn, noun_fn, _ = fake_fns(n_layers=len(layers))
    good = tokens_for("Pick", "up", "the", "banana", "and", "place", "it", "in", "the", "basket", ".")
    bad = tokens_for("Pick", "up", "the", "fruit", "and", "place", "it", "in", "the", "basket", ".")
    frames = [ex.feature_frame(bad if i == bad_frame else good, kv_fn=kv_fn, obs_fn=obs_fn,
                               noun_fn=noun_fn, noun="banana", ids=IDS, bos=BOS, n_layers=len(layers))
              for i in range(n)]
    stamp = ex.feat_stamp(key=key, prompt=PROMPT, noun="banana",
                          checkpoint=checkpoint or "/s/checkpoints/rby1/" + rc51.CHECKPOINT_TAIL,
                          config=rc51.CONFIG, fingerprint="abc", n_frames=n, layers=list(layers))
    return ex.assemble_feat(frames, [i * 8 for i in range(n)], stamp, extra={"decoded_prompt_frame0": "x"})


def test_assemble_feat_and_check(tmp_path):
    z = make_feat()
    assert z["kv_l4"].shape == (4, 512) and z["noun_v"].shape == (4, 3, 3, H)
    assert z["noun_hidden_final"].shape == (4, 3, D) and z["noun_layers"].tolist() == [0, 4, 17]
    assert [str(q) for q in z["query_names"]] == list(core.QUERY_NAMES)
    ex.save_npz_atomic(tmp_path / "f.npz", **z)
    back = dict(np.load(tmp_path / "f.npz", allow_pickle=False))
    assert ex.load_stamp(tmp_path / "f.npz")["noun_layers"] == [0, 4, 17]
    assert not list(tmp_path.glob("*.partial.npz"))
    t = [0, 8, 16, 24]
    assert ex.feat_problems(KEY, back, t, [0, 4, 17]) == []
    assert any("t_step" in p for p in ex.feat_problems(KEY, back, [0, 8, 16], [0, 4, 17]))
    assert any("noun_v shape" in p for p in ex.feat_problems(KEY, back, t, [0, 4]))
    assert any("record/noun" in p for p in ex.feat_problems("ep1809", back, t, [0, 4, 17]))
    z2 = make_feat(bad_frame=2)
    assert any("without noun feature" in p for p in ex.feat_problems(KEY, z2, t, [0, 4, 17]))
    z3 = make_feat(checkpoint="/x/other/1000")
    assert any("checkpoint" in p for p in ex.feat_problems(KEY, z3, t, [0, 4, 17]))
    with pytest.raises(ValueError):
        ex.assemble_feat([], [0], {"n_frames": 1})


# ------------------------------------------------------------------------------ noun record check


def test_noun_item_and_record_problems():
    item = ex.noun_item(KEY, PROMPT)
    assert item == {"key": KEY, "target": "banana", "variant": "orig", "prompt": PROMPT, "noun": "banana",
                    "swapped_from": None}
    stamp = {"record": KEY, "prompt": PROMPT, "noun": "banana", "variant": "orig"}
    ex_ok = {"stamp_dict": stamp, "t_step": np.asarray([0, 8]), "ok": np.asarray([True, True]),
             "error": np.asarray(["", ""])}
    assert ex.noun_record_problems(KEY, ex_ok, [0, 8], [0, 8], PROMPT) == []
    probs = ex.noun_record_problems(KEY, {**ex_ok, "ok": np.asarray([True, False]),
                                          "error": np.asarray(["", "target: 0 matches"])},
                                    [0, 8, 16], [0, 8], PROMPT)
    assert len(probs) == 2 and any("step record" in p for p in probs) and any("ok=False" in p for p in probs)
    probs = ex.noun_record_problems(KEY, {**ex_ok, "stamp_dict": {**stamp, "noun": "pear"}}, [0, 8], None,
                                    PROMPT)
    assert probs == ["stamp noun 'pear' != 'banana'"]


def test_array_format_problems():
    ref = {"t_step": np.zeros(3, np.int64), "class_cov": np.zeros((3, 3, 7, 16, 16), np.float32),
           "target": np.asarray("apple"), "rgb_mad": np.zeros((3, 3))}
    good = {"t_step": np.zeros(5, np.int64), "class_cov": np.zeros((5, 3, 7, 16, 16), np.float32),
            "target": np.asarray("banana"), "rgb_mad": np.zeros((2, 3)), "extra": np.zeros(1)}
    assert ex.array_format_problems(good, ref, ex.GT_PER_FRAME) == []
    bad = {**good, "class_cov": np.zeros((5, 3, 6, 16, 16), np.float64)}
    bad.pop("rgb_mad")
    probs = ex.array_format_problems(bad, ref, ex.GT_PER_FRAME)
    assert {p.split()[0] for p in probs} == {"missing", "class_cov"} and len(probs) == 3


# ------------------------------------------------------------------------------ lock + CLI


def test_lock_probe(tmp_path):
    p = tmp_path / "gpu.lock"
    assert ex.lock_is_held(str(p)) is None
    p.write_text("")
    assert ex.lock_is_held(str(p)) is False
    fd = os.open(p, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert ex.lock_is_held(str(p)) is True
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert ex.lock_is_held(str(p)) is False


@pytest.fixture
def dev_run(tmp_path, monkeypatch):
    """A run dir accepted by `common.run_output.resolve_output_path` (tmp results/development)."""
    import common.run_output as ro

    dev = tmp_path / "storage" / "results" / "development"
    run = dev / "RUN"
    run.mkdir(parents=True)
    monkeypatch.setattr(ro, "development_results_root", lambda *a, **k: dev)
    return run


@pytest.mark.parametrize("stage", ex.GPU_STAGES)
def test_gpu_stages_refuse_without_lock(stage, dev_run, tmp_path, monkeypatch):
    lock = tmp_path / "free.lock"
    lock.write_text("")

    def boom(*a, **k):
        raise AssertionError("GPU work started before the lock check")

    from AG3S.ag3s.experiments.studies import t51_extract as ex51
    from AG3S.ag3s.experiments.studies import t51b_noun_extract as ne

    monkeypatch.setattr(ex51, "gpu_guard", boom)
    monkeypatch.setattr(ne, "_gpu_session", boom)
    monkeypatch.setattr(ne, "gpu_guard", boom)
    with pytest.raises(SystemExit, match="shared lock"):
        ex.main(["--run-dir", str(dev_run), "--stage", stage, "--gpu-lock", str(lock)])


@pytest.mark.parametrize("argv,code", [
    (["--stage", "layout"], 2),                                           # no --run-dir
    (["--stage", "feat", "--noun-layers", "4", "18"], 2),
    (["--stage", "feat", "--noun-layers", "4", "4"], 2),
    (["--stage", "noun", "--config", "pi05_rby1_lora"], 2),
    (["--stage", "noun_parity", "--parity-record", "ep0"], 2),            # extra run: no stored ref
    (["--stage", "attention_parity", "--parity-frames", "0"], 2),
    (["--stage", "layout", "--server-roots", "/nonexistent/x"], 2),
    (["--stage", "nope"], 2),
])
def test_cli_argument_refusals(argv, code, dev_run):
    if "--run-dir" not in argv and argv != ["--stage", "layout"]:
        argv = argv + ["--run-dir", str(dev_run)]
    with pytest.raises(SystemExit) as e:
        ex.main(argv)
    assert e.value.code == code


def test_cli_refuses_records_and_foreign_run_dir(tmp_path, dev_run):
    with pytest.raises(SystemExit, match="unknown T53 key"):
        ex.main(["--run-dir", str(dev_run), "--stage", "layout", "--records", "ep1807"])
    with pytest.raises(SystemExit, match="--run-dir"):
        ex.main(["--run-dir", str(tmp_path / "elsewhere"), "--stage", "layout"])


def test_noun_layers_sorted():
    ap = ex.build_parser()
    args = ap.parse_args(["--stage", "feat", "--run-dir", "x", "--noun-layers", "8", "2", "4"])
    ex.validate_args(ap, args)
    assert args.noun_layers == [2, 4, 8]


# ------------------------------------------------------------------------------ layout / check stages


def write_dataset(tmp_path) -> pathlib.Path:
    ds = tmp_path / "dataset"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "randomized_episodes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _dataset_rows().values()))
    return ds


def test_stage_layout_synthetic(dev_run, tmp_path):
    t53 = (dev_run / "verify" / "T53").resolve()
    make_rollout(t53)
    ds = write_dataset(tmp_path)
    ex.main(["--run-dir", str(dev_run), "--stage", "layout", "--records", KEY, "--dataset", str(ds)])
    out = json.loads((t53 / "layout.json").read_text())
    assert out["all_ok"] and out["dataset_problems"] == []
    assert out["records"][KEY]["n_frames"] == 4 and out["records"][KEY]["policy_seed"] == 7
    assert len(out["registry"]) == 16
    with pytest.raises(SystemExit, match="layout"):  # episode not recorded -> problem, non-zero exit
        ex.main(["--run-dir", str(dev_run), "--stage", "layout", "--records", KEY, "ep1809",
                 "--dataset", str(ds)])
    out = json.loads((t53 / "layout.json").read_text())
    assert not out["all_ok"] and out["records"]["ep1809"]["problems"]


def test_stage_check_synthetic(dev_run):
    t53 = (dev_run / "verify" / "T53").resolve()
    d = make_rollout(t53)
    rec = d / "ag3s" / "run_0000"
    rc.configure(t53_dir=t53)
    write_attention(KEY, rec, n=4, prompt=PROMPT)
    ex.save_npz_atomic(rc.feat_path(KEY), **make_feat(layers=(0, 4, 17)))
    g = {"t_step": np.asarray([0, 8, 16, 24], np.int64), "target": np.asarray("banana"),
         "grasp_state": np.asarray(["pre", "pre", "held", "placed"]),
         "grasp_state_qpos": np.asarray(["pre", "pre", "held", "placed"]),
         "phys_state": np.asarray(["on_table", "grasping", "held", "in_crate"]),
         "closed_loop_gt": np.asarray(["reach", "held", "held", "released"]),
         "penetration_m": np.zeros(4)}
    ex.save_npz_atomic(rc.grasp_path(KEY), **g)
    base = ["--run-dir", str(dev_run), "--stage", "check", "--records", KEY]
    ex.main(base + ["--parts", "attention", "feat", "grasp"])
    out = json.loads((t53 / "check.json").read_text())
    assert out["all_ok"]
    assert out["records"][KEY]["grasp"]["grasp_state"]["held"] == 1
    assert out["records"][KEY]["grasp"]["first_held_frame"] == 2
    with pytest.raises(SystemExit, match="check"):  # noun / gt not produced -> reported, non-zero
        ex.main(base + ["--parts", "noun", "gt", "--t49-dir", str(dev_run), "--gt-reference",
                        str(dev_run / "none.npz")])
    out = json.loads((t53 / "check.json").read_text())
    assert not out["all_ok"]
    assert {"noun: missing", "gt: missing"} <= set(out["records"][KEY]["problems"])


# ------------------------------------------------------------------------------ real record smoke


def test_grasp_predicates_on_stored_ep1807():
    from AG3S.ag3s.experiments.studies import t47_head_selection as hs

    try:
        rec = hs.record_path("ep1807")
    except Exception:  # noqa: BLE001
        pytest.skip("storage unavailable")
    if not (rec / "meta.json").is_file():
        pytest.skip("ep1807 record not in storage")
    pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    a = ex.grasp_predicates(rec, target="apple", used_arm="left")
    n = len(a["t_step"])
    assert a["grasp_state"].shape == (n,) and a["finger_contact_lr"].shape == (n, 2)
    s = ex.grasp_summary(a)
    assert sum(s["grasp_state"].values()) == n
    gs = [str(x) for x in a["grasp_state"]]
    # the recorded run picks the apple up with the left hand and puts it in the crate
    assert s["grasp_state"]["held"] > 0 and s["grasp_state"]["placed"] > 0
    assert gs.index("held") < gs.index("placed") and gs[0] == "pre"
    held = np.asarray(gs) == "held"
    assert (a["finger_contact_lr"][held, 0] == 2).all()  # both left finger bodies touch the target
    assert all("table" not in str(t) and "crate" not in str(t) for t in a["touching"][held])
