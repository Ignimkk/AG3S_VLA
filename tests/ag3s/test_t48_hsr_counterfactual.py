"""T48a — HSR counterfactual: variants, baselines and metrics (no GPU; storage tests skip when absent).

Pinned on synthetic data:
  - C1: each distractor once, never the target, the rest of the sentence is the template; gt pick;
  - C1p: the other two templates with the same fruit; gt original;
  - C2g: only the used-arm gripper changes (closed <-> open representative), the other 15 dims stay;
  - C2s / C2i: one pair per host, partner of a different class in the same unit, image/state rows
    swapped symmetrically, reproducible with a fixed seed;
  - baselines (simple rule, strong-rule proxy, frame-index) against hand-computed sequences;
  - metrics (C1 to-pick, C1p retain, C2g changed, C2 follow, pair joint, episode median / IQR);
  - the numpy state-string replica, CLI argument validation, the AG3S_RUN_DIR refusal.
On shared storage (skipped when not mounted):
  - SUBTASK-b data parity of the val split (frames, state, labels, rule history) for one episode;
  - the C0 rule / frame-index / probe numbers on SUBTASK-b's stored features reproduce 0.9423 / 1.000 / 0.9808.
GPU (opt-in, `AG3S_GPU_TESTS=1`): C0 kv_L4 of three frames equals SUBTASK-b's stored feature.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t48_hsr_core as core
from AG3S.ag3s.experiments.studies import t48_hsr_counterfactual as cf
from AG3S.ag3s.experiments.studies import t48_hsr_io as io

TEMPLATES = sorted(["Pick up the {fruit} and place it in the basket.", "Put the {fruit} in the basket.",
                    "Place the {fruit} into the basket."])
REPS = core.GripperReps(open=1.0, closed=0.67)


def make_unit(y, *, grip_idx=7, prompt="Put the apple in the basket.", target="apple",
              distractors=("banana", "orange", "pear"), seq=None, rows=None, clock=None, cadence=1,
              uid="u0") -> core.Unit:
    y = np.asarray(y, np.int64)
    n = len(y)
    if seq is None:
        seq = np.zeros((n, 16))
        seq[:, 7] = seq[:, 15] = 1.0
        seq[:, :7] = np.arange(n)[:, None] * 0.01          # distinct arm states per frame
        seq[y == core.PLACE, grip_idx] = 0.7                # holding
    rows = np.arange(n) if rows is None else np.asarray(rows)
    clock = rows * 6 if clock is None else np.asarray(clock)
    return core.Unit(unit_id=uid, set_name="val", prompt=prompt, target=target, distractors=tuple(distractors),
                     gripper_index=grip_idx, seq_states=np.asarray(seq, np.float64), rows=rows, clock=clock,
                     y=y, proxy_cadence=cadence, seed_key=io.stable_key(uid))


Y = [0, 0, 0, 1, 1, 1, 2, 2, 2]


# ------------------------------------------------------------------------------- prompts


def test_prompt_template_and_templates_from_tasks():
    assert core.prompt_template("Put the pear in the basket.") == ("Put the {fruit} in the basket.", "pear")
    with pytest.raises(ValueError):
        core.prompt_template("Put the basket in the basket.")
    with pytest.raises(ValueError):
        core.prompt_template("Put the apple and the pear in the basket.")
    tasks = [core.fill(t, f) for t in TEMPLATES for f in core.FRUITS]
    assert core.templates_from_tasks(tasks) == TEMPLATES
    with pytest.raises(ValueError):          # a template that misses a fruit
        core.templates_from_tasks(tasks[:-1])


# ------------------------------------------------------------------------------- C1 / C1p


def test_c1_swaps_only_the_noun_to_each_distractor_and_answers_pick():
    u = make_unit(Y)
    t = core.build_variants(u, TEMPLATES, REPS)
    c1 = t.variant == "C1"
    assert c1.sum() == 3 * u.n
    for h in range(u.n):
        m = c1 & (t.host == h)
        assert sorted(t.sub[m]) == sorted(u.distractors)
        for v in np.flatnonzero(m):
            tmpl, fruit = core.prompt_template(str(t.prompt[v]))
            assert fruit == t.sub[v] != u.target
            assert tmpl == "Put the {fruit} in the basket."          # rest of the sentence unchanged
            assert t.gt[v] == core.PICK
            assert t.image_row[v] == t.state_row[v] == h
            assert np.array_equal(t.state[v], u.state(h))


def test_c1p_uses_the_other_two_templates_with_the_same_fruit():
    u = make_unit(Y)
    t = core.build_variants(u, TEMPLATES, REPS)
    m = t.variant == "C1p"
    assert m.sum() == 2 * u.n
    for v in np.flatnonzero(m):
        tmpl, fruit = core.prompt_template(str(t.prompt[v]))
        assert fruit == u.target and tmpl != "Put the {fruit} in the basket." and tmpl in TEMPLATES
        assert t.gt[v] == u.y[t.host[v]]
        assert t.sub[v] == f"T{TEMPLATES.index(tmpl)}"


def test_build_variants_rejects_bad_metadata():
    with pytest.raises(ValueError):
        core.build_variants(make_unit(Y, target="pear"), TEMPLATES, REPS)          # prompt says apple
    with pytest.raises(ValueError):
        core.build_variants(make_unit(Y, distractors=("apple", "orange", "pear")), TEMPLATES, REPS)
    with pytest.raises(ValueError):
        core.build_variants(make_unit(Y, distractors=("banana", "orange")), TEMPLATES, REPS)


# ------------------------------------------------------------------------------- C2g


@pytest.mark.parametrize("gi", [7, 15])
def test_c2g_flips_only_the_used_gripper(gi):
    u = make_unit(Y, grip_idx=gi)
    t = core.build_variants(u, TEMPLATES, REPS)
    for v in np.flatnonzero(t.variant == "C2g"):
        orig = u.state(int(t.host[v]))
        new = t.state[v]
        others = [d for d in range(16) if d != gi]
        assert np.array_equal(new[others], orig[others])                 # 15 dims untouched
        want = REPS.open if orig[gi] < 0.9 else REPS.closed
        assert new[gi] == want and new[gi] != orig[gi]
        assert t.gt[v] == u.y[t.host[v]]


def test_flip_gripper_threshold_and_validation():
    s = np.ones(16)
    s[7] = 0.9                                   # exactly the threshold counts as open
    assert core.flip_gripper(s, 7, REPS)[7] == REPS.closed
    s[7] = 0.8999
    assert core.flip_gripper(s, 7, REPS)[7] == REPS.open
    with pytest.raises(ValueError):
        core.flip_gripper(np.ones(14), 7, REPS)
    with pytest.raises(ValueError):
        core.flip_gripper(s, 6, REPS)


# ------------------------------------------------------------------------------- C2s / C2i


def test_pairs_are_other_class_same_unit_and_seed_reproducible():
    y = np.asarray(Y)
    p0 = core.sample_pairs(y, 0, 123)
    assert np.array_equal(p0, core.sample_pairs(y, 0, 123))
    assert all(0 <= b < len(y) and y[b] != y[a] for a, b in enumerate(p0))
    seeds = {tuple(core.sample_pairs(y, s, 123)) for s in range(6)}
    assert len(seeds) > 1                                              # the seed matters
    # both other classes get drawn somewhere over seeds
    drawn = {int(y[b]) for s in range(20) for a, b in enumerate(core.sample_pairs(y, s, 7)) if y[a] == 0}
    assert drawn == {1, 2}


def test_pairs_skip_unlabelled_and_single_class_units():
    assert np.array_equal(core.sample_pairs(np.asarray([0, 0, 0]), 0, 1), [-1, -1, -1])
    p = core.sample_pairs(np.asarray([0, -1, 1]), 0, 1)
    assert p[1] == -1 and p[0] == 2 and p[2] == 0


def test_c2s_c2i_swap_image_and_state_of_the_same_pair():
    u = make_unit(Y)
    t = core.build_variants(u, TEMPLATES, REPS, seed=3)
    pairs = core.sample_pairs(u.y, 3, u.seed_key)
    for a in range(u.n):
        s = np.flatnonzero((t.variant == "C2s") & (t.host == a))
        i = np.flatnonzero((t.variant == "C2i") & (t.host == a))
        assert len(s) == len(i) == 1
        s, i = int(s[0]), int(i[0])
        b = int(pairs[a])
        assert (t.image_row[s], t.state_row[s]) == (a, b)                  # image A + state B
        assert (t.image_row[i], t.state_row[i]) == (b, a)                  # image B + state A
        assert np.array_equal(t.state[s], u.state(b)) and np.array_equal(t.state[i], u.state(a))
        assert t.gt[s] == u.y[a] and t.gt[i] == u.y[b]                     # answer = image side
        assert t.y_image[s] != t.y_state[s] and t.y_image[i] != t.y_state[i]
        assert t.prompt[s] == t.prompt[i] == u.prompt
    # reproducible table (digest) with the same seed; different with another
    assert t.digest() == core.build_variants(u, TEMPLATES, REPS, seed=3).digest()
    assert t.digest() != core.build_variants(u, TEMPLATES, REPS, seed=4).digest()


def test_variant_counts_and_roundtrip():
    u = make_unit(Y)
    t = core.build_variants(u, TEMPLATES, REPS)
    assert len(t) == u.n * (1 + 3 + 2 + 1 + 1 + 1)
    t2 = core.VariantTable.from_arrays(t.as_arrays("plan__"), "plan__")
    assert t2.digest() == t.digest()


# ------------------------------------------------------------------------------- state string


def test_discretize_state_replica():
    q01, q99 = np.zeros(16), np.ones(16)
    s = np.full(16, 0.3)
    s[0], s[1], s[2] = 0.999999, 0.0001, 2.0
    b = core.discretize_state(s, q01, q99)
    norm = (s.astype(np.float32) - 0) / (1 + 1e-6) * 2 - 1
    want = np.clip(np.floor((norm + 1) * 128).astype(int), 0, 255)
    assert np.array_equal(b, want)
    assert b[3] == 76 and b[0] == 255 and b[2] == 255


def test_parse_state_string():
    t = "Task: Put the pear in the basket., State: 1 2 255 0;\nAction: "
    assert cf.parse_state_string(t).tolist() == [1, 2, 255, 0]
    assert cf.parse_state_string("no state here") is None


# ------------------------------------------------------------------------------- baselines


def test_simple_rule_hand_computed():
    s_open, s_closed = np.ones(16), np.ones(16)
    s_closed[15] = 0.6
    assert core.simple_rule(s_open, False) == core.PICK
    assert core.simple_rule(s_closed, False) == core.PLACE
    assert core.simple_rule(s_closed, True) == core.PLACE
    assert core.simple_rule(s_open, True) == core.HOME


def test_proxy_latch_hand_computed():
    run = core.proxy_run
    assert run([1.0, 1.0]) == core.PICK
    assert run([1.0, 0.70]) == core.PICK                     # CLOSING, not settled yet
    assert run([1.0, 0.70, 0.705]) == core.PLACE             # settled (0.005 < 0.02), dropped 0.295
    assert run([1.0, 0.70, 0.80]) == core.PICK               # still moving (0.10 >= 0.02)
    assert run([1.0, 0.70, 0.705, 1.0]) == core.PLACE        # one open request: not yet released
    assert run([1.0, 0.70, 0.705, 1.0, 1.0]) == core.HOME    # two -> PLACED
    assert run([1.0, 0.70, 0.705, 1.0, 0.70]) == core.PLACE  # open streak broken
    assert run([1.0, 0.70, 0.705, 1.0, 1.0, 0.7, 0.7]) == core.HOME   # PLACED is sticky
    assert run([1.0, 0.05, 0.05]) == core.PICK               # closed to the end = empty hand
    assert run([1.0, 0.70, 1.0, 0.70, 0.70]) == core.PLACE   # abandoned attempt, then a real grasp
    # no history: g_pre defaults to 1.0 -> first 0.84 = CLOSING, second settles with drop 0.16 -> HELD
    assert run([0.84]) == core.PICK
    assert run([0.84, 0.84]) == core.PLACE
    assert run([0.86, 0.86]) == core.PICK                    # above closed_below: never closing


def test_proxy_history_cadence():
    g = np.arange(20, dtype=float)
    assert core.proxy_history(g, 17, 8).tolist() == [1.0, 9.0]
    assert core.proxy_history(g, 5, 8).tolist() == []
    assert core.proxy_history(g, 3, 1).tolist() == [0.0, 1.0, 2.0]
    with pytest.raises(ValueError):
        core.proxy_history(g, 3, 0)


def test_frame_index_majority_and_fallbacks():
    maj = core.frame_index_majority(np.asarray([0, 0, 0, 6, 6, 12, 12]), np.asarray([0, 0, 1, 1, 2, 2, 1]))
    assert maj == {0: 0, 6: 1, 12: 1}                       # tie at 6 -> lowest class; 12 tie -> 1
    assert core.frame_index_predict(6, maj) == (1, False)
    assert core.frame_index_predict(8, maj, fallback="zero") == (0, True)
    assert core.frame_index_predict(8, maj, fallback="nearest") == (1, True)   # nearest 6
    assert core.frame_index_predict(9, maj, fallback="nearest") == (1, True)   # tie 6/12 -> 6
    assert core.frame_index_predict(100, maj, fallback="nearest") == (1, True)
    with pytest.raises(ValueError):
        core.frame_index_predict(8, maj, fallback="bogus")


def test_baselines_on_variants_follow_the_current_state_with_the_host_history():
    # dataset-like unit: gripper closes at seq 4-6, open afterwards; sampled every frame
    n = 10
    seq = np.ones((n, 16))
    seq[4:7, 7] = [0.70, 0.70, 0.70]
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    u = make_unit(y, seq=seq, cadence=1)
    t = core.build_variants(u, TEMPLATES, REPS, seed=0)
    maj = {int(c): int(k) for c, k in zip(u.clock, y)}
    b = core.baseline_predictions(u, t, maj, fallback="zero")
    c0 = t.variant == "C0"
    assert b["rule"][c0].tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert b["proxy"][c0].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]   # settle + 2 open requests
    assert b["frame_index"][c0].tolist() == y.tolist()
    # C1 / C1p: rules do not read the prompt -> same as C0
    for fam in ("C1", "C1p"):
        m = t.variant == fam
        for k in ("rule", "proxy", "frame_index"):
            assert all(b[k][v] == b[k][c0][t.host[v]] for v in np.flatnonzero(m))
    # C2g at host 2 (open, never closed before) -> rule place; proxy: one closed request -> pick
    v = int(np.flatnonzero((t.variant == "C2g") & (t.host == 2))[0])
    assert b["rule"][v] == core.PLACE and b["proxy"][v] == core.PICK
    # C2g at host 6 (closed, held) -> rule: open now + closed before -> home; proxy: one open -> place
    v = int(np.flatnonzero((t.variant == "C2g") & (t.host == 6))[0])
    assert b["rule"][v] == core.HOME and b["proxy"][v] == core.PLACE
    # C2g at host 5: history ends CLOSING (one closed request) -> an open request abandons it -> pick
    v = int(np.flatnonzero((t.variant == "C2g") & (t.host == 5))[0])
    assert b["rule"][v] == core.HOME and b["proxy"][v] == core.PICK
    # frame-index never changes with the variant (host clock)
    for v in range(len(t)):
        assert b["frame_index"][v] == y[t.host[v]]


# ------------------------------------------------------------------------------- metrics


def test_episode_summary_hand_computed():
    s = core.episode_summary({"a": (1, 2), "b": (3, 4), "c": (0, 0), "d": (0, 4)})
    assert s["k"] == 4 and s["n"] == 10 and s["pooled"] == 0.4 and s["n_episodes"] == 3
    assert s["episode_median"] == 0.5
    assert s["episode_q25"] == pytest.approx(0.25) and s["episode_q75"] == pytest.approx(0.625)


def _rows(spec):
    """spec rows: (unit, variant, sub, host, gt, y_host, y_image, y_state, pred)."""
    cols = list(zip(*spec))
    unit, variant, sub, host = (np.asarray(c) for c in cols[:4])
    gt, yh, yi, ys, pred = (np.asarray(c, np.int64) for c in cols[4:])
    hk = np.asarray([f"{u}:{h}" for u, h in zip(unit, host)])
    pk = np.asarray([k if v in ("C2s", "C2i") else "" for k, v in zip(hk, variant)])
    return core.score_method(unit, variant, sub, hk, gt, yh, yi, ys, pred, pair_key=pk)


def test_score_method_hand_computed():
    P, L, H = core.PICK, core.PLACE, core.HOME
    spec = [
        # unit A: host 0 (pick), host 1 (place), host 2 (home)
        ("A", "C0", "", 0, P, P, P, P, P),
        ("A", "C0", "", 1, L, L, L, L, L),
        ("A", "C0", "", 2, H, H, H, H, L),          # C0 error
        ("A", "C1", "x", 0, P, P, P, P, P),
        ("A", "C1", "x", 1, P, L, L, L, P),         # place -> pick
        ("A", "C1", "x", 2, P, H, H, H, H),         # home stays
        ("A", "C1p", "T0", 1, L, L, L, L, L),       # retained (== C0 pred L)
        ("A", "C1p", "T0", 2, H, H, H, H, H),       # C0 pred was L -> not retained (but correct)
        ("A", "C2g", "", 0, P, P, P, P, L),         # changed
        ("A", "C2g", "", 1, L, L, L, L, L),         # unchanged
        ("A", "C2s", "", 0, P, P, P, L, L),         # follows state
        ("A", "C2i", "", 0, L, P, L, P, L),         # follows image (B=place image, A state pick)
        ("A", "C2s", "", 1, L, L, L, H, P),         # other
        ("A", "C2i", "", 1, H, L, H, L, L),         # follows state
        # unit B: one pick host
        ("B", "C0", "", 0, P, P, P, P, P),
        ("B", "C1", "y", 0, P, P, P, P, P),
        ("B", "C2g", "", 0, P, P, P, P, P),
        ("B", "C2s", "", 0, P, P, P, H, -1),        # missing feature
    ]
    r = _rows(spec)
    assert r["n_missing"] == 1
    acc0 = r["accuracy"]["C0"]
    assert (acc0["k"], acc0["n"]) == (3, 4) and acc0["episode_median"] == pytest.approx((2 / 3 + 1) / 2)
    assert r["C0"]["confusion"] == [[2, 0, 0], [0, 1, 0], [0, 1, 0]]
    c1 = r["C1"]
    assert c1["to_pick"]["pick"]["pooled"] == 1.0 and c1["to_pick"]["place"]["pooled"] == 1.0
    assert c1["to_pick"]["home"]["pooled"] == 0.0 and c1["to_pick"]["home"]["n_episodes"] == 1
    assert c1["to_pick_all"]["k"] == 3 and c1["to_pick_all"]["n"] == 4
    assert c1["matrix"] == [[2, 0, 0], [1, 0, 0], [0, 0, 1]]
    assert c1["to_pick_by_distractor"] == {"x": pytest.approx(2 / 3), "y": 1.0}
    assert r["C1p"]["retain_all"]["k"] == 1 and r["C1p"]["retain_all"]["n"] == 2
    assert r["C1p"]["retain"]["home"]["pooled"] == 0.0
    assert r["accuracy"]["C1p"]["pooled"] == 1.0
    assert r["C2g"]["changed_all"]["k"] == 1 and r["C2g"]["changed_all"]["n"] == 3
    assert r["C2g"]["changed"]["pick"]["pooled"] == 0.5 and r["C2g"]["changed"]["place"]["pooled"] == 0.0
    f = r["C2s"]["follow"]
    assert (f["state"]["k"], f["image"]["k"], f["other"]["k"], f["state"]["n"]) == (1, 0, 1, 2)
    f = r["C2i"]["follow"]
    assert (f["state"]["k"], f["image"]["k"], f["other"]["k"]) == (1, 1, 0)
    assert r["C2s"]["by_image_state_class"]["img_pick__state_place"] == {"n": 1, "state": 1.0, "image": 0.0,
                                                                        "other": 0.0}
    j = r["C2_pairs"]
    assert j["n_pairs"] == 2
    assert j["joint"]["C2s_state__C2i_image"] == 1 and j["joint"]["C2s_other__C2i_state"] == 1
    assert sum(j["joint"].values()) == 2


# ------------------------------------------------------------------------------- CLI


def test_cli_argument_validation():
    a = cf.parse_args(["--stage", "plan"])
    assert a.sets == ["val", "ep1807"] and a.batch == 1 and a.seed == 0
    for bad in (["--stage", "extract", "--batch", "0"], ["--stage", "score", "--check-tokens"],
                ["--stage", "plan", "--max-frames", "0"], ["--stage", "bogus"],
                ["--stage", "plan", "--sets", "train"]):
        with pytest.raises(SystemExit):
            cf.parse_args(bad)
    assert cf.feat_tag(cf.parse_args(["--stage", "score", "--max-frames", "4"])) == "max4"


def test_cli_refuses_without_run_dir(monkeypatch):
    monkeypatch.delenv("AG3S_RUN_DIR", raising=False)
    with pytest.raises(SystemExit, match="AG3S_RUN_DIR"):
        cf.out_root()
    with pytest.raises(SystemExit, match="AG3S_RUN_DIR"):
        cf.main(["--stage", "plan"])


def test_closed_loop_gt_hand_computed():
    q = np.zeros(66)
    q[31:34] = [0.5, 0.0, 0.9]
    s = np.ones(16)
    q[38:41] = [0.5, 0.1, 0.95]                   # apple 0.1 m from the crate, low
    assert core.closed_loop_gt(q, s, "apple") == "released"
    s[7] = 0.7
    assert core.closed_loop_gt(q, s, "apple") == "held"
    s[7] = 1.0
    q[38:41] = [0.5, 0.3, 0.85]
    assert core.closed_loop_gt(q, s, "apple") == "reach"


# ------------------------------------------------------------------------------- shared storage

PATHS = io.Paths.from_storage()
have_data = pytest.mark.skipif(not PATHS.dataset.exists(), reason="dataset not on this machine")
have_feat = pytest.mark.skipif(not (PATHS.subtask_b_feat / "val").exists() or not PATHS.dataset.exists(),
                               reason="SUBTASK-b feat/val not on this machine")


@have_data
def test_val_split_matches_subtask_b_definition():
    meta, tasks = io.episode_meta(PATHS.dataset), io.task_prompts(PATHS.dataset)
    assert core.templates_from_tasks(tasks.values()) == TEMPLATES
    u = io.dataset_unit(PATHS.dataset, 1600, tasks, meta)
    assert u.n == 52 and (u.clock % 6 == 0).all() and u.gripper_index == 15
    assert u.target not in u.distractors and len(u.distractors) == 3
    assert max(u.extra["distractor_displacement_m"].values()) < 0.02


@have_feat
def test_subtask_b_data_parity_for_one_episode():
    meta, tasks = io.episode_meta(PATHS.dataset), io.task_prompts(PATHS.dataset)
    u = io.dataset_unit(PATHS.dataset, 1640, tasks, meta)
    with np.load(PATHS.subtask_b_feat / "val" / "ep_001640.npz", allow_pickle=False) as z:
        assert np.array_equal(u.clock, z["frame_index"])
        assert np.array_equal(u.seq_states[u.rows], z["state"])
        assert np.array_equal(u.y, core.phase_to_class(z["phase_index"]))
        ever = np.cumsum(core.gripper_min(u.seq_states) < 0.9) > 0
        assert np.array_equal(ever[u.rows], z["ever_closed"])
        assert str(z["prompt"]) == u.prompt


@have_feat
def test_c0_numbers_reproduce_subtask_b_on_stored_features():
    """rule 0.9423 (`probe_results.json` rule.val), frame-index 1.000 (`controls_results.json` C1),
    probe 0.98078 (`kv_L4_v1.json` parity) — through this study's own C0 path."""
    from AG3S.ag3s.stages.subtask_probe import SubtaskProbe

    meta, tasks = io.episode_meta(PATHS.dataset), io.task_prompts(PATHS.dataset)
    stats = io.train_stats(PATHS.dataset, meta)
    assert stats["frames"] == 10401 and stats["per_class"] == {"pick": 4000, "place": 3200, "home": 3201}
    maj = io.majority_from_stats(stats)
    probe = SubtaskProbe.load()
    hit = {"rule": 0, "frame_index": 0, "probe": 0}
    n = 0
    for ep in io.VAL_EPISODES:
        u = io.dataset_unit(PATHS.dataset, ep, tasks, meta)
        t = core.build_variants(u, TEMPLATES, io.reps_from_stats(stats), families=("C0",))
        b = core.baseline_predictions(u, t, maj, fallback="zero")
        with np.load(PATHS.subtask_b_feat / "val" / f"ep_{ep:06d}.npz", allow_pickle=False) as z:
            assert np.array_equal(z["frame_index"], u.clock)
            p = probe.proba(z["normal__kv_L4"]).argmax(1)
        hit["rule"] += int((b["rule"] == t.gt).sum())
        hit["frame_index"] += int((b["frame_index"] == t.gt).sum())
        hit["probe"] += int((p == t.gt).sum())
        n += len(t)
    assert n == 2601
    assert hit["rule"] / n == pytest.approx(0.9423298731257209, abs=1e-12)
    assert hit["frame_index"] == n
    assert hit["probe"] / n == pytest.approx(0.9807766243752403, abs=1e-12)


@have_feat
def test_score_stage_end_to_end_on_stored_features(tmp_path):
    """`--stage score` on a fake cache: every variant row gets SUBTASK-b's stored feature of its
    image-source frame (so C0 is the real feature). Checks the cache contract, the C0 gate on a
    partial run and the parity numbers (0 error), and that a stale digest is refused."""
    args = cf.parse_args(["--stage", "score", "--sets", "val", "--max-units", "2"])
    plan = cf.Plan(args, PATHS, tmp_path)
    fdir = tmp_path / "feat" / "full" / "val"
    fdir.mkdir(parents=True)
    for u in plan.units("val"):
        t = plan.table(u)
        with np.load(PATHS.subtask_b_feat / "val" / f"{u.unit_id}.npz") as z:
            f = np.asarray(z["normal__kv_L4"])[t.image_row]
        np.savez(fdir / f"{u.unit_id}.npz", feature=f, plan_digest=np.asarray(t.digest()),
                 n_text_tokens=np.full(len(t), 76), state_bins_replica_mismatch=np.asarray(0))
    m = cf.stage_score(args, PATHS, tmp_path)
    c0 = json.loads((tmp_path / "score" / "c0_check.json").read_text())
    assert c0["sets"]["val"]["n"] == 104 and c0["reproduced"] is True
    par = json.loads((tmp_path / "score" / "parity.json").read_text())["val"]["summary"]
    assert par["n_frames"] == 104 and par["rel_l2_max"] == 0.0 and par["argmax_agree_ratio"] == 1.0
    probe = m["sets"]["val"]["methods"]["probe"]
    assert probe["n_missing"] == 0 and "C2_pairs" in probe
    # fake features carry no prompt / state change -> C1p retain and C2g change are exact
    assert probe["C1p"]["retain_all"]["pooled"] == 1.0 and probe["C2g"]["changed_all"]["pooled"] == 0.0
    assert probe["C2i"]["follow"]["image"]["pooled"] > 0.9
    assert (tmp_path / "score" / "predictions_val.csv").exists()
    # a cache from another plan (seed) is refused
    args2 = cf.parse_args(["--stage", "score", "--sets", "val", "--max-units", "2", "--seed", "1"])
    with pytest.raises(SystemExit, match="digest"):
        cf.stage_score(args2, PATHS, tmp_path)


# ------------------------------------------------------------------------------- GPU (opt-in)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("AG3S_GPU_TESTS") != "1", reason="GPU + checkpoint; opt in with AG3S_GPU_TESTS=1")
def test_gpu_c0_feature_matches_subtask_b(capsys):
    from AG3S.ag3s.experiments.studies.t48_hsr_features import FeatureExtractor

    meta, tasks = io.episode_meta(PATHS.dataset), io.task_prompts(PATHS.dataset)
    u = io.dataset_unit(PATHS.dataset, 1600, tasks, meta, max_frames=3)
    vids = {c: io.decode_video(io.video_path(PATHS.dataset, 1600, c)) for c in io.CAMS}
    ex = FeatureExtractor(io.CONFIG_NAME, str(PATHS.checkpoint), batch=1)
    xs = [ex.inputs(u.state(a), {c: vids[c][int(u.clock[a])] for c in io.CAMS}, u.prompt) for a in range(3)]
    got = np.concatenate([ex.features([x]) for x in xs]).astype(np.float64)
    with np.load(PATHS.subtask_b_feat / "val" / "ep_001600.npz", allow_pickle=False) as z:
        want = np.asarray(z["normal__kv_L4"][:3], np.float64)
    rel = np.linalg.norm(got - want, axis=1) / np.linalg.norm(want, axis=1)
    with capsys.disabled():
        print(f"\n[T48 C0 parity] rel_l2={rel.tolist()} max_abs={np.abs(got - want).max():.3e} mem={ex.memory()}")
    assert rel.max() < 1e-2
