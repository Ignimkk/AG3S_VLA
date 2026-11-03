"""T77 -- 5-stage HSR: hidden-state summary extraction, closed-loop GT labels, probe (`handoff/T77.task.md` §5.4).

1. default CPU path of `pi05_hidden` unchanged (the T73 tests stay as they are; here: legacy CLI routing / defaults);
2. summary pooling: shapes, last valid text token position, masked means, kv_L4 = SUBTASK-b formula, numpy == jax;
3. label rules on small fake sequences (empty-hand close -> grasp, held -> carry, over the basket -> place,
   after release -> retreat), the basket box on the real scene XML;
4. scene split: test scenes never in train;
5. probe metrics / selection / an end-to-end cache + fit on synthetic files.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.sources import pi05_hidden as ph
from AG3S.ag3s.experiments.studies import t77_labels as tl
from AG3S.ag3s.experiments.studies import t77_probe as tp

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_imports_from_this_checkout():
    for mod in (ph, tl, tp):
        assert pathlib.Path(mod.__file__).resolve().is_relative_to(REPO), mod.__file__


# ------------------------------------------------------------------------------ 1. default path


def test_parser_defaults_unchanged():
    a = ph.build_parser().parse_args(["--records", "/x/run_0000"])
    assert a.device == "cpu" and a.pool == "none" and a.dtype == "float16" and a.tokens == "all"
    assert a.layers == ["all"] and a.threads == 16 and a.dataset is None and a.spec is None


class _Stop(Exception):
    pass


def test_legacy_cli_takes_the_cpu_path(monkeypatch):
    calls = []
    monkeypatch.setattr(ph, "force_cpu", lambda t: (calls.append(("cpu", t)), (_ for _ in ()).throw(_Stop()))[1])
    monkeypatch.setattr(ph, "setup_gpu", lambda: (_ for _ in ()).throw(AssertionError("gpu path")))
    monkeypatch.setattr(ph, "main_summary", lambda a: (_ for _ in ()).throw(AssertionError("summary path")))
    with pytest.raises(_Stop):
        ph.main(["--records", "/x/run_0000", "--threads", "8"])
    assert calls == [("cpu", 8)]
    with pytest.raises(SystemExit):  # several records only in summary mode
        ph.main(["--records", "/a", "/b"])
    with pytest.raises(SystemExit):  # dataset needs summary
        ph.main(["--dataset", "/d", "--episodes", "0"])


def test_summary_and_gpu_routing(monkeypatch):
    seen = []
    monkeypatch.setattr(ph, "main_summary", lambda a: seen.append(a.pool))
    ph.main(["--pool", "summary", "--dataset", "/d", "--episodes", "0:16:8"])
    assert seen == ["summary"]
    monkeypatch.setattr(ph, "setup_gpu", lambda: (_ for _ in ()).throw(_Stop()))
    monkeypatch.setattr(ph, "force_cpu", lambda t: (_ for _ in ()).throw(AssertionError("forced cpu")))
    with pytest.raises(_Stop):
        ph.main(["--records", "/x", "--device", "gpu"])


def test_parse_episodes_and_names():
    assert ph.parse_episodes(["0:1600:8"])[:3] == [0, 8, 16] and len(ph.parse_episodes(["0:1600:8"])) == 200
    assert len(ph.parse_episodes(["1600:1800:4"])) == 50
    assert ph.parse_episodes(["5", "3", "5", "0:4:2"]) == [5, 3, 0, 2]
    assert ph.safe_name("CE/FIN_k0/B_ROI_FPR_ep211_s2113") == "CE_FIN_k0_B_ROI_FPR_ep211_s2113"


# ------------------------------------------------------------------------------ 2. pooling


def _fake(T=968, D=16, L=3, n_text=37, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(L, T, D)).astype(np.float32)
    mask = np.zeros(T, bool)
    mask[:768] = True
    mask[768:768 + n_text] = True
    return h, mask


def test_summary_rows_shapes_last_token_and_means():
    h, mask = _fake()
    rows, last = ph.summary_rows(np, h, mask)
    assert rows.shape == (3, 5, 16)
    assert int(last) == 768 + 37 - 1                       # last valid text token
    np.testing.assert_allclose(rows[:, 4], h[:, 768 + 36])
    np.testing.assert_allclose(rows[:, 0], h[:, 0:256].mean(1), rtol=1e-5)
    np.testing.assert_allclose(rows[:, 2], h[:, 512:768].mean(1), rtol=1e-5)
    np.testing.assert_allclose(rows[:, 3], h[:, 768:805].mean(1), rtol=1e-5, atol=1e-6)  # padding excluded
    assert ph.SUMMARY_ROWS == ("cam_high", "cam_left_wrist", "cam_right_wrist", "text_mean", "text_last")
    # matches T73 --pool mean on the camera and text rows
    pooled, _ = ph.select_and_pool(h, mask, [0, 1, 2], "all", "mean")
    np.testing.assert_allclose(rows[:, :4], pooled, rtol=1e-5, atol=1e-6)


def test_kv_l4_is_subtask_b_formula():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(968, 256)).astype(np.float32)
    _, mask = _fake(n_text=50)
    got = ph.kv_l4_feature(np, v, mask)
    tm = mask[768:].astype(np.float32)                     # extract.py:_prefix verbatim
    ref = np.concatenate([(v[768:] * tm[:, None]).sum(0) / tm.sum(), v[:768].mean(0)])
    assert got.shape == (512,) and got.dtype == np.float32
    np.testing.assert_array_equal(got, ref)


def test_summary_rows_jax_matches_numpy():
    jnp = pytest.importorskip("jax.numpy")
    import jax

    h, mask = _fake(n_text=1)
    rows_n, last_n = ph.summary_rows(np, h, mask)
    rows_j, last_j = jax.jit(lambda a, m: ph.summary_rows(jnp, a, m))(h, mask)
    assert int(last_j) == int(last_n) == 768
    np.testing.assert_allclose(np.asarray(rows_j), rows_n, rtol=1e-5, atol=1e-6)


def test_summary_extractor_abstract_shapes():
    pytest.importorskip("flax")
    pytest.importorskip("openpi")
    import flax.nnx as nnx
    import jax

    from openpi.models import pi0_config

    cfg = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                               return_hidden_states=True)
    model = nnx.eval_shape(cfg.create, jax.random.key(0))
    ex = ph.SummaryExtractor(model, kv_layer=2)  # the dummy has 4 blocks
    rows, kv4, absmax, last, m = jax.eval_shape(ex._summary, ex._state, cfg.fake_obs())  # noqa: SLF001
    assert rows.shape == (5, 5, 64) and str(rows.dtype) == "float32"   # dummy: depth 4 -> 5 hidden layers, width 64
    assert kv4.shape == (2 * 16,)  # dummy head_dim 16 -> [text | image]
    assert absmax.shape == (5,) and last.shape == () and m.shape == (3 * 256 + cfg.max_token_len,)


# ------------------------------------------------------------------------------ 3. labels


def test_gt_held_thresholds():
    assert tl.gt_held(2, 20.0) and not tl.gt_held(1, 50.0) and not tl.gt_held(2, 19.9)


def test_phase5_sequence_rules():
    # far, near (open), near + closed on nothing (empty-hand close: not held), held, held over basket, released,
    # dropped then re-grasped, unmeasured request
    held = [False, False, False, True, True, False, True, None]
    palm = [0.20, 0.05, 0.03, 0.03, 0.03, 0.15, 0.03, None]
    bask = [False, False, False, False, True, True, False, None]
    assert tl.phase5_sequence(held, palm, bask) == [
        "approach", "grasp", "grasp", "carry", "place", "retreat", "carry", None]
    # 60 mm is inside the window, never-held stays pre-grasp however long
    assert tl.phase5_sequence([False, False], [0.060, 0.0601], [False, False]) == ["grasp", "approach"]
    assert tl.serving_to5("lift") == "carry" and tl.serving_to5("placement") == "place" and tl.serving_to5(None) is None
    assert tl.COARSE_OF_5["grasp"] == "pick" and tl.COARSE_OF_5["retreat"] == "home"


def test_map5_expert_phases():
    assert [tl.MAP5[i] for i in (0, 2, 3, 4, 5, 6, 7, 9, 10, 13, 20, 22)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0]
    assert 14 not in tl.MAP5 and 19 not in tl.MAP5


def _scene_probe():
    pytest.importorskip("mujoco")
    try:
        import rby1_manipulation.simulation.pick_place_obstacles  # noqa: F401
    except Exception:
        pytest.skip("rby1_manipulation not importable (PYTHONPATH)")
    xml = REPO / tl.XML_REL
    if not xml.exists():
        pytest.skip("scene XML missing")
    return tl.SceneProbe(str(xml), (None, "clear"))


def test_basket_box_and_measure_on_scene():
    import mujoco

    pr = _scene_probe()
    lo, hi = pr.basket_box
    np.testing.assert_allclose(lo, [-0.09, -0.15], atol=1e-6)   # crate_floor
    np.testing.assert_allclose(hi, [0.09, 0.15], atol=1e-6)
    m, d = pr.m, pr.d
    q = m.qpos0.copy()
    mujoco.mj_forward(m, d)
    adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "apple_free")]
    crate_adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "crate_free")]
    q[adr:adr + 3] = q[crate_adr:crate_adr + 3] + [0.05, -0.10, 0.10]   # over the floor, crate frame = world (qpos0)
    r = pr.measure(q, "left")
    assert r["in_basket"] and r["fingers_touch"] == 0
    q[adr:adr + 3] = q[crate_adr:crate_adr + 3] + [0.0, 0.20, 0.10]     # over the handle: outside the floor box
    assert not pr.measure(q, "left")["in_basket"]
    assert "error" in pr.measure(q[:-1], "left")


# ------------------------------------------------------------------------------ 4. split


def test_scene_split_keeps_test_scenes_out_of_train():
    cases = [("B_ROI_FPR_ep211_s2113", 211, 2113, "ep211_place"), ("HURDLE_ep954_s9543", 954, 9543, "ep954_placehurdle"),
             ("AG3S_case1_L2_s23453", 1345, 23453, "ep1345_noorange_case1_L2"),
             ("AG3S_halfpost_s23453", 1345, 23453, "ep1345_noorange_case1_plus_halfpost"),
             ("B_ROI_FPR_ep119_s1193", 119, 1193, "clear"), ("AG3S_case1_L2_s13453", 1345, 13453, "x_case1_L2"),
             ("HURDLE_ep1345_s13453", 1345, 13453, "ep1345_placehurdle"), ("B_ROI_FPR_ep850_s8503", 850, 8503, "p")]
    got = [tl.scene_of(*c) for c in cases]
    assert got == [("회귀", "211"), ("허들", "954"), ("가림", "122"), ("가림-halfpost", "122"), ("회귀", "119"),
                   ("가림", "121"), ("허들", "1345"), ("회귀", "850")]
    entries = [{"scene_number": n, "split": tl.split_of(n), "episode": c[1], "seed": c[2]}
               for c, (_, n) in zip(cases, got)]
    assert [e["split"] for e in entries] == ["test", "test", "test", "test", "test", "train", "train", "train"]
    sides = tl.check_split(entries)
    assert set(sides["test"]) == {"211", "954", "122", "119"} and not set(sides["train"]) & set(tl.TEST_SCENES)
    with pytest.raises(AssertionError):
        tl.check_split(entries + [{"scene_number": "211", "split": "train", "episode": 211, "seed": 2113}])
    with pytest.raises(AssertionError):  # same episode + seed on both sides
        tl.check_split(entries + [{"scene_number": "999", "split": "train", "episode": 1345, "seed": 23453}])


# ------------------------------------------------------------------------------ 5. probe


def test_false_carry_and_verdict():
    y = np.array([0, 0, 1, 1, 2, 3, 4])
    p = np.array([0, 2, 1, 4, 2, 3, 4])
    fc = tp.false_carry_rate(y, p)
    assert fc == {"rate": 0.5, "n_not_held": 4, "n_false_carry": 2}
    mk = lambda fc_, acc, f1: {"test": {"false_carry": {"rate": fc_}, "accuracy": acc}, "val": {"macro_f1": f1}}  # noqa: E731
    assert tp.verdict(mk(0.10, 0.80, 0.90), mk(0.20, 0.81, 0.91))["hidden_better"]
    assert not tp.verdict(mk(0.11, 0.80, 0.90), mk(0.20, 0.81, 0.91))["hidden_better"]   # not half
    assert not tp.verdict(mk(0.05, 0.78, 0.90), mk(0.20, 0.81, 0.91))["hidden_better"]   # accuracy 3 pp lower
    assert not tp.verdict(mk(0.05, 0.81, 0.88), mk(0.20, 0.81, 0.91))["hidden_better"]   # val F1 0.03 lower


def test_select_feature_uses_cv_only_with_tiebreak():
    cv = {"kv_L4": 0.99, "h4_img3+last": 0.80, "h4_last": 0.80, "h2_last": 0.80, "h8_img3+last": 0.79}
    assert tp.select_feature(cv) == "h2_last"
    assert tp.feature_names([0, 4], ["last"]) == ["kv_L4", "h0_last", "h4_last"]


def test_closed_metrics_boundary_delay():
    y = np.array([0, 1, 1, 2, 2, 2, 3, 3])
    p = np.array([0, 1, 1, 1, 2, 2, 2, 3])
    m = tp.closed_metrics(y, p, ["r"] * 8, np.arange(8) * 8)
    assert m["delay_grasp_to_carry"]["median_frames"] == 1.0 and m["delay_carry_to_place"]["median_frames"] == 1.0
    assert m["false_carry"]["rate"] == 0.0 and m["accuracy"] == 6 / 8


def _write_expert(path, ep, n, rng, layers):
    ph_idx = np.array([0, 3, 5, 7, 10, 14] * (n // 6 + 1))[:n]
    y5 = np.array([tl.MAP5.get(i, 0) for i in ph_idx])
    H = rng.normal(size=(n, len(layers), 5, 2048)).astype(np.float32)
    H[..., 0] += y5[:, None, None] * 2.0
    kv = rng.normal(size=(n, 512)).astype(np.float32)
    kv[:, 0] += y5 * 2.0
    np.savez(path / f"ep_{ep:06d}.npz", hidden_summary=H, kv_L4=kv, layers=np.array(layers), phase_index=ph_idx,
             frame_index=np.arange(n) * 6, episode=np.int64(ep), t_step=np.arange(n) * 6)


def test_probe_end_to_end_synthetic(tmp_path):
    rng = np.random.default_rng(0)
    layers = [0, 4]
    E = tmp_path / "expert"
    for split, eps in (("train", range(0, 10)), ("val", range(100, 103))):
        (E / split).mkdir(parents=True)
        for ep in eps:
            _write_expert(E / split, ep, 12, rng, layers)
    C = tmp_path / "closed"
    C.mkdir()
    runs = []
    for i, split in enumerate(["train"] * 5 + ["test"] * 2):
        lab = f"K/run{i}"
        seq = ["approach", "approach", "grasp", "grasp", "carry", "carry", "place", "retreat"]
        y5 = np.array([tl.PHASES5.index(s) for s in seq])
        ts = np.arange(8) * 8
        H = rng.normal(size=(8, 2, 5, 2048)).astype(np.float32)
        H[..., 0] += y5[:, None, None] * 2.0
        kv = rng.normal(size=(8, 512)).astype(np.float32)
        kv[:, 0] += y5 * 2.0
        np.savez(C / f"{tp.safe_name(lab)}.npz", hidden_summary=H, kv_L4=kv, layers=np.array(layers), t_step=ts)
        rows = [{"request": j, "t_step": int(t), "gt": {"phase5": s}, "serving": {"hsr6_to5": "carry", "subtask_label": "place"}}
                for j, (t, s) in enumerate(zip(ts, seq))]
        rows.append({"request": 8, "t_step": 64, "missing_step": True})
        runs.append({"label": lab, "split": split, "rows": rows})
    L = tmp_path / "labels.json"
    L.write_text(json.dumps({"runs": runs}))
    out = tmp_path / "out"
    tp.main([str(out), "--expert-dir", str(E), "--closed-dir", str(C), "--labels", str(L), "--layers", "0", "4",
             "--c-grid", "0.1"])
    res = json.loads((out / "t77_probe.json").read_text())
    info = res["cache"]
    assert info["rows_by_source_split"] == {"expert/train": 100, "expert/val": 30, "closed/train": 40, "closed/test": 16}
    assert info["excluded"]["expert_train_excluded_phase"] == 20                # phase 14 dropped (2 per episode)
    assert set(res["features"]) == {"kv_L4", "h0_img3+last", "h0_last", "h4_img3+last", "h4_last"}
    assert res["features"]["h0_img3+last"]["dim"] == 8192 and res["features"]["h4_last"]["dim"] == 2048
    assert res["selected_hidden"] in res["features"] and res["selected_hidden"] != "kv_L4"
    assert res["features"]["kv_L4"]["test"]["n"] == 16
    assert res["serving_reference"]["subtask3"]["false_carry_coarse"] == 1.0
    assert set(res["verdict"]) >= {"hidden_better", "values"}
    assert (out / "models" / "kv_L4.npz").exists()


class _StubProbe:
    """`SceneProbe.measure` stand-in: qpos[0] = palm distance (m), qpos[1] = fingers, qpos[2] = apple z, qpos[3] = basket."""

    def measure(self, qpos, side):
        return {"fingers_touch": int(qpos[1]), "apple_z_base": float(qpos[2]), "palm_dist_m": float(qpos[0]),
                "in_basket": bool(qpos[3]), "apple_crate_xy_mm": [0.0, 0.0]}


def test_label_run_pairs_requests_by_order(tmp_path):
    """Server chunk t_step is its request counter (1, 2, ...), the client's is the control step (0, 8, ...):
    the i-th chunk pairs with the i-th client step, never by equal t_step."""
    run = tmp_path / "client" / "ag3s" / "run_0000"
    run.mkdir(parents=True)
    qs = [[0.2, 0, 0.80, 0], [0.05, 0, 0.80, 0], [0.03, 2, 0.83, 0], [0.03, 2, 0.85, 1], [0.3, 0, 0.82, 1]]
    chunks = []
    for j, q in enumerate(qs):
        np.savez(run / f"step_{j:05d}.npz", t_step=np.int64(8 * j), qpos=np.asarray(q, float))
        f = tmp_path / f"chunk_{j:05d}.npz"
        summ = {"subtask": {"label": "pick"}, "hsr6": {"phase": ["approach", "grasp", "lift", "placement", "retreat"][j]}}
        np.savez(f, t_step=np.int64(j + 1), summary_json=np.asarray(json.dumps(summ)))
        chunks.append(str(f))
    chunks.append(str(tmp_path / "chunk_missing.npz"))
    np.savez(chunks[-1], t_step=np.int64(6), summary_json=np.asarray("{}"))
    entry = {"label": "x", "client_dir": str(tmp_path / "client"), "chunks": chunks, "side": "left", "split": "test"}
    r = tl.label_run(_StubProbe(), entry)
    assert [x.get("t_step") for x in r["rows"]] == [0, 8, 16, 24, 32, None]
    assert [x["chunk_t_step"] for x in r["rows"]] == [1, 2, 3, 4, 5, 6]
    assert [x["gt"]["phase5"] for x in r["rows"][:5]] == ["approach", "grasp", "carry", "place", "retreat"]
    assert [x["serving"]["hsr6_to5"] for x in r["rows"][:5]] == ["approach", "grasp", "carry", "place", "retreat"]
    assert r["rows"][5]["missing_step"] and r["n_missing"] == 1 and r["phase5_counts"]["carry"] == 1
