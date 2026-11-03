"""T79 -- ATS time tracking by hidden re-search (`handoff/T79.task.md` §4.4).

1. default paths of `pi05_hidden` unchanged (T73 / T77 tests run as they are; here: defaults, routing of --pool patches);
2. head token <-> image coordinates: SigLIP 14-px patches row-major on the 224 image, the 224 image = the 640 x 480
   frame stretched, so token k <-> depth box 40 x 30 px; the jitted slice = the full hidden's head tokens;
3. 3D back-projection / projection round trip, single-camera visibility = `task_dynamic_map_eval._visible`;
4. event rules (reference = last earlier ATS-correct request, a / b / ab / normal / no_ref), normal sampling;
5. re-search parts (component, centre, majority, score), train-only selection, verdict, an end-to-end synthetic run
   where train and test prefer different layers (the selection must follow train).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.sources import pi05_hidden as ph
from AG3S.ag3s.experiments.studies import t79_events as te
from AG3S.ag3s.experiments.studies import t79_track as tt

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_imports_from_this_checkout():
    for mod in (ph, te, tt):
        assert pathlib.Path(mod.__file__).resolve().is_relative_to(REPO), mod.__file__


# ------------------------------------------------------------------------------ 1. default paths


class _Stop(Exception):
    pass


def test_parser_defaults_unchanged_and_patches_routing(monkeypatch):
    a = ph.build_parser().parse_args(["--records", "/x/run_0000"])
    assert a.pool == "none" and a.requests is None and a.device == "cpu" and a.dtype == "float16"
    seen = []
    monkeypatch.setattr(ph, "main_patches", lambda a: seen.append(("patches", a.requests)))
    monkeypatch.setattr(ph, "main_summary", lambda a: seen.append(("summary", None)))
    ph.main(["--pool", "patches", "--spec", "/s.json", "--requests", "/e.json"])
    ph.main(["--pool", "summary", "--spec", "/s.json"])
    assert seen == [("patches", "/e.json"), ("summary", None)]
    # the legacy CPU path still goes to force_cpu, and --requests is refused outside --pool patches
    monkeypatch.setattr(ph, "force_cpu", lambda t: (_ for _ in ()).throw(_Stop()))
    with pytest.raises(_Stop):
        ph.main(["--records", "/x/run_0000"])
    with pytest.raises(SystemExit):
        ph.main(["--records", "/x/run_0000", "--requests", "/e.json"])


def test_patches_refuses_dataset_and_out(monkeypatch):
    monkeypatch.setattr(ph, "setup_gpu", lambda: (_ for _ in ()).throw(AssertionError("should refuse first")))
    with pytest.raises(SystemExit):
        ph.main(["--pool", "patches", "--dataset", "/d", "--episodes", "0", "--device", "gpu"])
    with pytest.raises(SystemExit):
        ph.main(["--pool", "patches", "--spec", "/s.json", "--out", "/o.npz", "--device", "gpu"])


def test_load_request_lists(tmp_path):
    f = tmp_path / "ev.json"
    f.write_text(json.dumps({"patch_requests": {"A/b": [3, 1, 3], "C": []}, "runs": []}))
    assert ph.load_request_lists(str(f)) == {"A/b": [1, 3], "C": []}
    f.write_text(json.dumps({"A/b": [2]}))
    assert ph.load_request_lists(str(f)) == {"A/b": [2]}
    assert ph.load_request_lists(None) is None


def test_request_files_sorted_by_t_step(tmp_path):
    run = tmp_path / "run_0000"
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"prompt": "p"}))
    for name, t in (("step_00000.npz", 16), ("step_00001.npz", 0), ("step_00002.npz", 8)):
        np.savez(run / name, t_step=np.int64(t))
    kind, files, prompt, ts = ph.request_files(str(run))
    assert kind == "ag3s" and prompt == "p" and ts == [0, 8, 16]
    assert [f.name for f in files] == ["step_00001.npz", "step_00002.npz", "step_00000.npz"]


# ------------------------------------------------------------------------------ 2. token <-> image coordinates


def test_policy_pixel_patch_equals_full_frame_patch():
    """224 image = the 640 x 480 frame stretched: pixel u640 -> floor(u640 * 224 / 640); same token either way."""
    u = np.arange(640)
    v = np.arange(480)
    assert np.array_equal(te.patch_of_pixel(u, np.zeros_like(u)) % 16,
                          te.patch_of_policy_pixel((u * 224) // 640, np.zeros_like(u)) % 16)
    assert np.array_equal(te.patch_of_pixel(np.zeros_like(v), v) // 16,
                          te.patch_of_policy_pixel(np.zeros_like(v), (v * 224) // 480) // 16)
    # one token = 14 x 14 px at 224, 40 x 30 px at 640 x 480
    assert te.PATCH_PX == 14
    for k in (0, 17, 165, 255):
        u0, u1, v0, v1 = te.patch_box(k)
        assert (u1 - u0, v1 - v0) == (40, 30)
        uu, vv = np.meshgrid(np.arange(u0, u1), np.arange(v0, v1))
        assert set(te.patch_of_pixel(uu.ravel(), vv.ravel()).tolist()) == {k}
    assert te.patch_of_pixel(639, 479) == 255 and te.patch_of_policy_pixel(223, 223) == 255


def test_siglip_tokens_are_row_major_14px_patches():
    pytest.importorskip("flax")
    pytest.importorskip("openpi")
    import jax
    import jax.numpy as jnp

    from openpi.models import siglip

    mod = siglip.Module(variant="mu/14", pool_type="none")
    img = jnp.zeros((1, 224, 224, 3))
    params = mod.init(jax.random.key(0), img)
    base = mod.apply(params, img)[1]["with_posemb"]
    assert base.shape[1] == 256
    for r, c in ((0, 0), (3, 11), (15, 15)):
        im2 = img.at[0, 14 * r:14 * r + 14, 14 * c:14 * c + 14, :].set(1.0)
        diff = np.abs(np.asarray(mod.apply(params, im2)[1]["with_posemb"] - base)).sum(-1)[0]
        assert np.flatnonzero(diff > 1e-6).tolist() == [r * 16 + c]
        assert te.patch_of_policy_pixel(14 * c + 7, 14 * r + 7) == r * 16 + c


def test_patch_extractor_slices_head_tokens():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    hidden = rng.normal(size=(19, 1, 968, 8)).astype(np.float32)
    ex = object.__new__(ph.PatchExtractor)
    ex.layers = (4, 8, 12, 16)
    ex.token_range = tuple(ph.CAMERA_BINDINGS["cam_high"][2])
    ex._prefix = lambda state, obs: (None, None, jnp.asarray(hidden), None)  # noqa: SLF001
    h, a = jax.jit(ex._patches)(None, None)  # noqa: SLF001
    assert ex.token_range == (0, 256)
    np.testing.assert_array_equal(np.asarray(h), hidden[[4, 8, 12, 16], 0, 0:256])
    np.testing.assert_allclose(np.asarray(a), np.abs(hidden[[4, 8, 12, 16], 0, 0:256]).max(axis=(1, 2)))


def test_patch_extractor_abstract_shapes():
    pytest.importorskip("flax")
    pytest.importorskip("openpi")
    import flax.nnx as nnx
    import jax

    from openpi.models import pi0_config

    cfg = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                               return_hidden_states=True)
    model = nnx.eval_shape(cfg.create, jax.random.key(0))
    ex = ph.PatchExtractor(model, [1, 3])  # the dummy has 4 blocks
    h, a = jax.eval_shape(ex._patches, ex._state, cfg.fake_obs())  # noqa: SLF001
    assert h.shape == (2, 256, 64) and str(h.dtype) == "float32" and a.shape == (2,)


# ------------------------------------------------------------------------------ 3. geometry


def _cam():
    K = np.array([[240.0, 0, 320], [0, 240.0, 240], [0, 0, 1]])
    T = np.eye(4)
    c, s = np.cos(0.3), np.sin(0.3)
    T[:3, :3] = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]).T
    T[:3, 3] = [0.1, 0.05, 1.45]
    return K, T


def test_backproject_project_round_trip():
    K, T = _cam()
    u = np.array([0, 320, 401, 639])
    v = np.array([0, 240, 77, 479])
    depth = np.zeros((480, 640))
    depth[v, u] = [0.5, 0.8, 1.2, 1.9]
    P = te.backproject(depth, K, T, u, v)
    uu, vv, z = te.project(P, K, T)
    assert np.array_equal(uu, u) and np.array_equal(vv, v)
    np.testing.assert_allclose(z, depth[v, u], atol=1e-12)
    # the optical-axis pixel lands on the camera z axis at its depth
    np.testing.assert_allclose(P[1], T[:3, 3] + T[:3, 2] * 0.8, atol=1e-12)


def test_visible_in_matches_tde_visible():
    pytest.importorskip("mujoco")
    from AG3S.ag3s.experiments.studies import task_dynamic_map_eval as tde

    K, T = _cam()
    rng = np.random.default_rng(1)
    u = rng.integers(0, 640, 50)
    v = rng.integers(0, 480, 50)
    depth = rng.uniform(0.4, 1.5, (480, 640))
    P = te.backproject(depth, K, T, u, v)
    P[::3] += T[:3, 2] * 0.03  # pushed 30 mm behind the surface -> not visible
    step = {"depth_zed_left": (depth * 1000).astype(np.uint16), "K_zed_left": K, "T_base_cam_zed_left": T}

    class S(dict):
        files = list(step)

    vis, inside, _, _ = te.visible_in(P, step["depth_zed_left"] / 1000.0, K, T)
    assert np.array_equal(vis, tde._visible(P, S(step)))  # noqa: SLF001
    assert inside.all() and not vis[::3].any() and vis.sum() >= 30


def test_appearance_patches_prefer_head_confirmed_points():
    K, T = _cam()
    depth = np.full((480, 640), 1.0)
    P = te.backproject(depth, K, T, np.array([20, 300, 610]), np.array([20, 200, 460]))
    hidden = P[2:] + T[:3, 2] * 0.2  # behind the surface
    patches, src = te.appearance_patches(np.vstack([P[:2], hidden]), depth, K, T)
    assert src == "visible" and patches == sorted({int(te.patch_of_pixel(20, 20)), int(te.patch_of_pixel(300, 200))})
    patches, src = te.appearance_patches(hidden, depth, K, T)
    assert src == "projected" and len(patches) == 1
    assert te.appearance_patches(np.zeros((0, 3)), depth, K, T) == ([], "none")


# ------------------------------------------------------------------------------ 4. event rules


def test_reference_and_event_rules():
    assert te.prev_reference([False, True, False, True, True, False]) == [-1, -1, 1, 1, 3, 4]
    assert te.ats_correct(0.8, "apple") and not te.ats_correct(0.79, "apple") and not te.ats_correct(0.9, "table")
    assert not te.ats_correct(None, None)
    assert te.event_type(False, None, "occluded", 50) == "no_ref"
    assert te.event_type(True, 20.0, "visible", 10) == "a"
    assert te.event_type(True, 19.9, "visible", 10) == "normal"
    assert te.event_type(True, 5.0, "occluded", 1) == "b"
    assert te.event_type(True, 5.0, "occluded", 0) == "normal"      # occluded and truly not seen: not an event
    assert te.event_type(True, 30.0, "occluded", 3) == "ab"
    assert te.event_type(True, 30.0, "occluded", 0) == "a"


def test_sample_normals_deterministic_and_capped():
    reqs = list(range(3, 40))
    a = te.sample_normals(reqs, 4, "run/x")
    assert a == te.sample_normals(reqs, 4, "run/x") and len(a) == 4 and set(a) <= set(reqs) and a == sorted(a)
    assert te.sample_normals(reqs, 4, "run/y") != a or te.sample_normals(reqs, 4, "run/y", seed=1) != a
    assert te.sample_normals([5, 2], 4, "r") == [2, 5] and te.sample_normals(reqs, 0, "r") == []


def test_patch_table_sums():
    P = np.array([[0, 0, 1.0], [0, 0, 3.0], [1, 1, 1.0]])
    N, S, C = te.patch_table(P, np.array([5, 5, 200]), np.array([0, 0, 5]))
    assert N[5] == 2 and N[200] == 1 and N.sum() == 3
    np.testing.assert_allclose(S[5] / N[5], [0, 0, 2.0])
    assert C[5, 0] == 2 and C[200, 5] == 1


def test_scene_gt_classifies_on_the_real_scene():
    pytest.importorskip("mujoco")
    try:
        import rby1_manipulation.simulation.pick_place_obstacles  # noqa: F401
    except Exception:
        pytest.skip("rby1_manipulation not importable (PYTHONPATH)")
    xml = REPO / "robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml"
    if not xml.exists():
        pytest.skip("scene XML missing")
    sc = te.SceneGT(str(xml), (None, "clear"))
    g = sc.at(sc.gt.m.qpos0.copy())
    cls = te.SceneGT.classify(g, np.vstack([g["apple"][:20], g["apple_c"] + [0, 0, 1.5]]))
    assert (cls[:20] == te.CLASSES.index("apple")).all() and cls[20] == te.CLASSES.index("none")
    a = te.ats_of(g, g["apple"][:30])
    assert a["precision_5mm"] == 1.0 and a["majority_object"] == "apple"
    assert te.ats_correct(a["precision_5mm"], a["majority_object"])
    g2 = sc.at(sc.gt.m.qpos0.copy())  # static tree reused (base did not move)
    assert g2["parts"][0][0] is g["parts"][0][0]


# ------------------------------------------------------------------------------ 5. re-search, selection, verdict


def test_largest_component_and_centre():
    s = np.zeros(256)
    s[[0, 1, 16]] = 0.9          # 3 connected
    s[[100, 101, 117, 116]] = 0.6  # 4 connected (larger)
    s[[200, 217]] = 0.95           # diagonal only: not 4-connected
    assert tt.largest_component(s, 0.5).tolist() == [100, 101, 116, 117]
    assert tt.largest_component(s, 0.85).tolist() == [0, 1, 16]
    assert len(tt.largest_component(s, 0.99)) == 0
    N = np.zeros(256, int)
    S = np.zeros((256, 3))
    C = np.zeros((256, len(te.CLASSES)), int)
    N[[0, 1]] = [2, 2]
    S[0], S[1] = [0, 0, 2], [0, 0, 6]
    C[0, te.CLASSES.index("apple")] = 1
    C[0, te.CLASSES.index("none")] = 3
    c, maj, n = tt.centre_of(np.array([0, 1, 16]), N, S, C)
    np.testing.assert_allclose(c, [0, 0, 2.0])
    assert n == 4 and maj == te.CLASSES.index("apple")  # none ignored unless all none
    assert tt.centre_of(np.array([16]), N, S, C)[0] is None


def test_score_rules():
    v = [0.5, 0.0, 0.9]
    assert tt.score([0.5, 0.019, 0.9], tt.APPLE, v)["found"]
    r = tt.score([0.5, 0.021, 0.9], tt.APPLE, v)
    assert not r["found"] and not r["wrong"]
    assert tt.score([0.5, 0.051, 0.9], tt.APPLE, v)["wrong"]
    assert tt.score([0.5, 0.01, 0.9], te.CLASSES.index("robot"), v) == {
        "abstain": False, "found": True, "wrong": True, "err_mm": 10.0}
    assert tt.score(None, tt.NONE, v)["abstain"]


def test_select_and_verdict():
    def row(l, tau, ef, ew, nf):
        return {"layer": l, "tau": tau, "event": {"found": ef, "wrong": ew}, "normal": {"found": nf}}

    table = [row(4, 0.5, 0.9, 0.2, 0.9),    # too many wrong
             row(8, 0.6, 0.7, 0.05, 0.80),  # admissible (normal 0.80 >= 0.85 - 0.05)
             row(8, 0.7, 0.7, 0.05, 0.86),  # admissible, same found, higher tau -> chosen
             row(12, 0.6, 0.8, 0.05, 0.79)]  # normal too low
    sel = tt.select(table, 0.85)
    assert (sel["layer"], sel["tau"], sel["admissible"], sel["n_admissible"]) == (8, 0.7, True, 2)
    sel = tt.select([row(4, 0.5, 0.9, 0.2, 0.9), row(8, 0.5, 0.5, 0.0, 0.1)], 0.85)
    assert not sel["admissible"] and sel["layer"] == 4  # 0.7 > 0.5
    v = tt.verdict({"found": 0.8, "wrong": 0.1}, {"found": 0.5}, {"found": 0.80}, {"found": 0.85})
    assert v["hidden_better"]
    assert not tt.verdict({"found": 0.79, "wrong": 0.1}, {"found": 0.5}, {"found": 0.9}, {"found": 0.85})["hidden_better"]
    assert not tt.verdict({"found": 0.9, "wrong": 0.11}, {"found": 0.5}, {"found": 0.9}, {"found": 0.85})["hidden_better"]
    assert not tt.verdict({"found": 0.9, "wrong": 0.0}, {"found": 0.5}, {"found": 0.79}, {"found": 0.85})["hidden_better"]


def _synthetic(tmp_path, D=8):
    """Two runs (train / test). Apple at patch 40 (train) / 90 (test). Hidden: layer 4 carries the apple direction
    only in the test run, layer 8 only in the train run -> selection must take layer 8 (train), not 4 (test)."""
    rng = np.random.default_rng(0)
    e1 = np.zeros(D)
    e1[0] = 1.0
    runs, Ns, Ss, Cs, As, G = [], [], [], [], [], 0
    for ri, (split, kp) in enumerate((("train", 40), ("test", 90))):
        label = f"K/{split}_run"
        apple = np.array([0.5, 0.1 * ri, 0.85])
        rows = [{"request": 0, "event": "no_ref", "ref": -1, "ats_ok": True, "appearance_patches": [kp],
                 "sampled": False, "vis_n": 10, "vis_c": apple.tolist(), "state": "visible", "moved_mm": None,
                 "apple_c": apple.tolist(), "centroid": apple.tolist(), "tp_centroid": apple.tolist(),
                 "class_counts_8mm": [5, 0, 0, 0, 0, 0, 0]}]
        H = np.zeros((7, 2, 256, D), np.float32)
        H[..., 1:] = rng.normal(size=(7, 2, 256, D - 1)) * 0.3
        H[..., 0] = -1.0
        good = 1 if split == "train" else 0  # layer index (0 = L4, 1 = L8) that sees the apple
        H[:, :, kp, 0] = np.where(np.arange(2) == good, 1.0, -1.0)[None, :]
        H[0, :, kp, 0] = 1.0  # the reference frame: apple direction at both layers
        for j in range(1, 7):
            ev = "ab" if j <= 4 else "normal"
            far = (apple + [0.0, 0.12, 0.0]).tolist()  # the stale ATS centroid of an event request
            rows.append({"request": j, "event": ev, "ref": 0, "ats_ok": False, "sampled": True, "vis_n": 10,
                         "vis_c": apple.tolist(), "state": "occluded" if ev == "ab" else "visible", "moved_mm": 30.0,
                         "apple_c": apple.tolist(), "centroid": far if ev == "ab" else apple.tolist(),
                         "tp_centroid": far if ev == "ab" else apple.tolist(),
                         "class_counts_8mm": [0, 0, 0, 0, 0, 0, 5], "geom_index": G})
            N = np.zeros(256, np.int32)
            S = np.zeros((256, 3))
            C = np.zeros((256, len(te.CLASSES)), np.int32)
            N[kp] = 10
            S[kp] = 10 * apple
            C[kp, 0] = 10
            N[0] = 10                      # elsewhere: the table, 30 cm away
            S[0] = 10 * (apple + [0.3, 0, 0])
            C[0, te.CLASSES.index("table")] = 10
            att = np.zeros(256, np.float32)
            att[kp] = 1.0
            Ns.append(N); Ss.append(S); Cs.append(C); As.append(att)
            G += 1
        runs.append({"label": label, "split": split, "scene": f"s{ri}", "rows": rows})
        np.savez(tmp_path / f"{tt.safe_name(label)}.npz", patches=H, layers=np.array([4, 8]),
                 request=np.arange(7))
    np.savez(tmp_path / "ev.geom.npz", N=np.array(Ns), S=np.array(Ss), C=np.array(Cs), attention=np.array(As))
    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps({"geom": str(tmp_path / "ev.geom.npz"), "runs": runs}))
    return ev


def test_track_end_to_end_selects_on_train_only(tmp_path):
    ev = _synthetic(tmp_path)
    out = tt.run_track(tmp_path / "out", ev, tmp_path, layers=[4, 8], taus=[0.5, 0.9], focus="K/test_run",
                       focus_requests=None)
    h = out["hidden"]
    assert h["selected"]["layer"] == 8 and h["selected"]["admissible"]   # train run's good layer
    assert h["train"]["event"]["found"] == 1.0 and h["train"]["event"]["wrong"] == 0.0
    assert h["test"]["event"]["found"] == 0.0                            # layer 8 does not see the test apple
    assert out["baseline"]["test"]["event"]["found"] == 0.0 and out["baseline"]["test"]["normal"]["found"] == 1.0
    assert not out["verdict"]["hidden_better"]
    curve = {c["layer"]: c for c in h["layer_curve"]}
    assert set(curve) == {4, 8} and curve[8]["tau"] == h["selected"]["tau"]
    # the test numbers of every setting are reported (graph data), but never used for the choice
    t4 = [t for t in h["table"] if t["layer"] == 4 and t["tau"] == 0.5][0]
    assert t4["test_event"]["found"] == 1.0 and t4["event"]["found"] == 0.0
    assert out["attention"]["test"]["event"]["found"] == 1.0
    assert out["counts"]["scored"] == 12 and len(out["focus"]["rows"]) == 6
    assert out["focus"]["rows"][0]["hidden"]["found"] is False and out["focus"]["rows"][0]["hidden_to_apple_centre_mm"] is None
    assert (tmp_path / "out" / "t79_track.json").exists()


def test_focus_keeps_rows_without_visible_gt(tmp_path):
    ev = _synthetic(tmp_path)
    d = json.loads(ev.read_text())
    x = d["runs"][1]["rows"][2]
    x["vis_n"], x["vis_c"] = 0, None   # apple hidden from the head camera at this request
    ev.write_text(json.dumps(d))
    out = tt.run_track(tmp_path / "out", ev, tmp_path, layers=[4, 8], taus=[0.5, 0.9], focus="K/test_run",
                       focus_requests=range(1, 7))
    assert out["counts"]["scored"] == 11 and out["counts"]["no_visible_gt"] == 1
    f = out["focus"]
    assert f["n_rows"] == 6 and f["n_without_visible_gt"] == 1
    r = [r for r in f["rows"] if r["request"] == 2][0]
    assert r["baseline"] is None and r["baseline_to_apple_centre_mm"] == 120.0


def test_evaluate_layer_cache_equals_brute_force():
    rng = np.random.default_rng(3)
    N = rng.integers(0, 5, (3, 256))
    S = rng.normal(size=(3, 256, 3))
    C = rng.integers(0, 3, (3, 256, len(te.CLASSES)))
    geom = {"N": N, "S": S, "C": C}
    items = [{"row": {"geom_index": g, "vis_c": [0.0, 0.0, 0.0]},
              "sims": {"hidden": rng.uniform(0.9, 1.0, (2, 256))}} for g in range(3)]
    taus = [round(x, 3) for x in np.arange(0.9, 1.0, 0.004)]
    fast = tt.evaluate_layer(items, geom, "hidden", 1, taus)
    for t in taus:
        brute = []
        for it in items:
            g = it["row"]["geom_index"]
            c, maj, _ = tt.centre_of(tt.largest_component(it["sims"]["hidden"][1], t), N[g], S[g], C[g])
            brute.append(tt.score(c, maj, [0.0, 0.0, 0.0]))
        assert fast[t] == brute


def test_size_options_default_off():
    a = ph.build_parser().parse_args(["--pool", "patches", "--spec", "/s.json"])
    assert a.patch_dtype == "float32" and a.requests is None
    import inspect

    assert inspect.signature(te.build_events).parameters["event_cap"].default == 0
    assert inspect.signature(te.events_run).parameters["event_cap"].default == 0
    # the cap draws from the run's own events, seeded per label (the same helper as the normal sample)
    ev = list(range(15, 75))
    a1 = te.sample_normals(ev, 20, "event:L")
    assert len(a1) == 20 and a1 == te.sample_normals(ev, 20, "event:L") and set(a1) <= set(ev)
