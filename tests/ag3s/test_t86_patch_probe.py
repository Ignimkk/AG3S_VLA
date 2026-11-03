"""T86 -- ATS target finding: hidden image-patch probe vs noun attention (`handoff/T86.task.md` §4.4).

1. default paths of `pi05_hidden` unchanged (T73 / T77 / T79 tests run as they are; here: parser defaults, routing of the
   three new flags, the new flags refused outside ``--pool patches``, ``load_hidden_policy`` default);
2. patch labels from a segmentation (known pixels): 224-grid mapping, 14 x 14 tokens, the 30 % thresholds, other-object
   share, the real scene's geom classes;
3. prompt override: the noun token row follows the prompt (real tokenizer when cached), the override json, counterfactual
   fruit / prompt rules;
4. the extractor body: hidden of the three cameras equals the T79 head slice, the noun row equals the attention
   probabilities' row, shapes on a dummy pi0.5 with both flags;
5. metrics (top-1, component majority, AUROC), train / test split without leakage (selection on train only), verdict.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.sources import pi05_hidden as ph
from AG3S.ag3s.experiments.studies import t79_events as te
from AG3S.ag3s.experiments.studies import t86_labels as tl
from AG3S.ag3s.experiments.studies import t86_probe as tp

REPO = pathlib.Path(__file__).resolve().parents[2]
APPLE, BANANA, ORANGE, PEAR, CRATE, ROBOT, OBST, SCENE = range(8)


def test_imports_from_this_checkout():
    for mod in (ph, tl, tp):
        assert pathlib.Path(mod.__file__).resolve().is_relative_to(REPO), mod.__file__


# ------------------------------------------------------------------------------ 1. default paths


class _Stop(Exception):
    pass


def test_parser_defaults_unchanged_and_t86_routing(monkeypatch):
    a = ph.build_parser().parse_args(["--pool", "patches", "--spec", "/s.json"])
    assert a.cameras is None and a.prompt_override is None and a.noun_attention is None
    assert a.patch_dtype == "float32" and a.requests is None and not ph.t86_mode(a)
    b = ph.build_parser().parse_args(["--pool", "patches", "--noun-attention"])
    assert b.noun_attention == "10:7" and ph.t86_mode(b)
    assert ph.parse_noun_cell("10:7") == (10, 7) and ph.parse_noun_cell(None) is None
    with pytest.raises(SystemExit):
        ph.parse_noun_cell("19:0")
    seen = []
    monkeypatch.setattr(ph, "main_patches_t86", lambda a: seen.append("t86"))
    monkeypatch.setattr(ph, "setup_gpu", lambda: seen.append("t79 path reached") or (_ for _ in ()).throw(_Stop()))
    ph.main(["--pool", "patches", "--spec", "/s.json", "--cameras", "cam_high", "cam_left_wrist"])
    ph.main(["--pool", "patches", "--spec", "/s.json", "--prompt-override", "/p.json"])
    ph.main(["--pool", "patches", "--spec", "/s.json", "--noun-attention"])
    assert seen == ["t86"] * 3
    seen.clear()
    with pytest.raises(_Stop):  # no T86 flag: the T79 `main_patches` body (reaches setup_gpu), not main_patches_t86
        ph.main(["--pool", "patches", "--spec", "/s.json", "--device", "gpu"])
    assert seen == ["t79 path reached"]


def test_t86_flags_refused_outside_patches():
    for extra in (["--cameras", "cam_high"], ["--prompt-override", "/p.json"], ["--noun-attention"]):
        for pool in ("none", "summary", "mean"):
            with pytest.raises(SystemExit):
                ph.main(["--records", "/x/run_0000", "--pool", pool, *extra])


def test_load_hidden_policy_default_has_no_attention_flag():
    import inspect

    assert inspect.signature(ph.load_hidden_policy).parameters["attn_probs"].default is False


def test_load_prompt_overrides(tmp_path):
    f = tmp_path / "o.json"
    f.write_text(json.dumps({"prompt_override": {"A/b": {"3": ["p3"], "7": ["p7", "q7"]}}, "runs": []}))
    assert ph.load_prompt_overrides(str(f)) == {"A/b": {3: ["p3"], 7: ["p7", "q7"]}}
    f.write_text(json.dumps({"A/b": {"1": "only"}}))
    assert ph.load_prompt_overrides(str(f)) == {"A/b": {1: ["only"]}}
    assert ph.load_prompt_overrides(None) == {}


# ------------------------------------------------------------------------------ 2. patch labels (known pixels)


def _grid(h=224, w=224):
    return np.full((h, w), -1, np.int64)


CLS = np.asarray([APPLE, BANANA, ORANGE, PEAR, CRATE, ROBOT, OBST, SCENE], np.int64)  # geom id i -> class i


def test_patch_fractions_known_pixels():
    g = _grid()
    g[28:42, 14:28] = APPLE          # token row 2, col 1 -> 33, fully apple
    g[0:14, 0:7] = BANANA            # token 0, half banana
    g[0:14, 7:14] = ROBOT            # ... half robot
    g[56:70, 42:49] = SCENE          # token 3*16+3 = 51: half table, half no-geom (-1 -> scene)
    f = tl.patch_fractions(g, CLS)
    assert f.shape == (256, 8)
    np.testing.assert_allclose(f.sum(1), 1.0)
    assert f[33, APPLE] == 1.0 and f[0, BANANA] == 0.5 and f[0, ROBOT] == 0.5 and f[51, SCENE] == 1.0
    assert f[255, SCENE] == 1.0 and f[255, :7].sum() == 0.0
    # the token grid is te.patch_of_policy_pixel (SigLIP 14-px patches, row-major), not a second implementation
    assert te.patch_of_policy_pixel(14 * 1 + 7, 14 * 2 + 7) == 33


def test_policy_grid_roundtrip_from_299_wide_render():
    """The render is 299 x 224 and the policy image the 4:3 frame stretched: a 224-grid picture expanded to 299 columns
    maps back to itself, so token <-> pixel membership is the one of the 224 image."""
    rng = np.random.default_rng(0)
    g224 = rng.integers(0, 8, size=(224, 224))
    ucol = np.floor((np.arange(299) + 0.5) * 224 / 299).astype(int)
    src = g224[:, ucol]
    assert src.shape == (224, 299)
    assert np.array_equal(tl.policy_grid_ids(src), g224)


def test_patch_labels_thresholds():
    f = np.zeros((256, 8))
    f[:, SCENE] = 1.0

    def put(k, **kw):
        f[k] = 0
        rest = 1.0 - sum(kw.values())
        f[k, SCENE] = rest
        for name, v in kw.items():
            f[k, tl.SEG_CLASSES.index(name)] = v

    put(0, apple=58 / 196)           # 29.6 % < 30 %
    put(1, apple=59 / 196)           # 30.1 % >= 30 %
    put(2, banana=0.5)               # another fruit -> other
    put(3, crate=0.2, robot=0.15)    # two objects, 35 % together -> other
    put(4, robot=0.2)                # 20 % -> background
    put(5, apple=0.35, robot=0.5)    # target wins
    assert tl.TARGET_FRAC == 0.20             # T86.task.md §2.1 (decided before results)
    lab = tl.patch_labels(f, "apple", 0.30)
    assert lab[:6].tolist() == [tl.BG, tl.TARGET, tl.OTHER, tl.OTHER, tl.BG, tl.TARGET] and (lab[6:] == tl.BG).all()
    # default 0.20, the same value for target and other: 58 px is a target, a 20 % robot token is now an other token
    lab2 = tl.patch_labels(f, "apple")
    assert lab2[0] == tl.TARGET and lab2[4] == tl.OTHER and lab2[2] == tl.OTHER
    g = np.zeros((256, 8))
    g[:, SCENE] = 1.0
    g[0] = [39 / 196, 0, 0, 0, 0, 0, 0, 1 - 39 / 196]
    g[1] = [40 / 196, 0, 0, 0, 0, 0, 0, 1 - 40 / 196]
    g[2] = [0, 0, 0, 0, 39 / 196, 0, 0, 1 - 39 / 196]
    g[3] = [0, 0, 0, 0, 40 / 196, 0, 0, 1 - 40 / 196]
    assert tl.patch_labels(g, "apple")[:4].tolist() == [tl.BG, tl.TARGET, tl.BG, tl.OTHER]
    lab_b = tl.patch_labels(f, "banana", 0.30)   # same image, the prompt names the banana
    assert lab_b[2] == tl.TARGET and lab_b[1] == tl.OTHER and lab_b[5] == tl.OTHER


def test_geom_classes_by_root_body():
    assert tl.geom_classes(["apple", "pear", "crate", "base", "pick_obstacle_hurdle_0", "table", "office",
                            "shelf", "world"]).tolist() == [APPLE, PEAR, CRATE, ROBOT, OBST, SCENE, SCENE, SCENE, SCENE]


def test_seg_scene_renders_real_scene_classes():
    pytest.importorskip("mujoco")
    try:
        import rby1_manipulation.simulation.pick_place_obstacles  # noqa: F401
    except Exception:
        pytest.skip("rby1_manipulation not importable (PYTHONPATH)")
    xml = REPO / "robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml"
    if not xml.exists():
        pytest.skip("scene XML missing")
    try:
        sc = tl.SegScene(str(xml), (None, "clear"))
        sc._pose(sc.m.qpos0.copy(), np.zeros(sc.m.nv), False)  # noqa: SLF001
        ids = sc._seg("zed_left")  # noqa: SLF001
    except Exception as e:  # no GL context
        pytest.skip(f"cannot render: {e}")
    assert ids.shape == (224, 299) and ids.max() < sc.m.ngeom
    cls = sc.geom_cls
    seen = set(cls[ids[ids >= 0]].tolist())
    assert SCENE in seen and seen <= set(range(8))     # the nominal pose looks at the room; classes are valid
    f = tl.patch_fractions(ids, cls)
    assert f.shape == (256, 8) and np.allclose(f.sum(1), 1.0)
    # the apple body is a fruit class whatever the scene: every geom of body 'apple' maps to APPLE
    import mujoco

    b = mujoco.mj_name2id(sc.m, mujoco.mjtObj.mjOBJ_BODY, "apple")
    assert all(cls[g] == APPLE for g in range(sc.m.ngeom) if int(sc.m.body_rootid[int(sc.m.geom_bodyid[g])]) == b)


RECORDED_RUN = pathlib.Path(
    "/mnt/dev/AG3S_VLA-storage/results/development/20261029T232703+0900_8ab3c7ef7757_7199b750125b/verify/T71/S1/"
    "REG_R0_k0/runs/B_ROI_FPR/B_ROI_FPR_ep211_s2113")
OBST_CFG = ("/mnt/dev/AG3S_VLA-storage/results/development/20261019T132719+0900_f15e587290f6_11408fd6d27f/verify/"
            "IFG-APPLE/obst/ep211_place.json", "ep211_place")


def test_seg_scene_on_a_recorded_request_matches_the_recorded_rgb():
    """The segmentation is rendered at the pose the recorded RGB came from: RGB parity with the recorded policy images
    (all three cameras), and the recorded request 30 of regression 211 shows the apple in the head and left wrist views."""
    pytest.importorskip("mujoco")
    step_dir = RECORDED_RUN / "ag3s" / "run_0000"
    if not step_dir.exists():
        pytest.skip("recorded run not on this machine")
    try:
        import rby1_manipulation.simulation.pick_place_obstacles  # noqa: F401
    except Exception:
        pytest.skip("rby1_manipulation not importable (PYTHONPATH)")
    from AG3S.ag3s.experiments.studies.t77_labels import default_xml

    steps = te._sorted_steps(str(RECORDED_RUN))  # noqa: SLF001
    try:
        sc = tl.SegScene(default_xml(), OBST_CFG)
        with np.load(steps[30]) as st:
            rec = {c: np.asarray(st[f"image_{c}"]) for c in ph.POLICY_CAMERA_NAMES}
            res = sc.render(np.asarray(st["qpos"], float), np.asarray(st["qvel"], float), rec, first_request=False)
    except Exception as e:  # no GL context
        pytest.skip(f"cannot render: {e}")
    assert max(res["mad"].values()) <= 0.5 and res["after_step"]
    with np.load(steps[30]) as st:  # the screening render (head only) equals the full render's head shares
        quick = sc.head_frac(np.asarray(st["qpos"], float), np.asarray(st["qvel"], float))
    np.testing.assert_array_equal(quick.astype(np.float32), res["frac"][0])
    lab = [tl.patch_labels(res["frac"][c], "apple") for c in range(3)]
    assert (lab[0] == tl.TARGET).sum() >= 1 and (lab[1] == tl.TARGET).sum() >= 1
    assert tl.pick_counterfactual(res["frac"][0], np.random.default_rng(0)) in tl.CF_FRUITS


# ------------------------------------------------------------------------------ 3. prompt override, counterfactual


def test_noun_token_row_follows_the_prompt():
    pytest.importorskip("openpi")
    pytest.importorskip("jax")
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer

        from AG3S.trajopt.attention_source import NounTokens, noun_lookup

        nt = NounTokens.from_config(ph.DEFAULT_CONFIG)
        tk = PaligemmaTokenizer(200)
    except Exception as e:  # tokenizer model not cached offline
        pytest.skip(f"tokenizer unavailable: {e}")
    state = np.zeros(16)
    seen = {}
    for prompt, noun in (("Place the apple into the basket.", "apple"),
                         ("Pick up the banana and place it in the basket.", "banana"),
                         ("Pick up the orange and place it in the basket.", "orange"),
                         ("Pick up the pear and place it in the basket.", "pear")):
        toks, mask = tk.tokenize(prompt, state)
        lk = noun_lookup({"tokenized_prompt": toks, "tokenized_prompt_mask": mask}, prompt, nt)
        assert lk.ok and lk.noun == noun
        assert int(lk.rows[0]) == ph.LANGUAGE_TOKEN_START + lk.token_index
        assert tk._tokenizer.id_to_piece(int(toks[lk.token_index])) == "▁" + noun  # noqa: SLF001
        seen[noun] = (int(lk.rows[0]), int(lk.rows[1]))
    assert len({v[0] for k, v in seen.items() if k != "apple"}) == 1   # same template -> same noun row
    assert seen["apple"][0] != seen["banana"][0]                        # recorded template differs
    # a prompt naming two fruits has no unambiguous noun row
    toks, mask = tk.tokenize("Put the apple next to the banana in the basket.", state)
    assert not noun_lookup({"tokenized_prompt": toks, "tokenized_prompt_mask": mask},
                           "Put the apple next to the banana in the basket.", nt).ok


def test_counterfactual_prompt_and_fruit_rules():
    assert tl.cf_prompt_for("pear", "Place the apple into the basket.") == "Pick up the pear and place it in the basket."
    assert tl.cf_prompt_for("pear", "Place the apple into the basket.", "same") == "Place the pear into the basket."
    assert tl.cf_prompt_for("pear", None, "Take the {fruit}.") == "Take the pear."
    with pytest.raises(ValueError):
        tl.cf_prompt_for("pear", None, "same")
    assert tl.fruit_of_prompt("Place the apple into the basket.") == "apple"
    with pytest.raises(ValueError):
        tl.fruit_of_prompt("apple and pear")
    f = np.zeros((256, 8))
    f[:, SCENE] = 1.0
    f[10] = [0, 1, 0, 0, 0, 0, 0, 0]   # a banana token in the head view
    f[20] = [1, 0, 0, 0, 0, 0, 0, 0]   # the apple itself
    rng = np.random.default_rng(0)
    assert tl.pick_counterfactual(f, rng) == "banana"             # the only non-apple fruit with a target token
    f[30] = [0, 0, 0, 1, 0, 0, 0, 0]
    picks = {tl.pick_counterfactual(f, np.random.default_rng(s)) for s in range(30)}
    assert picks == {"banana", "pear"}                             # never the original fruit, never an unseen one
    f[10] = f[30] = [0, 0, 0, 0, 0, 0, 0, 1]
    assert tl.pick_counterfactual(f, rng) is None


def test_request_order_is_seeded_and_uniform_per_run():
    cand = list(range(40))
    a = tl.request_order(cand, "K/run1", 0)
    assert a == tl.request_order(cand, "K/run1", 0) and sorted(a) == cand
    assert a != tl.request_order(cand, "K/run2", 0) and a != tl.request_order(cand, "K/run1", 1)
    assert tl.request_order([], "x") == []


# ------------------------------------------------------------------------------ 4. extractor body


def test_probe_outputs_numpy_slices_and_noun_row():
    rng = np.random.default_rng(0)
    L, T, D, G = 6, 968, 8, 8
    hidden = rng.normal(size=(L + 1, 1, T, D)).astype(np.float32)
    logits = rng.normal(size=(L, 1, 1, G, T, T))
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs = (probs / probs.sum(-1, keepdims=True)).astype(np.float32)
    rows = np.asarray([773, 776, 848], np.int32)
    layers, ranges = (2, 4), tuple(tuple(ph.CAMERA_BINDINGS[c][2]) for c in ph.POLICY_CAMERA_NAMES)
    out = ph.probe_outputs(np, hidden, probs, rows, layers, ranges, (3, 5))
    assert out["h"].shape == (3, 2, 256, D) and out["absmax"].shape == (2,)
    for ci, (lo, hi) in enumerate(ranges):
        np.testing.assert_array_equal(out["h"][ci], hidden[[2, 4], 0, lo:hi])
    # the head camera slice is exactly what T79's --pool patches stores
    np.testing.assert_array_equal(out["h"][0], hidden[[2, 4], 0, 0:256])
    np.testing.assert_allclose(out["att"], probs[3, 0, 0, 5, rows, :768], rtol=1e-6)
    np.testing.assert_allclose(out["image_mass"] + out["text_mass"], 1.0, atol=1e-5)
    off = ph.probe_outputs(np, hidden, None, rows, layers, ranges, None)
    assert set(off) == {"h", "absmax"}


def test_probe_outputs_jax_equals_numpy():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rng = np.random.default_rng(1)
    hidden = rng.normal(size=(5, 1, 968, 4)).astype(np.float32)
    probs = rng.random(size=(4, 1, 1, 8, 968, 968)).astype(np.float32)
    probs /= probs.sum(-1, keepdims=True)
    rows = np.asarray([770, 780, 850], np.int32)
    args = ((1, 3), tuple(tuple(ph.CAMERA_BINDINGS[c][2]) for c in ph.POLICY_CAMERA_NAMES), (2, 6))
    ref = ph.probe_outputs(np, hidden, probs, rows, *args)
    got = jax.jit(lambda h, p, r: ph.probe_outputs(jnp, h, p, r, *args))(hidden, probs, rows)
    for k in ref:
        np.testing.assert_allclose(np.asarray(got[k]), ref[k], rtol=1e-5, atol=1e-6)


def _dummy_cfg(**kw):
    from openpi.models import pi0_config

    return pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", **kw)


def test_probe_extractor_abstract_shapes_with_both_flags():
    pytest.importorskip("flax")
    pytest.importorskip("openpi")
    import flax.nnx as nnx
    import jax

    cfg = _dummy_cfg(return_hidden_states=True, return_attn_probs=True)
    model = nnx.eval_shape(cfg.create, jax.random.key(0))
    ex = ph.ProbeExtractor(model, [1, 3], ph.POLICY_CAMERA_NAMES, (3, 5))  # the dummy has 4 blocks, 8 heads
    obs = cfg.fake_obs()
    rows = jax.ShapeDtypeStruct((3,), np.int32)
    out = jax.eval_shape(ex._probe, ex._state, obs, rows)  # noqa: SLF001
    assert out["h"].shape == (3, 2, 256, 64) and str(out["h"].dtype) == "float32" and out["absmax"].shape == (2,)
    assert out["att"].shape == (3, 768) and out["image_mass"].shape == (3,)
    # the self-check path (HiddenExtractor contract) still works with the probabilities on
    prefix_out, kv_v, hidden, mask = jax.eval_shape(ex._prefix, ex._state, obs)  # noqa: SLF001
    T = 3 * 256 + cfg.max_token_len
    assert hidden.shape == (5, 1, T, 64) and kv_v.shape == (4, 1, T, 1, 16) and mask.shape == (1, T)

    cfg_h = _dummy_cfg(return_hidden_states=True)
    ex_h = ph.ProbeExtractor(nnx.eval_shape(cfg_h.create, jax.random.key(0)), [1, 3], ["cam_high"], None)
    out_h = jax.eval_shape(ex_h._probe, ex_h._state, cfg_h.fake_obs(), rows)  # noqa: SLF001
    assert set(out_h) == {"h", "absmax"} and out_h["h"].shape == (1, 2, 256, 64)
    with pytest.raises(ValueError):  # a cell without the attention model is refused, not silently empty
        bad = ph.ProbeExtractor(nnx.eval_shape(cfg_h.create, jax.random.key(0)), [1], ["cam_high"], (3, 5))
        jax.eval_shape(bad._probe, bad._state, cfg_h.fake_obs(), rows)  # noqa: SLF001


def test_gemma_both_flags_keep_each_output():
    """probs + hidden together = probs only = hidden only (the combined model changes nothing it returns)."""
    pytest.importorskip("flax")
    pytest.importorskip("openpi")
    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    import openpi.models.gemma as gemma
    import openpi.models.lora as lora
    from openpi.models.pi0 import make_attn_mask

    lc = {"attn": lora.LoRAConfig(rank=4, alpha=4.0), "ffn": lora.LoRAConfig(rank=4, alpha=4.0)}
    cfgs = [gemma.Config(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16, lora_configs=lc),
            gemma.Config(width=32, depth=4, mlp_dim=64, num_heads=8, num_kv_heads=1, head_dim=16, lora_configs=lc)]

    def mod(**kw):
        return gemma.Module(configs=cfgs, embed_dtype="float32", adarms=True, **kw)

    params = nn.Module.init(mod(), jax.random.key(0), use_adarms=[False, True], method="init")
    x0 = jax.random.normal(jax.random.key(2), (1, 12, 64))
    valid = jnp.ones((1, 12), bool)
    mask = make_attn_mask(valid, jnp.zeros(12, bool))
    pos = jnp.cumsum(valid, axis=1) - 1
    both = mod(return_probs=True, return_hidden=True).apply(params, [x0, None], pos, mask)
    probs_only = mod(return_probs=True).apply(params, [x0, None], pos, mask)
    hid_only = mod(return_hidden=True).apply(params, [x0, None], pos, mask)
    assert len(both) == 4 and len(probs_only) == 3 and len(hid_only) == 3
    np.testing.assert_array_equal(np.asarray(both[2]), np.asarray(probs_only[2]))
    np.testing.assert_array_equal(np.asarray(both[3]), np.asarray(hid_only[2]))
    np.testing.assert_array_equal(np.asarray(both[0][0]), np.asarray(probs_only[0][0]))


def test_main_patches_t86_writes_the_layout(tmp_path, monkeypatch):
    """The T86 driver end to end with a fake model: request selection (request j = j-th client step by t_step), the
    original and the counterfactual forward of the same image, the noun row per prompt, file layout, resume."""
    pytest.importorskip("openpi")
    import types

    from AG3S.ag3s.experiments.sources import pi05_attention

    run = tmp_path / "client" / "ag3s" / "run_0000"
    run.mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({"prompt": "Place the apple into the basket."}))
    for i, t in enumerate((16, 0, 8, 24)):   # files out of t_step order on purpose
        np.savez(run / f"step_{i:05d}.npz", t_step=np.int64(t), state=np.full(16, float(t)),
                 **{f"image_{c}": np.full((224, 224, 3), t, np.uint8) for c in ph.POLICY_CAMERA_NAMES})
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"runs": [{"label": "K/run", "client_dir": str(tmp_path / "client")}]}))
    sel = tmp_path / "samples.json"
    sel.write_text(json.dumps({"patch_requests": {"K/run": [1, 3]},
                               "prompt_override": {"K/run": {"3": ["Pick up the pear and place it in the basket."]}}}))

    calls = []

    class FakeExtractor:
        def __init__(self, model, layers, cameras, cell):
            self.layers, self.cameras, self.cell = tuple(layers), tuple(cameras), cell
            self.ranges = tuple(tuple(ph.CAMERA_BINDINGS[c][2]) for c in cameras)
            self.probe_compile_s = None

        def llm_params(self):
            return None

        def probe(self, obs, rows):
            self.probe_compile_s = 0.0
            calls.append((obs["prompt"], int(obs["state"][0]), np.asarray(rows).tolist()))
            n_c, n_l = len(self.cameras), len(self.layers)
            h = np.full((n_c, n_l, 256, 64), float(obs["state"][0]), np.float32)  # value = the request's t_step
            out = {"h": h, "absmax": np.full(n_l, 3.0, np.float32)}
            if self.cell is not None:
                att = np.zeros((3, 768), np.float32)
                att[0, int(rows[0]) - ph.LANGUAGE_TOKEN_START] = 1.0  # the noun row decides where the attention sits
                out.update(att=att, image_mass=np.asarray([0.6, 0.0, 0.0]), text_mass=np.asarray([0.4, 0.0, 0.0]))
            return out

    def fake_obs(policy, fr, prompt):
        return {"prompt": prompt, "state": np.asarray(fr.state), "tokenized_prompt": np.zeros((1, 200), np.int64),
                "tokenized_prompt_mask": np.ones((1, 200), bool)}

    class _Obs(dict):
        tokenized_prompt = property(lambda self: self["tokenized_prompt"])
        tokenized_prompt_mask = property(lambda self: self["tokenized_prompt_mask"])

    class FakeLookup:
        def __init__(self, prompt):
            self.ok = True
            self.token_index = 5 if "apple" in prompt else 6
            self.rows = np.asarray([ph.LANGUAGE_TOKEN_START + self.token_index, 900, 950], np.int32)

    import AG3S.trajopt.attention_source as asrc

    monkeypatch.setattr(ph, "setup_gpu", lambda: [0])
    monkeypatch.setattr(ph, "check_backend", lambda device: None)
    monkeypatch.setattr(ph, "load_hidden_policy", lambda cfg, ckpt, attn_probs=False: (
        types.SimpleNamespace(_model=None), types.SimpleNamespace(model=types.SimpleNamespace(paligemma_variant="dummy"))))
    monkeypatch.setattr(ph, "ProbeExtractor", FakeExtractor)
    monkeypatch.setattr(ph, "code_provenance", lambda: {"code_commit": "x", "code_dirty": False, "code_dirty_files": []})
    monkeypatch.setattr(pi05_attention, "observation_from_step", lambda p, fr, pr: _Obs(fake_obs(p, fr, pr)))
    monkeypatch.setattr(asrc.NounTokens, "from_config", classmethod(lambda cls, c: object()))
    monkeypatch.setattr(asrc, "noun_lookup", lambda inputs, prompt, tokens: FakeLookup(prompt))
    # the FakeExtractor reads obs["prompt"] / obs["state"]
    out_dir = tmp_path / "out"
    argv = ["--pool", "patches", "--device", "gpu", "--layers", "1", "3", "--cameras", "cam_high", "cam_right_wrist",
            "--noun-attention", "--patch-dtype", "float16", "--spec", str(spec), "--requests", str(sel),
            "--prompt-override", str(sel), "--out-dir", str(out_dir), "--no-check", "--checkpoint", "/ckpt"]
    ph.main(argv)
    z = np.load(out_dir / "K_run.npz")
    assert z["patches"].shape == (2, 2, 2, 256, 64) and z["patches"].dtype == np.float16
    assert z["cf_patches"].shape == (1, 2, 2, 256, 64)
    assert z["request"].tolist() == [1, 3] and z["t_step"].tolist() == [8, 24]      # j-th client step sorted by t_step
    assert z["cf_request"].tolist() == [1] and str(z["cf_prompt"][0]).startswith("Pick up the pear")
    assert float(z["patches"][0, 0, 0, 0, 0]) == 8.0 and float(z["patches"][1, 0, 0, 0, 0]) == 24.0
    assert float(z["cf_patches"][0, 0, 0, 0, 0]) == 24.0                            # the same image as request 3
    assert z["camera_names"].tolist() == ["cam_high", "cam_right_wrist"] and z["layers"].tolist() == [1, 3]
    assert z["noun_ok"].all() and z["cf_noun_ok"].all()
    assert int(np.argmax(z["noun_att"][0])) == 5 and int(np.argmax(z["cf_noun_att"][0])) == 6   # row follows the prompt
    assert z["noun_token_index"].tolist() == [5, 5] and z["cf_noun_token_index"].tolist() == [6]
    assert z["noun_cell"].tolist() == [10, 7]
    assert [c[0] for c in calls if c[1] == 24] == ["Place the apple into the basket.",
                                                  "Pick up the pear and place it in the basket."]
    n = len(calls)
    ph.main(argv)                      # resumable: the existing file is skipped, no forward is repeated
    assert len(calls) == n
    # an override for a request that is not extracted is refused
    sel.write_text(json.dumps({"patch_requests": {"K/run": [1]},
                               "prompt_override": {"K/run": {"3": ["Pick up the pear and place it in the basket."]}}}))
    with pytest.raises(SystemExit):
        ph.main([a if a != str(out_dir) else str(tmp_path / "out2") for a in argv])


# ------------------------------------------------------------------------------ 5. metrics, split, verdict


def test_map_metrics_top1_component_and_auroc():
    labels = np.full(256, tl.BG, np.int8)
    labels[[40, 41]] = tl.TARGET           # two adjacent target tokens (k = 2)
    labels[[100]] = tl.OTHER
    s = np.zeros(256)
    s[[40, 41]] = [1.0, 0.9]
    m = tp.map_metrics(s, labels)
    assert m["k"] == 2 and m["top1"] and m["selected"] and m["auroc"] == 1.0 and m["component"] == 2
    s2 = np.zeros(256)
    s2[40], s2[100] = 0.9, 1.0              # top-1 on the other object; k = 2 -> {100, 40} are not connected
    m2 = tp.map_metrics(s2, labels)
    assert not m2["top1"] and m2["component"] == 1
    # ties in the component: the larger summed score wins, so the better token decides (here the other object)
    assert not m2["selected"]
    # majority: a component of 2 target + 1 other tokens is a target selection, 1 + 1 is a tie -> not selected
    assert tp.majority_label(labels, np.asarray([40, 41, 100])) == tl.TARGET
    assert tp.majority_label(labels, np.asarray([40, 100])) == -1
    assert tp.map_metrics(np.zeros(256), np.full(256, tl.BG)) is None
    # 4-connectivity: diagonal neighbours are two components
    s3 = np.zeros(256)
    s3[[0, 17]] = 1.0                        # (0,0) and (1,1)
    lab3 = np.full(256, tl.BG, np.int8)
    lab3[[0, 17]] = tl.TARGET
    assert tp.top_component(s3, 2).tolist() == [0]
    assert tp.auroc(np.arange(10.0), np.arange(10) >= 5) == 1.0
    assert tp.auroc(np.zeros(10), np.arange(10) >= 5) == 0.5
    assert tp.auroc(np.zeros(3), np.zeros(3, bool)) is None


def test_cv_pick_layer_pick_and_verdict():
    assert tp.cv_pick({0.1: (0.5, 0.9), 1.0: (0.5, 0.9), 0.01: (0.4, 0.99)})[0] == 0.1   # ties -> smaller C
    assert tp.pick_layer({4: (0.6, 0.8), 8: (0.6, 0.8), 12: (0.5, 0.99)}) == 4
    a = {"selected": 0.30, "follow": 0.20}
    ok = tp.verdict(a, {"selected": 0.45, "follow": 0.85})
    assert ok["better"] and ok["cond_selected_ge_A_plus_10pp"] and ok["cond_follow_ge_0p8"] and ok["cond_follow_ge_A"]
    assert not tp.verdict(a, {"selected": 0.39, "follow": 0.85})["better"]        # gain 9 pp
    assert not tp.verdict(a, {"selected": 0.45, "follow": 0.79})["better"]        # follows < 0.8
    assert not tp.verdict({"selected": 0.3, "follow": 0.9}, {"selected": 0.5, "follow": 0.85})["better"]  # < A
    assert not tp.verdict({"selected": None, "follow": None}, {"selected": 0.5, "follow": 0.9})["better"]
    r = tp.rates([{"top1": True, "selected": True, "auroc": 1.0}, {"top1": False, "selected": False, "auroc": 0.5}, None])
    assert r["n"] == 2 and r["selected"] == 0.5 and r["auroc"] == 0.75


def test_group_folds_keep_scenes_together():
    scenes = np.repeat(np.asarray(["250", "320", "850", "121", "123", "1345"]), [5, 4, 6, 3, 5, 2])
    folds = tp.group_folds(scenes, 4)
    assert len(folds) == 4
    for tr, te_ in folds:
        assert not set(scenes[tr]) & set(scenes[te_])
    assert sorted(np.concatenate([te_ for _, te_ in folds]).tolist()) == list(range(len(scenes)))
    assert len(tp.group_folds(scenes[:9], 4)) == 2    # fewer scenes than folds


def test_scene_rates_equal_weight_and_exclusion():
    def row(kind, scene, ok):
        return {"kind": kind, "scene": scene,
                "m": {"top1": ok, "selected": ok, "auroc": 1.0 if ok else 0.0, "k": 1}}

    rows = ([row("orig", "A", True)] * 25                                  # scene A: 25 requests, all selected
            + [row("orig", "B", False)] * 20                               # scene B: exactly 20, none selected
            + [row("orig", "C", True)] * 19                                # scene C: 19 -> left out
            + [row("cf", "A", True)] * 30 + [row("cf", "B", True)] * 10)   # cf: only A has >= 20
    r = tp.scene_rates(rows)
    o, c = r["mean"]["orig"], r["mean"]["cf"]
    assert o["scenes"] == ["A", "B"] and o["excluded"] == ["C"] and o["n_scenes"] == 2
    assert o["selected"] == 0.5                       # (1.0 + 0.0) / 2: the scenes weigh the same, not 25 : 20
    assert tp.rates([x["m"] for x in rows if x["kind"] == "orig"])["selected"] == 44 / 64   # pooled differs
    assert c["scenes"] == ["A"] and c["excluded"] == ["B"] and c["selected"] == 1.0
    assert r["per_scene"]["orig"]["C"]["included"] is False and r["per_scene"]["orig"]["C"]["n"] == 19
    assert tp.scene_rates([])["mean"]["orig"]["selected"] is None
    assert tp.scene_rates(rows, min_n=19)["mean"]["orig"]["scenes"] == ["A", "B", "C"]


def test_centred_removes_the_common_component():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(3, 256, 8)).astype(np.float32) + 5.0
    c = tp.centred(h)
    np.testing.assert_allclose(c.mean(axis=-2), 0.0, atol=1e-5)
    np.testing.assert_allclose(c, h - h.mean(axis=-2, keepdims=True), atol=1e-6)


def _synthetic(tmp_path, scenes, test_noise_seed, D=12, good_layer=4):
    """Runs on several scenes (split by scene number). Head camera only informative: at `good_layer` the target fruit's
    tokens carry a fruit-specific direction, at the other layer nothing; the noun attention is random."""
    rng = np.random.default_rng(0)
    runs, fracs, run_idx, req_idx, cf_fruits = [], [], [], [], []
    dirs = {f: np.eye(D)[i] * 3.0 for i, f in enumerate(tl.FRUIT_CLASSES)}
    pdir = tmp_path / "patches"
    pdir.mkdir(exist_ok=True)
    for ri, (split, scene) in enumerate(scenes):
        label = f"K/{split}_{scene}_{ri}"
        r2 = np.random.default_rng(test_noise_seed if split == "test" else ri + 100)
        n_req = 2
        P = np.zeros((n_req, 3, 2, 256, D), np.float16)
        CF = np.zeros((n_req, 3, 2, 256, D), np.float16)
        att = np.zeros((n_req, 768), np.float32)
        cf_att = np.zeros((n_req, 768), np.float32)
        samples = []
        for q in range(n_req):
            tok_a = int(r2.integers(20, 200))
            tok_b = int(r2.integers(20, 200))
            while abs(tok_b - tok_a) < 3:
                tok_b = int(r2.integers(20, 200))
            fr = np.zeros((3, 256, 8), np.float16)
            fr[..., SCENE] = 1.0
            for c in range(3):
                fr[c, tok_a] = 0
                fr[c, tok_a, APPLE] = 0.9
                fr[c, tok_a, SCENE] = 0.1
                fr[c, tok_b] = 0
                fr[c, tok_b, BANANA] = 0.9
                fr[c, tok_b, SCENE] = 0.1
            for blk, tgt, fruit in ((P, tok_a, "apple"), (CF, tok_b, "banana")):
                x = r2.normal(size=(3, 2, 256, D)) * 0.3
                x[:, 0 if good_layer == 4 else 1, tgt] += dirs[fruit]
                blk[q] = x.astype(np.float16)
            att[q, :256] = r2.random(256)
            cf_att[q, :256] = r2.random(256)
            samples.append({"request": q, "cf_fruit": "banana", "cf_prompt": "Pick up the banana and place it in the basket."})
            fracs.append(fr)
            run_idx.append(ri)
            req_idx.append(q)
            cf_fruits.append("banana")
        runs.append({"label": label, "split": split, "scene_number": scene, "samples": samples})
        np.savez(pdir / f"{ph.safe_name(label)}.npz", patches=P, cf_patches=CF, cf_request=np.arange(n_req),
                 cf_prompt=np.asarray(["Pick up the banana and place it in the basket."] * n_req),
                 noun_att=att, cf_noun_att=cf_att, noun_ok=np.ones(n_req, bool), cf_noun_ok=np.ones(n_req, bool),
                 camera_names=np.asarray(ph.POLICY_CAMERA_NAMES), layers=np.asarray([4, 8]), request=np.arange(n_req),
                 prompt=np.asarray(["Place the apple into the basket."] * n_req))
    np.savez(tmp_path / "s.labels.npz", run=np.asarray(run_idx, np.int32), request=np.asarray(req_idx, np.int32),
             frac=np.asarray(fracs, np.float16), cf_fruit=np.asarray(cf_fruits), classes=np.asarray(tl.SEG_CLASSES))
    sj = tmp_path / "s.json"
    sj.write_text(json.dumps({"labels_npz": str(tmp_path / "s.labels.npz"), "params": {"target_frac": 0.2}, "runs": runs}))
    return str(sj), str(pdir)


SCENES = [("train", s) for s in ("250", "320", "850", "121", "123")] * 2 + [("test", s) for s in ("211", "954")] * 2


def _run(tmp_path, scenes, seed, good_layer=4):
    import unittest.mock as mock

    sj, pdir = _synthetic(tmp_path, scenes, seed, good_layer=good_layer)
    out = tmp_path / f"out{seed}"
    out.mkdir()
    # the synthetic scenes have a few requests each; the real threshold is 20
    with mock.patch.object(tp, "MIN_SCENE_REQUESTS", 2):
        return _run_camera(sj, pdir, out)


def _run_camera(sj, pdir, out):
    return tp.run_camera("cam_high", sj, pdir, [4, 8], (0.01, 1.0), neg=32, n_folds=3, seed=0, max_iter=100,
                         out_dir=out, log=lambda *a, **k: None)


def test_probe_selects_on_train_only_and_follows_the_prompt(tmp_path):
    res = _run(tmp_path, SCENES, 7)
    for v in tp.VARIANTS:
        assert res["selected"][v]["layer"] == 4                       # the layer that carries the fruit direction
    assert not set(res["train_scenes"]) & set(res["test_scenes"])
    t = res["test"]["raw_L4"]
    assert t["orig"]["selected"] == 1.0 and t["cf"]["selected"] == 1.0   # follows the prompt: banana tokens for banana
    assert t["orig"]["top1"] == 1.0 and t["orig"]["auroc"] > 0.95
    assert res["test"]["raw_L8"]["orig"]["selected"] < 0.5            # the uninformative layer is at chance
    assert res["test"]["A"]["orig"]["selected"] < 0.5                 # random noun attention
    v = res["verdict"]["raw"]
    assert v["better"] and v["selected_setting"]["layer"] == 4
    assert (tmp_path / "out7" / "test_maps_cam_high.npz").exists()


def test_selection_does_not_see_test_data(tmp_path):
    """Same train runs, two different test data sets (different noise AND an uninformative test layer 4): the CV table
    and the chosen (layer, C) are bit-identical, only the test numbers move."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _run(tmp_path / "a", SCENES, 1)
    b = _run(tmp_path / "b", SCENES, 2)
    assert a["cv"] == b["cv"] and a["selected"] == b["selected"] and a["train"] == b["train"]
    assert a["test"]["raw_L4"] != b["test"]["raw_L4"] or a["test"]["A"] != b["test"]["A"]


def test_run_camera_refuses_a_scene_on_both_sides(tmp_path):
    bad = SCENES + [("test", "250")]
    sj, pdir = _synthetic(tmp_path, bad, 3)
    with pytest.raises(SystemExit, match="both sides"):
        tp.run_camera("cam_high", sj, pdir, [4, 8], (0.1,), neg=16, n_folds=3, seed=0, max_iter=20, out_dir=tmp_path,
                      log=lambda *a, **k: None)


def test_new_tools_do_not_import_jax_at_import():
    import subprocess
    import sys

    code = ("import sys; import AG3S.ag3s.experiments.studies.t86_labels, AG3S.ag3s.experiments.studies.t86_probe; "
            "assert 'jax' not in sys.modules, 'jax imported'")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0, r.stderr
