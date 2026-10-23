"""Localization-heads provider (Implementer B): official-code parity, selection determinism, not-found path.

numpy-only tests run in ``.venv-ag3s``; tests marked ``torch`` need ``.venv-baselines`` (official ``analyze.py`` /
``bbox.py`` import torch). Official files are read from ``external/LocalizationHeads`` (pinned SHA, checked).
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Baseline.paper_baselines.providers import localization_heads as lh

EXT = Path(os.environ.get("PAPER_BASELINES_LOCHEADS_DIR",
                          Path(lh.__file__).resolve().parents[1] / "external" / "LocalizationHeads"))
try:
    import torch
except ImportError:  # .venv-ag3s
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="torch not importable (run in .venv-baselines)")
needs_ext = pytest.mark.skipif(not (EXT / "analyze.py").exists(), reason=f"{EXT} not fetched")


def _interface_or_skip():
    try:
        from Baseline.paper_baselines import interface
    except ImportError as exc:  # Implementer A's file not there yet
        pytest.skip(f"interface.py not importable: {exc}")
    return interface


# ---------------------------------------------------------------------------------------------- helpers
def _official_function(name: str):
    """One function of the pinned analyze.py compiled alone with numpy (no torch import)."""
    tree = ast.parse((EXT / "analyze.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    mod = ast.Module(body=[fn], type_ignores=[])
    ns = {"np": np, "List": list}
    exec(compile(mod, str(EXT / "analyze.py"), "exec"), ns)
    return ns[name]


def _load_official(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, EXT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _planted_sample(rng, L=6, H=4, P=8, planted=((3, 1), (4, 2)), peak=(2, 5)):
    """Diffuse random attention, plus heads that put a compact blob at ``peak`` (row, col)."""
    a = rng.random((L, H, P * P)).astype(np.float32) * 0.002
    yy, xx = np.mgrid[0:P, 0:P]
    blob = np.exp(-((yy - peak[0]) ** 2 + (xx - peak[1]) ** 2) / 1.5).astype(np.float32)
    for (l, h) in planted:
        a[l, h] = blob.ravel() / blob.sum() * 0.6 + rng.random(P * P).astype(np.float32) * 1e-3
    return a


def _multi_blob_sample(rng, L=8, H=4, P=8):
    """Heads with 1, 2 or 3 blobs at random places (distinct spatial entropies) over diffuse noise."""
    a = rng.random((L, H, P * P)).astype(np.float32) * 0.002
    yy, xx = np.mgrid[0:P, 0:P]
    for (l, h, k) in ((2, 0, 1), (3, 1, 2), (4, 2, 3), (5, 3, 2), (6, 1, 1), (7, 0, 3)):
        m = np.zeros((P, P), np.float32)
        for _ in range(k):
            cy, cx = rng.integers(1, P - 2, size=2)
            m += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / rng.uniform(0.6, 2.0)).astype(np.float32) \
                * rng.uniform(0.5, 1.0)
        a[l, h] = m.ravel() / m.sum() * rng.uniform(0.3, 0.7) + a[l, h]
    return a


def _write_fake_run(root: Path, name: str, episodes, rng, L=6, H=4, calls=3):
    run = root / name
    (run / "attention").mkdir(parents=True)
    rows = []
    for (task, init) in episodes:
        pooled = np.zeros((calls, L, H, 768), np.float16)
        for k in range(calls):
            pooled[k, :, :, 0:256] = _planted_sample(rng, L, H, 16, peak=(5, 3 + k))
            pooled[k, :, :, 256:512] = _planted_sample(rng, L, H, 16, planted=((2, 0),), peak=(10, 12))
        rel = f"attention/libero_spatial_t{task}_e{init}_attn.npz"
        np.savez_compressed(run / rel, pooled_f16=pooled)
        rows.append({"suite": "libero_spatial", "task_id": task, "episode_idx": init, "init_state_index": init,
                     "attn_npz": rel, "calls": calls})
    (run / "attention_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


# ---------------------------------------------------------------------------------------------- parity
@needs_ext
def test_pinned_checkout_sha():
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=EXT, capture_output=True, text=True).stdout.strip()
    assert sha == lh.SOURCE["sha"]


@needs_ext
def test_elbow_chord_matches_official_numpy():
    official = _official_function("elbow_chord")
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 17, 144):
        vals = list(rng.gamma(0.5, size=n))
        assert lh.elbow_chord(vals) == official(vals)
    assert lh.elbow_chord([0.3] * 10) == official([0.3] * 10)


@needs_ext
@needs_torch
def test_analyze_heads_matches_official_torch():
    analyze = _load_official("analyze.py", "_lh_official_analyze")
    cfg = SimpleNamespace(logic=SimpleNamespace(threshold=SimpleNamespace(method="chord", min_keep=1),
                                                entropy=SimpleNamespace(binarize_threshold=0.001)))
    rng = np.random.default_rng(1)
    distinct: set = set()
    for i in range(8):
        P = 16 if i % 2 else 8
        a = _multi_blob_sample(rng, L=8, H=4, P=P)
        ref = analyze.analyze_heads(cfg, torch.from_numpy(a)[:, :, None, :], {"patch_size": P})
        mine = lh.analyze_heads_code(a, P, lh.SelectionConfig())
        assert {(r["layer"], r["head"]) for r in mine} == {(r["layer"], r["head"]) for r in ref}
        by_head = {(r["layer"], r["head"]): r for r in ref}
        for m in mine:
            r = by_head[(m["layer"], m["head"])]
            assert m["bottom_row_focus"] == r["bottom_row_focus"]
            assert m["num_components"] == r["num_components"]
            assert np.isclose(m["spatial_entropy"], r["spatial_entropy"], rtol=1e-5, atol=1e-6)
        # same order once float32 rounding noise (|dSE| < 1e-6, e.g. single-component heads at SE ~ 0) is removed
        key = lambda r: (round(r["spatial_entropy"], 6), r["layer"], r["head"])  # noqa: E731
        assert [(r["layer"], r["head"]) for r in sorted(mine, key=key)] == \
               [(r["layer"], r["head"]) for r in sorted(ref, key=key)]
        distinct.update(round(r["spatial_entropy"], 6) for r in ref)
    assert len(distinct) >= 5  # the cases really rank heads by entropy (not only SE ~ 0 ties)


@needs_ext
@needs_torch
def test_combine_and_upscale_match_official_bbox():
    bbox = _load_official("bbox.py", "_lh_official_bbox")
    rng = np.random.default_rng(2)
    a = _planted_sample(rng, L=6, H=4, P=16)
    sel = [{"layer": 3, "head": 1}, {"layer": 4, "head": 2}, {"layer": 0, "head": 0}]
    ref = bbox.combine_heads(torch.from_numpy(a)[:, :, None, :], sel, P=16, sigma=1.0)
    mine = lh.combine_heads([a[s["layer"], s["head"]].reshape(16, 16) for s in sel], 1.0, 4.0)
    np.testing.assert_allclose(mine, ref, rtol=1e-6, atol=1e-9)
    np.testing.assert_array_equal(lh.binarize_mean_relu(mine).astype(np.uint8), bbox.binarize_mean_relu(ref))
    grid_mask = bbox.binarize_mean_relu(ref)
    for size in ((256, 256), (224, 224), (200, 320)):  # image_size is (W, H) in bbox.upscale_mask
        ref_up = bbox.upscale_mask(grid_mask, size)
        mine_up = lh.upscale_bilinear(grid_mask.astype(np.float32), (size[1], size[0])) > 0.5
        np.testing.assert_array_equal(mine_up, ref_up.astype(bool))
    cont = rng.random((16, 16)).astype(np.float32)
    ref_c = torch.nn.functional.interpolate(torch.from_numpy(cont)[None, None], size=(200, 320), mode="bilinear",
                                            align_corners=False)[0, 0].numpy()
    np.testing.assert_allclose(lh.upscale_bilinear(cont, (200, 320)), ref_c, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------------------------- selection
def test_select_heads_deterministic_and_finds_planted_heads():
    def source():
        rng = np.random.default_rng(3)
        return (_planted_sample(rng, peak=(2 + k % 3, 5)) for k in range(12))

    for protocol in ("official_code", "paper"):
        cfg = lh.SelectionConfig(protocol=protocol, top_k=2)
        s1, s2 = lh.select_heads(source, 8, cfg), lh.select_heads(source, 8, cfg)
        assert s1 == s2
        assert {(r["layer"], r["head"]) for r in s1["selected"]} == {(3, 1), (4, 2)}, protocol


def test_select_heads_excludes_first_two_layers():
    def source():
        rng = np.random.default_rng(4)
        return (_planted_sample(rng, planted=((0, 0), (1, 1), (5, 3))) for _ in range(5))

    sel = lh.select_heads(source, 8, lh.SelectionConfig(top_k=3))
    assert all(r["layer"] >= 2 for r in sel["selected"])
    assert sel["selected"][0]["layer"] == 5 and sel["selected"][0]["head"] == 3


def test_build_selection_dev_only_and_reproducible(tmp_path):
    rng = np.random.default_rng(5)
    run_a = _write_fake_run(tmp_path, "run_a", [(0, 0), (1, 0), (2, 7)], rng)  # init 7 = test split
    run_b = _write_fake_run(tmp_path, "run_b", [(0, 1), (1, 4)], rng)
    d1 = lh.build_selection([run_a, run_b])
    d2 = lh.build_selection([run_b, run_a])  # order of runs must not matter
    assert d1["selection_sha256"] == d2["selection_sha256"]
    assert d1["selection"] == d2["selection"]
    assert [e["init_state_index"] for e in d1["dev_split"]["excluded_non_dev"]] == [7]
    assert all(i["init_state_index"] in lh.DEV_INIT_STATES for i in d1["inputs"])
    assert d1["selection"]["head"]["n_samples"] == 4 * 3
    assert (2, 0) == tuple(d1["selection"]["left_wrist"]["selected"][0][k] for k in ("layer", "head"))
    # frozen json round trip + tamper detection
    path = tmp_path / "sel.json"
    path.write_text(json.dumps({**d1, "created_at": "2026-10-23T00:00:00+09:00"}))
    assert lh.load_selection(path)["selection_sha256"] == d1["selection_sha256"]
    bad = json.loads(path.read_text())
    bad["selection"]["head"]["selected"][0]["head"] += 1
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="selection_sha256"):
        lh.load_selection(path)


def test_cli_select_refuses_overwrite(tmp_path):
    rng = np.random.default_rng(6)
    run = _write_fake_run(tmp_path, "run", [(0, 0)], rng)
    out = tmp_path / "frozen.json"
    assert lh.main(["select", "--attention-run", str(run), "--out", str(out)]) == 0
    first = out.read_text()
    with pytest.raises(SystemExit):
        lh.main(["select", "--attention-run", str(run), "--out", str(out)])
    assert out.read_text() == first


# ---------------------------------------------------------------------------------------------- provider
def _selection_doc(tmp_path):
    rng = np.random.default_rng(7)
    run = _write_fake_run(tmp_path, "run", [(0, 0), (1, 1)], rng)
    return lh.build_selection([run], cameras=("head",))


def _query():
    iface = _interface_or_skip()
    return iface.TargetQuery(instruction="pick up the black bowl", target_phrase="black bowl")


def test_threshold_not_found_on_constant_attention(tmp_path):
    query = _query()
    prov = lh.LocalizationHeadsProvider(_selection_doc(tmp_path), mask_mode="threshold")
    pooled = np.full((6, 4, 768), 1.0 / 768, np.float32)
    rgb = {"head": np.zeros((256, 256, 3), np.uint8), "left_wrist": np.zeros((128, 160, 3), np.uint8)}
    res = prov.predict(rgb, query, aux={"attention_pooled": pooled, "attention_orientation": "image"})
    for cam, r in res.items():
        assert r.found is False and r.score is None and r.boxes is None
        assert r.mask.dtype == bool and r.mask.shape == rgb[cam].shape[:2] and not r.mask.any()
        assert "not_found_reason" in r.extras
    assert "constant" in res["head"].extras["not_found_reason"]
    assert "no frozen heads" in res["left_wrist"].extras["not_found_reason"]


def test_threshold_found_and_libero_mirror(tmp_path):
    query = _query()
    doc = _selection_doc(tmp_path)
    prov = lh.LocalizationHeadsProvider(doc, mask_mode="threshold")
    rng = np.random.default_rng(8)
    pooled = np.zeros((6, 4, 768), np.float32)
    pooled[:, :, 0:256] = _planted_sample(rng, P=16, peak=(4, 2))  # policy column 2 (left in policy image)
    rgb = {"head": np.zeros((256, 256, 3), np.uint8)}
    r_img = prov.predict(rgb, query, aux={"attention_pooled": pooled, "attention_orientation": "image"})["head"]
    r_pol = prov.predict(rgb, query, aux={"attention_pooled": pooled, "attention_orientation": "libero_policy"})["head"]
    assert r_img.found and r_pol.found
    np.testing.assert_array_equal(r_pol.mask, r_img.mask[:, ::-1])  # un-mirror = column reversal at 256 px
    assert r_img.extras["peak_xy"][0] < 64 and r_pol.extras["peak_xy"][0] > 192
    with pytest.raises(ValueError, match="attention_orientation"):
        prov.predict(rgb, query, aux={"attention_pooled": pooled})


def test_sam_modes_need_segmenter(tmp_path, monkeypatch):
    query = _query()
    doc = _selection_doc(tmp_path)
    rng = np.random.default_rng(9)
    pooled = np.zeros((6, 4, 768), np.float32)
    pooled[:, :, 0:256] = _planted_sample(rng, P=16, peak=(8, 8))
    aux = {"attention_pooled": pooled, "attention_orientation": "image"}
    rgb = {"head": np.zeros((64, 64, 3), np.uint8)}
    # no SAM object and no importable sam_backend -> explicit error, never a silent threshold fallback
    monkeypatch.setitem(sys.modules, "Baseline.paper_baselines.providers.sam_backend", None)
    with pytest.raises(NotImplementedError):
        lh.LocalizationHeadsProvider(doc, mask_mode="sam_point").predict(rgb, query, aux=aux)

    class FakeSamV1:  # the sam_backend.SamV1Backend call shape (Implementer A)
        def __init__(self):
            self.calls = []

        def _mask(self, rgb):
            m = np.zeros(rgb.shape[:2], bool)
            m[10:20, 10:20] = True
            return m

        def predict_points(self, rgb, points_xy, labels):
            self.calls.append(("points", np.asarray(points_xy).tolist(), np.asarray(labels).tolist()))
            return self._mask(rgb), 0.9

        def predict_box(self, rgb, box_xyxy):
            self.calls.append(("box", list(np.asarray(box_xyxy).tolist())))
            return self._mask(rgb), 0.8

    sam = FakeSamV1()
    r = lh.LocalizationHeadsProvider(doc, mask_mode="sam_point", sam=sam).predict(rgb, query, aux=aux)["head"]
    assert r.found and r.extras["sam_score"] == 0.9
    assert sam.calls[0] == ("points", [r.extras["peak_xy"]], [1])
    r = lh.LocalizationHeadsProvider(doc, mask_mode="sam_box", sam=sam).predict(rgb, query, aux=aux)["head"]
    assert r.found and r.extras["sam_score"] == 0.8 and sam.calls[1][0] == "box"
    assert sam.calls[1][1] == r.extras["prompt_box_xyxy"]


def test_mirror_rule_agrees_with_lift_policy_mask(tmp_path):
    """The grid-level un-mirror used here == lift.policy_mask_to_opencv applied to the policy-frame patch mask."""
    try:
        from Baseline.paper_baselines import lift
    except ImportError as exc:
        pytest.skip(f"lift.py not importable: {exc}")
    if not hasattr(lift, "policy_mask_to_opencv"):
        pytest.skip("lift.policy_mask_to_opencv missing")
    rng = np.random.default_rng(14)
    for _ in range(5):
        g = rng.random((16, 16)) > 0.7                        # policy-orientation patch mask
        policy224 = np.repeat(np.repeat(g, 14, 0), 14, 1)     # patch (r, c) = policy pixels [14r, 14r+14) x [14c, ..)
        via_lift = lift.policy_mask_to_opencv(policy224, (256, 256))
        mine = np.repeat(np.repeat(lh.to_image_orientation(g, "libero_policy"), 16, 0), 16, 1)
        np.testing.assert_array_equal(via_lift, mine)


def test_results_satisfy_interface_contract(tmp_path):
    iface = _interface_or_skip()
    query = _query()
    prov = lh.LocalizationHeadsProvider(_selection_doc(tmp_path))
    assert isinstance(prov, iface.TargetMaskProvider)
    rng = np.random.default_rng(15)
    pooled = np.zeros((6, 4, 768), np.float32)
    pooled[:, :, 0:256] = _planted_sample(rng, P=16, peak=(3, 12))
    rgb = {"head": np.zeros((256, 256, 3), np.uint8), "left_wrist": np.zeros((256, 256, 3), np.uint8)}
    res = prov.predict(rgb, query, aux={"attention_pooled": pooled, "attention_orientation": "libero_policy"})
    iface.validate_results(res, rgb)
    assert res["head"].found and not res["left_wrist"].found
    with pytest.raises(ValueError):
        prov.predict({"head": np.zeros((8, 8), np.float32)}, query,
                     aux={"attention_pooled": pooled, "attention_orientation": "image"})
