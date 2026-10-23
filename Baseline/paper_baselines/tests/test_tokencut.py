"""TokenCut re-implementation (Implementer B): parity with the pinned official ``object_discovery.ncut``.

The parity tests import the official function from ``external/TokenCut`` (pinned SHA, checked) and need torch
(``.venv-baselines``). The numpy-only tests (planted object, ``attention_overlap`` target rule, not-found path)
run in ``.venv-ag3s``.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from Baseline.paper_baselines.providers import tokencut as tc

EXT = Path(os.environ.get("PAPER_BASELINES_TOKENCUT_DIR", tc.EXTERNAL_DIR))
try:
    import torch
except ImportError:  # .venv-ag3s
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="torch not importable (run in .venv-baselines)")
needs_ext = pytest.mark.skipif(not (EXT / "object_discovery.py").exists(), reason=f"{EXT} not fetched")


def _interface_or_skip():
    try:
        from Baseline.paper_baselines import interface
    except ImportError as exc:
        pytest.skip(f"interface.py not importable: {exc}")
    return interface


def _object_features(rng, dims=(16, 16), D=64, boxes=(((4, 9), (3, 8)),), noise=0.35):
    """Background direction + one direction per planted box, with noise. Row-major (rows*cols, D)."""
    rows, cols = dims
    basis = np.linalg.qr(rng.normal(size=(D, 1 + len(boxes))))[0].T
    lab = np.zeros(dims, int)
    for i, ((r0, r1), (c0, c1)) in enumerate(boxes, start=1):
        lab[r0:r1, c0:c1] = i
    f = basis[lab.ravel()] + noise * rng.normal(size=(rows * cols, D)) / np.sqrt(D) * 4
    return f.astype(np.float32), lab


def _official_ncut():
    return tc.load_official_module("object_discovery.py", "_tokencut_official_od", EXT).ncut


# ---------------------------------------------------------------------------------------------- parity
@needs_ext
def test_pinned_checkout_sha():
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=EXT, capture_output=True, text=True).stdout.strip()
    assert sha == tc.SOURCE["sha"]


CASES = [  # (seed, dims, D, kind, binary_graph)
    (0, (16, 16), 384, "random", True),
    (1, (16, 16), 384, "random", True),
    (2, (14, 20), 64, "random", True),
    (3, (16, 16), 64, "objects", True),
    (4, (12, 18), 384, "objects", True),
    (5, (16, 16), 64, "objects", False),
    (6, (16, 16), 384, "random", False),
]


@needs_ext
@needs_torch
@pytest.mark.parametrize("seed,dims,D,kind,binary_graph", CASES)
def test_ncut_parity_with_official(seed, dims, D, kind, binary_graph):
    ncut_official = _official_ncut()
    rng = np.random.default_rng(seed)
    if kind == "random":
        # low-rank random features so the binarised graph is neither empty nor complete
        feats = (rng.normal(size=(dims[0] * dims[1], 6)) @ rng.normal(size=(6, D))).astype(np.float32)
    else:
        feats, _ = _object_features(rng, dims, D, boxes=(((2, 7), (3, 9)), ((8, 11), (10, 15))))
    cls = rng.normal(size=(1, D)).astype(np.float32)
    t = torch.from_numpy(np.concatenate([cls, feats])[None])  # (1, 1 + N, D), CLS first like main_tokencut
    init_image_size = (3, dims[0] * 16, dims[1] * 16)
    pred, objects, mask, seed_o, _, eigvec = ncut_official(t, list(dims), [16, 16], init_image_size,
                                                          tau=0.2, eps=1e-5, no_binary_graph=not binary_graph)
    mine = tc.ncut(feats, dims, tau=0.2, eps=1e-5, binary_graph=binary_graph)
    assert mine.seed == int(seed_o)
    np.testing.assert_array_equal(mine.mask, mask.astype(bool))
    np.testing.assert_array_equal(mine.labels, objects)
    # binary graph: A is exactly {1, eps} on both sides -> same matrices into the same scipy eigh -> same vector.
    # similarity graph (ablation): A keeps float32 cosine values, and numpy vs torch float32 matmul differ in the
    # last bits, so the eigenvector agrees to ~1e-7 (mask, seed and labels are still identical, asserted above).
    np.testing.assert_allclose(mine.eigvec, eigvec, rtol=0, atol=1e-10 if binary_graph else 1e-6)
    ys, xs = np.nonzero(mine.mask)  # official detect_box on the same component
    assert [xs.min() * 16, ys.min() * 16, (xs.max() + 1) * 16, (ys.max() + 1) * 16] == [float(v) for v in pred]


@needs_torch
def test_nearest_resize_matches_torch():
    rng = np.random.default_rng(10)
    g = rng.random((16, 16)) > 0.5
    for size in ((256, 256), (224, 224), (200, 333)):
        ref = torch.nn.functional.interpolate(torch.from_numpy(g.astype(np.float32))[None, None], size=size,
                                              mode="nearest")[0, 0].numpy().astype(bool)
        np.testing.assert_array_equal(tc.nearest_resize(g, size), ref)


# ---------------------------------------------------------------------------------------------- numpy-only
def test_ncut_recovers_planted_object():
    rng = np.random.default_rng(11)
    feats, lab = _object_features(rng)
    res = tc.ncut(feats, (16, 16))
    obj = lab == 1
    # the foreground (seed side) component is the planted box
    np.testing.assert_array_equal(res.mask, obj)
    assert res.mask[np.unravel_index(res.seed, (16, 16))]


def test_ncut_rejects_bad_input():
    with pytest.raises(ValueError):
        tc.ncut(np.zeros((10, 4), np.float32), (4, 4))
    bad = np.ones((16, 4), np.float32)
    bad[3, 1] = np.nan
    with pytest.raises(ValueError):
        tc.ncut(bad, (4, 4))


def _two_object_aux(rng, att_peak_rc, orientation="image"):
    feats, lab = _object_features(rng, boxes=(((2, 7), (2, 7)), ((9, 13), (9, 14))))
    att = np.zeros((16, 16), np.float32)
    att[att_peak_rc[0] - 1:att_peak_rc[0] + 2, att_peak_rc[1] - 1:att_peak_rc[1] + 2] = 1.0
    if orientation == "libero_policy":
        att = att[:, ::-1].copy()
    return {"patch_features": {"head": feats.reshape(16, 16, -1)}, "patch_features_orientation": "image",
            "attention_map": {"head": att}, "attention_orientation": orientation}, lab


@pytest.mark.parametrize("orientation", ["image", "libero_policy"])
def test_attention_overlap_picks_attended_component(orientation):
    iface = _interface_or_skip()
    query = iface.TargetQuery(instruction="pick up the bowl", target_phrase="bowl")
    rng = np.random.default_rng(12)
    rgb = {"head": np.zeros((256, 256, 3), np.uint8)}
    for peak, label_id in (((4, 4), 1), ((11, 11), 2)):
        aux, lab = _two_object_aux(rng, peak, orientation)
        r = tc.TokenCutProvider(feature_source="aux", target_select="attention_overlap").predict(rgb, query,
                                                                                             aux=aux)["head"]
        assert r.found and 0.0 < r.score <= 1.0
        expected = np.repeat(np.repeat(lab == label_id, 16, 0), 16, 1)
        np.testing.assert_array_equal(r.mask, expected)
        assert r.extras["chosen"]["overlap_mass"] == max(c["overlap_mass"] for c in r.extras["candidates"])
    # foreground mode ignores attention: same features -> same mask whatever the attention says
    fg = [tc.TokenCutProvider(feature_source="aux").predict(rgb, query, aux=_two_object_aux(
        np.random.default_rng(12), p)[0])["head"].mask for p in ((4, 4), (11, 11))]
    np.testing.assert_array_equal(fg[0], fg[1])


def test_not_found_paths():
    iface = _interface_or_skip()
    query = iface.TargetQuery(instruction="pick up the bowl", target_phrase="bowl")
    rng = np.random.default_rng(13)
    aux, _ = _two_object_aux(rng, (4, 4))
    rgb = {"head": np.zeros((256, 256, 3), np.uint8), "left_wrist": np.zeros((128, 128, 3), np.uint8)}
    aux["attention_map"]["head"] = np.full((16, 16), 0.3, np.float32)  # constant attention
    prov = tc.TokenCutProvider(feature_source="aux", target_select="attention_overlap")
    assert isinstance(prov, iface.TargetMaskProvider)
    res = prov.predict(rgb, query, aux=aux)
    iface.validate_results(res, rgb)
    assert not res["head"].found and "constant" in res["head"].extras["not_found_reason"]
    assert not res["left_wrist"].found and "no aux patch features" in res["left_wrist"].extras["not_found_reason"]
    for cam, r in res.items():
        assert r.mask.shape == rgb[cam].shape[:2] and not r.mask.any() and r.boxes is None and r.score is None
    del aux["attention_map"]
    r = tc.TokenCutProvider(feature_source="aux", target_select="attention_overlap").predict(
        {"head": rgb["head"]}, query, aux=aux)["head"]
    assert not r.found and "attention_map" in r.extras["not_found_reason"]


def test_provenance_flags_non_original_rule():
    p = tc.TokenCutProvider(feature_source="aux", target_select="attention_overlap").provenance()
    assert p["target_select_is_original_method"] is False and p["uses_query_text"] is False
    p = tc.TokenCutProvider(feature_source="dino_vits16").provenance()
    assert p["target_select_is_original_method"] is True and p["source"]["sha"] == tc.SOURCE["sha"]
