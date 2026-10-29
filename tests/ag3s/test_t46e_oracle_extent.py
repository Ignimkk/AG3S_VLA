"""T46e A13 — visible-target oracle extent (`esdf.target_extent = "oracle_visible"`, 기본 off). off 는 비트 동일이어야 한다.

## 무엇을 박나

| # | 성질 |
|---|---|
| 1 | config 기본값 · 왕복 · validation (F1 · F2 와 함께 거절), CLI (`--target-extent` · `--oracle-extent-identity` · `--oracle-max-dist-voxels`) 왕복 · 거절 |
| 2 | code 표 (`code_table` · `label_image`): target 1 · 정적 2 · 다른 과일 ≥ 3 · 로봇 / 안 맞음 0, 자식 body 는 root 의 code |
| 3 | `backproject_labels` = pinhole · `OracleMemory`: 새 관측이 덮는다 · ROI · 너머를 본 표본은 지운다 · reset · `instance_of` 다수결 |
| 4 | `oracle_visible_split` (합성 rig, 광선 추적한 두 카메라): **target 의 보인 표면 seed 만** 지우고 닿은 이웃 과일 · table seed 는 0 |
| 5 | `_build_tiers` (가짜 cuRobo): oracle 은 이웃 · table 표면을 남기고 target 표면을 지운다 · 기록 키 · `_exclusion_counts` |
| 6 | pipeline: 기본은 segmentation 을 무시하고 builder 호출 · stats 가 예전과 같다 · oracle 이면 identity 가 가리키는 물체의 표본을 넘긴다 (`pipeline` / `gt`) · 없으면 추정 범위로 짓고 이유를 남긴다 · reset |
| 7 | wire (옛 client = 키 없음 = off) · client (`oracle_target` 일 때만 code 영상) · 서버 metadata 짝 검사 |
| 8 | 기록 chunk (storage): `_replay_layer` 로 옮긴 뒤에도 off replay = 기록 (float16 비트), oracle replay 는 이웃 과일 seed 를 지우지 않는다 |
| 9 | GPU (실제 cuRobo, CUDA 있을 때): 기본 = `oracle_evidence=None` 비트 동일 · oracle 은 미세 target-free 계층만 바꾼다 |
"""

from __future__ import annotations

import json
import os
import pathlib
import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.fields.curobo_builder import OracleVisibleEvidence, oracle_candidates, oracle_visible_split
from AG3S.ag3s.runtime.oracle_segmentation import (
    CODE_NONE,
    CODE_OBJECT_BASE,
    CODE_STATIC,
    CODE_TARGET,
    OracleMemory,
    OracleView,
    backproject_labels,
    code_table,
    label_image,
    oracle_evidence,
)

VS = 0.005


# ============================================================================ 1 config · CLI


def test_config_defaults_round_trip_and_validation():
    d = AG3SConfig().esdf
    assert (d.target_extent, d.oracle_max_dist_voxels, d.oracle_extent_identity) == ("estimated", 6.0, "pipeline")
    cfg = AG3SConfig.from_dict({"esdf": {"target_exclusion": "ball_owned", "target_extent": "oracle_visible",
                                         "oracle_max_dist_voxels": 4.0, "oracle_extent_identity": "gt"}})
    again = AG3SConfig.from_dict(cfg.to_dict())
    assert again.esdf == cfg.esdf
    assert (again.esdf.target_extent, again.esdf.oracle_max_dist_voxels, again.esdf.oracle_extent_identity) == \
        ("oracle_visible", 4.0, "gt")
    for bad in ({"target_extent": "oracle"}, {"oracle_max_dist_voxels": 0.0}, {"oracle_max_dist_voxels": -1.0},
                {"oracle_max_dist_voxels": float("nan")}, {"oracle_max_dist_voxels": float("inf")},
                {"oracle_max_dist_voxels": True}, {"oracle_extent_identity": "attention"}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig.from_dict({"esdf": bad})


def test_config_rejects_oracle_with_f1_or_f2():
    """F1 · F2 change only the `ball_owned` fine split, which the oracle replaces — a no-op combination."""
    for f in ({"nontarget_evidence": "surface"}, {"owned_ambiguous": "remove"}):
        with pytest.raises(AG3SConfigError, match="oracle_visible"):
            AG3SConfig.from_dict({"esdf": {"target_exclusion": "ball_owned", "target_extent": "oracle_visible", **f}})
        AG3SConfig.from_dict({"esdf": {"target_exclusion": "ball_owned", **f}})   # without the oracle: fine


def _parse(*argv):
    from AG3S.trajopt.serve_safe import build_parser

    return build_parser().parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml", *argv])


def test_cli_defaults_leave_the_config_untouched():
    from AG3S.trajopt.serve_safe import target_exclusion_section

    for argv, want in (((), {}),
                       (("--target-exclusion", "ball-owned", "--owned-max-dist-voxels", "3"),
                        {"target_exclusion": "ball_owned", "owned_max_dist_voxels": 3.0}),
                       (("--target-extent", "estimated"), {})):
        args = _parse(*argv)
        assert target_exclusion_section(args) == want
    args = _parse()
    assert (args.target_extent, args.oracle_extent_identity, args.oracle_max_dist_voxels) == (None, None, None)
    cfg = AG3SConfig.from_dict({"esdf": target_exclusion_section(args)}).esdf
    assert cfg.target_extent == "estimated"


def test_cli_oracle_flags_map_to_esdf_keys_and_back():
    from AG3S.trajopt.serve_safe import target_exclusion_section

    args = _parse("--target-exclusion", "ball-owned", "--target-extent", "oracle-visible",
                  "--oracle-extent-identity", "gt", "--oracle-max-dist-voxels", "5")
    sec = target_exclusion_section(args)
    assert sec == {"target_exclusion": "ball_owned", "target_extent": "oracle_visible",
                   "oracle_extent_identity": "gt", "oracle_max_dist_voxels": 5.0}
    cfg = AG3SConfig.from_dict({"esdf": sec}).esdf
    assert {k: getattr(cfg, k) for k in sec} == sec          # what `build_ag3s` checks survived
    # the oracle on the C0 base (`ball`) too
    sec = target_exclusion_section(_parse("--target-extent", "oracle-visible"))
    assert sec == {"target_extent": "oracle_visible"}
    assert AG3SConfig.from_dict({"esdf": sec}).esdf.target_exclusion == "ball"


def _main_errors(monkeypatch, *argv):
    import sys

    from AG3S.trajopt import serve_safe

    monkeypatch.setattr(sys, "argv", ["serve_safe", "--config", "c", "--checkpoint", "k",
                                      "--model-xml", "x.xml", *argv])
    with pytest.raises(SystemExit) as e:
        serve_safe.main()
    return e.value.code


def test_cli_rejects_ineffective_oracle_combinations(monkeypatch, capsys):
    for flags in (("--oracle-extent-identity", "gt"), ("--oracle-max-dist-voxels", "4")):
        assert _main_errors(monkeypatch, *flags) == 2
        assert "oracle-visible" in capsys.readouterr().err
    base = ("--esdf-backend", "curobo", "--target-exclusion", "ball-owned", "--target-extent", "oracle-visible")
    assert _main_errors(monkeypatch, *base) == 2                                # relax policy
    assert "--target-field-policy" in capsys.readouterr().err
    on = (*base, "--target-field-policy", "exclude-authorized")
    for f in (("--nontarget-evidence", "surface"), ("--owned-ambiguous", "remove")):
        assert _main_errors(monkeypatch, *on, *f) == 2
        assert "F1" in capsys.readouterr().err
    for bad in ("0", "-1", "nan", "inf"):
        assert _main_errors(monkeypatch, *on, "--oracle-max-dist-voxels", bad) == 2
        assert "유한한 양수" in capsys.readouterr().err


# ============================================================================ 2 code table


def test_code_table_and_label_image():
    names = {0: "world", 1: "table", 2: "crate", 3: "apple", 4: "orange", 5: "pear", 6: "banana",
             7: "link_left_arm_6", 8: "ee_finger_l1", 9: "orange_child"}
    root = {i: i for i in names} | {9: 4}
    lut, codes = code_table(names, "orange", root_of=root)
    assert codes == {"orange": 1, "apple": 3, "banana": 4, "pear": 5}
    assert lut[0] == CODE_NONE                                                   # no hit (-1)
    want = {0: CODE_STATIC, 1: CODE_STATIC, 2: CODE_STATIC, 3: 3, 4: CODE_TARGET, 5: 5, 6: 4,
            7: CODE_NONE, 8: CODE_NONE, 9: CODE_TARGET}
    for bid, code in want.items():
        assert lut[bid + 1] == code, names[bid]
    img = np.array([[-1, 4, 9], [7, 1, 3], [99, -5, 6]])
    lab = label_image(img, lut)
    assert lab.dtype == np.uint8
    assert lab.tolist() == [[0, 1, 1], [0, 2, 3], [0, 0, 4]]
    with pytest.raises(ValueError, match="graspable"):
        code_table(names, "kiwi")


# ============================================================================ 3 rig: two touching fruits on a table

A = np.array([0.50, 0.30, 0.861])         # target centre
B = np.array([0.50, 0.363, 0.861])        # neighbour, 1 mm gap
R = 0.031
TABLE_Z = 0.830


def _look_at(eye, target):
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def _raycast(T, *, hw=(96, 96), f=110.0, spheres=((A, R, CODE_TARGET), (B, R, CODE_OBJECT_BASE)), robot=None,
             table_z=TABLE_Z):
    """Analytic depth (camera z, m) + code image: spheres, the table plane (code 2) and an optional robot sphere (0)."""
    h, w = hw
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
    v, u = np.mgrid[0:h, 0:w]
    dc = np.stack([(u - K[0, 2]) / f, (v - K[1, 2]) / f, np.ones_like(u, float)], -1).reshape(-1, 3)
    dw = dc @ T[:3, :3].T
    o = T[:3, 3]
    best = np.full(len(dw), np.inf)
    code = np.zeros(len(dw), np.uint8)
    hits = list(spheres) + ([] if robot is None else [(robot[0], robot[1], CODE_NONE)])
    for c, r, cd in hits:
        oc = o - c
        b = dw @ oc
        a = np.einsum("ij,ij->i", dw, dw)
        disc = b * b - a * (oc @ oc - r * r)
        ok = disc >= 0
        s = np.where(ok, (-b - np.sqrt(np.maximum(disc, 0))) / a, np.inf)
        s = np.where(s > 0, s, np.inf)
        better = s < best
        best[better], code[better] = s[better], cd
    s = (table_z - o[2]) / dw[:, 2]
    s = np.where(s > 0, s, np.inf)
    better = s < best
    best[better], code[better] = s[better], CODE_STATIC
    depth = np.where(np.isfinite(best), best, 0.0).reshape(h, w)          # s = camera z (dc has z = 1)
    return OracleView(depth_m=depth, K=K, T_base_cam=T, labels=code.reshape(h, w), name="cam")


def _views(robot=None):
    return [_raycast(_look_at(A + np.array([0.0, 0.0, 0.45]), A + np.array([0.0, 0.03, 0.0])), robot=robot),
            _raycast(_look_at(A + np.array([-0.35, -0.10, 0.30]), A))]


def _sdf(p):
    da = np.linalg.norm(p - A, axis=-1) - R
    db = np.linalg.norm(p - B, axis=-1) - R
    dt = p[..., 2] - TABLE_Z
    st = np.stack([da, db, dt], -1)
    return st.min(-1), st.argmin(-1)


def test_backproject_is_the_pinhole():
    view = _views()[0]
    p, c = backproject_labels(view.depth_m, view.K, view.T_base_cam, view.labels)
    assert len(p) == int((view.labels != 0).sum())
    sdf, owner = _sdf(p)
    assert np.abs(sdf).max() < 1e-6, "every back-projected pixel lies on the surface it was cast against"
    assert np.array_equal(np.where(owner == 0, CODE_TARGET, np.where(owner == 1, CODE_OBJECT_BASE, CODE_STATIC)), c)
    p2, _ = backproject_labels(view.depth_m, view.K, view.T_base_cam, view.labels, depth_min=0.0, depth_max=0.44)
    assert 0 < len(p2) < len(p)


def test_memory_newest_wins_roi_carving_and_reset():
    mem = OracleMemory(cell=0.0025, roi_margin=0.065, carve_tol=0.010)
    st = mem.update(_views(), legend={"orange": 1, "apple": 3})
    assert st["n_target_samples"] > 200 and len(mem) > st["n_target_samples"]
    tp, op = mem.evidence(CODE_TARGET)
    assert np.abs(np.linalg.norm(tp - A, axis=1) - R).max() < 1e-6
    # ROI: nothing farther than the margin from the fruits
    lo = np.minimum(A, B) - R - 0.065 - 1e-9
    hi = np.maximum(A, B) + R + 0.065 + 1e-9
    assert np.all((mem.points >= lo) & (mem.points <= hi))
    # one sample per cell
    assert len(np.unique(np.floor(mem.points / mem.cell).astype(np.int64), axis=0)) == len(mem)
    # the same views again: nothing carved, nothing new
    n = len(mem)
    st2 = mem.update(_views())
    assert st2["n_carved"] == 0 and len(mem) == n
    # the hand (a robot sphere, code 0) hides the target from the overhead camera: memory keeps the samples
    hand = (A + np.array([0.0, 0.0, 0.08]), 0.04)
    st3 = mem.update(_views(robot=hand))
    assert st3["n_carved"] == 0 and st3["n_target_samples"] == st["n_target_samples"]
    # the target moved away (+15 cm): the cameras see the table through its old samples → carved
    moved = [_raycast(v.T_base_cam, spheres=((A + [0.15, 0, 0], R, CODE_TARGET), (B, R, CODE_OBJECT_BASE)))
             for v in _views()]
    st4 = mem.update(moved)
    assert st4["n_carved"] > 0
    tp4, _ = mem.evidence(CODE_TARGET)
    old = np.abs(np.linalg.norm(tp4 - A, axis=1) - R) < 1e-3
    # what stays is only what no camera can see through now (the side facing the neighbour, behind it)
    assert old.sum() < 0.15 * st["n_target_samples"], (old.sum(), st["n_target_samples"])
    assert np.all(tp4[old][:, 1] > A[1]), "the samples left at the old place face the neighbour"
    assert np.all(np.linalg.norm(tp4[~old] - (A + [0.15, 0, 0]), axis=1) < R + 1e-3)
    mem.reset()
    assert len(mem) == 0 and mem.n_updates == 0


def test_instance_of_takes_the_majority_object():
    mem = OracleMemory()
    mem.update(_views(), legend={"orange": 1, "apple": 3})
    tp, _ = mem.evidence(CODE_TARGET)
    bp, _ = mem.evidence(CODE_OBJECT_BASE)
    code, info = mem.instance_of(tp[::5], radius=0.01)
    assert code == CODE_TARGET and info["fraction"] == 1.0
    code, _ = mem.instance_of(np.vstack([bp[::5], tp[:3]]), radius=0.01)
    assert code == CODE_OBJECT_BASE
    code, info = mem.instance_of(np.array([[2.0, 2.0, 2.0]]), radius=0.01)
    assert code is None and info["why"] == "no_object_sample"
    mixed = np.vstack([tp[:10], bp[:10]])
    code, info = mem.instance_of(mixed, radius=0.01)
    assert code in (CODE_TARGET, CODE_OBJECT_BASE) and info["fraction"] == 0.5
    code, info = mem.instance_of(mixed, radius=0.01, min_fraction=0.6)
    assert code is None and info["why"] == "no_majority"


# ============================================================================ 4 the split (pure numpy)


def _seed_grid():
    lo = np.minimum(A, B) - 0.06
    n = (np.ceil((np.maximum(A, B) + 0.06 - lo) / VS)).astype(int) + 1
    ax = [lo[i] + np.arange(n[i]) * VS for i in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)
    sdf, owner = _sdf(g)
    return g, lo, tuple(n), np.abs(sdf) < 0.75 * VS, owner


def test_oracle_split_removes_only_the_visible_target_seeds():
    mem = OracleMemory()
    mem.update(_views())
    tp, op = mem.evidence(CODE_TARGET)
    ev = OracleVisibleEvidence(target_points=tp, other_points=op, max_dist_voxels=6.0)
    g, lo, shape, seed_grid, owner_grid = _seed_grid()
    cand = oracle_candidates(tp, lo, VS, shape, 6.0 * VS)
    c = lo + cand * VS
    seed = seed_grid[cand[:, 0], cand[:, 1], cand[:, 2]]
    owner = owner_grid[cand[:, 0], cand[:, 1], cand[:, 2]]
    split = oracle_visible_split(c, seed, ev, VS)
    rm = split["removed"]
    assert not np.any(rm & ~seed)
    assert int((rm & (owner == 1)).sum()) == 0, "a seed of the touching neighbour was removed"
    assert int((rm & (owner == 2)).sum()) == 0, "a table seed was removed"
    # the target seeds facing the cameras (above the equator) are all removed
    vis = seed & (owner == 0) & (c[:, 2] > A[2]) & (c[:, 1] < A[1] + 0.5 * R)
    assert vis.sum() > 50 and rm[vis].mean() >= 0.98, rm[vis].mean()
    assert np.all(split["sign"][rm]), "sign ⊇ removed"
    assert not np.any(split["sign"] & (owner == 1) & (np.linalg.norm(c - B, axis=1) < R)), \
        "no sign correction inside the neighbour"
    # candidates reach: every candidate is within max_dist of a target sample, none beyond
    assert np.all(split["d_t"] <= 6.0 * VS + 1e-12)
    # tie → kept (the conservative side)
    tie = OracleVisibleEvidence(target_points=np.array([[0.0, 0.0, 0.0]]), other_points=np.array([[0.01, 0.0, 0.0]]))
    s2 = oracle_visible_split(np.array([[0.005, 0.0, 0.0], [0.004, 0.0, 0.0]]), np.array([True, True]), tie, VS)
    assert s2["removed"].tolist() == [False, True]


def test_band_memory_lies_behind_the_object_pixels_and_keys_match_the_builder():
    from AG3S.ag3s.fields.curobo_builder import oracle_band_masks, oracle_cell_keys
    from AG3S.ag3s.runtime.oracle_segmentation import _keys

    mem = OracleMemory(band_depth=0.030, band_stride=1)
    st = mem.update(_views())
    assert st["n_band_cells"] > 0 and st["n_band_points_this_frame"] > 0
    pts = np.random.default_rng(0).uniform(-1, 1, (50, 3))
    assert np.array_equal(oracle_cell_keys(pts, mem.cell), _keys(np.floor(pts / mem.cell).astype(np.int64)))
    bk, bm, cell = mem.band_lookup()
    assert np.all(np.diff(bk) > 0), "sorted, unique"
    # cells of the target's band are inside the target (behind its visible surface) or just behind its rim
    t_bit, b_bit = np.uint32(1 << CODE_TARGET), np.uint32(1 << CODE_OBJECT_BASE)
    probe_in_a = A + np.array([0.0, -0.005, 0.012])          # inside the target, ~19 mm under its visible top
    probe_in_b = B + np.array([0.0, 0.005, 0.012])
    m = oracle_band_masks(np.vstack([probe_in_a, probe_in_b, A + [0, 0, 0.2]]), bk, bm, cell)
    assert m[0] & t_bit and not m[0] & b_bit
    assert m[1] & b_bit and not m[1] & t_bit
    assert m[2] == 0, "free space above is no band"
    assert mem.band_depth == 0.030
    mem.reset()
    assert len(mem.band_keys) == 0
    assert OracleMemory(band_depth=0.0).update(_views())["n_band_cells"] == 0


def test_band_path_takes_the_shadow_seeds_and_respects_the_guards():
    """A seed behind the target's pixels, far from every sample (the shadow under / inside the fruit), goes by the band
    path; the same seed in a cell both objects' bands pass stays; near another sample or below the floor stays."""
    from AG3S.ag3s.fields.curobo_builder import oracle_band_masks, oracle_cell_keys

    mem = OracleMemory(band_depth=0.030, band_stride=1)
    views = _views()
    mem.update(views)
    tp, op = mem.evidence(CODE_TARGET)
    bk, bm, cell = mem.band_lookup()
    # the shadow: 20 mm behind the target's visible surface along the oblique camera's ray through the target centre
    v = views[1]
    pc = (A - v.T_base_cam[:3, 3]) @ v.T_base_cam[:3, :3]
    u, w = (int(np.floor(v.K[0, 0] * pc[0] / pc[2] + v.K[0, 2] + 0.5)),
            int(np.floor(v.K[1, 1] * pc[1] / pc[2] + v.K[1, 2] + 0.5)))
    assert v.labels[w, u] == CODE_TARGET
    zc = v.depth_m[w, u] + 0.020
    ray = np.array([(u - v.K[0, 2]) / v.K[0, 0], (w - v.K[1, 2]) / v.K[1, 1], 1.0]) * zc
    shadow = v.T_base_cam[:3, :3] @ ray + v.T_base_cam[:3, 3]
    assert np.linalg.norm(shadow - A) < R
    # a stray 'other' sample 8 mm from it: the surface path now says 'other' (d_o < d_t), the band path still takes it
    stray = shadow + np.array([0.0, 0.0, -0.008])
    op2 = np.vstack([op, stray])
    deep_b = B + np.array([0.0, 0.004, 0.012])                 # inside the neighbour, in its band
    floor = np.array([A[0], A[1] - 0.02, TABLE_Z + 0.003])     # just above the table under the target's side
    c = np.vstack([shadow, deep_b, floor])
    seed = np.ones(3, bool)
    ev = OracleVisibleEvidence(target_points=tp, other_points=op2, code=CODE_TARGET, band_cell=cell, band_keys=bk,
                               band_masks=bm, z_floor=TABLE_Z + 0.005)
    sp = oracle_visible_split(c, seed, ev, VS)
    assert not sp["surface"][0] and sp["band"][0] and sp["removed"][0], "the shadow seed goes by the band path"
    assert not sp["removed"][1], "the neighbour's inside is not the target's"
    assert not sp["removed"][2], "below the support floor"
    assert not sp["sign"][1] and not sp["sign"][2]
    # the stray sample within the protect distance (5 mm) → kept
    ev_near = OracleVisibleEvidence(target_points=tp, other_points=np.vstack([op, shadow + [0, 0, -0.004]]),
                                    code=CODE_TARGET, band_cell=cell, band_keys=bk, band_masks=bm)
    assert not oracle_visible_split(c[:1], seed[:1], ev_near, VS)["removed"][0]
    # without the band only the surface path is left
    nb = OracleVisibleEvidence(target_points=tp, other_points=op2, z_floor=TABLE_Z + 0.005)
    assert oracle_visible_split(c, seed, nb, VS)["band"].sum() == 0
    # a band cell both objects passed is shared → kept
    k = oracle_band_masks(c[:1], bk, bm, cell)
    shared = bm.copy()
    shared[np.searchsorted(bk, oracle_cell_keys(c[:1], cell))] = k[0] | np.uint32(1 << CODE_OBJECT_BASE)
    ev2 = OracleVisibleEvidence(target_points=tp, other_points=op2, code=CODE_TARGET, band_cell=cell, band_keys=bk,
                                band_masks=shared)
    s2 = oracle_visible_split(c[:1], seed[:1], ev2, VS)
    assert not s2["removed"][0], "a shared band cell does not remove by the band path"
    with pytest.raises(ValueError):
        OracleVisibleEvidence(target_points=tp, band_keys=bk, band_masks=bm[:-1], band_cell=cell)
    with pytest.raises(ValueError):
        OracleVisibleEvidence(target_points=tp, band_keys=bk, band_masks=bm, band_cell=0.0)


def test_oracle_candidates_and_empty_inputs():
    assert oracle_candidates(np.zeros((0, 3)), np.zeros(3), VS, (10, 10, 10), 0.03) is None
    cand = oracle_candidates(np.array([[0.025, 0.025, 0.025]]), np.zeros(3), VS, (10, 10, 10), 0.0101)
    d = np.linalg.norm(cand * VS - 0.025, axis=1)
    assert np.all(d <= 0.0101) and len(cand) == int(np.sum(np.linalg.norm(
        np.stack(np.meshgrid(*[np.arange(10)] * 3, indexing="ij"), -1).reshape(-1, 3) * VS - 0.025, axis=1) <= 0.0101))
    assert oracle_candidates(np.array([[5.0, 5.0, 5.0]]), np.zeros(3), VS, (10, 10, 10), 0.01) is None
    ev = OracleVisibleEvidence(target_points=np.zeros((0, 3)))
    assert len(ev.other_points) == 0
    with pytest.raises(ValueError):
        OracleVisibleEvidence(target_points=np.zeros((1, 3)), max_dist_voxels=0.0)


# ============================================================================ 5 tiers (fake cuRobo)


def test_tiers_oracle_keeps_the_neighbour_and_table_and_clears_the_target():
    torch = pytest.importorskip("torch")
    T44 = pytest.importorskip("tests.ag3s.test_t44_target_exclusion_owned")
    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder

    b = T44._builder(target_exclusion="ball_owned")
    g = T44._centres()
    sdf, owner = T44._sdf(g)
    # GT samples = the voxel-band surface points of each object (the rig's "cameras" see everything above the table)
    surf = np.abs(sdf) < 0.4 * VS
    pts = g[surf]
    own = owner[surf]
    ev = OracleVisibleEvidence(target_points=pts[own == 0], other_points=pts[own != 0], max_dist_voxels=6.0)
    itg = T44._FakeItg()
    (tier,) = b._build_tiers(itg, None, "cpu", torch, only="fine", tier_name="fine_no_target",
                             ball=T44._ball(), oracle=ev)
    (ball_tier,) = T44._tiers(b, T44._ball())
    (default,) = T44._tiers(b, T44._ball(), owned=T44._owned())
    (again,) = b._build_tiers(T44._FakeItg(), None, "cpu", torch, only="fine", tier_name="fine_no_target",
                              ball=T44._ball(), owned=T44._owned())
    assert torch.equal(default.values, again.values), "the oracle kwarg's absence is the old call"
    v = tier.values.numpy()
    above = g[..., 2] > T44.TABLE_Z + VS
    b_surf = (owner == 1) & (np.abs(sdf) < 0.5 * VS) & above
    t_surf = (owner == 0) & (np.abs(sdf) < 0.5 * VS)
    table = (owner == 2) & (np.abs(sdf) < 0.5 * VS) & (np.linalg.norm(g[..., :2] - T44.A[:2], axis=-1) < 0.03)
    assert np.abs(v[b_surf]).max() < 1e-6, "the neighbour surface stays"
    assert np.abs(v[table]).max() < 1e-6, "the table under the target stays"
    assert np.min(np.abs(v[t_surf])) > 0.0, "every target surface voxel is cleared"
    # the ball mode clears the neighbour's surface inside the ball — the oracle does not
    in_ball = np.linalg.norm(g - T44.A, axis=-1) <= T44._ball().radius
    assert np.min(np.abs(ball_tier.values.numpy()[b_surf & in_ball])) > 0.0
    ex = tier.extra["exclusion"]
    assert ex["exclusion_method"] == "oracle_visible"
    assert ex["n_removed"] == ex["n_target_owned"] > 0 and ex["n_ambiguous"] == 0
    assert ex["n_seeds_in_ball"] == ex["n_removed"] + ex["n_kept_nontarget"]
    assert ex["oracle"]["n_target_points"] == int((own == 0).sum()) and ex["oracle"]["n_candidates"] > 0
    tr = tier.extra["exclusion_trace"]
    assert len(tr["removed_ijk"]) == ex["n_removed"] and len(tr["ambiguous_ijk"]) == 0
    counts = CuroboFieldBuilder._exclusion_counts([tier])
    assert counts["exclusion_method"] == "oracle_visible" and counts["oracle_split"] == ex["oracle"]
    assert "oracle_split" not in CuroboFieldBuilder._exclusion_counts([default])
    assert tier.extra["ball"] == T44._ball().summary(), "the ball stays in the record"


# ============================================================================ 6 pipeline wiring (fake builder)


def _wired(monkeypatch, **esdf):
    """`test_fine_window`'s AG3S with a fake builder that writes a `target_free` stats block (as `test_t46` does)."""
    fw = pytest.importorskip("tests.ag3s.test_fine_window")

    class _Recording(fw._FakeBuilder):
        def update(self, cameras, **kw):
            field = super().update(cameras, **kw)
            if kw.get("target_free_points") is not None:
                field.stats["target_free"] = {"mode": "ball" if kw.get("target_free_ball") is not None else "points"}
            else:
                field.stats["target_free"] = {"mode": "none"}
            return field

    ag3s = fw._ag3s(monkeypatch, **esdf)
    ag3s._esdf_builder = _Recording()
    return fw, ag3s


def _apple_views(fw, other_offset=(0.0, 0.083, 0.0)):
    """Raycast views of the fine-window rig's apple (target) and a neighbour 3 mm away."""
    nb = fw.APPLE + np.asarray(other_offset)
    sph = ((fw.APPLE, fw.APPLE_R, CODE_TARGET), (nb, fw.APPLE_R, CODE_OBJECT_BASE))
    out = [_raycast(_look_at(eye, fw.APPLE), spheres=sph, table_z=fw.TABLE_Z)
           for eye in (fw.APPLE + [0.0, 0.0, 0.45], fw.APPLE + [-0.35, -0.1, 0.3])]
    return out, nb


def _observations(views):
    from AG3S.ag3s.types import CameraID

    ids = (CameraID.HEAD, CameraID.LEFT_WRIST)
    return [types.SimpleNamespace(camera_id=ids[i], depth=v.depth_m, camera_intrinsics=v.K, T_base_cam=v.T_base_cam)
            for i, v in enumerate(views)]


def _seg(views):
    return {"labels": {k: v.labels for k, v in zip(("head", "left_wrist"), views)},
            "meta": {"version": 1, "target": "apple", "codes": {"apple": 1, "orange": 3}}}


def test_pipeline_default_ignores_segmentation_and_keeps_the_builder_call(monkeypatch):
    fw, ag3s = _wired(monkeypatch, target_exclusion="ball_owned")
    views, _ = _apple_views(fw)
    ag3s.set_oracle_segmentation(_seg(views))
    assert ag3s._oracle_pending is None, "off: the segmentation is not even stored"
    field, _ = fw._build(ag3s)
    kw = ag3s._esdf_builder.calls[-1]
    assert "oracle_evidence" not in kw
    assert "oracle" not in field.stats["target_free"]
    assert ag3s._oracle_memory is None


def test_pipeline_oracle_passes_the_identified_objects_samples(monkeypatch):
    fw, ag3s = _wired(monkeypatch, target_exclusion="ball_owned", target_extent="oracle_visible")
    views, nb = _apple_views(fw)
    ag3s.set_oracle_segmentation(_seg(views))
    rec = ag3s._oracle_update(_observations(views))
    ag3s._oracle_frame = rec
    assert rec["segmentation"] and rec["memory"]["updated"] and rec["memory"]["n_target_samples"] > 100
    assert rec["cameras"] == ["head", "left_wrist"] and rec["target"] == "apple"
    field, _ = fw._build(ag3s)
    kw = ag3s._esdf_builder.calls[-1]
    ev = kw["oracle_evidence"]
    assert np.abs(np.linalg.norm(ev.target_points - fw.APPLE, axis=1) - fw.APPLE_R).max() < 1e-3
    assert np.any(np.abs(np.linalg.norm(ev.other_points - nb, axis=1) - fw.APPLE_R) < 1e-3), "neighbour samples"
    assert ev.max_dist_voxels == 6.0
    o = field.stats["target_free"]["oracle"]
    assert o["applied"] and o["code"] == CODE_TARGET and o["object"] == "apple"
    assert o["identity_matches_gt"] is True and o["identity_mode"] == "pipeline"
    assert o["memory"]["n_target_samples"] == len(ev.target_points)
    # the estimated extent's inputs are still passed (the coarse tier and the ball record read them)
    assert kw["target_free_ball"] is not None and kw["target_free_points"] is not None
    ag3s.reset()
    assert len(ag3s._oracle_memory) == 0 and ag3s._oracle_pending is None


def test_pipeline_oracle_identity_pipeline_or_gt(monkeypatch):
    """The pipeline's geometry on the neighbour: `pipeline` removes the neighbour (the identity is AG3S's, only the
    extent is GT); `gt` removes the GT target."""
    for mode, want in (("pipeline", CODE_OBJECT_BASE), ("gt", CODE_TARGET)):
        fw, ag3s = _wired(monkeypatch, target_extent="oracle_visible", oracle_extent_identity=mode)
        views, nb = _apple_views(fw)
        ag3s.set_oracle_segmentation(_seg(views))
        ag3s._oracle_frame = ag3s._oracle_update(_observations(views))
        geom = fw._apple_geometry()
        geom = types.SimpleNamespace(points=geom.points - fw.APPLE + nb)
        field, _ = ag3s._build_esdf([object()], geom, support_surfaces=[fw.TABLE], robot_state=np.zeros(2),
                                    contact_context=fw.LEFT)
        o = field.stats["target_free"]["oracle"]
        assert o["applied"] and o["code"] == want and o["identity_matches_gt"] is False, (mode, o)
        ev = ag3s._esdf_builder.calls[-1]["oracle_evidence"]
        centre = nb if want == CODE_OBJECT_BASE else fw.APPLE
        assert np.abs(np.linalg.norm(ev.target_points - centre, axis=1) - fw.APPLE_R).max() < 1e-3


def test_pipeline_oracle_without_segmentation_builds_the_estimated_extent(monkeypatch):
    fw, ag3s = _wired(monkeypatch, target_exclusion="ball_owned", target_extent="oracle_visible")
    ag3s._oracle_frame = ag3s._oracle_update(_observations(_apple_views(fw)[0]))   # nothing pending
    assert ag3s._oracle_frame == {"segmentation": False, "memory": {"n_samples": 0, "update": 0, "updated": False}}
    field, _ = fw._build(ag3s)
    assert "oracle_evidence" not in ag3s._esdf_builder.calls[-1]
    o = field.stats["target_free"]["oracle"]
    assert o["applied"] is False and o["why"] == "no_segmentation"
    # no manipulated geometry → no target-free layer → the record says so
    field, _ = ag3s._build_esdf([object()], None, support_surfaces=[fw.TABLE], robot_state=np.zeros(2),
                                contact_context=fw.LEFT)
    o = field.stats["target_free"]["oracle"]
    assert o["applied"] is False and o["why"] == "no_target_free_layer"


def test_oracle_evidence_unresolved_identity():
    mem = OracleMemory()
    mem.update(_views(), legend={"orange": 1, "apple": 3})
    ev, rec = oracle_evidence(mem, np.array([[3.0, 3.0, 3.0]]), radius=0.01)
    assert ev is None and rec["why"].startswith("identity_unresolved")
    ev, rec = oracle_evidence(mem, np.array([[3.0, 3.0, 3.0]]), radius=0.01, identity="gt")
    assert ev is not None and rec["code"] == CODE_TARGET and rec["identity_matches_gt"] is None
    ev, rec = oracle_evidence(None, np.zeros((1, 3)), radius=0.01)
    assert ev is None and rec["why"] == "no_segmentation"


# ============================================================================ 7 wire · client · metadata


def _pack(**kw):
    from AG3S.trajopt import wire

    d = {"c": np.full((4, 5), 1.234)}
    return wire.pack_request({"state": np.zeros(3)}, cameras=["c"], depth=d, intrinsics={"c": np.eye(3)},
                             extrinsics={"c": np.eye(4)}, robot_state={"c": np.zeros(2)}, stamps={"c": 1.0},
                             phase="approach", **kw)


def test_wire_is_unchanged_without_the_oracle_and_round_trips_with_it():
    from AG3S.trajopt import wire

    old = _pack()
    assert not any("oracle" in k for k in old)
    lab = np.arange(20, dtype=np.uint8).reshape(4, 5) % 4
    req = _pack(oracle_seg={"c": lab}, oracle_seg_meta={"version": 1, "target": "orange", "codes": {"orange": 1}})
    assert set(req) - set(old) == {"ag3s/oracle_seg/c", "ag3s/oracle_seg_meta"}
    assert all(np.array_equal(req[k], old[k]) if isinstance(old[k], np.ndarray) else req[k] == old[k] for k in old)
    _, scene = wire.strip_request(req)
    assert wire.unpack_oracle_segmentation(wire.strip_request(old)[1]) is None
    # a camera name the mount table knows maps to its CameraID value; an unknown one keeps its name
    seg = wire.unpack_oracle_segmentation(scene)
    assert list(seg["labels"]) == ["c"] and np.array_equal(seg["labels"]["c"], lab)
    assert seg["meta"]["target"] == "orange"
    d2 = {"zed_left": np.ones((2, 2))}
    r2 = wire.pack_request({}, cameras=["zed_left"], depth=d2, intrinsics={"zed_left": np.eye(3)},
                           extrinsics={"zed_left": np.eye(4)}, robot_state={"zed_left": np.zeros(1)},
                           stamps={"zed_left": 0.0}, phase="approach", oracle_seg={"zed_left": np.ones((2, 2))})
    assert list(wire.unpack_oracle_segmentation(wire.strip_request(r2)[1])["labels"]) == ["head"]
    with pytest.raises(ValueError, match="oracle_seg"):
        _pack(oracle_seg={"c": np.zeros((3, 3), np.uint8)})


def test_client_sends_code_images_only_with_an_oracle_target():
    from AG3S.trajopt import wire
    from AG3S.trajopt.client import SafeRemoteClient
    from tests.trajopt import test_safe_client as C

    class _SegFrame(C._Frame):
        def __init__(self, q):
            super().__init__(q)
            self.body_ids = np.full((8, 8), -1, np.int32)
            self.body_ids[2:4, 2:4] = 4       # orange
            self.body_ids[5, 5] = 3           # apple
            self.body_ids[0, :] = 1           # table
            self.body_ids[7, 7] = 7           # robot

    class _SegScene(C._Scene):
        _body_names = {0: "world", 1: "table", 3: "apple", 4: "orange", 7: "link_left_arm_6"}

        def capture(self, _camera):
            self.captures += 1
            return _SegFrame(self.q)

    plain = C._Server()
    SafeRemoteClient(policy=plain, scene=_SegScene(), cameras=("zed_left",)).infer({"state": np.zeros(C.DOF)})
    assert not any("oracle" in k for k in plain.requests[0])
    srv = C._Server()
    cl = SafeRemoteClient(policy=srv, scene=_SegScene(), cameras=("zed_left", "wrist_cam_l"), oracle_target="orange")
    cl.infer({"state": np.zeros(C.DOF)})
    req = srv.requests[0]
    lab = req["ag3s/oracle_seg/zed_left"]
    assert lab.dtype == np.uint8 and lab.shape == (8, 8)
    assert lab[2, 2] == CODE_TARGET and lab[5, 5] == CODE_OBJECT_BASE and lab[0, 0] == CODE_STATIC
    assert lab[7, 7] == CODE_NONE and lab[6, 6] == CODE_NONE
    assert req["ag3s/oracle_seg_meta"]["target"] == "orange"
    assert req["ag3s/oracle_seg_meta"]["codes"]["orange"] == 1
    assert set(k for k in req if "oracle" in k) == {"ag3s/oracle_seg/zed_left", "ag3s/oracle_seg/wrist_cam_l",
                                                    "ag3s/oracle_seg_meta"}
    # the server that needs the oracle refuses a client without it, at connect time
    need = C._Server(metadata={wire.ORACLE_SEG_REQUIRED: True})
    with pytest.raises(RuntimeError, match="oracle"):
        SafeRemoteClient(policy=need, scene=_SegScene(), cameras=("zed_left",))
    SafeRemoteClient(policy=need, scene=_SegScene(), cameras=("zed_left",), oracle_target="orange")


def _safe_policy_with(**esdf):
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.trajopt.safe_policy import SafePolicy, grasp_parent_links
    from tests.trajopt.test_safe_policy import _robot_model, _StubPolicy

    model = _robot_model()
    ag3s = AG3S(AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": esdf,
                                      "clustering": {"gripper_max_opening": 0.0715}}),
                robot_model=model, constraint_robot_model=model, attached_parent_links=grasp_parent_links())
    return SafePolicy(_StubPolicy(), ag3s=ag3s)


def test_server_metadata_carries_the_oracle_only_when_on():
    from AG3S.trajopt import wire

    base = _safe_policy_with().metadata
    assert not any("oracle" in k or k == "target_extent" for k in base)
    assert _safe_policy_with(target_extent="estimated").metadata == base
    on = _safe_policy_with(target_extent="oracle_visible", oracle_extent_identity="gt").metadata
    assert on.pop(wire.ORACLE_SEG_REQUIRED) is True
    assert on.pop("target_extent") == "oracle_visible" and on.pop("oracle_extent_identity") == "gt"
    assert on == base


# ============================================================================ 8 recorded chunks (storage)

STORE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage"))
CL3 = STORE / "results/development/20261012T155130+0900_218229ac6464_5fb5ff542c33/verify/T46/CL3/runs/C1a"
C1A = STORE / "results/workspace/verify/T44/CL/runs/C1a"
REPO = pathlib.Path(__file__).resolve().parents[2]
needs_cl3 = pytest.mark.skipif(not CL3.exists(), reason="CL3 record (shared storage) not mounted")


def _cl3_chunks(n=6):
    out = []
    if not CL3.exists():
        return out
    for p in sorted(CL3.glob("server_*/server_constraints/run_*/chunk_*.npz")):
        out.append(p)
    # spread over the run (pre-grasp chunks carry the ball_owned layer)
    return out[::max(1, len(out) // n)][:n]


@needs_cl3
@pytest.mark.parametrize("path", _cl3_chunks(), ids=lambda p: f"{p.parents[2].name}/{p.name}")
def test_cl3_off_replay_is_still_the_record(path):
    """`rebuild_target_free` after moving its tail into `_replay_layer`: the recorded fine target-free layer, bit for
    bit (float16 as recorded), on CL3 C1a chunks (the record T46e's oracle arm is compared with)."""
    from AG3S.trajopt.experiments.target_free_replay import rebuild_target_free, recorded_split_equal, summary_of

    with np.load(path, allow_pickle=True) as z:
        rep = rebuild_target_free(z, summary_of(z))
        if rep is None:
            pytest.skip("no ball_owned fine target-free layer in this chunk")
        assert np.array_equal(rep["values"].astype(np.float16), np.asarray(z["esdf_free_distance"]))
        # (the trace sets may differ by a seed on a few chunks — a known replay limit, `T46.verify.json` design162)
        assert isinstance(recorded_split_equal(z, rep), dict)


ACCEPTED_RUN = ("C1a_ep0_s1", "server_8280_20261010T172734", "run_0000", "chunk_00314.npz", 15)


@pytest.mark.skipif(not (C1A / ACCEPTED_RUN[1]).exists(), reason="C1a record (shared storage) not mounted")
def test_recorded_chunk_oracle_replay_keeps_the_neighbours():
    """C1a_ep0_s1 seq 15 (orange, D-TF collision chunk): re-render the run's requests up to this chunk into the
    server's memory, rebuild the fine target-free layer with the oracle and check, against the MuJoCo bodies,
    that no removed seed lies on another fruit (≤ 3 mm of its surface samples) and that it removes target seeds
    the record kept."""
    from AG3S.trajopt.experiments.target_free_replay import (
        OracleRenderer,
        oracle_evidence_for_chunk,
        rebuild_oracle_free,
        summary_of,
    )

    run = C1A / "R" / ACCEPTED_RUN[0]
    if not (run / "frames.jsonl").exists():
        pytest.skip("client run dir not found")
    manifest = json.loads((run / "manifest.json").read_text())
    rows = [json.loads(line) for line in open(run / "frames.jsonl") if line.strip()]
    rows = [r for r in rows if r.get("kind") == "planning"]
    target = manifest["scene"]["episode_target_fruit"]
    path = C1A / ACCEPTED_RUN[1] / "server_constraints" / ACCEPTED_RUN[2] / ACCEPTED_RUN[3]
    # this checkout's model (the manifest's absolute path points into a retired workspace)
    r = OracleRenderer(str(REPO / "robots/rby1/description/models/rby1a/mujoco/model_transport.xml"),
                       target=target, rows=rows)
    mem = r.memory_at(ACCEPTED_RUN[4])
    assert r.max_pose_error_m < 0.005
    with np.load(path, allow_pickle=True) as z:
        s = summary_of(z)
        assert int(z["t_step"]) >= 0
        ev, rec = oracle_evidence_for_chunk(mem, z["target_points"])
        assert rec["applied"] and rec["identity_matches_gt"] is True, rec
        rep = rebuild_oracle_free(z, s, ev)
        rec_rm = {tuple(v) for v in np.asarray(z["target_free_removed_ijk"], np.int64).tolist()}
    got = rep["removed_ijk"]
    assert len(got) > 0
    centres = rep["origin"] + got * rep["voxel_size"]
    others = mem.points[(mem.codes >= CODE_OBJECT_BASE)]
    if len(others):
        from scipy.spatial import cKDTree

        d, _ = cKDTree(others).query(centres)
        tgt = mem.points[mem.codes == CODE_TARGET]
        dt, _ = cKDTree(tgt).query(centres)
        assert np.all(dt < d), "every removed seed is nearer the target than any other fruit sample"
    assert rep["n_removed_not_in_record"] == len({tuple(v) for v in got.tolist()} - rec_rm)


# ============================================================================ 9 GPU (real cuRobo)


def _strip_ms(d):
    if isinstance(d, dict):
        return {k: _strip_ms(v) for k, v in d.items() if not k.endswith("_ms")}
    if isinstance(d, list):
        return [_strip_ms(v) for v in d]
    return d


def test_curobo_default_is_bit_identical_and_the_oracle_touches_only_the_fine_target_free_layer():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    pytest.importorskip("curobo")
    G = pytest.importorskip("tests.o4.test_gpu_esdf_field")
    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder, FineWindowRequest, TargetBall

    cams = G._scene_cameras()
    apple = np.random.default_rng(0).normal([0.51, -0.03, 0.85], 0.012, (300, 3))
    other = np.random.default_rng(3).normal([0.51, 0.03, 0.85], 0.012, (300, 3))
    kw = dict(target_points=apple, target_free_points=apple,
              target_free_ball=TargetBall(centre=[0.51, -0.03, 0.85], radius=0.045, z_min=0.825),
              fine_window=FineWindowRequest(manipulated_points=apple, placement="target_centroid"),
              nontarget_points=other)
    base = {"target_exclusion": "ball_owned", "target_free_coarse": True}
    ev = OracleVisibleEvidence(target_points=apple, other_points=other)
    runs = {"default": (base, {}),
            "explicit": ({**base, "target_extent": "estimated"}, {}),
            "oracle": ({**base, "target_extent": "oracle_visible"}, {"oracle_evidence": ev})}
    fields = {}
    for name, (over, extra) in runs.items():
        b = CuroboFieldBuilder(G._config(**over))
        for _ in range(2):
            fields[name] = b.update(cams, **kw, **extra)
    d, e, o = fields["default"], fields["explicit"], fields["oracle"]
    for la, lb in zip(d.layers + d.target_free_layers, e.layers + e.target_free_layers):
        assert torch.equal(la.values_device, lb.values_device)
    assert torch.equal(d.target_free_coarse_layer.values_device, e.target_free_coarse_layer.values_device)
    assert _strip_ms(d.stats["target_free"]) == _strip_ms(e.stats["target_free"])
    for la, lb in zip(d.layers, o.layers):
        assert torch.equal(la.values_device, lb.values_device), "the oracle changed a main layer"
    assert torch.equal(d.target_free_coarse_layer.values_device, o.target_free_coarse_layer.values_device), \
        "the oracle changed the coarse target-free layer"
    t = o.stats["target_free"]
    assert t["mode"] == "oracle_visible" and t["exclusion_method"] == "oracle_visible"
    assert t["oracle_split"]["n_target_points"] == len(apple) and t["n_removed"] > 0
    assert d.stats["target_free"]["mode"] == "ball" and "oracle_split" not in d.stats["target_free"]
    assert o.exclusion_trace["method"] == "oracle_visible"
