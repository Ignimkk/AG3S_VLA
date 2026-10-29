"""T46 A11 — target-free 층의 target 잔여: F1 (non-target 근거 정화, `nontarget_evidence`) · F2 (ambiguous 제거,
`owned_ambiguous`). 둘 다 기본 off 이고, off 는 비트 동일이어야 한다.

## 무엇을 박나

| # | 성질 |
|---|---|
| 1 | config 기본값 · 왕복 · validation, CLI (`--nontarget-evidence` · `--nontarget-shell-voxels` · `--owned-ambiguous`) 왕복 · 거절 |
| 2 | F2 `_ball_owned_split`: removed = keep 의 removed ∪ (seed ∧ ambiguous), 다른 분류는 그대로 · sign ⊇ removed |
| 3 | F2 `_build_tiers` (가짜 cuRobo): keep 은 기본과 비트 동일 · remove 는 ambiguous seed 를 지우고 `n_ambiguous_removed` 를 남긴다 |
| 4 | F1 `nontarget_surface_evidence` 합성 rig: target 표면 점은 빼고 이웃 과일 점은 남긴다 (접촉 띠 포함) · 문 (맞춤 물러남) · 공 밖 · z_min 아래 |
| 5 | pipeline: 기본은 `_nontarget_points` 결과 · builder 호출 · stats 가 예전과 같다 · F1 이면 `owned_nontarget_points` · trace |
| 6 | 기록 chunk (C1a, storage 있을 때): off replay = 기록 target-free 층 (float16 비트) · trace 집합 · stats 수, 기본 인자 = 명시 off, F2 removed 집합, F1 문 |
| 7 | GPU (실제 cuRobo, CUDA 있을 때): 기본 = 명시 off 비트 동일 · F2 는 미세 target-free 만 바꾼다 · F1 kwarg 는 거친 target-free 를 안 바꾼다 |
"""

from __future__ import annotations

import os
import pathlib
import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.fields.curobo_builder import (
    OwnedEvidence,
    TargetBall,
    _ball_owned_split,
    nontarget_surface_evidence,
)

VS = 0.005


# ============================================================================ 1 config · CLI


def test_config_defaults_round_trip_and_validation():
    d = AG3SConfig().esdf
    assert (d.nontarget_evidence, d.nontarget_shell_voxels, d.owned_ambiguous) == ("proximity", 2.0, "keep")
    cfg = AG3SConfig.from_dict({"esdf": {"target_exclusion": "ball_owned", "nontarget_evidence": "surface",
                                         "nontarget_shell_voxels": 1.5, "owned_ambiguous": "remove"}})
    again = AG3SConfig.from_dict(cfg.to_dict())
    assert again.esdf == cfg.esdf
    assert (again.esdf.nontarget_evidence, again.esdf.nontarget_shell_voxels, again.esdf.owned_ambiguous) == \
        ("surface", 1.5, "remove")
    for bad in ({"nontarget_evidence": "radius"}, {"nontarget_shell_voxels": 0.0},
                {"nontarget_shell_voxels": -1.0}, {"nontarget_shell_voxels": float("nan")},
                {"nontarget_shell_voxels": float("inf")}, {"owned_ambiguous": "drop"}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig.from_dict({"esdf": bad})


def _parse(*argv):
    from AG3S.trajopt.serve_safe import build_parser

    return build_parser().parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml", *argv])


def test_cli_defaults_leave_the_config_untouched():
    from AG3S.trajopt.serve_safe import target_exclusion_section

    args = _parse("--target-exclusion", "ball-owned", "--owned-max-dist-voxels", "3")
    assert (args.nontarget_evidence, args.nontarget_shell_voxels, args.owned_ambiguous) == (None, None, None)
    sec = target_exclusion_section(args)
    assert sec == {"target_exclusion": "ball_owned", "owned_max_dist_voxels": 3.0}
    cfg = AG3SConfig.from_dict({"esdf": sec}).esdf
    assert (cfg.nontarget_evidence, cfg.owned_ambiguous) == ("proximity", "keep")


def test_cli_f1_f2_map_to_esdf_keys_and_back():
    from AG3S.trajopt.serve_safe import target_exclusion_section

    args = _parse("--target-exclusion", "ball-owned", "--nontarget-evidence", "surface",
                  "--nontarget-shell-voxels", "1.5", "--owned-ambiguous", "remove")
    sec = target_exclusion_section(args)
    assert sec == {"target_exclusion": "ball_owned", "nontarget_evidence": "surface",
                   "nontarget_shell_voxels": 1.5, "owned_ambiguous": "remove"}
    cfg = AG3SConfig.from_dict({"esdf": sec}).esdf
    assert (cfg.nontarget_evidence, cfg.nontarget_shell_voxels, cfg.owned_ambiguous) == ("surface", 1.5, "remove")
    # config → the same keys (what `build_ag3s` checks survived)
    assert {k: getattr(cfg, k) for k in sec} == sec


def _main_errors(monkeypatch, *argv):
    import sys

    from AG3S.trajopt import serve_safe

    monkeypatch.setattr(sys, "argv", ["serve_safe", "--config", "c", "--checkpoint", "k",
                                      "--model-xml", "x.xml", *argv])
    with pytest.raises(SystemExit) as e:
        serve_safe.main()
    return e.value.code


def test_cli_rejects_f1_f2_without_ball_owned_and_shell_without_surface(monkeypatch, capsys):
    for flags in (("--nontarget-evidence", "surface"), ("--owned-ambiguous", "remove"),
                  ("--nontarget-shell-voxels", "2")):
        assert _main_errors(monkeypatch, *flags) == 2
        assert "ball-owned" in capsys.readouterr().err
    owned = ("--esdf-backend", "curobo", "--target-field-policy", "exclude-authorized",
             "--target-exclusion", "ball-owned")
    assert _main_errors(monkeypatch, *owned, "--nontarget-shell-voxels", "2") == 2
    assert "--nontarget-evidence surface" in capsys.readouterr().err
    for bad in ("0", "-1", "nan", "inf"):
        assert _main_errors(monkeypatch, *owned, "--nontarget-evidence", "surface",
                            "--nontarget-shell-voxels", bad) == 2
        assert "유한한 양수" in capsys.readouterr().err


def test_ta_offline_variants_and_their_replay_settings():
    from AG3S.trajopt.experiments import ta_offline as TA

    assert TA.variant_target_free("TF0") == {}
    assert TA.variant_target_free("F1") == {"nontarget_evidence": "surface"}
    assert TA.variant_target_free("F2") == {"owned_ambiguous": "remove"}
    assert TA.variant_target_free("F1F2") == {"nontarget_evidence": "surface", "owned_ambiguous": "remove"}
    assert TA.variant_target_free("base") is None and TA.variant_target_free("PC") is None
    # the TO config of these variants is the recorded server's (no overrides)
    for v in ("TF0", "F1", "F2", "F1F2"):
        assert TA.variant_server_args(v) == ()
        assert {k for k in TA.VARIANTS[v] if not k.startswith("_")} == set()


# ============================================================================ 2 F2 split (pure numpy)

A = np.array([0.475, 0.290, 0.853])          # target centre
B = np.array([0.475, 0.335, 0.853])          # neighbour centre (45 mm away → 5 mm gap)
R = 0.020
TABLE_Z = 0.832


def _sphere_surface(c, r, n, seed, upper_only=True):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(4 * n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    if upper_only:
        v = v[v[:, 2] > -0.3]
    return c + r * v[:n]


def _seed_rig():
    """Seed voxels of the two spheres inside the ball (`|sdf| < 0.75 vs`), as `test_t44_target_exclusion_owned`."""
    lo = A - 0.04
    ax = [lo[i] + np.arange(17) * VS for i in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
    ball = TargetBall(centre=A, radius=0.035, z_min=TABLE_Z + VS)
    inside = (np.linalg.norm(g - A, axis=1) <= ball.radius) & (g[:, 2] >= ball.z_min)
    c = g[inside]
    sdf = np.minimum(np.linalg.norm(c - A, axis=1) - R, np.linalg.norm(c - B, axis=1) - R)
    return c, np.abs(sdf) < 0.75 * VS, ball


def _owned(**kw):
    return OwnedEvidence(target_points=_sphere_surface(A, R, 120, 0),
                         nontarget_points=_sphere_surface(B, R, 300, 1, upper_only=False), **kw)


def test_f2_removed_is_keep_plus_the_ambiguous_seeds():
    c, seed, ball = _seed_rig()
    keep = _ball_owned_split(c, seed, ball, _owned(), VS)
    keep2 = _ball_owned_split(c, seed, ball, _owned(ambiguous="keep"), VS)
    rm = _ball_owned_split(c, seed, ball, _owned(ambiguous="remove"), VS)
    assert int(keep["ambiguous"].sum()) > 0, "the rig must have ambiguous seeds"
    for k in keep:
        a, b = keep[k], keep2[k]
        assert (a == b) if np.isscalar(a) else np.array_equal(a, b), f"explicit keep ≠ default: {k}"
    for k in ("target", "nontarget", "ambiguous", "unowned", "d_t", "d_o"):
        assert np.array_equal(keep[k], rm[k]), f"F2 changed the classification {k}"
    assert np.array_equal(rm["removed"], keep["removed"] | (seed & keep["ambiguous"]))
    assert np.all(rm["sign"][rm["removed"]]), "sign ⊇ removed"
    assert np.array_equal(rm["sign"], keep["sign"] | rm["removed"])
    assert not np.any(rm["removed"] & ~seed)


def test_owned_evidence_rejects_an_unknown_ambiguous_mode():
    with pytest.raises(ValueError, match="ambiguous"):
        _owned(ambiguous="drop")


# ============================================================================ 3 F2 tiers (fake cuRobo)


def test_f2_tiers_keep_is_bit_identical_and_remove_clears_the_ambiguous_seeds():
    torch = pytest.importorskip("torch")
    T44 = pytest.importorskip("tests.ag3s.test_t44_target_exclusion_owned")
    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder

    b = T44._builder(target_exclusion="ball_owned")
    (base,) = T44._tiers(b, T44._ball(), owned=T44._owned())
    (keep,) = T44._tiers(b, T44._ball(), owned=T44._owned(ambiguous="keep"))
    (rm,) = T44._tiers(b, T44._ball(), owned=T44._owned(ambiguous="remove"))
    assert torch.equal(base.values, keep.values)
    assert base.extra["exclusion"].keys() == keep.extra["exclusion"].keys()
    assert {k: v for k, v in base.extra["exclusion"].items() if k != "classify_ms"} == \
        {k: v for k, v in keep.extra["exclusion"].items() if k != "classify_ms"}
    assert "n_ambiguous_removed" not in base.extra["exclusion"]
    assert "ambiguous" not in base.extra["exclusion"]["owned"]

    eb, er = base.extra["exclusion"], rm.extra["exclusion"]
    assert er["n_ambiguous"] == eb["n_ambiguous"] > 0
    assert er["n_ambiguous_removed"] == eb["n_ambiguous"]
    assert er["n_removed"] == eb["n_removed"] + eb["n_ambiguous"]
    assert er["owned"]["ambiguous"] == "remove"
    assert er["n_seeds_in_ball"] == (er["n_removed"] + er["n_kept_nontarget"]
                                     + er["n_ambiguous"] - er["n_ambiguous_removed"] + er["n_unowned_kept"])
    amb = rm.extra["exclusion_trace"]["ambiguous_ijk"]
    rem = {tuple(v) for v in rm.extra["exclusion_trace"]["removed_ijk"].tolist()}
    assert all(tuple(v) in rem for v in amb.tolist())
    # the ambiguous voxels are no longer surfaces of the target-free layer
    vals = rm.values.numpy()[amb[:, 0], amb[:, 1], amb[:, 2]]
    assert np.all(np.abs(vals) > 0.0)
    assert np.all(base.values.numpy()[amb[:, 0], amb[:, 1], amb[:, 2]] == 0.0)
    counts = CuroboFieldBuilder._exclusion_counts([rm])
    assert counts["n_ambiguous_removed"] == er["n_ambiguous_removed"]
    assert "n_ambiguous_removed" not in CuroboFieldBuilder._exclusion_counts([base])


# ============================================================================ 4 F1 synthetic rig

TA_C = np.array([0.50, 0.30, 0.861])         # target (orange-sized) on the table
TB_C = np.array([0.50, 0.363, 0.861])        # neighbour, 1 mm gap
TR = 0.031
T_TABLE = 0.830


def _fused(points, voxel=0.010):
    """Fusion-like cloud: the centroid of the points in each 10 mm voxel (`timing.fusion_voxel_size`)."""
    key = np.floor(points / voxel).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    out = np.zeros((inv.max() + 1, 3))
    np.add.at(out, inv, points)
    return out / np.bincount(inv)[:, None]


def _f1_rig(fit_ok=True):
    from AG3S.ag3s.robot_models.held_object import fit_surface_sphere

    rng = np.random.default_rng(7)

    def surf(c, n):
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        p = c + TR * v
        return p[p[:, 2] > T_TABLE + 0.004]

    a, b = surf(TA_C, 6000), surf(TB_C, 6000)
    gx, gy = np.meshgrid(np.arange(0.35, 0.65, 0.004), np.arange(0.15, 0.50, 0.004), indexing="ij")
    table = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, T_TABLE)], axis=1)
    cloud = np.vstack([_fused(a), _fused(b), _fused(table)])
    label = np.concatenate([np.zeros(len(_fused(a)), int), np.ones(len(_fused(b)), int),
                            np.full(len(_fused(table)), 2)])
    support = label == 2
    # the anchor: an older frame's target points — sparse (~100) on the upper part, 2 mm off (the object
    # moved a little since); the current cloud is the 10 mm fused one, so its points fall between them
    anc = surf(TA_C, 3000)
    anc = anc[anc[:, 2] > TA_C[2] - 0.3 * TR][:100] + np.array([0.0, 0.0, 0.002])
    fit = fit_surface_sphere(anc)
    assert fit.accepted
    spread = float(np.linalg.norm(anc - fit.centre, axis=1).max())
    radius = max(spread, fit.radius) + VS
    prov = {"centre_method": "surface_fit" if fit_ok else "centroid",
            "fit": fit.record() if fit_ok else {**fit.record(), "method": "centroid", "reason": "test"}}
    ball = TargetBall(centre=fit.centre, radius=radius, z_min=T_TABLE + VS, provenance=prov)
    return cloud, label, support, anc, ball


def _proximity_evidence(cloud, support, anc):
    from AG3S.ag3s.runtime.pipeline import AG3S

    stub = types.SimpleNamespace(config=AG3SConfig.from_dict(
        {"esdf": {"backend": "curobo", "fine_voxel_size": VS, "voxel_size": 0.02}}))
    geom = types.SimpleNamespace(points=anc, point_indices=np.zeros(0, np.int64))
    return AG3S._nontarget_points(stub, types.SimpleNamespace(points=cloud), support, geom)


def _label_of(points, cloud, label):
    from scipy.spatial import cKDTree

    d, i = cKDTree(cloud).query(points)
    assert np.all(d < 1e-12)
    return label[i]


def test_f1_drops_the_target_surface_and_keeps_the_neighbour():
    """Touching fruits (1 mm gap), 10 mm fused cloud, a sparse 2 mm-off anchor. With the pipeline's settings
    (shell 2 voxels, tie = `owned_tie_voxels` 1 voxel) F1 takes ≥ 80 % of the target's left-over surface points
    out of the evidence. Neighbour points can only be taken in the contact band (inside the ball and the
    shell); the tie keeps all but ≤ 1 of them there, and protection never takes more than no protection."""
    cloud, label, support, anc, ball = _f1_rig()
    nt = _proximity_evidence(cloud, support, anc)
    assert nt["source"] == "proximity"
    lab = _label_of(nt["points"], cloud, label)
    n_target_left = int((lab == 0).sum())
    assert n_target_left >= 10, "precondition: proximity (5 mm) leaves target surface points as evidence"
    drop, info = nontarget_surface_evidence(nt["points"], anc, ball, shell_m=2.0 * VS, tie_m=VS)
    assert info["applied"] and info["n_dropped"] == int(drop.sum()) and info["tie_mm"] == 5.0
    assert int((drop & (lab == 0)).sum()) >= 0.8 * n_target_left, "F1 must take the target surface out"
    assert int((drop & (lab == 2)).sum()) == 0
    r = np.linalg.norm(nt["points"] - ball.centre, axis=1)
    band = (lab == 1) & (r <= info["limit_mm"] / 1000.0) & (nt["points"][:, 2] >= ball.z_min)
    assert int(band.sum()) >= 2, "precondition: neighbour points in the contact band"
    taken = drop & (lab == 1)
    assert np.all(band[taken]), "a neighbour point outside the contact band was taken"
    assert int(taken.sum()) <= 1
    d_surface = np.abs(np.linalg.norm(nt["points"][taken] - TA_C, axis=1) - TR)
    assert np.all(d_surface <= VS), "only a neighbour point within one voxel of the target surface"
    no_tie, _ = nontarget_surface_evidence(nt["points"], anc, ball, shell_m=2.0 * VS, tie_m=0.0)
    assert int((no_tie & (lab == 1)).sum()) >= int(taken.sum())
    assert np.all(no_tie[drop]), "the tie only ever keeps more"


def test_f1_gate_ball_and_zmin():
    cloud, label, support, anc, ball = _f1_rig(fit_ok=False)
    nt = _proximity_evidence(cloud, support, anc)
    drop, info = nontarget_surface_evidence(nt["points"], anc, ball, shell_m=2.0 * VS)
    assert not drop.any() and not info["applied"] and info["why"].startswith("no_surface_fit")
    # nothing outside the ball / below z_min, whatever the shell
    cloud, label, support, anc, ball = _f1_rig()
    nt = _proximity_evidence(cloud, support, anc)
    drop, info = nontarget_surface_evidence(nt["points"], anc, ball, shell_m=1.0)
    p = nt["points"][drop]
    assert np.all(np.linalg.norm(p - ball.centre, axis=1) <= ball.radius + 1e-12)
    assert np.all(p[:, 2] >= ball.z_min)
    assert info["limit_mm"] == pytest.approx(ball.radius * 1000.0, abs=1e-6)
    # no evidence / no target points: nothing to do
    d0, i0 = nontarget_surface_evidence(np.zeros((0, 3)), anc, ball, shell_m=VS)
    assert d0.shape == (0,) and i0["applied"] and i0["n_dropped"] == 0
    d1, i1 = nontarget_surface_evidence(nt["points"], np.zeros((0, 3)), ball, shell_m=VS)
    assert not d1.any() and i1["why"] == "no_target_points"


def test_f1_then_ball_owned_removes_the_target_seeds_the_evidence_was_holding():
    """End to end on the rig's seeds: F1 turns ambiguous/non-target target seeds into removed ones and keeps the
    neighbour's seeds (`_ball_owned_split` on voxel centres near both surfaces)."""
    cloud, label, support, anc, ball = _f1_rig()
    nt = _proximity_evidence(cloud, support, anc)
    drop, _ = nontarget_surface_evidence(nt["points"], anc, ball, shell_m=2.0 * VS, tie_m=VS)
    lo = ball.centre - ball.radius
    ax = [lo[i] + np.arange(int(2 * ball.radius / VS) + 2) * VS for i in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
    g = g[(np.linalg.norm(g - ball.centre, axis=1) <= ball.radius) & (g[:, 2] >= ball.z_min)]
    sa = np.abs(np.linalg.norm(g - TA_C, axis=1) - TR)
    sb = np.abs(np.linalg.norm(g - TB_C, axis=1) - TR)
    seed = np.minimum(sa, sb) < 0.75 * VS
    on_a, on_b = seed & (sa < sb), seed & (sb <= sa)
    kw = dict(tie_voxels=1.0, max_dist_voxels=3.0, unowned="remove")
    off = _ball_owned_split(g, seed, ball, OwnedEvidence(anc, nt["points"], **kw), VS)
    f1 = _ball_owned_split(g, seed, ball, OwnedEvidence(anc, nt["points"][~drop], **kw), VS)
    assert int((off["removed"] & on_a).sum()) < int(on_a.sum()), "precondition: off keeps target seeds"
    assert int((f1["removed"] & on_a).sum()) > int((off["removed"] & on_a).sum())
    assert int((f1["removed"] & on_a).sum()) >= 0.7 * int(on_a.sum())
    assert int((f1["removed"] & on_b).sum()) == int((off["removed"] & on_b).sum()) == 0, \
        "F1 must not remove neighbour seeds"
    # the contrast the task asks the verifier to measure: F2 removes the ambiguous seeds — here the
    # neighbour's seeds inside the ball are among them
    f2 = _ball_owned_split(g, seed, ball, OwnedEvidence(anc, nt["points"], ambiguous="remove", **kw), VS)
    assert int((f2["removed"] & on_a).sum()) > int((off["removed"] & on_a).sum())
    assert int((f2["removed"] & on_b).sum()) > 0


# ============================================================================ 5 pipeline wiring


def _old_nontarget_points(cfg, cloud, support_mask, geometry) -> dict:
    """`AG3S._nontarget_points` as of T44 I1 (HEAD 8efc1157) — the off reference."""
    from scipy.spatial import cKDTree

    pts = np.asarray(getattr(cloud, "points", cloud), np.float64).reshape(-1, 3)
    n = len(pts)
    keep = np.ones(n, bool)
    n_support = 0
    if support_mask is not None and len(support_mask) == n:
        sm = np.asarray(support_mask, bool)
        n_support = int(sm.sum())
        keep &= ~sm
    tp = np.asarray(geometry.points, np.float64).reshape(-1, 3)
    idx = np.asarray(getattr(geometry, "point_indices", ()), np.int64).reshape(-1)
    source = "proximity"
    if (len(idx) == len(tp) and len(idx) and int(idx.min()) >= 0 and int(idx.max()) < n
            and np.allclose(pts[idx], tp, rtol=0.0, atol=1e-9)):
        source = "indices"
        drop = np.zeros(n, bool)
        drop[idx] = True
    else:
        radius = float(getattr(cfg, "fine_voxel_size", 0.0) or cfg.voxel_size)
        drop = np.zeros(n, bool)
        if len(tp) and n:
            d, _ = cKDTree(tp).query(pts, k=1)
            drop = d <= radius
    n_target = int((drop & keep).sum())
    keep &= ~drop
    return {"points": pts[keep], "source": source, "n_cloud": int(n),
            "n_target_dropped": n_target, "n_support_dropped": n_support}


def _assert_same_evidence(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if k == "points":
            assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k])
        else:
            assert a[k] == b[k], k


def test_pipeline_default_nontarget_points_equal_t44_and_f1_records_why_proximity():
    from AG3S.ag3s.runtime.pipeline import AG3S

    cloud, label, support, anc, ball = _f1_rig()
    cfg = AG3SConfig.from_dict({"esdf": {"backend": "curobo", "fine_voxel_size": VS, "voxel_size": 0.02}})
    geom = types.SimpleNamespace(points=anc, point_indices=np.zeros(0, np.int64),
                                 metrics={"manipulated_subset_kept_anchor": 1.0})
    cl = types.SimpleNamespace(points=cloud)
    stub = types.SimpleNamespace(config=cfg)
    _assert_same_evidence(AG3S._nontarget_points(stub, cl, support, geom),
                          _old_nontarget_points(cfg.esdf, cl, support, geom))
    # indices path (this cloud's indices) — the same as before too
    idx = np.flatnonzero(label == 0)[:20]
    g2 = types.SimpleNamespace(points=cloud[idx], point_indices=idx)
    _assert_same_evidence(AG3S._nontarget_points(stub, cl, support, g2),
                          _old_nontarget_points(cfg.esdf, cl, support, g2))

    class _Stub:
        _proximity_reason = AG3S._proximity_reason

        def __init__(self, c):
            self.config = c
            self._grasp_active = True
            self.frame_index = 14
            self._gated_manipulated = types.SimpleNamespace(state="visible", last_seen_frame=14)

    f1cfg = AG3SConfig.from_dict({"esdf": {"backend": "curobo", "fine_voxel_size": VS, "voxel_size": 0.02,
                                           "target_exclusion": "ball_owned", "nontarget_evidence": "surface"}})
    out = AG3S._nontarget_points(_Stub(f1cfg), cl, support, geom)
    ref = _old_nontarget_points(cfg.esdf, cl, support, geom)
    assert np.array_equal(out["points"], ref["points"]), "F1 does not change the shared (coarse) evidence"
    pr = out["proximity_reason"]
    assert pr["why"] == "no_indices" and pr["stand_in_flags"] == ["manipulated_subset_kept_anchor"]
    assert pr["grasp_active"] is True and pr["request_frame"] == 14 and pr["observed_frame"] == 14
    # F1 is a `ball_owned` option: with `ball` the record stays as before
    ball_cfg = AG3SConfig.from_dict({"esdf": {"backend": "curobo", "fine_voxel_size": VS, "voxel_size": 0.02,
                                              "nontarget_evidence": "surface"}})
    _assert_same_evidence(AG3S._nontarget_points(_Stub(ball_cfg), cl, support, geom), ref)
    for bad_idx, why in ((np.arange(5), "length_mismatch"), (np.arange(len(anc)) + 10 ** 7, "out_of_range"),
                         (np.arange(len(anc)), "points_mismatch")):
        g3 = types.SimpleNamespace(points=anc, point_indices=bad_idx)
        assert AG3S._nontarget_points(_Stub(f1cfg), cl, support, g3)["proximity_reason"]["why"] == why


def _wired(monkeypatch, **esdf):
    fw = pytest.importorskip("tests.ag3s.test_fine_window")

    class _Recording(fw._FakeBuilder):
        def update(self, cameras, **kw):
            field = super().update(cameras, **kw)
            if kw.get("target_free_ball") is not None:
                field.stats["target_free"] = {"mode": "ball"}
            return field

    ag3s = fw._ag3s(monkeypatch, **esdf)
    ag3s._esdf_builder = _Recording()
    return fw, ag3s


def _apple_evidence(fw):
    """Evidence near the rig apple: points on its surface the proximity drop missed + a neighbour."""
    rng = np.random.default_rng(3)
    v = rng.normal(size=(200, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    on_apple = fw.APPLE + fw.APPLE_R * v
    on_apple = on_apple[on_apple[:, 2] > fw.TABLE_Z + 0.006][:40]
    other = fw.APPLE + np.array([0.0, 2 * fw.APPLE_R + 0.002, 0.0]) + fw.APPLE_R * v[:60]
    other = other[other[:, 2] > fw.TABLE_Z + 0.006]
    pts = np.vstack([on_apple, other])
    return {"points": pts, "source": "proximity", "n_cloud": 500, "n_target_dropped": 30,
            "n_support_dropped": 100}, len(on_apple)


def test_pipeline_default_builder_call_and_stats_are_unchanged(monkeypatch):
    fw, ag3s = _wired(monkeypatch, target_exclusion="ball_owned")
    nt, _ = _apple_evidence(fw)
    field, _ = fw._build(ag3s, nontarget=nt)
    kw = ag3s._esdf_builder.calls[-1]
    assert "owned_nontarget_points" not in kw
    assert kw["nontarget_points"] is nt["points"]
    assert field.stats["target_free"]["nontarget"] == {
        "source": "proximity", "n_cloud": 500, "n_target_dropped": 30, "n_support_dropped": 100,
        "n_points": len(nt["points"])}
    # F2 alone does not touch the evidence either (it rides on the builder's config)
    fw, ag3s = _wired(monkeypatch, target_exclusion="ball_owned", owned_ambiguous="remove")
    fw._build(ag3s, nontarget=nt)
    assert "owned_nontarget_points" not in ag3s._esdf_builder.calls[-1]


def test_pipeline_f1_passes_the_fine_evidence_and_records_it(monkeypatch):
    fw, ag3s = _wired(monkeypatch, target_exclusion="ball_owned", nontarget_evidence="surface")
    nt, n_apple = _apple_evidence(fw)
    field, _ = fw._build(ag3s, nontarget=nt)
    kw = ag3s._esdf_builder.calls[-1]
    ball = kw["target_free_ball"]
    assert ball is not None and ball.provenance["centre_method"] == "surface_fit"
    assert kw["nontarget_points"] is nt["points"], "the coarse tier keeps the original evidence"
    own = kw["owned_nontarget_points"]
    drop, info = nontarget_surface_evidence(nt["points"], kw["target_free_points"], ball,
                                            shell_m=ag3s.config.esdf.nontarget_shell_voxels * fw.FINE,
                                            tie_m=ag3s.config.esdf.owned_tie_voxels * fw.FINE)
    assert np.array_equal(own, nt["points"][~drop])
    assert info["n_dropped"] >= 0.9 * n_apple, "the apple's own surface points leave the fine evidence"
    assert np.all(np.linalg.norm(own - fw.APPLE, axis=1) > fw.APPLE_R + 0.001) or info["n_protected"] > 0
    rec = field.stats["target_free"]["nontarget"]
    assert rec["n_points"] == len(nt["points"]) and rec["n_points_owned"] == len(own)
    assert rec["evidence"] == info
    # `ball` mode never builds the fine evidence (F1 is a ball_owned option)
    fw, ag3s = _wired(monkeypatch, nontarget_evidence="surface")
    fw._build(ag3s, nontarget=nt)
    assert "owned_nontarget_points" not in ag3s._esdf_builder.calls[-1]


# ============================================================================ 6 recorded chunks (C1a)

STORE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage"))
C1A = STORE / "results/workspace/verify/T44/CL/runs/C1a"
#: (server dir, run dir, chunk file, what it is) — D-TF (`T45TF.verify.json`) chunks whose recorded executed chunk has
#: collision rows (C1a, grasp 직전)
ACCEPTED = (
    ("server_8280_20261010T172734", "run_0000", "chunk_00314.npz", "C1a_ep0_s1 seq 15 orange: fit accepted"),
    ("server_8280_20261010T194812", "run_0000", "chunk_00166.npz", "C1a_ep104_s1041 seq 17 pear: fit accepted"),
)
#: the ball fell back to the centroid — the target points hold neighbour-fruit points (D-TF F3)
REJECTED = (
    ("server_8280_20261010T194812", "run_0000", "chunk_00317.npz", "C1a_ep110_s1101 seq 18 pear: fit rejected"),
)


def _recorded(rows):
    out = []
    for srv, run, name, what in rows:
        p = C1A / srv / "server_constraints" / run / name
        if p.exists():
            out.append(pytest.param(p, id=what))
    return out


needs_record = pytest.mark.skipif(not C1A.exists(), reason="C1a record (shared storage) not mounted")


@needs_record
@pytest.mark.parametrize("path", _recorded(ACCEPTED + REJECTED))
def test_recorded_chunk_off_replay_is_the_record(path):
    """off (the server's unchanged functions with default arguments) rebuilds the recorded fine target-free layer
    bit for bit (float16 as recorded), the recorded trace sets and the recorded counts."""
    from AG3S.trajopt.experiments.target_free_replay import (
        rebuild_target_free,
        recorded_split_equal,
        summary_of,
        target_free_stats,
    )

    with np.load(path, allow_pickle=True) as z:
        s = summary_of(z)
        rep = rebuild_target_free(z, s)
        explicit = rebuild_target_free(z, s, nontarget_evidence="proximity", owned_ambiguous="keep")
        rec = np.asarray(z["esdf_free_distance"])
        assert np.array_equal(rep["values"].astype(np.float16), rec), "off replay ≠ recorded layer"
        assert recorded_split_equal(z, rep) == {"removed": True, "ambiguous": True, "nontarget_kept": True}
        assert np.array_equal(explicit["values"], rep["values"])
        tf = target_free_stats(s)
        for k in ("n_seeds_in_ball", "n_removed", "n_kept_nontarget", "n_ambiguous", "n_unowned_removed",
                  "n_unowned_kept", "n_target_owned", "n_sign_candidates"):
            assert rep["counts"][k] == int(tf[k]), k
        assert "n_ambiguous_removed" not in rep["counts"]
        nt = tf["nontarget"]
        for k in ("source", "n_cloud", "n_target_dropped", "n_support_dropped", "n_points"):
            assert rep["evidence"][k] == nt[k], k
        assert set(rep["evidence"]) == set(nt), "off evidence record has new keys"
        assert rep["n_removed_not_in_record"] == rep["n_record_removed_not_removed"] == 0


@needs_record
@pytest.mark.parametrize("path", _recorded(ACCEPTED))
def test_recorded_chunk_f2_removes_exactly_the_ambiguous_seeds(path):
    from AG3S.trajopt.experiments.target_free_replay import rebuild_target_free, summary_of

    with np.load(path, allow_pickle=True) as z:
        s = summary_of(z)
        off = rebuild_target_free(z, s)
        f2 = rebuild_target_free(z, s, owned_ambiguous="remove")
        amb = {tuple(v) for v in np.asarray(z["target_free_ambiguous_ijk"], np.int64).tolist()}
        rec_rm = {tuple(v) for v in np.asarray(z["target_free_removed_ijk"], np.int64).tolist()}
        got = {tuple(v) for v in f2["removed_ijk"].tolist()}
        assert len(amb) > 0 and got == rec_rm | amb and not (rec_rm & amb)
        assert f2["counts"]["n_ambiguous_removed"] == len(amb) == off["counts"]["n_ambiguous"]
        assert f2["n_removed_not_in_record"] == len(amb)
        # the ambiguous voxels are no longer surfaces of the replayed layer
        a = np.asarray(sorted(amb), np.int64)
        assert np.all(off["values"][a[:, 0], a[:, 1], a[:, 2]] == 0.0)
        assert np.all(f2["values"][a[:, 0], a[:, 1], a[:, 2]] != 0.0)


@needs_record
@pytest.mark.parametrize("path", _recorded(ACCEPTED))
def test_recorded_chunk_f1_takes_target_points_out_of_the_fine_evidence(path):
    from AG3S.trajopt.experiments.target_free_replay import rebuild_target_free, summary_of

    with np.load(path, allow_pickle=True) as z:
        s = summary_of(z)
        off = rebuild_target_free(z, s)
        f1 = rebuild_target_free(z, s, nontarget_evidence="surface")
    ev = f1["evidence"]
    assert ev["evidence"]["applied"] and ev["evidence"]["n_dropped"] > 0
    assert ev["n_points_owned"] == ev["n_points"] - ev["evidence"]["n_dropped"]
    assert ev["n_points"] == off["evidence"]["n_points"], "the shared evidence is unchanged"
    assert ev["proximity_reason"]["why"] == "no_indices"
    assert f1["counts"]["n_ambiguous"] < off["counts"]["n_ambiguous"]
    assert f1["counts"]["n_removed"] > off["counts"]["n_removed"]
    assert f1["counts"]["n_seeds_in_ball"] == off["counts"]["n_seeds_in_ball"]


@needs_record
@pytest.mark.parametrize("path", _recorded(REJECTED))
def test_recorded_fit_rejected_chunk_f1_is_off(path):
    """Neighbour protection, gate: a chunk whose ball fell back to the centroid (target points not one sphere —
    here a pear whose anchor holds neighbour-fruit points) gets nothing from F1: the same layer as off."""
    from AG3S.trajopt.experiments.target_free_replay import rebuild_target_free, summary_of

    with np.load(path, allow_pickle=True) as z:
        s = summary_of(z)
        off = rebuild_target_free(z, s)
        f1 = rebuild_target_free(z, s, nontarget_evidence="surface")
        rec = np.asarray(z["esdf_free_distance"])
    assert not f1["evidence"]["evidence"]["applied"]
    assert f1["evidence"]["evidence"]["why"].startswith("no_surface_fit")
    assert np.array_equal(f1["values"], off["values"])
    assert np.array_equal(off["values"].astype(np.float16), rec)


# ============================================================================ 7 GPU (real cuRobo)


def _strip_ms(d):
    if isinstance(d, dict):
        return {k: _strip_ms(v) for k, v in d.items() if not k.endswith("_ms")}
    if isinstance(d, list):
        return [_strip_ms(v) for v in d]
    return d


def test_curobo_off_is_bit_identical_and_f1_f2_touch_only_the_fine_target_free_layer():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    pytest.importorskip("curobo")
    G = pytest.importorskip("tests.o4.test_gpu_esdf_field")
    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder, FineWindowRequest

    cams = G._scene_cameras()
    apple = np.random.default_rng(0).normal([0.51, -0.03, 0.85], 0.012, (300, 3))
    other = np.random.default_rng(3).normal([0.51, 0.03, 0.85], 0.012, (300, 3))
    kw = dict(target_points=apple, target_free_points=apple,
              target_free_ball=TargetBall(centre=[0.51, -0.03, 0.85], radius=0.045, z_min=0.825),
              fine_window=FineWindowRequest(manipulated_points=apple, placement="target_centroid"),
              nontarget_points=other)
    base = {"target_exclusion": "ball_owned", "target_free_coarse": True}
    runs = {"default": (base, {}),
            "explicit_off": ({**base, "nontarget_evidence": "proximity", "owned_ambiguous": "keep"}, {}),
            "f2": ({**base, "owned_ambiguous": "remove"}, {}),
            "f1_kwarg": ({**base, "nontarget_evidence": "surface"}, {"owned_nontarget_points": other[:50]})}
    fields = {}
    for name, (over, extra) in runs.items():
        b = CuroboFieldBuilder(G._config(**over))
        for _ in range(2):
            fields[name] = b.update(cams, **kw, **extra)
    d, e = fields["default"], fields["explicit_off"]
    for la, lb in zip(d.layers + d.target_free_layers, e.layers + e.target_free_layers):
        assert torch.equal(la.values_device, lb.values_device)
    assert torch.equal(d.target_free_coarse_layer.values_device, e.target_free_coarse_layer.values_device)
    assert _strip_ms(d.stats["target_free"]) == _strip_ms(e.stats["target_free"])
    for name in ("f2", "f1_kwarg"):
        f = fields[name]
        for la, lb in zip(d.layers, f.layers):
            assert torch.equal(la.values_device, lb.values_device), f"{name} changed a main layer"
        assert torch.equal(d.target_free_coarse_layer.values_device, f.target_free_coarse_layer.values_device), \
            f"{name} changed the coarse target-free layer"
    t0, t2 = d.stats["target_free"], fields["f2"].stats["target_free"]
    assert t0["n_ambiguous"] > 0, "the scene must have ambiguous seeds"
    assert t2["n_ambiguous_removed"] == t2["n_ambiguous"] == t0["n_ambiguous"]
    assert t2["n_removed"] == t0["n_removed"] + t0["n_ambiguous"]
    assert not torch.equal(d.target_free_layers[0].values_device, fields["f2"].target_free_layers[0].values_device)
    t1 = fields["f1_kwarg"].stats["target_free"]
    assert t1["owned"]["n_nontarget_points"] == 50 and t0["owned"]["n_nontarget_points"] == len(other)
