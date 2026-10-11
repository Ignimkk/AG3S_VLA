"""T44e A8 — `coarse_no_target` 에서 지원면 (table) seed 제외 (`esdf.target_free_coarse_support`).

`T44dA.verify.json` Q1: 거친 target-free 계층의 테이블 seed (표면 ±18 mm 가 값 0) 가 공 아래 target-free 거리의
zero level 을 0 → 11.5 mm 로 들어 올렸다. `"exclude"` 는 공 아래 지원면의 seed 를 그 계층에서 지우고 (그 면의
inlier 점 근거 · 면에서 seed 띠 + 반 칸 · non-target 관측과 공유하지 않는 seed), 공의 seed guard 를 공의 `z_min`
으로 내린다. 테이블은 미세 `f` 와 창 밖 본 합성 `d` 가 답한다.

## 무엇을 박나

| # | 성질 |
|---|---|
| 1 | 기본 `"keep"` = A7 비트 동일 — 근거를 줘도 spec · 값 · site_index · counts · params 가 같다 |
| 2 | `"exclude"`: 근거가 있는 지원면 seed 는 거친 계층에 없다 · distractor 와 공유한 seed 는 남는다 · 지운 voxel 은 띠 안 · 근거 점 근처 |
| 3 | 테이블 위 (target 옆) 의 거친 target-free 값 회복 · 합성 값 ≥ keep · 성질 `min(old, d) ≤ new ≤ old` |
| 4 | distractor 내부 음수 유지 (계층 · 합성) · 테이블 내부 (평면 아래) 음수 유지 (부호 교정 없음) · target 바닥 내부 ≥ 0 |
| 5 | 다른 평면 (근거 점이 없는 곳) 의 seed 는 그대로 |
| 6 | exclude 에서 sign floor (plane · guard) 는 같은 계층 — guard 가 공 z_min 으로 내려가 띠가 빈다 |
| 7 | 근거가 없으면 keep (fail-closed) · spec validation · config 왕복 · CLI · harness · pipeline 배선 · record |
| 8 | (GPU) 실제 cuRobo: 본 계층 · 미세 target-free 불변, 거친 target-free 만 바뀌고 성질 유지 |
"""

from __future__ import annotations

import dataclasses
import json
import types

import numpy as np
import pytest

from benchmark.ag3s.config import AG3SConfig, AG3SConfigError
from benchmark.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays

torch = pytest.importorskip("torch")
A6 = pytest.importorskip("tests.ag3s.test_t44c_a6_sign_floor")

from benchmark.ag3s.fields.curobo_builder import (  # noqa: E402
    COARSE_SUPPORT_MODES,
    CoarseFreeSpec,
    coarse_exclusion_params,
)

VC, VF, BC, BN = A6.VC, A6.VF, A6.BC, A6.BN
TABLE, TC, RT, DC, RD = A6.TABLE, A6.TC, A6.RT, A6.DC, A6.RD
TARGET, NONTARGET = A6.TARGET, A6.NONTARGET
PLANE = np.array([0.0, 0.0, 1.0, TABLE])


def _table_points(spacing=0.005):
    """관측된 테이블 윗면: 격자 안, 두 물체 발자국 밖 (카메라가 못 보는 곳)."""
    lo, hi = BC[:2] + 0.01, BC[:2] + (np.asarray(BN[:2]) - 1) * VC - 0.01
    xs, ys = np.arange(lo[0], hi[0], spacing), np.arange(lo[1], hi[1], spacing)
    g = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    keep = (np.linalg.norm(g - TC[:2], axis=1) > RT) & (np.linalg.norm(g - DC[:2], axis=1) > RD)
    g = g[keep]
    return np.c_[g, np.full(len(g), TABLE)]


SUPPORT = _table_points()
EVIDENCE = {"points": SUPPORT, "plane": PLANE}


def _spec(b, *, evidence=EVIDENCE, floor=None, owned=None):
    s = b._coarse_free_spec(A6._ball(), owned, NONTARGET, target_points=TARGET,
                            support_evidence=evidence)
    return s if floor is None else dataclasses.replace(s, sign_floor=floor)


def _layer(t):
    return layer_from_arrays(A6._values(t), t.origin, VC)


def _cen():
    return A6._grid(BC, BN, VC)


def _composite(coarse_layer, *, table=TABLE):
    cen = _cen()
    main = layer_from_arrays(A6._sdf(cen).astype(np.float64), BC, VC)
    fo = TC - 0.08
    # 미세 target-free: target 을 뺀 장면 (table + distractor) 의 거리 — 미세가 본 곳
    g = A6._grid(fo, (33, 33, 33), VF)
    fsd = np.minimum(np.linalg.norm(g - DC, axis=-1) - RD, g[..., 2] - table)
    fine = layer_from_arrays(fsd, fo, VF)
    return CuroboEsdfField((main,), outside_distance=1.0, target_free_layers=(fine,), stats={},
                           target_free_coarse_layer=coarse_layer), fo


# ---------------------------------------------------------------------------- 1 keep = A7


def test_keep_default_is_a7_bit_for_bit_even_with_evidence():
    b = A6._builder()
    assert b.config.target_free_coarse_support == "keep"
    s_ev, s_no = _spec(b), _spec(b, evidence=None)
    assert s_ev.support == "keep" and s_ev.support_points is None and s_ev.support_plane is None
    itg_a, a = A6._tier(b, s_no)
    itg_b, k = A6._tier(b, s_ev)
    assert torch.equal(a.values, k.values) and torch.equal(itg_a._site_index, itg_b._site_index)
    strip = lambda e: {k_: v for k_, v in e.items() if k_ != "classify_ms"}  # noqa: E731
    assert strip(a.extra["exclusion"]) == strip(k.extra["exclusion"])
    assert "support" not in k.extra["exclusion"]
    assert "support_removed_ijk" not in k.extra["exclusion_trace"]
    p = coarse_exclusion_params(A6._ball(), VC, s_ev)
    assert "support" not in p and "support_reach_m" not in p
    assert p["z_min_m"] == pytest.approx(TABLE + VC)          # guard 그대로


# ---------------------------------------------------------------------------- 2 seeds


def _seeds(itg):
    return np.argwhere(itg._site_index.numpy() >= 0)


def test_exclude_removes_evidenced_support_seeds_and_keeps_shared_ones():
    b = A6._builder(target_free_coarse_support="exclude")
    s = _spec(b)
    assert s.support == "exclude" and s.support_points.shape == SUPPORT.shape
    itg_k, _ = A6._tier(A6._builder(), _spec(A6._builder()))
    itg_e, e = A6._tier(b, s)
    p = coarse_exclusion_params(A6._ball(), VC, s)
    reach = p["support_reach_m"]
    assert reach == pytest.approx(0.9 * VC + 0.5 * VC)
    from scipy.spatial import cKDTree

    c = BC + _seeds(itg_e) * VC
    d_sup = cKDTree(SUPPORT).query(c)[0]
    d_nt = cKDTree(NONTARGET).query(c)[0]
    in_band = np.abs(c[:, 2] - TABLE) <= reach
    left = in_band & (d_sup <= reach)
    assert np.all(d_nt[left] <= 0.9 * VC), "남은 지원면 seed 는 non-target 과 공유한 것뿐"
    assert left.sum() > 0, "distractor 밑의 공유 seed 가 남는다"
    rec = e.extra["exclusion"]["support"]
    assert rec["applied"] and rec["n_removed"] > 0 and rec["n_shared_kept"] == int(left.sum())
    assert rec["n_seeds"] == rec["n_removed"] + rec["n_shared_kept"]
    rem = e.extra["exclusion_trace"]["support_removed_ijk"]
    assert len(rem) == rec["n_removed"]
    rc = BC + rem * VC
    assert np.all(np.abs(rc[:, 2] - TABLE) <= reach + 1e-9)
    assert np.all(cKDTree(SUPPORT).query(rc)[0] <= reach + 1e-9)
    assert len(_seeds(itg_e)) < len(_seeds(itg_k))


# ---------------------------------------------------------------------------- 3 table recovered


def test_table_side_target_free_distance_recovers(monkeypatch):
    # 테이블 윗면이 거친 격자 층 사이 (0.86 · 0.88) 에 놓이면 두 층이 다 seed (|sdf| ≤ 18 mm) — 그 사이의
    # 값이 0 이 되어 zero level 이 테이블 위로 뜬다 (`T44dA` Q1 의 기전). 그 장면을 만든다.
    table = 0.869
    monkeypatch.setattr(A6, "TABLE", table)
    sup = SUPPORT.copy()
    sup[:, 2] = table
    ev = {"points": sup, "plane": np.array([0.0, 0.0, 1.0, table])}
    _, k = A6._tier(A6._builder(), _spec(A6._builder(), evidence=ev))
    b = A6._builder(target_free_coarse_support="exclude")
    _, e = A6._tier(b, _spec(b, evidence=ev))
    # (a) target 옆 테이블 위 (finger 행이 오는 높이), 두 물체에서 떨어진 곳 — 거친 계층 값
    side = np.array([[TC[0] - 0.07, TC[1] + dy, table + h]
                     for dy in (-0.04, 0.0, 0.04) for h in (0.002, 0.004, 0.008)])
    ck, ce = _layer(k).distance(side), _layer(e).distance(side)
    assert np.all(ck < 0.001), "keep: 두 seed 층 사이라 거친 값이 0 — 테이블이 위로 뜬다"
    assert np.all(ce > 0.02), "exclude: 테이블이 거친 target-free 계층에 없다"
    # (b) 공 아래 target 바닥 둘레 (본 합성 d < 0 이라 max(c, d) = c) — T44dA 의 zero level 자리 (ring)
    th = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    under = np.array([[TC[0] + 0.034 * np.cos(t), TC[1] + 0.034 * np.sin(t), table + h]
                      for t in th for h in (0.004, 0.006, 0.008)])
    nk, _ = _composite(_layer(k), table=table)
    ne, _ = _composite(_layer(e), table=table)
    assert np.all(nk.distance(under) < 0.0), "본 합성에는 target 이 있다"
    vk, ve = nk.target_free_distance(under), ne.target_free_distance(under)
    assert np.all(vk < 0.001), "keep: 거친 seed 가 테이블 위 4–8 mm 를 0 으로 누른다"
    # 회복된 값은 미세 계층 (테이블 높이) 이 답한다
    np.testing.assert_allclose(ve, under[:, 2] - table, atol=1e-6)
    assert ne.answer_tier(under, target_free=True).tolist() == ["target_free"] * len(under)


def test_composition_stays_bracketed_with_the_exclude_tier():
    b = A6._builder(target_free_coarse_support="exclude")
    _, e = A6._tier(b, _spec(b))
    new, fo = _composite(_layer(e))
    old, _ = _composite(None)
    pts = np.random.default_rng(7).uniform(fo + 0.01, fo + 0.15, (5000, 3))
    n, o, d = new.target_free_distance(pts), old.target_free_distance(pts), new.distance(pts)
    assert np.all(np.minimum(o, d) <= n + 1e-12) and np.all(n <= o + 1e-12)
    v, _, _ = new._answer(pts, True, want_gradient=False)
    np.testing.assert_array_equal(v, n)


# ---------------------------------------------------------------------------- 4 signs


def test_distractor_interior_and_table_interior_stay_negative_target_bottom_nonnegative():
    _, k = A6._tier(A6._builder(), _spec(A6._builder()))
    b = A6._builder(target_free_coarse_support="exclude")
    _, e = A6._tier(b, _spec(b))
    le, lk = _layer(e), _layer(k)
    dist_in = np.array([DC + [0.012, 0.0, 0.0], DC + [0.0, 0.0, 0.01]])
    assert np.all(le.distance(dist_in) < 0.0)
    ne, _ = _composite(le)
    assert np.all(ne.target_free_distance(dist_in) < 0.0)
    # 테이블 속 seed 띠 밖 (더 깊은 층): 부호 교정을 걸지 않는다 — 음수 그대로
    v, vk = A6._values(e), A6._values(k)
    cen = _cen()
    deep = cen[..., 2] < TABLE - 0.9 * VC - 1e-9
    assert np.all(v[deep] <= 0.0) and np.any(v[deep] < 0.0)
    # 지운 지원면 seed 의 부호만 바뀐다 (순수 판정) — 그 voxel 들은 띠 안
    flipped = np.argwhere((np.sign(v) != np.sign(vk)) & (vk < 0))
    if len(flipped):
        fz = BC[2] + flipped[:, 2] * VC
        assert np.all(np.abs(fz - TABLE) <= 0.9 * VC + 0.5 * VC + 1e-9) or np.all(
            np.linalg.norm(BC + flipped * VC - TC, axis=1) <= A6._ball().radius + 1e-9)
    # 합성은 테이블 속에서 음수 (미세 f 가 답한다)
    ne_, _ = _composite(le)
    under = np.array([[TC[0] - 0.08, TC[1], TABLE - 0.006], [TC[0] - 0.06, TC[1] + 0.03, TABLE - 0.01]])
    assert np.all(ne_.target_free_distance(under) < 0.0)
    # target 바닥 내부: keep(plane) 와 같이 ≥ 0
    bottom = np.array([[TC[0], TC[1], TABLE + 0.012], [TC[0], TC[1], TABLE + 0.02]])
    assert np.all(le.distance(bottom) >= 0.0)


# ---------------------------------------------------------------------------- 5 other planes


def test_seeds_without_support_evidence_are_kept():
    b = A6._builder(target_free_coarse_support="exclude")
    shelf = SUPPORT.copy()
    shelf[:, 2] = TABLE + 0.30             # 근거가 다른 높이 — 이 장면의 테이블 seed 와 무관
    ev = {"points": shelf, "plane": np.array([0.0, 0.0, 1.0, TABLE + 0.30])}
    itg_e, e = A6._tier(b, _spec(b, evidence=ev))
    rec = e.extra["exclusion"]["support"]
    assert rec["n_seeds"] == 0 and rec["n_removed"] == 0
    cen = _cen()
    seeds = _seeds(itg_e)
    c = BC + seeds * VC
    far = np.linalg.norm(c[:, :2] - TC[:2], axis=1) > 0.10
    assert np.any(np.abs(c[far, 2] - TABLE) <= 0.9 * VC), "테이블 seed 가 남아 있다"
    del cen


# ---------------------------------------------------------------------------- 6 sign floor


def test_sign_floor_is_moot_under_exclude():
    b = A6._builder(target_free_coarse_support="exclude")
    itg_p, p = A6._tier(b, _spec(b, floor="plane"))
    itg_g, g = A6._tier(b, _spec(b, floor="guard"))
    assert torch.equal(p.values, g.values) and torch.equal(itg_p._site_index, itg_g._site_index)
    par = coarse_exclusion_params(A6._ball(), VC, _spec(b, floor="plane"))
    assert par["z_min_m"] == pytest.approx(A6._ball().z_min)
    assert par["sign_z_min_m"] == pytest.approx(par["z_min_m"])
    assert "plane_band" not in p.extra["exclusion"], "띠가 비어 계산하지 않는다"


def test_ball_owned_works_with_exclude():
    from benchmark.ag3s.fields.curobo_builder import OwnedEvidence

    b = A6._builder(target_exclusion="ball_owned", owned_max_dist_voxels=3.0,
                    target_free_coarse_support="exclude")
    owned = OwnedEvidence(target_points=TARGET, nontarget_points=NONTARGET, tie_voxels=1.0,
                          max_dist_voxels=3.0)
    _, e = A6._tier(b, _spec(b, owned=owned))
    assert e.extra["exclusion"]["exclusion_method"] == "ball_owned"
    assert e.extra["exclusion"]["support"]["n_removed"] > 0
    le = _layer(e)
    assert le.distance(np.array([DC + [0.012, 0.0, 0.0]]))[0] < 0.0


# ---------------------------------------------------------------------------- 7 wiring


def test_without_evidence_exclude_falls_back_to_keep():
    b = A6._builder(target_free_coarse_support="exclude")
    for ev in (None, {}, {"points": SUPPORT}, {"plane": PLANE}):
        assert b._coarse_support_fields(ev) == {}
        assert _spec(b, evidence=ev).support == "keep"
    assert A6._builder()._coarse_support_fields(EVIDENCE) == {}
    with pytest.raises(ValueError):
        CoarseFreeSpec(support="exclude")
    with pytest.raises(ValueError):
        CoarseFreeSpec(support="drop")
    assert COARSE_SUPPORT_MODES == ("keep", "exclude", "exclude_target")


def test_config_round_trip_and_validation():
    cfg = AG3SConfig.from_dict({"esdf": {"target_free_coarse": True,
                                         "target_free_coarse_support": "exclude"}})
    assert cfg.esdf.target_free_coarse_support == "exclude"
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf.target_free_coarse_support == "exclude"
    assert AG3SConfig.from_dict({}).esdf.target_free_coarse_support == "keep"
    assert AG3SConfig.from_dict({}).esdf.TARGET_FREE_COARSE_SUPPORTS == COARSE_SUPPORT_MODES
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"esdf": {"target_free_coarse_support": "drop"}})


def test_cli_flag_maps_to_the_esdf_key_and_defaults_to_nothing():
    from benchmark.trajopt.serve_safe import target_free_coarse_support_section

    assert target_free_coarse_support_section(A6._parse()) == {}
    sec = target_free_coarse_support_section(
        A6._parse("--target-free-coarse", "--target-free-coarse-support", "exclude"))
    assert sec == {"target_free_coarse_support": "exclude"}
    assert AG3SConfig.from_dict({"esdf": sec}).esdf.target_free_coarse_support == "exclude"


def test_cli_rejects_support_without_target_free_coarse(monkeypatch, capsys):
    import sys

    from benchmark.trajopt import serve_safe

    monkeypatch.setattr(sys, "argv", ["serve_safe", "--config", "c", "--checkpoint", "k",
                                      "--model-xml", "x.xml", "--esdf-backend", "curobo",
                                      "--target-field-policy", "exclude-authorized",
                                      "--target-free-coarse-support", "exclude"])
    with pytest.raises(SystemExit) as e:
        serve_safe.main()
    assert e.value.code == 2
    assert "--target-free-coarse-support" in capsys.readouterr().err


def test_harness_passes_support_and_evidence(monkeypatch):
    from benchmark.ag3s.experiments.live import exclusion_e2 as H

    seen = {}

    class _Mapper:
        integrator = object()
        _device = "cpu"

    class _B:
        _mapper = _Mapper()
        _fine_mapper = None

        def _coarse_free_spec(self, ball, owned, nontarget, target_points=None, **kw):
            seen["kw"] = kw
            return CoarseFreeSpec(nontarget_points=nontarget, target_points=target_points)

        def _build_tiers(self, itg, att, dev, torch_, **kw):
            seen["spec"] = kw["coarse_free"]
            return ("tier",)

    monkeypatch.setattr(H, "_sync", lambda: None)
    H.build_coarse_free_tier(_B(), A6._ball(), nontarget=NONTARGET, target=TARGET)
    assert seen["kw"] == {} and seen["spec"].support == "keep"
    H.build_coarse_free_tier(_B(), A6._ball(), nontarget=NONTARGET, target=TARGET,
                             support="exclude", support_evidence=EVIDENCE)
    assert seen["spec"].support == "exclude"
    np.testing.assert_array_equal(seen["spec"].support_plane, PLANE)
    H.build_coarse_free_tier(_B(), A6._ball(), nontarget=NONTARGET, support="exclude")
    assert seen["spec"].support == "keep", "근거가 없으면 keep"
    assert H._parser().parse_args(["--coarse-support", "exclude"]).coarse_support == "exclude"
    assert H._parser().parse_args([]).coarse_support is None


def test_pipeline_picks_the_ball_surface_and_its_inliers():
    from benchmark.ag3s.runtime.pipeline import AG3S
    from benchmark.ag3s.types import SupportSurface

    pts = np.r_[SUPPORT[:50], np.c_[np.random.default_rng(1).uniform(0, 1, (30, 2)),
                                    np.full(30, 0.40)]]
    table = SupportSurface(id=0, normal=np.array([0.0, 0.0, 1.0]), offset=TABLE,
                           point_indices=np.arange(50), point_count=50, rms_error=0.001)
    floor = SupportSurface(id=1, normal=np.array([0.0, 0.0, 1.0]), offset=0.40,
                           point_indices=np.arange(50, 80), point_count=30, rms_error=0.001)
    wall = SupportSurface(id=2, normal=np.array([1.0, 0.0, 0.0]), offset=0.2,
                          point_indices=np.arange(3), point_count=3, rms_error=0.001)
    rng = np.random.default_rng(2)
    for _ in range(20):
        p = rng.uniform([0, 0, 0.3], [1, 1, 1.2])
        surfs = [floor, wall, table]
        s = AG3S._support_surface_under(surfs, p)
        z = AG3S._support_plane_z(surfs, p)
        assert (s is None) == (z is None)
        if s is not None:
            assert (float(s.offset) - s.normal[0] * p[0] - s.normal[1] * p[1]) / s.normal[2] == z
    ball = types.SimpleNamespace(centre=TC)
    ev = AG3S._support_seed_evidence(pts, [floor, wall, table], ball)
    assert ev["surface_id"] == 0 and ev["points"].shape == (50, 3)
    np.testing.assert_array_equal(ev["plane"], PLANE)
    assert ev["z_at_centre_m"] == pytest.approx(TABLE)
    assert AG3S._support_seed_evidence(pts, [wall], ball) is None


def test_pipeline_passes_support_evidence_only_with_exclude(monkeypatch):
    fw = pytest.importorskip("tests.ag3s.test_fine_window")
    nt = {"points": np.zeros((4, 3)), "source": "indices", "n_cloud": 9,
          "n_target_dropped": 3, "n_support_dropped": 2}
    cloud = np.random.default_rng(0).uniform(0, 1, (5, 3))
    ag3s = fw._ag3s(monkeypatch, target_free_coarse=True, target_free_coarse_support="exclude")
    fw._build(ag3s, nontarget=nt, support_cloud=cloud)
    kw = ag3s._esdf_builder.calls[-1]
    ev = kw["support_seed_evidence"]
    np.testing.assert_array_equal(ev["points"], cloud[fw.TABLE.point_indices])
    np.testing.assert_array_equal(ev["plane"], [0.0, 0.0, 1.0, fw.TABLE_Z])
    ag3s = fw._ag3s(monkeypatch, target_free_coarse=True)
    fw._build(ag3s, nontarget=nt)
    assert "support_seed_evidence" not in ag3s._esdf_builder.calls[-1], "keep 은 예전 호출 그대로"


def test_run_computes_support_cloud_only_for_exclude():
    """`_run` 은 `target_free_coarse_support == "exclude"` 일 때만 `support_cloud` 를 만든다 (AST)."""
    import ast
    import inspect
    import textwrap

    from benchmark.ag3s.runtime import pipeline

    src = textwrap.dedent(inspect.getsource(pipeline))
    tree = ast.parse(src)
    guards = []
    for n in ast.walk(tree):
        if isinstance(n, ast.If) and "target_free_coarse_support" in ast.unparse(n.test):
            body = "\n".join(ast.unparse(b) for b in n.body)
            if "support_cloud =" in body:
                guards.append(ast.unparse(n.test))
    assert guards and all("'exclude'" in g or '"exclude"' in g for g in guards)


def test_the_record_writes_the_support_trace():
    from benchmark.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    f = types.SimpleNamespace(
        stats={"target_free": {"mode": "ball", "coarse": {"mode": "ball",
                                                          "support": {"n_removed": 7}}}},
        exclusion_trace={"tier": "fine_no_target", "method": "ball", "origin": np.zeros(3),
                         "voxel_size": VF, "removed_ijk": np.zeros((2, 3), np.int64)},
        coarse_exclusion_trace={"tier": "coarse_no_target", "method": "ball",
                                "origin": np.zeros(3), "voxel_size": VC,
                                "removed_ijk": np.ones((5, 3), np.int64),
                                "shared_kept_ijk": np.ones((3, 3), np.int64),
                                "support_removed_ijk": np.ones((7, 3), np.int64)})
    payload: dict = {}
    info = ConstraintRecordWriter._write_exclusion_trace(f, payload)
    assert payload["target_free_coarse_support_removed_ijk"].shape == (7, 3)
    json.dumps(info)
    f.coarse_exclusion_trace.pop("support_removed_ijk")
    payload = {}
    ConstraintRecordWriter._write_exclusion_trace(f, payload)
    assert "target_free_coarse_support_removed_ijk" not in payload


# ============================================================================ 8 GPU (real cuRobo; skipped on CPU)


def test_curobo_exclude_changes_only_the_coarse_free_tier():
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    pytest.importorskip("curobo")
    G = pytest.importorskip("tests.o4.test_gpu_esdf_field")
    from benchmark.ag3s.fields.curobo_builder import FineWindowRequest, TargetBall

    cams = G._scene_cameras()
    apple = np.random.default_rng(0).normal([0.51, -0.03, 0.85], 0.012, (300, 3))
    other = np.random.default_rng(3).normal([0.51, 0.03, 0.85], 0.012, (300, 3))
    ball = TargetBall(centre=[0.51, -0.03, 0.85], radius=0.045, z_min=0.825,
                      provenance={"support_plane_z_m": 0.82, "pad_mm": 5.0})
    xy = np.stack(np.meshgrid(np.arange(0.30, 0.75, 0.005), np.arange(-0.25, 0.25, 0.005),
                              indexing="ij"), -1).reshape(-1, 2)
    sup = np.c_[xy, np.full(len(xy), 0.82)]
    ev = {"points": sup, "plane": np.array([0.0, 0.0, 1.0, 0.82])}
    kw = dict(target_points=apple, target_free_points=apple, target_free_ball=ball,
              fine_window=FineWindowRequest(manipulated_points=apple, placement="target_centroid"),
              nontarget_points=other)
    fields = {}
    for name, over, extra in (("keep", {"target_free_coarse": True}, {"support_seed_evidence": ev}),
                              ("exclude", {"target_free_coarse": True,
                                           "target_free_coarse_support": "exclude"},
                               {"support_seed_evidence": ev}),
                              ("noev", {"target_free_coarse": True,
                                        "target_free_coarse_support": "exclude"}, {})):
        b = CuroboFieldBuilder_(G._config(**over))
        for _ in range(2):
            fields[name] = b.update(cams, **kw, **extra)
    k, e, n = fields["keep"], fields["exclude"], fields["noev"]
    for lk, le in zip(k.layers, e.layers):
        assert torch.equal(lk.values_device, le.values_device)
    assert torch.equal(k.target_free_layers[0].values_device, e.target_free_layers[0].values_device)
    assert not torch.equal(k.target_free_coarse_layer.values_device,
                           e.target_free_coarse_layer.values_device)
    se = e.stats["target_free"]["coarse"]["support"]
    assert se["applied"] and se["n_removed"] > 0
    assert "support" not in k.stats["target_free"]["coarse"]
    assert n.stats["target_free"]["coarse"]["support"]["applied"] is False
    assert torch.equal(k.target_free_coarse_layer.values_device,
                       n.target_free_coarse_layer.values_device), "근거 없으면 keep 과 같다"
    old = CuroboEsdfField(tuple(e.layers), outside_distance=e.outside_distance,
                          target_free_layers=tuple(e.target_free_layers), stats={})
    pts = np.random.default_rng(5).normal([0.51, 0.0, 0.86], 0.04, (2000, 3))
    nv, ov, d = e.target_free_distance(pts), old.target_free_distance(pts), e.distance(pts)
    assert np.all(np.minimum(ov, d) <= nv + 1e-6) and np.all(nv <= ov + 1e-6)


def CuroboFieldBuilder_(cfg):
    from benchmark.ag3s.fields.curobo_builder import CuroboFieldBuilder

    return CuroboFieldBuilder(cfg)
