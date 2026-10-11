"""T44e A8b — `esdf.target_free_coarse_support = "exclude_target"`: target 발자국 아래 지원면 seed 만 지운다.

사용자 판정 (2026-10-11) "이웃 과일 아래 support 만 남김". A8 `"exclude"` 는 공 아래 지원면 seed 를 (non-target 과
공유한 것만 빼고) 모두 지워 distractor 바닥 띠 (지원면 위 0–15 mm) 를 off 수준으로 풀었다. `"exclude_target"` 은 같은
지원면 seed 중 **target 발자국 안 · non-target 발자국 밖** 의 것만 지운다. 발자국 = 지원면 위 그림자 (평면 거리, m):
target = 공 원판 (반지름 + D) ∪ target 점 (D), non-target = 평면 위 `h_min` 초과 관측점 (D), D = √2 · 거친 voxel.
겹치면 남기고, 그 겹침 seed 는 공 규칙 (guard 가 공 z_min 으로 내려갔다) 도 지우지 않는다.

## 무엇을 박나

| # | 성질 |
|---|---|
| 1 | `"exclude"` · `"keep"` 의 경로는 그대로 (A8 test 파일이 박는다) · footprint 이 전부 덮고 non-target 증거가 없으면 `exclude_target` = `exclude` 비트 동일 |
| 2 | 지운 seed ⊂ exclude 가 지운 seed · 지운 seed 는 전부 target 발자국 안 · non-target 발자국 밖 · 그 밖의 지원면 seed 는 남는다 |
| 3 | target 옆 테이블 위 값 회복 (exclude 와 같다) · distractor 발자국 아래 테이블 위 값 = keep |
| 4 | 겹침 seed 는 공 안에서도 남는다 (`n_support_protected_in_ball`) |
| 5 | `h_min` 아래 non-target 점 (테이블 잡음) 은 발자국 증거가 아니다 |
| 6 | 성질 `min(old, d) ≤ new ≤ old` · distractor 내부 음수 · 평면 투영 (기운 면) |
| 7 | config · CLI · harness · pipeline (`inlier_threshold_m` 은 exclude_target 일 때만) |
| 8 | (GPU) 실제 cuRobo |
"""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from benchmark.ag3s.config import AG3SConfig

torch = pytest.importorskip("torch")
A6 = pytest.importorskip("tests.ag3s.test_t44c_a6_sign_floor")
A8 = pytest.importorskip("tests.ag3s.test_t44e_a8_coarse_support")

import benchmark.ag3s.fields.curobo_builder as CB  # noqa: E402
from benchmark.ag3s.fields.curobo_builder import (  # noqa: E402
    COARSE_SUPPORT_FOOTPRINT_DILATION_VOXELS,
    COARSE_SUPPORT_FOOTPRINT_MIN_HEIGHT_M,
    CoarseFreeSpec,
    coarse_exclusion_params,
    coarse_support_spec_fields,
)

VC, BC, BN = A6.VC, A6.BC, A6.BN
TC, RT, DC, RD = A6.TC, A6.RT, A6.DC, A6.RD
NONTARGET, TARGET = A6.NONTARGET, A6.TARGET
D = COARSE_SUPPORT_FOOTPRINT_DILATION_VOXELS * VC


def _tier(mode, evidence=None, **kw):
    b = A6._builder(target_free_coarse_support=mode, **kw)
    s = A8._spec(b, evidence=A8.EVIDENCE if evidence is None else evidence)
    itg, t = A6._tier(b, s)
    return itg, t, s


def _seed_set(itg):
    return {tuple(v) for v in A8._seeds(itg)}


def _xy(ijk):
    return (BC + np.asarray(ijk) * VC)[:, :2]


def _in_n(c_xy, table):
    nt = NONTARGET[NONTARGET[:, 2] - table > COARSE_SUPPORT_FOOTPRINT_MIN_HEIGHT_M]
    from scipy.spatial import cKDTree

    return cKDTree(nt[:, :2]).query(c_xy)[0] <= D + 1e-9


# ---------------------------------------------------------------------------- 1 relation to exclude


def test_constants_and_params():
    assert D == pytest.approx(np.sqrt(2.0) * VC)
    s = dataclasses.replace(CoarseFreeSpec(), support="exclude_target",
                            support_points=A8.SUPPORT, support_plane=A8.PLANE)
    p = coarse_exclusion_params(A6._ball(), VC, s)
    assert p["support"] == "exclude_target" and p["z_min_m"] == pytest.approx(A6._ball().z_min)
    assert p["footprint_dilation_m"] == pytest.approx(D)
    assert p["footprint_min_height_m"] == pytest.approx(0.008)
    pe = coarse_exclusion_params(A6._ball(), VC, dataclasses.replace(s, support="exclude"))
    assert "footprint_dilation_m" not in pe and pe["support"] == "exclude"
    assert {k: v for k, v in p.items() if not k.startswith("footprint") and k != "support"} == \
           {k: v for k, v in pe.items() if k != "support"}


def test_full_footprint_and_no_nontarget_evidence_equals_exclude(monkeypatch):
    """발자국이 격자를 다 덮고 non-target 발자국 증거가 없으면 (h_min 아주 큼) exclude 와 같은 계층이다."""
    monkeypatch.setattr(CB, "COARSE_SUPPORT_FOOTPRINT_DILATION_VOXELS", 1000.0)
    ev = {**A8.EVIDENCE, "inlier_threshold_m": 10.0}
    itg_e, e, _ = _tier("exclude")
    itg_t, t, s = _tier("exclude_target", evidence=ev)
    assert s.support_min_height_m == 10.0
    assert torch.equal(e.values, t.values) and torch.equal(itg_e._site_index, itg_t._site_index)
    re_, rt = e.extra["exclusion"]["support"], t.extra["exclusion"]["support"]
    assert re_["n_removed"] == rt["n_removed"] and rt["footprint"]["n_overlap_kept"] == 0


# ---------------------------------------------------------------------------- 2 which seeds go


def test_removed_seeds_are_target_footprint_minus_nontarget_footprint():
    itg_k, _, _ = _tier("keep")
    itg_e, _, _ = _tier("exclude")
    itg_t, t, _ = _tier("exclude_target")
    rec = t.extra["exclusion"]["support"]
    assert rec["mode"] == "exclude_target" and rec["applied"]
    fp = rec["footprint"]
    assert fp["n_in_target"] > 0 and fp["n_overlap_kept"] > 0 and fp["n_outside_target_kept"] > 0
    assert rec["n_removed"] == fp["n_in_target"] - fp["n_overlap_kept"]
    rem = t.extra["exclusion_trace"]["support_removed_ijk"]
    assert len(rem) == rec["n_removed"] > 0
    # ⊂ exclude 가 지운 것
    gone_e = _seed_set(itg_k) - _seed_set(itg_e)
    assert {tuple(v) for v in rem} <= gone_e
    # target 발자국 안 · non-target 발자국 밖
    xy = _xy(rem)
    assert np.all(np.linalg.norm(xy - TC[:2], axis=1) <= A6._ball().radius + D + 1e-9)
    assert not np.any(_in_n(xy, A6.TABLE))
    # 겹침으로 남긴 seed 는 seed 그대로 · 두 발자국 안
    kept = t.extra["exclusion_trace"]["support_overlap_kept_ijk"]
    assert len(kept) >= fp["n_overlap_kept"] > 0
    assert {tuple(v) for v in kept} <= _seed_set(itg_t)
    assert np.all(_in_n(_xy(kept), A6.TABLE))
    # 발자국 밖 (공 원판 + D 밖) 의 테이블 seed 는 남는다 — keep 과 같다
    far = {v for v in _seed_set(itg_k)
           if np.linalg.norm(_xy([v])[0] - TC[:2]) > A6._ball().radius + D + 1e-9}
    assert far and far <= _seed_set(itg_t)


# ---------------------------------------------------------------------------- 3 values


def _raised_table(monkeypatch, table=0.869):
    monkeypatch.setattr(A6, "TABLE", table)
    sup = A8.SUPPORT.copy()
    sup[:, 2] = table
    return {"points": sup, "plane": np.array([0.0, 0.0, 1.0, table])}, table


def test_target_side_recovers_and_distractor_footprint_matches_keep(monkeypatch):
    ev, table = _raised_table(monkeypatch)
    _, k, _ = _tier("keep", evidence=ev)
    _, e, _ = _tier("exclude", evidence=ev)
    _, t, _ = _tier("exclude_target", evidence=ev)
    lk, le, lt = A8._layer(k), A8._layer(e), A8._layer(t)
    # (a) target 옆 (distractor 반대쪽) 테이블 위 — exclude 처럼 회복
    side = np.array([[TC[0] - 0.05, TC[1] + dy, table + h]
                     for dy in (-0.02, 0.0, 0.02) for h in (0.002, 0.004, 0.008)])
    assert np.all(lk.distance(side) < 0.001)
    assert np.all(lt.distance(side) > 0.02)
    # 발자국 밖 테이블 seed 가 남아 있어 exclude 보다는 낮다 (그 seed 까지의 거리)
    assert np.all(lt.distance(side) <= le.distance(side) + 1e-12)
    # (b) distractor 발자국 (target 발자국 밖) 아래 테이블 위 — keep 그대로 (이웃 과일 아래 seed 를 남긴다)
    under_d = np.array([[DC[0] + dx, DC[1] + dy, table + h]
                        for dx in (0.0, 0.02) for dy in (-0.02, 0.0, 0.02) for h in (0.002, 0.006)])
    np.testing.assert_allclose(lt.distance(under_d), lk.distance(under_d), atol=1e-9)
    assert np.all(lt.distance(under_d) <= le.distance(under_d) + 1e-12)
    # (c) target 아래 공 둘레 ring (본 합성 d < 0 → 미세 f 가 답한다) — distractor 반대쪽은 exclude 와 같다
    th = np.linspace(0.5 * np.pi, 1.5 * np.pi, 5)
    ring = np.array([[TC[0] + 0.034 * np.cos(a), TC[1] + 0.034 * np.sin(a), table + h]
                     for a in th for h in (0.004, 0.006, 0.008)])
    nt_, _ = A8._composite(lt, table=table)
    ne_, _ = A8._composite(le, table=table)
    np.testing.assert_allclose(nt_.target_free_distance(ring), ne_.target_free_distance(ring), atol=1e-9)
    np.testing.assert_allclose(nt_.target_free_distance(ring), ring[:, 2] - table, atol=1e-6)


def test_overlap_seeds_inside_the_ball_are_protected(monkeypatch):
    """공 안 · 두 발자국 겹침 seed 는 공 규칙 (guard 가 공 z_min 으로 내려갔다) 도 지우지 않는다.

    공유 판정 (non-target 점 3D 18 mm 안) 으로 이미 남는 것과 구별하려고, 공 위쪽 50 mm 에 매달린 non-target
    관측 (예: 이웃 물체의 윗부분) 을 둔다 — 평면 거리 10 mm (발자국 안), 3D 거리 > 18 mm (공유 아님).
    """
    ev, table = _raised_table(monkeypatch)
    hang = np.array([[0.55, 0.50 + dy, table + 0.061] for dy in (-0.01, 0.0, 0.01)])
    nt = np.r_[NONTARGET, hang]
    out = {}
    for mode in ("exclude", "exclude_target"):
        b = A6._builder(target_free_coarse_support=mode)
        s = b._coarse_free_spec(A6._ball(), None, nt, target_points=TARGET, support_evidence=ev)
        out[mode] = A6._tier(b, s)
    (itg_e, e), (itg_t, t) = out["exclude"], out["exclude_target"]
    ex = t.extra["exclusion"]
    assert ex["n_support_protected_in_ball"] > 0
    kept = {tuple(v) for v in t.extra["exclusion_trace"]["support_overlap_kept_ijk"]}
    seeds_t, seeds_e = _seed_set(itg_t), _seed_set(itg_e)
    assert kept <= seeds_t
    ball = A6._ball()
    in_ball = {v for v in kept if np.linalg.norm(BC + np.asarray(v) * VC - ball.centre) <= ball.radius
               and BC[2] + v[2] * VC >= ball.z_min}
    assert in_ball and not in_ball <= seeds_e, "exclude 는 공 안의 그 seed 를 지운다"
    assert not ({tuple(v) for v in t.extra["exclusion_trace"]["removed_ijk"]} & kept)


# ---------------------------------------------------------------------------- 5 h_min


def test_low_nontarget_points_are_not_footprint_evidence():
    """평면 위 h_min (8 mm) 이하 non-target 점 (inlier 에 못 든 테이블 잡음) 은 발자국을 만들지 않는다."""
    rng = np.random.default_rng(4)
    ang = rng.uniform(0, 2 * np.pi, 400)
    rad = rng.uniform(RT + 0.005, RT + 0.06, 400)
    noise = np.c_[TC[0] + rad * np.cos(ang), TC[1] + rad * np.sin(ang), np.full(400, A6.TABLE + 0.004)]
    spec = dataclasses.replace(CoarseFreeSpec(), support="exclude_target", support_points=A8.SUPPORT,
                               support_plane=A8.PLANE, nontarget_points=noise, target_points=TARGET)
    q = np.array([[TC[0] - 0.06, TC[1] + dy, 0.86] for dy in (-0.02, 0.0, 0.02)])
    p = {"footprint_dilation_m": D, "footprint_min_height_m": COARSE_SUPPORT_FOOTPRINT_MIN_HEIGHT_M}
    in_t, in_n, rec = CB.CuroboFieldBuilder._coarse_support_footprints(q, spec, p, A6._ball())
    assert in_t.all() and not in_n.any() and rec["n_nontarget_points"] == 0
    in_t0, in_n0, _ = CB.CuroboFieldBuilder._coarse_support_footprints(
        q, spec, {**p, "footprint_min_height_m": 0.0}, A6._ball())
    assert in_n0.all(), "h_min 0 이면 잡음이 target 둘레를 non-target 발자국으로 덮는다"


# ---------------------------------------------------------------------------- 6 properties


def test_bracket_property_and_distractor_interior():
    _, t, _ = _tier("exclude_target")
    lt = A8._layer(t)
    new, fo = A8._composite(lt)
    old, _ = A8._composite(None)
    pts = np.random.default_rng(7).uniform(fo + 0.01, fo + 0.15, (5000, 3))
    n, o, d = new.target_free_distance(pts), old.target_free_distance(pts), new.distance(pts)
    assert np.all(np.minimum(o, d) <= n + 1e-12) and np.all(n <= o + 1e-12)
    dist_in = np.array([DC + [0.012, 0.0, 0.0], DC + [0.0, 0.0, 0.01], DC + [0.0, 0.0, -0.02]])
    assert np.all(lt.distance(dist_in) < 0.0)
    assert np.all(new.target_free_distance(dist_in) < 0.0)


def test_footprints_project_onto_a_tilted_plane():
    n = np.array([0.0, np.sin(0.2), np.cos(0.2)])
    off = 0.5
    o = n * off                                            # 평면 위 한 점
    u = np.array([1.0, 0.0, 0.0])
    spec = CoarseFreeSpec(support="exclude_target", support_points=o[None], support_plane=np.r_[n, off],
                          nontarget_points=np.array([o + 0.3 * u + 0.05 * n,       # 위 50 mm
                                                     o - 0.3 * u + 0.004 * n]),   # 위 4 mm (증거 아님)
                          target_points=np.array([o + 0.10 * n]))
    params = {"footprint_dilation_m": 0.03, "footprint_min_height_m": 0.008}
    ball = types.SimpleNamespace(centre=o + 0.04 * n, radius=0.05)
    q = np.array([o + 0.07 * u - 0.01 * n,                 # 공 원판 + 0.02 안 (법선 방향 높이 무관)
                  o + 0.09 * u + 0.01 * n,                 # 밖 (0.09 > 0.05 + 0.03)
                  o + 0.28 * u, o + 0.33 * u - 0.01 * n,   # non-target 위 50 mm 점의 발자국 (target 밖)
                  o - 0.29 * u])
    in_t, in_n, rec = CB.CuroboFieldBuilder._coarse_support_footprints(q, spec, params, ball)
    assert in_t.tolist() == [True, False, False, False, False]
    assert not in_n.any(), "non-target 은 target 발자국 안 seed 에서만 본다"
    spec2 = dataclasses.replace(spec, nontarget_points=np.array([o + 0.09 * u + 0.05 * n]))
    in_t2, in_n2, _ = CB.CuroboFieldBuilder._coarse_support_footprints(q, spec2, params, ball)
    assert in_n2.tolist() == [True, False, False, False, False]
    assert rec["n_target_points"] == 1


# ---------------------------------------------------------------------------- 7 wiring


def test_spec_fields_config_and_cli():
    ev = {**A8.EVIDENCE, "inlier_threshold_m": 0.006}
    f = coarse_support_spec_fields("exclude_target", ev)
    assert f["support"] == "exclude_target" and f["support_min_height_m"] == 0.006
    assert "support_min_height_m" not in coarse_support_spec_fields("exclude", ev)
    assert coarse_support_spec_fields("exclude_target", {"points": A8.SUPPORT}) == {}
    assert coarse_support_spec_fields("keep", ev) == {}
    f0 = coarse_support_spec_fields("exclude_target", A8.EVIDENCE)
    assert "support_min_height_m" not in f0
    with pytest.raises(ValueError):
        CoarseFreeSpec(support="exclude_target")
    cfg = AG3SConfig.from_dict({"esdf": {"target_free_coarse": True,
                                         "target_free_coarse_support": "exclude_target"}})
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf.target_free_coarse_support == "exclude_target"
    from benchmark.trajopt.serve_safe import target_free_coarse_support_section

    sec = target_free_coarse_support_section(
        A6._parse("--target-free-coarse", "--target-free-coarse-support", "exclude_target"))
    assert sec == {"target_free_coarse_support": "exclude_target"}


def test_builder_records_fail_closed_for_exclude_target():
    b = A6._builder(target_free_coarse_support="exclude_target")
    assert b._coarse_support_fields(None) == {}
    assert A8._spec(b, evidence=None).support == "keep"


def test_harness_passes_exclude_target(monkeypatch):
    from benchmark.ag3s.experiments.live import exclusion_e2 as H

    seen = {}

    class _Mapper:
        integrator = object()
        _device = "cpu"

    class _B:
        _mapper = _Mapper()
        _fine_mapper = None

        def _coarse_free_spec(self, ball, owned, nontarget, target_points=None, **kw):
            return CoarseFreeSpec(nontarget_points=nontarget, target_points=target_points)

        def _build_tiers(self, itg, att, dev, torch_, **kw):
            seen["spec"] = kw["coarse_free"]
            return ("tier",)

    monkeypatch.setattr(H, "_sync", lambda: None)
    H.build_coarse_free_tier(_B(), A6._ball(), nontarget=NONTARGET, support="exclude_target",
                             support_evidence={**A8.EVIDENCE, "inlier_threshold_m": 0.007})
    assert seen["spec"].support == "exclude_target" and seen["spec"].support_min_height_m == 0.007
    H.build_coarse_free_tier(_B(), A6._ball(), nontarget=NONTARGET, support="exclude_target")
    assert seen["spec"].support == "keep"
    assert H._parser().parse_args(["--coarse-support", "exclude_target"]).coarse_support == "exclude_target"


def test_pipeline_adds_inlier_threshold_only_for_exclude_target(monkeypatch):
    fw = pytest.importorskip("tests.ag3s.test_fine_window")
    nt = {"points": np.zeros((4, 3)), "source": "indices", "n_cloud": 9,
          "n_target_dropped": 3, "n_support_dropped": 2}
    cloud = np.random.default_rng(0).uniform(0, 1, (5, 3))
    ag3s = fw._ag3s(monkeypatch, target_free_coarse=True, target_free_coarse_support="exclude_target")
    fw._build(ag3s, nontarget=nt, support_cloud=cloud)
    ev = ag3s._esdf_builder.calls[-1]["support_seed_evidence"]
    assert ev["inlier_threshold_m"] == ag3s.config.support_surface.distance_threshold == 0.008
    np.testing.assert_array_equal(ev["points"], cloud[fw.TABLE.point_indices])
    ag3s = fw._ag3s(monkeypatch, target_free_coarse=True, target_free_coarse_support="exclude")
    fw._build(ag3s, nontarget=nt, support_cloud=cloud)
    assert "inlier_threshold_m" not in ag3s._esdf_builder.calls[-1]["support_seed_evidence"]


# ============================================================================ 8 GPU


def test_curobo_exclude_target_keeps_the_neighbour_footprint():
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    pytest.importorskip("curobo")
    G = pytest.importorskip("tests.o4.test_gpu_esdf_field")
    from benchmark.ag3s.fields.curobo_builder import CuroboFieldBuilder, FineWindowRequest, TargetBall
    from benchmark.ag3s.fields.curobo_field import CuroboEsdfField

    cams = G._scene_cameras()
    apple = np.random.default_rng(0).normal([0.51, -0.03, 0.85], 0.012, (300, 3))
    other = np.random.default_rng(3).normal([0.51, 0.03, 0.85], 0.012, (300, 3))
    ball = TargetBall(centre=[0.51, -0.03, 0.85], radius=0.045, z_min=0.825,
                      provenance={"support_plane_z_m": 0.82, "pad_mm": 5.0})
    xy = np.stack(np.meshgrid(np.arange(0.30, 0.75, 0.005), np.arange(-0.25, 0.25, 0.005),
                              indexing="ij"), -1).reshape(-1, 2)
    ev = {"points": np.c_[xy, np.full(len(xy), 0.82)], "plane": np.array([0.0, 0.0, 1.0, 0.82]),
          "inlier_threshold_m": 0.008}
    kw = dict(target_points=apple, target_free_points=apple, target_free_ball=ball,
              fine_window=FineWindowRequest(manipulated_points=apple, placement="target_centroid"),
              nontarget_points=other, support_seed_evidence=ev)
    fields = {}
    for mode in ("keep", "exclude", "exclude_target"):
        b = CuroboFieldBuilder(G._config(target_free_coarse=True, target_free_coarse_support=mode))
        for _ in range(2):
            fields[mode] = b.update(cams, **kw)
    k, e, t = fields["keep"], fields["exclude"], fields["exclude_target"]
    for lk, lt in zip(k.layers, t.layers):
        assert torch.equal(lk.values_device, lt.values_device)
    assert torch.equal(k.target_free_layers[0].values_device, t.target_free_layers[0].values_device)
    st = t.stats["target_free"]["coarse"]["support"]
    se = e.stats["target_free"]["coarse"]["support"]
    assert st["mode"] == "exclude_target" and st["applied"]
    assert 0 < st["n_removed"] < se["n_removed"]
    assert "footprint" in st and "footprint" not in se
    old = CuroboEsdfField(tuple(t.layers), outside_distance=t.outside_distance,
                          target_free_layers=tuple(t.target_free_layers), stats={})
    pts = np.random.default_rng(5).normal([0.51, 0.0, 0.86], 0.04, (2000, 3))
    nv, ov, d = t.target_free_distance(pts), old.target_free_distance(pts), t.distance(pts)
    assert np.all(np.minimum(ov, d) <= nv + 1e-6) and np.all(nv <= ov + 1e-6)
