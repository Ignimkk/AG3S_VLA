"""T44c A6 — `coarse_no_target` 의 부호 교정 아래 한계 (`esdf.target_free_coarse_sign_floor`).

A5 는 거친 계층의 seed 제거 guard (지원면 + max(여유, 거친 voxel) = +20 mm) 와 같은 높이에서 부호 교정도
끊었다. 그 아래의 target 바닥 내부 (seed 가 아닌 음수 voxel) 가 장애물로 남았다 (`T44c.verify.json`:
target 내부 음수 11,180 / 67,361, 그중 11,057 이 지원면 + 20 mm 아래). `"plane"` 은 seed 는 그대로 두고
부호 교정만 공의 `z_min` (미세 계층과 같은 m 값) 까지 내린다.

## 무엇을 박나

| # | 성질 |
|---|---|
| 1 | `"guard"` (A7 부터 선택지, 기본은 `"plane"`) = A5 와 비트 동일 — 같은 seed · 같은 부호 집합 · 같은 값 · params/counts 키 그대로 |
| 2 | `"plane"`: seed 는 하나도 더 안 지운다 (테이블 seed 보존) — `site_index` 가 guard 와 같다 |
| 3 | `"plane"`: 값은 guard 값에서 **부호만** 바뀐다 (`|plane| = |guard|`, `plane ≥ guard`), 바뀐 voxel 은 전부 공 안 · 공의 z_min 위 · guard 아래 띠 |
| 4 | 지원면 아래 (테이블 내부) 의 음수 · 띠의 seed · 공유 영역 · non-target 쪽 voxel 은 그대로 |
| 5 | target 바닥 내부 점: guard 음수 → plane 양수 · distractor 내부 음수 그대로 |
| 6 | 합성 성질 `min(old, d) ≤ new ≤ old` 유지 (plane 계층으로) |
| 7 | `ball_owned` 의 띠 = 그 분류의 non-seed 규칙 (`d_t < d_o − tie`) |
| 8 | config 왕복 · validation · CLI (`--target-free-coarse-sign-floor`, `--target-free-coarse` 없이 거부) · spec 배선 |
| 9 | (GPU) 실제 cuRobo: plane 은 본 계층 · 미세 target-free 를 안 바꾸고, 거친 target-free 의 부호만 바꾼다 |
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from benchmark.ag3s.config import AG3SConfig, AG3SConfigError
from benchmark.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays

torch = pytest.importorskip("torch")

from benchmark.ag3s.fields.curobo_builder import (  # noqa: E402
    COARSE_SIGN_FLOORS,
    CoarseFreeSpec,
    CuroboFieldBuilder,
    OwnedEvidence,
    TargetBall,
    coarse_exclusion_params,
)

VC, VF = 0.02, 0.005
BC = np.array([0.30, 0.30, 0.80])                  # 거친 가짜 격자의 첫 voxel 중심 (z 0.80, 0.82, …)
BN = (24, 24, 24)
TABLE = 0.861                                      # 테이블 윗면: z 0.86 층이 테이블 seed
TC = np.array([0.52, 0.50, TABLE + 0.045])         # target (r 45 mm, 바닥이 테이블에 닿는다)
RT = 0.045
DC = np.array([0.60, 0.50, TABLE + 0.030])         # distractor (r 30 mm, 표면 간격 5 mm)
RD = 0.030


def _sdf(p):
    a = np.linalg.norm(p - TC, axis=-1) - RT
    b = np.linalg.norm(p - DC, axis=-1) - RD
    return np.minimum(np.minimum(a, b), p[..., 2] - TABLE)


def _grid(origin, shape, vs):
    ax = [origin[i] + np.arange(shape[i]) * vs for i in range(3)]
    return np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)


class _VG:
    def __init__(self, t):
        self.feature_tensor = t


class _Itg:
    """cuRobo 흉내: seed `|sdf| ≤ 0.9 vs` → propagate (남은 seed 최근접) → 부호 (그 voxel 의 sdf)."""

    def __init__(self):
        self.shape = BN
        self._esdf_voxel_size = torch.tensor([VC])
        self._origin = torch.tensor(BC + 0.5 * (np.asarray(BN) - 1) * VC, dtype=torch.float32)
        self._last_esdf_origin = self._origin.clone()
        self._site_index = torch.full(BN, -1, dtype=torch.int32)
        self._dist = torch.zeros(BN, dtype=torch.float32)

    def _g(self):
        vs = float(self._esdf_voxel_size[0])
        c = self._last_esdf_origin.numpy().astype(np.float64)
        return _grid(c - 0.5 * np.asarray(self.shape) * vs + 0.5 * vs, self.shape, vs)

    def _seed_esdf_impl(self, origin, vs):
        v = float(self._esdf_voxel_size[0])
        seed = np.abs(_sdf(self._g())) <= 0.9 * v
        ii, jj, kk = np.nonzero(seed)
        packed = np.full(self.shape, -1, np.int64)
        packed[ii, jj, kk] = (kk << 20) | (jj << 10) | ii
        self._site_index.copy_(torch.as_tensor(packed.astype(np.int32)))

    def _propagate_and_distance_impl(self, origin, vs):
        from scipy.spatial import cKDTree

        g = self._g()
        seeds = np.argwhere(self._site_index.numpy() >= 0)
        d, _ = cKDTree(g[seeds[:, 0], seeds[:, 1], seeds[:, 2]]).query(g.reshape(-1, 3), k=1)
        val = np.where(_sdf(g).reshape(-1) < 0.0, -d, d).reshape(self.shape)
        self._dist.copy_(torch.as_tensor(val.astype(np.float32)))

    def get_voxel_grid(self):
        return _VG(self._dist)


def _builder(**esdf):
    cfg = AG3SConfig.from_dict({"esdf": {"backend": "curobo", "voxel_size": VC,
                                         "fine_voxel_size": VF, **esdf}}).esdf
    return CuroboFieldBuilder(cfg)


def _ball():
    return TargetBall(centre=TC, radius=0.05, z_min=TABLE + 0.005,
                      provenance={"support_plane_z_m": TABLE, "pad_mm": 5.0})


def _sphere_pts(c, r, n, seed, *, upper_only=False):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(4 * n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    if upper_only:
        v = v[v[:, 2] > -0.3]
    return c + r * v[:n]


NONTARGET = _sphere_pts(DC, RD, 400, 1)
TARGET = _sphere_pts(TC, RT, 300, 0, upper_only=True)


def _spec(b, floor=None, *, owned=None, target=TARGET, nontarget=NONTARGET):
    s = b._coarse_free_spec(_ball(), owned, nontarget, target_points=target)
    return s if floor is None else dataclasses.replace(s, sign_floor=floor)


def _tier(b, spec):
    itg = _Itg()
    (t,) = b._build_tiers(itg, None, "cpu", torch, only="coarse", tier_name="coarse_no_target",
                          ball=_ball(), coarse_free=spec)
    return itg, t


def _values(t):
    return t.values.numpy().astype(np.float64)


# ---------------------------------------------------------------------------- 1 guard = A5


def test_guard_option_is_the_a5_tier_bit_for_bit():
    # T44c A7: 기본은 plane — guard 는 명시해야 한다 (config 로든 spec 으로든).
    b = _builder(target_free_coarse_sign_floor="guard")
    assert b.config.target_free_coarse_sign_floor == "guard"
    assert _spec(b).sign_floor == "guard"
    # A5 의 spec 은 target 점을 몰랐다 — guard 에서 target 점은 아무것도 바꾸지 않는다.
    itg_a5, a5 = _tier(b, dataclasses.replace(_spec(b), target_points=None))
    itg_g, g = _tier(b, _spec(b))
    assert torch.equal(a5.values, g.values)
    assert torch.equal(itg_a5._site_index, itg_g._site_index)
    strip = lambda e: {k: v for k, v in e.items() if k != "classify_ms"}  # noqa: E731 (시간)
    assert strip(a5.extra["exclusion"]) == strip(g.extra["exclusion"])
    assert "plane_band" not in g.extra["exclusion"]
    assert "sign_z_min_m" not in coarse_exclusion_params(_ball(), VC, _spec(b))
    assert a5.extra["n_sign_forced"] == g.extra["n_sign_forced"]


# ---------------------------------------------------------------------------- 2–5 plane


def test_plane_removes_no_more_seeds_than_guard():
    b = _builder()
    itg_g, g = _tier(b, _spec(b, "guard"))
    itg_p, p = _tier(b, _spec(b, "plane"))
    assert torch.equal(itg_g._site_index, itg_p._site_index), "테이블 seed 를 지우지 않는다"
    eg, ep = g.extra["exclusion"], p.extra["exclusion"]
    for k in ("n_seeds_in_ball", "n_removed", "n_shared_kept", "n_seeds_shared"):
        assert eg[k] == ep[k]
    np.testing.assert_array_equal(g.extra["exclusion_trace"]["removed_ijk"],
                                  p.extra["exclusion_trace"]["removed_ijk"])


def test_plane_only_flips_signs_inside_the_band():
    b = _builder()
    _, g = _tier(b, _spec(b, "guard"))
    _, p = _tier(b, _spec(b, "plane"))
    vg, vp = _values(g), _values(p)
    np.testing.assert_array_equal(np.abs(vg), np.abs(vp))
    assert np.all(vp >= vg)
    changed = np.argwhere(vp != vg)
    assert len(changed) > 0, "띠에 target 바닥 내부 voxel 이 있다"
    c = BC + changed * VC
    params = coarse_exclusion_params(_ball(), VC, _spec(b, "plane"))
    assert params["sign_z_min_m"] == pytest.approx(TABLE + 0.005)
    assert params["z_min_m"] == pytest.approx(TABLE + VC)
    assert np.all(c[:, 2] >= params["sign_z_min_m"] - 1e-9)
    assert np.all(c[:, 2] < params["z_min_m"])
    assert np.all(np.linalg.norm(c - TC, axis=1) <= _ball().radius + 1e-9)
    band = p.extra["exclusion"]["plane_band"]
    assert band["n_plane_band_sign"] >= len(changed)
    assert p.extra["n_sign_forced"] == g.extra["n_sign_forced"] + len(changed)


def test_table_interior_band_seeds_shared_and_nontarget_side_are_left_alone():
    b = _builder()
    _, g = _tier(b, _spec(b, "guard"))
    _, p = _tier(b, _spec(b, "plane"))
    vg, vp = _values(g), _values(p)
    cen = _grid(BC, BN, VC)
    below = cen[..., 2] < TABLE
    np.testing.assert_array_equal(vg[below], vp[below])
    assert np.any(vp[below] < 0), "테이블 내부 음수가 남는다"
    from scipy.spatial import cKDTree

    d_o = cKDTree(NONTARGET).query(cen.reshape(-1, 3))[0].reshape(BN)
    near_nt = d_o <= 0.9 * VC
    np.testing.assert_array_equal(vg[near_nt], vp[near_nt])
    seeds = np.abs(_sdf(cen)) <= 0.9 * VC
    np.testing.assert_array_equal(vg[seeds], vp[seeds])
    # target 쪽 판정: target 점을 멀리 두면 띠의 모든 non-seed 가 non-target 쪽 (d_o < d_t) 이 된다
    # → 띠에서 아무것도 안 바뀐다 (guard 와 같다).
    far_target = TARGET + np.array([0.0, 0.0, 0.5])
    _, p_far = _tier(b, _spec(b, "plane", target=far_target))
    np.testing.assert_array_equal(_values(p_far), vg)
    band = p_far.extra["exclusion"]["plane_band"]
    assert band["n_plane_band_sign"] == 0 and band["n_plane_band_nontarget_side"] > 0


def test_target_bottom_interior_turns_nonnegative_and_distractor_stays_negative():
    b = _builder()
    _, g = _tier(b, _spec(b, "guard"))
    _, p = _tier(b, _spec(b, "plane"))
    lg = layer_from_arrays(_values(g), g.origin, VC)
    lp = layer_from_arrays(_values(p), p.origin, VC)
    bottom = np.array([[TC[0], TC[1], TABLE + 0.019], [TC[0], TC[1], TABLE + 0.021]])
    assert np.any(lg.distance(bottom) < 0.0), "A5: target 바닥 내부가 음수로 남는다"
    assert np.all(lp.distance(bottom) >= 0.0)
    dist_in = np.array([DC + [0.012, 0.0, 0.0]])
    assert lp.distance(dist_in)[0] < 0.0
    assert lp.distance(dist_in)[0] == lg.distance(dist_in)[0]


# ---------------------------------------------------------------------------- 6 합성 성질


def test_composition_stays_bracketed_with_the_plane_tier():
    b = _builder()
    _, p = _tier(b, _spec(b, "plane"))
    cen = _grid(BC, BN, VC)
    main = layer_from_arrays(_sdf(cen).astype(np.float64), BC, VC)
    fo = TC - 0.08
    fine = layer_from_arrays(np.abs(_sdf(_grid(fo, (33, 33, 33), VF))), fo, VF)
    cf = layer_from_arrays(_values(p), p.origin, VC)
    new = CuroboEsdfField((main,), outside_distance=1.0, target_free_layers=(fine,), stats={},
                          target_free_coarse_layer=cf)
    old = CuroboEsdfField((layer_from_arrays(_sdf(cen).astype(np.float64), BC, VC),),
                          outside_distance=1.0,
                          target_free_layers=(layer_from_arrays(
                              np.abs(_sdf(_grid(fo, (33, 33, 33), VF))), fo, VF),), stats={})
    pts = np.random.default_rng(3).uniform(fo + 0.01, fo + 0.15, (4000, 3))
    n, o, d = new.target_free_distance(pts), old.target_free_distance(pts), new.distance(pts)
    assert np.all(np.minimum(o, d) <= n + 1e-12) and np.all(n <= o + 1e-12)
    v, _, _ = new._answer(pts, True, want_gradient=False)
    np.testing.assert_array_equal(v, n)


# ---------------------------------------------------------------------------- 7 ball_owned


def test_ball_owned_band_uses_its_nonseed_rule():
    b = _builder(target_exclusion="ball_owned", owned_max_dist_voxels=3.0)
    owned = OwnedEvidence(target_points=TARGET, nontarget_points=NONTARGET,
                          tie_voxels=1.0, max_dist_voxels=3.0)
    itg_g, g = _tier(b, _spec(b, "guard", owned=owned))
    itg_p, p = _tier(b, _spec(b, "plane", owned=owned))
    assert torch.equal(itg_g._site_index, itg_p._site_index)
    vg, vp = _values(g), _values(p)
    np.testing.assert_array_equal(np.abs(vg), np.abs(vp))
    ch = np.argwhere(vp != vg)
    assert len(ch) > 0
    from scipy.spatial import cKDTree

    c = BC + ch * VC
    d_t = cKDTree(TARGET).query(c)[0]
    d_o = cKDTree(NONTARGET).query(c)[0]
    assert np.all(d_t < d_o - 0.005), "ball_owned: tie 밖에서 target 쪽만"
    assert p.extra["exclusion"]["exclusion_method"] == "ball_owned"


# ---------------------------------------------------------------------------- 8 배선


def test_spec_rejects_unknown_floor_and_lists_choices():
    assert COARSE_SIGN_FLOORS == ("guard", "plane")
    with pytest.raises(ValueError):
        CoarseFreeSpec(sign_floor="nope")


def test_config_round_trip_and_validation():
    cfg = AG3SConfig.from_dict({"esdf": {"target_free_coarse": True,
                                         "target_free_coarse_sign_floor": "plane"}})
    assert cfg.esdf.target_free_coarse_sign_floor == "plane"
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf.target_free_coarse_sign_floor == "plane"
    assert AG3SConfig.from_dict({}).esdf.target_free_coarse_sign_floor == "plane"   # T44c A7
    g = AG3SConfig.from_dict({"esdf": {"target_free_coarse": True,
                                       "target_free_coarse_sign_floor": "guard"}})
    assert AG3SConfig.from_dict(g.to_dict()).esdf.target_free_coarse_sign_floor == "guard"
    assert AG3SConfig.from_dict({}).esdf.TARGET_FREE_COARSE_SIGN_FLOORS == COARSE_SIGN_FLOORS
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"esdf": {"target_free_coarse_sign_floor": "below"}})


def test_builder_spec_takes_the_config_floor_and_target_points():
    b = _builder(target_free_coarse_sign_floor="plane")
    s = b._coarse_free_spec(_ball(), None, NONTARGET, target_points=TARGET)
    assert s.sign_floor == "plane" and s.target_points.shape == TARGET.shape
    assert _builder()._coarse_free_spec(_ball(), None, NONTARGET).target_points is None
    assert _builder()._coarse_free_spec(_ball(), None, NONTARGET).sign_floor == "plane"  # T44c A7


def _parse(*argv):
    from benchmark.trajopt.serve_safe import build_parser

    return build_parser().parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml",
                                      *argv])


def test_cli_flag_maps_to_the_esdf_key_and_defaults_to_nothing():
    from benchmark.trajopt.serve_safe import target_free_coarse_sign_section

    assert target_free_coarse_sign_section(_parse()) == {}
    sec = target_free_coarse_sign_section(
        _parse("--target-free-coarse", "--target-free-coarse-sign-floor", "plane"))
    assert sec == {"target_free_coarse_sign_floor": "plane"}
    assert AG3SConfig.from_dict({"esdf": sec}).esdf.target_free_coarse_sign_floor == "plane"


def test_cli_rejects_the_floor_without_target_free_coarse(monkeypatch, capsys):
    import sys

    from benchmark.trajopt import serve_safe

    monkeypatch.setattr(sys, "argv", ["serve_safe", "--config", "c", "--checkpoint", "k",
                                      "--model-xml", "x.xml", "--esdf-backend", "curobo",
                                      "--target-field-policy", "exclude-authorized",
                                      "--target-free-coarse-sign-floor", "plane"])
    with pytest.raises(SystemExit) as e:
        serve_safe.main()
    assert e.value.code == 2
    assert "--target-free-coarse-sign-floor" in capsys.readouterr().err


def test_harness_passes_target_points_and_floor(monkeypatch):
    from benchmark.ag3s.experiments.live import exclusion_e2 as H

    seen = {}

    class _Mapper:
        integrator = object()
        _device = "cpu"

    class _B:
        _mapper = _Mapper()
        _fine_mapper = None

        def _coarse_free_spec(self, ball, owned, nontarget, target_points=None):
            seen["target"] = target_points
            return CoarseFreeSpec(nontarget_points=nontarget, target_points=target_points)

        def _build_tiers(self, itg, att, dev, torch_, **kw):
            seen["spec"] = kw["coarse_free"]
            return ("tier",)

    monkeypatch.setattr(H, "_sync", lambda: None)
    tier, _ = H.build_coarse_free_tier(_B(), _ball(), nontarget=NONTARGET, target=TARGET,
                                       sign_floor="plane")
    assert tier == "tier"
    assert seen["target"] is TARGET and seen["spec"].sign_floor == "plane"
    H.build_coarse_free_tier(_B(), _ball(), nontarget=NONTARGET)
    assert seen["spec"].sign_floor == "plane"            # T44c A7: spec 기본 = plane
    H.build_coarse_free_tier(_B(), _ball(), nontarget=NONTARGET, sign_floor="guard")
    assert seen["spec"].sign_floor == "guard"


# ============================================================================ 9 GPU (real cuRobo; skipped on CPU)


def test_curobo_plane_floor_only_flips_coarse_free_signs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    pytest.importorskip("curobo")
    G = pytest.importorskip("tests.o4.test_gpu_esdf_field")
    from benchmark.ag3s.fields.curobo_builder import FineWindowRequest

    cams = G._scene_cameras()
    apple = np.random.default_rng(0).normal([0.51, -0.03, 0.85], 0.012, (300, 3))
    other = np.random.default_rng(3).normal([0.51, 0.03, 0.85], 0.012, (300, 3))
    ball = TargetBall(centre=[0.51, -0.03, 0.85], radius=0.045, z_min=0.825,
                      provenance={"support_plane_z_m": 0.82, "pad_mm": 5.0})
    kw = dict(target_points=apple, target_free_points=apple, target_free_ball=ball,
              fine_window=FineWindowRequest(manipulated_points=apple, placement="target_centroid"),
              nontarget_points=other)
    fields = {}
    for name, over in (("off", {}), ("guard", {"target_free_coarse": True,
                                               "target_free_coarse_sign_floor": "guard"}),
                       ("plane", {"target_free_coarse": True})):            # T44c A7: 기본 plane
        b = CuroboFieldBuilder(G._config(**over))
        for _ in range(2):
            fields[name] = b.update(cams, **kw)
    off, g, p = fields["off"], fields["guard"], fields["plane"]
    for la, lg, lp in zip(off.layers, g.layers, p.layers):
        assert torch.equal(la.values_device, lp.values_device)
        assert torch.equal(lg.values_device, lp.values_device)
    assert torch.equal(off.target_free_layers[0].values_device, p.target_free_layers[0].values_device)
    cg, cp = g.target_free_coarse_layer.values_device, p.target_free_coarse_layer.values_device
    assert torch.equal(cg.abs(), cp.abs()), "seed 는 같다 — 부호만 바뀐다"
    assert bool((cp >= cg).all())
    sp = p.stats["target_free"]["coarse"]
    assert "plane_band" in sp and "plane_band" not in g.stats["target_free"]["coarse"]
    assert sp["n_removed"] == g.stats["target_free"]["coarse"]["n_removed"]
    pts = np.random.default_rng(5).normal([0.51, 0.0, 0.86], 0.04, (2000, 3))
    d, n, o = p.distance(pts), p.target_free_distance(pts), off.target_free_distance(pts)
    assert np.all(np.minimum(o, d) <= n + 1e-9) and np.all(n <= o + 1e-9)
