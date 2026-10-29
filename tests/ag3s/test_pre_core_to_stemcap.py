"""AG3S-PRE-CORE-TO-STEMCAP — 꼭지 덮개 (stem cap): `ball_owned` 미세 target-free 계층에서 공 위 좁은 원기둥 안 seed 도 지운다.

## 무엇을 박나

| # | 성질 |
|---|---|
| 1 | `stem_cap_mask` — 원기둥 안 (수평 ≤ 20 mm · c_z + 0.5 fit_r ≤ z ≤ c_z + fit_r + 25 mm) 만, 경계 · 반경 밖 · 아래 · 위 · `in_ball` |
| 2 | off (기본) = 명시 off: 값 · exclusion 수 · trace 키 · 기록 payload 비트 동일, `stem_cap` 키 없음 |
| 3 | 합성 사과 (공) + 꼭대기 위 가는 꼭지 (non-target 근거로 잘못 들어간) → `full` 이 지운다; 이웃 과일 · 공 아래 · 테이블 seed 는 남는다 |
| 4 | `in_ball` 은 공 밖 꼭지 seed 를 남기고, 지운 것 ⊂ `full` 이 지운 것 |
| 5 | 덮개가 더 지운 seed = 정확히 `stem_cap_ijk` · 그 수 = `stem_cap.n_removed` · 부호 교정 (지운 voxel 이 순수하면 양수) |
| 6 | 결정적 (두 번 같은 값 · 같은 ijk) · 맞춘 반지름이 없으면 건너뛰고 이유를 남긴다 |
| 7 | 기록: `target_exclusion.stem_cap` 블록 · npz `target_free_stem_cap_ijk` (int16) |
| 8 | config · CLI (`--target-stem-cap` · `--stem-cap-*`) 왕복 · 거절 (`ball` 과 함께 · flag 만 · oracle extent) |

가짜 integrator 는 `test_t44_target_exclusion_owned` 와 같은 방식 (seed = |sdf| < 0.75 vs, propagate = 남은 seed 최근접,
부호 = 그 voxel 의 sdf) 이고, 장면 SDF 만 이 파일의 것 (사과 · 꼭지 · 이웃 과일 · 테이블) 이다.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError  # noqa: E402
from AG3S.ag3s.fields.curobo_builder import (  # noqa: E402
    CuroboFieldBuilder,
    OwnedEvidence,
    StemCapSpec,
    TargetBall,
    stem_cap_bounds,
    stem_cap_mask,
)

VS = 0.005
N = 40
ORIGIN = np.array([0.40, 0.20, 0.78])            # 첫 voxel 중심
A = np.array([0.495, 0.300, 0.875])              # 사과 중심 (voxel 중심 위)
RA = 0.030                                       # 사과 반지름 = 맞춘 반지름
STEM_R = 0.003                                   # 꼭지 반지름
STEM_Z = (A[2] + 0.026, A[2] + 0.048)            # 꼭지 z 범위 (사과 꼭대기 A_z + 30 mm 위로 18 mm)
B = np.array([0.495, 0.355, 0.870])              # 이웃 과일 중심 (수평 55 mm 옆)
RB = 0.024                                       # → 표면 간격 1 mm, 꼭대기 A_z + 19 mm (덮개 높이 안, 반경 밖)
TABLE_Z = 0.843
BALL_R = RA + VS


def _centres(shape=(N, N, N), origin=ORIGIN, vs=VS):
    ax = [origin[i] + np.arange(shape[i]) * vs for i in range(3)]
    return np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)


def _sdf(p):
    """사과 · 꼭지 · 이웃 · 테이블의 합성 SDF 와 물체 id (0 사과, 1 꼭지, 2 이웃, 3 테이블)."""
    da = np.linalg.norm(p - A, axis=-1) - RA
    rxy = np.linalg.norm(p[..., :2] - A[:2], axis=-1)
    ds = np.maximum(rxy - STEM_R, np.maximum(STEM_Z[0] - p[..., 2], p[..., 2] - STEM_Z[1]))
    db = np.linalg.norm(p - B, axis=-1) - RB
    dt = p[..., 2] - TABLE_Z
    stack = np.stack([da, ds, db, dt], axis=-1)
    return stack.min(axis=-1), stack.argmin(axis=-1)


class _VoxelGrid:
    def __init__(self, t):
        self.feature_tensor = t


class _FakeItg:
    """`test_t44_target_exclusion_owned._FakeItg` 와 같은 계약 (이 파일의 `_sdf`)."""

    def __init__(self, shape=(N, N, N)):
        self.shape = shape
        self._esdf_voxel_size = torch.zeros(1)
        centre = ORIGIN + 0.5 * (np.asarray(shape) - 1) * VS
        self._origin = torch.tensor(centre, dtype=torch.float32)
        self._last_esdf_origin = torch.zeros(3)
        self._site_index = torch.full(shape, -1, dtype=torch.int32)
        self._dist = torch.zeros(shape, dtype=torch.float32)

    def _grid(self):
        vs = float(self._esdf_voxel_size[0])
        centre = self._last_esdf_origin.numpy().astype(np.float64)
        org = centre - 0.5 * np.asarray(self.shape) * vs + 0.5 * vs
        return _centres(self.shape, org, vs)

    def _seed_esdf_impl(self, origin, vs):
        sdf, _ = _sdf(self._grid())
        ii, jj, kk = np.nonzero(np.abs(sdf) < 0.75 * VS)
        packed = np.full(self.shape, -1, np.int64)
        packed[ii, jj, kk] = (kk << 20) | (jj << 10) | ii
        self._site_index.copy_(torch.as_tensor(packed.astype(np.int32)))

    def _propagate_and_distance_impl(self, origin, vs):
        from scipy.spatial import cKDTree

        g = self._grid()
        seeds = np.argwhere(self._site_index.numpy() >= 0)
        d, nn = cKDTree(g[seeds[:, 0], seeds[:, 1], seeds[:, 2]]).query(g.reshape(-1, 3), k=1)
        sdf, _ = _sdf(g)
        val = np.where(sdf.reshape(-1) < 0.0, -d, d).reshape(self.shape).astype(np.float32)
        self._dist.copy_(torch.as_tensor(val))
        s = seeds[nn]
        packed = ((s[:, 2] << 20) | (s[:, 1] << 10) | s[:, 0]).reshape(self.shape)
        self._site_index.copy_(torch.as_tensor(packed.astype(np.int32)))

    def get_voxel_grid(self):
        return _VoxelGrid(self._dist)


def _ball(fit_radius_mm=RA * 1e3):
    prov = {"centre_method": "surface_fit"}
    if fit_radius_mm is not None:
        prov["fit"] = {"method": "surface_fit", "radius_mm": fit_radius_mm}
    return TargetBall(centre=A, radius=BALL_R, z_min=TABLE_Z + VS, provenance=prov)


def _sphere_points(c, r, n, seed, zmin=-np.inf):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(6 * n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    p = c + r * v
    return p[p[:, 2] > zmin][:n]


def _stem_points(n=200, seed=3):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(STEM_Z[0] + 0.004, STEM_Z[1], n)
    return np.stack([A[0] + STEM_R * np.cos(t), A[1] + STEM_R * np.sin(t), z], axis=1)


def _owned():
    """target 점 = 사과 위쪽 표면, non-target 근거 = 이웃 과일 + **꼭지** (꼭지가 non-target 으로 잘못 간 실제 실패 모양)."""
    target = _sphere_points(A, RA, 300, 0, zmin=A[2] - 0.3 * RA)
    nontarget = np.vstack([_sphere_points(B, RB, 400, 1, zmin=TABLE_Z), _stem_points()])
    return OwnedEvidence(target_points=target, nontarget_points=nontarget)


def _builder(**esdf):
    cfg = AG3SConfig.from_dict({"esdf": {"backend": "curobo", "voxel_size": 0.02, "fine_voxel_size": VS,
                                         "target_exclusion": "ball_owned", **esdf}}).esdf
    b = CuroboFieldBuilder(cfg)
    b._fine_centre = ORIGIN + 0.5 * (N - 1) * VS
    return b


def _tier(builder, ball=None):
    (t,) = builder._build_tiers(_FakeItg(), None, "cpu", torch, only="fine", tier_name="fine_no_target",
                                ball=_ball() if ball is None else ball, owned=_owned())
    return t


def _seed_set(tier):
    """가짜 propagate 에서 seed voxel 만 값이 정확히 0 이다 (최근접 seed = 자기 자신)."""
    return {tuple(v) for v in np.argwhere(tier.values.numpy() == 0.0).tolist()}


SPEC_FULL = StemCapSpec(mode="full")
SPEC_IN = StemCapSpec(mode="in_ball")


# ============================================================================ 1 mask


def test_mask_geometry_matches_the_whatif_rule():
    ball = _ball()
    lo, hi = stem_cap_bounds(ball, SPEC_FULL, RA)
    assert lo == pytest.approx(A[2] + 0.5 * RA) and hi == pytest.approx(A[2] + RA + 0.025)
    pts = np.array([
        [A[0], A[1], A[2] + 0.034],              # 0 축 위, 공 안 → in
        [A[0] + 0.019, A[1], A[2] + 0.020],      # 1 수평 19 mm → in
        [A[0] + 0.021, A[1], A[2] + 0.020],      # 2 수평 21 mm → out (반경 밖)
        [A[0], A[1], A[2] + 0.014],              # 3 아래 끝 (15 mm) 아래 → out
        [A[0], A[1], A[2] + 0.056],              # 4 위 끝 (55 mm) 위 → out
        [A[0], A[1], A[2] + 0.050],              # 5 공 밖, 덮개 안 → full 만 in
        [A[0], A[1], A[2] - 0.020],              # 6 공 아래 → out
        [A[0], A[1] + 0.055, A[2] + 0.019],      # 7 이웃 과일 꼭대기 (수평 55 mm) → out
    ])
    full = stem_cap_mask(pts, ball, SPEC_FULL, RA)
    inb = stem_cap_mask(pts, ball, SPEC_IN, RA)
    assert full.tolist() == [True, True, False, False, False, True, False, False]
    assert inb.tolist() == [True, True, False, False, False, False, False, False]
    # parameters move the boundaries
    wide = StemCapSpec(mode="full", radius_m=0.022, top_m=0.0, bottom_fit_frac=0.0)
    assert stem_cap_mask(pts[[2, 3, 5]], ball, wide, RA).tolist() == [True, True, False]
    with pytest.raises(ValueError):
        StemCapSpec(mode="off")
    with pytest.raises(ValueError):
        StemCapSpec(mode="full", radius_m=0.0)


# ============================================================================ 2 off = bit-identical


def test_off_is_bit_identical_to_the_default_and_adds_no_keys(tmp_path):
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    default, off = _tier(_builder()), _tier(_builder(target_stem_cap="off"))
    assert AG3SConfig().esdf.target_stem_cap == "off"
    assert torch.equal(default.values, off.values)
    strip = lambda ex: {k: v for k, v in ex.items() if k != "classify_ms"}   # noqa: E731
    assert strip(default.extra["exclusion"]) == strip(off.extra["exclusion"])
    assert "stem_cap" not in off.extra["exclusion"]
    assert set(off.extra["exclusion_trace"]) == {"removed_ijk", "nontarget_kept_ijk", "ambiguous_ijk",
                                                 "unowned_kept_ijk"}
    assert "stem_cap" not in CuroboFieldBuilder._exclusion_counts([off])
    payloads = []
    for t in (default, off):
        field = types.SimpleNamespace(stats={"target_free": {"mode": "ball", **CuroboFieldBuilder._exclusion_counts([t])}},
                                      exclusion_trace=CuroboFieldBuilder._exclusion_trace([t]))
        payload = {}
        info = ConstraintRecordWriter._write_exclusion_trace(field, payload)
        payloads.append((payload, info))
    (p0, i0), (p1, i1) = payloads
    assert p0.keys() == p1.keys() and "target_free_stem_cap_ijk" not in p1
    assert all(np.array_equal(p0[k], p1[k]) for k in p0)
    i0["target_exclusion"].pop("classify_ms"), i1["target_exclusion"].pop("classify_ms")
    assert i0 == i1 and "stem_cap" not in i1["target_exclusion"]


# ============================================================================ 3-5 synthetic apple + stem


def _labels():
    g = _centres()
    _, owner = _sdf(g)
    return g, owner


def test_full_removes_the_stem_and_keeps_neighbour_below_and_table():
    off, full = _tier(_builder()), _tier(_builder(target_stem_cap="full"))
    s_off, s_full = _seed_set(off), _seed_set(full)
    cap = full.extra["exclusion_trace"]["stem_cap_ijk"]
    cap_set = {tuple(v) for v in cap.tolist()}
    sc = full.extra["exclusion"]["stem_cap"]
    # the cap removed exactly these seeds, nothing else changed in the seed set
    assert s_full <= s_off and s_off - s_full == cap_set
    assert sc["n_removed"] == len(cap) > 0 and sc["mode"] == "full" and sc["applied"] is True
    assert sc["n_cap_seeds"] == sc["n_removed"] + sc["n_already_removed"]
    # ball_owned's own record is unchanged by the cap
    for k in ("n_removed", "n_kept_nontarget", "n_ambiguous", "n_seeds_in_ball"):
        assert full.extra["exclusion"][k] == off.extra["exclusion"][k]
    assert np.array_equal(full.extra["exclusion_trace"]["removed_ijk"], off.extra["exclusion_trace"]["removed_ijk"])

    g, owner = _labels()
    # the stem: ball_owned kept stem seeds (the stem was non-target evidence) — full removes every one of them
    stem_off = {v for v in s_off if owner[v] == 1}
    assert len(stem_off) >= 5
    assert not ({v for v in s_full if owner[v] == 1}), "full 뒤에도 꼭지 seed 가 남았다"
    # every removed seed lies inside the cap; none is the neighbour, the table or below the cap
    pts = ORIGIN + cap * VS
    assert stem_cap_mask(pts, _ball(), SPEC_FULL, RA).all()
    assert {int(owner[tuple(v)]) for v in cap.tolist()} <= {0, 1}
    for lab in (2, 3):
        kept = {v for v in s_off if owner[v] == lab}
        assert kept and kept <= s_full, f"물체 {lab} 의 seed 를 덮개가 지웠다"
    below = {v for v in s_off if g[v][2] < A[2] + 0.5 * RA}
    assert below and below <= s_full
    # the neighbour fruit reaches into the cap's height band (so the radius, not z, keeps it)
    nb = [v for v in s_off if owner[v] == 2 and g[v][2] >= A[2] + 0.5 * RA]
    assert nb and all(np.linalg.norm(g[v][:2] - A[:2]) > 0.020 for v in nb)
    # sign: a removed voxel that is pure (|d| > 1.5 vs after the removal) is positive
    vals = full.values.numpy()[cap[:, 0], cap[:, 1], cap[:, 2]]
    assert np.all((vals >= 0.0) | (np.abs(vals) <= 1.5 * VS + 1e-7))
    assert sc["n_removed_outside_ball"] > 0


def test_in_ball_keeps_the_seeds_outside_the_ball():
    full, inb = _tier(_builder(target_stem_cap="full")), _tier(_builder(target_stem_cap="in_ball"))
    c_full = {tuple(v) for v in full.extra["exclusion_trace"]["stem_cap_ijk"].tolist()}
    c_in = {tuple(v) for v in inb.extra["exclusion_trace"]["stem_cap_ijk"].tolist()}
    assert c_in and c_in < c_full
    d = lambda S: np.array([np.linalg.norm(ORIGIN + np.asarray(v) * VS - A) for v in S])   # noqa: E731
    # the builder works in float32-origin metres: a voxel exactly on the ball (7 voxels above c) may fall either side
    assert np.all(d(c_in) <= BALL_R + 1e-6)
    assert np.all(d(c_full - c_in) > BALL_R - 1e-6)
    sc = inb.extra["exclusion"]["stem_cap"]
    assert sc["mode"] == "in_ball" and sc["n_removed_outside_ball"] == 0
    # the stem part above the ball survives in_ball
    g, owner = _labels()
    above = {v for v in _seed_set(inb) if owner[v] == 1 and np.linalg.norm(g[v] - A) > BALL_R}
    assert above


def test_deterministic_and_parameters_from_config():
    a, b = _tier(_builder(target_stem_cap="full")), _tier(_builder(target_stem_cap="full"))
    assert torch.equal(a.values, b.values)
    assert np.array_equal(a.extra["exclusion_trace"]["stem_cap_ijk"], b.extra["exclusion_trace"]["stem_cap_ijk"])
    narrow = _tier(_builder(target_stem_cap="full", stem_cap_radius=0.004, stem_cap_top=0.012,
                            stem_cap_bottom_fit_frac=1.0))
    sc = narrow.extra["exclusion"]["stem_cap"]
    assert (sc["radius_mm"], sc["top_mm"], sc["bottom_fit_frac"]) == (4.0, 12.0, 1.0)
    assert sc["z_bottom_m"] == pytest.approx(A[2] + RA) and sc["z_top_m"] == pytest.approx(A[2] + RA + 0.012)
    cap = narrow.extra["exclusion_trace"]["stem_cap_ijk"]
    assert np.all(cap[:, :2] == np.array([19, 20]))   # only the axis column (radius 4 mm < one voxel)
    assert 0 < sc["n_removed"] < a.extra["exclusion"]["stem_cap"]["n_removed"]


def test_no_fit_radius_skips_the_cap_and_says_why():
    t = _tier(_builder(target_stem_cap="full"), ball=_ball(fit_radius_mm=None))
    off = _tier(_builder(), ball=_ball(fit_radius_mm=None))
    sc = t.extra["exclusion"]["stem_cap"]
    assert sc["applied"] is False and sc["reason"] == "no_fit_radius" and sc["n_removed"] == 0
    assert len(t.extra["exclusion_trace"]["stem_cap_ijk"]) == 0
    assert torch.equal(t.values, off.values)


# ============================================================================ 7 record


def test_record_has_the_stem_cap_block_and_ijk_key():
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    t = _tier(_builder(target_stem_cap="full"))
    field = types.SimpleNamespace(stats={"target_free": {"mode": "ball", **CuroboFieldBuilder._exclusion_counts([t])}},
                                  exclusion_trace=CuroboFieldBuilder._exclusion_trace([t]))
    payload = {}
    info = ConstraintRecordWriter._write_exclusion_trace(field, payload)
    arr = payload["target_free_stem_cap_ijk"]
    assert arr.dtype == np.int16 and arr.shape[1] == 3
    sc = info["target_exclusion"]["stem_cap"]
    assert len(arr) == sc["n_removed"] > 0
    assert sc["mode"] == "full" and sc["radius_mm"] == 20.0 and sc["top_mm"] == 25.0 and sc["bottom_fit_frac"] == 0.5
    assert sc["fit_radius_mm"] == pytest.approx(RA * 1e3)
    # the ball_owned removal stays in its own key, disjoint from the cap's
    rem = {tuple(v) for v in payload["target_free_removed_ijk"].tolist()}
    assert not rem & {tuple(v) for v in arr.tolist()}
    json.dumps(info)


# ============================================================================ 8 config · CLI


def test_config_defaults_round_trip_and_validation():
    d = AG3SConfig().esdf
    assert (d.target_stem_cap, d.stem_cap_radius, d.stem_cap_top, d.stem_cap_bottom_fit_frac) == \
        ("off", 0.020, 0.025, 0.5)
    cfg = AG3SConfig.from_dict({"esdf": {"target_exclusion": "ball_owned", "target_stem_cap": "in_ball",
                                         "stem_cap_radius": 0.015, "stem_cap_top": 0.03,
                                         "stem_cap_bottom_fit_frac": 0.4}})
    assert AG3SConfig.from_dict(cfg.to_dict()).esdf == cfg.esdf
    for bad in ({"target_exclusion": "ball_owned", "target_stem_cap": "on"},
                {"target_stem_cap": "full"},                                   # `ball` (not owned)
                {"target_exclusion": "ball_owned", "target_stem_cap": "full", "stem_cap_radius": 0.0},
                {"target_exclusion": "ball_owned", "target_stem_cap": "full", "stem_cap_top": -0.001},
                {"target_exclusion": "ball_owned", "target_stem_cap": "full", "stem_cap_bottom_fit_frac": float("nan")},
                {"target_exclusion": "ball_owned", "target_stem_cap": "full", "target_extent": "oracle_visible"}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig.from_dict({"esdf": bad})


def _parse(*argv):
    from AG3S.trajopt.serve_safe import build_parser

    return build_parser().parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml", *argv])


def test_cli_defaults_and_mapping():
    from AG3S.trajopt.serve_safe import target_exclusion_section

    base = ("--target-exclusion", "ball-owned")
    assert target_exclusion_section(_parse(*base)) == {"target_exclusion": "ball_owned"}
    assert target_exclusion_section(_parse(*base, "--target-stem-cap", "off")) == {"target_exclusion": "ball_owned"}
    sec = target_exclusion_section(_parse(*base, "--target-stem-cap", "in_ball", "--stem-cap-radius-mm", "15",
                                          "--stem-cap-top-mm", "30", "--stem-cap-bottom-fit-frac", "0.4"))
    assert sec == {"target_exclusion": "ball_owned", "target_stem_cap": "in_ball", "stem_cap_radius": 0.015,
                   "stem_cap_top": 0.03, "stem_cap_bottom_fit_frac": 0.4}
    cfg = AG3SConfig.from_dict({"esdf": sec}).esdf
    assert {k: getattr(cfg, k) for k in sec} == sec   # what `build_ag3s` checks survived


def _main_errors(monkeypatch, *argv):
    import sys

    from AG3S.trajopt import serve_safe

    monkeypatch.setattr(sys, "argv", ["serve_safe", "--config", "c", "--checkpoint", "k",
                                      "--model-xml", "x.xml", *argv])
    with pytest.raises(SystemExit) as e:
        serve_safe.main()
    return e.value.code


def test_cli_rejects_cap_without_ball_owned_and_params_without_cap(monkeypatch, capsys):
    assert _main_errors(monkeypatch, "--target-stem-cap", "full") == 2
    assert "ball-owned" in capsys.readouterr().err
    owned = ("--esdf-backend", "curobo", "--target-field-policy", "exclude-authorized",
             "--target-exclusion", "ball-owned")
    assert _main_errors(monkeypatch, *owned, "--stem-cap-radius-mm", "15") == 2
    assert "--target-stem-cap" in capsys.readouterr().err
    assert _main_errors(monkeypatch, *owned, "--target-stem-cap", "off", "--stem-cap-top-mm", "10") == 2
    capsys.readouterr()
    for flag, bad in (("--stem-cap-radius-mm", "0"), ("--stem-cap-top-mm", "-1"), ("--stem-cap-bottom-fit-frac", "nan")):
        assert _main_errors(monkeypatch, *owned, "--target-stem-cap", "full", flag, bad) == 2
        assert flag in capsys.readouterr().err
