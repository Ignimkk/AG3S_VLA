"""T44 I0 + I1 — target-free 계층의 공 제외 trace 와 `ball_owned` (공 ∖ non-target 소유 seed).

## 무엇을 박나

| # | 성질 |
|---|---|
| 1 | `_ball_owned_split` — 공 안 이웃 물체 (distractor) seed 는 보존, target seed 는 제거 |
| 2 | ambiguous (`|d_t − d_o| ≤ tie`) 는 보존되고 trace 에 남는다 |
| 3 | sparse target (점 몇 개) — non-target 이 멀면 target seed 는 여전히 제거 (unowned=remove) |
| 4 | unowned `remove` / `keep` |
| 5 | support guard — `z_min` 아래는 어느 모드에서도 후보가 아니다 |
| 6 | `ball` 모드: `_ball_exclusion` = `_exclude_ball_seeds` (같은 seed · 같은 부호 집합) + trace |
| 7 | `ball` 모드 `_build_tiers` 값 = T13 알고리즘 (seed → `_exclude_ball_seeds` → propagate → 부호) 비트 동일 |
| 8 | `ball_owned` `_build_tiers`: distractor 표면 유지 · 내부 음수 유지, target 표면 제거 · 내부 양수 |
| 9 | stats 키 (`exclusion_method` …) · `field.exclusion_trace` · 기록 npz 키 |
| 10 | config 왕복 · validation · CLI (`--target-exclusion` …) 배선 · pipeline 배선 (`nontarget_points`) |
| 11 | `to_adapter` — `metrics["target_permission"]` (authorized links · 행별 vector) |

합성 씬은 5 mm 격자 위의 두 구 (target · distractor, 간격 5 mm) + 테이블 평면이다. 가짜 integrator 는
cuRobo 의 seed → propagate → 부호 경로를 brute force (KD-tree) 로 흉내 낸다 — `_build_tiers` 가 cuRobo
에 요구하는 표면 (`_site_index`, `_seed_esdf_impl`, `_propagate_and_distance_impl`, `get_voxel_grid`) 만.
실제 cuRobo 경로는 impl.md 의 GPU smoke 가 본다.
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
    TargetBall,
    _ball_owned_split,
)

VS = 0.005
N = 36
ORIGIN = np.array([0.40, 0.20, 0.78])          # 첫 voxel 중심
A = np.array([0.475, 0.290, 0.853])             # target 중심
B = np.array([0.475, 0.335, 0.853])             # distractor 중심 (45 mm 옆)
R = 0.020                                       # 두 구의 반지름 → 표면 간격 5 mm
TABLE_Z = 0.832


def _centres(shape=(N, N, N), origin=ORIGIN, vs=VS):
    ax = [origin[i] + np.arange(shape[i]) * vs for i in range(3)]
    return np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)


def _sdf(p):
    """target · distractor · 테이블 (z < TABLE_Z 반공간) 의 합성 SDF 와 물체 id (0 A, 1 B, 2 table)."""
    da = np.linalg.norm(p - A, axis=-1) - R
    db = np.linalg.norm(p - B, axis=-1) - R
    dt = p[..., 2] - TABLE_Z
    stack = np.stack([da, db, dt], axis=-1)
    return stack.min(axis=-1), stack.argmin(axis=-1)


def _surface_points(c, n, seed, *, upper_only=True):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(4 * n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    if upper_only:
        v = v[v[:, 2] > -0.3]
    return c + R * v[:n]


def _ball(radius=0.035):
    return TargetBall(centre=A, radius=radius, z_min=TABLE_Z + VS)


# ============================================================================ fake cuRobo


class _VoxelGrid:
    def __init__(self, t):
        self.feature_tensor = t


class _FakeItg:
    """seed = |sdf| < 0.75 vs 인 voxel · propagate = 남은 seed 중 최근접 · 부호 = 그 voxel 의 sdf."""

    def __init__(self, shape=(N, N, N)):
        self.shape = shape
        self._esdf_voxel_size = torch.zeros(1)
        centre = ORIGIN + 0.5 * (np.asarray(shape) - 1) * VS     # 격자 **중심** (함정 1)
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
        seed = np.abs(sdf) < 0.75 * VS
        ii, jj, kk = np.nonzero(seed)
        packed = np.full(self.shape, -1, np.int64)
        packed[ii, jj, kk] = (kk << 20) | (jj << 10) | ii
        self._site_index.copy_(torch.as_tensor(packed.astype(np.int32)))

    def _propagate_and_distance_impl(self, origin, vs):
        from scipy.spatial import cKDTree

        g = self._grid()
        site = self._site_index.numpy()
        seeds = np.argwhere(site >= 0)
        flat = g.reshape(-1, 3)
        d, nn = cKDTree(g[seeds[:, 0], seeds[:, 1], seeds[:, 2]]).query(flat, k=1)
        sdf, _ = _sdf(g)
        val = np.where(sdf.reshape(-1) < 0.0, -d, d).reshape(self.shape).astype(np.float32)
        self._dist.copy_(torch.as_tensor(val))
        s = seeds[nn]
        packed = ((s[:, 2] << 20) | (s[:, 1] << 10) | s[:, 0]).reshape(self.shape)
        self._site_index.copy_(torch.as_tensor(packed.astype(np.int32)))

    def get_voxel_grid(self):
        return _VoxelGrid(self._dist)


def _builder(**esdf):
    cfg = AG3SConfig.from_dict({"esdf": {"backend": "curobo", "voxel_size": 0.02,
                                         "fine_voxel_size": VS, **esdf}}).esdf
    b = CuroboFieldBuilder(cfg)
    b._fine_centre = ORIGIN + 0.5 * (N - 1) * VS
    return b


def _owned(target=None, nontarget=None, **kw):
    if target is None:
        target = _surface_points(A, 120, 0)
    if nontarget is None:
        # the distractor seen all round (its side facing the target included).
        nontarget = _surface_points(B, 300, 1, upper_only=False)
    return OwnedEvidence(target_points=target, nontarget_points=nontarget, **kw)


def _seed_scene():
    """`(itg, cand, centres, seed, owner)` — 공 안 cand · 그 중심 · seed 표시 · GT 소유 물체."""
    b = _builder()
    itg = _FakeItg()
    itg._esdf_voxel_size.copy_(torch.tensor([VS]))
    itg._last_esdf_origin.copy_(itg._origin)
    itg._seed_esdf_impl(None, None)
    cand = b._ball_candidates(_ball(), ORIGIN, VS, (N, N, N))
    centres = ORIGIN + cand * VS
    seed = itg._site_index.numpy()[cand[:, 0], cand[:, 1], cand[:, 2]] >= 0
    _, owner = _sdf(centres)
    return b, itg, cand, centres, seed, owner


# ============================================================================ 1-5 분류


def test_distractor_seeds_inside_the_ball_are_kept_and_target_seeds_removed():
    _, _, cand, centres, seed, owner = _seed_scene()
    assert int((seed & (owner == 1)).sum()) > 10, "공 안에 distractor seed 가 있어야 시험이 뜻을 갖는다"
    split = _ball_owned_split(centres, seed, _ball(), _owned(), VS)
    on_b = seed & (owner == 1)
    on_a = seed & (owner == 0)
    assert int((split["removed"] & on_b).sum()) == 0, "distractor seed 를 지웠다"
    kept = split["nontarget"] | split["ambiguous"]
    assert np.all(kept[on_b]), "distractor seed 는 non-target 이거나 (목에서는) ambiguous 로 보존"
    assert int((split["nontarget"] & on_b).sum()) > 0
    # target seed 는 거의 다 지운다 — 남는 것은 tie 띠 (두 구가 5 mm 떨어진 목) 뿐이다.
    assert int((split["removed"] & on_a).sum()) >= 0.85 * int(on_a.sum())
    # 지우는 것은 seed 뿐, 부호 후보는 지운 seed 를 포함한다.
    assert not np.any(split["removed"] & ~seed)
    assert np.all(split["sign"][split["removed"]])


def test_ambiguous_seeds_are_kept_and_counted():
    _, _, cand, centres, seed, owner = _seed_scene()
    split = _ball_owned_split(centres, seed, _ball(), _owned(tie_voxels=1.0), VS)
    amb = split["ambiguous"]
    assert int(amb.sum()) > 0
    assert not np.any(split["removed"] & amb)
    gap = np.abs(split["d_t"] - split["d_o"])
    assert np.all(gap[amb] <= VS + 1e-12)
    # tie 를 0 으로 두면 ambiguous 가 (거의) 사라진다 — 수가 tie 를 따른다.
    split0 = _ball_owned_split(centres, seed, _ball(), _owned(tie_voxels=0.0), VS)
    assert int(split0["ambiguous"].sum()) < int(amb.sum())


def test_sparse_target_points_still_remove_the_target_seeds():
    _, _, cand, centres, seed, owner = _seed_scene()
    sparse = _surface_points(A, 5, 3)
    split = _ball_owned_split(centres, seed, _ball(), _owned(target=sparse), VS)
    on_a_far = seed & (owner == 0) & (np.linalg.norm(centres - B, axis=1) > R + 3 * VS)
    assert int(on_a_far.sum()) > 20
    assert int((split["removed"] & on_a_far).sum()) == int(on_a_far.sum()), (
        "점 5 개의 target 도 (non-target 이 멀면) 공 안 seed 를 다 지워야 한다 — 공의 recall")
    assert int(split["unowned"].sum()) > 0, "점이 성기면 unowned 가 생기고 그것을 지운다"


def test_unowned_remove_versus_keep():
    _, _, cand, centres, seed, owner = _seed_scene()
    sparse = _surface_points(A, 5, 3)
    rm = _ball_owned_split(centres, seed, _ball(), _owned(target=sparse, unowned="remove"), VS)
    kp = _ball_owned_split(centres, seed, _ball(), _owned(target=sparse, unowned="keep"), VS)
    assert np.array_equal(rm["unowned"], kp["unowned"]) and kp["unowned"].any()
    assert np.all(rm["removed"][rm["unowned"]])
    assert not np.any(kp["removed"][kp["unowned"]])


def test_no_nontarget_evidence_is_the_ball_on_owned_seeds_and_never_wider():
    _, _, cand, centres, seed, owner = _seed_scene()
    split = _ball_owned_split(centres, seed, _ball(), _owned(nontarget=np.zeros((0, 3))), VS)
    assert not split["ambiguous"].any() and not split["nontarget"].any()
    assert np.array_equal(split["removed"], seed), "증거가 없으면 공 안 seed 전부 (= ball)"


def test_the_nontarget_crop_does_not_change_the_classification():
    _, _, cand, centres, seed, owner = _seed_scene()
    far = np.random.default_rng(5).uniform([-1, -1, 0], [2, 2, 2], (2000, 3))
    nt = np.vstack([_surface_points(B, 120, 1), far])
    a = _ball_owned_split(centres, seed, _ball(), _owned(nontarget=nt), VS)
    # 잘라내지 않은 기준: 같은 식을 전체 점으로.
    from scipy.spatial import cKDTree
    d_o = cKDTree(nt).query(centres, k=1)[0]
    assert np.all((a["d_o"] == d_o) | (a["d_o"] > a["d_t"] + VS))
    for k in ("target", "nontarget", "ambiguous", "unowned", "removed", "sign"):
        ref = _ball_owned_split(centres, seed, _ball(), _owned(nontarget=nt[:120]), VS)[k]
        assert np.array_equal(a[k], ref), k


def test_support_guard_holds_in_both_modes():
    for owned in (None, _owned()):
        b, itg, *_ = _seed_scene()
        before = itg._site_index.numpy().copy()
        b._ball_exclusion(itg, _ball(radius=0.05), ORIGIN, VS, (N, N, N), owned=owned)
        cleared = (before >= 0) & (itg._site_index.numpy() < 0)
        z = _centres()[..., 2]
        assert int((cleared & (z < TABLE_Z + VS)).sum()) == 0, "테이블에 구멍"
        assert int(cleared.sum()) > 0


# ============================================================================ 6-7 ball 비트 동일


def test_ball_exclusion_equals_the_t13_path_plus_a_trace():
    b, itg_new, cand, centres, seed, owner = _seed_scene()
    _, itg_old, *_ = _seed_scene()
    n_new, idx_new, excl = b._ball_exclusion(itg_new, _ball(), ORIGIN, VS, (N, N, N))
    n_old, idx_old = b._exclude_ball_seeds(itg_old, _ball(), ORIGIN, VS, (N, N, N))
    assert n_new == n_old and np.array_equal(idx_new, idx_old)
    assert torch.equal(itg_new._site_index, itg_old._site_index)
    c = excl["counts"]
    assert c["exclusion_method"] == "ball"
    assert c["n_removed"] == c["n_seeds_in_ball"] == n_new
    assert all(c[k] == 0 for k in ("n_kept_nontarget", "n_ambiguous", "n_unowned_removed",
                                   "n_unowned_kept"))
    tr = excl["trace"]
    assert len(tr["removed_ijk"]) == n_new
    assert np.all(itg_new._site_index.numpy()[tuple(tr["removed_ijk"].T)] < 0)
    assert len(tr["nontarget_kept_ijk"]) == len(tr["ambiguous_ijk"]) == 0


def _t13_reference(builder, ball):
    """T13 의 `_build_tiers` 를 그대로 — seed → `_exclude_ball_seeds` → propagate → 부호 교정."""
    itg = _FakeItg()
    itg._esdf_voxel_size.copy_(torch.tensor([VS]))
    origin = torch.as_tensor(builder._fine_centre, dtype=torch.float32)
    itg._last_esdf_origin.copy_(origin)
    itg._seed_esdf_impl(origin, itg._esdf_voxel_size)
    org = builder._origin_of(itg, itg.shape, VS)
    _, idx = builder._exclude_ball_seeds(itg, ball, org, VS, itg.shape)
    itg._propagate_and_distance_impl(origin, itg._esdf_voxel_size)
    values = itg.get_voxel_grid().feature_tensor.detach().to(torch.float32).clone()
    builder._force_sign_on_pure_voxels(values, idx, VS, threshold_voxels=1.5)
    return values


def _tiers(builder, ball, owned=None):
    itg = _FakeItg()
    return builder._build_tiers(itg, None, "cpu", torch, only="fine", tier_name="fine_no_target",
                                ball=ball, owned=owned)


def test_ball_mode_tier_values_are_bit_identical_to_t13():
    b = _builder()
    (tier,) = _tiers(b, _ball())
    ref = _t13_reference(b, _ball())
    assert torch.equal(tier.values, ref)
    assert tier.extra["n_ball_seeds_excluded"] == tier.extra["exclusion"]["n_removed"]
    assert tier.extra["exclusion"]["exclusion_method"] == "ball"


def test_ball_owned_keeps_the_distractor_and_removes_the_target():
    b = _builder(target_exclusion="ball_owned")
    (ball_tier,) = _tiers(b, _ball())
    (own_tier,) = _tiers(b, _ball(), owned=_owned())
    g = _centres()
    sdf, owner = _sdf(g)
    in_ball = np.linalg.norm(g - A, axis=-1) <= _ball().radius
    vb, vo = ball_tier.values.numpy(), own_tier.values.numpy()

    # distractor 표면 (공 안): owned 는 거리 ≈ 0 을 지킨다; ball 은 지웠다.
    b_surf = in_ball & (owner == 1) & (np.abs(sdf) < 0.5 * VS) & (g[..., 2] > TABLE_Z + VS)
    assert int(b_surf.sum()) > 3
    assert np.all(np.abs(vo[b_surf]) <= VS), "owned 가 distractor 표면을 잃었다"
    assert np.max(np.abs(vb[b_surf])) > VS, "ball 이 distractor 표면을 지운다는 전제 (대조군)"

    # distractor 내부 (공 안, 음수): owned 는 부호를 지킨다.
    b_in = in_ball & (owner == 1) & (sdf < -1.6 * VS)
    if b_in.any():
        assert np.all(vo[b_in] < 0.0), "distractor 내부가 양수로 바뀌었다 (관통 숨김)"

    # target 표면 (tie 띠 밖) 은 지워지고, target 내부는 양수다 — ball 과 같은 결과.
    a_surf = (owner == 0) & (np.abs(sdf) < 0.5 * VS) & (np.linalg.norm(g - B, axis=-1) > R + 3 * VS) \
        & (g[..., 2] > TABLE_Z + 2 * VS)
    assert np.all(vo[a_surf] > VS)
    a_in = (owner == 0) & (sdf < -2 * VS) & (g[..., 2] > TABLE_Z + 2 * VS) \
        & (np.linalg.norm(g - B, axis=-1) > R + 3 * VS)
    assert a_in.any() and np.all(vo[a_in] > 0.0) and np.all(vb[a_in] > 0.0)

    ex = own_tier.extra["exclusion"]
    assert ex["exclusion_method"] == "ball_owned"
    assert ex["n_kept_nontarget"] > 0 and ex["n_removed"] < ball_tier.extra["exclusion"]["n_removed"]
    assert ex["n_removed"] == len(own_tier.extra["exclusion_trace"]["removed_ijk"])
    assert ex["n_kept_nontarget"] == len(own_tier.extra["exclusion_trace"]["nontarget_kept_ijk"])
    assert ex["n_ambiguous"] == len(own_tier.extra["exclusion_trace"]["ambiguous_ijk"])


def test_stats_counts_and_trace_helpers():
    b = _builder(target_exclusion="ball_owned")
    tiers = _tiers(b, _ball(), owned=_owned())
    counts = CuroboFieldBuilder._exclusion_counts(tiers)
    for k in ("exclusion_method", "n_seeds_in_ball", "n_removed", "n_kept_nontarget",
              "n_ambiguous", "n_unowned_removed", "n_unowned_kept"):
        assert k in counts
    assert counts["n_seeds_in_ball"] == (counts["n_removed"] + counts["n_kept_nontarget"]
                                         + counts["n_ambiguous"] + counts["n_unowned_kept"])
    trace = CuroboFieldBuilder._exclusion_trace(tiers)
    assert trace["tier"] == "fine_no_target" and trace["voxel_size"] == VS
    np.testing.assert_allclose(trace["origin"], ORIGIN, atol=1e-6)
    assert CuroboFieldBuilder._exclusion_counts([])["exclusion_method"] is None
    assert CuroboFieldBuilder._exclusion_trace([]) is None


# ============================================================================ 9 기록


def test_the_record_writes_the_trace_keys(tmp_path):
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    b = _builder(target_exclusion="ball_owned")
    tiers = _tiers(b, _ball(), owned=_owned())
    field = types.SimpleNamespace(
        stats={"target_free": {"mode": "ball", **CuroboFieldBuilder._exclusion_counts(tiers),
                               "geometry_version": {"request_seq": 3}}},
        exclusion_trace=CuroboFieldBuilder._exclusion_trace(tiers))
    payload = {}
    info = ConstraintRecordWriter._write_exclusion_trace(field, payload)
    for k in ConstraintRecordWriter.EXCLUSION_TRACE_KEYS:
        assert payload[f"target_free_{k}"].dtype == np.int16
    assert len(payload["target_free_removed_ijk"]) == info["target_exclusion"]["n_removed"]
    assert info["target_exclusion"]["exclusion_method"] == "ball_owned"
    assert info["target_exclusion"]["geometry_version"] == {"request_seq": 3}
    json.dumps(info)   # real JSON
    # 공이 없는 프레임: 배열 없음.
    payload2 = {}
    ConstraintRecordWriter._write_exclusion_trace(
        types.SimpleNamespace(stats={"target_free": {"mode": "none"}}, exclusion_trace=None),
        payload2)
    assert payload2 == {}


# ============================================================================ 10 config · CLI · pipeline


def test_config_round_trip_and_validation():
    cfg = AG3SConfig.from_dict({"esdf": {"target_exclusion": "ball_owned", "owned_tie_voxels": 0.5,
                                         "owned_max_dist_voxels": 3.0, "owned_unowned": "keep"}})
    again = AG3SConfig.from_dict(cfg.to_dict())
    assert again.esdf == cfg.esdf
    d = AG3SConfig().esdf
    assert (d.target_exclusion, d.owned_tie_voxels, d.owned_max_dist_voxels, d.owned_unowned) == \
        ("ball", 1.0, 2.0, "remove")
    for bad in ({"target_exclusion": "owned"}, {"owned_unowned": "drop"},
                {"owned_tie_voxels": -1.0}, {"owned_max_dist_voxels": 0.0}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig.from_dict({"esdf": bad})


def _parse(*argv):
    from AG3S.trajopt.serve_safe import build_parser

    return build_parser().parse_args(["--config", "c", "--checkpoint", "k", "--model-xml", "x.xml",
                                      *argv])


def test_cli_defaults_leave_the_config_untouched():
    from AG3S.trajopt.serve_safe import target_exclusion_section

    assert target_exclusion_section(_parse()) == {}
    assert target_exclusion_section(_parse("--target-exclusion", "ball")) == {}


def test_cli_ball_owned_maps_to_esdf_keys():
    from AG3S.trajopt.serve_safe import target_exclusion_section

    args = _parse("--target-exclusion", "ball-owned", "--owned-tie-voxels", "0.5",
                  "--owned-max-dist-voxels", "3", "--owned-unowned", "keep")
    sec = target_exclusion_section(args)
    assert sec == {"target_exclusion": "ball_owned", "owned_tie_voxels": 0.5,
                   "owned_max_dist_voxels": 3.0, "owned_unowned": "keep"}
    cfg = AG3SConfig.from_dict({"esdf": sec})
    assert cfg.esdf.target_exclusion == "ball_owned"


def _main_errors(monkeypatch, *argv):
    """`main()` 의 인자 검사만 — 서버를 띄우기 전에 `ap.error` 로 죽는가."""
    import sys

    from AG3S.trajopt import serve_safe

    monkeypatch.setattr(sys, "argv", ["serve_safe", "--config", "c", "--checkpoint", "k",
                                      "--model-xml", "x.xml", *argv])
    with pytest.raises(SystemExit) as e:
        serve_safe.main()
    return e.value.code


def test_cli_rejects_owned_flags_without_ball_owned_and_ball_owned_with_relax(monkeypatch, capsys):
    assert _main_errors(monkeypatch, "--owned-tie-voxels", "0.5") == 2
    assert "ball-owned" in capsys.readouterr().err
    assert _main_errors(monkeypatch, "--esdf-backend", "curobo",
                        "--target-exclusion", "ball-owned") == 2
    assert "--target-field-policy" in capsys.readouterr().err


def _cloud(points):
    return types.SimpleNamespace(points=np.asarray(points, np.float64))


def test_pipeline_nontarget_points_from_indices_and_from_proximity():
    from AG3S.ag3s.runtime.pipeline import AG3S

    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, (50, 3))
    support = np.zeros(50, bool)
    support[:5] = True
    idx = np.arange(10, 20)
    stub = types.SimpleNamespace(config=AG3SConfig.from_dict(
        {"esdf": {"backend": "curobo", "fine_voxel_size": VS, "voxel_size": 0.02}}))
    geom = types.SimpleNamespace(points=pts[idx], point_indices=idx)
    out = AG3S._nontarget_points(stub, _cloud(pts), support, geom)
    assert out["source"] == "indices" and len(out["points"]) == 35
    assert out["n_target_dropped"] == 10 and out["n_support_dropped"] == 5
    # 옛 프레임의 기하: 인덱스가 이 cloud 를 가리키지 않는다 → 거리로 뺀다.
    stale = types.SimpleNamespace(points=pts[idx] + 1e-4, point_indices=idx + 1)
    out2 = AG3S._nontarget_points(stub, _cloud(pts), support, stale)
    assert out2["source"] == "proximity" and out2["n_target_dropped"] == 10
    assert len(out2["points"]) == 35


def test_pipeline_passes_nontarget_points_only_with_ball_owned(monkeypatch):
    fw = pytest.importorskip("tests.ag3s.test_fine_window")
    nt = {"points": np.zeros((4, 3)), "source": "indices", "n_cloud": 9,
          "n_target_dropped": 3, "n_support_dropped": 2}
    ag3s = fw._ag3s(monkeypatch)
    fw._build(ag3s, nontarget=nt)
    assert "nontarget_points" not in ag3s._esdf_builder.calls[-1], "ball 모드는 예전 호출 그대로"

    ag3s = fw._ag3s(monkeypatch, target_exclusion="ball_owned")
    fw._build(ag3s, nontarget=nt)
    kw = ag3s._esdf_builder.calls[-1]
    assert kw["target_free_ball"] is not None
    assert kw["nontarget_points"] is nt["points"]


def test_geometry_version_is_recorded():
    from AG3S.ag3s.runtime.pipeline import AG3S

    field = types.SimpleNamespace(stats={"target_free": {"mode": "ball"}})
    ident = types.SimpleNamespace(id=7, state="visible", last_seen_frame=12)
    stub = types.SimpleNamespace(_gated_manipulated=ident)
    tgt = types.SimpleNamespace(point_indices=np.arange(5))
    AG3S._record_geometry_version(stub, field, tgt, 12, None)
    gv = field.stats["target_free"]["geometry_version"]
    assert gv["request_seq"] == 12 and gv["manipulated_id"] == 7 and gv["n_points"] == 5
    assert len(gv["point_indices_sha1"]) == 12
    AG3S._record_geometry_version(stub, field, tgt, 12, {"points": np.zeros((3, 3)),
                                                         "source": "indices"})
    assert field.stats["target_free"]["nontarget"] == {"source": "indices", "n_points": 3}


# ============================================================================ 11 to_adapter


def test_target_permission_record():
    from AG3S.ag3s.constraints.to_adapter import _target_permission_record

    names = ["torso", "ee_left", "ee_finger_l1", "link_left_arm_5"]
    builder = types.SimpleNamespace(
        sphere_link_names=names,
        clearance_policy=types.SimpleNamespace(
            authorized_links=lambda ctx: frozenset({"ee_left", "ee_finger_l1", "ee_finger_l2"})))
    mask = np.array([False, True, True, False])
    rec = _target_permission_record(builder, None, "exclude_authorized", mask)
    assert rec["authorized_links"] == ["ee_finger_l1", "ee_left"]
    assert rec["target_field_exclude"] == [False, True, True, False]
    assert rec["n_rows_excluded"] == 2 and rec["excluded_links"] == ["ee_finger_l1", "ee_left"]
    none = _target_permission_record(builder, None, "relax", None)
    assert none["target_field_exclude"] is None and none["n_rows_excluded"] == 0
    json.dumps(rec)


# ============================================================================ GPU (real cuRobo; skipped on CPU)


def test_curobo_ball_owned_changes_only_the_target_free_layer():
    """Real cuRobo seed → exclusion → propagate → sign: `ball_owned` leaves the main layers bit-identical
    to `ball`, records the split, and keeps the seeds the neighbour's points own."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    pytest.importorskip("curobo")
    G = pytest.importorskip("tests.o4.test_gpu_esdf_field")
    from AG3S.ag3s.fields.curobo_builder import FineWindowRequest

    cams = G._scene_cameras()
    apple = np.random.default_rng(0).normal([0.51, -0.03, 0.85], 0.012, (300, 3))
    other = np.random.default_rng(3).normal([0.51, 0.03, 0.85], 0.012, (300, 3))
    kw = dict(target_points=apple, target_free_points=apple,
              target_free_ball=TargetBall(centre=[0.51, -0.03, 0.85], radius=0.045, z_min=0.825),
              fine_window=FineWindowRequest(manipulated_points=apple, placement="target_centroid"))
    fields = {}
    for name, over, extra in (("ball", {}, {}),
                              ("owned", {"target_exclusion": "ball_owned"},
                               {"nontarget_points": other})):
        b = CuroboFieldBuilder(G._config(**over))
        for _ in range(2):
            fields[name] = b.update(cams, **kw, **extra)
    fb, fo = fields["ball"], fields["owned"]
    for lb, lo in zip(fb.layers, fo.layers):
        assert torch.equal(lb.values_device, lo.values_device)
    tb, to = fb.stats["target_free"], fo.stats["target_free"]
    assert tb["exclusion_method"] == "ball" and tb["n_removed"] == tb["n_seeds_in_ball"] > 0
    assert to["exclusion_method"] == "ball_owned"
    assert to["n_seeds_in_ball"] == tb["n_seeds_in_ball"]
    assert to["n_removed"] + to["n_kept_nontarget"] + to["n_ambiguous"] + to["n_unowned_kept"] \
        == to["n_seeds_in_ball"]
    assert len(fo.exclusion_trace["removed_ijk"]) == to["n_removed"]
    assert len(fb.exclusion_trace["removed_ijk"]) == tb["n_removed"]
