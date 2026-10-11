"""T45 A10 — 손바닥 몸통 구 (`--palm-cover on`, 기본 off).

사용자 판정 (2026-10-11, T45 audit 뒤): "Palm 은 5 mm cover 를 기본 off 옵션으로 구현하고, 기존 coverage 생성 코드를
재사용. 구 수는 전체 466 개인지 확인". 손바닥 (`ee_left/right`, MJCF `EE_BODY_*`) 의 gap-filling capsule 사슬을
손가락 몸통 구 (T43 FC `finger_body_spheres`) 와 같은 계산의 구로 바꾼다.

이 파일이 고정하는 것:

1. **off 는 그대로** — `palm_cover=None/False/{}` 와 인자 없는 호출이 같은 모델이고, T44d C1a 서버 기록의 구 이름 ·
   반지름과 비트로 같다. CLI 기본 `off` 는 아무것도 넘기지 않는다 (`build_ag3s` · metadata 에 키가 없다).
2. **구 수** — `--links gripper --finger-cover both` 서버 모델 270 → 466 (손바닥 23 → 121 씩). `arms` 422 → 618.
3. **audit 후보 (b) 와 같은 구** — `outputs/verify/T45/palm_body_o5_*.npz` 와 중심 비트 동일 · 반지름 1 ulp 안.
4. **덮임** — 손바닥 겉면 표본 · mesh 정점 전부 구 안, 넘침 ≤ 5 mm (+ 표본 여유), T44d 손바닥–과일 접촉점 35 개 전부
   구 안 (off 모델에서는 전부 20 mm 넘게 밖).
5. **다른 구는 그대로** — 손가락 · 팔 구의 순서 · 값이 같고, 손바닥 구는 원래 손바닥 자리에 들어간다.
6. **권한** — 새 구도 link `ee_<side>` 이므로 `exclude_authorized` 의 target-free 마스크 · 마진 완화가 손바닥 구 전부에
   걸리고, 반대 손 · 비권한 link 에는 걸리지 않는다.
7. 배선: CLI → `build_ag3s` → `build_constraint_robot_model(palm_cover=...)` · `ta_offline.build_robot`.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

MODEL = pathlib.Path(
    "/mnt/dev/work/pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml")
#: T44d CL 서버 flag (`--capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32`).
SERVER = dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
C1A = pathlib.Path("/mnt/dev/work/outputs/verify/T44/CL/runs/C1a/server_8280_20261010T194812")
AUDIT_NPZ = pathlib.Path("/mnt/dev/work/outputs/verify/T45/palm_body_o5_0b966459a6ff.npz")
PALMS = ("ee_left", "ee_right")
FC = {"hands": "both"}

#: T45 audit (`outputs/verify/T45/events_cands.json`) — T44d C0 · C1a · C1b 의 손바닥–이웃 과일 접촉 8 event 의 접촉점
#: 35 개. 전부 MJCF `EE_BODY_R` geom 731 (`EE_BODY_collision_2`), **MJCF body frame, mm**. URDF link = `ee_right`.
PALM_CONTACTS_BODY_MM = np.array([
    [-33.6511, -27.885, -73.2259], [-33.8572, -27.885, -73.2259], [-34.0763, -27.885, -73.2259],
    [-34.1694, -27.8548, -73.2169], [-34.1162, -27.8623, -73.2192], [-34.1634, -27.885, -73.2259],
    [-41.872, -28.193, -52.2517], [-41.6365, -28.1603, -52.2521], [-41.7455, -28.1796, -52.2438],
    [-41.6056, -28.1581, -52.0199], [-41.5414, -28.1059, -51.6848], [-41.3788, -28.061, -51.5057],
    [-32.9504, -28.1216, -72.4084], [-32.8555, -28.0801, -72.3965], [-32.7112, -27.8455, -73.2143],
    [-32.6607, -27.8521, -73.2162], [-32.5688, -27.8557, -73.2172], [-32.5135, -27.8294, -73.2094],
    [-16.1563, -27.885, -73.2259], [-15.9674, -27.8111, -73.1933], [-15.9147, -27.8354, -73.2031],
    [-15.754, -27.8509, -73.21], [-15.673, -27.8516, -73.2102], [-40.9511, -28.193, -52.2517],
    [-40.7668, -28.174, -52.2518], [-41.6715, -28.1718, -54.4511], [-41.6489, -28.1735, -54.0667],
    [-41.6458, -28.193, -53.457], [-31.8311, -28.1216, -72.4084], [-31.8284, -28.1064, -72.4041],
    [-31.8528, -28.0853, -72.3981], [-31.791, -28.0719, -72.3942], [-32.7785, -28.1216, -72.4084],
    [-32.5928, -27.879, -73.2241], [-32.6275, -28.1141, -72.4063]])


@pytest.fixture(scope="module")
def mj():
    mujoco = pytest.importorskip("mujoco")
    if not MODEL.exists():  # pragma: no cover - asset-dependent
        pytest.skip(f"model not present: {MODEL}")
    return mujoco, mujoco.MjModel.from_xml_path(str(MODEL))


@pytest.fixture(scope="module")
def scene(mj):
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene

    mujoco, model = mj
    return TransportScene.attach(model, mujoco.MjData(model))


def _build(scene, links="gripper", sphere_options=None, **kw):
    from benchmark.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
    from benchmark.trajopt.serve_safe import constraint_link_filter

    return build_constraint_robot_model(scene, link_filter=constraint_link_filter(links),
                                        sphere_options={**SERVER, **dict(sphere_options or {})},
                                        **kw)


@pytest.fixture(scope="module")
def server(scene):
    """T44d CL 서버 모델 (`--links gripper --finger-cover both`)."""
    return _build(scene, finger_cover=FC)


@pytest.fixture(scope="module")
def pc(scene):
    return _build(scene, finger_cover=FC, palm_cover=True)


@pytest.fixture(scope="module")
def palm_surface(mj, server):
    from benchmark.ag3s.experiments.sources.gripper_cover import PALM_COVER_SPACING, link_surface

    _, model = mj
    return link_surface(model, server, "ee_right", spacing=PALM_COVER_SPACING)


def _fk(m, seed=3):
    q = np.random.default_rng(seed).uniform(-1.0, 1.0, m.nq)
    return m.sphere_centers_numeric(q)


# =============================================================================================
# 1. off 는 그대로
# =============================================================================================


def test_off_is_the_model_the_t44d_c1a_server_used(server):
    rec = sorted((C1A / "server_constraints" / "run_0000").glob("chunk_*.npz"))
    if not rec:
        pytest.skip("T44d C1a 서버 기록이 없다")
    z = np.load(rec[0], allow_pickle=True)
    names = json.loads(str(z["summary_json"]))["sphere_link_names"]
    assert list(server.sphere_link_names) == names
    assert np.array_equal(server.radii.astype(np.float32), z["sphere_radii"])
    assert not hasattr(server, "palm_cover_report")


@pytest.mark.parametrize("off", [None, False, {}])
@pytest.mark.parametrize("finger", [None, FC])
def test_off_keyword_paths_are_the_same_model(scene, server, off, finger):
    ref = server if finger else _build(scene)
    kw = {"finger_cover": finger} if finger else {}
    m = _build(scene, palm_cover=off, **kw)
    assert m.sphere_link_names == ref.sphere_link_names
    assert m.sphere_roles == ref.sphere_roles
    assert np.array_equal(m.radii, ref.radii)
    a, ra = _fk(ref)
    b, rb = _fk(m)
    assert np.array_equal(a, b) and np.array_equal(ra, rb)
    assert not hasattr(m, "palm_cover_report")


def test_cli_default_off_passes_nothing_and_on_builds_options(caplog):
    from benchmark.trajopt import serve_safe

    base = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
    p = serve_safe.build_parser()
    args = p.parse_args(base)
    assert args.palm_cover == "off" and args.palm_cover_overflow is None
    assert serve_safe.palm_cover_options(args) is None
    assert serve_safe.palm_cover_options(p.parse_args(base + ["--palm-cover", "off"])) is None
    assert serve_safe.palm_cover_options(p.parse_args(base + ["--palm-cover", "on"])) == \
        {"hands": "both"}
    assert serve_safe.palm_cover_options(p.parse_args(
        base + ["--palm-cover", "on", "--palm-cover-overflow", "2.5"])) == \
        {"hands": "both", "overflow": 0.0025}
    with pytest.raises(SystemExit):
        serve_safe.palm_cover_options(p.parse_args(base + ["--palm-cover-overflow", "5"]))
    for bad in ("0", "-1", "nan", "inf"):
        with pytest.raises(SystemExit):
            serve_safe.palm_cover_options(p.parse_args(
                base + ["--palm-cover", "on", "--palm-cover-overflow", bad]))
    with pytest.raises(SystemExit):
        p.parse_args(base + ["--palm-cover", "both"])
    # 옛 Namespace (속성 없음) 도 off
    import argparse

    assert serve_safe.palm_cover_options(argparse.Namespace()) is None
    with caplog.at_level("WARNING"):
        serve_safe.announce_palm_cover(None)
    assert not caplog.records


def test_recorded_c1a_command_line_is_off_and_appending_the_flag_turns_it_on():
    """`ta_offline.server_args` 처럼 기록 명령줄을 서버 parser 로 읽으면 off, `--palm-cover on` 을 붙이면 on."""
    proc = C1A / "server_proc.txt"
    if not proc.exists():
        pytest.skip("T44d C1a server_proc.txt 가 없다")
    from benchmark.trajopt import serve_safe

    line = [ln for ln in proc.read_text().splitlines() if "serve_safe" in ln][0]
    argv = line.split("benchmark.trajopt.serve_safe", 1)[1].split()
    p = serve_safe.build_parser()
    assert serve_safe.palm_cover_options(p.parse_args(argv)) is None
    assert serve_safe.palm_cover_options(p.parse_args(argv + ["--palm-cover", "on"])) == \
        {"hands": "both"}


@pytest.mark.parametrize("value", [None, {"hands": "both"}, {"hands": "both", "overflow": 0.004}])
def test_build_ag3s_passes_palm_cover_only_when_given(monkeypatch, value):
    """`build_ag3s` → `build_constraint_robot_model`: off 면 `palm_cover` 키가 아예 없다."""
    if not MODEL.exists():  # pragma: no cover
        pytest.skip("model not present")
    from benchmark.ag3s.experiments.reports import grounding_report
    from benchmark.trajopt import serve_safe

    seen = {}

    class Stop(Exception):
        pass

    def fake(scene, **kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(grounding_report, "build_constraint_robot_model", fake)
    with pytest.raises(Stop):
        serve_safe.build_ag3s(str(MODEL), voxel=0.02, range_max=2.0, links="gripper",
                              finger_cover=FC, palm_cover=value)
    if value is None:
        assert "palm_cover" not in seen
    else:
        assert seen["palm_cover"] == value
    assert seen["finger_cover"] == FC


def test_ta_offline_build_robot_passes_palm_cover_only_when_given(monkeypatch):
    if not MODEL.exists():  # pragma: no cover
        pytest.skip("model not present")
    from benchmark.ag3s.experiments.reports import grounding_report
    from benchmark.trajopt import serve_safe
    from benchmark.trajopt.experiments import ta_offline as TA

    seen = []

    def fake(scene, **kw):
        seen.append(kw)
        return object()

    monkeypatch.setattr(grounding_report, "build_constraint_robot_model", fake)
    base = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", str(MODEL),
            "--finger-cover", "both"]
    p = serve_safe.build_parser()
    TA.build_robot(p.parse_args(base))
    TA.build_robot(p.parse_args(base + ["--palm-cover", "on"]))
    assert "palm_cover" not in seen[0]
    assert seen[1]["palm_cover"] == {"hands": "both"}
    assert TA.variant_server_args("PC") == ("--palm-cover", "on")
    assert TA.variant_server_args("base") == ()


def test_no_safe_and_no_perception_reject_the_flag():
    from benchmark.trajopt import serve_safe

    base = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
    for extra in (["--no-safe"], ["--no-perception"]):
        p = serve_safe.build_parser()
        with pytest.raises(SystemExit):
            serve_safe.reject_bad_flag_combinations(
                p, p.parse_args(base + extra + ["--palm-cover", "on"]))
        # off 는 같이 써도 된다 (기본값)
        p = serve_safe.build_parser()
        serve_safe.reject_bad_flag_combinations(p, p.parse_args(base + extra + ["--palm-cover", "off"]))
    p = serve_safe.build_parser()
    serve_safe.reject_bad_flag_combinations(p, p.parse_args(base + ["--palm-cover", "on"]))


# =============================================================================================
# 2. 구 수 (사용자 판정: 전체 466)
# =============================================================================================


def test_gripper_server_model_goes_from_270_to_466(server, pc):
    from benchmark.ag3s.robot_models.urdf_sphere_chain import PALM_BODY_ROLE

    assert server.n_spheres == 270
    assert pc.n_spheres == 466
    names = np.asarray(pc.sphere_link_names)
    for palm in PALMS:
        assert int((np.asarray(server.sphere_link_names) == palm).sum()) == 23
        assert int((names == palm).sum()) == 121
    roles = np.asarray(pc.sphere_roles)
    assert set(roles[np.isin(names, PALMS)]) == {PALM_BODY_ROLE}
    rep = pc.palm_cover_report
    assert rep["options"] == {"hands": "both", "overflow": 0.005}
    assert rep["n_spheres_before"] == 270 and rep["n_spheres_after"] == 466
    for palm in PALMS:
        r = rep["links"][palm]
        assert r["n_spheres"] == 121 and r["n_uncovered"] == 0 and r["max_gap_mm"] == 0.0
        assert r["mujoco_body"] == ("EE_BODY_L" if palm == "ee_left" else "EE_BODY_R")
        assert r["radius_mm"][0] == pytest.approx(5.0, abs=0.01)
        assert r["radius_mm"][1] == pytest.approx(28.97, abs=0.01)
    assert pc.finger_cover_report == server.finger_cover_report


def test_arms_model_gets_the_same_palm_spheres(scene, pc):
    arms = _build(scene, "arms", finger_cover=FC)
    arms_pc = _build(scene, "arms", finger_cover=FC, palm_cover=True)
    assert arms.n_spheres == 422 and arms_pc.n_spheres == 618
    for palm in PALMS:
        ca, ra = arms_pc.local_spheres(palm)
        cg, rg = pc.local_spheres(palm)
        assert np.array_equal(ca, cg) and np.array_equal(ra, rg)


def test_matches_the_audit_candidate_b(pc):
    """T45 audit 후보 (b) `palm_body_o5` (`outputs/verify/T45/p/candidates.py`) 와 같은 구.

    반지름 차이는 overflow 표현 (`5.0 * 1e-3` 대 `0.005`) 의 1 ulp 뿐이다."""
    if not AUDIT_NPZ.exists():
        pytest.skip("T45 audit npz 가 없다")
    z = np.load(AUDIT_NPZ, allow_pickle=True)
    for palm in PALMS:
        c, r = pc.local_spheres(palm)
        assert np.array_equal(c, z["c"])
        assert np.allclose(r, z["r"], rtol=0.0, atol=1e-15)


# =============================================================================================
# 3. 덮임 · 넘침 · T44d 접촉점
# =============================================================================================


def test_every_palm_surface_sample_and_vertex_is_inside(pc, palm_surface):
    from benchmark.ag3s.experiments.sources.gripper_cover import _gap, finger_body_targets

    surf = palm_surface
    targets = surf.points[finger_body_targets(surf)]
    assert targets.shape[0] == 35287
    for palm in PALMS:          # 왼손 · 오른손은 link frame 에서 같은 mesh · 같은 구
        c, r = pc.local_spheres(palm)
        assert _gap(targets, c, r).max() <= 1e-9
        # mesh 정점 3866 개 중 5 개는 1 mm 표본 솎기에서 빠져 최대 0.039 mm 밖 (표본 간격 1 mm 의 근사)
        gv = _gap(surf.vertices, c, r)
        assert gv.max() <= 1e-4 and int((gv > 1e-9).sum()) <= 10
    cl, rl = pc.local_spheres("ee_left")
    cr, rr = pc.local_spheres("ee_right")
    assert np.array_equal(cl, cr) and np.array_equal(rl, rr)


def test_finer_samples_are_inside_up_to_the_sample_spacing(mj, pc):
    """0.5 mm 표본 (구를 고를 때의 2 배 촘촘) 에서도 0.1 % 미만만 밖이고 최대 0.5 mm 안 (audit: 최대 0.17–0.33 mm)."""
    from benchmark.ag3s.experiments.sources.gripper_cover import (
        _gap, finger_body_targets, link_surface)

    _, model = mj
    s = link_surface(model, pc, "ee_right", spacing=0.0005)
    t = s.points[finger_body_targets(s)]
    c, r = pc.local_spheres("ee_right")
    g = _gap(t, c, r)
    assert t.shape[0] > 100000
    assert (g > 1e-9).mean() < 1e-3
    assert g.max() <= 0.0005


def test_palm_spheres_leave_the_mesh_by_at_most_the_overflow(pc, palm_surface):
    from benchmark.ag3s.experiments.sources.gripper_cover import _sampled_overflow, hull_depth

    c, r = pc.local_spheres("ee_right")
    assert np.all(hull_depth(c, palm_surface.hulls) > 0.0), "구 중심은 mesh 안"
    over = _sampled_overflow(c, r, palm_surface.hulls, palm_surface.points)
    assert over <= 0.005 + 0.0003
    assert pc.palm_cover_report["links"]["ee_right"]["overflow_sampled_mm"] == pytest.approx(
        over * 1e3)
    lo, hi = palm_surface.vertices.min(0), palm_surface.vertices.max(0)
    assert np.all((c - r[:, None]).min(0) >= lo - 0.005 - 1e-9)
    assert np.all((c + r[:, None]).max(0) <= hi + 0.005 + 1e-9)


def test_t44d_palm_contact_points_are_inside_the_palm_spheres(mj, server, pc):
    """T45 audit A3: 손바닥–과일 접촉 8 event 의 35 점 — off 서버 구에서는 전부 20 mm 넘게 밖, palm cover 에서는 전부 안."""
    from benchmark.ag3s.experiments.sources.gripper_cover import _gap
    from benchmark.ag3s.experiments.sources.mujoco_source import static_urdf_ancestor

    _, model = mj
    found, T = static_urdf_ancestor(model, "EE_BODY_R", pc.model.links)
    assert found == "ee_right"
    T = np.asarray(T, np.float64)
    P = PALM_CONTACTS_BODY_MM * 1e-3 @ T[:3, :3].T + T[:3, 3]
    assert P.shape == (35, 3)
    c, r = pc.local_spheres("ee_right")
    g_on = _gap(P, c, r)
    assert np.all(g_on < 0.0), g_on.max()
    # audit (world frame, URDF FK) 값: −3.74 … −0.52 mm. 여기는 MJCF body → link 고정 변환이라 0.1 mm 안에서 같다.
    assert g_on.min() * 1e3 == pytest.approx(-3.735, abs=0.15)
    assert g_on.max() * 1e3 == pytest.approx(-0.525, abs=0.15)
    c0, r0 = server.local_spheres("ee_right")
    assert np.all(_gap(P, c0, r0) > 0.020), "off 서버 손바닥 구는 이 점들을 못 본다 (audit 24–43 mm)"


# =============================================================================================
# 4. 다른 구는 그대로
# =============================================================================================


def test_finger_spheres_unchanged_and_palm_spheres_take_the_palm_slots(server, pc):
    n0 = np.asarray(server.sphere_link_names)
    n1 = np.asarray(pc.sphere_link_names)
    k0, k1 = ~np.isin(n0, PALMS), ~np.isin(n1, PALMS)
    assert np.array_equal(n0[k0], n1[k1])
    assert np.asarray(server.sphere_roles)[k0].tolist() == np.asarray(pc.sphere_roles)[k1].tolist()
    a, ra = _fk(server)
    b, rb = _fk(pc)
    assert np.array_equal(np.asarray(a)[k0], np.asarray(b)[k1])
    assert np.array_equal(ra[k0], rb[k1])
    # gripper 모델에서 손바닥은 맨 앞 (gap-filling capsule 순서) — 그 자리에 121 구씩
    first = [i for i, n in enumerate(n0) if n in PALMS]
    assert first == list(range(46))
    assert set(n1[:242]) == set(PALMS) and not np.isin(n1[242:], PALMS).any()
    assert list(n0[:46:23]) == list(n1[:242:121])  # 손바닥 순서 (ee_left · ee_right) 그대로


def test_hands_option_and_bad_options(scene, pc):
    from benchmark.ag3s.robot_models.urdf_sphere_chain import PALM_BODY_ROLE

    right = _build(scene, finger_cover=FC, palm_cover={"hands": "right"})
    names = np.asarray(right.sphere_link_names)
    assert int((names == "ee_right").sum()) == 121 and int((names == "ee_left").sum()) == 23
    assert set(np.asarray(right.sphere_roles)[names == "ee_left"]) == {"collision"}
    assert set(np.asarray(right.sphere_roles)[names == "ee_right"]) == {PALM_BODY_ROLE}
    assert right.n_spheres == 270 + 98
    for bad in ({"hands": "middle"}, {"overflow": 0.0}, {"overflow": float("nan")},
                {"overflow": -1.0}, {"gap": 0.001}):
        with pytest.raises(ValueError):
            _build(scene, palm_cover=bad)


def test_radius_scale_does_not_touch_palm_body_spheres(scene, pc):
    big = _build(scene, finger_cover=FC, palm_cover=True,
                 sphere_options={"capsule_radius_scale_by_link": {"ee_left": 1.0, "ee_right": 1.0}})
    for palm in PALMS:
        c0, r0 = pc.local_spheres(palm)
        c1, r1 = big.local_spheres(palm)
        assert np.array_equal(c0, c1) and np.array_equal(r0, r1)


def test_max_opening_is_unchanged(server, pc):
    """T26 최대 개도는 손가락 구에서만 읽는다 — 손바닥 구는 바꾸지 않는다."""
    a = {g["parent_link"]: g for g in server.gripper_openings()}
    b = {g["parent_link"]: g for g in pc.gripper_openings()}
    assert a.keys() == b.keys() and a
    for k in a:
        assert a[k] == b[k]


# =============================================================================================
# 5. 권한 — 같은 link 이름이라 권한 행 · target-free 조회가 손바닥 구 전부에 걸린다
# =============================================================================================


@pytest.fixture(scope="module")
def permission(pc):
    import types

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.constraints.clearance import ClearancePolicy
    from benchmark.ag3s.constraints.constraint_builder import ConstraintBuilder
    from benchmark.ag3s.constraints.to_adapter import build_constraint_set
    from benchmark.ag3s.stages.geometry import fit_sphere
    from benchmark.ag3s.types import (
        ContactPolicyContext, Manipulator, Phase, PointCloud, TargetGeometry)

    cfg = AG3SConfig.from_dict({"constraint": {"horizon": 3,
                                               "target_field_policy": "exclude_authorized"}})
    builder = ConstraintBuilder(
        pc, cfg.constraint, cfg.geometry,
        clearance_policy=ClearancePolicy.from_config(cfg.contact, cfg.geometry))
    q = np.zeros(pc.nq)
    centres, _ = pc.sphere_centers_numeric(q)
    tip = np.asarray(centres)[list(pc.sphere_link_names).index("ee_finger_r1")]
    cloud = PointCloud(tip + np.random.default_rng(0).normal(0.0, 0.01, (60, 3)))
    target = TargetGeometry(
        id=0, points=cloud.points, point_indices=np.arange(len(cloud.points)),
        centroid=cloud.points.mean(axis=0), bounding_geometry=fit_sphere(cloud.points),
        attention_score=0.9, confidence=0.9)
    field = types.SimpleNamespace(stats={"n_occupied": 10}, has_target_free=True)
    cs = build_constraint_set(
        builder, candidates=[], support_surfaces=[], target=target, robot_state=q,
        phase=Phase.GRASP, esdf=field, build_spec=False,
        contact_context=ContactPolicyContext.make(Phase.GRASP, [Manipulator.RIGHT]))
    return builder, cs


def test_every_palm_sphere_of_the_active_hand_reads_the_target_free_layer(pc, permission):
    builder, cs = permission
    names = np.asarray(builder.sphere_link_names)
    assert names.size == 466
    mask = np.asarray(cs.target_field_exclude, bool)
    expected = np.isin(names, ("ee_right", "ee_finger_r1", "ee_finger_r2"))
    assert np.array_equal(mask, expected)
    assert int(mask[names == "ee_right"].sum()) == 121
    assert not mask[names == "ee_left"].any()
    relaxed = cs.manipulated_link_margin < builder.clearance_policy.safety_margin
    assert np.array_equal(relaxed, expected)
    rec = cs.metrics["target_permission"]
    assert "ee_right" in rec["authorized_links"] and "ee_left" not in rec["authorized_links"]
    assert rec["target_field_exclude"] == mask.tolist()


def test_the_trajopt_contact_rows_hold_every_palm_row(permission):
    """T44 I2 `target_contact_rows_of` (trajopt 쪽 권한 행) 도 손바닥 121 행 전부, 반대 손은 0."""
    from benchmark.trajopt.linearize import target_contact_rows_of

    builder, cs = permission
    names = np.asarray(builder.sphere_link_names)
    for mask, margin in ((cs.target_field_exclude, cs.manipulated_link_margin),
                         (None, cs.manipulated_link_margin)):
        rows = target_contact_rows_of(mask, None, margin, "exclude_authorized", manipulated=True)
        assert rows is not None and rows.shape == (466,)
        assert rows[names == "ee_right"].all() and int(rows[names == "ee_right"].sum()) == 121
        assert not rows[names == "ee_left"].any()
        assert np.array_equal(rows, np.isin(names, ("ee_right", "ee_finger_r1", "ee_finger_r2")))


def test_announce_names_the_sphere_counts(pc, caplog):
    from benchmark.trajopt import serve_safe

    with caplog.at_level("WARNING"):
        serve_safe.announce_palm_cover(pc.palm_cover_report)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "T45 A10" in text and "270 → 466" in text and "ee_right 121 구" in text
    assert "넘침 ≤ 5.0 mm" in text
