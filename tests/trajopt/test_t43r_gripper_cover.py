"""T43 R — 그리퍼 질의 구가 실제 collision mesh 를 덮는다 (`--gripper-cover`, 기본 꺼짐).

T40 I: E3b 의 손가락 질의 구는 손가락 collision mesh 정점의 4.3 % 만 담았고, 막대에 가장 가까운 손 점은
가장 가까운 구 밖 1.7–8.2 mm 였다. 사용자 지시는 "그리퍼 구는 최소로 (과일을 잡을 수 있게)" — 그래서
반지름을 키우지 않고 **내접 구를 더해** 덮는다 (`experiments/sources/gripper_cover.py`).

이 파일이 고정하는 것:

1. **꺼지면 E3b 서버의 모델 그대로** — 구 이름 · 반지름이 기록 (T40 G E3b) 과 비트 단위로 같고, 덮개를
   켜도 앞 122 구는 같다. CLI 를 안 주면 `gripper_cover` 가 아예 넘어가지 않는다.
2. **덮임 문턱 (양손)** — link 마다 바깥 면 표본의 99.5 % 이상이 구 합집합에서 2.5 mm 안, 최대 4 mm,
   손가락 관절 범위 전체에서. 덮개 없는 모델은 16–35 % 였다.
3. **덮개는 mesh 밖으로 안 나온다 · 파지 면 쪽으로 안 나온다** — 모든 덮개 구가 한 볼록 조각 안 (정확),
   손가락 안쪽 경계 · 손바닥 통로 경계를 지킨다, T26 max opening 이 그대로다.
4. **파지에 영향 없음** — 기록 replay (B ep1807 · ep1968 파지 구간, E3b 정책): 덮개 행이 target ·
   지지면 · 장애물 어느 분류로도 새로 위반하지 않는다. 쥔 물체의 자기충돌 mask 는 덮개 구도 같은 link
   이름으로 받는다.
5. **C1807 막대 접촉** — 접촉 직전 행 (T40 I) 에서 덮개 구는 막대까지 mesh 거리 + 2.5 mm 안이고
   mesh 보다 가깝지 않다; 접촉 청크 (130) 의 TO 거리장에서 덮개 행은 위반 (< 0) 을 보는데 덮개 없는
   모델은 못 본다.
6. 이름 뒤바뀜: URDF `ee_finger_l1` 의 덮개는 MJCF `ee_finger_l2` 의 mesh 위에 있다 (접촉 증인점).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

MODEL = pathlib.Path(
    "/mnt/dev/work/pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml")
#: E3b 서버 flag (`outputs/verify/T39/p/start_server.sh`).
E3B = dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
RUN = pathlib.Path("/mnt/dev/work/outputs/verify/T40/G/runs/E3b/server_8240_20261003T135313/"
                   "server_constraints/run_0000")
CASE = pathlib.Path("/mnt/dev/work/outputs/verify/T40/I/raw/case_C1807.json")
LEFT = ("ee_left", "ee_finger_l1", "ee_finger_l2")
MAX_GAP = 0.0025


# =============================================================================================
# fixtures
# =============================================================================================


@pytest.fixture(scope="module")
def mj():
    mujoco = pytest.importorskip("mujoco")
    if not MODEL.exists():  # pragma: no cover - asset-dependent
        pytest.skip(f"model not present: {MODEL}")
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    return mujoco, model


@pytest.fixture(scope="module")
def scene(mj):
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene

    mujoco, model = mj
    return TransportScene.attach(model, mujoco.MjData(model))


@pytest.fixture(scope="module")
def bare(scene):
    from benchmark.ag3s.experiments.reports.grounding_report import (
        GRIPPER_LINKS, build_constraint_robot_model)

    return build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=E3B)


@pytest.fixture(scope="module")
def covered(scene):
    from benchmark.ag3s.experiments.reports.grounding_report import (
        GRIPPER_LINKS, build_constraint_robot_model)

    return build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=E3B,
                                        gripper_cover=True)


@pytest.fixture(scope="module")
def surfaces(mj, bare):
    from benchmark.ag3s.experiments.reports.grounding_report import GRIPPER_LINKS
    from benchmark.ag3s.experiments.sources.gripper_cover import link_surface

    _, model = mj
    return {link: link_surface(model, bare, link) for link in GRIPPER_LINKS}


def _finger_values(x):
    return {"gripper_finger_l1": -x, "gripper_finger_l2": x,
            "gripper_finger_r1": -x, "gripper_finger_r2": x}


@pytest.fixture(scope="module")
def metrics(mj, bare, covered, surfaces):
    from benchmark.ag3s.experiments.reports.grounding_report import GRIPPER_LINKS
    from benchmark.ag3s.experiments.sources.gripper_cover import coverage_metrics

    _, model = mj
    fv = [_finger_values(x) for x in (0.0, 0.025, 0.05)]
    return {name: coverage_metrics(model, rob, GRIPPER_LINKS, finger_values=fv, surfaces=surfaces)
            for name, rob in (("bare", bare), ("covered", covered))}


def _kabsch(a, b):
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rot = vt.T @ d @ u.T
    return rot, cb - rot @ ca


def _place(bare, model, names_rec, world):
    """기록의 구 중심 (`names_rec` 순) → link 자세 (Kabsch, 덮개 없는 모델의 local 구) → `model` 의 구."""
    names_rec = np.asarray(names_rec)
    out = np.zeros((model.n_spheres, 3))
    names = np.asarray(model.sphere_link_names)
    for link in sorted(set(names)):
        c0, _ = bare.local_spheres(link)
        rot, t = _kabsch(c0, world[names_rec == link])
        assert np.abs(c0 @ rot.T + t - world[names_rec == link]).max() < 1e-5   # 같은 모델
        cl, _ = model.local_spheres(link)
        out[names == link] = cl @ rot.T + t
    return out


# =============================================================================================
# 1. 꺼지면 그대로
# =============================================================================================


def test_off_is_the_model_the_e3b_server_used(bare, covered):
    """덮개 없는 모델 = 기록 (T40 G E3b) 의 구 이름 · 반지름 (float32 비트), 덮개를 켜도 앞은 같다."""
    if not RUN.exists():
        pytest.skip("T40 G E3b 기록이 없다")
    z = np.load(RUN / "chunk_00130.npz", allow_pickle=True)
    names = json.loads(str(z["summary_json"]))["sphere_link_names"]
    assert list(bare.sphere_link_names) == names
    assert np.array_equal(bare.radii.astype(np.float32), z["sphere_radii"])
    n = bare.n_spheres
    assert list(covered.sphere_link_names[:n]) == names
    assert np.array_equal(covered.radii[:n], bare.radii)
    q = np.random.default_rng(3).uniform(-1.0, 1.0, bare.nq)
    a, ra = bare.sphere_centers_numeric(q)
    b, rb = covered.sphere_centers_numeric(q)
    assert np.array_equal(b[:n], a) and np.array_equal(rb[:n], ra)
    assert not hasattr(bare, "gripper_cover_report")


def test_off_keyword_paths_are_the_same_model(scene, bare):
    from benchmark.ag3s.experiments.reports.grounding_report import (
        GRIPPER_LINKS, build_constraint_robot_model)

    for off in (None, False, {}):
        m = build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=E3B,
                                         gripper_cover=off)
        assert m.n_spheres == bare.n_spheres
        assert np.array_equal(m.radii, bare.radii)
        assert m.sphere_roles == bare.sphere_roles


def test_cli_default_passes_nothing_and_the_flag_builds_options():
    from benchmark.trajopt import serve_safe

    base = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
    args = serve_safe.build_parser().parse_args(base)
    assert args.gripper_cover is None
    assert serve_safe.gripper_cover_options(args) is None
    on = serve_safe.build_parser().parse_args(base + ["--gripper-cover"])
    assert serve_safe.gripper_cover_options(on) == {"hands": "both"}
    left = serve_safe.build_parser().parse_args(
        base + ["--gripper-cover", "left", "--gripper-cover-max-gap", "3"])
    assert serve_safe.gripper_cover_options(left) == {"hands": "left", "max_gap": 0.003}
    lone = serve_safe.build_parser().parse_args(base + ["--gripper-cover-max-gap", "3"])
    with pytest.raises(SystemExit):
        serve_safe.gripper_cover_options(lone)
    bad = serve_safe.build_parser().parse_args(base + ["--gripper-cover",
                                                       "--gripper-cover-max-gap", "0"])
    with pytest.raises(SystemExit):
        serve_safe.gripper_cover_options(bad)


def test_announce_is_silent_when_off(caplog):
    from benchmark.trajopt import serve_safe

    with caplog.at_level("WARNING"):
        serve_safe.announce_gripper_cover(None)
    assert not caplog.records


def test_exact_sphere_role_is_used_as_is():
    """덮개 구는 배율 · 간격 inflation 을 받지 않는다 (0.05 배가 걸리면 덮개가 아니다)."""
    from benchmark.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf
    from benchmark.ag3s.robot_models.urdf_sphere_chain import EXACT_SPHERE_ROLE, UrdfCapsule

    urdf = parse_urdf(RBY1_URDF)
    origin = np.eye(4)
    origin[:3, 3] = [0.01, -0.02, 0.03]
    cap = UrdfCapsule(link="ee_left", origin=origin, radius=0.004, length=0.0,
                      role=EXACT_SPHERE_ROLE)
    m = UrdfSphereChain(urdf, extra_capsules=[cap], link_filter=("ee_left",), **E3B)
    c, r = m.local_spheres("ee_left")
    assert c.shape == (1, 3) and np.allclose(c[0], origin[:3, 3]) and r[0] == 0.004
    assert not m.coverage_shortfall
    capped = UrdfSphereChain(urdf, extra_capsules=[cap], link_filter=("ee_left",),
                             max_sphere_radius=0.003)
    assert capped.local_spheres("ee_left")[1][0] == 0.003
    assert capped.coverage_shortfall          # 상한이 덮개를 깎으면 말한다


# =============================================================================================
# 2. 덮임 문턱 (양손, 손가락 관절 범위)
# =============================================================================================


@pytest.mark.parametrize("link", ["ee_left", "ee_right", "ee_finger_l1", "ee_finger_l2",
                                  "ee_finger_r1", "ee_finger_r2"])
def test_outer_surface_is_covered_within_the_gap(metrics, link):
    before = metrics["bare"][link]["own"]["outer"]
    after = metrics["covered"][link]["own"]["outer"]
    assert before["frac_within_max_gap"] < 0.40          # 덮개 없는 모델: 손바닥 16 % · 손가락 35 %
    assert after["frac_within_max_gap"] >= 0.995
    assert after["p95_gap_mm"] <= MAX_GAP * 1e3
    assert after["max_gap_mm"] <= 4.0
    for row in metrics["covered"][link]["gripper_union"]:   # 닫힘 · 반 · 열림
        assert row["outer"]["max_gap_mm"] <= 4.0


def test_both_hands_get_the_same_cover(covered):
    rep = covered.gripper_cover_report["links"]
    assert set(rep) == {"ee_left", "ee_right", "ee_finger_l1", "ee_finger_l2",
                        "ee_finger_r1", "ee_finger_r2"}
    for a, b in (("ee_finger_l1", "ee_finger_r1"), ("ee_finger_l2", "ee_finger_r2")):
        assert rep[a]["n_spheres"] == rep[b]["n_spheres"] > 0
    assert abs(rep["ee_left"]["n_spheres"] - rep["ee_right"]["n_spheres"]) <= 10


def test_radii_are_not_inflated(bare, covered):
    """덮개 구 반지름 ≤ 그 link 의 지금 구 최대 반지름."""
    from benchmark.ag3s.robot_models.urdf_sphere_chain import EXACT_SPHERE_ROLE

    roles = np.asarray(covered.sphere_roles)
    names = np.asarray(covered.sphere_link_names)
    for link in set(names):
        cap = bare.local_spheres(link)[1].max()
        sel = (names == link) & (roles == EXACT_SPHERE_ROLE)
        assert sel.any()
        assert covered.radii[sel].max() <= cap + 1e-12


# =============================================================================================
# 3. mesh 밖으로 · 파지 쪽으로 안 나온다
# =============================================================================================


def test_every_cover_sphere_is_inscribed_and_respects_the_grasp_side(bare, covered, surfaces):
    from benchmark.ag3s.experiments.sources.gripper_cover import grasp_limits, hull_depth
    from benchmark.ag3s.robot_models.urdf_sphere_chain import EXACT_SPHERE_ROLE

    names = np.asarray(covered.sphere_link_names)
    roles = np.asarray(covered.sphere_roles)
    for link, surf in surfaces.items():
        sel = (names == link) & (roles == EXACT_SPHERE_ROLE)
        idx = np.flatnonzero(sel)
        c = np.asarray([covered._local_spheres[i][1] for i in idx])
        r = covered.radii[idx]
        assert np.all(hull_depth(c, surf.hulls) >= r - 1e-6)     # 한 볼록 조각 안 (정확)
        inner, channel = grasp_limits(bare, link)
        if inner is not None:
            u_in, h = inner
            assert np.all(c @ u_in + r <= h + 1e-9)               # 손가락 안쪽으로 안 나온다
        if channel is not None:
            u, half, tip, z_lim = channel
            in_ch = np.abs(c @ u) < half + r
            assert np.all(c[in_ch] @ tip + r[in_ch] <= z_lim + 1e-9)   # 손가락 사이로 안 내려온다
        assert grasp_limits(covered, link)[0] is None or np.allclose(
            grasp_limits(covered, link)[0][1], inner[1])


def test_max_opening_is_unchanged(bare, covered):
    a = bare.gripper_openings()
    b = covered.gripper_openings()
    assert [x["inner_gap_m"] for x in a] == [x["inner_gap_m"] for x in b]


# =============================================================================================
# 4. 파지에 영향 없음 — 기록 replay
# =============================================================================================

#: B ep1807 (chunk 12–19, t 13–20: 접근 · latched · closing · held) · B ep1968 (142–149, t 12–19).
GRASP_CHUNKS = tuple(range(12, 20)) + tuple(range(142, 150))


def _replay_rows(bare, model, z, *, auth=LEFT):
    from benchmark.trajopt.experiments.fixed_input_ablation import build_field
    from benchmark.trajopt.linearize import CollisionLinearizer, SceneSnapshot

    summary = json.loads(str(z["summary_json"]))
    centres = _place(bare, model, summary["sphere_link_names"],
                     z["sphere_centres"].astype(np.float64))[None]
    cloud = z["filtered_cloud"].astype(np.float64)
    sup = cloud[z["support_mask"]]
    sup = sup[sup[:, 2] > 0.4]
    mu = sup.mean(0)
    n = np.linalg.svd(sup - mu, full_matrices=False)[2][2]
    n = n * np.sign(n[2])
    names = np.asarray(model.sphere_link_names)
    radii = model.radii
    mask = np.isin(names, auth)
    lin = object.__new__(CollisionLinearizer)
    lin.robot_radii = radii
    lin.query_radii = radii
    lin.n_spheres = radii.size
    lin._attached_local = None
    scene = SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
        candidate_active=np.zeros(0, bool), d_safe=np.zeros((radii.size, 0)),
        plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=radii, candidate_ids=np.zeros(0, np.int64),
        esdf=build_field(z, outside_distance=1.0), esdf_margin=0.0,
        manipulated_points=z["target_points"].astype(np.float64),
        manipulated_link_margin=np.where(mask, 0.02, 0.05),
        target_free_mask=mask if "esdf_free_distance" in z.files else None,
        # 분류만 얻으려고 켠다 (1 nm) — 값은 아래에서 0 여유로 따로 잰다
        obstacle_margin=1e-9, support_planes=np.array([[*n, float(n @ mu)]]))
    lin._esdf_clearance(centres, scene)
    cls = lin._last_obstacle_class[0].copy()
    import dataclasses

    clear = lin._esdf_clearance(centres, dataclasses.replace(scene, obstacle_margin=0.0))[0, :, 0]
    return names, clear, cls


@pytest.mark.skipif(not RUN.exists(), reason="T40 G E3b 기록이 없다")
def test_recorded_grasp_rows_do_not_newly_violate(bare, covered):
    from benchmark.trajopt.linearize import OBSTACLE_CLASSES

    n = bare.n_spheres
    seen = 0
    for k in GRASP_CHUNKS:
        z = np.load(RUN / f"chunk_{k:05d}.npz", allow_pickle=True)
        _, c_bare, cls_bare = _replay_rows(bare, bare, z)
        _, c_cov, cls_cov = _replay_rows(bare, covered, z)
        assert np.array_equal(c_cov[:n], c_bare)             # 앞 122 행은 그대로
        new = slice(n, None)
        violating = c_cov[new] < 0.0
        for name in ("target", "held", "support", "obstacle"):
            hit = violating & (cls_cov[new] == OBSTACLE_CLASSES.index(name))
            assert not hit.any(), (k, name, float(c_cov[new][hit].min()))
        seen += 1
    assert seen == len(GRASP_CHUNKS)


def test_cover_spheres_take_the_held_object_exemption_by_link_name(covered):
    """쥔 물체 ↔ 로봇 자기충돌 mask 는 link 이름으로 면제한다 — 덮개 구도 같은 이름이라 같이 면제."""
    from benchmark.ag3s.constraints.attached import self_collision_mask
    import types

    allowed = types.SimpleNamespace(allowed_contact_links=frozenset(LEFT))
    mask = self_collision_mask(allowed, covered.sphere_link_names)
    names = np.asarray(covered.sphere_link_names)
    assert np.all(mask[np.isin(names, LEFT)] == 0.0)
    assert np.all(mask[~np.isin(names, LEFT)] == 1.0)


# =============================================================================================
# 5. C1807 막대 접촉
# =============================================================================================


@pytest.mark.skipif(not CASE.exists(), reason="T40 I 원자료가 없다")
def test_c1807_event_row_cover_sees_the_bar_like_the_mesh(bare, covered):
    case = json.loads(CASE.read_text())
    ev = case["event"]
    sw = ev["spheres_world_at_event_row"]
    world = np.asarray(sw["centres"], np.float64)
    bar = case["bar_geom"]
    bc = np.asarray(bar["centre_world"], np.float64)
    ba = np.asarray(bar["axis_world"], np.float64)
    ba /= np.linalg.norm(ba)
    half, rb = bar["half_length_mm"] / 1e3, bar["radius_mm"] / 1e3

    def to_bar(p):
        t = np.clip((p - bc) @ ba, -half, half)
        return np.linalg.norm(p - (bc + t[:, None] * ba), axis=1) - rb

    placed = _place(bare, covered, sw["links"], world)
    names = np.asarray(covered.sphere_link_names)
    n = bare.n_spheres
    for link in ("ee_left", "ee_finger_l1"):
        mesh = ev["per_link_min_geom_to_bar_mm_at_row"][link] / 1e3
        sel_b = names[:n] == link
        sel_c = names == link
        d_bare = float((to_bar(placed[:n][sel_b]) - covered.radii[:n][sel_b]).min())
        d_cov = float((to_bar(placed[sel_c]) - covered.radii[sel_c]).min())
        assert d_cov < d_bare                                # 덮개가 막대에 더 가까운 손 면을 본다
        assert mesh - 1e-4 <= d_cov <= mesh + MAX_GAP        # mesh 보다 가깝지 않고 틈 안


@pytest.mark.skipif(not RUN.exists(), reason="T40 G E3b 기록이 없다")
@pytest.mark.parametrize("k", [129, 130])
def test_c1807_contact_chunk_field_rows_see_the_contact(bare, covered, k):
    """접촉 청크 (t 27 · 28) 의 q_now 에서 TO 거리장: 덮개 없는 손 행은 위반이 없거나 적고, 덮개 행은 본다."""
    z = np.load(RUN / f"chunk_{k:05d}.npz", allow_pickle=True)
    names, c_bare, _ = _replay_rows(bare, bare, z)
    names_c, c_cov, _ = _replay_rows(bare, covered, z)
    finger = names_c == "ee_finger_l1"
    n = bare.n_spheres
    new = np.zeros(names_c.size, bool)
    new[n:] = True
    assert c_cov[finger & new].min() < 0.0
    assert c_cov[finger & new].min() < c_bare[names == "ee_finger_l1"].min()
    assert (c_cov[finger] < 0).sum() > (c_bare[names == "ee_finger_l1"] < 0).sum()


# =============================================================================================
# 6. 이름 뒤바뀜
# =============================================================================================


@pytest.mark.skipif(not CASE.exists(), reason="T40 I 원자료가 없다")
def test_urdf_l1_cover_sits_on_the_mjcf_l2_mesh(bare, surfaces):
    """접촉 증인점 (MJCF `ee_finger_l2` 의 geom#1086 위) 은 URDF `ee_finger_l1` frame 의 collision 껍질
    표면에 있고, URDF `ee_finger_l2` frame 의 껍질에서는 멀다 — 덮개가 올바른 mesh 위에 있다."""
    from benchmark.ag3s.experiments.sources.gripper_cover import hull_depth

    case = json.loads(CASE.read_text())
    ev = case["event"]
    sw = ev["spheres_world_at_event_row"]
    world = np.asarray(sw["centres"], np.float64)
    names = np.asarray(sw["links"])
    witness = np.asarray(ev["witness_on_geom_world"], np.float64)
    assert surfaces["ee_finger_l1"].body == "ee_finger_l2"
    assert surfaces["ee_finger_l2"].body == "ee_finger_l1"
    depth = {}
    for link in ("ee_finger_l1", "ee_finger_l2"):
        c0, _ = bare.local_spheres(link)
        rot, t = _kabsch(c0, world[names == link])
        local = (witness - t) @ rot
        depth[link] = float(hull_depth(local[None], surfaces[link].hulls)[0])
    assert abs(depth["ee_finger_l1"]) < 1e-4                  # 표면 위 (0.1 mm 안)
    assert depth["ee_finger_l2"] < -0.02                      # 다른 손가락에서는 2 cm 넘게 밖
