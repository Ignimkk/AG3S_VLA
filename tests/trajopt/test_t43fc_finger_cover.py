"""T43 FC — 손가락마다 따로, 실제 굵기의 몸통 구 (`--finger-cover`, 기본 꺼짐).

사용자 판정 (2026-10-06 10:00, "B"): 그리퍼마다 두 손가락을 **각각** 구로 감싼다. 두 손가락을 한
덩어리로 감싸지 않는다. 기준은 손가락 collision mesh 다 (URDF 상당 capsule 은 mesh 보다 두껍다).

이 파일이 고정하는 것:

1. **꺼지면 그대로** — `finger_cover=None/False/{}` 와 인자 없는 호출이 같은 모델이고, E3b 기록의 구
   이름 · 반지름과 비트로 같다. CLI 를 안 주면 아무것도 넘어가지 않는다.
2. **덮임 · 넘침** — 손가락마다 collision 겉면 표본 (0.5 mm, 파지 면 포함) 과 mesh 정점이 **전부** 구
   안에 있고, 구는 mesh 밖으로 `overflow` (+ 표본 여유 0.3 mm) 넘게 나가지 않는다.
3. **손가락은 따로** — 열린 손에서 두 손가락의 구 합집합이 서로 떨어져 있고 (사이 = mesh 안쪽 간격
   − 2 overflow), 손가락 사이 한가운데 점은 어느 구에도 들지 않는다. 손바닥 구는 그대로다.
4. 반지름은 배율 (`capsule_radius_scale`) 을 받지 않는다. 네 손가락은 같은 구 (link frame).
5. `--gripper-cover` 와 같이 주면 손가락 몸통 구는 그대로이고 덮개는 손바닥에 붙는다.
6. 배선: CLI → `build_ag3s` → `build_constraint_robot_model(finger_cover=...)`, 끄면 키가 없다.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

MODEL = pathlib.Path(
    "/mnt/dev/work/pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml")
#: E3b 서버 flag (V5 도 같다).
E3B = dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
RUN = pathlib.Path("/mnt/dev/work/outputs/verify/T40/G/runs/E3b/server_8240_20261003T135313/"
                   "server_constraints/run_0000")
FINGERS = ("ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2")
OVERFLOW = 0.0025


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


def _build(scene, **kw):
    from benchmark.ag3s.experiments.reports.grounding_report import (
        GRIPPER_LINKS, build_constraint_robot_model)

    opts = dict(E3B)
    opts.update(kw.pop("sphere_options", {}))
    return build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=opts, **kw)


@pytest.fixture(scope="module")
def bare(scene):
    return _build(scene)


@pytest.fixture(scope="module")
def fc(scene):
    return _build(scene, finger_cover=True)


@pytest.fixture(scope="module")
def surfaces(mj, bare):
    from benchmark.ag3s.experiments.sources.gripper_cover import link_surface

    _, model = mj
    return {link: link_surface(model, bare, link, spacing=0.0005) for link in FINGERS}


def _finger_q(x):
    return {"gripper_finger_l1": -x, "gripper_finger_l2": x,
            "gripper_finger_r1": -x, "gripper_finger_r2": x}


# =============================================================================================
# 1. 꺼지면 그대로
# =============================================================================================


def test_off_is_the_model_the_e3b_server_used(scene, bare):
    if not RUN.exists():
        pytest.skip("T40 G E3b 기록이 없다")
    z = np.load(RUN / "chunk_00130.npz", allow_pickle=True)
    names = json.loads(str(z["summary_json"]))["sphere_link_names"]
    assert list(bare.sphere_link_names) == names
    assert np.array_equal(bare.radii.astype(np.float32), z["sphere_radii"])
    assert not hasattr(bare, "finger_cover_report")


@pytest.mark.parametrize("off", [None, False, {}])
def test_off_keyword_paths_are_the_same_model(scene, bare, off):
    m = _build(scene, finger_cover=off)
    assert m.sphere_link_names == bare.sphere_link_names
    assert np.array_equal(m.radii, bare.radii)
    assert m.sphere_roles == bare.sphere_roles
    q = np.random.default_rng(5).uniform(-1.0, 1.0, bare.nq)
    a, ra = bare.sphere_centers_numeric(q)
    b, rb = m.sphere_centers_numeric(q)
    assert np.array_equal(a, b) and np.array_equal(ra, rb)
    assert not hasattr(m, "finger_cover_report")


def test_cli_default_passes_nothing_and_the_flag_builds_options(caplog):
    from benchmark.trajopt import serve_safe

    base = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
    p = serve_safe.build_parser()
    args = p.parse_args(base)
    assert args.finger_cover is None and args.finger_cover_overflow is None
    assert serve_safe.finger_cover_options(args) is None
    assert serve_safe.finger_cover_options(p.parse_args(base + ["--finger-cover"])) == \
        {"hands": "both"}
    assert serve_safe.finger_cover_options(p.parse_args(
        base + ["--finger-cover", "left", "--finger-cover-overflow", "2"])) == \
        {"hands": "left", "overflow": 0.002}
    with pytest.raises(SystemExit):
        serve_safe.finger_cover_options(p.parse_args(base + ["--finger-cover-overflow", "2"]))
    for bad in ("0", "-1", "nan"):
        with pytest.raises(SystemExit):
            serve_safe.finger_cover_options(p.parse_args(
                base + ["--finger-cover", "--finger-cover-overflow", bad]))
    with caplog.at_level("WARNING"):
        serve_safe.announce_finger_cover(None)
    assert not caplog.records


@pytest.mark.parametrize("value", [None, {"hands": "both", "overflow": 0.002}])
def test_build_ag3s_passes_finger_cover_only_when_given(monkeypatch, value):
    """`build_ag3s` → `build_constraint_robot_model`: 끄면 `finger_cover` 키가 아예 없다."""
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
                              finger_cover=value)
    if value is None:
        assert "finger_cover" not in seen
    else:
        assert seen["finger_cover"] == value


def test_no_safe_and_no_perception_reject_the_flag():
    from benchmark.trajopt import serve_safe

    base = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
    for extra in (["--no-safe"], ["--no-perception"]):
        p = serve_safe.build_parser()
        with pytest.raises(SystemExit):
            serve_safe.reject_bad_flag_combinations(p, p.parse_args(base + extra + ["--finger-cover"]))
    p = serve_safe.build_parser()
    serve_safe.reject_bad_flag_combinations(p, p.parse_args(base + ["--finger-cover"]))


# =============================================================================================
# 2. 덮임 · 넘침 (손가락마다)
# =============================================================================================


@pytest.mark.parametrize("link", FINGERS)
def test_every_finger_surface_sample_and_vertex_is_inside(fc, surfaces, link):
    from benchmark.ag3s.experiments.sources.gripper_cover import _gap, finger_body_targets

    surf = surfaces[link]
    c, r = fc.local_spheres(link)
    targets = surf.points[finger_body_targets(surf)]
    assert targets.shape[0] > 15000
    assert _gap(targets, c, r).max() <= 1e-9
    assert _gap(surf.vertices, c, r).max() <= 1e-9          # collision mesh 정점 971 개
    rep = fc.finger_cover_report["links"][link]
    assert rep["n_uncovered"] == 0 and rep["max_gap_mm"] == 0.0


@pytest.mark.parametrize("link", FINGERS)
def test_spheres_leave_the_mesh_by_at_most_the_overflow(fc, surfaces, link):
    from benchmark.ag3s.experiments.sources.gripper_cover import _sampled_overflow, hull_depth

    surf = surfaces[link]
    c, r = fc.local_spheres(link)
    assert _sampled_overflow(c, r, surf.hulls, surf.points) <= OVERFLOW + 0.0003
    # 반지름 − overflow 는 mesh 안 깊이 이하 (중심이 mesh 안이고 구의 '알맹이' 가 mesh 안)
    assert np.all(hull_depth(c, surf.hulls) > 0.0)
    assert r.min() >= OVERFLOW - 1e-12
    # 실제 굵기: 손가락 판 두께 9.5 mm (x −3.3…6.2 mm) — 구의 x 범위가 mesh + overflow 안
    lo, hi = surf.vertices.min(0), surf.vertices.max(0)
    assert np.all((c - r[:, None]).min(0) >= lo - OVERFLOW - 1e-9)
    assert np.all((c + r[:, None]).max(0) <= hi + OVERFLOW + 1e-9)


def test_the_bare_model_did_not_cover_the_fingers(bare, surfaces):
    """꺼진 모델의 손가락 구 (2.6–3.3 mm) 는 표본의 5 % 도 못 담는다 — 바꾼 이유."""
    from benchmark.ag3s.experiments.sources.gripper_cover import _gap

    c, r = bare.local_spheres("ee_finger_l1")
    g = _gap(surfaces["ee_finger_l1"].points, c, r)
    assert np.mean(g <= 0) < 0.05 and g.max() > 0.012


# =============================================================================================
# 3. 손가락은 따로 · 손바닥 그대로 · 배율 무관 · 네 손가락 같음
# =============================================================================================


def test_palm_spheres_are_unchanged_and_finger_chain_is_replaced(bare, fc):
    from benchmark.ag3s.robot_models.urdf_sphere_chain import FINGER_BODY_ROLE

    for palm in ("ee_left", "ee_right"):
        a, ra = bare.local_spheres(palm)
        b, rb = fc.local_spheres(palm)
        assert np.array_equal(a, b) and np.array_equal(ra, rb)
    n_palm = sum(1 for n in bare.sphere_link_names if n in ("ee_left", "ee_right"))
    assert fc.sphere_link_names[:n_palm] == bare.sphere_link_names[:n_palm]
    roles = np.asarray(fc.sphere_roles)
    names = np.asarray(fc.sphere_link_names)
    assert set(roles[np.isin(names, FINGERS)]) == {FINGER_BODY_ROLE}
    assert FINGER_BODY_ROLE not in set(bare.sphere_roles)
    rep = fc.finger_cover_report
    assert rep["n_spheres_before"] == bare.n_spheres == 122
    assert rep["n_spheres_after"] == fc.n_spheres
    assert not [s for s in fc.coverage_shortfall if s.link in FINGERS]


def test_four_fingers_get_the_same_spheres_in_their_own_frame(fc):
    c0, r0 = fc.local_spheres(FINGERS[0])
    for link in FINGERS[1:]:
        c, r = fc.local_spheres(link)
        assert np.array_equal(c, c0) and np.array_equal(r, r0)


def test_radius_scale_does_not_touch_finger_body_spheres(scene, fc):
    other = _build(scene, finger_cover=True, sphere_options={"capsule_radius_scale": 1.0})
    for link in FINGERS:
        a, ra = fc.local_spheres(link)
        b, rb = other.local_spheres(link)
        assert np.array_equal(a, b) and np.array_equal(ra, rb)


@pytest.mark.parametrize("x", [0.05, 0.035, 0.0167])
def test_the_two_fingers_stay_separate(fc, bare, x):
    """열린 손 (한쪽 50 · 35 · 16.7 mm — 사과 반폭 33.5 mm 근처까지): 두 손가락 구 합집합 사이가 비어
    있고, 손가락 사이 한가운데 점은 어느 구에도 들지 않는다."""
    from benchmark.ag3s.robot_models.urdf_sphere_chain import _joint_transform, _NumpyBackend

    be = _NumpyBackend()
    for pair in fc.gripper_joint_pairs():
        frames = []
        for joint_name, child in zip(pair["joints"], pair["children"]):
            joint = fc.model.joint_by_name(joint_name)
            q = _finger_q(x)[joint_name]
            T = np.asarray(_joint_transform(joint, q, be), np.float64)
            c, r = fc.local_spheres(child)
            frames.append((c @ T[:3, :3].T + T[:3, 3], r))
        (ca, ra), (cb, rb) = frames
        d = np.linalg.norm(ca[:, None] - cb[None], axis=2) - ra[:, None] - rb[None]
        assert d.min() > 0.0, (pair["parent_link"], x, d.min())
        mid = 0.5 * (ca.mean(0) + cb.mean(0))
        for c, r in frames:
            assert np.min(np.linalg.norm(c - mid, axis=1) - r) > 0.0


def test_max_opening_is_the_mesh_gap_minus_two_overflows(bare, fc):
    """T26 max opening: 102.35 mm (축선 사슬) → mesh 안쪽 간격 99.4 − 2 × 2.5 ≈ 94.3 mm."""
    before = {o["parent_link"]: o["inner_gap_m"] for o in bare.gripper_openings()}
    after = {o["parent_link"]: o["inner_gap_m"] for o in fc.gripper_openings()}
    for palm in ("ee_left", "ee_right"):
        assert before[palm] == pytest.approx(0.10235, abs=5e-5)
        assert after[palm] == pytest.approx(0.106 - 2 * 0.0033 - 2 * OVERFLOW, abs=2e-4)
        assert after[palm] > 0.0670 + 0.02            # 사과 66.6 mm + 여유


def test_hands_option_and_overflow_option(scene, fc):
    left = _build(scene, finger_cover={"hands": "left", "overflow": 0.003})
    rep = left.finger_cover_report
    assert set(rep["links"]) == {"ee_finger_l1", "ee_finger_l2"}
    assert rep["options"] == {"hands": "left", "overflow": 0.003}
    assert rep["links"]["ee_finger_l1"]["n_spheres"] < \
        fc.finger_cover_report["links"]["ee_finger_l1"]["n_spheres"]
    from benchmark.ag3s.robot_models.urdf_sphere_chain import FINGER_BODY_ROLE

    roles = dict(zip(left.sphere_link_names, left.sphere_roles))
    assert roles["ee_finger_r1"] != FINGER_BODY_ROLE and roles["ee_finger_l1"] == FINGER_BODY_ROLE
    with pytest.raises(ValueError):
        _build(scene, finger_cover={"hands": "middle"})
    with pytest.raises(ValueError):
        _build(scene, finger_cover={"overflow": 0.002, "max_gap": 1})


def test_with_gripper_cover_the_finger_body_stays_and_palms_get_the_cover(scene, fc):
    both = _build(scene, finger_cover=True, gripper_cover=True)
    for link in FINGERS:
        from benchmark.ag3s.robot_models.urdf_sphere_chain import FINGER_BODY_ROLE

        names = np.asarray(both.sphere_link_names)
        roles = np.asarray(both.sphere_roles)
        a, ra = fc.local_spheres(link)
        sel = (names == link) & (roles == FINGER_BODY_ROLE)
        idx = np.nonzero(sel)[0]
        b = np.asarray([both._local_spheres[i][1] for i in idx])
        assert np.array_equal(a, b)
    rep = both.gripper_cover_report["links"]
    assert rep["ee_left"]["n_spheres"] > 100
    assert both.finger_cover_report == fc.finger_cover_report



# =============================================================================================
# 4. 기록 재생 — CT 손가락–막대 접촉 (V5 ep1925 s19251, release 뒤 t 270, MJCF ee_finger_l2 = URDF l1)
# =============================================================================================

CT_RUN = pathlib.Path("/mnt/dev/work/outputs/verify/T43/W/runs/V5/H/V5_ep1925_s19251")
CT_REPLAY = pathlib.Path("/mnt/dev/work/outputs/verify/T43/CT/replay/V5_H_V5_ep1925_s19251.npz")
CT_CHUNK = pathlib.Path("/mnt/dev/work/outputs/verify/T43/W/runs/V5/server_8283_20261006T011616/"
                        "server_constraints/run_0000/chunk_00226.npz")


def test_ct_contact_row_finger_body_sees_the_bar_in_the_recorded_field(mj, bare, fc):
    """접촉 행의 post 상태 (CT replay 의 손 자세 · 개도) 에서, 그 청크의 기록 거리장으로 URDF ee_finger_l1 의
    몸통 구는 막대를 침투로 본다 (d − r < 0, 축선 사슬보다 깊게). 층은 기록된 행과 맞는 쪽 (offline 과 같은 규칙)."""
    if not (CT_RUN.exists() and CT_REPLAY.exists() and CT_CHUNK.exists()):
        pytest.skip("T43 W / CT 기록이 없다")
    import mujoco

    from benchmark.ag3s.robot_models.gripper_state import FingerJointMap
    from benchmark.trajopt.experiments import fixed_input_ablation as FIA
    from benchmark.trajopt.types import ChunkLayout

    _, model = mj
    data = mujoco.MjData(model)
    frames = [json.loads(line) for line in open(CT_RUN / "frames.jsonl") if line.strip()]
    ctrl = {r["t_step"]: r for r in frames if r.get("kind") == "control"}
    plan = {r["seq"]: r for r in frames if r.get("kind") == "planning"}
    rz = np.load(CT_REPLAY)
    t = 270
    qpos = np.asarray(ctrl[t]["qpos"], float)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_ee")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    Ts = np.eye(4); Ts[:3, :3] = data.site_xmat[sid].reshape(3, 3); Ts[:3, 3] = data.site_xpos[sid]
    Tb = np.eye(4); Tb[:3, :3] = data.xmat[bid].reshape(3, 3); Tb[:3, 3] = data.xpos[bid]
    i = int(np.flatnonzero(rz["t"] == t)[0]) + 1
    Tp = np.eye(4); Tp[:3, :3] = rz["tcp_R"][i]; Tp[:3, 3] = rz["tcp"][i]
    M = Tp @ np.linalg.inv(Ts)                       # 이 행 동안의 손 강체 운동
    qadr = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): int(model.jnt_qposadr[j])
            for j in range(model.njnt)}
    q20 = FIA.q_now_from_qpos(qpos, qadr)
    params = FingerJointMap(bare).joint_values({"left": float(rz["finger_open_norm"][i])})
    z = np.load(CT_CHUNK, allow_pickle=True)
    field = FIA.build_field(z, 0.4)
    # 층: 기록된 계획 행 (8 스텝) 과 맞는 쪽 — URDF ee_finger_l1 의 축선 사슬로 고른다
    seq = 34
    qn = FIA.q_now_from_qpos(np.asarray(plan[seq]["qpos"], float), qadr)
    layout = ChunkLayout.rby1(bare.joint_names)
    Q = layout.full_q(layout.chunk_to_trajectory(np.asarray(z["refined_chunk"], float))[:, :8], qn).T
    s = json.loads(str(z["summary_json"]))
    bare.set_joint_parameters(s["finger_joints"]["joints"])
    sel0 = np.asarray(bare.sphere_link_names) == "ee_finger_l1"
    err = {}
    for name, fn in (("full", field.distance), ("free", field.target_free_distance)):
        rows = [np.asarray(fn(np.asarray(bare.sphere_centers_numeric(q)[0])[sel0])) - bare.radii[sel0]
                for q in Q]
        err[name] = float(np.abs(np.asarray(rows) - np.asarray(z["clearance"])[:, sel0]).max())
    layer = min(err, key=err.get)
    assert err[layer] < 1e-4
    fn = field.distance if layer == "full" else field.target_free_distance
    out = {}
    for name, rob in (("bare", bare), ("fc", fc)):
        rob.set_joint_parameters(params)
        C, R = rob.sphere_centers_numeric(q20)
        sel = np.asarray(rob.sphere_link_names) == "ee_finger_l1"
        W = (np.asarray(C)[sel] @ Tb[:3, :3].T + Tb[:3, 3]) @ M[:3, :3].T + M[:3, 3]
        Pb = (W - Tb[:3, 3]) @ Tb[:3, :3]
        out[name] = float((np.asarray(fn(Pb)) - np.asarray(R)[sel]).min())
    bare.set_joint_parameters(None)
    fc.set_joint_parameters(None)
    assert out["fc"] < -0.003                         # 실측 −4.2 mm (CT 침투 2.56 mm)
    assert out["fc"] < out["bare"]                    # 축선 사슬 −3.3 mm
