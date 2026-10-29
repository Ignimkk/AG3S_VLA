"""T46f A14b — arm_6 inset 0.5 mm 와 실험 자세 점검 (사용자 판정 2026-10-13 10:56 KST, A2 · A7 · A8).

| # | 성질 |
|---|---|
| 1 | inset 기본값은 arm_6 양손 0.5 mm 뿐 (`WRIST_COVER_INSET`). FT sensor · arm_5 구는 A14 와 비트로 같다 (A14 지문) |
| 2 | `inset={}` 이면 A14 형상 그대로 (arm_6 지문 — A14 코드로 뽑은 값) |
| 3 | inset 형상: 고른 표본 전부가 구 표면에서 inset 넘게 안쪽 · 반지름 공식 (넘침 · keep-out) 그대로 · 손 껍질 밖 |
| 4 | 잘못된 inset 은 거절 (음수 · nan · 손목 밖 link · gap / ilp 와 같이) |
| 5 | 실험 자세 점검 (`wrist_pose_check`): 정확한 점–삼각형 거리 · 볼록 조각 부호 거리 · 사전 고정 기준 값 ·
      손가락 parameter 대응은 위치로 (왼손 이름 뒤바뀜) · CL3 planning 행에서 (a) 침범 ≤ 허용오차 · frame 검사,
      보정 전 arm_5 는 침범 (T46d 재현) |
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest

from common.workspace import shared_storage_root, workspace_path

MODEL = pathlib.Path(workspace_path("robots/rby1/description/models/rby1a/mujoco/model_transport.xml"))
SERVER = dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
FC = {"hands": "both"}
CL3 = (shared_storage_root() / "results/development/20261012T155130+0900_218229ac6464_5fb5ff542c33"
       / "verify/T46/CL3")

#: A14 (inset 없음, worktree HEAD ab9b3fba + A14 diff) 의 손목 구 지문 — link frame 중심 (float64) + 반지름
#: (float64) 의 sha1 앞 16 자리. A14b 실행 전 `--links gripper_wrist` 모델에서 뽑았다
#: (`impl/T46f-A14b/a14_model_spheres.npz`).
A14_FINGERPRINT = {
    "link_left_arm_5": (122, "3373adf98e2d04b6"),
    "link_left_arm_6": (84, "6340eb094826edcd"),
    "FT_sensor_L": (216, "dedacaecd68d7ac8"),
    "link_right_arm_5": (127, "3f1558b7e6a5896c"),
    "link_right_arm_6": (82, "bb58841da2385173"),
    "FT_sensor_R": (221, "6a6940652d29c8b9"),
}


def _fingerprint(c, r) -> tuple[int, str]:
    c = np.ascontiguousarray(c, np.float64)
    r = np.ascontiguousarray(r, np.float64)
    return int(r.size), hashlib.sha1(c.tobytes() + r.tobytes()).hexdigest()[:16]


@pytest.fixture(scope="module")
def mj():
    mujoco = pytest.importorskip("mujoco")
    if not MODEL.exists():  # pragma: no cover - asset-dependent
        pytest.skip(f"model not present: {MODEL}")
    return mujoco, mujoco.MjModel.from_xml_path(str(MODEL))


@pytest.fixture(scope="module")
def scene(mj):
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    mujoco, model = mj
    return TransportScene.attach(model, mujoco.MjData(model))


def _build(scene, links="gripper", **kw):
    from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
    from AG3S.trajopt.serve_safe import constraint_link_filter

    return build_constraint_robot_model(scene, link_filter=constraint_link_filter(links),
                                        sphere_options=dict(SERVER), **kw)


@pytest.fixture(scope="module")
def full(scene):
    """URDF 전체 link 모델 (mesh 찾기 · keep-out 용, `wrist_body_capsules` 의 robot_model)."""
    return _build(scene, "all")


@pytest.fixture(scope="module")
def wrist(scene):
    """`--links gripper_wrist --finger-cover both` 기본 (A14b: arm_6 inset 0.5 mm). CPU ~3–4 분 (프로세스 캐시 공유)."""
    return _build(scene, "gripper_wrist", finger_cover=FC, wrist_cover={"hands": "both"})


# =============================================================================================
# 1–2. 기본값 · A14 형상
# =============================================================================================


def test_inset_default_is_arm6_only():
    from AG3S.ag3s.experiments.sources import gripper_cover as gc

    assert gc.WRIST_COVER_INSET == {"link_left_arm_6": 0.0005, "link_right_arm_6": 0.0005}
    assert not set(gc.WRIST_COVER_INSET) & {"FT_sensor_L", "FT_sensor_R", "link_left_arm_5", "link_right_arm_5"}
    import inspect

    assert inspect.signature(gc.finger_body_spheres).parameters["inset"].default == 0.0
    assert inspect.signature(gc.wrist_body_capsules).parameters["inset"].default is None


def test_ft_and_arm5_are_the_a14_spheres_and_arm6_grew(wrist):
    for link, (n, h) in A14_FINGERPRINT.items():
        got = _fingerprint(*wrist.local_spheres(link))
        if link.endswith("arm_6"):
            assert got[0] > n, (link, got, n)                 # inset 으로 늘었다
        else:
            assert got == (n, h), link                        # FT · arm_5 는 A14 그대로
    rep = wrist.wrist_cover_report
    assert rep["options"] == {"hands": "both", "overflow": 0.005, "keep_out_hand": True,
                              "inset": {"link_left_arm_6": 0.0005, "link_right_arm_6": 0.0005}}
    for link, r in rep["links"].items():
        assert r["inset_mm"] == (0.5 if link.endswith("arm_6") else 0.0), link


def test_empty_inset_is_the_a14_arm6(mj, full):
    from AG3S.ag3s.experiments.sources import gripper_cover as gc

    _, model = mj
    for link in ("link_left_arm_6", "link_right_arm_6"):
        caps, rep = gc.wrist_body_capsules(model, full, [link], inset={})
        C = np.array([c.origin[:3, 3] for c in caps])
        R = np.array([c.radius for c in caps])
        assert _fingerprint(C, R) == A14_FINGERPRINT[link], link
        assert "n_short_of_inset" not in rep[link] and rep[link]["inset_mm"] == 0.0


# =============================================================================================
# 3. inset 형상
# =============================================================================================


def test_inset_cover_holds_every_target_inside_by_the_inset(mj, full, wrist):
    """서버 모델 (`wrist`) 의 arm_6 구: 고른 표본 (1 mm) 전부가 구 표면에서 0.5 mm 넘게 안쪽, 넘침 · keep-out 그대로."""
    from AG3S.ag3s.experiments.sources import gripper_cover as gc

    _, model = mj
    for link in ("link_left_arm_6", "link_right_arm_6"):
        surf = gc.wrist_surface(model, full, link)
        keep = gc.hand_keep_out(model, full, link)
        C, R = wrist.local_spheres(link)
        rep = wrist.wrist_cover_report["links"][link]
        target = surf.points[gc.finger_body_targets(surf) & ~surf.hidden]
        g = gc._gap(target, C, R)
        assert rep["n_short_of_inset"] == 0 and rep["n_uncovered"] == 0, rep
        assert g.max() <= -0.0005 + 1e-9, link                  # 고른 표본 전부 0.5 mm 넘게 안쪽
        assert rep["inset_mm"] == 0.5
        # 반지름 공식 그대로: 넘침 ≤ overflow + 표본 오차, keep-out 상한 (손 껍질 밖)
        assert rep["overflow_sampled_mm"] <= 5.5, rep["overflow_sampled_mm"]
        assert np.all(gc.keep_out_clearance(C, keep) >= R - 1e-9), link


# =============================================================================================
# 4. 거절
# =============================================================================================


def test_bad_inset_is_rejected(mj, full, scene):
    from AG3S.ag3s.experiments.sources import gripper_cover as gc

    _, model = mj
    surf = gc.wrist_surface(model, full, "link_left_arm_6")
    for bad in (-0.001, float("nan")):
        with pytest.raises(ValueError, match="inset"):
            gc.finger_body_spheres(surf, inset=bad)
    with pytest.raises(ValueError, match="inset"):
        gc.finger_body_spheres(surf, inset=0.0005, gap=0.001)
    with pytest.raises(ValueError, match="inset"):
        gc.finger_body_spheres(surf, inset=0.0005, solver="ilp")
    with pytest.raises(ValueError, match="inset"):
        _build(scene, "gripper_wrist", wrist_cover={"inset": {"ee_left": 0.0005}})
    with pytest.raises(ValueError, match="inset"):
        _build(scene, "gripper_wrist", wrist_cover={"inset": {"link_left_arm_6": -1e-4}})


# =============================================================================================
# 5. 실험 자세 점검
# =============================================================================================


def test_point_triangle_distance_is_exact():
    from scipy.spatial import cKDTree

    from AG3S.ag3s.experiments.reports.wrist_pose_check import point_triangle_distance

    rng = np.random.default_rng(0)
    tri = rng.normal(size=(4, 3, 3))
    P = rng.normal(size=(200, 3)) * 2.0
    u = rng.uniform(size=(100_000, 2))
    flip = u.sum(1) > 1
    u[flip] = 1 - u[flip]
    brute = np.full(P.shape[0], np.inf)
    for t in tri:
        S = t[0] + u[:, :1] * (t[1] - t[0]) + u[:, 1:] * (t[2] - t[0])
        brute = np.minimum(brute, cKDTree(S).query(P)[0])
    d = point_triangle_distance(P, tri)
    assert np.all(d <= brute + 1e-12)                       # 정확한 값은 표본 거리보다 크지 않다
    assert np.all(brute - d <= 0.02)                        # 표본 간격 안에서 같다
    # 꼭짓점 · 변 · 면 위의 점은 0
    on = np.vstack([tri[0][0], 0.5 * (tri[0][0] + tri[0][1]), tri[0].mean(0)])
    assert np.allclose(point_triangle_distance(on, tri[:1]), 0.0, atol=1e-12)


def test_convex_piece_signed_distance():
    from AG3S.ag3s.experiments.reports.wrist_pose_check import ConvexPiece, pieces_signed_distance

    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    p = ConvexPiece(cube)
    P = np.array([[0, 0, 0], [0.5, 0, 0], [2, 0, 0], [2, 2, 0], [3, 3, 3]], float)
    d, ex = p.signed_distance(P, exact_within=10.0)
    assert np.allclose(d, [-1.0, -0.5, 1.0, np.sqrt(2.0), np.sqrt(12.0)])
    assert ex.all()
    # 정확한 범위 밖은 하한 (≤ 정확한 거리)
    d2, ex2 = p.signed_distance(P, exact_within=0.5)
    assert np.all(d2 <= d + 1e-12) and not ex2[2:].any()
    T = np.eye(4)
    T[:3, 3] = [10.0, 0, 0]
    d3, _ = pieces_signed_distance(P + [10.0, 0, 0], [(p, T)], exact_within=10.0)
    assert np.allclose(d3, d)


def test_pose_thresholds_are_fixed():
    from AG3S.ag3s.experiments.reports import wrist_pose_check as W

    assert W.POSE_THRESHOLDS == {"a_mjcf_hand_penetration_mm": 0.001,
                                 "b_auth_overlap_excess_over_auth_protrusion_mm": 0.01,
                                 "c_false_constraint_points": 0, "frame_diff_mm": 0.01}
    assert W.HAND_REGION == 0.005 and W.TARGET_SPACING == 0.001
    assert W.CL3_ARMS == ("C1a", "C1aF1", "C1aF1F2")


def test_finger_param_map_is_positional(scene):
    """URDF 손가락 parameter ← MuJoCo joint 는 위치로 정한다: 왼손은 이름이 뒤바뀐다 (T43 OC), 오른손은 같다."""
    from AG3S.ag3s.experiments.reports import wrist_pose_check as W

    import mujoco

    m = _build(scene, "gripper", finger_cover=FC)
    mj = scene.model
    qadr = {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, i): int(mj.jnt_qposadr[i]) for i in range(mj.njnt)}
    assert W.finger_param_map(mj, m, qadr) == {
        "gripper_finger_l1": ("gripper_finger_l2", -1.0), "gripper_finger_l2": ("gripper_finger_l1", -1.0),
        "gripper_finger_r1": ("gripper_finger_r1", 1.0), "gripper_finger_r2": ("gripper_finger_r2", 1.0)}


def test_pose_check_on_cl3_planning_rows(scene, wrist):
    """CL3 C1aF1 planning 행 (처음 2 개 + 왼손가락 두 값이 0.209 mm 다른 행): (a) ≤ 허용오차 · frame ≤ 0.01 mm.
    보정 전 (`--links arms`) arm_5 는 손 안 (> 10 mm, T46d 재현)."""
    import mujoco

    from AG3S.ag3s.experiments.reports import wrist_pose_check as W

    name = "C1aF1_ep12_s122"
    ana = CL3 / "analysis" / f"{name}.json"
    if not ana.exists():
        pytest.skip("T46 CL3 기록이 없다")
    a = json.loads(ana.read_text())
    rows = [json.loads(l) for l in open(pathlib.Path(a["run_dir"]) / "frames.jsonl")]
    plan = [r for r in rows if r["kind"] == "planning" and r["seq"] in (1, 2, 16)]
    assert len(plan) == 3
    before = _build(scene, "arms", finger_cover=FC)
    ctx = W.Context(scene, {"after": wrist, "before_arms": before})
    data = mujoco.MjData(ctx.mj)
    for r in plan:
        res = W.pose_metrics(ctx, data, np.asarray(r["qpos"], np.float64), a["used_arm"], a["target_fruit"])
        aft = res["models"]["after"]
        assert aft["a_sweep_pen_max_mm"] <= W.POSE_THRESHOLDS["a_mjcf_hand_penetration_mm"]
        assert aft["a_current_pen_max_mm"] <= aft["a_sweep_pen_max_mm"] + 1e-9   # sweep ⊇ current
        assert aft["frame_diff_mm"] <= W.POSE_THRESHOLDS["frame_diff_mm"]
        for k in ("b_auth_overlap_max_mm", "c_n_points_inside_active_wrist", "d_active_wrist_to_other_hand_mm",
                  "d_other_wrist_to_active_hand_mm"):
            assert k in aft, k
        assert res["models"]["before_arms"]["a_current_pen_max_mm"] > 10.0
