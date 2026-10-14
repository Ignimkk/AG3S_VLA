"""T46f A14 — 손목 (비권한) 충돌 형상 보정 (`--links gripper_wrist`) 과 예외 없음 정책 (`--target-field-policy none`).

T46f §4 (사용자 판정 2026-10-13): DX-04 link-specific exemption 비교 전에

1. FT sensor coverage 구 (비권한, role `wrist_cover`) 를 MJCF `FT_SENSOR_*` collision mesh 에서, URDF link `FT_sensor_*` 에.
2. `link_*_arm_5` 의 URDF capsule (r 75 mm — 손을 삼킨다, T46d) 을 MJCF collision mesh 의 몸통 구로 (arm_6 도).
3. coverage audit (형상 누락 · 넘침 · 손 영역 침범).
4. 새 link 집합 `gripper_wrist` — 모든 조건이 같은 형상.
5. `--target-field-policy none` — 모든 link 이 target 을 일반 obstacle 로 (마진 완화 · target-free · 허용 접촉 없음).

이 파일이 고정하는 것:

| # | 성질 |
|---|---|
| 1 | 기본값 그대로: `gripper` · `relax` 가 첫 항목, `wrist_cover=None/False/{}` 는 같은 모델, CL3 기록 (C1aF1) 의 구 이름 · 반지름 비트 동일, CLI 기본은 아무것도 안 넘긴다 |
| 2 | `gripper_wrist` 모델: link 집합 · role · arm_5 URDF capsule 없음 · gripper 구는 gripper 모델과 같다 · 손목 link 이 없으면 ValueError |
| 3 | 비권한 구가 권한 손의 MJCF collision 껍질 안에 들어가지 않는다 (link frame 정확 · world frame MuJoCo FK) — 보정 전 arm_5 는 들어간다 (T46d 재현) |
| 4 | audit 기준 (사전 고정 `wrist_cover_audit.THRESHOLDS`): 형상 누락 · 넘침 |
| 5 | FT sensor 는 어떤 정책에서도 권한을 받지 않는다 (`PROTECTED_LINKS`): 권한 집합 · 쥔 물체 권한 사슬 (arm_6 깊이 1 유지) |
| 6 | `none` 의미: 마진 전부 full · target-free 마스크 없음 · 비권한 행 = 전 행 · 기록 · verdict 권한 없음 · trajopt 접촉 행 없음 · 쥔 뒤는 그대로 |
| 7 | 정책과 무관하게 같은 형상 (`build_ag3s` 가 넘기는 모델 인자) · config / CLI 왕복 |
"""

from __future__ import annotations

import inspect
import json
import pathlib
import types

import numpy as np
import pytest

from common.workspace import shared_storage_root, workspace_path

MODEL = pathlib.Path(workspace_path("robots/rby1/description/models/rby1a/mujoco/model_transport.xml"))
SERVER = dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
FC = {"hands": "both"}
#: T46 CL3 (C1aF1) 서버 기록 — `--links gripper --finger-cover both` + SERVER (T46d 입력).
CL3_C1AF1 = (shared_storage_root() / "results/development/20261012T155130+0900_218229ac6464_5fb5ff542c33"
             / "verify/T46/CL3/runs/C1aF1/server_8291_20261012T085210")
BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


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
def gripper(scene):
    return _build(scene, finger_cover=FC)


@pytest.fixture(scope="module")
def wrist(scene):
    """`--links gripper_wrist --finger-cover both` 서버 모델 (손목 보정 형상, 기본 overflow 5 mm). CPU ~3 분."""
    return _build(scene, "gripper_wrist", finger_cover=FC, wrist_cover={"hands": "both"})


# =============================================================================================
# 1. 기본값 그대로
# =============================================================================================


def test_default_lists_keep_their_first_item():
    from AG3S.ag3s.config import (TARGET_FIELD_POLICIES, TARGET_FREE_POLICIES, AG3SConfig,
                                  ConstraintConfig)
    from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
    from AG3S.trajopt import serve_safe

    assert serve_safe.LINK_GROUPS[0] == serve_safe.DEFAULT_LINKS == "gripper"
    assert serve_safe.LINK_GROUPS[-1] == "gripper_wrist"
    assert TARGET_FIELD_POLICIES[0] == "relax" and TARGET_FIELD_POLICIES[-1] == "none"
    assert TARGET_FREE_POLICIES == ("exclude_authorized", "exclude_all")
    assert AG3SConfig().constraint.target_field_policy == "relax"
    for policy, needs in (("relax", False), ("none", False), ("exclude_authorized", True),
                          ("exclude_all", True)):
        assert ConstraintConfig(target_field_policy=policy).excludes_target_from_field is needs
    assert inspect.signature(build_constraint_robot_model).parameters["wrist_cover"].default is None
    assert inspect.signature(serve_safe.build_ag3s).parameters["wrist_cover"].default is None


def test_off_is_the_model_the_cl3_server_used(gripper):
    rec = sorted((CL3_C1AF1 / "server_constraints" / "run_0000").glob("chunk_*.npz"))
    if not rec:
        pytest.skip("T46 CL3 C1aF1 서버 기록이 없다")
    proc = (CL3_C1AF1 / "server_proc.txt").read_text()
    assert "--links gripper " in proc and "--finger-cover both" in proc
    z = np.load(rec[0], allow_pickle=True)
    names = json.loads(str(z["summary_json"]))["sphere_link_names"]
    assert list(gripper.sphere_link_names) == names
    assert np.array_equal(gripper.radii.astype(np.float32), z["sphere_radii"])
    assert not hasattr(gripper, "wrist_cover_report")


@pytest.mark.parametrize("off", [None, False, {}])
@pytest.mark.parametrize("links", ["gripper", "arms"])
def test_off_keyword_paths_are_the_same_model(scene, gripper, off, links):
    ref = gripper if links == "gripper" else _build(scene, "arms", finger_cover=FC)
    m = _build(scene, links, finger_cover=FC, wrist_cover=off)
    assert m.sphere_link_names == ref.sphere_link_names
    assert m.sphere_roles == ref.sphere_roles
    assert np.array_equal(m.radii, ref.radii)
    q = np.random.default_rng(3).uniform(-1.0, 1.0, m.nq)
    a, ra = ref.sphere_centers_numeric(q)
    b, rb = m.sphere_centers_numeric(q)
    assert np.array_equal(a, b) and np.array_equal(ra, rb)
    assert not hasattr(m, "wrist_cover_report")


def test_existing_link_groups_did_not_move():
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, GRIPPER_LINKS, GRIPPER_WRIST_LINKS, WRIST_LINKS)
    from AG3S.trajopt.serve_safe import constraint_link_filter

    assert constraint_link_filter("gripper") == GRIPPER_LINKS
    assert constraint_link_filter("arms") == ARM_LINKS
    assert constraint_link_filter("all") is None
    assert constraint_link_filter("gripper_wrist") == GRIPPER_WRIST_LINKS == GRIPPER_LINKS + WRIST_LINKS
    assert set(WRIST_LINKS) == {"link_left_arm_5", "link_left_arm_6", "FT_sensor_L",
                                "link_right_arm_5", "link_right_arm_6", "FT_sensor_R"}
    assert not {"FT_sensor_L", "FT_sensor_R"} & set(ARM_LINKS)


def test_cli_default_passes_nothing():
    from AG3S.trajopt import serve_safe

    p = serve_safe.build_parser()
    args = p.parse_args(BASE_ARGV)
    assert args.links == "gripper" and args.wrist_cover_overflow is None
    assert serve_safe.wrist_cover_options(args) is None
    for links in ("arms", "all"):
        assert serve_safe.wrist_cover_options(p.parse_args(BASE_ARGV + ["--links", links])) is None
    with pytest.raises(SystemExit):   # overflow 만 주면 아무 일도 안 한다 → 거절
        serve_safe.wrist_cover_options(p.parse_args(BASE_ARGV + ["--wrist-cover-overflow", "5"]))
    src = inspect.getsource(serve_safe.main)
    assert "wrist_cover=wrist_cover_options(args)" in src


# =============================================================================================
# 2. gripper_wrist 모델
# =============================================================================================


def test_gripper_wrist_model(gripper, wrist):
    from AG3S.ag3s.experiments.reports.grounding_report import GRIPPER_WRIST_LINKS, WRIST_LINKS
    from AG3S.ag3s.robot_models.urdf_sphere_chain import (
        EXACT_SPHERE_ROLES, WRIST_BODY_ROLE, WRIST_COVER_ROLE)

    names = np.asarray(wrist.sphere_link_names)
    roles = np.asarray(wrist.sphere_roles)
    assert set(names) == set(GRIPPER_WRIST_LINKS)
    assert {WRIST_BODY_ROLE, WRIST_COVER_ROLE} <= EXACT_SPHERE_ROLES
    for link in WRIST_LINKS:
        want = WRIST_COVER_ROLE if link.startswith("FT_sensor") else WRIST_BODY_ROLE
        assert set(roles[names == link]) == {want}, link        # URDF arm_5 capsule 구 · arm_6 사슬 없음
        assert (names == link).sum() >= 50, link
    # 손 (권한) 구는 gripper 모델과 같다 — 이름 순서 · 반지름 · link frame 중심
    hand = ~np.isin(names, WRIST_LINKS)
    assert list(names[hand]) == list(gripper.sphere_link_names)
    assert np.array_equal(wrist.radii[hand], gripper.radii)
    for link in set(gripper.sphere_link_names):
        ca, ra = gripper.local_spheres(link)
        cb, rb = wrist.local_spheres(link)
        assert np.array_equal(ca, cb) and np.array_equal(ra, rb)
    rep = wrist.wrist_cover_report
    # T46f A14b: arm_6 양손 inset 0.5 mm 가 기본이다 (`gripper_cover.WRIST_COVER_INSET`)
    assert rep["options"] == {"hands": "both", "overflow": 0.005, "keep_out_hand": True,
                              "inset": {"link_left_arm_6": 0.0005, "link_right_arm_6": 0.0005}}
    assert rep["n_spheres_after"] == wrist.n_spheres
    assert set(rep["links"]) == set(WRIST_LINKS)
    # 시작 로그용 보고 키
    for r in rep["links"].values():
        assert {"n_spheres", "radius_mm", "n_uncovered", "n_targets", "max_gap_mm",
                "overflow_sampled_mm", "mujoco_body"} <= set(r)
    assert rep["links"]["FT_sensor_L"]["mujoco_body"] == "FT_SENSOR_L"


def test_ft_sensor_hangs_off_the_urdf_link_that_carries_it(mj, gripper):
    """URDF `tool_<side>` (fixed, arm_6 아래 z −0.1087) = MJCF `FT_SENSOR_<S>` body pos → 변환이 단위 행렬."""
    from AG3S.ag3s.experiments.sources import gripper_cover as gc

    _, model = mj
    for link, body in gc.WRIST_MJCF_BODY.items():
        T = gc._mjcf_body_in_link(model, gripper, link, body)
        assert np.allclose(T, np.eye(4), atol=1e-12), link
        assert np.allclose(gc._urdf_fixed_transform(gripper, f"link_{'left' if link.endswith('L') else 'right'}_arm_6",
                                                    link)[:3, 3], [0, 0, -0.1087])


def test_wrist_cover_needs_a_wrist_link_in_the_filter(scene):
    with pytest.raises(ValueError, match="손목 link"):
        _build(scene, "gripper", finger_cover=FC, wrist_cover=True)
    with pytest.raises(ValueError, match="모르는 키"):
        _build(scene, "gripper_wrist", wrist_cover={"bogus": 1})
    with pytest.raises(ValueError, match="overflow"):
        _build(scene, "gripper_wrist", wrist_cover={"overflow": 0.0})


def test_cli_gripper_wrist_builds_the_wrist_cover():
    from AG3S.trajopt import serve_safe

    p = serve_safe.build_parser()
    args = p.parse_args(BASE_ARGV + ["--links", "gripper_wrist"])
    assert serve_safe.wrist_cover_options(args) == {"hands": "both"}
    args = p.parse_args(BASE_ARGV + ["--links", "gripper_wrist", "--wrist-cover-overflow", "10"])
    assert serve_safe.wrist_cover_options(args) == {"hands": "both", "overflow": pytest.approx(0.010)}
    for bad in ("0", "-1", "nan"):
        with pytest.raises(SystemExit):
            serve_safe.wrist_cover_options(
                p.parse_args(BASE_ARGV + ["--links", "gripper_wrist", "--wrist-cover-overflow", bad]))
    for extra in (["--no-safe"], ["--no-perception"]):
        p = serve_safe.build_parser()
        with pytest.raises(SystemExit):
            serve_safe.reject_bad_flag_combinations(
                p, p.parse_args(BASE_ARGV + extra + ["--links", "gripper_wrist"]))
    p = serve_safe.build_parser()
    serve_safe.reject_bad_flag_combinations(p, p.parse_args(BASE_ARGV + ["--links", "gripper_wrist"]))


# =============================================================================================
# 3. 비권한 구는 권한 손 형상 안에 들어가지 않는다
# =============================================================================================


def test_no_wrist_sphere_inside_the_hand_hulls_link_frame(mj, wrist):
    """link frame, 정확: FT · arm_6 는 손바닥 껍질 + 손가락 sweep 껍질, arm_5 는 arm_6 회전 (z) 의 모든 각에서."""
    from AG3S.ag3s.experiments.sources import gripper_cover as gc

    _, model = mj
    for side in ("left", "right"):
        pieces = gc.hand_pieces(model, wrist, side)
        palm = gc.WRIST_HAND_LINKS[side][0]
        for link in gc.WRIST_COVER_LINKS[side]:
            c, r = wrist.local_spheres(link)
            if link.endswith("arm_5"):
                arm6 = next(j for j in wrist.model.joints if j.parent == link)
                T6 = gc._urdf_fixed_transform(wrist, arm6.child, palm)
                for a in np.linspace(arm6.lower, arm6.upper, 37):
                    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
                    hulls = [gc._hull_eq((v @ T6[:3, :3].T + T6[:3, 3]) @ Rz.T) for v in pieces]
                    assert np.all(gc.keep_out_clearance(c, hulls) >= r - 1e-9), (link, a)
            else:
                T = gc._urdf_fixed_transform(wrist, link, palm)
                hulls = [gc._hull_eq(v @ T[:3, :3].T + T[:3, 3]) for v in pieces]
                assert np.all(gc.keep_out_clearance(c, hulls) >= r - 1e-9), link


def test_world_frame_mujoco_check_and_the_t46d_arm5_error(scene, wrist):
    """world frame (MuJoCo FK 의 MJCF 손 geom 대 URDF FK 의 구): 보정 후 침범 ≤ 1 µm (접함 + 부동소수),
    frame 차 ≤ 0.01 mm. 보정 전 (`--links arms`) arm_5 는 손바닥 · 손가락 안으로 들어간다 (T46d)."""
    from AG3S.ag3s.experiments.reports import wrist_cover_audit as A

    before = _build(scene, "arms", finger_cover=FC)
    out = A.hand_overlap(scene, {"before_arms": before, "after": wrist},
                         ("link_left_arm_5", "link_left_arm_6", "FT_sensor_L",
                          "link_right_arm_5", "link_right_arm_6", "FT_sensor_R"), n_poses=6, seed=1)
    assert out["frame_depth_diff_max_mm"] <= A.THRESHOLDS["frame_depth_diff_mm"]
    for link, rec in out["by_model"]["after"].items():
        assert rec["mjcf_pen_lb_max_mm"] <= A.THRESHOLDS["mjcf_hand_penetration_lb_mm"], link
        assert rec["mjcf_pen_sampled_max_mm"] <= A.THRESHOLDS["mjcf_hand_penetration_sampled_mm"], link
    for side in ("left", "right"):
        rec = out["by_model"]["before_arms"][f"link_{side}_arm_5"]
        assert rec["mjcf_pen_sampled_max_mm"] > 10.0                  # URDF capsule 구가 손 안
        assert rec["n_sphere_pose_pen_sampled"] > 0


# =============================================================================================
# 4. audit 기준 (형상 누락 · 넘침)
# =============================================================================================


def test_coverage_meets_the_fixed_thresholds(mj, wrist):
    """사전 고정 기준 (`THRESHOLDS`) 의 형상 누락 · 넘침을 그대로 건다.

    A14 에서는 `link_left_arm_6` 의 표본 2 개 (0.97 mm, FT 와의 틈) 가 최대 밖 거리 기준 0.5 mm (arm) 를 넘었다.
    A14b (사용자 판정 A2) 는 arm_6 양손에 inset 0.5 mm 를 넣었고, 기준은 바꾸지 않았다. 그래서 여기서도
    `reachable_max_outside_mm` 를 link 종류마다 그대로 건다. 0.5 mm 넘게 밖인 표본이 남는 link (FT) 는 그 표본이
    전부 강체 이웃 (손바닥 · arm_6) 에서 2 mm 안이다."""
    from AG3S.ag3s.experiments.reports import wrist_cover_audit as A
    from AG3S.ag3s.experiments.reports.grounding_report import WRIST_LINKS
    from AG3S.ag3s.experiments.sources.gripper_cover import COVER_MAX_GAP

    _, model = mj
    cov = A.coverage(model, {"after": wrist}, WRIST_LINKS, 0.0005)
    th = A.THRESHOLDS
    for link, rec in cov.items():
        kind = A._kind(link)
        a = rec["after"]
        assert a["reachable"]["frac_outside_gt_0p5mm"] <= th["reachable_frac_outside_gt_0p5mm"][kind]
        assert a["reachable"]["outside_max_mm"] <= th["reachable_max_outside_mm"][kind], (link, a["reachable"])
        assert a["reachable"]["outside_max_mm"] <= COVER_MAX_GAP * 1e3, (link, a["reachable"])
        assert a["vertices_reachable"]["outside_max_mm"] <= COVER_MAX_GAP * 1e3, link
        where = a.get("outside_gt_0p5mm_where")
        if where is not None:
            assert where["dist_to_rigid_neighbour_mm"][2] <= 2.0, (link, where)
        assert a["overflow"]["max_mm"] <= 5.0 + th["overflow_max_excess_mm"], (link, a["overflow"])
        # 보정 전 (`--links arms`) 과 비교하면: arm_5 · arm_6 겉면의 95 % 넘게가 밖이었다 (FT 는 구가 없었다)
        assert a["reachable"]["frac_outside"] < 0.02


# =============================================================================================
# 5. FT sensor 는 권한을 받지 않는다
# =============================================================================================


def test_ft_sensor_is_never_authorized():
    from AG3S.ag3s.config import ContactConfig
    from AG3S.ag3s.constraints.clearance import PROTECTED_LINKS, ClearancePolicy
    from AG3S.ag3s.types import ContactPolicyContext

    ctx = ContactPolicyContext.make("grasp", ["left", "right"])
    assert PROTECTED_LINKS == {"FT_sensor_L", "FT_sensor_R"}
    assert not ClearancePolicy().authorized_links(ctx) & PROTECTED_LINKS
    # 설정이 FT 를 넣어도 빠진다
    cfg = ContactConfig(contact_links={"left": ("ee_left", "FT_sensor_L"), "right": ("ee_right",)})
    assert ClearancePolicy(contact=cfg).authorized_links(ctx) == {"ee_left", "ee_right"}
    # 기본 집합 (FT 없음) 은 그대로
    assert ClearancePolicy().authorized_links(ctx) == {
        "ee_left", "ee_finger_l1", "ee_finger_l2", "ee_right", "ee_finger_r1", "ee_finger_r2"}


def test_held_permission_chain_skips_the_ft_sensor(wrist):
    from AG3S.ag3s.constraints.to_adapter import held_contact_depth, urdf_parent_of

    names = list(wrist.sphere_link_names)
    depth, parents = held_contact_depth(names, "ee_left", ("ee_left", "ee_finger_l1", "ee_finger_l2"),
                                        urdf_parent_of(wrist))
    d = dict(zip(names, depth))
    assert parents[:2] == ["link_left_arm_6", "link_left_arm_5"]
    assert d["FT_sensor_L"] == -1 and d["link_left_arm_6"] == 1 and d["link_left_arm_5"] == 2
    assert d["ee_left"] == 0 and d["ee_finger_l1"] == 0 and d["FT_sensor_R"] == -1


# =============================================================================================
# 6. `none` — 예외 없음
# =============================================================================================

NAMES = ["ee_left", "ee_finger_l1", "ee_finger_l2", "ee_right", "ee_finger_r1", "ee_finger_r2",
         "link_left_arm_5", "link_left_arm_6", "FT_sensor_L"]


def _adapter_builder(policy, mode="full"):
    from AG3S.ag3s.config import ConstraintConfig, ContactConfig
    from AG3S.ag3s.constraints.clearance import ClearancePolicy

    return types.SimpleNamespace(
        build=None, max_candidates=32, max_support_surfaces=2,
        constraint_config=ConstraintConfig(target_field_policy=policy), robot_model=None,
        n_robot_spheres=len(NAMES), sphere_link_names=list(NAMES),
        clearance_policy=ClearancePolicy(contact=ContactConfig(nonauthorized_target_margin=mode)))


def _target():
    from AG3S.ag3s.stages.geometry import fit_sphere
    from AG3S.ag3s.types import TargetGeometry

    pts = np.random.default_rng(0).normal([0.5, 0.0, 0.85], 0.01, (40, 3))
    return TargetGeometry(id=1, points=pts, point_indices=np.arange(40), centroid=pts.mean(0),
                          bounding_geometry=fit_sphere(pts), attention_score=0.9, confidence=0.9)


class _FreeField:
    """target-free 계층이 **있는** 필드 — `none` 은 그래도 읽지 않아야 한다."""
    has_target_free = True


def _cs(policy, mode="full", phase="grasp"):
    from AG3S.ag3s.constraints.to_adapter import build_constraint_set
    from AG3S.ag3s.types import ContactPolicyContext

    return build_constraint_set(
        _adapter_builder(policy, mode), candidates=[], support_surfaces=[], target=_target(),
        robot_state=np.zeros(2), phase=phase, contact_context=ContactPolicyContext.make(phase, ["left"]),
        build_spec=False, esdf=_FreeField())


@pytest.mark.parametrize("phase", ["approach", "pre_grasp", "grasp"])
def test_none_relaxes_no_margin_and_reads_no_target_free_layer(phase):
    from AG3S.ag3s.constraints.clearance import ClearancePolicy

    full = ClearancePolicy().full_margin("target")
    relax, none, lte = _cs("relax", phase=phase), _cs("none", phase=phase), _cs("exclude_authorized", phase=phase)
    assert np.all(none.manipulated_link_margin == full)                        # 전 행 full
    assert none.target_field_exclude is None and none.target_field_policy == "none"
    rec = none.metrics["target_permission"]
    assert rec["policy"] == "none" and rec["authorized_links"] == [] and rec["n_rows_excluded"] == 0
    # 다른 정책은 예전 그대로: 권한 손 (왼손) 은 완화, LTE 는 권한 행만 target-free
    authorized = {"ee_left", "ee_finger_l1", "ee_finger_l2"}
    relaxed = np.asarray([n in authorized for n in NAMES])
    assert np.all(relax.manipulated_link_margin[~relaxed] == full)
    assert np.all(relax.manipulated_link_margin[relaxed] < full)        # 세 phase 모두 권한 손은 완화
    assert np.array_equal(lte.target_field_exclude, relaxed)
    assert np.array_equal(lte.manipulated_link_margin, relax.manipulated_link_margin)


def test_none_with_obstacle_margin_marks_every_row_nonauthorized():
    none = _cs("none", "obstacle")
    assert none.nonauthorized_target_rows.tolist() == [True] * len(NAMES)
    lte = _cs("exclude_authorized", "obstacle")
    assert {n for n, r in zip(NAMES, lte.nonauthorized_target_rows) if r} == {
        "ee_right", "ee_finger_r1", "ee_finger_r2", "link_left_arm_5", "link_left_arm_6", "FT_sensor_L"}


def test_none_leaves_the_held_object_rules_alone():
    from AG3S.ag3s.constraints.clearance import target_permission_context
    from AG3S.ag3s.types import ContactPolicyContext

    ctx = ContactPolicyContext.make("grasp", ["left"])
    assert target_permission_context(ctx, "none").active_manipulators == frozenset()
    assert target_permission_context(ctx, "none", attached=object()) is ctx
    for policy in ("relax", "exclude_authorized", "exclude_all"):
        assert target_permission_context(ctx, policy) is ctx
    assert target_permission_context(ctx, "none").phase == ctx.phase


def test_builder_margin_block_uses_the_same_context():
    from AG3S.ag3s.constraints.constraint_builder import ConstraintBuilder

    src = inspect.getsource(ConstraintBuilder.parameter_values)
    assert "target_permission_context(" in src and "context=margin_ctx" in src


def test_none_in_trajopt_names_no_contact_row_and_the_verdict_no_link():
    from AG3S.trajopt.linearize import target_contact_rows_of
    from AG3S.trajopt.safe_policy import SafePolicy
    from AG3S.ag3s.constraints.clearance import ClearancePolicy

    none = _cs("none")
    rows = target_contact_rows_of(none.target_field_exclude, None, none.manipulated_link_margin,
                                  "none", manipulated=True)
    assert rows is not None and not rows.any()
    relax = _cs("relax")
    rows = target_contact_rows_of(relax.target_field_exclude, None, relax.manipulated_link_margin,
                                  "relax", manipulated=True)
    assert {n for n, r in zip(NAMES, rows) if r} == {"ee_left", "ee_finger_l1", "ee_finger_l2"}
    me = types.SimpleNamespace(ag3s=types.SimpleNamespace(builder=types.SimpleNamespace(
        clearance_policy=ClearancePolicy())))
    assert SafePolicy._authorized_links(me, none) == frozenset()
    assert SafePolicy._authorized_links(me, relax) == {"ee_left", "ee_finger_l1", "ee_finger_l2"}
    held = types.SimpleNamespace(contact_context=none.contact_context, target_field_policy="none",
                                 attached=object())
    assert SafePolicy._authorized_links(me, held) == {"ee_left", "ee_finger_l1", "ee_finger_l2"}


# =============================================================================================
# 7. 정책과 무관하게 같은 형상 · config / CLI 왕복
# =============================================================================================


def test_config_round_trip_none():
    from AG3S.ag3s.config import AG3SConfig, AG3SConfigError

    cfg = AG3SConfig.from_dict({"constraint": {"target_field_policy": "none"}})
    assert cfg.constraint.target_field_policy == "none"
    assert not cfg.constraint.excludes_target_from_field
    assert AG3SConfig.from_dict(cfg.to_dict()).constraint.target_field_policy == "none"
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"constraint": {"target_field_policy": "nothing"}})


def test_cli_none():
    from AG3S.trajopt import serve_safe

    p = serve_safe.build_parser()
    args = p.parse_args(BASE_ARGV + ["--target-field-policy", "none"])
    assert serve_safe.resolve_target_field_policy(args.target_field_policy) == "none"
    # none 은 target-free 계층을 만들지 않으므로 curobo · fine voxel 을 요구하지 않는다
    serve_safe.reject_bad_flag_combinations(p, args)
    # target-free 계층에서만 도는 flag 는 none 과 거절 (relax 와 같은 규율)
    for extra in (["--target-exclusion", "ball-owned"], ["--target-free-coarse"]):
        p = serve_safe.build_parser()
        with pytest.raises(SystemExit):
            serve_safe.reject_bad_flag_combinations(
                p, p.parse_args(BASE_ARGV + ["--target-field-policy", "none", "--esdf-backend", "curobo",
                                             "--fine-voxel", "0.005"] + extra))
    # 다른 정책의 요구는 그대로
    p = serve_safe.build_parser()
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(
            p, p.parse_args(BASE_ARGV + ["--target-field-policy", "exclude-authorized"]))


class _Stop(Exception):
    pass


@pytest.mark.parametrize("links", ["gripper", "gripper_wrist"])
def test_same_geometry_for_every_policy(monkeypatch, links):
    """`build_ag3s` 가 제약 모델 builder 에 넘기는 인자는 정책과 무관하다 (세 조건 · CL4 가 같은 형상)."""
    from AG3S.ag3s.experiments.reports import grounding_report as G
    from AG3S.ag3s.experiments.sources import mujoco_source
    from AG3S.trajopt import serve_safe

    seen = []

    def fake_build(scene, link_filter=None, **kw):
        seen.append((link_filter, kw))
        raise _Stop()

    monkeypatch.setattr(G, "build_constraint_robot_model", fake_build)
    monkeypatch.setattr(G, "build_robot_model", lambda scene: types.SimpleNamespace(n_spheres=0))
    monkeypatch.setattr(mujoco_source.TransportScene, "attach", classmethod(lambda cls, m, d: None))
    p = serve_safe.build_parser()
    for policy in ("relax", "none", "exclude-authorized", "exclude-all"):
        args = p.parse_args(BASE_ARGV + ["--links", links, "--target-field-policy", policy,
                                         "--finger-cover", "both"])
        with pytest.raises(_Stop):
            serve_safe.build_ag3s(str(MODEL), voxel=0.02, range_max=2.0, links=args.links,
                                  constraint_sphere_options=dict(SERVER),
                                  finger_cover=serve_safe.finger_cover_options(args),
                                  wrist_cover=serve_safe.wrist_cover_options(args),
                                  target_field_policy=args.target_field_policy)
    assert len(seen) == 4 and all(s == seen[0] for s in seen)
    lf, kw = seen[0]
    if links == "gripper_wrist":
        assert kw["wrist_cover"] == {"hands": "both"}
    else:
        assert "wrist_cover" not in kw
    # 프로그램 호출이 wrist_cover 를 빼먹어도 gripper_wrist 는 보정 형상으로 짓는다
    seen.clear()
    if links == "gripper_wrist":
        with pytest.raises(_Stop):
            serve_safe.build_ag3s(str(MODEL), voxel=0.02, range_max=2.0, links="gripper_wrist")
        assert seen[0][1]["wrist_cover"] == {"hands": "both"}
