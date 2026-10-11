"""T19 — self-filter 가 조작 대상을 지우지 않는다 (link 별 inflation + 확정 target guard).

이 파일이 지키는 것:

* **기본값이면 지금과 글자 그대로 같다.** T19 이전의 필터 루프를 테스트가 따로 다시 적어 맞춰 본다 —
  코드가 스스로를 채점하지 않게.
* link 별 inflation: 집합(`arms` · `gripper`)을 펼치고, 우선순위가 dict 순서가 아니라 구체성이며,
  모르는 이름은 **오류**다 (조용히 전역값으로 물러나지 않는다).
* guard 는 **예외가 아니라 inflation 0** 이다 — 보호 구 안에서도 로봇 sphere 내부 점은 지워진다
  (지침 §3.3 의 경고).
* cloud 필터와 depth robot mask 가 **같은** inflation 객체와 **같은** guard centroid 를 쓴다.
* 실제 유효 설정이 stats 와 시작 로그에 남는다. YAML 이 서버 경로에 닿지 않는다는 문장도.
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import logging
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, PointCloudConfig
from AG3S.ag3s.runtime.pipeline import AG3S
from AG3S.ag3s.stages import robot_filter
from AG3S.ag3s.stages.reconstruction import backproject
from AG3S.ag3s.stages.robot_filter import (
    filter_robot_points,
    inflation_table,
    resolve_self_filter_inflation,
    robot_sphere_mask,
)
from AG3S.ag3s.types import PointCloud
from tests.ag3s.fixtures import (
    DEFAULT_ARM_Q,
    FINGERTIP_LINK,
    LABEL_ROBOT,
    PlanarTwoLinkArm,
    make_scene,
    make_three_camera_scene,
)

MODEL = pathlib.Path(
    _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
)


def _pre_t19_mask(points, centers, radii, inflation):
    """T19 이전 `robot_sphere_mask` 의 본문을 **테스트가 독립적으로** 다시 적은 것."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, np.float64).reshape(-1, 3)
    tree = cKDTree(points)
    inside = np.zeros(points.shape[0], bool)
    for centre, radius in zip(np.asarray(centers, np.float64), np.asarray(radii) + inflation):
        hits = tree.query_ball_point(centre, float(radius), workers=-1)
        if hits:
            inside[hits] = True
    return inside


@pytest.fixture(scope="module")
def arm():
    return PlanarTwoLinkArm()


@pytest.fixture(scope="module")
def scene(arm):
    return make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)


def _cfg(**pointcloud):
    return AG3SConfig.from_dict({"pointcloud": pointcloud})


# ------------------------------------------------------------------ 기본값 = 예전 그대로


def test_the_default_config_has_no_override_and_no_guard():
    cfg = PointCloudConfig()
    assert cfg.self_filter_inflation_by_link == {}
    assert cfg.self_filter_target_guard_radius == 0.0


def test_default_resolution_is_the_old_scalar_itself(arm):
    """재정의가 없으면 배열이 아니라 **그 float** 이 나간다 — 산술까지 예전과 같다."""
    cfg = PointCloudConfig()
    out = resolve_self_filter_inflation(arm.sphere_link_names, cfg)
    assert isinstance(out, float) and out == cfg.self_filter_inflation
    # 이름이 없는 모델이어도 기본 경로는 이름을 묻지 않는다.
    assert resolve_self_filter_inflation(None, cfg) == cfg.self_filter_inflation


def test_default_filter_output_equals_the_pre_t19_loop(arm, scene):
    cloud = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam)
    centres, radii = arm.sphere_centers_numeric(DEFAULT_ARM_Q)
    cfg = PointCloudConfig()
    expected = _pre_t19_mask(cloud.points, centres, radii, cfg.self_filter_inflation)

    out, stats = filter_robot_points(cloud, arm, DEFAULT_ARM_Q, cfg)
    assert stats["n_removed"] == int(expected.sum())
    np.testing.assert_array_equal(out.points, cloud.points[~expected])
    assert stats["guard_enabled"] is False and stats["n_guard_protected"] == 0

    # guard 반경이 0 이면 centroid 를 줘도 아무 일도 없다.
    out2, stats2 = filter_robot_points(cloud, arm, DEFAULT_ARM_Q, cfg,
                                       guard_centre=np.array([0.5, -0.3, 1.0]))
    np.testing.assert_array_equal(out2.points, out.points)
    assert stats2["guard_active"] is False


def test_default_pipeline_matches_a_pipeline_that_never_heard_of_t19(arm, scene):
    """AG3S 가 구한 inflation 객체가 예전 전역값과 같고, cloud 가 같은 점을 지운다."""
    ag3s = AG3S(AG3SConfig(), robot_model=arm)
    assert ag3s._self_filter_inflation == PointCloudConfig().self_filter_inflation
    _, debug = ag3s.process_debug(
        attention_map=scene.attention_grid, depth=scene.depth,
        camera_intrinsics=scene.camera_intrinsics, T_base_cam=scene.T_base_cam,
        robot_state=DEFAULT_ARM_Q, phase="approach", image_hw=scene.hw)
    raw = debug["raw_cloud"]
    centres, radii = arm.sphere_centers_numeric(DEFAULT_ARM_Q)
    expected = _pre_t19_mask(raw.points, centres, radii, PointCloudConfig().self_filter_inflation)
    np.testing.assert_array_equal(debug["filtered_cloud"].points, raw.points[~expected])


# ------------------------------------------------------------------ link 별 inflation


RBY1_LIKE = (
    ("base",) * 3
    + tuple(f"link_left_arm_{i}" for i in range(7))
    + ("ee_left", "ee_finger_l1", "ee_finger_l1", "ee_finger_l2")
    + tuple(f"link_right_arm_{i}" for i in range(7))
    + ("ee_right", "ee_finger_r1", "ee_finger_r2")
)


def test_groups_expand_and_the_more_specific_key_wins_whatever_the_dict_order():
    # T26: the default global is 0.0 now; a non-zero global keeps "unlisted → global" visible.
    a = PointCloudConfig(self_filter_inflation=0.05, self_filter_inflation_by_link={
        "ee_finger_l1": 0.0, "gripper": 0.01, "arms": 0.03})
    b = PointCloudConfig(self_filter_inflation=0.05, self_filter_inflation_by_link={
        "arms": 0.03, "gripper": 0.01, "ee_finger_l1": 0.0})
    for cfg in (a, b):
        infl = resolve_self_filter_inflation(RBY1_LIKE, cfg)
        table = inflation_table(RBY1_LIKE, infl)
        assert table["base"] == pytest.approx(0.05)  # 적지 않은 link 는 전역값
        assert table["link_left_arm_3"] == pytest.approx(0.03)
        assert table["ee_right"] == pytest.approx(0.01)  # gripper ⊂ arms, gripper 가 이긴다
        assert table["ee_finger_r2"] == pytest.approx(0.01)
        assert table["ee_finger_l1"] == 0.0  # link 이름이 집합을 이긴다
    np.testing.assert_array_equal(resolve_self_filter_inflation(RBY1_LIKE, a),
                                  resolve_self_filter_inflation(RBY1_LIKE, b))


def test_the_array_is_per_sphere_in_sphere_order():
    cfg = PointCloudConfig(self_filter_inflation=0.05,  # T26: explicit (default is 0.0)
                           self_filter_inflation_by_link={"ee_finger_l1": 0.004})
    infl = resolve_self_filter_inflation(RBY1_LIKE, cfg)
    assert infl.shape == (len(RBY1_LIKE),)
    for name, value in zip(RBY1_LIKE, infl):
        assert value == (0.004 if name == "ee_finger_l1" else 0.05)


def test_an_unknown_link_is_an_error_not_a_silent_fallback(arm):
    cfg = PointCloudConfig(self_filter_inflation_by_link={"ee_fingr_l1": 0.0})
    with pytest.raises(AG3SConfigError, match="ee_fingr_l1"):
        resolve_self_filter_inflation(RBY1_LIKE, cfg)
    # 파이프라인에서는 **생성할 때** 죽는다 — 첫 프레임이 아니라.
    with pytest.raises(AG3SConfigError, match="does not have"):
        AG3S(_cfg(self_filter_inflation_by_link={"link_9": 0.0}), robot_model=arm)


def test_a_group_with_no_member_in_the_model_is_an_error(arm):
    cfg = PointCloudConfig(self_filter_inflation_by_link={"gripper": 0.0})
    with pytest.raises(AG3SConfigError, match="gripper"):
        resolve_self_filter_inflation(arm.sphere_link_names, cfg)


@pytest.mark.parametrize("bad", [
    {"all": 0.01},
    {"gripper": -0.01},
    {"gripper": float("nan")},
    {"gripper": "wide"},
    {"": 0.01},
])
def test_config_rejects_malformed_overrides(bad):
    with pytest.raises(AG3SConfigError):
        _cfg(self_filter_inflation_by_link=bad)


@pytest.mark.parametrize("radius", [-0.01, float("inf")])
def test_config_rejects_a_bad_guard_radius(radius):
    with pytest.raises(AG3SConfigError):
        _cfg(self_filter_target_guard_radius=radius)


def test_overrides_round_trip_through_to_dict():
    cfg = _cfg(self_filter_inflation_by_link={"gripper": 0.01},
               self_filter_target_guard_radius=0.06)
    again = AG3SConfig.from_dict(cfg.to_dict())
    assert again.pointcloud.self_filter_inflation_by_link == {"gripper": 0.01}
    assert again.pointcloud.self_filter_target_guard_radius == 0.06


def test_a_per_link_margin_changes_only_that_links_spheres(arm, scene):
    """전역값을 두고 fingertip 만 0 으로 — fingertip 쪽 점만 새로 남는다."""
    cloud = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam)
    centres, radii = arm.sphere_centers_numeric(DEFAULT_ARM_Q)
    names = np.asarray(arm.sphere_link_names)
    cfg = PointCloudConfig(self_filter_inflation_by_link={FINGERTIP_LINK: 0.0})
    infl = resolve_self_filter_inflation(arm.sphere_link_names, cfg)
    assert infl[names == FINGERTIP_LINK].tolist() == [0.0]
    assert np.all(infl[names != FINGERTIP_LINK] == cfg.self_filter_inflation)

    got = robot_sphere_mask(cloud.points, centres, radii, infl)
    old = _pre_t19_mask(cloud.points, centres, radii, cfg.self_filter_inflation)
    assert not (got & ~old).any(), "a smaller margin cannot remove a point the old one kept"
    freed = old & ~got
    tip = centres[names == FINGERTIP_LINK][0]
    others = centres[names != FINGERTIP_LINK]
    d_tip = np.linalg.norm(cloud.points[freed] - tip, axis=1)
    d_others = np.linalg.norm(cloud.points[freed][:, None] - others[None], axis=2)
    # 새로 남은 점은 fingertip 의 여유 띠 안에 있고, 다른 구의 여유 띠 밖에 있다.
    assert np.all(d_tip <= radii[0] + cfg.self_filter_inflation + 1e-9)
    assert np.all(d_others > radii[0] + cfg.self_filter_inflation - 1e-9)


def test_the_inflation_array_must_match_the_sphere_count():
    with pytest.raises(ValueError, match="spheres"):
        robot_sphere_mask(np.zeros((1, 3)), np.zeros((2, 3)), np.ones(2), np.zeros(3))


# ------------------------------------------------------------------ target guard = inflation 0


CENTRE = np.array([[0.0, 0.0, 0.0]])
RADIUS = np.array([0.10])
INFLATION = 0.05


def test_the_guard_drops_the_margin_but_not_the_sphere():
    """지침 §3.3: 예외가 아니라 inflation 0 이다. 보호 구 안의 로봇 내부 점은 여전히 지워진다."""
    guard_centre = np.array([0.15, 0.0, 0.0])  # 사과가 손 옆 (구 표면에서 50 mm)
    guard_radius = 0.08
    pts = np.array([
        [0.05, 0.0, 0.0],   # 로봇 구 내부 + 보호 구 안 → 지운다
        [0.099, 0.0, 0.0],  # 로봇 표면 바로 안 + 보호 구 안 → 지운다
        [0.12, 0.0, 0.0],   # 여유 띠 + 보호 구 안 → 남긴다 (사과 표면)
        [-0.12, 0.0, 0.0],  # 여유 띠 + 보호 구 밖 → 예전처럼 지운다
        [0.30, 0.0, 0.0],   # 어느 구에도 없다 → 남긴다
    ])
    inside, protected = robot_sphere_mask(
        pts, CENTRE, RADIUS, INFLATION, guard_centre=guard_centre, guard_radius=guard_radius,
        return_protected=True)
    assert inside.tolist() == [True, True, False, True, False]
    assert protected.tolist() == [False, False, True, False, False]


def test_a_guard_covering_the_whole_robot_still_erases_every_interior_point():
    """보호 구가 로봇 구를 통째로 덮어도 그 **내부** 는 하나도 남지 않는다."""
    rng = np.random.default_rng(0)
    direction = rng.normal(size=(2000, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    interior = direction * (RADIUS[0] * rng.uniform(0.0, 0.999, size=(2000, 1)))
    shell = direction * (RADIUS[0] + INFLATION * rng.uniform(0.01, 0.99, size=(2000, 1)))
    inside, protected = robot_sphere_mask(
        np.vstack([interior, shell]), CENTRE, RADIUS, INFLATION,
        guard_centre=np.zeros(3), guard_radius=1.0, return_protected=True)
    assert inside[:2000].all(), "the guard must never leave robot interior points behind"
    assert not inside[2000:].any() and protected[2000:].all()


def test_the_guard_overlapping_two_spheres_keeps_a_point_only_if_outside_both_bare_radii():
    centres = np.array([[0.0, 0.0, 0.0], [0.18, 0.0, 0.0]])
    radii = np.array([0.10, 0.10])
    pts = np.array([[0.09, 0.0, 0.0], [0.0, 0.12, 0.0]])  # 첫째: 둘째 구의 bare 안
    inside, protected = robot_sphere_mask(pts, centres, radii, INFLATION,
                                          guard_centre=np.zeros(3), guard_radius=1.0,
                                          return_protected=True)
    assert inside.tolist() == [True, False] and protected.tolist() == [False, True]


def test_on_the_rendered_scene_the_guard_leaks_only_exact_surface_points(arm, scene):
    """렌더한 로봇 표면에서 guard(inflation 0) 가 무엇을 남기는지 **그대로 박아 둔다.**

    이 fixture 는 로봇을 구 그대로 렌더하므로 로봇 픽셀은 구 표면 **바로 위** (±1e-11 m) 에 있다.
    inflation 0 에서 그 점들은 부동소수 잡음만으로 반쯤 남는다 — 보호 구가 팔 전체를 덮으면
    9,714 px 중 6,355 px (2026-09-28 실측). 실제 depth 잡음은 mm 단위이므로 **보호 구 안의 로봇
    표면은 남을 수 있다** — 지침 §3.3 이 따로 검증하라고 한 바로 그 방향이다. 이 테스트는 두 가지를
    지킨다: (1) 남는 로봇 점은 전부 표면 점이고 내부 점은 하나도 없다, (2) guard 밖은 예전 필터
    그대로다.
    """
    cloud = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam)
    labels = scene.labels[cloud.uv[:, 1], cloud.uv[:, 0]]
    centres, radii = arm.sphere_centers_numeric(DEFAULT_ARM_Q)
    # T26: the default inflation is 0.0 now — the guard/no-guard contrast needs the old margin.
    cfg = PointCloudConfig(self_filter_inflation=0.05, self_filter_target_guard_radius=10.0)
    out, stats = filter_robot_points(cloud, arm, DEFAULT_ARM_Q, cfg, guard_centre=arm.base)
    assert stats["guard_active"] is True and stats["guard_radius"] == 10.0
    assert stats["guard_centroid"] == pytest.approx(arm.base.tolist())

    after = scene.labels[out.uv[:, 1], out.uv[:, 0]]
    survivors = out.points[after == LABEL_ROBOT]
    depth_into = (np.linalg.norm(survivors[:, None] - centres[None], axis=2) - radii[None]).min(1)
    assert np.all(np.abs(depth_into) < 1e-9), "only exact-surface robot points may survive"
    old_kept = len(cloud) - int(_pre_t19_mask(cloud.points, centres, radii,
                                              cfg.self_filter_inflation).sum())
    assert stats["n_guard_protected"] == stats["n_out"] - old_kept

    # guard 를 끄면 같은 장면에서 로봇은 예전처럼 지워진다.
    plain, _ = filter_robot_points(cloud, arm, DEFAULT_ARM_Q,
                                   PointCloudConfig(self_filter_inflation=0.05))
    plain_robot = int((scene.labels[plain.uv[:, 1], plain.uv[:, 0]] == LABEL_ROBOT).sum())
    assert plain_robot == 0


# ------------------------------------------------------------------ 두 호출부가 같은 것을 쓴다


def _spy(monkeypatch):
    calls = []
    real = robot_filter.robot_sphere_mask

    def spy(points, centers, radii, inflation=0.0, **kw):
        calls.append({"inflation": inflation, "guard_centre": kw.get("guard_centre"),
                      "guard_radius": kw.get("guard_radius")})
        return real(points, centers, radii, inflation, **kw)

    monkeypatch.setattr(robot_filter, "robot_sphere_mask", spy)
    return calls


def test_cloud_filter_and_depth_mask_get_the_same_inflation_and_guard(arm, scene, monkeypatch):
    ag3s = AG3S(_cfg(self_filter_inflation_by_link={FINGERTIP_LINK: 0.0},
                     self_filter_target_guard_radius=0.08), robot_model=arm)
    inputs = dict(attention_map=scene.attention_grid, depth=scene.depth,
                  camera_intrinsics=scene.camera_intrinsics, T_base_cam=scene.T_base_cam,
                  robot_state=DEFAULT_ARM_Q, phase="approach", image_hw=scene.hw)
    ag3s.process(**inputs)  # 프레임 1: target 을 확정한다
    held = ag3s._target_confirm.held_centroid
    if held is None:
        pytest.skip("grounding declined on this scene; nothing to guard")

    calls = _spy(monkeypatch)
    cs, debug = ag3s.process_debug(**inputs)  # 프레임 2: guard 가 켜진다
    mask = ag3s._robot_mask_for(scene.depth, scene.camera_intrinsics, scene.T_base_cam,
                                DEFAULT_ARM_Q)
    assert len(calls) == 2, "exactly one cloud-filter call and one depth-mask call"
    assert calls[0]["inflation"] is calls[1]["inflation"] is ag3s._self_filter_inflation
    np.testing.assert_array_equal(calls[0]["guard_centre"], held)
    np.testing.assert_array_equal(calls[1]["guard_centre"], held)
    assert calls[0]["guard_radius"] == calls[1]["guard_radius"] == 0.08

    # 같은 입력이면 같은 판정: depth mask 의 픽셀 = backproject 한 cloud 에서 지운 점의 픽셀.
    full = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam,
                       ag3s.config.pointcloud)
    kept, _ = filter_robot_points(full, arm, DEFAULT_ARM_Q, ag3s.config.pointcloud,
                                  inflation=ag3s._self_filter_inflation, guard_centre=held)
    removed = np.ones(len(full), bool)
    removed[np.isin(full.uv[:, 1] * 10_000 + full.uv[:, 0],
                    kept.uv[:, 1] * 10_000 + kept.uv[:, 0])] = False
    expect = np.zeros(scene.depth.shape, bool)
    expect[full.uv[removed, 1], full.uv[removed, 0]] = True
    np.testing.assert_array_equal(mask, expect)

    stats = debug["self_filter_stats"]
    assert stats["guard_active"] is True
    assert stats["guard_centroid"] == pytest.approx(held.tolist())
    sf = cs.metrics["self_filter"]
    assert sf["guard_active"] is True and sf["guard_radius"] == 0.08
    assert sf["inflation_by_link"] == {FINGERTIP_LINK: 0.0}
    assert sf["n_guard_protected_points"] == stats["n_guard_protected"]


def test_the_guard_centroid_is_the_one_held_before_this_frames_grounding(arm, scene):
    ag3s = AG3S(_cfg(self_filter_target_guard_radius=0.08), robot_model=arm)
    inputs = dict(attention_map=scene.attention_grid, depth=scene.depth,
                  camera_intrinsics=scene.camera_intrinsics, T_base_cam=scene.T_base_cam,
                  robot_state=DEFAULT_ARM_Q, phase="approach", image_hw=scene.hw)
    cs1, debug1 = ag3s.process_debug(**inputs)
    assert debug1["self_filter_stats"]["guard_enabled"] is True
    assert debug1["self_filter_stats"]["guard_active"] is False  # 첫 프레임엔 확정된 것이 없다
    assert cs1.metrics["self_filter"]["guard_centroid"] is None
    held = ag3s._target_confirm.held_centroid
    if held is None:
        pytest.skip("grounding declined on this scene; nothing to guard")
    _, debug2 = ag3s.process_debug(**inputs)
    assert debug2["self_filter_stats"]["guard_centroid"] == pytest.approx(held.tolist())


def test_the_fused_path_passes_the_same_objects_to_every_camera(arm, monkeypatch):
    scene3 = make_three_camera_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    ag3s = AG3S(_cfg(self_filter_inflation_by_link={FINGERTIP_LINK: 0.0},
                     self_filter_target_guard_radius=0.08), robot_model=arm)
    ag3s.process_multi(scene3.observations, phase="approach")
    held = ag3s._target_confirm.held_centroid
    if held is None:
        pytest.skip("grounding declined on this scene; nothing to guard")
    calls = _spy(monkeypatch)
    cs = ag3s.process_multi(scene3.observations, phase="approach")
    ag3s._depth_cameras_from(scene3.observations, None, None, None)
    n = len(scene3.observations)
    assert len(calls) == 2 * n  # 카메라마다 cloud 한 번 + depth mask 한 번
    for call in calls:
        assert call["inflation"] is ag3s._self_filter_inflation
        np.testing.assert_array_equal(call["guard_centre"], held)
    per_cam = cs.metrics["n_self_filter_guard_protected"]
    assert set(per_cam) == {o.camera_id.value for o in scene3.observations}
    assert cs.metrics["self_filter"]["n_guard_protected_points"] == sum(per_cam.values())
    assert cs.metrics["self_filter"]["guard_active"] is True


def test_self_filter_settings_report_the_effective_table(arm):
    ag3s = AG3S(_cfg(self_filter_inflation_by_link={FINGERTIP_LINK: 0.0},
                     self_filter_target_guard_radius=0.06), robot_model=arm)
    s = ag3s.self_filter_settings()
    assert s["enabled"] is True and s["n_spheres"] == arm.n_spheres
    # T26: the default is the sphere radius itself (0.0; it was 0.05 until 2026-09-28).
    assert s["inflation_default"] == 0.0
    assert s["inflation_effective"] == {"link_0": 0.0, "link_1": 0.0, FINGERTIP_LINK: 0.0}
    assert s["target_guard_radius"] == 0.06
    plain = AG3S(AG3SConfig(), robot_model=arm).self_filter_settings()
    assert set(plain["inflation_effective"].values()) == {0.0}
    assert AG3S(AG3SConfig()).self_filter_settings()["enabled"] is False


# ------------------------------------------------------------------ 실제 RB-Y1 자기 필터 모델


@pytest.fixture(scope="module")
def rby1_filter_model():
    mujoco = pytest.importorskip("mujoco")
    if not MODEL.exists():  # pragma: no cover - asset-dependent
        pytest.skip(f"model not present: {MODEL}")
    from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    scene = TransportScene.attach(model, data)
    mujoco.mj_forward(model, data)
    return build_robot_model(scene)


def test_group_names_land_on_the_real_self_filter_model(rby1_filter_model):
    """`gripper` 이 서버의 자기 필터 모델(전신)에서 손바닥 둘 + 손가락 넷 전부에 닿는다."""
    from AG3S.ag3s.experiments.reports.grounding_report import ARM_LINKS, GRIPPER_LINKS

    names = rby1_filter_model.sphere_link_names
    assert set(GRIPPER_LINKS) <= set(names) and set(ARM_LINKS) <= set(names)
    cfg = PointCloudConfig(self_filter_inflation=0.05,  # T26: explicit (default is 0.0)
                           self_filter_inflation_by_link={"gripper": 0.0})
    table = inflation_table(names, resolve_self_filter_inflation(names, cfg))
    assert {link for link, v in table.items() if v == 0.0} == set(GRIPPER_LINKS)
    assert table["base"] == cfg.self_filter_inflation  # base 누수(T1) 를 막는 값은 그대로


# ------------------------------------------------------------------ serve_safe: flag · 시작 로그


BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def test_no_flag_means_no_pointcloud_override():
    from AG3S.trajopt import serve_safe

    args = serve_safe.build_parser().parse_args(BASE_ARGV)
    assert serve_safe.self_filter_options(args) == {}


def test_flags_pass_group_names_through_unexpanded():
    from AG3S.trajopt import serve_safe

    args = serve_safe.build_parser().parse_args(BASE_ARGV + [
        "--self-filter-inflation-link", "gripper=0.01", "link_left_arm_6=0.03",
        "--self-filter-target-guard", "0.07"])
    opts = serve_safe.self_filter_options(args)
    assert opts == {"self_filter_inflation_by_link": {"gripper": 0.01, "link_left_arm_6": 0.03},
                    "self_filter_target_guard_radius": 0.07}
    # 그대로 config 가 된다.
    cfg = AG3SConfig.from_dict({"pointcloud": {"range_max": 2.0, **opts}})
    assert cfg.pointcloud.self_filter_inflation_by_link["gripper"] == 0.01


@pytest.mark.parametrize("bad", ["gripper", "all=0.01", "=0.01", "gripper=wide", "gripper=-0.01"])
def test_malformed_inflation_flags_die_at_parse_time(bad):
    from AG3S.trajopt import serve_safe

    with pytest.raises(ValueError):
        serve_safe.parse_self_filter_inflation([bad])
    ap = serve_safe.build_parser()
    args = ap.parse_args(BASE_ARGV + ["--self-filter-inflation-link", bad])
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)


def test_negative_guard_and_no_safe_combination_are_refused():
    from AG3S.trajopt import serve_safe

    ap = serve_safe.build_parser()
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(
            ap, ap.parse_args(BASE_ARGV + ["--self-filter-target-guard", "-0.01"]))
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(
            ap, ap.parse_args(BASE_ARGV + ["--no-safe", "--self-filter-target-guard", "0.05"]))
    # 기본 조합은 통과한다.
    serve_safe.reject_bad_flag_combinations(ap, ap.parse_args(BASE_ARGV))


def test_build_ag3s_forwards_the_flags_into_pointcloud():
    """`build_ag3s` 가 `self_filter` 를 `pointcloud` 에 얹는다 (소스로 확인 — MuJoCo 없이)."""
    import inspect

    from AG3S.trajopt import serve_safe

    src = inspect.getsource(serve_safe.build_ag3s)
    assert '"pointcloud": {"range_max": range_max, **dict(self_filter or {})}' in src
    assert "announce_self_filter(ag3s.self_filter_settings()" in src
    assert "self_filter=self_filter_options(args)" in inspect.getsource(serve_safe.main)


def test_start_log_default_is_one_line_and_says_yaml_does_not_reach(arm, caplog):
    from AG3S.trajopt import serve_safe

    caplog.set_level(logging.INFO, logger=serve_safe.__name__)
    serve_safe.announce_self_filter(AG3S(AG3SConfig(), robot_model=arm).self_filter_settings())
    msgs = [r for r in caplog.records if r.name == serve_safe.__name__]
    assert len(msgs) == 1 and msgs[0].levelno == logging.INFO
    text = msgs[0].getMessage()
    # T26: default inflation 0.0 mm (sphere radius itself; was 50.0 mm).
    assert f"{arm.n_spheres} 구" in text and "0.0 mm" in text and "50.0 mm" not in text
    assert "guard 끔" in text
    assert "닿지 않는다" in text and "코드 기본값" in text


def test_start_log_prints_the_effective_table_when_overridden(arm, caplog):
    from AG3S.trajopt import serve_safe

    caplog.set_level(logging.INFO, logger=serve_safe.__name__)
    cli = {"self_filter_inflation_by_link": {FINGERTIP_LINK: 0.0},
           "self_filter_target_guard_radius": 0.06}
    ag3s = AG3S(AG3SConfig.from_dict({"pointcloud": cli}), robot_model=arm)
    serve_safe.announce_self_filter(ag3s.self_filter_settings(), cli=cli)
    rec = [r for r in caplog.records if r.name == serve_safe.__name__]
    assert len(rec) == 1 and rec[0].levelno == logging.WARNING
    text = rec[0].getMessage()
    assert "NOT THE SINGLE-VALUE DEFAULT" in text and "60.0 mm" in text
    # T26: with the global at 0.0 the override equals it — every link reads 0.0 mm.
    assert f"0.0 mm :  3 link — link_0, link_1, {FINGERTIP_LINK}" in text
    assert "CLI" in text
