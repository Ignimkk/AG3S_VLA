"""T8b — capsule 을 link 별로 **짧게 자르고**, 손바닥을 제약 모델에 넣는다.

사용자 판정: *"팔꿈치에서 손가락까지 한 덩어리로 덮지 마라. 그리퍼의 2-finger 부분에 작은 구를
만들어라."*

측정된 사실이 그 판정의 근거다. `link_*_arm_5` 의 URDF capsule 은 **r 75 mm · L 250 mm** 이고
축이 link frame 에서 **z ∈ [−0.225, +0.025]** 인데, 같은 frame 에서 손목(`link_*_arm_6`) 원점이
0.0, `FT_sensor_*` 가 −0.1087, **손바닥(`ee_*`)이 −0.1548**, 2 지 그리퍼 뿌리가 −0.2278 이다.
capsule **하나가** 손목·손바닥·그리퍼를 전부 삼킨다. 다른 팔 capsule 은 r 40 / 35 mm 로 얇다.

그래서 축이 둘이고 이 파일은 둘을 갈라 지킨다.

* **굵기** (`capsule_radius_scale_by_link`, T7b) — 구를 얇게 한다. 손가락이 그 대상이다.
* **길이** (`capsule_extent_by_link`, 여기) — 구를 **덜 놓는다**. `arm_5` 가 그 대상이다.

그리고 셋을 더 지킨다.

* **자르면 손바닥이 벌거벗는다.** `ee_left`/`ee_right` 는 그 전까지 제약 모델에 없었고 (URDF 에
  collision capsule 이 없다 — geometry 는 MJCF mesh 실측에서 온다), `arm_5` 덩어리가 유일하게
  덮고 있었다. 그래서 `GRIPPER_LINKS` 에 손바닥이 들어왔고, 접촉 권한 집합에도 들어 있다.
* **`max_spheres_per_capsule` 이 `sphere_spacing` 을 무력화할 수 있다.** 상한이 먼저 걸리면 구가
  촘촘해지지 않고, 그러면 `r_eff = sqrt((r·scale)² + (간격/2)²)` 의 간격 항이 안 줄어 **얇게
  만들었다고 믿는데 안 얇아진다.** 그때 시작 로그가 크게 말한다.
* **기본값은 지금 그대로다.** 두 손잡이를 안 주면 구 수·반지름·로그가 예전과 같다.
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import logging
import math
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.config import DEFAULT_CONTACT_LINKS
from AG3S.ag3s.experiments.reports.grounding_report import (
    ARM_LINKS,
    FINGER_LINKS,
    GRIPPER_LINKS,
    PALM_LINKS,
)
from AG3S.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf
from AG3S.trajopt import serve_safe

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]

#: 손바닥·손가락 capsule 은 URDF 가 아니라 MJCF mesh 실측에서 온다 (`gap_filling_capsules`).
MODEL = pathlib.Path(
    _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
)

ARM5 = "link_left_arm_5"
#: 이 capsule 이 삼키고 있는 것들의 **link frame z** (관절값 0). 실측값이다.
WRIST_Z, FT_Z, PALM_Z, FINGER_ROOT_Z = 0.0, -0.1087, -0.1548, -0.2278
#: A2(T8c)가 고른 절단면. 이 파일은 값을 판정하지 않고 **기계가 그 값을 실현하는지**만 본다.
Z_CUT = -0.10


@pytest.fixture(scope="module")
def urdf():
    return parse_urdf(RBY1_URDF)


@pytest.fixture(scope="module")
def extra_capsules():
    mujoco = pytest.importorskip("mujoco")
    if not MODEL.exists():  # pragma: no cover - asset-dependent
        pytest.skip(f"model not present: {MODEL}")
    from AG3S.ag3s.experiments.sources.mujoco_source import gap_filling_capsules

    return tuple(gap_filling_capsules(mujoco.MjModel.from_xml_path(str(MODEL))))


def chain(urdf, extra_capsules, **kw):
    return UrdfSphereChain(urdf, extra_capsules=extra_capsules, link_filter=ARM_LINKS, **kw)


def radii_of(model, link: str) -> np.ndarray:
    return np.asarray([r for n, r in zip(model.sphere_link_names, model.radii) if n == link])


# --- 0. 지금 모델이 왜 이 손잡이를 필요로 하는가 (실측 고정) ---------------------------


def test_the_arm5_capsule_really_does_swallow_the_wrist_palm_and_gripper(urdf):
    """이 파일 전체의 근거. 숫자가 바뀌면 판정의 전제가 바뀐 것이므로 여기서 걸린다."""
    caps = [c for c in urdf.capsules if c.link == ARM5]
    assert len(caps) == 1, "arm_5 는 capsule 하나다 — 그것이 문제의 모양이다"
    cap = caps[0]
    assert cap.radius == pytest.approx(0.075, abs=1e-6)
    assert cap.length == pytest.approx(0.250, abs=1e-6)
    axis_z = float(cap.origin[2, 2])
    p_z = float(cap.origin[2, 3])
    assert abs(axis_z) == pytest.approx(1.0, abs=1e-3), "축이 link z 와 나란하다"
    lo, hi = p_z - cap.length / 2.0, p_z + cap.length / 2.0
    assert (lo, hi) == (pytest.approx(-0.225, abs=1e-6), pytest.approx(0.025, abs=1e-6))
    # 삼켜지는 것들이 정말 이 capsule **표면** 안에 있다. 기준이 축이 아니라 표면인 것이
    # 요점이다 — 축은 −0.225 에서 끝나는데 표면은 반지름만큼 더 내려가 −0.300 까지 간다.
    surface_lo = lo - cap.radius
    assert surface_lo == pytest.approx(-0.300, abs=1e-6)
    for z in (WRIST_Z, FT_Z, PALM_Z, FINGER_ROOT_Z):
        assert surface_lo <= z <= hi
    # **이웃한 팔 capsule 은 얇다** — 튀는 것은 이 하나다 (arm_3 40 mm · arm_4 35 mm).
    neighbours = {c.link: c.radius for c in urdf.capsules
                  if c.link in ("link_left_arm_3", "link_left_arm_4")}
    assert neighbours == {"link_left_arm_3": pytest.approx(0.040, abs=1e-6),
                          "link_left_arm_4": pytest.approx(0.035, abs=1e-6)}
    assert cap.radius > 1.8 * neighbours["link_left_arm_4"]
    # 그리고 이 팔에서 **가장 긴** capsule 이다 — 길이가 문제의 절반이다.
    assert cap.length == max(c.length for c in urdf.capsules
                             if c.link.startswith("link_left_arm_"))


def test_the_finger_capsule_axes_are_perpendicular_to_link_z(extra_capsules):
    """손가락은 **잘라서** 다룰 수 없다 — 축이 z 평면에 누워 있다. 그래서 굵기로 다룬다."""
    fingers = [c for c in extra_capsules if c.link == "ee_finger_l1"]
    assert fingers, "손가락 capsule 은 MJCF mesh 에서 온다"
    for cap in fingers:
        assert abs(float(cap.origin[2, 2])) < 1e-2


# --- 1. 기본값 ------------------------------------------------------------------------


def test_the_trim_defaults_to_nothing(urdf, extra_capsules):
    import inspect

    assert inspect.signature(UrdfSphereChain.__init__).parameters[
        "capsule_extent_by_link"].default is None
    plain = chain(urdf, extra_capsules)
    explicit = chain(urdf, extra_capsules, capsule_extent_by_link={})
    assert explicit.n_spheres == plain.n_spheres
    assert np.array_equal(explicit.radii, plain.radii)
    assert explicit.capsule_trims == () and plain.capsule_trims == ()
    assert "covers every capsule" in plain.coverage_report()


def test_the_server_flag_defaults_to_nothing():
    args = serve_safe.build_parser().parse_args(BASE_ARGV)
    assert tuple(args.capsule_extent_link) == ()
    assert serve_safe.sphere_options(args) == {}, "안 준 flag 는 키가 아예 없다"


def test_the_default_startup_log_says_nothing_about_the_sphere_model(caplog):
    with caplog.at_level(logging.WARNING, logger=serve_safe.__name__):
        # T27: `--links` 기본이 `arms` → `gripper`. 이 테스트가 박는 것은 "기본 실행은 구 모델에
        # 대해 조용하다" 이므로 **기본 범위**(이제 gripper)로 부른다 — `arms` 는 이제 비기본이라
        # 범위 경고가 따로 나간다 (`test_gripper_only_constraints`).
        serve_safe.announce_diagnostic_scope(
            links=serve_safe.DEFAULT_LINKS, self_collision=True, link_scales={}, link_extents={},
            radius_by_link="link_left_arm_5=5×81.25 mm", filter_spheres=218)
    assert caplog.text == ""


# --- 2. 자르면 무엇이 달라지나 --------------------------------------------------------


def test_cutting_arm5_removes_the_distal_spheres_and_touches_nothing_else(urdf, extra_capsules):
    base = chain(urdf, extra_capsules)
    cut = chain(urdf, extra_capsules, capsule_extent_by_link={ARM5: Z_CUT})
    assert cut.n_spheres < base.n_spheres
    # **다른 link 은 구 수도 반지름도 그대로다.** 자르는 것은 이름으로 하는 일이다.
    for link in set(base.sphere_link_names) - {ARM5}:
        assert np.array_equal(np.sort(radii_of(cut, link)), np.sort(radii_of(base, link))), link
    kept = [c for n, c in zip(cut.sphere_link_names, cut.sphere_centers_numeric(
        np.zeros(cut.nq))[0]) if n == ARM5]
    assert kept, "자른 뒤에도 유지 구간에는 구가 남는다"


def test_every_kept_sphere_centre_is_inside_the_window(urdf, extra_capsules):
    """유지 구간 밖에는 구 **중심**이 하나도 없다. 그것이 '자른다' 의 기계적 뜻이다."""
    cut = chain(urdf, extra_capsules, capsule_extent_by_link={ARM5: Z_CUT})
    cap = [c for c in urdf.capsules if c.link == ARM5][0]
    t_lo, t_hi = cut._kept_axis_span(cap)
    zs = [float(cap.origin[2, 3] + (cap.origin[:3, :3] @ np.array([0.0, 0.0, t]))[2])
          for t in np.linspace(t_lo, t_hi, 8)]
    assert min(zs) >= Z_CUT - 1e-9
    assert max(zs) <= 0.025 + 1e-9


def test_an_explicit_window_keeps_only_that_band(urdf, extra_capsules):
    both = chain(urdf, extra_capsules, capsule_extent_by_link={ARM5: (-0.05, 0.0)})
    cap = [c for c in urdf.capsules if c.link == ARM5][0]
    t_lo, t_hi = both._kept_axis_span(cap)
    assert t_hi - t_lo == pytest.approx(0.05, abs=1e-9)


def test_a_window_that_misses_the_capsule_drops_it_entirely(urdf, extra_capsules):
    """빈 교집합은 조용한 근사가 아니라 **구 0 개**다."""
    gone = chain(urdf, extra_capsules, capsule_extent_by_link={ARM5: (0.20, 0.30)})
    assert ARM5 not in set(gone.sphere_link_names)
    trim = [t for t in gone.capsule_trims if t.link == ARM5][0]
    assert trim.n_spheres_after == 0 and trim.n_spheres_before > 0


def test_cutting_alone_does_not_pull_the_surface_all_the_way_back(urdf, extra_capsules):
    """**잘라도 표면은 반지름만큼 부푼다.** 그래서 굵기 손잡이와 함께 쓰는 것이 정상이다."""
    cut = chain(urdf, extra_capsules, capsule_extent_by_link={ARM5: Z_CUT})
    trim = [t for t in cut.capsule_trims if t.link == ARM5][0]
    assert trim.reach_z_min_before < -0.24, "자르기 전 표면은 그리퍼 뿌리보다 아래까지 간다"
    assert trim.reach_z_min_after > trim.reach_z_min_before, "표면이 물러난다"
    assert trim.reach_z_min_after > Z_CUT - radii_of(cut, ARM5).max() - 1e-9
    # 자른 절단면 자체보다는 여전히 아래다 — 그 차이가 곧 유효 반지름이다.
    assert trim.reach_z_min_after < Z_CUT


def test_the_two_handles_compose_on_the_same_link(urdf, extra_capsules):
    """막지 않는다. 한 link 에 굵기와 길이를 동시에 걸 수 있어야 한다."""
    both = chain(urdf, extra_capsules,
                 capsule_radius_scale_by_link={ARM5: 0.25},
                 capsule_extent_by_link={ARM5: Z_CUT})
    only_cut = chain(urdf, extra_capsules, capsule_extent_by_link={ARM5: Z_CUT})
    assert radii_of(both, ARM5).max() < radii_of(only_cut, ARM5).max()
    trim = [t for t in both.capsule_trims if t.link == ARM5][0]
    assert trim.reach_z_min_after > [t for t in only_cut.capsule_trims
                                     if t.link == ARM5][0].reach_z_min_after


# --- 3. 무엇이 풀려났는지 반드시 말한다 -----------------------------------------------


def test_the_report_names_the_links_that_were_released(urdf, extra_capsules):
    cut = chain(urdf, extra_capsules, capsule_extent_by_link={ARM5: Z_CUT})
    trim = [t for t in cut.capsule_trims if t.link == ARM5][0]
    released = {name: covered for name, _, covered in trim.released_links}
    assert "ee_left" in released, "손바닥은 절단면 아래에 있으므로 반드시 언급돼야 한다"
    assert released["ee_left"] is True, "손바닥은 이제 자기 구를 갖는다 (T8b)"
    # 관절값 0 기준 z 가 실측과 같아야 한다 — 이 판정의 좌표계가 맞다는 뜻이다.
    z_of = {name: z for name, z, _ in trim.released_links}
    assert z_of["ee_left"] == pytest.approx(PALM_Z, abs=1e-3)
    report = cut.coverage_report()
    assert "CAPSULES ARE CUT SHORT" in report and "ee_left" in report


def test_a_released_link_without_spheres_is_called_naked(urdf, extra_capsules):
    """손바닥을 다시 빼 보면 그 사실이 로그에 **벌거벗었다**고 남는다."""
    no_palm = tuple(n for n in ARM_LINKS if n not in PALM_LINKS)
    model = UrdfSphereChain(urdf, extra_capsules=extra_capsules, link_filter=no_palm,
                            capsule_extent_by_link={ARM5: Z_CUT})
    trim = [t for t in model.capsule_trims if t.link == ARM5][0]
    naked = {n for n, _, covered in trim.released_links if not covered}
    assert "ee_left" in naked
    assert "자기 구가 없다" in model.coverage_report()


def test_the_trim_is_logged_at_construction(urdf, extra_capsules, caplog):
    with caplog.at_level(logging.WARNING,
                         logger="AG3S.ag3s.robot_models.urdf_sphere_chain"):
        chain(urdf, extra_capsules, capsule_extent_by_link={ARM5: Z_CUT})
    assert "CAPSULES ARE CUT SHORT" in caplog.text


# --- 4. 거절하는 것 -------------------------------------------------------------------


def test_an_unknown_link_name_is_refused(urdf, extra_capsules):
    with pytest.raises(ValueError, match="not in the URDF"):
        chain(urdf, extra_capsules, capsule_extent_by_link={"link_left_arm_9": -0.1})


def test_a_link_without_a_capsule_in_this_model_is_refused(urdf):
    """URDF 에는 있지만 이 모델에 capsule 이 없는 이름 — 자를 것이 없는데 잘랐다고 믿는다."""
    with pytest.raises(ValueError, match="no capsule for"):
        UrdfSphereChain(urdf, link_filter=ARM_LINKS,
                        capsule_extent_by_link={"ee_left": -0.1})


@pytest.mark.parametrize("value", [(0.1, 0.1), (0.2, 0.1), "x", None])
def test_a_malformed_window_is_refused(urdf, extra_capsules, value):
    with pytest.raises(ValueError):
        chain(urdf, extra_capsules, capsule_extent_by_link={ARM5: value})


def test_millimetres_are_refused_because_they_would_silently_cut_nothing():
    with pytest.raises(ValueError, match="미터"):
        serve_safe.parse_link_extents([f"{ARM5}=-100"])


def test_group_names_are_refused_for_a_cut():
    with pytest.raises(ValueError, match="집합 이름"):
        serve_safe.parse_link_extents(["gripper=-0.1"])


def test_the_flag_is_refused_together_with_no_safe():
    ap = serve_safe.build_parser()
    args = ap.parse_args(BASE_ARGV + ["--no-safe", "--capsule-extent-link", f"{ARM5}=-0.10"])
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)


@pytest.mark.parametrize("text,expect", [
    (f"{ARM5}=-0.10", {ARM5: (-0.10, math.inf)}),
    (f"{ARM5}=-0.10:0.03", {ARM5: (-0.10, 0.03)}),
])
def test_the_flag_parses_both_forms(text, expect):
    assert serve_safe.parse_link_extents([text]) == expect


# --- 5. 손바닥이 제약 모델에 들어왔다 --------------------------------------------------


def test_the_gripper_set_now_holds_the_palms_and_stays_a_subset_of_the_arm_set():
    assert PALM_LINKS == ("ee_left", "ee_right")
    assert GRIPPER_LINKS == PALM_LINKS + FINGER_LINKS
    assert set(GRIPPER_LINKS) < set(ARM_LINKS), "gripper 는 arms 의 부분집합이어야 한다"
    assert serve_safe.constraint_link_filter("gripper") == GRIPPER_LINKS


def test_the_palm_actually_gets_spheres_in_both_link_sets(urdf, extra_capsules):
    arms = chain(urdf, extra_capsules)
    gripper = UrdfSphereChain(urdf, extra_capsules=extra_capsules,
                              link_filter=GRIPPER_LINKS)
    for model in (arms, gripper):
        for palm in PALM_LINKS:
            assert radii_of(model, palm).size > 0, (palm, model)
    # 손바닥 구는 **양쪽에서 같다** — 두 모델이 같은 geometry 를 본다는 뜻이다.
    assert np.allclose(np.sort(radii_of(arms, "ee_left")),
                       np.sort(radii_of(gripper, "ee_left")))


def test_the_palm_comes_from_the_mjcf_mesh_not_the_urdf(urdf, extra_capsules):
    """URDF 에 collision capsule 이 없다. 그래서 씬 파일 없이는 손바닥 구가 안 생긴다."""
    assert not [c for c in urdf.capsules if c.link in PALM_LINKS]
    assert [c for c in extra_capsules if c.link in PALM_LINKS]


def test_the_palm_is_contact_authorized_out_of_the_box():
    """넣기만 하면 부족하다 — 사과를 잡을 때 손바닥이 사과 **바로 위**에 온다 (A2 T8e:
    배율 1.0 에서 `ee_left` × apple 이 exec8 279 행 · −31.55 mm). 권한 집합에 있어야 한다."""
    authorized = {n for names in DEFAULT_CONTACT_LINKS.values() for n in names}
    for palm in PALM_LINKS:
        assert palm in authorized
    for finger in FINGER_LINKS:
        assert finger in authorized


def test_the_self_filter_model_has_no_sphere_options_channel():
    """**비대칭은 의도된 것이다.** 자기 필터가 가늘어지면 그 link 의 점이 장애물로 샌다."""
    import inspect

    from AG3S.ag3s.experiments.reports import grounding_report

    assert "sphere_options" not in inspect.signature(
        grounding_report.build_robot_model).parameters
    assert "sphere_options" in inspect.signature(
        grounding_report.build_constraint_robot_model).parameters


# --- 6. 전역 배율 · 간격 · 상한 --------------------------------------------------------


def test_the_global_scale_reaches_every_constraint_link_including_the_palms(
        urdf, extra_capsules):
    base = chain(urdf, extra_capsules)
    thin = chain(urdf, extra_capsules, capsule_radius_scale=0.2)
    assert thin.n_spheres == base.n_spheres, "배율은 구 수를 바꾸지 않는다"
    for link in sorted(set(base.sphere_link_names)):
        assert radii_of(thin, link).max() < radii_of(base, link).max(), link
    for palm in PALM_LINKS:
        assert radii_of(thin, palm).max() < radii_of(base, palm).max()


def test_the_spacing_flags_are_exposed_on_the_server():
    args = serve_safe.build_parser().parse_args(
        BASE_ARGV + ["--sphere-spacing", "0.4", "--max-spheres-per-capsule", "32"])
    options = serve_safe.sphere_options(args)
    assert options["sphere_spacing"] == 0.4
    assert options["max_spheres_per_capsule"] == 32


def test_lowering_the_spacing_without_raising_the_cap_is_shouted_about(urdf, extra_capsules):
    """A2 T8e: 상한 8 이 구 수를 잘라 spacing 이 안 듣는다. 조용히 지나가면 안 된다."""
    capped = chain(urdf, extra_capsules, sphere_spacing=0.4)
    assert capped._spacing_capped, "상한이 걸렸다는 사실이 기록돼야 한다"
    report = capped.coverage_report()
    assert "max_spheres_per_capsule IS BINDING" in report
    raised = chain(urdf, extra_capsules, sphere_spacing=0.4, max_spheres_per_capsule=32)
    assert not raised._spacing_capped
    assert "IS BINDING" not in raised.coverage_report()


def test_the_cap_is_what_stops_the_radius_from_falling(urdf, extra_capsules):
    """`r_eff = sqrt((r·scale)² + (간격/2)²)` — 상한이 걸리면 간격 항이 안 줄고 반지름이 멈춘다."""
    capped = chain(urdf, extra_capsules, sphere_spacing=0.4, capsule_radius_scale=0.2)
    raised = chain(urdf, extra_capsules, sphere_spacing=0.4, capsule_radius_scale=0.2,
                   max_spheres_per_capsule=32)
    assert radii_of(raised, ARM5).max() < radii_of(capped, ARM5).max()
    assert raised.n_spheres > capped.n_spheres


def test_the_row_budget_separates_spheres_from_qp_rows(caplog):
    """구를 늘리면 질의 행은 늘고 **QP 행은 안 는다.** 로그가 그것을 갈라 말해야 한다."""
    with caplog.at_level(logging.INFO, logger=serve_safe.__name__):
        serve_safe.announce_row_budget(n_constraint_spheres=266, n_filter_spheres=218,
                                      planned=8, rows_per_step=24)
    assert "2128" in caplog.text, "ESDF 질의 행 = 8 × 266"
    assert "192" in caplog.text, "QP 행 = 8 × 24, 구 개수와 무관"
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=serve_safe.__name__):
        serve_safe.announce_row_budget(n_constraint_spheres=144, n_filter_spheres=218)
    assert "144" in caplog.text and "218" in caplog.text


def test_build_ag3s_announces_the_row_budget_and_the_sphere_model():
    import inspect

    src = inspect.getsource(serve_safe.build_ag3s)
    assert "announce_row_budget(" in src
    assert "radius_by_link=" in src and "link_extents=" in src


# --- 7. 절단 + 배율 + 간격을 같이 준 판 (launch 후보의 기계만 확인한다) ----------------


def test_the_launch_candidate_shape_is_reachable_by_flags_alone(urdf, extra_capsules):
    """값은 A2 가 정한다. 여기서 보는 것은 **flag 만으로 그 판이 만들어지는가** 다."""
    args = serve_safe.build_parser().parse_args(BASE_ARGV + [
        "--sphere-spacing", "0.4", "--capsule-radius-scale", "0.2",
        "--max-spheres-per-capsule", "32",
        "--capsule-extent-link", f"{ARM5}=-0.10", "link_right_arm_5=-0.10"])
    options = serve_safe.sphere_options(args)
    model = chain(urdf, extra_capsules, **options)
    assert not model._spacing_capped, "상한이 안 걸려야 spacing 이 실제로 듣는다"
    assert radii_of(model, ARM5).max() < 0.0466, "A2 의 r_crit 46.64 mm 아래"
    assert radii_of(model, "ee_finger_l1").max() < 0.00776, "A2 의 r_crit 7.75 mm 아래"
    for palm in PALM_LINKS:
        assert radii_of(model, palm).size > 0, "손바닥이 제약에 남아 있어야 한다"
    assert {t.link for t in model.capsule_trims} == {ARM5, "link_right_arm_5"}
