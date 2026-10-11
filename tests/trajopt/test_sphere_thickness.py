"""제약 모델의 구 굵기를 조절할 수 있게 한다 (T6f 고칠 것 2).

lead 가 MJCF mesh 로 실측한 값과 지금 제약 구가 다르다 — `link_*_arm_5` 는 mesh 65.4~68.4 mm
인데 구는 **81.2 mm** 다. 절반은 URDF capsule 자체(75 mm)가 mesh 보다 두꺼운 것이고, 나머지는
구 사슬이 capsule 을 담으려고 붙이는 팽창(`sqrt(r² + (간격/2)²)`)이다.

이 파일이 지키는 것:

* **기본값이면 지금과 글자 그대로 같다.** 옛 공식을 테스트가 따로 다시 계산해 맞춰 본다 —
  코드가 스스로를 채점하지 않게.
* 간격을 좁히면 반지름이 **내려가고 덮개는 유지된다**.
* 반지름에 상한을 걸면 덮개를 잃고, 그 사실이 **기록에도 로그에도** 남는다.
* **자기 필터 모델에는 이 통로가 없다.** 거기서 가늘어지면 로봇 점이 장애물로 샌다.
"""

from __future__ import annotations

import inspect
import logging
import math

import numpy as np
import pytest

from AG3S.ag3s.experiments.reports.grounding_report import (
    ARM_LINKS,
    build_constraint_robot_model,
    build_robot_model,
)
from AG3S.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf
from AG3S.trajopt import serve_safe

#: 필수 인자만. 여기에 아무것도 더하지 않은 것이 "지금과 같은 서버" 다.
BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]

#: 사용자가 가늘게 하고 싶어 한 link (T6f) — 양팔 팔뚝.
FOREARMS = ("link_left_arm_5", "link_right_arm_5")


@pytest.fixture(scope="module")
def urdf():
    return parse_urdf(RBY1_URDF)


@pytest.fixture(scope="module")
def default_model(urdf):
    return UrdfSphereChain(urdf, link_filter=ARM_LINKS)


def _max_radius_by_link(model) -> dict:
    out: dict = {}
    for link, radius in zip(model.sphere_link_names, model.radii):
        out[link] = max(out.get(link, 0.0), float(radius))
    return out


def _old_formula(cap, spacing=1.0, cap_count=8):
    """T6f 이전의 계산을 테스트가 **독립적으로** 다시 한다.

    같은 코드를 불러 비교하면 어떤 변경도 통과한다. 공식은 클래스 머리말에 글로 적혀 있고
    (`sqrt(r² + (간격/2)²)`), 그것을 여기 옮겨 적는 것이 기본값 고정의 유일한 방법이다.
    """
    n = int(math.ceil(cap.length / max(spacing * cap.radius, 1e-9))) + 1
    n = int(np.clip(n, 1, cap_count))
    if n < 2:
        return n, cap.radius + cap.length / 2.0
    step = cap.length / (n - 1)
    return n, math.sqrt(cap.radius ** 2 + (step / 2.0) ** 2)


# --- 기본값은 지금과 같다 ------------------------------------------------------------


def test_the_default_sphere_set_matches_the_formula_it_always_used(default_model):
    counts: dict = {}
    for link in default_model.sphere_link_names:
        counts[link] = counts.get(link, 0) + 1
    radii = _max_radius_by_link(default_model)

    for cap in default_model.capsules:
        n, radius = _old_formula(cap)
        assert counts[cap.link] == n, cap.link
        assert radii[cap.link] == pytest.approx(radius), cap.link


def test_the_default_forearm_sphere_is_the_number_t6d_measured(default_model):
    """81.2 mm — T6d 가 인용한 값. 기본값이 흔들리면 그 판정과의 연결이 끊긴다."""
    radii = _max_radius_by_link(default_model)
    for link in FOREARMS:
        assert radii[link] * 1000 == pytest.approx(81.2, abs=0.1)


def test_the_defaults_cover_every_capsule(default_model):
    assert default_model.coverage_shortfall == ()
    assert "covers every capsule" in default_model.coverage_report()


def test_the_knobs_default_to_the_old_values(urdf):
    params = inspect.signature(UrdfSphereChain.__init__).parameters
    assert params["sphere_spacing"].default == 1.0
    assert params["max_spheres_per_capsule"].default == 8
    assert params["capsule_radius_scale"].default == 1.0
    assert params["max_sphere_radius"].default is None
    # 명시적으로 기본값을 줘도 같은 모델이어야 한다 — 기본값이 두 곳에 적히지 않았다는 뜻이다.
    explicit = UrdfSphereChain(urdf, link_filter=ARM_LINKS, sphere_spacing=1.0,
                               max_spheres_per_capsule=8, capsule_radius_scale=1.0,
                               max_sphere_radius=None)
    plain = UrdfSphereChain(urdf, link_filter=ARM_LINKS)
    assert np.array_equal(explicit.radii, plain.radii)
    assert explicit.n_spheres == plain.n_spheres


def test_build_constraint_robot_model_passes_no_options_by_default():
    """`sphere_options=None` 이면 호출이 예전과 글자 그대로 같다."""
    params = inspect.signature(build_constraint_robot_model).parameters
    assert params["sphere_options"].default is None
    # 자기 필터 쪽에는 그 통로가 아예 없다.
    assert "sphere_options" not in inspect.signature(build_robot_model).parameters


# --- 촘촘하게 하면 가늘어진다 (덮개는 그대로) ----------------------------------------


def test_finer_spacing_thins_the_spheres_without_losing_coverage(urdf, default_model):
    fine = UrdfSphereChain(urdf, link_filter=ARM_LINKS,
                           sphere_spacing=0.4, max_spheres_per_capsule=16)
    before = _max_radius_by_link(default_model)
    after = _max_radius_by_link(fine)
    for link in FOREARMS:
        assert after[link] < before[link]
    # 값을 지어내지 않는다: 팽창 항은 `(간격/2)²` 이므로 URDF capsule 반지름이 하한이다.
    forearm = next(c for c in fine.capsules if c.link == FOREARMS[0])
    assert after[FOREARMS[0]] > forearm.radius
    assert fine.n_spheres > default_model.n_spheres
    assert fine.coverage_shortfall == ()


def test_the_sphere_cap_binds_before_the_spacing_does(urdf, default_model):
    """상한을 그대로 두고 간격만 낮추면 아무 일도 안 일어난다 — flag 를 하나만 주고
    가늘어졌다고 믿는 것을 막기 위해 이 성질을 글로 박는다."""
    capped = UrdfSphereChain(urdf, link_filter=ARM_LINKS, sphere_spacing=0.1)
    forearm_spheres = sum(1 for link in capped.sphere_link_names if link == FOREARMS[0])
    assert forearm_spheres == 8
    assert _max_radius_by_link(capped)[FOREARMS[0]] < _max_radius_by_link(default_model)[FOREARMS[0]]


# --- 덮지 못하면 크게 말한다 ---------------------------------------------------------


def test_a_radius_cap_is_recorded_as_a_coverage_shortfall(urdf):
    thin = UrdfSphereChain(urdf, link_filter=ARM_LINKS, max_sphere_radius=0.068)
    assert thin.coverage_shortfall, "덮개를 잃었는데 기록이 없다"
    links = {s.link for s in thin.coverage_shortfall}
    assert set(FOREARMS) <= links
    worst = max(thin.coverage_shortfall, key=lambda s: s.shortfall_m)
    assert worst.shortfall_m > 0.0
    assert worst.sphere_radius < worst.required_radius
    assert _max_radius_by_link(thin)[FOREARMS[0]] == pytest.approx(0.068)


def test_a_radius_scale_is_recorded_too(urdf):
    thin = UrdfSphereChain(urdf, link_filter=ARM_LINKS, capsule_radius_scale=0.9)
    assert {s.link for s in thin.coverage_shortfall} == {c.link for c in thin.capsules}


def test_the_shortfall_is_shouted_at_construction(urdf, caplog):
    """진입점이 여럿이라 클래스가 직접 찍는다 — 한 곳만 안 찍으면 그 실행이 기록에 "예전과
    같은 모델" 로 남는다."""
    with caplog.at_level(logging.WARNING):
        UrdfSphereChain(urdf, link_filter=ARM_LINKS, max_sphere_radius=0.068)
    text = caplog.text
    assert "DOES NOT COVER THE ROBOT" in text
    assert FOREARMS[0] in text
    # 무슨 설정이 그렇게 만들었는지도 같은 덩어리에 있다.
    assert "max_sphere_radius" in text


def test_nothing_is_shouted_for_the_defaults(urdf, caplog):
    with caplog.at_level(logging.WARNING):
        UrdfSphereChain(urdf, link_filter=ARM_LINKS)
    assert "DOES NOT COVER" not in caplog.text


def test_nonsense_knobs_are_refused(urdf):
    for kwargs in ({"sphere_spacing": 0.0}, {"max_spheres_per_capsule": 0},
                   {"capsule_radius_scale": -1.0}, {"max_sphere_radius": 0.0}):
        with pytest.raises(ValueError):
            UrdfSphereChain(urdf, link_filter=ARM_LINKS, **kwargs)


# --- 서버 flag -----------------------------------------------------------------------


def test_the_server_gives_no_sphere_options_by_default():
    args = serve_safe.build_parser().parse_args(BASE_ARGV)
    assert args.sphere_spacing is None
    assert args.max_spheres_per_capsule is None
    assert args.capsule_radius_scale is None
    assert args.max_sphere_radius is None
    # 빈 dict 가 "예전과 같은 호출" 의 계약이다.
    assert serve_safe.sphere_options(args) == {}


def test_only_the_flags_that_were_given_reach_the_model():
    args = serve_safe.build_parser().parse_args(
        BASE_ARGV + ["--max-sphere-radius", "0.068"])
    assert serve_safe.sphere_options(args) == {"max_sphere_radius": 0.068}


def test_all_four_flags_reach_the_model():
    args = serve_safe.build_parser().parse_args(BASE_ARGV + [
        "--sphere-spacing", "0.4", "--max-spheres-per-capsule", "16",
        "--capsule-radius-scale", "0.95", "--max-sphere-radius", "0.07"])
    assert serve_safe.sphere_options(args) == {
        "sphere_spacing": 0.4, "max_spheres_per_capsule": 16,
        "capsule_radius_scale": 0.95, "max_sphere_radius": 0.07}


def test_the_option_names_are_the_ones_the_model_accepts():
    """이름이 틀리면 `TypeError` 로 서버가 죽는다 — 조용하지는 않지만, 여기서 잡으면 서버를
    띄우기 전이다."""
    args = serve_safe.build_parser().parse_args(BASE_ARGV + [
        "--sphere-spacing", "0.4", "--max-spheres-per-capsule", "16",
        "--capsule-radius-scale", "0.95", "--max-sphere-radius", "0.07"])
    accepted = set(inspect.signature(UrdfSphereChain.__init__).parameters)
    assert set(serve_safe.sphere_options(args)) <= accepted


def test_sphere_flags_with_no_safe_are_rejected():
    ap = serve_safe.build_parser()
    args = ap.parse_args(BASE_ARGV + ["--no-safe", "--max-sphere-radius", "0.068"])
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)


def test_the_options_only_touch_the_constraint_model():
    """`build_ag3s` 의 소스로 경계를 고정한다. 자기 필터가 이 dict 를 받게 되는 순간 그 link 의
    점이 필터를 통과해 장애물로 샌다 — `--exclude-links` 와 같은 사고다."""
    src = inspect.getsource(serve_safe.build_ag3s)
    assert "filter_robot = build_robot_model(scene)" in src
    assert src.count("sphere_options=spheres") == 2
    # 자기 필터를 만드는 호출은 인자가 **하나**다. 여기에 무엇이든 끼면 그때부터 필터 모델도
    # 가늘어질 수 있다.
    assert "build_robot_model(scene," not in src
