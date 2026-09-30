"""`serve_safe --exclude-links` — 제약 모델에서만 link 를 뺀다 (T6e).

이 flag 는 **진단용**이다. 최악 위반 link 가 늘 팔뚝(`link_*_arm_5`)이라는 관찰이 팔 자세
때문인지를 가르기 위해 그 둘을 제약에서 빼고 한 번 돌려 보는 것이고, 빼는 동안 그 팔뚝이
무엇에 부딪혀도 아무도 막지 않는다.

그래서 이 테스트가 지키는 것은 성능이 아니라 **무엇이 조용히 달라지지 않는가** 다:

* flag 를 안 주면 제약 link 집합이 예전과 **글자 그대로** 같다 (`ARM_LINKS` 18 개).
* 모델에 없는 이름을 주면 **서버가 시작하지 않는다.** `link_filter` 는 집합 교집합이라 모르는
  이름을 조용히 버리므로(`urdf_sphere_chain.py:389`), 오타 하나로 아무것도 안 빠진 채 "뺐다"
  고 믿는 것이 이 flag 의 가장 위험한 실패다.
* **자기 필터 모델은 안 바뀐다.** 거기서 빼면 그 link 의 점이 필터를 통과해 장애물로 샌다 —
  제약을 푸는 것과 반대 방향의 사고다.
"""

from __future__ import annotations

import inspect

import pytest

from benchmark.ag3s.experiments.reports.grounding_report import ARM_LINKS
from benchmark.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf
from benchmark.trajopt import serve_safe

#: 사용자가 빼 보고 싶어 한 두 link (T6e) — 양팔 팔뚝.
FOREARMS = ("link_left_arm_5", "link_right_arm_5")

#: 필수 인자만. 여기에 아무것도 더하지 않은 것이 "지금과 같은 서버" 다.
BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


@pytest.fixture(scope="module")
def urdf():
    return parse_urdf(RBY1_URDF)


@pytest.fixture(scope="module")
def constraint_model(urdf):
    """서버의 제약 모델과 같은 link 범위. gap-filling capsule 은 MuJoCo 씬이 필요해 여기서는
    빼지만, link 이름 집합과 필터 동작은 그것과 무관하다."""
    return UrdfSphereChain(urdf, link_filter=ARM_LINKS)


# --- 기본 동작이 바뀌지 않는다 -----------------------------------------------------


def test_default_argv_excludes_nothing():
    args = serve_safe.build_parser().parse_args(BASE_ARGV)
    assert args.exclude_links == ()
    # T27: 기본 범위가 `arms` → `gripper` (사용자 판정 2026-09-28). 이 테스트의 요점(아무것도 안
    # 뺀다)은 그대로다.
    assert args.links == "gripper"


def test_build_ag3s_defaults_to_no_exclusion():
    assert inspect.signature(serve_safe.build_ag3s).parameters["exclude_links"].default == ()


def test_the_default_constraint_link_set_is_pinned():
    """`ARM_LINKS` 자체를 글자로 박는다. flag 를 더하면서 기본 집합이 흔들리면 여기서 걸린다.

    **2026-09-26 (T8b): 손바닥 `ee_left`/`ee_right` 가 들어왔다.** 그 전까지 손바닥은 제약
    모델에 없었고, `link_*_arm_5` 의 URDF capsule (r 75 mm · L 250 mm, 축이 link frame 에서
    z ∈ [−0.225, +0.025]) 이 손목·손바닥·그리퍼를 통째로 삼켜 티가 나지 않았다. 그 capsule 을
    `--capsule-extent-link` 로 자르는 순간 손바닥이 제약에서 사라지므로 둘은 같이 와야 한다.
    geometry 는 이미 있었다 — self-filter 는 2026-09-25 부터 같은 gap-filling capsule 을 쓴다.
    """
    assert ARM_LINKS == (
        "link_left_arm_0", "link_left_arm_1", "link_left_arm_2", "link_left_arm_3",
        "link_left_arm_4", "link_left_arm_5", "link_left_arm_6",
        "link_right_arm_0", "link_right_arm_1", "link_right_arm_2", "link_right_arm_3",
        "link_right_arm_4", "link_right_arm_5", "link_right_arm_6",
        "ee_left", "ee_right",
        "ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2")


def test_no_exclusion_keeps_every_present_link(constraint_model, urdf):
    kept = serve_safe.constraint_links_minus(
        constraint_model.sphere_link_names, (), known=urdf.links)
    assert set(kept) == set(constraint_model.sphere_link_names)


# --- 준 이름이 실제로 빠진다 -------------------------------------------------------


def test_excluding_the_forearms_drops_their_spheres_and_nothing_else(urdf, constraint_model):
    kept = serve_safe.constraint_links_minus(
        constraint_model.sphere_link_names, FOREARMS, known=urdf.links)
    assert set(kept) == set(constraint_model.sphere_link_names) - set(FOREARMS)

    reduced = UrdfSphereChain(urdf, link_filter=kept)
    assert set(reduced.sphere_link_names).isdisjoint(FOREARMS)
    # 남은 link 의 구는 하나도 안 잃는다 — 빠진 수는 정확히 팔뚝의 구 수다.
    dropped = sum(1 for link in constraint_model.sphere_link_names if link in FOREARMS)
    assert dropped > 0
    assert reduced.n_spheres == constraint_model.n_spheres - dropped


def test_duplicate_names_are_harmless(constraint_model, urdf):
    once = serve_safe.constraint_links_minus(
        constraint_model.sphere_link_names, FOREARMS, known=urdf.links)
    twice = serve_safe.constraint_links_minus(
        constraint_model.sphere_link_names, FOREARMS + FOREARMS[:1], known=urdf.links)
    assert once == twice


# --- 없는 이름이면 크게 말한다 -----------------------------------------------------


def test_a_typo_stops_everything(constraint_model, urdf):
    typo = "link_left_arm_55"
    with pytest.raises(ValueError) as err:
        serve_safe.constraint_links_minus(
            constraint_model.sphere_link_names, (typo,), known=urdf.links)
    assert typo in str(err.value)
    assert "URDF 에 없는 이름" in str(err.value)


def test_a_link_that_has_no_constraint_spheres_is_also_rejected(constraint_model, urdf):
    """URDF 에는 있지만 제약 모델에 구가 없는 link — 빼도 아무 일이 안 일어난다.

    조용히 통과시키면 "토르소를 뺐다" 고 믿은 채 제약이 그대로인 실행이 남는다.
    """
    present = set(constraint_model.sphere_link_names)
    sphereless = next(link for link in urdf.links if link not in present)
    with pytest.raises(ValueError) as err:
        serve_safe.constraint_links_minus(
            constraint_model.sphere_link_names, (sphereless,), known=urdf.links)
    assert sphereless in str(err.value)
    assert "구가 없는" in str(err.value)


def test_excluding_every_link_is_rejected(constraint_model, urdf):
    with pytest.raises(ValueError, match="전부"):
        serve_safe.constraint_links_minus(
            constraint_model.sphere_link_names, constraint_model.sphere_link_names,
            known=urdf.links)


# --- 자기 필터 모델은 그대로다 -----------------------------------------------------


def test_the_self_filter_model_still_carries_the_excluded_links(urdf):
    """전신 모델(자기 필터가 쓰는 것)에는 팔뚝이 그대로 있어야 한다."""
    whole_body = UrdfSphereChain(urdf, link_filter=None)
    assert set(FOREARMS) <= set(whole_body.sphere_link_names)


def test_exclusion_is_applied_to_the_constraint_model_only():
    """`build_ag3s` 가 자기 필터를 만드는 호출에는 link 인자가 없다는 것을 소스로 고정한다.

    두 모델이 한 변수를 공유하게 되는 순간 이 flag 는 안전 사고가 된다 — 그 경계는 주석이
    아니라 테스트가 지켜야 한다.
    """
    src = inspect.getsource(serve_safe.build_ag3s)
    # 자기 필터는 인자 없이 전신으로 만든다 — 여기에 link 목록이 끼면 곧바로 유령 장애물이다.
    assert "filter_robot = build_robot_model(scene)" in src
    # 제외는 제약 모델을 다시 만드는 자리에서만 일어난다.
    assert src.count("build_constraint_robot_model(") == 2
    assert src.count("build_robot_model(scene)") == 1


# --- 안전 계층을 끈 채로는 못 준다 --------------------------------------------------


def test_exclude_links_with_no_safe_is_rejected():
    ap = serve_safe.build_parser()
    args = ap.parse_args(BASE_ARGV + ["--no-safe", "--exclude-links", FOREARMS[0]])
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)


def test_shadow_with_no_safe_is_still_rejected():
    """옛 검사가 새 함수로 옮겨 오면서 살아 있는지 — 옮기다 잃는 것이 흔하다."""
    ap = serve_safe.build_parser()
    args = ap.parse_args(BASE_ARGV + ["--no-safe", "--shadow"])
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)


def test_a_plain_run_passes_the_flag_check():
    ap = serve_safe.build_parser()
    args = ap.parse_args(BASE_ARGV + ["--exclude-links", *FOREARMS])
    serve_safe.reject_bad_flag_combinations(ap, args)  # 죽지 않는다
    assert list(args.exclude_links) == list(FOREARMS)
