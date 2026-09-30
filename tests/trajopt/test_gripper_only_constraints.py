"""T7b — 그리퍼만 제약에 남기고, 자기 충돌을 끄고, 손가락만 얇게 한다.

셋 다 **진단용**이다. 이 설정에서 팔뚝·몸통·반대팔은 무엇에 부딪혀도 아무도 막지 않고, 쥔
물체가 로봇 자신을 통과해도 아무도 모른다. 그래서 이 파일이 지키는 것은 성능이 아니라
**되돌릴 수 있는가**와 **조용히 달라지지 않는가** 다:

* flag 를 하나도 안 주면 parser·모델·config 가 예전과 **글자 그대로** 같다.
* `--links gripper` 는 `ee_finger_l1/l2/r1/r2` **넷만** 남긴다. 이름을 손으로 적지 않는 것이
  요점이다 — `link_filter` 는 집합 교집합이라 오타 하나를 조용히 버린다.
* 자기 충돌을 끄면 그 블록의 행이 전부 상수 1 이 되고, 켜면 돌아온다. **행 수와 희소성은 양쪽
  에서 같다** — 그것이 "되돌릴 수 있다" 의 기계적 의미다.
* link 별 배율은 이름을 적은 link **만** 얇게 한다. 전역 배율로 손가락을 얇게 하면 굵기가
  필요한 팔뚝·반대팔이 같이 얇아진다.
* **꺼진 것은 시작 로그에 크게 나간다.** 조용히 다른 설정으로 떠 있는 것이 이 프로젝트가 여러
  번 밟은 함정이다.
* **자기 필터 모델은 어느 flag 로도 안 바뀐다.** 거기서 얇아지면 로봇 점이 장애물로 샌다.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import pathlib

import numpy as np
import pytest

from benchmark.ag3s.config import AG3SConfig, ConstraintConfig
from benchmark.ag3s.constraints.attached import self_collision_mask
from benchmark.ag3s.experiments.reports.grounding_report import (
    ARM_LINKS,
    GRIPPER_LINKS,
    build_robot_model,
)
from benchmark.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf
from benchmark.trajopt import serve_safe

#: 필수 인자만. 여기에 아무것도 더하지 않은 것이 "지금과 같은 서버" 다.
BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]

#: 손가락의 capsule 은 URDF 가 아니라 MJCF mesh 실측에서 온다 (`gap_filling_capsules`), 그래서
#: 씬 파일이 있어야 손가락 구가 생긴다. `test_gap_filling_capsules.py` 와 같은 경로·같은 skip.
MODEL = pathlib.Path(
    "/mnt/dev/work/pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml"
)


@pytest.fixture(scope="module")
def urdf():
    return parse_urdf(RBY1_URDF)


@pytest.fixture(scope="module")
def finger_capsules():
    """손가락 capsule 들. 없으면 이 파일의 굵기 절은 건너뛴다."""
    mujoco = pytest.importorskip("mujoco")
    if not MODEL.exists():  # pragma: no cover - asset-dependent
        pytest.skip(f"model not present: {MODEL}")
    from benchmark.ag3s.experiments.sources.mujoco_source import gap_filling_capsules

    return tuple(gap_filling_capsules(mujoco.MjModel.from_xml_path(str(MODEL))))


# --- flag 를 안 주면 지금과 같다 ------------------------------------------------------


def test_default_argv_keeps_every_new_switch_off():
    args = serve_safe.build_parser().parse_args(BASE_ARGV)
    # T27: `--links` 기본이 `arms` → `gripper` (사용자 판정 2026-09-28: "거리장 충돌제약은 gripper
    # 에만"). 나머지 스위치는 여전히 꺼져 있다.
    assert args.links == "gripper"
    assert args.no_self_collision is False
    assert tuple(args.capsule_radius_scale_link) == ()
    # 새 손잡이가 `sphere_options` 를 비어 있지 않게 만들면 `--no-safe` 검사가 오작동한다.
    assert serve_safe.sphere_options(args) == {}


def test_self_collision_is_on_by_default_everywhere():
    """스위치가 둘이면 언젠가 갈라지고, 갈라진 쪽이 안전 판정이면 그 결과는 조용하다."""
    assert ConstraintConfig().self_collision is True
    assert AG3SConfig().constraint.self_collision is True
    assert inspect.signature(serve_safe.build_ag3s).parameters["self_collision"].default is True
    assert inspect.signature(self_collision_mask).parameters["enabled"].default is True


def test_the_per_link_scale_defaults_to_nothing(urdf):
    params = inspect.signature(UrdfSphereChain.__init__).parameters
    assert params["capsule_radius_scale_by_link"].default is None
    # 빈 dict 를 명시해도 같은 모델이어야 한다 — 기본값이 두 곳에 안 적혔다는 뜻이다.
    explicit = UrdfSphereChain(urdf, link_filter=ARM_LINKS, capsule_radius_scale_by_link={})
    plain = UrdfSphereChain(urdf, link_filter=ARM_LINKS)
    assert np.array_equal(explicit.radii, plain.radii)
    assert explicit.n_spheres == plain.n_spheres


def test_the_default_links_choice_still_resolves_to_the_pinned_arm_set():
    """`arms` 는 옵션으로 남고 같은 집합을 가리킨다. **기본은 T27 부터 `gripper`** (첫 항목)."""
    assert serve_safe.constraint_link_filter("arms") == ARM_LINKS
    assert serve_safe.constraint_link_filter("all") is None
    assert "arms" in serve_safe.LINK_GROUPS
    assert serve_safe.LINK_GROUPS[0] == serve_safe.DEFAULT_LINKS == "gripper"
    assert serve_safe.constraint_link_filter(serve_safe.DEFAULT_LINKS) == GRIPPER_LINKS


# --- 손가락만 남는다 ----------------------------------------------------------------


def test_gripper_links_are_pinned_and_are_a_subset_of_the_arm_set():
    """이름 넷을 글자로 박는다. 여기가 `--links gripper` 의 정본이다."""
    # T8b: 손바닥이 들어왔다. `link_*_arm_5` 를 자르면 손바닥을 덮던 것이 사라지므로
    # (그 capsule 축이 z ∈ [−0.225, +0.025] 인데 손바닥 원점이 −0.1548 이다) 둘은 같이 와야 한다.
    assert GRIPPER_LINKS == ("ee_left", "ee_right",
                             "ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2")
    assert set(GRIPPER_LINKS) < set(ARM_LINKS)
    # `ARM_LINKS` 의 손끝은 `GRIPPER_LINKS` 에서 와야 한다 — 두 곳에 적히면 갈라진다.
    assert ARM_LINKS[-len(GRIPPER_LINKS):] == GRIPPER_LINKS


def test_the_gripper_choice_resolves_to_exactly_those_four_names():
    assert serve_safe.constraint_link_filter("gripper") == GRIPPER_LINKS
    assert "gripper" in serve_safe.LINK_GROUPS
    assert serve_safe.build_parser().parse_args(
        BASE_ARGV + ["--links", "gripper"]).links == "gripper"


def test_an_unknown_links_group_is_rejected_by_the_parser_and_the_resolver():
    with pytest.raises(SystemExit):
        serve_safe.build_parser().parse_args(BASE_ARGV + ["--links", "grippers"])
    with pytest.raises(ValueError, match="--links"):
        serve_safe.constraint_link_filter("grippers")


def test_a_gripper_only_model_carries_the_fingers_and_nothing_else(urdf, finger_capsules):
    arms = UrdfSphereChain(urdf, extra_capsules=finger_capsules, link_filter=ARM_LINKS)
    gripper = UrdfSphereChain(urdf, extra_capsules=finger_capsules, link_filter=GRIPPER_LINKS)

    assert set(gripper.sphere_link_names) == set(GRIPPER_LINKS)
    # 손가락 구는 하나도 잃지 않는다 — 빠진 것은 정확히 손가락이 아닌 구다.
    fingers_in_arms = sum(1 for n in arms.sphere_link_names if n in GRIPPER_LINKS)
    assert gripper.n_spheres == fingers_in_arms > 0
    assert gripper.n_spheres < arms.n_spheres
    # 남은 손가락 구의 굵기는 `--links` 때문에 달라지지 않는다.
    assert np.allclose(np.sort(gripper.radii),
                       np.sort([r for n, r in zip(arms.sphere_link_names, arms.radii)
                                if n in GRIPPER_LINKS]))


def test_build_ag3s_asks_the_resolver_rather_than_hardcoding_arm_links():
    """`--links` 의 해석이 한 곳에만 있어야 새 선택지가 조용히 무시되지 않는다."""
    src = inspect.getsource(serve_safe.build_ag3s)
    assert "link_filter=constraint_link_filter(links)" in src
    # 자기 필터는 예전처럼 인자 없이 전신이다 — 여기에 link 목록이 끼면 곧바로 유령 장애물이다.
    assert "filter_robot = build_robot_model(scene)" in src
    assert "sphere_options" not in inspect.signature(build_robot_model).parameters


def test_the_self_filter_model_still_carries_every_arm_link(urdf):
    """제약을 손가락으로 좁혀도 자기 필터는 전신이다.

    URDF 가 capsule 을 주는 것은 토르소와 arm 0~5 뿐이므로 (`link_*_arm_6` 이후는
    `gap_filling_capsules` 에서 온다) 여기서 지키는 것은 그 범위다.
    """
    whole_body = UrdfSphereChain(urdf, link_filter=None)
    from_urdf = {f"link_{side}_arm_{i}" for side in ("left", "right") for i in range(6)}
    assert from_urdf <= set(whole_body.sphere_link_names)
    assert from_urdf.isdisjoint(GRIPPER_LINKS)


# --- self-collision 을 끄면 그 행이 사라지고 켜면 돌아온다 ------------------------------


def test_disabling_self_collision_zeroes_every_pair():
    from benchmark.ag3s.stages.geometry import fit_sphere
    from benchmark.ag3s.types import AttachedCollisionGeometry

    names = ("link_left_arm_5", "ee_finger_l1", "ee_finger_l2", "link_right_arm_3")
    ball = fit_sphere(np.random.default_rng(0).normal(0.0, 0.01, (40, 3)))
    attached = AttachedCollisionGeometry(
        parent_link="ee_finger_l1", T_parent_object=np.eye(4), primitives=[ball],
        allowed_contact_links=frozenset({"ee_finger_l1"}))

    on = self_collision_mask(attached, names)
    off = self_collision_mask(attached, names, enabled=False)
    assert on.tolist() == [1.0, 0.0, 1.0, 1.0]
    assert off.tolist() == [0.0, 0.0, 0.0, 0.0]
    # 모양이 같아야 한다 — 같은 파라미터 칸에 다시 1 을 쓰는 것이 되돌리는 방법이다.
    assert off.shape == on.shape


def _attached_robot_rows(builder, spec, q) -> np.ndarray:
    """쥔 물체 대 **로봇 구** 행만 잘라 낸다 (step 0). `tests/ag3s/test_attached.py` 와 같은 계산."""
    from benchmark.ag3s.constraints.constraint_builder import evaluate

    values = evaluate(spec, np.tile(np.asarray(q, np.float64).reshape(-1, 1), (1, builder.horizon)),
                      builder)
    tail = values[builder.n_constraints - builder.n_attached_rows:]
    per_step = builder.n_attached_rows // builder.horizon
    m, k, s = builder.max_candidates, builder.max_support_surfaces, builder.n_robot_spheres
    width = m + k + s
    step0 = tail[:per_step]
    return np.concatenate([step0[slot * width + m + k: (slot + 1) * width]
                           for slot in range(builder.max_attached)])


def _engulfing_attachment(arm, fingertip, q):
    """팔의 구 여러 개를 삼킬 만큼 큰 물체. 손끝만 허용목록에 있다."""
    from benchmark.ag3s.stages.geometry import fit_sphere
    from benchmark.ag3s.types import AttachedCollisionGeometry

    tip = arm.end_effector_numeric(q)
    points = tip + np.random.default_rng(0).normal(0.0, 0.01, (40, 3))
    fat = dataclasses.replace(fit_sphere(points), center=np.zeros(3),
                              dimensions=np.full(3, 0.35))
    return AttachedCollisionGeometry(
        parent_link=fingertip, T_parent_object=np.eye(4), primitives=[fat],
        allowed_contact_links=frozenset({fingertip}))


def test_turning_self_collision_off_flattens_the_rows_and_on_brings_them_back():
    """끄면 그 블록이 전부 상수 1 이 되고, 켜면 위반이 돌아온다. **행 수는 양쪽에서 같다.**"""
    pytest.importorskip("casadi")
    from benchmark.ag3s.constraints.constraint_builder import ConstraintBuilder
    from tests.ag3s.fixtures import DEFAULT_ARM_Q, FINGERTIP_LINK, PlanarTwoLinkArm

    arm = PlanarTwoLinkArm()
    attached = _engulfing_attachment(arm, FINGERTIP_LINK, DEFAULT_ARM_Q)

    def rows(self_collision: bool):
        cfg = AG3SConfig().with_overrides({"constraint": {
            "horizon": 2, "max_attached_primitives": 2, "self_collision": self_collision}})
        builder = ConstraintBuilder(arm, cfg.constraint, cfg.geometry,
                                    attached_parent_links=(FINGERTIP_LINK,))
        spec = builder.build([], [], attached=attached)
        return builder, spec, _attached_robot_rows(builder, spec, DEFAULT_ARM_Q)

    on_builder, on_spec, on_rows = rows(True)
    off_builder, off_spec, off_rows = rows(False)

    assert (on_rows < 0.0).sum() > 0, "물체가 팔을 삼켰는데 위반 행이 하나도 없다"
    assert (off_rows < 0.0).sum() == 0, "자기 충돌을 껐는데 위반 행이 남았다"
    assert np.allclose(off_rows, 1.0), "꺼진 행은 상수 1 이어야 한다 (기울기 0)"
    # **되돌릴 수 있다는 것의 기계적 의미.** 행 수·파라미터 길이·경계가 양쪽에서 같다.
    assert off_builder.n_constraints == on_builder.n_constraints
    assert off_builder.n_attached_rows == on_builder.n_attached_rows
    assert off_spec.parameter_values.shape == on_spec.parameter_values.shape
    assert np.array_equal(off_spec.lower_bound, on_spec.lower_bound)
    # 마스크 블록만 달라진다.
    lo, hi = on_builder.layout["attached_self_mask"]
    assert off_spec.parameter_values[lo:hi].max() == 0.0
    assert on_spec.parameter_values[lo:hi].max() == 1.0


def test_the_builder_reports_the_config_value_to_the_mask():
    """마스크를 만드는 자리가 config 를 읽는다는 것을 소스로 고정한다.

    여기서 기본값(`True`)을 다시 적으면 스위치가 둘이 되고, 갈라진 쪽이 안전 판정이다.
    """
    from benchmark.ag3s.constraints import constraint_builder

    src = inspect.getsource(constraint_builder.ConstraintBuilder._pack_attached)
    assert "enabled=bool(self.constraint_config.self_collision)" in src


# --- link 별 배율은 적은 link 만 얇게 한다 ---------------------------------------------


def test_a_gripper_scale_thins_the_fingers_and_nothing_else(urdf, finger_capsules):
    def max_by_link(model):
        out: dict = {}
        for link, radius in zip(model.sphere_link_names, model.radii):
            out[link] = max(out.get(link, 0.0), float(radius))
        return out

    base = UrdfSphereChain(urdf, extra_capsules=finger_capsules, link_filter=ARM_LINKS)
    thin = UrdfSphereChain(urdf, extra_capsules=finger_capsules, link_filter=ARM_LINKS,
                           capsule_radius_scale_by_link={l: 0.35 for l in GRIPPER_LINKS})

    before, after = max_by_link(base), max_by_link(thin)
    for link in before:
        if link in GRIPPER_LINKS:
            assert after[link] < before[link], link
        else:
            assert after[link] == pytest.approx(before[link]), link
    # 얇게 하는 것은 굵기만이고 구 수도 행 수도 그대로다.
    assert thin.n_spheres == base.n_spheres


def test_a_global_scale_would_have_thinned_the_forearm_too(urdf, finger_capsules):
    """link 별 배율이 왜 필요한지를 테스트가 직접 보인다 — 전역 배율의 부작용."""
    global_scale = UrdfSphereChain(urdf, extra_capsules=finger_capsules, link_filter=ARM_LINKS,
                                   capsule_radius_scale=0.35)
    base = UrdfSphereChain(urdf, extra_capsules=finger_capsules, link_filter=ARM_LINKS)
    forearm = [r for n, r in zip(base.sphere_link_names, base.radii) if n == "link_left_arm_5"]
    thinned = [r for n, r in zip(global_scale.sphere_link_names, global_scale.radii)
               if n == "link_left_arm_5"]
    assert max(thinned) < max(forearm)


def test_the_per_link_scale_overrides_the_global_one_only_where_named(urdf, finger_capsules):
    model = UrdfSphereChain(urdf, extra_capsules=finger_capsules, link_filter=ARM_LINKS,
                            capsule_radius_scale=0.8,
                            capsule_radius_scale_by_link={"ee_finger_l1": 0.35})
    assert model.scale_for_link("ee_finger_l1") == pytest.approx(0.35)
    assert model.scale_for_link("ee_finger_l2") == pytest.approx(0.8)
    assert model.scale_for_link("link_left_arm_5") == pytest.approx(0.8)


def test_the_shortfall_and_the_report_name_the_per_link_scale(urdf, finger_capsules):
    """얇아진 사실은 **기록에도 로그에도** 남아야 한다."""
    model = UrdfSphereChain(urdf, extra_capsules=finger_capsules, link_filter=ARM_LINKS,
                            capsule_radius_scale_by_link={l: 0.35 for l in GRIPPER_LINKS})
    assert model.coverage_shortfall, "손가락을 얇게 했는데 덮개 손실이 기록되지 않았다"
    assert {s.link for s in model.coverage_shortfall} == set(GRIPPER_LINKS)
    report = model.coverage_report()
    assert "DOES NOT COVER" in report
    assert "per-link capsule_radius_scale" in report
    assert "ee_finger_l1=0.35" in report


def test_a_typo_in_the_per_link_scale_stops_everything(urdf):
    with pytest.raises(ValueError) as err:
        UrdfSphereChain(urdf, link_filter=ARM_LINKS,
                        capsule_radius_scale_by_link={"ee_finger_l11": 0.35})
    assert "ee_finger_l11" in str(err.value)


def test_scaling_a_link_this_model_has_no_capsule_for_is_rejected(urdf):
    """URDF 에는 있지만 이 모델이 구를 안 가진 link — 조용히 넘기면 얇게 했다고 믿는다."""
    present = {c.link for c in UrdfSphereChain(urdf, link_filter=ARM_LINKS).capsules}
    absent = next(link for link in urdf.links if link not in present)
    with pytest.raises(ValueError) as err:
        UrdfSphereChain(urdf, link_filter=ARM_LINKS,
                        capsule_radius_scale_by_link={absent: 0.35})
    assert absent in str(err.value)


def test_a_non_positive_per_link_scale_is_rejected(urdf):
    with pytest.raises(ValueError, match="> 0"):
        UrdfSphereChain(urdf, link_filter=ARM_LINKS,
                        capsule_radius_scale_by_link={"link_left_arm_5": 0.0})


# --- CLI 파싱: 집합 이름과 오타 -------------------------------------------------------


def test_the_group_name_expands_to_the_four_finger_links():
    assert serve_safe.parse_link_scales(["gripper=0.35"]) == {
        link: 0.35 for link in GRIPPER_LINKS}
    assert serve_safe.parse_link_scales(()) == {}


def test_a_bare_link_name_and_a_group_can_be_mixed():
    parsed = serve_safe.parse_link_scales(["gripper=0.35", "link_left_arm_5=0.8"])
    assert parsed["link_left_arm_5"] == pytest.approx(0.8)
    assert parsed["ee_finger_r2"] == pytest.approx(0.35)


@pytest.mark.parametrize("bad", ["gripper", "gripper=", "gripper=abc", "gripper=0",
                                 "gripper=-0.5", "all=0.5", "=0.5"])
def test_a_malformed_per_link_scale_is_rejected_before_the_server_starts(bad):
    ap = serve_safe.build_parser()
    args = ap.parse_args(BASE_ARGV + ["--capsule-radius-scale-link", bad])
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)


def test_the_diagnostic_flags_are_refused_with_no_safe():
    ap = serve_safe.build_parser()
    for extra in (["--no-self-collision"], ["--capsule-radius-scale-link", "gripper=0.35"]):
        args = ap.parse_args(BASE_ARGV + ["--no-safe", *extra])
        with pytest.raises(SystemExit):
            serve_safe.reject_bad_flag_combinations(ap, args)


def test_the_diagnostic_flags_pass_the_check_on_a_plain_run():
    ap = serve_safe.build_parser()
    args = ap.parse_args(BASE_ARGV + ["--links", "gripper", "--no-self-collision",
                                      "--capsule-radius-scale-link", "gripper=0.35"])
    serve_safe.reject_bad_flag_combinations(ap, args)  # 죽지 않는다
    assert serve_safe.sphere_options(args) == {
        "capsule_radius_scale_by_link": {link: 0.35 for link in GRIPPER_LINKS}}


# --- 시작 로그에 크게 찍힌다 ----------------------------------------------------------


def test_a_default_run_says_nothing_new(caplog):
    # T27: 기본 범위가 `gripper` 다. 기본 실행은 warning 을 내지 않는다 (범위는 info 한 줄).
    with caplog.at_level(logging.WARNING, logger=serve_safe.__name__):
        serve_safe.announce_diagnostic_scope(
            links="gripper", self_collision=True, link_scales={},
            constraint_links=GRIPPER_LINKS, filter_spheres=61)
    assert caplog.text == ""


def test_the_default_scope_is_still_named_in_the_startup_log(caplog):
    """조용하다고 말이 없는 것은 아니다 — 어느 link 에 걸렸고 무엇이 빠졌는지 info 로 남는다."""
    with caplog.at_level(logging.INFO, logger=serve_safe.__name__):
        serve_safe.announce_diagnostic_scope(
            links="gripper", self_collision=True, link_scales={},
            constraint_links=GRIPPER_LINKS, filter_spheres=61)
    assert "ee_finger_l1" in caplog.text and "아무도 막지 않습니다" in caplog.text


@pytest.mark.parametrize("links", ["arms", "all"])
def test_the_startup_log_shouts_about_a_non_default_scope(caplog, links):
    """T27 에서 **방향이 뒤집혔다.** 예전에는 `gripper` 가 비기본이라 크게 말했고, 이제는 `arms` ·
    `all` 이 비기본이다 — 팔뚝·손목(·몸통)까지 제약이 걸려 파지 접근이 그 행에 밀릴 수 있다."""
    with caplog.at_level(logging.WARNING, logger=serve_safe.__name__):
        serve_safe.announce_diagnostic_scope(
            links=links, self_collision=True, link_scales={},
            constraint_links=ARM_LINKS, filter_spheres=61)
    text = caplog.text
    assert f"CONSTRAINT SCOPE IS {links.upper()}" in text
    assert "NOT THE DEFAULT 'gripper'" in text
    assert "link_left_arm_5" in text or "(전신)" in text
    assert "gripper" in text


def test_the_startup_log_shouts_when_self_collision_is_off(caplog):
    with caplog.at_level(logging.WARNING, logger=serve_safe.__name__):
        serve_safe.announce_diagnostic_scope(
            links="arms", self_collision=False, link_scales={}, constraint_links=ARM_LINKS)
    assert "SELF-COLLISION IS OFF" in caplog.text


def test_the_startup_log_shouts_about_a_per_link_scale(caplog):
    with caplog.at_level(logging.WARNING, logger=serve_safe.__name__):
        serve_safe.announce_diagnostic_scope(
            links="arms", self_collision=True,
            link_scales={link: 0.35 for link in GRIPPER_LINKS},
            constraint_links=ARM_LINKS, filter_spheres=61)
    assert "PER-LINK CAPSULE RADIUS SCALE" in caplog.text
    assert "ee_finger_l1=0.35" in caplog.text


def test_build_ag3s_makes_that_announcement():
    src = inspect.getsource(serve_safe.build_ag3s)
    assert "announce_diagnostic_scope(" in src
    assert "self_collision=self_collision" in src


def test_the_constraint_builder_shouts_when_self_collision_is_off(caplog):
    """`serve_safe` 를 거치지 않는 호출자(실험 스크립트·테스트)도 반드시 듣는다."""
    pytest.importorskip("casadi")
    from benchmark.ag3s.constraints import constraint_builder
    from tests.ag3s.fixtures import FINGERTIP_LINK, PlanarTwoLinkArm

    cfg = AG3SConfig().with_overrides(
        {"constraint": {"horizon": 2, "max_attached_primitives": 1, "self_collision": False}})
    with caplog.at_level(logging.WARNING, logger=constraint_builder.__name__):
        constraint_builder.ConstraintBuilder(
            PlanarTwoLinkArm(), cfg.constraint, cfg.geometry,
            attached_parent_links=(FINGERTIP_LINK,))
    assert "SELF-COLLISION IS OFF" in caplog.text


def test_the_constraint_builder_is_quiet_when_self_collision_is_on(caplog):
    pytest.importorskip("casadi")
    from benchmark.ag3s.constraints import constraint_builder
    from tests.ag3s.fixtures import FINGERTIP_LINK, PlanarTwoLinkArm

    cfg = AG3SConfig().with_overrides(
        {"constraint": {"horizon": 2, "max_attached_primitives": 1}})
    with caplog.at_level(logging.WARNING, logger=constraint_builder.__name__):
        constraint_builder.ConstraintBuilder(
            PlanarTwoLinkArm(), cfg.constraint, cfg.geometry,
            attached_parent_links=(FINGERTIP_LINK,))
    assert "SELF-COLLISION" not in caplog.text
