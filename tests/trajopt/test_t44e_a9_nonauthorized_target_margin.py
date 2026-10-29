"""T44e A9 — 접촉 권한이 없는 link 의 target 여유 = 일반 obstacle 행의 여유 (`contact.nonauthorized_target_margin`).

`T44dA.verify.json` Q2: `ClearancePolicy.margin_matrix` 는 권한 없는 link 에 `full_margin(TARGET)` (= `safety_margin`
50 mm) 을 주고, `_esdf_clearance_core` 는 그 행의 최근접 표면이 target 이면 (`|d − d_object| ≤ voxel`) 그 값을 쓴다.
`--links arms` 에서 `link_*_arm_5` 가 153 chunk 에서 최악 행이었다. obstacle 행은 `esdf_margin` (V11AB 0) 에
`obstacle_margin > 0` 이면 `max(·, obstacle_margin)` (V11AB 10 mm) 를 받는다.

`"obstacle"`: AG3S 가 권한 없는 행 (`authorized_links` 밖) 을 `CollisionConstraintSet.nonauthorized_target_rows` 로 싣고,
trajopt 는 그 행을 margin 의 target 판정 두 곳 (`is_object`, `_obstacle_rows` 의 target class) 에서 뺀다 → 그 행의 여유가
obstacle 행과 **같은 식** 에서 나온다 (이중 적용 · 누락 없음).

Rig: `tests/trajopt/test_t44_target_permission.py` (z = 0 테이블, 사과 (target), 7 mm 옆 distractor, 막대). 행 0 손가락
(권한) 과 행 1 `link_right_arm_5` (권한 없음) 가 같은 점 (사과 1 mm 안) 에 있다.

| # | 성질 |
|---|---|
| 1 | 기본 `full`: 필드 · 스냅샷 · 값 · 분류 비트 동일 |
| 2 | `obstacle`: 권한 없는 행의 여유 = 같은 점의 obstacle 행 여유 (`obstacle_margin` 0 / 0.010, strict on/off, relax / exclude_authorized) |
| 3 | 권한 행 (phase 여유) · non-target 행 · 쥔 뒤 (carry) 불변 · look-ahead / sweep 은 strict 그대로 |
| 4 | verdict: 권한 없는 행의 위반은 여전히 `unauthorized_link` (접촉 금지) |
| 5 | AG3S: config 왕복 · validation · `build_constraint_set` 이 `authorized_links` 밖 행을 싣는다 (gripper 모델 = 반대손, arms 모델 = 팔 link 포함) · 쥔 뒤 / full 은 None · record |
| 6 | CLI (`--nonauthorized-target-margin`) · `build_ag3s(contact=)` dict 배선 |
"""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

P = pytest.importorskip("tests.trajopt.test_t44_target_permission")

from AG3S.trajopt.linearize import (  # noqa: E402
    OBSTACLE_CLASSES,
    CollisionLinearizer,
    SceneSnapshot,
    scene_from_constraint_set,
)

FINGER, FOREARM, BARROW, DISTROW, HELD = P.FINGER, P.FOREARM, P.BARROW, P.DISTROW, P.HELD
M, R_Q, FULL, APPROACH = P.M, P.R_Q, P.FULL, P.APPROACH
NONAUTH = ~P.CONTACT                                   # the rig's permission: row 1 only
OBSTACLE = OBSTACLE_CLASSES.index("obstacle")
TARGET = OBSTACLE_CLASSES.index("target")


def _clear(scene, steps=1, lookahead=None):
    lin = P._lin()
    lin.lookahead_from = lookahead
    out = lin._esdf_clearance(P._centres(steps), scene)[..., 0]
    return out, (None if getattr(lin, "_last_obstacle_class", None) is None
                 else lin._last_obstacle_class.copy()), lin


def _obstacle_reference(scene):
    """The same points with **no** manipulated object: every row's margin is the obstacle rule."""
    return dataclasses.replace(scene, manipulated_points=None, manipulated_link_margin=None,
                               target_contact_rows=None, nonauthorized_target_rows=None)


# =============================================================================================
# 1. default = bit-identical
# =============================================================================================


def test_defaults():
    assert SceneSnapshot.__dataclass_fields__["nonauthorized_target_rows"].default is None
    assert CollisionLinearizer._nonauthorized_target_rows(P._scene(), 5) is None
    on = CollisionLinearizer._nonauthorized_target_rows(
        P._scene(nonauthorized_target_rows=NONAUTH), 5)
    assert on.tolist() == [False, True, False, False, False]          # held padded False
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.types import CollisionConstraintSet

    assert AG3SConfig().contact.nonauthorized_target_margin == "full"
    assert CollisionConstraintSet.__dataclass_fields__["nonauthorized_target_rows"].default is None


@pytest.mark.parametrize("om", [0.0, M], ids=["no_obstacle_margin", "obstacle_margin"])
@pytest.mark.parametrize("strict", [False, True])
def test_scene_without_the_vector_is_unchanged(om, strict):
    field = P._Field()
    field.stats = {"target_free": {"ball": P._BALL_STATS}}
    radii = np.full(4, R_Q)
    cs = P._cs(field)
    cfg = P._config(obstacle_margin=om, target_permission_strict=strict)
    a = scene_from_constraint_set(cs, radii, cfg)
    cs.nonauthorized_target_rows = None                                 # `full` sends None
    b = scene_from_constraint_set(cs, radii, cfg)
    assert a.nonauthorized_target_rows is None and b.nonauthorized_target_rows is None
    for lookahead in (None, 1):
        ca, cla, la = _clear(a, 3, lookahead)
        cb, clb, lb = _clear(b, 3, lookahead)
        assert np.array_equal(ca, cb)
        assert (cla is None and clb is None) or np.array_equal(cla, clb)
        assert getattr(la, "_last_lookahead", None) == getattr(lb, "_last_lookahead", None)


# =============================================================================================
# 2. obstacle: unauthorized target row = an obstacle row at the same point
# =============================================================================================


@pytest.mark.parametrize("om", [0.0, M], ids=["no_obstacle_margin", "obstacle_margin_10mm"])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("mask", [None, P.CONTACT], ids=["relax", "exclude_authorized"])
def test_unauthorized_row_gets_the_obstacle_clearance(om, strict, mask):
    base = P._scene(mask=mask, strict=strict, obstacle_margin=om)
    off, _, _ = _clear(base)
    on, cls_on, _ = _clear(dataclasses.replace(base, nonauthorized_target_rows=NONAUTH))
    ref, _, _ = _clear(_obstacle_reference(base))
    d = -0.001                                                          # 1 mm inside the apple
    # off: `is_object` gives the forearm 50 mm (strict only changes its class, not that margin)
    assert off[0, FOREARM] == pytest.approx(d - R_Q - FULL, abs=1e-9)
    assert on[0, FOREARM] == ref[0, FOREARM]
    assert on[0, FOREARM] == pytest.approx(d - R_Q - max(0.0, om), abs=1e-9)
    if om > 0.0:
        assert cls_on[0, FOREARM] == OBSTACLE
    # every other row is untouched (the finger keeps its phase margin / target-free answer)
    keep = [i for i in range(on.shape[1]) if i != FOREARM]
    assert np.array_equal(on[:, keep], off[:, keep])


def test_v11ab_numbers():
    """V11AB: `esdf_margin` 0 + `obstacle_margin` 0.010 → the arm row next to the target asks 10 mm, not 50."""
    field = P._Field()
    field.stats = {"target_free": {"ball": P._BALL_STATS}}
    radii = np.full(4, R_Q)
    cs = P._cs(field, policy="exclude_authorized", mask=P.CONTACT)
    cs.nonauthorized_target_rows = NONAUTH
    cfg = P._config(obstacle_margin=0.010, target_permission_strict=True)
    s = scene_from_constraint_set(cs, radii, cfg)
    assert s.nonauthorized_target_rows.tolist() == NONAUTH.tolist()
    on, cls, _ = _clear(s)
    cs.nonauthorized_target_rows = None
    off, _, _ = _clear(scene_from_constraint_set(cs, radii, cfg))
    assert (on[0, FOREARM] - off[0, FOREARM]) == pytest.approx(FULL - 0.010, abs=1e-12)
    assert cls[0, FOREARM] == OBSTACLE


# =============================================================================================
# 3. what does not move
# =============================================================================================


def test_authorized_and_non_target_rows_and_classes_do_not_move():
    base = P._scene(obstacle_margin=M)
    off, cls_off, _ = _clear(base)
    on, cls_on, _ = _clear(dataclasses.replace(base, nonauthorized_target_rows=NONAUTH))
    for row in (FINGER, BARROW, DISTROW, HELD):
        assert on[0, row] == off[0, row] and cls_on[0, row] == cls_off[0, row]
    assert cls_on[0, FINGER] == TARGET


def test_carry_is_not_touched():
    field = P._Field()
    radii = np.full(4, R_Q)
    holding = P._cs(field, attached=types.SimpleNamespace(points=np.zeros((1, 3)),
                                                          parent_link="ee_right", primitives=[]))
    holding.nonauthorized_target_rows = NONAUTH                           # even if one were sent
    s = scene_from_constraint_set(holding, radii, P._config(obstacle_margin=M))
    assert s.nonauthorized_target_rows is None


def test_shape_mismatch_is_an_error():
    cs = P._cs(P._Field())
    cs.nonauthorized_target_rows = np.ones(7, bool)
    with pytest.raises(ValueError, match="nonauthorized_target_rows"):
        scene_from_constraint_set(cs, np.full(4, R_Q), P._config())


@pytest.mark.parametrize("strict", [False, True])
def test_lookahead_tail_and_sweep_follow_strict_only(strict):
    centres = P._centres(2)
    base = P._scene(strict=strict)
    on_scene = dataclasses.replace(base, nonauthorized_target_rows=NONAUTH)
    a, b = P._lin(), P._lin()
    a.lookahead_from = b.lookahead_from = 1
    ca = a._esdf_clearance(centres, base)[1, :, 0]
    cb = b._esdf_clearance(centres, on_scene)[1, :, 0]
    assert np.isinf(ca[FOREARM]) == np.isinf(cb[FOREARM])                 # same tail rows
    assert a._last_lookahead == b._last_lookahead
    ra, rb = P._checker(P._lin(), centres), P._checker(P._lin(), centres)
    ta, tb = ra._target_rows(centres, base), rb._target_rows(centres, on_scene)
    assert (ta is None and tb is None) or np.array_equal(ta, tb)


# =============================================================================================
# 4. verdict — contact stays forbidden
# =============================================================================================


def test_a_violating_unauthorized_row_is_still_unauthorized_link():
    from AG3S.trajopt.safe_policy import _classify_esdf_rows

    s = dataclasses.replace(P._scene(obstacle_margin=M, strict=True),
                            nonauthorized_target_rows=NONAUTH)
    values, _, _ = _clear(s)
    assert values[0, FOREARM] < 0.0                                       # 1 mm inside the apple
    hits = np.argwhere(values < -1e-6)
    names = P.NAMES + ["held"]
    rows = _classify_esdf_rows(
        hits, values, P._centres(1), np.array([R_Q] * 4 + [0.03]), s, lambda q: names[q], 4,
        frozenset(n for n, c in zip(P.NAMES, P.CONTACT) if c), {"centroid": None, "radius": 0.0},
        pad_m=0.01, target_label="target", allow_unresolved_tier=False)
    fore = [r for r in rows if r["query"] == FOREARM]
    assert fore and all(r["class"] == "collision" and r["why"] == "unauthorized_link" for r in fore)


# =============================================================================================
# 5. AG3S side
# =============================================================================================


def test_config_round_trip_and_validation():
    from AG3S.ag3s.config import AG3SConfig, AG3SConfigError

    cfg = AG3SConfig.from_dict({"contact": {"nonauthorized_target_margin": "obstacle"}})
    assert cfg.contact.nonauthorized_target_margin == "obstacle"
    assert AG3SConfig.from_dict(cfg.to_dict()).contact.nonauthorized_target_margin == "obstacle"
    assert cfg.contact.NONAUTHORIZED_TARGET_MARGINS == ("full", "obstacle")
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"contact": {"nonauthorized_target_margin": "zero"}})


GRIPPER = ["ee_left", "ee_finger_l1", "ee_finger_l2", "ee_right", "ee_finger_r1", "ee_finger_r2"]
ARMS = GRIPPER + ["link_right_arm_5", "link_right_arm_6", "link_left_arm_5"]


def _adapter_builder(names, mode):
    from AG3S.ag3s.config import ConstraintConfig, ContactConfig
    from AG3S.ag3s.constraints.clearance import ClearancePolicy

    policy = ClearancePolicy(contact=ContactConfig(nonauthorized_target_margin=mode))
    return types.SimpleNamespace(
        build=None, max_candidates=32, max_support_surfaces=2,
        constraint_config=ConstraintConfig(target_field_policy="exclude_authorized"),
        robot_model=None, n_robot_spheres=len(names), clearance_policy=policy,
        sphere_link_names=list(names))


def _target():
    from AG3S.ag3s.stages.geometry import fit_sphere
    from AG3S.ag3s.types import TargetGeometry

    pts = np.random.default_rng(0).normal([0.5, 0.0, 0.85], 0.01, (40, 3))
    return TargetGeometry(id=1, points=pts, point_indices=np.arange(40), centroid=pts.mean(0),
                          bounding_geometry=fit_sphere(pts), attention_score=0.9, confidence=0.9)


@pytest.mark.parametrize("names,expected", [
    (GRIPPER, {"ee_left", "ee_finger_l1", "ee_finger_l2"}),
    (ARMS, {"ee_left", "ee_finger_l1", "ee_finger_l2", "link_right_arm_5", "link_right_arm_6",
            "link_left_arm_5"}),
], ids=["links_gripper", "links_arms"])
def test_build_constraint_set_sends_the_unauthorized_rows(names, expected):
    from AG3S.ag3s.constraints.to_adapter import build_constraint_set
    from AG3S.ag3s.types import ContactPolicyContext

    ctx = ContactPolicyContext.make("approach", ["right"])
    kw = dict(candidates=[], support_surfaces=[], target=_target(), robot_state=np.zeros(2),
              phase="approach", contact_context=ctx, build_spec=False)
    on = build_constraint_set(_adapter_builder(names, "obstacle"), **kw)
    rows = on.nonauthorized_target_rows
    assert {n for n, r in zip(names, rows) if r} == expected
    rec = on.metrics["target_permission"]["nonauthorized_target_margin"]
    assert rec == {"mode": "obstacle", "n_rows": len(expected), "applied": True}
    # `manipulated_link_margin` (the permission `target_contact_rows_of` reads) is unchanged
    off = build_constraint_set(_adapter_builder(names, "full"), **kw)
    assert np.array_equal(on.manipulated_link_margin, off.manipulated_link_margin)
    assert off.nonauthorized_target_rows is None
    assert "nonauthorized_target_margin" not in off.metrics["target_permission"]
    # nothing manipulated → None (recorded as not applied)
    none = build_constraint_set(_adapter_builder(names, "obstacle"),
                                **{**kw, "target": None, "manipulated_geometry": None})
    assert none.nonauthorized_target_rows is None
    assert none.metrics["target_permission"]["nonauthorized_target_margin"]["applied"] is False


# =============================================================================================
# 6. CLI
# =============================================================================================


def test_cli_section():
    from AG3S.trajopt import serve_safe

    p = serve_safe.build_parser()
    assert serve_safe.nonauthorized_target_margin_section(p.parse_args(P.BASE_ARGV)) == {}
    args = p.parse_args(P.BASE_ARGV + ["--nonauthorized-target-margin", "obstacle"])
    sec = serve_safe.nonauthorized_target_margin_section(args)
    assert sec == {"nonauthorized_target_margin": "obstacle"}
    from AG3S.ag3s.config import AG3SConfig

    assert AG3SConfig.from_dict({"contact": sec}).contact.nonauthorized_target_margin == "obstacle"
    with pytest.raises(SystemExit):
        p.parse_args(P.BASE_ARGV + ["--nonauthorized-target-margin", "zero"])


def test_build_ag3s_takes_a_contact_section():
    import inspect

    from AG3S.trajopt import serve_safe

    sig = inspect.signature(serve_safe.build_ag3s)
    assert sig.parameters["contact"].default is None
    src = inspect.getsource(serve_safe.ag3s_from_args)   # `main` builds AG3S through it (AG3S-DYNAMIC-FIELD)
    assert "contact=nonauthorized_target_margin_section(args) or None" in src
