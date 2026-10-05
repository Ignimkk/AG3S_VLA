"""T43 R2 — 쥔 과일의 질의 구가 **몸통 전체** (바닥 반구 포함) 를 덮고, J1 lift 는 받침 가까이에서만.

T43 S (5): 쥔 사과의 질의 구는 사과 mesh 의 43–64 % 만 덮었고, 막대 접촉점은 사과 바닥 (중심 아래
24–30 mm) 에서 구 밖 3.6–15.5 mm 였다. J1 lift (T34) 가 운반 중에도 구를 14–17 mm 올려 둔 것이 큰
몫이다. 이 파일이 고정하는 것 (합성 — 기록 재생은 `tests/trajopt/test_t43r2_held_body_cover_replay.py`):

1. `body_cover_spheres` — 맞춘 구 + 받침면까지 내린 바닥 구 (+ 미끄럼) 의 합집합이 받침 위 사과를
   ≤ 3 mm 로 덮는다 (한쪽 cap · 손에 가린 작은 cap 모두). 구는 몇 개이고 (≤ 슬롯), 반지름은 최대
   개도의 반 + pad 를 넘지 않는다 — 손가락 파지 여유를 삼키는 큰 구 하나가 아니다. 관측 점은 모두 안.
2. `near_support` — 몸통 바닥 ≤ 띠 + 여유 일 때만 참.
3. `AG3S` 배선 — 기본 off 는 attach · 프레임 제약이 예전과 같은 객체 · 같은 배열. on 이면 받침 위
   (파지) 에서 J1 lift 된 몸통 구, 들어 올린 뒤에는 lift 없는 몸통 구; 오른손도 같은 길; detach 가 지운다.
4. `serve_safe --held-body-cover` — 기본 off, 안전 계층 없는 서버에서는 거절.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.ag3s.robot_models.held_object import (
    HELD_BODY_SLIP_M,
    HELD_LIFT_HYSTERESIS_M,
    SphereFit,
    body_cover_spheres,
    fit_surface_sphere,
    held_spheres_of,
    near_support,
    support_gap,
)
from benchmark.ag3s.types import SupportSurface

MAX_OPENING = 0.07146
PAD = 0.005
APPLE = np.array([0.0333, 0.0333, 0.0305])        # MuJoCo apple mesh: 66.6 × 66.6 × 61.0 mm


def _plane(normal, offset, sid=0):
    return SupportSurface(id=sid, normal=np.asarray(normal, float), offset=float(offset),
                          point_indices=np.zeros(0, np.int64), point_count=1000, rms_error=0.001)


def _ellipsoid_surface(centre, axes, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return centre + v * axes


def _cap(surface, centre, view, cos_min):
    """The part of `surface` a camera along `view` sees (one side)."""
    d = surface - centre
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    v = np.asarray(view, float) / np.linalg.norm(view)
    return surface[d @ v > cos_min]


def _gap(points, spheres):
    """Per point: distance outside the union (≤ 0 = inside)."""
    return np.min(np.stack([np.linalg.norm(points - c, axis=1) - r for c, r in spheres]), axis=0)


# =============================================================================================
# 1. body_cover_spheres
# =============================================================================================


@pytest.mark.parametrize("small_fit", [False, True])
def test_the_body_cover_spheres_cover_an_apple_on_the_table_to_3_mm(small_fit):
    """`small_fit=False`: the fit of a top + one-side view. `True`: the fit a hand-occluded cap gave
    in T41 ep1807 (radius 29.0 mm, centre 0.7 mm high, `T43R2.impl.md` §2). On the real apple mesh
    the H2 sphere then leaves the lower side out (replay test); this ellipsoid only fixes the bound."""
    table = 0.82
    c = np.array([0.45, 0.30, table + APPLE[2]])
    apple = _ellipsoid_surface(c, APPLE)
    obs = _cap(apple, c, [1.0, 0.0, 0.6], 0.0)
    if small_fit:
        fit = SphereFit(centre=c + [0.0, 0.0, 0.0007], radius=0.029, method="surface_fit",
                        n_points=len(obs))
        obs = obs[np.linalg.norm(obs - fit.centre, axis=1) <= 0.029 + PAD]   # what it saw
    else:
        fit = fit_surface_sphere(obs, max_radius=0.5 * MAX_OPENING)
        assert fit.accepted
    spheres, rec = body_cover_spheres(obs, fit, pad=PAD, support=([0, 0, 1], table),
                                      max_spheres=4, max_radius=0.5 * MAX_OPENING)
    gap = _gap(apple, spheres)
    bottom = apple[:, 2] < c[2] - 0.5 * APPLE[2]
    assert gap.max() <= 0.003, (gap.max(), rec)
    assert gap[bottom].max() <= 0.003                         # the bottom hemisphere too
    assert _gap(obs, spheres).max() <= 1e-12                   # every observed point inside
    assert 2 <= len(spheres) <= 4                              # a few spheres ...
    assert max(r for _, r in spheres) <= 0.5 * MAX_OPENING + PAD + 1e-9   # ... none swallows the hand
    from benchmark.ag3s.robot_models.held_object import covering_spheres

    h2, _ = covering_spheres(obs, fit, pad=PAD, max_spheres=4)
    h2_gap = _gap(apple, h2)
    assert h2_gap.max() >= gap.max() - 1e-12                   # never worse than H2


def test_the_bottom_sphere_is_the_body_moved_down_to_the_support_plus_slip():
    c = np.array([0.0, 0.0, 0.82 + 0.034])                     # centre 34 mm above the table
    fit = SphereFit(centre=c, radius=0.030, method="surface_fit", n_points=100)
    obs = c + 0.030 * np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    # radius floor off: the 4 mm the fit does not reach + slip, along −n
    sp, rec = body_cover_spheres(obs, fit, pad=PAD, support=([0, 0, 1], 0.82), slip=0.005,
                                 radius_from_support=False)
    (c0, r0), (c1, r1) = sp[:2]
    assert r0 == r1 == pytest.approx(0.035)
    assert np.allclose(c0, c) and np.allclose(c1, c - [0, 0, 0.009])
    assert rec["support_gap_mm"] == pytest.approx(4.0) and rec["bottom_shift_mm"] == pytest.approx(9.0)
    # radius floor on (default): a fit that misses the support by ≤ one pad is a radius
    # underestimate — the radius goes to 34 mm (≤ max_radius), only the slip remains below
    sp, rec = body_cover_spheres(obs, fit, pad=PAD, support=([0, 0, 1], 0.82), slip=0.005,
                                 max_radius=0.0357)
    assert sp[0][1] == pytest.approx(0.039) and np.allclose(sp[1][0], c - [0, 0, 0.005])
    assert rec["radius_raised_mm"] == pytest.approx(4.0)
    # capped at max_radius: the rest goes into the bottom shift
    sp, rec = body_cover_spheres(obs, fit, pad=PAD, support=([0, 0, 1], 0.82), slip=0.0,
                                 max_radius=0.032)
    assert sp[0][1] == pytest.approx(0.037) and np.allclose(sp[1][0], c - [0, 0, 0.002])
    # misses by more than a pad (a pear's upper cap: centre 43 mm up, radius 27 mm): not a sphere —
    # the radius stays, the bottom sphere reaches the support (+ slip)
    hi = np.array([0.0, 0.0, 0.82 + 0.043])
    pear = SphereFit(centre=hi, radius=0.027, method="surface_fit", n_points=100)
    sp, rec = body_cover_spheres(hi + 0.027 * np.eye(3), pear, pad=PAD, support=([0, 0, 1], 0.82),
                                 slip=0.005, max_radius=0.0357)
    assert "radius_raised_mm" not in rec and sp[0][1] == pytest.approx(0.032)
    assert np.allclose(sp[1][0], hi - [0, 0, 0.016 + 0.005]) and sp[1][1] == pytest.approx(0.032)


def test_no_support_or_a_fit_far_above_it_uses_slip_alone_and_down_is_minus_z():
    c = np.array([0.0, 0.0, 1.0])
    fit = SphereFit(centre=c, radius=0.030, method="surface_fit", n_points=100)
    obs = c + np.array([[0.03, 0, 0]])
    sp, rec = body_cover_spheres(obs, fit, pad=PAD, support=None, slip=0.005)
    assert np.allclose(sp[1][0], c - [0, 0, 0.005]) and rec["support_gap_mm"] is None
    sp, rec = body_cover_spheres(obs, fit, pad=PAD, support=([0, 0, 1], 0.82), slip=0.005)
    assert "support_ignored" in rec and np.allclose(sp[1][0], c - [0, 0, 0.005])
    sp, rec = body_cover_spheres(obs, fit, pad=PAD, support=None, slip=0.0)
    assert len(sp) == 1 and rec["n_body"] == 1 and rec["mode"] == "sphere"   # nothing to add


@pytest.mark.parametrize("rejected", [False, True])
def test_a_banana_is_covered_by_segments_along_its_axis_not_one_big_sphere(rejected):
    """A banana lying on the table (180 × 36 × 36 mm), seen from above: the H2 merge made one
    enclosing sphere of ~95 mm radius. The body cover splits the long axis into ≤ 4 segments,
    each ≤ ~50 mm, covering the banana down to the table (its footprint), observed points inside."""
    table = 0.82
    rng = np.random.default_rng(2)
    t = rng.uniform(-0.09, 0.09, 4000)
    ang = rng.uniform(0.0, 2.0 * np.pi, 4000)
    body = np.column_stack([t, 0.018 * np.cos(ang), table + 0.018 + 0.018 * np.sin(ang)])
    obs = body[body[:, 2] > table + 0.018]                     # the top half is what is seen
    if rejected:
        fit = SphereFit(centre=obs.mean(axis=0), radius=0.0, method="centroid", n_points=len(obs))
    else:
        fit = fit_surface_sphere(obs, max_radius=0.5 * MAX_OPENING)
        fit = SphereFit(centre=fit.centre, radius=min(fit.radius, 0.5 * MAX_OPENING),
                        method="surface_fit", n_points=len(obs))     # a capped "fit", as recorded
    sp, rec = body_cover_spheres(obs, fit, pad=PAD, support=([0, 0, 1], table), max_spheres=4,
                                 max_radius=0.5 * MAX_OPENING)
    assert rec["mode"] == "segments"
    assert rec["segments_reason"] == ("fit_rejected" if rejected else "elongated")
    assert len(sp) <= 4 and max(r for _, r in sp) <= 0.055
    assert _gap(obs, sp).max() <= 1e-12
    assert _gap(body, sp).max() <= 0.003                       # the whole banana, bottom too
    from benchmark.ag3s.robot_models.held_object import covering_spheres

    h2, _ = covering_spheres(obs, fit, pad=PAD, max_spheres=4)
    assert max(r for _, r in h2) > max(r for _, r in sp) or _gap(body, h2).max() > 0.003


def test_a_one_sided_apple_cap_is_not_taken_for_an_elongated_body():
    """A side view of an apple is a crescent (narrow in depth): the fit sphere's diameter is the
    width the elongation test uses, so the apple keeps the sphere mode."""
    table = 0.82
    c = np.array([0.45, 0.30, table + APPLE[2]])
    apple = _ellipsoid_surface(c, APPLE)
    obs = _cap(apple, c, [1.0, 0.0, 0.0], 0.6)                  # narrow crescent seen from +x
    fit = fit_surface_sphere(obs, max_radius=0.5 * MAX_OPENING)
    assert fit.accepted
    _, rec = body_cover_spheres(obs, fit, pad=PAD, support=([0, 0, 1], table),
                                max_radius=0.5 * MAX_OPENING)
    assert rec["horizontal_extent_mm"][0] > 1.6 * rec["horizontal_extent_mm"][1]   # raw: a crescent
    assert rec["mode"] == "sphere"


# =============================================================================================
# 2. near_support
# =============================================================================================


def test_near_support_is_the_body_bottom_within_band_plus_hysteresis():
    radii = np.array([0.035, 0.035])
    for bottom, near in ((-0.010, True), (0.0088, True), (0.0088 + HELD_LIFT_HYSTERESIS_M - 1e-6, True),
                         (0.0088 + HELD_LIFT_HYSTERESIS_M + 1e-4, False), (0.20, False)):
        centres = np.array([[0, 0, 0.82 + 0.035 + bottom + 0.005], [0, 0, 0.82 + 0.035 + bottom]])
        got, gap = near_support(centres, radii, [0, 0, 1], 0.82, band=0.0088)
        assert got is near and gap == pytest.approx(bottom)
    assert np.allclose(support_gap([[0, 0, 1.0]], [0.03], [0, 0, 2.0], 0.9), [0.07])


# =============================================================================================
# 3. AG3S wiring
# =============================================================================================


class _TableField:
    """d = 0 up to `band` above the table plane, then h − band (the recorded fine-ESDF band)."""

    def __init__(self, z0, band):
        self.z0, self.band = z0, band

    def distance(self, p):
        p = np.asarray(p, float).reshape(-1, 3)
        return np.maximum(p[:, 2] - self.z0 - self.band, 0.0)


@pytest.fixture(scope="module")
def rby1():
    from benchmark.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf

    return UrdfSphereChain(parse_urdf(RBY1_URDF),
                           link_filter=["ee_left", "ee_right", "ee_finger_l1", "ee_finger_l2",
                                        "ee_finger_r1", "ee_finger_r2"])


def _ag3s(model, body_cover):
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.runtime.pipeline import AG3S

    cfg = AG3SConfig().with_overrides({"constraint": {"horizon": 2, "max_attached_primitives": 4},
                                       "clustering": {"gripper_max_opening": MAX_OPENING},
                                       "esdf": {"backend": "curobo", "voxel_size": 0.02,
                                                "fine_voxel_size": 0.005,
                                                "tsdf_voxel_size": 0.005}})
    ag3s = AG3S(cfg, robot_model=model, constraint_robot_model=model,
                attached_parent_links=("ee_left", "ee_right"))
    ag3s.held_body_cover = body_cover
    return ag3s


def _target(points, tid=3):
    from benchmark.ag3s.stages.geometry import fit_sphere
    from benchmark.ag3s.types import TargetGeometry

    pts = np.asarray(points, np.float64)
    return TargetGeometry(id=tid, points=pts, point_indices=np.arange(len(pts)),
                          centroid=pts.mean(axis=0), bounding_geometry=fit_sphere(pts),
                          attention_score=0.9, confidence=0.9)


def _grasp(model, hand, body_cover):
    """An apple between the `hand`'s fingers at q = 0, standing on a table; attach on finger 1."""
    q = np.zeros(model.nq)
    s = "l" if hand == "left" else "r"
    T1, T2 = model.link_pose(q, f"ee_finger_{s}1"), model.link_pose(q, f"ee_finger_{s}2")
    c = 0.5 * (T1[:3, 3] + T2[:3, 3])
    table = c[2] - APPLE[2]
    apple = _ellipsoid_surface(c, APPLE)
    obs = _cap(apple, c, [1.0, 0.0, 0.6], 0.0)
    ag3s = _ag3s(model, body_cover)
    ag3s._last_support_surfaces = [_plane([0, 0, 1], table, 1)]             # noqa: SLF001
    ag3s._last_esdf_field = _TableField(table, 0.0088)                       # noqa: SLF001
    held = ag3s.attach(_target(obs), robot_state=q, parent_link=f"ee_finger_{s}1")
    return ag3s, held, q, table, apple


def _base(model, q, att):
    T = model.link_pose(q, att.parent_link)
    return [(T[:3, :3] @ c + T[:3, 3], r) for c, r in held_spheres_of(att)]


def test_default_off_keeps_the_attached_object_and_its_spheres(rby1):
    off, held_off, q, table, _ = _grasp(rby1, "left", False)
    on, held_on, _, _, _ = _grasp(rby1, "left", True)
    assert off.held_body_cover is False and "body_cover" not in off.held_record
    # the attach itself is the same with the flag on: J1 spheres, filter copy, record
    assert np.array_equal(np.array([c for c, _ in held_spheres_of(held_off)]),
                          np.array([c for c, _ in held_spheres_of(held_on)]))
    assert off.held_record["spheres_parent_frame"] == on.held_record["spheres_parent_frame"]
    # off: the frame's constraint set carries `_attached` itself — same object, same arrays
    surfaces = [_plane([0, 0, 1], table, 1)]
    assert off._frame_held_attached(q, surfaces) is held_off                # noqa: SLF001
    assert off.attached is held_off
    assert off._frame_attached is None                                      # noqa: SLF001


@pytest.mark.parametrize("hand", ["left", "right"])
def test_on_the_table_the_body_is_lifted_and_above_it_it_is_not(rby1, hand):
    ag3s, held, q, table, apple = _grasp(rby1, hand, True)
    palm = "ee_left" if hand == "left" else "ee_right"
    assert held.parent_link == palm
    bc = ag3s.held_record["body_cover"]
    assert bc["band_mm"] == pytest.approx(8.8, abs=0.01) and bc["lift"]["max_lift_mm"] > 0.0
    surfaces = [_plane([0, 0, 1], table, 1)]
    # at the grasp: near the support -> the J1-lifted body clears plane + band (no table row)
    at = ag3s._frame_held_attached(q, surfaces)                              # noqa: SLF001
    assert ag3s.attached is at and at.parent_link == palm
    assert ag3s.held_record["body_cover"]["frame"]["near_support"] is True
    for c, r in _base(rby1, q, at):
        assert c[2] - r - table >= 0.0088 - 1e-9
    # 60 mm up (lifted off): not near -> the unlifted body, which covers the apple bottom
    q_up = q.copy()
    up = 0.0
    T_now = rby1.link_pose(q, palm)

    class _Shift:                                     # the same robot, the palm 60 mm higher
        def __init__(self, m):
            self.m = m

        def __getattr__(self, k):
            return getattr(self.m, k)

        def link_pose(self, qq, link):
            T = np.array(self.m.link_pose(qq, link))
            T[2, 3] += up
            return T

    # pressed 6 mm lower than at the attach (still on the table): still near, and the lifted
    # spheres are raised further so they clear plane + band at this pose too
    up = -0.006
    ag3s.constraint_robot_model = _Shift(rby1)
    pressed = ag3s._frame_held_attached(q, surfaces)                         # noqa: SLF001
    fr = ag3s.held_record["body_cover"]["frame"]
    assert fr["near_support"] is True and max(fr["extra_lift_mm"]) > 0.0
    Tp = np.array(T_now)
    Tp[2, 3] += up
    for c, r in held_spheres_of(pressed):
        assert (Tp[:3, :3] @ c + Tp[:3, 3])[2] - r - table >= 0.0088 - 1e-9
    up = 0.060
    far = ag3s._frame_held_attached(q_up, surfaces)                          # noqa: SLF001
    rec = ag3s.held_record
    assert rec["body_cover"]["frame"]["near_support"] is False
    assert rec["body_cover"]["frame"]["extra_lift_mm"] is None
    assert rec["body_cover"]["frame"]["min_bottom_gap_mm"] > 0.0088e3 + HELD_LIFT_HYSTERESIS_M * 1e3
    Tu = np.array(T_now)
    Tu[2, 3] += up
    sph = [(Tu[:3, :3] @ c + Tu[:3, 3], r) for c, r in held_spheres_of(far)]
    assert _gap(apple + [0, 0, up], sph).max() <= 0.003              # the whole apple, bottom too
    assert rec["spheres_parent_frame"] == bc["body_parent_frame"]    # the record says which
    # the lifted copy would have left the apple bottom out
    lifted = [(Tu[:3, :3] @ np.asarray(x["centre_m"]) + Tu[:3, 3], x["radius_mm"] / 1000.0)
              for x in bc["lifted_parent_frame"]]
    assert _gap(apple + [0, 0, up], lifted).max() > 0.005
    # the per-frame TSDF free volume and the self-filter keep the pre-T43 spheres
    for (c0, r0), (c1, r1) in zip(held_spheres_of(ag3s._attached),           # noqa: SLF001
                                  held_spheres_of(held), strict=True):
        assert np.array_equal(c0, c1) and r0 == r1
    ag3s.detach()
    assert ag3s.attached is None and ag3s._held_query is None                # noqa: SLF001
    assert ag3s._frame_attached is None                                      # noqa: SLF001


def test_an_earlier_frame_record_is_not_rewritten_by_a_later_frame(rby1):
    ag3s, held, q, table, _ = _grasp(rby1, "left", True)
    surfaces = [_plane([0, 0, 1], table, 1)]
    ag3s._frame_held_attached(q, surfaces)                                   # noqa: SLF001
    first = ag3s.held_record
    snap = dict(first["body_cover"]["frame"])
    ag3s.frame_index += 1
    ag3s._frame_held_attached(q, [])          # no surface this frame: the attach plane stands in
    assert first["body_cover"]["frame"] == snap
    assert ag3s.held_record["body_cover"]["frame"]["near_support"] is True
    assert ag3s.held_record["body_cover"]["frame"]["plane"]["surface_id"] == -2


def test_a_prebuilt_attachment_is_untouched_by_the_flag(rby1):
    from benchmark.ag3s.types import AttachedCollisionGeometry, Primitive, PrimitiveType

    ag3s = _ag3s(rby1, True)
    geo = AttachedCollisionGeometry(
        parent_link="ee_left", T_parent_object=np.eye(4),
        primitives=[Primitive(type=PrimitiveType.SPHERE, center=np.array([0, 0, -0.1]),
                              dimensions=np.full(3, 0.03))],
        points=_ellipsoid_surface(np.array([0, 0, -0.1]), np.full(3, 0.03), n=200))
    assert ag3s.attach(geo) is geo
    assert ag3s._frame_held_attached(np.zeros(rby1.nq), []) is geo            # noqa: SLF001
    assert ag3s.attached is geo


def test_slip_default_is_the_measured_sag():
    assert HELD_BODY_SLIP_M == pytest.approx(0.005)


# =============================================================================================
# 4. serve_safe flag
# =============================================================================================


def test_cli_flag_is_off_by_default_and_rejected_without_the_safety_layer():
    from benchmark.trajopt.serve_safe import build_parser, reject_bad_flag_combinations

    ap = build_parser()
    base = ["--config", "c", "--checkpoint", "k", "--model-xml", "m"]
    assert ap.parse_args(base).held_body_cover is False
    assert ap.parse_args(base + ["--held-body-cover"]).held_body_cover is True
    for extra in (["--no-safe"], ["--no-perception"]):
        args = ap.parse_args(base + ["--held-body-cover"] + extra)
        with pytest.raises(SystemExit):
            reject_bad_flag_combinations(ap, args)
