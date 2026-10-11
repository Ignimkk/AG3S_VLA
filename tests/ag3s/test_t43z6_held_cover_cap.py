"""T43 Z6 (1) — held query sphere cap (`clustering.held_cover_cap`, `serve_safe --held-cover-cap-mm`).

User decision (a), 2026-10-06 07:10: at attach every held query sphere (body cover `sphere` and
`segments`, or the H2 covering spheres) gets radius ≤ observed half-width + slip allowance, a centre
within a fixed distance of the observed body centre, and duplicates removed. Default off = the
attach is bit-identical. This file fixes (synthetic; the T43 W record recomputation is in
`T43Z6.impl.md` §offline):

1. `cap_held_spheres` / `dedupe_spheres` / `held_cap_reference` — the rule and its envelope.
2. `AG3S.attach` — off: no `cap` key, the same spheres; on: query spheres capped (body cover on and
   off), the self-filter copy untouched, an elongated body not capped.
3. config + CLI wiring.
"""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.robot_models.held_object import (
    SphereFit,
    body_cover_spheres,
    cap_held_spheres,
    dedupe_spheres,
    held_cap_reference,
    held_spheres_of,
)
from tests.ag3s.test_t43r2_held_body_cover import (
    APPLE,
    MAX_OPENING,
    PAD,
    _TableField,
    _cap,
    _ellipsoid_surface,
    _gap,
    _plane,
    _target,
    rby1,  # noqa: F401 — pytest fixture
)

CAP = 0.010           # the allowance the impl report recommends (pad 5 + slip 5 mm)
CRATE_HALF_X = 0.082  # crate inner half-width (MJCF, T43Z.verify numbers.Z1)
DEST_MARGIN = 0.020


# =============================================================================================
# 1. the rule
# =============================================================================================


def test_radius_is_capped_and_a_far_centre_is_pulled_in_to_the_limit():
    c0 = np.array([0.5, 0.2, 0.9])
    spheres = [(c0 + [0.017, -0.002, 0.0], 0.0583), (c0 + [-0.0166, -0.0018, 0.0], 0.0558),
               (c0 + [0.0, 0.0, -0.004], 0.030)]
    out, rec = cap_held_spheres(spheres, c0, 0.0334, CAP)
    # the small sphere near the centre ends inside the capped first one and is removed
    assert len(out) == 2 and rec["n_removed"] == 1
    for (c, r), (c_in, r_in) in zip(out, spheres[:2]):
        assert r == pytest.approx(min(r_in, 0.0434))
        off_in = np.linalg.norm(c_in - c0)
        assert np.linalg.norm(c - c0) == pytest.approx(min(off_in, CAP))
        # pulled along its own direction
        if off_in > CAP:
            assert np.allclose((c - c0) / np.linalg.norm(c - c0), (c_in - c0) / off_in)
    assert rec["radius_cap_mm"] == pytest.approx(43.4) and rec["centre_limit_mm"] == pytest.approx(10.0)
    assert rec["n_radius_capped"] == 2 and rec["n_centre_moved"] == 2


def test_every_capped_sphere_lies_in_the_envelope_ball():
    rng = np.random.default_rng(3)
    c0 = np.zeros(3)
    for _ in range(50):
        spheres = [(rng.normal(scale=0.05, size=3), float(rng.uniform(0.005, 0.08)))
                   for _ in range(int(rng.integers(1, 5)))]
        out, _ = cap_held_spheres(spheres, c0, 0.033, CAP)
        for c, r in out:
            assert np.linalg.norm(c - c0) + r <= 0.033 + 2 * CAP + 1e-12


def test_the_segments_shape_of_v5_1925_fits_the_crate_after_the_cap():
    """V5 1925 s19253 (T43 Z1): two segments spheres 58.3 / 55.8 mm, ±17 mm along x of the apple
    centre — x half-span 73.8 mm, + 20 mm margin = 93.8 > 82. Capped at a fit radius of 33.4 mm."""
    c0 = np.zeros(3)
    spheres = [(np.array([0.0169, -0.0021, 0.0294]) - [0, 0, 0.025], 0.058332),
               (np.array([-0.0166, -0.0018, 0.0270]) - [0, 0, 0.025], 0.055795)]
    hx = lambda s: 0.5 * (max(c[0] + r for c, r in s) - min(c[0] - r for c, r in s))  # noqa: E731
    assert hx(spheres) + DEST_MARGIN > CRATE_HALF_X
    out, _ = cap_held_spheres(spheres, c0, 0.0334, CAP)
    assert hx(out) + DEST_MARGIN <= CRATE_HALF_X
    # and still covers an apple-size ellipsoid around the observed centre
    apple = _ellipsoid_surface(c0, APPLE, n=4000)
    assert _gap(apple, out).max() <= 1e-9


def test_duplicates_and_contained_spheres_are_removed_and_the_first_kept():
    a = (np.array([0.0, 0.0, 0.0]), 0.0385)
    out, n = dedupe_spheres([a, (a[0].copy(), a[1]), (np.array([0.016, 0.0067, 0.0218]), 0.0147),
                             (np.array([0.01, 0.0, 0.0]), 0.02)])
    # the duplicate and the sphere inside the first go; a sphere sticking out stays
    assert n == 2 and len(out) == 2
    assert np.array_equal(out[0][0], a[0]) and out[0][1] == a[1]
    assert np.allclose(out[1][0], [0.016, 0.0067, 0.0218])
    # nothing to remove: same list back
    same, n = dedupe_spheres([a, (np.array([0.06, 0, 0]), 0.01)])
    assert n == 0 and len(same) == 2


def test_the_reference_is_the_fit_centre_and_body_radius_or_the_observed_box():
    c = np.array([0.4, 0.1, 0.85])
    obs = _cap(_ellipsoid_surface(c, APPLE), c, [1.0, 0.0, 0.6], 0.0)
    fit = SphereFit(centre=c, radius=0.0330, method="surface_fit", n_points=len(obs))
    ref = held_cap_reference(obs, fit, body_radius=0.0334)
    assert ref["basis"] == "fit" and np.allclose(ref["centre"], c)
    assert ref["half_width"] == pytest.approx(0.0334)
    assert held_cap_reference(obs, fit)["half_width"] == pytest.approx(0.0330)
    rejected = SphereFit(centre=obs.mean(axis=0), radius=0.06, method="centroid",
                         n_points=len(obs), reason="residual")
    ref = held_cap_reference(obs, rejected)
    assert ref["basis"] == "observed_box"
    assert np.allclose(ref["centre"], 0.5 * (obs.min(axis=0) + obs.max(axis=0)))
    assert 0.020 < ref["half_width"] <= 0.0334 + 1e-9
    # a banana (180 × 36 mm from above): not capped
    banana = _ellipsoid_surface(c, [0.09, 0.018, 0.018])
    bfit = SphereFit(centre=c, radius=0.018, method="surface_fit", n_points=len(banana))
    assert held_cap_reference(banana, bfit) == {"skip": "elongated",
                                                "horizontal_extent_mm": pytest.approx(
                                                    held_cap_reference(banana, bfit)
                                                    ["horizontal_extent_mm"])}
    assert held_cap_reference(np.zeros((0, 3)), fit) == {"skip": "no_points"}


# =============================================================================================
# 2. AG3S.attach
# =============================================================================================


def _ag3s(model, *, body_cover, cap):
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S

    clustering = {"gripper_max_opening": MAX_OPENING}
    if cap is not None:
        clustering["held_cover_cap"] = cap
    cfg = AG3SConfig().with_overrides({"constraint": {"horizon": 2, "max_attached_primitives": 4},
                                       "clustering": clustering,
                                       "esdf": {"backend": "curobo", "voxel_size": 0.02,
                                                "fine_voxel_size": 0.005,
                                                "tsdf_voxel_size": 0.005}})
    ag3s = AG3S(cfg, robot_model=model, constraint_robot_model=model,
                attached_parent_links=("ee_left", "ee_right"))
    ag3s.held_body_cover = body_cover
    return ag3s


def _strays(c, n_groups=4, seed=1):
    """Small point groups 12 mm outside the upper apple surface (finger / leaf returns) — more
    groups than the cover slots, so R2 falls to `segments` (`covers_over_slots`, as in the V5 runs)."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_groups):
        ang = 2 * np.pi * k / n_groups
        d = np.array([np.cos(ang), np.sin(ang), 2.0])
        d /= np.linalg.norm(d)
        centre = c + d * (0.0333 + 0.012)
        out.append(centre + rng.normal(scale=0.002, size=(6, 3)))
    return np.vstack(out)


def _grasp(model, *, body_cover, cap, strays=False, shape=APPLE):
    q = np.zeros(model.nq)
    T1, T2 = model.link_pose(q, "ee_finger_l1"), model.link_pose(q, "ee_finger_l2")
    c = 0.5 * (T1[:3, 3] + T2[:3, 3])
    table = c[2] - shape[2]
    obs = _cap(_ellipsoid_surface(c, shape), c, [1.0, 0.0, 0.6], 0.0)
    if strays:
        obs = np.vstack([obs, _strays(c)])
    ag3s = _ag3s(model, body_cover=body_cover, cap=cap)
    ag3s._last_support_surfaces = [_plane([0, 0, 1], table, 1)]             # noqa: SLF001
    ag3s._last_esdf_field = _TableField(table, 0.0088)                       # noqa: SLF001
    held = ag3s.attach(_target(obs), robot_state=q, parent_link="ee_finger_l1")
    return ag3s, held, q, c


@pytest.mark.parametrize("body_cover", [False, True])
@pytest.mark.parametrize("strays", [False, True])
def test_off_is_the_attach_as_before(rby1, body_cover, strays):  # noqa: F811
    a, held_a, _, _ = _grasp(rby1, body_cover=body_cover, cap=None, strays=strays)
    b, held_b, _, _ = _grasp(rby1, body_cover=body_cover, cap=None, strays=strays)
    assert "cap" not in a.held_record
    assert "cap" not in (a.held_record.get("body_cover") or {})
    assert a.held_record == b.held_record
    assert a.config.clustering.held_cover_cap is None


def _query_base(ag3s, q):
    """The query spheres the frame's constraint set would carry (body cover per frame), base frame."""
    att = ag3s._frame_held_attached(q, ag3s._last_support_surfaces) or ag3s.attached  # noqa: SLF001
    T = ag3s.constraint_robot_model.link_pose(q, att.parent_link)
    return [(T[:3, :3] @ c + T[:3, 3], r) for c, r in held_spheres_of(att)]


def test_segments_body_cover_is_capped_and_the_filter_copy_is_not(rby1):  # noqa: F811
    off, _, q, c = _grasp(rby1, body_cover=True, cap=None, strays=True)
    on, _, _, _ = _grasp(rby1, body_cover=True, cap=CAP, strays=True)
    bc_off, bc_on = off.held_record["body_cover"], on.held_record["body_cover"]
    assert bc_off["mode"] == "segments" and bc_off["segments_reason"] == "covers_over_slots"
    r_off = [s["radius_mm"] for s in bc_off["body_parent_frame"]]
    assert max(r_off) > 50.0                                  # the V5 'segments' size
    cap = bc_on["cap"]
    assert cap["applied"] is True and cap["basis"] == "fit"
    w = cap["half_width_mm"]
    assert 30.0 < w < 40.0
    for s in bc_on["body_parent_frame"] + bc_on["lifted_parent_frame"]:
        assert s["radius_mm"] <= w + CAP * 1000.0 + 1e-6
    centre = np.asarray(cap["centre_m"])
    T = on.constraint_robot_model.link_pose(q, on.attached.parent_link)
    for s in bc_on["body_parent_frame"]:
        cb = T[:3, :3] @ np.asarray(s["centre_m"]) + T[:3, 3]      # attach pose == capture pose
        assert np.linalg.norm(cb - centre) <= CAP + 1e-6
    # the self-filter copy (fit spheres + slip) is what it was
    assert on.held_record["filter_spheres_parent_frame"] == off.held_record["filter_spheres_parent_frame"]
    # the capped body (unlifted) still covers the apple (to the fit's accuracy); near the table
    # the J1-lifted copy leaves the bottom out by design
    apple = _ellipsoid_surface(c, APPLE, n=3000)
    body = [(T[:3, :3] @ np.asarray(s["centre_m"]) + T[:3, 3], s["radius_mm"] / 1000.0)
            for s in bc_on["body_parent_frame"]]
    assert _gap(apple, body).max() <= 0.004
    assert len(_query_base(on, q)) == len(bc_on["lifted_parent_frame"])


def test_sphere_mode_keeps_its_body_and_loses_only_duplicates(rby1):  # noqa: F811
    off, _, q, _ = _grasp(rby1, body_cover=True, cap=None)
    on, _, _, _ = _grasp(rby1, body_cover=True, cap=CAP)
    bc_off, bc_on = off.held_record["body_cover"], on.held_record["body_cover"]
    assert bc_off["mode"] == "sphere"
    # the two body spheres (fit + the bottom sphere slip below) are inside the cap already
    assert bc_on["body_parent_frame"][:2] == bc_off["body_parent_frame"][:2]
    lifted_off = [(tuple(s["centre_m"]), s["radius_mm"]) for s in bc_off["lifted_parent_frame"]]
    lifted_on = [(tuple(s["centre_m"]), s["radius_mm"]) for s in bc_on["lifted_parent_frame"]]
    assert len(set(lifted_on)) == len(lifted_on)               # no duplicate left
    assert len(lifted_on) == len(lifted_off) - bc_on["cap"]["n_removed_after_lift"] \
        - bc_on["cap"]["n_removed"]


def test_h2_query_spheres_are_capped_when_body_cover_is_off(rby1):  # noqa: F811
    off, held_off, q, _ = _grasp(rby1, body_cover=False, cap=None, strays=True)
    on, held_on, _, _ = _grasp(rby1, body_cover=False, cap=CAP, strays=True)
    cap = on.held_record["cap"]
    assert cap["applied"] is True
    lim = cap["half_width_mm"] + CAP * 1000.0
    assert all(r * 1000.0 <= lim + 1e-3 for _, r in held_spheres_of(held_on))   # record rounding
    assert max(r for _, r in held_spheres_of(held_off)) * 1000.0 > lim      # it was bigger
    assert on.held_record["filter_spheres_parent_frame"] == off.held_record["filter_spheres_parent_frame"]


def test_an_elongated_body_is_not_capped(rby1):  # noqa: F811
    banana = np.array([0.09, 0.018, 0.018])
    off, _, _, _ = _grasp(rby1, body_cover=True, cap=None, shape=banana)
    on, _, _, _ = _grasp(rby1, body_cover=True, cap=CAP, shape=banana)
    assert on.held_record["body_cover"]["cap"]["applied"] is False
    assert on.held_record["body_cover"]["cap"]["skip"] == "elongated"
    assert on.held_record["body_cover"]["body_parent_frame"] == \
        off.held_record["body_cover"]["body_parent_frame"]


def test_body_cover_spheres_unchanged_by_the_new_code():
    """The cap lives outside `body_cover_spheres` — its output for a fixed input is as before."""
    c = np.array([0.0, 0.0, 0.82 + 0.034])
    fit = SphereFit(centre=c, radius=0.030, method="surface_fit", n_points=100)
    obs = c + 0.030 * np.eye(3)
    sp, _ = body_cover_spheres(obs, fit, pad=PAD, support=([0, 0, 1], 0.82), slip=0.005,
                               max_radius=0.0357)
    assert sp[0][1] == pytest.approx(0.039)


# =============================================================================================
# 3. config + CLI
# =============================================================================================


def test_config_default_validation_and_round_trip():
    from AG3S.ag3s.config import AG3SConfig, AG3SConfigError

    assert AG3SConfig().clustering.held_cover_cap is None
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf"}).with_overrides(
        {"clustering": {"held_cover_cap": 0.01}})
    assert cfg.clustering.held_cover_cap == 0.01 and cfg.collision_backend == "esdf"
    assert AG3SConfig.from_dict(cfg.to_dict()) == cfg
    for bad in (-0.001, True, "10", float("nan")):
        with pytest.raises(AG3SConfigError, match="held_cover_cap"):
            AG3SConfig.from_dict({"clustering": {"held_cover_cap": bad}})


def test_cli_flag_maps_to_metres_and_needs_the_safe_layer():
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.trajopt import serve_safe
    from tests.trajopt.test_t41a_obstacle_margin import BASE_ARGV

    ap = serve_safe.build_parser()
    assert "held_cover_cap" not in serve_safe.t43t_identity_section(ap.parse_args(BASE_ARGV))
    sec = serve_safe.t43t_identity_section(ap.parse_args(BASE_ARGV + ["--held-cover-cap-mm", "10"]))
    assert sec == {"held_cover_cap": pytest.approx(0.010)}
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": {"backend": "curobo"}}) \
        .with_overrides({"clustering": sec})
    serve_safe.assert_ag3s_config(cfg, collision_backend="esdf", esdf_backend="curobo", clustering=sec)
    for extra in (["--no-safe"], ["--held-cover-cap-mm", "-1"]):
        argv = BASE_ARGV + (["--held-cover-cap-mm", "10"] if extra == ["--no-safe"] else []) + extra
        with pytest.raises(SystemExit):
            serve_safe.reject_bad_flag_combinations(ap, ap.parse_args(argv))
