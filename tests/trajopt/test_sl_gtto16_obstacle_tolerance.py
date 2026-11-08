"""SL-GTTO16 (A, user ruling 2026-11-08 00:2x) — `serve_safe --passthrough-obstacle-tolerance-mm MM`.

F12's accepted model violation tau (25 mm) passed a held-object <-> obstacle row at -20.4 mm (SafeLIBERO t2e0, collision two
steps later). With MM set, tau does not cover the rows whose nearest surface is an obstacle object
(`linearize.obstacle_object_rows`: not the manipulated object · target volume · a support plane · the destination · the held
object's own remnant). Pinned here:
1. off (flag absent): no config key, `obstacle_object_rows` is never called;
2. on: a reference within tau passes only when no obstacle-object row is below its floor by more than MM;
3. the classifier keeps support / destination / target rows out and reports the worst obstacle row;
4. the CLI checks and the Panda server port; the F13 version = F12 + the flag.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError

pytest.importorskip("casadi")
pytest.importorskip("osqp")

from tests.trajopt.test_t88_passthrough_when_clear import BASE, _inside, _no_qp, _opt  # noqa: E402


def _violating(pen):
    from tests.trajopt.fixtures import make_scenario

    return _inside(make_scenario(horizon=8), pen)


TAU = {"sqp": {"passthrough_tolerance": 0.04}, "safety": {"violation_tolerance": 0.04}}


def test_default_has_no_key_and_validates():
    c = TrajOptConfig()
    assert c.sqp.passthrough_obstacle_tolerance is None
    assert "passthrough_obstacle_tolerance" not in c.to_dict()["sqp"]
    on = c.with_overrides({"sqp": {"passthrough_obstacle_tolerance": 0.0}})
    assert TrajOptConfig.from_dict(on.to_dict()) == on
    with pytest.raises(TrajOptConfigError, match="passthrough_obstacle_tolerance"):
        TrajOptConfig.from_dict({"sqp": {"passthrough_obstacle_tolerance": -0.001}})


def test_off_never_classifies(monkeypatch):
    from AG3S.trajopt.linearize import CollisionLinearizer as Linearizer

    def boom(*a, **k):
        raise AssertionError("classified with the flag off")

    monkeypatch.setattr(Linearizer, "obstacle_object_rows", boom)
    sc = _violating(0.01)
    _no_qp(monkeypatch)
    out = _opt(sc, on=True, **TAU).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    assert out.metrics["passthrough_clear"]["applied"] is True
    assert "obstacle_rows" not in out.metrics["passthrough_clear"]


@pytest.mark.parametrize("worst,applied", [(-0.02, False), (-0.0005, True), (float("inf"), True)])
def test_obstacle_row_beyond_its_tolerance_refuses(monkeypatch, worst, applied):
    from AG3S.trajopt.linearize import CollisionLinearizer as Linearizer

    pair = None if not np.isfinite(worst) else {"step": 1, "link": "held_0", "obstacle": None, "clearance_m": worst,
                                                "point_m": [0, 0, 0], "n_obstacle_rows": 3, "n_rows": 10}
    monkeypatch.setattr(Linearizer, "obstacle_object_rows", lambda self, *a, **k: (worst, pair))
    sc = _violating(0.01)
    opt = _opt(sc, on=True, sqp={**TAU["sqp"], "passthrough_obstacle_tolerance": 0.001}, safety=TAU["safety"])
    out = opt.solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    rec = out.metrics["passthrough_clear"]
    assert rec["applied"] is applied and rec["reason"] == ("clear" if applied else "obstacle")
    assert rec["obstacle_rows"]["tolerance_m"] == pytest.approx(0.001)
    assert rec["obstacle_rows"]["violation_m"] == pytest.approx(max(0.0, -worst) if np.isfinite(worst) else 0.0)


def test_classifier_error_fails_closed(monkeypatch):
    from AG3S.trajopt.linearize import CollisionLinearizer as Linearizer

    def bad(self, *a, **k):
        raise RuntimeError("no field")

    monkeypatch.setattr(Linearizer, "obstacle_object_rows", bad)
    sc = _violating(0.01)
    out = _opt(sc, on=True, sqp={**TAU["sqp"], "passthrough_obstacle_tolerance": 0.0},
               safety=TAU["safety"]).solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    assert out.metrics["passthrough_clear"]["reason"] == "obstacle"
    assert "error" in out.metrics["passthrough_clear"]["obstacle_rows"]


class _Field:
    """z = 0 table plane + one obstacle point at `obs`; points with x > 10 are labelled 'dest'."""

    has_labels = True
    label_names = ("dest",)

    def __init__(self, obs):
        self.obs = np.asarray(obs, float)
        self.grid = types.SimpleNamespace(voxel_size=0.005)

    def distance(self, p):
        p = np.asarray(p, float).reshape(-1, 3)
        return np.minimum(p[:, 2], np.linalg.norm(p - self.obs, axis=1))

    def is_label(self, p, name):
        return np.asarray(p, float).reshape(-1, 3)[:, 0] > 10.0

    def label(self, p):
        return np.where(self.is_label(p, "dest"), 0, -1)


def test_classifier_keeps_support_and_destination_out():
    sc = _violating(0.01)
    lin = sc.linearizer
    states = lin.sphere_states(sc.reference, sc.q_now)
    centres = np.asarray(states[0])
    h, s = centres.shape[:2]
    pick = centres[2, s // 2]
    field = _Field(pick + np.array([0.0, 0.0, 0.03]))
    scene = types.SimpleNamespace(esdf=field, target_free_mask=None, target_volume=None,
                                  support_planes=np.array([[0.0, 0.0, 1.0, 0.0]]), destination_label="dest",
                                  manipulated_spheres=None, manipulated_points=None)
    clear = np.full((h, s), 0.05)
    clear[2, s // 2] = -0.012                            # the row next to the obstacle point
    on_table = int(np.argmin(centres[3, :, 2]))
    clear[3, on_table] = -0.03                           # a deeper row whose nearest surface is the table
    lin_cls = type(lin)
    orig = lin_cls._esdf_clearance
    try:
        lin_cls._esdf_clearance = lambda self, c, sc_: clear
        worst, pair = lin.obstacle_object_rows(sc.reference, sc.q_now, scene)
    finally:
        lin_cls._esdf_clearance = orig
    d = field.distance(centres.reshape(-1, 3)).reshape(h, s)
    if d[3, on_table] == centres[3, on_table, 2]:        # the table really is that row's nearest surface
        assert worst == pytest.approx(-0.012)
    assert pair["step"] == 2 and pair["query"] == s // 2 and pair["n_obstacle_rows"] < h * s


def test_cli_port_and_version():
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    a = ap.parse_args(BASE + ["--passthrough-when-clear", "--passthrough-obstacle-tolerance-mm", "0"])
    assert ss.trajopt_config_from_args(a).sqp.passthrough_obstacle_tolerance == 0.0
    for bad in (["--passthrough-obstacle-tolerance-mm", "0"],
                ["--passthrough-when-clear", "--passthrough-obstacle-tolerance-mm", "-1"]):
        with pytest.raises(SystemExit):
            ss.reject_bad_flag_combinations(ap, ap.parse_args(BASE + bad))
    assert "passthrough_obstacle_tolerance_mm" in srv.PORTED_FINAL_CONFIG_DESTS
    f12 = pf.AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2"]
    f13 = pf.AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst"]
    assert tuple(f13) == tuple(f12) + ("--passthrough-obstacle-tolerance-mm", "0")
    assert pf.is_gtto_version("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst")
    assert pf.is_final_version("AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst-obs")


# ---- A' (2026-11-08 01:3x): the beneath rule --------------------------------------------------------------------------
class _PlaneField(_Field):
    """z = 0 plane only (no labels) with a gradient."""

    has_labels = False

    def distance(self, p):
        return np.asarray(p, float).reshape(-1, 3)[:, 2].copy()

    def gradient(self, p):
        g = np.zeros((len(np.asarray(p).reshape(-1, 3)), 3))
        g[:, 2] = 1.0
        return g


def _beneath_case(latch):
    sc = _violating(0.01)
    lin = sc.linearizer
    centres = np.asarray(lin.sphere_states(sc.reference, sc.q_now)[0])
    h, s = centres.shape[:2]
    k = s // 2
    p = centres[2, k]
    obj = p + np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])   # the object right above the row's surface
    obj[:, 2] = p[2] + 0.01
    shift = np.array([0.0, 0.0, -float(p[2])])                                   # the plane z = 0 under that row
    field = _PlaneField(np.zeros(3))
    scene = types.SimpleNamespace(esdf=field, target_free_mask=None, target_volume=None, support_planes=None,
                                  destination_label=None, manipulated_spheres=None, manipulated_points=obj,
                                  grasp_contact={"latch": latch})
    clear = np.full((h, s), 0.05)
    clear[2, k] = -0.012
    return lin, sc, scene, clear, shift


@pytest.mark.parametrize("latch,exempt", [("latched", True), ("closing", True), ("searching", False)])
def test_beneath_the_target_is_not_an_obstacle_while_grasping(latch, exempt):
    lin, sc, scene, clear, _ = _beneath_case(latch)
    cls = type(lin)
    orig = cls._esdf_clearance
    try:
        cls._esdf_clearance = lambda self, c, sc_: clear
        off = lin.obstacle_object_rows(sc.reference, sc.q_now, scene)
        on = lin.obstacle_object_rows(sc.reference, sc.q_now, scene, beneath=True)
    finally:
        cls._esdf_clearance = orig
    assert off[0] == pytest.approx(-0.012)
    if exempt:
        assert on[0] > 0.0 and on[1]["n_beneath"] >= 1
    else:
        assert on[0] == pytest.approx(-0.012)


def test_beneath_flag_cli_and_versions():
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    with pytest.raises(SystemExit):
        ss.reject_bad_flag_combinations(ap, ap.parse_args(BASE + ["--passthrough-when-clear", "--passthrough-obstacle-beneath"]))
    a = ap.parse_args(BASE + ["--passthrough-when-clear", "--passthrough-obstacle-tolerance-mm", "10",
                              "--passthrough-obstacle-beneath"])
    c = ss.trajopt_config_from_args(a)
    assert c.sqp.passthrough_obstacle_beneath is True and c.sqp.passthrough_obstacle_tolerance == pytest.approx(0.01)
    v = "AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst2"
    assert tuple(pf.AG3S_VERSIONS[v])[-3:] == ("10", "--passthrough-obstacle-tolerance-mm", "10")[0:0] + (
        "--passthrough-obstacle-tolerance-mm", "10", "--passthrough-obstacle-beneath")
    assert tuple(pf.AG3S_VERSIONS[v + "-pts"])[-2:] == ("--held-point-model-mm", "20")
    assert "--oracle-robot-mask" not in pf.AG3S_VERSIONS["AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst2-pts"]


# ---- A'' (SL-GTTO19, 2026-11-08 02:3x): GT obstacle-object rows ----------------------------------------------------------
def test_gt_obstacle_points_decide_the_rows():
    sc = _violating(0.01)
    lin = sc.linearizer
    centres = np.asarray(lin.sphere_states(sc.reference, sc.q_now)[0])
    h, s = centres.shape[:2]
    a, b = centres[2, s // 2], centres[3, 0]
    field = _PlaneField(np.zeros(3))
    clear = np.full((h, s), 0.05)
    clear[2, s // 2] = -0.012          # nearest surface (the plane right under it) is a GT obstacle object
    clear[3, 0] = -0.03                # nearest surface is not
    base = dict(esdf=field, target_free_mask=None, target_volume=None, support_planes=None, destination_label=None,
                manipulated_spheres=None, manipulated_points=None, grasp_contact=None)
    cls = type(lin)
    orig = cls._esdf_clearance
    try:
        cls._esdf_clearance = lambda self, c, sc_: clear
        on = lin.obstacle_object_rows(sc.reference, sc.q_now, types.SimpleNamespace(
            obstacle_object_points=np.array([[a[0], a[1], 0.0]]), **base))
        none = lin.obstacle_object_rows(sc.reference, sc.q_now, types.SimpleNamespace(
            obstacle_object_points=np.zeros((0, 3)), **base))
    finally:
        cls._esdf_clearance = orig
    assert on[0] == pytest.approx(-0.012) and on[1]["n_obstacle_rows"] >= 1
    assert none[0] == float("inf") and none[1]["n_obstacle_rows"] == 0 and none[1]["n_object_points"] == 0


def test_oracle_obstacle_points_exclude_target_destination_and_support():
    from AG3S.ag3s.runtime.pipeline import AG3S

    # 1 = target (bowl at z 0.15..0.2), 3 = cabinet under it, 4 = plate (destination), 5 = mug (obstacle)
    pts = np.array([[0.0, 0.0, 0.15], [0.01, 0.0, 0.16]] * 6 + [[0.0, 0.01, 0.14]] * 12
                   + [[0.5, 0.0, 0.0]] * 5 + [[0.3, 0.3, 0.05]] * 5, float)
    codes = np.array([1] * 12 + [3] * 12 + [4] * 5 + [5] * 5)
    p = AG3S.__new__(AG3S)
    p._oracle_pending = {"labels": {"head": np.zeros((2, 2), np.uint8)},
                         "meta": {"destination": "plate", "codes": {"bowl": 1, "cabinet": 3, "plate": 4, "mug": 5}}}
    p._oracle_code_of_points = lambda fusion: codes
    fusion = types.SimpleNamespace(cloud=types.SimpleNamespace(points=pts, n_observations=len(pts), __len__=None))
    fusion.cloud = type("C", (), {"points": pts, "n_observations": len(pts), "__len__": lambda self: len(pts)})()
    out, rec = p._oracle_obstacle_points(fusion)
    assert rec["obstacle_objects"] == ["mug"] and rec["excluded"] == {"destination": "plate", "target_support": "cabinet"}
    assert len(out) == 1 and np.allclose(out[0], [0.3, 0.3, 0.05])


# ---- SL-GTTO20 (2026-11-08 03:3x): the same set from AG3S's own perception -------------------------------------------------
def test_scene_obstacle_points_drop_target_support_and_destination():
    from AG3S.ag3s.runtime.pipeline import AG3S

    def box(c, n=6, s=0.03):
        g = np.linspace(-s, s, n)
        return np.array([[c[0] + x, c[1] + y, c[2] + z] for x in g for y in g for z in g])

    bowl, cab, plate, mug = box([0.0, 0.0, 0.20], s=0.02), box([0.0, 0.0, 0.14], s=0.03), box([0.5, 0.0, 0.03]), box([0.25, 0.25, 0.05])
    cab = cab[cab[:, 2] <= 0.17]                                   # the cabinet top under the bowl (not touching it)
    pts = np.concatenate([bowl, cab, plate, mug])
    p = AG3S.__new__(AG3S)
    out, rec = p._scene_obstacle_points(pts, np.zeros(len(pts), bool), bowl, plate)
    assert out is not None and rec["n_dropped_total"] in (2, 3)            # bowl + cabinet top may merge (10 mm gap)
    assert np.linalg.norm(out.mean(axis=0) - [0.25, 0.25, 0.05]) < 0.01      # only the mug is left


def test_non_gt_version_has_no_gt_flag():
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    v = "AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts"
    fl = pf.version_server_flags(v)
    assert "--scene-obstacle-objects" in fl and not any(t.startswith("--oracle") for t in fl)
    assert "--grounding-source" not in fl and "--target-extent" not in fl
    a = ss.build_parser().parse_args(pf.version_server_argv(v) + fl)
    ss.reject_bad_flag_combinations(ss.build_parser(), a)
    assert not pf.version_requires_oracle_segmentation(v)


def test_support_under_mode_does_not_take_a_neighbour():
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S

    # bowl on the table (lowest points z 0.00..0.01), mug (code 5) standing 15 mm beside the bowl's rim, low points at z 0
    bowl = np.array([[x, y, z] for x in np.linspace(-0.05, 0.05, 6) for y in np.linspace(-0.05, 0.05, 6) for z in (0.0, 0.04)])
    mug = np.array([[0.065 + dx, 0.0 + dy, z] for dx in (0.0, 0.01) for dy in (-0.01, 0.0, 0.01) for z in (-0.01, 0.0, 0.05)] * 2)
    pts = np.concatenate([bowl, mug])
    codes = np.array([1] * len(bowl) + [5] * len(mug))
    for mode, expect in (("near", "mug"), ("under", None)):
        p = AG3S.__new__(AG3S)
        p.config = AG3SConfig().with_overrides({"clustering": {"grounding_source": "oracle", "oracle_obstacle_objects": True,
                                                               "oracle_obstacle_support": mode}})
        p._oracle_pending = {"labels": {"head": np.zeros((2, 2), np.uint8)}, "meta": {"codes": {"bowl": 1, "mug": 5}}}
        p._oracle_code_of_points = lambda fusion, c=codes: c
        fusion = types.SimpleNamespace()
        fusion.cloud = type("C", (), {"points": pts, "n_observations": len(pts), "__len__": lambda self: len(pts)})()
        out, rec = p._oracle_obstacle_points(fusion)
        assert rec["excluded"]["target_support"] == expect
        assert (out is None) == (expect == "mug")
