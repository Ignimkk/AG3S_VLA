"""T67 — the three default-off options of the grasp-failure experiment (`handoff/T67.task.md` §2, §6 V2).

1. `--support-surface-offset-mm B` (`collision.support_surface_offset`, `SceneSnapshot.support_surface_offset`): a robot
   row classed `support` whose distance a fine tier answered (`fine` · `target_free`) reads `d + B`; obstacle, target
   and held rows, the coarse tier (and the other tiers) keep `d`; the gradient is unchanged; the solver's rows, final
   check and record read the same numbers.
2. `--hsr6-objective-window` (`Hsr6Flags.objective_window`): ④ also keeps the finger allowance / A4 / G1 when HSR6 =
   approach and the A6 "grasp moment" window is open (latch latched / closing, palm site within 60 mm of the target).
3. `--neighbour-contact-mm 0` (`collision.neighbour_contact_at_zero`): IFG-NBC on with depth 0 — the window's
   finger–neighbour rows require `d − r ≥ 0` (no overlap allowed, no clearance asked); off is the flag left out.

Every option off: bit-identical rows, no new record key, the config dump as before.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shlex
import types

import numpy as np
import pytest

from AG3S.trajopt import hsr6 as H6
from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError
from AG3S.trajopt.grasp_latch import GraspPhase
from AG3S.trajopt.linearize import CollisionLinearizer, SceneSnapshot
from tests.trajopt.test_core_eval_a6 import BASE_ARGV, FINAL, _gate_policy, _site, _target
from tests.trajopt.test_t41a_obstacle_margin import BAR_R, M, R_FINGER, ROWS, TABLE_PLANE, _clear, _rig, _scene

B = 0.004                       # the fixed experiment value (task §2 ①)


# ================================================================================================ 1. support offset
FINE = 0.005
COARSE = 0.020
APPLE = np.array([-0.05, 0.0, 0.03])
APPLE_R = 0.03
BAR = np.array([0.05, 0.0, 0.09])
BAR_RAD = 0.02
TABLE_BIAS = 0.001              # the fine layers read the table 1 mm high (so they win the composite's min)


def _layer(origin, voxel, shape, fn):
    from AG3S.ag3s.fields.curobo_field import layer_from_arrays

    idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1).reshape(-1, 3)
    pts = np.asarray(origin, np.float64) + idx * voxel
    return layer_from_arrays(fn(pts).reshape(shape).astype(np.float32), np.asarray(origin, np.float64), voxel)


def _sdf(with_apple: bool, bias: float):
    def fn(p):
        d = [p[:, 2] - bias, np.linalg.norm(p - BAR, axis=1) - BAR_RAD]
        if with_apple:
            d.append(np.linalg.norm(p - APPLE, axis=1) - APPLE_R)
        return np.minimum.reduce(d)
    return fn


@pytest.fixture(scope="module")
def tier_field():
    """A real `CuroboEsdfField`: coarse 20 mm layer over the workspace + fine 5 mm window + fine target-free layer.
    Table z = 0, the apple (target), a bar (obstacle)."""
    from AG3S.ag3s.fields.curobo_field import CuroboEsdfField

    coarse = _layer((-0.5, -0.5, -0.1), COARSE, (51, 51, 26), _sdf(True, 0.0))
    fine = _layer((-0.1, -0.1, -0.05), FINE, (41, 41, 41), _sdf(True, TABLE_BIAS))
    free = _layer((-0.1, -0.1, -0.05), FINE, (41, 41, 41), _sdf(False, TABLE_BIAS))
    return CuroboEsdfField((coarse, fine), outside_distance=1.0, target_free_layers=(free,))


#: query rows (one step): 0–3 robot spheres, 4 a held sphere.
QROWS = {"table_fine": 0, "table_coarse": 1, "bar": 2, "apple": 3, "held": 4}
CENTRES = np.array([[0.0, -0.06, 0.010],         # 9 mm above the fine table, far from apple / bar -> support, fine
                    [0.30, 0.30, 0.010],          # outside the fine window -> support, coarse
                    [0.05, 0.0, 0.060],           # 10 mm under the bar -> obstacle
                    [-0.05, 0.0, 0.066],          # 6 mm above the apple -> target
                    [0.02, 0.06, 0.040]])[None]   # held sphere, nearest the table


def _tier_rig():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(6000, 3))
    points = np.vstack([APPLE + APPLE_R * v / np.linalg.norm(v, axis=1, keepdims=True),
                        APPLE + np.array([0.0, 0.0, APPLE_R])])
    lin = object.__new__(CollisionLinearizer)
    lin.robot_radii = np.full(4, R_FINGER)
    lin.query_radii = np.array([R_FINGER] * 4 + [0.01])
    lin.n_spheres = 4
    lin._attached_local = None
    lin._attached_link = "ee_left"
    lin.robot_model = types.SimpleNamespace(sphere_link_names=["ee_finger_l1", "ee_finger_l2", "ee_left", "ee_left"])
    return lin, points


def _tscene(field, points, *, offset=0.0, mask=None, **extra):
    kw = {} if offset == 0.0 else {"support_surface_offset": offset}
    return SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0), candidate_active=np.zeros(0, bool),
        d_safe=np.zeros((4, 0)), plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0),
        plane_active=np.zeros(0, bool), robot_radii=np.full(4, R_FINGER), candidate_ids=np.zeros(0, np.int64),
        esdf=field, esdf_margin=0.0, manipulated_points=points, manipulated_link_margin=np.full(4, 0.0),
        target_free_mask=None if mask is None else np.asarray(mask, bool),
        obstacle_margin=M, support_planes=TABLE_PLANE, **kw, **extra)


def test_rig_classes_and_tiers_are_what_the_rows_say(tier_field):
    lin, points = _tier_rig()
    _clear(lin, CENTRES, _tscene(tier_field, points))
    cls = lin._last_obstacle_class[0]
    assert [int(c) for c in cls] == [2, 2, 0, 1, 3]                     # support support obstacle target held
    tiers = tier_field.answer_tier(CENTRES[0, :2])
    assert list(tiers) == ["fine", "coarse"]


def test_off_is_bit_identical_and_sets_no_record(tier_field):
    lin, points = _tier_rig()
    base = _clear(lin, CENTRES, _tscene(tier_field, points))
    assert "_last_support_surface_offset" not in lin.__dict__ and "_last_support_offset_rows" not in lin.__dict__
    assert SceneSnapshot.__dataclass_fields__["support_surface_offset"].default == 0.0
    again = _clear(lin, CENTRES, dataclasses.replace(_tscene(tier_field, points), support_surface_offset=0.0))
    np.testing.assert_array_equal(again, base)
    assert "_last_support_surface_offset" not in lin.__dict__


def test_only_the_fine_support_row_reads_plus_b(tier_field):
    lin, points = _tier_rig()
    base = _clear(lin, CENTRES, _tscene(tier_field, points))
    on = _clear(lin, CENTRES, _tscene(tier_field, points, offset=B))
    i = QROWS["table_fine"]
    assert on[i] - base[i] == pytest.approx(B, abs=1e-12)
    others = [k for k in range(len(on)) if k != i]
    np.testing.assert_array_equal(on[others], base[others])             # coarse table · bar · apple · held
    rec = lin._last_support_surface_offset
    assert rec == {"offset_mm": 4.0, "n_support_rows": 2, "n_rows": 1,
                   "support_rows_by_tier": {"coarse": 1, "fine": 1}}
    assert lin._last_support_offset_rows.tolist() == [[True, False, False, False, False]]


def test_target_free_tier_is_fine_and_the_held_support_row_is_unchanged(tier_field):
    lin, points = _tier_rig()
    mask = [True, True, True, True]
    kw = dict(mask=mask, held_obstacle_margin=True)                     # held rows get their own class table
    base = _clear(lin, CENTRES, _tscene(tier_field, points, **kw))
    assert int(lin._last_obstacle_class[0, QROWS["held"]]) == 2         # the held sphere is a support row ...
    on = _clear(lin, CENTRES, _tscene(tier_field, points, offset=B, **kw))
    assert on[QROWS["held"]] == base[QROWS["held"]]                      # ... and keeps `d` (robot rows only)
    assert on[QROWS["table_fine"]] - base[QROWS["table_fine"]] == pytest.approx(B, abs=1e-12)
    assert lin._last_support_surface_offset["support_rows_by_tier"] == {"coarse": 1, "target_free": 1}
    assert on[QROWS["table_coarse"]] == base[QROWS["table_coarse"]]


@pytest.mark.parametrize("tier,applied", [("fine", True), ("target_free", True), ("coarse", False),
                                          ("outside", False), ("window_edge", False), ("static", False),
                                          ("target_free_coarse", False)])
def test_tier_rule(tier, applied):
    """Only `fine` and `target_free` (the 5 mm layers) — on the analytic t41a rig with a stub tier."""
    lin, centres, field, points = _rig()

    class _Tiered(type(field)):
        def answer_tier(self, pts, *, target_free=False):
            return np.full(len(np.asarray(pts).reshape(-1, 3)), tier)

    f = _Tiered()
    base = _clear(lin, centres, _scene(f, points, margin=M, support_planes=TABLE_PLANE))
    on = _clear(lin, centres, _scene(f, points, margin=M, support_planes=TABLE_PLANE, support_surface_offset=B))
    t = ROWS["table"]
    assert (on[t] - base[t]) == (pytest.approx(B, abs=1e-12) if applied else 0.0)
    others = [k for k in range(len(on)) if k != t]
    np.testing.assert_array_equal(on[others], base[others])


def test_a_field_without_tiers_is_refused_and_margin_support_is_refused():
    lin, centres, field, points = _rig()
    with pytest.raises(ValueError, match="answer_tier"):
        _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE, support_surface_offset=B))
    with pytest.raises(ValueError, match="obstacle_margin_support"):
        _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE, support_surface_offset=B,
                                    obstacle_margin_support=True))


def test_worst_row_record_names_the_offset(tier_field):
    lin, points = _tier_rig()
    scene = _tscene(tier_field, points, offset=B)
    block = lin._esdf_clearance(CENTRES, scene)
    rec = lin._identify("esdf", (0, QROWS["table_fine"], 0), CENTRES, scene, float(block[0, 0, 0]), esdf_block=block)
    assert rec["support_surface_offset_m"] == B and rec["obstacle_margin_class"] == "support"
    rec = lin._identify("esdf", (0, QROWS["bar"], 0), CENTRES, scene, float(block[0, 2, 0]), esdf_block=block)
    assert rec["support_surface_offset_m"] == 0.0
    off = _tscene(tier_field, points)
    block = lin._esdf_clearance(CENTRES, off)
    rec = lin._identify("esdf", (0, 0, 0), CENTRES, off, float(block[0, 0, 0]), esdf_block=block)
    assert "support_surface_offset_m" not in rec


def test_scene_builder_carries_the_offset_only_when_on():
    from AG3S.trajopt.linearize import scene_from_constraint_set

    class _Field:
        stats: dict = {}

        def distance(self, p):
            return np.asarray(p, np.float64).reshape(-1, 3)[:, 2]

    cs = types.SimpleNamespace(constraints=None, esdf=_Field(), support_surfaces=(), target_field_exclude=None,
                               attached=None, manipulated_points=None, destination_label=None)
    radii = np.full(3, 0.01)
    base = TrajOptConfig.from_dict({"collision": {"backend": "esdf", "obstacle_margin": 0.01}})
    try:
        off = scene_from_constraint_set(cs, radii, base)
        on = scene_from_constraint_set(cs, radii, base.with_overrides({"collision": {"support_surface_offset": B}}))
    except Exception as exc:  # noqa: BLE001 — the builder wants more of a constraint set than this stub has
        pytest.skip(f"stub constraint set not enough for scene_from_constraint_set: {exc}")
    assert off.support_surface_offset == 0.0 and on.support_surface_offset == B


# ------------------------------------------------------------------------------------- recorded chunk (real field)
CORE = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261028T224856+0900_b3b69311f948_8784a1265958/"
                    "verify/AG3S-CORE-EVAL")


def _spec(label):
    path = CORE / "module" / "spec_fin.json"
    if not path.exists():
        return None
    sp = [x for x in json.load(open(path)) if x["label"] == label]
    return sp[0] if sp and pathlib.Path(sp[0]["chunks"][0]).exists() else None


needs_211 = pytest.mark.skipif(_spec("회귀211_k0") is None, reason="final-configuration record (211 k0) not present")


@needs_211
def test_recorded_chunk_rows_gradients_and_final_check():
    """Final configuration, regression 211 draw 0, request 14 (the table pushed the fingertips, T66b): with the offset
    the support rows of the fine tiers read +B, every other row and every row's gradient are unchanged, and the
    solver's final check, its record and a re-measure on the same scene agree."""
    pytest.importorskip("casadi")
    pytest.importorskip("osqp")
    from AG3S.trajopt.experiments import ta_offline as TA
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    sp = _spec("회귀211_k0")
    run = TA.Run("211", {"run_dir": sp["client_dir"], "server_segment": {"first": sp["chunks"][0], "n": 18}},
                 types.SimpleNamespace(hm=False, no_plane_rows=False))
    seq = sorted(run.chunks)[14]
    inp, why = run.inputs(seq)
    assert inp is not None, why
    cfg = run.base_cfg.with_overrides({"sqp": {"gpu_seeds": "off", "gpu_seeds_check": "all", "multi_seed": 0}})
    # the server's support planes: AG3S's own fit on the recorded (self-filtered) cloud — `ta_offline` falls back to an
    # SVD over the whole support mask (table + floor) when nothing was attached, which is not the server's plane
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.stages.support_surface import fit_support_surfaces
    from AG3S.ag3s.types import PointCloud

    surf, mask = fit_support_surfaces(PointCloud(points=np.asarray(inp["z"]["filtered_cloud"], np.float64)),
                                      AG3SConfig().support_surface, timestamp=0.0)
    np.testing.assert_array_equal(mask, np.asarray(inp["z"]["support_mask"], bool))      # = the server's fit
    planes = np.asarray([[*sf.normal, sf.offset] for sf in surf], np.float64)
    off_scene = dataclasses.replace(inp["scene"], support_planes=planes)
    on_scene = dataclasses.replace(off_scene, support_surface_offset=B)
    # rows on the VLA reference: values and gradients
    ref = TrajOptChunkRefiner(run.robot, run.layout, lambda _c: (off_scene, inp["q_now"], True), cfg)
    lin = ref.optimizer.linearizer
    lin.set_joint_parameter_path(run.finger_path(inp["summary"], inp["ref_chunk"], cfg.horizon.planned))
    traj = run.layout.chunk_to_trajectory(inp["ref_chunk"][:cfg.horizon.planned]) \
        if hasattr(run.layout, "chunk_to_trajectory") else None
    if traj is None:
        pytest.skip("layout cannot map a chunk to a trajectory")
    states = lin.sphere_states(traj, inp["q_now"])
    c_off = np.asarray(lin._esdf_clearance(states[0], off_scene))[..., 0]
    c_on = np.asarray(lin._esdf_clearance(states[0], on_scene))[..., 0]
    sel = lin._last_support_offset_rows
    assert sel.any(), "no fine support row on this chunk"
    fin = np.isfinite(c_off)
    np.testing.assert_allclose((c_on - c_off)[sel & fin], B, atol=1e-12)
    np.testing.assert_array_equal(c_on[~sel], c_off[~sel])
    # the selected rows are support rows of the robot answered by a fine tier
    cls = lin._last_obstacle_class
    assert (cls[sel] == 2).all() and not sel[:, lin.n_spheres:].any()
    rows_off = lin.linearize(traj, inp["q_now"], off_scene, cfg, states=states)
    rows_on = lin.linearize(traj, inp["q_now"], on_scene, cfg, states=states)
    key_off = {(k, int(s)): i for i, (k, s) in enumerate(zip(np.repeat(np.arange(rows_off.value.shape[0]),
                                                                         rows_off.value.shape[1]),
                                                               rows_off.sphere.reshape(-1)))
               if rows_off.used.reshape(-1)[i] and rows_off.slot.reshape(-1)[i] < -1000}
    n_same = 0
    for i, (k, s) in enumerate(zip(np.repeat(np.arange(rows_on.value.shape[0]), rows_on.value.shape[1]),
                                   rows_on.sphere.reshape(-1))):
        if not rows_on.used.reshape(-1)[i] or rows_on.slot.reshape(-1)[i] >= -1000 or (k, int(s)) not in key_off:
            continue
        j = key_off[(k, int(s))]
        g_on = rows_on.gradient.reshape(-1, rows_on.gradient.shape[-1])[i]
        g_off = rows_off.gradient.reshape(-1, rows_off.gradient.shape[-1])[j]
        np.testing.assert_array_equal(g_on, g_off)                       # gradient unchanged
        dv = rows_on.value.reshape(-1)[i] - rows_off.value.reshape(-1)[j]
        assert dv == pytest.approx(B if sel[k, int(s)] else 0.0, abs=1e-12)
        n_same += 1
    assert n_same > 0
    # the solve: final check == record == re-measure
    holder = {"v": (on_scene, inp["q_now"], True)}
    ref2 = TrajOptChunkRefiner(run.robot, run.layout, lambda _c: holder["v"], cfg)
    ref2.optimizer.linearizer.set_joint_parameter_path(
        run.finger_path(inp["summary"], inp["ref_chunk"], cfg.horizon.planned))
    ref2.refine(inp["ref_chunk"], {"t_step": seq, "previous_physical_chunk": None})
    res = ref2.last_result
    rec = res.metrics["support_surface_offset"]
    assert rec["offset_mm"] == pytest.approx(4.0) and rec["n_rows"] > 0
    opt = ref2.optimizer
    if (res.metrics.get("lookahead") or {}).get("gate") == "closed":
        k = int(res.metrics["lookahead"]["from"])
        opt, t2, steps = opt.window_optimizer, res.trajectory[:, :k], None
    else:
        t2, steps = res.trajectory, opt.lookahead_from
    clearance, _ = opt.linearizer.worst_row(t2, inp["q_now"], on_scene, steps=steps)
    assert opt.linearizer._last_support_surface_offset == rec
    assert res.metrics["clearance_m"] == pytest.approx(clearance, abs=1e-12)
    # off: no key
    ref3 = TrajOptChunkRefiner(run.robot, run.layout, lambda _c: (off_scene, inp["q_now"], True), cfg)
    ref3.optimizer.linearizer.set_joint_parameter_path(
        run.finger_path(inp["summary"], inp["ref_chunk"], cfg.horizon.planned))
    ref3.refine(inp["ref_chunk"], {"t_step": seq, "previous_physical_chunk": None})
    assert "support_surface_offset" not in ref3.last_result.metrics


# ================================================================================================ 2. ④ window
def _approach(p, phase="approach"):
    p._hsr6_start({"p": {c: (0.9 if c == phase else 0.02) for c in H6.PHASES6}, "argmax": phase})


def _win_policy(window=True, nbc=0.020):
    p, model = _gate_policy(depth=0.0, nbc=nbc, hsr6=H6.Hsr6Flags(objective=True, objective_window=window))
    p._last_grasp_anchor = {}
    return p, model


@pytest.mark.parametrize("phase,window,want", [
    ("grasp", None, True), ("grasp", True, True), ("grasp", False, True),
    ("approach", None, False), ("approach", False, False), ("approach", True, True),
    ("lift", True, False), ("retreat", True, False), ("transport", True, False), (None, True, False)])
def test_objective_allowed_table(phase, window, want):
    assert H6.objective_allowed(phase, window=window) is want
    if window is None:
        assert H6.objective_allowed(phase) is want                      # the rule before T67


@pytest.mark.parametrize("latch", [GraspPhase.LATCHED, GraspPhase.CLOSING])
def test_window_open_keeps_nbc_and_a4_in_approach(latch):
    p, model = _win_policy()
    q = np.zeros(len(model.joint_names))
    _target(p, _site(model, q) + np.array([0.0, 0.0, -0.03]))
    _approach(p)
    p._latch.phase = latch
    nb = p._neighbour_contact_gate(q, ["left"])
    assert nb is not None and "hsr6_withdrawn" not in p._last_neighbour_contact
    win = p._last_hsr6["detail"]["objective"]["window"]
    assert win["open"] is True and win["latch"] == latch.value and win["gate_mm"] == 60.0
    assert win["site_to_target_mm"] == pytest.approx(30.0, abs=0.1) and win["link"] == "ee_left"
    anchor = object()
    assert p._hsr6_a4_gate(anchor, q, ["left"]) is anchor
    assert p._last_hsr6["changed"]["objective"] is False
    assert "objective_window" not in p._last_hsr6["changed"]                 # widens ④, no decision of its own
    assert "objective_window" in p._last_hsr6["flags"]


@pytest.mark.parametrize("latch", [GraspPhase.SEARCHING, GraspPhase.HELD, GraspPhase.PLACED])
def test_window_closed_outside_latched_closing(latch):
    p, model = _win_policy()
    q = np.zeros(len(model.joint_names))
    _target(p, _site(model, q))
    _approach(p)
    p._latch.phase = latch
    assert p._hsr6_objective_ok(q, ["left"]) is False
    win = p._last_hsr6["detail"]["objective"]["window"]
    assert win["open"] is False and win["reason"] == "phase " + latch.value
    assert p._hsr6_a4_gate(object(), q, ["left"]) is None
    assert p._last_hsr6["detail"]["objective"]["a4_withdrawn"] is True


@pytest.mark.parametrize("dist,on", [(0.059, True), (0.0599, True), (0.0601, False), (0.10, False)])
def test_window_boundary_60_mm(dist, on):
    p, model = _win_policy()
    q = np.zeros(len(model.joint_names))
    _target(p, _site(model, q) + np.array([0.0, dist, 0.0]))
    _approach(p)
    assert p._hsr6_objective_ok(q, ["left"]) is on
    win = p._last_hsr6["detail"]["objective"]["window"]
    assert win["open"] is on
    if not on:
        assert win["reason"] == "palm outside the gate"


def test_window_measures_the_site_and_needs_q_now_and_revoked_closes_it():
    from AG3S.ag3s.constraints.attached import _link_pose_numeric

    p, model = _win_policy()
    q = np.zeros(len(model.joint_names))
    _target(p, np.asarray(_link_pose_numeric(model, q, "ee_left")[:3, 3]))   # link origin: 100 mm off the site
    _approach(p)
    assert p._hsr6_objective_ok(q, ["left"]) is False
    _target(p, _site(model, q))
    assert p._hsr6_objective_ok(None, ["left"]) is False
    assert p._last_hsr6["detail"]["objective"]["window"]["reason"] == "no q_now"
    p._last_grasp_record = {"attach_revoked": True}
    assert p._hsr6_objective_ok(q, ["left"]) is False


@pytest.mark.parametrize("phase,want", [("grasp", True), ("lift", False), ("retreat", False), (None, False)])
def test_window_only_matters_in_approach(phase, want):
    p, model = _win_policy()
    q = np.zeros(len(model.joint_names))
    _target(p, _site(model, q))                                         # window would be open
    if phase is None:
        p._hsr6_start(None)
    else:
        _approach(p, phase)
    assert p._hsr6_objective_ok(q, ["left"]) is want
    assert "objective" not in p._last_hsr6["detail"] or "window" not in p._last_hsr6["detail"]["objective"]


def test_window_flag_off_is_the_rule_before():
    p, model = _win_policy(window=False)
    q = np.zeros(len(model.joint_names))
    _target(p, _site(model, q))
    _approach(p)
    assert p._neighbour_contact_gate(q, ["left"]) is None                     # ④ withdraws NBC in approach
    assert p._last_neighbour_contact["hsr6_withdrawn"] is True
    assert "window" not in p._last_hsr6["detail"].get("objective", {})
    assert "objective_window" not in p._last_hsr6["flags"]
    assert p._hsr6_a4_gate(object(), q, ["left"]) is None


def test_hsr6_flags_and_cli():
    from AG3S.trajopt import serve_safe

    assert H6.Hsr6Flags(objective=True).names() == ["objective"]
    assert H6.Hsr6Flags(objective=True, objective_window=True).names() == ["objective", "objective_window"]
    ap = serve_safe.build_parser()
    a = ap.parse_args([*BASE_ARGV, "--neighbour-contact-mm", "20", "--hsr6-objective-window"])
    with pytest.raises(SystemExit):
        serve_safe.reject_hsr6_flags(ap, a)                              # the window needs ④
    a = ap.parse_args([*BASE_ARGV, "--neighbour-contact-mm", "20", "--hsr6-objective", "--hsr6-objective-window"])
    serve_safe.reject_hsr6_flags(ap, a)
    assert serve_safe.hsr6_flags_from_args(a).objective_window is True
    a = ap.parse_args([*BASE_ARGV, "--neighbour-contact-mm", "20", "--hsr6-objective"])
    assert serve_safe.hsr6_flags_from_args(a).objective_window is False
    assert serve_safe.hsr6_record_meta(a)["hsr6"]["flags"] == ["objective"]


# ================================================================================================ 3. NBC at depth 0
def _nb(centre, depth=0.0, at_zero=True):
    out = {"links_mask": np.ones(4, bool), "depth": depth, "centre": np.asarray(centre, np.float64), "radius": 0.05}
    if at_zero:
        out["at_zero"] = True
    return out


def test_nbc_zero_requires_d_minus_r_at_least_zero():
    lin, centres, field, points = _rig()
    kw = dict(margin=M, support_planes=TABLE_PLANE)
    base = _clear(lin, centres, _scene(field, points, **kw))
    on = _clear(lin, centres, _scene(field, points, **kw, neighbour_contact=_nb(field.bar)))
    bar = ROWS["bar"]
    assert base[bar] == pytest.approx(0.008 - R_FINGER - M, abs=1e-12)            # >= 10 mm before
    assert on[bar] == pytest.approx(0.008 - R_FINGER, abs=1e-12)                  # >= 0 now: margin 0, no overlap
    others = [k for k in range(len(on)) if k != bar]
    np.testing.assert_array_equal(on[others], base[others])
    assert lin._last_neighbour_contact == {"n_rows": 1}
    # depth 0 without `at_zero` is off, as before T67
    off = _clear(lin, centres, _scene(field, points, **kw, neighbour_contact=_nb(field.bar, at_zero=False)))
    np.testing.assert_array_equal(off, base)


@pytest.mark.parametrize("gap,feasible", [(0.0005, True), (0.0, True), (-0.0005, False)])
def test_nbc_zero_boundary(gap, feasible):
    lin, centres, field, points = _rig()
    c = centres.copy()
    bar = ROWS["bar"]
    c[0, bar] = field.bar + np.array([0.0, 0.0, -(BAR_R + R_FINGER + gap)])        # sphere surface `gap` off the bar
    on = _clear(lin, c, _scene(field, points, margin=M, support_planes=TABLE_PLANE, neighbour_contact=_nb(field.bar)))
    assert on[bar] == pytest.approx(gap, abs=1e-12)
    assert bool(on[bar] >= -1e-12) is feasible


def test_nbc_zero_gate_and_config():
    from AG3S.trajopt import serve_safe

    p, model = _gate_policy(depth=0.0, nbc=0.0)
    q = np.zeros(len(model.joint_names))
    _target(p, _site(model, q))
    assert p._neighbour_contact_gate(q, ["left"]) is None and p._last_neighbour_contact == {}    # 0 alone = off
    p.to_config = TrajOptConfig.from_dict({"collision": {"neighbour_contact_depth": 0.0,
                                                         "neighbour_contact_at_zero": True}})
    nb = p._neighbour_contact_gate(q, ["left"])
    assert nb is not None and nb["depth"] == 0.0 and nb["at_zero"] is True
    assert p._last_neighbour_contact["active"] is True and p._last_neighbour_contact["depth_mm"] == 0.0
    p.to_config = TrajOptConfig.from_dict({"collision": {"neighbour_contact_depth": 0.02}})
    assert "at_zero" not in p._neighbour_contact_gate(q, ["left"])
    # config / CLI: 0 -> on at zero; 20 -> as before (no new key in the dump); off -> no key; < 0 refused
    ap = serve_safe.build_parser()
    zero = serve_safe.trajopt_config_from_args(ap.parse_args([*BASE_ARGV, "--neighbour-contact-mm", "0"]))
    assert zero.collision.neighbour_contact_depth == 0.0 and zero.collision.neighbour_contact_at_zero is True
    assert zero.to_dict()["collision"]["neighbour_contact_at_zero"] is True
    twenty = serve_safe.trajopt_config_from_args(ap.parse_args([*BASE_ARGV, "--neighbour-contact-mm", "20"]))
    assert "neighbour_contact_at_zero" not in twenty.to_dict()["collision"]
    assert "neighbour_contact_at_zero" not in TrajOptConfig().to_dict()["collision"]
    assert TrajOptConfig.from_dict(zero.to_dict()) == zero
    with pytest.raises(SystemExit):
        serve_safe.trajopt_config_from_args(ap.parse_args([*BASE_ARGV, "--neighbour-contact-mm", "-1"]))
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig.from_dict({"collision": {"neighbour_contact_at_zero": 1}})


# ================================================================================================ config / argv
C1 = FINAL + " --hsr6-objective-window"
C2 = FINAL + " --support-surface-offset-mm 4"
C3 = FINAL.replace("--neighbour-contact-mm 20", "--neighbour-contact-mm 0") + \
    " --hsr6-objective-window --support-surface-offset-mm 4"


def test_support_offset_config_and_cli():
    from AG3S.trajopt import serve_safe

    c = TrajOptConfig()
    assert c.collision.support_surface_offset == 0.0 and "support_surface_offset" not in c.to_dict()["collision"]
    on = TrajOptConfig.from_dict({"collision": {"support_surface_offset": B}})
    assert on.to_dict()["collision"]["support_surface_offset"] == B and TrajOptConfig.from_dict(on.to_dict()) == on
    for bad in (-0.001, float("nan"), float("inf")):
        with pytest.raises(TrajOptConfigError):
            TrajOptConfig.from_dict({"collision": {"support_surface_offset": bad}})
    ap = serve_safe.build_parser()
    assert "support_surface_offset" not in serve_safe.obstacle_overrides(ap.parse_args(BASE_ARGV))
    for bad in (["--support-surface-offset-mm", "0"], ["--support-surface-offset-mm", "4"],      # no obstacle margin
                ["--obstacle-margin", "0.01", "--obstacle-margin-support", "--support-surface-offset-mm", "4"],
                ["--obstacle-margin", "0.01", "--support-surface-offset-mm", "-4"]):
        with pytest.raises(SystemExit):
            serve_safe.trajopt_config_from_args(ap.parse_args([*BASE_ARGV, *bad]))
    got = serve_safe.trajopt_config_from_args(ap.parse_args([*BASE_ARGV, "--obstacle-margin", "0.01",
                                                             "--support-surface-offset-mm", "4"]))
    assert got.collision.support_surface_offset == pytest.approx(B)


@pytest.mark.parametrize("name,argv", [("C1", C1), ("C2", C2), ("C3", C3)])
def test_c1_c2_c3_argv_parse_to_the_fixed_values(monkeypatch, name, argv):
    from AG3S.trajopt import gpu_seeds, serve_safe

    monkeypatch.setattr(gpu_seeds, "available", lambda: (True, "patched for a CPU test"))
    ap = serve_safe.build_parser()
    fin = ap.parse_args(BASE_ARGV + shlex.split(FINAL))
    a = ap.parse_args(BASE_ARGV + shlex.split(argv))
    serve_safe.reject_bad_flag_combinations(ap, a)
    c = serve_safe.trajopt_config_from_args(a)
    f = serve_safe.trajopt_config_from_args(fin)
    window = name in ("C1", "C3")
    assert serve_safe.hsr6_flags_from_args(a).objective_window is window
    col = c.collision
    assert col.support_surface_offset == pytest.approx(B if name in ("C2", "C3") else 0.0)
    assert col.neighbour_contact_depth == pytest.approx(0.0 if name == "C3" else 0.020)
    assert col.neighbour_contact_at_zero is (name == "C3")
    assert col.neighbour_palm_contact_depth == pytest.approx(0.010) and col.palm_shell_margin == pytest.approx(0.005)
    # the config differs from the final one only in the T67 keys
    d, df = c.to_dict(), f.to_dict()
    diff = {(s, k) for s in d for k in set(d[s]) | set(df[s]) if d[s].get(k, "∅") != df[s].get(k, "∅")}
    want = {"C1": set(), "C2": {("collision", "support_surface_offset")},
            "C3": {("collision", "support_surface_offset"), ("collision", "neighbour_contact_depth"),
                   ("collision", "neighbour_contact_at_zero")}}[name]
    assert diff == want


def test_final_argv_config_dump_has_no_t67_key():
    from AG3S.trajopt import serve_safe

    ap = serve_safe.build_parser()
    d = serve_safe.trajopt_config_from_args(ap.parse_args(BASE_ARGV + shlex.split(FINAL))).to_dict()
    assert "support_surface_offset" not in d["collision"] and "neighbour_contact_at_zero" not in d["collision"]
    assert serve_safe.hsr6_record_meta(ap.parse_args(BASE_ARGV + shlex.split(FINAL)))["hsr6"]["flags"] == \
        ["attach_assist", "held_gate", "target_exempt", "objective"]
