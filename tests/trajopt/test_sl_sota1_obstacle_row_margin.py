"""SL-SOTA1 (dev-57, 2026-11-11): the obstacle-object rows' own margin (`--obstacle-row-margin-mm`), slack priority
(`--obstacle-row-priority`) and surface tolerance (`--obstacle-surface-tol-mm`). All default off.

E1a SafeLIBERO pe123458 (outputs/2026-11-11_SOTA_개선/offline): the passed chunk may sit 0 mm from the modelled surface
(10 mm floor, 10 mm tolerance) while the world model misses 12-27 mm of an obstacle under the hand and the executed motion
is 10-15 mm off the QP plan; the L1 slack trades one obstacle row against the contact-surface rows.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

osqp = pytest.importorskip("osqp")

from AG3S.trajopt.linearize import ESDF_SLOT  # noqa: E402


def _sqp(**kw):
    base = dict(passthrough_obstacle_tolerance=0.0, passthrough_obstacle_full_horizon=True, passthrough_obstacle_beneath=False,
                passthrough_obstacle_class_tests=False, obstacle_row_margin=None, obstacle_row_priority=1.0,
                obstacle_object_surface_tolerance=None, obstacle_row_patience=None, ee_action_gate_rows=False,
                passthrough_tolerance=0.1)
    base.update(kw)
    return types.SimpleNamespace(**base)


# --- pass-through test -------------------------------------------------------------------------------------------------
class _RowsLin:
    def __init__(self, worst):
        self.worst, self.kwargs = worst, None

    def obstacle_object_rows(self, reference, q_now, scene, steps=None, **kw):
        self.kwargs = dict(kw, steps=steps)
        return self.worst, {"step": 0, "link": "panda_hand", "clearance_m": self.worst, "n_obstacle_rows": 3}


def _refuses(worst, **kw):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = TrajectoryOptimizer.__new__(TrajectoryOptimizer)
    opt.config = types.SimpleNamespace(sqp=_sqp(**kw))
    opt.linearizer = _RowsLin(worst)
    opt.lookahead_from = 5
    record: dict = {}
    return opt._obstacle_row_refuses(None, None, None, record), record, opt.linearizer.kwargs


def test_passthrough_without_margin_is_as_before():
    refused, record, kwargs = _refuses(0.012)
    assert not refused and record["obstacle_rows"]["violation_m"] == 0.0 and "margin_m" not in record["obstacle_rows"]
    assert kwargs == {"beneath": False, "steps": None}          # no new keyword reaches the linearizer when off
    refused, record, _ = _refuses(-0.004)
    assert refused and record["obstacle_rows"]["violation_m"] == pytest.approx(0.004)


def test_passthrough_margin_refuses_a_row_inside_it():
    refused, record, _ = _refuses(0.012, obstacle_row_margin=0.015)
    assert refused and record["obstacle_rows"]["violation_m"] == pytest.approx(0.003)
    assert record["obstacle_rows"]["margin_m"] == pytest.approx(0.015)
    assert not _refuses(0.016, obstacle_row_margin=0.015)[0]
    assert not _refuses(0.012, obstacle_row_margin=0.015, passthrough_obstacle_tolerance=0.005)[0]   # 3 mm <= 5 mm
    assert not _refuses(float("inf"), obstacle_row_margin=0.015)[0]                                   # no obstacle row


def test_surface_tolerance_and_class_tests_reach_the_linearizer():
    _, _, kwargs = _refuses(0.1, obstacle_object_surface_tolerance=0.04, passthrough_obstacle_class_tests=True,
                            passthrough_obstacle_beneath=True)
    assert kwargs == {"beneath": True, "class_tests": True, "surface_tol": 0.04, "steps": None}


# --- action QP ---------------------------------------------------------------------------------------------------------
class _Lin:
    """One step, one joint q. Row 0 (query 0, an obstacle object): clearance = q. Row 1 (query 1, a contact surface):
    clearance = 0.02 - q. Feasible without a margin (0 <= q <= 0.02); a 30 mm margin on row 0 conflicts with row 1."""

    n_spheres = 2

    def __init__(self):
        self.floors = []

    def sphere_states(self, traj, q_now):
        return np.zeros((1, 2, 3)), np.zeros((1, 2, 3, 1))

    def obstacle_row_floor(self, centres, scene, margin, **kw):
        floor = np.zeros((1, 2))
        floor[0, 0] = margin
        return floor, np.array([[True, False]])

    def linearize(self, traj, q_now, scene, cfg, states=None, row_floor=None):
        q = float(traj[0, 0])
        value = np.array([[q, 0.02 - q]])
        self.floors.append(None if row_floor is None else np.array(row_floor))
        if row_floor is not None:
            value = value - np.asarray(row_floor)
        grad = np.array([[[1.0], [-1.0]]])
        return types.SimpleNamespace(value=value, gradient=grad, used=np.ones((1, 2), bool),
                                     slot=np.full((1, 2), ESDF_SLOT), sphere=np.array([[0, 1]]))

    def full_violation(self, traj, q_now, scene):
        q = float(traj[0, 0])
        return max(0.0, -q, q - 0.02)


def _solve(**kw):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = TrajectoryOptimizer.__new__(TrajectoryOptimizer)
    opt.config = types.SimpleNamespace(reduction=types.SimpleNamespace(linearization_backoff=0.0),
                                       safety=types.SimpleNamespace(violation_tolerance=1e-6), sqp=_sqp(**kw))
    opt.linearizer = _Lin()
    opt.ee_action_map = {"M": np.ones((1, 1)), "a_ref": np.zeros((1, 1))}
    opt._finish = lambda traj, *a, **k: types.SimpleNamespace(trajectory=traj, metrics={})
    res = opt._solve_ee_action(np.zeros((1, 1)), np.zeros(1), None, None, 0.0, None, [], 0.0, True, None)
    return float(opt.last_ee_action["delta_action"][0, 0]), res.metrics["ee_action_qp"], opt.linearizer


def test_action_qp_without_margin_is_untouched():
    da, rec, lin = _solve()
    assert abs(da) < 1e-6 and all(f is None for f in lin.floors)
    assert "obstacle_row_margin_m" not in rec and "n_obstacle_rows" not in rec


def test_action_qp_margin_alone_leaves_the_obstacle_row_short():
    da, rec, lin = _solve(obstacle_row_margin=0.03)
    assert lin.floors[0] is not None and lin.floors[0][0, 0] == pytest.approx(0.03) and lin.floors[0][0, 1] == 0.0
    assert rec["obstacle_row_margin_m"] == pytest.approx(0.03) and rec["n_obstacle_rows"][0] == 1
    # equal L1 weights: every q in [0.02, 0.03] has the same summed slack, the obstacle-object row is not preferred
    assert 0.02 - 2e-4 <= da < 0.029


def test_action_qp_priority_meets_the_obstacle_row():
    da, rec, _ = _solve(obstacle_row_margin=0.03, obstacle_row_priority=10.0)
    assert da == pytest.approx(0.03, abs=2e-4)          # the obstacle-object row is met, the contact row takes the slack
    assert rec["obstacle_row_priority"] == 10.0 and rec["slack_max_m"] == pytest.approx(0.01, abs=2e-4)


def test_classification_failure_falls_back_to_the_plain_rows():
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = TrajectoryOptimizer.__new__(TrajectoryOptimizer)
    opt.config = types.SimpleNamespace(sqp=_sqp(obstacle_row_margin=0.03))
    lin = _Lin()
    lin.obstacle_row_floor = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no field"))
    opt.linearizer = lin
    rec: dict = {}
    rows, mask = opt._linearize_with_obstacle_floor(np.zeros((1, 1)), np.zeros(1), None, 0.03, rec)
    assert mask is None and lin.floors == [None] and "RuntimeError" in rec["obstacle_floor_error"]


# --- linearizer --------------------------------------------------------------------------------------------------------
def test_row_floor_layout_matches_linearize():
    from AG3S.trajopt.linearize import CollisionLinearizer

    lin = CollisionLinearizer.__new__(CollisionLinearizer)
    mask = np.array([[True, False, True], [False, False, True]])
    lin.obstacle_object_mask = lambda centres, scene, **kw: (np.zeros(mask.shape), mask.copy(), 0, None)
    type(lin).n_spheres = property(lambda self: 2)
    try:
        scene = types.SimpleNamespace(n_slots=4, n_planes=1)
        floor, got = lin.obstacle_row_floor(np.zeros((2, 3, 3)), scene, 0.015)
    finally:
        del type(lin).n_spheres
    assert floor.shape == (2, 2 * (4 + 1) + 3) and np.array_equal(got, mask)       # candidates, planes, then the ESDF rows
    assert np.all(floor[:, :10] == 0.0) and np.allclose(floor[:, 10:], np.where(mask, 0.015, 0.0))
    try:   # SL-SOTA4: `other` is the floor of the ESDF rows that are not obstacle-object rows (candidates / planes stay 0)
        type(lin).n_spheres = property(lambda self: 2)
        floor, _ = lin.obstacle_row_floor(np.zeros((2, 3, 3)), scene, 0.015, other=-0.1)
    finally:
        del type(lin).n_spheres
    assert np.all(floor[:, :10] == 0.0) and np.allclose(floor[:, 10:], np.where(mask, 0.015, -0.1))
    lin.obstacle_object_mask = lambda centres, scene, **kw: None
    assert lin.obstacle_row_floor(np.zeros((2, 3, 3)), scene, 0.015) == (None, None)


# --- flags, config, versions -------------------------------------------------------------------------------------------
def test_config_fields_default_off_and_validated():
    from AG3S.trajopt.config import SqpConfig, TrajOptConfigError

    cfg = SqpConfig()
    assert cfg.obstacle_row_margin is None and cfg.obstacle_row_priority == 1.0 and cfg.obstacle_object_surface_tolerance is None
    cfg.validate()
    with pytest.raises(TrajOptConfigError):
        SqpConfig(obstacle_row_margin=0.015).validate()                             # needs the obstacle-row tolerance
    with pytest.raises(TrajOptConfigError):
        SqpConfig(passthrough_obstacle_tolerance=0.0, obstacle_row_margin=-0.001).validate()
    with pytest.raises(TrajOptConfigError):
        SqpConfig(obstacle_row_priority=0.5).validate()
    SqpConfig(passthrough_obstacle_tolerance=0.0, obstacle_row_margin=0.015, obstacle_row_priority=10.0,
              obstacle_object_surface_tolerance=0.04).validate()


def test_flag_plumbing():
    from AG3S.trajopt import serve_safe as ss

    from tests.trajopt.test_t88_passthrough_when_clear import BASE as ARGV

    ap = ss.build_parser()
    a = ap.parse_args(ARGV)
    assert a.obstacle_row_margin_mm is None and a.obstacle_row_priority is None and a.obstacle_surface_tol_mm is None
    assert not {"obstacle_row_margin", "obstacle_row_priority", "obstacle_object_surface_tolerance"} & set(ss.passthrough_overrides(a))
    a = ap.parse_args(ARGV + ["--obstacle-row-margin-mm", "15", "--obstacle-row-priority", "10", "--obstacle-surface-tol-mm", "40"])
    out = ss.passthrough_overrides(a)
    assert out["obstacle_row_margin"] == pytest.approx(0.015) and out["obstacle_row_priority"] == 10.0
    assert out["obstacle_object_surface_tolerance"] == pytest.approx(0.04)


def test_versions():
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv

    base = "AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-"
    assert len(pf.SOTA_OM_VERSIONS) == 16 and set(pf.SOTA_OM_VERSIONS) <= set(pf.SOTA_VERSIONS) <= set(pf.GT15_VERSIONS)
    for b in (base + "pe123458-to0", base + "pe12345811-to0"):
        for suffix, flags in pf._SOTA_OM_SUFFIXES.items():
            assert tuple(pf.AG3S_VERSIONS[b + suffix]) == tuple(pf.AG3S_VERSIONS[b]) + tuple(flags)
            assert tuple(pf.AG3S_VERSIONS[b + suffix + "-tro"]) == tuple(pf.AG3S_VERSIONS[b + suffix]) + (
                "--ee-action-qp-translation-only",)
            assert pf.version_server_flags(b + suffix + "-tro")
    v = pf.AG3S_VERSIONS[base + "pe12345811-to0-om15p10s40-tro"]
    assert "--ee-action-qp" in v and "--scene-obstacle-target-cut" in v
    for d in ("obstacle_row_margin_mm", "obstacle_row_priority", "obstacle_surface_tol_mm"):
        assert d not in srv.UNPORTED_FINAL_CONFIG_DESTS
    fam = pf.version_flag_values(pf.parse_v11ab_args())                             # a V11AB record gains no key
    assert not {"obstacle_row_margin_mm", "obstacle_row_priority", "obstacle_surface_tol_mm", "palm_shell"} & set(fam)


def test_real_linearizer_applies_the_floor_to_its_esdf_block(monkeypatch):
    """The floor `obstacle_row_floor` builds has the width `linearize` concatenates (candidate, plane, ESDF rows), and the
    obstacle-object rows come back `margin` lower while every other row keeps its value."""
    pytest.importorskip("casadi")
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8)
    lin = sc.linearizer
    states = lin.sphere_states(sc.reference, sc.q_now)
    centres = np.asarray(states[0])
    h, n_query = centres.shape[:2]
    clear = np.full((h, n_query), 0.012)                 # every ESDF row 12 mm above its floor
    mask = np.zeros((h, n_query), bool)
    mask[:, 0] = True                                    # query 0 is next to an obstacle object
    cls = type(lin)
    monkeypatch.setattr(cls, "_esdf_clearance", lambda self, c, s: clear)
    monkeypatch.setattr(cls, "_esdf_directions", lambda self, pts, rows, scene, n: np.tile([0.0, 0.0, 1.0], (len(pts), 1)))
    monkeypatch.setattr(cls, "obstacle_object_mask", lambda self, c, s, **k: (clear, mask, 0, None))
    floor, got = lin.obstacle_row_floor(centres, sc.scene, 0.015)
    assert floor.shape == (h, lin.n_spheres * (sc.scene.n_slots + sc.scene.n_planes) + n_query) and np.array_equal(got, mask)
    plain = lin.linearize(sc.reference, sc.q_now, sc.scene, sc.config, states=states)
    rows = lin.linearize(sc.reference, sc.q_now, sc.scene, sc.config, states=states, row_floor=floor)
    esdf = (np.asarray(rows.slot) == ESDF_SLOT) & rows.used
    on_obstacle = esdf & (np.asarray(rows.sphere) == 0)
    assert on_obstacle.sum() == h and np.allclose(rows.value[on_obstacle], 0.012 - 0.015)
    assert np.allclose(rows.value[esdf & ~on_obstacle], 0.012)
    other = (np.asarray(plain.slot) != ESDF_SLOT) & plain.used                  # candidate / plane rows: untouched
    assert np.allclose(np.sort(plain.value[other]), np.sort(rows.value[(np.asarray(rows.slot) != ESDF_SLOT) & rows.used]))


# --- SL-SOTA2: the progress preference of the action QP -------------------------------------------------------------------
class _ProgLin:
    """Three steps, three action dims (x, y, z) that integrate into joints 0-2. The one collision row (step 2) wants
    joint 0 >= 0.01; no row sees joints 1 and 2. One 'hand' sphere at the origin."""

    n_spheres = 1

    def linearize(self, traj, q_now, scene, cfg):
        value = np.full((3, 1), np.inf)
        grad = np.zeros((3, 1, 3))
        used = np.zeros((3, 1), bool)
        value[2, 0] = -0.01 + traj[0, 2]
        grad[2, 0, 0] = 1.0
        used[2, 0] = True
        return types.SimpleNamespace(value=value, gradient=grad, used=used)

    def full_violation(self, traj, q_now, scene):
        return max(0.0, 0.01 - traj[0, 2])

    def sphere_states(self, traj, q_now):
        return np.zeros((3, 1, 3)), None

    def _query_name(self, i):
        return "panda_hand"


PROG = {"dims": (0, 1, 2), "gain": 0.4, "full_at_m": 0.02, "link": "panda_hand", "min_xy_m": 0.04}


def _prog_solve(prog, record, scene=None, R=None):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = TrajectoryOptimizer.__new__(TrajectoryOptimizer)
    opt.config = types.SimpleNamespace(reduction=types.SimpleNamespace(linearization_backoff=0.0),
                                       safety=types.SimpleNamespace(violation_tolerance=1e-6))
    opt.linearizer = _ProgLin()
    M = np.zeros((3 * 3, 3 * 3))
    for k in range(3):
        for j in range(k + 1):
            M[k * 3:(k + 1) * 3, j * 3:(j + 1) * 3] = np.eye(3)
    opt.ee_action_map = {"M": M, "a_ref": np.zeros((3, 3)), **({} if R is None else {"R_world_base": R})}
    opt.ee_action_progress = prog
    opt._finish = lambda traj, *a, **k: types.SimpleNamespace(trajectory=traj, metrics={})
    res = opt._solve_ee_action(np.zeros((3, 3)), np.zeros(3), scene, None, 0.01, None, [], 0.0, True, record)
    return opt.last_ee_action["delta_action"], res.metrics["ee_action_qp"]


VIOL = {"obstacle_rows": {"violation_m": 0.03}}
HELD = types.SimpleNamespace(grasp_contact={"latch": "held"}, manipulated_points=None)


def test_progress_default_off_and_only_with_an_obstacle_row_violation():
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    assert TrajectoryOptimizer.ee_action_progress is None
    da, rec = _prog_solve(None, VIOL, HELD)
    assert np.allclose(da[:, 0], 0.01 / 3, atol=1e-5) and np.allclose(da[:, 1:], 0.0, atol=1e-6) and "progress" not in rec
    for record in (None, {"obstacle_rows": {"violation_m": 0.0}}, {"reason": "collision"}):
        da, rec = _prog_solve(PROG, record, HELD)            # no obstacle-object row violated: the min-norm correction
        assert np.allclose(da[:, 1:], 0.0, atol=1e-6) and rec["progress"]["gain"] == 0.0 and rec["progress"]["direction"] is None


def test_progress_lifts_while_held_and_keeps_the_row():
    da, rec = _prog_solve(PROG, VIOL, HELD)
    assert np.allclose(da[:, 2], 0.4, atol=1e-4) and np.allclose(da[:, 1], 0.0, atol=1e-5)   # up, full gain above 20 mm
    assert np.allclose(da[:, 0], 0.01 / 3, atol=1e-4)                                        # the collision row as before
    assert rec["progress"]["mode"] == "held: up" and rec["progress"]["direction"] == [0.0, 0.0, 1.0]
    da, rec = _prog_solve(PROG, {"obstacle_rows": {"violation_m": 0.005}}, HELD)
    assert np.allclose(da[:, 2], 0.1, atol=1e-4) and rec["progress"]["gain"] == pytest.approx(0.1)   # scaled by the violation


def test_progress_goes_toward_the_target_before_the_grasp():
    scene = types.SimpleNamespace(grasp_contact={"latch": "latched"}, manipulated_points=np.array([[0.0, 0.2, -0.1]]))
    da, rec = _prog_solve(PROG, VIOL, scene)
    assert np.allclose(da[:, 1], 0.4, atol=1e-4) and np.allclose(da[:, 2], 0.0, atol=1e-5)   # +y toward the target, no z
    assert rec["progress"]["mode"].startswith("before the grasp")
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])                      # base +y is world +z
    da, _ = _prog_solve(PROG, VIOL, scene, R=Rx)
    assert np.allclose(da[:, 1], 0.0, atol=1e-4) and np.allclose(da[:, 2], 0.4, atol=1e-4)
    for sc, mode in ((types.SimpleNamespace(grasp_contact={"latch": "closing"}, manipulated_points=np.zeros((1, 3))), "closing"),
                     (types.SimpleNamespace(grasp_contact={"latch": "searching"}, manipulated_points=None), "no target"),
                     (types.SimpleNamespace(grasp_contact={"latch": "latched"}, manipulated_points=np.array([[0.0, 0.02, 0.0]])),
                      "over the target")):
        da, rec = _prog_solve(PROG, VIOL, sc)
        assert np.allclose(da[:, 1:], 0.0, atol=1e-5) and rec["progress"]["mode"] == mode


def test_progress_and_patience_flag_plumbing():
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss
    from tests.trajopt.test_t88_passthrough_when_clear import BASE as ARGV

    ap = ss.build_parser()
    a = ap.parse_args(ARGV)
    assert a.ee_action_qp_progress is None and a.obstacle_row_patience is None
    assert "obstacle_row_patience" not in ss.passthrough_overrides(a)
    a = ap.parse_args(ARGV + ["--ee-action-qp-progress", "0.4", "--obstacle-row-patience", "6"])
    assert a.ee_action_qp_progress == pytest.approx(0.4) and ss.passthrough_overrides(a)["obstacle_row_patience"] == 6
    assert {"ee_action_qp_progress", "obstacle_row_patience"} <= set(srv.PORTED_FINAL_CONFIG_DESTS)
    ok = ["--passthrough-when-clear", "--passthrough-obstacle-tolerance-mm", "0", "--ee-action-qp"]
    for bad in (["--ee-action-qp-progress", "0.4"], ok + ["--ee-action-qp-progress", "1.5"],
                ok + ["--obstacle-row-patience", "6"], ok + ["--obstacle-row-margin-mm", "15", "--obstacle-row-patience", "0"]):
        with pytest.raises(SystemExit):
            ss.reject_bad_flag_combinations(ap, ap.parse_args(ARGV + bad))


# --- SL-SOTA3: the margin's patience -----------------------------------------------------------------------------------
def test_patience_drops_the_margin_after_n_blocked_calls_and_rearms():
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = TrajectoryOptimizer.__new__(TrajectoryOptimizer)
    opt.config = types.SimpleNamespace(sqp=_sqp(obstacle_row_margin=0.015, obstacle_row_patience=2))
    opt.lookahead_from = 5
    seen = []
    for worst in (0.010, 0.010, 0.010, 0.010, -0.002, 0.020, 0.010):
        opt.linearizer = _RowsLin(worst)
        record: dict = {}
        refused = opt._obstacle_row_refuses(None, None, None, record)
        seen.append((refused, record["obstacle_rows"]["margin_m"], record["obstacle_rows"]["patience"]["relaxed"],
                     opt._obstacle_margin_now))
    # calls 1-2: inside the margin, refused; call 3: patience (2) spent, the margin is dropped and 10 mm passes;
    # call 5: below the floor itself, refused even without the margin; call 6: clear of the full margin, re-armed; call 7: refused
    assert [s[0] for s in seen] == [True, True, False, False, True, False, True]
    assert [s[2] for s in seen] == [False, False, True, True, True, False, False]
    assert [s[1] for s in seen] == [0.015, 0.015, 0.0, 0.0, 0.0, 0.015, 0.015] == [s[3] for s in seen]


def test_without_patience_the_margin_always_holds():
    for _ in range(3):
        refused, record, _kw = _refuses(0.010, obstacle_row_margin=0.015)
        assert refused and "patience" not in record["obstacle_rows"]


def test_relaxed_margin_reaches_the_action_qp():
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = TrajectoryOptimizer.__new__(TrajectoryOptimizer)
    opt.config = types.SimpleNamespace(reduction=types.SimpleNamespace(linearization_backoff=0.0),
                                       safety=types.SimpleNamespace(violation_tolerance=1e-6),
                                       sqp=_sqp(obstacle_row_margin=0.03, obstacle_row_priority=10.0, obstacle_row_patience=2))
    opt.linearizer = _Lin()
    opt.ee_action_map = {"M": np.ones((1, 1)), "a_ref": np.zeros((1, 1))}
    opt._finish = lambda traj, *a, **k: types.SimpleNamespace(trajectory=traj, metrics={})
    opt._obstacle_margin_now = 0.0                           # what `_obstacle_row_refuses` left: relaxed
    res = opt._solve_ee_action(np.zeros((1, 1)), np.zeros(1), None, None, 0.0, None, [], 0.0, True, None)
    assert abs(float(opt.last_ee_action["delta_action"][0, 0])) < 1e-6 and res.metrics["ee_action_qp"]["obstacle_row_margin_m"] == 0.0
    assert opt.linearizer.floors == [None] or all(f is None for f in opt.linearizer.floors)   # the plain rows, no floor
    assert "n_obstacle_rows" not in res.metrics["ee_action_qp"]


# --- SL-SOTA4: the action QP enforces what the pass-through test enforces ------------------------------------------------
class _GateLin(_Lin):
    """One step, one joint q. Row 0 (an obstacle object): clearance = q - 0.01. Row 1 (not an obstacle-object row, e.g. the
    held object's remnant): clearance = q - 0.06 -- a 60 mm model violation the pass-through test accepts (tau 100 mm)."""

    def obstacle_row_floor(self, centres, scene, margin, **kw):
        self.kw = dict(kw)
        return np.array([[margin, kw.get("other", 0.0)]]), np.array([[True, False]])

    def linearize(self, traj, q_now, scene, cfg, states=None, row_floor=None):
        q = float(traj[0, 0])
        value = np.array([[q - 0.01, q - 0.06]])
        self.floors.append(None if row_floor is None else np.array(row_floor))
        if row_floor is not None:
            value = value - np.asarray(row_floor)
        return types.SimpleNamespace(value=value, gradient=np.array([[[1.0], [1.0]]]), used=np.ones((1, 2), bool),
                                     slot=np.full((1, 2), ESDF_SLOT), sphere=np.array([[0, 1]]))

    def full_violation(self, traj, q_now, scene):
        return max(0.0, 0.01 - float(traj[0, 0]))


def _gate_solve(**kw):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = TrajectoryOptimizer.__new__(TrajectoryOptimizer)
    opt.config = types.SimpleNamespace(reduction=types.SimpleNamespace(linearization_backoff=0.0),
                                       safety=types.SimpleNamespace(violation_tolerance=1e-6), sqp=_sqp(**kw))
    opt.linearizer = _GateLin()
    opt.ee_action_map = {"M": np.ones((1, 1)), "a_ref": np.zeros((1, 1))}
    opt._finish = lambda traj, *a, **k: types.SimpleNamespace(trajectory=traj, metrics={})
    res = opt._solve_ee_action(np.zeros((1, 1)), np.zeros(1), None, None, 0.0, None, [], 0.0, True, None)
    return float(opt.last_ee_action["delta_action"][0, 0]), res.metrics["ee_action_qp"], opt.linearizer


def test_gate_rows_off_the_qp_pulls_every_row_back():
    da, rec, lin = _gate_solve()
    assert da == pytest.approx(0.06, abs=2e-4) and "gate_rows_relief_m" not in rec and all(f is None for f in lin.floors)


def test_gate_rows_only_the_obstacle_row_moves_the_action():
    da, rec, lin = _gate_solve(ee_action_gate_rows=True)
    assert da == pytest.approx(0.01, abs=2e-4) and rec["gate_rows_relief_m"] == pytest.approx(0.1)
    assert lin.kw["other"] == pytest.approx(-0.1) and np.allclose(lin.floors[0], [[0.0, -0.1]])
    assert rec["n_obstacle_rows"][0] == 1
    # with the margin: the obstacle-object row is held 15 mm higher, the other row still does not bind
    da, rec, lin = _gate_solve(ee_action_gate_rows=True, obstacle_row_margin=0.015, obstacle_row_priority=10.0)
    assert da == pytest.approx(0.025, abs=2e-4) and np.allclose(lin.floors[0], [[0.015, -0.1]])
    # a violation beyond tau binds again: tau 40 mm -> the 60 mm row is pulled back to 40 mm
    da, _, _ = _gate_solve(ee_action_gate_rows=True, passthrough_tolerance=0.04)
    assert da == pytest.approx(0.02, abs=2e-4)


def test_gate_rows_with_the_margin_relaxed_keeps_the_relief():
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = TrajectoryOptimizer.__new__(TrajectoryOptimizer)
    opt.config = types.SimpleNamespace(reduction=types.SimpleNamespace(linearization_backoff=0.0),
                                       safety=types.SimpleNamespace(violation_tolerance=1e-6),
                                       sqp=_sqp(ee_action_gate_rows=True, obstacle_row_margin=0.015, obstacle_row_priority=10.0,
                                                obstacle_row_patience=2))
    opt.linearizer = _GateLin()
    opt.ee_action_map = {"M": np.ones((1, 1)), "a_ref": np.zeros((1, 1))}
    opt._finish = lambda traj, *a, **k: types.SimpleNamespace(trajectory=traj, metrics={})
    opt._obstacle_margin_now = 0.0                           # relaxed by the patience
    opt._solve_ee_action(np.zeros((1, 1)), np.zeros(1), None, None, 0.0, None, [], 0.0, True, None)
    assert float(opt.last_ee_action["delta_action"][0, 0]) == pytest.approx(0.01, abs=2e-4)
    assert np.allclose(opt.linearizer.floors[0], [[0.0, -0.1]])


def test_gate_rows_config_flag_and_versions():
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.config import SqpConfig, TrajOptConfigError
    from tests.trajopt.test_t88_passthrough_when_clear import BASE as ARGV

    assert SqpConfig().ee_action_gate_rows is False
    with pytest.raises(TrajOptConfigError):
        SqpConfig(ee_action_gate_rows=True).validate()                              # needs the row classification
    SqpConfig(passthrough_obstacle_tolerance=0.0, ee_action_gate_rows=True).validate()
    ap = ss.build_parser()
    assert "ee_action_gate_rows" not in ss.passthrough_overrides(ap.parse_args(ARGV))
    ok = ["--passthrough-when-clear", "--passthrough-obstacle-tolerance-mm", "0", "--ee-action-qp"]
    assert ss.passthrough_overrides(ap.parse_args(ARGV + ok + ["--ee-action-qp-gate-rows"]))["ee_action_gate_rows"] is True
    with pytest.raises(SystemExit):
        ss.reject_bad_flag_combinations(ap, ap.parse_args(ARGV + ["--ee-action-qp-gate-rows"]))
    assert "ee_action_qp_gate_rows" in srv.PORTED_FINAL_CONFIG_DESTS
    base = "AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-"
    assert len(pf.SOTA_GATE_VERSIONS) == 2 * (4 + 116) and set(pf.SOTA_GATE_VERSIONS) <= set(pf.GT15_VERSIONS)
    for name in (base + "pe123458-gr-tro", base + "pe12345811-to0-om15p10s40-ntc-gr-tro",
                 base + "pe12345811-to0-om15p10s40-ntc-pt6-pg4-gr-tro", base + "pe12345811-to0-ntc-gr-tro"):
        v = tuple(pf.AG3S_VERSIONS[name])
        assert v[-2:] == ("--ee-action-qp-gate-rows", "--ee-action-qp-translation-only") and pf.version_server_flags(name)
    assert "ee_action_qp_gate_rows" not in pf.version_flag_values(pf.parse_v11ab_args())


def test_patience_state_is_shared_with_the_window_optimizer_and_cleared_by_reset():
    """E1a task 0 (HEAD 4f350f61): the count ran per optimizer and across episodes (an episode started at 11 blocked calls,
    already relaxed). One state per episode, over the optimizer and its window optimizer."""
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    main, window = TrajectoryOptimizer.__new__(TrajectoryOptimizer), TrajectoryOptimizer.__new__(TrajectoryOptimizer)
    for opt in (main, window):
        opt.config = types.SimpleNamespace(sqp=_sqp(obstacle_row_margin=0.015, obstacle_row_patience=2))
        opt.lookahead_from = 5
        opt.linearizer = _RowsLin(0.010)
    window._obstacle_patience = main._patience_state()       # what `__init__` does
    seen = []
    for opt in (main, window, main, window):                 # the gate opens and closes between requests
        record: dict = {}
        seen.append((opt._obstacle_row_refuses(None, None, None, record), record["obstacle_rows"]["patience"]["blocked_calls"]))
    assert seen == [(True, 1), (True, 2), (False, 3), (False, 4)]
    main.solver = types.SimpleNamespace(reset=lambda: None)
    main.observer, main.window_optimizer = None, None
    main.reset()
    assert main._patience_state() == {"blocked": 0, "relaxed": False} and window._patience_state() is main._patience_state()
    record = {}
    assert window._obstacle_row_refuses(None, None, None, record) and record["obstacle_rows"]["patience"]["blocked_calls"] == 1



# --------------------------------------------------------------------------- SL-SOTA6: the scaled action QP (F9a) under
# the margin / priority / progress terms (the ``-qs`` SOTA versions; the F7 form above is untouched)
def test_scaled_qp_with_margin_priority_and_progress(monkeypatch):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    monkeypatch.setattr(TrajectoryOptimizer, "ee_action_scaled", True)
    da, rec, _ = _solve(obstacle_row_margin=0.03, obstacle_row_priority=10.0)
    assert da == pytest.approx(0.03, abs=2e-4)               # the obstacle-object row is met, as in the F7 form
    assert max(rec["osqp_iter"]) < 20000 and all("solved" in s for s in rec["status"])
    da, rec, _ = _solve(obstacle_row_margin=0.03)             # equal weights: the two rows share the conflict
    assert da == pytest.approx(0.025, abs=5e-4)
    dz, rec = _prog_solve(PROG, VIOL, HELD)
    assert np.allclose(dz[:, 2], 0.4, atol=1e-3) and np.allclose(dz[:, 0], 0.01 / 3, atol=1e-3)
    assert max(rec["osqp_iter"]) < 20000
