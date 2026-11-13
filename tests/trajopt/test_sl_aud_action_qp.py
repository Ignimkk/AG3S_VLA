"""SL-AUD1-4 (code audit of the frozen version, 2026-11-13; `local_docs/2026-11-13_frozen_config_code_audit_KO.md`): four
switches of the action QP (`TrajectoryOptimizer._solve_ee_action`), all default off.

* SL-AUD1 `sqp.ee_action_reference_floor` / `--ee-action-qp-reference-floor` (version segment ``-rf``): an ESDF row that is not an
  obstacle-object row is held at min(backoff, its clearance on the policy chunk).
* SL-AUD2 `sqp.ee_action_limit_passthrough` / `--ee-action-qp-limit-passthrough` (``-lp``): a chunk refused only for its
  joint-limit overshoot is passed through when the solve would be the action QP (no limit rows).
* SL-AUD3 `sqp.ee_action_min_reach` / `--ee-action-qp-min-reach-mm` (``-mr8`` / ``-mr12``): rows the free action components
  cannot move are left out of the scaled QP.
* SL-AUD4 `sqp.ee_action_audit_fixes` / `--ee-action-qp-audit-fixes` (``-af``): the early exit reads `full_violation` with
  the right sign; the obstacle-row margin is this call's.

Off: the record keys, the rows and the result of every case below are the ones before these switches.
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


class _Lin:
    """One step, `nq` joints, rows = (gradient (nq,), offset): clearance_i(q) = g_i . q + c_i. Row 0 is the obstacle-object
    row, the others are not. `full_violation` follows the real linearizer: the worst clearance, positive = clear."""

    def __init__(self, rows, worst=None):
        self.g = np.array([r[0] for r in rows], np.float64).reshape(len(rows), -1)
        self.c = np.array([r[1] for r in rows], np.float64)
        self.n_spheres = len(rows)
        self.floors, self.kw, self.worst = [], None, worst

    def _clear(self, q):
        return self.g @ np.asarray(q, np.float64).reshape(-1) + self.c

    def sphere_states(self, traj, q_now):
        q = np.asarray(traj, np.float64)[:, 0]
        centres = np.zeros((1, self.n_spheres, 3))
        centres[0, :, :q.size] = q            # the centres carry q so that `_esdf_clearance` can read it back
        return centres, np.zeros((1, self.n_spheres, 3, q.size))

    def _esdf_clearance(self, centres, scene):
        return self._clear(centres[0, 0, :self.g.shape[1]])[None, :]

    def obstacle_row_floor(self, centres, scene, margin, **kw):
        self.kw = dict(kw)
        floor = np.full((1, self.n_spheres), float(kw.get("other", 0.0)))
        floor[0, 0] = margin
        mask = np.zeros((1, self.n_spheres), bool)
        mask[0, 0] = True
        return floor, mask

    def linearize(self, traj, q_now, scene, cfg, states=None, row_floor=None):
        value = self._clear(np.asarray(traj, np.float64)[:, 0])[None, :]
        self.floors.append(None if row_floor is None else np.array(row_floor))
        if row_floor is not None:
            value = value - np.asarray(row_floor)
        n = self.n_spheres
        return types.SimpleNamespace(value=value, gradient=self.g[None, :, :].copy(), used=np.ones((1, n), bool),
                                     slot=np.full((1, n), ESDF_SLOT), sphere=np.arange(n)[None, :])

    def full_violation(self, traj, q_now, scene):
        return float(self._clear(np.asarray(traj, np.float64)[:, 0]).min()) if self.worst is None else float(self.worst)


def _solve(rows, *, backoff=0.0, tolerance=1e-6, scaled=False, fixed=None, worst=None, margin_now=None, **kw):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    nq = len(np.atleast_1d(rows[0][0]))
    opt = TrajectoryOptimizer.__new__(TrajectoryOptimizer)
    opt.config = types.SimpleNamespace(reduction=types.SimpleNamespace(linearization_backoff=backoff),
                                       safety=types.SimpleNamespace(violation_tolerance=tolerance), sqp=_sqp(**kw))
    opt.linearizer = _Lin(rows, worst=worst)
    opt.ee_action_map = {"M": np.eye(nq), "a_ref": np.zeros((1, nq))}
    opt.ee_action_scaled = scaled
    opt.ee_action_fixed_dims = fixed
    if margin_now is not None:
        opt._obstacle_margin_now = margin_now
    opt._finish = lambda traj, *a, **k: types.SimpleNamespace(trajectory=traj, metrics={})
    res = opt._solve_ee_action(np.zeros((nq, 1)), np.zeros(nq), None, None, 0.0, None, [], 0.0, True, None)
    return np.array(opt.last_ee_action["delta_action"][0]), res.metrics["ee_action_qp"], opt.linearizer


# --- SL-AUD1: the reference floor --------------------------------------------------------------------------------------
#: row 0 (obstacle object): clearance = q - 0.01. Row 1 (not one, e.g. the held object's remnant): clearance = q - 0.06,
#: a 60 mm model violation the pass-through test accepts (tau 100 mm).
REMNANT = [([1.0], -0.01), ([1.0], -0.06)]
#: row 0 (obstacle object) wants q <= -0.02; moving there makes row 1 (accepted at -60 mm) worse.
INTO = [([-1.0], -0.02), ([1.0], -0.06)]


def test_reference_floor_off_is_as_before():
    da, rec, lin = _solve(REMNANT)
    assert da[0] == pytest.approx(0.06, abs=2e-4)                         # every row is pulled back to 0
    assert all(f is None for f in lin.floors) and "reference_floor" not in rec and "n_obstacle_rows" not in rec


def test_reference_floor_does_not_repair_an_accepted_row():
    da, rec, lin = _solve(REMNANT, ee_action_reference_floor=True)
    assert da[0] == pytest.approx(0.01, abs=2e-4)                         # only the obstacle-object row moves the action
    assert np.allclose(lin.floors[0], [[0.0, -0.06]])
    assert rec["reference_floor"] == {"n_rows_below_backoff": 2, "min_floor_m": pytest.approx(-0.06)}
    assert "other" not in lin.kw                                         # SL-SOTA4's keyword is not used
    # with the obstacle-row margin: that row 15 mm higher, the accepted row still does not bind
    da, _, lin = _solve(REMNANT, ee_action_reference_floor=True, obstacle_row_margin=0.015, obstacle_row_priority=10.0)
    assert da[0] == pytest.approx(0.025, abs=2e-4) and np.allclose(lin.floors[0], [[0.015, -0.06]])


def test_reference_floor_does_not_let_the_correction_make_an_accepted_row_worse():
    # SL-SOTA4 frees the accepted row down to -tau: the obstacle row is met by pushing 20 mm further into it
    da, _, _ = _solve(INTO, ee_action_gate_rows=True)
    assert da[0] == pytest.approx(-0.02, abs=2e-4)
    da, rec, _ = _solve(INTO, scaled=True, ee_action_gate_rows=True)
    assert da[0] == pytest.approx(-0.02, abs=2e-4) and rec["slack_max_m"] < 1e-5      # free: no row is left short
    # the reference floor holds the accepted row where the policy chunk has it: now the two rows conflict. It is a soft
    # row like the others, so the scaled form (equal weights) shares the 20 mm between them instead of spending it all
    # on the accepted row; the plain L1 form is indifferent inside the conflict and stays near the policy chunk.
    da, rec, _ = _solve(INTO, scaled=True, ee_action_reference_floor=True)
    assert da[0] == pytest.approx(-0.01, abs=5e-4) and rec["slack_max_m"] == pytest.approx(0.01, abs=5e-4)
    da, rec, _ = _solve(INTO, ee_action_reference_floor=True)
    assert -0.01 < da[0] <= 2e-4 and rec["slack_max_m"] > 0.009
    # with the obstacle-row priority the obstacle row wins, as it does without the switch
    da, _, _ = _solve(INTO, ee_action_reference_floor=True, obstacle_row_margin=0.0001, obstacle_row_priority=10.0)
    assert da[0] < -0.019


def test_reference_floor_keeps_the_backoff_where_the_policy_chunk_has_it():
    far = ([0.0], 1.0)                                                    # the obstacle-object row is far away
    # 2 mm on the policy chunk, backoff 5 mm: off -> lifted to 5 mm; on -> not worse than 2 mm
    da, _, _ = _solve([far, ([1.0], 0.002)], backoff=0.005)
    assert da[0] == pytest.approx(0.003, abs=2e-4)
    da, _, lin = _solve([far, ([1.0], 0.002)], backoff=0.005, ee_action_reference_floor=True)
    assert da[0] == pytest.approx(0.0, abs=2e-4) and np.allclose(lin.floors[0], [[0.0, -0.003]])
    # 20 mm on the policy chunk: the floor is 0, the row is the plain one
    da, _, lin = _solve([far, ([1.0], 0.02)], backoff=0.005, ee_action_reference_floor=True)
    assert da[0] == pytest.approx(0.0, abs=2e-4) and np.allclose(lin.floors[0], [[0.0, 0.0]])


def test_reference_floor_with_the_margin_relaxed_still_applies():
    da, rec, lin = _solve(REMNANT, ee_action_reference_floor=True, obstacle_row_margin=0.015, obstacle_row_priority=10.0,
                          obstacle_row_patience=2, margin_now=0.0)      # relaxed by the patience
    assert da[0] == pytest.approx(0.01, abs=2e-4) and np.allclose(lin.floors[0], [[0.0, -0.06]])
    assert rec["obstacle_row_margin_m"] == 0.0


def test_reference_floor_unreadable_falls_back_to_the_plain_rows(monkeypatch):
    def boom(self, centres, scene):
        raise RuntimeError("no field")

    monkeypatch.setattr(_Lin, "_esdf_clearance", boom)
    da, rec, lin = _solve(REMNANT, ee_action_reference_floor=True)
    assert da[0] == pytest.approx(0.06, abs=2e-4) and all(f is None for f in lin.floors)
    assert "no field" in rec["reference_floor"]["error"]


def test_real_linearizer_takes_the_reference_floor_on_its_esdf_block(monkeypatch):
    """On the real linearizer the reference floor has the shape of its ESDF block: the obstacle-object rows come back
    `margin` lower, an accepted row (30 mm inside on the policy chunk) comes back at the backoff, a clear row unchanged."""
    pytest.importorskip("casadi")
    pytest.importorskip("mujoco")
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8)
    lin = sc.linearizer
    centres = np.asarray(lin.sphere_states(sc.reference, sc.q_now)[0])
    h, n_query = centres.shape[:2]
    clear = np.full((h, n_query), 0.04)                  # most rows: 40 mm clear
    clear[:, 0] = 0.012                                  # query 0: next to an obstacle object
    clear[:, 1] = -0.03                                  # query 1: a model violation the pass-through test accepts
    mask = np.zeros((h, n_query), bool)
    mask[:, 0] = True
    cls = type(lin)
    monkeypatch.setattr(cls, "_esdf_clearance", lambda self, c, s: clear)
    monkeypatch.setattr(cls, "_esdf_directions", lambda self, pts, rows, scene, n: np.tile([0.0, 0.0, 1.0], (len(pts), 1)))
    monkeypatch.setattr(cls, "obstacle_object_mask", lambda self, c, s, **k: (clear, mask, 0, None))
    opt = TrajectoryOptimizer.__new__(TrajectoryOptimizer)
    opt.config, opt.linearizer = sc.config, lin
    backoff = 0.005
    ref_floor = np.minimum(0.0, clear - backoff)
    rec: dict = {}
    rows, got = opt._linearize_with_obstacle_floor(sc.reference, sc.q_now, sc.scene, 0.015, rec, ref_floor=ref_floor)
    assert np.array_equal(got, mask) and "reference_floor" not in rec               # no shape mismatch
    esdf = (np.asarray(rows.slot) == ESDF_SLOT) & rows.used
    sphere = np.asarray(rows.sphere)
    assert np.allclose(rows.value[esdf & (sphere == 0)], 0.012 - 0.015)
    assert (esdf & (sphere == 1)).sum() == h and np.allclose(rows.value[esdf & (sphere == 1)], backoff)
    assert np.allclose(rows.value[esdf & (sphere > 1)], 0.04)
    # a floor of another shape is not applied (recorded), the obstacle-row floor still is
    rec = {}
    rows, _ = opt._linearize_with_obstacle_floor(sc.reference, sc.q_now, sc.scene, 0.015, rec, ref_floor=ref_floor[:, :3])
    assert rec["reference_floor"]["shape_mismatch"] == [[h, 3], [h, n_query]]
    esdf = (np.asarray(rows.slot) == ESDF_SLOT) & rows.used
    assert np.allclose(rows.value[esdf & (np.asarray(rows.sphere) == 1)], -0.03)


# --- SL-AUD3: rows the free action components cannot move ----------------------------------------------------------------
#: two actions (M = I). Row 0: 3 mm per unit of action 0, 5 mm short (a step-0 row: needs |Da| 1.67, the box gives 1).
#: Row 1: 20 mm per unit of action 1, 4 mm short.
LEVER = [([0.003, 0.0], -0.005), ([0.0, 0.02], -0.004)]


def test_min_reach_off_the_weak_row_saturates_the_action():
    da, rec, _ = _solve(LEVER, scaled=True)
    assert da[0] == pytest.approx(1.0, abs=1e-3) and da[1] == pytest.approx(0.2, abs=2e-3)
    assert rec["saturated_components"] == 1 and rec["slack_max_m"] == pytest.approx(0.002, abs=1e-4)
    assert rec["n_rows_unreachable"][0] == 0 and "min_reach_m" not in rec


def test_min_reach_leaves_the_weak_row_out():
    da, rec, _ = _solve(LEVER, scaled=True, ee_action_min_reach=0.008)
    assert da[0] == pytest.approx(0.0, abs=1e-4) and da[1] == pytest.approx(0.2, abs=2e-3)
    assert rec["n_rows_unreachable"][0] == 1 and rec["min_reach_m"] == pytest.approx(0.008)
    assert rec["saturated_components"] == 0
    # a threshold below the row's reach keeps it
    da, rec, _ = _solve(LEVER, scaled=True, ee_action_min_reach=0.002)
    assert da[0] == pytest.approx(1.0, abs=1e-3) and rec["n_rows_unreachable"][0] == 0


def test_min_reach_counts_the_free_components_only():
    # the row's gradient is almost all on action 1, which is fixed (translation-only fixes the rotation columns)
    rows = [([0.001, 0.05], -0.004), ([0.02, 0.0], -0.004)]
    da, rec, _ = _solve(rows, scaled=True, fixed=(1,))
    assert rec["n_rows_unreachable"][0] == 0 and da[1] == 0.0            # off: the whole-row norm (0.05) keeps it
    da, rec, _ = _solve(rows, scaled=True, fixed=(1,), ee_action_min_reach=0.008)
    assert rec["n_rows_unreachable"][0] == 1 and da[0] == pytest.approx(0.2, abs=2e-3) and da[1] == 0.0


# --- SL-AUD5: the cap on one component's correction ---------------------------------------------------------------------
def test_max_delta_caps_the_correction_and_leaves_the_rest_as_slack():
    row = [([1.0], -0.8)]                                                 # wants Da = 0.8
    da, rec, _ = _solve(row)
    assert da[0] == pytest.approx(0.8, abs=2e-4) and "max_delta_cap" not in rec and "capped_components" not in rec
    da, rec, _ = _solve(row, ee_action_max_delta=0.5)
    assert da[0] == pytest.approx(0.5, abs=1e-6) and rec["max_delta_cap"] == 0.5 and rec["capped_components"] == 1
    assert rec["slack_max_m"] == pytest.approx(0.3, abs=2e-4)             # what the cap leaves short stays a slack
    da, rec, _ = _solve([([1.0], -0.2)], ee_action_max_delta=0.5)         # a correction inside the cap is untouched
    assert da[0] == pytest.approx(0.2, abs=2e-4) and rec["capped_components"] == 0
    # the +-1 box of a_ref + Da still holds, and a fixed column stays fixed
    da, rec, _ = _solve([([1.0, 1.0], -0.8), ([0.0, 1.0], -0.3)], scaled=True, fixed=(1,), ee_action_max_delta=0.5)
    assert da[0] == pytest.approx(0.5, abs=1e-6) and da[1] == 0.0


# --- SL-AUD4 (a): the early exit -----------------------------------------------------------------------------------------
def test_early_exit_reads_the_clearance_with_the_right_sign():
    met = [([1.0], -0.01)]                                                # met with zero slack from the first solve
    # the returned trajectory is 0.5 m clear: off -> `0.5 <= tau` is False, the loop runs to its cap; on -> stops
    assert _solve(met, tolerance=0.1, worst=0.5)[1]["iterations"] == 4
    assert _solve(met, tolerance=0.1, worst=0.5, ee_action_audit_fixes=True)[1]["iterations"] == 2
    # 0.2 m of violation remains outside the QP rows: off -> `-0.2 <= tau` stops; on -> keeps re-linearising
    assert _solve(met, tolerance=0.1, worst=-0.2)[1]["iterations"] == 2
    assert _solve(met, tolerance=0.1, worst=-0.2, ee_action_audit_fixes=True)[1]["iterations"] == 4
    # inside tau either way: the same exit
    assert _solve(met, tolerance=0.1, worst=-0.05)[1]["iterations"] == 2
    assert _solve(met, tolerance=0.1, worst=-0.05, ee_action_audit_fixes=True)[1]["iterations"] == 2


# --- SL-AUD2 / SL-AUD4 (b): on the real optimizer ----------------------------------------------------------------------
def _limit_case():
    """A clear scene and a policy chunk that jumps 0.5 rad at step 0: the pass-through test refuses it for `limit`."""
    pytest.importorskip("casadi")
    pytest.importorskip("mujoco")   # the fixture reads the joint ranges from the MJCF
    from tests.trajopt.test_t88_passthrough_when_clear import _clear

    sc = _clear()
    ref = sc.reference.copy()
    ref[0, :] += 0.5
    return sc, ref


def _real(sc, **sqp):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    cfg = sc.config.with_overrides({"sqp": {"passthrough_when_clear": True, **sqp}})
    return TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)


def _zero_map(ref):
    return {"M": np.zeros((ref.size, ref.shape[1] * 6)), "a_ref": np.zeros((ref.shape[1], 6))}


def test_limit_refusal_goes_to_the_action_qp_when_off():
    sc, ref = _limit_case()
    opt = _real(sc)
    opt.ee_action_map = _zero_map(ref)
    out = opt.solve(ref, sc.q_now, sc.scene, template=sc.chunk)
    rec = out.metrics["passthrough_clear"]
    assert rec["applied"] is False and rec["reason"] == "limit" and "limit_passthrough" not in rec
    assert "ee_action_qp" in out.metrics and opt.last_ee_action is not None


def test_limit_refusal_is_passed_through_with_the_action_map():
    sc, ref = _limit_case()
    opt = _real(sc, ee_action_limit_passthrough=True)
    opt.ee_action_map = _zero_map(ref)
    out = opt.solve(ref, sc.q_now, sc.scene, template=sc.chunk)
    rec = out.metrics["passthrough_clear"]
    assert rec["applied"] is True and rec["reason"] == "limit" and rec["limit_passthrough"] is True
    assert out.trajectory.tobytes() == ref.tobytes() and out.iterations == 0
    assert "ee_action_qp" not in out.metrics and opt.last_ee_action is None
    assert any("SL-AUD2" in n for n in out.notes) and not any(n.startswith("T88:") for n in out.notes)


def test_limit_passthrough_needs_the_action_map_and_the_limit_reason():
    sc, ref = _limit_case()
    # no action map (the joint SQP has limit rows): the same result as with the switch off
    on = _real(sc, ee_action_limit_passthrough=True).solve(ref, sc.q_now, sc.scene, template=sc.chunk)
    off = _real(sc).solve(ref, sc.q_now, sc.scene, template=sc.chunk)
    assert on.metrics["passthrough_clear"]["applied"] is False
    assert on.trajectory.tobytes() == off.trajectory.tobytes()
    # another reason (uncertified geometry) is not passed through
    opt = _real(sc, ee_action_limit_passthrough=True)
    opt.config = opt.config.with_overrides({"safety": {"require_certified_geometry": True}})
    opt.ee_action_map = _zero_map(sc.reference)
    out = opt.solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk, geometry_certified=False)
    assert out.metrics["passthrough_clear"]["reason"] == "geometry not certified"
    assert out.metrics["passthrough_clear"]["applied"] is False and "ee_action_qp" in out.metrics


def test_the_obstacle_row_margin_of_an_earlier_call_is_not_reused():
    sc, ref = _limit_case()
    for fixes, left in ((False, 0.0), (True, None)):
        opt = _real(sc, **({"ee_action_audit_fixes": True} if fixes else {}))
        opt._obstacle_margin_now = 0.0                    # left by an earlier call whose margin was relaxed
        opt.solve(ref, sc.q_now, sc.scene, template=sc.chunk)   # this call's obstacle test does not run (no tolerance set)
        assert opt.__dict__.get("_obstacle_margin_now") == left


# --- config, flags, versions -------------------------------------------------------------------------------------------
def test_config_and_flags():
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.config import SqpConfig, TrajOptConfig, TrajOptConfigError

    d = SqpConfig()
    assert (d.ee_action_reference_floor, d.ee_action_limit_passthrough, d.ee_action_min_reach, d.ee_action_audit_fixes) == (
        False, False, None, False)
    keys = {"ee_action_reference_floor", "ee_action_limit_passthrough", "ee_action_min_reach", "ee_action_audit_fixes",
            "ee_action_max_delta"}
    assert d.ee_action_max_delta is None
    assert not keys & set(TrajOptConfig().to_dict()["sqp"])               # off: the dump has no new key
    for bad in (dict(ee_action_reference_floor=True),                                         # needs the row classification
                dict(passthrough_obstacle_tolerance=0.0, ee_action_reference_floor=True, ee_action_gate_rows=True),
                dict(ee_action_limit_passthrough=True),                                       # needs the pass-through
                dict(ee_action_min_reach=0.0), dict(ee_action_min_reach=0.2), dict(ee_action_audit_fixes=1),
                dict(ee_action_max_delta=0.0), dict(ee_action_max_delta=3.0)):
        with pytest.raises(TrajOptConfigError):
            SqpConfig(**bad).validate()
    SqpConfig(passthrough_when_clear=True, passthrough_obstacle_tolerance=0.0, ee_action_reference_floor=True,
              ee_action_limit_passthrough=True, ee_action_min_reach=0.008, ee_action_audit_fixes=True, ee_action_max_delta=0.5).validate()

    base = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]
    ap = ss.build_parser()
    assert not keys & set(ss.passthrough_overrides(ap.parse_args(base)))
    ok = base + ["--passthrough-when-clear", "--passthrough-obstacle-tolerance-mm", "0", "--ee-action-qp", "--ee-action-qp-scaled"]
    new = ["--ee-action-qp-reference-floor", "--ee-action-qp-limit-passthrough", "--ee-action-qp-min-reach-mm", "8",
           "--ee-action-qp-audit-fixes", "--ee-action-qp-max-delta", "0.5"]
    args = ap.parse_args(ok + new)
    ss.reject_bad_flag_combinations(ap, args)
    over = ss.passthrough_overrides(args)
    assert over["ee_action_reference_floor"] is True and over["ee_action_limit_passthrough"] is True
    assert over["ee_action_min_reach"] == pytest.approx(0.008) and over["ee_action_audit_fixes"] is True
    assert over["ee_action_max_delta"] == pytest.approx(0.5)
    sqp = ss.trajopt_config_from_args(args).sqp
    assert sqp.ee_action_reference_floor and sqp.ee_action_limit_passthrough and sqp.ee_action_audit_fixes
    assert sqp.ee_action_min_reach == pytest.approx(0.008)
    for lone in (["--ee-action-qp-reference-floor"], ["--ee-action-qp-limit-passthrough"],
                 ["--ee-action-qp-min-reach-mm", "8"], ["--ee-action-qp-audit-fixes"], ["--ee-action-qp-max-delta", "0.5"]):
        with pytest.raises(SystemExit):
            ss.reject_bad_flag_combinations(ap, ap.parse_args(base + lone))
    with pytest.raises(SystemExit):                                       # the two floors of the same rows
        ss.reject_bad_flag_combinations(ap, ap.parse_args(ok + ["--ee-action-qp-reference-floor", "--ee-action-qp-gate-rows"]))
    with pytest.raises(SystemExit):
        ss.reject_bad_flag_combinations(ap, ap.parse_args(ok + ["--ee-action-qp-min-reach-mm", "80"]))

    dests = ("ee_action_qp_reference_floor", "ee_action_qp_limit_passthrough", "ee_action_qp_min_reach_mm",
             "ee_action_qp_audit_fixes", "ee_action_qp_max_delta")
    assert set(dests) <= set(srv.PORTED_FINAL_CONFIG_DESTS)
    assert not set(dests) & set(pf.version_flag_values(pf.parse_v11ab_args()))

    # the frozen version's config is untouched
    frozen = pf.FROZEN_20261112
    f = ss.trajopt_config_from_args(ap.parse_args(pf.version_server_argv(frozen) + list(pf.version_server_flags(frozen)))).sqp
    assert (f.ee_action_reference_floor, f.ee_action_limit_passthrough, f.ee_action_min_reach, f.ee_action_audit_fixes,
            f.ee_action_gate_rows, f.ee_action_max_delta) == (False, False, None, False, False, None)
