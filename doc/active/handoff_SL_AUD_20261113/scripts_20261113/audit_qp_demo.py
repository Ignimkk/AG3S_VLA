"""SL-AUD offline demo — the action QP (`TrajectoryOptimizer._solve_ee_action`) on a synthetic scene with the REAL Panda
action map (`action.ee_action_map`: response model + Jacobian), frozen settings vs the SL-AUD switches.

Scene (base frame), one sphere at the grip site:
* a wall the hand moves along (an obstacle-object row): clearance `wall_mm` on every step of the policy chunk
  (frozen: refused below the 15 mm margin; the QP holds it at margin + 5 mm backoff);
* a table under the hand (NOT an obstacle-object row): clearance `table_mm` on every step (the pass-through test
  accepts it down to -100 mm; the QP holds it at +5 mm).
The policy chunk moves +y at 0.6 (parallel to the wall).

Run: PYTHONUTF8=1 PYTHONPATH=. python doc/active/handoff_SL_AUD_20261113/scripts_20261113/audit_qp_demo.py   (needs numpy, scipy, osqp, pyyaml-free)
"""
from __future__ import annotations

import types

import numpy as np

from AG3S.adapters.panda import action as pa
from AG3S.adapters.panda import model as pm
from AG3S.ag3s.robot_models.urdf_sphere_chain import UrdfSphereChain
from AG3S.trajopt.linearize import ESDF_SLOT
from AG3S.trajopt.sqp import TrajectoryOptimizer

H = 10
Q_NOW = np.array([-0.001, -0.159, 0.017, -2.459, 0.005, 2.244, 0.798])   # the LIBERO start pose
kin = pa.PandaKinematics(UrdfSphereChain(pm.load_panda_urdf(), pm.PANDA_ARM_JOINTS))


class Lin:
    """Two rows per step on the grip-site position p: wall (obstacle object) x_wall - p_x, table p_z - z_table."""

    n_spheres = 2

    def __init__(self, p_ref, wall, table):
        self.x_wall = p_ref[:, 0] + wall          # the wall keeps `wall` from the reference at every step
        self.z_table = p_ref[:, 2] - table

    def _p(self, traj):
        return np.array([kin.fk(traj[:, k])[0] for k in range(traj.shape[1])])

    def _clear(self, p):
        return np.stack([self.x_wall - p[:, 0], p[:, 2] - self.z_table], axis=1)      # (H, 2)

    def sphere_states(self, traj, q_now):
        p = self._p(np.asarray(traj, np.float64))
        return np.repeat(p[:, None, :], 2, axis=1), None

    def _esdf_clearance(self, centres, scene):
        return self._clear(np.asarray(centres)[:, 0, :])

    def obstacle_row_floor(self, centres, scene, margin, **kw):
        floor = np.full((H, 2), float(kw.get("other", 0.0)))
        floor[:, 0] = margin
        mask = np.zeros((H, 2), bool)
        mask[:, 0] = True
        return floor, mask

    def linearize(self, traj, q_now, scene, cfg, states=None, row_floor=None):
        traj = np.asarray(traj, np.float64)
        value = self._clear(self._p(traj))
        if row_floor is not None:
            value = value - np.asarray(row_floor)
        grad = np.zeros((H, 2, 7))
        for k in range(H):
            J = kin.jacobian(traj[:, k])[:3]
            grad[k, 0], grad[k, 1] = -J[0], J[2]
        return types.SimpleNamespace(value=value, gradient=grad, used=np.ones((H, 2), bool),
                                     slot=np.full((H, 2), ESDF_SLOT), sphere=np.tile(np.arange(2), (H, 1)))

    def full_violation(self, traj, q_now, scene):
        return float(self._clear(self._p(np.asarray(traj, np.float64))).min())


def solve(wall_mm, table_mm, **sqp):
    A = np.zeros((H, 7), np.float32)
    A[:, 1] = 0.6
    A[:, 6] = -1.0
    conv = pa.delta_chunk_to_joint_chunk(kin, A, Q_NOW, np.eye(4))
    ref = conv.joint_chunk[:, :7].T.copy()                                # (7, H)
    p_ref = np.array([kin.fk(ref[:, k])[0] for k in range(H)])
    base = dict(passthrough_obstacle_tolerance=0.0, passthrough_obstacle_full_horizon=True, passthrough_obstacle_beneath=False,
                passthrough_obstacle_class_tests=False, obstacle_row_margin=0.015, obstacle_row_priority=10.0,
                obstacle_object_surface_tolerance=None, obstacle_row_patience=None, ee_action_gate_rows=False,
                passthrough_tolerance=0.1)
    base.update(sqp)
    opt = TrajectoryOptimizer.__new__(TrajectoryOptimizer)
    opt.config = types.SimpleNamespace(reduction=types.SimpleNamespace(linearization_backoff=0.005),
                                       safety=types.SimpleNamespace(violation_tolerance=0.1),
                                       sqp=types.SimpleNamespace(**base))
    opt.linearizer = Lin(p_ref, wall_mm / 1000.0, table_mm / 1000.0)
    opt.ee_action_map = pa.ee_action_map(kin, conv, H)
    opt.ee_action_scaled = True                 # frozen: -qs
    opt.ee_action_fixed_dims = (3, 4, 5)        # frozen: -tro
    opt._finish = lambda traj, *a, **k: types.SimpleNamespace(trajectory=traj, metrics={})
    res = opt._solve_ee_action(ref, Q_NOW, None, None, 0.0, None, [], 0.0, True, None)
    da = opt.last_ee_action["delta_action"]
    rec = res.metrics["ee_action_qp"]
    p_new = np.array([kin.fk(opt.last_ee_action["trajectory"][:, k])[0] for k in range(H)])
    shift = 1000 * np.linalg.norm(p_new[4] - p_ref[4])                    # EE displacement at the end of the executed rows
    return da, rec, shift


VARIANTS = (("frozen (pkgFrs)", {}),
            ("-gr  gate rows (SL-SOTA4)", dict(ee_action_gate_rows=True)),
            ("-rf  reference floor", dict(ee_action_reference_floor=True)),
            ("-mr8 min reach 8 mm", dict(ee_action_min_reach=0.008)),
            ("-mr12 min reach 12 mm", dict(ee_action_min_reach=0.012)),
            ("-rf-mr8", dict(ee_action_reference_floor=True, ee_action_min_reach=0.008)),
            ("-rf-mr12", dict(ee_action_reference_floor=True, ee_action_min_reach=0.012)),
            ("-md5 cap 0.5 (candidate B)", dict(ee_action_max_delta=0.5)),
            ("-rf-mr8-md5", dict(ee_action_reference_floor=True, ee_action_min_reach=0.008, ee_action_max_delta=0.5)))

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    for wall, table in ((14.0, 30.0), (14.0, 3.0), (5.0, 3.0), (14.0, -30.0)):
        print(f"\n== wall {wall:g} mm from the hand (margin 15 mm), table {table:g} mm under it "
              f"({'accepted model violation' if table < 0 else 'accepted'})")
        print(f"{'variant':<28} {'max|Da|':>7} {'saturated':>9} {'slack mm':>8} {'rows out':>8} {'EE shift @row4 mm':>17}   Da rows 0-2 (x y z)")
        for name, kw in VARIANTS:
            da, rec, shift = solve(wall, table, **kw)
            out = (rec.get("n_rows_unreachable") or [0])[0]
            print(f"{name:<28} {rec['max_abs_delta_action']:7.2f} {rec['saturated_components']:9d} "
                  f"{1000 * rec['slack_max_m']:8.1f} {out:8d} {shift:17.1f}   {da[0, :3]} {da[1, :3]} {da[2, :3]}")
