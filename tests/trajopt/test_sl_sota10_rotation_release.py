"""SL-SOTA10 (2026-11-11, dev-57) — `--ee-action-qp-rotation-release` (versions ``-rr``): the translation-only action QP
gets its rotation columns back while the obstacle-row patience is used up (`_patience_state()["relaxed"]`).

E1a ...-pt6-pg4-hr150-qs-tro task 1 init 6 / 8 / 9 / 12: the bowl is held 30-47 calls with link5 / link6 40 mm inside the
milk / moka pot row; the policy steps +y / -z, the translation-only QP cancels that step every call and the hand stands
still until `max_steps`. Off by default.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

osqp = pytest.importorskip("osqp")


class _Lin:
    """One step, six action dims (x y z rx ry rz) that are the six joints. The one collision row wants
    joint 0 + 2 * joint 4 >= 0.3: reachable by translation (x) and, twice as well, by rotation (ry)."""

    n_spheres = 1

    def linearize(self, traj, q_now, scene, cfg):
        value = np.array([[-0.3 + traj[0, 0] + 2.0 * traj[4, 0]]])
        grad = np.zeros((1, 1, 6))
        grad[0, 0, 0], grad[0, 0, 4] = 1.0, 2.0
        return types.SimpleNamespace(value=value, gradient=grad, used=np.ones((1, 1), bool))

    def full_violation(self, traj, q_now, scene):
        return max(0.0, 0.3 - traj[0, 0] - 2.0 * traj[4, 0])


def _solve(*, fixed=(3, 4, 5), release=False, relaxed=False):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = TrajectoryOptimizer.__new__(TrajectoryOptimizer)
    opt.config = types.SimpleNamespace(reduction=types.SimpleNamespace(linearization_backoff=0.0),
                                       safety=types.SimpleNamespace(violation_tolerance=1e-6))
    opt.linearizer = _Lin()
    opt.ee_action_map = {"M": np.eye(6), "a_ref": np.zeros((1, 6))}
    opt.ee_action_fixed_dims = fixed
    if release:
        opt.ee_action_rotation_release = True
    opt._patience_state().update(blocked=9 if relaxed else 0, relaxed=relaxed)
    opt._finish = lambda traj, *a, **k: types.SimpleNamespace(trajectory=traj, metrics={})
    res = opt._solve_ee_action(np.zeros((6, 1)), np.zeros(6), None, None, 0.3, None, [], 0.0, True, None)
    return opt.last_ee_action["delta_action"][0], res.metrics["ee_action_qp"]


def test_default_off_and_untouched_record():
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    assert TrajectoryOptimizer.ee_action_rotation_release is False
    da, rec = _solve(relaxed=True)                 # the patience is used up but the switch is off: translation only
    assert da[0] == pytest.approx(0.3, abs=1e-4) and np.abs(da[3:]).max() < 1e-9
    assert rec["fixed_dims"] == [3, 4, 5] and "rotation_released" not in rec


def test_rotation_stays_fixed_until_the_patience_is_used_up():
    da, rec = _solve(release=True, relaxed=False)
    assert da[0] == pytest.approx(0.3, abs=1e-4) and np.abs(da[3:]).max() < 1e-9
    assert rec["fixed_dims"] == [3, 4, 5] and rec["rotation_released"] is False


def test_rotation_is_free_while_relaxed():
    da, rec = _solve(release=True, relaxed=True)
    # min ||Da||^2 with x + 2 ry >= 0.3  ->  x = 0.06, ry = 0.12
    assert da[0] == pytest.approx(0.06, abs=1e-4) and da[4] == pytest.approx(0.12, abs=1e-4)
    assert np.abs(da[[1, 2, 3, 5]]).max() < 1e-6
    assert "fixed_dims" not in rec and rec["rotation_released"] is True


def test_without_fixed_dims_the_switch_does_nothing():
    da, rec = _solve(fixed=None, release=True, relaxed=False)
    assert da[4] == pytest.approx(0.12, abs=1e-4) and "rotation_released" not in rec


def test_flag_plumbing_and_versions():
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss
    from tests.trajopt.test_t88_passthrough_when_clear import BASE as ARGV

    ap = ss.build_parser()
    assert ap.parse_args(ARGV).ee_action_qp_rotation_release is False
    ok = ["--passthrough-when-clear", "--passthrough-obstacle-tolerance-mm", "0", "--ee-action-qp",
          "--obstacle-row-margin-mm", "15"]
    for bad in (ok + ["--ee-action-qp-rotation-release"],
                ok + ["--ee-action-qp-translation-only", "--ee-action-qp-rotation-release"],
                ok + ["--obstacle-row-patience", "6", "--ee-action-qp-rotation-release"]):
        with pytest.raises(SystemExit):
            ss.reject_bad_flag_combinations(ap, ap.parse_args(ARGV + bad))
    assert "ee_action_qp_rotation_release" in srv.PORTED_FINAL_CONFIG_DESTS
    base = "AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150"
    for mid in ("-qs", "-hb-qs", "-qs-ap", "-hb-qs-ap"):
        name = base + mid + "-rr-tro"
        assert name in pf.SOTA_ROTREL_VERSIONS and name in pf.GT15_VERSIONS
        v = tuple(pf.AG3S_VERSIONS[name])
        assert v[-2:] == ("--ee-action-qp-translation-only", "--ee-action-qp-rotation-release")
        assert v[:-1] == tuple(pf.AG3S_VERSIONS[base + mid + "-tro"])
        args = pf.parse_v11ab_args(argv=pf.version_server_argv(name) + list(v))
        assert args.ee_action_qp_rotation_release is True and args.obstacle_row_patience == 6
    assert all("-pt6-" in n and "-hr150" in n and n.endswith("-rr-tro") for n in pf.SOTA_ROTREL_VERSIONS)
    for n in pf.SOTA_ROTREL_VERSIONS:
        assert pf.version_server_flags(n)
    assert "ee_action_qp_rotation_release" not in pf.version_flag_values(pf.parse_v11ab_args())
