"""SL-QE2 (dev-e5, 2026-11-09): `--ee-action-qp-translation-only` — the action QP bends the policy's position only.

E1a SafeLIBERO task 3 (G110): the F7 action QP's first intervention moved rx / ry / rz by ~1.0 (normalised action), the
grasp pose was lost; the multi-link CBF (GT) changes x y z only and finished the task.
"""
import numpy as np
import pytest

osqp = pytest.importorskip("osqp")


def _solve(fixed):
    """The QP of `TrajectoryOptimizer._solve_ee_action` on a synthetic 1-row problem (same box / bound handling)."""
    import scipy.sparse as sp

    H, m = 2, 6
    a_ref = np.zeros((H, m))
    n_x = a_ref.size
    lo_x = (-1.0 - a_ref).reshape(-1)
    hi_x = (1.0 - a_ref).reshape(-1)
    if fixed:
        fmask = np.zeros(a_ref.shape, bool)
        fmask[:, list(fixed)] = True
        lo_x[fmask.reshape(-1)] = 0.0
        hi_x[fmask.reshape(-1)] = 0.0
    G = np.ones((1, n_x))          # one collision row reachable through every action dim
    b = np.array([0.6])
    n = 1
    P = sp.block_diag([sp.identity(n_x), sp.identity(n) * 1e-6], format="csc")
    q = np.concatenate([np.zeros(n_x), np.full(n, 1e6)])
    A = sp.vstack([sp.hstack([sp.csc_matrix(G), sp.identity(n)]),
                   sp.hstack([sp.identity(n_x), sp.csc_matrix((n_x, n))]),
                   sp.hstack([sp.csc_matrix((n, n_x)), sp.identity(n)])], format="csc")
    s = osqp.OSQP(algebra="builtin")
    s.setup(P=P, q=q, A=A, l=np.concatenate([b, lo_x, np.zeros(n)]), u=np.concatenate([np.full(n, np.inf), hi_x,
                                                                                       np.full(n, np.inf)]),
            verbose=False, eps_abs=1e-8, eps_rel=1e-8, polishing=True)
    x = np.clip(s.solve().x[:n_x], lo_x, hi_x)
    return x.reshape(H, m)


def test_free_qp_uses_rotation():
    da = _solve(None)
    assert np.abs(da[:, 3:]).max() > 1e-3          # the minimum-norm correction spreads over all six dims


def test_translation_only_keeps_rotation_and_meets_the_row():
    da = _solve((3, 4, 5))
    assert np.abs(da[:, 3:]).max() < 1e-9          # rotation untouched
    assert da.sum() >= 0.6 - 1e-6                  # the collision row is still met by x y z alone


def test_optimizer_attribute_default_off():
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    assert TrajectoryOptimizer.ee_action_fixed_dims is None


def test_flag_plumbing_and_versions():
    from AG3S.trajopt import serve_safe as ss
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as ps
    from tests.trajopt.test_t88_passthrough_when_clear import BASE
    ap = ss.build_parser()
    a = ap.parse_args(BASE + ["--ee-action-qp", "--ee-action-qp-translation-only"])
    assert a.ee_action_qp_translation_only is True
    assert "ee_action_qp_translation_only" in ps.PORTED_FINAL_CONFIG_DESTS
    for v in pf.QE2_VERSIONS:
        base = v[:-len("-tro")]
        assert tuple(pf.AG3S_VERSIONS[v]) == tuple(pf.AG3S_VERSIONS[base]) + ("--ee-action-qp-translation-only",)
        assert "--ee-action-qp" in pf.AG3S_VERSIONS[v]
        assert "--ee-action-qp-translation-only" not in pf.AG3S_VERSIONS[base]   # the evaluated B5 versions unchanged
