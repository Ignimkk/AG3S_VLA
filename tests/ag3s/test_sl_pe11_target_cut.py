"""SL-PE11 (2026-11-09, dev-4a; patch) — the manipulated object leaves the scene obstacle-object set by points, so an
obstacle touching it stays an obstacle (pe12345789 SafeLIBERO t3 e2 / e4). Off by default."""
from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.runtime.pipeline import AG3S


def _sphere(centre, radius, n=500, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return np.asarray(centre, float) + radius * v / np.linalg.norm(v, axis=1, keepdims=True)


BOWL = _sphere([0.50, 0.00, 0.05], 0.04, seed=1)
MUG = _sphere([0.50, 0.085, 0.05], 0.035, seed=2)        # 10 mm gap: one 15 mm component with the bowl
SEEN = BOWL[BOWL[:, 1] < 0.0]                              # the identity covers only half of the bowl


def test_defaults_off_and_validated():
    assert ClusteringConfig().scene_obstacle_target_cut is False
    with pytest.raises(AG3SConfigError):
        AG3SConfig().with_overrides({"clustering": {"scene_obstacle_target_cut": True}})
    AG3SConfig().with_overrides({"clustering": {"scene_obstacle_target_cut": True, "scene_obstacle_objects": True}})


def _near(out, c, r):
    return out is not None and len(out) and bool(np.any(np.linalg.norm(out - np.asarray(c), axis=1) < r))


def test_off_drops_the_touching_mug_on_keeps_it_and_drops_the_whole_bowl():
    cloud = np.concatenate([BOWL, MUG])
    off = types.SimpleNamespace(_obstacle_history=[])
    out, _ = AG3S._scene_obstacle_points(off, cloud, None, SEEN, None)
    assert not _near(out, [0.50, 0.085, 0.05], 0.04)                  # off: the mug left with the bowl's component
    on = types.SimpleNamespace(_obstacle_history=[])
    out, rec = AG3S._scene_obstacle_points(on, cloud, None, SEEN, None, target_cut=True)
    assert _near(out, [0.50, 0.10, 0.05], 0.03)                      # on: the mug's far side stays an obstacle
    assert not _near(out, [0.50, -0.03, 0.05], 0.02)                 # the seen half of the bowl is out
    assert rec["target_cut"]["n_dropped"] >= len(SEEN)


def test_server_flag_and_version():
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    base, v = pf.PE_VERSIONS[9], pf.PE_VERSIONS[10]
    assert v.endswith("-pe1234578911")
    assert list(pf.AG3S_VERSIONS[v]) == list(pf.AG3S_VERSIONS[base]) + ["--scene-obstacle-target-cut"]
    p = pf.parse_v11ab_args(argv=pf.version_server_argv(v) + list(pf.AG3S_VERSIONS[v]))
    assert ss.t43t_identity_section(p).get("scene_obstacle_target_cut") is True
