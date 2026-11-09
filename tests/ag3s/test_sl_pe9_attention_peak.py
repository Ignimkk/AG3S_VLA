"""SL-PE9 (2026-11-09, dev-4a) — before the chunk carries a close command, the object under the attention peak is cut out
of what it stands on (same cut as the grasp seed) and used as the grounding input when the attention bar holds.
Off by default; never while closing / held; a plan_close seed takes precedence."""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.runtime.pipeline import AG3S
from AG3S.ag3s.stages.target_grounding import TargetConfirm
from AG3S.ag3s.types import AttentionPointCloud, PointCloud

BASE = ClusteringConfig()


def _sphere(centre, radius, n=400, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return np.asarray(centre, float) + radius * v / np.linalg.norm(v, axis=1, keepdims=True)


# a bowl on a cabinet top: one connected blob for the plain clustering
SCENE = np.concatenate([np.random.default_rng(1).uniform([0.40, -0.10, 0.10], [0.60, 0.10, 0.1001], size=(2500, 3)),
                        _sphere([0.50, 0.0, 0.135], 0.03, seed=4)])


def _apc(att_top, att_bowl):
    att = np.r_[np.full(2500, att_top, np.float32), np.full(400, att_bowl, np.float32)]
    att[2500 + 7] *= 1.01                                              # one peak point on the bowl
    return AttentionPointCloud(PointCloud(SCENE), att, att)


def _fake(cfg, *, attached=None, frozen=False, seed=None, frame=10):
    tc = TargetConfirm(cfg, max_opening=0.078, overlap_distance=0.005)
    tc.frozen = frozen
    f = types.SimpleNamespace(_close_seed=seed, _target_confirm=tc, _close_seed_frame=None, _close_seed_plane=None,
                              _attached=attached, config=types.SimpleNamespace(clustering=cfg), frame_index=frame)
    f._add_support_plane = lambda plane: AG3S._add_support_plane(f, plane)
    f._attention_peak_point = AG3S._attention_peak_point
    return f


def test_defaults_off_and_validated():
    assert BASE.attention_peak_seed is None
    AG3SConfig().with_overrides({"clustering": {"close_seed_grounding": True, "attention_peak_seed": 5.0}})
    for bad in ({"attention_peak_seed": 5.0},                                   # needs close_seed_grounding
                {"close_seed_grounding": True, "attention_peak_seed": 0.0},
                {"close_seed_grounding": True, "attention_peak_seed": True}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig().with_overrides({"clustering": bad})


def test_off_no_seed_leaves_the_input():
    cfg = dataclasses.replace(BASE, close_seed_grounding=True)
    f = _fake(cfg)
    apc = _apc(0.1, 0.9)
    cloud, excl = AG3S._close_seed_grounding_input(f, apc, None)
    assert cloud is apc and f._close_seed_frame["applied"] is False and "source" not in f._close_seed_frame


def test_on_the_attended_bowl_is_cut_off_the_cabinet():
    cfg = dataclasses.replace(BASE, close_seed_grounding=True, attention_peak_seed=5.0)
    f = _fake(cfg)
    cloud, excl = AG3S._close_seed_grounding_input(f, _apc(0.1, 0.9), None)
    rec = f._close_seed_frame
    assert rec["applied"] and rec["source"] == "attention_peak" and rec["attention_ratio"] >= 5.0
    on = np.flatnonzero(np.asarray(cloud.attention) > 0)
    assert len(on) >= 300 and on.min() >= 2500                       # only the bowl, not the cabinet top
    assert f._target_confirm._close_seed is True


def test_on_gates_attention_closing_held_and_a_grasp_seed():
    cfg = dataclasses.replace(BASE, close_seed_grounding=True, attention_peak_seed=5.0)
    f = _fake(cfg)                                                    # attention spread: the cut is not attended enough
    apc = _apc(0.5, 0.6)
    cloud, _ = AG3S._close_seed_grounding_input(f, apc, None)
    assert cloud is apc and f._close_seed_frame["applied"] is False and "rejected" in f._close_seed_frame
    for kw in ({"frozen": True}, {"attached": object()}, {"frame": 4}):   # closing / held / first requests: never
        f = _fake(cfg, **kw)
        apc = _apc(0.1, 0.9)
        cloud, _ = AG3S._close_seed_grounding_input(f, apc, None)
        assert cloud is apc and "source" not in f._close_seed_frame
    f = _fake(cfg, seed=np.array([0.53, 0.0, 0.14]))                 # a grasp seed: that one is used, no peak
    AG3S._close_seed_grounding_input(f, _apc(0.1, 0.9), None)
    assert f._close_seed_frame["applied"] and "source" not in f._close_seed_frame


def test_server_flag_and_version():
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    pe123458, pe1234589 = pf.PE_VERSIONS[6], pf.PE_VERSIONS[8]
    assert pe1234589.endswith("-pe1234589")
    assert list(pf.AG3S_VERSIONS[pe1234589]) == list(pf.AG3S_VERSIONS[pe123458]) + ["--attention-peak-seed", "5"]
    p = pf.parse_v11ab_args(argv=pf.version_server_argv(pe1234589) + list(pf.AG3S_VERSIONS[pe1234589]))
    assert ss.t43t_identity_section(p).get("attention_peak_seed") == 5.0
    p0 = pf.parse_v11ab_args(argv=pf.version_server_argv(pe123458) + list(pf.AG3S_VERSIONS[pe123458]))
    assert "attention_peak_seed" not in ss.t43t_identity_section(p0)


def test_pe12345789_is_pe1234589_plus_pe7():
    from AG3S.adapters.panda import profile as pf

    pe1234589, pe12345789 = pf.PE_VERSIONS[8], pf.PE_VERSIONS[9]
    assert pe12345789.endswith("-pe12345789")
    assert list(pf.AG3S_VERSIONS[pe12345789]) == list(pf.AG3S_VERSIONS[pe1234589]) + ["--close-seed-attention-ratio", "5"]
    assert "--no-target-certify" not in pf.AG3S_VERSIONS[pe12345789]
    pf.parse_v11ab_args(argv=pf.version_server_argv(pe12345789) + list(pf.AG3S_VERSIONS[pe12345789]))
