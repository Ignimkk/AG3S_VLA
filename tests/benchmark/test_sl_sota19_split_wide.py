"""SL-SOTA19 -- the T31b attention split also re-clusters clusters wider than the opening in every direction
(`--attention-split-wide`, `clustering.attention_split_wide`; sdm-f6 2026-11-13). CPU.
(a) off: a wide cluster is never split (T31b); (b) on: two touching objects in one wide blob come apart and the attended one is
the target component; (c) on: one wide object (one connected surface) stays whole and no partition cut is made;
(d) the narrow-trigger behaviour is unchanged; (e) config / option / version plumbing."""
from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.stages import attention_split as asp

OPEN = 0.078
FROZEN = ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-"
          "hr150-hb-qs-rs-tro")


def _grid(lo, hi, step=0.006):
    ax = [np.arange(a, b + 1e-9, step) for a, b in zip(lo, hi)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    return g


def _two_objects(gap=0.03):
    a = _grid((0.0, 0.0, 0.0), (0.10, 0.10, 0.10))                    # attended object, 100 mm cube (wide)
    b = _grid((0.10 + gap, 0.0, 0.0), (0.22 + gap, 0.12, 0.12))         # neighbour
    pts = np.concatenate([a, b]); att = np.r_[np.full(len(a), 0.9), np.full(len(b), 0.05)]
    return pts, att, len(a)


def test_off_never_splits_a_wide_cluster():
    pts, att, _ = _two_objects()
    assert asp.split_by_attention(pts, att, np.arange(len(pts)), max_opening=OPEN, radius=0.012, min_points=20, min_ratio=2.0) is None


def test_on_splits_merged_objects_and_keeps_the_attended_one():
    pts, att, na = _two_objects()
    dec = asp.split_by_attention(pts, att, np.arange(len(pts)), max_opening=OPEN, radius=0.012, min_points=20, min_ratio=2.0,
                                 wide=True, partition=True)
    assert dec is not None and dec.accepted
    tgt = np.asarray(dec.target.indices)
    assert set(tgt.tolist()) == set(range(na))


def test_on_keeps_one_wide_object_whole_and_never_partitions():
    crate = _grid((0.0, 0.0, 0.0), (0.20, 0.18, 0.15))
    att = np.linspace(0.0, 1.0, len(crate))                             # uneven attention would invite a partition cut
    dec = asp.split_by_attention(crate, att, np.arange(len(crate)), max_opening=OPEN, radius=0.012, min_points=20, min_ratio=2.0,
                                 wide=True, partition=True)
    assert dec is not None and not dec.accepted and dec.reason == asp.ONE_COMPONENT


def test_narrow_trigger_unchanged():
    long_thin = _grid((0.0, 0.0, 0.0), (0.03, 0.05, 0.20))
    att = np.full(len(long_thin), 0.5)
    a = asp.split_by_attention(long_thin, att, np.arange(len(long_thin)), max_opening=OPEN, radius=0.012, min_points=20, min_ratio=2.0)
    b = asp.split_by_attention(long_thin, att, np.arange(len(long_thin)), max_opening=OPEN, radius=0.012, min_points=20, min_ratio=2.0, wide=True)
    assert (a is None) == (b is None) and (a is None or (a.reason == b.reason and a.accepted == b.accepted))


def test_plumbing():
    from AG3S.ag3s.config import AG3SConfigError, ClusteringConfig
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe

    assert ClusteringConfig().attention_split_wide is False
    ClusteringConfig(attention_split_radius=0.012, attention_split_wide=True).validate()
    with pytest.raises(AG3SConfigError):
        ClusteringConfig(attention_split_radius=None, attention_split_wide=True).validate()
    name = FROZEN[:-len("-tro")] + "-gh156a-sw-tro"
    assert pf.SOTA_SPLITWIDE_VERSIONS == (name,) and name in pf.SOTA_VERSIONS
    assert "attention_split_wide" in srv.PORTED_FINAL_CONFIG_DESTS
    args = pf.parse_v11ab_args(argv=pf.version_server_argv(name) + list(pf.AG3S_VERSIONS[name]))
    assert serve_safe.t43t_identity_section(args)["attention_split_wide"] is True
    base = pf.parse_v11ab_args(argv=pf.version_server_argv(FROZEN) + list(pf.AG3S_VERSIONS[FROZEN]))
    assert "attention_split_wide" not in serve_safe.t43t_identity_section(base)
    assert [v for v, f in pf.AG3S_VERSIONS.items() if "--attention-split-wide" in f] == [name]
