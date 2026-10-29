"""AG3S-GPU-STAGES M1 — sub-stage timing is measurement only.

  - off (the default), `part()` records nothing;
  - on, a part's time excludes the parts nested inside it;
  - grounding returns the same result with the timing on and off, and names its parts.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from AG3S.ag3s.config import ClusteringConfig
from AG3S.ag3s.runtime import substage
from AG3S.ag3s.stages.target_grounding import TargetConfirm, ground_target
from AG3S.ag3s.types import AttentionPointCloud, PointCloud


@pytest.fixture
def enabled():
    substage.pop()
    substage.ENABLED = True
    yield
    substage.ENABLED = False
    substage.pop()


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal([0.5, 0.0, 0.8], 0.02, (400, 3))
    b = rng.normal([0.5, 0.2, 0.8], 0.02, (400, 3))
    table = np.c_[rng.uniform(0.3, 0.7, 800), rng.uniform(-0.2, 0.4, 800), np.full(800, 0.75)]
    pts = np.vstack([a, b, table])
    att = np.r_[np.full(400, 0.9), np.full(400, 0.2), np.zeros(800)].astype(np.float32)
    att += rng.uniform(0, 0.01, len(att)).astype(np.float32)
    support = np.r_[np.zeros(800, bool), np.ones(800, bool)]
    return AttentionPointCloud(PointCloud(points=pts), att, att.copy()), support


def test_off_records_nothing():
    substage.pop()
    assert substage.ENABLED is False
    with substage.part("x"):
        time.sleep(0.001)
    assert substage.pop() == {}


def test_nested_parts_are_exclusive(enabled):
    with substage.part("outer"):
        time.sleep(0.02)
        with substage.part("inner"):
            time.sleep(0.02)
    parts = substage.pop()
    assert set(parts) == {"outer", "inner"}
    assert 15.0 <= parts["inner"] <= 60.0
    assert 15.0 <= parts["outer"] <= 60.0   # not ≈ 40: the inner 20 ms is subtracted
    assert substage.pop() == {}


def test_grounding_unchanged_and_named(enabled):
    cloud, support = _scene()
    cfg = ClusteringConfig()
    on = ground_target(cloud, cfg, exclude_mask=support, timestamp=0.0, confirm=TargetConfirm())
    parts = substage.pop()
    substage.ENABLED = False
    off = ground_target(cloud, cfg, exclude_mask=support, timestamp=0.0, confirm=TargetConfirm())
    assert on.status == off.status
    assert np.array_equal(on.target.point_indices, off.target.point_indices)
    assert np.array_equal(on.seed_indices, off.seed_indices)
    assert {"grounding.seeds", "grounding.grow", "grounding.components",
            "grounding.score", "grounding.confirm"} <= set(parts)
    assert all(v >= 0.0 for v in parts.values())
