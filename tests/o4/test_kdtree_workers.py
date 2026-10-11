"""T38 B1 — every AG3S KD-tree query runs single-worker, and the answer is the same as `workers=-1`.

`workers=-1` spawned one thread per core per query (256 cores here, ≈ 2,138 threads per chunk) and
cost hundreds of ms per chunk for queries over a few thousand points (T38.profile.verify.json
`micro_kdtree`). A KD-tree query is a pure per-point read of the same tree, so the worker count
cannot change the result; these tests pin both halves of that statement.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

config = pytest.importorskip("AG3S.ag3s.config")
if not hasattr(config, "KDTREE_WORKERS"):
    pytest.skip("KDTREE_WORKERS absent — this is not the O4 worktree", allow_module_level=True)

import benchmark  # noqa: E402
from AG3S.ag3s.stages import robot_filter as RF  # noqa: E402
from AG3S.ag3s.stages import target_grounding as TG  # noqa: E402

PKG = pathlib.Path(benchmark.__file__).parent


def test_import_location_is_reported():
    # Not an assertion about where it *must* be (CI may differ) — the header prints it; this keeps
    # the path in the -v output next to the results it explains.
    print("benchmark:", benchmark.__file__)


def test_the_worker_count_is_one():
    assert config.KDTREE_WORKERS == 1
    assert TG.KDTREE_WORKERS == 1 and RF.KDTREE_WORKERS == 1


def test_no_server_path_query_spawns_a_thread_per_core():
    """Offline experiment scripts may keep `-1` (one big query, where threads pay); the server path
    — stages, runtime, fields, constraints, robot_models, trajopt — may not."""
    offenders = []
    for sub in ("ag3s", "trajopt"):
        for f in (PKG / sub).rglob("*.py"):
            rel = f.relative_to(PKG).as_posix()
            if "/experiments/" in rel or "/asset/" in rel:
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                code = line.split("#", 1)[0]
                if re.search(r"workers\s*=\s*-1", code):
                    offenders.append(f"{rel}:{i}")
    assert offenders == []


def _cloud(seed, n=6000):
    """Three blobs and a sparse bridge — enough structure for seeds, borders and noise."""
    rng = np.random.default_rng(seed)
    blobs = [rng.normal(c, 0.025, size=(n // 3, 3)) for c in ((0.5, 0.0, 0.1), (0.6, 0.1, 0.1),
                                                             (0.4, -0.15, 0.05))]
    bridge = rng.uniform((0.4, -0.2, 0.0), (0.7, 0.2, 0.2), size=(200, 3))
    return np.concatenate(blobs + [bridge])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_grow_region_is_bitwise_the_same_with_all_workers(monkeypatch, seed):
    pts = _cloud(seed)
    seeds = np.arange(0, 300, 7)
    one = TG.grow_region(pts, seeds, 0.012)
    monkeypatch.setattr(TG, "KDTREE_WORKERS", -1)
    many = TG.grow_region(pts, seeds, 0.012)
    assert one.dtype == many.dtype and np.array_equal(one, many)
    assert one.size > seeds.size  # it actually grew


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dbscan_is_bitwise_the_same_with_all_workers(monkeypatch, seed):
    pts = _cloud(seed)
    one = TG.dbscan(pts, 0.01, 8)
    monkeypatch.setattr(TG, "KDTREE_WORKERS", -1)
    many = TG.dbscan(pts, 0.01, 8)
    assert np.array_equal(one, many)
    assert (one >= 0).any() and (one == -1).any()  # clusters and noise both present


@pytest.mark.parametrize("guard", [False, True])
def test_robot_sphere_mask_is_bitwise_the_same_with_all_workers(monkeypatch, guard):
    pts = _cloud(3)
    rng = np.random.default_rng(7)
    centres = rng.uniform((0.35, -0.2, 0.0), (0.7, 0.2, 0.2), size=(587, 3))
    radii = rng.uniform(0.005, 0.03, size=587)
    kw = dict(inflation=0.01, return_protected=True)
    if guard:
        kw.update(guard_centre=(0.5, 0.0, 0.1), guard_radius=0.05)
    one = RF.robot_sphere_mask(pts, centres, radii, **kw)
    monkeypatch.setattr(RF, "KDTREE_WORKERS", -1)
    many = RF.robot_sphere_mask(pts, centres, radii, **kw)
    for a, b in zip(one, many):
        assert np.array_equal(a, b)
    assert one[0].any() and not one[0].all()
