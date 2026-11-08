"""SL-PE1 — `stages.close_seed.split_at_seed` on synthetic scenes (2026-11-08, dev-30)."""
import numpy as np

from AG3S.ag3s.stages.close_seed import CloseSeedParams, split_at_seed, upward_mask, vertical_walls

RNG = np.random.default_rng(3)


def plane(x0, x1, y0, y1, z, step=0.006):
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, z)]


def wall_x(x, y0, y1, z0, z1, step=0.006):
    ys, zs = np.meshgrid(np.arange(y0, y1, step), np.arange(z0, z1, step))
    return np.c_[np.full(ys.size, x), ys.ravel(), zs.ravel()]


def bowl(cx, cy, z0, r=0.055, h=0.05, step=0.006):
    """open cylinder wall + inner bottom disc, as a depth camera above would see it."""
    out = []
    for z in np.arange(z0, z0 + h, step):
        n = int(2 * np.pi * r / step)
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        out.append(np.c_[cx + r * np.cos(a), cy + r * np.sin(a), np.full(n, z)])
    d = plane(cx - r, cx + r, cy - r, cy + r, z0 + 0.004)
    d = d[np.hypot(d[:, 0] - cx, d[:, 1] - cy) < r - 0.008]
    out.append(d)
    return np.concatenate(out)


def label(*parts):
    pts = np.concatenate([p for p, _ in parts])
    lab = np.concatenate([np.full(len(p), k) for p, k in parts])
    return pts + RNG.normal(0, 0.001, pts.shape), lab


def test_bowl_on_cabinet_top_is_cut_from_the_cabinet():
    top = 0.20
    cab = np.concatenate([plane(0.55, 0.85, -0.30, 0.0, top), wall_x(0.55, -0.30, 0.0, 0.0, top)])
    pts, lab = label((cab, "cabinet"), (bowl(0.70, -0.15, top + 0.002), "bowl"))
    seed = np.array([0.70 - 0.055, -0.15, top + 0.035])   # finger tips at the rim
    idx, rec = split_at_seed(pts, seed)
    assert idx is not None, rec
    got = lab[idx]
    assert (got == "bowl").mean() >= 0.95
    # by design the band up to ~1.4 cm above the support top goes with the support: recall on the wall above it
    above = (lab == "bowl") & (pts[:, 2] > top + 0.018)
    assert np.isin(np.flatnonzero(above), idx).mean() >= 0.9
    assert rec["local_support_top_m"] is not None and abs(rec["local_support_top_m"] - top) <= 0.012


def test_bowl_in_drawer_walls_are_removed():
    floor = 0.15
    drawer = np.concatenate([plane(0.60, 0.84, -0.25, -0.03, floor),
                             wall_x(0.60, -0.25, -0.03, floor, floor + 0.07),
                             wall_x(0.84, -0.25, -0.03, floor, floor + 0.07),
                             plane(0.60, 0.84, -0.25, -0.24, floor + 0.07)])   # front panel top edge
    b = bowl(0.665, -0.14, floor + 0.002)        # touching the left wall
    pts, lab = label((drawer, "drawer"), (b, "bowl"))
    seed = np.array([0.665 + 0.055, -0.14, floor + 0.04])
    idx, rec = split_at_seed(pts, seed)
    assert idx is not None, rec
    got = lab[idx]
    assert (got == "bowl").mean() >= 0.85, (rec, (got == "bowl").mean())
    above = (lab == "bowl") & (pts[:, 2] > floor + 0.018)
    assert np.isin(np.flatnonzero(above), idx).mean() >= 0.8, rec


def test_bowl_on_table_with_support_excluded():
    tab = plane(0.3, 0.8, -0.3, 0.3, 0.0)
    pts, lab = label((tab, "table"), (bowl(0.5, 0.0, 0.002), "bowl"), (bowl(0.5, 0.16, 0.002), "other"))
    excl = lab == "table"
    idx, rec = split_at_seed(pts, [0.5 + 0.055, 0.0, 0.03], exclude_mask=excl)
    assert idx is not None, rec
    assert set(lab[idx]) == {"bowl"}


def test_seed_far_from_everything_gives_none():
    pts, _ = label((bowl(0.5, 0.0, 0.0), "bowl"),)
    idx, rec = split_at_seed(pts, [0.5, 0.0, 0.40])
    assert idx is None and rec["why"]


def test_helpers():
    p = plane(0, 0.1, 0, 0.1, 0.0)
    assert upward_mask(p).mean() > 0.9
    w = wall_x(0.0, 0.0, 0.2, 0.0, 0.1)
    assert vertical_walls(np.concatenate([w, bowl(0.2, 0.1, 0.0)])).sum() >= 0.9 * len(w)
    assert CloseSeedParams().grid == 0.010
