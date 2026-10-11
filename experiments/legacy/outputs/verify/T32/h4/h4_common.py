"""T32 H4 — shared harness: build a field with the AG3S cuRobo builder exactly as the server does, then read it.

Import path: the snapshot ${AG3S_PROJECT_ROOT}/outputs/verify/T31diag/snap (md5-identical to the T30 server code for
ag3s/fields/curobo_builder.py, curobo_field.py, runtime/pipeline.py; see T32.h4.verify.json code_state).

Build path = CuroboFieldBuilder._ensure_mapper -> _integrate (per frame, same Mapper) -> _build_tiers
(seed -> [attached seed exclusion] -> propagate -> copy -> _force_sign_on_pure_voxels), i.e. the body of
CuroboFieldBuilder.update() minus labels/stats/window placement. The fine window centre is given explicitly
(from the recorded esdf_origin for the real chunk; a fixed centre for the synthetic wall).
Config = AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": {...}}) with the same esdf dict
trajopt/serve_safe.py builds (voxel_size, max_distance 0.4, exclude_support_surfaces False, backend curobo,
attached_sign_threshold_voxels, fine_voxel_size, tsdf_voxel_size) + truncation_voxels when varied.
MapperCfg knobs the builder does not expose (seeding_method, minimum_tsdf_weight) are varied by wrapping
curobo.perception.MapperCfg.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, functools
import numpy as np

SNAP = _ag3s_workspace_path('outputs/verify/T31diag/snap')
if SNAP not in sys.path:
    sys.path.insert(0, SNAP)
from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder
from AG3S.ag3s.fields.curobo_field import layer_from_arrays
from AG3S.ag3s.fields.esdf import CameraDepth

SERVER = dict(voxel=0.020, fine=0.005, tsdf=0.005, trunc_vox=3.0, thr=1.5, seeding="gather", min_weight=0.1)


def make_builder(voxel=0.020, fine=0.005, tsdf=0.005, trunc_vox=3.0, thr=1.5, seeding="gather",
                 min_weight=0.1, **_):
    esdf = {"voxel_size": voxel, "max_distance": 0.4, "exclude_support_surfaces": False,
            "backend": "curobo", "attached_sign_threshold_voxels": thr,
            "fine_voxel_size": fine or None, "tsdf_voxel_size": tsdf or None,
            "truncation_voxels": trunc_vox}
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": esdf}).esdf
    b = CuroboFieldBuilder(cfg)
    b._h4_mapper_overrides = {"seeding_method": seeding, "minimum_tsdf_weight": float(min_weight)}
    return b


def _patched_mapper(builder, cams):
    import curobo.perception as cp
    orig = cp.MapperCfg
    cp.MapperCfg = functools.partial(orig, **builder._h4_mapper_overrides)
    try:
        return builder._ensure_mapper(cams)
    finally:
        cp.MapperCfg = orig


def build(builder, frames, fine_centre, attached=None):
    """frames: list of list[CameraDepth]. Returns dict with coarse/fine layers + sign stats + the integrator."""
    import torch
    mapper = None
    for cams in frames:
        mapper = _patched_mapper(builder, cams)
        dev = mapper._device if hasattr(mapper, "_device") else "cuda:0"
        builder._integrate(mapper, cams, dev)
    itg = mapper.integrator
    builder._fine_centre = None if fine_centre is None else np.asarray(fine_centre, np.float64)
    att = None if attached is None or not len(attached) else np.asarray(attached, np.float64)
    tiers = builder._build_tiers(itg, att, dev, torch)
    out = {"itg": itg, "mapper": mapper, "tiers": {}}
    for t in tiers:
        lay = layer_from_arrays(t.values, t.origin, t.voxel_size)
        out["tiers"][t.name] = {"layer": lay, "values": t.values, "origin": np.asarray(t.origin), "vs": t.voxel_size,
                                "extra": dict(t.extra)}
    out["truncation_m"] = float(builder.config.truncation)
    out["mapper_cfg"] = {"truncation_distance": float(itg.truncation_distance), "tsdf_voxel": float(itg.voxel_size),
                         "tsdf_grid_shape_zyx": [int(v) for v in itg.grid_shape],
                         "seeding_method": mapper.integrator.config.seeding_method if hasattr(mapper.integrator, "config") else None,
                         "minimum_tsdf_weight": float(mapper.integrator.config.minimum_tsdf_weight) if hasattr(mapper.integrator, "config") else None}
    return out


def fine_centre_from_origin(origin, shape, vs):
    """inverse of CuroboFieldBuilder._origin_of: first-voxel centre -> grid centre."""
    return np.asarray(origin, float) + 0.5 * np.asarray(shape, float) * vs - 0.5 * vs


# ---------------------------------------------------------------- dense TSDF readout (sdf, weight) at points
def tsdf_at(itg, pts, min_weight=None):
    """(sdf, weight) of the TSDF voxel containing each point; sdf = nan where unallocated / weight <= min_weight."""
    ts = itg.tsdf
    data = ts.data if hasattr(ts, "data") else ts
    bd = data.block_data.detach().float().cpu().numpy()
    bc = data.block_coords.detach().cpu().numpy().reshape(-1, 3)
    slot = data.block_to_hash_slot.detach().cpu().numpy()
    nal = int(data.num_allocated.detach().cpu().numpy().reshape(-1)[0])
    D, H, W = (int(v) for v in itg.grid_shape)
    vs = float(itg.voxel_size)
    o = itg.origin.detach().cpu().numpy().reshape(-1)[:3].astype(np.float64)
    BS = int(round(bd.shape[1] ** (1 / 3)))
    off = np.array([-(-W // BS) // 2, -(-H // BS) // 2, -(-D // BS) // 2])
    valid = np.arange(len(slot)) < nal
    valid &= slot >= 0
    keys = bc[valid]; pools = np.nonzero(valid)[0]
    lut = {tuple(k): p for k, p in zip(keys.tolist(), pools.tolist())}
    pts = np.asarray(pts, np.float64)
    g = np.floor((pts - o) / vs + np.array([W, H, D]) * 0.5).astype(np.int64)
    bgrid = g // BS; loc = g % BS
    key = bgrid - off
    sdf = np.full(len(pts), np.nan); w = np.zeros(len(pts))
    mw = 0.1 if min_weight is None else min_weight
    for i in range(len(pts)):
        p = lut.get(tuple(key[i]))
        if p is None:
            continue
        li = loc[i, 2] * BS * BS + loc[i, 1] * BS + loc[i, 0]
        s, ww = bd[p, li, 0], bd[p, li, 1]
        w[i] = ww
        if ww > mw:
            sdf[i] = s / ww
    return sdf, w


# ---------------------------------------------------------------- line profiles
def line_points(p0, u, half=0.15, step=0.0005):
    s = np.arange(-half, half, step)
    return s, np.asarray(p0, float)[None, :] + s[:, None] * np.asarray(u, float)[None, :]


def inside(layer, pts):
    g = layer.grid; last = g.origin + (np.asarray(g.shape) - 1) * g.voxel_size
    return np.all((pts >= g.origin) & (pts <= last), axis=1)


def segments(s, mask):
    out = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append([float(s[i]), float(s[j])]); i = j + 1
        else:
            i += 1
    return out


def band(layer, p0, u, half=0.15, step=0.0005, split=None):
    """d<0 width along the line (mm; 0.5 mm samples, EsdfField trilinear distance - same as T31 D4 wall_profile).
    split: signed offset (m) along u; widths are also reported for s<split and s>=split."""
    s, pts = line_points(p0, u, half, step)
    ins = inside(layer, pts)
    if not ins.all():
        return {"covered": False, "covered_fraction": float(ins.mean())}
    v = np.asarray(layer.distance(pts), np.float64)
    r = {"covered": True, "width_lt0_mm": float(1000 * step * (v < 0).sum()),
         "width_lt5mm_mm": float(1000 * step * (v < 0.005).sum()), "min_mm": float(1000 * v.min()),
         "segments_lt0_m": segments(s, v < 0)}
    if split is not None:
        a = s < split
        r["width_lt0_mm_lo"] = float(1000 * step * (v[a] < 0).sum())
        r["width_lt0_mm_hi"] = float(1000 * step * (v[~a] < 0).sum())
    return r, s, v


def voxel_samples(layer, p0, u, half=0.15):
    """voxel-centre values along the line (nearest voxel), for plotting the raw grid."""
    g = layer.grid
    s = np.arange(-half, half, g.voxel_size * 0.5)
    pts = np.asarray(p0)[None] + s[:, None] * np.asarray(u)[None]
    idx = np.floor((pts - g.origin) / g.voxel_size + 0.5).astype(int)
    ok = np.all((idx >= 0) & (idx < np.asarray(g.shape)), axis=1)
    vals = np.full(len(s), np.nan)
    arr = layer_values(layer)
    vals[ok] = arr[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return s, vals


def layer_values(layer):
    for name in ("values", "_values", "distance_grid", "_distance", "grid_values", "data"):
        v = getattr(layer, name, None)
        if isinstance(v, np.ndarray) and v.ndim == 3:
            return v
    raise AttributeError("no 3D array on layer: " + ", ".join(vars(layer)))


def eikonal(values, vs, mask_free=0.02, cap=0.35):
    """|grad d| by central differences on voxel centres where every 6-neighbour is > mask_free and < cap."""
    v = np.asarray(values, np.float64)
    c = v[1:-1, 1:-1, 1:-1]
    gx = (v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / (2 * vs)
    gy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * vs)
    gz = (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2 * vs)
    nb = np.stack([v[2:, 1:-1, 1:-1], v[:-2, 1:-1, 1:-1], v[1:-1, 2:, 1:-1], v[1:-1, :-2, 1:-1],
                   v[1:-1, 1:-1, 2:], v[1:-1, 1:-1, :-2], c])
    ok = np.all((nb > mask_free) & (nb < cap), axis=0)
    gm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)[ok]
    if not gm.size:
        return {"n": 0}
    return {"n": int(gm.size), "median": float(np.median(gm)), "p05": float(np.percentile(gm, 5)),
            "p95": float(np.percentile(gm, 95)), "frac_within_0.9_1.1": float(((gm > 0.9) & (gm < 1.1)).mean()),
            "mask": f"all 6 neighbours and centre in ({mask_free}, {cap}) m"}
