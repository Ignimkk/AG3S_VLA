"""T43 DX — shared helpers (verifier, CPU only, recorded data only).

Process rule: one process per client snapshot (hold_w reads T43W_SNAP at import) -> set T43W_SNAP before importing this.
Run index: analysis json of each arm (server_segment = analyze_t41 pairing rule, as T43/W/p/analyze_t43w.py):
  V1 V3 V4 V5 E0 : T43/W/analysis/analysis.json        V5S V5B : T43/W/analysis_blind/analysis.json
  V5D            : T43/Z3V/analysis/analysis.json      V7g     : T43/CT/raw/analysis_v7g.json (CT rerun; V7g/analysis was stale)
Chunk of a control row = the server chunk npz of the segment whose t_step == that row's chunk_seq (the executing chunk).
Frames: base frame of the field = MuJoCo body 'base' (world <- base transform from the row qpos).
ESDF sampling: own trilinear on the recorded float16 grids (origin = centre of voxel (0,0,0), frame_record.py:157), NaN outside
  the grid; plus the composed TO field FIA.build_field (coarse -> fine; target-free layer) with distance / answer_tier.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import ast, json, os, pathlib, sys
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, mujoco
from scipy.ndimage import map_coordinates

T43 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
ANALYSES = {"W": T43 / "W/analysis/analysis.json", "WB": T43 / "W/analysis_blind/analysis.json",
            "Z3V": T43 / "Z3V/analysis/analysis.json", "V7g": T43 / "CT/raw/analysis_v7g.json"}
ARM_SRC = {"V1": "W", "V3": "W", "V4": "W", "V5": "W", "E0": "W", "V5S": "WB", "V5B": "WB", "V5D": "Z3V", "V7g": "V7g"}
SNAP_OF = {"V1": "${AG3S_PROJECT_ROOT}-t43w", "V3": "${AG3S_PROJECT_ROOT}-t43w", "V4": "${AG3S_PROJECT_ROOT}-t43w", "E0": "${AG3S_PROJECT_ROOT}-t43w",
           "V5": "${AG3S_PROJECT_ROOT}-t43w3", "V5S": "${AG3S_PROJECT_ROOT}-t43b", "V5B": "${AG3S_PROJECT_ROOT}-t43b",
           "V5D": "${AG3S_PROJECT_ROOT}-t43z3", "V7g": "${AG3S_PROJECT_ROOT}-t43v7"}
LEFT_Q = list(range(20, 27)); RIGHT_Q = list(range(11, 18))


def lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


_AN = {}


def run_entry(key):
    arm = key.split("/")[0]; src = ARM_SRC[arm]
    if src not in _AN:
        _AN[src] = json.load(open(ANALYSES[src]))["runs"]
    return _AN[src][key]


def frames(rd):
    rows = [json.loads(l) for l in open(pathlib.Path(rd) / "frames.jsonl") if l.strip()]
    ctrl = {r["t_step"]: r for r in rows if r["kind"] == "control"}
    plan = {r["seq"]: r for r in rows if r["kind"] == "planning"}
    return ctrl, plan


def segment_chunks(seg):
    """{t_step(seq): path} of the run's server segment."""
    first = pathlib.Path(seg["first"]); i0 = int(first.stem.split("_")[1]); out = {}
    for n in range(int(seg["n"])):
        p = first.parent / f"chunk_{i0 + n:05d}.npz"
        with np.load(p, allow_pickle=True) as z:
            out[int(z["t_step"])] = p
    return out


def samp(grid, origin, vox, p):
    """trilinear on a recorded grid; NaN outside. p (N,3) base frame -> (N,) metres."""
    p = np.atleast_2d(np.asarray(p, float))
    idx = (p - np.asarray(origin, float)) / float(vox)
    shp = np.asarray(grid.shape) - 1
    out = np.full(len(p), np.nan)
    ok = np.all((idx >= 0) & (idx <= shp), axis=1)
    if ok.any():
        out[ok] = map_coordinates(np.asarray(grid, np.float32), idx[ok].T, order=1)
    return out


def grid_index(origin, vox, shape, p):
    idx = (np.asarray(p, float) - np.asarray(origin, float)) / float(vox)
    inside = bool(np.all(idx >= 0) and np.all(idx <= np.asarray(shape) - 1))
    return idx.tolist(), inside


def qmat(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)


def T_world_base(m, d):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    T = np.eye(4); T[:3, :3] = d.xmat[b].reshape(3, 3); T[:3, 3] = d.xpos[b]; return T


def to_base(T, P):
    P = np.atleast_2d(P); return (P - T[:3, 3]) @ T[:3, :3]


def geom_point_sd(m, d, g, P):
    """signed distance (m) from points P (N,3) world to collision geom g (capsule 3, cylinder 5, box 6, sphere 2)."""
    P = np.atleast_2d(np.asarray(P, float)); c = d.geom_xpos[g]; R = d.geom_xmat[g].reshape(3, 3); s = m.geom_size[g]
    q = (P - c) @ R; t = int(m.geom_type[g])
    if t == 3:
        z = np.clip(q[:, 2], -s[1], s[1]); return np.linalg.norm(q - np.c_[np.zeros_like(z), np.zeros_like(z), z], axis=1) - s[0]
    if t == 5:
        rad = np.linalg.norm(q[:, :2], axis=1) - s[0]; ax = np.abs(q[:, 2]) - s[1]
        return np.minimum(np.maximum(rad, ax), 0) + np.linalg.norm(np.c_[np.maximum(rad, 0), np.maximum(ax, 0)], axis=1)
    if t == 6:
        dv = np.abs(q) - s[:3]; return np.linalg.norm(np.maximum(dv, 0), axis=1) + np.minimum(dv.max(1), 0)
    if t == 2:
        return np.linalg.norm(q, axis=1) - s[0]
    raise ValueError(f"geom type {t}")


def hurdle_sd(m, d, geoms, P):
    return np.min([geom_point_sd(m, d, g, P) for g in geoms], axis=0)


def geom_axis_point(m, d, g, p):
    """closest point on the geom's axis segment (capsule / cylinder) to world point p."""
    c = d.geom_xpos[g]; R = d.geom_xmat[g].reshape(3, 3); s = m.geom_size[g]
    z = float(np.clip((np.asarray(p) - c) @ R[:, 2], -s[1], s[1])); return c + z * R[:, 2]


def server_settings(seg, manifest_path):
    from AG3S.trajopt.experiments import fixed_input_ablation as FIA

    class _Args:
        links = None; sphere_spacing = None; max_spheres_per_capsule = None; capsule_radius_scale = None
        max_sphere_radius = None; esdf_margin = None; target_field_policy = None; phase = None
        manipulators = None; model_xml = None; hand = None; allow_uncertified = False; rows_per_step = None
    first = pathlib.Path(seg["first"]); log = first.parent.parent.parent / "serve_safe.log"
    st = FIA.resolve_settings(_Args, FIA.parse_server_log(str(log)), json.load(open(manifest_path)))
    _, _, _, maxd, unk = FIA.contact_policy(st, ["ee_left"], has_target=False, has_free=False, attached=False)
    outside = maxd if unk == "free" else -maxd
    om = None
    for line in open(log, errors="replace"):
        if "TO obstacle_margin:" in line:
            om = float(line.split("TO obstacle_margin:")[1].split("mm")[0]); break
    return st, outside, om, str(log)


def held_sets(summary):
    """(query set name, {name: [(centre_parent_frame (3,), radius_m)]}, frame_link, body_cover dict or None)."""
    g = summary.get("grasp") or {}; att = lit(g.get("attached"))
    if not isinstance(att, dict):
        return None, {}, None, None
    held = lit(att.get("held")) or {}; bc = lit(held.get("body_cover"))
    sets = {}
    for nm, src in (("body_parent_frame", bc), ("lifted_parent_frame", bc), ("spheres_parent_frame", held), ("filter_spheres_parent_frame", held)):
        L = (src or {}).get(nm) if isinstance(src, dict) else None
        if L:
            sets[nm] = [(np.asarray(x["centre_m"], float), 1e-3 * float(x["radius_mm"])) for x in L]
    if isinstance(bc, dict) and bc:
        near = bool((bc.get("frame") or {}).get("near_support"))
        used = "lifted_parent_frame" if near else "body_parent_frame"
    else:
        used = "spheres_parent_frame"
    return used, sets, held.get("frame_link") or att.get("parent_link"), bc if isinstance(bc, dict) else None


def place(T_body, loc):
    C = np.array([T_body[:3, :3] @ c + T_body[:3, 3] for c, _ in loc]); r = np.array([rr for _, rr in loc]); return C, r


def body_T(m, d, name):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    T = np.eye(4); T[:3, :3] = d.xmat[b].reshape(3, 3); T[:3, 3] = d.xpos[b]; return T


def union_sd(P, C, r):
    """signed distance of points to the union of spheres (< 0 inside)."""
    P = np.atleast_2d(P); return (np.linalg.norm(P[:, None, :] - C[None], axis=2) - r[None]).min(1)
