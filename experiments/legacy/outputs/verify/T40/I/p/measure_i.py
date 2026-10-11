"""T40 I (verifier, records only, CPU) — E3b approach-phase finger/EE contact with the hurdle bar.

Inputs (all recorded; no server, no GPU):
  client  : outputs/verify/T40/G/runs/E3b/{C,D}/<run>/frames.jsonl (planning + control rows, qpos 66)
  server  : .../server_<port>/server_constraints/run_0000/chunk_*.npz (--record-constraints, esdf=full):
            coarse 20 mm / fine 5 mm / target-free (esdf_free) grids, sphere_centres, clearance (8,122),
            reference_chunk, refined_chunk, summary_json
  hurdle  : the run's obstacle config + profile, placed with the snapshot client code
            (hurdle_metrics.manager -> PickPlaceObstacleManager.activate)
Code: snapshot ${AG3S_PROJECT_ROOT}-t40 (PYTHONPATH), venv .venv-openpi-live, CPU only.

Pairing: server npz t_step == client planning seq; refined_chunk vs planning 'actions' must agree <= 1e-5.
Robot model: rebuilt like fixed_input_ablation.build_robot (links=gripper, sphere options from serve_safe.log).
Finger parameters: server-side path = FingerJointMap.path(summary.finger_joints.opening_norm, reference gripper columns);
                   at measured MuJoCo poses = MuJoCo finger joint qpos.
Base frame -> world: MuJoCo body 'base' pose at the row's qpos (hurdle_metrics.base_to_world).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, math, pathlib, datetime
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/p'))
import numpy as np, mujoco
import hurdle_metrics as HM
from AG3S.trajopt.experiments import fixed_input_ablation as FIA
from AG3S.ag3s.fields.curobo_field import layer_from_arrays, CuroboEsdfField
from AG3S.ag3s.robot_models.gripper_state import FingerJointMap
from AG3S.trajopt.types import ChunkLayout
from AG3S.trajopt.limits import build_limits

assert FIA.__file__.startswith("${AG3S_PROJECT_ROOT}-t40/"), FIA.__file__
R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/G/runs/E3b'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/I/raw'))
S1 = R / "server_8240_20261003T135313"; S2 = R / "server_8242_20261004T054234"
A = json.load(open(_ag3s_workspace_path('outputs/verify/T40/G/analysis_all.json')))["runs"]

# event_t: failing runs = obstacle-stop control row (contact seen in the sim steps after that row);
#          successful runs = control row of the minimum gripper(A)-to-hurdle distance in the carry phase (analysis_all)
CASES = {
    "C1807": dict(key="E3b/C/E3b_ep1807_s18071", srv=S1, chunks=(103, 130), kind="contact"),
    "C1967": dict(key="E3b/C/E3b_ep1967_s19672", srv=S1, chunks=(578, 603), kind="contact"),
    "C1982": dict(key="E3b/C/E3b_ep1982_s19822", srv=S1, chunks=(317, 339), kind="contact"),
    "C1968": dict(key="E3b/C/E3b_ep1968_s19681", srv=S1, chunks=(206, 241), kind="pass"),
    "D1807": dict(key="E3b/D/E3b_ep1807_s18071", srv=S2, chunks=(0, 34), kind="pass"),
}
N_WINDOW = 5
LEFT_LINKS = ("ee_left", "ee_finger_l1", "ee_finger_l2")
MJ_BODY = {}  # urdf link -> MuJoCo body, filled by build_mapping()
GEOM_LINK = {}  # filled: mujoco geom id -> urdf link name


def jl(o):
    if isinstance(o, dict): return {str(k): jl(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jl(v) for v in o]
    if isinstance(o, np.ndarray): return jl(o.tolist())
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


def mm(x): return None if x is None else float(x) * 1000.0


def stats_mm(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not v.size: return None
    return {"n": int(v.size), "min": mm(v.min()), "median": mm(np.median(v)), "max": mm(v.max())}


# ---------------------------------------------------------------- setup (robot model as the server built it)
class Args: links = None; sphere_spacing = None; max_spheres_per_capsule = None; capsule_radius_scale = None
Args.max_sphere_radius = None; Args.esdf_margin = None; Args.target_field_policy = None; Args.phase = None
Args.manipulators = None; Args.model_xml = None; Args.hand = None; Args.allow_uncertified = False; Args.rows_per_step = None
log = FIA.parse_server_log(str(S1 / "serve_safe.log"))
manifest = json.load(open(R / "C/E3b_ep1807_s18071/manifest.json"))
manifest["scene"]["model_xml"] = "${AG3S_PROJECT_ROOT}-t40/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
settings = FIA.resolve_settings(Args, log, manifest)
robot, qadr, xml_path = FIA.build_robot(settings)
assert robot.n_spheres == settings.n_constraint_spheres == 122
layout = ChunkLayout.rby1(robot.joint_names)
_, _, _, max_distance, unknown_policy = FIA.contact_policy(settings, robot.sphere_link_names, has_target=False, has_free=False, attached=False)
OUTSIDE = max_distance if unknown_policy == "free" else -max_distance
planning = FIA.to_config(settings, collision=True, cost=settings.cost, sqp={})
limits = build_limits(robot, layout, dt=planning.horizon.dt, config=planning.limits)
meter = FIA.Meter(robot, layout, planning.horizon.planned, settings.hand, limits)
H = meter.lin.horizon
NAMES = np.array([str(n) for n in robot.sphere_link_names])
RADII = np.asarray(robot.radii, float)
fmap = FingerJointMap(robot)
PARAM_NAMES = tuple(robot.param_joint_names)
DEFAULT_P = robot.param_vector()

M = HM.model(); m, d = M["m"], M["d"]
MAPPING = {}
mj_jadr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): int(m.jnt_qposadr[i]) for i in range(m.njnt)}
assert all(p in mj_jadr for p in PARAM_NAMES), PARAM_NAMES


def gname(g): return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom#{int(g)}"


def mj_set(q):
    d.qpos[:] = q; mujoco.mj_kinematics(m, d)


def base_T(q):
    mj_set(q); b = M["bid"]["base"]
    return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


def to_base(pw, q):
    p0, Rb = base_T(q); return (np.asarray(pw) - p0) @ Rb


def to_world(pb, q):
    p0, Rb = base_T(q); return np.asarray(pb) @ Rb.T + p0


def params_from_qpos(q): return np.array([q[mj_jadr[p]] for p in PARAM_NAMES])


def spheres_at_qpos(q):
    """URDF sphere model at a MuJoCo qpos (arm + finger joints), base frame."""
    c, r = robot.sphere_centers_numeric(FIA.q_now_from_qpos(q, qadr), params_from_qpos(q))
    return c, r


def spheres_path(traj, q_now, P):
    full = layout.full_q(traj, q_now)
    return np.stack([robot.sphere_centers_numeric(full[:, k], P[:, k])[0] for k in range(full.shape[1])])



def qpos_with_traj(qpos_base, col):
    """MuJoCo qpos = qpos_base with the 14 arm joints replaced by a trajectory column (layout order)."""
    from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS
    q = np.array(qpos_base, float).copy()
    full = layout.full_q(np.asarray(col, float).reshape(-1, 1), FIA.q_now_from_qpos(q, qadr))[:, 0]
    for j, v in zip(DEFAULT_RBY1_JOINTS, full):
        q[qadr[j]] = v
    return q


def geom_bar_per_link(qp, mg, barw):
    mj_set(qp); ft = np.zeros(6)
    return {ln: mm(min(mg.pair_distance(barw["gid"], g, 0.30, ft) for g, l in GEOM_LINK.items() if l == ln)) for ln in LEFT_LINKS}

# ---------------------------------------------------------------- hurdle
def bar_of(mg):
    out = None; parts = []
    for g in sorted(mg.collision_geom_ids):
        R3 = d.geom_xmat[g].reshape(3, 3)
        e = dict(gid=int(g), name=gname(g), type=int(m.geom_type[g]), centre=d.geom_xpos[g].copy(), axis=R3[:, 2].copy(),
                 half_len=float(m.geom_size[g][1]), radius=float(m.geom_size[g][0]))
        parts.append(e)
        if e["name"].endswith("collision_bar"): out = e
    return out, parts


def seg_dist(p, a, b):
    """distance point(s) -> segment ab, and the closest point."""
    p = np.atleast_2d(p); ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    c = a + t[:, None] * ab
    return np.linalg.norm(p - c, axis=1), c


# ---------------------------------------------------------------- fields
def field_layers(z):
    L = {}
    def lay(prefix): return layer_from_arrays(z[f"{prefix}_distance"], z[f"{prefix}_origin"], float(z[f"{prefix}_voxel_size"]))
    L["coarse_20mm"] = lay("esdf_coarse"); L["fine_5mm"] = lay("esdf")
    if "esdf_free_distance" in z.files: L["target_free_5mm"] = lay("esdf_free")
    return L


def eval_layer(layer, P):
    P = np.atleast_2d(P)
    v = np.full(len(P), np.nan)
    cov = CuroboEsdfField._covers(layer.grid, P)
    if cov.any(): v[cov] = layer.distance(P[cov])
    return v, cov


def bar_samples(bar_b, n_axis=49, n_dir=16):
    a = bar_b["a"]; b = bar_b["b"]; ax = (b - a) / np.linalg.norm(b - a)
    tmp = np.array([0, 0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(ax, tmp); u /= np.linalg.norm(u); w = np.cross(ax, u)
    s = np.linspace(0, 1, n_axis)
    axis_pts = a + s[:, None] * (b - a)
    ang = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
    dirs = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w
    return axis_pts, dirs, u, w, ax


def bar_field_report(z, fieldc, bar_b, focus_b):
    """field values on the true bar (base frame). focus_b = bar axis point nearest the contact link."""
    L = field_layers(z)
    axis_pts, dirs, u, w, ax = bar_samples(bar_b)
    r = bar_b["radius"]
    surf = (axis_pts[:, None, :] + r * dirs[None, :, :]).reshape(-1, 3)
    radii = np.arange(0, 0.0805, 0.0005)
    out = {}
    tiers = dict(L)
    for name, layer in tiers.items():
        va, ca = eval_layer(layer, axis_pts); vs, cs = eval_layer(layer, surf)
        # exit radius along each ray from the axis: last r (contiguous from the axis) with field <= 0
        exit_r = []
        for p in axis_pts:
            if not CuroboEsdfField._covers(layer.grid, p[None])[0]: continue
            for dvec in dirs:
                pts = p + radii[:, None] * dvec
                vv, cc = eval_layer(layer, pts)
                if not (vv[0] <= 0): exit_r.append(np.nan); continue
                k = np.argmax(~(vv <= 0)) if (~(vv <= 0)).any() else len(radii)
                exit_r.append(radii[k - 1])
        exit_r = np.array(exit_r, float)
        vf, cf = eval_layer(layer, focus_b[None])
        out[name] = {
            "voxel_mm": layer.grid.voxel_size * 1000.0,
            "bar_axis_points_covered": f"{int(ca.sum())}/{len(axis_pts)}",
            "field_at_axis_mm": stats_mm(va), "true_at_axis_mm": -r * 1000.0,
            "n_axis_points_inside_field_le0": int(np.sum(va[ca] <= 0)),
            "field_at_true_surface_mm": stats_mm(vs), "true_at_surface_mm": 0.0,
            "field_surface_exit_radius_mm": (None if not exit_r.size else {
                "n_rays": int(exit_r.size), "n_rays_axis_not_inside": int(np.isnan(exit_r).sum()),
                **({"median": mm(np.nanmedian(exit_r)), "min": mm(np.nanmin(exit_r)), "max": mm(np.nanmax(exit_r))}
                   if np.isfinite(exit_r).any() else {})}),
            "true_radius_mm": r * 1000.0,
            "field_at_focus_axis_point_mm": mm(vf[0]) if cf[0] else None,
        }
    # composite as the TO reads it (min of coarse/fine) and the target-free composite
    va = fieldc.distance(axis_pts); vs = fieldc.distance(surf)
    out["TO_plain_composite"] = {"field_at_axis_mm": stats_mm(va), "field_at_true_surface_mm": stats_mm(vs),
                                 "field_at_focus_axis_point_mm": mm(fieldc.distance(focus_b[None])[0])}
    if fieldc.has_target_free:
        va = fieldc.target_free_distance(axis_pts); vs = fieldc.target_free_distance(surf)
        out["TO_target_free_composite"] = {"field_at_axis_mm": stats_mm(va), "field_at_true_surface_mm": stats_mm(vs),
                                           "field_at_focus_axis_point_mm": mm(fieldc.target_free_distance(focus_b[None])[0])}
    return out


def true_bar_clear(C, rad, bar_b):
    """analytic sphere-to-bar clearance: |c - axis segment| - r_bar - r_sphere. C (...,3)."""
    sh = C.shape[:-1]
    dd, _ = seg_dist(C.reshape(-1, 3), bar_b["a"], bar_b["b"])
    return dd.reshape(sh) - bar_b["radius"] - rad


# ---------------------------------------------------------------- coverage of the true finger geoms by the spheres
def geom_vertices_world(g):
    did = int(m.geom_dataid[g])
    va, nv = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
    V = m.mesh_vert[va:va + nv].astype(float)
    return d.geom_xpos[g] + V @ d.geom_xmat[g].reshape(3, 3).T


def coverage_at(q, bar_w, mg, contact_geom=None):
    """at MuJoCo qpos q: true geoms of EE_BODY_L/ee_finger_l1/l2 vs the URDF query spheres (world)."""
    C, rad = spheres_at_qpos(q)
    Cw = to_world(C, q)
    mj_set(q)
    res = {}
    ft = np.zeros(6)
    for ln in LEFT_LINKS:
        gs = [g for g, l in GEOM_LINK.items() if l == ln]
        sel = NAMES == ln
        Vall = []; per_geom = {}
        for g in gs:
            V = geom_vertices_world(g)
            gap = np.min(np.linalg.norm(V[:, None, :] - Cw[None, :, :], axis=2) - rad[None, :], axis=1)  # vs all 122 spheres
            gap_own = np.min(np.linalg.norm(V[:, None, :] - Cw[None, sel, :], axis=2) - rad[None, sel], axis=1)
            dist_bar = mg.pair_distance(bar_w["gid"], g, 0.30, ft)
            wit = ft[3:].copy() if np.linalg.norm(ft) > 0 else None
            vb, _ = seg_dist(V, bar_w["a"], bar_w["b"]); vb = vb - bar_w["radius"]
            near = vb <= 0.010
            per_geom[gname(g)] = {
                "n_vertices": int(len(V)), "frac_vertices_inside_a_sphere": float(np.mean(gap <= 0)),
                "vertex_gap_to_sphere_surface_mm": stats_mm(gap),
                "vertex_gap_to_own_link_spheres_mm": stats_mm(gap_own),
                "geom_to_bar_mm": mm(dist_bar),
                "n_vertices_within_10mm_of_bar": int(near.sum()),
                "gap_of_vertices_within_10mm_of_bar_mm": stats_mm(gap[near]) if near.any() else None,
                "witness_on_geom_gap_to_spheres_mm": (mm(np.min(np.linalg.norm(Cw - ft[3:], axis=1) - rad))
                                                      if dist_bar < 0.30 else None),
            }
            Vall.append(V)
        res[ln] = {"mujoco_body": MJ_BODY[ln], "n_spheres": int(sel.sum()),
                   "sphere_radius_mm": stats_mm(rad[sel]), "geoms": per_geom,
                   "sphere_to_bar_true_min_mm": mm(np.min(true_bar_clear(Cw[sel], rad[sel], bar_w)))}
    return res, Cw, rad


# ---------------------------------------------------------------- one case
def run_case(tag, spec):
    a = A[spec["key"]]; h = a["hurdle"]
    rd = pathlib.Path(a["run_dir"])
    rows = [json.loads(l) for l in open(rd / "frames.jsonl")]
    plan = {r["seq"]: r for r in rows if r["kind"] == "planning"}
    ctrl = {r["t_step"]: r for r in rows if r["kind"] == "control"}
    mg = HM.manager(h["config"], h["profile"])
    mj_set(np.array(plan[1]["qpos"]))
    barw, parts = bar_of(mg)
    hl = barw["half_len"]
    barw["a"] = barw["centre"] - hl * barw["axis"]; barw["b"] = barw["centre"] + hl * barw["axis"]
    # event
    if spec["kind"] == "contact":
        ev_t = max(ctrl)  # last control row = obstacle stop row (client stops after it)
    else:
        ph = h["phases"]["carry_lift_to_place_or_end"]["A"]; ev_t = int(ph["t"])
    seqs = sorted(plan)
    ev_seq = max(s for s in seqs if plan[s]["t_step"] <= ev_t)
    win = [s for s in seqs if ev_seq - N_WINDOW + 1 <= s <= ev_seq]
    files = {}
    for k in range(spec["chunks"][0], spec["chunks"][1] + 1):
        f = spec["srv"] / "server_constraints/run_0000" / f"chunk_{k:05d}.npz"
        with np.load(f) as z: files[int(z["t_step"])] = f
    q_ev = np.array(ctrl[ev_t]["qpos"])
    # contact geom at the event row (closest left-hand geom to the bar)
    mj_set(q_ev); ft = np.zeros(6)
    dists = {g: mg.pair_distance(barw["gid"], g, 0.30, ft) for g in GEOM_LINK}
    g_ev = min(dists, key=dists.get)
    cpairs = (a.get("hurdle") or {}).get("client_obstacle_safety", {}).get("contact_pairs") or []
    if spec["kind"] == "contact" and cpairs:  # the geom MuJoCo reported in contact with the bar (client record)
        gn = [x for x in cpairs[0] if x.startswith("geom#")][0]
        g_ev = int(gn[5:])
    ev_link = GEOM_LINK[g_ev]
    mg.pair_distance(barw["gid"], g_ev, 0.30, ft); witness_w = ft[3:].copy(); bar_pt_w = ft[:3].copy()
    out = {"run_dir": str(rd), "hurdle": h["hurdle"], "config": h["config"], "profile": h["profile"],
           "bar_geom": {"name": barw["name"], "radius_mm": barw["radius"] * 1000, "half_length_mm": hl * 1000,
                        "centre_world": barw["centre"], "axis_world": barw["axis"]},
           "event": {"kind": spec["kind"], "t": ev_t, "seq": ev_seq,
                     "definition": ("last control row (= client obstacle stop row; MuJoCo contact seen in the sim steps after it)"
                                    if spec["kind"] == "contact" else
                                    "control row of the carry-phase minimum gripper(EE_BODY_L/fingers)-to-hurdle distance (analysis_all)"),
                     "closest_left_geom": gname(g_ev), "closest_link": ev_link,
                     "event_geom_source": ("client MuJoCo contact pair" if spec["kind"] == "contact" and cpairs else "closest left-hand geom at the event row"),
                     "event_geom_to_bar_mm_at_row": mm(dists[g_ev]),
                     "per_link_min_geom_to_bar_mm_at_row": {ln: mm(min(v for g, v in dists.items() if GEOM_LINK[g] == ln)) for ln in LEFT_LINKS},
                     "witness_on_geom_world": witness_w, "witness_on_bar_world": bar_pt_w,
                     "client_contact_pairs": (A[spec["key"]].get("hurdle") or {}).get("client_obstacle_safety", {}).get("contact_pairs")},
           "window_seqs": win, "chunks": []}
    cov, Cw_ev, rad_ev = coverage_at(q_ev, barw, mg)
    out["event"]["sphere_coverage_at_event_row"] = cov
    out["event"]["spheres_world_at_event_row"] = {"centres": Cw_ev, "radii": rad_ev, "links": NAMES.tolist()}
    # 2D section data for the figure (plane through the bar point nearest the contact, normal = bar axis)
    sec = {}
    for s in win:
        p = plan[s]; q = np.array(p["qpos"]); fpath = files.get(s)
        z = np.load(fpath)
        summ = json.loads(str(z["summary_json"]))
        ref_c = np.asarray(z["reference_chunk"], float); fin_c = np.asarray(z["refined_chunk"], float)
        pair_ok = float(np.abs(np.asarray(p["actions"]) - fin_c).max())
        pair_ref = float(np.abs(np.asarray(p["actions_reference"]) - ref_c).max())
        q_now = FIA.q_now_from_qpos(q, qadr)
        # bar in base frame (base pose at this planning row)
        bar_b = {"a": to_base(barw["a"], q), "b": to_base(barw["b"], q), "radius": barw["radius"]}
        focus_b = to_base(seg_dist(witness_w, barw["a"], barw["b"])[1][0], q)
        fieldc = FIA.build_field(z, OUTSIDE)
        has_target = bool(summ.get("has_target")) and "target_points" in z.files
        attached = int((summ.get("esdf_stats") or {}).get("n_attached_points", 0) or 0) > 0
        margin, mask, authorized, _, _ = FIA.contact_policy(settings, robot.sphere_link_names, has_target=has_target,
                                                            has_free=bool(fieldc.target_free_layers), attached=attached)
        scene = FIA.make_scene(fieldc, RADII, settings.esdf_margin,
                               target_points=z["target_points"] if has_target else None, margin=margin, mask=mask)
        # finger parameter path as the server built it
        fj = summ.get("finger_joints") or {}
        measured = fj.get("opening_norm") or {}
        cmd = {"left": ref_c[:, 7], "right": ref_c[:, 15]}
        P, _ = fmap.path(measured, cmd, H, default=DEFAULT_P)
        meter.lin.set_joint_parameter_path(P)
        robot.set_joint_parameters(dict(zip(PARAM_NAMES, P[:, 0])))
        traj_ref = layout.chunk_to_trajectory(ref_c)[:, :H]; traj_fin = layout.chunk_to_trajectory(fin_c)[:, :H]
        fid = FIA.fidelity(meter, z, summ, traj_fin, q_now, fieldc, scene, RADII, settings.esdf_margin)
        det_fin = FIA.row_details(meter, traj_fin, q_now, scene, fieldc)
        det_ref = FIA.row_details(meter, traj_ref, q_now, scene, fieldc)
        rec_cl = np.asarray(z["clearance"], float)
        to = summ.get("to") or {}; met = to.get("metrics") or {}
        ver = summ.get("verdict") or {}
        Cfin = spheres_path(traj_fin, q_now, P); Cref = spheres_path(traj_ref, q_now, P)
        tb_fin = true_bar_clear(Cfin, RADII[None, :], bar_b); tb_ref = true_bar_clear(Cref, RADII[None, :], bar_b)
        c_now = robot.sphere_centers_numeric(q_now, P[:, 0])[0]
        sel_ev = NAMES == ev_link
        per_link = {}
        for ln in LEFT_LINKS:
            sl = NAMES == ln
            per_link[ln] = {
                "TO_view_clearance_min_mm_refined": mm(det_fin["clearance"][:, sl].min()),
                "TO_view_clearance_min_per_step_mm_refined": (det_fin["clearance"][:, sl].min(axis=1) * 1000).tolist(),
                "TO_view_clearance_min_mm_reference": mm(det_ref["clearance"][:, sl].min()),
                "recorded_plain_clearance_min_mm_refined": mm(rec_cl[:, sl].min()),
                "reads_target_free_layer": bool(det_fin["mask"][sl].any()),
                "true_sphere_to_bar_min_mm_refined": mm(tb_fin[:, sl].min()),
                "true_sphere_to_bar_min_per_step_mm_refined": (tb_fin[:, sl].min(axis=1) * 1000).tolist(),
                "true_sphere_to_bar_min_mm_reference": mm(tb_ref[:, sl].min()),
                "true_sphere_to_bar_min_per_step_mm_reference": (tb_ref[:, sl].min(axis=1) * 1000).tolist(),
                "true_sphere_to_bar_mm_at_q_now": mm(true_bar_clear(c_now[sl], RADII[sl], bar_b).min()),
            }
        # policy pull on the event link: centroid of its spheres
        def cen(Cs): return Cs[:, sel_ev].mean(axis=1)
        p_ref, p_fin, p_now = cen(Cref), cen(Cfin), c_now[sel_ev].mean(axis=0)
        dn, cp = seg_dist(p_fin, bar_b["a"], bar_b["b"]); nrm = (p_fin - cp) / dn[:, None]
        dn0, cp0 = seg_dist(p_now[None], bar_b["a"], bar_b["b"]); n0 = ((p_now - cp0[0]) / dn0[0])
        v_ref = p_ref[-1] - p_now; v_fin = p_fin[-1] - p_now
        cosang = lambda v, n: float(v @ (-n) / (np.linalg.norm(v) + 1e-12))
        pull = {
            "link": ev_link,
            "joint_abs_ref_minus_refined_max_deg": float(np.degrees(np.abs(traj_ref - traj_fin)).max()),
            "joint_abs_ref_minus_refined_max_deg_per_step": np.degrees(np.abs(traj_ref - traj_fin)).max(axis=0).tolist(),
            "link_centroid_ref_minus_refined_mm_per_step": (np.linalg.norm(p_ref - p_fin, axis=1) * 1000).tolist(),
            "link_centroid_ref_minus_refined_toward_bar_mm_per_step": (np.einsum("ij,ij->i", p_ref - p_fin, -nrm) * 1000).tolist(),
            "policy_displacement_over_window_mm": float(np.linalg.norm(v_ref) * 1000),
            "policy_displacement_toward_bar_mm": float(v_ref @ (-n0) * 1000),
            "policy_direction_cos_to_bar_inward_normal": cosang(v_ref, n0),
            "policy_direction_angle_to_bar_inward_normal_deg": float(np.degrees(np.arccos(np.clip(cosang(v_ref, n0), -1, 1)))),
            "refined_displacement_over_window_mm": float(np.linalg.norm(v_fin) * 1000),
            "refined_displacement_toward_bar_mm": float(v_fin @ (-n0) * 1000),
            "link_centroid_to_bar_axis_mm_at_q_now": float(dn0[0] * 1000),
            "definition": "link centroid = mean of that link's query spheres (URDF model, server finger path); "
                          "bar inward normal at q_now = unit vector from the centroid to the closest bar-axis point; "
                          "window = TO plan window (8 steps); toward-bar = projection on that normal (+ = closer to the bar)",
        }
        # what was executed in this chunk (control rows t..t+7)
        trows = [t for t in range(p["t_step"], p["t_step"] + 8) if t in ctrl]
        exe = {"rows": trows, "executed_flags": [bool(ctrl[t]["executed"]) for t in trows]}
        arm_idx = layout.q_indices
        if trows:
            applied = np.array([ctrl[t]["applied_ctrl"]["arm"] for t in trows])  # 14 arm targets
            def arm14(chunk):  # chunk columns 0..6, 8..14 = arm joints (16D layout)
                c = np.asarray(chunk, float); return np.concatenate([c[:, 0:7], c[:, 8:15]], axis=1)
            fin14 = arm14(fin_c)[:len(trows)]; ref14 = arm14(ref_c)[:len(trows)]
            exe["applied_vs_refined_max_abs_rad"] = float(np.abs(applied - fin14).max())
            exe["applied_vs_reference_max_abs_rad"] = float(np.abs(applied - ref14).max())
            hold = [ctrl[t].get("hold") for t in trows]
            if any(hold):
                qh = np.array(next(hh for hh in hold if hh)["q_hold"])
                exe["applied_vs_q_hold_max_abs_rad"] = float(np.abs(applied - qh[None]).max())
                exe["q_hold_source"] = next(hh for hh in hold if hh)["source"]
            exe["source_by_value"] = min(
                [(k, v) for k, v in (("refined", exe["applied_vs_refined_max_abs_rad"]),
                                     ("reference", exe["applied_vs_reference_max_abs_rad"]),
                                     ("q_hold", exe.get("applied_vs_q_hold_max_abs_rad", np.inf)))], key=lambda kv: kv[1])[0]
            # measured pose vs TO plan, per executed row (row k's qpos is the pose when row k was commanded)
            Qm = np.array([ctrl[t]["qpos"] for t in trows])
            qm = np.array([FIA.q_now_from_qpos(x, qadr) for x in Qm])[:, arm_idx]  # (n, 12)
            plan_q = traj_fin.T[:len(trows)]
            # measured row k vs plan step k-1 (plan step 0 is the target for row 0; the pose at row k reflects targets up to k-1)
            dev_now = np.degrees(np.abs(qm - np.vstack([q_now[arm_idx][None], plan_q[:-1]])))
            exe["measured_vs_TO_plan_prev_step_max_deg"] = float(dev_now.max())
            exe["measured_vs_TO_plan_prev_step_max_deg_per_row"] = dev_now.max(axis=1).tolist()
            Cm = np.stack([spheres_at_qpos(x)[0] for x in Qm])
            Cm_b = np.stack([to_base(to_world(Cm[i], Qm[i]), q) for i in range(len(Qm))])  # all in this chunk's base frame
            Cplan_prev = np.concatenate([c_now[None], Cfin[:-1]], axis=0)[:len(trows)]
            exe["measured_vs_TO_plan_link_centroid_mm_per_row"] = (np.linalg.norm(
                Cm_b[:, sel_ev].mean(axis=1) - Cplan_prev[:, sel_ev].mean(axis=1), axis=1) * 1000).tolist()
            # the measured poses through this chunk's field as the TO reads it (same tier rule) and vs the true bar
            flat = Cm_b.reshape(-1, 3)
            dd = fieldc.distance(flat).reshape(Cm_b.shape[:2])
            dfree = (fieldc.target_free_distance(flat).reshape(Cm_b.shape[:2]) if fieldc.has_target_free else dd)
            mk = det_fin["mask"]
            deff = np.where(mk[None, :], dfree, dd) - RADII[None, :] - settings.esdf_margin
            exe["measured_rows_TO_view_clearance_min_mm_per_row"] = {ln: (deff[:, NAMES == ln].min(axis=1) * 1000).tolist() for ln in LEFT_LINKS}
            exe["measured_rows_true_sphere_to_bar_min_mm_per_row"] = {ln: (true_bar_clear(Cm_b[:, NAMES == ln], RADII[NAMES == ln], bar_b).min(axis=1) * 1000).tolist() for ln in LEFT_LINKS}
            # true geom distance per executed row
            gd = []
            for x in Qm:
                mj_set(x); gd.append({ln: mm(min(mg.pair_distance(barw["gid"], g, 0.30, ft) for g, l in GEOM_LINK.items() if l == ln)) for ln in LEFT_LINKS})
            exe["measured_rows_true_geom_to_bar_mm"] = gd
            dev_same = np.degrees(np.abs(qm - plan_q))
            exe["measured_vs_TO_plan_same_step_max_deg_per_row"] = dev_same.max(axis=1).tolist()
            exe["measured_vs_TO_plan_same_step_link_centroid_mm_per_row"] = (np.linalg.norm(
                Cm_b[:, sel_ev].mean(axis=1) - Cfin[:len(trows), sel_ev].mean(axis=1), axis=1) * 1000).tolist()
            if any(hold):
                hh = next(x for x in hold if x)
                qh = np.array(hh["q_hold"])
                col = np.concatenate([qh[:7], qh[7:]])  # applied arm order = layout order (left 0-6, right 0-6)
                exe["q_hold_vs_measured_first_row_max_deg"] = float(np.degrees(np.abs(col - qm[0])).max())
                exe["hold_record_max_abs_dev_rad_last_row"] = [x for x in hold if x][-1].get("max_abs_dev_rad")
                exe["q_hold_pose_true_geom_to_bar_mm"] = geom_bar_per_link(qpos_with_traj(Qm[0], col), mg, barw)
                exe["measured_link_centroid_drift_during_rows_mm"] = (np.linalg.norm(
                    Cm_b[:, sel_ev].mean(axis=1) - Cm_b[0, sel_ev].mean(axis=0), axis=1) * 1000).tolist()
        # plan step 0 jump from q_now, and the refined/reference plans through the TRUE geometry (MuJoCo, fingers as at the planning row)
        exe["plan_step0_jump_from_q_now_max_deg"] = float(np.degrees(np.abs(traj_fin[:, 0] - q_now[arm_idx])).max())
        exe["plan_step0_jump_link_centroid_mm"] = float(np.linalg.norm(Cfin[0, sel_ev].mean(0) - c_now[sel_ev].mean(0)) * 1000)
        exe["refined_plan_true_geom_to_bar_mm_per_step"] = [geom_bar_per_link(qpos_with_traj(q, traj_fin[:, k]), mg, barw) for k in range(H)]
        exe["reference_plan_true_geom_to_bar_mm_per_step"] = [geom_bar_per_link(qpos_with_traj(q, traj_ref[:, k]), mg, barw) for k in range(H)]
        mvp = FIA._parse_pair(met.get("max_violation_pair"))
        rec = {
            "seq": s, "t_step": p["t_step"], "npz": str(fpath), "pairing_refined_vs_actions_max": pair_ok,
            "pairing_reference_vs_actions_reference_max": pair_ref,
            "gate": p.get("gate"), "ipc": p.get("ipc"), "hold_chunks": p.get("hold_chunks"),
            "verdict_reasons": [{k: r.get(k) for k in ("kind", "action", "detail")} for r in (p.get("verdict_reasons") or [])],
            "trajopt_status": (p.get("verdict") or {}).get("trajopt_status"),
            "max_violation_m": (p.get("verdict") or {}).get("max_violation_m"),
            "TO_worst_row": mvp, "TO_clearance_m": met.get("clearance_m"),
            "TO_returned": met.get("returned"), "TO_sqp_iterations": to.get("sqp_iterations"),
            "TO_initial_violation_m": met.get("initial_violation_m"),
            "ag3s_status": summ.get("status"), "grounding_status": summ.get("grounding_status"), "has_target": summ.get("has_target"),
            "exclusion_active": (summ.get("exclusion") or {}).get("active"), "attached_points": attached,
            "finger_opening_norm_left": measured.get("left"),
            "fine_window": {"origin_base": z["esdf_origin"], "extent_m": float(z["esdf_voxel_size"]) * int(z["esdf_shape"][0])},
            "target_free_window": ({"origin_base": z["esdf_free_origin"], "extent_m": float(z["esdf_free_voxel_size"]) * int(z["esdf_free_shape"][0])}
                                   if "esdf_free_origin" in z.files else None),
            "fidelity": fid,
            "bar_in_field": bar_field_report(z, fieldc, bar_b, focus_b),
            "per_link": per_link, "policy_pull": pull, "executed": exe,
        }
        out["chunks"].append(rec)
        # section arrays (bar cross-section at the focus point; plane normal = bar axis)
        axis_pts, dirs, u, w, axv = bar_samples(bar_b)
        g = np.arange(-0.10, 0.1001, 0.0025)
        U, W = np.meshgrid(g, g, indexing="xy")
        Pp = focus_b + U[..., None] * u + W[..., None] * w
        flat = Pp.reshape(-1, 3)
        Ls = field_layers(z)
        grids = {}
        for nm, layer in Ls.items():
            v, cov = eval_layer(layer, flat); grids[nm] = v.reshape(U.shape)
        grids["TO_plain"] = fieldc.distance(flat).reshape(U.shape)
        if fieldc.has_target_free: grids["TO_target_free"] = fieldc.target_free_distance(flat).reshape(U.shape)
        # spheres at q_now (left hand) in section coordinates
        sl = np.isin(NAMES, LEFT_LINKS)
        rel = c_now[sl] - focus_b
        sph = np.stack([rel @ u, rel @ w, rel @ axv, RADII[sl]], axis=1)
        relp = Cfin[-1][sl] - focus_b; relr = Cref[-1][sl] - focus_b
        sec[s] = {"grid_m": g, "fields": grids, "spheres_q_now_uvaxr": sph, "sphere_links": NAMES[sl].tolist(),
                  "spheres_refined_last_uvaxr": np.stack([relp @ u, relp @ w, relp @ axv, RADII[sl]], 1),
                  "spheres_reference_last_uvaxr": np.stack([relr @ u, relr @ w, relr @ axv, RADII[sl]], 1)}
        z.close()
        print(f"[{tag}] seq {s} t {p['t_step']} gate {p['gate']['action']} fid {fid.get('record_clearance', {}).get('max_abs_diff_mm')} "
              f"to_view {fid.get('to_view', {}).get('abs_diff_mm')}", flush=True)
    # event section: true geom vertices (left hand) at the event row near the focus plane, in the event chunk's base frame
    q_last = np.array(plan[ev_seq]["qpos"])
    bar_b = {"a": to_base(barw["a"], q_last), "b": to_base(barw["b"], q_last)}
    axis_pts, dirs, u, w, axv = bar_samples({**bar_b, "radius": barw["radius"]})
    focus_b = to_base(seg_dist(witness_w, barw["a"], barw["b"])[1][0], q_last)
    mj_set(q_ev)
    verts = []
    for g, ln in GEOM_LINK.items():
        V = to_base(geom_vertices_world(g), q_last) - focus_b
        verts.append(np.stack([V @ u, V @ w, V @ axv, np.full(len(V), LEFT_LINKS.index(ln))], 1))
    verts = np.concatenate(verts)
    Ce = to_base(Cw_ev, q_last) - focus_b
    sl = np.isin(NAMES, LEFT_LINKS)
    sec["event"] = {"geom_vertices_uvax_link": verts, "spheres_event_uvaxr": np.stack([Ce[sl] @ u, Ce[sl] @ w, Ce[sl] @ axv, rad_ev[sl]], 1),
                    "sphere_links": NAMES[sl].tolist(), "witness_uvax": [(to_base(witness_w, q_last) - focus_b) @ vv for vv in (u, w, axv)],
                    "bar_radius_m": barw["radius"], "ev_seq": ev_seq}
    # top-view locator (world): bar, feet, contact link path
    out["locator"] = {"bar_a_world": barw["a"], "bar_b_world": barw["b"], "witness_world": witness_w,
                      "parts": [{k: v for k, v in pp.items()} for pp in parts]}
    np.save(OUT / f"section_{tag}.npy", sec, allow_pickle=True)
    # clearance series for the graph: per control row true geom distance (analysis_all series A)
    out["series_true_gripper_to_hurdle"] = {"t": h["series"]["t"], "A_m": h["series"]["A"]}
    return out


def build_mapping():
    """MuJoCo left-hand body -> URDF sphere link, by geometry: median gap of the body's collision-mesh vertices to each
    URDF link's spheres at one recorded pose (C1807 planning seq 1). Fills MJ_BODY (urdf link -> MuJoCo body) and GEOM_LINK."""
    rows = [json.loads(l) for l in open(R / "C/E3b_ep1807_s18071/frames.jsonl")]
    q = np.array([r for r in rows if r["kind"] == "planning"][0]["qpos"])
    C, rad = spheres_at_qpos(q); Cw = to_world(C, q); mj_set(q)
    mat = {}
    for bn in ("EE_BODY_L", "ee_finger_l1", "ee_finger_l2"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
        gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid and (m.geom_contype[g] or m.geom_conaffinity[g])]
        V = np.concatenate([geom_vertices_world(g) for g in gs])
        mat[bn] = {ln: mm(np.median(np.min(np.linalg.norm(V[:, None] - Cw[None, NAMES == ln], axis=2) - rad[None, NAMES == ln], axis=1)))
                   for ln in LEFT_LINKS}
        ln = min(mat[bn], key=mat[bn].get)
        MJ_BODY[ln] = bn
        for g in gs: GEOM_LINK[g] = ln
    assert sorted(MJ_BODY) == sorted(LEFT_LINKS), MJ_BODY
    MAPPING.update({"median_vertex_gap_mm_body_x_urdf_link": mat, "urdf_link_to_mujoco_body": dict(MJ_BODY),
                    "pose": "C1807 planning seq 1 qpos"})


def main():
    build_mapping()
    res = {"generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "code_snapshot": "${AG3S_PROJECT_ROOT}-t40", "settings": settings.to_json(), "n_spheres": int(robot.n_spheres),
           "sphere_links": sorted(set(NAMES.tolist())), "body_link_mapping": MAPPING, "cases": {}}
    only = sys.argv[1:] or list(CASES)
    for tag in only:
        res["cases"][tag] = run_case(tag, CASES[tag])
        json.dump(jl(res["cases"][tag]), open(OUT / f"case_{tag}.json", "w"), indent=1)
    json.dump(jl({k: v for k, v in res.items() if k != "cases"}), open(OUT / "settings.json", "w"), indent=1)


if __name__ == "__main__":
    main()
