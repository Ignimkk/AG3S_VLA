"""T43 S (5) — is the held fruit part of the robot in the TO collision check? Records only (CPU, no server, no GPU).

Runs: every hurdle run (stage C / D) of T40 G, T40 J, T41 a, T41 c E3b (index raw/chunk_index_t40t41.json; client frames +
server npz --record-constraints esdf=full). Code for reconstruction: snapshot ${AG3S_PROJECT_ROOT}-t41c (PYTHONPATH) — pipeline.py,
held_object.py, attached.py, curobo_field.py, fixed_input_ablation.py are byte-identical in work-t40 / t40h / t41a / t41c.

Per control row (recorded qpos, MuJoCo obstacle scene of the run, hurdle placed by the snapshot client manager):
  apple_bar_mm     : manager.pair_distance(apple_geom, hurdle bar capsule)  (T40 H certified distance; <= 0 = contact)
  apple_hurdle_mm  : min over all hurdle collision geoms
Contact row t_c = first control row with apple_bar_mm <= 0 (after the target left the table by > 20 mm).
Per HELD planning chunk (summary grasp.state == held and grasp.attached present) up to t_c (all, or all if no contact):
  held spheres     : summary grasp.attached.held.spheres_parent_frame (centre in parent-link frame, radius) = the TO query
                     spheres (linearize: attached_points / attached_radii from held_spheres_of); parent = grasp.attached.parent_link
  filter spheres   : filter_spheres_parent_frame, radius + HELD_FILTER_SLIP_M (0.020 m) = the self-filter copy (pixels inside are
                     masked from the depth images)
  held_free        : the TSDF voxels set free each held frame = the held query spheres at the frame's robot state (minus below
                     the support plane + 1 fine voxel, minus within 2 fine voxels of the destination points) — pipeline.py
                     _held_free_points
  plan             : refined_chunk rows 0..7 (= planning 'actions', checked) -> URDF FK of the parent link per step
  TO view          : field = recorded composite (coarse 20 mm + fine 5 mm, CuroboEsdfField.distance); held rows have no
                     target-free layer (to_adapter: target_field_exclude only when nothing is attached); clearance =
                     d(centre) - r - margin, margin = esdf_margin (0; held rows never get obstacle_margin: class 'held').
                     Validated against the recorded TO worst row when that row is 'attached:*' (point and clearance).
  true             : apple mesh (MuJoCo) vs bar: at the planning row qpos, along the plan (apple moved rigidly with EE_BODY_L,
                     relative pose of the planning row), at the chunk's executed control rows (measured qpos)
  analytic         : held sphere vs bar capsule = |c - axis segment| - r_bar - r
  coverage         : apple mesh vertices vs held spheres at the planning qpos (frac inside, vertex gap = min_i |v - c_i| - r_i;
                     > 0 = vertex outside every sphere), and for the apple vertices within 15 mm of the bar
  bar in field     : 49 points on the true bar axis (true value -12 mm): field value (fine, coarse, composite); whether the axis
                     point is inside the held query sphere (= held_free volume) / inside the filter sphere at the planning qpos
usage: measure_held.py [N_WORKERS] [run substrings ...]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, math, pathlib, ast, datetime
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, mujoco
from multiprocessing import Pool

SNAP = pathlib.Path("${AG3S_PROJECT_ROOT}-t41c")
sys.path.insert(0, str(SNAP)); sys.path.insert(0, str(SNAP / "integration/rby1/manipulation/src"))
from AG3S.trajopt.experiments import fixed_input_ablation as FIA  # noqa: E402
from AG3S.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays  # noqa: E402
from AG3S.ag3s.constraints.attached import _link_pose_numeric  # noqa: E402
from AG3S.ag3s.robot_models.held_object import HELD_FILTER_SLIP_M  # noqa: E402
from AG3S.trajopt.types import ChunkLayout  # noqa: E402
from rby1_manipulation.simulation import pick_place_obstacles as PPO  # noqa: E402
assert FIA.__file__.startswith(str(SNAP)) and PPO.__file__.startswith(str(SNAP)), (FIA.__file__, PPO.__file__)

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/S/raw'))
IDX = json.loads((OUT / "chunk_index_t40t41.json").read_text())
XML = str(SNAP / "robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml")
APPLE_ADR = 38
N_WINDOW = 5
_G = {}


def jl(o):
    if isinstance(o, dict): return {str(k): jl(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jl(v) for v in o]
    if isinstance(o, np.ndarray): return jl(o.tolist())
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


def lit(x):
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except Exception: return x
    return x


def mm(x): return None if x is None else float(x) * 1000.0


def stats_mm(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not v.size: return None
    return {"n": int(v.size), "min": mm(v.min()), "median": mm(np.median(v)), "max": mm(v.max())}


def setup(server_log, manifest):
    key = str(server_log)
    if key in _G: return _G[key]
    class Args: links = None; sphere_spacing = None; max_spheres_per_capsule = None; capsule_radius_scale = None
    Args.max_sphere_radius = None; Args.esdf_margin = None; Args.target_field_policy = None; Args.phase = None
    Args.manipulators = None; Args.model_xml = None; Args.hand = None; Args.allow_uncertified = False; Args.rows_per_step = None
    log = FIA.parse_server_log(str(server_log))
    man = json.loads(json.dumps(manifest)); man["scene"]["model_xml"] = str(SNAP / "robots/rby1/description/models/rby1a/mujoco/model_transport.xml")
    settings = FIA.resolve_settings(Args, log, man)
    robot, qadr, _ = FIA.build_robot(settings)
    layout = ChunkLayout.rby1(robot.joint_names)
    _, _, _, max_distance, unknown_policy = FIA.contact_policy(settings, robot.sphere_link_names, has_target=False, has_free=False, attached=False)
    outside = max_distance if unknown_policy == "free" else -max_distance
    _G[key] = dict(settings=settings, robot=robot, qadr=qadr, layout=layout, outside=outside)
    return _G[key]


def mj():
    if "m" not in _G:
        m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
        _G.update(m=m, d=d, J={mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): int(m.jnt_qposadr[i]) for i in range(m.njnt)},
                  B=lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n),
                  apple_g=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom"))
    return _G["m"], _G["d"]


def set_q(q):
    m, d = mj(); d.qpos[:] = q; mujoco.mj_kinematics(m, d)


def base_T(q):
    m, d = mj(); set_q(q); b = _G["B"]("base"); return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


def to_base(pw, q): p0, R = base_T(q); return (np.asarray(pw) - p0) @ R


def to_world(pb, q): p0, R = base_T(q); return np.asarray(pb) @ R.T + p0


def geom_vertices_world(g):
    m, d = mj(); did = int(m.geom_dataid[g]); va, nv = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
    return d.geom_xpos[g] + m.mesh_vert[va:va + nv].astype(float) @ d.geom_xmat[g].reshape(3, 3).T


def seg_dist(p, a, b):
    p = np.atleast_2d(p); ab = b - a; t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1); c = a + t[:, None] * ab
    return np.linalg.norm(p - c, axis=1), c


def q_arms_from_chunk_row(q, row):
    q = np.array(q, float).copy(); J = _G["J"]
    for i in range(7):
        q[J[f"left_arm_{i}"]] = row[i]; q[J[f"right_arm_{i}"]] = row[8 + i]
    return q


def mat2quat(R):
    qq = np.zeros(4); mujoco.mju_mat2Quat(qq, np.asarray(R, float).reshape(-1)); return qq


def apple_moved_with_hand(q_ref, q_new):
    """q_new with the apple free joint set so the apple keeps its pose relative to EE_BODY_L at q_ref."""
    m, d = mj(); ee = _G["B"]("EE_BODY_L"); ab = _G["B"]("apple")
    set_q(q_ref); Re, pe = d.xmat[ee].reshape(3, 3).copy(), d.xpos[ee].copy(); Ra, pa = d.xmat[ab].reshape(3, 3).copy(), d.xpos[ab].copy()
    Rrel = Re.T @ Ra; prel = Re.T @ (pa - pe)
    set_q(q_new); Re2, pe2 = d.xmat[ee].reshape(3, 3).copy(), d.xpos[ee].copy()
    q = np.array(q_new, float).copy(); q[APPLE_ADR:APPLE_ADR + 3] = pe2 + Re2 @ prel; q[APPLE_ADR + 3:APPLE_ADR + 7] = mat2quat(Re2 @ Rrel)
    return q


def held_from_summary(s):
    g = s.get("grasp") or {}; att = g.get("attached") or {}; held = att.get("held") or {}
    sp = held.get("spheres_parent_frame"); fs = held.get("filter_spheres_parent_frame")
    if not sp: return None
    return {"parent_link": att.get("parent_link"), "grasp_link": held.get("grasp_link"),
            "local": np.array([x["centre_m"] for x in sp], float), "radii": np.array([x["radius_mm"] for x in sp], float) / 1000.0,
            "filter_local": None if not fs else np.array([x["centre_m"] for x in fs], float),
            "filter_radii": None if not fs else np.array([x["radius_mm"] for x in fs], float) / 1000.0 + HELD_FILTER_SLIP_M,
            "mode": held.get("mode"), "fit_radius_mm": (held.get("fit") or {}).get("radius_mm"),
            "fit_centre_base_at_capture": (held.get("fit") or {}).get("centre_m"),
            "lift_mm": [x.get("lift_mm") for x in ((held.get("support") or {}).get("spheres") or [])],
            "n_points": held.get("n_points"), "n_outside_primary": held.get("n_outside_primary"),
            "hand_shift_since_capture_mm": held.get("hand_shift_since_capture_mm"), "attached_at": att.get("attached_at")}


def spheres_base(G, held, q_full20):
    T = _link_pose_numeric(G["robot"], q_full20, held["parent_link"])
    c = held["local"] @ T[:3, :3].T + T[:3, 3]
    fc = None if held["filter_local"] is None else held["filter_local"] @ T[:3, :3].T + T[:3, 3]
    return c, fc


def run_one(key):
    ix = IDX[key]; rd = pathlib.Path(key)
    man = json.loads((rd / "manifest.json").read_text()); argv = man["extra"]["argv"]
    cfg_path = argv[argv.index("--obstacle-config") + 1]; profile = argv[argv.index("--obstacle-profile") + 1]
    srv = pathlib.Path(ix["server"]); G = setup(srv / "serve_safe.log", man)
    m, d = mj()
    mg = PPO.PickPlaceObstacleManager(m, d, PPO.load_pick_place_obstacle_config(cfg_path)); mg.activate(profile)
    rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    ctrl = {r["t_step"]: r for r in rows if r["kind"] == "control"}; plan = {r["seq"]: r for r in rows if r["kind"] == "planning"}
    q0 = np.asarray(plan[min(plan)]["qpos"], float); set_q(q0)
    bar = None; parts = []
    for g in sorted(mg.collision_geom_ids):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom#{g}"; parts.append(g)
        if nm.endswith("collision_bar"):
            R3 = d.geom_xmat[g].reshape(3, 3); hl = float(m.geom_size[g][1])
            bar = {"gid": int(g), "name": nm, "radius": float(m.geom_size[g][0]), "half_len": hl,
                   "a": d.geom_xpos[g] - hl * R3[:, 2], "b": d.geom_xpos[g] + hl * R3[:, 2], "centre": d.geom_xpos[g].copy()}
    ag = _G["apple_g"]; ft = np.zeros(6)
    z0 = float(q0[APPLE_ADR + 2])
    ts = sorted(ctrl); ab = []; ah = []; az = []
    for t in ts:
        set_q(np.asarray(ctrl[t]["qpos"], float))
        ab.append(mg.pair_distance(ag, bar["gid"], 0.30, ft)); ah.append(min(mg.pair_distance(ag, g, 0.30, ft) for g in parts))
        az.append(float(ctrl[t]["qpos"][APPLE_ADR + 2]) - z0)
    ab = np.array(ab); ah = np.array(ah); az = np.array(az)
    lifted = az > 0.020
    cidx = np.flatnonzero((ab <= 0) & lifted)
    t_c = int(ts[cidx[0]]) if len(cidx) else None
    out = {"run_dir": str(rd), "server": str(srv), "obstacle_config": cfg_path, "profile": profile,
           "bar": {"name": bar["name"], "radius_mm": 1e3 * bar["radius"], "half_len_mm": 1e3 * bar["half_len"], "centre_world": bar["centre"],
                   "a_world": bar["a"], "b_world": bar["b"]},
           "settings": {"esdf_margin": G["settings"].esdf_margin, "links": G["settings"].links, "target_field_policy": G["settings"].target_field_policy},
           "apple_bar_contact_t": t_c, "n_rows_apple_bar_le0": int(((ab <= 0) & lifted).sum()),
           "apple_bar_min_mm": mm(ab[lifted].min()) if lifted.any() else None,
           "apple_bar_argmin_t": int(ts[np.flatnonzero(lifted)[np.argmin(ab[lifted])]]) if lifted.any() else None,
           "series": {"t": ts, "apple_bar_mm": (1e3 * ab).tolist(), "apple_hurdle_mm": (1e3 * ah).tolist(), "apple_dz_mm": (1e3 * az).tolist()}}
    try:
        tr = np.load(rd / "traj.npz", allow_pickle=True)
        if "obstacle_safety_json" in tr.files:
            osj = json.loads(str(tr["obstacle_safety_json"]))
            out["client_obstacle_safety"] = {k: osj.get(k) for k in ("payload_collision", "robot_collision", "contact_steps", "contact_pairs", "min_robot_clearance_m")}
    except Exception as e:  # noqa: BLE001
        out["client_obstacle_safety"] = {"error": repr(e)}
    t_ref = t_c if t_c is not None else out["apple_bar_argmin_t"]
    chunks = []
    seqs = sorted(int(s) for s in ix["chunks"])
    held_seqs = []
    for s in seqs:
        with np.load(ix["chunks"][str(s)]["npz"]) as z:
            summ = json.loads(str(z["summary_json"]))
        if (summ.get("grasp") or {}).get("state") == "held" and held_from_summary(summ) is not None:
            if t_ref is None or plan[s]["t_step"] <= t_ref:
                held_seqs.append(s)
    window = set(held_seqs[-N_WINDOW:])
    for s in held_seqs:
        chunks.append(chunk_one(G, ix["chunks"][str(s)]["npz"], plan[s], ctrl, bar, mg, detail=(s in window)))
    out["held_chunks"] = chunks
    out["n_held_chunks_before_ref"] = len(held_seqs)
    out["ref_row_t"] = t_ref
    return jl(out)


def chunk_one(G, npz, p, ctrl, bar, mg, detail):
    m, d = mj(); robot = G["robot"]; layout = G["layout"]; qadr = G["qadr"]
    z = np.load(npz); summ = json.loads(str(z["summary_json"]))
    held = held_from_summary(summ)
    fin = np.asarray(z["refined_chunk"], float)
    q = np.asarray(p["qpos"], float)
    q_now = FIA.q_now_from_qpos(q, qadr)
    H = 8
    traj = layout.chunk_to_trajectory(fin)[:, :H]; full = layout.full_q(traj, q_now)
    field = FIA.build_field(z, G["outside"])
    to = summ.get("to") or {}; met = to.get("metrics") or {}; mvp = lit(met.get("max_violation_pair"))
    rad = held["radii"]
    # held centres along the plan (base frame) and at q_now
    C = np.stack([spheres_base(G, held, full[:, k])[0] for k in range(full.shape[1])])     # (H, n, 3)
    c_now, f_now = spheres_base(G, held, q_now)
    dfield = field.distance(C.reshape(-1, 3)).reshape(C.shape[:2])
    margin = float(G["settings"].esdf_margin)
    to_clear = dfield - rad[None, :] - margin
    bar_b = {"a": to_base(bar["a"], q), "b": to_base(bar["b"], q), "r": bar["radius"]}
    an = seg_dist(C.reshape(-1, 3), bar_b["a"], bar_b["b"])[0].reshape(C.shape[:2]) - bar_b["r"] - rad[None, :]
    # true apple-bar along the plan (apple rigid with EE_BODY_L)
    ft = np.zeros(6); true_plan = []
    for k in range(H):
        qk = apple_moved_with_hand(q, q_arms_from_chunk_row(q, fin[k])); set_q(qk)
        true_plan.append(mg.pair_distance(_G["apple_g"], bar["gid"], 0.30, ft))
    set_q(q); true_now = mg.pair_distance(_G["apple_g"], bar["gid"], 0.30, ft)
    # executed rows of this chunk
    trows = [t for t in range(p["t_step"], p["t_step"] + 8) if t in ctrl]
    ex = {"t": trows, "executed": [bool(ctrl[t].get("executed", True)) for t in trows], "true_apple_bar_mm": [], "TO_view_held_mm": [], "analytic_held_bar_mm": []}
    for t in trows:
        qm = np.asarray(ctrl[t]["qpos"], float); set_q(qm)
        ex["true_apple_bar_mm"].append(mm(mg.pair_distance(_G["apple_g"], bar["gid"], 0.30, ft)))
        cm, _ = spheres_base(G, held, FIA.q_now_from_qpos(qm, qadr))
        cm_b = to_base(to_world(cm, qm), q)                    # into this chunk's base frame
        ex["TO_view_held_mm"].append(mm((field.distance(cm_b) - rad - margin).min()))
        ex["analytic_held_bar_mm"].append(mm((seg_dist(cm_b, bar_b["a"], bar_b["b"])[0] - bar_b["r"] - rad).min()))
    rec = {"seq": p["seq"], "t_step": p["t_step"], "npz": str(npz),
           "gate": (p.get("gate") or {}).get("action"), "trajopt_status": (p.get("verdict") or {}).get("trajopt_status"),
           "held": {k: v for k, v in held.items() if k not in ("local", "filter_local", "radii", "filter_radii")},
           "held_local": held["local"], "held_radii_mm": 1e3 * rad,
           "filter_local": held["filter_local"], "filter_radii_mm": None if held["filter_radii"] is None else 1e3 * held["filter_radii"],
           "n_held_rows": int(len(rad)),
           "esdf_held_free": lit((summ.get("esdf_stats") or {}).get("held_free")),
           "target_free_layers": lit((summ.get("esdf_stats") or {}).get("target_free")),
           "npz_has_target_free_layer": "esdf_free_distance" in z.files,
           "exclusion": summ.get("exclusion"),
           "TO_worst_row": mvp if isinstance(mvp, dict) else None, "TO_clearance_mm": mm(met.get("clearance_m")),
           "TO_view_held_clearance_mm_per_step": (1e3 * to_clear.min(axis=1)).tolist(),
           "analytic_held_sphere_bar_mm_per_step": (1e3 * an.min(axis=1)).tolist(),
           "true_apple_bar_mm_per_step": [mm(x) for x in true_plan], "true_apple_bar_mm_at_planning_row": mm(true_now),
           "held_centre_displacement_over_plan_mm": float(1e3 * np.linalg.norm(C[-1].mean(0) - C[0].mean(0))),
           "held_centre_step0_minus_q_now_mm": float(1e3 * np.linalg.norm(C[0].mean(0) - c_now.mean(0))),
           "executed": ex}
    # validation against the recorded worst row (attached rows only)
    if isinstance(mvp, dict) and str(mvp.get("link", "")).startswith("attached:"):
        k = int(mvp["step"]); i = int(str(mvp["link"]).split("[")[1].rstrip("]"))
        pm = np.asarray(mvp.get("point_m"), float)
        rec["worst_row_check"] = {"step": k, "sphere": i, "recorded_point_m": pm, "reconstructed_point_m": C[k, i],
                                  "point_diff_mm": float(1e3 * np.linalg.norm(pm - C[k, i])),
                                  "recorded_clearance_mm": mm(mvp.get("clearance_m")), "reconstructed_clearance_mm": mm(to_clear[k, i]),
                                  "point_diff_vs_capture_pose_mm": None}
    if detail:
        # coverage of the true apple mesh by the held spheres at the planning qpos (base frame)
        set_q(q); Vw = geom_vertices_world(_G["apple_g"]); Vb = to_base(Vw, q)
        gap = np.min(np.linalg.norm(Vb[:, None, :] - c_now[None], axis=2) - rad[None], axis=1)
        vb_bar = seg_dist(Vb, bar_b["a"], bar_b["b"])[0] - bar_b["r"]
        near = vb_bar <= 0.015
        apple_c_b = to_base(d.xpos[_G["B"]("apple")].copy(), q)
        rec["coverage_at_planning_row"] = {
            "n_vertices": int(len(Vb)), "frac_inside_held_spheres": float(np.mean(gap <= 0)),
            "vertex_gap_mm": stats_mm(gap), "max_outside_mm": mm(gap.max()),
            "n_vertices_within_15mm_of_bar": int(near.sum()), "gap_of_vertices_within_15mm_of_bar_mm": stats_mm(gap[near]) if near.any() else None,
            "apple_centre_base": apple_c_b, "held_centres_base": c_now, "held_radii_mm": 1e3 * rad,
            "held_centre_minus_apple_centre_mm": (1e3 * (c_now - apple_c_b)).tolist(),
            "apple_bottom_below_held_sphere_bottom_mm": float(1e3 * ((c_now[:, 2] - rad).min() - Vb[:, 2].min())),
            "apple_vertical_extent_mm": float(1e3 * (Vb[:, 2].max() - Vb[:, 2].min())),
        }
        # bar in the recorded field + held_free / filter volumes
        s = np.linspace(0, 1, 49); axis = bar_b["a"] + s[:, None] * (bar_b["b"] - bar_b["a"])
        L = {}
        def lay(prefix): return layer_from_arrays(z[f"{prefix}_distance"], z[f"{prefix}_origin"], float(z[f"{prefix}_voxel_size"]))
        L["coarse_20mm"] = lay("esdf_coarse"); L["fine_5mm"] = lay("esdf")
        vals = {}
        for nm, layer in L.items():
            v = np.full(len(axis), np.nan); cov = CuroboEsdfField._covers(layer.grid, axis)
            if cov.any(): v[cov] = layer.distance(axis[cov])
            vals[nm] = v
        vals["composite"] = field.distance(axis)
        in_free = (np.linalg.norm(axis[:, None] - c_now[None], axis=2) - rad[None]).min(axis=1)
        in_filter = (None if f_now is None else (np.linalg.norm(axis[:, None] - f_now[None], axis=2) - held["filter_radii"][None]).min(axis=1))
        d_apple = np.linalg.norm(axis - apple_c_b, axis=1)
        rec["bar_in_field"] = {
            "true_axis_value_mm": -1e3 * bar_b["r"],
            "field_at_axis_mm": {k: (1e3 * v).tolist() for k, v in vals.items()},
            "axis_point_to_held_sphere_surface_mm": (1e3 * in_free).tolist(),
            "axis_point_to_filter_sphere_surface_mm": None if in_filter is None else (1e3 * in_filter).tolist(),
            "axis_point_to_apple_centre_mm": (1e3 * d_apple).tolist(),
            "n_axis_points_composite_gt0": int(np.sum(vals["composite"] > 0)),
            "n_axis_points_inside_held_free_volume": int(np.sum(in_free <= 0)),
            "n_axis_points_inside_filter_volume": None if in_filter is None else int(np.sum(in_filter <= 0)),
            "composite_at_axis_point_nearest_apple_mm": mm(vals["composite"][np.argmin(d_apple)]),
            "fine_at_axis_point_nearest_apple_mm": mm(vals["fine_5mm"][np.argmin(d_apple)]) if np.isfinite(vals["fine_5mm"][np.argmin(d_apple)]) else None,
            "nearest_axis_point_to_apple_mm": mm(d_apple.min()),
        }
        # arrays for the figure: section through the bar point nearest the apple, normal = bar axis
        ax = (bar_b["b"] - bar_b["a"]) / np.linalg.norm(bar_b["b"] - bar_b["a"])
        tmp = np.array([0, 0, 1.0]); u = np.cross(ax, tmp); u /= np.linalg.norm(u); w = np.cross(ax, u)
        focus = axis[np.argmin(d_apple)]
        g = np.arange(-0.12, 0.1201, 0.0025); U, W = np.meshgrid(g, g, indexing="xy")
        P = focus + U[..., None] * u + W[..., None] * w
        sec = {"grid_m": g, "composite": field.distance(P.reshape(-1, 3)).reshape(U.shape),
               "fine": None, "u": u, "w": w, "axis": ax, "focus_base": focus}
        vf = np.full(P.shape[:2], np.nan).reshape(-1); cov = CuroboEsdfField._covers(L["fine_5mm"].grid, P.reshape(-1, 3))
        if cov.any(): vf[cov] = L["fine_5mm"].distance(P.reshape(-1, 3)[cov])
        sec["fine"] = vf.reshape(U.shape)
        rel = Vb - focus; sec["apple_vertices_uvw"] = np.stack([rel @ u, rel @ w, rel @ ax], 1)
        relc = c_now - focus; sec["held_uvwr"] = np.stack([relc @ u, relc @ w, relc @ ax, rad], 1)
        if f_now is not None:
            relf = f_now - focus; sec["filter_uvwr"] = np.stack([relf @ u, relf @ w, relf @ ax, held["filter_radii"]], 1)
        # fingers / EE (MuJoCo mesh) for the scene
        verts = []
        for bn in ("EE_BODY_L", "ee_finger_l1", "ee_finger_l2"):
            bid = _G["B"](bn)
            for gg in range(m.ngeom):
                if int(m.geom_bodyid[gg]) == bid and (m.geom_contype[gg] or m.geom_conaffinity[gg]) and int(m.geom_type[gg]) == 7:
                    verts.append(to_base(geom_vertices_world(gg), q) - focus)
        Vh = np.concatenate(verts); sec["hand_vertices_uvw"] = np.stack([Vh @ u, Vh @ w, Vh @ ax], 1)
        rec["_section"] = sec
    z.close()
    return rec


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    subs = sys.argv[2:]
    keys = [k for k in IDX if ("/C/" in k or "/D/" in k) and (not subs or any(s in k for s in subs))]
    with Pool(nw) as pool:
        res = pool.map(run_one, keys, chunksize=1)
    secs = {}
    for k, r in zip(keys, res):
        for c in r.get("held_chunks", []):
            if "_section" in c: secs[f"{k}#{c['seq']}"] = c.pop("_section")
    tag = "all" if not subs else "_".join(s.replace("/", "-") for s in subs)
    (OUT / f"held_runs_{tag}.json").write_text(json.dumps({"definitions": __doc__, "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                                                           "runs": dict(zip(keys, res))}, indent=0))
    np.save(OUT / f"held_sections_{tag}.npy", secs, allow_pickle=True)
    print("done", len(res))


if __name__ == "__main__":
    main()
