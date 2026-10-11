"""T43 T step 2 — fixed-input replay of the near-fruit closing chunks (grasp state closing, not attached,
fingertip midpoint <= 60 mm from the fruit, T43 U's selection) of T41 a and T43 V1 runs (obstacle margin 10 mm).

off   = as recorded: the frame's (shrunk) target points, the recorded target-free layer.
(i)   = `closing_keep_geometry`: the last latched chunk's target points and its carve ball; the target-free
        layer **emulated** = the recorded main fine layer with that ball freed above its z_min (EDT, body_emul).
(i+ii)= (i) + `target_volume_exempt` (rows whose nearest surface point is in the ball: no obstacle margin).

Per variant: verdict (classification), apple (GT centre + 34 mm) inside the carve ball, deflection of the
fingertip midpoint at the last planned step (refined vs reference; radial > 0 = pushed away from the fruit),
and the robot rows on the recorded **main** field whose nearest surface point is outside the ball (bar, table,
crate ...): min clearance and the number below -0.1 mm."""
from common.workspace import workspace_path as _ag3s_workspace_path
import ast, json, sys, pathlib, dataclasses, glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))

U = pathlib.Path(_ag3s_workspace_path('legacy/outputs/verify/T43/U/raw'))
S = pathlib.Path(_ag3s_workspace_path('legacy/outputs/verify/T43/S/raw'))
IX = {**json.load(open(S / "chunk_index_t40t41.json")), **json.load(open(U / "chunk_index_t43v_t42_2.json"))}
XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
APPLE_R = 0.034


def run_dir_of(key):
    camp, rest = key.split("/", 1)[0], key
    if key.startswith("T41/a/"):
        return _ag3s_workspace_path('legacy/outputs/verify/T41/a/runs/E3b/') + key[len("T41/a/"):]
    if key.startswith("T43/V/H/"):
        name = key.split("/")[-1]; v = name.split("_")[0]
        return f"{_ag3s_workspace_path('legacy')}/outputs/verify/T43/V/runs/{v}/H/{name}"
    raise KeyError(key)


def todo():
    out = []
    for f in sorted(glob.glob(str(U / "runs/T41__a__*")) + glob.glob(str(U / "runs/T43__V__H__V1_*"))):
        r = json.load(open(f))
        for d in r["rows"]:
            if "error" in d or d["grasp_state"] != "closing" or (d.get("tip_now_to_fruit_mm") or 1e9) > 60:
                continue
            out.append((r["key"], d["seq"], r["fruit"], r["arm"]))
    return out


class _Args:
    links = None; sphere_spacing = None; max_spheres_per_capsule = None; capsule_radius_scale = None
    max_sphere_radius = None; esdf_margin = None; target_field_policy = None; phase = None
    manipulators = None; model_xml = None; hand = None; allow_uncertified = False; rows_per_step = None


_SET = {}


def setup(rd):
    from AG3S.trajopt.experiments import fixed_input_ablation as FIA
    from AG3S.trajopt.types import ChunkLayout
    from AG3S.ag3s.robot_models.gripper_state import FingerJointMap
    ix = IX[rd]
    key = ix["server"]
    if key not in _SET:
        man = json.loads((pathlib.Path(rd) / "manifest.json").read_text())
        man["scene"]["model_xml"] = XML
        log = FIA.parse_server_log(str(pathlib.Path(ix["server"]) / "serve_safe.log"))
        st = FIA.resolve_settings(_Args, log, man)
        robot, qadr, _ = FIA.build_robot(st)
        _, _, _, maxd, unk = FIA.contact_policy(st, robot.sphere_link_names, has_target=False, has_free=False, attached=False)
        _SET[key] = dict(FIA=FIA, st=st, robot=robot, qadr=qadr, layout=ChunkLayout.rby1(robot.joint_names),
                         outside=maxd if unk == "free" else -maxd, fmap=FingerJointMap(robot))
    G = dict(_SET[key])
    man = json.loads((pathlib.Path(rd) / "manifest.json").read_text())
    G["manipulators"] = man["policy"]["safe_manipulators"]
    return G


def planes(z):
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.stages.support_surface import fit_support_surfaces
    from AG3S.ag3s.types import PointCloud
    surf, _ = fit_support_surfaces(PointCloud(points=np.asarray(z["filtered_cloud"], np.float64)),
                                   AG3SConfig().support_surface, timestamp=0.0)
    return np.asarray([[*np.asarray(s.normal, float), float(s.offset)] for s in surf], float).reshape(-1, 4)


def lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def one(item):
    key, seq, fruit, arm = item
    import body_emul as BE
    from AG3S.trajopt.refiner import TrajOptChunkRefiner
    from AG3S.trajopt.safe_policy import classify_violations, fingertip_alignment
    rd = run_dir_of(key)
    G = setup(rd); FIA = G["FIA"]; st = G["st"]; robot = G["robot"]; layout = G["layout"]
    ix = IX[rd]["chunks"]
    import body_emul as BE0
    load = lambda s: BE0.Z({k: v for k, v in np.load(ix[str(s)]["npz"], allow_pickle=False).items()})  # noqa: E731
    z = load(seq)
    s = json.loads(str(z["summary_json"]))
    pre = seq - 1
    while pre >= 1 and json.loads(str(load(pre)["summary_json"]))["grasp"]["state"] != "latched":
        pre -= 1
    zp = load(pre); sp = json.loads(str(zp["summary_json"]))
    ball = (lit(sp["esdf_stats"]["target_free"]) or {}).get("ball")
    pl_z_rec = float(ball["support_plane_z_m"])
    plan = {}
    for l in (pathlib.Path(rd) / "frames.jsonl").read_text().splitlines():
        if '"planning"' in l[:40]:
            r = json.loads(l); plan[r["seq"]] = r
    qn = FIA.q_now_from_qpos(np.asarray(plan[seq]["qpos"], float), G["qadr"])
    apple = np.asarray(plan[seq]["object_poses"][fruit]["pos"], float)
    refc = np.asarray(z["reference_chunk"], float)
    pl = planes(z)
    pl_z = None   # set from the carve ball record below
    hand = arm
    fingers = ("ee_finger_l1", "ee_finger_l2") if hand == "left" else ("ee_finger_r1", "ee_finger_r2")
    hold = "ee_left" if hand == "left" else "ee_right"
    res = {"key": key, "seq": seq, "pre": pre, "ball": ball,
           "n_points": {"closing": int(len(z["target_points"])), "pre": int(len(zp["target_points"]))}}
    rec_ball = (lit(s["esdf_stats"]["target_free"]) or {}).get("ball")
    res["ball_closing"] = rec_ball

    def scene_for(variant):
        if variant == "off":
            field = FIA.build_field(z, G["outside"]); tp = np.asarray(z["target_points"], float)
        else:
            zz = BE.Z(z)
            c = np.asarray(ball["centre_m"], float); r = ball["radius_mm"] / 1000.0
            freed, n = BE.free_layer(z["esdf_distance"], z["esdf_origin"], float(z["esdf_voxel_size"]),
                                     [(c, r)], float(ball["z_min_m"]))
            zz["esdf_free_distance"] = freed; zz["esdf_free_origin"] = z["esdf_origin"]
            zz["esdf_free_voxel_size"] = z["esdf_voxel_size"]
            for k in list(zz):
                if k.startswith("esdf_free_") and k not in ("esdf_free_distance", "esdf_free_origin", "esdf_free_voxel_size"):
                    del zz[k]
            field = FIA.build_field(zz, G["outside"]); tp = np.asarray(zp["target_points"], float)
        margin, mask, auth, _, _ = FIA.contact_policy(st, robot.sphere_link_names, has_target=True,
                                                      has_free=bool(field.target_free_layers), attached=False)
        sc = FIA.make_scene(field, robot.radii, st.esdf_margin, target_points=tp, margin=margin, mask=mask)
        sc = dataclasses.replace(sc, obstacle_margin=0.010, support_planes=pl)
        if variant == "i+ii":
            sc = dataclasses.replace(sc, target_volume=np.asarray(
                [*ball["centre_m"], ball["radius_mm"] / 1000.0, ball["z_min_m"]], float))
        return sc, auth

    main_field = FIA.build_field(z, G["outside"])
    for variant in ("off", "i", "i+ii"):
        sc, auth = scene_for(variant)
        tm = (s.get("to") or {}).get("metrics") or {}
        cfg = FIA.to_config(st, collision=True, cost=dict(st.cost),
                            sqp={"time_budget_ms": FIA.NO_CLOCK_MS, "max_iterations": max(1, int(tm.get("sqp_iterations") or 1))})
        cfg = cfg.with_overrides({"collision": {"obstacle_margin": 0.010}})
        ref = TrajOptChunkRefiner(robot, layout, lambda _c: (sc, qn, True), cfg)
        ref.reset()
        lin = ref.optimizer.linearizer
        meas = (s.get("finger_joints") or {}).get("opening_norm") or {}
        path, _ = G["fmap"].path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, lin.horizon,
                                 default=robot.param_vector())
        lin.set_joint_parameter_path(path)
        ctx = {"t_step": seq}
        fb = s.get("exec_feedback") or {}
        if fb.get("available") and int(fb.get("n_exec") or 0) > 0:       # continuity, as recorded
            ctx["previous_physical_chunk"] = np.asarray(load(seq - 1)["refined_chunk"], np.float64)
            ctx["previous_executed_steps"] = int(fb["n_exec"])
        ref.refine(refc, ctx)
        r = ref.last_result
        traj = np.asarray(r.trajectory, float)
        reft = layout.chunk_to_trajectory(refc)[:, :lin.horizon]
        cls = classify_violations(lin, traj, qn, sc, authorized_links=auth, tolerance=1e-4, manipulated={})
        # fingertips at the last planned step
        robot.set_joint_parameters(G["fmap"].joint_values(meas))
        tip = lambda tr: np.asarray(fingertip_alignment(robot, layout.full_q(tr[:, -1:], qn)[:, 0], fingers, hold,  # noqa: E731
                                                        apple)["now"]["tip_m"])
        tf, tr_ = tip(traj), tip(reft)
        # rows on the recorded main field, nearest surface point outside the ball
        cen = lin.sphere_states(traj, qn)[0][:, :lin.n_spheres]
        flat = cen.reshape(-1, 3)
        d = np.asarray(main_field.distance(flat), float)
        g = np.asarray(main_field.gradient(flat), float); g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        surf = flat - d[:, None] * g
        c = np.asarray(ball["centre_m"]); rr = ball["radius_mm"] / 1000.0
        outside = ~((np.linalg.norm(surf - c, axis=1) <= rr) & (surf[:, 2] >= ball["z_min_m"]))
        cl = d - np.tile(lin.robot_radii, cen.shape[0])
        res[variant] = {
            "status": r.status.value, "returned": r.metrics.get("returned"),
            "initial_merit": r.metrics.get("initial_merit"), "continuity_active": r.metrics.get("continuity_active"),
            "rec_initial_merit": tm.get("initial_merit") if variant == "off" else None,
            "n_collision": cls["n_collision"], "n_unresolved": cls["n_unresolved"], "n_allowed": cls["n_allowed"],
            "hold": bool(cls["n_collision"] or cls["n_unresolved"]),
            "defl_mm": round(1e3 * float(np.linalg.norm(tf - tr_)), 2),
            "radial_mm": round(1e3 * float(np.linalg.norm(tf - apple) - np.linalg.norm(tr_ - apple)), 2),
            "nonfruit_min_mm": round(1e3 * float(cl[outside].min()), 2) if outside.any() else None,
            "nonfruit_n_below": int((cl[outside] < -1e-4).sum()),
            # where the rows below 0 meet the field: nearest surface point height over the table plane
            "nonfruit_below_surface_above_plane_mm": sorted({round(1e3 * float(surf[i, 2] - pl_z_rec), 1)
                                                             for i in np.flatnonzero(outside & (cl < -1e-4))})[:8],
            "refined_equal_record": bool(np.array_equal(np.asarray(r.chunk, float).astype(np.float32), z["refined_chunk"]))
            if variant == "off" else None,
            "refined_maxdiff_record_rad": round(float(np.abs(np.asarray(r.chunk, float)[:lin.horizon] - z["refined_chunk"][:lin.horizon]).max()), 5)
            if variant == "off" else None}
    cb = res["ball_closing"]
    res["apple_in_ball"] = {
        "closing": None if cb is None else bool(np.linalg.norm(apple - np.asarray(cb["centre_m"])) + APPLE_R <= cb["radius_mm"] / 1e3),
        "pre": bool(np.linalg.norm(apple - np.asarray(ball["centre_m"])) + APPLE_R <= ball["radius_mm"] / 1e3)}
    print(key, seq, res["apple_in_ball"], {v: (res[v]["hold"], res[v]["defl_mm"], res[v]["radial_mm"], res[v]["nonfruit_min_mm"], res[v]["nonfruit_n_below"]) for v in ("off", "i", "i+ii")},
          "fid", res["off"]["refined_equal_record"], flush=True)
    return res


if __name__ == "__main__":
    items = todo()
    print(len(items), "chunks", flush=True)
    with ProcessPoolExecutor(max_workers=int(sys.argv[2])) as ex:
        res = list(ex.map(one, items))
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=str))
