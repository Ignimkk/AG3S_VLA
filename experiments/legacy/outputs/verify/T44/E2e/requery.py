"""T44e V-E2e offline re-query (verifier). For each of the 7 T44d pairs (C1b != C1a at the first material chunk):
rebuild the C1b chunk's field from the recorded CL inputs (planning row qpos + object_poses -> TransportScene render, 3 history rows,
identity = the recorded `target_points` matched into the re-rendered cloud, method ball_owned tie 1 max_dist 3 = the CL server), build the coarse
target-free tier as keep / exclude / exclude_target, compose, and query the SAME q (recorded reference_chunk / refined_chunk of that chunk,
active hand authorized spheres, rows 0..7 = certified window) as in T44dA field_scan.
Also queries the RECORDED float16 grids (coarse ON = served keep, coarse OFF = C1a formula) with the current CuroboEsdfField (formula unchanged by A8).
Checks: re-rendered cloud vs recorded filtered_cloud; matched fraction of recorded target points; rebuilt ball vs recorded ball;
rebuilt keep vs recorded ON at the query spheres.
Output -> outputs/verify/T44/E2e/requery/<pair>.json   usage: requery.py PAIR ... (run from ${AG3S_PROJECT_ROOT}, GPU)"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json, pathlib, sys, time
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/E2b'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/E2'))
sys.path.insert(0, _ag3s_workspace_path(''))
import e2b_measure as B  # noqa: E402
from AG3S.ag3s.experiments.live import exclusion_e2 as H  # noqa: E402

CL = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CL'))
CLA = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CLA'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2e/requery'))
PAIRS = ["ep12_s122", "ep0_s1", "ep11_s112", "ep123_s1231", "ep123_s1232", "ep104_s1042", "ep110_s1101"]
HISTORY = 3
VARIANTS = ("recorded_off", "recorded_keep", "off", "keep", "exclude", "exclude_target")


def q_from_qpos(sc, qm, qpos):
    qpos = np.asarray(qpos, np.float64)
    return np.asarray([qpos[sc._qadr[n]] for n in qm.joint_names], np.float64)


def params_from_qpos(sc, qm, qpos):
    names = tuple(getattr(qm, "param_joint_names", ()) or ())
    if not names:
        return None
    qpos = np.asarray(qpos, np.float64)
    return np.asarray([qpos[sc._qadr[n]] for n in names], np.float64)


def chunk_q(sc, qm, qpos, chunk):
    base = q_from_qpos(sc, qm, qpos)
    names = list(qm.joint_names)
    out = np.repeat(base[None], len(chunk), 0)
    for i in range(7):
        out[:, names.index(f"left_arm_{i}")] = chunk[:, i]
        out[:, names.index(f"right_arm_{i}")] = chunk[:, 8 + i]
    return out


def sphere_xyz(qm, q, params):
    c = qm.sphere_centers_numeric(q, params) if params is not None else qm.sphere_centers_numeric(q)
    return np.asarray(c[0] if isinstance(c, tuple) else c, np.float64)


def zero_levels(field, ball, sz):
    zs = np.arange(sz - 0.010, sz + 0.040 + 1e-9, 0.001)
    R = float(ball.radius)
    fit = (ball.provenance or {}).get("fit") or {}
    Rf = float(fit.get("radius_mm") or (R * 1e3 - 5.0)) / 1e3
    c = np.asarray(ball.centre, np.float64)
    th = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    xy = [c[:2]] + [c[:2] + rr * np.array([np.cos(t), np.sin(t)]) for rr in (Rf, R + 0.010) for t in th]
    P = np.asarray([[x, y, z] for x, y in xy for z in zs])
    V = np.asarray(field.target_free_distance(P), np.float64).reshape(len(xy), -1)
    lv = []
    for col in V:
        k = np.where((zs >= sz - 1e-9) & (col > 0))[0]
        lv.append(1e3 * (zs[k[0]] - sz) if len(k) else np.nan)
    lv = np.asarray(lv)
    return {"centre": float(lv[0]), "ringfit": float(np.nanmedian(lv[1:9])), "ringout": float(np.nanmedian(lv[9:]))}


def run(pair):
    import torch
    from scipy.spatial import cKDTree
    from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene, set_finger_joints_from_scene
    from AG3S.ag3s.fields.curobo_builder import OwnedEvidence
    from AG3S.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays
    from AG3S.ag3s.fields.device_field import layer_from_device
    from AG3S.ag3s.runtime.asset_path import resolve_asset
    from AG3S.ag3s.types import TargetGeometry
    from AG3S.trajopt import wire
    from AG3S.trajopt.serve_safe import build_ag3s, constraint_link_filter

    dv = json.load(open(CLA / "diverge_C1a_C1b.json"))[pair]
    fm = int(dv["first_mat"])
    a = json.load(open(CL / "analysis" / f"C1b_{pair.replace('ep', 'ep', 1)}.json"))
    side = a["used_arm"]; s1 = "l" if side == "left" else "r"; target = a["target_fruit"]
    rows_all = [json.loads(l) for l in open(pathlib.Path(_ag3s_workspace_path('')) / a["run_dir"] / "frames.jsonl")]
    plan = [r for r in rows_all if r["kind"] == "planning" and r.get("qpos") is not None]
    seqs = [int(r["seq"]) for r in plan]
    i = seqs.index(fm)
    row = plan[i]; hist = plan[max(0, i - HISTORY):i]
    ch = [c for c in a["server_chunks"] if c["seq"] == fm][0]
    z = np.load(ch["server_file"], allow_pickle=True)
    s = json.loads(str(z["summary_json"]))
    mask = np.asarray(s["target_permission"]["target_field_exclude"], bool)
    tf = s["esdf_stats"]["target_free"]; tf = eval(tf) if isinstance(tf, str) else tf
    rball = tf["ball"]
    rec = {"pair": pair, "run": a["name"], "first_mat_seq": fm, "target": target, "used_arm": side, "history_rows": len(hist),
           "failure_mode": dv.get("failure_mode"), "server_file": ch["server_file"], "recorded_ball": rball}

    xml = str(resolve_asset("robots/rby1/description/models/rby1a/mujoco/model_transport.xml", what="transport model"))
    scene = TransportScene(xml, settle_steps=0)
    qm = build_constraint_robot_model(scene, link_filter=constraint_link_filter("gripper"),
                                      sphere_options={"capsule_radius_scale": 0.05, "sphere_spacing": 0.4, "max_spheres_per_capsule": 32},
                                      finger_cover={"hands": "both"})
    cameras = tuple(wire.DEFAULT_CAMERAS)
    ag3s = build_ag3s(xml, voxel=0.020, range_max=2.0, links="gripper", backend="curobo", fine_voxel=0.005, tsdf_voxel=0.005,
                      target_field_policy="exclude-authorized",
                      target_exclusion={"owned_tie_voxels": 1.0, "owned_max_dist_voxels": 3.0})
    ctx = {"scene": scene, "ag3s": ag3s, "target": target, "classes": H.label_classes(scene, target)}
    args = type("A", (), {"cameras": cameras, "phase": "approach"})()
    item = {"arm": side}
    for h in hist:
        H._history_step(ctx, (lambda sc, r=h: H.pose(sc, r)), item, args)
    pose_err = H.pose(scene, row)
    frames = {c: scene.capture(c) for c in cameras}
    gt_centre = H.body_in_base(scene, target)
    att = H.gt_attention(scene, frames, target, gt_centre)
    req = wire.pack_request({}, cameras=cameras, depth={c: f.depth for c, f in frames.items()},
                            intrinsics={c: f.camera_intrinsics for c, f in frames.items()},
                            extrinsics={c: f.T_base_cam for c, f in frames.items()},
                            robot_state={c: f.robot_state for c, f in frames.items()},
                            stamps={c: float(fm) for c in cameras}, phase="approach", active_manipulators=[side], seq=int(fm))
    _, sc_ = wire.strip_request(req)
    set_finger_joints_from_scene(scene, ag3s.robot_model)
    obs = wire.unpack_camera_observations(sc_, robot_model=ag3s.robot_model, attention=att)
    cs, debug = ag3s.process_multi_debug(obs, phase="approach", active_manipulators=[side])
    cloud = debug["filtered_cloud"]; pts = np.asarray(cloud.points, np.float64)
    surfaces = list(cs.support_surfaces)
    # ---- identity: recorded target points matched into the re-rendered cloud
    rt = np.asarray(z["target_points"], np.float64); rc = np.asarray(z["filtered_cloud"], np.float64)
    kd = cKDTree(pts)
    d_t, j_t = kd.query(rt)
    d_c, _ = kd.query(rc[:: max(1, len(rc) // 4000)])
    matched = d_t <= 0.002
    idx = np.unique(j_t[matched])
    rec["identity_check"] = {"n_recorded_target_points": int(len(rt)), "frac_matched_2mm": float(matched.mean()),
                             "n_geom_points": int(len(rt)), "n_recorded_in_current_cloud_2mm": int(len(idx)), "recorded_cloud_to_rerender_nn_mm_median": float(np.median(d_c) * 1e3),
                             "recorded_cloud_to_rerender_nn_mm_p90": float(np.percentile(d_c, 90) * 1e3),
                             "n_cloud_recorded": int(len(rc)), "n_cloud_rerender": int(len(pts)), "pose_err_mm": round(pose_err * 1e3, 3)}
    # The recorded target geometry can be an OLDER frame's (occluded; recorded `target_point_indices` is empty and the nontarget source is
    # `proximity`), so its points are NOT in the current cloud (median 8 mm away in ep11_s112). Use the recorded points as they are.
    geom = TargetGeometry(id=-1, points=rt, point_indices=np.zeros(0, np.int64), centroid=rt.mean(axis=0), bounding_geometry=None,
                          attention_score=0.0, confidence=0.0)
    ball = ag3s._target_ball(geom.points, surfaces, voxel=0.005)
    bs = ball.summary(); sz = bs["support_plane_z_m"]
    rec["rebuilt_ball"] = bs
    rec["ball_check"] = {"centre_diff_mm": float(np.linalg.norm(np.asarray(bs["centre_m"]) - np.asarray(rball["centre_m"])) * 1e3),
                         "radius_diff_mm": float(bs["radius_mm"] - rball["radius_mm"]),
                         "support_z_diff_mm": float((bs["support_plane_z_m"] - rball["support_plane_z_m"]) * 1e3)}
    support_mask = debug.get("support_mask")
    nt = ag3s._nontarget_points(cloud, support_mask, geom)["points"]
    # GT labels of the non-target evidence (diagnostic): how much of it is really the target (stale target geometry)?
    gt_pts, gt_lab = H.gt_points(frames.values(), ctx["classes"])
    lab_nt = H.point_labeler(gt_pts, gt_lab, radius=0.003)(nt)
    rec["nontarget_gt_labels"] = {H.LABELS[i]: int((lab_nt == i).sum()) for i in range(len(H.LABELS))}
    near = np.linalg.norm(nt - ball.centre, axis=1) <= ball.radius + 0.03
    rec["nontarget_gt_labels_within_ball_r_plus_30mm"] = {H.LABELS[i]: int(((lab_nt == i) & near).sum()) for i in range(len(H.LABELS))}
    builder = ag3s._esdf_builder
    owned = OwnedEvidence(target_points=geom.points, nontarget_points=nt, tie_voxels=1.0, max_dist_voxels=3.0, unowned="remove")
    B.build_tier(builder, ball); B.build_tier(builder, ball, owned=owned)
    F = B.build_tier(builder, ball, owned=owned)[0]
    ev = ag3s._support_seed_evidence(pts, surfaces, ball)
    ev_t = None if ev is None else {**ev, "inlier_threshold_m": float(ag3s.config.support_surface.distance_threshold)}
    H.build_coarse_free_tier(builder, ball, nontarget=nt, target=geom.points, sign_floor="plane")
    main = cs.esdf
    FL = layer_from_device(F.values, F.origin, F.voxel_size)
    fields, counts = {"off": H.compose_free_field(main, FL, None)}, {}
    for v in ("keep", "exclude", "exclude_target"):
        t, ms = H.build_coarse_free_tier(builder, ball, owned=owned, nontarget=nt, target=geom.points, sign_floor="plane",
                                         support=v, support_evidence=(ev_t if v == "exclude_target" else ev))
        fields[v] = H.compose_free_field(main, FL, layer_from_device(t.values, t.origin, t.voxel_size))
        ex = dict(t.extra.get("exclusion") or {}); ex.pop("params", None)
        counts[v] = {**ex, "n_sign_forced": int(t.extra.get("n_sign_forced", 0))}
    rec["coarse_counts"] = counts
    # ---- recorded grids
    def lay(key):
        return layer_from_arrays(np.asarray(z[key + "_distance"], np.float32), z[key + "_origin"], float(z[key + "_voxel_size"]))
    base = dict(layers=(lay("esdf_coarse"), lay("esdf")), outside_distance=0.0, target_free_layers=(lay("esdf_free"),))
    fields["recorded_keep"] = CuroboEsdfField(**base, target_free_coarse_layer=lay("esdf_free_coarse"))
    fields["recorded_off"] = CuroboEsdfField(**base, target_free_coarse_layer=None)
    # ---- spheres
    links = np.asarray([str(n) for n in qm.sphere_link_names]); rad = np.asarray(qm.radii, np.float64)
    assert mask.size == len(links), (mask.size, len(links))
    act = np.isin(links, [f"ee_finger_{s1}1", f"ee_finger_{s1}2", f"ee_{side}"]) & mask
    fing = np.char.startswith(links, f"ee_finger_{s1}") & mask
    qpos = row["qpos"]; pr = params_from_qpos(scene, qm, qpos)
    Q0 = chunk_q(scene, qm, qpos, np.asarray(z["reference_chunk"], np.float64)[:16])
    c0 = sphere_xyz(qm, Q0[0], pr)
    rec["fk_check_ref_row0_vs_recorded_sphere_centres_max_m"] = float(np.abs(c0 - np.asarray(z["sphere_centres"], np.float64)).max())
    pr_row = dict(zip(*np.unique(links[act], return_counts=True)))
    rec["active_authorized_rows"] = {k: int(v) for k, v in pr_row.items()}
    res = {}
    for key in ("reference_chunk", "refined_chunk"):
        Q = chunk_q(scene, qm, qpos, np.asarray(z[key], np.float64)[:16])
        S = np.stack([sphere_xyz(qm, q, pr) for q in Q])           # (16, n, 3)
        X = S[:, act].reshape(-1, 3); R = np.tile(rad[act], (16, 1)).reshape(-1)
        isf = np.tile(fing[act], (16, 1)).reshape(-1)
        names = np.tile(links[act], (16, 1)).reshape(-1)
        rowi = np.repeat(np.arange(16), int(act.sum()))
        D, T = {}, {}
        for vn in VARIANTS:
            d = np.asarray(fields[vn].target_free_distance(X), np.float64)
            D[vn] = (d - R) * 1e3
            T[vn] = np.asarray(fields[vn].answer_tier(X, target_free=True))
        kr = {}
        for sel_name, sel in (("finger_rows0_7", isf & (rowi < 8)), ("finger_rows0_15", isf), ("active_rows0_7", rowi < 8)):
            kk = {}
            for vn in VARIANTS:
                idxs = np.flatnonzero(sel)
                w = idxs[np.argmin(D[vn][idxs])]
                h_bottom = (X[w, 2] - R[w] - sz) * 1e3
                kk[vn] = {"clr_mm": float(D[vn][w]), "tier": str(T[vn][w]), "link": str(names[w]), "row": int(rowi[w]),
                          "h_bottom_mm": float(h_bottom), "inball": bool(np.linalg.norm(X[w] - ball.centre) <= ball.radius),
                          "clr_off_same_row_mm": float(D["off"][w]), "clr_recorded_off_same_row_mm": float(D["recorded_off"][w]),
                          "implied_lift_vs_off_mm": float(D["off"][w] - D[vn][w]),
                          "implied_lift_vs_recorded_off_mm": float(D["recorded_off"][w] - D[vn][w])}
            # per-variant rows lowered vs own off
            kr[sel_name] = kk
        lowered = {vn: {"n_lower_gt1mm_vs_off": int(((D["off"] - D[vn])[rowi < 8] > 1.0).sum()),
                        "n_cross10_rows0_7": int(((D["off"] >= 10) & (D[vn] < 10))[rowi < 8].sum()),
                        "n_cross0_rows0_7": int(((D["off"] >= 0) & (D[vn] < 0))[rowi < 8].sum()),
                        "max_lift_vs_off_mm_rows0_7": float((D["off"] - D[vn])[rowi < 8].max()),
                        "tier_counts_rows0_7": {t: int(((T[vn] == t) & (rowi < 8)).sum()) for t in np.unique(T[vn])}}
                   for vn in VARIANTS}
        # reproduction of recorded keep by the rebuilt keep, over rows 0..7
        m7 = rowi < 8
        repro = {"keep_vs_recorded_keep_absdiff_mm": {"median": float(np.median(np.abs(D["keep"] - D["recorded_keep"])[m7])),
                                                       "p90": float(np.percentile(np.abs(D["keep"] - D["recorded_keep"])[m7], 90)),
                                                       "max": float(np.abs(D["keep"] - D["recorded_keep"])[m7].max())},
                 "off_vs_recorded_off_absdiff_mm": {"median": float(np.median(np.abs(D["off"] - D["recorded_off"])[m7])),
                                                    "p90": float(np.percentile(np.abs(D["off"] - D["recorded_off"])[m7], 90)),
                                                    "max": float(np.abs(D["off"] - D["recorded_off"])[m7].max())}}
        res[key] = {"worst": kr, "rows": lowered, "repro": repro}
    rec["query"] = res
    rec["zero_level_mm"] = {vn: zero_levels(fields[vn], ball, float(sz)) for vn in VARIANTS}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{pair}.json").write_text(json.dumps(rec, indent=1, default=H._json_default))
    f = res["refined_chunk"]["worst"]["finger_rows0_7"]
    print(pair, "fm", fm, {vn: (round(f[vn]["clr_mm"], 2), f[vn]["tier"], round(f[vn]["implied_lift_vs_off_mm"], 2)) for vn in VARIANTS}, flush=True)
    scene.close()
    del ag3s; torch.cuda.empty_cache()


if __name__ == "__main__":
    for p in (sys.argv[1:] or PAIRS):
        if (OUT / f"{p}.json").exists():
            print("skip", p); continue
        run(p)
