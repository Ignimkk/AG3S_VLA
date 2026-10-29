"""T43 Z1 — held query spheres (body cover) at attach, apple mesh, closing carve-out ball and crate inner box, per run.
(verifier, CPU only; MUJOCO_GL=osmesa; no new runs — server chunk records of T43 W + MuJoCo FK)

usage: z1_held.py OUT_JSON

Runs: every V4 / V5 run of T43/W/analysis/analysis.json (H and B) and V5S / V5B of T43/W/analysis_blind/analysis.json.
Server segment = analysis server_segment (analyze_t41 pairing rule). Attach chunk = first server chunk with grasp.attach true.
At the attach chunk (planning row of that chunk, MuJoCo FK on its recorded qpos, model = hurdle_metrics.model()):
  apple       : body 'apple' pose; mesh = collision mesh apple_geom vertices (world); apple_bottom_z = min vertex z;
                apple half-extents along world x / y / z (max |vertex - apple centre|).
  held spheres: grasp.attached.held.body_cover.{body_parent_frame, lifted_parent_frame} (radius_mm, centre in the frame of
                frame_link ee_left|ee_right == MJCF body EE_BODY_L|R) placed with the FK pose of that body.
                query set used by the server = lifted_parent_frame if body_cover.frame.near_support else body_parent_frame (server_y rule).
                also: held.spheres (fit set, radius_mm) and body_cover mode / segments_reason.
  carve-out   : esdf_stats.target_free.ball (centre_m world, radius_mm) of the last chunk BEFORE the attach chunk whose
                grasp.state == 'closing' (else the last chunk before attach with a ball); stored relative to the apple centre at
                that chunk's planning row ('ball_rel_apple_mm') and placed on the attach-row apple centre.
Apple-centred frame = world axes, origin = apple body centre at the attach row. The crate body quaternion in these runs is
  reported (crate_quat_attach); crate axes == world axes when it is (1, 0, 0, 0).
Crate inner box (MJCF model_transport_pick_place_obstacles.xml, collision boxes): inner half-width x = 0.086 - 0.004 = 0.082 m,
  y = 0.146 - 0.004 = 0.142 m, floor top z = -0.056 + 0.004 = -0.052 m, wall top z = 0.004 + 0.056 = 0.060 m (crate frame).
Fit numbers per sphere set, in the apple-centred frame with the attach-row hand orientation (no rotation applied):
  half_extent_x_mm / _y_mm = (max(c + r) - min(c - r)) / 2 over the set's spheres, along world x / y;
  need_x_mm = half_extent_x_mm + margin (20 mm destination margin); fits_x = need_x_mm <= 82 (and y <= 142) — i.e. the set
  plus margin fits between the inner walls when the set's x-span is centred between them.
  bottom_below_apple_mm = apple_bottom_z - min(c_z - r) (> 0: the sphere bottom is below the apple's lowest vertex);
  apple_bottom_above_floor_needed_mm = bottom_below_apple_mm + margin (apple height above the crate floor top at which the
  lowest sphere's clearance to the floor equals the margin).
  cover_radius_about_apple_mm = max over spheres of |c - apple centre| + r."""
from common.workspace import workspace_path as _ag3s_workspace_path
import ast, json, os, pathlib, sys
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import hurdle_metrics as HM  # noqa: E402

W = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/W'))
MARGIN = 20.0; IN_X, IN_Y, FLOOR, TOP = 82.0, 142.0, -52.0, 60.0


def lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def qmat(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)


def fit_numbers(sph, apple_c, apple_bottom):
    c = np.array([s[0] for s in sph]); r = np.array([s[1] for s in sph])
    hx = 0.5 * (np.max(c[:, 0] + r) - np.min(c[:, 0] - r)); hy = 0.5 * (np.max(c[:, 1] + r) - np.min(c[:, 1] - r))
    bb = apple_bottom - np.min(c[:, 2] - r)
    return {"half_extent_x_mm": hx, "half_extent_y_mm": hy, "need_x_mm": hx + MARGIN, "need_y_mm": hy + MARGIN,
            "fits_x": bool(hx + MARGIN <= IN_X), "fits_y": bool(hy + MARGIN <= IN_Y), "spare_x_mm": IN_X - hx - MARGIN, "spare_y_mm": IN_Y - hy - MARGIN,
            "bottom_below_apple_mm": bb, "apple_bottom_above_floor_needed_mm": bb + MARGIN,
            "cover_radius_about_apple_mm": float(np.max(np.linalg.norm(c - apple_c, axis=1) + r)),
            "max_radius_mm": float(r.max())}


def one(key, r):
    side = r["side"]; HM.set_arm(side); HM._M.clear(); M = HM.model(); m, d = M["m"], M["d"]
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    SD = "L" if side == "left" else "R"; ee = B(f"EE_BODY_{SD}"); ab = B("apple"); cb = B("crate")
    ga = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom"); mid = int(m.geom_dataid[ga])
    av = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    af = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]].astype(int)
    rows = [json.loads(l) for l in open(pathlib.Path(r["run_dir"]) / "frames.jsonl") if l.strip()]
    plan = {p["seq"]: p for p in rows if p["kind"] == "planning"}
    seg = r.get("server_segment")
    if not seg:
        return {"key": key, "error": "no server segment"}
    f0 = pathlib.Path(seg["first"]); i0 = int(f0.stem.split("_")[1]); dd = f0.parent
    chunks = []
    for n in range(seg["n"]):
        z = np.load(dd / f"chunk_{i0 + n:05d}.npz", allow_pickle=True); s = json.loads(str(z["summary_json"]))
        chunks.append((int(z["t_step"]), s))
    ia = next((i for i, (_, s) in enumerate(chunks) if (s.get("grasp") or {}).get("attach")), None)
    if ia is None:
        return {"key": key, "attach": False, "grasp": r.get("grasp"), "success": r.get("success")}

    def pose(seq):
        p = plan[seq]; d.qpos[:] = p["qpos"]; mujoco.mj_kinematics(m, d)
        return p, d.xpos[ee].copy(), d.xmat[ee].reshape(3, 3).copy(), d.xpos[ab].copy(), d.xquat[ab].copy(), d.xpos[cb].copy(), d.xquat[cb].copy()
    seq, s = chunks[ia]; g = s["grasp"]; att = lit(g.get("attached")); held = lit(att.get("held")) or {}; bc = lit(held.get("body_cover")) or {}
    p, pe, Re, pa, qa, pc, qc = pose(seq)
    Ra = qmat(qa)
    V = pa + Ra @ m.geom_pos[ga] + (Ra @ qmat(m.geom_quat[ga]) @ av.T).T
    a_bottom = float(V[:, 2].min()) * 1e3; ac = pa * 1e3
    near = bool((bc.get("frame") or {}).get("near_support"))
    sets = {}
    for nm in ("body_parent_frame", "lifted_parent_frame"):
        L = bc.get(nm) or []
        sets[nm] = [((pe + Re @ np.asarray(x["centre_m"], float)) * 1e3, float(x["radius_mm"])) for x in L]
    used = "lifted_parent_frame" if near else "body_parent_frame"
    # carve-out ball
    ball = None; bseq = None; bstate = None
    for i in range(ia - 1, -1, -1):
        sq, ss = chunks[i]; st = lit((ss.get("esdf_stats") or {}).get("target_free")) or {}
        b = st.get("ball") if isinstance(st, dict) else None
        if not (isinstance(b, dict) and b.get("centre_m") is not None):
            continue
        stt = (ss.get("grasp") or {}).get("state")
        if stt == "closing":
            ball, bseq, bstate = b, sq, stt; break
        if ball is None:
            ball, bseq, bstate = b, sq, stt
    out = {"key": key, "attach": True, "side": side, "success": r.get("success"), "grasp": r.get("grasp"), "place": r.get("place"),
           "attach_seq": seq, "attach_t": p["t_step"], "apple_centre_mm": ac.tolist(), "apple_bottom_z_mm": a_bottom,
           "apple_half_extent_mm": (np.abs(V * 1e3 - ac).max(axis=0)).tolist(), "crate_quat_attach": qc.tolist(),
           "apple_mesh_rel_mm": (V * 1e3 - ac).tolist(), "apple_mesh_faces": af.tolist(),
           "ee_pos_mm": (pe * 1e3).tolist(), "ee_R": Re.tolist(),
           "body_cover_mode": bc.get("mode"), "segments_reason": bc.get("segments_reason"), "near_support": near, "query_set": used,
           "fit_radii_mm": [x.get("radius_mm") for x in (held.get("spheres") or [])], "fit_sphere_radius_mm": (held.get("fit") or {}).get("radius_mm"),
           "body_radius_mm": bc.get("body_radius_mm"), "pad_mm": bc.get("pad_mm"), "slip_mm": bc.get("slip_mm"),
           "sets": {}, "fit": {}}
    for nm, L in sets.items():
        out["sets"][nm] = [{"centre_rel_apple_mm": (c - ac).tolist(), "radius_mm": rr} for c, rr in L]
        if L:
            out["fit"][nm] = fit_numbers(L, ac, a_bottom)
    if ball is not None:
        pb, *_ , pab, _, _, _ = pose(bseq)
        out["carve_ball"] = {"seq": bseq, "t": plan[bseq]["t_step"], "grasp_state": bstate, "radius_mm": float(ball["radius_mm"]),
                             "radius_raw_mm": ball.get("radius_raw_mm"), "pad_mm": ball.get("pad_mm"),
                             "ball_rel_apple_mm": (np.asarray(ball["centre_m"]) * 1e3 - pab * 1e3).tolist()}
    return out


if __name__ == "__main__":
    res = {"definitions": __doc__, "margin_mm": MARGIN, "crate_inner_mm": {"half_x": IN_X, "half_y": IN_Y, "floor_top_z": FLOOR, "wall_top_z": TOP}, "runs": {}}
    for f, pre in ((W / "analysis/analysis.json", ("V4/", "V5/")), (W / "analysis_blind/analysis.json", ("V5S/", "V5B/"))):
        A = json.load(open(f))["runs"]
        for k in sorted(A):
            if k.startswith(pre):
                res["runs"][k] = one(k, A[k]); o = res["runs"][k]
                print(k, o.get("attach"), o.get("body_cover_mode"), o.get("query_set"), {n: round(v["max_radius_mm"], 1) for n, v in o.get("fit", {}).items()},
                      (o.get("carve_ball") or {}).get("radius_mm"), flush=True)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))
