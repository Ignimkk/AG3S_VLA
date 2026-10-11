"""T34 closed-loop: held-object query rows per attached chunk, from the T34 records.
usage: t34_held_rows.py COND   -> p/<COND>_held_rows.json   (needs p/<COND>_analysis.json)
Per run, per chunk whose server summary grasp.attached.held exists (attached in constraints for that chunk):
  spheres  : recorded held.spheres_parent_frame (T34 query spheres, lifted) on held.frame_link; also filter spheres
             (held.filter_spheres_parent_frame = fit + pad) for comparison.
  field    : that chunk's recorded coarse+fine field (fixed_input_ablation.build_field, outside 1.0 m).
  rows     : d(c) - r (esdf_margin 0) at q_now (planning row qpos) and at rows 0-7 of the recorded refined chunk and of the
             reference chunk (ChunkLayout.full_q); min over spheres x rows, per trajectory.
  worst    : for the min row of the refined trajectory: sphere index, row index, the field nearest-surface estimate
             p = c - d(c) grad/|grad| -> GT surface distance per body at p (MuJoCo probe, T28 P1-2) -> nearest GT body
             ('apple' with p above the apple = apple-top trace voxels), p height above the TLS table plane of that chunk
             (support_mask points z > 0.5), GT apple surface distance of p; and the sphere centre's GT gap (surface - r)
             to table / crate / apple.
Also per chunk: the recorded summary max_violation_pair link/clearance when its link starts with 'attached'."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, ast
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P
import AG3S.trajopt.experiments.fixed_input_ablation as T
from AG3S.trajopt.types import ChunkLayout

COND = sys.argv[1]
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T37'))
PORT = {"E3a": 8204, "E3b": 8206}[COND]
m, d, probe, obj, _ = P.build()
A = json.loads((V / "p" / f"{COND}_analysis.json").read_text())["runs"]
out = {"condition": COND, "doc": __doc__, "runs": {}}
rob = None


def lit(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except Exception:
        return x


for run in sorted(A):
    rec = V / f"{COND}_rec"
    if rob is None:
        args = T.build_parser().parse_args(["--constraints", str(rec / f"server_{PORT}/server_constraints/run_0000"),
                                            "--frames", str(rec / run / "frames.jsonl"), "--server-log", str(rec / f"serve_safe_{PORT}.log"), "--out", "/dev/null"])
        settings = T.resolve_settings(args, T.parse_server_log(args.server_log), json.loads((rec / run / "manifest.json").read_text()))
        rob = T.build_robot(settings)
    rb, qadr, _ = rob
    lay = ChunkLayout.rby1(rb.joint_names)
    rows = {int(x["seq"]): x for x in (json.loads(l) for l in (rec / run / "frames.jsonl").read_text().splitlines() if l.strip())
            if x.get("kind") == "planning"}
    res = []
    for c in A[run]["chunks"]:
        if not c.get("server_chunk"):
            continue
        z = np.load(rec / c["server_chunk"], allow_pickle=False)
        s = json.loads(str(z["summary_json"]))
        g = s.get("grasp") or {}
        held = ((g.get("attached") or {}).get("held"))
        if not held or not g.get("attached_in_constraints", True):
            continue
        link = held["frame_link"]
        loc = [(np.asarray(x["centre_m"], float), x["radius_mm"] / 1000) for x in held["spheres_parent_frame"]]
        locf = [(np.asarray(x["centre_m"], float), x["radius_mm"] / 1000) for x in held.get("filter_spheres_parent_frame") or []]
        row = rows[int(c["seq"])]
        field = T.build_field(z, outside_distance=1.0)
        q_now = T.q_now_from_qpos(row["qpos"], qadr)
        pts = np.asarray(z["raw_cloud"], float)[np.asarray(z["support_mask"], bool)]
        tab = pts[pts[:, 2] > 0.5]
        c0 = tab.mean(0); n = np.linalg.svd(tab - c0, full_matrices=False)[2][-1]; n = n if n[2] > 0 else -n; off = float(n @ c0)
        e = {"t": c["t"], "seq": c["seq"], "state": g.get("state"), "attach": g.get("attach"), "detach": g.get("detach"),
             "executed": c["n_executed"] == c["n_control"]}
        for traj_name in ("refined", "reference"):
            traj = lay.chunk_to_trajectory(np.asarray(z[f"{traj_name}_chunk"], float))[:, :8]
            Q = [q_now] + list(lay.full_q(traj, q_now).T)
            for sname, L in (("query", loc), ("filter", locf)):
                if not L:
                    continue
                best = (np.inf, None, None, None)
                for qi, q in enumerate(Q):
                    Tq = rb.link_pose(q, link)
                    cs = np.array([Tq[:3, :3] @ cc + Tq[:3, 3] for cc, _ in L])
                    v = np.asarray(field.distance(cs), float) - np.array([r for _, r in L])
                    k = int(np.argmin(v))
                    if v[k] < best[0]:
                        best = (float(v[k]), qi, k, cs[k])
                e[f"{traj_name}_{sname}_min_mm"] = round(1000 * best[0], 2)
                if traj_name == "refined" and sname == "query":
                    cc = best[3]; r_ = L[best[2]][1]
                    dc = float(np.asarray(field.distance(cc[None]), float)[0]); gv = np.asarray(field.gradient(cc[None]), float)[0]
                    gn = float(np.linalg.norm(gv))
                    P.set_q(m, d, row["qpos"])
                    _, _, per_c = P.nearest(m, d, probe, obj, P.base_to_world(m, d, cc))
                    w = {"sphere": best[2], "row_index": best[1], "centre_gt_gap_mm": {b: round(1000 * (v_ - r_), 2) for b, v_ in per_c.items() if b in ("table", "crate", "apple")},
                         "centre_height_above_plane_mm": round(1000 * float(n @ cc - off), 2)}
                    if gn > 1e-9:
                        p = cc - dc * gv / gn
                        _, _, per = P.nearest(m, d, probe, obj, P.base_to_world(m, d, p))
                        nb = min(per, key=per.get)
                        w.update({"surface_gt_nearest_body": nb, "surface_gt_nearest_mm": round(1000 * per[nb], 2),
                                  "surface_apple_mm": round(1000 * per.get("apple", np.inf), 2), "surface_table_mm": round(1000 * per.get("table", np.inf), 2),
                                  "surface_height_above_plane_mm": round(1000 * float(n @ p - off), 2)})
                    e["worst_refined_query"] = w
        mv = lit(((s.get("to") or {}).get("metrics") or {}).get("max_violation_pair"))
        if isinstance(mv, dict) and str(mv.get("link", "")).startswith("attached"):
            e["recorded_max_violation_pair"] = {k: mv.get(k) for k in ("link", "tier", "obstacle", "clearance_m", "step", "query")}
        res.append(e)
    out["runs"][run] = res
    print(run, len(res), [(x["t"], x.get("refined_query_min_mm"), (x.get("worst_refined_query") or {}).get("surface_gt_nearest_body")) for x in res][:40], flush=True)
    (V / "p" / f"{COND}_held_rows.json").write_text(json.dumps(out, indent=1, default=float))
