"""T42-2 invisible-obstacle diagnostic, run part (verifier, CPU). usage: invis_diag.py OUT_JSON
Runs: runs/{E0,E3b}/invis/* (profile t42_2_ep<EP>_place_invis = the place-phase placement with visual rgba [0,0,0,0]) and
runs/{E0,E3b}/clear42c/* (no obstacle flags, snapshot work-t42c). Seed rep 1.
Per run, vs three references of the same condition and seed: T42-1 no-obstacle twin (snapshot work-t42), clear42c (ep1834 only),
and the visible place run (runs/<COND>/place):
  first_step_diff_gt0 / gt_1e-3 : first control step where max_j |executed_actions diff| > 0 / > 1e-3 (traj.npz)
  first_chunk_diff_max_abs      : max |predicted_chunks[0] diff| (the policy's first chunk; E3b frames 'actions_reference' at planning
                                  seq 1 is reported as well when present)
  qpos_t0_equal                 : planning row 0 qpos bitwise equal
Outcome: analyze_t42 success / grasp / place, obstacle stop / phase / contact pairs (obst_metrics.run_obst on the invisible profile).
AG3S sees the obstacle?: E3b server chunk records paired by content; per chunk the number of raw_cloud points (robot base frame -> world
at that planning row's qpos) within 15 mm of the obstacle surface (analytic shapes of the placed slot), for the invisible run and for
the visible place run, and (baseline: other surfaces within 15 mm of the obstacle's volume) for the T42-1 no-obstacle twin with the
same obstacle placed virtually; reported as max over chunks and at chunk 1.
Offline part: raw/invis/probe_<EP>.json (render_probe.py)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/2/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/p'))
import analyze_t42 as AN, obst_metrics as OM, analyze_t42_2 as A2, plib as L  # noqa: E402
R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T42/2/runs'))


def cmp(a, b):
    za, zb = np.load(pathlib.Path(a) / "traj.npz"), np.load(pathlib.Path(b) / "traj.npz")
    ea, eb = za["executed_actions"], zb["executed_actions"]; n = min(len(ea), len(eb)); dd = np.abs(ea[:n] - eb[:n]).max(axis=1)
    pa = [json.loads(l) for l in (pathlib.Path(a) / "frames.jsonl").read_text().splitlines() if l.strip()]
    pb = [json.loads(l) for l in (pathlib.Path(b) / "frames.jsonl").read_text().splitlines() if l.strip()]
    p1a = next(x for x in pa if x["kind"] == "planning"); p1b = next(x for x in pb if x["kind"] == "planning")
    ar = None
    if p1a.get("actions_reference") is not None and p1b.get("actions_reference") is not None:
        ar = float(np.abs(np.asarray(p1a["actions_reference"], float) - np.asarray(p1b["actions_reference"], float)).max())
    return {"other": str(b), "n_compared": int(n), "first_step_diff_gt0": int(np.argmax(dd > 0)) if (dd > 0).any() else None,
            "first_step_diff_gt_1e-3": int(np.argmax(dd > 1e-3)) if (dd > 1e-3).any() else None,
            "first_chunk_diff_max_abs": float(np.abs(za["predicted_chunks"][0] - zb["predicted_chunks"][0]).max()),
            "actions_reference_seq1_diff_max_abs": ar, "qpos_t0_equal": bool(np.array_equal(np.asarray(p1a["qpos"]), np.asarray(p1b["qpos"])))}


def cloud_hits(rd, cfg, prof, segs):
    hits = A2.pair(segs, pathlib.Path(rd))
    if len(hits) != 1:
        return {"n_segments_matching": len(hits)}
    mg = OM.manager(cfg, prof); sh = OM.shapes(mg)
    plan = {p["seq"]: p for p in (json.loads(l) for l in (pathlib.Path(rd) / "frames.jsonl").read_text().splitlines() if l.strip()) if p["kind"] == "planning"}
    per = []
    for f in hits[0]:
        z = np.load(f, allow_pickle=True); seq = int(z["t_step"]); p = plan.get(seq)
        if p is None:
            continue
        q = np.array(p["qpos"]); C = np.asarray(z["raw_cloud"], float)
        M = OM.model(); m, d = M["m"], M["d"]; d.qpos[:] = q; import mujoco; mujoco.mj_kinematics(m, d); b = M["B"]("base")
        W = d.xpos[b] + C @ d.xmat[b].reshape(3, 3).T
        # cheap prefilter: points within 0.4 m of any obstacle geom centre
        cen = np.array([s[1] for s in sh]); near = np.min(np.linalg.norm(W[:, None, :] - cen[None], axis=2), axis=1) < 0.4
        n = int(sum(OM.point_to_obst(x, sh) <= 0.015 for x in W[near]))
        per.append({"seq": seq, "t": p["t_step"], "n_points": int(len(W)), "n_within_15mm_of_obstacle": n})
    return {"n_segments_matching": 1, "chunk1": per[0] if per else None, "max_over_chunks": max((e["n_within_15mm_of_obstacle"] for e in per), default=None),
            "n_chunks": len(per), "n_chunks_with_points_on_obstacle": sum(1 for e in per if e["n_within_15mm_of_obstacle"] > 0)}


out = {"definitions": __doc__, "runs": {}, "offline_probe": {}}
for ep in (1830, 983, 1215, 1834):
    pr = pathlib.Path(f"${AG3S_PROJECT_ROOT}/outputs/verify/T42/2/raw/invis/probe_{ep}.json")
    if pr.exists():
        out["offline_probe"][ep] = json.loads(pr.read_text())
segs = {c: A2.segments(R / c) for c in ("E3b",)}
segs_twin = A2.segments(pathlib.Path(_ag3s_workspace_path('outputs/verify/T42/1/runs/E3b'))) + A2.segments(pathlib.Path(_ag3s_workspace_path('outputs/verify/T39/runs/E3b')))
for c in ("E0", "E3b"):
    for kind in ("invis", "clear42c"):
        kd = R / c / kind
        if not kd.exists():
            continue
        for rd in sorted(p for p in kd.iterdir() if p.is_dir()):
            if not (kd / f"{rd.name}.done").exists():
                continue
            ep, seed = int(rd.name.split("_ep")[1].split("_")[0]), int(rd.name.split("_s")[-1])
            a = AN.analyze(rd); ob = A2.obstacle_of(kd / f"{rd.name}.start")
            r = {"cond": c, "kind": kind, "episode": ep, "seed": seed, "success": a["success"], "grasp": a["grasp"], "place": a["place"],
                 "grasp_t": a["grasp_t"], "place_t": a["place_t"], "n_control": a["n_control"], "holds": A2.holds(rd)}
            r["vs_T42_1_no_obstacle"] = cmp(rd, L.run_dir(c, ep, seed))
            vis = R / c / "place" / rd.name
            if vis.exists():
                r["vs_visible_place_run"] = cmp(rd, vis)
            cl = R / c / "clear42c" / rd.name
            if kind == "invis" and cl.exists() and (R / c / "clear42c" / f"{rd.name}.done").exists():
                r["vs_clear42c"] = cmp(rd, cl)
            if ob:
                o = OM.run_obst(rd, ob[0], ob[1])
                r.update(obstacle_stop=o["obstacle_stop"], stop_t=o["stop_t"], stop_phase=o["stop_phase"], lift_t=o["lift_t"],
                         client_contact={k: (o.get("client_obstacle_safety") or {}).get(k) for k in ("robot_collision", "payload_collision", "contact_steps", "contact_pairs")})
                if c == "E3b":
                    r["ag3s_raw_cloud_on_obstacle_invisible"] = cloud_hits(rd, ob[0], ob[1], segs["E3b"])
                    tw = L.run_dir(c, ep, seed)
                    r["ag3s_raw_cloud_on_obstacle_no_obstacle_twin_virtual"] = cloud_hits(tw, ob[0], ob[1], segs_twin)
                    vp = A2.obstacle_of(R / c / "place" / f"{rd.name}.start")
                    if vis.exists() and vp:
                        r["ag3s_raw_cloud_on_obstacle_visible_place_run"] = cloud_hits(vis, vp[0], vp[1], segs["E3b"])
            out["runs"][f"{c}/{kind}/{rd.name}"] = r
            print(c, kind, rd.name, "S" if r["success"] else "-", r["grasp"], "div T42-1", r["vs_T42_1_no_obstacle"]["first_step_diff_gt_1e-3"],
                  "chunk0", round(r["vs_T42_1_no_obstacle"]["first_chunk_diff_max_abs"], 5), "vs clear42c", (r.get("vs_clear42c") or {}).get("first_step_diff_gt_1e-3"),
                  "stop", r.get("stop_phase"), "cloud twin/invis/vis", (r.get("ag3s_raw_cloud_on_obstacle_no_obstacle_twin_virtual") or {}).get("max_over_chunks"), (r.get("ag3s_raw_cloud_on_obstacle_invisible") or {}).get("max_over_chunks"),
                  (r.get("ag3s_raw_cloud_on_obstacle_visible_place_run") or {}).get("max_over_chunks"), flush=True)
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
