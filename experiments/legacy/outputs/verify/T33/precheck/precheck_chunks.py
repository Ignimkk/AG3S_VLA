"""T33 precheck — first N chunk records of the first run (+ local observation rows) for the settings the task lists.
usage: precheck_chunks.py COND PORT RUN N OUT.json"""
import json, sys, pathlib
import numpy as np
COND, PORT, RUN, N, OUTP = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
REC = pathlib.Path(f"${AG3S_PROJECT_ROOT}/outputs/verify/T33/{COND}_rec")
rows = [json.loads(l) for l in (REC / RUN / "frames.jsonl").read_text().splitlines() if l.strip()]
obs = [r for r in rows if r["kind"] == "observation"]
out = {"run": RUN, "chunks": [], "observations": [{"t": o.get("t_step"), "stamp_mode": o.get("stamp_mode"), "camera_skew_sec": o.get("camera_skew_sec"),
        "render_spread_sec": o.get("render_spread_sec")} for o in obs[:N]]}
for k in range(N):
    f = REC / f"server_{PORT}/server_constraints/run_0000/chunk_{k:05d}.npz"
    if not f.exists():
        break
    z = np.load(f, allow_pickle=False); s = json.loads(str(z["summary_json"]))
    es = s.get("esdf_stats") or {}; tm = (s.get("to") or {}).get("metrics") or {}; man = s.get("manipulated") or {}
    g = s.get("grasp") or {}
    out["chunks"].append({
        "chunk": k, "t": int(z["t_step"]),
        "tsdf_truncation": es.get("tsdf_truncation"), "held_free": es.get("held_free"),
        "n_attached_points": es.get("n_attached_points"), "n_attached_seeds_excluded": es.get("n_attached_seeds_excluded"),
        "n_constraint_spheres": int(z["sphere_radii"].shape[0]), "constraint_sphere_links": sorted(set(s.get("sphere_link_names") or [])),
        "constraint_radius_mm_minmax": [float(1000 * z["sphere_radii"].min()), float(1000 * z["sphere_radii"].max())],
        "finger_joints": {"source": (s.get("finger_joints") or {}).get("source"), "opening_norm": (s.get("finger_joints") or {}).get("opening_norm"),
                          "applied_keys": sorted(((s.get("finger_joints") or {}).get("applied") or {}).keys())},
        "exclusion": s.get("exclusion"), "grasp_hold_link": g.get("hold_link"), "grasp_parent_link": g.get("parent_link"),
        "grasp_attached": g.get("attached"), "settle_epsilon": (g.get("evidence") or {}).get("settle_epsilon"),
        "manipulated_keys": sorted(man.keys()), "attention_split_present": "split" in man, "hand_present": "hand" in man,
        "split": man.get("split"), "hand": man.get("hand"),
        "to_min_iterations": tm.get("min_iterations"), "to_max_iterations": tm.get("max_iterations"), "to_sqp_iterations": tm.get("sqp_iterations"),
        "to_returned": tm.get("returned"), "to_projection": tm.get("projection"), "to_time_budget_ms": tm.get("time_budget_ms"),
        "esdf_layers": s.get("esdf_layers") if not isinstance(s.get("esdf_layers"), dict) else list(s["esdf_layers"].keys()),
        "notes_points": [n for n in (s.get("notes") or []) if n.startswith("points:")],
    })
pathlib.Path(OUTP).write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, default=str)[:6000])
