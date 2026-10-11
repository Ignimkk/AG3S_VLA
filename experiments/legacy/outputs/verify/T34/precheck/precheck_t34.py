"""T34 precheck — first N chunk records of RUN + the first attach chunk and the next HELD chunks: T33 precheck items +
T34 keys: grasp.attached.held.support {applied, plane, band, clearance_source, clearance_mm, max_lift_mm, spheres},
held.fit_spheres / filter_spheres_parent_frame / spheres_parent_frame, grasp.attach_revoked key, evidence.hold_window /
hold_checks on HELD chunks; LatchConfig revoke_* values read from the code (they are not written to the record except
hold_window.limit = revoke_chunks). usage: precheck_t34.py COND PORT RUN N OUT.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, dataclasses
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
COND, PORT, RUN, N, OUTP = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
REC = pathlib.Path(f"${AG3S_PROJECT_ROOT}/outputs/verify/T34/{COND}_rec")
files = sorted((REC / f"server_{PORT}/server_constraints").glob("run_*/chunk_*.npz"))
out = {"run": RUN, "n_chunk_files": len(files), "chunks": [], "attach": None, "held_after_attach": []}
try:
    from AG3S.trajopt.grasp_latch import LatchConfig
    out["latch_config_code_defaults"] = {k: v for k, v in dataclasses.asdict(LatchConfig()).items() if k.startswith("revoke")}
except Exception as e:
    out["latch_config_code_defaults"] = f"import failed: {e}"
att_i = None
for k, f in enumerate(files):
    z = np.load(f, allow_pickle=False); s = json.loads(str(z["summary_json"]))
    es = s.get("esdf_stats") or {}; tm = (s.get("to") or {}).get("metrics") or {}; g = s.get("grasp") or {}
    if k < N:
        out["chunks"].append({"chunk": k, "t_step": int(z["t_step"]), "tsdf_truncation": es.get("tsdf_truncation"),
            "n_constraint_spheres": int(z["sphere_radii"].shape[0]),
            "constraint_radius_mm_minmax": [float(1000 * z["sphere_radii"].min()), float(1000 * z["sphere_radii"].max())],
            "finger_joints_source": (s.get("finger_joints") or {}).get("source"),
            "grasp_keys": sorted(g.keys()), "attach_revoked_key_present": "attach_revoked" in g,
            "grasp_state": g.get("state"), "settle_epsilon": (g.get("evidence") or {}).get("settle_epsilon"),
            "to_min_iterations": tm.get("min_iterations"), "to_max_iterations": tm.get("max_iterations"),
            "exclusion": s.get("exclusion")})
    if g.get("attach") and att_i is None:
        att_i = k
        held = (g.get("attached") or {}).get("held") or {}
        out["attach"] = {"chunk": k, "t_step": int(z["t_step"]), "note": g.get("note"), "held_keys": sorted(held.keys()),
                         "support": held.get("support"), "fit_spheres": held.get("fit_spheres"), "spheres": held.get("spheres"),
                         "filter_spheres_parent_frame": held.get("filter_spheres_parent_frame"),
                         "spheres_parent_frame": held.get("spheres_parent_frame"), "fit": held.get("fit"),
                         "snapshot_state": held.get("snapshot_state"), "first_free_spheres": held.get("first_free_spheres"),
                         "attach_revoked": g.get("attach_revoked"), "evidence": g.get("evidence")}
    elif att_i is not None and k <= att_i + 6:
        ev = g.get("evidence") or {}
        out["held_after_attach"].append({"chunk": k, "t_step": int(z["t_step"]), "state": g.get("state"), "detach": g.get("detach"),
                                         "hold_window": ev.get("hold_window"), "hold_checks": ev.get("hold_checks"),
                                         "revoke": ev.get("revoke"), "attach_revoked": g.get("attach_revoked"), "note": g.get("note")})
pathlib.Path(OUTP).write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({"attach": out["attach"] and {k: out["attach"][k] for k in ("chunk", "t_step", "held_keys", "support")},
                  "held_after": out["held_after_attach"], "latch": out["latch_config_code_defaults"],
                  "first": out["chunks"][:2]}, default=str, ensure_ascii=False)[:8000])
