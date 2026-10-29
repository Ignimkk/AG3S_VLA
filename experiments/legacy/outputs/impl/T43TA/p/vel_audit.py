"""Velocity-row audit on recorded runs: anchor |Q0-q_now|, inner |dQ|, boundary |Q0-u[-1]| per joint / max_step."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, types
import numpy as np
from AG3S.trajopt.experiments.ta_offline import Run, lit
from AG3S.trajopt.experiments import fixed_input_ablation as FIA
T43 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
IDX = {"V8": T43 / "V8/analysis/analysis.json", "V7g": T43 / "CT/raw/analysis_v7g.json", "V5": T43 / "W/analysis/analysis.json"}
out = {}
run0 = None
for arm in sys.argv[1:]:
    idx = json.load(open(IDX[arm]))["runs"]
    rows = []
    for key, e in sorted(idx.items()):
        if not lit(e.get("server_segment")): continue
        if run0 is None:
            run0 = Run(key, e, types.SimpleNamespace(hm=False))
            step = np.asarray(run0.limits.max_step, float); print("max_step", step.round(4).tolist(), "dt", run0.base_cfg.horizon.dt, "vscale", run0.base_cfg.limits.velocity_scale, flush=True)
        seg = lit(e.get("server_segment"))
        if not seg: continue
        rd = pathlib.Path(e["run_dir"]); first = pathlib.Path(seg["first"]); i0 = int(first.stem.split("_")[1])
        fr = [json.loads(x) for x in open(rd / "frames.jsonl") if x.strip()]
        plan = {int(r["seq"]): r for r in fr if r["kind"] == "planning"}
        recs = {}
        for n in range(int(seg["n"])):
            p = first.parent / f"chunk_{i0 + n:05d}.npz"
            with np.load(p, allow_pickle=True) as z:
                s = json.loads(str(z["summary_json"]))
                recs[int(z["t_step"])] = (np.asarray(z["refined_chunk"], float), np.asarray(z["reference_chunk"], float), s)
        seqs = sorted(recs)
        for i, seq in enumerate(seqs):
            if seq not in plan: continue
            Rf, Rr, s = recs[seq]
            q_now = FIA.q_now_from_qpos(plan[seq]["qpos"], run0.qadr)[run0.layout.q_indices]
            Q = run0.layout.chunk_to_trajectory(Rf)[:, :8]; Qr = run0.layout.chunk_to_trajectory(Rr)[:, :8]
            fb = s.get("exec_feedback") or {}
            arm_ = fb.get("applied_arm"); arm_ = lit(arm_) if isinstance(arm_, str) else arm_
            u1 = np.asarray(arm_, float)[-1] if arm_ is not None and len(arm_) else None
            # executed rows of THIS chunk = next record's exec_feedback.executed
            nxt = recs[seqs[i + 1]][2].get("exec_feedback") if i + 1 < len(seqs) else None
            ex = list((nxt or {}).get("executed") or [])
            to = s.get("to") or {}
            r = {"key": key, "seq": seq, "status": to.get("status"), "returned": (to.get("metrics") or {}).get("returned"),
                 "anchor": float((np.abs(Q[:, 0] - q_now) / step).max()),
                 "inner": (np.abs(np.diff(Q, axis=1)) / step[:, None]).max(axis=0).tolist(),
                 "ref_anchor": float((np.abs(Qr[:, 0] - q_now) / step).max()),
                 "ref_inner_max": float((np.abs(np.diff(Qr, axis=1)) / step[:, None]).max()),
                 "boundary": None if u1 is None else float((np.abs(Q[:, 0] - u1) / step).max()),
                 "ref_boundary": None if u1 is None else float((np.abs(Qr[:, 0] - u1) / step).max()),
                 "lag": None if u1 is None else float((np.abs(u1 - q_now) / step).max()),
                 "executed": ex, "hold": bool(ex) and not all(ex)}
            rows.append(r)
    out[arm] = rows
    def summ(vals):
        v = np.asarray([x for x in vals if x is not None], float)
        return {"n": int(v.size), "median": round(float(np.median(v)), 3), "p99": round(float(np.quantile(v, .99)), 3), "max": round(float(v.max()), 3), "n_gt1": int((v > 1 + 1e-6).sum())}
    exe = [r for r in rows if r["executed"]]
    inner_exec = [r["inner"][k] for r in exe for k in range(min(7, len(r["executed"]) - 1)) if r["executed"][k + 1]]
    print(arm, "chunks", len(rows), "status", {s: sum(r["status"] == s for r in rows) for s in set(r["status"] for r in rows)})
    print("  anchor(refined)", summ([r["anchor"] for r in rows]))
    print("  inner(refined, executed rows)", summ(inner_exec))
    print("  boundary(refined, |Q0-u[-1]|)", summ([r["boundary"] for r in exe]))
    print("  ref boundary", summ([r["ref_boundary"] for r in exe]), "ref inner", summ([r["ref_inner_max"] for r in rows]))
    print("  lag |u[-1]-q_now|", summ([r["lag"] for r in rows]))
    jj = [int(np.argmax(np.abs(np.diff(run0.layout.chunk_to_trajectory(np.zeros((50,16))),axis=1)).max(axis=1))) ] if False else None
    big = sorted([r for r in rows if max(r["inner"]) > 1 + 1e-6 or r["anchor"] > 1 + 1e-6], key=lambda r: -max(max(r["inner"]), r["anchor"]))[:8]
    for r in big: print("   ", r["key"], r["seq"], r["status"], r["returned"], "anchor %.2f inner %.2f refinner %.2f" % (r["anchor"], max(r["inner"]), r["ref_inner_max"]), "bnd", r["boundary"])
    bb = sorted([r for r in exe if (r["boundary"] or 0) > 1 + 1e-6], key=lambda r: -r["boundary"])[:8]
    for r in bb: print("   B", r["key"], r["seq"], r["status"], "bnd %.2f refbnd %.2f lag %.2f anchor %.2f" % (r["boundary"], r["ref_boundary"], r["lag"], r["anchor"]))
json.dump(out, open(pathlib.Path(__file__).with_name("vel_audit.json"), "w"))
