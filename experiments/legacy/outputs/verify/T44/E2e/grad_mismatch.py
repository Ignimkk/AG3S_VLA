"""T44e: points answered by target_free_coarse whose composed gradient != coarse target-free layer gradient (> 1e-6), by point set (demo + lb, gt_label).
'nonzero_value' = |value| > 1e-6 m. Pools are the capped (4000/frame) gradient-check samples of e2e_measure.py. -> grad_mismatch.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import glob, json, numpy as np
out = {}
recs = []
for d in ("raw_demo", "raw_lb"):
    for f in sorted(glob.glob(f"{_ag3s_workspace_path()}/outputs/verify/T44/E2e/{d}/*.json")):
        recs.append((json.load(open(f)), f[:-5] + ".npz"))
for m in ("ball", "bo_d3"):
    for v in ("guard", "keep", "exclude", "exclude_target"):
        by = {}; nz = 0; tot = 0
        for r, nf in recs:
            ir = r["identities"].get("gt_label", {})
            gk = ir.get("grad_check", {}).get(f"{m}|{v}")
            if not gk or not gk["n"]: continue
            z = np.load(nf); names = gk["set_names"]
            d = z[f"gt_label__gc_{m}_{v}_grad_vs_layer_absdiff"]; src = z[f"gt_label__gc_{m}_{v}_src"]; val = z[f"gt_label__gc_{m}_{v}_value"]
            bad = d > 1e-6
            for i, nm in enumerate(names):
                s = src == i
                e = by.setdefault(nm, {"n": 0, "mismatch": 0})
                e["n"] += int(s.sum()); e["mismatch"] += int((bad & s).sum())
            nz += int((bad & (np.abs(val) > 1e-6)).sum()); tot += len(d)
        out[f"{m}|{v}"] = {"n_points": tot, "by_set": by, "mismatch_total": int(sum(e["mismatch"] for e in by.values())), "mismatch_with_nonzero_value": nz}
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T44/E2e/grad_mismatch.json'), "w"), indent=1)
for k, x in out.items(): print(k, x["n_points"], x["mismatch_total"], x["mismatch_with_nonzero_value"], {s: e["mismatch"] for s, e in x["by_set"].items()})
