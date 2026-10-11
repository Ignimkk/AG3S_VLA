"""Count points answered by target_free_coarse whose composed gradient != coarse target-free layer gradient (> 1e-6)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import numpy as np, glob, json
out = {}
for m in ("ball", "bo_d3", "oracle"):
    tot = bad = 0; ex = []
    for f in sorted(glob.glob(_ag3s_workspace_path('outputs/verify/T44/E2c/raw_*/*.npz'))):
        z = np.load(f)
        for I in ("gt", "gt_label"):
            k = f"{I}__gc_{m}_grad_vs_layer_absdiff"
            if k not in z.files:
                continue
            d = z[k]; b = d > 1e-6; tot += len(d); bad += int(b.sum())
            if b.any():
                ex.append({"npz": f.split("/")[-1], "identity": I, "n": int(b.sum()), "max_absdiff": float(d.max()),
                           "value_m": z[f"{I}__gc_{m}_value"][b].tolist()})
    out[m] = {"n_points": tot, "n_mismatch": bad, "examples": ex}
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T44/E2c/grad_mismatch.json'), "w"), indent=1)
print({m: (v["n_points"], v["n_mismatch"]) for m, v in out.items()})
