"""Error location for the best feature: distance (in samples) from each misclassified frame to the nearest GT label change."""
import json, pathlib
import numpy as np, pandas as pd
OUT = pathlib.Path(__file__).parent
R = json.load(open(OUT / "probe_results.json"))
P = pd.read_csv(OUT / "predictions.csv")
res = {}
for f in [R["best_feature"], "attn_L8", "prefix_last"]:
    for s in ("val", "closed414"):
        for mode in ("normal", "mask_a", "mask_b"):
            d = P[(P.set == s) & (P.feature == f) & (P["mode"] == mode)]
            dist = []
            for ep, g in d.groupby("episode"):
                g = g.sort_values("t")
                lab = g.label.to_numpy(); err = (g.pred != g.label).to_numpy()
                ch = np.where(lab[1:] != lab[:-1])[0] + 0.5
                for i in np.where(err)[0]:
                    dist.append(float(np.min(np.abs(ch - i))) if len(ch) else 1e9)
            dist = np.asarray(dist)
            res[f"{f}|{s}|{mode}"] = dict(n_err=int(len(dist)), within_1=int((dist <= 1.5).sum()),
                                          within_3=int((dist <= 3.5).sum()),
                                          err_by_phase=d[d.pred != d.label].phase_index.value_counts().sort_index().to_dict() if s == "val" else None)
json.dump(res, open(OUT / "boundary_results.json", "w"), indent=1, default=int)
for k, v in res.items(): print(k, v)
