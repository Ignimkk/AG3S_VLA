"""SUBTASK-extra M2 — per-episode accuracy + 3x3 confusion of the kv_L4 probe, mode=normal, set=closed414
(and closed246 for reference), read from outputs/verify/SUBTASK-b/predictions.csv. Rows = true label, cols = pred.
Label index 0/1/2 = pick/place/home (p_pick, p_place, p_home column order)."""
import json, pandas as pd, numpy as np
d = pd.read_csv("outputs/verify/SUBTASK-b/predictions.csv")
out = {}
for st in ["closed414", "closed246"]:
    q = d[(d.set == st) & (d.feature == "kv_L4") & (d["mode"] == "normal")]
    res = {}
    for ep, g in q.groupby("episode"):
        cm = np.zeros((3, 3), int)
        for a, b in zip(g.label, g.pred): cm[a, b] += 1
        res[ep] = dict(n=len(g), correct=int((g.label == g.pred).sum()),
                       accuracy=round(float((g.label == g.pred).mean()), 4),
                       confusion_true_rows_pred_cols=cm.tolist(),
                       label_counts=[int(x) for x in cm.sum(1)])
    cm = np.zeros((3, 3), int)
    for a, b in zip(q.label, q.pred): cm[a, b] += 1
    res["_all"] = dict(n=len(q), correct=int((q.label == q.pred).sum()),
                       accuracy=round(float((q.label == q.pred).mean()), 4), confusion_true_rows_pred_cols=cm.tolist())
    out[st] = res
json.dump(out, open("outputs/verify/SUBTASK-extra/m2_probe_per_run.json", "w"), indent=1)
print(json.dumps(out, indent=1))
