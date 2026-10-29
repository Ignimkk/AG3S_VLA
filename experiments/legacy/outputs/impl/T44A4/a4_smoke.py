"""T44b A4 smoke: reproduce the D-merge offline numbers through the pipeline's grounding call
(`ground_target` -> `_split_groups` -> `split_by_attention`) on the same VLA-attention dumps.

Input: outputs/verify/T44/merge/raw_vla/*.{json,npz} (fused cloud, normalised + raw attention, support
mask, per-point GT body id) — the cloud/attention the AG3S pipeline handed to `ground_target` in D-merge.
Variants: (radius, ratio source) in {15, 12} x {normalized, raw}. Default must reproduce the dumped
cluster labels exactly (sanity: same code path). Metric (T44M definitions): parent = pre-split cluster
holding most GT-target non-support points; segment = split 'target' component (else parent);
pure_and_complete = precision >= 0.95 and recall >= 0.90 vs the parent's GT-target points.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import dataclasses, json, pathlib, sys
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.stages.target_grounding import ground_target
from AG3S.ag3s.types import AttentionPointCloud, PointCloud

RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/merge/raw_vla'))
E2RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2/raw_demo'))
FRUITS = ("apple", "banana", "orange", "pear")
VARIANTS = {"r15_normalized": (0.015, "normalized"), "r12_normalized": (0.012, "normalized"),
            "r15_raw": (0.015, "raw"), "r12_raw": (0.012, "raw")}
base = AG3SConfig()
rows = []
for fj in sorted(RAW.glob("demo_*.json")):
    rec = json.loads(fj.read_text()); z = np.load(fj.with_suffix(".npz"))
    key, target = rec["key"], rec["target"]
    names = {int(k): v for k, v in json.loads(str(z["body_names"])).items()}
    nm = np.array([names.get(int(b), "none") for b in z["body_id"]])
    sup = np.asarray(z["support"], bool)
    is_t = (nm == target) & ~sup
    e2 = json.loads((E2RAW / f"{key}.json").read_text())["identities"]["gt"]["merge"]["target_cluster"]
    merged = bool(e2["target_fraction"] < 0.8 and e2["distractor_fraction"] > 0)
    apc = AttentionPointCloud(PointCloud(z["points"].astype(np.float64)), z["attention"], z["raw_attention"])
    assert np.array_equal(apc.peak_signal, z["peak_signal"])
    row = {"key": key, "target": target, "merged": merged}
    for vname, (radius, src) in VARIANTS.items():
        cfg = dataclasses.replace(base.clustering, gripper_max_opening=rec["config"]["max_opening_m"],
                                  attention_split_radius=radius, attention_split_ratio_source=src)
        g = ground_target(apc, cfg, seed_percentile=base.attention.seed_percentile,
                          seed_threshold=base.attention.seed_threshold, exclude_mask=sup,
                          min_radius=base.geometry.min_radius)
        cls = list(g.clusters)
        lab = np.full(len(apc), -1, np.int64)
        for ci, c in enumerate(cls):
            lab[c.point_indices] = ci
        if vname == "r15_normalized":
            # same order as the dump? (dump labels = the pipeline's clusters in their sorted order)
            row["labels_match_dump"] = bool(np.array_equal(lab, z["cluster_label"]))
            row["reasons_match_dump"] = [d.reason for d in g.splits] == [
                d["reason"] for d in (rec["split_record"] or {}).get("decisions", [])]
        # parents: split siblings are re-joined
        groups = {}
        for ci, c in enumerate(cls):
            sp = c.split
            gk = ("s", sp["split_index"]) if sp and sp.get("role") in ("target", "obstacle") else ("c", ci)
            groups.setdefault(gk, []).append(ci)
        best = max(groups.values(), key=lambda m: int(is_t[np.isin(lab, m)].sum()))
        pm = np.isin(lab, best)
        n_t = int(is_t[pm].sum())
        if len(best) > 1:
            seg_c = next(ci for ci in best if cls[ci].split["role"] == "target")
            seg = lab == seg_c; taken = True
        else:
            seg = pm; taken = False
        sp0 = cls[best[0]].split
        tp = int((seg & is_t).sum())
        prec, recl = tp / max(1, int(seg.sum())), tp / max(1, n_t)
        chosen_is_seg = g.target is not None and np.array_equal(np.sort(g.target.point_indices), np.nonzero(seg)[0])
        row[vname] = {"split_taken": taken, "reason": None if sp0 is None else sp0.get("reason", "split"),
                      "precision": round(prec, 4), "recall": round(recl, 4),
                      "pure_and_complete": bool(prec >= 0.95 and recl >= 0.90),
                      "wrong_pick": bool(taken and prec < 0.5), "chosen_is_segment": bool(chosen_is_seg)}
    rows.append(row)
    print(key, merged, row.get("labels_match_dump"), {v: row[v]["pure_and_complete"] for v in VARIANTS}, flush=True)

summ = {"n_frames": len(rows), "n_merged": sum(r["merged"] for r in rows),
        "labels_match_dump": sum(r["labels_match_dump"] for r in rows),
        "reasons_match_dump": sum(r["reasons_match_dump"] for r in rows)}
for v in VARIANTS:
    m = [r[v] for r in rows if r["merged"]]; nm_ = [r[v] for r in rows if not r["merged"]]
    summ[v] = {"merged_pure_and_complete": sum(x["pure_and_complete"] for x in m),
               "merged_split_taken": sum(x["split_taken"] for x in m),
               "merged_wrong_pick": sum(x["wrong_pick"] for x in m),
               "nonmerged_pure_and_complete": sum(x["pure_and_complete"] for x in nm_),
               "nonmerged_complete": sum(x["recall"] >= 0.90 for x in nm_),
               "nonmerged_wrong_pick": sum(x["wrong_pick"] for x in nm_),
               "chosen_is_segment_merged": sum(x["chosen_is_segment"] for x in m)}
out = pathlib.Path(sys.argv[1])
out.write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
print(json.dumps(summ, indent=1))
