"""T44 V-E2 verifier: fix the E2-demo list BEFORE measuring (rule from T44.task.md 'E2 입력 결정').
train split (episode < 1600 AND meta split == 'train'), success true, nearest other fruit xy distance
(initial_fruit_poses) bins <0.08 / 0.08-0.10 m; per (target_fruit x used_arm) x bin take the smallest
episode index; frames = 4 evenly spaced over [pregrasp_start_frame, grasp_close_start_frame]."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, numpy as np, collections
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.live import exclusion_e2 as H
ds = H.DEMO_DATASET
meta = H.load_demo_meta(ds)
FR = ("apple", "orange", "pear", "banana"); ARMS = ("left", "right")
BINS = (("lt_0.08", 0.0, 0.08), ("0.08_0.10", 0.08, 0.10))
cand = collections.defaultdict(list); n_split_mismatch = 0; counts = collections.Counter()
for ep, m in sorted(meta.items()):
    sp = m.get("split")
    if (ep < 1600) != (sp == "train"):
        n_split_mismatch += 1
    if not (ep < 1600 and sp == "train"):
        counts["not_train"] += 1; continue
    if m.get("success") is not True:
        counts["not_success"] += 1; continue
    t = np.asarray(m["initial_fruit_poses"][m["target_fruit"]][:2])
    nn = min((float(np.linalg.norm(np.asarray(v[:2]) - t)), k) for k, v in m["initial_fruit_poses"].items() if k != m["target_fruit"])
    for b, lo, hi in BINS:
        if lo <= nn[0] < hi:
            cand[(m["target_fruit"], m["used_arm"], b)].append((ep, nn))
            counts[b] += 1
sel, posed, table = [], [], []
for f in FR:
    for a in ARMS:
        for b, _, _ in BINS:
            c = sorted(cand[(f, a, b)])
            row = {"target_fruit": f, "used_arm": a, "bin": b, "n_candidates": len(c)}
            if c:
                ep, (d, other) = c[0]
                m = meta[ep]
                st = H.demo_state16(ds, ep)
                frs = H.demo_frames(m, 4)
                row.update(episode=ep, nearest_other=other, nearest_other_xy_m=round(d, 5), frames=frs,
                           pregrasp_start_frame=m["events"]["pregrasp_start_frame"],
                           grasp_close_start_frame=m["events"]["grasp_close_start_frame"])
                for fr in frs:
                    pf = H.posed_frame_from_demo(m, st, fr); pf["bin"] = b
                    posed.append(pf)
            else:
                row["episode"] = None
            table.append(row)
out = {"definitions": __doc__, "date": "2026-10-10", "dataset": str(ds),
       "split_rule": "episode_index < 1600 and meta split == 'train'",
       "n_split_rule_mismatch_(ep<1600 vs meta split)": n_split_mismatch,
       "pool_counts": dict(counts), "groups": table,
       "n_episodes": sum(r["episode"] is not None for r in table), "n_frames": len(posed)}
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T44/E2/e2_demo_frames.json'), "w"), indent=1)
json.dump(posed, open(_ag3s_workspace_path('outputs/verify/T44/E2/e2_demo_posed_frames.json'), "w"), indent=1)
for r in table: print(r["target_fruit"], r["used_arm"], r["bin"], r["n_candidates"], r.get("episode"), r.get("nearest_other"), r.get("nearest_other_xy_m"), r.get("frames"))
print(out["pool_counts"], out["n_episodes"], out["n_frames"], n_split_mismatch)
