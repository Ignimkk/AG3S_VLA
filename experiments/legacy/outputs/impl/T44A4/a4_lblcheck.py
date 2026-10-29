from common.workspace import workspace_path as _ag3s_workspace_path
import dataclasses, json, pathlib, sys
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.stages.target_grounding import ground_target
from AG3S.ag3s.types import AttentionPointCloud, PointCloud
d = json.load(open(_ag3s_workspace_path('outputs/verify/T44/A4/smoke_ground_target.json')))
RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/merge/raw_vla'))
base = AG3SConfig()
for r in d["rows"]:
    if r["labels_match_dump"]: continue
    rec = json.loads((RAW / f"{r['key']}.json").read_text()); z = np.load(RAW / f"{r['key']}.npz")
    apc = AttentionPointCloud(PointCloud(z["points"].astype(np.float64)), z["attention"], z["raw_attention"])
    cfg = dataclasses.replace(base.clustering, gripper_max_opening=rec["config"]["max_opening_m"])
    g = ground_target(apc, cfg, seed_percentile=95.0, exclude_mask=np.asarray(z["support"], bool), min_radius=base.geometry.min_radius)
    mine = sorted(tuple(np.sort(c.point_indices)) for c in g.clusters)
    lab = z["cluster_label"]
    theirs = sorted(tuple(np.nonzero(lab == k)[0]) for k in range(lab.max() + 1))
    print(r["key"], "same partition:", mine == theirs, "n mine/theirs", len(mine), len(theirs),
          "chosen dump", rec["chosen_cluster_id"], "mine", None if g.target is None else g.target.id,
          [ (c["n"], c["split"] and c["split"].get("role")) for c in rec["clusters"]], [(c.point_count, c.split and c.split.get("role")) for c in g.clusters])
