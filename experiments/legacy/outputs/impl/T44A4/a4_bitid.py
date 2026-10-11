"""default config: HEAD (attention_split, target_grounding) vs working tree, 64 VLA dumps.
usage: a4_bitid.py head|new OUT.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import dataclasses, importlib.util, json, pathlib, sys
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
S = pathlib.Path(__file__).parent
if sys.argv[1] == "head":
    import AG3S.ag3s.stages  # package
    for name in ("attention_split", "target_grounding"):
        full = f"AG3S.ag3s.stages.{name}"
        spec = importlib.util.spec_from_file_location(full, S / f"head_{name}.py")
        mod = importlib.util.module_from_spec(spec); sys.modules[full] = mod; spec.loader.exec_module(mod)
from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.stages import target_grounding as TG
from AG3S.ag3s.types import AttentionPointCloud, PointCloud
print(TG.__file__)
RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/merge/raw_vla'))
base = AG3SConfig()
out = {}
for fj in sorted(RAW.glob("demo_*.json")):
    rec = json.loads(fj.read_text()); z = np.load(fj.with_suffix(".npz"))
    apc = AttentionPointCloud(PointCloud(z["points"].astype(np.float64)), z["attention"], z["raw_attention"])
    mo = rec["config"]["max_opening_m"]
    conf = TG.TargetConfirm(base.clustering, max_opening=mo, overlap_distance=0.005)
    g = TG.ground_target(apc, base.clustering, seed_percentile=95.0, exclude_mask=np.asarray(z["support"], bool),
                         min_radius=base.geometry.min_radius, timestamp=0.0, confirm=conf)
    out[rec["key"]] = {"clusters": [[c.id, c.point_indices.tolist(), c.target_score, c.split] for c in g.clusters],
                       "target": None if g.target is None else [g.target.id, g.target.point_indices.tolist(),
                                                                {k: v for k, v in g.target.metrics.items()}],
                       "status": g.status.value, "split_record": conf._split,
                       "splits": [d.record() for d in g.splits]}
pathlib.Path(sys.argv[2]).write_text(json.dumps(out, sort_keys=True))
