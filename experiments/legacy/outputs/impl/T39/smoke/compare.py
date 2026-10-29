"""T39 S1 smoke — 같은 seed 비트 동일 / 다른 seed 다름 / seed 0 = 새 서버 seed 없음."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
D = pathlib.Path(_ag3s_workspace_path('outputs/impl/T39/smoke')) / sys.argv[1]
runs = ["r1_noseed", "r2_seed0", "r3_s18071", "r4_s18071", "r5_s18072", "r6_noseed"]
data = {}
for r in runs:
    z = np.load(D / r / "traj.npz")
    rows = [json.loads(l) for l in open(D / r / "frames.jsonl")]
    plan = [x for x in rows if x["kind"] == "planning"]
    ref = [x.get("actions_reference") for x in plan]
    man = json.load(open(D / r / "manifest.json"))
    data[r] = {"pred": z["predicted_chunks"], "exec": z["executed_actions"], "qpos": z["measured_qpos"],
               "ref": None if ref[0] is None else np.asarray(ref, np.float64),
               "seed_npz": int(z["policy_seed"]) if "policy_seed" in z else None,
               "seed_man": man.get("policy_seed"), "warm": man["timing"].get("warmup_steps"),
               "n": len(plan), "ctrl0": [x for x in rows if x["kind"] == "control"][0]["qpos"]}
def same(a, b, k):
    x, y = data[a][k], data[b][k]
    if x is None or y is None: return None
    return bool(x.shape == y.shape and np.array_equal(x, y))
def maxdiff(a, b, k):
    x, y = data[a][k], data[b][k]
    return float(np.max(np.abs(x - y))) if x is not None and y is not None else None
def chunks_differ(a, b, k="pred"):
    x, y = data[a][k], data[b][k]
    return int(sum(not np.array_equal(p, q) for p, q in zip(x, y)))
print("run  seed(manifest) seed(npz) warmup n_chunks")
for r in runs: d = data[r]; print(r, d["seed_man"], d["seed_npz"], d["warm"], d["n"], d["pred"].shape)
print("t=0 control qpos identical across all runs:", all(data[r]["ctrl0"] == data[runs[0]]["ctrl0"] for r in runs))
pairs = [("r3_s18071", "r4_s18071", "same seed -> identical"),
         ("r1_noseed", "r2_seed0", "fresh no-seed vs seed 0 -> identical"),
         ("r3_s18071", "r5_s18072", "seed s vs s' -> differ"),
         ("r1_noseed", "r6_noseed", "no seed on used server -> differs (old: request order)")]
for a, b, what in pairs:
    print(f"{what:55s} pred_eq={same(a,b,'pred')} exec_eq={same(a,b,'exec')} qpos_eq={same(a,b,'qpos')} ref_eq={same(a,b,'ref')} "
          f"max|dpred|={maxdiff(a,b,'pred'):.3e} chunks_differ={chunks_differ(a,b)}/{len(data[a]['pred'])}")
