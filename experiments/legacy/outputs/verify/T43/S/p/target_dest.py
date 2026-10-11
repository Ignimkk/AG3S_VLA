"""T43 S (4) — target / destination handling per arm (E3b T42-1 runs, records only).
Per planning chunk before the first close command: summary has_target, manipulated.state, target_centroid (npz, base frame) ->
world with MuJoCo body 'base' at the planning qpos; error = |centroid - target fruit body position (planning object_poses)|;
destination centroid (summary.destination.centroid, base) vs crate body position. Also: chunks after attach (held) with the
destination registered.
usage: target_dest.py [N_WORKERS]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import measure_pick as MP
import mujoco


def base_T(q):
    M = MP.model(); m, d = M["m"], M["d"]; d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    b = M["B"]("base"); return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


def one(key):
    if not key.startswith("E3b/"): return key, None
    r = MP.RUNS[key]; name = key.split("/")[1]
    ix = MP.IDX.get(f"{_ag3s_workspace_path()}/outputs/verify/T42/1/runs/E3b/noobs/{name}")
    rd = pathlib.Path(r["run_dir"]); fruit = r["target_fruit"]
    plan = {p["seq"]: p for p in (json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()) if p["kind"] == "planning"}
    ct = (r.get("close_cmd_t") or [None])[0]
    pre = {"n": 0, "has_target": 0, "err_mm": [], "state": {}}; post = {"n": 0, "dest_registered": 0, "dest_err_mm": []}
    for s, ch in ix["chunks"].items():
        p = plan[int(s)]
        with np.load(ch["npz"]) as z:
            summ = json.loads(str(z["summary_json"])); tc = np.asarray(z["target_centroid"], float) if "target_centroid" in z.files else None
        p0, Rb = base_T(np.asarray(p["qpos"], float))
        if ct is None or p["t_step"] + 7 < ct:
            pre["n"] += 1; st = str((summ.get("manipulated") or {}).get("state")); pre["state"][st] = pre["state"].get(st, 0) + 1
            if summ.get("has_target") and tc is not None and np.all(np.isfinite(tc)):
                pre["has_target"] += 1
                pre["err_mm"].append(float(1e3 * np.linalg.norm(tc @ Rb.T + p0 - np.asarray(p["object_poses"][fruit]["pos"]))))
        if (summ.get("grasp") or {}).get("state") == "held":
            post["n"] += 1
            de = summ.get("destination") or {}
            if de.get("centroid") is not None:
                post["dest_registered"] += 1
                post["dest_err_mm"].append(float(1e3 * np.linalg.norm(np.asarray(de["centroid"]) @ Rb.T + p0 - np.asarray(p["object_poses"]["crate"]["pos"]))))
    return key, {"arm": r["used_arm"], "fruit": fruit, "grasp": r["grasp"], "pre_close": pre, "held": post}


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with Pool(nw) as pool:
        res = {k: v for k, v in pool.map(one, sorted(MP.RUNS), chunksize=1) if v}
    agg = {}
    for v in res.values():
        a = agg.setdefault(v["arm"], {"runs": 0, "pre_chunks": 0, "pre_has_target": 0, "err": [], "held_chunks": 0, "held_dest": 0, "dest_err": []})
        a["runs"] += 1; a["pre_chunks"] += v["pre_close"]["n"]; a["pre_has_target"] += v["pre_close"]["has_target"]; a["err"] += v["pre_close"]["err_mm"]
        a["held_chunks"] += v["held"]["n"]; a["held_dest"] += v["held"]["dest_registered"]; a["dest_err"] += v["held"]["dest_err_mm"]
    for a in agg.values():
        for k in ("err", "dest_err"):
            x = np.asarray(a.pop(k)); a[k + "_mm"] = None if not x.size else {"n": int(x.size), "q25": float(np.percentile(x, 25)), "median": float(np.median(x)),
                                                                              "q75": float(np.percentile(x, 75)), "q95": float(np.percentile(x, 95))}
    pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/S/raw/target_dest.json')).write_text(json.dumps({"definitions": __doc__, "by_arm": agg, "runs": res}, indent=0))
    print(json.dumps(agg, indent=1))
