"""T43 S (4) — where did the POLICY (reference) vs the TO (refined, executed) put the fingertips at the grasp?  E3b T42-1 runs.

For each E3b run and each control row t in {close_t - 8, close_t, close_t + 8, close_t + 16} (close_t = first close command):
  chunk = the planning record whose 8 control rows contain t; k = t - chunk t_step
  measured : tip / finger geometry at the recorded qpos of row t (measure_pick.geom_at)
  reference: the same qpos with the arm joints replaced by actions_reference[k] (the policy's joint target for row t)
  refined  : the same qpos with the arm joints replaced by actions[k] (the TO's target = what was sent, unless HOLD)
  (fruit at its measured pose in all three; targets are positions the servo is driven to, not reached poses)
Output: raw/pick_ref_target.json  {run: {row_name: {measured|reference|refined: {tip_dz_mm, tip_dxy_mm, tilt_deg, ...}, executed}}}
usage: pick_ref_target.py [N_WORKERS]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import measure_pick as MP

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/S/raw'))
KEEP = ("tilt_deg", "approach_yaw_deg_mirrored", "closing_yaw_deg_mirrored_mod180", "tip_dz_mm", "tip_dxy_mm", "grasp_dz_mm", "grasp_dxy_mm")


def pick(g): return {k: g[k] for k in KEEP} | {"tip_mid": g["tip_mid"], "fruit_pos": g["fruit_pos"]}


def one(key):
    r = MP.RUNS[key]
    if not key.startswith("E3b/"): return key, None
    rd = pathlib.Path(r["run_dir"]); arm, fruit = r["used_arm"], r["target_fruit"]
    rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    ctrl = {c["t_step"]: c for c in rows if c["kind"] == "control"}; plan = [p for p in rows if p["kind"] == "planning"]
    ct = (r.get("close_cmd_t") or [None])[0]
    if ct is None: return key, None
    out = {}
    for nm, t in (("close_m8", ct - 8), ("close", ct), ("close_p8", ct + 8), ("close_p16", ct + 16)):
        if t not in ctrl: continue
        p = max((pp for pp in plan if pp["t_step"] <= t), key=lambda pp: pp["t_step"])
        k = t - p["t_step"]
        if k >= 8: continue
        q = np.asarray(ctrl[t]["qpos"], float)
        ref = np.asarray(p["actions_reference"], float)[k]; fin = np.asarray(p["actions"], float)[k]
        gm = MP.geom_at(q, arm, fruit); gr = MP.geom_at(MP.q_with_chunk_row(q, ref, arm), arm, fruit); gf = MP.geom_at(MP.q_with_chunk_row(q, fin, arm), arm, fruit)
        out[nm] = {"t": t, "seq": p["seq"], "k": k, "executed": bool(ctrl[t].get("executed", True)),
                   "measured": pick(gm), "reference": pick(gr), "refined": pick(gf),
                   "ref_minus_refined_tip_mm": float(1e3 * np.linalg.norm(gr["tip_mid"] - gf["tip_mid"])),
                   "ref_gripper_cmd": float(ref[MP.GRIP_COL[arm]]), "refined_gripper_cmd": float(fin[MP.GRIP_COL[arm]])}
    return key, MP.jl(out)


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with Pool(nw) as pool:
        res = dict(pool.map(one, sorted(MP.RUNS), chunksize=1))
    (OUT / "pick_ref_target.json").write_text(json.dumps({"definitions": __doc__, "runs": {k: v for k, v in res.items() if v}}, indent=0))
    print("done", sum(1 for v in res.values() if v))
