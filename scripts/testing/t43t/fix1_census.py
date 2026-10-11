"""T43 T fix 1 — on every recorded held chunk (attach .. detach) of the T42-1 / T41 c runs: does the
recovery floor engage (a non-allowed violated row on the hand / held object at q_now)? Where it does,
solve off vs on (fixed input) and compare the verdict. Scene = qnow.scene_for (held chunks only)."""
import json, sys, pathlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import qnow as Q

def runs():
    out = []
    for rd, v in Q.IX.items():
        if "chunks" in v and v.get("complete") and ("/T42/1/" in rd or "/T41/c/" in rd):
            out.append(rd)
    return out

def one(rd):
    import recov_replay as RR
    from AG3S.trajopt.linearize import CollisionLinearizer
    from AG3S.trajopt.safe_policy import recovery_floor_rows
    key = "X"; Q.RUNS[key] = rd
    run = RR.Run(key)
    G = run.G
    res = {"run": rd, "chunks": []}
    for seq in sorted(int(k) for k in run.ix["chunks"]):
        with np.load(run.ix["chunks"][str(seq)]["npz"], allow_pickle=False) as z:
            s = json.loads(str(z["summary_json"]))
        g = s.get("grasp") or {}
        if not g.get("attached") or seq not in run.plan:
            continue
        s, refc, _, qn, scene, held = run.inputs(seq)
        lin = CollisionLinearizer(G["robot"], G["layout"], 8)
        meas = (s.get("finger_joints") or {}).get("opening_norm") or {}
        path, _ = G["fmap"].path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, 8, default=G["robot"].param_vector())
        lin.set_joint_parameter_path(path)
        lin.set_attached(scene.attached_points, scene.attached_parent_link)
        auth = ((s.get("verdict") or {}).get("classification") or {}).get("authorized_links") or []
        floor, rec = recovery_floor_rows(lin, qn, scene, authorized_links=auth, manipulated=s.get("manipulated") or {}, tolerance=1e-4)
        hold_rec = bool((s.get("exec_feedback") or {}))  # placeholder
        kinds = [r.get("kind") for r in ((s.get("verdict") or {}).get("reasons") or [])]
        row = {"seq": seq, "engaged": floor is not None, "n_rows": len(rec["queries"]),
               "recorded_kinds": kinds}
        if floor is not None:
            off = run.solve(seq)
            on = run.solve(seq, recovery=True, overrides={"safety": {"recovery_floor": True}})
            row.update(off_hold=off["hold"], on_hold=on["hold"], off_reasons=off["reasons"], on_reasons=on["reasons"],
                       on_progress=on["progress"], on_raw_robot_min=on["raw_robot_min_mm"], on_raw_held_min=on["raw_held_min_mm"])
        res["chunks"].append(row)
    out_dir = pathlib.Path(sys.argv[1]).with_suffix(""); out_dir.mkdir(exist_ok=True)
    (out_dir / (rd.split("/runs/")[-1].replace("/", "_") + ".json")).write_text(json.dumps(res, default=str))
    print("done", rd, sum(c["engaged"] for c in res["chunks"]), "/", len(res["chunks"]), flush=True)
    return res

if __name__ == "__main__":
    rs = runs()
    with ProcessPoolExecutor(max_workers=int(sys.argv[2])) as ex:
        res = list(ex.map(one, rs))
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, default=str))
