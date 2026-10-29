"""T43 T fix 1 — what blocks a move toward the policy chunk from the stationary plan (floored scene):
plans Q(a) = stay + a (reference - stay); per a: worst rows by class/why/link (floored values)."""
import json, sys, pathlib, dataclasses, collections
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import recov_replay as RR
from AG3S.trajopt.safe_policy import classify_violations, recovery_floor_rows

def main(key, seqs, out):
    run = RR.Run(key); res = {}
    for seq in seqs:
        s, refc, _, qn, scene, held = run.inputs(seq)
        G = run.G
        from AG3S.trajopt.linearize import CollisionLinearizer
        lin = CollisionLinearizer(G["robot"], G["layout"], 8)
        meas = (s.get("finger_joints") or {}).get("opening_norm") or {}
        path, _ = G["fmap"].path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, 8, default=G["robot"].param_vector())
        lin.set_joint_parameter_path(path)
        lin.set_attached(scene.attached_points, scene.attached_parent_link)
        auth = s["verdict"]["classification"]["authorized_links"]; man = s.get("manipulated") or {}
        floor, rec = recovery_floor_rows(lin, qn, scene, authorized_links=auth, manipulated=man, tolerance=1e-4)
        fs = dataclasses.replace(scene, esdf_floor=floor) if floor is not None else scene
        stay = np.tile(qn[G["layout"].q_indices][:, None], (1, 8))
        ref = G["layout"].chunk_to_trajectory(refc)[:, :8]
        # end-effector lift of the reference (hold link origin z) per step
        hold = scene.attached_parent_link
        z0 = G["robot"].link_pose(qn, hold)[2, 3]
        zref = [G["robot"].link_pose(G["layout"].full_q(ref[:, k:k+1], qn)[:, 0], hold)[2, 3] - z0 for k in range(8)]
        rows = {}
        for a in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
            tr = stay + a * (ref - stay)
            cls = classify_violations(lin, tr, qn, fs, authorized_links=auth, tolerance=1e-4, manipulated=man, keep_rows=True)
            bad = [r for r in cls["rows"]]
            agg = collections.defaultdict(lambda: [0, 0.0])
            for r in bad:
                k = f'{r["class"]}|{r["why"]}|{r["link"]}|{"F" if floor is not None and floor[r["query"]] < 0 else "-"}'
                agg[k][0] += 1; agg[k][1] = min(agg[k][1], round(1e3 * r["clearance_m"], 2))
            rows[a] = dict(agg)
        res[seq] = {"recovery": rec, "ref_lift_mm": [round(1e3 * z, 1) for z in zref], "rows": rows}
        print(key, seq, "ref lift mm", res[seq]["ref_lift_mm"], flush=True)
        for a, agg in rows.items():
            print("   a", a, dict(sorted(agg.items(), key=lambda kv: kv[1][1])[:6]), flush=True)
    pathlib.Path(out).write_text(json.dumps(res, indent=1, default=str))

if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3])
