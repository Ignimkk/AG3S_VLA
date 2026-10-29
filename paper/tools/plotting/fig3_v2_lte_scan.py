"""Figure 3 v2 — scan recorded requests for the LTE slice figure: how completely the target-free field drops the target. CPU, read only.

  cd /mnt/dev/AG3S_VLA-development; MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python \
      paper/tools/plotting/fig3_v2_lte_scan.py --run <RUN> --eps 119 121 ... --out <json>

Same slice as `fig3_v2_lte_slices.py` (vertical plane through the target and nearest-neighbour GT centres). Per request, on the slice:
  target_left  fraction of GT-target-interior pixels where the target-free field is still <= 0 (0 = target fully dropped)
  target_main  fraction of GT-target-interior pixels where the main field is <= 0 (the scene field keeps the target)
  neigh_kept   fraction of GT-neighbour-interior pixels where the target-free field is <= 0 (1 = neighbour kept)
  neigh_main   the same for the main field
Only requests whose target-free layer exists (latched / closing) are scanned.
"""
import argparse, glob, json, pathlib, sys, time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--kit", default="IFG-APPLE-SPLIT"); ap.add_argument("--arm", default="SPL")
    ap.add_argument("--eps", type=int, nargs="+", required=True); ap.add_argument("--states", nargs="+", default=["latched", "closing"])
    ap.add_argument("--every", type=int, default=1); ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    from AG3S.ag3s.experiments.studies import ifg_audit_lte as L
    from AG3S.trajopt.experiments.fixed_input_ablation import build_field
    scene = L.Scene(a.run / "snapshot_split/robots/rby1/description/models/rby1a/mujoco/model_transport.xml")
    mj, m = scene.mj, scene.m
    rows = []
    for ep in a.eps:
        jf = next(f for f in sorted(glob.glob(str(a.run / "verify" / a.kit / "replay" / f"runs_{a.arm}_{a.arm}_ep{ep}_*.json"))) if not f.endswith("npz"))
        js = json.load(open(jf)); b = js["server_chunk_alignment"]["block"]
        rd = pathlib.Path(js["run_dir"])
        pl = [r for r in (json.loads(x) for x in open(rd / "frames.jsonl")) if r.get("kind") == "planning"]
        gs = js["geom_sets"]
        gid = lambda n: (int(n[5:]) if n.startswith("geom#") else mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, n))  # noqa: E731
        target_g = [gid(n) for n in gs["target"]]; fruit_g = [gid(n) for n in gs["fruit_other"]]
        t0 = time.time(); n = 0
        for k in range(b["block_len"]):
            cp = pathlib.Path(b["chunk_dir"]) / f"chunk_{k + b['offset']:05d}.npz"
            z = np.load(cp, allow_pickle=True)
            st = str((json.loads(str(z["summary_json"])).get("grasp") or {}).get("state"))
            if st not in a.states or k % a.every:
                continue
            field = build_field(z, outside_distance=1.0)
            if not field.has_target_free:
                continue
            scene.set_q(pl[k]["qpos"])
            fb = {}
            for g in fruit_g:
                fb.setdefault(scene.bname[g], []).append(g)
            gap = {bn: min(mj.mj_geomDistance(m, scene.d, t, g, 1.0, scene.ft) for t in target_g for g in gg) for bn, gg in fb.items()}
            nb = min(gap, key=gap.get); neigh_g = fb[nb]
            ct = scene.body_centre(target_g)[0]; cn = scene.body_centre(neigh_g)[0]
            e1 = cn - ct; e1[2] = 0; dist = float(np.linalg.norm(e1)); e1 /= dist
            us = np.arange(-0.07, dist + 0.07, 0.002); zs = np.arange(-0.04, 0.12, 0.002)
            U, Z = np.meshgrid(us, zs)
            G = ct[None] + U.ravel()[:, None] * e1 + Z.ravel()[:, None] * np.array([0, 0, 1.0])
            D, _ = scene.query(G, force=target_g + neigh_g)
            cls = L.classes_for(scene, set(target_g), set(neigh_g), set(fruit_g), set())
            inT = np.where((cls == 0)[None], D, np.inf).min(1) < -0.002
            inN = np.where((cls == 1)[None], D, np.inf).min(1) < -0.002
            fm = field.distance(G); ff = field.target_free_distance(G)
            r = {"ep": ep, "request": k, "state": st, "neighbour": nb, "gap_mm": round(float(gap[nb]) * 1000, 1),
                 "n_t": int(inT.sum()), "n_n": int(inN.sum()),
                 "target_left": round(float((ff[inT] <= 0).mean()), 4) if inT.any() else None,
                 "target_main": round(float((fm[inT] <= 0).mean()), 4) if inT.any() else None,
                 "neigh_kept": round(float((ff[inN] <= 0).mean()), 4) if inN.any() else None,
                 "neigh_main": round(float((fm[inN] <= 0).mean()), 4) if inN.any() else None}
            rows.append(r); n += 1
            del z, field
        print(f"ep{ep}: {n} requests {time.time() - t0:.0f}s", flush=True)
        a.out.write_text(json.dumps(rows, indent=0))


if __name__ == "__main__":
    main()
