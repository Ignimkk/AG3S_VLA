"""(.venv-ag3s, CPU, offline) candidate SL-PE9 (2026-11-09 03:4x, dev-4a): before the VLA's chunk carries a close command
(no plan_close seed), cut the object under the ATTENTION PEAK the same way as the grasp seed (`split_at_seed`: local support
band and flat walls removed) so that a bowl merged with a cabinet / drawer becomes its own graspable candidate.
Per pre-held call: seed = the non-support attention-cloud point of maximal raw attention; cut; GT body majority of the cut
(scoring only); PE7 ratio = mean raw attention of the cut / median raw attention of the non-support cloud. Also the recorded
identity's GT role (what the server had). Writes peak_<tag>.json, prints a table by bench.
Run: SCENE_DIR=<dir> [SAFELIBERO_PATCH=1] python att_peak_seed.py '<run glob>' <tag>"""
import collections
import json
import pathlib
import sys

import mujoco
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/mnt/dev/AG3S_VLA/outputs/2026-11-08_PE_경로끝점_identity")
import pe_signal as PS  # noqa: E402
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.stages.close_seed import split_at_seed  # noqa: E402

Q = PS.Q
OUT = pathlib.Path(__file__).resolve().parent


def main():
    runs = [p for p in sorted(Q.E.RUNS.glob(sys.argv[1])) if not p.name.startswith("INVALID") and (p / "episodes.jsonl").exists()]
    tag = sys.argv[2]
    rows = []
    for run in runs:
        eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
        order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
        succ = {o: bool(e["official_success"]) for o, e in zip(order, eps)}
        obst = {o: (e.get("safelibero") or {}).get("obstacle") for o, e in zip(order, eps)}
        ch = PS.episode_chunks(run, order)
        for stem in order:
            if not (Q.SCENE / f"{stem}_scene.xml").exists():
                continue
            info, m, rg, cache, stc = PS.scene_for(stem)
            d = mujoco.MjData(m)
            gb = np.array([m.body(b).name for b in m.geom_bodyid])
            geom = np.load(run / "episodes" / f"{stem}_geom.npz")
            grow = {int(o): i for i, o in enumerate(geom["obs_index"])}
            calls = {int(json.loads(l)["seq"]): json.loads(l) for l in open(run / "episodes" / f"{stem}_calls.jsonl")}
            for ts, cp in ch.get(stem, []):
                c = calls.get(ts)
                if c is None:
                    continue
                g = c.get("grasp") or {}
                if g.get("state") not in ("searching", "latched", "closing"):
                    continue
                z = np.load(cp)
                P = z["attention_cloud"].astype(float)
                sm = z["support_mask"].astype(bool)
                raw = z["attention_raw"].astype(float)
                if not len(P) or sm.all():
                    continue
                ns = np.flatnonzero(~sm)
                peak = ns[int(np.argmax(raw[ns]))]
                idx, _ = split_at_seed(P, P[peak], exclude_mask=sm)
                i = grow[int(c["obs_index"])]
                d.qpos[:] = geom["sim_state_image"][i][1:1 + m.nq]
                mujoco.mj_forward(m, d)
                S1, G1 = stc.world(d)
                S2, G2 = cache.world(d)
                S = np.r_[S1, S2]
                R = np.r_[rg[G1], rg[G2]]
                B = np.r_[gb[G1], gb[G2]]
                k = R != "robot"
                tree = cKDTree(S[k])
                Rk, Bk = R[k], B[k]
                Tw = geom["T_world_base"][i]

                def who(X):
                    if X is None or not len(X):
                        return None, None
                    dd, jj = tree.query(np.asarray(X, float) @ Tw[:3, :3].T + Tw[:3, 3])
                    ok = dd <= 0.015
                    if not ok.any():
                        return "none", None
                    r = collections.Counter(Rk[jj[ok]].tolist()).most_common(1)[0][0]
                    b = collections.Counter(Bk[jj[ok]].tolist()).most_common(1)[0][0]
                    if obst.get(stem) and str(b).startswith(obst[stem]):
                        r = "SAFELIBERO_OBSTACLE"
                    return r, b

                cut_role, cut_body = who(P[idx] if idx is not None else None)
                id_role, _ = who(z["target_points"] if "target_points" in z.files else None)
                ratio = float(raw[idx].mean() / max(np.median(raw[ns]), 1e-12)) if idx is not None else None
                rows.append({"run": run.name[16:60], "stem": stem, "success": succ[stem], "seq": ts, "state": g.get("state"),
                             "has_plan_close": bool((g.get("tip_alignment") or {}).get("plan_close")),
                             "cut": cut_role, "cut_body": cut_body, "n_cut": None if idx is None else int(len(idx)),
                             "ratio": ratio, "identity": id_role})
            print(stem, "done", flush=True)
    (OUT / f"peak_{tag}.json").write_text(json.dumps(rows, indent=0))
    for pcl in (False, True):
        sel = [r for r in rows if r["has_plan_close"] == pcl]
        print(f"\n== plan_close {'yes' if pcl else 'no'}: {len(sel)} calls")
        for key in ("identity", "cut"):
            c = collections.Counter(str(r[key]) for r in sel)
            print(f"  {key:8s}", ", ".join(f"{k} {v / max(len(sel), 1):.0%}" for k, v in c.most_common()))
        c5 = collections.Counter(str(r["cut"]) for r in sel if r["ratio"] is not None and r["ratio"] >= 5)
        n5 = sum(c5.values())
        print(f"  cut & ratio>=5 ({n5}):", ", ".join(f"{k} {v / max(n5, 1):.0%}" for k, v in c5.most_common()))


if __name__ == "__main__":
    main()
