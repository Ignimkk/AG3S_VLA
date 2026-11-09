"""(.venv-ag3s, CPU) SL-PE7 calibration (2026-11-09 00:0x KST, dev-30): does the attention tell a seed cut on the target from a seed
cut on another object? Per pre-held call with a plan_close seed on the gt24att / SL-PE records: cut = split_at_seed (same as
the server), GT majority body role of the cut, ratio = mean raw attention of the cut / median raw attention of the non-support
cloud. Writes att_gate_<tag>.json. Run: SCENE_DIR=... [SAFELIBERO_PATCH=1] python pe_att_gate.py '<run glob>' <tag>"""
import collections
import json
import pathlib
import sys

import mujoco
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pe_signal as PS  # noqa: E402
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.stages.close_seed import split_at_seed  # noqa: E402

Q = PS.Q
OUT = pathlib.Path(__file__).resolve().parent


def main():
    run = next(p for p in sorted(Q.E.RUNS.glob(sys.argv[1])) if not p.name.startswith("INVALID"))
    tag = sys.argv[2]
    eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
    order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
    ch = PS.episode_chunks(run, order)
    out = []
    for stem in order:
        if not (Q.SCENE / f"{stem}_scene.xml").exists():
            continue
        info, m, rg, cache, stc = PS.scene_for(stem)
        d = mujoco.MjData(m)
        geom = np.load(run / "episodes" / f"{stem}_geom.npz")
        grow = {int(o): i for i, o in enumerate(geom["obs_index"])}
        calls = {int(json.loads(l)["seq"]): json.loads(l) for l in open(run / "episodes" / f"{stem}_calls.jsonl")}
        for ts, cp in ch.get(stem, []):
            c = calls[ts]
            g = c.get("grasp") or {}
            if g.get("state") not in PS.PRE:
                continue
            pc = (g.get("tip_alignment") or {}).get("plan_close")
            if not pc:
                continue
            z = np.load(cp)
            pts = z["attention_cloud"].astype(float)
            att = z["attention_raw"].astype(float)
            sup = z["support_mask"].astype(bool)
            idx, rec = split_at_seed(pts, pc["tip_m"], exclude_mask=sup)
            if idx is None:
                continue
            i = grow[int(c["obs_index"])]
            d.qpos[:] = geom["sim_state_image"][i][1:1 + m.nq]
            mujoco.mj_forward(m, d)
            S1, G1 = stc.world(d)
            S2, G2 = cache.world(d)
            tree = Q.SplitTree(cKDTree(S1), rg[G1], S2, rg[G2])
            Tw = geom["T_world_base"][i]
            dd, j = tree.query(pts[idx] @ Tw[:3, :3].T + Tw[:3, 3])
            role = collections.Counter(np.where(dd <= 0.01, tree.roles[j], "none").tolist()).most_common(1)[0][0]
            med = float(np.median(att[~sup])) if (~sup).any() else 0.0
            ratio = float(att[idx].mean() / med) if med > 0 else float("inf")
            out.append({"ep": stem, "seq": ts, "latch": g.get("state"), "success": None, "role": role,
                        "ratio": ratio, "n": int(len(idx))})
        print(stem, flush=True)
    (OUT / f"att_gate_{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
