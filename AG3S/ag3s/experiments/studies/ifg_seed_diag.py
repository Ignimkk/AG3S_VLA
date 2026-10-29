"""IFG R4 diagnostic (F1 seed identity under the FK self-filter; dev only, GT for diagnosis).

Per frame and noun map N, compares the top-5 seed patches chosen
  restricted    R0 rule: patch needs >= 60 non-robot valid depth pixels
  unrestricted  any patch (raw map top-5)
and reports, per seed set: GT target pixel fraction (all pixels), target pixels the FK mask removed,
robot / gripper fraction. Also per frame: target pixels removed by the FK mask (all cameras).
"""
import argparse, json, multiprocessing as mp, pathlib

import numpy as np

from AG3S.ag3s.experiments.studies import ifg_reader as R
from AG3S.ag3s.experiments.studies import ifg_s1 as S1
from AG3S.ag3s.experiments.studies.t47_replay import CLS


def patch_index(h, w, cam):
    yy, xx = np.indices((h, w))
    return (yy * 16 // h) * 16 + xx * 16 // w + 256 * cam


def frame(args):
    ref, s0 = args
    with np.load(s0 / "replay" / f"{ref.frame_id}.npz") as z:
        depth, rmask = z["depth"], z["robot_mask"]
    with np.load(s0 / "labels" / f"{ref.frame_id}.npz") as z:
        cm = z["class_map"]
    with np.load(ref.path / "derived.npz") as z:
        n = z["noun_target_L10H7"].astype(np.float64).reshape(-1)
    P, V, RB, TG, GR = [], [], [], [], []
    for c in range(3):
        h, w = depth[c].shape
        P.append(patch_index(h, w, c).ravel())
        v = (depth[c] >= 0.05) & (depth[c] <= 3.0)
        V.append(v.ravel()); RB.append(rmask[c].ravel())
        TG.append((cm[c] == CLS["target"]).ravel())
        GR.append(np.isin(cm[c], [CLS["robot"], CLS["robot_gripper"]]).ravel())
    P, V, RB, TG, GR = map(np.concatenate, (P, V, RB, TG, GR))
    alg = np.bincount(P, weights=(V & ~RB), minlength=768)
    tot = np.bincount(P, minlength=768).astype(float)
    tgt = np.bincount(P, weights=TG, minlength=768)
    tgt_rm = np.bincount(P, weights=TG & V & RB, minlength=768)
    grp = np.bincount(P, weights=GR, minlength=768)
    out = dict(frame_id=ref.frame_id, episode_id=ref.episode_id, moment=ref.moment,
               target_px=int(TG.sum()), target_px_removed_by_fk=int((TG & V & RB).sum()))
    for name, valid in (("restricted", alg >= 60), ("unrestricted", None)):
        ix = S1.top_k(n, 5, valid)
        out[name] = dict(patches=ix.tolist(), target_frac=float((tgt[ix] / tot[ix]).mean()) if ix.size else None,
                         target_frac_after_fk=float(((tgt[ix] - tgt_rm[ix]) / tot[ix]).mean()) if ix.size else None,
                         robot_frac=float((grp[ix] / tot[ix]).mean()) if ix.size else None,
                         alg_px=[int(x) for x in alg[ix]])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--s0", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    refs = [r for r in R.list_frames() if (a.s0 / "replay" / f"{r.frame_id}.npz").exists()]
    with mp.get_context("spawn").Pool(4) as p:
        res = p.map(frame, [(r, a.s0) for r in refs])
    with open(a.out / "seed_diag.jsonl", "w") as f:
        for d in res:
            f.write(json.dumps(d) + "\n")
    print(len(res))


if __name__ == "__main__":
    main()
