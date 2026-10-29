"""IFG R8 diagnostic — do the dumped pixel gradients point at the target better than the 16x16 attention maps?

Maps per camera at the policy-image resolution 224x224 (dump `grad.npz`, collector definition):
  grip_gx   gripper_pixel_saliency   = sum_rgb |dy/dpixel * pixel|, y = mean first-10 gripper action
  arm_gx    arm_motion_pixel_saliency, y = mean first-10 ||used-arm joint delta||
  grip_g    ||dy/dpixel||_rgb (gripper), arm_g (arm motion)
  N, A      noun L10H7 / action L8H2 maps, each patch value spread over its 14x14 pixels
GT (evaluator only): replay class map (480x640) sampled at each policy pixel centre (same camera, 4:3 FOV).
Metrics over all 3 cameras (3 x 224 x 224 pixels):
  r_precision   precision of the top-|target| pixels (area-matched; ties broken by pixel index)
  hit1          the single highest pixel is a target pixel
  target_mass   share of the map's total mass lying on target pixels
  auc           P(score on a target pixel > score on a non-target pixel)
"""
import argparse, json, multiprocessing as mp, pathlib

import numpy as np

from AG3S.ag3s.experiments.studies import ifg_reader as R
from AG3S.ag3s.experiments.studies.t47_replay import CLS

MAPS = ("grip_gx", "arm_gx", "grip_g", "arm_g", "N", "A")


def gt_224(class_map):
    """(3,480,640) class map -> (3,224,224) at policy pixel centres."""
    u = ((np.arange(224) + 0.5) / 224 * 640).astype(int)
    v = ((np.arange(224) + 0.5) / 224 * 480).astype(int)
    return class_map[:, v][:, :, u]


def patch_up(m768):
    return np.repeat(np.repeat(np.asarray(m768, float).reshape(3, 16, 16), 14, axis=1), 14, axis=2)


def auc(s, y):
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    npos, nneg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else None


def frame(args):
    ref, run = args
    with np.load(ref.path / "grad.npz") as z:
        maps = {"grip_gx": z["gripper_pixel_saliency"].astype(float), "arm_gx": z["arm_motion_pixel_saliency"].astype(float),
                "grip_g": np.linalg.norm(z["gripper_d_pixels"].astype(float), axis=-1),
                "arm_g": np.linalg.norm(z["arm_motion_d_pixels"].astype(float), axis=-1)}
    with np.load(ref.path / "derived.npz") as z:
        maps["N"] = patch_up(z["noun_target_L10H7"]); maps["A"] = patch_up(z["serving_L8H2_last_d0"])
    with np.load(run / "s0/labels" / f"{ref.frame_id}.npz") as z:
        cls = gt_224(z["class_map"])
    y = (cls == CLS["target"]).ravel()
    robot = np.isin(cls, [CLS["robot"], CLS["robot_gripper"]]).ravel()
    out = dict(frame_id=ref.frame_id, episode_id=ref.episode_id, moment=ref.moment, target_px=int(y.sum()))
    k = int(y.sum())
    for name, m in maps.items():
        s = m.ravel()
        if k == 0:
            out[name] = None
            continue
        top = np.lexsort((np.arange(s.size), -s))[:k]
        out[name] = dict(r_precision=float(y[top].mean()), robot_in_top=float(robot[top].mean()), hit1=bool(y[int(np.argmax(s))]),
                         target_mass=float(s[y].sum() / max(s.sum(), 1e-30)), auc=auc(s, y),
                         top_camera=int(np.argmax(s) // (224 * 224)))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    refs = R.list_frames()
    with mp.get_context("spawn").Pool(4) as p:
        res = p.map(frame, [(r, a.run) for r in refs])
    with open(a.out / "grad_diag.jsonl", "w") as f:
        for d in res:
            f.write(json.dumps(d) + "\n")
    summ = {}
    for name in MAPS:
        v = [d[name] for d in res if d.get(name)]
        summ[name] = {k: float(np.mean([x[k] for x in v if x[k] is not None])) for k in ("r_precision", "robot_in_top", "hit1", "target_mass", "auc")}
        summ[name]["by_moment_r_precision"] = {m: float(np.mean([d[name]["r_precision"] for d in res if d.get(name) and d["moment"][:2] == m]))
                                               for m in sorted({d["moment"][:2] for d in res})}
    (a.out / "grad_summary.json").write_text(json.dumps(summ, indent=1))
    for name, s in summ.items():
        print(f"{name:8s} R-prec={s['r_precision']:.3f} robot_in_top={s['robot_in_top']:.3f} hit1={s['hit1']:.3f} target_mass={s['target_mass']:.3f} AUC={s['auc']:.3f}")
        print("          by moment:", {k: round(v, 2) for k, v in s["by_moment_r_precision"].items()})


if __name__ == "__main__":
    main()
