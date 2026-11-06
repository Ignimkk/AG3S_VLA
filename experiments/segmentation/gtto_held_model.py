#!/usr/bin/env python3
"""SL-GTTO4 §0 evidence: the attached held model vs the support plane and vs the true object (sim), held calls only.

Per held call (latch HELD, not HOLD) of the given runs:
  * ref_bottom_mm  — lowest point of the attached query spheres (`grasp.attached.held.spheres_parent_frame`, panda_hand
    frame) along the VLA reference (`calls.npz` joint_reference, FK) minus the support plane offset (`held.support.plane`)
  * deviation      — `to.reference_deviation`
  * model_err_mm   — at row 0 of the reference: model bottom minus the true object's bottom (sim `geom.npz` body
    position; the object's resting bottom is taken to lie on the support plane — an approximation)

usage: python3 experiments/segmentation/gtto_held_model.py --glob '*_gtto-gt2-libero9*' [--glob ...] --out DIR
"""
import argparse
import glob
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
BAND = 0.015


def _addr(names):
    a, i = {}, 0
    for n in names:
        n = str(n)
        a[n] = i
        i += 7 if "joint0" in n else 1
    return a


def episode_rows(f, model):
    rows = [json.loads(line) for line in open(f) if line.strip()]
    base = f.replace("_calls.jsonl", "")
    c = np.load(base + "_calls.npz")
    G = np.load(base + "_geom.npz", allow_pickle=True)
    tgt = next((((r.get("gt_perception") or {}).get("grounding") or {}).get("target") for r in rows
                if ((r.get("gt_perception") or {}).get("grounding") or {}).get("target")), None)
    A = _addr(G["model_joint_names"])
    jn = [k for k in A if tgt and k.startswith(tgt + "_joint")]
    ad = 1 + A[jn[0]] if jn else None

    def obj_z(o):
        T = G["T_world_base"][o]
        p = G["sim_state"][o][ad:ad + 3]
        return (np.linalg.inv(T) @ np.r_[p, 1])[2]

    rest = obj_z(rows[0]["obs_index"]) if ad is not None else None
    att, zf, out = None, None, []
    for k, r in enumerate(rows):
        gr = r.get("grasp") or {}
        a = gr.get("attached")
        if a and a.get("held") and a["held"].get("spheres_parent_frame"):
            att = [(np.array(s["centre_m"]), s["radius_mm"] / 1000.0) for s in a["held"]["spheres_parent_frame"]]
            pl = (a["held"].get("support") or {}).get("plane")
            zf = pl["offset_m"] if pl else zf
        if gr.get("state") != "held" or att is None or zf is None or r.get("hold"):
            continue

        def low(Q):
            m = 1e9
            for q in Q:
                T = np.asarray(model.link_pose(np.asarray(q[:7], np.float64), "panda_hand"))
                for cc, rad in att:
                    m = min(m, (T[:3, :3] @ cc + T[:3, 3])[2] - rad - zf)
            return m

        T0 = np.asarray(model.link_pose(np.asarray(c["joint_reference"][k][0][:7], np.float64), "panda_hand"))
        mb = min((T0[:3, :3] @ cc + T0[:3, 3])[2] - rad for cc, rad in att)
        err = None if rest is None else (mb - (obj_z(r["obs_index"]) - rest + zf)) * 1000.0
        out.append({"seq": r["seq"], "ref_bottom_mm": low(c["joint_reference"][k]) * 1000.0,
                    "refined_bottom_mm": low(c["joint_refined"][k]) * 1000.0,
                    "deviation": float((r.get("to") or {}).get("reference_deviation") or 0.0),
                    "model_err_mm": err,
                    "hbc": (r.get("to") or {}).get("held_beneath_contact")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from AG3S.adapters.panda.model import build_panda_bare_model
    model = build_panda_bare_model()
    res = {}
    for g in a.glob:
        for f in sorted(glob.glob(f"{RESULTS}/{g}/episodes/*_calls.jsonl")):
            run = f.split("/")[-3].split("_gtto-")[1].split("-vla")[0]
            ep = os.path.basename(f).replace("_calls.jsonl", "")
            rows = episode_rows(f, model)
            if rows:
                res[f"{run}/{ep}"] = rows
    summary = {}
    for key, rows in res.items():
        ref = np.array([r["ref_bottom_mm"] for r in rows])
        dev = np.array([r["deviation"] for r in rows])
        near = ref < BAND * 1000
        errs = [r["model_err_mm"] for r in rows if r["model_err_mm"] is not None]
        summary[key] = {"n_held_calls": len(rows), "n_ref_near_support": int(near.sum()),
                        "dev_median_near": float(np.median(dev[near])) if near.any() else None,
                        "dev_median_other": float(np.median(dev[~near])) if (~near).any() else None,
                        "model_err_median_mm": float(np.median(errs)) if errs else None,
                        "hbc_applied": sum(1 for r in rows if (r["hbc"] or {}).get("applied")),
                        "hbc_released_calls": sum(1 for r in rows if ((r["hbc"] or {}).get("rows_released") or 0) > 0)}
    (a.out / "held_model.json").write_text(json.dumps({"globs": a.glob, "band_mm": BAND * 1000, "summary": summary,
                                                       "calls": res}, indent=1, ensure_ascii=False))
    from AG3S.ag3s.experiments.common import figstyle
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ko = figstyle.use_korean()
    T = ({"a": "held call: VLA reference 의 쥔 모델 바닥 높이 vs TO 경로 수정량",
          "x": "reference 경로 중 쥔 모델 최저점 − 지지면 (mm)", "y": "reference_deviation",
          "b": "episode 별 쥔 모델 바닥 − 실제 물체 바닥 (중앙값, mm)"} if ko else
         {"a": "held calls: held-model bottom along the VLA reference vs TO path change",
          "x": "lowest held-model point on the reference − support (mm)", "y": "reference_deviation",
          "b": "per episode: held-model bottom − true object bottom (median, mm)"})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    runs = sorted({k.split("/")[0] for k in res})
    colors = dict(zip(runs, ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]))
    for key, rows in res.items():
        run = key.split("/")[0]
        ax[0].scatter([r["ref_bottom_mm"] for r in rows], [r["deviation"] for r in rows], s=10, alpha=0.6,
                      color=colors[run], label=run)
    h, l = ax[0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax[0].legend(uniq.values(), uniq.keys(), fontsize=7)
    ax[0].axvline(BAND * 1000, color="k", ls="--", lw=0.8)
    ax[0].set_xlabel(T["x"]); ax[0].set_ylabel(T["y"]); ax[0].set_title(T["a"], fontsize=10)
    ax[0].set_xlim(-150, 250); ax[0].set_yscale("symlog", linthresh=0.05)
    keys = [k for k in summary if summary[k]["model_err_median_mm"] is not None]
    vals = [summary[k]["model_err_median_mm"] for k in keys]
    ax[1].barh(range(len(keys)), vals, color=[colors[k.split("/")[0]] for k in keys])
    ax[1].set_yticks(range(len(keys)))
    ax[1].set_yticklabels([k.replace("libero_spatial_", "").replace("safelibero_spatial_", "SL ") for k in keys],
                          fontsize=6)
    ax[1].axvline(0, color="k", lw=0.8); ax[1].set_xlim(-150, 150); ax[1].set_title(T["b"], fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out / "그래프_쥔모델.png", dpi=130)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
