"""IFG summary — sample/episode tables and paired episode bootstrap (protocol v1.1 §12).

Reads `<s3>/rows/*.jsonl` (ifg_run) and writes:
  sample_metrics.csv     one row per (frame, arm, threshold | K)
  episode_metrics.csv    episode means of IoU / precision / recall / FP parts (valid frames only)
  paired_summary.json    per comparison: episode-mean difference and paired bootstrap 95% CI
                         (2000 replicates, seed 20261015, resampling episodes)
  latency.json           p50 / p95 per method of the per-arm compute time (graph build excluded)

Frame -> episode mean -> episodes are the resampling unit. Frames without a visible target are
excluded from IoU / recall means and counted (`n_frames_target_absent`).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import pathlib
from collections import defaultdict

import numpy as np

COLS = ["run_id", "episode_id", "frame_id", "moment", "kind", "split", "level", "arm_id", "method", "seed_source",
        "feature_name", "voxel_size", "threshold", "K", "valid", "missing_reason", "TP", "FP", "FN", "precision",
        "recall", "iou", "neighbor_fp", "robot_fp", "support_fp", "background_fp", "unknown_weight",
        "target_lost_to_robot_filter", "target_voxels", "pred_voxels", "seed_target_frac", "n_roi", "ms_solve"]
METRICS = ("iou", "precision", "recall", "neighbor_fp", "support_fp", "robot_fp", "unknown_weight")
B, SEED = 2000, 20261015


def s0_valid(s0: pathlib.Path) -> dict:
    """frame_id -> valid, last record wins (frames re-checked later are appended)."""
    out = {}
    for l in open(s0 / "provenance_checks.jsonl"):
        r = json.loads(l)
        out[r["frame_id"]] = bool(r["valid"])
    return out


def load_rows(s3: pathlib.Path, extra_dirs=(), valid: dict | None = None) -> tuple[list[dict], list]:
    rows, skipped = [], set()
    for d in (s3, *extra_dirs):
        for f in sorted(glob.glob(str(pathlib.Path(d) / "rows" / "*.jsonl"))):
            fid = pathlib.Path(f).stem
            if valid is not None and not valid.get(fid, False):
                skipped.add(fid)
                continue
            rows.extend(json.loads(l) for l in open(f))
    return rows, sorted(skipped)


def key(r):
    return (r["arm_id"], r.get("threshold") if r["level"] == "voxel" else r.get("K"), r["level"])


def episode_means(rows):
    """{(arm, thr, level): {episode: {metric: mean}}} over valid frames."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        if not r.get("valid", False):
            continue
        k = key(r)
        for m in METRICS:
            v = r.get(m)
            if v is not None:
                acc[k][r["episode_id"]][m].append(float(v))
    return {k: {e: {m: float(np.mean(v)) for m, v in d.items()} for e, d in eps.items()} for k, eps in acc.items()}


def paired(epm, a, b, metric="iou"):
    ea, eb = epm.get(a, {}), epm.get(b, {})
    common = sorted(set(ea) & set(eb))
    common = [e for e in common if metric in ea[e] and metric in eb[e]]
    if not common:
        return None
    d = np.array([ea[e][metric] - eb[e][metric] for e in common])
    rng = np.random.default_rng(SEED)
    boot = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return dict(a=list(a), b=list(b), metric=metric, n_episodes=len(common), mean_a=float(np.mean([ea[e][metric] for e in common])),
                mean_b=float(np.mean([eb[e][metric] for e in common])), diff=float(d.mean()),
                ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                episodes_a_better=int((d > 0).sum()), episodes_b_better=int((d < 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", type=pathlib.Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5, help="threshold of the headline table (all are written)")
    ap.add_argument("--s0", type=pathlib.Path, required=True, help="S0 dir: frames not S0-valid are excluded")
    ap.add_argument("--baseline", type=pathlib.Path, nargs="*", default=[], help="ifg_baseline out dirs (rows/)")
    a = ap.parse_args()
    rows, skipped = load_rows(a.s3, a.baseline, s0_valid(a.s0))
    # a frame enters the comparison only if both the IFG and (when given) the baseline rows exist
    have = defaultdict(set)
    for r in rows:
        have[r["frame_id"]].add("base" if str(r["arm_id"]).startswith("T54") else "ifg")
    need = {"ifg", "base"} if a.baseline else {"ifg"}
    incomplete = sorted(f for f, h in have.items() if not need <= h)
    rows = [r for r in rows if r["frame_id"] not in incomplete]
    run_id = a.s3.parent.name
    with open(a.s3 / "sample_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "run_id": run_id})
    epm = episode_means(rows)
    with open(a.s3 / "episode_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm_id", "threshold_or_K", "level", "episode_id", *METRICS])
        for k, eps in sorted(epm.items(), key=lambda t: (t[0][2], t[0][0], t[0][1] or 0)):
            for e, d in sorted(eps.items()):
                w.writerow([k[0], k[1], k[2], e, *[d.get(m) for m in METRICS]])
    # comparisons fixed before reading the numbers: same seed (and feature), method contrasts
    feats = sorted({r["feature_name"] for r in rows if r["level"] == "voxel" and r.get("feature_name")})
    comps = []
    for thr in sorted({r["threshold"] for r in rows if r["level"] == "voxel" and r.get("threshold") is not None}):
        for sd in ("A", "N", "P"):
            g = (f"G:{sd}", thr, "voxel")
            for fn in feats:
                for m in ("FG", "FGu", "FGe", "F"):
                    for metric in ("iou", "precision", "recall", "neighbor_fp"):
                        p = paired(epm, (f"{m}:{sd}:{fn}", thr, "voxel"), g, metric)
                        if p:
                            comps.append(p)
                for metric in ("iou", "precision", "recall"):
                    p = paired(epm, (f"FG:{sd}:{fn}", thr, "voxel"), (f"F:{sd}:{fn}", thr, "voxel"), metric)
                    if p:
                        comps.append(p)
    for thr in (a.threshold,):
        for fn in feats:
            for base_arm, sd in (("N", "N"), ("L8H2", "A"), ("B", "N")):
                for kind in ("region", "ball"):
                    for metric in ("iou", "precision", "recall", "neighbor_fp"):
                        for mine in (f"FG:{sd}:{fn}", f"G:{sd}"):
                            p = paired(epm, ((mine if mine.startswith("G:") else mine), thr, "voxel"),
                                       (f"T54{kind}:{base_arm}", 0.5, "voxel"), metric)
                            if p:
                                comps.append(p)
    n_frames = len({r["frame_id"] for r in rows})
    absent = len({r["frame_id"] for r in rows if r.get("missing_reason") == "target_not_visible"})
    headline = []
    for (arm, thr, lvl), eps in sorted(epm.items()):
        if lvl == "voxel" and thr == a.threshold:
            vals = [d["iou"] for d in eps.values() if "iou" in d]
            headline.append(dict(arm_id=arm, threshold=thr, n_episodes=len(vals), episode_mean_iou=float(np.mean(vals)),
                                 episode_mean_precision=float(np.mean([d["precision"] for d in eps.values() if "precision" in d] or [np.nan])),
                                 episode_mean_recall=float(np.mean([d["recall"] for d in eps.values() if "recall" in d]))))
    (a.s3 / "paired_summary.json").write_text(json.dumps(dict(
        excluded_s0_invalid=skipped, excluded_incomplete=incomplete, n_frames=n_frames, n_episodes=len({r["episode_id"] for r in rows}), n_frames_target_absent=absent,
        bootstrap=dict(replicates=B, seed=SEED, unit="episode"), headline=headline, comparisons=comps), indent=1))
    lat = defaultdict(list)
    for r in rows:
        if r["level"] == "voxel" and r.get("ms_solve") is not None and r.get("threshold") == a.threshold:
            lat[r.get("method") or r["arm_id"].split(":")[0]].append(r["ms_solve"])
    frames = [json.loads(open(f).read()) for f in glob.glob(str(a.s3 / "frames" / "*.json"))]
    lat_out = {m: dict(n=len(v), p50_ms=float(np.percentile(v, 50)), p95_ms=float(np.percentile(v, 95))) for m, v in lat.items()}
    lat_out["graph_build"] = dict(n=len(frames), p50_ms=float(np.percentile([f["geometry"]["ms_build"] for f in frames], 50)),
                                  p95_ms=float(np.percentile([f["geometry"]["ms_build"] for f in frames], 95)))
    lat_out["note"] = "offline CPU single-thread per worker; arm time excludes graph build (shared) and feature extraction"
    (a.s3 / "latency.json").write_text(json.dumps(lat_out, indent=1))
    print(f"{len(rows)} rows, {n_frames} frames, absent {absent}")
    for h in sorted(headline, key=lambda h: -h["episode_mean_iou"])[:12]:
        print(f"  {h['arm_id']:26s} IoU={h['episode_mean_iou']:.3f} P={h['episode_mean_precision']:.3f} R={h['episode_mean_recall']:.3f} (n_ep={h['n_episodes']})")


if __name__ == "__main__":
    main()
