"""T20 #1 — T14 server-record replay of ground_target + TargetConfirm identity (verifier script).

Input: outputs/live_test/20260928_t14/server_constraints/run_0000/chunk_{k}.npz (seq = k+1).
AttentionPointCloud(PointCloud(attention_cloud), attention_values, raw_attention=attention_raw),
exclude_mask=support_mask, AG3SConfig defaults (serve_safe builds AG3SConfig.from_dict without a
clustering/attention/geometry section).  Replays ALL chunks 0..30 in order so the identity state
at seq 10 is the one built from seq 1.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, time
from pathlib import Path
import numpy as np

from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.types import PointCloud, AttentionPointCloud
from AG3S.ag3s.stages.target_grounding import ground_target, TargetConfirm

ROOT = Path(_ag3s_workspace_path(''))
REC = ROOT / "outputs/live_test/20260928_t14/server_constraints/run_0000"
FRAMES = ROOT / "outputs/live_test/20260928_t14/execute_ep1807/frames.jsonl"
OUT = ROOT / "outputs/verify/T20"
PARTIAL = ROOT / "AG3S/ag3s/docs/handoff/T20.verify.partial.json"
APPLE = np.array([0.564, 0.320, 0.850])
CRATE = np.array([0.517, -0.016, 0.880])

cfg = AG3SConfig()
objs = None
for line in open(FRAMES):
    r = json.loads(line)
    if r.get("kind") == "planning" and "object_poses" in r:
        objs = {k: np.asarray(v["pos"] if isinstance(v, dict) else v, float) for k, v in r["object_poses"].items()}
        break


def nearest_obj(c):
    if c is None:
        return None
    d = {k: float(np.linalg.norm(np.asarray(c) - v)) for k, v in objs.items()}
    k = min(d, key=d.get)
    return [k, d[k] * 1000]


def mm(a, b):
    return None if a is None else float(np.linalg.norm(np.asarray(a, float) - b) * 1000)


chunks = []
for k in range(31):
    z = np.load(REC / f"chunk_{k:05d}.npz", allow_pickle=True)
    s = json.loads(str(z["summary_json"]))
    chunks.append(dict(
        k=k, seq=k + 1, t_step=int(z["t_step"]),
        pts=z["attention_cloud"].astype(np.float64), filt=z["filtered_cloud"],
        att=z["attention_values"], raw=z["attention_raw"], sup=z["support_mask"],
        seeds=(z["seed_indices"] if "seed_indices" in z.files else np.zeros(0, np.int32)),
        live_centroid=(z["target_centroid"].tolist() if "target_centroid" in z.files and z["target_centroid"].size == 3 else None),
        live_npts=(int(z["target_points"].shape[0]) if "target_points" in z.files else 0),
        summary=s))


def run_frame(c, confirm):
    apc = AttentionPointCloud(PointCloud(c["pts"]), c["att"], raw_attention=c["raw"])
    t0 = time.perf_counter()
    g = ground_target(apc, cfg.clustering, seed_percentile=cfg.attention.seed_percentile,
                      seed_threshold=cfg.attention.seed_threshold, exclude_mask=c["sup"],
                      min_radius=cfg.geometry.min_radius, timestamp=float(c["t_step"]),
                      confirm=confirm)
    ms = (time.perf_counter() - t0) * 1000
    return g, ms


def fidelity():
    rows = []
    for c in chunks:
        g, ms = run_frame(c, None)
        s = c["summary"]
        seeds = np.asarray(g.seed_indices if g.seed_indices is not None else [], int)
        rows.append(dict(
            seq=c["seq"], chunk=c["k"],
            attention_cloud_equals_filtered_cloud=bool(np.array_equal(c["pts"].astype(np.float32), c["filt"])),
            n_points=int(len(c["pts"])),
            seeds_equal=bool(np.array_equal(np.sort(seeds), np.sort(c["seeds"]))),
            n_seeds_replay=int(seeds.size), n_seeds_live=int(c["seeds"].size),
            status_replay=str(g.status.value), status_live=s.get("grounding_status"),
            n_clusters_replay=len(g.clusters or ()), n_clusters_live=s.get("n_clusters"),
            best_score_replay=(None if g.best_score is None else float(g.best_score)),
            best_score_live=s.get("grounding_best_score"),
            rank1_centroid_replay=(None if g.target is None else [float(v) for v in g.target.centroid]),
            live_target_centroid=c["live_centroid"],
            live_target_to_apple_mm=mm(c["live_centroid"], APPLE),
            live_target_nearest=nearest_obj(c["live_centroid"]),
            live_notes=[n for n in s.get("notes", []) if n.startswith("target")],
            live_has_target=s.get("has_target"),
            clusters_replay=[dict(score=float(cl.target_score), n=int(cl.point_count),
                                  centroid=[float(v) for v in cl.centroid],
                                  to_apple_mm=mm(cl.centroid, APPLE), nearest=nearest_obj(cl.centroid))
                             for cl in (g.clusters or ())],
            ms=ms))
        print(f"fid seq {c['seq']:2d} seeds_eq={rows[-1]['seeds_equal']} st={rows[-1]['status_replay']}/{rows[-1]['status_live']} "
              f"ncl={rows[-1]['n_clusters_replay']}/{rows[-1]['n_clusters_live']} best={rows[-1]['best_score_replay']}/{rows[-1]['best_score_live']}", flush=True)
    return rows


def sweep(min_score, lost):
    confirm = TargetConfirm(cfg.clustering, switch_min_score=min_score, lost_frames=lost)
    rows = []
    for c in chunks:
        g, ms = run_frame(c, confirm)
        m = confirm.manipulated
        rec = None if m is None else m.record()
        dec = confirm.last
        leader = g.clusters[0] if g.clusters else None
        row = dict(
            kind="t14_replay", min_score=min_score, lost_frames=lost, seq=c["seq"], chunk=c["k"],
            grounding_status=str(g.status.value),
            decision_mode=(None if dec is None else dec.mode),
            decision_frame_matches=(dec is not None and m is not None and dec is confirm.last),
            streak=confirm.streak,
            manipulated=rec,
            manip_to_apple_mm=(None if rec is None else mm(rec["centroid"], APPLE)),
            manip_to_crate_mm=(None if rec is None else mm(rec["centroid"], CRATE)),
            manip_nearest=(None if rec is None else nearest_obj(rec["centroid"])),
            target_present=g.target is not None,
            target_centroid=(None if g.target is None else [float(v) for v in g.target.centroid]),
            target_to_apple_mm=(None if g.target is None else mm(g.target.centroid, APPLE)),
            target_occluded_flag=(None if g.target is None else float(g.target.metrics.get("manipulated_occluded", 0.0))),
            n_clusters=len(g.clusters or ()),
            leader_score=(None if leader is None else float(leader.target_score)),
            leader_to_apple_mm=(None if leader is None else mm(leader.centroid, APPLE)),
            leader_nearest=(None if leader is None else nearest_obj(leader.centroid)),
            ms=ms)
        rows.append(row)
        with open(PARTIAL, "a") as f:
            f.write(json.dumps(row) + "\n")
    return rows


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "fidelity":
        rows = fidelity()
        json.dump(rows, open(OUT / "t14_fidelity.json", "w"), indent=1)
    else:
        ms_ = float(sys.argv[2]); lost = None if sys.argv[3] == "None" else int(sys.argv[3])
        rows = sweep(ms_, lost)
        json.dump(rows, open(OUT / f"t14_replay_min{ms_}_lost{lost}.json", "w"), indent=1)
        for r in rows:
            m = r["manipulated"] or {}
            print(f"min{ms_} lost{lost} seq {r['seq']:2d} {r['grounding_status']:9s} mode={r['decision_mode']} "
                  f"id={m.get('id')} st={m.get('state')} age={m.get('age_frames')} ->apple={r['manip_to_apple_mm']} "
                  f"lead={r['leader_score']} {r['leader_nearest']}", flush=True)
