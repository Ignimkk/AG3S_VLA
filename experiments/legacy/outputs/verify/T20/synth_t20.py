"""T20 #2/#3 — synthetic sequences through ground_target + TargetConfirm (verifier script).

Blobs reuse the T20 test fixture lattice (tests/ag3s/test_manipulated_identity._lattice, 27 pts,
8 mm spacing, CFG = ClusteringConfig(min_points=10), seed_threshold 0.005). One extra point far
from every blob carries the raw-attention peak and is support-excluded, so the geometric term of
every cluster is small and target_score ~ w_a * attention (lets a lone cluster score 0.015 as in
T14 seq 12). Scores reported are the ones ground_target computed.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, itertools
from pathlib import Path
import numpy as np

from AG3S.ag3s.stages.target_grounding import ground_target, TargetConfirm
from AG3S.ag3s.types import AttentionPointCloud, PointCloud
from tests.ag3s.test_manipulated_identity import _lattice, CFG, APPLE, CRATE

OUT = Path(_ag3s_workspace_path('outputs/verify/T20'))
PARTIAL = Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T20.verify.partial.json'))
PEAK = np.array([0.0, 0.0, 2.0])
FAR = np.array([0.0, 0.1, 2.0])


def frame(*blobs):
    """blobs: (centre, attention level). Always two excluded far points (peak 1.0, other 0.0)."""
    pts, att, exc = [], [], []
    for c, a in blobs:
        b = _lattice(c)
        pts.append(b); att.append(np.float32(a) + np.linspace(0.0, 1e-4, len(b), dtype=np.float32))
        exc.append(np.zeros(len(b), bool))
    pts.append(np.stack([PEAK, FAR])); att.append(np.array([0.0, 0.0], np.float32)); exc.append(np.ones(2, bool))
    P = np.concatenate(pts); A = np.concatenate(att); E = np.concatenate(exc)
    raw = A.copy(); raw[-2] = 10.0
    return AttentionPointCloud(PointCloud(P), A, raw), E


def ground(fr, confirm, ts):
    apc, exc = fr
    return ground_target(apc, CFG, seed_threshold=0.005, exclude_mask=exc, confirm=confirm, timestamp=ts)


def run(seq_frames, ms, lost, name, extra):
    confirm = TargetConfirm(CFG, switch_min_score=ms, lost_frames=lost)
    rows = []
    for i, (phase, fr) in enumerate(seq_frames):
        g = ground(fr, confirm, float(i))
        m = confirm.manipulated
        rec = None if m is None else m.record()
        lead = g.clusters[0] if g.clusters else None
        row = dict(kind="synthetic", sequence=name, min_score=ms, lost_frames=lost, frame=i, phase=phase,
                   **extra,
                   grounding_status=str(g.status.value),
                   decision_mode=(None if confirm.last is None else confirm.last.mode),
                   streak=confirm.streak, manipulated=rec,
                   manip_to_apple_mm=(None if rec is None else float(np.linalg.norm(np.asarray(rec["centroid"]) - APPLE) * 1000)),
                   manip_to_crate_mm=(None if rec is None else float(np.linalg.norm(np.asarray(rec["centroid"]) - CRATE) * 1000)),
                   target_present=g.target is not None,
                   leader_score=(None if lead is None else float(lead.target_score)),
                   scores=[(float(c.target_score), [round(float(v), 3) for v in c.centroid]) for c in (g.clusters or ())])
        rows.append(row)
        with open(PARTIAL, "a") as f:
            f.write(json.dumps(row) + "\n")
    return rows


def main():
    CRATE_LEVELS = [0.02, 0.07, 0.075, 0.143, 0.43, 1.0]   # calibrated below; actual scores recorded per frame
    rows = []
    for ms, lost in itertools.product([0.0, 0.05, 0.1, 0.3], [None, 10, 20]):
        pre = [("apple_visible", frame((APPLE, 0.8), (CRATE, 0.02)))] * 3
        for Lc in CRATE_LEVELS:
            # A: attention moves to the crate, apple still in view at low attention
            seqA = pre + [("attn_crate_apple_visible", frame((APPLE, 0.05), (CRATE, Lc)))] * 10
            rows += run(seqA, ms, lost, "A_attn_to_crate_apple_visible", dict(crate_level=Lc))
            # B: attention moves to the crate, apple not in the cloud (occluded / erased)
            seqB = pre + [("attn_crate_apple_absent", frame((CRATE, Lc)))] * 10
            rows += run(seqB, ms, lost, "B_attn_to_crate_apple_absent", dict(crate_level=Lc))
        for K, off_mm, occl in itertools.product([5, 15, 25], [0, 30, 80], ["crate_low", "no_cluster"]):
            occ_frame = frame((CRATE, 0.02)) if occl == "crate_low" else frame()
            back = APPLE + np.array([off_mm / 1000.0, 0, 0])
            seqC = pre + [("occluded", occ_frame)] * K + [("apple_back", frame((back, 0.8), (CRATE, 0.02)))] * 3
            rows += run(seqC, ms, lost, "C_occlusion_then_reobserve", dict(K=K, reobserve_offset_mm=off_mm, occluder=occl))
    json.dump(rows, open(OUT / "synth_rows.json", "w"))


if __name__ == "__main__":
    if sys.argv[1:] == ["calib"]:
        for L in [0.02, 0.07, 0.075, 0.143, 0.43, 1.0]:
            g = ground(frame((CRATE, L)), None, 0.0); h = ground(frame((APPLE, 0.8), (CRATE, L)), None, 0.0)
            k = ground(frame((APPLE, 0.05), (CRATE, L)), None, 0.0)
            print(L, g.status.value, [round(c.target_score, 4) for c in g.clusters], [round(c.target_score, 4) for c in h.clusters], [round(c.target_score, 4) for c in k.clusters])
        print("empty", ground(frame(), None, 0.0).status.value)
    else:
        main()
