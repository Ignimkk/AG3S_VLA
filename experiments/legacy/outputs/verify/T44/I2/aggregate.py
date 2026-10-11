"""Aggregate raw/{wt,head}_*.json (+ .npz) into agg.json (V-I2-2/3/4 numbers)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import glob
import json
import os

import numpy as np

R = _ag3s_workspace_path('outputs/verify/T44/I2/raw')
rows = []
tol_deg = lambda a, b: float(np.degrees(np.abs(a - b).max()))

runs = sorted(os.path.basename(p)[3:-5] for p in glob.glob(f"{R}/wt_*.json"))
agg = {"runs": [], "v2": {}, "v3": {}, "v4": {}}
v2 = dict(n_chunks=0, n_runs=0, compared={}, )
cmp_pairs = {  # name -> (A source, A arm, B source, B arm)
    "head_off_vs_wt_off": ("head", "off", "wt", "off"),
    "head_off_vs_wt_offR": ("head", "off", "wt", "offR"),
    "head_sw_vs_wt_sw": ("head", "sw", "wt", "sw"),
    "head_sw_vs_wt_swR": ("head", "sw", "wt", "swR"),
}
for n in cmp_pairs:
    v2["compared"][n] = dict(n_chunks=0, own_differs=0, traj_differs=0, status_differs=0,
                             hold_differs=0, max_joint_deg=0.0, n_collision_rows_differs=0)
per_chunk = []
for tag in runs:
    have_head = os.path.exists(f"{R}/head_{tag}.json")
    wt = json.load(open(f"{R}/wt_{tag}.json"))
    wz = np.load(f"{R}/wt_{tag}.npz")
    hd = json.load(open(f"{R}/head_{tag}.json")) if have_head else None
    hz = np.load(f"{R}/head_{tag}.npz") if have_head else None
    hmap = {c["seq"]: c for c in hd["chunks"]} if hd else {}
    run_rec = dict(tag=tag, n_chunks=0, has_head=have_head)
    for c in wt["chunks"]:
        if "skipped" in c:
            continue
        s = c["seq"]
        run_rec["n_chunks"] += 1
        A = c["arms"]
        rec = dict(run=tag, seq=s, held=c["held"], has_target=c["has_target"],
                   n_contact_rows=c["n_contact_rows"])
        # --- V-I2-2
        if have_head and s in hmap and "arms" in hmap[s]:
            for name, (sa, aa, sb, ab) in cmp_pairs.items():
                if aa not in hmap[s]["arms"]:
                    continue
                d = v2["compared"][name]
                d["n_chunks"] += 1
                ha, wb = hmap[s]["arms"][aa], A[ab]
                ownA, ownB = hz[f"s{s}_{aa}_own"], wz[f"s{s}_{ab}_own"]
                trjA, trjB = hz[f"s{s}_{aa}_traj"], wz[f"s{s}_{ab}_traj"]
                d["own_differs"] += int(not np.array_equal(ownA, ownB))
                d["traj_differs"] += int(not np.array_equal(trjA, trjB))
                d["status_differs"] += int(ha["status"] != wb["status"])
                d["hold_differs"] += int(ha["hold_candidate"] != wb["hold_candidate"])
                d["n_collision_rows_differs"] += int(ha["n_collision_rows"] != wb["n_collision_rows"])
                if ownA.shape == ownB.shape:
                    d["max_joint_deg"] = max(d["max_joint_deg"], tol_deg(ownA, ownB))
        # --- V-I2-3 (strict vs offR, V11AB config)
        o, st = A["offR"], A["strict"]
        ownO, ownS = wz[f"s{s}_offR_own"], wz[f"s{s}_strict_own"]
        rec["strict_own_differs"] = not np.array_equal(ownO, ownS)
        rec["strict_joint_deg_max"] = tol_deg(ownO, ownS)
        rec["strict_status"] = (o["status"], st["status"])
        rec["strict_hold"] = (o["hold_candidate"], st["hold_candidate"])
        rec["n_tail_target_unauthorized"] = (st["lookahead"] or {}).get("n_tail_target_unauthorized")
        rec["n_tail_dropped"] = ((o["lookahead"] or {}).get("n_tail_dropped"),
                                 (st["lookahead"] or {}).get("n_tail_dropped"))
        pr = c.get("probes") or {}
        for k in ("lookahead_reference", "lookahead_off_refined"):
            if k in pr:
                rec[k] = {kk: pr[k][kk] for kk in ("n_rows_changed", "reverse_changed", "kept_off",
                                                   "kept_strict", "changed_links")}
        # --- V-I2-4
        ro, rs = A["swR"], A["swS"]
        rec["sw_off_excluded"] = ro["path_check"]["target_rows_excluded"]
        rec["sw_strict_excluded"] = rs["path_check"]["target_rows_excluded"]
        rec["sw_strict_unauth_kept"] = rs["path_check"]["target_rows_unauthorized_kept"]
        rec["sw_own_differs"] = not np.array_equal(wz[f"s{s}_swR_own"], wz[f"s{s}_swS_own"])
        rec["sw_joint_deg_max"] = tol_deg(wz[f"s{s}_swR_own"], wz[f"s{s}_swS_own"])
        rec["sw_status"] = (ro["status"], rs["status"])
        rec["sw_hold"] = (ro["hold_candidate"], rs["hold_candidate"])
        rec["sw_path_clear_mm"] = (ro["path_check"]["clearance_m"], rs["path_check"]["clearance_m"])
        if "path_off" in pr:
            rec["path_same_traj"] = dict(off=pr["path_off"], strict=pr["path_strict"])
        rec["sw_off_links"] = ro["path_target_links"]
        per_chunk.append(rec)
    agg["runs"].append(run_rec)

v2["n_runs"] = len(runs)
v2["n_chunks"] = len(per_chunk)
agg["v2"] = v2
agg["per_chunk_file"] = "per_chunk.json"

P = per_chunk
tail_u = [r["n_tail_target_unauthorized"] or 0 for r in P]
lk_ref = [r.get("lookahead_reference", {}).get("n_rows_changed", 0) for r in P]
lk_off = [r.get("lookahead_off_refined", {}).get("n_rows_changed", 0) for r in P]
agg["v3"] = dict(
    n_chunks=len(P),
    n_chunks_with_contact_rows=sum(1 for r in P if r["n_contact_rows"]),
    n_chunks_tail_unauthorized_gt0=sum(1 for t in tail_u if t > 0),
    tail_unauthorized_rows_total=int(sum(tail_u)),
    lookahead_rows_changed_reference_total=int(sum(lk_ref)),
    lookahead_rows_changed_reference_chunks=sum(1 for t in lk_ref if t > 0),
    lookahead_rows_changed_offrefined_total=int(sum(lk_off)),
    lookahead_rows_changed_offrefined_chunks=sum(1 for t in lk_off if t > 0),
    refined_chunks_changed=sum(1 for r in P if r["strict_own_differs"]),
    refined_joint_deg_max=max([r["strict_joint_deg_max"] for r in P] or [0.0]),
    status_changes=sum(1 for r in P if r["strict_status"][0] != r["strict_status"][1]),
    hold_changes=sum(1 for r in P if r["strict_hold"][0] != r["strict_hold"][1]),
    hold_candidate_off=sum(1 for r in P if r["strict_hold"][0]),
    hold_candidate_strict=sum(1 for r in P if r["strict_hold"][1]),
)
unk = [r["sw_strict_unauth_kept"] or 0 for r in P]
agg["v4"] = dict(
    n_chunks=len(P),
    off_target_rows_excluded_total=int(sum(r["sw_off_excluded"] for r in P)),
    off_chunks_with_excluded=sum(1 for r in P if r["sw_off_excluded"] > 0),
    strict_target_rows_excluded_total=int(sum(r["sw_strict_excluded"] for r in P)),
    strict_chunks_with_excluded=sum(1 for r in P if r["sw_strict_excluded"] > 0),
    strict_unauthorized_kept_total=int(sum(unk)),
    strict_chunks_unauthorized_kept_gt0=sum(1 for u in unk if u > 0),
    refined_chunks_changed=sum(1 for r in P if r["sw_own_differs"]),
    refined_joint_deg_max=max([r["sw_joint_deg_max"] for r in P] or [0.0]),
    status_changes=sum(1 for r in P if r["sw_status"][0] != r["sw_status"][1]),
    hold_changes=sum(1 for r in P if r["sw_hold"][0] != r["sw_hold"][1]),
)
same = [r for r in P if "path_same_traj" in r]
agg["v4"]["same_traj"] = dict(
    n_chunks=len(same),
    off_excluded_total=int(sum(r["path_same_traj"]["off"]["target_rows_excluded"] for r in same)),
    strict_excluded_total=int(sum(r["path_same_traj"]["strict"]["target_rows_excluded"] for r in same)),
    strict_unauth_kept_total=int(sum(r["path_same_traj"]["strict"]["target_rows_unauthorized_kept"] or 0 for r in same)),
    new_path_violations=sum(1 for r in same if r["path_same_traj"]["strict"]["violation"]
                            and not r["path_same_traj"]["off"]["violation"]),
    path_violations_off=sum(1 for r in same if r["path_same_traj"]["off"]["violation"]),
    path_violations_strict=sum(1 for r in same if r["path_same_traj"]["strict"]["violation"]),
)
json.dump(agg, open(_ag3s_workspace_path('outputs/verify/T44/I2/agg.json'), "w"), indent=1)
json.dump(per_chunk, open(_ag3s_workspace_path('outputs/verify/T44/I2/per_chunk.json'), "w"))
print(json.dumps({k: agg[k] for k in ("v2", "v3", "v4")}, indent=1))
