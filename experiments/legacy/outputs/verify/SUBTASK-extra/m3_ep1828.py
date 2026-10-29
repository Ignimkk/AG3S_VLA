"""SUBTASK-extra M3 — ep1828 s18281, chunks 14..35 (k = chunk_index - episode_base = t_step - 1),
gate off (T39 E3b, server 8226, chunk_index 225..299) vs gate on (SUBTASK-d E3bG, server 8232, 75..149).
Server summary_json: grasp state, manipulated id/centroid, decision mode, exclusion, subtask (on).
Client planning record seq = k+1 (t_step = 8k): MuJoCo object_poses (world frame). The robot base
body sits at world origin when qpos[0:3] (joint_x, joint_y, joint_th) = 0; that is checked per chunk
and the centroid (robot base frame) is compared with object_poses directly. Read-only."""
import glob, json
import numpy as np

COND = {
    "off_T39_E3b": dict(server=glob.glob("outputs/verify/T39/runs/E3b/server_8226_*/server_constraints/run_0000")[0],
                        base=225, client="outputs/verify/T39/runs/E3b/E3b_ep1828_s18281/frames.jsonl"),
    "on_SUBTASKd_E3bG": dict(server=glob.glob("outputs/verify/SUBTASK-d/runs/E3bG/server_8232_*/server_constraints/run_0000")[0],
                             base=75, client="outputs/verify/SUBTASK-d/runs/E3bG/E3bG_ep1828_s18281/frames.jsonl"),
}
K = range(14, 36)

def r(x, n=4):
    return None if x is None else [round(float(v), n) for v in x]

def nearest(c, poses):
    best = None
    for name, p in poses.items():
        d = float(np.linalg.norm(np.asarray(c) - np.asarray(p["pos"])))
        dxy = float(np.linalg.norm(np.asarray(c[:2]) - np.asarray(p["pos"][:2])))
        if best is None or d < best[1]:
            best = (name, d, dxy)
    return dict(name=best[0], dist_m=round(best[1], 4), dist_xy_m=round(best[2], 4))

out = {}
for cname, cfg in COND.items():
    L = [json.loads(l) for l in open(cfg["client"])]
    P = {p["seq"]: p for p in L if p["kind"] == "planning"}
    rows = []
    seeds = set()
    # whole-episode checks: seed and t_step contiguity
    for k in range(75):
        with np.load(f'{cfg["server"]}/chunk_{cfg["base"]+k:05d}.npz', allow_pickle=False) as z:
            s = json.loads(str(z["summary_json"]))
        assert int(s["t_step"]) == k + 1, (cname, k, s["t_step"])
        seeds.add(json.dumps(s.get("policy_seed"), sort_keys=True))
        if k not in K:
            continue
        g = s.get("grasp") or {}
        m = s.get("manipulated") or {}
        dec = (s.get("admissibility") or {}).get("decision") or {}
        ex = s.get("exclusion") or {}
        sub = s.get("subtask")
        p = P.get(k + 1) or {}
        poses = p.get("object_poses") or {}
        q = p.get("qpos") or [None] * 3
        row = dict(k=k, chunk_index=int(s["chunk_index"]), client_t_step=p.get("t_step"),
                   grasp_state=g.get("state"), grasp_state_before=g.get("state_before"),
                   grasp_transition=g.get("transition"),
                   manipulated_id=m.get("id"), manipulated_state=m.get("state"),
                   switched_from=m.get("switched_from"), centroid=r(m.get("centroid")),
                   decision_mode=dec.get("mode"), decision_rank=dec.get("rank"), decision_reason=dec.get("reason"),
                   exclusion_active=ex.get("active"), exclusion_source=ex.get("source"),
                   exclusion_manipulated_id=ex.get("manipulated_id"),
                   destination_id=(s.get("destination") or {}).get("id"),
                   base_qpos_xyth=r(q[:3], 6),
                   apple_pos=r((poses.get("apple") or {}).get("pos")))
        if sub is not None:
            row.update(subtask_label=sub.get("label"), subtask_argmax=sub.get("argmax"),
                       subtask_p=({kk: round(v, 4) for kk, v in (sub.get("p") or {}).items()}),
                       subtask_blocked=sub.get("blocked"), subtask_would_block=sub.get("would_block"),
                       subtask_released=sub.get("released"), subtask_placed_seen=sub.get("placed_seen"),
                       subtask_streak=sub.get("streak"))
        if m.get("centroid") is not None and poses:
            row["nearest_object"] = nearest(m["centroid"], poses)
        rows.append(row)
    out[cname] = dict(server_dir=cfg["server"], client_frames=cfg["client"], chunk_base=cfg["base"],
                      policy_seeds=sorted(map(str, seeds)), rows=rows)

# chunk 30 (gate off) — new target id 1 vs previous target id 0
off = {row["k"]: row for row in out["off_T39_E3b"]["rows"]}
c30 = off[30]
prev = max((row for row in off.values() if row["k"] < 30 and row["manipulated_id"] == 0), key=lambda x: x["k"])
L = [json.loads(l) for l in open(COND["off_T39_E3b"]["client"])]
P = {p["seq"]: p for p in L if p["kind"] == "planning"}
poses30 = P[31]["object_poses"]
c1 = np.asarray(c30["centroid"]); c0 = np.asarray(prev["centroid"])
apple = np.asarray(poses30["apple"]["pos"])
crate = np.asarray(poses30["crate"]["pos"])
out["chunk30_off"] = dict(
    new_id=c30["manipulated_id"], switched_from=c30["switched_from"], decision_mode=c30["decision_mode"],
    centroid_id1=r(c1), prev_id0_chunk=prev["k"], centroid_id0=r(c0),
    dist_id1_id0_m=round(float(np.linalg.norm(c1 - c0)), 4),
    dist_id1_id0_xy_m=round(float(np.linalg.norm(c1[:2] - c0[:2])), 4),
    client_planning_seq=31, client_t_step=P[31]["t_step"], base_qpos_xyth=r(P[31]["qpos"][:3], 6),
    apple_pos=r(apple), crate_pos=r(crate),
    apple_to_crate_xy_m=round(float(np.linalg.norm(apple[:2] - crate[:2])), 4),
    dist_id0_centroid_to_apple_m=round(float(np.linalg.norm(c0 - apple)), 4),
    dist_id1_centroid_to_apple_m=round(float(np.linalg.norm(c1 - apple)), 4),
    all_objects_dist_to_id1_m={n: round(float(np.linalg.norm(c1 - np.asarray(p["pos"]))), 4) for n, p in poses30.items()},
    all_objects_dist_to_id0_m={n: round(float(np.linalg.norm(c0 - np.asarray(p["pos"]))), 4) for n, p in poses30.items()},
    nearest_to_id1=c30.get("nearest_object"),
    object_poses=poses30,
)
# apple trajectory over the whole episode for both conditions (MuJoCo truth)
for cname, cfg in COND.items():
    L = [json.loads(l) for l in open(cfg["client"])]
    PP = [p for p in L if p["kind"] == "planning"]
    a = np.array([p["object_poses"]["apple"]["pos"] for p in PP])
    cr = np.array([p["object_poses"]["crate"]["pos"] for p in PP])
    out[cname]["apple_episode"] = dict(first=r(a[0]), last=r(a[-1]),
        max_disp_from_start_m=round(float(np.linalg.norm(a - a[0], axis=1).max()), 4),
        last_xy_to_crate_m=round(float(np.linalg.norm(a[-1, :2] - cr[-1, :2])), 4),
        max_abs_base_qpos_xyth=round(float(np.abs(np.array([p["qpos"][:3] for p in PP])).max()), 6))

# chunk 29 (gate off) — the chunk whose decision.mode == "switch" (t_step 30, chunk_index 254)
c29 = off[29]
poses29 = P[30]["object_poses"]
c1b = np.asarray(c29["centroid"]); apple29 = np.asarray(poses29["apple"]["pos"])
out["chunk29_off_switch"] = dict(
    new_id=c29["manipulated_id"], switched_from=c29["switched_from"], decision_mode=c29["decision_mode"],
    chunk_index=c29["chunk_index"], server_t_step=30, centroid_id1=r(c1b), prev_id0_chunk=28, centroid_id0=r(c0),
    dist_id1_id0_m=round(float(np.linalg.norm(c1b - c0)), 4),
    client_planning_seq=30, client_t_step=P[30]["t_step"], apple_pos=r(apple29),
    all_objects_dist_to_id1_m={n: round(float(np.linalg.norm(c1b - np.asarray(q["pos"]))), 4) for n, q in poses29.items()},
    nearest_to_id1=c29.get("nearest_object"))
# whole-episode grasp-state / manipulated-id transitions and apple displacement from start
for cname, cfg in COND.items():
    L = [json.loads(l) for l in open(cfg["client"])]
    PP = {p["seq"]: p for p in L if p["kind"] == "planning"}
    a0 = np.asarray(PP[1]["object_poses"]["apple"]["pos"])
    tr, prev_state, prev_id, ad = [], None, None, {}
    for k in range(75):
        with np.load(f'{cfg["server"]}/chunk_{cfg["base"]+k:05d}.npz', allow_pickle=False) as z:
            s = json.loads(str(z["summary_json"]))
        st = (s.get("grasp") or {}).get("state"); mid = (s.get("manipulated") or {}).get("id")
        mode = ((s.get("admissibility") or {}).get("decision") or {}).get("mode")
        if st != prev_state or mid != prev_id:
            tr.append(dict(k=k, grasp_state=st, manipulated_id=mid, decision_mode=mode,
                           subtask_label=(s.get("subtask") or {}).get("label")))
        prev_state, prev_id = st, mid
        if k + 1 in PP:
            ad[k] = round(float(np.linalg.norm(np.asarray(PP[k + 1]["object_poses"]["apple"]["pos"]) - a0)), 4)
    out[cname]["transitions"] = tr
    out[cname]["apple_disp_from_start_m_by_k"] = ad

json.dump(out, open("outputs/verify/SUBTASK-extra/m3_ep1828.json", "w"), indent=1)
hdr = "k  | off: state  id  mode   excl  nearest(dist)        || on: state  id  mode   excl  label/argmax  nearest(dist)"
print(hdr)
on = {row["k"]: row for row in out["on_SUBTASKd_E3bG"]["rows"]}
for k in K:
    a, b = off[k], on[k]
    fa = lambda x: f'{x["grasp_state"]:<9} {str(x["manipulated_id"]):>3} {str(x["decision_mode"]):<7} {str(x["exclusion_active"]):<5} {(x.get("nearest_object") or {}).get("name","-")}({(x.get("nearest_object") or {}).get("dist_m","-")})'
    print(f'{k:2d} | {fa(a):<52} || {fa(b)}  {b.get("subtask_label")}/{b.get("subtask_argmax")}')
print(json.dumps({kk: v for kk, v in out["chunk30_off"].items() if kk != "object_poses"}, indent=1))
for c in COND: print(c, out[c]["policy_seeds"], json.dumps(out[c]["apple_episode"]))
print(json.dumps(out["chunk29_off_switch"], indent=1))
for c in COND:
    print(c, "transitions:"); [print("  ", json.dumps(t)) for t in out[c]["transitions"]]
    print(c, "apple disp k14..35:", {k: out[c]["apple_disp_from_start_m_by_k"][k] for k in K})
