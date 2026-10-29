"""T44 pre-work (verifier): fix the E2 episode / frame list BEFORE any E2 measurement. CPU, stdlib + numpy only.
usage: select_e2.py OUT_JSON
Inputs (read only): outputs/verify/T43/LB/runs/E0/*.check.json|.robust.json|<run>/frames.jsonl, LB/agg_final.json, dataset randomized_episodes.jsonl.
Order of rejection reasons is fixed (a run is counted under the first reason that applies)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, glob, os, sys, collections
import numpy as np
LB = _ag3s_workspace_path('outputs/verify/T43/LB'); RUNS = LB + "/runs/E0/"
META = _ag3s_workspace_path('legacy/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1/meta/randomized_episodes.jsonl')
FRUITS = ("apple", "orange", "pear", "banana"); ARMS = ("left", "right")
NN_STRICT_M = 0.10; N_EP = 12; N_FR = 4; STRIDE = 8  # control steps per planning row (600 control / 75 planning)
eps = {}
for l in open(META):
    j = json.loads(l); eps[j["episode_index"]] = j

def nn_of(poses, tf):
    d = {k: float(np.hypot(v[0] - poses[tf][0], v[1] - poses[tf][1])) for k, v in poses.items() if k != tf}
    k = min(d, key=d.get)
    return k, d[k], d

agg = json.load(open(LB + "/agg_final.json"))
agg_ok = {}
for e, v in agg["episodes"].items():
    for s in v["seeds"]:
        agg_ok[s["seed"]] = bool(s["E0_ok_B"])

reasons = collections.Counter(); runs = []
for rj in sorted(glob.glob(RUNS + "E0_ep*_s*.robust.json")):
    name = os.path.basename(rj)[:-len(".robust.json")]
    ep = int(name.split("_ep")[1].split("_")[0]); seed = int(name.split("_s")[-1])
    ck = json.load(open(RUNS + name + ".check.json")); rb = json.load(open(rj)); e = eps[ep]
    nn_name, nn_d, _ = nn_of(e["initial_fruit_poses"], e["target_fruit"])
    r = dict(name=name, episode=ep, seed=seed, split=e["split"], fruit=e["target_fruit"], arm=e["used_arm"], nn_fruit=nn_name, nn_xy_m=nn_d,
             success=bool(rb["success"]), robust_grasp=bool(rb["robust_grasp"]), close_cmd_t=rb["close_cmd_t"])
    r["complete_valid"] = bool(ck["complete"] and ck["checks"]["valid_no_safe_error"] and not ck["checks"]["conn_error"])
    r["ok"] = r["success"] and r["robust_grasp"]
    r["agg_final_E0_ok_B"] = agg_ok.get(seed)
    if not r["complete_valid"]: why = "incomplete_or_invalid_run"
    elif r["split"] != "train" or ep >= 1600: why = "not_train_split"
    elif not r["success"]: why = "E0_not_success"
    elif not r["robust_grasp"]: why = "success_but_not_robust_grasp"
    else: why = "ok_E0_run"
    r["stage1"] = why; reasons[why] += 1; runs.append(r)
assert all(r["agg_final_E0_ok_B"] in (None, r["ok"]) for r in runs), "agg_final disagrees with robust.json"
ok_runs = [r for r in runs if r["stage1"] == "ok_E0_run"]
# one run per episode: lowest seed among ok runs
by_ep = {}
for r in sorted(ok_runs, key=lambda r: r["seed"]):
    by_ep.setdefault(r["episode"], r)
ep_rows = list(by_ep.values())
strict_pool = [r for r in ep_rows if r["nn_xy_m"] < NN_STRICT_M]
reasons2 = {"ok_E0_runs": len(ok_runs), "ok_E0_episodes": len(ep_rows), "ok_episodes_nn_ge_0.10": len(ep_rows) - len(strict_pool),
            "ok_episodes_nn_lt_0.10": len(strict_pool)}
train_all = [(nn_of(e["initial_fruit_poses"], e["target_fruit"])[1], i) for i, e in eps.items() if e["split"] == "train"]
dataset_note = {"n_train_episodes": len(train_all), "n_train_nn_lt_0.10": sum(1 for d, _ in train_all if d < NN_STRICT_M),
                "LB_all_tried_episodes_min_nn_xy_m": min(r["nn_xy_m"] for r in runs), "LB_all_tried_episodes_max_nn_xy_m": max(r["nn_xy_m"] for r in runs),
                "LB_n_runs_nn_lt_0.10": sum(1 for r in runs if r["nn_xy_m"] < NN_STRICT_M),
                "why": "LB candidates_lb.json ranks by -crowd_score (largest = LEAST crowded); every LB episode therefore has nearest-other-fruit xy distance > 0.108 m"}

def pick(pool):
    groups = {f"{f}_{a}": sorted([r for r in pool if r["fruit"] == f and r["arm"] == a], key=lambda r: (r["nn_xy_m"], r["episode"])) for f in FRUITS for a in ARMS}
    out = []; i = 0
    while len(out) < N_EP and any(groups.values()):
        for g in groups:
            if groups[g] and len(out) < N_EP:
                out.append(groups[g].pop(0))
    return out

def frames_for(r):
    rd = RUNS + r["name"]
    rows = [json.loads(l) for l in open(rd + "/frames.jsonl") if l.strip()]
    pl = [x for x in rows if x["kind"] == "planning"]; ct = [x for x in rows if x["kind"] == "control"]
    ai = 0 if r["arm"] == "left" else 1
    tc = next(c["t_step"] for c in ct if c["applied_ctrl"]["gripper"][ai] < 0.5)
    assert abs(tc - r["close_cmd_t"][0]) <= 2, (r["name"], tc, r["close_cmd_t"])   # robust.json uses a slightly different threshold (differs by <= 1 step)
    sc = tc // STRIDE
    assert all(p["t_step"] == STRIDE * i for i, p in enumerate(pl)), r["name"]   # planning row i <-> t_step 8 i (E0 'seq' field == t_step)
    by_seq = dict(enumerate(pl))
    seqs = [s for s in range(sc - N_FR + 1, sc + 1)]
    assert all(s in by_seq and not by_seq[s]["duplicate_sequence"] for s in seqs), r["name"]
    e = eps[r["episode"]]; tf = e["target_fruit"]
    fr = []
    for s in seqs:
        p = by_seq[s]; op = p["object_poses"]
        nn_name, nn_d, _ = nn_of({k: v["pos"] for k, v in op.items() if k in FRUITS}, tf)
        fr.append(dict(planning_index=s, seq_server_equiv=s + 1, server_chunk_k=s, seq_in_E0_frames_jsonl=p["seq"], t_step=p["t_step"], steps_before_first_close_cmd=tc - p["t_step"], target_pos=[round(x, 5) for x in op[tf]["pos"]],
                       nearest_other_fruit=nn_name, nearest_other_xy_m=round(nn_d, 5)))
    # frame 0 object poses vs dataset initial poses (record sanity)
    p0 = by_seq[0]["object_poses"]
    d0 = max(float(np.max(np.abs(np.array(p0[k]["pos"]) - np.array(e["initial_fruit_poses"][k][:3])))) for k in FRUITS)
    return tc, sc, fr, d0

def entry(r, why):
    tc, sc, fr, d0 = frames_for(r)
    e = eps[r["episode"]]
    return dict(episode=r["episode"], seed=r["seed"], split=r["split"], run_dir=RUNS + r["name"], frames_jsonl=RUNS + r["name"] + "/frames.jsonl",
                target_fruit=r["fruit"], used_arm=r["arm"], group=f"{r['fruit']}_{r['arm']}", nearest_distractor=r["nn_fruit"], nearest_distractor_xy_m_initial=round(r["nn_xy_m"], 5),
                meets_strict_nn_lt_0_10=bool(r["nn_xy_m"] < NN_STRICT_M), first_close_cmd_t=tc, robust_close_cmd_t0=r["close_cmd_t"][0], close_planning_index=sc, frames=fr,
                record_vs_dataset_initial_pos_max_abs_diff_m=round(d0, 5), layout_index=e.get("layout_index"), reason_included=why)

sel_strict = pick(strict_pool)
sel_fb = pick(ep_rows)
E2 = dict(
    definitions=("T44 pre-work, verifier, 2026-10-09. List fixed BEFORE any E2 measurement. Candidate = completed, valid, TRAIN-split (episode < 1600) E0 run in "
                 "outputs/verify/T43/LB/runs/E0 with success and robust_grasp (= LB 'E0_ok_B'; cross-checked against LB/agg_final.json). One run per episode (lowest seed). "
                 "Neighbour distance = xy distance from the target fruit centre to the nearest OTHER fruit centre in meta initial_fruit_poses."),
    frame_rule=("closing command = first control row t_c whose active-arm gripper command < 0.5 (robust.json close_cmd_t[0] differs by at most 1 step, asserted <= 2). "
                "s_c = t_c // 8 = the planning row (0-based index i, t_step = 8 i) whose chunk first contains that command. Frames = the 4 planning rows i = s_c-3 .. s_c "
                "(the last four planning rows up to and including the one that issues the closing chunk; 8 control steps each, i.e. 24-31 steps = ~1.6-2.1 s of approach to pre-grasp). "
                "In the E0 frames.jsonl the planning 'seq' field equals t_step (0, 8, 16, ...); a server (TO) run numbers seq 1, 2, ... so planning_index i <-> seq i+1 <-> server chunk_k = i. E0 has no server chunk."),
    episode_selection_rule=(f"round-robin over the 8 groups in order apple_left, apple_right, orange_left, orange_right, pear_left, pear_right, banana_left, banana_right; inside a group sort by "
                            f"(nearest-other-fruit xy distance ascending, episode ascending); take {N_EP} episodes, {N_FR} frames each"),
    strict_criterion=dict(nn_xy_lt_m=NN_STRICT_M, n_candidate_episodes=len(strict_pool), selected=[entry(r, "strict") for r in sel_strict],
                          status="UNSATISFIABLE in LB: no completed E0 run has a neighbour closer than 0.10 m"),
    fallback_for_lead_decision=dict(
        description=("NOT the T44.task.md criterion. Same candidate pool and frame rule, nn < 0.10 m dropped; the 12 nearest-neighbour episodes by the round-robin rule. "
                     "Neighbour xy distances are still 0.108-0.115 m. A strict list needs new E0 runs (policy server) on train episodes with nn < 0.10 m, or frames from another source."),
        selected=[entry(r, "fallback_nearest") for r in sel_fb]),
    rejection_counts_runs=dict(reasons), rejection_counts_episodes=reasons2, dataset_note=dataset_note,
    n_runs_total=len(runs), n_episodes_total=len({r['episode'] for r in runs}))
E2["fallback_for_lead_decision"]["n_episodes"] = len(sel_fb); E2["fallback_for_lead_decision"]["n_frames"] = sum(len(x["frames"]) for x in E2["fallback_for_lead_decision"]["selected"])
E2["fallback_for_lead_decision"]["groups_covered"] = sorted({x["group"] for x in E2["fallback_for_lead_decision"]["selected"]})
E2["fallback_for_lead_decision"]["nn_xy_m_range"] = [min(x["nearest_distractor_xy_m_initial"] for x in E2["fallback_for_lead_decision"]["selected"]), max(x["nearest_distractor_xy_m_initial"] for x in E2["fallback_for_lead_decision"]["selected"])]
json.dump(E2, open(sys.argv[1], "w"), indent=1)
print("runs", len(runs), dict(reasons), reasons2, "strict", len(sel_strict), "fallback", len(sel_fb), E2["fallback_for_lead_decision"]["n_frames"])
