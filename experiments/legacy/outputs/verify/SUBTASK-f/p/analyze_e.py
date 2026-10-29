"""SUBTASK-e analysis — three sides paired by (episode, seed), CPU only:
  off  = gate off, T39 S2-2 E3b runs           (${AG3S_PROJECT_ROOT}/outputs/verify/T39/runs/E3b)
  on_d = gate on, before the SUBTASK-e change  (${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-d/runs/E3bG, snapshot ${AG3S_PROJECT_ROOT}-sd)
  on_e = gate on, after the SUBTASK-e change   (${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-e/runs/E3bG, snapshot ${AG3S_PROJECT_ROOT}-se)
usage: analyze_e.py OUT_JSON [--off DIR] [--on-d DIR] [--on-e DIR]

Per-run base numbers: p/analyze_v2.py run_one (copied unchanged from SUBTASK-d; success / grasp / place = T28 analyze_e0 rule,
HOLD chunk = planning row whose control rows were not all executed, hold kinds = client gate.hold_kinds, reason kinds =
client planning-row verdict_reasons[].kind, clearance = server summary clearance_min_mm of the refined chunk).

Added here (definitions):
  release chunk t_rel     : first chunk (server t_step) whose summary subtask.released is True.
  region of a chunk t     : 'before_first_held' (t < first latch 'held'), 'held_before_placed', 'placed_before_release' (t >= first
                            'placed', t < t_rel), 'from_release' (t >= t_rel).
  subtask_no_target chunk : client planning row verdict_reasons has kind 'subtask_no_target'; also counted from the server
                            record summary_json verdict.reasons (if that key exists).
  ready pose              : traj.npz measured_qpos[0] (16-D state at control step 0, after the 1000 warm-up steps); arm
                            columns L = 0..6, R = 8..14 (rby1_16d layout [L7, Lgrip, R7, Rgrip]).
  joint distance          : || q[s, arm] - q_ready[arm] ||_2 in rad; path length = sum_s || q[s+1, L] - q[s, L] ||.
  EE distance             : MuJoCo FK of the client control-row full qpos (nq 66), body 'link_left_arm_6' world position;
                            || p(s) - p(0) || in mm.
  reference step s_ref    : on runs with a release: (t_rel - 1) * 8 (first control step of the release chunk); off runs and
                            on runs without release but with latch PLACED: (t_placed - 1) * 8. Pose numbers at s_ref, at the
                            last control step 599, min over s >= s_ref.
  contacts                : per control step, mj_forward on the control-row full qpos; MuJoCo contacts with dist < 0 between a
                            robot geom (body root = root of 'link_left_arm_0') and a non-robot geom; counted per body category
                            (apple / crate / table / other_fruit / other, p12_records.CATEGORY) as steps with >= 1 such
                            contact and min dist (mm); split before / from s_ref.
  pairs                   : 2x2 (A success x B success) and exact two-sided McNemar for (off, on_e) and (on_d, on_e);
                            also grasp and place.
  on_e vs on_d identity   : first control step where traj.npz executed_actions differ (> 0) -> chunk = step // 8 + 1;
                            first planning seq where frames 'actions' differ; first t where server refined_chunk differs;
                            first t where the server decision tuple differs (fields listed in DEC); first t where the client
                            gate action or hold kinds differ. Compared to t_rel of on_d and to t_gh_d = first on_d chunk that is
                            HOLD with kind 'uncertified' while its grounding_status == 'subtask_gated'.
  first-blocked executes  : on_e chunks with subtask.blocked == 'first' (gate refused a first adoption) and the
                            subtask_no_target reason; listed with run, t, latch, label, region, gate action."""
from common.workspace import workspace_path as _ag3s_workspace_path
import argparse, json, math, pathlib, sys, collections
import numpy as np
P_DIR = _ag3s_workspace_path('outputs/verify/SUBTASK-f/p')  # copied unchanged from SUBTASK-e/p; only this path
sys.path.insert(0, P_DIR)
import analyze_v2 as V  # noqa: E402
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as PR  # noqa: E402
import mujoco  # noqa: E402

DEC = ("latch", "label", "argmax", "blocked", "released", "released_id", "placed_seen", "manip_id", "mode", "excl_active",
       "excl_source", "grounding_status", "to_status")
CATS = ("apple", "crate", "table", "other_fruit", "other")
_M = {}


def model():
    if "m" not in _M:
        m = mujoco.MjModel.from_xml_path(PR.XML); d = mujoco.MjData(m)
        root = m.body_rootid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link_left_arm_0")]
        robot = np.array([m.body_rootid[m.geom_bodyid[g]] == root for g in range(m.ngeom)])
        cat = []
        for g in range(m.ngeom):
            b = m.body(m.geom_bodyid[g]).name; rb = m.body(m.body_rootid[m.geom_bodyid[g]]).name
            cat.append(PR.CATEGORY.get(b) or PR.CATEGORY.get(rb) or "other")
        _M["m"] = (m, d, robot, cat, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link_left_arm_6"))
    return _M["m"]


def C(x):
    return dict(sorted(collections.Counter(str(v) for v in x).items()))


def add(cs):
    out = collections.Counter()
    for c in cs:
        out.update(c)
    return dict(sorted(out.items()))


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else float(min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n))


def region(t, tH, tP, trel):
    if trel is not None and t >= trel:
        return "from_release"
    if tP is not None and t >= tP:
        return "placed_before_release"
    if tH is not None and t >= tH:
        return "held_before_placed"
    return "before_first_held"


def extra(r):
    rd = pathlib.Path(r["run_dir"]); ch = r["_chunks"]; crow = r["_crow"]; bt = {c["t"]: c for c in ch}
    tH, tP = r["t_first_held"], r["t_placed"]
    trel = next((c["t"] for c in ch if c["released"]), None)
    r["t_release"] = trel
    # server verdict reasons + refined chunk
    srv = {}
    for c in ch:
        with np.load(c["file"], allow_pickle=False) as z:
            s = json.loads(str(z["summary_json"]))
            srv[c["t"]] = {"reasons": sorted({x.get("kind") for x in ((s.get("verdict") or {}).get("reasons") or []) if isinstance(x, dict)}),
                           "has_verdict_key": "verdict" in s, "refined": np.asarray(z["refined_chunk"])}
    rows = []
    for c in crow:
        k = bt.get(c["t"], {})
        rows.append({"t": c["t"], "region": region(c["t"], tH, tP, trel), "hold": c["hold"], "hold_kinds": c["hold_kinds"] or [],
                     "reason_kinds": c["reason_kinds"], "gate_action": c["gate_action"],
                     "server_reason_kinds": srv.get(c["t"], {}).get("reasons"),
                     **{f: k.get(f) for f in DEC}, "clearance_min_mm": k.get("clearance_min_mm"), "file": k.get("file")})
    r["_rows"] = rows; r["_srv"] = srv
    snt = [x for x in rows if "subtask_no_target" in x["reason_kinds"]]
    r["snt"] = {"n_chunks": len(snt), "n_chunks_server": sum(1 for x in rows if "subtask_no_target" in (x["server_reason_kinds"] or [])),
                "n_hold_among": sum(x["hold"] for x in snt), "by_region": C(x["region"] for x in snt), "by_latch": C(x["latch"] for x in snt),
                "by_label": C(x["label"] for x in snt), "by_blocked": C(x["blocked"] for x in snt), "by_grounding": C(x["grounding_status"] for x in snt),
                "gate_action": C(x["gate_action"] for x in snt), "t": [x["t"] for x in snt]}
    reg = {}
    for g in ("before_first_held", "held_before_placed", "placed_before_release", "from_release"):
        xs = [x for x in rows if x["region"] == g]
        cl = [x["clearance_min_mm"] for x in xs if x["clearance_min_mm"] is not None]
        reg[g] = {"n_chunks": len(xs), "n_hold": sum(x["hold"] for x in xs), "hold_kinds": C(k for x in xs if x["hold"] for k in x["hold_kinds"]),
                  "reason_kinds": C(k for x in xs for k in x["reason_kinds"]), "grounding_status": C(x["grounding_status"] for x in xs),
                  "gate_action": C(x["gate_action"] for x in xs), "to_status": C(x["to_status"] for x in xs),
                  "hold_grounding": C(x["grounding_status"] for x in xs if x["hold"]),
                  "clearance_min_mm": min(cl) if cl else None, "n_clearance_lt0": sum(1 for v in cl if v < 0)}
    r["regions"] = reg
    r["first_blocked_snt"] = [{"t": x["t"], "latch": x["latch"], "label": x["label"], "region": x["region"], "gate_action": x["gate_action"],
                               "hold": x["hold"], "hold_kinds": x["hold_kinds"]} for x in rows if x["blocked"] == "first"]
    r["t_gate_hold_subtask_gated"] = next((x["t"] for x in rows if x["hold"] and "uncertified" in x["hold_kinds"] and x["grounding_status"] == "subtask_gated"), None)
    # pose
    z = np.load(rd / "traj.npz"); q = z["measured_qpos"].astype(np.float64); q0 = q[0]
    dL = np.linalg.norm(q[:, 0:7] - q0[0:7], axis=1); dR = np.linalg.norm(q[:, 8:15] - q0[8:15], axis=1)
    m, d, robot, cat, ee = model()
    ctrl = {}
    for l in (rd / "frames.jsonl").read_text().splitlines():
        if l.strip():
            x = json.loads(l)
            if x["kind"] == "control":
                ctrl[x["t_step"]] = x["qpos"]
    steps = sorted(ctrl)
    eep = np.zeros((len(steps), 3)); contact = []
    for i, s in enumerate(steps):
        d.qpos[:] = ctrl[s]; mujoco.mj_forward(m, d)
        eep[i] = d.xpos[ee]
        hit = {}
        for k in range(d.ncon):
            cc = d.contact[k]
            if cc.dist >= 0:
                continue
            g1, g2 = int(cc.geom1), int(cc.geom2)
            if robot[g1] == robot[g2]:
                continue
            other = g2 if robot[g1] else g1
            hit[cat[other]] = min(hit.get(cat[other], 0.0), 1000 * float(cc.dist))
        contact.append(hit)
    dEE = 1000 * np.linalg.norm(eep - eep[0], axis=1)
    sref = (trel - 1) * 8 if trel is not None else ((tP - 1) * 8 if tP is not None else None)
    r["s_ref"] = sref; r["s_ref_kind"] = "release" if trel is not None else ("placed" if tP is not None else None)
    pose = {"dL_rad_last": float(dL[-1]), "dR_rad_last": float(dR[-1]), "dEE_mm_last": float(dEE[-1]), "n_steps": int(len(q)),
            "n_control_rows_with_qpos": len(steps)}
    if sref is not None:
        pose.update({"dL_rad_at_ref": float(dL[sref]), "dL_rad_min_from_ref": float(dL[sref:].min()), "argmin_step_from_ref": int(sref + dL[sref:].argmin()),
                     "dR_rad_at_ref": float(dR[sref]), "dEE_mm_at_ref": float(dEE[sref]), "dEE_mm_min_from_ref": float(dEE[sref:].min()),
                     "L_path_rad_from_ref": float(np.linalg.norm(np.diff(q[sref:, 0:7], axis=0), axis=1).sum()),
                     "L_path_rad_before_ref": float(np.linalg.norm(np.diff(q[:sref + 1, 0:7], axis=0), axis=1).sum()),
                     "n_steps_from_ref": int(len(q) - sref)})
    r["pose"] = pose
    r["_series"] = {"dL": dL.tolist(), "dR": dR.tolist(), "dEE": dEE.tolist(), "steps": steps}
    def csum(sel):
        out = {}
        for cg in CATS:
            v = [h[cg] for i, h in enumerate(contact) if sel(steps[i]) and cg in h]
            out[cg] = {"n_steps": len(v), "min_dist_mm": min(v) if v else None}
        return out
    r["contacts"] = {"all": csum(lambda s: True)}
    if sref is not None:
        r["contacts"]["before_ref"] = csum(lambda s: s < sref); r["contacts"]["from_ref"] = csum(lambda s: s >= sref)
    r["_contact_steps"] = [{"step": steps[i], **h} for i, h in enumerate(contact) if h]
    return r


def _work(tk):
    side, cond, ep, seed, rd, bs = tk
    r = V.run_one(cond, ep, seed, rd, bs); r["side"] = side; extra(r)
    return side, ep, seed, r


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def identity(re_, rd_):
    out = {}
    xe = np.load(pathlib.Path(re_["run_dir"]) / "traj.npz")["executed_actions"]; xd = np.load(pathlib.Path(rd_["run_dir"]) / "traj.npz")["executed_actions"]
    dif = np.abs(xe.astype(np.float64) - xd.astype(np.float64)).max(axis=1)
    s0 = int(np.argmax(dif > 0)) if (dif > 0).any() else None
    out["exec_first_step"] = s0; out["exec_first_chunk"] = None if s0 is None else s0 // 8 + 1
    def plan(r):
        rows = [json.loads(l) for l in (pathlib.Path(r["run_dir"]) / "frames.jsonl").read_text().splitlines() if l.strip()]
        return {x["seq"]: x.get("actions") for x in rows if x["kind"] == "planning"}
    pe, pd = plan(re_), plan(rd_)
    out["planning_actions_first_t"] = next((t for t in sorted(set(pe) & set(pd)) if pe[t] != pd[t]), None)
    se, sd = re_["_srv"], rd_["_srv"]
    out["refined_chunk_first_t"] = next((t for t in sorted(set(se) & set(sd)) if not np.array_equal(se[t]["refined"], sd[t]["refined"])), None)
    be = {x["t"]: x for x in re_["_rows"]}; bd = {x["t"]: x for x in rd_["_rows"]}
    ts = sorted(set(be) & set(bd))
    tdec = next((t for t in ts if any(json.dumps(be[t][f], default=str) != json.dumps(bd[t][f], default=str) for f in DEC)), None)
    out["decision_first_t"] = tdec
    out["decision_fields_at_first"] = None if tdec is None else [f for f in DEC if json.dumps(be[tdec][f], default=str) != json.dumps(bd[tdec][f], default=str)]
    tg = next((t for t in ts if (be[t]["gate_action"], be[t]["hold_kinds"]) != (bd[t]["gate_action"], bd[t]["hold_kinds"])), None)
    out["gate_first_t"] = tg
    if tg is not None:
        out["gate_at_first"] = {"e": {k: be[tg][k] for k in ("gate_action", "hold_kinds", "reason_kinds", "latch", "label", "blocked", "released", "grounding_status")},
                                "d": {k: bd[tg][k] for k in ("gate_action", "hold_kinds", "reason_kinds", "latch", "label", "blocked", "released", "grounding_status")}}
    out["t_rel_d"] = rd_["t_release"]; out["t_rel_e"] = re_["t_release"]; out["t_gh_d"] = rd_["t_gate_hold_subtask_gated"]
    firsts = [v for v in (out["exec_first_chunk"], out["planning_actions_first_t"], out["refined_chunk_first_t"], out["decision_first_t"], out["gate_first_t"]) if v is not None]
    out["earliest_any_t"] = min(firsts) if firsts else None
    lim = out["t_gh_d"] if out["t_gh_d"] is not None else 76
    out["identical_before_t_gh_d"] = out["earliest_any_t"] is None or out["earliest_any_t"] >= lim
    lim2 = out["t_rel_d"] if out["t_rel_d"] is not None else 76
    out["identical_before_t_rel_d"] = out["earliest_any_t"] is None or out["earliest_any_t"] >= lim2
    return out


def stat(v):
    return V.stat(v)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    ap.add_argument("--off", default=_ag3s_workspace_path('outputs/verify/T39/runs/E3b'))
    ap.add_argument("--on-d", default=_ag3s_workspace_path('outputs/verify/SUBTASK-d/runs/E3bG'))
    ap.add_argument("--on-e", default=_ag3s_workspace_path('outputs/verify/SUBTASK-e/runs/E3bG'))
    a = ap.parse_args()
    runs = {"off": {}, "on_d": {}, "on_e": {}}
    tasks = []
    for side, dd in (("off", a.off), ("on_d", a.on_d), ("on_e", a.on_e)):
        g = pathlib.Path(dd); bs = V.segments(g)
        for cond, ep, seed, rd in V.discover(g):
            tasks.append((side, cond, ep, seed, rd, {seed: bs.get(seed, [])}))
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(24, len(tasks))) as pool:
        for side, ep, seed, r in pool.imap_unordered(_work, tasks):
            runs[side][(ep, seed)] = r
            print(side, ep, seed, "S" if r["success"] else "-", "rel", r["t_release"], "hold", r["n_hold_chunks"], "snt", r["snt"]["n_chunks"],
                  "hold_from_rel", r["regions"]["from_release"]["n_hold"], "dL_last", round(r["pose"]["dL_rad_last"], 3), flush=True)
    keys = sorted(set(runs["off"]) & set(runs["on_d"]) & set(runs["on_e"]))
    res = {"definitions": __doc__, "args": vars(a), "n_runs": {s: len(v) for s, v in runs.items()}, "n_triples": len(keys)}
    t2 = {}
    for A_, B_ in (("off", "on_e"), ("on_d", "on_e"), ("off", "on_d")):
        for f in ("success", "grasp", "place"):
            t = {"both": 0, f"{A_}_only": 0, f"{B_}_only": 0, "neither": 0}
            for k in keys:
                x, y = runs[A_][k][f], runs[B_][k][f]
                t["both" if x and y else f"{A_}_only" if x else f"{B_}_only" if y else "neither"] += 1
            t["n_pairs"] = len(keys); t["mcnemar_exact_p"] = mcnemar(t[f"{A_}_only"], t[f"{B_}_only"]); t2[f"{A_}_vs_{B_}:{f}"] = t
    res["paired_2x2"] = t2
    idn = {}
    for k in keys:
        idn[f"{k[0]}_s{k[1]}"] = identity(runs["on_e"][k], runs["on_d"][k])
    res["identity_on_e_vs_on_d"] = idn
    iv = list(idn.values())
    res["identity_summary"] = {
        "n_pairs": len(iv), "n_fully_identical": sum(1 for x in iv if x["earliest_any_t"] is None),
        "n_identical_before_t_gh_d": sum(x["identical_before_t_gh_d"] for x in iv), "n_identical_before_t_rel_d": sum(x["identical_before_t_rel_d"] for x in iv),
        "n_pairs_with_t_rel_d": sum(1 for x in iv if x["t_rel_d"] is not None), "n_pairs_with_t_gh_d": sum(1 for x in iv if x["t_gh_d"] is not None),
        "earliest_minus_t_gh_d": C(x["earliest_any_t"] - x["t_gh_d"] for x in iv if x["earliest_any_t"] is not None and x["t_gh_d"] is not None),
        "t_gh_d_minus_t_rel_d": C(x["t_gh_d"] - x["t_rel_d"] for x in iv if x["t_gh_d"] is not None and x["t_rel_d"] is not None),
        "n_t_rel_e_equal_t_rel_d": sum(1 for x in iv if x["t_rel_e"] == x["t_rel_d"]),
        "exec_first_chunk_minus_t_gh_d": C(x["exec_first_chunk"] - x["t_gh_d"] for x in iv if x["exec_first_chunk"] is not None and x["t_gh_d"] is not None),
        "decision_first_t_minus_t_gh_d": C(x["decision_first_t"] - x["t_gh_d"] for x in iv if x["decision_first_t"] is not None and x["t_gh_d"] is not None),
        "decision_fields_at_first": C(",".join(x["decision_fields_at_first"]) for x in iv if x["decision_fields_at_first"]),
        "pairs_without_t_gh_d_but_differing": [k for k, x in idn.items() if x["t_gh_d"] is None and x["earliest_any_t"] is not None]}
    summ = {}
    for side, rr in runs.items():
        rs = [rr[k] for k in keys]
        rel = [r for r in rs if r["t_release"] is not None]
        fr = [r["regions"]["from_release"] for r in rs]
        pose_ref = [r for r in rs if r["s_ref"] is not None]
        summ[side] = {
            "n_runs": len(rs), "n_success": sum(r["success"] for r in rs), "n_grasp": sum(r["grasp"] for r in rs), "n_place": sum(r["place"] for r in rs),
            "n_placed_latch": sum(1 for r in rs if r["placed"]), "n_runs_released": len(rel),
            "n_hold_chunks_total": sum(r["n_hold_chunks"] for r in rs), "n_runs_with_hold": sum(1 for r in rs if r["n_hold_chunks"]),
            "hold_kind_counts": add(r["hold_kind_counts"] for r in rs), "gate_action_counts": add(r["gate_action_counts"] for r in rs),
            "reason_kind_counts": add(r["reason_kind_counts"] for r in rs),
            "regions": {g: {"n_chunks": sum(r["regions"][g]["n_chunks"] for r in rs), "n_hold": sum(r["regions"][g]["n_hold"] for r in rs),
                            "hold_kinds": add(r["regions"][g]["hold_kinds"] for r in rs), "reason_kinds": add(r["regions"][g]["reason_kinds"] for r in rs),
                            "grounding_status": add(r["regions"][g]["grounding_status"] for r in rs), "hold_grounding": add(r["regions"][g]["hold_grounding"] for r in rs),
                            "gate_action": add(r["regions"][g]["gate_action"] for r in rs), "to_status": add(r["regions"][g]["to_status"] for r in rs),
                            "clearance_min_mm": min([r["regions"][g]["clearance_min_mm"] for r in rs if r["regions"][g]["clearance_min_mm"] is not None], default=None),
                            "n_clearance_lt0": sum(r["regions"][g]["n_clearance_lt0"] for r in rs)}
                        for g in ("before_first_held", "held_before_placed", "placed_before_release", "from_release")},
            "snt": {"n_chunks": sum(r["snt"]["n_chunks"] for r in rs), "n_chunks_server": sum(r["snt"]["n_chunks_server"] for r in rs),
                    "n_runs": sum(1 for r in rs if r["snt"]["n_chunks"]), "n_hold_among": sum(r["snt"]["n_hold_among"] for r in rs),
                    **{k: add(r["snt"][k] for r in rs) for k in ("by_region", "by_latch", "by_label", "by_blocked", "by_grounding", "gate_action")}},
            "first_blocked": [{"run": f'{r["episode"]}_s{r["seed"]}', **x} for r in rs for x in r["first_blocked_snt"]],
            "blocked_counts_all": add(r["blocked_counts_all"] for r in rs),
            "clearance_min_mm_overall": min(r["clearance_min_mm"] for r in rs if r["clearance_min_mm"] is not None),
            "n_chunks_clearance_lt0": sum(r["n_chunks_clearance_lt0"] for r in rs),
            "from_release_clearance_min_mm_per_run": [r["regions"]["from_release"]["clearance_min_mm"] for r in rel],
            "pose": {"n_runs_with_ref": len(pose_ref), "ref_kind": C(r["s_ref_kind"] for r in pose_ref),
                     **{k: stat([r["pose"].get(k) for r in pose_ref]) for k in ("dL_rad_at_ref", "dL_rad_last", "dL_rad_min_from_ref", "dEE_mm_at_ref", "dEE_mm_last",
                                                                                "dEE_mm_min_from_ref", "L_path_rad_from_ref")},
                     "dL_rad_last_all_runs": stat([r["pose"]["dL_rad_last"] for r in rs]), "dEE_mm_last_all_runs": stat([r["pose"]["dEE_mm_last"] for r in rs])},
            "contacts_from_ref": {cg: {"n_runs": sum(1 for r in pose_ref if r["contacts"]["from_ref"][cg]["n_steps"]),
                                       "n_steps": sum(r["contacts"]["from_ref"][cg]["n_steps"] for r in pose_ref),
                                       "min_dist_mm": min([r["contacts"]["from_ref"][cg]["min_dist_mm"] for r in pose_ref if r["contacts"]["from_ref"][cg]["min_dist_mm"] is not None], default=None)}
                                  for cg in CATS},
            "contacts_all": {cg: {"n_runs": sum(1 for r in rs if r["contacts"]["all"][cg]["n_steps"]), "n_steps": sum(r["contacts"]["all"][cg]["n_steps"] for r in rs),
                                  "min_dist_mm": min([r["contacts"]["all"][cg]["min_dist_mm"] for r in rs if r["contacts"]["all"][cg]["min_dist_mm"] is not None], default=None)}
                             for cg in CATS},
            "n_budget_hit": sum(r["n_budget_hit"] for r in rs), "to_status_counts": add(r["to_status_counts"] for r in rs),
            "n_refined_mismatch": sum(r["n_refined_mismatch_gt_1e-5"] for r in rs),
            "n_runs_segment_found": sum(1 for r in rs if r["n_server_chunks"] == 75),
        }
    res["summary"] = summ
    res["per_run"] = {}
    for side, rr in runs.items():
        for k, r in rr.items():
            res["per_run"][f"{side}/{k[0]}_s{k[1]}"] = {
                **{f: r[f] for f in ("cond", "episode", "seed", "run_dir", "success", "grasp", "place", "t_first_held", "t_placed", "t_release", "s_ref", "s_ref_kind",
                                     "n_hold_chunks", "hold_t", "hold_kind_counts", "reason_kind_counts", "snt", "regions", "first_blocked_snt", "t_gate_hold_subtask_gated",
                                     "pose", "contacts", "clearance_min_mm", "n_chunks_clearance_lt0", "n_server_chunks", "n_refined_mismatch_gt_1e-5")},
                "rows": r["_rows"], "series": r["_series"], "contact_steps": r["_contact_steps"]}
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"paired_2x2": t2, "identity_summary": res["identity_summary"],
                      "summary": {s: {k: v for k, v in x.items() if k in ("n_runs", "n_success", "n_hold_chunks_total", "hold_kind_counts", "snt", "n_runs_released")}
                                  for s, x in summ.items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
