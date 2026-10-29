"""SUBTASK-d V2 — gate on (E3bG, SUBTASK-d runs) vs gate off (E3b, T39 S2-2 runs), paired by (episode, seed). CPU only.

usage: analyze_v2.py OUT_JSON --off DIR --on DIR [--extra-off DIR]   (DIR holds <COND>_ep<EP>_s<SEED>/ + NAME.done + server_*/)
Run discovery: sub-directories <COND>_ep<EP>_s<SEED> with frames.jsonl and NAME.done.
Server chunks: <group>/server_*/server_constraints/run_*/chunk_*.npz; segments start where t_step == 1; a run takes the segment
  whose first chunk has summary policy_seed.applied == its seed. Inside a run a chunk is addressed by (episode, seed, t_step)
  (server t_step == client planning seq), never by chunk_index (chunk_index continues across runs on one server).
  Each server chunk is re-checked against the client planning row: refined_chunk == planning actions within 1e-5 (counted).

Per run (definitions):
  grasp / place / success : outputs/verify/T28/E0/analyze_e0.py:analyze (unchanged; same as T39 analyze_t39.py).
  latch state            : summary grasp.state per chunk (searching / latched / closing / held / placed / ...).
  PLACED                 : first chunk (t_step) with grasp.state == 'placed'; "after PLACED" = that chunk and every later chunk.
  switch                 : summary admissibility.decision.mode == 'switch'; first target adoption = first chunk with mode == 'first'.
  exclusion active       : summary exclusion.active is True (source / mechanism counted too).
  released               : summary subtask.released is True (SUBTASK-c B3 release on that frame); delay = t_step(first released)
                           - t_step(PLACED), in chunks.
  gate blocked           : summary subtask.blocked in {'first', 'switch'} (gate on); would_block (gate off shadow, only where a
                           'subtask' key exists).
  label                  : summary subtask.label (confirmed, debounced; None before confirmation); argmax = per-request argmax.
  transition delay       : pick->place = t_step(first label=='place' at/after first 'held') - t_step(first 'held');
                           place->home = t_step(first label=='home' at/after PLACED) - t_step(PLACED). In chunks.
  place chunk            : place_t // 8 + 1 (place_t = first planning t_step of the final in-crate streak, analyze_e0 rule);
                           release_chunk_vs_place_chunk = first released chunk - place chunk (negative = released before in crate).
  HOLD chunk             : planning row whose control rows were not all executed (T39 rule); gate action per planning row;
                           verdict_reasons[].kind once per kind per row.
  clearance              : summary clearance_min_mm per chunk (server, the sphere-ESDF clearance of the refined chunk); per run min.
  chunk time             : client timing_ms.policy_infer and server verdict.timing_ms {infer, ag3s, trajopt, total}, first chunk excluded.
  time_budget_hit        : summary to.time_budget_hit count (SQP wall-clock deadline reached).
  carved object (after PLACED, exclusion active): manipulated.centroid (base frame) -> world with the base pose from FK of the
                           client planning row qpos of the same t_step (outputs/verify/T28/P1-2/p12_records.py, as T39);
                           nearest object centre among apple / banana / orange / pear / crate (planning-row object_poses), and
                           distance to the apple centre and to the apple's initial centre (first planning row), mm.
Per pair:
  2x2 (off success x on success) + exact two-sided McNemar p = min(1, 2 P[Binom(b+c, 1/2) <= min(b, c)]); same for grasp, place.
  first divergence: first control step where executed actions (traj.npz) differ (> 0) and > 1e-3; chunk = step // 8 + 1;
  latch state / label / exclusion of both runs at that chunk."""
from common.workspace import workspace_path as _ag3s_workspace_path
import argparse, json, math, pathlib, re, sys, collections
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/E0'))
import analyze_e0 as A  # noqa: E402
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
CENTRES = ("apple", "banana", "orange", "pear", "crate")
_G = {}


def geom():
    if "g" not in _G:
        import p12_records as P  # noqa: E402  (MuJoCo model, base->world FK; T28 P1-2 / T39 convention)
        _G["g"] = (P, P.build())
    return _G["g"]

NAME = re.compile(r"^(E3b|E3bG)_ep(\d+)_s(\d+)$")
STATES = ["searching", "latched", "closing", "held", "placed"]


def stat(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    a = np.asarray(v, float)
    return {"n": int(a.size), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max()),
            "min": float(a.min()), "mean": float(a.mean())}


def cnt(items):
    out = collections.Counter(str(x) for x in items)
    return dict(sorted(out.items()))


def discover(d):
    d = pathlib.Path(d)
    out = []
    for rd in sorted(p for p in d.iterdir() if p.is_dir()):
        m = NAME.match(rd.name)
        if m and (rd / "frames.jsonl").exists() and (d / f"{rd.name}.done").exists():
            out.append((m.group(1), int(m.group(2)), int(m.group(3)), rd))
    return out


def segments(group):
    segs = []
    for rec in sorted(group.glob("server_*/server_constraints/run_*")):
        cur = []
        for f in sorted(rec.glob("chunk_*.npz")):
            with np.load(f, allow_pickle=False) as z:
                t = int(z["t_step"])
            if t == 1 and cur:
                segs.append(cur); cur = []
            cur.append(f)
        if cur:
            segs.append(cur)
    by_seed = {}
    for sg in segs:
        with np.load(sg[0], allow_pickle=False) as z:
            s = json.loads(str(z["summary_json"]))
        seed = (s.get("policy_seed") or {}).get("applied")
        by_seed.setdefault(seed, []).append(sg)
    return by_seed


def run_one(cond, ep, seed, rd, by_seed):
    base = A.analyze(rd)
    rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    plan = {r["seq"]: r for r in rows if r["kind"] == "planning"}
    ctrl = [r for r in rows if r["kind"] == "control"]
    segs = by_seed.get(seed, [])
    out = {"cond": cond, "episode": ep, "seed": seed, "run_dir": str(rd), "n_server_segments_for_seed": len(segs),
           **{k: base[k] for k in ("success", "grasp", "place", "grasp_t", "place_t", "apple_dz_max_mm")}}
    seg = segs[-1] if segs else []
    ch = []; mism = 0
    for f in seg:
        with np.load(f, allow_pickle=False) as z:
            s = json.loads(str(z["summary_json"])); t = int(z["t_step"]); ref = np.asarray(z["refined_chunk"], np.float64)
            ci = int(z["chunk_index"])
        p = plan.get(t)
        if p is not None and p.get("actions") is not None and np.abs(ref - np.asarray(p["actions"], np.float64)).max() > 1e-5:
            mism += 1
        st = s.get("subtask"); m = s.get("manipulated") or {}; ex = s.get("exclusion") or {}
        dec = (s.get("admissibility") or {}).get("decision") or {}
        ch.append({"t": t, "chunk_index": ci, "file": str(f), "latch": (s.get("grasp") or {}).get("state"),
                   "has_subtask": st is not None,
                   "label": (st or {}).get("label"), "argmax": (st or {}).get("argmax"), "p": (st or {}).get("p"),
                   "blocked": (st or {}).get("blocked"), "would_block": (st or {}).get("would_block"),
                   "released": bool((st or {}).get("released")), "released_id": (st or {}).get("released_id"),
                   "placed_seen": (st or {}).get("placed_seen"), "streak": (st or {}).get("streak"),
                   "manip_id": m.get("id"), "manip_state": m.get("state"), "switched_from": m.get("switched_from"),
                   "mode": dec.get("mode"), "grounding_status": s.get("grounding_status"),
                   "excl_active": bool(ex.get("active")), "excl_source": ex.get("source"), "excl_mech": "+".join(ex.get("mechanism") or []),
                   "excl_manip_id": ex.get("manipulated_id"), "centroid": m.get("centroid"),
                   "clearance_min_mm": s.get("clearance_min_mm"), "budget_hit": (s.get("to") or {}).get("time_budget_hit"),
                   "to_status": (s.get("to") or {}).get("status")})
    out["n_server_chunks"] = len(ch); out["n_refined_mismatch_gt_1e-5"] = mism
    out["chunk_index_range"] = [ch[0]["chunk_index"], ch[-1]["chunk_index"]] if ch else None
    # client rows
    crow = []
    for seq in sorted(plan):
        p = plan[seq]; cs = [c for c in ctrl if c["chunk_seq"] == seq]; g = p.get("gate") or {}; v = p.get("verdict") or {}
        crow.append({"t": seq, "n_control": len(cs), "n_exec": sum(1 for c in cs if c.get("executed", True)),
                     "gate_action": g.get("action"), "hold_kinds": g.get("hold_kinds"), "ipc": p.get("ipc"),
                     "reason_kinds": sorted({r.get("kind") for r in (p.get("verdict_reasons") or []) if isinstance(r, dict)}),
                     "max_violation_m": v.get("max_violation_m"), "server_ms": v.get("timing_ms"),
                     "rt_ms": (p.get("timing_ms") or {}).get("policy_infer")})
    for c in crow:
        c["hold"] = c["n_exec"] < c["n_control"]
    bt = {c["t"]: c for c in ch}
    tP = next((c["t"] for c in ch if c["latch"] == "placed"), None)
    tH = next((c["t"] for c in ch if c["latch"] == "held"), None)
    after = [c for c in ch if tP is not None and c["t"] >= tP]
    before = [c for c in ch if tP is None or c["t"] < tP]
    out["latch_counts"] = cnt(c["latch"] for c in ch)
    out["t_first_held"] = tH; out["t_placed"] = tP; out["placed"] = tP is not None
    out["after_placed"] = {"n_chunks": len(after), "n_switch": sum(c["mode"] == "switch" for c in after),
                           "n_excl_active": sum(c["excl_active"] for c in after),
                           "excl_source_counts": cnt(f'{c["excl_source"]}|{c["excl_active"]}|{c["excl_mech"]}' for c in after),
                           "manip_ids": sorted({c["manip_id"] for c in after if c["manip_id"] is not None}),
                           "n_manip_none": sum(c["manip_id"] is None for c in after),
                           "first_switch_t": next((c["t"] for c in after if c["mode"] == "switch"), None),
                           "n_released_chunks": sum(c["released"] for c in after),
                           "first_released_t": next((c["t"] for c in after if c["released"]), None),
                           "mode_counts": cnt(c["mode"] for c in after),
                           "grounding_status_counts": cnt(c["grounding_status"] for c in after),
                           "n_blocked": cnt(c["blocked"] for c in after if c["blocked"]),
                           "n_would_block": cnt(c["would_block"] for c in after if c["would_block"])}
    a = out["after_placed"]
    a["placed_to_release_chunks"] = (a["first_released_t"] - tP) if (tP is not None and a["first_released_t"] is not None) else None
    a["placed_to_first_switch_chunks"] = (a["first_switch_t"] - tP) if (tP is not None and a["first_switch_t"] is not None) else None
    a["excl_active_frac"] = (a["n_excl_active"] / a["n_chunks"]) if a["n_chunks"] else None
    out["before_placed"] = {"n_chunks": len(before), "n_switch": sum(c["mode"] == "switch" for c in before),
                            "blocked": [{"t": c["t"], "blocked": c["blocked"], "latch": c["latch"], "label": c["label"]} for c in before if c["blocked"]],
                            "would_block": [{"t": c["t"], "would_block": c["would_block"], "latch": c["latch"], "label": c["label"]} for c in before if c["would_block"]],
                            "n_excl_active": sum(c["excl_active"] for c in before)}
    carved = []
    for c in after:
        if not c["excl_active"] or c.get("centroid") is None or c["t"] not in plan:
            continue
        P, (m, d, *_r) = geom(); p = plan[c["t"]]
        P.set_q(m, d, p["qpos"]); cw = P.base_to_world(m, d, c["centroid"]); op = p["object_poses"]
        dist = {b: float(1000 * np.linalg.norm(cw - np.asarray(op[b]["pos"]))) for b in CENTRES if b in op}
        a0 = np.asarray(plan[min(plan)]["object_poses"]["apple"]["pos"])
        carved.append({"t": c["t"], "manip_id": c["manip_id"], "nearest": min(dist, key=dist.get), "to_apple_mm": dist.get("apple"),
                       "to_apple_initial_mm": float(1000 * np.linalg.norm(cw - a0)), "centroid_world": cw.tolist(),
                       "to_nearest_mm": min(dist.values()), "source": c["excl_source"]})
    out["after_placed"]["carved"] = carved
    out["after_placed"]["carved_nearest_counts"] = cnt(x["nearest"] for x in carved)
    out["first_adoption_t"] = next((c["t"] for c in ch if c["mode"] == "first"), None)
    out["n_first_decisions"] = sum(c["mode"] == "first" for c in ch)
    out["n_switch_total"] = sum(c["mode"] == "switch" for c in ch)
    out["n_released_total"] = sum(c["released"] for c in ch)
    out["n_subtask_gated_status"] = sum(c["grounding_status"] == "subtask_gated" for c in ch)
    out["blocked_counts_all"] = cnt(c["blocked"] for c in ch if c["blocked"])
    # labels
    out["has_subtask_chunks"] = sum(c["has_subtask"] for c in ch)
    out["label_x_latch"] = cnt(f'{c["label"]}|{c["latch"]}' for c in ch if c["has_subtask"])
    out["argmax_x_latch"] = cnt(f'{c["argmax"]}|{c["latch"]}' for c in ch if c["has_subtask"])
    tpl = next((c["t"] for c in ch if tH is not None and c["t"] >= tH and c["label"] == "place"), None)
    tph = next((c["t"] for c in ch if tP is not None and c["t"] >= tP and c["label"] == "home"), None)
    out["delay_pick_to_place_chunks"] = (tpl - tH) if (tpl is not None and ch and ch[0]["has_subtask"]) else None
    out["delay_place_to_home_chunks"] = (tph - tP) if (tph is not None and ch and ch[0]["has_subtask"]) else None
    out["t_first_label_place"] = next((c["t"] for c in ch if c["label"] == "place"), None)
    out["t_first_label_home"] = next((c["t"] for c in ch if c["label"] == "home"), None)
    out["t_first_label_home_before_placed"] = next((c["t"] for c in before if c["label"] == "home"), None) if tP is not None else None
    # safety / time
    hold = [c for c in crow if c["hold"]]
    out["n_hold_chunks"] = len(hold); out["hold_t"] = [c["t"] for c in hold]
    trel = out["after_placed"]["first_released_t"]
    out["n_hold_after_placed"] = sum(1 for c in hold if tP is not None and c["t"] >= tP)
    out["n_hold_from_release"] = sum(1 for c in hold if trel is not None and c["t"] >= trel)
    out["n_chunks_from_release"] = (75 - trel + 1) if trel is not None else 0
    out["hold_kind_from_release"] = cnt(k for c in hold if trel is not None and c["t"] >= trel for k in (c["hold_kinds"] or []))
    out["release_chunk_vs_place_chunk"] = (trel - (base["place_t"] // 8 + 1)) if (trel is not None and base["place_t"] is not None) else None
    out["hold_latch"] = cnt(bt.get(c["t"], {}).get("latch") for c in hold)
    out["hold_kind_counts"] = cnt(k for c in hold for k in (c["hold_kinds"] or []))
    out["gate_action_counts"] = cnt(c["gate_action"] for c in crow)
    out["reason_kind_counts"] = cnt(k for c in crow for k in c["reason_kinds"])
    cl = [c["clearance_min_mm"] for c in ch if c["clearance_min_mm"] is not None]
    out["clearance_min_mm"] = min(cl) if cl else None
    out["n_chunks_clearance_lt0"] = sum(1 for x in cl if x < 0)
    out["max_violation_m_max"] = max((c["max_violation_m"] or 0.0) for c in crow) if crow else None
    out["n_budget_hit"] = sum(bool(c["budget_hit"]) for c in ch)
    out["to_status_counts"] = cnt(c["to_status"] for c in ch)
    rest = crow[1:]
    out["time_ms_excl_first"] = {"round_trip": stat([c["rt_ms"] for c in rest]),
                                 **{f"server_{k}": stat([(c["server_ms"] or {}).get(k) for c in rest]) for k in ("infer", "ag3s", "trajopt", "total")}}
    out["_chunks"] = ch; out["_crow"] = crow
    return out


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else float(min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--off", required=True); ap.add_argument("--on", required=True)
    ap.add_argument("--extra-off", nargs="*", default=[])
    a = ap.parse_args()
    res = {"definitions": __doc__, "args": vars(a), "runs": {}}
    exe = {}
    for d in [a.off, a.on] + list(a.extra_off):
        g = pathlib.Path(d); bs = segments(g)
        for cond, ep, seed, rd in discover(g):
            key = f"{g.parent.name}/{g.name}/{rd.name}"
            r = run_one(cond, ep, seed, rd, bs)
            r["side"] = "on" if cond == "E3bG" else ("off" if str(g) == str(pathlib.Path(a.off)) else "off_extra")
            res["runs"][key] = r
            exe[key] = np.load(rd / "traj.npz")["executed_actions"].astype(np.float64)
            print(key, "S" if r["success"] else "-", "P", r["t_placed"], "sw_after", r["after_placed"]["n_switch"],
                  "excl_after", r["after_placed"]["n_excl_active"], "/", r["after_placed"]["n_chunks"], "rel", r["n_released_total"],
                  "hold", r["n_hold_chunks"], "seg", r["n_server_segments_for_seed"], "mism", r["n_refined_mismatch_gt_1e-5"], flush=True)
    idx = {}
    for k, r in res["runs"].items():
        if r["side"] in ("on", "off"):
            idx.setdefault((r["episode"], r["seed"]), {})[r["side"]] = k
    pairs = []
    for (ep, seed), d in sorted(idx.items()):
        if "on" not in d or "off" not in d:
            continue
        ro, rn = res["runs"][d["off"]], res["runs"][d["on"]]
        x0, x1 = exe[d["off"]], exe[d["on"]]
        dif = np.abs(x0 - x1).max(axis=1)
        s0 = int(np.argmax(dif > 0)) if (dif > 0).any() else None
        s3 = int(np.argmax(dif > 1e-3)) if (dif > 1e-3).any() else None
        dv = {"first_step_gt0": s0, "first_step_gt_1e-3": s3, "max_abs": float(dif.max())}
        for nm, s in (("gt0", s0), ("gt_1e-3", s3)):
            if s is None:
                continue
            seq = s // 8 + 1
            co = next((c for c in ro["_chunks"] if c["t"] == seq), {}); cn = next((c for c in rn["_chunks"] if c["t"] == seq), {})
            dv[f"at_{nm}"] = {"chunk_t": seq, "off": {k: co.get(k) for k in ("latch", "label", "manip_id", "mode", "excl_active", "excl_source")},
                              "on": {k: cn.get(k) for k in ("latch", "label", "manip_id", "mode", "excl_active", "excl_source", "blocked", "released")},
                              "on_t_placed": rn["t_placed"], "off_t_placed": ro["t_placed"]}
        # first server-side decision difference (chunk keyed by t_step)
        fo = {c["t"]: c for c in ro["_chunks"]}; fn = {c["t"]: c for c in rn["_chunks"]}
        dd = next((t for t in sorted(set(fo) & set(fn)) if (fo[t]["manip_id"], fo[t]["excl_active"], fo[t]["excl_source"], fo[t]["mode"])
                   != (fn[t]["manip_id"], fn[t]["excl_active"], fn[t]["excl_source"], fn[t]["mode"])), None)
        dv["first_decision_diff_t"] = dd
        if dd is not None:
            dv["at_first_decision_diff"] = {"off": {k: fo[dd].get(k) for k in ("latch", "manip_id", "mode", "excl_active", "excl_source")},
                                           "on": {k: fn[dd].get(k) for k in ("latch", "label", "manip_id", "mode", "excl_active", "excl_source", "blocked", "released")}}
        pairs.append({"episode": ep, "seed": seed, "off_run": d["off"], "on_run": d["on"],
                      "off": {f: ro[f] for f in ("success", "grasp", "place")}, "on": {f: rn[f] for f in ("success", "grasp", "place")},
                      "divergence": dv})
    res["pairs"] = pairs
    t2 = {}
    for f in ("success", "grasp", "place"):
        t = {"both": 0, "off_only": 0, "on_only": 0, "neither": 0}
        for p in pairs:
            x, y = p["off"][f], p["on"][f]
            t["both" if x and y else "off_only" if x else "on_only" if y else "neither"] += 1
        t["n_pairs"] = len(pairs); t["mcnemar_exact_p"] = mcnemar(t["off_only"], t["on_only"]); t2[f] = t
    res["paired_2x2"] = t2
    summ = {}
    for side in ("off", "on", "off_extra"):
        rs = [r for r in res["runs"].values() if r["side"] == side]
        if not rs:
            continue
        pl = [r for r in rs if r["placed"]]
        allc = [c for r in rs for c in r["_crow"][1:]]
        summ[side] = {
            "n_runs": len(rs), "n_success": sum(r["success"] for r in rs), "n_grasp": sum(r["grasp"] for r in rs), "n_place": sum(r["place"] for r in rs),
            "n_placed_latch": len(pl), "n_placed_runs_with_switch_after": sum(1 for r in pl if r["after_placed"]["n_switch"]),
            "n_switch_after_placed_total": sum(r["after_placed"]["n_switch"] for r in pl),
            "after_placed_chunks_total": sum(r["after_placed"]["n_chunks"] for r in pl),
            "after_placed_excl_active_total": sum(r["after_placed"]["n_excl_active"] for r in pl),
            "after_placed_excl_source_counts": dict(sum((collections.Counter(r["after_placed"]["excl_source_counts"]) for r in pl), collections.Counter())),
            "after_placed_excl_active_frac_per_run": [r["after_placed"]["excl_active_frac"] for r in pl],
            "after_placed_carved_nearest_counts": dict(sum((collections.Counter(r["after_placed"]["carved_nearest_counts"]) for r in pl), collections.Counter())),
            "after_placed_carved_to_apple_mm": stat([x["to_apple_mm"] for r in pl for x in r["after_placed"]["carved"]]),
            "after_placed_carved_to_apple_initial_mm": stat([x["to_apple_initial_mm"] for r in pl for x in r["after_placed"]["carved"]]),
            "after_placed_carved_to_nearest_mm": stat([x["to_nearest_mm"] for r in pl for x in r["after_placed"]["carved"]]),
            "n_runs_released": sum(1 for r in rs if r["n_released_total"]), "n_released_chunks_total": sum(r["n_released_total"] for r in rs),
            "placed_to_release_chunks": [r["after_placed"]["placed_to_release_chunks"] for r in pl],
            "placed_to_first_switch_chunks": [r["after_placed"]["placed_to_first_switch_chunks"] for r in pl],
            "n_switch_before_placed_total": sum(r["before_placed"]["n_switch"] for r in rs),
            "n_switch_total": sum(r["n_switch_total"] for r in rs),
            "blocked_before_placed": [{"run": r["run_dir"].split("/")[-1], **b} for r in rs for b in r["before_placed"]["blocked"]],
            "blocked_counts_all": dict(sum((collections.Counter(r["blocked_counts_all"]) for r in rs), collections.Counter())),
            "n_subtask_gated_status": sum(r["n_subtask_gated_status"] for r in rs),
            "first_adoption_t": cnt(r["first_adoption_t"] for r in rs),
            "label_x_latch": dict(sum((collections.Counter(r["label_x_latch"]) for r in rs), collections.Counter())),
            "argmax_x_latch": dict(sum((collections.Counter(r["argmax_x_latch"]) for r in rs), collections.Counter())),
            "delay_pick_to_place_chunks": cnt(r["delay_pick_to_place_chunks"] for r in rs if r["t_first_held"] is not None),
            "delay_place_to_home_chunks": cnt(r["delay_place_to_home_chunks"] for r in pl),
            "n_label_home_before_placed": sum(1 for r in pl if r["t_first_label_home_before_placed"] is not None),
            "n_hold_chunks_total": sum(r["n_hold_chunks"] for r in rs), "n_runs_with_hold": sum(1 for r in rs if r["n_hold_chunks"]),
            "n_hold_after_placed_total": sum(r["n_hold_after_placed"] for r in rs),
            "n_hold_from_release_total": sum(r["n_hold_from_release"] for r in rs), "n_chunks_from_release_total": sum(r["n_chunks_from_release"] for r in rs),
            "hold_kind_from_release": dict(sum((collections.Counter(r["hold_kind_from_release"]) for r in rs), collections.Counter())),
            "release_chunk_minus_place_chunk": [r["release_chunk_vs_place_chunk"] for r in rs if r["n_released_total"]],
            "hold_latch_counts": dict(sum((collections.Counter(r["hold_latch"]) for r in rs), collections.Counter())),
            "hold_kind_counts": dict(sum((collections.Counter(r["hold_kind_counts"]) for r in rs), collections.Counter())),
            "gate_action_counts": dict(sum((collections.Counter(r["gate_action_counts"]) for r in rs), collections.Counter())),
            "reason_kind_counts": dict(sum((collections.Counter(r["reason_kind_counts"]) for r in rs), collections.Counter())),
            "clearance_min_mm": stat([r["clearance_min_mm"] for r in rs]), "clearance_min_mm_overall": min(r["clearance_min_mm"] for r in rs if r["clearance_min_mm"] is not None),
            "n_chunks_clearance_lt0": sum(r["n_chunks_clearance_lt0"] for r in rs),
            "n_budget_hit": sum(r["n_budget_hit"] for r in rs),
            "to_status_counts": dict(sum((collections.Counter(r["to_status_counts"]) for r in rs), collections.Counter())),
            "time_ms_pooled_excl_first": {"round_trip": stat([c["rt_ms"] for c in allc]),
                                          **{f"server_{k}": stat([(c["server_ms"] or {}).get(k) for c in allc]) for k in ("infer", "ag3s", "trajopt", "total")}},
            "n_refined_mismatch": sum(r["n_refined_mismatch_gt_1e-5"] for r in rs),
            "n_runs_segment_found": sum(1 for r in rs if r["n_server_chunks"] == 75),
            "n_runs_with_subtask_record": sum(1 for r in rs if r["has_subtask_chunks"]),
        }
    res["summary"] = summ
    for r in res["runs"].values():
        r["chunks"] = r.pop("_chunks"); r["client_rows"] = r.pop("_crow")
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"paired_2x2": t2, "summary": {s: {k: v for k, v in x.items() if not isinstance(v, (list, dict)) or k in ("gate_action_counts", "blocked_counts_all")} for s, x in summ.items()}}, indent=1))


if __name__ == "__main__":
    main()
