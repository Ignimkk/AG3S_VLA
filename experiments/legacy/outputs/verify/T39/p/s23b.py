"""T39 S2-3b — follow-up measurements on the S2-2 records (CPU only). usage: s23b.py ANALYSIS_JSON OUT_JSON

1. per pair (48), chunk seq k = 1..5 (rows 8(k-1) .. 8k-1 of the executed arrays, chunk rows 0-7):
   a_exec    = max |E3b executed (traj.npz executed_actions) - E3b actions_reference (frames planning row)|, all 16 cols
   a_refined = max |E3b actions (refined chunk sent; == server record refined_chunk, 0 mismatches > 1e-5) - actions_reference|, rows 0-7
   b         = max |E3b actions_reference - E0 executed (traj.npz executed_actions)|, rows 0-7
   c (k = 2, 3) = max |E0 planning-row qpos - E3b planning-row qpos| (full 66) and max |inference_states| difference (16D policy state)
   chunk 2: fraction of pairs with a_exec > 1e-3 and b < 1e-4.
   distribution of a_exec over all E3b chunks (48 x 75) by server summary grasp.state (latch) and by planning-row gate action +
   sorted verdict_reasons kinds.
2. HOLD vs outcome: runs with >= 1 HOLD chunk vs none (success / grasp / place); per discordant pair: E3b HOLD count, first HOLD
   chunk t and its offset from E3b grasp_t (t_hold - grasp_t, control steps; None when no grasp); per HOLD kind: runs that have
   >= 1 chunk of that kind and their success.
3. E3b latency per chunk (first chunk of each run excluded): round trip (client timing_ms.policy_infer) - server verdict.timing_ms.total.
   Server 'infer' wraps AttentionPolicy.infer (policy forward + attention-copy forward) in one timer (safe_policy.py:342-345);
   no record separates them. E0: no server time is recorded; E0 round trip is reported as the upper bound."""
import json, sys, pathlib
import numpy as np

A = json.loads(pathlib.Path(sys.argv[1]).read_text())
OUT = pathlib.Path(sys.argv[2])
RUNS = A["runs"]


def stat(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    a = np.asarray(v, float)
    return {"n": int(a.size), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max()), "min": float(a.min())}


def frames(rd, key="seq"):
    """planning rows by seq (E3b: 1..75) or by t_step (E0 records use seq == t_step)."""
    rows = [json.loads(l) for l in (pathlib.Path(rd) / "frames.jsonl").read_text().splitlines() if l.strip()]
    return {r[key]: r for r in rows if r["kind"] == "planning"}


res = {"definitions": __doc__, "per_pair": [], "per_chunk_k": {}}
by_latch, by_verdict, by_gate = {}, {}, {}
cache = {}
for p in A["pairs"]:
    r0, r3 = RUNS[p["E0_run"]], RUNS[p["E3b_run"]]
    z0, z3 = np.load(pathlib.Path(r0["run_dir"]) / "traj.npz"), np.load(pathlib.Path(r3["run_dir"]) / "traj.npz")
    x0, x3 = z0["executed_actions"], z3["executed_actions"]
    p0, p3 = frames(r0["run_dir"], "t_step"), frames(r3["run_dir"])
    e = {"episode": p["episode"], "seed": p["key"], "E0_success": p["E0"]["success"], "E3b_success": p["E3b"]["success"], "chunks": {}}
    for k in range(1, 6):
        ref = np.asarray(p3[k]["actions_reference"], np.float64)[:8]; act = np.asarray(p3[k]["actions"], np.float64)[:8]
        sl = slice(8 * (k - 1), 8 * k)
        e["chunks"][k] = {"a_exec": float(np.abs(x3[sl] - ref).max()), "a_refined": float(np.abs(act - ref).max()),
                          "b": float(np.abs(ref - x0[sl]).max()),
                          "E3b_gate": (p3[k].get("gate") or {}).get("action")}
        if k in (2, 3):
            e["chunks"][k]["c_qpos_full66"] = float(np.abs(np.asarray(p0[8 * (k - 1)]["qpos"]) - np.asarray(p3[k]["qpos"])).max())
            e["chunks"][k]["c_state16"] = float(np.abs(z0["inference_states"][k - 1] - z3["inference_states"][k - 1]).max())
    res["per_pair"].append(e)
    # all E3b chunks of this run: a_exec by latch / verdict
    ch_by_seq = {c["seq"]: c for c in r3["chunks"]}
    for k, pr in p3.items():
        if pr.get("actions_reference") is None:
            continue
        ref = np.asarray(pr["actions_reference"], np.float64)[:8]; sl = slice(8 * (k - 1), 8 * k)
        if x3[sl].shape[0] != 8:
            continue
        a = float(np.abs(x3[sl] - ref).max())
        c = ch_by_seq.get(k, {})
        latch = None
        if c.get("server_chunk"):
            zz = np.load(c["server_chunk"], allow_pickle=True)
            latch = (json.loads(str(zz["summary_json"])).get("grasp") or {}).get("state")
        g = (pr.get("gate") or {}).get("action")
        vk = "+".join(sorted({x.get("kind") for x in (pr.get("verdict_reasons") or []) if isinstance(x, dict)})) or "none"
        by_latch.setdefault(str(latch), []).append(a)
        by_verdict.setdefault(f"{g}:{vk}", []).append(a)
        by_gate.setdefault(str(g), []).append(a)
for k in range(1, 6):
    rows = [e["chunks"][k] for e in res["per_pair"]]
    d = {key: stat([r[key] for r in rows]) for key in ("a_exec", "a_refined", "b")}
    if k in (2, 3):
        d["c_qpos_full66"] = stat([r["c_qpos_full66"] for r in rows]); d["c_state16"] = stat([r["c_state16"] for r in rows])
    d["n_pairs_E3b_gate_hold"] = sum(1 for r in rows if r["E3b_gate"] == "hold")
    d["n_a_exec_gt_1e-3_and_b_lt_1e-4"] = sum(1 for r in rows if r["a_exec"] > 1e-3 and r["b"] < 1e-4)
    d["fraction_a_exec_gt_1e-3_and_b_lt_1e-4"] = d["n_a_exec_gt_1e-3_and_b_lt_1e-4"] / len(rows)
    d["n_b_gt_1e-4"] = sum(1 for r in rows if r["b"] > 1e-4)
    res["per_chunk_k"][k] = d
res["a_exec_all_E3b_chunks"] = {"by_latch_state": {k: stat(v) for k, v in sorted(by_latch.items())},
                                "by_gate_action": {k: stat(v) for k, v in sorted(by_gate.items())},
                                "by_gate_and_verdict_kinds": {k: stat(v) for k, v in sorted(by_verdict.items())}}
# 2. HOLD vs outcome
e3 = {n: r for n, r in RUNS.items() if r["cond"] == "E3b"}
def outc(rs):
    return {"n_runs": len(rs), "success": sum(r["success"] for r in rs), "grasp": sum(r["grasp"] for r in rs), "place": sum(r["place"] for r in rs)}
res["hold_vs_outcome"] = {"with_hold": outc([r for r in e3.values() if r["n_hold_chunks"] > 0]),
                          "without_hold": outc([r for r in e3.values() if r["n_hold_chunks"] == 0])}
kinds = {}
for r in e3.values():
    ks = {k for c in r["chunks"] if c["hold"] for k in (c["hold_kinds"] or [])}
    for k in ks:
        kinds.setdefault(k, []).append(r)
res["hold_kind_runs"] = {k: dict(outc(v), runs=sorted(x["run_dir"].split("/")[-1] for x in v)) for k, v in sorted(kinds.items())}
disc = []
for kind, ps in A["discordant"].items():
    for p in ps:
        r3 = RUNS[p["E3b_run"]]
        hold_t = [c["t"] for c in r3["chunks"] if c["hold"]]
        g = r3["grasp_t"]
        disc.append({"kind": kind, "episode": p["episode"], "seed": p["key"], "E3b_n_hold_chunks": len(hold_t),
                     "E3b_first_hold_t": hold_t[0] if hold_t else None, "E3b_grasp_t": g,
                     "first_hold_minus_grasp_t": (hold_t[0] - g) if (hold_t and g is not None) else None,
                     "E3b_hold_kinds": r3["hold_kind_counts"], "E3b_success": r3["success"], "E3b_grasp": r3["grasp"], "E0_grasp_t": RUNS[p["E0_run"]]["grasp_t"]})
res["discordant_hold"] = disc
res["discordant_hold_counts"] = {k: {"n": sum(1 for d in disc if d["kind"] == k), "with_hold": sum(1 for d in disc if d["kind"] == k and d["E3b_n_hold_chunks"] > 0)}
                                 for k in ("E3b_fail_E0_success", "E0_fail_E3b_success")}
# 3. latency
rt_minus = [c["client_round_trip_ms"] - c["server_ms"]["total"] for r in e3.values() for c in r["chunks"][1:]
            if c["client_round_trip_ms"] is not None and (c["server_ms"] or {}).get("total") is not None]
res["latency_E3b_ms_excl_first"] = {"round_trip_minus_server_total": stat(rt_minus),
                                    "round_trip": stat([c["client_round_trip_ms"] for r in e3.values() for c in r["chunks"][1:]]),
                                    "server_total": stat([(c["server_ms"] or {}).get("total") for r in e3.values() for c in r["chunks"][1:]]),
                                    "server_infer_policy_plus_attention_copy": stat([(c["server_ms"] or {}).get("infer") for r in e3.values() for c in r["chunks"][1:]]),
                                    "policy_vs_attention_copy_split": "not recorded (one timer around AttentionPolicy.infer, safe_policy.py:342-345)"}
res["latency_E0_ms_excl_first"] = {"server_time": "not recorded (--no-safe server records no timing)",
                                   "round_trip_upper_bound_on_policy_time": stat([c["client_round_trip_ms"] for r in RUNS.values() if r["cond"] == "E0" for c in r["chunks"][1:]])}
OUT.write_text(json.dumps(res, indent=1, default=float))
for k, d in res["per_chunk_k"].items():
    print(k, {kk: (round(v["median"], 8), round(v["max"], 8)) if isinstance(v, dict) else v for kk, v in d.items()})
print(json.dumps({k: (v["n"], round(v["median"], 5), round(v["max"], 4)) for k, v in res["a_exec_all_E3b_chunks"]["by_latch_state"].items()}))
print(json.dumps({k: (v["n"], round(v["median"], 5), round(v["max"], 4)) for k, v in res["a_exec_all_E3b_chunks"]["by_gate_and_verdict_kinds"].items()}))
print(res["hold_vs_outcome"], {k: (v["n_runs"], v["success"]) for k, v in res["hold_kind_runs"].items()}, res["discordant_hold_counts"])
for d in disc:
    print(d["kind"], d["episode"], d["seed"], d["E3b_n_hold_chunks"], d["E3b_first_hold_t"], d["E3b_grasp_t"], d["first_hold_minus_grasp_t"])
print({k: v for k, v in res["latency_E3b_ms_excl_first"].items()})
print(res["latency_E0_ms_excl_first"])
