"""SUBTASK-g — collect compare/<pair>.json (p/compare_v0.py) and per-chunk decision fields of both sides into analysis/summary.json.
Decision fields per chunk (server record summary_json of the run's segment, last segment if several):
  status, validity, grounding_status, has_target, verdict reasons [(kind, action)], grasp.state, manipulated.id, exclusion (whole),
  admissibility.decision.mode, subtask.gate, subtask.label, subtask.blocked, subtask.released, subtask.placed_seen.
Per field: number of chunks whose value differs between the two sides, first differing t_step.
Per side: success / grasp / place (T28/E0 analyze_e0.analyze, same as SUBTASK-f analyze_v2), verdict kind/action counts, grounding_status
counts, HOLD chunks, set of subtask.gate values, startup gate lines of the server log (new runs only).
usage: JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa summarize.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, collections
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/SUBTASK-g/p'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/E0'))
import compare_v0 as CV  # noqa: E402
import analyze_e0 as A  # noqa: E402
G = pathlib.Path(_ag3s_workspace_path('outputs/verify/SUBTASK-g'))
PAIRS = ["D_vs_SFon_ep1807_s18071", "D_vs_SFon_ep1800_s18001", "N_vs_T39_ep1807_s18071", "N_vs_T39_ep1800_s18001",
         "N_vs_SFoff_ep1807_s18071", "N_vs_SFoff_ep1800_s18001"]
FIELDS = {
    "status": lambda s: s.get("status"), "validity": lambda s: s.get("validity"), "grounding_status": lambda s: s.get("grounding_status"),
    "has_target": lambda s: s.get("has_target"),
    "verdict_reasons": lambda s: [(r.get("kind"), r.get("action")) for r in ((s.get("verdict") or {}).get("reasons") or [])],
    "grasp.state": lambda s: (s.get("grasp") or {}).get("state"), "manipulated.id": lambda s: (s.get("manipulated") or {}).get("id"),
    "exclusion": lambda s: s.get("exclusion"), "admissibility.decision.mode": lambda s: (((s.get("admissibility") or {}).get("decision")) or {}).get("mode"),
    "subtask.gate": lambda s: (s.get("subtask") or {}).get("gate"), "subtask.label": lambda s: (s.get("subtask") or {}).get("label"),
    "subtask.blocked": lambda s: (s.get("subtask") or {}).get("blocked"), "subtask.released": lambda s: (s.get("subtask") or {}).get("released"),
    "subtask.placed_seen": lambda s: (s.get("subtask") or {}).get("placed_seen")}


def chunks(run):
    segs = CV.server_segment(pathlib.Path(run))
    out = []
    for f in (segs[-1] if segs else []):
        with np.load(f, allow_pickle=False) as z:
            out.append((int(z["t_step"]), json.loads(str(z["summary_json"]))))
    return out, len(segs)


def side(run):
    ch, nseg = chunks(run)
    base = A.analyze(run)
    kinds = collections.Counter(); acts = collections.Counter(); gs = collections.Counter(); hold = 0
    for t, s in ch:
        rs = (s.get("verdict") or {}).get("reasons") or []
        for r in rs:
            kinds[r.get("kind")] += 1; acts[r.get("action")] += 1
        hold += any(r.get("action") == "hold" for r in rs)
        gs[s.get("grounding_status")] += 1
    return ch, {"run": str(run), "n_server_segments": nseg, "n_chunks": len(ch),
                **{k: base[k] for k in ("success", "grasp", "place")},
                "verdict_kind_counts": dict(kinds), "verdict_action_counts": dict(acts), "n_hold_chunks": hold,
                "grounding_status_counts": dict(gs),
                "subtask_gate_values": sorted({json.dumps((s.get("subtask") or {}).get("gate")) for _, s in ch}),
                "n_chunks_released": sum(bool((s.get("subtask") or {}).get("released")) for _, s in ch),
                "n_chunks_subtask_blocked": sum((s.get("subtask") or {}).get("blocked") is not None for _, s in ch),
                "n_chunks_subtask_key": sum("subtask" in s for _, s in ch)}


out = {"definitions": __doc__, "pairs": {}}
for p in PAIRS:
    f = G / "compare" / f"{p}.json"
    if not f.exists():
        out["pairs"][p] = {"missing": str(f)}; continue
    r = json.loads(f.read_text())
    ca, sa = side(r["run_t39"]); cb, sb = side(r["run_new"])
    fd = {}
    for name, fn in FIELDS.items():
        n = 0; first = None
        for (ta, a), (tb, b) in zip(ca, cb):
            if json.dumps(fn(a), sort_keys=True) != json.dumps(fn(b), sort_keys=True):
                n += 1; first = ta if first is None else first
        fd[name] = {"n_chunks_differ": n, "first_t": first}
    cl = r["classified"]
    out["pairs"][p] = {
        "compare_json": str(f), "ref": sa, "new": sb, "t_steps_equal": [t for t, _ in ca] == [t for t, _ in cb],
        "executed_actions_equal": r["executed_actions_equal"], "first_executed_step_diff": r["first_executed_step_diff"],
        "planning_actions_equal": r["planning_actions_equal"], "traj_npz_keys_differing": sorted(r["traj_npz_differ"]),
        "traj_npz_differ_other": r["traj_npz_differ_other"], "frames_rows": r["frames_n_rows"],
        "frames_other_leaf_diff": cl["frames_leaf_diff"]["other"], "frames_subtask_leaf_diff": cl["frames_leaf_diff"]["subtask"],
        "manifest_other_leaf_diff": cl["manifest_leaf_diff"]["other"], "manifest_subtask_leaf_diff": cl["manifest_leaf_diff"]["subtask"],
        "server_n_chunks": r.get("server_n_chunks"), "server_segments_found": r.get("server_segments_found"),
        "server_array_diff": r.get("server_array_diff"), "server_array_diff_other": r.get("server_array_diff_other"),
        "server_summary_other_leaf_diff": cl["server_summary_leaf_diff"]["other"],
        "server_summary_subtask_leaf_diff": cl["server_summary_leaf_diff"]["subtask"],
        "named_per_chunk": r.get("named_per_chunk"), "decision_field_diff": fd,
        "n_decision_fields_differing": sum(v["n_chunks_differ"] > 0 for v in fd.values()),
        "compare_v0_all_identical_flag": r["all_identical"]}
    for k, leaf in (("manifest_other_values", "scene.model_xml"),):
        if leaf in cl["manifest_leaf_diff"]["other"]:
            out["pairs"][p][k] = {leaf: [json.load(open(x + "/manifest.json"))["scene"]["model_xml"] for x in (r["run_t39"], r["run_new"])]}
    if "chunk_index" in (r.get("server_array_diff") or {}):
        fa, fb = r["server_files"]["t39"][0], r["server_files"]["new"][0]
        out["pairs"][p]["chunk_index_first"] = [int(np.load(fa)["chunk_index"]), int(np.load(fb)["chunk_index"])]
for c in ("E3bD", "E3bN"):
    out.setdefault("server_startup", {})[c] = {str(d): (d / "gate_lines.txt").read_text().splitlines()
                                               for d in sorted((G / "runs" / c).glob("server_*")) if (d / "gate_lines.txt").exists()}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))
for p, v in out["pairs"].items():
    if "missing" in v:
        print(p, "MISSING"); continue
    print(p, "exec", v["executed_actions_equal"], "plan", v["planning_actions_equal"], "dec_fields_diff", v["n_decision_fields_differing"],
          {k: x["n_chunks_differ"] for k, x in v["decision_field_diff"].items() if x["n_chunks_differ"]},
          "other:", v["frames_other_leaf_diff"], v["manifest_other_leaf_diff"], v["server_summary_other_leaf_diff"], v["server_array_diff_other"],
          "S/G/P", [v["ref"][k] for k in ("success", "grasp", "place")], [v["new"][k] for k in ("success", "grasp", "place")],
          "gate", v["ref"]["subtask_gate_values"], v["new"]["subtask_gate_values"])
