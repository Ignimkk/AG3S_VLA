"""T35-diag C — perception / identity per chunk around the first close command, all 12 T34 runs.

usage: MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} <py> c_perception.py -> c/c_perception.json
Sources: T34 p/<COND>_analysis.json (manipulated, exclusion, ball, grasp_latch, manip_extra) and p/<COND>_analysis_extra.json
(association, anchor, decision, frozen, ball_vs_gt_apple) — same definitions as T34 (analyze_e3_base.py / analyze_e3_extra.py).
Window: local seq s_close-10 .. s_close+10 (s_close from a/a_runs.json t_close). Also whole run up to t_close ('pre_close').
Per chunk:
  manip            : manipulated {id, state, to_apple_mm (centroid -> GT apple centre), nearest_centre, admissible_reason}
  manip_is_apple   : nearest_centre == 'apple'  (T28 rule)
  association      : extra association outcome ; decision reason ; anchor source ; switched_from
  split            : manip_extra split_n_triggered / split_n_split
  hand             : manip_extra.hand {occluded_by_hand, gap_mm}
  exclusion        : source / active / grasp_active / mechanism
  ball             : extra ball_vs_gt_apple {radius_mm, apple_centre_dist_mm, centre_inside, whole_inside (+34 mm)}
  held cover       : held chunks — recorded held.spheres_parent_frame ('query', T34 lifted spheres) and
                     held.filter_spheres_parent_frame ('filter', fit + pad) through constraint-model FK at q_now (fixed_input_ablation
                     build_robot) -> base -> world (P1-2 base_to_world); GT apple (planning-row qpos[38:41]):
                     max over apple surface of the distance outside the sphere union, sampled as the apple centre +
                     34 mm * 162 unit directions; 'whole_inside' = no sample outside any sphere; also the body-sphere
                     centre-to-apple-centre distance.
  latch            : grasp_latch state / attach / detach
Counts per run and window: n chunks manip_is_apple False, n occluded_by_hand, n split triggered / accepted, n ball whole_inside
False (when a ball exists), n frozen, association outcome counts.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
from collections import Counter
import numpy as np

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P  # noqa
import AG3S.trajopt.experiments.fixed_input_ablation as T  # noqa

V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T34'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T35diag/c')); OUT.mkdir(parents=True, exist_ok=True)
ARUNS = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T35diag/a/a_runs.json')).read_text())["runs"]
PORT = {"E3a": 8204, "E3b": 8206}
APPLE_R = 0.034

m, d, probe, objg, _ = P.build()
g = np.random.default_rng(0).normal(size=(162, 3)); DIRS = g / np.linalg.norm(g, axis=1, keepdims=True)

out = {"definitions": __doc__, "runs": {}}
rob = None
for cond in ("E3a", "E3b"):
    A = json.loads((V / "p" / f"{cond}_analysis.json").read_text())["runs"]
    X = json.loads((V / "p" / f"{cond}_analysis_extra.json").read_text())["runs"]
    for run in sorted(A):
        rec = V / f"{cond}_rec"
        if rob is None:
            args = T.build_parser().parse_args(["--constraints", str(rec / f"server_{PORT[cond]}/server_constraints/run_0000"),
                                                "--frames", str(rec / run / "frames.jsonl"), "--server-log", str(rec / f"serve_safe_{PORT[cond]}.log"), "--out", "/dev/null"])
            settings = T.resolve_settings(args, T.parse_server_log(args.server_log), json.loads((rec / run / "manifest.json").read_text()))
            rob = T.build_robot(settings)
        rb, qadr, _ = rob
        ar = ARUNS[f"{cond}_rec/{run}"]
        t_close = ar["t_close"]
        s_close = next(e["seq"] for e in ar["timeline"] if e["t"] == t_close)
        rows = {int(x["seq"]): x for x in (json.loads(l) for l in (rec / run / "frames.jsonl").read_text().splitlines() if l.strip())
                if x.get("kind") == "planning"}
        xs = {c["seq"]: c for c in X[run]}
        ch = []
        for c in A[run]["chunks"]:
            x = xs.get(c["seq"], {})
            me = c.get("manip_extra") or {}
            mn = c.get("manipulated") or {}
            bv = x.get("ball_vs_gt_apple")
            e = {"t": c["t"], "seq": c["seq"], "rel": c["seq"] - s_close, "phase": c.get("phase"),
                 "manip": {k: mn.get(k) for k in ("id", "state", "to_apple_mm", "nearest_centre", "admissible_reason", "narrowest_extent_mm")} if mn else None,
                 "manip_is_apple": (mn.get("nearest_centre") == "apple") if mn else False,
                 "association": (x.get("association") or {}).get("outcome") if isinstance(x.get("association"), dict) else x.get("association"),
                 "decision_reason": (x.get("decision") or {}).get("reason"), "decision_mode": (x.get("decision") or {}).get("mode"),
                 "anchor_source": (x.get("anchor") or {}).get("source"), "switched_from": x.get("manipulated_switched_from"),
                 "frozen": x.get("frozen"),
                 "split_n_triggered": me.get("split_n_triggered"), "split_n_split": me.get("split_n_split"),
                 "occluded_by_hand": (me.get("hand") or {}).get("occluded_by_hand"), "hand_gap_mm": (me.get("hand") or {}).get("gap_mm"),
                 "exclusion": c.get("exclusion"), "ball": bv,
                 "latch": {k: (c.get("grasp_latch") or {}).get(k) for k in ("state", "attach", "detach")},
                 "grounding_status": c.get("grounding_status")}
            if c.get("server_chunk") and (c.get("held") or {}).get("held") is not None and abs(e["rel"]) <= 10:
                z = np.load(rec / c["server_chunk"], allow_pickle=False)
                s = json.loads(str(z["summary_json"]))
                gr = s.get("grasp") or {}
                held = (gr.get("attached") or {}).get("held")
                if held and gr.get("attached_in_constraints"):
                    row = rows[c["seq"]]
                    q_now = T.q_now_from_qpos(row["qpos"], qadr)
                    Tq = np.asarray(rb.link_pose(q_now, held["frame_link"]), float)
                    P.set_q(m, d, row["qpos"])
                    ap = np.asarray(row["qpos"][38:41], float)
                    samples = ap + APPLE_R * DIRS
                    e["held_cover"] = {}
                    for key in ("spheres_parent_frame", "filter_spheres_parent_frame"):
                        L = held.get(key) or []
                        if not L:
                            continue
                        cw = [P.base_to_world(m, d, Tq[:3, :3] @ np.asarray(sp["centre_m"]) + Tq[:3, 3]) for sp in L]
                        rr = [sp["radius_mm"] / 1000 for sp in L]
                        outside = np.min(np.stack([np.linalg.norm(samples - cc, axis=1) - r for cc, r in zip(cw, rr)]), axis=0)
                        tag = "query" if key == "spheres_parent_frame" else "filter"
                        e["held_cover"][tag] = {"body_centre_to_apple_mm": float(1000 * np.linalg.norm(cw[0] - ap)),
                                                "body_centre_minus_apple_world_mm": (1000 * (cw[0] - ap)).tolist(), "radii_mm": [1000 * r for r in rr],
                                                "max_outside_mm": float(1000 * max(0.0, outside.max())), "n_samples_outside": int((outside > 0).sum()),
                                                "whole_inside": bool((outside <= 0).all())}
            ch.append(e)

        def cnt(rr):
            return {"n": len(rr), "n_manip_not_apple": sum(1 for e in rr if not e["manip_is_apple"]),
                    "n_manip_none": sum(1 for e in rr if e["manip"] is None),
                    "manip_state_counts": dict(Counter((e["manip"] or {}).get("state") for e in rr)),
                    "n_occluded_by_hand": sum(1 for e in rr if e["occluded_by_hand"]),
                    "n_split_triggered": sum(1 for e in rr if (e["split_n_triggered"] or 0) > 0),
                    "n_split_accepted": sum(1 for e in rr if (e["split_n_split"] or 0) > 0),
                    "n_ball": sum(1 for e in rr if e["ball"]), "n_ball_not_whole_inside": sum(1 for e in rr if e["ball"] and not e["ball"].get("whole_inside")),
                    "n_frozen": sum(1 for e in rr if e["frozen"]), "association_counts": dict(Counter(str(e["association"]) for e in rr)),
                    "decision_reason_counts": dict(Counter(str(e["decision_reason"]) for e in rr)),
                    "exclusion_source_counts": dict(Counter(str((e["exclusion"] or {}).get("source")) for e in rr)),
                    "n_switched": sum(1 for e in rr if e["switched_from"] is not None),
                    "max_to_apple_mm": max([(e["manip"] or {}).get("to_apple_mm") or 0 for e in rr] or [0]),
                    "held_cover": [dict(t=e["t"], **e["held_cover"]) for e in rr if e.get("held_cover")]}
        win = [e for e in ch if abs(e["rel"]) <= 10]
        pre = [e for e in ch if e["t"] <= t_close]
        out["runs"][run] = {"success": A[run]["summary"].get("success"), "t_close": t_close, "s_close": s_close,
                            "window": cnt(win), "pre_close": cnt(pre), "window_chunks": win}
        w = out["runs"][run]["window"]
        print(run, out["runs"][run]["success"], {k: w[k] for k in ("n", "n_manip_not_apple", "manip_state_counts", "n_occluded_by_hand", "n_split_triggered",
                                                                    "n_split_accepted", "n_ball", "n_ball_not_whole_inside", "n_frozen", "association_counts",
                                                                    "exclusion_source_counts", "n_switched", "max_to_apple_mm", "held_cover")}, flush=True)
(OUT / "c_perception.json").write_text(json.dumps(out, indent=1, default=float))
