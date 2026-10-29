"""T43 W (next round) — offline choice of the lighter sweep for V3 (= a + c + held body cover + --servo-model --sweep-check).
Verifier, CPU only, code = snapshot ${AG3S_PROJECT_ROOT}-t43v (benchmark f6af67c), harness = tests/trajopt/_t43q_replay.py and
tests/sim/_t43q_exec.py (T43 Q, unchanged; imported from ${AG3S_PROJECT_ROOT}/tests).

usage: tune.py detect OUT_JSON            detection on the recorded contact chunks (all candidates)
       tune.py timing OUT_JSON REPS [CANDIDATES]   solve time on T43 V V2 hurdle chunks (122-sphere model = no --gripper-cover)

Candidates (servo model on in all; 'off' = no servo / no sweep, i.e. V4; 'default' = Q defaults 12 mm / 16 / 24):
  name = m<sweep_max_mm>_s<sweep_max_samples>_r<sweep_rows_per_step>.
detect : chunks T40 G C1807 seq 28 (margin 0 and 0.010) and T41 a C1807 seq 35 (margin 0.010) — recorded plans touched the bar in MuJoCo.
         (1) check_plan of the RECORDED plan: path_clearance_m (servo / sweep rows) and waypoint clearance;
         (2) replay (re-solve, recorded SQP iterations, no clock): status, waypoint clearance, path_check clearance, solve ms;
         (3) MuJoCo execution of the re-solved plan from the bit-exact replayed state (_t43q_exec.execute, 8 rows): client contact flag,
             min true GJK hand-bar distance (mm).
timing : T43 V V2 hurdle runs (server_8280_20261005T134908, run_0000): seq 4, 10, ..., 70 of the T40-seed run of each episode
         (chunks with an attached object skipped), margin 0.010, recorded SQP iterations, no clock; each (chunk, candidate) solved REPS
         times interleaved, the minimum kept (shared pod). solve_ms = TrajOptResult.solve_time_ms (whole sqp.solve)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, time, pathlib
sys.path.insert(0, "${AG3S_PROJECT_ROOT}-t43v")
sys.path.append(_ag3s_workspace_path('tests/trajopt')); sys.path.append(_ag3s_workspace_path('tests/sim'))
import numpy as np
import benchmark
assert benchmark.__file__.startswith("${AG3S_PROJECT_ROOT}-t43v/"), benchmark.__file__
import _t43q_replay as R  # noqa: E402

V2S = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V/runs/V2/server_8280_20261005T134908'))
V2R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V/runs/V2/H'))
for nm, run, first in (("V2H1807", "V2_ep1807_s18071", 0), ("V2H1968", "V2_ep1968_s19681", 150),
                       ("V2H1982", "V2_ep1982_s19822", 300), ("V2H1967", "V2_ep1967_s19672", 450)):
    R.RUNS[nm] = (V2S, V2R / run, first - 1, 0.010)


def cand(mm, s, r):
    return {"limits": {"servo_model": True}, "reduction": {"sweep_check": True, "sweep_max_m": mm / 1000.0,
                                                            "sweep_max_samples": s, "sweep_rows_per_step": r}}


CANDS = {"off": None, "servo_only": {"limits": {"servo_model": True}}, "default": cand(12, 16, 24)}
for mm in (12, 24, 36):
    for s in (8, 4, 2):
        for r in (12, 6):
            CANDS[f"m{mm}_s{s}_r{r}"] = cand(mm, s, r)


def detect(out):
    import _t43q_exec as E
    cases = [("T40C1807", 28, 0.0), ("T40C1807", 28, 0.010), ("T41aC1807", 35, 0.010)]
    res = {"definitions": __doc__, "benchmark": benchmark.__file__, "rows": []}
    loaded = {}
    for name, seq, margin in cases:
        t0 = int(R._setup(name)["rows"][seq]["t_step"])
        if name not in loaded:
            loaded[name] = E.load(R.RUNS[name][1], t0)
        client, local, _ = loaded[name]
        recorded = np.asarray(R.npz(name, seq)["refined_chunk"], np.float64)
        ex0 = E.execute(client, local, recorded)
        for cn, ov in CANDS.items():
            row = {"chunk": f"{name}:{seq}", "margin_m": margin, "cand": cn,
                   "recorded_exec": {"contact": ex0["contact"], "true_mm_min": ex0["true_mm_min"]}}
            if ov is not None:
                c = R.check_plan(name, seq, recorded, overrides=ov, obstacle_margin=margin)
                row["recorded_plan_path_clearance_mm"] = 1e3 * c["path_clearance_m"]
                row["recorded_plan_waypoint_clearance_mm"] = 1e3 * c["waypoint_clearance_m"]
            t = time.perf_counter(); rp = R.replay(name, seq, overrides=ov, obstacle_margin=margin)
            row["resolve"] = {"status": rp["status"], "safe": rp["safe"], "waypoint_clearance_mm": 1e3 * rp["clearance_m"],
                              "path_clearance_mm": None if not rp["metrics"].get("path_check") else 1e3 * rp["metrics"]["path_check"]["clearance_m"],
                              "n_configs": (rp["metrics"].get("path_check") or {}).get("n_configs"),
                              "initial_source": (rp["metrics"].get("initial_source") or {}).get("source") if isinstance(rp["metrics"].get("initial_source"), dict) else rp["metrics"].get("initial_source"),
                              "solve_ms": rp["result"].solve_time_ms, "wall_ms": 1e3 * (time.perf_counter() - t)}
            ex = E.execute(client, local, rp["chunk"])
            row["resolved_exec"] = {"contact": ex["contact"], "true_mm_min": ex["true_mm_min"], "true_mm_q_now": ex["true_mm_q_now"],
                                    "first_contact_row_substep": ex["first_contact_row_substep"]}
            res["rows"].append(row); print(json.dumps({k: v for k, v in row.items()}), flush=True)
            pathlib.Path(out).write_text(json.dumps(res, indent=1, default=float))


def timing(out, reps, names):
    chunks = []
    for nm in ("V2H1807", "V2H1968", "V2H1982", "V2H1967"):
        for seq in range(4, 75, 6):
            s = R.summary(nm, seq)
            if int((s.get("esdf_stats") or {}).get("n_attached_points", 0) or 0) > 0 or (s.get("grasp") or {}).get("attached"):
                continue
            chunks.append((nm, seq))
    C = {k: CANDS[k] for k in names}
    for k, ov in C.items():                       # warm-up (FK maps, CasADi)
        R.replay(chunks[0][0], chunks[0][1], overrides=ov)
    rows = []
    for nm, seq in chunks:
        rec = {"chunk": f"{nm}:{seq}"}
        best = {}
        for _ in range(reps):
            for k, ov in C.items():
                r = R.replay(nm, seq, overrides=ov)
                t = r["result"].solve_time_ms
                if k not in best or t < best[k]["solve_ms"]:
                    pc = r["metrics"].get("path_check") or {}
                    best[k] = {"solve_ms": t, "status": r["status"], "iters": r["metrics"].get("sqp_iterations"),
                               "waypoint_clearance_mm": 1e3 * r["clearance_m"],
                               "path_clearance_mm": None if not pc else 1e3 * pc["clearance_m"], "n_configs": pc.get("n_configs"),
                               "timing_ms": r["metrics"].get("timing_ms")}
        rec.update(best); rows.append(rec); print(json.dumps({"chunk": rec["chunk"], **{k: round(v["solve_ms"]) for k, v in best.items()}}), flush=True)
        summ = {}
        for k in C:
            v = np.array([x[k]["solve_ms"] for x in rows])
            summ[k] = {"n": len(v), "median_ms": float(np.median(v)), "p90_ms": float(np.quantile(v, 0.9)), "max_ms": float(v.max()),
                       "violated": sum(x[k]["status"] != "feasible" for x in rows)}
        pathlib.Path(out).write_text(json.dumps({"definitions": __doc__, "reps": reps, "rows": rows, "summary": summ}, indent=1, default=float))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "detect":
        detect(sys.argv[2])
    else:
        timing(sys.argv[2], int(sys.argv[3]), sys.argv[4].split(",") if len(sys.argv) > 4 else list(CANDS))
