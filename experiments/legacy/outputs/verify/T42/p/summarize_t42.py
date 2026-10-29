"""T42-1 summary (verifier, 2026-10-05). CPU only.
usage: summarize_t42.py OUT_JSON
- every finished run (NAME.done) under outputs/verify/T42/1/runs/{E3b,E0}/noobs -> analyze_t42.analyze(series=True)
- robust grasp per run: grasp_robust.robust (full, with series)
- apple + left rep 1-2: the T39 runs outputs/verify/T39/runs/{E0,E3b}/<NAME> (snapshot ${AG3S_PROJECT_ROOT}-o1, benchmark 9fab500),
  analysed with the same analyze_t42 rule (source 'T39')
- reproduction: E0 / E3b ep1828 s18281 here vs T39 (and T40 G B) — traj.npz executed_actions and measured_qpos
  bitwise equality, frames.jsonl control-row qpos max |diff|
- per (fruit, arm) and episode: E3b / E0 results per rep (1-3), counts, library = E3b 3/3, selected = plan.select()
- server process facts per server record dir: cmdline flags (--no-subtask-gate present), XLA_FLAGS, cwd"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/p'))
import analyze_t42 as AN  # noqa: E402
import plan as PL  # noqa: E402
import grasp_robust as GR  # noqa: E402

R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T42/1/runs'))
T39R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T39/runs'))
T40B = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/G/runs'))


def runs_here():
    out = {}
    for cond in ("E3b", "E0"):
        d = R / cond / "noobs"
        for f in sorted(d.glob("*.done")) if d.exists() else []:
            name = f.name[:-5]; rd = d / name
            a = AN.analyze(rd, series=True); a["cond"] = cond; a["source"] = "T42"
            a["robust"] = GR.robust(rd)
            try:
                a["wall_s"] = None
                import datetime as dt
                st = dt.datetime.fromisoformat((d / f"{name}.start").read_text().splitlines()[0].strip())
                en = dt.datetime.fromisoformat((d / f"{name}.end").read_text().splitlines()[0].strip())
                a["wall_s"] = (en - st).total_seconds()
            except Exception:
                pass
            out[name] = a
    return out


def bitcmp(a, b):
    za, zb = np.load(pathlib.Path(a) / "traj.npz"), np.load(pathlib.Path(b) / "traj.npz")
    r = {}
    for k in ("executed_actions", "measured_qpos", "predicted_chunks"):
        if k in za.files and k in zb.files:
            x, y = za[k], zb[k]
            r[k] = {"shape_equal": x.shape == y.shape, "bit_identical": bool(x.shape == y.shape and np.array_equal(x, y)),
                    "max_abs_diff": float(np.abs(x - y).max()) if x.shape == y.shape else None}
    r["all_bit_identical"] = all(v["bit_identical"] for v in r.values() if isinstance(v, dict))
    return r


def main():
    out = pathlib.Path(sys.argv[1])
    runs = runs_here()
    # T39 apple-left rep 1-2 of the screening episodes
    for e in PL.screening()[("apple", "left")]:  # T39 rep 1-2
        for rep in (1, 2):
            for cond in ("E3b", "E0"):
                name = f"{cond}_ep{e}_s{10 * e + rep}"; rd = T39R / cond / name
                if rd.exists() and (T39R / cond / f"{name}.done").exists():
                    a = AN.analyze(rd, series=True); a["cond"] = cond; a["source"] = "T39"; a["robust"] = GR.robust(rd)
                    runs[f"T39/{name}"] = a
    repro = {}
    for cond in ("E3b", "E0"):
        name = f"{cond}_ep1828_s18281"; here = R / cond / "noobs" / name
        if (R / cond / "noobs" / f"{name}.done").exists():
            repro[cond] = {"vs_T39": bitcmp(here, T39R / cond / name), "vs_T40_G_B": bitcmp(here, T40B / cond / "B" / name)}
    table = {}
    for (f, a), eps in PL.screening().items():
        for e in eps:
            row = {"fruit": f, "arm": a, "score_m": PL.score(e), "split": PL.split_of(e), "E3b": {}, "E0": {}}
            for cond in ("E3b", "E0"):
                for rep in (1, 2, 3):
                    s = 10 * e + rep
                    k = f"T39/{cond}_ep{e}_s{s}" if ((f, a) == ("apple", "left") and rep < 3) else f"{cond}_ep{e}_s{s}"
                    r = runs.get(k)
                    rb = None if r is None else r["robust"]
                    row[cond][str(rep)] = None if r is None else {"seed": s, "success": r["success"], "grasp": r["grasp"], "place": r["place"],
                                                                  "source": r["source"], "n_hold_chunks": r["n_hold_chunks"],
                                                                  "non_target_in_crate_final": r["non_target_in_crate_final"],
                                                                  "robust_grasp": rb["robust_grasp"], "slip": rb.get("slip"), "regrasp": rb.get("regrasp"),
                                                                  "drift_mm_max": rb.get("drift_mm_max"), "finger_gap_mm_max": rb.get("finger_gap_mm_max"),
                                                                  "n_rows_no_finger_within_3mm": rb.get("n_rows_no_finger_within_3mm"),
                                                                  "n_close_before_lift": rb.get("n_close_before_lift"), "lift_t": rb.get("lift_t"),
                                                                  "release_t": rb.get("release_t"), "slip_B": rb.get("slip_B"),
                                                                  "loss_exempt_as_place": rb.get("loss_exempt_as_place"),
                                                                  "robust_grasp_strict": rb.get("robust_grasp_strict"),
                                                                  "first_slip_t": rb.get("first_slip_t"),
                                                                  "first_slip_xy_from_reset_m": rb.get("first_slip_xy_from_reset_m"),
                                                                  "first_slip_z_above_table_m": rb.get("first_slip_z_above_table_m"),
                                                                  "failure_mode": GR.failure_mode(rb)}
                done = [v for v in row[cond].values() if v is not None]
                row[f"{cond}_n_done"] = len(done); row[f"{cond}_n_success"] = sum(v["success"] for v in done)
                row[f"{cond}_n_success_robust"] = sum(v["success"] and v["robust_grasp"] for v in done)
                row[f"{cond}_n_success_robust_strict"] = sum(v["success"] and v["robust_grasp_strict"] for v in done)
            table[str(e)] = row
    servers = {}
    for p in sorted(R.glob("*/server_*/server_proc.txt")):
        t = p.read_text()
        servers[str(p.parent)] = {"no_subtask_gate_flag": "--no-subtask-gate" in t, "autotune0": "--xla_gpu_autotune_level=0" in t,
                                  "cwd_snapshot": "cwd=${AG3S_PROJECT_ROOT}-t42" in t,
                                  "pythonpath_snapshot": "PYTHONPATH=${AG3S_PROJECT_ROOT}-t42" in t,
                                  "log_autotuning_off": "XLA autotuning off" in (p.parent / "serve_safe.log").read_text(errors="replace")}
    lib = {f"{f}_{a}": v for (f, a), v in PL.library().items()}
    lib_strict = {}
    for (f, a), eps in PL.screening().items():
        lib_strict[f"{f}_{a}"] = [int(e) for e in eps if table[str(e)]["E3b_n_done"] == 3 and table[str(e)]["E3b_n_success_robust_strict"] == 3]
    fm = {}
    for e, row in table.items():
        for rep in ("1", "2", "3"):
            v = row["E3b"][rep]
            if v is not None:
                k = f'{row["fruit"]}_{row["arm"]}'; fm.setdefault(k, {}); fm[k][v["failure_mode"]] = fm[k].get(v["failure_mode"], 0) + 1
    res = {"definitions": __doc__ + "\n\nanalysis rule:\n" + AN.__doc__ + "\n\nrobust grasp:\n" + GR.__doc__, "runs": runs, "table": table,
           "library_E3b_3of3": lib, "library_E3b_3of3_strict_A": lib_strict, "E3b_failure_modes_per_fruit_arm": fm,
           "selected": PL.select(), "reproduction_ep1828_s18281": repro, "servers": servers}
    out.write_text(json.dumps(res, indent=1, default=float))
    for e, row in table.items():
        print(e, row["split"], row["fruit"], row["arm"], "E3b", row["E3b_n_success"], "robust", row["E3b_n_success_robust"], "/", row["E3b_n_done"],
              "E0", row["E0_n_success"], "robust", row["E0_n_success_robust"], "/", row["E0_n_done"])
    print("library", lib); print("selected", res["selected"]); print("repro", json.dumps({c: {k: v["all_bit_identical"] for k, v in d.items()} for c, d in repro.items()}))


if __name__ == "__main__":
    main()
