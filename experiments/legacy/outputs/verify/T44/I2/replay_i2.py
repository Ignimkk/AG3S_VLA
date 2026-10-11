"""T44 I2 verifier — offline replay of recorded V11AB chunks, per-arm.

Re-uses `AG3S.trajopt.experiments.ta_offline.Run` (reconstruction of the TO input from a server
record) of whatever tree PYTHONPATH points at (HEAD 498f713 archive or the working tree).

arms (each solved open-loop from the same reconstructed input; refiner reset before each):
  off     V11AB config, scene as ta_offline builds it (no target_contact_rows)           [HEAD + WT]
  offR    same config, scene carries target_contact_rows, strict False                    [WT only]
  strict  same config, rows + target_permission_strict=True                               [WT only]
  sw      lookahead off + plan_horizon=execution window + reduction.sweep_check, no rows  [HEAD + WT]
  swR     sw with rows, strict False                                                       [WT only]
  swS     sw with rows, strict True                                                        [WT only]
sqp.time_budget_ms is raised to 1e6 in every arm (wall-clock budget is not a bit-reproducible input).

same-trajectory probes (WT only, pure functions of (trajectory, scene)):
  lookahead  tail rows kept by scene off vs strict on the reference chunk and on the off arm's result
  path       PathChecker.violation on the sw arm's result under scene R-off vs R-strict
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time

import numpy as np

from AG3S.trajopt.experiments import ta_offline as TA

K = TA.K_EXEC


def jsonable(o):
    return TA._jsonable(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--first-chunk", type=int, default=0)
    ap.add_argument("--probes", action="store_true")
    cli = ap.parse_args()
    cli.hm = False
    cli.no_plane_rows = False
    cli.variants = ["base"]

    from AG3S.trajopt import linearize as L
    from AG3S.trajopt.refiner import TrajOptChunkRefiner
    from AG3S.trajopt.safe_policy import classify_violations

    has_i2 = hasattr(L, "target_contact_rows_of")
    idx = json.load(open(cli.index))["runs"][cli.key]
    run = TA.Run(cli.key, idx, cli)
    base = run.base_cfg
    budget = {"sqp": {"time_budget_ms": 1.0e6}}
    cfgs = {
        "off": base.with_overrides(budget), "offR": base.with_overrides(budget),
        "strict": base.with_overrides({**budget, "collision": {"target_permission_strict": True}})
        if has_i2 else None,
    }
    swo = {"horizon": {"lookahead": "off", "lookahead_gate": None, "plan_horizon": None},
           "reduction": {"sweep_check": True}, **budget}
    cfgs["sw"] = base.with_overrides(swo)
    cfgs["swR"] = base.with_overrides(swo)
    cfgs["swS"] = (base.with_overrides({**swo, "collision": {"target_permission_strict": True}})
                   if has_i2 else None)
    need_rows = {"offR", "strict", "swR", "swS"}
    holder = {}
    refiners = {a: TrajOptChunkRefiner(run.robot, run.layout, lambda _c: holder["value"], cfgs[a])
                for a in cli.arms}
    # wrap PathChecker._target_rows: which links were named
    captured = {}
    for a in cli.arms:
        p = refiners[a].optimizer.path
        if p is not None:
            orig = p._target_rows

            def wrap(centres, scene, _o=orig, _p=p):
                out = _o(centres, scene)
                width = _p._floor.shape[0]
                nq = centres.shape[1]
                cols = {}
                if out is not None:
                    sub = out[:, width - nq:]
                    for q in np.flatnonzero(sub.any(axis=0)):
                        cols[_p.lin._query_name(int(q))] = int(sub[:, q].sum())
                captured["links"] = cols
                return out
            p._target_rows = wrap

    out = {"key": cli.key, "tool": "outputs/verify/T44/I2/replay_i2.py", "has_i2": has_i2,
           "arms": cli.arms, "n_spheres": int(run.robot.n_spheres),
           "authorized": run.authorized, "chunks": []}
    arrays = {}
    seqs = sorted(run.chunks)
    if cli.max_chunks:
        seqs = seqs[cli.first_chunk:cli.first_chunk + cli.max_chunks]
    prev_seq = None
    state = {}
    for seq in seqs:
        inp, why = run.inputs(seq)
        if inp is None:
            out["chunks"].append({"seq": seq, "skipped": why})
            prev_seq = None
            continue
        s, q_now, fb = inp["summary"], inp["q_now"], inp["feedback"]
        scene0 = inp["scene"]
        arm_fb = np.asarray(fb.get("applied_arm"), np.float64) if fb.get("applied_arm") is not None else None
        n_exec = fb.get("n_exec") if fb.get("available") else None
        dhist = None
        prev_chunk = None
        if prev_seq is not None and prev_seq == seq - 1:
            with np.load(run.chunks[prev_seq], allow_pickle=True) as pz:
                prev_chunk = np.asarray(pz["refined_chunk"], np.float64)
                prev_ref = np.asarray(pz["reference_chunk"], np.float64)
            ex = list(fb.get("executed") or [])
            if arm_fb is not None and ex and ex[-1]:
                dhist = arm_fb[-1] - run.layout.chunk_to_trajectory(prev_ref)[:, len(ex) - 1]
        ctx = {"t_step": seq, "previous_physical_chunk": prev_chunk}
        if n_exec is not None:
            ctx["previous_executed_steps"] = int(n_exec)
        rows = None
        if has_i2:
            manip = scene0.manipulated_points is not None or scene0.manipulated_spheres is not None
            rows = L.target_contact_rows_of(scene0.target_free_mask, None, scene0.manipulated_link_margin,
                                            "exclude_authorized", manipulated=manip)
        rec = {"seq": seq, "t0": int(inp["row"]["t_step"]), "has_target": bool(s.get("has_target")),
               "held": inp["held"] is not None, "n_contact_rows": None if rows is None else int(rows.sum()),
               "contact_rows_links": None if rows is None else
               [run.names[i] for i in np.flatnonzero(rows)], "arms": {}}
        results = {}
        for a in cli.arms:
            scene = scene0
            if a in need_rows:
                scene = dataclasses.replace(scene0, target_contact_rows=rows,
                                            target_permission_strict=a in ("strict", "swS"))
            cfg = cfgs[a]
            Hh = cfg.horizon.planned
            ref = refiners[a]
            ref.reset()
            lin = ref.optimizer.linearizer
            lin.set_joint_parameter_path(run.finger_path(s, inp["ref_chunk"], Hh))
            holder["value"] = (scene, q_now, True)
            c = dict(ctx)
            if dhist is not None and cfg.cost.w_deflection_rate > 0:
                c["deflection_history"] = dhist
            captured.clear()
            t0 = time.process_time()
            own = ref.refine(inp["ref_chunk"], c)
            cpu = time.process_time() - t0
            res = ref.last_result
            traj = np.asarray(res.trajectory, np.float64)
            exw = traj[:, :K]
            lin8 = run.meter(K)
            lin8.set_joint_parameter_path(run.finger_path(s, inp["ref_chunk"], K))
            lin8.set_attached(scene0.attached_points, scene0.attached_parent_link)
            cls = classify_violations(lin8, exw, q_now, scene, authorized_links=run.authorized,
                                      tolerance=float(cfg.safety.violation_tolerance),
                                      manipulated=s.get("manipulated") or {}, steps=K)
            m = res.metrics
            pc = m.get("path_check")
            la = m.get("lookahead")
            rec["arms"][a] = {
                "status": res.status.value, "n_collision_rows": int(cls["n_collision"]),
                "n_allowed_rows": int(cls["n_allowed"]), "hold_candidate": bool(cls["n_collision"] > 0),
                "window_violation_mm": float(res.max_violation) * 1e3,
                "iterations": m.get("sqp_iterations"), "cpu_s": cpu,
                "time_budget_hit": m.get("time_budget_hit"),
                "lookahead": la, "path_check": None if pc is None else {
                    k: pc.get(k) for k in ("clearance_m", "target_rows_excluded",
                                           "target_rows_unauthorized_kept", "n_configs")},
                "path_target_links": dict(captured.get("links", {})) if pc is not None else None,
            }
            arrays[f"s{seq}_{a}_own"] = np.asarray(own, np.float64)
            arrays[f"s{seq}_{a}_traj"] = traj
            results[a] = (res, scene, ref)
        # ---------- same-trajectory probes (working tree, strict arm available) ----------
        if cli.probes and has_i2 and rows is not None and rows.any():
            pr = {}
            # look-ahead: scene off vs strict; the V11AB refiner's linearizer
            if "offR" in results and "strict" in results:
                ref = results["offR"][2]
                lin = ref.optimizer.linearizer
                Hh = base.horizon.planned
                lin.set_joint_parameter_path(run.finger_path(s, inp["ref_chunk"], Hh))
                lin.set_attached(getattr(scene0, "attached_points", None),
                                 getattr(scene0, "attached_parent_link", None))
                sc_off = dataclasses.replace(scene0, target_contact_rows=rows, target_permission_strict=False)
                sc_on = dataclasses.replace(scene0, target_contact_rows=rows, target_permission_strict=True)
                for name, trj in (("reference", run.layout.chunk_to_trajectory(inp["ref_chunk"])[:, :Hh]),
                                  ("off_refined", results["offR"][0].trajectory)):
                    trj = np.asarray(trj, np.float64)
                    cen = lin.sphere_states(trj, q_now)[0]
                    c_off = np.array(lin._esdf_clearance(cen, sc_off)[..., 0])
                    la_off = dict(lin._last_lookahead)
                    c_on = np.array(lin._esdf_clearance(cen, sc_on)[..., 0])
                    la_on = dict(lin._last_lookahead)
                    tail = slice(int(lin.lookahead_from), None)
                    f_off, f_on = np.isfinite(c_off[tail]), np.isfinite(c_on[tail])
                    changed = f_on & ~f_off          # rows kept only by strict
                    cols = np.flatnonzero(changed.any(axis=0))
                    pr[f"lookahead_{name}"] = {
                        "n_tail_rows": int(f_off.size), "kept_off": int(f_off.sum()), "kept_strict": int(f_on.sum()),
                        "n_rows_changed": int(changed.sum()), "reverse_changed": int((f_off & ~f_on).sum()),
                        "changed_links": {lin._query_name(int(q)): int(changed[:, q].sum()) for q in cols},
                        "n_tail_target_unauthorized": la_on.get("n_tail_target_unauthorized"),
                        "n_tail_dropped_off": la_off.get("n_tail_dropped"),
                        "n_tail_dropped_strict": la_on.get("n_tail_dropped"),
                        "value_diff_max_mm_on_common": float(np.nanmax(np.abs(np.where(f_off & f_on, c_off[tail] - c_on[tail], 0))) * 1e3),
                    }
            # path: sw arm trajectory under R-off vs R-strict
            if "swR" in results and "swS" in results:
                path = results["swR"][2].optimizer.path
                trj = np.asarray(results["swR"][0].trajectory, np.float64)
                sc_off = dataclasses.replace(scene0, target_contact_rows=rows, target_permission_strict=False)
                sc_on = dataclasses.replace(scene0, target_contact_rows=rows, target_permission_strict=True)
                for tag, sc in (("off", sc_off), ("strict", sc_on)):
                    captured.clear()
                    path.lin.set_joint_parameter_path(run.finger_path(s, inp["ref_chunk"], trj.shape[1]))
                    w, r = path.violation(trj, sc, record=True)
                    tol = float(cfgs["swR"].safety.violation_tolerance)
                    pr[f"path_{tag}"] = {
                        "worst_m": float(w), "violation": bool(-w > tol),
                        "target_rows_excluded": r["target_rows_excluded"],
                        "target_rows_unauthorized_kept": r.get("target_rows_unauthorized_kept"),
                        "excluded_links": dict(captured.get("links", {})),
                        "worst_link": (r.get("worst") or {}).get("link"),
                    }
            rec["probes"] = pr
        out["chunks"].append(rec)
        inp["z"].close()
        prev_seq = seq
        print(cli.key, seq, flush=True)
    pathlib.Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(jsonable(out), open(cli.out, "w"))
    np.savez_compressed(str(cli.out).replace(".json", ".npz"), **arrays)


if __name__ == "__main__":
    main()
