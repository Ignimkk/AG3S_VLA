"""V-T45 verifier driver — T44d C1a recorded chunks re-solved with palm cover off ("base") and on ("PC").

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 \
      nice -n 19 .venv-openpi-live/bin/python outputs/verify/T45/V/p/vt45.py --mode timing --field cuda --names ... --out DIR

modes
  timing : tp_timing's setting (server mode: one optimizer per variant kept across the run's chunks, variants
           alternate per chunk, recorded `sqp.time_budget_ms`, `--field cuda` = the server's DeviceEsdfField).
           Per solve: wall (server equivalent), cpu, stage times, ESDF rows, SQP/QP iterations.
  cmp    : T44 I2 replay convention (open loop, `ref.reset()` before every solve, `sqp.time_budget_ms = 1e6`).
           Per chunk and variant: refined chunk, status, verdict class (classify_violations over the executed
           window), worst row, palm sphere gaps (table / target / other obstacle), cross-evaluation.

The scene of every chunk carries what the C1a server handed the optimizer on top of ta_offline's: strict target
permission (`target_contact_rows`, held rows from AG3S' held permission record) and `layer_diff` membership.
Code is not modified; this file only calls benchmark functions.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse
import dataclasses
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from AG3S.trajopt.experiments import ta_offline as TA
from AG3S.trajopt.experiments import tp_timing as TP

K = TA.K_EXEC
VARS = ("base", "PC")
PC_ARGS = ("--palm-cover", "on")

COUNT: dict = {}
_DEPTH = [0]


def install_counters() -> None:
    """Row counters (no timing): core clearance rows (H x Q per computed call) and field points queried."""
    from AG3S.ag3s.fields.curobo_field import CuroboEsdfField
    from AG3S.trajopt.linearize import CollisionLinearizer

    core = CollisionLinearizer._esdf_clearance_core
    if not getattr(core, "__v_wrapped__", False):
        def core_w(self, centres, scene):
            if centres.ndim == 3:
                COUNT["core_calls"] = COUNT.get("core_calls", 0) + 1
                COUNT["core_rows"] = COUNT.get("core_rows", 0) + int(centres.shape[0] * centres.shape[1])
                COUNT["core_robot_rows"] = COUNT.get("core_robot_rows", 0) + int(
                    centres.shape[0] * self.n_spheres)
            return core(self, centres, scene)
        core_w.__v_wrapped__ = True
        CollisionLinearizer._esdf_clearance_core = core_w
    for name in ("distance", "gradient", "label", "target_free_distance", "target_free_gradient",
                 "answer_tier", "distance_and_label"):
        fn = getattr(CuroboEsdfField, name, None)
        if fn is None or getattr(fn, "__v_wrapped__", False):
            continue

        def mk(fn=fn, name=name):
            def w(self, points, *a, **k):
                if _DEPTH[0] == 0:
                    n = int(np.asarray(points).reshape(-1, 3).shape[0])
                    COUNT["field_points"] = COUNT.get("field_points", 0) + n
                    COUNT["field_calls"] = COUNT.get("field_calls", 0) + 1
                    COUNT[f"field_{name}"] = COUNT.get(f"field_{name}", 0) + n
                _DEPTH[0] += 1
                try:
                    return fn(self, points, *a, **k)
                finally:
                    _DEPTH[0] -= 1
            w.__v_wrapped__ = True
            return w
        setattr(CuroboEsdfField, name, mk())


def gpu_sample() -> dict:
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip().split(",")
        return {"mem_used_mib": int(o[0]), "util_pct": int(o[1])}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def load1() -> float:
    return float(open("/proc/loadavg").read().split()[0])


class Side:
    """One palm-cover setting: its Run (robot model, layout, reconstruction)."""

    def __init__(self, key, entry, ns, extra):
        self.run = TA.Run(key, entry, ns, extra_server_args=extra)
        from AG3S.ag3s.config import AG3SConfig
        from AG3S.ag3s.constraints import to_adapter as TOA

        contact_cfg = AG3SConfig.from_dict({"collision_backend": "esdf"}).contact
        self.TOA = TOA
        self.builder_ns = SimpleNamespace(sphere_link_names=self.run.names, robot_model=self.run.robot,
                                          clearance_policy=SimpleNamespace(contact=contact_cfg))

    def scene(self, inp):
        """ta_offline's scene + strict permission (what the C1a server handed the optimizer)."""
        from AG3S.trajopt import linearize as L

        scene0 = inp["scene"]
        manip = scene0.manipulated_points is not None or scene0.manipulated_spheres is not None
        rows = L.target_contact_rows_of(scene0.target_free_mask, None, scene0.manipulated_link_margin,
                                        "exclude_authorized", manipulated=manip)
        member = "layer_diff"
        if inp["held"] is not None:
            hrec = self.TOA._held_permission_record(
                self.builder_ns, SimpleNamespace(parent_link=str(inp["held"][0])))
            hr = L.held_contact_rows_of(hrec, 1, len(self.run.names))
            if hr is not None:
                rows = hr
            member = "ball"
        return dataclasses.replace(scene0, target_contact_rows=rows, target_permission_strict=True,
                                   target_membership=member)


def build_inputs(sides, seq, field_mode, prepare):
    """Per-variant (inp, scene) for chunk `seq`; `prepare` = tp_timing.prepare_field (timing mode)."""
    out = {}
    for v, sd in sides.items():
        inp, why = sd.run.inputs(seq)
        if inp is None:
            return None, why
        sc = inp["scene"]
        prew = 0.0
        if prepare:
            sc, prew = TP.prepare_field(sc, field_mode)
            inp = dict(inp)
            inp["scene"] = sc
        out[v] = (inp, sd.scene(inp), prew)
    return out, None


def chunk_context(run, inp, prev_seq, seq, cfg):
    s, fb = inp["summary"], inp["feedback"]
    arm = np.asarray(fb.get("applied_arm"), np.float64) if fb.get("applied_arm") is not None else None
    hist = np.ascontiguousarray(arm[-2:].T) if arm is not None and arm.ndim == 2 and arm.shape[0] >= 2 else None
    n_exec = fb.get("n_exec") if fb.get("available") else None
    prev_chunk = None
    dhist = None
    if prev_seq is not None and prev_seq == seq - 1:
        with np.load(run.chunks[prev_seq], allow_pickle=True) as pz:
            prev_chunk = np.asarray(pz["refined_chunk"], np.float64)
            prev_ref = np.asarray(pz["reference_chunk"], np.float64)
        ex = list(fb.get("executed") or [])
        if arm is not None and ex and ex[-1]:
            dhist = arm[-1] - run.layout.chunk_to_trajectory(prev_ref)[:, len(ex) - 1]
    ctx = {"t_step": seq, "previous_physical_chunk": prev_chunk}
    if n_exec is not None:
        ctx["previous_executed_steps"] = int(n_exec)
    return ctx, hist, dhist


def solve(ref, run, s, inp, scene, q_now, cfg, ctx0, hist, dhist, own_prev, mode, timed):
    lin = ref.optimizer.linearizer
    lin.set_joint_parameter_path(run.finger_path(s, inp["ref_chunk"], cfg.horizon.planned))
    holder = ref._v_holder
    holder["value"] = (scene, q_now, True)
    ctx = dict(ctx0)
    if hist is not None and cfg.limits.servo_accel != "off":
        ctx["command_history"] = hist
    if dhist is not None and cfg.cost.w_deflection_rate > 0:
        ctx["deflection_history"] = dhist
    if mode == "server" and own_prev is not None and ctx.get("previous_physical_chunk") is not None:
        ctx["previous_physical_chunk"] = own_prev
    if mode == "reset":
        ref.reset()
    TP.CLOCK.reset()
    COUNT.clear()
    TP.CLOCK.enabled = timed
    c0 = time.process_time()
    w0 = time.perf_counter()
    own = ref.refine(inp["ref_chunk"], ctx)
    wall = (time.perf_counter() - w0) * 1e3
    cpu = (time.process_time() - c0) * 1e3
    TP.CLOCK.enabled = False
    res = ref.last_result
    stages = {k: v * 1e3 for k, v in TP.CLOCK.acc.items()}
    harness = stages.pop("harness", 0.0)
    wall_adj = wall - harness
    if timed:
        stages["other"] = max(0.0, wall_adj - sum(stages.values()))
    m = res.metrics
    traj = np.ascontiguousarray(np.asarray(res.trajectory, np.float64))
    la = m.get("lookahead") or {}
    rec = {"wall_ms": wall_adj, "wall_raw_ms": wall, "harness_ms": harness, "cpu_ms": cpu - harness,
           "stages": {k: round(v, 4) for k, v in stages.items()}, "calls": dict(TP.CLOCK.calls),
           "counts": dict(COUNT), "status": res.status.value,
           "sqp_iterations": m.get("sqp_iterations"), "qp_iterations": m.get("qp_iterations"),
           "time_budget_hit": m.get("time_budget_hit"), "max_iterations_hit": m.get("max_iterations_hit"),
           "max_violation_mm": float(res.max_violation) * 1e3, "solve_ms": float(res.solve_time_ms),
           "gate": la.get("gate"), "sha1": hashlib.sha1(traj.tobytes()).hexdigest(),
           "worst_link_solver": ((m.get("max_violation_pair") or {}) if isinstance(m.get("max_violation_pair"), dict)
                                 else TA.lit(m.get("max_violation_pair")) or {}).get("link"),
           "worst_tier_solver": ((m.get("max_violation_pair") or {}) if isinstance(m.get("max_violation_pair"), dict)
                                 else TA.lit(m.get("max_violation_pair")) or {}).get("tier")}
    return own, traj, rec


def make_refiners(sides, cfgs):
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    out = {}
    for v, sd in sides.items():
        holder: dict = {}
        r = TrajOptChunkRefiner(sd.run.robot, sd.run.layout, lambda _c, h=holder: h["value"], cfgs[v])
        r._v_holder = holder
        out[v] = r
    return out


def load_entry(index, name):
    return json.load(open(index))["runs"][name]


def close_info(run):
    """First `closing` chunk (else first `attach`), from the recorded summaries."""
    state = {}
    for seq in sorted(run.chunks):
        with np.load(run.chunks[seq], allow_pickle=True) as z:
            s = json.loads(str(z["summary_json"]))
        g = s.get("grasp") or {}
        state[seq] = (g.get("state"), bool(g.get("attach")), bool(g.get("attached_in_constraints")))
    close = next((q for q in sorted(state) if state[q][0] == "closing"), None)
    attach = next((q for q in sorted(state) if state[q][1]), None)
    return {"close_seq": close, "attach_seq": attach, "states": {str(k): v[0] for k, v in state.items()}}


def run_timing(name, entry, cli, w):
    ns = argparse.Namespace(hm=False, no_plane_rows=False)
    TP.install()
    install_counters()
    sides = {"base": Side(name, entry, ns, ()), "PC": Side(name, entry, ns, PC_ARGS)}
    base_cfg = sides["base"].run.base_cfg
    cfgs = {}
    for v in VARS:
        over = {sec: dict(kv) for sec, kv in {}.items()}
        over.setdefault("collision", {})["held_obstacle_margin"] = False
        cfgs[v] = sides[v].run.base_cfg.with_overrides(over)
    refs = make_refiners(sides, cfgs)
    seqs = sorted(sides["base"].run.chunks)
    if cli.max_chunks:
        seqs = seqs[cli.first_chunk:cli.first_chunk + cli.max_chunks]
    out = {"name": name, "mode": "timing", "field": cli.field, "time_budget_ms": cfgs["base"].sqp.time_budget_ms,
           "n_spheres": {v: int(sides[v].run.robot.n_spheres) for v in VARS},
           "server_args_extra": {"PC": list(PC_ARGS)}, "load_before": TP._cgroup_cpu(), "gpu_before": gpu_sample(),
           "close": close_info(sides["base"].run), "chunks": []}
    prev_seq = None
    own_prev = {v: None for v in VARS}
    for ci, seq in enumerate(seqs):
        inps, why = build_inputs(sides, seq, cli.field, True)
        if inps is None:
            out["chunks"].append({"seq": seq, "skipped": why})
            prev_seq = None
            own_prev = {v: None for v in VARS}
            for r in refs.values():
                r.reset()
            continue
        inp0 = inps["base"][0]
        s, q_now = inp0["summary"], inp0["q_now"]
        ctx0, hist, dhist = chunk_context(sides["base"].run, inp0, prev_seq, seq, cfgs["base"])
        g = s.get("grasp") or {}
        rec = {"seq": seq, "grasp_state": g.get("state"), "held": inp0["held"] is not None,
               "recorded_to_ms": (s.get("to") or {}).get("solve_ms"),
               "recorded_status": (s.get("to") or {}).get("status"), "load1": load1(), "variants": {}}
        order = list(VARS[ci % 2:]) + list(VARS[:ci % 2])
        rec["order"] = order
        for v in order:
            inp, scene, _ = inps[v]
            own, traj, r = solve(refs[v], sides[v].run, s, inp, scene, q_now, cfgs[v], ctx0, hist, dhist,
                                 own_prev[v], "server", True)
            own_prev[v] = np.asarray(own, np.float64)
            rec["variants"][v] = r
        for v in VARS:
            inps[v][0]["z"].close()
        out["chunks"].append(rec)
        prev_seq = seq
        if cli.progress and ci % 10 == 0:
            print(f"{name} seq {seq} " + " ".join(f"{v}={rec['variants'][v]['wall_ms']:.0f}" for v in VARS),
                  flush=True)
    out["load_after"] = TP._cgroup_cpu()
    out["gpu_after"] = gpu_sample()
    return out


# ----------------------------------------------------------------------------------------------------------
# cmp mode
# ----------------------------------------------------------------------------------------------------------
GROUPS_OF = None


def group_indices(run):
    names = np.asarray(run.names)
    hand = run.hand
    fl = "l" if hand == "left" else "r"
    idx = {"palm": np.flatnonzero(names == f"ee_{hand}"),
           "fingers": np.flatnonzero(np.isin(names, [f"ee_finger_{fl}1", f"ee_finger_{fl}2"]))}
    return idx


def eval_traj(sd, inp, scene, traj, q_now, cfg, s, z_geom):
    """Verdict class + clearances of `traj` under side `sd`'s model and scene (executed window K)."""
    from AG3S.trajopt.safe_policy import classify_violations

    run = sd.run
    lin = run.meter(K)
    lin.set_joint_parameter_path(run.finger_path(s, inp["ref_chunk"], K))
    lin.set_attached(getattr(inp["scene"], "attached_points", None), getattr(inp["scene"], "attached_parent_link", None))
    exw = traj[:, :K]
    cls = classify_violations(lin, exw, q_now, scene, authorized_links=run.authorized,
                              tolerance=float(cfg.safety.violation_tolerance),
                              manipulated=s.get("manipulated") or {}, steps=K)
    centres = lin.sphere_states(exw, q_now)[0]                       # (K, Q, 3)
    to = np.asarray(lin._esdf_clearance(centres, scene)[..., 0], np.float64)   # (K, Q) d - r - margin
    S = len(run.names)
    idx = group_indices(run)
    names = run.names
    flat = np.argmin(to)
    k_w, q_w = np.unravel_index(flat, to.shape)
    worst_link = names[q_w] if q_w < S else "held"
    grp = {}
    for g, ii in idx.items():
        grp[g] = float(to[:, ii].min() * 1e3) if len(ii) else None
    other = np.setdiff1d(np.arange(S), np.concatenate([idx["palm"], idx["fingers"]]))
    grp["other"] = float(to[:, other].min() * 1e3) if len(other) else None
    grp["held"] = float(to[:, S:].min() * 1e3) if to.shape[1] > S else None
    out = {"n_collision": int(cls["n_collision"]), "n_allowed": int(cls["n_allowed"]),
           "n_unresolved": int(cls["n_unresolved"]), "n_violating": int(cls["n_violating"]),
           "collision_worst_link": (cls.get("collision_worst") or {}).get("link"),
           "to_min_mm": float(to.min() * 1e3), "worst_link_to_view": worst_link, "worst_step": int(k_w),
           "worst_is_palm": bool(worst_link == f"ee_{run.hand}"),
           "group_min_to_mm": grp}
    # palm sphere geometry
    pi = idx["palm"]
    if len(pi) and z_geom is not None:
        pc = centres[:, pi]                                          # (K, P, 3)
        pr = np.asarray(lin.robot_radii, np.float64)[pi]             # (P,)
        flatc = pc.reshape(-1, 3)
        rr = np.tile(pr[None, :], (K, 1)).reshape(-1)
        d_main = np.asarray(scene.esdf.distance(flatc), np.float64) - rr
        d_free = np.asarray(scene.esdf.target_free_distance(flatc), np.float64) - rr if getattr(
            scene.esdf, "target_free_layers", ()) else None
        tgt = z_geom["target_tree"]
        gt = (tgt.query(flatc)[0] - rr) if tgt is not None else None
        ot = z_geom["other_tree"]
        go = (ot.query(flatc)[0] - rr) if ot is not None else None
        pl = inp["plane"]
        gtab = (flatc @ pl[:3] - pl[3] - rr) if pl is not None else None
        mlm = scene.manipulated_link_margin
        marg = None if mlm is None else float(np.asarray(mlm)[pi].mean())

        def st(a, thr=()):
            if a is None:
                return None
            o = {"min_mm": float(a.min() * 1e3)}
            for t in thr:
                o[f"n_lt_{t:g}mm"] = int((a < t * 1e-3).sum())
            return o
        mask = scene.target_free_mask
        out["palm"] = {
            "n_spheres": int(len(pi)), "n_queries": int(flatc.shape[0]),
            "radius_mm_max": float(pr.max() * 1e3), "radius_mm_min": float(pr.min() * 1e3),
            "target_free_mask_palm": None if mask is None else int(np.asarray(mask)[pi].sum()),
            "target_contact_rows_palm": (None if scene.target_contact_rows is None
                                         else int(np.asarray(scene.target_contact_rows)[pi].sum())),
            "manipulated_link_margin_palm_mm": None if marg is None else marg * 1e3,
            "gap_target": st(gt, (0.0, 5.0, 10.0, 20.0)),
            "gap_table": st(gtab, (0.0, 5.0, 10.0)),
            "gap_other_obstacle": st(go, (0.0, 5.0, 10.0)),
            "gap_field_main": st(d_main, (0.0, 5.0, 10.0)),
            "gap_field_target_free": st(d_free, (0.0, 5.0, 10.0)),
            "n_queries_free_minus_main_gt_2mm": None if d_free is None else int(((d_free - d_main) > 2e-3).sum()),
            "n_queries_target_gap_lt_20mm_and_free_minus_main_gt_2mm": None if (d_free is None or gt is None) else int(
                (((d_free - d_main) > 2e-3) & (gt < 20e-3)).sum()),
            "centre_xyz_min_target_gap": None if gt is None else flatc[int(np.argmin(gt))].tolist()}
    return out, exw


def geom_of(inp):
    from scipy.spatial import cKDTree

    z = inp["z"]
    tp = np.asarray(z["target_points"], np.float64) if "target_points" in z.files else np.zeros((0, 3))
    tp = tp if len(tp) else None
    other = None
    if "filtered_cloud" in z.files:
        pts = np.asarray(z["filtered_cloud"], np.float64)
        keep = np.ones(len(pts), bool)
        if "support_mask" in z.files and np.asarray(z["support_mask"]).shape[0] == len(pts):
            keep &= ~np.asarray(z["support_mask"], bool)
        if "target_point_indices" in z.files:
            ti = np.asarray(z["target_point_indices"], np.int64).reshape(-1)
            ti = ti[(ti >= 0) & (ti < len(pts))]
            keep[ti] = False
        o = pts[keep]
        other = o if len(o) else None
    return {"target_tree": None if tp is None else cKDTree(tp),
            "other_tree": None if other is None else cKDTree(other), "n_target": 0 if tp is None else len(tp),
            "n_other": 0 if other is None else len(other)}


def run_cmp(name, entry, cli, w):
    ns = argparse.Namespace(hm=False, no_plane_rows=False)
    sides = {"base": Side(name, entry, ns, ()), "PC": Side(name, entry, ns, PC_ARGS)}
    budget = {"sqp": {"time_budget_ms": 1.0e6}, "collision": {"held_obstacle_margin": False}}
    cfgs = {v: sides[v].run.base_cfg.with_overrides(budget) for v in VARS}
    refs = make_refiners(sides, cfgs)
    seqs = sorted(sides["base"].run.chunks)
    if cli.max_chunks:
        seqs = seqs[cli.first_chunk:cli.first_chunk + cli.max_chunks]
    out = {"name": name, "mode": "cmp", "time_budget_ms": 1.0e6,
           "n_spheres": {v: int(sides[v].run.robot.n_spheres) for v in VARS},
           "close": close_info(sides["base"].run), "authorized": sides["base"].run.authorized,
           "hand": sides["base"].run.hand, "chunks": []}
    prev_seq = None
    arrays = {}
    for seq in seqs:
        inps, why = build_inputs(sides, seq, "cpu", False)
        if inps is None:
            out["chunks"].append({"seq": seq, "skipped": why})
            prev_seq = None
            continue
        inp0 = inps["base"][0]
        s, q_now = inp0["summary"], inp0["q_now"]
        ctx0, hist, dhist = chunk_context(sides["base"].run, inp0, prev_seq, seq, cfgs["base"])
        g = s.get("grasp") or {}
        vd = (s.get("verdict") or {}).get("classification") or {}
        rec = {"seq": seq, "t0": int(inp0["row"]["t_step"]), "grasp_state": g.get("state"),
               "held": inp0["held"] is not None, "has_target": bool(s.get("has_target")),
               "recorded": {"status": (s.get("to") or {}).get("status"),
                            "n_collision": vd.get("n_collision"), "n_violating": vd.get("n_violating"),
                            "gate_action": (s.get("verdict") or {}).get("action") or None,
                            "reasons": [r.get("kind") + ":" + str(r.get("action"))
                                        for r in ((s.get("verdict") or {}).get("reasons") or [])]},
               "variants": {}}
        geom = geom_of(inp0)
        rec["geom"] = {"n_target_points": geom["n_target"], "n_other_points": geom["n_other"]}
        rec_ref = np.asarray(inp0["rec_chunk"], np.float64)
        res = {}
        for v in VARS:
            inp, scene, _ = inps[v]
            own, traj, r = solve(refs[v], sides[v].run, s, inp, scene, q_now, cfgs[v], ctx0, hist, dhist,
                                 None, "reset", False)
            res[v] = (own, traj)
            r["own_vs_recorded_max_abs_rad"] = float(np.abs(np.asarray(own, np.float64) - rec_ref).max())
            rec["variants"][v] = r
            arrays[f"s{seq}_{v}_own"] = np.asarray(own, np.float64)
        # evaluate each trajectory under each model (own = diagonal, cross = off diagonal)
        for v in VARS:
            for u in VARS:
                inp, scene, _ = inps[u]
                ev, _ = eval_traj(sides[u], inp, scene, res[v][1], q_now, cfgs[u], s, geom)
                rec["variants"][v].setdefault("eval_under", {})[u] = ev
        rec["diff_own_max_abs_rad"] = float(np.abs(res["PC"][0] - res["base"][0]).max())
        rec["diff_own_max_abs_deg"] = float(np.degrees(rec["diff_own_max_abs_rad"]))
        # per joint on the executed window (8 control steps x 14 joints) in the trajectory form
        dj = np.abs(res["PC"][1][:, :K] - res["base"][1][:, :K])
        rec["diff_window_max_abs_rad"] = float(dj.max())
        rec["diff_window_argmax_joint"] = int(np.unravel_index(np.argmax(dj), dj.shape)[0])
        rec["same_sha1"] = rec["variants"]["base"]["sha1"] == rec["variants"]["PC"]["sha1"]
        for u in VARS:
            inps[u][0]["z"].close()
        out["chunks"].append(rec)
        prev_seq = seq
        if cli.progress:
            print(f"{name} seq {seq} dmax={rec['diff_window_max_abs_rad']:.4f} "
                  f"nc={rec['variants']['base']['eval_under']['base']['n_collision']}/"
                  f"{rec['variants']['PC']['eval_under']['PC']['n_collision']}", flush=True)
    np.savez_compressed(pathlib.Path(cli.out) / f"{name}.cmp.npz", **arrays)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("timing", "cmp"), required=True)
    ap.add_argument("--field", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--index", default=_ag3s_workspace_path('outputs/verify/T45/V/index_c1a.json'))
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--first-chunk", type=int, default=0)
    ap.add_argument("--progress", action="store_true")
    cli = ap.parse_args()
    out_dir = pathlib.Path(cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in cli.names:
        entry = load_entry(cli.index, name)
        t0 = time.time()
        res = (run_timing if cli.mode == "timing" else run_cmp)(name, entry, cli, None)
        res["wall_s"] = round(time.time() - t0, 1)
        tmp = out_dir / f"{name}.{cli.mode}.tmp"
        tmp.write_text(json.dumps(TA._jsonable(res)))
        tmp.replace(out_dir / f"{name}.{cli.mode}.json")
        print(f"{name}: {len(res['chunks'])} chunks {res['wall_s']} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
