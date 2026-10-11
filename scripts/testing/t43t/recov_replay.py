"""T43 T fix 1 — fixed-input TO replay of the held deadlock chunks, off vs `safety.recovery_floor`.
Scene = `qnow.scene_for` (record field + held spheres + manipulated + contact margin; reproduces the
recorded classification 31/31). Solve = `TrajOptChunkRefiner` with the server-log settings, the recorded
SQP iteration count, no wall clock. Verdict = `classify_violations` + the recovery progress check."""
import json, sys, pathlib, dataclasses
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import qnow as Q
from AG3S.trajopt.experiments import fixed_input_ablation as FIA
from AG3S.trajopt.safe_policy import classify_violations
from AG3S.trajopt import wire


class _Z(dict):
    files = property(lambda self: list(self.keys()))


class Run:
    def __init__(self, key):
        self.key = key; self.rd = Q.RUNS[key]
        self.ix, self.plan = Q.load(self.rd)
        self.G = Q.setup(str(pathlib.Path(self.ix["server"]) / "serve_safe.log"),
                         str(pathlib.Path(self.rd) / "manifest.json"))

    def capture_points(self):
        """Fix 2: the manipulated object's observed points on the last latched chunk before attach."""
        if not hasattr(self, "_cap"):
            seqs = sorted(int(k) for k in self.ix["chunks"])
            summ = {k: json.loads(str(np.load(self.ix["chunks"][str(k)]["npz"])["summary_json"])) for k in seqs}
            a = min(k for k in seqs if (summ[k].get("grasp") or {}).get("attach"))
            pre = max(k for k in seqs if k < a and summ[k]["grasp"]["state"] == "latched")
            with np.load(self.ix["chunks"][str(pre)]["npz"]) as zp:
                self._cap = (pre, np.asarray(zp["target_points"], float))
        return self._cap

    def inputs(self, seq, free_obs_pad=None):
        G = self.G
        with np.load(self.ix["chunks"][str(seq)]["npz"], allow_pickle=False) as zz:
            z = {k: zz[k] for k in zz.files}
        s = json.loads(str(z["summary_json"]))
        refc = np.asarray(z["reference_chunk"], float)
        rec_refined = np.asarray(z["refined_chunk"], float)
        qn = FIA.q_now_from_qpos(np.asarray(self.plan[seq]["qpos"], float), G["qadr"])
        zf = _Z(z)
        if free_obs_pad is not None:
            import body_emul as BE
            plane = s["grasp"]["attached"]["held"]["support"]["plane"]
            zf, self.last_freed = BE.emulated_npz(zf, [(p, free_obs_pad) for p in self.capture_points()[1]],
                                                  float(plane["offset_m"]) + 0.005)
        scene, field, held = Q.scene_for(G, zf, s, qn)
        return s, refc, rec_refined, qn, scene, held

    def solve(self, seq, *, recovery=False, progress_m=0.001, iters=None, overrides=None, free_obs_pad=None,
              scene_fn=None):
        from AG3S.trajopt.refiner import TrajOptChunkRefiner
        G = self.G
        s, refc, rec_refined, qn, scene, held = self.inputs(seq, free_obs_pad)
        if scene_fn is not None:
            scene = scene_fn(seq, s, qn)
        tm = (s.get("to") or {}).get("metrics") or {}
        n_it = iters or max(1, int(tm.get("sqp_iterations") or 1))
        cfg = FIA.to_config(G["st"], collision=True, cost=dict(G["st"].cost),
                            sqp={"time_budget_ms": FIA.NO_CLOCK_MS, "max_iterations": n_it})
        if overrides:
            cfg = cfg.with_overrides(overrides)
        holder = {"v": None}
        ref = TrajOptChunkRefiner(G["robot"], G["layout"], lambda _c: holder["v"], cfg)
        ref.reset()
        opt = ref.optimizer
        meas = (s.get("finger_joints") or {}).get("opening_norm") or {}
        path, _ = G["fmap"].path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, opt.linearizer.horizon,
                                 default=G["robot"].param_vector())
        opt.linearizer.set_joint_parameter_path(path)
        lin = opt.linearizer
        lin.set_attached(scene.attached_points, scene.attached_parent_link)
        auth = ((s.get("verdict") or {}).get("classification") or {}).get("authorized_links") or []
        man = s.get("manipulated") or {}
        rec = None
        if recovery:
            from AG3S.trajopt.safe_policy import recovery_floor_rows
            floor, rec = recovery_floor_rows(lin, qn, scene, authorized_links=auth, manipulated=man,
                                             tolerance=1e-4, pad_m=0.01)
            if floor is not None:
                scene = dataclasses.replace(scene, esdf_floor=floor)
        holder["v"] = (scene, qn, s.get("status") == "ok")
        ctx = {"t_step": seq}
        ref.refine(refc, ctx)
        r = ref.last_result
        traj = np.asarray(r.trajectory, float)
        cls = classify_violations(lin, traj, qn, scene, authorized_links=auth, tolerance=1e-4,
                                  manipulated=man, pad_m=0.01, keep_rows=True)
        bad_rows = [{k: (round(v * 1e3, 2) if k == "clearance_m" else v) for k, v in row.items()
                     if k in ("step", "query", "link", "clearance_m", "why", "tier", "class")}
                    for row in cls.get("rows", ()) if row["class"] != "allowed"]
        reasons = []
        if cls["n_allowed"]: reasons.append("allowed_contact")
        if cls["n_unresolved"]: reasons.append("unverified")
        if cls["n_collision"]: reasons.append("collision")
        prog = None
        if rec and rec.get("queries"):
            from AG3S.trajopt.safe_policy import recovery_progress
            prog = recovery_progress(lin, traj, qn, scene, rec, progress_m=progress_m, tolerance=1e-4)
            if not prog["ok"]:
                reasons.append("collision(recovery_no_progress)")
        hold = any(k.startswith(("collision", "unverified")) for k in reasons)
        raw = lin.esdf_clearance(traj, qn, scene)[..., 0]
        if getattr(scene, "esdf_floor", None) is not None:
            raw = raw + np.asarray(scene.esdf_floor)[None, :raw.shape[1]]
        return {"seq": seq, "status": r.status.value, "clearance_mm": 1e3 * float(r.metrics["clearance_m"]),
                "returned": r.metrics.get("returned"), "reasons": reasons, "hold": hold,
                "n_collision": cls["n_collision"], "collision_by_why": cls["collision_by_why"],
                "n_allowed": cls["n_allowed"], "n_unresolved": cls["n_unresolved"],
                "recovery": rec, "progress": prog, "bad_rows": bad_rows, "sel": {k: r.metrics.get(k) for k in ("candidates", "initial_merit", "initial_violation_m", "initial_source", "returned")},
                "raw_robot_min_mm": round(1e3 * float(raw[:, :lin.n_spheres].min()), 2),
                "raw_held_min_mm": None if raw.shape[1] == lin.n_spheres else round(1e3 * float(raw[:, lin.n_spheres:].min()), 2),
                "chunk": np.asarray(r.chunk, float), "trajectory": traj, "q_now": qn,
                "reference_traj": G["layout"].chunk_to_trajectory(refc)[:, :lin.horizon],
                "refined_equal_record": bool(np.array_equal(np.asarray(r.chunk, float).astype(np.float32), rec_refined)),
                "refined_maxdiff_record": float(np.abs(np.asarray(r.chunk, float) - rec_refined).max())}


def lift_mm(run, r):
    """Hold-link origin z at the last planned step minus at q_now (mm), planned vs the policy's."""
    G = run.G
    link = "ee_right" if "R" in run.key else "ee_left"
    z0 = G["robot"].link_pose(r["q_now"], link)[2, 3]
    zk = lambda tr: G["robot"].link_pose(G["layout"].full_q(tr[:, -1:], r["q_now"])[:, 0], link)[2, 3]  # noqa: E731
    return [round(1e3 * (zk(r["trajectory"]) - z0), 2), round(1e3 * (zk(r["reference_traj"]) - z0), 2)]


if __name__ == "__main__":
    key, s0, s1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    run = Run(key)
    out = {}
    for seq in range(s0, s1 + 1):
        for name, kw in (("off", {}), ("fix1", {"recovery": True, "overrides": {"safety": {"recovery_floor": True}}}),
                         ("fix2", {"free_obs_pad": 0.015}),
                         ("fix1+2", {"free_obs_pad": 0.015, "recovery": True, "overrides": {"safety": {"recovery_floor": True}}})):
            r = run.solve(seq, **kw)
            dz = None
            lift = lift_mm(run, r)
            r["hold_link_lift_mm"] = lift
            out.setdefault(seq, {})[name] = {k: v for k, v in r.items() if k not in ("chunk", "trajectory", "q_now", "reference_traj")}
            print(key, seq, name, "lift", lift, r["status"], round(r["clearance_mm"], 2), r["returned"], r["reasons"], "hold" if r["hold"] else "EXEC",
                  "prog", None if r["progress"] is None else (r["progress"]["worst_now_mm"], r["progress"]["worst_final_mm"], r["progress"]["ok"]),
                  "raw robot/held", r["raw_robot_min_mm"], r["raw_held_min_mm"], "rec==", r["refined_equal_record"], round(r["refined_maxdiff_record"], 6), flush=True)
    pathlib.Path(sys.argv[4]).write_text(json.dumps(out, indent=1, default=str))
