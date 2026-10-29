"""T33 offline (e) — G1: reference-chunk limit overshoot under URDF vs MJCF limits on recorded E3 runs (T28 + T30 E3a/E3b).

Per planning row (local frames.jsonl): Q = layout.chunk_to_trajectory(actions_reference)[:, :8] (the TO plan window, P=K=8),
q_now = planning-row qpos. Limits = trajopt.limits.build_limits(robot, layout, dt, LimitsConfig(source=...)) with position margin
0.02 rad (default), velocity/acceleration from URDF in both cases. Overshoot = trajopt.limits.limit_report (position / velocity /
acceleration, rad) + anchor (|Q[:,0]-q_now| - max_step, as sqp._limit_overshoot). A row 'overshoots' when any entry > 1e-9.
Per joint: which optimized joint carries the position overshoot (max over steps)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, collections
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T31diag/d2'))
import d2_rows_ablation as D2
import AG3S.trajopt.experiments.fixed_input_ablation as T
from AG3S.trajopt.limits import build_limits, limit_report
from AG3S.trajopt.config import LimitsConfig

XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T33/off/e'))
V = pathlib.Path(_ag3s_workspace_path('outputs/verify'))


def main():
    sh = D2.setup("E3b", "8206", V / "T30/E3b_rec/E3b_ep1807_r1")
    robot, layout, qadr = sh["robot"], sh["layout"], sh["qadr"]
    dt = T.to_config(sh["settings"], collision=True, cost=dict(sh["settings"].cost), sqp={}).horizon.dt
    L = {"urdf": build_limits(robot, layout, dt=dt, config=LimitsConfig(source="urdf")),
         "model_xml": build_limits(robot, layout, dt=dt, config=LimitsConfig(source="model_xml", model_xml=XML))}
    names = [robot.joint_names[i] for i in layout.q_indices] if hasattr(layout, "q_indices") else None
    res = {"definitions": __doc__, "dt": dt, "joint_order": names,
           "box": {k: {"lower": l.lower.tolist(), "upper": l.upper.tolist()} for k, l in L.items()}, "runs": {}}
    for tag in ("T28", "T30", "T33"):
        for cond in ("E3a", "E3b"):
            for rd in sorted((V / tag / f"{cond}_rec").glob(f"{cond}_ep*")):
                if not rd.is_dir() or not (rd / "frames.jsonl").exists():
                    continue
                rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
                plan = [r for r in rows if r["kind"] == "planning" and r.get("actions_reference") is not None]
                e = {"tag": tag, "cond": cond, "n_rows": len(plan)}
                for src, lim in L.items():
                    cnt = collections.Counter(); jc = collections.Counter(); mx = collections.defaultdict(float); ts = []
                    for p in plan:
                        Q = layout.chunk_to_trajectory(np.asarray(p["actions_reference"], np.float64))[:, :8]
                        q0 = T.q_now_from_qpos(p["qpos"], qadr)
                        rep = dict(limit_report(Q, lim))
                        step = np.asarray(lim.max_step, float)
                        rep["anchor"] = max(float(np.max(np.abs(Q[:, 0] - np.asarray(q0).reshape(-1)[layout.q_indices]) - step, initial=0.0)), 0.0)
                        any_ = False
                        for k, v in rep.items():
                            if v > 1e-9:
                                cnt[k] += 1; mx[k] = max(mx[k], v); any_ = True
                        if rep["position"] > 1e-9:
                            per = np.maximum(lim.lower[:, None] - Q, Q - lim.upper[:, None]).max(axis=1)
                            j = int(per.argmax()); jc[names[j] if names else j] += 1
                        if any_:
                            cnt["any"] += 1; ts.append(p["t_step"])
                    e[src] = {"n_rows_overshoot": dict(cnt), "max_overshoot_rad": dict(mx), "max_overshoot_deg": {k: float(np.degrees(v)) for k, v in mx.items()},
                              "position_overshoot_joint_counts": dict(jc), "t_any": ts}
                res["runs"][f"{tag}/{rd.name}"] = e
                print(tag, rd.name, len(plan), "urdf", dict(e["urdf"]["n_rows_overshoot"]), "mjcf", dict(e["model_xml"]["n_rows_overshoot"]), e["model_xml"]["position_overshoot_joint_counts"], flush=True)
    (OUT / "offline_e.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
