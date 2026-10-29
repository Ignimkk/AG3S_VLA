"""T43 TA2 figure 1 (scene): a hurdle crossing re-solved offline, side view (x–z) at the bar.

The recorded field's zero level (ESDF slice at the held apple's mean y) and, per variant, the held apple
centre over the executed rows 0-7 of consecutive chunks: commanded (solid) and servo-predicted (dashed,
evaluator only). Open loop: every chunk starts from the recorded state.
usage: fig_scene_ta.py KEY SEQ0 SEQ1 OUT_PNG VARIANT...
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, types, pathlib
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from AG3S.ag3s.experiments.common import figstyle
from AG3S.trajopt.experiments.ta_offline import Run, variant_config, K_EXEC
from AG3S.trajopt.refiner import TrajOptChunkRefiner

key, s0, s1, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
variants = sys.argv[5:]
idx = json.load(open(_ag3s_workspace_path('outputs/verify/T43/V8/analysis/analysis.json')))["runs"]
run = Run(key, idx[key], types.SimpleNamespace(hm=False, no_plane_rows=False))
paths = {v: {"cmd": [], "srv": []} for v in variants}
fields = []
for seq in range(s0, s1 + 1):
    inp, why = run.inputs(seq)
    if inp is None or inp["held"] is None:
        continue
    s, fb, q_now = inp["summary"], inp["feedback"], inp["q_now"]
    st = run.observer.update(q_now[run.layout.q_indices], fb if fb.get("available") else None, seq)
    ctx = {"t_step": seq}
    if seq - 1 in run.chunks:
        with np.load(run.chunks[seq - 1], allow_pickle=True) as pz:
            ctx["previous_physical_chunk"] = np.asarray(pz["refined_chunk"], np.float64)
    if fb.get("available") and fb.get("n_exec") is not None:
        ctx["previous_executed_steps"] = int(fb["n_exec"])
    arm = np.asarray(fb["applied_arm"], np.float64) if fb.get("applied_arm") is not None else None
    link, c_loc = inp["held"][0], inp["held"][1]
    fields.append(inp["scene"].esdf)
    for v in variants:
        cfg = variant_config(run.base_cfg, v, False)
        holder = {"v": (inp["scene"], q_now, True)}
        ref = TrajOptChunkRefiner(run.robot, run.layout, lambda c: holder["v"], cfg)
        H = cfg.horizon.planned
        ref.optimizer.linearizer.set_joint_parameter_path(run.finger_path(s, inp["ref_chunk"], H))
        c = dict(ctx)
        if arm is not None and arm.shape[0] >= 2 and cfg.limits.servo_accel != "off":
            c["command_history"] = np.ascontiguousarray(arm[-2:].T)
        ref.refine(inp["ref_chunk"], c)
        ex = np.asarray(ref.last_result.trajectory)[:, :K_EXEC]
        ro = run.servo.rollout(st.q, st.v, ex, st.tau, samples=(33,))
        qs = ro["q_samples"].reshape(ex.shape[0], -1)
        for key_, Q in (("cmd", ex), ("srv", qs)):
            full = run.layout.full_q(Q, q_now)
            for k in range(Q.shape[1]):
                T = np.asarray(run.robot.link_pose(full[:, k], link))
                paths[v][key_].append((c_loc @ T[:3, :3].T + T[:3, 3]).mean(axis=0))
    inp["z"].close()
P = {v: {k: np.asarray(x) for k, x in d.items()} for v, d in paths.items()}
allp = np.concatenate([P[v]["cmd"] for v in variants])
# vertical plane along the path's main horizontal direction
c0 = allp.mean(axis=0)
w, V = np.linalg.eigh(np.cov((allp[:, :2] - c0[:2]).T))
u = np.r_[V[:, -1], 0.0]
proj = lambda p: np.stack([(p - c0) @ u, p[:, 2]], axis=1)
S = proj(allp)
xs = np.linspace(S[:, 0].min() - 0.08, S[:, 0].max() + 0.08, 240)
zs = np.linspace(S[:, 1].min() - 0.12, S[:, 1].max() + 0.06, 180)
X, Z = np.meshgrid(xs, zs)
pts = c0[None, :] + X.reshape(-1, 1) * u[None, :]
pts[:, 2] = Z.ravel()
D = np.asarray(fields[len(fields) // 2].distance(pts)).reshape(X.shape)
n_rows = {"cmd": K_EXEC, "srv": K_EXEC}
ko = figstyle.use_korean()
fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.2))
figstyle.style_axes(fig, [ax])
ax.contourf(X, Z, D, levels=[-1, 0.0], colors=[figstyle.INK_2], alpha=0.35)
ax.contour(X, Z, D, levels=[0.0, 0.035], colors=[figstyle.INK_2, figstyle.GRID_INK], linewidths=[1.2, 0.8])
for i, v in enumerate(variants):
    col = figstyle.CATEGORICAL[i % len(figstyle.CATEGORICAL)]
    for kind, ls, lw in (("cmd", "-", 1.6), ("srv", "--", 1.0)):
        q = proj(P[v][kind])
        for j in range(0, len(q), K_EXEC):          # one segment per chunk (open loop: no joins)
            seg = q[j:j + K_EXEC]
            ax.plot(seg[:, 0], seg[:, 1], ls, color=col, lw=lw,
                    label=(f"{v} {'command' if kind == 'cmd' else 'servo-predicted'}" if j == 0 else None))
            if kind == "cmd":
                ax.plot(seg[:1, 0], seg[:1, 1], "o", color=col, ms=3)
ax.set_xlabel("along the path (m)" if not ko else "경로 방향 (m)"); ax.set_ylabel("z (m)"); ax.set_aspect("equal")
y0 = float(c0[1])
ttl = (f"{key} chunk {s0}-{s1}: 쥔 사과 중심 (청크마다 rows 0-7, 점 = 청크 시작; 회색 = ESDF ≤ 0, 연선 = 35 mm)" if ko else
       f"{key} chunks {s0}-{s1}: held apple centre, rows 0-7 per chunk (dot = chunk start; grey = ESDF <= 0)")
ax.set_title(ttl, fontsize=9)
ax.legend(fontsize=7, ncol=2, loc="upper right")
fig.tight_layout(); fig.savefig(out, dpi=150)
json.dump({"key": key, "seqs": [s0, s1], "plane_point_m": c0.round(4).tolist(), "plane_dir": u.round(4).tolist(), "variants": variants,
           "paths": {v: {k: x.round(5).tolist() for k, x in d.items()} for v, d in P.items()}},
          open(pathlib.Path(out).with_suffix(".json"), "w"))
print("wrote", out)
