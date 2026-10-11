"""T35-diag figure 1 — per failed T34 run: grasp timeline (graph).
Panels (shared t_step axis, control rows t_close-40 .. t_close+70):
  1 fingertip-apple GT distance (min over left finger geoms, mm): measured / reference-commanded arm pose / refined-commanded
  2 left gripper: applied command and measured opening (1 = open)
  3 GT apple dz from start (mm)
  4 refined - reference per chunk (deg, max over arm joints, rows 0-7; planning rows, drawn over the chunk's 8 control rows)
  5 grasp latch state per chunk (server summary grasp.state) with attach / attach_revoked marks
Sources: outputs/verify/T35diag/a/a_runs.json, T34 p/<COND>_analysis.json. Writes figures/t35/t35-timeline-<run>.png + .json.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as F

FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t35'))
A = json.load(open(_ag3s_workspace_path('outputs/verify/T35diag/a/a_runs.json')))["runs"]
ST = json.load(open(_ag3s_workspace_path('outputs/verify/T35diag/a/settle.json')))["runs"]
RUNS = ["E3b_ep1800_r1", "E3b_ep1808_r1", "E3b_ep1808_r2", "E3a_ep1800_r1"]
LATCH = ["latched", "closing", "held", "searching"]
C = F.CATEGORICAL

for run in RUNS:
    cond = run.split("_")[0]
    r = A[f"{cond}_rec/{run}"]; st = ST[f"{cond}_rec/{run}"]
    an = json.load(open(f"${AG3S_PROJECT_ROOT}/outputs/verify/T34/p/{cond}_analysis.json"))["runs"][run]["chunks"]
    tc = r["t_close"]; t0, t1 = tc - 40, tc + 70
    tl = [e for e in r["timeline"] if t0 <= e["t"] <= t1]
    t = np.array([e["t"] for e in tl])
    fig, ax = plt.subplots(5, 1, figsize=(8.6, 9.4), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2, 2, 0.8]})
    F.style_axes(fig, ax)
    clip = lambda v: np.clip(np.array(v, float), -10, 120)
    ax[0].plot(t, clip([e["d_exec_mm"] for e in tl]), color=C[0], lw=2, label="measured (executed)")
    ax[0].plot(t, clip([e["d_ref_mm"] for e in tl]), color=C[1], lw=2, label="reference cmd (policy)")
    ax[0].plot(t, clip([e["d_refined_mm"] for e in tl]), color=C[2], lw=2, ls="--", label="refined cmd (TO)")
    ax[0].axhline(0, color=F.GRID_INK, lw=1)
    ax[0].set_ylabel("finger-apple GT\ndistance (mm, clip 120)")
    ax[0].legend(fontsize=7, frameon=False, loc="upper left")
    ax[1].plot(t, [e["grip_cmd"] for e in tl], color=C[0], lw=2, label="applied cmd")
    ax[1].plot(t, [e["open_meas"] for e in tl], color=C[1], lw=2, label="measured opening")
    ax[1].axhline(0.70, color=F.GRID_INK, lw=1, ls=":")
    ax[1].text(t0 + 1, 0.72, "apple width ~0.70", fontsize=7, color=F.INK_2)
    ax[1].set_ylabel("left gripper\n(1 = open)"); ax[1].set_ylim(-0.05, 1.08)
    ax[1].legend(fontsize=7, frameon=False, loc="lower left")
    ax[2].plot(t, [e["dz_mm"] for e in tl], color=C[0], lw=2)
    ax[2].set_ylabel("GT apple dz (mm)")
    rr = [x for x in r["refined_minus_reference"] if t0 - 8 <= x["t"] <= t1]
    ax[3].bar([x["t"] + 3.5 for x in rr], [x["max_arm_deg"] for x in rr], width=7, color=C[0], edgecolor=F.SURFACE, lw=1)
    ax[3].set_ylabel("refined - reference\n(deg, arm max)")
    ch = {c["t"]: c for c in an}
    for tt, c in ch.items():
        if t0 - 8 <= tt <= t1:
            s = (c.get("grasp_latch") or {}).get("state")
            k = LATCH.index(s) if s in LATCH else 3
            ax[4].barh(0, 8, left=tt, color=C[k], edgecolor=F.SURFACE, lw=1)
            if (c.get("grasp_latch") or {}).get("attach"):
                ax[4].text(tt + 4, 0.55, "attach", ha="center", fontsize=7, color=F.INK)
            if (c.get("grasp_latch") or {}).get("detach"):
                ax[4].text(tt + 4, 0.55, "revoke", ha="center", fontsize=7, color=F.INK)
    for k, s in enumerate(LATCH[:3]):
        ax[4].barh(0, 0, color=C[k], label=s)
    ax[4].legend(fontsize=7, frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, -1.6))
    ax[4].set_yticks([]); ax[4].set_ylim(-0.5, 1.0); ax[4].set_ylabel("latch")
    for a in ax:
        a.axvline(tc, color=F.INK_2, lw=1, ls="--")
        a.axvline(st["t_settle"], color=F.INK_2, lw=1, ls=":")
        if st["contact_lost_t"]:
            a.axvline(st["contact_lost_t"], color=C[7], lw=1, ls="-.")
    ax[0].text(tc, ax[0].get_ylim()[1], " first close", fontsize=7, va="top", color=F.INK_2)
    ax[0].text(st["t_settle"], ax[0].get_ylim()[1] * 0.85, " settle", fontsize=7, va="top", color=F.INK_2)
    if st["contact_lost_t"]:
        ax[0].text(st["contact_lost_t"], ax[0].get_ylim()[1] * 0.7, " contact lost", fontsize=7, va="top", color=C[7])
    ax[4].set_xlabel("t_step (control row)")
    ax[4].set_xlim(t0, t1)
    fig.suptitle(f"T35-diag {run} (T34, failed): grasp timeline", fontsize=11, color=F.INK)
    fig.tight_layout()
    out = FIG / f"t35-timeline-{run}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    side = {"figure": str(out), "script": "outputs/verify/T35diag/figs/fig_timeline.py", "doc": __doc__, "run": run, "t_close": tc,
            "t_settle": st["t_settle"], "contact_lost_t": st["contact_lost_t"],
            "rows": [{k: e[k] for k in ("t", "d_exec_mm", "d_ref_mm", "d_refined_mm", "grip_cmd", "open_meas", "dz_mm")} for e in tl],
            "refined_minus_reference": [{"t": x["t"], "max_arm_deg": x["max_arm_deg"]} for x in rr],
            "latch": [{"t": tt, "state": (c.get("grasp_latch") or {}).get("state"), "attach": (c.get("grasp_latch") or {}).get("attach"),
                       "detach": (c.get("grasp_latch") or {}).get("detach")} for tt, c in sorted(ch.items()) if t0 - 8 <= tt <= t1]}
    out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float))
    print("wrote", out)
