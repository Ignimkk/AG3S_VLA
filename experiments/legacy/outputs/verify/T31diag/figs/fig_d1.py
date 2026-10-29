"""T31-diag D1 figures: (1) metric strip per condition, (2) timeline of executed left-arm joint velocity + HOLD + refined-reference."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs
KO = fs.use_korean(); C = fs.CATEGORICAL
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t31'))
D = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T31diag/d1/d1_oscillation.json')).read_text())
CC = {"E2": C[0], "E3a": C[1], "E3b": C[2]}
runs = D["runs"]
metrics = [("sign_reversals_per_s", "joint-velocity sign reversals / s (7 left-arm joints)" if not KO else "관절 속도 부호 반전 / s (왼팔 7 관절)"),
           ("jerk_rms", "jerk RMS (rad/s³)"),
           ("fingertip.mean_curvature_per_m", "fingertip mean curvature (1/m)" if not KO else "손끝 평균 곡률 (1/m)"),
           ("fingertip.path_over_chord", "fingertip path / chord" if not KO else "손끝 경로 길이 / 직선거리")]
def get(w, k):
    for p in k.split("."):
        w = w[p]
    return w
fig, axes = plt.subplots(1, 4, figsize=(19, 5.2))
side = {"source": "outputs/verify/T31diag/d1/d1_oscillation.json", "window": "all (grasp_t..release_t)", "values": {}}
for ax, (k, lab) in zip(axes, metrics):
    for i, cond in enumerate(("E2", "E3a", "E3b")):
        vals = [(n, get(r["windows"]["all"], k)) for n, r in runs.items() if r["condition"] == cond]
        xs = i + np.linspace(-0.15, 0.15, len(vals))
        ax.scatter(xs, [v for _, v in vals], color=CC[cond], s=40, zorder=3)
        ax.hlines(np.median([v for _, v in vals]), i - 0.25, i + 0.25, color=CC[cond], lw=2)
        for x, (n, v) in zip(xs, vals):
            ax.annotate(n.split("_", 1)[1].replace("ep", ""), (x, v), xytext=(4, 0), textcoords="offset points", fontsize=6.5, color=fs.INK_2)
            side["values"].setdefault(n, {})[k] = v
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["E2\n(no field, 6)", "E3a\n(ep1807 r1 r2 r3)", "E3b\n(ep1807 r1 r2)"] if not KO else ["E2\n(거리장 없음, 6)", "E3a\n(ep1807 r1 r2 r3)", "E3b\n(ep1807 r1 r2)"], fontsize=8)
    ax.set_title(lab, fontsize=10)
fs.style_axes(fig, list(axes))
fig.suptitle("D1 — executed trajectory, grasp → release window (bar = median; E3a r1 and r2 share the RNG sequence)" if not KO else
             "D1 — 실행 궤적, grasp → release 구간 (막대 = 중앙값; E3a r1·r2 는 같은 RNG 순서)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(F / "t31-d1-oscillation-metrics.png", dpi=120)
(F / "t31-d1-oscillation-metrics.json").write_text(json.dumps(side, indent=1))
# timeline
sel = ["E2_ep1807_r1", "E2_ep1807_r2", "E3a_ep1807_r1", "E3a_ep1807_r3", "E3b_ep1807_r1", "E3b_ep1807_r2"]
fig, axes = plt.subplots(len(sel), 1, figsize=(16, 3.0 * len(sel)), sharex=False)
side2 = {"source": "outputs/verify/T31diag/d1/d1_oscillation.json", "runs": {}}
for ax, n in zip(axes, sel):
    r = runs[n]
    P = np.array(r["windows"]["all"]["fingertip_path_world"])
    rows = [json.loads(l) for l in (pathlib.Path(r["run_dir"]) / "frames.jsonl").read_text().splitlines() if l.strip()]
    ctrl = [x for x in rows if x["kind"] == "control" and r["grasp_t"] <= x["t_step"] <= r["release_t"]]
    t = np.array([x["t_step"] for x in ctrl]); q = np.array([[x["qpos"][a] for a in range(20, 27)] for x in ctrl])
    v = np.diff(q, axis=0) * 15.0
    for j in range(7):
        ax.plot(t[:-1], v[:, j], color=C[j], lw=0.9, label=f"left_arm_{j}")
    for h in r["hold_intervals"]:
        if h["t_last"] >= r["grasp_t"] and h["t_first"] <= r["release_t"]:
            ax.axvspan(h["t_first"], h["t_last"] + 1, color=fs.GRID_INK, alpha=0.8, lw=0)
    ax2 = ax.twinx()
    rr = [x for x in r["refined_minus_reference"] if x["in_window"]]
    ax2.step([x["t"] for x in rr], [x["left_arm_max_deg"] for x in rr], where="post", color=fs.INK, lw=1.4, ls="--")
    ax2.set_ylim(0, 20); ax2.set_ylabel("refined−reference (deg)", fontsize=8)
    ax.set_ylabel("dq/dt (rad/s)"); ax.set_title(f"{n}  grasp t={r['grasp_t']}  place t={r['place_t']}  release t={r['release_t']}  "
                                                 f"reversals/s {r['windows']['all']['sign_reversals_per_s']:.2f}  jerk RMS {r['windows']['all']['jerk_rms']:.1f}", fontsize=10)
    side2["runs"][n] = {"hold_intervals": r["hold_intervals"], "refined_minus_reference": [(x["t"], x["left_arm_max_deg"]) for x in rr]}
axes[0].legend(ncol=7, fontsize=8, loc="upper right")
fs.style_axes(fig, list(axes))
fig.suptitle("D1 — executed left-arm joint velocity (lines), HOLD spans (grey), refined−reference per chunk (dashed, right axis)" if not KO else
             "D1 — 실행된 왼팔 관절 속도 (선), HOLD 구간 (회색), 청크별 refined−reference (점선, 오른쪽 축)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(F / "t31-d1-oscillation-timeline.png", dpi=110)
(F / "t31-d1-oscillation-timeline.json").write_text(json.dumps(side2, indent=1))
print("ok")
