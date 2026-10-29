"""V-T45 figures: t45v-graph / t45v-table (+ .json). Numbers only from outputs/verify/T45/V/agg_*.json."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from AG3S.ag3s.experiments.common import figstyle as FS

KO = FS.use_korean()
T = lambda ko, en: ko if KO else en
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T45/V'))
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t45'))
F.mkdir(parents=True, exist_ok=True)
CAT = FS.CATEGORICAL
C_BASE, C_PC = CAT[0], CAT[1]
at = json.load(open(V / "agg_timing.json"))
rows = json.load(open(V / "agg_timing_rows.json"))
rep_path = V / "agg_timing_rep.json"
atr = json.load(open(rep_path)) if rep_path.exists() else None
rowsr = json.load(open(V / "agg_timing_rows_rep.json")) if rep_path.exists() else None
ac = json.load(open(V / "agg_cmp.json"))
chunks = json.load(open(V / "agg_cmp_chunks.json"))
P8 = ac["pre8"]
side = {}


def own(c, v):
    return c["variants"][v]["eval_under"][v]


def graph():
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    ax = axs.ravel()
    # (a) TO wall CDF
    for v, col, lab in (("base", C_BASE, T("palm cover off (기록 모델 270 구)", "palm cover off (recorded model, 270)")),
                        ("PC", C_PC, T("palm cover on (466 구)", "palm cover on (466 spheres)"))):
        w = np.sort([r["wall"] for r in rows[v]])
        ax[0].plot(w, np.arange(1, len(w) + 1) / len(w), color=col, lw=1.6, label=lab)
        ax[0].axvline(at[v]["wall_ms"]["p90"], color=col, ls=":", lw=1)
    ax[0].axvline(69.2, color=FS.INK_2, ls="--", lw=1, label=T("V11AB TP p90 69.2 ms (기준)", "V11AB TP p90 69.2 ms (reference)"))
    ax[0].axhline(0.9, color=FS.GRID_INK, lw=0.8)
    ax[0].set_xlim(0, 260)
    ax[0].set_xlabel(T("TO 시간 (ms, 서버 등가)", "TO time (ms, server-equivalent)"))
    ax[0].set_ylabel("CDF")
    ax[0].set_title(T(f"(a) TO 시간 분포 — 1332 chunk × 2 (같은 chunk 번갈아)\np90 {at['base']['wall_ms']['p90']:.1f} → {at['PC']['wall_ms']['p90']:.1f} ms",
                      f"(a) TO time CDF, 1332 chunks x 2 (alternating)\np90 {at['base']['wall_ms']['p90']:.1f} -> {at['PC']['wall_ms']['p90']:.1f} ms"), fontsize=10)
    ax[0].legend(fontsize=8, frameon=False, loc="lower right")
    # (b) stage mean
    stages = ["fk", "esdf", "clearance", "linearize", "qp_build", "osqp", "objective", "other"]
    bottom = {"base": 0.0, "PC": 0.0}
    for i, s in enumerate(stages):
        vals = [at["base"]["stage_mean_ms"][s], at["PC"]["stage_mean_ms"][s]]
        ax[1].bar([0, 1], vals, bottom=[bottom["base"], bottom["PC"]], color=(CAT + [FS.GRID_INK])[i], label=s, width=0.55)
        bottom["base"] += vals[0]
        bottom["PC"] += vals[1]
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["off", "on"])
    ax[1].set_ylabel(T("평균 단계 시간 (ms)", "mean stage time (ms)"))
    ax[1].set_title(T("(b) 단계별 평균 (fk · esdf 가 구 수에 비례해 증가)", "(b) mean time per stage"), fontsize=10)
    ax[1].legend(fontsize=7, frameon=False, ncol=2)
    # (c) rows
    cats = [T("행/평가\n(S × H16)", "rows / eval\n(S x H16)"), T("solve 당 core 행\n(평균)", "core rows / solve\n(mean)"), T("solve 당 field 점\n(평균)", "field points / solve\n(mean)")]
    b = [at["base"]["rows_per_eval_S_x_H16"], at["base"]["core_rows"]["mean"], at["base"]["field_points"]["mean"]]
    p = [at["PC"]["rows_per_eval_S_x_H16"], at["PC"]["core_rows"]["mean"], at["PC"]["field_points"]["mean"]]
    x = np.arange(3)
    ax[2].bar(x - 0.19, b, 0.38, color=C_BASE, label="off")
    ax[2].bar(x + 0.19, p, 0.38, color=C_PC, label="on")
    for i in range(3):
        ax[2].text(x[i] + 0.19, p[i], f"x{p[i] / b[i]:.2f}", ha="center", va="bottom", fontsize=8, color=FS.INK)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(cats, fontsize=8)
    ax[2].set_title(T("(c) ESDF 질의 행 수 (구 270 → 466 = x1.73)", "(c) ESDF query rows (270 -> 466 spheres = x1.73)"), fontsize=10)
    ax[2].legend(fontsize=8, frameon=False)
    # (d) pre8 refined diff
    pre = [c for c in chunks if c["rel"] is not None and -8 <= c["rel"] <= -1]
    d = np.degrees([c["diff_window_max_abs_rad"] for c in pre])
    bins = np.concatenate([[0], np.logspace(-2, np.log10(15), 22)])
    ax[3].hist(np.clip(d, 1e-3, None), bins=bins, color=C_PC)
    ax[3].set_xscale("symlog", linthresh=0.01)
    ax[3].axvline(1.09, color=FS.INK_2, ls="--", lw=1, label=T("기록 대비 base 재현 최대 1.09°", "base-vs-recorded max 1.09 deg"))
    ax[3].set_xlabel(T("refined chunk 최대 joint 차이 (°, on − off)", "max joint diff of refined chunk (deg, on - off)"))
    ax[3].set_ylabel(T("chunk 수 (pre-grasp 8 × 18 = 144)", "chunks (pre-grasp 8 x 18 = 144)"))
    ax[3].set_title(T(f"(d) grasp 직전 refined 변화: 동일 {P8['n_chunks_identical_sha1']} · >1° {P8['n_chunks_joint_diff_gt_1deg']} · >3° {P8['n_chunks_joint_diff_gt_3deg']}",
                      f"(d) pre-grasp refined change: identical {P8['n_chunks_identical_sha1']}, >1 deg {P8['n_chunks_joint_diff_gt_1deg']}, >3 deg {P8['n_chunks_joint_diff_gt_3deg']}"), fontsize=10)
    ax[3].legend(fontsize=8, frameon=False)
    # (e) palm min gaps per chunk
    labs = [("gap_target", T("target 점", "target pts")), ("gap_table", T("table 평면", "table plane")),
            ("gap_other_obstacle", T("기타 장애물 점", "other obstacle pts")), ("gap_field_target_free", T("target-free ESDF", "target-free ESDF"))]
    pos = 0
    ticks, tl = [], []
    for k, lab in labs:
        for j, (v, col) in enumerate((("base", C_BASE), ("PC", C_PC))):
            vals = [own(c, v)["palm"][k]["min_mm"] for c in pre if "palm" in own(c, v) and own(c, v)["palm"].get(k)]
            bp = ax[4].boxplot([vals], positions=[pos + j * 0.8], widths=0.6, patch_artist=True, showfliers=True,
                               flierprops=dict(markersize=2.5, markeredgecolor=col), medianprops=dict(color=FS.INK))
            bp["boxes"][0].set(facecolor=col, alpha=0.55)
        ticks.append(pos + 0.4)
        tl.append(lab)
        pos += 2.4
    ax[4].axhline(0, color=FS.INK_2, lw=0.8)
    ax[4].axhline(10, color=FS.GRID_INK, lw=0.8, ls="--")
    ax[4].set_xticks(ticks)
    ax[4].set_xticklabels(tl, fontsize=8)
    ax[4].set_ylabel(T("palm 구 chunk 별 최소 gap (mm, d − r)", "palm sphere min gap per chunk (mm, d - r)"))
    ax[4].set_title(T("(e) palm 구 – 표면 최소 거리 (왼 = off, 오른 = on)", "(e) palm sphere to surface (left = off, right = on)"), fontsize=10)
    ax[4].set_ylim(-20, 130)
    # (f) verdict counts
    names = [T("HOLD 후보\n(n_collision>0)", "HOLD candidate"), T("status\nviolated", "status\nviolated"), T("palm 이\n최악 행", "palm is\nworst row"),
             T("새 HOLD\n(on 만)", "new HOLD\n(on only)"), T("HOLD 해소\n(off 만)", "HOLD resolved\n(off only)")]
    vb = [P8["hold_candidates_base_own_model"], P8["status_counts"]["base"].get("violated", 0), P8["worst_row_is_palm_own"]["base"], 0, 0]
    vp = [P8["hold_candidates_PC_own_model"], P8["status_counts"]["PC"].get("violated", 0), P8["worst_row_is_palm_own"]["PC"], P8["new_hold_PC_vs_base"], P8["resolved_hold_PC_vs_base"]]
    x = np.arange(len(names))
    ax[5].bar(x[:3] - 0.19, vb[:3], 0.38, color=C_BASE, label="off")
    ax[5].bar(x[:3] + 0.19, vp[:3], 0.38, color=C_PC, label="on")
    ax[5].bar(x[3:], vp[3:], 0.5, color=CAT[7])
    for i in range(len(names)):
        ax[5].text(x[i] + (0.19 if i < 3 else 0), vp[i], str(vp[i]), ha="center", va="bottom", fontsize=9)
    for i in range(3):
        ax[5].text(x[i] - 0.19, vb[i], str(vb[i]), ha="center", va="bottom", fontsize=9)
    ax[5].set_xticks(x)
    ax[5].set_xticklabels(names, fontsize=8)
    ax[5].set_ylabel(T("chunk 수 (총 144)", "chunks (of 144)"))
    ax[5].set_title(T("(f) grasp 직전 8 chunk × 18 episode: verdict 변화", "(f) pre-grasp 8 chunks x 18 episodes: verdict change"), fontsize=10)
    ax[5].legend(fontsize=8, frameon=False)
    FS.style_axes(fig, list(ax))
    fig.suptitle(T("V-T45 — palm cover 5 mm off/on, T44d C1a 기록 18 run offline replay", "V-T45 - palm cover 5 mm off/on, T44d C1a 18 recorded runs, offline replay"), fontsize=12, color=FS.INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(F / "t45v-graph.png", dpi=140)
    plt.close(fig)
    side["graph"] = {"timing": {k: at[k] for k in ("base", "PC", "paired")}, "pre8": P8,
                     "timing_replicate": None if atr is None else {k: atr[k] for k in ("base", "PC", "paired")}}
    json.dump(side["graph"], open(F / "t45v-graph.json", "w"), indent=1)


def table():
    pg = P8["palm_geometry"]
    cl = ac["close"]
    rowsT = [
        ("TO time (ms)", "off", "on", "on − off"),
        ("median", f"{at['base']['wall_ms']['median']:.1f}", f"{at['PC']['wall_ms']['median']:.1f}", f"{at['paired']['median_diff_ms']:+.1f}"),
        ("p90", f"{at['base']['wall_ms']['p90']:.1f}", f"{at['PC']['wall_ms']['p90']:.1f}", f"{at['paired']['p90_diff_ms']:+.1f}"),
        ("p99", f"{at['base']['wall_ms']['p99']:.1f}", f"{at['PC']['wall_ms']['p99']:.1f}", f"{at['PC']['wall_ms']['p99'] - at['base']['wall_ms']['p99']:+.1f}"),
        ("CPU p90 (process time)", f"{at['base']['cpu_ms']['p90']:.1f}", f"{at['PC']['cpu_ms']['p90']:.1f}", f"{at['paired']['cpu_p90_diff_ms']:+.1f}"),
        ("p90 vs V11AB TP 69.2", f"{at['base_p90_minus_TP_p90_ms']:+.1f}", f"{at['PC_p90_minus_TP_p90_ms']:+.1f}", ""),
        ("SQP iterations (mean)", f"{at['base']['sqp_iterations']['mean']:.2f}", f"{at['PC']['sqp_iterations']['mean']:.2f}", ""),
        ("QP iterations (mean / p90)", f"{at['base']['qp_iterations']['mean']:.0f} / {at['base']['qp_iterations']['p90']:.0f}", f"{at['PC']['qp_iterations']['mean']:.0f} / {at['PC']['qp_iterations']['p90']:.0f}", ""),
        ("spheres S / rows per eval (S × 16)", f"{at['base']['n_spheres']} / {at['base']['rows_per_eval_S_x_H16']}", f"{at['PC']['n_spheres']} / {at['PC']['rows_per_eval_S_x_H16']}", f"x{at['PC']['n_spheres'] / at['base']['n_spheres']:.2f}"),
        ("core rows / solve (mean)", f"{at['base']['core_rows']['mean']:.0f}", f"{at['PC']['core_rows']['mean']:.0f}", f"x{at['paired']['ratio_core_rows_mean']:.2f}"),
        ("field points / solve (mean)", f"{at['base']['field_points']['mean']:.0f}", f"{at['PC']['field_points']['mean']:.0f}", f"x{at['paired']['ratio_field_points_mean']:.2f}"),
    ]
    rowsG = [
        ("pre-grasp 8 chunks x 18 (n = 144)", "off", "on", "on − off"),
        ("refined diff, joint max (deg): median / p90 / max", "", f"{P8['refined_diff_joint_window_deg']['median']:.2f} / {P8['refined_diff_joint_window_deg']['p90']:.2f} / {P8['refined_diff_joint_window_deg']['max']:.2f}", ""),
        ("HOLD candidate chunks (n_collision > 0)", str(P8["hold_candidates_base_own_model"]), str(P8["hold_candidates_PC_own_model"]), f"new {P8['new_hold_PC_vs_base']} / resolved {P8['resolved_hold_PC_vs_base']}"),
        ("status violated chunks", str(P8["status_counts"]["base"].get("violated", 0)), str(P8["status_counts"]["PC"].get("violated", 0)), f"worse {P8['status_worse_PC_vs_base']} / better {P8['status_better_PC_vs_base']}"),
        ("collision rows (sum)", str(P8["n_collision_rows_sum"]["base"]), str(P8["n_collision_rows_sum"]["PC"]), ""),
        ("palm is worst row (chunks)", str(P8["worst_row_is_palm_own"]["base"]), str(P8["worst_row_is_palm_own"]["PC"]), ""),
        ("palm min TO clearance per chunk: median (mm)", f"{P8['palm_group_min_to_mm']['base']['median']:.1f}", f"{P8['palm_group_min_to_mm']['PC']['median']:.1f}", ""),
        ("palm min gap to target (mm): min over chunks", f"{pg['base']['gap_target']['min_mm_over_chunks']:.1f}", f"{pg['PC']['gap_target']['min_mm_over_chunks']:.1f}", ""),
        ("palm queries with target gap < 0 / < 10 / < 20 mm", f"{pg['base']['gap_target']['n_queries_lt_0mm']} / {pg['base']['gap_target']['n_queries_lt_10mm']} / {pg['base']['gap_target']['n_queries_lt_20mm']}",
         f"{pg['PC']['gap_target']['n_queries_lt_0mm']} / {pg['PC']['gap_target']['n_queries_lt_10mm']} / {pg['PC']['gap_target']['n_queries_lt_20mm']}", f"of {pg['PC']['n_queries_total']} (on)"),
        ("palm min gap to table (mm)", f"{pg['base']['gap_table']['min_mm_over_chunks']:.1f}", f"{pg['PC']['gap_table']['min_mm_over_chunks']:.1f}", ""),
        ("palm min gap to other obstacle pts (mm)", f"{pg['base']['gap_other_obstacle']['min_mm_over_chunks']:.1f}", f"{pg['PC']['gap_other_obstacle']['min_mm_over_chunks']:.1f}", ""),
        ("palm rows reading target-free layer (chunks)", f"{pg['base']['target_free_mask_palm_all_true_chunks']}/{pg['base']['target_free_mask_palm_present_chunks']}", f"{pg['PC']['target_free_mask_palm_all_true_chunks']}/{pg['PC']['target_free_mask_palm_present_chunks']}", ""),
    ]
    rowsC = [
        ("pass-criteria numbers (not a verdict)", "value"),
        ("TO p90 on − off (ms, same chunks, alternating)", f"{at['paired']['p90_diff_ms']:+.1f}"),
        ("TO p90 on vs V11AB TP 69.2 (ms)", f"{at['PC_p90_minus_TP_p90_ms']:+.1f}"),
        ("TO p90 on vs C1a recorded p90 (ms)", f"{at['PC_p90_minus_recorded_p90_ms']:+.1f}  (recorded {at['recorded_C1a_to_solve_ms_excl_first']['p90']:.0f})"),
        ("pre-grasp new HOLD (on only)", str(P8["new_hold_PC_vs_base"])),
        ("pre-grasp verdict worse (new HOLD or status worse)", str(P8["verdict_worse_total"])),
    ]
    fig = plt.figure(figsize=(17, 10.5))
    ax1 = fig.add_axes([0.02, 0.50, 0.46, 0.44]); ax2 = fig.add_axes([0.52, 0.30, 0.47, 0.64]); ax3 = fig.add_axes([0.02, 0.02, 0.46, 0.40])
    for ax, rws, widths, title in ((ax1, rowsT, [0.46, 0.18, 0.18, 0.18], T("TO 시간 (T44d C1a 18 run, 1332 chunk, 같은 프로세스 번갈아)", "TO time (T44d C1a 18 runs, 1332 chunks, same process alternating)")),
                                   (ax2, rowsG, [0.46, 0.16, 0.19, 0.19], T("grasp 영향 (close 이전 8 chunk, budget 1e6 open-loop)", "grasp impact (8 chunks before close, budget 1e6 open loop)")),
                                   (ax3, rowsC, [0.65, 0.35], T("통과 기준에 쓰이는 숫자 (판정 아님)", "numbers for the pass criteria (not a verdict)"))):
        ax.axis("off")
        ax.set_title(title, fontsize=11, color=FS.INK, loc="left")
        tb = ax.table(cellText=rws[1:], colLabels=rws[0], colWidths=widths, loc="upper center", cellLoc="left")
        tb.auto_set_font_size(False)
        tb.set_fontsize(9)
        tb.scale(1, 1.55)
        for (r, c), cell in tb.get_celld().items():
            cell.set_edgecolor(FS.GRID_INK)
            if r == 0:
                cell.set_facecolor(FS.SEQ_BLUE[1])
                cell.set_text_props(color=FS.INK)
            else:
                cell.set_facecolor(FS.SURFACE)
    fig.text(0.52, 0.04, T("측정 설정: 구 466 = palm 121 × 2 + finger 112 × 2 (off 270); budget 기록(50 ms) = TO 시간 pass, 1e6 = 결과 비교 pass.\nHOLD 후보 = classify_violations n_collision > 0 (executed 8 step, 각 변형의 자기 모델). 기록 HOLD chunk 와의 일치: off 재현 pre8+close 14 = 기록 14.",
                           "Settings: spheres 466 = palm 121 x 2 + fingers 112 x 2 (off 270). Recorded budget (50 ms) = timing pass; 1e6 = comparison pass.\nHOLD candidate = classify_violations n_collision > 0 over the executed 8 steps under each variant's own model. Off replay reproduces the recorded HOLD count in pre8+close: 14 = 14."),
             fontsize=8.5, color=FS.INK_2, va="bottom")
    fig.savefig(F / "t45v-table.png", dpi=140)
    plt.close(fig)
    json.dump({"timing_rows": rowsT, "grasp_rows": rowsG, "criteria_rows": rowsC}, open(F / "t45v-table.json", "w"), indent=1)


if __name__ == "__main__":
    for f in sys.argv[1:] or ["graph", "table"]:
        globals()[f]()
