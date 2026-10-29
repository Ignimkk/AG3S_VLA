"""Figures for the 16D attention-cell re-validation (numbers from summary.json / score.json / cubes.npz)."""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from AG3S.ag3s.experiments.common import figstyle
from AG3S.ag3s.experiments.common.figstyle import CATEGORICAL, GRID_INK, INK, INK_2, SURFACE

D = pathlib.Path(_ag3s_workspace_path('outputs/verify/attn16d'))
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/r-16d'))
S = json.loads((D / "score.json").read_text())
SUM = json.loads((D / "summary.json").read_text())
C = np.load(D / "cubes.npz")
KO = figstyle.use_korean()
DEN, AGG = S["config"]["denoise"], S["config"]["aggs"]
EPS = S["config"]["episodes"]
VIEW_LABEL = {"cam_high": "cam_high (머리)", "cam_left_wrist": "cam_left_wrist (왼손목)",
              "cam_right_wrist": "cam_right_wrist (오른손목)", "any": "세 카메라 중 하나라도"}
EP_LABEL = {"ep1800_t2": "ep1800", "ep1807": "ep1807", "ep1850": "ep1850", "ep1900": "ep1900"}


def save(fig, name, numbers):
    for ext in ("png", "json"):
        p = F / f"{name}.{ext}"
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    fig.savefig(F / f"{name}.png", facecolor=SURFACE, bbox_inches="tight", dpi=160)
    (F / f"{name}.json").write_text(json.dumps(numbers, indent=1))
    plt.close(fig)
    print("wrote", F / f"{name}.png")


def lab(r):
    return f"d{r['denoise']} {r['agg']} L{r['layer']}h{r['head']}"


# ------------------------------------------------------------------ (a) heatmap
def fig_heatmap():
    views = ["cam_high", "cam_left_wrist", "any"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 8.2))
    figstyle.style_axes(fig, axes)
    cmap = figstyle.sequential_cmap()
    nums = {}
    for ax, v in zip(axes, views):
        m = C[f"reach|pooled|{v}|peak"][DEN.index(0), AGG.index("last")]
        n = int(C[f"reach|pooled|{v}|n"])
        nums[v] = {"n_frames": n, "peak_d0_last": m.round(4).tolist()}
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for l in range(m.shape[0]):
            for h in range(m.shape[1]):
                val = m[l, h]
                ax.text(h, l, "1.0" if val >= 0.9995 else f"{val:.2f}"[1:], ha="center", va="center",
                        fontsize=6.3, color="#ffffff" if val > 0.55 else INK)
        for (l, h), col, name in (((8, 2), CATEGORICAL[1], "L8h2"), ((7, 0), CATEGORICAL[6], "L7h0")):
            ax.add_patch(plt.Rectangle((h - .5, l - .5), 1, 1, fill=False, edgecolor=col, linewidth=2.4))
        ax.set_xticks(range(8), [f"h{h}" for h in range(8)])
        ax.set_yticks(range(18), [f"L{l}" for l in range(18)])
        ax.set_xlabel("attention head")
        ax.set_title(f"{VIEW_LABEL[v]}  ·  채점 프레임 {n}", fontsize=10, color=INK, loc="left")
        ha = S["phases"]["reach"]["scopes"]["pooled"][v]["baseline_head_average_d0_last"]["peak"]
        ax.text(0, -0.075, f"L8h2 = {m[8, 2]:.3f}   L7h0 = {m[7, 0]:.3f}   head-average {ha:.3f}",
                transform=ax.transAxes, fontsize=8, color=INK_2)
    axes[0].set_ylabel("transformer 층")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("peak-on-target", color=INK_2, fontsize=8)
    fig.suptitle("16D · reach 구간 · Euler step 0, last pooling — (층, 헤드)별 peak-on-target, 4 에피소드 합산"
                 "   (주황 = L8h2, 보라 = L7h0)", x=0.01, ha="left", fontsize=11, color=INK)
    save(fig, "attention-cell-16d-heatmap", nums)


# ------------------------------------------------------------------ (b) top-15 bars
def fig_top15():
    views = ["cam_high", "cam_left_wrist", "any"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    figstyle.style_axes(fig, axes)
    nums = {}
    for ax, v in zip(axes, views):
        e = S["phases"]["reach"]["scopes"]["pooled"][v]
        rows = e["top15"]
        key = e["key_cells"]["L8h2_d0_last"]
        names = [lab(r) for r in rows]
        vals = [r["peak"] for r in rows]
        extra = []
        if "d0 last L8h2" not in names:
            names.append(f"d0 last L8h2 (#{key['rank']})")
            vals.append(key["peak"])
        colors = [CATEGORICAL[1] if n.startswith("d0 last L8h2") else CATEGORICAL[0] for n in names]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, color=colors, height=0.72, edgecolor=SURFACE)
        for yi, val in zip(y, vals):
            ax.text(val + 0.01, yi, f"{val:.3f}", va="center", fontsize=7, color=INK_2)
        ha = e["baseline_head_average_d0_last"]["peak"]
        un = e["baseline_uniform"]["peak"]
        ax.axvline(ha, color=INK_2, linestyle=(0, (4, 3)), linewidth=1.1)
        ax.text(ha, len(names) - 0.2, f" head-average {ha:.3f}", fontsize=7.5, color=INK_2, va="bottom")
        ax.set_yticks(y, [f"{i + 1}. {n}" if i < 15 else n for i, n in enumerate(names)], fontsize=7.5)
        ax.set_xlim(0, 1.12)
        ax.set_xlabel("peak-on-target")
        ax.set_title(f"{VIEW_LABEL[v]}  ·  n={e['n_frames']}  ·  uniform {un:.3f}", fontsize=10, color=INK, loc="left")
        ax.grid(axis="x", color=GRID_INK, linewidth=0.6)
        ax.set_axisbelow(True)
        nums[v] = {"n_frames": e["n_frames"], "top15": rows, "L8h2_d0_last": key,
                   "head_average_d0_last": e["baseline_head_average_d0_last"], "uniform": e["baseline_uniform"]}
    fig.suptitle("16D · reach 구간 · 1296 셀 중 상위 15 (순위 키: peak → hit β0.5 → lift), 주황 = d0 last L8h2",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "attention-cell-16d-top15", nums)


# ------------------------------------------------------------------ (c) per-episode
def fig_episodes():
    views = ["cam_high", "cam_left_wrist", "any"]
    phases = ["reach", "carry"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    figstyle.style_axes(fig, axes.ravel().tolist())
    nums = {}
    series = [("최고 셀", CATEGORICAL[0]), ("L8h2 d0 last", CATEGORICAL[1]),
              ("L7h0 d0 last", CATEGORICAL[6]), ("head-average", GRID_INK)]
    for pi, ph in enumerate(phases):
        for vi, v in enumerate(views):
            ax = axes[pi, vi]
            scopes = [s for s in EPS + ["pooled"] if S["phases"][ph]["scopes"][s][v] is not None]
            x = np.arange(len(scopes))
            w = 0.2
            rec = {}
            for si, (name, col) in enumerate(series):
                vals, ranks = [], []
                for s in scopes:
                    e = S["phases"][ph]["scopes"][s][v]
                    if name == "최고 셀":
                        vals.append(e["top15"][0]["peak"]); ranks.append(lab(e["top15"][0]))
                    elif name == "head-average":
                        vals.append(e["baseline_head_average_d0_last"]["peak"]); ranks.append(None)
                    else:
                        k = e["key_cells"]["L8h2_d0_last" if "L8h2" in name else "L7h0_d0_last"]
                        vals.append(k["peak"]); ranks.append(k["rank"])
                xs = x + (si - 1.5) * w
                ax.bar(xs, vals, w * 0.92, color=col, label=name, edgecolor=SURFACE)
                if name.startswith("L8h2"):
                    for xi, val, r in zip(xs, vals, ranks):
                        ax.text(xi, val + 0.02, f"#{r}", ha="center", fontsize=7, color=CATEGORICAL[1])
                if name.startswith("L7h0"):
                    for xi, val, r in zip(xs, vals, ranks):
                        ax.text(xi, val + 0.02, f"#{r}", ha="center", fontsize=6, color=CATEGORICAL[6], rotation=90)
                rec[name] = {"values": vals, "ranks_or_cells": ranks}
            ns = [S["phases"][ph]["scopes"][s][v]["n_frames"] for s in scopes]
            ax.set_xticks(x, [f"{EP_LABEL.get(s, '합산')}\nn={n}" for s, n in zip(scopes, ns)], fontsize=8)
            ax.set_ylim(0, 1.18)
            ax.grid(axis="y", color=GRID_INK, linewidth=0.6)
            ax.set_axisbelow(True)
            tgt = "과일 (프롬프트)" if ph == "reach" else "바구니 (crate)"
            ax.set_title(f"{ph} · target = {tgt} · {VIEW_LABEL[v]}", fontsize=9.5, color=INK, loc="left")
            nums[f"{ph}|{v}"] = {"scopes": scopes, "n_frames": ns, **rec}
        axes[pi, 0].set_ylabel("peak-on-target")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("에피소드별 — 최고 셀 vs L8h2 d0 last vs L7h0 d0 last vs head-average   (#숫자 = 1296 셀 중 순위)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(fig, "attention-cell-16d-episodes", nums)


# ------------------------------------------------------------------ (d) example frames
def fig_frames():
    picks = [("ep1807", 8), ("ep1850", 5), ("ep1900", 9)]
    cams = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    cmap = figstyle.sequential_cmap()
    fig, axes = plt.subplots(len(picks), 3, figsize=(10.5, 3.7 * len(picks)))
    fig.patch.set_facecolor(SURFACE)
    nums = []
    for ri, (ep, fi) in enumerate(picks):
        att = np.load(D / f"att_{ep}.npz")
        gt = np.load(D / f"gt_{ep}.npz")
        rec = json.loads(json.dumps(S["episodes"][ep]))
        bodies = [str(b) for b in gt["bodies"]]
        ti = bodies.index(rec["fruit"])
        from AG3S.ag3s.experiments.sources.policy_record import load_run
        run = load_run(str(gt["record"]), limit=fi + 1)
        step = run.steps[fi]
        for ci, cam in enumerate(cams):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID_INK)
            img = step.images[cam]
            a = np.asarray(att["attention"][fi, DEN.index(0), AGG.index("last"), 8, 2, ci], np.float32)
            cov = gt["cov_pol"][fi, ci, ti]
            area = float(gt["area_640"][fi, ci, ti])
            ax.imshow(img)
            ax.imshow(a / max(a.max(), 1e-12), cmap=cmap, alpha=0.55, extent=(0, 224, 224, 0),
                      interpolation="bilinear")
            if cov.max() > 0:
                px = np.arange(224) + 0.5  # pixel centres; image axes have y pointing down
                ax.contour(px, px, np.kron(cov, np.ones((14, 14))), levels=[0.05], colors=[CATEGORICAL[1]],
                           linewidths=1.6)
            r, c = np.unravel_index(a.argmax(), a.shape)
            hit = bool(cov[r, c] > 0.05)
            ax.plot((c + .5) * 14, (r + .5) * 14, marker="o", markersize=9, markerfacecolor="none",
                    markeredgecolor=CATEGORICAL[2] if hit else CATEGORICAL[7], markeredgewidth=2.0)
            kept = area >= 200
            status = ("적중" if hit else "빗나감") if kept else f"채점 제외 ({area:.0f}px<200)"
            ax.set_title(f"{cam} · {status}", fontsize=8.5, color=INK_2)
            if ci == 0:
                ax.set_ylabel(f"{EP_LABEL[ep]} · 프레임 {fi} (t={int(step.t_step)})\ntarget={rec['fruit']}",
                              fontsize=8.5, color=INK_2)
            nums.append({"episode": ep, "frame": fi, "t_step": int(step.t_step), "camera": cam,
                         "target": rec["fruit"], "target_px_640x480": area, "kept": kept,
                         "argmax_rc": [int(r), int(c)], "target_cov_at_argmax": float(cov[r, c]), "peak_hit": hit})
    fig.suptitle("reach 구간 예시 — 정책이 본 224×224 입력 위의 L8h2 d0 last attention\n"
                 "주황 윤곽 = target 점유 > 0.05 인 패치 (적중 기준과 같음) · 초록 ○ = argmax 가 target 위 · 빨강 ○ = 빗나감",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "attention-cell-16d-frames", nums)


# ------------------------------------------------------------------ table
def fig_table():
    rows = []
    hdr = ["구간", "카메라", "n", "1위 셀 (peak)", "L8h2 d0 last 순위 / peak / β0 β0.5 β1 / lift",
           "L7h0 d0 last 순위 / peak", "head-avg peak", "14D 판정 규칙"]
    nums = []
    for ph in ("reach", "carry"):
        for v in ("cam_high", "cam_left_wrist", "cam_right_wrist", "any"):
            p = SUM[ph][v]["pooled"]
            if not p:
                rows.append([ph, v, "0", "—", "target < 200 px 전 프레임", "—", "—", "—"])
                continue
            a, b = p["L8h2_d0_last"], p["L7h0_d0_last"]
            rule = "충족" if a["rule_14d_satisfied"] else ("미충족 (β " + ",".join(a["failing_betas"]) + ")"
                                                         if a["failing_betas"] else "미충족")
            if a["peak_margin_vs_max_baseline"] < 0.15:
                rule += f" margin {a['peak_margin_vs_max_baseline']:+.2f}"
            rows.append([ph, v, str(p["n_frames"]), f"{p['top1'].replace('_', ' ')} ({p['top1_metrics']['peak']:.3f})",
                         f"#{a['rank']} / {a['peak']:.3f} / {a['hit_b0']:.2f} {a['hit_b0.5']:.2f} {a['hit_b1']:.2f} / {a['lift']:.1f}",
                         f"#{b['rank']} / {b['peak']:.3f}", f"{p['baseline_head_average_d0_last']['peak']:.3f}", rule])
            nums.append({"phase": ph, "view": v, **{k: p[k] for k in ("n_frames", "top1", "top1_metrics")},
                         "L8h2_d0_last": a, "L7h0_d0_last": b,
                         "head_average_d0_last": p["baseline_head_average_d0_last"],
                         "uniform": p["baseline_uniform"]})
    rows.append(["14D 선택 시", "cam_high", "28", "d0 last L8h2 (1.000)", "#1 / 1.000 / 0.29 0.93 1.00 / 91.8", "—", "0.107", "PASS"])
    fig, ax = plt.subplots(figsize=(16, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=hdr, loc="center", cellLoc="left", colLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.45)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID_INK)
        if r == 0:
            cell.set_facecolor("#eef3fb"); cell.set_text_props(color=INK, weight="bold")
        else:
            cell.set_text_props(color=INK)
            if rows[r - 1][1] == "cam_high" and rows[r - 1][0] == "reach":
                cell.set_facecolor("#fdf0e9")
    t.auto_set_column_width(list(range(len(hdr))))
    ax.set_title("16D 4 에피소드 합산 — 1296 셀 중 L8h2 · L7h0 의 위치 (n = target 이 ≥200 px 인 프레임)",
                 loc="left", fontsize=11, color=INK)
    save(fig, "attention-cell-16d-table", {"rows": nums, "14d_reference": {
        "source": "AG3S/ag3s/docs/archive/step-verification-20260904/step-01-attention.json",
        "record": "run_0002", "n_frames": 44, "n_scored": 28, "peak": 1.0, "hit_b0": 0.286, "hit_b0.5": 0.929, "hit_b1": 1.0, "lift": 91.8,
        "head_average_peak": 0.107}})


if __name__ == "__main__":
    F.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["heatmap", "top15", "episodes", "frames", "table"]
    for w in which:
        globals()[f"fig_{w}"]()
