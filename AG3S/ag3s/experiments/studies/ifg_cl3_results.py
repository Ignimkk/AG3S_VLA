"""IFG-CL3 result figures (rule A: scene, graph, table) from the run's agg.json — numbers only, no judgement text.

  .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_cl3_results --agg <run>/verify/IFG-CL3/agg.json \
      --out AG3S/ag3s/docs/figures/ifg-cl3 [--runs <run>/verify/IFG-CL3/runs] [--prefix ifg-cl3] [--max-scenes 4]

Figures (each with a same-name .json sidecar holding every number drawn):
  <prefix>-paired-outcome   2x2 paired outcome (both / only C1aN / only SURF / neither) per stratum + paired difference with 95 % CI
  <prefix>-arm-metrics      per arm: failure modes, contacts, HOLD chunks per run, completion time
  <prefix>-latency          client round trip / server total (median, p95) against the 533 ms budget; server stage medians
  <prefix>-outcome-grid     every episode x arm outcome cell (contact-free success / success with contact / grasp fail / other fail)
  <prefix>-discordant-scenes third-person video frames of the discordant pairs (one arm succeeded, the other did not), both arms

Arm names are expanded in every title: C1aN = noun attention + ball extent (current serving with noun target choice);
SURF = C1aN + attention-surface target extent + rendered robot mask. Colours: figstyle CATEGORICAL slot 0 = C1aN, slot 1 = SURF,
slot 2 = both arms, INK_2 = neither; never re-picked per chart.
"""
import argparse, json, pathlib, shutil, subprocess, sys, tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from AG3S.ag3s.experiments.common import figstyle as FS  # noqa: E402

ARMS = ("C1aN", "SURF")
STRATA = ("clear", "adjacent", "place")
COL = {"C1aN": FS.CATEGORICAL[0], "SURF": FS.CATEGORICAL[1], "both": FS.CATEGORICAL[2], "neither": FS.INK_2}
BUDGET_MS = 533.3
MODES = ("no_grasp", "regrasp", "slip_at_lift", "slip_in_transport", "not_placed", "timeout", "other")
KO = FS.use_korean()


def T(ko, en):
    return ko if KO else en


ARM_LONG = {"C1aN": T("C1aN: noun attention + 공 영역 (현재 serving + 명사 target 선택)",
                      "C1aN: noun attention + ball extent (current serving + noun target choice)"),
            "SURF": T("SURF: C1aN + attention 표면 영역 + robot 렌더 mask",
                      "SURF: C1aN + attention-surface extent + rendered robot mask"),
            "E0": T("E0: VLA checkpoint 단독 (AG3S 없음)", "E0: VLA checkpoint alone (no AG3S)"),
            "S2": T("S2: C1aN + 안전 경보 + 학습 없는 3D 쥠 규칙", "S2: C1aN + confidence alarm + no-learning 3D held rule")}


def set_arms(arms, strata):
    """Pick the two compared arms (first = reference) and the strata; colours follow the fixed categorical slots."""
    global ARMS, STRATA, COL
    ARMS, STRATA = tuple(arms), tuple(strata)
    COL = {ARMS[0]: FS.CATEGORICAL[0], ARMS[1]: FS.CATEGORICAL[1], "both": FS.CATEGORICAL[2], "neither": FS.INK_2}


def save(fig, out, name, side):
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{name}.png"
    fig.savefig(png, dpi=200, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    (out / f"{name}.json").write_text(json.dumps(side, indent=1, ensure_ascii=False, default=float))
    print(png)


def cell_outcome(a):
    """One arm's run -> outcome class used by the grid: primary / contact / no_grasp / fail / missing."""
    if a is None or a.get("success") is None:
        return "missing"
    if a.get("primary"):
        return "primary"
    if a["success"]:
        return "contact"
    return "no_grasp" if not a.get("grasp") else "fail"


# ----------------------------------------------------------------------------------------------------------------- 1. paired outcome
def fig_paired(agg, out, prefix):
    keys = ("both", f"only_{ARMS[0]}", f"only_{ARMS[1]}", "neither")
    labels = {"both": T("둘 다 성공", "both succeed"), f"only_{ARMS[0]}": T(f"{ARMS[0]} 만", f"only {ARMS[0]}"),
              f"only_{ARMS[1]}": T(f"{ARMS[1]} 만", f"only {ARMS[1]}"), "neither": T("둘 다 실패", "neither")}
    colours = {"both": COL["both"], f"only_{ARMS[0]}": COL[ARMS[0]], f"only_{ARMS[1]}": COL[ARMS[1]], "neither": COL["neither"]}
    groups = [("all", agg["paired_primary"])] + [(s, agg["by_stratum"][s]["paired_primary"]) for s in STRATA]
    gl = {"all": T("전체", "all")} | {s: s for s in STRATA}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
    FS.style_axes(fig, (ax, bx))
    y = np.arange(len(groups))[::-1]
    for i, (g, p) in enumerate(groups):
        left = 0
        for k in keys:
            v = p.get(k, 0)
            ax.barh(y[i], v, left=left, color=colours[k], edgecolor=FS.SURFACE, linewidth=0.6, height=0.62)
            if v:
                ax.text(left + v / 2, y[i], str(v), ha="center", va="center", fontsize=8,
                        color=FS.SURFACE if k != "neither" else FS.SURFACE)
            left += v
        ax.text(left + 0.4, y[i], f"n={p.get('n_pairs', 0)}", va="center", fontsize=8, color=FS.INK_2)
    ax.set_yticks(y); ax.set_yticklabels([gl[g] for g, _ in groups], fontsize=9, color=FS.INK)
    ax.set_xlabel(T("episode 쌍 수", "episode pairs"))
    ax.set_title(T("접촉 없는 작업 성공: 2×2 paired outcome", "contact-free task success: 2x2 paired outcome"), fontsize=10, color=FS.INK, loc="left")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colours[k]) for k in keys], labels=[labels[k] for k in keys],
              fontsize=8, frameon=False, loc="lower right", ncol=2)
    ax.grid(axis="x", color=FS.GRID_INK, linewidth=0.5); ax.set_axisbelow(True)
    side = {"primary_metric": "contact-free task success (agg paired_primary)", "groups": {}}
    for i, (g, p) in enumerate(groups):
        d, ci = p.get(f"diff_{ARMS[1]}_minus_{ARMS[0]}"), p.get("ci95_bootstrap")
        side["groups"][g] = {k: p.get(k) for k in keys} | {"n_pairs": p.get("n_pairs"), f"rate_{ARMS[0]}": p.get(f"rate_{ARMS[0]}"), f"rate_{ARMS[1]}": p.get(f"rate_{ARMS[1]}"),
                                                          f"diff_{ARMS[1]}_minus_{ARMS[0]}": d, "ci95_bootstrap": ci, "mcnemar_exact_p": p.get("mcnemar_exact_p")}
        if d is None or ci is None:
            continue
        bx.plot([ci[0], ci[1]], [y[i], y[i]], color=FS.INK_2, linewidth=1.6, solid_capstyle="butt")
        bx.plot([d], [y[i]], "o", color=COL[ARMS[1]], markersize=6)
        bx.text(ci[1] + 0.02, y[i], f"{d:+.3f}  [{ci[0]:+.3f}, {ci[1]:+.3f}]" + (f"  p={p['mcnemar_exact_p']:.3g}" if p.get("mcnemar_exact_p") is not None else ""),
                va="center", fontsize=7.5, color=FS.INK_2)
    bx.axvline(0, color=FS.GRID_INK, linewidth=1)
    bx.set_yticks(y); bx.set_yticklabels([gl[g] for g, _ in groups], fontsize=9, color=FS.INK)
    bx.set_xlim(-1.05, 1.05); bx.set_xlabel(T(f"paired 차이 {ARMS[1]} − {ARMS[0]} (성공 비율), 95 % bootstrap CI", f"paired difference {ARMS[1]} - {ARMS[0]} (success rate), 95 % bootstrap CI"))
    bx.set_title(T("paired 차이와 CI (p = McNemar exact, 참고)", "paired difference and CI (p = exact McNemar, reference)"), fontsize=10, color=FS.INK, loc="left")
    bx.grid(axis="x", color=FS.GRID_INK, linewidth=0.5); bx.set_axisbelow(True)
    fig.suptitle(ARM_LONG[ARMS[0]] + "\n" + ARM_LONG[ARMS[1]], fontsize=8.5, color=FS.INK_2, x=0.01, ha="left", y=1.04)
    side["bootstrap"] = agg["paired_primary"].get("bootstrap")
    save(fig, out, f"{prefix}-paired-outcome", side)


# ----------------------------------------------------------------------------------------------------------------- 2. arm metrics
def fig_arm_metrics(agg, out, prefix):
    rows = agg["rows"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1.2, 1.1, 1, 1]})
    FS.style_axes(fig, axes)
    ax, bx, cx, dx = axes
    side = {"arms": {}}
    x = np.arange(len(ARMS))
    # failure modes (excluding ok), stacked
    present = [m for m in MODES if any((agg[a]["all"]["failure_modes"].get(m, 0)) for a in ARMS)]
    extra = sorted({m for a in ARMS for m in agg[a]["all"]["failure_modes"] if m not in MODES and m not in ("ok", "None")})
    present += extra
    ramp = FS.SEQ_BLUE[2::max(1, (len(FS.SEQ_BLUE) - 2) // max(1, len(present)))][:len(present)] if present else []
    for i, a in enumerate(ARMS):
        fm = agg[a]["all"]["failure_modes"]; left = 0
        for j, m in enumerate(present):
            v = fm.get(m, 0)
            ax.bar(x[i], v, bottom=left, color=ramp[j], edgecolor=FS.SURFACE, linewidth=0.6, width=0.6)
            if v:
                ax.text(x[i], left + v / 2, str(v), ha="center", va="center", fontsize=8, color=FS.INK)
            left += v
    ax.set_xticks(x)
    ax.set_xticklabels([a + "\n" + T(f"성공 {agg[a]['all']['n_success']}/{agg[a]['all']['n_runs']}", f"success {agg[a]['all']['n_success']}/{agg[a]['all']['n_runs']}")
                        for a in ARMS], fontsize=8.5, color=FS.INK)
    ax.set_xlim(-0.6, len(ARMS) + 0.9)
    ax.set_title(T("실패 모드 (ok 제외)", "failure modes (ok excluded)"), fontsize=10, loc="left", color=FS.INK)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ramp[j]) for j in range(len(present))], labels=present, fontsize=7, frameon=False,
              loc="upper right", bbox_to_anchor=(1.02, 1.0))
    ax.grid(axis="y", color=FS.GRID_INK, linewidth=0.5); ax.set_axisbelow(True)
    # contacts: runs with non-authorised target / obstacle / neighbour contact
    ck = (("n_nonauth_target", T("비허용 link–target", "non-auth link-target")), ("n_obstacle_contact", T("장애물", "obstacle")),
          ("n_neighbour_contact", T("이웃 과일", "neighbour fruit")))
    w = 0.26
    for j, (k, lab) in enumerate(ck):
        vals = [agg[a]["all"].get(k, 0) for a in ARMS]
        bx.bar(x + (j - 1) * w, vals, width=w, color=FS.SEQ_BLUE[3 + 3 * j], edgecolor=FS.SURFACE, label=lab)
        for i, v in enumerate(vals):
            bx.text(x[i] + (j - 1) * w, v + 0.1, str(v), ha="center", va="bottom", fontsize=7.5, color=FS.INK_2)
    bx.set_xticks(x); bx.set_xticklabels(ARMS, fontsize=9, color=FS.INK)
    top = max([agg[a]["all"].get(k, 0) or 0 for a in ARMS for k, _ in ck] + [1])
    bx.set_ylim(0, top * 1.6 + 1)
    bx.set_title(T("접촉이 있던 run 수 (MuJoCo replay)", "runs with contact (MuJoCo replay)"), fontsize=10, loc="left", color=FS.INK)
    bx.legend(fontsize=7, frameon=False, loc="upper center", ncol=3, columnspacing=0.8, handlelength=1.2)
    bx.grid(axis="y", color=FS.GRID_INK, linewidth=0.5); bx.set_axisbelow(True)
    # HOLD chunks per run (strip + median)
    rng = np.random.default_rng(0)
    for i, a in enumerate(ARMS):
        hc = [(r["arms"].get(a) or {}).get("hold_chunks") or 0 for r in rows if (r["arms"].get(a) or {}).get("success") is not None]
        cx.scatter(x[i] + rng.uniform(-0.18, 0.18, len(hc)), hc, s=12, color=COL[a], alpha=0.7, linewidths=0)
        if hc:
            cx.plot([x[i] - 0.3, x[i] + 0.3], [np.median(hc)] * 2, color=FS.INK, linewidth=1.2)
        side["arms"].setdefault(a, {})["hold_chunks_per_run"] = hc
    cx.set_xticks(x); cx.set_xticklabels(ARMS, fontsize=9, color=FS.INK)
    cx.set_title(T("run 당 HOLD chunk 수 (선 = median)", "HOLD chunks per run (line = median)"), fontsize=10, loc="left", color=FS.INK)
    cx.grid(axis="y", color=FS.GRID_INK, linewidth=0.5); cx.set_axisbelow(True)
    # completion time of successful runs
    for i, a in enumerate(ARMS):
        cs = [(r["arms"].get(a) or {}).get("completion_s") for r in rows if (r["arms"].get(a) or {}).get("completion_s") is not None]
        dx.scatter(x[i] + rng.uniform(-0.18, 0.18, len(cs)), cs, s=12, color=COL[a], alpha=0.7, linewidths=0)
        if cs:
            dx.plot([x[i] - 0.3, x[i] + 0.3], [np.median(cs)] * 2, color=FS.INK, linewidth=1.2)
        side["arms"].setdefault(a, {})["completion_s_success_runs"] = cs
    dx.set_xticks(x); dx.set_xticklabels(ARMS, fontsize=9, color=FS.INK)
    dx.set_ylabel("s", color=FS.INK_2)
    dx.set_title(T("성공 run 의 완료 시간 (place_t / 15 Hz)", "completion time of successful runs (place_t / 15 Hz)"), fontsize=10, loc="left", color=FS.INK)
    dx.grid(axis="y", color=FS.GRID_INK, linewidth=0.5); dx.set_axisbelow(True)
    for a in ARMS:
        side["arms"][a] |= {k: agg[a]["all"].get(k) for k in ("n_runs", "n_success", "n_primary", "n_grasp_fail", "failure_modes", "n_nonauth_target",
                                                            "n_obstacle_contact", "n_neighbour_contact", "hold_chunks_sum", "runs_with_hold",
                                                            "hold_chunks_per_run", "completion_s")}
        side["arms"][a]["by_stratum"] = {s: {k: agg[a][s].get(k) for k in ("n_runs", "n_success", "n_primary", "n_grasp_fail")} for s in STRATA}
    fig.suptitle(ARM_LONG[ARMS[0]] + "   |   " + ARM_LONG[ARMS[1]], fontsize=8.5, color=FS.INK_2, x=0.01, ha="left", y=1.03)
    save(fig, out, f"{prefix}-arm-metrics", side)


# ----------------------------------------------------------------------------------------------------------------- 3. latency
def fig_latency(agg, out, prefix):
    lat = agg.get("latency")
    if not lat:
        print("no latency block; skipping", file=sys.stderr); return
    per = lat["per_arm"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.55})
    FS.style_axes(fig, (ax, bx))
    side = {"budget_ms": lat.get("budget_ms", BUDGET_MS), "per_arm": {}}
    x = np.arange(2); w = 0.36
    for i, a in enumerate(ARMS):
        p = per.get(a) or {}
        rt, st = p.get("client_round_trip_ms") or {}, p.get("server_total_ms") or {}
        med = [rt.get("median"), st.get("median")]; p95 = [rt.get("p95"), st.get("p95")]
        ax.bar(x + (i - 0.5) * w, [m or 0 for m in med], width=w, color=COL[a], edgecolor=FS.SURFACE, label=a)
        for j in range(2):
            if med[j] is not None and p95[j] is not None:
                ax.plot([x[j] + (i - 0.5) * w] * 2, [med[j], p95[j]], color=FS.INK, linewidth=1)
                ax.plot([x[j] + (i - 0.5) * w], [p95[j]], "_", color=FS.INK, markersize=9)
                ax.text(x[j] + (i - 0.5) * w, p95[j] + 40, f"{med[j]:.0f} / {p95[j]:.0f}", ha="center", fontsize=7, color=FS.INK_2)
        side["per_arm"][a] = {"n_runs": p.get("n_runs"), "n_requests": p.get("n_requests"), "client_round_trip_ms": rt, "server_total_ms": st}
    ax.axhline(BUDGET_MS, color=FS.CATEGORICAL[7], linewidth=1, linestyle="--")
    ax.text(-0.55, BUDGET_MS, T("예산 533 ms", "budget 533 ms"), fontsize=7.5, color=FS.CATEGORICAL[7], va="bottom", ha="left")
    ax.set_xlim(-0.6, 1.6)
    ymax = max([((per.get(a) or {}).get(k) or {}).get("p95") or 0 for a in ARMS for k in ("client_round_trip_ms", "server_total_ms")] + [BUDGET_MS])
    ax.set_ylim(0, ymax * 1.15)
    ax.set_xticks(x); ax.set_xticklabels([T("client 왕복", "client round trip"), T("서버 합계", "server total")], fontsize=9, color=FS.INK)
    ax.set_ylabel("ms", color=FS.INK_2); ax.legend(fontsize=8, frameon=False)
    ax.set_title(T("요청 지연: median (막대) 와 p95 (선 끝)", "request latency: median (bar) and p95 (whisker)"), fontsize=10, loc="left", color=FS.INK)
    ax.grid(axis="y", color=FS.GRID_INK, linewidth=0.5); ax.set_axisbelow(True)
    stages = sorted({k for a in ARMS for k, v in ((per.get(a) or {}).get("server_stage_ms") or {}).items() if v and v.get("median") is not None},
                    key=lambda k: -max(((per.get(a) or {}).get("server_stage_ms") or {}).get(k, {}).get("median") or 0 for a in ARMS))
    surf_extra = {}
    for a in ARMS:
        for grp in ("attention_surface_ms", "render_ms"):
            for k, v in ((per.get(a) or {}).get(grp) or {}).items():
                if v and v.get("median") is not None:
                    surf_extra.setdefault(k, {})[a] = v
    allk = stages + [k for k in surf_extra if k not in stages]
    y = np.arange(len(allk))[::-1]
    for i, a in enumerate(ARMS):
        vals = []
        for k in allk:
            v = ((per.get(a) or {}).get("server_stage_ms") or {}).get(k) or surf_extra.get(k, {}).get(a)
            vals.append((v or {}).get("median"))
        bx.barh(y + (0.5 - i) * 0.38, [v or 0 for v in vals], height=0.36, color=COL[a], edgecolor=FS.SURFACE, label=a)
        for yy, v in zip(y + (0.5 - i) * 0.38, vals):
            if v is not None:
                bx.text(v + 8, yy, f"{v:.0f}", va="center", fontsize=6.5, color=FS.INK_2)
        side["per_arm"][a]["stage_median_ms"] = dict(zip(allk, vals))
    bx.set_yticks(y); bx.set_yticklabels(allk, fontsize=7.5, color=FS.INK)
    bx.set_xlabel("ms (median)", color=FS.INK_2); bx.legend(fontsize=8, frameon=False, loc="lower right")
    bx.set_title(T("서버 단계별 계산 비용 (첫 요청 제외)", "server stage cost (first request of each run excluded)"), fontsize=10, loc="left", color=FS.INK)
    bx.grid(axis="x", color=FS.GRID_INK, linewidth=0.5); bx.set_axisbelow(True)
    side["surf_minus_c1an"] = lat.get("surf_minus_c1an")
    fig.suptitle(ARM_LONG[ARMS[0]] + "   |   " + ARM_LONG[ARMS[1]], fontsize=8.5, color=FS.INK_2, x=0.01, ha="left", y=1.03)
    save(fig, out, f"{prefix}-latency", side)


# ----------------------------------------------------------------------------------------------------------------- 4. outcome grid
def fig_grid(agg, out, prefix):
    rows = sorted(agg["rows"], key=lambda r: (STRATA.index(r["stratum"]) if r["stratum"] in STRATA else 9, r["episode"]))
    classes = ("primary", "contact", "no_grasp", "fail", "missing")
    cl = {"primary": T("접촉 없는 성공", "contact-free success"), "contact": T("성공, 접촉 있음", "success with contact"),
          "no_grasp": T("잡기 실패", "grasp failure"), "fail": T("그 외 실패", "other failure"), "missing": T("기록 없음", "missing")}
    cc = {"primary": COL["both"], "contact": FS.CATEGORICAL[3], "no_grasp": FS.CATEGORICAL[7], "fail": FS.CATEGORICAL[6], "missing": FS.GRID_INK}
    n = len(rows); ncol = 2
    per_col = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(1, ncol, figsize=(10, 0.26 * per_col + 1.4))
    FS.style_axes(fig, axes)
    side = {"rows": [], "classes": cl}
    for c, ax in enumerate(np.atleast_1d(axes)):
        chunk = rows[c * per_col:(c + 1) * per_col]
        for i, r in enumerate(chunk):
            yy = per_col - 1 - i
            for j, a in enumerate(ARMS):
                k = cell_outcome(r["arms"].get(a))
                ax.add_patch(plt.Rectangle((j, yy), 1, 1, color=cc[k], ec=FS.SURFACE, lw=1))
                fm = (r["arms"].get(a) or {}).get("failure_mode")
                if k not in ("primary", "missing") and fm:
                    ax.text(j + 0.5, yy + 0.5, str(fm), ha="center", va="center", fontsize=5.5, color=FS.SURFACE)
            ax.text(-0.1, yy + 0.5, f"ep{r['episode']}  {r['stratum']}", ha="right", va="center", fontsize=7, color=FS.INK)
            side["rows"].append({"episode": r["episode"], "seed": r["seed"], "stratum": r["stratum"], "target_fruit": r.get("target_fruit"),
                                 **{a: {"outcome": cell_outcome(r["arms"].get(a)), "failure_mode": (r["arms"].get(a) or {}).get("failure_mode"),
                                        "hold_chunks": (r["arms"].get(a) or {}).get("hold_chunks"), "completion_s": (r["arms"].get(a) or {}).get("completion_s")} for a in ARMS}})
        ax.set_xlim(-2.6, ncol + 0.1); ax.set_ylim(0, per_col); ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(ARMS, fontsize=9, color=FS.INK)
        ax.xaxis.tick_top(); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cc[k]) for k in classes], labels=[cl[k] for k in classes], fontsize=8, frameon=False,
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(T("episode × arm 결과 (주 지표 = 접촉 없는 작업 성공)", "episode x arm outcome (primary metric = contact-free task success)") + "\n" +
                 ARM_LONG[ARMS[0]] + "   |   " + ARM_LONG[ARMS[1]], fontsize=9, color=FS.INK, x=0.01, ha="left", y=1.0)
    save(fig, out, f"{prefix}-outcome-grid", side)


# ----------------------------------------------------------------------------------------------------------------- 5. discordant scenes
def video_frame(mp4, frac, tmp):
    """Extract one frame at fraction `frac` of the video duration with ffmpeg; returns an RGB array or None."""
    if not mp4.exists() or shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return None, None
    try:
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp4)], text=True).strip())
    except (subprocess.CalledProcessError, ValueError):
        return None, None
    png = tmp / (mp4.parent.name + f"_{frac:.3f}.png")
    if frac >= 0.99:  # final frame: seek from the end (an absolute -ss past the last frame yields nothing)
        t = dur
        cmd = ["ffmpeg", "-v", "error", "-y", "-sseof", "-0.3", "-i", str(mp4), "-update", "1", "-frames:v", "1", str(png)]
    else:
        t = max(0.0, min(dur - 0.05, frac * dur))
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(mp4), "-frames:v", "1", str(png)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not png.exists():
        return None, None
    return plt.imread(png), {"duration_s": dur, "t_s": t}


def fig_scenes(agg, out, prefix, runs_root, max_scenes):
    rows = [r for r in agg["rows"] if all((r["arms"].get(a) or {}).get("primary") is not None for a in ARMS)
            and (r["arms"][ARMS[0]]["primary"] != r["arms"][ARMS[1]]["primary"])]
    kind = "discordant"
    if not rows:  # nothing discordant: show the first pairs instead so the step still has a scene figure
        rows = [r for r in agg["rows"] if all((r["arms"].get(a) or {}).get("success") is not None for a in ARMS)]
        kind = "first_pairs"
    rows = sorted(rows, key=lambda r: (STRATA.index(r["stratum"]) if r["stratum"] in STRATA else 9, r["episode"]))[:max_scenes]
    if not rows:
        print("no complete pairs; skipping scenes", file=sys.stderr); return
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ifg_cl3_frames_"))
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.1 * len(rows)), squeeze=False)
    FS.style_axes(fig, axes)
    side = {"kind": kind, "frames": []}
    for i, r in enumerate(rows):
        # both arms at the SAME time: the grasp row of the arm that grasped (C1aN first), else the 35 % mark; plus the final frame
        gts = [(a, r["arms"][a].get("grasp_t")) for a in ARMS if r["arms"][a].get("grasp_t") is not None]
        if gts:
            g_arm, gt = gts[0]; frac_g = gt / 600.0
            lab_g = T(f"{g_arm} 가 잡은 시각 (row {gt})", f"grasp row of {g_arm} (row {gt})")
        else:
            gt, frac_g, lab_g = None, 0.35, T("35 % 지점 (grasp 없음)", "35 % mark (no grasp)")
        for j, a in enumerate(ARMS):
            arm = r["arms"][a]; rd = runs_root / a / arm["name"]
            mp4 = rd / "third_person_front.mp4"
            for k, (frac, lab) in enumerate(((frac_g, lab_g), (0.999, T("마지막 frame", "final frame")))):
                ax = axes[i, 2 * j + k]
                img, meta = video_frame(mp4, frac, tmp)
                if img is not None:
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, T("영상 없음", "no video"), ha="center", va="center", transform=ax.transAxes, color=FS.INK_2)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color(COL[a]); sp.set_linewidth(2)
                oc = cell_outcome(arm)
                ax.set_title(f"{a} · {lab}\n" + T("결과: ", "outcome: ") + oc + (f" ({arm.get('failure_mode')})" if arm.get("failure_mode") not in (None, "ok") else ""),
                             fontsize=8, color=FS.INK, loc="left")
                side["frames"].append({"episode": r["episode"], "stratum": r["stratum"], "arm": a, "run": str(rd), "which": lab, "frac": frac,
                                       "grasp_t": gt, "video": meta, "outcome": oc, "failure_mode": arm.get("failure_mode"), "hold_chunks": arm.get("hold_chunks")})
        axes[i, 0].text(-0.02, 0.5, f"ep{r['episode']} s{r['seed']}\n{r['stratum']}\n{r.get('target_fruit') or ''}", transform=axes[i, 0].transAxes,
                        ha="right", va="center", fontsize=8, color=FS.INK)
    fig.suptitle((T("한쪽 arm 만 접촉 없는 성공인 episode 쌍 (third-person front 영상)", "episode pairs where only one arm had a contact-free success (third-person front video)")
                  if kind == "discordant" else T("불일치 쌍 없음: 앞의 쌍들 (third-person front 영상)", "no discordant pair: first pairs (third-person front video)"))
                 + "\n" + ARM_LONG[ARMS[0]] + "   |   " + ARM_LONG[ARMS[1]], fontsize=9, color=FS.INK, x=0.01, ha="left", y=1.0)
    fig.subplots_adjust(top=1 - 0.75 / fig.get_figheight(), hspace=0.35)
    save(fig, out, f"{prefix}-discordant-scenes", side)
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agg", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--runs", type=pathlib.Path, default=None, help="runs/ directory (default: <agg dir>/runs)")
    ap.add_argument("--prefix", default="ifg-cl3")
    ap.add_argument("--max-scenes", type=int, default=4)
    ap.add_argument("--arms", nargs=2, default=["C1aN", "SURF"], help="reference arm first (agg keys only_<arm> / diff_<B>_minus_<A>)")
    ap.add_argument("--strata", nargs="+", default=["clear", "adjacent", "place"])
    ap.add_argument("--only", nargs="*", default=None, choices=["paired", "metrics", "latency", "grid", "scenes"])
    a = ap.parse_args()
    agg = json.loads(a.agg.read_text())
    set_arms(a.arms, a.strata)
    runs_root = a.runs or (a.agg.parent / "runs")
    # A re-measurement of the same experiment must pass a new --prefix: a figure is evidence linked from the log and is never overwritten.
    want = set(a.only or ["paired", "metrics", "latency", "grid", "scenes"])
    if "paired" in want:
        fig_paired(agg, a.out, a.prefix)
    if "metrics" in want:
        fig_arm_metrics(agg, a.out, a.prefix)
    if "latency" in want:
        fig_latency(agg, a.out, a.prefix)
    if "grid" in want:
        fig_grid(agg, a.out, a.prefix)
    if "scenes" in want:
        fig_scenes(agg, a.out, a.prefix, runs_root, a.max_scenes)


if __name__ == "__main__":
    main()
