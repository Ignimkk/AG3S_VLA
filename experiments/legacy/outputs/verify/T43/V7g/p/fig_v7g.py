"""T43 V7g figures: t43v7g-hold-breakdown (graph), t43v7g-pairs-table (table), t43v7g-scene (real scene). Numbers from V7g/agg.json (+ server_chunks_*.json for the scene)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/Y/p'))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V7g')); FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
a = json.load(open(V / "agg.json")); C = figstyle.CATEGORICAL
kor = figstyle.use_korean()
ARMS = ["V5", "V5D", "V7g"]
for n in ("t43v7g-hold-breakdown", "t43v7g-pairs-table", "t43v7g-scene"):
    for ext in ("png", "json"):
        assert not (FIG / f"{n}.{ext}").exists(), n

# ---- 1. graph: HOLD breakdown
h = a["hold"]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.6)); figstyle.style_axes(fig, axes)
def stack(ax, key, order, title):
    bottom = np.zeros(3)
    for i, o in enumerate(order):
        v = np.array([h[x][key].get(o, 0) for x in ARMS], float)
        ax.bar(ARMS, v, bottom=bottom, color=C[i], label=o)
        for j, (b, y) in enumerate(zip(bottom, v)):
            if y > 0: ax.text(j, b + y / 2, int(y), ha="center", va="center", fontsize=8, color="white")
        bottom += v
    ax.set_title(title, fontsize=10, color=figstyle.INK); ax.legend(fontsize=7, frameon=False)
tot = [h[x]["total"] for x in ARMS]; trail = [h[x]["trailing"] for x in ARMS]
axes[0].bar(np.arange(3) - 0.2, tot, 0.4, color=C[0], label="HOLD chunks (total)"); axes[0].bar(np.arange(3) + 0.2, trail, 0.4, color=C[1], label="trailing HOLD chunks")
for j in range(3):
    axes[0].text(j - 0.2, tot[j] + 3, tot[j], ha="center", fontsize=9, color=figstyle.INK); axes[0].text(j + 0.2, trail[j] + 3, trail[j], ha="center", fontsize=9, color=figstyle.INK)
axes[0].set_xticks(range(3)); axes[0].set_xticklabels(ARMS); axes[0].legend(fontsize=8, frameon=False); axes[0].set_title("total / trailing HOLD chunks (19 runs)", fontsize=10, color=figstyle.INK)
stack(axes[1], "by_category", ["held", "base"], "binding row category")
stack(axes[2], "by_target", ["destination", "None"], "target label of the worst row")
stack(axes[3], "by_phase", ["pre_pick", "carry", "crate_above", "release_after"], "phase of the HOLD chunk")
fig.suptitle("T43 V7g — HOLD breakdown, V5 vs V5D vs V7g (cap 10 mm + slip detach + follow-TO 3 + subtask gate on)", fontsize=11, color=figstyle.INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = FIG / "t43v7g-hold-breakdown.png"; fig.savefig(out, dpi=120, facecolor=figstyle.SURFACE); plt.close(fig)
json.dump({"figure": str(out), "script": __file__, "source": str(V / "agg.json"),
           "hold": {x: {k: h[x][k] for k in ("total", "trailing", "runs_with_hold", "runs_with_trailing", "by_category", "by_target", "by_phase", "cells", "by_stage_total", "hold_kinds", "n_none_hurdle_le_50mm")} for x in ARMS}},
          open(FIG / "t43v7g-hold-breakdown.json", "w"), indent=1)

# ---- 2. table: pairs + counts
P = a["sgp"]["pairs"]; ar = a["sgp"]["arms"]
def cell(pair, st, f, A, B):
    x = P[pair][st][f]; return f"{x[A + '_total']} vs {x[B + '_total']}\n+{x[A + '_only']} / -{x[B + '_only']} / both {x['both']}\np {x['mcnemar_exact_p']:.2f}"
cols = ["metric", "V7g vs V5 (ALL 19)", "V7g vs V5D (ALL 19)", "V5D vs V5 (ALL 19)"]
rows = []
for lab, f in (("S old rule", "success"), ("G grasp", "grasp"), ("P place (old)", "place"), ("S new rule (release)", "success_rel"), ("P new rule", "place_rel"), ("released", "released")):
    rows.append([lab, cell("V7g_vs_V5", "ALL", f, "V7g", "V5"), cell("V7g_vs_V5D", "ALL", f, "V7g", "V5D"), cell("V5D_vs_V5", "ALL", f, "V5D", "V5")])
fig, axes = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [3.2, 2.2]}); figstyle.style_axes(fig, axes)
for ax in axes: ax.axis("off")
t = axes[0].table(cellText=rows, colLabels=cols, loc="center", cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 3.0)
axes[0].set_title("T43 V7g — paired 2x2 (A total vs B total; +A_only / -B_only / both; McNemar exact p), 14 hurdle + 5 no-obstacle", fontsize=10, color=figstyle.INK)
ft = a["follow_to"]; hh = a["hold"]; ra = a["attach_radii"]; sl = a["slip"]; pc = a["place_closed"]
rows2 = [["HOLD total / trailing", *[f"{hh[x]['total']} / {hh[x]['trailing']}" for x in ARMS]],
         ["runs with HOLD", *[str(hh[x]["runs_with_hold"]) for x in ARMS]],
         ["attach: body query max radius (mm) min / median / max", *[f"{ra[x]['body_parent_max_radius_mm']['min']:.1f} / {ra[x]['body_parent_max_radius_mm']['median']:.1f} / {ra[x]['body_parent_max_radius_mm']['max']:.1f}" for x in ARMS]],
         ["slip detach (runs with detach / slip flag)", *[f"{sl[x]['n_detach']} / {sl[x]['n_slip_flag_true']}" for x in ARMS]],
         ["released: closed-cmd / open-cmd (of 16)", *[f"{pc[x]['release_mode'].get('closed_cmd', 0)} / {pc[x]['release_mode'].get('open_cmd', 0)}" for x in ARMS]],
         ["release apple-floor gap, median (mm)", *[f"{pc[x]['release_apple_floor_gap_mm']['median']:.1f}" for x in ARMS]],
         ["follow_to executed / refused (V7g)", "-", "-", f"{ft['n_executed']} / {sum(ft['refused_counts'].values())}  {ft['refused_counts']}"]]
t2 = axes[1].table(cellText=rows2, colLabels=["", *ARMS], loc="center", cellLoc="center", colWidths=[0.42, 0.18, 0.18, 0.3]); t2.auto_set_font_size(False); t2.set_fontsize(9); t2.scale(1, 2.2)
fig.tight_layout()
out = FIG / "t43v7g-pairs-table.png"; fig.savefig(out, dpi=120, facecolor=figstyle.SURFACE); plt.close(fig)
json.dump({"figure": str(out), "script": __file__, "source": str(V / "agg.json"), "rows_pairs": rows, "cols_pairs": cols, "rows_summary": rows2,
           "arms_counts": ar, "pairs": P}, open(FIG / "t43v7g-pairs-table.json", "w"), indent=1, default=float)

# ---- 3. real scene: V5 (r 58 / 56 mm) vs V7g (cap 10 mm) held query spheres, ep1925 s19253 at t = 216 (V5 HOLD chunk)
import render_y as RY  # noqa: E402
S5 = json.load(open(V / "server_chunks_V5.json"))["runs"]["V5/H/V5_ep1925_s19253"]; S7 = json.load(open(V / "server_chunks_V7g.json"))["runs"]["V7g/H/V7g_ep1925_s19253"]
A5 = json.load(open(_ag3s_workspace_path('outputs/verify/T43/W/analysis/analysis.json')))["runs"]["V5/H/V5_ep1925_s19253"]; A7 = json.load(open(V / "analysis/analysis.json"))["runs"]["V7g/H/V7g_ep1925_s19253"]
CLOSE = dict(lookat=(0.50, -0.02, 0.93), distance=0.75, azimuth=90.0, elevation=-6.0); SPH = [(0.92, 0.41, 0.20, 0.35), (0.93, 0.63, 0.0, 0.35)]
T0 = 216; crop = (slice(60, 330), slice(150, 520))
fig, axes = plt.subplots(2, 3, figsize=(15, 7.6)); figstyle.style_axes(fig, axes); side = {}
for r, (key, an, S, lab) in enumerate((("V5", A5, S5, "V5 (no cap)"), ("V7g", A7, S7, "V7g (cap 10 mm)"))):
    rd = an["run_dir"]; M, ctrl = RY.scene_for(rd, "left")
    c = next(x for x in S["chunks"] if x["t"] == T0); hq = c.get("held_query") or {}
    cen = hq.get("centre_world_now") or []; sph = [(p, 1e-3 * rr, SPH[i % 2]) for i, (p, rr) in enumerate(zip(cen, hq.get("radii_mm") or []))]
    q = np.array(ctrl[T0]["qpos"], float)
    vid = RY.video_frame(rd, T0); full = RY.render(M, q, RY.REC, sph); close = RY.render(M, q, CLOSE, sph)
    axes[r, 0].imshow(vid[crop]); axes[r, 1].imshow(full[crop]); axes[r, 2].imshow(close)
    axes[r, 0].set_title(f"{lab}: third_person_front.mp4 frame {T0} (crop)", fontsize=9, color=figstyle.INK)
    axes[r, 1].set_title("MuJoCo re-render + held query spheres (body cover)", fontsize=9, color=figstyle.INK); axes[r, 2].set_title("close-up side view", fontsize=9, color=figstyle.INK)
    for ax in axes[r]: ax.set_xticks([]); ax.set_yticks([])
    sd = hq.get("crate_sd_mm_now")
    axes[r, 2].text(1.02, 0.98, f"{key} ep1925 s19253 t {T0}\nheld query radii: {', '.join('%.1f' % x for x in hq.get('radii_mm') or [])} mm\ngate: {'HOLD' if c['hold'] else 'execute'}\n"
                    f"apple centre in crate frame [{', '.join('%.0f' % v for v in c['apple_in_crate_mm'])}] mm\ntrue crate signed distance per sphere: {', '.join('%.1f' % x for x in sd or [])} mm",
                    transform=axes[r, 2].transAxes, fontsize=8, color=figstyle.INK_2, va="top")
    side[key] = {"t": T0, "radii_mm": hq.get("radii_mm"), "hold": c["hold"], "crate_sd_mm_now": sd, "apple_in_crate_mm": c["apple_in_crate_mm"], "spheres_world": [list(map(float, p)) for p, _, _ in sph]}
fig.subplots_adjust(left=0.01, right=0.80, top=0.90, bottom=0.01, hspace=0.18, wspace=0.03)
fig.suptitle("T43 V7g — ep1925 s19253 at t 216: held query spheres, V5 vs V7g (no follow_to chunk was executed in V7g, so the crate-entry moment is shown)", fontsize=10, color=figstyle.INK, y=0.985)
out = FIG / "t43v7g-scene.png"; fig.savefig(out, dpi=110, facecolor=figstyle.SURFACE); plt.close(fig)
json.dump({"figure": str(out), "script": __file__, "panels": side, "close_camera": CLOSE, "video_camera": RY.REC, "sources": [str(V / "server_chunks_V5.json"), str(V / "server_chunks_V7g.json")]},
          open(FIG / "t43v7g-scene.json", "w"), indent=1, default=float)
print("ok")
