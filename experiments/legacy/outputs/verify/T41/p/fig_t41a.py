"""T41a figures (verifier). usage: fig_t41a.py COMPARE_JSON T41A_ANALYSIS T40_REANALYSIS T40J_REANALYSIS
-> AG3S/ag3s/docs/figures/t41/t41a-scene-C.png, t41a-scene-D.png, t41a-distance.png, t41a-table.png (+ .json sidecars)

scene    : per hurdle episode (C = round-1 placement _r1, D = round-2 _r2), T40 (G/D record) | T41a third-person front frame (recorded
           third_person_front.mp4, one frame per control step) at the control row of the run's minimum gripper (EE_BODY_L + fingers)
           to hurdle distance (true MuJoCo geoms, t41a client distance code).
distance : gripper (solid) and apple (dotted) to hurdle distance (mm, capped 150) vs control step; T40 = CATEGORICAL slot 0,
           T41a = slot 1; x = last row of an obstacle-stopped run; grey lines at 0 and 10 mm (= --obstacle-margin).
table    : outcome per run, T40 vs T41a (T39 rules; contact = client MuJoCo contact + obstacle stop; mins over the whole run)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle

C = json.load(open(sys.argv[1]))["runs"]; A1 = json.load(open(sys.argv[2]))["runs"]; A0 = json.load(open(sys.argv[3]))["runs"]; AJ = json.load(open(sys.argv[4]))["runs"]
import os
OUT = pathlib.Path(os.environ.get("T41_FIGOUT", _ag3s_workspace_path('AG3S/ag3s/docs/figures/t41'))); OUT.mkdir(parents=True, exist_ok=True)
figstyle.use_korean()
COL = {"T40": figstyle.CATEGORICAL[0], "T41a": figstyle.CATEGORICAL[1], "T40J": figstyle.CATEGORICAL[2]}
WH = ("T40", "T40J", "T41a")
ORDER = [1807, 1968, 1982, 1828, 1995, 1967]


def save(fig, name, data):
    p = OUT / f"{name}.png"
    assert not p.exists() or "--overwrite" in sys.argv, f"{p} exists"
    fig.savefig(p, dpi=150, facecolor=figstyle.SURFACE); plt.close(fig)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, default=float)); print(p)


def runs_of(stage):
    ks = [k for k, v in C.items() if v["stage"] == stage]
    return sorted(ks, key=lambda k: ORDER.index(C[k]["episode"]) if C[k]["episode"] in ORDER else 99)


def raw(which, key):
    r = C[key]; st, ep, sd = r["stage"], r["episode"], r["seed"]
    k0 = f"E3b/{st}/E3b_ep{ep}_s{sd}"
    return A1.get(key) if which == "T41a" else (AJ.get(k0) if which == "T40J" else A0.get(k0))


def series(rr):
    s = rr["hurdle"]["series"]
    return np.asarray(s["t"]), np.asarray(s["A"]), np.asarray(s["B"])


def outcome_txt(f):
    if f.get("obstacle_stop"):
        c = ("contact" if (f.get("client_robot_collision") or f.get("client_payload_collision")) else "stop") + f" t={f['obstacle_stop_t']}"
    elif f.get("client_payload_collision") or f.get("client_robot_collision"):
        c = ("apple" if f.get("client_payload_collision") else "robot") + " contact (no stop)"
    else:
        c = ""
    return ("success" if f["success"] else "fail") + (f"  {c}" if c else "")


# ---- scene
for stage in ("C", "D"):
    ks = runs_of(stage)
    if not ks:
        continue
    fig, axes = plt.subplots(len(ks), 3, figsize=(13, 3.1 * len(ks)), squeeze=False)
    side = {"definitions": __doc__, "stage": stage, "episodes": {}}
    for row, k in zip(axes, ks):
        ep = C[k]["episode"]; side["episodes"][ep] = {}
        for ax, which in zip(row, WH):
            rr = raw(which, k); f = C[k].get(which)
            if rr is None or not rr.get("hurdle"):
                ax.set_axis_off(); ax.text(0.5, 0.5, f"{which}: no run" + (" (T40 J reran only false-zero stops)" if which == "T40J" else ""), ha="center", color=figstyle.INK_2, fontsize=8); continue
            t, a, b = series(rr); i = int(np.argmin(a))
            vid = pathlib.Path(rr["run_dir"]) / "third_person_front.mp4"
            rd = imageio.get_reader(str(vid)); n = rd.count_frames(); kf = min(i, n - 1); img = rd.get_data(kf); rd.close()
            Hh, Ww = img.shape[:2]; crop = (int(0.02 * Hh), int(0.80 * Hh), int(0.30 * Ww), int(0.98 * Ww))
            ax.imshow(img[crop[0]:crop[1], crop[2]:crop[3]]); ax.set_axis_off()
            fm = f["min_mm"]
            ax.set_title(f"{which} {stage} ep{ep}  t={int(t[i])}  gripper {1e3 * a[i]:.1f} mm\n"
                         f"fingers {fm['fingers']['all']['min_mm']}  EE {fm['EE_BODY_L']['all']['min_mm']} mm  {outcome_txt(f)}",
                         color=COL[which], fontsize=8.5)
            side["episodes"][ep][which] = {"run_dir": rr["run_dir"], "frame_index": kf, "n_frames": n, "t": int(t[i]),
                                           "gripper_min_m": float(a[i]), "fingers_min_mm": fm["fingers"]["all"]["min_mm"],
                                           "EE_BODY_L_min_mm": fm["EE_BODY_L"]["all"]["min_mm"], "success": f["success"], "crop_rows_cols": crop,
                                           "obstacle_stop_t": f["obstacle_stop_t"], "contact": f.get("client_robot_collision")}
    fig.suptitle(f"T40 | T40 J | T41a, hurdle {stage} ({'_r1' if stage == 'C' else '_r2'} placement): closest gripper approach "
                 "(third-person front, recorded runs)", color=figstyle.INK, fontsize=10)
    fig.tight_layout()
    save(fig, f"t41a-scene-{stage}", side)

# ---- distance graph
ks = runs_of("C") + runs_of("D")
nc = 5; nr = int(np.ceil(len(ks) / nc))
fig, axes = plt.subplots(nr, nc, figsize=(3.5 * nc, 3.1 * nr), sharey=True, squeeze=False)
figstyle.style_axes(fig, axes.ravel())
data = {"definitions": __doc__, "runs": {}}
for ax, k in zip(axes.ravel(), ks):
    r = C[k]; data["runs"][k] = {}
    for which in WH:
        rr = raw(which, k)
        if rr is None or not rr.get("hurdle"):
            continue
        t, a, b = series(rr)
        ax.plot(t, np.minimum(1e3 * a, 150), "-", color=COL[which], lw=1.3, label=f"{which} gripper")
        ax.plot(t, np.minimum(1e3 * b, 150), ":", color=COL[which], lw=1.0, label=f"{which} apple")
        stop = r[which]["obstacle_stop"]
        if stop:
            ax.plot(t[-1], min(1e3 * a[-1], 150), "x", color=COL[which], ms=8)
        data["runs"][k][which] = {"t": t.tolist(), "gripper_mm": (1e3 * a).tolist(), "apple_mm": (1e3 * b).tolist(), "obstacle_stop": stop,
                                  "success": r[which]["success"]}
    for y in (0, 10):
        ax.axhline(y, color=figstyle.INK_2, lw=0.7, ls="--" if y else "-")
    ax.set_title(f"{r['stage']} ep{r['episode']}", color=figstyle.INK, fontsize=10); ax.set_xlabel("control step")
for ax in axes.ravel()[len(ks):]:
    ax.set_axis_off()
for rr in axes[:, 0]:
    rr.set_ylabel("distance to hurdle (mm)")
hl = {}
for ax in axes.ravel():
    for h_, l_ in zip(*ax.get_legend_handles_labels()):
        hl.setdefault(l_, h_)
fig.legend(list(hl.values()), list(hl.keys()), loc="lower right", bbox_to_anchor=(0.985, 0.08), fontsize=8, frameon=False, ncol=2)
fig.suptitle("Distance to the hurdle over time, T40 / T40 J / T41a (gripper = EE_BODY_L + fingers; capped 150 mm; x = obstacle stop; dashed = 10 mm margin)",
             color=figstyle.INK, fontsize=10)
fig.tight_layout()
save(fig, "t41a-distance", data)

# ---- table
cols = ["stage", "ep", "run", "success", "grasp", "place", "HOLD (kinds)", "robot contact t / apple dist<=0 t (vs lift)", "fingers", "EE_BODY_L", "apple", "robot",
        "obst<10mm rows", "closest chunk finger: TO / mesh plan / mesh exec"]
rows = []; tdata = []; colors = []
for st in ("B", "C", "D"):
    for k in runs_of(st):
        r = C[k]
        for which in WH:
            f = r.get(which)
            if f is None:
                continue
            m = f["min_mm"]; g = lambda n: "-" if not m[n]["all"] else f"{m[n]['all']['min_mm']:.1f}"
            hk = ",".join(f"{a}:{b}" for a, b in f["hold_kind_counts"].items())
            ct = "-" if f["contact_t"] is None else f"{f['contact_t']} ({f['contact_vs_lift'].replace('_lift', '')})"
            if f["contact_t"] is None and f["obstacle_stop"]:
                ct = f"stop {f['obstacle_stop_t']} (no contact)"
            ap = (f.get("first_t_dist_le_0") or {}).get("apple")
            if ap and st != "B":
                ct = (ct if ct != "-" else "") + f"{'; ' if ct != '-' else ''}apple<=0 t={ap['t']} ({ap['vs_lift'].replace('_lift', '')})"
            ob = ca = "-"
            if which == "T41a" and r.get("T41a_TO_rows"):
                x = r["T41a_TO_rows"]; ob = str(sum(v.get("obstacle_lt10mm", 0) for v in x["by_phase"].values()))
                cc = x.get("closest_approach_chunk")
                if cc and st != "B":
                    tof = min(cc[l]["TO_view_on_min_mm"] for l in ("ee_finger_l1", "ee_finger_l2"))
                    mp = min(cc[l]["mesh_true_hurdle_min_mm_refined_plan"] for l in ("ee_finger_l1", "ee_finger_l2"))
                    me = min((cc[l]["mesh_true_hurdle_min_mm_executed_rows"] for l in ("ee_finger_l1", "ee_finger_l2")
                              if cc[l]["mesh_true_hurdle_min_mm_executed_rows"] is not None), default=None)
                    ca = f"{tof:.1f} / {mp:.1f} / {'-' if me is None else f'{me:.1f}'}"
            row = [st, r["episode"], which, "Y" if f["success"] else "N", "Y" if f["grasp"] else "N", "Y" if f["place"] else "N",
                   f"{f['n_hold_chunks']} ({hk})" if hk else str(f["n_hold_chunks"]), ct, g("fingers"), g("EE_BODY_L"), g("apple"), g("other_robot"), ob, ca]
            rows.append(row); tdata.append(dict(zip(cols, row))); colors.append(COL[which])
fig, ax = plt.subplots(figsize=(17, 0.26 * len(rows) + 1.0)); ax.set_axis_off(); fig.patch.set_facecolor(figstyle.SURFACE)
CW = [0.035, 0.035, 0.04, 0.045, 0.04, 0.04, 0.11, 0.15, 0.055, 0.065, 0.05, 0.05, 0.07, 0.17]
tb = ax.table(cellText=[[str(x) for x in r] for r in rows], colLabels=cols, cellLoc="center", bbox=[0, 0, 1, 1], colWidths=CW)
tb.auto_set_font_size(False); tb.set_fontsize(7.2)
for (i, j), cell in tb.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if i == 0:
        cell.set_text_props(color=figstyle.INK, weight="bold")
    elif j == 2:
        cell.set_text_props(color=colors[i - 1], weight="bold")
ax.set_title("T41a (--obstacle-margin 0.010) vs T40 G/D and T40 J (J = T40 code + client false-zero fix, reruns of the false-zero stops); all re-analysed with the same scripts.\nMin distance to hurdle, mm, "
             "whole run (B: hurdle placed virtually). obst rows<10mm = TO robot rows classed obstacle with field clearance < 10 mm (replay)",
             color=figstyle.INK, fontsize=8.5)
fig.tight_layout()
save(fig, "t41a-table", {"definitions": __doc__, "columns": cols, "rows": tdata})
