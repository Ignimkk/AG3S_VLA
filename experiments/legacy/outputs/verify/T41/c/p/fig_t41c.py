"""T41c figures (verifier). usage: fig_t41c.py COMPARE_C_JSON [--overwrite]
-> AG3S/ag3s/docs/figures/t41/t41c-scene-C.png, t41c-scene-D.png, t41c-scene-C1982.png, t41c-distance.png, t41c-hold.png,
   t41c-table.png (+ .json sidecars)

scene      : per hurdle episode, T41a | T41c third-person front frame (recorded third_person_front.mp4, one frame per control step) at
             the control row of the run's minimum gripper (EE_BODY_L + fingers) to hurdle distance (true MuJoCo geoms, t41a client
             distance code). T41c recordings carry the avoidance-arrow overlay (--viz-avoidance-arrow; green = execute
             refined - reference, red = HOLD), T41a recordings have none.
scene C1982: T40 | T41a | T41c, frame at the first row of each HOLD episode's last row (+1 = pose after the HOLD) or the contact row.
distance   : gripper (solid) to hurdle distance (mm, capped 150) vs control step; T40 = CATEGORICAL slot 0, T41a = slot 1,
             T41c = slot 3; shaded bands = T41c HOLD rows; x = last row of an obstacle-stopped run; lines at 0 and 10 mm.
hold       : per HOLD episode (hold_c.py physics replay): left-arm joint max |q - q_entry| (deg), TCP max displacement (mm),
             fruit-in-TCP slip max (mm; filled marker = apple in hand at entry), vs HOLD duration (control steps); T40 / T41a = mode fixed,
             T41c = mode measured.
table      : outcome per run, T40 / T40 J / T41a / T41c."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle

V = pathlib.Path(_ag3s_workspace_path('outputs/verify'))
C = json.load(open(sys.argv[1]))["runs"]
A = {"T41c": V / "T41/c/analysis/analysis_c.json", "T41a": V / "T41/a/analysis/analysis_a.json",
     "T40": V / "T41/a/analysis/t40g_reanalysis.json", "T40J": V / "T41/a/analysis/t40j_reanalysis.json"}
A = {k: json.load(open(p))["runs"] for k, p in A.items()}
OUT = pathlib.Path(os.environ.get("T41_FIGOUT", _ag3s_workspace_path('AG3S/ag3s/docs/figures/t41'))); OUT.mkdir(parents=True, exist_ok=True)
figstyle.use_korean()
COL = {"T40": figstyle.CATEGORICAL[0], "T41a": figstyle.CATEGORICAL[1], "T40J": figstyle.CATEGORICAL[2], "T41c": figstyle.CATEGORICAL[3]}
ORDER = [1807, 1968, 1982, 1828, 1995, 1967]


def save(fig, name, data):
    p = OUT / f"{name}.png"
    assert not p.exists() or "--overwrite" in sys.argv, f"{p} exists"
    fig.savefig(p, dpi=150, facecolor=figstyle.SURFACE); plt.close(fig)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, default=float)); print(p)


def runs_of(stage):
    ks = [k for k, v in C.items() if v["stage"] == stage]
    return sorted(ks, key=lambda k: ORDER.index(C[k]["episode"]) if C[k]["episode"] in ORDER else 99)


def series(rr):
    s = rr["hurdle"]["series"]
    return np.asarray(s["t"]), np.asarray(s["A"]), np.asarray(s["B"])


def frame(rr, t_row):
    vid = pathlib.Path(rr["run_dir"]) / "third_person_front.mp4"
    rd = imageio.get_reader(str(vid)); n = rd.count_frames(); kf = min(int(t_row), n - 1); img = rd.get_data(kf); rd.close()
    Hh, Ww = img.shape[:2]; crop = (int(0.02 * Hh), int(0.80 * Hh), int(0.30 * Ww), int(0.98 * Ww))
    return img[crop[0]:crop[1], crop[2]:crop[3]], kf, n, crop


def outcome_txt(f):
    ev = f.get("contact_events") or []
    c = "; ".join(f"{'robot' if e['robot_substeps'] else 'apple'} contact t={e['t_first']} {e['phase']}{' in HOLD' if e['in_hold_first_row'] else ''}" for e in ev)
    return ("success" if f["success"] else "fail") + (f"  {c}" if c else "") + (f"  stop t={f['obstacle_stop_t']}" if f["obstacle_stop"] else "")


# ---- scene: T41a | T41c closest approach
for stage in ("C", "D"):
    ks = runs_of(stage)
    if not ks:
        continue
    fig, axes = plt.subplots(len(ks), 2, figsize=(10, 3.3 * len(ks)), squeeze=False)
    side = {"definitions": __doc__, "stage": stage, "episodes": {}}
    for row, k in zip(axes, ks):
        ep = C[k]["episode"]; side["episodes"][ep] = {}
        for ax, which in zip(row, ("T41a", "T41c")):
            rr = A[which].get(k); f = C[k].get(which)
            if rr is None or f is None:
                ax.set_axis_off(); ax.text(0.5, 0.5, f"{which}: no run", ha="center", color=figstyle.INK_2); continue
            t, a, b = series(rr); i = int(np.argmin(a))
            img, kf, n, crop = frame(rr, i)
            ax.imshow(img); ax.set_axis_off(); fm = f["min_mm"]
            ax.set_title(f"{which} {stage} ep{ep}  t={int(t[i])}  gripper {1e3 * a[i]:.1f} mm  (fingers {fm['fingers']['all']['min_mm']}, EE {fm['EE_BODY_L']['all']['min_mm']})\n"
                         f"{outcome_txt(f)}", color=COL[which], fontsize=8)
            side["episodes"][ep][which] = {"run_dir": rr["run_dir"], "frame_index": kf, "n_frames": n, "t": int(t[i]), "gripper_min_m": float(a[i]),
                                           "fingers_min_mm": fm["fingers"]["all"]["min_mm"], "EE_BODY_L_min_mm": fm["EE_BODY_L"]["all"]["min_mm"],
                                           "success": f["success"], "crop_rows_cols": crop, "contact_events": f.get("contact_events")}
    fig.suptitle(f"T41a (HOLD fixed) | T41c (HOLD measured, arrow overlay), hurdle {stage} ({'_r1' if stage == 'C' else '_r2'}): closest gripper approach",
                 color=figstyle.INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, f"t41c-scene-{stage}", side)

# ---- scene C1982: HOLD frames, T40 | T41a | T41c
k = "E3b/C/E3b_ep1982_s19822"
if k in C:
    cols = [w for w in ("T40", "T41a", "T41c") if w in C[k]]
    picks = {}
    for w in cols:
        f = C[k][w]; hs = f.get("holds") or []
        p = []
        for h in hs:
            p.append(("HOLD entry", h["t_first"])); p.append(("HOLD last row", h["t_last"]))
        for e in f.get("contact_events") or []:
            p.append(("contact", e["t_first"]))
        t, a, b = series(A[w][k]); p.append(("closest approach", int(t[int(np.argmin(a))])))
        seen = set(); picks[w] = [x for x in sorted(p, key=lambda x: x[1]) if not (x[1] in seen or seen.add(x[1]))][:4]
    nr = max(len(v) for v in picks.values())
    fig, axes = plt.subplots(nr, len(cols), figsize=(4.6 * len(cols), 3.3 * nr), squeeze=False)
    side = {"definitions": __doc__, "run": k, "frames": {}}
    for j, w in enumerate(cols):
        rr = A[w][k]; t, a, b = series(rr); side["frames"][w] = []
        for i in range(nr):
            ax = axes[i, j]; ax.set_axis_off()
            if i >= len(picks[w]):
                continue
            lab, tt = picks[w][i]; ii = int(np.searchsorted(t, tt)); ii = min(ii, len(t) - 1)
            img, kf, n, crop = frame(rr, ii); ax.imshow(img)
            hold = next((h for h in C[k][w].get("holds") or [] if h["t_first"] <= tt <= h["t_last"]), None)
            extra = (f"\nHOLD {hold['mode']} {hold['duration_steps']} steps: joint {hold['joint_max_deg']:.2f} deg, TCP {hold['tcp_max_mm']:.1f} mm, "
                     f"contact substeps {hold['contacts_in_hold']['robot_substeps']}") if hold else ""
            ax.set_title(f"{w}  {lab}  t={tt}  gripper {1e3 * a[ii]:.1f} mm{extra}", color=COL[w], fontsize=7.5)
            side["frames"][w].append({"label": lab, "t": tt, "frame_index": kf, "gripper_mm": 1e3 * float(a[ii]), "hold": hold})
    fig.suptitle("C ep1982: T40 (HOLD fixed, contact inside a HOLD) | T41a (fixed) | T41c (measured)", color=figstyle.INK, fontsize=10)
    fig.tight_layout()
    save(fig, "t41c-scene-C1982", side)

# ---- distance graph
ks = runs_of("C") + runs_of("D")
nc = 5; nr = int(np.ceil(len(ks) / nc))
fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 3.2 * nr), sharey=True, squeeze=False)
figstyle.style_axes(fig, axes.ravel())
data = {"definitions": __doc__, "runs": {}}
for ax, k in zip(axes.ravel(), ks):
    r = C[k]; data["runs"][k] = {}
    for which in ("T40", "T41a", "T41c"):
        rr = A[which].get(k)
        if rr is None or which not in r:
            continue
        t, a, b = series(rr)
        ax.plot(t, np.minimum(1e3 * a, 150), "-", color=COL[which], lw=1.3, label=which)
        if r[which]["obstacle_stop"]:
            ax.plot(t[-1], min(1e3 * a[-1], 150), "x", color=COL[which], ms=8)
        for h in r[which].get("holds") or []:
            if which == "T41c":
                ax.axvspan(h["t_first"], h["t_last"] + 1, color=COL[which], alpha=0.18, lw=0)
            ax.plot([h["t_first"], h["t_last"] + 1], [-8 - 4 * ("T40", "T41a", "T41c").index(which)] * 2, "-", color=COL[which], lw=3)
        data["runs"][k][which] = {"t": t.tolist(), "gripper_mm": (1e3 * a).tolist(), "obstacle_stop": r[which]["obstacle_stop"],
                                  "holds": [(h["t_first"], h["t_last"]) for h in r[which].get("holds") or []], "success": r[which]["success"]}
    for y in (0, 10):
        ax.axhline(y, color=figstyle.INK_2, lw=0.7, ls="--" if y else "-")
    ax.set_ylim(-22, 155)
    ax.set_title(f"{r['stage']} ep{r['episode']}", color=figstyle.INK, fontsize=10); ax.set_xlabel("control step")
for ax in axes.ravel()[len(ks):]:
    ax.set_axis_off()
for rr in axes[:, 0]:
    rr.set_ylabel("gripper to hurdle (mm)")
hl = {}
for ax in axes.ravel():
    for h_, l_ in zip(*ax.get_legend_handles_labels()):
        hl.setdefault(l_, h_)
fig.legend(list(hl.values()), list(hl.keys()), loc="lower right", bbox_to_anchor=(0.985, 0.08), fontsize=9, frameon=False)
fig.suptitle("Gripper (EE_BODY_L + fingers) to hurdle over time, T40 / T41a / T41c (capped 150 mm; x = obstacle stop; bars below 0 = HOLD rows per run; "
             "shaded = T41c HOLD; dashed = 10 mm margin)", color=figstyle.INK, fontsize=9.5)
fig.tight_layout()
save(fig, "t41c-distance", data)

# ---- HOLD graph
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3)); figstyle.style_axes(fig, axes)
hd = {"definitions": __doc__, "holds": []}
for which in ("T40", "T41a", "T41c"):
    for k, r in C.items():
        f = r.get(which)
        for h in (f or {}).get("holds") or []:
            hd["holds"].append({"set": which, "run": k, **{x: h[x] for x in ("t_first", "duration_steps", "kinds", "mode", "phase", "joint_max_deg", "tcp_max_mm",
                                                                             "q_hold_vs_entry_measured_deg", "dev_from_q_hold_max_deg")},
                                "slip_max_mm": h["fruit"]["slip_max_mm"], "in_hand": h["fruit"]["in_hand_at_entry"],
                                "robot_contact_substeps": h["contacts_in_hold"]["robot_substeps"], "payload_contact_substeps": h["contacts_in_hold"]["payload_substeps"]})
for ax, (key, lab) in zip(axes, (("joint_max_deg", "left-arm joint max |q - q_entry| (deg)"), ("tcp_max_mm", "TCP max displacement (mm)"),
                                 ("slip_max_mm", "apple-in-TCP-frame max shift (mm)\n(open marker: apple not in hand, shift = TCP motion)"))):
    for j, which in enumerate(("T40", "T41a", "T41c")):
        hh = [h for h in hd["holds"] if h["set"] == which]
        if not hh:
            continue
        x = np.array([h["duration_steps"] for h in hh]) * (1 + 0.04 * (j - 1)); y = np.array([h[key] for h in hh])
        inh = np.array([h["in_hand"] for h in hh]); cont = np.array([h["robot_contact_substeps"] > 0 for h in hh])
        ax.scatter(x[~inh], y[~inh], s=34, facecolors="none", edgecolors=COL[which], lw=1.3, label=f"{which} ({hh[0]['mode']})")
        ax.scatter(x[inh], y[inh], s=34, color=COL[which], label=f"{which} apple in hand")
        if cont.any():
            ax.scatter(x[cont], y[cont], s=140, marker="x", color=COL[which], lw=1.6, label=f"{which} robot contact in HOLD")
    ax.set_xscale("log"); ax.set_xlabel("HOLD duration (control steps, log)"); ax.set_ylabel(lab)
axes[0].legend(fontsize=7, frameon=False)
fig.suptitle("Motion during each HOLD (physics replay of the recorded runs; window = entry pose .. pose after the last HOLD step). All stages B/C/D.",
             color=figstyle.INK, fontsize=10)
fig.tight_layout()
save(fig, "t41c-hold", hd)

# ---- table
cols = ["stage", "ep", "run", "S/G/P", "HOLD episodes: t x steps (kinds) joint deg / TCP mm / slip mm", "contacts: t phase (R robot, P apple, h in HOLD)",
        "fingers", "EE_BODY_L", "apple", "robot", "verdict reason kinds"]
rows = []; tdata = []; colors = []
for st in ("B", "C", "D"):
    for k in runs_of(st):
        r = C[k]
        for which in ("T40", "T40J", "T41a", "T41c"):
            f = r.get(which)
            if f is None:
                continue
            m = f["min_mm"]; g = lambda n: "-" if not m[n]["all"] else f"{m[n]['all']['min_mm']:.1f}"
            hs = "; ".join(f"{h['t_first']}x{h['duration_steps']} ({','.join(x[:5] for x in h['kinds'])}) {h['joint_max_deg']:.2f}/{h['tcp_max_mm']:.1f}/{h['fruit']['slip_max_mm']:.1f}"
                           f"{'H' if h['fruit']['in_hand_at_entry'] else ''}{' C' if h['contacts_in_hold']['robot_substeps'] else ''}" for h in (f.get("holds") or [])[:3])
            if len(f.get("holds") or []) > 3:
                hs += f"; +{len(f['holds']) - 3}"
            ce = "; ".join(f"{e['t_first']} {e['phase'].replace('before_lift', 'pre')} {'R' if e['robot_substeps'] else ''}{'P' if e['payload_substeps'] else ''}{'h' if e['in_hold_first_row'] else ''}"
                           for e in f.get("contact_events") or []) or "-"
            vk = ",".join(f"{a[:6]}:{b}" for a, b in f["verdict_reason_kind_counts"].items() if a != "budget_only")
            row = [st, r["episode"], which, ("S" if f["success"] else "-") + ("G" if f["grasp"] else "-") + ("P" if f["place"] else "-"),
                   hs or "-", ce, g("fingers"), g("EE_BODY_L"), g("apple"), g("other_robot"), vk or "-"]
            rows.append(row); tdata.append(dict(zip(cols, row))); colors.append(COL[which])
fig, ax = plt.subplots(figsize=(19, 0.27 * len(rows) + 1.2)); ax.set_axis_off(); fig.patch.set_facecolor(figstyle.SURFACE)
CW = [0.03, 0.035, 0.035, 0.035, 0.33, 0.14, 0.045, 0.055, 0.045, 0.045, 0.16]
tb = ax.table(cellText=[[str(x) for x in r] for r in rows], colLabels=cols, cellLoc="center", bbox=[0, 0, 1, 1], colWidths=CW)
tb.auto_set_font_size(False); tb.set_fontsize(6.8)
for (i, j), cell in tb.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if i == 0:
        cell.set_text_props(color=figstyle.INK, weight="bold")
    elif j == 2:
        cell.set_text_props(color=colors[i - 1], weight="bold")
ax.set_title("T41c (--obstacle-margin 0.010 + --safe-hold-mode measured) vs T41a (margin, HOLD fixed) vs T40 G/D and T40 J (no margin, fixed). Same analysis scripts.\n"
             "Min distance to hurdle, mm, whole run (B: hurdle placed virtually). HOLD: H = apple in hand at entry, C = robot contact inside the HOLD. Contacts from physics replay.",
             color=figstyle.INK, fontsize=8.5)
fig.tight_layout()
save(fig, "t41c-table", {"definitions": __doc__, "columns": cols, "rows": tdata})
