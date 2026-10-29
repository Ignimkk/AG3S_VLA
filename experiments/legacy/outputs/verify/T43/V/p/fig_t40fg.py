"""T40 F/G figures (verifier). usage: fig_t40fg.py ANALYSIS_JSON [ROUND_TAG]
-> AG3S/ag3s/docs/figures/t40/t40fg-scene-<EP>[ROUND].png, t40fg-distance[ROUND].png, t40fg-table[ROUND].png (+ .json sidecars)

scene    : per episode with a hurdle run, C-stage E0 | E3b third-person front frames (recorded third_person_front.mp4, one frame per
           control step) at the control row of the minimum hurdle distance min(gripper, apple, other robot) of that run.
distance : per episode, min(gripper, apple) hurdle distance (mm, client distance code, capped 300) vs control step; E0 / E3b
           (CATEGORICAL slots 0 / 1); solid = C (physical hurdle), dashed = B (same hurdle placed virtually, no physics);
           x marks the last row of a run that ended with an obstacle stop.
table    : outcome per episode x stage x condition (T39 analysis rules); 'apple leaves hand t' = hurdle_metrics apple_left_hand_t
           (in a successful run this is the release in the crate)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle

A = json.loads(pathlib.Path(sys.argv[1]).read_text()); RT = sys.argv[2] if len(sys.argv) > 2 else ""
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t40')); OUT.mkdir(parents=True, exist_ok=True)
KO = figstyle.use_korean()
COL = {"E0": figstyle.CATEGORICAL[0], "E3b": figstyle.CATEGORICAL[1]}
runs = A["runs"]
eps = sorted({r["episode"] for r in runs.values()}, key=lambda e: [1807, 1968, 1982, 1828, 1995, 1967].index(e) if e in [1807, 1968, 1982, 1828, 1995, 1967] else 99)
stages = [s for s in ("B", "C", "D") if any(r["stage"] == s for r in runs.values())]
hstage = "D" if RT == "-d" else "C"
refst = "C" if hstage == "D" else "B"   # dashed reference: B (virtual r1 hurdle) for round 1, C (round-1 hurdle) for D


def get(cond, stage, ep):
    return next((r for r in runs.values() if r["cond"] == cond and r["stage"] == stage and r["episode"] == ep), None)


def save(fig, name, data):
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=150, facecolor=figstyle.SURFACE); plt.close(fig)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, default=float)); print(p)


def mins(r):
    s = r["hurdle"]["series"]
    ab = np.minimum(np.asarray(s["A"]), np.asarray(s["B"]))
    allm = np.minimum(ab, np.asarray(s["C"]))
    return np.asarray(s["t"]), ab, allm


# ---- scene
for ep in eps:
    r0, r3 = get("E0", hstage, ep), get("E3b", hstage, ep)
    if not (r0 and r3 and r0.get("hurdle") and r3.get("hurdle")):
        continue
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    side = {}
    for ax, (cond, r) in zip(axes, (("E0", r0), ("E3b", r3))):
        t, ab, allm = mins(r); i = int(np.argmin(allm))
        vid = pathlib.Path(r["run_dir"]) / "third_person_front.mp4"
        rd = imageio.get_reader(str(vid)); n = rd.count_frames(); k = min(i, n - 1); img = rd.get_data(k); rd.close()
        ax.imshow(img); ax.set_axis_off()
        h = r["hurdle"]; stop = bool(h.get("client_obstacle_stop_lines"))
        ax.set_title(f"{cond}  t={int(t[i])}  min dist {1e3 * allm[i]:.1f} mm  {'success' if r['success'] else 'fail'}"
                     f"{'  obstacle stop' if stop else ''}", color=figstyle.INK, fontsize=10)
        side[cond] = {"run_dir": r["run_dir"], "frame_index": k, "n_frames": n, "t": int(t[i]), "min_dist_m": float(allm[i]),
                      "min_gripper_or_apple_m_at_t": float(ab[i]), "success": r["success"], "obstacle_stop": stop,
                      "hurdle": h["hurdle"]}
    fig.suptitle(f"T40 {hstage} ep{ep}: closest approach to the hurdle (third-person front, recorded run)", color=figstyle.INK, fontsize=11)
    fig.tight_layout()
    save(fig, f"t40fg-scene-{ep}{RT}", {"definitions": __doc__, "episode": ep, "stage": hstage, **side})

# ---- distance graph
E = [ep for ep in eps if any((get(c, s, ep) or {}).get("hurdle") for c in ("E0", "E3b") for s in stages)]
fig, axes = plt.subplots(1, len(E), figsize=(3.6 * len(E), 3.6), sharey=True, squeeze=False); axes = axes[0]
figstyle.style_axes(fig, axes)
data = {"definitions": __doc__, "episodes": {}}
for ax, ep in zip(axes, E):
    data["episodes"][ep] = {}
    for st, ls in ((refst, "--"), (hstage, "-")):
        for cond in ("E0", "E3b"):
            r = get(cond, st, ep)
            if not (r and r.get("hurdle")):
                continue
            t, ab, _ = mins(r); y = np.minimum(1e3 * ab, 300)
            ax.plot(t, y, ls, color=COL[cond], lw=1.3 if ls == "-" else 1.0, label=f"{cond} {st}")
            stop = bool(r["hurdle"].get("client_obstacle_stop_lines"))
            if stop:
                ax.plot(t[-1], y[-1], "x", color=COL[cond], ms=8)
            data["episodes"][ep][f"{cond}_{st}"] = {"t": t.tolist(), "min_gripper_or_apple_mm": (1e3 * ab).tolist(), "obstacle_stop": stop,
                                                   "success": r["success"]}
    ax.axhline(0, color=figstyle.INK_2, lw=0.8)
    ax.set_title(f"ep{ep}", color=figstyle.INK, fontsize=10); ax.set_xlabel("control step")
axes[0].set_ylabel("hurdle distance, gripper or apple (mm)")
axes[-1].legend(fontsize=8, frameon=False)
fig.suptitle(f"T40: distance to the hurdle over time (solid {hstage}, dashed {refst}; B = no hurdle, round-1 pose placed virtually)", color=figstyle.INK, fontsize=10)
fig.tight_layout()
save(fig, f"t40fg-distance{RT}", data)

# ---- table (one table for all stages; written only for the round-1 call)
if RT:
    sys.exit(0)
cols = ["ep", "stage", "cond", "success", "grasp", "place", "HOLD", "HOLD kinds", "min grip mm", "min apple mm", "min robot mm",
        "contact / stop", "apple off hand t", "AG3S tgt on hurdle", "AG3S tgt->hurdle mm", "bit-id T39"]
rows = []; tdata = []
for ep in eps:
    for st in stages:
        for cond in ("E0", "E3b"):
            r = get(cond, st, ep)
            if r is None:
                continue
            h = r.get("hurdle") or {}; ph = (h.get("phases") or {}).get("all") or {}
            cs = h.get("client_obstacle_safety") or {}
            f = lambda k: "-" if not ph.get(k) else f"{1e3 * ph[k]['min_m']:.1f}"
            ct = "-" if st == "B" else (("robot" if cs.get("robot_collision") else "") + ("+apple" if cs.get("payload_collision") else "")
                                         + (" stop" if h.get("client_obstacle_stop_lines") else "")).strip() or "none"
            ag = "-" if cond == "E0" or st == "B" else f"{h.get('ag3s_n_chunks_with_target_points_on_hurdle', 0)}/{h.get('ag3s_n_chunks', 0)}"
            agd = "-" if cond == "E0" or st == "B" or h.get("ag3s_min_target_to_hurdle_m") is None else f"{1e3 * h['ag3s_min_target_to_hurdle_m']:.0f}"
            bi = r.get("bit_identity_vs_T39")
            row = [ep, st, cond, "Y" if r["success"] else "N", "Y" if r["grasp"] else "N", "Y" if r["place"] else "N", r["n_hold_chunks"],
                   ",".join(f"{k}:{v}" for k, v in r["hold_kind_counts"].items()) or "-", f("A"), f("B"), f("C"), ct,
                   "-" if h.get("apple_left_hand_t") is None else str(h["apple_left_hand_t"]), ag, agd,
                   "-" if bi is None else ("Y" if bi["all_bit_identical"] else "N")]
            rows.append(row); tdata.append(dict(zip(cols, row)))
fig, ax = plt.subplots(figsize=(19, 0.27 * len(rows) + 0.9)); ax.set_axis_off(); fig.patch.set_facecolor(figstyle.SURFACE)
tb = ax.table(cellText=[[str(x) for x in r] for r in rows], colLabels=cols, cellLoc="center", bbox=[0, 0, 1, 1])
tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.0)
for (i, j), cell in tb.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if i == 0:
        cell.set_text_props(color=figstyle.INK, weight="bold")
ax.set_title("T40 outcomes (B = no obstacle, C = hurdle round 1, D = hurdle round 2). Min distances: whole run, control rows, client distance code "
             "(B: round-1 pose virtual). AG3S tgt-pts on hurdle = chunks with >= 1 target point within 15 mm of the hurdle / chunks", color=figstyle.INK, fontsize=9)
fig.tight_layout()
save(fig, f"t40fg-table{RT}", {"definitions": __doc__, "columns": cols, "rows": tdata})
