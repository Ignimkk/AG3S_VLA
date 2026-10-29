"""T42-2 figures (verifier). usage: fig_t42_2.py ANALYSIS_JSON [KIND]   (KIND default 'place')
-> AG3S/ag3s/docs/figures/t42/t42-2-scene-<EP>-<KIND>.png, t42-2-distance-<KIND>.png, t42-2-table-<KIND>.png (+ .json sidecars)
scene    : per episode, E0 | E3b third-person front frames (recorded third_person_front.mp4, E3b with the --viz-avoidance-arrow overlay;
           one frame per control step) at the control row of the minimum obstacle distance min(gripper, target, other robot) of that
           run (obst_metrics series), titled with t, the distance, outcome and obstacle stop.
distance : per episode, min(gripper, target) obstacle distance (mm, client distance code, capped 300) vs control step, E0 / E3b
           (CATEGORICAL slots 0 / 1); x = last row of a run that ended with an obstacle stop; dotted vertical = lift_t of the run.
table    : outcome per episode x condition: success / grasp / place, robust grasp (B), HOLD chunks and kinds, minimum distance per
           group (gripper, target, other robot, other fruits; whole run), client contact pairs and obstacle stop (phase, t),
           AG3S chunks with target points within 15 mm of the obstacle / chunks, min target centroid -> obstacle (mm)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle

A = json.loads(pathlib.Path(sys.argv[1]).read_text()); KIND = sys.argv[2] if len(sys.argv) > 2 else "place"
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t42')); OUT.mkdir(parents=True, exist_ok=True)
figstyle.use_korean()
COL = {"E0": figstyle.CATEGORICAL[0], "E3b": figstyle.CATEGORICAL[1]}
runs = {k: v for k, v in A["runs"].items() if v["kind"] == KIND and v["cond"] in ("E0", "E3b")}
eps = sorted({r["episode"] for r in runs.values()}, key=lambda e: [r["episode"] for r in runs.values()].index(e))


def get(cond, ep):
    return next((r for r in runs.values() if r["cond"] == cond and r["episode"] == ep), None)


def save(fig, name, data):
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=150, facecolor=figstyle.SURFACE); plt.close(fig)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, default=float)); print(p)


def mins(r):
    s = r["obst"]["series"]; t = np.asarray(s["t"]); slots = r["obst"]["slots"]
    g = lambda grp: np.min([np.asarray(s[f"{sl}:{grp}"]) for sl in slots], axis=0)
    ab = np.minimum(g("GRIP"), g("TARGET")); allm = np.minimum(ab, g("ROBOT_OTHER"))
    return t, ab, allm


for ep in eps:
    r0, r3 = get("E0", ep), get("E3b", ep)
    if not (r0 and r3):
        continue
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9)); side = {}
    for ax, (cond, r) in zip(axes, (("E0", r0), ("E3b", r3))):
        t, ab, allm = mins(r); i = int(np.argmin(allm))
        vid = pathlib.Path(A["runs"][next(k for k, v in A["runs"].items() if v is r)]["obst"]["run_dir"]) / "third_person_front.mp4"
        rd = imageio.get_reader(str(vid)); n = rd.count_frames(); k = min(i, n - 1); img = rd.get_data(k); rd.close()
        ax.imshow(img); ax.set_axis_off(); o = r["obst"]
        ax.set_title(f"{cond}  t={int(t[i])}  min dist {1e3 * allm[i]:.1f} mm  {'success' if r['success'] else 'fail'}"
                     f"{'  obstacle stop' if o['obstacle_stop'] else ''}", color=figstyle.INK, fontsize=10)
        side[cond] = {"run_dir": o["run_dir"], "frame_index": k, "n_frames": n, "t": int(t[i]), "min_dist_m": float(allm[i]),
                      "min_gripper_or_target_m": float(ab[i]), "success": r["success"], "obstacle_stop": o["obstacle_stop"], "obstacles": o["obstacles"]}
    slots = "+".join(r3["obst"]["slots"])
    fig.suptitle(f"T42-2 ep{ep} ({r3['target_fruit']}, {r3['used_arm']} arm), {slots}: closest approach (E3b video with the avoidance arrow)",
                 color=figstyle.INK, fontsize=11)
    fig.tight_layout()
    save(fig, f"t42-2-scene-{ep}-{KIND}", {"definitions": __doc__, "episode": ep, **side})

E = [ep for ep in eps if get("E0", ep) or get("E3b", ep)]
fig, axes = plt.subplots(1, len(E), figsize=(3.4 * len(E), 3.6), sharey=True, squeeze=False); axes = axes[0]
figstyle.style_axes(fig, axes); data = {"definitions": __doc__, "episodes": {}}
for ax, ep in zip(axes, E):
    data["episodes"][ep] = {}
    for cond in ("E0", "E3b"):
        r = get(cond, ep)
        if r is None:
            continue
        t, ab, _ = mins(r); y = np.minimum(1e3 * ab, 300)
        ax.plot(t, y, "-", color=COL[cond], lw=1.2, label=cond)
        if r["obst"]["obstacle_stop"]:
            ax.plot(t[-1], y[-1], "x", color=COL[cond], ms=8)
        if r["obst"]["lift_t"] is not None:
            ax.axvline(r["obst"]["lift_t"], color=COL[cond], lw=0.7, ls=":")
        data["episodes"][ep][cond] = {"t": t.tolist(), "min_gripper_or_target_mm": (1e3 * ab).tolist(), "obstacle_stop": r["obst"]["obstacle_stop"],
                                      "lift_t": r["obst"]["lift_t"], "success": r["success"]}
    r = get("E3b", ep) or get("E0", ep)
    ax.axhline(0, color=figstyle.INK_2, lw=0.8)
    ax.set_title(f"ep{ep} {r['target_fruit']} / {'+'.join(r['obst']['slots'])}", color=figstyle.INK, fontsize=9); ax.set_xlabel("control step")
axes[0].set_ylabel("obstacle distance, gripper or target (mm)"); axes[-1].legend(fontsize=8, frameon=False)
fig.suptitle(f"T42-2 {KIND}-phase obstacles: distance over time (x = obstacle stop, dotted = lift-off)", color=figstyle.INK, fontsize=10)
fig.tight_layout(); save(fig, f"t42-2-distance-{KIND}", data)

cols = ["ep", "fruit/arm", "slot", "cond", "success", "grasp", "place", "robust(B)", "HOLD", "HOLD kinds", "min grip mm", "min target mm",
        "min robot mm", "min fruits mm", "contact / stop", "AG3S tgt-pts on obst", "AG3S tgt->obst mm"]
rows = []; tdata = []
for ep in eps:
    for cond in ("E0", "E3b"):
        r = get(cond, ep)
        if r is None:
            continue
        o = r["obst"]; ph = o["phases"]["all"]; sl = o["slots"]
        f = lambda grp: "-" if not any(ph.get(f"{s}:{grp}") for s in sl) else f"{1e3 * min(ph[f'{s}:{grp}']['min_m'] for s in sl if ph.get(f'{s}:{grp}')):.1f}"
        cs = o.get("client_obstacle_safety") or {}
        ct = (("robot" if cs.get("robot_collision") else "") + ("+payload" if cs.get("payload_collision") else "")).strip("+")
        ct = (ct or "none") + (f" stop {o['stop_phase']} t={o['stop_t']}" if o["obstacle_stop"] else "")
        ag = "-" if cond == "E0" else f"{o.get('ag3s_n_chunks_with_target_points_on_obstacle', 0)}/{o.get('ag3s_n_chunks', 0)}"
        agd = "-" if (cond == "E0" or o.get("ag3s_min_target_to_obstacle_m") is None) else f"{1e3 * o['ag3s_min_target_to_obstacle_m']:.0f}"
        row = [ep, f"{r['target_fruit']}/{r['used_arm']}", "+".join(sl), cond, "Y" if r["success"] else "N", "Y" if r["grasp"] else "N", "Y" if r["place"] else "N",
               "Y" if r["robust"]["robust_grasp"] else "N", r["holds"]["n_hold"], ",".join(f"{k}:{v}" for k, v in r["holds"]["hold_kinds"].items()) or "-",
               f("GRIP"), f("TARGET"), f("ROBOT_OTHER"), f("FRUITS"), ct, ag, agd]
        rows.append([str(x) for x in row]); tdata.append(dict(zip(cols, row)))
fig, ax = plt.subplots(figsize=(19, 0.3 * len(rows) + 1.0)); ax.set_axis_off(); fig.patch.set_facecolor(figstyle.SURFACE)
tb = ax.table(cellText=rows, colLabels=cols, cellLoc="center", bbox=[0, 0, 1, 1]); tb.auto_set_font_size(False); tb.set_fontsize(7.5)
for (i, j), c in tb.get_celld().items():
    c.set_edgecolor(figstyle.GRID_INK)
    if i == 0:
        c.set_text_props(color=figstyle.INK, weight="bold")
    elif j == 4 and c.get_text().get_text() == "Y":
        c.set_facecolor(figstyle.SEQ_BLUE[1])
ax.set_title(f"T42-2 {KIND}-phase obstacle runs (seed rep 1, no-obstacle reference = T42-1). Min distances: whole run, control rows, client distance code",
             color=figstyle.INK, fontsize=9)
fig.tight_layout(); save(fig, f"t42-2-table-{KIND}", {"definitions": __doc__, "columns": cols, "rows": tdata})
