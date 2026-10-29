"""T42-1 figures (verifier). usage: fig_t42_1.py SUMMARY_JSON
-> AG3S/ag3s/docs/figures/t42/t42-1-table.png        per-fruit success table (episode x seed x condition), + .json
   AG3S/ag3s/docs/figures/t42/t42-1-scene-<EP>.png   per selected episode, + .json
   AG3S/ag3s/docs/figures/t42/t42-1-screening.png    graph: success count per (fruit, arm), E3b and E0, + .json

table : every screened episode, grouped by fruit and arm; E3b rep 1-3 and E0 rep 1-3 (seed = 10*ep + rep); Y = success
        (grasp and place, T28 rule generalized to the episode's target fruit), g = grasp only, N = neither, '-' = not run;
        apple + left rep 1-2 are T39 runs (marked *). Library = E3b 3/3; selected = up to 2 per fruit (plan.select()).
scene : left  = third-person front frame at control step 0 (recorded third_person_front.mp4, frame 0) of the E3b rep-1 run
                (rep 3 for apple + left), with the target-fruit carry path of every E3b run of that episode (solid) and every
                E0 run (dashed) projected into the image with the recording camera (pi05_infer configure_view_camera 'front':
                free camera lookat (0.45, 0, 0.85), distance 1.7, azimuth 180, elevation -18; fovy from the model, 640 x 480).
        right = top view (world x-y) of the same paths, crate outer rectangle (half 0.090 x 0.150, crate pose at planning row 0),
                fruit centres at planning row 0. Path = target centre qpos per control row, from t = 0 to the end of the run."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle

S = json.loads(pathlib.Path(sys.argv[1]).read_text())
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t42')); OUT.mkdir(parents=True, exist_ok=True)
KO = figstyle.use_korean()
FRUITS = ("apple", "orange", "banana", "pear"); ARMS = ("left", "right")
FCOL = {f: figstyle.CATEGORICAL[i] for i, f in enumerate(("apple", "banana", "orange", "pear"))}   # slot order = analysis CENTRES
CRATE_COL = figstyle.CATEGORICAL[6]
runs = S["runs"]; table = S["table"]; sel = S["selected"]


def save(fig, name, data):
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=150, facecolor=figstyle.SURFACE); plt.close(fig)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, default=float)); print(p)


def run_of(cond, ep, rep):
    s = 10 * ep + rep
    return runs.get(f"{cond}_ep{ep}_s{s}") or runs.get(f"T39/{cond}_ep{ep}_s{s}")


# ---------------- table
eps_order = sorted(table, key=lambda e: (FRUITS.index(table[e]["fruit"]), ARMS.index(table[e]["arm"]), -table[e]["score_m"], int(e)))
selset = {e for v in sel.values() for e in v}
lib = {e for v in S["library_E3b_3of3"].values() for e in v}
cols = ["fruit", "arm", "episode", "split", "crowd mm", "E3b r1", "E3b r2", "E3b r3", "E3b Y", "E0 r1", "E0 r2", "E0 r3", "E0 Y", "library", "selected"]
rows = []; tdata = []; fills = []


def cell(v):
    if v is None:
        return "-"
    if v["success"]:
        s = "Y" if v["robust_grasp"] else ("r" if v.get("regrasp") else "s")
    else:
        s = "g" if v["grasp"] else "N"
    return s + ("*" if v["source"] == "T39" else "")


for e in eps_order:
    t = table[e]; ie = int(e)
    row = [t["fruit"], t["arm"], e, t["split"], f'{1e3 * t["score_m"]:.0f}'] + [cell(t["E3b"][str(r)]) for r in (1, 2, 3)] + \
          [f'{t["E3b_n_success_robust"]}/{t["E3b_n_done"]}'] + [cell(t["E0"][str(r)]) for r in (1, 2, 3)] + \
          [f'{t["E0_n_success_robust"]}/{t["E0_n_done"]}' if t["E0_n_done"] else "-", "Y" if ie in lib else "", "Y" if ie in selset else ""]
    rows.append(row); tdata.append({**dict(zip(cols, row)), "raw": t})
fig, ax = plt.subplots(figsize=(12.5, 0.28 * len(rows) + 1.2)); ax.set_axis_off(); fig.patch.set_facecolor(figstyle.SURFACE)
tb = ax.table(cellText=rows, colLabels=cols, cellLoc="center", bbox=[0, 0, 1, 1])
tb.auto_set_font_size(False); tb.set_fontsize(8)
for (i, j), c in tb.get_celld().items():
    c.set_edgecolor(figstyle.GRID_INK)
    if i == 0:
        c.set_text_props(color=figstyle.INK, weight="bold"); continue
    txt = c.get_text().get_text()
    if j in (5, 6, 7, 9, 10, 11) and txt.startswith("Y"):
        c.set_facecolor(figstyle.SEQ_BLUE[1])
    if j in (0,):
        c.set_text_props(color=FCOL[txt])
    if j == 14 and txt == "Y":
        c.set_facecolor(figstyle.SEQ_BLUE[3])
ax.set_title("T42-1 no-obstacle runs (seed 10*ep+rep): Y = success with a robust grasp, s = success but the fruit slipped in the hand, "
             "r = success after a regrasp,\ng = grasp only, N = no grasp, - = not run, * = T39 run. Library = E3b Y on 3/3 seeds; selected = up to 2 per fruit",
             color=figstyle.INK, fontsize=9)
fig.tight_layout()
save(fig, "t42-1-table", {"definitions": __doc__, "columns": cols, "rows": tdata, "selected": sel, "library": S["library_E3b_3of3"]})

# ---------------- screening graph: per (fruit, arm) — screened episodes, E3b rep-1 (screen) success, E3b 3/3, E0 3/3 among selected
grp = [(f, a) for f in FRUITS for a in ARMS]
vals = {k: [] for k in ("screened", "E3b_screen_success", "E3b_3of3", "selected", "E0_3of3_selected")}
for f, a in grp:
    es = [e for e in table if table[e]["fruit"] == f and table[e]["arm"] == a]
    scr_rep = 3 if (f, a) == ("apple", "left") else 1
    vals["screened"].append(len(es))
    scr = lambda e: 3 if (f, a) == ("apple", "left") else 1
    vals["E3b_screen_success"].append(sum(bool((table[e]["E3b"][str(scr(e))] or {}).get("success") and (table[e]["E3b"][str(scr(e))] or {}).get("robust_grasp")) for e in es))
    vals["E3b_3of3"].append(sum(table[e]["E3b_n_success_robust"] == 3 for e in es))
    vals["selected"].append(sum(int(e) in selset for e in es))
    vals["E0_3of3_selected"].append(sum(int(e) in selset and table[e]["E0_n_success_robust"] == 3 for e in es))
fig, ax = plt.subplots(figsize=(10, 3.8)); figstyle.style_axes(fig, [ax])
x = np.arange(len(grp)); w = 0.16
for i, (k, lab) in enumerate((("screened", "screened"), ("E3b_screen_success", "E3b screening seed Y"),
                              ("E3b_3of3", "E3b Y 3/3"), ("selected", "selected"), ("E0_3of3_selected", "E0 Y 3/3 (selected)"))):
    ax.bar(x + (i - 2) * w, vals[k], w, color=figstyle.SEQ_BLUE[2 + 2 * i] if i < 4 else figstyle.CATEGORICAL[1], label=lab)
ax.set_xticks(x); ax.set_xticklabels([f"{f}\n{a}" for f, a in grp]); ax.set_ylabel("episodes")
ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
ax.set_title("T42-1 screening funnel per fruit and arm (no obstacle; Y = success with a robust grasp)", color=figstyle.INK, fontsize=10, pad=24)
fig.tight_layout()
save(fig, "t42-1-screening", {"definitions": __doc__, "groups": [f"{f}_{a}" for f, a in grp], **vals})


# ---------------- scene
def cam():
    fovy = 45.0
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_path("${AG3S_PROJECT_ROOT}-t42/robots/rby1/description/models/rby1a/mujoco/model_transport.xml")
        fovy = float(m.vis.global_.fovy)
    except Exception:
        pass
    look = np.array([0.45, 0.0, 0.85]); dist = 1.7; az = math.radians(180.0); el = math.radians(-18.0)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    pos = look - dist * fwd
    up = np.array([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)])
    right = np.cross(fwd, up)
    return {"pos": pos, "fwd": fwd, "up": up, "right": right, "fovy": fovy, "W": 640, "H": 480}


C = cam()


def project(P):
    P = np.asarray(P, float) - C["pos"]
    zc = P @ C["fwd"]; xc = P @ C["right"]; yc = P @ C["up"]
    f = 0.5 * C["H"] / math.tan(math.radians(C["fovy"]) / 2)
    return np.stack([C["W"] / 2 + f * xc / zc, C["H"] / 2 - f * yc / zc], 1)


def rect(crate):
    q = crate["quat"]; w, x, y, z = q
    Rm = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    c = np.array([[0.09, 0.15, 0], [-0.09, 0.15, 0], [-0.09, -0.15, 0], [0.09, -0.15, 0], [0.09, 0.15, 0]])
    return (Rm @ c.T).T + np.asarray(crate["pos"])


for f in FRUITS:
    for ep in sel.get(f, []):
        t = table[str(ep)]; arm = t["arm"]
        base = run_of("E3b", ep, 3 if (f, arm) == ("apple", "left") else 1)
        if base is None:
            continue
        vid = pathlib.Path(base["run_dir"]) / "third_person_front.mp4"
        rd = imageio.get_reader(str(vid)); img = rd.get_data(0); rd.close()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), gridspec_kw={"width_ratios": [1.35, 1]})
        ax = axes[0]; ax.imshow(img); ax.set_axis_off()
        side = {"episode": ep, "fruit": f, "arm": arm, "frame_source": str(vid), "frame_index": 0, "camera": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in C.items()},
                "paths": {}}
        ax2 = axes[1]; figstyle.style_axes(fig, [ax2])
        op = base["object_poses_plan0"]; cr = rect(base["crate_pose_plan0"])
        ax2.plot(cr[:, 1], cr[:, 0], color=CRATE_COL, lw=1.5, label="crate")
        for k in ("apple", "banana", "orange", "pear"):
            p = op[k]["pos"]; ax2.plot(p[1], p[0], "o", ms=9 if k == f else 6, color=FCOL[k], mfc=FCOL[k] if k == f else "none")
            ax2.annotate(k, (p[1], p[0]), textcoords="offset points", xytext=(5, 5), fontsize=7, color=FCOL[k])
        k_i = 0
        for cond, ls in (("E3b", "-"), ("E0", "--")):
            for rep in (1, 2, 3):
                r = run_of(cond, ep, rep)
                if r is None:
                    continue
                P = np.asarray(r["carry_path_xyz"]); uv = project(P)
                col = figstyle.CATEGORICAL[[1, 2, 5][rep - 1]]
                lab = f'{cond} s{10 * ep + rep}{"*" if r["source"] == "T39" else ""} {"ok" if r["success"] else "fail"}'
                ax.plot(uv[:, 0], uv[:, 1], ls, color=col, lw=1.4 if cond == "E3b" else 1.1, label=lab)
                ax2.plot(P[:, 1], P[:, 0], ls, color=col, lw=1.4 if cond == "E3b" else 1.1, label=lab)
                side["paths"][f"{cond}_s{10 * ep + rep}"] = {"source": r["source"], "run_dir": r["run_dir"], "success": r["success"],
                                                            "grasp_t": r["grasp_t"], "place_t": r["place_t"], "n_hold_chunks": r["n_hold_chunks"],
                                                            "xyz_every_8th": P[::8].tolist(), "uv_every_8th": uv[::8].tolist()}
        p0 = project(np.asarray([op[f]["pos"]]))[0]
        ax.plot(*p0, "o", ms=10, mfc="none", color=FCOL[f])
        ax.set_xlim(0, C["W"]); ax.set_ylim(C["H"], 0)
        ax.legend(fontsize=7, frameon=True, loc="lower left")
        ax.set_title(f"ep{ep} {f} ({arm} arm): t=0 front view + {f} carry path", color=figstyle.INK, fontsize=10)
        ax2.set_xlabel("world y (m)"); ax2.set_ylabel("world x (m)"); ax2.invert_yaxis(); ax2.set_aspect("equal")
        ax2.set_title("top view from +z, oriented as the front camera (robot at top)", color=figstyle.INK, fontsize=10)
        fig.suptitle(f"T42-1 library episode {ep}: {f}, {arm} arm, no obstacle (solid E3b, dashed E0)", color=figstyle.INK, fontsize=11)
        fig.tight_layout()
        save(fig, f"t42-1-scene-{ep}", {"definitions": __doc__, **side})
