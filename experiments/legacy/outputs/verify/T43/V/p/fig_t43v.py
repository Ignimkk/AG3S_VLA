"""T43V figures (verifier). usage: fig_t43v.py [--overwrite]
inputs : outputs/verify/T43/V/analysis/analysis.json (analyze_t43v.py), numbers.json (partial_v.py --final), hold_v.json.
outputs: AG3S/ag3s/docs/figures/t43/t43v-scene.png, t43v-distance.png, t43v-table.png, t43v-chunk-time.png (+ .json sidecars)

scene      : per hurdle episode (T40 seed), V0 | V2 third-person front frame (recorded third_person_front.mp4, one frame per control step,
             avoidance-arrow overlay drawn by the client: green = executed refined - reference TCP vector x8, red = HOLD) at the control row
             of the run's minimum hurdle distance over fingers, palm, other robot geoms and the apple (client pair_distance).
distance   : per hurdle run (episode x seed), min over robot sets (fingers Af, palm Ae, other robot C) to hurdle (solid) and apple B to
             hurdle (dotted) vs control step, mm, clipped at 150; arm colours E0 / V0 / V1 / V2 / V3 = CATEGORICAL slots 0-4 (V3 = V2 without --gripper-cover); x = last row of a
             run that ended in an obstacle stop; lines at 0 and 10 mm (V1 / V2 obstacle margin).
table      : outcome per hurdle run and arm: S (success) / G (grasp only) / - ; contact (client robot / payload collision, first contact
             row phase); HOLD chunks (trailing to the end in brackets); min mm fingers / palm / apple / other. Second panel: V2 no-obstacle
             runs vs T40 B / T41a B (outcome from T41c.verify.json numbers.outcome).
chunk-time : per arm, ECDF of the client round trip per chunk (first chunk excluded, pooled over hurdle runs) and of the server TO
             solve time (to.solve_ms); vertical line at 533.3 ms (8 control steps at 15 Hz)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle

AN = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V/analysis'))
A = json.load(open(AN / "analysis.json"))["runs"]
NUM = json.load(open(AN / "numbers.json"))
T41C = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T41c.verify.json')))["numbers"]["outcome"]
OUT = pathlib.Path(os.environ.get("T43V_FIGOUT", _ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))); OUT.mkdir(parents=True, exist_ok=True)
KO = figstyle.use_korean()
ARMS = ("E0", "V0", "V1", "V2", "V3")  # V3 = V2 without --gripper-cover (lead, added after the first figures)
COL = {a: figstyle.CATEGORICAL[i] for i, a in enumerate(ARMS)}
EPS = [1807, 1968, 1982, 1967]
SEEDS = {1807: (18071, 18073), 1968: (19681, 19683), 1982: (19822, 19823), 1967: (19672, 19673)}
PERIOD = 1000.0 * 8 / 15
PREFIX = os.environ.get("T43V_FIGPREFIX", "t43v")          # V3 addition: t43v-v3 (the committed t43v-* figures are not overwritten)
SCENE = tuple(os.environ.get("T43V_SCENE_ARMS", "V0,V2").split(","))


def save(fig, name, data):
    p = OUT / f"{name}.png"
    assert not p.exists() or "--overwrite" in sys.argv, f"{p} exists"
    fig.savefig(p, dpi=150, facecolor=figstyle.SURFACE); plt.close(fig)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, default=float)); print(p)


def run(arm, ep, seed, stage="H"):
    return A.get(f"{arm}/{stage}/{arm}_ep{ep}_s{seed}")


def nrun(arm, ep, seed, stage="H"):
    return NUM["per_run"].get(f"{arm}/{stage}/{arm}_ep{ep}_s{seed}")


def ser(r):
    s = r["hurdle"]["series"]
    t = np.asarray(s["t"]); rob = np.minimum.reduce([np.asarray(s[k]) for k in ("Af", "Ae", "C")]); ap = np.asarray(s["B"])
    return t, rob, ap


def frame(r, t_row):
    vid = pathlib.Path(r["run_dir"]) / "third_person_front.mp4"
    rd = imageio.get_reader(str(vid)); n = rd.count_frames(); kf = min(int(t_row), n - 1); img = rd.get_data(kf); rd.close()
    Hh, Ww = img.shape[:2]; crop = (int(0.02 * Hh), int(0.80 * Hh), int(0.30 * Ww), int(0.98 * Ww))
    return img[crop[0]:crop[1], crop[2]:crop[3]], kf, n, crop


def outcome(n):
    return "S" if n["success"] else ("G" if n["grasp"] else "-")


def contact_txt(n):
    out = []
    cr = n.get("contact_rows") or {}
    for k, lab in (("Af", "fing"), ("Ae", "palm"), ("C", "arm"), ("B", "apple")):
        c = cr.get(k) or {}
        if c.get("first_t") is not None:
            out.append(f"{lab}@{c['first_t']}({c['first_phase'].replace('before_lift', 'pre')})")
    return " ".join(out) if out else "none"


# ---- scene: V0 | V2 closest approach
fig, axes = plt.subplots(len(EPS), 2, figsize=(10, 3.3 * len(EPS)), squeeze=False)
side = {"definitions": __doc__, "episodes": {}}
for row, ep in zip(axes, EPS):
    seed = SEEDS[ep][0]; side["episodes"][ep] = {}
    for ax, arm in zip(row, SCENE):
        r = run(arm, ep, seed); n = nrun(arm, ep, seed)
        if r is None or n is None:
            ax.set_axis_off(); ax.text(0.5, 0.5, f"{arm}: no run", ha="center", color=figstyle.INK_2); continue
        t, rob, ap = ser(r); allm = np.minimum(rob, ap); i = int(np.argmin(allm))
        img, kf, nf, crop = frame(r, i)
        ax.imshow(img); ax.set_axis_off(); mm = n["hurdle_min_mm"]
        ax.set_title(f"{arm} ep{ep} s{seed}  t={int(t[i])}  closest {1e3 * allm[i]:.1f} mm ({'apple' if ap[i] < rob[i] else 'robot'})\n"
                     f"{outcome(n)}  fing {mm['Af']:.1f} palm {mm['Ae']:.1f} apple {mm['B']:.1f} arm {mm['C']:.1f} mm  contact: {contact_txt(n)}",
                     color=COL[arm], fontsize=8)
        side["episodes"][ep][arm] = {"run_dir": r["run_dir"], "frame_index": kf, "n_frames": nf, "t": int(t[i]), "closest_mm": 1e3 * float(allm[i]),
                                     "closest_is_apple": bool(ap[i] < rob[i]), "hurdle_min_mm": mm, "outcome": outcome(n), "crop_rows_cols": crop,
                                     "contact_rows": n.get("contact_rows")}
fig.suptitle(f"T43 V hurdle (T43 P placement): {' | '.join(SCENE)} (V0 = T34 E3b, V2 = all fixes, V3 = V2 without gripper cover), closest approach; arrow = refined - reference TCP x8 (red = HOLD)",
             color=figstyle.INK, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.975))
save(fig, f"{PREFIX}-scene", side)

# ---- distance vs time
fig, axes = plt.subplots(len(EPS), 2, figsize=(12, 2.6 * len(EPS)), squeeze=False, sharex=True)
figstyle.style_axes(fig, axes.ravel())
side = {"definitions": __doc__, "runs": {}}
for row, ep in zip(axes, EPS):
    for ax, seed in zip(row, SEEDS[ep]):
        for arm in ARMS:
            r = run(arm, ep, seed); n = nrun(arm, ep, seed)
            if r is None:
                continue
            t, rob, ap = ser(r)
            ax.plot(t, np.minimum(1e3 * rob, 150), color=COL[arm], lw=1.2, label=f"{arm} robot")
            ax.plot(t, np.minimum(1e3 * ap, 150), color=COL[arm], lw=1.0, ls=":", label=f"{arm} apple")
            if n["obstacle_stop"]:
                ax.plot([t[-1]], [min(1e3 * min(rob[-1], ap[-1]), 150)], "x", color=COL[arm], ms=8, mew=2)
            side["runs"][f"{arm}_ep{ep}_s{seed}"] = {"t": t.tolist(), "robot_min_mm": (1e3 * rob).tolist(), "apple_mm": (1e3 * ap).tolist(),
                                                     "obstacle_stop": n["obstacle_stop"], "outcome": outcome(n)}
        ax.axhline(0, color=figstyle.INK, lw=0.8); ax.axhline(10, color=figstyle.INK_2, lw=0.6, ls="--")
        ax.set_ylim(-5, 155); ax.set_title(f"ep{ep} s{seed}  " + "  ".join(f"{a}:{outcome(nrun(a, ep, seed))}" for a in ARMS if nrun(a, ep, seed)), fontsize=9, color=figstyle.INK)
        ax.set_ylabel("hurdle dist (mm)", fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("control step")
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=10, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, f"{PREFIX}-distance", side)

# ---- outcome table
cols = ["run"] + [f"{a}" for a in ARMS]
cells, side = [], {"definitions": __doc__, "rows": []}
for ep in EPS:
    for seed in SEEDS[ep]:
        row = [f"ep{ep} s{seed}"]; srow = {"episode": ep, "seed": seed}
        for arm in ARMS:
            n = nrun(arm, ep, seed)
            if n is None:
                row.append("(not run)"); continue
            mm = n["hurdle_min_mm"] or {}
            f = lambda k: "-" if mm.get(k) is None else f"{mm[k]:.0f}"
            hc = f"H{n['n_hold_chunks']}" + (f"[{n['trailing_hold_chunks']} to end]" if n["trailing_hold_chunks"] else "")
            con = ("robot " if n["client_robot_collision"] else "") + ("apple " if n["client_payload_collision"] else "")
            row.append(f"{outcome(n)} {hc} {con.strip() or 'no contact'}\n{f('Af')}/{f('Ae')}/{f('B')}/{f('C')} mm")
            srow[arm] = {"outcome": outcome(n), "hold_chunks": n["n_hold_chunks"], "trailing_hold_chunks": n["trailing_hold_chunks"],
                         "robot_collision": n["client_robot_collision"], "payload_collision": n["client_payload_collision"], "hurdle_min_mm": mm,
                         "contact_rows": n.get("contact_rows")}
        cells.append(row); side["rows"].append(srow)
summ = NUM["summary_per_arm"]
srow = ["total S/G/P"] + [(lambda s: f"{s['success']}/{s['grasp']}/{s['place']} of {s['n']}\nrobot {s['runs_client_robot_collision']} apple {s['runs_client_payload_collision']} HOLD {s['hold_chunks']}")(summ[f"{a}/H"]) if f"{a}/H" in summ else "-" for a in ARMS]
cells.append(srow); side["summary"] = {a: summ.get(f"{a}/H") for a in ARMS}
bcells = []; side["no_obstacle"] = []
for ep, seed in ((1807, 18071), (1968, 19681), (1982, 19822), (1995, 19952), (1967, 19672)):
    n = nrun("V2", ep, seed, "B"); t40 = (T41C.get(f"B_ep{ep}") or {}).get("T40"); t41a = (T41C.get(f"B_ep{ep}") or {}).get("T41a")
    o = lambda x: "-" if x is None else ("S" if x["success"] else ("G" if x["grasp"] else "-"))
    bcells.append([f"ep{ep} s{seed}", o(t40), o(t41a), "(not run)" if n is None else f"{outcome(n)} H{n['n_hold_chunks']}"])
    side["no_obstacle"].append({"episode": ep, "seed": seed, "T40_B": t40, "T41a_B": t41a, "V2": None if n is None else {"outcome": outcome(n), "hold_chunks": n["n_hold_chunks"], "grasp_t": n["grasp_t"], "place_t": n["place_t"]}})
pr = NUM["pairs"]
fig = plt.figure(figsize=(14, 9.5)); fig.patch.set_facecolor(figstyle.SURFACE)
ax = fig.add_axes([0.01, 0.30, 0.98, 0.65]); ax.set_axis_off()
tb = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center", colWidths=[0.1] + [0.18] * 5)
tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 2.6)
for (i, j), c in tb.get_celld().items():
    c.set_edgecolor(figstyle.GRID_INK)
    if i == 0 and j > 0:
        c.set_text_props(color=COL[ARMS[j - 1]])
ax.set_title("T43 V hurdle runs: S success / G grasp only / - ; H = HOLD chunks [trailing to end]; min mm fingers/palm/apple/arm",
             fontsize=9, color=figstyle.INK)
ax2 = fig.add_axes([0.01, 0.02, 0.45, 0.24]); ax2.set_axis_off()
tb2 = ax2.table(cellText=bcells, colLabels=["no obstacle", "T40 B", "T41a B", "V2"], loc="upper center", cellLoc="center")
tb2.auto_set_font_size(False); tb2.set_fontsize(8); tb2.scale(1, 1.6)
ax2.set_title("no-obstacle regression (T40 seeds)", fontsize=9, color=figstyle.INK)
ax3 = fig.add_axes([0.50, 0.02, 0.49, 0.24]); ax3.set_axis_off()
pc = [[k.replace("_vs_", " vs "), f"{v['n_pairs']}"] + [f"{v[f]['both']}/{v[f][k.split('_vs_')[0] + '_only']}/{v[f][k.split('_vs_')[1] + '_only']}/{v[f]['neither']}  p={v[f]['mcnemar_exact_p']:.2f}" for f in ("success", "grasp", "place")]
      for k, v in pr.items()]
tb3 = ax3.table(cellText=pc, colLabels=["pair (A vs B)", "n", "success both/A/B/neither", "grasp", "place"], loc="upper center", cellLoc="center")
tb3.auto_set_font_size(False); tb3.set_fontsize(7.5); tb3.scale(1, 1.6)
ax3.set_title("paired by (episode, seed), exact McNemar p", fontsize=9, color=figstyle.INK)
side["pairs"] = pr
save(fig, f"{PREFIX}-table", side)

# ---- chunk-time distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2)); figstyle.style_axes(fig, axes)
side = {"definitions": __doc__, "arms": {}}
for arm in ARMS:
    rt, so = [], []
    for k, r in A.items():
        if r["arm"] == arm and r["stage"] == "H":
            rt += [c["client_round_trip_ms"] for c in r["chunks"][1:] if c["client_round_trip_ms"] is not None]
            so += [c["solve_ms"] for c in (r.get("server_chunks") or [])[1:] if c["solve_ms"] is not None]
    side["arms"][arm] = {"round_trip_ms": rt, "to_solve_ms": so}
    for ax, v in zip(axes, (rt, so)):
        if v:
            x = np.sort(v); ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=COL[arm],
                                    label=f"{arm} n={len(x)} med {np.median(x):.0f} p90 {np.percentile(x, 90):.0f} ms")
for ax, tt in zip(axes, ("client round trip per chunk (hurdle runs, first chunk excluded)", "server TO solve per chunk (to.solve_ms)")):
    ax.axvline(PERIOD, color=figstyle.INK, lw=1, ls="--"); ax.text(PERIOD, 0.02, " 533 ms", fontsize=8, color=figstyle.INK)
    ax.set_xscale("log"); ax.set_xlabel("ms (log)"); ax.set_ylabel("fraction of chunks <= x"); ax.set_title(tt, fontsize=9, color=figstyle.INK)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
side["load_samples_file"] = "outputs/verify/T43/V/runs/load_samples.txt"
fig.tight_layout()
save(fig, f"{PREFIX}-chunk-time", side)
