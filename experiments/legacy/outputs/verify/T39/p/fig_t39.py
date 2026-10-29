"""T39 figures from p/analyze_t39.py output. Every png gets a .json sidecar with the numbers it draws.
  t39-success-matrix.png   : episode x (seed rep, condition) grid, cell = success / grasp only / no grasp
  t39-paired-2x2.png       : paired 2x2 tables (success, grasp, place) with exact McNemar p
  t39-contact-height.png   : contact midpoint z - apple centre z at settle (T37 settle rule) per run, by condition, colour = success
  t39-chunk-time.png       : client round trip per chunk (E0, E3b) + E3b server split (infer, ag3s, trajopt, total), first chunk
                             of each run excluded, 533.3 ms chunk period line
  t39-hold-verdict.png     : E3b HOLD chunks per run, hold kinds, gate actions, verdict reason kinds
  t39-scene-<ep>_<key>.png : discordant pairs — E0 and E3b third-person front frames at the same control steps
                             (start, first divergence chunk, E0 / E3b grasp t, last frame); mp4 frame for row t = max(t-1, 0)
  t39-summary-table.png    : condition summary numbers
usage: fig_t39.py ANALYSIS_JSON OUT_DIR [PREFIX] [--max-scenes N]"""
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, "${AG3S_PROJECT_ROOT}-o1")
from AG3S.ag3s.experiments.common import figstyle as fs  # noqa: E402

KO = fs.use_korean()
args = [a for a in sys.argv[1:] if not a.startswith("--")]
MAXS = int(sys.argv[sys.argv.index("--max-scenes") + 1]) if "--max-scenes" in sys.argv else 8
if "--max-scenes" in sys.argv:
    args.remove(str(MAXS))
R = json.loads(pathlib.Path(args[0]).read_text())
F = pathlib.Path(args[1]); F.mkdir(parents=True, exist_ok=True)
PRE = args[2] if len(args) > 2 else "t39"
C = fs.CATEGORICAL
COL = {"success": C[2], "grasp_only": C[3], "no_grasp": C[7]}
CONDC = {"E0": C[0], "E3b": C[1]}
PERIOD = 1000.0 * 8 / 15
RUNS = R["runs"]; PAIRS = R["pairs"]
# windows to shade on the wall-clock panel: from --windows JSON [[start_epoch, end_epoch, label], ...]
WINDOWS = []
if "--windows" in sys.argv:
    wf = sys.argv[sys.argv.index("--windows") + 1]; args.remove(wf)
    WINDOWS = [(w[0], w[1], w[2], C[3 + i]) for i, w in enumerate(json.loads(pathlib.Path(wf).read_text()))]


def new(name):
    p = F / f"{PRE}-{name}"
    assert not p.exists(), f"{p} exists"
    return p


def save(fig, name, side, dpi=140):
    p = new(name); fig.savefig(p, dpi=dpi); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float)); print(p)


def status(r):
    return "success" if r["success"] else ("grasp_only" if r["grasp"] else "no_grasp")


# ---- success matrix
def rep(p):  # T39 seed = 10*ep + rep -> column "rep"; T28/T37 keys are already reps
    return p["key"] - 10 * p["episode"] if p["key"] > 10 * p["episode"] else p["key"]
eps = sorted({p["episode"] for p in PAIRS}); keys = sorted({rep(p) for p in PAIRS})
cols = [(k, c) for k in keys for c in ("E0", "E3b")]
grid = np.full((len(eps), len(cols)), np.nan); side = {"columns": [f"{c} rep{k}" for k, c in cols], "seed_rule": "seed = 10*episode + rep", "rows": {}}
st_code = {"success": 2, "grasp_only": 1, "no_grasp": 0}
by = {(p["episode"], rep(p)): p for p in PAIRS}
for i, e in enumerate(eps):
    side["rows"][e] = {}
    for j, (k, c) in enumerate(cols):
        p = by.get((e, k))
        if p is None:
            continue
        r = RUNS[p[f"{c}_run"]]; s = status(r); grid[i, j] = st_code[s]
        side["rows"][e][f"{c} rep{k}"] = {"seed": p["key"], "status": s, "run": p[f"{c}_run"], "grasp_t": r["grasp_t"], "place_t": r["place_t"]}
fig, ax = plt.subplots(figsize=(2.2 + 1.1 * len(cols), 0.9 + 0.32 * len(eps)))
from matplotlib.colors import ListedColormap
ax.imshow(grid, cmap=ListedColormap([COL["no_grasp"], COL["grasp_only"], COL["success"]]), vmin=0, vmax=2, aspect="auto")
for i in range(len(eps)):
    for j in range(len(cols)):
        k, c = cols[j]; p = by.get((eps[i], k))
        if p is not None and p["discordant_success"]:
            ax.text(j, i, "x", ha="center", va="center", fontsize=8, color=fs.INK)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{c}\nseed 10ep+{k}" for k, c in cols], fontsize=7.5)
ax.set_yticks(range(len(eps))); ax.set_yticklabels([str(e) for e in eps], fontsize=7.5)
for j in range(1, len(keys)):
    ax.axvline(2 * j - 0.5, color=fs.SURFACE, lw=3)
fs.style_axes(fig, ax)
ax.legend(handles=[Patch(color=COL[s], label=s.replace("_", " ")) for s in ("success", "grasp_only", "no_grasp")] +
          [Patch(color=fs.SURFACE, label="x = discordant pair")], fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
ax.set_title("success by episode x seed x condition (pair = same episode and seed)", fontsize=9.5, color=fs.INK)
fig.tight_layout(); save(fig, "success-matrix.png", side)

# ---- paired 2x2
T = R["paired_2x2"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, f in zip(axes, ("success", "grasp", "place")):
    t = T[f]; m = np.array([[t["both"], t["E0_only"]], [t["E3b_only"], t["neither"]]])
    ax.imshow(m, cmap=fs.sequential_cmap(), vmin=0, vmax=max(1, m.max()))
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14, color=fs.INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"E3b {f}", f"E3b no {f}"], fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"E0 {f}", f"E0 no {f}"], fontsize=8)
    # rows = E0, cols = E3b: [[both, E0 yes/E3b no], [E0 no/E3b yes, neither]]
    m2 = np.array([[t["both"], t["E0_only"]], [t["E3b_only"], t["neither"]]])
    for (i, j), v in np.ndenumerate(m2):
        pass
    ax.set_title(f"{f}: n pairs {t['n_pairs']}, discordant {t['E0_only']} vs {t['E3b_only']}\nexact McNemar p = {t['mcnemar_exact_p']:.3g}",
                 fontsize=9, color=fs.INK)
fs.style_axes(fig, axes)
fig.suptitle("paired E0 (VLA only) vs E3b (AG3S + ESDF + TO, gate / HOLD)", fontsize=10, color=fs.INK)
fig.tight_layout(); save(fig, "paired-2x2.png", {"layout": "rows E0 yes/no, cols E3b yes/no", "tables": T})

# ---- contact height (left: all runs; right: same points, y limited to -15..25 mm, points outside counted in the title)
fig, axes2 = plt.subplots(1, 2, figsize=(11, 4.2))
side = {"metric": "contact_mid_rel_apple_mm[2] at t_settle (T37 settle rule)", "points": [], "zoom_ylim_mm": [-15, 25]}
for pi_, ax in enumerate(axes2):
    n_out = 0
    for x, c in enumerate(("E0", "E3b")):
        rs = [(n, r) for n, r in RUNS.items() if r["cond"] == c and isinstance(r.get("contact"), dict) and r["contact"].get("contact_mid_z_mm") is not None]
        rng = np.random.default_rng(0)
        for n, r in rs:
            z = r["contact"]["contact_mid_z_mm"]; jx = x + rng.uniform(-0.18, 0.18)
            if pi_ == 1 and not (-15 <= z <= 25):
                n_out += 1; continue
            ax.scatter(jx, z, s=26, color=COL[status(r)], marker="o" if r["contact"].get("first_grasp_held") else "^", edgecolor="none")
            if pi_ == 0:
                side["points"].append({"run": n, "cond": c, "contact_mid_z_mm": z, "status": status(r), "first_grasp_held": r["contact"].get("first_grasp_held")})
        zs = [r["contact"]["contact_mid_z_mm"] for n, r in rs]
        if zs:
            ax.hlines(np.median(zs), x - 0.3, x + 0.3, color=fs.INK_2, lw=1.2)
            side.setdefault("median_mm", {})[c] = float(np.median(zs))
    ax.axhline(0, color=fs.GRID_INK, lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["E0", "E3b"])
    if pi_ == 1:
        ax.set_ylim(-15, 25); ax.set_title(f"zoom -15..25 mm ({n_out} points outside not drawn)", fontsize=9, color=fs.INK)
        side["zoom_points_outside"] = n_out
    else:
        ax.set_ylabel("contact midpoint z - apple centre z at settle (mm)"); ax.set_title("all runs (line = median)", fontsize=9, color=fs.INK)
axes2[1].legend(handles=[Patch(color=COL[s_], label=s_.replace("_", " ")) for s_ in COL] +
                [plt.Line2D([], [], color=fs.INK_2, marker="o", ls="", label="first grasp held"),
                 plt.Line2D([], [], color=fs.INK_2, marker="^", ls="", label="contact lost in 24 rows")], fontsize=7.5, frameon=False, loc="upper right")
fs.style_axes(fig, axes2); fig.suptitle("grasp contact height", fontsize=10, color=fs.INK)
fig.tight_layout(); save(fig, "contact-height.png", side)

# ---- chunk time
import datetime as _dt
fig, axes = plt.subplots(1, 3, figsize=(18, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.4]})
side = {"excluded": "first chunk of each run", "chunk_period_ms": PERIOD, "series": {}}
ax = axes[0]
for c in ("E0", "E3b"):
    v = [ch["client_round_trip_ms"] for r in RUNS.values() if r["cond"] == c for ch in r["chunks"][1:] if ch["client_round_trip_ms"] is not None]
    if v:
        ax.hist(v, bins=60, color=CONDC[c], alpha=0.75, label=f"{c} (n={len(v)}, median {np.median(v):.0f} ms)")
        side["series"][f"{c}_round_trip"] = {"n": len(v), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(max(v)),
                                             "n_gt_period": int(sum(x > PERIOD for x in v))}
ax.axvline(PERIOD, color=fs.INK, ls="--", lw=1); ax.text(PERIOD, ax.get_ylim()[1] * 0.95, " 533 ms", fontsize=8, color=fs.INK)
ax.set_xlabel("client round trip per chunk (ms)"); ax.set_ylabel("chunks"); ax.legend(fontsize=8, frameon=False)
ax = axes[1]; keys_ = ("infer", "ag3s", "trajopt", "total"); data = []
for k in keys_:
    v = [(ch["server_ms"] or {}).get(k) for r in RUNS.values() if r["cond"] == "E3b" for ch in r["chunks"][1:]]
    v = [x for x in v if x is not None]; data.append(v)
    if v:
        side["series"][f"E3b_server_{k}"] = {"n": len(v), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(max(v))}
if any(data):
    bp = ax.boxplot([d if d else [np.nan] for d in data], tick_labels=list(keys_), showfliers=True, patch_artist=True)
    for b, cc in zip(bp["boxes"], C):
        b.set_facecolor(cc); b.set_alpha(0.6)
    ax.axhline(PERIOD, color=fs.INK, ls="--", lw=1)
ax.set_ylabel("E3b server time per chunk (ms)"); ax.set_title("E3b server split (infer = policy)", fontsize=9, color=fs.INK)
ax = axes[2]
for c in ("E0", "E3b"):
    pts = [(ch["wall_first_control"], ch["client_round_trip_ms"]) for r in RUNS.values() if r["cond"] == c for ch in r["chunks"][1:]
           if ch.get("wall_first_control") is not None and ch["client_round_trip_ms"] is not None]
    if pts:
        x = [_dt.datetime.utcfromtimestamp(t) for t, _ in pts]
        ax.scatter(x, [v for _, v in pts], s=2, color=CONDC[c], alpha=0.5, label=c, edgecolor="none")
for (w0, w1, lab, cc) in WINDOWS:
    ax.axvspan(_dt.datetime.utcfromtimestamp(w0), _dt.datetime.utcfromtimestamp(w1), color=cc, alpha=0.25, label=lab)
ax.axhline(PERIOD, color=fs.INK, ls="--", lw=1)
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")); ax.set_xlabel("UTC (first control row of the chunk)"); ax.set_ylabel("client round trip (ms)")
ax.legend(fontsize=7.5, frameon=False, markerscale=4); ax.set_title("round trip over the S2-2 wall clock", fontsize=9, color=fs.INK)
side["marked_windows"] = [{"start_epoch": w0, "end_epoch": w1, "label": lab} for (w0, w1, lab, _) in WINDOWS]
fs.style_axes(fig, axes); fig.suptitle("chunk time (first chunk of each run excluded)", fontsize=10, color=fs.INK)
fig.tight_layout(); save(fig, "chunk-time.png", side)

# ---- hold / verdict (E3b)
e3 = {n: r for n, r in RUNS.items() if r["cond"] == "E3b"}
if e3:
    S3 = R["condition_summary"]["E3b"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    names = sorted(e3, key=lambda n: (e3[n]["episode"], e3[n]["key"]))
    axes[0].bar(range(len(names)), [e3[n]["n_hold_chunks"] for n in names], color=[COL[status(e3[n])] for n in names])
    axes[0].set_xticks(range(len(names))); axes[0].set_xticklabels([f'{e3[n]["episode"]} {e3[n]["key"]}' for n in names], rotation=90, fontsize=6)
    axes[0].set_ylabel("HOLD chunks per run (of 75)")
    for ax, key, title in ((axes[1], "hold_kind_counts_total", "HOLD kinds (chunks)"), (axes[2], "verdict_reason_kind_counts_total", "verdict reason kinds (chunks)")):
        d = S3[key]; ks = list(d)
        ax.barh(range(len(ks)), [d[k] for k in ks], color=C[0]); ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks, fontsize=8)
        for i, k in enumerate(ks):
            ax.text(d[k], i, f" {d[k]}", va="center", fontsize=8, color=fs.INK)
        ax.set_title(title, fontsize=9, color=fs.INK)
    axes[0].set_title(f"gate actions: {S3['gate_action_counts_total']}", fontsize=9, color=fs.INK)
    fs.style_axes(fig, axes); fig.tight_layout()
    save(fig, "hold-verdict.png", {"hold_per_run": {n: e3[n]["n_hold_chunks"] for n in names}, "hold_kind_counts_total": S3["hold_kind_counts_total"],
                                   "gate_action_counts_total": S3["gate_action_counts_total"], "verdict_reason_kind_counts_total": S3["verdict_reason_kind_counts_total"]})

# ---- scene: discordant pairs
import imageio.v2 as iio
disc = R["discordant"]["E3b_fail_E0_success"] + R["discordant"]["E0_fail_E3b_success"]
for p in disc[:MAXS]:
    r0, r3 = RUNS[p["E0_run"]], RUNS[p["E3b_run"]]; dv = p["divergence"]
    m0, m3 = pathlib.Path(r0["run_dir"]) / "third_person_front.mp4", pathlib.Path(r3["run_dir"]) / "third_person_front.mp4"
    if not (m0.exists() and m3.exists()):
        continue
    t01 = ((dv.get("by_threshold") or {}).get("0.1") or {}).get("chunk_t")
    ts = [("start", 0), ("exec diff > 1e-3", dv.get("chunk_t")), ("exec diff > 1e-1", t01), ("E0 grasp t", r0["grasp_t"]), ("E3b grasp t", r3["grasp_t"]), ("last", 600)]
    ts = [(l, t) for l, t in ts if t is not None]
    rd0, rd3 = iio.get_reader(m0), iio.get_reader(m3)
    n0, n3 = rd0.count_frames(), rd3.count_frames()
    fig, axes = plt.subplots(2, len(ts), figsize=(3.3 * len(ts), 5.6)); side = {"pair": p, "frames": []}
    for j, (lab, t) in enumerate(ts):
        for i, (rd, n, r, c) in enumerate(((rd0, n0, r0, "E0"), (rd3, n3, r3, "E3b"))):
            k = min(max(t - 1, 0), n - 1); ax = axes[i, j]; ax.imshow(rd.get_data(k)[40:420, 60:600]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(COL[status(r)]); sp.set_linewidth(2)
            if i == 0:
                ax.set_title(f"{lab}  t={t}", fontsize=8.5, color=fs.INK)
            if j == 0:
                ax.set_ylabel(f"{c}\n{status(r).replace('_', ' ')}", fontsize=9, color=fs.INK)
            side["frames"].append({"cond": c, "label": lab, "t": t, "mp4_frame": k, "mp4": str(m0 if c == "E0" else m3)})
    ch = (dv.get("by_threshold") or {}).get("0.1") or {}
    ch = {"gate_action": ch.get("E3b_gate_action"), "manipulated": ch.get("E3b_manipulated"), "max_violation_m": ch.get("E3b_max_violation_m")}
    fig.suptitle(f"ep {p['episode']} seed {p['key']}: E0 {status(r0)} / E3b {status(r3)}; |exec diff| > 1e-3 from step {dv.get('first_step_gt_tol')}, > 1e-1 from step "
                 f"{((dv.get('by_threshold') or {}).get('0.1') or {}).get('first_step')}; at that chunk E3b gate {ch.get('gate_action')}, manipulated {((ch.get('manipulated') or {}).get('nearest_centre'))}, "
                 f"max violation {ch.get('max_violation_m')}", fontsize=8.5, color=fs.INK)
    fig.patch.set_facecolor(fs.SURFACE); fig.tight_layout()
    save(fig, f"scene-{p['episode']}_{p['key']}.png", side, dpi=110)
    rd0.close(); rd3.close()

# ---- summary table
S = R["condition_summary"]
rows = []
def g(c, *ks):
    v = S.get(c, {})
    for k in ks:
        v = (v or {}).get(k) if isinstance(v, dict) else None
    return v
f1 = lambda x: "-" if x is None else (f"{x:.0f}" if isinstance(x, float) else str(x))
for lab, ks in (("runs", ("n_runs",)), ("success", ("n_success",)), ("grasp", ("n_grasp",)), ("place", ("n_place",)),
                ("HOLD chunks total", ("n_hold_chunks_total",)), ("runs with HOLD", ("n_runs_with_hold",)),
                ("round trip median ms", ("time_ms_pooled_excl_first", "client_round_trip", "median")),
                ("round trip p95 ms", ("time_ms_pooled_excl_first", "client_round_trip", "p95")),
                ("round trips > 533 ms", ("n_round_trip_gt_chunk_period",)),
                ("server infer median ms", ("time_ms_pooled_excl_first", "server_infer", "median")),
                ("server ag3s median ms", ("time_ms_pooled_excl_first", "server_ag3s", "median")),
                ("server trajopt median ms", ("time_ms_pooled_excl_first", "server_trajopt", "median")),
                ("wall s per run median", ("wall_s", "median"))):
    rows.append([lab, f1(g("E0", *ks)), f1(g("E3b", *ks))])
t = R["paired_2x2"]["success"]
rows.append(["pairs (both / E0 only / E3b only / neither)", "", f"{t['both']} / {t['E0_only']} / {t['E3b_only']} / {t['neither']}"])
rows.append(["exact McNemar p (success)", "", f"{t['mcnemar_exact_p']:.3g}"])
fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows) + 0.8)); ax.axis("off")
tb = ax.table(cellText=rows, colLabels=["", "E0", "E3b"], loc="center", cellLoc="center", colWidths=[0.5, 0.2, 0.3])
tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.3)
fig.patch.set_facecolor(fs.SURFACE); ax.set_title("T39 condition summary (first chunk excluded from times)", fontsize=10, color=fs.INK)
fig.tight_layout(); save(fig, "summary-table.png", {"rows": rows})
