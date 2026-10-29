"""T33 offline figures (a) held-apple points after H1, (e) reference limit overshoot URDF vs MJCF. usage: fig_offline.py a|e ..."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs
KO = fs.use_korean()
C = fs.CATEGORICAL
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t33'))
O = pathlib.Path(_ag3s_workspace_path('outputs/verify/T33/off'))
WHAT = sys.argv[1:]


def save(fig, name, side):
    p = F / name
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float)); print(p)


if "a" in WHAT:
    a = json.load(open(O / "a/offline_a.json"))
    runs = list(a["runs"])
    fig, axes = plt.subplots(len(runs), 1, figsize=(12, 2.6 * len(runs)), sharex=False, squeeze=False)
    side = {"source": "outputs/verify/T33/off/a/offline_a.json", "definitions": a["definitions"], "runs": {}}
    for i, r in enumerate(runs):
        e = a["runs"][r]; ax = axes[i, 0]
        ch = [x for x in e["chunks"] if x["held_frame"]]
        t = [x["t"] for x in ch]
        ax.plot(t, [x["n_apple_5mm"] for x in ch], "-o", ms=3, color=C[0], label="GT apple pts in recorded cloud (T30 filter)")
        ax.plot(t, [x["n_apple_5mm_left_query_only"] for x in ch], "-o", ms=3, color=C[3], label="left after held query spheres (fit+5 mm)")
        ax.plot(t, [x["n_apple_5mm_left_H1"] for x in ch], "-o", ms=3, color=C[7], label="left after H1 self-filter spheres (fit+25 mm)")
        out = [x["t"] for x in ch if not x["in_hand"]]
        if out:
            ax.axvspan(min(out) - 4, max(out) + 4, color=fs.GRID_INK, alpha=0.5, lw=0)
        for tt, lab in ((e["grasp_t"], "grasp"), (e["place_t"], "place"), (e["release_t"], "release")):
            if tt is not None:
                ax.axvline(tt, color=fs.INK_2, lw=0.8, ls=":"); ax.text(tt, ax.get_ylim()[1] if False else 1, lab, fontsize=7, color=fs.INK_2, rotation=90, va="bottom")
        s = e["sum_grasp_to_release"]
        ax.set_title(f"{r} (T30 record): attach t={e['t_attach']}, held spheres {[round(x,1) for x in e['held_filter_radii_mm']]} mm (filter) — in hand grasp->release: "
                     f"{s['apple_5mm_left_H1']}/{s['apple_5mm']} apple pts left, {s['chunks_with_any_left_5mm']}/{s['n_chunks']} chunks with any; grey = apple out of hand (>30 mm drift)", fontsize=8.5)
        ax.set_ylabel("points"); ax.set_xlabel("t_step")
        side["runs"][r] = {"t": t, "n_apple_5mm": [x["n_apple_5mm"] for x in ch], "left_query_only": [x["n_apple_5mm_left_query_only"] for x in ch],
                           "left_H1": [x["n_apple_5mm_left_H1"] for x in ch], "in_hand": [x["in_hand"] for x in ch],
                           "drift_mm": [x["apple_offset_drift_palm_mm"] for x in ch], "sum_grasp_to_release": s, "last3": e["sum_last3_in_hand_chunks"]}
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("T33 offline (a): H1 self-filter (held spheres fit+25 mm on ee_left, H2 fit at capture pose) applied to recorded T30 clouds — GT held-apple points left", fontsize=10.5)
    fs.style_axes(fig, axes.ravel()); fig.tight_layout()
    save(fig, "t33-off-a-held-points.png", side)

if "e" in WHAT:
    e = json.load(open(O / "e/offline_e.json"))
    per = e["runs"]
    names = list(per)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]; x = np.arange(len(names))
    u = [per[n]["urdf"]["n_rows_overshoot"].get("position", 0) for n in names]
    m = [per[n]["model_xml"]["n_rows_overshoot"].get("position", 0) for n in names]
    ax.bar(x - 0.2, u, 0.4, color=C[1], label="URDF box (T30 server)")
    ax.bar(x + 0.2, m, 0.4, color=C[0], label="MJCF box (T31 G1)")
    ax.set_xticks(x); ax.set_xticklabels([n.replace("/", " ").replace("_ep", " ep").replace("_r", " r") for n in names], fontsize=7, rotation=60, ha="right")
    ax.set_ylabel("planning rows whose reference (8-step window) leaves the position box"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("per run (75 rows each)", fontsize=9)
    ax = axes[1]; ax.axis("off")
    agg = {}
    for n, v in per.items():
        tag, run = n.split("/"); k = f"{tag} {run.split('_')[0]}"
        g = agg.setdefault(k, [0, 0, 0, {}, 0.0, 0.0])
        g[0] += v["n_rows"]; g[1] += v["urdf"]["n_rows_overshoot"].get("position", 0); g[2] += v["model_xml"]["n_rows_overshoot"].get("position", 0)
        for j, c in v["model_xml"]["position_overshoot_joint_counts"].items():
            g[3][j] = g[3].get(j, 0) + c
        g[4] = max(g[4], v["urdf"]["max_overshoot_deg"].get("position", 0)); g[5] = max(g[5], v["model_xml"]["max_overshoot_deg"].get("position", 0))
    cell = [[k, g[0], g[1], f"{g[4]:.2f}", g[2], f"{g[5]:.2f}", ", ".join(f"{j} {c}" for j, c in g[3].items()) or "-"] for k, g in agg.items()]
    tb = ax.table(cellText=cell, colLabels=["records", "rows", "URDF\nrows", "URDF\nmax deg", "MJCF\nrows", "MJCF\nmax deg", "MJCF joints"], loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 2.2); tb.auto_set_column_width(list(range(7)))
    fig.suptitle("T33 offline (e): G1 — policy reference chunk vs joint position box (margin 0.02 rad), URDF vs MJCF model_transport.xml, recorded E3 runs", fontsize=10.5)
    fs.style_axes(fig, [axes[0]]); fig.tight_layout()
    save(fig, "t33-off-e-limit-overshoot.png", {"source": "outputs/verify/T33/off/e/offline_e.json", "definitions": e["definitions"],
                                                  "per_run_position_rows": {n: {"urdf": u[i], "mjcf": m[i]} for i, n in enumerate(names)},
                                                  "by_condition": {k: {"rows": g[0], "urdf_rows": g[1], "urdf_max_deg": g[4], "mjcf_rows": g[2], "mjcf_max_deg": g[5], "mjcf_joints": g[3]} for k, g in agg.items()}})

if "d" in WHAT:
    import glob
    s = json.load(open(O / "d/summ_d.json")); b = json.load(open(O / "d/boundary_cases.json"))
    P = O / "d/parts"
    G = {v: {"hidden": [], "visible": []} for v in ("pipe", "npz")}; pairs = []
    for var in ("pipe", "npz"):
        for f in sorted(P.glob(f"{var}__*.json")):
            for e in json.load(open(f))["rows"]:
                if e["frozen"] or not e["phase"].startswith("pre"):
                    continue
                g = e.get(f"hand_gap_anchor_pre_{var}_mm")
                if g is None:
                    continue
                v = e[var]; hid = v.get("state") == "occluded" or v.get("reason") == "subset_kept_anchor"
                G[var]["hidden" if hid else "visible"].append(g)
                if var == "pipe" and e.get("hand_gap_anchor_pre_npz_mm") is not None:
                    pairs.append((e["hand_gap_anchor_pre_npz_mm"], g, hid))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    bins = np.arange(-5, 105, 2.5)
    for ax, var, ttl in ((axes[0], "pipe", "hand = AG3S._hand_for_grounding (constraint model at q_now)"), (axes[1], "npz", "hand = recorded sphere_centres (T32b calibration)")):
        ax.hist(np.clip(G[var]["hidden"], -5, 102.4), bins=bins, color=C[1], alpha=0.8, label=f"anchor hidden (occluded / subset_kept_anchor) n={len(G[var]['hidden'])}")
        ax.hist(np.clip(G[var]["visible"], -5, 102.4), bins=bins, color=C[0], alpha=0.6, label=f"visible n={len(G[var]['visible'])}")
        ax.axvline(30, color=C[7], lw=1.5); ax.text(30.5, ax.get_ylim()[1] * 0.9, "reach 30 mm", color=C[7], fontsize=8)
        sv = s["variants"][var]
        ax.set_title(f"{ttl}\nhidden <=30 mm: {sv['hidden']['n_le_30']}/{sv['hidden']['n']}, visible <=30 mm: {sv['visible']['n_le_30']}/{sv['visible']['n']}, switches {sv['n_switches']}", fontsize=8.5)
        ax.set_xlabel("hand-sphere surface to anchor gap before decision (mm, clipped at 100)"); ax.set_ylabel("pre-grasp non-frozen chunks")
        ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    pa = np.array([(x, y) for x, y, h in pairs]); hh = np.array([h for *_, h in pairs])
    ax.scatter(pa[~hh, 0], pa[~hh, 1], s=6, color=C[0], label="visible"); ax.scatter(pa[hh, 0], pa[hh, 1], s=6, color=C[1], label="hidden")
    ax.plot([0, 120], [0, 120], color=fs.INK_2, lw=0.8); ax.axhline(30, color=C[7], lw=0.8); ax.axvline(30, color=C[7], lw=0.8)
    for k, key in ((0, "T30_E3a_ep1800_r3/npz"),):
        for r in b[key]:
            if r["t"] in (416, 424, 432):
                ax.annotate(f"ep1800 t={r['t']}", (r["gap_npz"], r["gap_pipe"]), fontsize=7, color=C[7])
    ax.set_xlim(-5, 120); ax.set_ylim(-5, 120)
    d = s["pipe_minus_npz_gap_mm"]
    ax.set_title(f"pipeline gap vs recorded-sphere gap (mm), same chunk\npipe - npz: median {d['median']:.1f}, q05 {d['q05']:.1f}, q95 {d['q95']:.1f} mm", fontsize=8.5)
    ax.set_xlabel("gap with recorded sphere_centres (mm)"); ax.set_ylabel("gap with _hand_for_grounding (mm)"); ax.legend(fontsize=7, frameon=False)
    fig.suptitle("T33 offline (d): S3 boundary — hand gap distribution against the 30 mm reach, T28+T30 E3 replays (27 runs), current tree grounding (S2+S3 on)", fontsize=10.5)
    fs.style_axes(fig, axes); fig.tight_layout()
    save(fig, "t33-off-d-s3-hand-gap.png", {"source": ["outputs/verify/T33/off/d/summ_d.json", "outputs/verify/T33/off/d/boundary_cases.json"],
                                            "summary": {v: {k: x for k, x in s["variants"][v].items() if k != "switches"} for v in s["variants"]},
                                            "pipe_minus_npz_gap_mm": d, "boundary_cases": b})

if "bc" in WHAT:
    from matplotlib.patches import Rectangle
    bc = json.load(open(O / "bc/bc_E3b_ep1807_r1.json")); S = np.load(O / "bc/bc_E3b_ep1807_r1.slices.npz")
    V = bc["variants"]; seqs_sl = sorted({int(k.split("/")[1][3:]) for k in S.files})
    # crate walls in the crate frame (half sizes from the model; crate is axis-aligned with base in these records, checked by walls ~ x/y lines)
    WALLS = {"nx": (-0.086, 0, 0.004, 0.004, 0.15, 0.056), "px": (0.086, 0, 0.004, 0.004, 0.15, 0.056), "ny": (0, -0.146, 0.004, 0.09, 0.004, 0.056),
             "py": (0, 0.146, 0.004, 0.09, 0.004, 0.056), "floor": (0, 0, -0.056, 0.09, 0.15, 0.004)}
    cm = fs.sequential_cmap()
    # (b) slices: seq 37 (t=288, over the crate) and 38 (release) — T32 vs T32_nofree vs difference, fine tier, xz and xy
    for sq in [s for s in (37, 38) if s in seqs_sl]:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
        side = {"seq": sq, "source": "outputs/verify/T33/off/bc/bc_E3b_ep1807_r1.slices.npz", "planes": {}}
        e0 = next(e for e in V["T32"] if e["seq"] == sq)
        for i, pl in enumerate(("xz", "xy")):
            a = S[f"T32/seq{sq:03d}/fine/{pl}"]; b = S[f"T32_nofree/seq{sq:03d}/fine/{pl}"]
            ext = [-0.2, 0.2, -0.2 + (0.004 if pl == "xz" else 0), 0.2 + (0.004 if pl == "xz" else 0)]
            for j, (arr, ttl) in enumerate(((a, "T32 (H3 free on)"), (b, "T32_nofree"), (None, "d<0 in one only"))):
                ax = axes[i, j]
                if arr is not None:
                    ax.imshow(np.clip(arr.T, 0, 0.1), origin="lower", extent=ext, cmap=cm, vmin=0, vmax=0.1)
                    ax.imshow(np.where(arr.T < 0, 1.0, np.nan), origin="lower", extent=ext, cmap=matplotlib.colors.ListedColormap([C[7]]), alpha=0.6)
                else:
                    only_nf = (b < 0) & ~(a < 0); only_f = (a < 0) & ~(b < 0)
                    ax.imshow(np.where(only_nf.T, 1.0, np.nan), origin="lower", extent=ext, cmap=matplotlib.colors.ListedColormap([C[1]]))
                    ax.imshow(np.where(only_f.T, 1.0, np.nan), origin="lower", extent=ext, cmap=matplotlib.colors.ListedColormap([C[0]]))
                    ttl += f": nofree-only {int(only_nf.sum())} px (orange), free-only {int(only_f.sum())} px (blue), 2.5 mm px"
                    side["planes"][pl] = {"n_lt0_T32": int((a < 0).sum()), "n_lt0_nofree": int((b < 0).sum()), "nofree_only": int(only_nf.sum()), "free_only": int(only_f.sum())}
                for wn, (cx, cy, cz, hx, hy, hz) in WALLS.items():
                    if pl == "xz":
                        if abs(cy) - hy <= 0 <= abs(cy) + hy or wn in ("nx", "px", "floor"):
                            ax.add_patch(Rectangle((cx - hx, cz - hz), 2 * hx, 2 * hz, fill=False, ec=C[2], lw=1.0))
                    else:
                        if cz - hz <= 0.004 <= cz + hz:
                            ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, fill=False, ec=C[2], lw=1.0))
                ax.set_title(f"{pl} fine, {ttl}", fontsize=8.5); ax.set_xlabel("x - crate x (m)"); ax.set_ylabel(("z" if pl == "xz" else "y") + " - crate (m)")
        fig.suptitle(f"T33 offline (b): H3 free vs no free — E3b_ep1807_r1 (T30 record rebuilt with the current builder), seq {sq} t={e0['t']}; green = GT crate collision boxes; "
                     f"slices through the GT crate centre (xz at crate y, xy at crate z + 4 mm)", fontsize=10)
        fs.style_axes(fig, axes.ravel()); fig.tight_layout()
        save(fig, f"t33-off-b-crate-free-slices-seq{sq:03d}.png", side)
    # (b)+(c) series
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    side = {"source": "outputs/verify/T33/off/bc/bc_E3b_ep1807_r1.json", "series": {}}
    for v, col in (("T32", C[0]), ("T32_nofree", C[1]), ("T32_trunc60", C[4])):
        L = [e for e in V[v] if e["seq"] >= bc["attach_seq"] - 1]
        t = [e["t"] for e in L]
        occ = [e["fine_crate_occupied"] for e in L]
        axes[0].plot(t, occ, "-o", ms=3, color=col, label=v)
        for k, ls in (("wall_x_fine", "-"), ("wall_y_fine", "--")):
            lo = [e[k].get("width_lt0_mm_lo") if e[k].get("covered") else np.nan for e in L]
            hi = [e[k].get("width_lt0_mm_hi") if e[k].get("covered") else np.nan for e in L]
            axes[1 if k == "wall_x_fine" else 2].plot(t, lo, ls, color=col, lw=1.4, label=f"{v} {'nx' if k == 'wall_x_fine' else 'ny'}")
            axes[1 if k == "wall_x_fine" else 2].plot(t, hi, ls, color=col, lw=0.8, marker=".", label=f"{v} {'px' if k == 'wall_x_fine' else 'py'}")
            side["series"].setdefault(v, {})[k] = {"t": t, "lo": lo, "hi": hi}
        side["series"][v]["fine_crate_occupied"] = occ
    axes[0].set_title("fine occupied voxels classified crate (GT within 30 mm)", fontsize=9); axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_title("crate walls nx (thick) / px (thin): fine d<0 width on the x line (mm); GT wall 8 mm", fontsize=9)
    axes[2].set_title("crate walls ny (thick) / py (thin): fine d<0 width on the y line (mm)", fontsize=9)
    for ax in axes[1:]:
        ax.axhline(8, color=C[2], lw=0.8, ls=":"); ax.legend(fontsize=6.5, frameon=False, ncol=2); ax.set_ylabel("mm")
    for ax in axes:
        ax.set_xlabel("t_step"); ax.axvline(bc["attach_t"], color=fs.INK_2, lw=0.7, ls=":")
    fig.suptitle("T33 offline (b)(c): E3b_ep1807_r1 rebuilt — H3 free (T32) vs no free vs 60 mm fine truncation; attach dotted", fontsize=10.5)
    fs.style_axes(fig, axes); fig.tight_layout()
    save(fig, "t33-off-bc-crate-walls-series.png", side)

if "c" in WHAT:
    t = json.load(open(O / "bc/bc_tables.json")); w = json.load(open(O / "bc/walls_recorded_summary.json"))
    fig, axes = plt.subplots(2, 1, figsize=(16, 9.5), gridspec_kw={"height_ratios": [1, 1.5]})
    ax = axes[0]; ax.axis("off")
    cell = []
    for ck, v in t["h4_chunks"].items():
        for var in ("T32_trunc60", "T32"):
            f = v[var]["fine"]; co = v[var]["coarse"]; y = v[var]["y_fine"]
            cell.append([ck, {"T32_trunc60": "fine trunc 60 mm (pre-fix)", "T32": "fine trunc 30 mm (H4-fix)"}[var],
                         f["width_lt0_mm"], f["width_lt0_mm_lo"], f["width_lt0_mm_hi"], f"{f['min_mm_hi']:.1f}",
                         "-" if not y.get("covered") else f"{y['width_lt0_mm_lo']} / {y['width_lt0_mm_hi']}", co["width_lt0_mm_lo"], co["width_lt0_mm_hi"]])
    tb = ax.table(cellText=cell, colLabels=["T30 E3b ep1807 r1", "variant", "fine x total", "fine nx", "fine px", "fine px min (mm)", "fine ny / py", "coarse nx", "coarse px"], loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.45)
    ax.set_title("rebuild (current builder, T32 H1/H3 on): crate wall d<0 widths (mm) on lines through the GT crate centre; GT wall 8 mm", fontsize=9.5)
    ax = axes[1]; ax.axis("off")
    cell = []
    for run, s in w.items():
        row = [run]
        for k in ("fine_nx", "fine_px", "fine_ny", "fine_py", "coarse_nx", "coarse_px"):
            x = s[k]; row.append("-" if not x["n_covered"] else f"{x['median_width_mm']:.1f}  ({x['n_zero_width']}/{x['n_covered']} zero)")
        cell.append(row)
    tb = ax.table(cellText=cell, colLabels=["T33 run (recorded fields)", "fine nx", "fine px", "fine ny", "fine py", "coarse nx", "coarse px"], loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.45)
    ax.set_title("T33 closed-loop records: median d<0 width (mm) per wall over chunks whose window covers the line (zero = no d<0 on that half-line; fine/free: min distance on every zero half-line = 0.0 mm; coarse nx zero half-lines: largest min distance 14.0 mm (E3a ep1800 r1), 2.5 mm (E3b ep1800 r1))", fontsize=9.5)
    fig.suptitle("T33 offline (c): H4-fix fine truncation 30 mm — crate wall bands", fontsize=11)
    fig.patch.set_facecolor(fs.SURFACE); fig.tight_layout()
    save(fig, "t33-off-c-wall-bands-table.png", {"rebuild": t["h4_chunks"], "recorded": w})
