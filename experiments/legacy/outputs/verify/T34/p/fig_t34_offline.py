"""T34 verifier figures from offline + closed-loop numbers (no new measurement).
  t34-band-thickness.png  : table zero-distance band (d <= 0.1 mm) above the TLS plane per run, fine / coarse / combined layer,
                            'table' columns, median (bar) and p90 (tick); T33 records and T34 records; plus bias h - d(h) at +20 mm.
                            Source off/band_T33_{E3a,E3b}.json, off/band_T34_{E3a,E3b}.json (band.py).
  t34-j1-offline-table.png: T33 attach 9 events replayed with the T34 code: sphere bottom above the TLS plane (snapshot /
                            attach pose), lift, band, held-row min at the attach chunk and over the T33 held chunks, T33 vs T34
                            spheres; J2 replay revoke t. Source off/offline_j1j2.json.
  t34-held-rows.png       : closed-loop T34: per held chunk min held query row (refined, recorded field) per run with the GT body
                            at the field's nearest-surface point; offline T33 records with T33 vs T34 spheres. Sources
                            p/{E3a,E3b}_held_rows.json, off/offline_j1j2.json.
usage: fig_t34_offline.py [band] [table] [rows]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs
fs.use_korean()
C = fs.CATEGORICAL
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T34'))
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t34'))
WHAT = sys.argv[1:] or ["band", "table", "rows"]


def new(n):
    p = F / n
    assert not p.exists(), p
    return p


if "band" in WHAT:
    B = {}
    for tag in ("T33", "T34"):
        for c in ("E3a", "E3b"):
            for r, v in json.loads((V / "off" / f"band_{tag}_{c}.json").read_text())["runs"].items():
                B[f"{tag} {r}"] = v
    names = sorted(B)
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    side = {"source": ["outputs/verify/T34/off/band_T33_E3a.json", "outputs/verify/T34/off/band_T33_E3b.json",
                       "outputs/verify/T34/off/band_T34_E3a.json", "outputs/verify/T34/off/band_T34_E3b.json"],
            "definition": json.loads((V / "off" / "band_T34_E3a.json").read_text())["doc"], "runs": {}}
    x = np.arange(len(names))
    for ai, (key, lab) in enumerate((("above_mm", "zero band top above the TLS plane (mm)"), ("bias_at_20mm_mm", "20 mm - d(+20 mm) (mm)"))):
        ax = axes[ai]
        for li, layer in enumerate(("fine", "coarse", "combined")):
            med = [B[n][f"table/{layer}"][key]["median"] for n in names]
            p90 = [B[n][f"table/{layer}"][key]["p90"] for n in names]
            ax.bar(x + (li - 1) * 0.27, med, 0.26, color=C[[0, 4, 7][li]], label=f"{layer} median")
            ax.plot(x + (li - 1) * 0.27, p90, "_", color=fs.INK, ms=9, mew=1.5, label="p90" if li == 0 else None)
            for n, a, b in zip(names, med, p90):
                side["runs"].setdefault(n, {}).setdefault(layer, {})[key] = {"median": a, "p90": b, "n_columns": B[n][f"table/{layer}"]["n_columns"]}
        ax.set_ylabel(lab); ax.legend(fontsize=8, ncol=4)
        ax.axhline(8.8, color=fs.INK_2, lw=0.8, ls=":")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[0].set_title("Table zero-distance band per run ('table' columns >= 90 mm from fruit, eikonal-checked), every 4th chunk; dotted = 8.8 mm", fontsize=10)
    fig.suptitle("T34 band thickness: fine (5 mm) vs coarse (20 mm) vs combined (min, what the TO reads) — T33 and T34 records", fontsize=11)
    fs.style_axes(fig, axes.ravel()); fig.tight_layout()
    p = new("t34-band-thickness.png"); fig.savefig(p, dpi=120); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps(side, indent=1)); print(p)

if "table" in WHAT:
    O = json.loads((V / "off" / "offline_j1j2.json").read_text())
    cols = ["event (T33 record)", "radius mm", "band q90 mm", "lift mm", "bottom-plane snap T33 / T34", "bottom-plane attach T33 / T34",
            "held row attach chunk T33 / T34", "held row min over T33 held chunks T33 / T34", "n held chunks row<0 T33 / T34",
            "J2 revoke t (default | obs only)", "T33 detach t"]
    cell, side = [], {"source": "outputs/verify/T34/off/offline_j1j2.json", "rows": {}}
    for k, e in O["j1"].items():
        run = k.split("@")[0]; s = e["support_record"]; rc = e["rows_per_chunk"]
        t33 = [x["row_min_mm_T33"] for x in rc]; t34 = [x["row_min_mm_T34"] for x in rc]
        j2 = O["j2"][run]
        row = [k, f"{e['radii_mm']['T34'][0]:.1f}", f"{(s.get('band') or {}).get('band_q10_q50_q90_mm', [None]*3)[2]}",
               f"{s['spheres'][0]['lift_mm']:.1f}",
               f"{e['bottom_above_tls_plane_mm_snapshot_T33'][0]:+.1f} / {e['bottom_above_tls_plane_mm_snapshot_T34'][0]:+.1f}",
               f"{e['bottom_above_tls_plane_mm_attach_T33'][0]:+.1f} / {e['bottom_above_tls_plane_mm_attach_T34'][0]:+.1f}",
               f"{e['attach_chunk_row_min_mm']['T33']:+.1f} / {e['attach_chunk_row_min_mm']['T34']:+.1f}",
               f"{min(t33):+.1f} / {min(t34):+.1f}", f"{sum(v < 0 for v in t33)}/{len(rc)}  |  {sum(v < 0 for v in t34)}/{len(rc)}",
               f"{j2['default']['revoke_t'] or '-'} | {j2['observation_only']['revoke_t'] or '-'}", f"{j2['default']['T33_detach_t'] or '-'}"]
        cell.append(row); side["rows"][k] = dict(zip(cols, row))
    fig, ax = plt.subplots(figsize=(22, 0.55 * len(cell) + 2)); ax.axis("off")
    tb = ax.table(cellText=cell, colLabels=cols, loc="upper center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.8)
    for (r_, c_), cl in tb.get_celld().items():
        if r_ == 0:
            cl.set_text_props(fontsize=7.5); cl.set_height(cl.get_height() * 1.6)
    ax.set_title("T34 offline replay on the 9 T33 attaches: primary held sphere bottom vs TLS table plane (mm), held<->field row (d - r, mm; q_now + reference rows 0-7), "
                 "J2 revoke t (GraspLatch replay on T33 signals)", fontsize=10, color=fs.INK)
    fig.patch.set_facecolor(fs.SURFACE); fig.tight_layout()
    p = new("t34-j1-offline-table.png"); fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps(side, indent=1)); print(p)

if "rows" in WHAT:
    BODY_COL = {"apple": C[1], "crate": C[0], "table": fs.INK, "banana": C[4], "orange": C[3], "pear": C[5]}
    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    side = {"closed_loop": {}, "offline": {}}
    ax = axes[0]
    k = 0
    for c in ("E3a", "E3b"):
        H = json.loads((V / "p" / f"{c}_held_rows.json").read_text())["runs"]
        for run, lst in sorted(H.items()):
            if not lst:
                continue
            ts = [x["t"] for x in lst]; vs = [x["refined_query_min_mm"] for x in lst]
            ax.plot(ts, vs, "-", color=C[k % 8], lw=0.9, alpha=0.7, label=run)
            for x in lst:
                b = (x.get("worst_refined_query") or {}).get("surface_gt_nearest_body")
                ax.plot(x["t"], x["refined_query_min_mm"], "o", ms=4, color=BODY_COL.get(b, fs.INK_2), mec=C[k % 8], mew=0.6)
            side["closed_loop"][run] = [{"t": x["t"], "row_mm": x["refined_query_min_mm"], "surface_body": (x.get("worst_refined_query") or {}).get("surface_gt_nearest_body")} for x in lst]
            k += 1
    ax.axhline(0, color=fs.INK, lw=0.8); ax.set_ylim(-30, 140)
    ax.set_xlabel("control t_step"); ax.set_ylabel("min held query row d - r (mm)")
    ax.set_title("T34 closed loop: min held query row per attached chunk (refined rows 0-7 + q_now, recorded field); dot fill = GT body at the field's nearest surface "
                 "(orange apple, blue crate, black table)", fontsize=9)
    ax.legend(fontsize=7, ncol=5)
    ax = axes[1]
    O = json.loads((V / "off" / "offline_j1j2.json").read_text())["j1"]
    for i, (kk, e) in enumerate(O.items()):
        rc = e["rows_per_chunk"]
        ts = [x["t"] for x in rc]
        ax.plot(ts, [x["row_min_mm_T33"] for x in rc], "--", color=C[i % 8], lw=0.9)
        ax.plot(ts, [x["row_min_mm_T34"] for x in rc], "-", color=C[i % 8], lw=1.4, label=kk)
        side["offline"][kk] = {"t": ts, "T33": [x["row_min_mm_T33"] for x in rc], "T34": [x["row_min_mm_T34"] for x in rc]}
    ax.axhline(0, color=fs.INK, lw=0.8); ax.set_ylim(-30, 140)
    ax.set_xlabel("control t_step"); ax.set_ylabel("min held row d - r (mm)")
    ax.set_title("Offline on T33 records (T33 held chunks): T33 spheres dashed vs T34 lifted spheres solid (q_now + reference rows 0-7)", fontsize=9)
    ax.legend(fontsize=7, ncol=5)
    fs.style_axes(fig, axes.ravel()); fig.tight_layout()
    p = new("t34-held-rows.png"); fig.savefig(p, dpi=120); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps(side, indent=1)); print(p)
