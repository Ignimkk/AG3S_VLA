"""T43 HX graph: crossing time lines + per-run minima + servo-model accuracy.  usage: fig_timeline_hx.py OUT_PNG
Time-line panels (rows of one phase window of one run, x = control row t):
  P geo (= C, orange), M geo (blue), S geo (servo prediction, green dashed) = MuJoCo signed distance group <-> hurdle (mm);
  TO clearance (purple) = reconstructed TO row of the executing chunk at refined row k: d_esdf - r - margin (min over the group's
  query spheres); ESDF d - r (grey); margin applied to the binding sphere (black step); grey bands = HOLD rows.
Scatter: per run (window rows, executed), min P geo vs min M geo, per arm (colour) and group (marker).
CDF: |EE body (servo prediction) - EE body (M)| and |EE (C) - EE (M)| over executed carry-window rows (mm).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

out_png = pathlib.Path(sys.argv[1]); assert not out_png.exists(), out_png
R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/HX/raw'))
D = {a: json.load(open(R / f"hx_{a}.json"))["runs"] for a in ("V7g", "V8", "V5")}
figstyle.use_korean()
C = figstyle.CATEGORICAL
PANELS = [("V8", "V8/H/V8_ep1807_s18073", "carry", "fingers"), ("V8", "V8/H/V8_ep1807_s18073", "carry", "apple"),
          ("V8", "V8/H/V8_ep1807_s18073", "return", "fingers"), ("V7g", "V7g/H/V7g_ep1982_s19822", "carry", "apple")]
fig, axes = plt.subplots(2, 3, figsize=(17, 9.5)); ax_list = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]
side = {"script": __file__, "panels": []}
for ax, (arm, key, ph, g) in zip(ax_list, PANELS):
    rows = [r for r in D[arm][key]["rows"] if r["phase"] == ph and r["in_window"]]
    t = np.array([r["t"] for r in rows]); P = np.array([r["geo_C"][g] for r in rows]); M = np.array([r["geo_M"][g] for r in rows])
    S = np.array([(r.get("geo_S") or {}).get(g, np.nan) for r in rows], float)
    TO = np.array([((r.get("to") or {}).get(g) or {}).get("to_clear_mm", np.nan) for r in rows], float)
    DR = np.array([((r.get("to") or {}).get(g) or {}).get("d_minus_r_mm", np.nan) for r in rows], float)
    MG = np.array([((r.get("to") or {}).get(g) or {}).get("margin_mm", np.nan) for r in rows], float)
    hold = np.array([not r["executed"] for r in rows])
    for tt, h in zip(t, hold):
        if h:
            ax.axvspan(tt - 0.5, tt + 0.5, color=figstyle.GRID_INK, alpha=0.6, lw=0)
    ax.plot(t, P, "-o", color=C[1], ms=3, lw=1.4, label="P = C geometric")
    ax.plot(t, M, "-o", color=C[0], ms=3, lw=2.2, label="M geometric (measured)")
    ax.plot(t, S, "--", color=C[2], lw=1.3, label="S geometric (servo model)")
    ax.plot(t, TO, "-", color=C[6], lw=1.4, label="TO clearance (ESDF - r - margin)")
    ax.plot(t, DR, ":", color=figstyle.INK_2, lw=1.2, label="ESDF d - r")
    ax.step(t, MG, where="mid", color=figstyle.INK, lw=1, label="margin applied (binding sphere)")
    ax.axhline(0, color=C[7], lw=0.8); ax.axhline(10, color=C[7], lw=0.8, ls="--")
    ax.set_ylim(-15, 80); ax.set_xlabel("control row t"); ax.set_ylabel("distance to hurdle (mm)")
    ax.set_title(f"{key.split('/')[-1]} {ph} - {g}  (grey = HOLD rows)", fontsize=9); ax.legend(fontsize=6.5, loc="upper right")
    side["panels"].append({"arm": arm, "key": key, "phase": ph, "group": g, "t": t.tolist(), "P_geo_mm": P.tolist(), "M_geo_mm": M.tolist(),
                           "S_geo_mm": S.tolist(), "TO_clear_mm": TO.tolist(), "ESDF_d_minus_r_mm": DR.tolist(), "margin_mm": MG.tolist(),
                           "hold": hold.tolist()})
# scatter minima
ax = axes[1, 1]; sc = []
for ai, arm in enumerate(("V7g", "V8", "V5")):
    for g, mk in (("fingers", "o"), ("palm", "^"), ("apple", "s")):
        xs, ys = [], []
        for key, run in D[arm].items():
            rr = [r for r in run["rows"] if r["in_window"] and r["executed"] and r["phase"] == "carry" and g in r["geo_M"]]
            if rr:
                xs.append(min(r["geo_C"][g] for r in rr)); ys.append(min(r["geo_M"][g] for r in rr))
                sc.append({"arm": arm, "group": g, "key": key, "min_P_mm": xs[-1], "min_M_mm": ys[-1]})
        ax.scatter(xs, ys, marker=mk, color=C[ai], s=28, alpha=0.85, label=f"{arm} {g}")
ax.plot([-20, 120], [-20, 120], color=figstyle.GRID_INK, lw=1); ax.axhline(10, color=C[7], ls="--", lw=0.8); ax.axhline(0, color=C[7], lw=0.8)
ax.axvline(10, color=C[7], ls="--", lw=0.8)
ax.set_xlim(-5, 60); ax.set_ylim(-5, 60); ax.set_xlabel("min P geometric over carry window (mm)"); ax.set_ylabel("min M geometric over carry window (mm)")
ax.set_title("per run, carry window, executed rows", fontsize=9); ax.legend(fontsize=6, ncol=3, loc="upper left")
side["scatter"] = sc
# CDF servo
ax = axes[1, 2]; cd = {}
for ai, arm in enumerate(("V7g", "V8", "V5")):
    se = np.array([r["servo_err_mm"] for run in D[arm].values() for r in run["rows"] if r["in_window"] and r["executed"] and r["phase"] == "carry" and "servo_err_mm" in r])
    ce = np.array([r["cmd_err_mm"] for run in D[arm].values() for r in run["rows"] if r["in_window"] and r["executed"] and r["phase"] == "carry" and "cmd_err_mm" in r])
    for v, ls, lab in ((se, "--", "servo model S"), (ce, "-", "command C")):
        v = np.sort(v); ax.plot(np.maximum(v, 1e-3), np.arange(1, len(v) + 1) / len(v), ls, color=C[ai], lw=1.5, label=f"{arm} {lab} (n={len(v)})")
    cd[arm] = {"servo_err_mm_sorted": np.sort(se).tolist(), "cmd_err_mm_sorted": np.sort(ce).tolist()}
ax.set_xscale("log"); ax.set_xlabel("|EE body - EE body of M| (mm)"); ax.set_ylabel("fraction of rows"); ax.legend(fontsize=6.5)
ax.set_title("prediction error of M: servo model vs command (carry window)", fontsize=9)
side["cdf"] = cd
figstyle.style_axes(fig, list(axes.reshape(-1)))
fig.suptitle("T43 HX - distance to the hurdle while crossing: plan P (= command C), measured M, servo prediction S, TO ESDF clearance", fontsize=11)
fig.tight_layout(); fig.savefig(out_png, dpi=125)
json.dump(side, open(out_png.with_suffix(".json"), "w"))
print("wrote", out_png)
