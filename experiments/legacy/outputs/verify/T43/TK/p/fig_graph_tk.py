"""T43 TK graph figure.  usage: fig_graph_tk.py OUT_PNG
(a)(b) time line of EE_BODY world y and z of P (refined row), C (applied command) and M (measured post state) around a contact row
       (V8_ep1807_s18071 t=206 apple carry; V8_ep1807_s18073 t=270 ee_finger_l2 after release); dotted vertical = chunk starts,
       solid vertical = contact row.
(c) |EE(C) - EE(M)| vs commanded EE speed, exec rows, per arm: median per speed bin (raw/tk_summary.json speed_bins_exec)
    + scatter of carry rows (V8).  Reference lines: speed x ramp lag (kv+b)/kp of the model (min / max over the 7 arm joints).
(d) distribution of the time offset tau_ee (rows, moving exec rows: speed_cmd >= 3 mm/row, t >= 7) per arm.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from AG3S.ag3s.experiments.common import figstyle

out = pathlib.Path(sys.argv[1]); assert not out.exists(), out
R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/TK/raw'))
J = {a: json.load(open(R / f"tk_{a}.json")) for a in ("V7g", "V8", "E0")}
S = json.load(open(R / "tk_summary.json"))
kor = figstyle.use_korean()
CAT = figstyle.CATEGORICAL
fig = plt.figure(figsize=(13, 12)); gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.6])
side = {"script": __file__, "panels": {}}
all_axes = []
for col, (key, t) in enumerate((("V8/H/V8_ep1807_s18071", 206), ("V8/H/V8_ep1807_s18073", 270))):
    run = J["V8"]["runs"][key]; rows = run["rows"]; ts = list(range(max(0, t - 22), min(len(rows), t + 3)))
    P = np.array([rows[i]["EE_P"] if rows[i]["EE_P"] else [np.nan] * 3 for i in ts]); C = np.array([rows[i]["EE_C"] for i in ts]); M = np.array([rows[i]["EE_M"] for i in ts])
    ref = np.array(rows[t]["EE_M"])
    c = next(x for x in run["contacts"] if x["t"] == t)
    for jj, cn in enumerate("xyz"):
        ax = fig.add_subplot(gs[jj, col]); all_axes.append(ax)
        for i in ts:
            if rows[i]["cls"] == "hold":
                ax.axvspan(i - 0.5, i + 0.5, color=figstyle.GRID_INK, alpha=0.35, lw=0)
            if rows[i]["k"] == 0:
                ax.axvline(i, color=figstyle.GRID_INK, ls=":", lw=1)
        ax.plot(ts, 1e3 * (P[:, jj] - ref[jj]), "-", color=CAT[1], lw=2, label="P refined row")
        ax.plot(ts, 1e3 * (C[:, jj] - ref[jj]), "x", color=CAT[2], ms=6, label="C applied")
        ax.plot(ts, 1e3 * (M[:, jj] - ref[jj]), "-", color=CAT[0], lw=2, label="M measured")
        ax.axvline(t, color=figstyle.INK, lw=1.2)
        ax.set_ylabel(f"EE {cn} - M_{cn}(t) (mm)", fontsize=8)
        if jj == 0:
            ax.set_title(f"{key.split('/')[-1]} contact t={t} ({c['kind']}:{c['body']}); shaded = HOLD rows, dotted = chunk start\n"
                         f"bar distance: P {c['bar_P_mm']:+.1f}  C {c['bar_C_mm']:+.1f}  M {c['bar_M_mm']:+.1f} mm;  |C-M| {c['CM_mm']:.1f} mm", fontsize=9)
            ax.legend(fontsize=7, ncol=3)
        if jj == 2:
            ax.set_xlabel("control row t (0.066 s)")
    side["panels"][key] = {"t": t, "rows": ts, "EE_P": P.tolist(), "EE_C": C.tolist(), "EE_M": M.tolist(), "ref_EE_M_t": ref.tolist(),
                           "hold_rows": [i for i in ts if rows[i]["cls"] == "hold"],
                           "chunk_start_rows": [i for i in ts if rows[i]["k"] == 0], "contact": c}
axes = np.array([[None, None], [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]], dtype=object)
all_axes += list(axes[1])
ax = axes[1, 0]
lag = [r["ramp_lag_steps"] for r in J["V8"]["model"]["left_arm"]]
for n, arm in enumerate(("V7g", "V8", "E0")):
    ex = [r for k, v in J[arm]["runs"].items() for r in v["rows"] if r["cls"] == "exec" and r["speed_cmd_mm"] is not None]
    xs = np.array([r["speed_cmd_mm"] for r in ex]); ys = np.array([r["CM_mm"] for r in ex])
    edges = np.array([0, 1, 3, 6, 10, 15, 20, 30, 45, 120]); mx, my, mp = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (xs >= lo) & (xs < hi)
        if k.sum() >= 5:
            mx.append(float(np.median(xs[k]))); my.append(float(np.median(ys[k]))); mp.append(float(np.percentile(ys[k], 90)))
    ax.plot(mx, my, "o-", color=CAT[n], label=f"{arm} median"); ax.plot(mx, mp, ":", color=CAT[n], label=f"{arm} p90")
    side["panels"][f"speed_{arm}"] = {"bin_edges_mm": edges.tolist(), "median_speed": mx, "median_CM": my, "p90_CM": mp}
    if arm == "V8":
        kc = np.array([r["phase"] == "carry" for r in ex]); ax.scatter(xs[kc], ys[kc], s=4, color=CAT[n], alpha=0.25, label="V8 carry rows")
xx = np.linspace(0, 60, 10)
ax.plot(xx, min(lag) * xx, color=figstyle.INK_2, lw=1, ls="--", label=f"speed x (kv+b)/kp, {min(lag):.2f} rows")
ax.plot(xx, max(lag) * xx, color=figstyle.INK_2, lw=1, ls="-.", label=f"speed x (kv+b)/kp, {max(lag):.2f} rows")
ax.set_xlim(0, 60); ax.set_ylim(0, 140); ax.set_xlabel("commanded EE speed |EE(C_t) - EE(C_t-1)| (mm/row)"); ax.set_ylabel("|EE(C) - EE(M)| (mm)")
ax.set_title("C - M vs speed (exec rows)", fontsize=9); ax.legend(fontsize=7, ncol=2)
side["panels"]["ramp_lag_rows_joint"] = lag
ax = axes[1, 1]
bins = np.arange(0, 6.01, 0.5)
for n, arm in enumerate(("V7g", "V8", "E0")):
    v = [r["tau_ee"] for k, run in J[arm]["runs"].items() for r in run["rows"] if r["cls"] == "exec" and (r["speed_cmd_mm"] or 0) >= 3 and r["t"] >= 7 and r["tau_ee"] is not None]
    h, _ = np.histogram(v, bins=bins); ax.step(bins[:-1], h / max(1, len(v)), where="post", color=CAT[n], lw=2, label=f"{arm} (n={len(v)}, median {np.median(v):.2f})")
    side["panels"][f"tau_{arm}"] = {"bins": bins.tolist(), "hist": h.tolist(), "n": len(v), "median": float(np.median(v))}
ax.set_xlabel("time offset tau: M(t) closest to C(t - tau) (rows)"); ax.set_ylabel("fraction of rows"); ax.legend(fontsize=8)
ax.set_title("time offset of M behind C (moving exec rows, EE space)", fontsize=9)
figstyle.style_axes(fig, all_axes)
fig.suptitle("T43 TK - plan P / command C / measured M", fontsize=11)
fig.tight_layout(); fig.savefig(out, dpi=130)
json.dump(side, open(out.with_suffix(".json"), "w"), indent=1)
print("wrote", out)
