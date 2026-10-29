"""T43 XM direction graph (verifier). usage: PYTHONPATH=${AG3S_PROJECT_ROOT} fig_graph_xm.py OUT_PNG
(a) carry_apple window chunks with |D_palm(k*)| > 5 mm: D components at k* (dz up, away = horizontal away from the bar on the robot's
    side, du = along the reference motion; mm) per arm.  (b) V9 runs with the apple in hand: executed-row palm D 'away' (solid) and
    'dz' (dotted) vs t - palm carry crossing t; x = apple leaves the hand; shaded = rows of stay-start chunks.  (c) |D_palm| max vs
    reference violation (reconstructed, incl. 10 mm margin) for carry_apple chunks whose reference violates; lines = LSQ slope per arm.
    (d) median |D_palm(k)| over plan row k = 0..7 (carry_apple chunks with |D| max > 5 mm).
Data: raw/cross_summary.json, raw/cross_V9.json (xm_cross.py, xm_agg.py).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

out = pathlib.Path(sys.argv[1]); assert not out.exists(), f"{out} exists"
RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/XM/raw'))
S = json.load(open(RAW / "cross_summary.json"))
ARMS = ("V5", "V7g", "V8", "V9"); C = figstyle.CATEGORICAL
figstyle.use_korean()
side = {"script": __file__}
fig, ax = plt.subplots(2, 2, figsize=(16, 11)); ax = ax.ravel()
# (a)
comp = {}
for arm in ARMS:
    R = json.load(open(RAW / f"cross_{arm}.json"))["runs"]
    vals = {"dz": [], "away": [], "du": []}
    for key, r in R.items():
        t_from = r["first_inhand_t"] if r["first_inhand_t"] is not None else (r["first_closing_t"] or 0)
        for c in r["chunks"]:
            if not c["in_window"] or c["grasp_state"] == "placed" or c["t0"] + 7 < t_from or not c["inhand_at_plan"]:
                continue
            dp = c["deflection"]["palm"]; k = max(range(len(dp)), key=lambda i: dp[i]["mag"])
            if dp[k]["mag"] > 5:
                for cc in vals:
                    vals[cc].append(dp[k][cc])
    comp[arm] = vals
pos = []; data = []; cols = []
for i, cc in enumerate(("dz", "away", "du")):
    for j, arm in enumerate(ARMS):
        pos.append(i * 5 + j); data.append(comp[arm][cc]); cols.append(C[j])
bp = ax[0].boxplot(data, positions=pos, widths=0.75, patch_artist=True, showfliers=True, flierprops=dict(markersize=3))
for p_, c_ in zip(bp["boxes"], cols):
    p_.set_facecolor(c_); p_.set_alpha(0.75)
ax[0].set_xticks([1.5, 6.5, 11.5]); ax[0].set_xticklabels(["dz (up +)", "away (from bar +)", "du (forward +)"])
ax[0].axhline(0, color="#888888", lw=0.8)
for j, arm in enumerate(ARMS):
    ax[0].plot([], [], color=C[j], lw=6, label=f"{arm} (n={len(comp[arm]['dz'])})")
ax[0].legend(fontsize=8); ax[0].set_ylabel("palm D component at k* (mm)")
ax[0].set_title("(a) carry with apple, window chunks |D|>5 mm: direction of D at k*", fontsize=10)
side["a_components_mm"] = comp
# (b)
R9 = json.load(open(RAW / "cross_V9.json"))["runs"]
tl_side = {}
i = 0
for key, v in S["v9_classes"].items():
    if not v["inhand_intervals"]:
        continue
    r = R9[key]; tx = r["crossing_rows"]["palm_M"]["carry"]; ch = {c["seq"]: c for c in r["chunks"]}
    t0 = v["inhand_intervals"][0][0] - 8; t1 = min(len(r["rows"]), tx + 24)
    T, A, Z, st = [], [], [], []
    for t in range(t0, t1):
        w = r["rows"][t]; c = ch.get(w["seq"])
        if c is None or w["k"] >= len(c["deflection"]["palm"]):
            continue
        d = c["deflection"]["palm"][w["k"]]
        T.append(t - tx); A.append(d["away"] if w["executed"] else np.nan); Z.append(d["dz"] if w["executed"] else np.nan)
        st.append(str(c["initial_source"]).startswith("stay"))
    col = C[i % 8]
    ax[1].plot(T, A, color=col, lw=1.6, label=f"{key.split('_', 1)[1]} ({v['class']})")
    ax[1].plot(T, Z, color=col, lw=1.0, ls=":")
    if v["drop_t"] is not None:
        dt = v["drop_t"] - tx
        if T and T[0] <= dt <= T[-1]:
            ax[1].scatter([dt], [A[T.index(dt)] if dt in T else 0], marker="x", s=80, color=col, zorder=5)
    for t_, s_ in zip(T, st):
        if s_:
            ax[1].axvspan(t_ - 0.5, t_ + 0.5, color=col, alpha=0.05, lw=0)
    tl_side[key] = {"palm_cross_t": tx, "t_minus_cross": T, "away_mm": A, "dz_mm": Z, "stay_rows": st, "drop_t": v["drop_t"], "class": v["class"]}
    i += 1
ax[1].axvline(0, color="#888888", lw=0.8); ax[1].axhline(0, color="#888888", lw=0.8)
ax[1].set_xlabel("t - palm carry crossing t (control rows, 15 Hz)"); ax[1].set_ylabel("executed-row palm D (mm): away solid, dz dotted")
ax[1].set_title("(b) V9 runs with the apple in hand: D direction over time (x = apple leaves hand)", fontsize=10)
ax[1].legend(fontsize=7, loc="upper left")
side["b_v9_timeline"] = tl_side
# (c)
sl = {}
for j, arm in enumerate(ARMS):
    pr = np.array(S["arms"][arm]["carry_apple"]["D_vs_ref_violation"]["pairs"])
    if len(pr):
        ax[2].scatter(pr[:, 1], pr[:, 0], s=14, color=C[j], alpha=0.8, label=f"{arm} n={len(pr)} slope {S['arms'][arm]['carry_apple']['D_vs_ref_violation']['slope_lsq']:.2f}")
        p = np.polyfit(pr[:, 1], pr[:, 0], 1); xx = np.linspace(0, pr[:, 1].max(), 10); ax[2].plot(xx, np.polyval(p, xx), color=C[j], lw=1)
        sl[arm] = {"pairs_D_mm_viol_mm": pr.tolist(), "lsq": p.tolist(), "pearson_r": S["arms"][arm]["carry_apple"]["D_vs_ref_violation"]["pearson_r"]}
ax[2].plot([0, 60], [0, 60], color="#888888", lw=0.8, ls="--", label="|D| = violation")
ax[2].set_xlabel("reference violation -v_ref (mm, reconstructed, incl. 10 mm margin)"); ax[2].set_ylabel("|D_palm| max over k (mm)")
ax[2].set_title("(c) deflection vs reference violation, carry with apple", fontsize=10); ax[2].legend(fontsize=8)
side["c_D_vs_violation"] = sl
# (d)
pf = {}
for j, arm in enumerate(ARMS):
    y = S["arms"][arm]["carry_apple"]["palm_D_profile_by_k_median_mm"]
    ax[3].plot(range(8), y, marker="o", color=C[j], label=arm); pf[arm] = y
    y2 = (S["arms"][arm]["carry_apple"].get("by_initial_source") or {})
ax[3].set_xlabel("plan row k (0 = first executed row)"); ax[3].set_ylabel("median |D_palm(k)| (mm)")
ax[3].set_title("(d) deflection along the 8-step plan window, carry with apple (chunks |D|max>5 mm)", fontsize=10); ax[3].legend(fontsize=8)
side["d_profile_median_mm"] = pf
figstyle.style_axes(fig, ax)
fig.suptitle("T43 XM: direction and size of the TO deflection D = refined - reference at the hurdle crossing", fontsize=12)
fig.tight_layout(); fig.savefig(out, dpi=110)
out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float))
print(out)
