"""T43 HM evidence figure: (left) distance of the held rows' nearest surface point to the attach-time volume, by GT group
(planned rows with d - r < 10 mm); (right) the 8 carry contacts: held clearance off vs on at the planned and servo poses."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = pathlib.Path(_ag3s_workspace_path('outputs/impl/T43HM/raw'))
G = ("hurdle", "table", "crate", "apple", "fruit", "shelf", "room", "robot")
vals = {}
for arm in ("V7g", "V8"):
    for key, run in json.load(open(RAW / f"survey_{arm}.json"))["runs"].items():
        for ch in run.get("chunks", []):
            for rr in ch["rows"]:
                p = rr["plan"]
                for i, dm in enumerate(p["d_minus_r_mm"]):
                    if dm >= 10 or p["self_cap_mm"] is None:
                        continue
                    lab = "lift (own image)" if ch["phase"] == "lift" else p["gt_group"][i]
                    vals.setdefault(lab, []).append(p["self_cap_mm"][i])
summ = json.load(open(RAW / "hm_summary.json"))
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
cols = {"lift (own image)": "#2a9d8f", "hurdle": "#e76f51", "crate": "#264653", "apple": "#e9c46a", "robot": "#8d99ae"}
order = [k for k in ("lift (own image)", "hurdle", "crate", "apple", "robot") if k in vals]
for j, k in enumerate(order):
    raw = np.asarray(vals[k]); v = np.maximum(raw, 0.5); y = j + 0.25 * (np.random.default_rng(j).random(len(v)) - 0.5)
    ax[0].scatter(v, y, s=8, color=cols[k], alpha=0.7)
    ax[0].text(1.05 * v.max(), j, f"n={len(v)}  min {raw.min():.1f} · max {raw.max():.1f}", va="center", fontsize=8)
ax[0].axvline(10.0, color="k", ls="--", lw=1); ax[0].text(10.5, 0.4, "tol 2 voxel = 10 mm", fontsize=8)
ax[0].set_xscale("log"); ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels(order)
ax[0].set_xlabel("nearest surface point → attach-time volume (mm, log; ≤0 drawn at 0.5)")
ax[0].set_title("held rows with d − r < 10 mm (planned, V7g+V8)")
ev = summ["a_events"]; x = np.arange(len(ev))
for src, mk, off_c, on_c in (("plan", "o", "#adb5bd", "#e76f51"), ("servo", "s", "#ced4da", "#9d0208")):
    off = [e[src]["off_min_mm"] for e in ev]; on = [e[src]["on_min_mm"] for e in ev]
    ax[1].scatter(x, off, marker=mk, facecolor="none", edgecolor=off_c, label=f"{src} pose — off (held margin 0)")
    ax[1].scatter(x, on, marker=mk, color=on_c, label=f"{src} pose — on (HM)")
ax[1].axhline(0, color="k", lw=1)
ax[1].set_xticks(x); ax[1].set_xticklabels([e["key"].split("/")[-1].replace("_ep", "\n") + f"\nt{e['first_t']}" for e in ev], fontsize=7)
ax[1].set_ylabel("min held-row TO clearance in window (mm)"); ax[1].set_title("(a) 8 apple–bar carry contacts: < 0 = TO sees it")
ax[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
fig.tight_layout(); fig.savefig(_ag3s_workspace_path('outputs/impl/T43HM/fig/t43hm-evidence.png'), dpi=130)
print("ok")
