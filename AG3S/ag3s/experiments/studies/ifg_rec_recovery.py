"""IFG-APPLE-REC — what the retreat-and-retry recovery did on the 10 VLA-alone-success apple scenes (CPU only, read only).

  python3 -m AG3S.ag3s.experiments.studies.ifg_rec_recovery --run <IFG-APPLE run> --fig AG3S/ag3s/docs/figures/ifg-split

Arm SPLE = target split (SPL) + `--recovery-retreat` (IFG-REC). From the agg3.py copies in IFG-APPLE-REC/agg4arm: locked contact-free
success of VLA alone (E0), SPL and SPLE. From the server chunk records (`summary_json.grasp`): per request the latch state, the recovery
record (`grasp.recovery`: hold / empty streaks, `armed`, `active`, `ended`, `skipped`, `palm_to_target_mm` = hold-link (palm) origin to the
manipulated object's centroid at the request's state).
"""
import argparse, glob, json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from AG3S.ag3s.experiments.common import figstyle as FS  # noqa: E402

KO = FS.use_korean()
T = (lambda ko, en: ko if KO else en)
P = (lambda x: eval(x) if isinstance(x, str) else x)   # noqa: E731  (recorded as a Python repr)
STATES = ["se", "la", "cl", "he", "pl"]


def episode_trace(agg: pathlib.Path, ep: int) -> dict:
    jf = glob.glob(str(agg / "replay" / f"runs_SPLE_SPLE_ep{ep}_*.json"))[0]
    al = json.load(open(jf))["server_chunk_alignment"]["block"]
    n = int(np.load(jf[:-5] + ".npz")["plan_index"].max()) + 1
    out = {"state": [], "palm_mm": [], "retreat": [], "armed": [], "ended": [], "skipped": [], "hold_streak": [], "empty_streak": []}
    for k in range(n):
        s = json.loads(str(np.load(f"{al['chunk_dir']}/chunk_{k + al['offset']:05d}.npz", allow_pickle=True)["summary_json"]))
        g = s.get("grasp") or {}
        r = P(g.get("recovery")) or {}
        out["state"].append(str(g.get("state"))[:2])
        out["palm_mm"].append(r.get("palm_to_target_mm"))
        out["retreat"].append(bool(r.get("active")))
        st = r.get("streaks") or {}
        out["hold_streak"].append(int(st.get("hold", 0))); out["empty_streak"].append(int(st.get("empty", 0)))
        if r.get("armed"):
            out["armed"].append({"request": k, "reason": r["armed"], "palm_to_target_mm": r.get("palm_to_target_mm"),
                                 "retreat_target_frame": r.get("target_frame"),
                                 "retreat_target_palm_to_target_mm": r.get("target_palm_to_target_mm")})
        if r.get("ended"):
            out["ended"].append({"request": k, "ended": r["ended"], "requests": r.get("requests")})
        if r.get("skipped"):
            out["skipped"].append(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--fig", type=pathlib.Path, required=True)
    a = ap.parse_args(); a.fig.mkdir(parents=True, exist_ok=True)
    agg = a.run / "verify/IFG-APPLE-REC/agg4arm"
    seeds = json.load(open(agg / "seeds.json"))["items"]
    eps = [it["episode"] for it in seeds]
    rows_s = {r["episode"]: r["arms"] for r in json.load(open(agg / "agg_SPL_vs_SPLE.json"))["rows"]}
    rows_e = {r["episode"]: r["arms"] for r in json.load(open(agg / "agg_E0_vs_SPLE.json"))["rows"]}
    table = {ep: {"E0": bool(rows_e[ep]["E0"]["primary"]), "SPL": bool(rows_s[ep]["SPL"]["primary"]),
                  "SPLE": bool(rows_s[ep]["SPLE"]["primary"]), "SPLE_failure_mode": rows_s[ep]["SPLE"]["failure_mode"],
                  "SPL_failure_mode": rows_s[ep]["SPL"]["failure_mode"]} for ep in eps}
    traces = {ep: episode_trace(agg, ep) for ep in eps}
    N = {k: sum(table[ep][k] for ep in eps) for k in ("E0", "SPL", "SPLE")}
    fired = [ep for ep in eps if traces[ep]["armed"]]

    fig = plt.figure(figsize=(17, 9.6))
    gs = fig.add_gridspec(len(fired), 2, width_ratios=[1, 2.6])
    ax = fig.add_subplot(gs[:, 0]); axes = [fig.add_subplot(gs[i, 1]) for i in range(len(fired))]
    FS.style_axes(fig, [ax, *axes])
    cols = [("E0", T("VLA 단독", "VLA alone")), ("SPL", T("target 분리", "target split")), ("SPLE", T("+ 후퇴 · 재시도", "+ retreat-retry"))]
    for i, ep in enumerate(eps):
        for j, (key, _) in enumerate(cols):
            ok = table[ep][key]
            ax.scatter(j, i, s=230, color=FS.CATEGORICAL[2] if ok else FS.CATEGORICAL[7], edgecolor=FS.INK_2, lw=0.5, zorder=3)
            ax.text(j, i, "✓" if ok else "×", ha="center", va="center", fontsize=8.5, color=FS.SURFACE, zorder=4)
        n_arm = len(traces[ep]["armed"])
        ax.text(2.45, i, T(f"후퇴 {n_arm}", f"retreat {n_arm}") if n_arm else "–", va="center", fontsize=7.5, color=FS.INK_2)
    ax.set_xticks(range(3)); ax.set_xticklabels([c[1] for c in cols], fontsize=8)
    ax.set_yticks(range(len(eps))); ax.set_yticklabels([f"ep{ep}" for ep in eps], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(-0.5, 3.0)
    ax.set_title(T(f"a  VLA 단독 성공 사과 10 장면\n   성공 {N['E0']} / {N['SPL']} / {N['SPLE']}, 성공 장면 후퇴 발동 0",
                   f"a  10 VLA-success apple scenes\n   success {N['E0']} / {N['SPL']} / {N['SPLE']}, retreats in successes 0"),
                 fontsize=9, loc="left")
    shade = {"cl": (FS.CATEGORICAL[1], T("닫는 중", "closing")), "he": (FS.CATEGORICAL[2], T("쥠 인정", "held"))}
    for bx, ep in zip(axes, fired):
        tr = traces[ep]; k = np.arange(len(tr["state"]))
        for st, (c, lab) in shade.items():
            m = np.array([s == st for s in tr["state"]])
            for x in np.flatnonzero(m):
                bx.axvspan(x - 0.5, x + 0.5, color=c, alpha=0.18, lw=0)
        for x in np.flatnonzero(tr["retreat"]):
            bx.axvspan(x - 0.5, x + 0.5, color=FS.CATEGORICAL[0], alpha=0.35, lw=0)
        y = np.array([np.nan if v is None else v for v in tr["palm_mm"]], float)
        bx.plot(k, y, color=FS.INK_2, lw=1.4)
        bx.axhline(120, color=FS.CATEGORICAL[0], ls="--", lw=0.8)
        for ev in tr["armed"]:
            bx.annotate(T(f"발동 ({ev['palm_to_target_mm']:.0f} mm)", f"armed ({ev['palm_to_target_mm']:.0f} mm)"),
                        (ev["request"], ev["palm_to_target_mm"]), xytext=(ev["request"] - 9, 470), fontsize=7.2,
                        arrowprops=dict(arrowstyle="->", color=FS.INK_2, lw=0.6))
        bx.set_ylim(0, 520); bx.set_xlim(-0.5, len(k) - 0.5)
        bx.set_ylabel(T("손바닥–target (mm)", "palm to target (mm)"), fontsize=7.5)
        bx.set_title(("\n" if bx is axes[0] else "") + T(f"ep{ep}  target 분리: {table[ep]['SPL_failure_mode']} → 후퇴 · 재시도: {table[ep]['SPLE_failure_mode']}",
                       f"ep{ep}  target split: {table[ep]['SPL_failure_mode']} -> retreat-retry: {table[ep]['SPLE_failure_mode']}"),
                     fontsize=8.5, loc="left")
    axes[-1].set_xlabel(T("요청 (chunk)", "request (chunk)"), fontsize=8)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Patch(color=FS.CATEGORICAL[1], alpha=0.3, label=shade["cl"][1]), Patch(color=FS.CATEGORICAL[2], alpha=0.3, label=shade["he"][1]),
                            Patch(color=FS.CATEGORICAL[0], alpha=0.5, label=T("후퇴 chunk", "retreat chunk")),
                            Line2D([], [], color=FS.CATEGORICAL[0], ls="--", label=T("후퇴 목표 기준 120 mm", "retreat clear 120 mm"))],
                   fontsize=7, frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.08))
    fig.suptitle(T("후퇴 · 재시도: 실패 3 장면 모두 '빈 손'으로 발동했지만, 발동 때 손바닥은 이미 target 에서 170–373 mm — 후퇴가 거의 제자리였다",
                   "retreat-retry fired on all 3 failures (empty grasp), but the palm was already 170-373 mm from the target: the retreat barely moved"),
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    p = a.fig / "ifg-rec-recovery.png"
    fig.savefig(p, dpi=175, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps({
        "episodes": eps, "success": N, "table": {str(e): v for e, v in table.items()},
        "recovery": {str(e): {x: traces[e][x] for x in ("armed", "ended", "skipped")} | {
            "retreat_requests": int(sum(traces[e]["retreat"])), "max_hold_streak": max(traces[e]["hold_streak"]),
            "max_empty_streak": max(traces[e]["empty_streak"])} for e in eps},
        "palm_to_target_at_each_close_start_mm": {str(e): [traces[e]["palm_mm"][k] for k in range(len(traces[e]["state"]))
                                                          if traces[e]["state"][k] == "cl" and (k == 0 or traces[e]["state"][k - 1] != "cl")]
                                                  for e in eps},
        "palm_to_target_mm": {str(e): traces[e]["palm_mm"] for e in fired}, "state": {str(e): traces[e]["state"] for e in fired},
        "definitions": __doc__, "script": "AG3S/ag3s/experiments/studies/ifg_rec_recovery.py"}, indent=1))
    print(p, N, {e: len(traces[e]["armed"]) for e in eps})


if __name__ == "__main__":
    main()
