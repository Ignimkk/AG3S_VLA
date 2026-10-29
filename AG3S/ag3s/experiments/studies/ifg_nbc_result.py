"""IFG-APPLE-NBC — neighbour contact (option A) on the 10 VLA-alone-success apple scenes (CPU only, read only).

  python3 -m AG3S.ag3s.experiments.studies.ifg_nbc_result --run <IFG-APPLE run> --fig AG3S/ag3s/docs/figures/ifg-split

Arms (same episode / seed), from the agg3.py copies in IFG-APPLE-NBC/agg4arm: VLA alone (E0), target split (SPL), + wrist-weighted
tracking (SPLW, IFG-TOW), + neighbour contact 20 mm (SPLT, IFG-NBC). Per scene: locked contact-free success, failure mode, HOLD chunks,
neighbour contact rows (agg `contacts.neighbour_rows`, summed; MuJoCo contacts of the robot with a non-target fruit). Grasp point at the
first close = replay TCP minus the apple position at t=0, from the `ifg_goal_drift` sidecars (`-wrist-track`, `-neighbour-contact`).
"""
import argparse, json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from AG3S.ag3s.experiments.common import figstyle as FS  # noqa: E402

KO = FS.use_korean()
T = (lambda ko, en: ko if KO else en)
ARMS = [("E0", T("VLA 단독", "VLA alone")), ("SPL", T("target 분리", "target split")),
        ("SPLW", T("+ 손목 가중치", "+ wrist weight")), ("SPLT", T("+ 이웃 접촉 20 mm", "+ neighbour contact"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--fig", type=pathlib.Path, required=True)
    a = ap.parse_args(); a.fig.mkdir(parents=True, exist_ok=True)
    agg = a.run / "verify/IFG-APPLE-NBC/agg4arm"
    eps = [it["episode"] for it in json.load(open(agg / "seeds.json"))["items"]]
    rows = {}
    for pair in ("SPLW_vs_SPLT", "E0_vs_SPLT", "SPL_vs_SPLT"):
        for r in json.load(open(agg / f"agg_{pair}.json"))["rows"]:
            rows.setdefault(r["episode"], {}).update(r["arms"])
    table = {ep: {k: {"primary": bool(rows[ep][k]["primary"]), "failure_mode": rows[ep][k]["failure_mode"],
                      "hold_chunks": int(rows[ep][k]["hold_chunks"]),
                      "neighbour_rows": int(sum(rows[ep][k]["contacts"]["neighbour_rows"].values()))} for k, _ in ARMS} for ep in eps}
    N = {k: sum(table[ep][k]["primary"] for ep in eps) for k, _ in ARMS}
    figdir = pathlib.Path("AG3S/ag3s/docs/figures/ifg-split")
    close = {}
    for ep in (954, 211):
        w = json.load(open(figdir / f"ifg-goal-drift-ep{ep}-wrist-track.json"))["arms"]
        n = json.load(open(figdir / f"ifg-goal-drift-ep{ep}-neighbour-contact.json"))["arms"]
        close[ep] = {"E0": w["E0"], "SPLW": w["SPL"], "SPLT": n["SPL"]}

    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(19, 6.2), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    FS.style_axes(fig, [ax, bx, cx])
    for i, ep in enumerate(eps):
        for j, (k, _) in enumerate(ARMS):
            x = table[ep][k]
            ax.scatter(j, i, s=230, color=FS.CATEGORICAL[2] if x["primary"] else FS.CATEGORICAL[7], edgecolor=FS.INK_2, lw=0.5, zorder=3)
            ax.text(j, i, "✓" if x["primary"] else "×", ha="center", va="center", fontsize=8.5, color=FS.SURFACE, zorder=4)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([l for _, l in ARMS], fontsize=7.8)
    ax.set_yticks(range(len(eps))); ax.set_yticklabels([f"ep{ep}" for ep in eps], fontsize=8); ax.invert_yaxis()
    ax.set_title(T("a  VLA 단독 성공 사과 10 장면\n   성공 ", "a  10 VLA-success apple scenes\n   success ") + " / ".join(str(N[k]) for k, _ in ARMS),
                 fontsize=9, loc="left")
    nb_eps = [ep for ep in eps if any(table[ep][k]["neighbour_rows"] for k, _ in ARMS)]
    xs = np.arange(len(nb_eps)); wd = 0.2
    for j, (k, lab) in enumerate(ARMS):
        bx.bar(xs + (j - 1.5) * wd, [table[ep][k]["neighbour_rows"] for ep in nb_eps], width=wd, color=FS.CATEGORICAL[j], label=lab)
    bx.set_xticks(xs); bx.set_xticklabels([f"ep{ep}" for ep in nb_eps], fontsize=8)
    bx.set_ylabel(T("이웃 과일 접촉 row 수", "neighbour contact rows"), fontsize=8); bx.legend(fontsize=7, frameon=False)
    bx.set_title(T("b  실제 이웃 과일 접촉 (MuJoCo) — 다른 8 장면은 모든 구성 0",
                   "b  actual neighbour-fruit contact (MuJoCo); other 8 scenes 0"), fontsize=9, loc="left")
    mk = {"E0": "o", "SPLW": "s", "SPLT": "^"}
    for ep, col in ((954, FS.CATEGORICAL[0]), (211, FS.CATEGORICAL[1])):
        for k in ("E0", "SPLW", "SPLT"):
            p = close[ep][k]["tcp_at_first_close_mm"]
            cx.scatter(p[0], p[1], marker=mk[k], s=70, color=col, edgecolor=FS.INK_2, lw=0.5)
            cx.annotate(f"ep{ep} {k}  z{p[2]:+.0f}", (p[0], p[1]), xytext=(4, 4), textcoords="offset points", fontsize=6.8)
    cx.axhline(0, color=FS.INK_2, lw=0.5); cx.axvline(0, color=FS.INK_2, lw=0.5)
    cx.set_xlabel(T("앞뒤 x (mm, + = 로봇에서 멀어짐)", "x forward (mm)"), fontsize=8); cx.set_ylabel(T("옆 y (mm)", "y lateral (mm)"), fontsize=8)
    cx.set_title(T("c  처음 닫을 때 손끝 위치 (사과 기준, t=0)\n   ○ VLA 단독  □ 손목 가중치  △ + 이웃 접촉",
                   "c  hand at the first close (rel. apple)\n   o VLA alone  s wrist weight  ^ + neighbour contact"), fontsize=9, loc="left")
    fig.suptitle(T("이웃 접촉 20 mm 허용: 954 는 VLA 단독과 같은 자리에서 닫아 성공 (접촉 17 → 8 row). 211 은 사과를 지나쳐 바나나를 쥐어 실패",
                   "neighbour contact: ep954 closes where VLA alone does and succeeds; ep211 passes the apple and grips the banana"),
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    p = a.fig / "ifg-nbc-result.png"
    fig.savefig(p, dpi=175, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps({"episodes": eps, "success": N, "table": {str(e): v for e, v in table.items()},
                                                  "first_close": {str(e): {k: {"t": v["first_close_t"], "tcp_rel_apple_t0_mm": v["tcp_at_first_close_mm"]}
                                                                           for k, v in d.items()} for e, d in close.items()},
                                                  "definitions": __doc__, "script": "AG3S/ag3s/experiments/studies/ifg_nbc_result.py"}, indent=1))
    print(p, N)


if __name__ == "__main__":
    main()
