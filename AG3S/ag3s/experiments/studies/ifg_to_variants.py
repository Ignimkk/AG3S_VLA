"""IFG TrajOpt variants on the 10 VLA-alone-success apple scenes (CPU only, read only).

  python3 -m AG3S.ag3s.experiments.studies.ifg_to_variants --run <IFG-APPLE run> --fig AG3S/ag3s/docs/figures/ifg-split

Configurations (same episode / seed): VLA alone (IFG-APPLE E0), target split (IFG-APPLE-SPLIT SPL), look-ahead off (IFG-APPLE-LOOKAHEADOFF
SPLN), deflection-rate term only after the grasp (IFG-APPLE-TOB SPLD), grasp-goal anchor (IFG-APPLE-TOA2 SPLA), anchor + safety-first
fallback (IFG-APPLE-TOA3 SPLA). Per run from the agg3.py copies in each kit's agg folder: locked contact-free success, failure mode,
HOLD chunks. Per anchor run, from the server records: requests whose TrajOpt result fell back to the plain solve (TOA3).
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
COLS = [("E0", "IFG-APPLE-LOOKAHEADOFF/agg3arm/agg_E0_vs_SPLN.json", "E0", T("VLA 단독", "VLA alone")),
        ("SPL", "IFG-APPLE-LOOKAHEADOFF/agg3arm/agg_SPL_vs_SPLN.json", "SPL", T("target 분리", "target split")),
        ("SPLN", "IFG-APPLE-LOOKAHEADOFF/agg3arm/agg_SPL_vs_SPLN.json", "SPLN", T("look-ahead 끔", "look-ahead off")),
        ("SPLD", "IFG-APPLE-TOB/agg4arm/agg_SPL_vs_SPLD.json", "SPLD", T("쥐기 전\n꺾임 유지 항 끔", "rate term\nafter grasp")),
        ("TOA2", "IFG-APPLE-TOA2/agg4arm/agg_SPL_vs_SPLA.json", "SPLA", T("잡을 지점 고정", "grasp anchor")),
        ("TOA3", "IFG-APPLE-TOA3/agg4arm/agg_SPL_vs_SPLA.json", "SPLA", T("고정 +\n안전 우선 대체", "anchor +\nsafety fallback"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--fig", type=pathlib.Path, required=True)
    a = ap.parse_args(); a.fig.mkdir(parents=True, exist_ok=True)
    v = a.run / "verify"
    seeds = json.load(open(v / "IFG-APPLE-TOA3/seeds.json"))["items"]
    eps = [it["episode"] for it in seeds]
    table = {}
    for key, path, arm, _ in COLS:
        rows = {r["episode"]: r["arms"][arm] for r in json.load(open(v / path))["rows"]}
        table[key] = {ep: {"primary": bool(rows[ep]["primary"]), "failure_mode": rows[ep]["failure_mode"],
                           "hold_chunks": int(rows[ep]["hold_chunks"])} for ep in eps}
    fallback = {}
    P = lambda x: eval(x) if isinstance(x, str) else x   # noqa: E731  (recorded as a Python repr)
    for ep in eps:
        jf = glob.glob(str(v / "IFG-APPLE-TOA3/replay" / f"runs_SPLA_SPLA_ep{ep}_*.json"))[0]
        al = json.load(open(jf))["server_chunk_alignment"]["block"]
        n = {"plain": 0, "anchored": 0}
        z = np.load(jf[:-5] + ".npz")
        for k in range(int(z["plan_index"].max()) + 1):
            c = np.load(f"{al['chunk_dir']}/chunk_{k + al['offset']:05d}.npz", allow_pickle=True)
            mm = P(((json.loads(str(c["summary_json"])).get("to") or {}).get("metrics") or {}).get("grasp_anchor")) or {}
            fb = mm.get("fallback")
            if fb:
                n[fb["used"]] += 1
        fallback[ep] = n
    N = {key: sum(table[key][ep]["primary"] for ep in eps) for key, *_ in COLS}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(17, 6.4), gridspec_kw={"width_ratios": [1.25, 1]}); FS.style_axes(fig, [ax, bx])
    for i, ep in enumerate(eps):
        for jx, (key, *_rest) in enumerate(COLS):
            x = table[key][ep]
            ax.scatter(jx, i, s=230, color=FS.CATEGORICAL[2] if x["primary"] else FS.CATEGORICAL[7], edgecolor=FS.INK_2, lw=0.5, zorder=3)
            ax.text(jx, i, "✓" if x["primary"] else "×", ha="center", va="center", fontsize=8.5, color=FS.SURFACE, zorder=4)
            if x["hold_chunks"] >= 5:
                ax.text(jx + 0.2, i - 0.2, f"H{x['hold_chunks']}", fontsize=6.5, color=FS.CATEGORICAL[1])
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c[3] for c in COLS], fontsize=7.6)
    ax.set_yticks(range(len(eps))); ax.set_yticklabels([f"ep{ep} [{T('바구니 옆 장애물', 'place') if it['stratum'] == 'place' else T('장애물 없음', 'clear')}]"
                                                         for ep, it in zip(eps, seeds)], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(-0.5, len(COLS) - 0.3)
    ax.set_title(T("a  VLA 단독 성공 사과 10 장면 · TrajOpt 변형별 결과 (H n = HOLD chunk 수)\n   성공 " + " / ".join(str(N[c[0]]) for c in COLS),
                   "a  10 VLA-success apple scenes by TrajOpt variant (H n = HOLD chunks)\n   success " + " / ".join(str(N[c[0]]) for c in COLS)),
                 fontsize=9, loc="left")
    xs = np.arange(len(eps)); w = 0.38
    bx.bar(xs - w / 2, [fallback[ep]["anchored"] for ep in eps], width=w, color=FS.CATEGORICAL[0], label=T("대체 풀이도 침범 → 고정 해 유지", "plain also violated: anchored kept"))
    bx.bar(xs + w / 2, [fallback[ep]["plain"] for ep in eps], width=w, color=FS.CATEGORICAL[1], label=T("대체 풀이 채택 (침범 더 작음)", "plain kept (smaller violation)"))
    bx.set_xticks(xs); bx.set_xticklabels([f"ep{ep}" for ep in eps], fontsize=8); bx.set_ylabel(T("요청 수", "requests"), fontsize=8)
    bx.legend(fontsize=7.5, frameon=False)
    bx.set_title(T("b  고정 + 안전 우선 대체(TOA3): 고정 해가 침범을 남긴 요청\n   211 은 고정 없이 풀어도 침범 19–27 mm 가 남는 교착 (HOLD → 같은 계획 → HOLD)",
                   "b  TOA3: requests whose anchored result left a violation\n   ep211: the plain solve is stuck too (HOLD loop)"), fontsize=9, loc="left")
    fig.suptitle(T("TrajOpt 변형 다섯 가지 모두 VLA 단독 성공 10 장면 중 7 장면. 실패 장면은 바뀌지만 수는 그대로다",
                   "all five TrajOpt variants: 7 of the 10 VLA-success scenes; which ones fail changes, the count does not"),
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    p = a.fig / "ifg-to-variants.png"
    fig.savefig(p, dpi=175, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps({"episodes": eps, "success": N, "table": {k: {str(e): x for e, x in d.items()} for k, d in table.items()},
                                                  "toa3_fallback": {str(e): n for e, n in fallback.items()}, "definitions": __doc__,
                                                  "script": "AG3S/ag3s/experiments/studies/ifg_to_variants.py"}, indent=1))
    print(p, N)


if __name__ == "__main__":
    main()
