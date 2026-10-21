"""IFG-APPLE-S2 diagnosis figure (rule A, graph): three-arm success on the 24 apple scenes, and how the two new serving features
behaved against their offline validation. CPU only, read only.

  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_apple_s2_diag --run <run> --s2fk <s2fk dir> --out <png>

Panels (+ same-name .json sidecar):
  a  contact-free task success per arm (E0 = VLA alone, C1aN = current serving + noun target, S2 = C1aN + confidence alarm + 3D held rule),
     per stratum, from verify/IFG-APPLE-3ARM/agg_*.json
  b  confidence-alarm score distribution: serving S2 records per grasp state vs the held-out offline FK-mask scores (R11b conf.jsonl),
     locked threshold 0.5904 drawn
  c  hand_d (target centre to nearest hand proxy): serving S2 records whose gripper rule says held vs offline GT-held frames (features_fk),
     D_hold 0.131 m drawn
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
THR, DHOLD = 0.5904207590672415, 0.0813 + 0.05
STATES = ("searching", "latched", "closing", "held", "placed")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--s2fk", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    V = a.run / "verify"
    M = V / "IFG-APPLE-3ARM"
    ag = {k: json.loads((M / f"agg_{k}.json").read_text()) for k in ("E0_vs_C1aN", "C1aN_vs_S2")}
    rate = {"E0": ag["E0_vs_C1aN"]["E0"], "C1aN": ag["E0_vs_C1aN"]["C1aN"], "S2": ag["C1aN_vs_S2"]["S2"]}
    sc = {s: [] for s in STATES}; hd = []
    for f in glob.glob(str(V / "IFG-APPLE-S2" / "runs" / "server_*" / "server_constraints" / "run_*" / "chunk_*.npz")):
        s = json.loads(str(np.load(f, allow_pickle=True)["summary_json"]))
        g = s.get("confidence_gate") or {}; gs = s.get("grasp") or {}
        v = g.get("gate_score"); st = str(gs.get("state"))
        if isinstance(v, (int, float)) and st in sc:
            sc[st].append(float(v))
        h = gs.get("held_3d")
        if isinstance(h, dict) and h.get("held_gripper") and h.get("hand_d") is not None:
            hd.append(float(h["hand_d"]))
    off = [json.loads(l).get("q_mem_chain:prefix_L8") for l in open(a.s2fk / "run/rounds/R11b/conf.jsonl")]
    off = [x for x in off if x is not None]
    ohd = [r["hand_d"] for r in (json.loads(l) for l in open(a.s2fk / "features_fk.jsonl")) if r["grasp_state"] == "held" and r["hand_d"] == r["hand_d"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1, 1.4, 1]})
    FS.style_axes(fig, axes)
    ax, bx, cx = axes
    arms = ("E0", "C1aN", "S2"); lab = {"E0": T("VLA 단독", "VLA alone"), "C1aN": T("기존 방식", "current serving"), "S2": T("새 구성", "new config")}
    x = np.arange(3); w = 0.26
    side = {"success": {}, "gate_score": {}, "hand_d": {}}
    for j, (sub, name) in enumerate((("all", T("전체 24", "all 24")), ("clear", T("장애물 없음 16", "clear 16")), ("place", T("바구니 옆 8", "basket post 8")))):
        vals = [rate[k][sub]["n_primary"] / rate[k][sub]["n_runs"] for k in arms]
        ax.bar(x + (j - 1) * w, vals, w, color=FS.SEQ_BLUE[4 + 3 * j], edgecolor=FS.SURFACE, label=name)
        for i, k in enumerate(arms):
            ax.text(x[i] + (j - 1) * w, vals[i] + 0.01, f"{rate[k][sub]['n_primary']}/{rate[k][sub]['n_runs']}", ha="center", fontsize=6.8, color=FS.INK_2)
            side["success"].setdefault(k, {})[sub] = {"n_primary": rate[k][sub]["n_primary"], "n_runs": rate[k][sub]["n_runs"]}
    ax.set_xticks(x); ax.set_xticklabels([lab[k] for k in arms], fontsize=9, color=FS.INK); ax.set_ylim(0, 1.05)
    ax.set_title(T("접촉 없는 작업 성공 (사과 24 장면, 같은 seed)", "contact-free success (24 apple scenes, same seeds)"), fontsize=10, loc="left", color=FS.INK)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=3); ax.grid(axis="y", color=FS.GRID_INK, lw=0.5); ax.set_axisbelow(True)
    data = [off] + [sc[s] for s in STATES]
    names = [T("offline\n(held-out)", "offline\n(held-out)")] + [f"{s}\n(n={len(sc[s])})" for s in STATES]
    bp = bx.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
    for k, b in enumerate(bp["boxes"]):
        b.set_facecolor(FS.GRID_INK if k == 0 else FS.CATEGORICAL[0]); b.set_alpha(0.8); b.set_edgecolor(FS.INK_2)
    for med in bp["medians"]:
        med.set_color(FS.INK)
    bx.axhline(THR, color=FS.CATEGORICAL[7], ls="--", lw=1); bx.text(6.45, THR, T("경보 기준 0.590", "alarm threshold 0.590"), fontsize=7.5, color=FS.CATEGORICAL[7], ha="right", va="bottom")
    bx.set_xticklabels(names, fontsize=7.5, color=FS.INK)
    for k, dset in enumerate(data):
        fr = float(np.mean(np.array(dset) < THR)) if dset else float("nan")
        bx.text(k + 1, -0.12, f"{fr:.0%}", ha="center", fontsize=7.5, color=FS.CATEGORICAL[7])
        side["gate_score"][names[k].split("\n")[0]] = {"n": len(dset), "frac_below": fr, "p10_50_90": [float(np.percentile(dset, p)) for p in (10, 50, 90)] if dset else None}
    bx.set_ylim(-0.2, 1.05)
    bx.set_title(T("경보 점수: 서버(새 구성, grasp 상태별) 대 offline — 아래 숫자 = 기준 미만 비율", "alarm score: serving by grasp state vs offline (below: fraction under threshold)"),
                 fontsize=9.5, loc="left", color=FS.INK)
    bx.grid(axis="y", color=FS.GRID_INK, lw=0.5); bx.set_axisbelow(True)
    bins = np.linspace(0, 0.6, 31)
    cx.hist(ohd, bins=bins, color=FS.GRID_INK, alpha=0.9, label=T(f"offline 정답 쥠 frame (n={len(ohd)})", f"offline GT-held frames (n={len(ohd)})"), density=True)
    cx.hist(hd, bins=bins, color=FS.CATEGORICAL[0], alpha=0.6, label=T(f"서버, gripper 닫힘 (n={len(hd)})", f"serving, gripper closed (n={len(hd)})"), density=True)
    cx.axvline(DHOLD, color=FS.CATEGORICAL[7], ls="--", lw=1); cx.text(DHOLD + 0.01, cx.get_ylim()[1] * 0.9, "0.131 m", fontsize=7.5, color=FS.CATEGORICAL[7])
    cx.set_xlabel(T("hand_d: target 중심 ↔ 손 대리점 (m)", "hand_d: target centre to hand proxy (m)"))
    cx.set_title(T("3D 쥠 규칙의 거리 입력: 서버 대 offline", "3D held rule distance input: serving vs offline"), fontsize=10, loc="left", color=FS.INK)
    cx.legend(fontsize=7.5, frameon=False); cx.grid(axis="y", color=FS.GRID_INK, lw=0.5); cx.set_axisbelow(True)
    side["hand_d"] = {"serving_gripper_closed": {"n": len(hd), "p10_50_90": [float(np.percentile(hd, p)) for p in (10, 50, 90)], "frac_le_Dhold": float(np.mean(np.array(hd) <= DHOLD))},
                      "offline_gt_held": {"n": len(ohd), "p10_50_90": [float(np.percentile(ohd, p)) for p in (10, 50, 90)], "frac_le_Dhold": float(np.mean(np.array(ohd) <= DHOLD))}}
    fig.suptitle(T("새 구성(기존 방식 + 안전 경보 + 3D 쥠 규칙): 결과와, 두 기능이 서버에서 offline 검증과 다르게 동작한 모습",
                   "new configuration (current serving + alarm + 3D held rule): outcome, and how both features behaved differently from offline validation"),
                 fontsize=10, color=FS.INK, x=0.01, ha="left", y=1.04)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    a.out.with_suffix(".json").write_text(json.dumps(side, indent=1, ensure_ascii=False, default=float))
    print(a.out)


if __name__ == "__main__":
    main()
