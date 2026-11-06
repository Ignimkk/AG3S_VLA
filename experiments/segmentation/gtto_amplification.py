#!/usr/bin/env python3
"""How large is the TO's correction when it does act? (lead diagnosis G45, 2026-11-05 14:0x KST)

For every call of the GT common-TO arms whose T88 pass-through was refused for a **collision**
(`to.passthrough_clear.applied == False and reason == "collision"`), read from `_calls.jsonl`:
  violation_mm   `to.passthrough_clear.reference_violation_m` (how far the policy chunk violates)
  change_mm      `conversion.realisation.refined_change_mm_max` (how far the refined path moves the EE)
  saturated      share of conversion-C rows saturated (`conversion.saturated_rows` / `conversion.rows`)
  da             `max_abs_action_change_executed` (executed − policy action, normalised units, range ±1)
  cap            `to.max_iterations_hit`
Outputs (DIR): amplification.json, 증폭표.md, 그래프_증폭.png.

usage: python3 experiments/segmentation/gtto_amplification.py --out DIR [--arms gt5zr gt8hand gt9it8]
"""
import argparse
import json
import pathlib
import re

import numpy as np

RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
ARM_RE = re.compile(r"_gtto-(?P<arm>gt10ee|gt9it8|gt8hand|gt7ochbc|gt6oc|gt5zr|gt3termsoff-pass|gt2)-(?P<bench>libero9|safelibero11-t\d|ls10)-")
BENCH = {"libero9": "LIBERO"}


def collect(arms):
    rows = []
    for run in sorted(RESULTS.iterdir()):
        m = ARM_RE.search(run.name)
        if not m or m.group("arm") not in arms or not (run / "episodes").is_dir():
            continue
        bench = BENCH.get(m.group("bench"), "SafeLIBERO" if m.group("bench").startswith("safe") else "LIBERO-Safety")
        for f in sorted((run / "episodes").glob("*_calls.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                c = json.loads(line)
                to = c.get("to") or {}
                pc = to.get("passthrough_clear") or {}
                if pc.get("applied") is not False or pc.get("reason") != "collision":
                    continue
                cv = c.get("conversion") or {}
                rz = cv.get("realisation") or {}
                rows.append({"arm": m.group("arm"), "bench": bench, "run": run.name, "episode": f.name.split("_calls")[0],
                             "seq": c.get("seq"), "violation_mm": 1000 * float(pc.get("reference_violation_m") or 0),
                             "change_mm": float(rz.get("refined_change_mm_max") or 0),
                             "saturated": len(cv.get("saturated_rows") or []) / max(1, int(cv.get("rows") or 1)),
                             "da": float(c.get("max_abs_action_change_executed") or 0),
                             "cap": bool(to.get("max_iterations_hit")), "iterations": to.get("sqp_iterations")})
    return rows


def summary(rows):
    out = {}
    for key in sorted({(r["arm"], r["bench"]) for r in rows}):
        rs = [r for r in rows if (r["arm"], r["bench"]) == key]
        v = np.array([r["violation_mm"] for r in rs]); ch = np.array([r["change_mm"] for r in rs])
        out[f"{key[0]}|{key[1]}"] = {
            "n_calls": len(rs), "violation_mm_median": round(float(np.median(v)), 1),
            "change_mm_median": round(float(np.median(ch)), 1), "change_mm_p90": round(float(np.percentile(ch, 90)), 1),
            "ratio_median": round(float(np.median(ch / np.maximum(v, 0.1))), 2),
            "saturated_median": round(float(np.median([r["saturated"] for r in rs])), 2),
            "da_median": round(float(np.median([r["da"] for r in rs])), 2),
            "cap_share": round(float(np.mean([r["cap"] for r in rs])), 2),
            "iterations_median": float(np.median([r["iterations"] or 0 for r in rs]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=["gt5zr", "gt8hand", "gt9it8", "gt10ee"])
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = collect(set(a.arms))
    summ = summary(rows)
    (out / "amplification.json").write_text(json.dumps({"summary": summ, "calls": rows}, indent=1, ensure_ascii=False))
    lines = ["| 판 | 벤치 | 충돌로 통과 거부된 call | 위반 중앙값 mm | 궤적 변경 중앙값 mm | 변경 p90 mm | 변경/위반 중앙값 | "
             "C 포화 행 비율 중앙값 | |Δa| 중앙값 (±1 범위) | 반복 cap 도달 | 반복 수 중앙값 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, s in summ.items():
        arm, bench = k.split("|")
        lines.append(f"| {arm} | {bench} | {s['n_calls']} | {s['violation_mm_median']} | {s['change_mm_median']} | "
                     f"{s['change_mm_p90']} | {s['ratio_median']} | {s['saturated_median']} | {s['da_median']} | "
                     f"{s['cap_share']:.0%} | {s['iterations_median']:g} |")
    lines.append("\n출처: 각 run 의 `episodes/*_calls.jsonl` (`to.passthrough_clear`, `conversion.realisation`, "
                 "`max_abs_action_change_executed`, `to.max_iterations_hit`). script `experiments/segmentation/gtto_amplification.py`.")
    (out / "증폭표.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    _figure(out, rows, a.arms)


def _figure(out, rows, arms):
    from AG3S.ag3s.experiments.common import figstyle
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ko = figstyle.use_korean()
    marks = {"LIBERO": "o", "SafeLIBERO": "s", "LIBERO-Safety": "^"}
    cols = {"gt5zr": "#4C72B0", "gt8hand": "#DD8452", "gt9it8": "#55A868", "gt10ee": "#C44E52", "gt7ochbc": "#8172B2", "gt6oc": "#937860"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for arm in arms:
        for b, mk in marks.items():
            rs = [r for r in rows if r["arm"] == arm and r["bench"] == b]
            if rs:
                ax[0].scatter([r["violation_mm"] for r in rs], [r["change_mm"] for r in rs], s=14, marker=mk,
                              color=cols.get(arm, "gray"), alpha=0.6, label=f"{arm} · {b}")
    lim = max([r["change_mm"] for r in rows] + [10])
    ax[0].plot([0, lim], [0, lim], "k--", lw=0.8, label="변경 = 위반" if ko else "change = violation")
    ax[0].set_xlabel("정책 chunk 위반 (mm)" if ko else "policy chunk violation (mm)")
    ax[0].set_ylabel("TO 궤적 변경 최대 (mm)" if ko else "refined path change max (mm)")
    ax[0].set_title("TO 가 개입할 때: 위반 대비 변경" if ko else "When the TO acts: change vs violation", fontsize=10)
    ax[0].legend(fontsize=6)
    bins = np.linspace(0, 2, 21)
    for arm in arms:
        d = [r["da"] for r in rows if r["arm"] == arm]
        if d:
            ax[1].hist(d, bins=bins, histtype="step", lw=1.6, color=cols.get(arm, "gray"), label=f"{arm} (n={len(d)})")
    ax[1].set_xlabel("실행 − 정책 action 최대 |Δa| (정규화, 범위 ±1)" if ko else "max |executed − policy action|")
    ax[1].set_ylabel("call 수" if ko else "calls")
    ax[1].set_title("실행 action 변화" if ko else "executed action change", fontsize=10)
    ax[1].legend(fontsize=7)
    for arm in arms:
        d = [r["saturated"] for r in rows if r["arm"] == arm]
        if d:
            ax[2].hist(d, bins=np.linspace(0, 1, 11), histtype="step", lw=1.6, color=cols.get(arm, "gray"), label=arm)
    ax[2].set_xlabel("conversion C 포화 행 비율" if ko else "conversion C saturated row share")
    ax[2].set_ylabel("call 수" if ko else "calls")
    ax[2].set_title("conversion C 포화" if ko else "conversion C saturation", fontsize=10)
    ax[2].legend(fontsize=7)
    fig.suptitle(("충돌로 pass-through 가 거부된 call 만 (GT 인식 공용 TO, seed 7, 장면당 1 episode)" if ko else
                  "calls whose pass-through was refused for a collision"), fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "그래프_증폭.png", dpi=130)


if __name__ == "__main__":
    main()
