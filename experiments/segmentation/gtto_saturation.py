#!/usr/bin/env python3
"""SL-GTTO3 evidence: how much the TrajOpt bends the VLA path vs how often conversion C clips (task SL-GTTO3 §0).

Per non-HOLD call of the given runs: `to.reference_deviation` (TO refined vs VLA reference) and
`conversion.saturated_rows` (rows of the 10-row chunk whose C correction hit the action bound) / `max_abs_correction`.
Also per grasp-latch phase (searching / latched / closing / held / placed): share of calls with >= 5 saturated rows.

usage: python3 experiments/segmentation/gtto_saturation.py --glob '*_gtto-gt-*' --glob '*gtto2-smoke*' --out DIR
"""
import argparse
import collections
import json
import pathlib

RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
BINS = [(0.0, 1e-4, "0"), (1e-4, 0.01, "<0.01"), (0.01, 0.05, "0.01–0.05"), (0.05, 0.2, "0.05–0.2"), (0.2, 1e9, "≥0.2")]
PHASES = ["searching", "latched", "closing", "held", "placed"]


def rows_of(globs):
    for g in globs:
        for p in sorted(RESULTS.glob(f"{g}/episodes/*_calls.jsonl")):
            for line in p.read_text().splitlines():
                if line.strip():
                    yield p.parent.parent.name, p.name, json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    bins = {lab: {"n": 0, "sat5": 0, "corr": []} for *_, lab in BINS}
    phase = {ph: {"n": 0, "sat5": 0} for ph in PHASES}
    for _, _, r in rows_of(a.glob):
        if r.get("hold"):
            continue
        dev = (r.get("to") or {}).get("reference_deviation")
        c = r.get("conversion") or {}
        sat = len(c.get("saturated_rows") or []) >= 5
        ph = (r.get("grasp") or {}).get("state")
        if ph in phase:
            phase[ph]["n"] += 1
            phase[ph]["sat5"] += sat
        if dev is None:
            continue
        lab = next(lab for lo, hi, lab in BINS if lo <= dev < hi)
        bins[lab]["n"] += 1
        bins[lab]["sat5"] += sat
        bins[lab]["corr"].append(float(c.get("max_abs_correction") or 0.0))
    for b in bins.values():
        cs = sorted(b.pop("corr"))
        b["median_max_abs_correction"] = cs[len(cs) // 2] if cs else None
    res = {"globs": a.glob, "by_deviation": bins, "by_phase": phase}
    (a.out / "saturation.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    from AG3S.ag3s.experiments.common import figstyle
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ko = figstyle.use_korean()
    T = ({"a": "TO 경로 수정량(reference_deviation)별 변환 C 포화 call 비율", "b": "grasp 단계별 C 포화 call 비율",
          "x": "reference_deviation", "y": "C 포화 call 비율 (≥5/10 행)", "ph": "grasp latch 단계"} if ko else
         {"a": "Share of calls where conversion C clips, by TO path change", "b": "C-clipped call share by grasp phase",
          "x": "reference_deviation", "y": "share of calls with ≥5/10 clipped rows", "ph": "grasp latch phase"})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    labs = [lab for *_, lab in BINS]
    ys = [bins[l]["sat5"] / bins[l]["n"] if bins[l]["n"] else 0 for l in labs]
    ax[0].bar(labs, ys, color="#4C72B0")
    for i, l in enumerate(labs):
        ax[0].text(i, ys[i] + 0.02, f"{bins[l]['sat5']}/{bins[l]['n']}", ha="center", fontsize=8)
    ax[0].set_ylim(0, 1.1); ax[0].set_xlabel(T["x"]); ax[0].set_ylabel(T["y"]); ax[0].set_title(T["a"], fontsize=10)
    ys = [phase[p]["sat5"] / phase[p]["n"] if phase[p]["n"] else 0 for p in PHASES]
    ax[1].bar(PHASES, ys, color="#DD8452")
    for i, p in enumerate(PHASES):
        ax[1].text(i, ys[i] + 0.02, f"{phase[p]['sat5']}/{phase[p]['n']}", ha="center", fontsize=8)
    ax[1].set_ylim(0, 1.1); ax[1].set_xlabel(T["ph"]); ax[1].set_title(T["b"], fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out / "그래프_C포화.png", dpi=130)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
