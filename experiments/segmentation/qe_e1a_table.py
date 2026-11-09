#!/usr/bin/env python3
"""E1a SafeLIBERO Spatial Level I table with the B5 versions (dev-e5/dev-38, 2026-11-08).

Arms on the same evaluator (test init 5-14, task 0-3, seed 7, main_aegis budget): the E1a arms
(``*_e1a-<arm>-spatial-LI-*``, launch_method.py) and the B5 versions (``*qe-<tag>-e1a-spatial-LI-t<k>-*``, launch_libero_paired.py).
Per arm: TSR / CAR / SSR (episode counts, SSR per episode), ETS (mean env steps excluding wait), Wilson 95% CI of SSR,
paired vs pi0.5 only on (task, init): discordant SSR pairs + exact two-sided McNemar p. CSC (approx.) and real-contact
counts from qe_csc_contacts.py outputs when present (``csc_<arm>.json`` in --out).
usage: python3 experiments/segmentation/qe_e1a_table.py --out outputs/2026-11-08_E1a_SafeLIBERO_B5
"""
import argparse
import glob
import json
import math
import pathlib

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
ARMS = [("VLA", "π0.5 only", "*_e1a-none-spatial-LI-*"),
        ("AEGIS", "AEGIS port (detector)", "*_e1a-aegis-spatial-LI-*"),
        ("AEGIS_GT", "AEGIS port (privileged names)", "*_e1a-aegis-gt-spatial-LI-*"),
        ("CBF", "multi-link CBF (detector)", "*_e1a-multilink_cbf-spatial-LI-*"),
        ("CBF_GT", "multi-link CBF (privileged GT)", "*_e1a-multilink_cbf-gt-spatial-LI-*"),
        ("KNOWS", "KNOWS port (detector)", "*_e1a-knows-spatial-LI-*"),
        ("KNOWS_GT", "KNOWS port (privileged GT)", "*_e1a-knows-gt-spatial-LI-*"),
        ("B5_GT", "**AG3S B5 (GT perception)**", "2*qe-gt23t100-e1a-spatial-LI-t*"),
        ("B5_noGT", "**AG3S B5 without GT (AG3S perception)**", "2*qe-gt24att-e1a-spatial-LI-t*"),
        ("PE_noGT", "**AG3S B5 noGT + SL-PE (pe123458)**", "2*qe-pe123458-e1a-spatial-LI-t*"),
        ("B5tro_GT", "**AG3S B5-tro (GT, translation-only QP)**", "2*qe-gt23tro-e1a-spatial-LI-t*"),
        ("PEtro_noGT", "**AG3S noGT + SL-PE + tro (pe123458-tro)**", "2*qe-pe123458tro-e1a-spatial-LI-t*"),
        ("PE11_noGT", "**AG3S noGT + SL-PE + PE11 (pe12345811)**", "2*qe-pe12345811-e1a-spatial-LI-t*"),
        ("PE11tro_noGT", "**AG3S noGT + SL-PE + PE11 + tro (pe12345811-tro)**", "2*qe-pe12345811tro-e1a-spatial-LI-t*")]


def load(pat):
    out = {}
    for f in sorted(glob.glob(B + pat + "/episodes.jsonl")):
        if "INVALID" in f:
            continue
        for l in open(f):
            e = json.loads(l)
            out[(int(e["task_id"]), int(e["episode_idx"]))] = e
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    data = {k: load(p) for k, _, p in ARMS}
    vla = data["VLA"]
    ssr = lambda e: bool(e["safelibero"]["safe_success"])
    rows, summary = [], {}
    rows.append("| arm | n | TSR | CAR | **SSR** | SSR % [Wilson 95 %] | ETS | vs π0.5 SSR (arm only / π0.5 only, McNemar p) | CSC approx. (robot · robot+target) | 실제 로봇 접촉 ep |")
    rows.append("|---|---:|---:|---:|---:|---|---:|---|---|---:|")
    for key, label, _ in ARMS:
        d = data[key]
        n = len(d)
        if n == 0:
            continue
        tsr = sum(bool(e["official_success"]) for e in d.values())
        car = sum(not e["safelibero"]["collided"] for e in d.values())
        s = sum(ssr(e) for e in d.values())
        ets = sum(int(e["env_steps_excluding_wait"]) for e in d.values()) / n
        lo, hi = wilson(s, n)
        both = [k for k in d if k in vla]
        b = sum(ssr(d[k]) and not ssr(vla[k]) for k in both)
        c = sum(ssr(vla[k]) and not ssr(d[k]) for k in both)
        csc_txt, real = "–", "–"
        f = out / f"csc_{key}.json"
        if f.exists():
            cs = json.loads(f.read_text())
            if cs:
                csc_txt = "%.1f · %.1f" % (sum(x["contact_steps_robot"] for x in cs) / len(cs),
                                           sum(x["contact_steps_robot_or_target"] for x in cs) / len(cs))
                real = str(sum(x["contact_steps_robot_or_target"] > 0 for x in cs))
        rows.append(f"| {label} | {n} | {tsr} | {car} | **{s}** | {100 * s / n:.1f} [{100 * lo:.0f}, {100 * hi:.0f}] | {ets:.1f} |"
                    f" {'' if key == 'VLA' else f'{b} / {c}, p={mcnemar(b, c):.3g}'} | {csc_txt} | {real} |")
        per_task = {}
        for t in range(4):
            E = [e for (tt, _), e in d.items() if tt == t]
            per_task[t] = (len(E), sum(bool(e["official_success"]) for e in E), sum(not e["safelibero"]["collided"] for e in E),
                           sum(ssr(e) for e in E))
        summary[key] = {"label": label, "n": n, "TSR": tsr, "CAR": car, "SSR": s, "ETS": ets, "per_task": per_task}
    rows += ["", "task 별 (n · TSR · CAR · SSR):", "",
             "| arm | task 0 | task 1 | task 2 | task 3 |", "|---|---|---|---|---|"]
    for key, v in summary.items():
        rows.append(f"| {v['label']} | " + " | ".join("%d · %d · %d · **%d**" % v["per_task"][t] for t in range(4)) + " |")
    (out / "E1a_spatial_LI_표.md").write_text("\n".join(rows) + "\n")
    (out / "E1a_spatial_LI_summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(rows))
    figure(summary, out)


def figure(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    keys = list(summary)
    names = [summary[k]["label"].replace("**", "").replace("AG3S ", "") for k in keys]
    fig, ax = plt.subplots(figsize=(12, 4.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    x = np.arange(len(keys)); w = 0.26
    for j, (m, col) in enumerate((("TSR", "#2a78d6"), ("CAR", "#eb6834"), ("SSR", "#1baf7a"))):
        v = [summary[k][m] / summary[k]["n"] for k in keys]
        ax.bar(x + (j - 1) * w, v, w - 0.02, color=col, label=m)
        for xi, vi in zip(x, v):
            if m == "SSR":
                ax.text(xi + (j - 1) * w, vi + 0.015, f"{vi * 100:.0f}", ha="center", fontsize=7, color="#0b0b0b")
    ax.set_xticks(x, names, rotation=22, ha="right", fontsize=8, color="#52514e")
    ax.set_ylim(0, 1.05); ax.set_ylabel("share of 40 episodes", color="#52514e")
    ax.grid(axis="y", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.set_title("SafeLIBERO Spatial Level I, E1a protocol (task 0-3 x test init 5-14, main_aegis budget, seed 7)", fontsize=10, pad=22)
    fig.tight_layout(); fig.savefig(out / "그래프_E1a_spatial_LI.png"); print("figure written")


if __name__ == "__main__":
    main()
