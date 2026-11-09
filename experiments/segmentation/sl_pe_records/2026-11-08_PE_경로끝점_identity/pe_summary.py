"""Summarise replay_*.json per episode: majority-role accuracy, purity, recall, M4 target-in-obstacles per rule.
(2026-11-08, dev-30)  python pe_summary.py [replay_*.json ...]"""
import collections
import glob
import json
import sys

import numpy as np

RULES = ("R0", "PE", "PEL")


def ep_stats(rows, rule, phases):
    rs = [r for r in rows if r["latch"] in phases]
    if not rs:
        return None
    ok = [r[rule].get("majority") == "target" for r in rs]
    pur = [r[rule]["purity"] for r in rs if "purity" in r[rule]]
    rec = [r[rule]["recall"] for r in rs if r[rule].get("recall") is not None]
    m4 = [r[rule]["M4"]["target"] >= 3 for r in rs if "M4" in r[rule]]
    m4x = []
    return {"n": len(rs), "acc": float(np.mean(ok)), "purity": float(np.median(pur)) if pur else None,
            "recall": float(np.median(rec)) if rec else None,
            "m4": float(np.mean(m4)) if m4 else None, "m4x": float(np.mean(m4x)) if m4x else None}


def main():
    files = sys.argv[1:] or sorted(f for f in glob.glob("replay_*.json") if "summary" not in f)
    out, lines = {}, []
    hdr = "| run | episode | ok | " + " | ".join(f"{r} acc approach/grasp · M4" for r in RULES) + " | locked first latch role |"
    lines += [hdr, "|" + "---|" * (4 + len(RULES))]
    agg = collections.defaultdict(list)
    for f in files:
        tag = f.split("replay_")[-1][:-5]
        for stem, e in json.load(open(f)).items():
            rows = e["rows"]
            cells, rec = [], {"success": e["success"]}
            for rule in RULES:
                a = ep_stats(rows, rule, ("searching", "latched"))
                gq = ep_stats(rows, rule, ("closing",))
                rec[rule] = {"approach": a, "grasp": gq}
                f_ = lambda s, k: "–" if not s or s[k] is None else f"{100 * s[k]:.0f}"
                cells.append(f"{f_(a, 'acc')} / {f_(gq, 'acc')} · {f_(a, 'm4')}")
                for ph, s in (("approach", a), ("grasp", gq)):
                    if s:
                        agg[(rule, ph, "acc")].append(s["acc"])
                        if s["m4"] is not None:
                            agg[(rule, ph, "m4")].append(s["m4"])
                        if s["m4x"] is not None:
                            agg[(rule, ph, "m4x")].append(s["m4x"])
            lr = next((r for r in rows if r.get("locked")), None)
            rec["lock_role"] = lr["PEL"].get("majority") if lr else None
            rec["lock_seq"] = lr["seq"] if lr else None
            out[f"{tag}/{stem}"] = rec
            lines.append(f"| {tag} | {stem.split('spatial_')[-1]} | {'ok' if e['success'] else 'FAIL'} | "
                         + " | ".join(cells) + f" | {rec['lock_role']} @{rec['lock_seq']} |")
    lines.append("")
    lines.append("| rule | approach acc (episode mean) | grasp acc | approach M4 (target in obstacles) |")
    lines.append("|---|---|---|---|")
    for rule in RULES:
        g = lambda ph, k: f"{100 * np.mean(agg[(rule, ph, k)]):.0f} %" if agg[(rule, ph, k)] else "–"
        lines.append(f"| {rule} | {g('approach', 'acc')} | {g('grasp', 'acc')} | {g('approach', 'm4')} |")
    print("\n".join(lines))
    json.dump(out, open("replay_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
