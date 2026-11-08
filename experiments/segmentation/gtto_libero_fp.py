#!/usr/bin/env python3
"""LIBERO false-positive audit (user ruling 2026-11-07 19:1x): on LIBERO the VLA alone succeeds 27/27, so every call where
our module changed (intervened) or stopped (HOLD) the VLA chunk is a candidate false positive. Per call: pass-through /
intervene / HOLD, the reason, the binding row (why · link · tier · clearance), latch state and |Da|; per episode: success,
first HOLD. usage: gtto_libero_fp.py --out outputs/<dir>"""
import argparse
import collections
import json
import pathlib

import numpy as np

RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
ARMS = {"F7": "gt10ee", "F8": "gt11hdp", "F9": "gt12qs"}


def runs(tag):
    out = {}
    for p in sorted(RESULTS.glob(f"*_gtto-{tag}-libero9*")):
        if not (p / "episodes.jsonl").is_file():
            continue
        name = p.name.split(f"gtto-{tag}-libero9")[1]
        seed = "s8" if name.startswith("s8") else "s9" if name.startswith("s9") else "s7"
        out[seed] = p   # newest wins
    return out


def hold_rows(r):
    """the hold-action reasons of a call: (kind, why, link, tier, clearance_mm)"""
    out = []
    for v in r.get("verdict_reasons") or []:
        if v.get("action") != "hold":
            continue
        row = ((v.get("evidence") or {}).get("row")) or {}
        link = str(row.get("link") or "")
        part = ("attached" if link.startswith("attached") else "finger" if "finger" in link
                else "hand" if "hand" in link else "arm" if link else "-")
        out.append((v.get("kind"), row.get("why") or "-", part, row.get("tier") or "-",
                    None if row.get("clearance_m") is None else 1000 * float(row["clearance_m"]), int(row.get("step", -1))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = {}
    for arm, tag in ARMS.items():
        cnt = collections.Counter()
        eps = []
        for seed, run in sorted(runs(tag).items()):
            for ep in (json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()):
                stem = f"libero_spatial_t{ep['task_id']}_e{ep['episode_idx']}"
                cf = run / "episodes" / f"{stem}_calls.jsonl"
                if not cf.is_file():
                    continue
                calls = [json.loads(l) for l in cf.read_text().splitlines() if l.strip()]
                z = np.load(run / "episodes" / f"{stem}_calls.npz", allow_pickle=True)
                da = np.abs(z["actions_response"][..., :6] - z["actions_policy"][..., :6]).max(axis=(1, 2))
                e = {"seed": seed, "task": ep["task_id"], "success": bool(ep["official_success"]), "calls": len(calls),
                     "intervene": 0, "hold": 0, "first_hold": None, "first_hold_row": None, "da_max": float(da.max())}
                for i, r in enumerate(calls):
                    pc = (r.get("to") or {}).get("passthrough_clear") or {}
                    latch = (r.get("grasp") or {}).get("state")
                    if r.get("hold"):
                        e["hold"] += 1
                        hr = hold_rows(r)
                        k = hr[0] if hr else ("hold", "-", "-", "-", None, -1)
                        cnt[("HOLD", latch, k[0], k[1], k[2], "step0" if k[5] == 0 else "step>0")] += 1
                        if e["first_hold"] is None:
                            e["first_hold"], e["first_hold_row"] = r["seq"], {"latch": latch, "row": k}
                    elif not pc.get("applied", True):
                        e["intervene"] += 1
                        cnt[("intervene", latch, pc.get("reason"), "-", "-", f"da>{0.3}" if da[i] > 0.3 else "da<=0.3")] += 1
                eps.append(e)
        res[arm] = {"episodes": eps, "counts": {"|".join(map(str, k)): v for k, v in cnt.most_common()}}
    (a.out / "libero_fp.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    L = ["# LIBERO 27 — VLA 경로를 바꾼 · 멈춘 call (VLA 단독 27/27)", ""]
    for arm in res:
        eps = res[arm]["episodes"]
        L += [f"## {arm} — 성공 {sum(e['success'] for e in eps)}/{len(eps)}", "",
              "| seed · task | 성공 | call | 개입 | HOLD | 첫 HOLD (latch · 사유 · row) | max \\|Δa\\| |", "|---|---|---|---|---|---|---|"]
        for e in eps:
            fh = e["first_hold_row"]
            fhs = "–" if fh is None else f"seq {e['first_hold']} · {fh['latch']} · {fh['row'][0]}/{fh['row'][1]}/{fh['row'][2]} {fh['row'][4] and round(fh['row'][4],1)} mm"
            L.append(f"| {e['seed']} t{e['task']} | {'✓' if e['success'] else '✗'} | {e['calls']} | {e['intervene']} | {e['hold']} | {fhs} | {e['da_max']:.2f} |")
        L += ["", "| 종류 | latch | 사유 | row why | 부위 | 비고 | call |", "|---|---|---|---|---|---|---|"]
        for k, v in list(res[arm]["counts"].items())[:25]:
            L.append("| " + " | ".join(k.split("|")) + f" | {v} |")
        L.append("")
    (a.out / "libero_fp표.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
