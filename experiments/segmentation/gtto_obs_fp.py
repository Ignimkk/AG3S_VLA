#!/usr/bin/env python3
"""G67 calibration: F8 observe-only (GT12 ``-hdp-obs``) on the VLA's own path. The env executes the VLA chunk bit for bit;
each call records what F8 would have done. LIBERO (seeds 10 · 11): in an episode the VLA alone succeeds, every 'would
change' / 'would hold' call is a FALSE POSITIVE. SafeLIBERO (seed 10): a 'would change' call in the window before the VLA's
recorded collision (``safelibero.collide_t``) is a TRUE POSITIVE.

Loosening candidates evaluated offline from the record (no re-run):
  T(tau) accepted model violation tau mm (pass-through and final check): a call stays only if reference violation > tau
         (violation is measured against the floors: obstacle rows 10 mm margin, target / support / destination rows 0)
  C     certified pass-through: a 'geometry not certified' call is judged by its reference violation alone (as collision)
usage: gtto_obs_fp.py --out DIR [--window 40]"""
import argparse
import collections
import json
import pathlib

import numpy as np

RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
TOLS = (0, 2, 5, 8, 10, 12, 15, 18, 21, 25, 30)   # tau mm: accepted model violation beyond the floors


def runs(pattern):
    return [p for p in sorted(RESULTS.glob(pattern)) if (p / "episodes.jsonl").is_file()]


def hold_row(r):
    for v in r.get("verdict_reasons") or []:
        if v.get("action") == "hold":
            row = ((v.get("evidence") or {}).get("row")) or {}
            link = str(row.get("link") or "")
            part = ("attached" if link.startswith("attached") else "finger" if "finger" in link
                    else "hand" if "hand" in link else "arm" if link else "-")
            return {"kind": v.get("kind"), "why": row.get("why") or "-", "part": part, "tier": row.get("tier"),
                    "clear_mm": None if row.get("clearance_m") is None else round(1000 * row["clearance_m"], 1),
                    "step": row.get("step")}
    return None


def call_rows(run):
    for ep in (json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()):
        stem = pathlib.Path(ep.get("npz") or "").stem or None
        cands = sorted((run / "episodes").glob(f"*_t{ep['task_id']}_e{ep['episode_idx']}_calls.jsonl"))
        if not cands:
            continue
        calls = [json.loads(l) for l in cands[0].read_text().splitlines() if l.strip()]
        out = []
        for r in calls:
            ob = r.get("observe") or {}
            pc = (r.get("to") or {}).get("passthrough_clear") or {}
            out.append({"seq": r["seq"], "t": r["obs_index"], "latch": (r.get("grasp") or {}).get("state"),
                        "would_hold": bool(ob.get("would_be_hold")),
                        "would_change": float(ob.get("would_be_max_abs_action_change") or 0.0),
                        "pass": bool(pc.get("applied", True)), "reason": pc.get("reason"),
                        "refviol_mm": 1000 * float(pc.get("reference_violation_m") or 0.0), "hold": hold_row(r)})
        yield ep, cands[0].name.replace("_calls.jsonl", ""), out


def flagged(c, tol, cert):
    """does the call still change / stop the VLA chunk under the candidate rule?"""
    if c["pass"]:
        return False
    if c["reason"] == "collision":
        return c["refviol_mm"] > tol + 1e-9
    if c["reason"] == "geometry not certified":   # checked before the collision test: refviol may be > 0 too
        return (not cert) or c["refviol_mm"] > tol + 1e-9
    return True   # limit / path: unchanged by these candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--tag", default="gt12obs", help="observe run tag (gt12obs = F8, gt14obs = F11)")
    ap.add_argument("--gate", action="store_true", help="exit 0 iff no false-positive call at the run's own rule (tau 0, cert off)")
    ap.add_argument("--window", type=int, default=40, help="env steps before collide_t that count as 'before the collision'")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lib, sl = [], []
    for run in runs(f"*_gtto-{a.tag}-libero9s*"):
        for ep, stem, calls in call_rows(run):
            lib.append({"run": run.name, "stem": stem, "seed": ep["seed"], "success": bool(ep["official_success"]), "calls": calls})
    for run in runs(f"*_gtto-{a.tag}-safelibero11s*"):
        for ep, stem, calls in call_rows(run):
            s = ep.get("safelibero") or {}
            sl.append({"run": run.name, "stem": stem, "success": bool(ep["official_success"]), "collided": bool(s.get("collided")),
                       "collide_t": s.get("collide_t"), "calls": calls})
    # --- false positives on the VLA's successful LIBERO paths
    fp = collections.Counter()
    for e in lib:
        if not e["success"]:
            continue
        for c in e["calls"]:
            if c["would_hold"]:
                h = c["hold"] or {}
                fp[("HOLD", c["latch"], h.get("why"), h.get("part"), "step0" if h.get("step") == 0 else "step>0")] += 1
            elif not c["pass"]:
                fp[("change", c["latch"], c["reason"], "refviol>0" if c["refviol_mm"] > 0 else "refviol=0",
                    "da>0.3" if c["would_change"] > 0.3 else "da<=0.3")] += 1
    # --- candidate sweep: LIBERO false-positive calls / episodes, SafeLIBERO true positives
    sweep = []
    for cert in (False, True):
        for m in TOLS:
            fpc = sum(flagged(c, m, cert) for e in lib if e["success"] for c in e["calls"])
            fpe = sum(any(flagged(c, m, cert) for c in e["calls"]) for e in lib if e["success"])
            tp = 0
            for e in sl:
                if e["collided"] and e["collide_t"] is not None:
                    tp += any(flagged(c, m, cert) and e["collide_t"] - a.window <= c["t"] < e["collide_t"] for c in e["calls"])
            sweep.append({"cert_pass": cert, "tol_mm": m, "libero_fp_calls": fpc, "libero_fp_episodes": fpe,
                          "safelibero_tp": tp})
    holds = sum(c["would_hold"] for e in lib if e["success"] for c in e["calls"])
    res = {"libero": [{k: v for k, v in e.items() if k != "calls"} | {"n_calls": len(e["calls"]),
                                                                       "would_hold": sum(c["would_hold"] for c in e["calls"]),
                                                                       "would_change": sum((not c["pass"]) for c in e["calls"])} for e in lib],
           "safelibero": [{k: v for k, v in e.items() if k != "calls"} | {"n_calls": len(e["calls"])} for e in sl],
           "fp_counts": {"|".join(map(str, k)): v for k, v in fp.most_common()}, "sweep": sweep, "libero_fp_holds": holds}
    (a.out / "obs_fp.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    ns = sum(e["success"] for e in lib)
    nc = sum(e["collided"] for e in sl)
    L = [f"# {a.tag} observe-only — VLA 경로 위의 거짓 양성 (LIBERO seed 10 · 11) · 참 양성 (SafeLIBERO seed 10)", "",
         f"LIBERO: VLA 성공 {ns}/{len(lib)} episode · SafeLIBERO: VLA 충돌 {nc}/{len(sl)} episode (창 = 충돌 전 {a.window} step)", "",
         "| episode | VLA 성공 | call | F8 이 바꿨을 call | F8 이 멈췄을 (HOLD) call |", "|---|---|---|---|---|"]
    for e in lib:
        L.append(f"| {e['stem']} s{e['seed']} | {'✓' if e['success'] else '✗'} | {len(e['calls'])} | "
                 f"{sum((not c['pass']) for c in e['calls'])} | {sum(c['would_hold'] for c in e['calls'])} |")
    L += ["", "## 거짓 양성 종류 (VLA 성공 episode)", "", "| 종류 | latch | 사유 | 1 | 2 | call |", "|---|---|---|---|---|---|"]
    for k, v in fp.most_common():
        L.append("| " + " | ".join(map(str, k)) + f" | {v} |")
    L += ["", "## 완화 후보 (기록으로 계산)", "",
          "| 인증 통과 (C) | 허용 위반 τ mm | LIBERO 거짓 양성 call | 그런 episode | SafeLIBERO 참 양성 (충돌 전 개입한 episode) |",
          "|---|---|---|---|---|"]
    for s in sweep:
        L.append(f"| {'on' if s['cert_pass'] else 'off'} | {s['tol_mm']} | {s['libero_fp_calls']} | {s['libero_fp_episodes']}/{ns} | {s['safelibero_tp']}/{nc} |")
    (a.out / "obs_fp표.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if a.gate:   # the observed version's own decisions: any change / hold on a VLA-success LIBERO episode = fail
        bad = sum((not c["pass"]) or c["would_hold"] for e in lib if e["success"] for c in e["calls"])
        print(f"GATE false-positive calls on VLA-success episodes: {bad}")
        raise SystemExit(0 if bad == 0 and lib else 1)


if __name__ == "__main__":
    main()
