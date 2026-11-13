#!/usr/bin/env python3
"""SL-AUD judge (sdm-f6, 2026-11-14). Criteria = doc/active/handoff_SL_AUD_20261113/2026-11-13_SL-AUD_test_plan_KO.md
§5 / §6.3, written into AG3S_PLAN_SCHEDULE_20261110_KO.md before any result.

S32 = Spatial L-I task 0-3 x init 15-18 (one run, ms-<TAG>-spatial-LI-i15_16_17_18) + Goal L-I / L-II task 0-3 x init 17, 18
(ms-<TAG>-goal-LI-i17_18, ms-<TAG>-goal-LII-i17_18); reference = the frozen version's runs of the same layout (ms-pkgFrssdm-*:
TSR 13, CAR 23, SSR 9, > 50 mm 3). 'lost' = frozen task failure and the same episode's pi0.5-alone task success (pi0.5 =
baseline tranche t1 blocks.csv, the source of the hand-off's qp_delta_latch.csv). 'big correction' = an action-QP call with
max_abs_delta_action >= 1.

usage:
  aud_judge.py s32 TAG {f1|f2}   -> exit 0 PASS, 1 FAIL, 2 INCOMPLETE; prints the decision line ("next=...") and writes
                                     outputs/2026-11-14_SL-AUD/s32_<TAG>.json (cells, 32 episode pairs, criteria)
  aud_judge.py same RUN_DIR_NEW RUN_DIR_REF EPISODE_FILE_STEM
                                  -> exit 0 if the calls are identical in count, max_abs_action_change_executed and
                                     to.passthrough_clear.reason; else 1 and the differing calls (with their joint/QP path)
F1 (§5): pass if SSR >= 12 and > 50 mm <= 4 and (frozen safe successes lost) <= 1.
  else next: SSR >= 12 and > 50 mm >= 5 -> close; SSR <= 11 and >= 7 of the 13 lost episodes still have >= 5 big-correction
  calls -> md5; otherwise close.
F2 (§6.3): pass if SSR >= 11 and > 50 mm <= 4 and (frozen safe successes lost) <= 1; else close.
md5 uses F1's criteria (TAG given with 'f1')."""
import csv, glob, json, os, sys

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
BLOCKS = "/mnt/dev/AG3S_VLA/outputs/2026-11-09_baseline_tranche/t1/blocks.csv"
OUT = "/mnt/dev/AG3S_VLA/outputs/2026-11-14_SL-AUD"
REF = "pkgFrssdm"
CELLS = (("spatial", "I", "i15_16_17_18"), ("goal", "I", "i17_18"), ("goal", "II", "i17_18"))


def latest_dir(pattern):
    ds = [d for d in sorted(glob.glob(B + pattern))
          if os.path.exists(d + "/episodes.jsonl") and not os.path.exists(d + "/INTERRUPTED.txt")]
    return ds[-1] if ds else None


def load(d):
    return {(int(e["task_id"]), int(e["init_state_index"])): e for e in
            (json.loads(l) for l in open(d + "/episodes.jsonl") if l.strip())}


def oc(e):
    s = e.get("safelibero") or {}; ok = bool(e["official_success"]); c = bool(s.get("collided"))
    return "SS" if ok and not c else "S+C" if ok else "F+C" if c else "F"


def disp_mm(e):
    return 1000 * float((e.get("safelibero") or {}).get("max_displacement_m") or 0)


def pi05(suite, lvl):
    rr = [r for r in csv.DictReader(open(BLOCKS)) if r["arm"] == "none" and r["suite"] == suite and r["level"] == lvl
          and r["status"] == "current"]
    return {k: oc(e) for k, e in load(B + rr[-1]["run"]).items()} if rr else {}


def calls_of(d, suite, t, i, e):
    cl = (e.get("safe_client") or {}).get("calls_log") or f"safelibero_{suite}_t{t}_e{i}_calls.jsonl"
    p = d + "/episodes/" + cl
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def qp_of(c):
    return ((c.get("to") or {}).get("ee_action_qp")) or ((c.get("conversion") or {}).get("ee_action_qp")) or {}


def big_calls(calls):
    return sum(1 for c in calls if (qp_of(c).get("max_abs_delta_action") or 0) >= 1.0)


def s32(tag):
    cells, pairs = {}, []
    for suite, lvl, ii in CELLS:
        dc = latest_dir(f"2*_ms-{tag}-{suite}-L{lvl}-{ii}-*")
        dr = latest_dir(f"2*_ms-{REF}-{suite}-L{lvl}-{ii}-*")
        if dc is None or dr is None:
            return None
        C, R, P = load(dc), load(dr), pi05(suite, lvl)
        if len(C) != len(R):
            return None
        es = list(C.values())
        cells[f"{suite} L-{lvl}"] = dict(run=os.path.basename(dc), n=len(es), tsr=sum(bool(e["official_success"]) for e in es),
                                         car=sum(not (e.get("safelibero") or {}).get("collided") for e in es),
                                         ssr=sum(bool((e.get("safelibero") or {}).get("safe_success")) for e in es),
                                         big50=sum(disp_mm(e) > 50 for e in es))
        for (t, i) in sorted(R):
            r, c = R[(t, i)], C[(t, i)]
            cr = calls_of(dc, suite, t, i, c); rr = calls_of(dr, suite, t, i, r)
            pairs.append(dict(cell=f"{suite} L-{lvl}", task=t, init=i, pi05=P.get((t, i), "-"),
                              ref=oc(r), cand=oc(c), ref_disp_mm=round(disp_mm(r), 1), cand_disp_mm=round(disp_mm(c), 1),
                              lost=oc(r)[0] == "F" and P.get((t, i), "-")[:1] == "S",
                              ref_big_calls=big_calls(rr), cand_big_calls=big_calls(cr), ref_calls=len(rr), cand_calls=len(cr)))
    tot = {k: sum(c[k] for c in cells.values()) for k in ("n", "tsr", "car", "ssr", "big50")}
    return dict(cells=cells, total=tot, pairs=pairs)


def judge_s32(tag, crit):
    r = s32(tag)
    if r is None:
        print(f"aud s32 {tag}: INCOMPLETE"); sys.exit(2)
    t, P = r["total"], r["pairs"]
    ss_lost = sum(1 for p in P if p["ref"] == "SS" and p["cand"] != "SS")
    need = 12 if crit == "f1" else 11
    cond = {f"SSR>={need}": t["ssr"] >= need, ">50mm<=4": t["big50"] <= 4, "frozen SS lost<=1": ss_lost <= 1}
    ok = all(cond.values())
    lost13 = [p for p in P if p["lost"]]
    lost_big5 = sum(1 for p in lost13 if p["cand_big_calls"] >= 5)
    if ok:
        nxt = "adoption"
    elif crit == "f1" and t["ssr"] >= 12 and t["big50"] >= 5:
        nxt = "close (collisions)"
    elif crit == "f1" and t["ssr"] <= 11 and lost_big5 >= 7:
        nxt = "md5"
    else:
        nxt = "close"
    r.update(tag=tag, crit=crit, criteria=cond, frozen_ss_lost=ss_lost, n_lost_ref=len(lost13),
             lost_with_ge5_big_calls=lost_big5, verdict="PASS" if ok else "FAIL", next=nxt)
    os.makedirs(OUT, exist_ok=True)
    json.dump(r, open(f"{OUT}/s32_{tag}.json", "w"), indent=1)
    print(f"aud s32 {tag} ({crit}): {'PASS' if ok else 'FAIL'}  total={json.dumps(t)}  criteria={json.dumps(cond)}  "
          f"frozen_SS_lost={ss_lost}  lost13_with_ge5_big={lost_big5}/{len(lost13)}  next={nxt}")
    sys.exit(0 if ok else 1)


def same(dn, dr, stem):
    def rows(d):
        out = []
        for c in (json.loads(l) for l in open(f"{d}/episodes/{stem}_calls.jsonl") if l.strip()):
            pc = (c.get("to") or {}).get("passthrough_clear") or {}
            q = (c.get("conversion") or {}).get("ee_action_qp") or {}
            out.append(dict(seq=c.get("seq"), mac=c.get("max_abs_action_change_executed"), reason=pc.get("reason"),
                            qp_used=q.get("used"), why=q.get("why"), trajopt_status=c.get("trajopt_status")))
        return out
    a, b = rows(dn), rows(dr)
    diff = [(x, y) for x, y in zip(a, b) if (x["mac"], x["reason"]) != (y["mac"], y["reason"])]
    ok = len(a) == len(b) and not diff
    print(f"aud same {stem}: {'SAME' if ok else 'DIFFERENT'}  calls new={len(a)} ref={len(b)}  differing={len(diff)}")
    for x, y in diff[:20]:
        print(f"  seq {x['seq']}: new {json.dumps(x)}  ref {json.dumps(y)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if sys.argv[1] == "s32":
        judge_s32(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "same":
        same(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(__doc__)
