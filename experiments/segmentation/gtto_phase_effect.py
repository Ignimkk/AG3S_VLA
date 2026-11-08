#!/usr/bin/env python3
"""User question 2026-11-08 "단계를 줄인 것에 대해서 효과는 어땠는지": what the grasp latch phases actually were, per arm, on
the evaluation protocol (LIBERO 27 + SafeLIBERO 11), from the call records.
Per arm: which latch states the episodes went through (sequence pattern), calls per state, TO changes / HOLDs per state, and
the gripper opening at attach (the first HELD call's `opening_after`; >= 0.9 = attached with the fingers still open).
Arms: F8 (`gt11hdp`, SL14 opening rule: LATCHED -> HELD on the first settled close call), F9 (`gt12qs`, + the 'moved' check
= closed confirmation), F12 (`gt15loose2`, + moved check + loosened rows), and any later arm tag given with --arm.
usage: gtto_phase_effect.py --out DIR [--arm NAME=tag ...]"""
import argparse
import collections
import json
import pathlib

RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
ARMS = {"F8": "gt11hdp", "F9": "gt12qs", "F12": "gt15loose2"}
STATES = ("searching", "latched", "closing", "held", "placed")


def newest(pattern):
    runs = [p for p in sorted(RESULTS.glob(pattern)) if (p / "episodes.jsonl").is_file()]
    return runs[-1] if runs else None


def runs_of(tag):
    out = []
    for s in (7, 8, 9):
        r = newest(f"*_gtto-{tag}-libero9s{s}-*") or (newest(f"*_gtto-{tag}-libero9-*") if s == 7 else None)
        if r:
            out.append(("LIBERO", r))
    for t in (1, 2, 3):
        r = newest(f"*_gtto-{tag}-safelibero11-t{t}-*")
        if r:
            out.append(("SafeLIBERO", r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--arm", action="append", default=[])
    a = ap.parse_args()
    arms = dict(ARMS)
    for spec in a.arm:
        k, v = spec.split("=", 1)
        arms[k] = v
    a.out.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, tag in arms.items():
        pat = collections.Counter()
        calls = collections.Counter()
        chg = collections.Counter()
        hold = collections.Counter()
        attach_open = []
        n_ep = ok = 0
        open_attach_fail = 0
        for bench, run in runs_of(tag):
            eps = {(e["task_id"], e["episode_idx"]): e for e in map(json.loads, (run / "episodes.jsonl").read_text().splitlines()) if e}
            for (t, e), ep in eps.items():
                f = sorted((run / "episodes").glob(f"*_t{t}_e{e}_calls.jsonl"))
                if not f:
                    continue
                rs = [json.loads(l) for l in f[0].read_text().splitlines() if l.strip()]
                n_ep += 1
                ok += bool(ep.get("official_success")) if bench == "LIBERO" else 0
                seq = []
                first_open = None
                for r in rs:
                    st = (r.get("grasp") or {}).get("state") or "-"
                    if not seq or seq[-1] != st:
                        seq.append(st)
                        if st == "held" and first_open is None:
                            first_open = ((r.get("grasp") or {}).get("evidence") or {}).get("opening_after")
                    calls[st] += 1
                    pc = (r.get("to") or {}).get("passthrough_clear") or {}
                    held_v = any(v.get("action") == "hold" for v in (r.get("verdict_reasons") or []))
                    if held_v:
                        hold[st] += 1
                    elif pc and not pc.get("applied", True):
                        chg[st] += 1
                pat["→".join(s[0].upper() for s in seq)] += 1
                if first_open is not None:
                    attach_open.append(float(first_open))
                    if float(first_open) >= 0.9 and bench == "LIBERO" and not ep.get("official_success"):
                        open_attach_fail += 1
        res[name] = {"tag": tag, "episodes": n_ep, "libero_success": ok, "patterns": dict(pat.most_common(6)),
                     "calls": dict(calls), "changed": dict(chg), "hold": dict(hold),
                     "n_attach": len(attach_open), "open_attach": sum(o >= 0.9 for o in attach_open),
                     "open_attach_libero_failures": open_attach_fail,
                     "attach_opening_median": (sorted(attach_open)[len(attach_open) // 2] if attach_open else None)}
    (a.out / "phase_effect.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    L = ["# grasp latch 단계 — 실제로 지난 상태와 상태별 개입 (평가 seed: LIBERO 27 + SafeLIBERO 11)", "",
         "상태 약자: S searching · L latched · C closing · H held · P placed. '열린 손 attach' = 첫 HELD call 의 개도 ≥ 0.9.", "",
         "| 판 | episode | 상태 순서 (상위) | attach | 열린 손 attach | 그로 인한 LIBERO 실패 | attach 개도 중앙 |", "|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        pats = ", ".join(f"{p} ×{n}" for p, n in r["patterns"].items())
        L.append(f"| {k} | {r['episodes']} | {pats} | {r['n_attach']} | {r['open_attach']} | {r['open_attach_libero_failures']} | "
                 f"{'-' if r['attach_opening_median'] is None else round(r['attach_opening_median'], 3)} |")
    L += ["", "| 판 | 상태 | call | TO 가 바꾼 call | HOLD call |", "|---|---|---|---|---|"]
    for k, r in res.items():
        for st in STATES:
            if r["calls"].get(st):
                L.append(f"| {k} | {st} | {r['calls'][st]} | {r['changed'].get(st, 0)} | {r['hold'].get(st, 0)} |")
    (a.out / "phase_effect표.md").write_text("\n".join(L) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    names = list(res)
    for i, st in enumerate(STATES):
        ax[0].bar([j + 0.16 * (i - 2) for j in range(len(names))], [res[n]["changed"].get(st, 0) + res[n]["hold"].get(st, 0) for n in names],
                  width=0.16, label=st)
    ax[0].set_xticks(range(len(names)), names)
    ax[0].set_title("TO changes + HOLDs per latch state (evaluation runs)")
    ax[0].legend(fontsize=8)
    ax[1].bar(names, [res[n]["open_attach"] for n in names], color="#d95f02", label="open-hand attach (opening >= 0.9)")
    ax[1].bar(names, [res[n]["open_attach_libero_failures"] for n in names], color="k", width=0.4, label="... of which LIBERO failures")
    ax[1].set_title("attach with the fingers still open")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out / "그래프_단계.png", dpi=110)
    print("\n".join(L))


if __name__ == "__main__":
    main()
