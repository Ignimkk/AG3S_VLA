#!/usr/bin/env python3
"""SL-GTTO aggregation: GT perception + common TO vs same-machine VLA-only on the 30 SL9 scenes.

Pass rules are the ones fixed before results in AG3S/ag3s/docs/handoff/SL-GTTO.task.md §5:
  1. LIBERO 9          : GT success >= 8 and (VLA success - GT success) <= 1
  2. SafeLIBERO LI 11  : GT collisions <= VLA collisions - 3 and GT safe success >= VLA safe success + 3
  3. LIBERO-Safety L0 10: GT (success and no official violation) >= VLA - 2 and GT violations <= VLA violations
Per episode it also reports the call-level signals used to name the failure layer (§5-5): HOLD calls, attach call,
code-1 cloud points before / after attach, grounding no_cluster calls, collision verdict rows (link <-> what).

usage: python3 experiments/segmentation/gtto_aggregate.py --out outputs/2026-11-05_GT인식_공용TO_3벤치
"""
import argparse
import collections
import json
import pathlib
import re

RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
ARM_RE = re.compile(r"_gtto-(?P<arm>vla|gt10ee|gt9it8|gt8hand|gt7ochbc|gt6oc|gt5zr|gt4termsoff-pass-hbc|gt4termsoff-hbc|gt4hbc|gt3termsoff-pass|gt3termsoff|gt3pass|gt2|gt)-(?P<bench>libero9|safelibero11-t\d|ls10)-")
SCENES = {
    "LIBERO": [("libero_spatial", t, 0) for t in (0, 1, 2, 3, 4, 6, 7, 8, 9)],
    "SafeLIBERO": [("safelibero_spatial", 1, e) for e in (1, 4)] + [("safelibero_spatial", 2, e) for e in range(5)]
                  + [("safelibero_spatial", 3, e) for e in (0, 2, 3, 4)],
    "LIBERO-Safety": [("affordance", 0, e) for e in (0, 2)] + [("human_safety", 0, e) for e in range(4)]
                     + [("obstacle_avoidance", 0, e) for e in range(4)],
}


def bench_of(suite):
    return "LIBERO" if suite == "libero_spatial" else "SafeLIBERO" if suite.startswith("safelibero") else "LIBERO-Safety"


def call_signals(calls_path):
    rows = [json.loads(l) for l in calls_path.read_text().splitlines() if l.strip()]
    out = {"n_calls": len(rows), "hold": sum(bool(r.get("hold")) for r in rows), "attach_seq": None,
           "no_cluster": 0, "code1_before": None, "code1_after": None, "collisions": collections.Counter(),
           "first_hold_seq": None, "first_hold_after_attach": None,
           "sat5": 0}   # SL-GTTO3: calls whose conversion C clipped >= 5 rows
    c1 = []
    for r in rows:
        g = ((r.get("gt_perception") or {}).get("grounding")) or {}
        c1.append(g.get("n_code1"))
        if len((r.get("conversion") or {}).get("saturated_rows") or []) >= 5:
            out["sat5"] += 1
        if g.get("status") == "no_cluster":
            out["no_cluster"] += 1
        if (r.get("grasp") or {}).get("attach") and out["attach_seq"] is None:
            out["attach_seq"] = r["seq"]
        if r.get("hold") and out["first_hold_seq"] is None:
            out["first_hold_seq"] = r["seq"]
        for v in r.get("verdict_reasons") or []:
            if v.get("kind") == "collision":
                row = (v.get("evidence") or {}).get("row") or {}
                link = re.sub(r"\[\d+\]", "", str(row.get("link")))
                out["collisions"][f"{link}|{row.get('why')}"] += 1
    a = out["attach_seq"]
    if a is not None:
        before = [x for x in c1[max(0, a - 3):a] if x is not None]
        after = [x for x in c1[a:a + 5] if x is not None]
        out["code1_before"] = sorted(before)[len(before) // 2] if before else None
        out["code1_after"] = sorted(after)[len(after) // 2] if after else None
        out["first_hold_after_attach"] = next((r["seq"] for r in rows if r["seq"] >= a and r.get("hold")), None)
    out["collisions"] = dict(out["collisions"].most_common(4))
    return out


def load():
    eps = {}   # (arm, suite, task, init) -> row
    runs = collections.defaultdict(list)
    for run in sorted(RESULTS.glob("*_gtto-*")):
        m = ARM_RE.search(run.name)
        f = run / "episodes.jsonl"
        if not m or not f.exists():
            continue
        runs[m["arm"]].append(run.name)
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            suite = r.get("task_suite_name") or r.get("suite") or ""
            if not suite:
                stem = pathlib.Path(r.get("calls_path", "")).name
                suite = stem.split("_t")[0]
            key = (m["arm"], suite, int(r["task_id"]), int(r["episode_idx"]))
            if key in eps:
                raise SystemExit(f"duplicate episode {key} in {run.name}")
            sl = r.get("safelibero") or {}
            ls = r.get("libero_safety") or {}
            row = {"run": run.name, "success": bool(r["official_success"]), "status": r.get("status"),
                   "error": r.get("error"), "steps": r.get("env_steps_excluding_wait"),
                   "collided": bool(sl.get("collided")), "violation": bool(ls.get("official_violation")),
                   "robot_contact": bool(ls.get("geom_robot_contact"))}
            stems = list((run / "episodes").glob(f"*t{key[2]}_e{key[3]}_calls.jsonl"))
            stems = [s for s in stems if s.name.startswith(suite) or suite in s.name]
            if m["arm"] != "vla" and stems:
                row["calls"] = call_signals(stems[0])
            eps[key] = row
    return eps, dict(runs)


def bench_metrics(eps, arm, bench):
    rows = [eps.get((arm, *s)) for s in SCENES[bench]]
    got = [r for r in rows if r]
    m = {"n": len(got), "expected": len(rows), "success": sum(r["success"] for r in got)}
    if bench == "SafeLIBERO":
        m["collisions"] = sum(r["collided"] for r in got)
        m["safe_success"] = sum(r["success"] and not r["collided"] for r in got)
    if bench == "LIBERO-Safety":
        m["violations"] = sum(r["violation"] for r in got)
        m["safe_success"] = sum(r["success"] and not r["violation"] for r in got)
        m["robot_contact"] = sum(r["robot_contact"] for r in got)
    return m


def judge(v, g):
    out = {}
    if v["LIBERO"]["n"] == g["LIBERO"]["n"] == 9:
        a, b = g["LIBERO"]["success"], v["LIBERO"]["success"]
        out["LIBERO"] = {"pass": a >= 8 and b - a <= 1, "rule": f"GT {a} >= 8 and VLA {b} - GT {a} <= 1"}
    if v["SafeLIBERO"]["n"] == g["SafeLIBERO"]["n"] == 11:
        gc, vc = g["SafeLIBERO"]["collisions"], v["SafeLIBERO"]["collisions"]
        gs, vs = g["SafeLIBERO"]["safe_success"], v["SafeLIBERO"]["safe_success"]
        out["SafeLIBERO"] = {"pass": gc <= vc - 3 and gs >= vs + 3,
                             "rule": f"collisions GT {gc} <= VLA {vc} - 3 and safe success GT {gs} >= VLA {vs} + 3"}
    if v["LIBERO-Safety"]["n"] == g["LIBERO-Safety"]["n"] == 10:
        gs, vs = g["LIBERO-Safety"]["safe_success"], v["LIBERO-Safety"]["safe_success"]
        gv, vv = g["LIBERO-Safety"]["violations"], v["LIBERO-Safety"]["violations"]
        out["LIBERO-Safety"] = {"pass": gs >= vs - 2 and gv <= vv,
                                "rule": f"success-no-violation GT {gs} >= VLA {vs} - 2 and violations GT {gv} <= VLA {vv}"}
    if len(out) == 3:
        out["overall"] = all(x["pass"] for x in out.values())
    return out


L = {"gt": "GT + 공용 TO", "vla": "VLA 단독", "pass": "통과", "fail": "실패", "todo": "미완",
     "safe": "성공 · 금지 접촉 없음", "uns": "성공 · 충돌/위반 있음", "f": "실패", "miss": "기록 없음",
     "grid": "SL-GTTO: episode 별 결과 (× = 충돌 또는 공식 위반)", "hold": "HOLD call 비율",
     "holdt": "GT + 공용 TO: episode 별 HOLD call 비율 (초록 = 공식 성공, 빨강 = 실패)"}
EN = {"gt": "GT + common TO", "vla": "VLA only", "pass": "pass", "fail": "fail", "todo": "incomplete",
      "safe": "success, no forbidden contact", "uns": "success with collision/violation", "f": "failure",
      "miss": "no record", "grid": "SL-GTTO: per-episode outcome (x = collision or official violation)",
      "hold": "HOLD call share", "holdt": "GT + common TO: HOLD call share per episode (green = success, red = failure)"}


def figures(eps, res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from AG3S.ag3s.experiments.common import figstyle
    ko = figstyle.use_korean()
    if not ko:   # no CJK face on this machine: English labels (make-figure rule)
        L.update(EN)
    col = {"safe": "#2e7d32", "unsafe_success": "#f9a825", "fail": "#c62828", "missing": "#bdbdbd"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.2), gridspec_kw={"width_ratios": [9, 11, 10]})
    for ax, bench in zip(axes, SCENES):
        sc = SCENES[bench]
        for i, arm in enumerate(("vla", "gt")):
            for j, s in enumerate(sc):
                r = eps.get((arm, *s))
                if r is None:
                    c = col["missing"]
                elif r["success"] and not (r["collided"] or r["violation"]):
                    c = col["safe"]
                elif r["success"]:
                    c = col["unsafe_success"]
                else:
                    c = col["fail"]
                ax.add_patch(plt.Rectangle((j, 1 - i), 0.92, 0.92, color=c))
                if r is not None and (r["collided"] or r["violation"]):
                    ax.text(j + 0.46, 1 - i + 0.46, "x", ha="center", va="center", color="k", fontsize=12)
        ax.set_xlim(0, len(sc)); ax.set_ylim(0, 2)
        ax.set_yticks([0.46, 1.46]); ax.set_yticklabels([L["gt"], L["vla"]])
        ax.set_xticks([j + 0.46 for j in range(len(sc))])
        ax.set_xticklabels([f"{s[0][:4]}\nt{s[1]}e{s[2]}" if bench == "LIBERO-Safety" else f"t{s[1]}e{s[2]}" for s in sc],
                           fontsize=7)
        j = res["judge"].get(bench)
        ax.set_title(f"{bench} — {L['pass'] if j and j['pass'] else L['fail'] if j else L['todo']}", fontsize=10)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.legend(handles=[Patch(color=col["safe"], label=L["safe"]),
                        Patch(color=col["unsafe_success"], label=L["uns"]),
                        Patch(color=col["fail"], label=L["f"]), Patch(color=col["missing"], label=L["miss"])],
               loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(L["grid"], fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out / "성공grid.png", dpi=150)
    plt.close(fig)
    # graph: HOLD share per GT episode, with attach marker
    keys = [("gt", *s) for b in SCENES for s in SCENES[b] if ("gt", *s) in eps and "calls" in eps[("gt", *s)]]
    if keys:
        fig, ax = plt.subplots(figsize=(14, 3.4))
        hs = [eps[k]["calls"]["hold"] / max(1, eps[k]["calls"]["n_calls"]) for k in keys]
        cs = ["#2e7d32" if eps[k]["success"] else "#c62828" for k in keys]
        ax.bar(range(len(keys)), hs, color=cs)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([f"{k[1].replace('safelibero_spatial', 'SL').replace('libero_spatial', 'L')[:6]}\nt{k[2]}e{k[3]}"
                            for k in keys], fontsize=7)
        ax.set_ylabel(L["hold"]); ax.set_ylim(0, 1)
        ax.set_title(L["holdt"], fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "그래프_HOLD비율.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--arm", default="gt", choices=["gt", "gt2", "gt3termsoff", "gt3pass", "gt3termsoff-pass", "gt4hbc", "gt4termsoff-hbc", "gt4termsoff-pass-hbc", "gt5zr", "gt6oc", "gt7ochbc", "gt8hand", "gt9it8", "gt10ee"], help="GT arm to judge (gt = AG3S-commonto-gt, gt2 = AG3S-commonto-gt2, gt3* = SL-GTTO3 variants)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    eps, runs = load()
    G = a.arm
    eps = {(("gt" if k[0] == G else "_" + k[0]) if k[0] != "vla" else "vla", *k[1:]): v for k, v in eps.items()}
    met = {arm: {b: bench_metrics(eps, arm, b) for b in SCENES} for arm in ("vla", "gt")}
    res = {"arm": G, "runs": runs, "metrics": met, "judge": judge(met["vla"], met["gt"]),
           "episodes": {"/".join(map(str, k)): v for k, v in sorted(eps.items())}}
    md = ["# SL-GTTO 결과 (GT 인식 + 공용 TO vs VLA 단독, 같은 기계 · 같은 commit)", "",
          "| 벤치 | 판 | n | 공식 성공 | 충돌 / 위반 | 안전 성공 | 판정 (task §5) |", "|---|---|---:|---:|---:|---:|---|"]
    for b in SCENES:
        for arm, lab in (("vla", "VLA 단독"), ("gt", "GT + 공용 TO")):
            m = met[arm][b]
            bad = m.get("collisions", m.get("violations", "–"))
            j = res["judge"].get(b) if arm == "gt" else None
            md.append(f"| {b} | {lab} | {m['n']}/{m['expected']} | {m['success']} | {bad} | {m.get('safe_success', m['success'])} | "
                      f"{'' if j is None else ('**통과** ' if j['pass'] else '**실패** ') + j['rule']} |")
    md += ["", f"**전체:** {res['judge'].get('overall', '미완')}", "",
           "## GT 판 episode 별 신호 (실패 층 분류용)", "",
           "| scene | 성공 | 충돌/위반 | status | call | HOLD | attach seq | code1 attach 전→후 | no_cluster | 첫 HOLD (attach 뒤) | C 포화 call (≥5행) | 충돌 행 (link · why) |",
           "|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|"]
    for b in SCENES:
        for s in SCENES[b]:
            r = eps.get(("gt", *s))
            if not r:
                md.append(f"| {s[0]} t{s[1]}e{s[2]} | 기록 없음 ||||||||||| ")
                continue
            c = r.get("calls") or {}
            md.append(f"| {s[0]} t{s[1]}e{s[2]} | {'O' if r['success'] else 'X'} | {'O' if r['collided'] or r['violation'] else ''} | "
                      f"{r['status']} | {c.get('n_calls')} | {c.get('hold')} | {c.get('attach_seq')} | "
                      f"{c.get('code1_before')}→{c.get('code1_after')} | {c.get('no_cluster')} | {c.get('first_hold_after_attach')} | {c.get('sat5')} | "
                      f"{'; '.join(f'{k} ×{v}' for k, v in (c.get('collisions') or {}).items())} |")
    res["sat_share"] = {b: (sum(eps[("gt", *x)]["calls"]["sat5"] / max(1, eps[("gt", *x)]["calls"]["n_calls"])
                                for x in SCENES[b] if ("gt", *x) in eps and eps[("gt", *x)].get("calls"))
                            / max(1, sum(1 for x in SCENES[b] if ("gt", *x) in eps and eps[("gt", *x)].get("calls"))))
                        for b in SCENES}
    md += ["", "episode 평균 C 포화 call 비율: " + ", ".join(f"{b} {v:.2f}" for b, v in res["sat_share"].items())]
    (a.out / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    (a.out / "결과표.md").write_text("\n".join(md) + "\n")
    figures(eps, res, a.out)
    print("\n".join(md))


if __name__ == "__main__":
    main()
