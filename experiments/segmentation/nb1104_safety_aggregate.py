#!/usr/bin/env python3
"""nb1104 — LIBERO-Safety Level 0 rollouts (``*_nb1104-safety-<arm>-<suite>*``): per suite x arm counts of the official
metrics of ``libero_safety_client.py`` (fixed before results): success (BDDL goal), official violation
(any(info['cost']) at a policy step; the paper terminates the episode on it), safe success (success and no violation),
also geom-level robot contact (diagnostic, not official) and LDLJ smoothness.

usage: .venv-ag3s/bin/python experiments/segmentation/nb1104_safety_aggregate.py --out DIR
"""
import argparse
import json
import pathlib
import statistics

RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res, md = {}, ["| suite | arm | n | success | official violation | safe success | violation-terminated | full-trace violation (diag., every env step) | geom robot contact (diag.) | per-task success/violation | run |",
                   "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for run in sorted(RESULTS.glob("*_nb1104-safety-*")):
        f = run / "episodes.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        name = run.name.split("_nb1104-safety-", 1)[1]
        arm = name.split("-")[0]
        suite = f'{rows[0]["suite"]} L{rows[0].get("level")}'
        ls = [r["libero_safety"] for r in rows]
        per_task = {}
        for r, s in zip(rows, ls):
            t = per_task.setdefault(int(r["task_id"]), [0, 0, 0])
            t[0] += 1; t[1] += bool(r["official_success"]); t[2] += bool(s["official_violation"])
        s = {"n": len(rows), "success": sum(bool(r["official_success"]) for r in rows),
             "violation": sum(bool(x["official_violation"]) for x in ls), "safe_success": sum(bool(x["safe_success"]) for x in ls),
             "terminated": sum(bool(x["terminated_by_violation"]) for x in ls),
             "geom_contact": sum(bool(x.get("geom_robot_contact")) for x in ls),
             "full_violation": sum(bool(x.get("full_violation")) for x in ls),
             "cost_keys": sorted({k for x in ls for k in x.get("official_cost_keys", [])}),
             "per_task": per_task, "run": run.name,
             "task_descriptions": sorted({r["task_description"] for r in rows})}
        res[f"{suite}/{arm}"] = s
        pt = "; ".join(f"t{k}: {v[1]}/{v[0]} · viol {v[2]}" for k, v in sorted(per_task.items()))
        md.append(f"| {suite} | {arm} | {s['n']} | {s['success']} | {s['violation']} | {s['safe_success']} | {s['terminated']} | "
                  f"{s['full_violation']} | {s['geom_contact']} | {pt} | `{run.name}` |")
    (a.out / "safety_summary.json").write_text(json.dumps(res, indent=1))
    (a.out / "safety_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
