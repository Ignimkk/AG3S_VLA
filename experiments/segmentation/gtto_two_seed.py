#!/usr/bin/env python3
"""SafeLIBERO over two seeds (7 + 8, 22 episodes): success, recorded collisions (1 mm rule), real collisions (robot / handled
object contact, gtto_collision_cause.py), safe success -- per arm. usage: gtto_two_seed.py --out DIR"""
import argparse
import json
import pathlib

R = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
C = pathlib.Path("/mnt/dev/AG3S_VLA/outputs/2026-11-08_SafeLIBERO_충돌_원인")
ARMS = [("VLA 단독", "vla"), ("B‴", "gt21pts"), ("B5", "gt23t100")]


def newest(p):
    runs = [x for x in sorted(R.glob(p)) if (x / "episodes.jsonl").is_file()]
    return runs[-1] if runs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cause = {7: json.loads((C / "collision_cause.json").read_text()), 8: json.loads((C / "s8" / "collision_cause.json").read_text())}
    rows = []
    for label, tag in ARMS:
        tot = {"n": 0, "success": 0, "collided": 0, "safe": 0, "real": 0}
        per = {}
        for seed, sfx in ((7, ""), (8, "s8")):
            k = {"n": 0, "success": 0, "collided": 0, "safe": 0}
            for t in (1, 2, 3):
                run = newest(f"*_gtto-{tag}-safelibero11{sfx}-t{t}-*")
                for e in ([json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()] if run else []):
                    s = e.get("safelibero") or {}
                    k["n"] += 1
                    k["success"] += bool(e.get("official_success"))
                    k["collided"] += bool(s.get("collided"))
                    k["safe"] += bool(s.get("safe_success"))
            k["real"] = sum(1 for c in cause[seed] if c["arm"] == tag and c["cause"] in ("robot", "target/object"))
            per[seed] = k
            for x in tot:
                tot[x] += k[x]
        rows.append({"arm": label, "tag": tag, "per_seed": per, "total": tot})
    (a.out / "two_seed.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    v = rows[0]["total"]
    L = ["| 판 | 성공 /22 (VLA 대비 %p) | 기록상 충돌 | 실제 충돌 | 안전 성공 | seed 7 (성공 · 충돌 · 실제) | seed 8 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["total"]
        gap = 100 * (t["success"] / t["n"] - v["success"] / v["n"]) if t["n"] else 0
        s7, s8 = r["per_seed"][7], r["per_seed"][8]
        L.append(f"| {r['arm']} | {t['success']} ({gap:+.1f}) | {t['collided']} | {t['real']} | {t['safe']} | "
                 f"{s7['success']} · {s7['collided']} · {s7['real']} | {s8['success']} · {s8['collided']} · {s8['real']} |")
    (a.out / "두seed표.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
