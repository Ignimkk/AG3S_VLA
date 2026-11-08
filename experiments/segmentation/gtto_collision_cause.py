#!/usr/bin/env python3
"""Summarise gtto_collision_contacts.py output: per arm, each SafeLIBERO 'collided' episode's cause =
robot (a robot0_/gripper0_ body touched the obstacle) · target (the grasped / target object touched it) · none (only the
table / furniture touched it: the 1 mm displacement rule fired without the robot). usage: gtto_collision_cause.py JSON... --out DIR"""
import argparse
import collections
import json
import pathlib
import re


def arm_of(run):
    m = re.search(r"_gtto-([a-z0-9]+)-safelibero", run)
    return m.group(1) if m else run


def cause(e):
    others = {h["other"] for fr in e["frames"] for h in fr["contacts"]}
    if any(o.startswith(("robot0_", "gripper0_", "mount0_")) for o in others):
        return "robot"
    # (2026-11-08 05:4x) the mj_geomDistance gap is recorded but NOT used: it reports overlaps of -2..-127 mm for geom pairs the
    # simulator never put in contact (filtered pairs / helper geoms), so it cannot tell a touch between frames.
    if any("bowl" in o or "plate" in o or "ramekin" in o or "cookie" in o for o in others):
        return "target/object"
    return "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()
    rows = []
    for f in a.files:
        for e in json.loads(pathlib.Path(f).read_text()):
            others = sorted({h["other"] for fr in e["frames"] for h in fr["contacts"]})
            first = next((fr["obs"] for fr in e["frames"] for h in fr["contacts"]
                          if h["other"].startswith(("robot0_", "gripper0_"))), None)
            gaps = [g["min_gap_mm"] for g in e.get("robot_gap") or []]
            rows.append({"arm": arm_of(e["run"]), "stem": e["stem"], "collide_t": e["collide_t"], "cause": cause(e),
                         "min_robot_gap_mm": min(gaps) if gaps else None,
                         "touching": others, "first_robot_contact_obs": first})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "collision_cause.json").write_text(json.dumps(rows, indent=1))
    L = ["# SafeLIBERO '충돌' 의 실제 원인 (MuJoCo contact, 기록된 sim state 재생)", "",
         "| 판 | episode | collide_t | 원인 | 장애물에 닿은 body | 첫 로봇 접촉 obs | 충돌 전후 로봇 최소 간격 mm (참고, 판정에 안 씀) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['arm']} | {r['stem']} | {r['collide_t']} | {r['cause']} | {', '.join(r['touching'])} | {r['first_robot_contact_obs']} | {r.get('min_robot_gap_mm')} |")
    L += ["", "| 판 | 기록상 충돌 | 로봇이 닿음 | 대상 물체가 닿음 | 로봇 없이 (가구 · 탁자만) |", "|---|---|---|---|---|"]
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r["arm"]][r["cause"]] += 1
    for arm, c in by.items():
        L.append(f"| {arm} | {sum(c.values())} | {c['robot']} | {c['target/object']} | {c['none']} |")
    (a.out / "collision_cause표.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
