#!/usr/bin/env python3
"""Queue gate for a pre-fixed widening rule (SL-GTTO, lead 2026-11-05): exit 0 iff the newest run dir matching
`--glob` under results/benchmark has at least `--min` official successes in its episodes.jsonl, else exit 1.

Used as `SOLO if .venv-ag3s/bin/python experiments/segmentation/gtto_gate.py --glob '*_gtto-gt8hand-libero9-*' --min 7;
then <launch>; else echo gate closed; fi` so the follow-up lines run only when the rule written before the result holds.
"""
import argparse
import json
import pathlib
import sys

RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--min", type=int, required=True)
    a = ap.parse_args()
    runs = sorted(RESULTS.glob(a.glob))
    if not runs:
        print(f"gate: no run matches {a.glob}")
        return 1
    p = runs[-1] / "episodes.jsonl"
    n = sum(1 for line in p.read_text().splitlines() if line.strip() and json.loads(line).get("official_success"))
    print(f"gate: {runs[-1].name} official successes {n} (need >= {a.min}) -> {'open' if n >= a.min else 'closed'}")
    return 0 if n >= a.min else 1


if __name__ == "__main__":
    sys.exit(main())
