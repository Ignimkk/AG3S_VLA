#!/usr/bin/env python3
"""SL-AUD §10 first-episode check (sdm-f6, 2026-11-14): after the -rf fix (0f1d90b4), every action-QP call of the checked
episode must carry `to.ee_action_qp.reference_floor` with `n_rows_below_backoff` and `min_floor_m`, and none of `why`,
`error`, `shape_mismatch`. Exit 0 = OK, 1 = not OK (prints the offending values, incl. clearance_shape / centres_shape).
usage: aud_rf_check.py RUN_DIR [EPISODE_STEM ...]   (no stem: every *_calls.jsonl of the run -- summary only)"""
import glob, json, os, sys


def check(paths):
    n_qp = n_ok = 0
    bad, below, floors = [], [], []
    for p in paths:
        for l in open(p):
            if not l.strip():
                continue
            c = json.loads(l)
            q = (c.get("to") or {}).get("ee_action_qp")
            if not q:
                continue
            n_qp += 1
            rf = q.get("reference_floor")
            ok = (isinstance(rf, dict) and "n_rows_below_backoff" in rf and "min_floor_m" in rf
                  and not any(k in rf for k in ("why", "error", "shape_mismatch")))
            if ok:
                n_ok += 1
                below.append(int(rf["n_rows_below_backoff"]))
                floors.append(float(rf["min_floor_m"]))
            else:
                bad.append((os.path.basename(p), c.get("seq"), rf))
    return n_qp, n_ok, bad, below, floors


def main():
    d = sys.argv[1]
    stems = sys.argv[2:]
    paths = ([f"{d}/episodes/{s}_calls.jsonl" for s in stems] if stems
             else sorted(glob.glob(f"{d}/episodes/*_calls.jsonl")))
    n_qp, n_ok, bad, below, floors = check(paths)
    med = sorted(below)[len(below) // 2] if below else None
    fmed = sorted(floors)[len(floors) // 2] if floors else None
    ok = n_qp > 0 and not bad
    print(f"aud rf-check {os.path.basename(d)[:60]}: {'OK' if ok else 'NOT OK'}  action-QP calls={n_qp} with floor={n_ok}  "
          f"n_rows_below_backoff sum={sum(below)} median/call={med}  min_floor_m median={fmed} min={min(floors) if floors else None}")
    for f, s, rf in bad[:10]:
        print(f"  {f} seq {s}: reference_floor={json.dumps(rf)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
