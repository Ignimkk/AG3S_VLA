"""(2026-11-09 12:2x dev-4a, offline) tau_obs what-if: over all recorded calls of a run glob, the pass-through calls
(passthrough_clear.applied) whose obstacle-row violation was > X mm (X = 0, 2, 5, 8) -- those would have gone to the action
QP with tau_obs = X. Split by episode outcome. usage: python tau_obs_scan.py '<run glob>'"""
import ast
import collections
import glob
import json
import sys

import numpy as np

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
XS = (0, 2, 5, 8)


def main(pat):
    tot = collections.Counter()
    for d in sorted(glob.glob(B + pat)):
        if "INVALID" in d:
            continue
        for c in sorted(glob.glob(d + "/server_constraints/run_*/chunk_*.npz")):
            sm = json.loads(str(np.load(c)["summary_json"]))
            m = ((sm.get("to") or {}).get("metrics") or {})
            pc = m.get("passthrough_clear")
            if isinstance(pc, str):
                try:
                    pc = ast.literal_eval(pc)
                except Exception:
                    pc = {}
            pc = pc or {}
            tot["calls"] += 1
            if not pc.get("applied"):
                continue
            tot["pass"] += 1
            v = 1000 * float(((pc.get("obstacle_rows") or {}).get("violation_m")) or 0.0)
            for x in XS:
                tot[f"pass_obsviol>{x}mm"] += v > x
    print(pat, dict(tot))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
