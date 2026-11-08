#!/usr/bin/env python3
"""G54 F9 adoption rule as a queue gate (fixed before results, E1_GTTO_DECISIONS_20261104_KO.md G54): exit 0 iff
(i) F9 mechanism: OSQP cap fraction <= 0.10 and no open-finger attach, and
(ii) per benchmark F9 is within 'cannot tell apart' of F8 or better (LIBERO 27: >= F8 - 2; LS 10: >= F8 - 1;
     SafeLIBERO 11: safe success >= F8 - 1 and collisions <= F8 + 1), and the three-benchmark success sum >= F8's.
Reads gtto_protocol27.py / gtto11_mech.py outputs (re-run here)."""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv-ag3s/bin/python")
OUT = ROOT / "outputs/2026-11-07_결과_27_10_11"
MOUT = ROOT / "outputs/2026-11-07_F9_스케일QP_닫힘확인/기제시험"


def main() -> int:
    subprocess.run([PY, str(ROOT / "experiments/segmentation/gtto_protocol27.py"), "--out", str(OUT)], check=True, capture_output=True)
    subprocess.run([PY, str(ROOT / "experiments/segmentation/gtto11_mech.py"), "--out", str(MOUT)], check=True, capture_output=True)
    p = json.loads((OUT / "protocol27.json").read_text())
    m = json.loads((MOUT / "mech11.json").read_text())
    f8, f9 = p["F8"], p["F9"]
    if not (f9["complete"] and f8["complete"]):
        print(f"gate F9: incomplete (F8 {f8['complete']}, F9 {f9['complete']}) -> closed")
        return 1
    s8 = f8["libero_total"] + f8["ls"]["ok"] + f8["safelibero"]["safe_success"]
    s9 = f9["libero_total"] + f9["ls"]["ok"] + f9["safelibero"]["safe_success"]
    checks = {
        "mech cap<=0.10": bool(m["verdict"]["(i-a) F9 cap_frac <= 0.10"]),
        "mech no open attach": bool(m["verdict"]["(i-c') F9 open attaches == 0"]),
        "LIBERO >= F8-2": f9["libero_total"] >= f8["libero_total"] - 2,
        "LS >= F8-1": f9["ls"]["ok"] >= f8["ls"]["ok"] - 1,
        "SafeLIBERO safe >= F8-1": f9["safelibero"]["safe_success"] >= f8["safelibero"]["safe_success"] - 1,
        "SafeLIBERO coll <= F8+1": f9["safelibero"]["collided"] <= f8["safelibero"]["collided"] + 1,
        "sum >= F8": s9 >= s8,
    }
    ok = all(checks.values())
    print(f"gate F9: {checks} sum F8 {s8} F9 {s9} -> {'open' if ok else 'closed'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
