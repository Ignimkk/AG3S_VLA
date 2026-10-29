"""T43 Z3V: obstacle-stop events per run (client log '[SAFETY] obstacle stop'), V5 (W/runs/V5) vs V5D (Z3V/runs/V5D). usage: stops.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import re, json, glob, sys, pathlib
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43')); out = {}
for arm, root in (("V5", T / "W/runs/V5"), ("V5D", T / "Z3V/runs/V5D")):
    for st in ("H", "B"):
        for lg in sorted(glob.glob(str(root / st / f"{arm}_ep*.client.log"))):
            n = pathlib.Path(lg).name[:-len(".client.log")]; m = re.match(rf"{arm}_ep(\d+)_s(\d+)", n)
            txt = open(lg, errors="replace").read()
            s = re.search(r"\[SAFETY\] obstacle stop: clearance=(-?[0-9.]+) m collision=(\w+)", txt)
            chk = json.load(open(str(root / st / n) + ".check.json"))["checks"]
            out[f"{arm}/{st}:{m.group(1)}_s{m.group(2)}"] = {"stop": bool(s), "clearance_m": float(s.group(1)) if s else None, "collision": s.group(2) if s else None,
                                                             "n_planning": chk["n_planning"], "n_control": chk["n_control"], "mode": chk["mode"]}
json.dump(out, open(sys.argv[1], "w"), indent=1)
for k, v in out.items():
    print(k, v)
