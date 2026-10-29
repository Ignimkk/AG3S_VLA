"""Run AG3S.trajopt.experiments.esdf_rollout in-process and record which files were imported.
usage: python wrap.py <imports_out.json> -- <esdf_rollout args...>"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, runpy, sys
out = sys.argv[1]
assert sys.argv[2] == "--"
args = sys.argv[3:]
sys.argv = ["esdf_rollout"] + args
rc = 0
try:
    runpy.run_module("AG3S.trajopt.experiments.esdf_rollout", run_name="__main__", alter_sys=True)
except SystemExit as e:
    rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
finally:
    mods = {n: getattr(m, "__file__", None) for n, m in list(sys.modules.items())
            if n == "benchmark" or n.startswith("benchmark.")}
    files = [f for f in mods.values() if f]
    bad = [f for f in files if not f.startswith(_ag3s_workspace_path('benchmark/'))]
    import benchmark
    json.dump({"benchmark.__file__": benchmark.__file__,
               "n_benchmark_modules": len(mods),
               "n_with_file": len(files),
               "n_outside_${AG3S_PROJECT_ROOT}/benchmark": len(bad),
               "outside": bad,
               "device_modules_loaded": sorted(n for n in mods if "device" in n),
               "modules": mods}, open(out, "w"), indent=1)
sys.exit(rc)
