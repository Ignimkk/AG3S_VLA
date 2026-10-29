"""T40 F: the client's own reset check for a written config: snapshot pi05_infer.main() with --obstacle-config/--obstacle-profile
(--obstacle-stop-distance 0) up to load_remote_policy (stubbed, no server). Prints the client's 'obstacles :' line or its error.
usage: dryrun.py EP CONFIG PROFILE OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, inspect, io, contextlib
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/F/p'))
import flib as F
P = F._client()
ep, cfg, prof, out = sys.argv[1:5]
cap = {}


class Reached(Exception):
    pass


def stub(remote):
    cap.update(inspect.currentframe().f_back.f_locals); raise Reached


P.load_remote_policy = stub
sys.argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1", "--episode-index", ep,
            "--headless", "--obstacle-config", cfg, "--obstacle-profile", prof, "--obstacle-stop-distance", "0"]
buf = io.StringIO(); res = {"ep": int(ep), "config": cfg, "profile": prof}
try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        P.main()
    res["outcome"] = "returned"
except Reached:
    mg = cap["obstacle_manager"]
    res["outcome"] = "reached load_remote_policy"
    res["model_xml"] = cap["model_xml"]; res["active_slots"] = list(mg.active_slots)
    res["robot_clearance_m"] = mg.robot_clearance(); res["object_clearance_m"] = mg.object_clearance()
except SystemExit as e:
    res["outcome"] = f"SystemExit {e.code}"
except Exception as e:
    res["outcome"] = f"exception {type(e).__name__}: {e}"
res["obstacles_line"] = next((l.strip() for l in buf.getvalue().splitlines() if "obstacles  :" in l), None)
res["error_lines"] = [l for l in buf.getvalue().splitlines() if "error" in l.lower()][-3:]
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res))
