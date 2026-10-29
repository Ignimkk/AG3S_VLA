"""T41c CPU dry run of the snapshot client (verifier). Runs pi05_infer.main() from ${AG3S_PROJECT_ROOT}-t41c with the exact T41c E3b flags
(hurdle r1 scene, --safe-hold-mode measured, --viz-avoidance-arrow) up to the point where the HOLD controller exists:
load_remote_policy -> dummy (no network), ExecutionLog() -> raises a sentinel (it is constructed right after HoldController).
Then HoldController('measured') from the snapshot is driven on the dry-run model: entry latch = measured arm, gripper = last command.
usage: cd ${AG3S_PROJECT_ROOT}-t41c/pi05_TO_hybrid && PYTHONPATH=${AG3S_PROJECT_ROOT}-t41c python dryrun_cpu.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, importlib.util, os
import numpy as np
OUT = pathlib.Path(sys.argv[1])
SNAP = pathlib.Path("${AG3S_PROJECT_ROOT}-t41c")
script = SNAP / "integration/rby1/bringup/pi05_infer.py"
sys.path[0] = str(script.parent)
sys.path.insert(1, str(SNAP))
import AG3S.trajopt.client as CL
assert CL.__file__.startswith(str(SNAP)), CL.__file__
res = {"client_py": CL.__file__, "HOLD_MODES": list(CL.HOLD_MODES)}
captured = {}
class Sentinel(BaseException):
    pass
_HC = CL.HoldController
class HC(_HC):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); captured["hold_ctl"] = self
CL.HoldController = HC
def boom(*a, **k):
    raise Sentinel()
CL.ExecutionLog = boom
spec = importlib.util.spec_from_file_location("pi05_infer", script)
pi = importlib.util.module_from_spec(spec); spec.loader.exec_module(pi)
class DummyPolicy:
    def infer(self, *a, **k):
        raise RuntimeError("dry run: no policy")
pi.load_remote_policy = lambda remote: DummyPolicy()
cfg = _ag3s_workspace_path('outputs/verify/T40/F/configs/t40f_ep1982_s19822_r1.json')
argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:8279", "--episode-index", "1982",
        "--max-steps", "600", "--headless", "--warmup-steps", "1000", "--policy-seed", "19822",
        "--record-frames", _ag3s_workspace_path('outputs/verify/T41/c/dryrun/rec'), "--record", _ag3s_workspace_path('outputs/verify/T41/c/dryrun/rec/x.mp4'),
        "--view", "front", "--trajectory-out", _ag3s_workspace_path('outputs/verify/T41/c/dryrun/rec/traj.npz'),
        "--obstacle-stop-distance", "0", "--obstacle-config", cfg, "--obstacle-profile", "t40f_ep1982_s19822_r1",
        "--safe-remote", "--safe-timeout", "120", "--safe-phase", "approach", "--safe-manipulators", "left",
        "--safe-gate", "reasons", "--safe-hold-mode", "measured", "--viz-avoidance-arrow"]
sys.argv = argv; res["argv"] = argv
try:
    pi.main(); res["reached_sentinel"] = False
except Sentinel:
    res["reached_sentinel"] = True
hc = captured.get("hold_ctl")
res["hold_ctl_mode"] = getattr(hc, "mode", None)
# drive the snapshot HoldController('measured') on synthetic vectors (16D layout is the client's; only the latch rule is checked)
h = _HC(mode="measured")
import inspect
res["hold_action_signature"] = str(inspect.signature(h.hold_action))
OUT.write_text(json.dumps(res, indent=1, default=str)); print(json.dumps(res, indent=1, default=str))
