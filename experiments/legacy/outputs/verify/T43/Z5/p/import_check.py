"""T40 G (copy of T39/s20/import_check.py): which files does a server / client process import? No model is loaded.
usage: import_check.py server|client  -> prints JSON {module: __file__}
server: run with cwd ${AG3S_PROJECT_ROOT}-t43w3, PYTHONPATH=${AG3S_PROJECT_ROOT}-t43w3, via `python -c` (sys.path[0]='' == `-m`),
        then the same sys.path insertion serve_safe.main() does (repo = serve_safe.__file__.parents[2]).
client: run as a script placed next to pi05_infer.py is NOT done (snapshot is immutable); instead sys.path[0] is set
        to the rby1_bringup dir (what `python rby1_bringup/pi05_infer.py` does), the module is exec'd from its real
        path (module level only: imports mujoco, rby1_manipulation, ...), then the workspace_root insertion that every
        benchmark-importing branch of main() does (REPO_ROOT.parent) is applied."""
import json, os, sys, pathlib, importlib, importlib.util
os.environ.setdefault("JAX_PLATFORMS", "cpu")
mode = sys.argv[1]
MODS = ["benchmark", "AG3S.trajopt.wire", "AG3S.trajopt.client", "AG3S.trajopt.policy_seed",
        "AG3S.trajopt.safe_policy", "AG3S.trajopt.serve_safe", "AG3S.ag3s.stages.device_recon",
        "AG3S.ag3s.runtime.frame_record", "AG3S.ag3s.robot_models.gripper_state"]
out = {"mode": mode, "cwd": os.getcwd(), "argv0": sys.argv[0], "PYTHONPATH": os.environ.get("PYTHONPATH")}
if mode == "server":
    import AG3S.trajopt.serve_safe as ss
    repo = pathlib.Path(ss.__file__).resolve().parents[2]
    for p in (repo, repo / "third_party" / "openpi" / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    out["serve_safe_repo"] = str(repo)
    from AG3S.ag3s.robot_models import gripper_state as gs
    try:
        out["gripper_client_script"] = str(gs.find_client_script())
    except Exception as e:
        out["gripper_client_script"] = f"ERR {e!r}"
    from AG3S.ag3s.runtime.asset_path import resolve_asset
    out["asset_model_transport"] = str(resolve_asset("robots/rby1/description/models/rby1a/mujoco/model_transport.xml"))
else:
    script = pathlib.Path(sys.argv[2])
    sys.path[0] = str(script.parent)          # what `python <script>` puts first
    sys.argv = [str(script), "--headless"]
    spec = importlib.util.spec_from_file_location("pi05_infer", script)
    pi = importlib.util.module_from_spec(spec); spec.loader.exec_module(pi)
    out["REPO_ROOT"] = str(pi.REPO_ROOT); out["workspace_root"] = str(pi.REPO_ROOT.parent)
    out["MODEL_XML_TRANSPORT"] = pi.MODEL_XML_TRANSPORT
    out["MODEL_XML_TRANSPORT_resolved"] = str(pathlib.Path(pi.MODEL_XML_TRANSPORT).resolve())
    out["RANDOMIZED_16D_DATASET"] = str(pi.RANDOMIZED_16D_DATASET)
    out["MANIPULATION_SRC"] = str(pi.MANIPULATION_SRC)
    ws = pi.REPO_ROOT.parent
    if str(ws) not in sys.path:
        sys.path.insert(0, str(ws))
    import rby1_manipulation
    out["rby1_manipulation"] = rby1_manipulation.__file__
    from rby1_manipulation.simulation import pick_place_obstacles as ppo
    out["pick_place_obstacles"] = ppo.__file__; out["hurdle_slot"] = "hurdle_0" in ppo.SLOTS
    out["MODEL_XML_TRANSPORT_PICK_PLACE_OBSTACLES_resolved"] = str(pathlib.Path(pi.MODEL_XML_TRANSPORT_PICK_PLACE_OBSTACLES).resolve())
out["sys_path_head"] = sys.path[:6]
for m in MODS:
    try:
        out[m] = importlib.import_module(m).__file__
    except Exception as e:
        out[m] = f"ERR {type(e).__name__}: {e}"
try:
    import openpi, openpi_client
    out["openpi"] = openpi.__file__; out["openpi_client"] = openpi_client.__file__
except Exception as e:
    out["openpi"] = f"ERR {e!r}"
bad = [m for m in MODS if not str(out[m]).startswith("${AG3S_PROJECT_ROOT}-t43w3/")]
if mode == "client":
    bad += [k for k in ("rby1_manipulation", "pick_place_obstacles") if not str(out[k]).startswith("${AG3S_PROJECT_ROOT}-t43w3/")]
out["all_benchmark_under_snapshot"] = not bad; out["not_under_snapshot"] = bad
print(json.dumps(out, indent=1))
