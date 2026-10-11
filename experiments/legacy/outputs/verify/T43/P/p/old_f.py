"""T43 P: evaluate the T40 F round-1 placements (F/configs/t40f_ep<EP>_s<SEED>_r1.json) with the T43 P metrics (psearch.PCtx,
bd6a002 distance) + timing. usage: old_f.py TAG... -> raw/old_f_<TAG>.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, time, pathlib
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P/p'))
import psearch as S
for tag in sys.argv[1:]:
    ep, seed = tag.split("_")
    cfgp = pathlib.Path(f"${AG3S_PROJECT_ROOT}/outputs/verify/T40/F/configs/t40f_ep{ep}_s{seed}_r1.json")
    cfg = json.loads(cfgp.read_text()); prof = [k for k in cfg["profiles"] if k != "clear"][0]
    ob = cfg["profiles"][prof]["obstacles"][0]
    P = S.PCtx(tag); t0 = time.time()
    e = P.evaluate(ob["position"][0], ob["position"][1], ob["yaw"], ob["bar_height"]); dt = time.time() - t0
    u, v = P.to_crate(ob["position"][0], ob["position"][1])
    e.update({"config": str(cfgp), "x": ob["position"][0], "y": ob["position"][1], "yaw": ob["yaw"], "h": ob["bar_height"], "u": u, "v": v,
              "eval_s": dt, "warmup_final_vs_row0": P.wu_final_vs_row0}); S.flags(e)
    pathlib.Path(f"${AG3S_PROJECT_ROOT}/outputs/verify/T43/P/raw/old_f_{tag}.json").write_text(json.dumps(e, indent=1, default=float))
    print(tag, f"eval {dt:.3f}s", {k: (round(e[k], 4) if isinstance(e[k], float) else e[k]) for k in ("u", "v", "client_reset_arms", "reset_robot", "wu_robot", "a_min", "b_crate_min", "d_min", "c_min_per_run", "pass_a", "pass_b", "pass_c", "pass_d")}, flush=True)
