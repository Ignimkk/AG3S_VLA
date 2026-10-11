"""T40 A step 5: write one obstacle config per selected pair and kind whose best placement blocks (exact replay: window
minimum of min(A gripper, B apple) <= 0 in BOTH E0 and E3b), in the pick_place_obstacles.json format (version 1,
table_surface_z from the stock file, profiles {clear, t40_<ep>_<seed>_<kind>}), then validate each with
rby1_manipulation.simulation.pick_place_obstacles.load_pick_place_obstacle_config.
usage: make_configs.py TAG [TAG ...]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys

sys.path.insert(0, _ag3s_workspace_path('integration/rby1/manipulation/src'))
from rby1_manipulation.simulation.pick_place_obstacles import load_pick_place_obstacle_config  # noqa: E402

RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/A/raw'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/A/configs'))
STOCK = json.load(open(_ag3s_workspace_path('integration/rby1/manipulation/integration/rby1/manipulation/config/pick_place_obstacles.json')))
KIND = {"bollard_0": "bollard", "divider_0": "divider"}
res = {}
for tag in sys.argv[1:]:
    rp = json.load(open(RAW / f"replay_{tag}.json"))
    ep, seed = tag.split("_")
    for slot, k in rp["kinds"].items():
        b = k["best"]
        if b is None or not (b["blocks_E0"] and b["blocks_E3b"]):
            res[f"{tag}_{KIND[slot]}"] = {"written": False, "reason": "best placement does not block both recorded paths"}
            continue
        prof = f"t40_ep{ep}_s{seed}_{KIND[slot]}"
        cfg = {"version": 1, "table_surface_z": STOCK["table_surface_z"],
               "profiles": {"clear": STOCK["profiles"]["clear"],
                            prof: {"scene": "fruit",
                                   "description": f"T40 A: {KIND[slot]} on the recorded apple carry path of T39 episode {ep} "
                                                  f"seed {seed} (E0 + E3b replay, verifier).",
                                   "obstacles": [{"slot": slot, "position": [round(b["xy"][0], 4), round(b["xy"][1], 4), STOCK["table_surface_z"]],
                                                  "yaw": round(b["yaw_rad"], 6)}]}}}
        p = OUT / f"{ep}_{seed}_{KIND[slot]}.json"
        p.write_text(json.dumps(cfg, indent=2) + "\n")
        v = load_pick_place_obstacle_config(p)
        res[f"{tag}_{KIND[slot]}"] = {"written": True, "path": str(p), "profile": prof,
                                      "validated_by": "load_pick_place_obstacle_config", "validated_obstacles": v["profiles"][prof]["obstacles"]}
        print(p, prof, v["profiles"][prof]["obstacles"])
(RAW / "configs_written.json").write_text(json.dumps(res, indent=1))
