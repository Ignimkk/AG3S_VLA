"""T40 A step 5: dry-check that pi05_infer.py accepts an obstacle config (verifier). No server, no GPU.
Imports rby1_bringup/pi05_infer.py unchanged and runs main() with a T39-like argv + --obstacle-config/--obstacle-profile;
load_remote_policy is replaced in THIS process only by a stub that records what main() had built up to the point the
policy would be contacted (scene loaded, obstacle manager activated, reset clearance checks passed) and stops.
usage: dryrun_infer.py OUT_JSON CONFIG PROFILE EPISODE SEED [--cf-transport-xml] [extra argv ...]
--cf-transport-xml: in this process only, point MODEL_XML_PICK_PLACE_OBSTACLES at model_transport_pick_place_obstacles.xml
(counterfactual for the 16D + obstacle path; no file is modified)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib, io, contextlib, traceback
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, _ag3s_workspace_path('integration/rby1/bringup'))
import pi05_infer as P  # noqa: E402

out, cfg, prof, ep, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
extra = sys.argv[6:]


class Reached(Exception):
    pass


def stub(remote):
    raise Reached(remote)


P.load_remote_policy = stub
CF = "--cf-transport-xml" in extra
if CF:
    # counterfactual, this process only (no file is changed): what main() would do if the 16D model with an obstacle
    # profile loaded model_transport_pick_place_obstacles.xml instead of model_pick_place_obstacles.xml
    extra = [a for a in extra if a != "--cf-transport-xml"]
    P.MODEL_XML_PICK_PLACE_OBSTACLES = P.MODEL_XML_TRANSPORT_PICK_PLACE_OBSTACLES
import numpy as _np, mujoco as _mj  # noqa: E402
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/A/p'))
import gjk as _G  # noqa: E402
PROBE = {}


class _ProbeManager(P.PickPlaceObstacleManager):
    """records, at the reset check, every obstacle <-> payload pair with mj_geomDistance < 0.25 plus the GJK value of
    the same pair, and the qpos the check saw (measurement only; the check itself is unchanged)."""

    def object_clearance(self, max_distance=0.25):
        v = super().object_clearance(max_distance)
        ft = _np.zeros(6); pairs = []
        for o in self.collision_geom_ids:
            for g in self.payload_geom_ids:
                mj = float(_mj.mj_geomDistance(self.model, self.data, int(o), int(g), max_distance, ft))
                if mj < max_distance:
                    sa, sb = _G.make_support(self.model, self.data, int(o)), _G.make_support(self.model, self.data, int(g))
                    gd = _G.gjk_dist(sa, sb, self.data.geom_xpos[g] - self.data.geom_xpos[o] + 1e-9)
                    pairs.append([_mj.mj_id2name(self.model, _mj.mjtObj.mjOBJ_GEOM, int(o)),
                                  _mj.mj_id2name(self.model, _mj.mjtObj.mjOBJ_GEOM, int(g)), mj, gd])
        PROBE["object_clearance_m"] = v; PROBE["object_pairs_mj_vs_gjk"] = pairs
        PROBE["qpos_at_check"] = self.data.qpos.tolist()
        return v


P.PickPlaceObstacleManager = _ProbeManager
argv = ["pi05_infer.py", "--model", "rby1_randomized_pick_place_16d", "--remote", "localhost:1", "--episode-index", ep,
        "--max-steps", "600", "--headless", "--warmup-steps", "1000", "--policy-seed", seed,
        "--obstacle-config", cfg, "--obstacle-profile", prof] + extra
sys.argv = argv
buf = io.StringIO(); res = {"argv": argv, "config": cfg, "profile": prof, "counterfactual_transport_xml": CF}
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    try:
        P.main(); res["outcome"] = "main returned"
    except Reached:
        res["outcome"] = "reached load_remote_policy (scene + obstacle checks passed)"
    except SystemExit as e:
        res["outcome"] = f"SystemExit({e.code})"
    except Exception as e:  # noqa: BLE001
        res["outcome"] = f"exception {type(e).__name__}: {e}"; res["traceback"] = traceback.format_exc()
res["stdout"] = buf.getvalue()
res["probe"] = PROBE
pathlib.Path(out).write_text(json.dumps(res, indent=1))
print(res["outcome"]); print("\n".join(l for l in res["stdout"].splitlines() if any(k in l for k in ("obstacle", "error", "Model", "scene", "model", "missing"))))
