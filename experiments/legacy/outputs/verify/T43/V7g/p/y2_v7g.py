"""T43 V7g — Y2 place metrics + Y1 time line per run for arm V5 / V5D / V7g (y_metrics.run unchanged; verifier, CPU only).
usage: y2_v7g.py ARM OUT_JSON      ARM = V5 | V5D | V7g
Inputs: analysis (V5: T43/W/analysis, V5D: T43/Z3V/analysis, V7g: T43/V7g/analysis), physics replay (replay_y.py; V5: T43/Y/replay,
V5D: T43/V7g/replay_V5D or T43/Z3V/replay, V7g: T43/V7g/replay), server chunk records (server_y.one; V5: T43/Y/raw/server_chunks.json).
Definitions of every field: y_metrics.py docstring (copied below)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
sys.path.insert(0, str(T / "Y/p")); sys.path.insert(0, str(T / "W/p"))
import y_metrics as YM  # noqa: E402
import server_y as SY  # noqa: E402

arm, outp = sys.argv[1], pathlib.Path(sys.argv[2])
AN = {"V5": T / "W/analysis/analysis.json", "V5D": T / "Z3V/analysis/analysis.json", "V7g": T / "V7g/analysis/analysis.json"}[arm]
RP = {"V5": T / "Y/replay", "V5D": (T / "V7g/replay_V5D") if (T / "V7g/replay_V5D").exists() or not (T / "Z3V/replay").exists() else (T / "Z3V/replay"), "V7g": T / "V7g/replay"}[arm]
root = T / "V7g/ymroot" / arm; root.mkdir(parents=True, exist_ok=True)
if (root / "replay").is_symlink() or (root / "replay").exists():
    (root / "replay").unlink()
(root / "replay").symlink_to(RP)
YM.Y = root
a = json.load(open(AN))
keys = sorted(k for k in a["runs"] if k.startswith(arm + "/"))
G = YM.Geo()
S = json.load(open(T / "Y/raw/server_chunks.json"))["runs"] if arm == "V5" else {}
res = {"definitions": YM.__doc__, "arm": arm, "replay_dir": str(RP), "runs": {}}
for k in keys:
    nm = k.split("/")
    if not (RP / f"{nm[0]}_{nm[1]}_{nm[2]}.npz").exists():
        print("no replay", k, flush=True); continue
    if k not in S:
        S[k] = SY.one(k, a)
    o = YM.run(k, a, S, G); res["runs"][k] = o
    print(k, "rel", o["release_t"], o.get("release_mode"), "aopen", o["applied_open_t"], flush=True)
    outp.write_text(json.dumps(res, indent=1, default=float))
(outp.parent / f"server_chunks_{arm}.json").write_text(json.dumps({"runs": S}, default=float))
