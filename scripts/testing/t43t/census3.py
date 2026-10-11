"""T43 T 3 — grounding replay census: off vs 3(a) / 3(c) / all on every recorded E3b run with an index.
Per chunk: manipulated id/state/mode, distance of the manipulated centroid to the episode's target fruit."""
import json, sys, pathlib, re
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import ground_replay as G

VARIANTS = {"off": {}, "3a": {"latched_identity_hold": True}, "3c": {"supported_max_bottom": 0.06},
            "all": {"latched_identity_hold": True, "refused_identity_release_frames": 3,
                    "supported_max_bottom": 0.06}}


def gate_of(server):
    t = (pathlib.Path(server) / "serve_safe.log").read_text(errors="replace")
    m = re.search(r"subtask label: .*· gate (on|off)", t)
    return None if m is None else m.group(1) == "on"


def one(rd):
    ix = G.IX[rd]
    gate = gate_of(ix["server"])
    man = json.loads((pathlib.Path(rd) / "manifest.json").read_text())
    fruit = man["scene"].get("episode_target_fruit")
    poses = {}
    for l in (pathlib.Path(rd) / "frames.jsonl").read_text().splitlines():
        if '"planning"' in l[:40]:
            r = json.loads(l); poses[r["seq"]] = (r.get("object_poses") or {}).get(fruit, {}).get("pos")
    out = {"run": rd, "gate": gate, "fruit": fruit, "obstacle": man["scene"].get("obstacle_profile"), "variants": {}}
    for name, ov in VARIANTS.items():
        rows = []
        for seq, s, m, last, ref in G.replay(rd, overrides={"clustering": {"subtask_gate": bool(gate), **ov}}):
            rm = s.get("manipulated") or {}
            p = poses.get(seq)
            rows.append({"seq": seq, "id": None if m is None else m.id, "state": None if m is None else m.state,
                         "mode": None if last is None else last.mode, "refused": ref,
                         "latched_hold": bool(last is not None and last.latched_hold),
                         "d_fruit_mm": None if m is None or p is None else round(1e3 * float(np.linalg.norm(np.asarray(m.centroid) - np.asarray(p))), 1),
                         "rec": None if not rm else [rm.get("id"), rm.get("state")], "grasp": (s.get("grasp") or {}).get("state")})
        out["variants"][name] = rows
    return out


def one_cached(rd):
    out_dir = pathlib.Path(sys.argv[1]).with_suffix("")
    out_dir.mkdir(exist_ok=True)
    f = out_dir / (rd.split("/verify/")[-1].replace("/", "_") + ".json")
    if f.exists():
        return json.loads(f.read_text())
    r = one(rd)
    f.write_text(json.dumps(r))
    print("done", rd, flush=True)
    return r


if __name__ == "__main__":
    runs = [rd for rd, v in G.IX.items() if "chunks" in v and v.get("complete")]
    with ProcessPoolExecutor(max_workers=int(sys.argv[2]) if len(sys.argv) > 2 else 2) as ex:
        res = list(ex.map(one_cached, runs))
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res))
    print(len(res))
