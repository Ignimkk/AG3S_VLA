"""T43 U — post-pass on raw/runs/*.json (measure_u.py):
 (1) recorded TO worst-row points (waypoint max_violation_pair.point_m, path_check.worst.point_m) -> GT nearest surface WITHOUT the
     robot hands (measure_u's timeline used fruit / other fruits / crate / table / obstacle + both hands, so a hand row's own point
     came back as 'hand'). Adds to.wp.point_nearest_nohand / point_dist_mm and the same for to.path.
 (2) T41 a vs T41 c: first planning seq where the two runs' actions differ (max |d| > 0), vs the T41a run's first close_t.
     Written to raw/t41_a_vs_c.json.
usage: [U_SNAP=...] fix_points.py N_WORKERS KEY_SUBSTRING_INCLUDE [KEY_SUBSTRING_EXCLUDE]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
from multiprocessing import Pool
import measure_u as MU

U = MU.U


def one(f):
    r = json.loads(f.read_text())
    if "timeline" not in r:
        return f.name, 0
    RU = json.loads((U / "raw/runs_u.json").read_text()); R = RU["server_runs"][r["key"]]
    obst = MU.place_obstacle(R["obstacle"]); surf = MU.surfaces(R["fruit"], obst)
    rows = [json.loads(l) for l in (pathlib.Path(R["run_dir"]) / "frames.jsonl").read_text().splitlines() if l.strip()]
    plan = {p["seq"]: p for p in rows if p["kind"] == "planning"}
    n = 0
    for e in r["timeline"]:
        q = np.asarray(plan[e["seq"]]["qpos"], float)
        for nm in ("wp", "path"):
            x = e["to"].get(nm)
            if x and x.get("point") is not None:
                o, Rb = MU.base_T(q)
                w = o + Rb @ np.asarray(x["point"], float)
                lab, dd = MU.nearest_label(w, surf)
                x["point_nearest_nohand"] = lab; x["point_dist_mm"] = {k: round(1e3 * v, 1) for k, v in dd.items()}; n += 1
    f.write_text(json.dumps(r, default=float))
    return f.name, n


def t41():
    out = {}
    V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T41'))
    for rd in sorted((V / "a/runs/E3b").glob("*/E3b_ep*")):
        if not rd.is_dir(): continue
        rc = V / "c/runs/E3b" / rd.parent.name / rd.name
        if not (rc / "frames.jsonl").exists(): continue
        a = [json.loads(l) for l in open(rd / "frames.jsonl") if '"planning"' in l[:40]]
        c = [json.loads(l) for l in open(rc / "frames.jsonl") if '"planning"' in l[:40]]
        m = min(len(a), len(c))
        first = next((i for i in range(m) if float(np.abs(np.asarray(a[i]["actions"]) - np.asarray(c[i]["actions"])).max()) > 0), None)
        ra = json.loads((U / "raw/runs" / f"T41__a__{rd.parent.name}__{rd.name}.json").read_text())
        ct = (ra.get("close_ts") or [None])[0]
        out[f"{rd.parent.name}/{rd.name}"] = {"n_a": len(a), "n_c": len(c), "first_diff_seq": None if first is None else a[first]["seq"],
                                              "first_diff_t": None if first is None else a[first]["t_step"], "a_first_close_t": ct,
                                              "identical_through_close_plus_64": bool(first is None or (ct is not None and a[first]["t_step"] > ct + 64)),
                                              "outcome_a": ra["outcome"]}
    (U / "raw/t41_a_vs_c.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    nw = int(sys.argv[1]); inc = sys.argv[2]; exc = sys.argv[3] if len(sys.argv) > 3 else None
    files = [f for f in sorted((U / "raw/runs").glob("*.json")) if inc in f.name and (exc is None or exc not in f.name) and "__E0_" not in f.name]
    with Pool(nw, maxtasksperchild=8) as pool:
        for nm, n in pool.imap_unordered(one, files):
            print(nm, n, flush=True)
    if inc == "T4" and exc:
        t41()
