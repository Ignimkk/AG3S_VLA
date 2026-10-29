"""T39 S2-1 — are two runs of the same (condition, episode, seed) on DIFFERENT server processes bit-identical?
usage: compare_pair.py OUT_JSON RUN_A RUN_B
traj.npz      : executed_actions, measured_qpos, predicted_chunks, inference_states, chunk_start_steps (array_equal + max abs)
frames.jsonl  : planning actions / actions_reference / qpos / object_poses; control qpos / applied_ctrl (row by row)
server record : (E3b) for each run the server_constraints segment whose first chunk has summary policy_seed.applied == seed
                (server dir = <run group>/server_<port>_*/server_constraints, port from NAME.start); refined_chunk /
                reference_chunk / clearance per chunk (array_equal + max abs).
Also the first index where anything differs (chunk / control step)."""
import json, sys, pathlib, re
import numpy as np


def load_frames(run):
    rows = [json.loads(l) for l in (run / "frames.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if r["kind"] == "planning"], [r for r in rows if r["kind"] == "control"]


def cmp_arr(a, b):
    a = np.asarray(a, dtype=np.float64) if a is not None else None
    b = np.asarray(b, dtype=np.float64) if b is not None else None
    if a is None or b is None:
        return {"equal": a is None and b is None, "max_abs": None, "note": "missing"}
    if a.shape != b.shape:
        return {"equal": False, "max_abs": None, "note": f"shape {a.shape} vs {b.shape}"}
    return {"equal": bool(np.array_equal(a, b)), "max_abs": float(np.abs(a - b).max()) if a.size else 0.0}


def series(pa, pb, key, sub=None):
    out = {"equal": True, "max_abs": 0.0, "first_diff_index": None, "n": min(len(pa), len(pb))}
    for i in range(out["n"]):
        x, y = pa[i].get(key), pb[i].get(key)
        if sub:
            x = None if x is None else [v for k in sorted(x) for v in (x[k]["pos"] + x[k]["quat"])]
            y = None if y is None else [v for k in sorted(y) for v in (y[k]["pos"] + y[k]["quat"])]
        if key == "applied_ctrl":
            x = None if x is None else list(x.get("arm", [])) + list(x.get("gripper", []))
            y = None if y is None else list(y.get("arm", [])) + list(y.get("gripper", []))
        c = cmp_arr(x, y)
        if not c["equal"]:
            out["equal"] = False
            if out["first_diff_index"] is None:
                out["first_diff_index"] = i
        if c["max_abs"] is not None:
            out["max_abs"] = max(out["max_abs"], c["max_abs"])
    if len(pa) != len(pb):
        out["equal"] = False; out["note"] = f"len {len(pa)} vs {len(pb)}"
    return out


def server_segment(run):
    name = run.name; m = re.match(r"^(E0|E3b)_ep(\d+)_s(\d+)$", name)
    st = run.parent / f"{name}.start"
    if not m or not st.exists():
        return []
    seed = int(m.group(3))
    port = next((l.split("=", 1)[1] for l in st.read_text().splitlines() if l.startswith("port=")), None)
    segs = []
    for rec in sorted(run.parent.glob(f"server_{port}_*/server_constraints/run_*")):
        cur = []
        for f in sorted(rec.glob("chunk_*.npz")):
            z = np.load(f, allow_pickle=True); s = json.loads(str(z["summary_json"]))
            if int(z["t_step"]) == 1 and cur:
                segs.append(cur); cur = []
            cur.append((f, z, s))
        if cur:
            segs.append(cur)
    hit = [sg for sg in segs if (sg[0][2].get("policy_seed") or {}).get("applied") == seed]
    return hit


def main():
    out_p, a, b = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    res = {"run_a": str(a), "run_b": str(b), "definitions": __doc__}
    za, zb = np.load(a / "traj.npz"), np.load(b / "traj.npz")
    res["traj_npz"] = {k: cmp_arr(za[k], zb[k]) for k in ("executed_actions", "measured_qpos", "predicted_chunks", "inference_states", "chunk_start_steps")}
    ex = np.abs(za["executed_actions"] - zb["executed_actions"]).max(axis=1)
    res["traj_npz"]["first_executed_step_diff"] = int(np.argmax(ex > 0)) if (ex > 0).any() else None
    pa, ca = load_frames(a); pb, cb = load_frames(b)
    res["frames"] = {"planning_actions": series(pa, pb, "actions"), "planning_actions_reference": series(pa, pb, "actions_reference"),
                     "planning_qpos": series(pa, pb, "qpos"), "planning_object_poses": series(pa, pb, "object_poses", sub=True),
                     "control_qpos": series(ca, cb, "qpos"), "control_applied_ctrl": series(ca, cb, "applied_ctrl")}
    sa, sb = server_segment(a), server_segment(b)
    res["server_segments_found"] = [len(sa), len(sb)]
    if sa and sb:
        A, B = sa[-1], sb[-1]
        rs = {"n_chunks": [len(A), len(B)]}
        for key in ("refined_chunk", "reference_chunk", "clearance"):
            eq, mx, first = True, 0.0, None
            for i in range(min(len(A), len(B))):
                c = cmp_arr(A[i][1][key], B[i][1][key])
                if not c["equal"]:
                    eq = False; first = i if first is None else first
                if c["max_abs"] is not None:
                    mx = max(mx, c["max_abs"])
            rs[key] = {"equal": eq and len(A) == len(B), "max_abs": mx, "first_diff_chunk": first}
        rs["files_a"] = [str(A[0][0]), str(A[-1][0])]; rs["files_b"] = [str(B[0][0]), str(B[-1][0])]
        res["server_record"] = rs
    flat = [v["equal"] for v in res["traj_npz"].values() if isinstance(v, dict)] + [v["equal"] for v in res["frames"].values()]
    if "server_record" in res:
        flat += [v["equal"] for k, v in res["server_record"].items() if isinstance(v, dict)]
    res["all_bit_identical"] = bool(all(flat))
    out_p.write_text(json.dumps(res, indent=1))
    print(json.dumps({"all_bit_identical": res["all_bit_identical"],
                      "traj": {k: v for k, v in res["traj_npz"].items()},
                      "frames": {k: (v["equal"], v["max_abs"], v["first_diff_index"]) for k, v in res["frames"].items()},
                      "server": res.get("server_record", {}) and {k: v for k, v in res["server_record"].items() if k.endswith("chunk") or k == "clearance"}}, indent=1))


if __name__ == "__main__":
    main()
