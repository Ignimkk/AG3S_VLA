"""SUBTASK-e V0 — is a gate-off run on the snapshot (main 7098e69) identical to the T39 run of the same (episode, seed)?
usage: compare_v0.py OUT_JSON RUN_T39 RUN_NEW
Compared (every leaf, not a chosen subset):
  traj.npz      every array key (array_equal); keys present in only one file listed.
  frames.jsonl  every row, every leaf (lists compared whole); per leaf path: number of rows differing, first row.
  manifest.json every leaf.
  server record (chunk_*.npz of the run's segment = consecutive chunks starting at t_step 1 whose summary
                policy_seed.applied == seed, server dir = <run group>/server_<port>_*/server_constraints/run_*):
                every array (array_equal) and every summary_json leaf.
Leaf paths that differ are split into
  subtask      : path contains 'subtask' (keys SUBTASK-c added)
  control_vary : path also differs between the two T39 S2-1 E3b runs (ep1807 s18071 on two server processes, same code)
                 -> CONTROL file p/../v0/control_T39_s21.json, made by this script with --control
  other        : everything else  -> all_identical requires other == {} and array/npz diffs == {}.
esdf_stats.target_free (a str embedding wall-clock build_ms) is compared with that one number masked.
Named fields reported separately: executed actions, planning actions, grasp.state (latch), manipulated.id, exclusion.*."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, re, collections
import numpy as np
CONTROL = pathlib.Path(_ag3s_workspace_path('outputs/verify/SUBTASK-e/v0/control_T39_s21.json'))


def flat(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{pre}{k}.")
    else:
        yield pre[:-1], x


def server_segment(run):
    m = re.match(r"^(E3b|E3bG)_ep(\d+)_s(\d+)$", run.name)
    seed = int(m.group(3))
    port = next(l.split("=", 1)[1] for l in (run.parent / f"{run.name}.start").read_text().splitlines() if l.startswith("port="))
    segs = []
    for rec in sorted(run.parent.glob(f"server_{port}_*/server_constraints/run_*")):
        cur = []
        for f in sorted(rec.glob("chunk_*.npz")):
            with np.load(f, allow_pickle=False) as z:
                t = int(z["t_step"])
                if t == 1 and cur:
                    segs.append(cur); cur = []
                cur.append(f)
        if cur:
            segs.append(cur)
    hit = []
    for sg in segs:
        with np.load(sg[0], allow_pickle=False) as z:
            if (json.loads(str(z["summary_json"])).get("policy_seed") or {}).get("applied") == seed:
                hit.append(sg)
    return hit


BUILD_MS = re.compile(r"'build_ms': [-+0-9.e]+")


def norm(d):
    # esdf_stats.target_free is a str(dict) that embeds a wall-clock 'build_ms'; drop only that number
    return {k: (BUILD_MS.sub("'build_ms': X", v) if isinstance(v, str) and "build_ms" in v else v) for k, v in d.items()}


def leafdiff(rows_a, rows_b):
    cnt, first = collections.Counter(), {}
    for i in range(min(len(rows_a), len(rows_b))):
        fa, fb = norm(dict(flat(rows_a[i]))), norm(dict(flat(rows_b[i])))
        for k in set(fa) | set(fb):
            if json.dumps(fa.get(k), sort_keys=True) != json.dumps(fb.get(k), sort_keys=True):
                cnt[k] += 1; first.setdefault(k, i)
    return {k: {"n_rows": cnt[k], "first_row": first[k]} for k in sorted(cnt)}


def compare(a, b):
    res = {"run_t39": str(a), "run_new": str(b)}
    za, zb = np.load(a / "traj.npz"), np.load(b / "traj.npz")
    res["traj_npz_only_in"] = {"t39": sorted(set(za.files) - set(zb.files)), "new": sorted(set(zb.files) - set(za.files))}
    res["traj_npz_differ"] = {}
    for k in sorted(set(za.files) & set(zb.files)):
        x, y = za[k], zb[k]
        if x.shape != y.shape or not np.array_equal(x, y):
            d = {"shape": [list(x.shape), list(y.shape)]}
            if x.shape == y.shape and x.dtype.kind in "fi":
                d["max_abs"] = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())
            res["traj_npz_differ"][k] = d
    ex = np.abs(za["executed_actions"].astype(np.float64) - zb["executed_actions"].astype(np.float64)).max(axis=1)
    res["executed_actions_equal"] = bool(np.array_equal(za["executed_actions"], zb["executed_actions"]))
    res["first_executed_step_diff"] = int(np.argmax(ex > 0)) if (ex > 0).any() else None
    ra = [json.loads(l) for l in (a / "frames.jsonl").read_text().splitlines() if l.strip()]
    rb = [json.loads(l) for l in (b / "frames.jsonl").read_text().splitlines() if l.strip()]
    res["frames_n_rows"] = [len(ra), len(rb)]
    res["frames_kinds_equal"] = [r["kind"] for r in ra] == [r["kind"] for r in rb]
    res["frames_leaf_diff"] = leafdiff(ra, rb)
    ma, mb = json.loads((a / "manifest.json").read_text()), json.loads((b / "manifest.json").read_text())
    res["manifest_leaf_diff"] = leafdiff([ma], [mb])
    sa, sb = server_segment(a), server_segment(b)
    res["server_segments_found"] = [len(sa), len(sb)]
    if sa and sb:
        A, B = sa[-1], sb[-1]
        res["server_n_chunks"] = [len(A), len(B)]
        res["server_files"] = {"t39": [str(A[0]), str(A[-1])], "new": [str(B[0]), str(B[-1])]}
        arr = collections.Counter(); arr_first = {}; sum_a, sum_b = [], []
        for i in range(min(len(A), len(B))):
            with np.load(A[i], allow_pickle=False) as x, np.load(B[i], allow_pickle=False) as y:
                for k in set(x.files) | set(y.files):
                    if k == "summary_json":
                        continue
                    if k not in x.files or k not in y.files or x[k].shape != y[k].shape or not np.array_equal(x[k], y[k]):
                        arr[k] += 1; arr_first.setdefault(k, i)
                sum_a.append(json.loads(str(x["summary_json"]))); sum_b.append(json.loads(str(y["summary_json"])))
        res["server_array_diff"] = {k: {"n_chunks": arr[k], "first_chunk": arr_first[k]} for k in sorted(arr)}
        res["server_summary_leaf_diff"] = leafdiff(sum_a, sum_b)
        res["named_per_chunk"] = {
            "grasp_state_equal": [s.get("grasp", {}).get("state") for s in sum_a] == [s.get("grasp", {}).get("state") for s in sum_b],
            "manipulated_id_equal": [(s.get("manipulated") or {}).get("id") for s in sum_a] == [(s.get("manipulated") or {}).get("id") for s in sum_b],
            "exclusion_equal": [json.dumps(s.get("exclusion"), sort_keys=True) for s in sum_a] == [json.dumps(s.get("exclusion"), sort_keys=True) for s in sum_b],
            "refined_chunk_equal": "refined_chunk" not in res["server_array_diff"],
            "subtask_key_in_t39": sum(("subtask" in s) for s in sum_a), "subtask_key_in_new": sum(("subtask" in s) for s in sum_b)}
    pa = [r for r in ra if r["kind"] == "planning"]; pb = [r for r in rb if r["kind"] == "planning"]
    res["planning_actions_equal"] = [r.get("actions") for r in pa] == [r.get("actions") for r in pb]
    return res


def classify(res, control):
    out = {}
    for sec in ("frames_leaf_diff", "manifest_leaf_diff", "server_summary_leaf_diff"):
        c = control.get(sec, {}) if control else {}
        d = res.get(sec, {})
        out[sec] = {"subtask": sorted(k for k in d if "subtask" in k),
                    "control_vary": sorted(k for k in d if "subtask" not in k and k in c),
                    "other": sorted(k for k in d if "subtask" not in k and k not in c)}
    res["classified"] = out
    ctl_arr = set(control.get("server_array_diff", {})) if control else set()
    res["server_array_diff_other"] = sorted(set(res.get("server_array_diff", {})) - ctl_arr)
    res["traj_npz_differ_other"] = sorted(set(res["traj_npz_differ"]) - set(control.get("traj_npz_differ", {}) if control else ()))
    res["all_identical"] = bool(
        not any(out[s]["other"] for s in out)
        and not res["server_array_diff_other"]
        and not res["traj_npz_differ_other"]
        and not res["traj_npz_only_in"]["t39"]
        and res["frames_kinds_equal"]
        and res["frames_n_rows"][0] == res["frames_n_rows"][1]
        and res.get("server_segments_found") == [1, 1]
        and res.get("server_n_chunks", [0, 1])[0] == res.get("server_n_chunks", [0, 1])[1])
    return res


if __name__ == "__main__":
    if sys.argv[1] == "--control":
        r = compare(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
        r["_comment"] = "control: same code, same (ep, seed), two server processes (T39 S2-1)"
        CONTROL.write_text(json.dumps(r, indent=1))
        print(json.dumps({k: (sorted(r[k]) if isinstance(r.get(k), dict) else r.get(k)) for k in
                          ("traj_npz_differ", "frames_leaf_diff", "manifest_leaf_diff", "server_array_diff", "server_summary_leaf_diff", "named_per_chunk")}, indent=1))
        sys.exit(0)
    control = json.loads(CONTROL.read_text())
    r = classify(compare(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])), control)
    r["definitions"] = __doc__
    pathlib.Path(sys.argv[1]).write_text(json.dumps(r, indent=1))
    print(json.dumps({"all_identical": r["all_identical"], "classified": r["classified"], "server_array_diff_other": r["server_array_diff_other"],
                      "traj_npz_differ_other": r["traj_npz_differ_other"], "traj_npz_only_in": r["traj_npz_only_in"],
                      "executed_actions_equal": r["executed_actions_equal"], "planning_actions_equal": r["planning_actions_equal"],
                      "named_per_chunk": r.get("named_per_chunk"), "server_segments_found": r["server_segments_found"]}, indent=1))
