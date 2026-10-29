"""SUBTASK-f V0 summary (copied from SUBTASK-e p/v0_summary.py): collects compare_<run>.json (compare_v0.py) into v0_summary.json. Leaves classified 'other' by
compare_v0.py are listed with their values on both sides so they can be read directly."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
import numpy as np
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/SUBTASK-f/v0'))
out = {"runs": {}}
for n in ("E3b_ep1807_s18071", "E3b_ep1800_s18001"):
    r = json.load(open(V / f"compare_{n}.json"))
    other = {s: r["classified"][s]["other"] for s in r["classified"]}
    vals = {}
    if "scene.model_xml" in other["manifest_leaf_diff"]:
        a = json.load(open(r["run_t39"] + "/manifest.json"))["scene"]["model_xml"]; b = json.load(open(r["run_new"] + "/manifest.json"))["scene"]["model_xml"]
        vals["manifest scene.model_xml"] = {"t39": a, "new": b}
    if "chunk_index" in r.get("server_array_diff", {}):
        ia = [int(np.load(f)["chunk_index"]) for f in sorted(pathlib.Path(r["server_files"]["t39"][0]).parent.glob("chunk_*.npz"))[:1]]
        fa = pathlib.Path(r["server_files"]["t39"][0]); fb = pathlib.Path(r["server_files"]["new"][0])
        za, zb = np.load(fa), np.load(fb)
        vals["server chunk_index (first chunk of the segment)"] = {"t39": int(za["chunk_index"]), "new": int(zb["chunk_index"]),
                                                                   "t39_file": str(fa), "new_file": str(fb),
                                                                   "t_step_first": [int(za["t_step"]), int(zb["t_step"])]}
    out["runs"][n] = {"compare_json": str(V / f"compare_{n}.json"), "run_t39": r["run_t39"], "run_new": r["run_new"],
                      "executed_actions_equal": r["executed_actions_equal"], "planning_actions_equal": r["planning_actions_equal"],
                      "traj_npz_keys_differing": sorted(r["traj_npz_differ"]), "traj_npz_differ_other": r["traj_npz_differ_other"],
                      "frames_rows": r["frames_n_rows"], "frames_other_leaf_diff": other["frames_leaf_diff"],
                      "frames_control_vary_leaf_diff": r["classified"]["frames_leaf_diff"]["control_vary"],
                      "manifest_other_leaf_diff": other["manifest_leaf_diff"],
                      "server_n_chunks": r.get("server_n_chunks"), "server_array_diff": r.get("server_array_diff"),
                      "server_summary_other_leaf_diff": other["server_summary_leaf_diff"],
                      "server_summary_subtask_leaf_diff": r["classified"]["server_summary_leaf_diff"]["subtask"],
                      "server_summary_control_vary_leaf_diff": r["classified"]["server_summary_leaf_diff"]["control_vary"],
                      "named_per_chunk": r["named_per_chunk"], "other_values": vals,
                      "compare_v0_all_identical_flag": r["all_identical"]}
out["note_fact"] = ("model_xml: work-sf rby1_description is git archive 22d8dae (work-se/T39 paths: symlink into ${AG3S_PROJECT_ROOT}/robots/rby1/description); "
                    "chunk_index: server-wide record counter")
for n in ("E3b_ep1807_s18071", "E3b_ep1800_s18001"):
    f = V / f"compare_vsSUBTASKe_{n}.json"
    if f.exists():
        r = json.load(open(f))
        out.setdefault("vs_SUBTASK_e_v0", {})[n] = {"compare_json": str(f), "run_subtask_e": r["run_t39"], "run_new": r["run_new"],
            "executed_actions_equal": r["executed_actions_equal"], "planning_actions_equal": r["planning_actions_equal"],
            "traj_npz_keys_differing": sorted(r["traj_npz_differ"]), "frames_other_leaf_diff": r["classified"]["frames_leaf_diff"]["other"],
            "manifest_other_leaf_diff": r["classified"]["manifest_leaf_diff"]["other"], "server_summary_other_leaf_diff": r["classified"]["server_summary_leaf_diff"]["other"],
            "server_summary_subtask_leaf_diff": r["classified"]["server_summary_leaf_diff"]["subtask"], "server_array_diff_other": r["server_array_diff_other"],
            "named_per_chunk": r["named_per_chunk"]}
(V / "v0_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps({n: {k: v[k] for k in ("executed_actions_equal", "planning_actions_equal", "traj_npz_differ_other", "frames_other_leaf_diff",
                                          "manifest_other_leaf_diff", "server_array_diff", "server_summary_other_leaf_diff", "named_per_chunk", "other_values")}
                  for n, v in out["runs"].items()}, indent=1))
