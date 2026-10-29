"""T39 S2-0: write the s2_0 section of AG3S/ag3s/docs/handoff/T39.verify.partial.json (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import fcntl, json, pathlib, subprocess, datetime, hashlib
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T39'))
PARTIAL = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T39.verify.partial.json'))
LOCK = T / "p" / ".partial.lock"
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
isv = json.loads((T / "s20/import_check_server.json").read_text())
icl = json.loads((T / "s20/import_check_client.json").read_text())
adr = json.loads((T / "s20/address_check.json").read_text())
dry = json.loads((T / "dry/dry_check.json").read_text())
cmp_ = json.loads((T / "dry/compare_test_smoke_E3b.json").read_text())
cmpn = json.loads((T / "dry/compare_test_smoke_E3b_neg.json").read_text())
s20 = {
  "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
  "gpu_used": False,
  "snapshot": {
    "root": "${AG3S_PROJECT_ROOT}-o1",
    "benchmark": {"source": "git -C ${AG3S_PROJECT_ROOT}/benchmark archive o1-eval", "commit": sh("git -C ${AG3S_PROJECT_ROOT}/benchmark rev-parse o1-eval"),
                  "symlinks": {"AG3S/ag3s/asset/data": _ag3s_workspace_path('AG3S/ag3s/asset/data')}},
    "pi05_TO_hybrid": {"rby1_bringup": "git -C ${AG3S_PROJECT_ROOT}/pi05_TO_hybrid archive 22d8dae rby1_bringup",
                       "commit": sh("git -C ${AG3S_PROJECT_ROOT}/pi05_TO_hybrid rev-parse 22d8dae"),
                       "symlinked_top_level": sorted(p.name for p in pathlib.Path("${AG3S_PROJECT_ROOT}-o1/pi05_TO_hybrid").iterdir() if p.is_symlink())},
    "extra_symlink_not_in_task": {"${AG3S_PROJECT_ROOT}-o1/src": "${AG3S_PROJECT_ROOT}-o1/pi05_TO_hybrid",
                                  "reason_fact": "main has ${AG3S_PROJECT_ROOT}/src -> ${AG3S_PROJECT_ROOT}/pi05_TO_hybrid; serve_safe.main inserts <repo>/third_party/openpi/src into sys.path and asset_path.ASSET_ROOTS tries 'src' first"},
    "code_md5_file": str(T / "p/code_md5_s20.txt"), "code_md5_n_files": len((T / "p/code_md5_s20.txt").read_text().splitlines()),
    "code_md5_file_md5": hashlib.md5((T / "p/code_md5_s20.txt").read_bytes()).hexdigest(),
  },
  "import_check": {
    "server": {"file": str(T / "s20/import_check_server.json"), "all_benchmark_under_work_o1": isv["all_benchmark_under_work_o1"],
               "modules": {k: isv[k] for k in isv if k.startswith("benchmark.")}, "openpi": isv["openpi"],
               "gripper_client_script": isv["gripper_client_script"], "asset_model_transport": isv["asset_model_transport"]},
    "client": {"file": str(T / "s20/import_check_client.json"), "all_benchmark_under_work_o1": icl["all_benchmark_under_work_o1"],
               "REPO_ROOT": icl["REPO_ROOT"], "workspace_root": icl["workspace_root"],
               "modules": {k: icl[k] for k in icl if k.startswith("benchmark.")}, "openpi": icl["openpi"],
               "rby1_manipulation": icl["rby1_manipulation"]},
    "paths_outside_snapshot_files": {
      "RANDOMIZED_16D_DATASET (pi05_infer.py:134-139, absolute default, env RBY1_16D_DATASET)": icl["RANDOMIZED_16D_DATASET"],
      "MODEL_XML_TRANSPORT (REPO_ROOT-relative, resolves through symlink)": icl["MODEL_XML_TRANSPORT_resolved"],
      "rby1_manipulation/src (REPO_ROOT-relative, symlink to main working tree)": _ag3s_workspace_path('integration/rby1/manipulation/src'),
      "openpi / openpi_client (venv .pth editable + serve_safe sys.path insert)": _ag3s_workspace_path('third_party/openpi'),
      "frame_record curobo_commit (frame_record.py:146 absolute)": _ag3s_workspace_path('third_party/curobo'),
      "checkpoint (server --checkpoint, absolute)": _ag3s_workspace_path('legacy/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999')},
    "guard": "run_chain.sh compares code_md5.sh (snapshot benchmark *.py, rby1_bringup *.py, rby1_manipulation/src *.py, mujoco *.xml, openpi src *.py) against p/code_md5_s20.txt and re-runs both import checks before every server start",
  },
  "address_check": {"file": str(T / "s20/address_check.json"), "qposadr_by_name": adr["qposadr_by_name"], "qposadr_hardcoded": adr["qposadr_hardcoded"],
                    "joint_names": adr["joint_names"], "nq": adr["nq"], "n_episodes": len(adr["episodes"]),
                    "dataset_scan_equals_task_list": adr["dataset_scan_equals_task_list"], "all_episodes_consistent": adr["all_episodes_consistent"],
                    "left_gripper_action_col": 7},
  "dry_check": {"file": str(T / "dry/dry_check.json"), "analysis": str(T / "dry/dry_T37E3b_T28E0.json"),
                "all_per_run_equal_to_recorded": dry["all_equal"], "T37_E3b": dry["E3b"], "T37_E3b_reference": dry["T37_E3b_reference"],
                "T28_E0_rec": dry["E0"],
                "hold_path_T30_E3b": str(T / "dry/dry_T30E3b_holdpath.json"),
                "compare_pair_smoke_same_seed_two_servers_bit_identical": cmp_["all_bit_identical"],
                "compare_pair_smoke_seed18071_vs_18072_bit_identical": cmpn["all_bit_identical"]},
  "scripts": {"env": str(T / "p/env.sh"), "start_server": str(T / "p/start_server.sh"), "run_client": str(T / "p/run_client.sh"),
              "run_chain": str(T / "p/run_chain.sh"), "check_run": str(T / "p/check_run.py"), "update_partial": str(T / "p/update_partial.py"),
              "code_md5": str(T / "p/code_md5.sh"), "launch_s21": str(T / "p/launch_s21.sh"), "launch_s22": str(T / "p/launch_s22.sh"),
              "run_lists": sorted(str(p) for p in (T / "p/lists").glob("*.txt")), "compare_pair": str(T / "p/compare_pair.py"),
              "analyze": str(T / "p/analyze_t39.py")},
  "ports": {"s21": [8220, 8221, 8222, 8223], "s22": {"E0": [8224, 8225], "E3b": [8226, 8227]}},
  "gpu_at_s20": {"memory_used_mib": int(sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").splitlines()[0]),
                 "memory_total_mib": int(sh("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits").splitlines()[0]),
                 "budget_gb": 84},
}
LOCK.touch(exist_ok=True)
with open(LOCK, "a") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    cur = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {
        "_comment": "ag3s-verifier partial record for T39 S2 (numbers and paths only). runs[] is rebuilt by outputs/verify/T39/p/update_partial.py after every run.",
        "step": "T39", "stage": "S2"}
    cur["s2_0"] = s20
    tmp = PARTIAL.with_suffix(".json.tmp"); tmp.write_text(json.dumps(cur, indent=1, ensure_ascii=False)); tmp.replace(PARTIAL)
print(json.dumps(s20["gpu_at_s20"]), s20["snapshot"]["benchmark"]["commit"], s20["snapshot"]["pi05_TO_hybrid"]["commit"])
