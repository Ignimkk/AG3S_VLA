"""T43W copy of T41/c/p/check_run.py (ARM E0 / V0 / V1 / V2: V0 hold mode fixed, V1 / V2 measured; every arm logs '[viz] avoidance arrow on'). T40 G — is one run complete? (copy of T39/p/check_run.py + obstacle-stop case)
usage: check_run.py RUN_DIR COND EP SEED [OBST_PROFILE]  -> JSON on stdout, exit 0 iff complete.
complete = client log ends with exit=0, manifest episode_index == EP and policy_seed == SEED and
timing.warmup_steps == 1000, traj.npz policy_seed == SEED, warmup_steps == 1000, obstacle_profile == OBST_PROFILE (default
'clear'), third_person_front.mp4 present and non-empty, and EITHER (full) completeness.json pass, frames.jsonl 75 planning + 600 control rows,
executed_actions (600, 16), predicted_chunks 75, OR (obstacle stop) the client log has '[SAFETY] obstacle stop', completeness.json consistent (0 duplicates / reversals /
unexplained, planned_missing = expected - captured),
n_control < 600, executed_actions (n_control, 16), n_planning == predicted_chunks == ceil(n_control / 8)."""
import json, math, sys, pathlib
import numpy as np
run, cond, ep, seed = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
prof = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else "clear"
c = {"expected_obstacle_profile": prof}
log = run.parent / f"{run.name}.client.log"
txt = log.read_text(errors="replace") if log.exists() else ""
c["exit0"] = txt.rstrip().endswith("exit=0")
c["obstacle_stop_in_log"] = "[SAFETY] obstacle stop" in txt
try:
    rows = [json.loads(l) for l in (run / "frames.jsonl").read_text().splitlines() if l.strip()]
    c["n_planning"] = sum(r["kind"] == "planning" for r in rows); c["n_control"] = sum(r["kind"] == "control" for r in rows)
except Exception as e:
    c["frames_error"] = repr(e); c["n_planning"] = c["n_control"] = None
try:
    cj = json.loads((run / "completeness.json").read_text())
    c["completeness_pass"] = bool(cj.get("pass"))
    # obstacle stop: frame_record expects 75 observations; the only allowed shortfall is the chunks after the stop
    c["completeness_consistent_if_stopped"] = bool(cj.get("duplicate_sequence_ids") == 0 and cj.get("timestamp_reversals") == 0
        and cj.get("unexplained_carried_or_stale") == 0 and cj.get("captured_observation_frames") == cj.get("planning_records")
        and cj.get("planned_missing_frames") == cj.get("expected_observation_frames", 75) - cj.get("captured_observation_frames", -1))
except Exception as e:
    c["completeness_pass"] = False; c["completeness_consistent_if_stopped"] = False; c["completeness_error"] = repr(e)
try:
    m = json.loads((run / "manifest.json").read_text())
    c["manifest_episode"] = m["scene"].get("episode_index"); c["manifest_seed"] = m.get("policy_seed")
    c["manifest_warmup_steps"] = (m.get("timing") or {}).get("warmup_steps")
    c["manifest_safe_hold_mode"] = (m.get("policy") or {}).get("safe_hold_mode")  # T41c: must be 'measured' for E3b
except Exception as e:
    c["manifest_error"] = repr(e)
try:
    z = np.load(run / "traj.npz")
    c["npz_executed_shape"] = list(z["executed_actions"].shape); c["npz_n_chunks"] = int(z["predicted_chunks"].shape[0])
    c["npz_seed"] = int(z["policy_seed"]) if "policy_seed" in z.files else None
    c["npz_warmup_steps"] = int(z["warmup_steps"]) if "warmup_steps" in z.files else None
    c["npz_obstacle_profile"] = str(z["obstacle_profile"]) if "obstacle_profile" in z.files else None
    c["npz_obstacle_safety"] = json.loads(str(z["obstacle_safety_json"])) if "obstacle_safety_json" in z.files else None
except Exception as e:
    c["npz_error"] = repr(e)
mp4 = run / "third_person_front.mp4"
c["mp4_bytes"] = mp4.stat().st_size if mp4.exists() else 0
common = (c["exit0"] and c.get("manifest_episode") == ep and c.get("manifest_seed") == seed
          and c.get("manifest_warmup_steps") == 1000 and c.get("npz_seed") == seed and c.get("npz_warmup_steps") == 1000
          and c.get("npz_obstacle_profile") == prof and c["mp4_bytes"] > 0
          and (cond == "E0" or c.get("manifest_safe_hold_mode") == {"V8": "measured"}.get(cond))
          and "[viz] avoidance arrow on" in txt)
full = (c["completeness_pass"] and c.get("n_planning") == 75 and c.get("n_control") == 600 and c.get("npz_executed_shape") == [600, 16] and c.get("npz_n_chunks") == 75)
nc = c.get("n_control") or 0
stopped = (c["obstacle_stop_in_log"] and c["completeness_consistent_if_stopped"] and 0 < nc < 600 and c.get("npz_executed_shape") == [nc, 16]
           and c.get("n_planning") == c.get("npz_n_chunks") == math.ceil(nc / 8))
c["mode"] = "full" if full else ("obstacle_stop" if stopped else None)
ok = bool(common and (full or stopped))
print(json.dumps({"run": str(run), "cond": cond, "ep": ep, "seed": seed, "complete": ok, "checks": c}, indent=1))
sys.exit(0 if ok else 1)
