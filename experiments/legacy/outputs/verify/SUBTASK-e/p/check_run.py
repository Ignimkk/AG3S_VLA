"""SUBTASK-e (copied unchanged from T39 p/check_run.py) — is one run complete? usage: check_run.py RUN_DIR COND EP SEED  -> JSON on stdout, exit 0 iff complete.
complete = client log ends with exit=0, frames.jsonl has 75 planning + 600 control rows, completeness.json pass,
manifest episode_index == EP and policy_seed == SEED and timing.warmup_steps == 1000, traj.npz executed_actions (600, 16) with
policy_seed == SEED, warmup_steps == 1000, predicted_chunks 75, third_person_front.mp4 present and non-empty."""
import json, sys, pathlib
import numpy as np
run, cond, ep, seed = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
c = {}
log = run.parent / f"{run.name}.client.log"
txt = log.read_text(errors="replace") if log.exists() else ""
c["exit0"] = txt.rstrip().endswith("exit=0")
try:
    rows = [json.loads(l) for l in (run / "frames.jsonl").read_text().splitlines() if l.strip()]
    c["n_planning"] = sum(r["kind"] == "planning" for r in rows); c["n_control"] = sum(r["kind"] == "control" for r in rows)
except Exception as e:
    c["frames_error"] = repr(e); c["n_planning"] = c["n_control"] = None
try:
    c["completeness_pass"] = bool(json.loads((run / "completeness.json").read_text()).get("pass"))
except Exception as e:
    c["completeness_pass"] = False; c["completeness_error"] = repr(e)
try:
    m = json.loads((run / "manifest.json").read_text())
    c["manifest_episode"] = m["scene"].get("episode_index"); c["manifest_seed"] = m.get("policy_seed")
    c["manifest_warmup_steps"] = (m.get("timing") or {}).get("warmup_steps")
except Exception as e:
    c["manifest_error"] = repr(e)
try:
    z = np.load(run / "traj.npz")
    c["npz_executed_shape"] = list(z["executed_actions"].shape); c["npz_n_chunks"] = int(z["predicted_chunks"].shape[0])
    c["npz_seed"] = int(z["policy_seed"]) if "policy_seed" in z.files else None
    c["npz_warmup_steps"] = int(z["warmup_steps"]) if "warmup_steps" in z.files else None
except Exception as e:
    c["npz_error"] = repr(e)
mp4 = run / "third_person_front.mp4"
c["mp4_bytes"] = mp4.stat().st_size if mp4.exists() else 0
ok = (c["exit0"] and c.get("n_planning") == 75 and c.get("n_control") == 600 and c["completeness_pass"]
      and c.get("manifest_episode") == ep and c.get("manifest_seed") == seed and c.get("manifest_warmup_steps") == 1000
      and c.get("npz_executed_shape") == [600, 16] and c.get("npz_n_chunks") == 75 and c.get("npz_seed") == seed
      and c.get("npz_warmup_steps") == 1000 and c["mp4_bytes"] > 0)
print(json.dumps({"run": str(run), "cond": cond, "ep": ep, "seed": seed, "complete": bool(ok), "checks": c}, indent=1))
sys.exit(0 if ok else 1)
