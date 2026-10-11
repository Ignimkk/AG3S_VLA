"""T37 precheck — key presence and gate records. usage: precheck_t37.py COND PORT RUN N OUT.json   (.venv-openpi-live)
From the server records (server_<PORT>/server_constraints/run_*/chunk_*.npz, first N files and every file):
  to.metrics keys continuity_active / continuity_reason / continuity_off_by present; exec_continuity.{source, off_by}; grasp.state
From the local frames.jsonl of RUN (planning rows): the response 'to' block keys continuity_active / continuity_reason.
From the server log serve_safe_<PORT>.log: every line containing 'continuity term' (gate OFF / back ON lines), with line number."""
import json, sys, pathlib
import numpy as np
COND, PORT, RUN, N, OUTP = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
REC = pathlib.Path(f"${AG3S_PROJECT_ROOT}/outputs/verify/T37/{COND}_rec")
files = sorted((REC / f"server_{PORT}/server_constraints").glob("run_*/chunk_*.npz"))
KEYS = ("continuity_active", "continuity_reason", "continuity_off_by")
out = {"cond": COND, "port": PORT, "run": RUN, "n_chunk_files": len(files), "first_chunks": [], "all_chunks_keys_present": None,
       "gate_on_chunks_server": [], "local_to_block": {}, "server_log_gate_lines": []}
allok = True
for k, f in enumerate(files):
    z = np.load(f, allow_pickle=False); s = json.loads(str(z["summary_json"]))
    tm = (s.get("to") or {}).get("metrics") or {}
    ok = all(x in tm for x in KEYS); allok &= ok
    ec = s.get("exec_continuity") or {}
    row = {"file": str(f.relative_to(REC)), "t_step": int(z["t_step"]), "keys_present": ok, **{x: tm.get(x) for x in KEYS},
           "exec_continuity_source": ec.get("source"), "exec_continuity_off_by": ec.get("off_by"), "grasp_state": (s.get("grasp") or {}).get("state")}
    if k < N:
        out["first_chunks"].append(row)
    if tm.get("continuity_off_by") is not None:
        out["gate_on_chunks_server"].append(row)
out["all_chunks_keys_present"] = allok
fr = REC / RUN / "frames.jsonl"
if fr.exists():
    plan = [json.loads(l) for l in fr.read_text().splitlines() if l.strip()]
    plan = [x for x in plan if x.get("kind") == "planning"]
    tb = [x.get("to") for x in plan]
    out["local_to_block"] = {"n_planning": len(plan), "n_with_to_block": sum(1 for x in tb if isinstance(x, dict)),
                             "n_with_continuity_active_key": sum(1 for x in tb if isinstance(x, dict) and "continuity_active" in x),
                             "n_with_continuity_reason_key": sum(1 for x in tb if isinstance(x, dict) and "continuity_reason" in x),
                             "first_to_block": tb[0] if tb else None,
                             "gate_rows": [{"t": x["t_step"], "to": x.get("to")} for x in plan if isinstance(x.get("to"), dict) and x["to"].get("continuity_active") is False][:40]}
for i, l in enumerate((REC / f"serve_safe_{PORT}.log").read_text(errors="replace").splitlines(), 1):
    if "continuity term" in l:
        out["server_log_gate_lines"].append([i, l.strip()])
pathlib.Path(OUTP).write_text(json.dumps(out, indent=1, default=str, ensure_ascii=False))
print(json.dumps({"n_files": len(files), "all_keys": allok, "first": out["first_chunks"][:3], "n_gate_on": len(out["gate_on_chunks_server"]),
                  "gate_on_t": [r["t_step"] for r in out["gate_on_chunks_server"]], "log_lines": out["server_log_gate_lines"][:10],
                  "local": {k: v for k, v in out["local_to_block"].items() if k != "gate_rows"}}, default=str, ensure_ascii=False, indent=0)[:6000])
