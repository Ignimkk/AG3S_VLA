"""T61 — partial verify (status · paths · sha256 only; no selection / held-out judgement) → handoff/T61.verify.json.

Written at the e4 → B hand-over (2026-10-14): what ran (pre-registration commit, snapshot, cuRobo baseline,
held-out rollouts, extraction stages, calibration selection outputs) and where the files are. B evaluates.

usage: t61_build_verify.py RUN61_DIR
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]


def sha(p: pathlib.Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def rd(p: pathlib.Path):
    return json.loads(p.read_text()) if p.is_file() else None


def main(run: pathlib.Path) -> None:
    v = run / "verify/T61"
    runs = v / "runs/C1a/R"
    names = sorted(p.name for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []
    done = sorted(p.stem for p in runs.glob("*.done")) if runs.is_dir() else []
    stages = (v / "extract_logs/stages.txt").read_text().splitlines() if (v / "extract_logs/stages.txt").is_file() else []
    sel = v / "select/combined"
    sel_files = {str(p.relative_to(v)): sha(p) for p in sorted(sel.rglob("*__scene.json"))} if sel.is_dir() else {}
    sel_files.update({str(p.relative_to(v)): sha(p) for p in sorted((sel / "filter").glob("*.json"))} if (sel / "filter").is_dir() else {})
    chain = (v / "logs/chain_select.log").read_text().splitlines() if (v / "logs/chain_select.log").is_file() else []
    curobo = rd(v / "baseline/curobo_base_summary.json")
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {
        "_comment": "numbers only; status partial — selection / held-out evaluation and judgement belong to session B",
        "step": "T61", "status": "partial", "date": _dt.date.today().isoformat(), "run_dir": str(run),
        "code_state": {"head_at_write": head, "pre_registration_commit": "95b6586c080cdcb630ef03523bc7a376d86c6807",
                       "snapshot_head": (v / "snapshot_head.txt").read_text().splitlines()[0] if (v / "snapshot_head.txt").is_file() else None,
                       "serving_md5_equal_to_T54c_snapshot": True},
        "runs": [{"id": "select", "command": f"{v}/chain_select.sh (t61_select.py over t51b_study, P window, records t2 ep1807 ep0 ep500 ep1200)"},
                 {"id": "rollout", "command": f"nohup setsid {v}/p/launch_t61.sh > {v}/launch_t61.out"},
                 {"id": "extract", "command": f"{v}/p/run_extract.sh"}],
        "baseline_check": {"curobo": curobo},
        "numbers": {
            "heldout_episodes": rd(v / "seeds.json"),
            "rollout": {"n_run_dirs": len(names), "n_done": len(done), "done": done,
                        "agg_partial": rd(v / "agg_partial.json")},
            "extraction_stages": stages,
            "selection_chain_log": chain,
            "selection_outputs_sha256": sel_files,
        },
        "not_measured": ["held-out 3D replay rows of the selected configuration and the baselines",
                         "T60 P / H endpoints of the held-out runs (replay_contacts → t60_termination)",
                         "held-out evaluation (episode-mean identity, paired difference, bootstrap seed 6102)",
                         "serving implementation of the selected configuration (session A)"],
    }
    p = ROOT / "AG3S/ag3s/docs/handoff/T61.verify.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(p, len(names), len(done), len(stages), len(sel_files))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
