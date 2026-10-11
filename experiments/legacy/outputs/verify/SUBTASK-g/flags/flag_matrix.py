"""SUBTASK-g item 4 (no model, no GPU): serve_safe argv -> (reject?, resolved args.subtask_gate, startup note, config section,
manifest subtask.gate value). Uses the snapshot's own functions: build_parser, reject_bad_flag_combinations, resolve_subtask_gate,
subtask_config_section; manifest value = bool(args.subtask_gate) (the expression in serve_safe.main, line printed below).
Also: AG3SConfig default clustering.subtask_gate. usage (cwd ${AG3S_PROJECT_ROOT}-sg, PYTHONPATH=${AG3S_PROJECT_ROOT}-sg): flag_matrix.py OUT_JSON"""
import io, json, sys, contextlib, inspect
import AG3S.trajopt.serve_safe as ss
from AG3S.ag3s import config as agcfg
BASE = ["--config", "c", "--checkpoint", "k", "--model-xml", "m.xml"]
CASES = {"no_flag": [], "--no-subtask-gate": ["--no-subtask-gate"], "--subtask-gate": ["--subtask-gate"],
         "--no-attention": ["--no-attention"], "--no-safe": ["--no-safe"], "--no-perception": ["--no-perception"],
         "--subtask-gate --no-attention": ["--subtask-gate", "--no-attention"],
         "--no-subtask-gate --no-attention": ["--no-subtask-gate", "--no-attention"],
         "--subtask-gate --no-subtask-gate": ["--subtask-gate", "--no-subtask-gate"]}
out = {"serve_safe": ss.__file__, "config": agcfg.__file__, "cases": {}}
for name, extra in CASES.items():
    ap = ss.build_parser(); err = io.StringIO(); r = {"argv_extra": extra}
    try:
        with contextlib.redirect_stderr(err):
            a = ap.parse_args(BASE + extra)
            r["parsed_subtask_gate"] = a.subtask_gate
            ss.reject_bad_flag_combinations(ap, a)
        note = ss.resolve_subtask_gate(a)
        r.update(rejected=False, resolved_subtask_gate=a.subtask_gate, startup_note=note,
                 config_section=ss.subtask_config_section(a.subtask_gate, a.subtask_probe),
                 manifest_subtask_gate=bool(a.subtask_gate))
    except SystemExit as e:
        r.update(rejected=True, exit_code=e.code, stderr_tail=err.getvalue().strip().splitlines()[-1:])
    out["cases"][name] = r
src = inspect.getsource(ss.main)
out["main_manifest_line"] = [l.strip() for l in src.splitlines() if '"subtask": {"gate"' in l]
out["main_build_ag3s_gate_arg"] = [l.strip() for l in src.splitlines() if "subtask_gate=" in l]
try:
    out["config_default_clustering_subtask_gate"] = agcfg.AG3SConfig().clustering.subtask_gate
except Exception as e:
    out["config_default_clustering_subtask_gate"] = f"ERR {e!r}"
out["ClusteringConfig_default_subtask_gate"] = agcfg.ClusteringConfig().subtask_gate
json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps({k: {kk: v.get(kk) for kk in ("rejected", "resolved_subtask_gate", "startup_note", "config_section")} for k, v in out["cases"].items()}, indent=1, ensure_ascii=False))
print(out["config_default_clustering_subtask_gate"], out["main_manifest_line"], out["main_build_ag3s_gate_arg"])
