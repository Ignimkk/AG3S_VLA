"""#3: T17 chunks t=72..120 — per-variant fingertip displacement vs reference (exec window, 8 steps)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, numpy as np
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/t24'))
VARS = ("replay", "track_only", "smooth_only", "continuity_only", "collision", "collision_budget_x2", "collision_budget_x4")
res = {}
for run in ("t17_clockoff", "t17_wall"):
    d = json.load(open(OUT / f'{run}.json'))
    chs = [c for c in d['chunks'] if c.get('variants') and 72 <= c['t_step'] <= 120]
    tab = {"chunks": [c['chunk_index'] for c in chs], "t_step": [c['t_step'] for c in chs], "rows": {}}
    def add(name, disp_lists, joint):
        allv = np.concatenate([np.asarray(x) for x in disp_lists])
        tab["rows"][name] = {"per_chunk_median_mm": [float(np.median(x)) for x in disp_lists],
                             "per_chunk_max_mm": [float(np.max(x)) for x in disp_lists],
                             "per_chunk_joint_max_deg": joint,
                             "pooled_median_mm": float(np.median(allv)), "pooled_max_mm": float(allv.max())}
    add("recorded_server_refined", [c['recorded_refined_vs_reference']['fingertip_disp_mm_per_step'] for c in chs],
        [c['recorded_refined_vs_reference']['joint_dev_max_deg'] for c in chs])
    for v in VARS:
        add(v, [c['variants'][v]['exec_window']['fingertip_disp_mm_per_step'] for c in chs],
            [c['variants'][v]['exec_window']['joint_dev_max_deg'] for c in chs])
        # legacy-rule trajectory of the same run
        lg = []
        lj = []
        for c in chs:
            L = c['variants'][v]['legacy']
            ew = L['exec_window'] if L.get('differs_from_new') else c['variants'][v]['exec_window']
            lg.append(ew['fingertip_disp_mm_per_step']); lj.append(ew['joint_dev_max_deg'])
        add(v + "__legacy_rule", lg, lj)
        tab["rows"][v]["returned"] = [c['variants'][v]['returned'] for c in chs]
        tab["rows"][v]["legacy_returned"] = [c['variants'][v]['legacy']['returned'] for c in chs]
        tab["rows"][v]["sqp_iterations"] = [c['variants'][v]['sqp_iterations'] for c in chs]
        tab["rows"][v]["status"] = [c['variants'][v]['status'] for c in chs]
    tab["chained_replay_new_rule_vs_reference"] = None
    res[run] = tab
json.dump(res, open(OUT / 'm3_t17_t72_120.json', 'w'), indent=1)
for run, tab in res.items():
    print('==', run, tab['chunks'], tab['t_step'])
    for k, r in tab['rows'].items():
        print(f"{k:34s} pooled med {r['pooled_median_mm']:8.3f} max {r['pooled_max_mm']:8.3f} | per-chunk med", [round(x, 2) for x in r['per_chunk_median_mm']], 'max', [round(x, 2) for x in r['per_chunk_max_mm']], 'jmax', [round(x, 2) for x in r['per_chunk_joint_max_deg']], r.get('returned', ''))
