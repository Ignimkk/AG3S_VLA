"""T24 verifier — tool JSON -> numbers (no interpretation). Writes summary_<run>.json and appends per-chunk rows to handoff/T24.verify.partial.json."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, collections, statistics, pathlib, datetime
import numpy as np

ROOT = pathlib.Path(_ag3s_workspace_path(''))
OUT = ROOT / 'outputs/verify/t24'
PARTIAL = ROOT / 'AG3S/ag3s/docs/handoff/T24.verify.partial.json'
VARS = ("replay", "track_only", "smooth_only", "continuity_only", "collision", "collision_budget_x2", "collision_budget_x4")


def load(name):
    return json.load(open(OUT / f'{name}.json'))


def per_chunk(c):
    row = {"chunk": c['chunk_index'], "seq": c['seq'], "t_step": c.get('t_step', c.get('npz_t_step')), "skipped_reason": c.get('skipped_reason')}
    f = c.get('fidelity') or {}
    if f:
        row["fid_record_clearance_max_mm"] = f['record_clearance']['max_abs_diff_mm']
        row["fid_to_view_mm"] = f['to_view']['abs_diff_mm']
        row["fid_to_view_same_row"] = f['to_view']['same_row']
        row["fid_sphere_step0_mm"] = f.get('sphere_centres_step0_max_diff_mm')
    ref = c.get('reference') or {}
    if ref:
        row["ref_limit_overshoot_max"] = max(ref['limit_overshoot'].values())
    rr = c.get('recorded_refined_vs_reference')
    if rr:
        d = np.array(rr['fingertip_disp_mm_per_step'])
        row["recorded"] = {"joint_dev_max_deg": rr['joint_dev_max_deg'], "tip_med_mm": float(np.median(d)), "tip_max_mm": float(d.max())}
    ch = c.get('replay_chained') or {}
    if ch.get('ran'):
        row["chained"] = {"intact": ch['chain_intact'], "returned": ch['returned'], "legacy_returned": ch['legacy_returned'],
                          "legacy_vs_rec_joint_max_deg": ch['legacy_vs_recorded_refined']['joint_dev_max_deg'],
                          "new_vs_rec_joint_max_deg": ch['new_vs_recorded_refined']['joint_dev_max_deg']}
    if c.get('variants'):
        V = {}
        for v in VARS:
            x = c['variants'].get(v)
            if x is None:
                continue
            ew = x['exec_window']; d = np.array(ew['fingertip_disp_mm_per_step'])
            lg = x.get('legacy') or {}
            e = {"returned": x['returned'], "legacy_returned": lg.get('returned'), "status": x['status'],
                 "joint_dev_max_deg": ew['joint_dev_max_deg'], "tip_med_mm": float(np.median(d)), "tip_max_mm": float(d.max()),
                 "palm_rot_max_deg": ew['palm_rot_deg_max'], "sqp_iter": x['sqp_iterations'], "qp_iter": x['qp_iterations'],
                 "budget_hit": x['time_budget_hit'], "n_cand": x['n_candidates'], "n_acc": x['n_accepted'],
                 "max_violation_mm": x['max_violation_m'] * 1000, "min_clearance_returned_mm": x['clearance_at_returned']['min_mm'],
                 "worst_row_link": x['worst_row'].get('link')}
            lew = lg.get('exec_window')
            if lg.get('differs_from_new') and lew is not None:
                ld = np.array(lew['fingertip_disp_mm_per_step'])
                # legacy exec_window is vs new-rule trajectory? check: recompute below in main
                e["legacy_ew"] = {"joint_dev_max_deg": lew['joint_dev_max_deg'], "tip_med_mm": float(np.median(ld)), "tip_max_mm": float(ld.max())}
            elif lg.get('returned') is not None:
                e["legacy_ew"] = {"joint_dev_max_deg": ew['joint_dev_max_deg'], "tip_med_mm": e['tip_med_mm'], "tip_max_mm": e['tip_max_mm'], "same_as_new": True}
            if v == 'replay':
                vr = x['vs_recorded_refined']; e["vs_recorded_joint_max_deg"] = vr['joint_dev_max_deg']
            V[v] = e
        row["variants"] = V
        ar = c['active_rows_at_reference']
        row["active_rows"] = {"n_selected": ar['n_selected'],
                              "by_link_tier_tf": {f"{k[0]}|{k[1]}|tf={k[2]}": n for k, n in collections.Counter((r['link'], r['tier'], r['target_free']) for r in ar['selected']).items()},
                              "n_negative": ar['n_negative_at_reference'], "negative_by_link": ar['negative_by_link'],
                              "min_clearance_mm": min(ar['min_clearance_per_step_mm']) if ar['min_clearance_per_step_mm'] else None}
    return row


def summarize(name, d, rows):
    s = {"run": name, "valid": d['valid'], "fidelity_summary": d['fidelity_summary'],
         "legacy_selection_counts": d['legacy_selection_counts'],
         "legacy_selection_counts_chained_replay": d['legacy_selection_counts_chained_replay'], "elapsed_s": d['elapsed_s']}
    ran = [r for r in rows if r.get('variants')]
    s["n_chunks"] = len(rows); s["n_ran"] = len(ran)
    s["n_same_row_to_view"] = sum(1 for r in rows if r.get('fid_to_view_same_row'))
    s["n_fid_measured"] = sum(1 for r in rows if 'fid_record_clearance_max_mm' in r)
    s["fid_record_clearance_max_mm"] = max(r['fid_record_clearance_max_mm'] for r in rows if 'fid_record_clearance_max_mm' in r)
    s["fid_to_view_max_mm"] = max(r['fid_to_view_mm'] for r in rows if 'fid_to_view_mm' in r)
    s["n_ref_overshoot_zero_ran"] = sum(1 for r in ran if r['ref_limit_overshoot_max'] == 0)
    s["returned_counts"] = {v: dict(collections.Counter(r['variants'][v]['returned'] for r in ran if v in r['variants'])) for v in VARS}
    s["legacy_counts_recount"] = {v: dict(collections.Counter(r['variants'][v]['legacy_returned'] for r in ran if v in r['variants'])) for v in VARS}
    to = [r['variants']['track_only'] for r in ran if r['ref_limit_overshoot_max'] == 0]
    s["track_only"] = {"n": len(to), "n_joint_dev_exact_zero": sum(1 for e in to if e['joint_dev_max_deg'] == 0.0),
                       "joint_dev_max_deg_over_chunks": max(e['joint_dev_max_deg'] for e in to),
                       "legacy_joint_dev_max_deg_over_chunks": max(e['legacy_ew']['joint_dev_max_deg'] for e in to if 'legacy_ew' in e),
                       "legacy_tip_max_mm_over_chunks": max(e['legacy_ew']['tip_max_mm'] for e in to if 'legacy_ew' in e)}
    ch = [r['chained'] for r in rows if r.get('chained')]
    chi = [x for x in ch if x['intact']]
    s["chained"] = {"n_ran": len(ch), "n_intact": len(chi),
                    "legacy_vs_rec_joint_max_deg_intact": max(x['legacy_vs_rec_joint_max_deg'] for x in chi) if chi else None,
                    "new_vs_rec_joint_max_deg_intact": max(x['new_vs_rec_joint_max_deg'] for x in chi) if chi else None,
                    "legacy_returned_intact": dict(collections.Counter(x['legacy_returned'] for x in chi)),
                    "legacy_returned_all": dict(collections.Counter(x['legacy_returned'] for x in ch)),
                    "returned_new_intact": dict(collections.Counter(x['returned'] for x in chi))}
    rep = [r['variants']['replay']['vs_recorded_joint_max_deg'] for r in ran]
    s["replay_fresh_vs_recorded_joint_max_deg"] = {"min": min(rep), "max": max(rep), "median": float(np.median(rep))}
    return s


def main(names):
    part = json.load(open(PARTIAL)) if PARTIAL.exists() else {"step": "T24", "writer": "ag3s-verifier", "entries": []}
    for name in names:
        d = load(name)
        rows = [per_chunk(c) for c in d['chunks']]
        s = summarize(name, d, rows)
        json.dump({"summary": s, "rows": rows}, open(OUT / f'summary_{name}.json', 'w'), indent=1, default=float)
        part['entries'] = [e for e in part['entries'] if e.get('run') != name]
        part['entries'].append({"run": name, "raw": f"outputs/verify/t24/{name}.json", "written": datetime.datetime.now().isoformat(timespec='seconds'),
                                "summary": s, "per_chunk": rows})
        json.dump(part, open(PARTIAL, 'w'), indent=1, default=float)
        print(name, json.dumps(s, indent=1, default=float))


if __name__ == '__main__':
    main(sys.argv[1:])
