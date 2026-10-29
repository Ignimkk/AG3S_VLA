from common.workspace import workspace_path as _ag3s_workspace_path
import json
r=json.load(open(_ag3s_workspace_path('outputs/verify/T28/P1-2/p12_records.json')))
for name in ('t16','t14'):
    s=r['runs'][name]['summary']
    print(name, {k:v for k,v in s.items() if k not in('argv','hold_chunks_with_policy_grip_cmd_lt_0p9','local_dir')})
    print(' t  ex kind      link          wr_mm  near surf_mm | tgt->apple tgt_near score npts | fing->apple dz  gripcmd0 gripmin meas  srv_inf cli_ms')
    for e in r['runs'][name]['chunks']:
        if not e.get('has_local_row'): print(' ',e['t_step'],'no local row'); continue
        wr=e['worst_row']
        print(f"{e['t_step']:4d} {int(e['executed'])} {str(e['hold_kind']):10s} {str(wr.get('link')):13s} {1000*(wr.get('clearance_m') or 0):6.1f} {str(e.get('worst_nearest_body')):6s} {e.get('worst_nearest_surface_mm',float('nan')):6.1f} {e.get('worst_point_to_apple_centre_mm',float('nan')):6.1f}| {e.get('target_to_apple_centre_mm',float('nan')):6.1f} {str(e.get('target_nearest_body_centre')):6s} {e.get('grounding_best_score') or 0:.3f} {e.get('target_n_points')} | {e['finger_to_apple_surface_mm']:6.1f} {e['apple_dz_mm']:6.1f} {e['policy_grip_l_cmd_rows0_7'][0]:.2f} {e['policy_grip_l_cmd_min_rows0_7']:.2f} {min(e['grip_l_measured_open_frac']):.2f} {e['server_timing_ms']['infer']:.0f} {e['client_policy_infer_ms']:.0f}")
