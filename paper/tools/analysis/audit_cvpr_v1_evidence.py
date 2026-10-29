#!/usr/bin/env python3
"""Read-only audit of the CVPR 2027 v1 evidence registry (EXP003-EXP006, R008-R026).

For every registered number this script re-reads the source JSON at the exact
pointer recorded in paper/evidence/NUMBER_MAP_CVPR_V1.md, compares it with the
registered value (floats: |diff| <= 5e-4; counts and strings: exact), prints
PASS/FAIL per result ID, pins the sha256 of every source file and writes
paper/experiments/summaries/CVPR_V1_evidence_audit.json.

Status vocabulary of a check
  PASS                value re-read from JSON equals the registered value
  FAIL                mismatch or missing pointer
  source=worklog      the item exists only in a markdown WORKLOG; the script
                      verifies the quoted text is present and pins the file
                      hash, but the number/sentence is NOT machine-verified
  DERIVED             computed by this script from re-read JSON values (ratio,
                      min/max, per-run count); the inputs are PASS-checked

This is a source-linked re-read (REGISTERED_SOURCE_LINKED), not a 16-D manifest
validation: no NPY array headers, no rollout re-aggregation from per-run
records, no model/simulator/serving code is run or modified.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STORAGE = Path('/mnt/dev/AG3S_VLA-storage/results/development')
OUT = ROOT / 'paper/experiments/summaries/CVPR_V1_evidence_audit.json'
FLOAT_TOL = 5e-4

HELDOUT_RUN = STORAGE / '20261016T030741+0900_f15e587290f6_19ea09c7a8b8'
HURDLE_RUN = STORAGE / '20261017T171201+0900_f15e587290f6_137b5f75d9e3/verify/IFG-HURDLE'
CL1_RUN = STORAGE / '20261016T042508+0900_f15e587290f6_86dde89fb754/verify/IFG-CL1'
HANDOFF = ROOT / 'AG3S/ag3s/docs/handoff'

SOURCES = {
    'HS': HELDOUT_RUN / 'heldout_locked/summary.json',
    'HF': ROOT / 'AG3S/ag3s/docs/figures/ifg/ifg-heldout-locked.json',
    'TW': HANDOFF / 'T43W.verify.json',
    'TV': HANDOFF / 'T43V11AB.verify.json',
    'HA': HURDLE_RUN / 'agg.json',
    'HT': HURDLE_RUN / 'timing.json',
    'CA': CL1_RUN / 'agg.json',
    'CT': CL1_RUN / 'timing.json',
    'T58': HANDOFF / 'T58.verify.json',
    'CL4': HANDOFF / 'CL4.verify.json',
    'T46d': HANDOFF / 'T46d.verify.json',
    'CL2': HANDOFF / 'T44e-CL2.verify.json',
    'T53': HANDOFF / 'T53.verify.json',
    'WL': ROOT / 'doc/active/reviews/vla_dump_20261015/WORKLOG_S0S3_KO.md',
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pget(doc, pointer):
    """RFC 6901 JSON pointer (~1 -> '/', ~0 -> '~'); list indices are integers."""
    cur = doc
    if pointer == '':
        return cur
    for raw in pointer.split('/')[1:]:
        key = raw.replace('~1', '/').replace('~0', '~')
        if isinstance(cur, list):
            cur = cur[int(key)]
        else:
            cur = cur[key]
    return cur


# ---------------------------------------------------------------- registry --
# (result_id, source, pointer, registered value, kind)
# kind: 'count' exact int, 'float' |diff|<=5e-4, 'str' exact, 'contains' substring
C = []


def reg(rid, src, pointer, value, kind):
    C.append(dict(result_id=rid, source=src, pointer=pointer, registered=value, kind=kind))


# ---- EXP003 held-out locked grounding (HS = heldout_locked/summary.json) ----
# R008 episode-mean IoU / precision / recall, 4 arms, population counts
for arm, iou, prec, rec in [
    ('G:N', 0.7730478249129286, 0.8941035687057404, 0.8381835766308009),
    ('A-fk', 0.8320612510959661, 0.9552711342181999, 0.8355078746336647),
    ('A', 0.876093971555441, 0.9490830942021894, 0.8838479458205702),
    ('B', 0.8908732856842837, 0.9656474792528674, 0.8986232884658625),
]:
    reg('R008', 'HS', f'/main_success/{arm}/iou/episode_mean', iou, 'float')
    reg('R008', 'HS', f'/main_success/{arm}/precision/episode_mean', prec, 'float')
    reg('R008', 'HS', f'/main_success/{arm}/recall/episode_mean', rec, 'float')
    reg('R008', 'HS', f'/main_success/{arm}/iou/n_episodes', 34, 'count')
    reg('R008', 'HS', f'/main_success/{arm}/n_valid_frames', 2523, 'count')
reg('R008', 'HS', '/by_stratum/clear/A/iou/n_episodes', 13, 'count')
reg('R008', 'HS', '/by_stratum/adjacent/A/iou/n_episodes', 12, 'count')
reg('R008', 'HS', '/by_stratum/place/A/iou/n_episodes', 9, 'count')
reg('R008', 'HS', '/by_stratum/clear/A/n_valid_frames', 966, 'count')
reg('R008', 'HS', '/by_stratum/adjacent/A/n_valid_frames', 892, 'count')
reg('R008', 'HS', '/by_stratum/place/A/n_valid_frames', 665, 'count')
reg('R008', 'HS', '/n_frames_total', 3196, 'count')
reg('R008', 'HS', '/skipped/s0_invalid', 29, 'count')
reg('R008', 'HS', '/skipped/no_prep', 0, 'count')
reg('R008', 'HS', '/failure_appendix/A/iou/n_episodes', 9, 'count')
reg('R008', 'HS', '/failure_appendix/A/n_valid_frames', 659, 'count')
# stratum-wise IoU (A and G:N) for Table 2 rows
reg('R008', 'HS', '/by_stratum/clear/A/iou/episode_mean', 0.9045400208022639, 'float')
reg('R008', 'HS', '/by_stratum/adjacent/A/iou/episode_mean', 0.8538756627971203, 'float')
reg('R008', 'HS', '/by_stratum/place/A/iou/episode_mean', 0.8646296454322352, 'float')
reg('R008', 'HS', '/by_stratum/clear/G:N/iou/episode_mean', 0.8146361930179686, 'float')
reg('R008', 'HS', '/by_stratum/adjacent/G:N/iou/episode_mean', 0.7685631795479471, 'float')
reg('R008', 'HS', '/by_stratum/place/G:N/iou/episode_mean', 0.7189552648034018, 'float')
reg('R008', 'HS', '/by_stratum/clear/B/iou/episode_mean', 0.9000382395052716, 'float')
reg('R008', 'HS', '/by_stratum/adjacent/B/iou/episode_mean', 0.8822170362604757, 'float')
reg('R008', 'HS', '/by_stratum/place/B/iou/episode_mean', 0.8891766849523789, 'float')
# figure sidecar cross-check (same numbers, points at the summary)
reg('R008', 'HF', '/source', str(SOURCES['HS']), 'str')
reg('R008', 'HF', '/main_success/A/iou/episode_mean', 0.876093971555441, 'float')
reg('R008', 'HF', '/main_success/G:N/iou/episode_mean', 0.7730478249129286, 'float')
reg('R008', 'HF', '/main_success/B/iou/episode_mean', 0.8908732856842837, 'float')

# R009 FP / TP voxel-mass sums (pooled over the 2,523 frames)
for arm, nb, sp, rb, fp, tp in [
    ('G:N', 176069.9713933181, 794.1824287812866, 226.91838952046052, 177091.97938555796, 646534.020614443),
    ('A-fk', 46237.61075299564, 502.3729128190385, 175.16828806817324, 46915.702689606966, 647208.2973103941),
    ('A', 46329.299182381736, 552.2908453349412, 2640.379443738322, 49529.18917135399, 673792.8108286464),
    ('B', 13581.713534193286, 517.4040243042206, 2604.9563646802735, 16711.370546153663, 676155.6294538468),
]:
    reg('R009', 'HS', f'/main_success/{arm}/neighbor_fp_sum', nb, 'float')
    reg('R009', 'HS', f'/main_success/{arm}/support_fp_sum', sp, 'float')
    reg('R009', 'HS', f'/main_success/{arm}/robot_fp_sum', rb, 'float')
    reg('R009', 'HS', f'/main_success/{arm}/FP_sum', fp, 'float')
    reg('R009', 'HS', f'/main_success/{arm}/TP_sum', tp, 'float')

# R010 paired episode differences (bootstrap 2000, seed 20261015)
for key, diff, lo, hi, better, worse in [
    ('A - G:N', 0.10304614664251237, 0.08323494981427215, 0.12313285145943233, 34, 0),
    ('A - A-fk', 0.04403272045947481, 0.03801383050689502, 0.050055427622609866, 34, 0),
    ('B - A', 0.014779314128842732, -0.0116789100514432, 0.03944222857471973, 14, 6),
    ('B - G:N', 0.11782546077135511, 0.08271813740590676, 0.15130667079948473, 32, 2),
]:
    reg('R010', 'HS', f'/main_success/paired_iou/{key}/diff', diff, 'float')
    reg('R010', 'HS', f'/main_success/paired_iou/{key}/ci95/0', lo, 'float')
    reg('R010', 'HS', f'/main_success/paired_iou/{key}/ci95/1', hi, 'float')
    reg('R010', 'HS', f'/main_success/paired_iou/{key}/better', better, 'count')
    reg('R010', 'HS', f'/main_success/paired_iou/{key}/worse', worse, 'count')
    reg('R010', 'HS', f'/main_success/paired_iou/{key}/n_episodes', 34, 'count')
reg('R010', 'HS', '/main_success/paired_precision/A - A-fk/diff', -0.006188040016010591, 'float')
reg('R010', 'HS', '/main_success/paired_precision/A - A-fk/ci95/0', -0.009116464059697633, 'float')
reg('R010', 'HS', '/main_success/paired_precision/A - A-fk/ci95/1', -0.004231042636771202, 'float')
reg('R010', 'HS', '/main_success/paired_precision/A - A-fk/better', 0, 'count')
reg('R010', 'HS', '/main_success/paired_precision/A - A-fk/worse', 34, 'count')
reg('R010', 'HS', '/main_success/paired_recall/A - A-fk/diff', 0.04834007118690548, 'float')
reg('R010', 'HS', '/main_success/paired_recall/A - A-fk/ci95/0', 0.041713586127408664, 'float')
reg('R010', 'HS', '/main_success/paired_recall/A - A-fk/ci95/1', 0.05489205054611608, 'float')
reg('R010', 'HS', '/main_success/paired_recall/A - A-fk/better', 34, 'count')
reg('R010', 'HS', '/main_success/paired_recall/A - A-fk/worse', 0, 'count')
reg('R010', 'HS', '/main_success/B_switch_frames', 125, 'count')
reg('R010', 'HS', '/by_stratum/clear/paired_iou/B - A/diff', -0.004501781296992498, 'float')
reg('R010', 'HS', '/by_stratum/adjacent/paired_iou/B - A/diff', 0.0283413734633552, 'float')
reg('R010', 'HS', '/by_stratum/place/paired_iou/B - A/diff', 0.02454703952014366, 'float')

# R011 IoU by grasp state (pre / held / placed), episode mean over 34 episodes
for state, nvf, gn, afk, a, b in [
    ('pre', 706, 0.8749510904201444, 0.9075024099794847, 0.9698507944447411, 0.9673310030050492),
    ('held', 326, 0.5377397793127421, 0.64040778567183, 0.8475770155578735, 0.8411570863823252),
    ('placed', 1491, 0.7767709801403806, 0.8449822910803859, 0.845807102001963, 0.8728342906503719),
]:
    reg('R011', 'HS', f'/by_grasp_state/{state}/G:N/iou/episode_mean', gn, 'float')
    reg('R011', 'HS', f'/by_grasp_state/{state}/A-fk/iou/episode_mean', afk, 'float')
    reg('R011', 'HS', f'/by_grasp_state/{state}/A/iou/episode_mean', a, 'float')
    reg('R011', 'HS', f'/by_grasp_state/{state}/B/iou/episode_mean', b, 'float')
    reg('R011', 'HS', f'/by_grasp_state/{state}/A/n_valid_frames', nvf, 'count')
    reg('R011', 'HS', f'/by_grasp_state/{state}/A/iou/n_episodes', 34, 'count')

# R012 confidence alert on the held-out successful-rollout frames (locked threshold)
reg('R012', 'HS', '/gate_success/threshold', 0.5904207590672415, 'float')
reg('R012', 'HS', '/gate_success/n_frames', 2420, 'count')
reg('R012', 'HS', '/gate_success/wrong_choices', 100, 'count')
reg('R012', 'HS', '/gate_success/alerts', 238, 'count')
reg('R012', 'HS', '/gate_success/alerts_on_wrong', 85, 'count')
reg('R012', 'HS', '/gate_success/wrong_caught_ratio', 0.85, 'float')
reg('R012', 'HS', '/gate_success/false_alarms', 153, 'count')
reg('R012', 'HS', '/gate_success/fp_removed_ratio', 0.8788327204996255, 'float')
reg('R012', 'HS', '/gate_success/tp_lost_ratio', 0.06698843965768998, 'float')
reg('R012', 'HS', '/gate_success/auroc_wrong', 0.9251551724137931, 'float')
reg('R012', 'HS', '/gate_all/n_frames', 3068, 'count')
reg('R012', 'HS', '/gate_all/wrong_choices', 178, 'count')
reg('R012', 'HS', '/gate_all/auroc_wrong', 0.9242758835193033, 'float')
reg('R012', 'HF', '/gate/auroc_wrong', 0.9251551724137931, 'float')

# ---- EXP004 hurdle closed-loop ------------------------------------------------
# R013 T43 W, arm E0 (VLA only), H set 7 ep x 2 seeds, recorded 2026-10-05
reg('R013', 'TW', '/date', '2026-10-05 (E0', 'contains')
reg('R013', 'TW', '/settings/E0', 'server --no-safe; client no safe flags', 'str')
reg('R013', 'TW', '/code_state/snapshot_1/arms', 'E0 V1 V3 V4', 'str')
reg('R013', 'TW', '/code_state/snapshot_1/snapshot_txt/1', '089516ef9c23be2a1d390d0eb8d6f8990b75d7c6', 'contains')
reg('R013', 'TW', '/numbers/summary_per_arm/E0~1H/n', 14, 'count')
reg('R013', 'TW', '/numbers/summary_per_arm/E0~1H/success', 0, 'count')
reg('R013', 'TW', '/numbers/summary_per_arm/E0~1H/grasp', 11, 'count')
reg('R013', 'TW', '/numbers/summary_per_arm/E0~1H/place', 0, 'count')
reg('R013', 'TW', '/numbers/summary_per_arm/E0~1H/runs_obstacle_stop', 13, 'count')
reg('R013', 'TW', '/numbers/summary_per_arm/E0~1H/runs_client_robot_collision', 13, 'count')
reg('R013', 'TW', '/numbers/summary_per_arm/E0~1H/runs_client_payload_collision', 0, 'count')
reg('R013', 'TW', '/numbers/summary_per_arm/E0~1H/hold_chunks', 0, 'count')
reg('R013', 'TW', '/numbers/summary_per_arm/E0~1H/runs_with_physics_replay_contact', 13, 'count')
reg('R013', 'TW', '/numbers/summary_per_arm/E0~1H/round_trip_ms_pooled/median', 199.33629035949707, 'float')

# R014 T43 V11AB H totals by rule (V8 / V11A / V11AB), 14 runs each, recorded 2026-10-07
reg('R014', 'TV', '/date', '2026-10-07', 'str')
reg('R014', 'TV', '/code_state/head', '3ca1b21', 'str')
for arm, s, g, p, rel, srel, sct in [
    ('V8', 13, 14, 13, 14, 13, 8),
    ('V11A', 10, 14, 10, 14, 10, 8),
    ('V11AB', 12, 14, 12, 13, 12, 12),
]:
    reg('R014', 'TV', f'/numbers/H_totals_by_rule/{arm}/success', s, 'count')
    reg('R014', 'TV', f'/numbers/H_totals_by_rule/{arm}/grasp', g, 'count')
    reg('R014', 'TV', f'/numbers/H_totals_by_rule/{arm}/place', p, 'count')
    reg('R014', 'TV', f'/numbers/H_totals_by_rule/{arm}/released', rel, 'count')
    reg('R014', 'TV', f'/numbers/H_totals_by_rule/{arm}/success_rel', srel, 'count')
    reg('R014', 'TV', f'/numbers/H_totals_by_rule/{arm}/success_ct', sct, 'count')
    reg('R014', 'TV', f'/numbers/totals_old_rule/H/{arm}/n', 14, 'count')
reg('R014', 'TV', '/numbers/H_pairs_z4_ct_rule/V11AB_vs_V8/success_ct/A_only', 5, 'count')
reg('R014', 'TV', '/numbers/H_pairs_z4_ct_rule/V11AB_vs_V8/success_ct/B_only', 1, 'count')
reg('R014', 'TV', '/numbers/H_pairs_z4_ct_rule/V11AB_vs_V8/success_ct/mcnemar_exact_p', 0.21875, 'float')

# R015 T43 V11AB carry-phase apple-hurdle contacts and CT failures among old successes
reg('R015', 'TV', '/numbers/contact_table_by_body_phase/0/body', 'apple', 'str')
reg('R015', 'TV', '/numbers/contact_table_by_body_phase/0/phase', 'carry', 'str')
reg('R015', 'TV', '/numbers/contact_table_by_body_phase/0/n_runs/V8', 5, 'count')
reg('R015', 'TV', '/numbers/contact_table_by_body_phase/0/n_runs/V11A', 2, 'count')
reg('R015', 'TV', '/numbers/contact_table_by_body_phase/0/n_runs/V11AB', 0, 'count')
reg('R015', 'TV', '/numbers/contact_table_by_body_phase/0/n_rows/V8', 10, 'count')
reg('R015', 'TV', '/numbers/contact_table_by_body_phase/0/n_rows/V11A', 9, 'count')
reg('R015', 'TV', '/numbers/contact_table_by_body_phase/0/n_rows/V11AB', 0, 'count')
reg('R015', 'TV', '/numbers/contact_table_by_body_phase/0/max_pen_mm/V11AB', 0.0, 'float')

# R016 IFG-HURDLE arms C1a / C1aN / SURF (same 14 (episode, seed) as T43 H), 2026-10-17
reg('R016', 'HA', '/n_items', 14, 'count')
reg('R016', 'HA', '/n_runs_scored', 42, 'count')
for arm, sold, sct, grasp, pre, nonauth, ostop, hold, cmed, cn, post in [
    ('C1a', 11, 11, 13, 2, 0, 2, 86, 16.533333333333335, 11, 0),
    ('C1aN', 11, 11, 13, 2, 0, 2, 82, 16.533333333333335, 11, 0),
    ('SURF', 8, 7, 9, 3, 1, 3, 11, 15.2, 8, 1),
]:
    reg('R016', 'HA', f'/arms/{arm}/n', 14, 'count')
    reg('R016', 'HA', f'/arms/{arm}/S_old', sold, 'count')
    reg('R016', 'HA', f'/arms/{arm}/S_contact', sct, 'count')
    reg('R016', 'HA', f'/arms/{arm}/grasp', grasp, 'count')
    reg('R016', 'HA', f'/arms/{arm}/pre_release_contact_runs', pre, 'count')
    reg('R016', 'HA', f'/arms/{arm}/post_release_contact_runs', post, 'count')
    reg('R016', 'HA', f'/arms/{arm}/nonauth_target_runs', nonauth, 'count')
    reg('R016', 'HA', f'/arms/{arm}/obstacle_stop_runs', ostop, 'count')
    reg('R016', 'HA', f'/arms/{arm}/hold_chunks_sum', hold, 'count')
    reg('R016', 'HA', f'/arms/{arm}/completion_s/median', cmed, 'float')
    reg('R016', 'HA', f'/arms/{arm}/completion_s/n', cn, 'count')
for arm, fm in [('C1a', {'ok': 11, 'slip_in_transport': 1, 'no_grasp': 1, 'not_placed': 1}),
                ('C1aN', {'ok': 11, 'slip_in_transport': 1, 'no_grasp': 1, 'not_placed': 1}),
                ('SURF', {'ok': 8, 'no_grasp': 5, 'slip_at_lift': 1})]:
    for k, v in fm.items():
        reg('R016', 'HA', f'/arms/{arm}/failure_mode/{k}', v, 'count')
reg('R016', 'HA', '/T43_V11AB_reference/n', 14, 'count')
reg('R016', 'HA', '/T43_V11AB_reference/S_old', 12, 'count')
reg('R016', 'HA', '/T43_V11AB_reference/S_contact', 12, 'count')
reg('R016', 'HA', '/arms/SURF/surface_applied_sum', 510, 'count')
reg('R016', 'HA', '/arms/SURF/surface_layer_chunks_sum', 580, 'count')

# R017 IFG-HURDLE pairs
for field, both, c1an_only, surf_only, neither, p in [
    ('success_ct', 6, 5, 1, 2, 0.21875),
    ('success', 7, 4, 1, 2, 0.375),
]:
    reg('R017', 'HA', f'/pairs/C1aN_vs_SURF/{field}/n_pairs', 14, 'count')
    reg('R017', 'HA', f'/pairs/C1aN_vs_SURF/{field}/both', both, 'count')
    reg('R017', 'HA', f'/pairs/C1aN_vs_SURF/{field}/C1aN_only', c1an_only, 'count')
    reg('R017', 'HA', f'/pairs/C1aN_vs_SURF/{field}/SURF_only', surf_only, 'count')
    reg('R017', 'HA', f'/pairs/C1aN_vs_SURF/{field}/neither', neither, 'count')
    reg('R017', 'HA', f'/pairs/C1aN_vs_SURF/{field}/mcnemar_exact_p', p, 'float')
reg('R017', 'HA', '/pairs/C1a_vs_C1aN/success_ct/both', 11, 'count')
reg('R017', 'HA', '/pairs/C1a_vs_C1aN/success_ct/C1a_only', 0, 'count')
reg('R017', 'HA', '/pairs/C1a_vs_C1aN/success_ct/C1aN_only', 0, 'count')
reg('R017', 'HA', '/pairs/C1a_vs_C1aN/success_ct/neither', 3, 'count')
reg('R017', 'HA', '/pairs/C1a_vs_C1aN/success_ct/mcnemar_exact_p', 1.0, 'float')

# R018 IFG-HURDLE client round trip (first request of each run excluded)
reg('R018', 'HT', '/budget_ms', 533.3333333333334, 'float')
for arm, n, med, p95, stot in [
    ('C1a', 976, 1256.0148239135742, 2034.055233001709, 910.2747109718621),
    ('C1aN', 976, 1177.14262008667, 2151.6876816749573, 855.4650004953146),
    ('SURF', 973, 3330.8560848236084, 4978.579664230347, 2998.2956275343895),
]:
    reg('R018', 'HT', f'/summary/{arm}/n_requests', n, 'count')
    reg('R018', 'HT', f'/summary/{arm}/stages/client_round_trip/median', med, 'float')
    reg('R018', 'HT', f'/summary/{arm}/stages/client_round_trip/p95', p95, 'float')
    reg('R018', 'HT', f'/summary/{arm}/stages/server_total/median', stot, 'float')

# ---- EXP005 held-out closed-loop C1aN vs SURF (IFG-CL1) ------------------------
reg('R019', 'CA', '/n_seeds', 34, 'count')
reg('R019', 'CA', '/C1a_reference/n', 34, 'count')
reg('R019', 'CA', '/C1a_reference/n_success', 34, 'count')
reg('R019', 'CA', '/C1aN/n_runs', 34, 'count')
reg('R019', 'CA', '/C1aN/n_success', 34, 'count')
reg('R019', 'CA', '/C1aN/n_primary', 34, 'count')
reg('R019', 'CA', '/SURF/n_runs', 34, 'count')
reg('R019', 'CA', '/SURF/n_success', 28, 'count')
reg('R019', 'CA', '/SURF/n_primary', 28, 'count')
for tbl in ('pair_primary', 'pair_success'):
    reg('R019', 'CA', f'/{tbl}/both', 28, 'count')
    reg('R019', 'CA', f'/{tbl}/only_C1aN', 6, 'count')
    reg('R019', 'CA', f'/{tbl}/only_SURF', 0, 'count')
    reg('R019', 'CA', f'/{tbl}/neither', 0, 'count')
    reg('R019', 'CA', f'/{tbl}/missing', 0, 'count')
    reg('R019', 'CA', f'/{tbl}/mcnemar_exact_p', 0.03125, 'float')
reg('R019', 'CA', '/SURF/failure_modes/no_grasp', 5, 'count')
reg('R019', 'CA', '/SURF/failure_modes/not_placed', 1, 'count')
reg('R019', 'CA', '/SURF/failure_modes/ok', 25, 'count')
reg('R019', 'CA', '/C1aN/failure_modes/ok', 27, 'count')
reg('R019', 'CA', '/by_stratum/clear/pair_primary/only_C1aN', 4, 'count')
reg('R019', 'CA', '/by_stratum/adjacent/pair_primary/only_C1aN', 1, 'count')
reg('R019', 'CA', '/by_stratum/place/pair_primary/only_C1aN', 1, 'count')
reg('R019', 'CA', '/by_stratum/clear/pair_primary/both', 9, 'count')
reg('R019', 'CA', '/by_stratum/adjacent/pair_primary/both', 11, 'count')
reg('R019', 'CA', '/by_stratum/place/pair_primary/both', 8, 'count')

reg('R020', 'CA', '/C1aN/hold_chunks_sum', 50, 'count')
reg('R020', 'CA', '/SURF/hold_chunks_sum', 19, 'count')
reg('R020', 'CA', '/C1aN/runs_with_hold', 23, 'count')
reg('R020', 'CA', '/SURF/runs_with_hold', 8, 'count')
reg('R020', 'CA', '/C1aN/completion_s/median', 16.0, 'float')
reg('R020', 'CA', '/C1aN/completion_s/n', 34, 'count')
reg('R020', 'CA', '/SURF/completion_s/median', 14.666666666666668, 'float')
reg('R020', 'CA', '/SURF/completion_s/n', 28, 'count')
reg('R020', 'CA', '/C1aN/n_neighbour', 2, 'count')
reg('R020', 'CA', '/SURF/n_neighbour', 2, 'count')
reg('R020', 'CA', '/C1aN/n_nonauth_target', 0, 'count')
reg('R020', 'CA', '/SURF/n_nonauth_target', 0, 'count')
reg('R020', 'CA', '/C1aN/n_obstacle', 0, 'count')
reg('R020', 'CA', '/SURF/n_obstacle', 0, 'count')
reg('R020', 'CA', '/SURF_diagnostics/n_chunks', 2550, 'count')
reg('R020', 'CA', '/SURF_diagnostics/n_applied', 908, 'count')
reg('R020', 'CA', '/SURF_diagnostics/oracle_not_applied_why/identity_unresolved', 43, 'count')
reg('R020', 'CA', '/SURF_diagnostics/oracle_not_applied_why/attention_surface_abstained (no_region)', 8, 'count')
reg('R020', 'CA', '/SURF_diagnostics/oracle_not_applied_why/no_target_free_layer', 1591, 'count')

reg('R021', 'CT', '/budget_ms', 533.3333333333334, 'float')
reg('R021', 'CT', '/summary/C1aN/n_requests', 2516, 'count')
reg('R021', 'CT', '/summary/SURF/n_requests', 2516, 'count')
reg('R021', 'CT', '/summary/C1aN/stages/client_round_trip/median', 1137.4754905700684, 'float')
reg('R021', 'CT', '/summary/C1aN/stages/client_round_trip/p95', 1923.847734928131, 'float')
reg('R021', 'CT', '/summary/SURF/stages/client_round_trip/median', 2857.689142227173, 'float')
reg('R021', 'CT', '/summary/SURF/stages/client_round_trip/p95', 3623.0645775794983, 'float')
reg('R021', 'CT', '/summary/C1aN/stages/server_total/median', 796.4624175801873, 'float')
reg('R021', 'CT', '/summary/SURF/stages/server_total/median', 2586.567224934697, 'float')
reg('R021', 'CT', '/summary/SURF/stages/profile_attention_surface/median', 1663.4485451504588, 'float')
reg('R021', 'CT', '/summary/SURF/stages/surface_build/median', 827.567, 'float')
reg('R021', 'CT', '/summary/SURF/stages/surface_solve/median', 285.10400000000004, 'float')
reg('R021', 'CT', '/summary/SURF/stages/surface_codes/median', 383.465, 'float')
reg('R021', 'CT', '/summary/SURF/stages/surface_codes/n', 2487, 'count')
reg('R021', 'CT', '/summary/SURF/stages/profile_robot_render_mask/median', 158.57859468087554, 'float')
reg('R021', 'CT', '/summary/SURF/stages/render_ms_record/median', 157.227, 'float')
reg('R021', 'CT', '/summary/C1aN/stages/client_round_trip/frac_over_budget', 1.0, 'float')
reg('R021', 'CT', '/summary/SURF/stages/client_round_trip/frac_over_budget', 1.0, 'float')
reg('R021', 'CT', '/summary/C1aN/stages/server_infer_vla_plus_attention/median', 235.64596055075526, 'float')
reg('R021', 'CT', '/summary/SURF/stages/server_infer_vla_plus_attention/median', 220.51269887015224, 'float')
reg('R021', 'CT', '/summary/C1aN/stages/server_ag3s/median', 288.6834470555186, 'float')
reg('R021', 'CT', '/summary/SURF/stages/server_ag3s/median', 2142.0471272431314, 'float')
reg('R021', 'CT', '/summary/C1aN/stages/server_trajopt/median', 187.97238916158676, 'float')
reg('R021', 'CT', '/summary/SURF/stages/server_trajopt/median', 153.17066758871078, 'float')

# ---- EXP006 context results for Limitations -----------------------------------
# R022 T58 adjacent post (3 episodes x E0/ON x with/without post)
reg('R022', 'T58', '/date', '2026-10-13', 'str')
reg('R022', 'T58', '/verdict', 'robot-post contact in 6/6 runs (E0 3/3, ON 3/3)', 'contains')
reg('R022', 'T58', '/verdict', '12.7-38.8 mm outside the hand constraint spheres', 'contains')
for ep in ('1925', '1830', '1623'):
    for arm in ('E0', 'ON'):
        reg('R022', 'T58', f'/numbers/runs/obs~1{arm}_ep{ep}_s{ep}1/success', False, 'str')
        reg('R022', 'T58', f'/numbers/runs/noobs~1{arm}_ep{ep}_s{ep}1/success', True, 'str')
        reg('R022', 'T58', f'/numbers/runs/obs~1{arm}_ep{ep}_s{ep}1/obstacle_rows_by_class/hand-obstacle', 1, 'count')
        reg('R022', 'T58', f'/numbers/runs/obs~1{arm}_ep{ep}_s{ep}1/qpos_mismatch_rows', 0, 'count')

# R023 CL4 (18 dev episode-seed pairs; final geometry)
reg('R023', 'CL4', '/date', '2026-10-15', 'str')
reg('R023', 'CL4', '/execution/runs', 90, 'count')
reg('R023', 'CL4', '/execution/done', 90, 'count')
for arm, s, sct in [('X1', 10, 10), ('X2', 7, 5), ('X3', 9, 7), ('X4', 9, 9), ('G4', 7, 7)]:
    reg('R023', 'CL4', f'/numbers/arms/{arm}/success', s, 'count')
    reg('R023', 'CL4', f'/numbers/arms/{arm}/success_ct', sct, 'count')
    reg('R023', 'CL4', f'/numbers/arms/{arm}/FT_target_contact_runs_any_window', 0, 'count')
    reg('R023', 'CL4', f'/numbers/FT_target_contact_runs/{arm}', 0, 'count')
reg('R023', 'CL4', '/numbers/pairs_agg/X4_vs_G4/success_ct/n_pairs', 18, 'count')
reg('R023', 'CL4', '/numbers/pairs_agg/X4_vs_G4/success_ct/both', 5, 'count')
reg('R023', 'CL4', '/numbers/pairs_agg/X4_vs_G4/success_ct/A_only', 4, 'count')
reg('R023', 'CL4', '/numbers/pairs_agg/X4_vs_G4/success_ct/B_only', 2, 'count')
reg('R023', 'CL4', '/numbers/pairs_agg/X4_vs_G4/success_ct/neither', 7, 'count')
reg('R023', 'CL4', '/numbers/pairs_agg/X4_vs_G4/success_ct/A_total', 9, 'count')
reg('R023', 'CL4', '/numbers/pairs_agg/X4_vs_G4/success_ct/B_total', 7, 'count')
reg('R023', 'CL4', '/numbers/pairs_agg/X4_vs_G4/success_ct/mcnemar_exact_p', 0.6875, 'float')
reg('R023', 'CL4', '/numbers/before/CL2_C0/success', 14, 'count')
reg('R023', 'CL4', '/numbers/before/CL2_C0/success_ct', 12, 'count')

# R024 T46d gripper-scope degeneracy (pre8close window)
reg('R024', 'T46d', '/date', '2026-10-12', 'str')
for arm in ('C1aF1', 'C1a', 'C2p'):
    reg('R024', 'T46d', f'/numbers/item1_degeneracy_gripper_scope/{arm}/pre8close/nonauth_rows_evaluated', 349920, 'count')
    reg('R024', 'T46d', f'/numbers/item1_degeneracy_gripper_scope/{arm}/pre8close/nonauth_rows_near_tgt_plan16', 0, 'count')
    reg('R024', 'T46d', f'/numbers/item1_degeneracy_gripper_scope/{arm}/pre8close/nonauth_rows_near_tgt_exec', 0, 'count')
    reg('R024', 'T46d', f'/numbers/item1_degeneracy_gripper_scope/{arm}/pre8close/chunks', 162, 'count')
reg('R024', 'T46d', '/verdict_facts/gripper_scope_degenerate_nonauth_rows_near_target', 0, 'count')

# R025 T44e CL2 arm C0 (V11AB flags + --to-fast-eval --esdf-host-mirror), 18 runs
reg('R025', 'CL2', '/date', '2026-10-12', 'str')
reg('R025', 'CL2', '/numbers/arms/C0/t1_success/n_runs', 18, 'count')
reg('R025', 'CL2', '/numbers/arms/C0/t1_success/task_success', 14, 'count')
reg('R025', 'CL2', '/numbers/arms/C0/t1_success/success_ct', 12, 'count')
reg('R025', 'CL2', '/numbers/arms/C0/t4_hold/hold_chunks_total', 101, 'count')

# R026 T53 current-serving test set (16 episodes, C1a)
reg('R026', 'T53', '/date', '2026-10-13', 'str')
reg('R026', 'T53', '/numbers/n_episodes', 16, 'count')
reg('R026', 'T53', '/numbers/runs_complete', 16, 'count')
reg('R026', 'T53', '/numbers/task_success', 9, 'count')
reg('R026', 'T53', '/numbers/success_ct', 7, 'count')
reg('R026', 'T53', '/numbers/hold_chunks_total', 37, 'count')
reg('R026', 'T53', '/numbers/hold_runs', 10, 'count')
reg('R026', 'T53', '/numbers/replay_qpos_mismatch_rows_total', 0, 'count')

# ---- WORKLOG-only items (text; cannot be machine-verified) --------------------
WORKLOG_ITEMS = [
    dict(result_id='R019', item='EXP005 registered interpretation sentence (WORKLOG 2026-10-18 14:30)',
         text='C1a가 성공한 34개 조건에서 C1aN은 34개, SURF는 28개를 성공했다. 이는 기준선 성공 사례의 보존 여부를 보여주지만, '
              '성공 여부로 선별하지 않은 전체 장면에서의 성능 차이를 추정하지는 못한다.'),
    dict(result_id='R017', item='WORKLOG 2026-10-17 23:40 discordant-pair wording that differs from agg.json (JSON is authoritative)',
         text='C1aN vs SURF: C1aN만 성공 6, SURF만 성공 2, p = 0.22'),
    dict(result_id='R020', item='WORKLOG 2026-10-17 03:30 completion-time scope wording ("둘 다 성공 28쌍"); agg.json C1aN median is over n=34',
         text='완료 시간 median (둘 다 성공 28쌍)'),
]


# ---------------------------------------------------------------- audit --------
def compare(kind, registered, found):
    if kind == 'count':
        return isinstance(found, int) and not isinstance(found, bool) and found == registered
    if kind == 'float':
        return isinstance(found, (int, float)) and not isinstance(found, bool) and math.isfinite(float(found)) \
            and abs(float(found) - float(registered)) <= FLOAT_TOL
    if kind == 'str':
        return found == registered
    if kind == 'contains':
        return isinstance(found, str) and registered in found
    raise ValueError(kind)


def derived(docs):
    """Values computed from re-read JSON (inputs are PASS-checked above)."""
    out = []
    hs = docs['HS']['main_success']
    gn, a, b = hs['G:N']['neighbor_fp_sum'], hs['A']['neighbor_fp_sum'], hs['B']['neighbor_fp_sum']
    out.append(dict(result_id='R009', name='neighbor_fp_reduction_A_vs_GN', value=dict(value=1 - a / gn),
                    formula='1 - /main_success/A/neighbor_fp_sum / /main_success/G:N/neighbor_fp_sum'))
    out.append(dict(result_id='R009', name='neighbor_fp_reduction_B_vs_GN', value=dict(value=1 - b / gn),
                    formula='1 - /main_success/B/neighbor_fp_sum / /main_success/G:N/neighbor_fp_sum'))
    pr = docs['TW']['numbers']['per_run']
    e0 = [v for k, v in pr.items() if k.startswith('E0/H/')]
    out.append(dict(result_id='R013', name='E0_H_per_run', value=dict(
        n=len(e0), success=sum(bool(r['success']) for r in e0), grasp=sum(bool(r['grasp']) for r in e0),
        place=sum(bool(r['place']) for r in e0), obstacle_stop=sum(bool(r['obstacle_stop']) for r in e0)),
        formula='counts over /numbers/per_run keys with prefix E0/H/ (expected n 14, success 0, grasp 11, place 0, obstacle_stop 13)'))
    ct = docs['TV']['numbers']['ct_contact_fail_among_old_success']
    out.append(dict(result_id='R015', name='ct_contact_fail_among_old_success_len',
                    value={k: len(v) for k, v in ct.items()},
                    formula='len(/numbers/ct_contact_fail_among_old_success/<arm>) (expected V8 5, V11A 2, V11AB 0)'))
    runs = docs['T58']['numbers']['runs']
    obs = {k: v for k, v in runs.items() if k.startswith('obs/')}
    contact = {k: sum(v['obstacle_rows_by_class'].values()) > 0 for k, v in obs.items()}
    out.append(dict(result_id='R022', name='T58_post_contact_runs', value=dict(
        with_post_runs=len(obs), with_post_contact=sum(contact.values()),
        E0=sum(c for k, c in contact.items() if '/E0_' in k), ON=sum(c for k, c in contact.items() if '/ON_' in k)),
        formula='sum(/numbers/runs/obs~1*/obstacle_rows_by_class) > 0 (expected 6 of 6; E0 3, ON 3)'))
    rows = [r['closest_point_outside_spheres_mm'] for ep in docs['T58']['numbers']['on_diagnosis'].values()
            for r in ep['cover_rows']]
    out.append(dict(result_id='R022', name='T58_closest_point_outside_spheres_mm', value=dict(min=min(rows), max=max(rows), n_rows=len(rows)),
                    formula='min/max over /numbers/on_diagnosis/*/cover_rows/*/closest_point_outside_spheres_mm (expected 12.72 / 38.8)'))
    return out


DERIVED_EXPECT = {
    # ratios implied by the registered R009 sums (manuscript rounds to -74 % / -92 %)
    ('R009', 'neighbor_fp_reduction_A_vs_GN'): dict(value=1 - 46329.299182381736 / 176069.9713933181),
    ('R009', 'neighbor_fp_reduction_B_vs_GN'): dict(value=1 - 13581.713534193286 / 176069.9713933181),
    ('R013', 'E0_H_per_run'): dict(n=14, success=0, grasp=11, place=0, obstacle_stop=13),
    ('R015', 'ct_contact_fail_among_old_success_len'): {'V8': 5, 'V11A': 2, 'V11AB': 0},
    ('R022', 'T58_post_contact_runs'): dict(with_post_runs=6, with_post_contact=6, E0=3, ON=3),
    ('R022', 'T58_closest_point_outside_spheres_mm'): dict(min=12.72, max=38.8),
}


def main():
    docs = {k: json.loads(p.read_text()) for k, p in SOURCES.items() if p.suffix == '.json'}
    worklog = SOURCES['WL'].read_text()
    checks = []
    for c in C:
        rec = dict(c)
        try:
            found = pget(docs[c['source']], c['pointer'])
            rec['found'] = found
            rec['status'] = 'PASS' if compare(c['kind'], c['registered'], found) else 'FAIL'
        except (KeyError, IndexError, ValueError, TypeError) as e:
            rec['found'] = None
            rec['status'] = 'FAIL'
            rec['error'] = f'{type(e).__name__}: {e}'
        checks.append(rec)
    for d in derived(docs):
        exp = DERIVED_EXPECT[(d['result_id'], d['name'])]
        ok = all(abs(d['value'][k] - v) <= FLOAT_TOL if isinstance(v, float) else d['value'][k] == v for k, v in exp.items())
        checks.append(dict(result_id=d['result_id'], source='derived', pointer=d['formula'], registered=exp,
                           kind='derived', found=d['value'], status='PASS' if ok else 'FAIL', note='DERIVED'))
    for w in WORKLOG_ITEMS:
        present = w['text'] in worklog
        checks.append(dict(result_id=w['result_id'], source='WL', pointer=w['item'], registered=w['text'], kind='worklog_text',
                           found=present, status='source=worklog', text_present=present))

    per_result = {}
    for c in checks:
        r = per_result.setdefault(c['result_id'], dict(n=0, n_pass=0, n_fail=0, n_worklog=0))
        r['n'] += 1
        if c['status'] == 'PASS':
            r['n_pass'] += 1
        elif c['status'] == 'FAIL':
            r['n_fail'] += 1
        else:
            r['n_worklog'] += 1
    for rid, r in per_result.items():
        r['status'] = 'FAIL' if r['n_fail'] else 'PASS'
        if r['n_worklog']:
            r['status'] += f' (+{r["n_worklog"]} source=worklog, text-presence only)'
    worklog_missing = [c for c in checks if c['status'] == 'source=worklog' and not c['text_present']]
    overall = 'PASS' if not any(r['n_fail'] for r in per_result.values()) and not worklog_missing else 'FAIL'

    experiments = {
        'EXP003': dict(results=['R008', 'R009', 'R010', 'R011', 'R012'], status='REGISTERED_SOURCE_LINKED',
                       scope='IFG held-out locked grounding; 34 successful-rollout episodes, 2,523 valid frames (clear 13 / adjacent 12 / place 9); offline replay evaluation with settings locked before evaluation'),
        'EXP004': dict(results=['R013', 'R014', 'R015', 'R016', 'R017', 'R018'], status='REGISTERED_SOURCE_LINKED',
                       scope='T43 H hurdle set, 7 episodes x 2 seeds, profile t43p2_overbar; E0 recorded 2026-10-05 (T43 W code state), V8/V11A/V11AB 2026-10-07 (T43 V11 code state), C1a/C1aN/SURF 2026-10-17 (IFG-HURDLE); same episodes and seeds, different code states'),
        'EXP005': dict(results=['R019', 'R020', 'R021'], status='REGISTERED_SOURCE_LINKED; EXPLORATORY_SELECTION_CONDITIONED',
                       scope='34 seed pairs selected from collector episodes in which C1a succeeded; C1aN vs SURF closed-loop, same cuRobo/TrajOpt flags'),
        'EXP006': dict(results=['R022', 'R023', 'R024', 'R025', 'R026'], status='REGISTERED_SOURCE_LINKED (context only)',
                       scope='Context results for Limitations: T58 adjacent post, CL4 extent/policy arms, T46d gripper-scope degeneracy, CL2 C0, T53 test 16'),
    }
    result = dict(
        audit='CVPR_V1_evidence_audit', generated_by='paper/tools/analysis/audit_cvpr_v1_evidence.py',
        status=overall, float_tolerance=FLOAT_TOL, verification_level='REGISTERED_SOURCE_LINKED',
        verification_note='Numbers re-read from the source JSON at the recorded pointer and sha256-pinned. Not a 16-D manifest validation: '
                          'no array headers, no per-run re-aggregation, no rollout or model execution.',
        experiments=experiments, per_result=per_result,
        n_checks=len(checks), n_pass=sum(c['status'] == 'PASS' for c in checks),
        n_fail=sum(c['status'] == 'FAIL' for c in checks), n_worklog=sum(c['status'] == 'source=worklog' for c in checks),
        sources={k: dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for k, p in SOURCES.items()},
        checks=checks,
    )
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    for rid in sorted(per_result):
        r = per_result[rid]
        print(f'{rid}: {r["status"]}  ({r["n_pass"]}/{r["n"]} checks PASS, {r["n_fail"]} FAIL)')
    for c in checks:
        if c['status'] == 'FAIL':
            print(f'  FAIL {c["result_id"]} {c["source"]} {c["pointer"]}: registered={c["registered"]!r} found={c.get("found")!r} {c.get("error", "")}')
    for c in checks:
        if c['status'] == 'source=worklog':
            print(f'  source=worklog {c["result_id"]}: {c["pointer"]} (text present: {c["text_present"]})')
    print(f'OVERALL: {overall}  checks={len(checks)} pass={result["n_pass"]} fail={result["n_fail"]} worklog={result["n_worklog"]}')
    print(f'wrote {OUT.relative_to(ROOT)}')
    return 0 if overall == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
