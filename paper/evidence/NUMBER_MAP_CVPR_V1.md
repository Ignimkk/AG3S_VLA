# Number Map — CVPR 2027 manuscript v1

Updated: 2026-10-18 KST. 원고에 들어갈 수치(반올림 표기) → result ID → source 파일 → JSON key/path → 상태.
전체 정밀도와 범위는 [RESULTS](RESULTS.md), 상태 정의는 [EXPERIMENTS](EXPERIMENTS.md), 기계 검증은
`python3 paper/tools/analysis/audit_cvpr_v1_evidence.py` → [audit JSON](../experiments/summaries/CVPR_V1_evidence_audit.json)
(source sha256 포함). 이 표는 등록 지도이지 원고 문장 승인이 아니다. 원고 문장은 CLAIMS C012–C020의 상태를 따른다.

Source 약어: **HS** `/mnt/dev/AG3S_VLA-storage/results/development/20261016T030741+0900_f15e587290f6_19ea09c7a8b8/heldout_locked/summary.json` ·
**TW** `AG3S/ag3s/docs/handoff/T43W.verify.json` · **TV** `AG3S/ag3s/docs/handoff/T43V11AB.verify.json` ·
**HA / HT** `…/20261017T171201+0900_f15e587290f6_137b5f75d9e3/verify/IFG-HURDLE/agg.json` / `timing.json` ·
**CA / CT** `…/20261016T042508+0900_f15e587290f6_86dde89fb754/verify/IFG-CL1/agg.json` / `timing.json` ·
**T58 / CL4 / T46d / CL2 / T53** `AG3S/ag3s/docs/handoff/{T58,CL4,T46d,T44e-CL2,T53}.verify.json` ·
**WL** `doc/active/reviews/vla_dump_20261015/WORKLOG_S0S3_KO.md` · **EXP002** `paper/experiments/summaries/EXP002_IFG_audit.json`과 그 안의 R9/R7/R11b/R12 경로.

상태: `RSL` = REGISTERED_SOURCE_LINKED (audit PASS) · `RSL/DERIVED` = audit이 PASS된 JSON 값에서 script가 계산 ·
`RSL/SEL` = REGISTERED_SOURCE_LINKED + EXPLORATORY_SELECTION_CONDITIONED · `worklog` = WL에만 있음(text-presence만 확인) ·
`EXP002` = 기존 REGISTERED 결과 cross-reference(중복 등록 없음) · `EXP001` = R001–R003 cross-reference.

## Abstract

| 원고 표기 | Result | Source | JSON key / path | 상태 |
|---|---|---|---|---|
| held-out IoU 0.773 → 0.876 (+0.103 [+0.083, +0.123], 34/0 episodes) | R008, R010 | HS | `/main_success/G:N/iou/episode_mean`, `/main_success/A/iou/episode_mean`, `/main_success/paired_iou/A - G:N/{diff,ci95,better,worse}` | RSL |
| neighbour-surface false-positive mass −74 % (176,070 → 46,329) | R009 | HS | `/main_success/{G:N,A}/neighbor_fp_sum`; ratio = 1 − A/G:N = .73687 | RSL/DERIVED |
| 34 held-out episodes, 2,523 frames | R008 | HS | `/main_success/A/iou/n_episodes`, `/main_success/A/n_valid_frames` | RSL |
| hurdle: VLA only 0/14 (obstacle stop 13/14) | R013 | TW | `/numbers/summary_per_arm/E0~1H/{success,runs_obstacle_stop,n}` (key `"E0/H"`) | RSL — 2026-10-05, T43 W code |
| hurdle: AG3S V11AB 12/14 (success_ct) | R014 | TV | `/numbers/H_totals_by_rule/V11AB/{success,success_ct}` | RSL — 2026-10-07, T43 V11 code; E0와 paired 표 없음 |
| hurdle: current implementation C1a 11 / C1aN 11 / SURF 7 of 14 (success_ct) | R016 | HA | `/arms/{C1a,C1aN,SURF}/S_contact` | RSL — 2026-10-17 |
| alert AUROC 0.925 (held-out) | R012 | HS | `/gate_success/auroc_wrong` | RSL |

## Introduction (성과 요약 문단)

| 원고 표기 | Result | Source | JSON key / path | 상태 |
|---|---|---|---|---|
| every one of 34 held-out episodes improved (34/0) | R010 | HS | `/main_success/paired_iou/A - G:N/{better,worse}` | RSL |
| precision 0.894 → 0.949 | R008 | HS | `/main_success/{G:N,A}/precision/episode_mean` | RSL |
| render mask +0.044 [+0.038, +0.050], 34/0 | R010 | HS | `/main_success/paired_iou/A - A-fk/{diff,ci95,better,worse}` | RSL |
| held-state IoU 0.640 → 0.848 | R011 | HS | `/by_grasp_state/held/{A-fk,A}/iou/episode_mean` | RSL |
| carry-phase apple–hurdle contact runs 5 → 2 → 0 (V8 → V11A → V11AB) | R015 | TV | `/numbers/contact_table_by_body_phase/0/n_runs/{V8,V11A,V11AB}` | RSL |
| 85/100 wrong picks flagged, 153 false alarms | R012 | HS | `/gate_success/{alerts_on_wrong,wrong_choices,false_alarms}` | RSL |

## Table 1 — closed-loop

| 원고 표기 | Result | Source | JSON key / path | 상태 |
|---|---|---|---|---|
| **Hurdle, 14 runs (7 ep × 2 seeds, `t43p2_overbar`)** | | | | |
| E0 (VLA only): success 0, grasp 11, place 0, obstacle stop 13, HOLD 0 | R013 | TW | `/numbers/summary_per_arm/E0~1H/{success,grasp,place,runs_obstacle_stop,hold_chunks}` | RSL (2026-10-05) |
| E0 per-run cross-check 14 rows: 0 / 11 / 0 / 13 | R013 | TW | `/numbers/per_run/E0~1H~1E0_ep*` (14 keys), counts | RSL/DERIVED |
| V8: success 13, success_ct 8; V11A: 10, 8; V11AB: 12, 12 | R014 | TV | `/numbers/H_totals_by_rule/{V8,V11A,V11AB}/{success,success_ct}` | RSL (2026-10-07) |
| V11AB grasp 14, released 13 | R014 | TV | `/numbers/H_totals_by_rule/V11AB/{grasp,released}` | RSL |
| apple–hurdle contact during carry: 5 / 2 / 0 runs (10 / 9 / 0 rows) | R015 | TV | `/numbers/contact_table_by_body_phase/0/{n_runs,n_rows}/{V8,V11A,V11AB}` | RSL |
| CT failures among old successes: V8 5, V11A 2, V11AB 0 | R015 | TV | `len(/numbers/ct_contact_fail_among_old_success/<arm>)` | RSL/DERIVED |
| C1a / C1aN / SURF: success (S_old) 11 / 11 / 8; success_ct 11 / 11 / 7; grasp 13 / 13 / 9 | R016 | HA | `/arms/<arm>/{S_old,S_contact,grasp}` | RSL (2026-10-17) |
| hurdle contact before release 2 / 2 / 3 runs; obstacle stop 2 / 2 / 3 | R016 | HA | `/arms/<arm>/{pre_release_contact_runs,obstacle_stop_runs}` | RSL |
| non-authorised link–target contact 0 / 0 / 1 runs | R016 | HA | `/arms/<arm>/nonauth_target_runs` | RSL |
| HOLD chunks 86 / 82 / 11 | R016 | HA | `/arms/<arm>/hold_chunks_sum` | RSL |
| completion median 16.5 / 16.5 / 15.2 s (n 11 / 11 / 8) | R016 | HA | `/arms/<arm>/completion_s/{median,n}` | RSL |
| SURF failure modes: no_grasp 5, slip_at_lift 1 | R016 | HA | `/arms/SURF/failure_mode` | RSL |
| C1aN vs SURF (success_ct): both 6, C1aN-only 5, SURF-only 1, neither 2, p 0.22 | R017 | HA | `/pairs/C1aN_vs_SURF/success_ct/{both,C1aN_only,SURF_only,neither,mcnemar_exact_p}` | RSL — WL 23:40의 "6 / 2"와 다름, JSON 정본 |
| C1a vs C1aN identical (11 both, 0 / 0, p 1.0) | R017 | HA | `/pairs/C1a_vs_C1aN/success_ct` | RSL |
| T43 V11AB reference column 12 / 12 (different code, flags) | R016 | HA | `/T43_V11AB_reference/{S_old,S_contact}` | RSL |
| **Held-out closed-loop (IFG-CL1), 34 seed pairs, C1a-success-selected** | | | | |
| C1aN 34/34, SURF 28/34 (success = primary) | R019 | CA | `/{C1aN,SURF}/{n_success,n_primary}` | RSL/SEL |
| pairs both 28 / C1aN-only 6 / SURF-only 0 / McNemar p 0.031 | R019 | CA | `/pair_primary/{both,only_C1aN,only_SURF,mcnemar_exact_p}` | RSL/SEL |
| SURF failures: no_grasp 5, not_placed 1 (clear 4 · adjacent 1 · place 1) | R019 | CA | `/SURF/failure_modes/{no_grasp,not_placed}`, `/by_stratum/<s>/pair_primary/only_C1aN` | RSL/SEL |
| HOLD 50 vs 19 | R020 | CA | `/{C1aN,SURF}/hold_chunks_sum` | RSL/SEL |
| completion median 16.0 (n 34) vs 14.7 s (n 28) | R020 | CA | `/{C1aN,SURF}/completion_s/{median,n}` | RSL/SEL — WL "28쌍" 한정값은 JSON에 없음 |
| neighbour-fruit contact runs 2 vs 2; non-authorised / obstacle contact 0 / 0 | R020 | CA | `/{C1aN,SURF}/{n_neighbour,n_nonauth_target,n_obstacle}` | RSL/SEL |
| selection: seeds from collector episodes where C1a succeeded (34/34) | R019 | CA | `/C1a_reference/{n,n_success}` | RSL/SEL |
| interpretation sentence (English, from WL 2026-10-18 14:30) | R019 / C018 | WL | 14:30 절 인용문 (text-presence 확인) | worklog |
| **Clear scene (cross-reference)** | | | | |
| T39 E0 30/48 vs E3b 29/48; 2×2 22 / 8 / 7 / 11 (p 1.0) | R001–R003 | T39.verify.json (EXP001) | `/numbers/{E0,E3b}/n_success`, `/numbers/paired_2x2/success` | EXP001 |

## Table 2 — held-out grounding (34 episodes, 2,523 frames)

| 원고 표기 | Result | Source | JSON key / path | 상태 |
|---|---|---|---|---|
| IoU G:N 0.773 · A-fk 0.832 · A 0.876 · B 0.891 | R008 | HS | `/main_success/<arm>/iou/episode_mean` | RSL |
| precision 0.894 · 0.955 · 0.949 · 0.966 | R008 | HS | `/main_success/<arm>/precision/episode_mean` | RSL |
| recall 0.838 · 0.836 · 0.884 · 0.899 | R008 | HS | `/main_success/<arm>/recall/episode_mean` | RSL |
| strata clear 13 / adjacent 12 / place 9 episodes (966 / 892 / 665 frames) | R008 | HS | `/by_stratum/<s>/A/{iou/n_episodes,n_valid_frames}` | RSL |
| stratum IoU (A) 0.905 / 0.854 / 0.865; (G:N) 0.815 / 0.769 / 0.719; (B) 0.900 / 0.882 / 0.889 | R008 | HS | `/by_stratum/<s>/<arm>/iou/episode_mean` | RSL |
| neighbour FP mass 176,070 / 46,238 / 46,329 / 13,582 | R009 | HS | `/main_success/<arm>/neighbor_fp_sum` | RSL |
| support FP 794 / 502 / 552 / 517; robot FP 227 / 175 / 2,640 / 2,605 | R009 | HS | `/main_success/<arm>/{support_fp_sum,robot_fp_sum}` | RSL |
| FP total 177,092 / 46,916 / 49,529 / 16,711; TP 646,534 / 647,208 / 673,793 / 676,156 | R009 | HS | `/main_success/<arm>/{FP_sum,TP_sum}` | RSL |
| A − G:N +0.103 [+0.083, +0.123] 34/0 | R010 | HS | `/main_success/paired_iou/A - G:N` | RSL |
| A − A-fk +0.044 [+0.038, +0.050] 34/0 | R010 | HS | `/main_success/paired_iou/A - A-fk` | RSL |
| B − A +0.015 [−0.012, +0.039] 14/6 | R010 | HS | `/main_success/paired_iou/B - A` | RSL |
| B − G:N +0.118 [+0.083, +0.151] 32/2 | R010 | HS | `/main_success/paired_iou/B - G:N` | RSL |
| A − A-fk precision −0.006 [−0.009, −0.004] 0/34; recall +0.048 [+0.042, +0.055] 34/0 | R010 | HS | `/main_success/{paired_precision,paired_recall}/A - A-fk` | RSL |
| pre-grasp 0.970 / held 0.848 (0.640 FK mask) / placed 0.846 (A) | R011 | HS | `/by_grasp_state/{pre,held,placed}/{A,A-fk}/iou/episode_mean` | RSL |
| grasp-state frames 706 / 326 / 1,491 | R011 | HS | `/by_grasp_state/<state>/A/n_valid_frames` | RSL |
| B switch frames 125 | R010 | HS | `/main_success/B_switch_frames` | RSL |
| dev matched-mask comparison 0.837 → 0.893 (footnote) | R004 | EXP002 (R9 `paired_summary.json`) | `/arms/T54region+rm:N/iou`, `/arms/top1-only-ns+rm:N/iou` | EXP002 |

## Table 3 — analysis

| 원고 표기 | Result | Source | JSON key / path | 상태 |
|---|---|---|---|---|
| held-out alert: threshold 0.590, 2,420 frames, 100 wrong, 85 caught, 153 false alarms | R012 | HS | `/gate_success/{threshold,n_frames,wrong_choices,alerts_on_wrong,false_alarms}` | RSL |
| 88 % wrong-surface FP removed at 6.7 % TP loss; AUROC 0.925 | R012 | HS | `/gate_success/{fp_removed_ratio,tp_lost_ratio,auroc_wrong}` | RSL |
| (all frames incl. failures) 3,068 frames, AUROC 0.924 | R012 | HS | `/gate_all/{n_frames,auroc_wrong}` | RSL |
| identity switch B − A +0.015 [−0.012, +0.039]; by stratum −0.005 / +0.028 / +0.025 | R010 | HS | `/main_success/paired_iou/B - A`, `/by_stratum/<s>/paired_iou/B - A/diff` | RSL |
| feature fusion (dev): G-Q+top1 0.893, FGe 0.893, FGu 0.827, FG 0.826, FG-R 0.755 | R005 | EXP002 (R7 `paired_summary.json`) | `/arms/{G-Q+top1,FGe-Q:prefix_L8+top1,FGu-Q:prefix_L8+top1,FG-Q:prefix_L8+top1,FG-R:prefix_L8+top1}/iou` | EXP002 |
| dev temporal consistency AUROC 0.998 vs 0.900 (147 matched frames, 6 wrong) | R006 | EXP002 audit | `/confidence_matched/{q_mem_chain:prefix_L8,c_mass}/auroc` | EXP002 |
| phase LOEO accuracy 0.878 → 0.939 | R007 | EXP002 (R12 `summary.json`) | `/results/phase/{LR[hsr+grip],LR[hsr+grip+geo]}` | EXP002 |

## Table 4 — runtime (ms, first request per run excluded, budget 533.3 ms)

| 원고 표기 | Result | Source | JSON key / path | 상태 |
|---|---|---|---|---|
| client round trip median / p95: C1aN 1,137 / 1,924; SURF 2,858 / 3,623 (2,516 requests each) | R021 | CT | `/summary/{C1aN,SURF}/stages/client_round_trip/{median,p95}`, `/summary/<arm>/n_requests` | RSL |
| server total median 796 vs 2,587 | R021 | CT | `/summary/{C1aN,SURF}/stages/server_total/median` | RSL |
| VLA inference + attention 236 vs 221; AG3S 289 vs 2,142; TrajOpt 188 vs 153 | R021 | CT | `/summary/<arm>/stages/{server_infer_vla_plus_attention,server_ag3s,server_trajopt}/median` | RSL |
| attention_surface 1,663 = graph build 828 + region solve 285 + label codes 383 (+ remainder) | R021 | CT | `/summary/SURF/stages/{profile_attention_surface,surface_build,surface_solve,surface_codes}/median` | RSL — medians, 합이 정확히 일치하지 않음 |
| render mask 159 | R021 | CT | `/summary/SURF/stages/profile_robot_render_mask/median` | RSL |
| over budget: 100 % of requests (both arms) | R021 | CT | `/summary/<arm>/stages/client_round_trip/frac_over_budget` | RSL |
| hurdle client round trip median / p95: C1a 1,256 / 2,034; C1aN 1,177 / 2,152; SURF 3,331 / 4,979 | R018 | HT | `/summary/<arm>/stages/client_round_trip/{median,p95}` | RSL |
| hurdle server total median 910 / 855 / 2,998 | R018 | HT | `/summary/<arm>/stages/server_total/median` | RSL |
| VLA-only round trip median 199 ms (E0, 398 chunks; different code state) | R013 | TW | `/numbers/summary_per_arm/E0~1H/round_trip_ms_pooled/{median,n}` | RSL |
| budget 533.3 ms = 8 steps @ 15 Hz | R018, R021 | HT, CT | `/budget_ms` | RSL |

## Limitations

| 원고 표기 | Result | Source | JSON key / path | 상태 |
|---|---|---|---|---|
| adjacent post: robot–post contact 6/6 runs (E0 3/3, ON 3/3); without post both 3/3 success | R022 | T58 | `/numbers/runs/obs~1<arm>_ep<ep>_s<ep>1/{success,obstacle_rows_by_class}`, `noobs~1…/success`; `/verdict` | RSL/DERIVED |
| hand point 12.7–38.8 mm outside the constraint spheres | R022 | T58 | min/max `/numbers/on_diagnosis/*/cover_rows/*/closest_point_outside_spheres_mm` (12.72 / 38.8) | RSL/DERIVED |
| CL4 success_ct of 18: ball 10, ball_owned 5, ball_owned+F1 7, oracle-visible 9 | R023 | CL4 | `/numbers/arms/{X1,X2,X3,X4}/success_ct` | RSL |
| link-specific vs global: 9 vs 7, p 0.69 (both 5 / 4 / 2 / 7) | R023 | CL4 | `/numbers/pairs_agg/X4_vs_G4/success_ct/{A_total,B_total,mcnemar_exact_p,both,A_only,B_only,neither}` | RSL |
| FT–target contact 0 in all arms | R023 | CL4 | `/numbers/FT_target_contact_runs/*` | RSL |
| gripper scope: 0 of 349,920 non-authorised rows within 50 mm of the target (degenerate) | R024 | T46d | `/numbers/item1_degeneracy_gripper_scope/{C1aF1,C1a,C2p}/pre8close/{nonauth_rows_near_tgt_plan16,nonauth_rows_evaluated}` | RSL |
| clear scenes: CL2 C0 14/18 (success_ct 12/18) | R025 | CL2 | `/numbers/arms/C0/t1_success/{task_success,success_ct,n_runs}` | RSL |
| current-serving test 16: 9/16 (success_ct 7/16), HOLD 37 | R026 | T53 | `/numbers/{task_success,success_ct,hold_chunks_total}` | RSL |
| T39 clear scene 30/48 vs 29/48 | R001–R003 | T39.verify.json (EXP001) | `/numbers/{E0,E3b}/n_success` | EXP001 |
| selection-conditioned held-out closed-loop (sentence) | R019 / C018 | WL + CA | 위 Table 1 참조 | RSL/SEL + worklog |
| latency exceeds budget in every arm | R018, R021 | HT, CT | `frac_over_budget` 1.0 | RSL |

## 기록되지 않은 것 (JSON에 없어 등록하지 않음)

- E0(2026-10-05) 대 V11AB(2026-10-07) 또는 대 SURF(2026-10-17)의 paired 2×2 — TV `H_pairs_z4_ct_rule`는 V11AB vs V8 / V11A뿐, HA `pairs`는 2026-10-17 세 arm 사이뿐.
- IFG-CL1 "둘 다 성공 28쌍" 한정 completion median — CA에는 arm별 성공 run 전체(n 34 / 28) median만 있다.
- 선별 없는 paired 재평가(IFG-CL3) 결과 — 실행 중이며 미등록.
- attention extraction 단독·VLA inference 단독 시간 — timing.json `not_measured`.
