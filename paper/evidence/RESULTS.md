# Results and Reported Values

## 2026-10-18 추가 등록 결과 — EXP003–EXP006 (REGISTERED_SOURCE_LINKED)

등록 수준은 **REGISTERED_SOURCE_LINKED**다. 아래 모든 수치는 `python3 paper/tools/analysis/audit_cvpr_v1_evidence.py`가 source JSON의 기록된 pointer에서 다시 읽어 등록값과 대조하고(float |diff| ≤ 5e-4, count·string은 exact), source 파일의 sha256을 [CVPR_V1 audit](../experiments/summaries/CVPR_V1_evidence_audit.json)에 고정한 것이다. **EXP001·EXP002와 달리 16-D manifest validation(array header·per-run 재집계·rollout 재실행)은 하지 않았다.** 원고에서는 반올림하되 registry에는 source 정밀도 그대로 적는다. WORKLOG에만 있는 항목은 `source=worklog`로 표시하며 기계 검증이 아니다.

Source 약어 (전체 경로·hash는 audit JSON `sources`): **HS** = `…/20261016T030741+0900_f15e587290f6_19ea09c7a8b8/heldout_locked/summary.json` (sha256 `b5fd560a7ec7c184…`), **HF** = `AG3S/ag3s/docs/figures/ifg/ifg-heldout-locked.json` (`453496f86ed72273…`), **TW** = `AG3S/ag3s/docs/handoff/T43W.verify.json` (`59e158d5c5ea11b3…`), **TV** = `T43V11AB.verify.json` (`693f7de281a00c9f…`), **HA/HT** = `…/20261017T171201+0900_f15e587290f6_137b5f75d9e3/verify/IFG-HURDLE/{agg,timing}.json` (`dcfbea1f03eb0fcd…` / `5b6441b77bae660a…`), **CA/CT** = `…/20261016T042508+0900_f15e587290f6_86dde89fb754/verify/IFG-CL1/{agg,timing}.json` (`dfda86f57fc4c745…` / `c7f9e7819d071025…`), **T58/CL4/T46d/CL2/T53** = `AG3S/ag3s/docs/handoff/{T58,CL4,T46d,T44e-CL2,T53}.verify.json`, **WL** = `doc/active/reviews/vla_dump_20261015/WORKLOG_S0S3_KO.md` (`a79d14c6c5e3c2e3…`). Storage 경로의 `…`는 `/mnt/dev/AG3S_VLA-storage/results/development`다.

Held-out arm 정의 (HS `/rule`): **G:N** = R0 reference arm (기하 radius + clustering extent, FK-sphere robot mask); **A-fk** = A와 같되 serving FK-sphere robot mask; **A** = `top1-only-ns+rm:N` (noun-attention seed + depth-continuous surface graph + rendered robot mask, memory switch 없음); **B** = A + centered-feature identity memory switch (`Mem-first:prefix_L8:d0.05`). Main = 성공 rollout episode; 실패 9 episode는 `/failure_appendix`로 등록 scope 밖이다.

### EXP003 — IFG held-out locked grounding

| ID | 결과와 source pointer | 범위 |
|---|---|---|
| R008 | HS `/main_success/<arm>/{iou,precision,recall}/episode_mean`: G:N .7730478249129286 / .8941035687057404 / .8381835766308009; A-fk .8320612510959661 / .9552711342181999 / .8355078746336647; **A .876093971555441 / .9490830942021894 / .8838479458205702**; B .8908732856842837 / .9656474792528674 / .8986232884658625. 모든 arm `iou/n_episodes` 34, `n_valid_frames` 2523. Stratum `/by_stratum/{clear,adjacent,place}/A/iou/n_episodes` 13 / 12 / 9 (`n_valid_frames` 966 / 892 / 665); stratum IoU A .9045400208022639 / .8538756627971203 / .8646296454322352, G:N .8146361930179686 / .7685631795479471 / .7189552648034018, B .9000382395052716 / .8822170362604757 / .8891766849523789. `/n_frames_total` 3196, `/skipped/s0_invalid` 29, `/skipped/no_prep` 0. HF `/source`가 HS를 가리키고 A·G:N·B IoU가 일치 | 성공 rollout 34 held-out episode, 2,523 valid frame; frame → episode mean → arm mean; replay depth; 평가 전 잠근 설정(`LOCK_heldout.json`). 실패 9 episode / 659 frame은 appendix (`/failure_appendix`), 등록 scope 밖 |
| R009 | HS `/main_success/<arm>/{neighbor_fp_sum,support_fp_sum,robot_fp_sum,FP_sum,TP_sum}`: G:N 176069.9713933181 / 794.1824287812866 / 226.91838952046052 / 177091.97938555796 / 646534.020614443; A-fk 46237.61075299564 / 502.3729128190385 / 175.16828806817324 / 46915.702689606966 / 647208.2973103941; A 46329.299182381736 / 552.2908453349412 / 2640.379443738322 / 49529.18917135399 / 673792.8108286464; B 13581.713534193286 / 517.4040243042206 / 2604.9563646802735 / 16711.370546153663 / 676155.6294538468. DERIVED (script 계산): neighbor FP 감소 A vs G:N = 1 − 46329.299…/176069.971… = .7368699567804897; B vs G:N = .922861840513091 | 2,523 frame pooled voxel-mass 합. 표면 FP는 proxy이며 실행 contact가 아니다. A의 `robot_fp_sum`(2640.38)이 A-fk(175.17)보다 큰 것은 rendered mask가 FK 구 mask보다 robot 표면을 덜 지운 것으로 기록된 값 그대로다 |
| R010 | HS `/main_success/paired_iou/<pair>/{diff,ci95,better,worse}` (n_episodes 34): **A − G:N +.10304614664251237 [.08323494981427215, .12313285145943233] 34/0**; **A − A-fk +.04403272045947481 [.03801383050689502, .050055427622609866] 34/0**; **B − A +.014779314128842732 [−.0116789100514432, .03944222857471973] 14/6**; B − G:N +.11782546077135511 [.08271813740590676, .15130667079948473] 32/2. `paired_precision/A - A-fk` −.006188040016010591 [−.009116464059697633, −.004231042636771202] 0/34; `paired_recall/A - A-fk` +.04834007118690548 [.041713586127408664, .05489205054611608] 34/0. `B_switch_frames` 125. Stratum `B - A` diff: clear −.004501781296992498 (5/4), adjacent +.0283413734633552 (6/1), place +.02454703952014366 (3/1) | Episode-paired bootstrap 2,000회(seed 20261015), HS `/rule`. better/worse는 episode 수(합이 34 미만이면 나머지는 동률). B − A의 CI는 0을 포함한다 |
| R011 | HS `/by_grasp_state/<state>/<arm>/iou/episode_mean` (n_episodes 34): pre (706 frames) G:N .8749510904201444, A-fk .9075024099794847, **A .9698507944447411**, B .9673310030050492; held (326 frames) G:N .5377397793127421, **A-fk .64040778567183, A .8475770155578735**, B .8411570863823252; placed (1,491 frames) G:N .7767709801403806, A-fk .8449822910803859, **A .845807102001963**, B .8728342906503719 | Grasp state는 평가 label(pre / held / placed)이며 HSR 출력이 아니다. frame 수는 state별 `A/n_valid_frames` |
| R012 | HS `/gate_success`: `threshold` .5904207590672415, `n_frames` 2420, `wrong_choices` 100, `alerts` 238, `alerts_on_wrong` 85, `wrong_caught_ratio` .85, `false_alarms` 153, `fp_removed_ratio` .8788327204996255, `tp_lost_ratio` .06698843965768998, **`auroc_wrong` .9251551724137931**. 참고 `/gate_all` (실패 episode 포함 3,068 frames): wrong 178, auroc .9242758835193033. HF `/gate/auroc_wrong` 일치 | Alert = `q_mem_chain:prefix_L8` < dev에서 잠근 threshold; wrong choice = `r6_iou` < 0.1. gate_success는 성공 rollout의 history-dependent cohort 2,420 frames(2,523보다 적음: chain feature가 없는 frame 제외) |

### EXP004 — hurdle closed-loop (T43 H set, 7 episodes × 2 seeds, `t43p2_overbar`)

| ID | 결과와 source pointer | 범위 |
|---|---|---|
| R013 | TW `/numbers/summary_per_arm/E0~1H` (key 문자열 `"E0/H"`): `n` 14, **`success` 0, `grasp` 11, `place` 0, `runs_obstacle_stop` 13**, `runs_client_robot_collision` 13, `runs_client_payload_collision` 0, `hold_chunks` 0, `runs_with_physics_replay_contact` 13, `round_trip_ms_pooled/median` 199.33629035949707 (n 398). DERIVED: `/numbers/per_run` 중 `E0/H/*` 14개에서 success 0 / grasp 11 / place 0 / obstacle_stop 13 재집계 일치. `/date` "2026-10-05 (E0 …)", `/settings/E0` "server --no-safe; client no safe flags", snapshot_1 benchmark git archive `089516ef9c23be2a1d390d0eb8d6f8990b75d7c6` | **E0 = VLA 단독, 2026-10-05, T43 W code state**(별도 snapshot `/mnt/dev/work-t43w`). 14 run = 7 episode × 2 seed. obstacle stop은 client가 첫 robot–obstacle 접촉에서 episode를 끝낸 run 수. 현재 code state의 결과가 아니다 |
| R014 | TV `/numbers/H_totals_by_rule/<arm>/{success,grasp,place,released,success_rel,success_ct}` (n 14): V8 13 / 14 / 13 / 14 / 13 / **8**; V11A 10 / 14 / 10 / 14 / 10 / **8**; **V11AB 12 / 14 / 12 / 13 / 12 / 12**. `/numbers/H_pairs_z4_ct_rule/V11AB_vs_V8/success_ct`: A_only 5, B_only 1, p .21875. `/date` 2026-10-07, `/code_state/head` 3ca1b21 | **2026-10-07, T43 V11 code state**(`/mnt/dev/work-t43v11`). success_ct = success ∧ Z4 release ∧ pre-release hurdle contact 없음. E0와 같은 14 (episode, seed)이지만 code state가 달라 **E0 대 V11AB의 paired 표는 등록 source에 없다**(TV pairs는 V11AB vs V8 / V11A뿐) |
| R015 | TV `/numbers/contact_table_by_body_phase/0` (`body` apple, `phase` carry): `n_runs` **V8 5 / V11A 2 / V11AB 0**, `n_rows` 10 / 9 / 0, `max_pen_mm/V11AB` 0.0. DERIVED: `len(/numbers/ct_contact_fail_among_old_success/<arm>)` V8 5 / V11A 2 / V11AB 0 | 운반 중 사과(attached object)–hurdle 접촉 run 수 (MuJoCo replay contact rows, phase = carry). V11AB의 다른 contact row(`link_right_arm_5` carry 1 run)는 같은 표 index 1에 있다 |
| R016 | HA `/arms/<arm>/{S_old,S_contact,grasp,pre_release_contact_runs,post_release_contact_runs,nonauth_target_runs,obstacle_stop_runs,hold_chunks_sum,completion_s/median (n)}` (n 14): **C1a 11 / 11 / 13 / 2 / 0 / 0 / 2 / 86 / 16.533333333333335 (11)**; **C1aN 11 / 11 / 13 / 2 / 0 / 0 / 2 / 82 / 16.533333333333335 (11)**; **SURF 8 / 7 / 9 / 3 / 1 / 1 / 3 / 11 / 15.2 (8)**. `failure_mode`: C1a·C1aN {ok 11, slip_in_transport 1, no_grasp 1, not_placed 1}; SURF {ok 8, no_grasp 5, slip_at_lift 1}. SURF `surface_applied_sum` 510 / `surface_layer_chunks_sum` 580. `/T43_V11AB_reference` S_old 12, S_contact 12 (n 14). `/n_items` 14, `/n_runs_scored` 42 | **2026-10-17, 현재 구현(commit f15e5872 계열 snapshot), 같은 14 (episode, seed)**. C1a = 현재 serving(L8H2·radius+clustering·ball-owned), C1aN = C1a + noun-attention seed, SURF = surface extent + rendered robot mask(두 요소를 함께 바꾼 구성). S_contact = CT 규칙(T43 복사). 참고 열 V11AB는 code·flag가 다르다 |
| R017 | HA `/pairs/C1aN_vs_SURF/success_ct`: n_pairs 14, **both 6, C1aN_only 5, SURF_only 1, neither 2, p .21875**; `/success`(S_old): both 7, C1aN_only 4, SURF_only 1, neither 2, p .375. `/pairs/C1a_vs_C1aN/success_ct`: both 11, 0 / 0, neither 3, p 1.0 | McNemar exact two-sided(HA `/definitions`). **WL 2026-10-17 23:40 절의 "C1aN만 성공 6, SURF만 성공 2"는 agg.json의 5 / 1과 다르다 — JSON이 정본**(`source=worklog` 항목으로 audit에 기록). 표본 14로 어느 방향으로도 판정하지 않는다 |
| R018 | HT `/summary/<arm>/stages/client_round_trip/{median,p95}` ms: C1a 1256.0148239135742 / 2034.055233001709 (n_requests 976); C1aN 1177.14262008667 / 2151.6876816749573 (976); SURF 3330.8560848236084 / 4978.579664230347 (973). `server_total/median` 910.2747109718621 / 855.4650004953146 / 2998.2956275343895. `/budget_ms` 533.3333333333334 | 각 run 첫 request 제외(JIT). client round trip = request→response. budget = 8 step @ 15 Hz. 세 arm 모두 `frac_over_budget` 1.0 |

### EXP005 — held-out closed-loop C1aN vs SURF (IFG-CL1) — EXPLORATORY_SELECTION_CONDITIONED

| ID | 결과와 source pointer | 범위 |
|---|---|---|
| R019 | CA `/n_seeds` 34; `/C1a_reference/{n,n_success}` 34 / 34; **`/C1aN/{n_runs,n_success,n_primary}` 34 / 34 / 34; `/SURF/{n_runs,n_success,n_primary}` 34 / 28 / 28**; `/pair_primary` = `/pair_success`: **both 28, only_C1aN 6, only_SURF 0, neither 0, missing 0, `mcnemar_exact_p` .03125**. `/SURF/failure_modes` {no_grasp 5, not_placed 1, ok 25, regrasp 1, slip_at_lift 2}; `/C1aN/failure_modes/ok` 27 (나머지 7은 성공 run의 grasp-quality 표기: regrasp 1, slip_at_lift 3, slip_in_transport 3). Stratum `pair_primary` both/only_C1aN: clear 9/4, adjacent 11/1, place 8/1. **등록 해석 문장(WL 2026-10-18 14:30, `source=worklog`; 원문 "C1a가 성공한 34개 조건에서 C1aN은 34개, SURF는 28개를 성공했다. 이는 기준선 성공 사례의 보존 여부를 보여주지만, 성공 여부로 선별하지 않은 전체 장면에서의 성능 차이를 추정하지는 못한다."): "Under the 34 conditions in which C1a succeeded, C1aN succeeded in 34 and SURF in 28. This shows whether the baseline's successful cases are retained, but it does not estimate the performance difference over the full set of scenes not selected by success."** | **34 seed 쌍은 collector에서 C1a가 성공한 episode에서 골랐다**(`/C1a_reference/n_success` 34/34). PRIMARY = success ∧ nonauth_target 없음 ∧ obstacle 접촉 없음(CA `/definitions`). 2026-10-17 03:30 실행. 선별 없는 재평가(IFG-CL3, 40 × 2)는 이 registry에 없다 |
| R020 | CA `/C1aN/hold_chunks_sum` **50** (`runs_with_hold` 23), `/SURF/hold_chunks_sum` **19** (8); `/C1aN/completion_s/{median,n}` **16.0 / 34**, `/SURF/completion_s/{median,n}` **14.666666666666668 / 28**; `/{C1aN,SURF}/n_neighbour` **2 / 2**; `n_nonauth_target` 0 / 0; `n_obstacle` 0 / 0. `/SURF_diagnostics`: n_chunks 2550, n_applied 908, not-applied why {identity_unresolved 43, attention_surface_abstained (no_region) 8, no_target_free_layer 1591} | completion = place_t / 15 s, 성공 run만. **WL은 "둘 다 성공 28쌍"이라 적었으나 agg.json의 C1aN median은 n 34 기준**이다(`source=worklog` 기록); 28쌍 한정값은 JSON에 없다. neighbour 접촉은 PRIMARY에 포함되지 않는 보고 항목 |
| R021 | CT `/summary/<arm>/stages/…/median` ms (n_requests 2516 / 2516): `client_round_trip` **C1aN 1137.4754905700684 (p95 1923.847734928131) vs SURF 2857.689142227173 (p95 3623.0645775794983)**; `server_total` **796.4624175801873 vs 2586.567224934697**; SURF `profile_attention_surface` **1663.4485451504588** = `surface_build` **827.567** + `surface_solve` **285.10400000000004** + `surface_codes` **383.465** (n 2487) (+ 잔여); SURF `profile_robot_render_mask` **158.57859468087554** (`render_ms_record` 157.227). `server_infer_vla_plus_attention` 235.64596055075526 / 220.51269887015224; `server_ag3s` 288.6834470555186 / 2142.0471272431314; `server_trajopt` 187.97238916158676 / 153.17066758871078. `/budget_ms` 533.3333333333334; 두 arm `client_round_trip/frac_over_budget` 1.0 | 첫 request 제외. attention extraction 단독·VLA inference 단독은 분리 측정되지 않음(`not_measured`). sub-step 합(1496.1)과 `profile_attention_surface` median(1663.4)은 median의 합이 아니므로 같지 않다 |

### EXP006 — Limitations용 context 결과

| ID | 결과와 source pointer | 범위 |
|---|---|---|
| R022 | T58 `/numbers/runs/obs~1<arm>_ep<ep>_s<ep>1`: 6 run 모두 `success` false, `obstacle_rows_by_class/hand-obstacle` 1, `qpos_mismatch_rows` 0 → DERIVED **기둥 있을 때 robot–post 접촉 6/6 (E0 3/3, ON 3/3)**; `noobs~1…/success` 6 run 모두 true (**E0 3/3, ON 3/3**). DERIVED `/numbers/on_diagnosis/*/cover_rows/*/closest_point_outside_spheres_mm` min **12.72** / max **38.8** (19 rows). `/verdict` 문자열에 "6/6 runs (E0 3/3, ON 3/3)"·"12.7-38.8 mm outside the hand constraint spheres" 포함. `/date` 2026-10-13 | episode 1925 (apple L) · 1830 (orange L) · 1623 (orange R, validation split), seed ep·10+1, post_0 r 0.02 m; ON = T53 C1a flag + F1 + W1. 접촉 = MuJoCo replay contact rows. 손 점은 hand mesh의 post 최근접점이 hand constraint sphere 밖에 있는 거리 |
| R023 | CL4 `/numbers/arms/<arm>/{success,success_ct}` (18 pairs): **X1 10 / 10, X2 7 / 5, X3 9 / 7, X4 9 / 9, G4 7 / 7**; `/numbers/pairs_agg/X4_vs_G4/success_ct`: n_pairs 18, both 5, A_only 4, B_only 2, neither 7, A_total 9, B_total 7, **p .6875**; `/numbers/FT_target_contact_runs/*` 및 `arms/*/FT_target_contact_runs_any_window` 모두 **0**; `/numbers/before/CL2_C0` success 14 / success_ct 12; `/execution/{runs,done}` 90 / 90; `/date` 2026-10-15 | X1 ball, X2 ball_owned, X3 ball_owned + F1, X4 oracle-visible(정책 ③ LTE 고정), G4 = X4 extent + 정책 ② global exclusion. 최종 형상(팔·손목·손바닥 sphere). CL4 `not_measured`: 성공률 하락 원인은 geometry / margin / W1 / snapshot이 교란됨 |
| R024 | T46d `/numbers/item1_degeneracy_gripper_scope/<arm>/pre8close`: C1aF1·C1a·C2p 모두 `nonauth_rows_evaluated` **349920**, `nonauth_rows_near_tgt_plan16` **0**, `nonauth_rows_near_tgt_exec` 0, `chunks` 162; `/verdict_facts/gripper_scope_degenerate_nonauth_rows_near_target` 0; `/date` 2026-10-12 | gripper scope(270 spheres)의 pre8close window에서 비권한 query row 중 target 50 mm 이내가 0 → LTE와 global exclusion이 이 scope에서 구분되지 않음(퇴화). 다른 window·arms scope 수치는 등록하지 않음 |
| R025 | CL2 `/numbers/arms/C0/t1_success/{n_runs,task_success,success_ct}` **18 / 14 / 12**; `t4_hold/hold_chunks_total` 101; `/date` 2026-10-12 | C0 = V11AB flags + `--to-fast-eval --esdf-host-mirror`, 9 scene × 2 seed, clear 인접 과일 장면 |
| R026 | T53 `/numbers/{n_episodes,runs_complete,task_success,success_ct,hold_chunks_total,hold_runs,replay_qpos_mismatch_rows_total}` **16 / 16 / 9 / 7 / 37 / 10 / 0**; `/date` 2026-10-13 | C1a 현재 serving, test 16 episode, seed ep·10+1 사전 등록 |

Cross-reference (중복 등록 없음): clear-scene paired T39 = R001–R003 (30/48 vs 29/48, 2×2 22/8/7/11); dev-set component 수치 = EXP002 R004–R007 (`/arms/T54region+rm:N/iou` .836552 → `/arms/top1-only-ns+rm:N/iou` .893162; feature fusion R005; 147-frame AUROC .997636 vs .899527 R006; phase LOEO .877778 → .938889 R007).

원고 위치별 번호 지도는 [NUMBER_MAP_CVPR_V1](NUMBER_MAP_CVPR_V1.md)을 따른다. 이 절의 어떤 수치도 물리적 접촉 보장, 동등성(non-inferiority 포함) 또는 현재 code state의 일반 성능을 확정하지 않는다.

## 2026-10-15 추가 등록 결과 — EXP002

현재 원고의 수치는 아래 EXP002만 사용한다. 기존 EXP001 기록은 아래에 이력으로 보존한다.

| ID | 결과와 source pointer | 범위 |
|---|---|---|
| R004 | R9 `paired_summary.json /arms/T54region+rm:N/iou`=.836552; `/arms/top1-only-ns+rm:N/iou`=.893162 | episode mean, 32 dev episodes; per-frame rows 재집계 |
| R005 | R7 `paired_summary.json /arms`: G-Q+top1 .893162, FGe L8 .893334, FGu L8 .827212, FG L8 .825845, FG-R L8 .754712 | 동일 geometric parent와 render mask |
| R006 | EXP002 audit `/confidence_matched`: chain .997636, c_mass .899527 | 같은 147 frame, wrong 6개; R11b per-frame 재집계 |
| R007 | R12 `summary.json /results/phase`: LR[hsr+grip] .877778, LR[hsr+grip+geo] .938889 | pooled accuracy; LOEO, dev upstream; paired +.067은 episode mean |

모든 source path/hash와 원래 summary는 [EXP002 audit](../experiments/summaries/EXP002_IFG_audit.json)에 있다. Precision 결측 수, confidence 분모, memory-chain 의존성은 [개정 기록](../notes/IFG_REVISION_20261015_KO.md)에 명시했다.

Updated: 2026-10-05 19:45 KST

## 등록 범위

현재 결과는 **EXP001의 descriptive task outcomes만**이다. 아래 값은 새 실험이나
simulation 재실행 결과가 아니라 기존 verifier source의 수치를 연결한 것이다.
원자료 hash: `74a172b64af191f71b94557952678bb9bafc2604a43fcded14d9712832b8a78d`.

| Result | Metric·조건 | 값 | 정확한 source key |
|---|---|---|---|
| R001 | M001 / E0: T39 VLA 단독 | 30 / 48 | `/numbers/E0/n_success`, `/numbers/E0/n_runs` |
| R002 | M001 / E3b: T39 통합 snapshot | 29 / 48 | `/numbers/E3b/n_success`, `/numbers/E3b/n_runs` |
| R003 | M001_PAIRED: 둘 다 성공 / E0만 / E3b만 / 둘 다 실패 | 22 / 8 / 7 / 11 | `/numbers/paired_2x2/success` |

Source: [T39.verify.json](../../benchmark/ag3s/docs/handoff/T39.verify.json),
[T39 research log](../../benchmark/ag3s/docs/AG3S_T0T6_LOG.md),
[등록 manifest](../experiments/manifests/EXP001_T39_task_outcomes.json).
R003의 원 source object에는 exact McNemar p = 1.0도 기록되어 있다.
이 값으로 equivalence, non-inferiority 또는 "조작 능력 유지"를 확정하지 않는다.

## 해석 범위

- Model은 기록된 16-D checkpoint, condition은 해당 시점의 E0/E3b snapshot이다.
- M001은 기존 planning-row grasp/place 규칙이다. 물리적 collision 판정과 다르다.
- Test split 표시는 metadata에서 확인했으나 training/tuning과의 독립성은 아직 검토하지 않았다.
- 다른 episode의 seed를 별도 task로 간주하거나 시간별 frame을 독립 trial로 세지 않는다.
- 현 main은 이후 변경됐다. 이 결과를 최신 default gate/config의 성능으로 쓰지 않는다.
- Collision rate, safe success, full-body/attached-object clearance, grounding accuracy,
  real-robot 성과는 이 registry에 없다. 미측정은 `NOT_MEASURED` 또는 `null`이며 0이 아니다.
- Raw replay·ESDF proxy·HOLD count를 실제 실행 contact의 대용물로 섞지 않는다.

## 결과 추가 시 필수 항목

Result ID, experiment/condition/metric ID, source hash와 JSON pointer, 값·unit·denominator,
sample unit·aggregation·uncertainty, raw source, exclusion/retry 규칙, limitations,
claim/artifact 연결을 기록한다. 다른 scope의 결과는 새 항목으로 등록한다.

[METRICS](METRICS.md), [EXPERIMENTS](EXPERIMENTS.md), [CLAIMS](CLAIMS.md)를 함께 갱신한다.
