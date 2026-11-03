# SL7 — 구현: development 최종 구성 P0m 을 Panda 서버에 이식

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 2026-11-03 KST · 기준 HEAD fe3abf21 (commit 안 함)
> task: `AG3S/ag3s/docs/handoff/SL7.task.md` + lead 추가 요청 (세 번째 판 `-p0m-nbc`). GPU · rollout 없음, CPU 테스트만.

## 무엇을 했나 (평이한 요약 먼저)
Panda 서버에 판 3개를 등록했다: `AG3S-final-panda-p0m` (P0 표면 추출 + hull 채우기 + 접촉 여유 10 mm + `--subtask-from-latch`),
`AG3S-final-panda-p0m-nohull` (hull 채우기 · 접촉 여유만 뺌), `AG3S-final-panda-p0m-nbc` (p0m + `--neighbour-contact-mm 20
--neighbour-palm-contact-mm 10`, D1 비용을 재는 대조). 세 판 모두 HSR6 규칙 · HSR6 probe · `--nontarget-evidence` · TrajOpt 여러 초기값 ·
G1 · A4 가 없다. 그 flag 들이 실제로 서버에서 적용되도록 연결했고 (표면 추출은 AG3S config 로, 단계 label 은 SafePolicy 인자로),
Panda 에서만 달라지는 곳 두 개를 고쳤다: (1) 표면 추출의 seed 지도는 RB-Y1 head `L10H7` 만 받던 것을 **Panda 서버에서만** `L8H6` 도 받는다.
(2) 표면 선택은 head · left_wrist · right_wrist 세 view 를 요구하는데 Panda 는 손목이 하나라 그대로면 매 요청 `missing_camera:right_wrist`
로 abstain 한다 — 없는 right_wrist 를 더미 view (깊이 1 픽셀, 로봇 mask 로 덮음, attention 0) 로 채운다 (기본 꺼짐, Panda 의 표면 추출 판에서만 켬).
기존 14 판 (V11AB · pb1~pb7 · 1판~pc판) 의 argv · 파싱 결과 · AG3S config · SafePolicy 인자 · `info` · recorder meta · `version_flag_values` 는
HEAD 와 byte 단위로 같다 (아래 "회귀"). RB-Y1 경로는 `_sl5a_golden` 포함 통과.

## 바뀐 파일
| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/adapters/panda/profile.py:348-375` | `P0M_REMOVED_FLAGS` · `P0M_SURFACE_FLAGS` · `P0M_HULL_FLAGS` · `P0M_SUBTASK_FLAGS` · `P0M_FLAGS` / `P0M_NOHULL_FLAGS` / `P0M_NBC_FLAGS` · `P0M_VERSIONS`, `AG3S_VERSIONS` 3 항목 (판 정의 위에 무엇을 뺐고 더했는지 주석) | task 0.1/0.3/0.4 + lead 추가 판. `FINAL_CONFIG_FLAGS` 에서 뺄 것만 `_drop_flags` 로 빼므로 1판이 바뀌면 따라간다 |
| `profile.py:386-405` | `is_final_version` 이 P0m 포함 (base argv 에 `V11AB_PRE_MERGE_DEFAULTS` 없음), `version_server_flags` 는 P0m 에 `--hsr6-probe` 를 붙이지 않고, probe 를 주면 ValueError | HSR6 끔 (0.1). `FINAL_CONFIG_VERSIONS` 는 그대로 |
| `profile.py:442-` | `version_flag_values`: P0m 전용 dest (surface_* · target_extent · subtask_from_latch …) 는 그중 하나가 기본에서 벗어날 때만 포함 | 이전 판 기록의 key 가 변하지 않게 (`match_ag3s_version` 읽기 포함) |
| `profile.py:378-381` | `PANDA_HSR6_PROBE = shared_storage_root()/PANDA_HSR6_PROBE_REL` (SHA-256 상수 그대로) | task 0.7. 기본 root 에서 기록 문자열 동일 (테스트) |
| `profile.py:498-503` | `parse_v11ab_args` 가 `reject_bad_flag_combinations(..., seed_map_labels=PANDA_SEED_MAP_LABELS)` | Panda profile 은 `--robot` 없이 파싱하는 경로도 있어 명시 |
| `profile.py:776,789` | `build_panda_ag3s`: `target_extent == attention_surface` 이면 `ag3s.surface_pad_missing_cameras = True`, `info["surface_pad_missing_cameras"]` (그때만) | 카메라 차이 (아래) |
| `AG3S/adapters/panda/attention.py:81` | `PANDA_SEED_MAP_LABELS = ("noun_target:L8H6",)` | Panda head 허용 (task 3) |
| `AG3S/trajopt/attention_source.py:183` | `seed_map_source_ok(src, extra_labels=())` | 기본 `()` = 이전과 같음 |
| `AG3S/trajopt/serve_safe.py:3448-3467,3633` | `panda_seed_map_labels(args)`, `reject_bad_flag_combinations(ap, args, *, seed_map_labels=None)`: None 이면 `args.robot == "panda"` 일 때만 Panda head 추가 | RB-Y1 서버 (`--robot rby1`) 의 L10H7 고정 유지. 설계 이유: 허용 label 을 Panda adapter 가 소유하고 (attention.py), 핵심은 "추가 label 목록" 만 받는다. 목록 밖 head (`L8H2` 등) 는 Panda 에서도 거절 |
| `AG3S/ag3s/runtime/attention_surface.py:101-125` | `dummy_view` · `pad_missing_views` (+`__all__`) | 없는 카메라 채움 |
| `AG3S/ag3s/runtime/pipeline.py:299,1005-1033,1053-1079` | `AG3S.surface_pad_missing_cameras = False` (기본), 두 view 조립 지점 (`_attention_surface_update`, `_surface_views`) 에서 켜졌을 때만 pad, `robot_mask_source` 에 `"dummy"` 기록 | 켜지 않으면 호출이 이전과 같다 (RB-Y1 · 기존 판) |
| `AG3S/adapters/panda/server.py:332-380` | `UNPORTED` → `PORTED` 로 9개 이동: `target_extent` · `surface_target_check` · `surface_keep_last` · `surface_grasp_hold` · `surface_no_ball` · `attention_surface_device` · `surface_hull_fill` · `surface_contact_allowance_mm` · `subtask_from_latch`. 나머지 (`surface_split` · crease 3종 · `surface_memory_target_sticky` · `surface_reattach_hull` · vacancy · grasp set/corridor · `hsr5` · `hsr_gt_stage` · `passthrough_when_clear` · `support_surface_offset_mm` · `subtask_from_phase_machine` · `fm_guide*` · `robot_mask` · oracle/hybrid) 는 거절 유지 | task 1.2 |
| `server.py:400-410` | `check_unported_flags`: `target_extent` 는 `attention-surface` 만 (`oracle-visible` · `attention-surface-hybrid` 거절) | dest 만 풀면 다른 extent 가 새어 들어온다 |
| `server.py:685,696-698` | SafePolicy 인자에 `ss.subtask_from_latch_kwargs(args)` (꺼짐 = `{}`), `announce_subtask_from_latch`, `--placed-no-seed-certify` 의 "Panda 에서 inert" 기록은 latch label 이 있으면 쓰지 않음 | latch 의 PLACED → `home` label 이 생기므로 규칙이 실제로 동작한다 |
| `server.py:471-481,738-775,825-` | recorder meta: `subtask.source="latch"`, `ifg_cl1_record_meta` (켠 것만); config dump `final_config.attention_surface` (`surface_record`) · `subtask_from_latch` (`subtask_latch_record`) — 켰을 때만 | 기록 |
| `tests/benchmark/test_sl7_panda_p0m.py` (신설, 36 test) | 아래 | |
| `tests/benchmark/test_merge_dev_main_20261103_panda.py` | 목록에서 9개를 `PORTED_MERGE_DESTS` 로 옮김 + 확인 test | task 1.2 |
| `tests/benchmark/test_sl6h_panda_placement_contact.py:59` | 옛 HEAD 와 비교하는 version 집합에서 `P0M_VERSIONS` 제외 | 판이 늘어서 |

## flag 가 Panda 경로에서 연결되는 방식
| flag | 경로 |
|---|---|
| 표면 추출 (`--target-extent attention-surface --surface-target-check overlap --surface-keep-last --surface-grasp-hold --surface-no-ball --surface-hull-fill --surface-contact-allowance-mm 10`) | `profile.v11ab_build_kwargs(args)` 가 `serve_safe.ag3s_from_args` 의 `build_ag3s(...)` 호출 식을 AST 로 읽어 `target_exclusion_section(args)` 를 평가 → `v11ab_config_dict` 가 `esdf.update(...)` → `AG3SConfig`. RB-Y1 과 **같은 해석 함수**를 쓴다 (복사 없음). `build_panda_ag3s` 의 survival check 도 같이 건다. nohull 판은 hull 과 allowance 가 config 에서 꺼짐 (테스트) |
| `--attention-surface-device cuda` | `serve_safe.main` 이 Panda 분기 **앞에서** `apply_attention_surface_device(args)` (모듈 전역 `ATTENTION_SURFACE_DEVICE` 설정 + GPU warm-up) 를 이미 부른다 — 서버 시작 때 적용. CPU 단위 테스트는 `build_safe_server` 만 부르므로 `cpu` 로 남는다 (config dump 에 `device_requested` · `device_in_effect` 둘 다 기록) |
| `--subtask-from-latch` | `build_safe_server` 의 `SafePolicy(..., **subtask_from_latch_kwargs(args))` + 시작 announce. 단계 label 은 `SafePolicy._subtask_gate_input` → `ag3s.set_subtask` (searching · latched · closing → pick, held → place, placed → home). Panda 의 `AttentionPolicy(subtask_probe=None)` 는 gate 에 쓰이지 않으므로 그대로 둠 (kv_L4_v1 없음) |
| `--attention-source noun_target:L8H6` | 기존 SL6 경로 (`attention_policy_kwargs`) + 위 `reject_bad_flag_combinations` 허용 |
| `--neighbour-*` (nbc 판만) | 기존 SL6-A A6 경로 (`safe_policy_injections`: 손가락 `panda_leftfinger/rightfinger`, 손바닥 site = `PANDA_HOLD_SITE_OFFSET_M` (0,0,0.097) of `panda_hand`) |

**3단계 gate 가 Panda 에서 실제 동작하는지:** gate 는 `clustering.subtask_gate` (서버가 bool 로 확정, Panda 에서 True 확인) 와 `SafePolicy → ag3s.set_subtask` 만 필요하고 probe 는 필요 없다. 저장된 SafeLIBERO 한 프레임 (아래 smoke) 에서 `place` label 3 요청을 먼저 넣고 같은 프레임을 처리하면 `subtask_blocked == "first"` 이고 조작 대상이 채택되지 않으며, `pick` label 이면 채택된다 (테스트 `test_the_latch_label_drives_the_gate_on_a_panda_frame`). `SafePolicy._subtask_gate_input` 이 latch phase 5개를 위 표대로 one-hot 으로 바꾸는 것도 테스트.

## Panda 와 RB-Y1 이 다른 점
| 항목 | RB-Y1 | Panda | 처리 |
|---|---|---|---|
| 카메라 · `CAMERA_ORDER` | head · left_wrist · right_wrist 3개 (attention 768 token = 3×256) | head · left_wrist 2개. pi05_libero 의 세 번째 token block (512..767) 은 영 이미지 + image mask 꺼짐 (`adapters/panda/attention.py` 헤더). 서버는 `ATTENTION_ALIAS` 로 `cam_right_wrist` 지도도 버린다 | `select_target_surface` 는 세 이름이 정확히 있어야 해서 더미 right_wrist (깊이 `[0,0]=1.0` 한 픽셀, 로봇 mask 전부 True, K · T 는 첫 view 것, attention 0) 를 채움. 오프라인 연구 adapter `experiments/segmentation/surface_pilot.py` 가 쓰던 것과 같은 더미. 안 채우면 같은 프레임이 `missing_camera:right_wrist` 로 abstain (테스트로 증명). Panda attention 은 서버가 이미 un-mirror 해서 넘긴다 (`ATTENTION_CAMERAS` transform) |
| robot mask | 서버 기본 `--robot-mask` 없음 → FK 구 mask (`ARM_SPHERES`); `--robot-mask render` 는 RB-Y1 MJCF 렌더 | P0m 도 `ARM_SPHERES`. Panda self-filter 모델 4271 구로 mask (한 프레임 12–20 k px). `--robot-mask render` 는 Panda 에서 계속 거절 (`robot_mask` 는 UNPORTED) | 변경 없음. `robot_mask_source` 에 head/left_wrist = `spheres`, right_wrist = `dummy` 기록 |
| depth 입력 | 미터 환산 후 `cam.depth` | 같음 — `perception_replay.build_observations` → `wire.unpack_camera_observations` 가 Panda 관측을 같은 `Camera` 로 만든다 (한 프레임 smoke 에서 확인) | 변경 없음 |
| hull 채우기의 손바닥 점 | 양손 중 권한 손가락 구의 평균 | 같은 코드 (`pipeline._grasp_palm_points`): `gripper_joint_pairs()` 가 `panda_hand` 의 두 prismatic 손가락을 구조적으로 찾는다. **hold site (`PANDA_HOLD_SITE_OFFSET_M`) 는 쓰이지 않는다** — site 는 A6 손바닥 접촉 허용 (nbc 판) 과 reach 만 | 기본 자세에서 손가락 구 평균과 hold site 거리 ≈ 15.6 mm (테스트 `< 30 mm`), 80 mm 창에는 영향이 작다 |
| gate label | `kv_L4_v1` probe (서버 기본) 또는 `--subtask-from-latch` | probe 없음, latch 뿐 (`--subtask-from-latch`). latch 문턱은 `panda_latch_config()` — `empty_below` 만 `gripper_calibration.json` (0.0253), 나머지는 RB-Y1 기본값 | 변경 없음 (0.6) |
| `--placed-no-seed-certify` | `home` label 로 동작 | SL6 에서는 label 이 없어 inert 로 기록. P0m 은 latch PLACED → `home` 이라 **실제로 동작**하며 inert 기록은 끈다 | 기록 키만 조건부 |
| 요청 간격 | 0.53 s | 0.25 s | `subtask_confirm_frames`(=`target_confirm_frames` 3) · `hold_follow_to` 3 은 요청 수 규칙이라 wall-clock 이 절반 |

## P0 (RB-Y1) 값 vs Panda P0m 값이 다른 항목 (task 0.6 — 값은 바꾸지 않았다)
| 항목 | P0 (RB-Y1) | Panda p0m | 이유 |
|---|---|---|---|
| attention head | `noun_target:L10H7` | `noun_target:L8H6` | 0.2 |
| 카메라 | 3 (head + 두 손목) | 2 + 더미 right_wrist | 위 |
| 손가락 이웃 허용 (`--neighbour-contact-mm`) / 손바닥 이웃 허용 (`--neighbour-palm-contact-mm`) | 20 / 10 mm | 끔 (nbc 판만 20 / 10) | 0.3 D1 |
| 구 모델 | capsule × 0.05 · 간격 0.4 · 최대 32/capsule · 손가락 덮개 `--finger-cover both` | **적용 안 됨**: franka.yml 구 (gripper 22 + 손바닥 껍질 111) — 서버가 `constraint_sphere_options` · `finger_cover` 를 Panda 모델에 적용하지 않는다 (`v11ab_build_kwargs_not_applied`). 손바닥 껍질 `--palm-shell on --palm-shell-margin 5` 는 적용 (111 구) | SL4 이래 |
| 계획 창 | `--plan-horizon 16`, H 50 · K 8, 15 Hz | H 10 · K 5 · plan 10, 20 Hz | pi05_libero chunk |
| 관절 한계 | RB-Y1 MJCF | Franka URDF + 가속도 7.5 rad/s² | |
| ESDF 범위 | z 하한 0.0 | -0.10 m | LIBERO 탁자 높이 |
| 접촉 link | 양손 `ee_*` | 왼쪽 하나 `panda_hand` + 두 손가락 | |
| latch 문턱 | 닫힘 0.85 · reach 0.12 m · 막힘 0.05 · 빈손 0.10 | `empty_below` 0.0253 (보정), 나머지 RB-Y1 값 | 0.6 |
| 거리 창 | 손바닥–target 60 · 80 · 150 mm, 손가락 행 120 mm | 같은 값 (Panda 그리퍼는 더 작음) | 0.6 — 재조정은 사용자 판정 |
| 쥔 뒤 연속성 · 꺾임 변화율 · 손목 추종 0.1 · hold-follow 3 · ROI carve · 장애물 여유 10 mm · `--esdf-margin 0` | P0 값 | 같은 값 (1판 이래 유지) | |
| `--placed-no-seed-certify` | 동작 | 동작 (P0m 부터) | 위 |
| HSR6 · hsr5 · G1 · A4 · multi-seed · `--nontarget-evidence` | 끔 | 끔 | 0.1/0.5 |

## 단위 검증
```bash
source scripts/setup/env.sh; MUJOCO_GL=osmesa PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest <파일> -q -p no:cacheprovider   # 파일별
```
| 파일 | 결과 |
|---|---|
| `tests/benchmark/test_sl7_panda_p0m.py` (신설) | **36 passed** |
| `tests/benchmark/test_merge_dev_main_20261103_panda.py` | 38 passed |
| `tests/benchmark/test_sl6a_panda_final.py` | 24 passed |
| `tests/benchmark/test_sl6h_panda_placement_contact.py` | 10 passed |
| `tests/benchmark/test_panda_sl5j_recorder.py` | 15 passed |
| `tests/benchmark/test_panda_sl5a.py` (RB-Y1 `_sl5a_golden`) | 14 passed |
| `tests/trajopt/test_t92_subtask_from_latch.py` | 37 passed |
| `tests/trajopt/test_sl6a_injections.py` | 14 passed |
| `tests/benchmark/test_panda_sl4.py` · `test_eval_matrix.py` · `test_safelibero_client.py` | 14 · 7 · 7 passed |
| `tests/trajopt/test_t57a_attention_source.py` · `test_t90_hidden_probe.py` | 68 passed 1 skipped · 20 passed 6 skipped |
| `tests/ag3s/test_ifg_cl1_attention_surface.py` · `test_ifg_cl2_hybrid_extent.py` · `test_ifg_s2serve_gate.py` | 23 · 24 · 25 passed |
| `tests/ag3s/test_t78a_surface_post.py` · `t78b_reattach_hull_fill` · `t78b_attention_surface_gpu` · `t78d_crease_cut` · `t78e_memory_target_sticky` · `t78f_contact_allowance` | 14 · 13+1s · 9+6s · 13+5s · 11 · 8+1s passed (skip = GPU / 자산 없음) |

`test_sl7_panda_p0m.py` 가 보이는 것: (a) 세 판의 flag = 1판 − 뺀 11개 + 더한 13개 토큰 (명시 검사), nohull = p0m − hull 3 토큰, nbc = p0m + neighbour 4 토큰, 금지 option 없음, HSR6 probe 없음, 이전 판 flag 집합 · 기록 key 불변; (b) launcher argv · `match_ag3s_version` 읽기; (c) RB-Y1 서버는 `L8H6` 거절, Panda 는 `L8H6` 만 허용; (d) 거절 목록 + P0m 의 `check_unported_flags` 통과 (probe 없이); (e) CPU build 3 판: `target_extent == attention_surface`, hull · allowance 가 nohull 에서 꺼짐, `hsr6 is None`, multi-seed 끔, neighbour 깊이 0 (nbc 만 0.02 / 0.01), SafePolicy `subtask_from_latch`, 5 phase one-hot; (f) 더미 view / pad 기본 꺼짐; (g) 저장된 SafeLIBERO `safelibero_spatial_t0_e0` 한 프레임 → `process_multi_debug` 표면 영역 (CPU) · 더미 없이는 abstain · gate 동작; (h) probe 경로가 `AG3S_STORAGE_ROOT` 를 따름 (다른 root 로 subprocess).

**회귀 (bit 동일):** 코드 변경 전 HEAD 파일 7개를 임시로 되돌려 14 판 (V11AB · pb1~pb7 · 1판 · g1a4 · v2 · v3 · v3-e0to · pc) 의 server argv · 파싱된 args 전체 · `version_flag_values` · `match_ag3s_version` · AG3S config dict · SafePolicy 생성자 인자 설명 · `build_safe_server` info · `build_panda_ag3s` info · `recorder_meta` 를 덤프하고 (`scratchpad/dump_old_versions.py`), 변경 후와 비교 — `cmp` 로 **완전 동일**. (이 비교는 일회성이다; test 로 남긴 것은 `test_older_versions_keep_their_dump_keys` 와 sl6a/sl6h/sl5j 의 HEAD 비교 test.)

## verifier 가 알아야 할 것
- 새로 생긴 flag 는 없다 (serve_safe 의 기존 option 을 Panda 에서 풀었다). 새 기본 꺼짐 속성 `AG3S.surface_pad_missing_cameras` (False).
- **GPU 에서만 확인할 수 있는 것 (내가 안 한 것):** (1) `exclusion_trace` 비어 있지 않음 — cuRobo field build 가 CUDA 를 요구해 CPU smoke 는 `_build_esdf` 를 막고 그 앞까지 (표면 영역 · oracle memory code-1 샘플 · target check) 만 검증. (2) `--attention-surface-device cuda` 의 GPU 선택 (`attention_surface_gpu.select`) 이 더미 view 를 CPU 와 같게 처리하는지 (코드는 view 목록에 일반적이지만 실행 안 함). (3) hull 채우기가 실제 voxel 을 지우는지 (`stats.target_free.hull_fill`).
- 재생산이 필요한 산출물: 없음. 옛 기록과 호환: 깨지지 않음 (위 bit 동일).
- 위 smoke 에서 본 것 (판정 아님, 측정은 verifier): `safelibero_spatial_t0_e0` call 0–2 는 표면 영역이 **2 node (≈ 5×1.5×9 mm) 로 아주 작게** 채택됨 (target check 통과, hull 는 `too_few_points` 로 저장 안 됨); call 9 이후 샘플한 거의 모든 call (3 요청 간격 19 개 중 16 개) 은 영역이 100–3000 node 의 다른 곳에 잡혀 `not_ats_target` 으로 기각 — 그 요청은 이전 영역 유지 (`surface_keep_last`). 이 episode 하나, no-grasp 상태, L8H6 저장 attention 기준이다. 표면 추출이 LIBERO 물체에서 잘 되는지는 SL7 의 시험 대상이다.

## 남은 위험
1. **영역이 작거나 엉뚱함** (위 smoke): 표면 추출 · L8H6 seed · 설정값 (RB-Y1 에서 정한 5 mm node · jump 10 mm · ROI 0.15 m) 이 Panda 물체 크기에 맞는지 미확인. 이 값들은 0.6 에 따라 바꾸지 않았다.
2. **더미 right_wrist 의 가정:** 영 attention + 로봇 mask 전체 → 그래프에 기여 0. CPU 경로는 오프라인 pilot 과 같은 구성. GPU 경로는 실행 못 함.
3. **요청 간격 0.25 s** (RB-Y1 0.53 s): gate debounce 3 요청 · hold-follow 3 요청 · hull 의 `hull_age_frames` 가 wall-clock 으로 절반.
4. **latch 문턱 · 거리 창이 RB-Y1 값**: `--subtask-from-latch` 는 이 latch 에 전적으로 의존한다 (SL5 의 `reach-at-capture` · `grasp-reach-sticky` · `closing-hand-exempt` 같은 pb 수정 없음 — p0m 은 1판 base 라 pb 없음). 거짓 label 은 `place` 가 pick 중에 target 선택을 막는 방향으로 나타날 수 있다.
5. **`--placed-no-seed-certify` 가 처음으로 Panda 에서 동작**: `home` label 후 no_seed 프레임 인증 — 이전 Panda 판과 다른 점.
6. **손가락 구 모델:** RB-Y1 P0 의 capsule 얇게 · 손가락 덮개가 Panda 에는 없다 (franka.yml 구). 손바닥 구 경고 (P0 문서 §5-2) 는 Panda 에서 확인하지 않았다.
7. `--attention-source` 를 `L8H6` 외 head 로 바꾼 P0m 판은 시작 시 거절된다 (의도). 오프라인 head 탐색 스크립트 (`eval_real_ats_offline.py --head`) 는 P0m 판에는 L8H6 만 쓸 수 있다.
8. 단위 테스트의 `ATTENTION_SURFACE_DEVICE` 는 `cpu` (서버 `main` 만 `cuda` 로 설정). 서버 로그의 "GPU warm-up" 줄과 config dump 의 `device_in_effect` 로 확인할 것.

## 내가 기대하는 결과
(verifier 는 측정 전에 읽지 않는다) 세 판 모두 서버가 뜨고 첫 요청부터 `surface_frame.applied` 가 반복적으로 True 가 된다고는 기대하지 않는다 — smoke 에서 기각이 많았다. 판정은 verifier 의 측정으로.
