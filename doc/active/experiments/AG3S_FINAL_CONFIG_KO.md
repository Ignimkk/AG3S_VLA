# AG3S 최종 구성 (Panda / LIBERO 계열)

현재 시점의 최종 구성 **전체** 와 그 설명만 둔다. 구성이 바뀌면 본문을 고치고, 맨 아래 '교체 기록' 에 **무엇 → 무엇, 이유** 를 한 줄로 더한다.
정본: 판 정의 `AG3S/adapters/panda/profile.py` (`AG3S_VERSIONS`) · flag 정의 `AG3S/trajopt/serve_safe.py` (argparse) · Panda 서버 `AG3S/adapters/panda/server.py` · 판정 근거 `E1_GTTO_DECISIONS_20261104_KO.md`.
본문의 argv 는 실제 run manifest (`…/20261108T232618_…-pe12345-C/run_manifest.json` `settings.server_argv`) 와 token 단위로 대조했고, 실효 값은 `serve_safe.build_parser()` 로 parse 해 확인했다. `(미확인)` = code 로 끝까지 확인하지 못한 것.

## 0. 판 · 상태 (2026-11-09 05:3x KST)

| 항목 | 내용 |
|---|---|
| **판** | `AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe123458` |
| 인식 | **GT 없음** — AG3S 자기 인식 (VLA attention + VLA 가 손을 닫는 지점 + depth 기하). GT 는 채점에만 |
| 상태 | 평가 규약 (§6) 끝 2026-11-09 07:22 (`experiments/segmentation/pe1k_eval_queue.sh`, `outputs/2026-11-09_집기놓기_장애물취급/eval_pe123458/`): LIBERO 27 **22**, SafeLIBERO 11 성공 8 · 안전 성공 6, LIBERO-Safety 10 **5** (위반 0) ; E1a 40 (2026-11-09 07:22–08:42, dev-38 `qe4_queue.sh`, 표 `outputs/2026-11-08_E1a_SafeLIBERO_B5/E1a_spatial_LI_표.md`): TSR 22 · CAR 23 · **SSR 15** (37.5 % [24, 53]; π0.5 대비 짝 9 / 0, McNemar p = 0.0039) — 목표 > 23 (CBF GT) 미달, GT 없는 이전 판 gt24att 5 · B5 GT 13 보다 높음 — §8 |
| 참조 상한 | **B5** `…-gt2-…-obst3-pts-nocarve-fh-t100` (같은 뼈대 + GT 인식, 사용자 판정 기준 판): LIBERO 27 / 27, SafeLIBERO 보정 11 ep 9 / 11 (실제 충돌 1), LIBERO-Safety 9 (위반 0) |
| 넘어야 할 목표 | E1a (SafeLIBERO Spatial LI, test init 5–14, n = 40, `main_aegis`) **SSR > 23 (CBF GT)**. 같은 셀: B5 (GT) 13 · GT 없는 B5 5 · π0.5 단독 6 · AEGIS 17 · KNOWS (GT) 21 (dev-38 G108–G110, `outputs/2026-11-08_E1a_SafeLIBERO_B5/E1a_spatial_LI_표.md`) |

## 1. 전체 흐름

LIBERO 시뮬레이터 (20 Hz) 가 5 step (0.25 s) 마다 client (`benchmark/runners/libero_safe_client.py`) 를 통해 서버에 요청한다. 요청 = 공식 π0.5 입력 + AG3S 용 카메라 기하 · 로봇 상태 (+ 둘째 call 부터 실행 feedback).
서버 (`serve_safe --robot panda` → `AG3S/adapters/panda/server.py`) 는 VLA chunk 를 받아 AG3S 가 만든 3D 장면 (target 을 뺀 cuRobo TSDF/ESDF) 위에서 TO 가 검사하고, 위반이 없으면 그대로, 있으면 정책 action 공간에서 최소로 고쳐 돌려준다.

| 단계 | 하는 일 | 핵심 값 | code |
|---|---|---|---|
| 0. client 관측 | `agentview`→`head`, `robot0_eye_in_hand`→`left_wrist` 두 카메라의 RGB · depth · K · `T_base_cam`, 촬영 순간 관절 상태 (`qpos_image`). 첫 call 에 `ag3s/reset` · `policy_seed`, 둘째 call 부터 exec feedback (`panda/exec_applied_osc`) | render 256 (정책 입력은 180° 회전 + pad 224), depth uint16 mm, `ag3s/phase = approach` 고정, replan 5 step, wait LIBERO 10 · SafeLIBERO 20 step, max_steps LIBERO-Spatial 220 · SafeLIBERO-Spatial 280 | `libero_safe_client.py`, `benchmark/adapters/libero_panda/camera.py` `CAMERA_TABLE` |
| 1. VLA | π0.5 `pi05_libero` 가 7-D EE delta chunk (OSC_POSE 6 + gripper 1) 10 행. policy seed = seed·100000 + task·1000 + ep | H 10, 실행 K 5 | `server.py` `ChunkToJointPolicy.infer` |
| 1a. attention | 같은 checkpoint 의 두 번째 사본에서 attention 추출, 좌우 반전 되돌림 | 셀 `action:last:L8H2` (layer 8 · head 2 · denoise step 0), 카메라 head · left_wrist | `server.py` `build_safe_server`, `attention_policy.py` |
| 2. chunk → 관절 | response model `y_t = α y_{t−1} + β x_t` 로 EE 경로를 예측하고 행마다 DLS IK | OSC scale 0.05 m / 0.5 rad, pos α 0.717 β 0.071, rot α 0.875 β 0.031, IK ≤ 60 회 | `adapters/panda/action.py` `delta_chunk_to_joint_chunk` |
| 3. grasp latch | exec feedback (측정 개도 · 실행 gripper 명령) 으로 SEARCHING → LATCHED → CLOSING → HELD → PLACED. 이 판은 **opening rule** 이 evidence 상태기계를 대신한다 | attach: 닫힘 명령 < 0.85 ∧ 개도 정지 (Δ < 0.02) ∧ 개도 ≥ 0.03 ∧ reach ≤ 120 mm ∧ 손가락이 0.05 이상 실제로 닫힘. detach: (열림 명령 ∧ 개도 > attach 개도 + 0.02) ∨ 개도 < 0.02 | `AG3S/ag3s/.../grasp_latch.py` (`_observe_opening_rule`) |
| 3a. 손 닫는 지점 (close seed) | attach 전 (SEARCHING · LATCHED · CLOSING) 이면 chunk 에서 gripper 명령이 0.85 아래로 처음 떨어지는 행의 손끝 중점을 AG3S 에 seed 로 준다 | 손끝 = 손가락 link 마다 접근축 방향으로 가장 먼 제약 구의 중심, 두 손끝의 중점 | `safe_policy.py` `_close_seed_request`, `fingertip_alignment` |
| 4.1 장면 재구성 | 카메라별 depth → 점 → base frame voxel 합침 | voxel 5 mm, depth 0.05–3.0 m, range_max 2.0 m, ≤ 200 000 점 | `pipeline.py` `AG3S._run` → `multiview.py` `fuse_observations` |
| 4.2 robot self-filter | 로봇 (+ 쥔 물체) 구 모델 안의 점을 지운다. 카메라마다 촬영 순간 상태. 같은 pixel mask 를 TSDF 에도 | self-filter 구 4271 (franka.yml 61 + `confined_hand_mount` cover), inflation 0 | `stages/robot_filter.py` |
| 4.3 받침면 | RANSAC 으로 위를 향한 평면 (탁자) → 받침면 mask | ≤ 2 평면, 문턱 8 mm, inlier ≥ 500, 법선 25° 이내 | `stages/support_surface.py` `fit_support_surfaces` |
| 4.4 attention lifting | 카메라별 attention 을 점마다 올려 max 로 합침. 정규화 · raw 둘 다 | 정규화 percentile [5, 99], seed percentile 95 | `stages/attention_lifting.py` `lift` |
| 4.5 target grounding | (a) **손 닫는 지점 컷**: seed 15 cm 안 비받침 점에서 국소 받침면 (위 향한 법선 띠) · 평평한 수직 벽을 빼고, 10 mm 격자 26-연결 성분 중 seed 최근접 (≤ 60 mm) 을 grounding 입력으로. 그 점의 attention = `clip(1 − d/0.15, 0.2, 1)`. (b) seed 가 없으면 attention grounding: DBSCAN cluster 점수 (attention 0.7 · geometry 0.3) + attention split / partition. `TargetConfirm` 이 identity · 전환 · 목적지 · admissibility 를 정한다 | DBSCAN eps 30 mm · min 20 점. 전환 확인 3 frame (seed 컷은 attach 전 즉시 채택, closing 부터 잠금). 최대 개도 78 mm. 받침 test: 바닥이 받침면 위 60 mm 넘게 떠 있으면 거부. 자른 물체의 국소 받침면은 episode 동안 받침 test 에 남는다 | `pipeline.py` `_close_seed_grounding_input`, `stages/close_seed.py` `split_at_seed`, `stages/target_grounding.py` `ground_target` · `TargetConfirm` |
| 4.6 제외 gate · 장애물 물체 | admissible 조작 대상만 target 제외 · 접촉 권한을 받는다. 장애물 물체 집합 = 받침면을 뺀 점군의 15 mm 연결 성분 중 (조작 대상 15 mm 안 · 그 밑면 아래 0–30 mm · 인식 목적지 15 mm 안 · 쥔 물체 snapshot 15 mm 안) 이 아닌 것에서, 쥔 동안 **손과 같이 움직인 점** (손 frame 에서 제자리 · 정지로 설명 안 됨 · 손끝 + 물체 폭 + 20 mm 안, 누적) 을 뺀 것, 최근 3 요청 합집합 | 15 mm 격자; 움직임 판정 match 10 mm · drift 25 mm | `pipeline.py` `_exclusion_gate`, `_scene_obstacle_points`, `_held_motion_mask`, `stages/held_motion.py` |
| 4.7 쥔 물체 | attach 때 관측 점으로 모델. self-filter · 기록용 body cover 구 (cap 10 mm), TO 질의용 관측 점 위 20 mm 격자 껍질 구 (반지름 17.3 mm, 받침면까지 채움) | `held_point_model_m = 0.02` | `pipeline.py` `_attach_body_cover` |
| 4.8 ESDF | cuRobo `Mapper` 로 TSDF 적분 → ESDF. coarse = 작업공간 전체, fine = 손 swept AABB + 조작 대상 창. **target-free 계층** (ball-owned 로 target seed 만 지움). ROI carve (조작 대상 · 목적지 · 손 경로 상자 안에서 이번 광선이 지나간 점유 voxel 비움). 쥔 물체는 seed 에서 빼고 부호 보정. 최근접 표면 label 은 `site_index` | coarse 20 mm, fine 5 mm (128³), TSDF 5 mm, truncation 3 voxel · fine 0.03 m, max_distance 0.4 m, 작업공간 base [-0.3, 1.2] × [-0.9, 0.9] × [-0.10, 1.6] m, ROI pad 0.1 m | `pipeline.py` `_build_esdf`, `fields/curobo_builder.py` `CuroboFieldBuilder.update` |
| 5.1 TO look-ahead gate | 계획 10 step 중 6–10 (꼬리) 은 장애물 행만 보는 look-ahead. 꼬리 reference 가 장애물 20 mm 안으로 오지 않으면 실행 창 5 step 만 푸는 window optimizer | plan 10, exec 5, gate 20 mm | `sqp.py` `_solve_window` |
| 5.2 pass-through | 순서대로: 기하 인증 (없으면 HOLD) → 일반 위반 ≤ τ 100 mm (실행 창) → **장애물 물체 행** 위반 ≤ 10 mm (chunk 10 step 전체, 행 = 집합 ∧ 기존 분류) → 관절 한계 초과 ≤ 0.2. 모두 통과하면 VLA chunk 를 그대로 (gripper 포함) | τ 100 mm, τ_obs 10 mm, 한계 0.2 | `sqp.py` `_clear_passthrough` · `_obstacle_row_refuses`, `linearize.py` `obstacle_object_rows` |
| 5.3 action QP | 통과 못하면 `min ‖Δa‖²` — 정책 OSC action 행 (10 × 6) 위의 선형화 충돌 행 (`value + grad·M Δa + s ≥ 5 mm backoff`) + `|a_ref + Δa| ≤ 1`. gripper 열은 안 바꿈 | 재선형화 4 회, slack 가중 1e6, OSQP eps 1e-6 | `sqp.py` `_solve_ee_action`, `action.py` `ee_action_map` |
| 5.3′ joint SQP | action map 이 없는 window optimizer 경로에서만: 관절 공간 SQP 3 회 · 50 ms | OSQP eps 1e-4, trust radius 0.15 | `sqp.py` `solve` |
| 6. verdict | 실제 기하로 최종 검사. HOLD 사유: 기하 미인증 (`occluded_target` 은 인증), solver 실패, 위반 > 100 mm (장애물 물체 행 포함 — τ_obs 10 mm 는 pass-through 결정에만). HOLD 대신 TO 궤적 따르기 (follow_to) ≤ 3 chunk | `safety.violation_tolerance = 0.1 m`, `require_certified_geometry = True` | `safe_policy.py` `_verdict` · `_reasons` |
| 7. 역변환 · 실행 | refined 관절 chunk → EE delta (conversion C `lag_inverse_tracking`; 바뀌지 않은 행은 원래 bit). action QP 를 푼 chunk 는 `clip(a_ref + Δa)` (refined 와 QP 궤적이 1e-4 rad 안에서 같을 때). client 는 앞 5 행 실행, 적용 OSC 행을 다음 call feedback 으로. `safe = False` 면 HOLD 행 (팔 delta 0, 마지막 gripper 명령) | `EE_ACTION_MATCH_RAD = 1e-4` | `server.py` `PandaSafeServer.infer`, `action.py` |

pe12345 run (LIBERO s7 · s8 · s9 t3 · 4 · 7 · 9 + SafeLIBERO t3, 553 call) 의 call 경로: pass-through 432 · 장애물 행 거절 → action QP 89 · → joint SQP 1 · 기하 미인증 → HOLD 31.

## 2. 역할별 인식 요약 (무엇이 target · 장애물 · 목적지인가)

| 역할 | 구성 | flag |
|---|---|---|
| target identity | VLA chunk 가 손을 닫는 지점 아래 물체 (국소 받침면 · 벽을 빼고 자른 성분), attach 전 즉시 채택, closing 부터 잠금, 자른 점 attention 은 손끝 거리로 가중. 그 지점이 없는 call 은 attention grounding (split · partition) | `--close-seed-grounding` (+ `--attention-split-*`, `--split-supersedes-anchor`) |
| target 받침 | 탁자 RANSAC 평면 + 자른 물체의 국소 받침면 (episode 동안) | (위에 포함), `--supported-max-bottom-mm 60` |
| target 제외 (ESDF) | target-free 계층 (ball-owned), 접촉 권한 link (손바닥 + 손가락 둘) 만 그 계층에 질의 | `--target-field-policy exclude-authorized` `--target-exclusion ball-owned` `--owned-max-dist-voxels 3` `--nontarget-evidence surface` `--target-permission-strict` |
| 장애물 물체 | 받침면을 뺀 점군의 15 mm 연결 성분 중 target · 그 받침 · 목적지 · 손 안 쥔 물체가 아닌 것 (3 call 합집합). 쥔 물체는 attach snapshot 근처 성분과, 손과 같이 움직인 점 (들어 올린 뒤 보이는 바깥 벽 · 바닥) 둘 다로 뺀다 | `--scene-obstacle-objects` `--scene-obstacle-held-exclude` `--scene-obstacle-held-motion` |
| 장애물 물체 행 (TO) | 최근접 표면이 장애물 물체 집합 위 **이고** 기존 분류 (조작 대상 · target 부피 · 받침면 · 목적지 label · 쥔 물체 잔재) 도 장애물일 때 | `--passthrough-obstacle-class-tests` |
| 목적지 | 쥔 동안에만 인식으로 등록. 손 (60 mm) 밖 attention 1 위 cluster 는 잡을 수 있는 크기여도 후보 (접시) | `--destination-after-held` `--destination-held-leader` |
| 로봇 · 쥔 물체 pixel | self-filter (FK 구 모델, GT mask 없음) | — |

## 3. Flag 전체 (실효 값)

반복된 flag 셋 (`--links`, `--w-deflection-rate`, `--violation-tolerance-mm`) 은 모두 `store` 라 뒤 값이 이긴다. 'Panda' 열 = Panda 서버가 실제로 그 값을 쓰는가.

### 3.1 입력 · VLA · 서버

| flag (실효 값) | 뜻 | Panda |
|---|---|---|
| `--config pi05_libero` | openpi train config | 사용 |
| `--checkpoint <storage>/checkpoints/panda/pi05_libero` | checkpoint (attention 용 사본을 하나 더 띄움) | 사용 |
| `--model-xml robots/rby1/…/model_transport.xml` | RB-Y1 씬 XML | **안 씀** — Panda 는 URDF `panda_libero.urdf` + 가속 7.5 rad/s² (`server.py` `_replace_horizon_limits`) |
| `--robot panda` | Panda 서버로 보냄 (launcher 가 붙임) | — |

### 3.2 재구성 · 받침 · attention · grounding

| flag (실효 값) | 뜻 | Panda |
|---|---|---|
| `--supported-max-bottom-mm 60` | 후보 cluster 바닥이 받침면 위 60 mm 보다 높고 손에도 없으면 admissible 아님. 자른 물체의 국소 받침면도 이 test 에 더해짐 | 사용 |
| `--attention-split-radius-mm 12` | attention split 재군집 반경 (기본 15) | 사용 |
| `--attention-split-ratio-source raw` | split ratio gate 가 정규화 전 attention 을 읽음 | 사용 |
| `--attention-split-partition` | 재군집이 한 덩어리로 끝난 cluster 를 attention 으로 가름 | 사용 |
| `--split-supersedes-anchor` | 쥐기 전 split 의 target 조각이 merge 된 anchor 를 대신함 | 사용 |
| `--close-seed-grounding` | SL-PE1: target identity = 손 닫는 지점 아래 자른 물체 (§1 4.5 (a)) | 사용 |
| `--latched-identity-hold` | LATCHED · CLOSING 동안 '안 보임' 만으로 identity 가 바뀌지 않음 | 사용 |
| `--closing-keep-geometry` | closing 중 (attach 전) 조작 대상 기하를 그 frame 관측으로 바꾸지 않음 | 사용 |
| `--kept-geometry-follow` | 유지한 기하를 이번 frame 비받침 점에 반지름 고정 구로 맞춰 평행 이동 (밀린 물체 따라감) | 사용 |
| `--placed-no-seed-certify` | PLACED 뒤 `home` label 에서 grounding `no_seed` frame 을 인증 | 사용 (`--subtask-from-latch` 가 home 을 만듦) |

### 3.3 target 제외 (target-free 계층)

| flag (실효 값) | 뜻 | Panda |
|---|---|---|
| `--target-field-policy exclude-authorized` | 아직 쥐지 않은 target 에 대해 접촉 권한 link (`panda_hand` + 손가락 둘) 질의점만 target 이 빠진 계층에 거리를 물음. 다른 link 에게 target 은 일반 장애물 | 사용 |
| `--target-exclusion ball-owned` | 공 안의 seed 마다 최근접 target 점 vs non-target 관측점 거리 비교, non-target 쪽 seed 는 남김 (이웃 보존) | 사용 |
| `--owned-max-dist-voxels 3` | 두 거리 모두 3 voxel 보다 멀면 unowned (지움) | 사용 |
| `--nontarget-evidence surface` | non-target 근거로 공 표면 구 맞춤 사용 (맞춘 구 + 2 voxel 껍질 안 점은 target 표면) | 사용 |
| `--target-permission-strict` | target 예외 (꼬리 drop · margin 면제 · sweep 제외) 를 권한 행에만 | 사용 |

### 3.4 grasp latch

| flag (실효 값) | 뜻 | Panda |
|---|---|---|
| `--grasp-opening-rule` | attach · detach 를 개도와 명령만 쓰는 규칙으로 (§1 단계 3) | 사용 — evidence 상태기계를 통째로 대신함 |
| `--grasp-opening-rule-moved 0.05` | attach 에 '손가락이 실제로 0.05 이상 닫힘' 검사 추가 | 사용 |
| `--reach-at-capture` | reach gate 가 조작 대상 마지막 관측 중심을 그 촬영 때 로봇 상태와 짝지음 | 사용 |
| `--latch-phase` | 접촉 phase 를 latch 에서: LATCHED → pre_grasp, CLOSING / HELD → grasp | 사용 |
| `--grasp-zone-mm 10` | LATCHED / CLOSING 동안 조작 대상 100 mm 안 손 행 중 target · 받침면 행에 10 mm 더 허용 | 사용 (같은 zone 의 손 행은 `--grasp-zone-release` 가 아예 풂) |
| `--grasp-reach-sticky` | 한 closing 시도 안에서 통과한 reach 는 계속 통과 | 사실상 inert (evidence 경로 전용) |
| `--grasp-drop-from-open` | 'dropped' 검사를 마지막 열림 요청 개도에서 잼 | 사실상 inert (evidence 경로 전용) |
| `--closing-hand-exempt` | CLOSING ∧ evidence `closed · not_empty · blocked` 면 손 행의 조작 대상 행 제외 | inert (opening rule 에 그 키 없음; 553 call 중 0 회) |
| `--held-hand-exempt` | HELD 동안 쥔 손 접촉 link 행 중 최근접 표면이 조작 대상인 행 제외 | 사용 (553 call 중 112 call) |
| `--held-slip-detach` | 닫힌 채 빠지면 detach | 사실상 inert (opening rule 은 '개도 < 0.02 → 빈손 detach' 를 따로 씀) |
| `--subtask-from-latch` | gate label 을 latch 에서: pick / place / home (probe 없음) | 사용 |

### 3.5 쥔 물체 · 목적지

| flag (실효 값) | 뜻 | Panda |
|---|---|---|
| `--held-body-cover` | 쥔 물체 body cover 구 (맞춘 구 + 받침면까지 바닥 구 + 미끄럼 5 mm) | self-filter · 기록에만 (TO 질의는 점 모델) |
| `--held-cover-cap-mm 10` | body cover 구 반지름 ≤ 관측 반폭 + 10 mm, 중심 10 mm 안 | body cover 에만 |
| `--held-point-model-mm 20` | TO 질의 구 = 관측 점 위 20 mm 격자 껍질 구 (반지름 17.3 mm) | 사용 |
| `--held-beneath-contact` | HELD 중 쥔 물체 자기 행 중 최근접 표면이 그 아래 (위 45° 안, 발자국 + 1 voxel) 인 행은 묶지 않음 | 사용 |
| `--no-self-collision` | 쥔 물체 ↔ 로봇 구 제약 끔 | 사용 |
| `--destination-after-held` | SL-PE2: 목적지 등록 · 갱신은 쥔 동안만 | 사용 |
| `--destination-held-leader` | SL-PE5: 쥔 동안 손 밖 attention 1 위 cluster 는 크기 무관 목적지 후보 | 사용 |
| `--destination-margin-mm 0` | 쥔 물체 행의 목적지 표면 여유 20 → 0 mm | 사용 |

### 3.6 장애물 물체 (인식)

| flag (실효 값) | 뜻 | Panda |
|---|---|---|
| `--scene-obstacle-objects` | SL-GTTO20: 자기 인식 장애물 물체 집합 (§2) | 사용 |
| `--scene-obstacle-held-exclude` | SL-PE3: attach 중 손 안 쥔 물체 snapshot 점 15 mm 안 성분 제외 | 사용 |
| `--scene-obstacle-held-motion` | SL-PE8: attach 중 손과 같이 움직인 점을 누적해 제외. 점 판정 = 손 frame 에서 snapshot · 검증된 점 · 직전 request 점 10 mm 안 ∧ 정지라면 손 frame 에서 25 mm 이상 밀렸어야 함 ∧ 직전 request 의 base-frame 점 10 mm 밖 (정지로 설명 안 됨) ∧ 손끝 중점에서 snapshot 폭 + 20 mm (≤ 150 mm) 안. snapshot 자체는 지우지 않음. 성분의 절반 이상이 그 점이고 성분 전체가 경계 안이면 성분째 | 사용 |

### 3.7 ESDF (cuRobo TSDF / ESDF)

| flag (실효 값) | 뜻 | Panda |
|---|---|---|
| `--esdf-backend curobo` | 필드 = cuRobo Mapper | 사용 |
| `--voxel 0.020` | coarse 계층 20 mm | 사용 |
| `--fine-voxel 0.005` | fine 계층 5 mm | 사용 |
| `--tsdf-voxel 0.005` | TSDF 5 mm | 사용 |
| `--esdf-margin 0` | 모든 표면 기본 여유 0.05 → 0 m (장애물만 `--obstacle-margin`) | 사용 |
| `--esdf-host-mirror` | GPU 격자를 chunk 마다 한 번 host 로 복사 (값 동일, 속도만) | 사용 |
| `--roi-carve` | 조작 대상 · 목적지 · 손 경로 상자 ± pad 안에서만 이번 광선이 지나간 점유 voxel 비움 | 사용 |
| `--roi-carve-footprint` | carve 판정을 voxel 투영 영역 전체로 (가는 물체 가장자리 보존) | 사용 |
| `--roi-carve-footprint-pad-px 0` | 영역 판정 여유 0 px | 사용 |
| `--roi-carve-footprint-robot ignore` | 영역 안 robot pixel 은 세지 않음 | 사용 |
| `--roi-carve-device cuda` | carve 판정을 GPU 에서 (같은 판정) | 사용 |

### 3.8 robot sphere model

| flag (실효 값) | 뜻 | Panda |
|---|---|---|
| `--links arms` (`gripper` 를 덮음) | 충돌 제약 link = `panda_link1..7` + `panda_hand` + 손가락 둘 | 제약 구 **170** (franka.yml 59 + 손바닥 껍질 111) |
| `--capsule-radius-scale 0.05` | capsule 반지름 배율 | **적용 안 됨** (`v11ab_build_kwargs_not_applied`) |
| `--sphere-spacing 0.4` | capsule 위 구 간격 | **적용 안 됨** |
| `--max-spheres-per-capsule 32` | capsule 당 구 상한 | **적용 안 됨** |
| `--finger-cover both` | 손가락 몸통 구 (RB-Y1) | **적용 안 됨** (franka.yml 손가락 구 그대로) |
| `--palm-shell on` | 손바닥 바깥 껍질 구 | 사용: `panda_hand` 111 구, 반지름 1.9–34.2 mm (`adapters/panda/panda_palm_shell.json`) |
| `--palm-shell-margin 5` | 껍질 행의 장애물 여유 5 mm (다른 행 10 mm) | 사용 |

### 3.9 TO · pass-through · verdict

| flag (실효 값) | 뜻 | Panda |
|---|---|---|
| `--plan-horizon 16` | 최적화 step 수 | **Panda 가 덮음**: `PANDA_HORIZON` = H 10 · K 5 · plan 10 |
| `--lookahead obstacles` | 실행 창 밖 step 은 장애물 행만 쓰고 인증하지 않음 | 사용 |
| `--lookahead-gate-mm 20` | 꼬리 reference 가 장애물 20 mm 안일 때만 꼬리 사용 | 사용 (닫히면 window optimizer → joint SQP) |
| `--qp-eps 1e-4` | joint SQP OSQP eps | joint SQP 경로에만 (action QP 는 1e-6) |
| `--w-deflection-rate 0` (`20` 을 덮음) | `‖ΔD‖²` 가중 | 0 = 항 없음 |
| `--w-continuity 0` | 앞 chunk 연속성 항 | 0 = 항 없음 |
| `--wrist-track-scale 0.1` | 쥐기 전 손바닥이 target 150 mm 안이면 joint5–7 추종 가중 × 0.1 | joint SQP 비용에만 |
| `--to-fast-eval` | FK / ESDF memo (출력 동일, 속도만) | 사용 |
| `--obstacle-margin 0.010` | target 이 아닌 장애물에만 10 mm 여유 | 사용 |
| `--passthrough-when-clear` | 위반 없고 한계 안이면 최적화 없이 VLA chunk 그대로 | 사용 |
| `--violation-tolerance-mm 100` (`25` 를 덮음) | 일반 τ = pass-through 문턱이자 verdict 허용 위반 | 사용 |
| `--passthrough-limit-tolerance 0.2` | 관절 한계 초과 0.2 이하면 통과 | 사용 |
| `--passthrough-obstacle-tolerance-mm 10` | 장애물 물체 행에는 τ 대신 10 mm | 사용 |
| `--passthrough-obstacle-full-horizon` | 장애물 물체 행을 chunk 10 step 전체에서 봄 | 사용 |
| `--passthrough-obstacle-class-tests` | SL-PE4: 장애물 물체 행 = 집합 ∧ 기존 분류 | 사용 |
| `--occluded-certify` | `occluded_target` frame 을 TO 도 인증으로 읽음 (충돌 검사는 그대로) | 사용 |
| `--hold-follow-to 3` | HOLD 대신 TO refined chunk 실행 (collision 사유뿐 · margin 안쪽 · clearance 악화 없음), 인증 뒤 ≤ 3 chunk | 사용 |
| `--grasp-zone-release` | LATCHED / CLOSING 동안 조작 대상 100 mm 안 손 행은 최근접 표면이 무엇이든 풂 (이웃 장애물 포함) | 사용 (225 call) |

### 3.10 action QP · 실행 환경

| flag (실효 값) | 뜻 | Panda |
|---|---|---|
| `--ee-action-qp` | 통과 못한 chunk 는 정책 OSC action 행 위 `min ‖Δa‖²` (6 자유도), 서버는 `a_ref + Δa` 를 보냄 | 주 optimizer 에만 (window optimizer 에는 map 없음) |
| `--blas-threads 1` | OpenBLAS 1 thread (parser 기본값과 같음) | 사용 |
| *(후보, 미채택 — 꺼짐)* `--ee-action-qp-translation-only` (SL-QE2, dev-38) | action QP 의 회전 3 자유도를 고정하고 평행이동 3 자유도만 바꿈 | B5 (GT) 에서 B5-tro 와 B5 비교 (2026-11-09, dev-38): E1a SSR 20 vs 13 (짝 9 / 2) · TSR 27 vs 22, LIBERO 27 27 vs 27, SafeLIBERO 11 안전 성공 9 vs 8, LIBERO-Safety 10 8 vs 9 (human_safety e2 시간 초과). GT 없는 판 결과 (pe123458-tro vs pe123458, 2026-11-09 12:10–15:11, 같은 평가 규약 · §8.3): 보정 13 vs 14 · LIBERO 27 23 vs 22 · SafeLIBERO 11 안전 성공 5 vs 6 · LIBERO-Safety 10 5 vs 5 · E1a 40 (개발 분할) SSR 17 vs 15 (p = 0.69), CAR 20 vs 23 → 사전 기준 (SafeLIBERO 안전 성공 ≥ 지금 구성) 미달로 **미채택 유지** (G172) |

launcher 가 붙이는 것: `--port <자동>`, 평가 queue 의 `--record-constraints <run>/server_constraints --record-constraints-esdf full`, 환경 `AG3S_PANDA_CONVERSION=C` (`launch_libero_paired.py --conversion C`).

## 4. argv 에 없는 implicit 값

| 무엇 | 값 | 출처 |
|---|---|---|
| horizon | H 10 · K 5 · 20 Hz · plan 10 | `server.py` `PANDA_HORIZON` |
| 관절 한계 | Panda URDF, 가속 7.5 rad/s² 스칼라, 속도 · 가속 scale 0.9, position margin 0.02 | `server.py`, TO config `limits` |
| 손 · 접촉 link | 한 손 `left`, 접촉 권한 `panda_hand` · `panda_leftfinger` · `panda_rightfinger`, 쥔 물체 부모 `panda_hand`, reach 는 `panda_leftfinger` 에서 | `profile.py`, config `contact.contact_links` |
| gripper | 정규화 개도 = `panda_finger_joint1 / 0.04` (1 = 열림), 관절 chunk gripper = `(1 − clip(a6)) / 2`, 최대 개도 **78 mm** (admissibility 문턱) | `server.py` `panda_gripper_convention`, config `gripper_max_opening` |
| latch 문턱 | `gripper_closed_below 0.85`, `settle_epsilon 0.02`, reach 0.12 m; opening rule attach 개도 ≥ 0.03, detach 상승 0.02, 빈손 < 0.02 | `grasp_latch.py`, `server.py` `panda_latch_config` |
| grasp zone | 100 mm (수평) | config `grasp_contact.grasp_zone_radius_m` |
| ESDF 그 밖 | unknown = free, 시간 · frustum 감쇠 끔, incremental, fine window `hand_swept`, Panda z 바닥 -0.10 m (탁자 윗면 z -0.012) | `AG3SConfig.esdf`, `profile.py` `PANDA_ESDF_BOUNDS_LOWER_Z` |
| 접촉 phase | transit 1.0 · approach 0.4 · pre_grasp 0.1 · grasp 0.0 (+ 접촉 권한); `contact_margin = 0` 이라 margin_scale 의 실제 효과 (미확인) | `AG3SConfig.contact` |
| action QP 상수 | 재선형화 4 회, slack 1e6, OSQP eps 1e-6 · max_iter 20000, backoff 5 mm, 10 행 × 6 자유도 | `sqp.py` |
| joint SQP 상수 | 3 회 · 50 ms, trust radius 0.15, w_track 1, w_smooth 0.05, w_slack 1000, 활성 band 0.35 m | TO config |
| Panda 가 거부하는 flag | 포팅되지 않은 merge flag 를 기본값이 아니게 주면 시작 전 거부 | `server.py` `UNPORTED_FINAL_CONFIG_DESTS` |

## 5. 판 이름 segment

GT 를 쓰는 B5 (`…-gt2-…-obst3-pts-nocarve-fh-t100`) 에서 GT flag 를 빼고 (`_NONGT_DROP`) SL-PE 를 더한 판이다. 아래는 각 segment 가 이 판에 남긴 flag.

| segment | 남긴 flag | 뜻 |
|---|---|---|
| (base V11AB) | `--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --obstacle-margin 0.010 --held-body-cover --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20` | RB-Y1 V11AB 서버 설정 (`scripts/benchmark/v11ab/start_server.sh` · `env.sh`) |
| `commonto` | `--to-fast-eval --esdf-host-mirror --target-exclusion ball-owned --owned-max-dist-voxels 3 --target-permission-strict --wrist-track-scale 0.1 --placed-no-seed-certify --kept-geometry-follow --roi-carve* (4) --blas-threads 1 --palm-shell on --palm-shell-margin 5` + Panda 수정 `--reach-at-capture --links arms --held-hand-exempt --latch-phase --grasp-zone-mm 10 --grasp-reach-sticky --grasp-drop-from-open --closing-hand-exempt` + `--subtask-from-latch` | 공통 TO (최종 구성 flag 에서 인식 · HSR6 · multi-seed 를 뺀 것 + Panda grasp 처리) |
| `att` | `--attention-split-radius-mm 12 --attention-split-ratio-source raw --attention-split-partition --split-supersedes-anchor --nontarget-evidence surface` | GT 인식 (`--target-extent oracle-visible --oracle-extent-identity gt --grounding-source oracle --oracle-admissible --oracle-held-target --oracle-destination --oracle-robot-mask --oracle-obstacle-*`) 을 빼고 attention grounding |
| `termsoff` | `--w-continuity 0 --w-deflection-rate 0` | 연속성 · deflection-rate 항 끔 |
| `pass` | `--passthrough-when-clear` | 위반 없는 chunk 는 통과 (T88) |
| `zr` | `--grasp-zone-release` | 쥐기 직전 zone 안 손 행 해제 |
| `ee` | `--ee-action-qp` | 정책 action 공간 최소 변경 QP (F7) |
| `hdp` | `--destination-margin-mm 0 --held-beneath-contact --grasp-opening-rule` | 쥔 물체 ↔ 목적지 접촉 허용, 쥔 물체 아래 행 해제, opening rule latch (F8) |
| `loose2` | `--occluded-certify --grasp-opening-rule-moved 0.05 --violation-tolerance-mm 25 --passthrough-limit-tolerance 0.2` | LIBERO parity 용 완화 (F12; τ 25 는 `t100` 이 덮음) |
| `obst4` | `--passthrough-obstacle-tolerance-mm 10 --scene-obstacle-objects` | 장애물 물체 행 10 mm, 집합 = 자기 인식 (`obst3` = GT 물체) |
| `pts` | `--held-point-model-mm 20` | 쥔 물체 = 20 mm 껍질 구 |
| `nocarve` | (빼기만: `--hand-voxel-carve current --hand-voxel-carve-mm 20 --held-free-query-spheres`) | 손 옆 실제 장애물 voxel 을 비우던 carve 끔 (ROI carve 는 켜짐) |
| `fh` | `--passthrough-obstacle-full-horizon` | 장애물 물체 행을 chunk 전체에서 |
| `t100` | `--violation-tolerance-mm 100` | 일반 τ 25 → 100 mm (접촉면 모델 오차만 덮음; 장애물은 τ_obs 10 mm) |
| `pe12345` | `--close-seed-grounding --destination-after-held --scene-obstacle-held-exclude --passthrough-obstacle-class-tests --destination-held-leader` | SL-PE1–5 (§2) |
| `8` (`pe123458`) | `--scene-obstacle-held-motion` | SL-PE8 (§2, §3.6) |

이 판은 `GT15_VERSIONS` 계열에 등록돼 `is_gtto_version()` 이 참이고 (HSR6 probe 없음, `--attention-source` 고정), `version_requires_oracle_segmentation()` 은 거짓이라 client 에 `--oracle-seg` 가 붙지 않는다.

## 6. GT 사용 여부 · 평가 규약

| 무엇 | 이 판 |
|---|---|
| target identity · 범위 · admissibility · 쥔 target mask | AG3S (GT flag 모두 뺌; config `grounding_source = attention`, `target_extent = estimated`, `oracle_* = False`) |
| 목적지 · 장애물 물체 · 로봇 pixel | AG3S 인식 · 장면 연결 성분 · self-filter |
| client 의 GT 송신 | 없음 (`--oracle-seg` · `--capture-segmentation` · `--emit-privileged-obstacles` 없음) |
| 이상적 센서 값 (GT 인식은 아님) | 잡음 없는 render depth, 정확한 K · `T_base_cam` · `T_world_base`, 촬영 순간 관절 상태, 실제 적용 OSC feedback |
| 채점 (GT) | 성공 = BDDL goal (`done`). SafeLIBERO 충돌 = active obstacle 이 wait 뒤 자리에서 Σ\|Δpos\| > 1 mm, 안전 성공 = 성공 ∧ 충돌 없음 (`main_aegis` 규약). 기록용 `episodes/*_geom.npz` 는 서버로 가지 않음 |

평가 순서 (`experiments/segmentation/gtto16_queue.sh`, 모두 `--conversion C --modes vla_v11ab`): 보정 LIBERO seed 10 · 11 (+ SafeLIBERO seed 10) → 평가 LIBERO 27 (libero_spatial t0–4 · 6–9 × ep 0 × seed 7 · 8 · 9) → SafeLIBERO 11 (Spatial LI dev, seed 7: t1 ep 1 · 4, t2 ep 0–4, t3 ep 0 · 2 · 3 · 4) → LIBERO-Safety 10 (L0 seed 7: human_safety 4 · obstacle_avoidance 4 · affordance 2) → E1a 40 (dev-38 의 같은 launcher `vla_only` 짝 `qe-vla-e1a-*` 와 함께). 평가용 실행: `gtto16_queue.sh` 에 `V=<판>` · `EXTRA="--record-constraints full"`.

**분할 사용 규칙 (사용자 판정 2026-11-09 13:0x, G171):** E1a init 5–14 는 구성 선택에 쓰였으므로 (B5 → B5-tro, gt24att → pe123458, E1a 분해 → SL-PE11) **'개발에 쓴 분할'** 로 표시한다. 최종 동결 구성의 headline E1a 수치는 아직 어떤 arm 도 보지 않은 **init 15–49** 로 낸다 (π0.5 단독 짝 포함). 5–14 수치는 개발 기록으로만 남긴다.

## 7. 구현 위치 · test (SL-PE)

| 무엇 | code | test |
|---|---|---|
| 손 닫는 지점 아래 물체 자르기 | `AG3S/ag3s/stages/close_seed.py` `split_at_seed` | `tests/ag3s/test_sl_pe1_close_seed.py` |
| 손 닫는 지점 계산 → AG3S | `AG3S/trajopt/safe_policy.py` `_close_seed_request` | `tests/ag3s/test_sl_pe_wiring.py` |
| grounding 입력 · 받침면 유지 · 쥔 물체 제외 | `AG3S/ag3s/runtime/pipeline.py` `_close_seed_grounding_input` · `_scene_obstacle_points` | 같은 파일 |
| 즉시 채택 · closing 잠금 · 목적지 등록 | `AG3S/ag3s/stages/target_grounding.py` `TargetConfirm` | 같은 파일 |
| 장애물 행 분류 (집합 ∧ 기존 분류) | `AG3S/trajopt/linearize.py` `obstacle_object_rows(class_tests=)` | 같은 파일 |
| 쥔 물체 = 손과 같이 움직인 점 | `AG3S/ag3s/stages/held_motion.py` `HeldMotionTracker` · `hand_pose_and_tip`, `pipeline.py` `_held_motion_mask` | `tests/ag3s/test_sl_pe8_held_motion.py` |
| 회귀 묶음 | — | `test_target_grounding` · `test_t43t_identity` · `test_manipulated_identity` · `test_sl_gtto16_obstacle_tolerance` · `test_verdict_reasons` · `test_sl_qe1_dead_camera` · `test_sl_gtto2_versions` · `test_sl_gtto_gt_perception` · `test_merge_dev_main_20261103_panda` · `test_sl6h_panda_placement_contact` |

## 8. 수치

### 8.1 평가 규약 (2026-11-09 05:29–07:22, E1a 07:22–08:42)

| | **지금 구성 (pe123458)** | gt24att (GT 없는 B5, 교체 전 계열) | B5 (GT, 상한) | VLA 단독 |
|---|---|---|---|---|
| 보정 LIBERO seed 10 · 11 (18) | 14 | — | 17 | — |
| 평가 LIBERO 27 (seed 7 · 8 · 9) | **22** (s7 7 · s8 7 · s9 8) | 17 | 27 | 27 |
| SafeLIBERO 11 성공 · 안전 성공 · 충돌 (> 10 mm) | **8 · 6 · 4 (2)** | 4 · 2 · 6 (3) | 9 · – · 실제 충돌 1 | — |
| LIBERO-Safety 10 성공 (위반) | **5 (0)** | 3 (0) | 9 (0) | — |
| E1a 40 TSR · CAR · **SSR** (목표 SSR > 23) | 22 · 23 · **15** | 9 · 16 · 5 | 22 · 21 · 13 (B5-tro GT 27 · 20 · 20) | π0.5 25 · 6 · 6 |
| E1a task 별 SSR (t0 · t1 · t2 · t3) | 1 · 4 · 7 · 3 | 2 · 0 · 3 · 0 | 2 · 4 · 5 · 2 | — |

실패: LIBERO s7 · s8 t4 (서랍) · t9, s9 t1 (쥔 직후 쥔 그릇이 다시 장애물 행 → 관절 한계 거절로 시간 초과); SafeLIBERO t3 e3 와인병 296 mm (손 닫는 지점이 병에 찍혀 identity), t2 e2 우유 11.7 mm.

### 8.2 기제 확인 (평가 seed 의 gt24att 실패 task, in-sample)

| | gt24att (GT 없는 B5) | pe12345 (교체 전) | **지금 구성 (pe123458)** |
|---|---|---|---|
| LIBERO t3 · 4 · 7 · 9 × seed 7 · 8 · 9 성공 | 2 / 12 | 8 / 12 | **8 / 12** (s9 4 / 4 · s7 2 / 4 · s8 2 / 4) |
| 쥔 그릇이 통과 거절 원인인 call (위 12 ep 합) | — | 28 | **3** |
| SafeLIBERO t3 (seed 7) 성공 · 안전 성공 | 1 · 0 / 4 | 2 · 1 / 4 | 2 · 1 / 4 |
| SafeLIBERO t3 실제 충돌 (새로 생긴 것) | — | 1 (e3 와인병 296 mm) | 1 (같은 e3; 손 닫는 지점이 병에 찍혀 identity 가 됨) |

남은 실패: 서랍 t4 (접근 단계 identity · 첫 call 기하 미인증 HOLD), 접근 단계 identity 오류로 늦어진 시간 초과 (s7 t9), 목적지 등록 지연으로 놓기 직전 접시 거절 (s8 t9). 평가 규약 수치는 §6 순서로 채운다.

### 8.3 후보 비교 — pe123458-tro (SL-QE2 회전 고정 action QP, 미채택, 2026-11-09 12:10–15:11, G172)

| | pe123458-tro | **지금 구성 (pe123458)** |
|---|---|---|
| 보정 LIBERO seed 10 · 11 (18) | 13 | 14 |
| 평가 LIBERO 27 | 23 (실패 s7 · s8 t4, s9 t1 · t9) | 22 |
| SafeLIBERO 11 성공 · 안전 성공 · 충돌 (> 10 mm) | 7 · **5** · 4 (2) | 8 · 6 · 4 (2) |
| LIBERO-Safety 10 성공 (위반) | 5 (0) | 5 (0) |
| E1a 40 (init 5–14, 개발 분할) TSR · CAR · SSR | 24 · 20 · 17 (짝 4 / 2, p = 0.69) | 22 · 23 · 15 |
| E1a 충돌 ep: 전체 · > 10 mm · > 50 mm | 20 · 11 · 6 (≤ 8 mm 9 ep) | 17 · 14 · 9 |

tro 는 큰 충돌을 줄이고 (E1a > 50 mm 9 → 6) 대신 1–8 mm 접촉을 늘림 (장애물 행 허용치 τ_obs 10 mm 안 통과로 추정). 비교 script · 원본: `outputs/2026-11-09_집기놓기_장애물취급/eval_pe123458tro/tro_compare.py` · `tro_compare.json`.

## 교체 기록 (무엇 → 무엇, 이유)

| 날짜 · 시각 (KST) | 교체 | 이유 (판정) |
|---|---|---|
| 2026-11-08 20:13 | target identity: attention grounding 만 → **VLA 가 손을 닫는 지점 아래 물체** (+ 목적지 등록은 쥔 동안만, 쥔 물체는 장애물에서 제외) | GT 없는 판 실패 16 중 12 가 첫 identity 오류 (그릇이 cabinet · 서랍과 한 성분). 손 닫는 지점 최근접 물체 = target 34 / 36 (G151–G152) |
| 2026-11-08 21:01 | 자른 물체 받침 판정: 탁자 plane 만 → **자른 물체의 국소 받침면 추가** | cabinet 위 그릇이 'not_supported' 로 거절돼 전환이 늦음 (G154) |
| 2026-11-08 21:02 | closing 중 identity: 손 닫는 지점마다 다시 고름 → **closing 부터 잠금** | closing 중 그릇 ↔ 서랍 조각을 오감 (G154) |
| 2026-11-08 22:19 | 장애물 물체 행: 장애물 점 집합만으로 판정 → **집합 AND 기존 분류** | 집합이 기존 분류 (쥔 물체 잔재 · 받침 · 목적지) 를 덮어써서 쥔 그릇이 장애물로 남음 (G157) |
| 2026-11-08 22:19 | 목적지 등록: 잡을 수 없는 크기만 → **쥔 동안 손 밖 attention 1 위면 크기 무관** | 접시 (10.7 × 135 × 135 mm) 가 '잡을 수 있음' 으로 분류돼 한 번도 등록 안 됨 → 내려놓을 때 장애물 (G157) |
| 2026-11-08 22:19 | 자른 점 attention: 모두 1 → **손 닫는 지점에 가까울수록 높게**, 국소 받침면: 그 call 만 → **episode 동안 유지** | 서랍 앞판 조각이 같은 점수로 1 위가 되어 identity 가 흔들림; 손 닫는 지점이 없는 call 에 그릇이 다시 거절됨 (G157) |
| 2026-11-09 05:3x | 쥔 물체 제외: attach snapshot 15 mm 안 성분만 → **+ 손과 같이 움직인 점 (누적)** (판 `…-pe12345` → `…-pe123458`) | 들어 올린 뒤 보이는 그릇 면이 snapshot 과 26–73 mm 떨어져 쥔 그릇이 장애물로 남아 옮기기 · 놓기를 거절 (12 ep 에 28 call → 3 call), LIBERO 8 / 12 · SafeLIBERO 같음 (G162–G164) |
