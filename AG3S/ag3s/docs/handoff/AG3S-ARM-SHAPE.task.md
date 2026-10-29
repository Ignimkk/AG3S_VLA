# AG3S-ARM-SHAPE — 팔 전체를 TrajOpt 충돌 검사에 넣기 위한 팔 형상 (구 모델) 비교 · 채택

> writer: lead · 읽는 쪽: implementer, verifier, scribe
> 작성 2026-10-27 16:50 KST (`date -u` 07:50 UTC + 9 h). 기준 · 지표 고정 시각 = 이 시각 (결과 전).
> RUN = `/mnt/dev/AG3S_VLA-storage/results/development/20261027T165001+0900_b3b69311f948_430762a2a2fc` (이 STEP 의 새 run).
> 원시 출력은 `$RUN/verify/AG3S-ARM-SHAPE/`. outputs 는 checkout `outputs/2026-10-27_팔전체_형상비교_offline/` (1단계) ·
> `outputs/<날짜>_팔전체_closedloop/` (2단계).

## 0. 사용자 판정 (2026-10-27)

- "팔 전체가 필요한지 확인하지 말고, 팔 전체로 우선 해봅시다." → 0단계 (필요성 확인) 없음.
- 팔 형상 방법 둘 (gripper_cover 확장 · cuRobo MorphIt) 을 **둘 다** 비교하고 하나를 채택한다.
- 1단계에 **TrajOpt 가 보는 로봇–장애물 거리의 GT 대비 정확도** 를 넣는다.
- 2단계 (closed-loop) 는 제안 그대로.
- 채택 기준에서 **실행 시간 상한은 뺀다** (보고만).
- 채택은 작업 성공률보다 **이 모듈의 역할을 잘 만족하는지** 를 우선한다.

## 1. 무엇을 정하나 — 세 결정 중 둘

| 결정 | 이번 STEP |
|---|---|
| 1. TrajOpt 충돌 범위 | **팔 전체로 정함** (양팔 `link_*_arm_0` … `link_*_arm_6` + `FT_SENSOR_*`). 손 (손가락 · 손바닥 · 껍질 구) 은 지금 구성 그대로 |
| 2. 팔 형상 (TrajOpt 구 모델) | **이번에 비교 · 채택** |
| 3. ESDF 의 로봇 자기 제거 | `--robot-mask render` 로 **고정** (시뮬레이션 한정, 사용자 2026-10-27 허용) |

- 팔 형상은 ESDF 를 만들지 않는다. TrajOpt 는 구 중심에서 ESDF 를 읽고 반지름을 빼 로봇–장애물 거리를 본다. 이 STEP 은 그 구를 정한다.
- 기준 구성 (= "지금 범위") 은 PRE-CORE-TO 의 껍질 구 + 렌더 mask 구성 (SCMR) 이다: `--links gripper` + `--finger-cover both` +
  `--palm-shell on --palm-shell-margin 5` + `--robot-mask render`, 코드 `snapshot_scpt`
  (`/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581/snapshot_scpt`), 명령줄 =
  `…/verify/AG3S-PRE-CORE-TO/PT/occ/launch_occ_SCMR.sh` 의 `SAFE`. **손목 FT sensor 는 지금 범위에 없다** (`--links gripper`).

## 2. 비교하는 팔 형상

모두 MJCF collision geom (MuJoCo 가 실제 접촉을 계산하는 형상) 기준, 구는 URDF link frame 에 고정. 손 쪽 구성은 바꾸지 않고 팔 구를 더한다.

| 이름 (보고용) | 방법 |
|---|---|
| URDF capsule 구 (대조 1, 지금 serving 굵기) | `UrdfSphereChain`, `--capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32` |
| URDF capsule 구 (대조 2, 원래 굵기) | `UrdfSphereChain`, `--capsule-radius-scale 1.0`, 나머지 같음 |
| **내접 덮개 구** (gripper_cover 확장) | `gripper_cover.py` 의 방식 (MJCF collision mesh 표면에 안쪽에서 접하는 구, set cover) 을 팔 link 에 적용 |
| **MorphIt 구** (cuRobo) | `third_party/curobo/.../sphere_fit/fit_spheres.py` `fit_spheres_to_mesh(fit_type=MORPHIT)` 를 각 link 의 MJCF collision geom 에 적용 |

- **매개변수는 기하 지표로만 정한다** (아래 §3 필수 2 · 3 을 만족하는 가장 적은 구 수). GT 자세 평가를 본 뒤에 매개변수를 바꾸지 않는다.
  바꿔야 하면 사용자에게 먼저 묻는다.
- 팔 구가 권한 있는 손 geom (`EE_BODY_*` · `ee_finger_*`) 안으로 들어가면 안 된다 (target 예외 유지). 두 방법 모두 같은 규칙으로 처리하고
  처리 전후를 보고한다.
- 서버 flag 는 기본 꺼짐. 꺼짐은 `snapshot_scpt` 와 bit 단위로 같아야 한다.

## 3. 1단계 — offline 비교 (기준 고정 16:50 KST)

### 3.1 자세 집합

| 집합 | 내용 | 쓰는 곳 |
|---|---|---|
| P1 기록 자세 | SCMR closed-loop 45 run (가림 121–123 · 회귀 119 · 211 · 250 · 320 · 850 · 954 × 5) 의 요청마다 계획 자세 (`refined_chunk`) 와 실제 자세 (`measured_qpos`) | 거리 정확도 · 형상 오차 · 거리장 오차 |
| P2 grasp 후보 자세 | AG3S-GEO-GRASP 의 후보 IK 자세 (`…/20261026T204126+0900_b3b69311f948_8d603092427b/verify/AG3S-GEO-GRASP/arm/`, 6 가림 장면, grasp · pre-grasp, MuJoCo 실제 침투 기록 있음) | 놓친 침투 · 거짓 충돌 |
| P3 흔들린 기록 자세 | P1 요청 자세에서 팔 관절을 균등 ±0.25 rad 로 흔든 자세, 요청마다 4 개, seed 고정 (0–3) | 놓친 침투 · 거짓 충돌 (link 0–4 의 실제 침투를 얻기 위해) |

### 3.2 GT 와 지표

- **GT 거리** = MuJoCo 의 팔 collision geom 과 로봇이 아닌 모든 collision geom (장애물 · 탁자 · 바구니 · 과일 · target) 사이 부호 거리
  (`mj_geomDistance`, 음수 = 침투). 팔 link 는 권한이 없으므로 target 도 장애물이다.
- **형상만의 거리** = 구 중심에서 GT 장애물 형상까지 정확한 거리 − 반지름 의 최소 (장애물 쪽은 정확, 로봇 쪽만 구).
- **TrajOpt 가 보는 거리** = 구 중심에서 그 요청의 **기록된 서버 ESDF** (팔 행이 실제로 읽는 거리장, `fixed_input_ablation.build_field`) − 반지름 의 최소.
- 분해: 형상 오차 = 형상만의 거리 − GT 거리 · 거리장 오차 = TrajOpt 가 보는 거리 − 형상만의 거리 (같은 구 중심) · 전체 오차 = 둘의 합.
  link 마다, GT 구간 ≤ 20 mm · 20–50 mm · > 50 mm 로 나눠 bias · |오차| 중앙 · p95 · max 를 낸다.
  부호를 따로 센다: **실제보다 멀다고 봄** (> 0, 접촉을 놓치는 쪽) · **실제보다 가깝다고 봄** (< 0, 거짓 충돌 · 불필요한 개입 쪽).
- **놓친 실제 침투** = GT < 0 인데 형상만의 거리 ≥ 0 인 (자세, link). **거짓 충돌** = 형상만의 거리 < 0 인데 GT ≥ 0. (P2 · P3)
- **기하 지표** (자세 무관): link 마다 MJCF collision mesh 표면 표본이 구 합집합 밖에 남는 거리 (max · p95), 구가 mesh 밖으로 나간 거리 (max · p95),
  팔 구와 권한 손 geom 의 겹침, 구 수.
- 보고만: 구 수, TrajOpt 1회 linearize 시간.

### 3.3 채택 기준 (순서대로)

| 순서 | 기준 | 역할 |
|---|---|---|
| 필수 1 | P2 · P3 에서 놓친 실제 침투 0 | 비타겟 장애물에 닿지 않음 |
| 필수 2 | 모든 팔 link 에서 mesh 표면이 구 밖에 남는 거리 max ≤ 5 mm | 비타겟 장애물에 닿지 않음 |
| 필수 3 | 팔 구가 권한 손 geom 을 침범하지 않음 (겹침 0) | target 예외 유지 |
| 고르기 1 | P1 · P3 에서 GT ≤ 50 mm 구간의 **형상 오차** \|오차\| p95 가 작은 쪽 | 닿지 않음 · 최소 수정 |
| 고르기 2 | 같으면 (차 < 1 mm) P2 · P3 의 거짓 충돌이 적은 쪽 | 최소 수정 |

- 거리장 오차는 두 방법에 공통이지만 크기를 보고한다 (구 중심 위치가 달라 값은 조금 다를 수 있다).
- 두 방법 모두 필수를 못 넘으면 채택하지 않고 사용자에게 묻는다. 대조 (URDF capsule) 는 채택 후보가 아니다.

## 4. 2단계 — closed-loop (1단계 채택 형상)

- 장면 · 반복: SCMR 과 같은 9 장면 × 5 번 (흔들림 없음 + 첫 chunk 1e-6 rad 흔들림 seed 1–4), `--stop-after-place 2.5`, 독점 GPU lock.
- 구성: **지금 범위** = SCMR 기록 45 run 을 다시 쓴다. 단 새 코드의 flag 꺼짐 구성으로 2 run (가림 121 seed 0 · 회귀 250 seed 0) 을 돌려
  실행 궤적이 SCMR 과 bit 단위로 같을 때만. 다르면 지금 범위도 새로 45 run.
  **팔 전체** = SCMR 명령줄 + 팔 형상 flag.
- 지표 (결과 전 고정):

| 순서 | 지표 | 역할 |
|---|---|---|
| 주 1 | 금지 접촉 run 수 (target 이 아닌 물체 · 권한 없는 link, 실제 MuJoCo 접촉, 접촉 body 별) | 닿지 않음 |
| 주 2 | 불필요한 개입: 그 요청의 VLA chunk (`reference_chunk`) 계획 자세의 GT 최소 거리 (로봇 collision geom 전체 대 target 이 아닌 geom) ≥ 20 mm 인데 TrajOpt 가 고쳤거나 (행 0–7 관절 차 > 0.5°) HOLD 한 요청의 비율 · 그 요청의 수정량 | 최소 수정 |
| 주 3 | 금지 접촉 없는 성공 | target 예외 유지 (쥐기) |
| 보고 | 테이블 · 바구니 접촉, 서버 처리 시간 · 533 ms 넘은 요청 | — |

- 효과 판정: 장면 하나에서 성공 · 금지 접촉 수의 차 ≥ 4 / 5 일 때만 구성 효과. HOLD 증가를 보호로 세지 않는다.
- 실패마다 원인 위치 (ATS · HSR · LTE · CCR) 를 적는다.

## 5. 역할 분담

| 일 | 누가 | 산출물 |
|---|---|---|
| 팔 형상 두 방법 구현 + 서버 flag + test + 기하 지표 + 매개변수 결정 | implementer (`$RUN/snapshot_arm` = `snapshot_scpt` 사본에서) | `AG3S-ARM-SHAPE.impl.md`, 팔 구 파일 2 개 |
| GT 평가 도구 (P1–P3, §3.2) + 대조 형상 측정 | verifier | `$RUN/verify/AG3S-ARM-SHAPE/eval/`, `AG3S-ARM-SHAPE.verify.json` |
| 1단계 판정 · outputs · 2단계 실행 | lead | outputs 폴더 |

## 6. 진행 기록

| KST | 내용 | 누가 |
|---|---|---|
| 2026-10-27 16:50 | task 작성, 기준 · 지표 고정 (사용자 판정 위 §0) | lead |
| 2026-10-27 17:15 · 17:30 | 측정 도구 보정 두 건 (방법 결과 전, verify.json `definitions.changes_after_fixing`): MuJoCo 3.11 `mj_geomDistance` 가 mesh–box 에서 떨어져 있어도 0.0 을 돌려주는 경우를 separating-axis 값으로 대체 · 형상만의 거리용 구 중심을 실제 base pose 로 보정 (MuJoCo base 가 최대 약 1 mm 움직임). 기준 · 지표 정의는 그대로 | verifier |
| 2026-10-27 (verifier 끝) | GT 평가 도구 `$RUN/verify/AG3S-ARM-SHAPE/eval/eval.py`, URDF capsule 대조 두 가지 측정 완료 (`AG3S-ARM-SHAPE.verify.json`). 자세 P1 계획 10,835 · 실제 17,223 · P2 376 · P3 8,668. 대조 둘 다 FT sensor 에 구가 없음 (URDF collision 없음) | verifier |
| 2026-10-27 18:15 | 내접 덮개 파일 (`$RUN/arm_spheres/inscribed.json`, 팔 구 2,607, 16 link 기하 기준 통과) · 서버 flag `--arm-spheres` · test 27 passed (`AG3S-ARM-SHAPE.impl.md`). MorphIt 는 CUDA 전용이라 GPU lock 대기 중 세션 종료로 멈춤 | implementer |
| 2026-10-27 21:52 | 세션 재시작 후 lead 가 이어받음: MorphIt fit + GPU 초기값 검사 (`fit/run_gpu.sh`, 공유 lock) 와 내접 덮개 GT 평가 재개 (평가 28 / 51 파일에서 멈췄던 것, 끝난 파일은 건너뜀) | lead |
| 2026-10-27 22:00 | 내접 덮개 평가 끝. grasp 후보 + 흔든 자세의 실제 침투 289 중 놓침 31 (모두 깊이 ≤ 3.8 mm, FT 20 · link 6 9 · link 5 2), 거짓 충돌 0, 형상 오차 p95 3.1 mm (≤ 50 mm). **필수 1 미통과**, 필수 2 · 3 통과. 거리장 오차가 큰 경우 (p95 133 mm) 는 모두 쥔 target 이 가장 가까운 경우 (쥔 사과는 ESDF 에서 지워지는 설계) — 정의 그대로 두고 따로 나눠 보고 (`compare/field_by_state_*.json`). MorphIt 는 GPU lock 대기 (geo-grasp-pilot4 독점). outputs 진행 중판 `outputs/2026-10-27_팔전체_형상비교_offline/` | lead |
| 2026-10-27 23:40 | MorphIt fit (기본 가중치 · 200 iteration · 구 수 8–512, GPU 공유 lock) — **16 link 모두 기하 기준 미통과** (가장 작은 구멍 5.7–16.2 mm, 밖으로 나감 7–27 mm, `fit/morphit_fit.log`, 기준 느슨하게 하지 않음). 비교용으로 link 별 최선 476 구 (`arm_spheres/morphit.FAILED.json`) 를 평가: 놓침 77 (max 17.4 mm, 5 mm 넘게 38, 모두 FT sensor), 거짓 18, 형상 오차 p95 17.6 mm. GPU FK 검사 (내접 덮개 3,213 구): 중심 차 0.0001 mm, 0.08 GiB, robot 설정 빌드 47.6 s. 결과 후 추가 분석 (채택 기준 아님, `compare/margin_4.json`): TrajOpt 가 팔 행에 지키는 비타겟 여유 10 mm 기준으로 내접 덮개 놓침 0 (형상 · TrajOpt 거리 모두), MorphIt 29 · 15. **두 방법 모두 필수 미통과 → §3.3 규칙대로 사용자 확인 대기.** outputs 최종판 `outputs/2026-10-27_팔전체_형상비교_offline/`, 그림 `figures/arm-shape-4/` | lead |
| 2026-10-27 23:55 | **사용자 판정:** 팔 형상 = **내접 덮개 (넘침 0, `arm_spheres/inscribed.json`)** 채택. 필수 1 은 "TrajOpt 가 팔 행에 지키는 10 mm 비타겟 여유 아래에서 놓침 0" 으로 읽었다 (결과 후 해석 — 고정 정의 (형상 거리 ≥ 0) 로는 31 건 미통과). 손바닥 껍질 구처럼 5 mm 넘치게 하는 안 (결과 후 추정: 놓침 0 · 거짓 43, 10 mm 여유에서 불필요하게 밀어내는 (자세, link) 59) 은 고르지 않음. 팔 행 target 최근접 여유 50 mm · 쥔 뒤 `link_*_arm_6` 권한 행 (`held_permission_parent_links = 1`) 은 기존 규칙 그대로 (사용자). → 2단계 시작 | 사용자 · lead |
| 2026-10-28 00:35 | 2단계 kit (`$RUN/verify/AG3S-ARM-SHAPE/CL/`, `make_kits.sh`): SCMR kit 복사 + snapshot_arm + B_ROI_FPR / SAFE 끝에 `--arm-spheres` (OFF kit 은 없음). 서버 argv: 팔 전체 = flag 꺼짐 + `--arm-spheres` 만 (`argv_check/`). code md5 차이 = 팔 형상 구현 7 파일만. `launch.sh` = A (flag 꺼짐 2 run: 가림 121 · 회귀 250 흔들림 없음) → B (팔 전체 45 run), 독점 lock 대기 (pre-core-to-e3g 뒤). 주 2 도구 `CL/intervention.py`: 요청마다 reference_chunk 행 0–15 의 GT 최소 거리 (움직이는 양팔 subtree geom 대 target 아닌 geom, 바닥 평면 제외 — 바퀴가 늘 닿아서; 두 구성 같은 정의), 개입 = 행 0–7 팔 관절 차 > 0.5° 또는 verdict hold. SCMR 45 run: 20 mm 이상 요청 1,536 중 개입 1,034 (67.3 %; hold 20 · 고침 1,014), 고친 양 중앙 1.28° · p95 3.44° (`CL/intervention_SCMR.json`) | lead |
| 2026-10-28 02:50 | lock 획득 02:43. A 의 가림 run (121 흔들림 없음, flag 꺼짐) = SCMR k0 와 `traj.npz` 의 모든 배열 bit 동일 (executed_actions · measured_qpos · predicted_chunks · inference_states 등), 다른 것은 벽시계 `inference_ms` 뿐. A 의 회귀 run (250) 은 돌지 않음: `edit_env.py` 가 kit env.sh 의 `V` 를 SCMR kit 경로로 남겨 lists 를 못 찾음 → 6 kit `V` 를 `CL/REG_<구성>_k<k>` 로 고침 (`edit_env.py` 에도 반영, launch.sh 는 실행 중이라 그대로). B 회귀 loop 은 env.sh 를 매번 새로 읽으므로 고친 경로로 돈다. 회귀 identity run 은 B 끝난 뒤 `run_regoff.sh` (독점 lock) 로. B 서버 log: 구 606 → 3,213 (inscribed 2,607, sha256 07841afc) 확인 | lead |
| 2026-10-28 03:50 | B 가림 15 run 정상 (rc 0). B 회귀는 실패: lead 가 가림 집계 (`agg4.py` 가 OCC 폴더 옆 `placement4` 를 읽음) 를 위해 03:28 에 만든 `CL/OCC_*/placement4` symlink 가 RUN 밖 (SCMR 기록) 을 가리켜, 그 뒤 시작한 client · server 가 모두 출력 경로 검사 ("Output must be inside RUN") 로 거부됨 — 회귀 k0 의 ep119 · ep211 만 완료. symlink → 실제 복사로 바꿈 (RUN 안 바깥 link 없음 확인). `relaunch_reg.sh` (독점 lock, 다른 세션 pilot7 뒤 대기): 회귀 팔 전체 k0–4 (`.done` 있는 run 은 건너뜀) → flag 꺼짐 회귀 identity run (250). 회귀 identity 대기 명령은 `pgrep -f` 가 자기 자신을 찾아 시작하지 못했음 (멈춤). 참고: 다시 돈 run 은 서버 안 순서가 SCMR 과 다름 (k0 8292 서버의 첫 run 이 320). 가림 15 run 결과 (`CL/agg/occ_*.json`): 접촉 없는 성공 팔 전체 12/15 · SCMR 10/15, 금지 접촉 0 · 1; 장면별 4/5 이상 차이 없음. flag 꺼짐 가림 run 은 집계도 SCMR 과 같음, 팔 전체 같은 draw 0 은 놓침 (slip). 불필요 개입 (가림만): 71.8 % · SCMR 71.0 % | lead |
| 2026-10-28 08:45 | 2단계 끝. 회귀 30 run 재실행 06:55–08:27 (`relaunch_reg.sh`), flag 꺼짐 회귀 identity run (250 회차 0) 도 SCMR 과 bit 동일 → SCMR 45 run 을 기준으로 씀. 회귀 집계는 kit 복사 때 빠진 `agg/address_check.json` (SCMR kit 과 같은 파일, model XML md5 같음) 을 넣고 다시 돌림. 결과 (`CL/figs_cl.py` → `AG3S/ag3s/docs/figures/arm-shape-cl/`, outputs `2026-10-28_팔전체내접덮개_45run_closedloop`): 금지 접촉 없는 성공 23/45 대 SCMR 24/45 (가림 12/15 · 10/15, 회귀 11/30 · 14/30), 금지 접촉 run 9 · 8, 불필요 개입 71.8 % · 67.3 % (HOLD 44 · 20), TrajOpt 중앙 246 · 70 ms, 533 ms 초과 요청 714 · 119. 효과 기준 (장면 차 ≥ 4) 넘은 장면: 회귀 250 (5 → 1, 나쁜 쪽). 250 회차 0: 팔 구는 70 mm 이상 떨어져 있었는데 TO 해가 3 번째 요청부터 달라지고, VLA 닫기가 step 93 → 198 로 늦어져 손가락이 사과에 밀린 뒤 운반 중 떨어뜨림 — 원인 위치 CCR, 멀리 있는 팔 구가 해를 바꾸는 이유는 미확인. 사용자 판정 대기 | lead |
| 2026-10-28 09:05 | 사용자 판정: 팔 덮개는 꺼 둔 채 원인 확인. 회귀 250 회차 0 진단 (`CL/diag/`, outputs `2026-10-28_팔덮개_회귀250_원인진단`): 요청 1 까지 입력 bit 동일 · TO 결과 2.98e-8 rad 차이. 팔 행은 75 요청 어디서도 QP 24 행에 들지 않음 (밀려난 손 행 0, 팔 행 최소 여유 36.8 mm). multi-seed 18 회는 모두 손가락 행 위반, 늦게 닫은 뒤의 결과. 같은 입력 offline 재풀이: 이전 구성 기록 29 요청은 서버와 1° 안, 팔 덮개 기록은 손이 사과에 밀린 채 닫지 않는 구간에서 팔 구 있음 · 없음 4.2–22.7° (요청 10 · 12 · 15 · 17 · 39), 팔 구 그대로 재풀이도 서버와 1° 넘는 요청 19 개 → 그 상태에서 TO 해가 불안정 (원인 위치 CCR). 팔 덮개 회차 4/5 가 늦게 닫는 체계적 경로는 못 찾음 | lead |
| 2026-10-28 22:05 | **사용자 최종 판정 (AG3S-PRE-CORE-TO 세션에서):** 렌더 로봇 mask 는 끈다 (핵심 평가 포함). §1 결정 3 의 "`--robot-mask render` 로 고정" 은 이 STEP 의 비교 조건으로만 남고 serving 기본이 아니다. 이 STEP 의 closed-loop 은 렌더를 켠 구성끼리의 짝 비교다 — 렌더를 끈 구성에 팔 덮개를 더한 closed-loop 은 없다. 근거 `doc/active/reviews/AG3S_decision/ADOPTED_KO.md` A7 · `NOT_ADOPTED_KO.md` N12 | lead (PRE-CORE-TO) |
