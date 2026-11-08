# AG3S Benchmark — V11AB Panda / LIBERO 계열 평가 기록

> **이 파일의 용도.** `feature/panda-benchmark` 에서 수행하는 **V11AB(paired v2) Panda benchmark 평가**의
> 구현과 검증 기록이다. 단계는 SL-0 부터 이어 쓴다. 2026-10-12 사용자 지시로 Panda/LIBERO 계열 기록은
> 이 파일이 정본이다.
>
> RB-Y1 16D 모델의 T0→T6 기록은 [`AG3S_T0T6_LOG.md`](AG3S_T0T6_LOG.md) 가 그대로 맡는다. 두 파일은
> 대상 로봇·환경·checkpoint 가 달라 서로의 숫자를 대신하지 않는다.
>
> **계획 정본** — [`doc/develope_PLAN_SUMMARY.md`](../../../doc/develope_PLAN_SUMMARY.md)
> (§0 한눈에 · §2.1 SL-0~SL-8 · §7 Phase 순서와 통과 기준).
>
> **수치 규칙.** 이 파일의 모든 수치는 `AG3S/ag3s/docs/handoff/<STEP>.verify.json` 의 `numbers` 를 출처로 한다.
> 출처가 없는 숫자는 쓰지 않고, 필요하면 verifier 에게 측정을 요청한다. 식별자(경로 · SHA · commit)는
> verify.json 또는 impl.md 에 있는 것만 옮긴다.

---

## 진행 현황

실행 경로는 plan §0 의 다섯 단계다: **Panda adapter → LIBERO Spatial 소규모 → SafeLIBERO Spatial I/II pilot →
전체 SafeLIBERO → LIBERO-Safety**.

| Phase (plan §7) | 범위 | 상태 | 비고 |
|---|---|---|---|
| **P1** Panda adapter | SL-0 pi0.5-only 1 episode, 독립 환경 · checkpoint · run 기록 | **완료 (2026-10-12, 사용자 확정)** | 아래 [SL0](#sl0--pi05-only-libero-spatial-1-episode-2026-10-12) |
| | SL-1 agent-view · wrist RGB/depth, intrinsics/extrinsics → AG3S 관측 형식 | **완료 (2026-10-12, 사용자 확정)** | 아래 [SL1](#sl1--libero-panda-camera-adapter-rgbdepth--k--t_base_cam--v11ab-wire-관측-2026-10-12). ~~판정 2건(K principal point 127.5, `robot_state` image 순간)은 SL2 에서 구현 예정~~ → SL2 에서 구현·측정 완료 |
| | SL-2 Panda geometry / self-filter | **완료 (2026-10-12, 사용자 확정)** | 아래 [SL2](#sl2--panda-geometry-sphere-chain--self-filter--contact-link-sl1-후속-2건-2026-10-12). self-filter 기본 = confined hand+mount cover, constraint model = `franka.yml` sphere 그대로 |
| | SL-3 attention read-out audit | **완료 (2026-10-12, 사용자 확정)** | 아래 [SL3](#sl3--v11ab-attention-read-out-audit-on-pi05_libero--sl2-후속-2건-2026-10-12). SL2 후속 2건(confined cover 기본값, `robot_geometry.py` CLI) 구현·확인 완료. V11AB cell L8H2 를 Panda 에 그대로 쓴다(recalibration 없음) |
| | SL-4 Panda 를 V11AB core 에 연결 (ATS / LTE / ESDF), offline | **완료 (2026-10-13, 사용자 확정)** | 아래 [SL4](#sl4--panda-관측을-v11ab-perception--esdf-에-연결-ats--lte--esdf-offline-2026-10-12). core diff 0 줄, 기준선 둘 다 일치. 접근 단계 ATS 는 대체로 target 을 잡았으나(101/105 call) **t4 · t5 · t9 에서 다른 물체를 target 으로 골랐다**. ~~SL3 열린 항목(head block mass, wrist mirror 비교 원인, core 주입 지점)~~ → SL4 에서 측정(head seed 기여, wrist mirror 질량 분해, offline 주입 profile). online 주입은 SL-5 |
| | SL-4b ATS 용 attention cell dev calibration 연구 | **완료 (2026-10-13, 사용자 확정)** | 아래 [SL4b](#sl4b--ats-용-attention-cell-dev-calibration-2026-10-13). ~~진행 중 (사전 등록)~~ → 사전 등록 규칙대로 **V11AB L8H2 유지**(최고 후보 L12H3 가 +1.5 %p, 기준 +10 %p 미달). oracle 도 t4 · t9 에서 0 이어서 오선택은 attention cell 이 아니라 그 뒤 grounding 단계 문제로 남는다. 같은 STEP 에서 LIBERO fixture reset 이력 문제를 찾아 export · replay 를 고쳤다. test 데이터는 쓰지 않았다 |
| | SL-4c ATS grounding 탈락 원인 진단 (t4 · t5 · t9) | ~~**진행 중 (2026-10-13)**~~ → **측정 완료 (verify 있음), 판정 기록 없음** | [`handoff/SL4c.task.md`](handoff/SL4c.task.md) (commit `cf4649ce`). **진단만** 한다. code 는 `85e6d9d1` worktree 에 고정. cluster 병합 · Panda 8 cm 개도 한계는 **가설**이며 측정이 확인 또는 반박한다 |
| | SL-5a Panda 주입(최소 core 변경) + EE delta ↔ joint 변환, offline 검증 | **구현 완료 (2026-10-13, commit `76ee59ba`), 2026-10-14 보정 추가** | [`handoff/SL5a.task.md`](handoff/SL5a.task.md). RB-Y1 golden bit 동일 · 무개입 chunk bit 동일 유지. 2026-10-14: gripper `empty_below` Panda 보정, 역변환 A/B 명시 모드 (아래 [SL5b-smoke](#sl5b-smoke--t0-init0-paired-smoke-재확인-conversion-a--b-2026-10-14)) |
| | SL-5b live closed-loop 1 episode | **연결 확인 완료 (2026-10-14)**, 한계 기록 | 아래 [SL5b-smoke](#sl5b-smoke--t0-init0-paired-smoke-재확인-conversion-a--b-2026-10-14) · [SL5b-feedback](#sl5b-feedback--exec_feedback-를-실제-실행-기준으로-확정-t0-재측정-2026-10-14) · [SL5b-libero10](#sl5b-libero10--libero-task-09--init-0-paired-2026-10-14). closed loop · 피드백 계약 · 무개입 동일성은 10 episode 전부 성립. V11AB-B 3/10, A 2/10 (VLA-only 9/10). 실패 7 건 중 6 건은 latch `reach`(occluded target) 로 attach 되지 않은 경우 — development 로 전달 |
| | SL-6 phase / gate source | 미착수 | Panda main 의 phase source = gripper-state strong rule 또는 pi05_libero feature 로 dev 에서 새로 학습한 probe(GT phase 는 oracle 비교용). **SL-6 에서 결정** |
| | SL-7 ~ SL-8 (contact evaluator, runner) | 미착수 | P1 통과 기준에는 pi0.5-only 뿐 아니라 V11AB 최소 1 episode 도 포함. ~~SL-3 은 attention → depth 사상에서 policy image 의 좌우 반전을 되돌려야 한다(SL1)~~ → SL3 에서 un-mirror mapping 구현·측정 완료(image 검사 221/221 call), **SL4 에서 ATS 결과로도 확인**(mirror 로 돌린 접근 단계 centroid 오차 median 236.2 mm 대 12.0 mm). 이후 단계는 이 mapping 을 그대로 쓴다 |
| **P2** LIBERO Spatial 소규모 평가 | 고정 task/seed paired 비교 | **소규모 확인 (2026-10-14)** | task 0~9 × init 0 (dev) paired. 통합 오류 없음, 뚜렷한 저하(9/10 → 3/10) 확인. 비저하 증명이 아니다 (사용자 판정 2026-10-14) |
| **P3** SafeLIBERO Spatial Level I/II pilot | contact 판정 검증 + paired 결과 | **진행 중 (2026-10-14)** | 아래 [SL5c](#sl5c--safelibero-spatial-dev-연결과-v11ab-기준선-2026-10-14). 환경 연결 · 충돌 판정 검증 완료. V11AB 기준선(B): Level I safe success 0/20 (VLA-only 2/20). Level II V11AB-B 는 미실행(보류). 2026-10-23 까지 Level I dev 에서 개선판을 시험했다(minimal intervention gate 판부터 파지 인식 + closing 면제판까지, 역변환 A · B 비교 포함). 최고 safe success 3/20(파지 인식 + closing 면제판) vs VLA-only 2/20 (같은 3/20 이 V11AB-A 기준선과 grasp zone 허용판 + 역변환 A 에도 있다). official success 는 3/20 vs VLA-only 11/20. 아래 [SL5i](#sl5i--파지-인식-수정과-closing-중-쥔-물체-면제-2026-10-22~23) |
| **P4** 전체 SafeLIBERO | official I/II 전체 | 미착수 | |
| **P5** LIBERO-Safety 추가 평가 | 독립 protocol · adapter | 미착수 | |

---

## 용어

기록에 쓰는 말을 여기 모은다. **새 용어를 쓸 때는 여기 추가한다.**

| 용어 | 뜻 |
|---|---|
| **V11AB** | 평가 대상으로 고정한 method. 원본 benchmark v2 와 pi05_TO_hybrid v2 를 통합한 paired-v2 기반 설정(plan §0.1). 개선 branch 의 후속 algorithm 은 섞지 않는다 |
| **pi0.5-only** | VLA(pi0.5)만 쓰고 AG3S safety layer 를 붙이지 않은 조건. V11AB 와 비교할 **기준 경로** |
| **frozen pi0.5 LIBERO checkpoint** | 이번 평가에서 다시 학습하지 않는 `pi05_libero` weight. RB-Y1 checkpoint 를 Panda 에 쓰지 않는다 |
| **official success** | 공식 LIBERO 예제와 같은 판정: `env.step()` 이 돌려주는 `done` 이 True 가 되면 성공. AG3S 가 따로 정의한 metric 과 구분한다 |
| **status** | episode 종료 사유. `success`(done) · `max_steps`(상한까지 done 없음) · `execution_error`(예외) 셋 중 하나 |
| **execution_error** | client 가 예외를 잡고 episode 를 끝낸 경우. 성공·실패로 조용히 섞지 않고 별도 status 와 traceback 으로 기록한다. 이후 단계의 오류 · UNKNOWN 집계가 이 구분에 기대므로 처음부터 분리했다 |
| **action chunk / replan** | policy 가 한 번의 호출로 돌려주는 연속 action 묶음. 이번 protocol 에서는 chunk 전체가 아니라 앞부분 `replan_steps` 개만 실행하고 다시 policy 를 호출한다 |
| **wait steps / dummy action** | episode 시작 직후 물체가 안정되도록 policy 없이 넣는 10 step. 고정 action `[0]*6 + [-1]` (gripper open) |
| **init state** | LIBERO 가 task 마다 제공하는 초기 물리 상태 목록. episode 번호가 그 인덱스를 고른다 |
| **OSC_POSE** | LIBERO 의 EE delta controller. action 앞 6 차원이 EE 위치·회전 delta, 7번째가 gripper 다 |
| **run dir / run manifest** | 실행마다 `results/benchmark/<시각_commit_run-id>/` 에 새로 만드는 고유 디렉터리와, 그 안의 `run_manifest.json`(설정 · 환경 · 실행 argv · checkpoint 식별). 기존 기록을 덮어쓰지 않는다 |
| **norm stats** | policy 입력/출력 정규화 통계(`norm_stats.json`). checkpoint 와 짝이 맞아야 하므로 SHA-256 을 함께 고정한다 |
| **aggregate SHA** | `CHECKSUMS.json` 의 파일별 SHA-256 들을 하나로 묶은 값. checkpoint 전체의 동일성 확인에 쓴다 |
| **JIT first call** | JAX 가 첫 호출에서 model 을 compile 하는 시간. steady-state latency 가 아니므로 나머지 호출과 분리해 보고한다 |
| **policy_records** | OpenPI server 가 호출마다 저장하는 입출력 기록(`policy_records/step_*.npy`) |
| **osmesa / EGL** | MuJoCo offscreen 렌더러. 공식 Docker 기본은 GPU EGL, 이번 실행은 CPU software 렌더링(osmesa) |
| **pod working set** | pod 의 `memory.current` 에서 inactive file cache 를 뺀 값. pod 메모리 압박의 실제 지표 |
| **paired 비교** | 같은 task · init state · seed · checkpoint · controller · horizon 에서 조건만 바꿔 비교하는 방식 |
| **camera adapter** | LIBERO/robosuite 관측(OpenGL row order RGB · normalized depth · camera pose)을 V11AB 가 읽는 형식(OpenCV row order · metric z-depth · K · `T_base_cam`)으로 바꾸는 Python 3.8 module (`benchmark/adapters/libero_panda/`) |
| **wire replica** | AG3S 를 import 할 수 없는 Python 3.8 쪽에서 `AG3S.trajopt.wire` 의 camera 직렬화(key · dtype)를 그대로 다시 구현한 것. 3.11 쪽 `unpack_camera_observations` 로 왕복 test 해 동치를 보장한다 |
| **K / `T_base_cam`** | K = pinhole intrinsics(3×3, fx fy cx cy). `T_base_cam` = camera 좌표(OpenCV 축)를 robot base frame 으로 옮기는 4×4. base frame 은 MuJoCo body `robot0_base` |
| **OpenCV / OpenGL row order** | OpenGL render 는 image 의 첫 행이 아래쪽(bottom-up)이고, OpenCV 는 첫 행이 위쪽(top-down, +y 가 아래, +z 가 앞). adapter 는 raw 를 `[::-1]` 로 OpenCV 순서로 바꾼다 |
| **z-depth** | camera 광축(+z) 방향 거리. ray 를 따라간 거리(range)가 아니다. robosuite 는 `near / (1 - d (1 - near/far))` 로 normalized depth 를 metric z-depth 로 바꾼다 |
| **back-project** | pixel (u, v)과 depth 로 3D 점을 만드는 연산. K 의 역 + `T_base_cam` |
| **principal point (cx, cy)** | 광축이 image 와 만나는 pixel 좌표. pixel 중심이 정수 index 인 좌표계에서 W = 256 이면 정중앙은 127.5 이고, robosuite 의 `W/2 = 128` 은 0.5 pixel 어긋난다 |
| **pixel centre / MSAA sample position** | RGB · segmentation 은 pixel 중심에서 sample 된다. 이 renderer(osmesa llvmpipe, MSAA `offsamples` 4)에서 depth 는 pixel 안 `(u+0.375, v+0.875)` 에서 sample 된다. renderer 에 따라 달라질 수 있다 |
| **physics substep / image instant** | MuJoCo 가 `env.step` 하나 안에서 여러 번 도는 시간 단위(0.002 s). robosuite 는 camera render 를 substep loop 안에서 하므로 `env.step` 이 돌려준 joint 값은 image 보다 한 substep 뒤다. image 가 찍힌 순간을 image instant 라 부른다 |
| **segmentation** | pixel 마다 그 pixel 에 보이는 MuJoCo geom id 를 담은 image. 이 STEP 에서는 depth 정합의 정답(어느 물체인가)으로만 쓴다 |
| **action replay** | 기존 run 의 기록 action 을 policy 없이 같은 task · init state · seed 로 다시 실행해 새 run 에 추가 정보(geometry)를 기록하는 방식. 원 기록과 byte 단위로 같은지 확인할 수 있다 |
| **left-right mirror (policy image)** | pi0.5 가 받는 image 는 OpenCV image 의 열만 뒤집은 것(`policy256 == opencv[:, ::-1]`)이다. 180° 회전이 아니다 |
| **co-visible** | 두 camera 가 모두 보고 있는 영역. cross-camera 일관성(#4)은 이 영역에서만 의미가 있다 |
| **self-filter** | depth 에서 robot 자신의 pixel 을 지우는 단계(V11AB 의 `DepthRobotMask`). robot 의 sphere 를 현재 joint 값으로 FK 해 camera 로 투영하고, 그 안에 들어온 depth 점을 제거한다. 지우지 못한 robot 점은 ESDF 에서 장애물이 된다 |
| **constraint model / self-filter model** | V11AB 가 robot 을 두 가지 sphere 집합으로 쓴다. constraint model = optimizer 의 clearance constraint 에 쓰는 sphere(실제 몸체에 가깝게). self-filter model = constraint model + 덮개(cover) capsule(몸체 표면이 depth 에 남지 않게 덮는다). 덮개는 constraint 에 넣지 않는다 |
| **sphere chain (`UrdfSphereChain`)** | URDF 의 link frame 에 sphere/capsule 을 붙이고 joint 값으로 FK 해 sphere 위치를 내는 V11AB 의 robot 표현. finger 개도는 학습 변수가 아닌 parameter joint 로 받는다 |
| **zero-length capsule** | 길이 0, 반지름 r 인 capsule = sphere 하나. cuRobo `franka.yml` 의 collision sphere 를 sphere chain 에 넣는 형태 |
| **cover capsule / self_filter_cover** | self-filter model 에만 들어가는 덮개. robot collision mesh 표면 중 `franka.yml` sphere 가 덮지 못한 부분을 V11AB 의 `self_filter_covering_capsules` fitter 로 맞춘 capsule(sphere role `self_filter_cover`) |
| **confined cover** | 덮개를 body 별로 그 body 의 collision hull 안쪽에 가두어 다시 맞춘 것. 큰 capsule 몇 개 대신 작은 capsule 을 많이 써서 capsule 끝 반구가 몸체 밖으로 나가는 양을 줄인다. V11AB 에서는 RB-Y1 `base` 만 이 방식이다 |
| **protrusion** | sphere union 의 경계가 robot collision hull 밖으로 나간 두께(mm). 덮개가 실제 몸체보다 크게 나간 정도를 본다 |
| **coverage** | robot collision mesh 표면 표본점 중 sphere union 안에 들어간 비율과, 밖인 점의 최대 거리 |
| **false removal (오제거)** | self-filter 가 robot 이 아닌 pixel(target · table · stove 등)을 지운 것. segmentation 이 robot 이 아니라고 한 pixel 중 제거된 비율 |
| **T17 (RB-Y1)** | RB-Y1 쪽 episode 이름. 옛 sphere 모델에서 쥔 사과가 17 frame 동안 통째로 지워졌고(`AG3S_T0T6_LOG.md` §3 T28 Phase 1: 사과 retention 0.472), 원인은 손목 부근 capsule 이 쥔 물체까지 덮은 것이었다. 같은 종류의 문제가 Panda 에서 재현되는지가 SL2 의 되돌아올 지점이었다 |
| **fixture (SL2)** | 금지 접촉(robot 의 arm link 나 hand 가 장애물에 파고드는 자세)을 만든 정적 scene. 관절 값을 random 하게 뽑아 MuJoCo contact 가 의도한 link-물체 쌍에 생기는 q 를 골랐다 |
| **FK (forward kinematics)** | joint 값에서 각 link frame 의 pose 를 계산하는 것. 여기서는 adapter URDF 의 FK 를 robosuite MuJoCo body pose 와 비교한다 |
| **MuJoCo contact** | MuJoCo 가 `mj_step` 중 geom 쌍의 접촉으로 보고하는 것(`dist` < 0 이 접촉 깊이). 독립 정답으로 쓴다 |
| **signed distance (`sphere_sd`)** | link 의 sphere 와 물체 collision geom 사이의 signed distance. 양수 = 떨어져 있음, 음수 = 침투(penetrate). link 별 최솟값을 쓴다 |
| **confusion table (contact × penetration)** | cell 마다 MuJoCo contact 가 있었는가(C) 와 sphere 가 침투했는가(P)로 4칸을 센 표. `C&P` = 접촉을 sphere 가 침투로 잡음, `C&noP` = 접촉을 놓침(miss), `noC&P` = 접촉 없는데 침투(false alarm), `noC&noP` = 둘 다 없음 |
| **manipulator side** | V11AB 의 16-D layout `[L7, Lg, R7, Rg]` 에서 robot 의 팔이 놓이는 쪽. Panda 한 팔은 `left`(wrist camera 가 `left_wrist` 이므로) |
| **attention cell** | V11AB 가 attention 을 읽는 고정 좌표 (denoise step, aggregation, layer, head). V11AB 의 cell = denoise step 0 · aggregation `last` · layer 8 · head 2 (이하 L8H2). RB-Y1 checkpoint 에서 고른 값이다 |
| **denoise step** | pi0.5 의 action expert 가 noise 에서 action chunk 를 만드는 반복(flow-matching) 단계. attention cell 의 step 0 은 첫 번째 단계이고, audit 은 한 call 에 10 step 을 돈다(`num_denoise` 10) |
| **aggregation `last`** | suffix 의 query token(예측 action step 마다 하나)을 하나로 합치는 방식(`AGGREGATIONS` 의 `mean` / `first` / `last`) 중 하나. `last` 는 action horizon 의 마지막 query token 의 attention 을 쓴다 |
| **token block** | VLA prefix 에서 image 한 장이 차지하는 token 구간. pi05_libero 는 `base_0_rgb`(0–256) · `left_wrist_0_rgb`(256–512) · `right_wrist_0_rgb`(512–768, LIBERO 에서는 padding)이고 각 16×16 grid |
| **block mass** | 한 attention cell 의 attention 합 중 token block 하나에 놓인 양. 이 cell 은 wrist block 에 거의 전부(median 0.995)를, head block 에 median 0.00105 를 둔다 |
| **mirror-undo (un-mirror) mapping** | policy image 는 OpenCV image 의 좌우 반전이므로, 16×16 attention grid 의 열을 뒤집고 224→256 scale 을 적용해 OpenCV pixel 로 옮기는 사상. 뒤집지 않은 비교용 mapping 은 mirror left in |
| **grounding** | attention 이 target object 에 놓이는 정도. 여기서는 target mass(target mask 가 덮는 patch 가중 attention 합), area 대비 mass/area, argmax 가 target 위인 비율, rank 로 잰다 |
| **uniform baseline** | attention 이 token 에 균일하게 퍼졌을 때 기대되는 target mass = target 이 image 에서 차지하는 면적 비율(area). mass/area > 1 이면 chance 보다 target 에 모인다 |
| **phase (approach / grasp / transport)** | MuJoCo contact · target 높이로 정한 단계. approach = 첫 gripper–target contact 이전, grasp = 첫 contact ~ lift(target z > z0 + 1 cm), transport = lift ~ 마지막 contact |
| **segP / segRec** | `segP` = image instant 로 다시 render 한 segmentation, `segRec` = `env.step` return 시점에 기록된 segmentation. attention 채점은 `segP` 를 기본으로 썼다 |
| **diagnostic grid** | 전체 18 layer × 8 head = 144 cell 의 grounding 지표 표. V11AB cell 의 순위를 보는 진단용이며 layer/head 선택에는 쓰지 않는다 |
| **dev / test split** | LIBERO Spatial 각 task 의 init state 0–4 를 dev(측정 · 필요 시 calibration), 5–49 를 test 로 분리(안 A, plan §6.0). layer/head 는 dev 에서만 정할 수 있고 test 에 쓰지 않는다 |
| **GPU lock** | 여러 checkout 이 한 GPU 를 쓰므로 GPU 작업은 `/tmp/ag3s-vla-gpu-validation.lock` 을 `flock` 으로 잡고, `nvidia-smi` 로 다른 compute process 가 없음을 확인한 뒤 시작하는 규칙 |
| **ATS** | Attention-based Target Segmentation. VLA attention 을 depth 점에 올리고(attention lifting) 가장 높은 점들을 seed 로 target 점 cluster 를 고르는 V11AB 단계 |
| **LTE** | Link-specific Target Exemption. target 접촉을 허가된(authorized) link 에만 허용한다. 이 기록에서는 허가 link 가 읽는 **target-free layer** 가 그 구현 |
| **target-free layer** | ESDF 에서 ATS target 점(+ 그 주변 target ball)을 제외하고 만든 별도 거리장. 허가 link(Panda 는 `panda_hand` · 두 finger)는 이 layer 를 읽고, 나머지 link 는 target 이 들어 있는 main layer 를 읽는다. 대비 용어: **main layer** |
| **profile (Panda AG3S profile)** | V11AB server 가 쓰는 AG3S 를 Panda 로 만드는 설정 묶음(`AG3S/adapters/panda/profile.py`). config 는 serve_safe 의 V11AB flag 에서 유도하고, Panda 와 다른 부분만 override 로 덮는다 |
| **override** | profile 이 V11AB 값 대신 쓰는 Panda 값. 반드시 근거를 manifest 에 적고 `diff_against_v11ab()` 가 이 목록 외의 차이를 `UNDECLARED` 로 잡는다. 이번 STEP 의 config override 는 `contact.contact_links` 와 `esdf.bounds_lower` 의 z 둘 |
| **offline harness (perception replay)** | 기록된 dev rollout 의 policy-call frame 을 V11AB 의 실제 perception 경로(`process_multi_debug`)에 다시 넣어 closed loop 없이 ATS · ESDF 를 측정하는 도구(`benchmark/runners/perception_replay.py`). 로봇이 움직이지 않으므로 측정이 policy 와 TrajOpt 의 영향을 받지 않는다. 한계: grasp latch · attach 가 없다 |
| **nearest-instance** | ATS target 점 하나가 어느 scene 물체의 표면에 가장 가까운가(10 mm 이내). 점들이 실제로 어떤 물체 위에 있는지의 분포로, GT target 외의 물체(plate, bowl_2 등)를 target 으로 골랐는지 본다. 10 mm 안에 어느 물체도 없으면 `none` |
| **grounding-only** | (SL4b) ESDF · constraint 단계를 건너뛰고 ATS grounding 까지만 실행하는 빠른 선별 모드. full 실행의 grounding 과 같아야 한다 |
| **target_correct** | (SL4b 1차 지표) call 에서 ATS 가 target 을 냈고, target 점의 nearest-instance 다수결이 진짜 target 이며, centroid 오차 ≤ 30 mm 이면 1 |
| **pre-registration (사전 등록)** | 결과를 보기 전에 지표 · 후보 · 채택 조건을 문서로 고정해 두는 것. dev 에서 cell 을 고르더라도 고른 뒤에 기준을 바꾸지 못하게 한다 |
| **oracle** | GT(MuJoCo segmentation 등 privileged 정보)로 만든 참고 값. 방법의 상한을 보는 용도이며 선택이나 실제 입력에 쓰지 않고 privileged 로 표시한다 |
| **fixture (LIBERO)** | scene 에 고정된 가구 · 설비 body(`wooden_cabinet_1`, `flat_stove_1`). joint 가 없어 기록된 `sim_state` 에 들어 있지 않고, LIBERO 가 `env.reset()` 때마다 위치를 다시 뽑는다. (SL2 의 **fixture** 는 금지 접촉 scene 을 뜻해 다른 말이다) |
| **reset history** | 한 run 에서 task 마다 env 하나를 만들고 episode 마다 `reset` 한 횟수와 순서. fixture 위치는 global numpy RNG 로 뽑히므로 k 번째 episode 의 scene 은 그 task 에서 k 번째 reset 의 결과다. export · replay 는 이 이력을 그대로 재현해야 같은 scene 이 된다 |
| **superseded (대체됨)** | 오류가 확인되어 새 산출물로 교체한 옛 산출물. 지우지 않고 `raw/superseded_*` 에 남기되 결과 해석에 쓰지 않는다 |
| **misselection (오선택)** | ATS 가 낸 target 점의 nearest-instance 다수결이 진짜 target 이 아닌 call (`wrong_instance`). 다수가 없거나(동률) 어느 물체도 아닌 `none` 도 포함한다 |
| **cause split** | L8H2 misselection call 을 attention 원인과 grounding 원인으로 나누는 규칙. 두 camera 의 raw attention grid 에서 joint argmax patch 가 진짜 target 의 image-instant segmentation 위에 없으면 attention 원인, 위에 있는데도 ATS 가 다른 물체를 골랐으면 grounding 원인 |
| **frozen worktree** | 한 진단이 끝날 때까지 code 를 한 commit 에 고정하기 위해 만든 detached `git worktree`(`.runtime/worktrees/sl4c-85e6d9d1`). 병행 STEP 의 code 변경이 측정에 섞이지 않게 한다 |
| **pb4 · pb5 (V11AB-pb4 · pb5)** | pb3(arm link 포함) 위에 contact 단계 규칙을 얹은 개선 version. pb4 = pb3 + **held exemption**(latch 가 held 일 때 손이 쥔 물체에 대해 가진 행을 제약에서 뺀다). pb5 = pb4 + **latch phase**(latched · closing 에서 V11AB phase 규칙을 client 의 고정 `approach` 가 아니라 latch 에서 받는다) + **grasp zone** 10 mm(손의 contact link 가 target 과 그 받침면에 10 mm 더 가까이 가도록 허용, 물체에서 수평 100 mm 안). flag 구성은 `AG3S/adapters/panda/profile.py` 의 `AG3S_VERSIONS` |
| **MI gate (minimal intervention gate)** | SQP 전에 reference chunk 의 step clearance 최소값을 보고, 기준(10 mm) 이상이면 reference 를 그대로 계획으로 내는 규칙(pb2). 통과한 call 은 `joint_refined = joint_reference` 다 |
| **HOLD call** | SafePolicy 가 HOLD 를 낸 call. episode 의 `safe_client.hold_calls` 로 센다 |
| **역변환 A · B · C** | TO 가 낸 joint 목표(refined)를 OSC action 으로 돌려주는 변환 모드. A = `lag_inverse`(OSC 1차 응답 모델의 역), B = `command_offset`(`Dx = Dy`), C = `lag_inverse_tracking`(response model 이 예측하는 실제 deviation 을 추적해 refined 누적 위치를 목표로 함). calls 기록의 `conversion.mode` 로 구분한다 |
| **over-lift** | 한 episode 에서 target 의 최고 rise(target z − z(wait))가 250 mm 를 넘는 것. VLA-only 가 성공한 episode 의 최고 rise 는 LI 186.3 mm · LIBERO 10 182.2 mm 이하였다. lift 는 두 손가락 면이 target body 에 닿고(MuJoCo contact dist ≤ 0) rise ≥ 30 mm 인 상태(SL5g #3) |
| **실패 class A / B / C / D (SL5g #3)** | VLA-only 성공 → pb 실패 episode 의 분류. A_no_lift_hold(lift 없음, 최장 HOLD 연속 ≥ 5 call) · A_no_lift_other(lift 없음, HOLD 연속 < 5) · B_lifted_lost(lift 뒤 ≥ 10 obs 연속 grasp 없음) · C_lifted_overlift(lift, 놓치지 않음, 최고 rise > 250 mm) · D_lifted_other(lift, 놓치지 않음, 최고 rise ≤ 250 mm). 위에서부터 처음 맞는 class |
| **saturation (포화, sat)** | 실행 action 의 arm 성분(x · y · z · rx · ry · rz, gripper 제외) 중 하나라도 \|a\| ≥ 0.999 인 것. 포화한 step 은 `lifted` 구간(위 lift 정의)의 env step 으로 센다 |
| **stage ①~⑤ · first stage** | action 이 거치는 단계: ① `actions_policy`(VLA 원본) → ② TO 수정(`joint_refined − joint_reference`) → ③ 역변환 C 의 clip → ④ `actions_response` → ⑤ `actions_executed`. first stage = 포화한 response row 가 처음 포화한 단계: `vla`(① 이 이미 포화) · `conv_on_to_row`(역변환이 clip 한 row 이고 TO 가 바꾼 row) · **`conv_carry`**(역변환이 clip 한 row 인데 TO 는 바꾸지 않음, 앞 row 의 부족분을 이월) · `unclipped`(clip 없이 포화) |
| **TO-changed row** | chunk 10 행 중 TrajOpt 가 바꾼 행(`joint_refined ≠ joint_reference`). MI gate 통과 call 의 행은 모두 TO-unchanged |
| **D\*** | TO 가 요청한 EE 변화 = FK(`joint_refined[k,4]`) − FK(`joint_reference[k,4]`) (mm). 역변환 C 가 받는 deviation 이다. D\*_z 는 z 성분 |
| **realised (realised_frac_obs)** | D\* 중 실제 EE 가 따라간 비율: (eef_obs(o+5) − FK(reference[k,4])) 를 D\* 방향에 사영한 값 / \|D\*\|. 1 이면 요청이 그대로 실현, 0 이면 실현 안 됨, 1 초과면 요청보다 더 움직임. `realised_fraction_model` 은 server 가 기록한 response model 의 예측 |
| **windup** | 역변환 C 의 clip 이 같은 방향으로 쌓이는 모양을 보는 조작 정의: `conversion.saturated_rows` 가 비어 있지 않고 cos(D\*_k, D\*_{k−1}) > 0.5 인 call 의 최대 연속 run. 역변환의 `d^`, `D^` 는 call 마다 0 에서 시작하므로 call 사이 상태는 이월되지 않는다 |
| **reach_mm · reach_limit_mm · `checks.reach`** | latch evidence 가 잰 manipulated centroid 와 손 사이 거리(mm)와 한도(120 mm), 그리고 그 한도 안인지의 검사. False 이면 latch 가 grasp 를 확인하지 못한다(note "파지 미확인: reach"). `manipulated_state` = `occluded` 이면 geometry 가 마지막 관측 frame 에 고정된다 |
| **규칙 (a) · (b) (후보 규칙)** | (a) VLA-only official fail → version official success, (b) VLA-only safe_success False → version safe_success True. 두 규칙을 따로 센다 |
| **병렬 GPU run · `AG3S_GPU_LOCK_HELD` · `held_by_caller`** | 한 caller(`benchmark/runners/run_parallel_gpu.sh`)가 공용 GPU lock 을 한 번 잡고 launcher job 을 최대 2 개 동시에 돌리는 방식(같은 checkout 안에서만). 환경 변수 `AG3S_GPU_LOCK_HELD=1` 이면 launcher 는 lock 을 직접 잡지 않고 caller 가 물려준 fd 로 lock 을 쥐고 있는지만 확인한다. run manifest 의 `launcher_result.gpu_lock_mode` = `held_by_caller`. checkout 사이 동시 실행은 계속 배제한다 |
| **병렬 bit 재현 (G0)** | 병렬 batch 에서 돈 run(repro)과 같은 code · 인자의 단독 run 의 episode npz · calls npz · geom npz 의 모든 array 를 byte 단위로 비교하는 것. 시각 · timing 기록만 달라야 한다 |
| **dirty run** | run manifest 의 `dirty = True`: 실행 시 working tree 에 추적되지 않는 파일이나 수정이 있었다. `source_diff` 가 비어 있으면 code 변경이 아니라 untracked 파일 때문이다 |
| **파지 인식 수정판 · 파지 인식 + closing 면제판 · grasp zone 허용판 (SL5i)** | 내부 key V11AB-pb6 · pb7 · pb5. grasp zone 허용판(pb5) = pb4 의 held 면제 + latch phase + grasp zone 10 mm. 파지 인식 수정판 = 허용판 + sticky reach + drop-from-open. 파지 인식 + closing 면제판 = 수정판 + closing 면제 |
| **닫힘 시도 (closing attempt)** | latch 가 LATCHED→CLOSING 으로 넘어간 때부터 reopen · revoke(attach 취소) · LATCHED 복귀 · reset 중 하나가 일어날 때까지의 구간. sticky reach 의 유효 범위다. attach 거절로 HELD→CLOSING 으로 되돌아온 것(`revert_attach`)은 같은 시도로 본다 |
| **sticky reach** | 한 닫힘 시도 안에서 reach check(손 – manipulated centroid 거리 ≤ 120 mm)가 한 번 통과했으면, 그 뒤 다른 capture 로 거리가 다시 재져 실패해도 시도가 끝날 때까지 통과로 두는 규칙. 같은 `manipulated_id` 일 때만 적용하고 AG3S `attach` 의 reach gate 에도 같은 판단을 넘긴다. flag `--grasp-reach-sticky` |
| **drop-from-open** | `dropped` check(`opening_before − opening ≥ 0.05`)의 기준 개도를 LATCHED→CLOSING 전이 순간의 개도가 아니라 닫힘 명령이 시작되기 전 마지막으로 열려 있던 측정 개도(명령 ≥ 0.85 이고 개도 ≥ 0.85)로 바꾸는 규칙. latch 가 손가락이 거의 닫힌 뒤에 잠겨도 기준이 열림(≈1.0)으로 잡힌다. flag `--grasp-drop-from-open` |
| **closing 면제 (closing exemption)** | latch 가 CLOSING 이고 그 call 의 `closed` · `not_empty` · `blocked` 가 모두 참일 때(손가락이 무언가를 쥐고 멈춘 상태), held 면제와 같은 규칙으로 손의 authorized link 행 중 nearest surface 가 manipulated object 인 행을 TO 제약에서 뺀다. 다른 물체 · table · obstacle 행은 남긴다. flag `--closing-hand-exempt`, record `closing_exempt_rows` |
| **거짓 held (false held)** | latch 가 held 로 넘어갔는데 그 call 의 obs 에서 양손가락이 target 에 닿지 않은 것(MuJoCo `both_fingers` X). "새로 생긴 거짓 held" = 기존 version 에는 같은 seq 에 없던 것 |
| **양손가락 (`both_fingers`)** | MuJoCo contact probe 에서 왼쪽 · 오른쪽 손가락 면이 모두 target body 에 닿아 있는 상태(dist ≤ 0). lifted = 양손가락 O 이고 rise ≥ 30 mm |
| **offline latch replay · 첫 divergence** | 기록된 calls log 의 latch 입력을 `GraspLatch` 에 다시 넣어 flag 를 켰을 때의 state 전이를 구하는 CPU 검사. 첫 divergence = flag on 의 출력이 기록과 처음 달라진 call(그 call 까지는 실제 run 과 입력이 같다. 첫 held 뒤에는 closed loop 가 갈라진다) |
| **revoke** | 이미 attach 된 물체를 latch 가 detach 하는 것(`attach_revoked`). 이후 latch 는 LATCHED 로 돌아가 같은 물체를 다시 시도할 수 있다 |
| **SL6 smoke 용어 (safe success · HOLD call · 쥔 bowl 미인식 · 포화 action step · E0 TrajOpt · pb1/pb6/pb7)** | 이 로그 끝의 "Panda 최종 구성 SafeLIBERO smoke 기록 (2026-10-29 ~ 10-30)" 머리말 "이 묶음에서 새로 쓰는 말" 표에 정의했다 |

---

## 고정 기준

SL0 에서 확정했고 이후 단계가 바꾸지 않는다. 바뀌면 그 STEP 이 사유를 기록한다.

| 항목 | 값 |
|---|---|
| branch / 기준 commit | `feature/panda-benchmark` / SL0 구현 `6ba81024`, 검증 `696abd52` |
| checkpoint | `/mnt/dev/AG3S_VLA-storage/checkpoints/panda/pi05_libero` (read-only storage input, 공개 GCS 에서 받음, OpenPI config `pi05_libero`) |
| checkpoint aggregate SHA-256 | `1cb098d84bde409fb4b3931ed676195b80b0755c6ee60177e3d6c96508f1d891` |
| norm stats SHA-256 | `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84` (`assets/physical-intelligence/libero/norm_stats.json`) |
| LIBERO client 환경 | `.venv-libero`, **Python 3.8.20**, OpenPI 예제 `requirements.txt` pin 그대로 (robosuite 1.4.1, mujoco 3.2.3, numpy 1.22.4, torch 1.11.0+cu113). 재현 기록: `environments/libero/` |
| policy server 환경 | `.venv-openpi-live`, Python 3.11.16, GPU |
| launcher 환경 | `.venv-ag3s`, Python 3.11.16 (CPU) |
| LIBERO source | submodule `benchmark/LIBERO` @ `f78abd68ee28` (수정 금지) |
| client 기준 | `third_party/openpi/examples/libero/main.py` |
| protocol 상수 | suite `libero_spatial`, seed 7, `num_steps_wait` 10, `replan_steps` 5, `resize_size` 224, render 256, `max_steps` 220 (Spatial), dummy action `[0]*6+[-1]`, video 10 fps |
| 판정 | official success = `env.step()` 의 `done` |
| 렌더러 | `MUJOCO_GL=osmesa` |

사용자 판정 (2026-10-12): `pi05_libero` 는 공개 GCS 에서 storage 의 새 input 으로 받는다. LIBERO client 는 공식
protocol 과 같은 별도 Python 3.8 `.venv-libero` 를 이 checkout 안에 만든다. 전역 `python3.11` shim 은 건드리지 않는다.

---

## SL0 — pi0.5-only LIBERO Spatial 1 episode (2026-10-12)

> 2026-10-12 KST. 구현 commit `6ba81024`(KST 04:40) · 검증 commit `696abd52`(KST 05:00). 사용자 확정.
> run dir 이름의 시각(예: `20261012T045129`)은 **KST 다.** `common/run_artifacts.py` 의 `create_run` 이 UTC+9 timezone 으로 시각을 만들기 때문이다(구조상 확정, 추정 아님).
> 원본: [`handoff/SL0.task.md`](handoff/SL0.task.md) · [`handoff/SL0.impl.md`](handoff/SL0.impl.md) ·
> [`handoff/SL0.verify.json`](handoff/SL0.verify.json). raw 출력은 `outputs/verify/SL0/`.

### 이 STEP 이 답하는 질문

이 checkout 만으로(이전 workspace 없이) frozen pi0.5 LIBERO checkpoint 가 공식 LIBERO Spatial protocol 의
1 episode 를 공식 판정까지 완주하고, 그 run 이 입력을 건드리지 않고 고유 디렉터리에 재현 가능하게 기록되는가.

```
LIBERO sim (.venv-libero, py3.8) --websocket--> OpenPI policy server (.venv-openpi-live, GPU)
   obs: agentview/wrist RGB(~~180 deg rotation~~ → SL1 측정: left-right mirror, 180° 회전 아님), eef pos + axis-angle + gripper qpos, prompt
   <-- action chunk (EE delta 6 + gripper 1), replan_steps 만 실행 후 다시 호출
   official success = env.step() 의 done
```

### 무엇을 만들었나

공식 LIBERO client 와 같은 동작의 client 와, run 생성 → 빈 port 선택 → storage checkpoint 로 server 기동 →
client 실행 → **자기가 띄운 server 만** 종료하는 launcher 를 만들었다. AG3S 코드와 `third_party/openpi`,
`benchmark/LIBERO` 는 수정하지 않았다.

| 파일 | 역할 |
|---|---|
| `scripts/setup/create_libero_env.sh` | 자기 `.python` 에 Python 3.8.20 을 설치하고 `.venv-libero` 생성. 공식 두 requirements 를 `uv pip sync` 로 설치, freeze 기록과 재현 확인(no-op 여부) |
| `environments/libero/{requirements,libero-requirements,validated-freeze}.txt` | 공식 requirements 의 byte 사본과 설치 결과 freeze |
| `benchmark/runners/libero_pi05_client.py` | Python 3.8 client. 공식 loop 와 같은 동작에 `episodes.jsonl` · step 별 npz · mp4 를 exclusive create 로 기록 |
| `benchmark/runners/launch_libero_pi05_only.py` | launcher. port 선택(8123 금지, bind 확인), checkpoint identity 확인, server/client 실행과 manifest 기록, `--dry-run` |
| `tests/benchmark/test_libero_pi05_sl0.py` | CPU unit test (공식 예제와 protocol 상수 · `_quat2axisangle` 일치, `--run-dir` 거부 규칙, port 선택, launcher command 구성 등) |

`.venv-libero` 는 Python 3.8 용으로 구현 중 세 가지 장애를 client/launcher 쪽에서 막았다.
(1) 이 fork 의 `openpi-client` 가 보내는 `ping_timeout` 을 Python 3.8 에서 쓸 수 있는 마지막 websockets 가 받지 못해 연결 시 `TypeError` 가 난다.
(2) pod 의 전역 `LD_LIBRARY_PATH` 가 다른 Python 의 `torch/lib` 를 가리켜 venv 의 torch 가 import 되지 않는다.
(3) LIBERO 최상위 `libero/` 에 `__init__.py` 가 없어 일반 editable 설치로는 import 되지 않는다.

### 실행

| 항목 | 값 |
|---|---|
| 명령 | `.venv-ag3s/bin/python benchmark/runners/launch_libero_pi05_only.py --task-suite-name libero_spatial --task-ids 0 --episode-ids 0 --seed 7 --run-id sl0-pi05only-t0e0` |
| run dir | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T045129_6ba81024d243_sl0-pi05only-t0e0` |
| task | libero_spatial task 0, episode 0 (init state 0): "pick up the black bowl between the plate and the ramekin and place it on the plate" |
| server port | 24038 |
| client / server returncode | 0 / -2 (launcher 가 보낸 종료 signal. `server_console.log` 의 `KeyboardInterrupt` traceback 은 이 종료다) |

### 결과 — Figure

세 figure 는 verifier 가 만들었다 (`outputs/verify/SL0/scripts/make_figures.py`, sidecar `.json` 동봉).

**실제 scene** — agentview · wrist 원본 frame 의 시작 / 첫 policy 호출 / 중간 / 끝 montage (raw 256×256, 공식 ~~180° 회전~~ 좌우 반전(SL1 #3) 전).

![SL0 scene](figures/sl0-scene.png)

**graph** — EEF xyz · gripper qpos · 실행 action · gripper 명령의 step 별 시계열(회색 = wait 10 step, 점선 = done step)과
호출별 latency. 첫 호출(JIT compile 포함)은 steady-state 와 자릿수가 달라 막대로 따로 그렸다.

![SL0 trend](figures/sl0-trend.png)

**table** — protocol · checkpoint · 환경 · 결과 · 입력 보호 · 회귀를 한 장에 모은 표.

![SL0 table](figures/sl0-table.png)

### 결과 — 수치

> **episode 1개의 사실이다. 성공률이 아니다.** task 0 · init state 0 · seed 7 에서 한 번 성공했다는 것 이상은
> 말하지 않는다. 이 episode 로 pi0.5 의 성능이나 공개 수치와의 일치 여부를 판정하지 않는다
> (성능 확인은 P2 소규모 평가의 일이다).

**Episode**

| 항목 | 값 |
|---|---|
| official_success | `true` |
| status | `success` (error 없음) |
| env step (wait 포함 / 제외) | 113 / 103 (wait 10, `max_steps` 220) |
| done 이 선 step index | 112 |
| policy 호출 수 | 21 |
| action chunk 배열 shape | 21 × 10 × 7 |
| mp4 | `libero_spatial_t0_e0_success.mp4`, 103 frames, 10 fps, 224×224 |
| 실행 action 과 chunk 불일치 step | 0 |
| wait 구간 action = dummy action | `true` |

**Latency** (client 는 `client.infer` 왕복, server 는 서버 측 `infer_ms`, policy 는 모델 추론 구간)

| 항목 | 값 |
|---|---|
| server ready (Popen → `/healthz` 200) | 58.03 s |
| orbax restore (server log) | 30.44 s |
| client episode 총 시간 | 90.42 s |

| 구간 | 첫 호출 (JIT 포함) | 호출 2~21 (n=20) p50 | p95 | max |
|---|---|---|---|---|
| client 왕복 | 36,497.7 ms | 88.6 ms | 220.6 ms | 305.6 ms |
| server infer | 36,493.9 ms | 85.8 ms | 182.9 ms | 303.6 ms |
| policy infer | 35,366.2 ms | 46.8 ms | 121.6 ms | 126.1 ms |

첫 호출 36.5 s 는 JIT compile 이 지배한다. 이 값을 steady-state latency 로 쓰지 않는다. 이후 호출이 V11AB 의 runtime
deadline 안에 드는지는 AG3S 가 붙은 뒤에 따로 잰다 — SL0 는 측정과 기록만 한다.

**기록 완결성** (plan §6.1 필수 항목 중 이 단계 해당분)

| 산출물 | 상태 |
|---|---|
| `episodes.jsonl` 필수 field | 누락 0 |
| episode npz 배열 | 누락 0 |
| mp4 | 있음 |
| `policy_records/` | 21 파일, 21 개 모두 npz 의 action chunk 와 일치 |
| server log | `logs/libero_client.log` · `logs/serve_policy_24038.log` · `logs/server_console.log` 있음 |
| `run_manifest.json` 필수 20 항목 | 16 present, **4 missing** (아래 열린 항목) |
| `source.patch` | 0 bytes, dirty = false |

**입력 보호** (실행 전후 storage 비교)

| 항목 | 값 |
|---|---|
| checkpoint 디렉터리 stat diff | 0 줄 |
| storage 기존 input (results 제외, 21,026 entries) stat diff | 0 줄 |
| `results/benchmark` 새 dir | 1개 (이번 run), 삭제된 dir 0 |
| `/mnt/dev/work` 경로 hit | 3 venv origin 0, run 의 text file 0 |

**checkpoint identity** — 독립 rehash script 로 16 파일 12,439,085,481 bytes 를 163.9 s 에 전부 다시 해시했다.
aggregate(파일 path 순 정렬) 재계산 값이 기록값 `1cb098d8…` 과 일치하고 per-file mismatch 는 0 이다.
(launcher 의 `--verify-checksums` 경로 자체는 이 STEP 에서 실행하지 않았다. 아래 not measured.)

**환경 독립성**

| venv | interpreter | 위치 |
|---|---|---|
| `.venv-libero` | Python 3.8.20 | `/mnt/dev/AG3S_VLA-benchmark/.python/cpython-3.8.20-linux-x86_64-gnu` |
| `.venv-openpi-live` | Python 3.11.16 | `/mnt/dev/AG3S_VLA-benchmark/.python/cpython-3.11.16-linux-x86_64-gnu` |
| `.venv-ag3s` | Python 3.11.16 | 같은 3.11.16 |

**회귀**

| 검사 | 결과 |
|---|---|
| live 보고서 범위 CPU test (`.venv-openpi-live`) | 73 passed, 0 skipped, 77 warnings |
| `tests/benchmark` (`.venv-ag3s`) | 16 passed, 0 skipped |
| `tests/benchmark` + `test_run_artifacts` + `test_policy_recorder_destination` | 20 passed, 1 skipped (`.venv-ag3s` 에 flax 가 없어 recorder test skip. impl.md 가 기존과 같은 skip 이라고 기록) |
| 16D legacy ESDF 기준선 | violated start 14/15 · has_target 15/15 · frame0 clearance -29.031 mm — 기준값과 일치 |
| 16D cuRobo ESDF 기준선 | violated start 10/15 · has_target 15/15 · frame0 clearance -9.172 mm — 기준값과 일치 |

SL0 가 새 파일만 추가했고 기존 코드를 바꾸지 않았으므로, 위 16D 기준선이 그대로인 것을 확인했다.

**자원** (실행 중 sampling 70 회)

| 항목 | 값 |
|---|---|
| pod working set 최대 | 61.403 GiB |
| GPU 전체 사용 최대 | 44,948 MiB (외부 2 process 36,136 MiB 포함) |
| 이 run 의 server GPU 최대 | 8,812 MiB |

### 공식 예제와의 차이

공식 `third_party/openpi/examples/libero/main.py` 와 **정책 입력 · 실행 action 은 같고**, 아래는 기록·운영 쪽 차이다
(출처 `SL0.impl.md`).

| # | 차이 | 영향 |
|---|---|---|
| 1 | `--task-ids` / `--episode-ids` 로 부분집합 실행 | task 0 episode 0 은 공식 sweep 의 첫 episode 와 같은 순서. 뒤쪽 episode 는 앞선 episode 를 거치지 않아 전체 sweep 의 같은 번호와 env 이력이 다를 수 있다 |
| 2 | 예외 시 `status=execution_error` + traceback 기록, client exit code 1. episode 시작 시 `done=False` | 공식은 예외 시 이전 episode 의 `done` 이 남을 수 있다 |
| 3 | 영상 이름 `videos/<suite>_t<task>_e<ep>_<success|failure>.mp4`, exclusive publish | 공식은 같은 task 의 다음 episode 가 덮어쓴다. frame · fps 동일 |
| 4 | 추가 기록: `episodes.jsonl`, step 별 npz, `client.infer` 왕복 시간, `server_timing` / `policy_timing` | 정책 입력 · 실행 action 불변 |
| 5 | 무거운 import 를 함수 안으로 이동 | 없음 (CPU test 가 Python 3.11 에서 module 을 import 하기 위함) |
| 6 | CLI 를 argparse 로, `--port` 필수, launcher 는 `--host 127.0.0.1` 전달 | 없음 |
| 7 | websockets 13.1 (Python 3.8) 에서 `ping_timeout` 없이 연결하는 subclass | upstream OpenPI client 와 같은 연결 |
| 8 | LIBERO import 전 `LIBERO_CONFIG_PATH/config.yaml` 이 없으면 거부 | 없음 (config 가 없으면 LIBERO 가 `input()` 으로 멈춘다) |
| 9 | 환경: `MUJOCO_GL=osmesa`, PYTHONPATH 제거, foreign torch `LD_LIBRARY_PATH` 제거, thread 수 제한 | **렌더러가 공식 GPU EGL 이 아니라 CPU llvmpipe.** pixel 단위 차이 가능성은 측정하지 않았다 |
| 10 | 설치 도구 버전, LIBERO editable compat 모드 | pin 은 예제 그대로 |

`pip check` 는 공식 방식(`uv pip sync` 는 의존성을 풀지 않는다)으로 LIBERO requirements 의 transitive 의존성이
설치되지 않아 불일치를 보고한다. 목록은 `SL0.impl.md`. 평가 경로(benchmark → env → `OffScreenRenderEnv` → step)는
이것들 없이 import · 실행됐다.

### 측정하지 않은 것

- renderer 차이(osmesa vs 공식 EGL)의 pixel 영향
- **성공률** (episode 1개)
- launcher 의 `--verify-checksums` 경로 자체 (독립 rehash script 로 대신 재계산)
- 다른 task / episode

### 열린 항목

| # | 내용 | 영향 / 다음 |
|---|---|---|
| 1 | ~~**manifest 필수 항목 4개 누락**: `protected_authorized_links`, `phase`, `level`, `runtime_schedule`~~ | ~~pi0.5-only 에는 해당하지 않는 항목이지만 "해당 없음" 이라는 **선언도 기록돼 있지 않다**. V11AB 와 같은 형식으로 비교하려면 N/A 를 명시해 채워야 한다. 처리 시점은 lead 판정 대기~~ **SL1 에서 처리**: manifest 에 `n/a` + 이유와 `runtime_schedule`(control 20 Hz, replan 5 step 등)을 명시하도록 launcher 를 고쳤다. 확인은 unit test(`run_cpu_regression.sh` B 44 passed) 수준이며, 실제 policy server 를 띄운 launcher 실행에서 manifest 를 다시 확인한 것은 아니다. |
| 2 | ~~manifest 의 `serve_policy` 자기 execution 항목에 `settings.env = "aloha_sim"` 이 기록됨~~ | ~~serve_policy `Args` 기본값이 그대로 기록된 것이다. 실제 policy 는 `policy:checkpoint` 경로(`pi05_libero`, storage checkpoint)로 로딩됐다. 그러나 manifest 만 읽으면 환경이 aloha_sim 처럼 보여 오독 위험이 있다. 같은 항목의 `record_dir` 은 null~~ **SL1 에서 처리**: checkpoint 경로로 서빙하면 `env = None`, `policy = {source, config, dir}` 로 기록한다(`serve_policy.py` 최소 변경). unit test 확인 수준이며 live 서버 실행의 manifest 는 재확인 전이다. 이전 run 의 `aloha_sim` 기록은 그대로 남아 있다. |
| 3 | `.venv-libero` 의 torch(1.11.0+cu113)가 standalone import 후 process 종료 시 `free(): invalid pointer` 로 abort | episode 는 영향받지 않았다(client returncode 0, 기록 완료). pod 의 `LD_LIBRARY_PATH` 를 그대로 쓰면 torch import 자체가 `undefined symbol` 로 실패해, launcher 가 client env 에서 foreign site/dist-packages 항목을 제거한다. 종료 시 abort 의 원인은 확인하지 않았다 |
| 4 | ~~**73-test 의 명령이 이전에 기록돼 있지 않았다.**~~ | ~~이전 live 보고서와 live-validation run 에 정확한 argv 가 없어, verifier 가 collected 수 4+2+23+6+38 = 73 과 warning 출처(t43ta 16+61 = 77)로 파일 구성을 **재구성**해 실행했다. 재구성 명령은 verify.json `pytest_regression73` 에 있다. 이 범위의 정본 명령으로 삼을지는 lead 판정 대기~~ **SL1 에서 처리**: 정본 명령은 `bash scripts/testing/run_cpu_regression.sh` 이다(A 가 SL0 의 5 파일 73 개). SL1 검증에서 A 73 passed, exit 0. |
| 5 | **pod working set 이 실행 중 최대 61.403 GiB.** lead 가 전달한 pod budget 은 64 GiB 인데(이 값은 verify.json 에 없다), 여유가 작다 | 측정 전 checkpoint page cache 를 `posix_fadvise DONTNEED` 로 비웠다(파일 내용 · mtime 불변, stat diff 0). 공유 host 의 load average 는 실행 직전 68~92 였다. server 와 client 가 한 pod 에서 돌 때의 memory 여유를 V11AB(추가 geometry 처리) 단계에서 다시 본다 |
| 6 | `CHECKSUMS.json` 의 `aggregate_rule` 문구와 실제 계산이 다르다 | 문구는 "sorted `<sha256>  <path>\n` lines" 인데 producer 는 record 를 **path 순** 으로 정렬한다. line 문자열 정렬로 재계산하면 `baffa4ca…` 가 나와 불일치, path 순이면 기록값과 일치한다. launcher 는 path 순을 쓴다. 문구 수정은 이 STEP 의 범위 밖 |
| 7 | manifest 의 `storage.rollouts`(`<run>/runs`) 경로가 만들어지지 않았다 | 이 run 의 산출물은 그 경로를 쓰지 않으므로 기록만 한다 |

### 판정

verifier 판정(`verify.json` 의 verdict): 16D 기준선 일치(legacy · cuRobo), `status=success` · `official_success=true`,
checkpoint aggregate 일치, input stat diff 0 · 새 results dir 1, 73 passed · `tests/benchmark` 16 passed.
**SL0 의 통과 기준은 "완주와 기록"이며(완주 = status 가 success 또는 max_steps), 충족했다.** 사용자가 2026-10-12 에 확정했다.
다음은 SL-1(agent-view · wrist RGB/depth 와 camera 변환 → AG3S 관측 형식)이다.

### 되돌아올 지점 (SL0)

| 전환 신호 | 돌아와서 할 일 |
|---|---|
| Python 3.8 pinned stack 이 이 host(CUDA driver 580, OSMesa)에서 설치 · 렌더가 안 된다 | 3.11 단일 env 대안은 공식 조건과 달라지므로 **사용자에게 다시 묻는다.** (이번에는 설치 · 렌더 모두 되어 해당 없음) |
| 1 episode 가 `execution_error` | adapter 단계로 넘어가지 않고 원인(관측 형식 · norm stats · port/통신)을 먼저 닫는다. (이번에는 `success` 라 해당 없음) |
| 이후 episode 가 failure | SL0 통과와 별개다(완주 · 기록이 기준). 성능 확인은 P2 소규모 평가에서 한다 |
| ~~V11AB 와 manifest 형식을 맞출 때 열린 항목 1 · 2 가 비교를 막는다~~ | ~~launcher 의 settings 에 N/A 선언과 policy source 를 명시하도록 고친다 (구현은 implementer)~~ SL1 에서 처리 |
| renderer 가 결과 차이의 원인으로 의심된다 | osmesa 와 EGL 의 pixel 영향을 verifier 에게 측정 요청한다 (이번에는 측정하지 않음) |

---

## SL1 — LIBERO Panda camera adapter: RGB/depth · K · T_base_cam → V11AB wire 관측 (2026-10-12)

> 2026-10-12 KST. 구현 commit `0e13395e`(KST 12:33) · 검증 commit `6c7ce76f`(KST 13:14). 사용자 확정 2026-10-12.
> 원본: [`handoff/SL1.task.md`](handoff/SL1.task.md) · [`handoff/SL1.impl.md`](handoff/SL1.impl.md) ·
> [`handoff/SL1.verify.json`](handoff/SL1.verify.json). raw 출력은 `outputs/verify/SL1/`.
> 이 절의 수치는 모두 `SL1.verify.json` 의 `numbers` 가 출처다. 단위는 mm 이고, 따로 적지 않으면 114 observation(task 0 · episode 0 의 replay 전체) 위의 값이다.

### 이 STEP 이 답하는 질문

LIBERO Panda 의 두 camera 관측을, **policy 입력을 바꾸지 않고**, V11AB 가 읽는 OpenCV / robot-base 규약의
metric depth · K · `T_base_cam` 으로 **물체 · table 에 mm 수준으로** 옮길 수 있는가?

```
robosuite render (OpenGL row order, depth = normalized z-buffer [0,1])
  -> [SL1 adapter, Python 3.8]  OpenCV row order, metric z-depth(m), K(fovy), T_base_cam = inv(T_world_base) * T_world_cam (OpenCV axes)
  -> V11AB wire key: ag3s/cameras, ag3s/depth/<cam>(uint16 mm), ag3s/K/<cam>, ag3s/T_base_cam/<cam>,
                     ag3s/robot_state/<cam>, ag3s/stamp/<cam>, ag3s/stamp_mode = sim_frozen
  -> AG3S.trajopt.wire.unpack_camera_observations -> CameraObservation (OpenCV, robot base frame)
camera 이름 table: agentview -> head, robot0_eye_in_hand -> left_wrist
```

`left_wrist` 로 매핑한 이유는 pi05_libero transform 이 wrist image 를 `left_wrist_0_rgb` slot 에 넣기 때문이다. SL-3 attention token 순서와 맞추기 위한 선택이며, SL-3 audit 결과가 다르면 adapter table 만 바꾼다.

### 무엇을 만들었나

| 구성 | 내용 |
|---|---|
| camera adapter (`benchmark/adapters/libero_panda/camera.py`, Python 3.8) | camera 별 metric z-depth(OpenCV row order) · K(fovy 식) · `T_base_cam`(MuJoCo body `robot0_base` 기준, OpenCV 축) · 9D `robot_state`(joint 7 + gripper 2) · stamp. robosuite `camera_utils` 의 세 공식을 다시 썼다(그 module 이 `h5py` 를 import 하는데 `.venv-libero` 는 공식 pin 이라 설치하면 freeze 재현이 깨진다). 원 공식과의 일치는 AST 로 꺼내 exec 한 값으로 비교한다 |
| policy image <-> OpenCV pixel 사상 | client 의 `[::-1, ::-1]` + `resize_with_pad` 256→224 를 명시적 함수(`opencv_to_policy_pixel` / `policy_to_opencv_pixel`)와 역함수로 만들었다 |
| `capture_state` observable | robosuite 는 camera render 를 `env.step` 끝이 아니라 physics substep loop 안에서 찍는다. 그래서 `env.step` 이 돌아온 뒤의 `env.sim` 은 image 보다 늦다. image 와 같은 substep 의 `T_world_cam` 등을 obs 에 싣는 읽기 전용 observable 을 geometry capture 가 켜졌을 때만 설치한다 |
| wire replica (`wire_pack.py`, Python 3.8) | `AG3S.trajopt.wire.pack_request` 의 camera 부분과 key · dtype 이 같은 dict/npz. AG3S 를 import 하지 않는 복제이므로 cross-env test 로 동치성을 보장한다 |
| client `--capture-geometry` (기본 off) | on 이면 `camera_depths=True` 로 env 를 만들고 step 마다 `episodes/<stem>_geom.npz` 기록. `--capture-segmentation` · `--capture-float-depth` 도 기본 off. off 이면 client argv · 출력이 SL0 와 같다 |
| action replay (`libero_replay.py` + `launch_libero_replay.py`) | 기존 run 의 기록 action 으로 policy 없이 같은 task · init state · seed 를 재생하고 새 run 에 geometry 를 기록한다. client 의 `run_tasks` 를 그대로 써서 seed · env 생성 · 대기 · replan · done 처리가 같다. 기록 `action` 과 chunk 가 bit 단위로 안 맞으면 시작을 거부하고, 기록이 모자라면 action 을 만들어내지 않고 `execution_error` 로 끝낸다 |
| manifest 해당 없음 항목 + `runtime_schedule` | SL0 launcher 의 manifest settings 에 protected/authorized links · phase · level 등 pi0.5-only 에 해당 없는 항목을 `n/a` + 이유로 명시. `runtime_schedule` 은 LIBERO source 에서 AST 로 읽은 control 20 Hz, replan 5 step, policy 호출 주기 5 step = 0.25 s sim 등을 적는다 |
| `serve_policy` env 기록 수정 | checkpoint 경로로 서빙하면 manifest 의 `env` 를 `None`, `policy = {source, config, dir}` 로 기록한다(이전에는 `Args` 기본값 `aloha_sim` 이 그대로 남았다). load · serve 동작은 변경 없음 |
| 고정 CPU 회귀 `scripts/testing/run_cpu_regression.sh` | A: SL0 verifier 가 재구성한 5 파일(`.venv-openpi-live` + `run_without_legacy.py`), B: `tests/benchmark`(같은 venv), C: `tests/benchmark`(`.venv-ag3s`). 하나라도 실패면 exit 1 |
| test | `tests/benchmark/test_libero_sl1.py` + `.venv-libero` 용 `libero_py38_checks.py`(pytest 가 subprocess 로 호출) |

`AG3S/**` algorithm · config, `robots/**`, `benchmark/LIBERO` submodule, storage 기존 input 은 건드리지 않았다.

**계획과 달라진 점** (출처 `SL1.impl.md`)

| # | 차이 | 영향 |
|---|---|---|
| 1 | robosuite `camera_utils` 를 import 하지 않고 공식을 다시 씀 | 값은 AST 비교로 확인 |
| 2 | depth 를 float64 로 계산 후 float32 저장 | robosuite 의 float32 계산과의 차이는 wire 양자화보다 작다고 impl.md 가 기록한다 (이 STEP 의 verify 수치는 아님). wire 왕복 오차는 #5 에서 측정 |
| 3 | geometry on 이면 env 에 `ag3s_capture_state` observable 추가 | 아래 #1 이 RGB 불변을 확인한다 |
| 4 | segmentation 은 `env.step` 이 돌아온 시점에 따로 render | image 순간보다 1 substep 뒤. 아래 "capture instant" 에서 크기를 잰다 |
| 5 | K 는 robosuite / AG3S RB-Y1 과 같은 `cx = W/2`, `cy = H/2` | pixel 중심이 정수 index 인 AG3S `backproject` 규약에서는 광축이 `(W-1)/2 = 127.5` 라 0.5 pixel 어긋난다. 이 STEP 에서는 고치지 않고 크기를 쟀다. 아래 판정 |

### 실행

| 항목 | 값 |
|---|---|
| source (replay 대상) | SL0 run `20261012T045129_6ba81024d243_sl0-pi05only-t0e0` (task 0, episode 0, 읽기만) |
| run a (geometry on, depth float 포함) | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T123505_0e13395ee6a6_sl1-replay-geom-t0e0` |
| run b (geometry + segmentation + depth float) | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T123554_0e13395ee6a6_sl1-replay-seg-t0e0` |
| run c (flag 없음 = SL0 경로 보존 확인) | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T123646_0e13395ee6a6_sl1-replay-off-t0e0` |
| 명령 | `.venv-ag3s/bin/python benchmark/runners/launch_libero_replay.py --source-run <SL0 run id> --task-ids 0 --episode-ids 0 [--capture-geometry --capture-float-depth [--capture-segmentation]] --run-id sl1-replay-{geom,seg,off}-t0e0` |
| 환경 | GPU · policy server 불필요. launcher `.venv-ag3s` -> replay `.venv-libero`. 렌더러 `llvmpipe (LLVM 15.0.7, 256 bits)`, Mesa 23.2.1 (OSMesa), `offsamples` 4 |
| 자원 | pod working set 최대 45.179 GiB, 새 results dir 3개 |

### 결과 — Figure

세 figure 는 verifier 가 만들었다 (`outputs/verify/SL1/scripts/make_figures.py`, sidecar `.json` 동봉).

**실제 scene** — run b 의 observation 55. 위 줄: head RGB(segmentation 윤곽 + 투영한 body origin + wrist depth 를 back-project 해 head 위에 얹은 점), wrist RGB(반대 방향 overlay), policy 가 실제로 받은 224×224 image(SL0 `policy_records` step_9), head depth(OpenCV row). 아래: 두 camera 를 `robot0_base` frame 으로 back-project 한 3D point cloud 와 camera frustum, table top z.

![SL1 scene](figures/sl1-scene.png)

**graph** — 왼쪽 위부터 #1 replay 차이(전부 0), #2 물체별 median 거리 막대와 전체 CDF(세 K 규약), #4 wrist -> head nearest neighbour CDF, depth sample 위치 격자(`|z_ray - depth| < 0.01 mm` 인 pixel 비율), capture instant 별 robot 표면 거리 CDF.

![SL1 graph](figures/sl1-graph.png)

**table** — 변환 규약 · K · T 와 #1~#7 결과를 한 장에 모은 표.

![SL1 table](figures/sl1-table.png)

### 결과 — #1 replay 재현성 (depth 활성화가 policy 입력을 바꾸지 않는가)

| 비교 대상 (SL0 원 기록 vs) | episode npz SHA-256 | video SHA-256 | 비교 key 수 / 최대 절대 차이 | done step | jsonl |
|---|---|---|---|---|---|
| run a (depth on) | 동일 | 동일 | 14 / 0.0 | 112 | success, env step 113 (대기 제외 103), policy 호출 21 |
| run b (depth + segmentation) | 동일 | 동일 | 14 / 0.0 | 112 | 동일 |
| run c (flag 없음) | 동일 | 동일 | 14 / 0.0 | 112 | 동일 |

- episode npz SHA-256 은 네 run 모두 `e8a15133b8e2ae4aa9c069532d6eca9eba96ed453b070f1b3523a8c48aaaa088`, mp4 SHA-256 은 `9ce6114fb5d99dc57549b28c54a6e047d9b967f4b15b381584c88899e7036f8a` 다. 즉 **RGB · eef/joint/gripper state · action · video 가 byte 단위로 같다.**
- run a 와 run b 의 geometry npz 는 공통 32 key 가 모두 같고, run b 에만 `segmentation_geom/head`, `segmentation_geom/left_wrist` 가 더 있다.
- SL0 의 policy image(21 호출 x image · wrist)는 기록된 raw 에 client 의 변환을 다시 적용한 값과 21/21 로 일치한다.

따라서 `camera_depths=True` 와 `ag3s_capture_state` observable 은 render 와 물리 상태를 바꾸지 않는다. task 문서의 "별도 render pass 로 전환" 조건에는 해당하지 않는다. #7 의 SL0 경로 보존(run c 가 SL0 와 동일)도 이 표로 확인됐다.

### 결과 — #2 depth <-> geom 표면 정합

segmentation 으로 고른 물체 pixel 을 back-project 한 base-frame 점과, 같은 순간(image instant)의 MuJoCo geom 표면 사이의 최근접 거리다. segmentation 과 pose 는 모두 image instant 로 맞췄다(아래 capture instant 의 V1). K 규약 세 가지를 같은 depth 에 적용했다.

| 이름 | principal point (cx, cy; pixel 중심이 정수 index 인 좌표) | 뜻 |
|---|---|---|
| `K_recorded` | (128.0, 128.0) | SL1 adapter 가 기록한 값 (robosuite 와 같은 `W/2`) |
| `K_pixel_centre` | (127.5, 127.5) | pixel 중심 규약. RGB · segmentation 이 sample 되는 위치와 일치 |
| `K_depth_sample` | (127.625, 127.125) | 이 renderer 가 depth 를 sample 하는 위치 (아래 (c)) |

**head (agentview)** — median / p95 (mm)

| 대상 | n (pixel) | `K_recorded` | `K_pixel_centre` | `K_depth_sample` |
|---|---|---|---|---|
| table | 3,153,421 | 2.00 / 2.85 | 0.90 / 1.28 | 0.18 / 0.41 |
| bowl_1 (target) | 75,545 | 1.20 / 2.56 | 0.54 / 1.17 | 0.16 / 0.44 |
| bowl_2 | 54,013 | 1.14 / 2.70 | 0.52 / 1.25 | 0.16 / 0.46 |
| plate | 148,629 | 2.03 / 2.57 | 0.86 / 1.19 | 0.19 / 0.39 |
| ramekin | 18,973 | 1.62 / 2.79 | 0.77 / 1.31 | 0.15 / 0.40 |
| cookies | 58,839 | 1.93 / 2.30 | 0.80 / 1.17 | 0.17 / 0.36 |
| cabinet | 943,236 | 1.56 / 2.03 | 0.56 / 0.97 | 0.13 / 0.28 |
| stove | 179,550 | 2.39 / 2.83 | 1.02 / 1.35 | 0.16 / 0.34 |
| robot | 854,825 | 1.26 / 2.79 | 0.58 / 1.34 | 0.17 / 0.42 |

**left_wrist (robot0_eye_in_hand)** — median / p95 (mm)

| 대상 | n (pixel) | `K_recorded` | `K_pixel_centre` | `K_depth_sample` |
|---|---|---|---|---|
| table | 3,854,155 | 0.25 / 0.51 | 0.25 / 0.49 | 0.25 / 0.47 |
| bowl_1 (target) | 642,523 | 0.33 / 0.99 | 0.23 / 0.56 | 0.18 / 0.46 |
| bowl_2 | 1,212 | 0.39 / 1.18 | 0.33 / 0.67 | 0.09 / 0.42 |
| plate | 924,562 | 0.28 / 0.74 | 0.25 / 0.55 | 0.25 / 0.49 |
| ramekin | 43,746 | 0.49 / 1.82 | 0.34 / 0.87 | 0.17 / 0.47 |
| cookies | 255,586 | 0.26 / 0.58 | 0.24 / 0.49 | 0.22 / 0.47 |
| cabinet | 130,177 | 1.12 / 1.58 | 0.28 / 0.89 | 0.12 / 0.33 |
| stove | 150,097 | 0.28 / 1.22 | 0.26 / 0.52 | 0.23 / 0.47 |
| robot | 769,260 | 0.36 / 0.56 | 0.19 / 0.39 | 0.15 / 0.31 |

wrist depth 가 0.05 m 보다 가까운 pixel(전체의 8.2 %)은 AG3S `backproject` 에서 빠지므로 이 표와 #4 에 들어가지 않는다(아래 #6).

**table plane 잔차** (114 frame 의 median)

| camera | K | plane fit RMS | tilt | z - table top 평균 | z - table top RMS |
|---|---|---|---|---|---|
| head | `K_recorded` | 0.226 | 0.104 deg | +2.001 | 2.048 |
| head | `K_pixel_centre` | 0.226 | 0.048 deg | +0.836 | 0.883 |
| head | `K_depth_sample` | 0.226 | 0.006 deg | -0.038 | 0.230 |
| left_wrist | `K_recorded` | 0.286 | 0.002 deg | +0.025 | 0.295 |
| left_wrist | `K_pixel_centre` | 0.286 | 0.002 deg | +0.011 | 0.288 |
| left_wrist | `K_depth_sample` | 0.286 | 0.001 deg | -0.000 | 0.286 |

읽는 법.

- **depth row 순서와 base frame 정의는 맞다.** table plane 이 평평하게(fit RMS 0.2 mm 대) 복원되고 z 가 table top 높이와 mm 안에서 맞는다.
- head 의 오차는 거의 전부 **principal point 규약 하나**에서 온다. table median 이 2.00 -> 0.90 -> 0.18 mm 로 줄고, plane fit RMS 는 그대로 0.226 mm 라 plane 의 모양(기하)은 이미 맞고 평행이동·기울기만 남는 형태다. wrist 는 camera 가 table 에 가까워 K 규약의 영향이 작다(table 0.25 mm 로 동일).
- **표면까지의 거리가 5 mm(V11AB fine layer voxel)를 넘는 pixel 비율**은 `K_recorded` 기준 head table 0.40 %, bowl_1 1.17 %, plate 0.87 %, ramekin 2.10 %, robot 1.06 %; wrist table 0.35 %, bowl_1 0.30 %, plate 0.46 %, ramekin 0.98 %. 20 mm(coarse layer voxel)를 넘는 비율은 head table 0.35 %, ramekin 1.94 %, robot 0.90 %, wrist table 0.19 %. 이 꼬리는 K 규약을 바꿔도 거의 같다(head ramekin 2.10 % / 1.94 % 가 세 K 모두 같다). 이 꼬리의 원인은 분리하지 않았다.
- 1-pixel 침식(경계 pixel 제외)으로 median 이 거의 안 변한다(head table 2.00 -> 1.99 mm). 경계 pixel 이 median 을 끌고 있지 않다.
- 독립 방법(`mj_multiRay` 로 쏜 ray 와 depth 의 차이, 일치한 pixel 만)도 같은 순서다: head table median 2.72 -> 1.18 -> 0.25 mm, wrist table 1.42 -> 0.68 -> 0.29 mm (`K_recorded` -> `K_pixel_centre` -> `K_depth_sample`).

**결론: 세 K 규약 모두 median · p95 가 5 mm(fine voxel) 아래이므로 task 의 "오차가 voxel 이상이면 SL-2 로 가지 않는다" 조건에는 걸리지 않는다.** 규약 선택의 근거는 아래 판정에 있다.

### 결과 — #3 policy image <-> OpenCV pixel 대응

**policy 입력이 OpenCV image 의 무엇인가.**

| 측정 | 결과 |
|---|---|
| `policy256 == opencv[:, ::-1]` (열 반전 = left-right mirror, 행 순서 그대로) | 228 / 228 image |
| `policy256 == opencv rot180` (180° 회전) | 0 / 228 |
| 점 하나를 policy 변환에 통과시킨 dot test (81 점, sigma 2 px) | 좌표 오차 최대 0.0246 px, `policy_to_opencv_pixel` 왕복 오차 0 |
| 방향 단서 (head, frame 0) | base +y 로 5 cm 가면 OpenCV du = +16.6, dv = 0.0 px; base +z 로 5 cm 가면 du = +0.69, dv = -11.50 px |
| 방향 단서 (wrist, frame 0) | base +y 5 cm: du = -35.40, dv = +1.17 px; base +z 5 cm: du = +4.62, dv = -11.06 px |
| 물체 삼각형 방향 (OpenCV pinhole 예측과 일치 / policy 예측과 일치) | head 4814 / 5073 vs 259 / 5073 (114 frame); wrist 1565 / 1566 vs 1 / 1566 |
| policy 좌표 규칙 | `policy224 (dx, dy) = 0.875 * (-du, +dv)` (`policy256[i, j] = opencv[i, 255 - j]`, 256 -> 224 scale 0.875, pad 0) |

예: head frame 0 에서 target bowl_1 의 mask centroid 는 OpenCV (194.39, 131.63), policy 224 image 에서 (52.97, 115.11) 이다. plate 는 OpenCV (201.15, 166.87) -> policy (47.06, 145.95). OpenCV 에서 오른쪽에 있던 물체가 policy 에서는 왼쪽에 보이고 세로 위치는 같은 비율로 유지된다.

**투영이 물체에 떨어지는가.** 물체의 body origin(MuJoCo body 기준점)을 K · `T_base_cam` 으로 투영한 점이 OpenCV segmentation 에서 그 물체 mask 안에 있는지(hit)다. 이 hit 는 body origin 한 점의 검사다.

| 대상 | head hit / visible (`K_recorded`) | wrist hit / visible (`K_recorded`) |
|---|---|---|
| bowl_1 (target) | 114 / 114 | 87 / 114 |
| bowl_2 | 96 / 114 | 0 / 0 (시야 밖) |
| plate | 113 / 114 | 102 / 113 |
| ramekin | 44 / 73 | 25 / 30 |
| cookies | 114 / 114 | 89 / 89 |
| cabinet | 114 / 114 | 19 / 19 |
| stove | 114 / 114 | 0 / 19 |
| **합계** | `K_recorded` 709 / 757, `K_pixel_centre` 712 / 757, `K_depth_sample` 713 / 757 | 322 / 384, 324 / 384, 325 / 384 |

hit 가 아닌 경우(예: wrist 의 stove 0/19)는 body origin 점이 다른 물체나 robot 에 가려지거나 물체 바깥에 있어서일 수 있으나, 원인은 분리하지 않았다. 투영이 mask 에서 어긋나는 크기를 보려면 아래 mask bbox 중심 오차가 더 직접적이다.

**mesh bbox 중심 - mask bbox 중심** (median du, dv / |오차| median / p95, pixel)

| camera | n | `K_recorded` | `K_pixel_centre` | `K_depth_sample` |
|---|---|---|---|---|
| head | 626 | (+0.32, +0.73) / 1.12 / 7.93 | (-0.18, +0.23) / 0.43 / 8.33 | (-0.05, -0.14) / 0.66 / 8.16 |
| left_wrist | 156 (`K_depth_sample` 은 155) | (+0.48, +0.58) / 0.84 / 20.85 | (-0.02, +0.08) / 0.33 / 17.96 | (+0.10, -0.30) / 0.50 / 17.75 |

p95 는 occlusion 을 보정하지 않은 값이라 가려진 물체를 포함한다. median 은 `K_pixel_centre` 가 가장 작다(head 0.43 px, wrist 0.33 px).

### 결과 — #4 두 camera 일관성

table + plate 의 static 표면에서 wrist 점 -> head 점 nearest neighbour 거리다(n = 4,778,717, 두 표면 합). `all_covisible` 은 verify.json 이 따로 제시한 부분집합으로, 두 camera 가 모두 보는 점으로 좁힌 값이다(n 은 K 별로 4,420,513 / 4,425,143 / 4,429,904).

| 대상 | 지표 | `K_recorded` | `K_pixel_centre` | `K_depth_sample` |
|---|---|---|---|---|
| all | median / p95 (mm) | 2.46 / 10.69 | 1.66 / 9.41 | 1.43 / 8.69 |
| all | 5 mm 초과 / 20 mm 초과 | 6.80 % / 3.33 % | 6.34 % / 3.22 % | 6.14 % / 3.14 % |
| all_covisible | median / p95 (mm) | 2.41 / 3.41 | 1.61 / 2.55 | 1.38 / 2.34 |
| all_covisible | 5 mm 초과 / 20 mm 초과 | 0.23 % / 0.15 % | 0.23 % / 0.16 % | 0.25 % / 0.15 % |

- 같은 table 위 점의 간격 자체가 head 2.82 mm, wrist 1.40 mm(median) 이므로, nearest 거리의 median 이 1.4~2.5 mm 인 것은 두 cloud 의 sampling 간격 수준이다.
- all 과 all_covisible 의 큰 차이(p95 10.69 -> 3.41 mm, 5 mm 초과 6.80 % -> 0.23 %)는 상대 camera 가 보지 않는 영역의 점 때문이다 (해석. 두 집합의 정의 외에 원인 분석은 하지 않았다).

### 결과 — #5 wire 왕복 (Python 3.8 pack -> 3.11 `unpack_camera_observations`)

| 항목 | head | left_wrist |
|---|---|---|
| unpack 성공 frame | 114 / 114 | 114 / 114 |
| 반환 type | `AG3S.ag3s.types.CameraObservation` (camera id `HEAD`, `LEFT_WRIST`) | 동일 |
| depth 양자화 오차 최대 | 0.499992 mm | 0.5 mm (0.5000000000000004, 부동소수 표현) |
| clip 된 pixel / 비유한 값 | 0 / 0 | 0 / 0 |
| K · T · robot_state · stamp 최대 절대 차이 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| wire depth dtype | uint16 | uint16 |

실패 0 (114 file, 228 camera frame). `stamp_mode` 는 `sim_frozen`. 양자화 오차는 0.5 mm 한계(uint16 mm 반올림)를 넘지 않는다.

### 결과 — capture instant (image 가 찍힌 순간)

task 에 없던 측정이다. robosuite 는 render 를 physics substep 안에서 하고, 그 substep 의 상태 적분은 render 뒤에 끝난다. 그래서 `env.step` 이 돌려주는 joint 값과 그 시점의 `sim_state` 는 image 보다 **한 substep(0.002 s) 뒤**다. verifier 는 image 순간을 `P = forward(integratePos(sim_state.qpos, qvel, -dt))` 로 정의하고 re-render 로 검증했다.

| 검증 | 결과 |
|---|---|
| P 에서 re-render 한 RGB · depth 가 기록과 일치 | head 114 / 114, left_wrist 114 / 114 (RGB, depth 모두 bit-identical) |
| 기록한 `T_world_cam` · `T_world_base` 와 P 의 차이 | 0.0 (head, wrist, base) |
| `sim_state`(1 substep 뒤)에서 re-render 한 image 와 기록의 차이 | frame 0 은 같고(아무것도 움직이기 전), frame 10 / 11 / 40 / 80 / 113 에서는 다르다. 예: frame 40 wrist RGB 21,446 pixel, depth 최대 47.8 mm; head RGB 3,486 pixel |
| `sim_state_at_return - sim_state` 시각 | 0 s 또는 0.002 s (`{0.0, 0.002}`) |
| `robot_state` 가 obs 의 `robot0_joint_pos` / `gripper_qpos` 와 차이 | 0.0 (bit 일치) |
| `robot_state` 와 image 순간 qpos 의 차이, 최대 | 0.0017 rad (두 camera 동일). joint 별 [0.0003, 0.0017, 0.0003, 0.0017, 0.0007, 0.0017, 0.0009, 0.0002, 0.0002] rad (joint 7 + gripper 2) |
| 기록한 segmentation 이 어느 순간의 것인가 | `sim_state` 에서 re-render 한 값과 114 / 114 일치 (image 순간보다 1 substep 뒤) |
| 기록 segmentation vs image 순간 segmentation | frame 별 다른 pixel 수 head median 81.5 · max 179 (같은 frame 4 개), wrist median 82.0 · max 316 (같은 frame 1 개) (256 x 256 = 65,536 pixel 중) |

segmentation / pose 의 순간이 표면 거리에 미치는 영향(`K_recorded`, median / p95, mm). V1 = segmentation 과 pose 모두 image 순간, V2 = 기록 segmentation + image 순간 pose, V3 = 기록 segmentation + `sim_state` pose, V4 = 기록 segmentation + `env.step` 반환 순간 pose.

| 대상 | V1 | V2 | V3 | V4 |
|---|---|---|---|---|
| head robot | 1.26 / 2.79 | 1.26 / 2.80 | 1.32 / 2.82 | 1.37 / 2.92 |
| head bowl_1 | 1.20 / 2.56 | 1.20 / 2.57 | 1.20 / 2.64 | 1.20 / 2.72 |
| wrist robot | 0.36 / 0.56 | 0.36 / 0.56 | 0.27 / 0.65 | 0.37 / 0.86 |
| wrist bowl_1 | 0.33 / 0.99 | 0.33 / 1.00 | 0.33 / 1.02 | 0.38 / 1.13 |

순간이 어긋나도 median 은 0.1 mm 안팎으로 변하고, wrist robot p95 가 0.56 -> 0.86 mm 로 가장 크다. 이 STEP 의 #2 수치는 모두 V1 이다.

### 결과 — #6 V11AB 범위 정보 (측정만, SL-4 ESDF box 판정 자료)

| 항목 | 값 |
|---|---|
| robot base 원점 world z | 0.912 m (`T_world_base` 평행이동 (-0.66, 0, 0.912), 에피소드 내 최대 변화 0) |
| table top z (world / base frame) | 0.9 m / **-0.012 m** (base 원점보다 12 mm 낮다) |
| table mesh bbox (base frame) | z -0.062 ~ -0.012 m, x 0.16 ~ 1.16, y -0.6 ~ 0.6 |
| 물체 높이 (base frame z 범위, frame 113: 놓인 뒤) | bowl_1 [-0.0043, 0.061] (plate 위), bowl_2 [-0.0136, 0.039], plate [-0.0091, 0.0061], ramekin [-0.0127, 0.0307], cookies [-0.0127, 0.0074] m |
| 고정 가구 (base frame z 범위) | cabinet [-0.0024, 0.2146], stove [-0.027, 0.0469] m |
| frame 0 의 물체 z 범위 | bowl_1 / bowl_2 / plate / ramekin 의 z 최소 0.058 m, cookies 0.048 m (frame 113 과 다르다. 원인은 분리하지 않았다) |
| 점 cloud 의 base frame z, `K_depth_sample`, 5 frame 간격 (head) | p50 -0.0119, min -0.7908, max 0.5008 m; z < 0 인 비율 **68.84 %**, z > 1.6 m 인 비율 0 |
| 같은 지표 (wrist) | p50 -0.0116, min -0.0125, max 0.3147 m; z < 0 비율 **69.28 %**, z > 1.6 m 0 |
| depth 범위 밖 pixel, head | 0.05 m 미만 0 %, **3.0 m 초과 2.53 %**, depth 최소 0.63 m, 최대 3.069 m |
| depth 범위 밖 pixel, wrist | **0.05 m 미만 8.2 %**, 3.0 m 초과 0 %, depth 최소 0.042 m, 최대 0.397 m |
| depth 가 0 인 pixel | 두 camera 모두 0 % |

V11AB 의 depth 유효 범위는 0.05 ~ 3.0 m, 기본 ESDF box 는 base frame z ∈ [0, 1.6] 이다(`AG3S/ag3s/config.py` · `fields/esdf.py`, task 문서의 audit). 이 값들은 아래 (d), (e) 의 근거다.

### 결과 — #7 SL0 경로 보존 · 회귀

| 검사 | 결과 |
|---|---|
| flag 없는 replay(run c) = SL0 | episode npz · video SHA-256 동일 (#1 표) |
| `run_cpu_regression.sh` A (SL0 5 파일, `.venv-openpi-live`) | **73 passed** (SL0 의 73 과 같다) |
| 같은 script B (`tests/benchmark`, `.venv-openpi-live`) | 44 passed |
| 같은 script C (`tests/benchmark`, `.venv-ag3s`) | 43 passed, 1 skipped (serve_policy test, `.venv-ag3s` 에 flax 없음) |
| script exit status | 0 |
| 16D legacy ESDF 기준선 | violated start 14/15 · has_target 15/15 · frame0 clearance -29.031 mm — 기준값과 일치 |
| 16D cuRobo ESDF 기준선 | 이 STEP 에서는 돌리지 않았다 (GPU 미사용 지시) |

### 이 STEP 에서 알게 된 것 (평이한 설명)

**(a) 공식 policy 전처리는 OpenCV image 를 180° 회전시키는 것이 아니라 좌우로 뒤집는다.** SL0 기록은 이 변환을 "180° 회전"으로 적었다(`~~`로 표시한 SL0 의 해당 줄). robosuite 의 raw image 는 OpenGL row order(아래가 첫 행)이고 client 는 `[::-1, ::-1]` 로 행과 열을 모두 뒤집는다. 행 뒤집기는 OpenGL -> OpenCV 변환과 상쇄되므로, 결과는 OpenCV image 에서 열만 뒤집은 것(left-right mirror)과 같다(`policy256 == opencv[:, ::-1]` 228/228, rot180 0/228). **policy 입력은 바꾸지 않았다.** 바뀌는 것은 읽는 쪽이다: SL-3 에서 attention token 위치를 depth pixel 로 옮길 때 `policy224 (dx, dy) = 0.875 * (-du, +dv)` 로 좌우를 되돌려야 한다. 그렇지 않으면 attention 이 물체의 좌우가 바뀐 곳에 놓인다.

**(b) `robot_state` 가 image 보다 한 physics substep 늦었다.** robosuite 는 image 를 substep loop 안에서 찍는데, `env.step` 이 돌려주는 joint 값은 그 substep 의 적분이 끝난 뒤의 값이다. 카메라 pose(`T_world_cam`)는 image 순간과 bit 단위로 일치하도록 만들었지만 `robot_state` 는 obs 의 joint 값과 같아서 image 보다 0.002 s 뒤다(최대 0.0017 rad). 이 크기가 표면 거리에 주는 영향은 작지만(위 V1~V4), self-filter 가 robot pose 와 image 를 같은 순간으로 보도록 SL-2 전에 맞춘다.

**(c) depth 는 pixel 안의 다른 위치에서 sample 된다.** 이 환경의 renderer(osmesa llvmpipe, MSAA `offsamples` 4)에서 depth 는 pixel 중심 `(u+0.5, v+0.5)` 가 아니라 `(u+0.375, v+0.875)` 에서 sample 된다. 격자 실험에서 head 의 `|z_ray - depth| < 0.01 mm` 비율은 `(0.375, 0.875)` 에서 최대(0.806 / 0.821 / 0.829, frame 0 / 56 / 113)이고 wrist 는 1.0 이다. pixel 중심 `(0.5, 0.5)` 에서는 `|z_ray - depth|` median 이 head 에서 1.34 / 1.26 / 1.23 mm 였다. 반면 RGB 와 segmentation 은 pixel 중심에서 sample 된다(ray 와 segmentation 의 일치율이 offset 0.5 에서 최대: head 0.969, wrist 0.853~0.861; offset 0.0 에서는 head 0.954). K 의 principal point 를 어디에 두느냐가 head table 에서 2.00 / 0.90 / 0.18 mm 의 차이를 만드는 이유다. 공식 EGL/GPU renderer 에서 depth sample 위치가 같은지는 측정하지 않았다.

**(d) table top 이 base 원점보다 12 mm 아래라, 기본 ESDF box(z >= 0)의 바깥이다.** table top 은 base frame z = -0.012 m 이고 plate · bowl_2 · ramekin · cookies 의 바닥(frame 113)은 -0.009 ~ -0.014 m 다. head 점 cloud 의 68.84 %, wrist 의 69.28 % 가 z < 0 이다. 기본 box 를 그대로 쓰면 table 과 놓인 물체의 아래쪽이 voxel grid 에 들어가지 않는다. box 를 낮출지, table 을 support surface 로 따로 다룰지는 **SL-4 에서 결정한다.**

**(e) wrist 의 가까운 pixel 이 depth 최소(0.05 m) 아래다.** wrist depth 최소는 0.042 m 이고 0.05 m 미만 pixel 이 8.2 % 다. V11AB 의 `backproject` 는 이 pixel 을 버린다(task 문서의 `config.py` 0.05–3.0 m 범위). head 는 3.0 m 를 넘는 pixel 이 2.53 % (최대 3.069 m)이며, 이것도 버려진다. 어떤 물체가 이 영역에 있는지(gripper 가 쥔 물체 등)는 SL-2 · SL-3 에서 본다.

### 판정 (2026-10-12, 사용자)

| # | 판정 |
|---|---|
| 1 | **SL1 확정.** |
| 2 | **K 의 principal point 를 pixel 중심 규약 `(W-1)/2 = 127.5` 로 바꾼다.** adapter 의 calibration 만 바꾸는 것이며 V11AB algorithm 은 바꾸지 않는다. 바뀐 규약은 manifest 에 기록한다. |
| 3 | **MSAA sample 위치의 K(`K_depth_sample`, 127.625 / 127.125)는 채택하지 않는다.** renderer 에 따라 달라지는 값이기 때문이다. |
| 4 | **`robot_state` 를 image 순간 값으로 고친다.** SL-2 전에 반영한다. |

~~위 2, 4 는 **SL2 에서 구현한다. 아직 구현도 측정도 되지 않았다.** 이 절의 `K_pixel_centre` 열은 현재 기록한 depth 에 오프라인으로 K 만 바꿔 적용해 본 값이며, SL2 구현 후의 adapter 출력을 다시 재야 한다. 그 값이 위 기대(head table median 약 0.9 mm, table plane 평균 +0.84 mm)와 같을지는 그때 확인한다.~~
→ **SL2 에서 구현했고 새 replay 로 측정했다.** 판정 2(K = 127.5) 와 판정 4(`robot_state` = image 순간)의 결과는 아래 [SL2 의 "SL1 후속 2건"](#결과--sl1-후속-2건-구현-후-재측정)에 있다. `K_pixel_centre` 열과 같은 값이 adapter 출력에서 나왔다. 이 절의 `K_recorded` 열과 "capture instant" 절의 `robot_state` 수치는 SL1 시점의 기록이다.

### 측정하지 않은 것

- 공식 EGL/GPU renderer 에서의 depth sample 위치 (측정은 osmesa llvmpipe, Mesa 23.2.1, `offsamples` 4 에서만)
- 다른 task · episode, 다른 해상도 (이 STEP 은 libero_spatial task 0 episode 0 의 replay 하나)
- 16D cuRobo 기준선 (GPU 미사용)
- `mj_multiRay` 가 gripper hand visual 과 plate 일부 pixel 을 통과해(ray geom != segmentation) ray 기반 수치는 일치한 pixel 만 쓴 값이다
- mask bbox 중심 오차의 occlusion 보정 (p95 는 occlusion 포함)
- #2 표면 거리의 5 mm 초과 꼬리(0.3~2.1 %)의 원인
- live policy server 와 `--capture-geometry` 를 함께 돌린 실행 (client 와 replay 가 같은 `run_tasks` 를 쓰므로 replay 로 실행됐다. server 와의 live 조합은 측정하지 않았다)
- 새로 추가한 manifest `n/a` 항목 · `runtime_schedule` · `serve_policy` 기록을 **실제 policy server 가 있는 launcher 실행**에서 확인한 것 (unit test 로만 확인: B 44 passed)

### 열린 항목

| # | 내용 | 영향 / 다음 |
|---|---|---|
| 1 | ~~SL2 에서 K 를 127.5 로, `robot_state` 를 image 순간으로 고친 뒤 #2 · capture instant 를 재측정~~ | ~~판정 2 · 4~~ → SL2 에서 완료 |
| 2 | ESDF box z 범위 · table 처리 | SL-4 에서 결정 ((d)) |
| 3 | SL-3 attention -> depth 사상에서 좌우 되돌리기 | `opencv_to_policy_pixel` / `policy_to_opencv_pixel` 사용 ((a)) |
| 4 | wrist 0.05 m 미만 pixel 과 head 3.0 m 초과 pixel 의 처리 | SL-2 · SL-3 ((e)) |
| 5 | segmentation 은 image 순간보다 1 substep 뒤에 render 된다 | image 순간 id 가 필요하면 `sim_state` 로 별도 env 에서 재생성(구현 차이 #4). V11AB 가 segmentation 을 쓰는 지점이 정해지면 판단 |
| 6 | 렌더러가 EGL 이면 depth sample 위치가 다를 수 있다 | 판정 3(MSAA K 비채택)의 근거. EGL 환경을 쓰게 되면 재측정 |

### 되돌아올 지점 (SL1)

| 전환 신호 | 돌아와서 할 일 |
|---|---|
| #1 에서 RGB 가 달라졌다면 | geometry 를 policy 와 별도 render pass 로 얻는 방식으로 전환하고 사용자에게 보고 (이번에는 bit-identical 이라 해당 없음) |
| #2 오차가 voxel 이상이었다면 | near/far · fovy · base pose 정의를 재점검, 원인 확정 전 SL-2 로 가지 않음 (이번에는 p95 가 2.85 mm 이하라 해당 없음) |
| SL-3 token 순서 audit 이 `left_wrist` slot 과 다르다 | adapter 의 camera 이름 table 만 변경 |
| 공식 EGL renderer 를 쓰게 된다 | depth sample 위치와 K 규약 재측정 (판정 3) |

---

## SL2 — Panda geometry: sphere chain · self-filter · contact link, SL1 후속 2건 (2026-10-12)

> 2026-10-12. 구현 commit `7ce24d9f` · 검증 commit `9b51a3c4` (git author date 2026-10-12 04:57 / 05:33 +0000). 사용자 확정 2026-10-12.
> 원본: [`handoff/SL2.task.md`](handoff/SL2.task.md) · [`handoff/SL2.impl.md`](handoff/SL2.impl.md) ·
> [`handoff/SL2.verify.json`](handoff/SL2.verify.json). raw 출력은 `outputs/verify/SL2/`.
> 이 절의 수치는 `SL2.verify.json` 의 `numbers` 가 출처다. 구현 단계에서만 나온 값(URDF 보정 전 오차 등)은 그 자리에서 "impl.md" 로 표시한다.
> 단위는 따로 적지 않으면 mm 이고, task 0 · episode 0 의 replay 114 observation 위의 값이다. frame 번호는 observation index(0–113)다.

### 이 STEP 이 답하는 질문

V11AB 와 **같은 구조**(self-filter 는 전신 + 덮개, constraint 는 gripper/arm)의 Panda sphere model 이 LIBERO simulator 의 실제 Panda 위치·형상과 맞아서,
**robot 점은 지우고 target 은 지우지 않으며, 실제 접촉을 sphere 침투로 나타내는가?** 그리고 SL1 에서 판정한 후속 2건(K principal point, `robot_state` image 순간)이 구현되었는가?

```
robot_state(q7 + finger2, image 순간) --FK--> Panda UrdfSphereChain (adapter URDF)
        |-- self-filter model (franka.yml 61 sphere + 덮개 capsule)  -> DepthRobotMask -> depth 에서 robot 점 제거
        `-- constraint model  (franka.yml 61 sphere, link_filter)    -> clearance constraint (SL-4/5 에서 연결)
```

Action 변환(SL-5), attention(SL-3), ESDF 연결(SL-4)은 이번 범위가 아니다.

### 무엇을 만들었나

**SL1 후속 2건 (camera adapter).**

| 항목 | 내용 |
|---|---|
| K principal point | `pixel_centre` 규약 `((W-1)/2, (H-1)/2) = (127.5, 127.5)`. 규약 이름과 값을 `conventions()` · manifest 에 기록 (`K_convention`). robosuite 의 `intrinsic_matrix` 는 비교 기준으로만 남겼다 |
| `robot_state` = image 순간 | geometry capture 의 observable 이 같은 physics substep 에서 두 후보(`mj_integratePos(qpos, qvel, -dt)` 와 현재 `qpos`)를 scratch `MjData` 에서 다시 풀어 render 가 본 kinematics 를 재현하는 쪽을 고른다. 어느 쪽도 1e-9 m 안에 재현하지 못하면 capture 가 예외를 낸다. 기록된 `sim_state` 는 그 substep 의 simulator 상태라는 원래 뜻 그대로 두고, `sim_state_image` 를 새로 추가했다 |
| `_geom.npz` 추가 key | `sim_state_image`, `qpos_image_source`, `qpos_image_residual_m` (검증용) |

**Panda robot model (`AG3S/adapters/panda/`, 새 파일. `AG3S/ag3s/**` algorithm 은 한 줄도 바꾸지 않았다).**

| 구성 | 내용 |
|---|---|
| adapter URDF (`robots/panda/description/panda_libero.urdf`) | cuRobo `franka_panda.urdf` 의 팔 관절 7 개를 그대로 쓰고, **손과 손가락 장착 offset 3 값만 robosuite(LIBERO `MountedPanda` + `PandaGripper`) 값으로 고쳤다**: `panda_joint8` z 0.107 → 0.1065, `panda_hand_joint` yaw −0.785398 → −0.7858908 (robot.xml quat 의 `2·atan2(−0.383, 0.924)`), finger joint z 0.0584 → 0.0524. root = `panda_link0` (= `robot0_base`). 보정 전 cuRobo URDF 는 4 고정 q 에서 팔(link0–7) 0.000 mm, hand 0.5 mm · 0.028°, finger 6.5 mm 어긋났다(impl.md 의 test 값). hand · finger collision mesh 의 bbox 가 두 쪽 link frame 에서 같아서 mesh 가 아닌 frame offset 의 문제였다. `panda_grip_site` · `panda_eye_in_hand`(OpenCV 축) fixed link 를 추가했다 |
| finger joint 대응 | `robot_state[7]` → `panda_finger_joint1` (+), `robot_state[8]` → `panda_finger_joint2` (부호 반전: MuJoCo `finger_joint2` 는 +y 축 · range [−0.04, 0], URDF 는 −y 축 · [0, 0.04]). finger 개도는 V11AB 의 T29 parameter joint 로 받는다 |
| constraint model | adapter URDF 위에 cuRobo `franka.yml` collision sphere 61 개를 zero-length `UrdfCapsule`(role `collision`)로 넣었다 |
| self-filter model | 위 61 개 + robosuite **collision mesh**(group 0) 에서 `franka.yml` sphere 가 덮지 못한 표면을 **V11AB 의 `self_filter_covering_capsules` 를 그대로 호출**해 맞춘 덮개 capsule(role `self_filter_cover`). 덮개는 constraint model 에 넣지 않는다 |
| 덮개 대상 body | Panda link0–7, hand, 손가락 + pad body, mount 3 개(`controller_box`, `pedestal_feet`, `pedestal`). SL1 segmentation 에서 head camera 가 `mount0_pedestal_vis` · `mount0_torso_vis` 를 보기 때문이다. RB-Y1 이 base 를 덮은 것과 같은 이유 |
| robosuite 모델 재구성 | robosuite 는 Python 3.8 쪽에만 있어서, 3.8 exporter(`robot_geometry.py`)가 robot subtree 의 body · joint · geom · mesh 를 npz 로 내고 3.11 쪽(`cover.py`)이 `MjSpec` 모델을 다시 짓는다. body 이름을 URDF link 이름으로 바꿔 두어 V11AB 함수의 `static_urdf_ancestor` 가 RB-Y1 에서처럼 link 를 찾는다 |
| 덮개 JSON | `libero_panda_self_filter_cover.json` — 51 capsule, provenance(geometry · kinematic digest, URDF · `franka.yml` SHA-256) 포함. confined variant 는 `cover.py --confine` 으로 만든다 |
| builder | `build_panda_self_filter_model()`, `build_panda_constraint_model(link_filter=...)`, `panda_depth_robot_mask(...)`(V11AB `backproject` + `depth_robot_mask` 연결) |

**link 집합 대응 (RB-Y1 ↔ Panda).** V11AB 의 link 집합 이름(`grounding_report`/`config`)을 같은 역할로 Panda 에 옮겼다. 오타 · 빈 집합 · 모르는 그룹은 `ValueError` 다.

| 역할 | RB-Y1 | Panda |
|---|---|---|
| `PALM_LINKS` | `ee_left`, `ee_right` | `panda_hand` |
| `FINGER_LINKS` | `ee_finger_{l1,l2,r1,r2}` | `panda_leftfinger`, `panda_rightfinger` |
| `GRIPPER_LINKS` (`--links gripper`, V11AB 기본) | PALM + FINGER (6) | PALM + FINGER (3 link, 22 sphere) |
| `ARM_LINKS` (`--links arms`) | `link_{left,right}_arm_0..6` + GRIPPER | `panda_link1..7` + GRIPPER (10 link, 59 sphere) |
| 결정 변수 없는 link (filter 만) | base, wheels, torso, head | `panda_link0`, mount (덮개만) |
| `DEFAULT_CONTACT_LINKS[side]` | left: `ee_left, ee_finger_l1, l2` / right: 대응 | left: `panda_hand, panda_leftfinger, panda_rightfinger` |

**Manipulator side = `left`.** Panda 단일 팔을 `left` 로 둔다. SL1 에서 wrist camera 를 pi05_libero 의 token slot 에 맞춰 `left_wrist` 로 정했으므로 camera 가 달린 팔과 contact link 의 side 가 일치해야 한다. V11AB 16-D layout `[L7, Lg, R7, Rg]` 의 L 슬롯에 대응한다(SL-5 에서 사용). camera.py · model.py · manifest 두 곳에 기록했다. SL-3/SL-5 의 camera↔arm audit 결과가 다르면 adapter 표만 바꾼다.

V11AB server 연결(`serve_safe.build_ag3s`)은 하지 않았다. RB-Y1 hardcode 지점 목록은 impl.md 에 있다(아래 "열린 항목" 5).

### 실행

| 항목 | 값 |
|---|---|
| source | SL0 run `20261012T045129_6ba81024d243_sl0-pi05only-t0e0` (읽기만) |
| replay + segmentation (측정 대상) | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T135846_7ce24d9ff7aa_sl2-replay-seg-t0e0` (`--capture-geometry --capture-float-depth --capture-segmentation`) |
| replay flag 없음 (SL0 경로 보존) | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T135938_7ce24d9ff7aa_sl2-replay-off-t0e0` |
| robot geometry ground truth | `outputs/verify/SL2/raw/geo_libero.npz` (실제 LIBERO env 의 robot subtree export), `fk_libero.npz` |
| 환경 | GPU · policy server 불필요. launcher `.venv-ag3s` → replay `.venv-libero`. cgroup working set 최대 57.357 GiB (다른 checkout 의 process 가 동시에 돌고 있었다: development checkout 의 pytest, `t47_head_selection`) |
| 명령 편차 | impl.md 가 준 `robot_geometry.py` 명령은 이 shell 에서 실패했다(`env.sh`/pod 의 `LD_LIBRARY_PATH` 가 `/usr/local/lib/python3.10/dist-packages/torch/lib` 를 포함해 `libtorch_python.so` 의 `PyObject_GET_WEAKREFS_LISTPTR` ImportError). verifier 가 `LD_LIBRARY_PATH` 를 비우고 `PYTHONPATH` 를 해제하는 wrapper `libero_env.sh` 로 실행했다. 코드 수정은 SL3 에서 한다(아래 "열린 항목" 1) |

### 결과 — Figure

세 figure 는 verifier 가 만들었다(`outputs/verify/SL2/scripts/`, sidecar `.json` 동봉).

**실제 scene** — frame 92(MuJoCo 가 두 finger pad 모두 target bowl 에 접촉으로 보고하는 frame). 위 줄: robosuite collision mesh(MuJoCo pose) 위에 sphere 를 겹친 정면(base frame y–z)·측면(x–z) 그림. constraint(61), self-filter default(61 + 184 cover = 245), hand close-up 에서 default 대 confined hand(61 + 1,066 cover = 1,127). 가운데 두 줄: wrist · head 에서 self-filter variant 5 개의 제거 결과(파랑 = 제거된 robot/기타, 빨강 = 제거된 target bowl, 초록 윤곽 = target bowl). 아래: head depth 를 back-project 한 point cloud(side view) 와 forbidden-contact fixture F1–F3.

![SL2 scene](figures/sl2-scene.png)

**graph** — FK 위치 오차(frame 별, log 축), robosuite collision mesh coverage, self-filter 의 물체별 제거율, wrist · head 에서 target bowl 제거 비율의 frame 별 추이(초록 띠 = finger pad 접촉, 노랑 띠 = hand 접촉, 점선 = bowl 의 lift), constraint sphere signed distance.

![SL2 graph](figures/sl2-graph.png)

**table** — link 집합 대응 · sphere 출처와 수 · self-filter variant 비교 · contact × penetration confusion table · SL1 후속, FK, 회귀 요약.

![SL2 table](figures/sl2-table.png)

### 결과 — SL1 후속 2건 (구현 후 재측정)

**robot_state 가 image 순간이 되었는가.** adapter URDF 로 FK 한 wrist camera pose 와 기록된 `T_world_cam` 의 차이다.

| 비교 | 위치 최대 | 회전 최대 |
|---|---|---|
| SL2 run: FK(`robot_state`) vs `T_world_cam` | 2.6e-16 m | 6.4e-6 deg |
| SL1 run 에 같은 검사 (`robot_state` = `sim_state.qpos`, 1 substep 늦음) | 0.574 mm (median 0.408 mm) | 0.041 deg |

- 기록된 `robot_state` 와 `sim_state_image` 의 qpos 차이: head · left_wrist 모두 0.0 (최대 절대값). `sim_state_image` 와 `integratePos` 로 만든 qpos 의 차이도 0.0.
- `sim_state_image` 에서 re-render 한 RGB · depth 가 기록과 같은 frame 수: head 114/114, left_wrist 114/114 (RGB · depth 모두). `T_world_cam` · `T_world_base` 차이는 0.0 이하 수준(wrist 3.3e-16, base 0.0).
- 고른 후보: `integrated_back` 113 frame, `current`(현재 qpos) 1 frame (frame 0, 아직 아무것도 움직이기 전).
- 기록된 segmentation 은 여전히 `env.step` 반환 시점 render 라 image 순간과 다르다(SL1 차이 #4 그대로): image 순간 segmentation 과 같은 frame 은 head 4 개, left_wrist 1 개, 다른 pixel 수 합계는 head 8,686 · left_wrist 11,938. 아래 self-filter 측정은 image 순간 segmentation(`segP`)을 쓰고, 기록된 segmentation(`segRec`)으로 낸 값과 함께 확인했다(차이는 0.2 %p 안).

**pixel-centre K 로 SL1 #2 · #4 를 다시 쟀다.** 기록된 K 의 principal point 는 (127.5, 127.5), SL1 run 의 K 와의 차이는 (−0.5, −0.5), 초점거리 차이는 0 이다. 새 adapter 출력의 값이 SL1 의 `K_pixel_centre` 열(depth 에 오프라인으로 K 만 바꿔 적용한 값)과 같다.

| 측정 | SL1 `K_recorded` (128.0) | SL2 adapter 출력 (127.5) | SL1 `K_pixel_centre` 열 |
|---|---|---|---|
| head table 표면 거리 median / p95 | 2.00 / 2.85 | **0.90 / 1.28** | 0.90 / 1.28 |
| head bowl_1 (target) median / p95 | 1.20 / 2.56 | 0.54 / 1.17 | 0.54 / 1.17 |
| head robot median / p95 | 1.26 / 2.79 | 0.58 / 1.34 | 0.58 / 1.34 |
| head stove median / p95 | 2.39 / 2.83 | 1.02 / 1.35 | 1.02 / 1.35 |
| wrist table median / p95 | 0.25 / 0.51 | 0.25 / 0.49 | 0.25 / 0.49 |
| wrist bowl_1 (target) median / p95 | 0.33 / 0.99 | 0.22 / 0.56 | 0.23 / 0.56 |
| head table plane z − table top 평균 | +2.001 | +0.836 | +0.836 |
| head table plane tilt | 0.104 deg | 0.048 deg | 0.048 deg |
| cross-camera nearest (all) median / p95 | 2.46 / 10.69 | 1.66 / 9.41 | 1.66 / 9.41 |
| cross-camera nearest (all_covisible) median / p95 | 2.41 / 3.41 | 1.61 / 2.54 | 1.61 / 2.55 |

wrist 쪽은 camera 가 table 에 가까워 K 규약의 영향이 작다(table plane z − top 평균 +0.011 mm, SL1 표와 같다). 물체별 전체 값은 `SL2.verify.json` `n1b_pixel_centre_K_remeasure` 에 있다. 이로써 SL1 판정 2(K = 127.5) 와 4(`robot_state` = image 순간)는 구현되었고 측정되었다.

### 결과 — #2 FK 일치 (adapter URDF vs LIBERO MuJoCo)

adapter URDF 의 FK 를 같은 q 에서 robosuite MuJoCo body pose 와 비교했다(`robot0_base` frame). 비교 link 는 link1–7 · hand · leftfinger · rightfinger(+ camera frame `panda_eye_in_hand`).

| 비교 | 위치 최대 | 위치 p95 | 회전 최대 | 회전 p95 |
|---|---|---|---|---|
| trajectory 114 frame × 10 link (link1–7, hand, 두 finger) | 4.8e-9 mm | 4.2e-9 mm | 5.6e-10 deg | 5.6e-10 deg |
| 고정 q 4 개 × 12 frame (10 link + `panda_grip_site` + `panda_eye_in_hand`) | 5.1e-9 mm (`panda_grip_site`) | | 8.3e-10 deg (`panda_link6`) | |

목표(≤ 1 mm, ≤ 0.1°)를 float 정밀도까지 만족한다(link 별 표: `n2_fk`). **adapter URDF 의 kinematics 는 LIBERO 의 Panda 와 같다.** 이 값은 위의 hand · finger offset 보정 뒤의 결과이고, 보정 전 cuRobo URDF 의 오차는 impl.md 가 낸 값만 있다(독립 측정 아님).

### 결과 — #3 sphere 출처 · 수 · coverage

**sphere model 별 수.**

| model | 출처 | sphere 수 | 덮개 sphere / capsule |
|---|---|---:|---|
| constraint `gripper` | `franka.yml` (hand 18 + finger 2 + 2) | 22 | — |
| constraint `arms` (builder 기본) | `franka.yml` link1–7 + gripper | 59 | — |
| constraint `all` | `franka.yml` 전부 (11 link, 반지름 11–60 mm, role `collision`) | 61 | 0 |
| self-filter **default** | `franka.yml` 61 + V11AB fitter 덮개 | 245 | 184 / 51 |
| self-filter confined hand | default 에서 hand 를 confined 로 다시 맞춤 | 1,127 | 1,066 / 294 |
| self-filter confined mount | default 에서 mount 를 confined 로 다시 맞춤 | 3,389 | 3,328 / 793 |
| self-filter **confined hand+mount** (Panda 기본으로 채택, 아래 판정) | 둘 다 confined | **4,271** | 4,210 / 1,036 (link0 3,192, hand 911) |

model 의 반지름과 `franka.yml` 반지름의 차이는 최대 0.0 이다. commit 된 덮개 51 capsule 은 이 export 에서 다시 맞추면 동일하고(`committed_cover_refit_from_this_export_identical = true`), 덮개를 맞춘 geometry 의 digest 는 이번 export 의 digest 와 같다.

**constraint model(`franka.yml` sphere)이 robosuite collision mesh 표면을 얼마나 덮는가.** mesh 표면 표본점 중 그 link 자신의 sphere union 안에 든 비율, 밖인 점의 최대 거리, 그리고 전체 sphere union 으로 본 값(19 frame 평균; 이웃 link sphere 가 덮어 주는 몫 포함)이다.

| link | 표본 수 | own-link 안 비율 | 밖 최대 거리 (mm) | 전체 union, q 19 frame 평균 |
|---|---:|---:|---:|---:|
| `panda_link0` | 30,208 | 2.3 % | 99.6 | 10.3 % |
| mount | 931,688 | 0.0 % | 1,095 | 0.0 % |
| `panda_link1` | 33,291 | 73.8 % | 27.3 | 73.8 % |
| `panda_link2` | 29,419 | 28.6 % | 42.5 | 36.3 % |
| `panda_link3` | 24,141 | 33.7 % | 34.3 | 38.3 % |
| `panda_link4` | 24,454 | 22.3 % | 36.7 | 25.8 % |
| `panda_link5` | 33,798 | 13.9 % | 65.2 | 17.4 % |
| `panda_link6` | 17,593 | 12.3 % | 90.3 | 39.1 % |
| `panda_link7` | 8,459 | 41.5 % | 12.2 | 47.2 % |
| `panda_hand` | 12,752 | 13.4 % | 15.6 | 27.2 % |
| `panda_leftfinger` | 1,528 | 23.4 % | 14.2 | 46.6 % |
| `panda_rightfinger` | 1,528 | 23.3 % | 14.2 | 44.8 % |

`franka.yml` sphere 는 몸체의 **일부만** 덮는다(link 별 12–74 %). 이것이 constraint 용으로는 충분한 근사인지는 sphere 가 몸체 밖으로 나간 두께와 contact 측정(아래 #5)으로 본다. 한편 self-filter default 모델(덮개 포함)은 모든 link · mount 의 collision 표면 표본을 100 % 담는다(밖 최대 0.0). visual 표면 표본으로 보면 default 는 모든 link 에서 100 % 이고, confined hand 는 hand 에서 99.76 %(밖 최대 1.16 mm), confined mount 는 mount 의 visual 표면에서 57.8 %(밖 최대 92.7 mm)다. mount 는 collision 으로 맞췄기 때문에 visual 만 있는 표면은 덮이지 않는다.

**sphere union 이 collision hull 밖으로 나간 두께 (protrusion, p95 / 최대, mm).** 값이 클수록 depth 에서 robot 이 아닌 곳을 robot 으로 오인해 지울 위험이 크다.

| link | constraint (`franka.yml`) | self-filter default | confined (hand 는 confined hand, link0 는 confined mount) |
|---|---|---|---|
| `panda_link0` (+ mount) | 23.0 / 28.2 | **215.0 / 249.6** | 61.3 / 129.8 |
| `panda_link1` | 20.6 / 37.7 | 62.8 / 71.1 | 62.8 / 71.1 |
| `panda_link2` | 14.8 / 30.9 | 100.5 / 110.7 | 100.5 / 110.7 |
| `panda_link3` | 22.3 / 38.7 | 92.5 / 101.3 | 92.5 / 101.3 |
| `panda_link4` | 13.7 / 25.6 | 90.2 / 97.9 | 90.2 / 97.9 |
| `panda_link5` | 30.5 / 61.7 | 105.1 / 112.1 | 105.1 / 112.1 |
| `panda_link6` | 2.9 / 8.6 | 72.6 / 82.8 | 72.6 / 82.8 |
| `panda_link7` | 13.4 / 26.8 | 46.6 / 52.7 | 46.6 / 52.7 |
| `panda_hand` | 1.6 / 5.1 | **49.1 / 56.4** | 17.5 / 43.1 |
| `panda_leftfinger` | 3.5 / 5.4 | 22.7 / 24.8 | 22.7 / 24.8 |
| `panda_rightfinger` | 3.5 / 5.5 | 22.8 / 24.8 | 22.8 / 24.8 |

constraint sphere(`franka.yml`)는 hand · finger 에서 몸체 밖으로 거의 나가지 않는다(hand p95 1.6 mm). 반면 self-filter 덮개는 link 몸체보다 수 cm 크게 나간다. 이 차이가 constraint 와 self-filter 를 따로 두는 이유이고, 아래 #4 가 그 대가를 보여 준다.

### 결과 — #4 self-filter (SL1 depth 에 Panda self-filter model 적용)

SL1 의 metric depth(유효 범위 0.05–3.0 m, inflation 0, guard radius 0)에 `DepthRobotMask` 를 적용했다. segmentation 이 정답이다: "robot 제거율" = segmentation 이 robot 인 pixel 중 제거된 비율, "오제거" = robot 이 아닌 물체 pixel 중 제거된 비율. 비교한 variant 는 5 개다.

- **default** — `franka.yml` 61 + V11AB fitter 덮개 51 capsule(245 sphere)
- **confined hand** — hand 덮개만 confined(1,127)
- **confined mount** — mount 덮개만 confined(3,389)
- **confined hand+mount** — 둘 다(4,271)
- **`franka.yml` only** — 덮개 없음(61)

114 frame 합산(removed / labelled valid-depth pixel):

| variant | sphere | robot 제거율 head / wrist | mount head 제거율 | target bowl 오제거 head / wrist | table 오제거 head / wrist | stove 오제거 head / wrist |
|---|---:|---|---|---|---|---|
| default | 245 | 99.23 % / 99.98 % | 99.93 % | 5.46 % / 11.00 % | 5.75 % / 1.62 % | **30.92 % / 11.68 %** |
| confined hand | 1,127 | 99.22 % / 99.98 % | 99.93 % | **1.91 % / 4.30 %** | 5.72 % / 1.41 % | 30.92 % / 11.68 % |
| confined mount | 3,389 | 99.17 % / 99.97 % | 98.70 % | 5.46 % / 11.00 % | **0.74 % / 0.60 %** | **0.00 % / 0.33 %** |
| **confined hand+mount** | 4,271 | 99.17 % / 99.97 % | 98.70 % | **1.91 % / 4.30 %** | **0.72 % / 0.38 %** | **0.00 % / 0.33 %** |
| `franka.yml` only | 61 | **41.86 % / 36.29 %** | 0.05 % | 0.02 % / 0.01 % | 0.00 % / 0.01 % | 0.00 % / 0.06 % |

(pixel 수: head target bowl 75,545, stove 179,550, table 3,153,421; wrist target bowl 642,523, stove 150,097, table 3,854,155. default 의 head stove 는 55,518 pixel 이 제거됐다.)

**target bowl 이 지워지는 frame (target bowl pixel 중 제거된 비율).**

| variant | head 최대 (frame) | wrist 최대 (frame) | grasp 구간 frame 90–113 평균 head / wrist | 제거가 0 보다 큰 frame 수 head / wrist |
|---|---|---|---|---|
| default · confined mount | 62.98 % (83) | 34.24 % (52) | 6.72 % / 7.68 % | 57 / 85 |
| confined hand · confined hand+mount | 18.41 % (83) | 15.53 % (53) | 3.64 % / 5.32 % | 55 / 85 |
| `franka.yml` only | 0.40 % (89) | 0.24 % (41) | 0.04 % / 0.00 % | 14 / 42 |

bowl 이 통째로 지워진 frame 은 없다(최대 62.98 %). default 에서 wrist 의 제거는 frame 49–56(finger 가 bowl 에 접근해 pad 접촉(frame 54)이 일어나는 구간)에 몰리고(frame 51–53 에서 31–34 %), head 의 제거는 frame 82–84(hand 가 bowl 에 접촉하는 구간)에서 61–63 % 로 커진다. 제거된 target pixel 을 가장 가까운 sphere 로 귀속시키면 default 에서 wrist 70,696 pixel 중 `panda_hand` 덮개가 65,135, `panda_rightfinger` 덮개가 5,561 pixel 이고, confined hand 에서는 wrist 27,656 pixel 중 `panda_rightfinger` 덮개 22,973, `panda_hand` 덮개 4,269, `panda_leftfinger` 덮개 414 pixel 이다.

**robot 이 아닌 다른 물체의 오제거는 variant 와 무관하게 작다.** bowl_2 0.51 % (head) · 0.50 % (wrist), ramekin 0.28 % · 0.17 %, plate 0 % · 0.036 %(default) → 0.011 %(confined hand), cookies · cabinet 0 %. 이 비율은 `franka.yml` only 에서는 거의 0 이다. segmentation 이 geom 이 아닌 pixel(seg −1)은 위 비율에서 뺐다: head 49,962 pixel(valid 7,282,370 중), wrist 86,922 pixel(valid 6,858,240 중)이며, 이 중 제거된 pixel 은 default 21,354 / 20,094 → confined hand+mount 18,453 / 3,362 이다(base frame z median 은 head 0.016 m, wrist −0.012 m).

#### 왜 default 덮개가 stove 와 target bowl 을 지웠나

**stove — mount 덮개가 table 위로 뻗어 있었다.** robot 의 mount(pedestal)는 크고 둥근 collision 몸체다. V11AB 의 덮개 fitter 는 `franka.yml` sphere 가 덮지 못한 표면을 **몇 개의 큰 capsule** 로 한꺼번에 덮는다. capsule 은 반지름이 크고 양 끝이 반구라서, 그 끝 반구가 mount 몸체 밖으로 나간다. default 에서 `panda_link0`(+ mount) 의 sphere union 은 collision hull 밖으로 p95 215.0 mm, 최대 249.6 mm 나가 있다(위 protrusion 표). 이 부분이 base 옆의 table 과 stove 위까지 닿아서, camera 가 본 table · stove 의 depth 점이 "robot sphere 안"으로 판정되어 지워졌다(head 에서 stove 30.92 %, table 5.75 %). 구현 단계의 smoke 도 같은 방향이다(impl.md, 검증 수치 아님: pedestal 덮개 중 위쪽 capsule 이 base frame z +0.173 m, x +0.394 m 까지 닿고 table top 은 z −0.012 m). **원인 분리 실험이 이를 확인한다.** hand 만 confined 로 바꾼 variant 는 stove 오제거가 그대로 30.92 % 이고, mount 만 confined 로 바꾼 variant 는 0.00 % 로 떨어진다(table 도 5.75 % → 0.74 %). confined mount 는 mount 의 큰 capsule 을 collision hull 안쪽에 가두어 작은 capsule 로 다시 맞춘 것(덮개 capsule 전체가 51 → 793 개)이고 protrusion 은 p95 61.3 mm · 최대 129.8 mm 로 줄었다.

**target bowl — hand 덮개가 손가락 사이로 내려와 있었다.** 같은 fitter 가 hand body 의 덮이지 않은 표면(옆면 등)을 한 덩어리로 보고 큰 capsule 몇 개로 덮었다. 그 capsule 의 끝 반구는 palm 아래, 즉 **두 손가락 사이 — 파지할 bowl 의 rim 이 들어오는 자리** 까지 나온다(hand protrusion p95 49.1 mm · 최대 56.4 mm; impl.md smoke: 손가락 z 0.052–0.106 m 사이의 파지 자리에 덮개 끝 반구가 hand z 0.112–0.120 m 까지 내려온다, 검증 수치 아님). 그래서 손가락이 bowl 을 잡으러 내려오고 bowl 이 손가락 사이로 들어올 때 bowl 의 depth 점이 hand 덮개 안에 들어가 robot 으로 지워졌다. 실제로 제거된 wrist target pixel 의 92 %(65,135 / 70,696)가 `panda_hand` 덮개에 귀속된다. **hand 만 confined 로 바꾸면 이 오제거가 11.00 % → 4.30 %(wrist), 5.46 % → 1.91 %(head)로 줄고** hand protrusion 은 p95 17.5 mm · 최대 43.1 mm 가 된다. mount 만 바꾼 variant 는 target 오제거가 그대로(5.46 / 11.00 %)이므로, target 오제거의 원인은 hand 덮개이고 mount 덮개와는 무관하다.

**RB-Y1 T17 과의 유사성.** T17 은 RB-Y1 쪽 episode 로, 옛 sphere 모델에서 손목 부근 capsule 이 쥔 사과를 덮어 사과가 **17 frame 동안 통째로 지워졌다**(사과 retention 0.472, `AG3S_T0T6_LOG.md` §3 T28 Phase 1; 이후 T29 의 손가락 개도 parameter joint + 덮개 재설계로 0 frame 이 되었다). 이번 Panda default 도 같은 종류의 기하다: **덮개 capsule 이 몸체 표면을 덮는 데 그치지 않고, 몸체가 잡으려는 물체가 들어올 공간까지 부풀어 있다.** 다만 Panda 에서는 bowl 이 통째로 지워진 frame 이 없다(최대 62.98 %). 그리고 finger 개도 parameter 의 문제가 아니다 — 손가락을 포함한 모든 variant 가 같은 finger joint 대응을 쓰고, 덮개를 hand 에서만 바꿔도 값이 바뀐다.

**남는 오제거는 finger 접촉 구간이다.** confined hand 에서도 wrist 에서 4.30 % 가 남는다. 제거된 pixel 의 대부분(27,656 중 22,973)이 `panda_rightfinger` 덮개에 귀속되고, 최대 15.53 %(frame 53)는 pad 접촉(frame 54) 직전이다. 이 구간은 MuJoCo 가 finger–bowl 접촉으로 보고하는 곳이므로 finger 덮개 표면과 bowl 표면이 맞닿아 있다(해석. 덮개 반지름별 분해는 측정하지 않았다). 한편 `franka.yml` only 는 target 을 거의 지우지 않지만(0.02 % · 0.01 %) robot 을 head 41.86 %, wrist 36.29 % 밖에 지우지 못해, 지우지 못한 robot 점이 ESDF 에서 장애물이 된다. 덮개는 필요하다.

### 결과 — #5 contact 장면 (MuJoCo contact × constraint sphere 침투)

**어떻게 쟀나.** link 별 sphere 와 물체 collision geom(이번 scene 의 LIBERO 물체 collision geom 은 모두 box 이고 바닥 plane 은 뺐다) 사이의 signed distance `sphere_sd` 의 최솟값을 구하고, 같은 (scene × 물체) cell 에서 MuJoCo contact 가 보고되는지와 대조했다. `sphere_sd < 0` 을 침투(P)로 센다(trajectory 접촉 cell 의 sd 부호와 일치). cell = (scene × 물체). trajectory 는 114 frame 위의 912 cell(link 별), fixture 는 7 개 scene 위의 56 cell. 아래 표는 `arms` constraint model(link1–7 + gripper, 59 sphere)이고, `gripper`(22 sphere)는 hand · finger 행과 같다.

**trajectory 의 MuJoCo contact frame (target bowl 과 접촉, image 순간 상태 기준).**

| MuJoCo geom → link | contact frame |
|---|---|
| `gripper0_rightfinger` (finger pad) → `panda_rightfinger` | 54, 90, 113 |
| `gripper0_right_gripper` (hand) → `panda_hand` | 82, 83, 84 |
| `gripper0_finger_joint1_tip` → `panda_leftfinger` | 90–113 (24 frame) |
| `gripper0_finger_joint2_tip` → `panda_rightfinger` | 90, 92–113 (23 frame) |

target bowl 의 lift(frame 10 대비 z 상승): frame 88 −0.0007 mm, 89 0.35 mm, 90 1.29 mm, 92 6.16 mm, 95 34.9 mm, 100 89.6 mm, 105 93.3 mm, 113 13.6 mm. 즉 frame 90 에서 파지가 되고 lift 가 시작된다.

**link 별 confusion table (C = MuJoCo contact, P = sphere 침투).**

| link (arms model) | trajectory C&P | C&noP (miss) | noC&P (false alarm) | noC&noP | fixture C&P | C&noP | noC&P | noC&noP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `panda_link1` | 0 | 0 | 0 | 912 | 0 | 0 | 0 | 56 |
| `panda_link2` | 0 | 0 | 0 | 912 | 0 | 0 | 0 | 56 |
| `panda_link3` | 0 | 0 | 0 | 912 | 0 | 0 | 0 | 56 |
| `panda_link4` | 0 | 0 | 0 | 912 | 0 | 0 | 0 | 56 |
| `panda_link5` | 0 | 0 | 0 | 912 | 1 | 0 | 0 | 55 |
| `panda_link6` | 0 | 0 | 0 | 912 | 1 | 0 | 0 | 55 |
| `panda_link7` | 0 | 0 | 0 | 912 | 2 | 0 | 0 | 54 |
| `panda_hand` | 0 | 3 | 0 | 909 | 2 | 1 | 0 | 53 |
| `panda_leftfinger` | 22 | 2 | 0 | 888 | 1 | 0 | 0 | 55 |
| `panda_rightfinger` | 0 | 24 | 0 | 888 | 0 | 0 | 0 | 56 |

- **false alarm 은 없다**: 접촉이 없는데 sphere 가 침투한 cell 은 trajectory 에서 0 이다(`no_contact_penetrate_cells` 가 비어 있다).
- **trajectory 의 접촉은 gripper 와 target bowl 의 파지 접촉뿐이다**(arm link1–7 은 접촉 cell 0). 접촉 cell 에서 sphere signed distance 는 `panda_leftfinger` −0.60 ~ +0.86 mm(24 cell 중 22 가 침투), `panda_rightfinger` **+0.47 ~ +2.02 mm**(24 cell 전부 침투 없음), `panda_hand` **+1.08 ~ +1.40 mm**(3 cell, 침투 없음)다. 같은 cell 의 MuJoCo contact 깊이는 0.02–0.49 mm 로 얕다(`contact dist` −0.0185 ~ −0.49 mm). 즉 gripper sphere 는 접촉 지점에서 **mm 수준(0.5–2 mm)으로 몸체보다 안쪽**에 있다(sphere 가 몸체보다 작다는 §#3 protrusion 과 일치).
- **비접촉 frame 의 최소 clearance (`sphere_sd`, bowl 에 가장 가까운 frame)**: `panda_hand` 3.3 mm(frame 85), `panda_leftfinger` 1.2 mm(frame 89), `panda_rightfinger` 0.74 mm(frame 55), `panda_link7` 57.1 mm(frame 83), `panda_link6` 112.1 mm, `panda_link5` 126.9 mm, `panda_link4` 323.7 mm, `panda_link3` 393.4 mm, `panda_link2` 325.3 mm, `panda_link1` 177.1 mm.

**금지 접촉 fixture 7 개.** joint 값을 random 하게 60,000 개 뽑아(seed 20261012) link–물체 간 MuJoCo contact 의 깊이가 −30 ~ −2 mm 인 자세를 골라, 의도한 link–물체 쌍마다 한 자세를 정했다(scene frame 10). 후보 수와 선택된 fixture 는 아래와 같다(탐색 27.4 s).

| fixture | 의도한 쌍 | 조건에 맞은 후보 수 | MuJoCo contact 깊이 (mm) | 해당 link 의 `sphere_sd` (mm) | 결과 |
|---|---|---:|---:|---:|---|
| F1 | link7 – cabinet | 317 | −10.50 | −8.24 | C&P |
| F2 | link6 – table | 988 | −10.04 | −8.34 | C&P |
| F3 | link5 – cabinet | 215 | −10.05 | −3.96 | C&P |
| F4 | hand – plate | 64 | hand −4.31 / leftfinger −6.20 | **hand +0.36** / leftfinger −4.62 | **hand 는 miss (C&noP)**, leftfinger 는 C&P |
| F5 | hand – table | 1,389 | −9.87 | −4.87 | C&P |
| F6 | link7 – stove | 352 | −8.41 | −5.65 | C&P |
| F7 | hand – bowl_2 | 169 | −8.86 | −3.45 | C&P |

**F4 의 miss.** hand 와 plate 사이 MuJoCo contact 깊이가 −4.31 mm 인데 hand sphere 의 signed distance 는 +0.36 mm(침투 아님)다. hand 의 constraint sphere 가 그 접촉 지점에서 mesh 보다 안쪽에 있기 때문이다. 같은 scene 에서 leftfinger 는 contact −6.20 mm, sphere −4.62 mm 로 침투를 잡으므로 scene 전체는 구별되지만, 이 한 칸은 hand 단독으로는 놓친다. 그 밖의 fixture 는 모두 접촉한 link 의 sphere 가 침투(−3.45 ~ −8.34 mm)로 나타났다. 관절 설정으로 만든 정적 scene 이므로 이 결과는 "fixture 7 개에서" 그렇다는 것이다.

`mj_geomDistance` 는 음수 값이 MuJoCo contact 깊이와 크게 달라(F2 link6–table: contact −10.04 mm, geomDistance −162.94 mm) 침투 깊이로 쓰지 않았고, figure 는 양수 값만 그렸다.

### 결과 — #6 SL0 경로 보존 · 회귀

| 검사 | 결과 |
|---|---|
| flag 없는 replay = SL0 | episode npz SHA-256 `e8a15133b8e2ae4aa9c069532d6eca9eba96ed453b070f1b3523a8c48aaaa088` 이 SL0 · geometry replay · flag 없음 replay 에서 모두 같고, mp4 SHA-256 `9ce6114fb5d99dc57549b28c54a6e047d9b967f4b15b381584c88899e7036f8a` 도 같다. flag 없는 run 은 `_geom.npz` 를 만들지 않았다 |
| `run_cpu_regression.sh` A (SL0 5 파일, `.venv-openpi-live`) | **73 passed** (SL0 · SL1 과 같다) |
| 같은 script B (`tests/benchmark`, `.venv-openpi-live`) | 58 passed (SL1 44 + 새 py38 check 1 + SL2 13) |
| 같은 script C (`tests/benchmark`, `.venv-ag3s`) | 57 passed, 1 skipped (serve_policy test, flax 없음 — SL1 과 같다) |
| script exit status | 0 |
| 16D legacy ESDF 기준선 | violated start 14/15 · has_target 15/15 · frame0 clearance −29.031 mm — 기준선과 일치 |
| 16D cuRobo ESDF 기준선 | 이 STEP 에서는 돌리지 않았다 (CPU 전용 task) |

`AG3S/ag3s/**` · `AG3S/trajopt/**` 를 바꾸지 않았으므로 16D 기준선이 그대로인 것은 구조상 예상된 결과다.

### 이 STEP 에서 알게 된 것 (평이한 설명)

**(a) Panda kinematics 는 맞다. 어긋난 것은 hand 장착 offset 뿐이었다.** cuRobo 의 `franka_panda.urdf` 는 팔 관절이 LIBERO 와 같고 손 장착 offset 만 robosuite 와 달랐다(impl.md: hand 0.5 mm, finger 6.5 mm). 이를 고친 adapter URDF 는 114 frame 에서 link1–7, hand, finger 의 FK 가 MuJoCo 와 4.8e-9 mm 이내로 같다.

**(b) `franka.yml` sphere 는 constraint 용이지 self-filter 용이 아니다.** 몸체를 12–74 % 만 덮고(arm), hand 는 13.4 % 다. 그런데 몸체 밖으로는 거의 나가지 않는다(hand protrusion p95 1.6 mm). constraint 로 쓸 때는 "실제 몸체보다 크지 않은 sphere" 가 맞는 성질이고, self-filter 로 쓰면 robot 의 depth 점을 head 41.86 %, wrist 36.29 % 밖에 못 지운다. 그래서 V11AB 는 덮개를 self-filter model 에만 넣는다.

**(c) 덮개를 두 곳에서 가두면 오제거가 거의 사라지고 robot 제거율은 그대로다.** mount 덮개가 stove · table 을(30.92 % → 0 %, 5.75 % → 0.72 %), hand 덮개가 target bowl 을(11.00 % → 4.30 %, 5.46 % → 1.91 %) 지웠다. 두 원인을 따로 바꾼 variant 가 서로의 값에 영향을 주지 않아 원인이 분리된다. robot 제거율은 head 99.23 % → 99.17 %, wrist 99.98 % → 99.97 % 로 거의 같다. 대가는 sphere 수다(245 → 4,271).

**(d) constraint sphere 는 gripper 에서 mm 수준으로 안쪽이다.** 접촉 지점에서 right finger 는 +0.47–2.02 mm, hand 는 +1.08–1.40 mm 떨어져 있고, trajectory 에서 접촉을 침투로 잡는 것은 left finger 뿐이다(22/24). 그래서 이 sphere 를 "접촉 판정"에 쓰면 mm 수준의 miss 가 있다(trajectory 에서 hand 3, right finger 24, left finger 2 cell; fixture F4 의 hand). false alarm 은 없다. clearance constraint 에 연결했을 때 이 mm 수준의 gap 이 margin 으로 흡수되는지는 이 STEP 에서 측정하지 않았다.

**(e) arm link 의 coverage 는 낮지만(12–74 %) trajectory 에서는 문제가 드러나지 않았다.** trajectory 는 arm 이 장애물에 가까워지지 않는 장면이라(link5–7 의 최소 sphere signed distance 57–127 mm 이상) 이 결과가 arm 보호가 충분하다는 뜻은 아니다. fixture 에서는 arm link(link5–7)가 접촉한 4 cell 이 모두 침투로 잡혔다.

### 판정 (2026-10-12, 사용자)

| # | 판정 |
|---|---|
| 1 | **SL2 확정.** |
| 2 | **self-filter model 의 Panda 기본 = confined hand+mount cover.** 이것은 algorithm 변경이 아니라 **robot geometry 보정**이다. (sphere 수 4,271 대 245; runtime 은 SL-4 에서 측정하고, 필요하면 compact refit 한다.) 남는 한계: mount pixel 1.3 % 가 남고(head mount 제거율 98.70 %), wrist 의 target 오제거 4.30 % 가 남는다. |
| 3 | **constraint model = cuRobo `franka.yml` sphere 그대로**(V11AB main). gripper-only protection 이다: sphere 가 mm 수준(0.5–2 mm)으로 gripper 몸체 안쪽이고, arm coverage 의 빈틈은 기록으로 남긴다(위 coverage 표). whole-body 보호가 필요해지면 MJCF-fit variant 를 따로 만든다. MuJoCo contact 를 정답으로 쓰는 독립 evaluator 는 SL-7 에서 만든다. |
| 4 | **다음은 SL-3 (attention audit).** |

SL2 의 "되돌아올 지점" 중 "`franka.yml` sphere coverage 가 크게 다르면 MJCF fit variant" 는 coverage 가 낮았지만(arm 12–74 %), 판정 3 이 V11AB main 에는 하나만 두기로 하여 **variant 를 만들지 않았다.**

### 측정하지 않은 것

- cuRobo 16D baseline (`.venv-openpi-live`, GPU) — CPU 전용 task
- URDF 쪽(cuRobo `franka_description`)의 mesh 에 대한 sphere coverage — robosuite mesh 만 쟀다
- `sphere_sd` 는 물체 collision box 기준이다(이 scene 의 모든 LIBERO 물체 collision geom 이 box). 바닥 plane 은 제외했다. 다른 task 의 비-box 물체는 쟀다고 할 수 없다
- `mj_geomDistance` 의 음수 값은 침투 깊이로 쓰지 않았다(MuJoCo contact 깊이와 달라서)
- segmentation 이 geom 이 아닌 pixel(seg −1)은 물체별 비율에서 제외했다(head 49,962 · wrist 86,922 pixel; 개수는 위에 기록)
- confined 4,271 sphere 모델의 runtime (SL-4 에서 측정)
- 다른 task · episode, 다른 해상도 (replay 한 task 0 · episode 0 의 114 frame 만)
- constraint clearance margin 이 mm 수준 gap 을 흡수하는지(clearance constraint 에 연결하지 않았다)

### 열린 항목

| # | 내용 | 영향 / 다음 |
|---|---|---|
| 1 | ~~구현 보류 2건: (a) confined hand+mount cover 를 기본 덮개 JSON 으로 commit (지금 commit 된 것은 default 51 capsule 이다), (b) `robot_geometry.py` 가 `LD_LIBRARY_PATH` 를 정리해서 `env.sh` 아래에서도 실행되게 하기~~ | ~~**SL3 에서 구현** (판정 2)~~ **SL3 에서 구현·확인 완료**: 기본 덮개 = confined(1,036 capsule), CLI 가 기본 shell 에서 returncode 0 |
| 2 | confined 모델 sphere 4,271 의 runtime | SL-4 에서 측정, 느리면 compact refit |
| 3 | arm coverage 의 빈틈(12–74 %) · gripper 의 mm 수준 gap | 판정 3 의 한계. whole-body 가 필요하면 MJCF-fit variant. 독립 contact evaluator 는 SL-7 |
| 4 | 남는 한계: mount pixel 1.3 %, wrist target 오제거 4.30 %, 기타 물체 오제거(bowl_2 0.51 % 등) | SL-4 에서 ESDF 결과로 영향 확인 |
| 5 | V11AB 의 RB-Y1 hardcode 지점 (impl.md 의 표): `serve_safe.build_ag3s` 의 model 주입, `constraint_link_filter`, `DEFAULT_CONTACT_LINKS` · `ContactConfig`, `safe_policy` 의 grasp/hold link, `multiview` 가 `robot_state` 를 그대로 q 로 쓰는 부분(Panda wire 는 9-D = q7 + finger 2 라 `split_robot_state` 가 필요), `wire.py` 의 camera 이름과 16-D layout(`ChunkLayout.rby1`), 관절 한계 model. **특히 `--capsule-radius-scale 0.05`(V11AB 의 RB-Y1 gap capsule 용)는 `franka.yml` sphere(role `collision`, zero-length)에 그대로 넘기면 반지름이 5 % 가 되므로 Panda 에서는 넘기지 않거나 exact role 로 두는 결정이 필요하다.** `--finger-cover both` 는 RB-Y1 손가락 구조 전제라 Panda 에 이식하지 않았다 | SL-4/5 입력. 구현은 하지 않았다 |
| 6 | wrist 0.05 m 미만 pixel(전체의 8.2 %, SL1)과 head 3.0 m 초과 pixel 의 처리 | SL-3 · SL-4 (SL1 열린 항목 4) |
| 7 | ESDF box z 범위와 table 처리 | SL-4 (SL1 열린 항목 2) |

### 되돌아올 지점 (SL2)

| 전환 신호 | 돌아와서 할 일 |
|---|---|
| FK 불일치가 장착 offset 으로 설명되지 않으면 | cuRobo URDF 대신 robosuite MJCF 에서 URDF 를 생성 (이번에는 hand · finger offset 3 값으로 설명되어 해당 없음) |
| `franka.yml` sphere coverage 가 robosuite mesh 와 크게 다르면 | MJCF mesh fit sphere 로 constraint model 을 만드는 variant 를 별도 이름으로 비교 (coverage 는 낮았지만 판정 3 으로 V11AB main 에는 `franka.yml` 만 두기로 하여 보류) |
| self-filter 가 grasp 중 target 을 지우면 | RB-Y1 T17 과 같은 원인인지 확인, 덮개 반지름 · finger parameter 재검토 (이번에는 hand 덮개가 원인임을 variant 로 분리하여 confined hand 로 처리) |
| confined 4,271 sphere 가 SL-4 runtime 을 넘으면 | compact refit(sphere 수를 줄이는 재맞춤) 또는 덮개를 hand+mount 에 한정한 채 반지름 정책 재검토 |
| Manipulator side `left` 가 SL-3/SL-5 의 camera↔arm audit 와 다르면 | adapter 의 side 표만 변경 |
| SL-4 에서 mm 수준 gripper gap 이 clearance 판정을 흔들면 | gripper sphere 반지름 보정 또는 MJCF-fit variant |

---

## SL3 — V11AB attention read-out audit on pi05_libero (+ SL2 후속 2건) (2026-10-12)

> 2026-10-12 KST. 구현 commit `ce29b9be` · 검증 commit `ae5d74a3`. 사용자 확정 2026-10-12.
> 관련 commit: `41339944`(공용 GPU lock · pod 자원 규칙 적용), `309e4c6d`(LIBERO dev/test 분리 확정).
> 원본: [`handoff/SL3.task.md`](handoff/SL3.task.md) · [`handoff/SL3.impl.md`](handoff/SL3.impl.md) ·
> [`handoff/SL3.verify.json`](handoff/SL3.verify.json). raw 출력은 `outputs/verify/SL3/`.
> 이 절의 수치는 모두 `SL3.verify.json` 의 `numbers` 가 출처다. **측정은 전부 dev 데이터**(LIBERO Spatial task 0–9 × init state 0, 10 episode, 221 policy call)이고, 이 STEP 에서 layer/head 를 **선택하지 않았다**(V11AB 고정 cell 을 그대로 측정했다).

### 이 STEP 이 답하는 질문

V11AB 의 ATS(Attention-based Target Segmentation: VLA attention 과 depth 로 target 을 고르는 단계)는 attention 을 RB-Y1 checkpoint 에서 고른
**고정된 attention cell** — denoise step 0, aggregation `last`, layer 8, head 2(이하 L8H2) — 에서 읽는다. 이 cell 을 `pi05_libero` 에 그대로 적용하고
policy image 의 좌우 반전(SL1 #3)을 되돌렸을 때, attention 이 LIBERO Spatial 의 target object 에 모이는가?

```
policy image 224×224 (= OpenCV 256 image 의 좌우 반전 + resize)
  -> pi05 prefix: base_0_rgb 256 token | left_wrist_0_rgb 256 token | right_wrist_0_rgb 256 token (LIBERO 는 padding, mask False)
  -> V11AB AttentionSampler (denoise 0, agg last, L8, H2) -> [768] -> 16×16 grid × camera
  -> [SL3] un-mirror(grid 열 뒤집기) + 224->256 scale -> OpenCV pixel (head / left_wrist) -> SL-4 attention lifting
```

### 무엇을 만들었나

**SL2 후속 2건** (SL2 판정 2 에 따른 구현 보류분)

| # | 내용 |
|---|---|
| 1 | self-filter 기본 덮개를 **confined hand + mount cover**(`confined_hand_mount`)로 바꿨다. 생성 JSON 은 provenance · digest 를 포함하고 `robots/panda/description/` 에 있다. 이전 기본(unconfined cover)은 내용을 바꾸지 않고 이름만 바꿔 **비교용으로 남겼다**(`..._unconfined.json`). run manifest 의 `robot_model.self_filter_cover` 에 cover 이름 · capsule 수 · sphere 수 · digest 를 기록한다 |
| 2 | `robot_geometry.py` CLI 가 기본 shell(`env.sh`)에서 import 에 실패하던 문제(pod `LD_LIBRARY_PATH` 의 외부 torch lib)를 고쳤다. launcher 와 **같은 함수**(`client_env.client_environment`)로 환경을 고치고 자기 자신을 한 번 re-exec 한다. launcher 의 규칙도 이 함수 하나로 합쳤다(동작 변경 없음) |

**Panda attention binding** (AG3S core 는 한 줄도 바꾸지 않았다)

| 구성 | 내용 |
|---|---|
| `AG3S/adapters/panda/attention.py` | V11AB 의 `AttentionSampler` · `AGGREGATIONS` · cell 상수(step 0, `last`, L8, H2)를 **import 해서 그대로** 쓴다(값을 다시 적지 않음). token block 표(`base_0_rgb -> head`, `left_wrist_0_rgb -> left_wrist`, `right_wrist_0_rgb -> padding`)는 가정이 아니라 model input 에서 **읽어서**(`read_token_layout`: `embed_prefix` 순서, image 별 token 수, image mask, V11AB prefix pass 가 실제로 쓴 prefix mask) 대조한다 |
| mirror-undo + 224→256 mapping | policy patch `(r, c)` 는 OpenCV 행 `[16r, 16r+16)`, 열 `[16(15-c), 16(16-c))` 와 정확히 일치한다. 따라서 un-mirror 는 grid 의 열 뒤집기이고 224→256 scale 은 V11AB 자신의 bilinear resampling 이 흡수한다. 반전을 **풀지 않은** mapping(`unmirror=False`, "mirror left in")도 비교용으로 같은 함수에 있다 |
| BDDL target 파싱 (`task_target.py`) | LIBERO BDDL 의 goal predicate 중 첫 인자가 `obj_of_interest` 에 있는 것을 target 으로 읽고 segmentation geom id 와 잇는다. **LIBERO Spatial 10 task 모두 target = `akita_black_bowl_1`, destination = `plate_1`** 이다. py3.8 check 가 LIBERO 자체 parser 와 10 task 모두 일치하고, live env 의 `obj_of_interest` · object root body 와도 일치함을 확인했다 |
| `benchmark/runners/attention_audit.py` | source run 의 `policy_records` + geom npz 를 읽어 call 별 attention(V11AB cell + 전체 [layer, head] 진단 배열 float16)을 **새 run** 에 저장한다. source run 은 읽기만 한다. policy record ↔ episode 의 join(state · prompt · image bit 동일)을 call 마다 검사하고 하나라도 다르면 멈춘다 |

### 실행

| run | 값 |
|---|---|
| dev rollout | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0`. `launch_libero_pi05_only.py --task-ids 0..9 --episode-ids 0 --capture-geometry --capture-segmentation`. 실행 창 2026-10-12 16:36:15–16:43:28 KST |
| attention audit | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T202459_309e4c6da477_sl3-attention-audit-dev` (`attention_audit.py`, `.venv-openpi-live`). GPU lock 을 잡고 compute process 가 없음을 확인한 뒤 실행. 실행 창 20:24:58–20:26:15 KST |
| flags-off replay (SL0 경로 보존) | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T195406_309e4c6da477_sl3-replay-off-t0e0` |
| 채점 | segmentation 은 **image instant**(image 가 찍힌 순간)로 다시 render 한 mask(`segP`)를 기본으로 쓰고, `env.step` return 시점에 기록된 mask(`segRec`)와 비교했다(SL1 의 capture instant 차이 때문) |

### 결과 — Figure

세 figure 는 verifier 가 만들었다 (`outputs/verify/SL3/scripts/make_figures.py`, sidecar `.json` 동봉).

**실제 scene** — task 0–9 의 대표 call(approach 단계) 의 head · left_wrist OpenCV RGB 위에 V11AB cell attention(파랑)과 target `akita_black_bowl_1` 의 segmentation contour(주황)를 겹쳤다. 각 camera 마다 **un-mirror 와 mirror left in 을 위아래로 대조**했다.
head 는 un-mirror 쪽에서 attention 이 target 주변에 놓이고, mirror left in 에서는 좌우가 어긋난 곳에 놓여 mass 가 0 에 가깝다(t0 · t5 · t8 은 0.000).

![SL3 scene](figures/sl3-scene.png)

**graph** — (a, b) call 별 target mass 추이(배경 색 = phase, 점선 = target area, 즉 uniform baseline), (c, d) task 별 mass 대 area, (e) phase 별, (f, g) 전체 layer×head 진단 grid(주황 상자 = V11AB L8H2), (h) call 별 mapping 비교(대각선 위 = un-mirror 가 큼).

![SL3 graph](figures/sl3-graph.png)

**table** — token layout · mapping 규약 · grounding 지표 · 진단 grid 순위 · 비용 · SL2 후속을 한 장에 모은 표.

![SL3 table](figures/sl3-table.png)

### 결과 — #1 SL2 후속: confined cover 기본값과 robot_geometry CLI

| 확인 | 결과 |
|---|---|
| `robot_geometry.py export` 를 기본 shell 에서 실행 | returncode 0 (시작 시 `LD_LIBRARY_PATH` 에 외부 torch lib 가 있었음). export 파일 SHA-256 이 SL2 export 와 같고 `geometry_digest` `ea38872f…` 가 같다 |
| manifest 의 cover 항목 (dry-run) | `confined_hand_mount`, capsule 1,036 · cover sphere 4,210 · self-filter sphere 4,271, file SHA-256 `b8ec2deb…` (비교용 unconfined SHA-256 `895b7dbf…`, 이전과 같음) |
| 무인자 builder 의 sphere 수 | 기본 4,271, `"unconfined"` 245 |
| SL2 측정과 일치 | 기본 = SL2 `confine_hand_mount` 와 44/44 cell 일치, unconfined = SL2 default 와 44/44 cell 일치 (6/6 row band 일치) |

**self-filter 제거율** (SL2 replay task 0 · episode 0, 114 frame, `segP`, 기본 confined 대 비교용 unconfined). 지운 pixel 비율이며 robot 은 높을수록, 나머지(target · stove · table)는 **낮을수록** 좋다.

| camera | 대상 (pixel 수) | confined (기본) | unconfined |
|---|---|---:|---:|
| head | robot (854,825) | 99.17 % | 99.23 % |
| head | mount (159,088) | 98.70 % | 99.93 % |
| head | target bowl (75,545) | 1.91 % | 5.46 % |
| head | stove (179,550) | 0.00 % | 30.92 % |
| head | table (3,153,421) | 0.72 % | 5.75 % |
| left_wrist | robot (769,260) | 99.97 % | 99.98 % |
| left_wrist | target bowl (642,523) | 4.30 % | 11.00 % |
| left_wrist | stove (150,097) | 0.33 % | 11.68 % |
| left_wrist | table (3,854,155) | 0.38 % | 1.62 % |

이 값은 SL2 의 confined hand+mount 결과와 같다(위 cell 비교). 대가는 sphere 수(245 → 4,271)이며 runtime 은 SL-4 에서 잰다.

### 결과 — #2 visual mesh 밖으로 나온 양 (confined cover 의 알려진 한계)

confined 규칙은 hand · mount capsule 을 body 의 **collision** 표면 안에 가둔다. 그래서 collision 표면은 덮지만 **visual(render) mesh** 는 그 밖으로 나올 수 있다. frame 0, 4 mm 표본에서 render mesh 표면의 sphere union 에 대한 최대 signed distance (양수 = sphere 밖):

| body | confined (기본) | unconfined | confined 에서 밖인 표본 / 전체 |
|---|---:|---:|---:|
| `gripper0_right_gripper` (hand) | −0.170 mm | −4.404 mm | 0 / 85,399 |
| `mount0_torso` | **+34.448 mm** | −122.468 mm | 773 / 4,056 |
| `mount0_pedestal` | **+93.215 mm** | −4.533 mm | 1,602,405 / 3,522,906 |

- hand 의 visual mesh 는 기본 cover 안에 들어간다. implementer 가 `SL3.impl.md` 에 적은 hand 0.880 mm 와 verifier 값(−0.170 mm)은 다르다. verifier 값을 정본으로 쓰며 차이의 원인은 조사하지 않았다.
- SL2 의 mount 제거율은 98.70 %(unconfined 99.93 %)였다. 제거되지 않고 남은 mount pixel 은 pedestal 에서만 나온다(같은 body 의 visual 초과와 일치). 수는 head 에서 head 에서 2,062 개(114 frame 합, 모두 `mount0_pedestal`)이고 sphere union 까지의 거리는 median 10.68 mm(min 0.38 mm, max 771.28 mm)이다. unconfined 에서는 114 개(median 661.76 mm)다. wrist 에는 mount 가 보이지 않는다(0 pixel).
- 남은 mount pixel 이 ESDF 에서 장애물이 되는지는 이 STEP 에서 재지 않았다(SL-4).

### 결과 — #3 dev rollout (기록만; 성공률 주장이 아니다)

task 0–9 × init state 0 각 1 episode. official success 는 `env.step()` 의 `done` 이다.

| task | 설명 (target 은 모두 `akita_black_bowl_1` → `plate_1`) | official_success | status | env step (wait 포함 / 제외) | policy call |
|---:|---|:---:|---|---:|---:|
| 0 | pick up the black bowl between the plate and the ramekin … | true | success | 113 / 103 | 21 |
| 1 | … black bowl next to the ramekin … | true | success | 120 / 110 | 22 |
| 2 | … black bowl from table center … | true | success | 142 / 132 | 27 |
| 3 | … black bowl on the cookie box … | true | success | 98 / 88 | 18 |
| 4 | … black bowl in the top drawer of the wooden cabinet … | true | success | 126 / 116 | 24 |
| 5 | … black bowl on the ramekin … | true | success | 107 / 97 | 20 |
| 6 | … black bowl next to the cookie box … | true | success | 114 / 104 | 21 |
| 7 | … black bowl on the stove … | true | success | 124 / 114 | 23 |
| 8 | … black bowl next to the plate … | true | success | 120 / 110 | 22 |
| 9 | … black bowl on the wooden cabinet … | true | success | 122 / 112 | 23 |

- 10/10 episode 가 `success`, policy call 합 221 = `policy_records` 221 개. task 0 의 npz 는 SL0 run 과 같다(`t0_npz_equal_to_SL0`).
- **이것은 dev 의 10 episode(task 당 init state 하나)이며 성공률이 아니다.** test(init state 5–49)는 쓰지 않았다.

### 결과 — #4 token layout (모델 input 에서 읽은 값)

| block | token 범위 | image_mask | prefix mask | camera | cell L8H2 block mass (median / mean / max) |
|---|---|---|---|---|---|
| `base_0_rgb` | 0–256 | True | True | head (agentview) | 0.00105 / 0.0462 / 0.935 |
| `left_wrist_0_rgb` | 256–512 | True | True | left_wrist (robot0_eye_in_hand) | 0.995 / 0.924 / 1.002 |
| `right_wrist_0_rgb` | 512–768 | **False** | False | padding (사용 안 함) | 0 / 0 / 0 |
| text | 768–968 | – | 유효 20 / 200 | – | – |

- `problems = []`: 읽어낸 layout 이 V11AB `CAMERA_BINDINGS` 와 camera 표에 모두 일치한다. prefix 길이 968.
- padding block 의 attention mass 는 **전체 18 × 8 [layer, head] 모든 call 에서 최대 0.0** 이다.
- V11AB 자신의 `AttentionPolicy` 와 대조한 2 call 은 bit 동일(max |Δ| 0.0). 저장한 float16 pooled 배열의 L8H2 와 float32 `cell_tokens` 의 차이는 최대 3.0e-8 이다.
- **block mass** = 한 cell 의 attention 이 image block 하나에 놓인 합. 이 cell 은 attention 을 **거의 전부 wrist image 에** 두고(wrist median 0.995), head block 은 median 0.00105 만 받는다. 이 값은 아래 "알게 된 것"에서 다시 다룬다.

### 결과 — #5 방향 대응: un-mirror 대 mirror left in

두 검사를 분리해서 읽는다.

**(a) image 검사 — un-mirror 가 기하학적으로 맞다.** policy image 를 OpenCV image 로 복원한 것과 16×16 block 단위로 비교한 MAE(0–255):

| camera | un-mirror | mirror left in | un-mirror 가 더 작은 call |
|---|---:|---:|---:|
| head | 0.077 | 42.20 | 221 / 221 |
| left_wrist | 0.077 | 30.72 | 221 / 221 |

**(b) attention 검사 — target mass 비교** (221 call, `segP`).

| camera | mapping | mass 평균 | mass / area (평균 비) | argmax 가 target 위 | call 별 un-mirror > mirror / < / 같음 |
|---|---|---:|---:|---:|---|
| head | un-mirror | **0.1202** | 10.20 | 0.317 | **167** / 54 / 0 |
| head | mirror left in | 0.0211 | 1.79 | 0.059 | |
| left_wrist | un-mirror | 0.1826 | 1.48 | 0.385 | 91 / **123** / 7 |
| left_wrist | mirror left in | **0.2939** | 2.39 | 0.380 | |

- head 는 un-mirror 가 맞다는 (a) 와 (b) 가 모두 같은 방향이다.
- **wrist 는 (a) 와 (b) 가 어긋난다.** image 로는 un-mirror 가 정확한데, target mass 평균은 mirror left in 이 높다(0.294 대 0.183). argmax 가 target 위인 비율은 거의 같다(0.385 대 0.380).
- phase 별로 보면 이 차이의 위치가 드러난다(wrist, mass 평균): approach 는 un-mirror 0.315 대 mirror 0.225, grasp 는 0.214 대 0.171 로 un-mirror 가 높고, **transport 에서만** un-mirror 0.022 대 mirror 0.407 로 뒤집힌다(transport 의 target area 평균은 0.176). 아래 phase 표 참고.
- ~~**미검증 가설 (lead 의 작업 가설)**: grasp 이후 bowl 이 wrist view 의 중앙에서 크게 보이므로, 좌우를 뒤집은 map 도 대칭성 때문에 target 과 겹친다. 이 가설은 이 STEP 에서 검증하지 않았다(per-phase · per-position 분해는 SL-4 로 예정). 위 transport 수치는 가설과 모순되지 않는 관찰일 뿐 증명이 아니다.~~ → **SL4 에서 질량 분해로 해소했다**(아래 [SL4 #4](#결과--4-wrist-mirror-진단--attention-mass-분해와-mirror-run)). transport 에서 policy 의 attention 은 target bowl 이 아니라 **destination plate** 로 옮겨 간다(head 0.578 · wrist 0.619). mirror left in 이 wrist transport 에서 높았던 것(target mass 0.407, plate mass 0.126)은 더 정확한 mapping 이어서가 아니라, plate 에 있던 attention 이 뒤집히며 면적이 큰 bowl(wrist image 면적 0.176, plate 0.049)에 우연히 걸린 결과로 읽는다. 겹친 위치 자체는 측정하지 않았다. un-mirror 는 확정이다.
- 따라서 **mapping 은 un-mirror 로 확정한다**(image 검사 기준). wrist 의 mass 점수만으로 mirror 를 택하지 않는다.

### 결과 — #6 V11AB cell 의 target grounding (L8H2, denoise 0, `last`; un-mirror; `segP`)

용어: **mass** = 그 camera 의 256 token attention 합을 1 로 정규화했을 때 target mask 가 덮는 patch 가중 합. **area** = 같은 patch 가중으로 센 target 의 면적 비율 = attention 이 균일하게 퍼졌을 때 기대되는 mass(**uniform baseline**). **mass/area** > 1 이면 chance 보다 target 에 모인다. **rank** = BDDL 물체 · fixture 중 target 보다 mass 가 큰 것의 수 + 1.

**전체**

| camera | n call (target 보임) | mass 평균 / median | area 평균 | mass/area (평균 비) | median mass/area | mass > area 인 call 비율 | argmax 가 target 위 | rank 1 비율 | rank 평균 (median) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| head | 221 (221) | 0.120 / 0.026 | 0.0118 | **10.20** | 3.13 | 0.548 | 0.317 | 0.059 | 3.54 (3) |
| left_wrist | 221 (214) | 0.183 / 0.060 | 0.123 | 1.48 | 0.53 | 0.444 | 0.385 (visible 0.397) | 0.220 | 2.24 (2) |

- 같은 camera 에서 평균 instance mass(head)는 `main_table` 0.425, `plate_1` 0.330, target `akita_black_bowl_1` 0.120, robot 0.025 순이다. 즉 head 의 attention 은 target 에 chance 의 약 10 배로 모이지만 **target 이 가장 많이 받는 물체는 아니다**(rank 1 은 call 의 5.9 %). `main_table` 은 면적이 크다.
- `segP` 와 `segRec` 의 차이는 작다: head mass 평균 절대차 0.00026, wrist 0.0016, rank 가 달라지는 call 은 head 0 · wrist 1. image instant 로 다시 render 한 RGB 는 모든 call frame 에서 기록된 RGB 와 같다.

**phase 별** — approach = 첫 gripper–target contact 이전, grasp = 첫 contact 부터 lift(target z > z0 + 1 cm)까지, transport = lift 부터 마지막 contact 까지(post_release 는 0 call). MuJoCo contact · target 높이로 정한 label 이며 subtask probe 는 쓰지 않았다.

| camera | phase (n) | mass 평균 | area 평균 | mass/area | argmax 가 target 위 | rank 평균 |
|---|---|---:|---:|---:|---:|---:|
| head | approach (105) | 0.214 | 0.0124 | **17.19** | 0.505 | 2.79 |
| head | grasp (25) | 0.0725 | 0.0081 | 8.99 | 0.080 | 4.24 |
| head | transport (91) | 0.0252 | 0.0120 | 2.09 | 0.165 | 4.21 |
| left_wrist | approach (105, 보임 98) | 0.315 | 0.0682 | 4.61 | 0.686 | 1.66 |
| left_wrist | grasp (25) | 0.214 | 0.163 | 1.31 | 0.240 | 2.00 |
| left_wrist | transport (91) | 0.0218 | 0.176 | 0.12 | 0.077 | 2.93 |

(참고: 같은 phase 의 mirror left in, wrist 는 approach 0.225 / grasp 0.171 / transport 0.407, head 는 0.0433 / 0.0023 / 0.0006.) grasp phase 는 task 당 0–8 call 로 짧다(t9 는 contact 와 lift 가 같은 frame 이라 0 call).

**task 별** (head · left_wrist, un-mirror). 한 task 는 episode 하나(call 18–27 개)이므로 call 들은 서로 독립이 아니다.

| task | head mass | head area | head mass/area | head argmax 가 target 위 | wrist mass | wrist area | wrist mass/area | wrist argmax 가 target 위 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.233 | 0.0100 | 23.34 | 0.429 | 0.352 | 0.0899 | 3.92 | 0.667 |
| 1 | 0.179 | 0.0091 | 19.77 | 0.455 | 0.211 | 0.0873 | 2.41 | 0.318 |
| 2 | 0.0714 | 0.0101 | 7.10 | 0.296 | 0.160 | 0.1268 | 1.26 | 0.481 |
| 3 | 0.176 | 0.0131 | 13.47 | 0.556 | 0.173 | 0.1015 | 1.71 | 0.500 |
| 4 | 0.0372 | 0.0158 | 2.35 | 0.125 | 0.110 | 0.1312 | 0.84 | 0.208 |
| 5 | 0.123 | 0.0110 | 11.16 | 0.550 | 0.240 | 0.0901 | 2.67 | 0.450 |
| 6 | 0.183 | 0.0145 | 12.62 | 0.524 | 0.191 | 0.1139 | 1.68 | 0.381 |
| 7 | **0.0047** | 0.0076 | **0.62** | 0.000 | 0.0855 | 0.2468 | 0.35 | 0.174 |
| 8 | 0.158 | 0.0122 | 12.90 | 0.227 | 0.193 | 0.0760 | 2.54 | 0.545 |
| 9 | 0.0818 | 0.0149 | 5.49 | 0.130 | 0.139 | 0.1509 | 0.92 | 0.174 |

head 는 10 task 중 8 task 에서 mass/area > 5 이고, task 7(`stove` 위의 bowl)은 mass/area 0.62 로 chance 아래다. task 4(drawer 안)도 2.35 로 낮다. wrist 의 task 7 area 평균이 0.247 로 가장 큰데 mass 는 0.0855 이다.

### 결과 — #7 진단 grid (dev only; 선택하지 않았다)

V11AB L8H2 가 전체 18 layer × 8 head = 144 cell(denoise 0, aggregation `last`, float16 pooled) 중 어디에 놓이는지만 센다. **이 표로 layer/head 를 고르지 않는다.**

| camera | 지표 | V11AB L8H2 (순위 / 144) | 144 cell median | 최대 (위치) |
|---|---|---|---:|---|
| head | mass 평균 | 0.121 (**5**) | 0.0352 | 0.248 (L12H3) |
| head | mass/area | 10.28 (**5**) | 2.99 | 21.06 (L12H3) |
| head | argmax 가 target 위 | 0.317 (10) | 0.0351 | 0.525 (L6H0) |
| head | rank 1 비율 | 0.059 (26) | 0.0068 | 0.419 (L17H6) |
| left_wrist | mass 평균 | 0.183 (12) | 0.0998 | 0.343 (L5H5) |
| left_wrist | mass/area | 1.48 (12) | 0.807 | 2.79 (L5H5) |
| left_wrist | argmax 가 target 위 | 0.385 (15) | 0.133 | 0.575 (L5H5) |
| left_wrist | rank 1 비율 | 0.220 (26) | 0.0701 | 0.607 (L5H5) |

- uniform baseline: head 0.0118, wrist 0.123. V11AB cell 의 head mass/area 10.28 은 144 cell 의 median(2.99)보다 크다.
- 일부 cell 은 어떤 call 에서 block 합이 float16 에서 0 이 되어 유효 call 수가 31–221 이다(head 10 cell, wrist 8 cell). V11AB L8H2 는 221 call 모두 유효하다.
- float16 pooled 로 계산한 L8H2 head mass 0.1211 은 float32 `cell_tokens` 의 0.1202 와 float16 오차 안에서 같다.

### 결과 — #8 비용

| 항목 | 값 | 조건 |
|---|---|---|
| attention pass 첫 call (JIT) | 2.094 s | audit, **uncontended**(GPU lock) |
| attention pass 이후 (n = 220) | p50 64.36 ms / p95 77.29 ms / mean 65.44 ms / max 94.60 ms | audit, uncontended. 기본 설정(num_denoise 10, subtask feature, full grid 저장) |
| model load | 34.86 s | audit |
| JAX device memory | load 후 6.98 GB(6,983,379,712 B), peak 7.21 GB(7,212,886,784 B) | audit |
| `nvidia-smi` 이 process 의 최대 | 8,868 MiB | audit |
| rollout client 왕복 latency, 첫 call (JIT) | 3,222.2 ms | dev rollout, **GPU contended** |
| rollout client 왕복 (n = 220) | p50 77.18 ms / p95 103.31 ms / max 264.75 ms | dev rollout, **GPU contended** |
| rollout server policy infer, 첫 call | 2,022.5 ms | dev rollout, **GPU contended** |
| rollout server policy infer (n = 220) | p50 41.05 ms / p95 54.36 ms / max 153.28 ms | dev rollout, **GPU contended** |

- **rollout latency 는 development checkout 의 CL3 closed-loop run 과 GPU 를 함께 쓰던 2026-10-12 16:36:15–16:43:28 KST 구간에서 쟀다.** 경합 없는 steady-state 값이 아니므로 다른 STEP 의 latency 와 직접 비교하지 않는다. **audit latency 는 GPU lock 을 잡고 다른 compute process 가 없음을 확인한 뒤 쟀다(경합 없음).**
- rollout 중 GPU 최대 사용: 다른 process 두 개가 각각 18,060 · 18,058 MiB, 이 run 의 server(pid 596005)가 8,802 MiB, GPU 전체 합 최대 45,530 MiB.
- audit 의 attention pass 는 V11AB `AttentionSampler` 로 계산한 비용이다(기본 설정: num_denoise 10, subtask feature on, full grid 저장 on; `SL3.impl.md`). runtime budget 판정은 SL-4 이후.

### 결과 — #9 회귀

| 검사 | 결과 |
|---|---|
| legacy 16D ESDF 기준선 | violated start 14/15 · has_target 15/15 · frame0 clearance −29.031 mm — 기준값과 일치 |
| `scripts/testing/run_cpu_regression.sh` | A 73 passed (77 warnings) / B 78 passed / C 76 passed, 2 skipped / exit 0 |
| flags-off replay (SL0 경로) | episode npz 가 SL0 와 모든 key 에서 같다. jsonl: official_success true · status success · env step 113 · policy call 21 (replay 와 SL0 동일) |

### 이 STEP 에서 알게 된 것 (평이한 설명)

1. **V11AB 의 head attention 은 pi05_libero 에서도 target 근처에 모인다.** target mask 가 차지하는 면적(uniform baseline) 대비 약 10 배(mass/area 10.20)이고, 144 개 (layer, head) 중 mass 순위는 5 위다. 접근(approach) 단계에서 특히 높다(17.19; 위 graph (a)). 하지만 target 이 가장 많이 받는 물체는 아니다(rank 1 은 5.9 %; table · plate 가 더 받는다). grasp 이후에는 급격히 떨어진다(transport 2.09, median 비는 0.07 정도로 chance 이하). task 7 · 4 는 약하다.
2. **이 cell 은 attention 의 거의 전부를 wrist image 에 둔다.** head block mass 의 median 이 0.00105 이고 wrist block 은 0.995 다. V11AB 가 raw mass 에 threshold 를 걸어 pixel 을 고르는 구조라면 head pixel 은 threshold 를 통과하지 못할 수 있다. 이것은 가능성이며 **SL-4 에서 측정한다**(여기서는 정규화한 mass 비율만 재었다).
3. **mirror 는 풀어야 한다.** image 검사는 221/221 call 에서 un-mirror 를 지지한다. head 에서는 attention 검사도 같은 결론이다(167/221 call, mass 0.120 대 0.021). **wrist 에서는 mirror left in 의 target mass 평균이 더 높았다**(0.294 대 0.183). 이 불일치는 transport phase 에 집중돼 있고(0.407 대 0.022), approach · grasp 에서는 un-mirror 가 높다. ~~lead 의 **미검증 가설**: grasp 이후 bowl 이 wrist view 에서 중앙에 크게 보이므로 좌우를 뒤집은 map 도 대칭성으로 겹친다. 이 가설과 phase 별 원인 분해는 SL-4 에서 한다.~~ → SL4 #4 에서 분해했다: transport 에서 attention 이 destination plate 로 옮겨 가며, 가설의 "중앙의 bowl" 설명은 맞지 않았다.
4. confined cover 로 target · stove · table 오제거는 줄었지만(head target 5.46 → 1.91 %, stove 30.92 → 0 %) mount 의 **visual mesh** 일부(pedestal 93 mm, torso 34 mm)가 sphere union 밖에 남는다.

### 판정 (2026-10-12, 사용자)

| # | 판정 |
|---|---|
| 1 | **SL3 확정.** |
| 2 | **LIBERO dev/test 분리 = 안 A.** dev = init state 0–4, test = 5–49 (plan §6.0). 이 STEP 의 측정은 모두 dev 다. |
| 3 | **Panda 는 V11AB attention cell L8H2 를 그대로 쓴다**(recalibration 하지 않는다). SL-4 에서 target geometry 가 부족하다고 확인된 경우에만 **dev 전용 calibration variant** 를 별도 이름으로 만든다. 진단 grid 의 최대(L12H3)를 자동으로 채택하지 않는다. |
| 4 | **다음은 SL-4**: Panda 를 V11AB core 에 연결(ATS / LTE / ESDF). |

### 측정하지 않은 것

- SL0 run 에 대한 attention audit (dev t0e0 npz 가 SL0 와 같아 반복하지 않았다)
- 경합 없는 rollout latency (rollout 은 development CL3 와 GPU 를 공유하는 동안 쟀다)
- cuRobo 16D baseline (legacy baseline 만 실행)
- frame 0 외의 frame 에서의 visual mesh 초과 (rigid body 이므로 한 frame)
- Panda feature 에서의 subtask probe `kv_L4_v1` (SL-6)
- phase 가 아닌 **위치** 별(wrist 중앙 / 가장자리) 분해, mirror left in 의 wrist 우세 원인
- head 의 raw mass threshold 가 pixel 을 고르는지(SL-4)
- init state 1–4(dev 의 나머지)와 test init state

### 운영 기록 (shared pod)

| 사건 | 내용 |
|---|---|
| development checkout 과의 GPU 겹침 | dev rollout(2026-10-12 16:36:15–16:43:28 KST)이 development checkout 의 CL3 closed-loop run 과 GPU 를 동시에 썼다. 그래서 rollout latency 는 경합 하에서 잰 값으로 표시한다. 이후 attention audit 은 `/tmp/ag3s-vla-gpu-validation.lock` 을 `flock` 으로 잡고 `nvidia-smi` 로 다른 compute process 가 없음을 확인한 뒤 실행했다 |
| pod 메모리 | benchmark 쪽 CPU 측정 job 이 pod working set 을 64 GiB 위로 올려 lead 가 중단했다(lead 보고; 최대값은 verify.json 에 없다). verify.json 의 self-filter 측정(`selffilter_default`)은 row band 16 단위 처리와 RSS 기록(`selffilter_rss.csv`)으로 다시 실행된 기록이 남아 있다 |
| 규칙 채택 | 사용자가 development checkout 과 같은 공용 GPU lock · pod 자원 규칙을 이 checkout 에도 적용하기로 결정했다(commit `41339944`): GPU 작업은 lock 을 잡고 다른 compute process 확인 후 시작, 다른 checkout 과 동시 실행 금지, pod working set ≤ 64 GiB(하드 80 GiB), CPU worker 세션당 ≤ 2(pod ≤ 4), 측정 job 은 `nohup` 없이 실행, 긴 GPU 작업 전후 다른 세션에 알림 |

### 열린 항목

| # | 내용 | 영향 / 다음 |
|---|---|---|
| 1 | ~~**head block mass 가 매우 작다**(median 0.00105). V11AB 의 raw-mass threshold 가 head pixel 을 고르는지~~ | → SL4 #3 에서 측정: head 가 seed 를 만든다(접근 단계 seed 의 head-argmax 비율 0.520). 해소 |
| 2 | ~~wrist 에서 mirror left in 이 mass 가 높은 원인(가설: grasp 후 bowl 이 중앙에서 크다)~~ | → SL4 #4 에서 분해. 원인 = transport 의 attention 이 destination plate 로 이동. 해소 |
| 3 | ~~**SL-4 가 필요로 하는 AG3S core 주입 지점**(구현하지 않았다): `serve_safe.py:1037-1058` `attention_extractor` 의 camera alias 를 Panda(`cam_high -> head`, `cam_left_wrist -> left_wrist`, `cam_right_wrist` 버림)로 바꾸고 **`lift` 전에 grid 열 뒤집기**(`attention.unmirror_grid`)를 넣어야 한다. core `GridAttentionAdapter` / `lift` 에는 flip 옵션이 없다. 그 밖에 SL2 열린 항목 5 의 RB-Y1 hardcode 지점이 남아 있다~~ | → offline 은 SL4 profile(생성자 인자 · config dict)로 core 변경 없이 해결. **online server 주입은 SL-5**(최소 core 변경, 사용자 판정 2026-10-13) |
| 4 | subtask probe `kv_L4_v1` 은 RB-Y1 feature 로 학습됐다. 이번 phase label 은 MuJoCo contact · lift 이고 probe 는 쓰지 않았다. Panda `kv_feature` 는 저장돼 있다 | SL-6 audit |
| 5 | ~~confined 4,271 sphere self-filter 의 runtime, mount visual mesh 초과(pedestal 93 mm, torso 34 mm)가 ESDF 에 장애물로 남는지~~ | → SL4 #8 · #9 에서 측정(self-filter 포함 scene_reconstruction p50 38.8 ms; mount 잔여 점은 call 당 최대 7 개, gripper 에서 326 mm 이상). 해소 |
| 6 | wrist 0.05 m 미만 pixel(전체 8.2 %, SL1)과 head 3.0 m 초과 pixel 처리 | SL-4 (SL2 열린 항목 6 에서 이월; 이 STEP 에서 다루지 않았다) |
| 7 | segmentation 은 `env.step` return 시점 render 가 기본 기록이다. attention 채점은 image instant 로 다시 render 한 `segP` 를 썼다. 두 결과 차이는 작았다(위 #6) | SL-4 에서 ATS 가 어느 시점 mask 를 쓰는지 정한다 |
| 8 | dev 는 task 당 episode 1 개. call 들이 episode 안에서 상관돼 있어 신뢰구간을 말하지 않는다 | init state 1–4 로 확장 시 |

### 되돌아올 지점 (SL3)

| 전환 신호 | 돌아와서 할 일 |
|---|---|
| SL-4 에서 head target geometry 가 ESDF 에 부족하다고 확인(threshold 가 head pixel 을 못 고름 등) | **dev 전용 calibration variant** 를 별도 이름으로 선언하고 init state 0–4 에서만 고른다. 사용자 판정 필요. 진단 grid 최대(L12H3)를 자동 채택하지 않는다(test init state 5–49 와 분리) |
| SL-4 per-phase 분해에서 wrist 의 mirror 우세가 가설과 다르게 나옴 | SL1 의 orientation 측정 · policy transform 경로를 다시 대조한다(원인 확정 전 wrist attention 을 depth 에 쓰지 않는다) |
| subtask probe 가 Panda 에서 쓸 수 없음 | SL-6 에서 strong-rule / GT phase 등 계획의 대안과 함께 blocker 로 기록 |
| confined 4,271 sphere 가 SL-4 runtime 을 넘음 | compact refit 또는 덮개를 hand + mount 에 한정한 채 반지름 정책 재검토 (SL2 에서 이월) |


---

## SL4 — Panda 관측을 V11AB perception → ESDF 에 연결 (ATS · LTE · ESDF), offline (2026-10-12)

> 2026-10-12 ~ 10-13 KST. 구현 commit `46a08d6f`(2026-10-12 22:53 KST) · 검증 commit `c2ba3fea`(2026-10-13 00:21 KST). 사용자 확정 2026-10-13.
> 관련 commit: `d53d8e07`(GPU lock `flock` 예시 수정, 아래 "운영 기록"), `41a5b28a`(SL4b task 정의, 아래 "판정").
> 원본: [`handoff/SL4.task.md`](handoff/SL4.task.md) · [`handoff/SL4.impl.md`](handoff/SL4.impl.md) ·
> [`handoff/SL4.verify.json`](handoff/SL4.verify.json). raw 출력은 `outputs/verify/SL4/`.
> 이 절의 수치는 모두 `SL4.verify.json` 의 `numbers` 가 출처다. **측정은 전부 dev 데이터**(LIBERO Spatial task 0–9 × init state 0,
> 10 episode, 221 policy call; SL3 dev rollout + SL3 attention audit)이고 이 STEP 에서 **아무것도 선택하지 않았다**(cell · 규칙 · threshold 모두 V11AB 값 그대로).

### 이 STEP 이 답하는 질문

SL1–SL3 으로 Panda 의 camera(depth · K · `T_base_cam`), robot geometry(self-filter · constraint sphere), attention(L8H2 + un-mirror)이 각각 V11AB 형식으로 준비됐다.
SL4 는 이것들을 **V11AB 의 실제 perception 코드**(`AG3S.process_multi_debug`)에 넣는다. AG3S core 는 수정하지 않는다. 질문은 이것이다:

> V11AB perception 코드를 바꾸지 않고 입력만 Panda 로 바꿨을 때, LIBERO scene 에서 (1) target bowl 을 추출하는가(ATS),
> (2) target 만 제외한 layer 를 만들면서 다른 장애물은 보존하는가(LTE), (3) gripper 주변 거리장이 실제 거리와 mm 수준으로 맞는가(ESDF)?

```
recorded dev frame (wire npz: depth mm, K, T_base_cam, robot_state 9-D) + attention(L8H2, un-mirror)
  -> robot_state 를 q7 + finger 로 split -> AG3S.set_finger_joints -> CameraObservation x2 (head, left_wrist)
  -> AG3S(robot_model = Panda self-filter, constraint_robot_model = Panda gripper, attached_parent_links = Panda,
          config = V11AB flag 에서 유도 + Panda override 둘)
  -> process_multi_debug(phase="approach", active_manipulators=["left"])
  -> grounding(ATS) · target-free layer · CuroboEsdfField · CollisionConstraintSet
```

closed loop(TrajOpt, action 변환)은 SL-5 다. 이 STEP 은 **offline** 이다: 기록된 frame 을 다시 넣을 뿐 로봇은 움직이지 않는다.

### 무엇을 만들었나

**1. Panda AG3S profile** (`AG3S/adapters/panda/profile.py`; AG3S core 는 한 줄도 바꾸지 않았다)

config 를 손으로 다시 적지 않았다. launcher 스크립트(`scripts/benchmark/v11ab/start_server.sh`)의 V11AB flag 를 **serve_safe 자신의 parser · 검사**로 읽고,
`serve_safe.main` 의 `build_ag3s(...)` 호출식(AST)에서 keyword 를 만들어, `build_ag3s` 의 config 조립(유일하게 피할 수 없는 복사본)으로 config 를 만든다.
test 가 **실제 `build_ag3s`** 를 같은 keyword 로 돌려 AG3S 생성자에 넘어가는 config 와 field 별로 대조한다.
Panda 와 달라지는 것은 아래 **override** 뿐이다. `diff_against_v11ab()` 가 이 목록 밖의 차이를 `UNDECLARED` 로 잡는다.

| override | V11AB | Panda | 이유 |
|---|---|---|---|
| `contact.contact_links` (config) | `left: (ee_left, ee_finger_l1, ee_finger_l2)`, `right: (ee_right, ee_finger_r1, ee_finger_r2)` | `left: (panda_hand, panda_leftfinger, panda_rightfinger)` | RB-Y1 link 이름이 Panda 에 없다. Panda 한 팔 = left manipulator(SL2). LTE 가 target 접촉을 허용하는 link 가 palm + 두 finger 가 된다 |
| `esdf.bounds_lower` z (config) | `(-0.3, -0.9, 0.0)` (`default_bounds()`) | `(-0.3, -0.9, -0.10)`, x · y 와 `bounds_upper` 는 그대로 | V11AB box 의 바닥은 base z 0.0(RB-Y1 의 table 은 base 보다 훨씬 위). **LIBERO table 윗면은 base z −0.012 m**(SL1)여서 기본 box 가 table 을 잘라 낸다 |

config 밖의 **constructor 차이**(로봇 geometry 에서 오는 것):

| 무엇 | V11AB | Panda |
|---|---|---|
| self-filter model | RB-Y1 `build_robot_model` | `build_panda_self_filter_model()` (`confined_hand_mount` cover, 4,271 sphere) |
| constraint model | `build_constraint_robot_model(links="gripper", …)` | `build_panda_constraint_model(PANDA_GRIPPER_LINKS)` (22 sphere) |
| `attached_parent_links` | `("ee_left", "ee_right")` | `("panda_hand",)` |
| constraint sphere options(`capsule_radius_scale` 0.05 등) | 적용 | **적용 안 함**: `franka.yml` sphere 는 길이 0 capsule 이라 scale 0.05 가 반지름을 5 % 로 줄인다(SL2) |
| `finger_cover: both` | 적용 | **적용 안 함**: RB-Y1 finger body capsule 용이다(SL2) |

V11AB flag 에서 유도한 값은 task 표와 대조해 일치했다(`esdf.voxel_size` 0.02, `max_distance` 0.4, `fine_voxel_size` 0.005, `tsdf_voxel_size` 0.005, `exclude_support_surfaces` False, backend `curobo`,
`attached_sign_threshold_voxels` 1.5, `pointcloud.range_max` 2.0, `self_collision` False). client 의 phase flag 는 `approach` 이다(V11AB client 는 `--safe-phase approach` 를 보낸다).

**2. offline harness** (`benchmark/runners/perception_replay.py`, `.venv-openpi-live`, GPU)

SL3 dev rollout 의 policy-call frame 221 개를 V11AB 의 실제 perception 경로로 다시 넣는다: `wire_pack` → `wire.strip_request` → `wire.unpack_camera_observations` →
robot_state 9-D 를 q7 + finger 로 split → `set_finger_joints` → SL3 audit 의 call 별 L8H2 grid 를 un-mirror 해 `CameraObservation.attention` 에 부착 →
`process_multi_debug(phase="approach", active_manipulators=["left"])`. 입력 join 은 dev run ↔ audit run 의 episode npz · geom SHA-256 과 `obs_index == chunk_t` 가 맞아야 시작한다.
`--attention-mapping {unmirror,mirror}`(진단용), `--esdf-bounds {panda,v11ab}`(#7 용)를 둔다. 저장 항목: frame 별 grounding 결과(status, target 점, seed 와 camera provenance),
fused cloud 요약, ESDF 를 GT 점 · gripper sphere 중심에서 query 한 값(main layer 와 target-free layer), constraint 요약, stage 별 시간, GPU memory.

**3. scene GT 도구** (`benchmark/adapters/libero_panda/scene_export.py`(py3.8) · `benchmark/evaluators/libero_scene_gt.py`(3.11))

episode 마다 LIBERO env 를 다시 만들어 call frame 의 MuJoCo 상태를 복원하고, 물체(target, plate, bowl_2, cookies, ramekin, stove, cabinet, table)와 robot mount 의
**visual mesh 표면 sample**(4 mm, instance 당 최대 3,000 점)을 base frame 으로 내보낸다. 점–삼각형 **정확 거리**, collision primitive 포함 판정, 기록 depth 와 점의 camera z 차이(depth 잔차)를 제공한다.
**visible GT sample** = `|기록 depth − sample 의 camera z| ≤ 10 mm`(head 또는 left_wrist). 바닥면 · 가려진 면은 sample 에 들어 있어서 이 조건으로 거른다.
검증 쪽에서 독립 구현(KD-pruned)과 brute force 가 300 점에서 최대 차 0.0 으로 같음을 확인했다.

**4. test** — `tests/benchmark/test_panda_sl4.py` 추가(CPU). profile config 대 실제 `build_ag3s` config, diff 가 선언한 override 와 정확히 같음, robot_state split · finger 부호, attention 부착 방향, ESDF bound 적용, harness dry-run, py3.8 export.

### 실행

| run | 값 |
|---|---|
| source dev rollout / attention audit | `.../20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0` / `.../20261012T202459_309e4c6da477_sl3-attention-audit-dev` (SL3, 읽기만) |
| scene GT | `outputs/verify/SL4/raw/scene` (`.venv-libero`, CPU) |
| 기본 run (un-mirror, Panda box) | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T225820_46a08d6ff644_sl4-perception-dev` — 실행 창 13:58:18–14:03:19 UTC, GPU lock 안 |
| mirror run (#4 진단) | `.../20261012T230325_46a08d6ff644_sl4-perception-dev-mirror` — `--attention-mapping mirror` |
| V11AB box run (#7) | `.../20261012T230818_46a08d6ff644_sl4-perception-dev-v11abbox` — `--esdf-bounds v11ab` |
| flags-off replay (SL0 경로 보존) | `.../20261012T232117_46a08d6ff644_sl4-replay-off-t0e0` |
| 기준선 | legacy 16D(`.venv-ag3s`, CPU), cuRobo 16D(`.venv-openpi-live`, GPU lock 안, compute process 없음 확인) |

세 run 모두 같은 frame 221 개에 대해 `gt_samples_equal_harness` · `obs_index_equal_sl3_phases` 가 true 다(GT sample 과 phase label 이 harness 와 같다). phase 는 SL3 와 같다: approach 105 · grasp 25 · transport 91 call(post_release 0).
**carry** = grasp + transport(116 call).

### 결과 — Figure

세 figure 는 verifier 가 만들었다 (`outputs/verify/SL4/scripts/make_figures.py`, sidecar `.json` 동봉).

**실제 scene** — 5 개 frame(t0 의 approach · grasp · transport, ATS 가 오선택한 t4 의 approach, t5 의 approach)을 행으로, 각 행은 (1) 위에서 본 배치도(xy slice box 점선, xz 절단선 빨강),
(2) 옆에서 본 배치도, (3–4) xy 평면의 **main layer** 와 **target-free layer** ESDF(검은 contour = ESDF 5 mm), (5–6) xz 단면의 main 과 target-free 이다. fused cloud 는 물체별 색,
ATS target 점은 검정, 22 gripper constraint sphere 는 흰 원, target-free ball 은 빨강 점선이다. t0 approach · grasp 행은 ATS 점이 target bowl 위에 있고 target-free layer 에서 그 bowl 만 지워진다(visible plate sample 1,994 중 erased 1 · 0, bowl_2 erased 0).
t0 transport 행(attach 없음)은 ATS 가 target 이 아닌 곳에 있고, t4 · t5 행은 ATS 점이 각각 plate · bowl_2 위에 있어 **target-free layer 에서 지워지는 것이 true target 이 아니라 그 물체다**(t4: visible plate sample 1,964 중 568 이 erased, t5: visible bowl_2 sample 2,141 중 1,747 이 erased).

![SL4 scene](figures/sl4-scene.png)

**graph** — (a) call 별 ATS precision · recall(+ mirror run 의 recall 선), (b) call 별 centroid 오차(log), (c) 물체별 visible GT 점의 occupied 비율(main · target-free), (d) phase 별 LTE,
(e) camera block 별 attention mass 분해(target · plate · gripper · other), (f) seed 의 head-argmax 비율, (g) ESDF box 효과, (h) gripper sphere 중심의 ESDF − 정확 거리, (i) stage 시간.

![SL4 graph](figures/sl4-graph.png)

**table** — profile 대 V11AB 표, 기준선 · 회귀, ATS · mirror, attention mass, LTE, field · box · mount · cost 요약.

![SL4 table](figures/sl4-table.png)

### 결과 — #1 V11AB 동일성

| 확인 | 결과 |
|---|---|
| core diff (`AG3S/ag3s/**` · `AG3S/trajopt/**`, 시작 HEAD `d1605030` 이후) | **0 줄**, untracked 0 |
| profile config 대 V11AB 차이 (profile 자체의 `diff_against_v11ab()`) | `contact.contact_links`, `esdf.bounds_lower`(z) 둘 — 선언한 override 와 정확히 같다 |
| 검증 쪽 독립 비교(`to_dict` 의 149 leaf) | Panda box: contact links + `bounds_lower` z(+ `bounds_upper` 의 `None` → 명시 기본값). V11AB box(`--esdf-bounds v11ab`): contact links 만 |
| 기준선: legacy 16D (15 frame, `.venv-ag3s`) | violated start 14/15 · has_target 15/15 · frame0 clearance −29.031 mm — **일치** |
| 기준선: cuRobo 16D (15 frame, `.venv-openpi-live`, GPU lock) | violated start 10/15 · has_target 15/15 · frame0 clearance −9.172 mm — **일치** |

**RB-Y1 기준선 둘이 그대로이므로** core 는 바뀌지 않았다는 독립 확인이다(task 의 중단 조건: 기준선이 어긋나면 즉시 중단 — 해당 없음).

### 결과 — #2 ATS: target 추출

용어: **precision** = ATS target 점 중 GT target(`akita_black_bowl_1`) 표면까지 정확 거리 ≤ 10 mm 인 비율. **recall** = visible GT target sample 중 10 mm 안에 ATS target 점이 있는 비율.
**centroid 오차** = |ATS target centroid − visible GT target sample 의 centroid|. `has_target` 은 ATS 가 target 점을 낸 call 이다. 실패 status: `no_seed`(seed 없음), `no_cluster`, `no_admissible`.

**phase 별** (un-mirror, 기본 run)

| phase | call | has_target | status | precision (pooled) | precision (call median) | recall (call median, target 있는 call) | recall (target 없음 = 0 평균) | centroid 오차 median / p95 (mm) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| all | 221 | 179 | ok 179, no_cluster 27, no_seed 11, no_admissible 4 | 0.497 | 0.671 | 0.231 | 0.306 | 33.2 / 382.6 |
| approach | 105 | 101 | ok 101, no_admissible 4 | 0.696 | 1.000 | 0.732 | 0.489 | 12.0 / 382.1 |
| grasp | 25 | 24 | ok 24, no_seed 1 | 0.769 | 1.000 | 0.673 | 0.565 | 16.3 / 312.2 |
| transport | 91 | 54 | ok 54, no_cluster 27, no_seed 10 | 0.056 | 0.000 | 0.000 | 0.024 | 155.3 / 420.3 |
| carry | 116 | 78 | ok 78, no_cluster 27, no_seed 11 | 0.270 | 0.000 | 0.000 | 0.141 | 96.9 / 407.4 |

- **접근(approach) 단계에서 ATS 는 대체로 target 을 잡았다**: 105 call 중 101 call 에서 target 을 냈고(실패 4 = `no_admissible`, 모두 t4), centroid 오차 median 12.0 mm, precision call median 1.000.
  단 p95 는 382.1 mm 로 **일부 task 에서 아예 다른 물체**를 골랐다(아래 task 별).
- transport 에서는 ATS 가 target 이 아닌 곳에 있다(precision 0.056). **이 구간은 attach 가 없는 offline 한계의 영향을 받는다**(아래 "알게 된 것" c).
- centroid 오차는 GT body 원점이 아니라 **visible 표면 sample 의 centroid** 와 비교한 값이다(GT body 원점은 bowl 바닥이라 표면 centroid 와 다르다; GT body 원점 기준 centroid 오차 median 은 approach 38.5 mm).
- ATS target centroid 가 직전 call 과 완전히 같은 call 이 target 이 있는 179 call 중 125 개다. 그 의미(target 유지 규칙인지)는 확인하지 않았다.

**task 별** (모든 phase, 한 task = episode 하나이므로 call 은 서로 독립이 아니다). 마지막 열은 ATS target 점의 **nearest-instance** 분포(0.05 이상만; `none` = 10 mm 안에 어느 물체도 없음).

| task | 설명 (target 은 모두 `akita_black_bowl_1`) | call | has_target | precision (pooled) | recall median | centroid 오차 median / p95 (mm) | target 점의 nearest-instance |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | between plate and ramekin | 21 | 19 | 0.909 | 0.750 | 12.6 / 46.7 | bowl_1 0.91, none 0.09 |
| 1 | next to ramekin | 22 | 16 | 0.739 | 0.209 | 41.7 / 325.2 | bowl_1 0.74, none 0.21 |
| 2 | table center | 27 | 22 | 0.818 | 0.765 | 6.1 / 240.8 | bowl_1 0.82, none 0.18 |
| 3 | on the cookie box | 18 | 14 | 0.771 | 0.765 | 9.1 / 169.7 | bowl_1 0.77, none 0.18 |
| 4 | in the top drawer of the cabinet | 24 | 19 | 0.000 | 0.000 | 376.9 / 407.3 | plate 1.00 |
| 5 | on the ramekin | 20 | 20 | 0.119 | 0.000 | 211.6 / 303.1 | bowl_2 0.49, none 0.23, ramekin 0.17, bowl_1 0.10 |
| 6 | next to the cookie box | 21 | 19 | 0.638 | 0.474 | 13.9 / 143.3 | bowl_1 0.64, none 0.22, plate 0.14 |
| 7 | on the stove | 23 | 15 | 0.621 | 0.485 | 14.0 / 462.3 | bowl_1 0.62, none 0.31 |
| 8 | next to the plate | 22 | 20 | 0.640 | 0.145 | 51.6 / 158.5 | bowl_1 0.64, none 0.36 |
| 9 | on the wooden cabinet | 23 | 15 | 0.000 | 0.000 | 371.3 / 393.3 | bowl_2 0.79, plate 0.12, stove 0.09 |

- **t4 · t5 · t9 에서 ATS 가 true target 이 아닌 물체를 골랐다.** t4(top drawer 안의 bowl): target 점의 nearest-instance 가 **plate 100 %**, precision 0.000(plate 는 destination 이다).
  t5(ramekin 위의 bowl): **bowl_2 0.49**, ramekin 0.17, true target(bowl_1) 0.10, precision 0.119. t9(cabinet 위의 bowl): **bowl_2 0.79**, plate 0.12, stove 0.09, bowl_1 은 0, precision 0.000.
- 나머지 7 task 는 target 점이 bowl_1 위에 놓인 비율이 0.62–0.91(t0 0.91 · t2 0.82 · t3 0.77 · t1 0.74 · t8 0.64 · t6 0.64 · t7 0.62)이다.

**phase 별 nearest-instance** (ATS target 점 pooled, 비율)

| phase | bowl_1 | bowl_2 | plate | ramekin | cookies | stove | table | robot_sphere | none |
|---|---|---|---|---|---|---|---|---|---|
| approach | 0.696 | 0.214 | 0.063 | 0.006 | 0.004 | 0.015 | 0 | 0.000 | 0.001 |
| grasp | 0.757 | 0 | 0.071 | 0.027 | 0.004 | 0.002 | 0 | 0.028 | 0.110 |
| transport | 0.053 | 0.125 | 0.250 | 0.054 | 0.006 | 0.019 | 0.004 | 0.004 | 0.485 |

### 결과 — #3 seed 를 만든 camera

**seed** = ATS 가 attention 상위(95 percentile) 점으로 고른 점이다. 점마다 per-camera attention 이 더 큰 camera 를 **argmax camera** 로 기록했다(core `_with_camera_provenance` 와 같은 규칙).
SL3 열린 항목 1(head block mass 가 매우 작은데 head pixel 이 seed 에 들어가는가)의 답이다.

| phase | seed 수 | argmax head / wrist | 관측한 camera: head 만 / wrist 만 / 둘 다 | head-argmax seed 가 하나라도 있는 call | head 가 과반인 call | call 별 head seed 비율 median | (mirror run) argmax head 비율 |
|---|---:|---|---|---:|---:|---:|---:|
| all | 19386 | 0.587 / 0.413 | 0.190 / 0.226 / 0.584 | 204 / 210 | 142 / 210 | 0.663 | 0.513 |
| approach | 12906 | 0.520 / 0.480 | 0.132 / 0.267 / 0.601 | 105 / 105 | 65 / 105 | 0.621 | 0.394 |
| grasp | 1670 | 0.640 / 0.360 | 0.346 / 0.210 / 0.444 | 23 / 24 | 16 / 24 | 0.609 | 0.664 |
| transport | 4810 | 0.749 / 0.251 | 0.294 / 0.122 / 0.584 | 76 / 81 | 61 / 81 | 0.791 | 0.598 |

- **head 가 seed 를 만든다.** 전체 seed 의 0.587 이 head-argmax 이고, seed 가 있는 210 call 중 204 call 에 head-argmax seed 가 있다. 접근 단계는 head 0.520 · wrist 0.480 으로 거의 반반이고 transport 로 갈수록 head 가 늘어난다(0.749).
  head block mass 가 작다는 SL3 의 우려는 **seed 구성에서는 문제가 되지 않았다**(fusion 에서 voxel 별 max, fused cloud 전체 percentile 정규화를 거치므로 block 합이 작아도 head 점이 상위에 들어간다는 구조적 설명은 이 STEP 에서 따로 검증하지 않았다).
- seed 점은 두 camera 가 모두 관측한 점이 0.584 로 가장 많다.
- ATS target 의 `seed_camera` 필드는 target 이 있는 179 call 중 head 23 · left_wrist 22 로 기록되고 134 call 은 빈 값이다(빈 값의 의미는 확인하지 않았다).

### 결과 — #4 wrist mirror 진단: attention mass 분해와 mirror run

SL3 #5 의 불일치(image 검사는 un-mirror 가 맞는데 wrist 의 target mass 평균은 mirror left in 이 높다)를 phase 별로 분해한다. **mass 분해** = call 평균. camera 의 256 token 합을 1 로 정규화했을 때 물체 mask 가 덮는 patch 가중 합이다
(`other` = 1 − target − destination − gripper; arm · fixture · 기타 물체 · rest 가 포함된다). 마지막 열은 같은 patch 가중으로 센 target / destination 의 면적 비율(uniform baseline)이다.

| camera | mapping | phase (call) | target | destination (plate) | gripper | other | area target / destination |
|---|---|---|---:|---:|---:|---:|---|
| head | unmirror | approach (105) | 0.214 | 0.098 | 0.018 | 0.670 | 0.0124 / 0.0208 |
| head | unmirror | grasp (25) | 0.072 | 0.397 | 0.017 | 0.514 | 0.0081 / 0.0198 |
| head | unmirror | transport (91) | 0.025 | 0.578 | 0.005 | 0.392 | 0.0120 / 0.0183 |
| head | mirror | approach (105) | 0.043 | 0.006 | 0.005 | 0.946 | 0.0124 / 0.0208 |
| head | mirror | grasp (25) | 0.002 | 0.001 | 0.005 | 0.991 | 0.0081 / 0.0198 |
| head | mirror | transport (91) | 0.001 | 0.000 | 0.001 | 0.999 | 0.0120 / 0.0183 |
| left_wrist | unmirror | approach (105) | 0.315 | 0.023 | 0.092 | 0.571 | 0.0682 / 0.0465 |
| left_wrist | unmirror | grasp (25) | 0.214 | 0.144 | 0.398 | 0.244 | 0.1634 / 0.1186 |
| left_wrist | unmirror | transport (91) | 0.022 | 0.619 | 0.056 | 0.303 | 0.1755 / 0.0488 |
| left_wrist | mirror | approach (105) | 0.225 | 0.033 | 0.092 | 0.650 | 0.0682 / 0.0465 |
| left_wrist | mirror | grasp (25) | 0.171 | 0.029 | 0.399 | 0.400 | 0.1634 / 0.1186 |
| left_wrist | mirror | transport (91) | 0.407 | 0.126 | 0.056 | 0.411 | 0.1755 / 0.0488 |

- **transport 에서 policy 의 attention 은 target bowl 을 떠나 destination plate 로 옮겨 간다.** un-mirror 로 head 는 plate 0.578 · target 0.025, wrist 는 plate 0.619 · target 0.022 다.
  grasp 에서 이미 옮겨 가는 중이다(head plate 0.397; wrist 는 gripper 0.398 · target 0.214 · plate 0.144).
- mirror left in 으로 바꾸면 head 는 전 phase 에서 target · plate mass 가 0 에 가까워진다(transport 0.001 · 0.000). 즉 head 는 **un-mirror 가 맞다**.
- wrist 는 approach · grasp 에서 un-mirror 가 target mass 가 높고(0.315 대 0.225, 0.214 대 0.171), transport 에서만 mirror 가 높다(0.407 대 0.022). 그 mirror 값의 내용은 plate 0.126(un-mirror 0.619)이다.
  **mirror left in 이 wrist transport 에서 높았던 것은 mapping 이 맞아서가 아니다.** attention 이 plate 에 있었는데 뒤집으면 plate(wrist image 면적 0.049)에서 빠져나와 면적이 큰 target bowl(0.176)에 걸린 것이다.

**mirror 로 harness 를 돌린 ATS** (같은 221 call, 다른 것은 mapping 뿐)

| phase | call | has_target | status | precision (pooled) | precision (call median) | recall (call median) | recall (target 없음 = 0 평균) | centroid 오차 median / p95 (mm) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| all | 221 | 202 | no_admissible 12, ok 202, no_cluster 5, no_seed 2 | 0.368 | 0.000 | 0.000 | 0.213 | 148.4 / 557.7 |
| approach | 105 | 89 | no_admissible 12, ok 89, no_cluster 3, no_seed 1 | 0.548 | 0.000 | 0.000 | 0.289 | 236.2 / 567.5 |
| grasp | 25 | 25 | ok 25 | 0.677 | 1.000 | 0.364 | 0.392 | 27.1 / 351.5 |
| transport | 91 | 88 | ok 88, no_cluster 2, no_seed 1 | 0.120 | 0.000 | 0.000 | 0.075 | 155.9 / 540.7 |
| carry | 116 | 113 | ok 113, no_cluster 2, no_seed 1 | 0.231 | 0.000 | 0.000 | 0.143 | 116.6 / 504.4 |

- 접근 단계에서 mirror 는 ATS 를 크게 해친다: precision(pooled) 0.696 → 0.548, recall median 0.732 → 0.000, centroid 오차 median **12.0 mm → 236.2 mm**. mirror 에서는 `no_admissible` 이 12 call 로 늘고 ATS 가 ramekin(nearest-instance 0.18) · cookies(0.11)를 target 으로 고르는 경우가 생긴다.
- **ATS 결과로도 un-mirror 가 확정**이다. 위 mirror 의 grasp precision(0.677)은 un-mirror(0.769)보다 낮고, transport 의 precision(0.120 대 0.056)은 높지만 둘 다 위 offline 한계 구간이다.
- mirror run 의 seed camera: 접근 단계 head-argmax 비율 0.394(un-mirror 0.520)로 head 의 기여가 줄어든다.

### 결과 — #5 장애물 보존과 LTE

용어: **occupied** = ESDF 값 ≤ 5 mm(20 mm 열은 따로 표시). **erased** = main layer 에서 occupied 이고 target-free layer 에서 > 5 mm. 대상은 **visible GT sample**(depth 잔차 ≤ 10 mm)이고 221 call 을 pooled 했다.
target-free layer 는 217 call 에 있다(approach 101 · grasp 25 · transport 91; 나머지 4 call 은 ATS 가 target 을 내지 못함).

| 물체 | visible 점 | main occupied (5 mm) | target-free occupied (5 mm) | main occupied (20 mm) | erased 비율 | erased: approach / grasp / transport | erased > 0 / ≥ 10 % 인 call |
|---|---:|---:|---:|---:|---:|---|---|
| plate | 419732 | 0.984 | 0.941 | 0.999 | 0.0436 | 0.023 / 0.050 / 0.068 | 46 / 26 |
| bowl_2 | 295531 | 0.911 | 0.776 | 0.990 | 0.1359 | 0.161 / 0.000 / 0.136 | 64 / 30 |
| cookies | 125777 | 0.996 | 0.992 | 1.000 | 0.0032 | 0.006 / 0.000 / 0.000 | 14 / 1 |
| ramekin | 193839 | 0.964 | 0.947 | 0.998 | 0.0176 | 0.009 / 0.028 / 0.032 | 103 / 10 |
| stove | 261022 | 0.958 | 0.942 | 1.000 | 0.0161 | 0.011 / 0.017 / 0.022 | 54 / 11 |
| cabinet | 84514 | 0.718 | 0.717 | 0.910 | 0.0006 | 0.001 / 0.001 / 0.000 | 33 / 0 |
| table | 93353 | 0.979 | 0.978 | 0.993 | 0.0004 | 0.001 / 0.000 / 0.000 | 19 / 0 |
| bowl_1 | 392444 | 0.823 | 0.575 | 0.959 | 0.2477 | 0.353 / 0.459 / 0.054 | 194 / 99 |
| mount | 11640 | 0.000 | 0.000 | 0.014 | 0.0000 | 0.000 / 0.000 / 0.000 | 0 / 0 |

- **비target 물체는 main layer 에서 거의 그대로 보존된다**: plate 0.984 · cookies 0.996 · table 0.979 · ramekin 0.964 · stove 0.958 · bowl_2 0.911(5 mm). 20 mm 로 보면 0.99 이상이다(cabinet 만 0.910).
  cabinet 의 main occupied 는 5 mm 에서 0.718 로 낮다(fine layer 구간에서는 0.992). 이 낮음의 원인은 분석하지 않았다.
- **target-free layer 에서 true target(bowl_1)은 부분적으로만 지워진다**: 5 mm 로 target-free 에서 제거된 비율 approach 0.364 · grasp 0.551 · transport 0.473(erased 만 세면 0.353 · 0.459 · 0.054). 이유는 둘이다 — ATS recall 이 완전하지 않고, **t4 · t5 · t9 에서는 다른 물체를 골랐다**. 두 이유의 몫은 분해하지 않았다.
- **오선택이 비target 물체를 지운다.** bowl_2 는 approach 에서 0.161 이 erased 이고 16 call 은 10 % 이상 지워진다(t5 · t9 가 bowl_2 를 target 으로 고른 것과 같은 방향이다; task 별로 분해하지는 않았다). plate 는 transport 0.068, 전체 26 call 에서 10 % 이상 지워진다(t4 의 plate 선택과 같은 방향).
  가장 큰 것은 **true target 이 target-free layer 에도 남는 것**이다(5 mm 로 approach 0.636, grasp 0.449, transport 0.527 가 target-free 에서도 occupied).
- cookies · ramekin · stove · table · cabinet 은 erased 비율이 0.032 이하다. mount 는 어느 layer 에도 occupied 가 아니다(0.000; 20 mm 0.014).

| phase | target visible 점 | main occupied (5 mm) | target-free occupied (5 mm) | target-free 에서 제거(5 mm) | target-free 에서 제거(20 mm) | erased | erased ≥ 10 % 인 call / call |
|---|---:|---:|---:|---:|---:|---:|---|
| approach | 201696 | 0.989 | 0.636 | 0.364 | 0.192 | 0.353 | 66 / 105 |
| grasp | 38729 | 0.909 | 0.449 | 0.551 | 0.257 | 0.459 | 23 / 25 |
| transport | 152019 | 0.581 | 0.527 | 0.473 | 0.184 | 0.054 | 10 / 91 |

transport 의 main layer 에서 target 이 0.581 만 occupied 인 것은 offline 한계 구간이라 원인을 분해하지 않았다.

### 결과 — #6 field 정확도: 22 gripper constraint sphere 중심의 ESDF 값

**err** = ESDF(sphere 중심) − 정확 거리(그 점에서 visual 표면까지의 최근접 거리, mount 제외, 정확 거리 ≤ 0.4 m, 중심이 collision primitive 안인 sphere 제외). 음수 = ESDF 가 실제보다 가깝다고 말함(안전 쪽).
`tf` 는 target-free layer 를 target 까지 뺀 정확 거리와 비교한 것이다. n 은 sphere × call 쌍 수다.

| layer | phase | n | err median (mm) | err p05 / p95 | \|err\| median | \|err\| p95 |
|---|---|---:|---:|---|---:|---:|
| main | all | 4862 | -2.26 | -34.35 / 19.19 | 4.73 | 38.60 |
| main | approach | 2310 | -2.67 | -8.36 / 0.48 | 2.73 | 8.36 |
| main | grasp | 550 | -3.91 | -13.01 / 6.62 | 4.74 | 13.38 |
| main | transport | 2002 | 4.12 | -55.25 / 25.57 | 11.93 | 55.25 |
| tf | all | 4862 | -21.09 | -189.43 / -0.92 | 21.09 | 189.43 |
| tf | approach | 2310 | -4.80 | -39.50 / -0.33 | 4.83 | 39.50 |
| tf | grasp | 550 | -11.38 | -41.71 / -1.18 | 11.38 | 41.71 |
| tf | transport | 2002 | -122.16 | -218.09 / -14.63 | 122.16 | 218.09 |

- **접근 단계에서 main layer 는 mm 수준으로 맞는다**: \|err\| median 2.73 mm, p95 8.36 mm(최대 23.4 mm), 오차는 일관되게 음수(median −2.67 mm)라 **안전 쪽 bias** 다. grasp 는 median 4.74 · p95 13.38 mm.
- **transport 에서는 오차가 커진다**(\|err\| median 11.93 · p95 55.25 mm, 부호도 양수로 바뀐다 median +4.12). 이 구간은 attach 가 없는 offline 구간이며 원인은 분해하지 않았다.
- approach 에서 coarse tier 의 \|err\| median 은 5.08 mm, fine tier 는 2.57 mm 로 fine layer 가 더 정확하다. transport 는 두 tier 모두 크다(12.12 · 11.90 mm).

| phase | tier | n | \|err\| median | \|err\| p95 |
|---|---|---:|---:|---:|
| approach | coarse | 287 | 5.08 | 11.45 |
| approach | fine | 2023 | 2.57 | 7.34 |
| grasp | coarse | 89 | 5.15 | 12.61 |
| grasp | fine | 461 | 4.68 | 13.69 |
| transport | coarse | 265 | 12.12 | 74.31 |
| transport | fine | 1737 | 11.90 | 51.21 |

정확 거리 구간별(\|err\| 의 n / median / p95, mm):

| phase | 0–30 mm | 30–100 mm | 100–400 mm |
|---|---|---|---|
| all | 472 / 9.34 / 24.73 | 2459 / 7.33 / 49.81 | 1931 / 2.67 / 9.30 |
| approach | 28 / 4.17 / 12.39 | 490 / 3.38 / 7.99 | 1792 / 2.57 / 8.36 |
| grasp | 81 / 6.07 / 17.66 | 393 / 4.75 / 13.37 | 76 / 4.05 / 7.68 |
| transport | 363 / 10.79 / 25.99 | 1576 / 12.22 / 58.56 | 63 / 12.90 / 92.14 |

- **target-free layer(`tf`)의 오차는 target 을 뺀 정확 거리와 비교하면 크다**(approach \|err\| median 4.83 · p95 39.50 mm, transport median −122.16 mm). 정확 거리에 **target 을 포함해** 비교하면 오차가 줄어든다(아래 표; transport median +5.25 mm).
  즉 tf layer 가 target bowl 을 아직 장애물로 갖고 있다는 뜻이다(위 #5: true target 이 tf 에 남는 것과 같은 현상).

| phase | n | tf − 정확 거리(target 포함) median (mm) | p05 / p95 |
|---|---:|---:|---|
| all | 4862 | 0.37 | -33.73 / 31.60 |
| approach | 2310 | -1.60 | -7.60 / 28.99 |
| grasp | 550 | 16.08 | -6.74 / 36.57 |
| transport | 2002 | 5.25 | -53.41 / 35.12 |

- eikonal 자가진단(`|grad ESDF|`, xz 단면; call 별 median 의 median): 0.992(p05 0.936), `|grad| > 1.1` 인 비율의 median 0.0005. xy 단면은 GT target 높이에서 자른 것이라 table 접촉면 근처의 0 gradient 가 섞여 이 지표로 쓰지 않았다.
- V11AB box 에서 돌린 run 의 값도 같은 수준이다(all, main \|err\| median 4.30 · p95 36.88 mm; approach \|err\| median 2.69 · p95 8.00 mm).

### 결과 — #7 ESDF box: z 하한 −0.10 m override 의 효과

같은 frame · 같은 ATS 로 두 box 만 바꿔 돌렸다(cloud 점 수와 call 별 `has_target` 은 둘이 같다).

| 지표 | Panda box (z ≥ −0.10) | V11AB box (z ≥ 0.0) |
|---|---:|---:|
| **table visible 점의 ESDF ≤ 5 mm 비율** | **0.979** | **0.005** |
| table visible 점의 ESDF ≤ 20 mm 비율 | 0.993 | 0.114 |
| main occupied coarse voxel 수 (call median) | 2,440 | 259 |
| fused cloud 점 중 box 밖 비율 | 0.0014 | 0.670 |

- table 점은 전부(1.000) base z < 0 에 있어 V11AB box 에서는 field 안에 표현되지 않는다. cloud 점의 0.670 이 box 밖(z < 0)이다.
- 그런데 gripper sphere 중심의 main ESDF 값 차이(V11AB box − Panda box)는 median 0.00 mm, p05 −4.79 · p95 +5.81 mm 로 작다(차이 > 1 mm 인 sphere 는 0.641 이지만 크기가 작다). 이 dev frame 에서 gripper 가 table 을 가장 가까운 장애물로 갖는 경우가 드물어 값이 거의 안 바뀐 것으로 보이나 분해하지는 않았다.
  **그래도 table 이 장애물로 field 에 없다는 점은 바뀌지 않는다**(위 0.005) — 이것이 override 가 필요한 이유다.

### 결과 — #8 mount 잔여

self-filter(confined hand + mount cover) 뒤에 남은 robot mount pedestal 점이 field 에 들어가는지 본다(SL3 열린 항목 5).

| 지표 | 값 |
|---|---|
| mount 잔여 cloud 점이 있는 call | 217 / 221 |
| mount 잔여 점 수 | 총 1,225, call 당 median 7(최대 7) |
| 잔여 점 → 22 gripper sphere 최소 거리 | median 616 mm(min 326 mm) |
| 잔여 점 → franka 61 sphere 전체 최소 거리 | median 150 mm(min 150 mm) |
| mount GT 표면 sample 중 main ESDF ≤ 5 mm | 1 / 663,000 (1 call; 그 점에서 gripper sphere 까지 336 mm) |
| 잔여 점 근처(10 mm 안) mount GT sample 의 ESDF | median 19.3 mm(164 call), ≤ 5 mm 인 call 1 |

mount 점이 field 에 남지만 call 당 최대 7 점이고 gripper sphere 에서 326 mm 이상 떨어져 있다. 잔여 점 자체의 ESDF 값은 재지 않았다(근처 GT sample 로만 봄).

### 결과 — #9 비용

각 episode 의 첫 call 은 제외했다(n = 211). 첫 call 은 JIT 때문에 6.73 s 였고, 각 episode 의 첫 call 에 3.80 s 까지 걸린 경우가 있다(원인은 확인하지 않았다).

| stage | p50 (ms) | p95 (ms) |
|---|---:|---:|
| scene_reconstruction (back-project · downsample · **self-filter 4,271 sphere** · fusion) | 38.8 | 44.1 |
| support_surface | 2.9 | 3.6 |
| attention_lifting | 0.007 | 0.010 |
| target_grounding (ATS) | 3.4 | 15.6 |
| esdf | 45.0 | 60.9 |
| constraint_generation (esdf 포함) | 46.1 | 61.9 |
| **t_process (frame 전체)** | **94.6** | **117.1** |
| (진단) self-filter mask 만: head / left_wrist | 17.1 / 19.1 | 17.8 / 22.9 |

- GPU: torch peak 0.867 GB(866,799,616 B), reserved 1.141 GB, `nvidia-smi` process 1,702 MiB. 진단용 self-filter mask 를 다시 만들어 frame 의 mask 와 비교한 것은 442 / 442 pixel 단위로 같다.
- **이 값은 perception 만이다**(policy 추론 · TrajOpt 제외). 250 ms policy 주기(lead 가 지정한 budget; `verify.json` 값이 아니다)에는 p95 117.1 ms 로 들어간다. policy 와 합친 online 비용은 SL-5 에서 잰다.
- harness 의 call 당 기록 오버헤드(npz · slice 저장)는 mean 1.09 s 이며 `t_process` 에 포함되지 않는다.

### 결과 — #10 회귀

| 검사 | 결과 |
|---|---|
| `scripts/testing/run_cpu_regression.sh` | A 73 passed / B 92 passed / C 90 passed, 2 skipped / exit 0 (SL3 은 A 73 · B 78 · C 76 + 2 skipped; B · C 의 증가는 SL4 test 추가) |
| flags-off replay (SL0 경로) | episode npz 가 SL0 와 모든 key 에서 같다. official_success true · status success · env step 113 · policy call 21 (replay 와 SL0 동일) |

### 이 STEP 에서 알게 된 것 (평이한 설명)

**(a) SL3 의 wrist mirror 수수께끼는 풀렸고, un-mirror 가 맞다.** SL3 에서 wrist 만 mirror left in 이 target mass 가 높게 나온 이유는 transport 에서 policy 의 attention 이
target bowl 이 아니라 **destination plate** 로 옮겨 가기 때문이다(un-mirror 의 plate mass: head 0.578, wrist 0.619; target 은 0.025 · 0.022). 옮겨 간 attention 을 좌우로 뒤집으면 plate 에서 빠져나와
(0.619 → 0.126) 면적이 큰 bowl 위에 걸릴 뿐(0.022 → 0.407)이라 mapping 이 맞아서가 아니다. 접근 단계에서는 mirror 가 ATS 를 실제로 해친다(centroid 오차 median 12.0 → 236.2 mm). SL3 의 가설("중앙의 bowl 이 대칭으로 겹친다")은 이 분해로 대체됐다.

**(b) ATS 는 일부 task 에서 다른 물체를 target 으로 고른다.** t4 는 plate(destination), t5 는 **같은 모양의 bowl_2**, t9 는 true bowl 위에 target 점이 하나도 없다(bowl_2 0.79). LIBERO Spatial 의 핵심 난점이
같은 모양 bowl 두 개를 **위치 관계**("between the plate and the ramekin", "on the ramekin" 등)로 구분하는 것인데, attention 이 그 구분을 못 하면 ATS 는 모양만 닮은 물체를 고른다.
**오선택의 비용은 대칭이 아니다**: 잘못 고른 물체가 target-free layer 에서 지워지고 true target 은 장애물로 남는다. LTE 는 target 만 면제하는 방식이라, 면제가 엉뚱한 곳에 적용되면 grasp 하려는 bowl 이
오히려 contact 금지 장애물이 된다. 이것이 V11AB 평가에 어떻게 반영될지는 closed loop(SL-5)에서 본다. 원인이 attention cell 인지 grounding 기하인지는 SL4b 가 분리한다.

**(c) offline 의 한계.** harness 에는 **grasp latch · attach** 가 없다. 그래서 grasp 이후에도 bowl 은 "쥐지 않은 target" 으로 처리되고(V11AB 였다면 gripper 에 attach 했을 것), **carry(grasp + transport) 구간의 숫자는 쥐지 않은 bowl 의 숫자**로 읽어야 한다.
또 fine window 는 `execution_path` 없이 `q_now` 로만 놓인다. Panda 에는 **phase · subtask probe 가 없다**(probe `kv_L4_v1` 은 RB-Y1 feature 로 학습). 그래서 subtask gate 는 켜져 있으되 label 이 없어 **inert**(ungated)이고, harness 는 V11AB client 처럼 `phase="approach"` 를 모든 frame 에 준다.

**(d) ESDF z 하한 override 는 필요하다.** V11AB 기본 box 에서는 table visible 점의 5 mm occupied 가 0.005 에 불과하고(Panda box 0.979), cloud 점의 0.670 이 box 밖이다. table 이 거리장에 없으면 table 과의 clearance 를 optimizer 가 볼 수 없다.
접근 단계의 gripper 근처 거리장은 이 override 와 별개로 mm 수준으로 맞는다(\|err\| median 2.73 mm).

**(e) 비용은 policy 주기에 들어간다.** perception 만 p50 94.6 ms · p95 117.1 ms(첫 call 제외)이고 가장 큰 stage 는 esdf(45.0 ms)와 self-filter 를 포함한 scene_reconstruction(38.8 ms)이다. 250 ms 주기(lead 지정)보다 작다.
policy 추론과의 합은 SL-5 에서 잰다.

그 밖에: **head 는 seed 를 만든다**(접근 단계 seed 의 head-argmax 0.520). SL3 의 head block mass 우려는 seed 구성에서는 문제가 아니었다. **mount 잔여** 는 call 당 최대 7 점이고 gripper 에서 326 mm 이상이다.

### 판정 (2026-10-13, 사용자)

| # | 판정 |
|---|---|
| 1 | **SL4 확정.** |
| 2 | **SL-4b 를 수행한다 — ATS 용 attention cell dev calibration 연구, 선택 규칙을 사전 등록**([`handoff/SL4b.task.md`](handoff/SL4b.task.md), commit `41a5b28a`). 아래 규칙 참고. test 데이터는 쓰지 않는다 |
| 3 | **SL-5 는 core 를 최소로 바꾼다**: server 에 robot model · contact link · bound 를 **주입**할 수 있게만 하고 RB-Y1 동작은 동일하게 유지한다(기준선으로 확인). 거기에 EE-delta ↔ joint 변환을 더한다 |
| 4 | **SL-6 의 Panda main phase source** 는 gripper-state strong rule 이거나 dev 데이터에서 pi05_libero feature 로 **새로 학습한 probe** 이다. GT phase 는 oracle 비교로만 쓴다. **SL-6 에서 결정** |

**SL4b 사전 등록 규칙** (`SL4b.task.md`; 결과를 보기 전에 고정, 바꾸지 않는다)

| 항목 | 정의 |
|---|---|
| 데이터 | dev 만: task 0–9 × init state 0–4(50 episode). init 0 은 SL3 run 재사용, init 1–4 는 새 rollout. **test(init 5–49)는 사용하지 않는다** |
| 평가 phase | approach + grasp(carry 는 attach 가 없는 offline 한계로 제외) |
| **1차 지표 `target_correct`** | 평가 phase call 중 ATS 가 target 을 냈고, target 점의 nearest-instance 다수결이 true target 이며, centroid 오차 ≤ 30 mm 인 call 의 비율 |
| 2차 지표 | pooled precision(10 mm), recall median, centroid 오차 median / p95, task 별 `target_correct` |
| 후보 | V11AB L8H2 + **선별 지표**(dev 50 episode attention 에서 approach + grasp call 의 head + wrist target mass 평균, un-mirror)의 상위 16 cell. 진단 grid 는 denoise 0, aggregation `last` |
| **채택 조건** | 후보 cell 이 L8H2 보다 1차 지표가 **≥ 10 %p 높고**, task 별로 **10 개 중 ≥ 6 개 task 에서 L8H2 이상**이면 "Panda calibrated cell" 후보로 사용자에게 제안한다. 여러 개면 1차 지표 최고, 동률이면 낮은 layer. 만족하는 cell 이 없으면 L8H2 유지 |
| 원인 분리 | L8H2 의 오선택 call 마다 attention argmax(head · wrist)가 true target 위에 있었는지 → attention 원인 / grounding 원인 |
| 참고(후보 아님) | **oracle**: GT target segmentation 으로 target pixel 만 attention = 1 로 둔 ATS 기하 상한. privileged input 이며 선택에 쓰지 않는다 |

채택 후보를 main 으로 쓸지 variant 로 둘지는 test 전에 사용자가 정한다. 규칙에 따라 판정 문장은 평가기 출력(사실)으로만 기록한다. **SL-4b 의 결과는 이 STEP 에 포함되지 않는다.**

### 측정하지 않은 것

- grasp 이후 attach / held-object 경로(offline harness 에 latch 가 없다; carry 구간은 쥐지 않은 bowl 의 숫자)
- `execution_path` 로 놓이는 fine window(harness 는 `q_now` 로만 놓았다)
- online `serve_safe` 경로(SL-5)
- init state 1–4(SL3 attention 이 init 0 에만 있다; SL4b 에서 만든다)
- mount 잔여 점 자체의 ESDF 값(그 근처 GT sample 로만 봄)
- cabinet 의 main layer 5 mm occupied 가 0.718 로 낮은 원인
- transport 의 main layer · field 오차가 커지는 원인 분해
- #5 의 erased 비율을 task 별로 분해한 것(오선택 task 와의 대응은 방향만 말했다)
- ATS target centroid 가 직전 call 과 같은 call 이 125 / 179 인 이유(target 유지 규칙인지)
- xy 단면의 eikonal 지표(GT target 높이에서 잘라 table 접촉면이 섞인다)

### 운영 기록

| 사건 | 내용 |
|---|---|
| GPU lock `flock` 형식 오류 | task 에 적힌 `flock -w 1800 9 CMD … 9>/tmp/ag3s-vla-gpu-validation.lock` 은 명령이 뒤에 붙으면 `9` 를 **파일 이름**으로 받아 cwd 에 `./9` 를 만들어 잠근다. 공용 lock 은 fd 9 로 열려 있기만 하고 **잡히지 않는다**(바깥 flock 아래에서 다른 `flock -n` 이 성공하는 것으로 재현). implementer 의 GPU smoke 두 번이 이 형식으로 돌아 공용 lock 을 잡지 않았다(시작 직전 compute process 는 비어 있었음). 실제로 lock 을 잡는 형식은 `( flock -w N 9 \|\| exit 1; CMD ) 9>/tmp/ag3s-vla-gpu-validation.lock` 또는 `flock -w N /tmp/ag3s-vla-gpu-validation.lock CMD` 이다. `d53d8e07` 이 규칙 예시를 고쳤다. 검증 run 은 모두 올바른 형식으로 lock 을 잡고 compute process 가 없음을 확인한 뒤 실행했다 |

### 열린 항목

| # | 내용 | 영향 / 다음 |
|---|---|---|
| 1 | **ATS 오선택**: t4(plate), t5(bowl_2), t9(true bowl 위 target 점 0). ~~원인이 attention cell 인지 grounding 기하인지~~ | ~~SL-4b 가 dev 에서 분리하고 사전 등록 규칙으로 cell 후보를 평가한다~~ → SL4b 에서 측정: cell 교체로 해결되지 않는다. 그 뒤 단계의 원인은 SL-4c 가 진단 |
| 2 | **attach · latch · execution_path 없음**(offline). carry 구간 ATS · field 숫자는 쥐지 않은 bowl 의 숫자 | SL-5 online 에서 재측정 |
| 3 | **phase / subtask 신호가 Panda 에 없다**(subtask gate inert) | SL-6: gripper-state strong rule 또는 새 probe, GT phase 는 oracle 비교 |
| 4 | **core 주입**: offline 은 profile 로 해결. online `serve_safe` 는 robot model · contact link · bound 주입 길이가 없다 | SL-5 에서 최소 core 변경(RB-Y1 동작은 기준선으로 확인) |
| 5 | true target 이 target-free layer 에도 남는 비율(5 mm 로 approach 0.636) 중 recall 부족 몫과 오선택 몫 | SL-4b 후 task 별 분해 |
| 6 | cabinet 의 main layer 5 mm occupied 0.718, transport 의 field 오차 증가 | 원인 분석 미실시. closed loop 에서 문제가 되면 다룸 |
| 7 | ~~dev 는 task 당 episode 하나.~~ call 이 episode 안에서 상관돼 신뢰구간을 말하지 않는다 | ~~SL4b 의 init state 1–4 로 확장~~ → SL4b 에서 50 episode 로 확장(신뢰구간은 아직 말하지 않는다) |
| 8 | Panda 에는 `constraint_sphere_options` · `finger_cover` 를 적용하지 않았다(RB-Y1 용 knob) | SL-5 이후 clearance 가 문제가 되면 재검토 |

### 되돌아올 지점 (SL4)

| 전환 신호 | 돌아와서 할 일 |
|---|---|
| SL-4b 에서 오선택의 원인이 attention cell 이 아니라 grounding(기하 · clustering)으로 나옴 | cell 교체로 해결되지 않는다. V11AB grounding 규칙 변경은 method variant 이므로 사용자 판정 |
| SL-4b 에서 채택 조건(≥ 10 %p, ≥ 6 / 10 task)을 넘는 cell 이 없음 | L8H2 유지, 오선택을 Panda 결과의 한계로 기록 |
| target ball 이 인접 장애물을 지우는 것이 closed loop 에서 확인됨 | V11AB `_target_ball` 반경 · 규칙이 Panda scene 스케일에 맞는지 확인(변경은 variant, 사용자 판정) |
| ESDF box 를 z 외에도 바꿔야 함 | 근거(측정)와 함께 사용자 판정 |
| 기준선이 어긋남 | 즉시 중단(core 가 바뀌지 않았어야 한다) |

---

## SL4b — ATS 용 attention cell dev calibration (2026-10-13)

> 2026-10-13 KST. 사전 등록 commit `41a5b28a`(2026-10-13 01:00 KST) · 구현 commit `714c8914`(09:23 KST) · reset history 수정 commit `8f0a07ec`(11:11 KST) ·
> 검증 commit `85e6d9d1`(13:57 KST) · 다음 STEP task commit `cf4649ce`(14:07 KST, SL4c). 직전 기록(SL4 log, SL5a task)은 `635a9762`. 사용자 확정 2026-10-13.
> 원본: [`handoff/SL4b.task.md`](handoff/SL4b.task.md) · [`handoff/SL4b.impl.md`](handoff/SL4b.impl.md)(마지막 절 "SL4b fix — reset history") ·
> [`handoff/SL4b.verify.json`](handoff/SL4b.verify.json). raw 출력은 `outputs/verify/SL4b/`.
> 이 절의 수치는 `SL4b.verify.json` 의 `numbers` 가 출처다. 예외 둘은 그 자리에 표시했다(impl.md 의 재현 표). **측정은 전부 dev 데이터**이고 test(init 5–49)는 쓰지 않았다.

### 이 STEP 이 답하는 질문

SL4 에서 V11AB attention cell(L8H2)로 만든 ATS 는 10 개 task 중 t4(plate 를 target 으로 고름), t5(같은 모양의 `bowl_2`), t9(true bowl 위 target 점 0)에서 진짜 target 을 놓쳤다.
LIBERO Spatial 은 같은 bowl 두 개를 위치 관계로 구분하는 과제라 이 오선택은 true target 을 장애물로 남기는 방향으로 작동할 수 있다. 질문은 둘이다.

1. 원인이 **attention cell** 인가, 그 뒤 **grounding**(attention → seed → cluster → 선택)인가?
2. cell 이 원인이라면 dev 에서만 고른 다른 cell 이 사전 등록한 기준을 넘는가?

### 사전 등록 (결과를 보기 전에 고정, 바꾸지 않았다)

`SL4b.task.md` 의 표를 문구 그대로 옮긴다. 이 표는 평가기 `benchmark/evaluators/ats_calibration.py` 의 `PREREGISTRATION_ROWS` 에 같은 문구로 들어 있고
test 가 task 파일과 문자 단위로 대조한다. 행 sha256 `77dfe79044da5df9f26c3a0a5e332586b99ca376a15a4d352b8b5256dfe32a55`, evaluator module sha256 `59c037c1ac351fe327604bba47abd45d830917fdf9da770456470395ca00643f`
(모든 SL4b replay run 의 manifest 와 평가기 출력에 기록).

| 항목 | 정의 (원문) |
|---|---|
| Dev 데이터 | LIBERO Spatial task 0–9 × init state 0–4 (50 episodes). init 0은 SL3 run 재사용, init 1–4는 새 rollout(pi0.5-only, 같은 protocol) |
| Phase | SL3/SL4와 같은 정의(MuJoCo finger–target contact, lift z0+10 mm). **평가 phase = approach + grasp** (운반 단계는 attach가 없는 offline 한계로 제외) |
| 1차 지표 (`target_correct`) | call별: ATS가 target을 냈고(has_target) **ATS target 점의 최근접 instance 다수결 = 진짜 target** 이고 centroid 오차 ≤ 30 mm 이면 1. 평가 phase call 전체에 대한 비율 |
| 2차 지표 | pooled precision(10 mm), recall median, centroid 오차 median/p95, task별 `target_correct` |
| 후보 cell | V11AB L8H2 + **선별 지표**(dev 50 episode attention에서 approach+grasp call의 head+wrist target mass 평균, un-mirror)의 상위 16개 cell. 진단 grid는 denoise 0, agg `last`(V11AB와 같은 집계) |
| 채택 조건 | 후보 cell이 L8H2보다 1차 지표가 **≥ 10 %p 높고**, task별로 **10개 중 ≥ 6개 task에서 L8H2 이상**이면 "Panda calibrated cell" 후보로 사용자에게 제안. 여러 개면 1차 지표 최고, 동률이면 낮은 layer. 조건을 만족하는 cell이 없으면 L8H2 유지 |
| 원인 분리 | L8H2의 오선택 call마다: attention argmax(head·wrist)가 진짜 target 위에 있었는지 → attention 원인 / grounding 원인 비율 |
| 참고(후보 아님) | GT target oracle seed(segmentation으로 target pixel만 attention=1) — ATS 기하 상한. privileged input으로 표시, 선택에 사용 안 함 |

표가 말하지 않는 해석(`OPERATIONAL_DEFINITIONS`)도 manifest 에 기록했다. 예: 분모는 target 이 없는 call 도 포함한 평가 phase call 전체이고, 다수결은 유일한 최대 label 만 인정(1 위 동률은 다수 없음 = `target_correct` 0),
채택 동률의 마지막 순서(낮은 head)는 등록 밖에서 결정적으로 완결하기 위해 정했다. 판정 문장은 평가기 출력(사실)으로만 기록한다.

### 데이터

| 항목 | 값 |
|---|---|
| dev episode | 50 (task 0–9 × init 0–4) |
| 평가 call (approach + grasp) | 607 |
| init 0 | SL3 dev rollout `20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0` + SL3 attention audit `20261012T202459_309e4c6da477_sl3-attention-audit-dev` (재사용) |
| init 1–4 | 새 rollout `20261013T092537_714c891415d9_sl4b-dev-pi05only-t0-9e1-4`(pi0.5-only, 같은 protocol; 09:25–09:47 KST, client 1270.96 s) + attention audit `20261013T094739_714c891415d9_sl4b-attention-audit-dev-e1-4`(09:47–09:49 KST) |
| scene GT · phase / segmentation | init 0 = `outputs/verify/SL4/raw/scene` · `outputs/verify/SL3/raw/gt`, init 1–4 = `outputs/verify/SL4b/raw/{scene,phase}` (아래 reset history 수정 후 두 번째 export) |
| test (init 5–49) | **사용하지 않았다** |

### 무엇을 만들었나

AG3S core(`AG3S/ag3s/**`, `AG3S/trajopt/**`)는 수정하지 않았다(SL4 와 같은 원칙).

**1. harness 옵션 넷** (`benchmark/runners/perception_replay.py`; 기본값은 SL4 경로 그대로)

| 옵션 | 하는 일 |
|---|---|
| `--attention-cell L,H` | attention audit 에 저장된 진단 grid(`[L,H,768]`, float16)에서 해당 cell 을 꺼내 쓴다. 기본 V11AB L8H2 는 SL4 가 쓴 float32 read-out 그대로이고 다른 cell 은 float16 `pooled_f16`. float16 underflow 로 block 합이 0 인 call 은 `attention_invalid` 로 표시 |
| `--grounding-only` | ESDF · constraint 단계를 건너뛰고 grounding 까지만 기록(빠른 선별용). 그 AG3S instance 의 `_build_esdf` 만 바꾸므로 core 파일은 그대로 |
| `--attention-source oracle` | GT target segmentation(image instant 로 다시 render 한 `segP`)으로 attention 을 만든 참고 실행. manifest 에 `privileged_attention` 과 "PRIVILEGED: …" 를 기록 |
| `--source-run` / `--attention-run` / `--scene-gt` 반복 | init 0 run 과 init 1–4 run 을 한 번에 받는다 |

**2. 평가기** (`benchmark/evaluators/ats_calibration.py`): 사전 등록 지표 · 후보 선별 `select` · 채택 조건 판정 · 원인 분리 · oracle 참고 `evaluate`. 채택 조건 문구와 숫자를 코드 상수로 두고 manifest 에 hash 와 함께 기록한다.
segmentation 은 rollout 이 `env.step` 반환 시점에 기록한 `segmentation_geom` 이 아니라 RGB · depth · attention 과 같은 순간(image instant)의 `segP` 를 쓴다(SL3 와 같은 입력). 새 `phase_export.py` 가 init 1–4 의 `segP` 를 만든다.

**3. test** — `tests/benchmark/test_panda_sl4b.py` 17 개(CPU)가 `--attention-cell 8,2` = 기존 경로, grounding-only = full 의 grounding, 평가기 지표 정의, 채택 경계(정확히 10 %p 통과 / 9 %p 실패 등)를 확인한다.

### 실행

| 단계 | 값 (시각은 KST) |
|---|---|
| dev rollout init 1–4 · audit | 09:25–09:47 · 09:47–09:49 (GPU lock 안, compute process 없음 확인) |
| 후보 선별 `select` | 50 episode 의 평가 call 에서 (call, camera) 쌍 1,214 개(head + wrist). 후보 17 cell(L8H2 + 상위 16) |
| grounding-only replay | 19 run(L8H2 · 후보 16 · oracle · L8H2-f16 대조) 모두 returncode 0, 11:36–12:16 KST, 각각 GPU lock 안. replay code 는 `8f0a07ec` |
| 평가 `evaluate` | 12:17–12:40 KST (CPU) |
| 채택 시 full replay (step 7) | **수행하지 않았다** — 평가기 `selected_by_rule` 이 null |

### 결과 — Figure

세 figure 는 verifier 가 만들었다(sidecar `.json` 동봉).

**실제 scene** — 행은 t4 · t5 · t9(L8H2 가 오선택한 call)와 대조 t7(L8H2 정답), 열은 head · left_wrist 별로 [L8H2(V11AB), L12H3(1차 지표 최고 후보), oracle(privileged 참고)] 이다.
흰 윤곽 = GT target(image instant segmentation), 주황 윤곽 = ATS 점 다수가 놓인 물체(GT target 이 아닐 때), 검은 점 = ATS target 점, 파란 = block 정규화한 cell attention, 빨강 사각형 = 그 camera 의 attention argmax patch.
oracle 열(빨강 fill = GT mask 가 attention)에서도 t4 는 ramekin, t5 는 stove, t9 는 `bowl_2` 위에 ATS 점이 놓인다(정확한 call 번호와 centroid 오차는 각 panel 제목).

![SL4b scene](figures/sl4b-scene.png)

**graph** — (a) cell 별 1차 지표(diamond, call 수 표기)와 task 별 점, (b) 선별 지표 대 1차 지표, (c) task × cell 의 `target_correct`(count / call), (d) L8H2 원인 분리, (e) task 별 L8H2 · L12H3 · oracle, (f) task 별 원인 분리.

![SL4b graph](figures/sl4b-graph.png)

**table** — 사전 등록 규칙, 후보 순위, 채택 조건 결과, rollout official success(pi0.5-only), fixture shift(아래 발견).

![SL4b table](figures/sl4b-table.png)

### 결과 — #1 후보 순위와 채택 조건

선별 지표 순위는 전체 144 cell 중의 순위다. L8H2(V11AB)는 2 위(선별 지표 0.2417), L12H3 이 1 위(0.2914)다. **1차 지표** `target_correct` = 평가 phase call 607 개 중 정답 call 수(L8H2 기준 400).
`gain` 은 L8H2 대비 %p, `tasks ≥ L8H2` 는 10 task 중 L8H2 이상인 task 수(채택 조건 ≥ 6).

| 선별 순위 | cell | 선별 지표 | correct / 607 | rate | gain (%p) | tasks ≥ L8H2 | gain ≥ 10 %p | tasks ≥ 6 |
|---:|---|---:|---:|---:|---:|---:|:---:|:---:|
| 2 | **L8H2 (V11AB, baseline)** | 0.2417 | **400** | 0.659 | — | — | — | — |
| 1 | L12H3 | 0.2914 | 409 | 0.674 | +1.48 | 6 | no | yes |
| 3 | L12H4 | 0.2104 | 373 | 0.614 | −4.45 | 5 | no | no |
| 4 | L5H5 | 0.2078 | 250 | 0.412 | −24.71 | 4 | no | no |
| 5 | L9H5 | 0.1820 | 400 | 0.659 | 0.00 | 7 | no | yes |
| 6 | L9H1 | 0.1742 | 303 | 0.499 | −15.98 | 3 | no | no |
| 7 | L8H3 | 0.1442 | 351 | 0.578 | −8.07 | 5 | no | no |
| 8 | L10H0 | 0.1433 | 356 | 0.586 | −7.25 | 4 | no | no |
| 9 | L7H3 | 0.1387 | 300 | 0.494 | −16.47 | 3 | no | no |
| 10 | L12H1 | 0.1370 | 386 | 0.636 | −2.31 | 5 | no | no |
| 11 | L12H2 | 0.1283 | 299 | 0.493 | −16.64 | 4 | no | no |
| 12 | L6H2 | 0.1260 | 301 | 0.496 | −16.31 | 3 | no | no |
| 13 | L3H1 | 0.1183 | 277 | 0.456 | −20.26 | 4 | no | no |
| 14 | L10H3 | 0.1145 | 315 | 0.519 | −14.00 | 4 | no | no |
| 15 | L5H4 | 0.1086 | 330 | 0.544 | −11.53 | 4 | no | no |
| 16 | L8H7 | 0.1082 | 172 | 0.283 | −37.56 | 3 | no | no |
| 17 | L15H0 | 0.1080 | 321 | 0.529 | −13.01 | 4 | no | no |

**채택 조건 결과** (평가기 출력 그대로): outcome = "no candidate satisfies the rule (rule: keep L8H2)", `selected_by_rule` = null, `satisfying` = [], `complete_dev_set_50` = true.
1 차 지표가 가장 높은 후보 L12H3 은 +9 call(+1.48 %p)로 +10 %p 기준에 못 미치고(task 조건은 만족), L9H5 는 L8H2 와 같다(400, task 조건 만족).
`tasks ≥ L8H2` 에는 모든 cell 이 0 인 t4 · t9 도 "이상"으로 세어진다는 점을 읽을 때 유의한다.

float16 대조(`L8H2-f16`, 후보 아님): L8H2 를 `pooled_f16` 로 읽어도 400 / 607, task 별 count 도 같고 `attention_invalid` call 은 1 개다(float32 read-out 은 0).

### 결과 — #2 L8H2 대 oracle, task 별

`n` = 평가 call 수. oracle 은 privileged 참고이며 선택에 쓰지 않았다. 열은 `target_correct` count 이다.

| task | n | L8H2 | L12H3 | oracle (privileged) |
|---|---:|---:|---:|---:|
| t0 | 66 | 56 | 55 | 56 |
| t1 | 65 | 44 | 48 | 48 |
| t2 | 61 | 60 | 57 | 61 |
| t3 | 61 | 54 | 53 | 57 |
| **t4** | 67 | **0** | 0 | **0** |
| t5 | 48 | 24 | 33 | 18 |
| t6 | 63 | 53 | 57 | 60 |
| t7 | 53 | 53 | 53 | 53 |
| t8 | 68 | 56 | 53 | 62 |
| **t9** | 55 | **0** | 0 | **0** |
| 합계 | 607 | 400 (0.659) | 409 (0.674) | 415 (0.684) |

L8H2 의 category(607 call): correct 400, centroid_far 27, no_target 30, wrong_instance 150. phase 별로는 approach 315 / 492(0.640), grasp 85 / 115(0.739).
pooled precision 0.705, recall median 0.688, centroid 오차 median 14.27 mm · p95 400.30 mm. oracle 의 category 는 correct 415, centroid_far 23, wrong_instance 169(no_target 0)이고 pooled precision 0.746 이다.

t4 · t5 · t9 에서 oracle ATS 가 고른 물체(nearest-instance 다수결, call 수):

| task | L8H2 의 오선택 다수 label | oracle 의 오선택 다수 label | oracle 의 ATS centroid 와 true target(`segP` back-projection)의 거리, median |
|---|---|---|---:|
| t4 | plate_1 42, ramekin 11 | ramekin 39, flat_stove_1 28 | 461.9 mm |
| t5 | `akita_black_bowl_2` 23 | flat_stove_1 30 (정답 18) | 382.4 mm |
| t9 | `akita_black_bowl_2` 44, plate_1 1 | `akita_black_bowl_2` 55 | 379.7 mm |

같은 거리가 정상 task 에서는 median 11.9–17.3 mm(t0 · t1 · t2 · t3 · t6 · t7 · t8)다. t1 의 p90 만 59.9 mm 이다.

### 결과 — #3 L8H2 오선택의 원인 분리

L8H2 의 `wrong_instance` call 150 개를 나눴다. 기준은 두 camera 의 raw attention grid 에서 joint argmax patch 가 true target 의 image-instant segmentation 위에 있는지다.

| | call | 비율 |
|---|---:|---:|
| 오선택 | 150 | |
| attention 원인 (argmax 가 target 밖) | 67 | 0.447 |
| grounding 원인 (argmax 가 target 위인데 ATS 가 다른 물체) | 83 | 0.553 |

camera 별: head argmax 가 target 위 53, wrist 82, 둘 중 하나라도 93, 둘 다 42. joint argmax 를 낸 camera 는 left_wrist 148, head 2. 다수 label 이 어느 물체도 아닌 call(`none`)은 2 개(모두 attention 원인)이고, 다른 instance 가 다수인 call 148 개 중 attention 65 · grounding 83 이다.

| task | 오선택 | attention 원인 | grounding 원인 | 다수 label |
|---|---:|---:|---:|---|
| t1 | 14 | 4 | 10 | ramekin 13, none 1 |
| t3 | 4 | 3 | 1 | plate 3, none 1 |
| **t4** | 53 | 25 | 28 | plate 42, ramekin 11 |
| **t5** | 23 | 4 | 19 | `bowl_2` 23 |
| t6 | 7 | 4 | 3 | `bowl_2` 7 |
| t8 | 4 | 1 | 3 | `bowl_2` 4 |
| **t9** | 45 | 26 | 19 | `bowl_2` 44, plate 1 |
| 합계 | 150 | 67 | 83 | |

### 결과 — #4 dev rollout official success (**pi0.5-only**, V11AB 결과가 아니다)

dev 50 episode 의 rollout 은 VLA(pi0.5)만 쓰고 AG3S safety layer 를 붙이지 않은 **pi0.5-only** 실행이다(init 0 = SL3 run, init 1–4 = SL4b run). 아래 `official_success` 는 그 기준 경로의 성공이며
**V11AB 의 성공이 아니다.** V11AB 와의 비교는 closed loop(SL-5 이후)에서 paired 로 한다.

| task | init 0 | init 1 | init 2 | init 3 | init 4 | task 합 |
|---|:---:|:---:|:---:|:---:|:---:|---:|
| t0 | success | success | success | success | success | 5 / 5 |
| t1 | success | success | success | success | success | 5 / 5 |
| t2 | success | success | success | success | success | 5 / 5 |
| t3 | success | success | success | success | success | 5 / 5 |
| t4 | success | success | success | success | success | 5 / 5 |
| t5 | success | success | success | success | success | 5 / 5 |
| t6 | success | success | success | **fail** | success | 4 / 5 |
| t7 | success | success | success | success | success | 5 / 5 |
| t8 | success | success | success | success | success | 5 / 5 |
| t9 | success | **fail** | success | success | success | 4 / 5 |
| init 합 | 10 / 10 | 9 / 10 | 10 / 10 | 9 / 10 | 10 / 10 | **48 / 50** |

### 결과 — #5 fixture reset 이력 발견과 수정

**평이한 설명.** LIBERO 는 `env.reset()` 을 부를 때마다 fixture(`wooden_cabinet_1`, `flat_stove_1`)의 위치를 global numpy RNG 로 **다시 뽑는다**(`bddl_base_domain.py:776`; `env.seed` = `np.random.seed`).
fixture 는 joint 가 없어서 기록된 `sim_state` 에 들어 있지 않다. rollout client 는 task 마다 env 를 하나 만들고 episode 마다 reset 하므로, 같은 task 의 k 번째 episode 는 그 task 의 k 번째 reset 에서 뽑힌 fixture 위치를 본다.
SL4b 의 첫 scene GT · phase / segmentation export 는 episode 마다 새 env 를 만들고 reset 을 **한 번만** 했다. 그래서 task 의 첫 episode(init 1–4 run 에서는 init 1)만 rollout 과 같은 scene 이었고, **init 2–4 는 fixture 가 달랐다.**
이 오류는 rollout 이 기록한 head segmentation 의 cabinet · stove 위치와 export 를 비교하다 드러났다.

| 측정 | 값 |
|---|---|
| 첫 export(대체됨)의 fixture 이동량: head 첫 call frame 에서 cabinet · stove centroid 의 rollout 기록 대비 최대 이동 | init 1 = 0.0 mm, init 2 = 57.7 mm, init 3 = 35.4 mm, init 4 = 23.1 mm |
| fixture entry(80 개) 중 1 mm 를 넘은 것 | 60 개 |
| 수정 후 같은 비교의 최대 이동 | 모든 init 에서 0.0 mm |
| 새 export 대 대체된 export, scene GT 40 episode | 10 episode 동일(init 1 전부), 30 episode 에서 fixture body 10 개만 이동: cabinet 계열 5.3–16.1 mm, stove 계열 1.9–15.3 mm |
| 수정 전후 L8H2 grounding-only 반복(50 episode) | **50 / 50 episode 에서 grounding 결과 동일**. 달라진 것은 init 2–4 30 episode 의 GT depth 잔차 두 key(`gt_depth_residual_head`, `gt_depth_residual_left_wrist`)뿐 |

즉 ATS 자체의 출력은 영향을 받지 않았고, 영향을 받은 것은 init 2–4 의 **GT 쪽 입력**(scene GT 의 fixture, `segP`)이다. 그래서 첫 export 를 쓴 산출물(선별 지표의 target coverage, oracle attention, GT 대조, 평가)은 모두 **superseded** 로 두고, 위 결과 표의 수치는 전부 수정 후 두 번째 export 로 다시 낸 것이다.
첫 export · 선별 · replay 는 `outputs/verify/SL4b/raw/superseded_*`, `select.superseded.json`, `replay_runs.superseded.tsv` 등에 보존했고 결과 해석에 쓰지 않는다. 후보 cell 17 개는 옛 선별과 새 선별이 같았다.
phase label 입력(`target_xpos`, `contact_grip_target`)은 40 / 40 episode 에서 같았다(impl.md).

**수정** (`8f0a07ec`; impl.md "SL4b fix — reset history"): export 와 action replay 가 rollout 의 **task 별 reset 이력을 그대로 재현**한다.

| 파일 | 하는 일 |
|---|---|
| `benchmark/runners/libero_pi05_client.py` | 동작을 바꾸지 않는 추출: `make_task_env`(env 생성 + seed), `start_episode`(reset → `set_init_state`). client 와 export 가 같은 두 함수를 부른다 |
| `benchmark/adapters/libero_panda/env_history.py` (새) | `episodes.jsonl` 의 순서가 manifest 의 `task_ids × episode_ids` 와 같음을 확인(어긋나면 **거부**)하고, task 마다 env 하나 + 앞선 episode 마다 `reset` + `set_init_state` 로 이력을 재현 |
| `scene_export.py` · `phase_export.py` | 위 이력으로 env 를 받고, 내보내는 call frame 마다 두 camera 의 RGB 재render 를 기록과 대조해 **하나라도 다르면 그 episode 는 예외로 멈추고 파일을 쓰지 않는다** |
| `benchmark/runners/libero_replay.py` · `launch_libero_replay.py` | replay 하지 않는 앞선 source episode 도 `start_episode` 로 reset 횟수를 맞춘다(`source_history_resets`) |
| `tests/benchmark/test_panda_sl4b_history.py` (새) | 10 test: 순서 규칙(거부 6 종), fake RNG env 와 실제 LIBERO t0 의 이력 일치, history 없는 fresh env 는 e3 불일치 |

**재현 증거** (`SL4b.impl.md` 의 재현 표 — `verify.json` 의 `numbers` 가 아니라 impl.md 가 출처다):

| 대상 | episode | call frame | head RGB 같음 | left_wrist RGB 같음 |
|---|---:|---:|---:|---:|
| `phase_export` init 1–4 | 40 | 873 | 873 / 873 | 873 / 873 |
| `scene_export` init 1–4 | 40 | 873 | 873 / 873 | 873 / 873 |
| `phase_export` · `scene_export` init 0 (SL3 run) | 10 | 214 | 214 / 214 | 214 / 214 |

init 0 의 새 scene export 는 `outputs/verify/SL4/raw/scene` 와 10 / 10 episode 에서 모든 배열이 bit 동일하다. e3 만 replay 한 경우 history 를 주면 rollout 의 e3 와 14 / 14 배열이 같고, history 없이(예전 동작) 돌리면 두 image 배열이 달랐다(impl.md).

**P2 로 넘어가는 결론.** paired 비교(V11AB 대 pi0.5-only 등)는 **task 별 episode 순서와 reset 횟수를 같게 유지해야** 같은 scene 에서 비교된다. episode 일부만 replay 하거나 순서를 바꾸면 fixture 가 달라진다(`libero_replay` 는 이제 source 순서의 부분열이 아니면 거부하고 앞선 reset 을 맞춘다).
기존 replay run(SL1 3 개, SL3 `replay-off`)은 source 가 task 당 episode 하나라 영향이 없다(impl.md).

### 결과 — #6 회귀

| 확인 | 결과 |
|---|---|
| core diff (`AG3S/ag3s/**` · `AG3S/trajopt/**`) | code tree 의 변경 파일 없음(verify.json `dirty_files_code_trees` 비어 있음) |
| 기준선: legacy 16D (15 frame) | violated start 14 / 15 · has_target 15 / 15 · frame0 clearance −29.031 mm — **일치** |
| 기준선: cuRobo 16D (15 frame, GPU lock) | violated start 10 / 15 · has_target 15 / 15 · frame0 clearance −9.172 mm — **일치** |
| `--attention-cell 8,2` grounding-only 대 SL4 default run | init 0 의 10 episode · 221 call · 32 key 모두 동일(attention grid 도 audit 의 `cell_grid_policy` 와 같음) |
| 대체된 GT 로 돌린 L8H2 run 대 SL4 default | 같은 10 episode · 221 call · 32 key 동일 |
| CPU regression (`run_cpu_regression.sh`) | exit 0 — A 73 passed / B 119 passed / C 117 passed, 2 skipped(flax import) |

### 알아둘 점 — replay 시간 측정은 기준이 아니다

grounding-only replay 19 개를 돌리는 동안 GPU lock 을 job 마다 풀었기 때문에 **다른 세션의 GPU 작업이 사이에 끼었다**(job 별 lock 대기 구간이 verify.json 에 기록됨).
그래서 이 run 들의 replay latency(`t_process_s`, `t_record_s`, `stage_ms`)는 시간 기준선이 아니고 인용하지 않는다. grounding-only 는 ESDF · constraint 단계를 건너뛰므로 어차피 SL4 의 stage 시간과 같은 값이 아니다.
이 verification 의 figure · verify.json 작업은 GPU 를 쓰지 않았다.

### 이 STEP 에서 알게 된 것 (평이한 설명)

**(a) cell 을 바꿔서는 해결되지 않는다.** 사전 등록 규칙에 따라 V11AB 의 L8H2 를 유지한다. 최고 후보 L12H3 도 607 call 중 +9 call(+1.48 %p)이어서 기준 +10 %p 의 15 % 수준이다.
선별 지표(target 위 attention 질량)와 1 차 지표(ATS 가 진짜 target 을 고르는가)는 같이 오르지 않는다(graph (b): 선별 지표 3 위 L12H4 의 1 차 지표가 L8H2 보다 낮다). attention 질량이 target 위에 있다는 것이 ATS 정답을 보장하지 않는다.

**(b) oracle 도 t4 · t9 에서 0, t5 에서는 일부만 맞는다.** oracle 은 attention 이 곧 GT target mask 이므로 attention 이 진짜 bowl 을 정확히 가리킨다. 그런데 t4(bowl 이 서랍 안), t9(bowl 이 cabinet 위)에서는 `target_correct` 가 0 이고
t5(bowl 이 ramekin 위)는 18 / 48 로 L8H2(24 / 48)보다도 낮다. 이 task 들에서 ATS target 점이 놓이는 물체는 ramekin · stove · `bowl_2` 로, cell 과 무관하게 진짜 bowl 이 아니다.
따라서 이 실패는 **attention cell 이 아니라 attention 이후의 grounding · cluster 선택 단계에 있다.** L8H2 오선택 150 call 중 83 call(0.553)이 이미 attention argmax 가 target 위에 있는데도 ATS 가 다른 물체를 골랐다는 원인 분리와도 방향이 같다(t5 에서 23 중 19).
반대로 attention 원인 67 call 은 t4 25 · t9 26 이지만, 두 task 는 oracle 도 0 이므로 cell 을 개선해도 풀리지 않는다.

**(c) 왜 그런지는 아직 모른다.** 작업 가설 둘이 있다. **둘 다 검증되지 않았다.**

| 가설 | 내용 | 상태 |
|---|---|---|
| H1 병합 | 진짜 bowl 이 받치고 있는 물체(서랍 · cabinet · ramekin)와 하나의 cluster 로 합쳐져 선택에서 밀린다 | **미검증** |
| H2 개도 한계 | admissibility 가 Panda 의 8 cm 개도 한계를 쓰고, 이것이 bowl 크기를 가진 cluster 를 탈락시킨다 | **미검증** |

이 중 어느 것도 이번 측정으로 확인하지 않았다. 진단은 SL-4c 가 `85e6d9d1` 에 고정한 code 로 한다.
oracle run 의 seed 와 ATS 선택 cluster 를 비교한 대표 call 의 숫자는 아래와 같고, 해석하지 않는다(`oracle_seed_vs_cluster.json`).

| call | oracle target pixel (head, wrist) | seed 점 | cluster 수 | ATS target 점 | ATS target centroid (base, m) |
|---|---|---:|---:|---:|---|
| t4 init 0 call 4 | 1244, 8557 | 937 | 4 | 123 | (0.458, 0.192, 0.020) |
| t5 init 0 call 2 | 708, 3566 | 708 | 3 | 63 | (0.248, −0.123, 0.025) |
| t9 init 0 call 0 | 1179, 0 | 761 | 4 | 261 | (0.410, −0.136, 0.045) |
| t7 init 0 call 3 (대조, 정답) | 548, 3494 | 700 | 4 | 236 | (0.399, −0.130, 0.048) |

**(d) oracle 은 call 단위 상한이 아니다.** oracle 합계 415 는 L8H2 보다 15 call 많지만 t5 는 오히려 6 call 적다(18 대 24). oracle 은 "ATS 기하가 attention 을 얼마나 그대로 따르는가"의 참고이며 방법의 상한으로 읽지 않는다.

**(e) 오류를 하나 찾고 고쳤다.** 위 발견(#5)의 fixture 문제다. 이 STEP 의 최종 숫자는 수정 후 export 로 낸 것이다. 이전 export 로 낸 중간 산출물은 superseded 로 보존했다.

### 판정 (2026-10-13, 사용자)

| # | 판정 |
|---|---|
| 1 | **SL4b 확정.** V11AB L8H2 를 사전 등록 규칙에 따라 유지한다(최고 후보 +1.5 %p < +10 %p 기준). dev rollout 48 / 50 은 **pi0.5-only** 결과로 따로 기록한다 |
| 2 | **SL4c 는 원인 진단까지만 한다.** 측정: 단계별 target 점 생존, cluster 의 물체 구성과 크기, **실제로 적용된** 개도 한계, 탈락 사유. 병합과 8 cm 한계는 **가설로 둔다.** code · parameter 변경은 진단 후 별도 판정한다. 물리적 robot 사양을 잘못 옮긴 오류는 adapter 보정 후보이고, 성능을 위해 clustering · 선택 기준을 바꾸는 것은 **별도의 method 변경**이다 |
| 3 | **SL5a 를 정의대로 병행 시작한다**: Panda 주입과 offline EE-delta ↔ joint 검증. **RB-Y1 golden 동일성**과 **무개입 bit 동일성**을 필수 조건으로 한다. SL4c 는 `85e6d9d1` 로 고정해(frozen worktree `.runtime/worktrees/sl4c-85e6d9d1`) SL5a 의 core 변경이 섞이지 않게 한다. live SL5b 는 SL5a 후에 판정한다 |

### 측정하지 않은 것

- 채택 후보의 full replay(ESDF 포함, task step 7): 채택 후보가 없어서 수행하지 않았다
- t4 · t5 · t9 에서 oracle 의 ATS target cluster 가 seed 가 놓인 cluster 가 아닌 이유(위 대표 call 숫자만 있다) → SL4c
- test split(init 5–49): 설계상 사용하지 않았다
- 시간: replay latency 는 GPU 를 번갈아 써서 기준이 아니다
- diagnostic grid 의 다른 denoise step · aggregation(grid 는 denoise 0 · `last` 로 고정)
- 1 차 지표의 신뢰구간: call 이 episode 안에서 상관되므로 말하지 않는다

### 열린 항목

| # | 내용 | 영향 / 다음 |
|---|---|---|
| 1 | t4 · t5 · t9 에서 ATS 가 true bowl 을 고르지 못하는 단계와 기준(H1 병합, H2 8 cm 개도 한계 포함) | SL-4c 진단. 결과에 따라 adapter 보정(사양 매핑 오류) 또는 별도 method 변경 판정 |
| 2 | 오선택이 closed loop 에서 true target 을 장애물로 남기는 비용 | SL-5 이후 online 측정 |
| 3 | paired 비교는 task 별 episode 순서와 reset 횟수를 같게 유지해야 한다 | P2 설계에 반영 |
| 4 | dev 는 50 episode 이며 신뢰구간을 말하지 않는다 | P2 이후 |

### 되돌아올 지점 (SL4b)

| 전환 신호 | 돌아와서 할 일 |
|---|---|
| SL-4c 진단이 Panda 사양(개도 한계 등)을 잘못 옮긴 것으로 나옴 | adapter 보정 후보. 보정 후 SL4b 지표를 같은 평가기로 다시 잰다 |
| SL-4c 진단이 clustering · 선택 기준이 원인으로 나옴 | 그 기준 변경은 V11AB 의 method 변경이므로 사용자 판정(variant 로 분리, test 전 고정) |
| test(init 5–49)를 열기 전에 cell 을 바꾸려는 요청 | 사전 등록 규칙(L8H2 유지)을 먼저 확인하고, 바꾸려면 새 사전 등록이 필요하다 |
| 동일 scene 으로 paired 비교한 결과가 episode 순서에 민감함 | reset history 재현 규칙(위 #5)을 먼저 확인 |

---

## SL5b-smoke — t0 init0 paired smoke 재확인, conversion A / B (2026-10-14)

**기록 시각** 2026-10-14 14:45 KST. **수치 출처** [`handoff/SL5b-smoke.verify.json`](handoff/SL5b-smoke.verify.json).
**그림** [`figures/sl5b-smoke4-t0e0-ab.png`](figures/sl5b-smoke4-t0e0-ab.png) (+ `.json` sidecar).
**code** run 은 `945b608c`, 호출별 grasp 기록과 그림은 `fa658e06`.

### 무엇을 했나

사용자 판정(2026-10-14) 두 가지를 반영하고 t0 init0 를 다시 돌렸다.

1. **gripper `empty_below` Panda 보정.** Panda 의 빈손 닫힘(normalised opening 0.0125)과 물체를 쥐고 들어올린
   순간의 opening(최소 0.0381)을 측정해 그 중간점 **0.025325** 를 Panda 값으로 정했다. RB-Y1 기본값 0.10 은
   Panda 의 bowl rim 파지(약 0.06)를 빈손으로 판정했다. 다른 `LatchConfig` 값은 V11AB 기본 그대로다.
   근거 파일 `robots/panda/description/gripper_calibration.json`.
2. **역변환을 명시 실행 모드로 분리.** A = `lag_inverse` (OSC 1차 응답 모델의 역, `Dx = (Dy − αDy_prev)/β`),
   B = `command_offset` (`Dx = Dy`). 둘 다 refined == reference 인 행은 원래 action bit 를 그대로 보낸다.
   launcher 는 `--conversion {A,B}` 를 필수로 받고 run id · manifest 에 모드를 남긴다.
   서버는 응답에 response model 이 예측한 EE 경로(`panda/predicted_ee_base`)와 실현도 기록을 싣는다.

### 결과 — official success

| 조건 | 결과 | steps | HOLD call | attach |
|---|---|---|---|---|
| VLA-only | **success** | 104 | — | — |
| V11AB-A | max_steps (실패) | 220 | 25 / 44 | 없음 |
| V11AB-B | max_steps (실패) | 220 | 3 / 44 | obs 145 (call 27) 에 attach, 그 뒤 VLA 가 gripper 를 열어 detach |

B 는 두 번(smoke3 · smoke4) 돌려 모든 점검 수치가 같았다. A 의 결과는 `empty_below` 0.10 으로 돌린 이전 smoke2 와
점검 수치가 같았다 — 보정은 A 에서 결과를 바꾸지 못했다(아래 원인).

### 결과 — 역변환 점검 (사용자 점검 항목)

| 항목 | A | B |
|---|---|---|
| 무개입 행 bit 동일 | 58 / 58 | 82 / 82 |
| 보정된 call 수 | 18 | 40 |
| 실현 비율 (response model 예측 / refined 변화, call median) | 1.00 | 0.15 |
| 예측 − refined 경로 (mm, p50 / max) | 1.7 / 119 | 14.8 / 143 |
| 예측 − 실제 EE (mm, 보정 call p50 / max) | 7.6 / 30.2 | 6.4 / 17.2 |
| 실행 arm action 변화 \|Δa\| (p50 / max) | 0.63 / 1.74 | 0.23 / 1.59 |
| saturated 행 · \|a\|≥1 항목 | 57 · 72 | 25 · 14 |

무개입 call 의 예측 − 실제 오차는 두 모드 모두 3.2 mm 다(response model 자체의 오차).

- **A** 는 refined 경로를 거의 그대로 실현하지만 action 을 크게 바꾸고 자주 포화한다.
- **B** 는 action 변화가 작지만 V11AB 가 요구한 경로 변화의 약 15 % 만 실현한다. B 에서 보정 call 이 40 개로 늘어난
  것은, 서버에 보내는 `exec_feedback` 가 "refined 경로를 실행했다"고 알리는데 실제로는 그 일부만 실행되어
  continuity 기준이 실제 경로와 어긋나기 때문일 가능성이 있다(**가설, 측정하지 않았다**).
- 두 모드 모두 **장애물 회피 장면에서의 실현 정도는 아직 측정하지 않았다** (LIBERO t0 에는 추가 장애물이 없다).

### 결과 — 실패 원인: latch `reach` 검사와 occluded target

호출별 grasp 기록(이번에 추가한 로그)으로 latch 의 attach 조건을 call 마다 확인했다.

| 구간 (B) | 상태 | 실패한 검사 |
|---|---|---|
| call 8 ~ 10 | 첫 닫힘 시작 | closed · dropped · settled (손가락이 아직 움직임) |
| call 11 ~ 14 | VLA 가 스스로 다시 열었다 (VLA-only 도 같은 동작) | settled · blocked · reach |
| call 15 ~ 18 | 두 번째 닫힘 (opening 약 0.06) | settled |
| call 19 ~ 26 | 들어올림, bowl **occluded** (geometry frame 17 에 고정) | **reach 만** (127.9 ~ 184.2 mm > 120 mm) |
| call 27 | 손이 내려와 reach 111.7 mm | 전부 통과 → attach |

- `reach` 는 parent link(`panda_leftfinger`) 원점과 조작 대상 centroid 의 거리다. 들어올린 뒤 bowl 은 카메라에서
  보이지 않아(`occluded`) centroid 가 마지막 관측(바닥에 있던 위치)에 고정되고, 손이 올라갈수록 거리가 커진다.
- attach 가 안 된 동안 쥔 bowl 은 장애물로 남아 `panda_hand ↔ esdf` 충돌 HOLD 가 났다.
- **A**: reach 가 178 mm 에 고정된 채 HOLD 가 끝까지 이어졌다(교착).
- **B**: 늦게 attach 했지만, 그 사이 경로가 VLA-only 와 달라져 VLA 가 plate 밖에서 gripper 를 열었다.
- 첫 닫힘의 `blocked` 실패는 VLA 가 열림 명령을 보낸 구간에서만 났다. 이번 실패의 원인이 아니다.

### 판정이 필요한 것

- `reach` 실패는 Panda 의 물리 사양 대응(adapter)이 아니라 **occluded target 의 마지막 관측을 유지하는 V11AB 동작**에서
  온다. 한도를 키우면 gate 를 바꾸는 것이 된다.
- B 의 `exec_feedback` 를 실제 실행에 맞추는 것은 adapter 범위이다. 다만 아직 판정 전이다.

### 측정하지 않은 것

- 장애물 장면에서 A · B 의 회피 실현도.
- task 1~9 와 다른 init.
- B feedback 불일치 가설.

### 판정 (2026-10-14, 사용자)

- **reach 실패 → development branch 로 전달.** Panda 의 거리 기준 문제가 아니라, 가려진 물체의 오래된 위치를 쓰는
  문제다. reach 120 mm 를 유지한다. benchmark branch 는 latch 를 바꾸지 않는다. 전달 문서
  [`handoff/SL5b-reach-to-development.md`](handoff/SL5b-reach-to-development.md).
- **B feedback → 수정 후 재측정.** 예측 경로를 `applied_arm` 에 넣는 안은 승인하지 않았다. `applied_arm` 은
  실제로 적용한 팔 목표 명령이다. 실제 적용 OSC 명령 · 예측 경로 · 측정 joint 경로를 구분하고, 피드백은 실제
  실행된 step 기준으로 확정한다.
  - 통과 기준: 피드백 의미의 일치와 무개입 동일성.
  - 관찰 지표: 보정 횟수와 action 변화.
- 진행 순서: 피드백 정합성 수정 → t0 B 재측정 → LIBERO 10 개 paired 로 한계 기록 → SafeLIBERO 사례 탐색.
  reach 문제를 해결할 때까지 benchmark 진행을 멈추지 않는다.

---

## SL5b-feedback — exec_feedback 를 실제 실행 기준으로 확정, t0 재측정 (2026-10-14)

**기록 시각** 2026-10-14 16:50 KST. **수치 출처** [`handoff/SL5b-feedback.verify.json`](handoff/SL5b-feedback.verify.json).
**그림** [`figures/sl5b-smoke5-t0e0-ab-feedback.png`](figures/sl5b-smoke5-t0e0-ab-feedback.png) (+ `.json`).
**code** `a06b2ac2`.

### 무엇을 고쳤나 (adapter 범위, 알고리즘 변경 없음)

V11AB 는 `exec_feedback` 를 세 곳에서 쓴다.

1. continuity 정렬: `executed_chunk` · `n_exec` 를 쓴다.
2. `w_deflection_rate` 항: `applied_arm[-1]` − 이전 계획의 reference.
3. grasp latch: `measured_gripper` · `applied_gripper` 를 쓴다.

기존 client 의 문제는 둘이었다.

- 피드백을 호출 시점에 미리 만들었다.
- `measured_gripper` 에 **한 chunk 전 촬영값**을 넣었다. wire 계약은 "이번 요청의 촬영 시점" 이다.

| 항목 | 수정 전 | 수정 후 |
|---|---|---|
| 확정 시점 | 호출 때 미리 | 다음 요청 때, 실제 진행한 env step 수로 (`n_steps`) |
| `measured_gripper` | 직전 호출의 촬영값 | 이번 요청의 촬영값 |
| `applied_arm` | refined joint 행 (B 에서는 실제 명령과 다름) | 실제 적용한 OSC 명령(HOLD 포함)을, 그 chunk 를 보낸 요청과 같은 forward map · 같은 시작 상태로 옮긴 joint 명령 행 (`action.applied_command_rows`) |
| 근거가 없을 때 | — | `available=False` 로 철회 |

`applied_arm` 은 명령 이력이고, 예측 경로나 측정 경로가 아니다. 무개입 행은 reference 행과 bit 동일하다(unit test).

### 결과 — 통과 기준

| 항목 | B | A |
|---|---|---|
| `n_steps` = 실제 진행 step | 22 / 22 | 43 / 43 |
| `measured_gripper` = 이번 촬영값 | 22 / 22 | 43 / 43 |
| 무개입 chunk 의 `applied_arm` = reference | 1 / 1 | 1 / 1 |
| 철회된 피드백 | 0 | 0 |
| 무개입 행 action bit 동일 | 100 / 100 | 58 / 58 |

두 모드 모두 통과했다. 다만 V11AB 가 거의 모든 chunk 를 보정해서(B 21 / 23, A 31 / 44) 무개입 chunk 표본은
각 1 개뿐이다.

### 결과 — official success (같은 init, 같은 policy_seed)

| 조건 | 결과 | steps | HOLD | attach |
|---|---|---|---|---|
| VLA-only (smoke3) | success | 104 | — | — |
| V11AB-B | **success** | 114 | 0 / 23 | obs 100 (call 18, reach 103.5 mm) |
| V11AB-A | max_steps (실패) | 220 | 12 / 44 | 없음 |

- **B**: 현재 촬영값으로 `settled` 가 한 호출 먼저 통과했다. 그래서 bowl 이 occluded 되는 바로 그 호출에서 attach 했다
  (수정 전 smoke4 의 call 18: latch 가 본 값 0.0724, 실제 그 순간 0.063).
  attach 뒤 reach 는 133.9 ~ 149.8 mm 였지만 held 상태는 유지되었다. 단, **attach 여부가 들어올리기 전에 `settled`
  가 통과하느냐에 달려 있다는 의존성은 그대로**다.
- **A**: 경로가 달라져 파지 개도가 0.043 ~ 0.059 였다. 일부 호출에서 `blocked`(0.05)도 실패했다.
  bowl 이 obs 80 부터 occluded 되어 reach 가 357.1 mm 까지 커졌고 attach 하지 못했다.
  A 는 saturated 104 행 · \|a\|≥1 217 항목으로, action 을 크게 바꾼다.

### 관찰 지표 (통과 기준 아님)

| 항목 | B | A |
|---|---|---|
| 보정 call | 21 | 31 |
| 실현 비율 p50 | 0.13 | 0.84 |
| 실행 arm \|Δa\| p50 / max | 0.01 / 0.94 | 1.63 / 1.93 |
| saturated 행 | 4 | 104 |

### 측정하지 않은 것

- 장애물 장면에서 B 의 회피 실현도. SafeLIBERO 에서 측정한다.
- t0 외 task · init.

---

## SL5b-libero10 — LIBERO task 0~9 × init 0 paired (2026-10-14)

**기록 시각** 2026-10-14 17:55 KST. **수치 출처** [`handoff/SL5b-libero10.verify.json`](handoff/SL5b-libero10.verify.json).
**그림** [`figures/sl5b-libero10-init0.png`](figures/sl5b-libero10-init0.png) (+ `.json`). **code** `1df077b8` (세 run 모두).
**split** dev (LIBERO init 0, plan §6.0).

목적은 사용자 판정대로 **통합 오류와 뚜렷한 저하의 확인**이다. 10 episode 로 비저하를 증명하지 않는다.

### 진행 중 생긴 문제 (기록)

첫 시도(run-id `sl5b-libero10-init0`)에서 문제가 있었다.

- VLA-only 는 `5b1e11ce` 에서 돌았다.
- V11AB-B 는 client 가 시작되기 전에 내가 SafeLIBERO 연결을 commit(`1df077b8`)했다. client 는 "다른 commit 의 run" 이라며
  거부했고 episode 가 없다.
- 그래서 VLA-only 와 V11AB-B 를 `1df077b8` 에서 다시 돌렸다(`-r2`).
- 첫 VLA-only 는 r2 VLA-only 와 task 별 성공이 10 / 10 같다. 하지만 짝이 없으므로 비교에 쓰지 않는다.
- 이후 run 진행 중에는 runner 코드를 고치지 않는다.

### 결과 — official success

| 조건 | 성공 | 성공 task |
|---|---|---|
| VLA-only | 9 / 10 | t5 제외 전부 |
| V11AB-B (paired) | 3 / 10 | t0 · t1 · t7 |
| V11AB-A (별도 run, 같은 commit) | 2 / 10 | t3 · t5 |

V11AB-B 기준 "VLA-only 실패 · V11AB 성공" 후보는 0 건이다.

V11AB-A 는 t5 에서 VLA-only 가 실패한 episode 를 성공했다. 다만 이 episode 에서 latch 는 한 번도 파지 상태가 되지
않았다(`searching`). HOLD 는 1 회이고 action 변화가 크다(saturated 50 행). 회피에 의한 성공으로 볼 근거가 없어
관찰로만 남긴다.

### 결과 — 통합 점검 (통과 기준)

| 항목 | B | A |
|---|---|---|
| 피드백 `n_steps` = 실제 진행 step | 372 / 372 | 383 / 383 |
| `measured_gripper` = 이번 촬영값 | 372 / 372 | 383 / 383 |
| 무개입 chunk 의 `applied_arm` = reference | 12 / 12 | 10 / 10 |
| 철회된 피드백 | 0 | 0 |
| 무개입 행 action bit 동일 | 940 / 940 | 902 / 902 |
| TrajOpt `time_budget_hit` | 0 | 0 |
| client 실행 오류 | 0 | 0 |

통합 오류는 없다. node load average 는 27 ~ 54 (16 core) 였지만 time budget 에 걸린 call 은 없었다.

### 결과 — 저하 원인 (V11AB-B)

| 구분 | episode | latch | 주된 HOLD |
|---|---|---|---|
| 성공 | t0 · t1 · t7 | 전부 attach | 없음 또는 1 회 |
| 실패 | t2 · t3 · t5 · t6 · t8 · t9 | attach 없음. 닫힘 중 `reach` 실패 26 ~ 35 회 | `panda_hand ↔ esdf` collision (쥔 bowl 이 장애물로 남음) |
| 실패 | t4 | attach 없음 (`reach` 20 · `settled` 18 · `blocked` 10) | `uncertified` 8 회 (target 을 잃음, SL4 에서 본 t4 grounding 문제) |

실패 7 건 중 6 건은 t0 smoke 와 같은 기전이다. 들어올린 bowl 이 occluded 되고, latch 의 `reach` 가 마지막 관측
centroid 기준으로 120 mm 를 넘는다. 그래서 attach 하지 못하고, 쥔 bowl 이 장애물로 남아 HOLD 가 난다.
이 문제는 development 로 전달했다([`handoff/SL5b-reach-to-development.md`](handoff/SL5b-reach-to-development.md)).
benchmark 에서는 latch 를 바꾸지 않는다.

### 측정하지 않은 것

- init 1~4 와 test split.
- 장애물 장면에서 B 의 회피 실현도 (SafeLIBERO 에서 측정).

### 판정 (2026-10-14, 사용자) — SafeLIBERO Spatial 시작 조건

| 항목 | 판정 |
|---|---|
| dev / test split | task 0~3 × Level I/II 마다 episode 0~4 = dev (40), 5~49 = test (360). plan §6.0 에 기록 |
| step protocol | 공식 `main_aegis.py`: wait 20, wait 포함 t < 300 (long 550) |
| 사례 탐색 조건 | VLA-only + V11AB-B paired. A 는 돌리지 않는다 |
| 성공 기준 | safe_success = official success 이고 obstacle 충돌 없음 (README 충돌 규칙: wait 직후 작업공간 안 obstacle 의 sum\|Δpos\| > 1 mm) |

---

## 버전 이름 규칙 (사용자 지시 2026-10-14)

2026-10-14 부터 이 branch 에서도 adapter 와 알고리즘을 고친다. 결과를 섞지 않도록 버전을 구분한다.

| 이름 | 뜻 | 기준 commit |
|---|---|---|
| **V11AB** | paired v2 V11AB 알고리즘을 바꾸지 않은 기준선 + Panda adapter (변환 모드 A / B 를 따로 표기) | SafeLIBERO `61f44671`, LIBERO `1df077b8` |
| **V11AB-pb1, pb2, …** | 이 branch 에서 고친 개선 버전. 번호마다 변경 내용과 commit 을 아래 개선 기록에 남긴다 | 각 개선 기록 |

기준선 결과는 비교 기준으로 보존한다. 개선 결과는 버전 이름과 commit 을 함께 쓴다.

**사용자 지시 2026-10-22:** "pb6 이렇게 표현하지 말고 실제 단계명을 작성하라. 내가 알아들을 수 있도록." 2026-10-22 이후 기록과 보고에서는 `pb` 번호 대신 version 에서 바꾼 내용(예: 파지 인식 수정판, 파지 인식 + closing 면제판, grasp zone 허용판, 역변환 C)으로 부르고, 내부 key 는 처음 한 번만 괄호나 대응표로 적는다. 아래 SL5i 절이 처음 적용한 곳이다.

---

## SL5c — SafeLIBERO Spatial dev 연결과 V11AB 기준선 (2026-10-14)

**기록 시각** 2026-10-14 22:30 KST. **수치 출처** [`handoff/SL5c-safelibero-dev.verify.json`](handoff/SL5c-safelibero-dev.verify.json).
**그림** [`figures/sl5c-safelibero-spatial-LI-dev.png`](figures/sl5c-safelibero-spatial-LI-dev.png) (+ `.json`).
**code** `61f44671` (V11AB 기준선, 변환 B). **split** dev (task 0~3 × episode 0~4, level 별).

### 환경 연결

- SafeLIBERO (`benchmark/SafeLIBERO`, vlsa-aegis 의 LIBERO fork) 는 package 이름이 `libero` 로 같다.
  client 가 `libero` 를 import 하기 전에 fork 를 `sys.path` 앞에 두고 전용 config(`.runtime/safelibero`)로 바꾼다.
  `libero.libero` 가 fork 에서 왔는지 확인한다(`1df077b8`).
- step protocol 은 공식 `main_aegis.py` 다(wait 20, wait 포함 t < 300). control 은 20 Hz, OSC_POSE 로 LIBERO 와 같다.
- 충돌 판정은 README 규칙이다. wait 직후 작업공간 안의 obstacle 이 그 위치에서 sum\|Δpos\| > 1 mm 움직이면 충돌이다.
  safe success = official success 이고 충돌 없음.

### 충돌 판정기 검증

robot 을 wait 뒤 100 step 동안 움직이지 않았을 때 결과다. dev 40 장면(Level I · II) 에서 충돌 판정은 0 건,
최대 obstacle 변위는 0.9 µm 였다. 판정기가 장면 자체의 움직임을 충돌로 잡지 않는다.

### 결과 — Level I (paired, 같은 init · policy_seed)

| 조건 | official success | 충돌 | safe success |
|---|---|---|---|
| VLA-only | 11 / 20 | 18 / 20 | 2 / 20 |
| V11AB-B | 3 / 20 | 18 / 20 | 0 / 20 |

"VLA-only 실패 → V11AB safe success" 사례는 0 건이다.

### 결과 — Level II

VLA-only 는 official success 11 / 20, 충돌 18 / 20, safe success 2 / 20 이다.
V11AB-B 는 run 을 할당했지만 실행하지 못했다. 다른 checkout 의 GPU 작업을 기다리던 중 session 이 끝났다.
`NOT_RUN.json` 으로 남기고, 사용자 판정으로 보류한다.

### 원인 — 계획한 회피가 실행되지 않는다 (B)

V11AB-B 충돌 18 건을 분석했다. 각 episode 에서 reference 가 obstacle 에 닿는 마지막 call 의 행들을 시뮬레이터에
놓고 robot ↔ obstacle 최소 거리를 쟀다.

| 분류 | 건수 |
|---|---|
| refined 계획은 obstacle 을 피함, 실행은 관통 | 14 |
| refined 계획도 관통 | 3 |
| 둘 다 닿지 않음 (기록된 순간 사이에서 접촉) | 1 |

- 예: t3e0 은 reference −69.9 mm, refined +19.1 mm, 실행 −69.8 mm 다.
- 충돌한 부위는 대부분 gripper link(`gripper0_leftfinger` / `right_gripper`)다. V11AB 가 모델링하는 link 다.
- 따라서 V11AB 는 obstacle 을 보고 피하는 계획을 만들었다. 문제는 변환 B 가 그 변화의 약 13 % 만 실행으로 옮겨서
  실제 경로가 reference 와 거의 같았다는 점이다.
- 이 결과로 변환 B 는 사용자 점검 항목 "장애물 장면에서 회피 보정이 충분히 실현되는가" 를 통과하지 못한다.

### 다음 (사용자 지시 2026-10-14)

1. Level I 에서 V11AB-A 를 같은 20 episode 로 돌리고, 기존 VLA-only 와 조건 일치를 검사해 사후 pairing 한다.
2. 실제 비교에서 확인된 영향 순서로 원인을 고친다. 후보는 셋이다.
   - B 의 실현 부족.
   - occluded target 의 오래된 centroid 와 latch `reach`.
   - 지지면 · cluster 처리로 진짜 target 이 탈락하는 문제.
3. 개선 버전은 V11AB-pb* 로 구분한다.

### SL5c-A — Level I, V11AB 변환 A (2026-10-14)

**기록 시각** 2026-10-15 00:45 KST. **수치 출처** [`handoff/SL5c-safelibero-dev-A.verify.json`](handoff/SL5c-safelibero-dev-A.verify.json).
**그림** [`figures/sl5c-safelibero-spatial-LI-dev-A.png`](figures/sl5c-safelibero-spatial-LI-dev-A.png).
**code** `38b82ae8` (알고리즘 · adapter code 는 `61f44671` 과 같다).

사용자 지시대로 Level I 같은 20 episode 를 V11AB-A 로만 돌렸다. 기존 VLA-only run 과는 사후 pairing 했다
(`outputs/verify/SL5c/scripts/posthoc_pair.py`). pairing 전에 다음 조건이 모두 같은지 검사했다.

- protocol · runtime schedule · task / episode 목록 · seed · level · split.
- checkpoint · episode 별 policy_seed.
- clean tree, code 차이 0.

| 조건 | official success | 충돌 | safe success |
|---|---|---|---|
| VLA-only | 11 / 20 | 18 / 20 | 2 / 20 |
| V11AB-B | 3 / 20 | 18 / 20 | 0 / 20 |
| **V11AB-A** | 3 / 20 | 14 / 20 | **3 / 20** |

**처음으로 "VLA-only 실패 → V11AB safe success" 사례 2 건이 나왔다.** 증거 묶음은 `AG3S/ag3s/docs/cases/` 에 있다.

| 사례 | VLA-only | V11AB-A | 구분 |
|---|---|---|---|
| [`sl5c-LI-t0e0-V11AB-A`](cases/sl5c-LI-t0e0-V11AB-A/keyframes.png) | task 실패, moka pot 충돌 (t 35) | task 성공, 충돌 없음 | task 실패 → 성공, 안전 실패 → safe success |
| [`sl5c-LI-t2e4-V11AB-A`](cases/sl5c-LI-t2e4-V11AB-A/keyframes.png) | task 성공, mug 충돌 (t 44) | task 성공, 충돌 없음 | 안전 실패 → safe success |

각 묶음에는 다음이 들어 있다.

- 양쪽 전체 영상 (`vla_only.mp4`, `v11ab.mp4`).
- 같은 순간 keyframe 과 개입 timeline (`keyframes.png`).
- 판정 · call 별 개입 log · pairing 검사 · 재현 명령 (`case.json`).

#### A 에서 남은 충돌 14 건의 원인

방법: 충돌 순간 obstacle 에 닿은 body 와, 충돌 직전 call 의 reference / refined / executed 경로의
robot ↔ obstacle 거리를 쟀다.

| 원인 | episode |
|---|---|
| **arm link 접촉** — V11AB 제약 모델은 `--links gripper` 라 arm(link5 ~ 7)을 모른다 | t0e1, t1e0, t2e0, t2e2, t3e3 |
| gripper 직접 접촉 (refined 도 관통 또는 실행이 덜 실현) | t0e3, t1e2, t3e2, t3e3 |
| 쥔 bowl 이 obstacle 을 밀었다 | t1e3 |
| task 3 — obstacle 이 놓인 cabinet 을 통해 전달 (VLA-only · B 에서도 t ≈ 35 에 같은 형태) | t3e0, t3e1, t3e4 |
| 기록된 순간에는 닿은 body 가 없음 (sub-step 사이 접촉) | t0e2, t1e4 |

A 는 B 보다 계획을 잘 실현한다. 충돌 시점이 t 35 ~ 51 에서 t 77 ~ 139 로 늦어졌다. 대가도 있다.

- saturation 이 크다 (episode 당 최대 285 행).
- official success 가 11 → 3 으로 떨어졌다.
- 실패한 episode 에서는 latch `reach` 실패가 다시 많다 (closing 중 12 ~ 45 회).

#### 다음 개선 순서 (실제 비교에서 확인된 영향 순)

1. **V11AB-pb1** — latch / attach `reach` 를 geometry 를 관측한 순간의 자세로 잰다. task success 저하의 공통 원인이다.
2. arm link 를 제약 모델에 넣는 것. A 충돌 5 건이 여기에 해당한다.
3. 변환 실현도 (B 의 실현 부족, A 의 saturation).

---

## 개선 기록 — V11AB-pb1 · 변환 C · V11AB-pb2 (2026-10-15, 구현 · CPU 검증까지)

**기록 시각** 2026-10-15 02:30 KST. GPU 검증은 다른 checkout 의 GPU 작업이 끝난 뒤 진행한다 (아래 계획).

### V11AB-pb1 — grasp reach 를 geometry 관측 시점의 자세로 잰다 (commit `3dea2bc0`)

| 항목 | 내용 |
|---|---|
| 재현 run · episode | LIBERO t0 init0 smoke4 A · B (`SL5b-smoke.verify.json`), LIBERO 10 V11AB-B 실패 7 중 6 (`SL5b-libero10.verify.json`), SafeLIBERO Level I A 실패 다수 (closing 중 `reach` 실패 12 ~ 45 회) |
| 관찰 근거 | 들어올린 bowl 이 `occluded` 가 되고 geometry 가 마지막 관측 frame 에 고정된다. latch 와 `AG3S.attach` 는 손의 **지금** 원점에서 재므로 reach 가 120 mm 를 넘는다. 다른 evidence 검사는 모두 통과한다 |
| 원인 | 시점이 다른 두 값을 비교했다. held snapshot 은 이미 관측 시점 자세를 쓴다(T32 H2). reach 검사만 지금 자세를 썼다 |
| 변경 | `AG3S.reach_at_capture` (기본 False = V11AB). True 면 latch evidence 와 attach 관문이 centroid 를 그 geometry 를 관측한 frame 의 robot state 와 짝지어 잰다. 그 frame 의 state 는 history 에서 지우지 않는다. 한도 120 mm 와 다른 검사는 그대로 |
| 파일 | `AG3S/ag3s/runtime/pipeline.py`, `AG3S/trajopt/safe_policy.py`, `AG3S/trajopt/serve_safe.py` (`--reach-at-capture`), `AG3S/adapters/panda/profile.py` (`AG3S_VERSIONS`), `benchmark/runners/launch_libero_paired.py` (`--ag3s-version`) |
| 검증 (CPU) | `tests/trajopt/test_pb1_reach_at_capture.py` 5 건. V11AB 는 거절하고, pb1 은 capture 시점 reach 50 mm 로 attach 한다. 빈 손 닫힘과 state 가 없을 때는 여전히 거절한다. RB-Y1 golden bit 동일 |
| development 이식 주의 | flag 하나로 켜진다. RB-Y1 에서도 같은 의미다 (손을 들어올린 뒤 occluded). `_state_by_frame` 이 last-seen frame 을 지우지 않도록 바뀐다(flag 켤 때만). 이 문제는 development(d2)에도 전달되어 있다 |

### 변환 C — `lag_inverse_tracking` (adapter, commit `6d1723de`)

| 항목 | 내용 |
|---|---|
| 재현 | SL5c Level I A: refined 는 obstacle 을 피했는데 실행은 관통한 call 이 있다 (t1e2, t3e1, t3e4 등) |
| 원인 | A 는 직전 행의 deviation 이 계획대로 실현됐다고 가정한다. 그래서 clip 된 행의 부족분을 버린다. refined 가 reference 로 돌아온 뒤에는 원래 bit 를 내보내서 1 차 lag 꼬리(α · d)가 계속 밀고 나간다 |
| 변경 | response model 이 예측하는 실제 deviation 을 추적하고, refined 누적 위치를 목표로 한다: `Dx_k = ((D*_k − D^_{k−1}) − α d^_{k−1}) / β` |
| offline 비교 | A run 의 보정 call 965 개에 같은 계획으로 적용했다 (open loop). 예측 경로 − refined p50 15.3 → 10.9 mm. C 가 더 가까운 call 617, 더 먼 call 40. saturated 행 2501 → 3645 |
| 한계 | refined 자체가 5 행 안에서 p90 110 mm 벗어난다. Panda OSC 의 정상상태 한 step 은 약 12.5 mm 다. 변환만으로는 다 실현할 수 없다 → pb2 |

### V11AB-pb2 — minimal intervention gate (commit `7740ac17`, pb2 = pb1 + gate 10 mm)

| 항목 | 내용 |
|---|---|
| 재현 · 근거 | SL5c Level I A 의 보정 call 중 status `feasible`, verdict `budget_only`(위반 없음)가 대부분이다: 15 mm 이하 321, 15 ~ 60 mm 292, 60 mm 초과 66. 위반이 없는 chunk 에서도 경로를 크게 바꾼다 |
| 원인 | TrajOpt 목적함수의 track + continuity(직전 refined 의 겹치는 꼬리, w 0.5) + deflection-rate(w 20) 가 VLA 의 새 chunk 를 이전 계획 쪽으로 당긴다. Panda 에서는 이것이 큰 action 변화가 된다. official success 11 → 3 의 유력한 원인이다 (**가설, GPU 검증 대상**) |
| 변경 | `SqpConfig.minimal_intervention` (m, 기본 None = V11AB). SQP 전에 reference 의 step clearance 최소값(검증과 같은 행 · 면제)을 본다. 기준 이상이면 reference 를 그대로 계획으로 낸다. 판정은 `_finish` 로 해서 실제 solve 결과와 같다 (certified geometry 요구 포함) |
| 기준 10 mm | response model 의 예측 오차를 덮는다 (무변경 chunk p50 약 3 mm, 최대 7.5 mm, SL5a). 이보다 가까운 reference 는 여전히 최적화한다 |
| 파일 | `AG3S/trajopt/config.py`, `sqp.py`, `safe_policy.py` (`to.minimal_intervention` 기록), `serve_safe.py` (`--minimal-intervention-mm`), `AG3S/adapters/panda/profile.py` |
| 검증 (CPU) | `tests/trajopt/test_pb2_minimal_intervention.py` 4 건. 깨끗한 reference 는 bit 그대로 OPTIMAL 이다. 기준 안쪽 · 충돌 reference 는 최적화되어 안전하다. V11AB 기본에는 gate 가 없다 |
| 회귀 | tests/benchmark + tests/trajopt 1584 passed, 6 skipped. 제외한 2 파일은 이 변경과 무관하게 이미 실패한다(legacy path) |
| development 이식 주의 | gate 는 RB-Y1 에서도 동작한다. 다만 RB-Y1 V11AB 는 free-space 보정을 의도적으로 허용해 왔다 (T43 TA 기록). 이식할 때는 RB-Y1 기준선에서 따로 판정해야 한다 |

### GPU 검증 계획 (다른 checkout 의 GPU 작업 종료 뒤, 이 순서)

1. 원인 재현: LIBERO task 0 ~ 9 × init 0, VLA-only + V11AB-pb1 (B). 기준은 V11AB-B 3 / 10 이고, reach 실패 6 건이 회복되는지 본다.
2. 같은 dev 집합: SafeLIBERO Spatial Level I dev 20, VLA-only + V11AB-pb2 (C). V11AB-A 3 / 20 safe success · B 0 / 20 과 비교한다.
3. 비저하 확인: LIBERO task 0 ~ 9 × init 0, VLA-only + V11AB-pb2 (C).

### SL5d-1 — V11AB-pb1 원인 재현 검증, LIBERO task 0~9 × init 0 (2026-10-15)

**기록 시각** 2026-10-15 12:15 KST. **수치 출처** [`handoff/SL5d-pb1-repro.verify.json`](handoff/SL5d-pb1-repro.verify.json).
**그림** [`figures/sl5d-pb1-libero10-repro.png`](figures/sl5d-pb1-libero10-repro.png). **code** `6ad2aef2`, 변환 B. VLA-only 는 같은 launcher 호출로 paired 했다.

| 조건 | official success | 성공 task |
|---|---|---|
| VLA-only | 9 / 10 | t5 제외 |
| V11AB-B (기준선, `1df077b8`) | 3 / 10 | t0 · t1 · t7 |
| **V11AB-pb1-B** | **7 / 10** | t0 · t1 · t2 · t3 · t6 · t7 · t8 |

- 기준선에서 reach 때문에 attach 하지 못해 실패한 6 건 중 4 건(t2 · t3 · t6 · t8)이 attach 하고 성공했다.
- 모든 latch 기록의 reach 는 capture 시점 자세로 쟀다(`reach_state = capture frame N`).
- 남은 실패는 세 건이다.
  - t4: target 을 잃었다(`uncertified` HOLD 8 회).
  - t5: VLA-only 도 실패하는 episode 다.
  - t9: capture 시점으로 재도 reach 가 120 mm 를 넘는다(closing 중 35 회). SL4 에서 t4 · t5 · t9 는 ATS 가 다른 물체를 target 으로 고른 task 다. 그래서 남은 실패는 세 번째 후보인 target 탈락 · 오선택과 연결된다 (**가설**).
- 이 비교에서 "VLA-only 실패 → pb1 성공" 후보는 0 건이다. pb1 의 목적은 V11AB 의 task 저하를 줄이는 것이다.

### SL5d-2 — V11AB-pb2 (변환 C) 같은 dev 집합 비교와 LIBERO 비저하 확인 (2026-10-15)

**기록 시각** 2026-10-15 19:40 KST. **수치 출처** [`handoff/SL5d-pb2.verify.json`](handoff/SL5d-pb2.verify.json).
**그림** [`figures/sl5d-safelibero-spatial-LI-dev-pb2C.png`](figures/sl5d-safelibero-spatial-LI-dev-pb2C.png). **code** `bd692d19`. 각 run 은 같은 launcher 호출로 VLA-only 와 paired 했다.

#### SafeLIBERO Spatial Level I dev 20

| 조건 | official success | 충돌 | safe success |
|---|---|---|---|
| VLA-only | 11 | 18 | 2 |
| V11AB-B (기준선) | 3 | 18 | 0 |
| V11AB-A (기준선) | 3 | 14 | 3 |
| **V11AB-pb2-C** | 2 | **11** | 1 |

- 충돌은 지금까지 가장 적다 (18 → 11).
- 새 사례 1 건: [`sl5d-LI-t2e2-V11AB-pb2-C`](cases/sl5d-LI-t2e2-V11AB-pb2-C/keyframes.png). VLA-only 는 task 를 성공했지만 milk 를 넘어뜨렸다(t 84). pb2-C 는 milk 를 건드리지 않고 bowl 을 plate 에 놓았다. 구분: 안전 실패 → safe success.
- 그러나 official success 는 2 / 20 으로 낮다. A 의 사례 2 건(t0e0 · t2e4)은 pb2-C 에서는 실패했다.
  - t0e0: 충돌 없이 task 실패.
  - t2e4: t 48 에 충돌.
- 남은 충돌 11 건의 원인:

| 원인 | episode |
|---|---|
| **arm link 접촉** (link5 ~ 7) | t0e1, t1e3, t3e2, t3e3, t3e4 |
| gripper | t1e0 |
| 쥔 bowl | t2e0 |
| 기록된 순간에는 닿은 body 가 없음 (refined 는 피했는데 실행이 관통한 예 포함) | t0e3, t1e2, t1e4, t2e4 |

  A · B · VLA-only 에 있던 task 3 의 cabinet 경유 충돌(t ≈ 35)은 사라졌다.

#### LIBERO task 0 ~ 9 × init 0

| 조건 | official success |
|---|---|
| VLA-only | 9 / 10 |
| V11AB-B (기준선) | 3 / 10 |
| V11AB-pb1-B | 7 / 10 |
| **V11AB-pb2-C** | 6 / 10 (pb1-B 대비 t1 이 실패, held 중 TO 보정 20 call) |

#### gate 가 왜 덜 작동하나 — reference 자체가 접촉 단계에서 위반한다

minimal intervention gate 를 적용한 비율과, reference 가 위반해서 최적화된 call 의 latch 상태:

| run | gate 적용 / 최적화 | reference 가 위반해서 최적화된 call 의 latch 상태 |
|---|---|---|
| SafeLIBERO LI | 360 / 709 | closing 262, latched 187, held 63, searching 25 |
| LIBERO 10 | 184 / 135 | closing 41, held 34, latched 28 |

- 장애물을 추가하지 않은 LIBERO 에서도 reference 의 33 % 가 위반이다.
- 대표 위반은 closing 중 `panda_hand ↔ esdf` −16 ~ −39 mm (`answer_excludes_target`) 다. 손바닥이 target 이 아닌 주변 기하(받침 · plate · table)에 닿는다고 판정된다.
- 들고 있을 때는 `authorized panda_hand ↔ manipulated` 가 위반으로 잡힌다.
- 즉 V11AB 가 잡기 · 옮기기 · 놓기 같은 접촉 단계의 정상 동작을 위반으로 보고 수정한다. 이것이 남은 task success 저하의 주된 원인이다 (**측정에서 나온 해석, 다음 단계에서 확인**).

#### 다음 후보 (영향 순)

1. 접촉 단계의 과잉 제약. closing · held 중 손바닥 ↔ target 주변 받침 · 쥔 물체 행.
2. arm link 를 제약 모델에 포함 (`--links arms`). pb2-C 충돌 11 건 중 5 건.
3. target 탈락 · 오선택 (LIBERO t4 · t9).

### SL5e — V11AB-pb3 (arm link 포함) 결과와 접촉 단계 과잉 제약 진단 (2026-10-17)

**기록 시각** 2026-10-17 14:30 KST. **수치 출처** [`handoff/SL5e-pb3.verify.json`](handoff/SL5e-pb3.verify.json). **code** `a2489584`, 변환 C.
pb3 = pb2 + `--links arms` (panda_link1 ~ 7 + gripper sphere). 먼저 t0e0 smoke 로 arm sphere 가 HOLD 를 계속 일으키지 않는지 확인했다: HOLD 1 / 56, time budget 초과 0.

| 조건 | SafeLIBERO LI official | 충돌 | safe | LIBERO 10 |
|---|---|---|---|---|
| VLA-only | 11 | 18 | 2 | 9 |
| V11AB-pb2-C | 2 | 11 | 1 | 6 |
| **V11AB-pb3-C** | **0** | **8** | 0 | 4 |

arm link 를 넣으니 충돌은 가장 적다(8). 그러나 task success 는 더 떨어졌다. 제약이 늘어난 만큼 접촉 단계의 과잉 개입도 늘었다.

#### 접촉 단계 진단 (LIBERO 10 pb2-C, CPU)

방법: reference 가 위반해서 최적화된 latched · closing · held call 을 고른다. 기록된 sim state 에 reference 자세를 놓고, gripper 와 가장 가까운 물체를 쟀다 (`outputs/verify/SL5e/scripts/palm_neighbors.py`).

| latch | 가장 가까운 물체 | call |
|---|---|---|
| held | 쥔 bowl 자신 | 34 |
| closing | table | 24 |
| latched | cabinet top (bowl 의 받침) | 19 |
| closing | target bowl | 17 |
| latched | target bowl | 6 |

원인은 두 가지다.

1. **contact phase 가 고정이다.** client 가 latch 상태와 무관하게 `ag3s/phase = approach` 를 보낸다(V11AB client 설정). 그래서 잡는 중에도 손가락이 target 에서 approach 규칙의 0.4 × 50 mm = 20 mm 를 지켜야 하고, rim 파지가 위반이 된다.
2. **받침면이 obstacle margin 을 받는다.** 손가락이 받침면(table · cabinet top · plate)에 내려가면 sphere 근사가 3 ~ 37 mm 겹친다.

#### 판정 (2026-10-17, 사용자)

"held 면제 + grasp zone" 방향으로 진행한다. 각 변경은 따로 검증한다.

### SL5f — pb4 · pb5 초기 구현 결과와 결함 수정 (2026-10-17)

**기록 시각** 2026-10-17 17:50 KST. **수치 출처** [`handoff/SL5f-pb4pb5-initial.verify.json`](handoff/SL5f-pb4pb5-initial.verify.json).

같은 commit(`38722284`)에서 VLA-only 1 회와 pb5-C · pb4-C 를 돌리고 사후 pairing 했다. condition 검사는 모두 통과했다.

| 조건 | SafeLIBERO LI official / 충돌 / safe | LIBERO 10 |
|---|---|---|
| VLA-only | 11 / 18 / 2 | 9 |
| pb3-C | 0 / 8 / 0 | 4 |
| pb4-C (초기) | 0 / 7 / 0 | 4 |
| pb5-C (초기) | 2 / 9 / 2 (사례 t0e0 재현) | 3 |

task success 가 거의 회복되지 않았다. 분석에서 **구현 결함 두 개**를 찾았다.

1. **authorized mask 오류.** latch 의 grasp links(손가락 둘)로 만들어서 손바닥(`panda_hand`)이 빠졌다. 진단에서 위반이 많았던 것이 손바닥이다.
2. **기록 오류.** `to.grasp_contact` 를 TO 가 쓰지 않는 linearizer 에서 읽어서 늘 `{}` 였다. 그래서 규칙이 적용됐는지 알 수 없었다.

수정(`f4f97337`): mask 를 제약을 지을 때 쓴 ClearancePolicy authorized links (손바닥 + 손가락, verdict 와 같은 집합)로 바꾸고, 기록은 TO 의 linearizer 에서 읽는다.

수정 뒤 smoke (LIBERO t0 · t3 init 0, pb5-C):

- 두 task 모두 성공했다 (150 · 91 step).
- grasp zone 이 latched · closing 에서 수백 행에 적용됐다. held 면제는 held 에서 수십 행에 적용됐다.
- 최소 개입 gate 는 latched · closing 의 거의 모든 call 에 적용됐다.

위 표의 pb4 · pb5 수치는 **결함이 있는 초기 구현**의 결과로 남긴다. 수정본의 재검증은 다음 GPU 슬롯에서 한다.

### SL5g — pb4 · pb5 수정본(`f4f97337`) 재검증 (2026-10-18)

**기록 시각** 2026-10-18 (lead 의 rollout 13:33 ~ 15:26 KST, 분석 task 15:40 KST, verify.json `date` 2026-10-18). **수치 출처** [`handoff/SL5g.verify.json`](handoff/SL5g.verify.json), 지시서 [`handoff/SL5g.task.md`](handoff/SL5g.task.md). **실행 code** `a8ad92ef` (pb4 · pb5 의 code 는 SL5f 에서 고친 `f4f97337` 과 같고, `a8ad92ef` 는 SL5f docs 만 더한다). 분석은 CPU 만 썼고 새 rollout 은 0 이다. legacy 기준선은 일치했다 (violated start 14/15, has_target 15/15, frame0 clearance -29.031 mm; cuRobo 기준선은 GPU 금지라 실행하지 않았다).

SL5f 의 두 결함(authorized mask 에서 `panda_hand` 누락, `grasp_contact` 기록이 늘 `{}`)을 고친 pb4 · pb5 를 SafeLIBERO Level I dev(task 0~3 × init 0~4, n=20) 와 LIBERO 10(task 0~9 × init 0, n=10)에서 다시 돌렸다. 두 질문에 숫자로만 답한다. (1) 수정한 contact 단계 규칙이 실제로 적용됐는가. (2) VLA-only 가 성공한 episode 가 pb4 · pb5 에서 실패하는 것은 어느 단계에서 시작되는가.

새 용어: **pb4** = pb3(arm link 포함) + held exemption(쥔 상태에서 손이 쥔 물체에 대해 가진 행을 제약에서 뺀다). **pb5** = pb4 + latch phase(latched · closing 에서 V11AB phase 규칙을 latch 에서 받는다) + grasp zone 10 mm(손의 contact link 가 target 과 그 받침면에 10 mm 더 가까이 가도록 허용, 물체에서 수평 100 mm 안). **변환 C** = `lag_inverse_tracking`(EE delta → joint 역변환, 용어 절). **HOLD calls** = `safe_client.hold_calls` 로 센, SafePolicy 가 HOLD 를 낸 call 수.

#### 실행 run 과 비교 기준

모든 mode 앞에서 guard 가 HEAD = `a8ad92ef` 와 clean tree 를 확인했다 (`SL5g.task.md`, 기록 위치는 각 run 의 `logs/sl5g_chain.log`). 4 run 모두 rc=0.

| 집합 | 조건 | code | run (`AG3S_VLA-storage/results/benchmark/…`) |
|---|---|---|---|
| SafeLIBERO LI dev | VLA-only (재사용) | 38722284 | `20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only` |
| SafeLIBERO LI dev | pb3-C (재사용) | a2489584 | `20261017T034100_a2489584ee5b_sl5e-safelibero-spatial-dev-LI-vla_v11ab-V11AB-pb3-C` |
| SafeLIBERO LI dev | pb4-C 초기 (재사용) | 38722284 | `20261017T161857_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_v11ab-V11AB-pb4-C` |
| SafeLIBERO LI dev | pb5-C 초기 (재사용) | 38722284 | `20261017T153442_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_v11ab-V11AB-pb5-C` |
| SafeLIBERO LI dev | pb4-C 수정본 (신규) | a8ad92ef | `20261018T141823_a8ad92ef3356_sl5g-safelibero-LI-dev-LI-vla_v11ab-V11AB-pb4-C` |
| SafeLIBERO LI dev | pb5-C 수정본 (신규) | a8ad92ef | `20261018T133339_a8ad92ef3356_sl5g-safelibero-LI-dev-LI-vla_v11ab-V11AB-pb5-C` |
| LIBERO 10 init 0 | VLA-only (재사용) | 38722284 | `20261017T170230_38722284e2fa_sl5f-libero10-init0-vla_only` |
| LIBERO 10 init 0 | pb3-C (재사용) | a2489584 | `20261017T042820_a2489584ee5b_sl5e-libero10-init0-pb3-vla_v11ab-V11AB-pb3-C` |
| LIBERO 10 init 0 | pb4-C 초기 (재사용) | 38722284 | `20261017T172203_38722284e2fa_sl5f-libero10-init0-vla_v11ab-V11AB-pb4-C` |
| LIBERO 10 init 0 | pb5-C 초기 (재사용) | 38722284 | `20261017T170859_38722284e2fa_sl5f-libero10-init0-vla_v11ab-V11AB-pb5-C` |
| LIBERO 10 init 0 | pb4-C 수정본 (신규) | a8ad92ef | `20261018T151346_a8ad92ef3356_sl5g-libero10-init0-vla_v11ab-V11AB-pb4-C` |
| LIBERO 10 init 0 | pb5-C 수정본 (신규) | a8ad92ef | `20261018T150153_a8ad92ef3356_sl5g-libero10-init0-vla_v11ab-V11AB-pb5-C` |

#### VLA-only 재사용과 서로 다른 commit 을 pairing 한 근거 (사용자 지시 2026-10-18)

VLA-only 는 다시 돌리지 않고 SL5f(`38722284`)의 run 을 재사용했다. pb3 는 SL5e(`a2489584`), 신규 pb4 · pb5 는 `a8ad92ef` 에서 돌아서 비교 대상이 서로 다른 commit 에서 나왔다. 재사용 조건은 **실행 경로 · 환경 · checkpoint · settings · task/init · policy seed · reset 이력이 같을 것**이고, 아래 숫자가 그 근거다. VLA-only 경로(`--no-safe` server 는 `SeededPolicy` 만 쓴다)는 SafePolicy method 를 거치지 않는다.

| 항목 | 측정 |
|---|---|
| `38722284` → `a8ad92ef` 에서 바뀐 file | 4 개: `AG3S_BENCHMARK_LOG.md`, `SL5f-pb4pb5-initial.verify.json`, `safe_policy.py`, `test_pb4_latch_phase.py` (SafePolicy method · test · docs 뿐) |
| VLA-only 가 지나가는 경로의 diff | 변화 없음 (True): `AG3S/trajopt/serve_safe.py`, `AG3S/trajopt/policy_seed.py`, `benchmark`, `AG3S/adapters`, `third_party/openpi`, `common`, `scripts` (7 / 7) |
| settings diff 수 (protocol · runtime_schedule · versions · checkpoint) | LI 0 · 0 · 0 · 0 / LIBERO 10 0 · 0 · 0 · 0 |
| dependencies 동일 | LI True / LIBERO 10 True |
| policy seed 규칙 · `init_state == episode_idx` | LI True · True (n=20) / LIBERO 10 True · True (n=10) |
| VLA-only npz 전 array 일치 (SL5e `a2489584` · SL5d `bd692d19` 와) | LI True · True / LIBERO 10 True · True |
| VLA-only outcome 일치 (같은 두 commit 과) | LI True · True / LIBERO 10 True · True (official 11 / 9) |
| `vla_only_identity` 재실행 vs run 사본 | byte 단위 일치 True, sha256 `05e88544…` 동일, 신규 4 run 의 md5 동일 True |
| pairing key (task · init · policy seed · obstacle) 일치 | 6 version × 2 집합 모두 `same_keys` · `same_init` · `same_policy_seed` · `same_obstacle` True (아래 집계 표의 전제) |

reset 이력(fixture 위치를 뽑는 `reset` 횟수와 순서)을 따로 비교한 key 는 verify.json 에 없다. 이 항목은 VLA-only npz 전 array 가 세 commit(SL5d · SL5e · SL5f)에서 bit 단위로 같다는 위 숫자로 간접 확인되며, 별도 측정은 하지 않았다.

#### 집계

figure: [`sl5g-graph-versions-timeline.png`](figures/sl5g-graph-versions-timeline.png) (위 두 panel = version 별 bar, 아래 = LI t2e4 pb5 의 call 별 timeline, sidecar [`.json`](figures/sl5g-graph-versions-timeline.json)), [`sl5g-tables.png`](figures/sl5g-tables.png) (전이표 · 실패 원인 분류 · 최근접 물체).

![SL5g version 별 결과](figures/sl5g-graph-versions-timeline.png)

| 조건 | LI official /20 | LI collided /20 | LI safe_success /20 | LI HOLD calls (episodes) | LIBERO 10 official /10 | LIBERO 10 HOLD calls (episodes) |
|---|---|---|---|---|---|---|
| VLA-only | 11 | 18 | 2 | 0 (0 ep) | 9 | 0 (0 ep) |
| pb3-C | 0 | 8 | 0 | 196 (18 ep) | 4 | 17 (6 ep) |
| pb4-C 초기 | 0 | 7 | 0 | 187 (17 ep) | 4 | 18 (6 ep) |
| pb5-C 초기 | 2 | 9 | 2 | 218 (18 ep) | 3 | 26 (4 ep) |
| pb4-C 수정본 | 0 | 7 | 0 | 211 (16 ep) | 3 | 18 (6 ep) |
| pb5-C 수정본 | 1 | 11 | 1 | 204 (16 ep) | 5 | 15 (4 ep) |

collided = SafeLIBERO CollisionMonitor 가 wait 이후 obstacle 변위 합 |dpos| > 1 mm 를 본 episode. safe_success = official success 이면서 무충돌.

**episode 전이표** (같은 task · init · policy seed. S = success, F = fail, T/F = 충돌 여부)

| 전이 | LI official (T→T / T→F / F→T / F→F) | LI collided | LI safe_success | LIBERO 10 official |
|---|---|---|---|---|
| VLA-only → pb3 | 0 / 11 / 0 / 9 | 7 / 11 / 1 / 1 | 0 / 2 / 0 / 18 | 4 / 5 / 0 / 1 |
| VLA-only → pb4 초기 | 0 / 11 / 0 / 9 | 6 / 12 / 1 / 1 | 0 / 2 / 0 / 18 | 4 / 5 / 0 / 1 |
| VLA-only → pb5 초기 | 1 / 10 / 1 / 8 | 8 / 10 / 1 / 1 | 1 / 1 / 1 / 17 | 3 / 6 / 0 / 1 |
| VLA-only → pb4 수정본 | 0 / 11 / 0 / 9 | 6 / 12 / 1 / 1 | 0 / 2 / 0 / 18 | 3 / 6 / 0 / 1 |
| VLA-only → pb5 수정본 | 1 / 10 / 0 / 9 | 10 / 8 / 1 / 1 | 1 / 1 / 0 / 18 | 5 / 4 / 0 / 1 |
| pb4 초기 → 수정본 | 0 / 0 / 0 / 20 | 7 / 0 / 0 / 13 | 0 / 0 / 0 / 20 | 3 / 1 / 0 / 6 |
| pb5 초기 → 수정본 | 1 / 1 / 0 / 18 | 8 / 1 / 3 / 8 | 1 / 1 / 0 / 18 | 3 / 0 / 2 / 5 |
| pb4 수정본 → pb5 수정본 | 0 / 0 / 1 / 19 | 7 / 0 / 4 / 9 | 0 / 0 / 1 / 19 | 3 / 0 / 2 / 5 |

**후보 규칙** (두 규칙을 따로 센다): (a) VLA-only official fail → version official success, (b) VLA-only safe_success False → version safe_success True.

| version | LI (a) | LI (b) | LI 충돌 → 무충돌 | 그 중 official success | LI 무충돌 → 충돌 | LI official S→F | LIBERO 10 (a) | LIBERO 10 official S→F |
|---|---|---|---|---|---|---|---|---|
| pb3 | 0 | 0 | 11 | 0 | 1 | 11 | 0 | 5 |
| pb4 초기 | 0 | 0 | 12 | 0 | 1 | 11 | 0 | 5 |
| pb5 초기 | 1 | 1 | 10 | 1 | 1 | 10 | 0 | 6 |
| pb4 수정본 | 0 | 0 | 12 | 0 | 1 | 11 | 0 | 6 |
| pb5 수정본 | 0 | 0 | 8 | 0 | 1 | 10 | 0 | 4 |

pb5 초기 run 에서 (a)(b) 가 1 건(LI t0e0)이던 것이 수정본에서는 0 건이다. 수정본에서 규칙 (a) · (b) 후보는 LI · LIBERO 10 어디에도 없다 (`rule_case_bundles` = 0). LI 에서 S→F 로 바뀐 episode: pb4 t1e1, t1e4, t2e0, t2e1, t2e2, t2e3, t2e4, t3e0, t3e2, t3e3, t3e4; pb5 t1e4, t2e0, t2e1, t2e2, t2e3, t2e4, t3e0, t3e2, t3e3, t3e4. LIBERO 10 에서 S→F: pb4 t0e0, t1e0, t4e0, t7e0, t8e0, t9e0; pb5 t4e0, t7e0, t8e0, t9e0.

#### 수정한 규칙이 실제로 적용됐는가 (#2)

행(row) = 한 call 의 linearizer 마지막 clearance 평가의 (plan step, constraint sphere) 쌍. `held_exempt_rows` = latch 가 held, 가장 가까운 면이 target, authorized link 의 sphere 인 행(margin 을 -1 m 로 둠). `zone_rows` = latch 가 latched · closing, class 가 target 또는 support, authorized link, 물체 centroid 에서 수평 0.10 m 이내인 행(margin ≤ -0.010 m). `mi_applied` = minimal intervention gate 통과(reference 최소 clearance ≥ 10 mm 이면 reference 를 그대로 낸다).

| 집합 | version | calls | `grasp_contact` 가 `{}` 인 call (초기) / 기록된 call (수정본) | held_exempt rows (call 수) | zone rows (call 수) | contact_phase 출처 latch / request | MI gate 통과 / reference 위반 call | HOLD calls |
|---|---|---|---|---|---|---|---|---|
| LI | pb3 | 1120 | 기록 없음 (pb3 는 규칙 없음) | – | – | 0 / 0 | 368 / 585 | 196 |
| LI | pb4 초기 | 1120 | 1120 / 1120 | – | – | 0 / 0 | 385 / 573 | 187 |
| LI | pb5 초기 | 1078 | 1078 / 1078 | – | – | 942 / 136 | 360 / 565 | 218 |
| LI | pb4 수정본 | 1120 | 기록 1120 / 1120 | 1823 (149 call) | 0 (0 call) | 0 / 0 | 394 / 561 | 211 |
| LI | pb5 수정본 | 1084 | 기록 1084 / 1084 | 1051 (105 call) | 5397 (137 call) | 952 / 132 | 353 / 564 | 204 |
| LIBERO 10 | pb3 | 367 | 기록 없음 (pb3 는 규칙 없음) | – | – | 0 / 0 | 177 / 146 | 17 |
| LIBERO 10 | pb4 초기 | 369 | 369 / 369 | – | – | 0 / 0 | 185 / 131 | 18 |
| LIBERO 10 | pb5 초기 | 376 | 376 / 376 | – | – | 311 / 65 | 187 / 135 | 26 |
| LIBERO 10 | pb4 수정본 | 375 | 기록 375 / 375 | 141 (26 call) | 0 (0 call) | 0 / 0 | 196 / 133 | 18 |
| LIBERO 10 | pb5 수정본 | 344 | 기록 344 / 344 | 151 (35 call) | 7285 (77 call) | 275 / 69 | 194 / 108 | 15 |

초기 run 은 `grasp_contact` 가 pb4 · pb5 모두 모든 call 에서 `{}` 여서 규칙 적용 여부를 알 수 없었다(위 표의 `{}` 수). 수정본은 모든 call 에 기록이 있다. pb4 수정본은 grasp zone · latch phase 를 쓰지 않으므로 zone rows 가 0, contact_phase 출처가 0 인 것이 예상된 값이다. pb5 에서 contact_phase 출처가 `request` 인 call 은 latch 가 searching · placed 이거나, latch 는 latched 인데 phase 가 아직 latch 를 따르지 않은 call 이다 (calls whose contact_phase.latch (searching) differs from grasp.state (latched): LI pb5 20, LIBERO10 pb5 10 (the phase is picked before the latch advances on that call)).

pb5 수정본의 latch 별 적용 (calls / held_exempt rows / zone rows / zone 이 켜진 call / MI gate 통과 / reference 위반 / HOLD):

| 집합 | latch | calls | held_exempt rows | zone rows | zone call | MI 통과 | reference 위반 | HOLD |
|---|---|---|---|---|---|---|---|---|
| LI | searching | 74 | 0 | 0 | 0 | 48 | 19 | 4 |
| LI | latched | 328 | 0 | 2663 | 74 | 119 | 159 | 70 |
| LI | closing | 440 | 0 | 2734 | 63 | 112 | 257 | 74 |
| LI | held | 204 | 1051 | 0 | 0 | 56 | 115 | 56 |
| LI | placed | 38 | 0 | 0 | 0 | 18 | 14 | 0 |
| LIBERO 10 | searching | 45 | 0 | 0 | 0 | 42 | 0 | 8 |
| LIBERO 10 | latched | 105 | 0 | 4221 | 48 | 67 | 24 | 0 |
| LIBERO 10 | closing | 88 | 0 | 3064 | 29 | 36 | 40 | 4 |
| LIBERO 10 | held | 92 | 151 | 0 | 0 | 47 | 33 | 1 |
| LIBERO 10 | placed | 14 | 0 | 0 | 0 | 2 | 11 | 2 |

#### VLA-only 성공 → pb4 · pb5 실패 episode 의 분류 (#3)

실패 episode 는 31 건(LI pb4 11 · pb5 10, LIBERO 10 pb4 6 · pb5 4)이다. 분류는 숫자 기준으로만 한다. MuJoCo 접촉으로 lift 를 정의한다: **grasped** = 두 손가락 면이 모두 target body 에 닿음(contact dist ≤ 0), **lifted** = grasped 이고 target z − z(wait) ≥ 30 mm, **lost** = 첫 lifted 뒤 10 obs 이상 연속으로 grasp 없음 · rise < 30 mm, **over-lift** = 최고 target rise > 250 mm (VLA-only 성공 episode 의 최고 rise 는 LI 186.3 mm, LIBERO 10 182.2 mm). HOLD 연속 기준 N = 5 call. 위에서부터 처음 맞는 class 를 쓴다.

- A_no_lift_hold: 한 번도 lift 되지 않았고 최장 HOLD 연속 ≥ 5 call
- A_no_lift_other: 한 번도 lift 되지 않았고 최장 HOLD 연속 < 5 call
- B_lifted_lost: lifted 뒤 놓침
- C_lifted_overlift: lifted, 놓치지 않음, 최고 rise > 250 mm
- D_lifted_other: lifted, 놓치지 않음, 최고 rise ≤ 250 mm

| class | LI pb4 | LI pb5 | LIBERO 10 pb4 | LIBERO 10 pb5 | total |
|---|---|---|---|---|---|
| A_no_lift_hold | 2 | 1 | 1 | 1 | 5 |
| A_no_lift_other | 4 | 3 | 1 | 1 | 9 |
| B_lifted_lost | 0 | 2 | 1 | 1 | 4 |
| C_lifted_overlift | 5 | 4 | 2 | 0 | 11 |
| D_lifted_other | 0 | 0 | 1 | 1 | 2 |
| 합계 (VLA-only S → pb F) | 11 | 10 | 6 | 4 | 31 |

episode 별 (div = 실행 궤적이 VLA-only 와 처음 갈라지는 call, 그때의 latch · MI gate · reference 최소 clearance):

| 집합 | version | ep | class | div call | div 의 latch | MI 통과 | div reference clearance (mm) | max rise (mm) | VLA-only max rise (mm) | 최장 HOLD 연속 | attach / detach call |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LI | pb4 | t1e1 | C_lifted_overlift | 6 | closing | F | 9.35 | 730.2 | 158.6 | 1 | [9] / [11] |
| LI | pb4 | t1e4 | C_lifted_overlift | 3 | searching | F | 4.64 | 296.4 | 165.4 | 38 | [12] / [] |
| LI | pb4 | t2e0 | C_lifted_overlift | 4 | searching | F | 2.36 | 926.5 | 174.7 | 1 | [] / [] |
| LI | pb4 | t2e1 | C_lifted_overlift | 3 | latched | F | 0.73 | 519.9 | 156.2 | 15 | [25] / [] |
| LI | pb4 | t2e2 | A_no_lift_other | 1 | searching | F | -2.18 | 6.9 | 172.1 | 1 | [46] / [] |
| LI | pb4 | t2e3 | A_no_lift_hold | 5 | latched | F | 4.83 | 6.6 | 160.9 | 8 | [] / [] |
| LI | pb4 | t2e4 | C_lifted_overlift | 3 | latched | F | -3.79 | 507.5 | 186.3 | 0 | [] / [] |
| LI | pb4 | t3e0 | A_no_lift_hold | 2 | latched | F | 4.48 | 0.0 | 126.8 | 33 | [] / [] |
| LI | pb4 | t3e2 | A_no_lift_other | 2 | searching | F | -16.62 | 4.5 | 105.0 | 4 | [] / [] |
| LI | pb4 | t3e3 | A_no_lift_other | 2 | latched | F | -8.42 | 0.0 | 138.5 | 0 | [] / [] |
| LI | pb4 | t3e4 | A_no_lift_other | 2 | latched | F | -13.56 | 0.0 | 99.8 | 4 | [] / [] |
| LI | pb5 | t1e4 | C_lifted_overlift | 3 | searching | F | 4.64 | 296.4 | 165.4 | 38 | [12] / [] |
| LI | pb5 | t2e0 | C_lifted_overlift | 4 | searching | F | 2.36 | 926.5 | 174.7 | 1 | [] / [] |
| LI | pb5 | t2e1 | B_lifted_lost | 3 | latched | F | 0.73 | 350.3 | 156.2 | 1 | [12, 47] / [13, 51] |
| LI | pb5 | t2e2 | C_lifted_overlift | 1 | searching | F | -2.18 | 454.3 | 172.1 | 4 | [10] / [] |
| LI | pb5 | t2e3 | B_lifted_lost | 5 | latched | F | 4.83 | 144.9 | 160.9 | 0 | [9] / [18] |
| LI | pb5 | t2e4 | C_lifted_overlift | 3 | latched | F | -3.79 | 531.3 | 186.3 | 1 | [] / [] |
| LI | pb5 | t3e0 | A_no_lift_hold | 2 | latched | F | 4.48 | 0.0 | 126.8 | 33 | [] / [] |
| LI | pb5 | t3e2 | A_no_lift_other | 2 | searching | F | -16.62 | 4.5 | 105.0 | 4 | [] / [] |
| LI | pb5 | t3e3 | A_no_lift_other | 2 | latched | F | -8.42 | 0.0 | 138.5 | 0 | [] / [] |
| LI | pb5 | t3e4 | A_no_lift_other | 2 | latched | F | -13.56 | 0.0 | 99.8 | 4 | [] / [] |
| LIBERO 10 | pb4 | t0e0 | C_lifted_overlift | 14 | closing | F | -1.64 | 330.9 | 98.3 | 1 | [15, 24] / [16] |
| LIBERO 10 | pb4 | t1e0 | C_lifted_overlift | 10 | latched | F | -2.13 | 527.0 | 163.7 | 2 | [] / [] |
| LIBERO 10 | pb4 | t4e0 | A_no_lift_hold | 0 | searching | T | 142.19 | 0.0 | 159.9 | 8 | [] / [] |
| LIBERO 10 | pb4 | t7e0 | D_lifted_other | 5 | latched | F | 5.53 | 196.0 | 182.2 | 1 | [10, 16, 21] / [13, 19] |
| LIBERO 10 | pb4 | t8e0 | B_lifted_lost | 11 | latched | F | -7.12 | 304.5 | 123.1 | 1 | [15] / [17] |
| LIBERO 10 | pb4 | t9e0 | A_no_lift_other | 6 | latched | F | 8.91 | 18.0 | 103.8 | 0 | [] / [] |
| LIBERO 10 | pb5 | t4e0 | A_no_lift_hold | 0 | searching | T | 142.19 | 0.0 | 159.9 | 8 | [] / [] |
| LIBERO 10 | pb5 | t7e0 | D_lifted_other | 5 | latched | F | 5.53 | 243.4 | 182.2 | 0 | [10, 16, 21] / [13, 19] |
| LIBERO 10 | pb5 | t8e0 | B_lifted_lost | 11 | latched | F | -7.12 | 237.3 | 123.1 | 1 | [15] / [30] |
| LIBERO 10 | pb5 | t9e0 | A_no_lift_other | 6 | latched | F | 8.91 | 18.0 | 103.8 | 0 | [] / [] |

갈라지는 call 의 집계 (실패 episode 수 기준, `divergence_summary`):

| 집합 | version | latch searching | latched | closing | MI 통과 False | MI 통과 True | ref clearance < 0 | 0 ~ 10 mm | ≥ 10 mm | verdict budget_only | verdict uncertified |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LI | pb4 | 4 | 6 | 1 | 11 | 0 | 5 | 6 | 0 | 11 | 0 |
| LI | pb5 | 4 | 6 | 0 | 10 | 0 | 5 | 5 | 0 | 10 | 0 |
| LIBERO 10 | pb4 | 1 | 4 | 1 | 5 | 1 | 3 | 2 | 1 | 5 | 1 |
| LIBERO 10 | pb5 | 1 | 3 | 0 | 3 | 1 | 1 | 2 | 1 | 3 | 1 |

LI 에서 실패한 episode 중 VLA-only 도 충돌했던 것은 pb4 9 / 11, pb5 9 / 10 건이다.

실행된 action 의 포화 (lift 구간 env step 중 실행 action 의 `dz ≥ 0.999`):

| 집합 | version | 실패 episode (포화 step / lifted step) | 같은 episode 의 VLA-only |
|---|---|---|---|
| LI | pb4 | 452 / 999 | 0 / 511 |
| LI | pb5 | 398 / 916 | 0 / 473 |
| LIBERO 10 | pb4 | 168 / 505 | 0 / 206 |
| LIBERO 10 | pb5 | 32 / 161 | 0 / 151 |

latch 와 실제 물리적 lift 의 관계 (call 단위, VLA-only 성공 → pb 실패 episode): LI pb5 에서 latch `closing` 인데 물리적으로 lifted 인 call 82, `held` 이고 lifted 91; LI pb4 는 각각 117 · 77.

#### contact 단계에 남은 제약 (#4)

SL5e 와 같은 방식(`palm_neighbors.py`)이다. 선택: `to.minimal_intervention.applied` 가 False 이고 reference clearance < 0 인 latched · closing · held call. 그 call 의 reference 5 행 자세를 sim state 에 놓고 gripper0 geom 에서 30 mm 이내인 non-robot body 를 센다. 표는 call 수이고 SL5e 열은 LIBERO 10 pb2-C 의 같은 측정이다.

| latch | 가장 가까운 물체 (gripper geom, 30 mm 안) | LIBERO 10 pb4 | LIBERO 10 pb5 | LI pb4 | LI pb5 |
|---|---|---|---|---|---|
| latched | target bowl | 13 | 10 | 36 | 38 |
| latched | obstacle | 0 | 0 | 65 | 67 |
| latched | other object | 12 | 12 | 19 | 22 |
| latched | table | 0 | 0 | 18 | 12 |
| latched | nothing ≤ 30 mm | 2 | 2 | 21 | 20 |
| closing | target bowl | 77 | 32 | 162 | 184 |
| closing | obstacle | 0 | 0 | 12 | 8 |
| closing | other object | 5 | 5 | 18 | 9 |
| closing | table | 0 | 0 | 6 | 7 |
| closing | nothing ≤ 30 mm | 3 | 3 | 47 | 49 |
| held | target bowl | 19 | 31 | 108 | 114 |
| held | other object | 1 | 0 | 9 | 0 |
| held | table | 1 | 1 | 4 | 0 |
| held | nothing ≤ 30 mm | 0 | 1 | 0 | 1 |
| calls | reference 위반으로 최적화된 call | 133 | 97 | 525 | 531 |

SL5e 의 LIBERO 10 pb2-C 는 같은 선택에서 103 call 이었다 (표는 [`sl5g-tables.png`](figures/sl5g-tables.png) (c) 에 나란히 있다). arm link(`robot0_link*`) 기준 최근접 body 의 같은 분류는 verify.json `nearest_body_by_latch_arm_links` 에 있다 (LI pb5 의 arm link 기준 `nothing within 30 mm` 는 closing 228 · latched 97 · held 84 call).

#### 충돌 (#5)

LI 충돌은 VLA-only 18 · pb4 7 · pb5 11 건이다. 충돌 body = 충돌 시점 전후(obs −1 ~ +2) 에 obstacle geom 과 MuJoCo 거리가 가장 작은 robot body 또는 target body. VLA-only 의 충돌 body 종류: 15 gripper · 3 held target.

| version | ep | 충돌 env step | call | latch | HOLD | MI 통과 | reference clearance (mm) | 직전 call HOLD | 충돌 body (거리, 종류) |
|---|---|---|---|---|---|---|---|---|---|
| pb4 | t1e0 | 36 | 3 | searching | F | F | -32.0 | F | `gripper0_right_gripper` (-3.0 mm, gripper) |
| pb4 | t1e4 | 70 | 10 | closing | F | T | 24.53 | F | `akita_black_bowl_1_main` (-27.2 mm, held_target) |
| pb4 | t2e0 | 63 | 8 | searching | F | F | -21.0 | T | `akita_black_bowl_1_main` (-172.2 mm, held_target) |
| pb4 | t3e0 | 93 | 14 | latched | F | T | 16.52 | F | `robot0_link5` (-1.6 mm, arm_link) |
| pb4 | t3e1 | 124 | 20 | latched | F | F | 1.1 | F | `robot0_link5` (-2.4 mm, arm_link) |
| pb4 | t3e3 | 105 | 17 | latched | F | F | -15.4 | F | `robot0_link5` (-1.8 mm, arm_link) |
| pb4 | t3e4 | 270 | 50 | latched | T | F | -42.0 | T | `gripper0_rightfinger` (-9.4 mm, gripper) |
| pb5 | t0e1 | 134 | 22 | closing | F | F | -27.39 | F | `akita_black_bowl_1_main` (-0.2 mm, held_target) |
| pb5 | t0e2 | 190 | 34 | latched | F | F | -26.97 | F | `robot0_link7` (2.2 mm, arm_link) |
| pb5 | t1e0 | 36 | 3 | searching | F | F | -32.0 | F | `gripper0_right_gripper` (-3.0 mm, gripper) |
| pb5 | t1e2 | 103 | 16 | closing | F | F | -14.46 | F | `akita_black_bowl_1_main` (0.0 mm, held_target) |
| pb5 | t1e3 | 152 | 26 | held | F | F | -30.13 | F | `robot0_link5` (-0.8 mm, arm_link) |
| pb5 | t1e4 | 70 | 10 | closing | F | T | 24.53 | F | `akita_black_bowl_1_main` (-27.2 mm, held_target) |
| pb5 | t2e0 | 63 | 8 | searching | F | F | -21.0 | T | `akita_black_bowl_1_main` (-172.2 mm, held_target) |
| pb5 | t3e0 | 93 | 14 | latched | F | T | 16.52 | F | `robot0_link5` (-1.6 mm, arm_link) |
| pb5 | t3e1 | 124 | 20 | latched | F | F | 1.1 | F | `robot0_link5` (-2.4 mm, arm_link) |
| pb5 | t3e3 | 105 | 17 | latched | F | F | -15.4 | F | `robot0_link5` (-1.8 mm, arm_link) |
| pb5 | t3e4 | 270 | 50 | latched | T | F | -42.0 | T | `gripper0_rightfinger` (-9.4 mm, gripper) |

요약: pb4 latch=searching 2, contact_kind=gripper 2, latch=closing 1, contact_kind=held_target 2, latch=latched 4, contact_kind=arm_link 3; pb5 latch=closing 3, contact_kind=held_target 4, latch=latched 5, contact_kind=arm_link 5, latch=searching 2, contact_kind=gripper 2, latch=held 1.

#### pb4 와 pb5 가 갈린 episode (#6)

| 집합 | ep | official pb4 / pb5 | collided pb4 / pb5 | 갈리는 env step / call | pb4 의 그 call | pb5 의 그 call | reference clearance (mm) pb4 → pb5 | TrajOpt status | 실행 수정량 (`max_abs_action_change_executed`) |
|---|---|---|---|---|---|---|---|---|---|
| LI | t0e1 | F / F | False / True | 65 / 9 | latched · MI False · zone 0 · phase – | latched · MI False · zone 49 · phase pre_grasp | -32.72 → -32.72 | feasible → feasible | 1.927 → 1.927 |
| LI | t0e2 | F / F | False / True | 60 / 8 | latched · MI False · zone 0 · phase – | latched · MI False · zone 33 · phase pre_grasp | -9.38 → -9.38 | feasible → feasible | 0.971 → 0.971 |
| LI | t1e1 | F / S | False / False | 50 / 6 | closing · MI False · zone 0 · phase – | closing · MI True · zone 67 · phase grasp | 9.35 → 11.95 | feasible → optimal | 0.093 → 0.000 |
| LI | t1e2 | F / F | False / True | 70 / 10 | latched · MI False · zone 0 · phase – | latched · MI False · zone 111 · phase approach | -0.03 → -0.03 | feasible → feasible | 1.016 → 1.016 |
| LI | t1e3 | F / F | False / True | 50 / 6 | closing · MI False · zone 0 · phase – | closing · MI False · zone 25 · phase grasp | 5.17 → 5.17 | feasible → feasible | 0.079 → 0.081 |
| LIBERO 10 | t0e0 | F / S | – / – | 80 / 14 | closing · MI False · zone 0 · phase – | closing · MI False · zone 190 · phase grasp | -1.64 → 8.36 | feasible → feasible | 0.303 → 0.186 |
| LIBERO 10 | t1e0 | F / S | – / – | 60 / 10 | latched · MI False · zone 0 · phase – | latched · MI False · zone 110 · phase pre_grasp | -2.13 → 3.15 | feasible → feasible | 0.370 → 0.084 |

갈리는 call 의 reference clearance 는 7 episode 중 4 건(LI t0e1 · t0e2 · t1e2 · t1e3)에서 두 version 이 같고, 3 건(LI t1e1, LIBERO 10 t0e0 · t1e0)에서 다르다. official 이 갈린 것은 LI t1e1, LIBERO 10 t0e0 · t1e0 세 건이고 모두 pb5 가 success 다. LI 의 나머지 4 건(t0e1 · t0e2 · t1e2 · t1e3)은 official 이 둘 다 fail 이고 pb4 는 무충돌, pb5 는 충돌이다.

#### 사례 4건

규칙 (a) · (b) 후보가 수정본에 없으므로 규칙 사례 묶음은 0 건이다. 안전 개선 2건과 대표 회귀(C_lifted_overlift) 2건을 묶었다. 묶음 위치 `outputs/verify/SL5g/cases/<case>/`, 큰 video 는 복사하지 않고 경로만 둔다. 재현 명령은 각 묶음의 `reproduce` 에 있다.

| 묶음 | 종류 | episode | VLA-only | 수정본 | 갈라지는 시점 | SL5g class |
|---|---|---|---|---|---|---|
| `safety_LI_t0e0_pb5` | safety_improvement | LI t0e0 | official False · collided True (collide_t 35) | pb5: official False · collided False | env step 30 / call 2 | – |
| `safety_LI_t2e2_pb5` | safety_improvement | LI t2e2 | official True · collided True (collide_t 84) | pb5: official False · collided False | env step 25 / call 1 | C_lifted_overlift |
| `regression_LI_t2e4_pb5` | regression_C_lifted_overlift | LI t2e4 | official True · collided True (collide_t 44) | pb5: official False · collided False | env step 35 / call 3 | C_lifted_overlift |
| `regression_LIBERO10_t1e0_pb4` | regression_C_lifted_overlift | LIBERO10 t1e0 | official True · collided – | pb4: official False · collided – | env step 60 / call 10 | C_lifted_overlift |

| 묶음 | VLA-only video | pb video |
|---|---|---|
| `safety_LI_t0e0_pb5` | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only/videos/safelibero_spatial_t0_e0_failure.mp4` | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261018T133339_a8ad92ef3356_sl5g-safelibero-LI-dev-LI-vla_v11ab-V11AB-pb5-C/videos/safelibero_spatial_t0_e0_failure.mp4` |
| `safety_LI_t2e2_pb5` | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only/videos/safelibero_spatial_t2_e2_success.mp4` | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261018T133339_a8ad92ef3356_sl5g-safelibero-LI-dev-LI-vla_v11ab-V11AB-pb5-C/videos/safelibero_spatial_t2_e2_failure.mp4` |
| `regression_LI_t2e4_pb5` | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only/videos/safelibero_spatial_t2_e4_success.mp4` | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261018T133339_a8ad92ef3356_sl5g-safelibero-LI-dev-LI-vla_v11ab-V11AB-pb5-C/videos/safelibero_spatial_t2_e4_failure.mp4` |
| `regression_LIBERO10_t1e0_pb4` | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261017T170230_38722284e2fa_sl5f-libero10-init0-vla_only/videos/libero_spatial_t1_e0_success.mp4` | `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261018T151346_a8ad92ef3356_sl5g-libero10-init0-vla_v11ab-V11AB-pb4-C/videos/libero_spatial_t1_e0_failure.mp4` |

figure: [`sl5g-scene-cases.png`](figures/sl5g-scene-cases.png) (위 두 줄 = LI t2e4 VLA-only · pb5, 아래 두 줄 = LI t0e0 VLA-only · pb5. + = target bowl, ○ = 가장 가까운 robot body ↔ 물체, □ / ◇ = 갈라진 call 의 reference · refined 최근접 행, sidecar [`.json`](figures/sl5g-scene-cases.json)).

![SL5g 사례 scene](figures/sl5g-scene-cases.png)

![SL5g 전이표 · 분류 · 최근접 물체](figures/sl5g-tables.png)

#### 가설별 숫자 (판정 없음)
verify.json `hypotheses_numbers` 의 6 개 항목을 순서대로 가설 1 ~ 6 으로 부른다. 표는 숫자만 옮긴 것이다.

**가설 1. grasp zone(pb5)이 contact 단계에서 binding 인 행을 덮지 못한다**

| pb5 | latched (zone 이 켜진 call / call) | closing (zone 이 켜진 call / call) |
|---|---|---|
| LI | 74 / 328 | 63 / 440 |
| LIBERO 10 | 48 / 105 | 29 / 88 |

closing 중 reference 위반 call 의 최근접 gripper body 가 target bowl 인 call: LI pb4 162 · pb5 184, LIBERO 10 pb4 77 · pb5 32. zone 조건: latch 가 latched · closing, class 가 target 또는 support, authorized link, 조작 중인 물체 centroid 에서 수평 0.10 m 이내.


| pb5 | latch | MI 통과 call / call | reference clearance < 0 인 call |
|---|---|---|---|
| LI | latched | 119 / 328 | 159 |
| LI | closing | 112 / 440 | 257 |
| LI | held | 56 / 204 | 115 |
| LIBERO 10 | latched | 67 / 105 | 24 |
| LIBERO 10 | closing | 36 / 88 | 40 |
| LIBERO 10 | held | 47 / 92 | 33 |

갈라지는 call 에서 MI gate 가 통과되지 않은 실패 episode: LI pb4 11 / 11, pb5 10 / 10, LIBERO 10 pb4 5 / 6, pb5 3 / 4 (위 episode 별 표의 `MI 통과` 열과 같다).

**가설 3. arm link · obstacle 행이 approach 를 막는다 (class A, lift 못 함)**

class A 수: LI pb4 6 · pb5 4, LIBERO 10 pb4 2 · pb5 2. 각 episode 의 target 까지 최소 EEF 거리, 갈라지는 call 의 reference 최근접 행, 최장 HOLD 연속:

| 집합 | version | ep | min eef–target (mm) | 갈라지는 call 의 reference 최근접 행 | 최장 HOLD 연속 |
|---|---|---|---|---|---|
| LI | pb4 | t2e2 | 43.0 | 30 mm 안에 없음 | 1 |
| LI | pb4 | t2e3 | 30.5 | `gripper0_right_gripper` ↔ `akita_black_bowl_1_main` -11.8 mm | 8 |
| LI | pb4 | t3e0 | 141.9 | 30 mm 안에 없음 | 33 |
| LI | pb4 | t3e2 | 76.4 | `gripper0_finger_joint1_tip` ↔ `yellow_book_obstacle_1_main` 12.6 mm | 4 |
| LI | pb4 | t3e3 | 94.2 | `gripper0_right_gripper` ↔ `wine_bottle_obstacle_1_main` 1.9 mm | 0 |
| LI | pb4 | t3e4 | 103.5 | `gripper0_leftfinger` ↔ `red_coffee_mug_obstacle_1_main` -5.9 mm | 4 |
| LI | pb5 | t3e0 | 141.9 | 30 mm 안에 없음 | 33 |
| LI | pb5 | t3e2 | 76.4 | `gripper0_finger_joint1_tip` ↔ `yellow_book_obstacle_1_main` 12.6 mm | 4 |
| LI | pb5 | t3e3 | 94.2 | `gripper0_right_gripper` ↔ `wine_bottle_obstacle_1_main` 1.9 mm | 0 |
| LI | pb5 | t3e4 | 103.5 | `gripper0_leftfinger` ↔ `red_coffee_mug_obstacle_1_main` -5.9 mm | 4 |
| LIBERO 10 | pb4 | t4e0 | 99.5 | 30 mm 안에 없음 | 8 |
| LIBERO 10 | pb4 | t9e0 | 108.9 | `gripper0_leftfinger` ↔ `akita_black_bowl_1_main` 18.4 mm | 0 |
| LIBERO 10 | pb5 | t4e0 | 99.5 | 30 mm 안에 없음 | 8 |
| LIBERO 10 | pb5 | t9e0 | 108.9 | `gripper0_leftfinger` ↔ `akita_black_bowl_1_main` 18.4 mm | 0 |

robot0_link(arm) 가 충돌 body 인 pb 충돌 episode: pb4 t3e0, t3e1, t3e3; pb5 t0e2, t1e3, t3e0, t3e1, t3e3.

**가설 4. latch 가 held 로 넘어가지 못한 채 물리적으로 lift 된 구간이 있다 (held exemption 이 적용되지 않는 구간)**

| 집합 | version | closing 인 call (lifted / not lifted), 실패 episode | held 인 call (lifted / not lifted) | 실패 episode 수 | VLA-only 성공 → pb 성공 episode 수 | 성공 episode 의 closing·lifted / held·lifted call |
|---|---|---|---|---|---|---|
| LI | pb4 | 117 / 204 | 77 / 10 | 11 | 0 | 0 / 0 |
| LI | pb5 | 82 / 148 | 91 / 13 | 10 | 1 | 0 / 7 |
| LIBERO 10 | pb4 | 51 / 50 | 44 / 8 | 6 | 3 | 3 / 17 |
| LIBERO 10 | pb5 | 3 / 29 | 20 / 24 | 4 | 5 | 0 / 41 |

lift 의 정의: both finger sides touch the target (MuJoCo contact) and target rise >= 30 mm, at the call's obs. 이 STEP 은 reach check 를 측정하지 않았다(다음 STEP SL5h 의 D3).

**가설 5. lift 뒤 실행 z command 가 포화한다 (class C)**

위 "실행된 action 의 포화" 표가 이 숫자다: 실패 episode 의 lift 구간에서 dz ≥ 0.999 인 step 이 LI pb4 452 / 999 · pb5 398 / 916, LIBERO 10 pb4 168 / 505 · pb5 32 / 161 이고, 같은 episode 의 VLA-only 는 모두 0 이다.

**가설 6. LI 의 S→F episode 는 VLA-only 가 충돌하면서 성공한 episode 다**

LI 에서 VLA-only 는 official 11 · collided 18 이고 safe_success 는 2 이다. S→F 인 episode 중 VLA-only 도 충돌했던 것은 pb4 9 / 11, pb5 9 / 10 이다.

#### 측정하지 않은 것

- cuRobo regression baseline: 이 STEP 은 GPU 사용 금지라 실행하지 않았다.
- 실행된(non-HOLD) call 의 binding constraint 행: calls log 는 이것을 collision / follow_to verdict 안에만 남긴다. #3 (i) 은 reference · refined 행의 MuJoCo 근접도를 대용으로 썼다.
- counterfactual 재실행(pb5 zone 확대, 변환 C 제거, gate threshold 변경 등): GPU rollout 이 필요해 실행하지 않았다.
- pb3 · 초기 run: 집계(#1)와 규칙 기록 수(#2)만 있다. #3 ~ #6 은 수정본 pb4 · pb5 만 다룬다.
- 물리적 grasp 상태는 obs 시점(`sim_state_at_return`)의 MuJoCo 접촉으로 정했다. obs 사이의 physics substep 은 보지 않았다.

**판정 대기 (사용자).** 원인 판정과 다음 수정 방향은 사용자 판정 전이다. 위는 가설별 숫자만 남긴 것이다.

### SL5h — over-lift · action 포화 출처 진단, CPU (2026-10-19)

**기록 시각** 2026-10-19 (SL5h task · verify.json `date` 2026-10-19 KST). **수치 출처** [`handoff/SL5h.verify.json`](handoff/SL5h.verify.json), 지시서 [`handoff/SL5h.task.md`](handoff/SL5h.task.md). **code** `8e7ce1f8`(SL5g docs commit. 실행 code 는 `a8ad92ef` 와 같다). 이 STEP 은 CPU 만 썼고(`CUDA_VISIBLE_DEVICES=`) code 를 고치지 않았다. legacy 기준선은 일치했다 (violated start 14/15, has_target 15/15, frame0 clearance -29.031 mm). cuRobo 기준선은 GPU 금지라 실행하지 않았다.

SL5g 에서 VLA-only 가 성공한 episode 31 건이 pb4 · pb5 에서 실패했고 가장 큰 유형이 C_lifted_overlift 11 건이었다. 이 STEP 은 over-lift 와 실행 action 의 포화가 **VLA · TrajOpt(TO) · 역변환 C · latch(reach check) 중 어느 단계에서 시작되는가**, 그리고 **pb2 이후 변경과 함께 나타났는가**를 숫자로 가른다. 판정 문장은 쓰지 않는다.

#### 용어 (이 절에서 새로 쓰는 말)

| 용어 | 뜻 (verify.json `_definitions`) |
|---|---|
| **over-lift** | episode 의 최고 target rise 가 250 mm 를 넘는 것. rise = target z − z(wait), lift 는 두 손가락 면이 target 에 닿고(MuJoCo contact dist ≤ 0) rise ≥ 30 mm (SL5g #3). VLA-only 성공 episode 의 최고 rise 는 186.3 mm 이하였다 |
| **saturation (포화, sat)** | 실행 action 의 arm 성분(x · y · z · rx · ry · rz, gripper 제외) 중 하나라도 \|a\| ≥ 0.999 |
| **stage ①~⑤** | ① `actions_policy`(VLA 원본 chunk) → ② TO 수정(`joint_refined ≠ joint_reference`, EE 변화 = FK(`panda_grip_site`)) → ③ 역변환 C 의 clip(`saturated_rows`, `max_abs_correction`) → ④ `actions_response`(client 로 돌려준 action) → ⑤ `actions_executed` |
| **TO-changed row** | chunk 10 행 중 TO 가 바꾼 행(`joint_refined ≠ joint_reference`). MI gate 가 통과된 call 은 refined = reference 라 모든 행이 TO-unchanged 다 |
| **D\*** | TO 가 요청한 EE 변화 = FK(`joint_refined[k,4]`) − FK(`joint_reference[k,4]`) (mm, base 축 = world 축). D\*_z 는 그 z 성분. 역변환 C 가 받는 입력 deviation 의 크기 |
| **realised (realised_frac_obs)** | D\* 중 실제 EE 가 따라간 비율: (eef_obs(o+5) − FK(reference[k,4])) 를 D\* 방향에 사영한 값 / \|D\*\|. 1 이면 요청이 그대로 실현, 0 이면 실현 안 됨, 1 초과면 요청보다 더 움직임. `realised_fraction_model` 은 server 가 기록한 response model 의 예측값 |
| **first stage** | 포화한 response row 가 처음 포화한 단계. `vla` = VLA 원본 row 가 이미 포화, `conv_on_to_row` = 역변환이 clip 한 row 이고 그 row 를 TO 가 바꿈, **`conv_carry`** = 역변환이 clip 한 row 인데 그 row 는 TO 가 바꾸지 않음(앞 row 의 부족분을 이월한 것), `unclipped` = clip 없이 포화 |
| **windup** | 역변환 C 의 clip 이 call 안에서 같은 방향으로 계속 쌓이는 모양. 이 STEP 의 조작 정의: `conversion.saturated_rows` 가 비어 있지 않고 cos(D\*_k, D\*_{k−1}) > 0.5 인 call 의 최대 연속 run. 길이 ≥ 3 · ≥ 5 call 인 run 의 수와 최장 길이를 센다 |
| **lag_target / lag_reference** | 목표 EE 대비 실제 EE 의 누적 오차: \|FK(refined 또는 reference [k,4]) − eef_obs(o+5)\| (mm) |
| **tracking_residual** | verify.json D2 에 `tracking_residual_mm` 으로 기록된 값. 정의 문장은 verify.json 에 없다. TO-unchanged call 은 전부 0 |
| **역변환 C 의 상태** | `d^`, `D^` 는 매 call 0 에서 시작한다(`AG3S/adapters/panda/action.py` `joint_chunk_to_delta_chunk`). call 사이에 이월되는 상태가 없다. 식(figure sidecar `sl5h-tables.json` `definitions`): `Dx_k = ((D*_k − D^_{k−1}) − α d^_{k−1}) / β`, `d^_k = α d^_{k−1} + β·clip(Dx_k)`, `D^_k = D^_{k−1} + d^_k`, 위치 α 0.717 β 0.071, 회전 α 0.875 β 0.031, `a'_k = clip(a_k + Dx_k / scale, ±1)` |
| **mujoco_ref_rows** | call 시점 sim state 에서 팔만 `joint_reference` 5 행 자세로 옮기고(bowl 은 그대로) 30 mm 이내인 non-robot geom 을 본다. `neither_within_30mm` = 5 행 모두에서 target 이 아닌 물체도 table 도 robot geom 에서 30 mm 안에 없음 |
| **centroid_offset** | AG3S 의 manipulated centroid(`grasp.tip_alignment.centroid_m`, base)에서 call 시점 MuJoCo target body 위치를 뺀 값 |


비교 대상 run: 수정본 SL5g pb4-C · pb5-C (LI, LIBERO 10), 이전 SL5e pb3-C · SL5d pb2-C · SL5d pb1-B(LIBERO 10 만) · SL5c V11AB-A · B, 그리고 VLA-only(SL5f). SL5c A · B 와 pb1 은 MI gate · arms · pb4/pb5 규칙이 없던 run 이다. 변환 표기는 A = `lag_inverse`, B = `command_offset`, C = `lag_inverse_tracking` (위 개선 기록 절).

**재현 확인** SL5g 의 lift 구간 실행 `dz ≥ 0.999` step 수(452/999, 398/916, 168/505, 32/161)가 정확히 재현됐다. MuJoCo lift 지표와 두 손가락 track 은 SL5g `mj_probe` 와 6 조합(LI · LIBERO 10 × vla_only · pb4 · pb5, n = 20 · 10) 모두 같다. 자원: probe peak RSS 2.49 GB (≤ 3 GB), pod working set 18405-18533 MiB.

#### D1 — 포화 분해: 어느 단계에서 처음 포화하는가

figure: [`sl5h-graph-stages-reach.png`](figures/sl5h-graph-stages-reach.png) (a ~ f), [`sl5h-tables.png`](figures/sl5h-tables.png) (T1 ~ T4), [`sl5h-scene-t2e4-pb5.png`](figures/sl5h-scene-t2e4-pb5.png). sidecar 는 같은 이름의 `.json`.

![SL5h 단계 분해와 reach](figures/sl5h-graph-stages-reach.png)

lift 구간의 env step 단위(실행 action ⑤)와 response row 단위(③ · ④)로 센다. `mi` = MI gate 가 통과된 lifted call 수 (그 call 들의 response sat row).

| 집합 | version | 변환 | lifted env step | 실행 any-axis sat | 실행 z ≥ 0.999 | TO-changed row 의 step (sat / step) | TO-unchanged row 의 step (sat / step) | ① VLA sat step | ③ first = vla (row) | conv_on_to_row (row) | conv_carry (row) | unclipped (row) | mi (그 call 의 response sat row) | TO 가 안 바꿨는데 clip 된 call (전 call) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LI | V11AB-A | A | 1424 | 451 / 1424 (32%) | 159 / 1424 (11%) | 451 / 1414 | 0 / 10 | 0 | 0 | 1651 | 41 | 0 | – | 0 |
| LI | V11AB-B | B | 1509 | 97 / 1509 (6%) | 31 / 1509 (2%) | 97 / 1479 | 0 / 30 | 0 | 0 | 191 | 0 | 0 | – | 0 |
| LI | pb2 | C | 1362 | 881 / 1362 (65%) | 577 / 1362 (42%) | 881 / 1094 | 0 / 268 | 0 | 0 | 1821 | 68 | 0 | 51 (0) | 0 |
| LI | pb3 | C | 1833 | 1336 / 1833 (73%) | 996 / 1833 (54%) | 1336 / 1672 | 0 / 161 | 0 | 0 | 2892 | 97 | 2 | 27 (0) | 0 |
| LI | pb4 | C | 1925 | 1193 / 1925 (62%) | 863 / 1925 (45%) | 1193 / 1583 | 0 / 342 | 0 | 0 | 2662 | 69 | 2 | 53 (0) | 0 |
| LI | pb5 | C | 1916 | 1158 / 1916 (60%) | 806 / 1916 (42%) | 1158 / 1527 | 0 / 389 | 0 | 0 | 2536 | 87 | 1 | 61 (0) | 0 |
| LIBERO 10 | V11AB-A | A | 425 | 166 / 425 (39%) | 66 / 425 (16%) | 166 / 419 | 0 / 6 | 0 | 0 | 392 | 17 | 0 | – | 0 |
| LIBERO 10 | V11AB-B | B | 677 | 30 / 677 (4%) | 8 / 677 (1%) | 30 / 639 | 0 / 38 | 0 | 0 | 109 | 0 | 0 | – | 0 |
| LIBERO 10 | pb1 | B | 361 | 13 / 361 (4%) | 0 / 361 (0%) | 13 / 343 | 0 / 18 | 0 | 0 | 20 | 0 | 0 | – | 0 |
| LIBERO 10 | pb2 | C | 497 | 131 / 497 (26%) | 88 / 497 (18%) | 131 / 284 | 0 / 213 | 0 | 0 | 472 | 10 | 0 | 39 (0) | 0 |
| LIBERO 10 | pb3 | C | 721 | 431 / 721 (60%) | 300 / 721 (42%) | 431 / 533 | 0 / 188 | 1 | 1 | 881 | 27 | 0 | 36 (0) | 0 |
| LIBERO 10 | pb4 | C | 780 | 374 / 780 (48%) | 256 / 780 (33%) | 374 / 498 | 0 / 282 | 1 | 1 | 735 | 27 | 0 | 55 (0) | 0 |
| LIBERO 10 | pb5 | C | 538 | 197 / 538 (37%) | 135 / 538 (25%) | 197 / 273 | 0 / 265 | 0 | 0 | 376 | 15 | 0 | 50 (0) | 0 |

예) LI pb5: lifted env step 1916 중 실행 any-axis sat 1158, axes ≥ 3 동시 sat 878, 축별 sat 은 x 685 · y 856 · z 978 · rx 459 · ry 699 · rz 575. lifted call 377 중 conversion clip 이 있는 call 294, TO-changed call 302, 실행 ≠ response 인 call 70. latch 별 lifted step: latched 10 (sat 10), closing 958 (sat 782), held 913 (sat 333), placed 19 (sat 17), searching 16 (sat 16).

#### D2 — 역변환 C 동역학: clip 의 쌓임과 실현 비율

lifted 이면서 conversion 이 clip 한 call(`lifted_conv_sat`)의 중앙값(p10 ~ p90)이다. windup run 정의는 위 용어 절.

| 집합 | version | 변환 | clip 된 lifted call | max_abs_correction | \|D\*\| (mm) | D\*_z (mm) | lag_target (mm) | realised_frac_obs | realised_fraction_model | tracking_residual (mm) | windup run: ≥3 / ≥5 / 최장 (≥3 run 의 call 수) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LI | V11AB-A | A | 271 | 9.1 (2.8 ~ 22.9) | 28 (4 ~ 75) | 7.7 (-24.1 ~ 56.2) | 50 (13 ~ 179) | 0.5 (-0.7 ~ 5.8) | 0.3 (0.1 ~ 1.9) | – | 112 / 56 / 21 (676) |
| LI | V11AB-B | B | 102 | 1.0 (0.5 ~ 2.0) | 58 (16 ~ 160) | 30.9 (3.1 ~ 124.7) | 55 (12 ~ 126) | 0.2 (-0.0 ~ 0.4) | 0.1 (0.1 ~ 0.2) | – | 42 / 18 / 19 (235) |
| LI | pb2 | C | 215 | 10.5 (4.0 ~ 27.0) | 31 (10 ~ 93) | 15.2 (-2.5 ~ 67.2) | 35 (11 ~ 84) | 1.0 (-0.2 ~ 2.4) | 0.6 (0.3 ~ 1.0) | 5.8 (0.0 ~ 63.1) | 69 / 36 / 26 (404) |
| LI | pb3 | C | 328 | 11.1 (4.4 ~ 27.0) | 38 (12 ~ 92) | 22.2 (-3.8 ~ 69.2) | 36 (13 ~ 91) | 0.8 (-0.3 ~ 2.0) | 0.6 (0.3 ~ 1.0) | 10.1 (0.0 ~ 79.2) | 92 / 37 / 29 (504) |
| LI | pb4 | C | 310 | 10.2 (3.5 ~ 26.2) | 34 (11 ~ 108) | 20.4 (-3.4 ~ 66.5) | 37 (11 ~ 107) | 0.9 (-0.4 ~ 2.3) | 0.6 (0.3 ~ 1.0) | 8.6 (0.0 ~ 64.2) | 85 / 36 / 29 (467) |
| LI | pb5 | C | 294 | 11.7 (3.9 ~ 24.8) | 39 (10 ~ 105) | 18.7 (-1.9 ~ 66.4) | 39 (13 ~ 106) | 0.8 (-0.5 ~ 2.1) | 0.6 (0.3 ~ 1.0) | 9.9 (0.0 ~ 55.9) | 80 / 33 / 29 (447) |
| LIBERO 10 | V11AB-A | A | 82 | 6.9 (1.0 ~ 31.2) | 32 (3 ~ 103) | 17.4 (-2.5 ~ 69.5) | 33 (9 ~ 94) | 0.2 (-1.2 ~ 1.9) | 0.5 (0.1 ~ 1.5) | – | 39 / 27 / 16 (247) |
| LIBERO 10 | V11AB-B | B | 46 | 1.0 (0.5 ~ 1.7) | 39 (15 ~ 136) | 21.8 (3.7 ~ 112.3) | 91 (18 ~ 202) | 0.2 (-0.0 ~ 4.3) | 0.1 (0.1 ~ 0.1) | – | 9 / 6 / 8 (52) |
| LIBERO 10 | pb1 | B | 11 | 0.6 (0.5 ~ 1.4) | 38 (13 ~ 123) | 18.9 (3.8 ~ 61.5) | 33 (9 ~ 95) | 0.2 (0.1 ~ 0.3) | 0.1 (0.1 ~ 0.1) | – | 5 / 4 / 8 (30) |
| LIBERO 10 | pb2 | C | 55 | 16.1 (3.6 ~ 27.3) | 41 (15 ~ 65) | 19.3 (5.0 ~ 50.7) | 48 (11 ~ 341) | 1.4 (0.7 ~ 5.3) | 0.4 (0.1 ~ 1.0) | 19.1 (0.0 ~ 85.3) | 12 / 6 / 21 (87) |
| LIBERO 10 | pb3 | C | 103 | 9.9 (3.9 ~ 24.5) | 34 (14 ~ 94) | 21.3 (-7.3 ~ 74.3) | 39 (11 ~ 93) | 1.1 (-0.6 ~ 2.3) | 0.7 (0.4 ~ 1.1) | 13.4 (0.0 ~ 74.7) | 22 / 11 / 16 (119) |
| LIBERO 10 | pb4 | C | 91 | 9.2 (2.7 ~ 22.5) | 33 (14 ~ 102) | 19.6 (-8.3 ~ 79.5) | 36 (11 ~ 93) | 1.2 (-0.7 ~ 2.3) | 0.8 (0.3 ~ 1.1) | 7.2 (0.0 ~ 51.7) | 19 / 8 / 16 (102) |
| LIBERO 10 | pb5 | C | 47 | 9.5 (2.6 ~ 22.5) | 29 (15 ~ 81) | 21.9 (0.4 ~ 70.3) | 31 (10 ~ 56) | 1.3 (0.4 ~ 2.5) | 0.9 (0.4 ~ 1.3) | 5.9 (0.0 ~ 58.5) | 15 / 6 / 10 (77) |

realised_frac_obs 의 구간별 call 수 (clip 된 lifted call)와 TO 변화 크기 대 correction 의 Spearman 상관 (lifted TO-changed call):

| 집합 | version | 변환 | n | realised < 0 | 0 ~ 1.5 | > 1.5 | TO-changed lifted call | Spearman(max\|dEE\|, correction) | Spearman(D\*, correction) | clip 없는 TO-changed lifted call | TO-unchanged call (전체) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LI | V11AB-A | A | 271 | 97 | 86 | 88 | 275 | 0.798 | 0.69 | 4 | 71 |
| LI | V11AB-B | B | 102 | 14 | 88 | 0 | 292 | 0.83 | 0.797 | 190 | 55 |
| LI | pb2 | C | 215 | 38 | 116 | 61 | 216 | 0.844 | 0.764 | 1 | 373 |
| LI | pb3 | C | 328 | 64 | 194 | 70 | 335 | 0.808 | 0.749 | 7 | 379 |
| LI | pb4 | C | 310 | 47 | 173 | 90 | 317 | 0.809 | 0.782 | 7 | 414 |
| LI | pb5 | C | 294 | 44 | 182 | 68 | 302 | 0.84 | 0.792 | 8 | 372 |
| LIBERO 10 | V11AB-A | A | 82 | 29 | 42 | 11 | 82 | 0.899 | 0.857 | 0 | 19 |
| LIBERO 10 | V11AB-B | B | 46 | 6 | 30 | 10 | 126 | 0.801 | 0.803 | 80 | 42 |
| LIBERO 10 | pb1 | B | 11 | 1 | 10 | 0 | 68 | 0.961 | 0.96 | 57 | 21 |
| LIBERO 10 | pb2 | C | 54 | 1 | 27 | 26 | 59 | 0.846 | 0.694 | 4 | 186 |
| LIBERO 10 | pb3 | C | 103 | 17 | 55 | 31 | 107 | 0.796 | 0.764 | 4 | 179 |
| LIBERO 10 | pb4 | C | 91 | 14 | 45 | 32 | 97 | 0.74 | 0.726 | 6 | 198 |
| LIBERO 10 | pb5 | C | 47 | 3 | 28 | 16 | 55 | 0.813 | 0.784 | 8 | 196 |

LI pb5 의 TO-unchanged call 372 개는 `max_abs_correction` · D\* · tracking_residual 이 모두 0 이다. lifted 이면서 clip 되고 `cos(D\*, 직전 D\*) > 0.5` 인 call 은 195 / 294.

#### D3 — latch 와 reach check

`reach_mm` = latch evidence 가 잰, manipulated centroid 와 손 사이 거리(capture frame 의 robot state 기준). `reach_limit_mm` = 120. `reach_now_mm` = 현재 자세로 잰 값(기록용). `checks.reach` 는 `reach_mm` 이 한도 안인지의 검사다. 대표 사례(LI t2e4 pb5 call 30)에서 이 check 가 False 여서 note 가 "파지 미확인: reach" 였다(`SL5h.task.md`). `manipulated_state` = `occluded` 는 target 이 시야에서 가려진 상태로, 이 경우 geometry 는 마지막 관측 frame 에 고정된다(개선 기록 pb1 절).

| 집합 | version | 변환 | lifted call 의 latch: closing / held / 그 밖 | closing 이면서 lifted 인 call | 그 중 reach False | reach_mm 중앙값 | p90 | max | manipulated_state occluded / visible | 실패한 check (call 수) | reach False call (전체) | reach False call (lifted) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LI | V11AB-A | A | 182 / 92 / 3 | 182 | 181 / 182 (99%) | 306.16 | 414.38 | 514.71 | 140 / 42 | reach 181 · blocked 97 · dropped 76 · settled 11 | 366 | 181 |
| LI | V11AB-B | B | 233 / 55 / 10 | 233 | 229 / 233 (98%) | 502.26 | 587.80 | 892.74 | 210 / 23 | reach 229 · dropped 43 · blocked 12 · settled 5 | 487 | 229 |
| LI | pb2 | C | 155 / 96 / 17 | 155 | 134 / 155 (86%) | 227.77 | 519.35 | 677.29 | 120 / 35 | dropped 146 · reach 134 · blocked 55 · not_empty 5 · settled 3 | 282 | 134 |
| LI | pb3 | C | 220 / 84 / 62 | 220 | 119 / 220 (54%) | 124.44 | 739.66 | 1039.09 | 185 / 35 | dropped 148 · reach 119 · blocked 76 · settled 12 · not_empty 3 | 337 | 119 |
| LI | pb4 | C | 185 / 174 / 24 | 185 | 119 / 185 (64%) | 124.64 | 793.86 | 1039.09 | 150 / 35 | reach 119 · dropped 113 · blocked 41 · settled 12 · not_empty 3 | 337 | 119 |
| LI | pb5 | C | 189 / 179 / 9 | 189 | 179 / 189 (95%) | 172.62 | 788.48 | 1039.09 | 129 / 60 | reach 179 · blocked 53 · dropped 42 · settled 17 · not_empty 1 | 382 | 179 |
| LIBERO 10 | V11AB-A | A | 51 / 14 / 19 | 51 | 51 / 51 (100%) | 297.25 | 495.30 | 527.57 | 50 / 1 | reach 51 · blocked 20 | 117 | 51 |
| LIBERO 10 | V11AB-B | B | 106 / 28 / 0 | 106 | 104 / 106 (98%) | 307.60 | 359.00 | 518.43 | 101 / 5 | reach 104 · dropped 11 · settled 4 | 203 | 104 |
| LIBERO 10 | pb1 | B | 21 / 51 / 0 | 21 | 18 / 21 (86%) | 121.87 | 298.15 | 298.15 | 18 / 3 | reach 18 · dropped 11 · settled 3 | 87 | 18 |
| LIBERO 10 | pb2 | C | 40 / 57 / 1 | 40 | 33 / 40 (82%) | 131.93 | 131.95 | 131.95 | 38 / 2 | dropped 33 · reach 33 · blocked 6 · settled 1 | 61 | 33 |
| LIBERO 10 | pb3 | C | 89 / 51 / 3 | 89 | 61 / 89 (69%) | 123.73 | 133.13 | 133.22 | 84 / 5 | reach 61 · dropped 55 · blocked 9 · settled 4 | 91 | 61 |
| LIBERO 10 | pb4 | C | 88 / 61 / 3 | 88 | 61 / 88 (69%) | 123.73 | 133.13 | 133.22 | 84 / 4 | reach 61 · dropped 55 · blocked 9 · settled 3 | 91 | 61 |
| LIBERO 10 | pb5 | C | 37 / 61 / 7 | 37 | 33 / 37 (89%) | 133.07 | 133.15 | 133.22 | 36 / 1 | dropped 36 · reach 33 · settled 2 | 61 | 33 |

첫 lift 전 `closing` 구간의 실패 check (call 수):

| 집합 | version | closing·lift 전 call | reach | settled | blocked | dropped | not_empty | closed |
|---|---|---|---|---|---|---|---|---|
| LI | pb4 | 301 | 214 | 188 | 199 | 20 | 101 | 30 |
| LI | pb5 | 211 | 172 | 129 | 132 | 8 | 77 | 20 |
| LIBERO 10 | pb4 | 58 | 30 | 52 | 23 | 5 | 3 | 8 |
| LIBERO 10 | pb5 | 51 | 28 | 45 | 20 | 2 | 3 | 7 |

LI pb5 의 closing·lifted 189 call 은 `reach_mm` min 72.64 · 중앙값 172.62 · max 1039.09 mm 이고, `reach_state` 는 189 call 모두 capture frame 이다. `reach_now_mm` 중앙값 534.66 mm, `geometry_age_frames` 중앙값 5.0 (최대 41.0). lifted 로 센 obs 의 latch 분포: closing 963 · held 917 · placed 19 · latched 10 · searching 16.

#### D4 — closing 중 쥔 물체에 걸린 제약

latch 가 `closing`(또는 `held`) 이면서 실제 lift 중인 call 에서, 최적화된 call(MI gate 미통과)의 TO 행이 hand 를 어느 방향으로 밀었는지 센다. 행의 위/아래는 FK z 변화 ±0.5 mm 기준, `cos away` = EE 변화가 쥔 bowl 에서 멀어지는 방향인 행(`rows_cos_away_gt_0` / `rows_cos_n`). TO row 가 어느 ESDF 물체에 binding 인지는 calls log 에 count 만 남아 있어 직접 확인하지 못했고, MuJoCo row probe 와 centroid offset 을 대용으로 썼다.

**latch = closing**

| 집합 | version | 변환 | lifted call | 최적화된 call | MI 통과 call | 행 dz 위 / 아래 / flat | cos away > 0 행 | TO 행 dz 중앙값 (mm) | reference clearance < 0 call | MuJoCo: neither within 30 mm / probe call | centroid − bowl 중앙값 (mm) | 그 dz 중앙값 (mm) | zone row 가 켜진 call (rows) | held_exempt rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LI | V11AB-A | A | 182 | 181 | 0 | 1139 / 588 / 38 | 1392 / 1765 | 10.9 | – | 111 / 181 | 279 | -75 | 0 (0) | 0 |
| LI | V11AB-B | B | 233 | 227 | 0 | 2049 / 143 / 38 | 2028 / 2230 | 40.0 | – | 203 / 227 | 476 | -119 | 0 (0) | 0 |
| LI | pb2 | C | 155 | 142 | 13 | 1198 / 149 / 18 | 1243 / 1365 | 20.7 | 116 | 141 / 142 | 494 | -297 | 0 (0) | 0 |
| LI | pb3 | C | 220 | 204 | 15 | 1695 / 222 / 48 | 1792 / 1965 | 26.4 | 167 | 203 / 204 | 513 | -392 | 0 (0) | 0 |
| LI | pb4 | C | 185 | 170 | 15 | 1374 / 210 / 46 | 1470 / 1630 | 22.1 | 135 | 169 / 170 | 575 | -439 | 0 (0) | 0 |
| LI | pb5 | C | 189 | 180 | 9 | 1431 / 243 / 51 | 1540 / 1725 | 23.5 | 150 | 175 / 180 | 477 | -369 | 7 (160) | 0 |
| LIBERO 10 | V11AB-A | A | 51 | 50 | 0 | 410 / 58 / 2 | 422 / 470 | 22.4 | – | 48 / 50 | 237 | -87 | 0 (0) | 0 |
| LIBERO 10 | V11AB-B | B | 106 | 102 | 0 | 830 / 122 / 13 | 880 / 965 | 27.8 | – | 89 / 102 | 265 | -48 | 0 (0) | 0 |
| LIBERO 10 | pb1 | B | 21 | 21 | 0 | 162 / 15 / 8 | 174 / 185 | 19.3 | – | 21 / 21 | 312 | -117 | 0 (0) | 0 |
| LIBERO 10 | pb2 | C | 40 | 36 | 4 | 195 / 136 / 24 | 175 / 355 | 2.8 | 35 | 12 / 36 | 403 | -160 | 0 (0) | 0 |
| LIBERO 10 | pb3 | C | 89 | 75 | 14 | 617 / 100 / 8 | 647 / 725 | 23.7 | 67 | 75 / 75 | 444 | -280 | 0 (0) | 0 |
| LIBERO 10 | pb4 | C | 88 | 75 | 13 | 617 / 100 / 8 | 647 / 725 | 23.7 | 67 | 75 / 75 | 444 | -280 | 0 (0) | 0 |
| LIBERO 10 | pb5 | C | 37 | 29 | 8 | 234 / 50 / 6 | 232 / 290 | 25.4 | 26 | 29 / 29 | 524 | -349 | 0 (0) | 0 |

**latch = held**

| 집합 | version | 변환 | lifted call | 최적화된 call | MI 통과 call | 행 dz 위 / 아래 / flat | cos away > 0 행 | TO 행 dz 중앙값 (mm) | reference clearance < 0 call | MuJoCo: neither within 30 mm / probe call | centroid − bowl 중앙값 (mm) | 그 dz 중앙값 (mm) | zone row 가 켜진 call (rows) | held_exempt rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LI | V11AB-A | A | 92 | 92 | 0 | 422 / 226 / 77 | 402 / 725 | 2.0 | – | 74 / 91 | 403 | -69 | 0 (0) | 0 |
| LI | V11AB-B | B | 55 | 55 | 0 | 358 / 91 / 36 | 437 / 485 | 3.9 | – | 46 / 55 | 263 | -64 | 0 (0) | 0 |
| LI | pb2 | C | 96 | 59 | 36 | 386 / 150 / 19 | 490 / 555 | 5.4 | 45 | 51 / 59 | 480 | -151 | 0 (0) | 0 |
| LI | pb3 | C | 84 | 71 | 10 | 559 / 106 / 10 | 545 / 675 | 18.4 | 56 | 48 / 71 | 339 | -185 | 0 (0) | 0 |
| LI | pb4 | C | 174 | 125 | 36 | 1029 / 154 / 27 | 949 / 1210 | 19.8 | 98 | 77 / 125 | 343 | -159 | 0 (0) | 1332 |
| LI | pb5 | C | 179 | 113 | 52 | 957 / 93 / 25 | 950 / 1075 | 20.9 | 89 | 78 / 113 | 319 | -159 | 0 (0) | 835 |
| LIBERO 10 | V11AB-A | A | 14 | 13 | 0 | 54 / 17 / 9 | 50 / 80 | 2.1 | – | 11 / 13 | 122 | -38 | 0 (0) | 0 |
| LIBERO 10 | V11AB-B | B | 28 | 24 | 0 | 109 / 6 / 60 | 137 / 175 | 1.5 | – | 22 / 24 | 221 | -71 | 0 (0) | 0 |
| LIBERO 10 | pb1 | B | 51 | 47 | 0 | 214 / 19 / 82 | 242 / 315 | 1.5 | – | 45 / 47 | 168 | -70 | 0 (0) | 0 |
| LIBERO 10 | pb2 | C | 57 | 23 | 34 | 165 / 39 / 11 | 142 / 215 | 16.7 | 19 | 22 / 23 | 482 | -111 | 0 (0) | 0 |
| LIBERO 10 | pb3 | C | 51 | 30 | 21 | 241 / 32 / 7 | 261 / 280 | 22.3 | 25 | 29 / 30 | 254 | -111 | 0 (0) | 0 |
| LIBERO 10 | pb4 | C | 61 | 20 | 41 | 148 / 14 / 18 | 149 / 180 | 13.0 | 14 | 13 / 20 | 388 | -80 | 0 (0) | 65 |
| LIBERO 10 | pb5 | C | 61 | 21 | 40 | 174 / 11 / 5 | 179 / 190 | 17.7 | 15 | 14 / 21 | 149 | -9 | 0 (0) | 58 |

LI pb5 closing: 최적화된 180 call 의 reference clearance 는 중앙값 -14.8 mm (min -62.2, max 9.0), 그 5 행 자세에서 gripper 의 target 최소 거리 중앙값(`grip_target_min_mm`) -30.4 mm 이고 probe call 모두(180) 두 손가락이 target 에 닿아 있다(`probe_grip_touching_target` 180). table 이 30 mm 안에 있는 call 0, target 이 아닌 물체가 30 mm 안에 있는 call 5, 둘 다 아닌 call 175. AG3S centroid 와 실제 bowl 의 차는 중앙값 477 mm (p90 796, max 1120), z 성분 중앙값 -369 mm.

#### D5 — version · 변환별 서명

같은 서명(over-lift 수, lifted step 의 실행 포화율, `checks.reach = False` 인 closing·lifted call)을 version 별로 센다. 아래 표의 VLA-only 는 SL5f.

| 집합 | version | 변환 | official | over-lift episode / lifted episode | max rise 중앙값 (mm) | lifted 실행 any-axis sat | lifted 실행 z ≥ 0.999 | closing·lifted 중 reach False | D\*_z 중앙값 (mm, clip 된 lifted call) | realised_obs 중앙값 | realised_model 중앙값 | max_abs_correction 중앙값 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LI | vla_only | – | 11 / 20 | 0 / 14 | 132.7 | 0 / 717 (0%) | 0 / 717 (0%) | – | – | – | – | – |
| LI | V11AB-A | A | 3 / 20 | 3 / 13 | 150.6 | 451 / 1424 (32%) | 159 / 1424 (11%) | 181 / 182 (99%) | 7.7 | 0.48 | 0.35 | 9.03 |
| LI | V11AB-B | B | 3 / 20 | 0 / 13 | 157.7 | 97 / 1509 (6%) | 31 / 1509 (2%) | 229 / 233 (98%) | 30.9 | 0.21 | 0.14 | 0.68 |
| LI | pb2 | C | 2 / 20 | 7 / 10 | 111.6 | 881 / 1362 (65%) | 577 / 1362 (42%) | 134 / 155 (86%) | 15.2 | 1.03 | 0.60 | 7.77 |
| LI | pb3 | C | 0 / 20 | 9 / 10 | 90.5 | 1336 / 1833 (73%) | 996 / 1833 (54%) | 119 / 220 (54%) | 22.2 | 0.82 | 0.56 | 10.21 |
| LI | pb4 | C | 0 / 20 | 9 / 10 | 128.7 | 1193 / 1925 (62%) | 863 / 1925 (45%) | 119 / 185 (64%) | 20.4 | 0.91 | 0.54 | 8.62 |
| LI | pb5 | C | 1 / 20 | 10 / 12 | 226.9 | 1158 / 1916 (60%) | 806 / 1916 (42%) | 179 / 189 (95%) | 18.7 | 0.83 | 0.55 | 9.06 |
| LIBERO 10 | vla_only | – | 9 / 10 | 0 / 10 | 147.3 | 0 / 348 (0%) | 0 / 348 (0%) | – | – | – | – | – |
| LIBERO 10 | V11AB-A | A | 2 / 10 | 0 / 7 | 131.7 | 166 / 425 (39%) | 66 / 425 (16%) | 51 / 51 (100%) | 17.3 | 0.20 | 0.45 | 6.85 |
| LIBERO 10 | V11AB-B | B | 3 / 10 | 0 / 9 | 140.4 | 30 / 677 (4%) | 8 / 677 (1%) | 104 / 106 (98%) | 21.8 | 0.22 | 0.13 | 0.57 |
| LIBERO 10 | pb1 | B | 7 / 10 | 0 / 9 | 137.1 | 13 / 361 (4%) | 0 / 361 (0%) | 18 / 21 (86%) | 18.9 | 0.16 | 0.13 | 0.04 |
| LIBERO 10 | pb2 | C | 6 / 10 | 1 / 8 | 138.1 | 131 / 497 (26%) | 88 / 497 (18%) | 33 / 40 (82%) | 19.3 | 1.36 | 0.40 | 3.57 |
| LIBERO 10 | pb3 | C | 4 / 10 | 4 / 8 | 184.5 | 431 / 721 (60%) | 300 / 721 (42%) | 61 / 89 (69%) | 21.3 | 1.07 | 0.74 | 7.14 |
| LIBERO 10 | pb4 | C | 3 / 10 | 4 / 8 | 184.5 | 374 / 780 (48%) | 256 / 780 (33%) | 61 / 88 (69%) | 19.6 | 1.21 | 0.73 | 4.02 |
| LIBERO 10 | pb5 | C | 5 / 10 | 1 / 8 | 154.4 | 197 / 538 (37%) | 135 / 538 (25%) | 33 / 37 (89%) | 21.9 | 1.28 | 0.99 | 0.18 |

over-lift episode 번호 (LI): A t0e1, t2e1, t2e2; B 없음; pb2 t0e3, t1e1, t1e2, t1e4, t2e0, t2e1, t2e3; pb3 t0e0, t0e1, t0e3, t1e1, t1e2, t1e4, t2e0, t2e1, t2e4; pb4 t0e0, t0e1, t0e3, t1e1, t1e2, t1e4, t2e0, t2e1, t2e4; pb5 t0e0, t0e1, t0e3, t1e2, t1e3, t1e4, t2e0, t2e1, t2e2, t2e4. LIBERO 10: A 없음; B 없음; pb1 없음; pb2 t1e0; pb3 t0e0, t1e0, t5e0, t8e0; pb4 t0e0, t1e0, t5e0, t8e0; pb5 t5e0.

한계: 비교 run 은 code · 설정이 version 마다 다르다 (A · B · pb1 은 MI gate · arms · pb4/pb5 규칙이 없고, SL5c · SL5d · SL5e · SL5g 는 서로 다른 commit). 그래서 위 표는 같은 서명이 어느 run 에 나타나는지를 보여 줄 뿐 한 변수만 바꾼 비교가 아니다. SL5h-G 가 같은 pb5 규칙에서 변환만 바꾼 run 을 추가했다.

#### D6 — pb5 성공 episode 와의 대조

| 군 | ep | official | max rise (mm) | eef z 최고 (m) | VLA-only eef z 최고 (m) | 첫 closing call 의 obs | 첫 reach 실패 call 의 obs | 첫 held call 의 obs | 첫 lifted obs | lifted obs 의 latch | closing·lifted call | closing·lifted 의 실패 check | lifted step: z sat / any sat / 전체 | lifted call: clip / TO-changed / 전체 | closing·lifted 행 dz 위 / 아래 | held·lifted 행 dz 위 / 아래 | z 기여 합 TO-changed (mm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LIBERO 10 pb5 성공 | t0e0 | True | 96.6 | 1.183 | 1.183 | 50 | 50 | 85 | 96 | closing 4, held 55 | 0 | – | 16 / 40 / 59 | 9 / 9 / 11 | 0 / 0 | 80 / 0 | 433.3 |
| LIBERO 10 pb5 성공 | t1e0 | True | 179.0 | 1.188 | 1.188 | 65 | 65 | 80 | 79 | closing 1, held 41 | 0 | – | 0 / 0 / 42 | 0 / 1 / 9 | 0 / 0 | 10 / 0 | 29.2 |
| LIBERO 10 pb5 성공 | t2e0 | True | 149.6 | 1.187 | 1.187 | 80 | – | 100 | 97 | closing 3, held 33 | 0 | – | 0 / 0 / 36 | 0 / 0 / 7 | 0 / 0 | 0 / 0 | 0 |
| LIBERO 10 pb5 성공 | t3e0 | True | 127.3 | 1.176 | 1.176 | 55 | – | 70 | 69 | closing 1, held 23 | 0 | – | 0 / 0 / 24 | 0 / 1 / 5 | 0 / 0 | 9 / 0 | 8.0 |
| LIBERO 10 pb5 성공 | t6e0 | True | 159.2 | 1.18 | 1.18 | 65 | – | 85 | 83 | closing 2, held 43 | 0 | – | 1 / 3 / 45 | 1 / 2 / 9 | 0 / 0 | 20 / 0 | 29.2 |
| LIBERO 10 pb5 실패 | t4e0 | False | 0.0 | 1.239 | 1.269 | – | – | – | – | – | 0 | – | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 | 0 / 0 | 0 |
| LIBERO 10 pb5 실패 | t5e0 | False | 689.0 | 1.609 | 1.167 | 60 | 65 | – | 59 | latched 1, closing 171 | 34 | dropped 34, reach 33 | 86 / 105 / 171 | 25 / 26 / 34 | 224 / 36 | 0 / 0 | 538.4 |
| LIBERO 10 pb5 실패 | t7e0 | False | 243.4 | 1.227 | 1.158 | 40 | 40 | 60 | 71 | held 74, latched 8, closing 15 | 3 | settled 2, dropped 2 | 18 / 28 / 97 | 6 / 7 / 18 | 10 / 14 | 17 / 7 | 103.7 |
| LIBERO 10 pb5 실패 | t8e0 | False | 237.3 | 1.202 | 1.181 | 70 | – | 85 | 82 | closing 3, held 32, placed 30 | 0 | – | 14 / 21 / 64 | 6 / 9 / 12 | 0 / 0 | 38 / 4 | 189.7 |
| LIBERO 10 pb5 실패 | t9e0 | False | 18.0 | 1.285 | 1.281 | 120 | 120 | – | – | – | 0 | – | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 | 0 / 0 | 0 |
| LI pb5 성공 | t1e1 | True | 157.4 | 1.176 | 1.176 | 50 | 50 | 70 | 67 | closing 3, held 35 | 0 | – | 0 / 0 / 38 | 1 / 2 / 7 | 0 / 0 | 9 / 4 | 3.7 |

성공 episode 5 건(LIBERO 10 t0e0 · t1e0 · t2e0 · t3e0 · t6e0)의 closing 이면서 lifted 인 call 은 모두 0 이다. 같은 지표에서 LI t1e1(LI pb5 의 유일한 성공)도 0 이다. LIBERO 10 실패 episode 중 t5e0 은 lifted obs 가 latched 1 · closing 171 이고 closing·lifted 34 call 중 reach 33 · dropped 34 call 이 실패 check 다.

#### 대표 사례 LI t2e4 pb5 (over-lift 531.3 mm)

`reach_mm` 은 capture frame 21 에 고정된 채 34 call 동안 123.22 ~ 123.94 mm 였다 (limit 120). lead 가 확인한 "call 30" 은 calls jsonl 의 30 번째 줄(아래 표의 k 29, obs 165)이다: reach_mm 123.46, 포화 row [0, 1], max_abs_correction 1.391. 다음 call(k 30, obs 170)은 reach_mm 123.47, 포화 row 9 개, max_abs_correction 5.521 이다. lifted 로 센 obs 의 latch 는 모두 `closing`(178 obs) 이다.

![SL5h LI t2e4 pb5 scene](figures/sl5h-scene-t2e4-pb5.png)

| call k | obs | latch | lifted | rise (mm) | eef z (m) | MI 통과 | reference clearance (mm) | ① policy max\|a\| | ② TO max\|dEE\| (mm) | ③ max_abs_correction | ③ clip 된 row | ④ response max\|a\| | ⑤ executed max\|a\| | reach_mm | manipulated_state | lag_target (mm) | realised_obs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 20 | searching | F | 0.0 | 1.185 | T | 113.1 | 0.86 | 0.0 | 0.00 | 0 | 0.86 | 0.54 | – | – | 10.8 | – |
| 4 | 40 | latched | F | 0.0 | 1.043 | F | -4.8 | 0.92 | 14.8 | 3.11 | 8 | 1.00 | 1.00 | 164.9 | visible | 15.1 | -0.07 |
| 22 | 130 | closing | T | 72.3 | 1.058 | F | 6.8 | 0.80 | 2.7 | 0.86 | 0 | 0.96 | 0.79 | 123.2 | visible | 5.3 | 2.49 |
| 29 | 165 | closing | T | 256.5 | 1.245 | F | 3.9 | 0.63 | 14.3 | 1.39 | 2 | 1.00 | 1.00 | 123.5 | occluded | 5.4 | 1.29 |
| 30 | 170 | closing | T | 257.2 | 1.245 | F | 1.3 | 0.72 | 44.2 | 5.52 | 9 | 1.00 | 1.00 | 123.5 | occluded | 11.1 | 1.48 |
| 46 | 250 | closing | T | 523.5 | 1.494 | T | 19.0 | 0.71 | 0.0 | 0.00 | 0 | 0.71 | 0.64 | 123.4 | occluded | 19.0 | – |
| 55 | 295 | closing | T | 483.1 | 1.401 | F | -14.1 | 0.71 | 45.0 | 8.60 | 10 | 1.00 | 1.00 | 123.9 | occluded | 112.1 | 2.71 |

같은 episode 의 call 별 전체는 `numbers.case_LI_t2e4_pb5.per_call` 56 행에 있다. 각 행은 이 표와 같은 열을 가진다 (obs 20 부터 5 obs 간격). 위 표는 첫 call(k 0), 처음 clip 된 call(k 4), 처음 lifted 인 call(k 22), 포화 시작 call(k 29, 30), rise 최고 call(k 46), 마지막 부근(k 55)만 옮겼다.

#### 보조 숫자: z 기여와 VLA 원본 z 부호

| 집합 | version | 변환 | z 기여 합: TO-changed lifted call (n / mm) | TO-unchanged lifted call (n / mm) | 실행 z ≥ 0.999 step 의 VLA 원본 z 부호 pos / neg / near0 | VLA-only 최고 eef z 를 넘은 step | 그 중 실행 z 포화 step | 그 중 VLA 원본이 포화한 step |
|---|---|---|---|---|---|---|---|---|
| LI | V11AB-A | A | 275 / 6346.9 | 2 / 17.4 | 106 / 42 / 11 | 127 | 15 | 0 |
| LI | V11AB-B | B | 291 / 2572.1 | 6 / 60.8 | 21 / 9 / 1 | 130 | 4 | 0 |
| LI | pb2 | C | 216 / 2458.1 | 52 / -271.2 | 419 / 114 / 44 | 827 | 401 | 0 |
| LI | pb3 | C | 335 / 4777.2 | 31 / -168.1 | 658 / 242 / 96 | 1346 | 803 | 0 |
| LI | pb4 | C | 317 / 6551.1 | 66 / -151.5 | 599 / 207 / 57 | 1074 | 666 | 0 |
| LI | pb5 | C | 302 / 3662.2 | 75 / -872.1 | 506 / 229 / 71 | 1019 | 581 | 0 |
| LIBERO 10 | V11AB-A | A | 82 / 81.2 | 2 / -18.6 | 35 / 23 / 8 | 12 | 1 | 0 |
| LIBERO 10 | V11AB-B | B | 126 / 3477.9 | 7 / 112.7 | 6 / 2 / 0 | 23 | 0 | 0 |
| LIBERO 10 | pb1 | B | 68 / 33.1 | 3 / 41.7 | 0 / 0 / 0 | 23 | 0 | 0 |
| LIBERO 10 | pb2 | C | 58 / 7743.9 | 39 / -293.3 | 38 / 34 / 16 | 66 | 48 | 0 |
| LIBERO 10 | pb3 | C | 107 / 1355.0 | 36 / -177.0 | 143 / 131 / 26 | 306 | 179 | 1 |
| LIBERO 10 | pb4 | C | 97 / 2166.7 | 55 / -248.3 | 114 / 121 / 21 | 289 | 169 | 1 |
| LIBERO 10 | pb5 | C | 55 / 1331.5 | 50 / -280.9 | 44 / 69 / 22 | 164 | 85 | 0 |

z 기여 = lifted call 마다 `eef_obs_z(o+5) − FK_z(joint_reference[k,4])` 의 합이다.

#### 가설 H1 ~ H4 별 숫자 (판정 없음)

verify.json `hypotheses_numbers` 가 각 가설에 붙인 `support` · `against` 구분을 그대로 옮긴다. 이 구분은 verifier 의 분류이고 이 로그의 판정이 아니다.

**H1. 포화는 역변환 C 의 windup 에서 생긴다**

support:

- lifted 로 센 포화 response row 의 first stage (LI pb4 / pb5): conv_on_to_row 2662 / 2536, conv_carry 69 / 87, unclipped 2 / 1
- lifted 실행 any-axis sat 비율: LI A / B / pb2-C / pb5-C = 0.317 / 0.064 / 0.647 / 0.604; LIBERO 10 B / pb1-B / pb2-C / pb5-C = 0.044 / 0.036 / 0.264 / 0.366
- over-lift episode 수: LI A / B / pb2 / pb3 / pb4 / pb5 = 3 / 0 / 7 / 9 / 9 / 10; LIBERO 10 B / pb1 / pb2 / pb3 / pb4 / pb5 = 0 / 0 / 1 / 4 / 4 / 1
- realised_frac_obs (LI pb5, clip 된 lifted call 294): > 1.5 인 call 68, < 0 인 call 44, 0 ~ 1.5 인 call 182
- windup run (LI pb5): ≥ 3 call 80 개, ≥ 5 call 33 개, 최장 29 call, ≥ 3 run 안의 call 447

against:

- 역변환 상태는 call 마다 `d^`, `D^` = 0 에서 시작한다 (code). conv_carry row / 포화 response row (lifted, LI pb5) = 87 / 2624
- TO-unchanged row 에서 포화한 lifted step: 13 개 (version × 집합) 모두 0; TO 가 안 바꿨는데 clip 된 call: 모두 0
- Spearman(max\|dEE\|, max_abs_correction) lifted TO-changed call: LI|A 0.798, LI|B 0.83, LI|pb2 0.844, LI|pb3 0.808, LI|pb4 0.809, LI|pb5 0.84, L10|A 0.899, L10|B 0.801, L10|pb1 0.961, L10|pb2 0.846, L10|pb3 0.796, L10|pb4 0.74, L10|pb5 0.813

**H2. 포화는 TO 수정에서 생긴다**

support:

| 집합 | version | 변환 | TO-changed row 의 lifted step (sat / step) | TO-unchanged row 의 lifted step (sat / step) | MI 통과 lifted call 의 response sat row | D\*_z 중앙값 (mm) | z 기여 합 TO-changed / unchanged (mm) |
|---|---|---|---|---|---|---|---|
| LI | V11AB-A | A | 451 / 1414 | 0 / 10 | – | 7.7 | 6346.9 / 17.4 |
| LI | V11AB-B | B | 97 / 1479 | 0 / 30 | – | 30.9 | 2572.1 / 60.8 |
| LI | pb2 | C | 881 / 1094 | 0 / 268 | 0 | 15.2 | 2458.1 / -271.2 |
| LI | pb3 | C | 1336 / 1672 | 0 / 161 | 0 | 22.2 | 4777.2 / -168.1 |
| LI | pb4 | C | 1193 / 1583 | 0 / 342 | 0 | 20.4 | 6551.1 / -151.5 |
| LI | pb5 | C | 1158 / 1527 | 0 / 389 | 0 | 18.7 | 3662.2 / -872.1 |
| LIBERO 10 | V11AB-A | A | 166 / 419 | 0 / 6 | – | 17.3 | 81.2 / -18.6 |
| LIBERO 10 | V11AB-B | B | 30 / 639 | 0 / 38 | – | 21.8 | 3477.9 / 112.7 |
| LIBERO 10 | pb1 | B | 13 / 343 | 0 / 18 | – | 18.9 | 33.1 / 41.7 |
| LIBERO 10 | pb2 | C | 131 / 284 | 0 / 213 | 0 | 19.3 | 7743.9 / -293.3 |
| LIBERO 10 | pb3 | C | 431 / 533 | 0 / 188 | 0 | 21.3 | 1355.0 / -177.0 |
| LIBERO 10 | pb4 | C | 374 / 498 | 0 / 282 | 0 | 19.6 | 2166.7 / -248.3 |
| LIBERO 10 | pb5 | C | 197 / 273 | 0 / 265 | 0 | 21.9 | 1331.5 / -280.9 |

- TO 행 dz 위 / 아래 (LI pb5, closing·lifted): 1431 / 243

against:

- LI B(변환 B): TO-changed lifted step 중 포화 / step = 97 / 1479, D\*_z 중앙값 30.9 mm, over-lift 0

**H3. latch 가 reach check 에서 막혀 held 로 못 가고, 쥔 bowl 이 제약으로 남아 hand 를 밀어 올린다**

support (표는 closing·lifted call 기준):

| 집합 | version | 변환 | reach False / closing·lifted call | ref clearance < 0 / 최적화 / MuJoCo neither within 30 mm (call) | TO 행 dz 위 / 아래 | centroid − bowl 중앙값 mm (dz) |
|---|---|---|---|---|---|---|
| LI | V11AB-A | A | 181 / 182 | 0 / 181 / 111 | 1139 / 588 | 279 (-75) |
| LI | V11AB-B | B | 229 / 233 | 0 / 227 / 203 | 2049 / 143 | 476 (-119) |
| LI | pb2 | C | 134 / 155 | 116 / 142 / 141 | 1198 / 149 | 494 (-297) |
| LI | pb3 | C | 119 / 220 | 167 / 204 / 203 | 1695 / 222 | 513 (-392) |
| LI | pb4 | C | 119 / 185 | 135 / 170 / 169 | 1374 / 210 | 575 (-439) |
| LI | pb5 | C | 179 / 189 | 150 / 180 / 175 | 1431 / 243 | 477 (-369) |
| LIBERO 10 | V11AB-A | A | 51 / 51 | 0 / 50 / 48 | 410 / 58 | 237 (-87) |
| LIBERO 10 | V11AB-B | B | 104 / 106 | 0 / 102 / 89 | 830 / 122 | 265 (-48) |
| LIBERO 10 | pb1 | B | 18 / 21 | 0 / 21 / 21 | 162 / 15 | 312 (-117) |
| LIBERO 10 | pb2 | C | 33 / 40 | 35 / 36 / 12 | 195 / 136 | 403 (-160) |
| LIBERO 10 | pb3 | C | 61 / 89 | 67 / 75 / 75 | 617 / 100 | 444 (-280) |
| LIBERO 10 | pb4 | C | 61 / 88 | 67 / 75 / 75 | 617 / 100 | 444 (-280) |
| LIBERO 10 | pb5 | C | 33 / 37 | 26 / 29 / 29 | 234 / 50 | 524 (-349) |

- lifted call 의 latch 분포 (call 수): LI|pb4 held 174, placed 18, closing 185, latched 3, searching 3; LI|pb5 closing 189, held 179, searching 3, latched 2, placed 4; LIBERO 10|pb4 held 61, closing 88, latched 3; LIBERO 10|pb5 held 61, closing 37, latched 2, placed 5
- closing·lifted call 중 zone_rows > 0 (pb5): LI 7, LIBERO 10 0
- LIBERO 10 pb5 성공 episode(t0e0 · t1e0 · t2e0 · t3e0 · t6e0)의 closing·lifted call: t0e0 0, t1e0 0, t2e0 0, t3e0 0, t6e0 0
- LI pb4 · pb3 · pb5 에서 closing·lifted 중 reach 외 실패 check: pb3 reach 119, settled 12, blocked 76, dropped 148, not_empty 3; pb4 reach 119, settled 12, blocked 41, dropped 113, not_empty 3; pb5 settled 17, reach 179, blocked 53, dropped 42, not_empty 1

against:

| 집합 | version | 변환 | held·lifted 의 TO 행 dz 위 / 아래 | ref clearance < 0 call | 최적화된 call |
|---|---|---|---|---|---|
| LI | V11AB-A | A | 422 / 226 | 0 | 92 |
| LI | V11AB-B | B | 358 / 91 | 0 | 55 |
| LI | pb2 | C | 386 / 150 | 45 | 59 |
| LI | pb3 | C | 559 / 106 | 56 | 71 |
| LI | pb4 | C | 1029 / 154 | 98 | 125 |
| LI | pb5 | C | 957 / 93 | 89 | 113 |
| LIBERO 10 | V11AB-A | A | 54 / 17 | 0 | 13 |
| LIBERO 10 | V11AB-B | B | 109 / 6 | 0 | 24 |
| LIBERO 10 | pb1 | B | 214 / 19 | 0 | 47 |
| LIBERO 10 | pb2 | C | 165 / 39 | 19 | 23 |
| LIBERO 10 | pb3 | C | 241 / 32 | 25 | 30 |
| LIBERO 10 | pb4 | C | 148 / 14 | 14 | 20 |
| LIBERO 10 | pb5 | C | 174 / 11 | 15 | 21 |

- over-lift 가 없는 run 의 closing·lifted reach False: LI B 229 / 233 (over-lift 0), LIBERO 10 B 104 / 106 (over-lift 0), LIBERO 10 pb1 18 / 21 (over-lift 0)

**H4. VLA reference 자체가 OOD 상태에서 포화한다**

support (lifted 실행 z ≥ 0.999 step 의 VLA 원본 z 부호, VLA-only 최고 eef z 를 넘은 step) 는 위 "보조 숫자" 표의 마지막 네 열과 같은 값이다.

against:

| 집합 | version | 변환 | VLA 원본(①)이 포화한 lifted step / lifted step | first stage 가 vla 인 포화 response row (lifted) |
|---|---|---|---|---|
| LI | V11AB-A | A | 0 / 1424 | 0 |
| LI | V11AB-B | B | 0 / 1509 | 0 |
| LI | pb2 | C | 0 / 1362 | 0 |
| LI | pb3 | C | 0 / 1833 | 0 |
| LI | pb4 | C | 0 / 1925 | 0 |
| LI | pb5 | C | 0 / 1916 | 0 |
| LIBERO 10 | V11AB-A | A | 0 / 425 | 0 |
| LIBERO 10 | V11AB-B | B | 0 / 677 | 0 |
| LIBERO 10 | pb1 | B | 0 / 361 | 0 |
| LIBERO 10 | pb2 | C | 0 / 497 | 0 |
| LIBERO 10 | pb3 | C | 1 / 721 | 1 |
| LIBERO 10 | pb4 | C | 1 / 780 | 1 |
| LIBERO 10 | pb5 | C | 0 / 538 | 0 |

- VLA-only 의 lifted step 중 VLA 원본 포화: LI 0 / 717, LIBERO 10 0 / 348

#### 측정하지 않은 것

- TO row 의 constraint 대상(어느 ESDF · attached 물체에 binding 인지): calls log 는 `zone_rows` · `held_exempt_rows` count 만 남긴다. server ESDF 가 cuRobo(GPU)라 offline replay 를 하지 않았다. D4 는 MuJoCo row probe 와 centroid offset 으로 대신했다.
- reach_mm 을 잰 capture frame 의 robot state: 기록되지 않았다. `reach_mm` · `reach_now_mm` · `reach_state` 만 있다.
- 역변환 C 의 row 별 response model 예측(p_hat): `predicted_ee_base` 5 행 이상은 없다. realised 는 관측 eef 로 계산했다.
- GPU counterfactual(pb5-B · pb5-A, 병렬 bit 재현): 이 STEP 에서 실행하지 않았다 (다음 절 SL5h-G).
- cuRobo regression baseline: GPU 금지.
- SL5f 초기 pb4_init · pb5_init run 은 D5 에 넣지 않았다.

**판정 대기 (사용자).** over-lift · 포화의 출처에 대한 판정과 다음 수정 방향은 사용자 판정 전이다. 위는 가설별 숫자만 남긴 것이다.

### SL5h-G — pb5 × 변환 A / B / C counterfactual 과 같은 checkout 안 병렬 GPU 실행 (2026-10-19)

**기록 시각** 병렬 launcher 구현 commit `e2b5deaa` 2026-10-19 18:22 UTC (2026-10-20 03:22 KST). rollout 은 2026-10-19 18:45 ~ 19:50 UTC = **2026-10-20 03:45 ~ 04:50 KST**, 분석 verify.json `date` = 2026-10-20 KST. **수치 출처** [`handoff/SL5h-G.verify.json`](handoff/SL5h-G.verify.json), 지시서 [`handoff/SL5h-G.task.md`](handoff/SL5h-G.task.md) · [`handoff/SL5h-launcher.task.md`](handoff/SL5h-launcher.task.md), 구현 기록 [`handoff/SL5h-launcher.impl.md`](handoff/SL5h-launcher.impl.md). **code** run 은 `743a1e76`(실행 경로는 SL5g 의 `a8ad92ef` 와 같고, 바뀐 것은 launcher 의 병렬 option 과 docs 뿐), 분석은 HEAD `4634ac11` 에서 CPU 만 썼다 (`CUDA_VISIBLE_DEVICES=`, code 수정 없음). legacy 기준선은 일치했다 (violated start 14/15, has_target 15/15, frame0 clearance -29.031 mm). cuRobo 기준선은 GPU 금지라 실행하지 않았다.

이 절은 두 가지를 기록한다. (1) 같은 checkout 안에서 GPU run 을 동시에 둘까지 돌리는 launcher option 과 그 결과가 직렬 실행과 bit 단위로 같았는지. (2) SL5h 가 남긴 질문, 즉 포화 · over-lift 가 변환(역변환) 에서 나오는가를 **같은 pb5 규칙에서 변환만 A / B / C 로 바꾼** run 으로 잰 숫자. 판정은 쓰지 않는다.

#### 병렬 실행 option (commit `e2b5deaa`, `benchmark/runners/run_parallel_gpu.sh`)

**무엇이 바뀌었나.** 사용자 지시(2026-10-19)는 GPU(H200 NVL 140 GB)를 적절히 활용하라는 것이었다. 그동안 `launch_libero_paired.py` 의 `run_mode` 가 mode 마다 공용 GPU lock (`/tmp/ag3s-vla-gpu-validation.lock`)을 직접 `flock` 하고 다른 GPU compute process 가 있으면 기다렸으므로 launcher 두 개를 동시에 띄울 수 없었다. 이제는 다음 두 가지다 (`SL5h-launcher.impl.md`).

| file | 내용 |
|---|---|
| `benchmark/runners/launch_libero_paired.py` | 환경 변수 `AG3S_GPU_LOCK_HELD=1` 이면 launcher 가 lock 을 직접 잡지 않고(busy-GPU 대기도 하지 않는다), **caller 가 lock 을 실제로 쥐고 있고 그 lock 이 물려받은 fd 로 잡혀 있는지**를 확인한 뒤 진행한다. 확인에 실패하면 run 디렉터리를 만들기 전에 `refused` 로 종료(rc 2). `launcher_result` 에 `gpu_lock_mode: "held_by_caller"`, caller pid, fd 를 기록한다. env 가 없으면 이전과 같은 동작이다 |
| `benchmark/runners/run_parallel_gpu.sh` (신규 140 줄) | lock 을 한 번 잡고(`exec 9>>lock; flock -w N 9`) → `nvidia-smi` 로 다른 compute process 가 없는지 확인 → pod working set + 추정(12 GiB/run × 동시 수)이 64 GiB 이하인지 확인 → job(launcher 인자 한 줄)을 **최대 2 개 동시**에 실행하고 하나가 끝나면 다음을 넣는다. job 의 시작 · 끝 · rc · working set 을 한 log 에 남긴다 |
| `tests/benchmark/test_parallel_gpu_launcher.py` (신규) | GPU 없이 mock · fake 로 확인. 구현 기록에는 기존 launcher test 2 개와 함께 13 passed (7.05 s) |

바뀌지 않은 것: server argv, client argv, `libero_safe_client.py`, `serve_safe.py`, adapter, profile (실행 경로). 기존 file 중 바뀐 것은 `launch_libero_paired.py` 한 파일이다 (구현 기록의 `git diff --stat`). CLAUDE.md 의 주의대로 `flock -w N 9 CMD 9>lock`(fd 번호 뒤에 command)은 쓰지 않는다.

**근거.** (i) 병렬로 돌려도 TrajOpt 결과가 바뀌지 않는다는 가정은 `SL5h-launcher.task.md` 가 든 것이다: benchmark V11AB 계열 TO 는 반복 수 상한(`max_iterations = 3`)으로만 끝나고 wall-clock 마감(`sqp.time_budget_ms`)에 걸리지 않는다. 그 근거로 lead 가 SL5c ~ SL5g 약 1 만 call 에서 `time_budget_hit = 0` 임을 적었다 (이 값은 verify.json 에 측정값으로 담겨 있지 않다. 지시서의 서술이다). (ii) 가정을 직접 확인한 것이 아래 **G0 bit 재현**이다.

**배제하는 것.** checkout 사이 동시 실행은 계속 하지 않는다 (development 두 세션과 합의). 같은 checkout 안 2 job 도 lock 을 쥔 한 caller 아래서만 허용한다.

**알려진 한계 (구현 기록).** 두 launcher 가 고른 port 가 같을 확률이 job 쌍마다 약 1/10⁴ 남는다(그러면 두 번째 server 의 bind 가 실패해 그 job 이 rc ≠ 0 이고 결과가 섞이지는 않는다). 메모리 추정 12 GiB/run 은 실측값이 아니다. 병렬 실행이 결과를 바꾸지 않는다는 것은 구현이 보장하지 않고 이 절의 G0 로만 확인한다.

#### run 과 병렬 batch

| 집합 | 조건 | code | run |
|---|---|---|---|
| SafeLIBERO LI dev (n=20) | VLA-only (SL5f 재사용) | 38722284 | `20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only` |
| SafeLIBERO LI dev (n=20) | pb5-C (SL5g 재사용) | a8ad92ef | `20261018T133339_a8ad92ef3356_sl5g-safelibero-LI-dev-LI-vla_v11ab-V11AB-pb5-C` |
| SafeLIBERO LI dev (n=20) | pb5-B (신규) | 743a1e76 | `20261020T035642_743a1e76874d_sl5h-safelibero-LI-dev-LI-vla_v11ab-V11AB-pb5-B` |
| SafeLIBERO LI dev (n=20) | pb5-A (신규) | 743a1e76 | `20261020T035857_743a1e76874d_sl5h-safelibero-LI-dev-LI-vla_v11ab-V11AB-pb5-A` |
| LIBERO 10 init 0 (n=10) | VLA-only (SL5f 재사용) | 38722284 | `20261017T170230_38722284e2fa_sl5f-libero10-init0-vla_only` |
| LIBERO 10 init 0 (n=10) | pb5-C (SL5g 재사용) | a8ad92ef | `20261018T150153_a8ad92ef3356_sl5g-libero10-init0-vla_v11ab-V11AB-pb5-C` |
| LIBERO 10 init 0 (n=10) | pb5-C repro (병렬 재실행) | 743a1e76 | `20261020T034523_743a1e76874d_sl5h-libero10-init0-repro-vla_v11ab-V11AB-pb5-C` |
| LIBERO 10 init 0 (n=10) | pb5-B (신규) | 743a1e76 | `20261020T034555_743a1e76874d_sl5h-libero10-init0-vla_v11ab-V11AB-pb5-B` |
| LIBERO 10 init 0 (n=10) | pb5-A (신규) | 743a1e76 | `20261020T043947_743a1e76874d_sl5h-libero10-init0-vla_v11ab-V11AB-pb5-A` |

변환: A = `lag_inverse`, B = `command_offset`, C = `lag_inverse_tracking` (calls 기록의 `conversion.mode` 로 확인: 모든 call 이 해당 변환; LI|C C 1084, LI|B B 1080, LI|A A 997, LIBERO10|C C 344, LIBERO10|repro C 344, LIBERO10|B B 286, LIBERO10|A A 345 call). pb5 규칙은 모두 `V11AB-pb5` 이다.

| job | 내용 | 시작 (UTC) | 끝 (UTC) | s | 동시에 돈 job (겹친 시간) |
|---|---|---|---|---|---|
| job0 | libero_spatial conversion C (SL5h-G-repro) | 2026-10-19T18:45:22Z | 2026-10-19T18:58:25Z | 783 | job1 618 s, job2 105 s |
| job1 | libero_spatial conversion B (SL5h-G-pb5B) | 2026-10-19T18:45:52Z | 2026-10-19T18:56:10Z | 618 | job0 618 s |
| job2 | safelibero_spatial conversion B (SL5h-G-pb5B) | 2026-10-19T18:56:40Z | 2026-10-19T19:39:16Z | 2556 | job0 105 s, job3 2420 s |
| job3 | safelibero_spatial conversion A (SL5h-G-pb5A) | 2026-10-19T18:58:56Z | 2026-10-19T19:39:46Z | 2450 | job2 2420 s |
| job4 | libero_spatial conversion A (SL5h-G-pb5A) | 2026-10-19T19:39:46Z | 2026-10-19T19:50:46Z | 660 | – |

최대 동시 실행은 2 이다 (job0 · job1, job2 · job3). 로그는 `outputs/verify/SL5h-G/sl5h_g.log`.

#### G0 — 병렬 bit 재현: repro(병렬) vs SL5g pb5-C(단독), LIBERO 10

비교: episode 10 개마다 `<stem>.npz` · `<stem>_calls.npz` · `<stem>_geom.npz` 의 모든 array 를 key 집합 · dtype · shape · byte 단위로, `episodes.jsonl` 의 모든 field 를, calls jsonl 의 모든 leaf 를 비교했다 (`g0_bitrepro.json`).

| 항목 | 결과 |
|---|---|
| npz array (episode 14 + calls 8 + geom 33 개 / episode) | 550 / 550 일치, 불일치 0, 한쪽에만 있는 key 0 |
| 그 중 episode + calls array | 220 / 220 일치 |
| 처음 달라진 episode · call · array | 없음 (`first_difference` = null) |
| `episodes.jsonl` 결과 field (`official_success`, `status`, `error`, `env_steps_including_wait`, `env_steps_excluding_wait`, `wait_steps`, `loop_t_final`, `policy_calls`, `init_state_index`, `seed`) | 10 / 10 episode 일치 |
| `episodes.jsonl` 에서 다른 field | `ended_at` 10 ep, `policy_latency_ms` 10 ep, `server_timing` 10 ep, `started_at` 10 ep (시각 · 지연 기록) |
| calls jsonl (344 call) 에서 다른 leaf | `/timing_ms/ag3s` 344, `/timing_ms/trajopt` 344, `/to/solve_ms` 343, `/timing_ms/infer` 344, `/t_client_ms` 344, `/timing_ms/total` 344, `/grasp/attached/attached_at` 92 (모두 timing · 시각 기록) |
| manifest | commit `a8ad92ef` → `743a1e76`, dirty [False, False]; conversion · ag3s_version · protocol · seed 동일, algorithm tree 는 `AG3S/ag3s` 하나만 다름(docs 를 포함하는 tree, 실행 경로는 같다), server argv 는 port 하나만 다름(index 10: 25269 → 28689) |
| gpu lock 기록 | 단독 run: wait 4.0e-05 s, 2026-10-18T06:01:55Z ~ 2026-10-18T06:13:41Z; repro: `gpu_lock_mode` = `held_by_caller`, caller pid 3265472, fd 9 |
| repro 와 동시에 돈 job | job1(pb5-B LIBERO 10) 618 s, job2(pb5-B LI) 105 s |

이 표가 가정을 확인하는 측정이다: 시각 · timing 기록을 뺀 모든 array 와 결과 field 가 병렬 · 단독 run 사이에서 같다. 다만 재실행한 것은 LIBERO 10 의 pb5-C 한 집합(10 episode)이다. 아래 not_measured 참조.

![SL5h-G tables](figures/sl5h-g-tables.png)

#### G1 — 집계, 전이표, 후보 규칙

figure: [`sl5h-g-graph-conversion.png`](figures/sl5h-g-graph-conversion.png) (a ~ f), [`sl5h-g-tables.png`](figures/sl5h-g-tables.png) (T1 ~ T5), [`sl5h-g-scene-C-vs-B.png`](figures/sl5h-g-scene-C-vs-B.png). sidecar 는 같은 이름의 `.json`. 모든 run 의 pairing key(task · init · policy seed · obstacle) 일치: LI 모두 True, LIBERO10 모두 True.

![SL5h-G 변환별 결과와 서명](figures/sl5h-g-graph-conversion.png)

| 집합 | 조건 | official | collided | safe_success | HOLD calls (episodes) | calls | status |
|---|---|---|---|---|---|---|---|
| LI | VLA-only | 11 / 20 | 18 | 2 | 0 (0 ep) | 796 | max_steps 9, success 11 |
| LI | pb5-A | 4 / 20 | 9 | 3 | 327 (18 ep) | 997 | max_steps 16, success 4 |
| LI | pb5-B | 2 / 20 | 17 | 1 | 246 (15 ep) | 1080 | success 2, max_steps 18 |
| LI | pb5-C | 1 / 20 | 11 | 1 | 204 (16 ep) | 1084 | max_steps 19, success 1 |
| LIBERO 10 | VLA-only | 9 / 10 | – | – | 0 (0 ep) | 248 | success 9, max_steps 1 |
| LIBERO 10 | pb5-A | 5 / 10 | – | – | 56 (6 ep) | 345 | max_steps 5, success 5 |
| LIBERO 10 | pb5-B | 7 / 10 | – | – | 12 (3 ep) | 286 | success 7, max_steps 3 |
| LIBERO 10 | pb5-C | 5 / 10 | – | – | 15 (4 ep) | 344 | success 5, max_steps 5 |
| LIBERO 10 | pb5-C repro | 5 / 10 | – | – | 15 (4 ep) | 344 | success 5, max_steps 5 |

전이표 (같은 task · init · policy seed. official: T→T / T→F / F→T / F→F):

| 집합 | 전이 | official | collided | safe_success |
|---|---|---|---|---|
| LI | vla_only → A | 2 / 9 / 2 / 7 | 8 / 10 / 1 / 1 | 1 / 1 / 2 / 16 |
| LI | vla_only → B | 1 / 10 / 1 / 8 | 17 / 1 / 0 / 2 | 1 / 1 / 0 / 18 |
| LI | vla_only → C | 1 / 10 / 0 / 9 | 10 / 8 / 1 / 1 | 1 / 1 / 0 / 18 |
| LI | C → B | 1 / 0 / 1 / 18 | 10 / 1 / 7 / 2 | 1 / 0 / 0 / 19 |
| LI | C → A | 1 / 0 / 3 / 16 | 8 / 3 / 1 / 8 | 1 / 0 / 2 / 17 |
| LI | A → B | 1 / 3 / 1 / 15 | 8 / 1 / 9 / 2 | 1 / 2 / 0 / 17 |
| LIBERO 10 | vla_only → A | 5 / 4 / 0 / 1 | – | – |
| LIBERO 10 | vla_only → B | 7 / 2 / 0 / 1 | – | – |
| LIBERO 10 | vla_only → C | 5 / 4 / 0 / 1 | – | – |
| LIBERO 10 | vla_only → repro | 5 / 4 / 0 / 1 | – | – |
| LIBERO 10 | C → B | 5 / 0 / 2 / 3 | – | – |
| LIBERO 10 | C → A | 3 / 2 / 2 / 3 | – | – |
| LIBERO 10 | A → B | 5 / 0 / 2 / 3 | – | – |
| LIBERO 10 | C → repro | 5 / 0 / 0 / 5 | – | – |

후보 규칙 (a) VLA-only official F→S, (b) VLA-only safe_success F→T 를 따로 센다. 표의 "충돌 → 무충돌" 은 VLA-only 에서 충돌했던 episode 가 그 변환에서 무충돌인 수다.

| 집합 | 조건 | (a) official F→S | (b) safe_success F→T | 충돌 → 무충돌 | 무충돌 → 충돌 | official T→F (regression) |
|---|---|---|---|---|---|---|
| LI | pb5-A | 2 (t0e2, t0e3) | 2 (t0e3, t2e0; 그 중 VLA-only 충돌 t0e3, t2e0) | 10 (t0e0, t0e1, t0e3, t0e4, t2e0, t2e1, t2e2, t2e4, t3e2, t3e3; official success t0e3, t2e0) | 1 (t1e4) | 9 (t1e4, t2e1, t2e2, t2e3, t2e4, t3e0, t3e2, t3e3, t3e4) |
| LI | pb5-B | 1 (t0e0) | 0 (–; 그 중 VLA-only 충돌 –) | 1 (t0e4; official success –) | 0 (–) | 10 (t1e4, t2e0, t2e1, t2e2, t2e3, t2e4, t3e0, t3e2, t3e3, t3e4) |
| LI | pb5-C | 0 (–) | 0 (–; 그 중 VLA-only 충돌 –) | 8 (t0e0, t0e3, t0e4, t2e1, t2e2, t2e3, t2e4, t3e2; official success –) | 1 (t1e4) | 10 (t1e4, t2e0, t2e1, t2e2, t2e3, t2e4, t3e0, t3e2, t3e3, t3e4) |
| LIBERO 10 | pb5-A | 0 (–) | – | – | – | 4 (t0e0, t4e0, t6e0, t9e0) |
| LIBERO 10 | pb5-B | 0 (–) | – | – | – | 2 (t4e0, t9e0) |
| LIBERO 10 | pb5-C | 0 (–) | – | – | – | 4 (t4e0, t7e0, t8e0, t9e0) |
| LIBERO 10 | pb5-C repro | 0 (–) | – | – | – | 4 (t4e0, t7e0, t8e0, t9e0) |

#### G2 — SL5h 서명: pb5 × 변환

SL5h 의 정의와 script 를 그대로 재사용했다 (용어는 위 SL5h 절). 아래 "ref" 행은 SL5h 에서 복사한 SL5c A · B, pb1-B, SL5g pb5-C 로 **MI gate · arms · pb4/pb5 규칙이 없던 run** 이다.

| 집합 | 조건 | official | over-lift episode / lifted episode | max rise 중앙값 (mm) | max rise 최대 (mm) | lifted 실행 any-axis sat | lifted 실행 z ≥ 0.999 | TO-unchanged row 의 sat / step | D\*_z 중앙값 (mm, clip 된 lifted call) | realised_obs 중앙값 (clip 된 call) | realised_model 중앙값 (lifted 전체) | closing·lifted 행 dz 위 / 아래 | closing·lifted 중 reach False | windup run: ≥3 (그 call 수) / ≥5 / 최장 | first stage: conv_on_to_row / conv_carry / unclipped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LI | VLA-only | 11 / 20 | 0 / 14 | 132.7 | 186.3 | 0 / 717 (0%) | 0 / 717 (0%) | – | – | – | – | – | – | – | – |
| LI | pb5-A | 4 / 20 | 2 / 11 | 104.7 | 403.5 | 342 / 1250 (27%) | 147 / 1250 (12%) | 0 / 139 | 17.4 | 0.29 | 0.43 | 841 / 82 | 52 / 99 (53%) | 60 (539) / 39 / 36 | 1073 / 3 / 0 |
| LI | pb5-B | 2 / 20 | 0 / 10 | 72.7 | 207.7 | 48 / 1166 (4%) | 18 / 1166 (2%) | 0 / 67 | 32.5 | 0.54 | 0.14 | 1009 / 344 | 59 / 144 (41%) | 35 (182) / 15 / 25 | 377 / 0 / 1 |
| LI | pb5-C | 1 / 20 | 10 / 12 | 226.9 | 926.5 | 1158 / 1916 (60%) | 806 / 1916 (42%) | 0 / 389 | 18.7 | 0.83 | 0.55 | 1431 / 243 | 179 / 189 (95%) | 80 (447) / 33 / 29 | 2536 / 87 / 1 |
| LIBERO 10 | VLA-only | 9 / 10 | 0 / 10 | 147.3 | 182.2 | 0 / 348 (0%) | 0 / 348 (0%) | – | – | – | – | – | – | – | – |
| LIBERO 10 | pb5-A | 5 / 10 | 0 / 8 | 138.4 | 247.4 | 87 / 383 (23%) | 38 / 383 (10%) | 0 / 207 | 26.3 | -0.08 | 0.31 | 204 / 21 | 6 / 28 (21%) | 18 (100) / 10 / 10 | 174 / 4 / 0 |
| LIBERO 10 | pb5-B | 7 / 10 | 0 / 9 | 136.7 | 193.0 | 7 / 315 (2%) | 0 / 315 (0%) | 0 / 214 | 22.1 | 0.22 | 0.13 | 137 / 10 | 7 / 19 (37%) | 3 (14) / 1 / 7 | 13 / 0 / 1 |
| LIBERO 10 | pb5-C | 5 / 10 | 1 / 8 | 154.4 | 689.0 | 197 / 538 (37%) | 135 / 538 (25%) | 0 / 265 | 21.9 | 1.28 | 0.99 | 234 / 50 | 33 / 37 (89%) | 15 (77) / 6 / 10 | 376 / 15 / 0 |
| LIBERO 10 | pb5-C repro | 5 / 10 | 1 / 8 | 154.4 | 689.0 | 197 / 538 (37%) | 135 / 538 (25%) | 0 / 265 | 21.9 | 1.28 | 0.99 | 234 / 50 | 33 / 37 (89%) | 15 (77) / 6 / 10 | 376 / 15 / 0 |
| LI | SL5c A (ref) | 3 / 20 | 3 / – | – | – | 0.317 | 0.112 | – | 7.7 | 0.48 | – | 1139 / 588 | 181 / 182 | – | – |
| LI | SL5c B (ref) | 3 / 20 | 0 / – | – | – | 0.064 | 0.021 | – | 30.9 | 0.21 | – | 2049 / 143 | 229 / 233 | – | – |
| LI | SL5g pb5-C (ref) | 1 / 20 | 10 / – | – | – | 0.604 | 0.421 | – | 18.7 | 0.83 | – | 1431 / 243 | 179 / 189 | – | – |
| LIBERO 10 | SL5c A (ref) | 2 / 10 | 0 / – | – | – | 0.391 | 0.155 | – | 17.3 | 0.20 | – | 410 / 58 | 51 / 51 | – | – |
| LIBERO 10 | SL5c B (ref) | 3 / 10 | 0 / – | – | – | 0.044 | 0.012 | – | 21.8 | 0.22 | – | 830 / 122 | 104 / 106 | – | – |
| LIBERO 10 | pb1-B (ref) | 7 / 10 | 0 / – | – | – | 0.036 | 0.000 | – | 18.9 | 0.16 | – | 162 / 15 | 18 / 21 | – | – |
| LIBERO 10 | SL5g pb5-C (ref) | 5 / 10 | 1 / – | – | – | 0.366 | 0.251 | – | 21.9 | 1.28 | – | 234 / 50 | 33 / 37 | – | – |

over-lift episode: LI pb5-A t1e4, t2e1; LI pb5-B 없음; LI pb5-C t0e0, t0e1, t0e3, t1e2, t1e3, t1e4, t2e0, t2e1, t2e2, t2e4; LIBERO 10 pb5-A 없음; LIBERO 10 pb5-B 없음; LIBERO 10 pb5-C t5e0; LIBERO 10 pb5-repro t5e0.

실행 포화가 TO-unchanged row 에서 나온 step 은 A · B · C · repro 모두 0 이고, MI gate 가 통과된 lifted call 의 response sat row 는 pb5-A 0 (21 call) · pb5-B 0 (9 call) · pb5-C 0 (61 call) (LI) 이다.

| 조건 | z 기여 합: TO-changed lifted call (n / mm) | TO-unchanged lifted call (n / mm) | 실행 z 포화 step 의 VLA 원본 z 부호 pos / neg / near0 |
|---|---|---|---|
| LI pb5-A | 219 / -2296.1 | 26 / -6.2 | 85 / 49 / 13 |
| LI pb5-B | 218 / 14891.4 | 10 / 22.9 | 2 / 14 / 2 |
| LI pb5-C | 302 / 3662.2 | 75 / -872.1 | 506 / 229 / 71 |
| LIBERO 10 pb5-A | 36 / -142.0 | 39 / -101.6 | 28 / 10 / 0 |
| LIBERO 10 pb5-B | 21 / 144.5 | 40 / -138.0 | 0 / 0 / 0 |
| LIBERO 10 pb5-C | 55 / 1331.5 | 50 / -280.9 | 44 / 69 / 22 |
| LIBERO 10 pb5-repro | 55 / 1331.5 | 50 / -280.9 | 44 / 69 / 22 |

z 기여 = lifted call 마다 `eef_obs_z(o+5) − FK_z(joint_reference[k,4])` 의 합 (SL5h 정의).

#### G3 — 실패 원인 분류 (SL5g #3 의 A / B / C / D)

VLA-only 성공 → pb5 실패 episode 에 같은 규칙(lift = 두 손가락 접촉 + rise ≥ 30 mm, lost = lifted 뒤 ≥ 10 obs 연속 grasp 없음, over-lift > 250 mm, HOLD 연속 ≥ 5 call)을 적용했다. pb5-C 는 SL5g 의 `failure_classes.json` pb5 와 일치한다 (LI True, LIBERO 10 True).

| 집합 | 조건 | A_no_lift_hold | A_no_lift_other | B_lifted_lost | C_lifted_overlift | D_lifted_other | 실패 episode 수 |
|---|---|---|---|---|---|---|---|
| LI | pb5-A | 3 | 1 | 0 | 2 | 3 | 9 |
| LI | pb5-B | 2 | 1 | 2 | 0 | 5 | 10 |
| LI | pb5-C | 1 | 3 | 2 | 4 | 0 | 10 |
| LIBERO 10 | pb5-A | 1 | 1 | 1 | 0 | 1 | 4 |
| LIBERO 10 | pb5-B | 1 | 0 | 0 | 0 | 1 | 2 |
| LIBERO 10 | pb5-C | 1 | 1 | 1 | 0 | 1 | 4 |
| 합계 | pb5-A | 4 | 2 | 1 | 2 | 4 | 13 |
| 합계 | pb5-B | 3 | 1 | 2 | 0 | 6 | 12 |
| 합계 | pb5-C | 2 | 4 | 3 | 4 | 1 | 14 |

episode 별 (분류 / 최고 rise mm / 충돌):

| 집합 | ep | VLA-only max rise (mm) | pb5-A | pb5-B | pb5-C |
|---|---|---|---|---|---|
| LI | t1e4 | 165.4 | C_lifted_overlift / 403.5 / True | B_lifted_lost / 173.6 / False | C_lifted_overlift / 296.4 / True |
| LI | t2e0 | 174.7 | success / 241.0 / False | D_lifted_other / 186.0 / True | C_lifted_overlift / 926.5 / True |
| LI | t2e1 | 156.2 | C_lifted_overlift / 250.6 / False | D_lifted_other / 193.7 / True | B_lifted_lost / 350.3 / False |
| LI | t2e2 | 172.1 | D_lifted_other / 136.3 / False | D_lifted_other / 204.5 / True | C_lifted_overlift / 454.3 / False |
| LI | t2e3 | 160.9 | D_lifted_other / 157.8 / True | D_lifted_other / 182.3 / True | B_lifted_lost / 144.9 / False |
| LI | t2e4 | 186.3 | D_lifted_other / 212.9 / False | D_lifted_other / 207.7 / True | C_lifted_overlift / 531.3 / False |
| LI | t3e0 | 126.8 | A_no_lift_hold / 7.9 / True | A_no_lift_hold / 14.7 / True | A_no_lift_hold / 0.0 / True |
| LI | t3e2 | 105.0 | A_no_lift_hold / 0.0 / False | A_no_lift_other / 28.1 / True | A_no_lift_other / 4.5 / False |
| LI | t3e3 | 138.5 | A_no_lift_other / 8.6 / False | A_no_lift_hold / 26.8 / True | A_no_lift_other / 0.0 / True |
| LI | t3e4 | 99.8 | A_no_lift_hold / 0.0 / True | B_lifted_lost / 129.0 / True | A_no_lift_other / 0.0 / True |
| LIBERO 10 | t0e0 | 98.3 | B_lifted_lost / 96.6 / – | success / 101.9 / – | success / 96.6 / – |
| LIBERO 10 | t4e0 | 159.9 | A_no_lift_hold / 13.1 / – | A_no_lift_hold / 12.8 / – | A_no_lift_hold / 0.0 / – |
| LIBERO 10 | t6e0 | 145.0 | D_lifted_other / 247.4 / – | success / 145.0 / – | success / 159.2 / – |
| LIBERO 10 | t7e0 | 182.2 | success / 181.3 / – | success / 193.0 / – | D_lifted_other / 243.4 / – |
| LIBERO 10 | t8e0 | 123.1 | success / 125.7 / – | success / 120.4 / – | B_lifted_lost / 237.3 / – |
| LIBERO 10 | t9e0 | 103.8 | A_no_lift_other / 7.6 / – | D_lifted_other / 116.9 / – | A_no_lift_other / 18.0 / – |

"success" 는 그 변환에서 official success 인 episode (VLA-only 성공 → 실패 집합에서 그 변환만 성공한 경우)다.

#### G4 — 사례: LI t0e0 (pb5-C 는 over-lift 로 실패, pb5-B 는 성공, VLA-only 는 실패)

조건 (i) pb5-C 가 over-lift 로 실패하고 pb5-B 가 성공한 episode: 1 건 (LI|t0e0). 조건 (ii) VLA-only 가 실패하고 pb5-B 가 성공한 episode: 1 건 (LI|t0e0). 같은 episode 하나가 두 조건을 만족해 사례 묶음은 1 개다. 묶음 위치 `outputs/verify/SL5h-G/cases/LI_t0e0_pb5_C_vs_B/` (`case.json`, calls csv, frame).

![SL5h-G C vs B scene](figures/sl5h-g-scene-C-vs-B.png)

| 조건 | official | collided | safe_success | max rise (mm) | rise 최고 obs | 첫 lifted obs | HOLD calls | lifted 실행 any-axis sat / lifted step |
|---|---|---|---|---|---|---|---|---|
| VLA-only | False | True | False | 8.1 | 250 | – | 0 | 0 / 0 |
| pb5-A | False | False | False | 99.2 | 131 | – | 12 | 0 / 0 |
| pb5-B | True | True | False | 113.7 | 150 | 136 | 9 | 0 / 23 |
| pb5-C | False | False | False | 515.6 | 178 | 88 | 0 | 144 / 158 |

B 와 갈라지는 시점: VLA-only · pb5-A · pb5-C 모두 env step 30 (B 의 call 2). figure 아래 줄은 LI t2e4(SL5h 의 대표 episode) 의 같은 조건 frame 이다: pb5-C 는 rise 531 mm(`lifted exec sat` 121/177), pb5-B 는 rise 208 mm, pb5-A 는 rise 213 mm.

#### dirty = True run 3 개와 그 원인

repro 를 뺀 SL5h-G 신규 run 4 개 중 3 개(pb5-B LI · pb5-A LI · pb5-A LIBERO 10)의 manifest 가 `dirty = True` 이다. 원인은 모두 같다: **untracked 파일 `AG3S/ag3s/docs/handoff/SL5h-G.task.md` 하나**이고 `source_diff` 는 비어 있다 (코드 변경 없음). 나머지 1 개(pb5-B LIBERO 10)는 `dirty = false` 이다.

| run | dirty 원인 (verify.json `other_runs_dirty_reason`) |
|---|---|
| `20261020T035642_743a1e76874d_sl5h-safelibero-LI-dev-LI-vla_v11ab-V11AB-pb5-B` | untracked AG3S/ag3s/docs/handoff/SL5h-G.task.md only (source_diff empty) |
| `20261020T035857_743a1e76874d_sl5h-safelibero-LI-dev-LI-vla_v11ab-V11AB-pb5-A` | untracked AG3S/ag3s/docs/handoff/SL5h-G.task.md only (source_diff empty) |
| `20261020T043947_743a1e76874d_sl5h-libero10-init0-vla_v11ab-V11AB-pb5-A` | untracked AG3S/ag3s/docs/handoff/SL5h-G.task.md only (source_diff empty) |
| `20261020T034555_743a1e76874d_sl5h-libero10-init0-vla_v11ab-V11AB-pb5-B` | dirty false |

repro 와 SL5g pb5-C 의 manifest 는 둘 다 `dirty = false` 이다 (G0 표).

#### 가설 H1 ~ H3 별 숫자 (판정 없음)

verify.json `hypotheses_numbers` 의 `support` · `against` 구분을 옮긴다. 이 구분은 verifier 의 분류이고 이 로그의 판정이 아니다. H4(VLA reference 포화) 는 이 STEP 의 측정에 없다 (VLA 원본 any-axis 포화 lifted step 은 pb5-A · B · C 모두 0 이다: H2 의 표).

**H1. 포화는 역변환 C 의 windup 에서 생긴다**

| 지표 | pb5-A | pb5-B | pb5-C |
|---|---|---|---|
| lifted 실행 any-axis sat 비율 (LI) | 342 / 1250 (0.274) | 48 / 1166 (0.041) | 1158 / 1916 (0.604) |
| lifted 실행 any-axis sat 비율 (LIBERO 10) | 87 / 383 (0.227) | 7 / 315 (0.022) | 197 / 538 (0.366) |
| lifted 실행 z ≥ 0.999 비율 (LI) | 0.1176 | 0.0154 | 0.4207 |
| lifted 실행 z ≥ 0.999 비율 (LIBERO 10) | 0.0992 | 0.0000 | 0.2509 |
| over-lift episode (LI n=20) | 2 | 0 | 10 |
| over-lift episode (LIBERO 10 n=10) | 0 | 0 | 1 |
| realised_frac_obs 중앙값 (LI, clip 된 lifted call) | 0.290 | 0.535 | 0.826 |
| realised_frac_obs 중앙값 (LIBERO 10) | -0.075 | 0.224 | 1.285 |
| windup run ≥ 3 call: run 수 (그 안 call 수) (LI) | 60 (539) | 35 (182) | 80 (447) |
| conversion clip 이 있는 lifted call / lifted call (LI) | 214 / 247 | 114 / 228 | 294 / 377 |
| VLA-only 성공 → 실패 중 class C_lifted_overlift (두 집합 합) | 2 | 0 | 4 |

against 로 분류된 숫자:

| 지표 | pb5-A | pb5-B | pb5-C |
|---|---|---|---|
| official (LI n=20) | 4 | 2 | 1 |
| official (LIBERO 10 n=10) | 5 | 7 | 5 |
| collided (LI), VLA-only 는 18 | 9 | 17 | 11 |
| official T→F regression (LI) | 9 | 10 | 10 |
| official T→F regression (LIBERO 10) | 4 | 2 | 4 |

- pb5-B(변환 C 없음)의 lifted 실행 any-axis sat step: LI 48, LIBERO 10 7 (0 이 아니다). pb5-B 의 windup run(≥ 3 call): LI 35, LIBERO 10 3.
- 변환 A(`lag_inverse`) over-lift: LI 2, LIBERO 10 0. conv_carry row / 포화 response row (lifted, pb5-C LI): 87 / 2624.

**H2. 포화는 TO 수정에서 생긴다**

| 지표 | pb5-A | pb5-B | pb5-C |
|---|---|---|---|
| TO-unchanged row 의 lifted step 중 포화 / step (LI) | 0 / 139 | 0 / 67 | 0 / 389 |
| TO-unchanged row 의 lifted step 중 포화 / step (LIBERO 10) | 0 / 207 | 0 / 214 | 0 / 265 |
| TO 가 안 바꿨는데 clip 된 call (LI) | 0 | 0 | 0 |
| TO 가 안 바꿨는데 clip 된 call (LIBERO 10) | 0 | 0 | 0 |
| MI gate 통과 lifted call 의 response sat row / call (LI) | 0 / 21 | 0 / 9 | 0 / 61 |
| VLA 원본 any-axis sat lifted step (LI) | 0 | 0 | 0 |
| VLA 원본 any-axis sat lifted step (LIBERO 10) | 0 | 0 | 0 |
| z 기여 합 TO-changed / unchanged, mm (LI) | -2296.1 / -6.2 | 14891.4 / 22.9 | 3662.2 / -872.1 |

against 로 분류된 숫자:

| 지표 | pb5-A | pb5-B | pb5-C |
|---|---|---|---|
| D\*_z 중앙값 mm (LI) | 17.4 | 32.5 | 18.7 |
| D\*_z 중앙값 mm (LIBERO 10) | 26.3 | 22.1 | 21.9 |

- 같은 pb5 TO 규칙에서 B 대 C: lifted 실행 any-axis sat 비율 LI 0.0412 대 0.6044, LIBERO 10 0.0222 대 0.3662; over-lift LI 0 대 10, LIBERO 10 0 대 1.

**H3. latch 가 reach check 에서 막혀 held 로 못 가고, 쥔 bowl 이 제약으로 남아 hand 를 밀어 올린다**

| 지표 | pb5-A | pb5-B | pb5-C |
|---|---|---|---|
| closing·lifted 중 reach False / call (LI) | 52 / 99 | 59 / 144 | 179 / 189 |
| closing·lifted 중 reach False / call (LIBERO 10) | 6 / 28 | 7 / 19 | 33 / 37 |
| closing·lifted 의 TO 행 dz 위 / 아래 (LI) | 841 / 82 | 1009 / 344 | 1431 / 243 |
| closing·lifted 의 TO 행 dz 위 / 아래 (LIBERO 10) | 204 / 21 | 137 / 10 | 234 / 50 |
| reference clearance < 0 / 최적화 / MuJoCo neither within 30 mm, call (LI) | 90 / 96 / 90 | 136 / 143 / 82 | 150 / 180 / 175 |
| lifted call 의 latch (LI) | held 146, closing 99, latched 2 | closing 144, held 63, placed 17, latched 3, searching 1 | closing 189, held 179, searching 3, latched 2, placed 4 |

against 로 분류된 숫자:

- pb5-B (LI): closing·lifted TO 행 dz 위 / 아래 1009 / 344, reach False / call 59 / 144, over-lift episode 0.
- closing·lifted 중 reach False 비율 (LI) pb5-A / B / C = 0.525 / 0.410 / 0.947.

#### 측정하지 않은 것

- LI 의 repro: 병렬 재실행은 LIBERO 10 의 pb5-C 한 집합만 했다.
- pb5-A · pb5-B 의 bit 재현: 각각 한 번만 돌렸고 직렬 쌍둥이 run 이 없다.
- cuRobo regression baseline: GPU 금지.
- TO row 의 constraint 대상(어느 ESDF · attached 물체에 binding 인지): calls log 는 count 만 남긴다 (SL5h 와 같다).

**판정 대기 (사용자).** 변환 A / B / C 의 차이에 대한 원인 판정과 다음 수정 방향은 사용자 판정 전이다. 위는 가설별 숫자만 남긴 것이다.

---

## SL5i — 파지 인식 수정과 closing 중 쥔 물체 면제 (2026-10-22~23)

이름은 사용자 지시(2026-10-22)대로 version 에서 바꾼 내용으로 부른다. 내부 key 는 아래 표에서 한 번만 적고, 이후 본문과 표에서는 쓰지 않는다 (figure 안의 범례에는 내부 key 가 괄호로 남아 있다).

| 부르는 이름 | 내부 key | 내용 |
|---|---|---|
| grasp zone 허용판 | V11AB-pb5 | 직전 최신판. SL5g 에서 측정 |
| 파지 인식 수정판 | V11AB-pb6 | grasp zone 허용판 + 같은 닫힘 시도 안에서 reach 유지 + 열린 개도 기준 `dropped` |
| 파지 인식 + closing 면제판 | V11AB-pb7 | 파지 인식 수정판 + closing 중 쥔 물체에 대한 손 제약 제외 |
| 역변환 C | `lag_inverse_tracking` | SL5h-G 에서 정의. 이 절의 모든 run 이 쓴다 |

### SL5i-1 — 구현과 offline 검사

**기록 시각** 2026-10-23 04:08 KST (구현 기록 작성 2026-10-23 02:40 KST). **code commit** `1ddb9376`.
**출처** [`handoff/SL5i.task.md`](handoff/SL5i.task.md) (무엇을 왜 바꾸나) · [`handoff/SL5i.impl.md`](handoff/SL5i.impl.md) (구현 · test · replay 표) ·
`outputs/verify/SL5i-pre/replay_summary.json` (offline latch replay 요약; 같은 폴더의 `latch_replay.json` · `replay_table.csv`).
GPU 는 쓰지 않았다 (CPU, replay 1.2 s, RSS 42 MB).

**사용자 판정 2026-10-22: "추천 방향에 맞게 수행".** 추천 방향은 두 가지다. ① latch(손가락이 실제로 물체를 쥐었는지 확인하는 상태기계)가 실제 파지를 인정하도록 파지 확인을 고친다. ② closing 중 쥔 물체에 대한 손의 제약을 뺀다. 변경마다 따로 검증하기로 해서 version 이 둘이다.

#### 원인 — 실제로 쥐고 있는데 latch 가 closing 에 머문다

SL5h 에서 over-lift(쥔 bowl 을 250 mm 넘게 들어 올림)의 출처를 따라가면 다음 연쇄가 나온다. latch 가 `held` 로 넘어가지 못하면 AG3S 가 보는 manipulated centroid 가 table 위 옛 위치에 남고, TO 가 손 근처의 쥔 bowl 을 obstacle 로 보고 손을 위로 민다(SL5h D4). 역변환 C 가 그 밀기를 그대로 실행해 over-lift 가 생긴다.

그 연쇄의 첫 고리, 즉 closing 인데 이미 bowl 을 든 call 의 수와 그 call 에서 실패한 check 는 다음과 같다 (grasp zone 허용판 + 역변환 C, 출처 [`SL5i-G.verify.json`](handoff/SL5i-G.verify.json) `numbers.I3.signature`, SL5g 와 같은 run).

| 집합 | closing 이면서 lifted 인 call | 실패한 check (call 수) |
|---|---|---|
| SafeLIBERO Level I dev 20 (LI) | 189 | reach 179 · blocked 53 · dropped 42 · settled 17 · not_empty 1 |
| LIBERO 10 | 37 | dropped 36 · reach 33 · settled 2 |

(lifted = 두 손가락 면이 target body 에 닿고 rise ≥ 30 mm. 한 call 이 여러 check 에서 동시에 실패할 수 있다.)

실패한 check 두 개의 구체적 모양은 다음과 같다 (`SL5i.task.md`, SL5g 기록에서 lead 가 다시 확인한 숫자).

- **reach 가 나중 capture 로 다시 재져 실패한다.** `reach-at-capture` 는 손-bowl 거리를 bowl 의 마지막 관측(capture) 시점 자세로 잰다. closing 에 들어갈 때는 통과한다(LI t2e4 seq 6: 91.1 mm, 한도 120 mm, geometry frame 5 / LIBERO t5e0 seq 10: 81.7 mm). 그 뒤 다른 capture 가 들어오면 실패하고 그대로 고정된다(t2e4: frame 21 에서 123 mm 로 34 call 고정 / t5e0: seq 11 부터 133 mm). bowl 은 쥐기 전에는 움직이지 않으므로, 한 번의 닫힘 시도 안에서 "그 bowl 에 손이 닿을 거리였다"는 사실은 바뀌지 않는다.
- **`dropped` 가 늦은 latch 때문에 기준을 잃는다.** `dropped` 는 `opening_before − opening ≥ 0.05` 인지 본다. `opening_before` 는 LATCHED→CLOSING 전이 순간의 직전 개도다. LIBERO t5e0 에서는 latch 가 손가락이 이미 거의 닫힌 뒤(seq 9, 개도 0.072)에 잠겨 `opening_before` 가 0.0719 로 잡혔고, 그 뒤 개도 0.055~0.072 범위에서 34 call 동안 `dropped=False` 였다. 손가락은 실제로 열림(≈1.0)에서 닫혔다. 기준이 잘못 잡혔을 뿐이다.
- **closing 중에도 같은 위쪽 밀기가 나온다.** 직전 version 의 held 면제(쥔 물체에 대해 손의 contact-link 행을 제약에서 뺀다)는 latch 가 `held` 일 때만 적용되므로, 손가락이 이미 닫혀 bowl 을 쥔 closing 구간은 면제되지 않는다.

#### 바꾼 것 — flag 3개, version 2개

모든 flag 는 기본값이 꺼짐이다.

| flag | 하는 일 | 기록(flag 가 켜졌을 때만 생기는 key) |
|---|---|---|
| `--grasp-reach-sticky` | **sticky reach**: 같은 닫힘 시도(LATCHED→CLOSING 부터 reopen · revoke · LATCHED 복귀 전까지) 안에서 reach check 가 한 번 통과했으면 그 시도가 끝날 때까지 통과로 둔다. 같은 `manipulated_id` 일 때만 적용(다른 물체로 바뀌면 지금 값으로, fail-closed). 같은 판단을 AG3S `attach` 의 reach gate 에도 적용해 "latch 는 held 인데 attach 가 거부되는" 어긋남을 막는다 | `grasp.evidence.reach_sticky`, `held_record.reach_gate` |
| `--grasp-drop-from-open` | **drop-from-open**: `dropped` 의 기준 개도를 "LATCHED→CLOSING 순간의 개도" 대신 "닫힘 명령이 시작되기 전 마지막으로 열려 있던 측정 개도"(명령 ≥ 0.85 이고 개도 ≥ 0.85 인 마지막 요청, SEARCHING 부터 추적)로 바꾼다. 그런 측정이 없으면 예전 값으로 돌아간다 | `grasp.evidence.drop_from_open` |
| `--closing-hand-exempt` | **closing 면제**: latch 가 CLOSING 이고 그 call 의 `closed` · `not_empty` · `blocked` 가 모두 참일 때(손가락이 무언가를 쥐고 멈춘 상태) held 면제와 같은 규칙을 적용한다. 손의 authorized link 행 중 nearest surface 가 manipulated object 인 행만 뺀다. 다른 물체 · table · obstacle 행은 남는다 | `to.grasp_contact.closing_exempt_rows` |

| version | 구성 |
|---|---|
| 파지 인식 수정판 | grasp zone 허용판 + `--grasp-reach-sticky` + `--grasp-drop-from-open` |
| 파지 인식 + closing 면제판 | 파지 인식 수정판 + `--closing-hand-exempt` |

바뀐 file 은 `AG3S/trajopt/grasp_latch.py` · `linearize.py` · `safe_policy.py` · `serve_safe.py`, `AG3S/ag3s/runtime/pipeline.py`, `AG3S/adapters/panda/profile.py` (code 6 file, +217 −16 에 golden allowlist `tests/trajopt/_sl5a_golden.py` 포함). file 별 줄 범위는 `SL5i.impl.md` 의 표가 정본이다. 새 test 는 `tests/trajopt/test_sl5i_grasp_recognition.py` 19 개(sticky 유지 · 초기화, drop 기준, attach gate, closing 면제의 조건과 빠지는 행, profile · CLI).

#### flag 가 꺼진 상태에서 기존 동작과 bit 동일

| 항목 | 변경 전 | 변경 후 |
|---|---|---|
| 회귀 기준선 legacy (`esdf_rollout … --constraint-links arms`, CPU) | 위반으로 시작 14/15 · has_target 15/15 · frame0 −29.031048280806342 mm · clearance_before sha1 `40798fb0…` | 같음 (sha1 까지) |
| RB-Y1 golden `sl5a_rby1_golden.npz` (SafePolicy 경로) | identical | identical, 0 differences |
| RB-Y1 golden `sl5a_serve_safe_golden.json` | identical | 새 argparse key 3개를 꺼진 값으로 allowlist(`PB_DEFAULT_OFF`)에 넣은 뒤 identical (이전 version 들과 같은 방식) |
| offline latch replay, flag off | – | SL5g 의 30 episode 모든 call 에서 latch state · checks 가 기록과 같다(0 diff). 다시 계산한 `closing_steps` 도 같다(0 mismatch) |
| flag off 일 때 code 경로 | – | 새 kwarg(`reach_context`, `reach_sticky`)를 넘기지 않고, 새 evidence · record key 를 만들지 않는다 |

test: 새 test 와 관련 test 47 passed, `tests/ag3s` 계열 1033 passed. `tests/trajopt tests/benchmark` 에는 변경 전부터 있던 실패 47 failed + 2 errors 가 남아 있다. 원인은 `.venv-ag3s` 에 `openpi_client` 가 없는 것(`test_panda_sl3.py`)과 legacy harness 의 chunk index 가 옛 path 를 key 로 쓰는 `KeyError` 42 개이며, 변경하지 않은 HEAD 사본에서도 같은 42 개가 실패했다. cuRobo 기준선은 GPU 가 필요해서 이 단계에서 돌리지 않았다.

#### offline latch replay — 기록된 입력으로 flag 를 켜 본다

**offline latch replay** 는 SL5g(grasp zone 허용판 + 역변환 C)의 calls log 에서 latch 입력(실행 피드백 개도 · 명령 · `closing_steps` · manipulated state · reach)을 꺼내 `GraspLatch` 에 다시 넣고, flag 를 켰을 때 **처음 held 로 넘어가는 seq** 를 구하는 것이다. 대상은 LI 20 + LIBERO 10 episode 다. SEARCHING→LATCHED 전이(grounding lock)는 attention 으로 다시 계산하지 않고 기록의 `latched` call 에서 강제로 잠갔다. 이 전이는 flag 와 상관이 없다.

**첫 divergence** = flag 를 켠 latch 의 출력(state · attach · detach)이 기록과 처음 달라진 call 이다. 첫 held 뒤에는 attach 가 제약을 바꿔 closed loop 가 갈라지므로, 정확한 비교는 첫 divergence 까지만 의미가 있다. **양손가락** = SL5h MuJoCo probe 에서 그 call 의 obs 에 왼쪽 · 오른쪽 손가락 면이 모두 target 에 닿아 있는가 (O/X). **거짓 held** = 양손가락이 target 에 닿지 않았는데 held 가 된 것. 이번 flag 때문에 새로 생긴 거짓 held 를 따로 센다.

| | LI (20) | LIBERO 10 (10) |
|---|---|---|
| flag off replay 와 기록의 불일치 (state · checks 다른 call / 입력 불일치) | 0 / 0 | 0 / 0 |
| held 에 도달한 episode: 기록(off) → sticky 만 / drop 만 / 파지 인식 수정판 | 8 → 10 / 8 / 10 | 7 → 7 / 8 / 8 |
| 첫 held 이전 closing 이면서 lifted 인 call: off → sticky 만 / drop 만 / 수정판 | 152 → 90 / 152 / 90 | 34 → 34 / 0 / 0 |
| 수정판의 divergence (모두 closing→held, 양손가락 O, lifted O) | 4 (sticky 3, drop 1) | 2 (drop 2) |
| **새 거짓 held** (수정판 flag 때문에 held 가 됐는데 양손가락 X) | **0** | **0** |
| 기록의 첫 held 가 이미 양손가락 X 였던 것 (수정판에서도 같은 seq, 새로 생긴 것 아님) | t0e1 seq 18 (rise 9.3 mm) · t1e3 seq 9 (33.2) · t2e1 seq 12 (68.7) | t0e0 seq 15 (0.0) |
| closing 면제가 적용될 call (첫 held 이전, `closed`·`not_empty`·`blocked` 모두 참): 전체 / 양손가락 O / X | 103 / 56 / 47 | 23 / 7 / 16 |

divergence 가 있는 6 episode 의 상세 (replay 표 `replay_table.csv` 에서):

| set | episode | 기록의 첫 held (seq · 양손가락 · rise) | 수정판 replay 의 첫 divergence (원인) | MuJoCo 첫 lifted call (obs) | closing·lifted call: off → 수정판 |
|---|---|---|---|---|---|
| LI | t0e0 | 없음 | seq 16 closing→held (sticky reach). held obs 100, 양손가락 O, rise 90.5 mm | 14 (90) | 32 → 2 |
| LI | t2e4 | 없음 | seq 24 closing→held (sticky reach). held obs 140, 양손가락 O, rise 160.0 mm | 22 (130) | 34 → 2 |
| LIBERO 10 | t5e0 | 없음 | seq 10 closing→held (drop-from-open). held obs 60, 양손가락 O, rise 41.9 mm | 10 (60) | 34 → 0 |
| LI | t0e1 | seq 18 · X · 9.3 mm | seq 25 closing→held (sticky reach, revoke 뒤 재시도). 양손가락 O, lifted O | 24 (140) | 0 → 0 |
| LI | t2e1 | seq 12 · X · 68.7 mm | seq 52 closing→held (drop-from-open, revoke 뒤 재시도). 양손가락 O, lifted O | 10 (70) | 2 → 2 |
| LIBERO 10 | t7e0 | seq 10 · O · 6.4 mm | seq 15 closing→held (drop-from-open, revoke 뒤 재시도). 양손가락 O, lifted O | 13 (75) | 0 → 0 |

- **처음 held 가 새로 생기는 episode 는 3개** (LI t0e0 · t2e4, LIBERO 10 t5e0)이고, 나머지 3개(t0e1 · t2e1 · t7e0)는 첫 held 가 revoke 된 뒤의 재시도에서 갈라진다. 6 곳 모두 양손가락 O 이고 bowl 이 들린 순간이며, 새 거짓 held 는 0 이다.
- 파지 인식 수정판에도 남는 실패 check (replay 에서): **LI t1e2** 는 `blocked` 가 43 call 실패한다(개도 0.027–0.037 이 명령 ≈ 0 보다 `blocked_gap` 0.05 이상 위에서 멈추지 않는다. Panda 손가락이 bowl 테두리를 끝까지 닫고 쥔 경우다). **LI t2e0** 는 `reach` 가 39 call 실패한다(latch 가 잠근 manipulated 가 손에서 400–600 mm 떨어져 있다. 손은 SEARCHING 중 이미 bowl 을 쥐었으므로 다시 잰 값이 아니라 대상 식별 문제이고, sticky 로는 통과하지 못한다).
- 한계: closing 면제는 replay 로 효과를 볼 수 없다. 면제가 바꾸는 것은 TO 의 제약이고 replay 는 latch 입력만 다시 넣기 때문이다. 위 면제 대상 call 수(103 / 23)는 기록된(면제 없는) 궤적 위에서 조건을 센 것이다. 양손가락 X 인 call 에서도 면제가 적용되므로 closed loop 에서 확인하도록 남겼다.

**측정 대기였던 것:** closed-loop GPU 비교(아래 SL5i-2)와 figure.

### SL5i-2 — closed-loop 검증 (SL5i-G)

**기록 시각** 2026-10-23 04:08 KST. **code commit** `1ddb9376` (두 version 의 run commit, dirty False). 측정 verify·figure commit `23efe983`.
**수치 출처** [`handoff/SL5i-G.verify.json`](handoff/SL5i-G.verify.json) 의 `numbers` (측정 task: [`handoff/SL5i-G.task.md`](handoff/SL5i-G.task.md)).
**raw** `outputs/verify/SL5i-G/` (`numbers.json` · `i0_identity.json` … `i5_exempt.json` · `calls/` · `cases/`).
run: 파지 인식 수정판 · 파지 인식 + closing 면제판 × {SafeLIBERO Level I dev 20 (LI), LIBERO 10 init 0} 4 run, 역변환 C, 같은 checkout 병렬 GPU(동시 ≤ 2, `held_by_caller`). 비교 기준은 SL5f VLA-only 와 SL5g grasp zone 허용판 run (재사용). 시작 전 `gpu compute apps` 없음, working set 18.35 GiB. 시각 UTC: LI 두 run 2026-10-22 17:43 ~ 18:28, LIBERO 10 두 run 18:28 ~ 18:40 (KST 로 2026-10-23 02:43 ~ 03:40), 4 job 모두 rc 0.

figure (규칙 A 의 셋. sidecar `.json` 은 같은 이름): 실제 scene [`sl5i-scene-grasp-moment.png`](figures/sl5i-scene-grasp-moment.png), graph [`sl5i-graph-versions.png`](figures/sl5i-graph-versions.png) (a ~ f), table [`sl5i-tables.png`](figures/sl5i-tables.png) (I0 ~ I6).

![SL5i-G version 별 결과 · lift 시점 · 규칙 적용](figures/sl5i-graph-versions.png)

#### I0 — 기존 동작 확인

| 확인 | 결과 |
|---|---|
| 회귀 기준선 legacy | 14/15 · has_target 15/15 · frame0 −29.031048280806342 mm, clearance_before sha1 이 SL5h-G · SL5i 구현 단계와 같다. cuRobo 기준선은 GPU 금지라 이번에도 돌리지 않았다 |
| run commit · dirty | 새 version 4 run 모두 `1ddb9376`, dirty False, source diff 비어 있음. grasp zone 허용판 run 은 `a8ad92ef`, dirty False |
| server argv | 새 flag 가 argv 와 server 시작 log(`SafePolicy grasp contact rules`)에 있다: 수정판은 `latch_reach_sticky` · `latch_drop_from_open`, 면제판은 거기에 `closing_hand_exempt` 추가. 같은 집합의 grasp zone 허용판 argv 와의 차이는 이 flag 들뿐이다 |
| conversion | 모든 call 이 역변환 C (LI 수정판 1049 · 면제판 1012 · grasp zone 허용판 1084, LIBERO 10 318 · 325 · 344) |
| 새 record key | 수정판 LI: reach_sticky 373 call · drop_from_open 383 call · closing_exempt_rows 0. 면제판 LI: 298 · 308 · 1012 (모든 call). LIBERO 10 수정판 65 · 65 · 0, 면제판 66 · 66 · 325. grasp zone 허용판은 모두 0 |
| VLA-only 재사용 근거 | VLA-only commit `38722284` → `1ddb9376` 사이 바뀐 code file 12개에서 `serve_safe.py` 의 diff hunk 두 개(parser 인자, `apply_grasp_contact_flags`)는 `--no-safe` 분기와 겹치지 않는다. VLA-only 경로의 `policy_seed.py` · `libero_safe_client.py` · `third_party/openpi` · `common` · `scripts` · `AG3S/adapters/panda/action.py` · `model.py` 는 변경 없음. launcher 변경은 병렬 option 뿐이며 그 bit 재현은 SL5h-G 에서 550/550 npz array 일치. VLA-only run 과 새 run 의 protocol · schedule · checkpoint · dependencies 차이 0 |

#### I1 — 집계 · 전이표 · 후보 규칙

pairing key(task · init · policy seed · obstacle)는 모든 run 에서 일치했다. VLA-only 와 grasp zone 허용판의 집계는 SL5g 의 값과 같다(official · collided · safe · HOLD 모두).

| 집합 | version | official | collided | safe_success | HOLD calls (episodes) | calls |
|---|---|---|---|---|---|---|
| LI | VLA-only | 11 / 20 | 18 | 2 | 0 (0) | 796 |
| LI | grasp zone 허용판 | 1 / 20 | 11 | 1 | 204 (16) | 1084 |
| LI | 파지 인식 수정판 | 2 / 20 | 11 | 2 | 204 (17) | 1049 |
| LI | 파지 인식 + closing 면제판 | **3 / 20** | 8 | **3** | 204 (16) | 1012 |
| LIBERO 10 | VLA-only | 9 / 10 | – | – | 0 (0) | 248 |
| LIBERO 10 | grasp zone 허용판 | 5 / 10 | – | – | 15 (4) | 344 |
| LIBERO 10 | 파지 인식 수정판 | 6 / 10 | – | – | 14 (4) | 318 |
| LIBERO 10 | 파지 인식 + closing 면제판 | 6 / 10 | – | – | 27 (3) | 325 |

episode 전이표 (같은 pairing, official T→T / T→F / F→T / F→F):

| 집합 | 비교 | official | 규칙 (a) official F→S | 규칙 (b) safe F→T | 충돌 → 무충돌 | 무충돌 → 충돌 |
|---|---|---|---|---|---|---|
| LI | VLA-only → grasp zone 허용판 | 1 / 10 / 0 / 9 | 0 | 0 | 8 | 1 (t1e4) |
| LI | VLA-only → 수정판 | 1 / 10 / 1 / 8 | 1 (t0e0) | 1 (t0e0, VLA-only 충돌) | 8 | 1 (t1e4) |
| LI | VLA-only → 면제판 | 1 / 10 / 2 / 7 | 2 (t0e0, t0e3) | 2 (t0e0, t0e3, 둘 다 VLA-only 충돌) | 11 | 1 (t1e4) |
| LIBERO 10 | VLA-only → grasp zone 허용판 | 5 / 4 / 0 / 1 | 0 | – | – | – |
| LIBERO 10 | VLA-only → 수정판 | 5 / 4 / 1 / 0 | 1 (t5e0) | – | – | – |
| LIBERO 10 | VLA-only → 면제판 | 5 / 4 / 1 / 0 | 1 (t5e0) | – | – | – |

- official T→F(regression, VLA-only 성공 → 실패)는 LI 에서 세 version 모두 10 episode(t1e4 · t2e0 · t2e1 · t2e2 · t2e3 · t2e4 · t3e0 · t3e2 · t3e3 · t3e4)다. LIBERO 10 은 4 episode 로, grasp zone 허용판 · 수정판은 t4e0 · t7e0 · t8e0 · t9e0, 면제판은 t4e0 · t6e0 · t8e0 · t9e0 이다.
- grasp zone 허용판 → 면제판: LI official F→T 2 (t0e0 · t0e3), 충돌 T→F 3 (t0e1 · t1e2 · t1e3), 충돌 T→T 8. LIBERO 10 은 F→T 2 (t5e0 · t7e0), T→F 1 (t6e0).
- grasp zone 허용판 → 수정판: LI 는 official F→T 1 (t0e0) 외에 collided 전이 없음(T→T 11 · F→F 9). LIBERO 10 은 F→T 1 (t5e0).
- 같은 safe_success 3/20 은 grasp zone 허용판 + 역변환 A(SL5h-G, 3/20)와 V11AB-A 기준선(SL5c-A, 3/20)에도 있다. 이번 면제판이 그 값을 넘지는 않았다.

#### I2 — 규칙이 실제로 적용됐나, 언제 held 가 되나

replay 가 예측한 대로 갈라졌는가: 수정판 run 이 grasp zone 허용판 run 과 처음 달라지는 call 이 offline replay 가 예측한 seq 와 **LI 20/20, LIBERO 10 10/10 episode** 에서 같고, 첫 HELD seq 도 같다. replay 가 예측한 divergence episode 수(LI 4 · LIBERO 10 2)와 run 에서 실제로 action 이 갈라진 episode 수가 같다. 수정판 run 과 면제판 run 의 action 이 끝까지 같은 episode 는 LI 12 · LIBERO 10 6 개다.

- sticky used = reach 를 지금 재면 실패인데 이 시도의 통과 기록이 True 로 둔 call. drop basis changed = 새 기준에서만 `dropped` 가 참인 call. HELD 진입 = latch state 가 held 가 아니다가 held 가 된 call.
- 양손가락 = MuJoCo probe 의 `both_fingers`(contact probe 의 left·right 와 모든 obs 에서 일치: LI 16982/16982, LIBERO 10 5240/5240).

| 집합 | version | sticky used (HELD 진입) | drop basis changed (HELD 진입) | closing 면제 call (빠진 행 수) | held 에 도달한 episode | lifted 인데 held 없음 | HELD 진입 중 양손가락 X (새로 생긴 것) | HELD − 첫 lift obs (median · ≤0 / >0) |
|---|---|---|---|---|---|---|---|---|
| LI | grasp zone 허용판 | 0 (0) | 0 (0) | 0 | 8 | t0e0 · t1e2 · t2e0 · t2e4 | 3 / 9 (0) | 1.0 (4 / 4) |
| LI | 파지 인식 수정판 | 69 (3) | 71 (1) | 0 | 10 | t1e2 · t2e0 | 3 / 13 (0) | 3.0 (4 / 6) |
| LI | 파지 인식 + closing 면제판 | 7 (2) | 67 (0) | 31 (872) | 11 | t2e0 | 2 / 12 (0) | 3.0 (4 / 7) |
| LIBERO 10 | grasp zone 허용판 | 0 (0) | 0 (0) | 0 | 7 | t5e0 | 1 / 10 (0) | 1.0 (2 / 5) |
| LIBERO 10 | 파지 인식 수정판 | 3 (0) | 3 (2) | 0 | 8 | 없음 | 1 / 11 (0) | 1.0 (2 / 6) |
| LIBERO 10 | 파지 인식 + closing 면제판 | 5 (1) | 3 (2) | 15 (211) | 8 | 없음 | 1 / 11 (0) | 1.0 (2 / 6) |

- **거짓 held 는 새로 생기지 않았다** (모든 version · 집합에서 grasp zone 허용판과 같은 seq 가 아닌 양손가락 X 진입 0). 양손가락 X 진입은 기존 episode(LI t0e1 · t1e3 · t2e1, LIBERO 10 t0e0)이며, 면제판 LI 에서는 t2e1 이 빠져 2 건이다.
- closing 이면서 lifted 인 call 의 실패 check 중 reach 는 LI 에서 179(허용판) → 67(수정판) → 65(면제판) call, `dropped` 는 42 → 0 → 0, LIBERO 10 에서 reach 33 → 0 → 0, `dropped` 36 → 0 → 0 이다. (I3 표)

#### I3 — SL5h 서명 (같은 정의, SL5h `analyze.py` 그대로)

| 집합 | version | over-lift episode | 실행 action 포화 (lifted step 중, any axis / z) | closing·lifted call | 그 중 reach False | closing·lifted TO 행 z 위 / 아래 | held·lifted TO 행 위 / 아래 | AG3S centroid − 실제 bowl 거리 median (held call, mm) |
|---|---|---|---|---|---|---|---|---|
| LI | VLA-only | 0 | 0 / 0 (0/717) | – | – | – | – | – |
| LI | grasp zone 허용판 | 10 | 0.604 / 0.421 | 189 | 179 | 1431 / 243 | 957 / 93 | 287.7 |
| LI | 파지 인식 수정판 | 9 | 0.587 / 0.404 | 119 | 67 | 927 / 134 | 1294 / 156 | 305.5 |
| LI | 파지 인식 + closing 면제판 | 7 | 0.506 / 0.329 | 75 | 65 | 530 / 93 | 1317 / 307 | 345.4 |
| LIBERO 10 | VLA-only | 0 | 0 / 0 (0/348) | – | – | – | – | – |
| LIBERO 10 | grasp zone 허용판 | 1 | 0.366 / 0.251 | 37 | 33 | 234 / 50 | 174 / 11 | 172.2 |
| LIBERO 10 | 파지 인식 수정판 | 1 | 0.202 / 0.105 | 2 | 0 | 10 / 4 | 196 / 18 | 169.5 |
| LIBERO 10 | 파지 인식 + closing 면제판 | 0 | 0.260 / 0.136 | 4 | 0 | 5 / 1 | 246 / 44 | 190.4 |

over-lift episode: LI 허용판 t0e0 · t0e1 · t0e3 · t1e2 · t1e3 · t1e4 · t2e0 · t2e1 · t2e2 · t2e4, 수정판 t0e1 · t0e3 · t1e2 · t1e3 · t1e4 · t2e0 · t2e1 · t2e2 · t2e4, 면제판 t0e1 · t1e2 · t1e4 · t2e0 · t2e1 · t2e2 · t2e4. LIBERO 10 허용판 t5e0, 수정판 t7e0. 최고 rise 의 median 은 LI 에서 VLA-only 132.65 mm → 허용판 226.9 → 수정판 151.15 → 면제판 125.9 mm (최댓값은 세 version 모두 926.5 mm, t2e0).

- **held 이후에도 AG3S centroid 와 실제 bowl 의 거리는 줄지 않았다.** held 인 call 의 median 이 LI 에서 287.7 → 305.5 → 345.4 mm 로 오히려 늘었다(first held 이후 모든 call 기준 240.6 → 240.7 → 325.8 mm). LIBERO 10 은 172.2 → 169.5 → 190.4 mm.
- closing 면제가 적용된 call(LI 31 · LIBERO 10 15)은 closing·lifted 이면서 TO 가 최적화한 call 과 **겹치지 않는다**: 이 구간의 면제 call 수와 그 행 수는 모든 version 에서 0 이고 위 표의 closing·lifted 행은 전부 면제되지 않은 행이다. 즉 면제판에서도 closing·lifted 의 위쪽 행 530 은 면제 대상이 아니었다.

#### I4 — 실패 원인 분류 (VLA-only 성공 → 실패 episode, SL5g class)

class: A no-lift(A_hold 는 최장 HOLD 연속 ≥ 5 call) · B lifted_lost · C lifted_overlift · D lifted_other. grasp zone 허용판의 분류는 SL5g 의 분류와 같다(LI · LIBERO 10 모두 일치).

| 집합 | version | C over-lift | B lost | A no-lift (hold / other) | D other |
|---|---|---|---|---|---|
| LI (10 실패) | grasp zone 허용판 | 4 | 2 | 1 / 3 | 0 |
| LI | 파지 인식 수정판 | 4 | 2 | 1 / 3 | 0 |
| LI | 파지 인식 + closing 면제판 | 5 | 1 | 1 / 3 | 0 |
| LIBERO 10 (4 실패) | grasp zone 허용판 | 0 | 1 | 1 / 1 | 1 |
| LIBERO 10 | 파지 인식 수정판 | 1 | 1 | 1 / 1 | 0 |
| LIBERO 10 | 파지 인식 + closing 면제판 | 0 | 1 | 1 / 1 | 1 |

- LI 의 class 가 바뀐 episode 는 t2e1 (허용판 · 수정판 B → 면제판 C, 최고 rise 288.6 → 537.3 mm)뿐이다. t2e4 는 수정판에서 held 가 생겼지만(seq 24) C 에 머문다(최고 rise 531.3 → 647.9 → 258.1 mm). t2e2 는 C 이고 면제판에서 최고 rise 가 454.3 → 279.7 mm 로 줄었다. LIBERO 10 에서는 t7e0 이 허용판 D(243.4 mm) → 수정판 C(261.8 mm) → 면제판 success(186.8 mm), t6e0 이 허용판 · 수정판 success → 면제판 D(237.2 mm)다.
- held 에 도달하지 못한 episode 의 CLOSING call 실패 check — 수정판 LI: t0e2 settled 19 · reach 18 · blocked 9, t1e0 reach 33 · blocked 24 · settled 26, t1e2 blocked 44(reach 는 0), t2e0 reach 43, t3e0 · t3e1 · t3e2 · t3e3 는 reach 와 blocked · not_empty 가 함께 실패, t0e4 는 closing call 없음.
- **초점 episode:**

| episode | version | official | first held (seq) | closing call 수 · 실패 check | max rise (mm) |
|---|---|---|---|---|---|
| LI t1e2 (`blocked` 가 막던 사례) | grasp zone 허용판 | 실패, 충돌 | 없음 | 45 · blocked 44 · reach 42 · settled 3 | 738.3 |
| | 파지 인식 수정판 | 실패, 충돌 | 없음 | 45 · blocked 44 · settled 3 (reach 0) | 738.3 |
| | 파지 인식 + closing 면제판 | 실패, 충돌 없음 | 13 (양손가락 O, rise 77.5 mm) | 2 · settled 2 | 370.9 |
| LI t2e0 (손에서 먼 cluster 를 잠금) | 세 version 모두 | 실패, 충돌 | 없음 | 43 · reach 43 (수정판 · 면제판), 허용판은 dropped 42 도 | 926.5 |

  t2e0 의 reach 는 세 version 모두 399.9 ~ 1039.1 mm 이고 manipulated id 는 1 로 같다. 수정판 · 면제판이 `dropped` 실패는 없앴지만 reach 는 그대로다. t1e2 의 면제판 held 는 note 에 "개도 0.994→0.081, 명령 0.001, reach 74.2 mm ≤ 120 mm" 로 남았다.

#### I5 — closing 면제의 부작용

조건 call = CLOSING 이고 그 call 의 `closed` · `not_empty` · `blocked` 가 모두 참인 call(면제 조건). 창 = call 뒤 실행 5 step(obs o+1 ~ o+5). 양손가락 O/X 는 call obs 기준.

| 집합 | version · call 집합 | calls | 빠진 행 | gripper–target 접촉(창) | robot–other 접촉(창) | target 수평 이동 ≥ 10 mm | 충돌(창) |
|---|---|---|---|---|---|---|---|
| LI | 면제판 조건 call 전체 | 135 | 872 | 104 | 2 | 95 | 1 |
| LI | 면제판 조건 call 중 양손가락 X | 48 | 539 | 17 | 2 | 12 | 0 |
| LI | 면제판 실제 면제 call 전체 | 31 | 872 | 30 | 0 | 21 | 1 |
| LI | 면제판 실제 면제 call 중 양손가락 X | 16 | 539 | 15 | 0 | 10 | 0 |
| LI | 비교: 허용판 조건 call 중 X | 55 | 0 | 23 | 2 | 20 | 1 |
| LI | 비교: 수정판 조건 call 중 X | 53 | 0 | 21 | 2 | 18 | 1 |
| LIBERO 10 | 면제판 실제 면제 call 전체 | 15 | 211 | 15 | 0 | 11 | 0 |
| LIBERO 10 | 면제판 실제 면제 call 중 양손가락 X | 9 | 120 | 9 | 0 | 6 | 0 |
| LIBERO 10 | 비교: 허용판 · 수정판 조건 call 중 X | 16 · 17 | 0 | 9 · 10 | 0 · 0 | 6 · 7 | 0 · 0 |

- 양손가락 X 인데 면제된 call(LI 16 · LIBERO 10 9)에서 robot–other 접촉과 SafeLIBERO 충돌은 창 안에서 0 이다. gripper–target 접촉은 LI 15 · LIBERO 10 9 call 에서 있고 target 이 10 mm 이상 수평으로 움직인 call 은 10 · 6 개다. 같은 조건의 비교 version 에서도 비슷한 규모(LI 허용판 20 · 수정판 18 · 면제판 12 / LIBERO 10 6 · 7 · 6)로 있다.
- 창 안 충돌 1 건은 양손가락 O 인 면제 call 에 있다. 조건 call 의 창 안 충돌 수는 LI 에서 허용판 2 (양손가락 O 1 · X 1) · 수정판 2 · 면제판 1 (O 1)이다.
- 면제 call 의 AG3S centroid − 실제 target 거리 median 은 LI 36.8 mm · LIBERO 10 36.3 mm 다(조건 call 전체의 median 은 LI 387.2 mm · LIBERO 10 38.5 mm).

#### I6 — 사례

기준: (a) VLA-only official F → 새 version 성공, (b) safe F → T(SafeLIBERO 만), (c) grasp zone 허용판이 over-lift(최고 rise > 250 mm)로 실패했는데 새 version 성공. 세 기준에 걸리는 episode 는 모두 같은 3개다 — LI t0e0 · LI t0e3 · LIBERO 10 t5e0. 각각 `outputs/verify/SL5i-G/cases/<set>_<ep>/` 에 case 묶음이 있다(조건 vla_only · 허용판 · 수정판 · 면제판의 `case.json` · call 표 · frame).

![같은 episode 의 이전 판과 새 판: 쥔 순간과 최고점](figures/sl5i-scene-grasp-moment.png)

(figure: 위 LI t0e0, 가운데 LI t0e3, 아래 LIBERO 10 t5e0. 열 = 허용판 key obs · 새 판 key obs · 허용판 최고점 · 새 판 최고점. + target, o gripper, x AG3S manipulated centroid.)

| 사례 | 조건 | official | collided | 최고 rise (mm) | 첫 lift obs | 첫 held (seq · obs) | 허용판과 처음 갈라지는 지점 |
|---|---|---|---|---|---|---|---|
| **LI t0e0** (sticky reach) | VLA-only | 실패 | 예 | 8.1 | – | – | – |
| | grasp zone 허용판 | 실패 | 아니오 | 515.6 | 88 | 없음 | – |
| | 파지 인식 수정판 | **성공** | 아니오 | 94.3 | 88 | 16 · 100 | env step 100 (call 16) |
| | 파지 인식 + closing 면제판 | **성공** | 아니오 | 94.3 | 88 | 16 · 100 | env step 75 (call 11) |
| **LI t0e3** (closing 면제) | VLA-only | 실패 | 예 | 150.3 | 120 | – | – |
| | grasp zone 허용판 | 실패 | 아니오 | 320.7 | 92 | 15 · 95 | – |
| | 파지 인식 수정판 | 실패 | 아니오 | 320.7 | 92 | 15 · 95 | 허용판과 action 이 같다 |
| | 파지 인식 + closing 면제판 | **성공** | 아니오 | 106.9 | 92 | 15 · 95 | env step 80 (call 12) |
| **LIBERO 10 t5e0** (drop-from-open) | VLA-only | 실패 | – | 173.5 | 63 | – | – |
| | grasp zone 허용판 | 실패 | – | 689.0 | 59 | 없음 | – |
| | 파지 인식 수정판 | **성공** | – | 133.7 | 59 | 10 · 60 | env step 60 (call 10) |
| | 파지 인식 + closing 면제판 | **성공** | – | 133.7 | 59 | 10 · 60 | env step 60 (call 10) |

- **LI t0e0:** 허용판은 obs 100 에서 이미 양손가락이 bowl 에 닿고 rise 90.5 mm 인데 latch 가 closing 이고 실패 check 는 reach 다(centroid − target 79.6 mm). 수정판은 같은 obs 에서 held 이고, 최고 rise 가 515.6 → 94.3 mm 로 줄었다. 면제판은 env step 75(call 11)에서 먼저 갈라지고 held 는 같은 seq 16 이다.
- **LI t0e3:** 수정판은 허용판과 action 이 같다(규칙 ①이 적용될 call 이 없다). 면제판만 call 12(obs 80)에서 갈라진다. 그 call 은 latch 가 closing 이고 실패 check 가 settled 뿐이며, 양손가락은 X(rise −1.1 mm)이고 빠진 행은 42 개다. 이후 첫 held 는 허용판과 같은 seq 15 이고 최고 rise 가 320.7 → 106.9 mm 로 줄었다.
- **LIBERO 10 t5e0:** 허용판은 obs 60 에서 양손가락 O 이고 rise 41.9 mm 인데 실패 check 가 `dropped` 이고(centroid − target 25 mm), 끝내 held 가 되지 못해 최고 rise 689.0 mm 로 over-lift 했다. 수정판 · 면제판은 같은 obs 60 에서 held 가 된다.

![SL5i-G 표 I0 ~ I6](figures/sl5i-tables.png)

#### 측정하지 않은 것

- cuRobo regression baseline(GPU 금지).
- 수정판 · 면제판 run 의 bit 재현(각 한 번만 돌렸다).
- 면제된 행의 TO row 수준 constraint 정체(calls log 는 count 만 남긴다). MuJoCo contact 창을 대리 지표로 썼다.
- latch 의 `manipulated_id` 가 가리키는 MuJoCo body(AG3S centroid − target 거리를 대리 지표로 썼다).

**판정 대기 (사용자).** 위는 version 별 숫자만 남긴 것이다. 규칙(sticky reach · drop-from-open · closing 면제)의 효과에 대한 해석과 다음 수정 방향은 사용자 판정 전이다.

---

## 문제 · 해결 정리 문서 신설 (2026-10-23)

사용자 지시(2026-10-23)로 지금까지 benchmark 에서 겪은 문제와 해결 방법을 문제별로 다시 모은 문서를 새로 만들었다: [`AG3S_BENCHMARK_ISSUES.md`](AG3S_BENCHMARK_ISSUES.md) (환경 · Panda adapter · 평가 · 알고리즘 개선판 · 미해결, 37 항목). 이 로그가 시간순 정본이고, 새 문서는 이 로그와 `handoff/*.verify.json` 에서 옮긴 요약 · 색인이다.

---

# Panda 최종 구성 SafeLIBERO smoke 기록 (2026-10-29 ~ 10-30) — 소급 기록

> **작성 경위 (2026-11-04, scribe).** 아래 8 개 절은 `outputs/2026-10-29_*` · `outputs/2026-10-30_*` 폴더의 README · summary.json 을 근거로 뒤늦게 옮긴 것이다. 숫자는 각 절에 적은 source 파일에서만 가져왔고, 계산하거나 보충하지 않았다. 원본 판정 규칙 · 사용자 판정은 [`handoff/SL6.task.md`](handoff/SL6.task.md) 의 §0 · §6 · §8 · §9 · §10 · §11 에 결과 전에 고정되어 있다. `outputs/` 는 git 에서 제외되므로 아래 `../../../outputs/...` 링크(README · summary.json · json)와 **영상(mp4)은 이 checkout 의 local 에만 있고** 추적되지 않는다. 그림(png)은 `figures/benchmark/<폴더 이름>/` 아래로 옮겨 링크한다.
> 모든 smoke 는 같은 장면이다: SafeLIBERO spatial Level I dev 11 episode (t1e1, t1e4, t2e0–e4, t3e0, t3e2–e4), seed 7, GPU 독점 lock, 동시 1 job. 같은 11 episode 를 반복해서 쓰므로 1 – 2 episode 차이는 우연과 구분되지 않는다 (각 README 한계 항목).
> 원인 층 약어는 README 의 정의를 따른다: **ATS** = target 분할 · 추적 (VLA attention 과 depth 로 target 을 고르는 단계), **HSR** = 작업 단계 · 쥠 판단, **LTE** = 링크별 target 제외 · 거리장, **CCR** = TrajOpt 궤적 수정 · HOLD.

### 이 묶음에서 새로 쓰는 말

| 용어 | 뜻 (출처: 각 README) |
|---|---|
| **Panda 최종 구성** | AG3S 최종 구성을 Panda 로 옮긴 판. HSR6-Panda probe, 손바닥 껍질 구, 변환 C, 연속성 · 꺾임 변화율 항 처리가 판마다 다르다 (각 절에서 정의) |
| **safe success** | official success 이면서 장애물을 움직이지 않은 episode (실제 MuJoCo 로 판정). 이 묶음의 주 지표 |
| **HOLD call** | SafePolicy 가 action 을 0 으로 내보낸 policy call. 표에서 `HOLD call / 전체 call` 로 쓴다. 이 값이 늘어도 보호를 뜻하지 않는다 (덜 움직여 충돌이 줄어든 결과일 수 있다) |
| **쥔 bowl 미인식** | MuJoCo 에서 target bowl 이 2 cm 넘게 들리고 손과 9 cm 안인 frame 이 20 개를 넘는데, 첫 닫기 뒤 grasp latch 가 CLOSING / LATCHED 에 머문 call 이 5 개를 넘는 episode ([2판 README](../../../outputs/2026-10-30_Panda최종구성2판_쥠인식pb_SafeLIBERO-LI-dev11/README.md) 한계) |
| **포화 action step** | 실행 action 의 arm 성분 하나라도 \|a\| ≥ 0.999 인 step (이 로그 용어 절의 saturation 과 같다). 3판 이후 README 에서 함께 보고 |
| **pb1 · pb6 · pb7 (README 설명)** | pb1: reach 검사가 가려진 target 중심을 손과 함께 옮긴다. pb6: 손이 닫힌 뒤 reach 판정을 유지하고 손을 열 때만 latch 를 푼다. pb7: CLOSING 동안 쥔 물체 위의 손 점을 충돌 대상에서 뺀다 (closing 면제) |
| **E0 TrajOpt** | AG3S-PRE-CORE-TO 계획의 TrajOpt 기준선 (GPU 여러 초기값 · 최소 개입 gate 이전 상태). 3판의 TrajOpt 를 이것으로 바꾼 판이 SL6-F 다 |

---

## SL6 — Panda 최종 구성 첫 판: 기하 grasp 보정 · 쥐기 목표 항 포함 / 제거 smoke (2026-10-29)

- 실행 2026-10-29 23:34 – 2026-10-30 00:41 KST, commit `a43faee3`. README 작성 2026-10-30 01:20 KST. source: [README](../../../outputs/2026-10-29_Panda최종구성_G1A4포함제거_SafeLIBERO-LI-dev11/README.md) · `summary.json` (같은 폴더). 판정 규칙은 결과 전 고정: [`handoff/SL6.task.md`](handoff/SL6.task.md) §6.

### 목적 → 변경

AG3S 최종 구성을 Panda 로 옮긴 뒤, 두 목적함수 항을 Panda 최종 구성에 넣을지 한 번의 smoke 로 정한다.
- **G1 (기하 grasp 보정):** VLA 가 닫으려는 곳 근처에서 점군으로 확인한 grasp 자세로 손바닥 목표를 고친다.
- **A4 (쥐기 목표 항):** VLA 가 처음 닫는 위치로 손바닥을 붙잡아 둔다.

두 판 모두 Panda 최종 구성이다: HSR6-Panda probe, 명사 attention `noun_target:L8H6`, 손가락 이웃 접촉 허용만 적용하는 범위(scope `nbc`), 쥐기 전 연속성 · 꺾임 변화율 항 끔, 변환 C. 차이는 G1 · A4 flag 6 개뿐이다 (dry-run 확인, README).
- **제거판** (`AG3S-final-panda`): G1 · A4 꺼짐. **작동판** (`AG3S-final-panda-g1a4`): 켜짐.
- 이전 Panda 개선(pb1–pb7)은 넣지 않았다 (사용자 판정: "이전과 동일한 이유로 실패하면 추가").
- 사전 확인 두 가지 통과 (README): G0 = 기존 V11AB-pb7 이 merge 검증 run 과 550/550 array bit 동일, G2 = 최종 구성 서버가 1 episode 를 정상 실행.

### 결과 (README 자체 보고 · `summary.json`)

| 조건 | 안전 성공 | 공식 성공 | 충돌 | HOLD call |
|---|---|---|---|---|
| VLA 단독 (SL5f) | 2 | 11 | 9 | – |
| pb7 · 이전 Panda (SL5i-G) | 1 | 1 | 5 | 121 / 580 |
| 최종 · G1A4 제거 | **0** | 0 | 3 | 167 / 616 |
| 최종 · G1A4 작동 | **1** | 1 | 3 | 171 / 581 |

판정 규칙(작동판 안전 성공 ≥ 제거판 + 2 이고 충돌이 늘지 않으면 포함): 1 < 0 + 2 이므로 **제거** (README).

![결과표](figures/benchmark/2026-10-29_Panda최종구성_G1A4포함제거_SafeLIBERO-LI-dev11/결과표.png)

![조건별 안전 성공 · 충돌, target 선택 오차, 쥔 뒤 TrajOpt 위쪽 수정 합](figures/benchmark/2026-10-29_Panda최종구성_G1A4포함제거_SafeLIBERO-LI-dev11/그래프.png)

episode 별 결과 (`summary.json`):

| episode | VLA 단독 | pb7 | 제거판 | 작동판 |
|---|---|---|---|---|
| t1e1 | 안전 성공 | 안전 성공 | 시간초과 | 안전 성공 |
| t1e4 | 안전 성공 | 시간초과+충돌 | 시간초과 | 시간초과 |
| t2e0 | 성공+충돌 | 시간초과+충돌 | 시간초과 | 시간초과 |
| t2e1 · t2e3 · t2e4 | 성공+충돌 | 시간초과 | 시간초과 | 시간초과 |
| t2e2 | 성공+충돌 | 시간초과 | 시간초과+충돌 | 시간초과+충돌 |
| t3e0 · t3e4 | 성공+충돌 | 시간초과+충돌 | 시간초과+충돌 | 시간초과+충돌 |
| t3e2 · t3e3 | 성공+충돌 | t3e2 시간초과 · t3e3 시간초과+충돌 | 시간초과 | 시간초과 |

("성공+충돌" = official success 이지만 장애물과 접촉. 11 episode 중 8 개는 두 판이 bit 단위로 같다. 갈린 것은 A4 가 켜진 t1e1 · t2e1 · t2e2 뿐이다 — README.)

#### G1 · A4 작동 횟수 (작동판, README)

| 항목 | 값 |
|---|---|
| G1 이 손바닥 목표를 채운 call | **0 / 581** |
| G1 후보 생성 | 249 회 중 241 회 "no antipodal pair" (두 손가락이 맞물릴 면 쌍 없음). 유효 grasp 가 있던 23 call 은 모두 closing · held 단계라 목표를 내지 않음 |
| A4 가 켜진 call | 21 (t1e1 4 · t2e1 13 · t2e2 4) |
| A4 call 의 HSR6 단계 | grasp 5 · lift 9 · approach 3 · 기록 없음 4 |
| scope `all` 이었다면 | 기록이 있는 17 call 중 12 call 이 막혔을 것 |

**이 smoke 는 G1 을 시험하지 못했다** (한 번도 켜지지 않음). 판정은 사실상 A4 하나의 효과다 (README).

#### 실패 원인 (제거판 기준, README)

| 원인 | 증상이 보이는 곳 | 원인 구성 요소 | episode | 근거 |
|---|---|---|---|---|
| 다른 물체를 target 으로 고름 | target 중심이 VLA 단독이 쥔 위치에서 223–503 mm | ATS | t2e0 · t2e3 · t3e0 · t3e2 · t3e3 · t3e4 (6) | 실제 장면에서 다른 bowl 이나 ramekin |
| target 을 못 찾음 → 전 구간 HOLD | 56/56 call `no_target` | ATS (grounding `no_admissible`) | t1e4 · t2e4 (2); t3e3 은 뒤 50 call | |
| 실제로 쥔 bowl 을 latch 가 인식 못함 → TrajOpt 가 손을 위로 밈 | bowl 이 손과 약 6 cm 를 유지하며 135–240 frame 들림, 최고 0.33–0.65 m (VLA 단독 약 0.17 m) | HSR (latch 가 CLOSING 에 머묾) → CCR | t1e1 · t2e0 · t2e1 · t2e2 · t2e3 (5) | t1e1 제거판 Σ(실행 z − VLA z) = +94, VLA 자체 +16 |
| 충돌 | – | 미진단 | t2e2 (milk) · t3e0 (wine bottle) · t3e4 (mug) | t3e0 · t3e4 는 pb7 도 같은 장애물과 충돌 |

README 는 HSR → CCR 실패가 pb6(파지 인식 수정)과 pb7(closing 면제)이 고친 원인과 같다고 기록한다.

#### 이전 대비 (before → after, README)

| 기준 | 안전 성공 | 공식 성공 | 충돌 | 비고 |
|---|---|---|---|---|
| pb7 → 제거판 | 1 → 0 | 1 → 0 | 5 → 3 | 충돌이 준 것은 시간초과 · HOLD 로 덜 움직인 결과 (HOLD ≠ 보호) |
| target 선택 오차 50 mm 이내(첫 기록) | 5/11 → 3/11 | | | 못 찾음 0 → 2. 명사가 모두 "bowl" 이라 같은 bowl 둘을 구분하지 못한다 (head 재선택 때 미리 적은 한계) |

![실제 장면: target 선택](figures/benchmark/2026-10-29_Panda최종구성_G1A4포함제거_SafeLIBERO-LI-dev11/실제장면_target선택.png)

(head 카메라 frame 25. ★ VLA 단독이 쥔 곳 · ● pb7 target · ✖ 최종 구성 target · ▲ 장애물 · ✚ 손 — README.)

![실제 장면: bowl 상승](figures/benchmark/2026-10-29_Panda최종구성_G1A4포함제거_SafeLIBERO-LI-dev11/실제장면_bowl상승.png)

#### 측정하지 않은 것

- G1 의 효과: 한 번도 켜지지 않아 미측정.
- 충돌 3 건의 원인 층: 미진단. HSR6 단계는 call 기록에 직접 남지 않아 A4 call 의 단계는 attach-assist 기록이 남은 17 call 로만 셌다.
- 요청 수 기반 parameter 의 시간 환산: Panda 요청 간격이 0.25 s 로 RB-Y1 (0.53 s) 의 절반이라 환산하지 않았다 (SL6-A 기록, README).

**읽는 법.** 규칙은 G1 · A4 를 제거하라고 나왔다. 그러나 이 숫자만으로는 두 항의 효과가 아니라 (1) 한 episode 차이 (t1e1) 와 (2) G1 이 시험되지 않았다는 사실이 남는다. 같은 smoke 에서 최종 구성은 pb7 보다 안전 성공 · 공식 성공이 모두 낮았고, 실패 원인은 target 선택 오류(ATS) 와 쥔 bowl 미인식 → TrajOpt 위쪽 수정(HSR → CCR) 두 갈래로 나뉘었다. 판정 뒤 사용자 판정(2026-10-30 ~01:30 KST, `SL6.task.md` §0)은 G1 · A4 제거를 확정하고 pb1 · pb6 · pb7 을 더하며 target 신호를 action attention 으로 되돌렸다. 이 결정 자체가 개선인지는 다음 절들의 숫자가 보여 준다.

---

## SL6-C — Panda 최종 구성 2판: 쥠 인식 수정 + action attention (2026-10-30)

- 실행 2026-10-30 02:50 – 03:16 KST, commit `1fb7a0d1`, README 작성 03:40 KST. GPU lock 은 01:23 KST 부터 기다렸다 (development 의 다른 작업이 사용 중). source: [README](../../../outputs/2026-10-30_Panda최종구성2판_쥠인식pb_SafeLIBERO-LI-dev11/README.md) · `summary.json` · `target_at_close.json`. 판정 규칙: [`handoff/SL6.task.md`](handoff/SL6.task.md) §8 (commit `342f6df1`).

### 목적 → 변경 (첫 판 → 2판)

첫 판 실패의 두 원인(쥔 bowl 미인식, 명사 attention 으로 인한 target 오류)을 같은 11 episode 에서 고친 판이 pb7 보다 나은지 본다. 변경 (사용자 판정 2026-10-30, README):
1. G1 · A4 제거 확정.
2. 쥠 인식 수정 세 가지 추가: pb1 (reach 검사가 가려진 target 중심을 손과 함께 옮김), pb6 (닫힌 뒤 reach 판정 유지), pb7 (CLOSING 중 쥔 물체 위의 손 점을 충돌 대상에서 뺌).
3. target 신호를 명사 attention (`L8H6`) 에서 action attention (`L8H2`, pb7 과 같음) 으로 되돌림.
- 나머지(HSR6-Panda probe, 쥐기 전 연속성 · 꺾임 변화율 항 끔, 변환 C, H 10 / K 5)는 첫 판과 같다. 기존 판 서버 인자 불변 (test 270 개 통과, skip 3 개; CPU 기동에서 pb1 · pb6 · pb7 켜짐, pb4 · pb5 꺼짐 확인 — README).
- 판정 규칙(결과 전 고정): 2판 안전 성공 ≥ pb7 (1) 이고 충돌 ≤ pb7 (5) 이면 Panda 최종 구성으로 삼는다.

### 결과 (README 자체 보고 · `summary.json`)

| 조건 | 안전 성공 | 공식 성공 | 충돌 | HOLD call | 닫는 시점 target 정답 (≤ 50 mm) | 쥔 bowl 미인식 |
|---|---|---|---|---|---|---|
| VLA 단독 | 2 | 11 | 9 | – | – | – |
| pb7 | 1 | 1 | 5 | 121 / 580 | 6 | 2 |
| 최종 1판 | 0 | 0 | 3 | 167 / 616 | 3 (5 개는 닫지 못함) | 5 |
| **최종 2판** | **1** | **2** | **6** | **38 / 574** | **5** (2 개는 닫지 못함) | **1** |

**판정: 미채택** (안전 성공 1 = 1, 충돌 6 > 5).

![결과표](figures/benchmark/2026-10-30_Panda최종구성2판_쥠인식pb_SafeLIBERO-LI-dev11/결과표.png)

![조건별 결과, target 선택 오차, 쥔 뒤 TrajOpt 위쪽 수정 합](figures/benchmark/2026-10-30_Panda최종구성2판_쥠인식pb_SafeLIBERO-LI-dev11/그래프.png)

episode 별 (`summary.json`): 2판 안전 성공 t2e3 · 성공+충돌 t2e0 · 충돌 시간초과 t1e1 · t1e4 · t3e2 · t3e3 · t3e4 · 충돌 없는 시간초과 t2e1 · t2e2 · t2e4 · t3e0. pb7 은 t1e1 이 안전 성공이었다.

### 이전 대비 (before → after)

| 지표 | 첫 판 → 2판 | pb7 → 2판 |
|---|---|---|
| 쥔 bowl 미인식 episode | 5 → 1 (남은 하나는 t1e1) | 2 → 1 |
| target 못 찾아 전 구간 HOLD | 2 → 0 | – |
| HOLD call | 167 → 38 | 121 → 38 |
| 안전 성공 | 0 → 1 | 1 → 1 (성공 장면이 t1e1 에서 t2e3 으로 바뀜) |
| 공식 성공 | 0 → 2 | 1 → 2 |
| 충돌 | 3 → 6 (첫 판의 충돌이 적었던 것은 덜 움직인 결과) | 5 → 6 |

### 2판의 실패 원인 (README, 진단 안 한 항목은 "미진단")

| episode | 2판 결과 | 관측된 사실 | 증상 / 원인 구성 요소 |
|---|---|---|---|
| t1e1 | 충돌 (storage box, frame 66) · 시간초과 | 첫 3 call 은 다른 물체를 target (310 mm), 4 번째부터 맞는 bowl (27 mm). latch 가 HELD 가 된 frame 65 에 수정량 0 → 1.02, frame 66 에 충돌 | ATS 일시 오류 뒤 HSR · CCR. 충돌 원인은 미진단 → 다음 절 진단 |
| t1e4 | 충돌 (mug, frame 112) | 첫 기록 target 은 다른 물체 (282 mm), 닫기 전 맞는 bowl (28 mm). VLA 단독은 안전 성공 | 미진단 → 다음 절 |
| t2e0 | 성공+충돌 (moka pot) | 닫는 시점 target 106 mm. VLA 단독 · pb7 도 같은 장애물과 충돌 | ATS (+ 충돌 미진단) |
| t2e1 | 시간초과 | bowl 이 0.08 m 까지만 들림. HOLD 33 call (모두 `uncertified`). latch 가 CLOSING / LATCHED 에 53 call. Σ(실행 z − VLA z) = +102 (VLA 자체 −120) | CCR 후보 (미진단) |
| t2e2 | 시간초과 | bowl 이 0.45 m 까지 들렸다 떨어짐. latch 가 PLACED 로 39 call 판정 (실제로 놓지 않음) | HSR 후보 (미진단) |
| t2e4 | 시간초과 | bowl 이 들리지 않음. latch CLOSING / LATCHED 50 call. Σ = +23 (VLA −27) | CCR 후보 (미진단) |
| t3e0 · e2 · e3 · e4 | 시간초과 1 · 충돌 3 (book · wine bottle · mug) | target 이 VLA 단독이 쥔 위치에서 276–487 mm | ATS (task 3 은 pb7 · 첫 판 · 2판 모두 target 을 틀림) |

충돌 6 개 분해 (README): task 3 의 target 오류 3 (t3e2 · t3e3 · t3e4) + VLA 단독도 충돌한 t2e0 1 + **VLA 단독은 안전했는데 2판이 만든 task 1 의 2 (t1e1 · t1e4)**.

target 선택: 2판은 pb7 과 같은 attention head 인데 첫 기록 call 이 pb7 과 다른 episode 가 있다 (t1e1 · t3e2 · t3e3). 원인은 확인하지 않았다 (README).

![실제 장면: target 선택](figures/benchmark/2026-10-30_Panda최종구성2판_쥠인식pb_SafeLIBERO-LI-dev11/실제장면_target선택.png)

![실제 장면: bowl 상승](figures/benchmark/2026-10-30_Panda최종구성2판_쥠인식pb_SafeLIBERO-LI-dev11/실제장면_bowl상승.png)

#### 측정하지 않은 것

- 충돌 원인 층: task 3 의 target 오류 외에는 진단하지 않았다 (t1e1 · t1e4 가 우선 — 다음 절).
- 첫 기록 call 이 pb7 과 달라진 이유.

**읽는 법.** 쥔 bowl 미인식과 target 못 찾음은 줄었고 HOLD 도 줄었으나, 안전 성공은 pb7 과 같고 충돌은 1 개 늘었다. 늘어난 충돌은 VLA 단독이 안전했던 task 1 에서 생겼다. 11 episode 에서 1 episode 차이라 우연과 구분되지 않는다는 점은 README 의 한계이며, 판정은 결과 전 규칙대로 미채택이었다.

---

## SL6-C-diag — task 1 충돌 진단: 2판 정의에서 빠진 pb7 기능과 포화 action (2026-10-30)

- README 작성 2026-10-30 06:50 KST. 저장된 기록만 사용 (CPU, GPU 없음, 새 closed-loop 실행 없음). 사용자 판정 2026-10-30: "task 1 충돌 진단". source: [README](../../../outputs/2026-10-30_task1충돌진단_Panda최종구성2판/README.md) · `summary.json`.

### 목적 → 방법

2판에서 늘어난 충돌(6 vs 5)이 VLA 단독이 안전했던 t1e1 · t1e4 에서 나왔다. 장애물에 무엇이 닿았고 그 직전에 어느 층이 무엇을 했는지 찾는다. 방법 (README):
1. **MuJoCo 접촉 재생:** 기록된 simulator 상태(`_geom.npz` `sim_state`)를 frame 마다 SafeLIBERO env 에 다시 넣어 장애물 geom 과 접촉한 몸체를 뽑는다. 대상은 2판 충돌 6 개 · pb7 충돌 5 개.
2. **substep 재현:** 기록 frame 에 접촉이 없는데 장애물이 움직인 경우(t1e1 · t3e2)에 그 frame 상태에서 기록된 action 으로 `env.step` 을 다시 돌려 물리 substep 25 개마다 접촉을 기록. bit 단위 재현이 아니다 (다음 frame 상태와의 최대 차이 qpos + qvel 0.003–0.23, 일부 frame 2.6–3.0) — 보조 근거.
3. **call 기록 비교:** 충돌 직전 TrajOpt 수정량, conversion C(관절 궤적 → 끝점 delta action 변환)의 포화 row, latch 상태, HOLD, 손 이동량(`obs_robot0_eef_pos/quat`).

### 결과

#### 1. 두 충돌의 직접 경로 (README)

경로: TrajOpt 가 정책 궤적을 15–25 mm 고친다 → 변환 C 가 이를 ±1 포화 action 으로 바꾼다 (회전 성분 포함) → 손이 VLA 보다 크게 움직인다 → 충돌.

| | t1e1 (storage box, frame 66) | t1e4 (mug, frame 112) |
|---|---|---|
| 닿은 몸체 | 들고 있던 target bowl (substep 재현) | 손 본체 `gripper0_right_gripper` (기록 frame) |
| 직전 call | call 9 (frame 65): latch CLOSING → HELD 전환 call | call 17 (frame 105, CLOSING) → call 18 (frame 110, HOLD) |
| TrajOpt 수정 | 끝점 20.1 mm, 관절 0.018 rad | 끝점 24.2 mm, 관절 0.028 rad (call 13 은 73.1 mm) |
| 변환 C | 포화 전 최대 수정량 3.96, 10 row 중 8 row 포화 | 3.69, 8 row 포화 (call 13: 19.9, 10 row 전부) |
| 손 이동 / step | 22.4 · 23.2 mm (VLA 단독 최대 13.9) | 11.8 · 28.7 mm; HOLD(action 0) 후에도 13.4 · 9.9 · 8.0 mm 계속 움직여 frame 112 에 mug 와 닿음 |

pb7 의 같은 t1e1 장면: HELD 가 된 call 의 수정 1.1 mm, 포화 없음. HELD 전 call 들은 최소 개입 gate 가 적용되어 기준 궤적과 장애물 사이가 30 mm 라 TrajOpt 가 궤적을 고치지 않았다 (README).
t1e1 추가 사실: HELD 가 된 순간 쥔 물체 모델(bowl 을 덮는 구)의 바닥이 지지면보다 **43.5 mm 아래** 였다 (pb7 은 같은 순간 44.5 mm 위). 위로 20 mm 고친 원인일 가능성이 있으나 어느 제약이 만들었는지는 TrajOpt 를 다시 풀어야 하며 하지 않았다 (GPU 필요).

#### 2. episode 전체 비교 (`summary.json` · README)

| | VLA 단독 | pb7 | 2판 |
|---|---|---|---|
| t1e1 포화 action step | 0 / 91 | 4 / 99 | **207 / 280** |
| t1e1 TrajOpt 가 고치지 않은 call (`optimal`) | – | 16 / 20 | 6 / 56 |
| t1e1 최소 개입 gate 적용 call | – | 16 / 20 | **0 (기능 없음)** |
| t1e4 포화 action step | 0 / 88 | 12 / 280 | **91 / 280** |
| t1e4 최소 개입 gate 적용 call | – | 10 / 56 | **0** |
| 11 episode 전체 포화 step | – | 955 / 2899 (33 %) | 1349 / 2866 (47 %) |

#### 3. 충돌마다 닿은 몸체

| 조건 | 팔 링크 (link5–7) | 손 · 손가락 | bowl (쥔 · 민 물체) |
|---|---|---|---|
| 2판 (충돌 6) | **4** (t2e0 link7 · t3e2 link7 · t3e3 link6 · t3e4 link7) | 1 (t1e4) | 1 (t1e1) |
| pb7 (충돌 5) | 2 (t3e0 · t3e3 link5) | 1 (t3e4) | 2 (t1e4 · t2e0) |

![충돌 11 개의 접촉 몸체, 2판에서 빠진 기능](figures/benchmark/2026-10-30_task1충돌진단_Panda최종구성2판/결과표_충돌몸체_빠진기능.png)

![step 당 손 이동 · 회전 (t1e1 · t1e4)](figures/benchmark/2026-10-30_task1충돌진단_Panda최종구성2판/그래프_손움직임.png)

![충돌 직전 4 frame 과 실행 action](figures/benchmark/2026-10-30_task1충돌진단_Panda최종구성2판/실제장면_충돌직전.png)

#### 4. 2판 정의에서 빠진 것 — 정의 단계의 실수 (README)

pb7 은 pb1–pb7 의 누적판이며 서버 flag 9 개다: `--reach-at-capture` · `--minimal-intervention-mm 10` · `--links arms` · `--held-hand-exempt` · `--latch-phase` · `--grasp-zone-mm 10` · `--grasp-reach-sticky` · `--grasp-drop-from-open` · `--closing-hand-exempt`. 2판에는 pb1 · pb6 · pb7 고유 flag 4 개만 넣어 다섯 기능이 빠졌다.

| 빠진 기능 | 뜻 | 이 진단과의 관련 |
|---|---|---|
| pb2 최소 개입 | 기준 궤적이 모든 row 에서 10 mm 넘게 떨어져 있으면 TrajOpt 가 고치지 않음 | task 1 에서 TrajOpt 가 거의 매 call 궤적을 고쳤다 |
| pb3 팔 링크 모델 | 2판은 `--links gripper` 라 팔 link1–7 이 충돌 모델에 없음 | 2판 충돌 4 개가 팔 링크 |
| pb4 HELD 손 면제 | HELD 동안 쥔 물체 위의 손 점을 충돌 대상에서 뺌 | – |
| pb5 latch 단계 · 쥐기 영역 | 접촉 단계를 latch 에서 정하고 target 주변 10 mm 접근을 허용 | – |

### 원인 층 (README)

| episode | 증상 위치 | 원인 구성 요소 |
|---|---|---|
| t1e1 | HELD 전환 call 의 포화 action 으로 쥔 bowl 이 storage box 에 닿음 | CCR. 수정을 만든 쥔 물체 모델 위치는 HSR · LTE 후보 (미확인) |
| t1e4 | CLOSING 중 수정이 포화 action → HOLD 가 손 관성을 못 멈춰 손 본체가 mug 에 닿음 | CCR |

#### 측정하지 않은 것

- TrajOpt 의 수정 이유(어느 제약 · 비용): 기록에 없고 다시 풀어야 한다.
- 최소 개입 gate 나 팔 링크 모델이 이 충돌을 막는지: 실행이 필요해 미확인.
- t1e1 "bowl–box 접촉" 은 재현 기반. 기록 frame 만으로 확인된 사실은 frame 66→67 에 box 가 1.2 cm 움직였고, 그때 손은 box 에서 27 cm 떨어져 있었으며 bowl 은 손에 들려 있었다는 것.

**읽는 법.** 2판은 pb7 누적판에서 다섯 기능이 빠진 채 정의되었고, 그 상태에서 task 1 의 두 충돌은 모두 TrajOpt 수정 → 변환 C 포화 action 직후에 일어났다. 이 진단은 빠진 기능이 충돌을 막는지는 확인하지 않았다. 사용자 판정(2026-10-30 ~07:00 KST, `SL6.task.md` §0)에 따라 다음 3판은 pb7 의 flag 9 개를 모두 넣고 연속성 · 꺾임 변화율 항을 끈다.

---

## SL6-E — Panda 최종 구성 3판: pb7 전체 + 연속성 · 꺾임 변화율 항 전 구간 끔 (2026-10-30)

- 실행 08:27–08:44 KST (task 1 · 2), 10:02–10:12 KST (task 3 재실행), commit `665bcc6e`. README 작성 10:45 KST. task 3 의 첫 실행은 08:45 KST 에 세션이 끝나며 중단되었고(episode 기록 0) 같은 설정으로 다시 돌린 run 만 썼다. source: [README](../../../outputs/2026-10-30_Panda최종구성3판_pb7전체_연속성꺾임끔_SafeLIBERO-LI-dev11/README.md) · `summary.json`. 판정 규칙: [`handoff/SL6.task.md`](handoff/SL6.task.md) §9 (commit `665bcc6e`).

### 목적 → 변경 (2판 → 3판)

앞 절의 두 확인(2판에 pb7 누적판의 pb2–pb5 가 빠졌고, t1e1 의 충돌 직전 수정이 연속성 · 꺾임 항이 켜지는 HELD 전환 call 에서 나왔다)을 고친 판이 pb7 보다 나은지 본다 (사용자 판정 2026-10-30 ~07:00 KST).
- pb7 의 flag 9 개를 모두 넣었다 (최소 개입 gate 10 mm · 팔 link1–7 충돌 모델 · HELD 손 면제 · latch 단계 · 쥐기 영역이 새로 들어감).
- 연속성 항과 꺾임 변화율 항을 **전 구간에서 껐다** (무게 0).
- 그대로: action attention, HSR6-Panda probe, 손바닥 껍질 구, 변환 C. CPU 기동 확인: pb1–pb7 모두 켜짐, 두 항 무게 0, 제약 구 170 개 (손 133 + 팔 37).
- 판정 규칙(결과 전 고정): 3판 안전 성공 ≥ pb7 (1) 이고 충돌 ≤ pb7 (5) 이면 채택.

### 결과 (README 자체 보고 · `summary.json`)

| 조건 | 안전 성공 | 공식 성공 | 충돌 | HOLD call | 닫는 시점 target 정답 | 쥔 bowl 미인식 | 포화 action step |
|---|---|---|---|---|---|---|---|
| VLA 단독 | 2 | 11 | 9 | – | – | – | – |
| pb7 | 1 | 1 | 5 | 121 / 580 | 6 | 2 | 955 |
| 최종 2판 | 1 | 2 | 6 | 38 / 574 | 5 | 1 | 1349 |
| **최종 3판** | **1** | **1** | **7** | 39 / 600 | 4 | 3 | 1194 |

**판정: 미채택** (충돌 7 > 5). 안전 성공은 t2e2 하나(VLA 단독 · pb7 · 2판 모두 실패한 장면). t1e1 의 storage box 충돌은 없어졌지만 시간이 다 되었다. task 3 은 네 episode 모두 충돌했다 (2판 3, pb7 3).

![결과표](figures/benchmark/2026-10-30_Panda최종구성3판_pb7전체_연속성꺾임끔_SafeLIBERO-LI-dev11/결과표.png)

![조건별 결과, target 선택 오차, 쥔 뒤 TrajOpt 위쪽 수정 합](figures/benchmark/2026-10-30_Panda최종구성3판_pb7전체_연속성꺾임끔_SafeLIBERO-LI-dev11/그래프.png)

#### 충돌마다 장애물에 닿은 몸체 (기록 재생, README)

| 조건 | 팔 링크 | 손 · 손가락 | bowl | 합 |
|---|---|---|---|---|
| pb7 | 2 (link5 ×2) | 1 | 2 | 5 |
| 2판 (팔 모델 없음) | 4 (link6 · 7) | 1 | 1 | 6 |
| **3판 (팔 모델 있음)** | **3** (t2e0 link7 · t3e0 link5 · t3e2 link6) | **3** (t1e4 손가락* · t3e3 손* · t3e4 손) | **1** (t2e3) | 7 |

\* 기록 frame 에 접촉이 없어 substep 재현으로 찾았다 (bit 단위 재현 아님).
팔 링크를 충돌 모델에 넣어도 팔 접촉이 남았다 (pb7 도 같다). 남은 접촉의 원인은 보지 않았다. README 가 든 후보: 거리장에 장애물 표면이 없음(가림) / 실행 action 이 계획과 달랐음(포화) / TrajOpt 가 위반을 못 풂.

#### 실패 원인 (README)

| 원인 | 증상 / 원인 구성 요소 | episode |
|---|---|---|
| 처음 닫는 시점에 다른 물체를 target 으로 함 | ATS | t1e4 · t2e0 · t2e3 · t3e0 · t3e2 · t3e3 · t3e4 (7). task 3 은 네 조건 모두 틀림 |
| 쥔 bowl 미인식, TrajOpt 가 손을 위로 밈 | HSR → CCR | t1e1 · t2e1 · t2e3 (3; 2판은 1) |
| 충돌 없는 시간초과 | 미진단 | t2e4 |

같은 action attention 인데 처음 닫는 시점의 target 이 2판과 다른 episode 가 있다: t2e0 은 2판 106 mm → 3판 463 mm, t2e3 은 2판 34 mm (안전 성공) → 3판 513 mm (충돌). closed-loop 궤적이 갈라지면 관측과 target 선택이 달라진다 (README).

![실제 장면: target 선택](figures/benchmark/2026-10-30_Panda최종구성3판_pb7전체_연속성꺾임끔_SafeLIBERO-LI-dev11/실제장면_target선택.png)

![실제 장면: bowl 상승](figures/benchmark/2026-10-30_Panda최종구성3판_pb7전체_연속성꺾임끔_SafeLIBERO-LI-dev11/실제장면_bowl상승.png)

#### 이전 대비 (before → after)

| 기준 | 안전 성공 | 충돌 | HOLD | 포화 step | 쥔 bowl 미인식 |
|---|---|---|---|---|---|
| pb7 → 3판 | 1 → 1 | 5 → 7 | 121 → 39 | 955 → 1194 | 2 → 3 |
| 2판 → 3판 | 1 → 1 | 6 → 7 | 38 → 39 | 1349 → 1194 | 1 → 3 |

2판 → 3판에서 바뀐 episode: t1e1 (충돌 → 충돌 없음), t2e2 (시간초과 → 안전 성공), t2e3 (안전 성공 → 충돌). 최소 개입 gate 는 2판보다 TrajOpt 가 덜 고치게 했지만 포화 step 이 pb7 수준까지 줄지는 않았다 (README).

#### 측정하지 않은 것

- 남은 팔 링크 접촉의 원인.
- substep 재현은 보조 근거. t1e4 · t3e3 의 다음 frame 상태 최대 차이는 0.004–0.35, 첫 frame 은 1.3. 기록 frame 에서 확인된 첫 접촉은 t1e4 가 frame 58 에 오른손가락이고, t3e3 은 기록 frame 에 접촉이 없다.

**읽는 법.** pb7 의 flag 를 모두 넣고 두 항을 끈 3판은 충돌이 7 개로 pb7 보다 많았고, 안전 성공 장면이 t2e2 로 바뀌었다. 한 episode 의 결과가 closed-loop 분기(target 선택 포함)에 따라 바뀌는 폭이 커서 1–2 episode 차이는 우연과 구분되지 않는다. 판정은 결과 전 규칙대로 미채택이다.

---

## SL6-F — 3판 + E0 TrajOpt (GPU 여러 초기값 · 최소 개입 없음, 연속성 · 꺾임 항은 쥔 뒤에만) (2026-10-30)

- 실행 10:19–10:47 KST, commit `d53745f5`, README 작성 11:10 KST. source: [README](../../../outputs/2026-10-30_Panda최종구성3판_TO-E0_SafeLIBERO-LI-dev11/README.md) · `summary.json`. 판정 규칙: [`handoff/SL6.task.md`](handoff/SL6.task.md) §10 (commit `d53745f5`).

### 목적 → 변경 (3판 → E0판)

3판이 충돌 7 로 미채택이었으므로, 같은 구성에서 **TrajOpt 만 E0 버전**(AG3S-PRE-CORE-TO 계획의 TrajOpt 기준선; 이후 단계에서 E1 쥐기 목표 항, E2 · E3 여러 초기값이 더해졌고 E0 는 그 이전 상태)으로 바꾸면 나아지는지 본다 (사용자 지시, ~10:10 KST).
- 뺐다: GPU 여러 초기값 (위반이 남은 chunk 를 초기값 16 개로 다시 풂), 최소 개입 gate (pb2, 10 mm).
- 연속성 항 · 꺾임 변화율 항: 3판은 전 구간 끔이었으나 E0판은 **쥔 뒤 (HELD 부터) 에만** 켠다. 무게는 V11AB 기본 0.5 · 20.
- 그대로: 로봇 충돌 모델 (팔 link1–7 + 손 + 손바닥 껍질 구), 쥠 판단 · 접촉 규칙 (pb1 · pb4–pb7, HSR6), 거리장 · target 제외, action attention, 변환 C.
- 검증: CPU 기동 test, launcher dry-run 으로 서버 인자 확인, tests/benchmark 274 통과, skip 3 (README).
- 판정 규칙(결과 전 고정): E0판 안전 성공 ≥ 3판 이고 충돌 ≤ 3판 이면 E0 TrajOpt 채택 (같으면 더 단순한 쪽).

### 결과 (README 자체 보고 · `summary.json`)

| 조건 | 안전 성공 | 공식 성공 | 충돌 | HOLD call | 닫는 시점 target 정답 | 쥔 bowl 미인식 | 포화 action step |
|---|---|---|---|---|---|---|---|
| VLA 단독 | 2 | 11 | 9 | – | – | – | 0 |
| pb7 | 1 | 1 | 5 | 121 / 580 | 6 | 2 | 955 |
| 최종 3판 | 1 | 1 | 7 | 39 / 600 | 4 | 3 | 1194 |
| **E0판** | **1** | **1** | **5** | 50 / 615 | 6 | 5 | 1621 |

**판정: E0 TrajOpt 채택** (안전 성공 1 ≥ 1, 충돌 5 ≤ 7). 숫자는 pb7 과 같다 (1 · 1 · 5).

![결과표](figures/benchmark/2026-10-30_Panda최종구성3판_TO-E0_SafeLIBERO-LI-dev11/결과표.png)

![조건별 결과, target 선택 오차, 쥔 뒤 TrajOpt 위쪽 수정 합](figures/benchmark/2026-10-30_Panda최종구성3판_TO-E0_SafeLIBERO-LI-dev11/그래프.png)

#### 3판 → E0판 전이 (11 episode 중 5 개가 바뀜, README)

| episode | 3판 | E0판 |
|---|---|---|
| t2e2 | 안전 성공 | 시간초과 |
| t2e3 | 충돌 (wine bottle) | **안전 성공** |
| t2e4 | 시간초과 | 충돌 (mug) |
| t3e0 | 충돌 (wine bottle) | 시간초과 |
| t3e3 | 충돌 (wine bottle) | 시간초과 |

task 3 의 두 충돌이 시간초과로 바뀐 것은 target 을 여전히 틀린 상태에서 덜 움직인 결과이며 보호로 해석하지 않는다 (README).

#### 충돌마다 장애물에 닿은 몸체 (기록 재생, README)

| 조건 | 팔 링크 | 손 · 손가락 | bowl | 합 |
|---|---|---|---|---|
| pb7 | 2 | 1 | 2 | 5 |
| 3판 | 3 | 3 | 1 | 7 |
| **E0판** | **1** (t3e2 link5) | **2** (t2e4 · t3e4 손) | **2** (t1e4*, t2e0) | 5 |

\* substep 재현으로 찾았다 (기록 frame 에 접촉 없음; 재현 상태 차이 0.04–2.3, 보조 근거).

#### 실패 원인 (README)

| 원인 | 증상 / 원인 구성 요소 | episode |
|---|---|---|
| 다른 물체를 target 으로 | ATS | t1e4 (첫 기록만) · t2e0 · t3e0 · t3e2 · t3e3 · t3e4. task 3 은 네 조건 모두 틀림 |
| 쥔 bowl 미인식, TrajOpt 가 손을 위로 밈 | HSR → CCR | t1e1 · t1e4 · t2e0 · t2e1 · t2e4 (5; 3판은 3) |
| 충돌 없는 시간초과 | 미진단 | t2e2 |

포화 action step 은 1194 → 1621 로 늘었다. README 의 설명: 쥔 뒤 연속성 · 꺾임 항이 켜지고 최소 개입이 없어 TrajOpt 가 더 자주 고쳤다 (원인 확인 실험은 없음). 그래도 충돌은 줄었다.

![실제 장면: target 선택](figures/benchmark/2026-10-30_Panda최종구성3판_TO-E0_SafeLIBERO-LI-dev11/실제장면_target선택.png)

![실제 장면: bowl 상승](figures/benchmark/2026-10-30_Panda최종구성3판_TO-E0_SafeLIBERO-LI-dev11/실제장면_bowl상승.png)

#### 이전 대비 (before → after)

| 기준 | 안전 성공 | 공식 성공 | 충돌 | HOLD | 쥔 bowl 미인식 | 포화 step |
|---|---|---|---|---|---|---|
| 3판 → E0판 | 1 → 1 | 1 → 1 | 7 → 5 | 39 → 50 | 3 → 5 | 1194 → 1621 |
| pb7 → E0판 | 1 → 1 | 1 → 1 | 5 → 5 | 121 → 50 | 닫는 시점 target 정답 6 = 6 | 955 → 1621 |

README 요약: 지금까지 Panda 최종 구성의 어느 판도 pb7 보다 나은 결과를 내지 못했다. 가장 큰 남은 원인은 target 선택(ATS, 특히 task 3)과 쥔 bowl 미인식(HSR)이다.

#### 측정하지 않은 것

- 팔 · 손 접촉이 왜 남는지(충돌별 원인 층).
- substep 재현은 bit 단위가 아니다.
- 3판과 E0판 사이에서 5 episode 가 바뀐 것에서 TrajOpt 차이와 closed-loop 분기를 분리하지 않았다.

**읽는 법.** TrajOpt 만 E0 로 바꾼 판은 충돌이 7 → 5 로 줄어 규칙상 채택되었지만, 안전 성공 · 공식 성공은 3판 · pb7 과 같은 1 개이고 포화 step 과 쥔 bowl 미인식은 늘었다. 5 개 episode 가 바뀐 이유는 분리하지 않았다.

---

## SL6-diag-1 — Panda 에서 AG3S 를 켜면 작업 성공률이 떨어지는 원인 진단 (2026-10-30)

> 이 진단에는 handoff 에 부여된 SL6 번호가 없다 (미확인). scribe 가 구분용으로 붙인 이름이다.

- README 작성 2026-10-30 11:50 KST. 저장된 기록만 사용 (CPU, 새 실행 없음). 사용자 지시 ~11:15 KST: "현재 작업 성공률이 떨어지는 원인을 진단하고 분석한 후에 수행합시다". source: [README](../../../outputs/2026-10-30_작업성공률저하_원인진단_Panda/README.md) · `summary.json` (stage_counts · per_episode · place · carry).

### 목적 → 방법

같은 11 episode 에서 VLA 단독은 공식 성공 11 / 11 인데 AG3S 를 켠 모든 판(pb7 · 3판 · E0판)은 1 / 11 이다. 어디에서 막히는지 단계별로 가르고 원인 층을 찾는다.
- 조건: VLA 단독 (SL5f) · pb7 (SL5i-G) · 3판 (SL6-E) · E0판 (SL6-F). target bowl = VLA 단독이 실제로 들어 올린 bowl (MuJoCo 상태).
- 도달 단계 (MuJoCo 기준, 대기 20 frame 뒤): 놓기 성공 = official success / 쥐고 들었으나 못 놓음 = target bowl 이 2 cm 넘게 들리고 손과 9 cm 안인 frame 이 20 개 넘음 / target 에서 닫았으나 못 듦 = 처음 닫을 때 손이 target bowl 에서 40 mm 안 / target 밖에서 닫음 / 닫지 않음.
- 놓기 단계: bowl 이 plate 중심 30 mm 안에 처음 온 순간의 높이, 쥔 뒤 call 마다 실행 z 와 VLA z 의 차이(latch 상태별), 같은 call 의 쥔 물체 모델 기록(바닥 여유, 지지면 근접).
- target 선택: 첫 25 call 에서 AG3S target 중심에 가장 가까운 물체를 셌다 (예: plate×24 = 24 call 동안 plate 를 골랐다).

### 결과

#### 1. 도달 단계 (`summary.json` stage_counts · README)

| 조건 | 놓기 성공 | 쥐고 들었으나 못 놓음 | target 에서 닫았으나 못 듦 | target 밖에서 닫음 | 닫지 않음 |
|---|---|---|---|---|---|
| VLA 단독 | 11 | 0 | 0 | 0 | 0 |
| pb7 | 1 | 6 | 0 | 4 | 0 |
| 최종 3판 | 1 | 3 | 1 | 5 | 1 |
| **E0판** | 1 | **6** | 1 | **3** | 0 |

![도달 단계 표](figures/benchmark/2026-10-30_작업성공률저하_원인진단_Panda/결과표_도달단계.png)

![조건별 도달 단계, 쥔 bowl 높이, Σ(실행 − VLA) z](figures/benchmark/2026-10-30_작업성공률저하_원인진단_Panda/그래프_실패단계.png)

실패는 두 갈래이고 pb7 에서도 같은 구조다 (README). Panda 최종 구성으로 바꾸면서 새로 생긴 문제가 아니라 Panda AG3S 에 계속 있던 문제다: **task 1 · 2 (7 episode)** 는 맞는 bowl 을 쥐어 들지만 내려놓지 못하고, **task 3 (4 episode)** 는 진짜 target bowl 에 가지 못한다.

#### 2. task 1 · 2 — 놓기 단계 (E0판, README)

E0판 실패 6 개 중 5 개가 맞는 bowl 을 쥐고 plate 위까지 갔다. "하강 명령이 위로 바뀐 row" = VLA z < −0.1 인데 실행 z 가 VLA 보다 0.3 넘게 큰 row.

| episode | bowl 이 plate 위에 왔을 때 높이 | bowl 최고 높이 | 하강 명령이 위로 바뀐 row | Σ(실행 − VLA) z | 쥔 동안 latch 상태 (call 수) |
|---|---|---|---|---|---|
| VLA 단독 (7 개) | 0.10–0.15 m | 0.16–0.19 m | – | – | – |
| t1e1 | 0.435 m | 0.784 m | 66 | +120 | HELD 26 · CLOSING 19 |
| t1e4 | 0.143 m | 0.314 m | 32 | +32 | CLOSING 38 (미인식) · LATCHED 9 |
| t2e0 | 0.265 m | 0.398 m | 14 | +31 | CLOSING 31 · SEARCHING 12 |
| t2e2 | 0.078 m | 0.269 m | 14 | +7 | HELD 45 |
| t2e4 | 0.375 m | 0.541 m | 25 | +60 | CLOSING 29 (미인식) |
| t2e1 | plate 위에 못 옴 | 0.234 m | 4 | −3 | 쥐기가 늦음 (frame 188, VLA 122) |
| (t2e3 성공) | 0.058 m | 0.211 m | 92 | +88 | HELD 40 |

- VLA 는 0.10–0.15 m 높이로 plate 위에 와서 바로 내려놓는다. AG3S 는 bowl 을 더 높이 들고, 내려가려는 VLA 명령을 TrajOpt 가 위로 바꾼다.
- latch 가 쥔 bowl 을 인식한 경우(HELD: t1e1 · t2e2)와 인식하지 못한 경우(CLOSING: t1e4 · t2e0 · t2e4) 모두 일어난다. 그래서 latch 인식만 고쳐서는 풀리지 않는다.
- 지지면에 가까워질 때 위로 뒤집힌다. pb7 t2e4 에서 쥔 bowl 모델의 바닥 여유가 34 → −10 mm 로 줄 때 VLA z −1.7 이 실행 z +5 (위로 포화) 로 바뀐다. E0판 t2e3 도 바닥 여유가 0 근처일 때 같은 뒤집힘이 반복되었고 놓기는 frame 288 에 성공했다.
- **Panda 기록에는 목적지 (plate) 가 없다.** RB-Y1 AG3S 는 목적지를 인식 · 등록해 따로 처리한다 (`DestinationRegistry`, `DESTINATION` 여유). Panda 서버 설정과 call 기록에는 destination 항목이 없다. plate 가 일반 장애물로 거리장에 들어가 쥔 bowl 구 (반지름 44 mm + 여유) 가 내려가지 못하는 것인지는 **확인 필요**이고, 어느 제약 row 가 걸리는지는 기록에 없다 (`--record-constraints` 로 다시 돌려야 확인).

#### 3. task 3 — target 을 다른 물체로 고름 (README)

명령: "pick up the black bowl **on the wooden cabinet** and place it on the plate".

| episode | pb7 이 고른 target | 3판 | E0판 | 진짜 target |
|---|---|---|---|---|
| t3e0 | 다른 bowl ×24 | 다른 bowl ×24 | 다른 bowl ×4 → plate ×20 | cabinet 위 bowl |
| t3e2 | 다른 bowl ×24 | plate ×24 | plate ×24 | cabinet 위 bowl |
| t3e3 | 다른 bowl ×24 | plate ×24 | plate ×24 | cabinet 위 bowl |
| t3e4 | 다른 bowl ×24 | 다른 bowl ×24 | 다른 bowl ×24 | cabinet 위 bowl |

- action attention (`L8H2`) 이 task 3 에서 다른 bowl (stove 옆) 이나 목적지 plate 를 가리킨다. 진짜 bowl 은 target 제외를 받지 못하고 장애물로 남는다.
- VLA 는 frame 62–71 에 진짜 bowl 에서 닫는다. AG3S 는 frame 68–294 에 엉뚱한 곳에서 닫는다. E0판은 첫 닫기 전 call 의 79–91 % 를 고쳤다 (t3e0 15/19 · t3e3 19/21 · t3e4 50/55).

![실제 장면: 놓기 순간과 task 3 target](figures/benchmark/2026-10-30_작업성공률저하_원인진단_Panda/실제장면_놓기_target.png)

#### 원인 정리 (E0판 실패 10 개, README)

| 원인 | 증상 → 원인 구성 요소 | episode | 근거 |
|---|---|---|---|
| 놓기 단계에서 TrajOpt 가 하강을 막음 (쥔 bowl 이 지지면 · plate 에 가까워지면 위로 뒤집음) | CCR (후보: LTE — 쥔 물체 모델 · 목적지 처리 없음) | 5 (t1e1 · t1e4 · t2e0 · t2e2 · t2e4) | plate 위 높이 0.08–0.44 m (VLA 0.10–0.15), 하강이 뒤집힌 row 14–66 |
| target 을 다른 bowl 이나 plate 로 고름 | ATS | 4 (task 3 전부) | 첫 25 call target 이 다른 bowl · plate |
| 쥐기가 늦어 시간 부족 | CCR 후보 (닫기 전 수정 4/8) | 1 (t2e1) | bowl 들림 frame 188 vs VLA 122 |

#### 측정하지 않은 것

- 어느 제약이 하강을 막았는가 (기록에 없음); plate 가 장애물로 들어갔는지.
- task 3 의 attention 이 왜 plate · 다른 bowl 을 가리키는지 (명령의 어느 token, head 선택) → 다음 절에서 일부 분리.
- 비율은 이 11 episode 기준이다.

**읽는 법.** 공식 성공 11 → 1 로 떨어지는 구간은 (a) 쥔 뒤 놓기 단계와 (b) task 3 의 target 선택으로 나뉘며, pb7 에서도 같은 구조였다. 앞 절들의 최종 구성 변경으로 새로 생긴 문제가 아니라는 것이 이 진단이 보인 사실이다. 사용자 판정(~12:00 KST, `SL6.task.md` §0)으로 task 3 원인 분리와 placement 단계 접촉 허용 flag 두 가지가 이어졌다.

---

## SL6-diag-2 — task 3 target 오류: attention 원인인가, grounding 원인인가 (2026-10-30)

> 이 진단도 handoff 번호가 없다 (미확인). scribe 가 구분용으로 붙인 이름이다.

- README 작성 2026-10-30 15:30 KST. 사용자 판정 ~12:00 KST: "원인 분리 먼저". GPU (공유 lock, attention 추출만) + CPU 분석, closed-loop 실행 없음. source: [README](../../../outputs/2026-10-30_task3_target오류_원인분리/README.md) · `t3_split.json` (call 별 카메라별 질량, 진짜 bowl 몫, 화면 안 여부, AG3S target).

### 목적 → 방법

task 3 에서 pb7 · 3판 · E0판 모두 진짜 bowl 대신 다른 bowl 이나 plate 를 target 으로 골랐다. 지금 쓰는 action attention (`action:last:L8H2`) 이 처음부터 틀린 곳을 보는지(attention 원인), 맞는 곳을 보는데 AG3S 가 다른 물체를 고르는지(grounding 원인)를 가른다.
- 대상: E0판과 pb7 의 task 3 episode (e0 · e2 · e3 · e4), 각 첫 6 call (frame 20–45). VLA 단독은 frame 62–71 에 진짜 bowl 에서 닫는다.
- **GPU 재추출:** 기록된 episode 에서 policy 요청을 client 와 같은 방식으로 다시 만들어(그 frame 의 두 카메라 이미지, `policy_state`, 명령문) 서버와 같은 경로(`PandaAttentionReader`, denoise step 0, `last`, layer 8 · head 2)로 attention 을 읽었다.
- **CPU 분석:** 기록된 MuJoCo 상태로 각 물체 중심을 head · 손목 카메라에 투영하고 물체 반경 원 안의 attention 질량을 셌다 (분할 mask 가 아닌 근사).

### 결과 (README, `t3_split.json`)

진짜 bowl 위 attention 몫과 AG3S target (E0판):

| episode | frame 20–30 (call 0–2) | frame 35–45 (call 3–5) | AG3S target (call 1–5) | latch |
|---|---|---|---|---|
| t3e2 | 0.00 – 0.13 (plate · 다른 bowl 위) | **0.90 · 0.97 · 0.99** | plate (계속) | SEARCHING |
| t3e3 | 0.00 (plate 위 0.87–0.93) | 0.00 · 0.03 · **0.90** | plate (계속) | frame 30 부터 LATCHED |
| t3e4 | 0.00 – 0.01 (다른 bowl · plate 위) | 0.07 · **0.97 · 0.97** | 다른 bowl (계속) | frame 30 부터 LATCHED |
| t3e0 | 0.01 – 0.02 | 0.09 · 0.17 · 0.27 (wine bottle 과 나눔) | 다른 bowl → plate | SEARCHING |

pb7 도 같은 모양이다: t3e2 는 frame 30–45 에 0.42–0.92, t3e4 는 frame 35–45 에 0.43–0.98 인데 AG3S target 은 처음부터 끝까지 다른 bowl 이다.

![pb7 · E0판 episode 별 진짜 bowl 위 attention 몫](figures/benchmark/2026-10-30_task3_target오류_원인분리/그래프_task3_attention몫.png)

![E0판 t3e2 · t3e4 의 head · 손목 이미지와 L8H2 attention](figures/benchmark/2026-10-30_task3_target오류_원인분리/실제장면_task3_attention.png)

#### 두 겹의 원인

| 구간 | 관측 | 원인 구성 요소 |
|---|---|---|
| 처음 (frame 20–30) | L8H2 attention 질량의 65–99 % 가 손목 카메라에 있고, 이때 진짜 bowl 은 손목 화면 밖이다 (투영이 256×256 밖, E0판 frame 20–30 의 12 call 중 12). attention 이 손목 화면 안의 plate · 다른 bowl 에 앉는다. head 에서는 진짜 bowl 이 보이지만 head 질량이 작아(0.1–35 %) target 선택을 이기지 못한다 | ATS (attention 입력 · 카메라 비중) |
| 이후 (frame 35–45) | 손이 cabinet 쪽으로 가 진짜 bowl 이 손목 화면에 들어오면 attention 의 90–99 % 가 진짜 bowl 위 (E0판 4 개 중 3 개, pb7 4 개 중 2 개). AG3S target 은 처음 고른 물체 그대로. t3e3 · t3e4 는 frame 30 에 latch 가 틀린 물체에 LATCHED 로 잠겼다 (잠긴 동안 target 정체를 쉽게 바꾸지 않는 규칙 `--latched-identity-hold`, V11AB 기본). t3e2 는 latch 가 SEARCHING 인데도 plate 를 유지했다 | ATS (grounding · 추적) |

결과적으로 진짜 bowl 은 끝까지 장애물로 남고 TrajOpt 가 손을 그 bowl 에서 떼어 놓는다 (닫기 전 call 의 79–91 % 수정, README).

#### 측정하지 않은 것

- 투영 원 근사이며 분할 mask 가 아니다 (bowl 끼리 가까운 경우는 없었다).
- t3e2 의 SEARCHING 중 target 유지 규칙의 이름: 미확인.
- 첫 6 call (frame 45 까지)만 봤다.

**읽는 법.** task 3 의 target 오류는 한 가지가 아니라 두 겹이다. 첫 target 결정 때는 진짜 bowl 이 손목 화면 밖이라 attention 이 틀린 곳에 앉고, 이후에는 attention 이 진짜 bowl 로 옮겨 가도 target 이 따라가지 않는다. 이 분리 뒤 사용자 판정(2026-10-30 ~15:40 KST)으로 attention 이 옮겨 가면 target 을 교체하는 기본 꺼짐 flag 가 계획되었다 (`SL6.task.md` §12). 그 flag 의 효과는 이 기록에 없다 (미측정).

---

## SL6-H — E0판 + placement 단계 쥔 물체 접촉 허용 smoke (2026-10-30)

- 실행 16:15–16:44 KST, commit `96fda38a`, README 작성 17:05 KST. source: [README](../../../outputs/2026-10-30_Panda_placement접촉허용_SafeLIBERO-LI-dev11/README.md) · `summary.json` · `placement_activation.json`. 규칙 정의: [`handoff/SL6.task.md`](handoff/SL6.task.md) §11, 구현 상세: [`handoff/SL6-H.impl.md`](handoff/SL6-H.impl.md) (README 가 인용).

### 목적 → 변경 (E0판 → placement 접촉 허용판)

앞의 성공률 진단에서 E0판 실패 10 개 중 5 개는 놓기 단계에서 막혔다. 쥔 bowl 이 plate · 지지면에 가까워지면 TrajOpt 가 하강을 위로 뒤집었다. placement 단계에서 쥔 물체와 아래 표면의 접촉을 허용하면 내려놓을 수 있는지 본다 (사용자 판정 ~12:00 KST).
- flag `--hsr6-placement-contact` (기본 꺼짐; profile `AG3S-final-panda-v3-e0to-pc`).
- **켜지는 call:** HSR6 argmax = placement ∧ (latch HELD ∨ (CLOSING ∧ closed ∧ blocked)).
- **푸는 것:** attach 된 쥔 물체 질의 구 행 중 "아래 표면"에 대한 행만 (margin −1 m). 아래 표면 = 거리장 기울기가 위로 45° 안 **이고** 최근접 표면점이 쥔 물체 xy 발자국 + 5 mm 안.
- **그대로:** 손 · 손가락 · 팔 행과 옆 장애물. call 기록에 HSR6 단계 · 확률과 이 규칙의 on/off · 풀린 행을 함께 남긴다 (action 불변).
- 판정(결과 전 고정): pc판 안전 성공 ≥ E0판 (1) 이고 충돌 ≤ E0판 (5) 이면 채택.

### 결과 (README 자체 보고 · `summary.json`)

| 조건 | 안전 성공 | 공식 성공 | 충돌 | HOLD call | 쥔 bowl 미인식 | 포화 action step |
|---|---|---|---|---|---|---|
| VLA 단독 | 2 | 11 | 9 | – | – | 0 |
| pb7 | 1 | 1 | 5 | 121 / 580 | 2 | 955 |
| E0판 | 1 | 1 | 5 | 50 / 615 | 5 | 1621 |
| **pc판** | **1** | **1** | **5** | 53 / 595 | 5 | 1519 |

**판정(규칙): 채택** (1 ≥ 1, 5 ≤ 5). **그러나 11 episode 의 결과가 E0판과 episode 단위로 같다.** 8 개는 실행 action 까지 E0판과 bit 동일하고 (README 보고), 다른 3 개(t1e1 · t2e2 · t2e3)도 결과(시간초과 · 시간초과 · 안전 성공)는 같다 (결과 일치는 `summary.json` 으로 확인). 규칙상 채택이지만 놓기 실패를 고친 근거는 없다 (README).

![결과표](figures/benchmark/2026-10-30_Panda_placement접촉허용_SafeLIBERO-LI-dev11/결과표.png)

![조건별 결과, target 선택 오차, 쥔 뒤 TrajOpt 위쪽 수정](figures/benchmark/2026-10-30_Panda_placement접촉허용_SafeLIBERO-LI-dev11/그래프.png)

#### 규칙이 작동한 정도 (`placement_activation.json` · README, task 1 · 2)

"bowl 이 plate 위인 call" = bowl–plate xy 60 mm 안이고 3 cm 이상 들린 call.

| episode | plate 위 call | 그중 placement 확률 ≥ 0.5 | 규칙 켜짐 | 행을 품 | 푼 쥔 물체 행 | 하강이 위로 뒤집힌 call |
|---|---|---|---|---|---|---|
| t1e1 | 5 | 0 | 4 | 4 | 2 | 4 |
| t1e4 | 1 | 1 | 7 | 0 | 0 | 0 |
| t2e0 | 2 | 0 | 0 | 0 | 0 | 2 |
| t2e1 | 0 | 0 | 0 | 0 | 0 | 0 |
| t2e2 | 3 | 2 | 2 | 2 | 4 | 3 |
| t2e3 | 5 | 0 | 4 | 4 | 8 | 2 |
| t2e4 | 3 | 0 | 0 | 0 | 0 | 2 |

![plate 위 call 의 placement 확률 대 위쪽 수정](figures/benchmark/2026-10-30_Panda_placement접촉허용_SafeLIBERO-LI-dev11/그래프_placement허용_작동.png)

README 가 든 세 가지 이유 (증상 위치 → 원인 위치):

| 이유 | 증상 위치 | 원인 구성 요소 | 근거 |
|---|---|---|---|
| bowl 이 plate 위에 있을 때 HSR6 가 대부분 transport 로 판정 | placement 확률 0.02–0.40 (예: t2e4 frame 215 · 220 에서 0.08 · 0.02, t2e0 frame 125 에서 0.08) | HSR (HSR6 단계 판정) | 높이 든 채 plate 위에 온 장면이 학습 label 의 placement 와 다르게 보이는 것으로 **추정** (확인 필요) |
| 실제로 쥐고 있는데 latch 가 인식 못함 → attach 된 물체가 없어 풀 행도 없음 | t2e0 은 bowl 이 plate 위 0.27 m 인데 latch SEARCHING, t2e4 · t1e1 후반은 CLOSING | HSR (latch) | t1e4 는 규칙이 7 call 켜졌지만 모두 "attach 된 쥔 물체 없음" |
| 켜진 call 에서도 하강이 막힘 | t2e2 frame 170 에서 쥔 물체 행 2 개를 풀었는데 VLA z −2.0 → 실행 z +3.0, frame 175 는 −3.2 → +0.5 | CCR (위로 미는 제약은 쥔 물체 행만이 아님. 손 · 손가락 행이 plate · 지지면에 막히는지는 기록에 없음) | `--record-constraints` 로 확인 필요 |

#### 이전 대비 (before → after)

| 기준 | 안전 성공 | 공식 성공 | 충돌 | HOLD | 포화 step | 쥔 bowl 미인식 |
|---|---|---|---|---|---|---|
| E0판 → pc판 | 1 → 1 | 1 → 1 | 5 → 5 | 50 → 53 | 1621 → 1519 | 5 → 5 |
| pb7 → pc판 | 1 → 1 | 1 → 1 | 5 → 5 | 121 → 53 | 955 → 1519 | 2 → 5 |

놓기 실패 5 개 (t1e1 · t1e4 · t2e0 · t2e2 · t2e4)는 모두 그대로 실패다 (README).

![실제 장면: target 선택](figures/benchmark/2026-10-30_Panda_placement접촉허용_SafeLIBERO-LI-dev11/실제장면_target선택.png)

![실제 장면: bowl 상승](figures/benchmark/2026-10-30_Panda_placement접촉허용_SafeLIBERO-LI-dev11/실제장면_bowl상승.png)

#### 측정하지 않은 것

- 위로 미는 제약이 무엇인지 (call 기록에 없음).
- HSR6 가 plate 위에서 transport 로 판정하는 이유 (확인하지 않음).
- 충돌마다 닿은 몸체, 도달 단계 재집계: 사전 고정 항목에 있었으나 이 README 에는 없다 (미확인).

**읽는 법.** 사전 규칙으로는 채택이지만, 이 규칙은 거의 켜지지 않았고(plate 위 call 중 placement 확률 ≥ 0.5 인 것이 task 1 · 2 표에서 t1e4 1 · t2e2 2 뿐) 켜진 call 에서도 하강이 풀리지 않았다. 따라서 E0판 대비 결과 차이는 없고, 놓기 실패의 원인은 이 flag 가 다룬 쥔 물체 행 밖에 있거나 HSR6/latch 단계 판정에서 규칙이 켜지지 않는 데 있을 수 있다 — 어느 쪽인지는 이 기록으로 가를 수 없다. 이 결과의 해석과 다음 수정은 사용자 판정 전이다.

---

## SL-GTTO — GT 인식 + 공용 TO 3 벤치 검증 (2026-11-04 – 11-05)

- 목적 (사용자 지시 2026-11-04): baseline 비교 전에 "공용 TO"가 GT 인식으로 LIBERO 9 · SafeLIBERO 11 · LIBERO-Safety 10 (SL9의 30 장면)에서 작동하는지 본다. 통과하면 같은 TO에 인식만 바꿔 끼운다. 통과하지 못하면 TO(또는 Panda 변환 C · 쥐기 처리) 층을 먼저 고친다.
- 판정 규칙(결과 전 고정, [`handoff/SL-GTTO.task.md`](handoff/SL-GTTO.task.md) §5): LIBERO GT 성공 ≥ 8 ∧ VLA 대비 저하 ≤ 1 / SafeLIBERO 충돌 ≤ VLA−3 ∧ 안전 성공 ≥ VLA+3 / LIBERO-Safety (성공 ∧ 무위반) ≥ VLA−2 ∧ 위반 ≤ VLA.
- 기준: 같은 기계 · 같은 commit `89b42459`(+ 미commit patch)의 VLA 단독 run `*_gtto-vla-*` (2026-11-04 23:1x–23:5x).
- 판 결정 기록: [`doc/active/experiments/E1_GTTO_DECISIONS_20261104_KO.md`](../../../doc/active/experiments/E1_GTTO_DECISIONS_20261104_KO.md) G1–G27 (G18부터 lead 자율 판정, 사용자 검토 대상). 그림 · 표: [`outputs/2026-11-05_GT인식_공용TO_3벤치/`](../../../outputs/2026-11-05_GT인식_공용TO_3벤치/README.md).

### SL-GTTO v1 — `AG3S-commonto-gt` 본 실행 (2026-11-04 23:50 – 11-05 01:29 KST)

판 정의 · flag 분류표: [`handoff/SL-GTTO.impl.md`](handoff/SL-GTTO.impl.md). 공용 TO = Panda 최종 flag − AG3S 인식 flag − HSR6 − multi-seed − D1 이웃 허용 + pc판 쥐기 처리(pandafix 8) + latch 단계 label + GT 인식(grounding = code 1 점, 미세 층 `oracle-visible` identity gt).

| 벤치 | VLA 단독 | GT + 공용 TO | 규칙 |
|---|---|---|---|
| LIBERO 9 | 성공 9 | 성공 2 (t1e0, t8e0) | 실패 |
| SafeLIBERO 11 | 성공 11 · 충돌 9 · 안전 성공 2 | 성공 0 · 충돌 2 · 안전 성공 0 | 실패 |
| LIBERO-Safety 10 | 성공∧무위반 7 · 위반 1 | 성공∧무위반 2 · 위반 1 | 실패 |

![v1 성공 grid](../../../outputs/2026-11-05_GT인식_공용TO_3벤치/gt_v1/성공grid.png)
![v1 HOLD 비율](../../../outputs/2026-11-05_GT인식_공용TO_3벤치/gt_v1/그래프_HOLD비율.png)

실패 층 (call 기록 분류, `experiments/segmentation/gtto_aggregate.py`):

| 층 | 증상 | episode |
|---|---|---|
| ① admissibility가 GT target 거절 | 모든 call `no_admissible` (`not_supported`) → uncertified HOLD. 원인은 base argv `--supported-max-bottom-mm 60` (T43 T3(c)): 서랍 안 · 캐비닛 위 그릇 바닥이 탁자 지지면 위 143–234 mm | LIBERO t4e0 · t9e0, SafeLIBERO t3 4개 (44/44 · 56/56 HOLD) |
| ② 쥔 물체 잔여 점 | attach 뒤 cloud code 1 점 ~170 → 0–24 (`no_cluster`). ESDF 근거는 closing · held 동안 손안 GT 그릇 pixel이 TSDF에 적분된 것(self-filter 구 ↔ attached 구 53–77 mm 어긋남, 60 mm truncation band 잔존) → `attached:panda_hand ↔ esdf` HOLD | LIBERO t0e0 · t7e0, SafeLIBERO t1e1 · t1e4 · t2e0 · t2e1 |
| ③ 팔 link5–7 `unauthorized_link` | closing 중 움직인 GT 그릇 + 가장자리 static pixel (진단만) | LIBERO t3e0 등 |

### SL-GTTO2 — GT identity · 범위가 TO까지 가게 (구현 2026-11-05 01:1x–02:1x, smoke 03:09–03:23)

task [`handoff/SL-GTTO2.task.md`](handoff/SL-GTTO2.task.md) (결과 전 고정), impl [`handoff/SL-GTTO2.impl.md`](handoff/SL-GTTO2.impl.md). 새 판 `AG3S-commonto-gt2` = gt + 두 flag (기본 꺼짐, 꺼짐이면 모든 기존 판 argv · 기록 동일).

| flag | 변경 | CPU 재생 검증 (impl) |
|---|---|---|
| F1 `--oracle-admissible` | oracle grounding에서 TargetConfirm admissibility · `_exclusion_gate`가 GT cluster를 거절하지 않음. 검사 결과는 `admissibility_would_refuse` · `exclusion_gate_would_refuse`로 기록만 | 6 episode 312/312 call `no_admissible` → `ok` |
| F2 `--oracle-held-target` | closing · held 동안 code 1 pixel을 TSDF depth mask에 넣음 (`held_target_mask` 기록) | closing/held code 1 TSDF 입력 0, `attached_point` 5행의 code 1 근거 제거 |

CPU test: 새 18 + 기존 SL-GTTO 46 + 주변 344 passed, 환경 문제 4 (worktree venv 부재), SL5a golden 차이 24 = 변경 전과 같음.

GPU smoke (SOLO, 4 episode):

| episode | 확인 키 (task §2) | 키 | 공식 결과 | 남은 증상 |
|---|---|---|---|---|
| LIBERO t4e0 | 첫 call grounding ok, HOLD 없음 | 통과 | 실패 (max_steps) | latched 42 call 중 C 포화 36, attach 없음. HOLD 14 (`answer_excludes_target` hand · `unauthorized_link` link6 coarse) |
| SafeLIBERO t3e0 | 첫 call grounding ok, HOLD 없음 | 통과 | 실패 (max_steps, 충돌 없음) | latched 27 중 포화 22, attach 없음 |
| LIBERO t0e0 | attach 뒤 `attached_point` 0 | 통과 | 실패 (max_steps) | held 20 call 모두 C 포화, dA 1.5–1.9, 놓기(placed) 뒤 성공 판정 없음 |
| SafeLIBERO t1e1 | attach 뒤 `attached_point` 0 | 통과 | **성공 · 충돌 없음** | — |

판정 G25: 키 통과 → gt2 30 장면 본 실행 (queue 41–47, 결과는 아래에 추가).

### 새 층 ④ — TO 경로 수정이 변환 C 포화로 (2026-11-05 03:3x–03:5x, 기록 분석)

![C 포화](../../../outputs/2026-11-05_GT인식_공용TO_3벤치/gt2_smoke/그래프_C포화.png)

| `reference_deviation` | call | C 포화 (≥5/10행) | median `max_abs_correction` |
|---|---:|---:|---:|
| 0 | 289 | 0 | 0.00 |
| < 0.01 | 172 | 1 | 0.05 |
| 0.01–0.05 | 227 | 52 | 1.46 |
| 0.05–0.2 | 386 | 269 | 3.91 |
| ≥ 0.2 | 315 | 310 | 12.33 |

(v1 30 + gt2 smoke 4 episode, HOLD 아닌 call. source `gt2_smoke/saturation.json`, script `experiments/segmentation/gtto_saturation.py`.)

- held 구간의 `trajopt_status = violated`는 모두 verdict `budget_only` (SQP 반복 상한 3, 최종 검사 통과)다. 위반이 아니라 경로 수정이다.
- 성공한 episode(LIBERO t1e0 v1, SafeLIBERO t1e1 gt2)도 held 구간에서 같은 포화가 있다. 포화가 곧 실패는 아니다 — 실패 episode와의 차이는 본 실행 30 장면으로 센다.
- 공용 TO는 `--continuity-after-grasp --deflection-rate-after-grasp`를 켜고(V11AB 무게 0.5 / 20) pass-through가 없다. SL5b/pb2 기록은 이 두 항이 위반 없는 chunk에서도 VLA의 새 chunk를 이전 계획 쪽으로 당긴다고 진단했다.
- SL7b(P0m 위 같은 변경, 모두 1/9)의 "TO는 원인 아님"은 인식이 먼저 막고 있던 조건의 결론이다.

### SL-GTTO2 본 실행 — `AG3S-commonto-gt2` 30 장면 (2026-11-05 05:31 – 07:00 KST, 판정 G28)

같은 기계 · commit `89b42459` + GTTO · GTTO2 patch, seed 7, conversion C. 집계 `experiments/segmentation/gtto_aggregate.py --arm gt2` → [`outputs/2026-11-05_GT인식_공용TO_3벤치/gt_v2/`](../../../outputs/2026-11-05_GT인식_공용TO_3벤치/gt_v2/) (`결과표.md` · `summary.json` · 그림).

| 벤치 | VLA 단독 | v1 (`-gt`) | gt2 | §5 규칙 | 판정 |
|---|---|---|---|---|---|
| LIBERO 9 공식 성공 | 9 | 2 | **2** | GT ≥ 8 ∧ VLA − GT ≤ 1 | 실패 |
| SafeLIBERO 11 공식 성공 / 충돌 / 안전 성공 | 11 / 9 / 2 | 0 / 2 / 0 | **4 / 6 / 2** | 충돌 ≤ VLA − 3 (= 6 ✓) ∧ 안전 성공 ≥ VLA + 3 (= 5 ✗) | 실패 |
| LIBERO-Safety 10 성공∧무위반 / 위반 | 7 / 1 | 2 / 1 | **2 / 0** | ≥ VLA − 2 (= 5) ∧ 위반 ≤ VLA | 실패 |
| episode 평균 C 포화 call 비율 (LIBERO / SafeLIBERO / LIBERO-Safety) | – | 0.31 / 0.39 / 0.50 | 0.48 / 0.61 / 0.49 | – | – |

- F1 은 SafeLIBERO t3 (캐비닛 위 그릇) 2 episode 에서 attach 를 열었다 (v1 은 t3 4 episode 모두 attach 없음). F2 뒤에도 `attached_point` HOLD 가 LIBERO t1e0 14 · t3e0 5, SafeLIBERO t2e3 14 call 남는다.
- 결정성: LIBERO t1e0 은 v1 · gt2 의 seq 0–12 기록(tip · 목표 centroid · reference deviation)이 같고, seq 13 에서 손가락 닫힘 결과(개도 0.025 vs 0.059)가 갈린 뒤 달라진다 — 같은 조건의 반복 차이가 아니라 TO 출력 차이에서 갈린다.

#### 새 층 ⑤ — 쥔 물체 모델이 실제보다 아래 (기록 + sim 진실, G29)

| 측정 (held call, HOLD 제외) | 수치 |
|---|---|
| VLA reference 경로 중 attached 질의 구 최저점 − 지지면 < 15 mm 인 call | gt2 LIBERO: t0e0 11/20 · t1e0 4/7 · t2e0 10/15 · t6e0 3/5 · t8e0 3/6 (v1 도 같은 경향: t2e0 21/26) |
| 그때 `reference_deviation` 중앙값 vs 그 밖 | gt2 t0e0 0.29 vs 0.18, t1e0 0.27 vs 0.06, t2e0 0.18 vs 0.03 · v1 t2e0 0.25 vs 0.03 |
| row 0 자세의 모델 바닥 − 실제 물체 바닥 (sim, 중앙값) | gt2 −98 · −106 · −34 · −74 · −50 · −4 · −15 mm (t0 · t1 · t2 · t3 · t6 · t7 · t8), v1 −50 … +17 mm |

해석: 놓기 하강에서 쥔 모델이 지지면 · plate 안으로 들어가므로, TO 가 하강을 위로 고치고 (④b 의 held 구간 포화) 더 깊으면 `attached_point` HOLD 가 된다.
snapshot 시점 가설(마지막 관측 frame 의 자세 × 들기 전 기하)은 finger 정지 frame 으로 바꾼 반사실 계산에서 오차가 줄지 않아 기각했다.
SL6 E0판 진단의 "놓기 하강 뒤집힘" 과 같은 증상이다. SL6-H 의 행 규칙은 HSR6 분류기가 plate 위를 `transport` 로 보아 거의 켜지지 않았다 (위 SL6-H 절).

![쥔 모델 바닥 vs TO 경로 수정, 모델 바닥 오차](../../../outputs/2026-11-05_GT인식_공용TO_3벤치/gt_v2/그래프_쥔모델.png)

source: `experiments/segmentation/gtto_held_model.py` → `gt_v2/held_model.json`.

#### SL-GTTO4 — F3 `--held-beneath-contact` (구현 2026-11-05 07:1x–07:2x, 판정 G29 · G30)

task [`handoff/SL-GTTO4.task.md`](handoff/SL-GTTO4.task.md) (결과 전 고정). SL6-H 의 placement-contact 행 규칙(쥔 물체 자신의 행 중 아래 표면에 대한 행만 풀림, 손 · 손가락 · 팔 · 옆 장애물은 그대로)을 HSR6 없이 **latch HELD + attach** 로 켠다.
변경: `SafePolicy._held_beneath_contact` · `_held_beneath_record` (+ `_scene_fn` hook, `to.held_beneath_contact` 기록), serve_safe `--held-beneath-contact` (`apply_grasp_contact_flags`), Panda server ported 목록 1줄, profile `AG3S-commonto-gt2-hbc` · `AG3S-commonto-gt2-termsoff-hbc`.
CPU test `tests/trajopt/test_sl_gtto4_held_beneath.py` 10개 + 관련 suite 108 passed, 의존 suite 195 passed / 5 skipped. 기본 꺼짐: 다른 판의 argv · 기록 key 불변.

### SL-GTTO3 — gt2 위 TO 변형 3판 (등록 2026-11-05 03:5x, 실행 대기)

task [`handoff/SL-GTTO3.task.md`](handoff/SL-GTTO3.task.md) (결과 전 고정). profile 등록만, 새 code 없음: `AG3S-commonto-gt2-termsoff` (두 항 끔, 3판 · SL7b (c)와 같은 변경), `-pass` (`--passthrough-when-clear`, T88), `-termsoff-pass`. test `tests/benchmark/test_sl_gtto3_versions.py` 등 96 passed / 5 skipped.
LIBERO 9로 가르고, gt2보다 ≥ 2 episode 높은 판이 있으면 SafeLIBERO · LIBERO-Safety까지 돌려 §5 전체를 판정한다. 공용 TO 정의를 바꾸는 것 자체는 사용자 판정으로 남긴다.

#### SL-GTTO3 결과 — LIBERO 9 (2026-11-05 07:00 – 07:42 KST, 판정 G31)

| 판 | 공식 성공 | 성공 episode | episode 평균 C 포화 call 비율 | HOLD call (9 episode 합) |
|---|---:|---|---:|---:|
| gt2 (기준) | 2 | t2 · t8 | 0.477 | 82 |
| termsoff | 3 | t0 · t2 · t8 | 0.405 | 67 |
| pass | 3 | t1 · t2 · t8 | 0.2185 | 8 |
| termsoff-pass | 4 | t1 · t2 · t7 · t8 | 0.2189 | 14 |

- 규칙 1 을 글자대로 적용하면 pass (포화 0.0003 낮음) → 규칙 2 (≥ gt2 + 2) 불성립. lead 는 차이가 해상도 아래라 보고 termsoff-pass 를 SafeLIBERO · LIBERO-Safety 에 **정보용으로** 돌린다 (규칙 이탈, G31).
- 예측 대조: ④a (t4 · t9 서랍 · 캐비닛 못 바꿈) 맞음 — 4 판 모두 attach 없이 실패. termsoff 가 held 포화를 줄여 t0e0 되찾음 (포화 call 24 → 5). "pass 효과 없음" 예측은 틀림: HOLD 82 → 8, t1e0 · t7e0 회복.
- 새 실패: pass 두 판 t0e0 · t6e0 은 attach 없음 — TO 는 통과(dev 0)인데 VLA 가 빈손 닫힘 · 재시도를 반복 (경로가 앞 call 에서 갈린 뒤).
- 그림 · 표: `outputs/2026-11-05_GT인식_공용TO_3벤치/gt3_*/`.

#### SL-GTTO4 결과 — F3 LIBERO 9 (2026-11-05 07:42 – 08:27 KST, 판정 G32)

| 짝 | 짝 성공 | F3 판 성공 | F3 판 포화 비율 | F3 판 HOLD 합 |
|---|---:|---:|---:|---:|
| gt2 → hbc | 2 | 2 | 0.463 (gt2 0.477) | 61 (gt2 82) |
| termsoff → termsoff-hbc | 3 | 2 | 0.386 (0.405) | 51 (67) |
| termsoff-pass → termsoff-pass-hbc | 4 | 3 | 0.191 (0.219) | 16 (14) |

- F3 는 작동했다: hbc 의 held call 63 개 모두 applied, 43 call 에서 행을 풀었다 (`to.held_beneath_contact`). 그래도 held 구간에서 reference 가 지지면 근처일 때 deviation 중앙값이 줄지 않았다 (t2e0 0.18 → 0.24).
- **판정: F3 는 실패를 설명하지 못한다 (SL-GTTO4 §3-3). ⑤ 의 "쥔 모델이 held 경로 수정을 만든다" 해석은 철회** (상관이었음). 모델 바닥 −15 … −106 mm 오차 측정은 그대로 남는다.
- 다음 진단: 제약 기록(`--record-constraints occupancy`)으로 held 구간에서 어떤 행이 경로를 미는지 직접 본다 (gt2 t0e0 · t2e0).
- 그림 · 표: `outputs/2026-11-05_GT인식_공용TO_3벤치/gt4_*/` (`gt4_hbc/held_model.json` · `그래프_쥔모델.png` = gt2 vs hbc).

#### termsoff-pass 세 벤치 (정보용, G31 · G36) 와 단계 분석 (2026-11-05 10:0x – 10:2x KST, G37 · G38)

| 벤치 | VLA 단독 | gt2 | termsoff-pass | §5 |
|---|---|---|---|---|
| LIBERO 9 성공 | 9 | 2 | 4 | 실패 |
| SafeLIBERO 11 공식 성공 / 충돌 / 안전 성공 | 11 / 9 / 2 | 4 / 6 / 2 | 1 / 3 / 0 | 실패 |
| LIBERO-Safety 10 성공∧무위반 / 위반 | 7 / 1 | 2 / 0 | **5 / 0** | **통과** |

단계 정의 (sim 진실, 모든 판 공통): 쥐기 전 = 첫 gripper 닫힘 명령 전 · 쥐는 중 = 닫힘 ~ target 3 cm 들림 · 쥔 후 = 그 뒤. script `experiments/segmentation/gtto_phase.py`.

| 판 | 벤치 | n | 성공 | 실패: 쥐기 전 | 쥐는 중 | 쥔 후 | 쥐기 전 step | 쥐는 중 step | 쥔 후 step | HOLD 비율 전/중/후 | dev 중앙값 전/중/후 | C 포화 전/중/후 | |Δa| 전/중/후 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| vla | LIBERO | 9 | 9 | 0 | 0 | 0 | 43.0 | 18.0 | 38.0 | – / – / – | – / – / – | – / – / – | – / – / – |
| vla | SafeLIBERO | 11 | 11 | 0 | 0 | 0 | 29.0 | 19.0 | 63.0 | – / – / – | – / – / – | – / – / – | – / – / – |
| vla | LIBERO-Safety | 10 | 7 | 0 | 1 | 2 | 53.0 | 27.0 | 77.0 | – / – / – | – / – / – | – / – / – | – / – / – |
| gt | LIBERO | 9 | 2 | 2 | 0 | 5 | 53.0 | 23.0 | 133.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.011 / 0.118 | 0.0 / 0.25 / 0.533 | 0.006 / 0.12 / 0.397 |
| gt | SafeLIBERO | 11 | 0 | 4 | 0 | 7 | 38.0 | 27.0 | 221.0 | 0.0 / 0.167 / 0.512 | 0.0 / 0.093 / 0.053 | 0.167 / 0.5 / 0.571 | 0.088 / 0.522 / 0.381 |
| gt | LIBERO-Safety | 10 | 2 | 0 | 1 | 7 | 65.5 | 63.5 | 448.0 | 0.098 / 0.059 / 0.185 | 0.0 / 0.042 / 0.104 | 0.143 / 0.433 / 0.545 | 0.114 / 0.375 / 0.43 |
| gt2 | LIBERO | 9 | 2 | 1 | 1 | 5 | 52.0 | 20.5 | 142.0 | 0.0 / 0.0 / 0.062 | 0.0 / 0.012 / 0.059 | 0.0 / 0.292 / 0.545 | 0.008 / 0.232 / 0.37 |
| gt2 | SafeLIBERO | 11 | 4 | 0 | 2 | 5 | 38.0 | 22.0 | 178.0 | 0.0 / 0.2 / 0.023 | 0.0 / 0.084 / 0.165 | 0.167 / 0.5 / 0.842 | 0.171 / 0.458 / 0.505 |
| gt2 | LIBERO-Safety | 10 | 2 | 0 | 2 | 6 | 64.5 | 78.5 | 419.0 | 0.098 / 0.005 / 0.093 | 0.0 / 0.023 / 0.135 | 0.148 / 0.29 / 0.562 | 0.147 / 0.297 / 0.508 |
| gt3termsoff | LIBERO | 9 | 3 | 1 | 1 | 4 | 52.0 | 20.5 | 142.0 | 0.0 / 0.0 / 0.036 | 0.0 / 0.009 / 0.029 | 0.0 / 0.292 / 0.333 | 0.008 / 0.227 / 0.335 |
| gt3pass | LIBERO | 9 | 3 | 0 | 2 | 4 | 55.0 | 21.0 | 58.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.037 | 0.0 / 0.12 / 0.2 | 0.045 / 0.093 / 0.133 |
| gt3termsoff-pass | LIBERO | 9 | 4 | 0 | 2 | 3 | 55.0 | 22.0 | 72.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.12 / 0.0 | 0.045 / 0.093 / 0.025 |
| gt3termsoff-pass | SafeLIBERO | 11 | 1 | 0 | 4 | 6 | 34.0 | 103.0 | 202.0 | 0.0 / 0.0 / 0.25 | 0.0 / 0.0 / 0.033 | 0.2 / 0.333 / 0.429 | 0.144 / 0.147 / 0.278 |
| gt3termsoff-pass | LIBERO-Safety | 10 | 5 | 0 | 2 | 3 | 56.5 | 91.0 | 268.0 | 0.0 / 0.005 / 0.087 | 0.0 / 0.0 / 0.04 | 0.143 / 0.12 / 0.452 | 0.118 / 0.162 / 0.361 |
| gt4hbc | LIBERO | 9 | 2 | 1 | 1 | 5 | 52.0 | 20.5 | 142.0 | 0.0 / 0.0 / 0.036 | 0.0 / 0.012 / 0.094 | 0.0 / 0.292 / 0.654 | 0.008 / 0.232 / 0.486 |
| gt4termsoff-hbc | LIBERO | 9 | 2 | 1 | 1 | 5 | 52.0 | 20.5 | 142.0 | 0.0 / 0.0 / 0.062 | 0.0 / 0.009 / 0.006 | 0.0 / 0.292 / 0.296 | 0.008 / 0.227 / 0.314 |
| gt4termsoff-pass-hbc | LIBERO | 9 | 3 | 0 | 2 | 4 | 55.0 | 22.0 | 72.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.12 / 0.0 | 0.045 / 0.093 / 0.02 |

step = 중앙값 (그 단계에 들어간 episode 만). HOLD · dev · C 포화 · |Δa| = episode 별 값의 중앙값. |Δa| = 실행 − 정책 action xyz 평균 (TO 개입 크기).

![단계별 결과와 TO 개입](../../../outputs/2026-11-05_GT인식_공용TO_3벤치/단계분석/그래프_단계별.png)

- 쥐기 전 실패는 드물고, 실패는 쥐는 중 (닫았는데 못 듦) 과 쥔 후 (들었는데 못 놓음) 에 몰린다. TO 개입은 모든 판에서 쥐기 전 < 쥐는 중 < 쥔 후.
- termsoff-pass 쥔 후 changed call 은 모두 `budget_only` — reference 를 막은 행이 기록에 없어 T88 `passthrough_clear` 기록을 `to` 에 싣도록 했다 (기록만).

#### SL-GTTO5 · SL-GTTO6 — 최소 개입 + 쥐기 직전 손 행 해제 + 가려진 쥔 물체 인증 (2026-11-05 09:4x – 11:5x KST, G34 – G42)

사용자 지시 (09:3x): Panda TO · 충돌회피 논문 13 편 조사 뒤 "최선의 방향으로 바꾸세요". 문헌의 VLA safety filter (AEGIS · KNOWS · multi-link · VLPSA · OSCBF) 는 모두 `min ‖u − u_nom‖²` + 비활성이면 통과, cuRobo `plan_grasp` · MoveIt MTC 는 마지막 접근에서 손–world 충돌을 끈다.

| 판 | 정의 | LIBERO 9 | SafeLIBERO 11 (성공 / 충돌 / 안전 성공) | LIBERO-Safety 10 |
|---|---|---:|---|---|
| termsoff-pass | 연속성 · 꺾임 항 0 + 위반 없으면 통과 | 4 | 1 / 3 / 0 | **5 / 0 (통과)** |
| **zr** (F4) | + LATCHED · CLOSING 손 · 손가락 행 중 target 수평 100 mm 안 해제 | **6** | **5 / 5 / 4** (안전 성공 규칙 ≥ 5 에 1 부족) | 4 / 1 (≥ 5 에 1 부족) |
| zr-oc (F5) | + 판정의 `occluded_target` frame 을 TO 에도 인증 | 5 | — | — |
| zr-oc-hbc | + F3 (쥔 물체 아래 표면 행 해제) | 6 | — | — |

- F4 는 쥐는 중 실패를 LIBERO 에서 0 으로 만들었다 (termsoff-pass 2). 남은 LIBERO 실패 t4 · t6 · t9 는 모두 쥔 후.
- 쥔 후 원인 (새 기록 `to.passthrough_clear`): zr HELD call 의 통과 거절 = 기하 미인증 67 · 충돌 6 · 한계 6. F5 는 미인증을 0 으로 만들었으나 (거절 = 충돌 32 · 한계 14) LIBERO 는 5 (1 episode 차이는 구분 못 함).
- SafeLIBERO t3 (캐비닛 위 그릇) 4/4 실패: latched 구간 통과 거절이 충돌 22 · 한계 13–18, HOLD 행 = 손 ↔ 비-target (F4 영역 밖) · 팔 link5 · 7.
- 기록 추가 (동작 불변): `to.passthrough_clear` (T88 결정), 그 안의 `limit_overshoot` (어느 한계를 얼마나 넘었나, `sqp._clear_passthrough`).

#### SL-GTTO7 · SL-GTTO8 — 쥔 동안 손 행 해제 (F3b) 와 TO 개입 증폭 진단 (2026-11-05 12:3x – 15:4x KST, G44 – G46)

| 판 | 정의 | LIBERO 9 | 실패 (단계) | held collision 거절 |
|---|---|---:|---|---:|
| zr-oc-hbc | zr + F5 + F3 | 6 | t4 · t7 · t9 | 30 |
| zr-oc-hbch (F3b) | + HELD 에서 손 · 손가락 행도 "아래 · 쥔 물체 발자국 안" 표면 해제 | 6 | t4 (쥔 후) · t7 (쥐는 중) · t9 (쥔 후) | 8 |
| zr-it8 (F6) | zr + `--sqp-min-iterations 8 --sqp-max-iterations 8` | 6 | t4 · t7 (쥐는 중) · t9 | — |

- F3b 는 작동했으나 (손 행 해제 call 당 수백 행, 거절 30 → 8) 성공을 바꾸지 못함 → gate (≥ 7) 닫힘, SafeLIBERO · LIBERO-Safety 미실행.
- **증폭 진단** (`experiments/segmentation/gtto_amplification.py`, `outputs/2026-11-05_GT인식_공용TO_3벤치/증폭진단/`): collision 으로 pass-through 가 거절된 call 은 zr 세 벤치 · oc · ochbc · hbch 모두 SQP 반복 cap 3 에 100 % 도달, 궤적 변경 / 정책 위반 중앙값 2.3–3.2 (변경 p90 79–171 mm), C 포화 행 비율 중앙값 0.6–1.0, \|Δa\| 중앙값 1.24–1.6 (±1 범위). 실패 episode 의 이런 call 은 episode 당 7.1 (zr) vs 성공 1.5 (상관).
- F6 (반복 8, 15:31–15:43): 6/9, 변경/위반 중앙값 3.02 → 2.43 (결과 전 규칙 "절반 이하" 불성립), 반복 8 에서도 cap 97 %, 변경 p90 101 → 63 mm. → 반복 수는 원인 층이 아니다; 증폭은 joint 공간 TO + conversion C 구조 쪽 (EE-delta QP 는 사용자 판정, G46).

![증폭](../../../outputs/2026-11-05_GT인식_공용TO_3벤치/증폭진단/그래프_증폭.png)

#### SL-GTTO9 — F7 정책 action 공간 QP (`--ee-action-qp`) 세 벤치 (2026-11-05 17:4x – 20:2x KST, G47 – G50)

- 구현 (17:4x): pass-through 가 거절된 chunk 에서 joint SQP 대신 `min ‖Δa‖²` (정책 OSC action 공간), 제약 = 같은 선형화 충돌 행 `value + g·M Δa + s ≥ backoff`
  + action box (hard), 4 회 재선형화, OSQP. M = `AG3S/adapters/panda/action.py` `ee_action_map` (OSC 응답 모델 × base 회전 × EE Jacobian 감쇠 pseudo-inverse).
  결과 궤적은 기존 `_finish` 를 거치고 SafePolicy 가 그 궤적을 반환할 때만 server 가 `clip(a_ref + Δa)` 를 보낸다. 판 `AG3S-commonto-gt2-termsoff-pass-zr-ee`. test 7 (`tests/trajopt/test_sl_gtto9_ee_action_qp.py`).
- run: LIBERO 18:37–18:48, SafeLIBERO 18:48–19:23, LIBERO-Safety 19:23–20:23 (SOLO, gate LIBERO ≥ 7 열림).

| 벤치 | VLA 단독 | zr | **zr-ee (F7)** | §5 |
|---|---|---|---|---|
| LIBERO 9 성공 | 9 | 6 | **8** | ✓ (처음) |
| SafeLIBERO 11 공식 / 충돌 / 안전 성공 | 11 / 9 / 2 | 5 / 5 / 4 | **6 / 3 / 6** | ✓ (처음) |
| LIBERO-Safety 10 성공∧무위반 / 위반 | 7 / 1 | 4 / 1 | 3 / 0 | ✗ (기준 5) |
| └ human_safety (4) | 3 | 1 | 0 | |
| 개입 call C 포화 중앙값 LIBERO · SafeLIBERO · LIBERO-Safety | — | 0.9 · 0.9 · 0.6 | 0.0 · 0.2 · 0.5 | |
| 개입 call \|Δa\| 중앙값 | — | 1.6 · 1.34 · 1.24 | 1.16 · 0.95 · 0.73 | |

- LIBERO-Safety 실패 7 모두 쥔 후, 쥔 후 HOLD 비율 0.356 (zr 0.03). HOLD 충돌 행 step 0 = 188 · 1 = 132 · 2–4 = 69 — 지금 자세가 이미 위반이면 어떤 action 으로도
  못 고치고 (step 1 도 action 1 당 3.6 mm), hard box 의 F7 은 slack → 최종 검사 HOLD → 접촉 자세에서 deadlock. 원인 층 = 쥔 후 grasp handling · 판정 규칙 (solver 아님).
- 한계: OSQP 대부분 `maximum iterations reached` (20000), solve ~1.5 s. offline (scratchpad `ee_qp_scaling.py`) 행 정규화 + quadratic slack 은 150 회 · 5 ms 에 엄밀해 — 미적용 (사용자 판정).
- 그림: `outputs/2026-11-05_GT인식_공용TO_3벤치/gt10_ee/` (`성공grid.png`, `그래프_HOLD비율.png`, `결과표.md`), `단계분석/`, `증폭진단/`.

![F7 성공 grid](../../../outputs/2026-11-05_GT인식_공용TO_3벤치/gt10_ee/성공grid.png)

### 측정하지 않은 것 (현재)

- gt2 30 장면 본 실행 · SL-GTTO3 결과 (queue 대기, 이 절에 추가 예정).
- ③ 팔 link 충돌은 고치지 않았다. `answer_excludes_target` (서랍 · 캐비닛 같은 실제 비-target 기하와 손 · 손가락) 행은 세기만 했다.
- 변환 C 자체(포화 시 부족분 처리)는 바꾸지 않았다.

## SL-GTTO10/11 — 쥔 물체 · 목적지 접촉 · 작업 단계 (2026-11-07 00:2x – KST, 사용자 주의 1–3)

판정 정본 `doc/active/experiments/E1_GTTO_DECISIONS_20261104_KO.md` G51–G55+, 보고 `doc/active/experiments/E1_GTTO_REPORT_20261107_KO.md`.
merge `ff21caa5` (원격 SL9–SL16 + SL-GTTO) 위 미commit patch. 평가 규약은 원격 것 (LIBERO 9 × seed 7·8·9 = 27, LS 10, SafeLIBERO 11).

### 진단 (00:2x – 02:3x)

- **00:21 사고:** 진단 run 을 바깥 `flock` 안에서 띄워 launcher 내부 lock 과 self-deadlock (GPU 미사용 1.5 시간). 01:58 lock 없이 재실행 (`AG3S_GPU_LOCK_HELD=1` 없이 바깥 flock 금지).
- **J0:** F7 LS human_safety e0 기록 재실행이 원래 run 과 call 배열 · env action bit 동일 → merge · 기록이 F7 동작 불변.
- **voxel 주인 (새 도구 `gtto_ls_geomsnap.py` · `gtto_held_voxel_origin.py`):** e0 carry HOLD 76 행 = 쥔 그릇 ↔ **사람 손 (목적지)**, e2 carry HOLD 58 행 = 손 ↔ **로봇 손 자기 voxel** (손 주변에 쥔 그릇 voxel 1571 / call).
  human_safety 다섯 과제는 모두 `On(쥔 물체, 사람 손 | 손 위 접시)` + `CheckRobotContact` — 쥔 물체는 닿아야, 로봇은 닿으면 안 된다. GT code image 에 목적지가 없어 서버 `destination = null`.
- **LS VLA 기준:** 이 기계 VLA 단독 human_safety 두 run (다른 commit) 이 bit 동일 → 7/10 이 이 기계 값, 원격 10/10 은 다른 기계.

![F7 e0 voxel 주인](../../../outputs/2026-11-07_쥔물체_놓기접촉_단계진단/voxel출처/장면_human_safety_L0_t0_e0_seq44.png)

### F8 `AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp` (02:37 – )

F7 + (F8a) 새 `--oracle-destination` (BDDL 목적지 code 의 fused 점을 매 frame 목적지로 주입, 쥔 물체 행만 destination margin) + 새 `--destination-margin-mm 0` + F3 `--held-beneath-contact`;
(F8b) 원격 `--held-free-query-spheres` + `--hand-voxel-carve current 20`; (F8c) 원격 `--grasp-opening-rule`. test 새 7 + 관련 suite passed.

| | VLA 단독 | F7 | **F8** |
|---|---|---|---|
| LIBERO-Safety 10 성공∧무위반 (위반) | 7 (1) | 3 (0) | **7 (0)** |
| human_safety · obstacle · affordance | 3/4 · 3/4 · 1/2 | 0/4 · 3/4 · 0/2 | 3/4 · 4/4 · 0/2 |
| LIBERO seed 7 · 8 · 9 | 9 · 9 · 9 = 27/27 | 8 · 6 · 6 = 20/27 | **8 · 7 · 7 = 22/27** |
| SafeLIBERO 성공 · 충돌 · 안전 성공 | 11 · 9 · 2 | 6 · 3 · 6 | **7 · 3 · 5** |

- 기제 (human_safety, F7 같은 episode): 목적지 보유 0 → 1.00, carry `attached_point` HOLD 행 82 → 0, 손 · 쥔 물체 HOLD 행 165 → 0, carry 중 latch held 0.77 → 0.96 (모두 G53 규칙 통과).
- **F8 의 남은 실패 = 원격 opening rule 의 '열린 손 attach'** (닫힘 명령 첫 call 의 개도 변화 < settle 0.02 를 정지로 읽음): LIBERO 실패 5 개 전부 + LS human_safety e3. attach 25 회 중 7 회가 개도 > 0.9.

![F7 vs F8 실제 장면](../../../outputs/2026-11-07_F8_쥔물체목적지단계/장면영상/장면비교_F7_vs_F8_hs.png)

### F9 `…-hdp-qs` (등록 03:0x, queue `gtto11_queue.sh`)

F8 + (a) `--ee-action-qp-scaled` (행 정규화 + quadratic slack) + (c′) `--grasp-opening-rule-moved 0.05` (마지막 열림 명령 때 개도 − 지금 개도 ≥ 0.05 일 때만 attach). test 새 13.

**F8 판정 (G57):** 원격 후보 규칙 (LIBERO ≥ 17/27 · LS ≥ 6 ∧ 위반 0 · SafeLIBERO 충돌 ≤ 5 ∧ 안전 성공 ≥ 1) 세 벤치 모두 처음 통과. §5 는 LIBERO seed 별 ≥ 8 만 미달.

| (04:45 – 06:13) | F8 | **F9** |
|---|---|---|
| LIBERO seed 7 · 8 · 9 | 22 (8 · 7 · 7) | **23 (7 · 8 · 8)** |
| SafeLIBERO 성공 · 충돌 · 안전 성공 | 7 · 3 · 5 | **10 · 5 · 6** |
| LIBERO-Safety (위반) [aff · hs · oa] | 7 (0) [0 · 3 · 4] | **8 (0) [0 · 4 · 4]** |
| OSQP 상한 도달 · 반복 중앙값 | 0.89 · 20000 | 0.20 · 300 |
| 개도 > 0.9 attach | 16 | 0 |

- **F9 채택 판정 (G54 규칙, G59): 미달** — OSQP 상한 0.20 > 0.10 (상한 풀이의 92 % 는 slack > 1 mm 실현 불가능 풀이), SafeLIBERO 충돌 5 > F8 + 1. 후보 = F8. 사용자 판정 항목.
- 남은 실패 층: LIBERO t4 (서랍) 접근 단계 collision 개입, 쥔 뒤 팔 link7 `unauthorized_link` HOLD (F9 s7 t7, SafeLIBERO t1e4 (F8)),
  affordance 의 closing 중 아주 작은 관절 한계 초과 (위치 0.009 rad, 가속도 0.0007) 로 pass-through 거절 → |Δa| 0.2–0.9 개입.

![27/10/11](../../../outputs/2026-11-07_결과_27_10_11/그래프_27_10_11.png)

![F8 vs F9 실제 장면](../../../outputs/2026-11-07_F9_스케일QP_닫힘확인/장면영상/장면비교_F8_vs_F9_열린손attach.png)

### seed 8 짝 확인 · HVC 가설 (06:24 – 08:24, G56 · G61 · G63 · G64)

- **후보 F8 (G54 gate 가 F9 를 탈락시킴) seed 8:** LS 5 (위반 1) · SafeLIBERO 성공 9 / 충돌 3 / 안전 성공 7 — VLA seed 8 (LS 6 · 위반 2, SafeLIBERO 9 / 9 / 2) 대비 G56 세 규칙 모두 통과.
  F8 의 첫 공식 위반 = seed 8 oa e0 쥔 그릇 ↔ 장애물 (서버 필드가 직전에 위반을 못 봄; 기록 없음).
- VLA 단독 LIBERO seed 8 · 9 = 9 · 9 → 이 기계에서도 27/27.
- **HVC 가설 기각:** F7 · F8 SafeLIBERO t2 기록 run (채점 run 과 call 배열 bit 동일) 의 t2e4 seq 3 필드에서 머그컵 GT 표면 3 cm 안 점유 voxel 9768 · 9753.
  F8 · F9 의 t2e4 충돌은 reference 위반 0.12 mm vs 0 의 문턱 뒤집힘.

![seed 7 vs 8](../../../outputs/2026-11-07_후보_seed8확인/그래프_seed7_8.png)
- **F9 seed 8 (정보용, 08:24 – 09:40):** LS 2 (위반 1) [aff 0 · hs 1 · oa 1] · SafeLIBERO 6 / 6 / 3 → seed 8 규칙 셋 중 둘 미달 (F8 은 셋 다 통과).
  hs 실패 = (c′) 로 attach 가 늦어 닫는 동안 손가락 ↔ 쥘 물체 행이 남아 쥐기 반복. oa e1 · e3 = 놓기 단계 손 step 0 deadlock (`panda_hand` ↔ coarse −20.7 mm, 손이 탁자 높이)
  — 승인 계획 (b) CBF 회복 규칙이 겨냥하는 첫 실제 사례. **최종 후보 = F8** (G66).
- test (09:0x): `tests/benchmark` + `tests/trajopt/test_sl_gtto*` 에서 기존 실패 밖의 새 실패 0 (env.sh 환경). `test_tool_does_not_modify_server_adapter_or_recorder` 는 미commit 수정 때문에 실패.

## SL-GTTO12–15 — LIBERO 동률 (2026-11-07 19:10–21:52 KST, 사용자 판정 19:1x)

**판정:** "LIBERO 에서 VLA 단독이 충돌 없이 성공한 episode 는 우리 모듈에서도 반드시 성공해야 한다 — 제약을 느슨하게 해서 LIBERO 성공률을 VLA 단독과 같게." 판정 기록 G67–G73.

| 시각 (KST) | 일 | 결과 |
|---|---|---|
| 19:1x | F7 · F8 · F9 LIBERO 27 의 '바꾼 · 멈춘 call' 원인 분류 (`gtto_libero_fp.py`) | F8: 인증 안 됨 188 (175 는 VLA 경로 위반 0) · collision 36 (중앙값 8 mm) · HOLD 쥔 물체 119 |
| 19:20–20:09 | GT12 관찰 전용 F8 — LIBERO seed 10 · 11, SafeLIBERO seed 10, `--record-constraints full` | VLA 성공 17 episode 에서 바꿨을 162 · 멈췄을 9; SafeLIBERO VLA 충돌 10/11, F8 충돌 전 개입 10/10 |
| 19:4x | 남은 거짓 양성 voxel 주인 (geom snapshot) | 열린 손 attach (개도 0.986 · 0.988) 로 쥔 물체 구가 탁자 속 −84 mm; 팔 · 손 행 ↔ 로봇 자기 voxel |
| 19:3x–20:0x | 새 기본 꺼짐 flag: `--violation-tolerance-mm` (pass-through + 최종 판정 τ), `--oracle-robot-mask` (GT code 0 pixel 을 TSDF 에서 뺌), `--passthrough-limit-tolerance`; 기록 전용 `passthrough_clear.reference_worst` | test `tests/trajopt/test_sl_gtto14_violation_tolerance.py` 9 + 판 · T88 suite 171 passed |
| 20:09–20:31 | GT14 관찰 전용 F11 (F8 + F5 + (c′) + robot mask + τ 25 + limit 0.01) — seed 10 · 11 | 바꿨을 7 · 멈췄을 0, 남은 7 개 모두 관절 한계 추정치 초과 (acc 최대 0.128) |
| 20:32–21:00 | **F12** (`…-hdp-loose2`, limit 0.2) closed loop LIBERO 27 | **27/27 = VLA**, 27 episode 모두 개입 0 · HOLD 0 |
| 21:00–21:52 | F12 SafeLIBERO 11 · LIBERO-Safety 10 | SafeLIBERO 9 · 충돌 8 · 안전 성공 3 (VLA 11 · 9 · 2, F8 7 · 3 · 5); LS 7 (1) = VLA |

**결론:** LIBERO 동률은 달성했으나 SafeLIBERO 충돌 회피는 거의 VLA 수준으로 돌아갔다 (결과 전 F8 관찰 계산: τ 25 → 충돌 전 개입 10/10 → 4/10). F8 의 SafeLIBERO 회피 일부는 '인증 안 됨' QP 의 흔들림이 만든 우연이었다 (충돌 직전 VLA 경로 위반 0). F12 가 놓친 진짜 위험은 쥔 물체 ↔ 장애물 행 (−20.4 mm) 이었다. **다음 판 제안:** 장애물 물체 (GT code ≥ 3, target · 목적지 아님) 행에는 τ 를 적용하지 않음. 산출: `outputs/2026-11-07_LIBERO_과민제약_진단/`, `outputs/2026-11-07_LIBERO_동률_F12/`.


## AG3S 모듈별 리뷰 · 검증 (offline, 2026-11-08 00:45 – KST, 세션 dev-49)

**사용자 지시 (2026-11-08 00:5x):** 로봇 마스크처럼 AG3S 의 모든 모듈을 같은 방식으로 검증한다. 잘 되면 잘 됐다, 안 되면 이유를 적고 넘어간다. 13:00 까지 자율 판정한다.

- **정본:** `doc/active/experiments/AG3S_MODULE_REVIEW_20261108_KO.md` (요약 판정표 + §1–§20).
- **방법:** 모두 CPU offline 이다. GPU 는 dev-ec 가 독점으로 쓰고 있었다.
  - 기록된 run 의 장면을 rollout env 이력 그대로 재구성했다 (RGB bit 동일 확인).
  - 서버 함수를 기록 입력에 다시 돌려 MuJoCo 진실 (image instant segmentation · 접촉 · geom 표면) 과 대조했다.
- **공유 코드는 수정하지 않았다.** figure 와 스크립트는 `outputs/2026-11-08_*` 에 있다.

| 판정 | 모듈 |
|---|---|
| 잘 됨 | depth → point cloud (오차 중앙 1.1 / 0.3 mm) · RGB 매핑 (unmirror) · Panda 수정 render mask (recall 0.96–1.00) · attach 판정 (26 / 26 양손 접촉) |
| 부분 | attention (접근 단계만) · TSDF / ESDF (보이는 면만, 뒷면 2–3 cm 그림자) · target-free 층 · sphere clearance (놓침 45 / 132, 거짓 566) · 응답 모델 (개입 call 33–39 mm 오차) · zone release (코드상 구멍) |
| 안 됨 | GT 없는 grounding (접근 정답 0.54) · P0m surface extent (쥐기 전 target 표면 ~6 % 만 지움; 두 번째 RANSAC 평면) · latch × grounding (틀린 첫 identity 고정, 0.54 → 0.44) · HVC (7 / 7 episode 에서 장애물 voxel 8215 개 지움) · 동적 거리장 (ghost) · F12 τ 25 mm (SafeLIBERO 충돌 8 / 8 통과) |


### AG3S 모듈 리뷰 — 추가 (2026-11-08 04:00–09:30 KST, 세션 dev-49)

정본: `doc/active/experiments/AG3S_MODULE_REVIEW_20261108_KO.md`. dev-ec 의 아침 run 기록 (gt19pts · B3 gt21pts · B5 gt23t100 · GT 없는 B5 gt24att) 을 같은 방식 (장면 재구성 + MuJoCo 진실 대조) 으로 다시 쟀다.

| 항목 | 결과 |
|---|---|
| HVC (§17.5–17.7) | 켜면 최신 closed-loop 에서도 9 / 11 episode 에서 장애물 voxel 을 지운다. 공통 9 episode 장애물 놓침: gt19pts 16 / 16, B3 (끔) 4 / 34, B5 10 / 11 (빈손 deadlock 한 순간) |
| HVC 끔의 비용 (§17.6) | B3 t3 e3: 쓰러진 와인병의 경로 ghost 로 HOLD 45 call. 근본은 동적 거리장 (Panda 미이식) |
| 쥔 물체 점 모델 (§13.5) | 들어 올린 뒤 그릇 아랫면이 모델에서 빠진다 (바닥이 8–45 mm 위). 반쯤 닫힌 attach (개도 0.375) 뒤 덮임이 0.39 → 0.06 |
| 판정 층 (§14.4) | gt19pts 충돌 5 / 11: 쥔 물체 행 통과 1 · 장애물 τ 10 통과 1 · TO 개입 후 충돌 3 |
| latch (§6.7–6.8) | reach 문턱 120 mm 로 판당 4–6 / 48 episode 가 attach 못 함. B5 t3 e0 는 빈손 CLOSING + 손목 카메라 기여 0 으로 HOLD deadlock |
| GT 없는 판 closed-loop (§21) | LIBERO s7: 7 task 는 target 을 맞게 잡음. task 4 · 9 는 다른 물체를 target 으로 잡아 그것을 지우고 실패 (offline 예측과 일치) |

## SL-GTTO16–24 — 장애물 행 엄격 (A) · 쥔 물체 점모델 (B) · GT 없는 판 (2026-11-08 00:25–10:4x KST, 세션 dev-ec)

**사용자 지시 (00:2x):** A (τ 를 장애물에 적용 안 함) → B (쥔 물체를 실제 형상 점으로) → 각 실패 원인 분석, 13:00 까지 자율. 목표: LIBERO · SafeLIBERO 성공률 VLA 대비 ≤ 10 %p 차 + 충돌 회피 훨씬 좋게; 좋으면 GT 없이. 판정 기록 `doc/active/experiments/E1_GTTO_DECISIONS_20261104_KO.md` G74–G99, 요약 `outputs/2026-11-08_최종요약/README.md`.

| 시각 | 일 | 결과 |
|---|---|---|
| 00:3x–01:3x | A (`--passthrough-obstacle-tolerance-mm`, TO 표면 분류) · 보정 | 첫 실행 shape 오류로 무효 → 고침; seed 10 거짓 양성 6/9 episode (접시 · 캐비닛 윗면이 장애물로) |
| 01:3x–02:1x | A′ (beneath + τ_obs 10) 관찰 · B (`--held-point-model-mm 20`) 구현 | A′ 보정 3/17; SafeLIBERO 충돌 전 개입 8/10 |
| 02:09–02:46 | B′ closed loop | LIBERO 23/27 (A′ 분류 실패) |
| 02:3x–04:22 | A″ (`--oracle-obstacle-objects`, GT code ≥ 3 − target · 목적지 · 받침) + B = B″ | LIBERO 27/27 · SafeLIBERO 7 · 5 · LS 7 (0) |
| 03:4x–06:55 | 받침 'under' + HVC · held-free carve 끔 (dev-49 §14) → A‴ · B‴ | A‴ 27 · 10 · 6 · 6 (1); **B‴ 27 · 10 · 6 · 7 (0)** |
| 04:3x–08:12 | B5 = B‴ + 장애물 행 chunk 전체 + 일반 τ 100 | 27 · 9 · 2 · **9 (0)** |
| 08:51–09:52 | SafeLIBERO seed 8 (B‴ · B5) | 2 seed 22: VLA 20 · 실제 충돌 11, **B‴ 18 (−9.1 %p) · 4**, B5 17 (−13.6 %p) · 4 (기록상 7) |
| 08:11–10:4x | GT 없는 B5 (`--scene-obstacle-objects`) | LIBERO 17/27 · SafeLIBERO 4/11 · LS 3 (0); 거절 행 옆 = target 자신 49/62 (인식 병목) |
| 01:5x– | SafeLIBERO '충돌' 재생 분석 (`gtto_collision_contacts.py`) | 공식 1 mm 규칙의 충돌 중 다수는 로봇 접촉 없음 (VLA seed 7: 9 중 4) |

**결론:** 목표 (성공 ≤ 10 %p + 충돌 회피 훨씬 좋게) 를 채운 판 = **B‴** `AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst3-pts-nocarve` (GT 인식). 회피 최우선이면 B5. 남은 한계: SafeLIBERO t3 팔 행 반복 개입, 빈손 닫힘 → 손목 카메라 dead → HOLD 고리, TSDF decay 부재, GT 없는 target identity. 실제 장면: 판마다 `outputs/2026-11-08_*/영상_agent+wrist/` · `AG3S_기능장면/`. 모든 새 flag 기본 꺼짐; commit 안 함.


### AG3S 모듈 리뷰 — 마감 추가 (2026-11-08 09:30–11:30 KST, 세션 dev-49)

| 항목 | 결과 (정본 `doc/active/experiments/AG3S_MODULE_REVIEW_20261108_KO.md`) |
|---|---|
| noun head L8H6 (§2.5, GPU 10:5x–11:15 offline 추출, lock 사용 · peer 알림) | head approach target mass 0.45 (L8H2 0.21), 쥔 뒤에도 target 유지 (wrist 0.90). 대신 같은 범주 distractor 0.24 |
| P0m 본판 서버 경로 (§5.7) | noun head 로도 쥐기 전 target 표면의 평균 11 % 만 지움 (L8H2 6 %). memory 의 35 % 는 distractor |
| GT 없는 판 closed-loop (§21, B5 att LIBERO 27 + SafeLIBERO) | LIBERO 실패 10 중 8 이 identity (task 3 · 4 · 9: 다른 물체를 target 으로 잡아 그것을 지움), 2 는 운반 중 obstacle 거절 (task 7). SafeLIBERO t3 는 모두 다른 물체. offline 예측 (§4 · §16) 과 일치 |
