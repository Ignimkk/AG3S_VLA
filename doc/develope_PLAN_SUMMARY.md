# develope_PLAN 요약 — V11AB Benchmark · Baseline · Ablation · 실물 로봇

작성: 2026-10-07 KST · 갱신: 2026-10-12 01:11 KST · 문서 버전: **1.1**
상태: `PLANNED_V11AB` — 이번 수정은 실험 계획 변경이며 benchmark 실행 완료를 뜻하지 않는다. 실제 수행 상태는 최신 handoff와 run manifest로 확인한다.
작성 배경: 기존 `develope_PLAN.md`, `paper/notes/SAFELIBERO_PLAN.md`, `paper/notes/REFERENCE_ANALYSIS_R2.md`.
이 자료들은 이전 설계의 참고 기록이다. 실행 기준은 본 문서와 benchmark checkout의 `doc/BRANCHES.md`, `doc/migration/source-manifest.json` 및 실제 source/config다.
용도: Claude Code가 V11AB Panda adapter와 benchmark 평가를 단계별로 구현·검증할 때의 실행 계획.

---

## 0. 한눈에

- **평가 대상 확정:** 기존 benchmark/v2와 pi05_TO_hybrid/v2를 통합한 **V11AB**. 개선 branch의 후속 algorithm을 섞지 않는다.
- **논문 한 줄(검증할 가설):** 외부 segmentation model 없이 VLA attention + depth로 target geometry를 얻고, 이를 link-specific contact constraint에 연결해 frozen VLA의 조작 실행을 개선한다. Phase 정보는 contact exemption의 유효 시점을 조절한다.
- **중심 검증:** ATS → target geometry → LTE. HSR·CCR의 독립 기여는 V11AB의 실제 구현과 ablation에서 추가 이득이 확인될 때만 주장한다.
- **실행 순서:** **Panda adapter → LIBERO Spatial 소규모 평가 → SafeLIBERO Spatial Level I/II pilot → 전체 SafeLIBERO → LIBERO-Safety 추가 평가**.
- **비교 순서:** pi0.5-only vs pi0.5+V11AB 먼저, 그다음 AEGIS, 이후 KNOWS 재구현·Lan-o3dp 관련 실험.
- **본문 목표:** Figure 3개 + Main/Ablation Table + 짧은 runtime/설정 표. 실제로 검증한 범위만 주장한다.
- **CVPR 2027 제출 일정:** 등록 2026-11-10 · 본문 11-16 · 보충자료 11-23 (AoE). 2026-10-12에 [공식 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)와 [Dates](https://cvpr.thecvf.com/Conferences/2027/Dates)를 재확인했다. 내부 주장 고정 목표 10/21은 진행 상태에 따라 재판정한다.

| 약어 | 뜻 / 적용 조건 |
|---|---|
| ATS | Attention-based Target Segmentation — attention + depth로 target 선택·추출 |
| HSR | Hidden-state Sub-task Recognition — frozen VLA feature 위의 phase read-out; V11AB의 실제 구현·학습 상태를 audit한 후 적용 범위를 기록 |
| LTE | Link-specific Target Exemption — 허용된 link에만 target 접촉을 허용 |
| CCR | Constrained Chunk Refinement — 실행할 action chunk를 constraint 하에서 정제 |
| Attached object | 잡은 물체를 collision query에 포함 |

이 약어는 연구·분석 용어다. 해당 기능이나 ablation switch가 V11AB에 이미 있다고 가정하지 않는다.

### 0.1 V11AB 고정 기준과 branch 운영

| 항목 | 고정 기준 |
|---|---|
| 작업 checkout / branch | `/mnt/dev/AG3S_VLA-benchmark` / `feature/panda-benchmark` |
| benchmark 원본 v2 commit | `legacy/benchmark/v2` → `498f713b20bf9b75b4b065cd9bcdd508d7cdaecf` |
| pi05 원본 v2 commit | `legacy/pi05_TO_hybrid/v2` → `f62ab814e867912af3cecea52519d00c470d39e3` |
| 실제 실행 버전 | 매 run의 monorepo commit·adapter commit/config·dependency pin·checkpoint checksum을 manifest에 기록 |
| V11AB 설정 확인 자료 | `scripts/benchmark/v11ab/env.sh`, `start_server.sh` 및 실제 server/client config |
| 보호 원칙 | V7g/V8로 기준을 재선정하지 않고, 개선 branch 전체 merge 및 원본 tag checkout/reset을 하지 않는다 |

현재 V11AB launcher의 추가 설정은 A+B다: `plan-horizon 16`, obstacle lookahead,
`lookahead-gate-mm 20`, `qp-eps 1e-4`, `w-deflection-rate 20`. 공통 설정에는
`links gripper`, `no-self-collision`, `target-field-policy exclude-authorized`, held-body/finger coverage 등이 있다.
이 launcher는 **RB-Y1용 역사적 설정 확인 자료**이며 Panda runner가 아니다. 전체 flag와 default를 audit한다.
Gripper protection을 전신 collision avoidance로 표현하지 않으며, 독립 evaluator의 검사 범위도 따로 기록한다.

Panda의 robot geometry·camera calibration·관측/action 좌표 변환·VLA checkpoint/norm stats는 adapter에서 연결한다.
V11AB의 목적함수·constraint·gate·chunk refinement 동작은 유지한다. 기존 threshold 변경이 필요하면
단위/좌표 변환인지 method 변경인지 근거를 남긴다. Method 변경·추가 HSR 학습·보호 link 확대·
EE 공간 최적화는 별도 variant로 표시하며 V11AB main 결과에 합치지 않는다.

---

## 1. 이전 계획에서 확정·변경된 사항

| 항목 | 이번 실행 기준 | 영향 |
|---|---|---|
| AG3S snapshot | **V11AB 확정** | 이전 V7g/V8 후보 및 결과는 historical reference |
| Benchmark | LIBERO → SafeLIBERO → LIBERO-Safety; RB-Y1 MuJoCo는 상세 mechanism/ablation | 작은 평가로 adapter를 확인한 뒤 전체 평가로 확장 |
| Baseline | pi0.5-only 필수 → AEGIS 우선 → KNOWS/Lan-o3dp 추가 | 다섯 조건을 동시에 구현하는 일정으로 잡지 않음 |
| 공개 코드 여부 | 이전 원문 정리의 기록을 참고하되 구현 착수 시 official source·revision 확인 | 미공개/미확인 방법은 재구현 범위·차이를 명시 |
| Phase 처리 | V11AB 동작 유지 | strong rule 대체를 기본값으로 넣지 않음; 대체 조건은 독립 variant |
| 실물 로봇 | RB-Y1 AG3S 실행 가능성 시연 우선 | 비교 trial이 부족하면 정량 개선을 주장하지 않음 |

---

## 2. Benchmark — 무엇을, 어디서

| 환경 | 역할 | 실행 조건 | 규모 |
|---|---|---|---|
| **LIBERO** | Spatial에서 adapter smoke/pilot; 이후 기본 task 성능 보존 평가 | Panda-compatible pi0.5 checkpoint, audited adapter; official success | Pilot task/episode subset를 먼저 고정. 전체 범위는 suite·task·initial-state 목록으로 선언 |
| **SafeLIBERO** | 충돌회피 main 결과 | 동일 pi0.5 checkpoint·adapter; Level I(target 근처)/II(경로 차단); OSC_POSE와 20 Hz 설정은 pinned source/runtime에서 확인 | **4 suites × suite당 4 tasks × 2 levels × 50 episodes = 1,600 episodes/조건**. Spatial 전체는 400/조건이며 pilot은 선언한 subset |
| **LIBERO-Safety** | 다른 안전 task 분포에서 추가 평가 | 해당 benchmark의 독립 adapter·공식 protocol·success/safety 지표 | 선택 suite·task·level·episode 수와 제외 범위를 명시. SafeLIBERO level 정의를 적용하지 않음 |
| **MuJoCo (RB-Y1 자체 과제)** | V11AB 상세 ablation·mechanism figure·held-out 평가 | 기존 16-D joint stack과 V11AB 고정 config | 개발 scene과 분리한 held-out scene family |

SafeLIBERO 규모는 [official README](https://github.com/THU-RCSCT/vlsa-aegis#-overview) 기준이다.
표의 16 tasks는 전체 suite 합계이며 suite마다 16개가 아니다. 다섯 조건을 모두 수행하면 E1만 8,000 episodes다.
KNOWS 관련 Level III는 별도 확장 실험이며 공식 SafeLIBERO I/II main과 분리한다.
원본 LIBERO는 **추가 safety obstacle을 주입하지 않은 환경**이다. 기존 scene의 다른 물체·지지면이 없는 환경이라는 뜻은 아니다.
LIBERO-Safety의 범위는 [official repository](https://github.com/LIBERO-SAFETY/LIBERO-Safety)와 pinned task map에서 확인한다.

### 2.1 Panda adapter와 평가 구현 (SL-0 ~ SL-8)

| ID | 구현 | 완료 확인 |
|---|---|---|
| SL-0 | 자기 checkout의 OpenPI Python 3.11.x·cuRobo 환경, pinned benchmark/assets, Panda-compatible pi0.5 checkpoint/norm stats 연결 | own-root import, GPU·checkpoint 로딩, pi0.5-only LIBERO Spatial 1 episode 공식 판정/기록 |
| SL-1 | Agent-view·wrist RGB/depth, intrinsics/extrinsics, image orientation 및 frame convention → AG3S 관측 형식 | depth back-projection·target/obstacle overlay가 실제 scene과 일치 |
| SL-2 | Panda joint/gripper/action 단위와 geometry, query sphere, self-filter mapping | mesh–sphere overlay 및 allowed/forbidden contact scene 검증; protected link 목록 확정 |
| SL-3 | V11AB attention/read-out의 실제 동작 audit 및 pi0.5-LIBERO 입력·출력 연결 | 추출 단계·head·token 대응을 기록. KNOWS의 L12·H3를 자동 채택하지 않음; 선택이 필요하면 dev-only calibration/variant를 선언 |
| SL-4 | 기존 ATS/LTE·multi-view geometry·ESDF 연결; authorized link별 target 처리 확인 | target 제거가 non-target obstacle이나 비허용 contact를 지우지 않는지 overlay/fixture로 검증 |
| SL-5 | EE delta ↔ 기존 joint-space CCR 변환 adapter; IK·OSC의 frame·scale·rotation convention·gripper 처리 명시 | 변환의 consistency, joint limit, 실제 controller 실행 후 clearance/contact 재검사. EE 공간 직접 최적화는 별도 variant |
| SL-6 | V11AB phase/gate source·상태 이력 유지 및 Panda 입력 mapping | 지원 여부/동작을 audit. 필요한 read-out이 이식되지 않으면 blocker로 기록; strong-rule 대체 결과는 별도 표시 |
| SL-7 | Official task success + 독립 contact evaluator + 원자료 logger + benchmark별 metric 구현 | 알려진 contact scene에서 expected 판정 일치; 동일 조건 baseline 재현 차이를 분석하고 허용 범위를 사전 기록 |
| SL-8 | pi0.5-only vs V11AB 비교 runner, 이후 AEGIS·추가 baseline 연결 | 동일 task·initial state·seed·checkpoint·controller·episode horizon; 각 셀의 출처 및 설정 표시 |

`.venv-ag3s`는 CPU 회귀 환경이다. VLA 실행 전 자기 `.venv-openpi-live`와 필요 시 별도 cuRobo 환경을 검증한다.
RB-Y1 checkpoint를 Panda에 그대로 사용하지 않는다. Camera/action norm·quaternion/axis-angle·delta composition 및
predicted trajectory와 실제 OSC 실행 차이를 확인한다. 계획상 clearance를 실제 무충돌 결과로 대체하지 않는다.

### 2.2 Contact evaluator와 보호 범위

- RB-Y1과 Panda 모두 V11AB의 검사 link·geometry·gate를 고정한다. Geometry 변경 시 영향받는 모든 비교 조건을 재실행한다.
- LTE 비교 전에 실제 protected links와 authorized links를 audit한다. 비허용 link가 없어 두 조건이 동일하면 그 사실을 보고한다. 차이를 만들기 위한 보호 범위 확대는 main에 섞지 않는다.
- 독립 evaluator는 method가 예측한 target/mask와 무관하게 simulator task identity·actual contact로 allowed/forbidden을 판정한다.
- Simulator GT target/geometry를 쓰는 evaluator와 AG3S의 입력을 구분한다. GT planning 입력은 oracle 조건에만 사용하며 privileged input으로 표시한다.

| Contact pair | 자체 evaluator 판정 |
|---|---|
| task/phase별 허용 link – 실제 target | 허용 (grasp 등 task 수행에 필요한 접촉) |
| 비허용 link – target | forbidden |
| robot – 비대상 물체/추가 장애물 | task protocol에서 허용한 contact 외에는 forbidden |
| 잡은 물체 – 장애물 | forbidden |
| 물체 – 목적지/지지면 | 정상 placement 등 선언한 task contact만 허용; table 일괄 면제 금지 |

이 확장 evaluator와 benchmark 공식 collision 판정이 다르면 두 결과를 구분해 보고한다.
손잡이·서랍·스위치 등 pick-and-place 외 task도 사전에 contact 허용 규칙을 정의한다.

---

## 3. Baseline — 누구와, 어떻게 비교

아래 방법 설명은 이전 원문 정리의 요약이다. 구현 시 official paper/code를 확인하고 실제 사용 범위를 기록한다.

| Baseline | 방법 핵심 / 비교 역할 | 구현·표기 방식 | 우선 |
|---|---|---|---|
| **pi0.5-only** | 동일 frozen VLA에서 AG3S safety layer를 사용하지 않는 기준 | 동일 checkpoint·norm stats·controller·task/seed; AG3S 비활성화 경로 audit | 필수, 최초 |
| **pi0.5+V11AB** | 현재 고정 AG3S 구성의 효과 | 실제 V11AB config 및 Panda adapter 차이를 manifest에 기록 | 필수, 최초 |
| **VLSA / AEGIS** | VLM/detector geometry 기반 CBF safety layer; 공개 구현 비교 | Official code·revision·지원 checkpoint 확인 후 re-run. Checkpoint가 다르면 별도 블록 | 우선 추가 |
| **KNOWS** | 외부 segmentation·attention target 선택·ellipsoid·EE CBF-QP | 공개 구현을 먼저 확인. 재구현 시 `KNOWS (re-impl.)`; 핵심만 구현했다면 partial 범위·누락 기능 명시 | 추가 |
| **Lan-o3dp** | Point-cloud diffusion policy의 cost guidance | 원래 policy 재실행, perception ablation, pi0.5 guidance 이식을 구분; 아래 3.1 참조 | 선택 추가 |

### 3.1 공정 비교와 variant 이름

- **Reported·re-run·re-implemented·adapted 결과를 구분한다.** Checkpoint·task·level·episode 수·metric 정의가 같을 때만 동일 비교 블록에 둔다.
- 같은 initial-state/seed 목록으로 paired 비교한다. Dev와 test를 분리하고 test 결과로 attention head·threshold·baseline hyperparameter를 선택하지 않는다.
- Raw contact event·official success를 보존하고 CAR/TSR/ETS, SR/CR/SSR는 원문의 정의와 집계 방식이 확인된 것만 계산한다. 이름이 같아도 같은 지표라고 가정하지 않는다.
- 표에 protected links, privileged input(RGB/depth/GT), 추가 model/학습, controller 및 runtime 비용을 기록한다. Gripper-only 결과를 whole-body 결과로 표시하지 않는다.
- KNOWS의 원문 `Naive` 조건과 우리 재구현 명칭을 혼동하지 않는다. Partial 재구현으로 원래 method 전체를 대표하지 않는다.
- Lan-o3dp **perception만 사용**하면 `Lan-o3dp-inspired perception` ablation이다. Guidance를 pi0.5에 이식하면 `pi0.5 + Lan-o3dp-inspired guidance (adapted)`다. 둘 다 원래 Lan-o3dp policy 결과로 표시하지 않는다.
- 공식 원래 Lan-o3dp policy를 해당 protocol에 맞춰 재실행한 경우에만 원래 method 이름으로 표시하며, 필요한 학습·입력·checkpoint 차이는 별도 공개한다.
- SafeLIBERO와 LIBERO-Safety는 별도 benchmark이며 결과와 level 정의를 합치지 않는다.

---

## 4. 실물 로봇 (RB-Y1)

| 단계 | 내용 | 주장 범위 |
|---|---|---|
| 기본 | 고정 V11AB AG3S로 대표 task 시연 | 실행 가능성; 소수 trial로 성능 개선을 주장하지 않음 |
| 시간 여유 시 | baseline 1개 추가, 같은 task·초기 배치·trial 조건 | 규모에 맞춰 정성/정량 비교 범위 구분 |

선행: simulation의 main/ablation config와 평가기 검증 후 착수한다. 실물 검증 없이 sim 설정을 바로 배포하지 않는다.

---

## 5. 실험 목록

**실행 필수**는 합의한 평가 경로에 필요한 항목, **논문 핵심**은 해당 기여를 주장하려면 필요한 항목이다.
괄호의 DX-ID는 이전 상세 계획과의 대응이며 그 문서의 V7g/V8 설정을 재사용하라는 의미가 아니다.

### 5.1 선행 검증

| # | 실험 | 환경 | 내용 / 통과 기준 | 우선 |
|---|---|---|---|---|
| E0 | 평가기·geometry·adapter 검증 (DX-00) | LIBERO/Panda + SafeLIBERO + RB-Y1 | 알려진 allowed/forbidden fixtures의 기대 판정, sphere coverage, non-target leakage, action mapping 확인. Pi0.5-only/AG3S 각각 최소 1 episode를 official 판정과 기록까지 완주 | 실행 필수 |

Official score 재현은 동일 protocol·checkpoint·normalization·reset state·seed·horizon·controller를 먼저 맞춘 뒤
trial 변동·CI와 구현 차이를 분석한다. 완료한 평가 규모와 별개로 논문의 단일 점수와 정확히 같아야 통과한다고 정하지 않는다.
재현 허용 차이·통계 기준은 결과를 보기 전에 기록한다. Evaluator logic 오류나 환경 불일치는 통과시키지 않는다.

### 5.2 Main 비교

| # | 실험 | 환경 | 비교 조건 | 측정 / 역할 | 우선 |
|---|---|---|---|---|---|
| E2 | **LIBERO adapter pilot / task 보존** | Spatial 소규모 → 선언한 전체 범위 | pi0.5-only vs pi0.5+V11AB | Official success 차이·stop/timeout·runtime. Non-inferiority margin과 CI 기준은 test 전에 고정 | Pilot 실행 필수; 전체 task 보존은 논문 핵심 |
| E1 | **SafeLIBERO main** (DX-07) | Spatial I/II pilot → 전체 I/II | 처음 두 조건 → AEGIS → KNOWS/re-impl. 등 추가 | Official safety/task metrics·safe success·episode 수·CI | 실행 필수 / 주 결과 |
| E13 | **LIBERO-Safety 추가 평가** | 독립 official suite/task/level | pi0.5-only vs pi0.5+V11AB; compatible baseline 추가 | 해당 benchmark official 지표 + 구분된 공통 지표 | 합의한 추가 평가 |
| E3 | **RB-Y1 MuJoCo held-out** (DX-07) | 개발 scene과 분리된 family | V11AB vs pi0.5-only; 검증된 compatible safety baseline 추가 | Task success·forbidden contact·safe success·stop·완료 시간 | 자체 과제 주장을 위한 논문 핵심 |

### 5.3 Ablation — V11AB에서 부품만 변경

상세 matrix는 RB-Y1에서 수행하고, **Panda의 대표 SafeLIBERO task subset에서도 E4/E6/E7 핵심 조건을 확인**한다.
Task subset·episode 수·dev/test 분리 및 한 번에 바꿀 요소를 먼저 고정한다. 전체 benchmark 성능으로 과장하지 않는다.

| # | 요소 | 대안 | 질문 / 적용 조건 | 우선 |
|---|---|---|---|---|
| E4 | **ATS** (DX-01) | 외부 segmenter / 기하 규칙 / GT target(oracle) | Attention-based target geometry의 효과. V11AB에 해당 경로가 있는지 먼저 audit | 논문 핵심; Panda subset 포함 |
| E5 | **HSR/phase gate** (DX-02) | Gate off / strong rule / GT phase(oracle) | 기존 read-out이 rule보다 유리한가. V11AB에 HSR이 없으면 효과를 주장하지 않고 별도 개선 실험으로 분리 | 해당 phase 기여 주장 시 필수 |
| E6 | **LTE** (DX-04) | Exemption off / 모든 protected link에서 target 제외 | Link-specific 처리가 필요한가. 실제 authorized/protected 범위에서 조건 차이가 성립해야 함 | 논문 핵심; Panda subset 포함 |
| E7 | **Attached object** (DX-05) | Off / on | 잡은 물체까지 collision 계산하는 효과 | 논문 핵심; Panda subset 포함 |
| E8 | Target × phase 2×2 (DX-03) | {기하 규칙, ATS} × {strong rule, HSR} | 두 신호의 개별·결합 효과; HSR 실제 지원 시 수행 | 선택 |
| E9 | CCR (DX-06) | Stop-only / per-command correction / K-step CCR | Chunk refinement의 추가 효과; 구현·검사 비용도 비교 | CCR 독립 기여 주장 시 필수 |

Strong rule은 gripper 명령·개도·grasp latch·이력으로 구성하고 dev에서 고정한다.
Panda에서 phase 기능을 검증하지 않았다면 HSR의 cross-robot 효과를 주장하지 않는다.
별도 variant는 명칭·차이·source/config를 기록하고 V11AB main 행에 합치지 않는다.

### 5.4 보조 실험

| # | 실험 | 내용 | 우선 |
|---|---|---|---|
| E10 | Runtime (DX-09) | Stage/end-to-end latency p50/p95/p99, memory, attention pass 비용, intervention/stop, control 및 policy 호출 deadline miss | 실행 필수 / 짧은 표 |
| E11 | Robustness (DX-08) | Target–distractor gap, occlusion/depth noise, instruction 변경, retry; 한 축씩 | 권장 |
| E12 | 실물 RB-Y1 | 4절 | 선택 시연 |

Runtime은 **controller 주기와 policy chunk 호출 주기를 분리**한다. 20 Hz가 실제 설정이면 control tick은 50 ms지만,
policy latency의 deadline은 실행할 action 수·replanning 간격·async 여부에 따라 별도로 정한다.
이전 RB-Y1의 533 ms를 Panda에 일괄 적용하지 않는다. 실제 시간을 측정하고 각 run의 schedule·miss 정의를 manifest에 기록한다.

### 5.5 공통 metric

| Metric | 정의 |
|---|---|
| Task success | 공개 benchmark는 각 task의 **official success predicate/aggregation**. RB-Y1 pick-and-place에만 lift → release → stable placement 정의 적용 |
| Forbidden-contact rate | 선언한 독립 contact 규칙에서 금지 접촉이 1회 이상 발생한 episode 비율; official collision 지표와 구분 |
| Safe success | 같은 episode에서 성공 AND forbidden contact 없음. 두 비율을 곱하지 않음 |
| Stop / timeout / completion time | 정지 증가로 얻은 충돌 감소를 분리; horizon·timeout·재시도 규칙 고정 |
| 기록 누락 / execution error | `UNKNOWN` 및 별도 원인 집계. 0 contact나 성공으로 처리하거나 denominator에서 조용히 제외하지 않음 |
| 통계 | Method별 episode 수·분모·task별 결과·CI·paired 차이 보고. Initial-state/seed pairing과 집계/CI 방법을 test 전에 고정 |

공식 CAR/TSR/ETS·SR/CR/SSR·LIBERO-Safety 지표의 numerator/denominator 및 aggregation을 각각 확인한다.
우리 common metric을 official metric과 같은 이름으로 임의 치환하거나 benchmark 간 비율을 합산하지 않는다.

### 5.6 결과 → Figure/Table 연결

| 산출물 | 채우는 실험 |
|---|---|
| Main Table | E1 + E2, E13 별도 블록; E3는 자체 환경 블록 |
| Ablation Table | E4/E6/E7 + 실제 지원하는 E5; Panda subset 결과의 범위 명시 |
| Figure 1 | E1/E3 실제 rollout의 문제와 개입 |
| Figure 2 | 구현 audit를 반영한 V11AB+adapter pipeline |
| Figure 3 | 같은 scene에서 target instruction 변경 → geometry/constraint/실행 차이; 검증한 phase timeline만 추가 |
| Runtime 표 | E10; robot/benchmark별 schedule 분리 |

---

## 6. Test 전에 확정할 항목

| # | 항목 | 판정 기준 |
|---|---|---|
| Q1 | LIBERO 범위와 task 보존 주장 | Spatial pilot 필수; 전체 suite/task 범위, paired episode 목록, non-inferiority margin/CI를 사전 고정 |
| Q2 | Lan-o3dp 관련 조건 | Perception ablation / adapted guidance / original policy 재실행 중 범위를 선언; 원래 method와 adapted 행 분리 |
| Q3 | Baseline 재현 허용 범위 | Official protocol·checkpoint·metric 일치 여부, 비교 규모·통계 기준과 mismatch 분석 계획 사전 기록 |
| Q4 | V11AB 실제 module·link·geometry·phase | Source/config audit로 고정. V7g/V8 선택 문제는 종료; 없는 기능을 main에 추가하지 않음 |
| Q5 | Panda subset / LIBERO-Safety protocol | Task/level/seed 목록, official 지표, adapter 차이와 제외 범위를 고정 |

### 6.1 환경·저장·재현 기록

- 실행 코드와 venv는 자기 benchmark checkout에 둔다. 이전 `/mnt/dev/work` source/venv를 runtime에 사용하지 않는다.
- 공유 checkpoint/dataset/assets는 `/mnt/dev/AG3S_VLA-storage`에서 참조한다.
- 새 video·trajectory·contact event·log·metrics·manifest는 `/mnt/dev/AG3S_VLA-storage/results/benchmark/<KST날짜_commit_run-id>/`에 저장한다. 이 규칙은 runner에서 실제 구현·검증해야 한다.
- OpenPI `--record`의 상대 `policy_records`가 기존 공유 기록을 덮어쓰지 않도록 고유한 run 경로로 연결한다. Recording step 번호는 각 run 내부에만 사용한다.
- Manifest 필수 항목: 원본 v2 SHA, monorepo/adapter SHA 및 dirty 여부, dependency pin, checkpoint identity/checksum, norm stats, method/variant, protected/authorized links, privileged input, camera/robot config, controller/phase/action 설정, suite/task/level, initial state/seed, horizon, metric 정의, runtime schedule, output 경로.
- Run id를 유일하게 생성하며 development 결과 경로와 섞지 않는다. 다른 세션의 파일·프로세스는 수정/종료하지 않는다.

---

## 7. Claude Code 실행 순서와 통과 기준

판정은 **milestone 기반**이다. 2026-10-12 기준 이미 지난 10/10 중단 기준은 폐기한다.
10/21 주장 고정·10/28 전체 결과 목표는 내부 계획이며, 각 gate의 실제 진행과 episode당 비용으로 재산정한다.
공식 제출 마감과 내부 목표를 구분한다. 각 단계는 구현 → 독립 검증 → handoff 기록 후 다음 단계로 진행한다.

| Phase | 범위 | 다음 단계 통과 기준 |
|---|---|---|
| P0 | Source/config·환경·checkpoint·저장 경로 audit | V11AB 기준과 module 대응, own-root import/GPU, 고유 run logging 확인 |
| P1 | **Panda adapter**: SL-0~SL-6 + E0 fixtures | Camera/depth/action/geometry/phase mapping 정상; pi0.5-only 및 V11AB 최소 1 episode 공식 판정·기록까지 완주 |
| P2 | **LIBERO Spatial 소규모 평가**: E2 + E10 기본 측정 | 고정 task/seed paired 비교, 평가기·logging 정상, action 의미/성능 저하 및 stop/timeout 원인 점검 |
| P3 | **SafeLIBERO Spatial I/II pilot**: E1 + SL-7/8 | Contact 판정 검증, 두 주 조건의 paired 결과·CI·runtime 확인; compatible AEGIS re-run 착수 |
| P4 | **전체 SafeLIBERO**: E1 + E10 | Pilot 후 config/test 목록 고정; official I/II 1,600 episodes/조건 및 오류·UNKNOWN 집계 |
| P5 | **LIBERO-Safety 추가 평가**: E13 | 독립 protocol과 adapter 검증 후 선언한 범위 완료; 공식 지표와 variant 차이 기록 |
| P6 | RB-Y1 상세 E4~E7 및 Panda subset 검증, E3, 추가 baseline | 고정 V11AB에서 조건별 변경만 적용; 핵심 mechanism 주장에 맞는 증거 확보 |
| P7 | E11 robustness / E12 실물 시연 | 앞선 검증과 자원·시간 여유가 있을 때 수행 |

P6의 RB-Y1 ablation은 주 평가 경로와 독립적인 자원이 확보되면 병행할 수 있다. Panda subset은 P3 이후 검증된 adapter에서 실행한다.
추가 baseline 구현 때문에 최초 pi0.5-only vs V11AB 평가를 지연시키지 않는다.

**재판정 기준:** own-root 환경·checkpoint가 준비되지 않으면 환경 단계에서 원인을 해결한다.
Adapter/contact evaluator 오류가 있으면 전체 rollout을 확장하지 않는다. 공식 점수와 차이가 나면 protocol과
통계적 변동을 분석하며 날짜 경과나 단일 점수 차이만으로 benchmark를 포기하지 않는다.
시간·자원이 부족해 범위를 줄일 경우에는 축소 task/level/episode와 그 근거를 명시하고 전체 benchmark 완료로 표시하지 않는다.
RB-Y1 결과는 별도 자체 환경 결과로 유지하며 public benchmark 성능을 대신 증명한다고 주장하지 않는다.

### 7.1 이번 개정 내역

- V11AB paired-v2 source·config 고정 및 method/adapter/ablation variant 구분.
- 합의한 다섯 단계 실행 경로 적용, LIBERO-Safety 평가(E13) 추가.
- SafeLIBERO 16 tasks의 합계/산식 수정 및 단계별 평가 비용 명시.
- Panda 핵심 ablation subset, phase·CCR·protected-link 적용 범위 audit 추가.
- Official success와 자체 metric 분리, paired 비교·CI·UNKNOWN·non-inferiority 규칙 명시.
- Baseline 우선순위·partial 재구현·Lan-o3dp adapted 표기 및 재현 허용 기준 정리.
- Robot별 control/chunk runtime deadline 분리, 과거 날짜 중단 기준을 milestone gate로 전환.
- 자기 OpenPI 환경과 storage의 고유 run 경로·manifest·기존 기록 보호 규칙 추가.
