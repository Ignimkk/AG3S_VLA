# AG3S-VLA CVPR 2027 최종 실험 계획

> 작성 기준: 2026-11-04 KST. 목적: 논문 **4. Experiments**의 실험과 비교표를 미리 준비한다. AG3S 최종 방법은 병행 개발 중이며, AG3S 수집 결과는 본 문서에 넣지 않는다.

## 1. 연구 목적과 진행 원칙

AG3S-VLA는 VLA 내부 표현과 다중시점 RGB-D로부터 작업 관련 3D 객체 표면과 identity를 추정하고, 이를 작업 조건부 충돌 제약으로 변환하는 방법을 목표로 한다. 후단 trajectory optimization(TO)은 제약의 유효성을 검증하는 고정 구성 요소이며 신규 최적화 기법으로 주장하지 않는다. 최종 구현의 구성 요소와 입력은 구현이 확정되면 기록한다.

**진행 방식:** (1) AG3S 구현과 독립적으로 문헌 수집·평가 환경·baseline을 준비하고 기존 방법의 원문 수치 또는 직접 측정치를 등록한다. (2) AG3S 최종 구현 확정 후 같은 평가 계획을 적용하여 표의 빈칸을 채운다. (3) 결과를 확인한 뒤에만 기여 및 SOTA 범위를 확정한다. AG3S 수치·성능 추정·이전 개발 결과를 결과표에 넣지 않는다.

### Research questions

| RQ | 질문 | 주요 실험 |
|---|---|---|
| RQ1 | VLA 내부 표현과 RGB-D로 지시에 맞는 관측 3D target 표면을 식별할 수 있는가? | E2 segmentation, 지시 변경 |
| RQ2 | 유사 객체와 가려짐에서 target identity를 유지할 수 있는가? | E3 memory·occlusion |
| RQ3 | 객체·링크·단계에 맞는 허용/금지 접촉 제약을 생성할 수 있는가? | E4 제약 진단, E5 ablation |
| RQ4 | 고정된 후단 TO에서 안전한 작업 수행에 기여하는가? | E1 benchmark, E4 downstream, E6–E7 |

C1은 내부 표현에서 제약 생성까지의 framework, C2는 3D grounding과 identity, C3는 안전 조작에 대한 실험적 유효성이다. **SOTA·일반화·최초 여부는 검증 후에만 주장한다.** RQ의 구체적인 구성 요소는 최종 구현에 맞춰 동결한다.

### 결과 표기 규칙

- **P**: 논문 원문 보고값. 논문 버전·표·조건을 옆에 명시한다. 원문 값은 현재 환경에서 재현된 값이 아니다.
- **R**: 우리 평가기에서 직접 재현한 결과. 실제 실행·검증 후 manifest와 원본을 연결한다.
- **TBD**: 아직 측정·재현하지 않은 값. **—**: 원문에 해당 지표가 없거나 동일 정의로 추출할 수 없는 값. **AG3S `빈칸`**: 최종 구현 후 평가 예정.
- P와 R을 별도 표로 관리한다. 논문에 함께 배치할 필요가 있으면 출처와 조건 차이를 명확히 표시하며, 동일 protocol 확인 전 순위·차이·SOTA 판정에 사용하지 않는다. 지표가 비슷해도 분모·안전 판정·집계가 다르면 합산하지 않는다.

## 2. 실험 구조와 수행 순서

| ID | 실험 | 주요 산출물 | AG3S 완성 전 준비 |
|---|---|---|---|
| E1 | SafeLIBERO / LIBERO-Safety | 공식 protocol 결과, 강한 기존 방법과 비교 | 문헌 표, 환경·평가기·baseline rollout |
| E2 | Task-conditioned 3D segmentation | 관측 표면의 품질과 target 선택 | 공통 장면·GT·외부/geometry baseline |
| E3 | Identity 및 occlusion | identity 유지·재식별·기권 | 연속 시퀀스·가림 GT·tracking baseline |
| E4 | Constraint validity 및 downstream | 금지 충돌 누락·필요 접촉 차단·TO 효과 | 접촉 정책 GT, configuration/trajectory 평가기, 공통 TO |
| E5 | Component ablation | 표현·memory·contact logic의 기여 | 구현 가능한 대조군 인터페이스·factorial 설계 |
| E6 | Held-out generalization·efficiency | 미관측 조건과 계산 비용 | split, timing instrumentation·baseline timing |
| E7 | Real robot | 실환경 안전 성공·실패 | 로봇·센서·접촉 판정·baseline |

**우선순위:** P0 = 공식 환경·평가기 및 π₀.₅ baseline, E2의 3D GT, E4 제약/접촉 평가기. P1 = 외부 segmenter 및 generic geometry, 주요 공개 안전 방법 수집·재현, E3 시퀀스. P2 = 추가 OOD 및 실로봇. E1과 E4의 downstream은 가능한 한 같은 rollout을 재사용한다. 문헌값은 지금 채우고 자체 baseline은 실행 후 채운다. 이후 AG3S 확정 → 설정/평가 split 잠금 → 평가 → AG3S 칸 입력 순으로 진행한다.

## 3. 비교와 판정의 공통 규약

### 3.1 공식 benchmark 비교와 통제 비교

1. **공개 연구 결과:** 각 논문의 checkpoint, 학습 데이터, 센서, simulator·action space, 안전 평가 및 집계 방식을 기재한다. 문헌별 다른 조건은 별도 표로 둔다.
2. **동일 조건 제약 비교:** π₀.₅ checkpoint, RGB-D 입력·카메라, robot·controller, action chunk, 초기 상태·seed, TO solver·목적함수·예산, timeout, 독립 evaluator를 고정한다. geometry 또는 constraint generator만 바꾼다. VLA 단독은 TO가 없는 별도 기준선이다.
3. **Safety method의 시스템 비교:** AEGIS, SafeDojo 등은 고유한 control·훈련 방식을 유지해야 하므로 TO까지 같다고 가정하지 않는다. 동일 평가 환경에서 실행하더라도 backbone·센서·학습량·추가 GT 정보와 입력 차이를 표기한다.

문헌값을 자체 R 표에 미리 삽입하지 않는다. 재현이 어려운 경우 원문값과 미재현 사유를 남긴다. 새로운 구현이 원 논문과 다르면 `our reimplementation`으로 표시한다.

### 3.2 안전 판정과 통계

공식 평가는 해당 benchmark의 원 evaluator를 우선한다. 추가 분석은 사전 고정된 독립 simulator contact/geometry 또는 실로봇 센서·영상으로 평가한다. target과의 모든 접촉을 일괄 허용하지 않고 robot link, object, 작업 단계, grasp 및 부착 상태별로 허용 접촉을 정의한다. 예측 제약 자체는 안전 정답이 아니다.

Task success, 안전 위반, safe success의 **개별 분자·분모**와 task/suite/level별 수치를 보존한다. paired 초기 상태·seed를 사용하고 episode 단위 paired bootstrap 등으로 95% 구간을 보고한다. 같은 episode의 frame은 독립 표본이 아니다. 학습 seed 간 변동은 evaluation episode의 변동과 구분한다. 표본 수는 사전에 목표 효과·실행 비용·예상 정밀도를 기준으로 확정하고, 결과 확인 후 필요한 확대는 추가 수집임을 기록한다. 성능 유지·비열등성은 사전 허용 한계와 불확실성 평가 없이는 주장하지 않는다.

## 4. E1 — Main benchmark comparison

### 4.1 SafeLIBERO 원문 보고값: Level I / II 분리

[P: SafeDojo v2 Table 1–2](https://arxiv.org/html/2606.20698v2), Spatial·Goal·Object·Long 평균. Level II는 Level I demonstrations만으로 학습한 평가다. 이 표는 SafeDojo 논문의 조건이며 AEGIS 원 논문 또는 다른 논문의 수치를 섞지 않는다.

| Level | Method | TSR ↑ (%) | SSR ↑ (%) | ETS ↓ (steps/episode) | 출처 |
|---|---|---:|---:|---:|---|
| I | OpenVLA-OFT | 52.12 | 39.00 | 287.94 | P, Table 1 |
| I | AEGIS | 32.38 | 28.50 | 310.67 | P, Table 1 |
| I | SafeVLA | 54.25 | 45.00 | 294.69 | P, Table 1 |
| I | WMPO | 60.62 | 44.88 | 274.02 | P, Table 1 |
| I | SafeDojo | 64.50 | 53.25 | 269.41 | P, Table 1 |
| II | OpenVLA-OFT | 52.12 | 43.38 | 286.72 | P, Table 2 |
| II | AEGIS | 33.13 | 30.50 | 302.44 | P, Table 2 |
| II | SafeVLA | 56.50 | 49.50 | 289.87 | P, Table 2 |
| II | WMPO | 55.75 | 46.12 | 277.17 | P, Table 2 |
| II | SafeDojo | 57.00 | 49.62 | 277.00 | P, Table 2 |

TSR = 작업 완료 / 전체 episode; SSR = 작업 완료 및 해당 evaluator의 무충돌 조건 충족 / 전체; CAR = 무충돌 episode / 전체; CSC = 전체 episode당 접촉 simulation step 평균; ETS = **전체 episode당** 실행 step 평균. SafeDojo에서 CAR·CSC는 본문에 정의되지만 위 표에는 제공되지 않으므로 별도 검증 없이 숫자를 채우지 않는다. TSR과 CAR에서 SSR을 추정하지 않는다. 기존 32 scenarios × 50 episodes = 1,600회/방법은 계획 가정으로, 공식 코드의 task·level·초기 상태·평가 횟수를 실행 전 확인한다.

### 4.2 SafeLIBERO 자체 평가 표 — 각 Level별 별도 작성

동일 evaluator·집계의 R만 기입한다. 비교 가능한 센서와 원 방법 구현 여부는 별도 열/각주에 기록한다. CAR·CSC·ETS, suite별 수치와 성공/전체 횟수는 같은 결과 파일에서 함께 제공한다.

| Method | Level | TSR ↑ | SSR ↑ | CAR ↑ | ETS ↓ | 상태 |
|---|---|---:|---:|---:|---:|---|
| π₀.₅ | I / II | TBD | TBD | TBD | TBD | R 수집 예정 |
| π₀.₅ + generic geometry + fixed TO | I / II | TBD | TBD | TBD | TBD | R 구현·수집 예정 |
| π₀.₅ + external segmentation + fixed TO | I / II | TBD | TBD | TBD | TBD | R 통합·수집 예정 |
| AEGIS | I / II | TBD | TBD | TBD | TBD | 원 구현 확인 후 R |
| SafeDojo | I / II | TBD | TBD | TBD | TBD | checkpoint·재현 가능성 확인 후 R |
| **AG3S-VLA** | **I / II** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **최종 구현 후 평가** |

표의 `I / II`는 계획 표시이며 최종 산출물은 level별 별도 행 또는 표로 기록한다. 각 방법의 본래 알고리즘을 유지한 재현과 같은 TO를 사용하는 controlled baseline은 구분한다.

### 4.3 LIBERO-Safety 원문과 자체 평가

[P: LIBERO-Safety Table 3](https://arxiv.org/html/2606.23686), physical track L2, SR(%) 평균 ± 세 학습 seed의 표준편차. SR은 안전 제약 위반 없이 작업을 완료한 비율이다. 아래는 관련 suite의 **원문 baseline**이며 전체 벤치마크의 최신 통합 순위를 뜻하지 않는다.

| Method | AAG L2 SR ↑ | TSA L2 SR ↑ | FSHOA L2 SR ↑ | 출처 |
|---|---:|---:|---:|---|
| OpenVLA | 4.0 ± 0.0 | 12.7 ± 0.9 | 0.0 ± 0.0 | P, Table 3 |
| OpenVLA-OFT | 1.3 ± 0.9 | 40.0 ± 1.6 | 42.7 ± 2.5 | P, Table 3 |
| π₀ | 11.3 ± 2.5 | 0.7 ± 0.9 | 0.7 ± 0.9 | P, Table 3 |
| π₀.₅ | 35.3 ± 2.5 | 56.7 ± 3.8 | 51.3 ± 6.1 | P, Table 3 |
| GR00T N1.6 | 19.3 ± 3.4 | 52.7 ± 2.5 | 50.0 ± 1.6 | P, Table 3 |

AAG = Affordance-Aware Grasping, TSA = Tabletop Spatial Avoidance, FSHOA = Free-Space Hand-Object Avoidance. 공식 physical track 전체를 주장하려면 HRI 및 L0–L2도 평가한다. Semantic Safety Reasoning은 별도 track으로 범위 밖이다. 공식 보고의 보조 지표는 CR, 실행 시간, LDLJ다. 물리 4 suite × 5 task × 3 level × 10 trials × 3 training seeds = 1,800은 계획 계산치이며 checkpoint·학습 seed 및 실행 단위를 확인한다. 3개 평가 seed와 3개 독립 학습 seed를 혼동하지 않는다.

| Method | AAG L2 SR ↑ | TSA L2 SR ↑ | FSHOA L2 SR ↑ | CR ↓ | 상태 |
|---|---:|---:|---:|---:|---|
| π₀.₅, benchmark-compatible checkpoint | TBD | TBD | TBD | TBD | R: checkpoint·controller 확인 후 |
| Generic geometry + fixed TO | TBD | TBD | TBD | TBD | R 수집 예정 |
| External segmentation + fixed TO | TBD | TBD | TBD | TBD | R 수집 예정 |
| **AG3S-VLA** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **최종 구현 후 평가** |

최종 결과는 각 L0–L2 및 HRI를 별도 표로 기록한다. **실행 전 필수:** 공식 checkpoint/config·데이터, action 변환과 controller 설정의 동등성을 검증한다. 기존 내부 protocol에 기록된 controller 모델 차이를 해결하거나 실험 범위를 제한하고, 정책 action을 다른 task에 옮긴 stress test를 공식 rollout으로 취급하지 않는다. 참고: [action transfer 규약](../../doc/active/experiments/BASELINE_LIBERO_SAFETY_ACTION_TRANSFER_PROTOCOL_20261102_KO.md), [공식 코드](https://github.com/LIBERO-SAFETY/LIBERO-Safety).

## 5. E2 — Task-conditioned 3D segmentation

**질문:** 동일 관측 support에서 지시에 맞는 target의 관측 표면을 식별하는가? Single target, 유사 객체, 인접 clutter, 부분 가려짐 장면을 구성하고 task·물체·배치 단위로 development/test를 분리한다. 같은 장면·로봇 상태에서 instruction만 바꾸어 target과 최종 접촉 제약이 함께 바뀌는지 검증한다. 모호한 지시와 비가시 target에서는 기권을 별도 평가한다.

GT는 모델 입력 없이 image-time instance mask·depth·camera calibration을 통해 생성하고, 공통 point/voxel support와 visibility를 고정한다. 관측되지 않은 뒷면은 점수에 포함하지 않는다. 빈 예측과 미관측 GT의 처리 및 frame → episode → task 집계 순서를 잠근다. 대상 단서와 self-mask의 오류를 진단하며, 과거 BDDL 정답 이름을 이용한 prompt는 instruction-only 결과와 분리한다.

| Method | 3D IoU ↑ | Target ID ↑ | Precision / Recall ↑ | Abstain rate ↓ | Latency | 상태 |
|---|---:|---:|---:|---:|---:|---|
| Geometry-only + 명시된 target 선택 규칙 | TBD | TBD | TBD | TBD | TBD | R 구현·수집 예정 |
| Grounded-SAM / SAM2 등 외부 segmentation + 동일 3D 변환 | TBD | TBD | TBD | TBD | TBD | R: instruction-only prompt |
| Oracle target/geometry | TBD | TBD | TBD | — | — | R 진단용, GT 입력 표시 |
| **AG3S-VLA** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **최종 구현 후 평가** |

외부 모델은 상한이 아니라 별도의 baseline이다. 모델별 checkpoint·prompt·추가 학습과 입력 차이, 2D→3D 변환·multi-view fusion 및 inference hardware를 기록한다. Boundary F1은 정의와 GT 품질을 검증한 경우 보조 지표로 보고한다. 동일 장면의 서로 다른 지시를 포함한 target-ID/제약 변화 결과도 별도 표로 저장한다. [기존 baseline catalog](../../Baseline/paper_baselines/README.md), [실험 자료](../../doc/active/experiments/AG3S_SEGMENTATION_FACTORIAL_RESULTS_20261101_KO.md)는 재사용 가능성을 점검하되 서로 다른 입력·평가 범위를 혼합하지 않는다.

## 6. E3 — Identity 및 occlusion

**질문:** 연속 관측에서 target identity를 유지하고 가려진 후 다시 식별하는가? 유사 객체 및 가림 없는 장면부터 부분 가림, 완전 가림 후 재등장까지 독립 시퀀스를 구성한다. 가림 비율은 GT 기준으로 정의하고 target이 화면 밖인 경우를 별도로 기록한다. 완전 가림 동안 identity가 관측만으로 결정되지 않는 상황에서는 기권 및 잘못된 재식별을 구분한다. 보이지 않는 표면을 알고 있다고 가정하지 않는다.

| Method | Target ID accuracy ↑ | ID switches ↓ | Re-ID ↑ | 기권·오인식률 | 상태 |
|---|---:|---:|---:|---:|---|
| Frame-by-frame external segmentation | TBD | TBD | TBD | TBD | R 수집 예정 |
| External segmentation + tracker | TBD | TBD | TBD | TBD | R 수집 예정 |
| Geometry-based tracking | TBD | TBD | TBD | TBD | R 수집 예정 |
| **AG3S w/o memory** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **구현 확정 후 ablation** |
| **AG3S full** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **최종 구현 후 평가** |

ID switch, 재등장 window, Re-ID 분모와 occlusion 구간을 고정한다. baseline별 입력·초기 prompt·tracking 상태를 기록한다. memory가 최종 방법에 포함되지 않으면 이 RQ 및 ablation의 범위를 수정한다.

## 7. E4 — Collision constraint validity 및 downstream

**질문:** 필요한 접촉은 허용하고 비허용 충돌을 포착하며, 고정 TO에서 안전한 작업 실행에 도움이 되는가?

### 7.1 독립 제약 평가기

- **접촉 정책:** 사전에 정의한 `(phase, robot link, object, contact type)`별 허용/금지 정답과 예측 정책을 비교한다. grasp 및 attached object 상태·support contact를 명시한다.
- **기하:** 미리 잠근 robot configuration 및 후보 trajectory의 링크–객체 pair에서 독립 GT collision/contact와 제약 판정을 비교한다. threshold와 horizon을 고정한다. 불가능한 state 또는 예측 범위 밖 지점은 표시한다.
- **오류 분해:** 금지 충돌 FN, 안전 상태 FP, 필요한 target 접촉 차단, non-target의 잘못된 허용을 각각 센다. 로봇 링크–장애물과 운반 물체–장애물, 접근·파지·운반·배치를 나눈다. 난이도 균형 표본의 precision은 자연 발생 비율로 일반화하지 않는다.
- **미관측:** 관측 지원 밖 공간을 별도 표시하고, unknown 처리와 관측 지연을 기록한다. 표면/robot geometry 미포함, 제약 생성 실패, TO infeasibility 또는 실행 불일치를 분리한다.

| 제약 생성 | 금지 접촉 recall ↑ | 금지 접촉 precision ↑ | 필요 접촉 차단률 ↓ | Constraint latency | 상태 |
|---|---:|---:|---:|---:|---|
| Generic geometry + uniform policy | TBD | TBD | TBD | TBD | R 구현·수집 예정 |
| External segmentation + 동일 contact policy | TBD | TBD | TBD | TBD | R 통합·수집 예정 |
| Oracle geometry + oracle policy | TBD | TBD | TBD | TBD | R 진단용, 특권 정보 |
| **AG3S constraint generation** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **최종 구현 후 평가** |

Oracle은 진단 참고치이며 TO 최적화나 보수적 근사 때문에 모든 downstream 지표의 수학적 상한으로 취급하지 않는다. 후보 state 샘플링과 양성·음성 분모, 지정된 제약의 역할을 결과마다 함께 기록한다.

### 7.2 고정 TO downstream

동일 π₀.₅와 TO에서 generic geometry / external segmentation / AG3S 제약을 바꾼다. 이때 동일한 관측, 초기 상태, objective, solver tolerance, iteration·time budget, fallback을 사용한다. E1 rollout을 재사용하고 필요한 인접 물체·hurdle·운반 장애물 사례만 별도 추가한다.

| 방법 | Task SR ↑ | Safe SR ↑ | 금지 접촉률 ↓ | TO feasible rate ↑ | TO p95 latency | 상태 |
|---|---:|---:|---:|---:|---:|---|
| π₀.₅, no TO | TBD | TBD | TBD | — | — | R 수집 예정 |
| Generic geometry + fixed TO | TBD | TBD | TBD | TBD | TBD | R 수집 예정 |
| External segmentation + fixed TO | TBD | TBD | TBD | TBD | TBD | R 수집 예정 |
| **AG3S + fixed TO** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **최종 구현 후 평가** |

공식 benchmark 지표와 이 표의 추가 contact evaluator 정의가 다르면 결과를 따로 보고한다. 제약 평가와 실제 실행의 연관성을 분석하되, 상관관계만으로 인과 효과를 단정하지 않는다.

## 8. E5 — 구성 요소 및 contact policy ablation

최종 구현이 확정된 후 각 요소를 실제로 제거하거나 동등한 대조로 교체할 수 있는지 확인한다. 제안된 attention, hidden state, action conditioning, geometry, single/multi-view, memory, contact policy 중 **실제로 사용하는 요소만** 평가한다. 단독 요소가 출력 자체를 만들 수 없다면 임의 decoder를 사용하지 않고 재현 가능한 대조를 설계한다. 핵심 rollout은 E1/E4와 공유한다.

| 비교 | 주요 측정 | 상태 |
|---|---|---|
| Geometry vs 외부 segmentation | 3D IoU, target ID, 제약 오류 | R 준비 |
| Target 항상 장애물 vs target 항상 허용 vs phase/link-aware policy | 필요한 접촉 차단, 금지 접촉 FN, safe SR | R 정책·평가기 준비 |
| AG3S full vs attention / geometry / memory 등 제거 | 해당 요소의 grounding·identity·제약·safe SR | AG3S 확정 후; 값 빈칸 |
| AG3S constraint 없음 vs 있음 | paired safe SR·금지 접촉 | AG3S 확정 후; 값 빈칸 |

**2×2 조건부 설계:** generic/AG3S 표면 표현 × generic/task-aware 제약 로직. 네 군 모두 사용할 수 있는 동일 target identity 입력과 정보 출처를 먼저 정의한다. GT identity를 사용하면 oracle 진단으로 별도 표시한다. G2가 구현 불가능하면 억지로 채우지 않고 설계를 수정한다. 독립 효과 및 interaction은 각 차이와 차이의 차이를 paired 데이터로 산출한다.

## 9. E6 — 일반화 및 효율성

**Held-out 평가:** 학습·개발·임계값 조정에서 제외한 유사 객체, 배치·clutter, occlusion을 우선한다. 외형·형상, camera calibration, depth noise는 여력이 있을 때 개별 축으로 추가한다. 공식 benchmark를 수정한 경우 Extended OOD로 별도 명시한다. 동적 장애물은 최종 방법이 시간 변화와 예측을 지원하고 독립 evaluator를 구축한 경우에만 포함한다. 실험별 held-out 근거와 split을 저장한다.

**효율성:** policy inference, 특징 추출, 3D fusion, constraint 생성, TO 및 capture→response→execution end-to-end를 동일 경계에서 p50/p95로 측정한다. 하드웨어·동시 작업·warmup·요청 수, fallback 및 timeout을 기록한다. 15 Hz에서 8 step을 실행하는 설정이라면 실행 창은 약 533 ms다. 이는 해당 설정의 deadline이며 모든 시스템에 대한 보편 기준이 아니다. 동기식 deadline miss와 비동기식 observation age·action age·update frequency를 구분한다.

| Method | Held-out safe SR ↑ | Constraint p50/p95 | End-to-end p50/p95 | Deadline miss ↓ | 상태 |
|---|---:|---:|---:|---:|---|
| Generic geometry + fixed TO | TBD | TBD | TBD | TBD | R 수집 예정 |
| External segmentation + fixed TO | TBD | TBD | TBD | TBD | R 수집 예정 |
| **AG3S-VLA** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **최종 구현 후 평가** |

## 10. E7 — Real-world manipulation

실로봇 조건에서 clear pick-and-place, 유사 인접 물체, 운반 경로 장애물, held-out 물체/배치를 우선한다. 로봇·카메라 calibration, 속도 및 emergency stop, 실행 예산, 관측 가능한 접촉 범위를 문서화한다. AG3S 추정과 독립적인 접촉 센서·토크·외부 영상 또는 관찰을 사용하고 검출 한계를 기록한다. 작업별 초기 배치를 짝지어 순서를 무작위화한다.

| Method | Task SR ↑ | Safe SR ↑ | Unsafe contact ↓ | 성공/전체 | 상태 |
|---|---:|---:|---:|---:|---|
| π₀.₅ | TBD | TBD | TBD | TBD | R 수집 예정 |
| Generic geometry + fixed TO | TBD | TBD | TBD | TBD | R 수집 예정 |
| External segmentation + fixed TO | TBD | TBD | TBD | TBD | 자원 허용 시 R |
| **AG3S-VLA** | **빈칸** | **빈칸** | **빈칸** | **빈칸** | **최종 구현 후 평가** |

시나리오별 방법당 20회는 **초기 계획값**이지 충분성이 입증된 표본 수가 아니다. pilot 후 목표 효과와 신뢰구간·자원을 고려해 사전에 최종 횟수를 정한다. known 장면과 held-out 장면을 분리하며, 새로운 조건에서의 결과 없이 일반화를 주장하지 않는다.

## 11. Baseline 준비 및 출처 관리

| 대상 | 현재 수집 경로 | 다음 작업 |
|---|---|---|
| π₀.₅ | SafeLIBERO 및 LIBERO-Safety 원문값 P 확보 | benchmark별 checkpoint·controller·공식 평가 확인 후 R |
| AEGIS | SafeDojo 및 다른 논문에 각기 다른 P 존재; [원 프로젝트](https://vlsa-aegis.github.io/) | 공식 코드·입력·action space 확인; 자체 evaluator R |
| SafeVLA / SafeDojo | SafeDojo Table 1–2 P 확보 | checkpoint·코드·학습 데이터·수행 비용 확인; 가능하면 R |
| Generic geometry / external segmenter | 공통 평가로 직접 산출 필요 | 동일 입력·3D fusion·contact policy·TO 연결 및 R |
| Oracle geometry/contact policy | simulator GT | 평가기 검증 및 특권 진단만 수행 |

[기존 조사](../../Baseline/paper_baselines/docs/survey.md)는 관련 방법의 코드·checkpoint·license를 점검한 기록이나 2026-10-23 snapshot이다. 실행 전 현재 공개 상태를 재확인한다. 문헌 원본에서 직접 추출한 수치는 논문 버전·표·suite·level·metric·집계·입력과 함께 보존한다. 코드나 checkpoint를 사용할 수 없으면 구현 사실을 만들어내지 않고 `미재현`으로 남긴다.

## 12. 논문 구성 및 산출물

| Section | 본문에 넣을 핵심 결과 | 추가 자료 |
|---|---|---|
| 4.1 Experimental Setup | benchmark·baseline·입력·접촉 정책 및 평가 | 전체 설정과 split |
| 4.2 Main Results | E1 benchmark 및 동일 조건 비교 | suite/level/task별 전체 결과 |
| 4.3 Task-Conditioned 3D Grounding | E2와 필요 시 E3 identity·occlusion | 3D GT, 장면·prompt·추가 시각화 |
| 4.4 Constraint Quality and Ablations | E4 독립 제약 평가와 E5 핵심 ablation | contact pair별 오류와 TO 분석 |
| 4.5 Generalization and Real Robot | E6 held-out·runtime 및 E7 실로봇 | 상세 latency·실패·비디오 |

결과 파일마다 코드 revision, 환경·evaluator 버전, checkpoint/hash, 공개 원문 version/table 또는 자체 manifest, 입력·특권 정보, 초기 상태·seed, per-episode 원본, metric 정의와 집계 script, confidence interval을 보관한다. 보고값과 재현값을 혼동하지 않는다. 표와 수치는 [RESULTS](RESULTS.md), 주장과 범위는 [CLAIMS](CLAIMS.md)에 등록한다.

### 완료 체크리스트

- [ ] 공개 문헌 표의 값·버전·프로토콜 확인 및 최신 연구 재검색
- [ ] benchmark controller/action·checkpoint·evaluator 및 independent contact 판정 검증
- [ ] 자체 baseline의 각 R 결과, source 및 불확실성 등록
- [ ] E2/E3/E4의 GT, split, 샘플링·기권 및 측정 단위 동결
- [ ] AG3S 최종 구성·입력·fallback·실제 활성 경로 동결 후 평가
- [ ] 각 표의 AG3S 빈칸을 실제 결과로 채우고 표본·실패·latency 함께 등록
- [ ] 최신 강력 baseline과 동등 조건이 확인된 범위에서만 SOTA 판단; 미확인 시 구체적인 비교 결과만 기술

## 참고 자료

- [SafeDojo v2, SafeLIBERO Table 1–2](https://arxiv.org/html/2606.20698v2)
- [LIBERO-Safety 논문](https://arxiv.org/html/2606.23686), [공식 프로젝트](https://libero-safety.github.io/), [공식 코드](https://github.com/LIBERO-SAFETY/LIBERO-Safety)
- [별도 SOTA 조사](SOTA_COMPARISON_KO.md), [baseline 조사](../../Baseline/paper_baselines/docs/survey.md)
