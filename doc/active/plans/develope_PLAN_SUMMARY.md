# develope_PLAN 요약 — Benchmark · Baseline · 실물 로봇 · 실험 목록

> **2026-10-23 우선 적용:** [한 페이지 연구 범위](AG3S_RESEARCH_SCOPE_20261023_KO.md)와 [핵심 평가 task](../../../AG3S/ag3s/docs/handoff/AG3S-CORE-EVALUATION.task.md)를 먼저 읽는다. 아래는 이전 계획 요약이며, 이번 회차의 후보·비교·종료 기준은 새 문서에 고정했다.

작성: 2026-10-07 KST · 상태: `PLANNED` (아직 아무것도 실행하지 않음)
원본: [develope_PLAN.md](develope_PLAN.md) (638줄) · [SAFELIBERO_PLAN.md](paper/notes/SAFELIBERO_PLAN.md) · baseline 원문 정리 [REFERENCE_ANALYSIS_R2.md](paper/notes/REFERENCE_ANALYSIS_R2.md)
용도: 이후 Claude Code에 "1·2번을 구현·검증해 달라"고 넘길 때의 입력 문서.

---

## 0. 한눈에

- **논문 한 줄:** 외부 segmentation model 없이 VLA attention + depth로 target geometry를 얻고, 이를 link-specific contact constraint에 연결해 frozen VLA의 조작 실행을 개선한다. Phase 정보는 contact exemption이 유효한 시점을 조절한다.
- **중심 기여:** ATS → target geometry → LTE (link-specific constraints). HSR·CCR은 우선 "구성 요소"로 두고, 추가 이득이 확인될 때만 독립 기여로 올린다.
- **본문 목표 산출물:** Figure 3개 + Table 2개(Main · Ablation) + 짧은 runtime/설정 표. 원본의 10개 실험·8 Figure·8 Table을 다 할 필요는 없다.
- **마감 (CONFIRMED_2027):** 등록 11/10 · 본문 11/16 · 보충자료 11/23 (AoE). 주장 고정 10/21. 웹페이지는 제출 후.

| 약어 | 뜻 |
|---|---|
| ATS | Attention-based Target Segmentation — attention + depth로 target 선택·추출 |
| HSR | Hidden-state Sub-task Recognition — frozen VLA feature 위의 linear probe로 phase를 읽음 |
| LTE | Link-specific Target Exemption — 허용된 link에만 target 접촉을 허용 |
| CCR | Constrained Chunk Refinement — 실행할 action chunk 전체를 cuRobo로 정제 |
| Attached object | 잡은 물체를 collision query에 포함 |

## 1. 원래 계획과 이번 결정의 차이 (확인 필요)

| 항목 | develope_PLAN 0.9절 | 이번 결정 | 영향 |
|---|---|---|---|
| Benchmark | SafeLIBERO + MuJoCo | **LIBERO 추가** | LIBERO의 역할을 정해야 함 (아래 6절 Q1) |
| Baseline | VLSA · KNOWS · WBAG | **KNOWS · Lan-o3dp · VLSA/AEGIS** (WBAG 제외) | Lan-o3dp는 VLA가 아니므로 이식 방법을 정해야 함 (Q2) |
| 코드 공개 | "미확인" | KNOWS 미공개, Lan-o3dp · VLSA/AEGIS 공개 | KNOWS는 재구현 후 "re-implemented"로 표시 |
| 실물 로봇 | AG3S만, 개선 주장 없음 | AG3S 우선, 시간이 남으면 baseline 1개 추가 | 동일 (추가 baseline은 정성 비교 수준) |

---

## 2. Benchmark — 무엇을, 어디서

| 환경 | 역할 | 실행 조건 | 규모 |
|---|---|---|---|
| **SafeLIBERO** (최우선) | **주 결과 표** | pi0.5-LIBERO, Franka Panda, OSC_POSE(EE 변위), 20 Hz | 4 suite × 16 task × Level I(target 근처 장애물)/II(경로 차단) = 32 scenario × 50 ep = **1,600 ep / 조건**. 선택: KNOWS의 Level III(이동 장애물) |
| **LIBERO** (원본, 장애물 없음) | 제안: AG3S가 **task success를 떨어뜨리지 않는지** 확인 (안전 layer의 비용 측정) | SafeLIBERO와 같은 checkpoint·adapter | 4 suite 표준 평가 |
| **MuJoCo (RB-Y1, 자체 과제)** | 자체 과제 결과 + **ablation 전부** + mechanism figure | 16-D joint 공간, 기존 AG3S/trajopt stack | held-out scene family (개발 scene과 분리) |

### 2.1 SafeLIBERO/LIBERO에 AG3S를 붙이려면 필요한 구현 (SL-0 ~ SL-8)

| ID | 구현 | 완료 확인 |
|---|---|---|
| SL-0 | SafeLIBERO 코드·asset, pi0.5-LIBERO checkpoint, openpi LIBERO client 연결 | pi0.5 단독 1 episode가 공식 판정으로 종료 |
| SL-1 | Observation adapter: agent-view·wrist RGB + **depth**, intrinsics/extrinsics → AG3S wire 형식 | depth back-projection이 scene과 겹침 (overlay) |
| SL-2 | Franka gripper(palm·finger) query sphere, self-filter용 전신 sphere | mesh–sphere overlay |
| SL-3 | pi0.5-LIBERO의 action→image attention 추출, head 선택 (KNOWS의 L12·H3에서 출발, **dev split에서만** 선택) | dev split에서 target 위 peak 비율 |
| SL-4 | ATS·LTE 연결: multi-view fusion, target cluster, target-free ESDF (cuRobo) | target/obstacle 분리 overlay |
| SL-5 | **CCR adapter** — LIBERO는 EE 변위, CCR은 joint 공간. (a) EE 변위 chunk를 직접 최적화 또는 (b) IK로 joint 공간에 옮겨 정제 후 복귀. smoke test로 선택 | 정제된 chunk가 OSC로 실행되고 clearance 재검사 통과 |
| SL-6 | Phase 처리: LIBERO 시연엔 phase label이 없음 → **strong rule을 기본**, HSR은 MuJoCo에서만 평가 | gate 로직 동일, 출처만 교체 가능 |
| SL-7 | 평가기: 공식 성공 판정 + contact event 원자료 → CAR·TSR·ETS(VLSA) / SR·CR·SSR(KNOWS) 모두 계산 | **pi0.5 공식 수치를 우리 평가기로 재현** |
| SL-8 | Baseline 실행 (3절) | 각 셀에 출처 표시 |

### 2.2 MuJoCo에서 먼저 정할 것 (선행 조건)

- 기준 snapshot (V7g 또는 V8) 1개 확정. geometry를 바꾸면 영향받는 조건을 모두 다시 돌린다.
- **검사 link 범위** 확정 — palm·finger만 검사하면 LTE 대 "전 link에서 target 제거"의 차이가 거의 안 나온다. LTE 실험 전에 결정.
- 독립 contact evaluator: allowed/forbidden contact를 method 출력과 무관하게 정의 (아래 표).

| Contact pair | 판정 |
|---|---|
| 허용 link – 실제 target | 허용 (grasp에 필요) |
| 비허용 link – target | forbidden |
| robot – 비대상 물체/장애물 | forbidden |
| 잡은 물체 – 장애물 | forbidden |
| 물체 – 목적지/지지면 | 정상 placement만 허용 (table 일괄 면제 금지) |

---

## 3. Baseline — 누구와, 어떻게 비교

| Baseline | 코드 | 방법 핵심 | 우리와의 차이 | 구현 방식 |
|---|---|---|---|---|
| **pi0.5 단독** | 있음 | 안전 layer 없음 | — | 직접 실행 (필수) |
| **KNOWS** (arXiv 2606.09749) | **미공개** | YOLOE로 모든 물체 segmentation → 물체별 MVEE ellipsoid → **pi0.5 attention(L12·H3)으로 target 선택** → target 제외 → EE 1-step CBF-QP | **가장 가까운 선행 연구.** 우리는 TSDF/ESDF(segment 안 된 geometry도 포함), chunk refinement, link별 exemption, attached object | 핵심만 재구현, 표에 "re-implemented" 표시. YOLOE는 우리 물체에 fine-tune 필요할 수 있음 |
| **VLSA / AEGIS** | 공개 | VLM(GLM-4.5V)이 장애물 1개 이름 지목 → GroundingDINO → ellipsoid → EE CBF-QP. SafeLIBERO를 만든 논문 | 외부 VLM·detector 필요, 장애물 1개, EE만 보호 | 공개 코드 재실행 |
| **Lan-o3dp** (ICRA 2025) | 공개 | GPT-4가 instruction에서 target/obstacle 분리 → Grounded-SAM point cloud → **diffusion policy의 denoising 중 cost guidance** | VLA가 아님 (point-cloud diffusion policy를 scratch 학습). 장애물은 최근접 1점, EE만 | **이식 방법 결정 필요 (Q2)** |

### 3.1 공정 비교 규칙

- **reported 수치와 re-run 수치를 같은 열에 섞지 않는다.** checkpoint·level·episode 수·지표 정의가 같을 때만 같은 블록. 다르면 별도 블록 + †/* 표시.
- **지표 이름이 같아도 정의가 다르다** (VLSA: CAR/TSR/ETS, KNOWS: SR/CR/SSR). 원자료(contact event, 성공 판정)를 남겨 모두 계산 가능하게 한다.
- 표에 **protected links**, **privileged input**, **추가 model/학습** 열을 둔다 (SafeLIBERO에서 우리가 gripper만 검사하면 VLSA·KNOWS와 같은 범위).
- "Naive"라는 단어 주의: KNOWS 논문의 "Naive"는 **VLSA를 GT segmentation으로 흉내 낸 재구현**을 뜻한다. 우리 KNOWS 재구현은 "KNOWS (re-impl.)"로 부른다.
- SafeLIBERO ≠ LIBERO-Safety (별도 benchmark).

---

## 4. 실물 로봇 (RB-Y1)

| 단계 | 내용 | 주장 범위 |
|---|---|---|
| 기본 | AG3S만 적용, 대표 task 몇 개 실행 | **실행 가능성 시연** — 개선 주장 없음 |
| 시간 여유 시 | baseline 1개 추가 (같은 task·같은 초기 배치) | 소수 trial이므로 정성 비교 수준 |

선행: sim에서 main/ablation이 고정(10/21)된 뒤 착수. 영상은 supplementary video에 사용.

---

## 5. 실험 목록

우선순위: **필수** = 본문 Main/Ablation 표에 필요 · **권장** = 약점 보완 · **선택** = 여유 있을 때.
(괄호 안 DX-ID는 원본 6절의 상세 명세)

### 5.1 선행 검증 (모든 실험 전에)

| # | 실험 | 환경 | 내용 | 통과 기준 | 우선 |
|---|---|---|---|---|---|
| E0 | 평가기·geometry 검증 (DX-00) | MuJoCo + SafeLIBERO | 알려진 allowed/forbidden contact scene으로 evaluator 판정, mesh–sphere coverage, target ball 안 non-target surface leakage | evaluator 오판 원인이 설명됨. SafeLIBERO는 **pi0.5 공식 수치 재현** | 필수 |

### 5.2 Main 비교 (Benchmark × Baseline)

| # | 실험 | 환경 | 비교 조건 | 측정 | 우선 |
|---|---|---|---|---|---|
| E1 | **SafeLIBERO main** (DX-07) | SafeLIBERO L-I/II (+L-III 선택) | pi0.5 · pi0.5+AG3S · VLSA/AEGIS · KNOWS(re-impl.) · Lan-o3dp | CAR·TSR·ETS / SR·CR·SSR, episode 수, CI | 필수 |
| E2 | **LIBERO task 보존** | LIBERO (장애물 없음) | pi0.5 · pi0.5+AG3S (+baseline) | task success 차이, stop/timeout | 권장 (Q1) |
| E3 | **MuJoCo main** (DX-07) | MuJoCo held-out | AG3S vs KNOWS → VLSA → Lan-o3dp (2개 먼저) | task success · forbidden contact · safe success · stop · 완료 시간 | 필수 |

### 5.3 Ablation — 부품 대 부품 (MuJoCo, 나머지는 모두 고정)

| # | 바꾸는 부품 | 대안 | 답하는 질문 | 우선 |
|---|---|---|---|---|
| E4 | **ATS** (DX-01) | ① 외부 segmenter (VLSA·KNOWS식) ② 기하 규칙 (손에 가장 가까운 물체) ③ GT target (참고용) | perception model 없이도 target을 충분히 찾는가 | **필수 (최중요)** |
| E5 | **HSR** (DX-02) | ① gate 없음 ② **strong rule** (gripper 명령·개도·grasp latch·이력) ③ GT phase (참고용) | 학습한 read-out이 단순 규칙보다 나은가 | **필수 (최중요, 가장 먼저)** |
| E6 | **LTE** (DX-04) | ① exemption 없음 ② 전 link에서 target 제거 (MoManipVLA식) | link별 처리가 필요한가 | 필수 (검사 link 범위 확정 후) |
| E7 | **Attached object** (DX-05) | off / on | 잡은 물체를 collision 계산에 넣어야 하는가 | 필수 |
| E8 | Target × phase 2×2 (DX-03) | {기하 규칙, ATS} × {strong rule, HSR} | 두 신호의 개별·결합 효과 | 선택 |
| E9 | CCR (DX-06) | stop-only / per-command correction / K-step CCR | chunk 전체 refinement가 이득인가 | 선택 (CCR을 독립 기여로 주장할 때만) |

E5 주의: strong rule을 gripper 개도 하나만 보는 약한 규칙으로 만들지 않는다. dev set에서 정하고 test 전에 고정.

### 5.4 보조 실험

| # | 실험 | 내용 | 우선 |
|---|---|---|---|
| E10 | Runtime (DX-09) | stage별·end-to-end latency (p50/p95/p99), 533 ms chunk 주기 대비 deadline miss, memory, attention 추가 pass 비용 | 필수 (짧은 표) |
| E11 | Robustness (DX-08) | target–distractor gap, occlusion/depth noise, instruction target 변경, retry. 한 축씩 | 권장 (실패 범위·limitation용) |
| E12 | 실물 RB-Y1 | 4절 | 기본 AG3S만 |

### 5.5 공통 metric

| Metric | 정의 |
|---|---|
| Task success | lift → release → stable placement 완료 episode 비율 |
| Forbidden-contact rate | 금지 접촉이 1번 이상 발생한 episode 비율 |
| Safe success | **같은 episode에서** 성공 + 금지 접촉 없음 (두 비율을 곱하지 않음) |
| Stop / timeout | 정지만 늘려 얻은 충돌 감소를 분리하기 위해 함께 보고 |
| 기록 누락 | `UNKNOWN` — 0 contact로 간주하지 않음 |

### 5.6 결과 → Figure/Table 연결

| 산출물 | 채우는 실험 |
|---|---|
| Main Table | E1 (+E2), E3 |
| Ablation Table | E4 · E5 (핵심 행), E6 · E7 |
| Figure 1 (실제 장면의 문제와 개선) | E1/E3 rollout |
| Figure 2 (pipeline) | 구현 audit |
| Figure 3 (같은 장면, target만 바뀔 때 attention → constraint → 실행 차이) | E4 (+E5 phase timeline) |
| Runtime 표 | E10 |

---

## 6. 넘기기 전에 정할 것

| # | 질문 | 제 의견 |
|---|---|---|
| Q1 | LIBERO(장애물 없음)의 역할은? | "AG3S가 task success를 깎지 않는다"를 보이는 용도가 가장 자연스럽습니다. 이 경우 non-inferiority margin을 미리 정해야 합니다 |
| Q2 | Lan-o3dp를 어떻게 이식할까? | (a) 그 perception 부분(LLM target/obstacle 분리 + Grounded-SAM)만 E4의 "외부 segmenter" 대안으로 사용, (b) cost guidance를 pi0.5 flow-matching sampling에 이식, (c) Lan-o3dp policy 자체를 LIBERO 시연으로 학습. 비용은 a < b < c. Main 표용이면 b, ablation용이면 a |
| Q3 | VLSA/AEGIS 공개 코드가 SafeLIBERO 공식 수치를 재현하는가? | SL-0 단계에서 확인 후 re-run vs reported 결정 |
| Q4 | MuJoCo 기준 snapshot (V7g / V8)과 검사 link 범위 | E6 전에 반드시 확정 |

## 7. Claude Code에 넘길 때의 순서 제안

한 세션에 한 phase씩 진행하고, 각 phase는 implementer → verifier → scribe 순서로 처리합니다.

| Phase | 범위 | 판정 시점 |
|---|---|---|
| P1 | SL-0 + E0 (SafeLIBERO에서 pi0.5 단독 공식 수치 재현) | 10/8 – 10/14 |
| P2 | SL-1 ~ SL-6 (AG3S adapter) → AG3S 1 episode 완주 | **10/10까지 완주 못 하면 재판정** |
| P3 | E1 pilot (suite 1개, Level I/II) + VLSA/AEGIS re-run | 10/14 |
| P4 | MuJoCo: E5 → E4 → E6 / E7 (ablation) | 10/21 주장 고정 |
| P5 | Baseline 구현: KNOWS(re-impl.) → Lan-o3dp (Q2 결정대로) → E3 | 10/21 – 10/28 |
| P6 | E1 전체 실행 + E10 runtime | 10/28 |
| P7 | 실물 RB-Y1 (AG3S) | 여유 시 |

**중단 기준:** 10/10에 1 episode를 완주하지 못하거나 10/14에 pi0.5 공식 수치를 재현하지 못하면, 원인을 기록하고 SafeLIBERO를 suite 일부로 축소하거나 MuJoCo held-out을 주 결과로 전환한다.
