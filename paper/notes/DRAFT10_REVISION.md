# Draft 10 — CVPR 기여 중심으로 Introduction·Related Work·Method 개정

기록 날짜: 2026-10-08 KST (중단 후 재개; 클라이언트 날짜 기준)  
상태: `DRAFT — EVIDENCE PENDING`  
원고: [main.tex](../latex/main.tex)  
실행 기준: [develope_PLAN](../../develope_PLAN.md) 0절과 [확정된 실험 구성](SAFELIBERO_PLAN.md)

## 1. 이번 개정의 판단

핵심 기여는 **외부 segmentation model 없이 VLA attention과 depth에서 얻은 대상 기하를 작업 단계·링크별 접촉 제약으로 연결하는 것**이다. ATS·HSR·LTE·CCR을 각각 새로운 독립 기여로 주장하지 않는다. HSR은 target identity와 exemption의 유지 시점을 조절하는 보조 구성이고, CCR은 구성한 제약을 실행하는 optimizer다.

영문 working title을 `AG3S VLA: Grounding VLA Attention in 3D for Task-Dependent Contact Constraints`로 변경했다. 최종 제목은 결과와 주장 범위에 맞춰 조정할 수 있다.

| 부분 | 개정 내용 |
|---|---|
| Introduction | 필요한 접촉과 금지 충돌의 차이 → geometry만으로 부족한 정보 → 기존 연구와의 관계 → attention/depth에서 contact constraint로 이어지는 아이디어 |
| Related Work | VLA/기하 refinement, internal grounding, task-dependent collision geometry, frozen-feature probing의 네 문단 |
| Method | Problem/query 정의 → ATS → HSR와 grasp state → LTE/attached geometry → CCR와 실행 범위 |
| Abstract/Conclusion | 같은 기여 방향으로 정리; 확인되지 않은 성능은 구체적인 TBD |
| Experiments | 확정된 SafeLIBERO 주 평가, RB-Y1 자체 비교/ablation, 실물 feasibility 구성과 일치시킨 작성 골격 |
| Figure 1/3 | 실제 원자료로 채울 storyboard; 측정 결과로 표시하지 않음 |
| Figure 2 | action-serving pass와 별도 attention-extraction pass, 학습된 probe를 구분한 개념도 |

## 2. Claim·구현 근거 연결

아래는 방법 서술의 근거이며 성능 claim의 등록이 아니다. 기존 C001–C005/C007의 상태를 바꾸지 않았다. 기존 EXP001/R001–R003의 수치를 이번 원고에 전용하지 않았다.

| 원고 주장/위치 | 연결 | 확인 근거와 범위 |
|---|---|---|
| ATS: attention으로 대상 후보를 선택하고 depth로 관측 기하 구성 | K001 / C002, P002–P003 | `benchmark/ag3s/stages/target_grounding.py`; 기존 METHOD_FACTS와 V7G_FACTS. 대상 선택 정확도·extent 품질은 미확정 |
| Additional extraction pass, 동일 checkpoint | Method ATS | `benchmark/trajopt/attention_policy.py`: 별도 model instance, `infer`, `_attention_and_feature`; serving pass의 attention이라고 서술하지 않음 |
| HSR는 학습된 linear classifier | Method HSR | `benchmark/ag3s/stages/subtask_probe.py`, SUBTASK_FACTS. Frozen VLA와 probe training을 구분; feature는 text/state와 image token mean |
| Phase는 identity를 조절하고 grasp state는 attachment를 결정 | K002 / C003 | `target_grounding.py`: `note_subtask`, `note_placed`, `_release_if_placed_home`, `observe`; `grasp_latch.py`; V7G_FACTS |
| Link별 field query와 window-boundary 합성 | K002 / C003 | `benchmark/ag3s/fields/curobo_field.py::target_free_distance`, `curobo_builder.py::update`; 제시 수식과 대조 |
| Target-free ball의 공간적 근사 | C002–C003 | `curobo_builder.py::TargetBall`; 정확한 instance mask처럼 표현하지 않고, 주변 obstacle 포함 가능성과 extent 평가를 연결 |
| Attached-object spheres와 clearing | K002 / C003 | V7G_FACTS의 held-body-cover/cap/slip-detach 설정과 기존 구현 audit. 관측 기하 근사이지 true-object coverage 증명이 아님 |
| CCR objective, overlap, slack | C004 / P003 | `benchmark/trajopt/problem.py::build_problem/objective`, `safe_policy.py::_continuity_input`. Frobenius 내적으로 행렬 미분 표기 정리 |
| Query coverage, waypoint checks, timing budget | C007 / P005 | V7g 설정: gripper links, self-collision off, H=50/K=8. 533 ms는 실행 horizon의 명목 주기이며 latency 성능이 아님 |

V7g 방법과 현재 코드의 관련 interface를 읽어 서술했다. 최종 실험 snapshot(V7g/V8), protected links, SafeLIBERO adapter와 phase 대안은 아직 확정된 실행 결과가 아니다. 특히 active hand의 query만 있으면 LTE와 전 링크 제거의 차이를 충분히 시험하지 못할 수 있으므로, 이 차이를 검증할 modeled non-contact link가 있는지 DX-04 전에 확인해야 한다.

## 3. 문헌 근거

기존 [R2](REFERENCE_ANALYSIS_R2.md), [R3](REFERENCE_ANALYSIS_R3.md)와 로컬 원문을 사용하고, 새 서술에 필요한 다음 primary source를 재확인했다. 전체 bibliography·novelty survey를 다시 수행한 것은 아니다.

| 문헌 | 확인 범위 | 원고에 남긴 비교 |
|---|---|---|
| [KNOWS](https://arxiv.org/html/2606.09749v1) | Method 3.2–3.3, 로컬 PDF의 기존 R2 audit | Segmentation으로 만든 객체와 attention 기반 target 선택; 우리의 depth-grown geometry와 구분 |
| [VLSA](https://arxiv.org/html/2512.11891v1) | Primary HTML와 R2의 perception/filter 분석 | 외부 visual grounding에 의한 obstacle 구성 |
| MoManipVLA | 로컬 PDF p.7, R2 | Real-world Grounding-SAM target mask를 ESDF에서 제외; 모든 실험에 동일하다고 일반화하지 않음 |
| AutoMoMa | 로컬 `cvpr2026_ref_paper/Scalable_Trajectory_Generation_for_Whole-Body_Mobile_Manipulation.pdf`, 기존 highlight 분석 | 알려진 scene의 trajectory generation과 manipulated-object kinematic representation |
| [WBAG](https://arxiv.org/html/2610.01083v1) | Primary HTML와 R3 | Whole-body/attached geometry의 선행 맥락; 우리의 현재 query coverage와 구분 |
| [Kang et al.](https://arxiv.org/html/2503.06287v1) | Abstract/Introduction/Method 개요 | Frozen VLM attention grounding의 선행 맥락 |
| [Bhardwaj et al.](https://arxiv.org/abs/2608.13474), [Zhang et al.](https://arxiv.org/abs/2605.28527) | Primary abstract pages와 R3 | Frozen-feature progress/success probing; HSR의 독립 novelty로 전용하지 않음 |

기존 bibliography key를 재사용했다. CVF 페이지 일부의 도구 접근 실패는 원문의 부재로 해석하지 않았고 MoManipVLA는 로컬 원문을 직접 확인했다. 최신 venue·전체 author 표기 등 서지 정리는 최종 bibliography audit에 남아 있다.

## 4. 중요한 정정

- `no extra training` → **no additional VLA training for collision avoidance**. Probe는 별도 학습.
- `same forward pass` → action serving과 별도 extraction pass. HSR만 extraction cache를 재사용.
- `no external model` → **no external detector or segmentation network for target extraction**.
- `every other object keeps the full field` → non-contact query는 full field, contact query 안의 기하 근사 오류는 별도 고려.
- `every returned trajectory is safe`에 해당하는 표현 → configured geometry/waypoint post-check와 명시된 fallback.
- `Qprev = previous actually executed trajectory` → 실행 feedback으로 정렬한 이전 chunk의 overlap.
- `place` 자체가 attachment를 발생시키는 설명 → 물리적 grasp state가 attachment의 조건.
- 전체 target/destination contact를 충돌 지표에서 제외하던 정의 → object pair·link·phase별 allowed contact를 TBD로 정의.
- 역사적 V5→V7g 누적 실험을 순수 HSR ablation으로 해석하던 표 → 다른 요소를 고정한 부품별 비교 골격.

## 5. 남은 TBD와 실행 연결

| 필요한 확인 | 연결 작업 | 원고에서 채울 부분 |
|---|---|---|
| 최종 snapshot, 링크 coverage, 실행 gate | PA-1/PA-2 | Method implementation, baseline parity, Appendix D |
| Head 선택 자료와 개발/test 분리 | DX-01 / P002 | ATS 설정과 Appendix B |
| Probe fitting/split/phase label 명세 | DX-02 / P003 | HSR와 Appendix C |
| Main evaluation | SafeLIBERO 계획, P001 | Abstract/Introduction/Conclusion 결과 문장, Main Table |
| ATS·HSR·LTE·attached geometry의 고정-controller 비교 | P002–P003 | Ablation Table, Figure 3, Results Q2–Q4 |
| RGB/point cloud/ESDF/contact 원자료 | P0-A/B | Figure 1·3의 실제 패널 |
| End-to-end runtime | P005 | Results Q5, Appendix D |
| 실물 실행 기록 | 확정된 feasibility 계획 | 시연 패널; baseline 없는 개선 주장 금지 |

Appendix B–D는 계획상의 배치이며 이번 작업에서 해당 본문을 새로 완성하지 않았다. 연구 코드·실험 원자료·성능 registry는 수정하지 않았다. 기존 supplementary PDF scaffold는 유지한다.

## 6. 검증 기록

빌드·PDF 점검 결과는 [DRAFT10_BUILD_AUDIT.json](DRAFT10_BUILD_AUDIT.json)에 기록한다. 이 검증은 문서·source 연결·렌더링의 확인이며 실험 유효성, 전체 novelty, 최종 submission readiness를 인증하지 않는다.
