# Figures and Tables: Evidence Map

Updated: 2026-10-05 21:01 KST

## Manuscript 자료 — 계획·conceptual draft·실측 결과를 구분

| ID | 역할 | 연결할 claim·experiment | 준비 상태 |
|---|---|---|---|
| F01 | Motivation/teaser | C001, P001; 같은 scene의 baseline/ours와 실제 outcome | `PLANNED`; 성공 장면 선정 전 |
| F02 | Architecture | C002–C004; source/code version과 단계별 data flow | `CONCEPTUAL_DRAFT`; native-LaTeX diagram, 구현 사실·성과 인증 아님 |
| F03 | Attention→3D grounding | C002, P002·P003; 2D/3D/target GT | `PLANNED` |
| F04 | Qualitative comparison | C004, P001; matched scene·seed, 성공과 실패 | `PLANNED` |
| F05 | Failure/runtime/tradeoff | C007, P005 및 실패 진단 | `PLANNED`; page budget에 따라 supplementary 배치 |
| T01 | Main task·safety 결과 | C001·C004, P001 | `PLANNED`; safety 값 미등록 |
| T02 | Generalization | C005, P004 | `PLANNED` |
| T03 | Ablation | C002–C004, P003 | `PLANNED` |
| T04 | Efficiency | C007, P005 | `PLANNED` |

## 기존 진단 자료의 index

| Source 자료 | 용도 | 현재 상태 |
|---|---|---|
| [T39 paired table](../../benchmark/ag3s/docs/figures/t39/t39-paired-2x2.png) / [sidecar](../../benchmark/ag3s/docs/figures/t39/t39-paired-2x2.json) | R003의 source 맥락 | `INDEXED`; 그대로 publication-ready라고 판정하지 않음 |
| [T39 summary table](../../benchmark/ag3s/docs/figures/t39/t39-summary-table.png) / [sidecar](../../benchmark/ag3s/docs/figures/t39/t39-summary-table.json) | 기존 outcome와 기타 진단 | `INDEXED`; registry 밖 수치를 자동 사용하지 않음 |

새 figure/table마다 result ID, manifest, verify key, sidecar, plotting script/revision,
input hash, caption의 scope, anonymity를 연결한다. Cell·error bar마다 numerator·denominator·unit·
aggregation이 확인되어야 한다. 결측값은 미측정으로 표시한다.

Step 3에서는 artifact를 생성하지 않았다. Step 5에서 F02 conceptual diagram·A01 proposed
algorithm·T00 descriptive table와 나머지 figure/table의 placeholder·PLAN·sidecar를 구성했다.
기존 research figure는 덮어쓰지 않았다. Scaffold의 렌더링 검증과 publication-ready 판정은 별개다.

## Step 5 artifact 상태와 경로

| ID | 상태 | 파일 / 범위 |
|---|---|---|
| F01 | `PLANNED_PLACEHOLDER` | [Teaser plan](../figures/fig01_teaser/PLAN.md); 실측 비교 없음 |
| F02 | `CONCEPTUAL_DRAFT` | [Overview](../figures/fig02_overview/PLAN.md); C002–C004의 proposed narrative |
| F03 | `PLANNED_PLACEHOLDER` | [Grounding plan](../figures/fig03_attention_to_3d/PLAN.md); GT·결과 필요 |
| F04 | `PLANNED_PLACEHOLDER` | [Qualitative plan](../figures/fig04_qualitative/PLAN.md); matched outcomes 필요 |
| F05 | `PLANNED_PLACEHOLDER` | [Analysis plan](../figures/fig05_analysis/PLAN.md); failure/runtime scope 필요 |
| T00 | `DRAFT_FROM_REGISTERED_RESULTS` | [Descriptive table](../tables/table00_descriptive/PLAN.md); C006·EXP001·R001/R002·M001만 |
| T01 | `PLANNED_PLACEHOLDER` | [Main plan](../tables/table01_main/PLAN.md); P001의 task/safety는 미측정 |
| T02 | `PLANNED_PLACEHOLDER` | [Generalization](../tables/table02_generalization/PLAN.md); unseen split 필요 |
| T03 | `PLANNED_PLACEHOLDER` | [Ablation](../tables/table03_ablation/PLAN.md); controlled intervention 필요 |
| T04 | `PLANNED_PLACEHOLDER` | [Efficiency](../tables/table04_efficiency/PLAN.md); timing boundary 필요 |
| A01 | `CONCEPTUAL_DRAFT` | [Algorithm plan](../algorithms/PLAN.md); source 구현 audit 아님 |

T00을 P001 main result로 바꾸거나 T01의 missing safety cell에 EXP001 task count를 넣지 않는다.
F02/A01에는 source-note·native render-source hash를, T00에는 exact verify pointer와 값·분모·
source hash를 기록했다. Placeholder sidecar의 수치 값은 null이다.
Default main은 F02/A01/T00/T01을 포함한다. 나머지 placeholder는 소스만 준비했다.
Section 연결과 working page budget은 [MANUSCRIPT_MAP](../notes/MANUSCRIPT_MAP.md)을 따른다.
