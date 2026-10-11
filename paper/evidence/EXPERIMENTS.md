# Experiments and Evidence Eligibility

Updated: 2026-10-05 19:45 KST

## 등록된 실험

| ID | 범위 | 상태 | 사용 가능한 결과 |
|---|---|---|---|
| EXP001 | T39 snapshot; apple·left-arm·clear scene; episode와 policy seed를 맞춘 E0/E3b closed-loop simulation | `REGISTERED`, `DESCRIPTIVE_TASK_OUTCOME_ONLY` | R001–R003; physical safety·runtime·현재 main의 성능은 포함하지 않음 |

[Manifest](../experiments/manifests/EXP001_T39_task_outcomes.json),
[source audit](../experiments/summaries/EXP001_source_audit.json),
[RESULTS](RESULTS.md)를 함께 읽는다. 이번 단계에서는 기존 기록의 source identity와
배열·metadata를 확인했으며 scientific experiment를 다시 실행하지 않았다.

## 계획된 실험 — 결과나 whitelist가 아님

| 계획 ID | 원문 파일명 예시 | 질문·비교 | 필요 metric | 상태 |
|---|---|---|---|---|
| P001 | E001_main_safety | 동일 checkpoint·scene·seed에서 vanilla VLA와 통합 method 비교 | Task success, forbidden-contact rate, safe success, geometry clearance, HOLD | `PLANNED` |
| P002 | E002_attention_3d | 실측 attention이 phase별 task target과 맞는가? 단일 view·multi-view·대조 attention 비교 | Grounding error, target selection, obstacle retention | `PLANNED` |
| P003 | E003_ablation | Grounding, target contact policy, safety field, refinement, gate의 기여 분리 | P001·P002 metric, action modification | `PLANNED` |
| P004 | E004_generalization | Unseen object/obstacle/layout/occlusion과 scene별 평가 | 조건별 task·safety, split provenance | `PLANNED` |
| P005 | E005_runtime | 같은 측정 경계의 latency, cold/warm start, deadline miss, memory | End-to-end 및 stage timing | `PLANNED` |
| P006 | E006_real_robot | Simulator에서 확인한 범위를 hardware에서 확인할 수 있는가? | Hardware task·contact, emergency stop, 시간 경계 | `OPTIONAL_PLANNED`; 실행·자료 없음 |

원문 E001–E006은 제안된 파일명이며 실행 완료를 뜻하지 않는다. 계획은 이 문서에서
관리하고, 실제 16-D source audit를 통과한 자료만 `manifests/`에 등록한다.
샘플 수, task success 판정, allowed contact, test split, non-inferiority margin은 실행 전에
고정해야 한다. 현재 미정인 값을 임의로 채우지 않는다.

## 기존 후속 자료의 처리

| 자료 | 확인한 범위 | 현재 처리 |
|---|---|---|
| [T40FG.verify.json](../../benchmark/ag3s/docs/handoff/T40FG.verify.json) | Obstacle 시연과 제한된 검사 범위, false-zero 문제를 research log에서 확인 | Candidate; geometry/contact 정의와 selection bias 검토 전 미등록 |
| [T40I.verify.json](../../benchmark/ag3s/docs/handoff/T40I.verify.json) | Records-only 진단과 재구성하지 못한 held-object query가 있음 | Candidate; scope·log 연결·전체 provenance 별도 확인 필요 |
| [T41a.verify.json](../../benchmark/ag3s/docs/handoff/T41a.verify.json) | 별도 code snapshot과 후속 실행 자료가 있음 | Candidate; 현재 읽은 정본 log에서 해당 절 연결을 확인하지 못해 미등록 |
| `outputs/verify/T42/` | 후속 산출물 위치를 발견함 | 미검토·미등록; 진행 중 파일을 완료된 결과로 읽지 않음 |

Candidate의 미등록은 결과가 틀렸다는 판정이 아니다. 이 Step에서 검토한 범위로는
논문 근거의 요건을 확정할 수 없다는 뜻이다.

## 등록 요건

1. Model/checkpoint·실제 state/action dimension·source version과 config를 확인한다.
2. Episode·seed·condition·split·실행 command·완료 상태·raw 자료를 연결한다.
3. Research log와 verify.json의 **정확한 key**에 결과를 연결한다.
4. 측정 범위와 미측정 항목을 기록하고 source identity를 고정한다.
5. Schema와 semantic validator를 통과한 `REGISTERED` 항목만 해당 evidence scope에서 사용한다.

Validator는 연구적 타당성·novelty·physical correctness를 자동으로 판정하지 않는다.
자세한 운영은 [manifest README](../experiments/manifests/README.md)를 따른다.
