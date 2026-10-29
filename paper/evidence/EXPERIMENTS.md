# Experiments and Evidence Eligibility

## 2026-10-18 추가: EXP003–EXP006 (REGISTERED_SOURCE_LINKED)

**등록 수준의 뜻.** `REGISTERED_SOURCE_LINKED` = 등록한 모든 수치를 `python3 paper/tools/analysis/audit_cvpr_v1_evidence.py`가 source JSON의 기록된 JSON pointer에서 다시 읽어 대조하고(float tolerance 5e-4, count·string exact) source 파일 sha256을 [audit](../experiments/summaries/CVPR_V1_evidence_audit.json)에 고정했다는 뜻이다. **EXP001·EXP002의 `REGISTERED`와 달리 16-D manifest validation(NPY array header, per-run record 재집계, rollout·model 실행)은 하지 않았다.** 따라서 이 네 실험은 "source-linked number registry"이며, 원고 제출 전 `/cvpr-evidence full`에서 manifest 수준 검증을 추가할지 결정한다. WORKLOG에만 있는 항목은 `source=worklog`(text-presence만 확인)로 남긴다. 2026-10-18 audit: 434 checks, 431 PASS, 0 FAIL, 3 `source=worklog`.

| ID | 범위 (population · 조건) | 상태 | 결과 |
|---|---|---|---|
| EXP003 | IFG held-out locked grounding. 성공 rollout **34 held-out episode, 2,523 valid frame** (clear 13 / adjacent 12 / place 9); 3,196 frame 중 s0_invalid 29 제외; 평가 전 잠근 설정(`LOCK_heldout.json`); replay depth, offline; arm G:N / A-fk / A / B | `REGISTERED_SOURCE_LINKED`, `OFFLINE_HELDOUT_GROUNDING` | R008–R012. 실행 성능·permission builder·runtime 근거 아님. 실패 9 episode는 appendix로 scope 밖 |
| EXP004 | Hurdle closed-loop, T43 H set **7 episode × 2 seed = 14 run**, profile `t43p2_overbar`. 세 code state의 결과를 같은 (episode, seed)에서 나란히 등록: **E0 (VLA 단독) 2026-10-05, T43 W snapshot** (TW); **V8 / V11A / V11AB 2026-10-07, T43 V11 snapshot** (TV); **C1a / C1aN / SURF 2026-10-17, 현재 구현** (HA/HT) | `REGISTERED_SOURCE_LINKED`, `MIXED_CODE_STATES` | R013–R018. E0 대 V11AB·대 SURF의 paired 2×2는 source에 없어 등록하지 않음(TV pairs는 V11AB vs V8 / V11A, HA pairs는 2026-10-17 세 arm 사이만). 표본 14 |
| EXP005 | Held-out closed-loop C1aN vs SURF (IFG-CL1), **34 seed 쌍 = collector에서 C1a가 성공한 episode에서 선별** (`/C1a_reference/n_success` 34/34); 같은 cuRobo·TrajOpt·flag; 2026-10-17 03:30 | `REGISTERED_SOURCE_LINKED`, **`EXPLORATORY_SELECTION_CONDITIONED`** | R019–R021. 등록 해석 문장은 WL 2026-10-18 14:30을 영어로 옮긴 것(RESULTS R019). 선별 없는 재평가 IFG-CL3는 미등록 |
| EXP006 | Limitations용 context: T58 adjacent post (3 ep × E0/ON × with/without post, 2026-10-13), CL4 (18 pair, 5 arm, 2026-10-15), T46d gripper-scope degeneracy (2026-10-12), T44e-CL2 C0 (18 run, 2026-10-12), T53 test 16 (2026-10-13) | `REGISTERED_SOURCE_LINKED`, `CONTEXT_ONLY` | R022–R026. 각 결과는 자기 code state·scope에 한정되며 서로 비교하지 않는다 |

EXP002 dev-set component 수치(R004–R007)는 cross-reference만 하고 중복 등록하지 않는다. 원고 위치별 지도: [NUMBER_MAP_CVPR_V1](NUMBER_MAP_CVPR_V1.md).

## 2026-10-15 추가: EXP002

**REGISTERED / OFFLINE_DEVELOPMENT_ONLY.** T53/T54c 180 selected frames / 32 episodes, replay depth, 실제 state/action 16-D. [Manifest](../experiments/manifests/ifg/EXP002_IFG_development.json), [audit](../experiments/summaries/EXP002_IFG_audit.json). 검증은 `python3 paper/tools/analysis/audit_ifg_evidence.py`. 기존 EXP001의 trajectory 형식을 변경하지 않고 per-request IFG 자료에 맞는 별도 validator를 사용한다. 새 rollout·held-out·permission builder·runtime 근거가 아니다.

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
5. 등록·validator 상태와 무관하게 사용할 수 있으며, 자료 유형·source·scope·검증 상태를 함께 기록한다.

Validator는 연구적 타당성·novelty·physical correctness를 자동으로 판정하지 않는다.
자세한 운영은 [manifest README](../experiments/manifests/README.md)를 따른다.
