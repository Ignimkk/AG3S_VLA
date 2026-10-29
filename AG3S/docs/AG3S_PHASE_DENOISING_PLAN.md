# AG3S: Target Exclusion 개선, Attention–Sub-task 결합 및 Denoising 계획

작성일: 2026-10-09 (UTC)  
개정일: 2026-10-09 — 코드 확인을 바탕으로 논문 질문 4개와 Claude Code 구현 명세 보강.  
상태: **PROPOSED — 방법론 및 실행 계획. 신규 구현·실험 결과 없음.**  
목표: CVPR 2027 논문의 핵심인 **VLA 내부 신호 → 대상 기하 → 작업 단계·링크별 접촉 제약**을 강화한다. 먼저 target exclusion의 정확도를 개선하고, attention–phase 결합과 denoising 내부 적용을 순서대로 평가한다.

이 문서는 [develope_PLAN.md](../../develope_PLAN.md)의 PAPER-FIRST 원칙과 실험 구성을 따른다. SafeLIBERO는 주 평가, RB-Y1 MuJoCo는 메커니즘 진단과 ablation, 실제 RB-Y1은 feasibility demonstration으로 사용한다. 새 guidance가 기존 논문 완성의 필수 조건이 되는 것은 아니다.

## 1. 핵심 결정

| 방향 | 답할 질문 | 우선 구현 | 채택 근거 |
|---|---|---|---|
| **최우선: ATS→LTE 연결** | 올바른 target을 선택했을 때, 주변 obstacle을 보존하면서 필요한 target contact를 허용하는가? | 실제 target extent에 맞는 seed exclusion과 공통 contact 권한 | 동일 target identity에서 obstacle 보존과 실제 contact·grasp 결과 개선 |
| Attention–Sub-task 결합 | 동일한 attention이라도 phase에 따라 다르게 사용하면 잘못된 target 변경·접촉 예외를 줄이는가? | 기존 3-class probe와 grasp state를 결합한 target lifecycle | 같은 controller에서 strong rule 대비 실행상의 추가 이득 |
| Denoising 내부 TO / guidance | 같은 AG3S 제약을 생성 중에 적용하면 최종 action만 보정하는 것보다 유리한가? | Clean-action TO hook, 이후 velocity guidance | Task·contact 성능과 실제 latency의 개선 또는 명확한 trade-off |

**실험은 분리한다.** 외부 TO를 고정한 상태에서 identity → extent/exclusion → link permission → phase lifecycle 순으로 확인한다. 이후 채택한 제약 구성을 고정하고 TO 적용 방식만 바꾼다. Geometry 오류가 남은 상태에서 denoising 개선을 먼저 수행하지 않는다. 두 방향을 동시에 변경한 성능 차이를 각각의 기여로 해석하지 않는다.

**기존 문서의 범위 평가:** Attention과 HSR 비교는 포함되어 있었으나, geometry 오류의 원인 분리, LTE의 정확한 수정 경로, non-contact link 실험, attached coverage 검증은 부족했다. 아래 2.1–2.4절과 8–9절을 구현 명세로 추가한다. 문서가 질문의 답을 이미 입증한 것은 아니며, 답을 얻을 비교와 채택 기준을 정의한다.

## 2. 현재 구현과 제안의 경계

| 항목 | 현재 확인한 구현 | 이번 계획에서 추가할 부분 |
|---|---|---|
| Attention 추출 | Serving model과 별도의 동일 checkpoint 사본에서 prefix와 첫 suffix pass 수행 | 우선 추출을 action 생성 앞으로 이동; serving pass 재사용은 후속 최적화 |
| Sub-task | Prefix KV의 V feature를 학습된 linear probe로 `pick/place/home` 분류 | 강한 물리 규칙과 비교하고 결측·지연 처리 명시 |
| Phase 활용 | `place/home`에서 새 target 채택·변경 억제; placed 이후 home에서 identity 해제 | 상태 전이와 접촉 권한의 일관된 interface 및 trace |
| Grasp / attachment | Grasp latch가 물리 상태를 바탕으로 부착·해제 결정 | 이 역할 유지; phase 예측만으로 attachment 전환 금지 |
| TO | 최종 decoded action chunk를 외부에서 보정 | 생성 중 clean-action 보정 또는 gradient guidance |
| Sampler hook | `guidance_fn(x_next, time_next)`가 Euler step 직후 호출됨 | Clean estimate, velocity, step index 및 constraint snapshot 접근 추가 |

`no additional VLA training`은 **충돌 회피를 위해 VLA weight를 추가 학습하지 않는다**는 뜻이다. Probe 학습과 inference 중 action gradient 계산은 별도로 명시한다. 외부 segmenter를 제거했다는 사실만으로 전체 비용 감소가 입증되지는 않는다.

### 2.1 논문 질문 4개 → 원인 분리 실험

| 질문 / 실험 | 비교와 통제 | 장면 및 판정 | Figure / Table |
|---|---|---|---|
| **E1: Attention이 필요한가?** | 같은 depth 후보·extent builder·LTE·HSR·TO에서 target selector만 attention vs strong spatial rule로 교체 | 비슷한 거리의 두 graspable object, nearest object가 지시 대상이 아닌 배치, 좌우 위치 교환과 지시 대상 변경. Identity accuracy와 safe/task success를 함께 비교 | Fig. E1: 위치만으로 구별하기 어려운 장면과 선택·실행 결과. Main ablation의 selector 행 |
| **E2: Target 기하가 중요한가?** | **동일한 올바른 target identity**에서 현재 ball vs 관측 extent 기반 exclusion. 소수 oracle extent는 오류 위치 진단용 | Target 옆 distractor·얇은 obstacle·support 근접. 제거 seed, query distance, 실제 forbidden contact, grasp success를 연결 | **Fig. E2 우선:** RGB/3D → 유지·제거된 seed → link query → 실제 contact. Table E2: geometry와 실행 지표 |
| **E3: 링크별 처리가 필요한가?** | 동일 exclusion geometry에서 strict no-exemption / global exemption / authorized-links-only | 손가락 grasp가 필요하면서 전완·손목 등 **실제 검사되는 non-contact link**가 target에 접근하는 장면. Grasp와 non-contact-link contact를 별도 보고 | Table E3: grasp success·비허용 링크 접촉·safe success. Fig. E2에 query link별 결과 통합 |
| **E4: HSR이 필요한가?** | 동일 selector·exclusion·link set에서 none / matched strong rule / learned probe; 필요 시 oracle phase | Place 이후 재채택, 잘못 유지된 exemption, attention focus 전환. 상태 개선이 forbidden contact 또는 task/stop 결과까지 바꾸는지 확인 | 3절의 Table P / Figure P 재사용 |

E1의 attention-off는 ranking만 바꾸는 실험과 **attention 기반 split까지 끄는 전체 ATS 대체**를 구분한다. Ranking 실험에서는 후보를 고정해 선택 능력을 분리하고, 전체 비교에서는 baseline도 동일 depth의 geometry-only split을 사용할 수 있게 한다. 강한 baseline의 기능을 인위적으로 약화하지 않는다. 위치·지시 조합은 development에서 설계하고 held-out 목록을 고정한다.

E2의 correct-identity subset은 원인 진단 표임을 표시한다. 동일 trace의 selector 출력을 고정하거나 진단용 oracle identity로 강제할 수 있으나, **실제 방법의 end-to-end 전체 episode 결과도 별도로 보고**한다. Identity 실패를 제외한 숫자를 전체 성능처럼 사용하지 않는다.

E3의 `relax` 설정은 margin relaxation이므로 strict no-exemption과 같지 않다. 실제 mask·margin을 확인해 세 조건을 구현한다. 접촉 허용 여부와 geometry 제거 방식은 독립 switch로 분리한다. 세 조건의 link set과 collision sphere는 동일하게 유지한다.

### 2.2 코드에서 확인한 약점과 수정 경계

아래는 실행으로 새 결함을 확정한 결과가 아니라 **현재 코드 구조와 그로부터 도출한 검증 과제**이다. 과거 source 주석에 적힌 수치는 새로운 논문 evidence로 취급하지 않는다.

| 모듈 | 현재 존재하는 기능 / 근거 | 남은 약점 | Claude Code가 수행할 변경·확인 |
|---|---|---|---|
| ATS | `ground_target`, `split_by_attention`, admissibility, anchor coverage·extent association이 이미 존재 | 올바른 peak가 전체 target extent를 뜻하지 않음. Split trigger·connectivity에 따라 합쳐진 물체가 남거나 길쭉한 단일 물체가 잘릴 수 있음 | 기존 split을 재사용하고 identity / merge / missing extent / over-split을 분리 측정. 새 splitter부터 만들지 않음 |
| LTE seed exclusion | `AG3S._target_ball`의 fitted sphere/centroid + radius padding; `_exclude_ball_seeds`는 ball 안·support 높이 위 seed 제거 | Support 아래 보호는 있지만 **ball 안의 이웃 object seed를 구별하는 ownership 조건은 없음** | 관측 extent에 근거한 seed mask를 대안으로 추가. Main field와 target-free field를 계속 분리 |
| LTE query·권한 | `to_adapter`가 authorized link mask 생성; `target_free_distance/gradient`가 대체 field 조회 | Field만 고쳐도 ball 기반 `target_volume` 판정·path exclusion에 이전 의미가 남을 수 있음 | 아래 2.3절의 공통 geometry/permission 계약을 field, margin classification, sweep, final verdict까지 전달 |
| HSR | `pick/place/home` probe와 debounce; physical latch·identity guard가 따로 존재 | Phase accuracy와 strong-rule 대비 실행 유용성은 다름. 지속적으로 잘못된 attention은 일관성 검사도 통과할 수 있음 | 같은 state machine에 phase source만 교체. 신호 충돌 시 유지·축소·종료 규칙을 명시 |
| Attached geometry | Attach snapshot, fitted/body spheres, optional `held_cover_cap`, slip detach | 관측되지 않은 면·형상 근사·cap이 coverage를 제한할 수 있음 | Cap 전후의 observed coverage와 mesh-reference coverage를 분리 측정. 문제 장면에서만 primitive/uncertainty 개선 |
| CCR | Waypoint 검사 외에 optional `sweep_check`·servo prediction·최종 path check 구현됨 | Sampling cap, servo 오차, 미관측 기하, latency로 실제 보호 범위가 제한됨 | 활성 설정·실제 sample spacing·누락 링크·fallback 사용을 기록. Continuous collision guarantee로 설명하지 않음 |

**우선 점검할 연결부:** `sweep.py::PathChecker._target_rows`는 manipulated distance/`target_volume`으로 robot-sphere target 행을 제외하며, 함수 내부에 authorized contact-link 교집합이 명시되어 있지 않다. Non-contact link가 target에 접근하는 경우를 재현해 전체 caller 경로에서 권한이 보존되는지 확인한다. `linearize.py::in_target_volume` 역시 ball 내부라는 기하 판정이지 접촉 권한 판정이 아니다. 이는 **감사할 잠재적 권한 불일치**이며, 현재 모든 실행에서 충돌을 일으킨다고 단정하지 않는다.

### 2.3 최우선 구현 — Target identity를 정확한 exclusion으로 연결

**최소 변경안:** 기존 ball baseline을 유지하고 `observed_seed_mask` 대안을 추가한다. OBB나 convex hull로 모양만 바꾸는 것 역시 사이의 obstacle을 포함할 수 있으므로, 기본 대안은 **관측된 target 표면에 대응하는 seed만 선택적으로 제외**하는 방식이다. 이름은 제안이며 실제 config에 아직 존재하지 않는다.

1. **표본과 원본의 연결 보존.** ATS가 선택한 cluster의 source frame/view/pixel 또는 dense cloud index를 유지한다. Downsampled grounding point만 바로 seed 제거에 쓰지 않는다. 현재 ball은 sparse point 제거에서 target seed가 남는 문제를 해결하려고 도입됐으므로, 단순히 `_exclude_seed_points`로 되돌리면 regression이 생길 수 있다.
2. **Target extent와 obstacle evidence 분리.** 선택한 cluster를 원 depth/dense surface로 lift한다. Depth discontinuity, support plane, 다른 cluster와의 경계를 사용해 불필요한 연결을 제한한다. Target 후보 점, non-target 표면 점, ownership 불명 점을 분리한다. Attention score를 곧바로 voxel 소유 확률로 해석하지 않는다.
3. **ESDF seed와 정합.** 같은 origin/voxel convention에서 target 지지 표면에 근접하고, non-target/support evidence와 충돌하지 않는 seed만 제거한다. Distance threshold와 최소 관측 조건은 development에서 결정한다. Multi-view 상충 또는 target/obstacle이 같은 voxel을 공유하면 우선 보존하고 ambiguity를 기록한다. 필요할 때만 해당 영역의 resolution을 높인다.
4. **오래된 기하 처리.** ESDF에 과거 frame의 seed가 남을 수 있으므로 현재 보이는 표면만 지웠다는 사실을 전체 target 제거와 혼동하지 않는다. 유지된 target anchor와 제한된 local history를 이용하되 pose/identity 일치가 확인된 evidence만 합친다. Unknown 영역을 무조건 target로 확장하지 않는다.
5. **Field 계약 유지.** Target-free field는 별도 생성하고 unauthorized link는 main field를 조회한다. Support·static obstacle은 유지한다. 기존 seed 제외 후 sign correction, fine-window boundary 합성, 값과 gradient의 일치 경로를 재사용·검증한다. Window boundary의 보수성은 내부 seed ownership 오류를 해결해주지 않는다.
6. **모든 소비자 갱신.** `target_volume`을 계속 쓴다면 ball baseline 전용임을 명시하고 새 방식은 shared target-membership 판정을 사용한다. `linearize`, `sweep`, margin classification, final `allowed_contact`/fallback 판정에 같은 geometry version과 link permission을 전달한다. 공간 내부라는 이유만으로 접촉을 허용하지 않는다.

권장 계약은 아래와 같다. 새 범용 framework를 만드는 대신 기존 snapshot/record에 필요한 필드만 추가한다.

```text
TargetEvidence:
  identity_id, geometry_version, observation_time, coordinate_frame
  observed_target_surface, non_target_surface, ambiguous_support
  source_point_or_pixel_ids, validity_reason
ExclusionSpec:
  method = ball | observed_seed_mask
  geometry_version, seed_mask_or_membership, support_guard
  allowed_links, phase_state, physical_grasp_state, expires_at
```

**완료 조건:** 같은 correct identity에서 target contact가 가능할 만큼 target seed는 제거되고, 인접 obstacle은 더 잘 보존되어야 한다. 제거량을 줄였다는 사실만으로 성공 처리하지 않는다. Grasp 실패·hold 증가 없이 실제 forbidden contact 개선까지 확인해야 논문의 기술적 기여로 채택한다. Unknown/ambiguous 입력에서 임의로 ball로 넓혀 fallback하지 말고, 적용 축소·기존 유효 geometry 유지·hold 중 선택과 원인을 기록한다.

### 2.4 Geometry → query → contact의 측정 정의

Metrics의 ROI, voxel resolution, surface matching tolerance는 평가 전에 고정한다. Simulator instance/mesh 정보는 **평가 reference**로만 사용한다.

| 층위 | 측정 | 해석 |
|---|---|---|
| Identity | 지시 대상 선택 accuracy / wrong-target adoption | 무엇을 골랐는가 |
| Observed extent | Target-visible-point precision/recall, distractor contamination, merge/over-split | 선택 영역에 무엇이 포함·누락됐는가; 가려진 면의 recall로 혼동 금지 |
| Seed exclusion | `target removal recall = removed target seeds / target seeds in ROI`; `obstacle retention = retained non-target seeds / non-target seeds in ROI` | 필요한 제거와 주변 보존을 동시에 측정. Shared/unknown seed는 별도 건수로 보고 |
| Distance query | 권한/비권한 링크 각각의 oracle-no-target field 대비 distance overestimation, 잘못 feasible로 바뀐 query 수 | Geometry 오류가 실제 controller 입력을 얼마나 낙관적으로 바꿨는가 |
| Execution | Grasp success, target에 대한 non-contact-link contact, distractor contact, safe/task success, hold/timeout | 모델의 기하 차이가 실행 결과로 이어지는가 |

Seed count는 volume 자체가 아니다. Dense reference에서 제거된 obstacle volume을 계산할 때는 동일 voxel grid와 실제 제거 mask의 overlap을 별도로 사용한다. Query 오차 진단에서는 같은 `q`와 link query를 평가해 trajectory 변화와 geometry 변화를 분리한다. End-to-end 실행에서는 각 방법의 실제 trajectory/contact를 평가한다.

**Fig. E2는 핵심 Figure 후보이다.** 같은 장면에서 현재 ball / 관측 extent / oracle reference를 나란히 놓고, non-target 보존 영역·target 잔여 표면·query 오류·실제 접촉을 연결한다. 평균적인 예시와 대표 실패 예시를 포함하며, 단일 유리한 장면을 정량 결과 대신 쓰지 않는다.

## 3. 방향 1 — Attention과 Sub-task의 상호보완성

### 3.1 방법론 정의

Attention은 **현재 시각적 후보가 어디에 있는지**, depth는 **관측된 후보의 3D 범위**, sub-task와 grasp state는 **그 후보를 언제 어떤 접촉 역할로 사용할지**를 제공한다. Attention을 완전한 segmentation mask 또는 지속적인 manipulated-object identity로 취급하지 않는다.

Request `n`의 attention `A_n`, depth `D_n`, hidden feature `H_n`, 물리 grasp state `g_n`으로 다음을 구성한다.

```text
candidate geometry G_n = Ground(A_n, D_n)
phase belief p_n       = Probe(H_n)
target state S_n       = Update(S_{n-1}, G_n, p_n, g_n, execution feedback)
constraint set C_n     = Build(scene geometry, S_n, link-specific contact rules)
```

`S_n`은 manipulated-object identity, 관측 기하, attached geometry, 접촉 권한과 유효성 정보를 보관한다. **운반 중 attention이 사과에서 바구니로 이동해도 들고 있는 사과의 identity를 바꾸지 않는다.** 목적지 후보가 보였다는 이유만으로 바구니 전체를 obstacle에서 제거하지 않는다.

| 물리적 구간 | Attention / geometry 사용 | Phase의 역할 | 접촉 제약 |
|---|---|---|---|
| Approach | 조작 대상 후보 선택·관측 기하 갱신 | 새 manipulated target 채택 허용 여부 | 명시된 contact links와 지역에만 target 예외; 다른 링크·장애물 유지 |
| Grasp / closing | 기존 identity와 pre-grasp geometry 유지 | 성급한 target 변경 억제 | Grasp latch가 attachment 전환 판정 |
| Transport | Attached geometry를 손과 함께 이동 | Attention focus가 달라져도 held identity 유지 | 로봇 및 held object와 주변 환경의 충돌 검사 |
| Put / release | 놓기 전후 기하·권한 갱신 | 목적지 focus와 manipulated identity 구분 | 필요한 object–support contact만 별도 정의; 광범위한 destination 제거 금지 |
| Retreat / home | 완료된 identity와 접촉 예외 정리 | 물리적 release 이후 정리 시점 보조 | 놓인 물체를 다시 scene obstacle로 반영 |

표의 다섯 구간은 **동작 해석용**이다. 첫 구현에서 5-class classifier를 새로 만들지 않는다. 기존 `pick/place/home`과 grasp latch의 조합으로 표현한다. 목적지 grounding과 object–support contact의 새 모델링은 필요성이 확인될 때만 추가한다. 현재 시스템이 이미 모든 표의 동작을 구현했다고 서술하지 않는다.

### 3.2 구현

1. **Interface와 trace 통일.** `candidate_id`, `manipulated_id`, `phase_probs`, confirmed phase, grasp state, geometry version, allowed links, adopt/switch/release reason을 request별로 기록한다. Episode reset에서 모두 초기화한다.
2. **Phase source 교체 가능하게 구성.** `none / strong_rule / learned_probe / oracle`가 동일한 target-state update와 constraint builder를 사용하게 한다. `none`도 물리 grasp latch는 유지한다.
3. **Strong rule 구현.** 측정 gripper 상태, grasp/release feedback, 접근 가능한 거리·운동 정보로 상태를 판단한다. Threshold는 development split에서 고정하며 simulator 정답 identity나 task 성공 label을 입력으로 사용하지 않는다.
4. **불확실성 처리 고정.** 현재 probe가 연속 결측이면 ungated 동작으로 돌아간다. 제안 버전은 결측 시 새 예외의 확대를 막고 기존 identity를 물리 상태와 정해진 timeout으로 관리한다. 이 변경은 learned probe에만 유리한 별도 규칙이 되지 않도록 비교군에도 동일하게 적용한다.
5. **Geometry 검사.** Target 주변 제거 영역이 인접 obstacle까지 지우는지 확인한다. Target 선택이 맞는 것과 target extent가 정확한 것은 별개로 기록한다.

### 3.3 핵심 실험

모든 행에서 VLA checkpoint, depth, TO, link set, grasp latch와 evaluator를 동일하게 유지한다.

| 실험 | 비교 | 확인할 내용 | 필요한 Figure / Table |
|---|---|---|---|
| P1. Phase의 실행 기여 | Attention + none / strong rule / learned probe | 올바른 lifecycle 관리가 contact와 task 결과를 바꾸는가? | **Table P:** safe success, forbidden contact, task success, stop/timeout, 잘못된 target 전환 |
| P2. 두 신호의 결합 | Target source `{attention, strong geometry rule}` × phase source `{none, learned probe}` | Phase 효과가 attention의 잘못된 사용 시점을 줄이는가? | Table P에 필요한 행만 추가; full factorial 확장 불필요 |
| P3. 전환 장면 진단 | 같은 observation trace에서 phase source만 교체; 소수 oracle phase 참고 | Place 중 잘못된 새 target 채택, release 뒤 남은 예외, 전환 지연 | **Figure P:** 실제 RGB → attention → 3D target → phase/grasp timeline → 허용 링크·실제 contact |

Strong geometry rule은 외부 segmenter·정답 identity 없이 같은 depth와 robot state에서 후보를 선택하는 baseline으로 먼저 정의한다. 기존 ATS 대 external segmenter 실험은 재사용하고 여기서 새 비교군을 무제한 추가하지 않는다.

P2의 상호작용은 `phase 추가 효과(attention) − phase 추가 효과(geometry rule)`로 참고할 수 있다. 그러나 이 값이 유의하게 양수인 것만을 상호보완성의 정의로 삼지 않는다. 핵심은 **어떤 spatial 오류와 temporal 오류를 각각 줄였고, 그것이 실제 접촉·실행에 연결되는가**이다. Replay 분석은 lifecycle 진단이며 closed-loop success 근거로 대체하지 않는다.

### 3.4 검증 및 판단

- Phase accuracy 외에 transition delay, false transition, 잘못된 target 채택·변경, release 후 stale exemption을 확인한다.
- Occlusion, attention의 목적지 이동, probe 결측, grasp 실패·slip, episode reset에서 identity와 attachment가 일관되는지 검사한다.
- Probe split은 frame 무작위 분할 대신 episode 단위로 분리한다. 가능하면 held-out task에서도 확인한다. 새 checkpoint의 feature/head/probe는 development split에서 선택한다.
- Learned probe가 strong rule보다 실행을 개선하면 핵심 ablation으로 제시한다. 결과가 비슷하면 **phase-aware 처리의 필요성**을 주장하고 HSR은 구현 선택 또는 보조 결과로 둔다.

### 3.5 신호 불일치 시 exemption의 유지·축소·종료

`pick/place/home`는 HSR label이고 `approach/pre_grasp/grasp`는 별도의 contact policy 상태일 수 있다. 두 상태의 mapping과 physical latch 우선순위를 명시해, label 이름이 같아 보인다는 이유로 직접 치환하지 않는다. Strong rule과 learned probe 모두 아래 **동일한** conflict resolver를 사용한다.

| 입력 상황 | 결정 | 기록·검증 |
|---|---|---|
| Attention만 순간적으로 다른 물체를 가리킴; 기존 geometry 일관성 유지 | 기존 identity 유지, 새 물체로 exemption 확장 금지 | Temporal streak, candidate/held ID와 geometry overlap |
| Closing 중 target 일부가 가려짐 | Pre-closing anchor를 제한된 유효 기간 동안 유지; 실제 attachment는 물리 evidence로 결정 | 유지 시간, 새 관측과 anchor 차이, grasp 결과 |
| Attention·geometry가 지속적으로 충돌하고 grasp evidence 없음 | 신규 예외를 막고 유효한 관측 부분으로 축소; 만료 시 예외 종료. 필요한 경우 hold | 축소/만료 이유, held로 잘못 승격되지 않음 |
| 물리적으로 attached인데 attention/HSR은 다른 대상을 지시 | Held identity와 attached collision geometry 유지; attention으로 detach하지 않음 | Attach evidence, current FK, identity consistency |
| Slip/release가 물리적으로 확인됨 | Attachment 종료; 새 관측으로 world obstacle 반영. Contact exemption 정리 | Release 시간과 field 복구 시간, ghost geometry 여부 |
| Probe 결측·저신뢰 또는 observation 오래됨 | Phase만으로 권한을 확대하지 않음; 기존 예외의 유효기간과 물리 상태로 판단 | 입력 age·TTL, fallback 원인, stop 증가 |

TTL과 uncertainty threshold는 초·request 수 등 단위를 명시하고 development에서 고정한다. 예외 종료는 충돌 제약을 다시 활성화하는 것이므로, 로봇이 target와 이미 접촉 중이라면 다음 경로의 feasibility/hold를 검사한다. 만료만으로 실행 중인 chunk를 안전하게 중단할 수 있다고 가정하지 않는다.

**한계:** Temporal stability와 geometry consistency는 독립적인 semantic 정답 검사가 아니다. VLA가 지속적으로 잘못된 물체를 선택하면 두 조건을 모두 만족할 수 있다. 잘못된 attention을 replay에서 주입한 진단과 실제 wrong-target episode를 구분하고, 검출되지 않는 오류도 결과로 남긴다.

**기존 HSR 근거의 위치:** [V7G_FACTS.md](../../paper/notes/V7G_FACTS.md)는 미등록 비교에서 placement 이후 target switch/carve 변화와 gate on/off의 task 성공 차이 부재를 기록한다. 이는 E4의 출발 가설이며, strong rule 대비 task/contact 우수성의 근거가 아니다. 새 비교 전 해당 관찰을 본문의 검증된 성능 주장으로 승격하지 않는다.

## 4. 방향 2 — Denoising 내부 TO 또는 Guidance

### 4.1 공통 문제 정의

한 request에서 `C_n`을 만든 후 denoising 동안 고정한다. **Physical task phase와 denoising timestep은 서로 다른 변수**이다. ODE step마다 probe나 clustering을 다시 실행하지 않는다.

로봇 configuration `q`, collision sphere `j`의 위치 `p_j(q)`, 반경 `r_j`, clearance margin `m_j`에 대해 다음을 사용한다.

\[
h_j(q;C_n)=d_j(p_j(q);C_n)-r_j-m_j,
\qquad E(Q;C_n)=\sum_{k,j}w_j[\max(0,-h_j(q_k;C_n))]^2.
\]

`d_j`는 해당 phase·link에 맞는 distance query이며 attached-object sphere도 포함한다. `h_j >= 0`은 **현재 관측 기하와 검사 지점에 대한 margin 충족**이다. 현실의 모든 충돌 부재를 보장하는 표현으로 사용하지 않는다. Above energy는 비교를 위한 제안이며 OmniGuide 원문 cost의 복제가 아니다.

### 4.2 비교할 세 방식

| 방식 | 적용 위치 | 보정 연산 | 목적 |
|---|---|---|---|
| A. External TO | 모든 denoising 종료 후 | 현재 constrained trajectory optimization | 기준선 |
| B. Internal TO hook | 선택된 denoising step | Predicted clean chunk를 TO로 보정하고 latent에 반영 | 생성 도중 명시적 제약 보정의 효과 |
| C. Velocity guidance | 선택된 denoising step | Clean-action collision energy의 gradient로 velocity 수정 | 생성 방향을 연속적으로 유도하는 효과 |

**SEAM-inspired hook과 SEAM 원 방법을 구분한다.** SEAM은 이전 chunk의 미실행 tail에 대한 analytic consistency correction을 Euler update 뒤에 적용한다. 우리의 collision TO hook은 그 개입 위치를 참고하는 별도 확장이다. [SEAM 원문](https://arxiv.org/html/2607.04609v1)

**OmniGuide-inspired guidance와 재현을 구분한다.** OmniGuide는 predicted clean action을 통해 robot geometry의 energy를 평가하고 action에 대한 gradient로 생성 경로를 수정한다. Network Jacobian을 생략하거나 cost를 바꾸면 차이를 명시한다. [OmniGuide 원문](https://arxiv.org/html/2603.10052v1)

### 4.3 Clean-action 좌표와 update 정의

현재 sampler는 `t=1`이 noise, `t=0`이 clean이며 `Δt<0`이다. Local velocity를 `vθ(x_t,t)`로 쓰면:

\[
\hat{x}_0=x_t-t\,v_\theta(x_t,t),\qquad \bar{x}_{next}=x_t+\Delta t\,v_\theta(x_t,t).
\]

`x`는 normalized model action이다. `Q = Decode(x̂₀, observation)`로 실제 joint trajectory를 얻은 뒤 collision을 평가한다. **Noisy latent를 실제 joint trajectory처럼 검사하지 않는다.** `Decode`에는 normalization, delta/absolute 변환, active action dimensions를 포함한다. Gripper와 padding은 초기 비교에서 보정하지 않는다.

**B의 초기 구현안:** 같은 tracking·smoothness·continuity·collision 목적을 사용하는 TO로 `Q*`를 얻는다. 보정 대상 차원에서 `x*₀ = Encode(Q*, observation)`를 구성하고:

\[
x_{next}=\bar{x}_{next}+\beta(t_{next})(x^*_0-\hat{x}_0).
\]

초기 schedule 후보는 `β(t)=1−t`와 마지막 일부 step 적용이다. 이는 **설계할 heuristic**이며 SEAM 식의 재현이나 최종 feasibility 보장이 아니다. TO가 사용한 동일한 clean estimate를 기준으로 correction을 계산하고, 다음 step에서 policy velocity를 새로 평가한다. 보정량 상한과 solver 실패 시 처리를 명시한다.

**C의 초기 구현안:** `g_t = ∇_{x_t} E(Decode(x_t−t vθ(x_t,t)); C_n)`로 두고:

\[
x_{next}=x_t+\Delta t\,[v_\theta(x_t,t)+\lambda(t)\operatorname{clip}(g_t)].
\]

`Δt<0`이므로 **velocity에 `+gradient`를 더해야 실제 state update는 energy 감소 방향**이 된다. Noise→clean 시간이 증가하는 논문의 부호를 그대로 복사하지 않는다. 기본 C는 한 step의 clean estimate를 통한 full gradient로 정의한다. `stop_gradient(vθ)`를 사용한 근사는 `C-SG`로 따로 기록하고 동등한 방법이라고 부르지 않는다. VLA weights는 고정한다.

같은 위치에서 같은 correction `δx`를 적용한다면 post-Euler update와 velocity의 `δx/Δt` 추가는 대수적으로 같다. 따라서 B/C의 비교 대상은 hook의 이름이 아니라 **TO와 gradient라는 연산, clean estimate, gradient 경로, schedule, 계산 예산**이다.

### 4.4 구현 순서

1. **Constraint 생성 선행.** `observation/feedback → attention+phase 추출 → target/grasp state 갱신 → C_n 생성 → guided sampling → final check` 순서로 분리한다. 현재 scene 생성이 decoded chunk에 의존하는 부분을 점검하고, 관측 상태로 결정할 수 있는 처리부터 분리한다. 예측 action 때문에 실제 grasp state를 앞당겨 전환하지 않는다.
2. **공통 snapshot.** ESDF grid와 좌표계, query mask, link spheres, attached geometry, margin, joint limits, 현재 state와 유효한 이전 tail을 하나의 버전으로 전달한다. 최초 비교는 기존 별도 attention pass를 사용한다.
3. **좌표 adapter.** Physical joint ↔ model action 변환의 round-trip을 먼저 확인한다. C에서는 decode, FK, distance query가 action에 대해 미분 가능해야 한다. Discrete phase/clustering은 고정 입력으로 취급한다.
4. **B부터 구현.** 기존 post-Euler hook을 확장해 step의 `x_t/v_t/t/Δt`와 snapshot에 접근한다. CPU solver를 JAX `lax.while_loop`에 바로 넣을 수 없으므로 초기에는 step별 jitted model 호출을 host loop에서 orchestration하는 경로를 고려한다. 이때 같은 host loop의 A도 측정해 loop overhead를 분리한다.
5. **C 구현.** Device에서 실행 가능한 FK·distance interpolation/gradient를 연결한다. 기존 cuRobo field의 query 값·gradient와 일치성을 먼저 확인한다. Network backward, device transfer와 동기화를 latency에 포함한다.
6. **공통 final check와 trace.** A/B/C에 동일한 실행 전 검사·hold/fallback 정책을 적용한다. B/C에 추가 external TO를 붙이면 별도 hybrid variant로 표시한다. 첫 비교에서는 숨은 후처리 TO를 넣지 않는다.

Previous-tail continuity는 실제 실행 feedback으로 정렬한다. Phase/attachment 변화나 새로운 obstacle로 tail이 부적절해지면 해당 reference를 무효화하거나 재검사한다. 이전 model chunk와 이전 TO-corrected physical chunk 중 무엇을 사용하는지 모든 비교군에서 맞춘다.

### 4.5 핵심 실험

| 실험 | 조건 | 핵심 지표 | 필요한 Figure / Table |
|---|---|---|---|
| D1. Offline mechanism | 같은 observation, noise, `C_n`에서 A/B/C | Step별 clean prediction의 constraint violation, 보정량, task trajectory 변형, 처리 시간 | **Figure D:** clean trajectory가 denoising에 따라 변하는 과정과 violation curve |
| D2. Closed-loop 비교 | 같은 episode/seed 목록, 동일 checkpoint·제약 구성 | Task success, safe success, forbidden contact, stop/timeout, chunk boundary discontinuity, latency | **Table D:** A/B/C 성능–비용 비교 |
| D3. 제한된 schedule 선택 | Development split에서 마지막 step / 마지막 일부 / 모든 step 중 소수 비교 | 성능–latency trade-off, final gate rejection | Table D의 작은 보조 표 또는 appendix curve |

동일 denoising step 수의 비교와, 가능한 범위의 동일 wall-clock 예산 비교를 구분한다. TO iteration, guidance 호출·gradient 횟수, 추가 model evaluation, GPU memory를 기록한다. Multiple-noise candidate selection은 초기 비교에서 제외한다.

Offline에서는 snapshot과 initial noise를 정확히 공유한다. Closed-loop에서는 action 차이로 observation이 달라지므로 같은 초기 조건과 request seed schedule을 공유하며, 이후 observation까지 같다고 표현하지 않는다. Gripper를 보정하지 않아도 팔 경로가 달라져 grasp timing이 어긋날 수 있으므로 grasp 실패 유형도 기록한다.

### 4.6 검증 및 판단

- `hook=None`, `β=0`, `λ=0`에서 해당 실행 경로의 baseline과 출력이 사전 지정 tolerance 안에서 일치해야 한다.
- Decode/encode round-trip, dimension mask, time 부호, finite values를 검사한다. Known obstacle 장면에서 작은 guidance update가 energy를 낮추는지 확인한다.
- Smooth 영역에서 FK·distance·clean-estimate gradient를 finite difference와 비교한다. ESDF 경계의 비미분점은 별도 표시한다. B의 correction과 C의 gradient clipping이 크기를 제한하는지 확인한다.
- 장애물이 없을 때 원래 동작을 불필요하게 바꾸지 않는지, 좁은 통로·target 근접·운반·release에서 오작동하는지 확인한다.
- Gate 이전 raw constraint violation과 gate rejection/hold, 실제 실행 contact를 함께 보고한다. 정지 증가로 collision만 감소한 것을 성능 개선으로 판단하지 않는다.
- 내부 방식의 이득이 작거나 latency 부담이 크면 A를 본문 기본 방법으로 유지한다. B/C는 보조 분석으로 정리한다. 이득이 명확할 때만 채택한 방식을 main pipeline에 반영한다.

## 5. 공통 실험·검증 원칙

**Evaluation을 먼저 고정한다.** Benchmark 공식 contact metric은 그대로 보고한다. 별도로 robot-link–object–phase별 허용 contact 목록을 정의하고, 이 목록 밖의 contact를 forbidden contact로 채점한다. Evaluation의 정답 phase/object 정보는 점수 계산에만 쓰고 oracle 이외의 controller에 전달하지 않는다. 예측 phase가 틀렸다는 이유로 evaluator까지 접촉을 허용하면 안 된다.

`safe success`는 사전 정의한 성공 조건과 forbidden-contact 부재를 동시에 만족한 episode 비율이다. 어떤 링크·물체·시간 구간을 관측하고 채점했는지 명시한다. LTE와 global target 제거를 비교할 때 실제 검사 링크와 배치에서 두 방식의 차이가 생기는지 먼저 확인한다.

- Pilot은 소수 development scene으로 interface와 실패 유형을 확인하는 용도다. 최종 결론의 표본 수는 baseline 변동성과 남은 예산을 보고 test 실행 전에 고정한다. 결과를 보고 유리한 seed만 선택하지 않는다.
- 최종 비교는 같은 episode 목록을 사용한다. Episode 수·성공 건수·95% uncertainty interval을 보고하고 차이는 paired episode 단위로 분석한다. 동일 episode 내 frame을 독립 표본으로 세지 않는다.
- Warm-up/compile time과 steady-state latency를 분리한다. Device 완료를 동기화해 end-to-end median/p95를 측정하고, 서버 추론 시간과 실제 robot cycle 시간을 구분한다.
- Run마다 commit, checkpoint/probe/config hash, scene/seed, initial noise seed, constraint version, phase/target trace, raw/final chunk, execution feedback, contact event, timing을 연결한다. Denoising 전체 trace 저장은 진단 subset에 한정할 수 있다.
- Solver 실패, timeout, hold와 중단 episode를 분모에서 제거하지 않는다. 복구·재실행 규칙을 사전에 정한다.

## 6. 실행 우선순위와 논문 반영

| 순서 | 수행 범위 | 완료 기준 | 논문 반영 |
|---|---|---|---|
| 1 | Snapshot·evaluator·identity/seed/query trace 고정 | E1–E4의 원인 분리 가능 | Method의 problem/contact definition |
| **2** | **ATS→LTE 개선 및 E2 우선 진단** | 필요한 target 제거와 주변 obstacle 보존을 동시에 확인 | Fig. E2와 geometry/contact 표 |
| 3 | E1 attention, E3 link permission, E4/P1–P3 phase 비교 | 4개 논문 질문의 답과 claim 범위 결정 | 기존 main ablation·Figure 3에 통합 |
| 4 | Attached coverage·실제 CCR 검사 범위 확인 | 핵심 결과의 보호 범위·latency 설명 가능 | Method/runtime 및 appendix |
| 5 | 좌표 adapter + B, 이후 C offline prototype/D1 | Baseline·좌표·제약·gradient 검증, 비용 측정 | Figure D 후보 확보 |
| 6 | D2 및 필요한 D3만 실행 | A/B/C 채택 판단 가능 | Main runtime/ablation 또는 appendix의 Table D |
| 7 | 선택한 구성만 고정·집필 | 코드·표·그림·주장 일치 | Figure 2 pipeline 갱신, implementation/appendix 정리 |

Figure P/D와 Table P/D는 **자료의 용도 이름**이며 네 개를 모두 본문에 추가하라는 뜻이 아니다. 본문은 기존 Figure 1–3과 주요 표에 통합하고 상세 denoising trace·schedule·실패 사례는 appendix 또는 프로젝트 페이지에 배치한다.

논문의 우선 주장은 **내부 시각·작업 신호로 task-dependent contact constraints를 구성한다**는 것이다. Denoising hook은 이 제약을 적용하는 방법으로 먼저 설명한다. 단순히 hook 위치를 바꾼 것을 독립적인 큰 novelty로 내세우지 않는다. 최종 성능·비용·학습 범위의 표현은 위 검증 결과에 맞춰 결정한다.

## 7. 구현 시 확인할 파일

- [AttentionPolicy](../trajopt/attention_policy.py): 별도 attention pass, probe feature와 serving 결과 연결.
- [SubtaskProbe / Debounce](../ag3s/stages/subtask_probe.py): phase feature, label, 결측 처리.
- [Target grounding](../ag3s/stages/target_grounding.py): target 채택·변경·release 및 phase gate.
- [Attention split](../ag3s/stages/attention_split.py), [admissibility](../ag3s/stages/admissibility.py): 기존 merged-cluster 분리와 geometry consistency 관련 기능.
- [Runtime pipeline](../ag3s/runtime/pipeline.py): `_target_ball`, exclusion gate, attach snapshot 및 field 요청 조립.
- [cuRobo builder](../ag3s/fields/curobo_builder.py): sparse-point/ball seed 제거, ESDF 재생성 및 sign 처리.
- [TO adapter](../ag3s/constraints/to_adapter.py): link별 target-free mask 생성.
- [Grasp latch](../trajopt/grasp_latch.py): 물리 grasp/attachment 상태.
- [cuRobo field](../ag3s/fields/curobo_field.py): link별 distance query와 gradient.
- [Safe policy](../trajopt/safe_policy.py): inference·scene 생성·TO·execution feedback 순서.
- [TO problem](../trajopt/problem.py), [refiner](../trajopt/refiner.py): 목적함수·제약·previous-tail 계약.
- [Linearizer](../trajopt/linearize.py): distance/gradient, target-volume 분류와 margin 적용.
- [Sweep](../trajopt/sweep.py), [SQP](../trajopt/sqp.py): 중간 자세와 최종 trajectory 검사.
- [Held-object geometry](../ag3s/robot_models/held_object.py), [attachment](../ag3s/constraints/attached.py): fitted geometry, cap, 부착 좌표 변환.
- [Pi0 sampler](../../pi05_TO_hybrid/openpi/src/openpi/models/pi0.py): reverse-time flow와 기존 `guidance_fn`.
- [Policy transforms](../../pi05_TO_hybrid/openpi/src/openpi/policies/policy.py): normalized model action과 physical action 변환 경계.

위 파일들은 문서 작성 시 읽기 전용으로 확인했다. 이 문서의 계획은 아직 구현하거나 실험하지 않았다.

## 8. Claude Code 구현 인계 — 작업 단위와 완료 기준

**시작 지시:** 2.1–2.4절을 먼저 읽고 현재 active run의 resolved config와 source revision을 확인한다. 이 문서의 함수명은 2026-10-09 확인 기준이다. YAML default를 실제 실행 설정으로 가정하지 않는다. 기존 사용자 변경·기록·결과를 덮어쓰지 않는다. 이번 문서 작성 요청 자체가 GPU/MuJoCo 실행이나 코드 수정 완료를 뜻하지 않는다.

### 8.1 우선 작업 묶음

| 작업 | 주 수정 위치 | 구현 내용 | 완료 확인 |
|---|---|---|---|
| **I0: 오류 분리 기록** | `target_grounding`, runtime pipeline, builder stats, evaluator | Identity→source points→actual removed seeds→query→contact의 version 연결 | 하나의 contact event가 어느 seed 제거·권한 결정과 관련되는지 추적 가능 |
| **I1: 관측 extent 기반 exclusion** | Runtime `_target_ball` 호출부, `curobo_builder`, config | Ball baseline 보존, dense target evidence와 `observed_seed_mask` 대안; 보존된 obstacle·ambiguous seed 기록 | Correct identity에서 target 잔여/obstacle 손실 trade-off 확인; E2 offline 후 closed-loop |
| **I2: 권한 일관성** | `to_adapter`, `linearize`, `sweep`, `safe_policy` | Link permission과 spatial target-membership 분리; old ball 판정이 새 모드에 섞이지 않게 처리 | Same spatial point라도 authorized/non-contact link의 target 예외가 다름; non-target는 모두 검사 |
| **I3: HSR matched rule** | `subtask_probe`, `TargetConfirm`, `grasp_latch` 연결부 | Phase source adapter와 공통 conflict resolver; oracle은 eval mode만 | Grasp evidence·debounce·TTL을 맞춘 E4 비교, 상태 변화와 실행 결과 연결 |
| **I4: attached / CCR 범위** | `held_object`, `attached`, `sweep`, `sqp`, timing/evaluator | Coverage audit와 실제 검사 범위 기록; 필요한 부분만 수정 | 아래 8.3절의 coverage·path·latency 자료 확보 |
| **I5: denoising 비교** | Pi0 sampler, policy transforms, safe policy | 4절 B/C 구현; I1/I2에서 확정한 동일 geometry/permission 사용 | D1 후 D2, 출력·성능·실측 비용으로 채택 판단 |

각 묶음의 인계에는 변경 파일/함수, config delta, 실행 명령, baseline와 새 variant의 resolved config, test 결과, Figure/Table 위치, 남은 결함을 남긴다. 연구 실행 기록은 workspace의 기존 기록 규칙을 따른다. 본 문서에는 진행 상태와 artifact 링크만 추가하고 계획을 실험 사실처럼 바꾸지 않는다.

### 8.2 반드시 검증할 regression / invariant

기존 [target ball 테스트](../../tests/ag3s/test_target_ball_exclusion.py), [attention split 테스트](../../tests/ag3s/test_attention_split.py), [field policy 테스트](../../tests/trajopt/test_target_field_policy.py), [sweep/servo 테스트](../../tests/trajopt/test_t43q_servo_path.py)를 먼저 읽고 관련 검증을 확장한다. 기존 테스트만 통과했다고 새로운 방법이 검증된 것으로 처리하지 않는다.

| 상황 | 기대 동작 |
|---|---|
| Target와 distractor가 떨어져 있지만 하나의 ball 안에 들어감 | 새 exclusion에서 distractor seed 보존; 그 표면의 거리·gradient가 실제 query에 반영 |
| 두 물체가 같은 voxel 또는 depth 경계에서 구별되지 않음 | Ambiguous로 기록하고 일방적으로 target로 삭제하지 않음; 필요한 contact를 막는 비용도 측정 |
| Sparse target point / partial view | Dense evidence 또는 검증된 anchor를 사용; 고립 점만 지워 target surface가 거의 그대로 남는 regression 검출 |
| Table 접촉·fine window 안팎 | Support seed와 static obstacle 유지; main/target-free distance와 gradient가 동일 branch 사용 |
| 같은 geometry에서 authorized finger / non-contact forearm | Target 예외는 허용 링크만 적용; waypoint·sweep·final verdict에서 같은 권한 유지 |
| Ball→mask 모드 교체 | Legacy `target_volume` 및 serialized stats가 새 방법의 obstacle을 target로 재분류하지 않음 |
| Grasp attach / slip / release / reset | Identity·geometry version·field 제거·attached query가 모순되지 않음; perception 결측만으로 held object 삭제 금지 |
| 모든 새 switch OFF | 기존 baseline behavior 재현; baseline의 기존 한계는 별도로 기록 |

**피해야 할 수정:** Ball radius만 줄이고 해결했다고 결론 내리기, contact link의 collision row 전체 비활성화, `exclude_all`을 기본값으로 바꾸기, optimizer와 evaluator가 같은 잘못된 predicted target로 contact를 허용하기, hold/실패 episode를 평가에서 제외하기. CPU mock은 seed-index 논리 검증에 사용하고, 실제 cuRobo integration→seed→propagate→sign→query 경로는 저장된 scene replay 또는 승인된 실행에서 별도로 확인한다.

### 8.3 Attached geometry·CCR의 최소 검증

1. **Coverage:** Attach 순간 target observed points와 이동 중 held spheres를 같은 frame에 겹친다. `observed point coverage`, uncovered distance의 상위 분위수/최댓값, 과도한 부피를 기록한다. Simulator mesh reference의 coverage는 별도 열로 둔다. Visible points를 전부 덮었다고 unseen shape 전체를 덮었다고 주장하지 않는다.
2. **Cap 영향:** `held_cover_cap` 전후를 같은 object와 pose에서 비교한다. 원형 물체·길쭉한 물체·부분 관측 대표 사례만 우선 확인한다. Sphere를 무조건 키우거나 줄이지 말고, contact 누락과 불필요한 hold 중 어느 문제가 실제로 발생하는지 보고 수정한다.
3. **경로 범위:** Checked link 목록·sphere 수, 실행 prefix `K`, plan horizon `H`, 실제 collision horizon, waypoint stride, sweep/servo 활성 여부, `cap_bound_steps`, achieved `max_sweep_mm`를 저장한다. Sampling된 경로와 continuous-time collision 검사는 구분한다.
4. **실행과 시간:** Commanded path와 measured robot path의 편차, 관측 age, attention/field/TO/guidance 시간 및 p95 end-to-end latency를 기록한다. Stale snapshot을 실행한 시간과 hold/fallback을 함께 확인한다. Simulation이 server 응답을 기다려 정지했다면 그 시간을 포함해 설명한다.

산출물은 **한 장의 coverage/path overlay**와 **간결한 coverage·latency 표**를 기본으로 한다. 전면적인 robot model 재구축이나 전체 shape-completion 연구를 선행 조건으로 만들지 않는다.

## 9. 아이디어의 주장 범위와 Contact-rule learning 확장

### 9.1 이번 논문에서 끝까지 답할 질문

> VLA의 task relevance를 충돌 검사에 사용할 수 있는 공간적 제약으로 바꿀 때, 어떤 정보가 필요하며 어떤 변환이 잘못된 접촉을 막는가?

이를 `task identity → observed extent → obstacle-preserving exclusion → link permission → phase validity → executed contact`로 분해한다. Attention-only heatmap이나 phase accuracy에서 설명을 끝내지 않는다. **Semantic selection과 정확한 공간적 exclusion 사이의 연결**을 구현하고 E1–E4로 검증하는 것이 현재 가장 유망한 차별점이다.

External-segmenter-free 방식의 실질적 이득은 아래 중 **검증할 수 있는 한 가지 이상**을 선택해 제시한다.

| 후보 이득 | 필요한 비교 / 범위 |
|---|---|
| Segmentation 학습·annotation 의존성 감소 | 어떤 추가 mask annotation·segmenter adaptation을 제거했는지 명시. 기존 pretrained segmenter가 annotation 없이 동작하면 그 상대 이득을 과장하지 않음. HSR phase supervision은 별도 공개 |
| Segmentation이 불안정한 장면에서의 유용성 | 동일 RGB-D와 target 지시 정보로 적절한 외부 segmenter baseline을 실행하고 held-out occlusion/clutter에서 geometry와 contact 비교 |
| 충분한 geometry 품질을 더 적은 비용으로 제공 | 같은 hardware에서 추출용 model 사본·추가 pass·probe·depth/ESDF까지 포함한 incremental time/memory 비교. 품질이 다르면 성능–비용 trade-off로 보고 |

하나도 확인되지 않으면 `external-segmenter-free`는 방법의 구성 설명으로만 남긴다. 기존 VLA pretraining의 데이터 비용까지 사라진 것처럼 말하지 않는다.

### 9.2 VLA가 접촉 규칙을 이해하도록 하려면?

**현재 방법의 정확한 정의:** VLA에서 추정한 task target/phase를 사람이 정한 contact rule에 연결한다. 이것만으로 VLA가 접촉 규칙을 학습했다고 할 수 없다. 더 강한 방향은 상황에 따른 **contact relation 자체를 예측하고 실행에 활용하도록 학습**하는 것이다. 아래는 신규 연구 제안이며 기존 구현의 기능이나 검증된 성과가 아니다.

예측 단위는 단일 `safe/unsafe`가 아니라 다음과 같은 구조가 적합하다.

```text
ContactRelation = (
  object/region, robot link or held object,
  required | allowed | forbidden | unknown,
  activation event, termination event, confidence
)
```

예를 들어 손가락–사과는 grasp 동안 허용되지만 전완–사과는 금지되고, 운반 중 사과–바구니 rim은 금지되며, 올바른 placement의 사과–바구니 내부 support는 허용될 수 있다. 같은 object도 link·region·phase에 따라 다른 label을 갖도록 정의한다.

| 단계 | 구현·학습 | 무엇까지 주장할 수 있는가 |
|---|---|---|
| **L0: 현재 논문** | Frozen VLA attention/probe + explicit rule + 정확한 geometry 변환 | 내부 task 정보를 접촉 제약으로 연결 |
| **L1: 선택적 후속 실험** | Frozen VLA feature + object/region geometry + link embedding에서 contact-relation head만 학습; explicit rule 대비 비교 | Frozen feature에서 contact permission을 예측·활용 가능. VLA backbone 자체가 학습했다고 주장하지 않음 |
| **L2: 별도 연구 범위** | Adapter/LoRA 또는 VLA fine-tuning에 contact-relation loss와 action imitation loss를 결합; 필요 시 짝지은 선호/실패 trajectory supervision | 추가 학습된 정책의 contact-rule generalization을 검증할 수 있음. **현재의 no additional VLA training 주장과는 다른 설정** |

필요한 학습 데이터는 instruction·RGB-D·object/region·link·phase/event와 contact relation의 짝이다. 성공 demonstration의 접촉은 positive evidence이지만, **관측되지 않은 접촉을 모두 forbidden negative로 삼지 않는다.** 물리 contact 발생 여부도 normative permission과 다르다. Simulator contact labels에 task specification을 결합하고 필요한 부분을 검수하거나, 같은 장면의 허용/금지 trajectory 쌍을 구성한다. Training/evaluation의 simulator privilege와 annotation 비용을 명시한다.

Shortcut을 막으려면 같은 물체에서 link를 바꾼 사례, 같은 link에서 phase를 바꾼 사례, 물체 위치는 같고 instruction만 바꾼 사례를 포함한다. Test는 미관측 object–link–phase 조합 또는 새 task에서 수행한다. 같은 수동 rule을 그대로 label로 만들어 같은 조합에서 복원한 accuracy만으로 “이해”를 주장하지 않는다.

검증은 relation accuracy/calibration → incorrect permission → 실제 task/contact 결과 순서로 한다. Learned permission이 기존 explicit rule보다 어디서 일반화되는지 확인하고, `unknown` 처리와 geometry/grasp evidence 확인은 유지한다. 일반적인 geometry collision checking을 학습된 단일 분류기로 대체하지 않는다.

**남은 기간의 권고:** L0의 정확한 ATS→LTE 연결과 E1–E4를 먼저 완성한다. L1은 위 결과에서 explicit rule의 한계가 분명하고 데이터·시간이 남을 때만 진행한다. L2는 논문의 학습 설정과 contribution을 바꾸는 별도 연구 결정으로 둔다. Denoising guidance 자체는 학습이 없으면 VLA에 contact rule을 학습시키는 방법이 아니다.
