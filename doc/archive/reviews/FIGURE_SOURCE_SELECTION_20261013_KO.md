# Figure 1–3 실제 씬 자료 선정안

작성: 2026-10-13 KST. 상태: **후보 선정 / 그림 제작 전**.

사용자 지정 역할을 따른다: Figure 1 = demo teaser(감탄), Figure 2 = conceptual overview(납득), Figure 3 = full method pipeline(이해). 기존 계획의 Figure 2 pipeline / Figure 3 grounding 배치보다 이번 역할 분담이 우선한다. 이번 작업에서는 기존 이미지·sidecar·실험 기록·관련 구현을 읽었으며 새 이미지, crop, montage, 렌더링, 실험을 생성하지 않았다. 원고와 기존 figure 자료도 수정하지 않았다.

## 1. 선정 결론

**주 장면은 T53 C1a ep1842: 오른손으로 오렌지 옆의 배를 집어 바구니에 넣는 실행.** Figure 1은 이 실행의 자연색 장면, Figure 2는 대상과 이웃 물체 경계의 확대 장면, Figure 3은 동일 실행의 실제 중간 출력으로 연결한다. 세 그림에 같은 과제를 사용하되 질문·시점·확대 범위를 달리한다.

선정 이유는 배의 모양과 초록색이 잘 보이고, 오렌지와 매우 가까우며, 손–대상–이웃–목적지가 한 장면에 있기 때문이다. 1600×1200 재렌더 이미지와 paired server dump가 이미 있다. 연구 verify의 해당 episode는 `success=true`, `success_ct=true`, `failure_mode=ok`이며 HOLD는 1 chunk다. 이는 T53에서 정의한 contact 범위의 결과이며, 모든 충돌·전신 안전이나 다른 방법 대비 개선을 뜻하지 않는다.

| Figure | 독자가 알아야 할 한 문장 | 주 자료 | 권하는 시각 구성 |
|---|---|---|---|
| 1 | 인접 물체가 있는 장면에서도 지시 대상을 집어 옮기는 실행을 보여준다 | ep1842 grasp/carry/release, 필요시 작은 grounding inset | 큰 carry 장면 + 작은 grasp/완료 장면; 최소한의 annotation |
| 2 | 대상 위치를 아는 것과 접촉 예외의 경계를 결정하는 것은 다르다 | ep1842 chunk 17의 all-surface / ball / ownership, T46 경계 개선 진단 | 동일 viewpoint의 기하 처리 세 칸 + 작은 link permission 설명 |
| 3 | 정책의 내부 신호와 depth가 접촉 제약·행동 보정으로 이어진다 | RGB/attention, lifted cloud, target cluster, scene field, attached object, CCR 기록 | 모듈·입출력·화살표 중심, 실제 중간 출력은 작은 inset |

## 2. Figure 1 — 실행 자체를 전면에

권장 메시지: **Task-grounded contact in cluttered manipulation.** 과제가 좁은 pick-and-place라는 점은 caption에서 명시한다. 제목에 obstacle avoidance improvement나 collision-free를 먼저 걸지 않는다.

### 2.1 우선 사용할 이미지

| 역할 / 우선순위 | 실제 파일 | 시점 | 시각 판단과 사용법 |
|---|---|---|---|
| Hero / 1순위 | [f1_key_carry.png](../../../paper/figures/paperbanana/figure1/assets/f1_key_carry.png) | chunk 26, control step 208 | 가장 추천. 손가락 사이 배가 분명하고 로봇·바구니·남아 있는 오렌지를 함께 볼 수 있다. 나중에 편집할 때 바닥·왼쪽 선반 여백을 줄여 조작 영역을 키운다 |
| 근접 grasp / 1순위 | [f1_key_grasp.png](../../../paper/figures/paperbanana/figure1/assets/f1_key_grasp.png) | chunk 20, step 160 | 배와 오렌지의 가까운 배치 및 필요한 손가락 접촉을 보여준다. Carry와 같은 크기의 반복 장면보다 작은 시작 inset으로 적합 |
| 완료 / 조건부 | [f1_key_release.png](../../../paper/figures/paperbanana/figure1/assets/f1_key_release.png) | chunk 34, step 272 | 손이 열린 것은 보이나 바구니 앞벽 때문에 배가 보이지 않는다. 완료를 직관적으로 보여주려면 후속 제작 때 같은 기록 상태를 위쪽 viewpoint에서 재렌더하는 편이 낫다 |
| 내부 신호→기하 / 선택 | [f1_grounding_scene.png](../../../paper/figures/paperbanana/figure1/assets/f1_grounding_scene.png) | chunk 17 | 대상과 이웃의 대비가 명확하다. 작은 inset 하나로만 사용. 파란 이웃 색은 GT centre를 이용한 표시이며 모델의 instance prediction이 아님 |
| 접촉 허용 / 대안 | [f1_contact_scene.png](../../../paper/figures/paperbanana/figure1/assets/f1_contact_scene.png) | chunk 18 | 금색 손과 target cluster가 잘 보인다. 큰 점 glyph가 실제 과일을 가리므로 자연색 hero를 대체하기보다는 Fig. 2 설명용 후보 |
| Attention / 후순위 | [f1_wrist_attention.png](../../../paper/figures/paperbanana/figure1/assets/f1_wrist_attention.png) | chunk 10 | 밝은 peak가 두 과일 경계 부근에 있고 배 하단이 잘린다. “정확한 object mask”의 대표 이미지로 부적합. Fig. 3의 seed cue 설명에 더 적합 |

권장 배치 비율은 hero 약 55–65%, 나머지는 grasp·완료·작은 기하 inset이다. 4단계 장면을 같은 크기로 나열하면 로봇 전신의 반복 때문에 조작 대상이 작아진다. Scene은 자연색, 관측/추정 overlay만 일정한 색으로 표시한다. 모듈 이름·수식·latency·ablation 표는 Figure 1에 넣지 않는다.

### 2.2 대안 episode와 제외할 비교

[T53 initial layout](../../../AG3S/ag3s/docs/figures/t53/t53-initial-layout.png)와 [T53 verify](../../../AG3S/ag3s/docs/handoff/T53.verify.json)를 대조했다.

- **ep1845, 왼손 배:** task/contact 기준 성공, HOLD 0. 반대 손의 시연이 필요할 때 대안. 현재 이번 검토는 initial layout까지이며 최종 carry/grasp frame의 미관 우위는 확인하지 않았다.
- **ep1809, 왼손 바나나:** task/contact 기준 성공, HOLD 0. 길쭉한 물체 시연용 후보. 원본 영상은 verify의 `per_episode.run_dir`에 있다. 개별 keyframe 선정은 후속 작업이다.
- **ep1860 또는 ep1866, 오렌지:** 둘 다 해당 task/contact 기준 성공. 물체 다양성의 작은 보조 컷 후보이며 첫 세 Figure에 모두 넣을 필요는 없다.
- **ep1862, 사과:** task는 성공했지만 `success_ct=false`이고 이웃 과일 접촉이 기록되어 있어 무접촉 성공 teaser 후보에서 제외한다.
- ep1842에 현재 설정과 짝지어진 VLA-only 결과는 확인되지 않았다. T39의 과거 clear-scene baseline 또는 다른 episode 실패를 붙여 현재 방법의 before/after로 만들지 않는다.
- 옛 `cvprfig_assets/demo_*.png`는 설정 미확인 `ep1807_apple.mp4` stand-in에 기반하므로 현재 T53 자료보다 후순위다.

좋은 성공 사례를 teaser에 선택하되, 최종 결과 표의 전체 평가 분모는 그대로 유지한다. 이번 선정은 “시각적으로 이해하기 좋은 정성 예시”이며 전체 성능의 대표 통계가 아니다.

## 3. Figure 2 — 같은 장면에서 제약의 차이를 납득시키기

권장 메시지: **Localizing the target does not determine where contact may be allowed.** 큰 주제는 공간적 extent이며, link specificity는 작은 별도 설명으로 분리한다. Ball→ownership 비교에서 extent와 link policy를 동시에 바꾸고 한 요인의 효과처럼 말하지 않는다.

### 3.1 주 패널

모두 ep1842 chunk 17(마지막 pre-grasp 부근)의 같은 viewpoint를 사용한다.

| 패널 | 파일 | 전달할 내용 | 자료 성격 |
|---|---|---|---|
| A. 모든 관측 표면 | [f2a_scene_render.png](../../../paper/figures/paperbanana/figure2/assets/f2a_scene_render.png) | 대상도 충돌 기하로 남으면 필요한 접촉을 제한할 수 있음 | 기록된 observed cloud를 1 cm voxel로 표시한 개념 시각화. 실제 controller가 멈춘 before run은 아님 |
| B. 구형 예외 | [f2b_ball_render.png](../../../paper/figures/paperbanana/figure2/assets/f2b_ball_render.png) | 하나의 ball이 target과 이웃 표면을 함께 포함 | 기록 ball과 observed cloud에 따른 시각화. 모든 link에서 제외하는 global policy는 extent 축과 구분 |
| C. 경계 보존 — 원 실행 일치 | [f2c_ownership_asrun_proximity.png](../../../paper/figures/paperbanana/figure2/assets/f2c_ownership_asrun_proximity.png) | 이웃을 남기는 ownership 처리와 여전히 남는 ambiguous target 표면 | T53 C1a의 proximity 설정에 맞춘 동일 chunk의 offline 1 cm surface-grid 계산 |
| C의 최신 개선 대안 | [f2c_ownership_render.png](../../../paper/figures/paperbanana/figure2/assets/f2c_ownership_render.png) | F1 surface evidence 적용으로 대상 표면의 불필요한 잔류를 줄이는 원리 | 같은 기록에 F1을 **오프라인 적용**. T53 C1a가 실제 실행한 설정은 아님 |
| 작은 link 설명 | [f2c_link_render.png](../../../paper/figures/paperbanana/figure2/assets/f2c_link_render.png) | Active-hand contact links와 다른 query links가 다른 field를 사용 | 기록 permission 기반. 현재 query scope는 양손 gripper이며 wrist/forearm 보호 결과가 아님 |

**권장 선택:** 최신 개선을 적극 설명하려면 C에 F1 이미지를 쓰고 `F1 applied offline to the recorded scene`으로 구분한다. Figure 1과 완전히 같은 실행 설정으로 통일하려면 proximity 이미지를 주 패널로 쓰고 F1은 별도 작은 개선 inset에 둔다. 두 이미지를 같은 결과라고 섞지 않는다.

이 그림의 1 cm display voxel은 서버 TSDF seed의 원본 5 mm lattice와 같지 않다. “실제 seed가 정확히 이렇게 삭제됐다”는 그림이 필요하면 후속 제작에서 실제 field/seed 기록으로 대체한다. 기존 그림은 geometry 처리 원리 설명으로 사용할 수 있다.

최종 시각 편집에서는 ball과 이웃이 겹치는 작은 경계 영역을 확대하고 유지/제외되는 표면을 대비시킨다. 전체 voxel을 강하게 그리면 pear/orange 모양이 사라지므로 mesh의 가독성과 overlay 밀도를 함께 조절한다. 지금은 편집을 수행하지 않았다.

### 3.2 최신 개선 이미지의 적극적인 활용

| 기존 연구 이미지 | 추천 사용 위치 | 실제로 유용한 부분 / 해석 범위 |
|---|---|---|
| [T46 scene](../../../AG3S/ag3s/docs/figures/t46/t46-scene.png) | Figure 2의 개선 근거 inset, 또는 supplementary | 첫 행 `ep104_s1042:14`: off→F1에서 이웃 표면을 유지하면서 target 잔류와 constraint row가 줄어드는 동일 상태 비교가 선명함. Offline rows이며 physical contact 횟수가 아님 |
| [T43 DX finger scene](../../../AG3S/ag3s/docs/figures/t43/t43dx-4-finger-scene.png) | Figure 2 문제 제기용 대안 | 실제 bar contact 장면과 full/target-free ESDF에서 bar가 사라지는 현상이 직관적. 과거 snapshot 실패를 현재 개선이 같은 run에서 해결했다고 연결하지 않음 |
| [T45 target-free scene](../../../AG3S/ag3s/docs/figures/t45/t45tf-scene.png) | Figure 2 설계 참고·부록 | Target 잔류의 원인을 분해한 진단. 아름다운 대표 씬보다는 technical failure 설명에 적합 |
| [T46 oracle scene](../../../AG3S/ag3s/docs/figures/t46/t46e-d-scene.png) | 부록·error analysis | GT visible extent를 넣어도 남는 이웃/table/robot 관련 제약을 보여줌. Oracle을 제안 방법으로 표시하지 않음 |
| [T46 CL3 scene](../../../AG3S/ag3s/docs/figures/t46/t46-cl3-scene.png) | 부록·정성 비교 후보 | Paired C1a/F1의 성공과 실패가 함께 있음. 본문 hero로 넣기에는 HUD·작은 대상·반복 장면이 많고 일관된 우위도 아님 |

T46 CL3의 task success는 C1a/C1aF1/C1aF1F2 각각 14/13/13 of 18로 기록되어 있다. 경계 처리의 closed-loop 개선이 입증됐다는 결론은 이 자료에서 나오지 않는다. 이 수치는 후보 선정의 해석에만 사용했으며 paper result로 새로 등록하지 않았다.

## 4. Figure 3 — 실제 출력으로 pipeline을 설명하기

권장 구성은 **정책 action 경로**, **attention/depth geometry 경로**, **phase/grasp 상태 경로**를 구분해 마지막 contact constraints와 CCR에서 합치는 것이다. Pipeline 도형은 후속 제작에서 구성하며 이번에는 아래 원본을 지정한다.

| 구조상 위치 | 사용할 실제 출력 | 시점·주의 |
|---|---|---|
| RGB 입력 | [head](../../../paper/figures/paperbanana/figure3/assets/f3_in_head.png), [left wrist](../../../paper/figures/paperbanana/figure3/assets/f3_in_left_wrist.png), [right wrist](../../../paper/figures/paperbanana/figure3/assets/f3_in_right_wrist.png) | chunk 10의 카메라 재렌더. 정책에 실제 입력된 224² pixels 자체는 아니므로 고해상도 보기용으로 구분 |
| Attention cue | [right wrist attention](../../../paper/figures/paperbanana/figure3/assets/f3_att_right_wrist.png), 기존 [policy inputs/attention diagnostic](../../../paper/figures/paperbanana/figure3/assets/diagnostic/f3_policy_inputs_attention.png) | L8H2. 한 개 overlay + 작은 다중 view 표기로 충분. 최종 사용 전 preprocessing/crop/padding에 따른 overlay 좌표 확인 |
| Multi-view lifting / seeds | [f3_lifted_attention.png](../../../paper/figures/paperbanana/figure3/assets/f3_lifted_attention.png) | chunk 17. Teal seeds가 이웃·basket에도 존재하므로 정밀 segmentation 결과가 아닌 cue 단계로 표시 |
| Target geometry | [f3_target_cluster.png](../../../paper/figures/paperbanana/figure3/assets/f3_target_cluster.png) | 같은 chunk 17. Lifted cloud와 동일 viewpoint로 전후 연결하기 좋음 |
| Scene geometry | [f3_scene_field.png](../../../paper/figures/paperbanana/figure3/assets/f3_scene_field.png) | 기록된 fine ESDF의 near-surface voxel. 현재 그림은 밀도가 높으므로 작은 thumbnail은 과밀함; 최종 제작 때 표시 밀도/범위 조절 |
| Contact queries | Fig. 2의 ownership/link 자료를 작은 요약 inset으로 연결 | Fig. 2의 세 패널 비교를 그대로 반복하지 않음 |
| Attached geometry | [f3_attached_object.png](../../../paper/figures/paperbanana/figure3/assets/f3_attached_object.png) | chunk 19 attach sphere를 FK로 chunk 26 carry pose에 옮긴 재구성. 새 추정 결과나 true-object 전체 coverage가 아님 |
| CCR | [f3_ccr_paths.png](../../../paper/figures/paperbanana/figure3/assets/f3_ccr_paths.png), [기존 joint diagnostic](../../../paper/figures/paperbanana/figure3/assets/diagnostic/f3_ccr_refinement.png) | chunk 19. 현재 3D 이미지에서 두 TCP 경로가 잘 구분되지 않아 그대로 작게 넣기에는 부적합. 최종 제작 때 projection/선 굵기를 개선하거나 joint inset 사용 |
| State / lifecycle | [state timeline](../../../paper/figures/paperbanana/figure3/assets/diagnostic/f3_state_timeline.png) | 전 75 chunks. 본문에는 작은 lifecycle strip만 고려하고 probability plot 전체는 부록에 적합 |

Figure 3에서 보존할 실제 구현 관계:

1. Frozen VLA의 action serving과 attention extraction은 별도 pass다. 현재 extraction은 L8H2이며 최근 noun/head 실험의 우수 후보를 serving 경로에 이미 적용했다고 그리지 않는다.
2. Depth + calibration + capture-time pose가 3D geometry를 만든다. Low-attention surface가 scene map에서 모두 지워지는 화살표를 그리지 않는다.
3. HSR은 frozen feature의 학습된 probe로 target adoption/switching 등을 제어한다. Grasp state machine은 실행 feedback/측정 opening/geometry로 attachment를 관리한다. 둘의 화살표를 합치지 않는다.
4. HSR의 pick/place/home과 client의 `safe-phase=approach`는 다른 변수다. 후자는 T53에서 주입된 상수이며 학습된 phase output으로 표시하지 않는다.
5. T53 C1a는 ball-owned + proximity evidence + exclude-authorized + gripper query다. 최신 F1 surface evidence를 pipeline의 선택 가능한 개선으로 표시하려면 해당 flag와 offline/별도 arm 사용을 구분한다.
6. 평가 arm의 CCR는 16 planned rows(8 executed + 8 look-ahead)를 사용한다. 원고의 과거 K=8-only 설명을 그대로 복사하지 않는다. Gripper command 경로는 arm-joint refinement와 분리한다.
7. Refined/reference TCP는 예정 command의 FK path다. 실제 measured execution trajectory 또는 장애물 회피 성공 궤적으로 이름 붙이지 않는다. `f3_ccr_paths`는 해당 run에서 수정량이 가장 큰 chunk를 고른 자료다.

## 5. 최근 T47–T56 자료의 배치

| 연구 자료 | Figure 1–3에서의 역할 | 권고 |
|---|---|---|
| T47–T49 attention/head/noun 비교 | Attention이 완성된 mask가 아니라는 설명·head 선택의 근거 | 메인 pipeline의 attention inset을 고를 때 참고; detailed heatmap/후보 표는 부록 |
| [T49b scene variants](../../../AG3S/ag3s/docs/figures/t49/t49b-scene-variants.png) | 같은 관측에서 query/head별 grounding 차이 | exploratory 결과. 새 noun-query branch가 현 serving인 것처럼 Fig. 3에 넣지 않음 |
| [T51 self-label scene](../../../AG3S/ag3s/docs/figures/t51/t51-selflabel-scene.png) | Grasp event를 통한 label 구성 이해 | calibration/selection 부록에 적합. GT segmentation 색을 runtime segmentation으로 오해시키지 않음 |
| [T52 scene1800 stages](../../../AG3S/ag3s/docs/figures/t52/t52-scene1800-stages.png) | 2D target peak와 3D cluster purity가 다를 수 있음을 설명 | Figure 2의 작은 identity-versus-extent 보조 사례 후보. 전체 9행을 본문에 넣기에는 과밀함 |
| T53 | 최신 완료된 rollout과 중간 출력 | 이번 세 Figure의 주 원자료 |
| T54 | 최신 replay 비교 | 검토 당시 `verify.status=in_progress`. 최종 결과가 확인되면 후보 교체를 검토 |
| T55/T56 | Empty/slip 진단, feature/self-score 구현 | 구현 문서는 있으나 이번 확인에서 최종 결과·완성 figure를 확보하지 못함. 성능 개선 예시로 선반영하지 않음 |
| T46f wrist geometry | 향후 link-specific 대비 장면 | 별도 worktree 검증 단계로 확인. 현재 T53이 wrist/forearm을 보호했다고 그리지 않음 |

**T52를 넣는다면 가장 쓸 만한 부분:** `headfix f4`의 동일 관측에서 L8H2와 noun-query 결과를 비교하는 첫 세 행. 2D cue가 강해 보여도 target과 이웃이 같은 3D cluster로 합쳐질 수 있다는 문제를 설명한다. 행의 `purity`와 source는 해당 JSON/verify로 추적하며, 정답 여부와 전체 성능 우위를 혼동하지 않는다.

옛 hurdle 장면 중 [T43 V7g](../../../AG3S/ag3s/docs/figures/t43/t43v7g-scene.png)의 ep1925/s19253, t=216은 attached-object 크기와 basket entry의 관계를 잘 보여준다. 현재 T53과 다른 snapshot이므로 Fig. 1의 동일 실행 strip에 혼합하지 않고 Fig. 3/부록의 역사적 기하 진단에 두는 편이 낫다.

## 6. 기존 자료 패키지에서 발견한 불일치와 후속 편집 포인트

- `paperbanana/figure*/reference_montage.png`는 초기 diagnostic montage다. 개별 1600×1200 scene render가 이후 추가되어 현재 개별 PNG·JSON이 더 풍부하다. Montage만 보고 후보를 결정하지 않는다.
- `figure1/evidence.md`의 video frame 84/160/200/268과 새 scene render의 chunk 10/20/26/34(step 80/160/208/272)는 서로 다른 산출물이다. 최종 caption은 선택한 **개별 sidecar**를 따른다.
- `figure2/evidence.md`는 proximity 설명 중심이지만 `f2c_ownership_render.json`은 F1 offline 적용을 명시한다. 더 깔끔한 그림이라는 이유로 실제 T53 설정으로 바꿔 설명하지 않는다.
- `figure3/evidence.md`는 attach 시점 그림만 설명하지만 새 `f3_attached_object.json`은 attach→carry FK 이동을 명시한다. 후속 편집 때 자료 설명을 동기화한다.
- `f1_wrist_attention`은 16×16을 4:3 재렌더에 선형 대응시킨 표시다. 실제 policy resize/crop과 camera FOV parity는 이번에 계산 검증하지 않았다. 논문에서 픽셀 단위 정확성을 강조하려면 원 관측/전처리 변환을 대조한다.
- `f1_key_release`는 성공 종료 상태를 담지만 물체가 가려져 있다. 같은 기록 상태의 viewpoint만 바꿔 배가 바구니에 놓인 것을 보여주는 장면이 후속 제작의 첫 보완 대상이다.
- `f3_ccr_paths`는 수정량의 근거는 있지만 현재 렌더에서는 경로가 가려져 있다. 동작을 바꾸거나 보정량을 과장하지 않고 viewpoint/표시 방법만 개선한다.
- `f3_attached_object`의 구는 배 하단을 완전히 덮지 않는다. 관측 기하 근사로 표시하고 mesh 전체 coverage를 암시하지 않는다.

## 7. 원자료와 확인 범위

- 주 원자료: `/mnt/dev/AG3S_VLA-storage/results/development/20261013T100210+0900_4a540f1a18d7_cb037c425d2f/verify/T53/runs/C1a/R/C1a_ep1842_s18421/`.
- Client: `ag3s/run_0000/step_*.npz`, `third_person_front.mp4`. Server의 ep1842 local chunk 17/19는 sidecar에서 server chunk 00167/00169에 연결되어 있다. 서로 다른 index를 혼동하지 않는다.
- [T53 verify](../../../AG3S/ag3s/docs/handoff/T53.verify.json), [CL3 verify](../../../AG3S/ag3s/docs/handoff/T46-CL3.verify.json), [T52 verify](../../../AG3S/ag3s/docs/handoff/T52.verify.json), [T54 상태](../../../AG3S/ag3s/docs/handoff/T54.verify.json)를 읽었다.
- 기존 27개 PNG를 직접 열어 확인했다. Episode 대안은 T53 initial-layout montage와 결과표 기준이며 모든 영상 프레임을 전수 검토한 것은 아니다.
- 주요 `paperbanana` 개별 JSON의 source·chunk·render note를 읽고 `_ball_owned_split` 및 render script의 관련 구현과 대조했다. 이 문서는 전체 source hash 재감사, 실험 재계산, paper evidence 등록 또는 submission readiness 판정이 아니다.
- 현재 후보는 **기록된 시뮬레이션 상태의 재렌더 / 실제 관측·출력 overlay / offline 비교 시각화**다. 실물 로봇 사진으로 부르지 않는다. 단순 display 보간을 모델의 고해상도 출력으로 부르지 않는다.

다음 제작 단계에서 먼저 고정할 것은 Figure 2의 **T53 as-run 중심 또는 F1 offline 개선 중심**이라는 설명 범위다. 이번 추천은 Figure 1의 T53 성공 데모, Figure 2의 명시적으로 구분한 최신 F1 메커니즘, Figure 3의 T53 pipeline과 선택적 F1 branch 조합이다. 이를 고정한 뒤 release viewpoint·CCR 가독성·overlay 좌표만 보완하면 된다.
