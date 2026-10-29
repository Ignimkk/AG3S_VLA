**AG3S 내부 feature + RGB-D 실험 실행 명세 v1.2**

작성: 2026-10-15 KST. 상태: **실행할 방법론 명세이며, 완료된 대규모 실험 결과가 아니다.** 다른 agent는 이 문서를 실행 정본으로 사용한다. [기계 판독 설정](experiment_spec.json), [덤프 저장 명세](../../../archive/reviews/vla_dump_20261015/DUMP_POLICY_KO.md), [기존 단일 프레임 결과](../../../archive/reviews/vla_dump_20261015/DEPTH_GROUNDING_REVIEW_KO.md)를 함께 읽는다. 이전 [연구 프롬프트](../../../archive/reviews/AG3S_VLA_Internal_Feature_Grounding_Prompt.md)의 A 성공 후에만 C를 진행한다는 순서는 이 문서의 병렬 비교 설계로 대체한다.

**0. 목표·범위·완료 조건**

목표 S: frozen π0.5 내부 특징과 관측 RGB-D로 조작 대상의 **identity와 visible surface extent**를 추정하고, target 주변 다른 물체를 잘못 포함하는 오류를 줄인다.

목표 P: 기존 HSR/KV에 3D 관계·상대 운동·gripper 관측을 추가했을 때 현재 phase와 attachment 추정이 향상되는지 검증한다. 작업 단계와 물리 attachment를 같은 label로 합치지 않는다.

VLA weight와 baseline optimizer를 고정한다. 외부 segmenter·mask-supervised weight training을 제안 방법의 필수 구성으로 사용하지 않는다. dev GT로 hyperparameter를 선택하는 것은 허용하되 이를 명시하며 완전 label-free 선택이라고 주장하지 않는다. Phase probe를 별도 학습하는 실험은 명시적 supervised ablation으로 표시한다. 초기 P baseline은 기존 probe+물리 rule+시간 filter다.

완료 조건: provenance/정합 검사 → 사전 고정 split/config → offline 비교 → 실패 원인 분리 → 결과표/개별 episode 결과/실행 비용/다음 결정이 모두 산출되어야 한다. 성능이 나쁘면 최초 비교 결과를 보존하고 §13.1의 원인 진단·근거 있는 개선·재실험을 수행한다. 그 후 개선 미확인과 남은 한계를 보고하는 것도 완료 결과다. 최초 실패 한 번만 보고 전체 작업을 종료하지 않는다. 성능이 좋아 보인다는 이유로 serving default를 변경하지 않는다. 이 명세는 실제 로봇 실행이나 운영 배포 요청이 아니다.

**0.1 T43–T61 감사 반영: 기존 방법과 추가 검증을 구분한다**

먼저 [기존 실험 대조 보고서](../../../archive/reviews/vla_dump_20261015/HISTORICAL_PROTOCOL_AUDIT_KO.md)를 읽는다. v1에서 T55/T56/T61 재사용을 충분히 명시하지 못한 부분을 정정한다. T54는 action/noun 역할별 융합, T55는 strong-rule+HSR/noun+시간 지속성, T56은 손 좌표계 co-motion self-score, T61은 head 선택과 held-out 평가를 이미 수행했다. 이들 자체를 새 방법으로 제안하지 않는다.

주 추가 검증은 **내부 feature prototype + depth surface 경계를 보존한 graph가 기존 geometry extent보다 나은가**다. 개발 계획 §0.10에 따라 이 경계 방법 하나를 먼저 검증하고 P 확장은 보조로 둔다. 기존 serving region/ball 또는 ball_owned 경로와 T54 N/B를 재사용 baseline으로 포함한다. 같은 identity/source에서 extent 효과를 분리한다.

T61 `action:mean:L10H7`은 추가 고정 source control이다(denoise 0, action query 평균). `noun_target:L10H7`과 다른 map이다. 기본 L8H2를 대체하지 않으며, T61의 주 P 창 episode CI는 0을 포함했다. 새 source를 선택했다는 사실만으로 extent 개선을 주장하지 않는다. T61 offline replay와 새 graph의 voxel/설정도 맞춰 비교한다.

T55의 task-ready 이진 판별과 pick/place/home 3-class phase는 별도 평가다. T56 S2는 HELD 조건부 사후 self-label이므로 독립 hold onset detector가 아니다. P4는 그 좌표·거리 primitive를 재사용하되 world-static/hand-static 비교 및 online onset 평가를 추가하는 실험으로 정의한다.

**현재 분담:** 새 프레임 dump 수집은 다른 agent가 진행 중이다(사용자 확인). 분석 agent는 별도 수집을 시작하지 않고 수집 산출물의 manifest/schema를 받아 reader·정합 검사·실험에 연결한다. 아래 collector 구현 지침은 실제 누락/불일치가 확인될 때의 기술 명세이며 중복 수집 지시가 아니다. 산출물을 기다리는 동안 기존 frame 재현과 graph/evaluator 구현을 진행한다.

**1. 다음 agent의 첫 작업 순서**

1. 현재 repository의 AGENTS.md가 있으면 확인하고 `git status`를 기록한다. 기존 WIP를 덮어쓰지 않는다. 원본 dump·checkpoint·GT는 읽기 전용으로 둔다.
2. 아래 “기존 구현 재사용” 목록을 읽고 기능·지문·입력 shape를 확인한다. compact schema/CLI가 이미 구현되었다고 가정하지 않는다.
3. `experiment_spec.json`을 새 run에 복사하여 source hash, 실제 episode split, head/query registry를 채운다. 아직 비어 있는 fields는 pending으로 남긴다.
4. pilot으로 기존 frame022를 재현한다. 이 frame은 **debug_regression**이며 최종 성능 평가에 포함하지 않는다.
5. compact collector/reader를 구현할 경우 `DUMP_POLICY_KO.md`의 source-of-truth 규칙과 full-vs-compact parity부터 통과시킨다.
6. dev split에서 S0–S3의 extent 비교를 우선 진행하고, 실패하면 §13.1의 진단·개선 반복을 수행한다. S4와 P0–P4는 확인된 병목에 따라 후속으로 진행한다. 공간 feature-only 개선 유무와 관계없이 geometry-only 및 결합 비교를 수행한다.
7. config와 test split을 잠그고 held-out replay 평가를 한 번 수행한다. 수정 후 같은 test를 다시 보면 그 set은 더 이상 미관찰 test가 아니다.
8. 끝에 실행한 것과 미실행한 것을 구분한 보고서, 실패 episode 목록, 재현 명령을 남긴다.

**기존 구현 재사용**

- `AG3S/ag3s/experiments/sources/policy_record.py`: camera binding, record loader, fingerprint, 시간축, replay pose.
- `AG3S/ag3s/experiments/sources/mujoco_source.py`: metric depth, K, T_base_cam, simulator capture.
- `AG3S/ag3s/experiments/studies/t47_replay.py`: 기존 GT class coverage; algorithm에 GT를 전달하지 않도록 evaluator에서만 사용.
- `AG3S/ag3s/experiments/studies/t53_extract.py`: 기존 noun rows/KV/GT producer. 기존 T49/T53/T58 evidence와 중복 실험 여부 확인.
- `AG3S/ag3s/stages/reconstruction.py`, `attention_lifting.py`, `robot_filter.py`: depth validity·좌표·robot filtering 재사용.
- `AG3S/ag3s/stages/subtask_probe.py`: 기존 HSR 입력/정규화/class order.
- `AG3S/ag3s/experiments/studies/t54_arms.py`: 기존 N/A/B attention 역할별 융합 baseline, `t54_study.py`의 replay 입력.
- `AG3S/ag3s/experiments/studies/t55_study.py`, `t55_predicate.py`: strong/HSR/noun, causal persistence, 독립 P_grasp/P_task, event/censoring 및 group 평가. 과거 prereg/model을 고정할 때 지문도 보존한다.
- `AG3S/ag3s/experiments/studies/t56_selfscore.py`: `hand_frame`, `s2_extension`의 co-motion primitive. HELD 조건부 self-score와 online attachment inference를 구분한다.
- `AG3S/ag3s/experiments/studies/t60_timestamps.py`, `t55_windows.py`, `t56_windows.py`: 기존 P/H 평가 창·시간 정합. 적합한 window 생성 경로를 각 study에서 확인한다.
- `AG3S/ag3s/experiments/studies/t61_select.py`, `t61_heldout.py`, `AG3S/ag3s/docs/handoff/T61.selection.json`: 고정 source 및 선택/held-out 평가. 재선택이 필요할 때만 기존 fold-local 선택기를 확장한다.

- `doc/archive/reviews/vla_dump_20261015/grounding_probe.py`: **특정 frame022 전용 진단**. 범용 batch runner가 아니므로 그대로 여러 episode 평가에 사용하지 않는다.

**2. 데이터 분할·수집: frame보다 episode가 표본 단위**

Split은 scene/layout/trajectory family/물체 조합의 누수를 막는 group 단위로 나눈다. 동일 원본 rollout, 재덤프, prompt swap, noise seed 변화, 그 rollout에서 뽑은 모든 frame은 동일 split에 둔다. 기존 모델 training set과 downstream 연구 split을 각각 manifest에 기록한다. 기존 ep1839와 이미 결과를 검토한 T49/T53/T54c/T55/T56/T58/T61 사례는 regression/stress 진단으로 표시하고 새 독립 test라고 부르지 않는다.

다음은 **초기 수집 제안**이며 statistical power를 보장하는 표본 수가 아니다.

- Pilot/dev: 최소 12개 독립 episode, clear/adjacent-obstacle/partial-occlusion/grasp-or-release-failure 네 stratum을 포함한다. 기존 debug 사례는 별도다.
- 방법 선택 뒤 locked test: 다른 group의 최소 12개 episode, 가능한 한 같은 네 stratum을 포함한다. 더 적으면 feasibility 결과로 제한하고 CI/성공률 개선을 강하게 주장하지 않는다. pilot 효과 크기와 episode 분산으로 본실험 sample size를 재산정한다.
- 정해진 episode에서는 성공/실패에 관계없이 전 observation timeline을 compact로 저장한다. 결과가 좋아 보이는 frame만 추출하지 않는다.
- Full audit는 dev에서 구간별 1개씩, 총 5–10개 observation을 우선 선정한다. transition 전후와 가림/실패를 포함하되 phase GT는 평가·샘플 설명에만 사용한다.
- Depth·RGB는 최소 policy observation마다 동기 수집한다. robot state/gripper/실행 action index/HOLD/contact event는 가능한 control cadence로 기록한다. 부족하면 transition delay의 분해능을 그 관측 간격으로 제한한다.

실제 timestamp와 `t_step/ctrl_hz`를 사용한다. ep1839의 15Hz 제어·8-step 관측 간격은 약 0.533초다. 한 frame에서의 denoise 10 step은 10개의 task time sample이 아니다.

Manifest 필수 열: `episode_id, group_id, split, scene_id, object_layout_id, task, target_role, destination_role, record_uri, record_fingerprint, checkpoint_hash, source_commit, seed, observation_times, outcome_label_source, prior_inspection`.

**3. 입력 계약과 GT 격리**

Algorithm 입력: 원본/정책 입력 RGB, depth 또는 lossless 참조, K/T/timestamp, 원본→policy 전처리 mapping, raw/normalized robot state, prompt/token/mask/span, frozen feature/map, 실제 grasp feedback, 과거 관측. prediction에는 현재와 과거만 사용한다. 예측 action은 미래 **의도** feature로 명시할 수 있으나 실제 미래 결과를 넣지 않는다.

Evaluator 전용 입력: simulator body/geom ID mask, GT target pose, GT phase/attachment/contact/completion label, oracle seed/extent. 별도 객체/모듈에서 관리한다. label로 component를 선택하거나 GT object pose로 ROI를 만드는 것은 oracle 결과로만 보고한다.

Surface output 계약:
`frame_id, timestamp, frame_convention, node_ids, xyz, normals, geometric_valid, feature_refs, view_ids, identity_scores, membership_score, target_mask, unknown_mask, confidence_source, timing`.

Semantic role은 target/destination/robot/support/other를 구분한다. non-target 전체를 “항상 충돌 금지 물체”와 동일시하지 않는다. permission builder는 link·attachment·phase별로 따로 판단한다. unknown은 기존 충돌 표현에서 제거하지 않는다. Target membership만으로 environment geometry를 광범위하게 삭제하지 않는다.

Phase output 계약:
`timestamp, phase_probs[pick,place,home], attachment_probs[free,held,uncertain], confirmed_phase, manipulated_id, destination_id, transition_reason, evidence_available, observation_age`.

분류체계를 확장할 때 기존 3-class와의 매핑 및 독립 GT를 별도 명시한다. 물리 상태 held/free는 기존 task phase 3-class와 별개다.

**4. S0 — 정합·parity: 통과 전 성능 분석 금지**

- record fingerprint, checkpoint/config, frame time, raw state, camera order가 맞는지 검사한다. 실패하면 해당 sample을 조용히 생략하지 말고 invalid 이유를 기록하고 분석을 중단한다.
- `base/left/right` → 0:256/256:512/512:768 mapping을 검사한다. image mask와 valid text mask를 보존한다.
- depth 단위, z-depth인지 radial인지, camera-to-base/world convention, crop/resize/pad, distortion 처리를 명시한다. 다른 camera/해상도에 기존 normalized scaling을 무조건 재사용하지 않는다.
- roundtrip backproject/project 오차를 계산한다. 제안 구현 검사 기준은 합성 유효점의 pixel reprojection error ≤0.5px이다. 이는 실제 sensor calibration 정확도 보장이 아니다.
- 실물 RGB/depth registration은 checker/대응점 등 독립 증거로 평가하고 오차를 geometry margin에 반영한다. simulator replay는 RGB 비교와 GT/geometry 정합을 기록한다. GT 평가용 render를 algorithm 입력에 섞지 않는다.
- compact의 저장된 feature slice는 같은 full forward에서 캡처한 값과 equality/허용 오차를 확인한다. 별도 layer replay가 forward를 바꾸면 원본 action과 replay action을 분리한다. FP16 cast는 값 범위를 검사한다; BF16→FP16 overflow를 허용하지 않는다.
- original/checkpoint/dtype/noise가 같아도 backend 연산차가 존재할 수 있다. 사전 pilot에서 action·map repeatability 오차를 측정해 parity tolerance를 고정한다. 효과가 noise floor보다 작으면 개선으로 해석하지 않는다.

현재 참고값: prefix replay parity는 추출 로그상 0, suffix x0 최대 차이 0.00390625. SigLIP attention 재계산의 sa maxabs 차이 0.499–0.574가 있어 정확한 original attention으로 인증되지 않았다. 이 수치를 새 collector의 자동 통과 기준으로 복사하지 않는다.

**5. S1 — Seed × feature factorial: 가장 먼저 구현할 비교**

고정 seed source:
A=`action L8H2, s=0, j=last`.
N=`target noun prefix L10H7`.
P=`normalize(A) * normalize(N)`.
추가 고정 control A61=`action L10H7, s=0, j=mean(all action queries)`. 기존 15개 prototype factorial은 그대로 두고 A61 map 자체를 별도 control로 보고한다. A61 seed의 prototype/graph를 추가할 때는 실행 전에 arm registry에 명시하고 총 arm 수를 수정한다.
문장에 해당 noun이 없거나 span이 모호하면 missing으로 기록한다. 여러 subword면 해당 span의 개별 query row를 같은 head에서 산술 평균한다. `noun_prefix_row`/piece를 반드시 로그에 남긴다. destination noun과 verb row도 보존한다.

seed 크기는 초기 `k=5`; 안정성 분석에만 `k={1,5,10}`을 사용한다. top-k tie는 global token index 오름차순으로 결정한다. masked camera/invalid projected support는 제외한 valid score 위에서 선택한다. score가 모두 0이면 abstain한다. raw-all-patch 진단과 geometry-valid seed 실험은 별도 arm으로 표시한다.

Feature set은 `{SigLIP encoded, projector/image_tokens, prefix L4/L8/L12 resid_out}`으로 고정한다. 모든 번호 0-based. 처음에는 unit cosine만 쓴다.

`fhat_i=f_i/max(norm(f_i),eps)`; `mu=normalize(sum(seed_score_i*fhat_i)/sum(seed_score))`; `s_i=fhat_i dot mu`. feature 차원을 임의로 잘라 비교하지 않는다.

3 seed × 5 feature=15개 prototype arm, A/N/P map 자체 3개 control을 모두 기록한다. 기존 first8 action map 비교 1개는 query diagnostic이다. raw top-K 진단은 K={5,10,20,40} 고정하며 결과를 보고 K를 골라 대표값으로 바꾸지 않는다. raw map과 prototype의 score scale이 다르므로 공통 수치 threshold로 공정 비교했다고 주장하지 않는다.

주요 진단: seed purity, target/neighbor/robot/support mass, fixed-area precision/recall/IoU, prototype spread, camera bias. GT에 의한 seed purity는 평가값이며 seed 선택 입력이 아니다. full surface threshold는 S3 dev calibration에서 별도로 결정한다.

**6. S2 — Point cloud 기본 표현과 geometry-only baseline**

1. 유효 depth를 backproject한다. 기본값은 현 설정에서 읽고 manifest에 저장한다. 현재 기본은 0.05≤z≤3.0m이다. 실제 sensor 유효 범위가 다르면 config를 분리한다.
2. robot self-filter는 기존 FK/geometry를 사용하되 held object 보호 동작을 audit한다. GT robot mask는 oracle 대조군이다. support plane은 RANSAC 등 관측 기반 estimator로 찾고, GT plane을 입력으로 쓰지 않는다. 물체가 plane에 닿았다는 이유만으로 함께 제거하지 않는다.
3. 이미지의 4-neighbor depth surface를 우선 분리한다. jump gate는 `max(10mm, 3*calibrated_depth_sigma)`를 초기 제안값으로 쓰되 dev sensitivity에 5/10/20mm를 둔다. 실측 sigma가 없으면 “모델 미보정”이라고 표시한다.
4. 분리한 surface 안에서 voxelization한다. 초기 voxel 5mm, dev 민감도 10mm. 같은 voxel에 서로 다른 depth layer가 들어오면 하나의 평균점으로 합치지 않는다. node key에 surface/source provenance를 유지한다.
5. node normal은 관측 neighborhood PCA로 추정하고 ill-conditioned normal은 unavailable로 표시한다. feature는 raw dense point마다 복제하지 않고 patch/view 참조로 연결한다.
6. local graph는 k≤12, radius≤3×voxel_size를 초기값으로 쓴다. 유효 surface adjacency가 없는 edge는 만들지 않는다. 3D 근접도만으로 gripper–object/target–neighbor 경계를 연결하지 않는다.
7. patch seed가 여러 surface를 덮으면 각각 candidate evidence를 받지만 모두 hard-positive로 확정하지 않는다. patch evidence를 point 수만큼 복제하지 말고 투영 면적/가시성으로 분배하여 source patch당 mass가 보존되게 한다.

Geometry-only G baseline: 같은 seed 증거·robot/support filter·node set에서 appearance affinity 없이 connected/geodesic region score를 계산한다. Feature+geometry FG와 candidate ROI/normalization/permission builder는 동일하게 유지한다. 특정 arm만 GT ROI나 더 좋은 robot filter를 사용하면 안 된다.

장면이 큰 경우 seed 기반 ROI를 사용할 수 있으나 GT target pose를 쓰지 않는다. 초기 ROI는 valid seed surface 주변 0.15m union이며, ROI 밖 타깃 누락도 recall 분모에 포함한다. ROI 제한을 적용한 모든 비교군에 동일하게 적용하고 전체 valid-scene run을 dev control로 남긴다. node 수/edge 수/시간을 기록하며 density cap이 필요하면 deterministic voxel downsampling을 사용한다.

**7. S3 — Feature+geometry graph**

G와 FG가 같은 nodes/edges 및 attention source를 사용하게 한다. G는 appearance unary와 feature edge 항을 모두 제외하고, FG는 이를 추가한다. Feature unary off/on × feature edge off/on의 2×2 mechanism control로 두 효과를 분리한다. Attention seed·기하 mask·ROI는 고정한다. Feature unary를 남긴 채 edge에서만 feature를 뺀 것은 G 전체가 아니라 edge ablation이라고 표시한다.

`w_ij = valid_edge * exp(-dpos²/sigma_p² - (1-|ni·nj|)/sigma_n - ||fhat_i-fhat_j||²/sigma_f²)`.

normal invalid이면 normal 항을 생략하고 기록한다. 초기 `sigma_p=2*voxel_size`, `sigma_n=0.2`, `sigma_f=0.5`; dev sensitivity는 sigma_f={0.25,0.5,1.0}. geometry-only는 feature 항만 0으로 둔다.

unary는 bounded score를 사용한다. seed positive score는 provenance와 함께 soft 값으로 부여하고, 확인된 robot/support는 negative evidence로 사용한다. low attention만으로 negative label을 만들지 않는다. node score s의 변환식과 reliability Λ의 생성 규칙을 resolved config에 고정한 뒤 `(m-s)^TΛ(m-s)+λm^TLm`을 0≤m≤1에서 최소화한다. λ 후보={0.1,1,10}. sparse solver의 residual/수렴을 기록한다. unary가 없는 component는 unknown이다. 양성 anchor만 있는 connected component는 m=1이 최적해가 될 수 있으므로, all-target 누출을 λ 조절만으로 해결하지 않는다. 상세 퇴화 진단과 Q/C 목적함수 구분은 개선 가이드 §15.5를 따른다.

실행 threshold 후보는 τ={0.5,0.7,0.9}; score가 probability로 calibrated되었다고 주장하지 않는다. unknown interval은 initial [0.3,τ)이며 reliable-negative와 단순 low-score를 구분한다. operating point는 dev에서 다음의 사전 고정 규칙으로 선정한다: false-target exclusion ≤baseline, target recall ≥baseline-0.02를 만족하는 후보 중 macro 3D IoU 최대. 만족 후보가 없으면 “해당 기준 통과 없음”으로 보고한다. 이 0.02는 제안된 dev 선택 허용치이며 검증된 안전 margin이 아니다.

feature/voxel/sigma/λ/τ 전체 Cartesian grid는 금지한다. 첫 pass는 기본값으로 15 prototype·G/FG 효과를 비교하고, dev에서 상위 2개 seed-feature pair를 위 규칙으로 선택한다. 이후 해당 pair의 작은 one-factor sensitivity만 수행하며 평가한 모든 config를 registry에 남긴다. test를 보고 수정하지 않는다.

S3에 추가할 후순위 ablation: positive–negative prototype 대비, multi-prototype, graph geodesic, normal 제거, robot-filter 제거(offline만), shuffled feature. 최초 비교를 완료하기 전에 방법을 무제한 확장하지 않는다.

**8. S4 — Multi-view와 시간적 identity**

단일 view 처리 결과와 공동 base/world surface graph를 비교한다. projective depth consistency+distance gate로 대응시키고 occluded/unobserved는 negative evidence로 넣지 않는다. 동일 source feature/head에서 나온 evidence를 독립 관측처럼 곱하지 않는다.

view별 semantic evidence는 같은 surfel에 연결하되 total surface weight를 중복 횟수만큼 늘리지 않는다. 누락 view/두 view/세 view ablation을 같은 frame 집합에서 수행한다. valid view가 0개인 물체는 current observed segmentation을 생성하지 않는다; memory는 age와 source를 붙여 별도 관리한다.

움직이는 물체는 world-fixed fusion이 ghost geometry를 만들 수 있으므로 timestamp와 object/hand-frame motion을 고려한다. object tracking은 관측 feature+geometry로 하고 GT pose는 평가에만 쓴다. 초기 baseline은 gated nearest-neighbor/rigid registration이며 partial OT는 비용 대비 이득이 확인될 때 추가한다.

타깃 identity memory와 current extent는 분리한다. 기존 identity를 유지해도 잘못 붙은 voxel은 수정 가능해야 한다. 한 번 잘못 섞인 경계를 memory로 고착시키지 않는다.

**9. P0–P4 — 현재 작업 단계와 물리 attachment**

P0 state-only/physical strong rule. P1 기존 KV_L4 HSR. P2 HSR+기하 관계 score. P3 P2+시간 filter. P4 P3+관측 motion consistency. 모든 arm에 같은 physical feedback와 lifecycle 구현을 제공하고 입력 evidence만 바꾼다. phase probe 학습을 추가하면 별도 arm P_supervised로 분리한다.

기하 feature 정의:
- hand–estimated-target surface 최소 robust distance, target–destination 상대 위치/height, 실제 gripper 상태와 변화량.
- action mass의 target/destination 분해는 estimated membership으로 계산한다. GT mask 분해는 oracle diagnostic이다.
- tracked object motion: world-static residual과 gripper-frame-static residual을 비교한다. FK pose와 point coordinates를 같은 frame으로 맞춘다. 이동 base는 world transform으로 보정한다.
- correspondence residual은 robust loss를 쓰며 visibility/outlier 비율을 함께 보고한다. gripper 자체를 held object라고 판단하지 않도록 robot geometry를 분리한다. 손이 움직이지 않아 두 가설을 구분 못하면 uncertain으로 유지한다.
- predicted FK action 경로는 intention feature로 별도 표시한다. predicted 성공과 실제 observed completion을 동일시하지 않는다.

시간 baseline: P2 emission score에 기존 phase 순서와 failure/retry transition을 허용하는 유한-state filter를 적용한다. 첫 구현은 hysteresis/physical-event rule로 시작한다. HMM/HSMM을 비교할 때 transition/duration은 train/dev에서 고정하고 실제 초 단위를 쓴다. 모든 VLA 신호가 독립 likelihood라고 가정하지 않는다.

GT phase는 기존 independent task annotation을 재사용하거나 별도 annotation protocol을 고정한다. HSR 예측값으로 GT를 만들지 않는다. held label은 attachment 평가에는 사용할 수 있지만 pick/place/home phase label을 자동 대체하지 않는다. GT가 없다면 transition evidence/consistency diagnostic까지만 수행하고 phase accuracy를 보고하지 않는다.

지표: macro-F1/class별 recall, Brier score 또는 calibration, event별 transition delay(조기 전환은 별도), false transitions/episode, attachment false-positive/miss, identity switches, release 후 stale exemption 지속시간. 성공 transition만 골라 delay를 측정하지 말고 missed/censored event 수도 보고한다. 정지/완전 가림/실패/retry/episode reset을 포함한다.

**10. S5 — 인과 검증: offline 재추론**

그림이 좋은 것과 action에 실제 기여하는 것을 분리한다. original 관측·checkpoint·initial noise·dtype·solver를 고정하고 다음을 비교한다: target-region perturbation, neighbor-region perturbation, equal-area random perturbation, low-score perturbation. 각 region은 predicted와 oracle 결과를 분리한다.

input masking과 hidden-value patching은 별도 실험이다. input perturbation은 prefix부터 다시 계산한다. 후반 hidden에서 image row를 빼는 것만으로 state/text 입력을 제거했다고 주장하지 않는다. value patch와 key patch를 분리해 content와 routing을 검사한다. 영벡터 replacement의 분포 밖 artifact를 보완할 mean/norm-matched control을 둔다.

paired noise 목록은 dev에서 고정하고 최소 3 seed로 repeatability를 점검한다. `x_initial` actual tensor를 저장한다. phase가 같은 원본/perturbed 비교와 명령 target 변경 비교를 섞지 않는다.

지표: active16 normalized action 차이, 역변환 joint/gripper별 변화, FK end-effector 경로, 실제 실행 K개 구간 변화, predicted clearance. 전체32 norm은 padding 영향을 포함하므로 보조 지표다. action 변화가 크다는 사실만으로 target segmentation이 맞다고 판정하지 않는다.

**11. S6 — downstream replay/closed-loop 구분**

Extent만 바꾸는 비교는 기존 phase/lifecycle을 고정하고 P 확장 없이 수행할 수 있다. Phase 처리까지 바꾸거나 그 개선을 주장할 때 P 평가를 선행한다.

고정 recorded trace에 새 segmentation을 적용하는 것은 geometry/lifecycle/constraint 진단이다. 새로운 policy/controller 행동으로 미래 observation이 달라지는 closed-loop success를 증명하지 않는다.

offline 통과 후 simulator closed-loop를 별도 실험으로 수행할 때 checkpoint, optimizer, permission rules, initial scene/seed를 고정하고 paired episode로 비교한다. baseline/proposed의 TO 수정·HOLD·latency·접촉·성공 종료 조건을 같은 정의로 집계한다. 실패 episode의 종료 후 유리한 prefix만 잘라내지 않는다. 실제 robot 테스트는 별도 요청·환경 준비가 있는 경우에만 범위에 포함한다.

**12. 평가 지표의 정확한 정의**

Visible target GT surface를 G, predicted target을 M, 공통 관측 surface node weight를 w라 한다. TP=sum(w on M∩G), FP=sum(w on M\G), FN=sum(w on G\M). Precision=TP/(TP+FP), Recall=TP/(TP+FN), IoU=TP/(TP+FP+FN).

평가 node set은 같은 geometry discretization을 공유한다. dense pixel 수로 fused-view 면적을 중복 집계하지 않는다. point density의 영향을 피하려면 공통 voxel occupancy 또는 surface area weight 중 하나를 사전 고정하고 두 지표를 혼용하지 않는다. voxel resolution도 결과표에 표시한다.

G가 비었으면 recall/IoU를 임의로 1로 채우지 않는다. target-absent frame에서 false-positive 면적을 별도 보고하고 absent sample 수를 남긴다. G가 있는데 M이 비었으면 recall=0/IoU=0, precision은 undefined로 기록한다. unknown target surface는 FN이며 abstain으로 지표가 좋아 보이지 않게 한다.

Contamination은 predicted region 내부 non-target 비율이며, neighbor/robot/support별로 분해한다. **False-exemption**은 실제 permission builder가 제외한 non-permitted surface의 면적/voxel 수다. segmentation FP와 같다고 가정하지 않는다. 전체 observed non-target 대비 비율과 절대 면적/수를 함께 보고한다.

통계: frame 지표를 episode별 평균으로 만든 후 episode/group 수준 paired bootstrap 2000회(seed=20261015)로 difference CI를 계산한다. dependence group이 있으면 group resampling을 사용한다. 성공률은 episode binary outcome, missed transition은 episode/event count다. 천 개 frame을 천 개 독립 trial로 취급하지 않는다.

runtime: warmup과 compile/I/O 시간을 분리하고 extraction+transfer+segmentation+fusion의 전체 p50/p95를 보고한다. 원래 VLA 계산 시간에 이미 있는 feature라도 hook materialization·host copy 비용은 0이 아니다. memory는 baseline 대비 peak 증가량을 기록한다. collector의 debug replay 시간과 live serving latency를 혼동하지 않는다.

**13. 통과·중단·보고 기준**

- Integrity 실패: 성능 평가 중단, 원인/affected samples/복구 방법 기록. invalid frame을 정상 negative로 대체하지 않는다.
- A/feature-only 악화: seed 오류 vs extent 오류를 oracle 진단으로 분리한 후 G/FG를 계속 비교한다.
- Geometry-only와 FG가 비슷함: 현재 추가 이득 미확인으로 기록하고 §13.1의 seed/node/edge/unary 진단을 수행한다. 근거가 있는 최소 개선을 재실험하며, 근거 없이 복잡한 방법을 자동 채택하지 않는다.
- Stage completion: 모든 사전 지정 arm, 실패/결측, 비용, 산출물을 기록한다. 실패한 FG 방법은 §13.1에 따른 개선 분기의 채택/기각·종료 근거까지 남겨야 전체 작업 완료다.
- Held-out benefit claim: episode-level 효과와 CI, recall–false-exemption tradeoff를 제시한다. 기준을 통과하지 못하면 개선 미확인으로 표시한다. CI가 넓으면 inconclusive다. 이 절차 자체는 안전 보장을 제공하지 않는다.

필수 산출물: `manifest.json`, `split_manifest.jsonl`, `resolved_config.json`, `provenance_checks.json`, `sample_metrics.csv`, `episode_metrics.csv`, `paired_summary.json`, `latency.json`, `failures.jsonl`, representative overlays/3D views, 실행 명령과 환경을 포함한 `REPORT_KO.md`.

CSV 최소 열은 `run_id, episode_id, frame_id, split, arm_id, seed_source, feature_name, layer, voxel_size, threshold, valid, missing_reason, TP, FP, FN, precision, recall, iou, neighbor_fp, unknown_weight, latency_ms`다. phase CSV는 `timestamp, gt_source, gt_phase, predicted_phase, attachment_gt, attachment_prediction, evidence_available, transition_event_id`를 추가한다.

**13.1 실패 후 개선·재실험 — 필수 실행 절차**

[3D Surface Graph 및 튜닝 가이드 §15](../AG3S_3D_Surface_Graph_and_Parameter_Tuning_Guide.md#failure-improvement-loop)를 실행한다. 최초 실패를 저장한 뒤 **실패 분류 → 증거/반증 가능한 가설 → 최소 수정 → 같은 parent/reference 대조 → 전체 dev 재평가 → 채택/rollback**을 반복한다. F0 정합, F1 seed, F2 node, F3 edge, F4 feature, F5 unary, F6 solver, F7 시간, F8 비용 중 근거 있는 원인을 선택한다. 가림이 실패 원인이면 공간 단계의 선행 성공 없이도 F7의 제한된 개선을 진행할 수 있다.

한 cycle은 한 원인·한 모듈, parent 포함 최대 3개 설정을 초기 범위로 둔다. 최초 R0 이후 3개 개선 cycle마다 작업 묶음을 요약하되, 남은 원인과 근거가 있으면 다음 묶음을 등록하여 계속한다. 같은 원인에 2회 연속으로 진단 지표와 최종 품질의 개선이 모두 없으면 해당 분기를 기각한다. 이는 전체 작업을 즉시 포기하는 규칙이 아니다.

Reference·평가 surface/grid·전체 dev manifest를 고정한다. 기존 `Δfalse-exemption≤0`, `Δrecall≥−0.02` 제약은 유지한다. Runtime/최소 실용 차이는 후보 결과 전에 baseline 반복 오차와 목적에 따라 config에 고정한다. Oracle은 dev 진단으로만 쓰고, runtime에는 GT를 넣지 않는다. 개선을 위해 test 결과를 사용했다면 해당 episode family는 후속 버전에서 inspected/dev로 표시하고 새 독립 test로 확인한다. 미사용 데이터가 없으면 개발 작업은 계속하되 최종 검증은 pending으로 보고한다.

Solver는 `FG-R`(region growing), `FG-Q`(본 §7 연속 energy), `FG-C`(binary Potts cut)를 구별한다. Q가 기본 graph 기준이며 C는 근거 있는 추가 arm이다. Objective·affinity scale·unknown 처리가 다르므로 같은 solver의 설정 변경으로 보고하지 않는다.

각 cycle은 `hypothesis.json`, `resolved_config.json`, `diagnostics.json`, `episode_metrics.csv`, `paired_summary.json`, `latency.json`, `decision.json`, 실제 `commands.txt`를 남긴다. 상위 registry의 필수 열은 가이드 §15.8을 따른다. [개선 round 템플릿](improvement_round_template.json)은 실행 전 등록용이며 자동 runner가 아니다. 성공뿐 아니라 **원인별 개선과 반증 실험을 마친 음성 결과**도 완료다. 초기 실패 한 번·임의 반복 횟수 도달은 그 자체로 완료 사유가 아니다.

**14. 이미 실행 가능한 재현과 아직 구현할 작업**

기존 단일 frame 진단은 다음 명령으로 재현한다. 이것은 smoke/regression이며 위 범용 실험 전체의 실행 명령이 아니다.

```bash
MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/mnt/dev/AG3S_VLA-development \
.venv-ag3s/bin/python doc/archive/reviews/vla_dump_20261015/grounding_probe.py \
  --dump /mnt/dev/AG3S_VLA-storage/results/development/20261015T150325+0900_ddc7a00060c4_d857a08dcb63/vla_dump \
  --t53 /mnt/dev/AG3S_VLA-storage/results/development/20261013T100210+0900_4a540f1a18d7_cb037c425d2f/verify/T53 \
  --out /tmp/ag3s_grounding_smoke --replay
```

이 명세의 compact collector/reader, feature+geometry graph 및 이를 연결한 통합 batch runner는 아직 구현되지 않았다. 기존 attention/feature batch producer와 temporal·predicate evaluator는 T53/T55/T56/T60 등에 이미 있으므로 재사용·확장한다. agent는 기존 producer를 재사용하며 위 stage에 맞춰 구현하고 **실제로 실행된 CLI만** 최종 보고서에 기록한다. 가상의 CLI가 이미 동작한다고 안내하지 않는다. 새 대규모 raw 산출물은 repository의 `common.run_output` 규칙에 따라 새 run 디렉터리에 저장하고, checkpoint·과거 결과를 덮어쓰지 않는다.
