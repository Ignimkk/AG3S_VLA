# AG3S T47~ — attention head 재선택 · HSR counterfactual · 명사 token query 기록

> **이 파일은 최신 정본이다 (2026-10-12 개편, 사용자 지시).** STEP 번호 **T47 이상**은 여기에 이어 쓴다.
> T46 까지의 기록은 [`AG3S_T0T6_LOG.md`](AG3S_T0T6_LOG.md) (이전 정본) 에 있다. 읽을 때는 **최신 → 오래된** 순서다.

**프로젝트 목적.** AG3S 에서 VLA 의 visual attention 을 활용하여 작업 대상(Target)과 장애물(Obstacle)을
구분하고, Target 을 충돌 대상에서 제외한 3D 환경 정보를 cuRobo 의 TSDF/ESDF 파이프라인으로 전달하는
전체 시스템을 구축한다. T47 이후는 그 중 **attention 으로 target 을 가리키는 입력 쪽**(어느 head 가
target 을 가리키는가, 명사 token query 가 더 나은가)과 **frozen VLA feature 위의 phase probe 가 무엇을
읽는가**(HSR counterfactual)를 다룬다.

> **계획서** — [`AG3S_REVIEW_PLAN.md`](AG3S_REVIEW_PLAN.md) (무엇을 왜 하는가),
> [`AG3S_CUROBO_LIVE_TEST_PLAN.md`](AG3S_CUROBO_LIVE_TEST_PLAN.md) (T0~T6 실행 계획, 이전 국면).
> T47 이후 연구 방향: `doc/reviews/AG3S_TWO_PAPER_REVIEW_20261012_KO.md`.

## 문서 상태 — 로그 두 파일 (최신 → 오래된)

| 순서 | 파일 | 상태 | 범위 | 언제 읽나 |
|---|---|---|---|---|
| 1 | [`AG3S_T47_LOG.md`](AG3S_T47_LOG.md) (이 파일) | **최신 · 이어 쓰는 정본** | T47 이후 모든 STEP (T47·T47b, T48a·T48b, T49 …) | 항상 먼저. 맨 위 표 (용어 · 진행 현황 · 물려받은 결정 · 재측정 대기) 로 현재 위치를 안다 |
| 2 | [`AG3S_T0T6_LOG.md`](AG3S_T0T6_LOG.md) | **이전 정본 · T46 까지** | 16D 모델 위의 T0→T46 실행 기록. T46 (V-T46 · CL3) 은 거기서 마무리한다 | T46 이전 결정 · 발견 ID 의 출처, 옛 STEP 의 원 측정을 찾을 때 |

- 새 STEP 의 판단 근거가 옛 로그에 있으면 그 줄을 **링크**하고 다시 옮겨 적지 않는다.
- 옛 로그의 수치와 이 로그의 수치가 다르면 **이 로그가 우선**하고 그 차이를 여기 적는다.

---

## 용어 (T47 이후)

새 용어는 여기 추가한다 (규칙 C). T46 이전의 용어는 [`AG3S_T0T6_LOG.md`](AG3S_T0T6_LOG.md) 의 "용어" 절에 있다.

**T47 · T47b (2026-10-12 추가)** — attention head 재선택 (localization head 기준) 과 그 보강 측정의 용어.

| 용어 | 뜻 |
|---|---|
| **localization head** | text (또는 query) 가 가리키는 이미지 영역에 attention 이 모이는 head. Kang et al. (arXiv 2503.06287) 이 label 없는 기준으로 고르는 대상. 우리 query 는 action token 이라 "target 에 모인다" 는 보장이 없어 GT class 로 분해해 확인한다 |
| **`S_cam` · `S_img` (image attention sum)** | `S_cam` = 한 camera 의 16×16 (256 token) attention 질량 합, `S_img` = 3 camera 768 image token 의 합. 정규화하지 않은 raw softmax mass 다. 이 값이 작은 head 는 이미지를 거의 보지 않는 head 로 보고 거른다 |
| **spatial entropy** | `ReLU(A − mean A)` 를 binarize → 8-connectivity 성분 → `P(C) = |C| / Σ|C|` → `H = −Σ P ln P` (nats). 한 덩어리면 0, 같은 크기 두 덩어리면 ln 2. attention 확률 자체의 Shannon entropy 와 다르다 |
| **τ (Kneedle knee)** | 오름차순 정렬한 `S` 곡선의 knee. `S ≥ τ` 인 cell 만 eligible. sample (frame, camera) 마다 그 sample 의 128 cell 로 계산 (primary). 대안: 이산 curvature, 전체 `S_img` 기준 |
| **eligible cell** | 그 sample 에서 `S ≥ τ` 인 cell. selection frequency 의 후보 |
| **selection frequency · 공통 / camera 별 top-k** | sample 마다 eligible 중 spatial entropy 가 가장 낮은 10 개를 고르고 cell 별로 고른 비율을 센 값. 비율 순위의 앞 k 개가 **고정 head set**. 공통 = 모든 camera 의 sample 을 합쳐 하나, camera 별 = camera 마다 하나. 후보 cell = L2–17 × 8 head = 128 (L0 · L1 제외) |
| **`raw_sum` · `mass_norm_joint` · `mass_norm_per_camera`** | head set 을 합치는 방식. `raw_sum` = 논문 그대로 smoothing 후 element-wise 합. `mass_norm_joint` = head 마다 3 camera 768 token 질량으로 나눈 뒤 합 (camera 간 비율 보존). `mass_norm_per_camera` = (head, camera) 마다 16×16 질량으로 나눔 (camera 간 비율 소멸) |
| **σ (smoothing)** | cell 별 Gaussian smoothing 의 표준편차, 16×16 grid 단위. σ=0 은 smoothing 없음. reflect 경계라 map 마다 질량이 보존된다 |
| **head-average** | L2–17 의 128 cell 평균 (σ 적용). mass 모드에서는 cell 마다 질량을 1 로 만든 뒤 평균 |
| **GT-oracle (T47)** | selection record 의 GT 로 `peak_on_target` (3 camera pooled) 순위를 매겨 고른 head set. 상한 참고용이고 채택 후보가 아니다. d0 · last 의 top-3 = L8H3 · L10H7 · L12H4 |
| **target mass · lift · IoU (2D 지표)** | target mass = attention 질량 중 target 칸의 몫. lift = target mass ÷ target 면적 몫 (1.0 이 uniform). IoU = 평균으로 binarize 한 map 대 GT target patch mask (patch coverage ≥ 0.5, 0.25 변형). `peak_on_target` 은 위 용어 표 |
| **class mass (분해)** | attention 질량을 GT class 로 쪼갠 몫: target · destination (crate) · other_object (나머지 과일) · robot_gripper · robot · support (table · shelf) · background |
| **held-out same-scene / new-scene** | held-out 4 record 를 selection 과의 관계로 가른 것. same-scene = selection 의 `long` · `t2` 와 같은 episode 1800 에서 출발 (`headfix` · `ep1800`, apple), new-scene = 다른 episode (`ep1850` banana · `ep1900` pear). 일반화 판단의 주 근거는 new-scene 이고 둘 다 pilot |
| **pre-registration (사전 고정)** | 결과를 열기 전에 primary 설정 (head set · k · σ · query · τ · 합산) 과 채택 판단의 근거를 고정해 두는 것. 고정 밖의 변형은 exploratory 로 보고만 한다 |
| **pre-grasp / post-grasp (phase group)** | pre-grasp = approach · pre_grasp · transit, post-grasp = 그 밖의 phase (grasp 이후). offline e2e 에는 grasp latch 가 없어 post-grasp frame 의 target 고정이 재현되지 않는다 |
| **`identity_correct`** | chosen cluster 의 label 이 있는 점 중 **엄격히 가장 많은 class 가 target** 인 frame. 동점 · target 없음 = 거짓. `on_target` (centroid 가 target 원점 10 cm 안) 이 놓치는 이웃 과일 오분리를 잡는다. `identity_failure` = `has_target ∧ ¬identity_correct` |
| **`nontarget_seed_loss`** | chosen cluster 에 든 GT non-target 점 수 (other_object · destination …). T44 의 `distractor_seeds_lost` (TSDF seed voxel 단위, voxel 다수결 label) 와 **정의가 다르다** — 숫자를 직접 비교하지 않는다 |
| **`target_residual` (T47b)** | GT target 의 관측 점 중 chosen cluster 에 못 든 수와 그 비율. (T45 의 "target 잔여" 는 target-free 층에 남은 target 표면이고 단위가 다르다) |
| **fixed-patch sink · broad single blob** | 진단용 읽기 이름. fixed-patch sink = frame 이 바뀌어도 argmax 가 늘 같은 patch (modal ≥ 0.8 이고 top-1 patch 몫 ≥ 0.3). broad single blob = 질량이 넓게 퍼져 평균 위 pixel 의 최대 component 가 grid 의 0.15 이상. threshold 는 읽기용이고 어느 선택에도 쓰지 않았다 |
| **`modal_fraction` · top-1 patch 몫 (sink 진단)** | modal_fraction = frame 별 argmax patch 중 가장 흔한 patch 의 비율. top-1 몫 = 최대 patch 값 ÷ `S_cam` |
| **B3 (raw amplitude 점검)** | lifting 에서 grounding 까지 raw (정규화 전) attention 에 절대값 비교를 하는 곳을 file:line 으로 찾아 ×k 배율 불변 여부를 판정하고 수치로 시험한 것 (24 지점). `decision_identical` = grounding 의 결정 (cloud · seed · peak · cluster · status · chosen) 이 같음, `bit_identical` = 그에 더해 정규화 값 · score · centroid 가 bit 단위로 같음 |
| **T47 · T47b · T49** | T47 = head 재선택 (label-free 기준) 의 2D · 3D 검증. T47b = 합산 방식 · label 기반 3D 지표 · raw amplitude · sink 진단. T49 = 명사 token query (D) 로 넘어가는 다음 STEP (~~아직 시작 전~~ → 진행 중 (2026-10-12, 진행 현황 표 참조), `handoff/T49.task.md`) |

**T48a (2026-10-12 추가)** — HSR counterfactual 의 용어.

| 용어 | 뜻 |
|---|---|
| **HSR (Hidden-state Sub-task Recognition)** | 얼린 pi0.5 prefix 의 hidden state (KV cache V · layer 4 · head 0, `[text 평균 \| image 평균]` 512 차원 `kv_L4`) 위에 logistic regression `kv_L4_v1` 하나를 얹어 지금 하는 일을 `pick` / `place` / `home` 으로 읽는 구성 요소. label 은 `TargetConfirm` 의 subtask gate 로 들어간다 (위 SUBTASK 용어의 hidden-state probe 를 이 구성 요소의 이름으로 부른 것) |
| **weak decodability** | feature 에서 label 이 읽힌다 (held-out accuracy 가 높다). SUBTASK-b 의 val 0.9808 이 이것이다. 읽는 *이유* (작업 상태인지 시간 순서인지) 는 말하지 않는다 |
| **strong decodability** | 관련 입력이 바뀌면 출력이 올바르게 반응한다 (instruction 의 target 을 바꾸면, 물리 상태를 바꾸면). 검토 문서 §4 가 Bhardwaj et al. (arXiv:2608.13474) 의 구분으로 가져왔다. T48 이 처음 재기 시작했다 |
| **counterfactual (입력 변형)** | 한 frame 의 입력 셋 (prompt · 16D state · RGB 3 장) 중 일부만 바꾸고 나머지는 고정해 같은 probe 에 다시 넣는 것. 바뀐 입력과 예측 변화의 관계만 본다 |
| **C0 · C1 · C1p · C2g · C2s · C2i (T48a)** | C0 = 변형 없음 (재현 점검). C1 = prompt 의 target 명사를 같은 scene 의 distractor 과일로 교체 (정답 = instruction 기준 `pick`). C1p = 같은 target, 다른 phrasing (통제). C2g = used gripper 값 하나만 반대로. C2s = image A + state B (state 이식). C2i = image B + state A (image 이식). A = host frame, B = 같은 episode 의 다른 class frame |
| **wrong-object 정답 = instruction 기준 `pick`** | 지시된 물체가 아닌 것을 쥐고 있을 때의 정답을 사용자 판정으로 정한 것 (2026-10-12). 예: 사과를 쥔 frame + prompt "pear" → `pick` |
| **strong rule** | 실제 grasp latch (`grasp_latch.py`) — command · measured opening · latch · history 와 실행 feedback (`applied_gripper`, `blocked_gap`) 를 쓰는 상태 기계. T48a 는 이것을 측정하지 않았다 |
| **strong-rule proxy (offline)** | strong rule 을 offline 기록으로 흉내 낸 근사 — measured opening 과 이력만 쓰는 latch 상태 기계 (PRE → CLOSING → HELD → PLACED), 요청 단위. **strong rule 이 아니며** 그 결과를 strong rule 과의 비교로 읽지 않는다 |
| **simple rule** | SUBTASK-b 의 rule — 양 gripper 중 하나가 닫힘 문턱 아래면 `place`, 아니면 지금까지 닫힌 적이 있으면 `home`, 아니면 `pick`. prompt 를 보지 않는다 |
| **frame-index baseline** | train frame 의 frame index 별 다수 class 로 예측. T48a 의 ep1807 은 train 격자에 없는 frame index 를 가장 가까운 것으로 대신한다 (nearest fallback, 549 / 675 row) |
| **host frame A · partner frame B** | C2s · C2i 의 쌍. A 는 원래 frame, B 는 A 와 같은 episode 의 **다른 class** frame (seed 0 고정). C2s = image A + state B, C2i = image B + state A |
| **`to_pick` · `retain` · `changed` · `follow` (T48a 지표)** | `to_pick` = C1 에서 `pick` 으로 예측된 비율 (원래 class 별). `retain` = C1p 에서 같은 host frame 의 C0 예측이 유지된 비율. `changed` = C2g 에서 예측이 C0 예측과 달라진 비율. `follow` = C2s · C2i 에서 예측 class 가 state 쪽 class / image 쪽 class / 기타 인 비율 |
| **serving state bin · replica 불일치** | 서빙 tokenizer 가 16D state 를 `np.digitize` 로 256 bin 으로 만든 값 (model 이 실제로 본 prompt 의 `State:` 숫자). numpy replica 는 보고용이고, val 27 row (`ep_001648` 차원 13, 1 bin 차이) 에서 서빙 bin 과 달랐다 |
| **T48 · T48a · T48b** | T48 = HSR counterfactual. T48a = prompt · state · image 변형 (rendering 불필요, 기록 frame 만 사용). T48b = 물체의 유무를 바꾼 image (empty grasp · slip · phantom hold) 를 MuJoCo 로 다시 render 해 같은 팔 자세에서 probe 가 쥠 상태를 읽는지 가르는 다음 STEP (`handoff/T48b.task.md`) |

---

## 진행 현황 — T47 이후

| 게이트 | 무엇을 묻나 | 상태 |
|---|---|---|
| **T47 · T47b** attention head 재선택 (label-free: attention sum + spatial entropy, 고정 top-k) 과 보강 (합산 방식 · label 기반 3D 지표 · raw amplitude · sink 진단) | 16D · camera 3 대에서 다시 고른 head set 이 L8H2 보다 held-out 에서 target 을 더 잘 가리키는가 | **완료 (2026-10-12) — L8H2 유지 (사용자 판정).** 사전 고정 primary (L4H1 · L2H0 · L14H5, σ=1) 는 3D `on_target` 0 / 60 (L8H2 55 / 60), target mass 0.0134 (L8H2 0.1498). 선택된 head = sink 1 + broad blob 2. 합산 방식은 3D 를 거의 안 바꿈, `on_target` 이 놓치는 `identity_correct` 불일치와 σ=1 의 이웃 과일 흡수를 측정. held-out 은 pilot. 후속 T49 (명사 token query). 아래 "T47 · T47b" 절 |
| **T48a** HSR counterfactual (a): prompt target 교체 · state 이식 · image 이식 · gripper 반전 (rendering 불필요) | frozen VLA feature 위의 phase probe (`kv_L4_v1`) 가 instruction · state 문자열 · image 가 바뀌면 어떻게 반응하는가 (strong decodability) | **측정 완료 (2026-10-12).** C0 재현 (val 0.9804 대 0.9808). 사실 셋: (a) probe 는 image 를 따르고 state 문자열을 따르지 않는다 (C2s · C2i 에서 image 쪽 0.979, gripper 반전 시 예측 변화 0.005), (b) instruction 의 target 교체에 거의 반응하지 않는다 (C1 원래 place → `pick` 0.040 = C0 의 place → pick 오류율 0.040), (c) grasp state 를 읽는지 팔 자세 순서를 읽는지는 **미결**. strong-rule proxy 는 strong rule 이 아니다. **T48b 진행 (사용자 판정)**. 아래 "T48a" 절 |
| **T48b** HSR counterfactual (b): empty grasp · slip · phantom hold (물체의 유무를 바꾼 image 를 MuJoCo 로 re-render) | probe 가 "물체를 쥐고 있다" 를 읽는가, "팔이 이 자세에 있다" 를 읽는가 | **진행 중 (2026-10-12)** — 16D policy image re-render parity 확인이 먼저. task: [`handoff/T48b.task.md`](handoff/T48b.task.md). 아직 기록할 측정 없음 |
| **T49** 명사 token query: prefix 의 target 명사 · "basket" token → image attention 으로 target · 목적지 map | action token query (L8H2) 대신 명사 token 을 query 로 쓰면 target · 목적지를 더 잘 가리키는가 | **진행 중 (2026-10-12)** — 구현 단계 (`handoff/T49.impl.md` 있음), verify 대기. task: [`handoff/T49.task.md`](handoff/T49.task.md). 아직 기록할 측정 없음 |

> T46 까지의 게이트 표는 [`AG3S_T0T6_LOG.md` 의 "진행 현황"](AG3S_T0T6_LOG.md#진행-현황--t0t6) 에 있다.

---

## 물려받은 결정 — T47 이후

**T46 이전 결정은 [`AG3S_T0T6_LOG.md`](AG3S_T0T6_LOG.md) 의 "물려받은 결정" 표를 본다.** 여기 있는 것은
T47 이후에 판정된 것이고, 다시 판정하지 않는다.

| 결정 | 무엇 | 모델 |
|---|---|---|
| T47 — 서빙 attention cell 은 L8H2 (`denoise 0 · agg last · L8 · h2`) 를 유지한다 | 사용자 판정 (2026-10-12). label-free 기준 (attention sum + spatial entropy) 으로 다시 고른 공통 top-3 가 held-out 에서 실패했다 (3D `on_target` 0 / 60 대 L8H2 55 / 60). 교체는 새 근거가 있는 별도 STEP 에서만 판정한다 | 16D (pilot 규모) |
| T47b — grounding 평가는 `on_target` 에 `identity_correct` (label 기반) 를 함께 본다 (**제안, 판정 대기**) | `on_target` 이 참이어도 chosen cluster 의 다수 label 이 target 이 아닌 frame 이 있다 (60 frame 중 6 – 13, 반대 방향 0) | 16D |
| T48 — HSR counterfactual 은 T48a (rendering 불필요) 먼저, wrong-object 의 정답은 instruction 기준 `pick`, T48b (empty grasp · slip) 는 T48a 결과 후 판정 → **진행** | 사용자 판정 (2026-10-12). 사과를 쥔 frame + prompt "pear" → `pick`. offline GPU feature 추출 (prefix pass, server 없음) 승인. T48b 는 16D policy image re-render parity 를 먼저 확인하고 GPU 추출은 T46 의 CL3 후 (`handoff/T48b.task.md`) | 16D |

---

## 재측정 대기 — T47 이후

각 항목의 번호는 이 파일 안의 번호이고, 괄호는 [`AG3S_T0T6_LOG.md`](AG3S_T0T6_LOG.md) "재측정 대기" 표에서 이 항목들이
원래 받았던 번호다 (그쪽에서는 옮겼다는 한 줄만 남겼다). T46 이전 항목은 그쪽 표를 본다.

| # | 무엇 | 값 | 상태 |
|---|---|---|---|
| 1 (T0T6 30) | (T47 · T47b 가 추가) **grasp latch 를 포함한 e2e** 에서의 head 비교, cuRobo ESDF backend e2e, T47b source 의 selection record 3D | offline e2e 는 CPU legacy ESDF 이고 grasp latch 가 없다 (`T47.verify.json not_measured`, `T47b.verify.json not_measured`). selection record 3D 는 T47 의 5 source 만 있다 | **측정 안 됨** |
| 2 (T0T6 31) | (T47 · T47b 가 추가) **held-out 의 일반화** — banana · pear 각 1 record 와 same-scene 2 record 뿐이다. `cam_right_wrist` 2D 는 4 record 모두 target < 200 px 라 제외 | record 를 늘리려면 GPU attention pass 가 필요하다 (사용자 승인 사항) | **측정 안 됨** |
| 3 (T0T6 32) | (T47b 가 추가) **σ=1 이 이웃 과일 흡수를 늘리는 경로**, 그리고 B3 수치 시험의 다른 source (L8H2 · oracle top-3 σ=1 에서만 시험) | `identity_correct` 54 → 49 (L8H2, σ=1) 는 측정했으나 smoothing 이 seed 를 번지게 하는지는 분리해 재지 않았다 | **원인 미확정** |
| 4 (T0T6 33) | (T48a 가 추가) **probe 가 "물체를 쥐고 있다" 를 읽는가, "팔이 이 자세에 있다" 를 읽는가** — 같은 팔 자세에서 물체의 유무만 바꾼 image 로 가른다 | T48a 의 C2 는 같은 episode 의 다른 frame 의 image 를 통째로 이식해 팔 자세와 물체 상태가 함께 바뀐다. dataset 은 scripted 라 둘이 같이 움직인다 (frame index 만으로 val 1.000) | **T48b 로 진행 (사용자 판정, 2026-10-12)** — E0 re-render parity 먼저 |
| 5 (T0T6 34) | (T48a 가 추가) **실제 strong rule (`grasp_latch.py` + 실행 feedback) 과의 비교** | T48a 는 offline strong-rule proxy 만 썼다 (`T48.verify.json not_measured`) | **측정 안 됨** |
| 6 (T0T6 35) | (T48a 가 추가) **batch 1 (서빙 batch size) 전체 extract · cuRobo 회귀 기준선 · extract 시작 시점의 pod working set** | extract 는 `--batch 4` (batch parity 는 첫 batch n = 4 에서만), cuRobo 변형은 돌리지 않았고, lead 요청이 extract 도중 도착해 시작 시점 working set 이 없다 (`T48.verify.json not_measured`) | **측정 안 됨** |

---

## 실행 기록

## T47 · T47b — label-free 기준으로 attention head 를 다시 고르면 L8H2 보다 나은가 (2026-10-12 KST)

**이 절이 답하는 물음 (목적).** AG3S 가 attention 을 쓰는 곳은 하나다 — 재구성한 3D cluster 중 어느 것이 **target** 인지 이름을 붙이는 일이다. 지금 그 attention 은 고정된 한 cell (`denoise 0 · agg last · layer 8 · head 2`, 이하 **L8H2**, `AG3S/trajopt/attention_policy.py:37-41`) 에서 온다. 이 cell 은 14D 모델 · `cam_high` 1 대 · 28 frame 에서 `peak_on_target` (attention 의 최대 칸이 대상 위에 떨어진 frame 수) 하나로 골랐고 (`T47.task.md` 의 서술, 2026-09-04 step-01), 지금 운영 조건 (16D 모델 · camera 3 대) 에서는 다시 고르지 않았다. T47 은 논문 Kang et al. (*Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding*, arXiv 2503.06287) 의 **label 이 필요 없는 head 선택 절차** (image attention sum + spatial entropy → selection frequency → 고정 k=3 합산) 를 16D · camera 3 대에 옮겨, 그렇게 다시 고른 head set 이 L8H2 보다 held-out frame 에서 target 을 더 잘 가리키는지 2D 와 3D grounding 에서 잰다. T47b 는 T47 의 실패를 가르는 보강 측정이다 — (a) 같은 head set 의 **합산 방식** (raw 합 대 head 별 mass 정규화) 이 결과를 바꾸는지, (b) 10 cm centroid 기준 (`on_target`) 이 놓치는 이웃 과일 오분리를 GT label 로 재는 지표, (c) raw attention 크기에 절대값 threshold 를 거는 곳이 있는지, (d) 선택된 head 가 왜 실패했는지 (sink 진단).

**한 줄 결과 (before → after).**

- **사전 고정 primary 는 실패했다.** 공통 top-3 = **L4H1 · L2H0 · L14H5** (σ=1, k=3, denoise 0 · agg last, raw 합) 는 held-out 에서 target mass 0.0134 (σ=1) 이고 3D `on_target` **0 / 60** (σ=1, σ=0 모두) 이다. L8H2 는 55 / 60. → **사용자 판정: L8H2 유지, 서빙 cell 은 바꾸지 않는다.**
- **label-free 기준 (attention sum + spatial entropy) 은 action-token query 에서 target head 를 고르지 못했다** (pilot 규모, 사용자 판정으로 확정한 이 STEP 의 답). 선택된 세 head 는 (T47b #5) L4H1 · L2H0 = 어느 camera 에서도 `broad_single_blob` (넓게 퍼진 한 덩어리, top-1 patch 몫 ≈ 0.02), L14H5 = `cam_high` 에서 frame 의 95.8 % 가 같은 patch (r1, c11) 에 몰리는 **fixed-patch sink** (top-1 patch 가 `S_cam` 의 0.669).
- **합산 방식은 결과를 거의 바꾸지 않는다** (T47b B1). oracle top-3 의 `raw_sum` / `mass_norm_joint` / `mass_norm_per_camera` 는 3D `on_target` 55 / 54 / 55 (60 frame), `identity_correct` 49 / 48 / 48.
- **`on_target` 이 참이어도 cluster 가 target 이 아닌 frame 이 있다** (T47b B2). L8H2 (서빙 cell, σ 없음) 도 60 frame 중 `on_target` 55 대 `identity_correct` 54 이고, 이웃 과일이 섞이는 쪽은 **Gaussian smoothing σ=1** 이다: L8H2 에 σ=1 을 걸면 `identity_correct` 49 / 60, chosen cluster 에 든 non-target 점 195 → 2,678.
- **raw amplitude 의 절대값 threshold 는 default 경로에 `1e-12` degenerate guard 하나뿐** (T47b B3, 24 지점 점검). 같은 frame 에 attention 을 ×4 하면 60 / 60 bit-identical, ×3 이면 decision 은 60 / 60 같고 bit 는 0 / 60.

**읽는 법과 출처 규약.**

- 수치는 `T47.verify.json` · `T47b.verify.json` 의 `numbers` 에서만 왔다. 같은 파일 숫자의 덧셈 · 개수 세기는 "(산출)" 로 적었다. figure sidecar (`figures/t47/*.json`) 의 값은 verify.json 이 같은 값을 갖고 있을 때만 옮겼다. 논문의 수치는 이 로그가 측정한 것이 아니어서 옮기지 않았다.
- **해석의 출처를 가른다.** 사용자 판정은 "(사용자 판정)", lead 가 task 문서에 쓴 것은 "(lead)", 측정으로 확인하지 않은 해석은 "(미측정 해석)" 으로 적었다. 적은 해석은 모두 그렇게 표시했다.
- **시각.** pod 의 `date` 는 UTC 다. KST 는 run 디렉터리 이름의 `+0900` 접미사 또는 verify.json 의 UTC 필드에 +9 h 를 더해 환산했다 (파일 시각도 같다).
- **환경.** CPU 만 썼다 (T47 · T47b 모두 `.venv-ag3s`, legacy ESDF, MuJoCo osmesa; T47b verify.json `gpu_used: false`). 회귀 기준선은 둘 다 일치했다 — legacy 14 / 15 · 15 / 15 · frame0 `clearance_before` −29.031048 mm (T47 은 HEAD `bd778e5e` + T46 의 uncommitted 파일, T47b 는 HEAD `218229ac`). cuRobo 기준선은 GPU 를 쓰지 않아 돌리지 않았다.
- **HEAD 와 코드.** T47 은 select · gt · validate 를 HEAD `bd778e5e` 에서, e2e 는 다른 session 이 commit 한 `c906cda2` (T47 verify.json `code_state.head_change_note`: `common/workspace.py` 추가 변경 등, T47 study 가 import 하는 부분은 불변) 에서 돌렸다. T47b 는 HEAD `218229ac` 에서 돌렸고, `pipeline.py` · `config.py` · `curobo_builder.py` · `serve_safe.py` 의 md5 가 T47 e2e import 시점과 **같다** (T47b verify.json `code_state`). 즉 T46 option 은 기본값 (off) 이고 T47 과 T47b 의 3D 는 같은 코드 경로다.

### 1. 사용자 판정과 타임라인 (KST)

| # | 판정 · 사건 (2026-10-12) | 이 절에서 |
|---|---|---|
| 1 | **(사용자 판정)** 논문 Kang et al. 의 head 선택 절차 (A) 와 검증 방법론 (E) 을 먼저 적용한다. SAM 은 쓰지 않는다. entropy 실행 중 신뢰도 (B) · camera 가중 (C) · 명사 token query (D) · sampler loop 수정은 이 STEP 범위가 아니다 | §2 |
| 2 | **(사용자 판정)** 승인: MuJoCo replay (osmesa, CPU). "GPU 가 필요하면 사용해도 되지만 다른 agent 가 쓰는 중일 수 있으니 상대쪽을 종료시키지 않도록 주의" | 이 STEP 은 GPU 를 쓰지 않았다 |
| 3 | **(사용자 판정)** 사전 고정 (pre-registration) — verifier 결과를 lead 가 열기 전에 기록 | §3 |
| 4 | **(lead)** `ep1800` step record 를 storage 에서 찾아 held-out 에 포함. held-out 을 **same-scene** (`headfix` · `ep1800`, apple) 과 **new-scene** (`ep1850` banana · `ep1900` pear) 로 가른다 | §4 |
| 5 | **(사용자 판정, T47 verify.json 확인 후)** **L8H2 유지.** 후속: (1) 실패 원인의 정식 측정 → T47b #5, (2) 명사 token query (D) → `T49.task.md` | §9 |
| 6 | **(사용자 판정)** T47b 범위 변경: B1 의 비교 대상을 oracle top-3 · camera 별 top-3 · head-average 로 교체, sink 진단 (#5) 을 정식 기록 | §6 – §8 |
| 7 | **(lead 보정, T47b.impl.md 확인 후)** T47 의 e2e · class mass 는 σ=0 으로 돌았고 사전 고정은 σ=1 이었다 → T47b 가 σ=1 로 다시 쟀다 | §5 |

| 시각 (KST) | 무엇 | 출처 |
|---|---|---|
| 13:57 | 검토 문서 `doc/reviews/AG3S_TWO_PAPER_REVIEW_20261012_KO.md` 작성 (context) | 문서 머리 |
| 14:14:12 | T47 run 디렉터리 생성 `20261012T141412+0900_bd778e5e3453_d7a51b456f36` | 디렉터리 이름 |
| 14:14:56 – 14:15:16 | GT cache (8 파일) 생성 (gt stage 31.5 s) | T47b `m0_check.gt_cache_mtime_utc` · T47 `gt_stage_seconds_total` |
| 14:20:58 | 다른 session 이 `c906cda2` commit (T47 validate 와 e2e 사이) | T47 `code_state` |
| 15:53:26 | T47b 회귀 기준선 실행 (일치) | T47b `concurrency.baseline_launch` |
| 15:55:12 | T47b e2e (σ=1, 11 source) 시작 | `e2e_s1_launch` |
| 16:00:43 | T47b B3 수치 시험 시작 | `b3_launch` |
| 16:09:45 | T47b 종료 | `concurrency.end` |

### 2. 왜 이 STEP 인가

| | L8H2 를 고른 때 (step-01, 2026-09-04) | 지금 운영 |
|---|---|---|
| model | 14D | **16D** |
| camera | `cam_high` 1 대 | **3 대** (head + 양 wrist, multi-view max fusion) |
| frame | 28 (선택 = 평가가 같은 frame) | 여러 episode · 과일 |
| 기준 | `peak_on_target` 하나 (GT 필요) | — |

(표는 `T47.task.md` 에서 옮겼다. 28 frame 과 14D 는 옛 기록의 서술이며 이 STEP 의 측정이 아니다.)

- T1-a (attention 셀 선택은 16D 에서 성립하는가) 는 원인을 **배선** (offline 이 `zed_left` 에만 attention 을 붙임) 으로 닫았고 판정 8 로 배선을 고쳤다. **head 자체는 16D · 3 camera 에서 다시 고르지 않았다.**
- 논문 근거 (논문 서술, 이 로그의 측정이 아님): 모든 head 평균은 쓸 수 없고, label 없는 두 기준 ① image attention sum ② spatial entropy 와 여러 sample 에 걸친 **selection frequency** 로 고른 **고정 k=3 head 합산**이 1 head 보다 낫다.

**우리에게 맞춘 차이 (논문 그대로가 아닌 곳, `T47.task.md`).**

1. **query 가 다르다.** 논문은 마지막 *text* token, 우리는 *action* (suffix) token 이다. action token 의 attention 이 한 덩어리로 모이는 대상이 target 이 아니라 gripper · 목적지일 수 있다 → 그래서 GT 로 "무엇에 모였나" 를 class 별로 분해했다 (§5-2).
2. **camera 가 3 대.** 논문은 이미지 1 장이다. 기준을 camera 별 16×16 map 에 적용하고 "모든 camera 공통 head set" 과 "camera 별 head set" 을 둘 다 만들었다. 3 camera 를 이어 붙인 768 token 에 2D connectivity 를 적용하지 않는다.
3. attention 값은 renormalize 안 된 **raw softmax mass** 다 (`attention_policy.py:57-58`). npz 의 768 image token 합이 논문의 `S_img` 이다.
4. grid 는 16×16 (논문 LLaVA-1.5 는 24×24), Gaussian smoothing σ 는 grid 단위 {0, 1}.
5. 최종 판정은 **3D end-to-end** 다 — 2D 개선이 3D 개선을 보장하지 않는다.

### 3. 사전 고정 (pre-registration, 사용자 판정 2026-10-12)

verifier 결과를 lead 가 열기 전에 고정했다. 근거: `doc/reviews/AG3S_TWO_PAPER_REVIEW_20261012_KO.md` §3 (보강할 항목) · §6 ("k 나 threshold 를 held-out 에서 선택했다면 그 split 은 최종 test 가 아니다").

| 항목 | **primary (채택 판단 근거)** | exploratory (보고만) |
|---|---|---|
| head set | **모든 camera 공통 top-3** | camera 별 top-3 |
| k | **3** | 1, 2, 4, 5 |
| smoothing σ (grid 단위) | **1** | 0 |
| query | **denoise 0 · agg last** (서빙과 같음) | agg mean / first, denoise 4 / 9 |
| τ (attention sum 의 knee) | **camera 별 `S_cam` · Kneedle** | 전체 `S_img` 기준 |
| 합산 | **raw 합 (논문 재현)** | mass-normalized 변형은 T47b |

- held-out (same-scene 2 + new-scene 2 record) 은 **pilot** 이다 — object generalization 확정 근거가 아니다. frame 을 독립 trial 로 세지 않고 record (episode) 단위로 본다.
- attention source 는 별도 checkpoint 사본 · 고정 noise seed 0 에서 나온 것이다. 서빙 action 을 낸 forward 의 attention 이라고 주장하지 않는다.

### 4. 입력과 구현 (무엇을 어떻게 쟀나)

**입력 (7 record, 전부 기존 storage · 수정 없음).** attention npz 의 축 = `(frame, denoise[0,4,9], agg[mean,first,last], layer 18, head 8, camera 3, 16, 16)`.

| record | frame | episode | target | 역할 |
|---|---:|---:|---|---|
| `long` | 50 | 1800 | apple | selection |
| `t2` | 75 | 1800 | apple | selection |
| `ep1807` | 75 | 1807 | apple | selection |
| `headfix` | 15 | 1800 | apple | held-out **same-scene** |
| `ep1800` | 15 | 1800 | apple | held-out **same-scene** (step record 는 `recordings/run_16d_ep1800`, 디렉터리 이름만 `run_0000 → run_16d_ep1800` 로 바뀜) |
| `ep1850` | 15 | 1850 | banana | held-out **new-scene** |
| `ep1900` | 15 | 1900 | pear | held-out **new-scene** |

(`T47.verify.json` `gt_selfcheck`, held-out 4 record = 60 frame.) **독립성 주의:** same-scene held-out 2 개는 selection 의 `long` · `t2` 와 같은 초기 scene (episode 1800) 에서 출발한다. 일반화 판단의 주 근거는 new-scene 2 record 이고, 그 둘도 record 단위 pilot 이다.

**구현 (전부 새 파일, 기존 파일 무수정; `T47.impl.md` · `T47b.impl.md`).**

| 구성 | 정의 |
|---|---|
| `S_cam` · `S_img` | camera 별 16×16 attention 합 (`S_cam`) 과 3 camera 768 token 합 (`S_img`) |
| spatial entropy | `ReLU(A − mean A) > 0` → **8-connectivity** 성분 → `P(C) = |C|/Σ|C|` (pixel 수) → `H = −Σ P ln P` (nats). 성분 0 개 (상수 map) 는 NaN → 선택 불가 |
| τ | 오름차순 정렬 곡선의 knee. primary = **Kneedle**, variant = 이산 curvature. sample (frame, camera) 마다 그 sample 의 128 cell `S_cam` 으로 계산. eligible = `S ≥ τ` |
| selection frequency | sample 마다 (eligible ∧ entropy 유한) 중 entropy 최저 10 개를 고르고 cell 별 선택 비율. L0 · L1 제외 → cell = L2–17 × h0–7 = 128 |
| top-k 합산 | cell 별 Gaussian smoothing (`gaussian_filter`, reflect, map 마다 질량 보존) 후 element-wise raw 합 |
| GT | MuJoCo replay (osmesa) 로 frame × camera 별 body 의 16×16 patch coverage 를 한 번 만들어 cache (`gt_cache/*.npz`). class = target · destination · other_object · robot_gripper · robot · support · background |
| 2D 지표 | `target_mass` (attention 질량 중 target 칸의 몫), `lift` (= target mass ÷ target 면적 몫), `peak_on_target`, IoU (평균으로 binarize 한 map 대 GT target patch mask, coverage ≥ 0.5 와 ≥ 0.25). target 이 한 camera 에서 `< 200 px` 인 (frame, camera) 는 2D 에서 제외 |
| 3D 지표 | live 서빙 경로 (`wire.pack_request → unpack_camera_observations(attention=…) → AG3S.process_multi_debug`) 를 그대로 조립, attention source 만 바꿈. `has_target`, **`on_target`** (grounding centroid 가 GT target body 원점에서 0.10 m 안, T1-a `on_apple` 정의), **silent failure** (`has_target ∧ ¬on_target`), seed precision / recall (GT seg 기준), chosen cluster 의 target 점 비율 |

**GT replay 자가 점검** (`T47.verify.json` `gt_selfcheck`). replay RGB 대 기록 RGB: 7 record 의 pixel 평균 절대차 (0–255) 최대 0.2334, 채널 최대 차 > 16 인 pixel 비율 최대 0.00466. held-out 에서 target 이 ≥ 200 px 로 보이는 frame 수: `cam_left_wrist` 는 headfix 15 · ep1850 15 · ep1900 15 · ep1800 13, `cam_high` 는 11 · 9 · 12 · 0, **`cam_right_wrist` 는 4 record 모두 0** (60 frame 전부 제외). 그래서 held-out 2D 지표는 `cam_left_wrist` 와 `cam_high` 만 본다 (n = 90 / 180 sample 유지, 제외 90). 3D fusion 은 세 camera 를 다 쓴다.

**e2e 의 한계 (T47.impl.md).** offline e2e 에는 grasp latch 가 없다 — 파지 후 frame 의 target 고정 (TargetConfirm freeze) 이 재현되지 않는다. 그래서 3D 결과는 pre-grasp / post-grasp 로 나눠 읽고, post-grasp 실패 수를 서빙 성능으로 읽지 않는다.

### 5. T47 결과

#### 5-1. 선택 (select) — GT 없이 selection 3 record 200 frame

![정렬 S_cam 곡선과 τ](figures/t47/t47-sorted-S-tau.png)

[`t47-sorted-S-tau.png`](figures/t47/t47-sorted-S-tau.png) — camera 별 정렬 `S_cam` 곡선과 knee τ (논문 Fig.3 대응). 3 camera 각각 200 sample, d0 · last.

| camera | τ (Kneedle) 평균 | τ 중앙 | eligible cell 평균 (/128) | knee 찾음 비율 |
|---|---:|---:|---:|---:|
| cam_high | 0.1106 | 0.1071 | 19.285 | 1.0 |
| cam_left_wrist | 0.1937 | 0.2062 | 27.43 | 1.0 |
| cam_right_wrist | 0.1492 | 0.1466 | 21.565 | 1.0 |
| (variant) 전체 `S_img` 기준 | 0.2847 | 0.2967 | 50.0 | 1.0 |

![selection frequency 와 rank](figures/t47/t47-selection-frequency-rank.png)

[`t47-selection-frequency-rank.png`](figures/t47/t47-selection-frequency-rank.png) — 공통 selection frequency 상위 cell 과 camera 별 상위 10 (논문 Fig.6 대응).

| 순위 | cell | 공통 selection frequency (600 sample = 200 frame × 3 camera) |
|---:|---|---:|
| 1 | **L4H1** | 0.6183 |
| 2 | **L2H0** | 0.4750 |
| 3 | **L14H5** | 0.4383 |
| 4 | L3H3 | 0.4333 |
| 5 | L4H6 | 0.3967 |
| 6 | L4H7 | 0.3600 |
| 7 | L4H2 | 0.3450 |
| 8 | L2H3 | 0.3083 |
| 9 | L3H1 | 0.2667 |
| 10 | **L8H2** | 0.2600 |

- 공통 top-3 (primary) = **L4H1 · L2H0 · L14H5**. 후보가 하나도 없는 sample 은 0 개. L8H2 의 공통 순위는 10 (0.26).
- camera 별 top-3: `cam_high` = L4H7 · L3H3 · L4H1 (L8H2 순위 18, 0.17), `cam_left_wrist` = L15H2 · L9H6 · L9H2 (L8H2 순위 4, 0.50), `cam_right_wrist` = L4H1 · L4H6 · L2H3 (L8H2 순위 19, 0.11).
- **selection 의 안정성.** record 하나를 빼고 다시 고른 (leave-one-record-out) 공통 top-3 가 전체 top-3 와 같은 집합인 횟수는 d0 · last 에서 **0 / 3** (각각 2 개만 겹침). agg 변형은 d0 · mean 3 / 3 (공통 top-3 = L4H2 · L4H1 · L2H0), d0 · first 1 / 3 (L4H1 · L2H0 · L5H2).
- τ 정의를 curvature 로 바꾸면 공통 top-3 = L4H1 · L14H5 · L3H3. 전체 `S_img` 기준 (exploratory) 의 공통 top-3 = **L4H1 · L4H6 · L8H2** (L8H2 포함).
- (산출) 같은 계산을 denoise 4 · 9 에도 해 두었고 (`selection` 의 9 setting) 공통 top-3 가 setting 마다 달랐다. 예: d4 · last = L4H1 · L14H5 · L4H6, d9 · last = L3H3 · L14H5 · L13H3.

**GT-oracle (상한 참고용, 채택 후보 아님).** selection record 의 GT 로 `peak_on_target` (3 camera pooled) 순위를 매긴 d0 · last 의 top-1 = L8H3, **top-3 = L8H3 · L10H7 · L12H4**. 이 세 head 는 L8H2 가 아니고, 공통 top-3 와도 겹치지 않는다.

#### 5-2. 2D 검증 (held-out) — 무엇에 모였나

![class 별 mass 분해](figures/t47/t47-class-mass.png)

[`t47-class-mass.png`](figures/t47/t47-class-mass.png) — source 별 attention 질량의 class 분해 stacked bar (same-scene / new-scene × pooled / camera 별). **이 그림은 σ=0 으로 만들었다** (sidecar `sigma: 0`).

held-out 2D pooled (90 sample), d0 · last, `target_mass` · `lift` · `peak_on_target` · IoU@0.5 · IoU@0.25 (`T47.verify.json` `validate_2d`):

| source | target mass | lift | peak_on_target | IoU@0.5 | IoU@0.25 |
|---|---:|---:|---:|---:|---:|
| **L8H2 (서빙)** | **0.1498** | **59.60** | 0.689 | 0.091 | 0.173 |
| 공통 top-3, σ=1 (**primary**) | 0.0134 | 1.19 | 0.111 | 0.041 | 0.039 |
| 공통 top-3, σ=0 | 0.0143 | 1.44 | 0.022 | 0.041 | 0.048 |
| camera 별 top-3, σ=0 | 0.0225 | 4.06 | 0.011 | 0.024 | 0.035 |
| camera 별 top-3, σ=1 | 0.0260 | 3.04 | 0.089 | 0.039 | 0.046 |
| head-average (L2–17, σ=0) | 0.0400 | 6.25 | 0.322 | 0.039 | 0.046 |
| uniform (면적) | 0.0112 | 1.00 | 0.000 | 0.000 | 0.000 |
| GT-oracle top-3, σ=0 | 0.1021 | 22.78 | 0.878 | 0.050 | 0.062 |
| (exploratory) 전체 `S_img` 기준 공통 top-3, σ=1, k=3 | 0.0696 | 8.41 | 0.411 | 0.080 | 0.088 |

- 공통 top-3 (σ=1) 의 target mass 는 same-scene 0.0155 · new-scene 0.0118 이다 (pooled sample 기준). uniform 의 0.0112 에 가깝다.
- class 분해 (pooled, `class_mass`): L8H2 = target 0.150 · **support 0.705** · robot_gripper 0.056 · destination 0.021. 공통 top-3 (σ=1) = target 0.013 · destination 0.084 · robot_gripper 0.042 · robot 0.066 · support 0.179 · **background 0.609**. camera 별 top-3 (σ=0) = target 0.022 · **robot_gripper 0.184** · background 0.357. 즉 선택된 공통 top-3 는 gripper 나 목적지보다 **background 에** 질량이 갔고, camera 별 top-3 는 gripper 에 질량이 상대적으로 많이 갔다. (L8H2 의 질량이 support 에 0.705 있다는 점은 측정된 사실이다; 그 의미는 이 STEP 이 분석하지 않았다.)
- **rank 가 held-out 지표를 얼마나 예측하나** (Spearman ρ, 128 cell, selection frequency 대 held-out pooled, 공통 head set, `cam` τ, d0 · last, all held-out): `lift` ρ 0.233 (p 0.0081), `peak_on_target` 0.345 (p 0.0001), `target_mass` 0.210 (p 0.0174), IoU@0.5 0.142 (p 0.111), IoU@0.25 0.169 (p 0.0566). same-scene 과 new-scene 으로 가르면 `lift` 0.231 / 0.237, IoU@0.5 0.086 (p 0.336) / 0.202 (p 0.0225). 양의 상관이 있으나 약하다. 다중비교 보정은 하지 않았다.

![held-out scene 의 attention overlay](figures/t47/t47-scene-overlay.png)

[`t47-scene-overlay.png`](figures/t47/t47-scene-overlay.png) — held-out frame (`headfix` frame 1 · 5, `ep1850` frame 1 · 6, `ep1900` frame 1 · 5; transit 과 approach) 의 camera 3 대 RGB 위에 L8H2 · 공통 top-3 · camera 별 top-3 와 GT target 윤곽을 겹친 실제 scene (σ=0). 예: `headfix` frame 1 `cam_high` 에서 L8H2 의 target mass 는 0.2712, 공통 top-3 는 0.0041.

#### 5-3. 3D end-to-end (held-out, σ=**0**) — T47 의 e2e 는 사전 고정과 다른 σ 로 돌았다

**편차 기록.** T47 의 e2e 와 위 class mass figure 는 σ=0 으로 돌았다 (`summary__heldout.json` `args.sigma = 0`, T47 CLI 기본값). 사전 고정 primary 는 σ=1 이었다. 이 편차는 lead 가 T47b.impl.md 확인 후 발견했고 σ=1 은 T47b 가 다시 쟀다 (§6). 아래 표는 **σ=0** 이다. 2D 의 `validate_2d` 는 σ=0, 1 둘 다 있다 (§5-2 표).

![T47 3D e2e 표](figures/t47/t47-e2e-table.png)

[`t47-e2e-table.png`](figures/t47/t47-e2e-table.png) — source × 지표 표 (σ=0).

| source (σ=0) | scene (n=30) | has_target | **on_target** | silent failure | seed precision | seed recall | chosen target 점 비율 |
|---|---|---:|---:|---:|---:|---:|---:|
| **L8H2** | same | 30 | **26** | 4 | 0.393 | 0.943 | 0.761 |
| | new | 30 | **29** | 1 | 0.339 | 0.785 | 0.808 |
| 공통 top-3 | same | 27 | **0** | 27 | 0.004 | 0.002 | 0.000 |
| | new | 30 | **0** | 30 | 0.000 | 0.000 | 0.000 |
| camera 별 top-3 | same | 29 | 24 | 5 | 0.490 | 0.405 | 0.650 |
| | new | 30 | 29 | 1 | 0.315 | 0.410 | 0.608 |
| head-average | same | 30 | 26 | 4 | 0.859 | 0.850 | 0.761 |
| | new | 30 | 29 | 1 | 0.653 | 0.631 | 0.608 |
| GT-oracle top-3 | same | 30 | 26 | 4 | 0.595 | 0.972 | 0.762 |
| | new | 30 | 29 | 1 | 0.529 | 0.802 | 0.808 |

- held-out 전체 (60 frame) `on_target`: L8H2 **55**, 공통 top-3 **0**, camera 별 53, head-average 55, GT-oracle 55. silent failure 는 5 · 57 · 6 · 5 · 5.
- **record 별 `on_target` (15 frame 중, held-out)**: L8H2 · head-average = ep1800 11 · ep1850 14 · ep1900 15 · headfix 15. camera 별 = 10 · 14 · 15 · 14. 공통 top-3 = 전부 0.
- L8H2 · head-average · GT-oracle 가 `on_target` 에서 같은 값인 것은 4 개 record 가 pilot 이라 변별력이 작다는 뜻이다 (record 단위 4 개). 차이가 나는 곳은 **seed precision / recall** 이다: held-out 전체에서 head-average 는 precision 0.756 · recall 0.740, L8H2 는 0.366 · 0.864 (`by_source.*.all_heldout`).
- phase: L8H2 same-scene 은 pre-grasp 14 / 18 · post-grasp 12 / 12, new-scene 은 19 / 19 · 10 / 11.
- **selection record 3D (200 frame, 참고값, GT-oracle 제외)** `on_target`: L8H2 146 (silent 54: pre-grasp 4 · post-grasp 50), 공통 top-3 **0** (silent 200), camera 별 77 (silent 119), head-average 135 (silent 65). 파지 후 frame (136) 에서 L8H2 86 · head-average 75 · camera 별 19 이다. offline e2e 에 grasp latch 가 없다는 점 (§4) 을 함께 읽는다.

**이 STEP 의 답 (사용자 판정, pilot 규모).** label-free 기준 (attention sum + spatial entropy, k=3 고정 합산) 은 action-token query 에서 target head 를 고르지 못했다. 사전 고정 primary 는 target mass 가 uniform 수준이고 3D `on_target` 이 0 이다. head 교체 (L8H2 → 다른 head) 는 하지 않는다.

### 6. T47b — σ=1 로 primary 를 다시 쟀다 (정식 값)

lead 보정 (§1 #7): T47b verifier 가 **공통 top-3 (raw 합) 를 σ=1 로 2D · 3D 다시 쟀다.** 이 값이 T47 primary 의 정식 값이고 σ=0 값과 나란히 둔다. L8H2 와 head-average 도 같은 σ 두 값으로 낸다. 검증으로 T47b 의 `--sigma 0` 실행이 T47 의 5 source × 60 frame 을 모든 필드에서 재현했다 (parity 60 / 60 × 5; 독립 재계산도 일치).

| source | σ | 2D target mass | 2D lift | 2D peak | IoU@0.5 | IoU@0.25 | 3D 같은 scene on / has / silent (n=30) | 3D new-scene on / has / silent (n=30) |
|---|---|---:|---:|---:|---:|---:|---|---|
| 공통 top-3 (primary) | **1** | 0.0134 | 1.19 | 0.111 | 0.041 | 0.039 | **0** / 8 / 8 | **0** / 28 / 28 |
| | 0 | 0.0143 | 1.44 | 0.022 | 0.041 | 0.048 | 0 / 27 / 27 | 0 / 30 / 30 |
| L8H2 | σ 없음 (서빙 raw cell) | 0.1498 | 59.60 | 0.689 | 0.091 | 0.173 | 26 / 30 / 4 | 29 / 30 / 1 |
| | 1 | 0.1015 | 20.07 | 0.722 | 0.076 | 0.094 | 26 / 30 / 4 | 29 / 30 / 1 |
| head-average | 1 | 0.0331 | 3.61 | 0.378 | 0.048 | 0.050 | 26 / 30 / 4 | 29 / 30 / 1 |
| | 0 | 0.0400 | 6.25 | 0.322 | 0.039 | 0.046 | 26 / 30 / 4 | 29 / 30 / 1 |

- 공통 top-3 의 `on_target` 은 σ=1 · σ=0 모두 **0** 이다. σ=1 에서는 has_target 자체가 줄었다 (same-scene 8 / 30; σ=0 에서는 27 / 30) — 그 이유는 측정하지 않았다.
- 같은 L8H2 에 σ=1 을 걸면 2D lift 59.60 → 20.07, 3D `on_target` 은 같지만 `identity_correct` 가 same-scene 26 → 23, new-scene 28 → 26 으로 줄었다 (§8).
- (T47b 의 `l8h2` 는 CLI 에서 언제나 unsmoothed cell 이다. σ=1 L8H2 는 verifier 가 `combine_heads(block, [(8, 2)], 1, raw_sum)` 로 같은 `run_b2_record` 경로에서 따로 돌렸고, σ=0 에서 CLI l8h2 row 를 `ep1850` 15 / 15 로 재현함으로 대조했다.)

### 7. T47b #5 — sink 진단: 선택된 head 가 어떤 head 인가

사용자 판정으로 정식 기록한다. 128 cell × camera 3 대 마다 d0 · last, 7 record 전 260 frame 에서 다섯 수치를 쟀다. 용어: **fixed-patch sink** = 입력이 바뀌어도 늘 같은 patch 에 질량이 몰리는 head. **broad single blob** = 질량이 넓게 퍼져 있어 평균 위 pixel 이 큰 한 덩어리를 이루는 head.

| | 정의 (frame 평균) |
|---|---|
| (a) `modal_fraction` | frame 별 argmax patch 중 가장 흔한 patch 의 frame 비율 |
| (b) `top1_share` | top-1 patch 값 ÷ `S_cam` |
| (c) component | 평균 binarize 후 8-connectivity component 수, 최대 component ÷ 256 (grid) |
| (d) entropy | spatial entropy (nats) |
| (e) `S_cam` | camera 질량 |

![sink 진단 scatter](figures/t47/t47b-sink-scatter.png)

[`t47b-sink-scatter.png`](figures/t47/t47b-sink-scatter.png) — 위: camera 별 (b) top-1 patch 몫 대 (c) 최대 component 의 grid 몫 scatter (cell 마다 한 점, 공통 top-3 · L8H2 · oracle top-3 강조). 아래: 대표 head 의 frame 평균 map (x = 최빈 argmax patch).

선택된 head 와 비교 head 의 값 (`T47b.verify.json` `m5_sink.marked_cells`):

| cell | camera | (a) modal fraction | 최빈 patch (r, c) | (b) top-1 몫 | (c) 최대 component / 256 | (d) entropy | (e) `S_cam` | descriptive label |
|---|---|---:|---|---:|---:|---:|---:|---|
| L4H1 | cam_high | 0.200 | (6, 4) | 0.0203 | 0.181 | 1.425 | 0.137 | broad_single_blob |
| L4H1 | cam_left_wrist | 0.135 | (3, 0) | 0.0174 | 0.186 | 1.434 | 0.149 | broad_single_blob |
| L4H1 | cam_right_wrist | 0.515 | (13, 15) | 0.0279 | 0.243 | 0.989 | 0.344 | broad_single_blob |
| L2H0 | cam_high | 0.454 | (15, 3) | 0.0164 | 0.201 | 1.464 | 0.156 | broad_single_blob |
| L2H0 | cam_left_wrist | 0.585 | (14, 1) | 0.0184 | 0.166 | 1.706 | 0.174 | broad_single_blob |
| L2H0 | cam_right_wrist | 0.269 | (0, 13) | 0.0121 | 0.232 | 1.338 | 0.281 | broad_single_blob |
| **L14H5** | **cam_high** | **0.958** | **(1, 11)** | **0.669** | 0.012 | 1.486 | 0.245 | **fixed_patch_sink** |
| L14H5 | cam_left_wrist | 0.223 | (1, 10) | 0.259 | 0.064 | 1.636 | 0.105 | other |
| L14H5 | cam_right_wrist | 0.623 | (2, 11) | 0.250 | 0.020 | 1.358 | 0.329 | other |
| L8H2 (서빙) | cam_high | 0.215 | (10, 5) | 0.336 | 0.072 | 0.918 | 0.068 | other |
| L8H2 | cam_left_wrist | 0.092 | (14, 9) | 0.262 | 0.064 | 0.773 | 0.631 | other |
| L8H2 | cam_right_wrist | 0.331 | (15, 7) | 0.343 | 0.045 | 1.787 | 0.077 | other |

- **L14H5 `cam_high`** 는 frame 의 95.8 % 가 같은 patch (r1, c11) 에서 최대이고 top-1 patch 가 camera 질량의 0.669 를 가진다 (lead 의 빠른 점검 — 최빈 patch 96 %, top-1 0.67 — 을 verifier 가 독립 재계산으로 확인했고, 독립 재계산은 7 수치 모두 최대 절대차 0, 최빈 patch 384 / 384 일치).
- **L4H1 · L2H0** 는 세 camera 모두 top-1 patch 몫이 0.012 – 0.028 로 낮고, 최대 component 가 grid 의 0.166 – 0.243 을 차지한다 — 질량이 넓게 퍼진 한 덩어리다.
- oracle top-3 (L8H3 · L10H7 · L12H4) 는 세 head × 세 camera 모두 `other` (sink 도 broad blob 도 아님). L8H2 도 `other` 이고 그 (b) top-1 몫이 0.26 – 0.34 로 공통 top-3 의 L4H1 · L2H0 보다 크다.
- descriptive label 의 threshold (sink: modal ≥ 0.8 ∧ top-1 ≥ 0.3, blob: 최대 component ≥ grid 의 0.15) 는 implementer 가 읽기용으로 정했다. **무엇을 고르는 데도 쓰지 않았다.**
- 128 cell 의 camera 별 분포: modal fraction ≥ 0.8 인 cell 이 `cam_high` 12 · `cam_left_wrist` 12 · `cam_right_wrist` 23 개다. 값이 0 인 map (frame × cell × camera) 은 99,840 중 1,389 개.
- (미측정 해석) 두 기준 (attention sum, spatial entropy) 이 이 두 종류를 뽑는 이유는 측정하지 않았다. 측정된 것은 공통 top-3 가 "sink 1 + broad blob 2" 라는 사실과 그 head 들의 target mass 가 uniform 수준이라는 사실 (§5-2) 이다.

### 8. T47b B1 · B2 — 합산 방식과 label 기반 3D 지표

**B1 (합산 방식, σ=1, k=3, d0 · last).** 같은 head set 의 세 합산 — `raw_sum` (논문 합, T47 과 bit 단위 동일), `mass_norm_joint` (head 마다 3 camera 768 token 질량으로 나눔 — camera 간 비율 보존), `mass_norm_per_camera` (head · camera 마다 질량으로 나눔 — camera 간 비율 소멸) — 을 oracle top-3 (`mass_norm_joint` 의 사전 선언 secondary 가 이쪽으로 옮겨짐, 사용자 판정 §1 #6) · camera 별 top-3 · head-average 에 적용했다. 공통 top-3 는 `raw_sum` 1 줄만 둔다.

**B2 (label 기반 3D 지표).** chosen cluster = `ConstraintSet.target.point_indices` (indices 가 이 cloud 를 가리키지 않으면 5 mm proximity 로 대신). 점마다 GT label 은 세 camera GT depth 점 중 3 mm 안 최근접 점의 label (T47 e2e 의 labeller).

| 지표 | 정의 |
|---|---|
| **`identity_correct`** | chosen cluster 의 label 이 있는 점 중 **엄격히 가장 많은 class 가 target** 인가 (동점이면 거짓). centroid 가 아니라 label 을 본다 |
| `identity_failure` | `has_target ∧ ¬identity_correct` (silent failure 의 label 판) |
| `on_target_not_identity` | `on_target` 이 참인데 `identity_correct` 가 거짓인 frame 수 |
| **`nontarget_seed_loss`** | chosen cluster 에 든 GT non-target 점 수 (other_object · destination · robot · table …). T44 의 "distractor seed loss" 와 **같은 정의가 아니다** (T44 는 TSDF seed voxel · voxel 다수결 label, T47b 는 cloud 점 · 최근접 depth 점 label — 숫자를 직접 비교하지 않는다, T47b.impl.md 대조표) |
| **`target_residual`** | GT target 의 관측 점 중 chosen cluster 에 못 든 수와 그 비율 |

(verify.json 의 `nontarget_seed_loss_sum` · `target_residual_*` 는 구현 주 지표 층 (chosen cluster) 의 값이고 seed 층은 `seedlayer_*` 로 따로 있다.)

#### 8-1. 합산 방식 × source (held-out 60 frame 전체)

![합산 방식 × source 의 identity_correct 와 non-target 점](figures/t47/t47b-identity-nontarget.png)

[`t47b-identity-nontarget.png`](figures/t47/t47b-identity-nontarget.png) — 왼쪽 열: record 별 `identity_correct` 비율, 오른쪽 열: record 별 chosen cluster 의 `nontarget_seed_loss` / frame. 행: 전체 · pre-grasp · post-grasp. 속이 빈 점 = same-scene (headfix · ep1800), 채운 점 = new-scene (ep1850 · ep1900). 한 점이 record 하나다 (frame 을 독립 trial 로 세지 않는다).

| source (σ) | has_target | **on_target** | **identity_correct** | silent failure | `on_target_not_identity` | non-target 점 (other_object / destination) | target 잔여 (잔여 ÷ 관측) |
|---|---:|---:|---:|---:|---:|---|---|
| **L8H2 (서빙, σ 없음)** | 60 | 55 | **54** | 5 | 1 | 195 (0 / 195) | 273 / 4609 = 0.0592 |
| L8H2, σ=1 | 60 | 55 | 49 | 5 | 6 | 2,678 (2,678 / 0) | 233 / 4609 = 0.0506 |
| 공통 top-3, σ=1 (raw) | 36 | 0 | 0 | 36 | 0 | 4,427 (4,323 / 104) | 4609 / 4609 = 1.000 |
| 공통 top-3, σ=0 (raw) | 57 | 0 | 0 | 57 | 0 | 6,967 (6,967 / 0) | 1.000 |
| oracle top-3, σ=1 · `raw_sum` | 60 | 55 | 49 | 5 | 6 | 2,449 (2,254 / 195) | 227 / 4609 = 0.0493 |
| oracle top-3, σ=1 · `mass_norm_joint` | 60 | 54 | 48 | 6 | 6 | 2,498 (2,254 / 244) | 308 / 4609 = 0.0668 |
| oracle top-3, σ=1 · `mass_norm_per_camera` | 60 | 55 | 48 | 5 | 7 | 2,563 (2,563 / 0) | 286 / 4609 = 0.0621 |
| oracle top-3, σ=0 · `raw_sum` | 60 | 55 | 54 | 5 | 1 | 251 (251 / 0) | 271 / 4609 = 0.0588 |
| camera 별 top-3, σ=1 · `raw_sum` | 47 | 41 | 31 | 6 | 10 | 2,162 (1,677 / 485) | 2046 / 4609 = 0.4439 |
| camera 별 top-3, σ=1 · `mass_norm_joint` | 52 | 46 | 33 | 6 | 13 | 2,155 (1,783 / 372) | 1995 / 4609 = 0.4328 |
| camera 별 top-3, σ=1 · `mass_norm_per_camera` | 52 | 47 | 40 | 5 | 7 | 2,074 (2,074 / 0) | 1214 / 4609 = 0.2634 |
| camera 별 top-3, σ=0 · `raw_sum` | 59 | 53 | 50 | 6 | 3 | 1,709 (1,313 / 396) | 525 / 4609 = 0.1139 |
| head-average, σ=1 · `raw_sum` | 60 | 55 | 48 | 5 | 7 | 2,547 (2,547 / 0) | 295 / 4609 = 0.0640 |
| head-average, σ=1 · `mass_norm_joint` | 60 | 55 | 49 | 5 | 6 | 2,470 (2,470 / 0) | 306 / 4609 = 0.0664 |
| head-average, σ=1 · `mass_norm_per_camera` | 60 | 55 | 49 | 5 | 6 | 2,487 (2,487 / 0) | 243 / 4609 = 0.0527 |
| head-average, σ=0 · `raw_sum` | 60 | 55 | 52 | 5 | 3 | 1,506 (1,311 / 195) | 331 / 4609 = 0.0718 |

(전부 `T47b.verify.json` `m3_m4_3d_heldout_full` `all|all` 의 합이다. 이 표의 모든 `identity_not_on_target` 은 0 이다.)

관찰된 사실:

- **합산 방식은 3D 결과를 거의 바꾸지 않는다.** oracle top-3 의 세 합산 사이에서 `on_target` 54 – 55, `identity_correct` 48 – 49, silent failure 5 – 6 이다. 2D 도 target mass 0.0603 – 0.0611, lift 8.74 – 8.92 로 같다. camera 별 top-3 의 `mass_norm_per_camera` 는 `identity_correct` 40 (raw 31) 으로 달랐으나 head-average 와 oracle 에서는 같은 방향의 개선이 없다.
- **합산 방식은 camera 선택은 바꾼다.** 같은 frame 에서 각 합산 방식의 "질량이 가장 큰 camera" 가 `raw_sum` 의 것과 같은 frame 비율 (record 별 최소 – 최대, σ=1): oracle top-3 의 `mass_norm_joint` 0.6 – 1.0, `mass_norm_per_camera` 0.133 – 0.733; camera 별 top-3 는 `mass_norm_joint` 1.0, `mass_norm_per_camera` 0.933 – 1.0; head-average 는 `mass_norm_joint` 0.267 – 0.6, `mass_norm_per_camera` 0 – 0.067. 즉 mass 정규화는 camera 선택을 실제로 바꾸지만 (특히 head-average), 이번 pilot 의 3D `on_target` · `identity_correct` 는 그에 따라 달라지지 않았다.
- head 하나가 `raw_sum` 을 지배하는 정도 (dominance 평균, record 별): 공통 top-3 0.357 – 0.378 (`headfix` 에서 세 head 의 합 몫 0.337 · 0.334 · 0.330), oracle top-3 0.437 – 0.464 (`headfix` 에서 head 별 몫 0.414 · 0.194 · 0.392). 질량이 0 인 head 는 없었다.
- **σ=1 smoothing 은 이웃 과일 흡수를 늘린다.** L8H2: `identity_correct` 54 → 49, other_object 점 0 → 2,678. oracle top-3: 54 → 49 (`raw_sum`), other_object 251 → 2,254. head-average: 52 → 48, other_object 1,311 → 2,547. 이 σ 의 효과는 사전 고정 primary 가 σ=1 이었다는 사실과 별개로 측정된 것이다 (primary 는 어느 σ 에서도 `on_target` 0 이다).
- (미측정 해석) σ=1 이 cluster 를 넓히는 경로 (smoothing 이 seed 를 이웃 과일 쪽으로 번지게 하는지) 는 이 STEP 이 분리해서 재지 않았다.

#### 8-2. `on_target` 은 참이고 `identity_correct` 는 거짓인 frame

![on_target 은 참, identity_correct 는 거짓인 frame](figures/t47/t47b-scene-identity.png)

[`t47b-scene-identity.png`](figures/t47/t47b-scene-identity.png) — chosen cluster 의 점을 GT label 색으로 그린 실제 scene (왼쪽: wrist camera 의 GT class map 위 투영, 가운데: filtered cloud top view, 오른쪽: side view, 점선 = 10 cm `on_target` 반경). 두 사례 (σ=1):

| 사례 | centroid 오차 | chosen cluster (target / other_object) | 무엇이 보이나 |
|---|---:|---|---|
| `headfix` frame 0 (transit) · camera 별 top-3 · `raw_sum` | 0.094 m | **0 / 67** | cluster 가 전부 이웃 과일인데 centroid 가 target 원점 10 cm 안에 들어 `on_target` 이 참 |
| `ep1900` frame 13 (grasp) · oracle top-3 · `raw_sum` | 0.035 m | 71 / 85 | target 점도 있으나 이웃 과일 점이 더 많아 plurality 가 other_object |

(두 사례는 verifier 가 `on_target ∧ ¬identity_correct` frame 목록에서 고른 것이다. 목록의 범주는 `chosen_n=0` (`ep1850` frame 13 grasp, proximity 로 cloud 점 0 개) 과 "cluster plurality 가 other_object" 두 가지다.)

**`on_target` 과 `identity_correct` 가 갈린다.** `on_target_not_identity` 는 L8H2 σ=1 6 · oracle top-3 σ=1 6 – 7 · camera 별 top-3 σ=1 7 – 13 · head-average σ=1 6 – 7 (60 frame 기준) 이고, 그 반대 방향 (`identity_not_on_target`) 은 모든 source 에서 0 이다. centroid 10 cm 기준은 이웃 과일 오분리를 놓친다 (lead 의 가설 — 10 cm centroid 만으로 작은 이웃 과일 오분리를 가릴 수 없다 — 이 이번 pilot 에서 수치로 확인됐다).

### 9. T47b B3 — raw amplitude 에 절대값 threshold 가 있는가

attention 이 lifting 에서 grounding 까지 가는 경로에서 raw (정규화 전) 값을 읽는 곳을 file:line 으로 점검했다 (24 지점, `moved_or_missing` 0 — T46 이 줄을 옮기지 않았다).

![B3 점검표](figures/t47/t47b-b3-audit-table.png)

[`t47b-b3-audit-table.png`](figures/t47/t47b-b3-audit-table.png) — 24 지점 × (file:line · 읽는 값 · 절대 상수 · default 여부 · ×k · camera 별 · head 재가중 · 수치 시험).

- **raw 를 읽는 지점은 10 곳이다** (L1 · L2 · L3 · L4 · L5 · L6 · M2 · M3 · G2 · P1; M2 = `multiview.py:384` camera 간 raw max fusion, G2 = `target_grounding.py:1672` raw argmax peak). 그중 **default 로 켜진 절대 상수는 L3 하나**다 — `attention_lifting.py:278`, percentile 폭 `hi − lo ≤ 1e-12` (`_EPS`, `:33`), 상수 map 만 걸리는 degenerate guard. L5 (softmax temperature 1.0) 와 L6 (`normalization == "none"`) 은 default off 다.
- **절대 threshold 가 정규화된 값 · score 에 걸린 곳**: `target_switch_min_score` 0.1 (G8 `target_grounding.py:1198`, D1 `admissibility.py:312`), split ratio 2.0 (S1, 비율). 이들은 ×k 에 불변이지만 head 재가중으로 정규화 분포의 모양이 바뀌면 score 수준이 움직여 threshold 통과 여부가 달라질 수 있다 (`head 재가중` 열 = variant). **head set 을 바꿀 때 다시 볼 곳이다.**
- camera 간 상대 크기는 M2 → (M3, G2) 로 들어간다 — 그래서 `mass_norm_per_camera` 처럼 camera 간 비율을 바꾸는 합산은 camera 선택을 바꾼다 (§8-1 의 camera 선택 일치 비율).
- verifier 가 점검표에 없는 지점을 하나 더 찾았다: `attention_split.py:272` `elif rest <= _EPS` (별도 행으로 올리지 않음).

**수치 시험 (B3 numeric).** held-out 60 frame × source 2 (L8H2 · oracle top-3 σ=1 `raw_sum`), 같은 frame 에 attention 을 ×k 하고 grounding 출력을 비교:

| source · 배율 | frame | `decision_identical` | `bit_identical` | 처음 달라진 지점 |
|---|---:|---:|---:|---|
| L8H2 ×4 | 60 | **60** | **60** | — |
| L8H2 ×3 | 60 | **60** | 0 | `raw_order` 60 |
| oracle top-3 ×4 | 60 | **60** | **60** | — |
| oracle top-3 ×3 | 60 | **60** | 0 | `raw_order` 59 · `normalized` 1 |

- `decision_identical` = cloud · seed · peak · cluster 분할 · split 사유 · status · chosen 이 같음. `bit_identical` = 그에 더해 raw 를 `float32(k·raw)` 와 비교한 값 · 정규화 값 · score · centroid 가 bit 단위로 같음. ×4 는 float 에서 정확한 대조군 (2 의 거듭제곱) 이라 bit-identical 이어야 한다 — **60 / 60 으로 확인됐다.** ×3 은 decision 이 60 / 60 같다.
- (측정 사실 그대로) ×3 의 `bit_identical` 이 0 / 60 인데 `max_raw_rel_err` · `max_norm_abs_diff` 는 0.0 으로 기록돼 있다. 두 값이 어떻게 함께 나오는지는 verify.json 의 요약만으로는 알 수 없어 해석하지 않는다 (`first_difference` 가 `raw_order` 로 잡힌 것은 raw 값 비교의 첫 차이라는 뜻이다).
- 이 수치 시험은 **L8H2 와 oracle top-3 σ=1 에서만** 돌았다 (다른 source 는 미측정).

### 10. 측정의 한계

| 항목 | 내용 |
|---|---|
| held-out 은 pilot | 4 record (60 frame, 같은 scene 2 + 새 scene 2). same-scene 2 record 는 selection 의 `long` · `t2` 와 같은 episode (1800) 에서 출발한다. 새 scene 은 banana 1 · pear 1 뿐이다. frame 을 독립 trial 로 세지 않았으나 record 가 4 개라 record 단위 통계도 약하다 |
| 일부 camera · 일부 지표 | held-out 의 `cam_right_wrist` 는 target ≥ 200 px 가 0 frame 이라 2D 에서 제외, `ep1800` 의 `cam_high` 는 15 / 15 제외. 3D fusion 은 세 camera 를 모두 쓴다 |
| σ 편차 | T47 의 e2e 와 class mass figure 는 σ=0 (사전 고정은 σ=1). σ=1 은 T47b 가 2D · 3D 로 재측정 (§6). T47 의 σ=0 3D 값과 T47b 의 σ=1 값은 같은 표에 섞지 않았다 |
| grasp latch 없음 | offline e2e 에 TargetConfirm freeze · `GraspLatch` 가 없다. post-grasp frame 의 실패 수는 서빙 성능이 아니다 (selection record 3D 의 post-grasp 실패 50 / 136 등) |
| legacy ESDF | e2e 는 CPU legacy ESDF 다. `ball` 층 지표는 사후 계산이고 (`verify.json not_measured`) record 별로만 보고했다 |
| attention source | 별도 checkpoint 사본 · noise seed 0. 서빙 action 을 낸 forward 의 attention 이 아니다 |
| 미측정 | T47: `cam_right_wrist` · `ep1800 cam_high` 2D, grasp latch e2e, cuRobo ESDF e2e, selection record 에서 GT-oracle, `S_img` variant 의 e2e, d0 · last 밖 setting 과 σ=1 의 e2e. T47b: cuRobo 기준선, T47b source 의 selection record 3D, B3 수치 시험의 다른 source |
| p 값 | Spearman p 는 다중비교 보정 없음. 128 cell 은 독립이 아니다 (같은 layer 의 head 들) — 이 의존성은 측정하지 않았다 |
| 회귀 기준선 | legacy 일치 (T47 · T47b 모두). cuRobo 변형은 돌리지 않았다 |
| 논문 수치 | 논문의 점수 (cIoU 등) 는 이 로그가 측정한 것이 아니라서 옮기지 않았고, 우리 pilot 의 숫자와 직접 비교하지 않는다 |

### 11. 사용자 판정 · 되돌아올 지점 · 산출물

**사용자 판정 (2026-10-12, T47 verify.json 확인 후).**

- **L8H2 유지.** 사전 고정 primary (공통 top-3 = L4H1 · L2H0 · L14H5) 는 held-out 에서 target mass 0.012 – 0.017, 3D `on_target` 0 / 30 (same-scene · new-scene 모두, σ=0) 로 L8H2 (26 / 30 · 29 / 30) 보다 나빴다 → 채택하지 않는다. 서빙 cell 은 바꾸지 않는다 (`AG3S/trajopt/attention_policy.py` 무수정).
- 후속: (1) 실패 원인 (고정 patch sink · 넓게 퍼진 map 의 단일 성분) 의 정식 측정 → T47b #5 (이 절 §7), (2) 명사 token query (D) → `handoff/T49.task.md`.

**되돌아올 지점 (task 문서의 전환 신호와 측정 결과).**

| 신호 | 측정 결과 | 다음 |
|---|---|---|
| 선택된 head 가 gripper · 목적지에 모이면 label-free 기준만으로는 못 고른다 → (i) GT 지표를 섞는 supervised 선택 (sim 한정) 또는 (ii) 명사 token query | 공통 top-3 는 gripper · 목적지가 아니라 background 에 모였고 (§5-2) sink 1 + broad blob 2 였다 (§7). camera 별 top-3 는 gripper 몫이 상대적으로 컸다 | (ii) 는 T49 로 진행 중. (i) 는 이 STEP 에서 하지 않았다 |
| 2D 는 좋아지는데 3D 가 그대로면 병목은 head 가 아니라 3D 쪽 | 2D 가 좋아진 source 가 없다 (target mass · lift 는 L8H2 가 가장 높다). 3D 병목 후보로는 σ=1 의 이웃 과일 흡수 (§8-1) 가 측정됐다 | head 교체 보류 |
| held-out 이 작아 차이가 frame 몇 개 수준이면 record 추가를 제안 | L8H2 · head-average · oracle 이 `on_target` 에서 구분되지 않는다 (§5-3). 구분되는 지표는 seed precision / recall · `identity_correct` | record 를 늘리려면 GPU attention pass 가 필요하다 (사용자 승인 사항, 이 STEP 에서 하지 않음) |
| `identity_correct` 와 `on_target` 이 자주 갈리면 주 지표를 label 기반으로 바꾸는 것을 제안 | 갈린다 (§8-2, 60 frame 중 6 – 13 frame, 반대 방향 0) | **제안 (판정 대기):** 이후 grounding 평가의 주 지표에 `identity_correct` 를 함께 보고 |
| B3 에서 절대값 threshold 가 발견되면 head set 교체 전에 따로 판정 | default 경로의 raw 절대 상수는 degenerate guard 하나. score 에 걸린 0.1 은 head 재가중에 가변 (§9) | head set 을 바꾸는 STEP 에서 G8 · D1 의 0.1 을 다시 본다 |
| `mass_norm_joint` 가 `raw_sum` 보다 3D 에서 낫다면 합산 방식도 판정 대상 | 낫지 않았다 (§8-1) | 합산 방식은 판정 대상으로 올리지 않았다 |

**사용자 판정 대기.** 이 절 시점에 새로 사용자에게 물을 항목: (a) 이후 grounding 평가에서 `identity_correct` 를 주 지표 중 하나로 쓸 것인가 (위 표, 제안). 나머지 방향은 이미 판정됐다.

**산출물.**

- **측정**: `handoff/T47.verify.json` · `handoff/T47b.verify.json`. 원자료는 run 디렉터리 `/mnt/dev/AG3S_VLA-storage/results/development/20261012T141412+0900_bd778e5e3453_d7a51b456f36/verify/T47/` (gt_cache · select · validate · e2e) 와 `…/verify/T47b/` (check · sink · b2d · e2e · b3 · supp · indep).
- **figure** (각각 `.json` sidecar 있음): T47 [`scene`](figures/t47/t47-scene-overlay.png) · [`graph 1`](figures/t47/t47-sorted-S-tau.png) · [`graph 2`](figures/t47/t47-selection-frequency-rank.png) · [`graph 3`](figures/t47/t47-class-mass.png) · [`table`](figures/t47/t47-e2e-table.png), T47b [`scene`](figures/t47/t47b-scene-identity.png) · [`graph 1`](figures/t47/t47b-identity-nontarget.png) · [`graph 2`](figures/t47/t47b-sink-scatter.png) · [`table`](figures/t47/t47b-b3-audit-table.png). T47 figure 는 σ=0 (sidecar 의 `sigma`).
- **구현** (전부 새 파일, 기존 파일 무수정): `AG3S/ag3s/experiments/studies/t47_*.py` · `t47b_*.py`, `tests/ag3s/test_t47_head_selection.py` (45 passed) · `test_t47b_*.py` (33 passed). 문서: `handoff/T47.task.md` · `T47.impl.md` · `T47b.task.md` · `T47b.impl.md`.
- commit 은 하지 않았다.

---

## T48a — HSR counterfactual: phase probe 는 instruction · state 문자열 · image 가 바뀌면 어떻게 반응하는가 (2026-10-12 KST)

**이 절이 답하는 물음 (목적).** HSR (Hidden-state Sub-task Recognition) 은 얼린 pi0.5 의 prefix KV cache 에서 뽑은 feature `kv_L4` (layer 4 · **V** · KV head 0, 512 차원 = `[valid text token 평균 | image token 768 개 평균]`) 위에 logistic regression 하나 (`kv_L4_v1`) 를 얹어 지금 하는 일을 `pick` / `place` / `home` 으로 읽는 구성 요소다 (SUBTASK 절). 이 label 은 `TargetConfirm` 의 **subtask gate** 로 들어간다 — 확정 label 이 `place` · `home` 이면 새 target 채택을 막고, PLACED 뒤 `home` 이면 쥐고 있던 물체를 놓아 준다 (`T48.task.md` 가 가리키는 `target_grounding.py:945, 959-968, 1124-1135, 1204-1209`, `config.py:342-360` `subtask_gate=True`).

지금까지의 근거는 **weak decodability** (feature 에서 label 이 읽힌다) 하나였다. `paper/notes/SUBTASK_FACTS.md` §1 · §3 이 기록한 사실로, val accuracy 0.9808 이지만 **frame index 만으로도 val 1.000** (250 episode 중 248 이 같은 phase 순서의 scripted dataset) 이라 그 정확도가 *작업 상태* 를 읽은 것인지 *시간 · 팔 자세의 순서* 를 읽은 것인지 가를 수 없다. gripper 값을 가려도 0.9808 (mask (a)) 이고 단순 rule 은 0.9423 이었다. 다른 하나는 **strong decodability** (관련 입력이 바뀌면 올바르게 반응한다) 이다. 검토 문서 `doc/reviews/AG3S_TWO_PAPER_REVIEW_20261012_KO.md` §4 가 둘째 논문 (Bhardwaj et al., arXiv:2608.13474) 의 구분으로 가져온 것이고, 그 원문이 일반 probe 의 R² 0.95 에도 instruction target 을 바꾸면 출력이 거의 바뀌지 않았다고 보고한다고 적는다 (원문의 수치이며 이 로그가 측정한 것이 아니다). 여기에 우리 구성의 특수한 사정이 하나 더 있다 — **text 평균 안에 discretize 된 16D state 문자열 (gripper · joint 숫자) 이 들어 있어** probe 가 image 대신 state 문자열을 읽을 수 있다.

T48a 는 한 frame 의 입력 셋 (prompt · 16D state · RGB 3 장) 중 **하나만** 바꾼 변형 (counterfactual) 을 만들어 같은 probe 에 다시 넣고 예측이 어떻게 변하는지 잰다. rendering 이 필요 없고 (dataset 의 기록 frame 만 쓴다) GPU 는 offline prefix pass 에만 쓴다. empty grasp · slip (물체 없는 image 를 새로 만드는 것) 은 T48b 다.

**한 줄 결과 (before → after).**

- **before.** val accuracy 0.9808 (SUBTASK-b, 위 `SUBTASK_FACTS.md` 의 기록) 이고 frame-index baseline 도 1.000 이라 *무엇을 읽는지* 는 열려 있었다.
- **after — 측정된 사실 셋.** 아래 (a) (b) 는 이 STEP 이 처음 잰 것이고 (c) 는 **아직 정해지지 않았다**.
  - **(a) probe 의 예측은 image 를 따르고 state 문자열을 따르지 않았다.** image A + state B (C2s) 에서 예측 class 가 image 쪽인 비율 0.979 (2546/2601), state 쪽 0.012 (31/2601). image B + state A (C2i) 도 image 쪽 0.979 (2547/2601), state 쪽 0.013 (33/2601). used gripper 값 하나만 반대로 한 C2g 에서 예측이 C0 예측과 달라진 비율은 0.005 (14/2601). 완전 held-out ep1807 (75 frame, 1 episode) 도 같은 방향이다 (C2s image 쪽 72/75, C2i image 쪽 75/75, C2g 0/75). 같은 변형에서 simple rule 은 C2i 의 state 쪽이 0.942 (2451/2601) 이다.
  - **(b) probe 는 instruction 의 target 명사 교체에 거의 반응하지 않았다.** C1 (prompt 의 target 명사 → 같은 scene 의 distractor 과일) 에서 원래 place 인 frame 이 `pick` 으로 바뀐 비율 0.040 (95/2400), 원래 home 은 0/2403. 이 0.040 은 C0 (변형 없음) 에서 probe 가 이미 place → pick 으로 읽던 비율 32/800 = 0.040 (산출) 과 구분되지 않는다. 같은 target 에 phrasing 만 바꾼 C1p 에서는 예측이 5191/5202 (0.998) 유지된다. ep1807 에서도 place → pick 3/39, home → pick 0/129.
  - **(c) probe 가 "물체를 쥐고 있다" 는 작업 상태를 읽는지, "팔이 이 자세에 있다" 는 자세 순서를 읽는지는 이 측정으로 가려지지 않는다.** C2 의 image 이식은 같은 episode 의 다른 frame 의 image 를 통째로 옮기므로 image 안에 팔 자세와 물체 상태가 함께 따라온다. dataset 은 실패 · retry 가 없는 scripted 시연이라 둘이 항상 같이 움직인다. → **T48b** (같은 팔 자세에서 물체의 유무만 바꾼 image, 사용자 판정: 진행 — §9).
- **strong-rule proxy 는 실제 strong rule 이 아니다.** 비교에 쓴 proxy 는 실제 grasp latch (`grasp_latch.py`) 를 offline 으로 흉내 낸 것이고 (§3), 아래 표의 proxy 열을 "strong rule 과 비교한 결과" 로 읽지 않는다.
- **C0 재현.** 새로 뽑은 feature 의 val accuracy 0.9804 (2550/2601) 는 SUBTASK-b 의 0.9808 과 0.000385 차이로, 통과 기준 ±0.002 안이다 (§5-1).

**읽는 법과 출처 규약.**

- 수치는 `T48.verify.json` 의 `numbers` 에서만 왔다. 같은 파일 숫자의 덧셈 · 곱셈 · 비율은 "(산출)" 로 적었다. figure sidecar (`figures/t48/*.json`) 는 verify.json 이 같은 값을 갖고 있을 때만 옮겼다. `SUBTASK_FACTS.md` 와 검토 문서 (논문) 의 수치는 이 로그가 측정한 것이 아니므로 "이미 기록된 사실" 로 출처와 함께만 적었다.
- 해석의 출처를 가른다. 사용자 판정은 "(사용자 판정)", lead 가 task 문서에 쓴 것은 "(lead)", 측정으로 확인하지 않은 해석은 "(미측정 해석)" 으로 적었다.
- 시각은 KST. pod 의 `date` 는 UTC 이고 verify.json 의 UTC 필드에 +9 h 를 더했다.
- **frame 은 독립 trial 이 아니다.** val 은 50 episode × (stride 6 으로 뽑은) 약 52 frame 이다. 표는 frame 합산 (pooled k/n) 을 주로 적고, episode 단위 비율의 median · IQR 은 figure 표에 있다. 대부분의 값에서 episode median 이 pooled 와 같은 것은 episode 마다 비율이 거의 같기 때문이다 (예: probe C0 episode median 0.9808 = 51/52 (산출), IQR 폭 0).
- **환경.** GPU 는 offline feature 추출 한 번 (H200, `flock -n` 으로 `/tmp/ag3s-vla-gpu-validation.lock` 획득, rc 0) 에만 썼다. 회귀 기준선은 legacy 가 일치했다 (§5-3). HEAD `218229ac`, 코드는 새 파일 4 개 + test 뿐이고 기존 파일은 수정하지 않았다 (`T48.impl.md`). commit 은 하지 않았다.

### 1. 사용자 판정과 타임라인 (KST)

| # | 판정 · 사건 (2026-10-12) | 이 절에서 |
|---|---|---|
| 1 | **(사용자 판정)** 검토 문서 §4 · §6 의 실험 3 (HSR counterfactual) 을 T47 다음으로 한다. **T48a (rendering 불필요) 먼저**, empty grasp · slip (T48b) 은 결과를 본 뒤 판정한다 | §2 |
| 2 | **(사용자 판정)** 지시된 물체가 아닌 것을 쥐고 있을 때 (wrong-object) 의 정답은 **instruction 기준 `pick`** 이다 (예: 사과를 쥔 frame + prompt "pear" → `pick`) | §2 (C1) |
| 3 | **(사용자 판정)** VLA checkpoint 의 offline GPU feature 추출을 승인한다 (policy server 없음, prefix pass). GPU lock 과 `nvidia-smi` 를 확인하고 다른 process 는 건드리지 않는다 | §5, §6-7 |
| 4 | **(lead)** T46 session 의 CL3 일정과 겹치지 않게 조율한다 | §6-7 |
| 5 | **(사용자 판정, T48a 결과 후)** **T48b 진행.** 세부는 lead 의 `T48b.task.md` — 16D policy image re-render parity 를 먼저 확인하고 통과하지 못하면 거기서 멈춰 보고, GPU 추출은 T46 의 CL3 가 끝난 뒤에 한다 | §9 |

| 시각 (KST) | 무엇 | 출처 |
|---|---|---|
| 15:52:29 | 측정 run 디렉터리 `20261012T155229+0900_218229ac6464_2e66675de9ee` 생성 (verifier 구간 15:52 – 16:40) | 디렉터리 이름 · verify.json `time_kst` |
| 15:54:21 | GPU extract 시작 (`--batch 4`, 24084 prefix pass) | `runs[extract_gpu].start` (06:54:21Z) |
| 16:06:34 – 16:09:35 | GPU 사용 sampler 창 (10 sample, compute process 1 개 = 이 extract) | `cost.concurrent_gpu_state` |
| 16:09:47 | extract 종료 `rc=0` | `runs[extract_gpu].end_rc` (07:09:47Z) |
| 16:11 – 16:12 | figure 3 개 생성 | 파일 시각 (UTC 07:11 – 07:12) |

### 2. 변형 설계

dataset 의 모든 episode 는 target 1 + distractor 3 이고 모두 `place_one`, 성공, retry 0 이다. distractor 는 episode 내내 건드려지지 않는다고 본다 (`T48.task.md`). 그래서 prompt 를 distractor 로 바꾸면 그 instruction 기준 정답은 **모든 frame 에서 `pick`** 이다 (아직 안 잡았고 basket 에도 없다). 사용자 판정 #2 와 같다.

| ID | 바꾸는 것 | 고정 | 정답 | 무엇을 묻나 |
|---|---|---|---|---|
| **C0** | 없음 (원본) | — | dataset `phase_index` → class (pick 0–3 · place 4–8 · home 9–13, `kv_L4_v1.json`) | 재현 점검 |
| **C1** | prompt 의 target 명사 → 같은 scene 의 distractor 과일 3 개 각각 | image · state | **`pick` (모든 frame)** | language 민감도 — 원래 place · home frame 이 `pick` 으로 바뀌는가 |
| **C1p** | 같은 target, **다른 phrasing** (dataset 의 prompt template 3 개 중 나머지 2 개) | image · state | 원래 class | 통제 — 말만 바꿨을 때 흔들리지 않는가 |
| **C2g** | state 의 **used gripper 값 하나만** 반대로 (열림 1.0 ↔ 닫힘 0.6702, dataset train split 에서 정한 대표값) | image · prompt | 원래 class (image 가 보여 주는 실제 상태) | gripper 숫자에 끌려가는가 |
| **C2s** | **state 전체** 를 같은 episode 의 **다른 class frame** (B) 에서 이식 → image A + state B | image · prompt | A 의 class (image 쪽) | state 문자열이 phase 를 결정하는가 |
| **C2i** | **image 3 장** 을 같은 episode 의 다른 class frame (B) 에서 이식 → image B + state A | state · prompt | B 의 class (image 쪽) | image 가 phase 를 결정하는가 |

- 변형은 16D float **state 배열** 을 바꾸는 것이고 문자열을 직접 편집하지 않는다. state 문자열은 서빙과 같은 tokenizer 경로 (`np.digitize` 로 256 bin 화) 가 만든다.
- C2s 와 C2i 는 대칭이다. 같은 (host frame A, partner frame B) 쌍에서 둘 다 만든다. B 는 A 와 같은 episode 의 **다른 class** frame 이고 seed 0 으로 고정한다. 예측이 state 쪽 class 인지 image 쪽 class 인지 쌍마다 센다.
- 입력은 SUBTASK-b 의 val split 과 같다 (episode 1600–1796 중 4 번째마다 50 개, stride 6). 추가로 **ep1807 recorded run** (75 step, 유일한 완전 held-out) 에 같은 변형을 적용한다. 그 정답은 SUBTASK-b `closed_loop_gt` (held / in-crate / reach) 정의를 쓴다.

| 항목 | val | ep1807 |
|---|---:|---:|
| unit (episode · run) | 50 | 1 |
| frame 수 (pick / place / home) | 2601 (1000 / 800 / 801) | 75 (19 / 13 / 43) |
| 변형 row 수 C0 / C1 / C1p / C2g / C2s / C2i | 2601 / 7803 / 5202 / 2601 / 2601 / 2601 | 75 / 225 / 150 / 75 / 75 / 75 |
| 합계 (prefix pass) | 23409 | 675 (전체 24084) |
| C2s · C2i 쌍 수 | 2601 | 75 |

(전부 `T48.verify.json` `plan` · `sets.*.n_variants` · `independent_checks.swap_sanity`. 전체 24084 = `cost.n_inputs`.)

**정답 정의의 전제.** C1 의 "모든 frame `pick`" 은 distractor 가 episode 내내 건드려지지 않는다는 데 기댄다. `T48.impl.md` §2 는 metadata 의 initial → final pose 이동량으로 이를 확인했고 val distractor 중 이동량이 큰 것이 있다고 적는다. 그 수치는 `T48.verify.json` 에 없어 이 로그에 옮기지 않는다 (verifier 재측정 없음). 정답 pick 정의는 그대로 둔다.

### 3. 비교 대상 (baseline 셋)

같은 frame · 같은 입력 변형에 probe 와 나란히 적용한다.

| baseline | 정의 | 주의 |
|---|---|---|
| **simple rule** | SUBTASK-b `probe.py` 의 rule — 양 gripper 중 하나가 닫힘 문턱 아래 → `place`, 아니면 이 episode 에서 지금까지 닫힌 적이 있음 → `home`, 아니면 `pick` | prompt 를 보지 않는다 |
| **offline strong-rule proxy** | state 의 gripper 값 (측정 finger 개도) 을 요청 단위로 넣는 latch 상태 기계 (PRE → CLOSING → HELD → PLACED) 와 이력. 문턱은 `grasp_latch.py` `LatchConfig` 를 따른다 (`T48.impl.md` §5) | **strong rule 이 아니라 proxy 다.** 실제 strong rule (`grasp_latch.py`) 은 실행 feedback (`applied_gripper`, `blocked_gap`) 과 조작 대상 reach 를 쓰는데 offline 기록에는 없어 뺐다 |
| **frame-index** | train frame 의 frame index 별 다수 class | ep1807 은 stride-6 격자에 없는 frame index 가 많아 가장 가까운 train frame index 로 대신한다 (`frame_index_fallback_rows` 549, 전체 675 row 중 (산출)). SUBTASK-b 는 closed loop 에서 이 baseline 을 재지 않았다 |

**정의상 정해지는 칸 (해석하지 않는다).** simple rule · proxy · frame-index 는 prompt 를 보지 않으므로 C1 에서 자기 C0 예측을 그대로 낸다 — 그 실패는 rule 의 설계에서 온다. 또 rule · proxy 는 host frame A 의 이력을, frame-index 는 A 의 clock 을 쓰므로 C2s · C2i 에서 frame-index 는 정의상 C2s image 쪽 · C2i state 쪽과 같은 class 를 낸다. proxy 도 이력의 영향이 커서 같은 방향으로 기운다. 이 칸들은 probe 의 반응과 같은 의미의 측정이 아니다.

### 4. 구현 (전부 새 파일, 기존 파일 무수정; `T48.impl.md`)

| 파일 | 역할 |
|---|---|
| `AG3S/ag3s/experiments/studies/t48_hsr_core.py` | 변형 생성 (C1 · C1p · C2g · C2s · C2i), prompt template 추출, state 문자열 numpy replica, baseline 셋, 지표, `closed_loop_gt`. 순수 numpy |
| `.../t48_hsr_io.py` | dataset val · train · ep1807 reader, unit 구성, train 통계 (대표 gripper 값 · frame-index), video decode |
| `.../t48_hsr_features.py` | `FeatureExtractor` (GPU, **prefix-only**, 서빙의 `_input_transform` 을 그대로 사용), model 없이 서빙 input transform 을 조립하는 CPU 함수 |
| `.../t48_hsr_counterfactual.py` | CLI `--stage plan / extract / score` — SUBTASK-b data parity, token check, C0 gate, feature parity, 지표, predictions CSV |
| `tests/ag3s/test_t48_hsr_counterfactual.py` | 합성 데이터 손계산 test (변형 · baseline · 지표) + storage 의 SUBTASK-b feature 로 C0 재현 + GPU opt-in 1 |

test: 27 passed · 1 skipped (GPU opt-in) · 0 failed (`T48.verify.json` `numbers.pytest_t48`). extract 는 plan row 의 digest 가 같으면 이어 하고 다르면 덮어쓰지 않고 멈춘다. `--stage score` 는 C0 가 SUBTASK-b 를 재현하지 못하면 중단한다 (task 의 "되돌아올 지점").

### 5. 검증 — 변형이 정의대로 만들어졌고 C0 가 재현되는가

#### 5-1. C0 재현 (gate)

| 항목 | 값 |
|---|---|
| SUBTASK-b val accuracy (기준) | 0.980777 |
| 새 feature 의 val accuracy | **0.980392** (2550/2601) |
| 차이 · 허용 | 0.000385 ≤ 0.002 → `reproduced = true` |
| val confusion (행 = 원래 pick · place · home, 열 = 예측) | 새 [[982, 18, 0], [32, 768, 0], [0, 1, 800]] 대 SUBTASK-b [[983, 17, 0], [32, 768, 0], [0, 1, 800]] — pick → place 한 frame 차이 |
| feature 수치 parity (val 2601 frame, 새 대 SUBTASK-b 저장 feature) | rel L2 median 0.00241 · p99 0.00335 · max 0.00612, argmax 일치 2600/2601 |
| feature 수치 parity (ep1807 75 frame) | rel L2 median 0.00247 · max 0.00343, argmax 일치 75/75, accuracy 0.96 = 저장 feature 와 같음 |
| SUBTASK-b 저장 feature 로 독립 재계산 (verifier 의 별도 script) | probe 0.980777 · simple rule 0.942330 (SUBTASK-b 의 값과 같다) |
| ep1807 새 feature accuracy | 0.96 (72/75), confusion [[19, 0, 0], [1, 12, 0], [0, 2, 41]] |
| batch 4 대 batch 1 (첫 batch, n = 4) | rel L2 max 0.00289, max abs 0.00297 |

task 가 정한 통과 기준은 **C0 accuracy 의 재현** 이고 충족했다. 위 rel L2 가 어느 수준의 잡음인지는 이 STEP 에서 따로 판정하지 않았다 (task 는 feature parity 를 "있으면" 의 보조로 두었다).

#### 5-2. 독립 재계산과 변형 점검 (verifier)

| 점검 | 결과 |
|---|---|
| cache 의 feature 에서 probe 를 다시 계산해 CLI 의 예측과 비교 | 불일치 val 0/23409 · ep1807 0/675 |
| `metrics.json` 의 지표를 독립 script 로 다시 계산 | 1104 field 비교 · 차이 0 |
| 변형 state 문자열 (model 이 실제로 본 bin) — C2s = B 의 C0 | val 2601/2601 · ep1807 75/75 |
| C2i = A 의 C0 | val 2601/2601 · ep1807 75/75 |
| C1 · C1p 의 state bin 이 C0 와 같다 | val 7803/7803 · 5202/5202, ep1807 225/225 · 150/150 |
| C2g 가 바꾼 bin 수 | 모든 row 에서 정확히 1 개 (val 2601 · ep1807 75) |
| C2s feature 의 image 반쪽이 A 의 C0 에 더 가깝다 | val 2601/2601 · ep1807 75/75 |
| C2s feature 의 text 반쪽이 **B** 의 C0 에 더 가깝다 | val 33/2601 · ep1807 0/75 |
| C2i feature 의 image 반쪽이 B 의 C0 에 더 가깝다 | val 2601/2601 · ep1807 75/75 |
| C2i feature 의 text 반쪽이 A 의 C0 에 더 가깝다 | val 35/2601 · ep1807 0/75 |

앞 여섯 행은 "변형이 정의대로 만들어졌다" 의 확인이다. 마지막 네 행은 `kv_L4` 512 차원을 반으로 가른 거리 비교로, verifier 의 구성 점검이며 이 로그는 분류 이유의 근거로 쓰지 않았다. 다만 state 문자열을 B 로 바꿨는데도 text 반쪽이 B 쪽에 가까운 frame 이 33/2601 뿐이라는 것은, text token 의 V 값이 state 문자열보다 image 에 더 크게 좌우될 수 있음을 시사한다 (미측정 해석 — prefix 가 image · text 를 서로 attend 하기 때문일 수 있으나 여기서 분리해 재지 않았다).

**replica 불일치.** 보고용 numpy `discretize_state` replica 와 서빙 경로에서 decode 한 bin 이 val 의 27 row (전부 `ep_001648`, 차원 13, |bin 차이| 1) 에서 다르다. model 이 본 것은 서빙 bin 이고 replica 는 보고용이다. ep1807 은 0 이다 (`notes_facts`). 지표는 서빙 경로의 입력으로 만든 feature 를 쓰므로 이 불일치가 지표에 들어가지 않는다.

#### 5-3. 회귀 기준선

legacy 기준선 (`esdf_rollout`, `.venv-ag3s`) 은 일치했다 — violated 시작 14 / 15 · has_target 15 / 15 · frame0 `clearance_before` −29.031048 mm (skill 의 기준값과 같다). baseline 실행 시점의 1 분 load 평균은 9.95 였다 (`concurrent_load_1min`). cuRobo 변형은 GPU 사용을 이 STEP 의 단일 승인 extract 로 한정하기 위해 돌리지 않았다 (`not_measured`).

### 6. 결과 (val 50 episode · 2601 frame, ep1807 75 frame)

#### 6-1. variant × 방법 정답률

![T48a variant × 방법 정답률 표](figures/t48/t48-accuracy-table.png)

[`t48-accuracy-table.png`](figures/t48/t48-accuracy-table.png) — 위 val, 아래 ep1807. 각 칸이 pooled 정답률이고 val 칸에 episode median [IQR] 도 있다. C1 의 정답은 모두 `pick`, C2s · C2i 의 정답은 image 쪽 frame 의 class 다. 아래는 같은 값의 표다.

**val**

| variant | 정답 | rows | probe `kv_L4_v1` | simple rule | strong-rule proxy (실제 strong rule 아님) | frame-index |
|---|---|---:|---:|---:|---:|---:|
| C0 | 원래 class | 2601 | 0.9804 (2550/2601) | 0.9423 (2451/2601) | 0.9954 (2589/2601) | 1.0000 (2601/2601) |
| C1 | `pick` (전부) | 7803 | 0.3897 (3041/7803) | 0.3460 (2700/7803) | 0.3799 (2964/7803) | 0.3845 (3000/7803) |
| C1p | 원래 class | 5202 | 0.9802 (5099/5202) | 0.9423 (4902/5202) | 0.9954 (5178/5202) | 1.0000 (5202/5202) |
| C2g | 원래 class | 2601 | 0.9804 (2550/2601) | 0.0192 (50/2601) | 0.9616 (2501/2601) | 1.0000 (2601/2601) |
| C2s | image 쪽 (A) class | 2601 | 0.9789 (2546/2601) | 0.3441 (895/2601) | 0.9700 (2523/2601) | 1.0000 (2601/2601) |
| C2i | image 쪽 (B) class | 2601 | 0.9792 (2547/2601) | 0.0254 (66/2601) | 0.0027 (7/2601) | 0.0000 (0/2601) |

**ep1807** (1 episode — episode median · IQR 은 의미가 없어 k/n 만 적는다)

| variant | 정답 | rows | probe `kv_L4_v1` | simple rule | strong-rule proxy (실제 strong rule 아님) | frame-index |
|---|---|---:|---:|---:|---:|---:|
| C0 | 원래 class | 75 | 0.9600 (72/75) | 1.0000 (75/75) | 0.9600 (72/75) | 0.8800 (66/75) |
| C1 | `pick` (전부) | 225 | 0.2667 (60/225) | 0.2533 (57/225) | 0.2800 (63/225) | 0.2000 (45/225) |
| C1p | 원래 class | 150 | 0.9600 (144/150) | 1.0000 (150/150) | 0.9600 (144/150) | 0.8800 (132/150) |
| C2g | 원래 class | 75 | 0.9600 (72/75) | 0.0000 (0/75) | 0.9333 (70/75) | 0.8800 (66/75) |
| C2s | image 쪽 (A) class | 75 | 0.9600 (72/75) | 0.4400 (33/75) | 0.9467 (71/75) | 0.8800 (66/75) |
| C2i | image 쪽 (B) class | 75 | 1.0000 (75/75) | 0.0000 (0/75) | 0.0000 (0/75) | 0.0400 (3/75) |

C0 에서 simple rule 이 ep1807 에 1.0000 인 것은 SUBTASK-b 가 ep1807 의 정답을 gripper 로 정의했기 때문이다 (`SUBTASK_FACTS.md` §3). frame-index 의 ep1807 값은 위의 nearest fallback 을 쓴 확장 정의다.

#### 6-2. C1 — prompt 의 target 명사를 distractor 로 (language 민감도)

![T48a C1 · C2 · C2g 그래프](figures/t48/t48-c1-c2-graph.png)

[`t48-c1-c2-graph.png`](figures/t48/t48-c1-c2-graph.png) — 위 행 val, 아래 행 ep1807. 왼쪽 = C1 의 원래 class 별 예측 class stacked bar, 가운데 = C2s · C2i 가 따른 쪽, 오른쪽 = C2g 로 예측이 C0 와 달라진 비율. 막대 위 숫자는 `pick` 으로 간 비율이고 정답은 모든 class 에서 `pick` 이다.

**val — 원래 class 별 `pick` 예측 비율 (정답은 전부 1.0)**

| 원래 class (rows) | probe `kv_L4_v1` | simple rule | strong-rule proxy (실제 strong rule 아님) | frame-index |
|---|---:|---:|---:|---:|
| pick (3000) | 2946/3000 (0.982) | 2700/3000 (0.900) | 2964/3000 (0.988) | 3000/3000 (1.000) |
| place (2400) | 95/2400 (0.040) | 0/2400 (0.000) | 0/2400 (0.000) | 0/2400 (0.000) |
| home (2403) | 0/2403 (0.000) | 0/2403 (0.000) | 0/2403 (0.000) | 0/2403 (0.000) |

**probe 의 C1 matrix 와 C0 confusion 의 3 배 비교 (산출).** C1 은 같은 frame 에 distractor 과일 3 종을 각각 넣으므로 C0 의 모든 row 가 3 배가 된다. 예측이 하나도 안 바뀌었다면 C1 matrix 는 C0 confusion 의 정확히 3 배다.

val

| 원래 class | C0 confusion (pick · place · home 예측) | C0 × 3 (산출) | C1 matrix (측정) |
|---|---|---|---|
| pick | [982, 18, 0] | [2946, 54, 0] | [2946, 54, 0] |
| place | [32, 768, 0] | [96, 2304, 0] | [95, 2303, 2] |
| home | [0, 1, 800] | [0, 3, 2400] | [0, 3, 2400] |

ep1807

| 원래 class | C0 confusion (pick · place · home 예측) | C0 × 3 (산출) | C1 matrix (측정) |
|---|---|---|---|
| pick | [19, 0, 0] | [57, 0, 0] | [57, 0, 0] |
| place | [1, 12, 0] | [3, 36, 0] | [3, 36, 0] |
| home | [0, 2, 41] | [0, 6, 123] | [0, 6, 123] |

- val 의 pick · home row 는 3 배와 정확히 같고 place row 는 한두 칸 (−1, −1, +2) 만 다르다. ep1807 은 세 row 모두 정확히 3 배다. 이것은 **집계 수준의 일치** 이고 frame 마다 예측이 같다는 것은 아니다 (C0 대 C1 의 frame 별 예측 비교는 verifier 가 따로 내지 않았다).
- C1 accuracy 는 `pick` 으로 간 비율이며 probe 0.3897 (3041/7803), distractor 별로는 apple 0.3875 · banana 0.3910 · orange 0.3885 · pear 0.3916 이다. episode 단위로 원래 place 의 `pick` 전환 비율은 median 0.0625 [0.0, 0.0625] (원래 place frame 이 episode 당 16 개 (산출) 이므로 episode 당 0 – 1 개), pick 의 유지는 median 1.0 [0.967, 1.0].
- 같은 정의로 simple rule · proxy · frame-index 는 place · home 에서 `pick` 전환이 0 이다 (위 표). 이들은 prompt 를 보지 않는다.
- (미측정 해석) probe 의 학습 prompt 는 12 개의 완전한 instruction 뿐이고 paraphrase 에 대한 강건성은 이전에 재지 않았다 (`SUBTASK_FACTS.md` §3). 그 사실과 위 결과가 어떻게 이어지는지는 이 STEP 이 가리지 않았다.

#### 6-3. C1p — 같은 target, 다른 phrasing (통제)

**val — 예측 유지 비율 (같은 host frame 의 C0 예측 대비)**

| 원래 class | probe `kv_L4_v1` | simple rule | strong-rule proxy (실제 strong rule 아님) | frame-index |
|---|---:|---:|---:|---:|
| pick | 1994/2000 (0.997) | 2000/2000 (1.000) | 2000/2000 (1.000) | 2000/2000 (1.000) |
| place | 1595/1600 (0.997) | 1600/1600 (1.000) | 1600/1600 (1.000) | 1600/1600 (1.000) |
| home | 1602/1602 (1.000) | 1602/1602 (1.000) | 1602/1602 (1.000) | 1602/1602 (1.000) |
| all | 5191/5202 (0.998) | 5202/5202 (1.000) | 5202/5202 (1.000) | 5202/5202 (1.000) |

probe 는 예측이 5191/5202 유지되고 (바뀐 11 frame), rule · proxy · frame-index 는 prompt 를 보지 않으므로 5202/5202 이다. ep1807 은 네 방법 모두 150/150.

#### 6-4. C2g — used gripper 값만 반대로

**val — 원래 class 별 예측이 C0 예측에서 바뀐 비율**

| 원래 class | probe `kv_L4_v1` | simple rule | strong-rule proxy (실제 strong rule 아님) | frame-index |
|---|---:|---:|---:|---:|
| pick | 7/1000 (0.007) | 1000/1000 (1.000) | 12/1000 (0.012) | 0/1000 (0.000) |
| place | 6/800 (0.007) | 800/800 (1.000) | 50/800 (0.062) | 0/800 (0.000) |
| home | 1/801 (0.001) | 801/801 (1.000) | 50/801 (0.062) | 0/801 (0.000) |
| all | 14/2601 (0.005) | 2601/2601 (1.000) | 112/2601 (0.043) | 0/2601 (0.000) |

- probe 는 모든 class 에서 0.001 – 0.007 이다 (전체 14/2601). C2g 입력의 원래 class × 예측 matrix 는 pick → place 11 · place → pick 38 · home → place 2 이고 (`C2g_matrix_origclass_x_pred`), 같은 칸의 C0 confusion 은 18 · 32 · 1 이다.
- simple rule 은 정의상 gripper 문턱을 넘기므로 전부 바뀐다. proxy 는 이력을 쓰므로 일부만 바뀐다 (전체 112/2601).
- ep1807 은 probe 0/75 · simple rule 75/75 · proxy 2/75 · frame-index 0/75.

#### 6-5. C2s · C2i — state 를 바꾸면 따라가는가, image 를 바꾸면 따라가는가

**val — 예측 class 가 state 쪽 / image 쪽 / 기타 인 비율**

| 방법 | 변형 | state 쪽 | image 쪽 | 기타 |
|---|---|---:|---:|---:|
| probe `kv_L4_v1` | C2s | 31/2601 (0.012) | 2546/2601 (0.979) | 24/2601 (0.009) |
| probe `kv_L4_v1` | C2i | 33/2601 (0.013) | 2547/2601 (0.979) | 21/2601 (0.008) |
| simple rule | C2s | 1288/2601 (0.495) | 895/2601 (0.344) | 418/2601 (0.161) |
| simple rule | C2i | 2451/2601 (0.942) | 66/2601 (0.025) | 84/2601 (0.032) |
| strong-rule proxy (실제 strong rule 아님) | C2s | 54/2601 (0.021) | 2523/2601 (0.970) | 24/2601 (0.009) |
| strong-rule proxy (실제 strong rule 아님) | C2i | 2589/2601 (0.995) | 7/2601 (0.003) | 5/2601 (0.002) |
| frame-index | C2s | 0/2601 (0.000) | 2601/2601 (1.000) | 0/2601 (0.000) |
| frame-index | C2i | 2601/2601 (1.000) | 0/2601 (0.000) | 0/2601 (0.000) |

(simple rule · proxy · frame-index 의 값은 위 §3 의 정의상 결과이므로 probe 의 반응과 같은 의미로 읽지 않는다.)

**probe — (image class, state class) 조합별.** C2s 는 (image A class, state B class), C2i 는 (image B class, state A class) 이다. 값은 `state / image / 기타` 비율이다.

| (image class, state class) | C2s rows | C2s: state / image / 기타 | C2i rows | C2i: state / image / 기타 |
|---|---:|---|---:|---|
| image `pick` · state `place` | 506 | 0.022 / 0.978 / 0.000 | 416 | 0.014 / 0.986 / 0.000 |
| image `pick` · state `home` | 494 | 0.000 / 0.984 / 0.016 | 405 | 0.000 / 0.983 / 0.017 |
| image `place` · state `pick` | 416 | 0.048 / 0.950 / 0.002 | 506 | 0.051 / 0.947 / 0.002 |
| image `place` · state `home` | 384 | 0.000 / 0.961 / 0.039 | 396 | 0.003 / 0.965 / 0.033 |
| image `home` · state `pick` | 405 | 0.000 / 1.000 / 0.000 | 494 | 0.000 / 1.000 / 0.000 |
| image `home` · state `place` | 396 | 0.000 / 1.000 / 0.000 | 384 | 0.000 / 1.000 / 0.000 |

- probe 의 state 쪽 비율이 가장 큰 조합은 image `place` · state `pick` 으로 C2s 0.048 · C2i 0.051 이고, 이 조합은 place ↔ pick 경계 frame 이 섞이는 쪽이다 (C0 confusion 의 pick ↔ place 오류와 같은 방향이다). image 가 `home` 인 두 조합은 양쪽에서 image 쪽 1.000 이다.
- 쌍 단위 (같은 (A, B) 에서 C2s 와 C2i 를 함께 본 결과, 산출 포함):

| 방법 | C2s 와 C2i 모두 image 쪽 | C2s image · C2i state | C2s state · C2i image | 둘 다 state 쪽 | 그 밖 (산출) | 쌍 수 |
|---|---:|---:|---:|---:|---:|---:|
| probe `kv_L4_v1` | 2492 | 33 | 31 | 0 | 45 | 2601 |
| simple rule | 0 | 887 | 62 | 1177 | 475 | 2601 |
| strong-rule proxy (실제 strong rule 아님) | 0 | 2518 | 7 | 47 | 29 | 2601 |
| frame-index | 0 | 2601 | 0 | 0 | 0 | 2601 |

probe 는 2601 쌍 중 2492 쌍에서 C2s · C2i 모두 image 쪽을 따랐고, 둘 다 state 쪽을 따른 쌍은 0 이다.

#### 6-6. ep1807 (완전 held-out, 1 episode · 75 frame)

| 항목 (ep1807, 75 frame) | probe `kv_L4_v1` | simple rule | strong-rule proxy (실제 strong rule 아님) | frame-index |
|---|---:|---:|---:|---:|
| C0 accuracy | 72/75 | 75/75 | 72/75 | 66/75 |
| C1: 원래 pick → pick | 57/57 | 57/57 | 57/57 | 45/57 |
| C1: 원래 place → pick | 3/39 | 0/39 | 6/39 | 0/39 |
| C1: 원래 home → pick | 0/129 | 0/129 | 0/129 | 0/129 |
| C1p: 예측 유지 | 150/150 | 150/150 | 150/150 | 150/150 |
| C2g: 예측이 바뀜 | 0/75 | 75/75 | 2/75 | 0/75 |
| C2s: state / image / 기타 | 0 / 72 / 3 | 33 / 33 / 9 | 0 / 71 / 4 | 3 / 66 / 6 |
| C2i: state / image / 기타 | 0 / 75 / 0 | 75 / 0 / 0 | 72 / 0 / 3 | 66 / 3 / 6 |

위 표는 1 episode 이므로 비율의 불확실성은 크다 (예: 75 frame 에서 1 frame = 0.013). 방향은 val 과 같지만 val 과 독립된 증거로 세지 않는다.

#### 6-7. 비용 (GPU, offline prefix pass)

| 항목 | 값 |
|---|---|
| batch · 입력 수 · 호출 수 | 4 · 24084 · 6022 |
| prefix pass + `kv_L4` reduction (입력당) | median **15.05 ms** · p90 15.12 ms (batch 4 호출 median 60.2 ms) |
| prefix 시간 합 · extract wall time | 363.4 s · **922.7 s** (≈ 15.4 분, 산출) |
| checkpoint load | 12.0 s |
| unit 당 wall (전형) · 첫 unit (JIT 포함) | 17.0 s · 46.3 s |
| GPU memory | JAX peak 7,600,011,264 byte · `nvidia-smi` process 8,808 MiB |
| pod working set (extract 중) | 41.7 – 43.3 GB (extract 시작 시점은 측정하지 못했다), 끝 39.0 GB |
| 다른 GPU compute process | extract 전 없음, sampler 10 회 모두 compute process 1 개 (이 extract) |

timing 은 prefix pass 와 `kv_L4` reduction 의 device 시간이며 input transform · video decode · warm-up/JIT 를 뺀다 (`cost.timing_note`). SUBTASK-b 의 prefix pass median 24.50 ms (H200 bf16, batch 1, `SUBTASK_FACTS.md` §3) 와는 batch 크기 (4 대 1) 가 달라 같은 조건의 비교가 아니다 — batch 1 전체 extract 는 재지 않았다.

#### 6-8. 실제 scene — C2s · C2i 쌍

![T48a C2s · C2i 쌍의 실제 scene](figures/t48/t48-scene-c2-pairs.png)

[`t48-scene-c2-pairs.png`](figures/t48/t48-scene-c2-pairs.png) — 네 쌍. 각 쌍의 왼쪽 두 열이 host frame A (cam_high · wrist 한 대), 가운데 두 열이 partner frame B, 오른쪽 표가 C0(A) · C0(B) · C2s (image A + state B) · C2i (image B + state A) 에 대한 probe · rule · proxy · frame-index 예측과 probe 확률이다. 선택 규칙은 sidecar `selection` 에 있다 — val ep_001600 에서 (image class, state class) = (pick, place) · (place, home) · (home, pick) 인 첫 C2s row 세 개와 ep1807 의 첫 (place, pick) row 이며 결과를 보고 고르지 않았다. 네 쌍 모두 probe 의 C2s 예측은 C0(A) 예측과, C2i 예측은 C0(B) 예측과 같다. 두 번째 쌍은 C0(A) 에서 이미 probe 가 정답 (place) 이 아닌 pick 으로 읽는 경계 frame 이다 (확률이 거의 반반).

### 7. 읽을 때의 한계

| 한계 | 내용 |
|---|---|
| **proxy ≠ strong rule** | proxy 는 offline 기록에 없는 실행 feedback 과 조작 대상 reach 를 뺀 근사다. 실제 strong rule 과의 비교는 하지 않았다 (§8) |
| **frame-index fallback** | ep1807 의 frame-index 는 549 / 675 row 에서 nearest 정의다. SUBTASK-b 가 이 baseline 을 closed loop 에서 재지 않았으므로 ep1807 의 frame-index 값은 확장 정의의 값이다 |
| **replica 불일치** | val 27 row (`ep_001648`, 1 bin 차이). 지표는 서빙 bin 으로 만든 feature 를 쓴다 (§5-2) |
| **C2 에서 baseline 의 의미** | rule · proxy · frame-index 는 host 의 이력 · clock 을 쓰므로 C2 의 결과가 정의에서 나온다 (§3) |
| **C1 정답 pick 의 전제** | distractor 가 거의 움직이지 않는다는 dataset 성질에 기댄다. 이동량의 수치는 verify.json 에 없다 (§2) |
| **일반화** | 완전 held-out 은 ep1807 한 episode 뿐이다. val 50 episode 는 probe 학습 split (episode 0–1599) 과 겹치지 않으나 (`SUBTASK_FACTS.md` §1) 모두 같은 scripted 시연이다 |
| **image 안의 정보** | C2 의 image 이식은 팔 자세와 물체 상태를 함께 옮긴다. image 를 따랐다는 것이 *무엇을* 읽는지는 말하지 않는다 (결과 (c)) |

### 8. 측정하지 않은 것 (`T48.verify.json` `not_measured`)

- 실제 strong rule (`grasp_latch.py` + 실행 feedback) 과의 비교. offline proxy 만.
- batch 1 (서빙 batch size) 전체 extract. extract 는 `--batch 4` 였고 batch parity 는 첫 batch (n = 4) 에서만 쟀다.
- extract 시작 시점의 pod working set (lead 요청이 extract 도중 도착).
- cuRobo 회귀 기준선 (GPU 를 단일 extract 로 한정).
- T48b (empty grasp · slip · phantom hold).

### 9. 되돌아올 지점과 다음 STEP

| task 의 신호 (`T48.task.md`) | 측정 결과 | 다음 |
|---|---|---|
| C1 에서 place · home frame 의 `pick` 전환이 거의 없으면 probe 는 instruction 을 반영하지 못한다. 그때 HSR 은 wrong-object 상황에서 strong rule 보다 나을 근거가 없고, 검토 문서 §6 기준대로 **보조 구성 요소** 로 두며, language 민감 probe 는 별도 판정한다 | 전환이 거의 없었다 (val place 0.040 · home 0.000, ep1807 3/39 · 0/129). 다만 strong rule 은 측정하지 않았고 proxy 도 같은 칸에서 0 이다 | 신호의 조건에 해당하는 값이다. HSR 의 위치를 보조 구성 요소로 정할지, language 민감 probe 를 따로 판정할지는 **사용자 판정 대기** — 이 STEP 은 정하지 않았다 |
| C2 에서 예측이 state 를 따르면 HSR 은 사실상 proprio rule 이다 → state token 을 뺀 변형을 후속으로 제안 | 따르지 않았다 (state 쪽 0.012 · 0.013) | 해당 없음. state token 을 뺀 변형은 제안하지 않는다 |
| C0 가 재현되지 않으면 T48 전체를 멈추고 feature 경로 차이를 찾는다 | 재현됐다 (0.9804 대 0.9808) | 해당 없음 |
| T48b (empty grasp · slip, 16D policy image re-render parity 필요) 는 T48a 결과를 보고 판정 | (c) 가 미결이다 | **T48b 진행 (사용자 판정).** 16D policy image re-render parity (E0: re-render 한 image 의 예측이 기록 image 의 예측과 같은가) 를 먼저 확인하고 통과하지 못하면 거기서 멈춰 보고한다. GPU 추출은 T46 의 CL3 가 끝난 뒤. 구체 설계는 `handoff/T48b.task.md` |

**산출물.**

- **측정**: `handoff/T48.verify.json`. 원자료는 run 디렉터리 `/mnt/dev/AG3S_VLA-storage/results/development/20261012T155229+0900_218229ac6464_2e66675de9ee/verify/T48/` (`plan` · `feat/full` · `extract` · `score` · `indep` · `c0_stored` · `cost` · `baseline`).
- **figure** (각각 `.json` sidecar 있음): [`scene`](figures/t48/t48-scene-c2-pairs.png) · [`graph`](figures/t48/t48-c1-c2-graph.png) · [`table`](figures/t48/t48-accuracy-table.png).
- **구현**: `AG3S/ag3s/experiments/studies/t48_hsr_*.py` (4 개) · `tests/ag3s/test_t48_hsr_counterfactual.py` (27 passed · 1 skipped). `handoff/T48.impl.md`.
- **task 문서**: `handoff/T48.task.md` (T48a) · `handoff/T48b.task.md` (다음 STEP).
- commit 은 하지 않았다.
