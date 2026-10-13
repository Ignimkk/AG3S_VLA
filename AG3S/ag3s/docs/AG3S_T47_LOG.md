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
| **T47 · T47b · T49** | T47 = head 재선택 (label-free 기준) 의 2D · 3D 검증. T47b = 합산 방식 · label 기반 3D 지표 · raw amplitude · sink 진단. T49 = 명사 token query (D) 로 넘어가는 다음 STEP (~~아직 시작 전~~ → ~~진행 중 (2026-10-12, 진행 현황 표 참조)~~ → 완료 (아래 "T49" 절), `handoff/T49.task.md`) |

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
| **T48 · T48a · T48b** | T48 = HSR counterfactual. T48a = prompt · state · image 변형 (rendering 불필요, 기록 frame 만 사용). T48b = 물체의 유무를 바꾼 image (empty grasp · slip · phantom hold) 를 MuJoCo 로 다시 render 해 같은 팔 자세에서 probe 가 쥠 상태를 읽는지 가르는 다음 STEP (`handoff/T48b.task.md`; ~~진행 중~~ → 완료, 아래 "T48b" 절) |

**T48b (2026-10-12 추가)** — 물체의 유무를 바꾼 counterfactual 의 용어.

| 용어 | 뜻 |
|---|---|
| **re-render parity (E0)** | 기록된 qpos 를 MuJoCo 로 policy image 규약 (224×224, reflection off) 그대로 다시 그린 image 가 기록 RGB 와 같은가. `bit-identical` = 3 camera 의 모든 pixel 값이 같음 (MAD 0 · max abs 0). 편집 image 를 믿어도 되는지의 전제 |
| **E0 · E1 · E1s · E1z · E2 · E3** | E0 = 편집 없이 re-render 만 (parity 점검). 나머지는 대상 과일의 free joint 와 쓰는 손 finger joint 만 바꾸고 `mj_forward` 후 다시 그린 것. **E1 (empty grasp)** = 쥔 과일을 초기 table 위치로 옮기고 finger 를 "물체 없이 닫힌" 값 (state 0.457) 으로 둔다, 정답 `pick`. **E1s** = 과일만 옮기고 finger · state 는 그대로, 정답 `pick`. **E1z** (exploratory) = E1 과 같되 finger 를 joint 한계 (state 0, 완전히 닫힘) 로 둔다. **E2 (slip)** = 과일을 손 바로 아래 table 위에 놓고 finger · state 는 E1 과 같게, 정답 `pick`. **E3 (phantom hold)** = 파지 전 접근 frame 에서 과일을 finger 사이에 넣고 finger 를 과일 폭에 맞춘다, 정답 `place` (물리적으로 쥐어짐) |
| **물리 predicate (`held` · `reach` · `on_table` · `in_crate`)** | MuJoCo contact 로 정한 과일 상태. `held` = 한 손의 finger 두 개 모두와 접촉하고 table · crate 와 비접촉. `reach` host = 쓰는 손이 처음 닫히기 전 frame. 편집의 정답은 gripper 값이 아니라 이 predicate 로 정한다 |
| **E0 이중 채점 (`closed_loop_gt` · contact state)** | E0 정답을 두 기준으로 채점. `closed_loop_gt` = SUBTASK-b 정의 (gripper 가 닫혔으면 `held`), contact state = 위 물리 predicate (모호 frame 제외). long · t2 의 빈 손 닫힘 frame 에서 둘이 다르다 |
| **관통 제외 (penetration exclusion)** | 편집 뒤 대상 과일 또는 finger 가 낀 contact 의 최대 관통 깊이가 2.0 mm 를 넘는 편집은 제외. 기준은 편집하지 않은 기록 frame 의 최대 관통 (held-out 1.83 mm) 에 맞춰 결과 전에 고정 |
| **held-out primary · train_range secondary** | primary = ep1807 · long · t2 · headfix · ep1800 · ep1850 · ep1900 (VLA · probe 어느 학습에도 없는 7 run). secondary = ep0 · ep500 · ep1200 (VLA fine-tune episode, ep0 · ep1200 은 probe 학습 episode 이기도 함) — 편집 image 는 학습에 없었으나 장면은 익숙할 수 있어 분리 보고. long · t2 · headfix · ep1800 은 같은 scene (episode 1800) 이라 독립 표본 1 개로 센다 |

**T49 · T49b (2026-10-12 추가)** — 명사 token query 의 용어.

| 용어 | 뜻 |
|---|---|
| **명사 token query** | prefix 의 prompt 안 target 과일 명사 token (예 `▁apple`) 의 행 (query row) 이 image 768 열에 둔 attention 을 camera 별 16×16 map 으로 쓴 것. action (suffix) token query 와 달리 prefix pass 한 번으로 끝나고 denoise · noise seed 가 없다 (결정적). `basket` token 행은 같은 방식의 목적지 map |
| **image mass · BOS sink** | image mass = 명사 행 attention 중 image 768 열의 합 (나머지는 text 열, 그중 BOS 열이 큼). BOS 열에 몰리는 head 는 image 를 보지 않는 head 로 읽는다. layer 별 median 이 0.01 미만이면 "image 를 거의 보지 않음" 으로 표시 (표시만 하고 선택에서 제외하지 않음 — 결과 전 고정) |
| **cell 선택: GT primary · label-free secondary** | primary (`noun_gt`) = selection record 3 개 (long · t2 · ep1807) 에서 target map 의 lift 평균으로 18×8 cell 순위를 매기고 top-3 를 σ=1 raw 합. secondary (`noun_lf`) = 같은 명사 행에 T47 의 label-free 절차 (S_cam τ Kneedle + spatial entropy + selection frequency, k=3). 목적지 map (`basket_gt` · `basket_lf`) 도 같은 방식, crate coverage 기준 |
| **`noun_gt_k{k}_s{σ}`** | 명사 GT cell 의 top-k 를 σ 로 smoothing 후 합산한 source 이름 (primary = `k3_s1`) |
| **separation (분리)** | 같은 prefix pass 에서 target map 과 basket map 이 서로 다른 물체를 가리키는가. 3 camera joint argmax 가 target map 은 target body 위 (`target_peak_on_target`), basket map 은 crate 위 (`basket_peak_on_crate`) 에 있는 frame 의 비율이 `both_correct_rate`. 섞임 mass = target map 의 crate mass, basket map 의 target mass |
| **prompt swap (교체)** | 같은 frame (image · state 그대로) 에서 prompt 의 target 명사만 scene 의 다른 과일로 바꿔 map 이 새 과일로 옮겨가는가를 본다 (exploratory). `moved` = 원 prompt map 의 peak 가 원 target 위에 있고 동시에 교체 prompt map 의 peak 가 새 과일 위에 있음 (둘 다 ≥ 200 px 보이는 frame 에서) |
| **`lift_new_gain` · `map_corr`** | lift_new_gain = 교체 prompt 의 새 과일 lift − 원 prompt 의 새 과일 lift. map_corr = 두 map 의 Pearson 상관 (낮을수록 map 이 많이 바뀜) |
| **record-mean** | 지표를 record (episode) 안에서 frame 으로 pool 한 뒤 record 가중 없이 평균한 값. T49 · T49b 의 3D · 분리 · swap 표는 모두 이 집계다 (T47 의 pooled 값과 집계가 달라 같은 source 의 값이 조금 다르다) |
| **T49b (exploratory)** | 이미 추출한 명사 attention 위에서 σ ∈ {0, 1} × k ∈ {1, 2, 3} 과 label-free σ=0 을 3D e2e 로 다시 잰 보강 측정. 채택 근거가 아니다 |

**T50 (2026-10-12 추가)** — representation 분리 측정의 용어.

| 용어 | 뜻 |
|---|---|
| **`pi05_base`** | 우리 16D LoRA 학습의 초기 weight (로컬 12 GB). 같은 16D config 의 입력 변환 그대로 params 만 바꿔 같은 변형의 `kv_L4` feature 를 뽑아 LoRA 와 비교한다 |
| **`r` ratio (feature 민감도)** | frame 마다 `r = d(C0, C1) / d(C0, C1p)`. d = host frame 의 C0 feature 에 대한 상대 L2 (각 절반 안에서), C1 = 다른 과일 3 개를 지시한 prompt 의 평균, C1p = 같은 target 의 다른 phrasing 2 개의 평균. r < 1 이면 target 교체가 표현만 바꾼 것보다 feature 를 덜 움직인다. text 절반 (dim 0–255) · image 절반 (256–511) · full 을 따로 낸다 |
| **`d_adj`** | 같은 episode 의 다음 sampled frame (stride 6 = 0.4 s) 의 C0 와의 상대 L2. feature 가 시간에 따라 움직이는 기준선 |
| **counterfactual probe** | 같은 512D feature 위의 multinomial LogReg 를 원본 C0 행 + C1 행 (distractor 지시, 정답 `pick`) 1 : 1 로 학습한 것. **standard probe** = C0 행만 학습 (`kv_L4_v1` 과 같은 방식). 지표 `C0` = C0 정확도, `C1_instr` = 원래 place · home 인 host 의 C1 행 중 `pick` 으로 예측한 비율 |
| **`valcv` · `trainval`** | `valcv` (T50a primary) = val 50 episode 안의 episode 단위 5-fold 교차검증. `trainval` (T50b) = train split 200 episode 의 C0 + C1 로 학습해 val 50 episode 전체 평가 |
| **통제 (control)** | text 절반만 · image 절반만 · label 섞음 (C1 행의 label 을 무작위로). label 을 섞으면 C1_instr 가 올라가지 않아야 학습이 label 을 쓴 것이다 |
| **decision rule (사전 고정)** | LoRA 에서 ② 가 C0 ≥ 0.95 이고 C1_instr ≥ 0.80 이면 representation 에 있음 → 원인은 (a) probe 학습. 아니면 (b) representation 또는 (c) fine-tuning. base 에서 기준을 넘고 LoRA 에서 넘지 못하면 (c), 둘 다 넘지 못하면 fine-tuning 탓이 아니다. 0.95 · 0.80 은 episode 별 정확도의 median 에 적용 (lead 가 결과 전 고정) |
| **T50 · T50a · T50b** | T50 = HSR 의 instruction 둔감이 probe 학습 · representation · fine-tuning 중 어디서 오는지 가르는 STEP. T50a = LoRA feature 위 ① 민감도 · ② probe (CPU). T50b = base feature 를 GPU 로 뽑아 같은 표 (+ train split C1) |

**T46b · T46c (2026-10-12 추가)** — F1 을 켠 뒤 늘어난 close 이후 HOLD 의 진단과 F1 채택의 용어. (T46 의 F1 · F2 · C1a · CL3 · HOLD kind · `ball_owned` · SUBTASK-f 의 원 정의는 [`AG3S_T0T6_LOG.md`](AG3S_T0T6_LOG.md) 의 "용어" 절.)

| 용어 | 뜻 |
|---|---|
| **arm 이름 (C0 · C1a · C1aF1 · C1aF1F2)** | C0 = V11AB (기본 ball 제거, CL2 의 첫 arm). C1a = `ball_owned` (ball 안 seed 를 target 소유 / non-target 소유로 나눠 non-target 을 보존). C1aF1 = C1a + F1 (`--nontarget-evidence surface`, non-target 근거에서 target 자신의 표면 점을 뺌). C1aF1F2 = C1aF1 + F2 (`--owned-ambiguous remove`, ambiguous seed 도 제거). 같은 18 (episode, seed) 쌍 |
| **window (pre8 + close)** | grasp 직전 구간: close 가 시작된 chunk 이전 8 chunk 부터 close chunk 까지 (CL3 정의, 18 run 합쳐 162 chunk). **close 이후** = 그 뒤 모든 chunk |
| **phase (closing · lift · carry · placed)** | close 이후를 recorded `grasp.state` 로 나눈 것. closing = gripper 가 닫히는 중, lift = 쥐고 target 이 처음 높이 (z0 + 50 mm) 에 이르기 전, carry = 그 이후, **placed = 과일을 놓은 뒤** (release 이후의 place · home) |
| **seed · support mask** | seed = VLA attention 상위 5 % 점 (`seed_percentile` 95). support mask = 테이블 · 바닥 등 지지면으로 분류된 점의 mask. grounding 은 seed 에서 support mask 안의 것을 지운 뒤 clustering 한다 |
| **`no_seed` (grounding status)** | support mask 를 지운 뒤 남은 seed 가 0 개라 cluster 가 하나도 없는 상태 (`n_clusters = 0`). target 이 없는 것과 같은 결과지만 "gate 가 일부러 비운 것 (`subtask_gated`)" 이나 "후보가 있었으나 admissible 하지 않음 (`no_admissible`)" 과 구분된다 |
| **`geometry_certified` (인증 술어)** | server 가 chunk 의 기하를 믿어도 되는지 판정하는 술어: grounding 이 `ok` 이거나, `subtask_no_target` 예외 (gate 가 의도적으로 target 을 비웠음) 에 해당. False 이면 위반 행이 없어도 TO verdict 가 `violated` 로 내려가 chunk 가 HOLD (`uncertified` kind) 된다 |
| **certification gap (인증 공백)** | 인증 술어의 예외 목록 (`subtask_gated` · `no_admissible`) 에 `no_seed` 가 없어서, placed · home 에서 위반이 없는데도 HOLD 가 되는 규칙상 빈틈. SUBTASK-f (놓은 뒤 home 이면 `no_admissible` 도 인증, 2026-10-03) 가 `no_admissible` 만 추가하고 `no_seed` 를 남겼다 ([T0T6 SUBTASK 절 되돌아올 지점](AG3S_T0T6_LOG.md)) |
| **trailing HOLD** | episode 가 끝날 때까지 이어진 HOLD (한 번 걸리면 풀리지 않음) |
| **W1 (A12)** | placed · home 의 `no_seed` 를 SUBTASK-f 인증 조건 (gate on ∧ label home ∧ `placed_seen` ∧ valid ∧ carve 없음) 아래서 `no_admissible` 과 같이 인증하도록 인증 술어를 넓히는 수정안. 기본 off 의 flag. 위반 행이 있으면 HOLD 는 그대로다 (인증은 "설명 안 됨" 만 해소). T46b 에서는 기록 chunk 위 offline what-if 로만 계산했다 |
| **phase-gated F1** | F1 을 grasp 전 phase (pre-grasp · approach) 에만 적용하고 close 이후에는 끄는 변형안 |
| **첫 궤적 분기 (first divergence)** | 같은 (episode, seed) 의 두 arm 에서 실행 action 이 처음 1e-6 이상 달라지는 control step. 그 step 이 속한 server chunk 의 phase 를 본다 |
| **McNemar exact p** | 같은 pair 안에서 두 arm 의 이분 결과 (예: run 에 placed `no_seed` 가 있는가) 가 어긋난 pair 수 (한쪽만 · 반대쪽만) 로 계산하는 정확 검정의 p 값 |
| **extent method adoption (extent 방법 채택)** | develope_PLAN §0.10.1-3 의 "경계 보존 방법 하나" 로 F1 을 정한 것 (T46c, 사용자 판정). 같은 identity 에서 extent 처리만 바꾼 비교가 근거이고, 한계 L1–L9 를 함께 기록한다. "성공률을 올린다" 는 채택 근거가 아니다 |
| **L1–L9 (F1 한계)** | T46c §2 의 한계 표 번호. 각 한계는 논문 주장 범위의 제약을 짝으로 갖는다 |
| **PA-1 / DT-02 / DT-03 / DT-05A** | develope_PLAN 의 논문 설계 단위. PA-1 = 논문 설계와 evidence 선별, DT-02 = identity source 와 고정 identity 의 extent 비교 표, DT-03 = identity · extent · contact handling 의 핵심 contrast (closed-loop), DT-05A = field policy (no exemption · global exclusion · LTE) 별 contact · success · stop |

**T51 · T51b (2026-10-12 추가)** — attention head 선택 protocol (filter → calibration → bootstrap → k 선택 → held-out → ablation) 과 명사 query 추가의 용어.

| 용어 | 뜻 |
|---|---|
| **Ω (후보 집합)** | head 선택이 고르는 후보 cell 전체. T51 의 Ω = {action query × layer L2–17 × head 0–7}. action query = action token, denoise 0, aggregation `agg ∈ {last, mean, first}` → **384 cell** (L0 · L1 제외). T51b 는 Ω 에 명사 token 행 (`noun_target`) 을 넣는다: **명사만 (a) = 128 cell**, **action + 명사 (b) = 512 cell** |
| **P1–P6 (protocol 6 단계, 결과 전 사전 고정)** | P1 filter (label 없이 이상한 head 를 거름) → P2 calibration 점수 (3D `identity_correct` 비율로 cell 순위) → P3 안정성 (scene bootstrap) → P4 k 선택 (nested CV) → P5 held-out (calibration 에 쓰지 않은 record 평가) → P6 ablation (다른 선택 규칙과 비교) |
| **P1 filter rule — `low_mass` · `fixed_patch_sink` · `broad_map`** | (cell × camera 마다, 한 camera 라도 해당하면 그 cell 제외) **`low_mass`** = image mass 의 frame median < 0.01 (image 를 거의 보지 않는 head). **`fixed_patch_sink`** = 최빈 argmax patch 의 frame 비율 > 0.5 **이고** top-1 patch mass 몫 > 0.4 (frame 이 바뀌어도 늘 같은 patch 에 몰리는 head). **`broad_map`** = top-1 patch mass 몫 < 0.03 **이고** 평균 위 pixel 의 최대 component 면적 > grid 의 15 % (넓게 퍼진 map). T47 · T47b 의 sink · broad blob 진단 (위 "fixed-patch sink · broad single blob") 을 제외 규칙으로 바꾼 것이고, threshold 는 결과 전에 고정했고 다시 맞추지 않는다 |
| **2D → 3D 2 단계 선별 · forced cell** | 3D 계산 (MuJoCo replay, 비싸다) 은 P1 을 통과한 cell 중 **2D lift 상위 16 개** 와 **forced cell** (서빙 `last:L8H2` + T47 oracle top-3 `last:L8H3` · `last:L10H7` · `last:L12H4`) 에만 한다. P1 에 걸린 forced cell 은 3D 점수만 보고하고 순위 경쟁에는 넣지 않는다 |
| **scene · scene mean** | **scene** = 같은 초기 배치 (episode index) 에서 출발한 run 의 묶음. calibration 의 scene 은 5 개: 1800 (`long` · `t2`), 1807, 0, 500, 1200. **scene mean** = 같은 scene 의 run 을 먼저 평균한 뒤 scene 끼리 평균한 값 (P2 의 사전 고정 점 추정). run 이 많은 scene 이 가중을 더 받지 않는다 |
| **record-mean (T51 의 deviation)** | run (record) 6 개를 가중 없이 평균한 값. 사전 고정은 scene mean 이었으나 첫 구현이 이것으로 순위를 매겼다 (이 절 §4). 보고만 하고 판정에 쓰지 않는다 |
| **VLA 학습 episode (calibration 확장)** | `ep0` · `ep500` · `ep1200` 은 VLA fine-tune 에 쓰인 episode 의 기록이다 (사용자 판정으로 calibration 에 추가). held-out 은 그대로 VLA 학습에 없는 4 record 다 |
| **scene-level bootstrap** | scene 을 단위로 복원 추출한다 (같은 scene 의 run 은 함께 뽑힌다). B = 1000, seed 5103. scene 이 5 개라 서로 다른 resample 은 126 개뿐이라 모든 resample 을 세는 exact enumeration 값 (`exact_p_rank1`) 도 함께 낸다 |
| **P(rank1) · P(top3) · rank-1 tie rate** | resample 마다 P2 순위를 다시 매겼을 때 그 cell 이 1 위 / 상위 3 위 안인 비율. **rank-1 tie rate** = 1 위 점수가 동점이라 2D lift tie-break 가 순위를 정한 resample 의 비율. `stable_single_cell` = full-sample 1 위의 P(rank1) ≥ 0.5 (사전 고정, 되돌아올 지점의 기준) |
| **k 선택 (nested CV, leave-one-scene-out)** | k ∈ {1, 2, 3} (상위 k cell 의 raw 합) 을 scene 단위 교차검증으로 고른다. test scene 을 뺀 나머지로 순위를 매기고 top-k 합을 test scene 에서 잰다. 규칙: CV 평균이 가장 높은 k (best) 와의 쌍 차이 bootstrap 95 % CI 하한이 0 이하인 가장 작은 k 를 쓴다 (단순성 원칙). **nested** = 각 outer fold 의 k 를 그 안쪽 CV 로 정한다 (`k_inner`) |
| **held-out scene mean** | held-out 4 record (`headfix` · `ep1800` · `ep1850` · `ep1900`) 도 scene 으로 묶는다: scene1800 = `headfix` + `ep1800`, scene1850, scene1900 → **3 scene**. P5 · P6 의 판정 값 = scene 평균 (record 평균은 함께 보고) |
| **self label (grasp-event pseudo label)** | 사람 annotation 없이 얻는 target 위치 label. 한 손의 두 finger 가 처음 target 에 닿고 그 손이 이후 `held` 가 되는 시점 (`both_contact`) 의 fingertip pad 중점 (3D) 을 그 **이전 frame** 의 camera 로 투영해 16×16 label 과 3D ball (반지름 = 물체 반지름 + 2 cm) 을 만든다. 물체가 쥐기 전에 정지해 있다는 가정에 기댄다. 실물에서는 `grasp_latch` 의 HELD 가 같은 역할을 한다 (**이 STEP 은 sim 에서만 시험**) |
| **`proj_err_px` · `hit_pixel` · `hit_patch` · `target_disp_m`** | self label 의 sim 점검. `proj_err_px` = 투영점과 GT target pixel 중심의 거리 (480×640 px, target ≥ 200 px 일 때). `hit_pixel` = 투영점이 GT target pixel 위, `hit_patch` = 그 16×16 patch 의 GT target coverage > 0.05. `target_disp_m` = label frame 들과 grasp frame 사이 target 원점의 이동 최대 (1 cm 초과 frame 수 `n_frames_gt_tol` 를 함께 낸다) — "쥐기 전 정지" 가정의 직접 점검 |
| **`self_identity`** | self label 로 판정한 identity: chosen cluster 의 점 중 ball 안이 **엄격한 과반** 인 frame 의 비율 (grasp 이전 frame 만). GT `identity_correct` 와 정의가 다르다 |
| **annotation curve** | GT 로 고른 선택을 소량 사람 annotation 으로 재현하려면 frame 이 몇 개 필요한가: calibration frame n 개를 비복원으로 뽑아 순위를 다시 매기고, 전체 1 위와 같을 확률 (`P(same top-1)`) 과 1 위가 전체 상위 3 안일 확률 (`P(top-1 in reference)`) 을 낸다 (exploratory, frame 을 pool 해서 뽑는다) |
| **LIBERO task aggregation** | LIBERO 는 task (20 개) × episode (2 개) 다. 같은 task 의 두 episode 를 함께 묶어 점 추정과 bootstrap 을 한다 (scene 단위와 같은 원칙). primary 지표는 **agent view 2D lift** (3D `identity_correct` 불가) |
| **noun-only Ω (a) · combined Ω (b)** | (a) 후보가 명사 행 128 cell 뿐, (b) action 384 + 명사 128 = 512 cell. 같은 P1–P6 를 각각 돌려 어느 query 의 cell 이 뽑히는가를 본다. `p_rank1_sum_by_query_family` = P(rank1) 을 query family (action / noun) 별로 합친 값 |
| **scene-level paired difference (T51b (c))** | 두 선택 (예: 명사 best 와 action best) 의 held-out scene 평균 `identity_correct` 의 **scene 별 차를 평균** 한 값과 그 scene bootstrap 95 % CI (B = 1000, seed 5107 + 3^3 = 27 resample 전수). frame 단위 McNemar 는 frame 을 독립으로 세므로 **참고값** |
| **T51 · T51b** | T51 = head 선택 protocol 공식화 (RB-Y1 sim P1–P6 + self label + LIBERO 2D). T51b = 같은 protocol 에 명사 query 행을 추가해 action query 와 같은 기준으로 비교 |

**T46c-W1 · T46d · T46f (2026-10-13 추가)** — placed `no_seed` 인증 수정 (W1) 의 재현 검증, DX-04 (link 별 target 예외) 선행 확인, LTE 검증 블록 계획의 용어. (W1 · `no_seed` · `geometry_certified` 는 위 "T46b · T46c" 표, PA-1 · DT-02 · DT-05A 도 그 표.)

| 용어 | 뜻 |
|---|---|
| **authorized / non-authorized link (허용 / 비허용 link)** | authorized = target 에 닿는 것이 작업의 일부인 link: active hand 의 손바닥 `ee_<side>` (MJCF `EE_BODY_*`) 와 손가락 `ee_finger_<side>1/2`. non-authorized = 그 밖의 link: 손목 F/T 센서 `FT_SENSOR_<side>` · `link_*_arm_5` · `link_*_arm_6` · 반대 손 · 팔. 허용 여부는 link 의 축이고 target 영역 (어느 물체인가) 과 다른 축이다 |
| **field policy: no exemption · global exclusion · LTE** | query sphere 가 ESDF 에서 target 을 어떻게 읽는가의 세 방식 (develope_PLAN DX-04). **no exemption** = 모든 link 가 target 을 일반 obstacle 로 읽음 (full field, 권한 link margin 완화도 없음). **global exclusion** = 모든 link 가 target-free 합성 field 를 읽음 (CLI `--target-field-policy exclude-all`). **LTE (link-specific target exemption)** = authorized link 만 target-free field 를 읽고 나머지는 full field (`exclude-authorized`, 이 STEP 은 `--target-permission-strict` + `--nonauthorized-target-margin obstacle` 조합을 쓴다고 정함 — T46f §4). 현재 CLI 선택지는 `relax` / `exclude-authorized` / `exclude-all` 이고 no exemption 은 별도 선택지가 아니다. T46f 가 `none` 을 추가한다 (아래 "no exemption `none`") |
| **no exemption policy `none` (T46f 계획)** | `--target-field-policy none` — 모든 link 가 target 을 일반 obstacle 로 읽고 권한 link margin 완화도 하지 않는 정책. 아직 구현되지 않았다 (T46f A14, 계획) |
| **퇴화 (degenerate) 비교** | LTE 와 global exclusion 이 같은 판정을 내서 비교할 수 없는 상태. query set 의 모든 link 가 authorized 이면 두 정책이 같은 field 를 읽는다. 측정은 "target 근처 non-authorized 행 수 = 0" 으로 했다 |
| **near-target (criterion 2, T46d 사전 고정)** | link 의 sphere 표면에서 target 표면까지 거리가 50 mm 이하인 chunk 가 그 run 의 window 안에 하나라도 있으면 그 link 는 그 run 에서 target 에 접근한 것. 50 mm 는 target 의 full margin 이고, 측정 전에 고정했다 |
| **window (pre8close) · policy-active chunk** | pre8close = close 가 시작된 chunk 이전 8 chunk 부터 close chunk 까지 (T46b 의 window 와 같은 정의, 18 run × 9 = 162 chunk). policy-active = 그 chunk 의 target permission 정책이 `exclude_authorized` 이고 held 가 아니며 target 이 있고 target-free field 가 기록된 chunk. LTE 와 global 의 차이는 이 chunk 에서만 생길 수 있다 |
| **field-level re-query (LTE ≠ global chunk)** | 기록된 field (full / target-free 두 층) 를 plan 의 16 행 sphere 위치에서 다시 읽어 `d − r < m` 인 행을 위반으로 세는 offline 재계산. m ∈ {0, 10 mm (obstacle margin), 50 mm (target full margin)}. 한 chunk 에서 global (모두 target-free) 과 LTE (authorized 만 target-free) 의 위반 행 집합이 1 행이라도 다르면 그 chunk 는 **LTE ≠ global**. TrajOpt 를 다시 푼 것이 아니다 |
| **binding row · C2′** | binding row = 그 chunk 의 TO 에서 가장 제한적인 (최악) query 행의 link. C2′ = 이전 CL2 의 `--links arms` (양팔 14 link + 손가락, 422 구) arm ([이전 로그 정의](AG3S_T0T6_LOG.md)) |
| **sphere model 대 MJCF geom** | sphere model = URDF capsule 을 구로 덮은 constraint model (TrajOpt 가 보는 것). MJCF geom = MuJoCo 가 실제 접촉을 계산하는 collision geom. 두 모델이 어긋나면 constraint 가 실제 접촉과 다른 것을 막는다 (**형상 근사 오류**) |
| **FT sensor cover (`wrist_cover`, T46f 계획)** | 손목 F/T 센서 (MJCF `FT_SENSOR_*`, `link_*_arm_6` 의 자식 body) mesh 를 덮는 coverage 구의 role 이름 (예). 현재 URDF constraint model 에는 이 body 를 덮는 구가 없다. non-authorized 로 둔다 |
| **coverage audit** | 각 보정 link 의 MJCF geom 표면 점 중 구 밖 비율 · 밖 거리 (median / p95 / max) 로 형상 **누락** 을, 구가 geom 밖으로 나간 정도의 max 로 **과도한 팽창** 을, 비허용 구와 허용 (손바닥 · 손가락) 구 · MJCF 손 geom 의 **겹침** 을 재는 검사 (T46f §4). 선례: 이전 로그의 finger · palm coverage audit ([이전 로그](AG3S_T0T6_LOG.md)) |
| **diagnostic scene (진단 장면)** | 손의 target 접촉과 non-authorized link (FT sensor 등) 의 회피가 **동시에** 필요하도록 고른 배치. 기존 18 쌍은 일반 작업 성능 · 부작용 평가로 유지하고 이 장면을 따로 둔다 (T46f §5) |
| **S1 – S4 (diagnostic scene 선정 기준, 결과 전 고정)** | S1 = 실제 MuJoCo geom 기준 non-authorized 접촉 위험이 있음 (E0 rollout 의 FT sensor–target 최소 표면 거리 ≤ 50 mm 는 **후보 선별** 기준일 뿐 충돌 · 보호 효과의 증거가 아님). S2 = 접근 경로나 손목 자세를 바꾸면 작업 완료가 가능 (성공 시연 또는 IK / 계획 feasibility 1 개 이상). S3 = 초기 관통 (t = 0 접촉) 이나 구조적으로 성공 불가능한 배치 제외. S4 = 개발용 배치와 최종 평가용 배치를 episode id 로 사전 분리 (예: dev = train split, eval = held-out test split 1800–1999) |
| **visible observed surface oracle (T46e)** | 시뮬레이터의 GT 로 **카메라에 실제 보인 target 표면 점** (target geom id 의 visible pixel 의 depth 점) 에서 온 seed 만 target 으로 지우는 extent 상한. 전체 형상 (full mesh) oracle 이 아니다. 기본 off 이고 T46e 가 격리 worktree 에서 구현 (미적용) |
| **비허용 접촉 없는 작업 성공률 (T46f 주 지표)** | task success ∧ 실제 MuJoCo geom 기준 non-authorized link–target 접촉 0. HOLD · TO 실패가 늘어 접촉률이 낮아진 결과는 보호 효과로 읽지 않는다 |
| **T46c-W1 · T46d · T46f · T46e** | T46c-W1 = placed `no_seed` 인증 수정 (flag 기본 off) 의 검증. T46d = DX-04 선행 확인 (어떤 non-authorized link 를 넣어야 LTE 와 global 비교가 성립하는가, offline). T46e = visible observed surface oracle extent 구현 (`handoff/T46e.task.md`). T46f = LTE 검증 블록 (형상 보정 → 진단 장면 → 3 조건 closed-loop) 의 계획 (`handoff/T46f.task.md`) |

---

## 진행 현황 — T47 이후

| 게이트 | 무엇을 묻나 | 상태 |
|---|---|---|
| **T47 · T47b** attention head 재선택 (label-free: attention sum + spatial entropy, 고정 top-k) 과 보강 (합산 방식 · label 기반 3D 지표 · raw amplitude · sink 진단) | 16D · camera 3 대에서 다시 고른 head set 이 L8H2 보다 held-out 에서 target 을 더 잘 가리키는가 | **완료 (2026-10-12) — L8H2 유지 (사용자 판정).** 사전 고정 primary (L4H1 · L2H0 · L14H5, σ=1) 는 3D `on_target` 0 / 60 (L8H2 55 / 60), target mass 0.0134 (L8H2 0.1498). 선택된 head = sink 1 + broad blob 2. 합산 방식은 3D 를 거의 안 바꿈, `on_target` 이 놓치는 `identity_correct` 불일치와 σ=1 의 이웃 과일 흡수를 측정. held-out 은 pilot. 후속 T49 (명사 token query). 아래 "T47 · T47b" 절 |
| **T48a** HSR counterfactual (a): prompt target 교체 · state 이식 · image 이식 · gripper 반전 (rendering 불필요) | frozen VLA feature 위의 phase probe (`kv_L4_v1`) 가 instruction · state 문자열 · image 가 바뀌면 어떻게 반응하는가 (strong decodability) | **측정 완료 (2026-10-12).** C0 재현 (val 0.9804 대 0.9808). 사실 셋: (a) probe 는 image 를 따르고 state 문자열을 따르지 않는다 (C2s · C2i 에서 image 쪽 0.979, gripper 반전 시 예측 변화 0.005), (b) instruction 의 target 교체에 거의 반응하지 않는다 (C1 원래 place → `pick` 0.040 = C0 의 place → pick 오류율 0.040), (c) grasp state 를 읽는지 팔 자세 순서를 읽는지는 **미결**. strong-rule proxy 는 strong rule 이 아니다. **T48b 진행 (사용자 판정)**. 아래 "T48a" 절 |
| **T48b** HSR counterfactual (b): empty grasp · slip · phantom hold (물체의 유무를 바꾼 image 를 MuJoCo 로 re-render) | probe 가 "물체를 쥐고 있다" 를 읽는가, "팔이 이 자세에 있다" 를 읽는가 | ~~**진행 중 (2026-10-12)** — 16D policy image re-render parity 확인이 먼저. task: [`handoff/T48b.task.md`](handoff/T48b.task.md). 아직 기록할 측정 없음~~ → **측정 완료 (2026-10-12).** E0 re-render 가 기록 image 와 599/599 frame bit-identical (parity 통과). held-out primary (7 run) 의 편집 결과: empty grasp (E1) 에서 probe `pick` 1/13 (`place` 유지 12/13), E1s 4/13, slip (E2) 3/7, phantom hold (E3) `place` 97/99 (simple rule 99/99, offline proxy 0/99). **E1 · E2 는 2 run · 2 scene (ep1807 · ep1850) 에서 나온 pilot** (n = 13 · 7). "자세를 읽는가" 의 해석과 HSR 위치는 **판정 대기** (아래 "T48b" 절 §7). 실제 strong rule 은 측정 안 함. task: [`handoff/T48b.task.md`](handoff/T48b.task.md) |
| **T49** 명사 token query: prefix 의 target 명사 · "basket" token → image attention 으로 target · 목적지 map | action token query (L8H2) 대신 명사 token 을 query 로 쓰면 target · 목적지를 더 잘 가리키는가 | ~~**진행 중 (2026-10-12)** — 구현 단계 (`handoff/T49.impl.md` 있음), verify 대기. task: [`handoff/T49.task.md`](handoff/T49.task.md). 아직 기록할 측정 없음~~ → **측정 완료 (2026-10-12).** GPU extract 1040 frame 오류 0. held-out (4 record pilot, record-mean): 2D `peak_on_target` 1.000 (L8H2 0.693), 잡은 뒤 1.000 (L8H2 0.382), target ∧ 목적지 분리 `both_correct` 0.981, prompt 교체 시 map 이 새 과일로 옮겨간 frame 157/164. **그러나 3D `identity_correct` 0.800 (L8H2 0.900), purity 0.700 (0.933), non-target seed 손실 420.3 / frame (190.3).** 서빙 cell 은 L8H2 유지, 서빙 반영 없음. task: [`handoff/T49.task.md`](handoff/T49.task.md). 아래 "T49" 절 |
| **T49b** 명사 query 3D 열세 확인 (exploratory): σ ∈ {0, 1} × k ∈ {1, 2, 3} | 명사 query 의 3D 열세가 smoothing · 합산 설정 때문인가 | **측정 완료 (2026-10-12, 채택 근거 아님).** 모든 변형에서 `identity_correct` 0.800 – 0.850, purity 0.700 – 0.831 로 L8H2 (0.900 · 0.933) 에 못 미침. non-target seed 손실은 σ=0 에서 오히려 큼 (512.8 – 583.4 대 σ=1 394.5 – 420.3, L8H2 190.3). cell · k · σ 선택은 T51b 로. 아래 "T49b" 절 |
| **T50** HSR 의 instruction 둔감은 fine-tuning 탓인가 (T50a LoRA feature CPU + T50b `pi05_base` GPU) | probe 학습 · representation · fine-tuning 중 원인 분리 | **측정 완료 (2026-10-12).** 사전 고정 decision rule 의 출력: LoRA · base 모두 counterfactual probe 가 기준 (C0 ≥ 0.95 ∧ C1_instr ≥ 0.80, episode median) 미달 → **"fine-tuning 탓이 아니다"** (규칙이 정한 문구). feature 민감도 r (full, episode median) 은 LoRA 0.611 · base 0.999. 아래 "T50" 절 |
| **T51** attention head 선택 protocol 공식화 (RB-Y1 sim + 실물용 grasp-event label + LIBERO) | head · layer 를 하나로 고정하는 선택의 당위성 | ~~**진행 중 (2026-10-12)** — task: [`handoff/T51.task.md`](handoff/T51.task.md). 아직 기록할 측정 없음~~ → **측정 완료 (2026-10-12, 사용자 판정 대기).** calibration 6 record · 5 scene (ep0 · ep500 · ep1200 = VLA 학습 episode 추가), scene mean primary. P1 filter 가 L8H2 를 `low_mass` 로 제외 (cam_high mass median 0.006293 < 0.01). P2 GT 3D `identity_correct` 1 위 `last:L12H4` 0.691. **사전 고정 안정성 기준 (P(rank1) ≥ 0.5) 미충족: 0.429 → 단일 head 고정은 주장하지 않는다.** P4 는 단순성 규칙으로 k\* = 1 (CV 평균 최고는 k = 3). held-out (3 scene) 에서 `last:L12H4` · `last:L12H3` · `last:L8H2` 가 0.911 로 동률. self label 은 `self_identity` 1.0 으로 포화해 2D lift tie-break 가 선택을 정했다. LIBERO (2D lift, task 단위) 1 위 `mean:L12H3`. 집계 단위 deviation (record-mean → scene mean) 을 중간에 발견해 고침. 아래 "T51 · T51b" 절 |
| **T51b** head 선택 protocol 에 명사 query 행 추가 | action query 와 명사 query 를 같은 기준으로 비교 | ~~**진행 중 (2026-10-12)** — task: [`handoff/T51b.task.md`](handoff/T51b.task.md). 아직 기록할 측정 없음~~ → **측정 완료 (2026-10-12, 사용자 판정 대기).** (a) 명사만 Ω: GT 1 위 `noun_target:L10H7` P(rank1) 0.687 (상위 10 cell 의 점수 차 0.007 이내), self 1 위 `noun_target:L8H6` 0.529. (b) action + 명사 Ω: GT 1 위 `last:L12H4` P(rank1) 0.753 (action family 합 0.918), self 1 위 `noun_target:L8H6`. (c) held-out scene mean: 명사 best 0.867 대 action best 0.911, scene 단위 차 −0.044 (95 % CI [−0.133, 0.000], scene 3 개). 명사가 더 낫다는 증거는 없다. 아래 "T51 · T51b" 절 §12 |
| **T46b** F1 을 켠 뒤 늘어난 close 이후 uncertified HOLD 의 원인 (진단, CPU offline, T46 의 후속) | F1 이 grasp 이후 처리와 충돌해서 HOLD 가 늘었는가, 아니면 다른 원인인가 — F1 을 develope_PLAN §0.10.1-3 의 방법으로 채택할 수 있는가 | **측정 완료 (2026-10-12).** close 이후 uncertified HOLD 는 전부 **placed phase** (C1a 37 / 37 · C1aF1 130 / 130 · C1aF1F2 118 / 118 chunk) 이고, 실패 항목은 grounding `no_seed` (C1a 37 · C1aF1 129 · C1aF1F2 116 chunk, `n_clusters = 0`). 그 HOLD chunk 의 F1 기록은 0 / 130. 가설 (a) F1 이 grasp 이후 처리와 충돌 · (b) non-target 근거 감소 · (c) 분모 효과는 **반박**, (d) placed 뒤 `no_seed` 인증 공백은 **확인**. 같은 HOLD 가 F1 이 없는 CL2 에도 있다 (C0 73 · C1bp 70). F1 이 `no_seed` run 을 계통적으로 늘렸는지는 **모름** (3 / 0 / 1, p = 0.25). 아래 "T46b" 절 |
| **T46c** F1 채택 (한계 L1–L9 명시) · W1 (A12) 구현 · PA-1 ablation 초안 | F1 을 현재 extent 방법으로 채택하는가, close 이후 HOLD 를 어떻게 다루는가, ablation 을 어떻게 구성하는가 | **F1 채택 (사용자 판정, 2026-10-12) — 한계 L1–L9 를 함께 기록.** W1 구현은 **진행 중 (결과 없음)**. PA-1 ablation 구성은 **사용자 판정 대기**. 아래 "T46c" 절 |
| **T46c-W1** placed `no_seed` 인증 수정 (flag 기본 off) 의 검증 (V-T46c, CPU offline) | W1 이 T46b 의 what-if 를 코드로 재현하는가, off 에서 기록과 같은가 | **측정 완료 (2026-10-12 23:40 KST, PASS).** 기준선 일치 (legacy 14 / 15 · 15 / 15 · −29.031048 mm, cuRobo 10 / 15 · 15 / 15 · −9.171877 mm). off (W0) 가 기록 reason · certified 와 3 arm × 1350 / 1350 chunk 에서 같다. W1 의 close 이후 HOLD (total / uncertified) 는 C1a 107 / 37 → 70 / 0, C1aF1 183 / 130 → 54 / 1, C1aF1F2 122 / 118 → 6 / 2. **per-chunk what-if 이고 closed-loop 효과는 측정하지 않았다.** 사용자 판정으로 모든 Ablation arm 에 켠다 (바로 위 T46c 행의 "W1 구현은 진행 중 (결과 없음)" 은 이 행으로 대체). 아래 "T46c-W1 · T46d · T46f" 절 |
| **T46d** DX-04 선행 확인: LTE 대 global exclusion 비교가 성립하는 최소 non-authorized link 집합 (CPU offline, 18 run × 3 arm) | 현재 gripper scope 에서 LTE 와 global exclusion 이 다른 판정을 내는가, 어떤 link 를 넣어야 비교가 성립하는가 | **측정 완료 (2026-10-13 00:17 KST), 사용자 판정 (2026-10-13).** 기록 문구 (사용자 지정): **현재 gripper scope 에서는 LTE 와 global exclusion 의 비교가 퇴화하며, `link_*_arm_5` 를 추가했을 때 나타난 차이는 형상 근사 오류의 영향을 받는다.** ("효과 없음" 이 아니다.) gripper scope 에서 target 근처 non-authorized 행 0 / 349,920 (C1aF1, pre8close). arm_5 sphere 는 18 / 18 run 에서 target 50 mm 안이지만 MJCF arm_5 geom 은 18 / 18 run 에서 100 mm cap 이상. 판정: **A (주장 축소) 가 아니라 B 를 확장해 진행, C (arm_5 근사 구 그대로) 제외.** 아래 절 |
| **T46f** LTE 검증 블록: 형상 보정 (A14) → 진단 장면 → oracle extent 로 no exemption · global exclusion · LTE → AG3S extent | 지정 link 에만 target 접촉을 허용하면 필요한 손의 접촉을 유지하면서 비허용 link 의 접촉을 줄이는가 (가설, 사전 고정) | **계획 단계 (2026-10-13 01:23 KST). 측정 결과 없음.** 형상 보정 (A14) 이 진행 중이다. LTE 의 우수성을 전제하지 않는다. 아래 절 |

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
| T48b 범위 — held-out 7 run 의 held frame 이 **primary**, VLA 학습 episode 의 recorded run (ep0 · ep500 · ep1200) 은 **secondary 로 분리 보고**, E1z 는 exploratory | 사용자 판정 (2026-10-12, T48b.impl.md 확인 후 · 결과 전). 같은 scene (episode 1800) 의 run 은 독립 표본으로 세지 않는다. 빈 손 닫힘 값 0.457 · 관통 제외 2.0 mm 는 impl.md 의 근거대로 고정. E0 는 `closed_loop_gt` 와 contact state 로 이중 채점 (`handoff/T48b.task.md` 사전 고정 보완) | 16D (pilot 규모) |
| T49 — 명사 token query 를 시험한다. cell 선택은 **GT primary + label-free secondary**, **prompt 교체 시험 포함** (exploratory). 서빙 cell (L8H2) 은 유지 | 사용자 판정 (2026-10-12, T47 결과 후). offline GPU prefix pass 승인. 서빙 반영 (`attention_policy.py` 의 source 교체) 은 이 STEP 에서 하지 않고, 결과 → 사용자 판정 → 별도 STEP (`handoff/T49.task.md`) | 16D (pilot 규모) |
| T49b — 명사 query 의 σ · k 변형은 **exploratory**, 채택 근거가 아니다. cell · k · σ 선택은 T51b (공식 protocol) 로 정한다 | 사용자 판정 (2026-10-12, T49 결과 후). 이미 추출된 data 로 CPU 만 사용 (`handoff/T49b.task.md`) | 16D (pilot 규모) |
| T50 — **T50a (CPU) + T50b (GPU, `pi05_base`) 를 모두 진행**한다. 판정 임계 (C0 ≥ 0.95 · C1_instr ≥ 0.80) 는 episode 별 정확도의 **median** 에 적용 | 사용자 판정 (2026-10-12) 으로 T50a + T50b 진행. median 적용은 lead 가 결과 전 고정 (`handoff/T50.task.md`) | 16D |
| T50 — decision rule 의 출력은 "fine-tuning 탓이 아니다" (LoRA · base 모두 기준 미달) | **사전 고정 규칙의 출력이다 (사용자 판정이 아니다).** 규칙을 어떻게 읽고 무엇을 하는지는 사용자 판정 대기 (T50 절 §7) | 16D |
| (**제안, 판정 대기 — 미측정 해석 포함**) HSR 을 phase source 의 **보조 구성 요소** 로 둔다 | 검토 문서 §6 의 채택 기준 ("HSR 이 strong rule 보다 실행상 도움이 없으면 보조 구성 요소로 남긴다"). T48a (C1 둔감), T48b (E1 `place` 유지 12/13), T50 (counterfactual probe 도 기준 미달) 은 이 방향과 같은 쪽이나, **실제 strong rule 과의 비교와 gate on/off 실행 이득은 측정하지 않았다** (미측정 해석) | 16D |
| (**제안, 판정 대기 — 미측정 해석 포함**) **명사 query 로 identity · destination 을, L8H2 로 3D seed 를 얻는** 결합 | T49: 명사 query 가 2D peak · 분리 · prompt 교체에서 앞서고 L8H2 가 3D `identity_correct` · purity · non-target 손실에서 앞섰다. 그 결합이 실제로 3D 를 개선하는지는 **돌리지 않았다** (미측정 해석). 공식 비교는 T51b | 16D |
| T46c — **F1 (`--nontarget-evidence surface`) 을 develope_PLAN §0.10.1-3 의 현재 extent 방법으로 채택한다. 한계 L1–L9 는 명확하게 기록한다. F1 이 close 이후 HOLD 를 늘린 원인도 기록한다** | 사용자 판정 (2026-10-12): "권장하는 A 로 진행. 한계는 명확하게 작성하고 넘어갈 것. F1 이 grasp 이후 HOLD 를 늘린 원인을 작성할 것." 근거와 한계는 T46c 절 §1 · §2, 원인은 T46b · T46c §3 | 16D (pilot 규모, 18 쌍) |
| T46c — F2 (`--owned-ambiguous remove`) 는 채택하지 않고 대조 arm 으로만 남긴다 | **lead 의 정리 (T46c.task.md §1, 사용자 판정 문장에 직접 나오지 않음).** 이유: 이웃 과일 seed 를 지운다 (offline held-out 에서 잃은 이웃 seed 2,463) | 16D |
| T46c — close 이후 HOLD 의 고칠 곳은 F1 이 아니라 인증 규칙 (W1), W1 은 모든 arm 에 같이 적용한다 | **lead 의 설계 결정 (T46c.task.md §3 · §4).** W1 구현 (A12) 은 진행 중이고 결과가 없다. 사용자 판정이 아니다 | 16D |
| (**제안, 판정 대기**) PA-1 ablation 구성 — DT-02 extent (주) · identity · DT-05A contact handling | **lead 의 초안 (T46c.task.md §5).** 판정 필요: (a) visible-target oracle extent arm 을 만들지, (b) DT-05A 용 비권한 link 범위, (c) 표본 수 (18 쌍 유지 · 확대) | 16D |
| T51 — head · layer 를 하나로 고정해 쓰려면 선택의 당위성이 있어야 하고 실물 로봇 · LIBERO 에서도 같은 절차로 고를 수 있어야 한다. 범위는 **RB-Y1 sim protocol + 실물용 grasp-event self label 검증 + LIBERO**, primary 선택 지표는 **3D `identity_correct`**, 2D lift 는 depth 가 없는 플랫폼용 보조 지표 | 사용자 판정 (2026-10-12, `handoff/T51.task.md`) | 16D (+ `pi05_libero`) |
| T51 판정 보완 — **LIBERO L1 승인 (KNOWS 재현 확인 없이)**: 저장 frame 에 `pi05_libero` attention pass 를 한 번 돌린다 (새 rollout 없음). **RB-Y1 calibration 확장 승인**: `ep0` · `ep500` · `ep1200` (**VLA 학습에 쓰인 episode 의 기록**) 을 calibration 에 추가, 결과에 학습 episode 임을 명시. bootstrap 은 **scene 단위** | 사용자 판정 (2026-10-12, T51.impl.md 확인 후 · 결과 전). 근거: calibration 이 3 record (2.5 scene) 라 서로 다른 resample 이 10 개뿐이고 P4 inner CV 가 1 episode 로 학습 | 16D (calibration 6 record · 5 scene) |
| T51 집계 단위 — P2 점 추정은 **scene mean** (사전 고정). 첫 구현이 record 평균으로 순위를 매긴 것을 verifier 가 발견해 `--aggregate scene` 으로 고쳤고, **scene mean 결과가 primary**, record 평균은 "사전 고정과 다른 방식" 으로 보고만 한다. held-out 비교도 held-out scene mean, LIBERO 는 task 단위 | lead (2026-10-12, 결과 전 고정 · 수정). 사용자 판정이 아니다. 이 절 §4 | 16D |
| T51 — **서빙 반영은 이 STEP 에서 하지 않는다.** 선택 결과가 L8H2 와 다르더라도 교체는 별도 판정 | task 의 되돌아올 지점 (lead). 서빙 cell 은 T47 판정대로 L8H2 | 16D |
| T51 — 사전 고정 안정성 기준 (full-sample 1 위의 P(rank1) ≥ 0.5) 를 GT 선택이 **충족하지 못했으므로 (0.429) 단일 head 고정을 주장하지 않는다** | **사전 고정 규칙의 출력이다 (사용자 판정이 아니다).** task 의 되돌아올 지점이 정한 문구. 후속 (k > 1 · 명사 query) 은 T51b 와 판정 대기 항목 | 16D (5 scene) |
| T51b — action query 와 같은 protocol 을 명사 query 행에 돌려 비교한다 (offline GPU prefix pass, GPU lock) | 사용자 판정 (2026-10-12, T49 결과 후). T51 은 그대로 끝내고 추가 | 16D |
| T51b — **명사 단독 Ω 의 3D 강제 포함 cell = T49 primary 명사 cell L8H6 · L5H4 · L4H2** (합친 Ω 에서는 T51 의 forced 에 이 3 개를 더한다). **(c) 비교의 primary 통계 = scene 단위 paired 차이 + scene bootstrap 95 % CI**, frame 단위 McNemar 는 참고값 | lead (T51b.impl.md ★ 두 항목에 대한 판정, 결과 전 고정). 사용자 판정이 아니다. 구현이 이 forced 를 코드에 넣지 않은 일은 §12 에 기록 | 16D |
| T46c — **W1 (`--placed-no-seed-certify`) 을 Ablation 의 모든 arm 에 켠다. 코드의 기본값은 off 로 두고 실험 arm 에서 명시적으로 켠다** | 사용자 판정 (2026-10-12, 추천안 채택). **2026-10-04 판정 ("`no_seed` HOLD 는 그대로 둔다") 을 이번에 바꾼다.** 근거: 그 판정의 전환 신호 관측 (T46b) · V11AB (C0) 에도 같은 HOLD. 위 "close 이후 HOLD 의 고칠 곳은 F1 이 아니라 인증 규칙 (W1)" 행 (lead 의 설계 결정) 을 사용자 판정으로 확정한다 (`handoff/T46c.task.md` "사용자 판정" 절) | 16D (pilot 규모) |
| T46c — DX-04 선행 확인의 비허용 link 범위는 develope_PLAN §0.10.1-1 의 **최소 범위** (wrist 등) 부터, 표본은 **18 쌍 유지** (CL2 · CL3 와 같은 9 episode × 2 seed), 순서는 develope_PLAN §11 (DX-04 선행 확인 → DX-01 identity / extent 분리 + visible-target oracle → contact handling → Ablation closed-loop) | 사용자 판정 (2026-10-12, 추천안 채택). 근거 (task 문서 인용): develope_PLAN §0.4 "유리한 결과를 위해 표본을 늘리지 않는다". 범위는 T46d 에서 기록으로 고른다 | 16D |
| T46d — **기록 문구는 "현재 gripper scope 에서는 LTE 와 global exclusion 의 비교가 퇴화하며, `link_*_arm_5` 를 추가했을 때 나타난 차이는 형상 근사 오류의 영향을 받는다"** 이다. "효과 없음" 으로 쓰지 않는다 | 사용자 지정 (2026-10-13 KST, `handoff/T46d.task.md` "결과 기록과 사용자 판정") | 16D |
| T46d · T46f — **T46d 의 A (link-specific 효과 주장 축소) 가 아니라 B 를 확장해 진행한다: 형상 보정 (FT sensor coverage · arm_5 근사 보정) → 진단 장면 → oracle extent 3 조건 closed-loop. C (arm_5 근사 구 그대로) 는 제외한다. LTE 의 우수성을 전제하지 않는다** | 사용자 판정 (2026-10-13 KST, `handoff/T46f.task.md` 머리말). A · C 의 괄호 설명은 `T46f.task.md` 의 문구이고, B 는 "B 를 확장" 으로만 적혀 있어 내용은 T46f 의 단계 (아래 절) 로 읽는다 | 16D |
| T46f — 가설 · 주 지표 · 접촉 정의 · DT-02 와 DT-05A 블록 분리 · 진단 장면 기준 S1 – S4 · 단계 · 중단 기준 | **lead 의 사전 고정 (`handoff/T46f.task.md`).** 사용자 판정 문장에는 직접 나오지 않는다. FT sensor 비접촉을 이 작업의 요구사항으로 정의한 것도 lead 의 정의다 ("센서 보호") | 16D |

---

## 재측정 대기 — T47 이후

각 항목의 번호는 이 파일 안의 번호이고, 괄호는 [`AG3S_T0T6_LOG.md`](AG3S_T0T6_LOG.md) "재측정 대기" 표에서 이 항목들이
원래 받았던 번호다 (그쪽에서는 옮겼다는 한 줄만 남겼다). T46 이전 항목은 그쪽 표를 본다.

| # | 무엇 | 값 | 상태 |
|---|---|---|---|
| 1 (T0T6 30) | (T47 · T47b 가 추가) **grasp latch 를 포함한 e2e** 에서의 head 비교, cuRobo ESDF backend e2e, T47b source 의 selection record 3D | offline e2e 는 CPU legacy ESDF 이고 grasp latch 가 없다 (`T47.verify.json not_measured`, `T47b.verify.json not_measured`). selection record 3D 는 T47 의 5 source 만 있다 | **측정 안 됨** |
| 2 (T0T6 31) | (T47 · T47b 가 추가) **held-out 의 일반화** — banana · pear 각 1 record 와 same-scene 2 record 뿐이다. `cam_right_wrist` 2D 는 4 record 모두 target < 200 px 라 제외 | record 를 늘리려면 GPU attention pass 가 필요하다 (사용자 승인 사항) | **측정 안 됨** |
| 3 (T0T6 32) | (T47b 가 추가) **σ=1 이 이웃 과일 흡수를 늘리는 경로**, 그리고 B3 수치 시험의 다른 source (L8H2 · oracle top-3 σ=1 에서만 시험) | `identity_correct` 54 → 49 (L8H2, σ=1) 는 측정했으나 smoothing 이 seed 를 번지게 하는지는 분리해 재지 않았다 | **원인 미확정** |
| 4 (T0T6 33) | (T48a 가 추가) **probe 가 "물체를 쥐고 있다" 를 읽는가, "팔이 이 자세에 있다" 를 읽는가** — 같은 팔 자세에서 물체의 유무만 바꾼 image 로 가른다 | T48a 의 C2 는 같은 episode 의 다른 frame 의 image 를 통째로 이식해 팔 자세와 물체 상태가 함께 바뀐다. dataset 은 scripted 라 둘이 같이 움직인다 (frame index 만으로 val 1.000) | ~~**T48b 로 진행 (사용자 판정, 2026-10-12)** — E0 re-render parity 먼저~~ → **T48b 측정 완료, 결론은 판정 대기.** E0 parity 통과, E1 · E2 · E3 측정 (E1 n = 13 · E2 n = 7 의 pilot). "자세를 읽는가" 의 해석은 T48b 절 §7 |
| 5 (T0T6 34) | (T48a 가 추가) **실제 strong rule (`grasp_latch.py` + 실행 feedback) 과의 비교** | T48a 는 offline strong-rule proxy 만 썼다 (`T48.verify.json not_measured`) | **측정 안 됨** |
| 6 (T0T6 35) | (T48a 가 추가) **batch 1 (서빙 batch size) 전체 extract · cuRobo 회귀 기준선 · extract 시작 시점의 pod working set** | extract 는 `--batch 4` (batch parity 는 첫 batch n = 4 에서만), cuRobo 변형은 돌리지 않았고, lead 요청이 extract 도중 도착해 시작 시점 working set 이 없다 (`T48.verify.json not_measured`) | **측정 안 됨** |
| 7 | (T48b 가 추가) **E1 · E2 의 표본 확대** — held-out 의 `held` host 16 개 중 편집이 남은 것은 E1 13 · E2 7 이고 2 run · 2 scene (ep1807 · ep1850) 에서 나왔다 | 관통 제외로 줄었다 (E2 는 16 → 7). 더 늘리려면 held 구간이 긴 run 의 recorded data 가 필요 | **측정 안 됨** |
| 8 | (T48b 가 추가) **E3 에서 probe 의 `place` 가 물체를 읽은 것인지 finger 닫힘을 읽은 것인지** | E3 는 과일과 finger 를 함께 바꾼다 (finger 를 과일 폭에 맞춤). simple rule (finger 값만 봄) 도 99/99 로 `place` 라 이 편집만으로는 가를 수 없다 (미측정 해석) | **원인 미확정** |
| 9 | (T49 · T49b 가 추가) **명사 query 의 3D `identity_correct` · purity · non-target 손실 열세의 원인**, 명사 map 과 L8H2 의 결합 (제안 (b)) | σ · k 변형 (T49b) 으로는 L8H2 수준에 못 미쳤다. 명사 map 으로 seed 를 고르는 경로와 L8H2 seed 의 결합은 돌리지 않았다 | **원인 미확정 · 결합 측정 안 됨** → T51b |
| 10 | (T49 가 추가) **grasp latch 를 포함한 e2e · cuRobo ESDF backend · held-out 4 record 보다 큰 표본** (T47 의 1 · 2 번과 같은 한계) | T49 e2e 는 legacy ESDF, grasp latch 없음, held-out 4 record × 15 frame. cuRobo 기준선은 GPU 미사용 지시로 돌리지 않음 (`T49.verify.json not_measured`) | **측정 안 됨** |
| 11 | (T50 이 추가) **base · LoRA 모두 `kv_L4` 위 counterfactual probe 가 기준 미달인 이유** — 다른 layer · 다른 feature · contrastive objective 에서는 어떤가 | T50 은 `kv_L4_v1` 의 feature 정의 (layer 4 · V · head 0, text · image 평균) 와 LogReg 만 시험했다 | **측정 안 됨** |
| 12 | (T50 이 추가) **실제 gate on / off 실행 이득과 실제 strong rule 비교** (HSR 위치 판정의 근거) | T48a · T48b · T50 모두 offline 이다 | **측정 안 됨** |
| 13 (T0T6 T46-1) | (T46b 가 닫음 · T46c 가 추가) **close 이후 uncertified HOLD 증가의 원인** — 이전 로그의 T46-1 ("원인 미측정") 에 대한 답 | T46b: placed phase 의 grounding `no_seed` 인증 공백 (C1a 37 · C1aF1 130 · C1aF1F2 118 중 `no_seed` 37 · 129 · 116). F1 이 직접 만든 것이 아니다. 이전 로그의 T46-1 행은 그대로 남아 있다 (이 STEP 에서 수정하지 않음) | **원인 확인 (d). `no_seed` 가 왜 생기는지는 미확정 (#14)** |
| 14 | (T46b 가 추가) **placed 뒤 attention 상위 5 % seed 가 전부 support mask 안에 놓이는 이유** | 그 chunk 에서 seed 가 전부 support mask 에 있음은 측정했다 (`no_seed` 129 chunk 모두 min 1.0). 왜 그렇게 되는지는 중단 기준에 따라 추적하지 않았다 | **원인 미확정** |
| 15 | (T46b 가 추가) **F1 이 placed `no_seed` run 을 계통적으로 늘리는가** | F1 만 3 · C1a 만 0 · 둘 다 1 (15 pair, McNemar exact p = 0.25). 연결은 궤적 분기를 통해서만 있고 (F1 기록 0 / 130), 표본이 작다 | **모름** |
| 16 | (T46b 가 추가) **W1 · phase-gated F1 의 closed-loop 효과, 그리고 placed HOLD 가 completion time · safe success 에 주는 영향** | W1 수치는 기록 chunk 위 per-chunk what-if 이고, HOLD 가 풀리면 이후 궤적이 달라진다. F1 의 `no_seed` run 4 개는 모두 task success 이고 `no_seed` 시작이 place 이후다. 홈 복귀까지의 시간은 재지 않았다 (`T46b.verify.json not_measured`) | **측정 안 됨** (W1 구현은 T46c A12 진행 중) |
| 17 | (T46c 가 추가) **F1 한계 L1–L9 를 줄이는 측정** — 표본 확대 · banana · 비구형 물체 · 실물 · 다른 checkpoint · real-time | T46c §2 의 한계 표. 이 중 F1 의 closed-loop 효과는 18 쌍 pilot 이다 | **측정 안 됨** |
| 18 | (T51 이 추가) **GT 선택의 안정성 — calibration scene 확대** | scene 5 개 (그중 3 개는 VLA 학습 episode) 에서 `last:L12H4` 의 scene 별 `identity_correct` 가 0.363 – 0.885 로 흩어진다. P(rank1) 0.429 < 0.5. 학습에 쓰이지 않은 scene 을 더한 calibration 은 하지 않았다 | **측정 안 됨** (원인은 scene 수 부족인지 head 간 실제 동률인지 미확정) |
| 19 | (T51 이 추가) **held-out 의 변별력** — 선택 `last:L12H4` · `last:L12H3` · serving `last:L8H2` 가 3 scene · 60 frame 에서 같은 frame 을 맞히고 같은 frame 을 틀린다 (0.911 동률) | 이 held-out 으로는 세 cell 을 가를 수 없다. 더 큰 held-out · 학습에 없는 물체 · 다른 scene 이 필요 (#2 와 같은 한계) | **측정 안 됨** |
| 20 | (T51 이 추가) **self label 이 cell 을 가르는가, 실물에서 되는가** | sim 에서 `self_identity` 가 상위 9 cell 모두 1.0 으로 포화해 선택은 2D lift tie-break 가 정했다. grasp event 가 없는 record 가 10 개 중 4 개, "쥐기 전 정지" 가정은 6 개 중 3 개 record 에서 1 cm 를 넘어 움직였다. 실물 (`grasp_latch` HELD) 은 시험하지 않았다 | **측정 안 됨** (포화의 원인은 미확정) |
| 21 | (T51 이 추가) **P1 `low_mass` 제외의 의미** — L8H2 가 cam_high 의 mass median 0.006293 으로 제외됐으나 held-out 에서 선택과 동률이고 LIBERO 에서도 2D lift 4 위 (제외 안 했을 때) | threshold (0.01) 는 사전 고정이라 다시 맞추지 않았다. `low_mass` 가 head 의 유용성을 반영하는지는 재지 않았다 | **측정 안 됨** |
| 22 | (T51 이 추가) **LIBERO 3D · KNOWS 재현** — LIBERO 는 agent view 2D lift 만 있고 3D `identity_correct` 가 없다. KNOWS 재현 (`--init zeros`) 은 사용자 판정으로 하지 않았다. wrist GT 는 저장되어 있지 않다 | RB-Y1 에서 2D lift 1 위 (`last:L12H3`) 는 3D 에서 6 위였다 (§6). 2D 만으로 고른 LIBERO 선택이 3D 에서도 같은지는 모른다 | **측정 안 됨** |
| 23 | (T51b 가 추가) **명사 대 action query 의 일반화 · 후보 구성 의존** — held-out 3 scene 의 scene 단위 차의 CI 가 0 에 닿는다. 합친 Ω 의 2D 상위 16 은 전부 명사 cell 이라 action cell 은 forced 3 개 (`last:L8H3` · `last:L10H7` · `last:L12H4`) 로만 3D 에 들어갔다 | (b) 의 P(rank1) 0.753 은 그 후보 구성 아래의 값이다. action 후보를 넓힌 합친 Ω 는 돌리지 않았다. LIBERO 의 명사 행은 범위 밖 | **측정 안 됨** |
| 24 | (T51 · T51b 가 추가) **grasp latch 를 포함한 e2e · cuRobo backend · 서빙 반영 후의 효과** | T51 의 3D 는 legacy ESDF offline replay (subtask gate off, grasp latch 없음). cuRobo 회귀 기준선은 GPU 미사용 지시로 돌리지 않았다 (`T51.verify.json not_measured`) | **측정 안 됨** |
| 25 | (T46c-W1 이 추가) **W1 의 closed-loop 효과** — HOLD 가 풀리면 이후 궤적이 달라진다. home 복귀 시간 · safe success · 새 위반 · 접촉 | replay 는 기록 chunk 의 per-chunk what-if 다 (`T46c.verify.json not_measured`). #16 (T46b 가 추가한 같은 항목) 의 구현 쪽 | **측정 안 됨** (모든 Ablation arm 에 W1 on 이므로 CL4 이후의 closed-loop 에서 처음 잰다) |
| 26 | (T46c-W1 이 추가) **"위반 행이 있으면 HOLD 유지" 의 기록 chunk 시험** | CL3 기록에는 그런 chunk 가 없다 (`not_measured`). 단위 test 로만 덮인다 | **측정 안 됨** (기록으로는) |
| 27 | (T46c-W1 이 추가) **V11AB (C0) 에서의 W1 what-if · 전체 pytest · `ta_offline` 의 TO 재풀이 variant (W0 / W1 / F1W1)** | certification replay 의 입력은 CL3 의 세 arm (C1a · C1aF1 · C1aF1F2) 이다. verifier 는 전체 pytest 와 TO 재풀이 variant 를 돌리지 않았다 (`not_measured`) | **측정 안 됨** |
| 28 | (T46d 가 추가) **형상을 보정한 뒤의 LTE 대 global exclusion 차이** — arm_5 에서 나타난 LTE ≠ global 의 크기가 형상 근사 오류를 뺀 뒤에도 남는가 | T46d 는 구 모델 위의 field-level re-query 만 했다. 보정 형상 (FT sensor cover · arm_5 보정) 은 T46f A14 가 만든다 | **측정 안 됨** (T46f 단계 1 이후) |
| 29 | (T46d 가 추가) **TrajOpt 재풀이 · closed-loop 에서의 TO 시간 · 성공 · 접촉** | LTE ≠ global 은 `d − r` 을 임계값과 비교한 field-level 재계산이고 TO verdict 는 재현하지 않았다. TO 시간은 구 수 선형 추정이다 (`T46d.verify.json not_measured`). 재계산 fidelity: 기록 clearance 와 재조회 `d_full` 의 차가 policy-active chunk 에서 최대 22.00 mm (C1aF1), 손가락 행은 closing chunk 에서 기록과 최대 26.3 mm 달랐다 (손가락 관절 경로를 재현하지 않음) | **측정 안 됨** |
| 30 | (T46d 가 추가) **MJCF 거리의 100 mm cap 이후 · `FT_SENSOR` / `EE_BODY` 의 0.0 mm 거리가 접촉 필터를 거친 겹침과 구별되지 않음 · held 구간 (lift / carry 대부분) 의 LTE ≠ global · cuRobo 회귀 기준선 (GPU 미사용)** | `mj_geomDistance` 의 distmax 100 mm 로 arm_5 / arm_6 거리는 "100 mm 이상" 으로만 안다 (`T46d.verify.json not_measured`) | **측정 안 됨** |
| 31 | (T46f 가 추가) **T46f 의 모든 단계 결과** — 형상 보정 audit · 진단 장면 수 (S1 – S4 를 만족하는 dev / eval 배치) · 3 조건 closed-loop (oracle · AG3S extent) · 주 지표 | 계획 단계다. 결과 없음 | **측정 안 됨 (계획만)** |

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

---

## T48b — HSR counterfactual (b): 같은 팔 자세에서 물체의 유무만 바꾸면 phase probe 는 반응하는가 (2026-10-12 KST)

**이 절이 답하는 물음 (목적).** T48a 는 probe (`kv_L4_v1`, 위 "T48a" 절) 가 state 문자열이 아니라 image 를 따른다는 것까지 보였지만, image 안에서 **"물체를 쥐고 있다" 는 작업 상태** 를 읽는지 **"팔이 이 자세에 있다" 는 자세 순서** 를 읽는지는 가리지 못했다 (T48a 결과 (c)). 데이터가 실패 · retry 없는 scripted 시연이라 팔 자세와 쥠 상태가 항상 같이 움직이기 때문이다. T48b 는 기록된 16D step 의 qpos 에서 **대상 과일의 free joint 와 쓰는 손 finger joint 만** 바꾸고 MuJoCo (osmesa, CPU) 로 policy image 규약 그대로 다시 그려 (팔 자세는 그대로, 물체의 유무 · 위치만 다름) 같은 probe 에 넣는다. 편집의 종류는 용어 절 "T48b" 표 (E0 · E1 · E1s · E1z · E2 · E3) 에 풀어 두었다. 정답은 gripper 값이 아니라 **MuJoCo contact 로 정한 물리 predicate** 다.

**한 줄 결과 (before → after).**

- **before.** T48a 후: probe 는 image 를 따르나 무엇을 읽는지는 열려 있었다.
- **after — 측정된 사실.**
  - **re-render parity 통과.** E0 (편집 없이 re-render) 가 기록 image 와 **599 / 599 frame (held-out 260 + train_range 339) 에서 3 camera 모두 bit-identical** (MAD 0 · max abs 0, state 차이 0.0), probe 예측 일치 599 / 599 (통과 기준 ≥ 0.95).
  - **held-out primary 의 편집 결과** (frame 합산 k/n, "= target" 은 정답과 같은 예측). 정답이 `pick` 인 편집에서 probe 가 `pick` 으로 읽은 비율: **E1 (empty grasp) 1/13** (`place` 유지 12/13), **E1s 4/13**, **E2 (slip) 3/7**, E1z (exploratory) 5/13. 정답이 `place` 인 **E3 (phantom hold)** 에서 probe `place` **97/99**.
  - **simple rule 과 offline proxy** (proxy 는 strong rule 이 아니다): E1 에서 rule 0/13 · proxy 1/13, E3 에서 rule 99/99 · proxy 0/99. rule 은 finger 값 (state) 만 보고 proxy 는 host record 의 이력 (PRE → CLOSING → HELD) 을 쓰기 때문에 같은 편집에서 서로 반대 방향의 값이 나온다 (정의에서 오는 값이며 이 로그는 더 해석하지 않는다).
  - **규모.** E1 · E2 의 held-out 값은 **2 run · 2 scene (ep1807 · ep1850) 에서 나온 pilot** 이다 (E1 은 ep1807 11 + ep1850 2 frame, E2 는 5 + 2). E3 는 4 scene (ep1807 17 · scene ep1800 56 · ep1850 11 · ep1900 15 frame).
- **"자세를 읽는가" 의 해석은 하지 않았다.** task 의 되돌아올 지점이 "E1 · E2 에서 probe 가 `place` 를 유지하면 자세 순서를 읽는다" 로 적어 둔 조건에 해당하는 값은 §7 에 두고, HSR 의 위치는 판정 대기로 둔다.

**읽는 법과 출처 규약.**

- 수치는 `T48b.verify.json` 의 `numbers` 에서만 왔다. 같은 파일 숫자의 덧셈은 "(산출)" 로 적었다. 구현의 설계 값 (host 선택 규칙 · 편집 정의 · 렌더 자세 규칙) 은 `T48b.impl.md` 의 서술이며 그 수치는 verify.json 에 있는 것만 옮겼다.
- 해석의 출처를 가른다: 사용자 판정 "(사용자 판정)", lead 가 task 에 쓴 것 "(lead)", 측정으로 확인하지 않은 해석 "(미측정 해석)".
- **독립 단위.** frame 은 독립 trial 이 아니다. 표는 frame 합산 k/n 을 적고, scene (독립 단위) 별 probe 값은 figure 의 마커와 verify.json `probe_per_scene` 에 있다. long · t2 · headfix · ep1800 은 같은 scene (episode 1800) 이라 1 scene 이다.
- **환경.** CPU render (`.venv-ag3s`, osmesa, 1 thread) + GPU extract 1 회 (offline prefix pass, GPU lock 아래, 시작 전 compute process 0) + CPU score. 회귀 기준선은 render 앞과 score 앞 두 번 일치했다 — legacy 14 / 15 · has_target 15 / 15 · frame0 `clearance_before` −29.031048 mm. cuRobo 기준선은 이 STEP 에서 돌리지 않았다. HEAD `28b98160` (T48b 파일 3 개는 untracked, 새 파일뿐).

### 1. 사용자 판정과 타임라인 (KST)

| # | 판정 · 사건 (2026-10-12) | 이 절에서 |
|---|---|---|
| 1 | **(사용자 판정, T48a 결과 후)** T48b 진행. 16D policy image re-render parity (E0) 를 먼저 확인하고 통과하지 못하면 거기서 멈춰 보고. GPU 추출은 T46 의 CL3 가 끝난 뒤 | §3 |
| 2 | **(사용자 판정, T48b.impl.md 확인 후 · 결과 전)** held-out 7 run 의 `held` frame 이 **primary**, VLA 학습 episode 의 recorded run (ep0 · ep500 · ep1200) 은 **secondary 로 분리 보고** (편집 image 는 학습에 없으나 장면은 익숙할 수 있음을 명시). 같은 scene 의 run 은 독립 표본으로 세지 않는다. E1z 는 exploratory | §4 |
| 3 | **(lead)** 정답은 물리 predicate. 빈 손 닫힘 값 0.457 (기록된 빈 손 닫힘 22 개의 median) · 관통 제외 기준 2.0 mm (편집하지 않은 기록 frame 의 최대 1.83 mm) 고정. E0 는 `closed_loop_gt` 와 contact state 두 기준으로 채점 | §2, §3 |

| 시각 (KST) | 무엇 | 출처 |
|---|---|---|
| 17:03 – 17:10 | CPU render (`--stage render --extra-runs train_range`, rc 0, stage wall 443.5 s). 앞서 회귀 기준선 | `runs[render]` 08:03:03Z – 08:10:28Z · `cost_5.render_stage_wall_s` |
| 19:23:35 – 19:28:16 | GPU extract (`--stage extract --batch 4`, rc 0, wall 280.9 s, retry 0, nvidia-smi peak 8804 MiB) | `gpu_extract[0]` 10:23:35Z – 10:28:16Z |
| 21:00 – 21:20 | CPU score · figure (회귀 기준선 재확인 일치) | verify.json `time_kst` |

### 2. 설계와 구현 (새 파일만, `T48b.impl.md`)

| ID | host frame | 편집 | 정답 | 묻는 것 |
|---|---|---|---|---|
| **E0** | 사용 frame 전부 | 없음 (re-render 만) | 원래 정답 (이중 채점) | parity |
| **E1** | `held` (파지 후 · 놓기 전) | 과일 → 초기 table 위치, finger → 0.457, state 도 같게 | `pick` | 손은 닫혔고 팔은 운반 자세인데 물체가 없다 |
| **E1s** | `held` | 과일만 옮김, finger · state 그대로 | `pick` | E1 에서 finger image · state 효과를 뺀다 |
| **E1z** (exploratory) | `held` | E1 과 같되 finger → 0 (joint 한계) | `pick` | 기록된 빈 손 (0.457) 과 완전히 닫힌 손의 차이 |
| **E2** | `held` | 과일 → 손 바로 아래 table 위 (xy 그대로, z · 자세 = 초기), finger · state 는 E1 과 같게 | `pick` | 물체가 손 근처에 있으나 쥐어져 있지 않다 |
| **E3** | `reach` (쓰는 손이 처음 닫히기 전, 과일 `on_table`) | 과일 → finger 사이, finger → 과일 폭에 맞춤, state 도 맞춤 | `place` | 팔은 접근 자세인데 물체는 손 안에 있다 (E1 의 대칭) |

- **렌더 자세 규칙.** 기록의 qpos 는 마지막 `mj_step` 뒤 값이지만 camera 가 본 kinematics 는 그 step 앞의 것이라 `mj_integratePos(qpos, qvel, −dt)` 로 한 step 되돌려 그린다 (첫 record 는 예외). 이 규칙으로 parity 가 bit 단위로 맞았다 (`T48b.impl.md` §1; 규칙 없이는 움직이는 link 가장자리가 다르다고 impl 이 smoke 로 확인).
- **편집 지역성 (verifier 독립 점검, 10 run).** E1s 는 과일의 free joint 7 개만 바뀌고 state 는 그대로다. E1 · E1z · E3 는 과일 7 개 + 쓰는 손 finger joint 2 개가 바뀌고 state 는 쓰는 gripper 한 차원만 바뀐다. E2 는 과일의 z · 자세 5 개 + finger 가 바뀌고 xy 는 그대로다.
- **빈 손 닫힘 값 `g_empty` = 0.457** (n = 22 기록, p0 0.389 · p5 0.402 · p95 0.495 · p100 0.688). 기준 hold 값 (dataset 의 닫는 명령): apple 0.723 · banana 0.495 · orange 0.667 · pear 0.673.
- 테스트: `test_t48b_*` 43 passed · 0 failed (`numbers.pytest_t48b`).

### 3. 검증 — E0 re-render parity (통과 기준 = 예측 일치 ≥ 0.95, MAD 가 render 잡음 수준)

![T48b CPU parity · 제외 표](figures/t48/t48b-cpu-parity-exclusions-table.png)

[`t48b-cpu-parity-exclusions-table.png`](figures/t48/t48b-cpu-parity-exclusions-table.png) — #0 pixel · state parity, #4 편집별 포함 / 계획 frame 수, render 비용.

| 점검 | held-out primary (7 run) | train_range secondary (3 run) |
|---|---|---|
| E0 frame 수 | 260 | 339 |
| 3 camera 모두 bit-identical | **260 / 260** | **339 / 339** |
| MAD 최대 · max abs · state max abs | 0.0 · 0 · 0.0 | 0.0 · 0 · 0.0 |
| probe 예측 일치 (re-render 대 기록 image) | **260 / 260** | **339 / 339** |
| probe 확률 변화 (599 frame) | median · p99 · max 모두 0.0 | |
| 같은 자세 재렌더가 동일 | 전 run 참 | |

- 일치율 1.0 (≥ 0.95), pixel gate (frame · camera MAD 최대 ≤ 1.0) 통과 → **E1 – E3 진행.** rule · proxy · frame-index 의 예측 일치도 599 / 599 이다.
- state 재구성 (`build_obs("rby1_16d")` 규칙) 과 기록 state 의 차이 최대 0.0 (전 frame).
- 독립 확인: 편집 · 확률을 verifier 가 따로 다시 계산했다 — probe 를 numpy 로 다시 적용한 argmax 불일치 0 / 1510 (확률 최대 차 5.0e-7), `metrics.json` 대 독립 재채점 124 / 124 field 일치.

**E0 이중 채점** (정답 기준을 가름; frame 합산 k/n). held-out primary 에서 `closed_loop_gt` 와 contact state 가 다른 (모호하지 않은) frame 은 39 개, 모호 frame 은 6 개다 (long · t2 의 빈 손으로 닫힌 frame 에서 `closed_loop_gt` 는 `held` 로 친다). train_range 는 1 개 · 5 개.

| 채점 | 대상 | probe | simple rule | offline proxy | frame-index |
|---|---|---:|---:|---:|---:|
| `closed_loop_gt` | held-out primary | 228 / 260 | 227 / 260 | 204 / 260 | 164 / 260 |
| | ep1807 만 | 72 / 75 | 75 / 75 | 72 / 75 | 66 / 75 |
| | train_range | 325 / 339 | 339 / 339 | 332 / 339 | 296 / 339 |
| contact state | held-out primary | 215 / 254 | 182 / 254 | 181 / 254 | 152 / 254 |
| | ep1807 만 | 72 / 74 | 74 / 74 | 72 / 74 | 65 / 74 |
| | train_range | 324 / 334 | 333 / 334 | 330 / 334 | 292 / 334 |

(frame-index baseline 은 채점 1510 row 중 1210 row 에서 nearest-`t_step` fallback 을 썼다 — T48a 의 ep1807 과 같은 정의.)

**편집 · 제외.** 관통 > 2.0 mm 인 편집은 제외했다. 제외 사유는 held-out 에서 전부 `penetration` 이다 (train_range 는 `penetration` 과 `finger_touch`: 쓰는 손 finger 가 과일에 닿은 채 `on_table` 로 판정).

| 편집 | held-out primary 계획 → 포함 (제외) | ep1807 만 | train_range 계획 → 포함 (제외) |
|---|---|---|---|
| E1 | 16 → 13 (3) | 12 → 11 | 38 → 30 (8) |
| E1s | 16 → 13 (3) | 12 → 11 | 38 → 31 (7) |
| E1z | 16 → 13 (3) | 12 → 11 | 38 → 31 (7) |
| E2 | 16 → **7** (9) | 12 → 5 | 38 → 13 (25) |
| E3 | 123 → 99 (24) | 19 → 17 | 66 → 62 (4) |

- 독립 scene 단위로 쓰면 held-out 의 **E1 · E1s · E1z 는 ep1807 11 + ep1850 2 (scene 2 개), E2 는 5 + 2, E3 는 ep1807 17 · scene ep1800 56 · ep1850 11 · ep1900 15** 이다. scene ep1800 (long + t2 + headfix + ep1800) 과 ep1900 에는 `held` host 가 없어 E1 · E2 가 0 이다.
- E1 에서 과일을 초기 table 위치로 옮긴 결과 host 기준 xy 50 mm 안에 놓인 포함 편집이 held-out 6 / 13, train_range 10 / 30 이다 (정의와 값은 `indep/e1_e2_fruit_xy.json`; 이 로그는 해석하지 않는다). train_range 의 기록된 관통 최대는 2.10 mm 로 (held-out 1.83 mm) 편집 제외 기준 2.0 mm 와 가깝다.

**비용 (#5).** re-render 는 frame 당 median 0.286 s (3 camera, 1 thread, osmesa; mean 0.348 · p95 0.603, n = 599), 편집 image 는 median 0.291 s (n = 312), E3 의 배치 계산 median 0.442 s (n = 189) — render 중 host load average 16.7 → 25.9 (다른 session 의 GPU closed-loop job 포함). prefix pass 는 input 당 median 14.98 ms (batch 4, 1510 input, 385 call, device 시간), JAX peak 7.6 GB, nvidia-smi peak 8804 MiB, load 10.9 s, extract wall 278.4 s.

### 4. 결과 — 편집 × 방법 (정답과 같은 예측 비율, frame 합산 k/n)

![T48b 편집 × 방법](figures/t48/t48b-score-accuracy-bars.png)

[`t48b-score-accuracy-bars.png`](figures/t48/t48b-score-accuracy-bars.png) — 세 panel (held-out primary / ep1807 만 / train_range secondary). 막대 = frame 합산, 마름모 = scene 별 probe 값 (독립 단위). E1z 는 음영 (exploratory). 표는 [`t48b-score-table.png`](figures/t48/t48b-score-table.png) (sidecar 에 같은 수치).

**held-out primary (정답과 같은 예측: E1 · E1s · E1z · E2 는 `pick`, E3 는 `place`)**

| 편집 (정답) | n | probe | simple rule | offline proxy | frame-index |
|---|---:|---:|---:|---:|---:|
| E1 empty grasp (`pick`) | 13 | **1 / 13** | 0 / 13 | 1 / 13 | 2 / 13 |
| E1s empty, state 고정 (`pick`) | 13 | **4 / 13** | 0 / 13 | 0 / 13 | 2 / 13 |
| E2 slip (`pick`) | 7 | **3 / 7** | 0 / 7 | 1 / 7 | 2 / 7 |
| E3 phantom hold (`place`) | 99 | **97 / 99** | 99 / 99 | 0 / 99 | 4 / 99 |
| E1z, finger 완전 닫힘 (exploratory) | 13 | 5 / 13 | 0 / 13 | 1 / 13 | 2 / 13 |

- probe 의 예측 분포 (`pick` / `place` / `home`): E1 [1, 12, 0], E1s [4, 9, 0], E2 [3, 4, 0], E1z [5, 8, 0], E3 [2, 97, 0]. 같은 host frame 의 E0 에서 probe 의 평균 확률은 E1 의 host 13 개에서 `place` 0.997 이고, E1 편집 뒤 평균은 `pick` 0.110 · `place` 0.877 · `home` 0.012 다. E1s 는 0.282 / 0.699 / 0.019, E2 는 0.375 / 0.609 / 0.017, E1z 는 0.392 / 0.552 / 0.056. E3 는 E0 에서 `pick` 0.995 이던 평균이 편집 뒤 `place` 0.962 로 바뀐다.
- E0 대비 예측이 바뀐 frame 수 (probe): E1 1 / 13, E1s 4 / 13, E2 3 / 7, E1z 5 / 13, E3 97 / 99. rule 은 E3 99 / 99 만 바뀌고 (E1 · E1s · E1z · E2 는 0), proxy 는 E1 1 · E1z 1 · E2 1 · E3 0.
- scene 별 probe (`pick` 으로 읽은 수): E1 ep1807 1 / 11 · ep1850 0 / 2, E1s 4 / 11 · 0 / 2, E2 3 / 5 · 0 / 2, E1z 3 / 11 · 2 / 2. E3 ep1807 17 / 17 · scene ep1800 56 / 56 · ep1850 11 / 11 · ep1900 13 / 15 (`place` 으로 읽은 수).

**ep1807 만 (유일하게 probe · VLA 어느 학습에도 없고 SUBTASK-b closed-loop 평가에 쓰인 run)**

| 편집 | n | probe | rule | proxy | frame-index |
|---|---:|---:|---:|---:|---:|
| E1 | 11 | 1 / 11 | 0 / 11 | 1 / 11 | 0 / 11 |
| E1s | 11 | 4 / 11 | 0 / 11 | 0 / 11 | 0 / 11 |
| E2 | 5 | 3 / 5 | 0 / 5 | 1 / 5 | 0 / 5 |
| E3 | 17 | 17 / 17 | 17 / 17 | 0 / 17 | 2 / 17 |
| E1z | 11 | 3 / 11 | 0 / 11 | 1 / 11 | 0 / 11 |

**train_range secondary (VLA fine-tune episode; ep0 · ep1200 은 probe 학습 episode 이기도 함 — 장면이 익숙할 수 있다)**

| 편집 | n | probe | rule | proxy | frame-index |
|---|---:|---:|---:|---:|---:|
| E1 | 30 | 5 / 30 | 0 / 30 | 0 / 30 | 0 / 30 |
| E1s | 31 | 7 / 31 | 0 / 31 | 0 / 31 | 0 / 31 |
| E2 | 13 | 5 / 13 | 0 / 13 | 0 / 13 | 0 / 13 |
| E3 | 62 | 60 / 62 | 62 / 62 | 0 / 62 | 12 / 62 |
| E1z | 31 | 9 / 31 | 0 / 31 | 0 / 31 | 0 / 31 |

- train_range 의 scene 별 probe: E1 ep0 1 / 10 · ep500 2 / 10 · ep1200 2 / 10, E2 0 / 4 · 1 / 4 · 4 / 5.
- 정의에서 정해지는 칸: E1s 는 state 가 그대로라 rule · proxy 예측이 정의상 E0 와 같다 (그래서 0 / 13). proxy 는 host record 의 원래 이력을 쓰므로 E3 에서 `place` 로 못 바뀐다 (0 / 99). 이 칸들은 probe 의 반응과 같은 의미의 측정이 아니다.

### 5. 실제 scene

![T48b held host 의 편집 image](figures/t48/t48b-scene-held-edits.png)

[`t48b-scene-held-edits.png`](figures/t48/t48b-scene-held-edits.png) — held-out 두 frame (ep1807 r23, ep1850 r14) 의 E0 (기록 image 와 bit-identical) · E1 · E1s · E1z · E2 를 3 camera 로 보이고 각 열 제목에 probe · rule · proxy · frame-index 예측과 probe 확률을 적었다. 이 두 frame 은 선택 기준이 sidecar 에 있고 결과로 고르지 않았다. 그림에서: ep1807 r23 은 E1s · E1z · E2 가 `pick` 으로, ep1850 r14 는 E1z 만 `pick` 으로 읽힌다 (같은 host frame 이라도 편집마다 예측이 갈린다). wrist camera 에서 E1 · E2 는 과일이 finger 사이에 없고 E1z 는 finger 가 붙어 있다.

![T48b E3 phantom hold](figures/t48/t48b-scene-e3-phantom.png)

[`t48b-scene-e3-phantom.png`](figures/t48/t48b-scene-e3-phantom.png) — E3 (ep1807 r10 · ep1850 r5 · ep1900 r10 (template 가 없는 pear 의 `pad_midpoint` 배치)) 의 E0 대 E3 를 3 camera 와 예측으로. E3 는 접근 자세의 팔이 과일을 finger 사이에 쥐고 있는 image 다.

### 6. 읽을 때의 한계

| 한계 | 내용 |
|---|---|
| **E1 · E2 는 pilot** | held-out 에서 E1 13 · E2 7 frame, 2 scene. 관통 제외로 줄었다 (E2 16 → 7). 한 scene 안의 frame 은 독립 trial 이 아니다 |
| **E3 의 template** | 손 안의 과일 자세는 같은 과일 · 같은 손의 실제 `held` record 에서 잰 상대 자세를 옮긴 것이다. ep1807 · long · headfix · ep1800 의 template 은 `ep1807:31`. pear (ep1900) 는 template 이 없어 finger pad 중점 규칙을 쓴다 |
| **E3 에서 과일과 finger 가 함께 바뀐다** | 과일의 유무와 finger 닫힘을 분리해 재지 않았다 (재측정 대기 8) |
| **proxy ≠ strong rule** | 실제 strong rule (`grasp_latch.py` + 실행 feedback) 은 측정하지 않았다 |
| **secondary 의 장면 익숙함** | train_range 3 run 은 VLA fine-tune episode 이고 ep0 · ep1200 은 probe 학습 episode 이기도 하다. 편집 image 는 학습에 없으나 장면은 익숙할 수 있다 |
| **E0 정답 기준** | `closed_loop_gt` 와 contact state 가 다른 frame 이 있다 (held-out 39 frame). 편집 (E1 – E3) 의 정답은 contact state 계열의 물리 predicate 만 쓴다 |

### 7. 되돌아올 지점 · 판정 대기 제안

| task 의 신호 (`T48b.task.md`) | 측정 | 다음 |
|---|---|---|
| E0 parity 미달 → 이후 진행 없음 | 통과 (599 / 599 일치, bit-identical) | 해당 없음 |
| E1 · E2 에서 probe 가 `place` 를 유지하면 probe 는 자세 순서를 읽는다. 그러면 HSR 은 empty grasp · slip 에서 strong rule (measured opening) 보다 못하다 → 검토 문서 §6 대로 HSR 을 **보조 구성 요소** 로 둔다 | held-out 에서 E1 `place` 유지 12 / 13, E2 4 / 7 (n = 13 · 7, 2 scene) | 조건에 해당하는 값이다. 단 **strong rule 은 측정하지 않았다** (proxy 는 strong rule 이 아니다). 해석과 HSR 의 위치는 **판정 대기** (아래 제안 (a)) |
| E1 · E2 · E3 에서 probe 가 물체를 따라가면 image 의 쥠 상태를 읽는 것이다 → 실험 4 (phase source ablation) 의 HSR arm 근거 | E3 에서 `pick` → `place` 로 바뀐 frame 97 / 99 (rule 도 99 / 99), E1 은 대부분 `place` 유지 (12 / 13), E2 는 4 / 7 | 한 방향의 신호가 아니다. 판정 대기 |

**판정 대기 제안 (a) — HSR 을 보조 구성 요소로 둔다 (미측정 해석 포함).** 검토 문서 §6 의 채택 기준 ("HSR 이 strong rule 보다 실행상 도움이 없으면 HSR 은 보조 구성 요소로 남긴다") 을 쓰면, T48a (instruction 교체 반응 0.040), T48b (E1 `place` 유지 12 / 13), T50 (counterfactual probe 도 기준 미달, 아래 T50 절) 은 같은 쪽을 가리킨다. 그러나 이 기준의 전제인 **strong rule 과의 실행 비교와 gate on / off 이득은 어떤 STEP 에서도 측정하지 않았다 (미측정 해석).** 사용자 판정 전이므로 HSR 의 위치를 이 로그가 정하지 않는다.

**측정하지 않은 것 (`T48b.verify.json` `not_measured`).** cuRobo 회귀 기준선 (GPU, 이 STEP 에서 돌리지 않음). 그 밖에 이 절에서 밝힌 것: 실제 strong rule, E3 의 과일 · finger 분리, held-out E1 · E2 의 표본 확대.

**산출물.**

- **측정**: `handoff/T48b.verify.json`. 원자료는 run 디렉터리 `/mnt/dev/AG3S_VLA-storage/results/development/20261012T170147+0900_28b981600bf7_d97d2cd2e13a/verify/T48b/` (`plan` · `render` · `feat` · `extract` · `score` · `indep`).
- **figure** (각각 `.json` sidecar 있음): [`scene (held edits)`](figures/t48/t48b-scene-held-edits.png) · [`scene (E3)`](figures/t48/t48b-scene-e3-phantom.png) · [`graph`](figures/t48/t48b-score-accuracy-bars.png) · [`table (parity · 제외)`](figures/t48/t48b-cpu-parity-exclusions-table.png) · [`table (score)`](figures/t48/t48b-score-table.png).
- **구현**: `AG3S/ag3s/experiments/studies/t48b_{scene,metrics,hsr}.py` · `tests/ag3s/test_t48b_{scene,hsr_cli}.py` (43 passed). `handoff/T48b.impl.md`.
- commit 은 하지 않았다.

---

## T49 — 명사 token query: prompt 의 target 명사 · "basket" token 이 image 를 보는 attention 은 target · 목적지를 가리키는가 (2026-10-12 KST)

**이 절이 답하는 물음 (목적).** T47 은 논문 (Kang et al.) 의 label-free 기준을 **action (suffix) token query** 에 적용했고 실패했다 (선택된 head 가 fixed-patch sink 와 넓게 퍼진 map). 논문의 원래 query 는 **text** 이고 여러 물체를 다룰 때 **명사 token** 마다 map 을 따로 만든다. pi0.5 의 prefix 는 완전 bidirectional 이라 (image · text 모두 `ar_mask = False`) text 행이 image 열을 본다. T49 는 얼린 pi0.5 의 prefix pass **한 번** 에서 (denoise · noise seed 불필요, 결정적) target 과일 명사 token 과 `basket` token 의 행이 image 768 열에 둔 attention 을 꺼내 camera 별 16×16 map 으로 만들고, (1) target 을 더 잘 가리키는가, (2) target 과 목적지를 서로 다른 단어로 따로 얻을 수 있는가 (separation), (3) prompt 의 명사를 바꾸면 map 이 따라 움직이는가 (prompt swap), (4) 3D 파이프라인에 넣으면 L8H2 보다 나은가를 잰다. query 의 성격이 다르다: action query 는 policy 가 "어디로 움직이려 하는가" 를 보고 (잡은 뒤 basket 으로 옮겨갈 수 있다), 명사 query 는 "이 단어가 무엇을 가리키는가" 를 본다.

**한 줄 결과 (before → after).**

- **before.** 서빙 query (L8H2, action token) 는 held-out 에서 `peak_on_target` 0.693 이고 잡은 뒤 (post-grasp) 에는 0.382 로 떨어진다. 목적지는 규칙 (잡을 수 없음 + 3 frame 우위) 으로만 판정했다.
- **after — 2D · 분리 · prompt 교체는 명사 query 가 앞선다** (held-out 4 record pilot, record-mean).
  - 명사 GT cell (primary, `noun_gt_k3_s1` = L8H6 · L5H4 · L4H2, σ=1) `peak_on_target` **1.000** (pre-grasp 1.000 · post-grasp 1.000), target mass 0.236. 같은 cell 을 σ=0 으로 합산하면 target mass 0.528 · lift 93.9. L8H2 는 0.693 (post-grasp 0.382), target mass 0.147.
  - **분리**: target map 의 peak 가 target 위 · basket map 의 peak 가 crate 위인 frame 비율 `both_correct` **0.981** (target map 의 crate 위 peak 0.000, basket map 의 target 위 peak 0.000). label 없이 고른 secondary 는 0.914.
  - **prompt 교체**: 새로 지명한 과일 위로 map peak 가 옮겨간 (`moved`) frame 이 held-out **157 / 164** (pre-grasp 98 / 98, post-grasp 59 / 66).
- **after — 그러나 3D 는 명사 primary 가 L8H2 보다 낮다** (held-out 60 frame, record-mean): `identity_correct` **0.800 대 0.900**, chosen cluster purity **0.700 대 0.933**, non-target seed 손실 **420.3 대 190.3** point / frame. `on_target` 은 0.900 대 0.917.
- **비용**: 명사 prefix pass 의 device 시간 median 26.5 ms (1025 frame). 같은 frame 에서 전처리 포함 49.2 ms 대 L8H2 의 prefix + suffix 10 회 107.5 ms.
- 서빙 cell 은 L8H2 를 유지한다 (T47 판정). 서빙 반영은 이 STEP 에서 하지 않았다.

**읽는 법과 출처 규약.**

- 수치는 `T49.verify.json` 의 `numbers` 에서만 왔다. 같은 파일 숫자의 덧셈 · 개수 세기는 "(산출)". 모든 집계는 verify.json `_scope` 의 **record-mean** (record 안에서 frame pool → record 가중 없는 평균) 이라 T47 의 pooled 값과 같은 source 도 조금 다르다 (L8H2 의 target mass · lift · peak: T47 pooled 0.1498 · 59.60 · 0.689 대 T49 record-mean 0.1465 · 55.19 · 0.693).
- 해석의 출처를 가른다: "(사용자 판정)" · "(lead)" · "(미측정 해석)".
- **held-out 은 pilot** 이다 (same-scene `headfix` · `ep1800`, new-scene `ep1850` · `ep1900`; record 당 15 frame, 같은 scene 의 두 record 는 독립 표본이 아니다). `noun_gt` · `basket_gt` 는 selection record (`long` · `t2` · `ep1807`) 의 GT 로 고른 cell 이라 **held-out 에서만 채점**하고 e2e 는 selection record 에서 거절한다.
- **환경.** GPU 는 offline prefix pass (checkpoint `…/29999`, 16D LoRA) 한 번에만 썼다 (GPU lock 아래, 시작 전 nvidia-smi memory 0 MiB). 나머지는 CPU (`.venv-ag3s`, legacy ESDF). 회귀 기준선: legacy 14 / 15 · has_target 15 / 15 · frame0 `clearance_before` −29.031048 mm 일치 (분석 시 load 1 분 평균 12.34 → 10.74). cuRobo 기준선은 돌리지 않았다. 분석 HEAD `ec2e79fe`, extract HEAD `28b98160` (T49 파일은 untracked).

### 1. 사용자 판정과 타임라인 (KST)

| # | 판정 · 사건 (2026-10-12) | 이 절에서 |
|---|---|---|
| 1 | **(사용자 판정, T47 결과 후)** 명사 token query (D) 를 시험한다. 서빙 cell L8H2 는 유지 (T47 판정) | 전체 |
| 2 | **(사용자 판정)** cell 선택 = **GT 기반 primary + label-free secondary** (결과 보기 전 고정) | §2 |
| 3 | **(사용자 판정)** **prompt 교체 시험 포함** (exploratory) | §8 |
| 4 | **(사용자 판정)** VLA checkpoint 의 offline GPU prefix pass 승인. GPU 를 동시에 쓸 수 있으면 동시 실행 — 단 **(lead 판정) T46 의 CL3 와는 겹치지 않는다** (CL3 의 TO 가 wall-clock 마감에서 멈추므로 GPU 를 나눠 쓰면 refined action 이 바뀌어 paired 비교가 무너진다). 그래서 GPU 추출은 CL3 종료 후 | §9 |
| 5 | **(lead, T49.impl.md 확인 후 · 결과 전)** 명사 행 image mass 의 record 통합 median < 0.01 인 layer 는 "image 를 거의 보지 않음" 으로 **표시만** 하고 primary GT top-3 선택에서 제외하지 않는다 | §3 |
| 6 | **(사용자 판정, T49 결과 후)** T49b exploratory 진행 (§T49b), T51b 로 명사 행을 공식 protocol 에 추가 | T49b 절 |

| 시각 (KST) | 무엇 | 출처 |
|---|---|---|
| 19:18:17 – 19:19:15 | GPU smoke (`--records ep1850 --no-swaps --cost-frames 4`, rc 0, wall 57.8 s) | `gpu_extract[0]` 10:18:17Z – 10:19:15Z |
| 19:19:18 – 19:23:34 | GPU full extract (1040 frame = 7 record × 4 variant, rc 0, wall 255.9 s, retry 0, nvidia-smi peak 8806 MiB) | `gpu_extract[1]` 10:19:18Z – 10:23:34Z |
| (이후) | CPU select 11 s · validate 8 s · swap 13 s · e2e (4 source) 173 s | `n8_cost.cpu_stage_wall_s` |

### 2. 사전 고정과 구현 (새 파일만, `T49.impl.md`)

| 항목 | **primary** | secondary / exploratory |
|---|---|---|
| query 행 | prompt 의 target 과일 명사 1 token (target map) · `basket` 1 token (목적지 map). `last_text` (마지막 text token) 행은 기록만 하고 판정에 쓰지 않음 | — |
| cell 선택 | **GT 기반**: selection record 3 개에서 target map 의 lift 평균으로 18×8 = 144 cell 의 순위를 매기고 top-3 를 σ=1 raw 합 | **secondary**: 같은 명사 행에 T47 의 label-free 절차 (S_cam τ Kneedle + spatial entropy + selection frequency, L0 · L1 제외, k = 3) |
| 목적지 map cell | target 과 같은 방식으로 따로 (crate coverage 기준 lift) | target 의 GT cell 을 그대로 쓴 변형 (`basket_samecells`) |
| 평가 | held-out 4 record (pilot) | k = 1..5, σ = 0 |
| 비교 기준 | L8H2 (action query, 서빙) · T47 oracle top-3 (action, L8H3 · L10H7 · L12H4) | head-average (명사 행, L2–17) |

- **추출** (GPU): `pi05_attention._prefix` 를 복제한 jitted 함수가 probs 를 jit 안에서 `[18 layer, 8 head, 3 query row, 768]` 로 잘라 host 로 보낸다 (전체 `[18,1,1,8,968,968]` 는 device 밖으로 안 나간다). 명사 위치는 `obs.tokenized_prompt` 안에서 piece id 매칭으로 찾는다 (index 하드코딩 없음, 0 개 · 2 개 이상이면 오류). **1040 frame 에서 오류 0** (`extract_frames_not_ok` 0).
- **e2e**: T47b 의 `run_b2_record` (live 서빙 경로 `wire.pack_request → process_multi_debug` 를 조립한 것 + label 기반 3D 지표) 를 쓴다. attention source 만 바뀐다. 3D 는 legacy ESDF, grasp latch 없음 (T47 과 같은 한계).
- 테스트: `.venv-ag3s` 30 passed · 3 skipped (jax · sentencepiece 없는 venv 의 3 test).
- **L8H2 의 e2e row 일치 점검**: T49 의 L8H2 row 는 T47 row 와 field 불일치 0, T47b row 와는 2 (모두 ep1850 frame 13 · 14 의 `chosen_cluster_purity` NaN 대 NaN 비교, 다른 field 는 일치).

### 3. 명사 행의 mass 배분 (#1) — 이 query 는 image 를 보는가

![명사 · basket 행의 layer 별 mass 배분과 분리](figures/t49/t49-separation-mass.png)

[`t49-separation-mass.png`](figures/t49/t49-separation-mass.png) — 왼쪽 두 panel = target · basket 행의 layer 별 median mass 배분 (image · text 제외 BOS · BOS, 로그 축, 0.01 점선), 오른쪽 = §6 의 분리 (scene × phase). 아래는 같은 수치 (7 record, 260 frame, (frame, head) median; `image + text` 의 최대 편차 0.0031 · 0.0028 · 0.0029).

| layer | target 행 image | target 행 BOS | basket 행 image | basket 행 BOS | last_text 행 image |
|---:|---:|---:|---:|---:|---:|
| L0 | **0.0002** | 0.9805 | **0.0013** | 0.8691 | **0.0004** |
| L1 | 0.174 | 0.1357 | 0.3833 | 0.0498 | 0.1144 |
| L2 | 0.1715 | 0.3428 | 0.3064 | 0.1572 | 0.0941 |
| L3 | 0.3356 | 0.0952 | 0.432 | 0.0614 | 0.0647 |
| L4 | 0.2159 | 0.1597 | 0.3559 | 0.0935 | 0.0601 |
| L5 | 0.1498 | 0.2051 | 0.1306 | 0.0774 | 0.0929 |
| L6 | 0.418 | 0.0469 | 0.4583 | 0.0747 | 0.1083 |
| L7 | 0.1308 | 0.1548 | 0.1822 | 0.1445 | 0.0703 |
| L8 | 0.4611 | 0.1982 | 0.5873 | 0.0684 | 0.127 |
| L9 | 0.0979 | 0.1997 | 0.1128 | 0.1367 | 0.0258 |
| L10 | 0.2294 | 0.2388 | 0.2923 | 0.126 | 0.1475 |
| L11 | 0.0609 | 0.5742 | 0.0847 | 0.0806 | 0.1042 |
| L12 | 0.207 | 0.4873 | 0.1567 | 0.1328 | 0.0595 |
| L13 | 0.0486 | 0.4814 | 0.094 | 0.1008 | 0.0427 |
| L14 | 0.0343 | 0.8086 | 0.0302 | 0.4355 | 0.2175 |
| L15 | 0.0501 | 0.7031 | 0.0498 | 0.6621 | 0.0786 |
| L16 | 0.1904 | 0.1948 | 0.148 | 0.4424 | 0.1702 |
| L17 | 0.4767 | 0.0806 | 0.4999 | 0.146 | **0.0012** |

- **image mass < 0.01 로 표시된 layer**: target · basket 행 = **L0 만**, `last_text` 행 = L0 · L17. 나머지 layer 는 image 에 0.03 – 0.59 를 둔다. 모든 layer 에서 `image + text ≈ 1` 이라 slicing 이 맞다 (사전에 둔 확인 기준 ≈ 1e-2 이하).
- 선택된 cell 은 layer median 보다 훨씬 image 를 본다: 명사 GT cell L8H6 0.9716 · L5H4 0.9948 · L4H2 0.9618, label-free L4H7 0.8436 · L4H6 0.9204, basket GT L8H6 0.9975 · L4H2 0.9531 · L8H0 0.7658 (cell 별 median image mass).
- (미측정 해석) 명사 행이 layer 에 따라 BOS (L0 · L14 · L15 에서 큼) 나 text 에 질량을 두는 것은 head 마다 역할이 다르다는 뜻일 수 있다. 이 STEP 은 그 이유를 분리해 재지 않았다.

### 4. cell 선택 (#2)

selection record 3 개 (long · t2 · ep1807) 만 쓴다. `n` = 선택에 쓰인 (frame, camera) sample 수.

| query 행 | 방식 | top-3 (layer 번호는 L, head 번호는 H) | 비고 |
|---|---|---|---|
| target 명사 | **GT primary** (n = 219) | **L8H6 · L5H4 · L4H2** | lift 86.86 · 86.33 · 84.36, `peak_on_target` 1.000 · 0.995 · 0.995 |
| | label-free secondary (τ = camera 별 Kneedle) | L4H7 · L4H6 · L8H6 | selection frequency 0.490 · 0.485 · 0.403 |
| | (변형) label-free, τ = 전체 `S_img` | L6H7 · L8H6 · L4H6 | frequency 0.667 · 0.613 · 0.458 |
| basket | **GT primary** (n = 550, crate 기준) | **L8H6 · L4H2 · L8H0** | lift 10.02 · 8.17 · 8.165, peak on crate 0.980 · 0.913 · 0.913 |
| | label-free secondary | L4H7 · L8H2 · L8H0 | frequency 0.600 · 0.512 · 0.497 |
| | (변형) label-free, τ = `S_img` | L8H6 · L4H2 · L3H4 | |
| (참고) `last_text` 행, target | GT | L8H7 · L12H2 · L10H2 | lift 31.26 · 30.04 · 29.32. 판정에 쓰지 않음 |
| (참고) `last_text` 행, destination | GT | L3H6 · L0H4 · L0H6 | lift 2.12 · 1.93 · 1.87. 판정에 쓰지 않음 |

- record 별 GT top-3 (target): long = L8H6 · L8H0 · L4H2, t2 = L4H2 · L5H4 · L8H0, ep1807 = L8H6 · L5H4 · L4H2. record 하나를 빼고 다시 고른 top-3 가 전체 top-3 와 겹친 cell 수 (3 이면 같은 집합): GT target **3 · 3 · 3**, label-free target 2 · 2 · 2, GT basket 2 · 2 · 3, label-free basket 3 · 2 · 2. (T47 의 action query 공통 top-3 는 같은 지표가 0 / 3 이었다.)
- 서빙 L8H2 (L8, head 2) 는 명사 target GT top-3 에 들지 않고 basket label-free top-3 에 든다. T47 의 action oracle top-3 = L8H3 · L10H7 · L12H4, 명사 GT cell 은 L8H6 · L5H4 · L4H2 로 겹치지 않는다.
- 명사 label-free 절차의 top-3 는 GT primary 와 L8H6 하나만 겹친다 (target).

### 5. 2D 검증 (#3 · #4) — held-out, record-mean

![layer × head lift heatmap](figures/t49/t49-lift-heatmaps.png)

[`t49-lift-heatmaps.png`](figures/t49/t49-lift-heatmaps.png) — 18 × 8 cell 의 lift (target · basket, selection record 와 held-out). 상자 = GT top-3, 점 = label-free top-3, x = L8H2. 정의와 수치는 sidecar.

**target map** (target body 기준, 모든 phase; same-scene · new-scene 은 record 2 개씩의 평균)

| source | target mass | lift | `peak_on_target` | IoU@0.25 | IoU@0.5 | same-scene mass · lift | new-scene mass · lift |
|---|---:|---:|---:|---:|---:|---|---|
| **명사 GT k3 σ1 (primary)** | 0.2359 | 28.42 | **1.000** | 0.181 | 0.165 | 0.267 · 31.15 | 0.205 · 25.68 |
| 명사 GT k3 σ0 | **0.5281** | **93.94** | 1.000 | 0.652 | 0.481 | 0.573 · 98.32 | 0.483 · 89.56 |
| 명사 label-free k3 σ1 (secondary) | 0.1649 | 17.39 | 0.9375 | 0.124 | 0.114 | 0.174 · 18.66 | 0.156 · 16.12 |
| 명사 head-average (L2–17) | 0.2346 | 35.60 | 0.9375 | 0.099 | 0.092 | 0.262 · 42.59 | 0.208 · 28.61 |
| **L8H2 (action, 서빙)** | 0.1465 | 55.19 | 0.693 | 0.169 | 0.092 | 0.167 · 56.25 | 0.126 · 54.14 |
| T47 oracle top-3 σ1 (action) | 0.0644 | 8.78 | 0.746 | 0.058 | 0.052 | 0.076 · 10.52 | 0.053 · 7.03 |
| T47 oracle top-3 σ0 (action) | 0.1057 | 22.07 | 0.880 | 0.061 | 0.050 | 0.129 · 28.01 | 0.083 · 16.13 |

- **phase 별** (target mass · lift · `peak_on_target`): 명사 GT σ1 pre-grasp 0.175 · 33.68 · 1.000, post-grasp 0.370 · 18.75 · **1.000**. L8H2 pre-grasp 0.182 · 74.47 · 0.839, post-grasp 0.076 · 15.06 · **0.382**. 명사 GT σ0 pre 0.481 · 116.8, post 0.650 · 49.20. T47 oracle σ1 pre `peak` 0.770, post 0.697.
- **lift 는 L8H2 가 명사 σ1 primary 보다 크다 (55.2 대 28.4)** 이고 target mass 와 `peak_on_target` 은 명사가 앞선다. 명사 σ0 는 셋 모두 앞선다. 지표마다 앞서는 쪽이 다르다 — 어느 쪽으로 읽을지는 §7 의 3D 와 함께 본다.
- **class mass 분해** (all phases, same-scene / new-scene):

| source | target | destination | other_object | robot_gripper | robot | support | background |
|---|---|---|---|---|---|---|---|
| 명사 GT σ1 | 0.267 / 0.205 | 0.030 / 0.065 | 0.019 / 0.018 | 0.170 / 0.105 | 0.011 / 0.050 | 0.458 / 0.456 | 0.046 / 0.102 |
| 명사 GT σ0 | 0.573 / 0.483 | 0.007 / 0.029 | 0.015 / 0.012 | 0.161 / 0.065 | 0.002 / 0.033 | 0.231 / 0.330 | 0.011 / 0.048 |
| L8H2 | 0.167 / 0.126 | 0.004 / 0.035 | 0.008 / 0.006 | 0.079 / 0.035 | 0.003 / 0.038 | 0.723 / 0.711 | 0.016 / 0.050 |

**목적지 map** (crate 기준; 아래 "crate mass" = crate 칸의 attention 질량 몫, verify.json 의 `target_mass` 필드가 이 표에서는 crate 를 뜻한다)

| source | crate mass | lift | peak on crate | IoU@0.25 | IoU@0.5 |
|---|---:|---:|---:|---:|---:|
| **basket GT k3 σ1 (primary)** | **0.6052** | 7.20 | **0.958** | 0.499 | 0.451 |
| basket GT k3 σ0 | 0.7173 | 9.39 | 0.956 | 0.455 | 0.418 |
| basket 같은 cell (target 의 GT cell) σ1 | 0.6151 | 7.31 | 0.975 | 0.509 | 0.457 |
| basket label-free σ1 | 0.2886 | 3.37 | 0.719 | 0.272 | 0.235 |
| basket head-average | 0.2802 | 4.13 | 0.367 | 0.231 | 0.205 |
| L8H2 (action) | 0.1751 | 2.41 | 0.222 | 0.128 | 0.120 |
| T47 oracle σ1 (action) | 0.1150 | 1.23 | 0.072 | 0.109 | 0.099 |
| (참고) target 명사 map 을 crate 로 채점 | 0.0982 | 1.46 | 0.028 | 0.125 | 0.097 |

- basket map 을 **target** 으로 채점하면 (섞임) target mass 0.0237 · lift 2.31 · peak 0.048 이다 (pre-grasp peak 0.000).
- 목적지 class mass (basket GT σ1): destination 0.653 (same) / 0.557 (new), target 0.014 / 0.012.
- L8H2 의 목적지 질량은 같은 지표에서 crate mass 0.175, peak 0.222 로, post-grasp 에서 높다 (crate mass pre 0.117 → post 0.275).

### 6. 분리 (#5) — target 과 목적지를 한 번의 prefix pass 에서 따로 얻는가

(§3 그림의 오른쪽 panel.) peak 는 3 camera joint argmax, 적중은 patch coverage > 0.05. 4 record 의 record-mean.

| 쌍 | target peak ∈ target | basket peak ∈ crate | **both_correct** | target peak ∈ crate | basket peak ∈ target | 같은 patch | target map 의 target mass · crate mass | basket map 의 crate mass · target mass |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **primary** (`noun_gt` / `basket_gt` k3 σ1) | 1.000 | 0.983 | **0.981** | 0.000 | 0.000 | 0.000 | 0.275 · 0.035 | 0.553 · 0.007 |
| secondary (label-free) | 1.000 | 0.917 | 0.914 | 0.000 | 0.000 | 0.000 | 0.161 · 0.099 | 0.251 · 0.014 |
| basket 에 target 의 cell 사용 | 1.000 | 0.917 | 0.904 | 0.000 | 0.000 | 0.000 | 0.275 · 0.035 | 0.511 · 0.004 |
| head-average | 1.000 | 0.767 | 0.769 | 0.000 | 0.077 | 0.067 | 0.191 · 0.095 | 0.252 · 0.032 |

- primary 의 `both_correct`: scene 별 same-scene 0.962 (pre-grasp 0.929 · post-grasp 1.000), new-scene 1.000. phase 별 pre-grasp 0.964 · post-grasp 1.000.
- record · phase 별 개수 (primary): both_correct / both_visible = ep1800 pre 6 / 7 · post 6 / 6, headfix 9 / 9 · 6 / 6, ep1850 10 / 10 · 5 / 5, ep1900 9 / 9 · 6 / 6 → 합 57 / 58 (산출; 60 frame 중 ep1800 의 pre-grasp 2 frame 은 target 이 가려져 eligible 이 아님). 이 표의 mass 열은 분리 집계의 frame 단위 값이라 §5 표의 camera-sample 집계 값과 다르다 (미측정 해석 — 집계 단위 차이로 읽었으나 verify.json 이 이 차이를 직접 서술하지 않는다).
- L8H2 대응 값은 없다 — action query 에는 basket map 이 없다 (`not_measured`).

![scene overlay](figures/t49/t49-scene-overlay.png)

[`t49-scene-overlay.png`](figures/t49/t49-scene-overlay.png) — held-out 4 frame (ep1800 frame 6 · 11, ep1850 frame 6 · 12; pre-grasp 와 post-grasp 각 2) 의 3 camera 위에 target 명사 map · basket map · L8H2 map (파랑, 3 camera 최대로 정규화) 과 GT target 윤곽 (초록) · crate 윤곽 (주황) 을 겹쳤다. 빨간 x = joint 3-camera argmax. 열 제목에 per-camera target · crate mass 몫. 선택 규칙은 sidecar (σ=1, k=3, cell 은 §4). 이 그림에서 명사 target map 은 `cam_left_wrist` 의 과일 윤곽 안에, basket map 은 crate 윤곽 안에 놓이고, 같은 frame 의 L8H2 map 은 post-grasp 의 ep1850 frame 12 `cam_left_wrist` 에서 과일 밖의 pixel 들에 흩어진다.

### 7. 3D end-to-end (#6) — held-out 60 frame, record-mean

![T49 3D 비교 표](figures/t49/t49-e2e-table.png)

[`t49-e2e-table.png`](figures/t49/t49-e2e-table.png) — source × 지표 표 (같은 수치의 scene · phase 분해 포함).

| source (σ=1) | `on_target` | `identity_correct` | silent failure | chosen cluster purity | non-target seed 손실 (point / frame) | target 잔여 비율 |
|---|---:|---:|---:|---:|---:|---:|
| **명사 GT k3 (primary)** | 0.900 | **0.800** | 0.100 | **0.700** | **420.3** | 0.0031 |
| 명사 label-free k3 (secondary) | 0.883 | 0.783 | 0.117 | 0.684 | 579.3 | 0.0051 |
| **L8H2 (서빙, σ 없음)** | 0.917 | **0.900** | 0.083 | **0.933** | **190.3** | 0.0926 |
| T47 oracle top-3 (action) | 0.917 | 0.817 | 0.083 | 0.717 | 114.7 | 0.0828 |

- **scene · phase 로 나눈 `identity_correct` · purity · non-target 손실** — 명사 GT: same-scene 0.733 · 0.629 · 342.5, new-scene 0.867 · 0.772 · 498.1, pre-grasp 0.861 · 0.661 · 374.0, post-grasp 0.692 · 0.760 · 496.3. L8H2: same 0.867 · 0.867 · 119.9, new 0.933 · 1.000 · 260.6, pre 0.889 · 0.889 · 148.3, post 0.900 · 1.000 · 273.5.
- non-target seed 손실의 class 별 합 (4 record 평균의 합; 명사 GT 대 L8H2): destination 4052.5 대 1944.3, other_object 1118.5 대 499.5, table 1133.5 대 410.3. L8H2 는 target 잔여 비율이 크다 (0.0926 대 0.0031).
- **pooled (60 frame, T47 style)**: `on_target` 명사 GT 54 · 명사 label-free 53 · L8H2 55 · oracle 55, silent failure 6 · 7 · 5 · 5. seed precision / recall: 명사 GT 0.173 / 0.995, L8H2 0.366 / 0.864, oracle 0.520 / 0.917. chosen target 점 비율 0.565 · 0.785 · 0.614 (명사 GT · L8H2 · oracle), centroid 오차 median 0.0346 · 0.0169 · 0.0345 m. 한 frame 의 e2e 시간 median 456 · 441 · 440 ms (명사 GT · L8H2 · oracle).
- record 단위 pilot (4 record × 15 frame) 이므로 `identity_correct` 0.800 과 0.900 의 차이를 크게 읽지 않는다. 그러나 purity (0.700 대 0.933) 와 non-target 손실 (420 대 190) 은 같은 방향으로 갈린다.
- **task 의 되돌아올 지점**: "2D 는 좋은데 3D 가 L8H2 와 같으면 head 처럼 3D 집계가 흡수하는 것이다. 이득은 목적지 분리 · 결정성 · 비용 쪽에서만 주장한다." 측정된 3D 는 같지 않고 **낮다** (identity · purity · non-target 손실). 그래서 3D 에서의 이득은 주장하지 않고, 이득의 후보는 분리 · 결정성 · 비용 쪽에 둔다. 원인은 T49b 에서 σ · k 로 확인했으나 닫히지 않았다.

### 8. prompt 교체 (#7, exploratory) — map 은 instruction 을 따라 움직이는가

같은 frame (image · state 그대로) 에서 target 명사만 scene 의 다른 과일로 바꾼 prompt 로 다시 뽑았다 (record 당 3 개, 7 record × 3 = 21 variant). held-out 4 record 의 12 variant (180 frame 중 두 과일이 모두 ≥ 200 px 보이는 164 frame 이 eligible, 산출 180 = 4 × 3 × 15). 선택 record 는 cell 선택에 쓰였으므로 판단 근거로 쓰지 않는다.

| source (held-out, record-mean) | 새 과일 위 peak (교체 prompt) | 새 과일 위 peak (원 prompt) | 원 target 위 peak (교체 prompt) | **moved** | pooled moved | 새 과일 lift 교체 / 원 | lift_new_gain | map_corr |
|---|---:|---:|---:|---:|---|---|---:|---:|
| **명사 GT k3 σ1** | 0.959 | 0.017 | 0.104 | **0.958** | **157 / 164** | 29.47 / 5.86 | 23.61 | 0.210 |
| 명사 head-average | 0.939 | 0.022 | 0.125 | 0.939 | 153 / 164 | 51.13 / 4.09 | 47.03 | 0.369 |
| 명사 label-free σ1 | 0.757 | 0.011 | 0.305 | 0.755 | 121 / 164 | 18.75 / 7.27 | 11.48 | 0.696 |

- 명사 GT 의 `moved`: same-scene 0.927 · new-scene 0.989. pre-grasp 98 / 98, post-grasp 59 / 66 (same-scene post-grasp 0.828, new-scene post-grasp 0.967).
- 같은 교체에서 **basket map 은 흔들리지 않는다**: basket GT map 의 원 prompt 대 교체 prompt 상관 0.989 (held-out), crate lift 7.20 → 7.23.
- 참고 (selection record 포함, cell 이 그 record 로 골라진 것이라 판단 근거 아님): 명사 GT `moved` 0.896 (455 / 501).
- **되돌아올 지점 점검**: "prompt 교체에서 map 이 옮겨가지 않으면 명사 map 은 instruction 이 아니라 장면의 현저한 물체를 따른다 → wrong-object 진단에는 쓸 수 없다." 측정은 옮겨갔다 (157 / 164). 다만 이것은 (wrong-object 정의 · 사용자 판정을 거친) 별도의 wrong-object 진단 STEP 이 아니다.

### 9. 비용 (#8)

| 항목 | 값 |
|---|---|
| 명사 prefix pass (device, embed_prefix + 18 layer + jit 안 slice) | median **26.52 ms** · p90 26.93 ms (1025 frame) |
| 같은 frame (`long` 앞 8 frame) 전처리 포함 | 명사 49.21 ms 대 L8H2 의 prefix + suffix 10 회 **107.5 ms** (median) |
| model load · compile | 27.8 s · 1.61 s |
| memory | JAX peak 7.60 GB, nvidia-smi peak 8806 MiB |
| 추출 wall | smoke 57.8 s, full 255.9 s |

(L8H2 와의 memory 비교는 별도 process 로 재지 않았다 — `not_measured`.)

### 10. 읽을 때의 한계

| 한계 | 내용 |
|---|---|
| **pilot** | held-out 4 record × 15 frame, 같은 scene 의 두 record 는 독립 아님. 모든 비율의 단위가 크다 |
| **GT 로 고른 cell** | primary cell 은 selection record 3 개의 GT lift 로 골랐다 (label 이 필요). label-free 로 고른 secondary 는 record-mean 전체 기준 2D · 3D 지표 모두 primary 보다 나쁘다 (2D target mass 0.165 대 0.236, 3D non-target 손실 579.3 대 420.3) |
| **primary 는 σ=1** | 사전 고정이 σ=1 이다. σ=0 합산은 exploratory (target mass 0.528 · lift 93.9) 이고 3D 는 T49b 에서 잰다 |
| **3D 는 legacy ESDF · grasp latch 없음** | cuRobo ESDF backend e2e 와 latch 포함 e2e 는 측정하지 않았다 (재측정 대기 10) |
| **L8H2 와 명사의 집계가 다른 값을 줄 수 있다** | T47 pooled 와 T49 record-mean (§ 읽는 법) |

### 11. 되돌아올 지점 · 판정 대기 제안

| task 의 신호 (`T49.task.md`) | 측정 | 다음 |
|---|---|---|
| #1 에서 명사 행의 image mass 가 거의 없으면 명사 query 는 쓸 수 없다 | image mass < 0.01 은 L0 하나 (target · basket 행), 선택된 cell 의 image mass 는 0.77 – 0.99 | 해당 없음 |
| 2D 는 좋은데 3D 가 L8H2 와 같으면 이득은 분리 · 결정성 · 비용 쪽에서만 | 3D 는 L8H2 보다 낮다 (identity 0.800 대 0.900 · purity 0.700 대 0.933 · non-target 손실 420 대 190) | 3D 이득은 주장하지 않는다. T49b (σ · k) 후 T51b 로 |
| prompt 교체에서 map 이 옮겨가지 않으면 wrong-object 진단에 쓸 수 없다 | 옮겨갔다 (157 / 164) | wrong-object 진단 후보 (검토 문서 §3 후속 후보) 는 별도 판정 |
| 서빙 반영 (`attention_policy.py` source 교체) 은 별도 STEP | 하지 않았다 | 결과 보고 → 사용자 판정 → 별도 STEP (T46 session 에 먼저 알린다) |

**판정 대기 제안 (b) — 명사 query 로 identity · destination, L8H2 로 3D seed 를 얻는 결합 (미측정 해석 포함).** 측정된 비대칭은 이렇다. 명사 query 는 prompt 가 가리키는 물체 (2D `peak_on_target` 1.000, post-grasp 에서도), 목적지 (`both_correct` 0.981), instruction 교체 (157 / 164) 를 따라가고, L8H2 는 3D 에서 `identity_correct` · purity · non-target 손실이 낫다. 이 두 장점을 합치면 (예: 명사 map 으로 어느 물체인지, L8H2 로 3D seed 를) 좋아질 수 있다고 **볼 수 있으나, 그 결합을 어떤 방식으로도 돌리지 않았다 (미측정 해석).** 어떤 경로로 결합하는지도 정해지지 않았다. 공식 비교는 T51b (명사 행을 T51 의 protocol 에 추가) 의 몫이고, 사용자 판정 전이므로 이 로그는 결합을 채택하지 않는다.

**측정하지 않은 것 (`T49.verify.json` `not_measured`).** cuRobo 회귀 기준선 (GPU 미사용 지시) · 명사 prefix 의 GPU memory 를 L8H2 serving 과 별도 process 로 잰 값 · selection record 의 e2e 3D · separation 의 L8H2 대응 (action query 에 basket map 이 없음).

**산출물.**

- **측정**: `handoff/T49.verify.json`. 원자료는 run 디렉터리 `/mnt/dev/AG3S_VLA-storage/results/development/20261012T190547+0900_28b981600bf7_c39a577332fe/verify/T49/` (`extract` · `select` · `validate` · `swap` · `e2e`). T47 의 GT cache 는 읽기 전용 입력.
- **figure** (각각 `.json` sidecar 있음): [`scene`](figures/t49/t49-scene-overlay.png) · [`lift heatmap`](figures/t49/t49-lift-heatmaps.png) · [`mass · 분리`](figures/t49/t49-separation-mass.png) · [`3D 표`](figures/t49/t49-e2e-table.png).
- **구현**: `AG3S/ag3s/experiments/studies/t49_noun_query{,_core,_extract}.py` · `tests/ag3s/test_t49_noun_query.py`. `handoff/T49.impl.md`.
- commit 은 하지 않았다.

---

## T49b — 명사 query 의 3D 약점 확인: smoothing (σ) 과 cell 수 (k) 변형 (exploratory, 2026-10-12 KST)

**이 절이 답하는 물음 (목적).** T49 에서 명사 query 는 2D · 분리 · prompt 교체에서 L8H2 보다 앞섰지만 3D 의 `identity_correct` · purity · non-target 손실은 L8H2 보다 나빴다. 의심한 원인은 사전 고정한 **σ=1 smoothing 과 top-3 raw 합산** 이다 (T47b 에서 σ=1 이 L8H2 의 3D identity 를 떨어뜨린 전례: held-out 60 frame 의 `identity_correct` 54 → 49, 위 "T47 · T47b" 절 §8). T49b 는 이미 추출된 명사 attention 위에서 **σ ∈ {0, 1} × k ∈ {1, 2, 3}** 과 label-free top-3 σ=0 을 3D e2e 로 다시 잰다. k=1 은 GT 순위 1 위 L8H6 단독이다. **이 STEP 은 exploratory 이고 채택 근거가 아니다** (사용자 판정, T49 결과 후): 명사 query 의 cell · k · σ 선택은 T51b (공식 protocol) 로 정한다.

**한 줄 결과 (before → after).**

- **before.** T49 primary (`noun_gt_k3_s1`): `identity_correct` 0.800, purity 0.700, non-target 손실 420.3 (L8H2 0.900 · 0.933 · 190.3).
- **after.** **어떤 σ · k 변형도 L8H2 의 3D 수준에 못 미쳤다.** σ=0 이 identity 를 0.800 → 0.850, purity 를 0.700 → 0.831 로 올렸고 (k 와 무관), k 는 identity · purity 를 바꾸지 않았다 (σ 가 같으면 0.850 / 0.800 으로 같음). 대신 **non-target 손실은 σ=0 에서 오히려 커졌다** (512.8 → 583.4, k 가 커질수록 증가; σ=1 은 394.5 → 420.3). L8H2 는 0.900 · 0.933 · 190.3. label-free top-3 σ=0 은 0.800 · 0.700 · 675.6.
- 2D 는 σ=0 에서 target mass 0.47 – 0.59, lift 89 – 98.5 로 σ=1 (0.20 – 0.27, 25 – 31) 보다 높다. 3D 와 2D 가 같은 방향으로 움직이지 않았다.

**읽는 법과 출처 규약.** 수치는 `T49b.verify.json` 의 `numbers` 에서만 왔다. 모두 **exploratory** 이며 해석은 "(미측정 해석)". 집계는 T49 와 같은 record-mean (held-out 4 record × 15 frame, 추출 오류 frame 0). 같은 입력 (T49 의 select · extract) 을 읽기 전용으로 쓰고 `verify/T49b/variants/<name>/e2e/` 에만 썼다 (T49 e2e 출력의 sha1 은 전후 동일). 3D 는 CPU, legacy ESDF, grasp latch 없음. 회귀 기준선 일치 (legacy 14 / 15 · 15 / 15 · −29.031048 mm), cuRobo 기준선은 돌리지 않았다. 검증: 6 개 e2e run 의 L8H2 row 와 `gt_k3_s1` 의 명사 row 가 T49 e2e row 와 field 단위 일치 (불일치 0, 비교한 field 4260 개 / 60 frame, `_ms` 제외), 2D 재계산이 T49 validate 와 최대 절대차 0.0. HEAD `ed47b976` (e2e 는 `de2042e9` 에서 시작해 중간 docs-only commit 만 바뀜, .py 변경 없음).

### 1. 3D 결과 (held-out, record-mean, 모든 phase)

![T49b σ × k 3D 그래프](figures/t49/t49b-sigma-k-3d.png)

[`t49b-sigma-k-3d.png`](figures/t49/t49b-sigma-k-3d.png) — σ × k 별 `identity_correct` · purity 와 L8H2 기준선. 전 변형 표는 [`t49b-variants-table.png`](figures/t49/t49b-variants-table.png).

| source | `on_target` | `identity_correct` | silent | chosen cluster purity | non-target seed 손실 (point / frame) | `on_target` ∧ ¬`identity` (record 평균 frame 수) |
|---|---:|---:|---:|---:|---:|---:|
| 명사 GT k1 σ0 (L8H6 단독) | 0.917 | 0.850 | 0.083 | 0.831 | 512.8 | 1.00 |
| 명사 GT k2 σ0 | 0.917 | 0.850 | 0.083 | 0.831 | 564.9 | 1.00 |
| 명사 GT k3 σ0 | 0.917 | 0.850 | 0.083 | 0.831 | 583.4 | 1.00 |
| 명사 GT k1 σ1 | 0.900 | 0.800 | 0.100 | 0.700 | 394.5 | 1.50 |
| 명사 GT k2 σ1 | 0.900 | 0.800 | 0.100 | 0.700 | 411.0 | 1.50 |
| 명사 GT k3 σ1 (T49 primary) | 0.900 | 0.800 | 0.100 | 0.700 | 420.3 | 1.50 |
| 명사 label-free k3 σ0 | 0.900 | 0.800 | 0.100 | 0.700 | 675.6 | 1.50 |
| **L8H2 (σ 없음)** | 0.917 | **0.900** | 0.083 | **0.933** | **190.3** | 0.25 |

- 합산 개수 (산출, 4 record × 15 frame = 60): `identity_correct` σ0 변형 51 / 60, σ1 변형 48 / 60, L8H2 54 / 60. `on_target` 55 · 54 · 55. record 별로 보면 명사 변형은 모두 `headfix` 에서 `identity_correct` 12 / 15 (L8H2 15 / 15) 이고, σ1 변형과 label-free 는 `ep1900` 에서 13 / 15 (σ0 변형과 L8H2 15 / 15). 차이가 나는 record 가 둘뿐이라 4 record pilot 이다.
- **scene · phase 별** (identity · purity · non-target 손실):

| source | same-scene | new-scene | pre-grasp | post-grasp |
|---|---|---|---|---|
| 명사 GT k1 σ0 | 0.767 · 0.663 · 398.7 | 0.933 · 1.000 · 626.9 | 0.889 · 0.798 · 442.8 | 0.775 · 0.881 · 629.9 |
| 명사 GT k2 σ0 | 0.767 · 0.662 · 430.6 | 0.933 · 1.000 · 699.2 | 0.889 · 0.798 · 551.9 | 0.775 · 0.881 · 585.9 |
| 명사 GT k3 σ0 | 0.767 · 0.663 · 460.5 | 0.933 · 1.000 · 706.2 | 0.889 · 0.798 · 579.1 | 0.775 · 0.881 · 591.0 |
| 명사 GT k1 σ1 | 0.733 · 0.629 · 333.8 | 0.867 · 0.772 · 455.2 | 0.861 · 0.661 · 335.1 | 0.692 · 0.760 · 493.3 |
| 명사 GT k2 σ1 | 0.733 · 0.629 · 324.5 | 0.867 · 0.772 · 497.6 | 0.861 · 0.661 · 358.1 | 0.692 · 0.760 · 496.8 |
| 명사 GT k3 σ1 | 0.733 · 0.629 · 342.5 | 0.867 · 0.772 · 498.1 | 0.861 · 0.661 · 374.0 | 0.692 · 0.760 · 496.3 |
| 명사 label-free k3 σ0 | 0.733 · 0.629 · 582.2 | 0.867 · 0.772 · 768.9 | 0.861 · 0.661 · 659.7 | 0.692 · 0.760 · 697.6 |
| L8H2 | 0.867 · 0.867 · 119.9 | 0.933 · 1.000 · 260.6 | 0.889 · 0.889 · 148.3 | 0.900 · 1.000 · 273.5 |

### 2. 2D (같은 변형, held-out, T49 validate 에서 읽음)

| source | same-scene target mass · lift · peak | new-scene target mass · lift · peak |
|---|---|---|
| 명사 GT k1 σ0 | 0.587 · 98.5 · 1.000 | 0.494 · 98.5 · 1.000 |
| 명사 GT k2 σ0 | 0.571 · 98.3 · 1.000 | 0.470 · 89.1 · 1.000 |
| 명사 GT k3 σ0 | 0.573 · 98.3 · 1.000 | 0.483 · 89.6 · 1.000 |
| 명사 GT k1 σ1 | 0.269 · 31.0 · 1.000 | 0.207 · 27.8 · 1.000 |
| 명사 GT k2 σ1 | 0.265 · 31.2 · 1.000 | 0.198 · 25.4 · 1.000 |
| 명사 GT k3 σ1 | 0.267 · 31.1 · 1.000 | 0.204 · 25.7 · 1.000 |
| 명사 label-free k3 σ0 | 0.353 · 51.2 · 1.000 | 0.333 · 43.1 · 0.854 |
| L8H2 | 0.167 · 56.3 · 0.731 | 0.126 · 54.1 · 0.655 |

(scene × phase 별 2D 는 `numbers.2d_vs_target_scene_phase_recomputed` 와 위 그림 sidecar. 표본 수는 same-scene 39 · new-scene 51 (frame, camera) sample.)

![T49b 변형의 실제 scene](figures/t49/t49b-scene-variants.png)

[`t49b-scene-variants.png`](figures/t49/t49b-scene-variants.png) — ep1900 frame 13 (post-grasp) 에서 변형 별 map 과 그 frame 의 3D 결과 (`on_target` · `identity_correct` · purity · non-target 손실). 정의는 sidecar.

### 3. 이 STEP 이 닫은 것과 닫지 못한 것

- 사전 질문 "명사 query 의 3D 열세는 smoothing · 합산 설정 때문인가? σ=0 이나 k=1 에서 L8H2 와의 차이가 줄어드는가?" 에 대한 측정: σ=0 에서 identity · purity 는 올랐으나 (0.800 → 0.850, 0.700 → 0.831) L8H2 (0.900 · 0.933) 에 못 미쳤고, non-target 손실은 σ=0 에서 커졌다 (512.8 – 583.4 대 σ=1 의 394.5 – 420.3, L8H2 190.3). k=1 도 같다. **σ · k 만으로는 3D 열세가 닫히지 않았다.**
- 원인은 확정하지 못했다 (재측정 대기 9). (미측정 해석) 명사 map 이 target 과 목적지 · 주변을 넓게 덮는 것 (T49 의 class mass: support 0.46, destination 0.03 – 0.06) 과 3D lifting 이 그 map 을 seed 로 쓰는 방식의 상호작용일 수 있으나, 이 STEP 은 분리해 재지 않았다.
- `noun_lf` 는 σ=0 · k=3 한 변형만 돌렸다 (k 1 · 2 와 σ=1 은 task 범위 밖, `not_measured`). selection record 의 3D 는 CLI 가 `noun_gt` 를 selection record 에서 거절해 측정하지 않았다. swap · separation 은 다시 재지 않았다.
- **채택 근거가 아니다.** 4 record 의 pilot 이고 변형은 held-out 에서 비교한 것이므로, 최종 설정은 T51b 의 calibration → held-out 분리 아래서 다시 정해야 한다.

**산출물.** `handoff/T49b.verify.json`, 원자료 `/mnt/dev/AG3S_VLA-storage/results/development/20261012T190547+0900_28b981600bf7_c39a577332fe/verify/T49b/` (`variants/<name>/e2e` · `analysis`). figure: [`graph`](figures/t49/t49b-sigma-k-3d.png) · [`scene`](figures/t49/t49b-scene-variants.png) · [`table`](figures/t49/t49b-variants-table.png). 코드 변경 없음 (T49 의 `--k` · `--sigma` 를 wrapper 로 호출). commit 은 하지 않았다.

---

## T50 — HSR 의 instruction 둔감은 fine-tuning 탓인가: probe 학습 · representation · fine-tuning 의 분리 (2026-10-12 KST)

**이 절이 답하는 물음 (목적).** T48a 에서 phase probe (`kv_L4_v1`) 는 image 를 따랐으나 instruction 의 target 교체에는 거의 반응하지 않았다 (C1 에서 원래 place 인 frame 이 `pick` 으로 바뀐 비율 0.040, home 0.000). 사용자 질문: **이것이 우리 fine-tuning (16D LoRA) 때문인가?** 원인 후보는 셋이다 — **(a) probe 학습**: probe 는 prompt 와 장면이 항상 맞는 frame 으로만 학습돼 언어를 쓸 이유가 없다 (fine-tuning 탓 아님). **(b) representation**: L4 V feature 에 "지시된 물체의 진행 상태" 가 선형으로 들어 있지 않다. **(c) fine-tuning data**: 2000 episode 가 모두 성공한 scripted 시연 (target 1 + distractor 3) 이라 언어는 처음에 과일을 고를 때만 필요하다. `pi05_base` (우리 LoRA 학습의 초기 weight) 가 로컬에 있어 **같은 feature 정의 · 같은 입력 변형을 base 와 LoRA 에 똑같이 적용해 비교**할 수 있다. ① feature 가 prompt 교체에 얼마나 움직이나 (`r` ratio), ② counterfactual 변형 (C1, 정답 `pick`) 을 학습에 넣은 probe 가 선형으로 "instruction 기준 phase" 를 읽을 수 있나를 잰다.

**한 줄 결과 (before → after).**

- **before.** probe 가 instruction 에 둔감한 것은 확인됐으나 원인 후보 셋은 가려지지 않았다.
- **after — 사전 고정 decision rule 의 출력.** LoRA 와 base 모두 counterfactual probe 가 기준 (**C0 ≥ 0.95 이고 C1_instr ≥ 0.80**, episode 별 정확도의 median) 에 미달했다. LoRA: C0 0.781 · C1_instr 0.672 (valcv), 0.827 · 0.745 (trainval). base: 0.673 · 0.698 (valcv), 0.769 · 0.656 (trainval). 규칙이 정한 문구는 "**둘 다 넘지 못하면 fine-tuning 탓이 아니다**" 이다. 이 문구가 이 STEP 의 규칙 출력이고 사용자 판정이 아니다.
- **① 민감도 r (full, episode median)**: LoRA **0.611**, base **0.999** (text 절반 0.585 / 0.791, image 절반 1.177 / 1.407). r < 1 은 target 교체가 표현만 바꾼 것보다 feature 를 덜 움직인다는 뜻이다. LoRA 가 base 보다 낮다 (이 값의 의미는 (미측정 해석), 규칙은 ② 로 판정).
- **통제**: C1 의 label 을 섞으면 C1_instr 가 0.03 으로 올라가지 않는다 (학습이 label 을 쓴다). counterfactual probe 는 C1_instr 를 올렸으나 (standard 0.02 → 0.66 – 0.71, LoRA) C0 를 0.98 에서 0.77 – 0.83 으로 낮췄다.
- base 위 **표준 probe** (원본 data 로만 학습) 의 C1 반응은 LoRA 와 같은 수준이다: place → `pick` 0.047 (base) 대 0.040 (LoRA), home → `pick` 0.002 대 0.000.

**읽는 법과 출처 규약.**

- 수치는 `T50.verify.json` 의 `numbers` (T50a = 최상위, T50b = `numbers.T50b`) 에서만 왔다. 같은 파일 숫자의 나눗셈 · 덧셈은 "(산출)". 새 probe 는 sklearn 이 없어 numpy / scipy 로 다시 쓴 LogReg 이며 (SUBTASK-b 의 `kv_L4_v1` 을 재현함은 `T50.impl.md` §1, 그 수치는 verify.json 의 `numbers` 에 없어 옮기지 않았다), verifier 가 probe metric 을 독립 재계산해 **base 108 / 108 · LoRA 108 / 108 field 가 1e-9 안에서 일치**, `r` frame 값의 최대 절대차 ≤ 5e-6.
- **집계.** 판정에 쓰는 값은 episode (50 개) 별 정확도의 **median** (lead 가 결과 전 고정). frame 합산 (pooled) 은 함께 보고만 한다. frame 은 독립 trial 이 아니다.
- **환경.** T50a = CPU (T48 의 LoRA feature cache 읽기 전용). T50b = GPU offline prefix pass 2 회 (base: val 6 family + train C0 · C1, LoRA: train C0 · C1; GPU lock 아래, 각각 시작 전 memory 0 MiB) + CPU 분석. 회귀 기준선: T50a 와 T50b 시점 모두 legacy 14 / 15 · 15 / 15 · −29.031048 mm 일치 (T50a 시점 다른 session 의 CL3 server 가 돌고 있어 load 약 24 – 27). cuRobo 기준선은 돌리지 않았다.

### 1. 사용자 판정과 타임라인 (KST)

| # | 판정 · 사건 (2026-10-12) | 이 절에서 |
|---|---|---|
| 1 | **(사용자 판정)** T50a (CPU, 바로) + T50b (GPU, T46 CL3 종료 후) 진행. offline GPU prefix pass 승인 (`pi05_base` 와 16D LoRA checkpoint) | 전체 |
| 2 | **(lead, T50.impl.md 확인 후 · 결과 전 고정)** 판정 임계 (0.95 · 0.80) 는 episode 별 정확도의 median 에 적용한다 | §3 |

| 시각 (KST) | 무엇 | 출처 |
|---|---|---|
| 17:20 – 17:35 | T50a (CPU): sensitivity · probe (valcv) · figure. 앞서 회귀 기준선 | `time_kst` · `runs` |
| 19:37:44 – 20:24:53 | T50b GPU: base extract (`--sets val train`, 65013 input, rc 0, wall 2829.7 s) | `gpu_extract[0]` 10:37:44Z – 11:24:53Z |
| 20:26:15 – 21:00:21 | T50b GPU: LoRA train extract (`--sets train`, 41604 input, rc 0, wall 2045.3 s) | `gpu_extract[1]` 11:26:15Z – 12:00:21Z |
| 21:02 – 21:30 | T50b CPU 분석 (base ① · ② valcv · trainval, LoRA ② trainval, compare) · figure | `time_kst` |

### 2. ① feature 민감도 — prompt 교체는 feature 를 얼마나 움직이는가

val 50 episode · 2601 frame. `r = d(C0, C1) / d(C0, C1p)` (C1 = distractor 3 개 prompt 평균, C1p = 같은 target 의 다른 phrasing 2 개 평균), d = host C0 feature 에 대한 상대 L2 (절반 안에서). 기준선 `d_adj` = 같은 episode 의 다음 sampled frame (stride 6 = 0.4 s) 과의 C0 거리. episode median (괄호 = q25 – q75).

![T50a 민감도 · probe 그래프](figures/t50/t50a-sensitivity-probe-graph.png)

[`t50a-sensitivity-probe-graph.png`](figures/t50/t50a-sensitivity-probe-graph.png) — LoRA 의 r (frame 분포 · episode 분포) 과 probe 정확도. base 와 LoRA 의 나란한 비교는 [`t50b-base-vs-lora-graph.png`](figures/t50/t50b-base-vs-lora-graph.png).

![T50b base 대 LoRA](figures/t50/t50b-base-vs-lora-graph.png)

| 모델 · 절반 | r (all) | d_c1 | d_c1p | d_adj |
|---|---|---:|---:|---:|
| **LoRA full** | **0.611** (0.523 – 0.662) | 0.0409 | 0.0662 | 0.1309 |
| LoRA text 절반 | 0.585 (0.496 – 0.627) | 0.0662 | 0.1125 | 0.1282 |
| LoRA image 절반 | 1.177 (1.055 – 1.314) | 0.0158 | 0.0141 | 0.1310 |
| **base full** | **0.999** (0.870 – 1.155) | 0.1071 | 0.1125 | 0.1000 |
| base text 절반 | 0.791 (0.676 – 0.896) | 0.1441 | 0.1939 | 0.1452 |
| base image 절반 | 1.407 (1.276 – 1.602) | 0.0915 | 0.0672 | 0.0779 |

- 원래 class 별 r (full, episode median): LoRA pick 0.622 · place 0.582 · home 0.633, base pick 0.980 · place 1.006 · home 0.998. frame 합산 median: LoRA full 0.603 · text 0.574 · image 1.208, base full 1.000 · text 0.788 · image 1.401.
- (산출) target 교체의 feature 이동 `d_c1` 은 인접 frame 이동 `d_adj` 의 LoRA full 0.0409 / 0.1309 = 0.31, base full 0.1071 / 0.1000 = 1.07 배다. LoRA image 절반은 0.0158 / 0.1310 = 0.12 배다.
- (미측정 해석) LoRA fine-tuning 이 prompt 교체에 대한 feature 반응을 줄였다고 읽을 수 있으나, 이 STEP 의 판정 규칙은 ② 에 건 것이다. r 의 감소가 읽기 가능성 (②) 에는 영향이 없었다는 것이 아래 §3 이다.

### 3. ② counterfactual probe 와 decision rule

![T50a 판정 규칙 표](figures/t50/t50a-decision-rule-table.png)

[`t50a-decision-rule-table.png`](figures/t50/t50a-decision-rule-table.png) (LoRA valcv) · [`t50b-decision-rule-table.png`](figures/t50/t50b-decision-rule-table.png) (base 대 LoRA, valcv · trainval). `valcv` = val 50 episode 의 episode 단위 5-fold CV (T50a primary), `trainval` = train 200 episode 의 C0 + C1 로 학습해 val 50 episode 평가 (T50b). 모든 probe 는 같은 512D feature, multinomial LogReg (C 는 inner GroupKFold 로 선택, 모든 fold 수렴).

![T50b 판정 규칙 표 (base 대 LoRA)](figures/t50/t50b-decision-rule-table.png)

**사전 고정 규칙 (`T50.task.md`).**

1. LoRA 에서 ② 가 C0 ≥ 0.95 이고 C1 instruction-기준 ≥ 0.80 이면 "representation 에 있음 → 원인은 (a) probe 학습".
2. 아니면 (b) 또는 (c). **(b) 와 (c) 의 구분은 base 결과로 한다**: base 에서 기준을 넘고 LoRA 에서 넘지 못하면 (c) fine-tuning 탓. **둘 다 넘지 못하면 fine-tuning 탓이 아니다.**

**episode median 값 (판정에 쓰는 값)**: counterfactual probe (primary, `full__counterfactual`) 의 C0 / C1_instr.

| 모델 | ② (i) valcv C0 · C1_instr | ② (ii) trainval C0 · C1_instr | 기준 (≥ 0.95 ∧ ≥ 0.80) |
|---|---|---|---|
| **LoRA** | 0.781 · 0.672 | 0.827 · 0.745 | 미달 (둘 다) |
| **base** | 0.673 · 0.698 | 0.769 · 0.656 | 미달 (둘 다) |

frame 합산 (보고만): LoRA valcv 0.766 · 0.662, trainval 0.825 · 0.708. base valcv 0.703 · 0.667, trainval 0.756 · 0.669.

**통제와 참고 cell (episode median, C0 · C1_instr).** 정의: `text` · `image` = 그 절반만, `shuffled` = C1 행 label 무작위, `standard` = C0 행만으로 학습.

| cell | LoRA valcv | base valcv | LoRA trainval | base trainval |
|---|---|---|---|---|
| counterfactual, full (primary) | 0.781 · 0.672 | 0.673 · 0.698 | 0.827 · 0.745 | 0.769 · 0.656 |
| counterfactual, text 절반 | 0.750 · 0.625 | 0.692 · 0.661 | 0.827 · 0.682 | 0.731 · 0.641 |
| counterfactual, image 절반 | 0.606 · 0.625 | 0.683 · 0.635 | 0.702 · 0.474 | 0.692 · 0.589 |
| shuffled label, full | 0.981 · 0.031 | 0.962 · 0.031 | 0.981 · 0.031 | 0.962 · 0.031 |
| standard, full (`kv_L4_v1` 과 같은 방식) | 0.981 · 0.031 | 0.962 · 0.031 | 0.981 · 0.031 | 0.981 · 0.031 |

- standard probe 의 C0 는 base · LoRA 모두 기준을 넘는다 (episode median 0.962 – 0.981; frame 합산 base 0.953 / 0.972, LoRA 0.978 / 0.981). 즉 **base feature 도 phase 를 읽는다** — task 의 "base 가 16D 입력 변환과 맞지 않아 feature 가 무의미" 되돌아올 지점에는 해당하지 않는다.
- 같은 probe 를 counterfactual 로 학습하면 C1_instr 는 올라가지만 (LoRA valcv frame 합산: standard 0.021 → counterfactual 0.662, `pick` 예측 비율 place 0.637 · home 0.687) C0 가 같이 떨어진다 (0.978 → 0.766). label 을 섞은 통제는 C1_instr 0.027 로, label 이 학습에 쓰였음을 확인한다.
- (미측정 해석) 이 512D · 이 층 · 이 선형 probe 위에서 C0 와 C1 을 동시에 만족하는 해가 학습되지 않았다. VLA 의 다른 layer 나 다른 objective (예: contrastive) 에서는 어떤지 이 STEP 은 시험하지 않았다 (재측정 대기 11).

**규칙의 출력.**

| 질문 | 측정 | 규칙의 출력 |
|---|---|---|
| LoRA 에서 ② 가 C0 ≥ 0.95 ∧ C1_instr ≥ 0.80 인가 | 아니다 (valcv · trainval 모두) | (a) "representation 에 있음 → probe 학습" 은 **확인되지 않았다** |
| base 에서 기준을 넘고 LoRA 에서 넘지 못하는가 | 아니다 (base 도 둘 다 미달) | (c) **해당 없음.** **"둘 다 넘지 못하면 fine-tuning 탓이 아니다"** |

**따라서 사전 고정 규칙에 따른 이 STEP 의 결론은 "instruction 둔감은 fine-tuning 탓이 아니다" 이다.** 규칙이 (b) 와 (c) 를 base 의 통과 여부로 가르므로, 규칙으로 남는 후보는 (b) representation (이 feature 에서 선형으로 읽히지 않음) 이다 — 이것은 규칙에서 나온 소거이고 **(b) 를 직접 확인한 측정은 아니다 (미측정 해석).** 사용자 판정 전이므로 다음 STEP (probe 를 counterfactual-aware 로 다시 학습할지, 다른 layer 를 볼지) 은 이 로그가 정하지 않는다.

### 4. base 위 표준 probe 의 T48a C1 반응 (#6)

| `pick` 으로 예측된 비율 (val C1 행, 원래 class 별, pooled) | place (n = 2400) | home (n = 2403) |
|---|---:|---:|
| base 위 표준 probe (train 200 episode C0 로 학습, `trainval` standard) | 0.0471 (113 / 2400) | 0.0017 (4 / 2403) |
| LoRA 위 표준 probe (같은 조건) | 0.0400 (96 / 2400) | 0.0000 (0 / 2403) |
| (참고) T48a 의 `kv_L4_v1` (LoRA, 원 probe) | 0.0396 (95 / 2400) | 0.0000 (0 / 2403) |

### 5. 실제 scene — C1 에서 feature 반응이 큰 frame · 작은 frame

![T50a C1 반응이 큰 / 작은 frame](figures/t50/t50a-scene-c1-response.png)

[`t50a-scene-c1-response.png`](figures/t50/t50a-scene-c1-response.png) — LoRA feature 에서 full `r` 이 가장 큰 frame 과 가장 작은 frame (전체 · 원래 place / home 중, episode 마다 하나), 그 frame 의 image 와 C0 · C1 · C1p 의 prompt, probe 예측 · 확률. 선택 규칙은 sidecar (`selection_rule`: full__r 로 정렬, 가장 큰 / 작은 것). **T50b (base) 의 scene figure 는 만들지 않았다** (`not_measured`).

### 6. 비용 (T50b #4)

| 항목 | base | LoRA (train) |
|---|---|---|
| 추출 input · call (batch 4) | 65013 · 16254 | 41604 · 10401 |
| unit · variant (val / train) | val 50 · 23409, train 200 · 41604 | train 200 · 41604 (LoRA val 은 T48 cache 재사용) |
| prefix pass (device) input 당 median · p90 | **13.72 ms** · 13.77 ms | 14.99 ms · 15.04 ms |
| prefix 합 · load · wall | 892.5 s · 58.1 s · 2824 s | 623.7 s · 9.85 s · 2039 s |
| JAX peak · nvidia-smi peak | 7.38 GB · 8820 MiB | 7.47 GB · 8804 MiB |
| batch parity (n = 4) rel L2 max | 0.0065 | 0.0025 |
| state bin replica 불일치 row (val / train) | 27 / 4 | — / 4 |

- 비교의 전제: val plan digest 가 base 와 LoRA 에서 **50 / 50 같다** (같은 입력 변형을 비교한다).

### 7. 읽을 때의 한계 · 측정하지 않은 것 · 판정 대기

| 한계 | 내용 |
|---|---|
| **feature 정의는 하나** | `kv_L4_v1` 의 정의 (layer 4 · V · head 0, text · image 평균 512D) 와 LogReg 만 시험했다. 다른 layer · head · pooling · objective 는 측정하지 않았다 |
| **probe 학습 data** | `valcv` 의 학습은 val 40 episode 라 `kv_L4_v1` (train 200 episode) 과 같은 probe 가 아니다. `trainval` 은 train 200 episode 를 쓴다. 두 방식 모두 같은 결론 (미달) |
| **C1 정답 pick 의 전제** | T48a 와 같다 — distractor 가 episode 내내 건드려지지 않는다는 dataset 성질에 기댄다 |
| **fine-tuning 의 다른 요인** | base 대 LoRA 의 차이는 SigLIP fine-tune 과 gemma_2b LoRA adapter 의 합이다 (`T50.impl.md` §3 의 서술). 어느 쪽이 r 의 변화를 만들었는지는 가르지 않았다 |
| **본 측정이 아닌 smoke** | impl 의 smoke 수치는 이 로그에 쓰지 않았다 |

**측정하지 않은 것 (`T50.verify.json` `not_measured`).** cuRobo 회귀 기준선 · T50b 의 scene figure (실제 image + prompt, T50a 의 scene figure 가 LoRA 만 다룸).

**판정 대기.** (1) decision rule 의 출력 ("fine-tuning 탓이 아니다") 을 받아들이고 후속을 정할지 — 후속 후보는 task 의 되돌아올 지점 (a) 에 적힌 "probe 를 counterfactual-aware 로 다시 학습" (Bhardwaj et al. 의 contrastive objective 와 비교) 이나, 이번 결과는 (a) 가 아니라 규칙상 소거로 (b) 쪽이다. (2) HSR 의 위치 (T48b 절 제안 (a)). 둘 다 사용자 판정 전이다.

**산출물.**

- **측정**: `handoff/T50.verify.json`. 원자료는 run 디렉터리 `/mnt/dev/AG3S_VLA-storage/results/development/20261012T172032+0900_28b981600bf7_2d9c4266f2e1/verify/T50/` (`lora` · `base` · `compare/compare_full.{json,csv}` · `T50b`).
- **figure** (각각 `.json` sidecar 있음): [`scene`](figures/t50/t50a-scene-c1-response.png) · [`graph (T50a)`](figures/t50/t50a-sensitivity-probe-graph.png) · [`graph (T50b)`](figures/t50/t50b-base-vs-lora-graph.png) · [`table (T50a)`](figures/t50/t50a-decision-rule-table.png) · [`table (T50b)`](figures/t50/t50b-decision-rule-table.png).
- **구현**: `AG3S/ag3s/experiments/studies/t50_{core,extract,hsr_rep}.py` · `tests/ag3s/test_t50_hsr_rep.py` (T50a 시점 19 passed · 1 skipped). `handoff/T50.impl.md`.
- commit 은 하지 않았다.


---

## T46b — F1 을 켠 뒤 늘어난 close 이후 HOLD 는 어디서 왔는가 (진단, 2026-10-12 KST)

**이 절이 답하는 물음 (목적).** T46 의 closed-loop 비교 (CL3) 에서 F1 (non-target 근거에서 target 자신의 표면 점을 빼는 extent 후보) 은 grasp 직전 HOLD run 을 12 → 6 으로 줄였지만, **close 이후** HOLD 는 107 → 183 chunk 로 늘었고 그중 `uncertified` (기하를 인증하지 못해 막은 HOLD) 가 37 → 130 이었다. develope_PLAN §0.10.1 은 "target identity 와 extent 를 분리하고 필요한 target 접촉은 허용하되 인접 장애물 경계를 보존하는 3D 방법 **하나**" 를 집중 개발하라고 하고, §1.2 DC-1 · DC-4 는 그 방법이 잘못된 접촉 허용 · 차단을 줄이면서 task 와 forbidden contact 의 trade-off 를 해치지 않을 것을 요구한다. 늘어난 stop 은 safe success 와 completion time 을 해친다. 그래서 이 STEP 은 **F1 을 그 방법으로 채택할 수 있는가** 에 답하는 데 필요한 진단만 한다 (코드 수정 없음, 새 closed-loop 없음).

**한 줄 결과 (before → after).**

- **before.** close 이후 uncertified HOLD 가 F1 때문에 늘었는지, F1 과 무관한 규칙의 문제인지 가려져 있지 않았다 (이전 로그 T46-1 "원인 미측정").
- **after.** 늘어난 uncertified HOLD 는 **전부 placed phase (과일을 놓은 뒤)** 에 있다 — closing · lift · carry 에서는 세 arm 모두 0 이다 (C1a 37 · C1aF1 130 · C1aF1F2 118 chunk 가 모두 placed). 실패한 인증 항목은 grounding `no_seed` (`n_clusters = 0`) 이고, 그 chunk 에서 **F1 기록은 0 / 130** 이다.
- **원인은 F1 이 아니라 인증 규칙의 공백이다.** placed · home 에서 `no_seed` 는 인증 예외 목록에 없어서, 위반 행이 0 이어도 chunk 가 HOLD 된다. **같은 HOLD 가 F1 이 없는 CL2 에도 있다** (C0 73 · C1bp 70 chunk).
- **F1 의 역할은 간접이다.** F1 은 grasp 직전 field 를 바꿔 18 쌍 중 12 쌍의 궤적을 grasp 직전 window 에서 처음 갈라놓았다. 궤적이 다르면 놓은 뒤 attention 도 달라지고, placed `no_seed` 가 생긴 run 은 F1 만 3 · C1a 만 0 · 둘 다 1 이다 (15 pair, McNemar exact p = 0.25) — **F1 이 계통적으로 늘린다고 말할 수 없다 (모름).**
- **가설 판정.** (a) F1 이 grasp 이후 처리와 충돌 = 반박, (b) F1 이 non-target 근거를 줄여 인증 실패 = 반박, (c) 분모 효과 = 반박, (d) placed 뒤 `no_seed` 인증 공백 = **확인**, (d′) F1 이 `no_seed` 를 계통적으로 늘림 = **모름**.
- **offline what-if (closed-loop 아님).** `no_seed` 를 SUBTASK-f 조건 아래서 인증하면 (W1) 기록 chunk 중 execute 로 바뀌는 것이 37 / 129 / 116 개이고, close 이후 uncertified HOLD 는 0 / 1 / 2 chunk 로 줄어든다. phase-gated F1 (close 이후 F1 off) 은 그 uncertified chunk 130 개 중 0 개에만 F1 기록이 있어 이 HOLD 에 효과가 없다.

**읽는 법과 출처 규약.**

- 수치는 [`T46b.verify.json`](handoff/T46b.verify.json) 의 `numbers` · `hypotheses` · `not_measured` 에서만 왔다. CL3 의 pair 통계 · 성공 지표 · 시간 지표는 [`T46-CL3.verify.json`](handoff/T46-CL3.verify.json), offline 충돌 행 · seed 손실은 [`T46.verify.json`](handoff/T46.verify.json) 에서 왔고 그 자리에 적었다. 같은 파일 숫자의 덧셈 · 뺄셈은 "(산출)" 로 적었다.
- **해석의 출처를 가른다.** 사용자 판정은 "(사용자 판정)", lead 가 task 문서에 쓴 것은 "(lead)", 측정으로 확인하지 않은 해석은 "(미측정 해석)" 이다.
- **arm 과 지표.** C0 · C1a · C1aF1 · C1aF1F2 는 용어 절. `HOLD` · `uncertified` 의 정의는 [이전 로그 용어](AG3S_T0T6_LOG.md). phase 는 recorded `grasp.state` 로 정했다 (`design.phase_definition`). HOLD 는 client 가 planning row 의 control row 를 모두 실행하지 않은 chunk 다.
- **환경.** CPU offline, `.venv-ag3s` (legacy ESDF), CL3 run 의 snapshot `218229ac` 코드를 PYTHONPATH 에 올려 읽기 전용으로 분석했다. 회귀 기준선 (legacy): start_violated 14 / 15 · has_target 15 / 15 · frame0 clearance −29.031048 mm 일치. cuRobo 기준선은 GPU 가 없어 **돌리지 않았다**. snapshot 의 6 파일은 당시 HEAD `ec2e79fe` 와 같다 (`code_state.snapshot_vs_head_identical_files`); 아래 file:line 은 그 코드 기준이다 (지금 working tree 는 다른 session 이 수정 중이라 줄 번호가 달라질 수 있다).

### 1. 사용자 판정과 타임라인 (KST)

KST 는 pod 에 tzdata 가 없어 UTC 에 +9 h 를 더해 환산했다 (run 디렉터리 이름의 `+0900` 은 그대로 읽음; 파일 시각은 mtime, commit 시각은 `git log`).

| # | 판정 · 사건 (2026-10-12) | 이 절에서 |
|---|---|---|
| 1 | **(사용자 판정)** "추천 단계로 진행. **develope_PLAN.md 가 진행 방향이므로 방향성을 잃지 않도록.**" — F1 의 close 이후 HOLD 원인만 진단하고 원인 추적을 넓히지 않는다 | 전체 |
| 2 | **(lead, 측정 전 고정 · [`T46b.task.md`](handoff/T46b.task.md))** 중단 기준: CPU offline, verifier 한 번, 새 closed-loop 없음. 원인이 (a)–(c) 로 설명되지 않으면 추가 추적 없이 "모름" 으로 닫고 사용자 판정 (F1 을 grasp 직전 효과로만 주장, limitation 기록) | §6 |

| 시각 (KST) | 무엇 | 출처 |
|---|---|---|
| 22:05 | T46b task 작성 (lead) | `T46b.task.md` mtime 13:05Z |
| 22:07:12 | verifier run 시작 | run 디렉터리 `20261012T220712+0900_de2042e9be89_ead5f3483344` |
| 22:16 | figure 3 종 생성 | `t46b-*.png` mtime 13:16Z |
| 22:17 | `T46b.verify.json` 작성 | mtime 13:17Z |
| 22:19:42 | T46b 진단 commit `ed47b976` | `git log` |

### 2. 어디서 늘었나 — phase × HOLD kind (§1 of task)

CL3 의 18 쌍 × 3 arm 의 recorded chunk (4050 server chunk) 를 phase 별로 나눴다. 칸의 값은 HOLD chunk / 그 phase 의 chunk 수 (괄호 = kind). 위반 행이 있어서 막은 것은 `collision`, 인증이 없어서 막은 것은 `uncertified`.

| phase | 도달 run (C1a · C1aF1 · C1aF1F2) | C1a | C1aF1 | C1aF1F2 |
|---|---|---|---|---|
| pre-grasp (window 이전) | 18 · 18 · 18 | 0 / 134 | 0 / 133 | 0 / 133 |
| **window (pre8 + close)** | 18 · 18 · 18 | 12 / 162 (collision 12) | **6 / 162** (collision 6) | **2 / 162** (collision 2) |
| closing | 3 · 5 · 6 | 50 / 77 (collision) | 50 / 100 (collision) | 1 / 82 (collision) |
| lift | 18 · 18 · 18 | 0 / 77 | 0 / 113 | 0 / 105 |
| carry | 16 · 16 · 16 | 2 / 211 (collision) | 3 / 201 (collision) | 3 / 222 (collision) |
| **placed (놓은 뒤)** | 16 · 15 · 15 | 55 / 689 (collision 18 · **uncertified 37**) | **130 / 641** (**uncertified 130**) | **118 / 646** (**uncertified 118**) |
| **close 이후 합** | | **107** (uncertified 37) | **183** (uncertified 130) | **122** (uncertified 118) |

(출처: `numbers.phase_table`, `numbers.after_close_hold`. closing 은 도달 run 수가 arm 마다 다르다 (3 · 5 · 6).)

- close 이후 HOLD 중 `uncertified` 가 아닌 것 (= collision) 은 C1a 70 · C1aF1 53 · C1aF1F2 4 chunk 다 (산출: 107 − 37, 183 − 130, 122 − 118). **F1 은 close 이후 collision HOLD 를 오히려 줄였고 (70 → 53), 늘어난 것은 placed 의 uncertified 뿐이다.**
- closing · lift · carry 의 uncertified 는 세 arm 모두 **0** 이다 (`numbers.phase_table` 의 `uncertified_by_grounding` 이 비어 있음).

![T46b 그래프: close 이후 HOLD 의 phase × kind, placed 의 grounding 시계열, CL2 base rate](figures/t46/t46b-graph.png)

[`t46b-graph.png`](figures/t46/t46b-graph.png) (sidecar [`.json`](figures/t46/t46b-graph.json)) — (a) close 이후 HOLD 를 phase × kind 로 쌓은 막대 (진한 색 = uncertified, 연한 색 = collision; 라벨 = HOLD / 그 phase chunk). placed 에서만 uncertified 막대가 솟는다. (b) 같은 (episode, seed) 의 C1a 와 C1aF1 의 placed chunk 별 `grounding_status` 띠 (세로 눈금 = HOLD). F1 쪽 띠가 `no_seed` (빨강) 로 이어지고 HOLD 가 끝까지 찍힌 run 이 보인다. (c) placed `no_seed` HOLD chunk 수를 CL2 의 arm (F1 없음) 과 CL3 의 arm 에 걸쳐 놓은 막대 (§6).

### 3. 직접 원인 — 인증 규칙의 공백 (certification gap)

**placed · home 에서 `no_seed` 가 일어나는 경로 (file:line, snapshot `218229ac` = 당시 HEAD 와 같은 파일).**

| 단계 | 일어나는 일 | 위치 |
|---|---|---|
| 1 | grounding 이 attention 상위 5 % 를 seed 로 뽑고 `~exclude_mask` 를 곱한다. `exclude_mask` 는 pipeline 이 support mask 로 넘긴다. 남은 seed 가 0 이면 곧바로 `NO_SEED` 를 돌려준다 (admissibility · subtask gate 판정 이전) | `AG3S/ag3s/stages/target_grounding.py:1679-1686` (gate 는 `:1124-1136`), `AG3S/ag3s/runtime/pipeline.py:1513-1519` |
| 2 | 놓은 뒤 home 에서 server 는 subtask 인증 `subtask_no_target` 을 쓴다. 이 술어는 grounding 이 `subtask_gated` 이거나, `no_admissible` 이면서 (gate on ∧ label home ∧ `placed_seen`) 일 때만 참이다. **`no_seed` 는 목록에 없다** | `AG3S/ag3s/types.py:1069-1113` (`_placed_home` gate: `:1061-1066`) |
| 3 | 인증 술어 `geometry_certified = status ok ∨ subtask_no_target` 가 거짓이면 reason `uncertified` | `AG3S/trajopt/safe_policy.py:1853-1866`, reason 기록 `:1513-1517` · `:1583` |
| 4 | safe status 이면서 인증이 요구되고 인증이 없으면 TO verdict 를 `violated` 로 강등 → HOLD | `AG3S/trajopt/sqp.py:922` |

- **인증 술어를 chunk 마다 다시 계산한 결과가 기록된 reason 과 일치한다**: 세 arm 모두 **1350 / 1350** chunk (`numbers.certification_replay`). 즉 위 술어 서술이 실제 판정과 같다.
- **seed 를 offline 으로 다시 뽑은 결과도 일치한다**: placed chunk 에서 "남은 seed = 0 이면 `no_seed`" 의 예측이 기록과 C1a 689 / 689 · C1aF1 641 / 641 · C1aF1F2 646 / 646 이다 (`numbers.offline_seed_requery`). `no_seed` chunk 에서 attention 상위 5 % seed 중 support mask 안에 있는 비율은 **전부 1.0** (최솟값 1.0; 37 · 129 · 116 chunk). destination AABB 안의 seed 비율 중앙값은 `no_seed` chunk 에서 0.0, 그 밖의 placed chunk 에서 0.093 · 0.148 · 0.151 이다.
- 그 HOLD chunk 의 증거 필드: `n_clusters` 0 (C1a 37 · C1aF1 129 · C1aF1F2 116), 위반 행 > 0 인 chunk **0**, `has_target` true **0**, TO status 는 전부 `violated` (강등), 그 chunk 의 subtask gate · label · `placed_seen` 는 `home | True | True` (C1aF1 129 chunk; 나머지 1 chunk 는 label `pick` · grounding `no_admissible`, C1aF1F2 는 2 chunk).
- **위반 행이 없는데 막혔다.** 이 chunk 들은 collision 이 아니라 "설명되지 않음" 으로 막힌 것이다. 인증은 collision 이 아니라 "target 이 없는 이유" 를 요구한다.

**같은 seq 에서 C1a 는 통과하고 F1 은 막힌 chunk.** 93 chunk (`numbers.paired_C1a_execute_vs_C1aF1_uncertified_same_seq`): ep11_s111 40 · ep30_s301 36 · ep54_s542 15 · ep104_s1042 1 · ep54_s541 1, phase 는 모두 placed. 그 chunk 에서 C1a 의 grounding 은 `subtask_gated` 19 · `no_admissible` 74 (둘 다 인증 예외 목록에 있다), C1aF1 은 `no_seed` 92 · `no_admissible` 1 이다. 같은 episode · 같은 seq 라도 attention 이 어디에 떨어지느냐에 따라 예외 목록 안 / 밖으로 갈린다.

![T46b 대표 장면: 같은 episode/seed 의 같은 seq 에서 C1a 는 execute, C1aF1 은 no_seed 로 uncertified HOLD](figures/t46/t46b-scene.png)

[`t46b-scene.png`](figures/t46/t46b-scene.png) (sidecar [`.json`](figures/t46/t46b-scene.json)) — ep11_s111, seq 36 (control step t = 280–287, video frame 284), grasp.state 는 `placed`, subtask label 은 `home`. 위쪽은 같은 시각의 MuJoCo 장면, 아래쪽은 table 영역 top view 에 attention 상위 5 % seed 를 올린 것 (파랑 = support 밖 seed, 주황 = support 위라 제외된 seed, 초록 사각형 = destination AABB, 보라 원 = 로봇 구 중심).

| 같은 seq 36 (ep11_s111) | C1a | C1aF1 |
|---|---|---|
| attention 상위 5 % seed | 1803 | 1812 |
| 그중 support 밖 seed | **7** | **0** |
| `n_clusters` | 3 | **0** |
| grounding_status | `subtask_gated` | **`no_seed`** |
| subtask label · `placed_seen` | home · True | home · True |
| 서버 판정 | execute (TO `optimal`) | **HOLD** (`uncertified` + `budget_only`, TO `violated`) |
| 이 chunk 의 F1 기록 | 없음 | **없음** |

(출처: `numbers.representative_scene`. 이 장면은 boundary · contact 장면이 아니다 — F1 의 경계 효과가 아니라 HOLD 의 원인을 보인다. develope_PLAN Figure 3 의 경계 장면은 CL3 의 [`t46-cl3-scene.png`](figures/t46/t46-cl3-scene.png), 이 STEP 의 장면은 §7 참조.)

### 4. F1 과의 연결 — 직접이 아니라 간접이다

**F1 은 그 HOLD chunk 에서 돌지 않았다.**

| phase (C1aF1) | chunk | F1 기록 있음 | F1 적용 | F1 이 뺀 점 (합) |
|---|---|---|---|---|
| pre-grasp | 133 | 133 | 76 | 16 (2 chunk) |
| window | 162 | 162 | 108 | 444 (76 chunk) |
| closing | 100 | 100 | 99 | 1148 (93 chunk) |
| lift | 113 | 71 | 71 | 1391 (63 chunk) |
| carry | 201 | 1 | 1 | 0 |
| **placed** | 641 | 92 | 41 | 4 (4 chunk) |
| **placed 중 uncertified HOLD 130 chunk** | 130 | **0** | **0** | — |

(출처: `numbers.f1_activity_by_phase`, `numbers.uncertified_hold_evidence`. F1 이 close 이후 closing · lift 에서는 동작했지만 그 phase 의 uncertified 는 0 이다. 같은 HOLD chunk 에 F1 기록이 없는 것은 C1aF1F2 도 같다: 0 / 118.)

**F1 이 바꾼 것은 grasp 직전이다.**

| 지표 | C1aF1 | C1aF1F2 |
|---|---|---|
| C1a 와 실행 action 이 끝까지 같은 pair | 6 | 0 |
| 첫 궤적 분기가 있는 pair | **12** | **18** |
| 그 분기가 window 안 · pre-grasp | 12 · 0 | 17 · 1 |
| 분기 chunk 에서 F1 적용 · 뺀 점 > 0 | 12 · 12 | 12 · 12 |

(출처: `numbers.divergence_summary`. 분기 pair 수는 [CL3 verify](handoff/T46-CL3.verify.json) 의 `executed_actions_identity` (C1a 대 C1aF1 같음 6 / 18, C1aF1F2 같음 0 / 18) 와 일치한다.)

- 궤적이 갈라지면 놓은 뒤 자세와 attention 도 달라진다. 그 결과 placed `no_seed` 가 생긴 run 은 **F1 만 3, C1a 만 0, 둘 다 1** (placed 에 둘 다 도달한 15 pair, McNemar exact p = 0.25; C1aF1F2 도 3 · 0 · 1, p = 0.25). run 수로는 C1a 1 → C1aF1 4 · C1aF1F2 4 (산출).
- **이 연결은 궤적 분기를 통한 간접이다.** F1 기록 0 / 130 이므로 F1 이 `no_seed` 를 직접 만든 증거는 없다. 계통적으로 늘린다는 증거도 없다 (p = 0.25, 표본 15 pair) — **모름**.
- `no_seed` 가 시작된 시점은 모두 place 이후다:

| run | arm | 첫 `no_seed` seq · t | place_t | `no_seed` 연속 chunk | episode 끝까지 | task success |
|---|---|---|---|---|---|---|
| ep11_s111 | C1aF1 · C1aF1F2 | 36 · 280 | 184 | 40 | 예 (trailing HOLD 40) | 예 |
| ep30_s301 | C1aF1 · C1aF1F2 | 40 · 312 | 224 | 36 | 예 (36) | 예 |
| ep54_s541 | C1a | 39 · 304 | 224 | 37 | 예 (37) | 예 |
| ep54_s541 | C1aF1 · C1aF1F2 | 38 · 296 | 216 | 38 | 예 (38) | 예 |
| ep54_s542 | C1aF1 | 41 · 320 | 240 | 14 + 1 | 아니오 (풀림) | 예 |
| ep54_s542 | C1aF1F2 | 41 · 320 | 240 | 1 + 1 | 아니오 (풀림) | 예 |

(출처: `numbers.no_seed_onset_vs_place`. **정정 메모:** T46c task 문서 §3 은 "두 run (ep11_s111 · ep30_s301) 이 episode 끝까지 HOLD" 라고 쓰지만 verify.json 은 C1aF1 의 끝까지 HOLD 가 세 run (ep11_s111 · ep30_s301 · ep54_s541) 이다. ep54_s541 은 C1a 도 같은 모양이라, **F1 이 새로 만든** 끝까지 HOLD 는 앞의 두 run 이다 (산출: C1a 에서는 ep11_s111 · ep30_s301 이 `no_seed` 를 겪지 않음 — 위 F1 만 3 중 두 개).)

### 5. 가설 판정 (측정 전 고정한 후보 → 측정 결과)

![T46b 표: close 이후 HOLD 의 출처 · 실패 항목 · offline what-if, 가설 판정](figures/t46/t46b-table.png)

[`t46b-table.png`](figures/t46/t46b-table.png) (sidecar [`.json`](figures/t46/t46b-table.json)) — 위 표의 수치와 가설 판정을 한 장에 모았다.

| 가설 (task 문서가 측정 전 세운 후보) | 판정 | 근거 (`T46b.verify.json`) |
|---|---|---|
| **(a)** close 이후 target 이 held object 가 되는데 F1 이 grasp 이후에도 동작해 held / target 표면 처리와 충돌한다 | **반박** | uncertified HOLD 가 closing · lift · carry 에서 0 · 0 · 0. 그 HOLD chunk 의 F1 기록 0 / 130, `has_target` true 0 / 130. F1 은 closing 99 chunk · lift 71 chunk 에서 동작했지만 uncertified 를 만들지 않았다 |
| **(b)** F1 이 non-target 근거를 줄여 "설명되지 않은 표면" 이 늘어 인증이 실패한다 | **반박** | 실패 항목은 grounding `no_seed` (129 chunk) 와 `no_admissible` + label `pick` (1 chunk). `n_clusters` 0 이 129 chunk, 위반 행 > 0 인 chunk 0. `no_seed` 129 chunk 모두 상위 5 % seed 가 전부 support mask 안 (최솟값 1.0) — field 는 관여하지 않았다 |
| **(c)** 같은 run 이 C1a 에서는 더 일찍 실패해 close 이후에 도달하지 않는다 (분모 효과) | **반박** | placed 도달 run C1a 16 · C1aF1 15 · C1aF1F2 15, placed chunk 689 · 641 · 646. 분모는 C1a 가 오히려 같거나 크다. placed chunk 당 `no_seed` HOLD 는 37 / 689 · 129 / 641 · 116 / 646 |
| **(d)** 그 밖: placed 뒤 `no_seed` 인증 공백 + 고착 HOLD | **확인** | `no_seed` HOLD C1a 37 · C1aF1 129 · C1aF1F2 116. CL2 (F1 없음) 에도 C0 73 · C1bp 70. 인증 술어 재생이 1350 / 1350 일치. W1 가정 시 129 / 129 chunk 가 execute |
| **(d′)** F1 이 placed `no_seed` run 을 계통적으로 늘린다 | **모름** | F1 만 3 · C1a 만 0 · 둘 다 1 (15 pair), McNemar exact p = 0.25. 연결은 궤적 분기 (§4) 를 통해서만 있다 |

### 6. 같은 HOLD 가 F1 이 없는 arm 에도 있다 (CL2 base rate) · offline what-if

**CL2 (F1 없음) 의 placed `no_seed`.** 같은 18 쌍 harness 의 이전 closed-loop (CL2, 별도 run · snapshot `f14ae3e6`) 에서 읽었다. CL3 의 C1a 는 CL2 의 C1a 와 18 / 18 쌍에서 실행 action 이 같다 (`CL3 verify` `executed_actions_identity`).

| arm | placed 도달 run | placed chunk | placed HOLD | placed `no_seed` | 그중 HOLD | `no_seed` 가 있는 run (chunk) |
|---|---|---|---|---|---|---|
| **CL2 C0 (V11AB)** | 14 | 608 | 73 | 78 | **73** | ep11_s112 (37) · ep30_s301 (36) · ep3_s32 (5) |
| CL2 C1a | 16 | 689 | 55 | 37 | 37 | ep54_s541 (37) |
| CL2 C1bp | 13 | 540 | 70 | 70 | **70** | ep11_s112 (35) · ep54_s541 (1) · ep54_s542 (34) |
| CL2 C2p | 2 | 35 | 0 | 0 | 0 | — |
| CL3 C1aF1 | 15 | 641 | 130 | 129 | 129 | 4 run (§4 표) |
| CL3 C1aF1F2 | 15 | 646 | 118 | 116 | 116 | 4 run |

(출처: `numbers.cl2_base_rate_placed_no_seed`, `hypotheses.d_other_post_place_no_seed_certification_gap`. CL2 의 C1bp · C2p 는 같은 CL2 의 다른 arm 이고 정의는 [이전 로그 T44e-CL2 절](AG3S_T0T6_LOG.md) 에 있다. CL2 의 uncertified HOLD 총합은 C0 73 · C1a 37 · C1bp 70 · C2p 0 chunk, 전체 HOLD 는 101 · 119 · 87 · 40.)

- **V11AB (C0) 에도 이 HOLD 가 73 chunk 있다.** 인증 공백은 F1 이전부터 있던 규칙의 문제다. F1 arm 의 129 chunk 가 가장 크기는 하지만 run 수가 3 – 4 개 단위라 크기 차이를 F1 탓으로 읽을 수 없다 (§4 의 p = 0.25).
- 이 공백은 이미 한 번 마주쳤고 남겨둔 것이다: SUBTASK-f (2026-10-03) 가 `no_admissible` 만 인증하고 `no_seed` 는 "사용자 판정 — 그대로 둔다" 로 남겼다. 그때 적은 **되돌아올 신호가 "`no_seed` 가 다른 run 에서도 놓은 뒤 길게 이어질 때"** 였고 ([이전 로그 SUBTASK 절 되돌아올 지점](AG3S_T0T6_LOG.md)), 이 STEP 이 그 신호를 충족했다 (§8).

**phase-gated F1 (close 이후 F1 off) — offline.** 기록 chunk 에서 "그 chunk 가 F1 기록을 가진 uncertified HOLD" 를 센다 (그 chunk 만 바뀔 수 있다): C1aF1 **0 / 130**, C1aF1F2 **0 / 118**. 즉 이 변형은 close 이후 uncertified HOLD 를 한 chunk 도 바꾸지 못한다. 궤적을 통한 closed-loop 효과는 offline 으로 계산할 수 없다 (`design.phase_gated_F1_offline`). closing · lift 에서 F1 이 동작한 chunk 가 있으므로 (99 · 71) F1 을 끄면 그 구간의 collision 처리 효과를 포기한다 — 그 크기는 재지 않았다.

**W1 offline what-if (closed-loop 아님).** `no_seed` chunk 의 grounding_status 를 `no_admissible` 로 바꿔 (= SUBTASK-f 조건: gate on ∧ label home ∧ `placed_seen` ∧ valid ∧ carve 없음) 인증을 다시 계산하고, 다른 hold-action reason 이 기록에 없을 때 execute 로 센 값이다 (`design.W1_what_if`).

| | C1a | C1aF1 | C1aF1F2 |
|---|---|---|---|
| `no_seed` 인데 W1 인증 → execute 로 바뀌는 chunk | 37 | 129 | 116 |
| 그 chunk 의 phase | placed 37 / 37 | placed 129 / 129 | placed 116 / 116 |
| close 이후 HOLD 합 (W1 가정) | 70 | 54 | 6 |
| 그중 uncertified (W1 가정) | **0** | **1** | **2** |
| (기록 그대로) close 이후 HOLD 합 · uncertified | 107 · 37 | 183 · 130 | 122 · 118 |

(출처: `numbers.W1_execute_chunks`, `numbers.certification_replay`, `numbers.after_close_hold_under_W1_per_chunk`. 남는 uncertified chunk 수 1 · 2 는 실패 항목 `no_admissible` + label `pick` 의 chunk 수 (C1aF1 1 · C1aF1F2 2) 와 같다. label 이 `pick` 이라 SUBTASK-f 조건에 맞지 않는다 (산출: 개수 일치로 읽음).) 이 값은 **기록된 chunk 를 per-chunk 로 다시 판정한 것**이다. HOLD 가 execute 로 바뀌면 이후 궤적이 달라지므로 실제 closed-loop 의 HOLD 수가 아니다.

### 7. develope_PLAN 의 어느 표 · 그림이 이 결과를 쓰는가

| 계획서 단위 | 이 STEP 이 주는 것 | 주의 |
|---|---|---|
| **Ablation DT-02 (extent block)** · DT-03 (closed-loop 핵심 contrast) 의 contact handling 행 | grasp 직전 HOLD chunk C1a 12 · C1aF1 6 · C1aF1F2 2 (출처 `T46-CL3.verify.json numbers.arms.*.t7_pre_grasp.hold_chunks_pre8_close`) | close 이후 uncertified HOLD (37 · 130 · 118) 는 **F1 의 경계 효과와 같은 열에 넣지 않는다** — 별도 열 (원인이 인증 규칙) 로 분리하고, W1 이 끝나면 모든 arm 에 같이 적용한 뒤 다시 잰다 (T46c §4) |
| **Figure 3 B (같은 identity 의 ball / 제안 extent / 관측 oracle)** | 경계 장면은 CL3 의 [`t46-cl3-scene.png`](figures/t46/t46-cl3-scene.png) (pre-grasp contact 장면). [`t46b-scene.png`](figures/t46/t46b-scene.png) 는 verifier 가 "post-place, boundary · contact 장면 아님" 이라고 표시한 **HOLD 원인 장면** (`paper_mapping_numbers.figure3_candidate_scene`) | 이 장면을 Figure 3 의 경계 보존 근거로 쓰지 않는다. 한계 L8 (close 이후 HOLD) 의 설명 · supplementary 후보다 |

### 8. 읽을 때의 한계 · 측정하지 않은 것 · 중단 기준 · 되돌아올 지점

| 한계 | 내용 |
|---|---|
| **진단은 offline 이다** | W1 · phase-gated F1 의 수치는 기록 chunk 위 per-chunk what-if 다. closed-loop 효과는 측정하지 않았다 (중단 기준: 새 closed-loop 없음) |
| **표본이 작다** | 18 쌍, placed `no_seed` run 은 arm 당 1 – 4 개. (d′) 의 p = 0.25 는 "F1 이 늘린다 / 늘리지 않는다" 를 가르지 못한다 |
| **`manipulated_link_margin` 은 기록에 없다** | 인증 술어 재생은 그 값을 `None` 으로 가정했고 1350 / 1350 일치했다 |
| **왜 attention 이 support 에 몰리는지는 추적하지 않았다** | 중단 기준에 따라 실패한 인증 항목 (`no_seed`) 까지만 측정했다 (`not_measured`) |
| **HOLD 가 task 결과에 주는 영향** | F1 의 `no_seed` run 4 개는 모두 task success 이고 `no_seed` 가 place 이후에 시작됐다. 홈으로 돌아오는 시간 · safe success 영향은 재지 않았다 (`not_measured`) |
| **grasp 직전 HOLD 감소의 유의성은 집계 기준에 민감하다** | CL3 의 C1a vs C1aF1 의 grasp 직전 HOLD (12 → 6, paired 6 / 0, p = 0.03125, 18 pair) 는 foreign GPU 겹침이 있던 두 pair (ep12_s121 · ep30_s301; 겹친 run 은 보충 재실행으로 executed_actions 가 원본과 같음을 확인 — CL3 verdict) 를 뺀 16 pair 에서 11 → 6, paired 5 / 0, p = 0.0625 다 (`T46-CL3.verify.json numbers.pair_hold.no_overlap`). 이 STEP 의 진단 자체는 server 기록 chunk 를 쓰므로 겹침의 영향을 받지 않는다고 확인하지는 않았다 (미측정) |

**측정하지 않은 것 (`T46b.verify.json not_measured`).** W1 · phase-gated F1 의 closed-loop 효과 (per-chunk what-if 만) · placed 뒤 attention 이 support 에 몰리는 이유 · `manipulated_link_margin` · cuRobo 회귀 기준선 (GPU) · HOLD 의 completion time / safe success 영향.

**되돌아올 지점.**

| 안 고른 것 | 왜 | 되돌아올 신호 |
|---|---|---|
| **phase-gated F1** (close 이후 F1 off) | 0 / 130 · 0 / 118 chunk 만 건드려 HOLD 에 효과가 없다 | closing · lift 의 F1 동작이 collision HOLD 를 만들거나 held object 처리를 해치는 측정이 나올 때 |
| `no_seed` HOLD 를 그대로 둔다 (SUBTASK-f 의 이전 사용자 판정) | 그때는 1 run 의 현상이었다 | **신호 충족:** 다른 run 에서도 놓은 뒤 길게 이어짐 — C1aF1 4 run (끝까지 3), CL2 C0 3 run · C1bp 3 run. T46c 의 W1 (A12) 이 이 판정을 바꾸는 구현이다 (T46c §4 · §6) |
| 원인 추적을 더 넓힘 (attention 이 support 에 몰리는 이유) | 중단 기준 | W1 이 막는 `no_seed` 외에 같은 모양의 HOLD 가 다른 phase 에서 나타날 때 |

**산출물.**

- **측정**: [`handoff/T46b.verify.json`](handoff/T46b.verify.json). 원자료는 run 디렉터리 `/mnt/dev/AG3S_VLA-storage/results/development/20261012T220712+0900_de2042e9be89_ead5f3483344/verify/T46b/` (`raw/diagnosis.json` · `raw/chunks.jsonl` · `raw/cl2_base_rate.json` · `baseline/legacy_base.json`). 입력 CL3 run 은 `.../20261012T155130+0900_218229ac6464_5fb5ff542c33/verify/T46/CL3/` (읽기 전용).
- **figure** (각각 `.json` sidecar 있음): [`scene`](figures/t46/t46b-scene.png) · [`graph`](figures/t46/t46b-graph.png) · [`table`](figures/t46/t46b-table.png).
- **스크립트** (verify.json `code_state.scripts_md5`): `build_verify.py` · `cl2_base_rate.py` · `diagnose.py` · `extract_chunks.py` · `fig_t46b.py`. 코드 수정 없음.
- 요청 문서: [`handoff/T46b.task.md`](handoff/T46b.task.md). commit `ed47b976`.


---

## T46c — F1 채택 (한계 L1–L9 를 함께 기록) · close 이후 HOLD 원인 · W1 구현 (~~진행 중~~ → 완료, 아래 "T46c-W1 · T46d · T46f" 절) · PA-1 ablation 초안 (2026-10-12 KST)

**이 절이 답하는 물음 (목적).** T46b 의 진단 뒤에 정할 것이 셋이다. (1) F1 (non-target 근거에서 target 자신의 표면 점을 빼는 surface 근거) 을 develope_PLAN §0.10.1-3 의 **"경계 보존 방법 하나"** 로 현재 범위에서 채택할 것인가. (2) 채택하면서 close 이후 HOLD 증가를 어떻게 기록할 것인가. (3) 그 방법을 논문의 ablation 에 어떻게 넣을 것인가. 이 절은 (1) 과 (2) 의 판정을 기록하고, 그 판정이 요구한 구현 (W1) 이 **진행 중임** 과 (3) 의 초안이 **판정 대기임** 을 적는다. **W1 의 결과는 아직 없다.**

**한 줄 결과 (before → after).**

- **before.** F1 은 grasp 직전 경계 효과를 보였지만 (HOLD run 12 → 6, 이웃 과일 접촉 run 4 → 1) 성공률은 같았고 (task 14 → 13) close 이후 HOLD 가 107 → 183 chunk 로 늘어, 채택 여부와 논문 주장 범위가 정해지지 않았다.
- **after (사용자 판정, 2026-10-12).** **F1 을 현재 extent 방법으로 채택한다.** "권장하는 A 로 진행. **한계는 명확하게 작성하고 넘어갈 것.** F1 이 grasp 이후 HOLD 를 늘린 원인을 작성할 것." 한계 9 개 (L1–L9) 는 §3 에 논문 주장 범위의 제약과 함께 적었다. 채택 근거는 성공률이 아니라 **같은 identity 에서 extent 처리만 바꿨을 때 grasp 직전 충돌 행 887 → 401 (offline), HOLD run 12 → 6 (closed-loop), 잃은 이웃 과일 seed 0 → 0** 이다.
- **close 이후 HOLD 의 원인** 은 F1 이 아니라 placed 뒤 `no_seed` 인증 공백이다 (T46b). 늘어난 uncertified HOLD 130 chunk 에서 F1 기록은 0 이고, F1 이 없는 V11AB (C0) 에도 73 chunk 가 있다.
- **고칠 곳은 인증 규칙이다 (lead).** W1 (placed · home 의 `no_seed` 를 인증) 을 ~~구현 중이다. **결과 없음.**~~ → **구현 · offline 재현 완료 (2026-10-12 23:40 KST, 아래 "T46c-W1 · T46d · T46f" 절 §2).**
- **PA-1 ablation 구성** (extent 를 주 block 으로) 은 **판정 대기** 다.

**읽는 법과 출처 규약.**

- 수치는 [`T46.verify.json`](handoff/T46.verify.json) (offline) · [`T46-CL3.verify.json`](handoff/T46-CL3.verify.json) (closed-loop) · [`T46b.verify.json`](handoff/T46b.verify.json) (HOLD 진단) 의 `numbers` 에서 왔다. **세 파일 어디에도 없는 내용은 "(task 문서)" 로 표시했다** — 그것은 lead 가 [`T46c.task.md`](handoff/T46c.task.md) 에 쓴 서술이고 verifier 가 재지 않았다. 이 절의 한계 표 (§3) 의 "논문 표현에 주는 제약" 열은 전부 lead 의 문장이다.
- 판정의 출처: **(사용자 판정)** = 위 인용, **(lead)** = task 문서의 결정 · 초안, **(산출)** = 같은 파일 숫자의 뺄셈 · 나눗셈.
- **C0 비교의 한계.** C0 (V11AB) 수치는 CL2 (같은 18 쌍 harness 의 이전 run) 에서 왔고, C1a · C1aF1 은 CL3 다. CL3 의 C1a 와 CL2 의 C1a 는 18 / 18 쌍에서 실행 action 이 같아 (`CL3 verify` `executed_actions_identity`) CL2 의 C0 를 같은 기준으로 놓았지만, C0 와 F1 은 같은 run 이 아니다.

### 1. 사용자 판정과 타임라인 (KST)

| # | 판정 · 사건 (2026-10-12) | 이 절에서 |
|---|---|---|
| 1 | **(사용자 판정)** "권장하는 A 로 진행. 한계는 명확하게 작성하고 넘어갈 것. F1 이 grasp 이후 HOLD 를 늘린 원인을 작성할 것." | §2 · §3 · §4 |
| 2 | **(lead)** F2 (`--owned-ambiguous remove`) 는 채택하지 않고 대조 arm 으로만 남긴다 — 이웃 과일 seed 를 지워 §0.10 의 "인접 장애물 보존" 목적과 어긋난다 | §2 |
| 3 | **(lead)** W1 (A12) 구현 → offline 재현 (verifier) → 사용자 보고. 병행: 이 기록. PA-1 판정 뒤 ablation closed-loop task | §5 · §6 |

| 시각 (KST) | 무엇 | 출처 |
|---|---|---|
| 23:02 | T46c task 작성 (lead) | `T46c.task.md` mtime 14:02Z |
| 23:03:13 | T46c task commit `6d134a69` | `git log` |

(pod 에 tzdata 가 없어 UTC + 9 h 로 환산했다.)

### 2. F1 채택의 근거 — 같은 identity 에서 extent 처리만 바꾼 비교 (before → after)

**offline (grasp 직전 window, held-out 162 chunk, CL2 C1a 의 기록 chunk 를 재분류).** 대조는 C1a (`ball_owned`) 하나이고 C0 는 이 offline 에 없다. 숫자 앞의 `off` 는 F1 · F2 를 끈 기록과 bit 단위로 같다 (162 / 162 chunk, 최대 차이 0.0 mm).

| 지표 | C1a (`off`) | **C1a + F1** | (참고) F2 | (참고) F1 + F2 | (상한) F1GT |
|---|---|---|---|---|---|
| 충돌 행 (grasp 직전 162 chunk) | 887 | **401** | 136 | 107 | 113 |
| 충돌 행이 남은 chunk (14 개 중) | 14 | **8** | 8 | 6 | 6 |
| 충돌 행 0 이 된 chunk | 0 | **6** | 6 | 8 | 8 |
| 충돌 행 0 이 된 recorded HOLD chunk (12 개 중) | 0 | **5** | 4 | 6 | 6 |
| 새로 충돌이 생긴 chunk | — | **0** | 0 | 0 | 0 |
| **잃은 이웃 과일 seed (전 chunk)** | 0 | **0** | **2463** | 2463 | 48 |

(출처: `T46.verify.json numbers.3_4_5_heldout.variants`. F1GT = GT target 점을 non-target 근거에서 빼는 상한 대조로, GT 는 대조 · 평가에만 쓴다. F2 는 이웃 과일 seed 2463 개를 잃어 채택하지 않는다.)

**closed-loop (CL3, 18 쌍).** 18 쌍 = 9 episode × 2 seed (산출: `per_run` key). 각 칸은 run 수 (또는 chunk 수), 괄호는 C1a 대 C1aF1 의 paired 결과 (한쪽만 / 반대쪽만, McNemar exact p).

| 지표 | V11AB (C0, CL2) | C1a | **C1a + F1** | 비교 (C1a 대 F1) |
|---|---|---|---|---|
| grasp 직전 (window) HOLD run | 1 | 12 | **6** | 6 / 0, **p = 0.031** |
| grasp 직전 HOLD chunk | 1 | 12 | **6** | |
| pre-release 새 접촉 run (hand–이웃 과일 · 비손 link–target · 비손 link–이웃 과일) | 4 | 4 | **1** | 3 / 0, p = 0.25 |
| task success | 14 | 14 | 13 | 1 / 0, p = 1.0 |
| CT success (task + 접촉 없음) | 12 | 11 | 12 | 1 / 2, p = 1.0 |
| grasp (final) | 16 | 17 | 16 | 1 / 0, p = 1.0 |
| HOLD 전체 chunk | 101 | 119 | 189 | |
| 그중 uncertified | 73 | 37 | 130 | (원인은 §4) |

(출처: `T46-CL3.verify.json numbers.pairs` · `numbers.pair_hold` · `numbers.arms.*` · `numbers.reproducibility_vs_CL2` (C0 열). `pre_release_new_contact` 의 class 정의는 CL3 verify `definitions`. window HOLD 는 C1aF1F2 가 2 chunk 다.)

![T46 CL3 대표 장면 (pre-grasp contact)](figures/t46/t46-cl3-scene.png)

[`t46-cl3-scene.png`](figures/t46/t46-cl3-scene.png) — 이 채택의 경계 효과를 보이는 CL3 의 pre-grasp 장면 (CL3 verifier 가 만든 그림, 상세는 [이전 로그 T46 절](AG3S_T0T6_LOG.md)). [`t46-cl3-graph.png`](figures/t46/t46-cl3-graph.png) · [`t46-cl3-table.png`](figures/t46/t46-cl3-table.png) 는 같은 비교의 그래프 · 표다.

**이 표를 읽는 법 — 채택이 주장하는 것과 주장하지 않는 것.**

- 주장하는 것: F1 은 같은 identity · 같은 controller 에서 extent 처리만 바꿨을 때 **grasp 직전 충돌 행 · HOLD · 이웃 과일 접촉을 줄이고 이웃 seed 를 잃지 않는다**. 이는 §0.10.1-3 의 "잘못된 접촉 차단을 줄이면서 인접 장애물 경계를 보존한다" 와 같은 방향이다 (DC-1).
- **주장하지 않는 것: 성공률 향상.** task success 는 14 → 13 으로 같은 수준이다 (유의하지 않음, §3 L1).
- **추가 관찰 (scribe 정리, 위 표의 수치에서).** V11AB (C0) 의 grasp 직전 HOLD 는 1 chunk 라 F1 의 6 chunk 보다 적다. F1 의 12 → 6 은 `ball_owned` (C1a) 가 늘린 grasp 직전 HOLD 를 **일부** 되돌린 것이지 C0 보다 줄인 것이 아니다. 같은 표에서 이웃 과일 접촉 run 은 C0 4 · C1a 4 · F1 1 이다. "F1 이 V11AB 보다 grasp 직전 HOLD 를 줄인다" 는 주장은 이 근거로는 성립하지 않는다 (이 trade-off 의 해석은 DC-4 의 몫이고 아직 정하지 않았다).
- 12 → 6 의 p = 0.031 은 18 pair 의 값이고, foreign GPU 겹침 pair 를 뺀 16 pair 에서는 11 → 6, paired 5 / 0, p = 0.0625 다 (`numbers.pair_hold.no_overlap`). 유의성은 집계 기준에 민감하다.

### 3. F1 의 한계 L1–L9 (채택과 함께 기록, 논문 주장 범위를 정한다)

(사용자 지시: "한계는 명확하게 작성하고 넘어갈 것.") **"논문 표현에 주는 제약" 열은 lead 의 문장이다 (task 문서 §2).** "근거" 열은 verify.json 의 수치이고, 그렇지 않은 것은 "(task 문서)" 로 표시했다.

| # | 한계 | 근거 | 논문 표현에 주는 제약 (lead) |
|---|---|---|---|
| **L1** | **task success 를 개선하지 못했다** | task success 14 → 13, CT success 11 → 12, grasp 17 → 16 (C1a 대 F1, 18 쌍, 모두 McNemar p = 1.0) | "성공률 향상" 을 주장하지 않는다. grasp 직전 불필요한 정지 · 이웃 접촉 감소로만 쓴다 |
| **L2** | **표본이 작다** | 18 쌍 = 9 episode × 2 seed (`per_run` key 에서 산출). train split · 이웃 과일 < 0.10 m 장면 · apple / orange / pear · banana 0 은 (task 문서); banana 는 CL3 verify `not_measured` 에 "no passed episode in LB2" | 효과 크기와 p 값을 같이 쓰고 일반화를 주장하지 않는다. 주 평가 (SafeLIBERO) 결과 전에는 보조 근거 |
| **L3** | **표면 구 맞춤이 물러나면 F1 이 동작하지 않는다** | CL3 에서 F1 이 적용된 chunk 396 / 559, 적용되지 않은 163 chunk 의 이유는 모두 `no_surface_fit` (163 / 559 = 29.2 %, 산출). offline 에서는 충돌 chunk 14 개 중 6 개, 전체 162 chunk 중 54 개가 fit 물러남. 그 6 개는 target 점에 이웃 과일 점이 섞인 경우다 (task 문서) | 이웃과 맞붙은 · 점이 섞인 경우의 경계 오류는 남는다 (F3 = target 점 오염 · identity / split 쪽 과제) |
| **L4** | **잔여를 다 지우지 못한다** | offline 충돌 행 887 → 401, GT 상한 (F1GT) 113 | "잔여 제거" 가 아니라 "감소" 로 쓴다 |
| **L5** | **구형 가정** | F1 은 ball 이 맞춘 표면 구 + 껍질 10 mm (task 문서; 껍질 기본값은 [이전 로그 용어](AG3S_T0T6_LOG.md)) | 비구형 물체 (banana · 상자 · 길쭉한 물체) 는 검증하지 않았다 |
| **L6** | **설계에 GT 를 본 chunk 가 있다** | 구현자가 D-TF (target-free 층의 target 잔여) 17 chunk 의 GT 라벨을 보고 설계했다 (task 문서). 판정은 held-out 162 chunk 로만 했다 | design set 수치는 근거로 쓰지 않는다 |
| **L7** | **비권한 link (C2′) 문제를 풀지 못한다** | C2′ (= C1b′ + `--links arms` + `--nonauthorized-target-margin obstacle`, [이전 로그 정의](AG3S_T0T6_LOG.md)) window 206 chunk 의 단일 chunk 재풀이 (closed-loop 아님) 에서 row 7 중앙값 69.70 → 68.45 mm (task 문서는 이를 손가락–target 거리로 부른다), 짝지은 차의 중앙 0.0 mm, status `violated` 34 → 28 | DX-04 (link 별 예외) 의 근거로 쓰지 않는다 |
| **L8** | **close 이후 HOLD 가 늘었다 (간접 영향)** | close 이후 HOLD 107 → 183 chunk, uncertified 37 → 130 (§4) | W1 적용 전 수치를 그대로 쓰면 safe success · stop 에 불리하게 보인다. W1 은 모든 arm 에 같이 적용한다 |
| **L9** | **시뮬레이션 · 단일 checkpoint · 실시간 아님** | RB-Y1 MuJoCo · 16D LoRA 30k (task 문서). 시간: TO solve p90 599.9 (C1a) · 586.0 (F1) ms, client round trip 중앙값 1053.2 · 1057.8 ms · p90 1674.8 · 1662.8 ms (chunk 1332 개씩, 18 run) — F1 이 시간을 늘리지 않았다 | 실물 · 다른 checkpoint · real-time 을 주장하지 않는다 |

(출처: L1 `T46-CL3.verify.json numbers.pairs` · L3 `numbers.arms.C1aF1.t8_t46` 와 `T46.verify.json numbers.3_4_5_heldout` · L4 `T46.verify.json` · L7 `numbers.6_c2p_row7` · L8 `numbers.after_close_hold` (`T46b.verify.json`) · L9 `T46-CL3.verify.json numbers.arms.*.t5_timing`. F1 의 offline 추가 비용은 evidence 단계 p90 증가 0.20 ms (`T46.verify.json numbers.7_build_ms`, 36 frame).)

### 4. F1 이 grasp 이후 HOLD 를 늘린 원인 (`T46b.verify.json`, 사용자 지시 "원인을 작성할 것")

자세한 근거는 위 "T46b" 절의 §2 – §6. 여기서는 사용자 지시에 맞춰 원인을 한 곳에 모은다.

1. **늘어난 곳.** 늘어난 uncertified HOLD 는 전부 **과일을 놓은 뒤 (placed phase, subtask label `home`)** 다. closing · lift · carry 에서는 0 이다 (C1a 37 · C1aF1 130 · C1aF1F2 118 chunk 가 모두 placed).
2. **직접 원인 — 인증 규칙의 공백 (F1 이전부터 있었다).**
   - 놓은 뒤 VLA attention 상위 5 % seed 가 모두 support mask 안에 있어 (`no_seed` chunk 마다 최솟값 1.0) support mask 로 지우면 seed 가 0 → grounding `no_seed` (`AG3S/ag3s/stages/target_grounding.py:1679-1686`, `AG3S/ag3s/runtime/pipeline.py:1513-1519`).
   - placed · home 의 인증 `subtask_no_target` (`AG3S/ag3s/types.py:1069-1113`) 은 `subtask_gated` 와 `no_admissible` 만 인증하고 **`no_seed` 는 목록에 없다**.
   - 그래서 위반 행이 0 인데도 "기하 인증 없음" 으로 TO verdict 가 `violated` 로 강등되고 (`AG3S/trajopt/sqp.py:922`) HOLD 가 난다. episode 끝까지 HOLD 가 이어진 run 은 C1aF1 에서 3 개 (ep11_s111 · ep30_s301 · ep54_s541) 이고 그중 ep54_s541 은 C1a 도 같다 (T46b §4 의 정정 메모).
   - 같은 HOLD 가 F1 이 없는 CL2 에도 있다: C0 73 · C1a 37 · C1bp 70 chunk.
3. **F1 의 역할 — 간접.**
   - 그 HOLD chunk 130 개에서 F1 경로는 한 번도 돌지 않았다 (F1 기록 0 / 130).
   - F1 은 grasp 직전 field 를 바꿔 (점 1–12 개 제외 — task 문서; verify.json 은 window 의 뺀 점 합 444) 18 쌍 중 12 쌍의 궤적이 grasp 직전 window 에서 처음 갈라졌다.
   - 궤적이 달라지면 놓은 뒤 자세 · attention 도 달라지고, 그중 `no_seed` 가 생긴 run 은 F1 만 3 · C1a 만 0 · 둘 다 1 이었다 (15 pair, p = 0.25) — **F1 이 계통적으로 늘린다고 말할 수 없다.**
4. **고칠 곳.** F1 이 아니라 인증 규칙. **phase-gated F1 (close 이후 F1 off) 은 0 / 130 chunk 만 바꿔 효과가 없다.**

| 원인 후보 | 판정 | 요약 |
|---|---|---|
| (a) F1 이 grasp 이후 held / target 처리와 충돌 | 반박 | uncertified 는 closing · lift · carry 에서 0 |
| (b) F1 이 non-target 근거를 줄여 설명되지 않은 표면이 늘어남 | 반박 | 실패 항목은 `no_seed` (seed 가 전부 support 안) |
| (c) C1a 가 일찍 실패해 분모가 작음 | 반박 | placed 도달 run 16 대 15, chunk 689 대 641 |
| (d) placed 뒤 `no_seed` 인증 공백 | **확인** | CL2 C0 73 · C1bp 70 (F1 없음) |
| (d′) F1 이 `no_seed` 를 계통적으로 늘림 | 모름 | 3 / 0 / 1, p = 0.25 |

### 5. W1 (A12) — placed `no_seed` 인증 공백의 구현 (~~**진행 중, 결과 없음**~~ → **완료, 2026-10-12 23:40 KST 갱신: 아래 "T46c-W1 · T46d · T46f" 절 §2**)

**목적 (lead).** F1 채택과 함께 close 이후 HOLD 증가를 인증 규칙 쪽에서 해소하고, 모든 arm 에 같은 수정을 적용해 비교 대상에서 뺀다.

| 항목 | 내용 (task 문서 §4, lead) |
|---|---|
| **무엇** | placed · home 의 `no_seed` 를 기존 SUBTASK-f 인증 조건 (gate on ∧ label home ∧ `placed_seen` ∧ valid ∧ carve 없음) 아래서 `no_admissible` 과 같이 인증한다. 위치: `AG3S/ag3s/types.py:1069-1113` (`subtask_no_target`) 와 그 소비자. 기준 술어는 `T46b.verify.json` 의 `certification_replay` |
| **flag** | 기본 off (예: `--placed-no-seed-certify` / config 필드). off 에서 bit-identical. 켜도 placed 이전 phase · label `pick` 의 동작은 바뀌지 않음을 test 로 확인 |
| **안전 조건 (test)** | `no_seed` 라도 위반 행이 있으면 HOLD 유지 (인증은 "설명 안 됨" 만 해소, collision 은 그대로). `placed_seen` 이 없거나 label 이 `pick` 이면 인증하지 않음 |
| **offline 재현 목표** | CL3 기록 chunk 에서 variant `W1` · `F1W1` 로 T46b 의 W1 수치가 재현되는지: execute 로 바뀌는 chunk 37 / 129 / 116, close 이후 uncertified 0 / 1 / 2 chunk (C1a · C1aF1 · C1aF1F2). 이 목표 수치는 `T46b.verify.json` (`W1_execute_chunks` · `after_close_hold_under_W1_per_chunk`) 에서 왔다 |
| **손대지 않을 것** | F1 · F2 · ball_owned · coarse · permission 로직, T47 session 파일 |
| **절차** | implementer 구현 + 단위 test + 전체 pytest (GPU 는 공용 lock) → `T46c.impl.md` → verifier 의 offline 재현 → 사용자 보고 |

- ~~**상태.** 이 기록을 쓰는 시점에 `T46c.impl.md` · `T46c.verify.json` 이 없다.~~ (2026-10-12 23:40 KST 이후 둘 다 있다.) ~~**W1 수치는 아직 측정된 것이 없으며** 위 목표 수치는 T46b 의 per-chunk what-if 다 (closed-loop 아님).~~ → 위 목표 수치는 V-T46c 가 재현했다 (아래 절 §2). 그 수치는 여전히 per-chunk what-if 이며 closed-loop 이 아니다.
- ~~**주의 (lead 확인 필요).** W1 은 SUBTASK 절 되돌아올 지점 표의 "`no_seed` HOLD 를 인증 — **사용자 판정: 그대로 둔다**" 를 바꾸는 구현이다 ([이전 로그](AG3S_T0T6_LOG.md)). 그 표가 적은 전환 신호 (`no_seed` 가 다른 run 에서도 놓은 뒤 길게 이어짐) 는 T46b 에서 충족됐다. 사용자 문장 "권장하는 A 로 진행" 이 W1 구현을 명시적으로 포함하는지는 이 로그에 근거가 없어 **적지 않는다** — T46c task 문서는 W1 을 지시했으나 "이전 판정을 번복한다" 는 사용자 확인은 기록돼 있지 않다.~~ → **해소: 사용자 판정 (2026-10-12, 추천안 채택) 이 2026-10-04 판정을 바꾸는 것을 명시했다 (아래 절 §1).**

### 6. PA-1 — Ablation 구성 초안 (~~**판정 대기**~~ → 일부 판정됨: (b) 최소 범위 · (c) 18 쌍 유지 · W1 모든 arm on, 아래 절 §1. lead 초안)

develope_PLAN DX-01 (identity / extent 분리) · DX-04 (contact handling) · DT-02 · DT-05A 에 맞춘 초안이다. 모든 arm 에 W1 을 같이 적용한다 (공통 인증 수정은 비교 대상이 아니다).

| block | 고정 | 바꾸는 것 (arm) | 측정 | 비고 |
|---|---|---|---|---|
| **DT-02 extent (주)** | identity = ATS, LTE, controller | ball (C0) · `ball_owned` (C1a) · **`ball_owned` + F1 (제안)** · visible-target oracle | extent leakage (이웃 seed 손실) · retention (target 잔여) · grasp 직전 HOLD · task / safe success · forbidden contact | oracle arm 은 새 구현 필요 (GT 관측 mask) — 범위 판정 필요 |
| DT-02 identity | extent = 제안 | ATS · geometry / proximity rule · oracle identity | identity error · false exemption · 실행 | geometry rule arm 의 구현 상태 확인 필요 |
| **DT-05A contact handling** | identity · extent = 제안 | no exemption · global exclusion · LTE | 비권한 link–target contact · grasp · stop | **선행 조건 (DX-04):** query set 에 비권한 link 가 있어야 한다. `--links gripper` 는 퇴화 (task 문서), `--links arms` (C2′) 는 "1 / 18" (task 문서; 이유를 적지 않았다 — 이전 로그의 CL2 C2′ task success 1 / 18 과 같은 수로 보인다). 범위 판정 필요 |
| 공통 | 같은 18 쌍 (또는 확대), 같은 snapshot · W1 on | — | episode 단위 paired 비교 · CI | 주 평가 SafeLIBERO 는 benchmark branch (별도) |

**지금까지 이 초안의 DT-02 extent 열에 이미 들어갈 수 있는 값** (ball · `ball_owned` · F1 은 §2 의 표, oracle 은 없음):

| arm | extent leakage (이웃 seed 손실, offline) | grasp 직전 HOLD run (closed-loop) | task success | 출처 |
|---|---|---|---|---|
| ball (C0) | 측정 안 됨 (offline 대조 없음) | 1 | 14 | `T46-CL3.verify.json` (CL2 열) |
| `ball_owned` (C1a) | 0 | 12 | 14 | `T46.verify.json` · `T46-CL3.verify.json` |
| `ball_owned` + F1 | 0 | 6 | 13 | 같음 |
| visible-target oracle | 구현 없음 | 구현 없음 | 구현 없음 | — |

**판정이 필요한 것 (lead, task 문서 §5).** (a) visible-target oracle extent arm 을 만들지 (GT 관측 mask 신규 구현), (b) DT-05A 를 위한 비권한 link 범위 (wrist 만 추가 등 §0.10.1-1 의 "최소 범위"), (c) 표본 수 (18 쌍 유지 · 확대).

### 7. 되돌아올 지점 · 판정 대기

| 안 고른 것 | 왜 | 되돌아올 신호 |
|---|---|---|
| F1 을 "grasp 직전 효과로만 주장" 하고 채택하지 않음 (T46b 중단 기준의 대안) | 사용자가 채택 (A) 을 택했다 | F1 의 한계 L3 · L5 · L7 이 주 평가 (SafeLIBERO) 에서 지배적인 실패 원인으로 나타날 때 |
| F2 채택 | 이웃 과일 seed 2463 개 손실 | 이웃 과일 보존이 문제가 되지 않는 장면에서 F2 의 추가 감소 (충돌 행 401 → 107, F1 + F2) 가 실행 이득으로 이어진다는 측정이 나올 때 |
| phase-gated F1 | 0 / 130 chunk 만 바꾼다 | T46b §8 |

~~**판정 대기.** (1) PA-1 의 (a) – (c) (§6). (2) W1 이 SUBTASK 절의 이전 사용자 판정을 바꾸는 점을 사용자가 확인했는지 (§5 주의). (3) W1 구현 · offline 재현 뒤 close 이후 HOLD 를 모든 arm 에 W1 을 켠 상태로 다시 잴 것인가 (closed-loop 은 새 task).~~ → **2026-10-12 갱신:** (1) (b) 비허용 link 범위는 최소 범위부터 (T46d 에서 기록으로 선택), (c) 표본은 18 쌍 유지로 판정. (a) visible-target oracle extent arm 은 `T46e.task.md` 가 "계획서상 필요" 로 적고 구현을 지시했다 (구현 완료 · 미적용, 검증 없음). (2) 해소 (위). (3) W1 on 으로 다시 재는 closed-loop 은 Ablation 단계 (T46f 단계 4 이후) 의 몫이다.

**산출물.**

- **측정 (이 절이 인용)**: `handoff/T46.verify.json` · `handoff/T46-CL3.verify.json` · `handoff/T46b.verify.json`. ~~**T46c 자체의 verify.json 은 아직 없다** (W1 결과 없음).~~ → [`handoff/T46c.verify.json`](handoff/T46c.verify.json) 이 2026-10-12 23:40 KST 에 생겼다 (아래 절).
- **figure**: 새로 만든 것 없음 — [`t46-cl3-*`](figures/t46/) (CL3) · [`t46b-*`](figures/t46/) (T46b) 를 링크했다.
- 요청 문서: [`handoff/T46c.task.md`](handoff/T46c.task.md) (commit `6d134a69`). commit 은 하지 않았다.


## T51 · T51b — attention head 를 하나로 고정해 써도 되는가: 사전 고정한 선택 protocol 과 명사 query 비교 (2026-10-12 KST)

**이 절이 답하는 물음 (목적).** AG3S 는 attention 이 가리키는 3D cluster 에 target 이름을 붙이는데, 그 attention 은 지금 고정된 cell 하나 (`denoise 0 · agg last · layer 8 · head 2`, 이하 **L8H2**) 에서 온다. 이 cell 은 14D 모델 · `cam_high` 1 대 · 28 frame 에서 `peak_on_target` 하나로 골랐고 (T47 절 서술), 절차로 설명되지 않는다. T47 은 논문 (Kang et al., arXiv 2503.06287) 의 label-free 기준이 action-token query 에서 sink head 와 넓게 퍼진 map 을 고른다는 것을 보였다. 리뷰어가 납득하려면 **"이 규칙을 따르면 어느 모델 · 로봇에서도 같은 방식으로 cell 이 정해지고, 그 선택이 얼마나 안정적인지 수치로 말할 수 있다"** 가 필요하다. T51 은 그 규칙을 **결과를 보기 전에 고정**한 protocol (P1 filter → P2 calibration → P3 bootstrap 안정성 → P4 k 선택 → P5 held-out → P6 ablation) 로 만들어 RB-Y1 sim 에서 돌리고, 실물용 label 없는 대안 (grasp-event self label) 과 다른 모델 (LIBERO `pi05_libero`) 에서도 돌려 본다. T51b 는 T49 의 명사 token query 행을 같은 후보 집합 Ω 에 넣어 **action query 와 명사 query 를 같은 기준으로** 비교한다.

**한 줄 결과 (before → after).**

- **before.** 서빙 cell L8H2 는 절차 없이 골라졌고, 시험한 label-free 대안 (T47) 은 실패했다. "하나로 고정" 에 데이터의 근거가 없었다.
- **after (RB-Y1, scene mean primary).** 사전 고정 절차의 출력은 **GT 3D `identity_correct` 1 위 `last:L12H4` (scene mean 0.691, 95 % CI [0.515, 0.840])** 이다. 그러나 **bootstrap 에서 1 위가 되는 확률 P(rank1) 은 0.429 로 사전 고정 기준 (≥ 0.5) 에 못 미쳤다.** 규칙대로 **"단일 head 고정" 은 주장하지 않는다.** P4 는 단순성 규칙으로 k\* = 1 을 냈으나 CV 평균이 가장 높은 것은 k = 3 이다 (차이 CI 가 0 을 포함).
- **held-out (3 scene · 60 frame).** 선택 `last:L12H4`, self label 로 고른 `last:L12H3`, 서빙 `last:L8H2` 가 **모두 scene mean 0.911 로 동률**이고 frame 단위로도 같은 frame 을 맞힌다. 논문 label-free top-3 는 0.000, head 평균 0.867, 무작위 cell 20 개의 median 은 0.756. 이 held-out 은 세 cell 을 가르지 못한다.
- **P1 은 L8H2 를 제외한다** (`low_mass`, cam_high 의 image mass median 0.006293 < 0.01). 그런데 held-out 에서 선택과 동률이다. threshold 는 다시 맞추지 않고 측정된 대로 기록한다.
- **self label** (사람 annotation 없음): 선택 cell 은 GT 와 다르다 (`last:L12H3` 대 `last:L12H4`, 상위 3 의 겹침 1) 그러나 `self_identity` 가 상위 9 cell 모두 1.0 으로 **포화**해 선택을 2D lift tie-break 가 정했다. held-out 의 GT `identity_correct` 차이는 4 / 4 record 에서 0.
- **LIBERO** (2D lift primary, task 단위, `pi05_libero`): 1 위 `mean:L12H3`, task bootstrap P(rank1) 1.0. 3D 지표는 없다.
- **T51b.** 명사만: GT 1 위 `noun_target:L10H7` (P(rank1) 0.687), 합친 Ω: GT 1 위 `last:L12H4` (P(rank1) 0.753, action family 합 0.918). held-out scene mean 은 **명사 best 0.867 대 action best 0.911 (scene 단위 차 −0.044, 95 % CI [−0.133, 0.000])**. 명사가 더 낫다는 증거는 없다.
- **서빙 반영 없음.** 서빙 cell 은 L8H2 그대로다.

**정직하게 먼저 적어 둘 것 (결과를 읽기 전에).**

| # | 무엇 | 어디 |
|---|---|---|
| 1 | **집계 단위 deviation.** 사전 고정은 scene mean 인데 첫 구현은 record 평균으로 순위를 매겼다. 실행 중 verifier 가 발견했고 `--aggregate scene` 으로 고쳐 **scene 단위로 전 단계를 다시 돌렸다.** 이 절의 primary 는 scene mean 이다. record 평균 결과는 "사전 고정과 다른 방식" 으로 따로 보고한다 | §4 |
| 2 | **안정성 기준 미충족.** GT 선택의 P(rank1) = 0.429 < 0.5. 되돌아올 지점에 따라 단일 head 고정을 주장하지 않는다 | §7 |
| 3 | **L8H2 는 P1 low_mass 로 제외됐지만 held-out 에서 선택과 같다.** threshold 를 다시 맞추지 않았다 | §5 · §9 |
| 4 | **self label 포화.** rank-1 tie rate 1.0 이라 self 선택은 2D lift tie-break 로 정해졌다. "self 와 GT 의 선택이 (부분) 일치한다" 를 self label 의 검증으로 읽지 않는다 | §10 |
| 5 | **T51b 의 forced cell 구현 불일치.** lead 가 고정한 명사 forced cell 3 개가 코드의 forced 목록에 없었다. 이 3 개가 이미 모든 선별에 들어 있어 결과는 바뀌지 않았다 | §12 |
| 6 | **LIBERO 는 2D primary 만.** 3D `identity_correct` 가 불가능하고 KNOWS 재현은 하지 않았다 | §11 |

**읽는 법과 출처 규약.**

- 수치는 `T51.verify.json` · `T51b.verify.json` 의 `numbers` 에서만 왔다. **T51 은 `numbers.scene_mean_primary` 블록이 PRIMARY** 이고, 그 밖의 블록 (`p3_bootstrap` · `p4_ksel_record_mean_code_deviation` · `p5_heldout_record_mean_selection` · `p6_ablation_heldout` · `self_label` · `annotate_curve_gt_exploratory`) 은 record 평균 chain (사전 고정과 다른 방식) 이다. 해당 표에 "record-mean" 이라 적었다. 같은 파일 숫자의 나눗셈 · 덧셈은 "(산출)", 해석은 "(미측정 해석)" 이다.
- **frame 은 독립 trial 이 아니다.** 판정은 scene 단위 (calibration 5 scene, held-out 3 scene) 로 한다. frame 단위 값은 참고다.
- **calibration 의 `ep0` · `ep500` · `ep1200` 은 VLA 학습에 쓰인 episode 의 기록이다** (사용자 판정으로 추가). held-out 4 record 는 VLA 학습에 없다.
- 회귀 기준선: legacy `esdf_rollout` (`run_16d_ep1800`) 이 start 시점과 scene 재실행 시점 모두 `violated_start` 14 / 15 · `has_target` 15 / 15 · frame 0 clearance_before −29.031048 mm 로 일치. cuRobo 기준선은 GPU 미사용 지시로 돌리지 않았다.

### 1. 사용자 판정과 사전 고정 (타임라인 · KST)

| # | 판정 · 사건 (2026-10-12) | 이 절에서 |
|---|---|---|
| 1 | **(사용자 판정)** head · layer 를 하나로 고정해 쓰려면 선택의 당위성이 있어야 하고 실물 · LIBERO 에서도 같은 절차로 고를 수 있어야 한다. 범위 = RB-Y1 sim protocol + 실물용 grasp-event self label + LIBERO. **primary = 3D `identity_correct`**, 2D lift 는 depth 없는 플랫폼용 보조 | 전체 |
| 2 | **(사용자 판정, T51.impl.md 확인 후 · 결과 전)** LIBERO L1 승인 — **KNOWS 재현 확인 없이** 저장 frame 에 `pi05_libero` attention pass 를 한 번 돌린다 (새 rollout 없음). primary = 2D lift. **calibration 확장 승인** — VLA 학습 episode `ep0` · `ep500` · `ep1200` 추가 (결과에 학습 episode 임을 명시). bootstrap 은 scene 단위 | §3 · §11 |
| 3 | **(lead, 결과 전 고정)** P2 점 추정은 scene mean, LIBERO 는 task 단위 | §4 |
| 4 | **(lead, verifier 발견 후 수정)** 첫 구현이 record 평균이었음 → `--aggregate scene` 추가, scene mean 이 primary. held-out 비교도 held-out scene mean | §4 |
| 5 | **(사용자 판정, T49 결과 후)** T51b 추가. offline GPU prefix pass (GPU lock) 승인 | §12 |
| 6 | **(lead, T51b.impl.md ★ 판정, 결과 전)** 명사 forced cell = T49 primary 명사 cell L8H6 · L5H4 · L4H2. (c) 의 primary 통계 = scene 단위 paired 차이 + scene bootstrap CI, McNemar 는 참고 | §12 |

| 시각 (KST) | 무엇 | 출처 |
|---|---|---|
| 19:28:18 – 19:29:01 | GPU: RB-Y1 extra attention parity (`ep1807` 첫 4 frame, rc 0, wall 43.9 s, 시작 전 memory 0 MiB) | `gpu_extract[0]` 10:28:18Z – 10:29:01Z |
| 19:29:03 – 19:32:35 | GPU: `ep0` · `ep500` · `ep1200` attention extract (rc 0, wall 211.9 s) | `gpu_extract[1]` |
| 19:32:40 – 19:37:43 | GPU: LIBERO `pi05_libero` attention pass (40 episode, rc 0, wall 302.9 s) | `gpu_extract[2]` |
| (그 뒤, CPU) | `gt_extra` → P1–P6 (record 평균 chain) → verifier 가 집계 단위 deviation 발견 → scene 평균으로 전 단계 재실행 → T51b 추출 · P1–P6 · (c) | `runs` |

### 2. Protocol (사전 고정)

후보 Ω = {query × layer L2–17 × head 0–7}. T51 의 query = action token (denoise 0, agg ∈ {last, mean, first}) → 384 cell. 모든 threshold (0.01 · 0.5 · 0.4 · 0.03 · 15 % · 상위 16 · k ∈ {1, 2, 3}) 는 lead 가 결과를 보기 전에 정했다. T47b 에서 관찰한 sink (L14H5) 와 넓은 map (L4H1 · L2H0) 을 걸러내는 최소 조건이다. **"이 값들로 L8H2 가 제외되는지는 결과에서 보고만 한다. 값을 다시 맞추지 않는다"** 가 task 에 적혀 있었다.

| 단계 | 정의 | label |
|---|---|---|
| **P1 filter** | calibration 전체에서 cell × camera 마다 계산. 제외: `low_mass` (image mass median < 0.01) · `fixed_patch_sink` (최빈 argmax patch 비율 > 0.5 이고 top-1 patch 몫 > 0.4) · `broad_map` (top-1 몫 < 0.03 이고 최대 component > grid 15 %). 한 camera 라도 해당하면 cell 제외 | 아니오 |
| **P2 calibration 점수** | cell 마다 primary **3D `identity_correct` 비율** (T47b 정의, raw · σ=0, scene mean). 3D 는 비싸서 **2D lift 상위 16 + forced (L8H2 + T47 oracle 3)** 에만 | 예 (GT 또는 self label) |
| **P3 안정성** | scene bootstrap B = 1000: cell 별 P(rank1) · P(top3) · 점수 95 % CI. 기준 P(rank1) ≥ 0.5 | 예 |
| **P4 k 선택** | k ∈ {1, 2, 3} (상위 cell raw 합) 을 leave-one-scene-out nested CV 로. 차이가 CI 안이면 더 작은 k | 예 |
| **P5 held-out** | calibration 에 쓰지 않은 record: same-scene `headfix` · `ep1800`, new-scene `ep1850` · `ep1900` (T47 과 같은 split). 항상 GT `identity_correct` 로 채점 | 예 |
| **P6 ablation** | 같은 held-out: 논문 label-free top-3 (σ=0 · σ=1) · head 평균 · 무작위 cell 20 개 (P1 통과 cell, seed 5106) · GT-oracle (T47 top-3) · 서빙 L8H2 · restricted oracle (후보 중 held-out 최고 단일 cell, 상한 참고) | 예 |

- **"단일 head 고정 여부" 는 P4 와 P3 의 안정성으로 정해진다** — 고정이 데이터의 결론이 되게 한다는 것이 task 의 원칙이다.
- 누수 방지: P4 의 모든 training set 마다 그 set 만으로 2D → 3D 선별을 다시 한다 (scene fold 5 + 안쪽 10 = 15 개, `candidates_3d` 는 그 합집합). 구현 단위 시험 (63 test) 이 test scene 점수를 바꿔도 fold 선택이 불변임을 확인했다 (`pytest_t51` 63 passed, 24.0 s).

### 3. calibration 확장과 GPU 추출 (RB-Y1)

| record | scene | frame | target | VLA 학습 episode | 비고 |
|---|---|---:|---|---|---|
| `long` | scene1800 | 50 | — | 아니오 | T47 selection record |
| `t2` | scene1800 | 75 | — | 아니오 | `long` 과 같은 scene |
| `ep1807` | scene1807 | 75 | — | 아니오 | |
| **`ep0`** | scene0 | 113 | orange | **예** | 추가. target ≥ 200 px 인 frame: cam_high 97 · left wrist 113 · right wrist 85 |
| **`ep500`** | scene500 | 113 | banana | **예** | 추가. 24 · 25 · 111 |
| **`ep1200`** | scene1200 | 113 | pear | **예** | 추가. 94 · 23 · 113 |
| held-out `headfix` · `ep1800` · `ep1850` · `ep1900` | scene1800 · scene1800 · scene1850 · scene1900 | 각 15 | | 아니오 | P5 · P6 |

- 추가 3 run 의 attention 은 T47 asset 을 만든 producer (`pi05_attention`) 를 수정 없이 subprocess 로 돌려 같은 축 · 같은 checkpoint 에서 뽑았다. 이 환경이 T47 asset 을 재현하는지 `ep1807` 첫 4 frame 으로 확인했다: **bit 동일은 아니다** (`bitwise_equal` false, max |Δ| 0.003906, mean |Δ| 1e-06, argmax patch 일치 0.995885 (d0 0.995563), 서빙 cell 의 Pearson r 0.999998). float16 저장과 GPU reduction 순서가 원인일 수 있으나 분리하지 않았다 (미측정 해석).
- GT cache (CPU, MuJoCo osmesa) 는 T47 의 `extract_gt` 를 그대로 불러 만들었다. 3 run 의 render 대 기록 RGB: `rgb_mad_max` 0.009619 · 0.015745 · 0.195871 (`ep0` · `ep500` · `ep1200`), `rgb_frac_gt16_max` 0.00012 · 0.000199 · 0.003647.
- GPU: 시작 전 compute process 없음 (memory 0 MiB), nvidia-smi peak 8802 MiB (RB-Y1), 8804 MiB (LIBERO).
- T51 실행 중 다른 세션이 T46c 관련 파일 5 개 (`config.py` · `pipeline.py` · `types.py` · `wire.py` · `serve_safe.py`) 를 바꿨다. verifier 가 같은 입력으로 재측정해 **108 / 108 row 가 같음** (`ag3s_ms` 제외) 을 확인했다.
- (`numbers.check` 는 GPU 추출 전 snapshot 이다: `all_ok` false, "no attention npz". 이후 `check` 는 rc 0.)

### 4. 집계 단위 deviation — 실행 중 발견 (정직한 기록 1)

**사전 고정.** P2 의 점 추정은 **scene mean** (같은 scene 의 run 을 먼저 평균 → scene 끼리 평균). `long` · `t2` 는 같은 scene 이라 한 scene 으로 센다.

**발견.** T51 verifier 가 보고했다: 처음 구현은 P2 · P3 · P4 의 순위를 **record 6 개의 가중 없는 평균** 으로 매겼다. 그래서 scene1800 (`long` · `t2`) 이 6 개 중 2 개의 가중을 받았다. P3 의 scene bootstrap 도 resample 안에서는 record 평균이었다. 이 구현의 GT 1 위는 `mean:L12H4`, 사전 고정 (scene mean) 의 1 위는 `last:L12H4` 로 달랐다.

**수정.** lead 가 `--aggregate scene` (P2 · P3 · P4 · P5 · P6 전부 scene mean, held-out 은 held-out scene mean) 을 요청했다. 구현은 기본값 `record` 를 유지해 (진행 중이던 verifier stage 를 깨지 않기 위해) `--aggregate scene` 을 더했고, 출력은 `__scene` 이름으로 따로 썼다. **verifier 가 scene 단위로 P2 · P3 · P4 · P5 · P6 을 전부 다시 돌렸고, 이 절의 primary 는 그 결과다.** 3D row cache 는 집계와 무관해 공유했다.

**검증.** scene mean 의 P2 · P3 를 코드 출력과 verifier 의 독립 스크립트로 각각 구해 대조했다: **순위 동일, 최대 절대차 0.0** (P2 · P(rank1) · P(top3) · CI), 서로 다른 resample 126 개 동일. 재실행 중 감시한 파일의 md5 는 변하지 않았다. scene mean 의 2D → 3D 후보 선별 (GT · self top-16) 은 record 평균과 같았다 (`gt_top16_identical` · `self_top16_identical` true).

**두 집계가 가른 것 (GT label).**

| 단계 | record-mean chain (사전 고정과 다른 방식) | **scene mean (primary)** |
|---|---|---|
| P2 3D 1 위 | `mean:L12H4` 0.6819 (`last:L12H4` 0.6766 이 2 위) | **`last:L12H4` 0.6913** (`mean:L12H4` 0.6909 가 2 위) |
| P3 1 위 확률 | `mean:L12H4` P(rank1) 0.388 · P(top3) 0.826 (record 단위 resample 은 0.372, 6 단위 · 462 개) | `last:L12H4` P(rank1) **0.429** · P(top3) 0.879 |
| P4 CV 평균 k = 1 / 2 / 3, 선택 | 0.586 / 0.596 / 0.537, k\* = 1 `mean:L12H4` (best k = 2) | 0.586 / 0.596 / 0.647, k\* = 1 `last:L12H4` (best k = 3) |
| P5 held-out | record mean 0.900 (선택 · L8H2 · `last:L12H3` 모두) | held-out scene mean 0.911 (record mean 0.900) |

- 두 집계의 선택은 같은 head (L12H4) 의 `mean` 과 `last` aggregation 이고 점수 차는 작다 (scene mean 기준 0.6913 대 0.6909). 그러나 **두 chain 어느 쪽도 P(rank1) ≥ 0.5 를 넘지 못한다.**
- P4 의 15 fold 에서 record 평균과 scene 평균의 fold 순위를 비교하면 top-3 의 순서가 다른 fold 가 8 개, top-1 이 다른 fold 가 3 개다.
- 같은 이름에 `__scene` 이 붙지 않은 figure ([`t51-bootstrap-rank1.png`](figures/t51/t51-bootstrap-rank1.png) · [`t51-ablation-heldout.png`](figures/t51/t51-ablation-heldout.png) · [`t51-protocol-table.png`](figures/t51/t51-protocol-table.png)) 는 **record-mean chain 의 증거로 보존**했고 primary 가 아니다. primary 는 `__scene` 이 붙은 아래 figure 들이다.

### 5. P1 filter — 무엇이 걸러졌나 (label 없음)

calibration 6 record 539 frame 전부, cell × camera.

| query | eligible | 제외 (한 camera 라도): `low_mass` | `fixed_patch_sink` | `broad_map` |
|---|---:|---:|---:|---:|
| last | 81 | 37 | 3 | 8 |
| mean | 87 | 29 | 3 | 11 |
| first | 74 | 43 | 10 | 8 |
| **합 (384 cell)** | **242** | 사유 집합별 cell 수: `low_mass` 만 99 · `broad_map` 만 26 · `fixed_patch_sink` + `low_mass` 9 · `fixed_patch_sink` 만 7 · `broad_map` + `low_mass` 1 | | |

![P1 filter 후 2D heatmap](figures/t51/t51-filter-2d-heatmap.png)

[`t51-filter-2d-heatmap.png`](figures/t51/t51-filter-2d-heatmap.png) ([sidecar](figures/t51/t51-filter-2d-heatmap.json)) — layer × head, query · camera 별 2D lift (GT label, scene mean of record means). × = P1 제외, 네모 = 전역 2D 상위 16.

**표시한 cell 의 판정.**

| cell | 판정 | 이유 · 통계 |
|---|---|---|
| **`last:L8H2` (서빙)** | **제외** | cam_high `low_mass`: mass median **0.006293** (< 0.01). left wrist 0.015043, right wrist 0.081629 는 통과. 다른 두 통계는 규칙 안 (modal_fraction 0.165 – 0.315, top-1 몫 0.165 – 0.225, 최대 component 7.8 – 9.3 %) |
| T47 label-free top-3 `last:L4H1` · `last:L2H0` | 제외 | 3 camera 모두 `broad_map` (top-1 몫 0.012 – 0.025, 최대 component 15.8 – 23.5 %) |
| `last:L14H5` | 제외 | cam_high `fixed_patch_sink` (modal_fraction 0.931354, top-1 몫 0.583195) |
| T47 oracle top-3 `last:L8H3` · `last:L10H7` · `last:L12H4` | 통과 | |

- **L8H2 의 제외는 사전 고정 규칙의 출력이다.** 2D lift 로는 L8H2 가 통과한 1 위 `last:L12H3` 보다 높다 (record 평균 2D lift 27.95 대 19.83). 즉 이 규칙은 "image 질량이 작은 head" 를 거르는데 그 head 가 target 위에 몰려 있다 (미측정 해석: image 질량의 절대량이 작아도 그 질량이 target 에 몰릴 수 있다). threshold 는 바꾸지 않았다.
- 제외된 L8H2 는 **forced cell 이라 3D 점수만 보고하고 순위 경쟁에는 넣지 않았다** (§6).

### 6. P2 calibration 점수 — 2D 와 3D

**2D lift (GT label, scene mean 선별).** 통과한 cell 의 2D lift 1 위는 `last:L12H3` (scene mean 15.22). 2D → 3D 후보는 GT 20 개, self 23 개. 선별 집합은 record 평균과 같다.

**3D `identity_correct` (GT label, scene mean, primary).** 표는 상위 cell 의 scene mean 과 scene 별 값이다.

| cell | scene mean | scene1800 (`long`+`t2`) | scene1807 | scene0\* | scene500\* | scene1200\* |
|---|---:|---:|---:|---:|---:|---:|
| **`last:L12H4`** | **0.6913** | 0.6033 | 0.8267 | 0.8850 | 0.7788 | 0.3628 |
| `mean:L12H4` | 0.6909 | 0.6367 | 0.8267 | 0.8850 | 0.7434 | 0.3628 |
| `last:L10H7` | 0.6069 | 0.6167 | 0.4000 | 0.8850 | 0.7611 | 0.3717 |
| `mean:L10H7` | 0.6055 | 0.6500 | 0.3333 | 0.8850 | 0.7965 | 0.3628 |
| `last:L8H3` | 0.5905 | 0.5833 | 0.3867 | 0.8761 | 0.7522 | 0.3540 |
| `last:L12H3` (2D 1 위) | 0.5772 | 0.6500 | 0.2800 | 0.8850 | 0.6991 | 0.3717 |
| `mean:L12H3` | 0.5683 | 0.6500 | 0.2800 | 0.8496 | 0.6903 | 0.3717 |
| `last:L8H2` (forced · P1 제외 · 순위 밖) | 0.5858 | — | — | — | — | — |

(\* = VLA 학습 episode.) 순위 밖 L8H2 의 0.5858 은 순위 5 위 `last:L8H3` (0.5905) 와 6 위 `last:L12H3` (0.5772) 사이다 (산출 비교).

- **2D lift 1 위 (`last:L12H3`) 는 3D 에서 6 위다.** 2D 와 3D 의 순위가 다르다. 이것이 primary 를 3D 로 정한 이유의 직접 증거이고, 2D 만 있는 플랫폼 (LIBERO, §11) 의 선택은 이 한계를 가진다.
- scene 별 값이 크게 흩어진다: `last:L12H4` 는 scene0 0.8850 에서 scene1200 0.3628 까지. scene1200 (pear) 은 상위 cell 대부분이 0.35 – 0.37 이다. 이 scene 만 낮은 이유는 측정하지 않았다 (이 값이 P3 의 불안정을 만든다, §7).

### 7. P3 안정성 — 사전 고정 기준 미충족 (정직한 기록 2)

![P3 bootstrap 1 위 확률 (scene mean)](figures/t51/t51-bootstrap-rank1__scene.png)

[`t51-bootstrap-rank1__scene.png`](figures/t51/t51-bootstrap-rank1__scene.png) ([sidecar](figures/t51/t51-bootstrap-rank1__scene.json)) — scene bootstrap (B = 1000, seed 5103, 5 scene, 서로 다른 resample 126 개) 의 cell 별 P(rank1) (파랑) · P(top3) (주황) 과 primary 점수 95 % CI. 점선 = 0.5 안정성 기준. 위 = GT label, 아래 = self label.

| cell (GT, scene mean) | P(rank1) | exact P(rank1) | P(top3) | 점수 95 % CI |
|---|---:|---:|---:|---|
| **`last:L12H4`** | **0.429** | 0.4355 | 0.879 | [0.515, 0.840] |
| `mean:L12H4` | 0.256 | 0.2368 | 0.794 | [0.522, 0.833] |
| `mean:L10H7` | 0.232 | 0.2483 | 0.459 | [0.403, 0.797] |
| `last:L12H3` | 0.080 | 0.0774 | 0.127 | [0.372, 0.764] |
| `last:L10H7` | 0.002 | 0.0016 | 0.350 | [0.426, 0.782] |

- **full-sample 1 위의 P(rank1) = 0.429 < 0.5 → `stable_single_cell` = false.** rank-1 tie rate 0.099 (동점이 순위를 정한 resample 비율). 1 위 확률이 `last:L12H4` · `mean:L12H4` · `mean:L10H7` 세 cell 에 나뉜다.
- **사전 고정 되돌아올 지점의 적용: "선택이 불안정하면 단일 head 고정을 주장하지 않는다. k > 1 또는 명사 query 를 Ω 에 넣어 다시 돌린다."** → 단일 head 고정은 주장하지 않는다. k > 1 은 §8, 명사 query 는 §12.
- 읽을 때의 한계: scene 이 5 개이고 그중 3 개가 VLA 학습 episode 라서 이 값은 "이 5 scene 을 다시 뽑았을 때" 의 변동이다. 새 scene 의 변동은 아니다 (§13).
- (미측정 해석) 1 위 cell 들이 같은 head L12H4 의 aggregation 이라 head 단위로는 `last` · `mean` 합 P(rank1) 가 0.685 로 높지만 (산출 0.429 + 0.256), 이 합은 사전 고정 기준이 아니다.

### 8. P4 k 선택

| k | CV 평균 (scene 5 fold) | 95 % CI |
|---|---:|---|
| 1 | 0.586 | [0.405, 0.772] |
| 2 | 0.596 | [0.412, 0.786] |
| 3 | **0.647** (best) | [0.481, 0.812] |

- best = k 3. 단순성 규칙: k = 1 대 best 의 쌍 차이 0.061, 95 % CI [−0.119, 0.296] 이 0 을 포함 → **k\* = 1, 선택 `last:L12H4`.** nested CV 평균 0.586, 안쪽 fold 의 k 는 [1, 1, 1, 1, 2] (scene1800 · 1807 · 0 · 500 · 1200 순).
- **읽는 법.** k = 1 은 "k = 3 이 더 낫지 않다" 는 증거가 아니라 **"차이가 5 scene 으로는 구별되지 않아 단순한 쪽을 택하는" 규칙의 출력** 이다. 코드가 내는 `single_head_fixed: true` 는 k\* = 1 이라는 뜻일 뿐, §7 의 안정성 기준을 충족했다는 뜻이 아니다.
- self label (§10): k 별 CV 평균이 모두 1.0 → k\* = 1, `last:L12H3`.

### 9. P5 held-out · P6 ablation (정직한 기록 3 포함)

![P5 · P6 held-out (scene mean)](figures/t51/t51-ablation-heldout__scene.png)

[`t51-ablation-heldout__scene.png`](figures/t51/t51-ablation-heldout__scene.png) ([sidecar](figures/t51/t51-ablation-heldout__scene.json)) — 왼쪽: 무작위 eligible cell 20 개 (seed 5106) 의 held-out GT `identity_correct` 분포 위에 이름 붙은 선택들. 오른쪽: held-out scene 별 값. held-out = 4 record · 3 scene · record 당 15 frame.

| source | held-out scene mean | scene1800 (`headfix`+`ep1800`) | scene1850 | scene1900 | record mean |
|---|---:|---:|---:|---:|---:|
| **선택 GT (k\* = 1) `last:L12H4`** | **0.911** | 0.867 | 0.867 | 1.000 | 0.900 |
| 선택 self label `last:L12H3` | 0.911 | 0.867 | 0.867 | 1.000 | 0.900 |
| **서빙 `last:L8H2`** | **0.911** | 0.867 | 0.867 | 1.000 | 0.900 |
| T47 GT-oracle top-3 합 | 0.911 | | | | 0.900 |
| restricted oracle (`mean:L8H2`, held-out 최고) | 0.911 | | | | |
| head 평균 (last, L2–17) | 0.867 | | | | 0.867 |
| 논문 label-free top-3, σ=0 · σ=1 | 0.000 · 0.000 | | | | 0.000 · 0.000 |
| 무작위 cell 20 개 | median 0.756 (p05 0.022, p95 0.912) | | | | median 0.733 |

- record 별 (세 선택 공통): `headfix` 1.000 · `ep1850` 0.867 · `ep1900` 1.000 · `ep1800` 0.733.
- **세 선택은 frame 단위로도 같다.** 4 record 모두에서 모든 frame (15 / 15) 의 identity 판정이 세 source 에서 일치한다. 서빙 L8H2 의 row 는 T47b σ=0 의 `l8h2` row 와 4 record 모두 15 / 15 같다 (T51 replay loop 가 T47b 와 갈라지지 않았음).
- 선택이 무작위 cell 20 개 중 `share_at_or_above_selected` 0.10 (2 개, 0.9111 · 0.9222) 만 같거나 높다 — 무작위에서 좋은 cell 이 거의 없지는 않다. 논문 top-3 은 σ 와 무관하게 0.000.

**L8H2 는 P1 에서 제외됐는데 held-out 에서 선택과 같다 (정직한 기록 3).** 사전 고정 규칙은 L8H2 를 low_mass 로 제외했고, 그 cell 은 calibration 3D 에서 순위 5 · 6 위 사이 (0.5858), held-out 에서는 선택과 모든 frame 에서 동률이다. 규칙이 그 cell 을 거른 것이 틀렸다고도 맞았다고도 이 측정은 말하지 못한다 — held-out 이 세 cell 을 가르지 못하기 때문이다 (재측정 대기 #19). **threshold 를 다시 맞추지 않았고 그 선택을 제안하지도 않는다.**

### 10. 자기 label (grasp-event pseudo label) — 실물 적용 가능성의 sim 검증 (정직한 기록 4)

![자기 label 투영 (실제 scene)](figures/t51/t51-selflabel-scene.png)

[`t51-selflabel-scene.png`](figures/t51/t51-selflabel-scene.png) ([sidecar](figures/t51/t51-selflabel-scene.json)) — 6 record (행: `t2` · `ep1807` · `ep0`\* · `ep500`\* · `ep1200`\* · `ep1850`) 의 grasp event 에서 구한 fingertip pad 중점 (붉은 점 + 반지름 disc) 을 이전 frame 의 camera 로 투영해 MuJoCo GT segmentation 위에 겹쳤다. 열: cam_high 첫 frame (전체) · 같은 frame (80×60 확대) · grasp 직전 frame (확대) · grasp 손의 wrist camera. 검은 + = GT target centroid, 색 = GT class (파랑 target, 주황 destination, 초록 other_object, 노랑 robot_gripper).

**event 와 투영 오차 (GT 점검, 6 record).**

| record | 손 · 시각 | label frame | r_obj (m) | 쥐기 전 target 이동 max · 1 cm 초과 frame | cam_high proj 오차 median (px) · hit_patch | 손 wrist camera 오차 median (px) |
|---|---|---:|---:|---|---|---:|
| `t2` | 왼 · t = 120 | 15 | 0.0314 | **1.11 cm · 15 / 15** | 4.65 · 0.933 | 19.97 |
| `ep1807` | 왼 · t = 152 | 19 | 0.0309 | **1.24 cm · 19 / 19** | 2.31 · 0.895 | 9.28 |
| `ep1850` (held-out) | 왼 · t = 88 | 11 | 0.0222 | 0.65 cm · 0 | 5.11 · 0.818 | 11.75 |
| `ep0`\* | 왼 · t = 128 | 16 | 0.0297 | 0.73 cm · 0 | 4.56 · 0.813 | 12.52 |
| `ep500`\* | 오른 · t = 112 | 14 | 0.0236 | 0.18 cm · 0 | 3.66 · 1.000 | 21.56 |
| `ep1200`\* | 오른 · t = 296 | 37 | 0.0301 | **1.47 cm · 37 / 37** | 5.33 · 0.865 | 13.09 |

- **grasp event 가 검출되지 않은 record 가 10 개 중 4 개** (`long` · `headfix` · `ep1800` · `ep1900`) 다. 그 record 는 self label 이 없다 (원인은 확인하지 않았다).
- **"쥐기 전 정지" 가정은 6 개 중 3 개 record 에서 깨진다.** target 이 label frame 과 grasp frame 사이에 1 cm 넘게 움직였다 (`t2` · `ep1807` · `ep1200`, 그 record 의 모든 label frame). 그런데 cam_high 의 투영 오차 median 은 2.3 – 5.3 px 로 가정이 깨진 record 와 안 깨진 record 가 비슷하다. wrist camera 의 오차 (9.3 – 21.6 px) 는 disc 가 크고 gripper 가 가려서 (그림의 4 열) cam_high 보다 크다 (원인은 미측정 해석).
- 그림의 `ep1850` grasp 직전 frame 에서 투영점이 gripper 위에 놓이고 target 은 가려진다 — 이 label 은 occlusion 을 모른다 (실물과 같은 조건).

**P2 · P3 · P4 에서 self label 로 고른 결과 (scene mean).**

| | GT label | self label |
|---|---|---|
| P2 1 위 | `last:L12H4` | `last:L12H3` |
| P2 상위 3 | `last:L12H4` · `mean:L12H4` · `last:L10H7` | `last:L12H3` · `mean:L12H3` · `last:L12H4` (겹침 1) |
| P3 1 위 확률 | `last:L12H4` 0.429, rank-1 tie rate 0.099 | `last:L12H3` 0.720, **rank-1 tie rate 1.0** |
| 교차 확률 | self label bootstrap 이 GT 1 위 `last:L12H4` 를 1 위로 뽑는 확률 **0.279** (self 1 위 자신은 0.720) | GT label bootstrap 이 self 1 위 `last:L12H3` 를 1 위로 뽑는 확률 **0.080** (GT 1 위 자신은 0.429) |
| P4 | k\* = 1 `last:L12H4` | k\* = 1 `last:L12H3` (같은 k, 다른 cell) |
| held-out GT `identity_correct` | 0.911 | 0.911 (record 별 차이 4 / 4 에서 0.0) |

- **포화 (정직한 기록 4).** `self_identity` 의 scene mean 이 상위 **9 cell 모두 1.0** (95 % CI [1.0, 1.0], 10 번째 `last:L8H3` 0.974) 이고 **rank-1 tie rate 가 1.0** 이다. 즉 모든 bootstrap resample 에서 1 위가 동점이라 **2D lift tie-break 가 순위를 정했다.** 실제로 self 의 상위 9 cell 은 2D lift 내림차순 (35.14, 29.70, 29.61, 24.78, 23.58, 23.32, 19.96, 13.17, 11.42) 이다. GT 의 2D lift 1 위도 `last:L12H3` 이다. 따라서 self 의 P(rank1) 0.720 은 **3D 지표가 cell 을 가른 확률이 아니라 2D tie-break 의 안정성** 이고, "self label 이 GT 와 같은 cell 을 골랐다" 는 성립하지 않았으며 (1 위가 다르다) "다른 cell 을 골랐다" 도 self label 의 결함이라고 읽지 않는다.
- 포화가 frame 부분집합 때문인지 보려고 같은 frame (self label 이 있는 grasp 이전 frame) 에서 GT `identity_correct` 를 재면 상위 3 GT cell (`mean:L12H4` · `last:L12H4` · `mean:L10H7`) 이 모두 1.0 이다 (record-mean chain 의 전체 frame 값은 0.682 · 0.677 · 0.613). **grasp 이전 frame 에서는 GT 로 재도 cell 을 가를 수 없다** (미측정 해석: 오분리가 grasp 이후 phase 에서 생긴다. phase 별로 쪼개 측정하지 않았다).
- **되돌아올 지점의 적용: "자기 label 의 결과가 GT 와 다르면 실물 적용에는 소량 사람 annotation 이 필요하다고 명시한다."** P2 1 위가 다르므로 명시한다. 단 위 포화 때문에 이것이 self label 의 부정확을 뜻하는지 지표의 해상도 한계를 뜻하는지는 가려지지 않는다.
- 실물 (`grasp_latch` 의 HELD) 의 label 은 시험하지 않았다. sim 의 `both_contact` 물리 predicate 만 썼다.

**annotation curve (GT, exploratory — 필요 frame 수 추정).** calibration frame (539 개 pool) 에서 n 개를 비복원으로 뽑아 (B = 1000) 순위를 다시 매긴다. 기준 상위 3 = `last:L12H4` · `mean:L12H4` · `last:L10H7`.

| n frame | P(same top-1) | P(top-1 이 기준 상위 3 안) |
|---:|---:|---:|
| 5 | 0.269 | 0.282 |
| 10 | 0.467 | 0.532 |
| 20 | 0.670 | 0.827 |
| 40 | 0.685 | 0.962 |
| 80 | 0.615 | 0.991 |
| 120 | 0.550 | 0.998 |
| 539 (전부) | 0.000 | 1.000 |

- frame 을 pool 해서 순위를 매기므로 (record · scene 가중이 달라짐) n = 539 에서 `P(same top-1)` 이 0.000 이다 (미측정 해석: scene mean 기준 1 위와 pool 평균 1 위가 달라서일 수 있다). **이 열은 해석에 쓰지 않는다.** `P(top-1 in reference)` 는 약 40 frame 에서 0.962 가 된다 — frame 이 같은 scene 에서 오는 상관을 무시한 값이라 scene 수가 아니라 frame 수의 효과를 과대평가할 수 있다 (미측정 해석).

### 11. LIBERO — 2D primary, task 단위 (정직한 기록 6)

**L0 조사.** KNOWS P0b 수집물 (`datasets/workspace/knows_p0b/`, 40 episode · 1,479 step) 에는 frame · state · GT segmentation · depth 는 있으나 **cell 별 attention 이 저장되어 있지 않았다.** 사용자 판정 (L1 승인, KNOWS 재현 확인 없이) 으로 저장 frame 에 `pi05_libero` attention pass 를 이 checkout 의 `.venv-openpi-live` 에서 한 번 돌렸다 (40 episode · 1,479 step, extract 240.8 s). **GT 가 agent view 의 segmentation 뿐이라 primary 는 2D lift (agent view) 이고 3D `identity_correct` 는 불가능하다.** KNOWS 재현 (`--init zeros`) 은 하지 않았다.

![LIBERO layer × head 2D lift](figures/t51/t51-libero-heatmap.png)

[`t51-libero-heatmap.png`](figures/t51/t51-libero-heatmap.png) ([sidecar](figures/t51/t51-libero-heatmap.json)) — LIBERO agent view 2D lift 의 layer × head (query 별). 집계 방식과 무관한 heatmap 이다.

- **P1.** eligible cell 287 개. 제외 사유 (cell × camera 수): agent view `low_mass` 81 · `fixed_patch_sink` 5 · `broad_map` 4, wrist `low_mass` 31 · `fixed_patch_sink` 10 · `broad_map` 4.
- **P2 · P3 (task 단위, 20 task × 2 episode).**

| 순위 (eligible) | cell | 2D lift (task mean) | 95 % CI | `peak_on_target` | KNOWS hit |
|---:|---|---:|---|---:|---:|
| 1 | **`mean:L12H3`** | **23.35** | [17.69, 28.50] | 0.756 | 0.711 |
| 2 | `last:L12H3` | 22.23 | [16.91, 26.98] | 0.751 | 0.760 |
| 3 | `first:L12H3` | 20.21 | [15.53, 24.68] | 0.614 | 0.648 |
| 4 | `first:L12H0` | 14.51 | [11.30, 17.58] | 0.552 | 0.670 |

- task bootstrap: `mean:L12H3` P(rank1) **1.0** (`stable_single_cell` true), `last:L12H3` · `first:L12H3` 도 P(top3) 1.0. task 평균과 episode 평균은 같다 (상위 20 의 순서 동일, 최대 절대차 7.1e-15).
- **LIBERO 에서도 L12H3 이 1 위** — RB-Y1 의 2D lift 1 위도 `last:L12H3` 이다. **head index 가 두 모델에서 같은 것은 관측이고** (미측정 해석: 같은 base weight 에서 fine-tune 된 영향일 수 있다 — 시험하지 않았다), 3D 로는 확인할 수 없다. §6 에서 RB-Y1 의 2D 1 위가 3D 에서 6 위였다는 점이 이 선택의 한계다.
- **L8H2 (LIBERO).** `last:L8H2` 는 P1 로 제외됐지만 (eligible 아님) 2D lift 15.83 으로 **전체 cell 중 4 위**, KNOWS hit 0.809 로 전체 cell 중 1 위다. `mean:L8H2` 는 eligible 이고 lift 12.20 (eligible 중 16 위). `mean:L12H4` 는 13.40 (eligible 중 10 위, KNOWS hit 0.739). RB-Y1 과 같이 **P1 이 제외한 L8H2 가 다른 지표에서는 상위** 에 있다. threshold 는 바꾸지 않았다.
- KNOWS 재현 확인을 하지 않았으므로 이 추출이 KNOWS 의 attention 과 같다는 점검은 없다 (사용자 판정). 이 점검 대신 RB-Y1 에서 같은 sampler 가 T47 asset 을 재현한 parity (§3, argmax 0.995885) 가 있다.

### 12. T51b — 명사 query 행을 같은 protocol 에 추가

**설정.** 명사 행 = T49 `target` query (prompt 의 target 명사 token 이 image 를 보는 attention, prefix 1 회, denoise 없음, raw σ=0). T47 record 의 명사 행은 T49 run 에서, 추가 3 run 은 새로 추출했다 (GPU lock). 같은 P1–P6 · 같은 threshold · scene 집계로 (a) 명사 128 cell, (b) action 384 + 명사 128 = 512 cell 을 돌린다. 3D row 는 T51 의 action row 를 읽기만 하고 명사 row 는 새로 replay 했다 (row 출처 집계: 명사 Ω 는 GT 108 · self 120 row 모두 `t51b`, 합친 Ω 는 GT 24 row 가 T51 cache · 108 row 가 `t51b`, self 30 row 가 T51 cache · 108 row 가 `t51b`).

**추출 검증.** `ep0` · `ep500` · `ep1200` 모두 113 / 113 frame 에서 명사가 정확히 1 회 매칭되어 오류 0, `check` 통과. `ep1807` 첫 4 frame 을 T49 의 `orig` 추출과 비교하면 **bit 동일** (max |Δ| 0.0, argmax 일치 1.0, Pearson r 1.0, t_step · token index 같음). prefix 시간 median 24.19 ms, GPU peak 7,527,781,120 byte. 시작 전 GPU compute process 없음.

**forced cell 구현 불일치 (정직한 기록 5).**

- lead 판정 (결과 전 고정): 명사 단독 Ω 의 3D 강제 포함 = **`noun_target:L8H6` · `L5H4` · `L4H2`** (T49 primary 명사 cell), 합친 Ω 에서는 T51 forced 에 이 3 개를 더한다.
- 구현은 T51 의 규칙 그대로 forced = [`last:L8H2` + T47 oracle 3] ∩ Ω 만 썼다. **명사 cell 3 개는 forced 목록에 넣지 않았다.** (a) 에서는 forced 가 Ω 안에 하나도 없어 실효 forced 가 비었다. (코드 규칙과 lead 판정의 차이가 verifier 에 의해 확인되었다.)
- verifier 가 영향을 점검했다: 세 cell 은 (a) · (b) × GT · self 네 경우 모두 **P1 eligible, 전역 2D 상위 16 안, 15 개 fold 모두의 순위 집합 안** 에 있다. 즉 forced 로 넣었어도 3D 후보 집합이 바뀌지 않는다 → **결과는 바뀌지 않았다.** 코드는 고치지 않았다.
- verifier 가 P2 3D · bootstrap 을 독립 재계산해 코드 값과 최대 절대차 1.1e-16 으로 일치시켰다 (명사 · 합친 Ω, GT).

![T51b bootstrap 1 위 확률 (query 별 색)](figures/t51/t51b-bootstrap-rank1-by-query.png)

[`t51b-bootstrap-rank1-by-query.png`](figures/t51/t51b-bootstrap-rank1-by-query.png) ([sidecar](figures/t51/t51b-bootstrap-rank1-by-query.json)) — 위 = GT label, 아래 = self label. 열: action Ω (T51) · (a) 명사 Ω · (b) action + 명사 Ω. 채움 = P(rank1), 윤곽 = P(top3), 파랑 = action query, 주황 = 명사 query. 점선 = 0.5.

**(a) 명사만 Ω.**

- **P1**: eligible 83 / 128. 제외 45 (cell 수, 한 camera 라도): `low_mass` 43, `fixed_patch_sink` 4, `broad_map` 0. lead 의 세 cell 은 모두 eligible.
- **P2 3D (GT, scene mean)**: 1 위 `noun_target:L10H7` 0.5957. 이어 `L8H0` · `L5H6` · `L6H5` 0.5931, `L6H4` 0.5913, `L4H2` 0.5911, `L8H6` 0.5891. **상위 10 cell 의 점수가 0.589 – 0.596 안에 있다 (차 0.007).** 비교: action 의 상위 점수는 0.691 (§6).
- **P3 (GT)**: `noun_target:L10H7` P(rank1) **0.687** (P(top3) 0.687, CI [0.406, 0.779], rank-1 tie rate 0.313, `stable_single_cell` true). 이어 `L4H2` 0.232 · `L8H6` 0.070 · `L5H4` 0.010. **P(rank1) 은 0.5 를 넘지만 점수가 거의 같은 cell 사이의 동점이 순위를 정하는 구조다** (tie rate 0.313, 미측정 해석: 어느 명사 cell 을 골라도 3D 에서는 구별되지 않는다).
- **P4 (GT)**: CV 평균 k = 1 · 2 · 3 = 0.5911 · 0.5911 · 0.5931 (best k = 3, 차 CI 가 0 을 포함) → k\* = 1 `noun_target:L10H7`.
- **self label**: 1 위 `noun_target:L8H6` P(rank1) 0.529 (rank-1 tie rate 1.0 — T51 과 같은 포화), `L5H4` 0.424, `L4H2` 0.047. k\* = 1 `noun_target:L8H6`.
- **P5 · P6 (held-out scene mean, GT 채점)**: 선택 GT `noun_target:L10H7` **0.867** (scene1800 0.733, scene1850 0.867, scene1900 1.000; record mean 0.833), 선택 self `noun_target:L8H6` 0.878 (scene1800 0.767; record mean 0.850), T49 primary 세 cell (`L8H6` · `L5H4` · `L4H2`) 각각 0.878, 서빙 `last:L8H2` 0.911. 무작위 명사 cell 20 개의 median 0.661, 선택 이상인 비율 0.40. restricted oracle `noun_target:L8H7` 0.878. (T49 절의 명사 수치는 k = 3 · σ = 1 합이라 이 단일 cell raw 값과 같은 source 가 아니다.)

**(b) action + 명사 합친 Ω.**

- **P1**: eligible 325 / 512 (last 81 · mean 87 · first 74 · 명사 83).
- **후보 구성.** 전역 2D lift 상위 16 은 **전부 명사 cell 이다 (명사 16 · action 0).** action cell 이 3D 후보로 들어간 것은 forced (`last:L8H3` · `last:L10H7` · `last:L12H4`, L8H2 는 P1 로 제외) 뿐이다. 3D 후보는 GT 22 개 · self 23 개다. 따라서 T51 의 action 단독 순위에서 2 · 3 위였던 `mean:L12H4` · `mean:L10H7` 와 2D 1 위 `last:L12H3` 는 이 순위 집합에 없다.
- **P2 3D (GT)**: 1 위 `last:L12H4` 0.6913, 이어 `last:L10H7` 0.6069, `noun_target:L10H7` 0.5957, `noun_target:L8H0` · `L5H6` · `L6H5` 0.5931 ….
- **P3 (GT)**: `last:L12H4` P(rank1) **0.753** (P(top3) 0.921, rank-1 tie rate 0.076, `stable_single_cell` true). **query family 별 P(rank1) 합 = action 0.918 · noun 0.082.** 이어 `last:L10H7` 0.165, `noun_target:L4H2` 0.065.
- **(미측정 해석) 0.429 (action 단독) 에서 0.753 으로 올라간 것은 같은 실험의 결과로 비교하기 어렵다.** action 단독에서 P(rank1) 을 나눠 가졌던 `mean:L12H4` (0.256) · `mean:L10H7` (0.232) 가 이 순위 집합에 없다. 합친 Ω 의 action 후보를 넓힌 실험은 하지 않았다.
- **P4 (GT)**: CV 평균 k = 1 · 2 · 3 = 0.606 · 0.588 · 0.466 → best = k 1, k\* = 1 `last:L12H4`.
- **self label**: 1 위 `noun_target:L8H6` P(rank1) 0.529 (family 합 noun 1.0 · action 0.0, tie rate 1.0). 고른 것이 명사 cell 인 것은 self 의 2D lift tie-break 가 명사 cell (lift 94.5 등) 을 앞세운 결과다 (자기 label 의 3D 점수는 상위 cell 모두 1.0).
- **P6 (GT 선택 `last:L12H4`, held-out scene mean 0.911)**: 무작위 20 개 (합친 Ω) median 0.639 (p05 0.032, p95 0.879), 선택 이상 비율 0.0, restricted oracle `last:L8H2` 0.911.

**(c) action 최선 대 명사 최선의 held-out 비교 (primary = scene 단위 paired 차이).**

![T51b held-out 비교 표](figures/t51/t51b-heldout-compare-table.png)

[`t51b-heldout-compare-table.png`](figures/t51/t51b-heldout-compare-table.png) ([sidecar](figures/t51/t51b-heldout-compare-table.json)) — 위: 선택별 held-out scene 별 `identity_correct` · scene mean · record mean. 아래: scene 단위 paired 차이 (seed 5107, B = 1000 과 3^3 = 27 resample 전수) 와 frame 단위 McNemar (참고).

| 선택 | cell | scene1800 | scene1850 | scene1900 | **scene mean** | record mean |
|---|---|---:|---:|---:|---:|---:|
| action best (GT) | `last:L12H4` | 0.867 | 0.867 | 1.000 | **0.911** | 0.900 |
| **명사 best (GT)** | `noun_target:L10H7` | **0.733** | 0.867 | 1.000 | **0.867** | 0.833 |
| 합친 best (GT) | `last:L12H4` | 0.867 | 0.867 | 1.000 | 0.911 | 0.900 |
| action best (self) | `last:L12H3` | 0.867 | 0.867 | 1.000 | 0.911 | 0.900 |
| 명사 best (self) = 합친 best (self) | `noun_target:L8H6` | 0.767 | 0.867 | 1.000 | 0.878 | 0.850 |
| 서빙 | `last:L8H2` | 0.867 | 0.867 | 1.000 | 0.911 | 0.900 |

| 쌍 (a − b) | scene 별 차 (1800 · 1850 · 1900) | scene mean 차 | 95 % CI (scene bootstrap MC) | 95 % CI (전수 27) | frame only_a / only_b | McNemar p (참고) |
|---|---|---:|---|---|---|---:|
| 명사 best − action best (GT) | −0.133 · 0 · 0 | **−0.044** | [−0.133, 0.000] | [−0.104, 0.000] | 0 / 4 | 0.125 |
| 합친 best − action best (GT) | 0 · 0 · 0 | 0.000 | [0, 0] | [0, 0] | 0 / 0 | 1 |
| 명사 best − 서빙 L8H2 | −0.133 · 0 · 0 | −0.044 | [−0.133, 0.000] | [−0.104, 0.000] | 0 / 4 | 0.125 |
| action best − 서빙 L8H2 | 0 · 0 · 0 | 0.000 | [0, 0] | [0, 0] | 0 / 0 | 1 |
| 명사 best − action best (self) | −0.100 · 0 · 0 | −0.033 | [−0.100, 0.000] | [−0.078, 0.000] | 0 / 3 | 0.25 |

- **차이는 scene1800 한 곳에서만 난다** (`headfix`: 명사 best 0.733 대 action best 1.000, `ep1800` 은 둘 다 0.733). scene1850 · scene1900 은 같다. scene 이 3 개라 bootstrap CI 의 상한이 0 에 걸리고 (MC resample 의 69.2 % 가 음수, 양수는 0 %), 전수 27 resample 에서 서로 다른 값은 4 개뿐이다.
- 따라서 **명사 query 가 action query 보다 3D target 을 더 잘 고른다는 증거는 없다.** 방향은 명사 ≤ action 이지만 유의하다고도 하지 않는다 (scene 3 개, frame 60 개). frame 단위 McNemar 는 frame 을 독립으로 세므로 참고다.
- 이 결과는 T49 의 3D 열세 (명사 query 의 `identity_correct` · purity · non-target 손실이 L8H2 보다 낮음) 와 같은 방향이고, 같은 source 가 아니라 크기를 직접 비교하지 않는다.
- (미측정 해석) T49 의 제안 (b) — 명사 query 로 identity · destination 을, L8H2 로 3D seed 를 얻는 결합 — 은 이 STEP 에서도 측정하지 않았다. T51b 의 "합친 Ω" 는 한 cell 을 고르는 것이지 두 query 를 결합하는 것이 아니다.

### 13. 읽을 때의 한계 · 측정하지 않은 것 · 되돌아올 지점 · 판정 대기

| 한계 | 내용 |
|---|---|
| **calibration scene 5 개** | 그중 3 개 (`ep0` · `ep500` · `ep1200`) 는 VLA 학습 episode 의 기록이다. P3 의 변동은 "이 5 scene 을 다시 뽑을 때" 의 변동이고 새 scene 의 일반화가 아니다. 서로 다른 resample 이 126 개뿐이다 |
| **held-out 3 scene · 60 frame** | 세 선택이 frame 단위로 동률이라 변별력이 없다 (#19). held-out 은 물체 종류 3 (apple · banana · pear) 정도로 좁다 |
| **3D 는 legacy ESDF offline replay** | subtask gate off, grasp latch 없음. cuRobo ESDF · closed-loop 효과는 측정하지 않았다 (#24) |
| **self label 은 sim 만** | `both_contact` 물리 predicate. 포화 · 쥐기 전 이동 (6 중 3) · event 없는 record (10 중 4) (#20) |
| **LIBERO 는 2D 만** | 3D · wrist · KNOWS 재현 없음 (#22) |
| **T51b 의 후보 구성** | 합친 Ω 의 2D 상위 16 이 모두 명사. action 후보는 forced 3 개 (#23) |
| **집계 단위 deviation** | 수정되었고 scene 로 재측정했다. record-mean chain 은 보고만 한다 (§4) |

**측정하지 않은 것 (`not_measured`).** T51: cuRobo 회귀 기준선 (GPU 미사용) · LIBERO KNOWS 재현 (`--init zeros`, protocol 밖) · 명사 행은 T51b 로 분리. T51b: T51b 용 MuJoCo scene figure (T51 의 scene figure 가 같은 held-out record 를 다룬다) · LIBERO 의 명사 행 (범위 밖) · record-aggregate (`__scene` 이 아닌) T51b 출력 · cuRobo 기준선.

**되돌아올 지점 (task 가 미리 정한 전환 신호의 적용).**

| 신호 | 결과 | 취한 행동 |
|---|---|---|
| 선택이 불안정 (1 위 확률 < 0.5) | 해당 (0.429) | 단일 head 고정을 주장하지 않는다. k > 1 은 P4 에서 CV 최고가 k = 3 (0.647) 이나 차 CI 가 0 을 포함해 단순성 규칙으로 k = 1. 명사 query 를 Ω 에 넣은 T51b 의 합친 Ω 는 P(rank1) 0.753 이나 후보 구성에 의존 (§12) |
| self label 결과가 GT 와 다름 | 해당 (P2 1 위 다름) | 실물 적용에는 소량 사람 annotation 이 필요하다고 명시한다. 필요 frame 수는 annotation curve (exploratory, 40 frame 에서 P(top-1 in reference) 0.962). 단 self 지표 포화 때문에 이 결론의 강도는 약하다 (§10) |
| LIBERO 수집물 없음 | 해당 (attention 없음) | 사용자 L1 승인 후 `pi05_libero` pass 수행 (§11) |
| 서빙 반영 | 하지 않음 | 별도 판정 |

**판정 대기 (제안, 미측정 해석 포함).**

1. **서빙 cell 을 어떻게 할 것인가.** 이 STEP 은 L8H2 를 유지한다. P1 이 L8H2 를 제외했지만 held-out 에서 `last:L12H4` · `last:L12H3` 와 frame 단위로 동률이어서, 이 측정만으로 교체나 유지의 근거는 확정되지 않는다.
2. **논문에서 이 protocol 을 어떻게 기술할 것인가 (제안).** 근거가 있는 주장은 "사전 고정한 절차로 cell 을 고르고 선택의 안정성을 수치로 보고한다" 까지다. "단일 head 가 최적이다" · "self label 이 GT 를 대체한다" · "명사 query 가 낫다" 는 이 측정이 받치지 않는다.
3. **후속 측정 (제안).** (i) 학습에 쓰이지 않은 scene 을 calibration 에 더해 P3 를 다시 (#18), (ii) 더 큰 held-out (#19), (iii) self label 을 phase 별 · 실물 `grasp_latch` 로 (#20), (iv) 합친 Ω 에서 action 후보를 넓힌 재실행 (#23).

**산출물.**

- **측정 (이 절이 인용)**: [`handoff/T51.verify.json`](handoff/T51.verify.json) (`numbers.scene_mean_primary` 가 primary) · [`handoff/T51b.verify.json`](handoff/T51b.verify.json). 원자료 run: `/mnt/dev/AG3S_VLA-storage/results/development/20261012T180318+0900_28b981600bf7_852011fdcae0/verify/{T51,T51b}/`.
- **figure** (각각 `.json` sidecar 있음, 위에서 링크): primary — [`t51-bootstrap-rank1__scene`](figures/t51/t51-bootstrap-rank1__scene.png) · [`t51-ablation-heldout__scene`](figures/t51/t51-ablation-heldout__scene.png) · [`t51-protocol-table__scene`](figures/t51/t51-protocol-table__scene.png) · [`t51-filter-2d-heatmap`](figures/t51/t51-filter-2d-heatmap.png) · [`t51-selflabel-scene`](figures/t51/t51-selflabel-scene.png) · [`t51-libero-heatmap`](figures/t51/t51-libero-heatmap.png) · [`t51b-bootstrap-rank1-by-query`](figures/t51/t51b-bootstrap-rank1-by-query.png) · [`t51b-heldout-compare-table`](figures/t51/t51b-heldout-compare-table.png). record-mean chain (deviation 증거로 보존) — [`t51-bootstrap-rank1`](figures/t51/t51-bootstrap-rank1.png) · [`t51-ablation-heldout`](figures/t51/t51-ablation-heldout.png) · [`t51-protocol-table`](figures/t51/t51-protocol-table.png).

![T51 protocol 단계별 결과 요약 (scene mean)](figures/t51/t51-protocol-table__scene.png)

[`t51-protocol-table__scene.png`](figures/t51/t51-protocol-table__scene.png) ([sidecar](figures/t51/t51-protocol-table__scene.json)) — P1–P6 · self label · annotation curve · LIBERO 의 단계별 결과 요약.

- **구현**: `AG3S/ag3s/experiments/studies/t51_{protocol,records,extract,selflabel,replay,study,libero}.py` · `t51b_{noun_extract,study}.py` · `tests/ag3s/test_t51*.py` · `test_t51b_*.py`. [`handoff/T51.impl.md`](handoff/T51.impl.md) · [`handoff/T51b.impl.md`](handoff/T51b.impl.md). commit 은 하지 않았다 (T51 · T51b 파일은 untracked).


---

## T46c-W1 · T46d · T46f — W1 의 재현 검증, DX-04 (link 별 target 예외) 선행 확인, LTE 검증 블록 계획 (2026-10-12 ~ 2026-10-13 KST)

> 이 절은 위 "T46c" 절의 **갱신**이다 (그 절의 W1 "진행 중, 결과 없음" 을 이 절 §2 가 닫는다). 이전 서술은 지우지 않고 취소선으로 표시했다.
> 시각은 pod 에 tzdata 가 없어 UTC 에 +9 h 를 더해 환산한 KST 다 (파일 mtime · `git log`; run 디렉터리 이름의 `+0900` 은 그대로 읽음).

**이 절이 답하는 물음 (목적).** develope_PLAN 은 P0-B ("대상 identity, 경계 추정, 접촉 처리 중 무엇이 실행을 개선하는가") 의 접촉 처리 쪽을 DX-04 (link-specific exemption: 허용 link 에만 target 접촉을 허용하는 LTE 가 target 을 전역으로 지우는 방식보다 필요한 접촉과 금지 접촉을 더 잘 구분하는가) 로 묻고, 결과를 **DT-05A** (세 field policy 별 pair 의 contact · success · stop 표) 와 **DF-04B** (같은 ESDF slice 에서 link 별 distance · gradient 와 실제 contact 그림) 에 싣는다. 이 주장 (DC-3: link-specific exemption 이 필요한 접촉과 금지 접촉을 더 잘 구분한다, DC-4: 전체 시스템이 task completion 과 forbidden contact 의 trade-off 를 개선한다) 이 성립하려면 먼저 (i) F1 채택 뒤 늘어난 close 이후 HOLD 가 비교를 오염시키지 않아야 하고 (T46c-W1), (ii) query set 에 허용 link 와 비허용 link 가 함께 있어야 한다 (T46d). 이 절은 (i) 의 재현 검증, (ii) 의 선행 확인 결과와 사용자 판정, 그리고 그 판정이 정한 다음 블록 (T46f) 의 **계획**을 기록한다.

**한 줄 결과 (before → after).**

- **W1 (T46c, 결과 있음).** before: F1 을 켠 C1a + F1 은 close 이후 HOLD 가 183 chunk (uncertified 130) 였고 그 원인은 placed `no_seed` 의 인증 공백이었다 (T46b). after: 인증 술어를 넓힌 W1 을 off 로 두면 3 arm 모두 기록과 **1350 / 1350** chunk 에서 같고, W1 을 켜면 close 이후 HOLD (total / uncertified) 가 C1a 107 / 37 → **70 / 0**, C1a + F1 183 / 130 → **54 / 1**, C1a + F1 + F2 122 / 118 → **6 / 2** 로 줄어든다 (T46b 의 what-if 와 같은 값). **per-chunk what-if 이고 closed-loop 효과는 아니다.** 사용자 판정으로 W1 은 모든 Ablation arm 에 켠다.
- **DX-04 선행 확인 (T46d, 결과 있음).** 현재 gripper scope (손바닥 2 + 손가락 4, 270 구) 에서는 non-authorized 행이 target 50 mm 안에 한 번도 오지 않아 LTE 와 global exclusion 이 **퇴화** 한다 (target 근처 non-authorized 행 0 / 349,920, LTE ≠ global chunk 0 / 162). `link_*_arm_5` 를 더하면 두 정책이 갈리는 chunk 가 생기지만 (C1a + F1, 162 chunk 중 89), 그 arm_5 구는 **실제 MJCF 접촉 geom 이 target 에서 100 mm cap 이상 떨어져 있는 18 / 18 run 에서도** target 50 mm 안에 놓인다. 기록 문구 (사용자 지정): **현재 gripper scope 에서는 LTE 와 global exclusion 의 비교가 퇴화하며, `link_*_arm_5` 를 추가했을 때 나타난 차이는 형상 근사 오류의 영향을 받는다.** "효과 없음" 이 아니다.
- **T46f (계획, 결과 없음).** 사용자 판정으로 link-specific 효과 주장을 줄이지 않고 **형상 보정 → 진단 장면 → oracle extent 3 조건 closed-loop** 으로 확장한다. LTE 의 우수성은 전제하지 않는다.

**이 절이 주장하지 않는 것.**

- **LTE 가 global exclusion 보다 낫다** 는 주장. T46d 는 비교가 성립하는 조건을 찾는 선행 확인이고 효과를 재지 않았다.
- **arm_5 에서 나타난 LTE ≠ global chunk 수가 LTE 의 효과다** 라는 주장. 그 차이는 형상 근사 오류의 영향을 받는다 (사용자 지정 문구).
- **LTE ≠ global chunk 수는 contribution 의 근거가 아니다.** 메커니즘 설명용 **보조 지표**일 뿐이다 (T46f §1, lead). 주 지표는 비허용 접촉 없는 작업 성공률이다.
- **정지 (HOLD · TO 실패) 가 늘어 접촉률이 낮아진 결과는 보호 효과가 아니다.** 보호 효과로 해석하지 않는다 (T46f §1).
- **W1 이 task success · completion time 을 개선한다** 는 주장. closed-loop 으로 재지 않았다.

**읽는 법과 출처 규약.**

- 수치는 [`T46c.verify.json`](handoff/T46c.verify.json) (W1) · [`T46d.verify.json`](handoff/T46d.verify.json) (DX-04 선행 확인) 의 `numbers` 에서 왔다. 이전 STEP 의 수치는 [`T46b.verify.json`](handoff/T46b.verify.json) · [`T46-CL3.verify.json`](handoff/T46-CL3.verify.json) 이고 그 자리에 적었다. 같은 파일 숫자의 덧셈 · 뺄셈은 "(산출)" 로 적었다. **T46f 는 계획이라 verify.json 이 없다**; 그 절의 서술은 전부 lead 의 [`T46f.task.md`](handoff/T46f.task.md) 이고 "(lead)" 로 표시했다.
- 판정의 출처: **(사용자 판정)** · **(lead)** · **(산출)** 은 위와 같다. 측정으로 확인하지 않은 해석은 "(미측정 해석)" 이다.
- 이 STEP 의 측정 입력은 모두 **읽기 전용** 이고 코드 수정은 없었다 (T46d). T46c 의 코드 변경은 implementer 의 W1 구현이다 ([`T46c.impl.md`](handoff/T46c.impl.md)).

### 1. 사용자 판정과 타임라인 (KST)

| # | 판정 · 사건 | 이 절에서 |
|---|---|---|
| 1 | **(사용자 판정, 2026-10-12, 추천안 채택)** W1 을 Ablation 의 **모든 arm 에 켠다** — 2026-10-04 판정 ("남은 `no_seed` HOLD 는 그대로 둔다") 을 이번에 바꾼다. 근거: 그 판정의 전환 신호 관측 (T46b), V11AB 에도 같은 HOLD. W1 의 기본값은 off 로 두고 실험 arm 에서 명시적으로 켠다 | §2 |
| 2 | **(사용자 판정, 2026-10-12, 추천안 채택)** DX-04 선행 확인의 비허용 link 범위는 develope_PLAN §0.10.1-1 대로 **최소 범위** (wrist 등) 부터 — T46d 에서 기록으로 고른다 | §3 |
| 3 | **(사용자 판정, 2026-10-12, 추천안 채택)** 표본은 **18 쌍 유지** (CL2 · CL3 와 같은 9 episode × 2 seed) | §4 |
| 4 | **(사용자 판정, 2026-10-12, 추천안 채택)** 순서는 develope_PLAN §11: DX-04 선행 확인 (T46d) → DX-01 identity / extent 분리 + visible-target oracle (T46e 이후) → contact handling → Ablation closed-loop | §4 |
| 5 | **(사용자 지정, 2026-10-13)** T46d 의 기록 문구: "현재 gripper scope 에서는 LTE 와 global exclusion 의 비교가 퇴화하며, `link_*_arm_5` 를 추가했을 때 나타난 차이는 형상 근사 오류의 영향을 받는다." ("효과 없음" 으로 기록하지 않는다) | §3 |
| 6 | **(사용자 판정, 2026-10-13)** A (link-specific 효과 주장 축소) 가 아니라 **B 를 확장해 진행** (형상 보정 → 진단 장면 → oracle extent 3 조건 closed-loop). **C (arm_5 근사 구 그대로) 는 제외.** LTE 의 우수성을 전제하지 않는다 | §4 |

| 시각 (KST) | 무엇 | 출처 |
|---|---|---|
| 2026-10-12 23:02 | T46c task 작성 (lead) | `T46c.task.md` |
| 23:32 | W1 구현 보고 `T46c.impl.md` (implementer) | mtime 14:32Z |
| 23:35:15 | V-T46c verifier run 시작 | run 디렉터리 `20261012T233515+0900_972f28056cd8_318f98f36f37` |
| 23:36 | 사용자 판정 (위 1 – 4) 과 T46d task 를 task 문서에 기록 | commit `39246257` 14:36:40Z |
| 23:39 | `t46c-table.png` 생성 | mtime 14:39Z |
| 23:39:20 | V-T46d verifier run 시작 | run 디렉터리 `20261012T233920+0900_392462577320_70c141f47dbe` |
| 23:40 | **`T46c.verify.json` 작성 (PASS)** | mtime 14:40Z |
| 23:41:40 | T46d 판정 기준 (`thresholds_fixed_before_measuring.json`) 을 측정 전에 고정 | `T46d.verify.json definitions.written_utc` 14:41:40Z. 그 전에는 legacy 기준선과 구 모델 확인만 돌았다 |
| 23:41:51 | W1 코드 · test · replay harness commit | commit `6dfaf7bb` 14:41:51Z |
| 23:42:37 | T46e task (visible-target oracle extent 구현 지시) commit | commit `15314ba3` 14:42:37Z |
| 2026-10-13 00:06 – 00:16 | `t46d-scene` · `t46d-graph` · `t46d-table` figure 생성 | mtime 15:06Z · 15:15Z · 15:16Z |
| 00:17 | **`T46d.verify.json` 작성** | mtime 15:17Z |
| 00:19:35 | T46d 결과 commit | commit `e7157636` 15:19:35Z |
| 01:23 – 01:24 | 사용자 지정 기록 문구 · 판정을 `T46d.task.md` 에 기록, **T46f task 작성** (lead) | `T46f.task.md` mtime 16:23Z, commit `ab9b3fba` 16:24:23Z |

### 2. T46c — W1 (placed `no_seed` 인증) 의 재현 검증 (V-T46c: PASS)

**무엇을 검증했나.** W1 (`--placed-no-seed-certify`, 기본 off) 은 인증 술어 `subtask_no_target` 에 한 갈래를 더한다: gate on ∧ label `home` ∧ `placed_seen` ∧ validity `valid` ∧ carve 없음 ∧ grounding `no_seed` 이면 인증한다 (구현: `AG3S/ag3s/types.py` 의 `subtask_no_target`, `AG3S/trajopt/safe_policy.py`, CLI `AG3S/trajopt/serve_safe.py`; 상세 [`T46c.impl.md`](handoff/T46c.impl.md)). 다른 조건 — 위반 행이 있음, `placed_seen` 없음, label `pick` / `place` — 은 그대로 HOLD 다. 인증은 "AG3S 가 장면을 다 설명했는가" 라는 질문만 바꾸고 field 는 그대로 모든 물체를 full clearance 로 잡는다. 검증 기준 (verifier, 측정 전):

| 기준 | 내용 | 결과 |
|---|---|---|
| 1 | 회귀 기준선 일치 (legacy · cuRobo) | PASS |
| 2 | off (W0) 가 CL3 기록의 reason · `geometry_certified` 와 chunk 단위로 같은가 (3 arm × 1350 chunk) | PASS |
| 3 | W1 이 T46b 의 what-if (execute 전환 37 / 129 / 116, close 이후 HOLD) 를 재현하는가 | PASS |
| 4 | 새 test · 기존 SUBTASK test 통과 | PASS |

**기준 1 — 회귀 기준선.**

| backend | start-violated | has_target | frame 0 clearance_before | 기대값과 |
|---|---|---|---|---|
| legacy (CPU, `.venv-ag3s`) | 14 / 15 | 15 / 15 | −29.031048 mm | 일치 |
| cuRobo (GPU, `.venv-openpi-live`) | 10 / 15 | 15 / 15 | −9.171877 mm | 일치 |

(출처: `numbers.criterion_1_baselines`, `baseline_check.matched = true`. 부기: legacy 의 final status 는 feasible 15, cuRobo 는 feasible 14 · violated 1 이고 verifier 는 이것을 기준선이 아니라고 적었다 (SQP wall-clock). legacy 실행 중 다른 session 때문에 machine load average 가 약 14 였다.)

**기준 2 · 3 — off bit-identical, W1 offline 재현 (before → after).** 입력은 CL3 의 기록 chunk (arm 당 1350 chunk, [`T46-CL3`](handoff/T46-CL3.verify.json) 의 18 쌍). **C1a · C1a + F1 · C1a + F1 + F2 의 세 arm 이 입력이다.** V11AB (C0) 는 CL2 의 run 이라 이 replay 에 들어 있지 않다 (재측정 대기 #27).

| 기록 arm | off (W0) 가 기록 reason 과 같음 | off (W0) 가 기록 `certified` 와 같음 | close 이후 HOLD, 기록 그대로 (total / uncertified) | **W1: execute 로 바뀌는 chunk** | **W1: close 이후 HOLD (total / uncertified)** |
|---|---|---|---|---|---|
| C1a | 1350 / 1350 | 1350 / 1350 | 107 / 37 | **37** | **70 / 0** |
| C1a + F1 | 1350 / 1350 | 1350 / 1350 | 183 / 130 | **129** | **54 / 1** |
| C1a + F1 + F2 | 1350 / 1350 | 1350 / 1350 | 122 / 118 | **116** | **6 / 2** |

(출처: `numbers.criterion_2_off_bit_identical`, `numbers.criterion_3_W1`. 기록 그대로의 값은 `W0_after_close_hold` 와 `recorded_after_close_hold` 가 같다. T46b 의 what-if 와 대조하면 전환 37 / 129 / 116 · HOLD 70 / 54 / 6 · uncertified 0 / 1 / 2 가 모두 같다.)

![T46c verifier 표: 회귀 기준선과 CL3 certification replay](figures/t46/t46c-table.png)

[`t46c-table.png`](figures/t46/t46c-table.png) (sidecar [`.json`](figures/t46/t46c-table.json)) — 위: 두 backend 의 기준선. 아래: arm 별 off fidelity · W1 전환 · close 이후 HOLD · 바뀐 칸.

- **바뀐 chunk 는 전부 한 칸이다:** `grounding = no_seed` · `phase = placed` · `label = home` · `placed_seen = True` 에서 C1a 37 / 37 · C1a + F1 129 / 129 · C1a + F1 + F2 116 / 116 chunk 이고, `no_seed` · placed · home 이 아닌 칸에서 인증이 바뀐 chunk 는 없다 (`conversions_by_cell`, `flipped`). 인증이 바뀌었는데 다른 사유로 HOLD 가 남은 chunk 와 인증이 바뀐 chunk 가 기록에서 실행이었던 경우는 세 arm 모두 0 이다 (`certified_but_still_hold`, `certified_recorded_execute`).
- **전환된 run (chunk 수):** C1a `ep54_s541` 37 · C1a + F1 `ep11_s111` 40 · `ep30_s301` 36 · `ep54_s541` 38 · `ep54_s542` 15 · C1a + F1 + F2 `ep11_s111` 40 · `ep30_s301` 36 · `ep54_s541` 38 · `ep54_s542` 2 (`conversion_runs`). 이 run 들은 T46b §4 에서 placed `no_seed` 가 episode 끝까지 이어졌거나 풀린 run 과 같다.
- **W1 후에 남는 uncertified 1 · 2 chunk** 는 T46b 가 `no_admissible` + label `pick` chunk 수 (C1a + F1 1, C1a + F1 + F2 2) 와 같다고 정리했다 (개수 일치로 읽음). T46c verify 는 그 chunk 의 칸을 따로 분해하지 않았다.

**기준 4 — test.** W1 test (`tests/trajopt/test_t46c_placed_no_seed_certify.py`) 는 passed 67 · failed 0 · skipped 0. 기존 SUBTASK suite 5 개 파일은 passed 316 · failed 0 · skipped 1 (`numbers.criterion_4_tests`). 구현자는 전체 pytest 도 돌렸으나 그 결과는 `T46c.impl.md` 에만 있고 verifier 는 재지 않았다 (`not_measured`, #27). 측정은 working tree (HEAD `972f2805` 에 미커밋 변경) 위에서 했고 verify.json 의 md5 (`config.py` · `types.py` · `safe_policy.py` · `certification_replay.py`) 는 이 로그를 쓰는 시점의 working tree 와 같다.

**before → after 요약 (C0 = V11AB · C1a · C1a + F1).**

| 지표 | V11AB (C0) | C1a | C1a + F1 | 출처 |
|---|---|---|---|---|
| close 이후 uncertified HOLD, 기록 그대로 (W0) (uncertified 는 placed phase 에서만 발생) | 73 chunk (CL2 의 uncertified HOLD 총합) | 37 | 130 | T46b §6 (C0), `T46c.verify.json` (C1a · F1) |
| **close 이후 HOLD (total / uncertified), W1 what-if** | **측정 안 됨** (replay 입력에 없음) | **70 / 0** | **54 / 1** | `numbers.criterion_3_W1` |

(C0 의 73 은 T46b 가 CL2 의 placed `no_seed` HOLD chunk 로 센 값이고, C0 의 close 이후 total 은 이 로그에 없다.)

**읽는 법 — W1 이 주장하는 것과 주장하지 않는 것.**

- 주장하는 것: 인증 술어를 `no_seed` 까지 넓히면 기록된 chunk 의 인증이 T46b 가 예측한 칸에서만, 예측한 수만큼 바뀐다. off 일 때는 기록과 chunk 단위로 같다.
- **주장하지 않는 것:** (1) closed-loop 에서 HOLD 가 풀린 뒤의 궤적 · home 복귀 시간 · safe success (#25). (2) "위반 행이 있으면 HOLD 를 유지한다" 의 **기록 chunk 시험** — CL3 의 `no_seed` chunk 는 전부 `uncertified` 하나로 멈춰 있어 이 갈래를 밟는 chunk 가 없다 (`not_measured`, #26). 이 안전 조건은 단위 test 가 맡는다. (3) "F1 이 close 이후 HOLD 를 늘린다 / 늘리지 않는다" 의 판정 — 원인은 T46b 의 결론 (인증 공백) 이고 W1 은 그 공백을 메운 것이다. F1 이 placed `no_seed` run 을 계통적으로 늘리는지는 여전히 모른다 (#15).
- **알려진 위험 (구현자, [`T46c.impl.md`](handoff/T46c.impl.md) "판정이 필요한 것" 2).** `no_seed` 는 `no_admissible` 보다 근거가 약하다. `no_admissible` 은 "cluster 를 보고 잡을 것이 없다고 판단했다" 이고, `no_seed` 는 "attention 이 지지면만 본다" 이며 지지면 밖 물체를 하나도 보지 않았다는 뜻이 아니다. 완화 근거: target 이 없어 carve 가 없고, field 는 모든 관측 물체를 full clearance 로 잡아 위반 행이 있으면 HOLD 이며, `validity` 가 `valid` 일 때만 열린다 (구현자의 서술이고 verifier 가 이 경로를 closed-loop 로 시험하지 않았다).
- **되돌아올 지점.** 코드 기본값은 off 로 둔다 (사용자 판정). W1 on 에서 새 위반 · 접촉이 home 경로에서 나타나는 closed-loop 측정이 나오면 인증 범위를 다시 판정한다.

### 3. T46d — DX-04 선행 확인: LTE 대 global exclusion 비교는 어디서 성립하는가

**무엇을 쟀나 (lead 의 task, verifier 측정, CPU offline, 코드 수정 없음).** 입력은 CL3 의 C1a + F1 · C1a (둘 다 gripper scope 로 서빙, 각 18 run) 와 CL2 C2′ (arms scope 로 서빙, 18 run). 구 모델은 `--capsule-radius-scale 0.05` 등 서버와 같은 flag 로 지은 constraint model (gripper 270 구, arms 422 구) 이고, 실제 접촉 판정은 MJCF collision geom 의 MuJoCo 거리와 replay contact 로 했다. 판정 기준은 측정 전에 고정했다 (2026-10-12 23:41:40 KST, §1 표). 중단 기준 (lead): 한 번의 offline 실행, 새 closed-loop 없음, 어떤 최소 집합도 criterion 2 를 만족하지 않으면 "DX-04 비교 불성립" 으로 닫고 사용자 판정에 올린다. 회귀 기준선: legacy 14 / 15 · 15 / 15 · −29.031048 mm 일치 (`numbers.baseline_legacy`). **cuRobo 기준선은 GPU 를 쓰지 않아 돌리지 않았다** (#30).

**3.1 gripper scope 에서는 비교가 퇴화한다.** policy-active chunk 에서 target 표면 50 mm 안에 든 non-authorized query 행 수가 0 이다.

| arm (gripper scope 로 서빙) | window | policy-active chunk | non-authorized 행 평가 수 | **target 50 mm 안 non-authorized 행** | **LTE ≠ global chunk (m = 10 mm)** | non-authorized 행의 target 최소 거리 (min / median, plan 16 행) |
|---|---|---|---|---|---|---|
| C1a + F1 | pre8close | 162 | 349,920 | **0** | **0** | 475.67 / 508.34 mm |
| C1a + F1 | lift | 2 | 4,320 | 0 | 0 | 505.18 / 508.19 mm |
| C1a + F1 | carry | 63 | 136,080 | 0 | 0 | 247.87 / 338.78 mm |
| C1a | pre8close | 162 | 349,920 | 0 | 0 | 474.73 / 508.10 mm |
| C1a | carry | 102 | 220,320 | 0 | 0 | 180.64 / 386.96 mm |
| CL2 C2′ 기록 (같은 gripper scope 정의로 평가) | pre8close | 162 | 349,920 | 0 | 0 | 476.84 / 510.12 mm |

(출처: `numbers.item1_degeneracy_gripper_scope`, `numbers.windows_chunks`. 같은 표의 m = 0 · 50 mm 에서도 LTE ≠ global chunk 는 모두 0 이다. C1a 의 lift 는 1 chunk 이고 C2′ 의 lift 는 policy-active chunk 가 0 이라 평가 대상이 없다.) 반대로 **authorized 행은 target 근처에 있다**: C1a + F1 pre8close 에서 authorized 행 349,920 개 중 172,801 개가 plan 16 행 기준 target 50 mm 안이다. 즉 이 scope 에서 target 근처에 오는 것은 전부 손이라 LTE 와 global exclusion 이 같은 field 를 읽는다. 이것이 **퇴화** 다.

**3.2 어떤 non-authorized link 가 target 에 접근하는가 — arms scope 구 모델 (pre8close, executed rows, run 당 최소 거리).** criterion 2 = window 안 chunk 중 하나라도 구 표면–target 표면 거리 ≤ 50 mm 인 run.

| link (used = 실행에 쓴 쪽 팔) | C1a + F1 criterion 2 run | C1a | C2′ | C1a + F1 의 run 최소 거리 (median, 범위) |
|---|---|---|---|---|
| **`link_*_arm_5`** | **18 / 18** | **18 / 18** | **18 / 18** | **−2.16 mm (−12.41 – 1.57)** |
| `link_*_arm_6` | 0 / 18 | 0 / 18 | 0 / 18 | 130.23 mm (122.35 – 134.81) |
| `link_*_arm_4` | 0 / 18 | 0 / 18 | 0 / 18 | 206.51 mm (193.14 – 214.83) |
| `link_*_arm_3` · 반대 팔 · 반대 손 | 0 / 18 | 0 / 18 | 0 / 18 | (그래프 (a) 참조) |

(출처: `numbers.item2_approach_sphere_model`, `verdict`. 음수는 구 표면이 target hull 안으로 들어간 것이다. C2′ 의 arm_5 run 최소 거리는 median 16.45 mm (10.86 – 38.86): arms scope 로 서빙한 C2′ 는 arm_5 구가 constraint 에 들어 있어 target 에서 떨어져 있었던 것으로 읽힌다 (미측정 해석). arm_5 구가 target 50 mm 안이었던 chunk 는 C1a + F1 90 (1449 행) · C1a 91 (1482 행) · C2′ 125 (1798 행). 같은 arm_5 구가 이웃 과일 50 mm 안에 든 run 은 C1a + F1 17 · C1a 17 · C2′ 9 / 18.)

**3.3 그 접근은 실제 접촉 geom 에서도 일어나는가 — 아니다 (형상 근사 오류).** 같은 run · window 의 MuJoCo MJCF geom 과 replay contact (C1a + F1, 18 run, 1296 control step):

| 항목 | 값 | 출처 |
|---|---|---|
| MJCF `link_<used>_arm_5` geom 의 run 최소 target 거리 | **18 / 18 run 모두 100 mm cap 이상** (`mj_geomDistance` distmax 100 mm, 그래서 "100 mm 이상" 으로만 안다), C1a 도 18 / 18 | `item2_approach_mjcf`, `item2_arm5_geometry`, `verdict_facts.mjcf_arm_5_runs_beyond_100mm_C1aF1 = 18` |
| replay contact: 손 이외 로봇 ↔ target | **0 / 1296 step**, 해당 run 0 (C1a 도 0 / 1296) | `replay_contacts.robot_other-target_steps` |
| replay contact: 손 ↔ target | 321 step | 같음 |
| MJCF `FT_SENSOR` (`link_*_arm_6` 의 자식) 의 run 최소 target 거리 ≤ 50 mm | **3 / 18 run (C1a + F1), 4 / 18 (C1a)**, median 72.77 / 68.08 mm | `tgt_run_min_mm.A_FT_SENSOR` |
| 같은 run 의 URDF `link_*_arm_6` 구 최소 target 거리 | ≥ 122 mm (C1a + F1: 122.35 mm) | §3.2 |
| URDF `link_*_arm_5` capsule | 반지름 75 mm · 길이 250 mm, 먼 끝이 joint 에서 225 mm. `--capsule-radius-scale 0.05` 로 지은 구의 effective 반지름 14.39 mm (arm_6 구는 5.25 – 5.35 mm) | `item2_arm5_geometry` |
| target 50 mm 안에 든 arm_5 구 (36 CL3 run 의 가장 가까운 executed step) | **87 개**. joint 에서 169.4 – 225.0 mm 떨어져 있고, **가장 가까운 authorized 손 구의 중심에서 11.0 – 47.5 mm** | 같음 |

해석 (사용자 지정 문구에 맞춰): URDF 의 긴 arm_5 capsule 의 먼 끝 구가 손 영역 안에 놓여 있고, 그 구가 target 에 접근하는 것은 실제 arm_5 geom 의 접근이 아니라 **근사의 산물** 이다. 반대로 실제 MJCF 에서 target 에 가까워진 비허용 body 는 URDF 구가 덮지 않는 `FT_SENSOR` (3 – 4 / 18 run) 였다. 이 두 사실이 T46f 의 형상 보정 (FT sensor cover 추가 · arm_5 근사 보정) 의 근거다.

![T46d 장면: gripper scope run 에서 non-authorized arm_5 구가 target 에 놓인 실제 frame, 같은 단면의 full field 와 target-free field](figures/t46/t46d-scene.png)

[`t46d-scene.png`](figures/t46/t46d-scene.png) (sidecar [`.json`](figures/t46/t46d-scene.json)) — `C1aF1_ep104_s1042`, chunk seq 14, control step 104 (close seq 16, 쓴 팔 left, target pear). (a) front camera, (b) 같은 상태의 MuJoCo render 에 arms scope 구를 겹침 (주황 = arm_5, 초록 = arm_6, 노랑 = arm_4, 파랑 = 허용 손 구), (c) 배치도 (top view, plan 16 행의 구 중심, GT hull, 단면선 y = 0.297 m), (d) 단면에서 full field (LTE 에서 non-authorized 행이 읽는 것), (e) 같은 단면의 target-free field (global exclusion 에서 모든 행이 읽는 것). executed step 104 의 구 표면–GT hull 거리: `link_left_arm_5` −11.72 mm, `link_left_arm_6` 122.49 mm, `link_left_arm_4` 204.98 mm, `ee_left` 28.22 mm, `ee_finger_l1` 20.11 mm, `ee_finger_l2` 5.50 mm. 이 chunk 에서 m = 10 mm 에 LTE 와 global 이 갈리는 arm_5 행은 31 개이고, 가장 가까운 행의 full field 는 −14.39 mm, target-free field 는 34.26 mm, GT target 거리는 −13.62 mm 다. **이 장면은 gripper scope 로 서빙한 run 을 arms scope 구 모델로 다시 읽은 것이라 서버는 이 arm_5 구를 constraint 로 쓰지 않았다.**

**3.4 후보 집합과 비용, 그리고 세 정책의 offline 차이.** 후보 = gripper scope 에 non-authorized link 를 (양 팔 모두) 더한 것. TO 시간은 구 수에 선형이라고 가정한 **추정** 이다 (CL2 C1b′ 270 구 median 160.34 ms, C2′ 422 구 median 402.87 ms 사이에서 보간, 재풀이 없음). LTE ≠ global 은 pre8close 162 chunk 중 chunk 수이고 field-level re-query 다 (용어). C2′ binding = C2′ 의 chunk 에서 그 link 집합이 최악 행이었던 chunk 수 (pre8close 162 chunk / 전체 1350 chunk).

| 후보 | 더한 구 (양 팔) | 구 합계 | ESDF 행 / chunk (gripper 대비) | TO solve ms (**추정**) | criterion 2 run (C1a + F1 / C1a / C2′) | LTE ≠ global chunk, C1a + F1 (m = 0 / 10 / 50 mm) | C1a | C2′ | C2′ binding (pre8close / 전체) |
|---|---|---|---|---|---|---|---|---|---|
| `--links gripper` | 0 | 270 | 4320 (1.00×) | 160 (C1b′ 실측 median) | 0 / 0 / 0 (반대 손) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | — |
| + `link_*_arm_6` | 58 | 328 | 5248 (1.21×) | 253 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 1 |
| **+ `link_*_arm_5`** | **20** | **290** | **4640 (1.07×)** | **192** | **18 / 18 / 18** | **78 / 89 / 84** | **77 / 81 / 83** | **0 / 4 / 82** | **52 / 86** |
| + `link_*_arm_5–6` | 78 | 348 | 5568 (1.29×) | 285 | 18 / 18 / 18 | 같음 | 같음 | 같음 | 52 / 87 |
| + `link_*_arm_4–6` | 94 | 364 | 5824 (1.35×) | 310 | 18 / 18 / 18 | 같음 | 같음 | 같음 | 52 / 87 |
| + `link_*_arm_3–6` | 112 | 382 | 6112 (1.41×) | 339 | 18 / 18 / 18 | 같음 | 같음 | 같음 | 52 / 87 |
| `--links arms` (C2′) | 152 | 422 | 6752 (1.56×) | 403 | 18 / 18 / 18 | 같음 | 같음 | 같음 | 52 / 87 |

(출처: `numbers.item3_candidates`, `numbers.item3_to_time_basis`. "같음" = arm_5 를 포함하는 집합은 arm_5 를 넘어 더해도 숫자가 바뀌지 않는다 — 비교가 갈리는 것은 전부 arm_5 구에서만 나온다: C1a + F1 의 m = 10 mm 위반 행 차이 1048 행 전부가 `A_arm_5` 이다 (`flip_rows_by_link_m10`).) C2′ 의 pre8close binding link 는 `ee_finger_r2` 60 · `ee_finger_l1` 45 · `link_right_arm_5` 43 · `link_left_arm_5` 9 · `ee_right` 4 · `ee_finger_r1` 1 chunk (합 162, `numbers.item3_C2p_binding_by_link_pre8close`) — **arm_5 가 최악 행인 chunk 는 52 / 162 이고 손 link 가 110 / 162 이다** (산출).

![T46d 그래프: 어느 non-authorized link 가 target 50 mm 안에 오는가, 후보 집합별 변화와 비용](figures/t46/t46d-graph.png)

[`t46d-graph.png`](figures/t46/t46d-graph.png) (sidecar [`.json`](figures/t46/t46d-graph.json)) — (a) arms scope 구 모델의 link 별 run 최소 거리 (50 mm 점선, 위 숫자 = criterion 2 를 만족한 run 수). (b) 같은 run 의 MJCF geom 최소 거리 (100 mm cap), replay contact 가 위 제목에 있다. (c) 후보 집합별 LTE ≠ global chunk 와 C2′ binding chunk. (d) 후보별 ESDF query 행 (정확) 과 TO 시간 (추정).

![T46d 표: 후보 constraint 집합과 arm_5 · non-authorized 형상 사실](figures/t46/t46d-table.png)

[`t46d-table.png`](figures/t46/t46d-table.png) (sidecar [`.json`](figures/t46/t46d-table.json)) — 위 후보 표 (§3.4) 와 arm_5 형상 사실 (§3.3) 을 한 장에 모았다.

**3.5 verifier 의 결론과 사용자 판정이 읽은 것.**

- **verifier (사전 고정 기준 2 기준).** "기준 2 를 만족하는 비권한 link: `link_<used>_arm_5` (sphere model) 18 / 18 run (C1a + F1 · C1a · C2′ 각각); arm_6 / arm_4 / arm_3 / 반대팔 / 반대손 0 / 18. gripper scope 비권한 행 target 50 mm 이내 0. 최소 집합 = gripper + `link_*_arm_5` (290 구, 4640 행 / chunk). **측정 정의상 성립.**" (`verdict`, 원문 요약). 비교가 갈리는 chunk 는 C1a + F1 에서 pre8close 162 chunk 중 89 (m = 10 mm, 18 run 모두) 였다.
- **사용자 지정 기록 문구.** "현재 gripper scope 에서는 LTE 와 global exclusion 의 비교가 퇴화하며, `link_*_arm_5` 를 추가했을 때 나타난 차이는 형상 근사 오류의 영향을 받는다." 위 §3.3 이 이 문구의 근거다: arm_5 구가 target 에 접근하는 18 / 18 run 에서 실제 arm_5 geom 은 100 mm cap 이상이고 손 이외 로봇 ↔ target 접촉은 0 / 1296 step 이며, 접근하는 구는 authorized 손 구 중심에서 11.0 – 47.5 mm 에 있다. **verifier 의 "성립" 은 사전 고정한 측정 정의 (구 표면 ≤ 50 mm) 에서의 말이고, 그 접근이 실제 접촉 위험인지는 구 모델의 형상에 달려 있다.**
- **이 결과를 "효과 없음" 으로 기록하지 않는다 (사용자).** 무엇이 측정됐는가: (a) gripper scope 에서 비교가 퇴화함, (b) arm_5 를 더하면 두 정책이 갈리는 chunk 가 생김, (c) 그 갈림이 근사 오류의 영향을 받음. 무엇이 측정되지 않았는가: 형상을 바로잡은 뒤 갈림이 남는가 (#28), 갈림이 closed-loop 의 접촉 · 성공 차이로 이어지는가 (#29).
- **판정 (사용자, 2026-10-13).** A (link-specific 효과 주장 축소) 가 아니라 **B 를 확장해 진행** (형상 보정 → 진단 장면 → oracle extent 3 조건 closed-loop), **C (arm_5 근사 구 그대로) 는 제외**. 이유는 위 해석 (C 는 근사 오류를 그대로 비교에 넣는다 — 이 문장은 scribe 의 정리이고 사용자 문장에는 이유가 적혀 있지 않다). 상세 계획은 §4.

**3.6 읽을 때의 한계 · 측정하지 않은 것 (`T46d.verify.json not_measured`, `fidelity`).**

| 한계 | 내용 |
|---|---|
| **field-level 재계산이다** | LTE ≠ global 은 `d − r` 을 임계값 (m = 0 · 10 · 50 mm) 과 비교한 재조회이고 서빙 설정의 per-row margin (support 0 · obstacle 10 mm · non-authorized target 50 mm 또는 obstacle) 과 TO verdict 는 재현하지 않았다. `--nonauthorized-target-margin` 이 full (기본 50 mm) 인가 obstacle 인가에 따라 LTE arm 의 arm_5 행이 달라진다: C2′ 의 LTE ≠ global chunk 는 m = 10 mm 에서 4, m = 50 mm 에서 82 |
| **재조회 fidelity** | 기록 clearance 와 재조회 `d_full` 의 차가 policy-active chunk 에서 최대 22.00 mm (C1a + F1, 559 chunk 중 0.5 mm 초과 60 chunk), C2′ 는 최대 66.08 mm (1303 chunk 중 190). plan 16 행의 손가락 관절 경로를 재현하지 않아 (planning row 값으로 고정) closing chunk 의 손가락 행이 기록과 최대 26.3 mm 어긋난다. 팔 · 손바닥 행은 정확하다 |
| **구 모델 구성** | 구 위치 재현은 기록과 최대 0.21 mm (C1a) 차이. arm_5 구 87 개는 36 CL3 run 의 가장 가까운 step 에서 센 값 |
| **MJCF 거리의 cap** | 100 mm 에서 잘려 arm_5 · arm_6 의 실제 거리는 "100 mm 이상" 으로만 안다. `FT_SENSOR` · `EE_BODY` 의 0.0 mm 거리는 접촉 필터를 거친 겹침과 구별하지 못했다 (#30) |
| **held 구간** | lift · carry 는 대부분 held (policy 가 active 가 아님) 라 LTE ≠ global 을 평가한 chunk 가 적다 (C1a + F1 lift 2 · carry 63 chunk) |
| **TO 시간은 추정이다** | 구 수 선형 보간이고 실측이 아니다 |
| **표본** | 18 run × 3 arm. no exemption 은 "모든 행이 full field 를 읽는다" 로만 평가했고 서빙된 `relax` 정책은 재현하지 않았다 |

**구현 요구 (`implementation_needs`).** 최소 집합 (gripper + arm_5, 290 구) 은 기존 flag 로 표현할 수 있다: `--links arms --exclude-links link_{left,right}_arm_{0,1,2,3,4,6}` (12 개 이름). 서버는 이를 "arms minus ..." 로 표시하고 진단용 EXCLUDING 경고를 낸다. 이름 붙은 `--links` 값 (예: gripper + arm_5) 은 없다 (`LINK_GROUPS` 는 gripper / arms / all). `--target-field-policy` 의 선택지는 `relax` / `exclude-authorized` / `exclude-all` 이고 **no exemption 은 별도 선택지가 아니다** (`relax` 는 권한 link margin 을 완화한다).

### 4. T46f — LTE 검증 블록 (계획, **결과 없음**)

> 이 절의 모든 서술은 [`T46f.task.md`](handoff/T46f.task.md) (lead, 2026-10-13 01:23 KST) 의 **사전 고정** 이다. 측정된 것이 없으므로 verify.json 이 없고 이 절에 결과 수치는 없다. 상태: 단계 1 (형상 보정, A14) 이 진행 중이다.

**검증할 가설 (사전 고정).** "지정된 link 에만 target 접촉 예외를 적용하면, 필요한 손의 접촉을 유지하면서 비허용 link 의 접촉을 줄일 수 있다."

| 항목 | 정의 |
|---|---|
| **주 지표** | **비허용 접촉 없는 작업 성공률** = task success ∧ 실제 MuJoCo geom 기준 non-authorized link–target 접촉 0 |
| 함께 보고 | 일반 task success · 실제 geom 기준 non-authorized 접촉 발생률 · 접촉 지속 시간 (step) · non-authorized link–target 최소 표면 거리 · 정지 (HOLD) · TO 실패율 · 작업 완료 시간 · **실측** TO 계산 시간 · 같은 episode · seed 의 실패 → 성공 / 성공 → 실패 쌍 수 (McNemar) |
| 보조 지표 | LTE ≠ global chunk 수 — **메커니즘 설명용이고 contribution 의 근거가 아니다** |
| 해석 규칙 | HOLD · TO 실패가 늘어 접촉률이 낮아진 결과는 보호 효과로 해석하지 않는다. 정확한 형상과 oracle extent 에서도 LTE 가 주 지표를 개선하지 못하면 그대로 보고하고 가설을 재검토한다. **사후 기준 변경 금지** |
| 전제 | **LTE 의 우수성을 전제하지 않는다** (사용자 판정) |

**접촉 허용의 정의 (lead, §2).**

| 구분 | link / 접촉면 | target 에 대한 제약 |
|---|---|---|
| authorized | active hand 의 손바닥 `ee_<side>` (MJCF `EE_BODY_*`) 와 손가락 `ee_finger_<side>1/2` | LTE 에서 target-free 층을 읽음 (접촉 허용) |
| non-authorized | **`FT_SENSOR_<side>`** (손목 F/T 센서), `link_*_arm_5` · `link_*_arm_6`, 반대 손 · 팔 | target 을 일반 obstacle 로 (`nonauthorized_target_margin` = obstacle) |

- **FT sensor 비접촉은 이 작업의 요구사항으로 정의한다** (센서 보호; lead 의 정의). 실제 접촉 판정은 MuJoCo `FT_SENSOR_*` geom 과 target geom 의 contact 다.
- target 영역 = target 과일. oracle 단계에서는 T46e 의 **visible observed surface oracle** — 카메라에 실제 보인 target 표면 점에서 온 seed — 를 쓴다. 전체 형상 (full mesh) oracle 이 아니다. 접촉 허용 link 집합과 target 영역은 서로 다른 축이다.

**블록 분리 — DT-02 와 DT-05A 를 섞지 않는다 (lead, §3).**

| 블록 | 질문 | 바꾸는 것 | 고정 |
|---|---|---|---|
| **DT-02 extent (T46e · CL4)** | 같은 identity · 같은 예외 정책에서 extent 처리의 효과 | ball · ball_owned · + F1 · oracle | 예외 정책 = LTE, 형상 = 보정 형상 |
| **DT-05A link 예외 (T46f)** | 같은 extent 에서 예외 정책의 효과 | ① 예외 없음 · ② global exclusion · ③ LTE | extent = 1차 oracle (visible observed surface), 2차 AG3S (ball_owned + F1) |

공통: 같은 보정 형상 · 초기 상태 · seed · TO 예산 · W1 on · snapshot. CL4 (DT-02 closed-loop) 는 단계 1 의 형상 보정이 끝난 뒤 보정 형상으로 실행한다 (두 블록이 같은 형상을 쓰게). 정책 구현: ① = `--target-field-policy none` (새로 추가, 모든 link 가 target 을 일반 obstacle 로, 권한 link margin 완화도 없음), ② = `exclude_all`, ③ = `exclude_authorized` + `--target-permission-strict` + `--nonauthorized-target-margin obstacle`. 정책 외 조건은 동일하다.

**단계 (lead, §4 – §6).**

| 단계 | 내용 | 상태 |
|---|---|---|
| **1. 충돌 형상 보정 (A14, implementer)** | (1) **FT sensor cover (`wrist_cover`) 구 추가** — 기존 coverage 코드 (`gripper_cover.link_surface` · `finger_body_spheres` · `_body_capsules`) 를 MJCF `FT_SENSOR_*` mesh 에 재사용, non-authorized. (2) **arm_5 근사 구 보정** — MJCF 실제 arm_5 geom 기반 coverage 구로 교체하거나 손 영역과 겹치는 구를 제거. 보정하지 못하면 그 구에 의존한 보호 효과를 주장하지 않는다. (3) **coverage audit** (필수): 구 밖 비율 · 밖 거리 (median / p95 / max) = 형상 누락, 구가 geom 밖으로 나간 정도의 max = 과도한 팽창, 비허용 구와 허용 구 · MJCF 손 geom 의 겹침. (4) 새 link 집합 (예: `--links gripper_wrist` = gripper + 보정 arm_5 · arm_6 · FT sensor) 을 세 조건 · CL4 모든 arm 이 같이 쓴다 (LTE 에만 coverage 추가 금지). (5) `--target-field-policy none` 추가. (6) 기본값은 모두 기존 그대로 (off bit-identical), 격리 worktree 에서 | **진행 중 (결과 없음)** |
| **2. 진단 장면 (A15 / V)** | 기존 18 쌍은 일반 작업 성능 · 부작용 평가로 유지. 별도로 **손의 target 접촉과 FT sensor 회피가 동시에 필요한 진단 장면** 을 만든다 (아래 S1 – S4) | 대기 |
| 3. 소규모 진단 (dev) | 회피 가능성 확인: ③ 에서 성공하는 경로가 존재하는가, 측정 장치가 정상인가 | 대기 |
| 4. oracle extent 로 ① · ② · ③ | 진단 장면 (dev → eval) + 기존 18 쌍 | 대기 (T46e 의 검증 · 적용이 선행) |
| 5. AG3S extent (ball_owned + F1) 로 반복 | 단계 4 에서 효과가 확인되면 같은 비교를 반복 | 대기 |

**진단 장면 선정 기준 S1 – S4 (방법별 결과를 보기 전에 고정, lead §5).**

| 기준 | 내용 |
|---|---|
| S1 | 실제 MuJoCo geom 기준 non-authorized (FT sensor · arm) 접촉 위험이 존재. 정책 rollout (E0: VLA 단독) 에서 FT sensor–target 최소 표면 거리 ≤ 50 mm 는 **후보 선별 기준일 뿐 충돌 · 보호 효과의 증거가 아니다** |
| S2 | 접근 경로나 손목 자세를 바꾸면 작업 완료가 가능 — 같은 배치에서 다른 손목 yaw / 접근 방향의 성공 시연 (데이터셋 demo 또는 IK / 계획 feasibility) 이 1 개 이상 |
| S3 | 초기 관통 (t = 0 접촉) 이나 구조적으로 성공 불가능한 배치 제외 |
| S4 | 개발용 배치와 최종 평가용 배치를 episode id 로 사전 분리 (예: dev = train split, eval = held-out test split 1800 – 1999) |

후보 생성 방법 (사전 고정): E0 rollout 과 기존 기록에서 S1 을 만족하는 배치를 찾고, 부족하면 target 을 crate 벽 · 이웃 과일 사이 등 손목이 낮게 들어가야 하는 위치로 옮긴 변형 배치를 규칙으로 생성한다.

**진행 · 중단 기준 (측정 전 고정, lead §8).**

- 단계 1 의 audit 에서 FT sensor 형상 누락 (구 밖 표면 비율) 이나 손 영역 침범이 남으면 단계 2 로 넘어가지 않는다.
- 단계 2 에서 S1 – S3 를 만족하는 eval 배치가 사전 정한 최소 수 (예: 10) 에 못 미치면 사용자에게 보고하고 장면 생성 규칙을 판정받는다.
- 단계 4 전에 비교 조건 · 장면 수 · 예상 GPU 시간 · 진행 / 중단 기준을 정리해 **사용자 확인 후** 실행한다.

**의존성 (lead §9).** T46e (oracle extent) 는 격리 worktree 에서 구현 완료 · 미적용이다. 단계 4 의 oracle 은 T46e 를 쓰므로 T46e 의 검증 · 적용이 선행한다. T47 session 의 측정 보호를 위해 메인 tree 의 pipeline · config · curobo_builder 변경 전에 알린다.

**develope_PLAN 과의 연결.**

| 계획서 단위 | 이 절이 주는 것 | 주의 |
|---|---|---|
| **DX-04 선행 확인 (§11 순서 1)** | T46d: gripper scope 퇴화, 비교는 arm_5 에서만 갈리나 그 구가 형상 근사 오류의 영향을 받음 → link-specific 효과 주장을 줄이지 않고 형상을 보정한 뒤 다시 확인 (사용자 판정). 계획서의 선택지 중 "관련 비허용 link 를 모든 조건에 동일하게 추가" 쪽 | 계획서의 다른 선택지 ("link-specific 효과 주장 축소") 는 고르지 않았다 |
| **DT-05A (세 field policy 의 pair 별 contact · success · stop)** | T46f 단계 4 – 5 가 채운다. **지금은 비어 있다** | DX-04 판정: "No exemption 이 멈춘다는 관찰만으로 LTE 전체 효과를 결론내리지 않는다" (develope_PLAN) |
| **DF-04B (같은 ESDF slice 의 link 별 distance · gradient 와 실제 contact)** | [`t46d-scene.png`](figures/t46/t46d-scene.png) 는 full field 와 target-free field 의 같은 단면을 보이는 **후보 장면** 이다. 단 서버가 그 arm_5 구를 constraint 로 쓰지 않은 run 의 offline 재조회이고 형상 근사 오류의 영향을 받는다 | 보정 형상 · 진단 장면 뒤에 새로 만든다. 지금 장면을 DF-04B 로 쓰지 않는다 |
| **DC-3** (link-specific exemption 이 필요한 접촉과 금지 접촉을 더 잘 구분한다) | 근거 없음 (측정 안 됨). T46f 가 가설을 사전 고정했다 | 효과가 확인된 link · phase · scene 만 서술한다 (계획서의 판정 규칙) |
| **DC-4** (task completion 과 forbidden contact 의 trade-off 를 개선한다) | 주 지표 (비허용 접촉 없는 작업 성공률) 와 정지 · 시간을 함께 보고하는 것으로 구조화 | 충돌만 줄고 성공 손실이 크면 trade-off 와 실패 조건을 명시한다 |
| **DT-02 extent block** (T46c §6 의 초안) | W1 on 을 공통 조건으로 확정 (사용자). 비허용 link 범위는 T46f 의 보정 형상이 정한다 | oracle arm 은 T46e 의 검증 · 적용 뒤 |

### 5. 되돌아올 지점 · 판정 대기

| 안 고른 것 | 왜 | 되돌아올 신호 |
|---|---|---|
| **A: link-specific 효과 주장 축소** (T46d 가 올린 선택지) | 사용자가 B 확장을 택했다 | 단계 1 audit 에서 FT sensor 형상 누락 · 손 영역 침범이 남음, 또는 단계 2 에서 S1 – S3 를 만족하는 eval 배치가 최소 수 (예: 10) 에 못 미치고 장면 생성 규칙도 성립하지 않음 (T46f §8) |
| **C: arm_5 근사 구를 그대로 비교** | 사용자가 제외했다 | 형상을 보정하지 못하는 구가 남으면 그 구에 의존한 보호 효과를 주장하지 않는다 (T46f §4-2) |
| W1 기본 on | 기본값은 off (사용자 판정) | 실험 arm 이 아니라 서버 기본값에서 `no_seed` HOLD 가 문제가 되는 측정 |
| W1 을 `place` label 로 넓힘 | CL3 의 placed `no_seed` 는 전부 `home` 이었다 (구현자, `T46c.impl.md`) | `place` 칸에서 같은 HOLD 가 관찰될 때 |
| 표본 확대 | 사용자가 18 쌍 유지를 택했다 (develope_PLAN §0.4) | 사용자가 새로 판정할 때만 |

**판정 대기.** (1) T46f 단계 4 전의 비교 조건 · 장면 수 · 예상 GPU 시간 · 진행 / 중단 기준 확인 (단계 4 직전, lead 가 정리해 올림). (2) T46e (visible observed surface oracle) 의 검증 · 적용 시점. (3) T46f 단계 2 의 S1 – S3 만족 eval 배치 최소 수 (예: 10) 가 사전 정한 값으로 확정되는가.

**산출물.**

- **측정**: [`handoff/T46c.verify.json`](handoff/T46c.verify.json) (원자료 `/mnt/dev/AG3S_VLA-storage/results/development/20261012T233515+0900_972f28056cd8_318f98f36f37/verify/T46c/`), [`handoff/T46d.verify.json`](handoff/T46d.verify.json) (원자료 `.../20261012T233920+0900_392462577320_70c141f47dbe/verify/T46d/`, 입력 CL3 · CL2 · CL-A 는 읽기 전용).
- **figure** (각각 `.json` sidecar 있음): [`t46c-table`](figures/t46/t46c-table.png) · [`t46d-scene`](figures/t46/t46d-scene.png) · [`t46d-graph`](figures/t46/t46d-graph.png) · [`t46d-table`](figures/t46/t46d-table.png). T46f 는 figure 가 없다 (계획).
- **구현** (T46c-W1): `AG3S/ag3s/config.py` · `AG3S/ag3s/runtime/pipeline.py` · `AG3S/ag3s/types.py` · `AG3S/trajopt/safe_policy.py` · `serve_safe.py` · `wire.py` · `experiments/certification_replay.py` · `experiments/ta_offline.py` · `tests/trajopt/test_t46c_placed_no_seed_certify.py` (commit `6dfaf7bb`). 상세 [`handoff/T46c.impl.md`](handoff/T46c.impl.md).
- 요청 문서: [`handoff/T46c.task.md`](handoff/T46c.task.md) · [`handoff/T46d.task.md`](handoff/T46d.task.md) · [`handoff/T46e.task.md`](handoff/T46e.task.md) · [`handoff/T46f.task.md`](handoff/T46f.task.md). commit 은 하지 않았다.
