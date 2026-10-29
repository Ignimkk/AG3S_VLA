# SL6-B — 구현 (LIBERO target 명사 · HSR6 부분집합 · Panda 단계 label · HSR6-Panda probe · noun-row 추출 · head 선택 계산)

> writer: ag3s-implementer (SL6-B) · 읽는 쪽: lead, verifier, scribe · 작성 2026-10-29 19:50 KST (`date -u` 10:50 + 9 h)
> task: [SL6.task.md](SL6.task.md) §0 · §3 · §3.1 · §4 (§2 는 implementer A) · checkout `feature/panda-benchmark` HEAD `5ea56af4` (작업 트리, commit 없음)
> 새 run (B3 · B4 출력): `/mnt/dev/AG3S_VLA-storage/results/benchmark/20261029T192223_5ea56af44722_sl6b-hsr6-panda-labels/` (아래 `$R`)

## 무엇을 했나 (평이한 요약 먼저)

- **B1** LIBERO 문장에서 target 명사를 찾는 규칙을 넣었다. 기본 꺼짐: `NounTokens(rule="libero")` 일 때만 쓴다.
  - 예: "pick up the black bowl between the plate and the ramekin and place it on the plate" → target 명사 `bowl`. query row 는 `▁bowl` 이다.
  - 기본값 (`rule="fruit"`) 은 T57-A 조회와 같다. RB-Y1 과일 · basket 문장의 rows 와 record 가 bit 동일하다.
- **B2** HSR6 probe 가 6 단계의 순서 있는 부분집합이어도 받는다. Panda 는 retreat 가 없어 5 단계다.
  - 없는 단계는 확률 0.0 으로 기록한다. 그래서 retreat 를 쓰는 ③ 은 작동하지 않는다.
  - RB-Y1 6 class probe 는 전과 같은 `SubtaskProbe` 객체로 읽힌다.
- **B3** 기존 dev 기록 40 episode 의 MuJoCo 요청 시점 상태에서 Panda 물리 단계를 요청 단위로 매겼다. 모두 873 요청이다.
  - 손가락별 접촉 · 지지면 접촉 · 목적지 xy box 를 새로 export 했다 (`phase_export --finger-contacts`, 기본 꺼짐).
- **B4** HSR6-Panda probe 를 학습했다.
  - 고른 C 는 **0.01** 이다. episode 단위 5-fold macro-F1 은 **0.834**, grasp recall 은 **0.851** 이다. VB3 조건 (≥ 0.70 · ≥ 0.6) 을 넘는다.
  - probe 는 `$R/probe/hsr6_panda_kv_L4.npz` 이다.
- **B5** `attention_audit.py --noun-rows` 를 넣었다 (기본 꺼짐). GPU 실행은 lead 가 한다.
  - 작은 pi05_libero 모델 (JAX CPU) 에서 추출값이 서버의 `noun_target:L<l>H<h>` cell 과 **bit 동일**하다. 64 cell 모두 같다.
- **B6** §4 의 head 선택 계산을 넣었다 (`sl6_head_select.py`). B5 출력이 생기면 lead 가 아래 명령으로 돌린다.
- VB2 용 단계 띠 json 을 냈다: `$R/labels/vb2_bands.json` (t0e1 · t3e1 · t5e1 · t8e1).

## 1. 바뀐 파일 · md5

| 파일 | md5 (작업 트리) | 무엇이 |
|---|---|---|
| `AG3S/trajopt/attention_source.py` | `6ed314699880097add41b515e9bd4361` (전 `24d543d3…`) | B1 규칙, `NounTokens.rule/sp`, `NounLookup.rule/destination/pieces`, `noun_lookup` 분기 |
| `AG3S/trajopt/hsr6.py` | `fd27eb17817768979456c9c07a09d49f` (전 `b2d07388…`) | B2 `is_ordered_subset`, `Hsr6SubsetProbe`, `load_hsr6_probe` 완화 |
| `benchmark/adapters/libero_panda/phase_export.py` | `61c7948b78d39833e7b275ac7917dec4` (전 `a5653ee3…`) | B3 `--finger-contacts` (기본 꺼짐) |
| `benchmark/runners/attention_audit.py` | `2c99f5066519e645d41e15676784b06a` (전 `2622b29b…`) | B5 `--noun-rows` (기본 꺼짐) |
| `AG3S/ag3s/experiments/studies/sl6_phase_labels.py` (새) | `e104a83336d47df787beee237f088b57` | B3 단계 규칙 (순수 numpy) |
| `AG3S/ag3s/experiments/studies/sl6_hsr6_panda.py` (새) | `673cdb47c4f0b78225c78961f00afce7` | B3 label 계산 + B4 학습 + VB2 json |
| `AG3S/ag3s/experiments/studies/sl6_head_select.py` (새) | `8c0c559e7cda6782466f8e742417a1a4` | B6 head 선택 계산 |
| `tests/trajopt/test_sl6_b1_libero_noun.py` (새) | `5c8d18c207ccb295a8faaf6b1973e0df` | VB1 B1 |
| `tests/trajopt/test_sl6_b2_hsr6_subset.py` (새) | `072022c2cc413736596e9344f49fd503` | VB1 B2 |
| `tests/benchmark/test_sl6_b3_phase_labels.py` (새) | `c2f15b7602adae3aac4f31c375771a3e` | VB1 B3 (합성 상태 경계) |
| `tests/benchmark/test_sl6_b5_noun_rows.py` (새) | `0167285df58d5aab50af3fb8a85eb4da` | B5 CPU 부분 + tiny model |
| `tests/benchmark/sl6_tiny_noun_rows.py` (새) | `e952f8f635c9b5567c6f08561f479fdf` | B5 tiny model 보조 (subprocess, JAX CPU) |
| `tests/ag3s/test_sl6_b6_head_select.py` (새) | `64c9ab28d367273460603b1d4088dfa0` | B6 합성 |

`AG3S/ag3s/stages/*` 는 고치지 않았다. `SubtaskProbe` 를 상속만 한다.

## 2. 규칙과 코드 위치

### B1 — LIBERO target 명사 (`AG3S/trajopt/attention_source.py`)

| 무엇 | 위치 | 내용 |
|---|---|---|
| 문장 형식 | `:187` `_LIBERO_PICK_PLACE` | `^pick up the <target> and place it <prep> the <dest>[.]$` (대소문자 무시). tokenizer 의 정리 (`clean_prompt` `:213` = `tokenizer.py:23`) 를 먼저 한다 |
| head word | `:191` `PHRASE_STOP_WORDS` · `:217` `_phrase_head` | target phrase = "pick up the" 뒤에서 첫 stop word (between · next · from · on · in · of · to · …) 까지. head = 그 직전 단어다. 목적지도 같은 방법으로 찾는다 |
| 위치 찾기 | `:255` `token_char_spans` · `:272` `_word_tokens` · `:285` `_libero_lookup` | piece 를 글자로 풀어 (`▁`=공백) **글자 위치**로 head word 를 찾는다. 그래서 두 번 나온 목적지 명사 (plate) 를 target 으로 잡지 않는다. 단어 양 끝이 piece 경계여야 한다. 아니면 not ok 로 둔다 |
| query row | `:285` | target 의 **마지막 piece** 다. rows = `[768+target, 768+destination 마지막 piece, 768+last_text]` (`QUERY_NAMES` 순서 그대로, jit 모양 동일) |
| 켜는 법 | `:324` `NounTokens(rule, sp)` · `:339` `from_config(config, *, rule="fruit")` | `rule="libero"` 일 때만 tokenizer 를 들고 B1 을 먼저 시도한다. 기본 `"fruit"` 는 T57 객체와 같다 |
| 분기 | `:378` `noun_lookup` | libero mode 에서 문장이 형식에 맞지 않으면 과일 규칙으로 간다. 이때 record 에 `noun_rule: "libero:fruit"` 와 두 이유를 남긴다. fruit mode 는 전과 같은 코드 경로다 |
| record | `NounLookup.record` (`:362` 부근) | fruit mode 는 키 6 개로 전과 같다. libero mode 만 `noun_rule · destination_noun · destination_row · noun_pieces` 를 더한다 |

결과 (실제 PaliGemma tokenizer, `.venv-openpi-live`):

- libero_spatial 10 문장과 SafeLIBERO spatial 4 문장 모두 noun `bowl` (`▁bowl` 한 piece), 목적지 `plate` (`▁plate`) 를 찾는다.
- pi0 형식과 pi05 형식 모두 그렇다.
- `pi05_libero` 는 `discrete_state_input=False` 다. 그래서 pi0 형식 `<bos>+문장+"\n"` 이고, 유효 token 은 20 개다.

**규칙을 옮기며 정한 것 (lead 확인 요청):**

1. LIBERO-object `"pick up the orange juice and place it in the basket"` 에서 결과가 mode 마다 다르다.
   - libero mode: B1 규칙대로 `juice` 다.
   - fruit mode (기본): 전처럼 `orange` 다.
   - 이번 단계의 데이터 (libero_spatial) · smoke (SafeLIBERO spatial) 에는 이 문장이 없다.
2. B1 은 Panda 서버가 `NounTokens.from_config(config, rule="libero")` 로 켜야 작동한다. A 의 작업 트리 `AG3S/adapters/panda/server.py` `_noun_tokens` 가 이미 그렇게 부른다. 아래 "A 에게" 참고.

### B2 — HSR6 class 부분집합 (`AG3S/trajopt/hsr6.py`)

- `:79` `is_ordered_subset`: 2 개 이상, 중복 없음, `PHASES6` 순서를 지킬 때만 받는다.
- `:88` `load_hsr6_probe`: 6 class 면 전처럼 plain `SubtaskProbe` 를 돌려준다. 부분집합이면 `Hsr6SubsetProbe` 로 감싼다. dim 512 검사는 그대로다.
- `:106` `Hsr6SubsetProbe` (`SubtaskProbe` 상속):
  - `predict` 의 `p` 에 6 단계를 모두 싣는다. 없는 단계는 0.0 이다.
  - argmax 는 probe 의 class 중에서만 고른다.
  - `phase_of` (변경 없음) 는 retreat 를 고를 수 없다. 그래서 ③ 은 작동하지 않는다.
- `record()` 에 `phases_absent` 를 더한다.

### B3 — Panda 물리 단계 (`sl6_phase_labels.py`, export 는 `phase_export.py`)

export (`phase_export.py`, 기본 꺼짐 `--finger-contacts` `:276`). 매 기록 frame 의 `sim_state_image` 에서 다음을 낸다 (`:181`, `:215`, `:246`).

- `contact_left_finger_target`: `gripper0_leftfinger` + `gripper0_finger_joint1_tip` 몸체의 geom 과 target geom 의 접촉 (`FINGER_BODIES` `:67`, `finger_flags` `:111`).
- `contact_right_finger_target`: `gripper0_rightfinger` + `gripper0_finger_joint2_tip` 몸체의 geom 과 target geom 의 접촉.
- `n_contact_target_env`: target 과 로봇이 아닌 geom 의 접촉 수.
- `destination_xy_box`: 목적지 collision geom 의 world AABB (`geom_aabb` 모서리, `geoms_xy_box` `:127`). libero_spatial 의 목적지는 `plate_1` 이다. region site 는 쓰지 않았다 (목적지가 물체라서 geom box 로 정의된다).

규칙 (`sl6_phase_labels.py`, 상수 `RISE_M 0.02` `:43` · `TRAVEL_M 0.05` `:44`, 본문 `frame_phases` `:78`):

| 단계 | 코드 | 읽는 법 (task 문장을 좁힌 곳) |
|---|---|---|
| 닫기 명령 | `close_frames` `:49` | frame f 는 action f−1 의 결과다. `action[f-1,-1] > 0` 이면 닫기 frame. 연속 구간 = close run (`runs_of` `:58`) |
| grasp | `:126` | 두 손가락 접촉 frame 이 있는 close run 의 첫 frame 부터 그 run 의 첫 lift 전까지 grasp. lift 뒤 두 손가락인데 lift 조건이 아니면 (지지면에 다시 놓임) 도 grasp (T64: held ∧ rise < 2 cm). 두 손가락 접촉이 없는 close run 은 approach (실패한 닫기) |
| lift | `:96` · `:124` | 두 손가락 ∧ (z − z0 ≥ 2 cm ∨ target-env 접촉 0). z0 = 첫 요청 frame 의 target z. 대기 step 동안 bowl 이 내려앉아서 frame 0 은 쓰지 않는다 (`libero_phases` 와 같다) |
| transport | `:124` | lift ∧ lift 지점 xy 에서 ≥ 5 cm. lift 지점 = 마지막 close run 이 시작된 뒤 처음 lift 인 frame 의 target xy |
| placement | `:122` | 첫 lift 뒤 target xy 가 목적지 box 안. lift · transport 보다 먼저 판정한다 (T64: 쥐었든 놓았든 crate 위) |
| approach | `:128` | 그 밖. lift 뒤 두 손가락도 아니고 box 밖도 아니면 approach 다 (drop / 실패). 이것을 `lost_hold_frames` 로 센다 |

요청 label = `chunk_t[k]` frame 의 label 이다 (`request_phases` `:144`). kv feature 의 `obs_index` 와 같은지 확인했다 (`sl6_hsr6_panda.load_episodes`).

### B4 — 학습 (`sl6_hsr6_panda.py`)

- 모델은 `t50_core.fit_logreg` 다 (StandardScaler + multinomial lbfgs, max_iter 3000, tol 1e-4 = `t64_phase6.fit`).
- C 후보는 `C_GRID` `:45` 의 {0.003, 0.01, 0.03, 0.1, 0.3} 이다.
- `cross_validate` `:146` 가 `t50_core.group_kfold(5)` 로 episode 단위 fold 를 나눈다. fold macro-F1 의 평균이 가장 큰 C 를 고른다. 동점이면 작은 C 다 (`t64_phase6.choose_C` 와 같다).
- fold macro-F1 = 그 fold 의 정답 ∪ 예측에 있는 class 의 F1 평균이다 (sklearn `average="macro"`, `macro_f1_present` `:124`).
- grasp 시작 지연은 `grasp_onset` `:166` 가 낸다. episode 마다 (첫 예측 grasp − 첫 정답 grasp) 요청 수를 잰다. `t64_phase6.boundary_delays` 의 approach→grasp 도 함께 낸다.
- 40 episode 전체로 학습한 probe 를 `save_probe` `:193` 로 쓴다 (`probe_6.npz` 형식 + `name`). 그 파일을 `hsr6.load_hsr6_probe` 로 다시 읽어 학습 feature 873 개에서 argmax 를 비교했다. fit 과 서빙 경로가 873/873 같다.
- 입력 검사 (`load_episodes` `:65`):
  - libero_spatial init 1–4 가 아니면 거절한다 (init 0 · SafeLIBERO 는 들어올 수 없다).
  - audit run 이 읽은 episode npz 의 sha256 이 같아야 한다.
  - phase track 의 geom npz sha256 이 같아야 한다.

### B5 — `attention_audit.py --noun-rows`

- `:72` `NOUN_LAYERS` = 2–17 이다.
- `:391` `NounRowReader` 는 요청마다 다음을 한다.
  - `policy._input_transform` 을 거친다.
  - `noun_lookup(…, NounTokens.from_config(config, rule="libero"))` 로 query row 를 찾는다.
  - `t49_noun_query_extract.NounQueryExtractor.rows` 로 prefix pass 한 번을 돌린다. `slice_probs` 는 jit 안에서 한다.
  - target row 의 `[layers, 8 heads, 768]` float32 와 mass 요약을 낸다.
- `:442` `audit_noun_episode` 의 join 검사는 SL3 audit 와 같다 (state · prompt · 두 image bit-exact).
- 출력은 `noun_rows/<stem>_noun.npz` 와 `noun_index.jsonl` 이다. index 에는 BDDL target 과 instances_geom_ids 가 들어간다.
- `:496` `run_noun_rows` 는 새 run 을 만든다 (`create_run`, step `SL6-B5`).
- `:618` main 에서 flag 가 꺼지면 전의 `run(args)` 를 그대로 부른다.
- 명사를 못 찾은 요청은 0 으로 두고 `noun_ok=False` 로 기록한다. 지어내지 않는다.

### B6 — head 선택 (`sl6_head_select.py`, §4 그대로)

- 후보는 layer 2–17 × head 0–7 (128) 이다. 데이터는 B3 label 이 approach 인 요청의 head 카메라다.
- 2D 지도 (`pixel_maps` `:77`):
  - head 16×16 grid 를 un-mirror 한다.
  - V11AB 의 `attention_pixel_map` 과 같은 bilinear 로 256×256 으로 올린다. 같은 식을 batch 로 옮긴 `resample_batch` `:55` 다 (test 로 bit 동일 확인).
- identity (`request_metrics` `:118`):
  - 상위 1 % pixel = 656 개다 (`top_mask` `:84`, 동점은 낮은 index, stable argsort 와 같다).
  - 그 안에서 가장 많은 label 이 target 이면 1 이다. 다른 모든 label 보다 **엄격히** 많아야 한다.
  - label 은 BDDL 물체 · fixture 각각, `robot`, `other`, `none` 이다 (`label_image` `:99`).
- mass = target pixel attention 합 / 전체 합이다.
- coverage = target pixel 중 상위 10 % 에 든 비율이다. target 이 안 보이면 정의하지 않는다 (nanmean).
- 점수는 task 평균이다 (`task_average` `:139`). 순위는 identity → mass → coverage → 낮은 layer → 낮은 head (`rank` `:146`).
- 결정 (`decide` `:152`): 최고 cell 과 L10H7 의 identity 차가 2 %p 미만이면 L10H7 을 유지한다.
- 참고 줄로 Panda V11AB `action:last:L8H2` 를 같은 지표로 낸다 (SL4b audit `pooled_f16[:, 8, 2]`). 후보에는 넣지 않는다.

**§4 를 옮기며 정한 것 (lead 확인 요청):** 정답 segmentation 의 출처.

- §4 는 "요청 시점 head 카메라 segmentation (`segmentation_geom/head`)" 이라고 적었다. 그런데 rollout 의 `segmentation_geom/head` 는 **반환 시점** 상태로 그려졌다 (SL3: frame 당 100–300 px 다름).
- 그래서 기본값은 요청 시점 상태로 다시 그린 `segP/head` 로 했다 (`$R/phase/`, SL4b 선택 지표와 같은 출처).
- `--seg-source rollout` 이면 `segmentation_geom/head` 를 쓴다. 둘 다 돌려 선택이 같은지 보면 된다.

## 3. 단위 검증 (VB1)

```bash
source scripts/setup/env.sh
# .venv-ag3s (CPU) — 새 test + 바꾼 module 에 닿는 기존 test 23 파일
nice -n 10 .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/ag3s/test_ifg_s2serve_gate.py tests/ag3s/test_subtask_gate.py \
  tests/ag3s/test_t48_hsr_counterfactual.py tests/ag3s/test_t48b_hsr_cli.py tests/ag3s/test_t50_hsr_rep.py tests/ag3s/test_t55_attention.py \
  tests/ag3s/test_t49_noun_query.py tests/ag3s/test_sl6_b6_head_select.py tests/benchmark/test_panda_sl3.py tests/benchmark/test_panda_sl4b.py \
  tests/benchmark/test_panda_sl4b_history.py tests/benchmark/test_sl6_b3_phase_labels.py tests/benchmark/test_sl6_b5_noun_rows.py \
  tests/trajopt/test_core_eval_a6.py tests/trajopt/test_core_eval_hsr6.py tests/trajopt/test_ifg_s2serve_resid.py \
  tests/trajopt/test_sl6_b1_libero_noun.py tests/trajopt/test_sl6_b2_hsr6_subset.py tests/trajopt/test_subtask_gate_default.py \
  tests/trajopt/test_subtask_wiring.py tests/trajopt/test_t57a_attention_source.py tests/trajopt/test_subtask_no_target.py \
  tests/trajopt/test_subtask_no_admissible_home.py
# .venv-openpi-live — 실제 tokenizer · tiny pi05_libero (JAX_PLATFORMS=cpu subprocess)
nice -n 10 .venv-openpi-live/bin/python -m pytest -q tests/trajopt/test_sl6_b1_libero_noun.py
nice -n 10 .venv-openpi-live/bin/python -m pytest -q tests/benchmark/test_sl6_b5_noun_rows.py -k tiny
```

| 묶음 | 결과 |
|---|---|
| `.venv-ag3s` 23 파일 | **694 passed, 12 skipped, 0 failed** (5 분 7 초). skip 은 tokenizer · jax 가 필요한 test |
| `.venv-openpi-live` B1 | **27 passed** (실제 PaliGemma tokenizer: libero_spatial 10 · SafeLIBERO 4 · RB-Y1 6 문장, pi0 · pi05 형식) |
| `.venv-openpi-live` B5 tiny | **1 passed** (3 분 24 초). layer 2–9 × head 0–7 = 64 cell 이 `JaxPrefixBackend.run(noun_target)` 와 bit 동일. 오른손목 block mass 0 |

VB1 항목별:

- **B1**
  - fruit mode 에서 RB-Y1 문장 · LIBERO 문장 · crate 문장 · 빈 문장의 record 와 rows 가 T57-A 본문을 그대로 옮긴 `_legacy` 와 같다. stub tokenizer 와 실제 tokenizer 모두 그렇다.
  - libero mode 에서 libero_spatial 10 문장과 SafeLIBERO 4 문장은 bowl / plate 다.
  - RB-Y1 문장은 libero mode 에서도 fruit mode 와 같은 rows 를 낸다.
  - 여러 piece 단어 (`ramekin` → `▁rame`+`kin`) 는 `kin` row 를 쓴다.
  - plate 가 두 번 나와도 target 은 bowl 이다.
  - 경계가 어긋나거나 tokenizer 가 없으면 not ok 다.
- **B2**
  - RB-Y1 probe 의 type · proba · predict · record 가 전과 같다.
  - 5 class probe 는 retreat 를 0.0 으로 채운다. 나머지 확률과 argmax 는 `SubtaskProbe` 와 같다. ③ 은 작동하지 않는다.
  - 순서가 틀리거나, 모르는 class 거나, 중복이거나, 1 class 면 거절한다.
- **B3**: 합성 상태로 각 경계를 고정했다. 닫기 시작 (action f−1), 2 cm (1.9 cm 는 아님), 지지면 이탈, 5 cm (4.99 cm 는 아님), 목적지 box, 실패한 닫기, drop, 놓은 뒤 placement 유지, lift 전 box 는 placement 아님. `finger_flags` 와 회전 `geoms_xy_box` 도 확인했다.
- **phase_export 꺼짐 동등성**: `--finger-contacts` 로 낸 40 episode 의 기존 키 11 개 (frames · target_xpos · 접촉 4 개 (contact_grip_target · n_contact_grip_target · contact_robot_target · contact_target_any) · segP/head · segP/left_wrist · chunk_t · call_frames · target_geoms) 가 `outputs/verify/SL4b/raw/phase/` 와 **40/40 array 동일**하다. flag 가 꺼지면 코드 경로가 이전과 같다 (분기 3 곳 모두 `if finger_contacts`).
- **B6 실제 데이터 smoke**: 숫자는 선택 결과가 아니다. B5 출력이 아직 없어서 SL4b action-query `pooled_f16` 를 대신 넣었다. 40 episode · approach 404 요청이 101 초에 돈다. 참고 줄 L8H2 의 identity · mass · coverage 가 128 cell 표의 (8, 2) 와 같다. scratch 에만 썼다.

## 4. B3 label 수

모두 873 요청 = 40 episode (38 성공, t6e3 · t9e1 실패). retreat 는 0 이다.

| class | approach | grasp | lift | transport | placement |
|---|---|---|---|---|---|
| 전체 | 404 | 94 | 91 | 126 | 158 |

| task | approach | grasp | lift | transport | placement |
|---|---|---|---|---|---|
| 0 | 39 | 10 | 7 | 2 | 11 |
| 1 | 43 | 8 | 9 | 9 | 14 |
| 2 | 31 | 11 | 10 | 9 | 19 |
| 3 | 38 | 8 | 9 | 5 | 10 |
| 4 | 44 | 9 | 9 | 19 | 20 |
| 5 | 26 | 9 | 12 | 11 | 15 |
| 6 | 68 | 9 | 8 | 12 | 9 |
| 7 | 30 | 11 | 10 | 26 | 20 |
| 8 | 41 | 10 | 9 | 0 | 19 |
| 9 | 44 | 9 | 8 | 33 | 21 |

episode 별 띠와 사건 frame 은 `$R/labels/request_labels.jsonl` 에 있다 (frame 띠 · close run 수 · lost_hold_frames). 눈에 띄는 것:

- **t0e1**: 첫 close run (frame 48–72) 에서 오른손가락만 닿았다. 그래서 approach 다. 두 번째 run 에서 grasp 가 된다. HSR6 grasp 예측이 7 요청 빠른 곳이다.
- **t6e3 (실패)**: transport 중 떨어뜨렸다. 그 뒤 28 요청은 approach 다 (lost_hold_frames 144).
- **t9e1 (실패)**: 떨어뜨림과 재시도 때문에 `…TTT AAA PPP AAA PPPP` 다. 첫 lift 뒤 box 안이면 placement 라서 그렇다.
- **t7e1**: transport 사이 1 요청이 approach 다. 두 손가락 접촉이 2 frame 끊겼다. 규칙을 글자 그대로 적용한 것이다. 873 요청 중 이 한 곳이다.
- **t8**: transport 0 이다. bowl 이 plate 옆이라 5 cm 를 가기 전에 box 에 든다.
- **t3e1**: 첫 두 손가락 frame 이 곧 lift 다. 그 순간 지지면 접촉이 0 이었다.

VB2 json: `$R/labels/vb2_bands.json`.

- 요청마다 `k`, `frame` (episode npz · 영상의 관측 index), `global_step` (policy record), `label` 이 있다.
- frame 단위 띠 문자열 (A/G/L/T/P) 과 사건 frame 도 있다.
- 대상은 `libero_spatial_t0_e1` · `t3_e1` · `t5_e1` · `t8_e1` 이다.

## 5. B4 숫자

| 항목 | 값 |
|---|---|
| 고른 C | **0.01** |
| C 별 fold 평균 macro-F1 | 0.003: 0.8090 · **0.01: 0.8342** · 0.03: 0.8249 · 0.1: 0.8302 · 0.3: 0.8253 (모든 fit 수렴) |
| fold macro-F1 (C 0.01) | 0.754 · 0.783 · 0.866 · 0.869 · 0.898 |
| episode 단위 5-fold macro-F1 (VB3 숫자, fold 평균) | **0.834** (pooled 0.832, accuracy 0.866) |
| recall | approach 0.923 · **grasp 0.851** · lift 0.769 · transport 0.786 · placement 0.848 |
| precision | approach 0.949 · grasp 0.777 · lift 0.814 · transport 0.853 · placement 0.766 |
| confusion (행 = 정답, 열 = 예측, A G L T P) | A [373 10 1 1 19] · G [7 80 6 0 1] · L [1 13 70 5 2] · T [1 0 7 99 19] · P [11 0 2 11 134] |
| grasp 시작 지연 (첫 예측 − 첫 정답, 요청) | 39 episode: 중앙값 **0**, 평균 −0.13, 범위 −7 (t0e1) … +1. 1 episode (t6e3) 는 out-of-fold 에서 grasp 를 한 번도 예측하지 않았다 |
| approach→grasp 경계 (`boundary_delays`) | 40 경계, 놓침 1, 중앙값 0 요청, 앞섬 비율 0.13 |
| VB3 판정 | macro-F1 0.834 ≥ 0.70 ∧ grasp recall 0.851 ≥ 0.6 → **통과** |
| probe | `$R/probe/hsr6_panda_kv_L4.npz` |
| sha256 | `e2bdd1290f5770ce16ad0235ff261a1d343757e37a321047708cd85ded498029` |
| md5 | `d5a43792b74223e00a166e3afef4b31b` |
| class | approach · grasp · lift · transport · placement (`load_hsr6_probe` → `Hsr6SubsetProbe`, absent = retreat) |
| 전체 fit | 62 iter 수렴, 학습 정확도 0.989, 서빙 경로 argmax = fit argmax 873/873 |

전체 기록은 `$R/sl6b_hsr6_panda.json` 이다 (C 별 fold 점수, 경계 지연 전체, episode 사건, 한계).

## 6. lead 가 돌릴 명령

**B5 (GPU, 공유 lock, 873 요청)**. SL4b audit 기준 load 약 35 초, 7 GiB, 요청당 약 60 ms 다.

```bash
cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
python3 scripts/setup/gpu_lock.py shared --est-gib 12 --tag sl6b-noun-rows -- \
  .venv-openpi-live/bin/python benchmark/runners/attention_audit.py --noun-rows \
  --source-run 20261013T092537_714c891415d9_sl4b-dev-pi05only-t0-9e1-4 --run-id sl6b-noun-rows-dev-e1-4
```

출력은 다음과 같다.

- `<B5 run>/noun_rows/libero_spatial_t*_e*_noun.npz` 40 개. `noun_image` 는 (calls, 16, 8, 768) float32 이고 약 340 MB 이다.
- `noun_index.jsonl`, `summary.json`.

먼저 `--dry-run` 으로 873 calls 와 settings 를 볼 수 있다.

**B6 (CPU, `.venv-ag3s`, 약 2 분)**:

```bash
cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
nice -n 10 .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.sl6_head_select \
  --noun-run <B5 run dir> \
  --labels-run /mnt/dev/AG3S_VLA-storage/results/benchmark/20261029T192223_5ea56af44722_sl6b-hsr6-panda-labels \
  --attn-run /mnt/dev/AG3S_VLA-storage/results/benchmark/20261013T094739_714c891415d9_sl4b-attention-audit-dev-e1-4 \
  --out <B5 run dir>/sl6b_head_select.json
# 출처 대조 (선택): --seg-source rollout --out <B5 run dir>/sl6b_head_select_rollout_seg.json
```

`decision.spec` 이 서버 인자 `--attention-source` 값이다 (예: `noun_target:L10H7`).

**서버 인자 (A 의 Panda 최종 구성에 넣을 B 쪽 값)**:

- `--hsr6-probe /mnt/dev/AG3S_VLA-storage/results/benchmark/20261029T192223_5ea56af44722_sl6b-hsr6-panda-labels/probe/hsr6_panda_kv_L4.npz`
- `--attention-source <B6 decision.spec>`

## 7. verifier 가 알아야 할 것

- **새 flag · 기본값** (모두 기본 꺼짐):
  - `NounTokens(rule="fruit")` 기본, `"libero"` 는 opt-in.
  - `phase_export.py --finger-contacts`.
  - `attention_audit.py --noun-rows` / `--noun-layers 2-17`.
  - `load_hsr6_probe` 는 받는 범위만 넓혔다. 6 class 면 같다.
- **재생산 산출물**: B3 phase export 40 개 (`$R/phase/`, 670 초, `.venv-libero`). label · probe · VB2 json (`$R/labels/`, `$R/probe/`, 5 초).
- **옛 기록과 호환**:
  - fruit mode record 키는 그대로다. libero mode record 는 키 4 개를 더한다.
  - `hsr6.py` 는 이제 module import 때 `AG3S.ag3s.stages.subtask_probe` 를 import 한다 (numpy 만 쓰는 module).
- run manifest 에 실행 기록 (`record_execution`, 명령 2 줄 · code md5) 이 있다. `source.patch` 는 run 을 만든 19:21 KST 시점 것이다. 그 뒤 쓴 `sl6_hsr6_panda.py` 의 md5 는 실행 기록에 있다.
- 쓰지 않은 데이터: LIBERO init 0 · SafeLIBERO dev (`load_episodes` · `select` 가 init 1–4 외를 거절한다).

## 8. A 에게 (요청 · 확인)

- A 의 작업 트리 `AG3S/adapters/panda/server.py` `_noun_tokens` 가 `NounTokens.from_config(config, rule="libero")` 를 부른다. 이 서명은 내 최종 서명과 같다. 바꿀 것 없음.
  - 들어오는 libero mode record 의 `attention_source` 에는 `noun_rule` · `destination_noun` · `destination_row` · `noun_pieces` 가 더 붙는다.
- `load_hsr6_probe(5 class)` 는 `Hsr6SubsetProbe` 를 돌려준다. 다음 두 가지가 맞는다.
  - `getattr(probe, "classes")` = 5 개다.
  - `predict()["p"]` 는 6 키다 (retreat 0.0).
- A 가 config dump 에 classes 를 기록하면 retreat 는 `probe.absent` (또는 `record()["phases_absent"]`) 로 보면 된다.
- VB4 parity (서버 HSR6 단계 = offline B4 단계): offline 쪽 숫자는 `$R/labels/request_labels.jsonl` 이 아니라 probe 의 argmax 다. `sl6_hsr6_panda.run` 이 873 feature 에서 서빙 경로 = fit 임을 이미 확인했다. 서버가 같은 kv_L4 를 내면 같아야 한다.

## 9. 한계

- C 고르기와 macro-F1 보고가 같은 5 fold 다 (nested CV 아님). 보고 F1 은 낙관적이다.
- 40 episode 는 libero_spatial 10 task × init 4 다. init 마다 rollout 1 개다. 두 실패 episode 가 drop 이후 approach label 을 많이 만든다 (t6e3 28 요청).
- B3 의 "두 손가락 접촉" 은 MuJoCo 접촉 그대로다. 접촉이 잠깐 끊기면 approach 가 된다 (t7e1 1 요청).
- B6 은 §4 대로 2D 기준만 쓴다. libero_spatial 은 검은 bowl 이 둘이라 명사만으로 구분이 안 될 수 있다.
- 요청 간격은 Panda 0.25 s (RB-Y1 0.53 s) 다. 요청 수로 정한 지연 · 창은 시간으로 절반이다.

## 내가 기대하는 결과

> verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.

- B5 는 873 요청 모두 `noun_ok` 일 것이다 (dev 40 episode 가 모두 libero_spatial 형식이고 `▁bowl` 이 한 번 나온다).
- B6 에서 L10H7 이 RB-Y1 처럼 상위일지는 모르겠다. pi05_libero 는 다른 checkpoint 다. 최고 cell 이 L10H7 보다 2 %p 이상 높아 바뀔 가능성이 크다고 본다.
- identity 정답률은 두 bowl 때문에 0.5 근처의 cell 이 많을 것이다.
- VB2 띠는 grasp 시작이 영상의 손가락 닫힘과 맞을 것이다. 다만 t0e1 의 첫 닫기 (오른손가락만 닿음) 를 approach 로 본 것은 사용자가 grasp 로 볼 수도 있다.
- VB4 parity 는 99 % 이상일 것이다 (서버와 audit 의 kv_L4 가 SL4b 에서 bit 동일했다).
