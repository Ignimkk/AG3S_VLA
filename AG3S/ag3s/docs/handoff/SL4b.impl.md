# SL4b — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 작성 2026-10-13 09:30 KST
> branch `feature/panda-benchmark`, 시작 HEAD `635a9762` (commit 안 함 — 수정 2, 새 파일 4)
> 앞선 implementer 가 API 한도로 중간에 끊긴 작업을 이어받았다. 남은 부분 작업을 처음부터 검토하고 고친 뒤 마쳤다 (아래 "이어받은 작업").

## 무엇을 했나 (평이한 요약 먼저)

SL4 harness(`perception_replay.py`)에 dev calibration 연구용 옵션 넷을 붙였다: audit 진단 grid에서 다른 attention cell을 꺼내 쓰는 `--attention-cell L,H`,
ESDF를 건너뛰고 grounding까지만 기록하는 `--grounding-only`, GT target segmentation을 attention으로 쓰는 privileged 참고 실행 `--attention-source oracle`,
여러 (rollout, audit) run 쌍을 한 번에 받는 반복 `--source-run/--attention-run/--scene-gt`. 기본값은 SL4 경로 그대로이고 `--attention-cell 8,2` 도 SL4 경로다
(CPU test에서 3 call의 모든 기록 배열이, GPU smoke에서 4 call의 grounding·cloud·ESDF 값이 SL4 default run과 bit 단위로 같다).

사전 등록 표(`SL4b.task.md` "사전 등록")는 평가기 `benchmark/evaluators/ats_calibration.py` 에 **문구 그대로**(`PREREGISTRATION_ROWS`, test가 task 파일과 대조)와
숫자 상수로 들어 있고, 표가 말하지 않는 부분의 해석은 `OPERATIONAL_DEFINITIONS` 에 적었다. 둘 다 hash와 함께 모든 SL4b replay run의 manifest(`settings.preregistration`)와
평가기 출력에 기록된다. 평가기는 `select`(선별 지표 → L8H2 + 상위 16 cell)와 `evaluate`(1차·2차 지표, 채택 조건 판정, 원인 분리, oracle 참고)를 한다.

Phase label과 선별 지표·oracle·원인 분리에 쓰는 target segmentation은 SL3 정의를 그대로 쓰도록 했다: init 0은 SL3 verifier의 `gt_` export, init 1–4는 새
`phase_export.py`(py3.8)가 같은 규칙으로 만든다. 이 둘은 같은 episode에서 bit 단위로 같다(t0·t4, init 0). 평가기는 SL4 default run(init 0)에서 SL4 verifier의
call별 label·precision·recall·centroid 오차를 130 call 전부 그대로 재현한다.

## 이어받은 작업 — 유지한 것 / 바꾼 것

| 부분 작업 | 판단 | 무엇을 |
|---|---|---|
| `perception_replay.py` `AttentionSpec`, `episode_attention_grids`, `--attention-cell`, `--cell-from-pooled-f16`, `install_grounding_only`, 반복 `--source-run`/`--scene-gt`, settings 기록 | **유지** (검토 후 test 추가) | 8,2 는 float32 `cell_grid_policy`(SL4가 쓴 배열), 다른 cell 은 `pooled_f16[:, L, H]` |
| oracle attention = rollout 의 `segmentation_geom/<cam>` | **바꿈** | 그 segmentation 은 env.step **반환 시점** 상태로 render 된 것이다(`camera.capture` docstring; SL3: frame 당 100–300 px 다름). RGB·depth·attention 의 순간(image instant)과 맞는 SL3 `segP` 를 `--phases DIR` 에서 읽는다 |
| 선별 지표·원인 분리의 target coverage = `segmentation_geom` | **바꿈** | 위와 같은 이유로 `segP`. SL3 #6 grid 표와 같은 입력이 된다 (smoke: 221 call × 144 cell, SL3 `grid_metrics.npz` 와 차 ≤ 2.2e-16). 반환 시점 segmentation 결과는 `metric_recorded_segmentation` 으로만 보고 |
| `phase_export.py` (render 없음) | **바꿈** | call frame 마다 image-instant segmentation `segP/<cam>` 과 RGB 재render 일치 검사를 추가 (SL3 `export_gt_dev.py` 와 같은 규칙, 같은 key) |
| oracle `attention_invalid` = 어느 camera 든 target pixel 0 | **바꿈** | target 이 안 보이는 camera 의 attention 0 은 정의상 정상 — oracle 은 `attention_invalid` 를 세우지 않고 `oracle_target_pixels` 만 기록 |
| `index_sources` 가 다른 run 을 읽은 audit 행을 조용히 건너뜀 | **바꿈** | 여러 run 을 읽은 audit 은 지원하되, 이 dev run 의 행이 없고 다른 run 의 행만 있으면 SL4 의 명시적 오류("attention run read X, not Y")를 그대로 낸다 |
| `check_dev_split` 이 모든 run 에 적용 | **바꿈** | SL4b 옵션을 쓴 run 에만 강제 (기본 SL4 경로 동작 불변). `data_split` 문구는 init 집합으로 |
| `point_triangles_distance_near` (모든 점×삼각형 AABB) | **바꿈** | 점 KD-tree + 삼각형 외접구로 후보 쌍 → 정확 거리 (같은 값, 평가 43 s → 17 s / 130 call) |
| `ats_calibration.py` 지표·채택·선별·원인 분리 | **유지** + 보완 | `episode_phases` 가 `chunk_t` 대조·`segP` 로드, 원인 분리를 오선택 종류별(`by_majority_kind`)로도, `evaluate` 가 episode 집합 = selection 의 것 검사, `complete_dev_set_50` |
| `libero_scene_gt.py` `closest_points_pairwise`, `exact_distance_near` | **유지** | test: pairwise = all-pairs 결과와 bit 동일 |
| `libero_phases.py` | **유지** + `segmentation_at`, `load_track(chunk_t=, segmentation=)` | |
| test, impl.md | 없었음 → 작성 | |

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/runners/perception_replay.py:53-74` | 모듈 docstring 의 SL4b 옵션 절 | 요청 1 |
| `perception_replay.py:131-182` | `index_sources`: audit 행을 dev run 경로로 고름; 다른 run 의 행만 있으면 명시적 오류; episode 에 `dev_run_dir/attention_run_dir` | 여러 source run |
| `perception_replay.py:185-207` | `index_source_pairs`: 순서대로 짝, stem 중복 거부, 짝 수 불일치 거부; 한 쌍이면 SL4 layout(`dev`, `attention`) 유지 | 요청 1 "여러 source run" |
| `perception_replay.py:210-244` | `index_scene_gt`: 디렉터리 여러 개, stem 은 정확히 한 곳 | init 0 GT(SL4) + init 1–4 GT |
| `perception_replay.py:248-301` | `AttentionSpec(source, cell, from_pooled_f16, mapping)`: `grid_key()` = 8,2 이면 `cell_grid_policy`(float32), 아니면 `pooled_f16`; oracle 은 cell·mirror 거부; `record()` 에 `privileged` | 요청 1 |
| `perception_replay.py:311-342` | `parse_cell`, `episode_attention_grids`: `pooled_f16[:, L, H, block]` → (N,2,16,16) float32; block 합 0/비유한 → `invalid` (grid 는 그대로 사용) | float16 underflow 표시 |
| `perception_replay.py:345-371` | `oracle_target_geoms`, `oracle_track`(phase track + `segP`, geom SHA·`chunk_t` 대조), `oracle_attention`(target pixel 1/0, 256×256, OpenCV) | 요청 1 oracle |
| `perception_replay.py:374-405` | `build_observations(attention=)`: OpenCV 방향 map 을 그대로 부착 (oracle) | |
| `perception_replay.py:536-546, 572-710` | `FrameOptions.grounding_only`; `run_frame`: grounding-only 면 field 질의·slice·정확 구 거리·front-end probe 를 건너뜀, field 가 생기면 오류; GT 가시성(depth 잔차)·target 위치는 기록 | 요청 1 grounding-only |
| `perception_replay.py:722-809` | `run_episode(attention=, phase_dirs=)`: call 별 `attention_grid_policy`, `attention_block_sum`, `attention_invalid` (oracle: `oracle_target_pixels`), episode 별 `attention_source/cell/grid_array`, `grounding_only`, `oracle_segmentation` | |
| `perception_replay.py:821-838` | `install_grounding_only`: 그 AG3S **instance** 의 `_build_esdf` 를 `(None, [note])` 로 (core 파일 무변경) | grounding-only |
| `perception_replay.py:841-920` | `is_sl4b`, `attention_spec`, `episode_inits`, `check_dev_split`(init ∉ 0–4 거부, SL4b 만), `build_settings`: step `SL4b`, `study`, `preregistration`, `privileged_attention`, `options.attention`, `options.phases`, `options.grounding_only`, oracle 이면 `privileged_inputs` = "PRIVILEGED: …" | manifest 기록 |
| `perception_replay.py:956-1063` | `run`: oracle 은 `--phases` 필수, dry-run 이 cell/oracle 을 검사, episode 별 dev run 경로 | |
| `perception_replay.py:1074-1110` | 새 플래그: `--source-run/--attention-run/--scene-gt` 반복, `--attention-cell L,H`, `--cell-from-pooled-f16`, `--attention-source {vla,oracle}`, `--phases DIR`(반복), `--grounding-only` | |
| `benchmark/evaluators/ats_calibration.py` (새) `:53-75` | `PREREGISTRATION_ROWS` (task 표 8 행 문구 그대로), `PREREGISTRATION_SOURCE`(commit `41a5b28a`, 그 뒤 파일 변경 없음), `ADOPTION_RULE_TEXT` | 요청 2 "채택 조건 문구를 코드 상수로" |
| `ats_calibration.py:78-90` | 숫자: task 0–9, init 0–4, phase approach+grasp, L8H2, 16, 30 mm, 10 mm, 가시성 10 mm, gain 1/10, ≥6 task, grid denoise 0 / `last` | |
| `ats_calibration.py:92-136` | `OPERATIONAL_DEFINITIONS` (아래 표) | 표가 말하지 않는 해석을 명시 |
| `ats_calibration.py:143-176` | `preregistration_record()`(행·hash·상수·해석·module SHA), `check_baseline_cell()`(V11AB cell = L8H2·denoise 0·`last` 아니면 오류) | manifest |
| `ats_calibration.py:179-241` | `label_points`, `majority_label`(동률 = 다수 없음), `call_metrics`(category: correct / centroid_far / wrong_instance / no_visible_gt / no_target) | 1차 지표 |
| `ats_calibration.py:249-284` | `aggregate`: 1차(정수 count), task 별, pooled precision, recall median(두 가지), centroid median/p95, category·majority 표, `attention_invalid` 수 | 2차 지표 |
| `ats_calibration.py:287-323` | `adoption_decision`: `10·(c−b) ≥ n`(정수), task 별 `c_t ≥ b_t` 가 ≥ 6, 같은 call 이 아니면 오류; 여럿이면 (−correct, layer, head) | 채택 조건 |
| `ats_calibration.py:326-352` | `selection_mass`, `rank_cells`(L8H2 제외 상위 16, 동률 낮은 layer→head) | 후보 선별 |
| `ats_calibration.py:355-368, 645-681` | `attention_cause`(두 camera raw 값의 joint argmax 가 target 위 → grounding 원인, 아니면 attention 원인; camera 별 flag 도), `cause_split`(+ `by_majority_kind`, `by_task`) | 원인 분리 |
| `ats_calibration.py:411-499` | `select`: 쌍 index → dev split 검사 → phase(`segP` 포함) → 평가 phase call 의 (call, camera) 평균 mass; 결과에 `metric`, `invalid_blocks`, `metric_nanmean_valid_only`(SL3 #6 방식), `metric_recorded_segmentation`, `baseline_metric_from_float32_cell`, `candidates`(+`replay_arg`), `complete_dev_set_50` | |
| `ats_calibration.py:503-642, 684-749` | `read_run`(label: `L{l}H{h}`, f16 L8H2 → `-f16`, mirror → `-mirror`, oracle), `evaluate_run`(scene GT·sample 이 run 의 것과 같은지 검사), `evaluate`(baseline 1 개 = float32 L8H2 un-mirror, 선택된 cell 마다 `pooled_f16` run, 나머지는 reference; 모든 run 의 call key·episode 집합 동일 검사) | |
| `ats_calibration.py:752-800` | CLI `select` / `evaluate` (출력 파일 exclusive 생성; `evaluate` 는 selection 의 `rows_sha256` 이 현재 사전 등록과 다르면 거부) | |
| `benchmark/evaluators/libero_phases.py` (새) | SL3 `phases_of` 규칙(`phase_labels`), `find_track`(`phase_`/`gt_` 정확히 한 파일), `load_track`(geom SHA·`chunk_t` 대조, `segP`), `segmentation_at` | phase 재사용 |
| `benchmark/adapters/libero_panda/phase_export.py` (새, py3.8) | episode 마다 env 재생성 → 모든 frame 의 `sim_state_image` 복원 → target 위치·gripper-target contact(SL3 규칙); call frame 마다 `segP/<cam>` render + RGB 재render 일치 수; `--no-segmentation`; 덮어쓰기 거부 | init 1–4 phase / segmentation |
| `benchmark/evaluators/libero_scene_gt.py:176-249, 420-433` | `closest_points_pairwise`, `point_triangles_distance_near`(KD-tree 후보 → 정확), `SceneGT.exact_distance_near` | 10 mm label 계산 속도 |
| `tests/benchmark/test_panda_sl4b.py` (새, 17 test) | 아래 표 | 요청 3 |

손대지 않은 것: `AG3S/ag3s/**`, `AG3S/trajopt/**` (`git diff --stat HEAD -- AG3S/ag3s AG3S/trajopt` 0 줄, handoff 외 untracked 0), SL3/SL4 verifier script(읽기·import 만), 기존 run.

### 표가 말하지 않는 부분의 해석 (`OPERATIONAL_DEFINITIONS`, manifest 에 기록)

| 문구 | 코드가 읽는 방식 | 근거 |
|---|---|---|
| 평가 call | phase ∈ {approach, grasp} 인 call 전부. target 없는 call·`attention_invalid` call 도 분모에 | "평가 phase call 전체에 대한 비율" |
| 최근접 instance | target 점마다 visual 표면 정확 거리 최소 instance (≤ 10 mm), 아니면 franka 구 표면 ≤ 10 mm → `robot`, 아니면 `none` | SL4 verifier `analyze.py` 의 label 그대로 (재현 확인) |
| 다수결 | count 가 **유일하게** 최대인 label; 1 위 동률은 다수 없음 (→ target_correct 0) | |
| centroid 오차 | ATS centroid − **보이는** GT target sample centroid; 보임 = head 또는 wrist depth 잔차 ≤ 10 mm; 보이는 sample 이 없으면 0 | SL4 `centroid_err_vis` |
| 선별 지표 | `pooled_f16[L,H]` block 정규화 × `segP` target coverage(un-mirror), (평가 call, camera) 평균; float16 underflow block 은 mass 0 | SL3 #6 (SL3 은 underflow 를 nan 으로 뺐다 → `metric_nanmean_valid_only` 로 같이 보고) |
| 채택 동률 | 1차 지표 최고 → 낮은 layer (등록) → 낮은 head (등록 밖, 결정적 완결) | |
| 오선택 | L8H2 call 중 `wrong_instance` (다른 instance·`none`·`robot`·동률); 종류별로도 나눔 | |
| 원인 분리 | replay 가 붙인 두 camera grid(raw)의 joint argmax patch 에 `segP` target coverage > 0 → grounding 원인, 아니면 attention 원인. camera 별 argmax 결과(`alt_either/both`)도 보고 | fusion 이 camera 별 raw attention 을 max 로 비교 (SL4 impl) |

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
OMP_NUM_THREADS=2 .venv-ag3s/bin/python scripts/setup/run_without_legacy.py -m pytest -q -p no:cacheprovider -rs tests/benchmark/test_panda_sl4b.py
# 17 passed (~28 s; phase_export 1 episode 22 s 포함)
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv-openpi-live/bin/python scripts/setup/run_without_legacy.py -m pytest -q -p no:cacheprovider tests/benchmark/test_panda_sl4b.py tests/benchmark/test_panda_sl4.py
# 30 passed in 64.58s   (SL4b 16 개 시점 + SL4 14; evaluate test 추가 전)
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu bash scripts/testing/run_cpu_regression.sh
# == A: 73 passed, 77 warnings in 36.61s
# == B: tests/benchmark (.venv-openpi-live): 109 passed in 202.12s      (SL4 92 + SL4b 17)
# == C: tests/benchmark (.venv-ag3s): 107 passed, 2 skipped in 140.51s  (SL4 90 + SL4b 17; skip 2 = SL3 과 같음)
# == cpu regression exit status: 0
```

| test (`test_panda_sl4b.py`) | 무엇을 |
|---|---|
| `test_preregistration_is_the_task_table_verbatim_and_its_numbers` | task 파일 표를 parse 해 `PREREGISTRATION_ROWS` 와 문자 단위 대조, 상수, record 내용, V11AB cell = L8H2 |
| `test_parse_cell_and_attention_spec_rules` | `L,H` parse·거부, 8,2 = default grid(`cell_grid_policy`), f16/다른 cell = `pooled_f16`, oracle 규칙·privileged, CLI → spec, `is_sl4b` |
| `test_episode_attention_grids_cell_blocks_and_underflow` | 합성 grid: default·8,2 → `cell_grid_policy` 와 bit 동일; 다른 cell 의 block 위치; underflow block → invalid, 0 grid 그대로; 범위 밖·grid 없음 거부 |
| `test_several_source_runs_are_paired_in_order` | 짝 순서·layout, stem 중복·짝 수 거부, dev split 거부(init 5), scene GT 여러 디렉터리(없음/중복 거부) |
| `test_phase_labels_rule` | 접촉 없음 / grasp / lift(> z0+10 mm, = 는 lift 아님) / transport / post_release; SL3 verifier `phases_of` 를 import 해 같은 label |
| `test_phase_export_is_python38_and_uses_the_sl3_contact_rule` | py3.8 parse, contact 규칙(gripper·robot·any 수) |
| `test_phase_labels_of_init0_equal_the_sl3_series` (storage, SL3 GT) | t0·t4·t9 init 0 label = SL3 `series.json`, `segmentation_at` |
| `test_point_labels_majority_and_call_metric_categories` | label(10 mm 경계, robot), 다수결 동률, category 다섯, centroid 29.3/30.1 mm, aggregate(분모에 no-target 포함, precision pooled) |
| `test_adoption_rule_boundaries_and_tie_break` | 정확히 10 %p 통과 / 9 %p 실패, 6 task 통과 / 5 task 실패, 동률 → 낮은 layer, 더 높은 1차 지표 우선, 다른 call 거부 |
| `test_selection_mass_ranking_and_cause_split` | block 정규화 mass, underflow → 0·invalid, 순위(L8H2 제외, 동률 순서), joint argmax 원인, `by_majority_kind` |
| `test_pairwise_closest_points_and_near_distance_are_exact` | pairwise = all-pairs (bit), near 거리 = 정확 거리 (≤ r, `pair_budget` 97 로 chunk 경계 포함) |
| `test_v11ab_cell_options_read_the_sl4_grid_and_dry_run` (storage) | 실제 audit t0: default = 8,2 = `cell_grid_policy`, f16 차 < 1e-6; `--attention-cell 8,2 --grounding-only --dry-run` settings (step SL4b, prereg hash, grid, grounding-only, audit opencv 일치) |
| `test_phase_export_reproduces_the_sl3_export` (storage + libero) | t0 export: RGB 재render 21/21, `segP` shape, SHA·`chunk_t` 대조 오류, 덮어쓰기 거부, SL3 `gt_` 파일과 6 배열 bit 동일 |
| `test_oracle_dry_run_is_privileged` (storage + libero) | `--phases` 없으면 거부; settings privileged·condition·"PRIVILEGED", target pixel > 0, segmentation SHA |
| `test_cell_8_2_is_the_sl4_path_and_grounding_only_equals_the_full_grounding` (storage, CPU) | t0 처음 3 call 을 CPU(legacy ESDF, test 전용 override)로 세 번: default / `cell=(8,2)` / default + grounding-only. 8,2 = default 의 **모든** 배열(시간 제외) bit 동일; grounding-only = full 의 grounding·seed·manipulated·cloud·구 중심 bit 동일, field 질의 없음, note |
| `test_oracle_attention_reaches_ag3s_at_full_resolution` (storage + libero, CPU) | 256×256 oracle map 이 AG3S 에 들어가 두 camera attention 으로 fusion, 기록(`oracle_target_pixels`, segmentation SHA, invalid 없음) |
| `test_evaluate_roles_adoption_and_cause_split` | 합성 per-call 행으로 `evaluate`: baseline/candidate/reference(privileged) 역할, 채택, 원인 분리, 누락 cell·baseline 없음·episode 집합 불일치 거부 |

### Smoke (결과 수치는 보고하지 않음 — 일치·동작만)

| smoke | 결과 |
|---|---|
| `phase_export.py` t0·t4 (init 0, CPU, `.venv-libero`) | 42 s(≈ 20 s/episode). 10 배열(`target_xpos`, contact 4 종, `segP` 2, `call_frames`, `chunk_t`, `target_geoms`) 이 SL3 `gt_` 파일과 bit 동일; RGB 재render 21/21·24/24; segP-vs-recorded px 합이 SL3 `gt_checks.json` 과 같음 |
| 선별 지표 (init 0, CPU) | phase 10/10 episode = SL3 `series.json`; L8H2 float32 mass − SL3 series ≤ 1.1e-16; 144 cell mass − SL3 `grid_metrics.npz` ≤ 2.2e-16 (nan 위치 동일) |
| `ats_calibration select` CLI (init 0 만, scratchpad) | 17 후보(L8H2 + 16), grid 18×8, `complete_dev_set_50` False(정상 — init 0 만), 다시 쓰기 거부 |
| `evaluate_run` on SL4 default run (init 0, CPU) | 평가 phase 130 call: SL4 verifier `analysis/default/frames.json` 과 label count·precision·recall·centroid·phase 불일치 **0**; 17 s |
| GPU smoke (lock 00:11:20–00:12:02 UTC, 시작 시 compute apps 없음, scratchpad storage, main 에 시작·끝 알림) | t0 4 call: `--attention-cell 8,2 --grounding-only` 와 `--attention-cell 8,2`(full) 의 grounding·seed·cloud·구 중심이 SL4 default run 과 bit 동일, full 의 `sph_esdf_main`·`gt_esdf_main`·`slice_xy_main` 도 차 0.0. grounding-only `t_process` p50 45 ms (full 92 ms), 첫 call 0.8 s (full 7.6 s). oracle(2 call) privileged, has_target 2/2; `3,1`(2 call) `pooled_f16`, invalid 없음 |

## verifier 가 알아야 할 것

- **새 플래그**(harness, 모두 기본값 = SL4 경로): `--attention-cell L,H`, `--cell-from-pooled-f16`(정밀도 대조), `--attention-source {vla,oracle}`, `--phases DIR`(oracle 용, 반복),
  `--grounding-only`, 반복 가능해진 `--source-run`/`--attention-run`(순서대로 짝)/`--scene-gt`. 새 도구: `phase_export.py`(`--run --out [--task-ids] [--no-segmentation]`),
  `python -m benchmark.evaluators.ats_calibration {select,evaluate}`.
- **기록되는 것**: SL4b 옵션이 하나라도 있으면 manifest `settings.step = "SL4b"`, `settings.preregistration`(문구·hash·상수·해석), `options.attention`(source·cell·grid 배열·정밀도·privileged), oracle 이면 `privileged_attention: true` 와 `privileged_inputs: "PRIVILEGED: …"`.
  episode npz 에 `attention_grid_policy`(oracle 제외), `attention_block_sum`, `attention_invalid`, `attention_source/cell/grid_array`, `grounding_only`, oracle 이면 `oracle_target_pixels`, `oracle_segmentation`.
- **정밀도 비대칭**: 등록 규칙상 L8H2 는 SL4 의 float32 read-out(`cell_grid_policy`), 후보 cell 은 float16 `pooled_f16` 이다 (task: "기본은 L8H2 — 기존과 bit 동일"). init 0 smoke 에서 L8H2 의 선별 지표가 f16 와 f32 사이에서 7.5e-4 다르다 (block 이 작을 때 f16 상대오차). 이 효과를 재려면 `--attention-cell 8,2 --cell-from-pooled-f16 --grounding-only` run 을 reference(`L8H2-f16`)로 같이 `evaluate` 에 넣는다 — 후보가 아니다.
- **segmentation 출처**: init 0 = `outputs/verify/SL3/raw/gt` (`gt_*.npz`), init 1–4 = `phase_export.py` 출력. 둘 다 `--phases` 로 준다. rollout 의 `segmentation_geom` 은 쓰지 않는다 (반환 시점 render).
- **`evaluate` 입력 조건**: L8H2 baseline run 은 정확히 하나(float32 un-mirror, 50 episode — SL4 default run 은 init 0 만이라 쓰면 episode 불일치로 거부된다), selection 의 16 cell 마다 `pooled_f16` run 하나, 나머지(oracle, `L8H2-f16`)는 reference. 모든 run 의 평가 call 이 같아야 한다.
- **시간 추정**: grounding-only replay ≈ 5–6 s / episode (GPU smoke 기준) → 50 episode ≈ 5 min / run, 18 run ≈ 1.5 h (한 lock 안에서 연달아 돌려도 됨). `phase_export` ≈ 20 s / episode (CPU). `evaluate` ≈ 1.3 s / 10 평가 call / run → 18 run ≈ 25–30 min (CPU).
- **attention audit**: init 1–4 audit 은 기본값(`pooled_f16` 저장)으로 돌린다. `--no-full-grid` 면 `--attention-cell` 이 거부한다.
- **실행 명령 (순서)**:
  ```bash
  cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
  LOCK=/tmp/ag3s-vla-gpu-validation.lock
  DEV0=20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0      # init 0 (SL3)
  AUD0=20261012T202459_309e4c6da477_sl3-attention-audit-dev       # its audit (SL3)
  V=outputs/verify/SL4b
  # 1. dev rollout init 1-4 (GPU)
  ( flock -w 1800 9 || exit 1; nvidia-smi --query-compute-apps=pid --format=csv,noheader
    .venv-ag3s/bin/python benchmark/runners/launch_libero_pi05_only.py --task-ids 0,1,2,3,4,5,6,7,8,9 --episode-ids 1,2,3,4 \
        --capture-geometry --capture-segmentation --run-id sl4b-dev-pi05only-t0-9e1-4 ) 9>$LOCK
  DEV1=<run printed above>
  # 2. attention audit of DEV1 (GPU)
  ( flock -w 1800 9 || exit 1; nvidia-smi --query-compute-apps=pid --format=csv,noheader
    .venv-openpi-live/bin/python benchmark/runners/attention_audit.py --source-run $DEV1 --run-id sl4b-attention-audit-dev-e1-4 ) 9>$LOCK
  AUD1=<run printed above>
  # 3. scene GT + phase / segmentation for init 1-4 (CPU, .venv-libero)
  .venv-libero/bin/python benchmark/adapters/libero_panda/scene_export.py --run $AG3S_STORAGE_ROOT/results/benchmark/$DEV1 --out $V/raw/scene
  .venv-libero/bin/python benchmark/adapters/libero_panda/phase_export.py --run $AG3S_STORAGE_ROOT/results/benchmark/$DEV1 --out $V/raw/phase
  PAIRS="--source-run $DEV0 --attention-run $AUD0 --source-run $DEV1 --attention-run $AUD1"
  GT="--scene-gt outputs/verify/SL4/raw/scene --scene-gt $V/raw/scene"
  PH="--phases outputs/verify/SL3/raw/gt --phases $V/raw/phase"
  # 4. candidate selection (CPU)
  .venv-ag3s/bin/python -m benchmark.evaluators.ats_calibration select $PAIRS $PH --out $V/select.json
  # (dry-run check, CPU) .venv-openpi-live/bin/python benchmark/runners/perception_replay.py $PAIRS $GT --attention-cell 8,2 --grounding-only --dry-run
  # 5. grounding-only replays (GPU): L8H2 + the 16 cells of select.json (candidates[*].replay_arg) + oracle (+ L8H2-f16 control)
  ( flock -w 1800 9 || exit 1; nvidia-smi --query-compute-apps=pid --format=csv,noheader
    for C in 8,2 <16 cells from select.json>; do
      .venv-openpi-live/bin/python benchmark/runners/perception_replay.py $PAIRS $GT --grounding-only --attention-cell $C \
          --run-id sl4b-gonly-L${C%,*}H${C#*,} || exit 1
    done
    .venv-openpi-live/bin/python benchmark/runners/perception_replay.py $PAIRS $GT $PH --grounding-only --attention-source oracle --run-id sl4b-gonly-oracle
    .venv-openpi-live/bin/python benchmark/runners/perception_replay.py $PAIRS $GT --grounding-only --attention-cell 8,2 --cell-from-pooled-f16 --run-id sl4b-gonly-L8H2-f16
  ) 9>$LOCK
  # 6. evaluation (CPU): pre-registered metric, adoption, cause split, oracle reference
  .venv-ag3s/bin/python -m benchmark.evaluators.ats_calibration evaluate --selection $V/select.json \
      --run <L8H2 run> --run <16 candidate runs ...> --run <oracle run> --run <L8H2-f16 run> $PH \
      --out $V/evaluate.json --rows-out $V/rows.jsonl
  # 7. (if adoption.selected_by_rule is not null) full replay with that cell, ESDF included (GPU):
  #    perception_replay.py $PAIRS $GT --attention-cell <L,H> --run-id sl4b-full-<cell>   (and the same with --attention-cell 8,2)
  # 8. regression: CPU suite; --attention-cell 8,2 grounding = SL4 default grounding (compare the init-0 episodes of the
  #    step-5 L8H2 run with 20261012T225820_46a08d6ff644_sl4-perception-dev, keys as in the CPU test GROUNDING list)
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu bash scripts/testing/run_cpu_regression.sh
  ```
  stdout 첫 줄이 새 run 경로. harness 는 `nvidia-smi` 에 compute process 가 있으면 시작하지 않는다 (`--allow-busy-gpu`). `select`/`evaluate` 출력은 exclusive 생성.
- **재생산이 필요한 산출물**: init 1–4 rollout·audit·scene GT·phase (위 1–3). init 0 은 기존 SL3 run·SL3 `gt_`·SL4 scene 재사용.
- **옛 기록과 호환이 깨지는가**: 없음. 기본 실행은 SL4 와 같은 step `SL4`·같은 배열(새 배열이 추가될 뿐), SL4 run 은 `read_run` 이 L8H2 float32 로 읽는다.
- **AG3S 내부 속성**: `install_grounding_only` 가 instance 의 `_build_esdf` 를 바꾼다 (쓰기는 그 instance 에만). grounding 이 ESDF 산출물을 읽지 않는 근거: field 는 `_last_esdf_field` 와 `_last_object_fit/_points` 로만 남고 둘 다 grasp latch 뒤 attach 경로(`pipeline.py:861, 926`)에서만 읽힌다 — harness 는 latch 하지 않는다. test·GPU smoke 가 같음을 확인.
- **offline 한계** (SL4 와 같음): execution path 없음, grasp latch·attach 없음 → grasp phase 이후 target 은 쥐지 않은 물체. 평가 phase 가 approach + grasp 인 이유.

## SL4b fix — reset history (2026-10-13 KST, verifier finding 대응)

**평이한 요약.** LIBERO 는 `env.reset()` 때마다 fixture(`wooden_cabinet_1`, `flat_stove_1`)의 위치를 global numpy RNG 로 다시 뽑는다.
fixture 는 joint 가 없어 기록된 sim state 에 들어 있지 않다. rollout client 는 task 마다 env 를 하나 만들고 episode 마다 reset 한다.
그래서 같은 task 의 k 번째 episode 는 k 번째 reset 의 fixture 를 본다. 예전 export 는 episode 마다 새 env 에서 reset 을 한 번만 했다.
이 때문에 각 task 의 첫 episode(init 1–4 run 에서는 e1)만 맞았다. 이제 export 는 run 에 기록된 순서대로 env history 를 재현한다.
재현은 task 마다 env 하나, 앞선 episode 마다 `reset` + `set_init_state` 이다. 내보내는 모든 call frame 은 두 camera 모두 RGB 재render 가 기록과
**bit 단위로 같아야** 하며, 다르면 그 episode 는 예외로 멈추고 파일을 쓰지 않는다.

**코드 근거.** `env.seed` = `np.random.seed` (`benchmark/LIBERO/libero/libero/envs/bddl_base_domain.py:162-163`). hard reset 때 fixture
`body_pos` 를 다시 뽑는다 (`bddl_base_domain.py:776`). client 는 task 마다 env 하나를 만들고 episode 마다 reset 한다 (`libero_pi05_client.py:558`, `:332`).
robosuite step 경로는 기본 설정(identity corrupter, delay 0)에서 RNG 를 쓰지 않는다. hard reset 은 sim 을 새로 만들므로 다음 episode 로
넘어가는 것은 reset 횟수뿐이다. 이것은 아래 873/873 결과가 실측으로 확인한다.

| 파일 | 무엇을 |
|---|---|
| `benchmark/runners/libero_pi05_client.py` | **동작 불변 추출**: `make_task_env` (`:187`, env 생성 + seed + capture observable), `start_episode` (`:199`, reset → `set_init_state`). `run_tasks`·`run_episode` 가 이 둘을 부른다. `run_tasks(..., prepare_episode=None)` hook 추가 (`:564`; client 는 None = 예전 loop 그대로) |
| `benchmark/adapters/libero_panda/env_history.py` (새, py3.8) | `history_blocks`/`run_history` (`:51`): `episodes.jsonl` 줄 순서 = manifest `protocol.task_ids × episode_ids` 를 확인하고, `started_at` 단조·block 별 suite/seed/geometry 단일을 검사한다. 하나라도 어긋나면 **거부**. `plan` (`:98`), `episode_envs` (`:133`, client 의 `make_task_env`/`start_episode` 를 그대로 호출), `rgb_rerender_equal` (`:155`), `require_rgb_equal` (`:166`) |
| `scene_export.py` (SCHEMA 1→2) | `episode_envs` 로 env 를 받는다 (`:253`). 내보내는 call frame 마다 RGB 2 camera 를 대조하고 불일치하면 raise (`:207`). `checks.rgb_rerender_equal_call_frames`, provenance `env_history` (resets_before, preceding_episodes) |
| `phase_export.py` (SCHEMA 2→3) | 위와 같다 (`:217`, `:169`). RGB 대조는 `--no-segmentation` 이어도 항상 한다. provenance `env_history`, `rgb_rerender_equal_call_frames` |
| `benchmark/runners/libero_replay.py` | `source_history_resets` (`:121`): replay 하지 않는 앞선 source episode 를 `prepare_episode` 로 `start_episode` 한다 (`:222`). source 순서의 부분열이 아니면 거부한다. jsonl `replay.source_history_resets` 에 기록 |
| `benchmark/runners/launch_libero_replay.py` | dry-run/manifest 에 episode 별 `source_history_resets` 를 넣는다 (잘못된 순서는 run 할당 전에 거부) |
| `tests/benchmark/test_panda_sl4b_history.py` (새, 10 test), `libero_py38_history_checks.py` (새) | 아래. `test_libero_sl1.py` fake manifest 에 `task_ids/episode_ids` 추가 (실제 manifest 는 모두 가짐) |

**재현 증거 (scratchpad, CPU, export 는 한 번에 하나, peak RSS ≤ 2.78 GB).**

| 대상 | episode | call frame | head RGB 같음 | left_wrist RGB 같음 |
|---|---|---|---|---|
| `phase_export` — SL4b dev run (init 1–4) | 40 | 873 | **873/873** | **873/873** |
| `scene_export` — SL4b dev run (init 1–4) | 40 | 873 | **873/873** | **873/873** |
| `phase_export` — SL3 dev run (init 0) | 10 | 214 | 214/214 | 214/214 |
| `scene_export` — SL3 dev run (init 0) | 10 | 214 | 214/214 | 214/214 |

| 비교 (배열 단위, `provenance_json` 제외) | 결과 |
|---|---|
| init 0 scene 새 export vs `outputs/verify/SL4/raw/scene` | 10/10 episode, 37 배열 모두 bit 동일 |
| init 0 phase 새 export vs `outputs/verify/SL3/raw/gt` | 10/10 episode, 공통 10 배열(`segP` 포함) 모두 bit 동일 |
| init 1–4 새 export vs 기존 `outputs/verify/SL4b/raw/{scene,phase}` | e1 10/10 배열 동일. e2–e4 30/30: scene 은 **fixture body 10 개만** 이동 (stove·cabinet root, 1.95–16.07 mm). 자유 물체·robot 은 같다. phase 는 `segP/*` 30/30 다르고 `contact_target_any` 가 t4_e2 1 개 다르다. **phase label 입력(`target_xpos`, `contact_grip_target`)은 40/40 동일** |
| 실제 action replay (scratch storage, `libero_replay.py`) | SL0 t0e0 replay npz = SL0 rollout npz 14/14 배열 (client 추출이 bit 불변). SL4b t0 **e3 만** replay (source resets [1, 2]) = rollout e3 14/14 배열. 같은 replay 를 history 없이(예전 동작) 돌리면 두 image 배열이 다르고 agentview 0/89 frame 만 같다 |

**libero_replay 영향.** 영향을 받았다 (잠재적). replay 의 task/episode 목록이 source 와 같으면 history 가 같지만, source 의 일부(예: e3 만)를 replay 하면
fixture 가 달랐다. 기존 replay run(SL1 3 개, SL3 `replay-off`)은 모두 source 가 task 당 episode 하나라 해당 없고 결과는 그대로다.

**단위 검증.** `test_panda_sl4b_history.py` 10 passed (.venv-openpi-live·.venv-ag3s 모두). 내용:
- 순서 규칙(거부 6 종), plan, replay 부분열 규칙, RGB 검사 raise, py3.8 문법.
- fake RNG env 로 rollout(`run_tasks` 2 task × e1–3) → `episode_envs` 전체와 부분 → fresh env(예전 동작은 e1 만 맞음) → `prepare_episode` replay.
- 실제 LIBERO t0 e1·e3: history 일치, fresh e3 불일치, replay e3 단독 일치.

CPU regression (`run_cpu_regression.sh` 의 세 suite 를 같은 명령으로 따로 실행): **A 73 passed / B 119 passed / C 117 passed + 2 skip**.
B·C 는 위 10 개가 늘었고, SL0/SL1 replay·`run_tasks` 동등성 test 와 SL3 export bit 동일 test 를 포함한다.

**verifier 가 알아야 할 것.**
- `outputs/verify/SL4b/raw/scene`·`raw/phase` 의 init 1–4 파일은 **다시 만들어야 한다**. e2–e4 30 episode 는 fixture 가 틀렸다. e1 은 배열이 같지만 schema·provenance 가 바뀌었다.
  명령은 위 3 단계와 같다. export 는 RSS ≈ 2.7 GB 이므로 한 번에 하나씩 돌린다. 시간은 40 episode 기준 scene ≈ 12.5 min, phase ≈ 14 min (CPU, 이번 실측).
- 그 scene/phase 를 읽은 산출물(선별 지표의 target coverage, oracle attention, grounding-only replay 의 GT 대조, 평가)은 e2–e4 에서 영향을 받는다.
  phase label 은 바뀌지 않는다 (`target_xpos`·`contact_grip_target` 40/40 동일).
- 이번 수정은 `perception_replay.py`·`ats_calibration.py` 를 건드리지 않았다. 출력 key 는 늘기만 했다 (`checks.rgb_rerender_*`, provenance `env_history`).

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다 -->
- 회귀: A 73 / B 109 / C 107 + 2 skip. `--attention-cell 8,2` grounding-only run 의 init 0 grounding 이 SL4 default run 과 bit 동일 (GPU smoke t0 4 call 에서 그랬다).
- 선별: 상위 cell 은 L8H2 와 같은 중간 layer 대에 몰리고 L8H2 자신도 전체 144 중 상위권일 것. underflow block 은 깊은 layer 에 몰려 그쪽 cell 이 지표에서 밀린다.
- 1차 지표: L8H2 의 오선택은 SL4 와 같이 t4(plate)·t5(bowl_2)·t9 근처 task 에 몰릴 것. 다른 cell 이 t5 처럼 같은 모양 bowl 을 구분해야 하는 task 에서 크게 나아지기는 어렵고, ≥ 10 %p·≥ 6 task 를 동시에 넘는 cell 은 없거나 하나일 것으로 본다.
- 원인 분리: 오선택 중 상당 부분이 attention argmax 가 이미 다른 물체 위에 있는 경우(attention 원인)일 것 — 특히 head camera 가 seed 를 주도하는 접근 초반.
- oracle: 1차 지표가 모든 후보보다 높고, 남는 실패는 target 이 거의 안 보이는 call(no_visible_gt / no_target)과 grasp 중 gripper 에 가린 call 일 것.
- L8H2-f16 reference 는 L8H2 와 1차 지표가 거의 같을 것 (정밀도 효과 작음).
