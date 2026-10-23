# BASELINE-EVAL (F) — 구현: 지표 registry · grounding 고유 지표 · ATS cloud 공정 비교 · methods offline 평가

> writer: ag3s-implementer (Implementer F) · 읽는 쪽: verifier V1, scribe, lead · 작성 2026-10-23 KST
> checkout `/mnt/dev/AG3S_VLA-benchmark`, branch `feature/panda-benchmark`, HEAD `5752c64c`. commit 하지 않았다. GPU 를 쓰지 않았다.

## 무엇을 했나 (평이한 요약 먼저)

각 baseline 논문과 benchmark 가 쓰는 지표를 primary source (논문 절 · 표, 공식 code 줄) 에서 확인했다. 그 결과를
`docs/metrics.md` 에 registry 로 정리했다. 항목마다 지금 잴 수 있는지 (`offline-now` / `closed-loop` / `not-applicable`) 를 적었다.

새 도구 세 개가 지금 잴 수 있는 것을 계산한다. 모두 CPU 로 돈다.

1. **`eval/native_metrics.py`**: SL3 run 의 저장 mask 에서 각 논문 고유의 2D 지표를 사후 계산한다.
   - 지표: Acc@0.5, CorLoc, pointing game, mIoU / oIoU(cIoU), maxFβ / IoU / Acc, J / F / J&F, detector conf sweep.
   - J / F 는 pinned Grounded-SAM-2 의 `sav_benchmark.py` 와, saliency 지표는 pinned TokenCut 의 `metric.py` 와 같은 값이 나오는지 test 로 확인했다.
2. **`eval/cloud_labeling.py`**: ATS 와 segmenter 를 **V11AB 가 그 call 에 만든 같은 fused / filtered cloud 점** 위에서 비교한다. 같은 점마다 GT / ATS / segmenter label 을 매기고, point IoU · P · R · neighbour 비율과 ATS 대비 paired bootstrap 을 낸다.
   - SL4 perception run 이 `cloud_points` 를 기록해 두었으므로 대안 (`--no-clouds` 의 경우) 은 쓰지 않았다. 출력만 준비했다.
3. **`methods/evaluate_offline.py`**: SL3 dev 의 기록된 π0.5 chunk 를 9 개 arm (7 method, AEGIS 3 변형) 에 넣고, call 마다 offline 지표를 계산한다. 계산은 두 venv 로 나눈다 (filter 는 cvxpy, guidance 는 JAX).
   - 지표: 개입률, ‖Δa‖, 자기 barrier / cost, QP 실패, 계산 시간, **scene GT 기준 EE / link clearance (모든 method 에 같은 경로 decode · IK · sphere 모델)**.
   - 핵심 기능 figure 용 npz 를 남긴다: ellipsoid, h trace, path 전 / 후, denoising step 별 수정량, featured call 의 field 단면.

1 episode smoke 까지 했다. 10 episode 전체 계산은 V1 이 아래 명령으로 한다.

## 바뀐 파일 (모두 새 파일)

| 파일 | 무엇이 | 왜 |
|---|---|---|
| `Baseline/paper_baselines/docs/metrics.md` | 지표 registry: 대상 21 행 (AG3S, grounding 6, ATS, method 7, benchmark 3, pointing game). 항목마다 정의 · 출처 (절 · 표 · code 줄) · status · 우리 대응 (tool → summary key). lead 초안과 달라진 점 6 개 | task F-1 |
| `Baseline/paper_baselines/eval/native_metrics.py` | 지표 함수 (`mask_box` `box_iou` `pointing_hit` `precision_recall_tokencut` `f_beta` `threshold_pr` `max_f_beta` `davis_j` `seg2bmap` `davis_f` `db_statistics` `conf_sweep`), 집계 (`bootstrap_ratio` = episode resample 의 pooled 비율 CI), loader (`load_mask_npz`, `read_frame_rows`, `EpisodeGT`, `SaliencyMaps` = LH 결합 map · V11AB L8H2 cell 을 audit 기록에서 재구성), figure dump (`TokenCutRecompute` = DINO CPU 재계산 eigvec · bipartition · seed), CLI | task F-2 |
| `Baseline/paper_baselines/eval/cloud_labeling.py` | `mask_positive` (투영 + visibility `observed`/`depth`/`any`), `gt_classes` (`ats_calibration.label_points` 재사용, 10 mm), `point_metrics`, `summarize` (paired vs ATS), `PerceptionEpisode` (cloud 미기록이면 `LookupError` → 대안 출력 후 exit 2), CLI | task F-3 |
| `Baseline/paper_baselines/methods/evaluate_offline.py` | `filters` / `guidance` / `summarize` 하위 명령. `GTClearance` (scene GT 정확 거리 + 충돌 primitive 부호, DLS IK, 61 sphere), `osc_path`, `surrogate_sample_traced` (D 의 대체 sampler + step 별 보정 기록, bit-equal 확인), `own_costs` (jit vmap), `scene_sdf_metrics` (D 의 `path_metrics` 를 고정 padding 으로 jit), `field_slices` (SDF · OmniGuide energy · NUDGE cost · CFM h · Lan push), dump writer | task F-4 |
| `Baseline/paper_baselines/tests/test_native_metrics.py` | 15 test (아래) | task F-5 |
| `Baseline/paper_baselines/tests/test_cloud_labeling.py` | 7 test | task F-5 |
| `Baseline/paper_baselines/tests/test_evaluate_offline.py` | 11 test (JAX test 1 개는 `.venv-openpi-live` 에서만) | task F-5 |

수정하지 않은 것: `eval/offline_grounding.py`, `providers/grounded_sam2.py`, `lift.py`, `eval/metrics.py` (G 가 수정 중),
`methods/<name>/**`, `AG3S/**`, `benchmark/**`, `third_party/**`. 설치한 package 없음.

### 새 파일이 기존 code 에서 import 하는 것 (G 의 변경이 깨면 안 되는 API)

| module | 이름 |
|---|---|
| `eval/offline_grounding.py` | `Episode` (`.row`, `.seg`, `.lut`, `.raw`, `.depth_mm`, `.K`, `.T`, `.target`, `.eval_frames`, `.stem`), `AttentionRun.aux`, `load_episodes`, `seg_labels`, `DEFAULT_LH_SELECTION` |
| `eval/metrics.py` | `CLASSES`, `aggregate`, `episode_means`, `paired_bootstrap`, `found_rate` |
| `lift.py` | `project_points`, `opencv_rgb_from_raw` |
| `providers/localization_heads.py` · `tokencut.py` | `LocalizationHeadsProvider.combined_map`, `upscale_bilinear`, `to_image_orientation`; `TokenCutProvider._features`, `ncut`, `patch_mask_to_pixels` |
| `benchmark/evaluators/ats_calibration.py` · `libero_scene_gt.py` | `label_points`; `SceneGT` (`exact_distance_near`, `inside`, `frame_row`, `instances`) |
| methods (C / D) | `get_method`, `validate_filter_result`, `safety_common.load_recorded_call` · `rollout_positions`, `guidance_common.recorded.load_recorded_call`, `core.surrogate_sample` · `chunk_to_model_space`, `evaluation.injected_obs` · `SCENE_KW`, `geometry.signed_distance` |

## 단위 검증

```bash
source Baseline/paper_baselines/setup/paper_baselines_env.sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=$PWD .venv-baselines/bin/python -m pytest -q -p no:cacheprovider \
  Baseline/paper_baselines/tests/test_native_metrics.py Baseline/paper_baselines/tests/test_cloud_labeling.py \
  Baseline/paper_baselines/tests/test_evaluate_offline.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=$PWD .venv-openpi-live/bin/python -m pytest -q \
  -p no:cacheprovider Baseline/paper_baselines/tests/test_evaluate_offline.py
```
- `.venv-baselines`: **32 passed, 1 skipped** (skip = JAX test).
- `.venv-openpi-live`: `test_evaluate_offline.py` **11 passed**.
- 기존 test 는 고치지 않았다.
- G 의 변경 (`offline_grounding.py` · `metrics.py` · `lift.py`, diff 3 파일 +367/−95 줄; `BASELINE-EVAL-G.impl.md` 가 생긴 뒤의 working tree) 위에서 두 명령을 다시 돌려 같은 결과 (32 passed 1 skipped / 11 passed) 를 얻었다. G 가 더 고치면 V1 실행 전에 다시 돌린다.

| test | 무엇을 보나 |
|---|---|
| `test_davis_parity_with_sam2_benchmark` | 무작위 mask 15 쌍에서 J · F · `seg2bmap` 이 pinned `sav_benchmark.Evaluator` (cv2 + skimage) 와 1e-12 안에서 같다 |
| `test_tokencut_metric_parity` | maxFβ · Acc · IoU 가 pinned TokenCut `metric.metrics` (torch) 와 1e-6 안에서 같다 |
| `test_db_statistics` | davis2017 decay bin 경계 (8 frame → [0:3], [5:8]) |
| `test_pointing_game_tolerance`, `test_mask_box_and_iou`, `test_fbeta_conventions`, `test_binary_map_maxf_equals_mean_f`, `test_sample_row_and_summary`, `test_conf_sweep`, … | 손 계산 사례 (CorLoc 이 distractor 물체도 맞음으로 세는 것 포함) |
| `test_recorded_episode_matches_harness_2d_iou` | 실제 SL3 t0_e0 2 frame: 우리 IoU 가 harness `frames.jsonl` 의 `2d_iou` 와 같다 (차이 0); gt_oracle 행 = 1 |
| `test_mask_positive_projection_and_visibility` | 투영 pixel, camera bit, depth occlusion, 카메라 뒤 점 |
| `test_recorded_calls_gt_oracle_agrees` | 실제 SL4 cloud + SL3 GT mask 2 call: GT mask 투영의 point recall > 0.9, precision > 0.8 |
| `test_gt_clearance_ik_and_sign` | 실제 scene GT: IK 가 경로를 1 mm 안에서 따라감; cookies 충돌 상자 중심의 거리가 음수 |
| `test_fast_clearance_equals_exact_below_cap` | 실제 scene GT 의 instance 주변 무작위 점 수백 개: pruned 거리 = `exact_distance_near` (bit 단위), 상한 밖은 inf |
| `test_traced_sampler_bit_equal_to_surrogate` | trace 판 sampler 가 D 의 `surrogate_sample` 과 bit 단위로 같다 (tracking / constant / guidance 없음) |
| `test_summarize_family_and_multilink_cases` | multi-link 은 개입하고 AEGIS 는 개입하지 않은 call 의 목록과 binding body |

## 1 episode smoke (CPU, 한 번에 process 하나)

`libero_spatial_t0_e0` (21 call · 21 평가 frame), 출력 `outputs/baseline_eval_smoke/1ep/` (git 추적 안 함).
smoke 명령은 아래 V1 명령에 `--episodes libero_spatial_t0_e0 --out-dir outputs/...` 를 붙인 것이다.

| 도구 | wall | peak RSS | 확인한 것 |
|---|---|---|---|
| native_metrics | 7 s | 0.74 GB | `check_vs_harness_2d_iou`: 504 sample, 최대 차이 0. TokenCut DINO CPU 재계산 mask = 저장 mask (4 / 4). 12 mask arm + 연속 map 2 arm 모두 행 생성 |
| cloud_labeling | 9 s | 0.19 GB | 14 arm (ATS, ATS seed, mask 12) × 21 call. `paired_vs_ats` 13 개 |
| evaluate_offline filters | 97 s | 2.6 GB (GDINO tiny) | 5 arm × 21 call. contract 위반 0, gripper 불변. dump 5 개 |
| evaluate_offline guidance | 242 s | 1.5 GB | 4 method × 2 variant × 2 surrogate × 21 call. 출력 유한. **trace sampler = D 의 `surrogate_sample` bit-equal 16 / 16**. featured call 단면 dump 8 개 |
| summarize | <1 s | — | `summary_all.json` 21 arm 행 |

smoke 중 고친 것 (최종 code 에 반영됨):
1. scene-GT clearance 가 물체 근처에서 경로 하나에 ~6 s 걸렸다. pruned exact 계산 (`clearance_distance`) 으로 바꾸고 상한을 0.05 m 로 낮췄다. 측정 대상이 아닌 instance (target, mount) 는 건너뛴다. 경로당 ~0.07 s 이고, 결과는 exact 와 bit 단위로 같다 (test).
2. `_gt_summary` 가 `*_neg_rows` count 를 per-row 배열로 잘못 알고 지웠다. per-row 배열 key 를 `*_by_row` 로 바꿨다 (test 추가).
3. NUDGE 단면이 grid × sphere × 장애물 크기로 한 번에 계산돼 RSS 가 4 GB 까지 올랐다. chunk 로 나눠 1.5 GB 가 됐다.
4. guidance 는 call 마다 closure 를 새로 compile 하므로 `jax.clear_caches()` 를 call 마다 부른다 (`--no-clear-jax-caches` 로 끈다). 6 call 동안 RSS 가 ~0.75 GB 로 평평했다.

예상 전체 시간 (10 ep, 221 call, 위 smoke 비례): native ~1.5 분, cloud ~1.5 분, filters ~15 분, guidance ~45 분.

## V1 실행 명령 (전체 10 episode, CPU, worker 1 개, 순서대로)

```bash
cd /mnt/dev/AG3S_VLA-benchmark
source Baseline/paper_baselines/setup/paper_baselines_env.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=$PWD
G=20261023T134909_bf1403d54bc6_paper-baselines-2-sl3dev-notrack     # SL3 dev grounding run (--save-masks)
A=20261012T225820_46a08d6ff644_sl4-perception-dev                   # V11AB perception run (cloud_points)

# 1) grounding 고유 지표 (source / scene GT / attention run 은 G 의 provenance 에서 읽는다)
.venv-baselines/bin/python -m Baseline.paper_baselines.eval.native_metrics --grounding-run $G \
  --dump-figure-data --dump-per-episode 2 --tokencut-dino --run-id baseline-eval-native-sl3dev

# 2) ATS 공정 비교 (같은 V11AB cloud 점)
.venv-baselines/bin/python -m Baseline.paper_baselines.eval.cloud_labeling --ats-run $A --grounding-run $G \
  --dump-figure-data --dump-per-episode 2 --run-id baseline-eval-cloud-sl3dev

# 3) methods offline: filter (새 run 을 만든다) -> guidance (같은 run 에 붙인다) -> 합본
.venv-baselines/bin/python -m Baseline.paper_baselines.methods.evaluate_offline filters --dump \
  --run-id baseline-eval-methods-sl3dev 2>&1 | grep -v "^Polishing not needed"
R=$(ls -d /mnt/dev/AG3S_VLA-storage/results/benchmark/*_baseline-eval-methods-sl3dev | tail -1)
JAX_PLATFORMS=cpu .venv-openpi-live/bin/python -m Baseline.paper_baselines.methods.evaluate_offline guidance --dump --into $R
.venv-baselines/bin/python -m Baseline.paper_baselines.methods.evaluate_offline summarize --into $R
```

산출물 (각 run dir):
- native: `native_frames.jsonl` (arm × frame × camera), `summary.json` (`arms.<arm>.{all,head,left_wrist}`, `check_vs_harness_2d_iou`, `tokencut_dino_parity`), `figure_dump/grounding_<stem>_<t>.npz` (rgb · depth · K · T · GT target / instance map · arm 별 mask · box / score meta · LH head map 3 개 + 결합 map · V11AB cell · TokenCut eigvec / bipartition / seed).
- cloud: `calls.jsonl`, `summary.json` (`arms`, `paired_vs_ats`), `figure_dump/cloud_<stem>_<t>.npz` (points · gt_class · arm 별 flag · cam_bits).
- methods: `filters/` · `guidance/` 각각 `calls.jsonl`, `summary.json`, `provenance.json`, `dump/<arm>/<stem>.npz` (call 별 `cXXX/...` key). 맨 위 `summary_all.json` 은 arm 표.

## verifier 가 알아야 할 것

- 새로 생긴 플래그 · 기본값 변경: 없음 (새 도구만). 기존 harness · method 동작은 바뀌지 않았다.
- 재생산이 필요한 산출물: 없음.
- 옛 기록과 호환이 깨지는가: 없음.
- **guidance 의 한계** (모든 guidance 출력의 `limitation` 에 기록됨):
  - π0.5 대신 D 의 대체 velocity 를 쓴다. `tracking` / `constant` 는 양 끝 경우다.
  - open loop 이다. 수정한 chunk 를 실행하지 않는다.
  - guidance 의 `ms_*` 는 call 마다 새로 만드는 closure 의 jit compile 을 포함한다. 실제 sampling 시간이 아니다.
- **filter 의 한계**: chunk 안의 뒤 action 은 각 method 자기 EE 예측 (gain 0.0115) 위에서 filter 된다. 공식 per-step 호출이 아니다.
- **scene-GT clearance** (`gt_nominal` / `gt_modified`):
  - 모든 method 에 같은 decode (OSC response, 자세 고정) · IK · sphere 를 쓴다. method 내부 예측과 다를 수 있다.
  - 0.05 m 에서 자른다 (cap). 상한 아래의 값은 `SceneGT.exact_distance_near` 와 bit 단위로 같다 (`test_fast_clearance_equals_exact_below_cap`). 처음 구현 (0.10 m, `exact_distance_near` 직접 호출) 은 물체 근처 경로 하나에 ~6 s 가 걸려, centroid KD-tree pruning 으로 바꿨다 (~0.07 s).
  - 표면 거리는 visual mesh 로, 안 / 밖 부호는 collision primitive 로 정한다.
  - obstacle 군은 target · destination · 받침대 (`main_table` 등) 를 뺀 instance 다.
- AEGIS / multi-link 의 `h` 와 KNOWS 의 `h` 는 정의가 달라 method 끼리 비교하지 않는다.
- KNOWS 는 h < 0 인 step 에서도 개입하지 않을 수 있다 (smoke t0_e0 call 2: 10 step 모두 h < 0, 개입 0). 이것은 method 의 동작이며 이 도구가 바꾸지 않는다.
- detector 두 arm 의 첫 call 은 Grounding DINO tiny CPU 탐지 (~17-43 s) 를 포함한다. 장애물은 episode 첫 call 에 고정된다 (AEGIS 원 동작).
- `cloud_labeling` 의 ATS 는 `has_target=False` 인 call 에서 positive 가 없다 (= harness 의 found_any 와 같은 처리). `ats_v11ab_seeds` 는 attention seed 로, 참고 행이다.
- `native_metrics` 의 TokenCut 재계산은 dump frame 에만 한다. DINO 가중치 (`.cache/paper_baselines/torch/hub/checkpoints/dino_deitsmall16_pretrain.pth`) 가 있어야 한다.
- G 가 `offline_grounding.py` / `metrics.py` / `lift.py` 를 고치는 중이다. 위 import 목록이 G 이후에도 성립하는지 V1 실행 전에 test 를 다시 돌려 확인해야 한다 (test 가 실제 run 과 harness `2d_iou` 일치를 본다).
- figure 요구 중 F 가 dump 를 남기지 않은 것:
  - benchmark 대표 장면 (V1 이 기존 run · env render 로 만든다).
  - YOLOE 의 conf 0 부터의 P/R 곡선 (저장 후보가 provider threshold 이상뿐. GPU 재실행 필요).
  - track 의 J/F 시계열 (V2 mask 이후 같은 `native_metrics` 명령).

## 기존 파일에서 고칠 것 (F 는 고치지 않았다)

- `methods/aegis/adapter.py` `solve_cbf_qp`: QP 를 풀 때마다 OSQP 가 "Polishing not needed - no active set detected at optimal point" 를 stdout 에 쓴다 (smoke 에서 관측). 전체 실행 log 가 커진다. 출력 억제 방법은 소유자 C 가 정한다. V1 명령은 `grep -v` 로 거른다.
- `methods/guidance_common/core.py` `make_guidance_fn`: guidance closure 가 scene 을 상수로 잡아 call 마다 재컴파일한다 (D 가 이미 적음). offline 전체에서 guidance 시간의 대부분이 compile 이다.
- `Baseline/paper_baselines/README.md`: `eval/` 행에 `native_metrics.py` · `cloud_labeling.py`, `methods/` 에 `evaluate_offline.py`, `docs/` 에 `metrics.md` 를 추가해야 한다 (README 소유자).

## 내가 기대하는 결과

(verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.)

- cloud 위 공정 비교에서 ATS 의 point precision 은 높고 recall 은 낮을 것이다. ATS target cluster 가 target 표면의 일부만 덮기 때문이다. harness 의 recall 0.068 보다는 높게 나올 것이다.
- phrase prompt 의 detector arm 은 같은 범주 distractor 를 집어 neighbour 비율이 높을 것이다. instruction prompt 는 낮을 것이다.
- `corloc_any_object` 는 `acc05_mask` 보다 높을 것이다. 다른 물체를 맞힌 것도 class-agnostic CorLoc 에서는 정답이다.
- methods 에서 개입은 접근 구간에 몰리고, scene-GT clearance 의 nominal 음수 행은 드물 것이다. SL3 dev 는 장애물을 일부러 놓은 장면이 아니다.
