# BASELINE-EVAL (Implementer G) — 구현

> writer: ag3s-implementer (Implementer G) · 읽는 쪽: verifier V2, lead, scribe · 2026-10-23 KST
> checkout `/mnt/dev/AG3S_VLA-benchmark`, branch `feature/panda-benchmark`, HEAD `5752c64c` (commit 하지 않았다)
> CPU 만 썼다 (`CUDA_VISIBLE_DEVICES=""`, process 1 개, thread 1 개). GPU 측정은 V2 가 한다.

## 무엇을 했나 (평이한 요약 먼저)

세 가지를 고쳤다.

1. **`grounded_sam2:track` memory 상한.** SAM 2 streaming state 가 frame 마다 계속 커지던 문제를 provider 쪽에서 막았다.
   - 원인: pinned code 의 `add_new_frame` 은 모든 frame 을 `torch.cat` 으로 `state["images"]` 에 쌓는다. 모든 frame 의 backbone feature 도 `cached_features` 에 남는다 (`sam2_video_predictor.py:779-802`).
   - 고친 것: 다음 frame 이 실제로 읽는 것만 남긴다. 남기는 것은 prompt (cond) frame, 최근 6 frame 의 mask memory, 최근 15 frame 의 object pointer 다.
   - 결과: mask 는 원래 (unbounded) state 와 bit 단위로 같다. state 크기는 15 frame 이후 평평하다.
   - pinned clone (`external/Grounded-SAM-2`) 은 수정하지 않았다.
2. **평가 CPU 경로 속도.**
   - cProfile 결과, 병목은 거의 전부 `lift_check` 였다. 이 함수가 arm 마다 target mesh (42,522 triangle) 까지의 거리를 처음부터 다시 계산했다.
   - 고친 것 (a): frame 마다 모든 유효 depth pixel 의 거리를 한 번만 계산한다. 각 arm 은 자기 mask 로 그 값을 index 한다.
   - 고친 것 (b): 거리 계산 자체는 reference 와 결과가 bit 단위로 같은 exact pruning 으로 바꿨다.
   - 효과: 같은 일 (1 ep, 12 arm + ATS) 이 **549.5 s → 9.9 s** (55 배) 로 줄었다.
   - 검증: SL3 run 의 `frames.jsonl` 전체 2,873 행을 새 code 로 다시 계산했고, **byte 단위로 같다.**
3. **provider 예외 격리.**
   - provider 하나가 build · reset · predict 에서 예외를 내거나 mask 계약을 어기면, 그 arm 의 그 episode 남은 평가 frame 만 `missing_reason="provider_error"` 행으로 기록한다.
   - run 은 다음 arm 으로 계속 간다.
   - `--fail-fast` 를 주면 예전 동작 (첫 예외에서 run 중단) 으로 돌아간다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `Baseline/paper_baselines/providers/grounded_sam2.py:19-31` | docstring: 무엇이 쌓였는지, 다음 frame 이 무엇을 읽는지 (`sam2_base.py:534-568` mask memory, `:611-620` object pointer), 무엇을 버리는지 | task G-1 근거 |
| `…/grounded_sam2.py:53-62` | `read_window(vp)`. 모델 속성 (`num_maskmem` 7, `memory_temporal_stride_for_eval` 1, `max_obj_ptrs_in_encoder` 16) 에서 읽기 창을 계산한다: mask memory 6 frame, object pointer 15 frame, 합 15 frame | 하드코딩하지 않으려고 |
| `…/grounded_sam2.py:68-143` | `CausalTrack(bounded_memory=True)`. `_add_frame` (97-111): 새 frame 을 1-frame batch 로 넘긴다. `state["images"]=None` 으로 두면 `add_new_frame` 이 `torch.cat` 을 하지 않는다. 그 뒤 진짜 frame index · `num_frames` · `cached_features` key 를 되돌린다. `_trim` (122-143): infer 뒤에 처리한다. `cached_features` 를 비운다. 창 밖의 non-cond output 은 `output_dict` 와 `output_dict_per_obj` 둘 다에서 지운다. mask-memory 창 밖이고 pointer 창 안인 frame 은 `obj_ptr` · `object_score_logits` 만 남긴다. `frames_already_tracked` 도 정리한다 | task G-1 |
| `…/grounded_sam2.py:146-173` | `state_footprint(state)`: state 에서 닿는 tensor 수와 bytes 를 센다 (같은 storage 는 한 번만) | memory 증거 |
| `…/grounded_sam2.py:182, 214, 243` | provider 인자 `bounded_memory=True` (기본). `provenance()["track_memory"]` 에 창을 기록한다 | 재현 기록 |
| `Baseline/paper_baselines/eval/offline_grounding.py:27-35` | module docstring: 공유 geometry, provider_error 행, `--fail-fast` | |
| `…/offline_grounding.py:170, 203-209` | `Episode.geometry(fr)`: frame 별 `FrameGeometry` cache | task G-2 |
| `…/offline_grounding.py:472-550` | `point_triangles_distance_near_pruned`. reference (`benchmark/evaluators/libero_scene_gt.py:212`) 와 같은 수를 낸다. 구조: 같은 `closest_points_pairwise` 를 쓰고 (element-wise), 같은 min 을 낸다. 다른 점은 최소값도 될 수 없고 `≤ 10 mm` 도 될 수 없는 쌍을 풀지 않는 것이다 (점 별 상한 = regular triangle 의 가장 가까운 centroid 거리, 쌍 별 하한 = `|p-c|-r`, margin 1 µm). sliver triangle (`|ab×ac| ≤ 1e-3·edge²`) 은 상한에 쓰지 않는다. 그 쌍은 reference 와 똑같은 KD-tree 호출로 고른다 | task G-2 (쌍 수 2.8 M → 4.7 만) |
| `…/offline_grounding.py:553-600` | `lift_check_from_distances`, `lift_check` (ATS 용, pruned 사용), `FrameGeometry`. frame 당 back-projection 1 회, 유효 pixel 전체의 거리 1 회 (camera 별, 처음 쓸 때 계산), arm 별 `lift_check(selections)` 는 index 만 한다 | task G-2 |
| `…/offline_grounding.py:603-626` | `evaluate_masks` 가 `FrameGeometry` 를 쓴다. 숫자 의미는 그대로다. `eval_grid` 는 eval grid 를 만들 때 같은 back-projection 을 다시 쓴다 | task G-2 |
| `…/offline_grounding.py:405-468` | `build_providers(errors=dict)`: build 예외를 모아 둔다. `arm_list` 는 spec 중복을 제거한다. `ProviderFailure`, `_release_device_memory` (gc + `torch.cuda.empty_cache`, torch 가 이미 load 되었을 때만) | task G-3 |
| `…/offline_grounding.py:629-705` | `run()` main loop 를 `arm_rows` / `ats_rows` 생성기로 나눴다. 행의 key 순서와 내용은 그대로다. episode 로그 줄에 episode 시간을 더했다. `provenance.json["provider_build_errors"]` | task G-2/3 |
| `…/offline_grounding.py:714-805` | `arm_rows`: provider 호출 (`reset`, `predict`, `validate_results`) 만 `_call_provider` 로 감싼다. 평가 code 의 예외는 그대로 위로 올린다. 실패하면 except 블록 밖에서 `prov.reset()` 과 device memory 해제를 한 뒤 남은 평가 frame 을 `provider_error` 행으로 낸다 (이 순서여야 traceback 이 잡고 있는 tensor 가 풀린다). `ats_rows` 는 예전 ATS 블록 그대로다 | task G-3 |
| `…/offline_grounding.py:892, 894-901` | `summary.json`: arm 별 `n_provider_error`, 최상위 `provider_errors` [{arm, episode_id, obs_index, error, n_frames}] | task G-3 |
| `…/offline_grounding.py:931-933` | CLI `--fail-fast` | task G-3 |
| `Baseline/paper_baselines/lift.py:101-118` | `lift_selection(mask, u, v, depth_shape, …)`: 이미 back-project 한 pixel 위의 선택이다. `lift_mask` 는 이것을 써서 다시 구현했다 (결과 같음) | 한 frame 을 한 번만 back-project |
| `Baseline/paper_baselines/eval/metrics.py:88-101` | `build_eval_grid`: camera dict 에 `"backprojected": (pts, u, v)` 가 있으면 다시 계산하지 않는다 | 같음 |
| `Baseline/paper_baselines/scripts/check_track_memory.py` (새 파일) | 실제 SAM 2.1 로 bounded vs unbounded 를 비교한다. 기록 episode 1 개, GT mask 로 prompt 를 주고 매 frame 의 footprint · RSS · (CUDA 면) allocated 를 적고 mask 동일성을 본다 | memory 증거 (GPU 에서도 그대로 쓸 수 있다) |
| `Baseline/paper_baselines/tests/test_grounded_sam2_track.py:112-246` | `StreamingVP` fake 를 추가했다. 저장은 pinned code 처럼 (`images` cat, `cached_features`, output, per-obj slice) 하고, 읽기는 `sam2_base.py` 규칙대로 (stride, pointer 창) 한다. 창 안에서 지워진 것을 읽으면 실패한다. test 2 개를 추가했다 | |
| `Baseline/paper_baselines/tests/test_offline_grounding_speed.py` (새 파일) | test 4 개 (아래) | |

## profile 전후 (cProfile, `.venv-baselines`, CPU 1 thread)

| 측정 | 전 (원 code `5752c64c`) | 후 |
|---|---|---|
| `offline_grounding` 실행, SL3 `libero_spatial_t0_e0`, `gt_oracle` phrase+instruction + ATS (63 arm-frame), episode 시간 | **127.8 s** | **7.7 s** |
| 위 실행의 `lift_check` 누적 | 126.1 s (63 호출, 2.0 s/호출). 그 안에 `point_triangles_distance_near` 123.5 s, `closest_points_pairwise` 68.0 s, `cKDTree`·`minimum.at` 등 자체 43.0 s | `point_triangles_distance_near_pruned` 6.4 s (61 호출) |
| 같은 작업, 1 ep **전체 13 arm** (12 provider arm 은 저장된 mask 를 replay, + ATS, 273 arm-frame). provider 시간 ≈ 0 이므로 평가 CPU 경로만이다 | **549.5 s** | **9.9 s** |
| SL3 10 ep 전체 재계산 (같은 replay) | (verify.json 추정 ~600 s/ep) | 7.6–13.8 s/ep, 평균 9.96 s/ep |
| 후의 남은 시간 (1 ep, 13 arm, `prof_after_full`) | — | per-pixel 거리 4.6 s (frame·camera 당 1 회, 42 회), `voxel_keys` 1.6 s, ATS 0.7 s, attention load 0.9 s |

쌍 수 (frame 1 개, 모든 유효 pixel 108,893 점): reference 2,811,995 쌍, pruned 46,770 쌍 (`libero_spatial_t0_e0` obs 10).

산출물: `outputs/paper_baselines_smoke/G_profile/{prof_before,prof_after,prof_after_full}.{out,txt}`, `outputs/paper_baselines_smoke/G_prof_{before,after}/` (1 ep gt_oracle+ATS run), `outputs/paper_baselines_smoke/G_bit_identity/time_old_vs_new_t0_e0.json`.

## bit 동일성 증거

| 비교 | 결과 |
|---|---|
| SL3 run `20261023T134909_bf1403d54bc6_paper-baselines-2-sl3dev-notrack` 의 **10 ep 전체** `frames.jsonl` 를 새 code 로 다시 계산했다. 비교 대상 13 arm: 12 provider arm + `ats_v11ab`. gt_oracle 은 다시 돌렸고, GPU provider 는 저장된 mask 에 기록된 found/score/boxes/phrases/latency/extras 를 붙여 replay 했다 | **2,873 / 2,873 행이 byte 단위로 같다.** 비교는 JSON 한 줄 문자열 그대로 했다. 예외는 gt_oracle 의 wall-clock `latency_ms`·`latency_ms_total` 두 칸뿐이다. `outputs/paper_baselines_smoke/G_bit_identity/rescore_sl3_all.json` (sha256 `74f1ddad…`) |
| 1 ep gt_oracle ×2 + ATS 를 `offline_grounding` CLI 로 실행 (새 code / 원 code) | 63/63 행 같음 (latency 제외), 원 code 출력도 63/63 같음 |
| `point_triangles_distance_near_pruned` vs reference, synthetic 900 회 (seed 1000–1299 × md 3 종). 섞은 것: sliver, 점 triangle, ratio 1e-5…1e-1 의 얇은 triangle, 큰 triangle | 2,700,000 점, **불일치 0** (1,666,420 점이 finite) |
| 같은 비교, 기록 frame 의 모든 유효 depth pixel (t0_e0 obs 10, t1_e0 obs 20, camera 2 개) | bit 같음 (test) |

**numpy version 주의.** stored run 은 `.venv-baselines` (numpy 2.2.6) 에서 나왔다. `.venv-ag3s` (numpy 2.4.6) 에서는 **원 code 도** 저장 행과 다르다. 21 frame 중 7 frame 에서 `3d_target_voxels`·`3d_FN`·`3d_iou`·`3d_recall` 의 마지막 bit 가 다르다 (float sum 구현 차이). 같은 venv 안에서 새 code 와 원 code 는 21/21 frame 모두 같다. 그래서 stored-run byte test 는 numpy 2.2.6 에서만 돈다 (다른 version 에서는 skip 하고 이유를 남긴다).

이번 실패에서 배운 것: 처음 구현은 점 별 상한에 degenerate triangle 의 centroid 도 썼다. 그런데 reference 의 `closest_points_pairwise` 는 degenerate triangle 에서 closest point 가 triangle 위에 있지 않다. 그래서 그 거리가 centroid 거리보다 클 수 있다 (zero-length edge 예: centroid 3.86 mm, 풀린 거리 16.8 mm). synthetic test 가 8 점 불일치로 잡았다. 그래서 상한은 regular triangle 로만 만들고, sliver 기준을 `1e-3` 으로 보수적으로 잡았다. 그 결과 regular triangle 의 rounding 은 ≲1e-12 m 로 margin 1 µm 보다 훨씬 작다. 실제 target mesh (bowl, 42,522 triangle) 의 최소 ratio 는 1.02e-3 이라 sliver 가 0 개다.

## memory 상한 증거

**1) 실제 SAM 2.1 hiera-tiny, CPU, SL3 `libero_spatial_t0_e0` head camera.** 명령은 `scripts/check_track_memory.py --frames 60 --compare-frames 30` 이다. GT mask 로 prompt 를 준 뒤 매 frame `CausalTrack.step` 를 부른다.

| track frame | bounded tensors | bounded state bytes | bounded RSS MB | unbounded tensors | unbounded state bytes | unbounded RSS MB |
|---|---|---|---|---|---|---|
| 0 | 13 | 31,196,164 | 1136 | 21 | 148,636,676 | 1323 |
| 5 | 63 | 35,133,464 | 1255 | 111 | 865,605,656 | 2031 |
| 10 | 89 | 35,925,036 | 1343 | 201 | 1,582,574,636 | 2715 |
| 15 | 109 | 35,930,176 | 1335 | 291 | 2,299,543,616 | 3510 |
| 29 | 109 | 35,930,176 | 1353 | 543 | 4,307,056,760 | 5563 |
| 59 | 109 | 35,930,176 | 1403 | — | — | — |

- bounded 는 frame 15 부터 59 까지 tensor 수 109, bytes 35,930,176 으로 **변하지 않는다**.
- 그 상태에서 non-cond output 은 15 개, `images` 는 1 frame, `cached_features` 는 0 이다.
- unbounded 는 frame 마다 +136.8 MB (CPU, tiny) 씩 커진다.
- **mask 는 30 / 30 frame 이 bit 단위로 같다.** 이 30 frame 은 pointer 창 (15) 을 넘으므로, trim 이 시작된 뒤의 frame 도 비교에 들어간다.
- frame 당 약 10 s (CPU 1 thread).
- 산출물: `outputs/paper_baselines_smoke/G_track_memory/track_memory_head.json` (sha256 `cab101d0…`), `track_memory.log`.

**2) unit test (fake, 120 frame).** bounded 와 unbounded 의 mask 가 같다. footprint (bytes · tensor 수) 는 window+1 frame 뒤로 일정하다. unbounded 는 매 frame 증가한다. 마지막 state 는 non-cond 15 개 (가장 최근), full output 6 개, `images` 1 frame, `cached_features` 비어 있음, prompt frame 유지다.

**3) harness 배선 (CPU, smoke weight).** `offline_grounding --providers grounded_sam2:track gt_oracle --size smoke --device cpu --limit-episodes 1 --limit-frames 2` 로 돌렸다. track 은 obs 0 에서 시작했고, obs 15 에서 `track_frames=16` (창을 넘음) 이다. `provenance.track_memory = {"bounded": true, "mask_memory_frames": 6, "obj_ptr_frames": 15, "frames": 15}`, `provider_errors` 는 [] 다. `outputs/paper_baselines_smoke/G_track_wiring/`. **숫자는 배선 확인용이다. 판정에 쓰지 않는다.**

GPU 에서의 확인 (frame 당 allocated / reserved 가 평평한가) 은 하지 않았다. V2 가 한다.

## 단위 검증

```bash
source scripts/setup/env.sh; source Baseline/paper_baselines/setup/paper_baselines_env.sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=$PWD \
  .venv-baselines/bin/python -m pytest -p no:cacheprovider -q Baseline/paper_baselines/tests Baseline/paper_baselines/methods/tests
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest -p no:cacheprovider -q \
  Baseline/paper_baselines/tests/{test_offline_grounding,test_localization_heads,test_offline_grounding_speed,test_lift,test_metrics,test_grounded_sam2_track}.py
```

- `.venv-baselines`: **111 passed, 2 skipped** (29.9 s, 마지막 code 기준). skip 2 개는 `jax` 없음 (`test_methods_guidance.py`, F 의 `test_evaluate_offline.py`) 이다.
  - 이 수에는 F 의 새 test (`test_native_metrics.py`, `test_cloud_labeling.py`, `test_evaluate_offline.py`) 도 들어 있다. working tree 에 있었다.
- `.venv-ag3s`: **42 passed, 3 skipped**. skip 은 torch 2 개, numpy version 1 개 (위 주의).

| 새 test | 확인하는 것 |
|---|---|
| `test_grounded_sam2_track.py::test_bounded_memory_same_masks_and_flat_footprint` | bounded == unbounded mask (120 frame). 창 (6/15/15). footprint 가 평평한지, unbounded 가 증가하는지. 마지막 state 의 내용 |
| `…::test_bounded_memory_is_causal` | bounded 에서도 t 이후 frame 을 바꿔도 t 까지의 mask 가 같은지 |
| (기존) `test_track_is_causal`, `test_reset_drops_state`, `test_per_frame_mode_has_no_state` | 기본 `bounded_memory=True` 로도 통과한다 |
| `test_offline_grounding_speed.py::test_pruned_distance_bit_identical_synthetic` | 12 soup × 2 md 에서 bit 동일 (sliver, zero edge, 얇은 triangle, 큰 triangle 포함) |
| `…::test_pruned_distance_bit_identical_recorded_pixels` | 기록 frame 2 개의 모든 유효 pixel 에서 bit 동일. mask index 결과 == reference 로 lift 한 점의 `lift_check` |
| `…::test_recorded_episode_rows_byte_identical` | SL3 t0_e0 의 273 행 (13 arm × 21 frame) 이 저장 행과 byte 동일 |
| `…::test_provider_failure_is_isolated_to_its_arm` | predict 2 번째 frame 에서 예외 → 그 arm 의 2 frame 이 `provider_error` 이고 `provider_error_obs_index=15`. build 실패 arm → 3 frame `provider not built`. 뒤 arm (`gt_oracle`) 은 정상이다. `summary.provider_errors` · `n_provider_error`, `provenance.provider_build_errors` 를 확인하고, `--fail-fast` 이면 예외가 난다 |

## verifier 가 알아야 할 것

**새 플래그 · 기본값 변경**
- `GroundedSAM2Provider(bounded_memory=True)` 가 기본이다. `offline_grounding` 에는 이것을 끄는 플래그가 없다.
  - 끄는 것은 test 와 `check_track_memory.py` 에서만 한다.
  - `provenance.providers["grounded_sam2:track"].track_memory` 에 기록된다.
- `offline_grounding --fail-fast` 를 새로 만들었다. 기본은 격리다.

**새 열 · 새 key**
- `frames.jsonl`: 실패한 arm-frame 에 `missing_reason: "provider_error"`, `provider_error` (`"Type: message"`), `provider_error_obs_index` 가 생긴다.
- `summary.json`: arm 별 `n_provider_error`, 최상위 `provider_errors`.
- `provenance.json`: `provider_build_errors`.
- 실패가 없으면 행은 예전과 byte 단위로 같다.
- `provider_error` 행은 `n_missing` 에도 세어진다. `found_any` rate 에서는 빠진다 (예전 `missing_reason` 행과 같은 규칙).

**V2 가 판정 전에 확인할 것**
- `summary.json["provider_errors"]` 가 비어 있는지 본다.
- 비어 있지 않으면 그 arm·episode 는 측정이 아니다. 특히 CUDA context 가 깨지면 (`CUDA error: unrecognized error code`) 그 뒤 GPU arm 이 모두 `provider_error` 가 될 수 있다. run 이 끝까지 돈다는 사실을 측정이 끝났다는 뜻으로 읽지 않는다.

**로그 형식**
- episode 로그 줄이 `…: N frames, episode X s, total Y s` 로 바뀌었다. 예전에는 누적 시간만 있었다.

**그 밖**
- 재생산이 필요한 산출물: 없다. SL3 notrack run 의 숫자는 그대로다 (위 2,873 행).
- 옛 기록과 호환이 깨지는가: 없다. `methods/guidance_common/recorded.py` 가 쓰는 `Episode` / `frame()` 는 그대로다.
- 내가 바꾸지 않은 working tree 변경이 있다. F 의 파일과 `AG3S/ag3s/docs/AG3S_BASELINE_LOG.md` 다. 손대지 않았다.

**track 실행 비용 (참고, 측정 아님)**
- track 은 평가 frame 이 아닌 frame 까지 매 frame SAM 2 를 돈다 (episode 의 obs 0 ~ 마지막 평가 frame).
- GPU 시간은 frame 수 × camera 2 × prompt mode 2 에 비례한다.

## V2 가 그대로 쓸 GPU 명령

시작 전 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` 가 비어 있는지 본다. 같은 pod 의 다른 session (`ag3s-vla-benchmark-67`) 에 시작과 끝을 알린다. `.venv-baselines` 에서 돌린다.

```bash
cd /mnt/dev/AG3S_VLA-benchmark
source scripts/setup/env.sh; source Baseline/paper_baselines/setup/paper_baselines_env.sh

# 1  SL3 dev (10 ep, 221 frame): track 추가 측정
#    per_frame · gt_oracle · ATS 를 같이 넣으면 같은 run 안에서 paired 비교가 된다.
#    per_frame 은 notrack run 과 재현성도 비교할 수 있다.
flock -w 3600 /tmp/ag3s-vla-gpu-validation.lock env PYTHONPATH=$PWD .venv-baselines/bin/python -m Baseline.paper_baselines.eval.offline_grounding \
  --source-run 20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0 --scene-gt outputs/verify/SL4/raw/scene \
  --providers gt_oracle grounded_sam2:per_frame grounded_sam2:track --prompt-modes phrase instruction --size default \
  --ats-run 20261012T225820_46a08d6ff644_sl4-perception-dev --save-masks --run-id baseline-eval-v2-sl3dev-track

# 2  SL4b dev (40 ep, 873 frame): 전체 arm + track (SL4b 에는 ATS perception run 이 없다)
flock -w 3600 /tmp/ag3s-vla-gpu-validation.lock env PYTHONPATH=$PWD .venv-baselines/bin/python -m Baseline.paper_baselines.eval.offline_grounding \
  --source-run 20261013T092537_714c891415d9_sl4b-dev-pi05only-t0-9e1-4 --scene-gt outputs/verify/SL4b/raw/scene \
  --providers gt_oracle grounded_sam grounded_sam2:per_frame grounded_sam2:track yoloe \
              localization_heads:threshold localization_heads:sam_point tokencut:foreground tokencut:attention_overlap \
  --prompt-modes phrase instruction --size default \
  --attention-run 20261013T094739_714c891415d9_sl4b-attention-audit-dev-e1-4 --save-masks --run-id baseline-eval-v2-sl4bdev
```

- 두 run 사이에 lock 을 풀어도 된다. 하나로 묶으려면 `flock -w … LOCK bash -c '…; …'` 를 쓴다.
- GPU 에서 memory 가 평평한지 따로 보려면 아래 명령을 lock 아래에서 돌린다. 출력의 `cuda_allocated_mb` / `cuda_reserved_mb` 를 frame 별로 기록한다.
  ```bash
  .venv-baselines/bin/python Baseline/paper_baselines/scripts/check_track_memory.py \
    --weight sam2.1-hiera-large --device cuda --frames 150 --compare-frames 40 --threads 2 \
    --out outputs/verify/BASELINE-EVAL-V2/track_memory
  ```
- 위 명령은 PAPER-BASELINES-2 처럼 provider 를 한 process 에 모두 올린다. 한 arm 이 실패해도 run 은 멈추지 않는다 (위 "V2 가 판정 전에 확인할 것").

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다. -->
- track 의 GPU state 는 15 frame 이후 평평하다. 이 상태로 SL3 · SL4b 의 모든 episode (평가 frame 이 obs 120 이상인 것 포함) 가 OOM 없이 끝나고, `provider_errors` 는 비어 있을 것이다.
- 1 번 run 의 `grounded_sam2:per_frame` 행은 notrack run 과 거의 같을 것이다. GPU 비결정성이 있으면 아주 조금 다를 수 있다.
- episode 당 평가 CPU 경로는 약 10 s 로, 실행 시간은 provider (GPU) 시간이 거의 정한다.
