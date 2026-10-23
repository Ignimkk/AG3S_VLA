# PAPER-BASELINES (Implementer B) — 구현

> writer: ag3s-implementer (Implementer B) · 읽는 쪽: verifier, scribe, lead · 2026-10-23 KST
> checkout `/mnt/dev/AG3S_VLA-benchmark`, branch `feature/panda-benchmark`, HEAD `8cd02b1e` (commit 하지 않았다)

## 무엇을 했나 (평이한 요약 먼저)

VLA 내부 grounding baseline 두 개를 이 checkout 의 공통 interface (A 의 `interface.py`) 에 맞춰 만들었다.

- **localization heads:** Kang et al. 의 head 선택 기준을 공식 code 에서 옮겨 pi0.5 의 기록된 attention 에 적용했다. dev episode (init 0–4) 에서만 head 를 고르고, 그 결과를 frozen json 두 개로 남겼다 (공식 code 방식 / 논문 방식).
- **TokenCut:** normalized cut 을 numpy 로 다시 썼다. 같은 입력에서 공식 함수와 mask · seed · label 이 같다.

문서 세 개도 썼다.
- `survey.md`: 구현하지 않는 8편의 공개 code · license · 수치
- `exemption_ablations.md`: No / All-link exemption 을 현재 flag 로 표현하는 방법. code 는 고치지 않았다
- `catalog_vla_internal.md`: README catalog 행 초안과 원 논문과 다른 점

**측정은 하지 않았다.** grounding 품질 숫자는 verifier 몫이다. 아래 smoke 는 배선만 본다.

## 바뀐 파일 (전부 새 파일. 기존 파일은 한 줄도 고치지 않았다)

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `Baseline/paper_baselines/providers/localization_heads.py:128-215` | `elbow_chord` · `spatial_entropy_code` · `analyze_heads_code` 는 `analyze.py:20-119` 를 그대로 옮겼다. `spatial_entropy_paper` 는 논문 Eq. 3 이다 | head 선택 기준을 공식 code 에서 읽는다 (task B-1) |
| `…/localization_heads.py:238-305` | `select_heads`: 논문 Sec. 4.2 의 빈도 집계 (표본마다 entropy 낮은 10 개 안에 든 횟수, 상위 k = 3). `protocol` 은 `official_code` (기본, 표본별 chord + code entropy + bottom-row) 또는 `paper` (평균 S 의 chord + 논문 entropy). 동점은 평균 entropy → layer → head 로 정해 결정적이다 | 공식 code 에는 표본 간 집계가 없어서 논문 규칙을 썼다 |
| `…/localization_heads.py:318-411` | `index_attention_runs` · `build_selection` · `load_selection` · `aux_from_attention_npz`. dev 가 아닌 episode 는 제외하고 기록한다. episode npz 의 sha256 을 남기고, `selection_sha256` 로 편집 여부를 검사한다 | dev 전용 frozen json |
| `…/localization_heads.py:416-471` | `combine_heads` · `binarize_mean_relu` 는 `bbox.py` 다. `upscale_bilinear` 는 `F.interpolate(bilinear, align_corners=False)` 를 numpy 로 쓴 것이다. `to_image_orientation` 은 `AG3S.adapters.panda.attention.unmirror_grid` 를 import 한다 | mask 생성. 좌우 반전 규칙은 한 곳에서만 가져온다 |
| `…/localization_heads.py:483-620` | `LocalizationHeadsProvider` (mask_mode `threshold` · `sam_point` · `sam_box`). SAM 은 A 의 `sam_backend.SamV1Backend` 의 `predict_points` / `predict_box` 를 쓴다 | provider |
| `…/localization_heads.py:625-655` | CLI `select` (같은 이름의 frozen 파일이 있으면 덮어쓰지 않는다) | 재현 |
| `Baseline/paper_baselines/providers/frozen/localization_heads_pi05_libero_spatial_dev_init0-4_official_code.json` | 선택 결과 (sha256 `baa85c42…`, `selection_sha256` `07eb43c3…`) | frozen head 목록 |
| `Baseline/paper_baselines/providers/frozen/localization_heads_pi05_libero_spatial_dev_init0-4_paper.json` | 선택 결과 (sha256 `e1538282…`, `selection_sha256` `b959ce73…`) | 논문 방식 비교용 |
| `Baseline/paper_baselines/providers/tokencut.py:95-131` | `ncut` 은 `object_discovery.py:12-92` 를 numpy 로 옮겼다 (binary graph τ 0.2 / ε 1e-5, `eigh(D−A, D)`, 평균 이분, seed 의 4-연결 component) | re-impl (task B-2) |
| `…/tokencut.py:157-187` | `select_attention_overlap` — **원 방법이 아니다.** 양쪽 bipartition 의 모든 component 중 above-mean attention 질량이 가장 큰 것을 고른다 | task 의 `attention_overlap` 옵션 |
| `…/tokencut.py:198-271` | `load_official_module` (pinned clone 의 file 을 private 이름으로 import), `DinoKeyExtractor` (`main_tokencut.py:185-266` 의 key 추출, weight URL 은 `networks.py`) | DINO ViT-S/16 feature |
| `…/tokencut.py:286-397` | `TokenCutProvider` (`feature_source` `dino_vits16` · `aux`, `target_select` `foreground` · `attention_overlap`) | provider |
| `Baseline/paper_baselines/tests/test_localization_heads.py` | 13 test (아래) | parity · 결정성 · not-found |
| `Baseline/paper_baselines/tests/test_tokencut.py` | 15 test (parity 7 case 포함) | parity · not-found |
| `Baseline/paper_baselines/scripts/b_smoke_localization_heads.py` · `b_smoke_tokencut_randinit.py` | smoke 스크립트 (출력 `outputs/paper_baselines_smoke/B_*.json`) | 배선 확인 |
| `Baseline/paper_baselines/docs/survey.md` · `exemption_ablations.md` · `catalog_vla_internal.md` | 문서 | task B-3 · B-4 · B-5 |

## 단위 검증

```bash
source scripts/setup/env.sh
# torch 있는 venv (A 가 만든 .venv-baselines, 설치는 하지 않았다)
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=$PWD .venv-baselines/bin/python -m pytest -p no:cacheprovider -v \
    Baseline/paper_baselines/tests/test_localization_heads.py Baseline/paper_baselines/tests/test_tokencut.py
# numpy 만 있는 venv
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider \
    Baseline/paper_baselines/tests/test_localization_heads.py Baseline/paper_baselines/tests/test_tokencut.py
```

실제 출력 (2026-10-23 KST, 마지막 실행):

```
== .venv-baselines
test_localization_heads.py::test_pinned_checkout_sha PASSED
test_localization_heads.py::test_elbow_chord_matches_official_numpy PASSED
test_localization_heads.py::test_analyze_heads_matches_official_torch PASSED
test_localization_heads.py::test_combine_and_upscale_match_official_bbox PASSED
test_localization_heads.py::test_select_heads_deterministic_and_finds_planted_heads PASSED
test_localization_heads.py::test_select_heads_excludes_first_two_layers PASSED
test_localization_heads.py::test_build_selection_dev_only_and_reproducible PASSED
test_localization_heads.py::test_cli_select_refuses_overwrite PASSED
test_localization_heads.py::test_threshold_not_found_on_constant_attention PASSED
test_localization_heads.py::test_threshold_found_and_libero_mirror PASSED
test_localization_heads.py::test_sam_modes_need_segmenter PASSED
test_localization_heads.py::test_mirror_rule_agrees_with_lift_policy_mask PASSED
test_localization_heads.py::test_results_satisfy_interface_contract PASSED
test_tokencut.py::test_pinned_checkout_sha PASSED
test_tokencut.py::test_ncut_parity_with_official[0-dims0-384-random-True] PASSED
test_tokencut.py::test_ncut_parity_with_official[1-dims1-384-random-True] PASSED
test_tokencut.py::test_ncut_parity_with_official[2-dims2-64-random-True] PASSED
test_tokencut.py::test_ncut_parity_with_official[3-dims3-64-objects-True] PASSED
test_tokencut.py::test_ncut_parity_with_official[4-dims4-384-objects-True] PASSED
test_tokencut.py::test_ncut_parity_with_official[5-dims5-64-objects-False] PASSED
test_tokencut.py::test_ncut_parity_with_official[6-dims6-384-random-False] PASSED
test_tokencut.py::test_nearest_resize_matches_torch PASSED
test_tokencut.py::test_ncut_recovers_planted_object PASSED
test_tokencut.py::test_ncut_rejects_bad_input PASSED
test_tokencut.py::test_attention_overlap_picks_attended_component[image] PASSED
test_tokencut.py::test_attention_overlap_picks_attended_component[libero_policy] PASSED
test_tokencut.py::test_not_found_paths PASSED
test_tokencut.py::test_provenance_flags_non_original_rule PASSED
============================== 28 passed in 3.68s ==============================
== .venv-ag3s
18 passed, 10 skipped in 2.45s        (skip 10 = torch 가 필요한 parity test)
```

| task 가 요구한 test | 어디 | 판정 |
|---|---|---|
| tokencut parity (같은 random feature → 공식 code 와 같은 mask) | `test_ncut_parity_with_official` 7 case. 비정사각 grid 2 case, similarity-graph ablation 2 case 포함 | mask · seed · label · bbox 동일. 고유벡터: binary graph 1e-10, similarity graph 1e-6 이내 |
| localization head 선택의 결정성 | `test_select_heads_deterministic…` (두 protocol), `test_build_selection_dev_only_and_reproducible` (run 순서를 바꿔도 `selection_sha256` 같음, init 7 은 제외). 실제 dev 에서도 run 순서를 바꿔 다시 돌렸고 `selection_sha256` 이 `07eb43c3…` 로 같았다 | 통과 |
| threshold mask 의 not-found 경로 | `test_threshold_not_found_on_constant_attention` (상수 attention, frozen head 없는 camera) | `found=False`, 빈 mask, score/boxes `None`, 사유가 기록된다 |
| 추가 | 공식 `analyze_heads` / `bbox.py` / `F.interpolate` 와의 parity. A 의 `lift.policy_mask_to_opencv` 와 좌우 반전 규칙 일치. `interface.validate_results` 계약 | 통과 |

**test 를 고치며 확인한 수치 사실 두 개.**
1. component 가 하나뿐인 head 는 entropy 가 ~0 이다. 그런 head 끼리의 순서는 torch 와 numpy 의 float32 합 순서 차이 (|ΔSE| ≤ 1.2e-7) 로 바뀐다. 처음 test (blob 1개 head 만 있음) 가 이 때문에 실패했다. 지금 test 는 entropy 가 실제로 다른 head (5 종 이상) 를 쓰고, 1e-6 미만의 잡음을 뺀 순서를 비교한다.
2. TokenCut 의 similarity-graph 모드는 affinity 를 float32 로 유지한다. 그래서 numpy 와 torch 의 matmul 끝자리 차이가 고유벡터에 2.8e-8 로 남는다. mask 는 같다.

## smoke (배선만, 숫자 판정 없음)

| smoke | 명령 | 결과 |
|---|---|---|
| localization heads, 실제 dev frame 1 장 (SL3 audit `libero_spatial_t0_e0`, call 0, obs 10) | `CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD .venv-ag3s/bin/python Baseline/paper_baselines/scripts/b_smoke_localization_heads.py` | 두 frozen json 모두 head · left_wrist `found=True`, mask 256×256, 지연 1.4–5.1 ms (CPU, mask 생성만). 출력 `outputs/paper_baselines_smoke/B_localization_heads_smoke.json` |
| TokenCut DINO 경로, **random-init** ViT-S/16 (다운로드 없음) | `source Baseline/paper_baselines/setup/paper_baselines_env.sh; CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD .venv-baselines/bin/python Baseline/paper_baselines/scripts/b_smoke_tokencut_randinit.py` | 공식 `dino/vision_transformer.py` load "All keys matched", feature (256, 384), 256×256 와 200×250 RGB 모두 RGB 크기 mask. CPU 72–90 ms/camera, maxrss 688 MB. 출력 `outputs/paper_baselines_smoke/B_tokencut_randinit_smoke.json` |
| head 선택 (dev 50 episode, 1,094 call) | `.venv-ag3s/bin/python -m Baseline.paper_baselines.providers.localization_heads select --protocol {official_code,paper} --attention-run <SL3 audit> --attention-run <SL4b audit> --out <json>` | 각 8–11 s, maxrss 76–83 MB |

## 확인하지 못한 것

- **DINO ViT-S/16 공식 weight 로 TokenCut 을 돌리지 않았다.**
  - `dino_deitsmall16_pretrain.pth` (87 MB) 가 `.cache/paper_baselines/torch/hub/checkpoints/` 에 없다. 다운로드는 A 담당이라 받지 않았다.
  - weight 가 생기면 `get_provider("tokencut")` 가 그 경로를 자동으로 읽는다.
- **실제 SAM 으로 `sam_point` / `sam_box` 를 돌리지 않았다.**
  - 구현 중에는 `sam_backend.py` 가 없었다. 마지막에 생긴 것을 보고 호출 모양 (`predict_points(rgb, points_xy, labels)`, `predict_box(rgb, box_xyxy)`) 을 맞췄다.
  - 같은 모양의 fake 로만 test 했다. 기본 weight 는 `SamV1Backend()` = `sam-vit-huge` 다. smoke 는 `sam=SamV1Backend("sam-vit-base")` 처럼 작은 weight 로 넘기면 된다.
- **VLA patch feature 생산자는 없다.** TokenCut `feature_source="aux"` 의 입력을 만드는 code 가 없다. attention audit 은 512-d pooled `kv_feature` 만 남긴다.
- **localization heads 는 audit 이 있는 frame 에서만 돈다.** `pooled_f16` 이 있는 frame 은 SL3 · SL4b dev call 뿐이다. 다른 frame (예: test split) 에 쓰려면 같은 runner 로 audit 을 다시 떠야 한다 (GPU).
- **exemption ablation 의 runtime 동작은 확인하지 않았다.**
  - server 를 띄우지 않았다. 확인한 것은 parser · config 단계뿐이다 (`exemption_ablations.md` §4).
  - `relax` 에서 target-free 계층이 없을 때 기록 경로가 빈 값을 견디는지는 정적으로만 읽었다.
- **survey 표 수치는 WebFetch 로 arXiv HTML 을 요약 추출한 것이다.** 논문에 옮기기 전에 PDF 표와 다시 대조해야 한다. [5] 와 [9] 는 원문 자체에 표 사이 불일치가 있다.

## verifier 가 알아야 할 것

- **새로 생긴 플래그 · 기본값 변경:** 없음. V11AB · AG3S core · runner 는 손대지 않았다.
- **provider 기본값:**
  - `localization_heads`: `protocol="official_code"`, `top_k=3`, `mask_mode="threshold"`. `aux["attention_orientation"]` 은 필수다 (`"libero_policy"` | `"image"`). 없으면 ValueError 로 멈춘다.
  - `tokencut`: `feature_source="dino_vits16"`, `target_select="foreground"`.
- **재생산이 필요한 산출물:** 없음.
  - frozen json 두 개는 dev audit 에서 다시 만들 수 있다. 같은 이름이 있으면 CLI 가 덮어쓰지 않으므로 다른 `--out` 을 준다.
  - 입력 episode npz 의 sha256 이 json `inputs` 에 있다.
- **옛 기록과 호환이 깨지는가:** 없음.
- **pixel frame:**
  - localization heads 는 grid 단계에서 un-mirror 한 뒤 RGB 크기로 bilinear resample 한다. V11AB `attention_pixel_map` 과 같은 순서다.
  - 결과 mask 는 입력 RGB (= depth) frame 이다. `lift.policy_mask_to_opencv` 를 **다시 적용하면 안 된다.** 두 번 뒤집힌다.
  - test `test_mirror_rule_agrees_with_lift_policy_mask` 가 두 경로의 일치를 고정한다.
- **head 의 query 정의:** 원 논문의 "마지막 text token" 이 아니라 "마지막 action token, denoise step 0" 이다 (`catalog_vla_internal.md` §2.1). 결과를 해석할 때 이 차이를 함께 적어야 한다.
- **exemption ablation:**
  - V11AB (`--links gripper`) 위에서 `--target-field-policy exclude-all` 은 target 을 빼는 sphere 집합이 V11AB 와 같다. All-link 비교는 `--links arms` 판 (pb3 이상) 에서만 의미가 있다.
  - 지금 수동으로 flag 를 덧붙이면 server 기록의 version 이름이 원래 이름 (`V11AB` 등) 으로 잘못 남는다. 최소 변경은 `profile.py:236-237` 사이에 `AG3S_VERSIONS` 항목을 추가하는 것이다 (`exemption_ablations.md` §3).
- **pip 패키지 추가 요청:** 없음 (`.venv-baselines` 의 torch · scipy · numpy 로 충분).

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다. -->
- `localization_heads` threshold mask 는 mean 위 전부를 포함한다. 그래서 큰 영역이 잡힐 것으로 본다 (smoke 에서 head camera 14k / 65k px). 3D precision 은 낮고 recall 은 높을 것으로 예상한다. `sam_box` 가 precision 을 올릴 것이다.
- 공식 code 방식으로 고른 left_wrist head 첫 번째가 (8, 2) 다. 이것은 V11AB 의 고정 cell L8H2 와 같다. 그래서 wrist 에서는 V11AB 와 비슷한 grounding 이 나올 것으로 본다.
- TokenCut `foreground` 는 LIBERO 256 px 장면에서 탁자나 큰 물체를 전경으로 잡는 경우가 많을 것이다. target IoU 는 낮을 것으로 예상한다.
