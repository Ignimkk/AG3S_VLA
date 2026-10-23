# PAPER-BASELINES (A) — 구현

> writer: ag3s-implementer (Implementer A) · 읽는 쪽: verifier, scribe, lead · 2026-10-23 KST (작업 11:58–13:00)
> checkout `/mnt/dev/AG3S_VLA-benchmark`, branch `feature/panda-benchmark`, HEAD `8cd02b1e` (commit 하지 않음)

## 무엇을 했나 (평이한 요약 먼저)

논문 Tab. 2 의 "External segmenter" 행을 채울 수 있도록 새 폴더 `Baseline/paper_baselines/` 를 만들었다.
모든 baseline 은 같은 interface 로 "RGB → camera 별 target mask" 를 내고, 그 mask 는 같은 depth · 같은 V11AB
back-projection · 같은 robot 제거 옵션으로 3D 로 올린 뒤 development branch 와 같은 5 mm soft-voxel IoU 로 잰다.
provider 는 GT oracle, Grounded-SAM(v1 recipe), Grounded-SAM-2(per-frame / causal track), YOLOE 네 개다.
평가 harness 는 recorded `_geom.npz`(segmentation 포함) 위에서 끝까지 돌아가는 것을 CPU 로 확인했다.
숫자 측정과 판정은 하지 않았다(verifier 몫). `.venv-baselines` 는 2026-10-23 12:20 KST 에 준비됐다.

## 바뀐 파일

기존 tracked 파일은 하나도 고치지 않았다(`git diff HEAD` 비어 있음). 아래는 모두 새 파일이다.

| 파일 | 무엇이 | 왜 |
|---|---|---|
| `Baseline/paper_baselines/interface.py` | `TargetQuery` · `TargetMaskResult`(`not_found`) · `TargetMaskProvider` Protocol · `select_top1` · `validate_results` · lazy registry (`get_provider`, B 의 `localization_heads` / `tokencut` 포함) | 공통 계약. registry 는 torch 를 import 하지 않는다 |
| `Baseline/paper_baselines/lift.py` | raw→OpenCV(`raw[::-1]`), policy-224 ↔ OpenCV mirror(`camera.opencv_to_policy_pixel` 재사용), robot 제거 `none \| gt_robot_seg`, V11AB `AG3S.ag3s.stages.reconstruction.backproject`(`range_max` 2 m) 재사용, 5 mm voxel key | frame 변환을 한 곳에 모으고 test 로 고정 |
| `Baseline/paper_baselines/weights.py` | weights 목록(HF revision / release URL 고정), sha256 sidecar cache, preset `default`(큰 모델) / `smoke`(작은 모델) | provenance |
| `providers/gt_oracle.py` | `segmentation_geom` ∩ BDDL target geom ids (`task_target` / scene `target_json` 재사용), `phrase_from_object` | 상한선 · lift 확인 |
| `providers/sam_backend.py` | `SamV1Backend`(HF SamModel) · `Sam2ImageBackend`(pinned sam2) — `predict_box`, `predict_points` | A 의 provider 와 B 의 `sam_point` 가 공유 |
| `providers/grounding_dino.py` | HF Grounding DINO, caption 규칙(소문자 + ".") | grounded_sam(2) 공유 |
| `providers/grounded_sam.py` | DINO box(0.3/0.25, v1 demo 기본값) → SAM v1 box mask | Grounded-SAM (사용자 필수) |
| `providers/grounded_sam2.py` | `per_frame` / `track`(`CausalTrack`: streaming `init_state`→`add_new_frame`→`add_new_mask`→`infer_single_frame`), 0.4/0.3 | Grounded-SAM-2 |
| `providers/yoloe.py` | ultralytics YOLOE text prompt, conf 0.25 / IoU 0.7, `retina_masks=True`, cache·config 를 checkout 안으로 | YOLOE [25] |
| `eval/metrics.py` | 2D IoU/P/R, soft voxel grid · 지표(dev 정의), hard 지표, per-class FP mass, 집계, paired bootstrap(2000, seed 20261015), latency, found rate | 측정 정의 |
| `eval/offline_grounding.py` | harness: `--source-run --scene-gt --ats-run --providers name[:mode] --prompt-modes --robot-removal --size --save-masks --dry-run`, `create_run` 또는 `--out-dir`(outputs/ 아래만) | 측정 실행기 |
| `external/fetch_sources.py` · `sources.json` · `.gitignore` | 5 repo 를 pinned SHA 로 shallow fetch(detached), license 파일 sha256 기록, clone 본체 untracked | submodule 없이 고정 |
| `setup/requirements-baselines.in` · `requirements-baselines.txt`(lock, 84 줄) · `setup_venv.sh` · `paper_baselines_env.sh` | venv 생성, lock 설치, sam2 editable(`SAM2_BUILD_CUDA=0`, `--no-build-isolation`), cache 경로 | 환경 |
| `scripts/smoke_providers.py` | provider 별 CPU smoke(1 frame), track 은 causality check(3 frame), overlay png | 배선 확인 |
| `tests/test_interface.py` · `test_lift.py` · `test_metrics.py` · `test_grounded_sam2_track.py` · `test_offline_grounding.py` | 29 test | 단위 검증 |
| `README.md` · `docs/providers.md` | catalog(12 행), 실행법, recipe 대조표 | 문서 |

B 소유 파일(`providers/localization_heads.py`, `providers/tokencut.py`, `providers/frozen/`, B 의 docs · tests · `scripts/b_smoke_*`)은 수정하지 않았다.

## 환경 · 설치

- `.venv-baselines`: checkout `.python/cpython-3.11.16`, uv 0.12.13 (`AG3S_VLA-storage/tools/uv`), `--link-mode copy`, uv cache `.cache/uv`.
  - 첫 설치에서 NFS copy 오류(`Bad file descriptor`, torch 의 한 파일)가 나서 lock 전체를 `--reinstall` 로 다시 깔았다. 이후 import 정상.
  - `uv python find` 가 활성 `VIRTUAL_ENV`(.venv-openpi-live) 를 돌려줘서 setup script 는 interpreter 를 경로로 직접 지정한다.
- 주요 version (lock 발췌): torch 2.7.1+cu126 · torchvision 0.22.1+cu126 (`torch.version.cuda` = 12.6, PyPI CUDA wheel; nvidia-cudnn-cu12 9.5.1.17, triton 3.3.1) ·
  transformers 4.53.2 · huggingface-hub 0.34.4 · tokenizers 0.21.4 · safetensors 0.5.3 · accelerate 1.10.1 · sam-2 1.0 (editable, Grounded-SAM-2 @ b7a9c29) ·
  hydra-core 1.3.2 · iopath 0.1.10 · ultralytics 8.4.184 · clip (ultralytics/CLIP @ 7ffa84b) · opencv-python 4.12.0.88 · numpy 2.2.6 · scipy 1.16.1 ·
  scikit-image 0.25.2 · pillow 11.3.0 · matplotlib 3.10.5 · timm 1.0.19 · pytest 8.4.1.
- task 와 다른 점: `opencv-python-headless` 대신 `opencv-python` 하나만 pin 했다. ultralytics 가 `opencv-python` 을 요구하고 두 package 가 같은 `cv2` 를 덮어쓴다. `libGL.so.1` 은 시스템에 있다.
- YOLOE text prompt 는 `clip` (ultralytics fork) 이 필요하다. ultralytics 가 runtime 에 pip install 하지 않도록 pin 해 두고 `YOLO_AUTOINSTALL=False` 를 쓴다.
- 내가 만든 `/root/.config/Ultralytics/settings.json`(첫 import 때 생김)은 지웠다. 이후에는 `YOLO_CONFIG_DIR=.cache/paper_baselines/ultralytics_config` 를 쓴다.

## external sources (`external/fetch_sources.py`, 2026-10-23 fetch)

| name | SHA | LICENSE 파일 | license |
|---|---|---|---|
| Grounded-SAM-2 | b7a9c29f196edff0eb54dbe14588d7ae5e3dde28 | sha256 2e4d1dce… | Apache-2.0 |
| Grounded-Segment-Anything | 126abe633ffe333e16e4a0a4e946bc1003caf757 | 5e86c044… | Apache-2.0 |
| yoloe | 40cd606cabdbe2b566d6f14a6b162c89206e9a1b | 0d96a4ff… | **AGPL-3.0** (README 에 표시) |
| TokenCut | fed52cd5b60891baefd8ec7110dafa73be816ee1 | a8b8f8d9… | MIT |
| LocalizationHeads | 9ffe219d20ec376eb4dd14d42c54bb3299ffdb4a | **없음** | README 는 MIT 라고 적지만 LICENSE 파일이 없다 |

## weights (`.cache/paper_baselines/`, 전부 6.0 GB)

| id | 출처 · revision | 파일 | sha256 |
|---|---|---|---|
| grounding-dino-tiny (smoke) | HF IDEA-Research/grounding-dino-tiny @ a2bb814d | model.safetensors | 1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3 |
| grounding-dino-base (default) | HF IDEA-Research/grounding-dino-base @ 12bdfa31 | model.safetensors | 5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21 |
| sam-vit-base (smoke) | HF facebook/sam-vit-base @ 70c1a07f | model.safetensors | 892c410e496344e527255ccdcb2cb7244a609acb5389c7c4fdba1288f861c579 |
| sam-vit-huge (default) | HF facebook/sam-vit-huge @ 87aecf0d | model.safetensors | edfb0462392541fca9af44ff039bfb32dbd0c939997f3abb77a26e23af7afd7c |
| sam2.1-hiera-tiny (smoke) | HF facebook/sam2.1-hiera-tiny @ de431c40 | sam2.1_hiera_tiny.pt | 7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69 |
| sam2.1-hiera-large (default) | HF facebook/sam2.1-hiera-large @ 665f8e2a | sam2.1_hiera_large.pt | 2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318 |
| yoloe-11s-seg (smoke) | ultralytics assets v8.4.0 | yoloe-11s-seg.pt | 8e439445c87338b79d9ce21dec109f4621e26df67e94d26ea1a98c1e64dce3e3 |
| yoloe-11l-seg (default) | ultralytics assets v8.4.0 | yoloe-11l-seg.pt | a993fb0fc7c8830939ae14e6434a925dd1179428158c2761482eb8a8d8a3699f |
| mobileclip_blt (YOLOE text) | ultralytics assets v8.4.0 | mobileclip_blt.ts | a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54 |
| DINO ViT-S/16 (B 의 TokenCut, lead 요청) | dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth (TokenCut `networks.py:31,52` 의 URL) | `.cache/paper_baselines/torch/hub/checkpoints/dino_deitsmall16_pretrain.pth` (86,710,517 B) | 1566d50496f27f52f07fea6094fa29b2fdd6fae89da65bdd3ebc3b24ef6b7eb7 |

default(큰) weights 도 미리 받아 두었다. 측정 때 다운로드는 필요 없다.

## soft IoU 정의 (development 에서 읽기만 함)

출처: `/mnt/dev/AG3S_VLA-development/AG3S/ag3s/experiments/studies/ifg_graph.py` 의 `build_geometry`·`evaluate`, 그리고 `ifg_heldout_summary.py` (import 하지 않았다).
- eval voxel: 모든 camera 의 유효 depth pixel 을 `backproject(range_max=2.0)` 로 올려 `floor(p/0.005)` 격자에 넣은 합집합이다.
- `t(v)`: voxel 안 pixel 중 target class 의 비율.
- `TP = Σ_{v∈P} t`, `FP = Σ_{v∈P}(1−t)`, `FN = Σ_{v∉P} t`. IoU 는 이 셋으로 계산한다.
- `neighbor_fp`: other_object + destination class 비율을 P 위에서 더한 값.
- 집계: frame → episode 평균 → episode 간 평균. paired bootstrap 은 2000 회, seed 20261015, percentile 95 % 구간.

여기서 더한 것은 세 가지다.
- (a) eval voxel 에 없는 예측 voxel 은 FP 1 로 센다. 이런 경우는 ATS 점에서만 생기고, `pred_offgrid` 로 따로 보고한다.
- (b) hard 지표(`t ≥ 0.5`)도 함께 낸다.
- (c) scene GT 가 있으면 lift 된 점과 target 의 실제 visual surface 사이 거리를 낸다(`lift_within_10mm`, `lift_dist_median_mm`). 독립적인 lift 확인용이다.

class 는 `background, target, destination, other_object, fixture, support(main_table 등), robot, mount` 이다. LIBERO 의 cabinet · stove 같은 fixture 는 neighbor 에 넣지 않고 `fixture_fp` 로 따로 낸다.

## 단위 검증

```bash
source Baseline/paper_baselines/setup/paper_baselines_env.sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 .venv-baselines/bin/python -m pytest -q -p no:cacheprovider \
  Baseline/paper_baselines/tests/test_interface.py Baseline/paper_baselines/tests/test_lift.py \
  Baseline/paper_baselines/tests/test_metrics.py Baseline/paper_baselines/tests/test_grounded_sam2_track.py \
  Baseline/paper_baselines/tests/test_offline_grounding.py
# 29 passed in 2.74s
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 .venv-baselines/bin/python -m pytest -q -p no:cacheprovider Baseline/paper_baselines/tests
# 57 passed in 5.66s   (A 29 + B 28)
```
test 범위:
- interface 계약: not-found 경로, shape · dtype · 지어낸 mask 거부, top-1 tie 규칙, registry.
- gt_oracle.
- lift: 알려진 K · T round-trip, 무효 depth, robot 제거, policy mirror 규약(`camera.policy_to_opencv_pixel` 과 대조).
- metrics: 손으로 계산한 2-voxel soft IoU, off-grid, 집계, bootstrap 재현성(dev 식과 같은 CI).
- track causality: fake streaming predictor 로 t 이후 frame 이 달라도 t 까지의 결과가 같은지 본다.
- harness: label LUT, ATS slicing, 실제 SL3 dev frame 1 장의 배선.

## smoke (CPU, `CUDA_VISIBLE_DEVICES=""`, OMP 1 thread)

`scripts/smoke_providers.py` 를 SL3 dev `libero_spatial_t0_e0` obs 10 에 smoke weights 로 돌렸다. 출력은 `outputs/paper_baselines_smoke/smoke_<provider>.{json,png}` 다. 숫자는 배선 확인용이고 판정하지 않는다.

| provider | load s | RSS MB | head | left_wrist |
|---|---|---|---|---|
| gt_oracle | 0.0 | 150 | found, 769 px | found, 2281 px |
| grounded_sam (dino-tiny + sam-vit-base) | 32.7 | 4508 | not found (box < 0.3) | found 0.66, 2232 px |
| grounded_sam2:per_frame (dino-tiny + sam2.1-tiny) | 23.9 | 2743 | not found | found 0.66, 2254 px |
| grounded_sam2:track | 12.9 | 3342 | not found | found (track), 2284 px; **causality: frame 8, 9 mask 가 3 번째 frame 이 10 이든 40 이든 4/4 같음** |
| yoloe (11s) | 5.3 | 1389 | not found | not found |
| tokencut (B, 실제 DINO weight, lead 요청) | 0.4 | 719 | found, 8192 px | found, 7680 px; `load_msg` "All keys matched successfully" |
| localization_heads:sam_point (B, `SamV1Backend("sam-vit-base")`, lead 요청) | 38.4 | 3678 | found, 1748 px | found, 2232 px |

- YOLOE-11s 는 이 256 px LIBERO frame 에서 기본 conf 0.25 를 넘는 instance 가 없었다. 배선 확인 결과:
  - 같은 model 이 ultralytics `bus.jpg` 에서는 person · bus 를 0.53–0.92 로 찾는다.
  - conf 0.01 로 낮추면 "bowl" 후보 score 는 0.058(head) / 0.032(wrist) 였다.
  - 큰 모델(11l)의 결과는 측정하지 않았다.
- harness end-to-end: `outputs/paper_baselines_smoke/harness_smoke_t0e0_f10/` 에 결과가 있다.
  - 실행 조건: 1 episode × 1 frame × 5 provider-arm × prompt 2 종 + ATS, smoke weights, CPU, 539 s.
  - `frames.jsonl` 11 행, `episodes.jsonl`, `summary.json`(paired bootstrap 포함), `provenance.json`, `masks/` 가 나왔다.
  - `wiring_gt_oracle_ats/` 는 gt_oracle + ATS 를 3 frame 에 돌린 것이다.
- overlay `smoke_grounded_sam.png` 를 육안으로 봤다. GT target 외곽선이 OpenCV frame RGB 의 bowl 위에 맞게 놓인다.

## verifier 가 알아야 할 것

- 새 플래그 · 기본값 변경: 기존 code 에는 없음. 새 harness flag 는 위 "바뀐 파일" 참조. `--robot-removal` 기본값은 `none` 이다.
- 재생산이 필요한 산출물: 없음. 입력 위치는 다음과 같다.
  - segmentation 이 있는 recorded run 은 SL3 dev `20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0`(10 ep, scene GT `outputs/verify/SL4/raw/scene`, ATS `20261012T225820_46a08d6ff644_sl4-perception-dev`)와 SL4b dev `20261013T092537_714c891415d9_sl4b-dev-pi05only-t0-9e1-4`(40 ep, scene GT `outputs/verify/SL4b/raw/scene`) 두 개뿐이다.
  - SL1/SL2 replay-seg 는 t0e0 1 ep 씩이다. SL5* closed-loop run 에는 segmentation 이 없다.
- 옛 기록과 호환이 깨지는가: 없음 (기존 tracked 파일 무수정).
- 측정 전 주의:
  - (1) GPU 측정은 `--device cuda` 를 주거나 기본(cuda 자동)으로 돌린다. 공용 lock `/tmp/ag3s-vla-gpu-validation.lock` 아래에서만 돌린다. CPU smoke 의 latency 는 GPU latency 가 아니다.
  - (2) `grounded_sam2:track` 은 0 부터 마지막 평가 frame 까지 모든 recorded frame 을 순서대로 넣는다. 그래서 per-frame 보다 오래 걸린다. streaming state 의 image tensor 는 episode 안에서 계속 쌓인다(frame 당 1024² float).
  - (3) ATS 행은 perception npz 의 `target_points` 다. V11AB 의 voxel-downsample 대표점이라 5 mm voxel recall 이 구조적으로 낮게 나올 수 있다. latency 는 `t_process_s`(V11AB 전체 처리)라서 grounding 단독 시간이 아니다.
  - (4) segmentation 은 RGB · depth 보다 한 physics substep 늦게 render 된다(conventions_json). 그래서 gt_oracle 도 정확히 1 이 아니다.
  - (5) 2D 지표에서 not-found 인 camera 는 GT 가 보이면 IoU 0 으로 들어간다. GT 가 없으면 None 이다.

## 확인하지 못한 것

- GPU 실행, 큰(default) weights 로 돌리는 것, latency 측정. 모두 하지 않았다(금지 범위).
- HF Grounding DINO / SAM port 와 원 repo(GroundingDINO CUDA op) 출력 사이의 수치 동등성. 같은 weights 를 쓰고 같은 threshold 규칙을 적용한다는 근거만 있다(`docs/providers.md`).
- SAM2 CUDA extension 은 build 하지 않았다(`SAM2_BUILD_CUDA=0`). video predictor 의 `fill_hole_area=8` 후처리가 경고와 함께 생략된다(`provenance()["sam2_cuda_ext"]`). GPU 측정 전에 build 할지는 lead 가 정한다.
- task 예시의 "box 0.35" 는 공식 code 기본값이 아니다. `grounded_sam_demo.py:165-166` 의 0.3 / 0.25 를 썼다.
- 논문 bib 에 Grounded-SAM · Grounded-SAM-2 항목이 없다(YOLOE [25] 는 있음). catalog 에는 "not in bib" 으로 적었다.
- `CUDA_VISIBLE_DEVICES=""` 일 때 `/home/infinitix/ix-nvidia-lib/src/cuda/hook.c:475 [ixGPU ERROR]: cuInit failed:100` 이 stderr 에 찍힌다. 주입된 hook library 의 메시지이고 CPU 실행에는 영향이 없었다.
- B 관련 메모(수정하지 않음): `providers/localization_heads.py` docstring 41–43 행은 SAM wrapper 를 "`grounded_sam.py`" 라고 적는다. 실제 code(525–546 행)는 `sam_backend.SamV1Backend` 를 쓰고, 그 경로로 smoke 가 통과했다.

## 내가 기대하는 결과

(verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.)
- gt_oracle 의 3D soft IoU 는 1 에 가깝지만 1 은 아니다. 경계 voxel 의 혼합 비율과 segmentation substep 때문이다. smoke 3 frame 에서 0.965–0.977, `lift_within_10mm` ≈ 0.99 였다.
- phrase prompt 의 Grounded-SAM 계열은 wrist 에서는 잘 잡지만 head(멀리 · 작게 보임)에서는 threshold 를 못 넘는 frame 이 많을 것이다.
- instruction prompt 는 "the ramekin" 같은 다른 명사를 top-1 로 고르는 wrong-object 가 생긴다(smoke 에서 head 에서 관찰).
