# Catalog 초안 — VLA-internal grounding providers (Implementer B)

> writer: Implementer B (PAPER-BASELINES) · 2026-10-23 KST · lead 가 `README.md` catalog 에 합친다.
> 열은 task 문서의 catalog 열과 같다. 인용 번호는 `main.pdf` References: [14] Kang et al. · [26] Wang et al. (TokenCut).

## 1. catalog 행

| baseline | 인용 | 역할 (논문의 어느 표 · 주장) | 출처 repo + SHA | license | 구현 형태 | 상태 (2026-10-23) | 실행 명령 |
|---|---|---|---|---|---|---|---|
| `localization_heads` (mask `threshold`) | [14] | "VLA attention 으로 grounding" 의 **head 선택을 문헌 기준으로 한** 대조군. 우리 fixed cell 대신 Kang et al. 의 기준으로 고른 head 를 같은 lift · 지표로 잰다 (Tab. 2 비교 행 후보) | seilk/LocalizationHeads @ `9ffe219d20ec376eb4dd14d42c54bb3299ffdb4a` | **UNKNOWN**. pinned checkout 에 LICENSE 파일이 없다 (README 는 MIT 라고 적는다). code 는 가져오지 않고 다시 썼다 | `partial` (re-impl, numpy) | head 선택 완료 (dev 50 episode, 1,094 call) · frozen json 2개 · unit test 통과 · 실제 frame smoke 1장 (배선만) | 선택: `.venv-ag3s/bin/python -m Baseline.paper_baselines.providers.localization_heads select --attention-run <audit run> … --out <json>` · 예측: `get_provider("localization_heads", selection=<json>, mask_mode="threshold")` + `aux_from_attention_npz(npz, k)` |
| `localization_heads_sam_point` / `_sam_box` | [14] | 위와 같은 head + SAM 으로 mask 를 다듬는다. `sam_box` 가 논문의 recipe (box → SAM). `sam_point` 는 attention peak 점 prompt (task 지시) | 위 + SAM (A 의 `providers/sam_backend.py`) | 위 + SAM Apache-2.0 | `partial` | code · fake-SAM test 완료. **실제 SAM 실행 안 함** (A 의 `sam_backend` 가 아직 없었다) | `get_provider("localization_heads", selection=<json>, mask_mode="sam_box")` (`.venv-baselines`) |
| `tokencut` (target `foreground`) | [26] | language 없는 object discovery 의 대조군. "segmenter 없이 VLA 로" 주장의 반대편 (self-supervised feature 만으로 물체를 자른다) | YangtaoWANG95/TokenCut @ `fed52cd5b60891baefd8ec7110dafa73be816ee1` | MIT | `re-impl` + parity test (ncut 은 공식 함수와 mask · seed · eigenvector 일치) | parity 7 case 통과. DINO 경로는 random-init weight 로만 배선 확인. **DINO ViT-S/16 공식 weight 미다운로드** | `get_provider("tokencut")` (`.venv-baselines`, weight `dino_deitsmall16_pretrain.pth` 를 `$TORCH_HOME/hub/checkpoints/` 에) |
| `tokencut_attention_overlap` | [26] (변형) | **원 방법이 아니다.** TokenCut 의 component 중 noun/VLA attention 과 가장 많이 겹치는 것을 target 으로 고른다. "attention 이 물체를 고르고 feature 가 경계를 준다" 의 단순 대조군 | 위 | MIT | `re-impl` + 우리 선택 규칙 | unit test 통과 (synthetic). attention 입력과 VLA patch feature 생산자는 다음 STEP | `get_provider("tokencut", target_select="attention_overlap")` + `aux["attention_map"][cam]`, `aux["attention_orientation"]` |

## 2. 원 논문과 다른 점

### 2.1 Localization heads (Kang et al., CVPR 2025)

| 항목 | 원 논문 / 공식 code | 여기 | 영향 |
|---|---|---|---|
| 모델 | LLaVA-1.5 등 LVLM (32 layer × 32 head 류) | pi0.5 (`pi05_libero`). PaliGemma Gemma-2B prefix, 18 layer × 8 head | head 수가 144 개라 chord elbow 의 모양이 다르다 |
| query token | referring expression 의 **마지막 text token** (`collector.py:103`, 논문 Sec. 3) | **action expert 의 마지막 action token**, Euler denoise step 0 (`AGGREGATIONS["last"]`, `attention_audit.py` 의 `pooled_f16`) | 원 방법의 "text query → image" 가 아니라 "action query → image" 다. noun 하나를 따로 묻지 않는다. instruction 전체가 prompt 로 들어가 있다 |
| 이미지 | 한 장, P×P (LLaVA 24×24) | 카메라 2 대 (head · left_wrist), 각 16×16 | 카메라마다 따로 P×P 로 보고 **head 도 카메라별로** 고른다 |
| softmax 범위 | text + image token | prefix (image 768 + text) + suffix (action) key | image 몫의 합이 1 보다 작다 (예: t0 e0 의 21 call × 144 head 평균 head 0.10, left_wrist 0.29). 기준 1 의 절대값은 LLaVA 와 비교할 수 없다 |
| 표본 | RefCOCO train 1,000 장 × 5 trial | LIBERO Spatial **dev** (init 0–4) 50 episode, policy call 1,094 개. test (init 5–49) 는 쓰지 않았다 | 같은 장면이 반복되는 표본이다 (task 10 개) |
| 기준 1 | 논문: 1,000 표본 **평균** S_img 에 max-curvature 임계. 공식 code: **표본마다** chord elbow (`analyze.py:82`) | `protocol="official_code"` (기본) = code 대로 표본마다. `protocol="paper"` = 평균 | 두 json 모두 남겼다 |
| 기준 2 | 논문: mean 위를 1, component **면적**으로 p (Eq. 3, App. B.1). code: `relu(S − 2·mean) > 1e-3`, component **질량**으로 p, 아래 행 >0.05 인 head 제외 (`analyze.py:20-42`, `:98`) | `official_code` = code 식 그대로 (bottom-row 규칙 포함). `paper` = 논문 식 (bottom-row 규칙 없음) | bottom-row 규칙은 LLaVA 의 attention sink 에 맞춘 절대 문턱 (0.05) 이다. pi0.5 에도 그대로 적용했다 |
| 층 제외 | 처음 두 층 (논문 Sec. 4, code `layer > 1`) | 같음 (`min_layer=2`) | — |
| 표본 간 집계 | 논문 Sec. 4.2: 10-lowest entropy 에 든 빈도, 상위 k. **공식 code 에는 집계가 없다** (표본별 순위만) | 논문 빈도 규칙 (top-10, k = 3). 동점은 평균 entropy → layer → head 순 | code 의 demo yaml `top_k: 5` 대신 논문 main 의 k = 3 을 썼다 (`--top-k` 로 변경 가능) |
| mask | Gaussian σ 1 (논문 kernel 7, code 는 scipy 기본 = 9 tap) → 합 → mean 위 → (논문) 가장 큰 convex hull 의 bbox → SAM box prompt | code 그대로: σ 1, scipy 기본 truncate, mean 위, bilinear upsample > 0.5 (`bbox.py`). **convex hull 단계는 없다** (code 에도 없다) | `threshold` mask 는 component 가 여럿이면 모두 포함한다 |
| SAM | box prompt (논문 Sec. 5) | `sam_box` = 논문 방식. `sam_point` = attention peak 한 점 (task 지시, 원 방법 아님) | — |
| 픽셀 frame | — | pi05_libero grid 는 policy 방향 (좌우 반전). `aux["attention_orientation"]="libero_policy"` 면 `AG3S.adapters.panda.attention.unmirror_grid` 로 grid 를 되돌린 뒤 RGB 크기로 resample 한다. test 가 A 의 `lift.policy_mask_to_opencv` 와 같은 결과임을 확인한다 | orientation 을 안 주면 오류로 멈춘다 |
| 수치 정밀도 | torch float32 | numpy float32 | entropy 가 0 근처 (component 1 개) 인 head 끼리는 순서가 float 잡음 (|ΔSE| < 1e-6) 으로 바뀔 수 있다. test 는 이 잡음을 뺀 순서 일치를 본다 |

**frozen selection (dev 만, 2026-10-23):** `Baseline/paper_baselines/providers/frozen/`

| 파일 | camera | 선택 head (layer, head) · 빈도 / 1,094 |
|---|---|---|
| `localization_heads_pi05_libero_spatial_dev_init0-4_official_code.json` (`selection_sha256` `07eb43c3…`) | head | (4,0) 1076 · (4,4) 1054 · (6,0) 863 |
| | left_wrist | (8,2) 816 · (5,5) 813 · (7,3) 776 |
| `localization_heads_pi05_libero_spatial_dev_init0-4_paper.json` (`b959ce73…`) | head | (4,0) 1088 · (4,4) 1020 · (7,7) 927 |
| | left_wrist | (5,5) 984 · (5,3) 924 · (7,3) 905 |

- 입력은 SL3 dev audit `20261012T202459_309e4c6da477_sl3-attention-audit-dev` (init 0) 와 SL4b `20261013T094739_714c891415d9_sl4b-attention-audit-dev-e1-4` (init 1–4) 다.
- episode npz 50 개의 sha256 이 json `inputs` 에 있다.
- run 순서를 바꿔 다시 돌려도 `selection_sha256` 이 같았다.
- 이 표는 head 목록일 뿐이다. grounding 품질은 verifier 가 잰다.

### 2.2 TokenCut (Wang et al., CVPR 2022)

| 항목 | 원 방법 / 공식 code | 여기 | 영향 |
|---|---|---|---|
| ncut | `object_discovery.py:12-92` | 같은 식을 numpy 로 다시 썼다. binary graph (τ 0.2, ε 1e-5), `eigh(D−A, D)` 2 번째 고유벡터, 평균 이분, \|y\| 최대 seed 쪽이 전경 | parity: mask · seed · label · bbox 동일. 고유벡터는 binary graph 에서 1e-10, similarity graph (ablation) 에서 1e-6 이내 (float32 matmul 차이) |
| 전경 component | **seed 가 든 connected component** (4-연결, `ndimage.label` 기본) | 같음 | task 문서의 "largest connected component" 는 공식 code 와 다르다. code 를 따랐다 |
| feature | DINO ViT-S/16, 마지막 block 의 key (`--which_features k`), ImageNet 정규화, 16 배수로 zero-pad | 같음. 모델 code 는 pinned clone 의 `dino/vision_transformer.py` 를 import 하고, weight URL 은 `networks.py` 와 같다 | 공식 weight 는 아직 받지 않았다 (impl.md) |
| 해상도 | 원본 이미지 (VOC 등) | LIBERO 256×256 → 16×16 patch | 물체가 몇 patch 밖에 안 된다. 경계가 16 px 단위다 |
| VLA feature 옵션 | — | `feature_source="aux"`: `(rows, cols, D)` patch feature + orientation | 이 feature 를 남기는 생산자가 아직 없다. attention audit 의 `kv_feature` 는 512-d pooled 값이다 |
| language | 쓰지 않는다 | `foreground` 는 query 를 무시한다 | 여러 물체 장면에서 target 이 아닌 물체를 잡을 수 있다. 원 방법의 성질이다 |
| `attention_overlap` | **없음** | 우리 규칙: 양쪽 bipartition 의 모든 4-연결 component 중 `Σ relu(att − mean(att))` 가 가장 큰 것. 동점이면 seed component. attention 이 상수면 `found=False` | 원 방법이 아니다. provenance 에 `target_select_is_original_method: false` 로 남긴다. target 과 배경이 같은 component 로 붙으면 배경까지 함께 고른다 |

## 3. 다음 STEP 에 필요한 것

- `localization_heads` 를 다른 frame 에 쓰려면 그 frame 의 `pooled_f16` 이 있어야 한다. 현재는 SL3 · SL4b audit 의 policy call 만 있다. ATS 비교 frame 과 같은 call 이면 바로 쓸 수 있다.
- TokenCut DINO 경로에는 `dino_deitsmall16_pretrain.pth` (87 MB) 가 필요하다. 다운로드는 A 담당이다.
- VLA patch feature 옵션에는 pi0.5 SigLIP patch token 을 남기는 audit 확장이 필요하다. `benchmark/runners/**` 수정이라 run 이 없을 때 한다.
