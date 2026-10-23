# PAPER-BASELINES — 논문 비교 baseline 폴더 (external segmenter · VLA-internal grounding · GT oracle)

> writer: lead · 읽는 쪽: implementer A / B (다음에 verifier) · 작성 2026-10-23 KST
> 사용자 지시 2026-10-23: "논문 baseline 목록을 하나의 baseline 폴더로 만들고, 오픈소스가 있으면 가져오고 없으면 일정 부분 구현".
> 사용자 요청: **Grounded-SAM 은 반드시 포함.**

## 왜 지금 이것인가

논문(`main.pdf`, CVPR 2027 draft)의 핵심 주장 하나가 "observed target surface 를 **external segmenter 없이** frozen VLA 의
attention 으로 grounding 한다"이다. 그런데 비교 숫자가 비어 있다.

- Tab. 2 의 "External segmenter" 행
- §4.1 의 TBD "Grounded-SAM / YOLOE masks lifted with the same depth"
- Fig. 2 의 "Prior: external segmenter" 열

이 STEP 은 그 비교를 돌릴 수 있는 **baseline 코드와 offline 평가 harness** 를 만든다.
측정은 이 STEP 범위 밖이며, 다음 verifier STEP 이 GPU lock 아래에서 한다.

```
RGB(+instruction) ──[baseline provider]──> per-camera target mask
                                  │
depth + K + T_base_cam ──[lift]───┴──> base-frame target points / 5 mm voxels ──> metrics vs GT
                                       (closed-loop: 같은 points 를 LTE/CCR 에 넣는 것은 다음 STEP)
```

## 고정 결정 (lead)

| 항목 | 결정 | 이유 |
|---|---|---|
| 폴더 | `Baseline/paper_baselines/` 하나 | 사용자 지시. 기존 `Baseline/knows_vla`, `Baseline/seam_vla`, `benchmark/SafeLIBERO`(VLSA)는 **옮기지 않고** README catalog 에서 링크만 한다 (옮기면 기존 run · import 가 깨진다) |
| 외부 code | `external/<name>/` 에 **pinned SHA 로 fetch** 하는 script + `external/sources.json` manifest. `external/.gitignore` 로 clone 본체는 추적하지 않는다 | `git submodule add` 는 index 를 바꾸는데, 같은 checkout 에서 다른 session 이 commit 중이다. submodule 전환은 사용자가 정한다 |
| venv | 새 `.venv-baselines` (py3.11, checkout `.python` 사용, uv) | `.venv-ag3s`(numpy 2.4.6, pipeline)와 `.venv-openpi-live` 를 오염시키지 않는다 |
| cache · weights | `HF_HOME` · `TORCH_HOME` · ultralytics weights → checkout `.cache/paper_baselines/` | 규칙: cache 는 checkout 마다 독립. 공유 storage 에 새 checkpoint 를 쓰지 않는다 |
| GPU | **이 STEP 에서는 쓰지 않는다.** smoke 는 `CUDA_VISIBLE_DEVICES=""` CPU, 가장 작은 weight | GPU 는 다른 session 이 쓰고 있다. 측정은 verifier 가 lock 아래에서 |
| 손대지 않을 것 | `AG3S/**`, `benchmark/runners/**`, `AG3S/adapters/panda/profile.py`, `integration/**`, `third_party/**`, `.gitmodules`, 루트 `.gitignore` | V11AB 평가 대상 불변 + 진행 중 run 보호 (memory: no runner edits during runs) |
| 결과 위치 | 평가 harness 는 `common/run_artifacts.create_run` 으로 `AG3S_VLA-storage/results/benchmark/<날짜_commit_run-id>/` 에 쓴다. implementer smoke 출력은 `outputs/paper_baselines_smoke/` | 기존 규칙 |

## 폴더 구조 (목표)

```
Baseline/paper_baselines/
├── README.md                 catalog (A 소유) — 아래 "catalog 열" 참조
├── __init__.py
├── interface.py              (A) TargetQuery · TargetMaskResult · TargetMaskProvider · registry
├── lift.py                   (A) mask + depth + K + T → base-frame points / voxels
├── providers/
│   ├── gt_oracle.py          (A) simulator segmentation → mask (상한선)
│   ├── grounded_sam.py       (A) Grounding DINO + SAM (v1 recipe)
│   ├── grounded_sam2.py      (A) Grounding DINO + SAM 2 (+ video tracking)
│   ├── yoloe.py              (A) YOLOE text prompt seg
│   ├── localization_heads.py (B) Kang et al. CVPR 2025 의 head 선택을 pi0.5 attention 에 적용
│   └── tokencut.py           (B) TokenCut normalized cut (re-impl + parity test)
├── eval/
│   ├── metrics.py            (A)
│   └── offline_grounding.py  (A) recorded frames → providers → metrics (create_run)
├── external/                 (A) fetch_sources.py · sources.json · .gitignore
├── setup/                    (A) requirements-baselines.txt (pinned) · setup_venv.sh
├── docs/                     (A: providers.md / B: survey.md, exemption_ablations.md, catalog_vla_internal.md)
└── tests/                    (A, B 각자 자기 provider 의 test)
```

## 공통 interface (A 가 먼저 만든다. B 는 이 spec 대로 코딩한다)

```python
@dataclass(frozen=True)
class TargetQuery:
    instruction: str                 # 원문 instruction
    target_phrase: str               # 예: "black bowl" — BDDL/GT 에서 온 target 이름 또는 noun phrase
    prompt_mode: str = "phrase"      # "phrase" | "instruction": 어떤 text 를 detector 에 줄지
    episode_id: str | None = None
    frame_index: int | None = None

@dataclass
class TargetMaskResult:
    camera: str
    mask: np.ndarray                 # bool (H, W), **입력 RGB/depth 와 같은 pixel frame**
    found: bool                      # 검출 실패면 False + 빈 mask. 절대 지어내지 않는다 (ATS 원칙과 같음)
    score: float | None
    boxes: np.ndarray | None         # (N, 4) xyxy, 후보 전부
    phrases: list[str]
    latency_ms: float
    extras: dict                     # provider 고유 기록 (선택한 box index, head 목록 등)

class TargetMaskProvider(Protocol):
    name: str
    def provenance(self) -> dict: ...      # repo + SHA, weights id + sha256, threshold 등 전부
    def reset(self) -> None: ...           # episode 경계 (tracker 초기화)
    def predict(self, rgb: dict[str, np.ndarray], query: TargetQuery,
                *, aux: dict | None = None) -> dict[str, TargetMaskResult]: ...
```

- `aux` 에는 provider 가 따로 필요한 입력을 넣는다: GT segmentation(oracle), attention grid(localization heads), VLA patch feature(TokenCut 옵션).
- **pixel frame 주의:** `AG3S_BENCHMARK_LOG.md` 562행 부근에 따르면 OpenCV image 와 policy 224 image 는 좌우가 뒤집혀 있다. mask 는 depth 와 같은 frame 이어야 lift 가 맞는다. 변환이 필요하면 `lift.py` 한 곳에서 하고 test 로 고정한다.
- 후보가 여럿이면 **score 최고 1개** 를 target 으로 고른다. GT 는 쓰지 않는다. 선택 규칙은 provenance 에 남긴다.
- dataset 에 의존하지 않는다. LIBERO 전용 코드는 `eval/` 과 `gt_oracle` 에만 둔다. 나중에 development branch 의 RB-Y1 held-out 34 episode 에도 같은 provider 를 쓴다.

## Implementer A — 공통 · external segmenter · oracle · 평가

1. **환경**
   - `setup/setup_venv.sh` 로 `.venv-baselines` 를 만든다. 버전은 전부 pin.
     - torch / torchvision: driver 에 맞는 CUDA wheel. 나중에 GPU 측정을 하므로 CPU wheel 이 아니다.
     - 그 외: transformers, `sam2`(Grounded-SAM-2 의 pinned clone 을 editable 설치하거나 같은 SHA 의 공식 `facebookresearch/sam2`), ultralytics(YOLOE 포함 버전), scipy, opencv-python-headless, pillow, pytest.
   - B 가 쓸 패키지도 여기에 포함한다. B 는 설치하지 않는다.
2. **external fetch** — `external/fetch_sources.py` + `sources.json`. 확인된 HEAD(2026-10-23, `git ls-remote`):

   | name | repo | HEAD |
   |---|---|---|
   | Grounded-SAM-2 | IDEA-Research/Grounded-SAM-2 | `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28` |
   | Grounded-Segment-Anything (v1, 참조만) | IDEA-Research/Grounded-Segment-Anything | `126abe633ffe333e16e4a0a4e946bc1003caf757` |
   | yoloe (참조. 실행은 pinned ultralytics) | THU-MIG/yoloe | `40cd606cabdbe2b566d6f14a6b162c89206e9a1b` |
   | TokenCut (B parity test 용) | YangtaoWANG95/TokenCut | `fed52cd5b60891baefd8ec7110dafa73be816ee1` |
   | LocalizationHeads (B 용) | seilk/LocalizationHeads | `9ffe219d20ec376eb4dd14d42c54bb3299ffdb4a` |

   - 각 repo 의 license 를 읽어 `sources.json` 에 기록한다. YOLOE 는 AGPL-3.0 으로 알려져 있으니 확인해 README 에 표시한다.
3. **providers**
   - `grounded_sam`: HF `GroundingDinoForObjectDetection` + `SamModel`(공식 weights) 로 Grounded-SAM v1 recipe 를 만든다. 원 repo 의 GroundingDINO CUDA op build 를 피하기 위해서다. 동등성 근거(같은 weights, 같은 threshold 기본값)를 `docs/providers.md` 에 쓴다.
   - `grounded_sam2`: Grounded-SAM-2 의 HF-DINO demo 경로를 따른다.
     - 모드 `per_frame`: 매 frame 검출 + SAM2 image predictor.
     - 모드 `track`: 처음 found 된 frame 에서 prompt 한 뒤 SAM2 video predictor 로 forward 전파한다. **causal 이어야 한다** — frame t 의 결과가 t 이후 frame 에 의존하지 않는다는 것을 test 로 확인한다.
   - `yoloe`: ultralytics YOLOE seg, text prompt = `target_phrase`.
   - 기본 threshold 는 각 공식 demo 값을 쓴다 (예: Grounded-SAM 의 box 0.35 / text 0.25 — 공식 code 에서 확인). **held-out 결과로 고르지 않는다.**
   - weights: 기본은 각 계열의 큰 모델, smoke 는 가장 작은 모델. id 와 sha256 을 provenance 에 남긴다.
   - `gt_oracle`: `_geom.npz` 의 `segmentation_geom/<camera>` + GT target geom id → mask.
     - GT target 해석은 `Baseline/knows_vla/libero_env.py`(BDDL goal parsing), `benchmark/evaluators/libero_scene_gt.py` 를 재사용한다.
4. **lift** — mask + depth + intrinsics/extrinsics → base-frame points.
   - 이미 검증된 camera 변환(`wire_pack` / `libero_scene_gt` 등)을 import 해 재사용한다. 새로 짜지 않는다.
   - robot 제거 옵션 `none | gt_robot_seg`: 모든 provider 에 같은 옵션을 적용해야 공정하다.
   - test: 알려진 K · T 로 round-trip, 그리고 mirror 규약.
5. **metrics + harness** — `eval/offline_grounding.py`
   - 입력: `--capture-segmentation` 이 켜진 rollout run 의 `_geom.npz`, `--scene-gt` 디렉터리. 위치는 `AG3S_BENCHMARK_LOG.md` 426 · 482 · 1452 · 2064행 부근과 SL-1/SL-4 run 을 보고 찾는다.
   - 옵션 `--ats-run <perception_replay run>`: V11AB ATS 의 target points 를 같은 frame 에 대한 비교 행으로 읽는다. pseudo-provider `ats_v11ab`.
   - 지표:
     - 2D: per-camera IoU · precision · recall
     - 3D (5 mm voxel): IoU · precision · recall, neighbour leak (GT 비-target 물체 voxel 에 들어간 예측 voxel 수), found rate, latency median/p95
     - 집계: episode 평균 → episode 간 평균
     - paired bootstrap 2,000회 (provider 쌍, episode 단위)
   - **soft IoU:** 논문 Tab. 2 는 soft IoU 를 쓴다. `/mnt/dev/AG3S_VLA-development` 의 정의를 **읽기만** 하고(import 금지) 같은 정의로 구현한다. 정의를 찾지 못하면 hard IoU 만 구현하고 impl.md 에 그 차이를 적는다.
   - 출력: `create_run` → `frames.jsonl`, `episodes.jsonl`, `summary.json`, provenance(provider + source run SHA-256).
6. **tests** (`Baseline/paper_baselines/tests/`)
   - synthetic: interface 계약, not-found 경로, lift round-trip, metrics toy case, bootstrap 재현성(seed)
   - provider 별 CPU smoke 는 test 가 아니라 `outputs/paper_baselines_smoke/` 에 결과를 남기는 script 로 둔다. recorded frame 1장, 최소 weight. 숫자 판정은 하지 않고 배선만 확인한다.
7. **README.md catalog** — 열: baseline · 논문 인용 번호 · 역할(논문의 어느 표 · 주장) · 출처 repo + SHA · license · 구현 형태(`vendored-run` / `wrapper` / `re-impl` / `partial` / `survey-only` / `existing`) · 상태 · 실행 명령.
   - 행:
     - 이 STEP 의 6 provider
     - 기존 KNOWS(`Baseline/knows_vla`), VLSA/AEGIS(`benchmark/SafeLIBERO`), SEAM(`Baseline/seam_vla`)
     - π0.5-only, Coarse(V11AB)
   - B 의 행은 B 가 `docs/catalog_vla_internal.md` 에 쓰고 lead 가 나중에 합친다.

## Implementer B — VLA-internal grounding · survey · exemption ablation 설계

1. **`localization_heads`** (partial re-impl)
   - 고정 SHA 의 `external/LocalizationHeads` code(없으면 A 의 fetch script 를 실행한다)와 논문(arXiv:2503.06287)에서 **head 선택 기준**을 그대로 읽는다. 예: image attention 비율 상위 layer, spatial entropy, 빈도 기준 top-k.
   - 그 기준을 pi0.5(PaliGemma) prefix 의 recorded per-(layer, head) noun→image attention grid 에 적용한다. grid 는 `benchmark/runners/attention_audit.py` 가 남기는 diagnostic grid 다. 정확한 key · shape 를 확인한다.
   - head 선택은 **dev episode 에서만** 하고, frozen head 목록 json 으로 저장한다.
   - mask 모드:
     - `threshold`: 선택 head map 합 → upsample → 임계
     - `sam_point`: peak point 로 SAM 을 prompt. A 의 SAM wrapper 를 import 하고, 없으면 TODO 로 남긴다.
   - 원 논문과 다른 점(모델 · token 정의 · 단일 noun query 등)을 `docs/catalog_vla_internal.md` 에 적는다.
2. **`tokencut`** (re-impl + parity)
   - 공식 `object_discovery.ncut` 를 재구현한다: binarized cosine affinity(τ=0.2, ε=1e-5), generalized eigenproblem 의 second smallest eigenvector, mean bipartition, foreground = argmax |eigvec| 쪽, largest connected component.
   - **parity test**: 같은 random feature 에서 pinned 공식 code 와 같은 mask 가 나와야 한다.
   - feature: DINO ViT-S/16(공식 기본) 또는 `aux` 로 받은 VLA patch feature.
   - TokenCut 은 언어를 쓰지 않는다. target 선택 옵션 `foreground` | `attention_overlap`(noun attention 과 가장 많이 겹치는 component)을 두고, 후자가 원 방법이 아니라는 점을 명시한다.
3. **survey (구현하지 않음)** — `docs/survey.md`
   - 대상: [1] multi-link safety filter, [5] constrained flow matching, [9] zero-shot reactive avoidance, [11] RoboGround, [13] GuidedVLA, [15] Lan-o3dp (arXiv:2407.00451), [19] OmniGuide (arXiv:2603.10052), [24] OBEYED-VLA.
   - 각 항목: 공개 code 유무(URL + HEAD SHA 또는 "없음, 확인 날짜"), license, checkpoint, SafeLIBERO / LIBERO 보고 수치 유무, 우리 protocol 에 넣는 비용 추정.
   - primary source 만 쓴다.
4. **exemption ablation 설계 (code 수정 금지)** — `docs/exemption_ablations.md`
   - **No exemption**(모든 observed surface 가 obstacle)과 **All-link exemption**(accepted target 을 모든 link 에서 제거)을 다룬다.
   - 현재 V11AB flag / config 로 표현 가능한지 조사한다: `AG3S/adapters/panda/profile.py` 의 argv tuple, `AG3S/ag3s/constraints/constraint_builder.py` 의 `contact.contact_links`.
   - 결과로 정확한 argv · config 를 제시하거나, 필요한 최소 code 변경(flag, 기본 off, 기본 bit-identical)을 `파일:줄` 단위로 제안한다. 실제 수정은 다음 STEP 에서 run 이 없을 때 한다.
5. **tests**: tokencut parity, localization-head 선택의 결정성(같은 입력 → 같은 head 목록), threshold mask 의 not-found 경로.
6. torch 가 필요한 test 는 `.venv-baselines/bin/python -c "import torch"` 가 될 때만 돌린다.
   - 아직 안 되면 numpy 부분과 survey · 설계를 먼저 끝내고, 마지막에 다시 확인한다.
   - 그래도 안 되면 미실행으로 impl.md 에 쓴다. **B 는 venv 에 설치하지 않는다.**

## 공용 자원 규칙 (둘 다)

- CPU worker 는 각자 1개. 큰 설치와 download 는 A 만 한다.
- RSS 를 작게 유지한다. smoke 는 최소 weight, frame 1장으로 한다.
- GPU 를 쓰지 않는다 (`CUDA_VISIBLE_DEVICES=""`). 다른 session 의 process 를 건드리지 않는다.
- git commit 을 하지 않는다. 사용자가 한다.
- 끝나면 `AG3S/ag3s/docs/handoff/PAPER-BASELINES-<A|B>.impl.md` 를 쓴다(템플릿 `_TEMPLATE.impl.md`). 내용:
  - 바꾼 파일
  - test 결과 (실제 출력)
  - smoke 결과
  - 확인하지 못한 것

## 측정에 요청하는 것 (→ verifier, 다음 STEP · 사용자 판정 후)

| # | 무엇을 재나 | 어떤 숫자로 | 통과 기준 |
|---|---|---|---|
| 1 | LIBERO dev frame 에서 provider 별 target grounding | 2D/3D IoU · precision · neighbour leak · found rate | 비교용. 통과 기준 없음. 단 gt_oracle 3D IoU ≈ 1 (lift 검증) |
| 2 | 같은 frame 의 ATS(V11AB) 대비 paired gain | bootstrap 95% CI | 보고만 |
| 3 | latency | median / p95 ms per frame per camera (GPU) | 보고만 |
| 4 | "two black bowls" 류 referring-expression task | phrase / instruction prompt 별 wrong-object rate | 보고만 |

## 이 STEP 이 답하는 질문

논문의 "external segmenter" 비교 행(Grounded-SAM 포함)을 **같은 depth · 같은 lift · 같은 지표로** 채울 수 있는 실행 가능한 baseline 이 이 checkout 에 있는가?

## 되돌아올 지점

- **closed-loop(G2)** 은 다음 STEP 이다. baseline points 로 ATS target 을 대체하는 hook 을 V11AB server 에 flag 로 넣는다. run 이 없을 때 하고, 기본은 bit-identical 이어야 한다.
- **submodule 전환:** 사용자가 원하면 `external/` fetch 방식을 pinned submodule 로 바꾼다.
- **RB-Y1 held-out 34 episode (논문 Tab. 2 의 실제 집합):** development checkout 의 eval 에 같은 provider 를 붙이는 일은 별도 task 다.
