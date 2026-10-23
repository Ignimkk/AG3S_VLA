# PAPER-BASELINES-2 (Implementer E) — 구현

> writer: ag3s-implementer (Implementer E) · 읽는 쪽: verifier, lead, scribe · 2026-10-23 13:34 KST
> checkout `/mnt/dev/AG3S_VLA-benchmark`, branch `feature/panda-benchmark`, HEAD `bf1403d5` (commit 하지 않았다)

## 무엇을 했나 (평이한 요약 먼저)

`offline_grounding.py` 가 VLA 내부 baseline 두 개(localization heads, TokenCut)를 다른 provider 와 같은 frame · 같은 lift · 같은 지표로 돌릴 수 있게 했다.

- `--attention-run <attention_audit run>` 을 주면 평가 frame 마다 그 frame 의 policy call 에서 기록한 pi0.5 attention 을 `aux` 로 넘긴다. 짝은 `obs_index` 로 맞춘다.
- `--providers` 에 `localization_heads:threshold|sam_point|sam_box`, `tokencut:foreground|attention_overlap` 을 쓸 수 있다.
- 두 provider 는 query text 를 읽지 않는다. 그래서 prompt mode 마다 같은 결과를 두 번 내지 않고 한 번만 돈다. arm 이름은 `<spec>|no_text` 다.

그 밖의 변경:
- `setup_venv.sh` 의 SAM2 CUDA build 기본값을 1 로 바꿨다. lead 의 build 명령과 같다.
- `localization_heads.py` docstring 의 SAM wrapper 위치를 `sam_backend` 로 고쳤다.

**측정은 하지 않았다.** CPU 에서 dry-run 두 개와 1 episode × 2 frame 배선 확인만 했다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `Baseline/paper_baselines/eval/offline_grounding.py:13-19, 38-43` | module docstring 에 `--attention-run` 설명과 SL3 실행 예를 추가 | 사용법 |
| `…/offline_grounding.py:242-337` | `AttentionRun` 클래스. audit run 의 `attention_index.jsonl` 을 `<suite>_t<task>_e<ep>` 로 색인한다. `aux(stem, t)` 가 넘기는 것 두 가지: (1) `localization_heads` 용 `pooled_f16[k]` (B 의 `aux_from_attention_npz` 그대로); (2) `tokencut:attention_overlap` 용 `attention_map[camera] = cell_grid_policy[k, i]` (V11AB cell L8H2). 두 입력 모두 `attention_orientation="libero_policy"` 다. `check_episode` 는 audit 의 `source_run` 과 `npz_sha256` 이 `--source-run` episode 와 다르면 멈춘다. `describe` 는 dry-run 에서 입력 grid 를 점검한다 | task E-1 |
| `…/offline_grounding.py:340-400` | `MODE_KWARG` 는 `:mode` 를 각 provider 의 생성자 인자로 보낸다 (`localization_heads` → `mask_mode`, `tokencut` → `target_select`, 그 밖에는 `mode`). `TEXT_FREE` · `DEFAULT_LH_SELECTION` (frozen `…_official_code.json`) · `needs_attention` · `provider_kwargs` 를 추가했다. `size` 와 `device` 는 그것을 받는 provider 에만 넘긴다. 예전에는 모든 provider 에 `size=` 를 넘겨서 localization_heads · tokencut 생성이 TypeError 로 실패했다. `sam_point` / `sam_box` 의 SAM 은 `grounded_sam` 과 같은 preset weight 다 (`default` = `sam-vit-huge`, `smoke` = `sam-vit-base`). `--device` 가 없으면 tokencut 은 `default_device()` 를 쓴다. 이유: TokenCutProvider 자체 기본값은 cpu 다 | task E-1 |
| `…/offline_grounding.py:441-444` | attention 이 필요한 provider 가 있는데 `--attention-run` 이 없으면 시작 전에 멈춘다 (dry-run 은 예외) | 조용한 실패 방지 |
| `…/offline_grounding.py:455-459, 503-525` | text-free provider 의 arm 은 한 번만 만든다 (`no_text`, `prompt=None`). 평가 frame 에 audit call 이 없으면 `missing_reason="no_attention_record_for_frame"` 행으로 남긴다. not-found 로 세지 않는다. 행에는 `attention_call` (call index) 이 들어간다 | task E-1 |
| `…/offline_grounding.py:473-474, 493-498` | episode 마다 audit join 을 검사한다. dry-run 행에 attention 점검 값과 `ats_record` 를 넣는다 | task E-4 |
| `…/offline_grounding.py:564-574` | `provenance.json["attention"]`: audit run, manifest sha256, episode attn npz sha256, provider 별 `selection_in_sample` (frozen selection 의 `inputs` 에 이 audit run 이 들어 있으면 `true`) | verifier 의 `in_sample: true` 근거 |
| `…/offline_grounding.py:625-631, 647` | dry-run summary 에 `n_episodes` · `n_frames_with_attention` · `n_frames_with_ats` 를 넣었다. arm summary 에 `n_missing` 을 넣었다 | 점검 |
| `…/offline_grounding.py:661-667` | CLI `--attention-run`, `--lh-selection`. `--providers` help 갱신 | |
| `Baseline/paper_baselines/setup/setup_venv.sh:6-14, 35-44, 62` | `SAM2_BUILD_CUDA=${SAM2_BUILD_CUDA:-1}`. 1 이면 `CUDA_HOME` (기본 `/usr/local/cuda-12.6`), `TORCH_CUDA_ARCH_LIST` (기본 9.0), `MAX_JOBS` (기본 2) 를 export 한다. `$CUDA_HOME/bin/nvcc` 가 없으면 exit 3. 설치는 `--no-build-isolation --no-deps --reinstall-package sam2 -e external/Grounded-SAM-2` 로 lead 명령과 같다. `--check` 목록에 `sam2._C` 를 추가했다 | task E-2 |
| `Baseline/paper_baselines/providers/localization_heads.py:42-44` | docstring 의 "SAM wrapper (`grounded_sam.py`)" 를 "`providers.sam_backend.SamV1Backend` (`predict_points` / `predict_box`)" 로 고쳤다. code 는 바꾸지 않았다 | task E-3 (A 의 지적) |
| `Baseline/paper_baselines/tests/test_offline_grounding.py:68-177` | 새 test 5 개 (아래) | |

`setup_venv.sh` 는 문법 (`bash -n`) 만 확인했다. 다시 돌리지 않았다. 이유: lead 가 이미 build 했고, 다시 돌리면 sam2 를 reinstall 한다.

## 단위 검증

```bash
source scripts/setup/env.sh
source Baseline/paper_baselines/setup/paper_baselines_env.sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=$PWD .venv-baselines/bin/python -m pytest -p no:cacheprovider -q Baseline/paper_baselines/tests
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest -p no:cacheprovider -q \
    Baseline/paper_baselines/tests/test_offline_grounding.py Baseline/paper_baselines/tests/test_localization_heads.py
```

- `.venv-baselines`: **62 passed** (전체 `Baseline/paper_baselines/tests`, 7.2 s)
- `.venv-ag3s`: 20 passed, 2 skipped (torch parity test)

| 새 test | 확인하는 것 |
|---|---|
| `test_parse_arm_vla_internal_modes` | `:mode` 가 `mask_mode` / `target_select` 로 가는지. `needs_attention` 이 localization_heads (모든 mode) 와 `tokencut:attention_overlap` 에서만 참인지 |
| `test_provider_kwargs_per_provider` | provider 별 생성자 인자. SAM weight 가 `grounded_sam` preset 과 같은지. frozen selection 파일이 있는지 |
| `test_attention_run_aux_and_join` | synthetic audit 에서 `aux` 값이 `pooled_f16[k]` · `cell_grid_policy[k, cam]` 와 같은지. frame 이 없으면 `None` 인지. source run 이나 sha256 이 다르면 ValueError 인지 |
| `test_aux_feeds_both_vla_internal_providers` | 같은 `aux` 를 실제 `LocalizationHeadsProvider` (threshold) 와 `TokenCutProvider(attention_overlap)` 에 넣으면 found 이고, `validate_results` 를 통과하는지. TokenCut 은 가짜 feature extractor 를 쓴다 |
| `test_real_attention_frame_wiring` | 실제 SL3 `libero_spatial_t0_e0` 첫 평가 frame (obs 10) 에서 audit join 과 sha256 이 맞는지. cell grid 가 `pooled[8,2]` block 과 1e-3 이내인지. localization_heads threshold → `evaluate_masks` 가 끝까지 도는지 |

## dry-run (입력 grid 확인, CPU)

| 집합 | 명령 (provider 목록은 아래 "실행 명령" 과 같음, `--dry-run --out-dir …`) | 결과 |
|---|---|---|
| SL3 dev | `--out-dir outputs/paper_baselines_smoke/E_dryrun_sl3dev` | 10 ep, 평가 frame 221. **221 / 221 frame 에 audit call 이 있다.** ATS 기록도 221 / 221. pooled shape (18, 8, 768), 전부 finite. `cell_grid_opencv == cell_grid_policy[:, ::-1]` 모두 참. cell 과 `pooled_f16[8,2]` 의 최대 절대 차 2.98e-8. GT target 이 두 camera 모두 0 px 인 frame 0. episode 당 frame 18–27. 11.8 s |
| SL4b dev | `--out-dir outputs/paper_baselines_smoke/E_dryrun_sl4bdev` | 40 ep, 평가 frame 873. **873 / 873 frame 에 audit call 이 있다.** 나머지 점검 값은 SL3 와 같다 (최대 절대 차 2.98e-8, GT 0 px frame 0). episode 당 frame 15–44. 35.8 s |
| 배선 (non-dry) | SL3, `gt_oracle localization_heads:threshold tokencut:foreground tokencut:attention_overlap` + ATS, `--limit-episodes 1 --limit-frames 2 --device cpu`, `--out-dir outputs/paper_baselines_smoke/E_wiring_sl3dev` | 모든 arm 이 2 frame 을 냈다. `n_missing` 은 0. arm 은 `localization_heads:threshold\|no_text`, `tokencut:*\|no_text` 로 각 한 개. `provenance.attention.selection_in_sample` 은 `{"localization_heads:threshold": {"in_sample": true, …}}`. DINO 공식 weight 를 읽었다 (sha256 `1566d504…`, "All keys matched"). 18 s. **숫자는 배선 확인용이다. 판정에 쓰지 않는다** |

## verifier 가 알아야 할 것

**새 플래그 · 기본값 변경**
- `offline_grounding`: `--attention-run`, `--lh-selection` 를 새로 만들었다. `--lh-selection` 기본값은 `providers/frozen/localization_heads_pi05_libero_spatial_dev_init0-4_official_code.json` 이다.
- localization_heads · tokencut arm 이름은 `|no_text` 다. prompt mode 수와 관계없이 한 개씩이다.
- `--device` 가 없을 때 tokencut 은 cpu 가 아니라 `default_device()` (GPU 가 있으면 cuda) 를 쓴다.
- `setup_venv.sh`: `SAM2_BUILD_CUDA` 기본값이 1 이다.

**join 과 입력**
- audit call 의 `obs_index` 를 평가 frame 과 맞춘다. 두 dev 집합 모두 scene-GT frame 이 전부 audit call 이다. 그래서 `missing_reason` 행은 0 이어야 한다.
- `tokencut:attention_overlap` 의 attention 은 V11AB 와 같은 cell (L8H2, last action token, denoise step 0) 이다. localization_heads 의 입력은 frozen head 3 개 (camera 별) 다.

**in-sample**
- frozen selection 의 `inputs` 는 바로 이 두 audit run 이다 (SL3 10 ep + SL4b 40 ep).
- `provenance.json["attention"]["selection_in_sample"]` 에 `in_sample: true` 가 자동으로 기록된다.

**`grounded_sam2:track` feed 와 attention**
- track 은 frame 0 부터 넣는다. track 은 attention 을 쓰지 않으므로 둘이 섞이지 않는다.
- attention provider 는 평가 frame 에만 돈다.

**`sam_point` 의 SAM**
- `SamV1Backend("sam-vit-huge", device=--device)` 를 하나 더 올린다. grounded_sam 의 SAM 과 공유하지 않는다. GPU memory 를 그만큼 더 쓴다.

**latency 와 첫 call**
- `latency_ms` 는 provider 가 잰 camera 별 값을 그대로 쓴다.
- 첫 call (warm-up) 을 빼는 것은 summary 에서 하지 않는다. verifier 가 `frames.jsonl` 에서 뺀다.
- localization_heads threshold 의 latency 는 기록된 attention 에서 mask 를 만드는 시간뿐이다. pi0.5 forward 는 들어 있지 않다.

**재생산이 필요한 산출물:** 없음.

**옛 기록과 호환이 깨지는가:** 없음.
- 기존 provider 의 kwargs 는 그대로다 (`size` · `device` · `mode`).
- 새 provider 를 쓰지 않으면 출력이 이전과 같다.
- dry-run 행에 열이 더 생겼을 뿐이다.

**working tree 에 다른 변경이 있다.** `external/sources.json` 과 `methods/` 의 변경은 내 것이 아니다 (C · D 또는 lead). 나는 손대지 않았다.

## 실행 명령 (verifier 용, GPU 는 lock 아래에서)

```bash
source scripts/setup/env.sh; source Baseline/paper_baselines/setup/paper_baselines_env.sh
P="gt_oracle grounded_sam grounded_sam2:per_frame grounded_sam2:track yoloe \
   localization_heads:threshold localization_heads:sam_point tokencut:foreground tokencut:attention_overlap"

# 1  SL3 dev (10 ep, 221 frames) + ATS
flock -w 3600 /tmp/ag3s-vla-gpu-validation.lock env PYTHONPATH=$PWD .venv-baselines/bin/python -m Baseline.paper_baselines.eval.offline_grounding \
  --source-run 20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0 --scene-gt outputs/verify/SL4/raw/scene \
  --providers $P --prompt-modes phrase instruction --size default \
  --attention-run 20261012T202459_309e4c6da477_sl3-attention-audit-dev \
  --ats-run 20261012T225820_46a08d6ff644_sl4-perception-dev --save-masks --run-id paper-baselines-2-sl3dev

# 2  SL4b dev (40 ep, 873 frames), ATS 없음
flock -w 3600 /tmp/ag3s-vla-gpu-validation.lock env PYTHONPATH=$PWD .venv-baselines/bin/python -m Baseline.paper_baselines.eval.offline_grounding \
  --source-run 20261013T092537_714c891415d9_sl4b-dev-pi05only-t0-9e1-4 --scene-gt outputs/verify/SL4b/raw/scene \
  --providers $P --prompt-modes phrase instruction --size default \
  --attention-run 20261013T094739_714c891415d9_sl4b-attention-audit-dev-e1-4 --save-masks --run-id paper-baselines-2-sl4bdev
```

`--save-masks` 는 overlay figure 에 쓰려고 넣었다. 필요 없으면 빼도 된다.

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다. -->
- `localization_heads:threshold|no_text` 와 `tokencut:attention_overlap|no_text` 는 모든 평가 frame 에서 `n_missing = 0` 일 것이다 (dry-run 에서 coverage 가 100 % 였다).
- 그 밖의 숫자에 대한 기대는 B 의 impl.md 와 같다.
