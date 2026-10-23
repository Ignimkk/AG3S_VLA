# PAPER-BASELINES-2 — grounding baseline GPU 측정 (Grounded-SAM 계열 · YOLOE · VLA-internal · oracle vs ATS)

> writer: lead · 읽는 쪽: implementer E (harness 보강, 짧음) → verifier · 작성 2026-10-23 KST
> 사용자 지시 2026-10-23: "GPU 측정도 수행". 선행: PAPER-BASELINES (`bf1403d5`).
> SAM2 CUDA extension 은 lead 가 build 했다 (2026-10-23, `SAM2_BUILD_CUDA=1`, CUDA 12.6, sm_90).

## Implementer E (먼저 · 짧게)

1. `eval/offline_grounding.py` 에 `--attention-run <audit run>` 을 추가한다.
   - `localization_heads` provider 가 그 call 의 per-(layer, head) grid 를 `aux` 로 받게 한다. 형식은 `providers/localization_heads.py` 가 기대하는 것을 따른다.
   - `--providers` 에 `localization_heads:threshold`, `localization_heads:sam_point`, `tokencut:foreground`, `tokencut:attention_overlap` 이 들어가게 한다.
   - SL3 dev 는 audit `20261012T...`(SL3 audit, B 의 impl.md 참조)와 짝을 맞춘다. SL4b 도 같은 방식이다.
2. `setup/setup_venv.sh`: `SAM2_BUILD_CUDA=${SAM2_BUILD_CUDA:-1}` 로 바꾸고 nvcc 경로 주석을 단다. lead 의 build 명령과 같게 한다:

   ```
   CUDA_HOME=/usr/local/cuda-12.6 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=2
   uv pip install --no-build-isolation --no-deps --reinstall-package sam2 -e external/Grounded-SAM-2
   ```

3. `providers/localization_heads.py` 41–43행 docstring 의 SAM wrapper 위치를 `sam_backend` 로 고친다 (A 의 지적).
4. 전체 test(`Baseline/paper_baselines/tests`) 통과를 확인한다. `--dry-run` 으로 두 run 의 입력 grid 를 확인한다. CPU 만 쓴다.
5. `PAPER-BASELINES-2-E.impl.md` 를 쓴다.

## 측정에 요청하는 것 (→ verifier, E 이후)

- **GPU 규칙:**
  - `nvidia-smi --query-compute-apps` 가 비어 있을 때 `flock -w <초> /tmp/ag3s-vla-gpu-validation.lock CMD` 로 시작한다. fd 형식은 쓰지 않는다.
  - 시작 전과 끝난 뒤 같은 pod 의 다른 session 에 알린다(ListAgents → SendMessage). 특히 같은 checkout 의 `ag3s-vla-benchmark-67` 이다.
  - nohup 없이 자기 shell 에서 실행한다.
- **weights:** `--size default`(큰 weights)로 GPU 에서 돌린다.

| # | 집합 | provider × prompt | 숫자 | 통과 기준 |
|---|---|---|---|---|
| 1 | SL3 dev 10 ep (+ `--ats-run`, scene GT `outputs/verify/SL4/raw/scene`) | gt_oracle, grounded_sam, grounded_sam2:per_frame, grounded_sam2:track, yoloe × {phrase, instruction}; localization_heads:{threshold, sam_point}; tokencut:{foreground, attention_overlap}; ats_v11ab | 2D IoU · precision · recall; 3D soft IoU · precision · neighbour leak; found rate | 비교용. gt_oracle 3D soft IoU 가 다른 모든 행보다 높아야 lift 가 성하다 |
| 2 | SL4b dev 40 ep (scene GT `outputs/verify/SL4b/raw/scene`, ATS 없음) | #1 에서 ats_v11ab 를 뺀 같은 목록 | 같은 숫자 | 같음 |
| 3 | #1 의 ATS 대비 paired gain | provider 별 | episode-paired bootstrap 95% CI | 보고만 |
| 4 | latency | provider 별 GPU ms per call (median / p95, 첫 call 제외) | 보고만 | 보고만 |
| 5 | "two black bowls" 류 referring-expression task | phrase vs instruction prompt | wrong-object rate (예측 voxel 이 주로 비-target 물체 위인 frame 비율) | 보고만 |

- **in-sample 주의:** localization_heads 의 head 는 SL3 + SL4b dev(init 0–4)에서 골랐다. 이 측정은 head 선택에 쓴 바로 그 집합이므로 verify.json 에 `in_sample: true` 로 표시한다.
- **figure:** `AG3S/ag3s/docs/figures/` 에 `.json` sidecar 와 함께 만든다(skill `make-figure`).
  - 실제 장면: 같은 frame 에서 provider 별 mask overlay 를 비교한다. referring-expression 실패 예시를 1장 포함한다.
  - graph: provider 별 3D soft IoU 와 neighbour leak (episode 점 + 평균).
  - 표: 위 숫자 전체.
- 출력은 `outputs/verify/PAPER-BASELINES-2/` 와 `AG3S/ag3s/docs/handoff/PAPER-BASELINES-2.verify.json` 이다. 해석은 쓰지 않는다.

## 이 STEP 이 답하는 질문

같은 depth · 같은 lift · 같은 지표에서 external segmenter(Grounded-SAM 계열 · YOLOE)와 VLA-internal baseline 이 LIBERO dev frame 의 target 표면을 얼마나 맞히고, V11AB ATS(Coarse)와 비교해 어떤가?
