# BASELINE-EVAL — baseline 별 고유 지표 · 시각 자료 · 공정 비교 · SL4b/track 측정, 그리고 AG3S_BASELINE_LOG.md

> writer: lead · 읽는 쪽: scribe, implementer F / G, verifier V1 / V2 · 작성 2026-10-23 KST
> 사용자 지시 2026-10-23:
> 1. `AG3S_BASELINE_LOG.md` 를 만들어 기록한다.
> 2. 권장 순서로 진행한다.
> 3. 각 baseline 과 benchmark 가 요구하는 지표를 **전부** 기록하고 시각 자료를 남긴다. 실제 장면, segmentation 결과, 그 module 의 핵심 기능에 관한 그림이 필요하다.
>
> 선행: `bf1403d5` (PAPER-BASELINES), `5752c64c` (PAPER-METHODS, PAPER-BASELINES-2).

## 권장 순서 (lead)

| 순서 | 무엇 | 왜 이 순서인가 | 자원 |
|---|---|---|---|
| 1 | **ATS 와의 공정 비교 + grounding 고유 지표 + methods offline 지표와 figure** (F → V1) | 논문 핵심 주장에 바로 필요하다. 지금의 ATS 행은 대표점 표현 때문에 recall 0.068 로, 비교로 쓸 수 없다. SL3 run 이 저장한 mask 를 다시 쓰므로 CPU 만 든다 | CPU |
| 2 | **track memory 수정 + 평가 CPU 경로 속도 개선 → SL4b 40 ep + track 측정** (G → V2) | 남은 GPU 측정이다. 속도 개선이 lock 점유 시간을 줄인다 | CPU → GPU |
| 3 | **methods closed-loop 연결 → benchmark 고유 지표(TSR · CAR · ETS 등)** | `benchmark/runners` 수정이 필요하다. 같은 checkout 의 다른 session 과 조율한 뒤 별도 task 로 한다 | GPU, 조율 |

## 지표 registry — `Baseline/paper_baselines/docs/metrics.md` (F)

**F 는 각 원 논문과 benchmark 문서(primary source)에서 지표 정의를 확인해 아래 표를 채운다.** 칸마다 다음을 적는다.
- 정의 (식)
- 출처: 논문 절 · 표 번호, 또는 공식 code 의 줄
- 우리가 지금 잴 수 있는지: `offline-now` / `closed-loop` / `not-applicable` 과 그 이유

lead 가 아는 범위의 초안이다. 확인되지 않은 항목은 F 가 고친다.

| 대상 | 원 논문 · benchmark 의 지표 (확인 대상) | 우리 offline 대응 |
|---|---|---|
| AG3S (ours, main.pdf) | observed-surface soft IoU, precision, neighbour leak, paired gain, success / safe success / hold, contact class | harness |
| Grounded-SAM | zero-shot mask AP · mIoU (SegInW 등) | mask IoU · P · R, 2D / 3D |
| Grounded-SAM-2 (track) | video seg J (region IoU) · F (contour) · J&F | **frame 별 J, F, J&F, temporal IoU 안정성** |
| YOLOE | AP, AP_r (LVIS), mask AP, FPS | conf 별 P/R 곡선, detection recall, FPS |
| Localization heads | REC Acc@0.5 (box IoU ≥ 0.5), RES mIoU / oIoU | **box Acc@0.5**, mask mIoU / oIoU, pointing game |
| TokenCut | CorLoc (box IoU ≥ 0.5), saliency maxFβ · IoU · Acc | **CorLoc**, maxFβ, IoU |
| GT oracle | — (상한선) | lift 검증 |
| VLSA / AEGIS, SafeLIBERO | CAR, TSR, ETS (SafeLIBERO 정의) | offline: intervention rate, ‖Δa‖, barrier h_min, QP 실패율, 예측 clearance(GT 기하 기준), solve time |
| KNOWS | (원 논문 지표 확인) | 같은 offline 세트 |
| Multi-link [1] | (SafeLIBERO TSR / CAR + link 별 충돌 — 확인) | + link 별 h_min, EE-only 대비 추가 개입 |
| CFM [5] | (SafeLIBERO 지표 + 제약 만족률 — 확인) | + 제약 위반 수 |
| NUDGE [9] | (success, collision — 확인) | + path SDF min |
| OmniGuide [19] | success rate, safety rate (확인) | + guided/unguided path 차, energy |
| Lan-o3dp [15] | success rate, collision (RLBench — 확인) | + cost 감소량 |
| LIBERO | success rate | — (closed-loop) |
| SafeLIBERO | TSR, CAR, ETS (level I / II) | — (closed-loop) |
| LIBERO-Safety | (공식 지표 확인, plan 의 SR / CR / SSR) | — (closed-loop) |

## Implementer F (순서 1, 새 파일만)

1. `docs/metrics.md`: 위 registry 를 완성한다. 출처는 primary source 만 쓴다.
2. **grounding 고유 지표** — `eval/native_metrics.py`. SL3 run 의 `--save-masks` 산출물을 읽어 사후 계산한다. harness 는 수정하지 않는다.
   - box Acc@0.5, CorLoc
   - pointing game (예측 peak 이 GT 안에 있는가)
   - mIoU / oIoU
   - maxFβ
   - J / F / J&F (track 결과가 생기면 같은 code 로 계산)
   - 각 arm 에 대해 계산한다.
3. **ATS 와의 공정 비교** — `eval/cloud_labeling.py`.
   - V11AB perception run 이 기록한 **같은 fused / filtered cloud 점 집합** 위에서 점마다 label 을 매긴다:
     - GT: scene GT 의 가장 가까운 물체 (거리 임계 포함)
     - ATS: target flag
     - 각 segmenter: 그 점이 어느 camera 에서든 mask 안에 투영되는가
   - 그 위에서 point-level IoU · P · R · neighbour 비율과 paired bootstrap 을 낸다. 두 쪽이 같은 표현 · 같은 해상도가 되게 하려는 것이다.
   - perception run 이 cloud 를 기록하지 않았다면(`--no-clouds`) 그 사실을 보고하고, 대안(ATS target 점을 같은 voxel 로 dilate 하는 등)을 제안만 한다.
4. **methods offline 평가** — `methods/evaluate_offline.py`.
   - SL3 dev 10 ep 전체 call 에서 7 method(KNOWS, AEGIS 3 변형, multi-link, OmniGuide, CFM, Lan-o3dp adapted, NUDGE)를 돌린다.
   - 지표: intervention rate, ‖Δa‖, barrier h_min, 제약 위반, QP 실패, solve time, scene GT 기준 예측 EE / link clearance.
   - guidance 는 D 의 recorded-chunk 대체 velocity 를 쓰고, 그 한계를 산출물에 기록한다.
   - **figure 용 dump 를 남긴다:** 아래 "핵심 기능 figure" 에 필요한 ellipsoid · h trace · energy slice · path before/after · attention map · eigenvector 등을 npz 로 둔다.
   - CPU 만 쓴다. guidance 는 `.venv-openpi-live` 의 `JAX_PLATFORMS=cpu`, filter 는 `.venv-baselines` 다.
5. tests 를 두고 `BASELINE-EVAL-F.impl.md` 를 쓴다.

**수정 금지:** `eval/offline_grounding.py`, `providers/grounded_sam2.py`, `lift.py` (G 소유), `AG3S/**`, `benchmark/**`, `third_party/**`.

## Implementer G (순서 2, CPU)

1. **`grounded_sam2:track` memory 상한.** frame 마다 약 0.9 GB 가 쌓인다. streaming state 의 memory bank 를 최근 N frame(SAM2 기본 `num_maskmem` = 7)과 conditioning frame 만 남기도록 정리한다. 처리된 이전 frame 의 feature 와 output 을 버린다.
   - causal 성질은 유지한다. 기존 causality test 를 통과해야 한다.
   - CPU 에서 긴 sequence(합성 또는 recorded 1 ep 전체)로 memory 가 평평해지는 것을 보인다. GPU 확인은 V2 가 한다.
2. **평가 CPU 경로 속도 개선.** episode 당 약 700 s 중 provider 시간은 약 100 s 다.
   - `offline_grounding.py` / `lift.py` / `eval/metrics.py` 를 profile 한다(cProfile).
   - 병목을 vectorize 하거나 cache 한다. 예: frame 마다 반복되는 GT voxel 계산, soft voxel 집계.
   - **숫자는 bit 단위로 같아야 한다.** SL3 run 의 `frames.jsonl` 중 1 ep 를 새 code 로 다시 계산해 같은지 test 한다.
3. provider 예외 하나가 run 전체를 멈추는 문제: arm 단위로 격리하고 실패 행을 기록한 뒤 계속 진행한다.
4. `BASELINE-EVAL-G.impl.md` 를 쓴다.

**수정 금지:** F 의 새 파일, `methods/**`, `AG3S/**`, `benchmark/**`.

## 측정 · figure (→ verifier)

**V1 (F 이후, CPU):** F 의 산출물을 실행해 `BASELINE-EVAL-V1.verify.json` 과 figure 를 만든다.
**V2 (G 이후, GPU, lock):** SL3 track + SL4b 40 ep 전체 arm(track 포함)을 측정하고 `BASELINE-EVAL-V2.verify.json` 을 만든다.
- GPU 규칙: `nvidia-smi` 가 비어 있음을 확인하고, `flock -w <초> LOCK CMD` 로 돌린다. 시작 전과 후에 다른 session(`ag3s-vla-benchmark-67`)에 알린다.

### figure 요구 (규칙 A) — `AG3S/ag3s/docs/figures/baselines/<module>/` + `.json` sidecar

module 마다 아래 세 종류를 모두 만든다.

| 종류 | 무엇 |
|---|---|
| 실제 장면 | 실제 frame 위에서 보이는 결과 |
| 핵심 기능 | 그 module 이 어떻게 결정하는지 |
| graph 또는 표 | 고유 지표 |

| module | 실제 장면 · segmentation | 핵심 기능 figure |
|---|---|---|
| grounded_sam / grounded_sam2 | 두 camera 에 DINO box + score + SAM mask, GT 윤곽, 3D lifted 점 vs GT voxel | phrase vs instruction 의 box 선택. track: frame strip 과 J/F 시계열 |
| yoloe | detection + conf | conf 임계 별 P/R 곡선 |
| localization_heads | mask overlay | 선택된 head 3개의 attention map, 선택 기준(attention 비율 · spatial entropy · 빈도) 막대 |
| tokencut | mask overlay | affinity graph 의 2nd eigenvector map, bipartition, seed |
| gt_oracle | GT mask | lift 검증 (3D 점 vs GT voxel) |
| ats_v11ab | seed · cluster · target 점 | cloud-labeling 공정 비교: 같은 점 집합 위의 GT / ATS / segmenter label |
| knows | 장면 위 fitted ellipsoid | EE path nominal vs filtered, h(t) |
| aegis (3 변형) | detection → ellipsoid | h(t), nominal vs filtered action, 변형 간 차이 |
| multilink_cbf | link 별 ellipsoid | link 별 h(t), EE-only 가 놓치는 forearm 사례 |
| omniguide / cfm / lan_o3dp / nudge | 장면 위 guided vs unguided EE path | energy · cost · SDF 단면(배치도 먼저), denoising step 별 수정량 |
| benchmarks (LIBERO, SafeLIBERO I/II, LIBERO-Safety) | 각 suite · level 의 대표 장면(장애물 표시). 기존 run 기록이나 env render 를 쓴다 | 지표 정의 표(registry 에서) |

- 지표 표는 `docs/metrics.md` 의 모든 행에 대해 "측정값 또는 `closed-loop 에서 측정`" 이 채워진 표 figure 하나로 만든다.
- 해석은 쓰지 않는다. 숫자는 verify.json 의 `numbers` 에만 둔다.
- 같은 이름의 figure 가 이미 있으면 새 이름을 쓴다.

## Scribe — `AG3S/ag3s/docs/AG3S_BASELINE_LOG.md` (새 파일)

- 형식은 `AG3S_BENCHMARK_LOG.md` 를 따른다: 머리말(용도 · 수치 규칙), 진행 현황 표, 용어 표, 단계별 절.
  - 각 단계 절의 내용: 날짜 · 시간 KST, 무엇을 · 왜, 표 · figure 링크, 판정 상태.
  - 숫자는 `*.verify.json` 의 `numbers` 에서만 가져온다.
- 단계:
  - BL-0 **PAPER-BASELINES**: 폴더 · interface · providers. 근거 문서는 impl.md 이고, 숫자 없이 구성과 test 통과 사실만 쓴다.
  - BL-1 **PAPER-BASELINES-2**: SL3 dev GPU 측정. 숫자는 `PAPER-BASELINES-2.verify.json` 에서 가져오고, figure 는 `figures/paper-baselines-2/` 를 링크한다.
  - BL-2 **PAPER-METHODS**: methods 구성. impl.md 근거로 구성 · 원 방법과의 차이 · test 통과 사실만 쓴다. 수치는 V1 이후에 채운다.
  - 이후 단계(BASELINE-EVAL V1 · V2)는 verify 가 생기면 이어 쓴다.
- 반드시 남길 사실 (verify.json 의 issues · not_measured 에 있다):
  - ATS 행은 대표점 표현이라 비교로 쓸 수 없다.
  - track OOM.
  - SL4b 는 deferred.
  - SAM2 sm_90 원인과 재빌드.
  - localization_heads 는 in-sample.
  - KNOWS rename 이 commit `88f0c735` 에 섞인 경위.
- 이번에는 판정을 쓰지 않는다. "사용자 판정 대기" 로 둔다.
