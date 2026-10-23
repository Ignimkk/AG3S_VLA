# 지표 registry — 각 baseline 논문과 benchmark 가 쓰는 지표, 그리고 우리가 지금 잴 수 있는가

> writer: Implementer F (BASELINE-EVAL) · 확인 2026-10-23 KST · 읽는 쪽: verifier V1, scribe, lead
> **primary source 만 썼다**: 논문 본문 (arXiv HTML / PDF 의 절 · 표 · 식) 과 공식 code 의 파일:줄 (pinned clone).
> arXiv HTML 은 WebFetch 요약을 거쳐 읽었다. 숫자를 인용하기 전에는 PDF 표와 한 번 더 대조해야 한다 (§5).
> 인용 번호는 `main.pdf` (CVPR 2027 draft) References 의 번호다.

## 읽는 법

| status | 뜻 |
|---|---|
| `offline-now` | 지금 있는 기록 (SL3 dev 의 저장 mask · V11AB perception cloud · 기록된 action chunk) 만으로 CPU 에서 잰다. 이 STEP 의 tool 이 계산한다 |
| `closed-loop` | rollout 이 있어야 한다 (성공 · 충돌 · episode 길이). 순서 3 (methods closed-loop 연결) 에서 잰다 |
| `not-applicable` | 우리 setting 에서 정의가 성립하지 않는다 (dataset · 학습 방식 · task 가 다름). 이유를 적는다 |

tool 약자: **NM** = `eval/native_metrics.py`, **CL** = `eval/cloud_labeling.py`, **EO** = `methods/evaluate_offline.py`,
**H** = 기존 harness `eval/offline_grounding.py` + `eval/metrics.py` (PAPER-BASELINES-2 에서 이미 측정).

## 1. 한눈에

| 대상 | 원 논문 · benchmark 의 지표 (출처) | status | 우리 대응 (tool → `summary.json` key) |
|---|---|---|---|
| AG3S (ours) | observed-surface soft IoU · Prec. · neighbour leak · paired gain (main.pdf Sec. 4.1 l.424-429, Tab. 2) | `offline-now` (ATS V11AB 행) | H: `3d_soft_*`, `neighbour_leak_*`, `paired_vs_ats`; CL: point IoU · P · R · neighbour |
| AG3S (ours) | success · safe success · hold · contact class (Sec. 4.1 l.413-424, Tab. 1); latency (Tab. 3); SafeLIBERO TSR · CAR (Tab. 4) | `closed-loop` | — |
| Grounded-SAM | SegInW zero-shot mean AP, 25 dataset (arXiv:2401.14159 Abstract, Sec. 4, Tab. 1) | AP: `not-applicable` | H: 2D IoU · P · R, 3D soft IoU; NM: `mIoU` `oIoU` `acc05_det` `acc05_mask` `davis` `conf_sweep` |
| Grounded-SAM-2 | SAM 2 의 video 지표 J&F (arXiv:2408.00714v2 Sec. 6), 계산 code = `sav_benchmark.py:215-303` | `offline-now` (per_frame). track 은 같은 code, mask 는 V2 이후 | NM: `davis.J_mean` `F_mean` `JF_mean` `J_recall` `J_decay` `temporal_iou_*` |
| YOLOE [25] | LVIS minival Fixed AP · AP_r/c/f, LVIS val AP^m, COCO AP^b/AP^m, FPS (T4 TensorRT / iPhone CoreML) (arXiv:2503.07465v2 Tab. 1-4) | AP: `not-applicable`; 속도: `offline-now` | NM: `conf_sweep` (conf 별 P/R), `acc05_det`; H: `latency_*` (H200, PAPER-BASELINES-2) |
| Localization heads [14] | REC Acc@0.5, RES · ReasonSeg cIoU (arXiv:2503.06287 Sec. 6.1, Tab. 1-2) | `offline-now` (in-sample) | NM: `acc05_mask`, `oIoU` (= cIoU), `mIoU`; pointing game (추가, 아래) |
| (추가) pointing game | Zhang et al. ECCV 2016, arXiv:1608.00507 Sec. 4.1 (hit = 최대점이 GT 안, tolerance 15 px) | `offline-now` | NM: `pointing_tol0`, `pointing_tol15` (LH peak · TokenCut seed · V11AB L8H2 peak), `pointing_box_centre_*` (변형) |
| TokenCut [26] | object discovery CorLoc (arXiv:2202.11539v2 Sec. 4.1, Tab. 1; code `main_tokencut.py:295-298`); saliency maxFβ · IoU · Acc (Sec. 4.3, Tab. 3; code `metric.py:5-81`) | `offline-now` | NM: `corloc_any_object`, `corloc_target`, `fbeta_binary` (= maxFβ of a binary mask), `mIoU`, `pixel_acc`; 연속 map `maxFbeta` |
| GT oracle | — (상한) | `offline-now` | H: `lift_within_10mm`; CL: gt_oracle 행 (projection 상한) |
| ATS V11AB | (AG3S 이전판) | `offline-now` | CL: `ats_v11ab|noun_attention` (cloud_target), `ats_v11ab_seeds|…` |
| VLSA / AEGIS [10] | SafeLIBERO CAR · TSR · ETS (arXiv:2512.11891v2 Sec. V-A, Tab. I); 충돌 판정 code `main_aegis.py:370` | `closed-loop` | EO offline 대리: `intervention_rate_*`, `da_xyz_l2_*`, `h_min`, `h_neg_steps`, `qp_fail_steps`, `ms_per_call`, `gt_*` clearance |
| KNOWS [17] | SR · CR · SSR (primary) (arXiv:2606.09749v1 Sec. 4.1, Tab. 1); per-step latency (Tab. 2) | `closed-loop` | EO: 같은 offline 세트 |
| Multi-link [1] | collision (hazard 변위 > 11 mm) · contact · S · safe-success · p50/p99 latency · energy (arXiv:2609.40007v1 Sec. IV, Tab. I-II). link 별 충돌 수는 보고하지 않음 | `closed-loop` | EO: + `h_by_body` (dump), `gt_*.arm_obstacle_*`, `obstacle_min_by_link`, `multilink_intervenes_aegis_not` |
| CFM [5] | CAR · TSR · ETS (arXiv:2607.01378v1 Sec. 5.1, Tab. 1-2); 안전 조건 B_ij ≥ 0 (Eq. 3) | `closed-loop`; 자기 제약: `offline-now` | EO: `own_*.cfm_min_g`, `cfm_violated_rows_before/after` |
| NUDGE [9] | LIBERO SR · CR (= contact 가 있는 control step 비율) (arXiv:2609.35231v2 Sec. IV-C, Tab. III); sphere task Coll. Rate · Coll. Ep. (Tab. II); latency (Tab. IV) | `closed-loop` | EO: `own_*.nudge_c_sdf`, `scene_hand_min_sdf_*`, `gt_*` |
| OmniGuide [19] | Success Rate · Safety Rate (arXiv:2603.10052v1 Sec. IV-A, Fig. 5); latency (Fig. 10) | `closed-loop` | EO: `own_*.omniguide_cost`, guided vs unguided path, `frozen_calls` |
| Lan-o3dp [15] | RLBench success rate (상위 5 eval 평균) (arXiv:2407.00451v3 Sec. IV-A); 실물 obstacle avoidance 성공률 (Sec. IV-B3, Tab. II) | `not-applicable` (원 policy · RLBench 없음); adapted guidance 는 `closed-loop` | EO: `own_*.lan_rows_inside_circle` |
| LIBERO | success rate (env goal predicate); lifelong 지표 FWT · NBT · AUC (arXiv:2306.03310v2 Sec. 5.1 Eq. 3) | SR: `closed-loop`; FWT/NBT/AUC: `not-applicable` | — |
| SafeLIBERO | CAR · TSR · ETS, Level I / II, scenario 당 50 episode (VLSA Sec. V-A) | `closed-loop` | — |
| LIBERO-Safety | SR · CR · Execution Time · LDLJ (Eq. 2) (physical track); Refusal Rate (SSR suite) (arXiv:2606.23686v2 Sec. 4.1, Tab. 3-6) | SR · CR · time · LDLJ: `closed-loop`; RR: `not-applicable` | — |

lead 초안과 달라진 점:
- Localization heads 의 RES 지표는 **cIoU** 다 (논문은 mIoU / oIoU 라는 이름을 쓰지 않는다). cIoU = oIoU (누적 교집합 / 누적 합집합) 라서 NM 의 `oIoU` 가 그것이다. `mIoU` 는 참고로 같이 낸다.
- **pointing game 은 [14] 에 없다.** 원 정의는 Zhang et al. 2016 이다. 우리가 추가한 지표로 표시한다.
- Grounded-SAM 의 SegInW AP 가 box AP 인지 mask AP 인지 논문은 밝히지 않는다.
- Multi-link [1] 은 link 별 충돌 수를 보고하지 않는다. coverage 변형 (EE-only vs multi-link) 비교만 있다.
- CFM [5] 은 제약 위반률을 따로 정의하지 않는다. 자기 제약 (Eq. 3, B ≥ 0) 은 우리가 offline 으로 센다.
- KNOWS [17] 는 CAR / TSR 대신 SR / CR / SSR 을 쓴다 (CR = 비대상 장애물과 접촉한 episode 비율).

## 2. 항목별

### AG3S (ours) — main.pdf

- **정의** (Sec. 4.1 "Protocol and definitions", 줄 413-429):
  - grounding: 5 mm voxel grid 위 soft IoU, frame 지표를 episode 마다 평균한 뒤 episode 에 걸쳐 평균, 2,000 회 episode-paired bootstrap.
  - neighbour leak = 이웃 물체 위 false-positive voxel mass 의 pooled 합 / radius growth 의 같은 합 (ratio of sums).
  - task success = 물체가 basket 안에 놓이고 놓아짐. safe success = 추가로 release 전 금지 접촉 없음. hold = prefix check 가 실패해 팔이 기다린 chunk 하나.
  - 접촉은 기록 상태를 simulator 에서 replay 해서 잰다. robot–obstacle · carried-object–obstacle · hand–neighbour · non-contact-link–target · (허용) hand–target 으로 나눈다.
- **표**: Tab. 1 (closed-loop success / safe / hold), Tab. 2 (held-out grounding IoU / Prec. / Leak / Gain), Tab. 3 (latency), Tab. 4 (SafeLIBERO TSR / CAR).
- **status**: grounding 지표는 `offline-now` (H 가 같은 정의로 잰다. benchmark branch 에서는 V11AB ATS 행). closed-loop 지표는 `closed-loop`.

### Grounded-SAM

- **정의**: SegInW (Segmentation in the Wild, 25 dataset) zero-shot 의 mean AP. Grounding DINO-Base + SAM-Huge 가 48.7 (arXiv:2401.14159 Abstract; Sec. 4; Tab. 1).
- **status**: AP 는 `not-applicable`. SegInW dataset 이 없고, 우리 task 는 instruction 이 지정한 target 하나 (referring) 라서 category AP 가 정의되지 않는다.
- **우리 대응** (`offline-now`):
  - H: 2D IoU / P / R, 3D soft IoU.
  - NM: mIoU · oIoU, Acc@0.5 (detector box 와 mask box), J / F / J&F (per-frame 이지만 같은 sequence 로 계산), `conf_sweep` (top-1 box 의 conf 별 P / R, `any_candidate_recall`).

### Grounded-SAM-2 (per_frame / track)

- **정의**: SAM 2 는 "standard J&F metric" (Pont-Tuset et al. 2017) 을 video 에, mIoU 를 image 에 쓴다 (arXiv:2408.00714v2 Sec. 6). 계산 code 는 pinned clone `Grounded-SAM-2@b7a9c29` 의 `sav_dataset/utils/sav_benchmark.py`:
  - `get_iou` (215-223): J = |P∩G| / |P∪G|, 둘 다 비면 1.
  - `Evaluator.feed_frame` (237-303): F = boundary precision / recall 의 F-measure. boundary 는 `_seg2bmap` (156-212). 허용 반경 `bound_pix = ceil(0.008 · ‖(H, W)‖)` (256 px 영상에서 3 px), `disk` 로 dilate. 경계가 한쪽만 비면 P 또는 R = 0.
  - sequence 통계: davis2017-evaluation `utils.db_statistics` — Mean, Recall = mean(v > 0.5), Decay = 첫 1/4 구간 평균 − 마지막 1/4 구간 평균.
  - SA-V 평가 (`VideoEvaluator`) 는 첫 / 마지막 frame 을 뺀다 (`skip_first_and_last`). **DAVIS 는 빼지 않는다. 우리는 DAVIS 쪽을 따른다** (모든 평가 frame).
- **status**: `offline-now`.
  - per_frame 은 저장 mask 로 지금 잰다.
  - track 은 같은 code 로 계산되지만 mask 가 없다 (OOM, PAPER-BASELINES-2). V2 측정 뒤에 같은 명령으로 잰다.
- **우리 대응**: NM `davis` 블록. sequence = (episode, camera). 5 step 간격의 정책 call frame 이 sequence 의 frame 이다 (원 benchmark 의 연속 frame 이 아님). `temporal_iou_pred / gt / abs_gap` 는 연속 평가 frame 사이 mask IoU 로, 논문 지표가 아니다.

### YOLOE [25]

- **정의** (arXiv:2503.07465v2):
  - Tab. 1: LVIS minival zero-shot Fixed AP, AP_r / AP_c / AP_f, FPS (T4 TensorRT, iPhone 12 CoreML).
  - Tab. 2: LVIS val mask AP^m.
  - Tab. 3: prompt-free Fixed AP, FPS on T4 PyTorch.
  - Tab. 4: COCO transfer AP^b · AP^m.
- **status**:
  - AP 계열은 `not-applicable` (LVIS / COCO annotation 없음, referring target 하나).
  - 속도는 `offline-now` (H 가 provider latency 를 이미 기록했다. H200, PAPER-BASELINES-2. T4 / TensorRT 조건은 아니다).
- **우리 대응**: NM `conf_sweep`. 저장된 후보는 provider 자신의 conf threshold 이상뿐이라, 곡선은 그 threshold 에서 시작한다. conf 0 부터의 곡선은 YOLOE 를 conf 0 으로 다시 돌려야 한다 (GPU).

### Localization heads [14]

- **정의** (arXiv:2503.06287 Sec. 6.1 "Benchmarks"):
  - REC: "Acc@0.5 metric, which is the standard detection metric for REC" (예측 box 의 IoU ≥ 0.5 이면 정답).
  - RES · ReasonSeg: "cIoU" (= cumulative IoU = Σ교집합 / Σ합집합).
  - 결과: Tab. 1 (REC), Tab. 2 (RES).
- **status**: `offline-now`. 단 **head 선택이 같은 dev 기록에서 이뤄져 in-sample** 이다 (PAPER-BASELINES-2 verify.json `in_sample`).
- **우리 대응** (NM):
  - `acc05_mask`: 저장 mask 의 tight box 로 계산. provider 의 box 도 mask 의 box 다.
  - `oIoU` (= cIoU), `mIoU` (참고).
  - `pointing_tol*`: 결합 head map 의 최대점 `peak_xy`.
  - 연속 map `lh_combined_map`: frozen head 3 개의 smoothed sum 을 attention audit 기록에서 다시 계산한다. `maxFbeta` · pointing 을 낸다.

### (추가) pointing game — Zhang et al.

- **정의** (arXiv:1608.00507 Sec. 4.1, p. 10-11):
  - attention map 의 최대점이 cue 된 category 의 annotated instance 위에 있으면 hit, 아니면 miss. Acc = #Hits / (#Hits + #Misses).
  - "we expand the groundtruth region by a tolerance margin of 15 pixels" (입력은 짧은 변 224 로 resize).
- **status**: `offline-now`. 원 논문의 category 평균 대신 sample (frame × camera) 에 대한 pooled 비율을 낸다 (category 가 target 하나).
- **우리 대응** (NM `pointing_tol0`, `pointing_tol15`):
  - 점이 있는 arm 만 낸다: localization heads (peak), TokenCut (seed patch 중심), `v11ab_attention_L8H2` (V11AB attention cell 최대점), `lh_combined_map`.
  - detector arm 의 box 중심은 `pointing_box_centre_*` 로 따로 둔다. **원 정의의 변형이다** (box 에는 saliency 최대점이 없다).

### TokenCut [26]

- **CorLoc**:
  - 정의 (arXiv:2202.11539v2 Sec. 4.1 "Evaluation metric"): 예측 box 가 GT box 중 하나와 IoU > 0.5 이면 정답, 정답 image 비율.
  - code 는 `>= 0.5` 다 (`main_tokencut.py:297-298`, `torch.any(ious >= 0.5)`). 우리는 code 를 따른다.
  - box IoU 는 `datasets.py:289` `bbox_iou` (pixel-edge box, 면적 w·h).
- **saliency** (Sec. 4.3 "Evaluation Metrics", Tab. 3; code `unsupervised_saliency_detection/metric.py`):
  - Fβ = (1+β²)·P·R / (β²·P + R). code `F_score(..., betta_sq=0.3)` (`metric.py:32`) 이므로 **β² = 0.3**. 논문 HTML 은 "β=0.3" 이라고 적는다. code 와 saliency 관례 (β² = 0.3) 를 따른다.
  - maxFβ = 255 개 균등 threshold (`np.arange(0, 1, 1/255)`, `metric.py:63`) 에서 sample 평균 Fβ 의 최댓값 (`F_max`, `metric.py:38-40`). 빈 예측은 P = 1, 빈 GT 는 R = 1 (`metric.py:17-29`).
  - IoU · Acc 는 0.5 로 binarize 한 mask 의 IoU 와 pixel 정확도의 sample 평균 (`metric.py:5-14`).
- **status**: `offline-now`.
- **우리 대응** (NM):
  - `corloc_any_object`: 원 정의처럼 class-agnostic. GT box = 장면의 모든 BDDL 물체 instance (target · destination · other object).
  - `corloc_target`: target box 만 (= `acc05_mask`).
  - `fbeta_binary`: binary mask 는 모든 threshold 에서 같은 mask 라 maxFβ = 평균 Fβ.
  - `mIoU`, `pixel_acc`.
  - TokenCut 의 eigenvector 는 figure dump 에서만 다시 계산한다 (`--tokencut-dino`). 저장 mask 와 같은지를 `tokencut_dino_parity` 에 남긴다.

### GT oracle

- 원 논문이 없다 (상한).
- H 의 `lift_within_10mm` 가 lift 검증이다.
- CL 의 gt_oracle 행은 "mask 를 cloud 에 투영했을 때의 상한" 을 보인다. projection 반올림과 경계 pixel 때문에 1 이 아니다 (값은 V1 의 verify.json).

### ATS V11AB — 공정 비교 (CL)

- **문제**: H 는 ATS 를 대표 target 점 (`target_points`, call 당 ~150 점) 으로 voxel grid 에 올린다. 그래서 recall 0.068 은 표현 차이다.
- **CL 의 방법**: V11AB 가 그 call 에 만든 fused / filtered cloud (`cloud_points`, perception run 이 기록함) 의 같은 점마다 label 을 매긴다.
  - GT: scene GT instance 의 visual surface 까지 정확한 거리 ≤ 10 mm 인 최근접 instance, 아니면 robot sphere 표면 10 mm 이내면 robot, 아니면 none (SL4 · SL4b `ats_calibration.label_points` 와 같은 규칙).
  - ATS: `cloud_target`.
  - segmenter: 그 점을 관측한 camera (`cloud_camera_mask` bit) 에서 mask 안에 투영되면 positive.
- **지표**: point IoU · P · R · neighbour 비율 (+ support / robot / none 비율). call → episode → 평균과 pooled. 각 arm − ATS 의 paired bootstrap (episode, 2,000, seed 20261015).
- **기록 여부**: SL4 perception run `20261012T225820_46a08d6ff644_sl4-perception-dev` 는 cloud 를 기록했다 (`cloud_points`). `--no-clouds` run 이면 CL 이 멈추고 대안을 출력한다 (`NO_CLOUD_PROPOSAL`).

### VLSA / AEGIS [10] — SafeLIBERO 를 만든 논문

- **정의** (arXiv:2512.11891v2 Sec. V-A "Evaluation Metrics"):
  - CAR = "the percentage of strictly collision-free episodes".
  - TSR = "the percentage of tasks successfully completed within the time limit" (충돌해도 episode 는 끝나지 않는다).
  - ETS = "the average episode length (including timeouts)".
- **공식 code** (`benchmark/SafeLIBERO/main/main_aegis.py` @ `2457feed`):
  - 충돌 = 장애물 위치가 시작 위치에서 Σ|Δxyz| > 1 mm 움직임 (`:370`).
  - safe success = `done and not collide_flag` (`:420`).
  - episode 길이 = `time_steps.append(t)` (`:404`).
- **benchmark**: 16 task (4 suite × 4), Level I (target 옆 장애물) / II (경로 위), scenario 당 50 episode, 총 1,600.
- **status**: `closed-loop`.
- **offline 대리** (EO, `offline-now`): 개입률, ‖Δa‖, 자기 barrier h_min · h<0 step, QP 실패, 계산 시간, scene-GT clearance (§3).

### KNOWS [17]

- **정의** (arXiv:2606.09749v1 Sec. 4.1 "Metrics"):
  - SR = task 완료.
  - CR = "the episode contacted any non-target obstacle".
  - SSR = 완료이면서 충돌 없음. "the primary metric".
- **benchmark**: SafeLIBERO 4 suite × Level I / II + 자체 Level III (움직이는 장애물), task 당 50 episode, step 예산 300 / 550.
- **표**: Tab. 1 (SR / CR / SSR), Tab. 2 (per-step latency).
- **status**: `closed-loop`. offline 대리는 EO 같은 세트.

### Multi-link safety filter [1]

- **정의** (arXiv:2609.40007v1 Sec. IV):
  - collision: hazard 위치 차의 축별 절대값 합이 어느 step 에서든 11 mm 를 넘으면 (AEGIS benchmark code 에서 가져옴. 공식 SafeLIBERO code 는 1 mm 다 — 위).
  - robot–hazard contact 는 MuJoCo 에서 따로 기록한다.
  - safe-success = "task completion without collision".
  - latency p50 / p99 (Tab. II), energy (RAPL).
  - 32 scene × 1,600 episode (정적), moving-hazard 6 조건 × 624 episode.
- **link 별 충돌 수는 보고하지 않는다.** EE vs multi-link coverage 변형 비교뿐이다 (Tab. I).
- **status**: `closed-loop`.
- **offline** (EO): link 별 barrier (`h_by_body`, dump), 팔 sphere 의 scene-GT clearance (`arm_*`, `obstacle_min_by_link`), multi-link 은 개입했고 EE-only AEGIS 는 개입하지 않은 call 목록 (`multilink_intervenes_aegis_not`).

### Constrained flow matching [5]

- **정의** (arXiv:2607.01378v1 Sec. 5.1):
  - CAR = "percentage of episodes with no collisions".
  - TSR = "percentage of episodes completing the task".
  - ETS = "mean number of time steps until episode completion".
- 충돌과 완료는 simulator 물체의 최종 위치로 정한다 (Sec. 2.4).
- 안전 조건: 모든 궤적 점 · 장애물에서 B_ij ≥ 0 (Eq. 3; B = signed distance − margin, Eq. 6, 8).
- **status**: CAR / TSR / ETS 는 `closed-loop`. 자기 제약 (이산 CBF 행 g ≥ 0) 은 `offline-now`.
- **우리 대응** (EO): `cfm_min_g`, `cfm_violated_rows` (전 / 후).

### NUDGE [9]

- **정의** (arXiv:2609.35231v2):
  - LIBERO (Sec. IV-C, Tab. III): SR = task success rate. CR = "the percentage of recorded control steps with robot-obstacle contact".
  - sphere task (Sec. IV-B, Tab. II): Coll. Rate = 충돌 중인 control step 비율, Coll. Ep. = 충돌이 있었던 episode 비율, 성공 = 목표 도달이면서 내내 충돌 없음.
  - latency 는 Tab. IV.
  - 논문 내 불일치: Discussion 의 41 % vs 16 % 와 Tab. III (44 / 30) 가 다르다.
- **status**: `closed-loop`.
- **offline** (EO): `nudge_c_sdf` (논문 Eq. 2 의 squared hinge, 전 / 후), hand sphere 의 scene SDF, scene-GT clearance.

### OmniGuide [19]

- **정의** (arXiv:2603.10052v1 Sec. IV-A):
  - Success Rate (RoboCasa 원 정의 그대로).
  - Safety Rate: "an episode is considered a safety failure if even one collision occurs between the robot body and static furniture".
  - 50 trial. latency 는 Fig. 10 (RTX 5090).
- **status**: `closed-loop`.
- **offline** (EO): `omniguide_cost` (collision log-barrier + attractor, 최대화), guided / unguided path, freeze-zone call 수.

### Lan-o3dp [15]

- **정의** (arXiv:2407.00451v3):
  - Sec. IV-A: RLBench 21 task 성공률. 50 epoch 마다 20 episode 를 평가하고 상위 5 개를 평균한다.
  - Sec. IV-B3, Tab. II: 실물 장애물 회피 성공률 (장애물 당 5 trial). "We count failures when the robot or the grasped object hits the obstacles".
- **status**:
  - 원 지표는 `not-applicable` (원 policy · RLBench · checkpoint 없음).
  - 우리가 π0.5 에 옮긴 guidance (adapted) 의 closed-loop 지표는 `closed-loop`.
- **offline** (EO): `lan_rows_inside_circle` (원 안 waypoint 수, 전 / 후).

### LIBERO

- **정의** (arXiv:2306.03310v2 Sec. 5.1, Eq. 3):
  - 모든 지표가 success rate 에서 나온다. task 당 20 rollout, 최대 600 step (App. D).
  - lifelong 지표: FWT_k = (1/11) Σ_e c_{k,k,e}, NBT, AUC.
- **status**:
  - success rate 는 `closed-loop` (env 의 goal predicate).
  - FWT / NBT / AUC 는 `not-applicable`. frozen policy 라 lifelong 학습 과정이 없다.

### SafeLIBERO

- 정의는 VLSA [10] 위의 것 (CAR / TSR / ETS, Level I / II, scenario 당 50 episode).
- **status**: `closed-loop`.

### LIBERO-Safety

- **정의** (arXiv:2606.23686v2 Sec. 4.1):
  - physical track (AAG · HRI · TSA · FSHOA, L0–L2):
    - SR = "strictly requires goal completion without any constraint violations". 위반은 episode 를 즉시 끝내고 실패로 센다.
    - CR = "isolates collision-induced terminations".
    - Execution Time (s).
    - LDLJ = −ln((T³ / v_peak²) ∫‖d³x/dt³‖² dt) (Eq. 2).
  - semantic track (SSR suite): Refusal Rate (task 당 10 회 추론).
  - 표: Tab. 3 (SR), Tab. 5-6 (CR · time · LDLJ), Tab. 4 (RR).
- **status**:
  - SR · CR · time · LDLJ 는 `closed-loop`.
  - RR 은 `not-applicable` (거부 판단은 rollout 이 아니고, 우리 arm 은 그것을 하지 않는다. BM-SETUP-B 범위 밖).

## 3. 우리 offline 지표의 정의 (이 STEP 의 tool)

### NM — `eval/native_metrics.py` (저장 mask 사후 계산)

- **sample**:
  - (frame, camera) 중 GT target mask 가 비지 않은 것.
  - 못 찾은 arm (빈 mask) 은 miss / IoU 0 으로 남는다.
  - GT box = GT segmentation 의 보이는 부분의 tight box (amodal box 는 기록에 없다).
- **집계**: sample 전체 pooled 비율 + episode resample 2,000 회의 95 % CI (seed 20261015). camera 별과 두 camera 합.
- **DAVIS 블록만 예외**: sequence 평균을 쓰고, GT 가 비는 frame 도 포함한다.
- **연속 map arm** (`lh_combined_map`, `v11ab_attention_L8H2`):
  - attention audit run 의 기록을 provider 와 같은 방향 변환 (`to_image_orientation`) 과 bilinear 확대로 256 px 로 올린다.
  - min-max 정규화 후 maxFβ, 0.5 binarize 의 IoU / Acc, 최대점 pointing 을 낸다.
- **검증 장치**:
  - `check_vs_harness_2d_iou`: 같은 mask · 같은 GT 의 IoU 가 harness `2d_iou` 와 같아야 한다.
  - `tokencut_dino_parity`: CPU 재계산 mask = 저장 mask.

### CL — `eval/cloud_labeling.py`

§2 ATS 절 참조.
- visibility 기본은 `observed` (V11AB 가 그 점을 관측한 camera 만).
- `depth` (|depth − z| ≤ 10 mm) 와 `any` 도 고를 수 있다.

### EO — `methods/evaluate_offline.py`

- **입력**: SL3 dev 10 episode · 221 call 의 기록된 π0.5 chunk.
- **replay 방식**: open loop. 수정한 chunk 를 실행하지 않는다. 상태를 가진 method 는 episode 마다 reset 하고 시간 순서로 부른다.

| 지표 | 정의 |
|---|---|
| 개입 | filter: method 의 `intervened` (max \|Δa[:6]\| > 1e-6). guidance: \|Δa_xyz\| > 1e-4 raw 인 행 (D 의 규칙) |
| `intervention_rate_rows` / `_calls` | 개입 행 / 전체 행, 개입이 하나라도 있는 call 의 비율 |
| `da_xyz_l2_*` | 행마다 ‖Δa_xyz‖₂ (LIBERO raw action 단위). 전체 평균, 개입 행 평균, 최대 |
| `h_min`, `h_neg_steps` | method 자신의 barrier. KNOWS = EE ellipsoid–물체 MVEE, AEGIS = 장애물별 최소, multi-link = (body, 장애물) 최소. **method 사이 비교 불가**. h < 0 = 그 예측 상태가 제약을 어김 |
| `qp_fail_steps`, `emergency_stops` | infeasible / solver error / nominal 로 fallback 한 step 수, KNOWS 비상 정지 수 |
| `ms_per_call` / `ms_build` · `ms_jit_and_sample` | CPU 1 thread wall ms. guidance 는 jit compile 을 포함한다 (call 마다 closure 가 새로 만들어진다) |
| 자기 cost (guidance) | OmniGuide cost (최대화), NUDGE C_SDF, CFM min g · 위반 행, Lan-o3dp 원 안 행. 전 / 후 |
| scene SDF (guidance) | D 의 `path_metrics`: EE · hand sphere 의 depth-lift 장애물 (또는 injected sphere) 까지 signed distance 최소 |
| **scene-GT clearance** (전 method 공통) | chunk → OSC response 모델 EE 경로 (자세 고정, action ∈ [−1, 1]) → 기록 q 에서 DLS IK → cuRobo `franka.yml` sphere 61 개 → scene GT 표면까지 정확한 거리 (안쪽은 음수) − 반지름, 0.05 m 상한 (상한 아래에서는 `exact_distance_near` 와 bit 단위로 같은 pruned 계산). 군: obstacle (target · destination · 받침대 제외), destination, support. sphere 군: ee (hand + finger), arm (link1-7). nominal 과 수정본 각각 행 1..T 최소와 음수 행 수 |

guidance 의 한계 (출력에 기록됨): **π0.5 network 대신 D 의 대체 velocity** 를 쓴다 (기록 chunk 로 가는 직선 flow).
`tracking` 은 이후 step 이 수정을 되돌리는 쪽 (최악), `constant` 는 수정이 끝까지 남는 쪽 (최선) 이다.
대체 sampler 를 trace 판 (`surrogate_sample_traced`) 으로 돌리고, D 의 `surrogate_sample` 과 bit 단위로 같은지
arm × variant × surrogate 마다 첫 call 에서 확인한다 (`trace_matches_surrogate_sample`).

## 4. 출처 목록

| 출처 | 판 / SHA | 본 곳 |
|---|---|---|
| main.pdf (AG3S) | CVPR 2027 draft, repo root (2026-10-23) | Sec. 4.1, Tab. 1-4 |
| Grounded-SAM, arXiv:2401.14159 | v1 | Abstract, Sec. 4, Tab. 1 |
| SAM 2, arXiv:2408.00714 | v2 | Sec. 6 |
| Grounded-SAM-2 `sav_dataset/utils/sav_benchmark.py` | `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28` | 156-303 |
| davis2017-evaluation `davis2017/metrics.py`, `utils.py` | GitHub master (raw, 2026-10-23) | `f_measure` (bound_th 0.008), `db_eval_iou`, `db_statistics` |
| YOLOE, arXiv:2503.07465 | v2 (ICCV 2025 camera-ready) | Tab. 1-4 |
| Localization heads, arXiv:2503.06287 | v1 (CVPR 2025) | Sec. 6.1, Tab. 1-2 |
| Zhang et al., Top-down Neural Attention by Excitation Backprop, arXiv:1608.00507 | PDF | Sec. 4.1, p. 10-11 |
| TokenCut, arXiv:2202.11539 | v2 (CVPR 2022) | Sec. 4.1, 4.3, Tab. 1, 3 |
| TokenCut code | `fed52cd5b60891baefd8ec7110dafa73be816ee1` | `main_tokencut.py:295-298`, `datasets.py:289`, `unsupervised_saliency_detection/metric.py:5-81` |
| VLSA (AEGIS, SafeLIBERO), arXiv:2512.11891 | v2 | Sec. V-A, Tab. I |
| SafeLIBERO code `main/main_aegis.py` | `2457feed5968ae803926e178c8ce8243b9ecdcf9` | 370, 404, 420 |
| KNOWS, arXiv:2606.09749 | v1 | Sec. 4.1, Tab. 1-2 |
| Multi-link safety filtering, arXiv:2609.40007 | v1 | Sec. IV, V-A, Tab. I-II |
| Constrained flow matching, arXiv:2607.01378 | v1 | Sec. 2.4, 5.1, Eq. 3, Tab. 1-2 |
| NUDGE, arXiv:2609.35231 | v2 | Sec. IV-B, IV-C, Tab. II-IV |
| OmniGuide, arXiv:2603.10052 | v1 | Sec. IV-A, Fig. 5, 10 |
| Lan-o3dp, arXiv:2407.00451 | v3 | Sec. IV-A, IV-B3, Tab. II |
| LIBERO, arXiv:2306.03310 | v2 | Sec. 5.1 Eq. 3, App. D |
| LIBERO-Safety, arXiv:2606.23686 | v2 | Sec. 4.1, Eq. 2, Tab. 3-6 |

## 5. 확인하지 못한 것

- 논문 HTML 은 WebFetch 요약으로 읽었다. 정의 문장 · 표 번호는 위와 같이 확인했지만, 표의 숫자는 옮기지 않았다.
- davis2017-evaluation 은 pinned clone 이 없다 (GitHub master 를 읽음). `db_statistics` 의 decay bin 은 기억과 요약을 맞춰 구현했다 (`tests/test_native_metrics.py::test_db_statistics` 가 bin 경계를 고정한다).
- SAM 2 가 DAVIS 평가에 쓴 toolkit 의 정확한 판은 논문 본문에서 확인하지 못했다 (부록 F 미확인).
- TokenCut 논문 HTML 의 "β=0.3" 과 code 의 `betta_sq=0.3` 가 다르다. code 를 따랐다.
- Multi-link [1] 의 11 mm 와 공식 SafeLIBERO code 의 1 mm 는 서로 다른 collision 기준이다. closed-loop 비교 때 어느 쪽을 쓰는지 명시해야 한다.
