# IFG S0–S3 작업 기록 (internal feature + RGB-D target extent)

이 파일은 [START_HERE_KO.md](../../../archive/reviews/vla_dump_20261015/START_HERE_KO.md)에 따라 진행한 S0–S3 작업의 시간순 기록이다. 실행 정본은 [EXPERIMENT_PROTOCOL_KO.md](EXPERIMENT_PROTOCOL_KO.md) v1.1이다. 각 항목에는 실행한 것과 실행하지 않은 것을 함께 적는다. 수치는 아래 run 디렉터리의 산출물에서 직접 읽은 값이다.

- Checkout: `/mnt/dev/AG3S_VLA-development`, branch `feature/ag3s-development`, HEAD `30955b6c`(작업 시작 시점, 커밋 전 WIP 포함)
- Run: `/mnt/dev/AG3S_VLA-storage/results/development/20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba/` (아래 `$RUN`)
- Dump(다른 session의 collector, 읽기 전용): `.../20261015T150325+0900_ddc7a00060c4_d857a08dcb63/phase_dump/`
- Interpreter: `.venv-ag3s` (CPU, MuJoCo osmesa). GPU는 사용하지 않았다.

---

## 2026-10-15 18:17–18:30 KST — S0: frame022 재현과 phase dump 정합 검사

### 1. Frame022 기존 진단의 재현

[grounding_probe.py](../../../archive/reviews/vla_dump_20261015/grounding_probe.py)를 문서의 명령 그대로 새 run에 실행했다(`$RUN/s0/frame022_repro/`, 6.8 s).

| 비교 | 결과 |
|---|---|
| 기존 [grounding_metrics.json](../../../archive/reviews/vla_dump_20261015/grounding_metrics.json)과 float 822개 대조 | 최대 차이 **0** (bitwise 동일) |
| 새 범용 S1 모듈(`ifg_s1.py`)로 같은 frame 재계산, 14 arm × K{5,10,20,40} × P/R/IoU = 56칸 | 최대 차이 **0** |
| 16×16 whole-patch oracle IoU | 0.7087 (기존과 동일) |

참고로, 기존 진단에 없던 T61 control map A61(`action:mean:L10H7`, d0)의 frame022 top-20 IoU는 0.1117이었다. 이 frame은 `debug_regression`이며 성능 평가에 넣지 않는다.

### 2. Phase dump 구조 확인 (collector는 건드리지 않음)

- Collector: `phase_dump/vla_phase_dump.py`. T53 + T54c record에서 GT grasp 상태로 **미리 고정한 phase key-moment** frame을 뽑는다. 성공 episode는 M1 approach … M7 retreat의 7개, 실패 episode는 F1–F3의 3개이며 계획은 총 180 frame이다.
- 저장 항목: `inputs`(224 RGB, state, token, named query row), `embed`(`image_tokens`=projector, SigLIP `encoded`), `prefix`(18 layer `resid_out` 등), `suffix`(`attn_probs` 10×18×8×50×1018), `derived`(serving L8H2 map, noun L10H7 map, `kv_L4`, HSR), `grad`.
- **Depth, K, T는 dump에 없다.** 원본 record도 `depth_cameras=[]`다. 따라서 frame022와 같이 record의 qpos로 MuJoCo를 replay해 simulated depth를 만든다(실제 센서 기록이 아님).
- S1의 5 feature(SigLIP encoded, projector, prefix L4/L8/L12)와 4 map(A, N, P, A61)은 모두 이 dump에서 계산된다. A61은 `attn_probs[d0, L10, H7, :, :768]`의 query 평균이다.

### 3. S0 전수 검사 (`ifg_s0.py`)

18:23 시점에 완료된 97 frame(18 episode: T53 16개 + T54c 2개 진행분)에 대해 실행했다. CPU worker 4개, 203 s.

| 검사 | 기준 | 결과 (97 frame / 291 camera) |
|---|---|---|
| dump `raw_state` vs record `state` | ≤1e-5 | 최대 1.07e-7 |
| dump image vs record uint8 image | ≤1 (8-bit) | 최대 1.5e-5 |
| `derived` serving map vs `attn_probs[0,8,2,-1]` | — | 최대 3.0e-8 |
| GT fingerprint vs record | 일치 | 97/97 |
| Replay patch class coverage vs 저장 GT | 0 | 최대 **0** |
| Replay RGB MAD (8-bit) | ≤1 | 0.0011–0.433 |
| Depth convention: `mj_ray` 거리로 검사 | planar z ≤1 mm | planar z 최대 오차 **2.1e-6 m** (radial로 해석하면 0.66–2.21 m 어긋남) |
| Back-project → project roundtrip | ≤0.5 px | 최대 ~1e-12 px |
| S1 입력 feature/map의 non-finite | 0 | 0 |

**결론: 97 frame 모두 S0 통과.** Depth는 planar z-depth(OpenCV, camera→base `T_base_cam`)로 확정했다.

발견·주의:

- **S0-F1 (FP16 overflow):** collector의 float16 cast에서 41 frame의 `prefix resid_out`·`mlp_out`에 inf가 있다. 위치는 22,368개 모두 **L17(마지막 layer)의 text row**였다. S1–S3가 쓰는 L4/L8/L12 image row, encoded, image_tokens, attention map에는 없다. 프로토콜 §4의 "BF16→FP16 overflow 불허"에 따라 L17 text row를 쓰는 후속 분석(phase/value 연구)에서는 이 member를 사용 불가로 표시해야 한다.
- **S0-F2 (action parity):** collector가 기록한 `x0_vs_orig_maxabs`는 대부분 ≤0.021이지만 T54c ep1806 f019는 0.0885였다. S1–S3(공간 grounding)에는 영향이 없다. S5(같은 noise의 action 인과 검증)를 한다면 이 frame의 repeatability를 먼저 확인한다.
- **S0-F3 (ray 검사 구현 주의):** 처음에는 `zed_left` ray가 camera housing body(4 mm 거리, renderer near plane 2.3 cm에서 잘림)에 맞아 실패로 나왔다. Camera가 붙은 body를 ray에서 제외하고 렌더 group 0–2만 쓰도록 고친 뒤 통과했다. Depth 자체의 문제가 아니다.

산출물: `$RUN/s0/provenance_checks.jsonl`, `$RUN/s0/replay/<frame_id>.npz`(depth·K·T, algorithm 입력), `$RUN/s0/labels/<frame_id>.npz`(body id·class map, evaluator 전용).

재현:

```bash
cd /mnt/dev/AG3S_VLA-development
MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv-ag3s/bin/python \
  -m AG3S.ag3s.experiments.studies.ifg_s0 --out $RUN/s0 --workers 4   # 이미 처리한 frame은 건너뜀
```

미실행: 실제 센서 depth 정합(없음), compact collector parity(새 profile collector가 아님), S5 repeatability.

### 4. Split 관련 사실

현재 dump의 episode는 모두 T53·T54c에서 왔다. 프로토콜 §2는 "이미 결과를 검토한 T53/T54c 사례는 regression/stress 진단으로 표시하고 새 독립 test라고 부르지 않는다"고 정한다. 따라서 현 dump만으로는 **locked held-out test가 없다.**

## 2026-10-15 18:35 KST — 사용자 판정 3건

| 항목 | 판정 |
|---|---|
| Split | **현 dump(T53+T54c) 전부 dev.** 방법 선택에만 쓴다. Locked test는 collector가 새 seed/배치 episode를 수집하면 그때 한 번 평가한다. |
| 기록 위치 | **이 WORKLOG만.** `AG3S_T47_LOG.md`에 STEP 번호를 예약하지 않는다. |
| Baseline 조건 | **T54 `current`(frame마다 AG3S reset, history 없음) 먼저.** `hold` 조건은 공간 비교 결론 뒤에 판단한다. |
| GPU | S0–S3는 CPU만 쓴다. GPU lock은 dump collector가 보유 중이며 이 작업은 GPU를 요청하지 않았다. |

## 2026-10-15 18:25–18:45 KST — S1·S2·S3 구현과 smoke 중 발견

### 1. 구현한 모듈 (모두 `AG3S/ag3s/experiments/studies/`, serving 코드는 수정하지 않음)

| 파일 | 역할 |
|---|---|
| `ifg_reader.py` | phase dump reader. 5 feature(SigLIP encoded, projector, prefix L4/L8/L12 `resid_out`)와 map A·N·P·A61·A_first8. Record·GT(evaluator 전용) 연결 |
| `ifg_s0.py` | S0 정합 검사 + replay depth/K/T + **serving FK robot mask** 저장(`build_robot_model` + `set_finger_joints_from_scene` + `depth_robot_mask`, serving c1a와 같은 기본 self-filter) |
| `ifg_s1.py` | S1 seed×feature prototype(15 arm) + map control 5종. Patch top-K 진단. Frame022 56칸 재현 차이 0 |
| `ifg_graph.py` | S2/S3 surface graph와 G/F/FGe/FGu/FG membership |
| `ifg_run.py` | frame 단위 batch runner (worker 4) |
| `ifg_baseline.py` | T54 arm(L8H2·N·B)을 `current` 조건으로 재실행하고 serving이 고른 cluster(region)와 bounding primitive(ball)를 같은 eval voxel로 평가 |
| `ifg_summary.py`, `ifg_figures.py` | episode 평균·paired bootstrap(2000회, seed 20261015, episode 단위)·figure |

### 2. S2/S3 graph 정의 (실행 전 고정, 결과를 보고 바꾼 항목은 아래 3절에 따로 적음)

- **Pixel → 점:** serving `backproject`(정수 pixel 좌표)와 serving `range_max=2.0 m`.
- **Surface:** 4-neighbour pixel 쌍의 z 차이가 10 mm 이하일 때만 같은 surface. 이 gate를 넘는 edge는 만들지 않는다.
- **Node:** 5 mm base-frame voxel. 세 camera를 하나의 grid로 합치므로 여러 camera가 본 표면도 node는 하나다.
- **Eval voxel:** robot pixel을 포함한 모든 유효 pixel의 같은 grid. 각 voxel weight는 1(occupancy)이고, target 비율은 soft 값이다.
- **Edge:** 어떤 camera에서 같은 surface의 인접 pixel 쌍이 두 voxel에 걸치고, centroid 거리가 15 mm 이하일 때만 만든다. 3D 근접만으로는 연결하지 않는다.
- **Normal:** node와 1-ring centroid의 PCA. 점이 4개 미만이면 unavailable.
- **Unary:**
  - seed: 상위 5개 seed patch의 질량(w/max w)을 pixel 비율대로 node에 나눈다(질량 보존).
  - support: serving RANSAC(`fit_support_surfaces`) plane에 s=0, Λ=1.
  - prototype(FG·FGu만): s=clip((cos−c0)/(c1−c0)), c1은 seed patch 평균 cosine, c0은 frame 중앙값이다. Λ=0.1.
- **해:** component별로 (Λ+λL)m=Λs를 푼다(λ=1). Unary가 없는 component는 unknown, ROI는 seed node에서 0.15 m 안이다.
- **Arm:**
  - G: geometry edge + seed/support unary
  - F: prototype score만
  - FGe: G에 feature affinity edge를 더함(σ_f=0.5)
  - FGu: G에 prototype unary를 더함
  - FG: FGe와 FGu를 모두 적용

### 3. Smoke 중 발견과 수정 (결과를 본 뒤 고친 것은 명시)

- **S2-F1 (support plane 미검출 → 수정):** 처음에는 PointCloudConfig 기본값(z≤3 m, range gate 없음)을 썼다. 이때 RANSAC은 바닥(z≈0.002 m)만 찾았고 table 상판(z≈0.82 m)은 하나도 support로 잡지 못했다. 그 결과 mixed seed patch의 작은 positive evidence가 table component 전체를 m=1로 만들었다(ep1839 f023 G precision 0.012). Serving의 `--range-max` 기본 2.0 m를 적용하자 table node의 81–86%가 support로 검출됐고 target 오검출은 0이었다. 문서의 "z≤3.0 m" 기본과 다르며, serving 조건과 맞추기 위한 변경이다.
- **S2-F2 (속도 → 수정):** 전체 장면 461k node에 2048-d node feature를 만들어 frame당 178 s가 걸렸다. ROI sub-graph에서만 계산하도록 바꿔 5–6 s가 됐다.
- **S2-F3 (serving FK self-filter가 held target을 지움, 공통 입력):** 141 frame에서 FK mask의 GT robot pixel recall은 세 camera 모두 1.0이었다. 반면 grasp(M3) frame에서는 target pixel을 중앙값 4,069개(최대 10,165개) 지웠다. Serving c1a에도 target guard가 없으므로 모든 arm과 baseline에 같은 조건이다. 이렇게 사라진 target은 recall 분모에 남는다(`target_lost_to_robot_filter`).
- **S0-F4 (pixel 중심 규약):** Serving `backproject`는 pixel 중심 +0.5 없이 정수 좌표를 쓴다. MuJoCo 렌더 기준으로는 약 0.5 px(1 m에서 약 2 mm)의 계통 offset이다. 모든 arm이 같은 함수를 쓰므로 비교에는 영향이 없다. Serving은 수정하지 않았다.
- **S0 판정 기준 개정 (결과를 본 뒤):** T54c 2 frame(ep1820 f006, ep1817 f010)이 `zed_left` planar-z 검사에서 실패했다. 원인은 24개 ray 중 1개가 banana mesh 표면에서 1.4–1.6 mm 어긋난 것이다(나머지는 µm 수준). 기준을 "최대 ≤1 mm"에서 "중앙값 ≤0.1 mm 그리고 90% 이상 ray ≤1 mm"로 바꾸고 두 frame을 다시 검사했다.

## 2026-10-15 18:45–19:05 KST — R0 최초 비교 (dev 180 frame / 32 episode)

Collector가 계획한 180 frame(T53 16 ep + T54c 16 ep)이 모두 완료됐다. S0 v2(robot mask 포함)는 180 frame 모두 통과했다(개정 기준 적용, 위 S0 판정 기준 개정 항목 참조). Baseline은 T54 arm(L8H2·N·B)을 c1a profile, `current` 조건으로 같은 180 frame에서 다시 실행했다. Serving fusion voxel은 0.01 m였다.

**고정 reference와 기준** (`improvement_registry.jsonl`의 R0, baseline 숫자를 열기 전에 등록):

- 고정 reference: `G:N`. B0는 `T54region:N`이다.
- 선택 규칙: Δrecall ≥ −0.02, Δfalse-exemption proxy ≤ 0, episode macro 3D IoU 최대.
- 실용적 최소 효과: +0.02 IoU(pipeline이 결정적이라 반복 오차 0).
- Runtime 예산: arm p95 ≤ 500 ms.
- False-exemption은 permission builder를 돌리지 않았으므로 segmentation FP(non-target voxel)를 **proxy**로 쓴다.
- G/F/FG 예비 수치(139 frame)는 등록 전에 이미 봤다.

| Arm (τ=0.5) | IoU | Precision | Recall |
|---|---:|---:|---:|
| T54region:B | 0.787 | 0.899 | 0.828 |
| **T54region:N (B0)** | 0.785 | 0.896 | 0.828 |
| T54ball:N | 0.764 | 0.854 | 0.844 |
| **G:N (reference)** | 0.728 | 0.873 | 0.820 |
| FGe:N:projector (feature edge만) | 0.728 | 0.869 | 0.820 |
| FGu:N:projector (feature unary만) | 0.585 | 0.692 | 0.777 |
| FG:N:projector | 0.582 | 0.690 | 0.774 |
| T54region:L8H2 (serving 기본 source) | 0.596 | 0.723 | 0.632 |
| G:P / G:A | 0.413 / 0.309 | 0.572 / 0.481 | 0.484 / 0.365 |
| F:N:projector | 0.256 | 0.292 | 0.721 |

- FG − G(seed N)는 5개 feature 모두 음수이고 CI가 0을 포함하지 않는다(−0.094 ~ −0.187).
- G:N − B0 = −0.057 [−0.114, +0.001]이다.
- 2×2 mechanism control(v1.2 §7)에서 FGe≈G, FGu≈FG다. 즉 **악화는 feature unary에서 온다.**

**R0 진단** (`rounds/R0/diag/diagnostics.jsonl`):

- G의 FP는 14,005 voxel이고 그중 13,978개가 이웃 물체다. 모두 seed가 닿은 component 안에 있다. Mixed seed patch가 이웃 물체 component에 작은 양성 질량을 주고, positive-only component가 m≡1이 되는 퇴화(F5)다.
- FG의 FP는 74,600 voxel(anchored 65,812 + unanchored 8,788)이다. 증가분은 M7 retreat(+31.6k)과 M4 carry(+14.2k)에 몰려 있다.
- Prototype score 자체는 target과 이웃을 구분한다(node 평균 0.775 대 0.233). 원인은 patch 해상도 score를 그 patch 안 모든 surface(예: crate 벽)에 복사하는 데 있다.

## 2026-10-15 19:05–19:20 KST — 개선 cycle R1–R3 (가이드 §15, v1.2 §13.1)

v1.2 갱신(사용자 문서, 18:59 KST)에 따라 최초 실패에서 멈추지 않고 cycle을 돌렸다. 각 round는 `$RUN/rounds/Rx/`에 다음을 남겼다: `hypothesis.json`(실행 전 registry 등록), `commands.txt`, rows, `paired_summary.json`, `episode_metrics.csv`, `decision.json`. 모든 round에서 parent G:N을 같은 코드로 다시 계산해 R0와의 parity가 **0**임을 확인했다.

| Round | 원인 ID · 가설 | 바꾼 것 | 결과 (vs G:N, episode paired 95% CI) | 판정 |
|---|---|---|---|---|
| R1 | F5: feature는 patch 해상도이므로 node마다 복사하지 말고 component 단위 identity gate로 써야 한다 | anchored graph component의 pixel 가중 prototype score q<0.5면 기각 | FGc:N:prefix_L8 IoU −0.086, recall −0.165. 이웃 FP 86→14. Shuffle IoU 0.390. **Oracle component gate도 recall −0.149** | 기각. Component가 target과 table을 함께 묶는다(새 증거) |
| (진단) | F3 edge 누출인가 | `rounds/R2/edge_diag` | Target 경계 edge는 180 frame 전체에서 317개뿐. Feature 거리는 경계 구분 불가(AUC 0.49–0.54, 양쪽이 같은 patch feature). Normal(0.27)·convexity(0.20)는 구분 | F3는 주 원인이 아님. R1 실패는 gate **단위** 문제 |
| R2 | F5: gate 단위를 G 예측 영역 {m^G≥0.5}의 connected region으로 | region별 q | FGr:N:prefix_L8 IoU **+0.026** [−0.013, +0.074], recall **−0.050** [−0.081, −0.022]. 이웃 FP 86→15. Oracle region IoU +0.070 | 기각(recall 제약). 진단 지표가 개선돼 분기는 유지 |
| R3 | F5: seed 질량 1위 region은 target(166/172)이므로 보호하고 부차 region만 gate | 1위 region 보호 | FGr1:N:prefix_L8 IoU +0.057 [+0.022, +0.102], recall −0.011. **그러나 shuffle +0.055, feature 없는 top1-only +0.065 [+0.031, +0.111]** | **Feature gate 분기 음성 종료.** 이득은 feature가 아니라 "seed 질량 1위 region" 규칙에서 온다 |

**R3 이후 상태 (3-cycle checkpoint):**

- **Feature의 추가 가치: dev에서 확인되지 않음.**
  - Feature는 identity 정보를 담는다. R1·R2의 shuffle control이 훨씬 나빴다(IoU 0.36–0.43).
  - 그러나 noun attention seed의 질량 순위를 쓰고 나면 같은 정보가 중복된다.
  - Feature는 boundary edge 수준에서 경계를 구분하지 못한다. 경계 양쪽이 같은 16×16 patch feature를 공유하기 때문이다.
- **제약을 통과한 후보는 geometry-only `top1-only:N`이다.** G graph에서 noun seed 질량이 가장 큰 예측 영역 하나만 남긴다.
  - IoU 0.794, precision 0.961, recall 0.807.
  - vs G: IoU +0.065 [+0.031, +0.111], recall −0.013.
  - vs B0: IoU +0.009 [−0.036, +0.055], recall −0.020 [−0.062, +0.023]. **기존 serving region과 동등 수준이며, 더 낫다는 근거는 없다.**
  - Dev GT를 보고 설계했으므로 label-free 선택이 아니고, held-out 검증도 없다.
- **남은 실패:**
  - (a) FN 7,711 voxel 중 3,579개(46%)가 FK self-filter가 지운 held target이다(M3–M5). Serving과 공통 입력이다.
  - (b) IoU≈0 frame이 12개이고 그중 10개는 B0가 성공한다. 원인은 seed identity(F1)다: 들고 있는 과일 pixel이 FK filter로 지워져 그 patch가 seed 자격(비robot depth ≥60 px)을 잃는다.

**Figure** (`AG3S/ag3s/docs/figures/ifg/`, 같은 이름 `.json` sidecar에 수치):

- `ifg-scene-T53_ep1839_f023.png`, `ifg-scene-T54c_ep1806_f031.png`(FG가 crate 전체로 번지는 실패, G IoU 1.00 대 FG 0.04), `ifg-scene-T53_ep1849_f024.png`(FG가 G보다 나은 사례): 실제 policy image에 5 mm node를 투영한 TP/FP/FN.
- `ifg-episode-iou.png`: 전 arm의 episode IoU와 FG−G paired CI.
- `ifg-table.png`: episode 평균 표.
- `ifg-rounds.png`: R0–R3 paired 차이(IoU, recall)와 제약선.

**산출물** (`$RUN`): `split_manifest.jsonl`(32 episode, 전부 dev), `improvement_registry.jsonl`(R0–R3), `s3/{sample_metrics.csv, episode_metrics.csv, paired_summary.json, latency.json, resolved_config.json}`, `baseline/`, `rounds/R0–R3/`.

**재현 명령:**

```bash
cd /mnt/dev/AG3S_VLA-development; RUN=/mnt/dev/AG3S_VLA-storage/results/development/20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba
bash $RUN/pipeline.sh                      # S0 -> S1-S3 runner -> T54 baseline replay/eval (CPU, worker 4)
PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_summary --s3 $RUN/s3 --s0 $RUN/s0 --baseline $RUN/baseline
PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_diag --s0 $RUN/s0 --out $RUN/rounds/R0/diag
bash $RUN/rounds/R1/commands.txt; bash $RUN/rounds/R2/commands.txt; bash $RUN/rounds/R3/commands.txt
PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_edge_diag --s0 $RUN/s0 --out $RUN/rounds/R2/edge_diag
PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_figures --run $RUN --out AG3S/ag3s/docs/figures/ifg --baseline
PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_round_figure --run $RUN --out AG3S/ag3s/docs/figures/ifg
```

**비용:**

- Frame당 graph build 약 0.4–0.6 s(serving range 적용 후), seed당 G solve 수십 ms.
- R0 batch: 180 frame에 worker 4개로 약 5분. Baseline replay: 약 15분.
- `s3/latency.json`의 p50/p95는 offline CPU 수치이며 serving latency가 아니다.

**미실행:**

- Locked test(사용자 판정: 새 episode 수집 후)
- 실제 permission builder의 false-exemption(segmentation FP proxy만 측정)
- T54 `hold` 조건
- S4 multiview/time 비교
- P0–P4
- S5 인과 검증
- closed-loop
- 15 prototype 전체에 대한 dev pair 선택 이후의 σ_f/λ/τ one-factor sensitivity. R0 FG가 G보다 일관되게 나빠서 v1.2 §13.1 원인 진단을 먼저 했다.

메모리 회수: `reclaim: ws 36.1->26.7 GiB | cache 38.5->0.0 GiB, 3704 files, 2 paths`.

## 2026-10-15 19:25 KST — 사용자 판정: 다음 묶음 방향

| 항목 | 판정 |
|---|---|
| R4 이후 묶음 | **F1 seed + held object 제거** (feature gate 분기는 R3에서 음성 종료) |

## 2026-10-15 21:40–21:50 KST — 개선 cycle R4–R6 (F1 seed, held 제거)

시작 전에 `ifg_*.py` 파일이 바뀌었다는 알림이 있어 확인했다. 모든 파일의 마지막 수정이 19:20 KST(R3 종료) 이전이어서 외부 수정은 없었다.

| Round | 원인 ID · 가설 | 바꾼 것 | 결과 (episode paired 95% CI) | 판정 |
|---|---|---|---|---|
| (진단) | F1: FK filter가 held 과일을 지워 seed patch가 자격을 잃는가 | `rounds/R4/diag/seed_diag.jsonl` | **기각.** 실패 frame에서 자격 제한 유무와 관계없이 top-5 seed가 동일하다. Noun map의 patch 최댓값 자체가 target 밖에 있다(seed target 비율 <0.05인 frame 9/180) | 다른 가설로 |
| R4 | F1: serving처럼 region 단위 attention **질량 합**으로 identity 선택 | depth-surface region(support·robot 제외)별 noun 질량, 최대 region | MassTop1:N IoU 0.688(top1-only 대비 −0.106), 이웃 FP 1,006. Top1 실패 12개 중 4개를 구하고 성공 22개를 망침. **O-region-best(GT region 선택) IoU 0.843, B0 대비 +0.058 [+0.029, +0.090]** | 기각(질량의 크기 편향). Region 표현의 상한은 serving보다 높다 → 병목은 identity |
| R5 | F1: seed 자격 pixel에 table이 포함됨(실패 frame seed pixel의 71%가 support, 성공 frame은 34%) | seed 자격을 비robot·**비support** pixel ≥60으로(serving은 support를 seed에서 제외) | top1-only-ns:N IoU 0.817. vs G +0.089 [+0.049, +0.137], recall +0.011. **vs R3 +0.023 [+0.005, +0.049](7/0 episode)**. Moment 회귀 없음 | **채택(dev 후보)**. vs B0 +0.032 [−0.008, +0.070], 우위 미확인 |
| R6 | F2: serving sphere/capsule self-filter가 손가락 사이 과일을 지운다(FN의 46%) | robot mesh를 FK 자세로 렌더한 depth와 관측 depth가 10 mm 안에서 일치할 때만 robot (`ifg_render_mask.py`) | top1-only-ns+rm:N IoU **0.893**, recall 0.915, precision 0.938. **vs R5 +0.076 [+0.060, +0.093](31/0)**. M3 0.724→0.935, M4 0.598→0.748, M5 0.762→0.913. GT robot recall 1.000(540/540), 지운 target px 311,296→12,218, robot FP voxel 0.1→1.6/frame | **채택(dev 후보)** |

**R6 해석의 한계:**

- vs B0 +0.108 [+0.072, +0.143]은 **robot filter 변경 효과를 포함**한다. B0(serving)는 sphere filter 그대로이고, 같은 filter 조건의 비교는 R5(+0.032, CI가 0 포함)다. Serving을 render mask로 다시 돌린 비교는 하지 않았다.
- Simulation의 mesh와 렌더가 정확히 일치하므로 실제 센서 depth에서는 유리한 조건이다. Tolerance는 10 mm다.
- 모두 dev GT를 보고 설계했고 held-out은 없다.

**Moment별 frame 평균 IoU** (R6 / R5 / B0):

| Moment | F1 | F2 | F3 | M1 | M2 | M3 | M4 | M5 | M6 | M7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R6 | 0.940 | 0.910 | **0.745** | 0.998 | 0.967 | 0.935 | 0.748 | 0.913 | 0.911 | 0.882 |
| R5 | 0.927 | 0.768 | 0.738 | 0.983 | 0.896 | 0.724 | 0.598 | 0.762 | 0.896 | 0.882 |
| B0 | 0.925 | 0.581 | 0.918 | 0.982 | 0.911 | 0.717 | 0.624 | 0.811 | 0.748 | 0.617 |

남은 약점은 F3(실패 grasp 직후, B0 0.918 대 0.745)와 M4 carry다. F3는 seed identity 실패(noun 최댓값이 target 밖)로 남아 있다.

**Figure:**

- `AG3S/ag3s/docs/figures/ifg/ifg-rounds-r4r6.png` (+json): R3–R6 paired 차이와 moment별 IoU.
- `ifg-scene-r6-T54c_ep1817_f020.png` (+json): M3 grasp, 오른손 camera. Sphere mask는 과일 4,061 px를 지워 IoU 0.55, render mask는 219 px만 지워 IoU 0.96.
- R0–R3 figure는 증거로 남겨 두고 덮어쓰지 않았다.

**재현:** `bash $RUN/rounds/R5/commands.txt`, `bash $RUN/rounds/R6/commands.txt`, `PYTHONPATH=. MUJOCO_GL=osmesa .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_r6_figures --run $RUN --out AG3S/ag3s/docs/figures/ifg`

**새 코드:** `ifg_seed_diag.py`, `ifg_round_mass.py`(R4), `ifg_round_ns.py`(R5/R6, `--mask-dir`), `ifg_render_mask.py`, `ifg_r6_figures.py`. Registry: `$RUN/improvement_registry.jsonl` R0–R6.

## 2026-10-15 22:30–22:20 KST — R7 A+F+3D 재검증, R8 pixel gradient 진단 (사용자 요청)

사용자 요청: (1) 고친 기반에서 Attention + Feature graph + 3D 재검증, (2) dump의 pixel gradient 시험(CPU), (3) [3D Surface Graph 가이드](../AG3S_3D_Surface_Graph_and_Parameter_Tuning_Guide.md) 참고. 가이드에서 반영한 것:

- §10 ablation 중 아직 없던 **A3 Feature+Geometry Region Growing(FG-R)**과 A2 G-R
- A5 oracle seed
- §13: "R과 Q 차이가 없으면 단순한 쪽을 택한다"
- Binary cut C는 solver 문제(F6) 근거가 없어 추가하지 않았다.

### R7 — 고친 기반(R6 render mask + R5 non-support seed)에서 A+F+3D 재검증

- 기본값은 R0와 같다. Parity: G-Q / G-Q+top1이 R6의 G-ns+rm / top1-only-ns+rm과 frame 단위로 일치했다.
- 비교군: 2×2(G/FGe/FGu/FG) × feature 5종, 각 arm의 +top1 버전, R arm(G-R, FG-R), oracle seed, shuffle. 180 frame, worker 4, 198 s.

| Arm (+top1, parent = G-Q+top1 0.893) | IoU | parent 대비 IoU [95% CI] | recall 차 |
|---|---:|---|---:|
| FGe-Q (feature edge) × 4 feature | 0.893–0.894 | ±0.000 | 0.000 |
| FGe-Q:siglip_encoded | 0.872 | −0.021 [−0.047, −0.001] | −0.021 |
| FGu-Q (feature unary) | 0.790–0.831 | −0.062 ~ −0.104 | −0.070 ~ −0.114 |
| FG-Q (둘 다) | 0.787–0.826 | −0.067 ~ −0.106 | −0.073 ~ −0.116 |
| FG-R (가이드 A3) | 0.736–0.755 | −0.138 ~ −0.157 | −0.142 ~ −0.166 |
| G-R (가이드 A2) | 0.832 | −0.061 [−0.091, −0.034] | −0.040 |
| FG-Q-shuffle:prefix_L8 | 0.485 | −0.408 | −0.428 |
| G-Q:O (oracle seed) | 0.918 | +0.025 [+0.004, +0.051] | +0.025 |
| FG-Q:prefix_L8:O (oracle seed) | 0.901 | +0.008 [−0.020, +0.039] | 0.000 |

**판정: 음성.** 등록한 반증 조건(feature arm ≥ parent + 0.02, recall ≥ −0.02, shuffle이 재현하지 않음)을 통과한 arm이 없다.

- **Feature edge 항:** 효과가 없다. 경계 node가 같은 patch feature를 공유한다(R2 edge 진단 AUC 0.49–0.54와 일치).
- **Feature unary 항:** recall을 깎는다. GT seed를 줘도 G보다 낮으므로 seed 오류 때문이 아니다.
- **Feature 내용 자체는 진짜다.** Shuffle은 훨씬 나쁘다. 그러나 noun seed와 depth surface 위에 추가 정보를 주지 못한다.
- **A3 대 A4:** Region growing이 연속 energy보다 낮다(support negative unary의 효과).
- **남은 seed identity 비용:** 약 0.025 IoU(oracle 대비).
- **Sensitivity:** σ_f·λ는 사전 등록 조건("FG arm이 통과할 때만")에 따라 실행하지 않았다. κ_proto→0이면 FGu는 G와 같아지므로, unary 강도 조절로 기대할 수 있는 상한은 G 수준이다.

### R8 — dump의 pixel gradient (224×224) 진단

- 대상: collector가 저장한 `gripper/arm_motion_pixel_saliency`(Σ_rgb |∂y/∂pixel × pixel|)와 ‖∂y/∂pixel‖.
- 비교: noun N, action A. Target GT는 replay class map을 policy pixel 중심에서 샘플했다(평가 전용).
- 비용: 180 frame, CPU worker 4개로 1분 미만.

| Map | 해상도 | 면적 일치 precision | argmax가 target | target 질량 | 상위 pixel 중 robot | AUC |
|---|---|---:|---:|---:|---:|---:|
| noun N | 16×16 | 0.495 | 0.500 | 0.297 | 0.085 | 0.972 |
| gripper grad×input | 224×224 | 0.069 | 0.050 | 0.041 | 0.269 | 0.823 |
| gripper ‖grad‖ | 224×224 | 0.052 | 0.044 | 0.031 | 0.189 | 0.776 |
| arm grad×input | 224×224 | 0.024 | 0.000 | 0.026 | 0.265 | 0.797 |
| arm ‖grad‖ | 224×224 | 0.017 | 0.000 | 0.021 | 0.200 | 0.743 |
| action A L8H2 | 16×16 | 0.037 | 0.022 | 0.043 | 0.032 | 0.799 |

**판정: 음성(진단만 수행, 방법 수정 없음).**

- Gradient는 해상도가 높지만 target을 가리키는 정도가 noun attention의 1/7–1/20이다.
- 상위 pixel의 19–27%가 robot이다.
- Grasp·carry(M3–M4)의 gripper gradient만 precision 0.18–0.23으로 다소 높다.
- "행동에 영향을 주는 pixel"과 "명령한 물체"는 다르다. Seed/identity 신호로 쓸 근거(가이드 §15.3)가 없어 graph round로 진행하지 않았다.

**Figure:**

- `ifg-r7-feature-revalidation.png` (+json): +top1 arm 전체의 paired IoU/recall 차이와 제약선.
- `ifg-r8-gradient.png` (+json): 지표 막대와 실제 frame(T53 ep1839 f023, 오른손 camera)의 RGB / GT / noun / gripper·arm gradient.

**재현:**

- `bash $RUN/rounds/R7/commands.txt`
- `PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_round7_summary --run $RUN`
- `bash $RUN/rounds/R8/commands_diag.txt`
- `PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_r7r8_figures --run $RUN --out AG3S/ag3s/docs/figures/ifg`

**새 코드:** `ifg_round7.py`, `ifg_round7_summary.py`, `ifg_grad_diag.py`, `ifg_r7r8_figures.py`. Registry: R7·R8 기록.

## 2026-10-15 22:40 KST — Feature가 추가 가치를 내지 못한 원인 (측정)

`ifg_feature_rootcause.py`, R7 기반(render mask, non-support seed, seed N), 180 frame. `$RUN/rounds/R9_rootcause/rootcause_summary.json`. GT는 진단에만 썼다.

| 가설 | 측정 | prefix_L8 | projector | 해석 |
|---|---|---:|---:|---|
| H1 해상도 | target node 중 target 비율 <0.5인 patch(혼합 patch)에 속한 비율 | 0.424 | (동일) | target 표면의 42%가 다른 물체와 섞인 patch에 있다 |
| H1 해상도 | target 경계 edge 양쪽 node의 주 patch가 같은 비율 (경계 edge가 있는 100 frame) | 0.735 | (동일) | 경계의 3/4은 양쪽 feature가 같다 → feature edge로 자를 수 없다(R7 FGe=G) |
| H3 recall 손실 | prototype 점수 <0.5인 target node 비율 | 0.188 | 0.198 | target의 약 1/5이 "target 같지 않다"는 unary를 받는다 |
| H3 recall 손실 | 그중 혼합 patch에 있는 비율 (점수 ≥0.5인 target node는) | 0.798 (0.371) | 0.830 (0.362) | 낮은 점수는 대부분 혼합 patch에서 온다 → feature unary가 recall을 깎는 기전 |
| H2 중복 | 예측 region이 2개 이상인 frame(48)에서 target region을 맞힌 수: seed 질량 / prototype | 42 / 34 | 42 / 32 | identity 선택은 seed가 더 정확하다 |
| H2 중복 | seed가 틀리고 prototype이 맞힘 / seed가 맞고 prototype이 틀림 | 1 / 9 | 0 / 10 | feature가 seed 오류를 구하는 경우는 거의 없고 해치는 경우가 많다 |
| H4 seed 의존 | node 단위 AUC(prototype, target): seed가 target 위(149 frame) / seed가 target 밖(9 frame) | 0.934 / 0.779 | 0.937 / 0.743 | prototype은 seed에서 만들어지므로 seed가 틀리면 함께 틀린다 |

**원인 결론:**

1. **경계에서는 쓸 수 없다(해상도).** VLA feature는 14×14 pixel patch 하나에 값 하나다. Target 경계 edge의 73%는 양쪽이 같은 patch여서 feature 차이가 0이다. 경계는 depth가 이미 나누고 있다.
2. **Unary로 쓰면 해롭다(혼합 patch).** Target 표면의 42%가 혼합 patch에 있고, 이 node들이 낮은 prototype 점수를 받아 target에서 빠진다. 이것이 FGu/FG의 recall −0.07 ~ −0.11의 기전이다.
3. **Identity 정보는 중복이고 덜 정확하다.** Prototype은 noun seed의 feature 평균이라 seed가 가리킨 물체를 다시 가리킨다. Seed가 맞을 때는 새 정보가 없고, 틀릴 때는 함께 틀린다(AUC 0.93→0.74–0.78). Region 선택에서도 seed 질량이 prototype보다 정확하다(42 대 32–34).

## 같은 시각 — Attention + 3D(R6 후보)와 기존 serving 방식(B0)의 차이

| 단계 | 기존 serving grounding (T54 N, c1a) | Attention + 3D (R6 후보, top1-only-ns+rm) |
|---|---|---|
| Robot 제거 | FK sphere/capsule **부피** 안의 점 제거 (손가락 사이 과일도 제거) | robot mesh를 FK로 렌더한 depth와 **표면 일치**(≤10 mm)할 때만 제거 |
| 3D 점 | 1 cm fusion voxel, 점마다 patch attention 값을 올림(lifting) | 5 mm voxel node, patch evidence를 pixel 비율로 나눠 질량 보존 |
| Seed | lifted attention 상위 5% 점(support 점은 connectivity에서 제외) | noun map 상위 5 patch 중 비robot·**비support** pixel ≥60인 patch |
| 연결 | **3D 반경** 3 cm 성장 + DBSCAN(eps 3 cm): 3 cm 안이면 다른 물체도 연결 | **image 상 인접 pixel이면서 depth 차 ≤10 mm**일 때만 연결: 3D로 가까워도 depth가 끊기면 연결하지 않음 |
| Target 결정 | cluster 점수(attention)가 가장 높은 cluster 하나, attention split 등 규칙 | 연속 energy(seed=1, support=0, 표면 smoothness)를 푼 뒤 seed 질량 최대 region 하나 |
| 출력 | cluster 점 + bounding sphere(ball) | region의 5 mm node(ball 미생성) |
| 시간 | TargetConfirm history(이번 비교는 `current`라 꺼짐) | 단일 frame |

성능 비교(dev):

- 같은 serving filter 조건(R5): 0.817 대 0.785, +0.032 [−0.008, +0.070].
- Render mask 조건(R6): 0.893. 다만 B0는 아직 sphere filter다.
- → R9에서 B0에도 같은 render mask를 넣어 filter 효과를 분리 중(background 실행).

## 2026-10-15 22:46 KST — R9 robot filter를 맞춘 B0 비교 (교란 제거)

**동기:** R6 후보 0.893 대 B0 0.785의 차이에는 robot filter 변경(render mask)이 섞여 있었다. B0(T54 c1a, `current`)에도 같은 render mask를 적용해 다시 실행했다. 방법은 masked pixel의 depth를 0으로 두고 serving sphere self-filter를 끄는 것이다(`ifg_baseline.py --render-mask`). 180 frame, worker 4.

| Arm | IoU | Precision | Recall |
|---|---:|---:|---:|
| B0 region, sphere filter | 0.785 | 0.896 | 0.828 |
| B0 region + render mask | 0.837 | 0.865 | 0.910 |
| B0 ball + render mask | 0.778 | 0.809 | 0.903 |
| Attention + 3D, sphere filter (R5) | 0.817 | 0.947 | 0.831 |
| Attention + 3D + render mask (R6) | 0.893 | 0.938 | 0.915 |

- **Render mask의 B0 효과:** IoU +0.052 [+0.028, +0.081], 28/3 episode. Recall +0.082, precision −0.031.
- **같은 mask에서 방법 효과(Attention + 3D − B0 region):** IoU **+0.057 [+0.006, +0.103]**, 27/5 episode. Precision +0.073 [+0.017, +0.121], recall +0.005 [−0.041, +0.047]. 주로 다른 물체를 덜 포함해서 얻는 이득이다.
- **Moment별(같은 mask):**
  - 앞섬: F2 0.910/0.758, M3 0.935/0.823, M6 0.911/0.758, M7 0.882/0.617
  - 뒤짐: F3 0.745/0.920
- **주의:**
  - CI 하한이 0에 가깝다.
  - Dev에서 설계하고 같은 dev에서 재평가했으므로 독립적인 유의성 증거가 아니다(가이드 §15.6).
  - Held-out은 대기 중이다(미사용 episode 없음, collector 180 frame에서 종료).

Figure: `AG3S/ag3s/docs/figures/ifg/ifg-r9-filter-matched.png` (+json).
재현: `bash $RUN/rounds/R9/commands.txt`, `PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_r9_figure --run $RUN --out AG3S/ag3s/docs/figures/ifg`.

## 2026-10-15 ~23:07 KST — Feature 재활용 3종 (사용자 요청): R10/R10b 기억, R11/R11b 경보, R12 단계·파지

### R10 시간축 identity 기억 (`ifg_memory.py`)

- **방법:** episode의 첫 frame에서 R6가 고른 영역의 feature를 기억으로 둔다(chain 변형 포함). 이후 frame에서 물체 크기 영역 중 기억과 더 닮은 영역이 있으면 δ 이상일 때 전환한다.
- **결과(raw cosine):**
  - δ ≥ 0.05: 전환 0회
  - δ = 0: prefix_L8 기준 27회 전환(6 개선 / 21 악화), IoU −0.055
  - 첫 frame 기억 자체는 146/148 frame에서 맞았다.
- **진단(`rounds/R10/failed_frame_diag.json`):**
  - R6 실패 7 frame 중 5개에서 기억과 가장 닮은 영역이 실제 target이었다.
  - 그러나 cos 차이가 0.002–0.015에 불과했다. 모든 영역의 cos가 0.96–0.98에 몰리는 anisotropy 때문이다.
  - 나머지 2개(M7)는 target이 crate 영역과 합쳐져 있었다.
- **R10b(centering, frame 평균 patch feature 제거):**
  - 최선은 Mem-first:prefix_L8:δ0.05로 IoU 0.912다.
  - R6 대비 +0.019 [−0.020, +0.060], recall +0.018, 7회 전환(5 개선/2 악화).
  - **판정: 미확정(채택 안 함)**. 사전 기준 +0.02 미달이고 CI가 0을 포함한다.
  - Projector 기억은 centering 후에도 해롭다.

### R11 신뢰도 경보 (`ifg_confidence.py`)

- **정의:** "잘못 고름" = R6 IoU < 0.1(6 frame). 같은 147 frame에서 비교했다.

| 점수 | AUROC (raw) | AUROC (centered) | 10% 보류 시(centered): 탐지 / FP 제거 / TP 손실 |
|---|---:|---:|---|
| seed 질량 비율(feature 없음) | 0.900 | 0.914* | 4/6 · 48% · 9%* |
| 같은 frame seed prototype(prefix_L8) | 0.879 | 0.928* | 4/6 · 48% · 9%* |
| 첫 frame 기억(prefix_L8) | 0.954 | 0.992 | 6/6 · 82% · 9% |
| chain 기억(prefix_L8) | 0.982 | 0.998 | 6/6 · 82% · 6% |

(*는 179 frame 기준 값이다.)

- Raw chain 대 feature 없는 점수의 AUROC 차이는 +0.096 [0.000, +0.266]이다.
- **판정: 긍정(dev, 작은 표본).**

### R12 작업 단계·파지 판단 (`ifg_phase.py`, supervised ablation, LOEO)

| 모델 | 단계 정확도 | 파지 정확도 |
|---|---:|---:|
| HSR argmax / gripper 규칙 | 0.783 / — | — / 0.828 |
| LR[hsr] | 0.856 | — |
| LR[hsr+grip] | 0.878 | 0.894 |
| LR[grip+geo] | 0.733 | 0.922 |
| LR[hsr+grip+geo] | **0.939** | 0.933 |
| LR[hsr+grip+geo+feat] | 0.933 | 0.939 |

- 3D 관계의 추가 효과(단계): +0.067 [+0.015, +0.118].
- vs HSR argmax: +0.149 [+0.095, +0.196].
- Feature identity의 추가 효과: 단계 −0.010, 파지 +0.004. 이 dump에는 엉뚱한 물체를 쥔 사례가 없다.

**Figure:** `AG3S/ag3s/docs/figures/ifg/ifg-r10-r12-feature-reuse.png` (+json).

**전체 정리:** [SUMMARY_METHODS_RESULTS_KO.md](SUMMARY_METHODS_RESULTS_KO.md)(이전 방식 / 현재 방식 / 개선·비슷·악화 분류 / 원인 / 한계).

## 2026-10-15 23:39 KST — 방법 잠금, held-out 수집 요청서, 사용자 판정

**사용자 판정:**

| 항목 | 판정 |
|---|---|
| Target을 가리는 장애물 시나리오 | **테스트용.** 결과는 별도 표로만 보고하고 주 결론·채택 판단에 섞지 않는다. 비중은 사용자가 판단한다 |
| 실패 기록 | 실패·개선 실패도 정리 문서에 모두 포함한다 |

**작업 내용:**

- **방법 잠금(`$RUN/LOCK_heldout.json`):** held-out 수집 전에 고정했다.
  - 후보 extent 설정
  - 기억 전환(δ = 0.05, centered, 보조 arm)
  - 경보 threshold **0.5904**(dev chain 기억 점수의 10% 분위수)
  - 단계·파지 LR weight(dev 전체로 학습, 표준화 계수 포함)
  - `ifg_*.py` sha256
  - 가림 시나리오 별도 보고 원칙
- **수집 요청서:** [HELDOUT_COLLECTION_REQUEST_KO.md](../../../archive/reviews/vla_dump_20261015/HELDOUT_COLLECTION_REQUEST_KO.md). 이전 사용 episode 목록은 `$RUN/used_episodes_prior.json`(T53, T54c, T61, CL4, T58, T47 dev).
  - 주 held-out: clear / adjacent / place 장애물
  - 테스트용 가림: 20–60%, 60% 이상
  - 모든 step, depth 기록, 최소 dump profile, instruction swap 재추론
- **정리 문서 보강:** [SUMMARY_METHODS_RESULTS_KO.md](SUMMARY_METHODS_RESULTS_KO.md)
  - §3A 실패·개선 실패 기록(모든 시도의 이유·결과·원인·교훈)
  - §3B 가림 시나리오 계획
  - R10b 주의점(결과를 본 뒤 넣은 수정, δ 척도 미조정)

## 2026-10-15 23:49 KST — 요청서 v2 (사용자 답변 3건)

| 질문 | 사용자 답변 | 반영 |
|---|---|---|
| 학습 split 밖 장면 | 필요 없음. 성공 episode 위주로 | in-distribution 장면 허용, 주 평가 = simulator 판정 성공 rollout |
| 저장 범위 | 우리가 쓰는 데이터 위주 | 최소 4항목(inputs, derived 3 key, prefix L8 image row, done.json), step당 약 5 MB |
| 가림 장애물 | 잡지 못하는 위치 금지(feasible, reachable), head camera에서 target이 보여야 함, 가림 0 / 1/3 / 1/2 테스트 | T42-2 탐색 틀을 반대 목적으로 재사용: 기준 궤적 clearance ≥ 0.03 m, grasp 간격 ≥ 0.04 m, 가림 비율 측정 후 장애물 rollout 성공 확인 |

**정직성 원칙(분석 session 판단, 요청서 §2에 명시):**

- 성공 기준은 우리 결과를 보기 전에 정해지는 simulator 판정만 쓴다.
- 이미 분석한 episode는 제외한다.
- 실패 episode는 보관하고 부록에 보고한다.
- 분석 결과를 보고 episode를 고르지 않는다.

`LOCK_heldout.json`에도 같은 내용을 반영했다.

## 2026-10-15 23:58 KST — 최종 보고서

- **[FINAL_REPORT_KO.md](FINAL_REPORT_KO.md):** 가이드 §15.8 순서(최초 실패 → 개선 이유 → 바꾼 것 → 전후 수치 → 남은 한계). 사본은 `$RUN/FINAL_REPORT_KO.md`.
- **`$RUN/manifest.json`:** 입력, `ifg_*.py` sha256, 산출물 목록. 프로토콜 필수 산출물 중 빠져 있던 항목이다.
- **Figure `ifg-final-overview.png`(+json):** B0 3종, R0 G/FG, R3/R5/R6 후보, R7 FG, oracle seed의 episode-mean IoU와 95% CI.

## 2026-10-15 00:29 KST — 커밋, 실험별 figure, 쉬운 설명판 보고서, 방법론 변화 문서 (사용자 요청 4건)

- **커밋:** `4f36ff51`(ifg 코드·figure·문서 4종). 사용자가 추가한 `Novelty_contribution.pdf`를 읽고 방법론 문서에 반영했다.
- **실험별 figure:** `AG3S/ag3s/docs/figures/ifg/exp/`, `ifg_experiment_figures.py`. 실제 장면 투영 9장(+json). 예시 frame은 "효과가 가장 큰 frame" 규칙으로 자동 선택하고 sidecar에 규칙을 기록했다.
  - S0 정합, R1–R3 gate 단위, R4 질량 편향, R5 탁자 시작점, R7 feature 점수 손실, R9 기존 방법 비교, R10 centering, R11 경보, R12 단계 신호
- **검토 중 고친 것:**
  - R1–R3 / R9 / R11 제목이 그림 내용을 과장하던 부분을 실제 수치로 바꿨다. 예: R11 = 경보 15회, 그중 잘못된 선택 6개, 오경보 9회.
  - R5는 세 camera를 모두 표시하도록 바꿨다.
- **R12 입력 정의 정정:**
  - "탁자 위 높이"는 실제로 base 좌표계 높이였다. support에 바닥(z ≈ 0.003)과 탁자(z ≈ 0.82)가 함께 있고, 주변 중앙값이 바닥에서 나왔다.
  - 모든 장면의 탁자 높이가 같아 결과에는 영향이 없다. 잠근 weight는 유지하고 이름·문서만 고쳤다.
  - `LOCK_heldout.json`에 amendment를 기록했다(`ifg_phase.py` docstring만 변경).
- **문서:**
  - [FINAL_REPORT_KO.md](FINAL_REPORT_KO.md)를 쉬운 설명판으로 다시 썼다(용어표, 실험별 질문→한 일→결과→그림, 약어 미사용).
  - [METHODOLOGY_EVOLUTION_KO.md](METHODOLOGY_EVOLUTION_KO.md): 이전 → 원래 계획 → 현재 후보 → 계획, novelty 대응, **채택 후보 A–D(사용자 판정 대기)**.
  - 도식: `ifg-method-evolution.png`.

## 2026-10-15 00:31 KST — 사용자 판정: 채택 방법론 B

| 항목 | 판정 |
|---|---|
| 채택 방법론 | **B = A + 시간축 identity 전환.** 구성: Attention + 3D 범위(R6), feature 기억 영역 전환(R10b, centered, δ = 0.05), feature 기억 안전 경보(R11), 3D 단계·잡기 판단(R12) |

**반영:**

- `LOCK_heldout.json`에 `adopted_method`를 기록했다.
- Held-out arm: B, A(전환 없음, 전환 효과 분리용), B0 + render mask, B0, G:N.
- 영역 전환이 dev에서 미확정(+0.019 [−0.020, +0.060])이라는 점과 δ 척도를 조정하지 않았다는 점을 문서에 명시했다.
- 수정한 문서: METHODOLOGY_EVOLUTION §5, FINAL_REPORT §4.

## 2026-10-16 03:10 KST — held-out 검증 · closed-loop 착수 (사용자 판정 3건)

**사용자 판정:**

| 질문 | 판정 |
|---|---|
| GPU 사용 | collector 수집(+ 내부값 dump)이 끝날 때까지 대기. 그동안 CPU 작업만 한다 |
| closed-loop 범위 | 단계적. 1 단계 = attention + depth 표면 영역 + rendered-robot 표면 제거. 2 단계 = feature 기억 전환 · 경보 |
| 비교 설정 | held-out 성공 seed 짝 비교: C1a(현재 serving) vs 제안. 같은 cuRobo ESDF · TrajOpt · flag. 주 지표 = 비허용 접촉 없는 작업 성공률, 정지 · 시간 함께 |

**collector 현황:**

- run: `20261016T000420+0900_30955b6ca8ef_2498252a873f/heldout`
- 성공 주 평가 rollout: 장애물 없음 6/12, 이웃 장애물 6/12, 바구니 옆 5/8
- 가림 시나리오 미수집. 내부값 dump(`derived.npz` · `prefix_L8.npz`) 0개(rollout 후 GPU 단계)
- 장애물 run은 `--safe-phase approach` 고정으로 돌았다. serving의 단계 판단은 subtask gate(kv_L4 probe, pick/place/home) + 잡기 상태(held 처리) 그대로다

**한 일:**

- **held-out 준비** (`ifg_heldout.py`, 새 파일; 잠근 코드 아님): finished primary run의 모든 step에서 다음을 만든다.
  - 알고리즘 입력: **기록된 rollout depth**(dev는 replay depth였다)와 K/T
  - 평가용 정답: snapshot scene + 장애물 profile을 재적용한 replay 라벨(장애물은 `other_object`)
  - robot mask: FK 구 mask, rendered-robot mask
  - smoke 1 run(75 frame) 결과:
    - 기록 depth와 replay depth의 차: 중앙값 0.25 mm(uint16 mm 양자화), ≥ 99.4%가 2 mm 이내
    - RGB MAD ≤ 0.3
    - wrist_cam_r T 차 ≤ 0.36 mm(움직이는 중) → 판정 기준을 1 mm로 둔다
  - 전체 실행: analysis run `20261016T030741+0900_f15e587290f6_19ea09c7a8b8/heldout_eval`, worker 2(collector 2개와 합쳐 pod 4 이내)
- **serving 1 단계 구현을 implementer에 위임:** `AG3S/ag3s/docs/handoff/IFG-CL1.task.md`
  - 기존 `--target-extent oracle-visible` 경로(카메라별 code 영상 → target-free 미세 계층)를 재사용한다
  - code 영상만 GT 대신 서버의 attention + depth graph(잠근 `top1-only-ns+rm:N`)로 만드는 `attention_surface` 모드를 추가한다
  - `--robot-mask render`도 추가한다
  - CPU 구현과 unit test만 한다(GPU 금지)

## 2026-10-16 04:40 KST — serving 1 단계 구현 완료 · closed-loop 설정 판정

**held-out 준비 완료:**

- 1,650 frame(run 22). 정합 검사 통과 1,635개(99.1%)
- 제외 15개: 손목 카메라에서 기록 depth와 replay가 어긋남(빠른 동작 중). 평가 분모에서 제외하고 사유를 보고한다

**구현 (implementer, `IFG-CL1.impl.md`):**

- `runtime/attention_surface.py`
  - 잠근 `build_geometry` · `eligible` · `g_and_top1`을 import한다
  - 카메라별 code 영상(1 target, 2 static, 0 robot/무효)을 만들어 기존 oracle_visible 경로(memory → target-free 미세 계층 분할)에 넣는다
- `runtime/robot_render_mask.py`: 로봇 mesh만 요청 관절로 렌더하고, 기록 depth와 10 mm 이내로 맞는 pixel을 robot으로 본다
- flag: `--target-extent attention-surface`(noun_target:L10H7 필수), `--robot-mask render`. 기본값은 off이고, off일 때 동작은 비트 단위로 같다
- **parity:** dev 180 frame에서 서버 선택과 offline LOCK arm의 IoU 1.0(180/180). render mask와 offline R6의 pixel 일치율은 최소 0.99974
- **lead 재실행:** `.venv-ag3s`에서 새 test + oracle test + attention source test → 119 passed, 3 skipped
- **추가 지연:** 요청당 약 1.0–1.15 s(median, CPU). 대부분 잠근 graph 생성(build_geometry 678 ms)이다. 잠근 방법을 지키려고 줄이지 않았다
- **사고 기록:** implementer가 `.venv-openpi-live`로 기존 test를 돌리다가 GPU test 1개가 lock 없이 GPU에서 수 초 돌았다. collector process는 건드리지 않았다
- **기존 회귀 1건 실패:** `test_t43t_replay::test_off_is_bit_identical_to_the_pre_fix_commit`. HEAD의 clean archive에서도 똑같이 실패하므로 이번 변경과는 무관하다

**사용자 판정:**

| 항목 | 판정 |
|---|---|
| 비교군 | C1aN(C1a + noun_target:L10H7) vs SURF(C1aN + attention-surface + render mask). 기존 C1a(L8H2)는 collector 기록을 참고로만 쓴다 |
| TSDF robot 제거 | 구 ∪ render(현재 구현) |

**실행:**

- verifier에 `IFG-CL1-run.task.md`를 위임했다: collector 검증 틀 복사, 임시 index snapshot(commit 없음), GPU lock을 blocking으로 대기
- smoke gate: SURF에서 attention_surface가 실제로 적용된 frame 비율 ≥ 30%
- 실시간성 측정: 단계별 ms, 왕복, 533 ms 예산 초과 비율

## 2026-10-16 05:10 KST — closed-loop launcher 대기 시작

- **run:** `20261016T042508+0900_f15e587290f6_86dde89fb754/verify/IFG-CL1`. launcher(`cl1_launch.sh`)가 GPU lock을 blocking으로 기다린다
- **snapshot:** 임시 index tree `f0d8046b`(commit 없음). 실제 index와 HEAD는 바뀌지 않았다
- **사전 검사:** flag · import · address · legacy baseline 모두 일치
- **seed:** lock 획득 직후 collector의 성공 seed 전부. 지금 기준 dry-run은 25개
- **lead 수정:** smoke gate G2의 분모를 "target-free 층이 만들어진 chunk"로 바꿨다. 잡은 뒤 chunk는 원래 층이 없어서, 전체 chunk를 분모로 두면 정상 작동해도 멈춘다. 사유는 `p/LEAD_AMENDMENTS.md`에 남겼다
- **주 지표 정의 수락:** 성공 ∧ 손 밖 robot–target 접촉 0 ∧ 장애물 접촉 0. 이웃 과일 접촉은 따로 보고한다
- **실시간성 사전 관찰:**
  - collector의 C1a run 1개에서 client 왕복 median이 1038 ms였다
  - 74/74 요청이 533 ms 예산을 넘었다. 즉 **현재 serving도 실시간 예산을 넘는다**
  - SURF는 여기에 약 1.1 s가 더해질 것으로 본다

## 2026-10-16 21:35 KST — collector 종료 · closed-loop 재시작 · held-out 잠근 평가 착수

**collector 종료:**

- 성공 rollout: clear 13, adjacent 12, place 9(+ 실패 run). DONE
- dump: 3,225 step(43 run × 75), serving map parity 75/75

**closed-loop launcher 실패 2건과 수정** (`verify/IFG-CL1/p/LEAD_AMENDMENTS.md`):

1. 첫 lock 획득(10-15 21:29 UTC) 때 두 server가 argparse에서 종료됐다
   - 원인: `common/run_output`은 run 안에 run 밖을 가리키는 symlink가 있으면 거부한다. verifier가 사전 시험용으로 만든 `pre/atest`에 이런 symlink가 5개 있었다
   - 조치: symlink만 삭제(대상 파일은 그대로)
2. 재시작 후 C1aN server가 정상으로 떴는데도 거부됐다
   - 원인: log 검사 `grep "IFG-CL1"`이 기록 경로 문자열에 걸렸다(오탐)
   - 조치: 검사를 경고 접두어 `!!! AG3S IFG-CL1`로 좁혔다. 파일은 mv로 원자 교체했다
   - SURF smoke는 계속 돈다. launcher가 멈추면 watcher(`lead_relaunch_once.sh`)가 C1aN만의 사유일 때 한 번 재시작한다

- **seed:** 34개(clear 13, adjacent 12, place 9). 그중 primary 32개

**held-out 잠근 평가** (`ifg_heldout_eval.py`, 어댑터):

- 잠근 파일은 수정하지 않았다
- 어댑터가 하는 일
  - 같은 process 안에서 `ifg_reader.load_frame`을 held-out dump reader로 바꿔 끼운다
  - `ifg_memory.FEATS`를 prefix_L8만으로 줄인다. dump에 projector가 없고, 잠근 arm은 prefix_L8만 쓴다
- smoke(HO_adjacent_ep64_f010): top1-only-ns+rm:N IoU 0.995. memory 점수가 정상으로 계산된다
- 실행 순서: 준비(run 43개)가 끝나면 ns(R5/R6) → memory(R10b/R11b)를 자동 실행한다. worker 2개

## 2026-10-16 22:50 KST — held-out 잠근 평가 결과 (offline)

- **데이터:** 성공 rollout 34 episode, 유효 frame 2,523개(clear 13 · adjacent 12 · place 9). 실패 9 episode는 부록이다. S0 정합에서 29 frame을 제외했다
- **집계:** dev와 같은 규칙. frame → episode 평균 → episode 짝 bootstrap 2,000회(seed 20261015)
- **산출물:** `heldout_locked/summary.json`, figure `ifg-heldout-locked.png`(+json)

| arm | IoU (held-out) | dev 참고 |
|---|---:|---:|
| G:N (R0 기준, FK 구 mask) | 0.773 | 0.728 |
| A안, FK 구 mask (top1-only-ns:N) | 0.832 | 0.817 |
| **A안 (top1-only-ns+rm:N)** | **0.876** | 0.893 |
| **B안 (A + 기억 전환, 채택)** | **0.891** | 0.912(A + 0.019) |

| 짝 비교 (IoU) | 차이 [95% CI] | 좋아짐/나빠짐 |
|---|---|---|
| A − G:N | +0.103 [+0.083, +0.123] | 34/0 |
| A − A(FK mask): robot 표면 제거 효과 | +0.044 [+0.038, +0.050] | 34/0 |
| B − A: 기억 전환 효과 | +0.015 [−0.012, +0.039] | 14/6, 미확정(dev와 같다) |

- **단계별 IoU (A안):** 잡기 전 0.970, 손에 든 동안 0.848(FK mask일 때 0.640), 놓은 뒤 0.846
  - B안: adjacent +0.028, place +0.024, clear −0.005
- **신뢰도 경보 (잠근 threshold 0.5904):**
  - 잘못 고른 frame 100개 중 85개를 잡았다. 오경보는 153개다
  - 경보가 난 frame의 영역을 버리면 잘못 포함된 표면의 88%가 사라지고, target 표면은 6.7%를 잃는다
  - AUROC 0.925
- **아직 안 한 것:** B0(serving ball) offline 재현. 이 비교는 closed-loop가 직접 대신한다

## 2026-10-17 03:30 KST — closed-loop 결과 (C1aN vs SURF, held-out 34 seed 짝)

- **실행 정보**
  - run: `20261016T042508+0900_f15e587290f6_86dde89fb754/verify/IFG-CL1`
  - 결과 파일: `agg.json`, `timing.json`
  - rollout 68개 완료. 두 arm 모두 같은 cuRobo · TrajOpt · flag를 썼다
- **분석 중 수정 1건**
  - 접촉 replay가 처음에 실패했다(rc 123). 복사해 온 `hold_w.py`가 장애물 profile을 거부했기 때문이다
  - run의 `.start`에 적힌 장애물 설정을 그대로 넘기도록 고친 뒤 다시 돌렸다(`LEAD_AMENDMENTS.md`)

| 항목 | C1aN (대조군) | SURF (제안) |
|---|---:|---:|
| 작업 성공 | **34/34** | **28/34** |
| 주 지표 (성공 ∧ 비허용 접촉 0 ∧ 장애물 접촉 0) | 34/34 | 28/34 |
| 짝 비교 | both 28 · only C1aN 6 · only SURF 0 · McNemar p = 0.031 | |
| 실패 유형 | — | no_grasp 5, not_placed 1 (clear 4 · adjacent 1 · place 1) |
| HOLD chunk 합 | 50 | 19 |
| 완료 시간 median (둘 다 성공 28쌍) | 16.0 s | 14.7 s |
| 이웃 과일 접촉 run | 2 | 2 |
| 비허용 link–target / 장애물 접촉 | 0 | 0 |

- **SURF 적용률**
  - 층이 만들어진 chunk에서 거의 다 적용됐다: applied 908
  - 적용되지 않은 경우: identity_unresolved 43, no_region 8. 나머지 1,591은 층 자체가 없었다
- **실패 run의 공통점: ball 안의 seed를 비-target으로 남긴 수가 크다**
  - `n_kept_nontarget`가 C1aN에서는 수십~수백인데, SURF에서는 수만이다
  - 가설(검증 전): visible-surface extent는 보인 면과 그 바로 뒤 띠만 지운다. 그래서 target의 안 보인 쪽 TSDF seed가 손가락 앞에 장애물로 남아 잡기를 막는다
  - oracle_visible 경로 자체의 closed-loop는 T46e 이후 측정된 적이 없다
- **실시간성** (요청 2,516개씩, median / p95 ms)

| 항목 | C1aN | SURF |
|---|---:|---:|
| client 왕복 | 1,137 / 1,924 | 2,858 / 3,623 |
| server 총시간 | 796 | 2,587 |
| attention_surface | — | 1,663 (graph 828 · 영역 285 · 라벨 383) |
| render mask | — | 159 |

  - 두 arm 모두 533 ms 예산을 넘는다

## 2026-10-17 15:20 KST — SURF 실패 원인 진단 (정답 대조 + 진단 rollout)

**정답 대조** (`ifg_cl1_keptseeds.py`, CPU, 68 run): target-free 층이 "target 아님"으로 남긴 seed가 실제로는 정답 target(convex hull ±5 mm)에 속하는지 확인했다.

| 그룹 | 정답 target인데 남긴 seed (chunk당 median) | 지운 seed 중 정답 target 비율 | 다른 물체를 지운 chunk가 있는 run |
|---|---:|---:|---:|
| C1aN 성공 34 | 1 | 0.89 | 0 |
| SURF 성공 28 | 79 | 0.92 | 1 |
| SURF 실패 6 | 134 | 0.50 | 5 |

**진단 rollout** (실패 6 seed, GPU, 05:20–06:10 UTC):

- SURFR(SURF 재실행): **6/6 실패 재현.** 실패 유형도 같아 우연이 아니다(결정적)
- ORAC(같은 설정, target 영역만 GT visible segmentation): **2/6 성공**
  - 391, 1769 성공. 둘 다 SURF에서 "다른 물체를 지운" chunk가 많았던 run이다
  - 나머지는 no_grasp 2, slip_at_lift 2

**해석 (측정 기반, 원인은 2가지가 겹침):**

1. **"보인 표면만 지우는" 경로가 잡기를 어렵게 한다.** GT 영역(ORAC)으로도 6 중 4가 실패했다. 이 경로는 T46e 이후 closed-loop에서 측정된 적이 없었다
   - render mask 효과와는 아직 분리하지 못했다. ORAC와 SURF 모두 render mask를 썼기 때문이다
2. **우리 attention 영역의 identity 오류가 실패를 더한다.** GT로 바꾸면 2 run이 구제된다

**영상 경로:** `runs/{SURF,C1aN,ORAC,SURFR}/<arm>_ep<ep>_s<seed>/third_person_front.mp4`

## 2026-10-17 17:25 KST — 허들 1차 launcher 대기 · 하이브리드 구현 착수

**사용자 판정:**

- 허들은 현재 구현 그대로 1차, 하이브리드가 준비되면 2차를 따로 돌린다
- C1a(현재 serving)를 기준선 arm으로 넣는다

**허들 1차** (`IFG-HURDLE-run.task.md`):

- run: `20261017T171201+0900_f15e587290f6_137b5f75d9e3/verify/IFG-HURDLE`. launcher pid 1199646
- 장면: T43 H 세트 14개(7 episode × 2 seed, `t43p2_overbar`). config는 storage 사본이고 md5가 원본과 같다
- client argv: T43 V11AB와 같다(경로와 기록 옵션만 다름)
- arm: C1a / C1aN / SURF. snapshot은 IFG-CL1의 tree `f0d8046b`를 그대로 쓴다. 하이브리드 수정 중인 파일은 섞지 않았다
- 시작 조건: benchmark 작업 process가 10분 동안 없을 때만 GPU lock을 잡는다
- 실행 순서: smoke → phase A/B/C → CPU 분석. CT 규칙 등 T43 규칙을 그대로 쓴다
- 예상: 시작 후 GPU 약 3시간, 분석 약 40분. lead가 확인한 결과 run 밖을 가리키는 symlink는 없다

**하이브리드** (`IFG-CL2.task.md`, implementer, CPU):

- 지우는 범위: ball_owned 그대로(target 부피 전체)
- attention-surface의 역할: 선택 영역 밖의 비지지면 표본에 더 가까운 공 안 seed는 지우지 않는다(이웃 보호)
- identity 안전장치를 둔다
- 수락 조건: offline 재생에서 정답 대조로 확인한다

## 2026-10-17 19:00 KST — 정정: 15:20 절의 "SURF가 target 뒷면을 남긴다(134 vs 1)" 진단은 성립하지 않는다 · 하이브리드 결과

**정정 (implementer 재생으로 확인):**

- 15:20 절의 대조(`ifg_cl1_keptseeds.py`)에는 두 가지 오류가 있었다
  - 정답 범위를 target convex hull ±5 mm로 잡아서 **과일 바로 아래 탁자 seed**를 target으로 셌다
  - arm마다 기록되는 seed 집합이 다르다. SURF는 oracle 후보 seed를 쓰고, C1aN은 ball 안 seed만 기록한다. 그런데 이 둘을 그대로 비교했다
- 같은 TSDF · 같은 seed 집합으로 다시 계산(지지면 위, chunk 평균): SURF가 남긴 정답 target seed는 35.5, 같은 TSDF에 C1aN 규칙을 적용하면 113.6이다
- 즉 **SURF가 target을 더 많이 남긴 것이 아니다.** "보인 표면만 지우는 경로가 잡기를 막는다"는 원인 1은 근거를 잃었다

**남는 측정 사실:**

1. SURF 실패는 재현된다(SURFR 6/6)
2. GT visible segmentation(ORAC)으로는 2/6 성공이다
3. **SURF 실패 6개 중 5개는 chunk당 crate seed 159–431개를 지웠다. 성공 run은 0개다.** identity 오류(영역이 crate 등)와 함께 나타나는 신호이고, 상관이지 원인은 아니다
- 실패 원인은 **미확정**이다. render mask 효과도 아직 분리하지 않았다

**하이브리드 (`IFG-CL2.impl.md`, `--target-extent attention-surface-hybrid`):**

- 구현과 test 완료(39 pass, 관련 회귀 2,045 pass). torch builder test는 GPU에서 확인해야 한다
- offline 재생: 68 run, pre-grasp 1,430 chunk
  - 이웃 seed 제거: C1aN 규칙 대비 −62%(35.9 → 13.7). **수락 조건 충족**
  - 정답 target seed 보존: 55.5 → 65.9로 오히려 늘었다(ball 안 median 0 → 35). **수락 조건 미충족**
  - 원인: 선택 영역이 과일 일부만 덮어, 같은 과일의 선택되지 않은 표면이 "다른 물체" 근거가 된다(보호된 target seed의 71%)
- implementer가 짚은 수정 후보: manipulated cloud 근처의 "other" node를 빼는 것. held-out 기록으로 고르면 test 데이터에 맞추는 셈이라 구현하지 않았다
- 추가 시간: SURF의 선택 비용 + 약 10 ms. oracle memory는 쓰지 않는다

## 2026-10-17 21:00 KST — 하이브리드 보완 규칙(CL2b) 결과: 1.0에서 C1aN과 거의 같아진다

- **규칙:** manipulated 기하 점에서 1 × fusion voxel(10 mm) 안의 "other" node는 보호 근거에서 뺀다. 값은 고정이고, 0과 1.0 두 행만 재생했다
- **구현:** flag `--hybrid-other-exclude-voxels`(기본 1.0, 0 = CL2와 같은 동작). test 47 pass, 관련 회귀 2,053 pass
- **재생:** `20261017T202707+0900_f15e587290f6_7024ee2643b6/cl2b_replay`. 같은 68 run, 1,430 chunk. 값은 chunk 평균이고, 실패 / 성공 순서다

| 변형 | 정답 target 남김 (지지면 위) | 이웃 seed 제거 |
|---|---:|---:|
| C1aN 규칙 | 113.6 / 55.5 | 36.7 / 35.9 |
| hybrid (보완 없음) | 121.1 / 65.9 | 14.0 / 13.7 |
| hybrid + 보완 1.0 | 114.6 / 56.3 | 36.3 / 34.8 |

- **결과:** 보호된 seed가 93% 줄어 hybrid가 C1aN과 거의 같아졌다
- **원인 (scratch 진단):** manipulated 기하 점의 35%가 이웃 과일 위에 있다. 그래서 이웃 보호 근거까지 같이 지워진다. identity guard는 점의 과반이 target이라 통과한다
- **결론:** 현재 근거(serving manipulated 기하)로는 "target 과소 제거"와 "이웃 보호"를 동시에 풀 수 없다. 허들 2차 구성은 render 진단(④)과 함께 사용자 판정을 받는다

## 2026-10-17 23:40 KST — 허들 1차 결과 (T43 H 14 run × C1a / C1aN / SURF, 현재 구현 그대로)

- **실행:** run `20261017T171201+0900_f15e587290f6_137b5f75d9e3/verify/IFG-HURDLE`
  - rollout 11:35–14:13 UTC. 42 run 모두 완료됐고 replay 교차 확인에서 불일치는 0이다
  - 08:40–11:30에는 시작하지 못했다. bench_watch가 benchmark 세션의 Claude 대기 셸을 GPU 작업으로 잘못 셌기 때문이다. lead가 수정했다(`p/LEAD_AMENDMENTS.md`)

| arm | 기존 성공 | 접촉 포함 성공 (CT) | 잡기 | 허들 접촉 run | obstacle stop | HOLD chunk |
|---|---:|---:|---:|---:|---:|---:|
| C1a (현재 serving) | 11 | **11** | 13 | 2 | 2 | 86 |
| C1aN | 11 | **11** | 13 | 2 | 2 | 82 |
| SURF | 8 | **7** | 9 | 3 | 3 | 11 |
| 참고: T43 V11AB (코드 · flag 다름) | 12 | 12 | — | — | — | — |

- **짝 비교 (CT)**
  - C1aN vs SURF: C1aN만 성공 6, SURF만 성공 2, p = 0.22
  - C1a vs C1aN: 완전히 같은 결과
- **SURF 실패 유형:** no_grasp 5(C1aN은 성공), slip_at_lift 1. SURF가 C1a/C1aN보다 나았던 run은 1807 s18073 하나다(C1a/C1aN은 운반 중 미끄러짐)
- **비허용 link–target 접촉:** SURF 1 run, C1a/C1aN 0
- **실시간성** (client 왕복 median / p95 ms)
  - C1a 1,256 / 2,034
  - C1aN 1,177 / 2,152
  - SURF 3,331 / 4,979
- **영상 길이:** 모두 40초다. `--max-steps 600` @ 15 Hz 고정 길이이고, 성공해도 조기 종료하지 않는다. 판정 규칙이 놓은 뒤의 분리 유지까지 보기 때문이다

## 2026-10-18 13:10 KST — render mask 분리 진단(diag2) 결과 · GPU 반납 알림 지연

- **실행:** SURFS(우리 영역 + 구 mask)와 ORACS(GT 영역 + 구 mask)를 실패 6 seed에 돌렸다. 10-17 14:34–15:22 UTC
- **알림 지연:** lead가 benchmark 세션에 GPU 반납을 알리지 못했다. 이전 소켓 주소가 사라져 있었다. 10-18 04:10 UTC에 재알림했다

| seed | C1aN | SURF (우리+render) | SURFR (재실행) | ORAC (GT+render) | SURFS (우리+구) | ORACS (GT+구) |
|---|---|---|---|---|---|---|
| 391 | ok | no_grasp | no_grasp | ok | no_grasp | ok |
| 764 | ok(regrasp) | no_grasp | no_grasp | no_grasp | no_grasp | no_grasp |
| 1769 | ok(slip_at_lift) | no_grasp | no_grasp | ok | regrasp(실패) | ok |
| 623 | ok | no_grasp | no_grasp | no_grasp | no_grasp | no_grasp |
| 1893 | ok | no_grasp | no_grasp | slip_at_lift | **ok** | slip_at_lift |
| 485 | ok | not_placed | not_placed | slip_at_lift | **ok** | slip_at_lift |

**해석 (측정 기반):**

- **GT 영역은 robot mask와 무관하게 2/6이다.** ORAC와 ORACS가 같은 seed(391, 1769)에서 성공했다
  - target 영역을 정답으로 줘도 oracle_visible 경로로는 6 중 4가 실패한다. C1aN(ball_owned 경로)은 6/6이다
  - 따라서 **실패의 주 요인은 oracle_visible 경로 자체**다. 15:20 절의 "target을 남긴다"는 기전은 implementer 재생으로 기각됐으므로, 어떤 기전인지는 미확정이다
- **우리 영역에서는 구 mask가 2 seed를 살린다**(SURF 0/6 → SURFS 2/6). render mask가 실패에 일부 기여한다
- **우리 영역과 GT 영역은 성공하는 seed가 서로 다르다**(391·1769 vs 1893·485). identity 차이도 영향을 준다
- **결론:** 허들 2차는 oracle_visible 경로를 쓰지 않는 구성이어야 한다. 후보는 CL2 hybrid(ball_owned 위에서 보호만 함)다

## 2026-10-18 14:00 KST — 성공률 하락 원인 분석: held-out 결과는 seed 선택 편향으로 설명된다

**확인한 것과 기각한 것:**

| 가설 | 측정 | 판정 |
|---|---|---|
| 경계면이 부정확하다 | GT 경계면(ORAC/ORACS)으로도 실패 seed 6개 중 4개가 실패 | 기각 |
| TrajOpt가 동작을 크게 바꾼다 | 수정량이 두 방식 모두 작다(median ~0.0006 rad, 최대 ~0.05). HOLD는 갈라지기 전까지 0 | 기각 |
| target 뒷면을 남긴다 | implementer 재생으로 반박됨 | 기각 |
| 그리퍼를 닫을 때 HOLD가 팔을 붙잡아 준다 | 닫는 구간 HOLD는 C1aN 23/34, SURF 4/34로 차이가 크다. 그러나 닫는 동안 팔 이동량은 두 방식이 같다(~0.006–0.007 rad). C1aN은 HOLD가 없어도 11/11 성공 | 상관만 있고 기전은 없음 |
| **seed 선택 편향** | 아래 | **지지됨** |

**seed 선택 편향:**

- 34 seed는 collector의 **C1a가 성공한** 장면만 골랐다
- 같은 seed에서 실행 동작(팔 관절)을 비교했다
  - C1a vs C1aN: 0.05 rad 이상 차이가 처음 나는 시점의 median이 control step 280이다. 대부분 놓은 뒤이므로, **C1aN은 잡기까지 C1a의 성공 궤적을 거의 그대로 재현한다**
  - C1aN vs SURF: 33/34 run이 step 129(잡기 즈음)에 이미 크게 갈라진다. 그리퍼 명령도 31/34에서 달라진다. 즉 SURF는 **새로 굴린 독립 시도**와 같다
- 정책의 기본 성공률: collector의 주 평가 run 기준 34/43 = **0.79**
- SURF 28/34 = **0.82**로 기본 성공률과 같은 수준이다. C1aN의 34/34는 선택된 성공 궤적을 재현한 결과다
- **결론:** held-out에서 "SURF가 성공률을 떨어뜨린다(p 0.031)"는 판정은 성립하지 않는다. 앞선 보고를 정정한다
  - 실패 6 seed에서 GT 영역도 2/6만 성공한 것도 이 seed들이 어려운 장면이기 때문이라는 설명과 맞는다
- **허들:** seed를 성공 여부로 고르지 않았다. 결과는 C1a 11, C1aN 11(궤적이 거의 같음), SURF 7/14(p 0.22)로 표본이 작아 판정할 수 없다
- **올바른 비교:** 어느 arm의 성공으로도 고르지 않은 seed에서 충분한 표본으로 비교해야 한다(예: 실패 run을 포함한 새 seed, 허들 seed 확대)

## 2026-10-18 14:30 KST — 정정 (사용자 지시): 결론의 강도를 바로잡는다

**held-out closed-loop 결과의 서술 (14:00 절과 그 이전 보고를 대체):**

> C1a가 성공한 34개 조건에서 C1aN은 34개, SURF는 28개를 성공했다. 이는 기준선 성공 사례의 보존 여부를 보여주지만, 성공 여부로 선별하지 않은 전체 장면에서의 성능 차이를 추정하지는 못한다.

- 원래 수집 요청은 **성공 rollout에서의 offline 분석**을 위한 것이었다. 그 목록을 작업 성공률 비교에 그대로 쓰면서 평가 범위가 좁아졌다. 기존 실패 결과(SURF 6 실패)는 무효로 처리하지 않는다

**14:00 절에서 철회하는 해석:**

- "SURF는 궤적이 달라졌으므로 독립 시도다" — 같은 초기 조건에서 방법을 바꿔 궤적이 달라진 것은 paired 비교가 측정하려는 효과 그 자체다. 독립 표본으로 바뀌지 않는다
- "82%와 79%가 비슷하므로 기본 성공률과 같다" — 두 비율은 선정 조건이 다른 분모다. 비율이 비슷하다고 성능이 유지됐다고 판정할 수 없다
- "실패 6개는 원래 어려운 장면이다" — SURF 실패를 보고 고른 집합이다. 방법과 무관한 난이도가 확인된 것이 아니다

**원인 가설별로 허용하는 결론:**

| 항목 | 허용하는 결론 |
|---|---|
| 정답 경계면으로도 4/6 실패 | 경계 정확도만 고쳐서는 모든 실패가 해결되지 않는다. 경계 오류의 기여 전체를 배제하지는 못한다 |
| TO 수정량 median이 작음 | 전반적인 수정량은 작다. 파지 직전의 국소 수정이나 누적 영향은 배제하지 못한다 |
| target 뒷면 잔존 | 이전 수치에 근거한 설명은 철회한다. 전체 잔존량만으로 위치별 방해 여부까지 판정하지 않는다 |
| 닫는 동안 HOLD | "정지가 팔을 붙잡아 성공시킨다"는 단순한 설명은 지지되지 않는다 |
| 선택 편향 | 평가의 일반화를 제한하는 요인으로 확인됐다. 개별 실패의 물리적 원인은 여전히 미확정이다 |

**허들:**

- 11/14 대 7/14는 SURF에 불리한 관측이지만 불확실성이 크다. p = 0.22는 "판단할 수 없다"는 뜻도 "차이가 없다"는 뜻도 아니다
- C1a와 C1aN은 별도 방법 조건이지만, 거의 같은 결과를 두 개의 독립 증거로 합산하지 않는다
- 허들 seed 확대는 보류한다

**현재 판정:**

> SURF 유지 효과 미확정, 성공 조건부 평가의 한계 확인, 선별 없는 paired 재평가 진행.

**SURF 구성에 대한 주의:** SURF는 표면 extent와 render mask를 함께 바꾼 구성이다. 효과를 extent 하나의 효과로 설명하지 않는다.

**승인된 재평가:**

- 규모: 40 장면 × seed 1 × arm 2(C1aN vs SURF) = 80 run
- 장면 구성: clear 14, adjacent 13, place 13
- 선정 방식: metadata로만 고르고, 어느 arm의 성공도 보지 않는다. seed 규칙과 목록은 실행 전에 고정한다
- 우선순위: 설계·진단에 쓰지 않은 장면을 먼저 쓴다. 기존 장면에 새 seed만 쓰는 경우는 따로 표시한다
- 실패는 모두 남긴다. 성공 장면으로 바꿔 넣지 않는다
- 방법은 현재 SURF로 고정한다. 중간에 수정안을 넣지 않는다
- 보고 항목: 주 지표 = 접촉 없는 작업 성공. 함께 보고할 것
  - paired 차이와 CI
  - 한쪽만 성공한 쌍
  - grasp 실패 · 접촉 · HOLD · 완료 시간
  - 요청 지연과 계산 비용
- 판정 원칙: 동등성을 미리 선언하지 않는다. p값이 유의하지 않다고 "동등"으로 판정하지 않는다. seed를 자동으로 추가하지 않는다

## 2026-10-18 14:20 KST — 선별 없는 paired 재평가 준비 완료 · launcher 대기

- **run:** `20261018T135457+0900_f15e587290f6_2cd4b0b6acad/verify/IFG-CL3`(`IFG-CL3-run.task.md`)
  - snapshot tree `f0d8046b`(현재 SURF, 하이브리드 미포함)
  - launcher pid 1777937. benchmark 작업이 10분 동안 없으면 lock을 잡고 시작한다
- **seeds.json:** 40행, sha256 `88f6b37e…190f2b`
  - 제외: 168 episode(IFG dev 64, held-out collector 99, 허들 7)
  - 순서: 남은 1,832 후보를 `sha256("IFG-CL3:"+episode)` 순으로 정렬했다. 앞 40개를 썼고 건너뛴 장면은 없다
  - seed = ep·10+3. reused_scene는 0개다
- **장애물 배치:** 기하 규칙만 쓰고 rollout은 쓰지 않았다
  - adjacent 13개: 모두 target에서 10 cm 위치(5 cm와 7.5 cm 후보는 robot이나 손가락 자리 조건에 걸렸다). 가림 0–0.9%, 보이는 target pixel 59–128
  - place 13개: 모두 첫 번째 후보를 썼다. 기둥 방향(먼 쪽 10, 로봇 쪽 3)은 거리가 같은 두 후보를 부동소수점 반올림이 가른 결과다. 결정적이고 성공 여부와 무관하므로 그대로 고정했다(`p/LEAD_AMENDMENTS.md`)
- **사전 검사:** flag · import · address · legacy baseline · client_args 80/80 · 장애물 config 26/26 모두 일치
- **집계 스크립트:** `agg_cl3.py`를 IFG-CL1 기록에 돌려 기존 집계와 모든 공통 수치가 같음을 확인했다
- **figure:** `AG3S/ag3s/docs/figures/ifg-cl3/` (장면 4개, 배치 검사 값, 40행 표)
- **예상 소요:** GPU 약 4시간 50분 + 분석 약 10분

## 2026-10-18 17:00 KST — 사용자 지시: 최종 실험 사슬 · figure 우선순위 · run 이름 (기록만, CL3는 계속 진행)

- 지시 전문 요약과 반영 계획: [FINAL_EXPERIMENT_AND_FIGURES_PLAN_KO.md](FINAL_EXPERIMENT_AND_FIGURES_PLAN_KO.md)
- 즉시 반영: storage에 `results/development/RUN_INDEX_IFG.md`(IFG run 폴더별 실험 설명)를 만들었다. 새 run label은 공용 모듈 변경이라 CL3 후 implementer가 구현한다

## 2026-10-18 18:00 KST — doc/ 정리: active / archive (사용자 지시)

- 이 기록은 `doc/active/reviews/vla_dump_20261015/WORKLOG_S0S3_KO.md`로 옮겼다
- **이동:** 199 파일 → `doc/active/`(18), `doc/archive/`(181). 각 폴더 README에 파일별 설명을 적었다
  - git 등록 파일은 rename으로 기록했다(commit은 하지 않음)
- **참조 수정:** 68개 파일의 경로 문자열을 바꿨다
  - 대상: CLAUDE.md, `.claude`(hook · agent · skill · project-context), scripts, AG3S docs · handoff, paper 도구
  - 옮긴 md 20개의 상대 링크 166개를 다시 계산했다
  - JSON · log 같은 과거 기록과 `.runtime`, 잠근 `ifg_*.py`(주석 1줄)는 고치지 않았다
- **후속 수정:** 자동 치환에서 빠진 `paper/tools/analysis/audit_ifg_evidence.py`의 조각 경로를 고쳤다. memory 2개의 경로도 고쳤다
- **검사:**
  - hook 문법 OK. setup 스크립트 py_compile OK
  - 남은 깨진 링크 22개는 이동 전부터 깨져 있던 것이다(루트 기준 경로, `:줄번호` 접미사, 저장소에 없는 PDF)

## 2026-10-19 01:15 KST — 선별 없는 paired 재평가 결과 (40 장면 × 기존 방식 / 표면 영역 방식)

**비교한 두 방식**

| 이름 | 뜻 |
|---|---|
| 기존 방식 (C1aN) | 현재 serving 구성에서 target 선택만 noun attention(명령의 과일 단어)으로 바꾼 것. target 주변 **공 모양 영역**을 잡는 손의 충돌 지도에서 지운다 |
| 표면 영역 방식 (SURF) | 기존 방식 + attention과 depth로 복원한 **target 표면**만 지움 + robot을 **렌더한 표면**으로 지움. 두 가지를 함께 바꾼 구성이다 |

- 장면은 어느 방식의 성공도 보지 않고 metadata 순서로 고정했다(`seeds.json` sha256 `88f6b37e…190f2b`). clear 14 · adjacent 13 · place 13, seed 1, 80 run
- run: `20261018T135457+0900_f15e587290f6_2cd4b0b6acad/verify/IFG-CL3`, 수치: `AG3S/ag3s/docs/handoff/IFG-CL3.verify.json`

**주 지표: 접촉 없는 작업 성공 (paired, 40쌍)**

| | 기존 방식 | 표면 영역 방식 | 차이 (표면 − 기존) | 95% CI (episode 쌍 bootstrap 10,000) | McNemar exact p (참고) |
|---|---|---|---|---|---|
| 전체 | 17/40 | 19/40 | +0.050 | [−0.125, +0.225] | 0.774 |
| clear | 5/14 | 4/14 | −0.071 | [−0.286, +0.143] | 1.0 |
| adjacent | 7/13 | 9/13 | +0.154 | [−0.231, +0.538] | 0.688 |
| place | 5/13 | 6/13 | +0.077 | [−0.154, +0.308] | 1.0 |

**2×2 (둘 다 성공 / 기존만 / 표면만 / 둘 다 실패)**

| | 둘 다 성공 | 기존만 | 표면만 | 둘 다 실패 |
|---|---|---|---|---|
| 전체 | 12 | 5 | 7 | 16 |
| clear | 3 | 2 | 1 | 8 |
| adjacent | 5 | 2 | 4 | 2 |
| place | 4 | 1 | 2 | 6 |

- 이 측정으로 할 수 있는 말: 선별 없는 40쌍에서 성공 차이의 95% 구간은 −12.5%p에서 +22.5%p까지다. 차이의 방향도 크기도 이 표본으로 정해지지 않는다
- 하지 않는 말: "차이 없음", "동등", "SURF가 낫다". p가 크다는 것은 판단할 수 없다는 뜻이다
- 성공 조건부 평가(10-17, C1a 성공 34 장면에서 34 대 28)와 분모가 다르다. 두 결과를 하나의 성공률로 합치지 않는다. 선별 없는 장면에서는 두 방식 모두 절반 이하만 성공했다

**함께 보는 수치**

| | 기존 방식 | 표면 영역 방식 |
|---|---|---|
| 잡기 실패 (no_grasp) | 12 | 13 |
| 들어 올리다 놓침 (slip_at_lift) | 9 | 3 |
| 운반 중 놓침 (slip_in_transport) | 3 | 2 |
| 다시 잡기 (regrasp) | 3 | 4 |
| 비허용 link–target 접촉이 있던 run | 1 | 1 |
| 장애물 접촉이 있던 run | 0 | 2 (손–장애물 1행, target–장애물 1행; adjacent) |
| 이웃 과일 접촉이 있던 run | 6 | 4 |
| HOLD chunk 합 / HOLD가 있던 run | 80 / 25 | 32 / 12 |
| 완료 시간 median (성공 run) | 14.9 s | 14.9 s |

- 잡기 성공 paired 차이는 −0.025 [−0.175, +0.125]다
- 놓침과 HOLD 수의 차이는 관측값이다. 원인은 이 측정으로 정하지 않는다

**지연과 계산 비용 (각 run 첫 요청 제외)**

| | 기존 방식 | 표면 영역 방식 |
|---|---|---|
| client 왕복 median / p95 | 1,253 / 2,109 ms | 3,079 / 4,011 ms |
| 서버 합계 median / p95 | 905 / 1,583 ms | 2,783 / 3,670 ms |
| 533 ms 예산 초과 비율 (client 왕복) | 100 % | 100 % |
| attention 표면 계산 median | — | 1,742 ms (build 867 · codes 396 · solve 291) |
| robot 렌더 mask median | — | 166 ms |

- 표면 영역 방식의 서버 계산 중 attention 표면 단계가 가장 크다. 두 방식 모두 실시간 예산을 넘는다
- 표면 영역 방식 진단: chunk 2,877개 중 표면 영역이 실제로 적용된 chunk 1,538개(53%). 나머지는 target-free 층 없음 1,017, identity 미확정 311, 영역 없음 11

**인프라 사고와 처리 (결과 해석과 무관, 기록용)**

- 18–19 KST 사이 NFS에서 서버 기록 파일 하나(`chunk_00720.npz`, 표면 영역 방식 ep194 run의 75 chunk 중 1개)의 속성 조회가 끝나지 않았다. client는 시작할 때 run 폴더 전체를 `lstat`하므로 19:52 KST 이후 시작한 client 두 개가 멈췄다
- 그 파일을 run 밖 `<RUN>__nfs_quarantine/`로 옮기고 나머지 749 chunk를 hard link로 복원했다. 멈춘 두 run(표면 영역 방식 ep1135, 기존 방식 ep1034)은 PHASE B 뒤 같은 seed로 다시 돌렸다. 재시도 횟수는 `agg.json` attempts에 남았다
- 성공·접촉 판정은 client 기록과 MuJoCo replay에서 나오므로 격리한 chunk 1개와 무관하다. 영향은 표면 영역 방식 진단 수치에서 1 chunk가 빠지는 것뿐이다
- 이 pod에는 같은 NFS 대기로 며칠째 멈춘 다른 process도 있다. 앞으로 run 폴더가 커질수록 client 시작 때의 전체 `lstat`이 같은 위험을 진다(run label 작업 때 implementer에게 함께 넘길 후보)

**figure** (`AG3S/ag3s/docs/figures/ifg-cl3/`, 각 `.json` sidecar)

- `ifg-cl3-paired-outcome` 2×2와 paired 차이·CI · `ifg-cl3-arm-metrics` 실패 모드·접촉·HOLD·완료 시간 · `ifg-cl3-latency` 지연과 단계별 비용 · `ifg-cl3-outcome-grid` 40 장면 × 방식 결과 · `ifg-cl3-discordant-scenes` 한쪽만 성공한 쌍의 실제 장면(같은 시각 frame)
- 그림 script: `AG3S/ag3s/experiments/studies/ifg_cl3_results.py`

**현재 판정:** 표면 영역 방식의 효과는 이 40쌍으로 정해지지 않았다. seed는 자동으로 늘리지 않는다. 다음 단계(run label, 최종 사슬 탐색)는 사용자 확인 뒤 진행한다

## 2026-10-19 02:30 KST — 실패 원인 분석(CPU) · 로봇 렌더 mask 보류(사용자 판정) · 사과 시나리오 정리

**궤적 최적화가 정책 action을 얼마나 바꿨나 (80 run, GPU 없음)**

- 방법: 서버 기록의 chunk마다 VLA 원래 출력(`reference_chunk`)과 최적화 결과(`refined_chunk`)를 손끝(TCP) 위치로 바꿔 차이를 쟀다. 처음 그리퍼를 닫기 전(접근 구간)만 본다
- script `AG3S/ag3s/experiments/studies/ifg_cl3_deflection.py`, 결과 `IFG-CL3/failure_analysis/cl3_deflection.json`

| 그룹 (run 수) | chunk당 수정량 median | chunk 최대 수정량 median | 접근 중 HOLD가 있던 run |
|---|---|---|---|
| 기존 방식 성공 (17) | 2.3 mm | 13.4 mm | 2 |
| 기존 방식 잡기 실패 (12) | 2.5 mm | 13.9 mm | 1 |
| 표면 영역 방식 성공 (19) | 2.3 mm | 8.3 mm | 0 |
| 표면 영역 방식 잡기 실패 (13) | 2.5 mm | 14.5 mm | 5 |

- 대부분의 실패 run에서 수정량은 성공 run과 같은 크기다. 예외: 표면 영역 방식 ep1060(닫기 직전 150 mm), 기존 방식 ep495(122 mm + HOLD 4), 표면 영역 방식 ep1256(닫기 직전 HOLD)
- 두 방식이 모두 잡기에 실패한 장면 8개(target: 오렌지 4, 바나나 3, 배 1, 사과 0)
- 하지 않는 말: "궤적 최적화는 원인이 아니다". 작은 수정도 닫힌 루프에서 이후 궤적을 바꾼다. 같은 장면의 VLA 단독 결과가 없어 VLA 원인과 공통 최적화 원인을 가르지 못했다
- 과일별 접촉 없는 성공(기존 / 표면 영역): 사과 11 장면 5 / 7, 오렌지 16 장면 8 / 6, 바나나 8 장면 2 / 4, 배 5 장면 2 / 2

**사용자 판정 (2026-10-19)**

- **로봇 렌더 mask 보류.** 실물 로봇(custom wrist camera 기구물 · custom 그리퍼)에서 정확한 mesh와 보정을 요구해 쓰기 어렵고, 계산 시간(median 166 ms)이 실시간성을 해친다
- **다른 과일은 성공 1회만 쓰고 사과 위주로 진행한다**
- 기록의 사과 성공 시나리오 정리: [APPLE_SCENARIOS_KO.md](APPLE_SCENARIOS_KO.md). 기록 성공으로 고른 목록이므로 시연 · figure · 사슬 탐색용이다. 성공률 평가에 쓰면 성공 조건부 결과가 된다

## 2026-10-19 03:50 KST — 2단계 요소를 렌더 mask 없이(FK 구 mask) held-out offline으로 재측정 (IFG-S2FK)

- 사용자 지시: "1번부터 진행" — 렌더 mask 보류 조건에서 기억 전환 · 안전 경보 · 3D 파지 · 단계 판단을 CPU offline으로 다시 잰다
- task(결과 전 고정): `AG3S/ag3s/docs/handoff/IFG-S2FK.task.md`, 수치: `IFG-S2FK.verify.json`, run `20261019T025853+0900_f15e587290f6_93594d3dffc0/s2fk`
- 데이터: 잠근 held-out 평가와 같은 3,196 frame(성공 rollout 34 episode 2,528 frame + 실패 9 episode). 잠근 코드는 고치지 않고 robot mask 경로만 FK 구 mask로 바꿨다
- 일관성: FK 조건의 "전환 없음" IoU가 잠근 평가의 FK 조건 값과 3,196 frame 모두 같다

**기억 feature 영역 전환과 안전 경보 (성공 34 episode)**

| | 렌더 mask (잠근 평가) | FK 구 mask (이번) |
|---|---|---|
| 표면 IoU 전환 없음 → 전환 | 0.876 → 0.891 | 0.832 → 0.845 |
| 전환 효과 (paired) | +0.015 [−0.012, +0.039] | +0.013 [−0.012, +0.037], 좋아짐 14 / 나빠짐 6 |
| 이웃 물체 오포함 voxel | 46.3k → 13.6k | 46.2k → 13.8k |
| 경보: 잘못 고른 frame 중 잡은 수 | 85 / 100 | 83 / 97 |
| 경보: 오경보 | 153 | 153 |
| 경보 AUROC | 0.925 | 0.924 |
| 경보 frame을 버릴 때 FP 제거 / TP 손실 | 88 % / 6.7 % | 93 % / 6.9 % |

- 경보는 렌더 mask 없이도 같은 수준으로 동작한다. 기준값(0.590)은 다시 정하지 않았다
- 기억 전환의 IoU 효과는 두 조건 모두 미확정이다. 이웃 물체 오포함은 두 조건 모두 약 70 % 줄었다
- 렌더 mask를 빼면 표면 IoU 자체가 0.044 낮다(손에 든 동안 0.848 → 0.640, 잠근 평가의 단계별 값)

**3D 관계로 파지 · 단계 판단 (성공 rollout 2,528 frame, FK 조건)**

| 판단 | 현재 serving 규칙 | 학습 없는 3D 규칙 | 학습형 (dev weight 고정) |
|---|---|---|---|
| 파지 정확도 | gripper < 0.9: 0.898 | 0.935 (+0.036 [+0.019, +0.057], 24 / 1) | 0.976 (+0.078 [+0.057, +0.102]) |
| 단계 정확도 | HSR: 0.930 | 0.894 (−0.036 [−0.059, −0.017], 4 / 26) | 0.956 (+0.026 [+0.001, +0.050]) |

- 3D 쥠 규칙(결과 전 고정): gripper < 0.9 그리고 target 중심이 손 대리점에서 0.131 m 안(모델에서 잰 손–파지점 0.081 m + 과일 반지름 0.05 m)
- 렌더 조건과 FK 조건의 결과가 거의 같다. 파지 · 단계 판단은 robot mask 선택에 거의 영향을 받지 않는다
- 학습 없는 쥠 규칙은 serving gripper 규칙보다 낫다. 닫힌 gripper가 target에서 떨어져 있는 frame을 쥠에서 뺀다
- 학습 없는 단계 규칙은 HSR보다 낮다. 원인: 정답이 "잡기 전"인 frame 142개(gripper가 닫히는 중, 손이 target 근처)를 place로 판단했다. 시뮬레이터 정답은 두 손가락이 닿아야 "쥠"이다. 규칙은 고정이라 고치지 않았다
- 학습형은 dev 렌더 조건에서 학습한 weight를 그대로 썼는데 held-out FK 조건에서도 두 판단 모두 serving보다 높다. 다만 시뮬레이터 정답으로 학습한 weight다
- 참고(채택 근거 아님): held-out에서 다시 학습하면 단계 0.990 · 파지 0.983

**figure** (`AG3S/ag3s/docs/figures/ifg-s2fk/`, sidecar `.json`): `ifg-s2fk-memory-gate` · `ifg-s2fk-phase-held` · `ifg-s2fk-scenes`(실제 held-out frame: 규칙이 맞고 gripper 규칙이 틀린 frame, 규칙이 틀린 닫힘 중 frame)

## 2026-10-19 13:30 KST — 2번 서버 연결 · 사과 장면 closed-loop 착수 (사용자: "권장사항대로 수행")

**권장안으로 확정한 것**

| 항목 | 결정 |
|---|---|
| 서버에 올릴 요소 | 안전 경보(feature 기억 신뢰도) · 학습 없는 3D 쥠 규칙. 둘 다 flag, 기본 off |
| 작업 단계 판단 | HSR(subtask gate) 유지. 학습형 weight는 쓰지 않는다 |
| 경보가 울린 frame | target 접촉 예외 보류(target을 장애물로 유지) |
| 경보가 쓰는 영역 | offline에서 검증한 표면 영역 선택 그대로(FK 구 mask). target extent는 공 영역 그대로. code image 단계는 건너뛴다. 약 1.2 s 추가 지연이 예상되어 closed-loop에서 잰다 |

**병렬로 시작한 작업**

- implementer: `AG3S/ag3s/docs/handoff/IFG-S2SERVE.task.md` — 두 flag 구현과 test, flag off 시 출력 불변
- verifier: `AG3S/ag3s/docs/handoff/IFG-APPLE-run.task.md` — 사과 target 24 장면(장애물 없음 16, 바구니 옆 장애물 8, metadata 해시 순서, 결과 전 고정) × VLA 단독 · 기존 방식, seed 1. 새 구성은 구현 뒤 같은 seed로 추가한다
- target 옆 기둥 배치는 쓰지 않는다(사용자: 가림이 작고 궤적과 간섭하지 않는 배치는 의미가 약하다)

## 2026-10-19 15:00 KST — 서버 연결 구현 완료 · 사과 장면 closed-loop 진행 · 새 구성 arm 대기

**서버 연결 (IFG-S2SERVE, implementer, `AG3S/ag3s/docs/handoff/IFG-S2SERVE.impl.md`)**

- 새 flag(기본 off): `--confidence-gate on`(안전 경보), `--held-3d-rule on`(학습 없는 3D 쥠 규칙). off면 serving 출력 불변(test, legacy 회귀 기준선 동일)
- 경보: 서버가 표면 영역(FK 구 mask)과 VLA layer 8 feature로 점수를 내고, 잠근 기준값 0.5904 미만이면 그 frame의 target 접촉 예외를 보류. 물체를 쥔 동안은 적용하지 않고 기록만
  - layer 8 feature: openpi를 고치지 않고 별도 계산(held-out dump 코드와 같은 layer 0–8). CPU에서 GPU dump와 token cosine ≥ 0.9997, 경보 점수 차 ≤ 0.0055, 경보 frame 동일
  - 서버 경로 점수는 offline 함수와 같은 frame에서 차이 0.0
- 3D 쥠 규칙: latch가 쥠을 거는 조건에 `gripper < 0.9 그리고 hand_d ≤ 0.131 m`를 AND. target 중심은 직전 요청의 target(lead 판정: 영역 없으면 gripper만)
- 알려진 비용 · 위험: 경보 영역 선택 CPU 약 1 s/요청(예산 533 ms 초과), layer 8 별도 계산의 GPU 지연 미측정, 경보 frame에서 HOLD 증가 가능
- 전체 회귀 test(`tests/ag3s tests/trajopt`)는 rollout CPU 경합을 피해 사과 실험 뒤 lead가 돌린다

**사과 장면 closed-loop (IFG-APPLE, lead 직접)**

- run `20261019T132719+0900_f15e587290f6_11408fd6d27f/verify/IFG-APPLE`: 사과 target 24 장면(장애물 없음 16 · 바구니 옆 8, metadata 해시 순서) × VLA 단독 · 기존 방식
- smoke 1차: VLA 단독 client가 AG3S 서버에서 seed를 적용받지 못해 스스로 멈춤(구성 오류). VLA 단독 chain을 T39처럼 `--no-safe` 서버로 바꿈
- smoke 2차 통과: VLA 단독 첫 chunk와 기존 방식 서버의 VLA 원 출력 차 1.1e-7(같은 VLA 출력)
- 경합: implementer의 pytest 2개 동시 실행으로 load 44 → client 1회 21분. test를 1개씩 · 1 thread · nice로 제한하도록 요청

**새 구성 arm (IFG-APPLE-S2)**: 같은 24 장면 · seed, snapshot_s2(새 코드), 기존 방식 + 경보 + 3D 쥠. flag 수락 CPU 검사 통과. 사과 실험이 GPU lock을 놓으면 바로 이어서 시작한다

## 2026-10-19 16:50 KST — 사과 장면 결과: VLA 단독 대 기존 방식 (IFG-APPLE)

- 장면: 사과 target 24(장애물 없음 16 · 바구니 옆 기둥 8), metadata 해시 순서로 결과 전 고정. seed 1. 수치 `AG3S/ag3s/docs/handoff/IFG-APPLE.verify.json`
- 짝 확인: VLA 단독 첫 chunk와 기존 방식 서버의 VLA 원 출력 최대 차 1.1e-7

| | VLA 단독 | 기존 방식 | 차이 (기존 − VLA) | 95 % CI |
|---|---|---|---|---|
| 접촉 없는 작업 성공 | 18 / 24 | 16 / 24 | −0.083 | [−0.292, +0.125] |
| 장애물 없음 | 12 / 16 | 12 / 16 | 0.000 | [−0.250, +0.250] |
| 바구니 옆 기둥 | 6 / 8 | 4 / 8 | −0.250 | [−0.625, +0.250] |
| 잡기 실패 | 5 | 7 | | |
| 장애물 접촉이 있던 run | 0 | 0 | | |
| 비허용 link–target 접촉이 있던 run | 0 | 2 | | |
| HOLD chunk 합 (run 수) | 0 | 119 (17) | | |
| client 왕복 median | 123 ms | 893 ms | | |

- 2×2: 둘 다 성공 13, VLA 단독만 5, 기존 방식만 3, 둘 다 실패 3
- 사용자 질문 "baseline checkpoint는 성공하는가"에 대한 측정: 이 사과 24 장면에서 VLA 단독은 18 / 24(75 %)
- 기존 방식만 실패한 5 장면 중: ep211은 잡기 전 충돌 HOLD 30번으로 접근이 막혔다. ep320 · 954 · 1345는 HOLD 없이 잡기 실패. 원인은 이 측정으로 정하지 않는다
- 동등성이나 "AG3S가 해친다"를 판정하지 않는다. CI가 0을 포함한다
- 인프라: 분석 1차 실패(주소 검사 파일 누락, lead 준비 누락) → 파일을 만들고 분석 재실행
- figure: `AG3S/ag3s/docs/figures/ifg-apple/`(2×2 · CI, 실패 모드 · HOLD, 지연, 24 장면 결과표, 한쪽만 성공한 장면의 실제 영상 frame)
- 다음: 같은 24 장면 · seed로 새 구성(기존 방식 + 안전 경보 + 3D 쥠 규칙) 진행 중

## 2026-10-20 03:10 KST — 새 구성(기존 방식 + 안전 경보 + 3D 쥠 규칙) 결과 · 두 기능이 서버에서 offline과 다르게 동작함

- 같은 사과 24 장면 · seed. 새 구성 rollout은 10-19 18:34 KST에 끝났다(24/24, 실패 0). 세 arm 집계는 합본 폴더 `verify/IFG-APPLE-3ARM`. 수치 `AG3S/ag3s/docs/handoff/IFG-APPLE-S2.verify.json`
- 짝 확인: 새 구성 서버의 VLA 원 출력 = VLA 단독 첫 chunk(차 1.1e-7)

| 접촉 없는 작업 성공 | VLA 단독 | 기존 방식 | 새 구성 |
|---|---|---|---|
| 전체 24 | 18 | 16 | 15 |
| 장애물 없음 16 | 12 | 12 | 12 |
| 바구니 옆 기둥 8 | 6 | 4 | 3 |

- 새 구성 − 기존 방식: −0.042 [−0.125, 0.000]. 24 장면 중 23 장면의 결과가 같고, 다른 1 장면(ep572)은 기존 방식 성공 · 새 구성 들어 올리다 놓침
- 새 구성 − VLA 단독: −0.125 [−0.375, +0.125]
- HOLD 합: 기존 방식 119, 새 구성 105(둘 다 17 run). client 왕복 median 893 → 2,478 ms

**두 기능의 서버 동작 (진단, 수정 없음)**

| | 서버 (새 구성) | held-out offline |
|---|---|---|
| 경보 점수가 기준 0.590 미만인 비율 | 잡기 전 0 % · 접근(latched) 24 % · 닫는 중 59 % · 쥔 동안 85 % · 놓은 뒤 93 % | 13 % |
| 경보 1,800 요청 중 | 1,072 울림, 307 적용, 596 예외 없음, 169 쥔 동안이라 건너뜀 | — |
| gripper 닫힘 때 hand_d median | 0.313 m (0.131 m 이하 비율 낮음) | 정답 쥠 frame 0.088 m |
| 3D 쥠 규칙 | gripper가 쥠이라 한 458 요청 중 375 거절 | — |

- 하는 말: 두 기능 모두 서버에서 offline 검증과 다른 입력 분포를 받았다. 이 상태의 결과는 offline에서 검증한 기능을 시험한 것이 아니다
- 하지 않는 말: 경보 · 3D 규칙의 효과에 대한 결론. 원인(서버 영역 선택 · 기억 갱신 · target 중심의 출처)은 확정하지 않았다
- 후보 원인(미확인): 손에 든 과일이 FK 구 mask에 지워진 뒤 서버의 target이 다른 물체로 옮겨가 hand_d가 커지고, 그 영역의 feature가 기억과 달라 점수가 떨어진다
- figure: `AG3S/ag3s/docs/figures/ifg-apple/ifg-apple-s2-diagnosis`(세 arm 성공 · 경보 점수 분포 · hand_d 분포), `ifg-apple-s2-*`(기존 방식 대 새 구성)

## 2026-10-20 03:15 KST — 새 구성 원인 진단 (IFG-APPLE-S2-CAUSE): 경보는 기억 고착, 3D 쥠 규칙은 결정에 관여하지 않음

- 03:10 절의 "후보 원인(미확인)"을 서버 기록과 replay GT로 확인했다. CPU · 읽기 전용이고 서버 코드는 고치지 않았다
- script `AG3S/ag3s/experiments/studies/ifg_apple_s2_cause.py`, 출력 `verify/IFG-APPLE-S2-CAUSE/`(요청 1,800행 `rows.jsonl`, `numbers.json`), 수치 `AG3S/ag3s/docs/handoff/IFG-APPLE-S2-CAUSE.verify.json`
- GT 단계(replay에서 각 요청의 첫 control row로 정함. 평가 전용): 접근(손끝–사과 > 0.08 m) · 손 근처(≤ 0.08 m) · 들림(사과가 시작 높이보다 0.02 m 이상 위) · 놓은 뒤(들린 뒤 손끝 > 0.10 m)

**1. 안전 경보: 같은 GT 단계에서 서버 점수가 offline보다 크게 낮다**

| | offline 잡기 전 | offline 쥠 | offline 놓은 뒤 | 서버 접근 | 서버 손 근처 | 서버 들림 | 서버 놓은 뒤 |
|---|---|---|---|---|---|---|---|
| 점수 median | 0.979 | 0.977 | 0.998 | 0.962 | 0.963 | 0.236 | 0.401 |
| 기준 0.590 미만 | 2 % | 4 % | 15 % | 29 % | 13 % | 82 % | 94 % |
| 선택 영역 중심–GT 사과 median | — | — | — | 0.019 m | 0.029 m | 0.165 m | 0.018 m |

- 서버 "접근" 29 %에는 잡기에 실패한 뒤 다시 접근하는 요청도 들어 있다(실패 episode는 "들림"에 도달하지 않음)

**원인 (측정으로 확인한 사슬)**

1. **쥔 동안 사과가 서버 depth에서 지워진다.**
   - 경보는 서버 카메라의 robot mask를 쓴다(`pipeline._confidence_gate_step`). attach된 동안 이 mask는 쥔 물체의 sphere까지 포함한다(T32 H1, `_depth_robot_mask`). offline S2FK의 FK mask는 로봇만 지웠다
   - 기록된 filtered cloud에서 GT 사과 4 cm 안의 점 수: attach 전 들린 frame은 median 58(6 frame)이다. attach된 채 들린 frame은 median 0이고, 83 frame 중 99 %가 0이다
   - 그래서 들린 동안 선택 영역이 다른 물체로 간다(중심 오차 median 0.165 m)
2. **chain memory가 다른 물체로 전환되어 그 자리에 고착된다.**
   - 기억은 영역이 있는 모든 frame에서 갱신된다(`ConfidenceGate.step`: switch면 가장 비슷한 후보, 아니면 이번 선택)
   - 한 번 다른 물체로 갱신되면, 다음 frame에 선택이 사과로 돌아와도 기억과 가까운 다른 물체 쪽으로 다시 switch한다
   - 24 장면 중 19 장면에서 첫 switch 이후 요청의 절반 넘게 switch가 이어졌다. 그래서 놓은 뒤 선택 영역이 사과 위에 있어도(오차 median 0.018 m) 점수가 낮다
3. **반대 방향의 오류도 있다.** 선택 영역이 사과에서 10 cm 넘게 벗어난 333 요청 중 71 요청은 점수가 기준 이상이었다(경보 없음). 예: ep1313 k ≥ 35는 영역이 0.65 m 떨어진 물체 위에 있는데 점수는 0.96–0.998이다
- 경보가 실제로 target 예외를 보류한 307 요청의 GT 단계: 접근 162 · 놓은 뒤 135 · 손 근처 5 · 들림 5
- 하지 않는 말: robot-only mask나 기억 동결로 고치면 offline 수준이 된다는 것. 서버 기록에는 카메라별 depth와 layer-8 feature가 없어 그 조건을 다시 계산하지 못했다

**2. 3D 쥠 규칙: 24 장면에서 attach 결정을 한 번도 바꾸지 않았다 (03:10 절 정정)**

- 03:10 절의 "458 요청 중 375 거절"과 "hand_d median 0.313 m"에는 이미 attach된 뒤의 요청이 섞여 있었다. 3D 규칙은 attach를 거는 판단(CLOSING → HELD)에만 쓰인다
- attach된 뒤 hand_d가 큰 이유도 1번과 같다. 사과가 지워져 target 중심이 들기 전 위치에 머문다. 들린 동안 target 중심–GT 오차 median 0.255 m, GT 중심 기준 hand_d median 0.101 m(기준 0.131 m 안)

| CLOSING에서의 attach 판단 요청 | 기존 검사 통과 | 기존 검사 거절 |
|---|---|---|
| 손끝이 사과 위(≤ 0.06 m), 3D 통과 | 17 | 6 |
| 손끝이 사과 위, 3D 거절 | **0** | 6 |
| 손끝이 사과에서 떨어짐, 3D 통과 | 0 | 13 |
| 손끝이 사과에서 떨어짐, 3D 거절 | **0** | 201 |

- 3D 규칙이 거절한 207 요청은 모두 기존 검사(reach 등)도 거절했다. 처음 attach한 요청 번호는 24 장면 모두 기존 방식과 같다(attach한 장면 17)
- 하는 말: 이 24 장면에서 3D 규칙은 결과에 영향을 주지 않았다. 하지 않는 말: 기존 reach 검사가 놓치는 경우가 없다는 것

**3. 결과가 갈린 유일한 장면 ep572**

- 두 arm은 k=17까지 실행 action이 완전히 같다(차 0)
- k=18:
  - 사과가 17 cm 들린 채 손에 있었다. 두 arm 모두 latch가 attach를 풀었다(`not_following`)
  - attach가 풀리자 새 구성의 경보가 처음으로 target 예외를 보류했다(점수 0.17). 실행 action은 이 요청에서 처음 갈라졌다(0.05 rad)
- 새 구성은 k=18–29 내내 예외를 보류했다. 사과는 k=20→21 사이 바구니 밖으로 떨어졌다. 기존 방식은 예외를 유지했고 k=26에 바구니에 놓았다
- 하는 말: 이 장면의 실패는 경보가 손에 든 사과에 대한 예외를 보류한 시점부터 갈라졌다. 하지 않는 말: 다른 장면 일반에 대한 경보의 효과

**figure** (`AG3S/ag3s/docs/figures/ifg-apple/`, sidecar `.json`)

- `ifg-apple-s2-cause`: (a) 24 장면 경보 점수 시간축(들린 구간, switch 표시), (b) GT 단계별 점수 offline 대 서버, (c) attach 판단 교차표, (d) ep572 사과 높이와 두 arm 실행 action 차
- `ifg-apple-s2-cause-scene`: ep572 실제 장면(정면 client 영상) k=17 · 18 · 20 · 22, 기존 방식 대 새 구성

**현재 판정:** 서버의 경보는 offline에서 검증한 기능과 다른 입력(쥔 물체가 지워진 mask)을 받았다. 그 때문에 기억이 다른 물체에 고착됐다. 3D 쥠 규칙은 이 장면들에서 기존 검사와 중복됐다. 수정 방향(경보 mask를 robot-only로 바꿈, attach 중 기억 동결, 경보를 끄고 다음 단계로 감)은 사용자 판정 뒤 진행한다

## 2026-10-20 03:50 KST — IFG-S2SERVE 뒤 전체 회귀 test 결과

- 명령: `.venv-ag3s` · `JAX_PLATFORMS=cpu` · `nice -n 15` · OMP/BLAS 2 thread, `pytest tests/ag3s tests/trajopt`(26 분 11 초)
- 결과: **3,555 passed, 1 failed, 23 skipped**
- 실패 1건: `tests/trajopt/test_t43t_replay.py::test_off_is_bit_identical_to_the_pre_fix_commit`(key `to_T42_L1992_16`). 10-16 IFG-CL1 때와 같은 test, 같은 key다. 그때 HEAD의 clean archive에서도 똑같이 실패하는 것을 확인했다(`IFG-CL1.impl.md:64`). IFG-S2SERVE 변경과는 무관하며, 이번에 다시 돌리지 않았다
- 인프라: 이 test process(RSS 약 16 GB)를 benchmark · development-22 세션이 pod 메모리 문제로 문의했다. 이 세션 소유임을 확인하고 끝까지 실행한 뒤, 종료를 두 세션에 알렸다

## 2026-10-20 09:40 KST — 사과 24 장면 실패 원인 분석 (IFG-APPLE-FAIL, 사용자 지시 1·2)

- 사용자 지시
  1. 실패한 8–9 episode의 원인을 먼저 분석한다
  2. 안전 경보와 3D 쥠 규칙이 성공률에 영향을 준 문제를 파악한다
  3. 그 뒤 경보 수정·재검증과 기능 끄고 다음 계획 중 어느 쪽으로 갈지 판단한다
- 방법: CPU · 읽기 전용, 서버 코드 변경 없음. 세 arm의 replay, 기존 방식·새 구성의 서버 chunk 기록, run snapshot 모델 FK를 썼다. 각 장면의 초기 과일 위치는 replay와 같은 reset으로 다시 만들었다
- script `ifg_apple_fail.py` · `ifg_apple_scene_bodies.py`, 출력 `verify/IFG-APPLE-FAIL/`, 수치 `AG3S/ag3s/docs/handoff/IFG-APPLE-FAIL.verify.json`
- **주의:** 아래의 "이웃 과일 8.2 cm" 구분과 실패 유형은 기록을 읽은 뒤 정했다(post hoc). 미리 정한 검정이 아니다

**1. 실패 장면의 유형 (기존 방식 8 · 새 구성 9)**

| 유형 | 장면 | 무엇이 일어났나 |
|---|---|---|
| VLA 단독도 실패 | 526 · 979 · 1313 | VLA 자체가 이 장면에서 실패한다. 1313은 AG3S가 손끝을 위로 평균 7.9 mm 밀어 손이 사과에서 더 멀어졌다 |
| 접근이 막힘 / 위로 밀림 | 211 · 954 | 사과 바로 옆 바나나(7.2 · 8.2 cm)를 피하느라 손가락이 사과에 닿지 못했다 |
| 운반 중 attach 해제 → 낙하 | 250 (두 arm) · 572 (새 구성만) | 들어 올린 뒤 서버가 "쥔 물체가 따라오지 않는다"고 판단해 attach를 풀었다 → 손에 든 사과가 다시 장애물이 되어 공중 HOLD → 낙하 |
| 뚜렷한 서버 결정 없음 | 320 · 1345 | 사과 근처 HOLD도 큰 수정도 없다. 접근 초반(t≈18–33)부터 궤적이 VLA 단독과 갈라졌고, VLA가 더 일찍(t=123 · 107, VLA 단독은 211 · 180) 높게 닫았다 |

**2. 공통 조건: 사과 옆 과일이 가까운 장면**

| 사과와 가장 가까운 과일의 중심 거리 | 장면 수 | VLA 단독 | 기존 방식 | 새 구성 |
|---|---|---|---|---|
| ≤ 8.2 cm | 7 | 7 | 3 | 2 |
| > 8.2 cm | 17 | 11 | 13 | 13 |

- AG3S만 실패한 장면 6개 중 5개(211 · 250 · 954 · 1345 · 572)가 ≤ 8.2 cm 쪽에 있다. AG3S만 성공한 3 장면(342 · 421 · 1222)은 모두 > 8.2 cm 쪽이다
- **target 덩어리에 이웃 과일이 합쳐짐 (4 장면: 211 · 250 · 572 바나나, 1345 오렌지)**
  - 서버 target 덩어리의 최대 길이는 136–151 mm다(사과 하나만이면 66–76 mm)
  - target 점의 약 1/3이 사과에서 8.1–8.7 cm 떨어진 탁자 높이에 있다. 그 중심은 reset 때 이웃 과일 위치와 1 cm 안에서 같다
- **ep211:** 사과 표면 중 바나나 쪽이 판정이 애매한 voxel(255–423개)로 남았다. 접촉이 허가된 손가락(`ee_finger_l1`)이 거기에 20–24 mm 겹친다고 판정되어 34 chunk 연속 HOLD했다. VLA 단독이 잡은 시각(12.3 s)에 기존 방식은 손이 사과 위에 멈춰 있었다
- **ep954:** 덩어리는 합쳐지지 않았다. 사과 근처에서 가장 강하게 걸린 제약이 손가락–(target-free 거리장) 5.0 mm 여유였다. 그래서 손끝이 평균 5.2 mm 위로 밀렸다. 같은 값이 성공 run에서는 −2.4 ~ +3.0 mm였다
- **ep250 · 572의 attach 해제:**
  - 해제 때 서버가 본 "쥔 물체" 위치는 탁자 위 바나나 자리와 1 cm 안에서 같다
  - 들고 있는 동안 사과는 robot mask에 지워져 보이지 않았다(03:15 절). 그래서 합쳐진 덩어리에서 남은 바나나 쪽만 추적됐다
- 닫는 순간 손끝 중점이 사과 중심보다 높은 정도의 median: 성공 3.5–3.6 cm, 실패 3.9–4.1 cm

**3. 주 지표의 정의가 미친 영향**

- 주 지표("접촉 없는 작업 성공")는 이웃 과일 접촉을 실패로 세지 않는다(`agg_apple.py`)
- VLA 단독이 ≤ 8.2 cm 장면에서 성공한 7개 중 3개(211: 8 · 954: 17 · 1345: 31 row)는 손이 이웃 과일에 닿으면서 잡았다
- 이웃 과일 접촉도 실패로 세면 전체 VLA 단독 15 · 기존 방식 14 · 새 구성 13이다. ≤ 8.2 cm 장면에서는 4 · 3 · 2다. 잠근 주 지표를 바꾸자는 것이 아니라 함께 보고하는 값이다

**4. 안전 경보 · 3D 쥠 규칙이 결과에 영향을 준 경로 (지시 2)**

- 3D 쥠 규칙: 24 장면에서 attach 결정을 바꾼 적이 없다(03:15 절). 결과 영향 0
- 안전 경보가 결과를 바꾼 장면은 ep572 하나다. 경로는 다음 순서다
  1. 이웃 바나나가 target 덩어리에 합쳐진다
  2. attach 중 robot mask가 손에 든 사과를 지운다
  3. tracker가 바나나 쪽을 쥔 물체로 보고 attach를 푼다(두 arm 공통)
  4. 경보는 기억이 이미 다른 물체에 고정돼 있어 점수가 0.17이다 → 손에 든 사과의 target 예외를 보류한다(새 구성만)
  5. TrajOpt가 손에 든 사과를 장애물로 다룬다 → 낙하
- 경보 자체의 결함(기억 고착)과 기존 방식의 결함(이웃 과일 병합, attach 중 사과가 mask에 지워짐)이 겹친 결과다
- 경보는 307 요청에서 예외를 보류했다. 그러나 다른 23 장면의 결과는 기존 방식과 같았다

**figure** (`AG3S/ag3s/docs/figures/ifg-apple/`, sidecar `.json`)

- `ifg-apple-fail-overview`: 24 장면을 이웃 과일 거리순으로 놓은 그림. (a) 세 arm 결과와 실패 유형, 이웃 과일 접촉, (b) target 덩어리 길이, (c) 사과 근처에서 위로 민 양, (d) 사과 근처 HOLD, (e) 닫을 때 손끝 높이
- `ifg-apple-fail-scenes`: 실제 장면. ep211 · 954는 닫는 순간 VLA 단독과 기존 방식, ep250은 attach 해제와 낙하 직전

**하는 말 / 하지 않는 말**

- 하는 말: AG3S만 실패한 장면의 대부분은 사과 바로 옆 과일과 관련된 서버 판단(덩어리 병합, 이웃 회피 제약, attach 해제)에서 갈렸다
- 하지 않는 말: 이 원인을 고치면 성공률이 VLA 단독 이상이 된다는 것. 320 · 1345처럼 서버 결정 하나로 설명되지 않는 갈림도 있다. 장면 수도 작다

## 2026-10-20 10:40 KST — IFG-SPLIT: 이웃 과일과 붙은 target 을 attention 으로 떼어 내기 (AG3S 핵심 수정 · offline 검증 · closed-loop 착수)

- 사용자 판정: 궤적에서 생기는 원인은 TO로, 관측·기하 원인은 AG3S 핵심을 고친다. 권장안대로 진행한다
- task(결과 전 고정): `AG3S/ag3s/docs/handoff/IFG-SPLIT.task.md`

**원인을 코드에서 확인**

- T31b split은 merge된 덩어리를 12 mm 연결 반경으로 다시 묶는다. 사과–바나나 틈이 4–12 mm라 대부분 `one_component`로 끝난다
- 가끔 split이 성공해도 사과 조각은 첫 frame에 merge된 채 잡힌 anchor의 일부(약 66 %)다. 그래서 "손에 가려 일부만 보임"(T30 F3b `subset_kept_anchor`)으로 처리되고, anchor는 바뀌지 않는다
- C1aN 서버는 이미 `--closing-keep-geometry`를 켜고 있었다. 그래서 손을 닫는 동안 merge된 anchor가 그대로 유지됐다. attach 뒤 사과는 self-filter에 지워지고, 남은 바나나 쪽이 "쥔 물체"로 관측되어 `observed_elsewhere / not_following` 해제가 났다
- 합쳐진 덩어리 안에서 점별 raw attention은 두 과일을 분명히 가른다. median이 사과 0.05–0.18, 이웃 과일 3e-5–4e-3이다(AUC 0.86–0.99)

**수정 (flag, 기본 꺼짐)**

| flag | 내용 |
|---|---|
| `--attention-split-partition` | 재군집이 한 덩어리로 끝나면 점별 log raw attention을 둘로 나눈다(Otsu, 9-NN 다수결). 가장 큰 질량의 고-attention 연결 부분이 target이다. 기존 비율 gate(2.0)를 쓰고, 최대 길이가 30 mm 이상 줄 때만 받아들인다 |
| `--split-supersedes-anchor` | 쥐기 전 split의 target 조각이 merge된 anchor의 일부면, 손 가림이 아니라 이웃이 떨어져 나간 것으로 보고 anchor를 대체한다. 조각이 anchor보다 30 mm 이상 짧을 때만이다 |

- 두 번째 조건(30 mm)은 결과를 본 뒤 더했다. 처음 구현에서 ep1313의 T31b split이 손 그늘에 잘린 사과 반쪽(113 → 59점, 72.7 → 61.7 mm)을 anchor로 받아들였기 때문이다
- 파일: `stages/attention_split.py`(`_partition`), `stages/target_grounding.py`(`SPLIT_SUPERSEDED_ANCHOR`), `config.py`(3 key), `trajopt/serve_safe.py`(2 flag, `--no-safe`면 거절)
- test: 새 `tests/ag3s/test_ifg_split_partition.py` 15개, 관련 기존 test 203개 통과(split · grounding · confirm · T44 · T54 replay)

**offline 검증 (기록 replay, CPU, `ifg_split_replay.py` → `verify/IFG-SPLIT-OFFLINE`)**

- replay 방식: T43 `ground_replay`와 같다. 기록된 attention cloud · support mask · 손 sphere · latch 단계를 넣고, C1aN 서버의 clustering key(T2 + T44A)를 모두 쓴다
- 이 replay가 기록과 75 chunk 전부 같은 장면은 24 중 23이다(상태 · 덩어리 길이 · 중심 차 0). ep979만 일부 frame이 다르다

| 첫 닫기 전 (median) | 기존 방식 | 수정 |
|---|---|---|
| merge 4 장면의 target 중 이웃 과일 점 비율 | 0.34–0.41 | 0.05–0.14 |
| merge 4 장면의 anchor 최대 길이 | 135–150 mm | 58–81 mm |
| 나머지 20 장면 | — | 18 장면 동일. ep979는 늦게 merge된 anchor 105 → 67 mm, ep325는 일부 frame만 차이 |

- ep250 · 572의 attach 중 frame(k=15 / 17): 기존 방식은 바나나를 `visible`로 관측해 해제가 났다. 수정에서는 `occluded`다(사과 형상 유지, 바나나와 7–8 cm)
- 남은 문제: 사과–이웃 경계에 이웃 과일 점이 얇은 띠로 남는다(5–18 %). attention 해상도(patch) 때문으로 보이며 확인하지 않았다. ep1345 anchor 81 mm가 그 예다
- figure: `AG3S/ag3s/docs/figures/ifg-split/ifg-split-offline`. 4 장면 위에서 본 target 점(기존 / 수정, 색 = 사과 · 이웃 과일, 배경 = log raw attention)과 24 장면 이웃 비율 막대

**closed-loop 착수**

- `verify/IFG-APPLE-SPLIT`: S2 kit 복사, arm SPL = C1aN + 두 flag, `snapshot_split`(tree ab054291)
- 같은 24 장면 · seed. CPU flag check 통과(`pre/flag_check.json`). benchmark 세션에 GPU 사용을 알렸고, launcher는 benchmark 작업이 600초 동안 없으면 GPU lock을 잡는다

## 2026-10-20 12:20 KST — IFG-SPLIT closed-loop 결과 (사과 24 장면, seed 1)

- GPU 01:42–02:55 UTC(10:42–11:55 KST), 24/24 완료, 재시도 0
- smoke 통과: SPL 첫 VLA 출력 = VLA 단독(차 1.1e-7), split 기록 있음, 경보 · 3D 쥠 기록 없음
- 분석 준비 누락 2건을 고쳐 다시 돌렸다: `agg/address_check.json` 미생성, `run_replays_cl.sh`의 run 목록이 `runs/S2`로 남아 있었음. rollout과는 무관하며 `LEAD_AMENDMENTS.md`에 적었다
- 합본: `verify/IFG-APPLE-4ARM`(E0 · C1aN · S2 · SPL 링크, agg3.py). 수치: `AG3S/ag3s/docs/handoff/IFG-SPLIT.verify.json`

**주 지표: 접촉 없는 작업 성공 (같은 장면 짝)**

| 짝 | 둘 다 | 앞만 | 뒤만 | 둘 다 실패 | 차 (뒤 − 앞) [95 % CI] |
|---|---|---|---|---|---|
| 기존 방식 → SPL | 16 | 0 | 2 | 6 | +0.083 [0.000, +0.208] |
| VLA 단독 → SPL | 15 | 3 | 3 | 3 | 0.000 [−0.208, +0.208] |
| +경보·3D쥠 → SPL | 15 | 0 | 3 | 6 | +0.125 [0.000, +0.292] |
| VLA 단독 → 기존 방식 (확인용, 이전 값과 같음) | 13 | 5 | 3 | 3 | −0.083 [−0.292, +0.125] |

| 성공 수 (괄호 = 이웃 과일 접촉도 실패로 셈) | VLA 단독 | 기존 방식 | +경보·3D쥠 | SPL |
|---|---|---|---|---|
| 이웃 과일 ≤ 8.2 cm (7) | 7 (4) | 3 (3) | 2 (2) | 5 (4) |
| 그 밖 (17) | 11 (11) | 13 (11) | 13 (11) | 13 (11) |
| 전체 (24) | 18 (15) | 16 (14) | 15 (13) | 18 (15) |

- 기존 방식 → SPL로 바뀐 장면: 실패 → 성공 2개(ep250 · ep1345), 성공 → 실패 0개
- 실행 동작이 기존 방식과 bit 단위로 같은 장면: 24 중 19. 달라진 장면은 merge되던 4 장면과 ep325다(ep325 결과는 같음)
- 운반 중 attach 해제: 기존 방식 2번(ep250 k16 · ep572 k18) → SPL 0번

**merge되던 4 장면 (첫 닫기 전)**

| 장면 | 덩어리 길이 median | 사과 근처 HOLD | 닫는 시각 | 결과 |
|---|---|---|---|---|
| ep211 (바나나 7.2 cm) | 139 → 77 mm | 34 → 2 | t=398 → 163 | 실패 → 실패 (쥐기 재시도) |
| ep250 (바나나 8.2 cm) | 150 → 85 mm | 0 → 0 | 87 → 81 | 실패(운반 중 낙하) → 성공 |
| ep572 (바나나 7.3 cm) | 142 → 82 mm | 0 → 0 | 107 → 107 | 성공 → 성공 (attach 해제 사라짐) |
| ep1345 (오렌지 7.2 cm) | 136 → 92 mm | 0 → 0 | 107 → 127 | 실패 → 성공 |

**아직 실패하는 가까운 장면 (원인 층: 궤적 / TO)**

- ep211:
  - t=163에 사과에서 7 cm 떨어져 닫았다. 닫는 무렵 바나나와의 충돌로 HOLD 3번이 났다(`nearest_is_not_manip`, 손 – 바나나 26 mm)
  - 다시 잡아 7.9 cm 들었지만 미끄러졌다. 손이 바나나에 34 row 닿았다
  - 바나나가 잡는 경로 위에 있어서 VLA 단독도 바나나에 닿으면서(8 row) 잡았다
- ep954: 덩어리는 처음부터 사과뿐이었다. TrajOpt가 바나나와 5 mm 여유를 지키느라 손이 위로 밀렸다. 이 실행은 SPL과 기존 방식이 bit 단위로 같다

**하는 말 / 하지 않는 말**

- 하는 말: 이 24 장면에서 SPL은 기존 방식이 실패한 merge 장면 2개를 성공으로 바꿨고, 성공한 장면을 하나도 잃지 않았다. 나머지 19 장면은 실행이 같다. 전체 성공 수는 VLA 단독과 같다(18 = 18, 이웃 접촉 포함 15 = 15)
- 하지 않는 말: SPL이 기존 방식보다 낫다는 일반 결론. CI 하한이 0이고, 이 24 장면은 실패 분석에 쓴 장면이다(post hoc). 다른 장면에서 확인이 필요하다

**figure** (`AG3S/ag3s/docs/figures/ifg-split/`, sidecar `.json`)

- `ifg-split-closedloop`:
  - (a) 24 장면 × 4 arm 결과, 이웃 접촉, 기존 방식과 같은 실행 표시
  - (b) 이웃 거리별 성공률(엄격 기준 함께)
  - (c) 짝 비교 표
  - (d) merge 4 장면의 덩어리 길이와 서버 판단
- `ifg-split-closedloop-scenes`: 실제 장면. ep250은 기존 방식 해제 → 낙하, SPL 같은 시각 운반 → 바구니. ep1345는 기존 방식 높게 닫음 → 실패, SPL 닫음 → 들어 올림

메모리 회수: `reclaim: ws 21.5->21.5 GiB | cache 1.1->0.0 GiB, 26623 files, 4 paths`

## 2026-10-20 17:00 KST — IFG-SPLIT 독립 장면 확인 (IFG-CL3 40 장면, 사용자 "수행하세요")

- 결과 전 고정: `IFG-SPLIT.task.md` §4
  - 비교 기준은 IFG-CL3 C1aN이다. 같은 snapshot tree f0d8046b, 같은 client 인자(경로만 다름), 같은 서버 flag를 썼다
  - 미리 정한 규칙: "성공 → 실패 장면이 SPL 판단에서 시작되면 진단하고 채택하지 않는다"
- run: `verify/IFG-CL3-SPLIT`(seeds.json byte 동일, sha256 88f6b37e…). GPU 05:39–07:35 UTC, 40/40, 재시도 0
- smoke 1차 실패: 서버가 시작을 거부했다. lead가 미리 만든 집계 폴더에 run 밖을 가리키는 symlink가 있었기 때문이다. 폴더를 지우고 그대로 다시 띄웠다. 집계 폴더는 rollout 뒤 링크 없이 복사본으로 만들었다(`IFG-CL3-SPLIT-AGG`)
- 집계 확인: 같은 script가 IFG-CL3의 C1aN 대 SURF 결과(12/5/7/16)를 그대로 재현한다
- 수치: `IFG-SPLIT.verify.json`의 `cl3_independent`. script: `ifg_split_cl3.py`

| 접촉 없는 성공 (괄호 = 이웃 과일 접촉도 실패) | 기존 방식 | SPL | 표면 영역 (참고) |
|---|---|---|---|
| 전체 40 | 17 (16) | 18 (17) | 19 (19) |
| 가장 가까운 과일 ≤ 8.2 cm (6) | 2 (2) | 3 (3) | 4 (4) |
| 그 밖 (34) | 15 (14) | 15 (14) | 15 (15) |
| clear 14 / adjacent 13 / place 13 | 5 / 7 / 5 | 6 / 7 / 5 | — |

- 짝 비교(기존 → SPL): 둘 다 16, 기존만 1, SPL만 2, 둘 다 실패 21. 차 +0.025 [−0.050, +0.100]
- 실행이 기존 방식과 bit 단위로 같은 장면은 34개다. ep1034는 NFS에 멈춘 process가 잡고 있는 폴더라 읽지 않았다. 집계 수치는 두 arm이 같다
- 실패 → 성공:
  - ep1582(오렌지, 바나나 7.6 cm): k=0부터 partition이 merge 덩어리를 145 → 69 mm로 나눴다. 의도한 효과다
  - ep1407(사과): target 형상은 두 arm이 같았다. 다른 덩어리의 partition에서 row 200부터 갈라졌다. target 분리 효과가 아니다
- **성공 → 실패: ep936 (오렌지, 기둥 옆, 바나나 8.3 cm)**
  - SPL 판단에서 시작됐다. k=7에 T31b split이 오렌지만 떼어 냈고(61 mm), anchor 대체로 merge anchor(148 mm)가 바뀌었다. target 판단 자체는 맞았다
  - 실패 경로: k=25 닫는 중 `settled` 검사 실패 → 정책이 한 chunk 안에 들어 올림 → `reach`(닫기 전 탁자 위에 유지된 형상까지 거리)가 159–273 mm로 실패 → attach가 걸리지 않음 → 손에 든 오렌지가 장애물로 남음 → TrajOpt가 최대 155 mm 밀었고 k=29에 낙하
- 같은 latch 경로(들렸는데 attach 안 됨 ≥ 2 chunk)는 기존 방식에도 있다: ep423 · ep831(두 arm 모두 실패, 실행 같음), ep1863(성공). IFG-APPLE C1aN의 ep250 · 572도 같은 상태를 거쳤다
- 미리 정한 규칙에 따라 SPL을 기본 구성으로 채택하지 않는다. 원인은 latch의 `reach`가 닫기 전 형상을 기준으로 쓰는 기존 약점이다. 이를 다루는 옵션 `--latch-reach-hand-relative`(T46g / SL5b)가 이미 있다. 가려진 대상의 마지막 관측 centroid를 손에 붙여 옮긴 점에서 reach를 잰다. C1aN 서버 구성에는 들어 있지 않다
- figure:
  - `ifg-split-cl3`: (a) 40 장면 × 3 arm, latch 미attach 표시, (b) 성공 수 표, (c) ep936의 reach · 높이 · TrajOpt 수정 시간축
  - `ifg-split-cl3-scenes`: ep1582 · ep936의 실제 장면

## 2026-10-20 18:20 KST — 사용자 규칙 3개 · 실패를 AG3S 구성 요소별로 진단 · 정렬 오류 정정 · 10 episode 진단 실행 착수

**사용자 규칙 (2026-10-20, memory에 저장)**

1. 설명과 기록에서 줄임말을 쓰지 않는다. 이 기록 앞부분의 "SPL"은 "target 분리 수정 구성"(기존 방식 + attention 분할 + 분할 조각의 기준 형상 대체)을, "C1aN"은 "기존 방식"을 뜻한다
2. 실패마다 AG3S의 어느 구성 요소 문제일 가능성이 높은지 판별한다(정의: `paper/METHOD_GUIDE_KO.md`)
   - ATS(Attention-based Target Segmentation): 목표 표면 · 추적
   - HSR(Hidden-state Sub-task Recognition): 작업 단계 · grasp 상태
   - LTE(Link-specific Target Exemption): 링크별 거리장 · 쥔 물체
   - CCR(Constrained Chunk Refinement): TrajOpt 수정 · 검증 · 멈춤
3. closed-loop 검증은 한 번에 10 episode 정도로 작게 돌린다

**지금까지의 실패를 구성 요소별로 정리**

| 실패 현상 | 장면 | 증상이 나타난 곳 | 원인 위치 (가능성 높음) | 근거 · 상태 |
|---|---|---|---|---|
| 목표와 이웃 과일이 한 덩어리로 잡힘 | 사과 211 · 250 · 572 · 1345, CL3 1582 · 936 | ATS 1단계 | ATS 1단계 (거리 기준 재군집이 4–12 mm 틈을 못 가름) | target 분리 수정으로 해결 (replay 이웃 비율 0.34–0.44 → 0.05–0.14) |
| 첫 frame의 병합 형상이 기준 형상으로 고정 | 같은 장면 | ATS 2단계 | ATS 2단계 (분할 조각을 "손에 가린 일부"로 처리) | 분할 조각의 기준 형상 대체로 해결 |
| 운반 중 attach 해제 → 낙하 | 사과 250 · 572 | HSR grasp (해제) | ATS (병합 덩어리 중 이웃 과일 쪽이 "쥔 물체"로 관측) | target 분리 수정 후 해제 0 |
| 들어 올렸는데 attach가 안 걸림 → 손에 든 물체가 장애물 → TrajOpt가 최대 155 mm 밀어 낙하 | CL3 423 · 831 (기존 · 수정 모두), 936 (수정 구성) | LTE · CCR (밀기) | HSR grasp (attach 검사의 `reach`를 닫기 전 탁자 위 형상 기준으로 잼) | 손 기준 reach(T46g)로 다시 재면 49–66 mm로 문턱 120 mm 안 → 다음 실험에서 확인 |
| 이웃 과일과 5 mm 여유를 지키느라 손이 위로 밀려 높게 쥠 | 사과 954 | CCR | CCR · LTE (이웃 과일 여유 거리 정책) | 미수정. 이웃 과일 접촉 허용 정책과 함께 정해야 함 |
| 바나나가 잡는 경로 위, 충돌 HOLD | 사과 211 (수정 후 남은 실패) | CCR | CCR (경로 위 이웃 장애물). VLA 단독도 바나나에 닿으며 잡음 | 미수정 |
| 안전 경보의 기억이 다른 물체에 고착 | 새 구성 전반 | ATS 2단계 경보 | ATS 2단계 (쥔 물체가 mask에 지워진 채 기억 갱신) | 경보는 끔 |
| 3D 쥠 규칙 | — | HSR grasp | 기존 검사와 중복, 결정 변화 0 | 끔 |
| 뚜렷한 서버 결정 없이 VLA가 다른 시점·위치에서 닫음 | 사과 320 · 1345(기존 방식) | — | CCR의 작은 수정이 closed-loop에서 커진 것 (특정 결정 없음) | 원인 하나로 특정하지 않음 |

**정정: IFG-CL3 서버 기록 정렬 오류**

- IFG-CL3 기존 방식 run 중 ep1096 · 1211 · 1529 · 1863의 replay 정렬이 다른 run의 서버 chunk를 가리킨다
  - 정렬 자체 검사에서 plan 0의 target 중심이 기록된 target 위치와 7.6–63 cm 떨어져 있다
  - 같은 서버(8293) 구간에서 NFS 문제로 다시 돌린 ep1034 뒤에 있던 run들이다
- 17:00 절의 "기존 방식 ep1863도 attach 안 됨"은 ep831의 데이터였다. 이제 그 4 run은 "판단 불가"로 뺀다(`ifg_split_cl3.py` 수정, 그림 · 수치 재생성)
- 주 지표(접촉 없는 성공)는 client 기록과 MuJoCo replay에서 나오므로 영향이 없다. target 분리 수정 구성의 정렬 40개는 모두 정상이다
- 이 정렬 오류는 이전 세션의 IFG-CL3 서버 기록 분석에도 영향을 줬을 수 있다. 확인하지 않았다

**10 episode 진단 실행 (결과 전 고정)**

- 구성: 기존 방식 + target 분리 수정(ATS) + 손 기준 reach `--latch-reach-hand-relative`(HSR, T46g 기존 기능)
- 장면 10개:
  - HSR attach 실패: CL3 423 · 831 · 936
  - ATS 병합: 사과 211 · 250 · 572 · 1345, CL3 1582
  - 회귀 확인용 대조: 사과 126, CL3 1404
- 성공률 표본이 아니라 수정이 의도대로 동작하는지 보는 진단용이다
- 비교: 같은 장면 · seed의 기존 방식과 target 분리 수정 구성(이미 있음)
- run: `verify/IFG-10EP-TARGETSPLIT-HANDREACH`. CPU flag 검사 통과. benchmark 세션에 알림

## 2026-10-20 19:30 KST — 10 episode 진단: target 분리 수정(ATS) + 손 기준 reach(HSR)

- run `verify/IFG-10EP-TARGETSPLIT-HANDREACH`, GPU 09:27–10:04 UTC(37분). smoke 통과(첫 VLA 출력 차 1.0e-7)
- 수치: `AG3S/ag3s/docs/handoff/IFG-10EP-TARGETSPLIT-HANDREACH.verify.json`, script `ifg_10ep_handreach.py`
- 세 구성 비교: 기존 방식 / + target 분리 수정 / + target 분리 수정 + 손 기준 reach. 같은 장면 · seed이고, 앞의 두 구성은 이미 있는 기록이다

| 장면 (고른 이유) | 기존 방식 | + target 분리 | + target 분리 + 손 기준 reach | 들렸는데 쥠 인정 안 된 chunk (기존 → 분리 → 분리+손) |
|---|---|---|---|---|
| 사과 250 (ATS) | 실패 (운반 중 낙하) | 성공 | 성공 | 14 → 0 → 0 |
| 사과 211 (ATS) | 실패 | 실패 (재시도) | 실패 (재시도) | 0 → 0 → 0 |
| 사과 572 (ATS) | 성공 | 성공 | 성공 | 8 → 0 → 0 |
| 사과 1345 (ATS) | 실패 | 성공 | 성공 | 0 |
| 사과 126 (대조) | 성공 | 성공 | 성공 | 0 |
| CL3 423 (HSR) | 실패 | 실패 | 실패 (들어 올린 뒤 낙하) | 8 → 8 → 1 |
| CL3 831 (HSR) | 실패 | 실패 | **성공** | 8 → 8 → 1 |
| CL3 936 (HSR) | 성공 | 실패 | 실패 (운반 중 낙하) | 0 → 4 → 1 |
| CL3 1582 (ATS) | 실패 | 성공 | 성공 | 0 |
| CL3 1404 (대조) | 성공 | 성공 | 성공 | 1 → 1 → 1 |
| **합계** | **4 / 10** | **6 / 10** | **7 / 10** | |

**구성 요소별 판정**

- **ATS (target 분리 수정):** 이웃 과일 병합 장면 5개 중 3개(250 · 1345 · 1582)를 성공으로 바꿨다. 211은 CCR 문제로 남는다(바나나가 경로 위). 936은 분리로 궤적이 달라진 뒤 HSR 문제에 걸렸다
- **HSR (손 기준 reach):** 쥠 인정 실패 3 장면 모두에서 쥠이 인정됐다(들린 뒤 미인정 chunk 8 → 1, 4 → 1). 831은 성공으로 바뀌었다. 나머지 7 장면은 target 분리만 쓴 구성과 실행이 bit 단위로 같아, 이 기능으로 인한 퇴보는 없다
- **남은 문제 1 (HSR 시점):** 들린 첫 chunk에서는 물체가 아직 "보임" 상태다(닫는 동안 유지한 탁자 위 옛 형상). 그래서 손 기준 reach가 적용되지 않고, 쥠 인정이 한 chunk 늦어 물체가 이미 17–19 cm 들린 뒤에 걸린다
- **남은 문제 2 (LTE → CCR):** 그 지연 구간과 닫는 chunk에서 손에 든 물체가 쥔 물체로도, 목표 예외로도 처리되지 않는다. 그래서 장애물이 되고, TrajOpt가 손끝을 크게 민다(831 k21 121 mm, 936 k25 130 mm, 423 k14 47 mm)
- **남은 문제 3 (LTE):** 423은 늦게 쥠이 인정된 뒤 손가락과 쥔 물체 구가 거리장의 무언가와 26–31 mm 겹친다는 판정이 계속됐다(겹친 대상이 쥔 물체가 아님). 그래서 공중에서 HOLD 35 chunk 뒤 떨어졌다
  - 쥔 물체를 거리장에서 지우는 처리가 들기 전 탁자 위치를 기준으로 되어, 손에 든 사과가 거리장에 남은 것으로 보인다
  - 확인 전이다. 거리장 기록으로 확인할 일이다

**figure** (`AG3S/ag3s/docs/figures/ifg-split/`)

- `ifg-10ep-handreach`: 10 장면 × 3 구성 결과, 831 · 936 · 423의 물체 높이 · 쥠 인정 · HOLD · TrajOpt 밀기 시간축
- `ifg-10ep-handreach-scenes`: 831 · 423 실제 장면(target 분리만 대 + 손 기준 reach, 같은 시각과 6초 뒤)

## 2026-10-20 19:35 KST — 세션 인계

- 사용자가 세션을 clear하고 이어가기로 했다. 인계 문서: [HANDOFF_20261020_KO.md](HANDOFF_20261020_KO.md)
  - 진행 중: `IFG-10EP-HSR-KEPTUNSEEN`. 같은 10 장면에 "닫는 동안 유지한 형상을 관측으로 세지 않기" 설정 하나만 더한 비교. launcher가 실행 중이다
  - 할 일 · 사용자 판정 대기 · 함정 · arm 코드 대응표를 인계 문서에 정리했다
- HSR 쥠 인정 시점 수정 구현(이 절 직전 작업):
  - `latch_reach_kept_unseen` 설정과 `--latch-reach-kept-unseen` flag를 추가했다(기본 꺼짐, `--latch-reach-hand-relative` 필요)
  - test: 새 4개, 관련 기존 91개 통과
  - snapshot: `snapshot_hsr`. snapshot_split 대비 실행 코드 차이는 4 파일(config · pipeline · safe_policy · serve_safe)이다

## 2026-10-20 20:20 KST — 10 episode 진단: 닫는 동안 유지한 형상을 관측으로 세지 않기 (HSR) · 423 원인 확인 · 1404 퇴보

- run `verify/IFG-10EP-HSR-KEPTUNSEEN`, GPU 10:32–11:10 UTC(38분). smoke 통과(첫 VLA 출력 차 1.0e-7). 끝난 뒤 benchmark 세션에 GPU 반납 알림
- 비교: 같은 10 장면 · seed(seeds.json 동일). 앞의 세 구성은 `IFG-10EP-TARGETSPLIT-HANDREACH`에 이미 있는 기록이다. 집계는 `agg4arm`(복사본, symlink 없음)에서 agg3.py로 짝 비교했다
- 수치: `AG3S/ag3s/docs/handoff/IFG-10EP-HSR-KEPTUNSEEN.verify.json`, script `ifg_10ep_keptunseen.py`
- 진단용이며 성공률 표본이 아니다

| 장면 (고른 이유) | 기존 방식 | + target 분리 | + 손 기준 쥠 거리 | + 닫는 동안 유지한 형상은 관측 아님 | 들렸는데 쥠 미인정 chunk (마지막 두 구성) |
|---|---|---|---|---|---|
| 사과 250 (ATS) | 실패 | 성공 | 성공 | 성공 | 0 → 0 |
| 사과 211 (ATS) | 실패 | 실패 | 실패 | **성공** | 0 → 0 |
| 사과 572 (ATS) | 성공 | 성공 | 성공 | 성공 | 0 → 0 |
| 사과 1345 (ATS) | 실패 | 성공 | 성공 | 성공 | 0 → 0 |
| 사과 126 (대조) | 성공 | 성공 | 성공 | 성공 | 0 → 0 |
| CL3 423 (HSR) | 실패 | 실패 | 실패 (공중 HOLD 35 → 낙하) | **성공** | 1 → 0 |
| CL3 831 (HSR) | 실패 | 실패 | 성공 | 성공 | 1 → 0 |
| CL3 936 (HSR) | 성공 | 실패 | 실패 | 실패 (운반 중 낙하) | 1 → 1 |
| CL3 1582 (ATS) | 실패 | 성공 | 성공 | 성공 | 0 → 0 |
| CL3 1404 (대조) | 성공 | 성공 | 성공 | 성공 (쥠 철회 1) | **1 → 7** |
| **합계** | **4 / 10** | **6 / 10** | **7 / 10** | **9 / 10** | |

- 짝 비교: 직전 구성 대비 실패→성공 2(211 · 423), 성공→실패 0. 기존 방식 대비 실패→성공 6, 성공→실패 1(936)

**423 원인 확인 (LTE, 원인 층은 HSR)** — 직전 구성의 서버 기록(ESDF · 쥔 물체 기록)으로 확인했다

- 공중 HOLD 동안 TrajOpt 최악 행이 걸린 거리장 점유는 손에 든 사과 자신의 표면이었다
  - 점유 voxel은 실제 사과 중심에서 21–45 mm 거리에 있다
  - 최악 행 점이 쥔 사과 위에 놓인 위반 요청은 8개다
- 쥠 인정 때 붙인 형상은 닫기 전 탁자 위의 옛 형상이다(fit 중심 z 0.850 m, 실제 사과 z 1.107 m)
  - 이 형상을 15번째 요청의 손 자세 기준으로 손에 붙였다. 15번째는 닫는 동안 옛 형상을 유지한 요청인데 "관측"으로 세어졌다
  - 그래서 손이 옮긴 사과 중심이 실제와 31 mm 어긋났다. 쥔 물체 구와 거리장에서 지우는 영역(held_free)이 실제 사과를 덮지 못했다
- 다른 9 장면은 쥠 인정 1–2 요청 뒤 사과 위치의 점유가 0이 된다. 423은 첫 6 요청에서 5–9 voxel이 남았고, 처음 2 요청 뒤에도 점유가 남은 요청이 11개다
- 새 구성에서는 중심 오차가 12 mm로 줄고 HOLD가 0이 되어 성공했다. 17번째 요청의 TrajOpt 손끝 밀기는 58 mm에서 1 mm로 줄었다

**1404 퇴보 (HSR, 결과는 성공 유지)**

- 새 구성에서 손을 따라 옮기기 시작하는 기준 자세가 44번째 요청으로 남았다. 44번째는 손이 아직 접근 중이던 마지막 "관측" frame이다
- 사과는 닫기 시작한 54번째까지 탁자 위에 있었다. 그런데 손이 접근하며 움직인 만큼 계산한 중심도 같이 옮겨져 실제와 56 mm 어긋났다(쥠 거리 99.7 mm, 직전 구성 58.0 mm)
- 그 결과 55번째에 잘못된 위치로 쥠이 인정됐고, 56번째에 "다른 곳에서 관측됨"(중심 거리 82.9 mm)으로 철회됐다. 이후 사과가 최대 24 cm 들린 채 7 chunk 동안 쥠이 인정되지 않았고 HOLD가 2번 있었다
- 실행 궤적은 55번째 요청 직전까지 두 구성이 같다

**공통 원인 (미수정, 사용자 판정 필요)**

- 물체를 손에 따라 옮기기 시작하는 자세가 두 방향으로 틀린다
  - flag 없이는 너무 늦다(423: 들어 올리는 중)
  - flag를 켜면 너무 이를 수 있다(1404: 접근 중)
- 후보 규칙: "닫기 시작한 요청의 자세부터 옮긴다". 물체는 손가락이 닫히기 전에는 손을 따라 움직이지 않는다
- 수치로 확인하지 않았다

**나머지**

- 936: 두 구성 모두 27번째 요청에서 쥠이 인정되고 운반 중 손에서 빠졌다. 닫는 chunk(25번째)의 TrajOpt 손끝 밀기 130 mm도 그대로다(LTE → CCR: 닫는 동안 손에 든 물체 처리). 이번 수정과 무관하다
- 211: 성공 전환은 이번 수정의 의도된 효과로 설명되지 않는다. 64번째 쥠 인정 시점의 TrajOpt 밀기는 두 구성 모두 8 mm다. 직전 구성에서는 65번째에 사과가 빠져 쥠이 철회됐다

**figure** (`AG3S/ag3s/docs/figures/ifg-split/`)

- `ifg-10ep-keptunseen`: 10 장면 × 4 구성 결과, 423 사과 높이 · 손이 옮긴 중심 오차 · 쥔 사과 위 점유 voxel · HOLD, 1404 쥠 인정 · 철회 · 중심 오차
- `ifg-10ep-keptunseen-scenes`: 423 · 1404 실제 장면(직전 구성 대 새 구성, 같은 시각)

## 2026-10-20 22:50 KST — 사용자 판정 · 닫기 시작한 요청부터 손과 함께 옮기기 (HSR2) 10 episode · VLA 단독 성공 → AG3S 실패 원인 추적 · 명령 대 실제 궤적

**사용자 판정 (memory 저장)**

- 모듈 시험 장면 = VLA 단독 성공 기록이 있는 장면(손 · 과일 제한 없음)
- VLA 단독으로 성공하던 장면이 AG3S를 켜서 실패하면 원인과 그때 작동한 규칙(판정 · 문턱)을 반드시 특정한다
- "왼손 사과가 성공률이 높다"는 기록으로 확인되지 않았다(사과 24 장면 VLA 단독: 왼손 9/14, 오른손 9/10)
- MuJoCo 댐핑 · 스프링 계수 때문에 명령 궤적이 실제 궤적으로 그대로 이어지지 않는다. TrajOpt 분석은 실제 궤적으로도 본다

**구현: `--latch-carry-from-closing` (HSR2, 기본 꺼짐)**

- 손 기준 쥠 거리가 관측 중심을 손과 함께 옮기기 시작하는 자세, 그리고 쥠 인정 때 형상을 붙이는 자세를 정하는 규칙이다
  - 관측 기준 frame이 닫기 시작한 요청보다 앞서면, 닫기 시작한 요청의 자세를 쓴다
  - 닫기 전에는 물체가 손을 따라 움직이지 않기 때문이다
- 파일: `config.py` · `safe_policy.py` · `serve_safe.py`
- test: `tests/trajopt/test_ifg_hsr_carry_from_closing.py` 4개 통과, 관련 기존 test 157개 통과(1개 건너뜀)

**10 episode 결과 (`verify/IFG-10EP-HSR-CARRYCLOSE`, GPU 13:00–13:41 UTC)**

- 성공: 기존 방식 4 → target 분리 6 → 손 기준 쥠 거리 7 → 닫는 동안 유지한 형상은 관측 아님 9 → **닫기 시작한 요청부터 옮김 8** / 10
- 1404(대조): 54번째 요청(닫기 시작)부터 옮겨 55번째에 쥠이 인정됐다. 철회 없이 들어 올렸고(손이 옮긴 중심 오차 19–22 mm), HOLD는 3에서 0이 됐다. 직전 실험의 퇴보가 사라졌다
- 423: 성공을 유지했다(쥔 사과 위치의 점유 0)
- 211: 성공에서 실패로 바뀌었지만 이 flag 때문은 아니다
  - target 분리 · 손 기준 쥠 거리 · 직전 구성 · 이번 구성 모두 첫 쥠이 같은 방식으로 실패한다. 닫는 chunk에서 바나나 충돌 판정을 TrajOpt가 풀지 못해 닫는 중 HOLD가 걸리고, 25번째 요청에 쥠이 철회된다
  - 결과는 그 뒤 다시 잡기에서 갈렸다
- 936: 변화 없음(닫는 chunk에서 손끝 밀기 130 mm, 운반 중 낙하)
- 수치: `AG3S/ag3s/docs/handoff/IFG-10EP-HSR-CARRYCLOSE.verify.json`
- figure: `ifg-split/ifg-10ep-carryclose`, `ifg-10ep-carryclose-scenes`

**VLA 단독 성공 → AG3S 실패 원인 (사과 24 장면, target 분리 수정 기록, `ifg_regression_trace.py`)**

| 장면 | 작동한 규칙 | 근거 |
|---|---|---|
| 954 | 손가락 link가 이웃 과일(바나나) voxel에서 5 mm 여유를 지켜야 하는 충돌 제약. look-ahead도 켜져 있었다 | 접근 12–16번째 요청에서 VLA 계획의 여유가 1–3 mm였고, TrajOpt가 6–19 mm를 앞 · 옆 · 위로 밀었다. 닫는 위치가 옆으로 20 mm, 위로 10 mm 어긋났다. VLA 단독은 바나나와 접촉 17행을 남기며 성공했다 |
| 211 | 같은 이웃 과일(바나나) 충돌 판정. 닫는 chunk에서 TrajOpt가 위반을 풀지 못해 "충돌 위반이 남음"으로 HOLD | 21번째 요청에서 −8.8 mm 위반, 22번째 요청에 HOLD, 25번째에 쥠 철회. 접근 중 look-ahead 켜짐(밀기 54 · 28 · 14 mm). VLA 단독은 바나나 접촉 8행으로 성공 |
| 320 | 닫는 chunk에서 손가락이 목표를 지운 거리장의 남은 voxel을 침범한 행을 TrajOpt가 해소. look-ahead는 꺼져 있었다 | 16번째 요청에서 초기 위반 17.5 mm, 명령 기준 155 mm 이동(로봇 쪽 119 mm). 침범한 voxel이 어느 물체인지는 미확인 |

- 원인 후보에서 뺀 것: 목표를 지운 거리장에 사과 voxel이 남는 현상은 성공 장면(126 · 572 · 1345)에서도 닫는 순간 160–530개로 똑같이 생겨, 실패를 가르지 못한다
- 처음 닫는 순간 손가락 중점의 위치(사과 24 장면 중앙값)는 VLA 단독, 기존 방식, target 분리 수정이 앞뒤 방향 1 mm 이내로 같다. AG3S가 체계적으로 앞쪽을 잡지는 않는다

**명령 대 실제 궤적 (사과 24 장면, traj.npz `executed_actions` 대 `measured_qpos`, 손끝 위치)**

- MuJoCo 설정
  - 팔은 position actuator로 kp 1500 / 2000 / 6000, dampratio 1이고, 관절 damping 30과 armature 10이 붙어 있다
  - 15 Hz로 제어해 한 목표값을 33 physics step 동안 유지한다
- 추종 오차(명령한 손끝 대 한 step 뒤 실제 손끝)
  - 중앙값 7.2 mm, p95 16 mm. VLA 단독과 target 분리 수정이 같다
  - 계통 오차: 실제가 명령보다 약 3 mm 로봇 쪽, 6 mm 아래에 있다
- 큰 수정은 걸러진다: 320 닫는 chunk에서 명령이 118 mm 점프했지만 한 step 동안 실제는 약 30 mm만 움직였다(오차 102 mm)
- TrajOpt의 여유(손가락 5 mm, 장애물 10 mm)가 추종 오차와 같은 크기다. 명령 공간에서 통과한 것이 실제 경로에서도 통과한다고 볼 수 없다

**진행 중:** `verify/IFG-APPLE-LOOKAHEADOFF`

- look-ahead(`--plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20`)만 뺀 target 분리 수정을 돌리고 있다
- 장면은 VLA 단독 성공 사과 10개다: 211 · 320 · 954 + 해시 순서 7개(56 · 1837 · 753 · 309 · 250 · 1668 · 126)
- 실험 전 예상: look-ahead가 원인이면 954 · 211이 회복되고, 320은 그대로다

## 2026-10-20 23:30 KST — look-ahead(허들 대비 TrajOpt 기능)만 끈 비교: 원인 아님

- run `verify/IFG-APPLE-LOOKAHEADOFF`, GPU 13:41–14:16 UTC
- 구성: target 분리 수정에서 `--plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20`만 뺐다. 비교 대상은 같은 seed의 기존 target 분리 수정 기록과 VLA 단독 기록이다
- 장면: VLA 단독 성공 사과 10개(결과 전 고정, 진단용)
- 실험 전 예상: look-ahead가 원인이면 954 · 211이 회복된다

| | VLA 단독 | target 분리 (look-ahead 켬) | look-ahead 끔 |
|---|---|---|---|
| 성공 / 10 | 10 | 7 (211 · 320 · 954 실패) | 7 (211 · 954 · **1837** 실패, **320** 회복) |

- **예상과 다르다.** 954 · 211은 look-ahead를 꺼도 그대로 실패한다. 이웃 과일(바나나) 여유 규칙과 맞는다
- 두 구성 모두 닫기 4 요청 전부터 2 요청 뒤 사이에 TrajOpt가 손끝을 명령 기준 10–155 mm 밀거나 HOLD를 건다
  - 목표를 지운 거리장에 남은 사과 voxel에 걸린 위반이 성공 장면에도 거의 다 있다
  - 그래서 이 10 장면에서는 미는 크기나 남은 voxel 위반만으로 성공과 실패가 갈리지 않는다
- 수치 `AG3S/ag3s/docs/handoff/IFG-APPLE-LOOKAHEADOFF.verify.json`, figure `ifg-split/ifg-lookahead-off`

## 2026-10-20 23:21 KST — IFG-CONVEX (표면 graph 에 오목 edge 절단 · attention 양쪽 확인, D1 offline): 성과 없음 — 사용자 판정으로 폐기, 코드 · 그림 · run 삭제

## 2026-10-20 23:55 KST — 954: TrajOpt가 손을 민 뒤 VLA가 잡을 지점을 옮긴다 (사용자 영상 분석 확인)

- 사용자 분석: 바나나를 피하려고 궤적이 앞으로 옮겨지면서 잡는 목표 지점까지 바뀌었다. target이 정지해 있었으므로 목표 지점은 바뀌면 안 된다
- 확인 방법: 요청마다 VLA 예측 chunk(TrajOpt 전)에서 처음 닫는 행의 손끝 위치를 t=0 사과 위치 기준으로 비교했다(`ifg_goal_drift.py`, figure `ifg-split/ifg-goal-drift-ep954`)
- VLA 단독: 잡으려는 지점이 12–18번째 요청 내내 사과 기준 약 (−3, 17, 31) mm에 머문다
- AG3S(target 분리 수정):
  - 12번째 요청까지는 같은 지점을 노린다
  - 12번째부터 TrajOpt가 바나나 여유 때문에 손을 밀었다(chunk 끝 +12 / +5 / +12 mm)
  - 허들 대응 꺾임 유지 항(`--w-deflection-rate 20`, T43 TA-B) 때문에 13–16번째 요청의 첫 행에도 꺾임이 이어졌다(10 → 7 → 3 mm)
  - 그 사이 VLA가 잡을 지점을 손을 따라 옮겼다(16번째 요청에서 (13, 39, 30) mm)
- 닫는 chunk(17–18번째 요청)에서 TrajOpt 꺾임은 0이다. 실제 닫은 손끝은 VLA 단독보다 앞 16, 옆 23, 위 11 mm 어긋났다
- TrajOpt 비용식에는 VLA 관절값 추종 · 부드러움 · 연속성 · 충돌 벌점만 있다. 정지한 target의 목표 지점으로 돌아오게 하는 항이 없다

## 2026-10-21 00:10 KST — 사용자 정정 · B(쥐기 전 꺾임 유지 항 끄기) 구현과 실행 착수

**정정 (사용자 판정):** 23:55 절에서 "목표를 옮긴 직접 주체는 VLA"라고 쓴 것은 잘못된 표현이다

- VLA 단독은 바나나에 닿으면서 제자리에서 잡았다
- AG3S에서 손을 앞 · 옆으로 옮긴 것은 TrajOpt의 궤적 수정이다
- VLA는 옮겨진 손을 따라갔을 뿐이다
- 따라서 실패 원인은 TrajOpt(CCR)의 궤적 수정이다. 정지한 target이므로 목표 지점은 바뀌면 안 된다

**사용자 판정:** B로 원인을 확인한 뒤 A(목표 지점 고정)로 넘어간다

**구현: `--deflection-rate-after-grasp` (IFG-TOB, 기본 꺼짐)**

- 쥠 상태가 탐색 · 목표 고정 · 닫는 중인 chunk에서는 꺾임 유지 항(T43 TA-B)과 직전 꺾임 이력을 빼고 푼다
- 쥔 뒤(운반 · 놓기 · 복귀)는 그대로 둔다. 허들은 운반 경로에 있기 때문이다
- 파일: trajopt `config.py` · `refiner.py` · `safe_policy.py` · `serve_safe.py`
- test: `tests/trajopt/test_ifg_tob_deflection_rate_after_grasp.py` 4개 통과, 관련 기존 207개 통과(3개 건너뜀)

**실행: `verify/IFG-APPLE-TOB`**

- 구성: target 분리 수정 + 새 flag. look-ahead를 포함한 나머지 flag는 그대로다
- 장면: VLA 단독 성공 사과 10개(look-ahead 비교 실험과 같다). 비교 대상은 기존 target 분리 수정 기록이다
- snapshot: `snapshot_tob`(tree 3b9b3a68)
- 실험 전 예상: 쥐기 전에는 다음 chunk로 이어지는 꺾임이 0이 되고, 954에서 잡으려는 지점이 덜 흘러간다. 바나나 여유 때문에 미는 것 자체는 남는다

## 2026-10-21 01:10 KST — B 결과: 쥐기 전 꺾임 유지 항을 꺼도 954의 목표 지점은 똑같이 흘러간다

- run `verify/IFG-APPLE-TOB`, GPU 15:16–16:02 UTC
- 성공 / 10: VLA 단독 10, target 분리 7, look-ahead 끔 7, **쥐기 전 꺾임 유지 항 끔 7**
  - 954 · 211 실패는 그대로다. 320은 회복, 1668은 새로 실패(not_placed)했다
- 954 메커니즘(`ifg-goal-drift-ep954-deflection-rate-after-grasp`)
  - 다음 chunk 첫 행으로 이어지는 꺾임(13 · 14 · 15번째 요청, 앞 / 옆 / 위 mm)은 (10, 5, 10) · (10, 6, 11) · (7, 4, 8)에서 (5, 4, 7) · (4, 0, 4) · (0, −1, 0)으로 줄었다. 0이 되지 않은 것은 연속성 항(`w_continuity` 0.5, 직전 수정 꼬리 기준) 때문으로 보인다(미확인)
  - 12번째 요청 chunk 끝의 밀기는 24 mm로 오히려 커졌다
  - 닫은 손끝은 사과 기준 (13, 36, 22) mm로 기존(12, 38, 23)과 같다. VLA 단독은 (−4, 15, 12)다
- 결론
  - 목표 지점을 옮기는 것은 이월 항이 아니다. 한 chunk 안의 밀기만으로 손이 옮겨지고, 정책이 그 손에서 다시 계획한다
  - VLA 단독은 954에서 바나나에 닿으면서(8.2–10.4 s, 닫기 9.6 s) 잡는다. 손가락이 이웃 과일에서 5 mm 여유를 지키는 한, 어떤 TrajOpt 계획도 그 잡는 자리에 닿을 수 없다
  - 따라서 A(목표 지점 고정)는 목표 근처에서 이웃 과일 여유를 어떻게 다룰지와 함께 정해야 한다

## 2026-10-21 01:40 KST — 954 영상 기반 사용자 질문 두 가지: 자세 변화 없는 회피, 쥐지 못한 뒤의 머뭇거림

- figure `ifg-split/ifg-954-pose-hesitation`, script `ifg_954_pose_hesitation.py`

**질문 1 (사용자 관찰: 바나나를 피하려면 궤적보다 grasp 자세가 바뀌어야 한다)** — 측정이 관찰과 맞다

- 12–16번째 요청의 TrajOpt 수정(명령 기준, 실행 8행 최대)은 이동 6–19 mm, 회전 0.5–1.7°다. 거의 평행 이동이다
- 닫는 순간 그리퍼 방향은 VLA 단독과 6.6°만 다르다
- TrajOpt 비용식은 VLA 관절값을 모든 관절 같은 가중으로 추종한다. 손목을 돌려 손가락을 사과와 바나나 사이에 넣는 선택지를 우대하는 항이 없다

**질문 2 (쥐지 못한 뒤 바구니로 가며 머뭇거림)** — 측정한 것

- 바구니로 가는 것 자체는 VLA의 계획이다. VLA 계획의 15번째 행은 VLA 단독과 같은 방식으로 바구니에 다가간다
- 쥐지 못한 뒤(20–26번째 요청) 요청 사이 VLA 계획 변화는 9–52 mm다. VLA 단독(쥠)은 같은 구간에서 3–11 mm다
- 같은 구간 HSR p(pick)은 0.13 → 0.09 → 0.34 → 0.29 → 0.17 → 0.45 → 0.16으로 흔들린다
- TrajOpt는 같은 구간에서 손을 2–20 mm 민다. VLA 원래 계획의 장애물 여유는 42–53 mm라 충돌 회피 때문이 아니다
  - 꺾임 유지 항을 쥐기 전에 끈 B 실험에서도 7–17 mm가 남는다. 직전 chunk 꼬리와 맞추는 연속성 항이 VLA의 계획 변화를 직전 계획 쪽으로 당기는 것으로 보인다(미확인)
- 성공 운반(250)은 요청 사이 VLA 계획 변화 중앙값 6.5 mm, TrajOpt 밀기 1.7 mm다
- 쥠 판단(latch)은 운반 구간 내내 closing으로 쥠을 인정하지 않았다. 판단 자체는 옳다

## 2026-10-21 02:40 KST — A: 잡을 지점 고정(IFG-TOA) 구현 · 실행 착수 · 실패 회복 문헌 조사

**사용자 판정:** 잡을 지점은 고정한다. 허용 범위 안에서 모든 관절이 움직여도 된다(이 문제는 손목 회전으로도 가능할 것으로 봄)

**구현: `--grasp-anchor W` (IFG-TOA, 기본 꺼짐)**

- 새 모듈 `AG3S/trajopt/grasp_anchor.py`
- 적용 조건: 쥐기 전(latch latched · closing)이고, 손바닥(hold link 원점)이 고정된 목표에서 150 mm 안에 있는 chunk
- 손바닥에 두 비용을 더한다
  1. 모든 계획 행에서 정책의 손바닥 경로를 따라간다. 평행 이동이 비싸지고 자세 변경은 싸다
  2. 정책이 닫는 행부터 손바닥을 목표에 둔다
- 목표: 정책 chunk(TrajOpt 전)가 처음 닫기를 계획한 요청에서, 그 행의 손바닥 위치로 고정한다. 단 target 중심에서 120 mm 이내일 때만이다. 쥔 뒤 · 철회 · 대상 변경 때 해제한다
- 두 항 모두 SQP 반복마다 손바닥 Jacobian으로 선형화해 QP에 넣고, 단계 수락 판정(merit)에도 같은 항을 정확히 넣는다. 관절은 모두 자유이고 제약이 아닌 비용이다
- 가중치 주의
  - 고정 항은 이탈의 제곱, 충돌 벌점은 선형이다(1000 / m)
  - W = 2000이면 이탈 25 cm까지는 충돌이 우선한다
  - test fixture에서 W = 20000은 12 mm 침범을 남겼다
- test: `tests/trajopt/test_ifg_toa_grasp_anchor.py` 7개 통과, 관련 기존 299개 통과(3개 건너뜀)
  - fixture: 장애물에 걸린 구보다 안쪽 link 원점을 고정하자 정책 경로 이탈이 21.5 mm에서 3.9 mm로 줄었다. 충돌은 해소됐다

**실행: `verify/IFG-APPLE-TOA`** (target 분리 수정 + `--grasp-anchor 2000`, 같은 VLA 단독 성공 사과 10 장면, snapshot_toa tree d2326902)

- 실험 전 예상: 954에서 TrajOpt 수정이 주로 회전이 되고(손바닥 이동이 기존 6–19 mm보다 작음), 닫는 손바닥이 목표 근처에 머문다. 성공 여부는 미정이다(VLA 단독은 바나나에 닿으며 잡는다)

**문헌 조사 (사용자 지시, literature-analyst sub-agent):** [FAILURE_RECOVERY_SURVEY_KO.md](FAILURE_RECOVERY_SURVEY_KO.md)

- 가장 가까운 선행연구
  - ProbeAct(2606.09740): hidden-state probe + kinematic 빈 grasp 탐지 → 회복. probe는 학습한다
  - Sheng et al.(2610.06921): frozen π0.5, 실패 감지 시 몇 초 전 자세로 후퇴 → 정책 재질의. training-free이고 학습한 corrector와 차이가 검출되지 않았다
- 그 밖: CoRe(anchor rollback, π0.5 실물), Sentinel(탐지만), BID · FOREWARN(샘플 선택)
- 수치는 각 논문 보고값이며 재현하지 않았다

## 2026-10-21 03:10 KST — IFG-APPLE-TOA 중단 · 목표 행 수정 · IFG-APPLE-TOA2 재실행

- smoke 장면(56) 서버 기록으로 고정 항이 실제로 켜지는 것을 확인했다. 목표 고정은 7번째 요청(닫는 행 45, target에서 104 mm)이었고, 고정 항이 켜진 chunk는 7개였다
- 결함을 발견했다: 목표 행이 닫는 행부터 chunk 끝까지 전부였다
  - 이미 닫힌 상태로 시작하는 chunk(13번째 요청, 닫는 행 0)에서는 들어 올리는 행까지 목표에 붙잡도록 되어 있었다(목표 오차 117.7 mm)
  - 이번에는 충돌이 없어 TrajOpt가 손대지 않았다. 하지만 닫는 순간 충돌이 있는 장면(954 · 320)에서는 들어 올리는 손을 끌어내릴 수 있다
- 조치
  - 결과가 오염되기 전에 lead가 자기 launcher process group만 종료했다(STOPPED 파일에 사유 기록, GPU 비움)
  - 목표 행을 닫는 행부터 3행(`cost.grasp_goal_rows`)으로 줄이고, 닫는 행이 0이면 목표 항을 쓰지 않게 했다. test 146개 통과
  - snapshot_toa2(tree cba727f0)로 `verify/IFG-APPLE-TOA2`를 재실행했다. 구성 · 장면 · flag는 같다

## 2026-10-21 04:20 KST — IFG-APPLE-TOA2 결과: 954는 회전 회피가 됐지만 실패, 211 · 1668은 HOLD 고착 → 안전 우선 대체 풀이 추가

- run `verify/IFG-APPLE-TOA2`, GPU 18:14–19:05 UTC. 성공 / 10: VLA 단독 10, target 분리 7, **잡을 지점 고정 7**
  - 320 회복, 1668 새 실패, 954 · 211 실패 유지
- 수치: `AG3S/ag3s/docs/handoff/IFG-APPLE-TOA2.verify.json`

**954: 의도한 메커니즘은 작동했다**

- 고정 항은 10–16번째 요청에서 켜졌다
- 12–14번째 요청의 TrajOpt 수정
  - 손끝 이동 5–16 mm, 회전 6–14°. 손바닥의 정책 경로 이탈은 ≤ 4 mm다
  - target 분리 수정만 쓴 구성은 이동 16–19 mm에 회전 ≤ 1.7°였다
- 처음 닫는 순간 손끝 위치(사과 기준 앞 / 옆 / 위 mm)
  - 고정: (5, 15, 25)
  - target 분리: (13, 38, 23)
  - VLA 단독: (−4, 15, 12)
  - 옆 방향 밀림은 사라졌다
- 그래도 쥐지 못했다. 정책이 2초 일찍(113 step, VLA 단독 144) 닫았고, VLA 단독보다 13 mm 높았다

**211 · 1668: 설계 결함**

- 211은 11번째 요청부터, 1668은 12–22번째 요청에서 고정 항을 켠 풀이가 "violated"(침범 0.7–24 mm)로 끝났다. HOLD가 반복됐고, 211은 약 50 chunk 연속이다
- 211의 17번째 요청: SQP 후보 3개가 모두 침범을 거의 줄이지 못했고(42.9 → 39.6–43.5 mm), 고정 비용만 20–30 늘어 모두 기각됐다. TrajOpt는 정책 계획을 그대로 반환했다
- 원인
  - 고정 항은 계획 16행의 합인데, 충돌 값은 선형 slack 하나다
  - 평행 이동으로만 피할 수 있는 장애물에서 고정 항이 그 이동을 막는다
  - 앞 절에서 "W = 2000이면 이탈 25 cm까지 충돌이 우선한다"고 쓴 계산은 행 수를 빠뜨려 틀렸다
- 같은 장면에서 고정 항이 없으면 손을 54 mm 옮겨 해소했었다

**수정 (안전 우선)**

- `TrajectoryOptimizer.solve`: 고정 항을 켠 해가 침범을 허용 오차(`safety.violation_tolerance`) 안으로 해소하지 못하면, 고정 항 없이 다시 풀고 침범이 더 작은 쪽을 쓴다
- 기록: `to.metrics.grasp_anchor.fallback`
- test 추가: fixture에서 W = 20000이 남긴 12.4 mm 침범이 대체 풀이로 허용 오차 안이 된다. 관련 test 195개 통과(3개 건너뜀)
- 재실행: `verify/IFG-APPLE-TOA3`

## 2026-10-21 05:30 KST — IFG-APPLE-TOA3 결과: 안전 우선 대체 풀이도 211 교착은 못 푼다 · TrajOpt 변형 다섯 가지 모두 7 / 10

- run `verify/IFG-APPLE-TOA3`, GPU 19:24–20:18 UTC(T63 세션 뒤에 lock 대기). 성공 7 / 10(320 회복, 1668 · 211 · 954 실패)
- 수치 `AG3S/ag3s/docs/handoff/IFG-APPLE-TOA3.verify.json`, figure `ifg-split/ifg-to-variants`

**대체 풀이**

- 211에서 51번 실행됐다(고정 해 유지 36, 고정 없는 해 채택 15)
- 그러나 12번째 요청부터는 고정 없이 풀어도 침범이 19–27 mm 남는다. 두 풀이의 침범이 똑같은 요청이 많다. 둘 다 후보를 받아들이지 못하고 정책 계획을 그대로 돌려줬다는 뜻이다
- 경과
  - 고정 항은 손바닥이 목표에서 95 mm일 때(10번째 요청)부터 켜져, 손을 정책 경로에 붙잡아 둔다
  - 그 결과 손이 3번의 SQP 반복으로는 빠져나올 수 없는 자세까지 들어갔다. 이후 HOLD → 같은 계획 → HOLD가 반복됐다(HOLD 49 chunk)
  - 고정 항이 없던 target 분리 수정은 14번째 요청에서 손을 54 mm 옮겨 그 자세를 피했었다
- 1668도 12–13번째 요청에서 두 풀이의 침범이 같았고(10.8 · 8.2 mm), 쥐지 못했다
- 954에서는 대체 풀이가 필요 없었다. 실패 원인(일찍 · 높게 닫음)은 고정 항과 무관하다

**TrajOpt 변형 비교 (VLA 단독 성공 사과 10 장면)**

| VLA 단독 | target 분리 | look-ahead 끔 | 쥐기 전 꺾임 유지 항 끔 | 잡을 지점 고정 | 고정 + 안전 우선 대체 |
|---|---|---|---|---|---|
| 10 | 7 | 7 | 7 | 7 | 7 |

- 실패하는 장면은 바뀌지만 수는 그대로다
- 바구니 옆 장애물이 있고 이웃 바나나가 붙은 211 · 954는 어느 변형에서도 성공하지 못했다

## 2026-10-21 14:35 KST — 2번 교착 · 빈 손 회복(IFG-REC) 구현 · 실행 착수, 관절 가중치 아이디어(사용자)

**구현: `--recovery-retreat` (IFG-REC, 기본 꺼짐, SafePolicy)**

- 발동 조건(쥐기 전 latched · closing에서만)
  - 교착: 직전 chunk가 충돌 판정으로 8 step 전부 HOLD된 요청이 3번 연속(211 유형)
  - 빈 손: 닫는 중 쥠 근거가 "손가락이 멈춤(settled)인데 막히지 않음(blocked False)"인 요청이 2번 연속(954 유형)
- 동작
  - 다음 요청부터 VLA chunk를 후퇴 chunk로 바꾼다. 그리퍼를 열고, 관절을 행당 0.03 rad 이하로 움직인다
  - 후퇴 목표는 손바닥이 target에서 120 mm 이상 떨어져 있던 가장 최근 자세다
  - 후퇴 chunk도 TrajOpt를 거친다. 도착(0.02 rad 이내)하거나 8 요청이 지나면 VLA에 다시 맡긴다
- 제한: episode당 최대 2회. 기록: 각 chunk의 `grasp.recovery`
- test: `tests/trajopt/test_ifg_rec_recovery.py` 6개 통과, 관련 기존 227개 통과(3개 건너뜀)
- 실행: `verify/IFG-APPLE-REC`(target 분리 수정 + 이 flag 하나, 같은 VLA 단독 성공 사과 10 장면, snapshot_rec tree d63c7292). T63 세션 뒤에 GPU lock을 기다린다
- 실험 전 예상: 211(교착)과 954(빈 손)에서 후퇴가 일어난다. 재시도가 성공할지는 미정이다

**사용자 아이디어 (관절 가중치, 아직 구현 안 함)**

- 내용: 앞쪽 관절(1–3)은 위치를, 뒤쪽 관절(4–6)은 방향을 주로 바꾼다. 그러니 뒤쪽 관절의 수정량을 늘리도록 가중하면, 위치 변화는 줄고 방향 변화로 피하게 할 수 있다
- 사용자 우려: 허들을 넘거나 궤적 자체를 바꿔야 하는 경우도 있다

## 2026-10-21 15:45 KST — 교착 · 빈 손 후퇴 · 재시도(IFG-REC) 결과, 손목 관절 가중치(IFG-TOW) 구현 · 실행 착수

**IFG-APPLE-REC 결과** (`verify/IFG-APPLE-REC`, GPU 05:42–06:38 UTC, smoke 통과, 재실행 0)

- 수치: `AG3S/ag3s/docs/handoff/IFG-APPLE-REC.verify.json`, 그림 `AG3S/ag3s/docs/figures/ifg-split/ifg-rec-recovery.png`
- 성공: VLA 단독 10, target 분리 7, + 후퇴 · 재시도 7. target 분리와 짝 비교하면 바뀐 장면 0

| 확인 항목 | 결과 |
|---|---|
| 성공 7 장면에서 오경보 | 0 (발동 없음) |
| 교착 발동 (HOLD 3번 연속) | 0. 211의 최대 연속 HOLD는 2 |
| 빈 손 발동 | 실패 3 장면 모두: 211 2회, 954 2회, 320 1회 |
| 발동 때 손바닥–target 거리 | 211 316 · 373 mm, 954 311 · 315 mm, 320 171 mm |
| 후퇴 목표 손바닥–target 거리 | 160–381 mm (지금 자세이거나 1 요청 전 자세) |
| 재시도 때 닫기 시작한 손바닥–target 거리 | 954 156 · 181 mm, 211 180 · 269 mm, 320 102–131 mm |
| 비교: 성공 7 장면의 닫기 시작 거리 | 101–107 mm |

- 원인 1 — 탐지가 늦다: "손가락이 멈췄는데 막히지 않음"이 2번 연속일 때까지 기다리는 동안, VLA는 빈 손을 이미 30 cm 들어 올렸다
- 원인 2 — 후퇴 목표가 틀렸다: "지금부터 거슬러 올라가 손바닥이 120 mm 이상 떨어진 가장 최근 자세"가 곧 지금 자세였다. 그래서 후퇴가 거의 제자리였다
- 관찰: VLA는 후퇴 뒤 다시 다가가 다시 닫았다. 그러나 사과보다 50–170 mm 앞에서 닫았다(954 · 211)
- 다음 후보(미구현)
  - 닫는 순간 탐지: 손가락이 닫힘 한계까지 갔는데 막히지 않으면 바로 발동
  - 닫기 시작 이전 자세로 후퇴: 닫기 시작한 시점부터 거슬러 찾는다
  - 이렇게 해도 재시도가 제자리에서 닫을지는 모른다

**IFG-TOW 구현 (사용자 아이디어: 뒤쪽 관절 수정량을 늘려 위치 대신 방향으로 피하기)**

- `--wrist-track-scale S` (기본 꺼짐)
  - 쥐기 전(latched · closing)이고 손바닥이 target 중심 150 mm 안일 때만 켜진다
  - 쥐는 팔의 4 · 5 · 6번 관절은 VLA 경로를 따르는 비용에 S를 곱한다
  - 이 가중치는 QP와 해 채택 판단(merit), look-ahead 경로에 똑같이 들어간다
- 허들 · 150 mm 밖 접근 · 운반 구간은 지금처럼 균일 가중치다(사용자 우려 반영). 앞쪽 관절도 잠그지 않는다
- 근거(954 관절별 분석): 관절 1도당 손끝 이동은 0 · 1번 12.6 mm, 2번 3.5–7, 3번 7.6, 4번 3.5, 5번 4.4, 6번 0 mm. target 분리 수정은 주로 0번과 3번으로 피했다
- test `tests/trajopt/test_ifg_tow_wrist_track.py` 4개 통과, 관련 106개 통과
  - test 장면에서 S = 0.1이면 5번 관절 수정량이 0.33°에서 2.68°로 늘었다. 충돌은 두 경우 모두 0 mm로 해결됐다
- 실행: `verify/IFG-APPLE-TOW` (SPL + `--wrist-track-scale 0.1`, 같은 10 장면, snapshot_tow tree 45bdd832). GPU lock 06:38 UTC 획득. T63 세션에 순서를 알렸다
- 실험 전 예상
  - 954: 바나나를 피하는 수정이 0 · 3번에서 4 · 5번으로 옮겨가고, 손바닥 이동이 줄어든다
  - 211: 닫는 chunk에서 바나나와 충돌해 멈추는 문제여서 바뀌지 않는다
  - 나머지 8 장면: 그대로다

## 2026-10-21 20:00 KST — 손목 가중치(IFG-TOW) 결과, B(손목 회전만) 불가 확인, A(이웃 접촉 허용, IFG-NBC) 결과

- 수치: `AG3S/ag3s/docs/handoff/IFG-APPLE-NBC.verify.json`
- 그림: `AG3S/ag3s/docs/figures/ifg-split/ifg-nbc-result.png`, `ifg-roll-feasibility-ep954.png` · `-ep211.png`, `ifg-goal-drift-ep{954,211}-{wrist-track,neighbour-contact}.png`
- 같은 VLA 단독 성공 사과 10 장면 · seed. 재실행 0, smoke 통과

| 장면 | VLA 단독 | target 분리 | + 손목 가중치 | + 이웃 접촉 20 mm |
|---|---|---|---|---|
| 성공 수 | 10 | 7 | 8 | **9** |
| 954 (바나나 옆) | 성공, 접촉 17 row | 실패 | 실패 | **성공, 접촉 8 row** |
| 211 (바나나 옆) | 성공, 접촉 8 row | 실패 | 실패 | 실패, 접촉 14 row |
| 320 | 성공 | 실패 | **성공** | 성공 |

- 짝 비교: 이웃 접촉 구성은 손목 가중치 대비 실패→성공 1(954), 성공→실패 0. target 분리 대비 실패→성공 2(320 · 954), 성공→실패 0
- 다른 7 장면은 네 구성 모두 이웃 과일 접촉 0

**손목 가중치 (IFG-TOW, `--wrist-track-scale 0.1`)**

- 320: target 분리는 앞쪽 관절 0 · 1번을 9.7° · 8.8° 고쳤다. 손목 가중치에서는 모든 관절 수정이 1.2° 이하로 줄었고 성공했다
- 954: 손목 수정 합계는 29.3°에서 65.7°로 늘었지만, 앞쪽 수정은 줄지 않았다(29.9°에서 33.7°). 닫을 때 손끝이 사과 기준 (0.5, 37.2, 21.9) mm로, VLA 단독(−3.8, 14.7, 12.3)보다 옆으로 23 mm, 위로 10 mm 밀려 실패했다

**B — 이웃 규칙은 유지하고 접근 방향만 바꾸기: 불가** (사용자 판정 "B 한번 해보고 안 되면 A")

- VLA 단독이 닫던 자세에서 마지막 손목 관절을 −90°~+90° 돌려 보고, 기록된 거리장에서 손 · 손가락 구 거리를 쟀다(`ifg_roll_feasibility.py`). 요구 여유는 10 mm다
- 954: 회전 없이 −6.6 mm(손가락이 바나나 침범). 가장 좋은 −50°에서도 −1.8 mm(탁자 · 사과 아랫면 쪽)
- 211: 닫기 시작할 때 +10°에서 +0.7 mm가 최대다. 15 step 뒤 손가락이 닫히면 모든 각도에서 −6.5 mm 이하
- 그래서 잡을 지점을 유지한 채 이웃 여유를 지키는 자세가 없다. closed-loop 실험은 하지 않았다

**A — 이웃 접촉 허용 (IFG-NBC, `--neighbour-contact-mm 20`, 기본 꺼짐)**

- 동작 범위
  - 쥐기 전(latched · closing)이고 손바닥이 target 150 mm 안일 때만 켜진다
  - 쥐는 손의 손가락 구가 target 120 mm 안에서 이웃 장애물(target · 지지면 · 쥔 물체가 아닌 것)과 겹치는 것을 20 mm까지 허용한다
  - 손바닥 · 팔 · 탁자 · 운반 구간은 그대로다
- 구현: `SceneSnapshot.neighbour_contact`, `CollisionLinearizer` margin 규칙, `SafePolicy._neighbour_contact_gate`
- test `tests/trajopt/test_ifg_nbc_neighbour_contact.py` 3개 통과
  - trajopt 전체 91개 파일 중 89개 통과, 2개 건너뜀
  - `test_t43t_replay.py` bit-identity 1개 실패: 깨끗한 HEAD worktree에서도 똑같이 실패한다. 옛 commit(682976e)의 기준 파일과 비교하는 기존 실패다
- 954: VLA 단독과 같은 자리(−7.3, 16.2, 11.4 mm)에서 닫고 성공했다. 이웃 과일 접촉은 17 row에서 8 row로 VLA 단독보다 적다
- 211은 새 실패 유형이다
  - 손이 VLA 단독보다 약 2초 일찍 사과에 닿는다(t=143, VLA 단독은 닫는 t=184 근처)
  - 손이 사과를 약 70 mm 지나친 뒤 t=163에 닫았다. 손가락은 개도 0.52에서 멈췄다(바나나 접촉 t=165부터)
  - 22번째 요청에서 쥠 인정, 23번째에 철회됐다(개도 감소 · 막힘 없음)
  - 경로는 이웃 접촉 규칙이 켜지기 전(손바닥 163 mm, 요청 12–13의 HOLD)부터 달라진다. 일찍 도착하는 원인은 아직 분리하지 못했다

## 2026-10-21 23:30 KST — 관측 모듈 점검 (ATS · HSR · LTE, GT 비교)

- 사용자 지시: TrajOpt 문제는 CCR로 남기고, 관측에서 할 수 있는 최선을 찾는다. 분할 능력, 작업 단계 기억, attention이 바뀌어도 기억하는 능력, 3D 기하 활용, 거리장 대 GT 거리, 단계별 링크 처리를 파악한다
- 데이터: `IFG-APPLE-SPLIT`(사과 24 장면, 지금 serving 구성) 요청 1,800개 전부. GT는 각 요청의 MuJoCo 상태
- 측정: sub-agent 3개(ATS · HSR · LTE)가 병렬로 했다. 실제 씬 gallery와 링크 표는 lead가 만들었다
- 수치: `AG3S/ag3s/docs/handoff/IFG-AUDIT-ATS.verify.json`, `IFG-AUDIT-HSR.verify.json`, `IFG-AUDIT-LTE.verify.json`
- 그림 · 핵심 수치 표 · 개선 제안: `AG3S/ag3s/docs/figures/ifg-audit/README.md`

| 능력 | 결과 |
|---|---|
| 분할 (ATS) | 보일 때 precision 1.00 · IoU 0.93(접근). 닫는 중 IoU 0.36(이웃 있으면 0.20). 중심이 보이는 면 쪽으로 17.9 mm 치우침 |
| 작업 단계 (HSR latch) | 정확도 0.965. 쥔 상태 recall 1.000, 쥔 동안 잘못된 철회 0 |
| 작업 단계 (VLA probe) | label 정확도 0.843. 빈손 실패 5 run 중 4 run에서 "place"로 넘어감 |
| attention 변화 대 기억 | 다른 물체로 바뀐 적 0. 그러나 가려지면 기억이 옛 world 위치에 멈춤(568 요청 중 302) |
| 3D 기하 (LTE) | 사과 제외 recall 0.79(접근) / 0.59(닫는 중). 남은 사과 voxel 251 / 517. 이웃 과일 오제거 약 2% |
| 거리장 대 GT | 고운 층은 정확함(5 mm 넘게 멀게 봄 0.06%). 거친 층은 +6.0 mm, 55%가 5 mm 넘게 멀게 봄 |
| 링크 처리 | 단계별 예외는 의도대로 동작. 충돌 검사 대상은 손 링크 6개뿐(팔 미검사) |

- 개선 제안 1–5는 README의 "개선할 부분" 절에 있다. 사용자 판정을 기다린다(권장: 1 · 2를 함께)
- 정의 주의: gallery의 "상위 attention 비율"(≥ 0.5인 점 중 사과 위, 약 0.2)과 ATS 점검의 attention 비율(물체 간 정규화, 접근 중 0.63)은 정의가 다르다

## 2026-10-21 23:57 KST — 사용자 판정: 관측 개선 순서

- 근거: 위 절의 점검 결과와 `AG3S/ag3s/docs/figures/ifg-audit/README.md` "개선할 부분"
- README 원안은 1 · 2를 함께 하는 것이었다. 사용자가 고른 순서는 구 맞춤을 먼저 하고 기억 이동을 그 위에 얹는 것이다

| 순서 | README 번호 | 내용 | 구성 요소 |
|---|---|---|---|
| 1 | 2 | 보이는 점의 평균 대신 표면에 맞춘 구로 target 중심 · 크기 · 제외 영역을 잡는다 | ATS → LTE |
| 2 | 1 | 손이 닿은 뒤 기억한 형상을 손 기준으로 옮기고, 비어 있는 것이 관측된 옛 자리는 무효화한다. 순서 1과 한 STEP으로 묶되 ablation(구 맞춤만 / 구 맞춤 + 기억 이동)으로 나눠 본다 | ATS · LTE |
| 3 | 3 | 거친 층의 거리 과대평가 보정. 성공률 변화를 따로 보려고 분리한다 | LTE |
| 4 | 4 | latch에 빈손 상태를 두고 probe와 묶는다. 회복(IFG-REC) 재실험 때 같이 한다 | HSR |
| 5 | 5 | 팔 링크 미검사. 관측 범위 밖이라 기록만 한다 | 링크 설정 |

- 착수 전 확인 (CPU offline, 아직 하지 않음)
  - 순서 1 착수 전: 211 · 954 · 320 실패 장면에서 TrajOpt 제약을 켠 voxel이 남은 사과인지 이웃 바나나인지 나눠 센다. 211의 "일찍 도착" 원인(손바닥 163 mm, 요청 12–13 HOLD)도 함께 본다
  - 순서 3 착수 전: SPLIT 기록에서 바구니와 실제로 접촉한 사례를 센다. 접촉이 있으면 순서 3을 앞으로 올린다
- 검증 방식: SPLIT 24 장면에서 GT와 비교하는 offline 측정 → VLA 단독 성공 사과 10 장면 closed-loop 짝 비교(기준: 이웃 접촉 20 mm 구성 9 / 10)

## 2026-10-22 00:36 KST — 착수 전 확인 결과 (IFG-PRECHECK A · B, CPU offline)

- task: `AG3S/ag3s/docs/handoff/IFG-PRECHECK.task.md` (결과 전 고정). 수치: `IFG-PRECHECK-A.verify.json` · `IFG-PRECHECK-B.verify.json`. 그림: `AG3S/ag3s/docs/figures/ifg-precheck/`
- 데이터: `IFG-APPLE-SPLIT` 사과 24 장면 (A4는 `IFG-APPLE-NBC`도). 두 측정 모두 회귀 기준선 일치(legacy 14/15, has_target 15/15)
- 재구성 충실도: A는 측정 요청 530개 중 528개가 기록된 TrajOpt 여유와 1 mm 안에서 일치했다(나머지 2개는 target이 가려진 요청). B는 24 장면 모두 qpos 불일치 0

**A — 쥐기 전(latched · closing) TrajOpt 활성 제약의 출처** (출처 = 구 중심에서 GT 표면 거리가 거리장 값에 가장 가까운 물체, 허용 오차 10 / 20 mm)

| 장면 | 활성 row | target 사과 | 이웃 과일 | 바구니 · 기둥 | 탁자 | 미상 |
|---|---|---|---|---|---|---|
| 실패 6 장면 합 | 5,423 | 18.8% | 23.9% | 4.8% | 5.0% | 47.5% |
| 211 | 1,160 | 29% | 58% | 6% | — | 8% |
| 320 | 1,907 | 23% | ~0% | — | 13% | 64% |
| 526 | 354 | 5% | — | 38% | — | 56% |
| 954 | 571 | 1% | 94% | 2% | — | 4% |
| 979 | 66 | 88% | — | 12% | — | — |
| 1313 | 1,365 | 13% | 6% | 3% | 1% | 76% |
| 대조 성공 3 장면 합 (250 · 126 · 572) | 205 | 82.0% | 6.3% | — | — | 10.7% |

- "미상" row는 가장 가까운 GT 물체에서 중앙값 31 mm 떨어져 있다. 가장 가까운 물체는 탁자 1,171 · 바구니 1,154 · target 260 row다. 원인은 확인하지 않았다
- 요청별 가장 크게 침범한 row의 출처로 묶은 실패 장면 HOLD: target 3(211 k11 · k12 · k58), 이웃 과일 3(211 k21 · k22, 1313 k13)
- 211 요청 10–16 (A4): k10–14는 SPLIT와 NBC가 같다. 가장 크게 침범한 row는 k10 바구니, k11–14 target이다. HOLD는 k11(0.5 mm) · k12(1.1 mm)이다. 이웃 접촉 규칙은 k15–16에서만 켜져 이 HOLD를 바꾸지 않는다. 손바닥–target 거리(GT)는 k10–16에서 208 → 108 mm다

**B — 실제 접촉** (손이 target 사과에 닿은 것은 제외, 침범 ≥ 0 mm)

| 상대 | 접촉 step | 장면 | 최대 침범 | 거리장 − GT (거친 / 고운, 중앙값) |
|---|---|---|---|---|
| 이웃 과일 | 72 | 119 · 1345 · 211 · 342 | 1.84 mm | +8.2 / +4.5 mm |
| 바구니 | 24 (놓은 뒤) | 901 (쓰지 않는 손 손가락) | 0.06 mm | +13.8 mm / 고운 범위 밖 |
| 탁자 | 21 (닫는 중) | 320 | 1.40 mm | 0 / 0 mm |
| 바닥의 target 사과 (바퀴, 미검사 링크) | 15 | 320 | — | — |
| 기둥 · 팔 · 손목 | 0 | — | — | — |

**미리 정한 해석과의 대조**

- A: target 출처가 실패를 "주로" 설명하지도, 이웃 과일 출처가 "대부분"이지도 않다. 미상이 47.5%로 가장 크다. 장면별로 갈린다(954는 이웃 94%, 979는 target 88%, 320 · 1313 · 526은 미상이 절반 이상). 사용자 판정을 기다린다
- B: 조건(검사 대상 링크의 바구니 접촉 + 거친 층이 멀게 봄)은 901의 한 건(놓은 뒤, 0.06 mm)으로 글자 그대로 충족된다

## 2026-10-22 00:50 KST — 사용자 판정: "미상" 출처 확인과 순서 1을 함께 진행

- **IFG-UNKNOWN** (`AG3S/ag3s/docs/handoff/IFG-UNKNOWN.task.md`, verifier): 실패 장면 활성 제약의 47.5%를 차지한 미상 voxel이 어디서 왔는지 분류한다
- **IFG-SPHERE 1단계** (`AG3S/ag3s/docs/handoff/IFG-SPHERE.task.md`, implementer, 코드 수정 없음): 남은 target voxel이 어느 판단 때문에 남는지 분해하고 설계안을 낸다
  - task를 쓰며 확인한 사실: 제외 공의 중심은 이미 표면 구 맞춤(`centre_method == "surface_fit"`)이다. LTE 점검에서 맞춤 중심 오차는 stale 아닌 요청에서 중앙값 2.01 mm였다
  - 그러니 "평균 대신 구 맞춤"만으로는 닫는 중 recall 0.59를 설명하지 못한다. 공 안에서 voxel을 남기는 판단(소유 판정, 표면 보호, 반지름 제한, stale 공 등)을 먼저 가른다
  - 설계안은 사용자 판정 뒤 2단계에서 구현한다
- 사전 확인 A agent가 filesystem 루트에 남긴 `/w2.out`(426 byte, argparse 오류 메시지)은 안전 검사 때문에 lead가 지우지 못했고, 사용자가 삭제했다(2026-10-22)

## 2026-10-22 02:58 KST — "미상" 활성 제약의 출처 (IFG-UNKNOWN)

- 수치: `AG3S/ag3s/docs/handoff/IFG-UNKNOWN.verify.json`. 그림: `AG3S/ag3s/docs/figures/ifg-unknown/`. 회귀 기준선 일치
- 방법: 미상 row마다 거리장 기울기를 따라 표면 voxel을 찾고, 그 근처 10 mm 안의 관측 depth 점(frame 0..k, 카메라 3대, 서버 FK 구 mask 뒤)을 segmentation render로 분류했다. 근처 관측이 없으면 거리장 artefact로 본다
- 규칙 변경 1건(02:55 KST, 1차 결과를 본 뒤):
  - "옮겨짐" 판정을 충돌 geom 거리 대신 물체 pose 추적으로 바꿨다
  - 바구니 테두리의 visual mesh가 충돌 벽보다 7 mm 높아, 움직이지 않은 점이 옮겨진 것으로 판정됐기 때문이다
  - 1차 수치는 `numbers.v1_moved_test_gt_distance`에 남겼다

| 출처 | 실패 6 장면 row (2,573) | voxel (406) |
|---|---|---|
| 옛 관측 기억(stale, 지금은 그 자리에 없는 물체) | 44.5% | 38.9% |
| 거리장 artefact(관측 점이 근처에 없음) | 44.1% | 45.3% |
| 지지면(탁자 · 바구니) 표면 | 5.1% | 3.2% |
| 로봇 자기 몸 | 3.0% | 0.5% |
| 그 요청의 target 사과 | 2.8% | 9.4% |
| ambiguous | 14 row | 2.7% |

- 장면별
  - 320: 1,217 row 중 1,139가 기억이다. 모두 frame 17 이전에 본 target 사과다. 사과는 k = 18에 높이 0.85 → 0.23 m, k = 19에 0.03 m로 떨어졌다
  - 1313: 1,042 row 중 1,009가 artefact다
  - 526: 로봇 76 row(쓰지 않는 손, 앞 frame에서만), 바구니 표면 121 row
  - 211: 93 row 중 59가 artefact다
- artefact row: 가장 가까운 관측 점까지 중앙값 34.4 mm(p95 41.8). 가장 가까운 점의 라벨은 바구니 1,065 row, target 69 row
- 그 순간 보이는가(카메라에서 표면 voxel까지 시선이 막히지 않음 / depth가 그 voxel에 닿음): 기억 0.06 / 0.002, artefact 0.95 / 0.004
- U3: 미상 row가 가장 크게 침범한 22 요청에서 HOLD 0. TrajOpt 수정량 중앙값 artefact 9.15 mm, 기억 12.44 mm. 가장 큰 것은 320 k17 120.5 mm(artefact), k18 66.3 mm(artefact)
- 대조 성공 3 장면: 미상 22 row 중 target 12, 기억 9, ambiguous 1
- 한계
  - 기록에 frame별 TSDF weight가 없어, 어느 frame이 voxel을 썼는지는 근처 점으로 추정했다
  - 211 frame 23–24에서 mask 불일치가 최대 2,864 px다

## 2026-10-22 03:16 KST — IFG-SPHERE 1단계: 남은 target voxel의 원인은 공이 아니라 non-target 근거였다

- 결과: `AG3S/ag3s/docs/handoff/IFG-SPHERE.impl.md`
- script: `AG3S/ag3s/experiments/studies/ifg_sphere_attrib.py`
- raw: `20261022T023623+0900_055f2ea2f00b_8e1a8f54bb5e/verify/IFG-SPHERE-ATTRIB/`
- 재현: latched · closing 요청 669개 중 661개를 측정했다. offline replay는 기록과 660 / 661 요청에서 일치한다. 남은 voxel 중앙값 251 · 520은 LTE 점검(251 · 516.5)과 같다

| 남은 GT target voxel의 원인 | latched | closing |
|---|---|---|
| 공 밖(중심 · 반지름) | ≈ 0 | ≈ 0 |
| `z_min` 아래 | 10.9% | 11.0% |
| non-target으로 보존 | 4.2% | 5.1% |
| ambiguous로 보존 | 18.9% | 21.2% |
| seed는 지웠지만 부호가 음수로 남음 | 29.5% | 25.5% |
| 내부, 부호 보정 대상 아님 | 27.9% | 28.3% |
| 내부, 부호 검사 실패 | 8.5% | 7.2% |
| stale 공 | 0.1% | 1.75% |

- 보존 · 내부 voxel의 99.6%(122,295 / 122,797)에서 가장 가까운 non-target 근거 점이 GT target 점, 즉 사과 자신의 표면이다
- 사전 확인 A의 target 출처 활성 row 중 stale 아닌 785개의 99%가 이 오염된 근거로 남은 voxel에 닿는다(211 k11–14 102 row, 979 58 row, 320 k16 42 row)
- 추적 · 기억 중심(17.5 mm 치우침)은 제외 공에 쓰이지 않는다(661 / 661 요청 surface_fit)
  - 쓰이는 곳: 쥠 도달 판정(`safe_policy.py:1399` → `grasp_latch.py:714`), self-filter guard 중심(`pipeline.py:1805`), NBC row 선택(`safe_policy.py:1731`)
- **lead 확인: IFG serving 구성에는 T46c에서 채택한 F1(`--nontarget-evidence surface`)이 없다**
  - IFG kit의 `run_chain.sh`는 C1aN · SPL · S2 · SURF arm에 이 flag가 있으면 실행을 거부한다
  - `start_server.sh`의 계보는 T53("arm C1a only = current serving FLAGS") → T54c → held-out collector → IFG-CL1 → IFG-APPLE*이다
  - F1의 close 이후 HOLD 증가를 고친 W1(`--placed-no-seed-certify`)도 없다
  - 의도된 제외인지는 기록에서 확인하지 못했다
- 설계안
  - **P1:** 코드 수정 없이 F1을 켠다. offline 효과
    - 남은 voxel 중앙값: 251 → 90, 520 → 136
    - recall: 0.79 → 0.91, 0.58 → 0.85
    - stale 아닌 target 활성 row: 785 → 133 (211 k11–14 102 → 19, 979 58 → 0, 320 k16 42 → 0)
    - 이웃 과일 표면 voxel 손실: 3 / 24,647 · 7 / 12,303. 탁자 손실 0
    - 위험: T46 CL3에서 close 이후 uncertified HOLD가 37 → 130으로 늘었다(W1로 해소)
  - **P2:** 새 flag `--target-free-support-sign`. P1 위에서 남은 voxel 중앙값 90 → 40 · 136 → 75. 활성 row는 줄지 않는다(133). 부호를 바꾼 voxel 중 0.24%가 이웃 과일 안이다
- 측정하지 않은 것: HOLD 변화(TrajOpt 재풀이 필요)와 closed-loop
- 320 k17 393 row는 stale 공(닫는 중 사과가 밀려 맞춤 중심이 54 mm 어긋남)이라 P1 · P2로는 줄지 않는다. 이것은 순서 2(기억 이동 · 무효화)의 대상이다

## 2026-10-22 23:39 KST — 사용자 판정: 장애물 근거에서 target 표면 점을 빼는 설정의 closed-loop 확인

- 사용자 지적(세 번째): 줄인 이름(이전 STEP의 설정 기호, 제안 번호, "arm", 작업자 기호)을 쓰지 말고 무슨 실험이고 무엇을 쓴 실험인지로 말한다. 이 절부터 그렇게 쓴다
- 판정
  1. 지금 최선 구성(이웃 과일과 붙은 target 떼어 내기 + 쥐기 직전 손목 관절 경로 가중 0.1 + 손가락–이웃 과일 접촉 20 mm 허용, 9 / 10)에 두 설정을 더한다
     - 장애물 근거에서 target 자신의 표면 점 빼기(`--nontarget-evidence surface`)
     - 놓은 뒤 target을 못 찾아도 안전 인증(`--placed-no-seed-certify`)
     - 같은 VLA 단독 성공 사과 10 장면 · seed에서 closed-loop로 짝 비교한다
  2. 실험 틀은 새 구성에서만 두 설정을 허용한다. 기존 구성의 "있으면 거부" 검사는 그대로 둔다
- 정정: 앞 절들에서 비교 기준을 "target 분리 + 이웃 접촉 20 mm"로 적었다. 실제 서버 flag에는 손목 관절 경로 가중 0.1(`--wrist-track-scale 0.1`)도 들어 있다
- task: `AG3S/ag3s/docs/handoff/IFG-APPLE-TSURF.task.md`. run 폴더: `verify/IFG-APPLE-NBC-TARGETSURFACE`
  - 사용자 규칙대로 place 판정 2.5 초 뒤 기록을 끝낸다
  - 비교 기준은 끝까지 기록됐으므로, 두 쪽 모두 같은 구간(place 뒤 2.5 초까지)으로 잘라 비교한다

## 2026-10-23 01:32 KST — 결과: target 표면 점 빼기 + 놓은 뒤 인증 구성 (사과 10 장면 closed-loop)

- run: `verify/IFG-APPLE-NBC-TARGETSURFACE`. 수치: `AG3S/ag3s/docs/handoff/IFG-APPLE-NBC-TARGETSURFACE.verify.json`
- 그림: `AG3S/ag3s/docs/figures/ifg-split/ifg-targetsurface-{scene-ep211,graph,table}.png`
- 회귀 기준선 일치. smoke 통과(첫 VLA chunk 차이 0, surface 근거 적용 확인)
- 두 구성 모두 place 판정 후 2.5 초까지로 잘라 비교했다
- code snapshot: HEAD a6ff7dd9 + 아직 커밋하지 않은 client `--stop-after-place`. 서버 runtime은 비교 기준 snapshot과 같다

| 지표 (10 장면 합) | 지금 최선 구성 | + target 표면 점 빼기 + 놓은 뒤 인증 |
|---|---|---|
| 작업 성공 | 9 / 10 | 9 / 10 (실패→성공 0, 성공→실패 0, 211 둘 다 실패) |
| 다시 잡기(regrasp)로 성공 | 3 장면 (56 · 1668 · 211 실패) | 0 장면 |
| 쥐기 전 HOLD chunk | 9 | 0 |
| 놓은 뒤 2.5 초 안 HOLD | 0 | 0 |
| 이웃 과일 실제 접촉 | 22 행 (2 장면, 최대 2.3 mm) | 9 행 (1 장면, 1.5 mm) |
| 바구니 · 탁자 · 기둥 · 미검사 링크 접촉 | 0 | 0 |
| 남은 GT 사과 voxel 중앙값 (stale 제외) | 168 | 77.5 (latched 189 → 77, closing 117.5 → 142) |
| 사과 출처 활성 제약 행 | 2,117 | 39 |

- 211 요청 10–16
  - 지금 최선 구성: HOLD k11 · k12, 사과 출처 활성 행 0 · 24 · 18 · 18 · 42 · 4 · 0
  - 새 구성: HOLD 0, 사과 출처 활성 행 0
  - 새 구성의 손바닥–사과 거리: 208.5 · 181.2 · 155.6 · 121.5 · 106.9 · 107.2 · 107.2 mm
  - 닫기 시작: 지금 최선 구성 t = 163(손끝–사과 75.4 mm), 새 구성 t = 233(17.2 mm)
  - 새 구성에서는 그 뒤 손바닥–사과 약 107 mm가 유지되고 쥠이 일어나지 않았다(no_grasp). 원인은 아직 진단하지 않았다
- 처음 갈라진 요청: 10 장면 모두 k6–16. 갈라진 이유는 target 제외 층의 non-target · ambiguous 보존 감소다(211 k9: 164 → 7 · 112 → 37)
- 측정 한계
  - offline 원인 분류 재구성(`ifg_sphere_attrib`)이 이 설정을 읽지 않아, 새 구성의 voxel 원인 분류는 쓰지 않았다
  - 탁자 접촉은 확장 replay(`replay_contacts_ext.py`)로 셌다. 기존 class 접촉 행은 표준 replay와 20 run 모두 같다
- 미리 정한 해석과의 대조: "211이 그대로 실패하면 요청 10–16의 사과 출처 제약이 줄었는지 본다" → 줄었다(112 → 0, HOLD 2 → 0). 211의 실패 양상은 쥐기 직전 멈춤에서 "가까이서 닫았는데 쥐지 못함"으로 바뀌었다
- pod 관찰: 2026-10-18부터 D state로 멈춘 다른 세션의 IFG-CL3 client process 4개가 있다(socket 없음). 건드리지 않았다

## 2026-10-23 01:43 KST — 사용자 판정: 새 비교 기준, 211 쥐기 실패 진단, 커밋, 멈춘 process 정리

- **새 비교 기준 구성:** 이전 최선 구성에 다음 두 설정을 더한다. 이후 사과 실험은 이 구성에 하나씩 더해 비교한다
  - 이전 최선 구성: 이웃 과일과 붙은 target 떼어 내기 + 쥐기 직전 손목 관절 경로 가중 0.1 + 손가락–이웃 과일 접촉 20 mm 허용
  - 더한 설정: 장애물 근거에서 사과 표면 점 빼기 + 놓은 뒤 target을 못 찾아도 안전 인증
- **211 쥐기 실패 진단:** task `AG3S/ag3s/docs/handoff/IFG-211-GRASP.task.md`, CPU offline, 진행 중
  - 질문: 새 비교 기준 구성에서 사과 17 mm 앞에서 닫고도 쥐지 못한 원인
  - 후보: 바나나 · 탁자 선접촉, 사과 밀림, VLA 계획, TrajOpt 이동, 쥠 판단, 그 밖
- **커밋:** `27e79f8c`
  - client의 place 뒤 기록 종료(`--stop-after-place`), 메모리 회수 script, 작업 규칙 문서 2개
  - 이번 실험 기록은 이 절 뒤에 따로 커밋한다
- **멈춘 process 정리 (사용자 허락):** 2026-10-18 IFG-CL3 run의 client 4개(pid 2150267 · 2151352 · 2394708 · 2395140)
  - 상태: 부모 pid 1, 각 약 155 MB
  - SIGTERM에 이어 SIGKILL을 보냈다. 커널의 페이지 입출력 대기(D state, `folio_wait_bit_common`)라 신호가 대기 상태로만 남고, 아직 종료되지 않았다
  - 입출력이 풀리면 종료된다. 사용자 공간에서 더 할 수 있는 조치는 없다
