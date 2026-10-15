# IFG S0–S3 작업 기록 (internal feature + RGB-D target extent)

이 파일은 [START_HERE_KO.md](START_HERE_KO.md)에 따라 진행한 S0–S3 작업의 시간순 기록이다. 실행 정본은 [EXPERIMENT_PROTOCOL_KO.md](EXPERIMENT_PROTOCOL_KO.md) v1.1이다. 각 항목에는 실행한 것과 실행하지 않은 것을 함께 적는다. 수치는 아래 run 디렉터리의 산출물에서 직접 읽은 값이다.

- Checkout: `/mnt/dev/AG3S_VLA-development`, branch `feature/ag3s-development`, HEAD `30955b6c`(작업 시작 시점, 커밋 전 WIP 포함)
- Run: `/mnt/dev/AG3S_VLA-storage/results/development/20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba/` (아래 `$RUN`)
- Dump(다른 session의 collector, 읽기 전용): `.../20261015T150325+0900_ddc7a00060c4_d857a08dcb63/phase_dump/`
- Interpreter: `.venv-ag3s` (CPU, MuJoCo osmesa). GPU는 사용하지 않았다.

---

## 2026-10-15 18:17–18:30 KST — S0: frame022 재현과 phase dump 정합 검사

### 1. Frame022 기존 진단의 재현

[grounding_probe.py](grounding_probe.py)를 문서의 명령 그대로 새 run에 실행했다(`$RUN/s0/frame022_repro/`, 6.8 s).

| 비교 | 결과 |
|---|---|
| 기존 [grounding_metrics.json](grounding_metrics.json)과 float 822개 대조 | 최대 차이 **0** (bitwise 동일) |
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
- **수집 요청서:** [HELDOUT_COLLECTION_REQUEST_KO.md](HELDOUT_COLLECTION_REQUEST_KO.md). 이전 사용 episode 목록은 `$RUN/used_episodes_prior.json`(T53, T54c, T61, CL4, T58, T47 dev).
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
