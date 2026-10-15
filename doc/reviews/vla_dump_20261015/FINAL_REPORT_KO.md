# 최종 보고서 — VLA 내부 feature + RGB-D target 영역 추정 (S0–S3 및 개선 cycle)

작성: 2026-10-15 KST. 분석 session(branch `feature/ag3s-development`, HEAD `30955b6c` + 미커밋 `ifg_*.py`).

- 정본 문서: [실험 프로토콜 v1.2](EXPERIMENT_PROTOCOL_KO.md), [3D Surface Graph 가이드 §15](../AG3S_3D_Surface_Graph_and_Parameter_Tuning_Guide.md)
- 상세 결과: [SUMMARY_METHODS_RESULTS_KO.md](SUMMARY_METHODS_RESULTS_KO.md)
- 시간순 기록: [WORKLOG_S0S3_KO.md](WORKLOG_S0S3_KO.md)
- Held-out 요청: [HELDOUT_COLLECTION_REQUEST_KO.md](HELDOUT_COLLECTION_REQUEST_KO.md)
- Run: `/mnt/dev/AG3S_VLA-storage/results/development/20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba` (`manifest.json`, `improvement_registry.jsonl`, `LOCK_heldout.json`)

> **범위:** 모든 수치는 **dev**(T53 + T54c 32 episode, 180 phase key-moment frame, 이미 결과를 본 episode)에서 나왔다. 개선은 dev GT를 보고 설계했고, 같은 dev에서 평가한 CI는 독립적인 유의성 증거가 아니다. **Held-out 검증은 아직 없다.** Depth는 MuJoCo replay이고, false-exemption은 segmentation FP proxy다.

![overview](../../../AG3S/ag3s/docs/figures/ifg/ifg-final-overview.png)

## 1. 결론

| 질문 | 답 (dev) |
|---|---|
| 같은 seed·관측 표면에서 **feature + geometry가 geometry-only보다 target 경계를 더 잘 잡는가** (원래 연구 질문) | **아니오.** 프로토콜 기본값(R0), 원인별 개선 6 cycle(R1–R6), 개선된 기반에서의 재검증(R7) 모두 feature 추가 이득이 없었다. Feature를 edge에 넣으면 효과 0, unary로 넣으면 recall이 떨어졌다 |
| Attention + 3D(feature 없이)는 기존 serving보다 나은가 | **같은 robot mask 조건에서 IoU +0.057 [+0.006, +0.103]**(0.893 대 0.837, 27/5 episode). 이득은 주로 precision(+0.073)에서 온다. CI 하한이 0에 가깝고 dev 결과다 |
| Feature를 쓸 곳은 없는가 | **과거 frame 기억과의 일치도를 안전 경보로 쓰면 유용하다.** 잘못 고른 frame 탐지 AUROC는 0.998(147 frame)이고, feature 없는 점수는 0.914(179 frame)다. 같은 frame 집합에서의 차이는 +0.096 [0.000, +0.266]이다. 잘못된 frame이 6개뿐이다 |
| 3D는 작업 단계·파지 판단을 돕는가 | **돕는다**(supervised ablation): 단계 정확도 0.783 → 0.939, 파지 0.828 → 0.933. Feature identity는 추가 효과가 없다(이 dump에는 엉뚱한 물체를 쥔 사례가 없음) |

## 2. 최초 실패 (R0, 프로토콜 그대로)

**구현:** S0 정합 → S1 seed × feature 15 prototype → S2 depth surface graph → S3 연속 energy와 2×2 mechanism control(G / FGe / FGu / FG).

**정합(S0):**
- frame022 기존 진단을 bitwise로 재현했다(float 822개 차이 0).
- 180 frame 전부 통과: state·image·GT fingerprint 일치, depth planar z(ray 검사 오차 ≤ 2 µm), roundtrip ~1e-12 px.

**결과(seed N, τ = 0.5):**

| Arm | IoU |
|---|---:|
| B0 serving noun | 0.785 |
| **G**(attention + 3D graph) | 0.728 |
| FGe | 0.728 |
| FG(projector) | 0.582 |
| F(feature-only) | 0.256 |

FG − G는 5개 feature 모두 −0.10 ~ −0.19였다. 설정 문제 하나도 발견했다: 문서 기본 depth 범위(z ≤ 3 m)에서는 table plane이 검출되지 않았다. Serving `range_max` 2.0 m로 맞췄다(결과를 본 뒤 수정했고, 기록함).

## 3. 개선 이유 → 바꾼 것 → 전후 수치

각 round는 실행 전에 가설·반증 대조군·채택 규칙을 registry에 등록했다. 채택 규칙: Δrecall ≥ −0.02, FP proxy ≤ reference, IoU ≥ parent + 0.02, shuffle 대조가 효과를 재현하지 않을 것.

| Round | 관측 증거 → 원인 가설 | 바꾼 것 | 전 → 후 (IoU) | 판정 |
|---|---|---|---|---|
| R1 | FG의 FP가 다른 과일·crate로 번짐. Patch 점수가 표면마다 복사됨 | feature를 graph component identity gate로 | 0.728 → 0.642 | 기각. Component가 target과 table을 묶음 |
| R2 | R1의 단위 오류 | gate 단위를 G 예측 영역으로 | 0.728 → 0.755(recall −0.050) | 기각 |
| R3 | 기각된 영역 중 seed 질량 1위가 대부분 target(166/172) | 1위 영역 보호 | 0.755 → 0.785. **shuffle 0.784, feature 없는 top1-only 0.794** | feature gate 음성 종료. **top1-only 채택** |
| R4 | serving은 cluster 질량으로 선택 | region 질량 합으로 선택 | 0.794 → 0.688 | 기각(크기 편향). Oracle region 0.843 → 병목은 identity |
| (진단) | "FK filter가 seed 자격을 빼앗는다" | — | — | **가설 틀림.** 실패 frame의 seed pixel 71%가 table |
| R5 | seed 후보에 table pixel이 섞임 | seed 자격에서 support 제외(serving 원칙과 동일) | 0.794 → **0.817**(+0.023 [+0.005, +0.049], 7/0) | **채택** |
| R6 | FK sphere filter가 손가락 사이 과일을 지움(FN의 46%) | robot mesh 렌더 depth와 일치하는 pixel만 robot | 0.817 → **0.893**(+0.076 [+0.060, +0.093], 31/0) | **채택** |
| R7 | R0 실패에 seed·mask 결함이 섞였을 수 있음 | R6 기반에서 A+F+3D 전 arm + 가이드 A3 region growing 재실행 | FGe = G, FG 0.79–0.83, FG-R 0.74–0.76, GT seed를 줘도 FG < G | feature graph **최종 음성** |
| R8 | 16×16보다 촘촘한 신호가 필요한가 | dump의 224×224 pixel gradient 진단 | 면적 일치 precision 0.02–0.07(noun 0.50) | 음성(진단만) |
| R9 | R6 vs B0에 filter 차이가 섞임 | B0에도 같은 render mask | B0 0.785 → 0.837. 후보 − B0 = **+0.057 [+0.006, +0.103]** | 방법 효과 분리 |
| R10 / R10b | 같은 frame prototype은 seed 오류를 물려받음 | 과거 frame 기억으로 영역 전환. 전환이 안 되는 원인이 anisotropy(cos 0.96–0.98)여서 centering 추가 | R10 악화 또는 전환 없음, R10b 0.893 → 0.912(+0.019, n.s.) | 미확정(δ 척도 미조정, 결과를 본 뒤 수정) |
| R11 | 기억 일치도가 오류를 알릴 수 있음 | 경보 점수 비교 | AUROC(centered): 기억 chain 0.998(147 frame) / 같은 frame seed 0.928 / feature 없음 0.914(179 frame). 같은 147 frame의 raw 비교: 0.982 대 0.900, 차이 +0.096 [0.000, +0.266]. 10% 보류 시 FP 82% 제거, TP 6% 손실 | 긍정(표본 작음) |
| R12 | 3D 관계가 단계·파지 증거 | HSR + gripper + 3D logistic(LOEO) | 단계 0.783 → 0.939, 파지 0.828 → 0.933. Feature 추가 효과 없음 | 3D 긍정, feature 효과 없음 |

## 4. Feature가 영역 추정에 기여하지 못한 원인 (측정)

1. **해상도:** target 경계 edge의 73%는 양쪽이 같은 patch여서 feature 차이가 0이다. 경계는 depth가 이미 나눈다.
2. **혼합 patch:** target 표면의 42%가 다른 물체와 섞인 patch에 있다. 거기서 prototype 점수가 낮게 나와 target이 빠진다.
3. **정보 중복:** prototype은 seed에서 만들어지므로 seed가 맞으면 새 정보가 없고, 틀리면 함께 틀린다(AUC 0.93 → 0.74–0.78).
4. **Anisotropy:** 모든 feature가 큰 공통 성분을 공유해 cosine이 0.96–0.98에 몰린다. Centering으로 해소된다.

**정리:** feature에는 identity 정보가 있다(shuffle은 크게 나쁨). 그러나 같은 frame의 경계 추정에는 쓸 수 없고, **다른 시점과의 identity 일치 확인**에는 쓸 수 있다.

## 5. 현재 dev 후보 (held-out 전 잠금)

- **영역 추정:** noun L10H7 top-5 seed(비robot·비support pixel ≥ 60) → 5 mm depth-surface graph(image 인접 + depth 연속 edge) → 연속 energy(seed 1, support 0) → seed 질량 1위 영역 → render robot mask.
- **안전 경보:** centered prefix_L8 기억 일치도, threshold 0.5904(dev 10% 분위수).
- **단계·파지:** LR[HSR + gripper + 3D] weight.
- 모두 `LOCK_heldout.json`에 코드 hash와 함께 고정했다.

## 6. 남은 한계

- **Held-out 없음.** 요청서는 작성했다(사용자 답변 반영: in-distribution 장면 허용, 성공 rollout 중심 평가, 이미 분석한 67 episode 제외, 실패 episode는 부록으로 보관, 최소 dump 항목).
- **약점:** grasp 실패 직후(F3)에서 후보가 B0보다 낮다(0.745 대 0.920). 남은 seed identity 비용은 약 0.025 IoU다(oracle seed 0.918).
- **실제 안전 지표 미측정:** permission builder의 false-exemption, ball/ESDF 연결, closed-loop.
- **Simulation 이점:** render mask와 replay depth가 simulation에서 정확하므로 실제 센서보다 유리하다.
- **Key moment만 사용:** 전환 지연과 시간 안정성은 미평가다. Held-out은 모든 step을 요청했다.
- **미실행:** T54 `hold` 조건, S4 multiview/time 비교, S5 인과 검증, σ_f/λ sensitivity(사전 조건 미충족).
- **가림 장애물 시나리오(0 / 1/3 / 1/2):** 테스트용으로 요청했고, 결과는 별도 보고하며 비중은 사용자가 판단한다.

## 7. 산출물

| 종류 | 위치 |
|---|---|
| Run manifest, registry, lock | `$RUN/manifest.json`, `$RUN/improvement_registry.jsonl`, `$RUN/LOCK_heldout.json` |
| Round별 가설·명령·결과·판정 | `$RUN/rounds/R0…R12/` |
| Figure (+json) | `AG3S/ag3s/docs/figures/ifg/`: `ifg-final-overview`, `ifg-scene-*`, `ifg-episode-iou`, `ifg-table`, `ifg-rounds`, `ifg-rounds-r4r6`, `ifg-r7-feature-revalidation`, `ifg-r8-gradient`, `ifg-r9-filter-matched`, `ifg-r10-r12-feature-reuse` |
| Code | `AG3S/ag3s/experiments/studies/ifg_*.py`(미커밋 WIP, sha256은 manifest에 기록) |
| 재현 | `bash $RUN/pipeline.sh` 후 `bash $RUN/rounds/<R>/commands.txt` |
