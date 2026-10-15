# IFG 정리: 이전 방식 / 현재 구현 방식 / 결과 분류

작성: 2026-10-15 KST. 시간순 상세 기록은 [WORKLOG_S0S3_KO.md](WORKLOG_S0S3_KO.md), round별 가설·판정은 `$RUN/improvement_registry.jsonl`에 있다. 이 문서는 나중에 보고서·논문으로 옮기기 위한 요약본이다. 수치는 모두 아래 run의 산출물에서 옮겼다.

- Run: `/mnt/dev/AG3S_VLA-storage/results/development/20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba` (`$RUN`)
- Figure: `AG3S/ag3s/docs/figures/ifg/` (각 PNG에 같은 이름의 `.json` sidecar)
- Code: `AG3S/ag3s/experiments/studies/ifg_*.py` (serving 코드는 수정하지 않음)

## 0. 데이터·평가 범위 (먼저 읽을 것)

| 항목 | 내용 |
|---|---|
| 데이터 | phase dump 180 frame / 32 episode(T53 16 + T54c 16). Episode마다 key moment(성공 M1–M7, 실패 F1–F3)만 있다 |
| Split | **전부 dev**(사용자 판정). T53·T54c는 이전 연구에서 결과를 본 episode이므로 held-out이 아니다. **Held-out 검증은 아직 없다** |
| Depth | 기록에 depth가 없어 MuJoCo로 replay한 simulated depth를 썼다(S0에서 planar z, K/T, GT coverage 정합 확인) |
| 주 지표 | 관측 표면의 5 mm voxel 위 3D IoU / precision / recall(occupancy weight 1, target 비율 soft). Frame → episode 평균 → episode paired bootstrap 2000회(seed 20261015) |
| False-exemption | 실제 permission builder는 돌리지 않았다. Non-target voxel이 target으로 예측된 양(FP)을 **proxy**로 썼다 |
| Baseline 조건 | T54 `current`(frame마다 reset, history 없음). `hold` 조건은 미실행 |
| 설계 방식 | 개선은 dev GT를 보고 설계했다(가이드 §15). Dev에서 반복 평가한 CI는 독립적인 유의성 증거가 아니다 |

## 1. 이전 방식

### 1.1 Target 영역 (serving grounding, 이하 B0)

T54 `t54_arms`의 N arm(noun target L10H7 map)을 c1a serving 설정(T53 rollout server와 같은 flag)으로 실행했다.

1. **Robot 제거:** URDF sphere/capsule + 덮개 capsule로 만든 FK 모델을 쓴다. **부피 안의 점을 모두 지운다**(`depth_robot_mask`). 손가락 열림은 렌더된 상태를 따른다.
2. **3D 점:** serving back-projection(range ≤ 2.0 m) 후 1 cm fusion voxel.
3. **Attention lifting:** 각 점에 그 점이 속한 16×16 patch의 attention 값을 붙인다.
4. **Seed:** lifted attention 상위 5%(percentile 95)의 점. RANSAC support 점은 connectivity에서 제외한다.
5. **성장·clustering:** 3D 반경 3 cm BFS 성장 + DBSCAN(eps 3 cm). 3 cm 안이면 다른 물체도 같은 cluster가 될 수 있다.
6. **선택:** cluster 점수(attention 기반)가 가장 높은 cluster 하나(attention split 등 serving 규칙 포함).
7. **출력:** cluster 점(region) + bounding sphere(ball).

### 1.2 작업 단계 판단

HSR subtask probe(prefix KV L4 → logistic, class pick/place/home)를 argmax로 쓴다. 파지는 gripper 상태 규칙으로 판단한다.

### 1.3 원래 연구 계획 (Attention + Feature graph + 3D, 프로토콜 v1.1/v1.2)

1. Noun attention top-5 patch → feature prototype(cosine).
2. Depth surface graph.
3. 위치·법선·feature edge + prototype unary.
4. 연속 energy `(m−s)ᵀΛ(m−s)+λmᵀLm`으로 target extent를 추정한다.

## 2. 현재 구현한 방식

### 2.1 Target 영역: Attention + 3D (현재 dev 후보, `top1-only-ns+rm:N`)

| 단계 | 구현 | B0와 다른 점 |
|---|---|---|
| Robot 제거 (R6) | robot mesh(robot body의 geom만)를 FK 자세로 렌더한 depth와 관측 depth가 **10 mm 안에서 일치**하는 pixel만 robot (`ifg_render_mask.py`) | 부피 → 표면. 손가락 사이 과일이 남는다 |
| Node (S2) | serving back-projection(range ≤ 2.0 m), **5 mm voxel**. 세 camera를 한 grid로 합친다 | 1 cm → 5 mm |
| Edge (S2) | **image에서 이웃한 pixel이면서 z 차 ≤ 10 mm**인 쌍이 두 voxel에 걸칠 때만 연결(+centroid 거리 ≤ 15 mm) | 3D 반경 연결 → 관측 표면 연속성 연결. 3D로 가까워도 depth가 끊기면 연결하지 않는다 |
| Seed (R5) | noun map(N, L10H7) 상위 5 patch. 단 patch 안에 **robot도 support도 아닌** pixel이 60개 이상이어야 한다. Patch 질량은 pixel 비율로 node에 나눈다(질량 보존) | 점 percentile → patch 단위, table 위주 patch 제외 |
| Unary (S3) | seed node s=1(나눈 질량을 신뢰도로), RANSAC support node s=0(Λ=1) | 없음 → support를 명시적 negative로 |
| 해 (S3) | component별 `(Λ+λL)m=Λs`(λ=1). Unary 없는 component는 unknown | DBSCAN → 연속 energy |
| 선택 (R3) | {m ≥ 0.5}의 연결 영역 중 **seed 질량이 가장 큰 영역 하나** | cluster 점수 → seed 질량 |
| 출력 | 그 영역의 5 mm node | ball은 아직 만들지 않음 |

### 2.2 시험한 feature 사용 방식 (모두 채택하지 않음)

| ID | 방식 |
|---|---|
| F | prototype cosine을 node로 옮긴 feature-only |
| FGe | edge 가중치에 feature 거리 추가 |
| FGu | prototype을 unary로 추가 |
| FG | FGe + FGu |
| FG-R | feature 조건을 건 region growing(가이드 A3) |
| FGc / FGr / FGr1 | component·region 단위 feature gate(R1–R3) |
| Mem | 과거 frame 기억 feature로 영역 전환(R10, R10b centering) |

### 2.3 Feature를 쓰는 안전 경보 (R11, 후보)

- 선택한 영역의 feature와 **과거 frame 기억**(첫 frame 또는 직전 선택)의 cosine을 신뢰도로 쓴다.
- R10b부터는 frame 평균 patch feature를 빼는 centering을 적용했다.
- 점수가 낮으면 그 frame의 접촉 예외를 보류(abstain)한다.

### 2.4 작업 단계·파지 판단 (R12, supervised ablation)

입력:
- HSR log-prob
- gripper 상태(두 gripper 열의 최솟값)
- 3D 관계: 선택 영역 중심과 두 gripper TCP(FK) 사이 최소 거리, 주변 support 위 높이, 영역 크기
- (선택) 기억 feature 일치도

numpy L2 logistic regression(L2 = 1.0 사전 고정)을 episode 단위 leave-one-out CV로 평가했다.

## 3. 결과 분류

### 3.1 개선이 확인된 것 (dev)

| 항목 | 비교 | 결과 [95% CI] | 근거 |
|---|---|---|---|
| Attention + 3D extent (같은 robot mask) | 후보 0.893 vs B0 region + render mask 0.837 | IoU **+0.057 [+0.006, +0.103]**, 27/5 episode. Precision +0.073, recall +0.005 | R9, `ifg-r9-filter-matched.png` |
| Render robot mask (B0에 적용) | 0.837 vs 0.785 | IoU +0.052 [+0.028, +0.081], recall +0.082 | R9 |
| Render robot mask (후보에 적용) | 0.893 vs 0.817 | IoU +0.076 [+0.060, +0.093], 31/0 episode | R6 |
| Non-support seed 자격 | 0.817 vs 0.794 | IoU +0.023 [+0.005, +0.049], 7/0 episode | R5 |
| 1위 region 규칙(geometry-only) | 0.794 vs G 0.728 | IoU +0.065 [+0.031, +0.111] | R3 |
| 3D 관계로 작업 단계 판단 | LR[hsr+grip+geo] 0.939 vs HSR argmax 0.783 | 정확도 +0.149 [+0.095, +0.196]. 3D만 더한 효과 +0.067 [+0.015, +0.118] | R12 (supervised) |
| 3D 관계로 파지 판단 | LR[grip+geo] 0.922 vs gripper 규칙 0.828 | 정확도 +0.147 [+0.079, +0.225] | R12 (supervised) |
| 기억 feature 안전 경보 | AUROC 0.998(chain, centered) vs feature 없는 점수 0.914 | 10% 보류 시 잘못된 frame 6/6 탐지, FP voxel 82% 제거, TP 6% 손실. **잘못된 frame이 6개뿐** | R11/R11b, `ifg-r10-r12-feature-reuse.png` |

Moment별(같은 render mask, 후보 대 B0):
- 앞섬: F2 0.910 대 0.758, M3 0.935 대 0.823, M6 0.911 대 0.758, M7 0.882 대 0.617
- 뒤짐: F3 0.745 대 0.920

### 3.2 이전 방식과 비슷한 것 (차이 미확인)

| 항목 | 결과 | 근거 |
|---|---|---|
| Attention + 3D vs B0, **serving sphere filter 조건** | R3 top1 +0.009 [−0.036, +0.055], R5 +0.032 [−0.008, +0.070] | R3, R5 |
| Feature를 edge에만 추가(FGe) | G와 동일(±0.001), 5개 feature 공통 | R0, R7 |
| 기억 feature로 영역 전환(centering 후) | +0.019 [−0.020, +0.060], 7번 전환(5 개선/2 악화). 사전 기준 +0.02 미달. **주의:** (1) centering은 R10 실패를 dev에서 진단한 뒤 넣은 수정이다. (2) 전환 기준 δ는 centering 전 척도(cos 0.96–0.98)로 정한 값이라 centering 후 척도(차이 약 0.2)에 맞게 다시 정하지 않았다 | R10b |
| 작업 단계·파지 판단에 feature identity 추가 | 단계 −0.010, 파지 +0.004 | R12 |
| B0 ball(render mask) vs B0 ball(sphere) | +0.013 [−0.008, +0.038] | R9 |

### 3.3 나빠진 것

| 항목 | 결과 | 근거 |
|---|---|---|
| Feature unary(FGu/FG) | R0 IoU −0.10 ~ −0.19(seed N, 180 frame). R7(개선 기반)에서도 −0.06 ~ −0.11, recall −0.07 ~ −0.11. GT seed를 줘도 G보다 낮음 | R0, R7 |
| Feature-only(F) | IoU 0.26(R0) | R0 |
| Feature gate 단위 오류(FGc) | recall −0.165 | R1 |
| Feature region growing(FG-R) | −0.14 ~ −0.16 | R7 |
| Region attention 질량 합으로 identity(MassTop1) | IoU 0.688, 이웃 FP 급증(크기 편향) | R4 |
| Pixel gradient(224×224)를 target 신호로 | 면적 일치 precision 0.02–0.07 vs noun 0.50 | R8 |
| 기억 전환, raw cosine | δ=0: −0.055(27번 전환, 21번 악화), δ≥0.05: 전환 없음 | R10 |
| Action map seed(A), product seed(P) | G:A 0.309, G:P 0.413 | R0 |

### 3.4 진단만 하고 방법으로 쓰지 않은 것

- **Oracle seed(GT top-5 patch):** 0.918. 남은 seed 오류 비용은 약 0.025 IoU다.
- **Oracle region(GT로 region 선택):** 0.843(R4 full-scene region). Region 표현의 상한이 B0보다 높다.
- **Oracle component/region gate:** R1·R2 원인 분리에 썼다.

## 3A. 실패·개선 실패 기록 (모든 시도)

각 줄은 시도 하나다. "실패"는 사전에 정한 채택 기준(Δrecall ≥ −0.02, Δfalse-exemption proxy ≤ 0, IoU ≥ parent + 0.02, shuffle 대조가 효과를 재현하지 않음)을 통과하지 못했다는 뜻이다. 실패한 결과는 지우지 않고 `$RUN/rounds/R*/`에 보존했다.

| Round | 시도 | 시도한 이유 | 결과 | 확인한 원인 | 교훈 / 다음 행동 |
|---|---|---|---|---|---|
| R0 | 프로토콜 그대로의 Attention + Feature graph + 3D(FG, 3 seed × 5 feature) | 연구 가설: feature가 경계를 넓히거나 정교하게 한다 | FG < G(seed N에서 −0.10 ~ −0.19), F 0.26, A·P seed 0.31·0.41. 기존 B0(0.785)보다도 낮음 | 다른 과일·crate로 번짐(이웃 FP 112 → 368). Mixed patch의 prototype 점수가 patch 안 모든 표면에 복사됨 | 원인 진단으로 이동(v1.2 §13.1) |
| R0(설정) | 문서 기본 depth 범위 z ≤ 3 m | 프로토콜 기본값 | table plane 미검출(바닥만), G precision 0.012 | 먼 바닥·벽이 RANSAC을 지배 | serving `range_max` 2.0 m로 수정(결과를 본 뒤 수정, 기록함) |
| R1 | feature를 graph component 단위 identity gate로 | feature는 patch 해상도이므로 표면 단위에서만 쓰자 | IoU −0.086, recall −0.165 | component가 target과 table을 함께 묶어 q가 희석됨(oracle gate도 recall −0.149) | 기각. gate 단위 수정으로 |
| R2 | gate 단위를 G 예측 영역으로 | R1의 단위 오류 수정 | IoU +0.026(n.s.), recall −0.050 | seed 질량 1위인 진짜 target 영역 10개 중 7개를 잘못 기각 | 기각. 1위 영역 보호로 |
| R3 | 1위 영역 보호 + 나머지만 feature gate | R2의 recall 손실 원인 | IoU +0.057이지만 **shuffle도 +0.055, feature 없는 top1-only가 +0.065** | 이득은 feature가 아니라 1위 영역 규칙 | feature gate 분기 **음성 종료**. top1-only를 후보로 |
| R4 | region별 attention 질량 합으로 identity 선택 | serving은 cluster 질량으로 고른다 | IoU 0.688(−0.106), 이웃 FP 1,006. Top1 실패 12개 중 4개를 구하고 성공 22개를 망침 | 큰 영역(crate)이 크기만으로 질량을 모음 | 기각. oracle region 0.843에서 병목이 identity임을 확인 |
| (R4 진단) | "FK filter가 seed 자격을 빼앗는다" 가설 | 실패 frame이 held 구간에 몰림 | **가설 틀림**: 자격 제한 유무와 관계없이 seed 동일 | noun 최댓값이 table 위주 patch(실패 frame seed pixel의 71%가 support) | R5(seed에서 support 제외)로 |
| R7 | 고친 기반(R5+R6)에서 A+F+3D 재검증 + region growing(가이드 A3) | R0 실패에 seed·mask 결함이 섞였을 가능성 | FGe = G, FGu/FG −0.06 ~ −0.11, FG-R −0.14 ~ −0.16. GT seed를 줘도 FG < G | 해상도(경계 edge 73%가 같은 patch), 혼합 patch(target node 42%), seed와 정보 중복 | **feature graph 최종 음성**. sensitivity는 사전 조건 미충족으로 미실행 |
| R8 | pixel gradient(224×224)를 target 신호로 | 16×16보다 촘촘한 신호 | 면적 일치 precision 0.02–0.07(noun 0.50), 상위 pixel의 19–27%가 robot | gradient는 "행동에 영향을 주는 pixel"이지 "명령한 물체"가 아님 | 진단만 하고 방법 수정 없음 |
| R10 | 첫 frame 기억으로 영역 전환(raw cosine) | 같은 frame prototype은 seed 오류를 물려받음(H4) | δ ≥ 0.05에서 전환 0회, δ = 0에서 27회 전환 중 21회 악화(IoU −0.055) | anisotropy: 모든 영역 cos 0.96–0.98. target이 1등이어도 차이 0.002–0.015 | centering으로(R10b) |
| R10b | centering 후 기억 전환 | R10 원인 수정 | +0.019 [−0.020, +0.060], 기준 미달 | δ를 새 척도로 다시 정하지 않음, 표본 작음 | **미확정**. held-out에서 보조 arm으로 재확인(잠근 δ = 0.05) |
| R12(feature) | 단계·파지 판단에 기억 feature 추가 | identity가 상태 판단을 도울 것 | 단계 −0.010, 파지 +0.004 | 이 dump에 엉뚱한 물체를 쥔 사례가 없음 | swap dump로 재검증(요청서 §5.1) |
| — | 남은 실패: F3(grasp 실패 직후) | — | 후보 0.745 대 B0 0.920 | noun seed가 target 밖. 기억 전환·경보로 일부만 대응 | held-out에서 경보 작동 여부 확인 |

## 3B. 앞으로 볼 시나리오: target을 가리는 장애물 (테스트용)

- **목적:** 장애물이 camera 시야에서 target을 20–60% 또는 60% 이상 가릴 때 후보·B0·경보가 어떻게 동작하는지 확인한다.
- **수집:** [HELDOUT_COLLECTION_REQUEST_KO.md](HELDOUT_COLLECTION_REQUEST_KO.md) §3.2.
- **사용자 판정(2026-10-15):** 이 시나리오는 **테스트용**이다. 결과는 별도 표로만 보고하고 주 결론이나 채택 판단에 섞지 않는다. 비중은 사용자가 판단한다.
- **dev에서 미리 알 수 있는 위험(검증 전, 가설):**
  - 가림으로 target 표면이 작아지면 noun seed의 자격 patch(비robot·비support 60 px)가 줄어든다.
  - 기억 경보는 보이는 표면이 다른 면일 때 cos가 내려갈 수 있다.

## 4. Feature가 영역 추정에 기여하지 못한 원인 (측정, `rounds/R9_rootcause`)

1. **해상도:** feature는 14×14 pixel patch당 값 하나다.
   - Target 경계 edge의 73%는 양쪽 node가 같은 주 patch를 공유한다. 그래서 feature 차이가 0이고 경계를 자르지 못한다(FGe=G).
   - 경계는 depth가 이미 나누고 있다.
2. **혼합 patch:** target node의 42%가 target 비율 <0.5인 patch에 속한다.
   - Prototype 점수가 <0.5인 target node(약 19%)의 80%가 혼합 patch에 있다.
   - 이것이 feature unary가 recall을 깎는 기전이다.
3. **중복과 오류 승계:** prototype은 같은 frame seed의 feature 평균이다.
   - Seed가 맞으면 새 정보가 없다.
   - Seed가 틀리면 함께 틀린다(node AUC 0.93 → 0.74–0.78).
   - Region 선택 정확도는 seed 42 대 prototype 32–34(48 frame)이다.
4. **Anisotropy (R10):** 모든 region feature의 cosine이 0.96–0.98에 몰려 있다.
   - 기억과 가장 닮은 영역이 실제 target인 경우(실패 7 frame 중 5)에도 차이가 0.002–0.015에 그친다.
   - Frame 평균을 빼는 centering 뒤에는 AUROC 0.99 이상으로 분리된다.

**정리:** feature에는 물체 identity 정보가 있다. Shuffle 대조는 크게 나쁘고, centering한 기억은 경보로 강하다. 그러나 **같은 frame에서 경계를 정하는 용도로는 해상도와 정보 중복 때문에 추가 가치가 없다.** 쓸모가 확인된 자리는 **다른 시점과의 identity 일치 확인(안전 경보)**이다.

## 5. 한계와 미실행

- **Held-out 없음:** 모든 수치는 dev이고 설계와 평가가 같은 데이터다. 수집 요청서: [HELDOUT_COLLECTION_REQUEST_KO.md](HELDOUT_COLLECTION_REQUEST_KO.md). 방법과 threshold는 `$RUN/LOCK_heldout.json`에 잠갔다(경보 threshold 0.5904).
- **False-exemption은 proxy:** 실제 permission builder와 ball/ESDF 연결 효과는 미측정이다.
- **Simulation 이점:** depth와 render mask가 simulation에서 정확히 일치하므로 실제 센서보다 유리하다.
- **Key moment frame만 사용:** 연속 timeline이 없어 단계 전환 지연(transition delay)이나 기억의 시간적 안정성은 평가하지 못했다.
- **R12 범위:** supervised ablation이다. 이 dump에는 엉뚱한 물체를 쥔 경우가 없어 identity 기반 P_task를 검증할 수 없다.
- **경보 표본:** R11의 잘못된 frame은 6개다.
- **미실행:** T54 `hold` 조건, S4 multiview/time 비교, S5 인과 검증, closed-loop, σ_f/λ sensitivity(사전 조건 미충족).

## 6. 재현 순서 (요약)

```bash
cd /mnt/dev/AG3S_VLA-development; RUN=/mnt/dev/AG3S_VLA-storage/results/development/20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba
bash $RUN/pipeline.sh                                   # S0, S1-S3(R0), B0 baseline
for r in R1 R2 R3 R4 R5 R6 R7 R9 R10 R10b R12; do bash $RUN/rounds/$r/commands.txt; done
```

Round별 `hypothesis.json`(실행 전 등록), `commands.txt`, `paired_summary.json`, `decision.json`은 `$RUN/rounds/R*/`에 있다.
