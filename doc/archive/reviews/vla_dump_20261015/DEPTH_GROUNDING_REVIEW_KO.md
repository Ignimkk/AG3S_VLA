> 후속 감사: [T43–T61 대조](HISTORICAL_PROTOCOL_AUDIT_KO.md). 아래 제안 중 hand-frame co-motion은 T56에 이미 구현되었으며, T55/T61도 재사용 대상이다. 신규성·미구현 범위는 후속 감사와 v1.1 프로토콜을 따른다.

**Depth/3D를 전제로 한 내부 feature grounding 문서 평가 및 후속 분석**

작성: 2026-10-15 KST. 평가 대상: [AG3S_VLA_Internal_Feature_Grounding_Prompt.md](../AG3S_VLA_Internal_Feature_Grounding_Prompt.md). 원문은 수정하지 않았다. 신규 구현은 아래 offline 진단에 한정하며 serving·모델·원본 기록은 수정하지 않았다.

**1. 판단**

문서의 중심 가설인 “VLA 내부 특징으로 identity 후보를 찾고 geometry로 extent를 정한다”는 연구할 가치가 있다. 특히 unknown 유지, 잘못된 seed 검증, 동일 optimizer 조건, independent simulator GT, 동일 noise의 action intervention은 유지해야 한다.

그러나 현재 순서와 단순 prototype 가정은 보완해야 한다. **Depth→3D point cloud는 뒤에 붙이는 보정 도구가 아니라 분석의 기본 표현**으로 두는 것이 적절하다. 16×16 token은 정체에 대한 증거를 제공하고, 세밀한 표면 단위는 depth가 제공한다. 이 두 해상도를 분리해야 한다.

| 원문 항목 | 평가 | 구체적 수정 |
|---|---|---|
| Action attention을 Target Seed로 사용 | 행동 relevance가 manipulated object에 한정되지 않음 | action / noun / product seed를 분리 비교; destination·robot·support 반응 평가 |
| WeightedMean으로 단일 prototype | 구현은 간단하지만 seed 오염과 여러 면의 특징을 평균해버림 | robust/복수 prototype, positive–negative 대비, 3D region별 prototype 비교 |
| A 개선 후 B→C 순차 진행 | 2D 실패가 geometry 결합의 실패를 뜻하지 않음 | 최초부터 attention-only / geometry-only / feature-only / feature+geometry 비교 |
| Patch 또는 point graph | 두 노드 정의의 해상도 한계가 다름 | patch는 semantic evidence, surfel/supervoxel은 segmentation node로 분리 |
| TokenCut·CutLER 참고 | graph 아이디어는 적합 | CutLER의 detector 학습 단계는 채택하지 않는다고 명시; SigLIP에 같은 성질이 있는지는 별도 검증 |
| Multi-view fusion | 방향 적절 | 중복 관측·밀도 편향·움직이는 물체·좌표 시각 동기화를 추가 |
| No mask supervision | 학습과 선택 과정의 범위가 불명확 | mask로 weight 학습은 안 하는지, dev mask로 layer/threshold 선택은 허용하는지 명시 |
| Segmentation-Free | 실제로 segmentation을 수행하므로 표현 부정확 | “without external segmenters or mask-supervised training” 또는 “segmentation-model-free” 사용 |
| 결합을 novelty로 제시 | 현 단계에서는 가설 | geometry-only 대비 개선, seed와 extent 효과 분리, held-out 실제 접촉 결과로 검증 |
| 작업 단계 추론 | 현재 문서에는 별도 실험 설계가 없음 | spatial grounding과 별개로 temporal state/phase 실험 추가 |

TokenCut은 self-supervised feature affinity를 graph로 만들고 normalized cut으로 물체를 찾는다. 따라서 graph spectral baseline의 근거는 되지만 “SigLIP cosine이 곧 같은 객체”라는 보장은 아니다. [TokenCut 원논문](https://arxiv.org/abs/2209.00383). CutLER는 pseudo-mask 생성 후 별도 detector 학습 및 self-training을 포함하므로, 전체 방법을 따라가면 원문의 frozen/no-external-segmenter 조건과 달라진다. [CutLER 원논문](https://arxiv.org/abs/2301.11320).

**2. 이번에 추가로 실행한 진단**

[grounding_probe.py](grounding_probe.py), [grounding_metrics.json](grounding_metrics.json), [top20 patch 시각화](grounding_top20.png).

대상은 이전과 동일한 ep1839/frame022, `t_step=176`이다. 기존 T53의 class coverage GT, grasp GT, 원본 record 지문을 대조했다. VLA 특징·map으로 먼저 score를 만들고 그 뒤 GT로 평가했다. GT seed, GT negative, GT 입력 feature는 사용하지 않았다. 시뮬레이터 장면을 CPU replay해 depth·카메라 보정도 재생성했다. 이 depth는 원본 센서 기록을 새로 얻은 것이 아니라 저장된 시뮬레이션 상태의 replay이다.

원본 record는 `depth_cameras=[]`이지만 snapshot XML과 qpos/qvel이 있어 재생 가능했다. 카메라별 replay RGB와 기록 RGB의 평균 절대 차이는 8-bit 값 기준 0.0485 / 0.0104 / 0.1150이었다. 기존 GT patch coverage와 이번 replay coverage의 최대 차이는 세 카메라 모두 0이었다. 따라서 아래 혼합 patch 진단은 기존 GT와 일치하는 장면에서 수행되었다.

실행 설정은 `ctrl_hz=15`, `open_loop_horizon=8`이다. 이전 first8 비교는 실제 설정과 관련이 있음이 확인됐다. 단, HOLD·중단·TO 수정까지 반영한 실제 실행 여부는 controller trace로 별도 확인해야 한다.

**평가 정의:** 모든 카메라의 768 patch에서 score 상위 K개를 선택하고 선택된 patch 전체를 positive라고 간주했다. 각 patch의 실제 target 면적 비율을 적분해 precision/recall/IoU를 계산했다. K=5,10,20,40을 모두 JSON에 저장했다. 아래는 동일 예산 K=20 비교이며 운영 threshold·3D segmentation 결과가 아니다. seed는 각 map의 상위 5개, prototype은 L2-normalized feature의 seed score 가중평균을 다시 정규화했다. feature별 output은 전체 patch와 prototype의 cosine이다.

| Score/feature | Seed | Precision | Recall | IoU |
|---|---|---:|---:|---:|
| 기존 action L8H2 last | 해당 map 자체 | 0.0792 | 0.1527 | 0.0550 |
| action L8H2 first8 평균 | 해당 map 자체 | 0.0000 | 0.0000 | 0.0000 |
| noun orange L10H7 | 해당 map 자체 | 0.3255 | 0.6280 | 0.2729 |
| action×noun | 해당 map 자체 | 0.2822 | 0.5444 | 0.2283 |
| SigLIP encoded prototype | action | 0.1077 | 0.2077 | 0.0763 |
| SigLIP encoded prototype | action×noun | 0.1849 | 0.3567 | 0.1387 |
| SigLIP encoded prototype | noun | 0.4832 | 0.9322 | 0.4669 |
| Projector prototype | noun | 0.4837 | 0.9332 | 0.4676 |
| Prefix L4 output prototype | noun | 0.4554 | 0.8785 | 0.4284 |
| Prefix L8 output prototype | noun | 0.4271 | 0.8239 | 0.3914 |
| Prefix L12 output prototype | noun | 0.3771 | 0.7274 | 0.3304 |

**관찰의 해석:**

- 같은 SigLIP feature라도 seed에 따라 결과가 크게 달랐다. feature 평가에 앞서 seed identity 오류를 분리해야 한다.
- noun+SigLIP은 이 프레임에서 noun map 단독보다 target recall과 IoU가 높았다. 실제로 영역 확장에 활용할 가능성을 보여준다.
- 그러나 precision 48.3%는 선택된 면적의 절반 이상이 다른 표면이라는 뜻이다. 이 상태로 접촉 예외를 만들 수 없다. 3D 표면 분리와 robot/support 배제의 필요성이 드러난다.
- action last 상위20 면적 중 50%가 destination이었다. action relevance와 조작 물체 identity가 다른 사례다. first8 상위20은 90% background였으므로 실행 구간 query를 사용한다는 이유만으로 좋은 segmentation이 되는 것도 아니다.
- 앞서 action×noun가 더 좁아졌다는 관찰은 정확도 개선이 아니었다. product 상위5 patch의 평균 면적은 target 10.7%, support 70.3%였다. 실제 prototype 가중치를 반영해도 target 19.8%, support 63.7%였다.
- 특정 feature/layer의 일반적 우월성은 입증하지 않았다. 이 test 출처 사례를 보고 결정한 방법은 새 held-out episode에서 평가해야 한다. 수치가 큰 방법을 이 프레임에서 선택해 최종 성능으로 발표하면 안 된다.

**3. Depth가 이번 실패를 어떻게 바꾸는가**

오른손 카메라의 global patch 726에는 다음 표면이 함께 있었다. 각 patch는 replay 640×480에서 40×30=1200 pixel이다. 아래 class는 설명용 GT이며 algorithm 입력으로 쓰지 않았다.

| Patch 726 안의 표면 | Pixel 수 | Median z-depth |
|---|---:|---:|
| orange target | 629 | 0.0859 m |
| robot gripper | 356 | 0.0990 m |
| table support | 215 | 0.3287 m |

다른 seed patch 713에는 target 약 0.0965m, gripper 0.1035m, table 0.4109m, background 1.3072m가 함께 있었다. seed 679/695/696은 모두 support였다.

이것은 다음을 뜻한다.

1. 하나의 patch feature를 하나의 3D 평균점에 붙이면 서로 떨어진 표면 사이의 빈 공간에 점을 만들 수 있다. **depth mode/connected surface를 먼저 분리**해야 한다.
2. patch 726의 table은 target보다 약 24cm 뒤에 있어 depth로 분리할 강한 단서가 있다. 반면 gripper와 target은 가까우므로 depth threshold만으로 완전 분리는 어렵다. robot FK/model mask, normal, 외관, 시간적 물체 identity가 필요하다.
3. 같은 patch feature를 여러 3D surface에 복사해도 semantic evidence가 새로 생기는 것은 아니다. uncertainty가 큰 mixed patch로 취급해야 한다. 보간으로 feature 해상도를 올리는 것도 새 semantic 정보를 만드는 것은 아니다.
4. 이 프레임의 16×16 whole-patch binary mask는 GT를 알고 최적으로 선택하더라도 IoU 상한이 약 0.7087이었다. 이는 해당 프레임·평가 표현의 oracle bound이며 일반 모델 상한이 아니다. subpatch 3D surface를 노드로 쓰면 이 표현 제한을 피할 여지가 있다.

따라서 제안 표현은 **관측 surface 단위의 `(position, normal, uncertainty, projected feature evidence, semantic role, history)`**이다. 보이지 않은 물체 뒷면을 지금 관측한 것처럼 완성하지 않는다. 현재 관측 표면과 시간에 걸친 추정 geometry를 구분한다.

**4. 수학적 관점 A — 대비 학습 없이도 가능한 기하 제약 prototype**

최소 비용 첫 방법은 positive prototype 하나를 전역에 확장하는 대신, 3D 후보 영역과 negative evidence를 함께 쓰는 것이다.

\[
s_i=\hat f_i^\top\hat\mu_T-
\max_{b\in\mathcal B}\hat f_i^\top\hat\mu_b.
\]

`B`는 robot-model surface, 확인된 support plane, 다른 안정된 3D component 등이다. 낮은 attention이라는 이유만으로 negative로 지정하면 가려진 타깃을 negative로 만들 수 있다. basket은 destination이라는 독립 role로 유지한다. 무엇이 obstacle인지는 링크·phase별 접촉 규칙에서 판단한다.

다중 prototype `max_k cos(f_i, μ_Tk)` 또는 normalized log-sum-exp로 물체의 서로 다른 면을 표현할 수 있다. seed의 강한 outlier는 weighted medoid/trimmed mean으로 줄인다. 이때 trimming으로 작은 진짜 target seed가 제거되는 경우도 대조한다.

feature anisotropy가 크면 중심화·regularized whitening을 비교할 수 있다. `d≫n`이므로 2048×2048 sample covariance의 역행렬을 직접 쓰지 말고 shrinkage 또는 저랭크 투영을 쓴다. 어느 정규화가 좋은지는 dev에서 정하고 raw cosine baseline을 유지한다.

검증 질문: **feature가 target recall을 늘린 상태에서 geometry/negative evidence가 contamination을 줄이는가?**

**5. 수학적 관점 B — 3D graph와 Dirichlet energy**

3D point/surfel/supervoxel을 node로 하고, 실재하는 가까운 surface 사이에만 edge를 만든다. 각종 비용을 곱하기 전에 geometric gate를 둔다.

\[
w_{ij}=\mathbf1[\text{valid adjacency}]
\exp[-\alpha\|\hat f_i-\hat f_j\|^2
-\beta\|p_i-p_j\|^2
-\gamma(1-|n_i^Tn_j|)].
\]

각 계수는 해당 feature/거리의 단위를 흡수한다. normal 방향이 정해졌다면 signed alignment를 사용할 수 있다. depth jump·가려짐 경계·다른 관측 layer를 가로지르는 edge는 제외한다. 가까운 3D 거리만으로 edge를 만들면 맞닿은 다른 물체까지 연결되므로 local surface/feature consistency도 필요하다.

soft target membership `m`은 다음의 convex 문제로 시작할 수 있다.

\[
\min_{0\le m\le1}(m-s)^T\Lambda(m-s)+\lambda m^TLm,
\quad L=D-W.
\]

unconstrained solution은 `(Λ+λL)m=Λs`이며 실제로는 boundary seed/box constraint 및 연결되지 않은 component 처리가 필요하다. seed가 없는 component를 target으로 자동 결정하지 않는다. 이는 random walker/graph diffusion과 연결되는 방법이며 기존 방법이다. [Grady, Random Walks for Image Segmentation](https://www.cs.bu.edu/groups/ivc/pubs/Grady06.pdf).

반대로 normalized cut의 `Lz=λDz`는 geometry component의 분할 후보를 생성하는 baseline이다. 가장 큰/두드러진 물체가 task target이라는 보장은 없으므로 명사 evidence로 component를 식별한다. single hard cut보다 target/destination/robot/support/other의 role score를 유지하고 unknown은 별도 confidence decision으로 남긴다.

**6. 수학적 관점 C — 거리 대신 표면을 따라가는 geodesic과 scale 안정성**

Euclidean ball은 타깃 옆 장애물을 포함하기 쉽다. 대신 edge cost를 `c_ij=-log(w_ij+ε)`처럼 두고 seed에서 표면 graph를 따라가는 geodesic distance를 이용한다. 유효한 surface 연결이 없는 옆 물체로는 짧은 직선거리만으로 퍼지지 않는다. 다만 실제로 닿은 물체/탁자에서는 path가 이어질 수 있으므로 support와 semantic boundary를 함께 쓴다.

threshold를 조금 바꿀 때 target component가 계속 유지되는지, 특정 edge 하나에서 갑자기 table과 합쳐지는지를 기록한다. 이는 여러 threshold에서 연결 구조의 안정성을 보는 diagnostic이다. formal persistent homology 구현/성과를 주장하는 것은 아니다. 경계가 불안정한 연결부를 unknown으로 두는 기준 후보가 된다. 안정된 component라도 잘못된 물체일 수 있어 identity 검증을 대체하지 않는다.

**7. 수학적 관점 D — SE(3) 운동 일관성: segmentation과 grasp 상태를 함께 검증**

Depth의 중요한 추가 이점은 시간에 따른 실제 물체 운동을 볼 수 있다는 것이다. 양 시점의 보이는 surface correspondence `(p_i,t-1, p_i,t)`와 FK로 얻은 gripper pose `Tg,t`를 이용해 두 가설을 비교한다. 점과 pose는 동일한 안정된 world frame이어야 하며 base가 움직이면 보정한다.

\[
E_{\rm static}=\operatorname{robust\ mean}_i
\|p_{i,t}-p_{i,t-1}\|_{\Sigma_i^{-1}}^2,
\]

\[
E_{\rm held}=\operatorname{robust\ mean}_i
\|T_{g,t}^{-1}p_{i,t}-T_{g,t-1}^{-1}p_{i,t-1}\|_{\Sigma_i^{-1}}^2.
\]

`Σ`는 위치/정합 오차 covariance이며 SE(3) 변환 시 함께 좌표변환한다. 실제 구현에서는 visibility-aware ICP/feature association과 outlier 처리가 필요하다.

- held 가설이 좋고 hand가 충분히 움직이면, 손과 함께 운반되는 surface라는 증거가 된다.
- release 후 world-static 가설로 전환되면 놓임/attachment 해제의 증거가 된다.
- 손에 붙어 있던 identity와 현재 surface가 불일치하면 slip 또는 association 실패를 점검한다.

robot gripper 자체도 hand frame에서 정지하므로 robot model mask를 먼저 구분해야 한다. 손과 물체가 모두 정지한 구간에는 두 가설을 구분할 관측 정보가 없다. 정보가 없는 상황에서 held를 확정하지 않고 grasp feedback과 history를 유지한다. object가 밀리거나 독립적으로 이동하는 가설도 추가해 binary 강제 판정을 피한다.

이 접근은 “feature가 비슷한 점”에 더해 **“같은 rigid object로 움직이는 점”**을 묶는다. segmentation과 phase가 실제로 서로 도움을 줄 수 있는 가장 직접적인 물리적 연결이다.

**8. 수학적 관점 E — 작업 단계는 관측 기반 시간 추론**

세 시간축을 혼동하지 않는다: 실제 관측 시간 `t`, 미래 action index `j`, noise 제거 step `s`. 한 frame의 10 denoise step이 pick→place→home 진행을 뜻하지 않는다.

현재 계산 중 phase에 활용할 수 있는 것은 다음이다.

| 신호 | 가능한 용도 | 해석 한계 |
|---|---|---|
| HSR KV 및 place 확률 0.99865 | 기존 phase evidence | pose/state 의존성과 calibration 확인 필요 |
| target / destination surface에 할당된 action mass | 조작 대상에서 목적지로 행동 관심이 옮겨가는지 | 같은 head의 query별 역할을 먼저 검증 |
| noun feature와 object prototype의 일치 | manipulated identity 유지 | 작업 단계 자체를 바로 결정하지 않음 |
| 3D hand–object 거리·상대 운동 | approach/held/slip/release 증거 | 가림·정합 실패·stationary ambiguity |
| object–basket 상대 위치·운동 | 운반/배치/안정 residence 증거 | centroid만 basket 안에 있다고 안정 배치가 아님 |
| 실제 gripper 상태·feedback | 물리 attach/release 보조 | gripper 닫힘만으로 grasp 성공 아님 |
| 예측 action의 FK 경로 | 다음 행동 의도 예측 | 아직 실행되지 않은 future intention |

예를 들어 추정한 target/destination surface membership `m_T,m_D`를 action attention에 투영하여

\[
r_T(j)=\sum_i a_{ji}m_T(i),\quad r_D(j)=\sum_i a_{ji}m_D(i),\quad
z_j=\log\frac{r_D(j)+\epsilon}{r_T(j)+\epsilon}
\]

를 만들 수 있다. 각 patch 내 surface ownership의 모호성을 반영하고 camera visibility를 고려해야 한다. 이것은 역할 선호 score이며 calibrated probability가 아니다. target attention이 감소했다고 manipulated identity를 바꾸지 않는다.

관측 feature `o_t`를 결합한 HMM/HSMM은 다음 구조를 갖는다.

\[
b_t(s)\propto L(o_t\mid s)\sum_{s'}P(s\mid s')b_{t-1}(s').
\]

HSMM은 state duration을 명시적으로 다룰 수 있다. 기존 이론이며 여기서 필요한 것은 작은 유한-state 모델이다. [Explicit-duration HSMM 참고](https://www.jmlr.org/papers/volume14/johnson13a/johnson13a.pdf). backward slip/grasp failure/retry transition을 허용하고, 시간을 지나면 자동으로 성공 단계로 가는 규칙은 피한다. duration은 실제 초 단위로 다루며 15Hz 제어와 8-step 관측 간격을 구분한다.

`L`을 만들 때 HSR·noun·action attention은 같은 VLA에서 나온 상관된 evidence이므로 독립 likelihood라고 단순 곱하지 않는다. calibrated log-linear emission 또는 공동 feature 모델을 baseline rule과 비교한다. 새 phase 학습을 금지하는 조건이라면 명시적 physical-event transition을 사용한다. mask supervision 금지와 phase supervision 금지는 서로 다른 제약이다.

**현재 frame의 기존 grasp GT는 held이고, 오른손 finger contact count는 [0,2]다.** 이것은 물체를 들고 있다는 독립된 물리 근거이지 HSR의 place 분류 전체가 맞다는 증명은 아니다. 단일 frame과 단일 episode로 phase 정확도 향상을 주장할 수 없다.

**9. 수학적 관점 F — 다중 시점 정합과 불완전 대응**

단순 3-camera 평균 대신, geometry/normal/feature를 결합해 같은 surface를 연결한다. sparse nearest-neighbor association을 먼저 baseline으로 두고, 모호한 대응이 남을 때 partial/unbalanced optimal transport를 후보로 검토한다.

\[
C_{ij}=\alpha\|p_i-p_j\|^2+\beta(1-\cos(f_i,f_j))
+\gamma(1-|n_i^Tn_j|).
\]

관측이 안 된 점을 다른 물체에 강제로 대응시키지 않도록 unmatched mass를 허용한다. camera pose가 다른 시점에 취득되었다면 motion compensation 후 적용한다. 같은 voxel을 여러 카메라가 봤다는 이유로 표면 면적·표본 수·신뢰도를 무조건 세 배로 늘리지 않는다. point density 대신 voxel/surfel area 가중치를 사용하고, occluded view의 missing evidence와 실제 contradictory evidence를 구분한다.

이 방법은 후보 설계이며 이번 진단에서 OT를 실행하거나 효과를 확인하지 않았다. 단순 association보다 비용이 크므로 이점이 확인될 때만 채택한다.

**10. 두 문제를 연결하되 오류가 자기강화되지 않게 하기**

최종적으로 관심 변수는 target identity `I`, surface membership `M`, physical attachment `A`, task phase `S`이다. 관측은 RGB/VLA features `F`, depth `D`, robot state `Q`, history다.

공동 factorization의 개념은 다음과 같다.

\[
p(I,M,A,S\mid F,D,Q,history)\propto
\psi_{semantic}(I,M,F)\psi_{geometry}(M,D)
\psi_{motion}(M,A,D,Q,history)\psi_{phase}(S,A,F,Q,history).
\]

이는 joint model의 목표 구조이며 factor들의 조건부 독립성이 입증되었다는 뜻이 아니다. 초기 구현은 결합 전체를 한 번에 최적화하지 말고 모듈별 출처와 confidence를 기록한다. 예를 들어 높은 place 확률이 잘못된 basket segment를 만들고 그 segment가 다시 place를 강화하는 순환을 피해야 한다. 물리 grasp feedback·관측 geometry는 별도 근거로 유지한다.

**11. 수정한 첫 실험과 통과 기준**

먼저 네 비교군을 동일 조건으로 실행한다.

1. 기존 attention-only.
2. 동일 seed + geometry-only: depth component, support plane, robot filter.
3. 동일 seed + feature-only: 원문 A의 prototype.
4. 동일 seed + geometry + feature: subpatch surface graph/region score.

각 비교군에서 seed `{action, noun, product}`를 교차하고, 작은 고정 feature set `{SigLIP encoded, projector, prefix L4/L8/L12}`를 비교한다. 실제 실행 query는 별도 축으로 하되 모든 head/layer/query를 무제한 탐색하지 않는다. feature effect와 seed effect를 섞지 않는다.

진단용 oracle은 별도로 둔다: oracle identity/seed + predicted extent, predicted identity + oracle extent, 그리고 전체 oracle. GT는 이 진단과 평가에만 쓴다. 특히 feature-only 실패 때문에 geometry 결합 실험을 중단하지 않는다.

공간 지표는 visible-surface 3D precision/recall/IoU, neighbor contamination, robot/support leakage, false-exemption surface area, unknown rate, target fragmentation을 기록한다. 보이지 않는 영역은 현재 visible recall 분모에 넣지 않는다. fused point density와 중복 view를 보정한다. 최종 접촉 지표는 link·phase·실제 실행 prefix별로 따로 평가한다.

시간 지표는 independent phase label에 대한 macro-F1뿐 아니라 transition delay, false transition, identity switch, slip miss, release 후 stale attachment, latency다. state-only, physical strong rule, 기존 HSR, HSR+geometry, HSR+geometry+temporal filter를 비교한다. full-dump 없는 다른 frame에도 기존 T53 KV/noun feature와 record trace가 있어 compact temporal 분석을 시작할 수 있다.

결과는 episode 단위 split과 paired CI로 평가한다. 이 프레임은 진단에 사용했으므로 새 test claim의 독립성에 포함하지 않는다. GT로 threshold를 선택했다면 “no mask-supervised weight training”이라고 표현하고 “완전 label-free 선택”이라고 표현하지 않는다.

**12. 재현과 실행 범위**

```bash
MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/mnt/dev/AG3S_VLA-development \
.venv-ag3s/bin/python doc/archive/reviews/vla_dump_20261015/grounding_probe.py \
  --dump /mnt/dev/AG3S_VLA-storage/results/development/20261015T150325+0900_ddc7a00060c4_d857a08dcb63/vla_dump \
  --t53 /mnt/dev/AG3S_VLA-storage/results/development/20261013T100210+0900_4a540f1a18d7_cb037c425d2f/verify/T53 \
  --out doc/reviews/vla_dump_20261015 --replay
```

실행 완료: fixed seed/feature prototype 진단, GT 지문·state·frame 대응 검사, 3-camera depth/GT/RGB replay, patch 표현 oracle bound, 기존 grasp label 확인. 미실행: 새로운 3D segmentation algorithm, temporal classifier 학습/평가, model counterfactual inference, 실제 robot/closed-loop 실험.

이번 진단이 가장 강하게 지지하는 후속 가설은 **“명사로 물체를 식별하고, point cloud로 표면을 분리하고, 손과 물체의 상대 운동으로 attachment를 추정하며, 그 증거로 phase 추론을 보강한다”**이다.
