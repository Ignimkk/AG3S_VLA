# AG3S Method 이해 가이드

이 문서는 논문의 Method 절을 이 분야를 처음 접하는 주니어 연구자가 이해할 수 있도록 풀어쓴 설명서다. 수식의 증명보다 **각 모듈이 어떤 문제를 해결하고, 무엇을 입력받아 무엇을 출력하며, 왜 다음 모듈에 필요한지**에 초점을 둔다.

## 1. AG3S가 해결하려는 문제

로봇이 “사과를 집어 바구니에 놓아라”라는 지시를 수행한다고 하자. 이때 일반적인 충돌 회피만 적용하면 모순이 생긴다.

- 손가락은 사과에 닿아야 한다.
- 손목과 반대쪽 팔은 사과에 부딪히면 안 된다.
- 사과 옆의 다른 과일이나 지지대는 계속 피해야 한다.
- 사과를 집은 뒤에는 사과 자체도 로봇과 함께 움직이는 충돌 물체가 되어야 한다.
- 사과를 놓은 뒤에는 다시 환경 속 장애물로 취급해야 한다.

즉, 모든 물체를 무조건 피해서도 안 되고, 목표 물체 전체를 무조건 장애물에서 지워서도 안 된다. **어떤 표면을, 어떤 로봇 링크가, 어느 시점에 접촉해도 되는지**를 구분해야 한다.

AG3S는 이 문제를 다음 세 질문으로 나눈다.

1. **Identity — 무엇인가?** 지시가 가리키는 물체가 어느 것인가?
2. **Extent — 어디까지인가?** 관측된 표면 중 어디까지가 그 물체에 속하는가?
3. **Permission — 누가 언제 닿아도 되는가?** 어떤 링크에 어떤 작업 단계 동안 접촉을 허용할 것인가?

핵심 원칙은 세 판단에 서로 다른 증거를 사용하는 것이다.

| 판단 | 사용하는 증거 | 이유 |
|---|---|---|
| 목표 물체의 정체성 | VLA의 명사 attention과 중심화된 내부 feature | 정책이 지시문과 영상의 의미 관계를 알고 있기 때문 |
| 물체 표면의 경계 | depth의 표면 연속성 | metric 단위의 실제 경계는 영상 attention보다 depth가 더 잘 표현하기 때문 |
| 접촉 허용 시점 | 작업 phase와 물리적 grasp 증거 | 정책의 의도와 실제로 물체가 잡혔는지는 다르기 때문 |

## 2. 전체 구조 한눈에 보기

AG3S는 기존 VLA 정책을 다시 학습하지 않고 그대로 둔다. 이 **frozen VLA**는 RGB 영상 \(I_t\), 로봇 상태 \(q_t\), 언어 지시 \(l\)을 받아 여러 시점의 행동 묶음인 action chunk를 출력한다.

\[
A_t = \pi_\theta(I_t, q_t, l)
\]

AG3S는 같은 체크포인트를 계측한 복사본에서 명사 attention, 내부 feature, hidden state를 읽고, 여기에 동기화된 depth, 카메라 보정값, 로봇 모델을 결합한다. 처리 단계는 네 개다.

1. **ATS (Attention-based Target Segmentation)**: 목표 물체의 관측 표면을 찾고 시간에 따라 같은 물체를 추적한다.
2. **HSR (Hidden-state Sub-task Recognition)**: 현재 작업 단계와 실제 grasp 상태를 각각 판단한다.
3. **LTE (Link-specific Target Exemption)**: 링크와 상태별로 서로 다른 거리장을 제공한다.
4. **CCR (Constrained Chunk Refinement)**: VLA가 제안한 action chunk를 제약에 맞게 다듬고, 안전성이 확인된 앞부분만 실행한다.

데이터 흐름은 다음과 같다.

```text
RGB + 지시 + 로봇 상태 ──> frozen VLA ──> 원래 action chunk
                │
                ├─ 명사 attention ─┐
                ├─ 내부 feature ───┼─> ATS: 목표의 identity + extent
                └─ hidden state ───┘

depth + 카메라 + 로봇 모델 ─────────> ATS: 실제 표면 경계
hidden state + gripper + 기하 정보 ─> HSR: phase + grasp state
ATS + HSR ──────────────────────────> LTE: 링크별 거리장
원래 action chunk + 링크별 거리장 ──> CCR: refine + 검증 ──> 실행 또는 hold
```

여기서 active arm, 접촉을 허용할 링크 집합 \(\mathcal L\), 놓기 시 접촉을 허용할 support는 작업 설정으로 주어진다. AG3S가 affordance를 추론해서 자동으로 정하는 값이 아니다.

시스템이 request 사이에 유지하는 핵심 상태는 다음과 같다.

| 기호 | 의미 |
|---|---|
| \(j_t\) | 현재 추적하는 목표의 identity |
| \(\mathcal T_t\) | 현재 관측에서 승인된 목표 표면 집합 |
| \(v_t\) | grounding이 존재하고 alert가 없는지를 나타내는 validity |
| \(s_t\) | 확인된 task phase: pick, place, home |
| \(z_t\) | 물리적 grasp 상태: free, held, uncertain |

이 상태들은 하나의 confidence 값으로 합쳐지지 않는다. 예를 들어 identity를 알고 있어도 현재 표면이 관측되지 않으면 geometry를 지울 수 없고, phase가 place여도 grasp가 확인되지 않으면 물체를 attached geometry로 바꿀 수 없다.

## 3. 최종적으로 만족해야 하는 충돌 제약

로봇의 복잡한 형상을 여러 개의 query sphere로 근사한다. sphere \(i\)의 중심은 현재 관절 자세 \(q\)에서 \(c_i(q)\), 반지름은 \(r_i\), 추가 안전 여유는 \(\mu_i\)다. 거리장 \(\delta_{i,t}\)가 알려 주는 가장 가까운 표면까지의 거리를 이용해 다음 clearance를 정의한다.

\[
h_{ik}(q_k)=\delta_{i,t}(c_i(q_k))-r_i-\mu_i \ge 0
\]

쉽게 말해, **sphere 중심에서 장애물까지의 거리**가 **sphere 반지름과 안전 여유의 합**보다 커야 한다. 이 조건이 모든 sphere \(i\)와 waypoint \(k\)에서 만족되어야 충돌하지 않는다.

AG3S의 핵심은 optimizer 자체보다 \(\delta_{i,t}\)를 올바르게 만드는 데 있다. 같은 위치를 질의해도:

- 목표에 접근하는 손가락은 목표 표면만 제외한 거리장을 보고,
- 손목, 반대쪽 팔, 그 밖의 링크는 목표를 포함한 전체 장면 거리장을 보며,
- 물체를 잡은 뒤에는 그 물체도 attached sphere가 되어 주변 장애물과의 거리를 검사한다.

## 4. ATS 1단계: 현재 프레임에서 목표 표면 찾기

### 4.1 로봇 표면만 정확히 제거하기

카메라의 depth에는 환경뿐 아니라 로봇 팔과 gripper도 보인다. 이를 그대로 사용하면 로봇이 자기 자신을 장애물로 인식한다.

흔한 방법은 로봇 주변의 sphere나 capsule 부피를 통째로 지우는 것이다. 그러나 손가락 사이에 사과가 있으면 이 부피 안의 사과 표면까지 사라진다. AG3S는 현재 관절 자세의 로봇 모델을 카메라에 렌더링하여 예상 로봇 depth \(\widehat D^v_{\text{robot},t}\)를 만든다. 실제 depth \(D_t^v\)와 렌더링 depth가 충분히 가까운 pixel만 제거한다.

\[
|D_t^v(u)-\widehat D^v_{\text{robot},t}(u)|\le\epsilon_r
\]

따라서 로봇의 **부피 근처**가 아니라 실제로 관측된 **로봇 표면과 일치하는 pixel**만 제거된다. 손가락 사이의 물체나 gripper 앞의 물체는 depth가 다르면 남는다.

### 4.2 Depth-continuous surface graph 만들기

로봇 pixel을 제거한 뒤, 각 카메라의 유효 depth pixel을 촬영 시점의 카메라 pose를 이용해 공통 robot frame의 3D 점으로 변환한다. 이 점들을 voxel 단위의 node로 묶는다. 각 node는 다음 정보를 가진다.

- 중심 위치 \(x_i\)
- 가능한 경우 표면 normal \(n_i\)
- 어떤 영상 patch와 pixel에서 왔는지에 대한 연결 정보

단순히 3D 거리가 가까운 점끼리 연결하면 서로 닿아 있는 두 과일이 하나의 물체로 합쳐질 수 있다. AG3S는 다음 조건을 모두 만족할 때만 node 사이에 edge를 만든다.

- 원래 영상에서 이웃한 pixel들이 두 node를 연결한다.
- 두 pixel의 depth 차이가 임계값보다 작다.
- 두 node가 3D에서도 가깝다.

즉, **3D 근접성만으로 보이는 depth 불연속을 건너지 못하게 한다.** 허용된 edge의 유사도는 다음과 같다.

\[
w_{ij}=\exp\left(
-\frac{\|x_i-x_j\|^2}{\sigma_x^2}
-\frac{1-|n_i^\top n_j|}{\sigma_n}
\right)
\]

위치가 가깝고 표면 방향이 비슷할수록 큰 weight를 가진다. local PCA로 normal을 신뢰성 있게 구하지 못하면 normal 항은 생략한다.

RANSAC으로 찾은 support plane, 예를 들어 테이블 표면은 “목표가 아님”을 나타내는 negative evidence가 된다. 동시에 support는 계속 보호해야 할 장면 geometry로 남는다. 여기에는 별도의 semantic class label을 사용하지 않는다.

### 4.3 명사 attention을 seed로 바꾸기

VLA에서 지시문의 목표 명사와 image patch 사이 attention 점수 \(a_p\)를 읽는다. attention은 “대략 어느 patch가 사과와 관련 있는가”를 알려 주지만, patch 해상도가 거칠어서 물체의 정밀 경계를 직접 만들기에는 부족하다. 따라서 attention은 경계 자체가 아니라 graph 위의 **seed**, 즉 시작 증거로만 사용한다.

두 가지 단순한 방법에는 문제가 있다.

- attention 최고점 하나만 3D로 옮기면 depth sample이 너무 적고 프레임마다 쉽게 흔들린다.
- patch 점수를 그 아래 모든 sample에 그대로 복사하면, pixel이 많은 큰 표면이 작은 목표보다 지나치게 강한 표를 얻는다.

AG3S는 robot과 support가 아닌 pixel을 충분히 포함한 상위 patch 집합 \(\mathcal P\)를 선택한 뒤, 각 patch가 가진 evidence 총량을 그 patch가 관측한 node에 pixel 비율대로 나눈다.

\[
\lambda_i^+=\sum_{p\in\mathcal P}
\frac{N_{ip}}{N_p}
\frac{a_p}{\max_{r\in\mathcal P}a_r}
\]

- \(N_p\): patch \(p\)에 남은 전체 pixel 수
- \(N_{ip}\): 그중 node \(i\)에 속하는 pixel 수
- \(\lambda_i^+\): node \(i\)가 목표라는 positive evidence

이를 **patch mass conservation**으로 이해하면 쉽다. 한 patch가 넓은 표면을 본다고 해서 총 투표량이 커지지 않고, 가진 한정된 evidence를 node들 사이에 나눈다. 유효한 seed가 전혀 없으면 새로운 목표 영역을 승인하지 않는다.

### 4.4 Seed를 연속 표면으로 퍼뜨리기

support node에는 negative evidence \(\lambda_i^-=1\), 나머지에는 0을 준다. positive와 negative evidence가 있는 node를 기준점으로 삼아 graph 전체의 membership \(m_i\), 즉 각 node가 목표에 속할 정도를 계산한다.

\[
m^*=\arg\min_m (m-y)^\top\Lambda(m-y)+m^\top L_{\mathcal G}m
\]

이는 다음 선형 시스템을 푸는 것과 같다.

\[
(\Lambda+L_{\mathcal G})m^*=\Lambda y
\]

두 항의 의미는 간단하다.

- \((m-y)^\top\Lambda(m-y)\): seed나 support 같은 관측 증거를 따르도록 한다.
- \(m^\top L_{\mathcal G}m\): 강하게 연결된 이웃 node는 비슷한 membership을 갖도록 한다.

따라서 **unary evidence가 어느 표면인지 지정하고, geometry가 evidence가 어디까지 흐를 수 있는지 제한한다.** evidence가 전혀 없는 connected component는 자동으로 free space가 되지 않고 unknown으로 남는다.

마지막으로 membership을 threshold하고, seed mass가 가장 큰 connected region을 골라 현재 프레임의 기본 목표 표면 \(\mathcal T_t^0\)로 만든다. 시각 feature를 graph edge나 unary에 섞지 않는 이유는, robot 제거와 seed가 올바를 때 feature가 정밀 경계를 개선하지 않았기 때문이다. 이 방법에서 geometry는 extent를, feature는 다음 절의 identity를 담당한다.

## 5. ATS 2단계: 시간에 따라 같은 물체 유지하기

Attention은 다음 프레임에서 옆 물체로 이동할 수 있다. 이를 그대로 믿으면 목표 identity가 바뀐다. AG3S는 VLA 내부 feature를 이용해 현재 선택한 영역이 처음 선택한 목표와 같은지 확인한다.

### 5.1 Feature 중심화와 region descriptor

원래 VLA patch feature에는 화면 전체가 공유하는 큰 공통 성분이 있다. 그 결과 서로 다른 물체도 cosine similarity가 약 0.97로 높게 나올 수 있다. 각 patch의 unit feature \(u_{p,t}\)에서 현재 프레임의 평균 \(\bar u_t\)를 빼고 다시 정규화한다.

\[
\widetilde u_{p,t}=\operatorname{norm}(u_{p,t}-\bar u_t)
\]

영역 \(R\)의 descriptor는 각 patch 안에서 그 영역이 차지하는 pixel 수 \(b_p(R)\)로 중심화된 feature를 가중합하여 만든다.

\[
g_t(R)=\operatorname{norm}\left(\sum_p b_p(R)\widetilde u_{p,t}\right)
\]

프레임 평균을 제거함으로써 장면 공통 정보보다 물체 간 차이가 더 잘 드러난다.

### 5.2 Reference를 이용한 재연결

처음으로 유효하게 선택된 base region의 descriptor를 reference \(g_\star\)로 저장한다. 이후 프레임에서는 robot과 support를 제외한 graph의 크기 조건을 만족하는 connected component들을 후보 \(\mathcal C_t\)로 만들고, reference와 가장 유사한 후보 \(R_t^\star\)를 찾는다.

현재 attention 기반 영역보다 다른 후보가 reference와 더 비슷하고, 그 차이가 margin \(\kappa\) 이상이면 후보로 교체한다.

\[
g_t(R_t^\star)^\top g_\star
-g_t(\mathcal T_t^0)^\top g_\star\ge\kappa
\]

이 switch는 이미 관측된 connected region 중 하나를 고를 뿐, feature로 표면 경계를 새로 그리지 않는다. 물체가 실제로 잡혀 attached 상태가 되면 carried identity를 고정해야 하므로 재연결을 중지한다.

### 5.3 Causal consistency check와 alert

Reference 비교와 별도로, 현재 선택 영역을 바로 이전에 승인한 영역의 descriptor와 비교하는 causal memory chain을 유지한다.

\[
c_t=g_t(\mathcal T_t)^\top g_{\text{chain},t-1}
\]

\(c_t\)가 고정 임계값보다 낮으면 alert를 발생시킨다. 중요한 점은 alert가 곧바로 tracking을 중단하는 것이 아니라는 점이다.

- 영역 추적은 계속한다.
- 그러나 그 영역에 **새로운 contact exception은 허용하지 않는다.**
- switch된 영역도 geometry를 허용하기 전에 consistency 검사를 받는다.
- 비교할 과거 정보가 없으면 “일치”가 아니라 “판단 불가”로 처리한다.

Descriptor memory는 오직 identity를 기억한다. 과거에 본 표면을 현재도 있다고 가정하거나, 현재 관측되지 않은 공간을 접촉 가능하게 만들지는 않는다.

## 6. HSR: 작업 의도와 실제 grasp를 분리하기

같은 사과도 접근 중에는 손가락이 들어갈 목표이고, 운반 중에는 로봇에 붙은 충돌 geometry이며, 놓은 뒤에는 다시 장면 geometry다. 이 역할 변화를 결정하기 위해 HSR은 두 상태를 따로 관리한다.

### 6.1 Task phase: 정책이 무엇을 하려는가

행동용으로 fine-tuning된 VLA는 현재 단계를 말로 출력하지 않는다. AG3S는 VLA prefix의 pooled key/value feature \(f_t\)에 작은 linear probe를 붙여 다음 세 phase의 확률을 계산한다.

- **pick**: 목표에 접근하고 잡는 단계
- **place**: 잡은 물체를 옮기고 놓는 단계
- **home**: 작업 후 손을 복귀시키는 단계

\[
p_t=\operatorname{softmax}\left(W\frac{f_t-\bar f}{\sigma_f}+b\right)
\]

VLA 본체는 frozen 상태이고, simulator 개발 episode의 phase label로 normalization과 linear classifier만 학습한다.

이 probe는 영상 변화에는 민감하지만 robot state 문자열을 직접 보지 않는다. 그래서 두 번째 regularized linear classifier가 다음 신호를 함께 사용해 최종 phase \(s_t\)를 보완한다.

- 첫 probe의 log-probability
- gripper opening
- 목표 centroid와 더 가까운 손 사이의 거리
- 추정 support 위에서 목표 centroid의 높이
- 목표 region 크기의 log 값

한 프레임의 오분류로 규칙이 바뀌지 않도록 같은 추정이 여러 연속 관측에서 지속되어야 phase 변경을 확정한다.

- confirmed **pick**: 새 목표 채택과 조건을 만족한 재연결 허용
- confirmed **place**: 조작 중인 identity 고정
- confirmed **home**: 실제 release가 확인된 뒤에만 identity 제거

### 6.2 Grasp state: 실제로 무엇이 손과 함께 움직이는가

Phase는 정책의 **의도**이지 물리적 접촉의 증거가 아니다. 정책이 place 단계라고 생각해도 물체를 놓쳤을 수 있다. 따라서 독립적인 grasp state \(z_t\)를 유지한다.

\[
z_t\in\{\textsc{free},\textsc{held},\textsc{uncertain}\}
\]

판단에는 다음 실제 신호를 사용한다.

- 측정된 gripper opening
- 실행된 gripper command
- 목표와 손의 움직임이 일치하는지 여부

confirmed attachment일 때만 물체를 carried geometry로 바꾼다. 신호가 충돌하면 uncertain으로 두고, 새로운 permissive transition을 모두 막는다. 따라서 “정책이 잡으려 한다”와 “물체가 실제로 잡혔다”가 섞이지 않는다.

## 7. LTE: 링크와 상태에 맞는 거리장 만들기

### 7.1 실제로 관측되고 승인된 표면만 제외하기

목표 중심 주위의 구나 bounding volume 전체를 장애물에서 지우면 옆 물체나 아직 보지 못한 빈 공간까지 안전하다고 잘못 판단할 수 있다. LTE가 거리장에서 제외할 수 있는 sample 집합 \(\mathcal X_t\)는 다음 조건을 모두 만족해야 한다.

\[
\mathcal X_t=\left\{x\in\mathcal S_t\cap\mathcal B_t:
d_T(x)\le\rho,\;
d_T(x)+\tau<d_N(x),\;
\nu_t(x)=1\right\}
\]

- \(\mathcal S_t\): 현재 장면 sample
- \(\mathcal B_t\): 계산 범위를 제한하는 local search bound일 뿐, 제거할 부피 자체는 아님
- \(d_T(x)\): 승인된 목표 표면까지의 거리
- \(d_N(x)\): 경쟁 표면 및 보호된 support까지의 거리
- \(\rho\): 목표 표면과 충분히 가까운 sample만 허용
- \(\tau\): 목표와 이웃 표면 사이에 남기는 안전 boundary band
- \(\nu_t(x)\): 승인된 membership, 현재 관측, support/경쟁 표면의 veto 없음이라는 조건

즉, 목표 표면과 가깝고, 다른 표면보다 명확히 목표 쪽이며, 지금 실제로 관측되고 승인된 sample만 제외된다. 애매한 sample은 계속 장애물로 남는다. 과거의 identity memory만으로 현재 보이지 않는 공간을 비우지 않는다.

### 7.2 링크별 거리장 선택

\(d_t\)는 robot을 제거한 전체 장면의 depth-fused distance field이고, \(d_t^{\setminus\mathcal X}\)는 승인된 target sample \(\mathcal X_t\)만 제외한 local alternative field다.

물체가 아직 붙지 않은 상태에서 sphere \(i\)가 사용할 거리장은 다음과 같다.

\[
\delta_{i,t}(x)=
\begin{cases}
d_t^{\setminus\mathcal X}(x), & i\in\mathcal L \land e_t=1\\
d_t(x), & \text{otherwise}
\end{cases}
\]

여기서 \(\mathcal L\)은 접촉 허용 링크 집합이고,

\[
e_t=v_t\land[s_t\in\mathcal S_{\text{ok}}]
\]

이다. 즉, grounding이 존재하고 alert가 없는 유효 상태 \(v_t\)이며, 현재 phase가 접촉 허용 phase 집합 \(\mathcal S_{\text{ok}}\)에 포함될 때만 exception이 켜진다.

결과적으로:

- 허용된 손가락 링크만 목표가 빠진 field를 사용한다.
- 손목, 팔, 반대쪽 gripper 등 나머지는 목표를 포함한 full scene field를 사용한다.
- local field의 범위 밖에서는 항상 full scene field로 fallback한다.
- unknown space에는 exception을 주지 않는다.
- \(\mathcal X_t\)는 매 request마다 현재 sample에서 다시 만들기 때문에 membership이 철회되면 다음 request에서 즉시 장애물이 복구된다.

### 7.3 Grasp, carry, release

confirmed grasp가 발생하면 closing 직전 승인된 목표 geometry를 hand frame으로 변환하고 attached query sphere들로 표현한다.

- 물체 자신의 depth sample은 환경 layer에서 빠진다. 이는 주변 부피를 지워서가 아니라 **소유권이 환경에서 손으로 이동했기 때문**이다.
- attached sphere는 로봇 sphere와 함께 남은 환경을 질의한다. 따라서 운반 중 사과가 기둥에 부딪히는 경로를 막을 수 있다.
- 설정된 placement rule은 목적지에서 물체와 support의 접촉을 허용할 수 있다. 이때도 robot link가 그 support에 닿는 것은 허용되지 않는다.
- release가 확인되면 저장한 물체 geometry를 장면으로 되돌린다.
- release 여부가 uncertain이면 새로운 permissive transition을 막는다. 오래된 attachment 때문에 공간이 비어 있는 상태가 되지 않게 하기 위함이다.

## 8. CCR: action chunk를 수정하고 검증하기

VLA는 한 번에 여러 waypoint로 이루어진 action chunk를 제안한다. 첫 command 하나만 다듬으면 뒤쪽 chunk가 검사되지 않고, chunk 전체를 오래된 장면에 맞춰 확정하면 환경 정보가 stale해진다. CCR은 앞의 \(J\)개 waypoint를 최적화하되 그보다 짧은 \(K\)개만 실행한다.

\[
\begin{aligned}
\min_Q\quad
&w_t\|Q-Q^{\mathrm{ref}}\|_F^2
+w_s\|\Delta^2Q\|_F^2
+w_c\|Q_{\mathcal O}-\bar Q^{\mathrm{prev}}_{\mathcal O}\|_F^2\\
\text{s.t.}\quad
&h_{ik}(q_k)\ge0\quad\forall i,k,\qquad Q\in\mathcal Q
\end{aligned}
\]

목적함수의 세 항은 각각 다음 의미다.

1. **Tracking 항**: 수정한 궤적 \(Q\)가 VLA의 원래 제안 \(Q^{\mathrm{ref}}\)에서 지나치게 벗어나지 않게 한다.
2. **Smoothness 항**: 2차 차분 \(\Delta^2Q\)를 줄여 궤적이 급격히 꺾이지 않게 한다.
3. **Continuity 항**: 이전 계획과 현재 계획이 겹치는 구간 \(\mathcal O\)에서 계획이 갑자기 바뀌지 않게 한다. 실행 feedback으로 시간축을 맞춰 동일한 미래 구간끼리 비교한다.

제약 \(\mathcal Q\)는 관절 한계와 motion limit를 포함하고, 앞에서 정의한 clearance 조건을 모든 sphere와 waypoint에 적용한다.

최적화는 SQP(Sequential Quadratic Programming)로 수행한다. 각 반복에서 비선형 clearance를 trust region 안에서 선형화하고, 만들어진 quadratic subproblem을 OSQP로 푼다.

### 실행 전 마지막 검증

최적화가 답을 냈다는 사실만으로 바로 실행하지 않는다.

1. 최적화한 \(J\)개 waypoint 중 앞의 \(K\le J\)개를 candidate prefix로 정한다. 나머지는 look-ahead 용도다.
2. 측정된 현재 pose에서 시작해 prefix waypoint와 그 사이의 보간 자세를 모두 검사한다.
3. 모든 robot sphere와 attached sphere의 **비선형 실제 거리**를 다시 계산한다.
4. refiner와 동일한 geometry snapshot 및 상태를 사용해 clearance를 확인한다.
5. 하나라도 실패하면 실행하지 않고 **hold**한다.

Hold 중에는 측정된 팔 pose를 유지하고, 물체를 놓치지 않도록 현재 grasp와 일관된 gripper command를 유지한 채 다음 관측을 기다린다. Hold는 성공이 아니라 safety intervention으로 기록된다.

## 9. 사과를 집어 옮기는 전체 시나리오

### 단계 A: 접근 전

1. VLA가 RGB, 로봇 상태, “사과를 바구니에 놓아라”라는 지시로 action chunk를 만든다.
2. ATS가 “사과”에 대한 noun attention을 graph seed로 사용한다.
3. Depth continuity가 사과와 옆 과일 또는 테이블의 경계를 나눈다.
4. 첫 유효 사과 descriptor를 reference로 저장한다.
5. HSR이 pick phase를 연속적으로 확인한다.
6. 유효한 grounding이고 alert가 없으면 손가락에만 사과 표면 exception을 준다.
7. CCR은 손가락이 사과에는 접근할 수 있지만 손목과 나머지 링크는 충돌하지 않는 궤적으로 chunk를 다듬는다.

### 단계 B: 프레임 사이 추적

1. 다음 프레임의 attention이 옆 물체로 흔들릴 수 있다.
2. 중심화된 descriptor가 처음 사과와 가장 비슷한 connected region을 찾는다.
3. 다른 후보가 충분한 margin으로 이기면 그 후보로 재연결한다.
4. 이전 승인 영역과의 consistency가 낮으면 alert를 발생시켜 새 exception을 막는다.

### 단계 C: grasp

1. HSR의 phase가 place 쪽으로 변하는 것만으로는 attachment가 활성화되지 않는다.
2. gripper opening, 실행 command, 손과 사과의 동반 움직임이 grasp를 확인해야 한다.
3. confirmed held가 되면 사과의 pre-closure geometry를 hand frame의 attached sphere로 바꾼다.
4. 이후 identity는 고정되고 attention drift 때문에 다른 물체로 바뀌지 않는다.

### 단계 D: carry와 place

1. attached 사과와 robot sphere가 모두 남은 환경에 대한 clearance를 검사한다.
2. 따라서 손 자체가 기둥을 피하는 것뿐 아니라 손에 든 사과도 기둥을 피한다.
3. 목적지에서는 설정된 규칙에 따라 사과와 바구니/support의 접촉만 허용할 수 있다.
4. robot link는 여전히 support를 피한다.

### 단계 E: release와 복귀

1. 실제 release가 확인되면 사과 geometry를 환경으로 돌려놓는다.
2. uncertain release이면 안전한 상태를 유지하고 새 exception을 열지 않는다.
3. confirmed home과 physical release가 모두 충족된 뒤에 저장한 manipulated identity를 지운다.

## 10. 반드시 구분해야 하는 개념

### Attention은 segmentation mask가 아니다

Attention은 어떤 patch가 목표와 관련 있는지 알려 주는 seed다. 정확한 표면 경계는 depth-continuous graph가 정한다.

### Identity memory는 geometry memory가 아니다

Feature memory는 “같은 물체인가?”만 판단한다. 과거에 관측한 표면이나 현재 보이지 않는 공간에 접촉 권한을 주지 않는다.

### Phase는 grasp가 아니다

Phase는 정책의 의도이고, grasp state는 물리적 사실이다. 물체를 carried geometry로 바꾸려면 반드시 물리적 attachment가 확인되어야 한다.

### Target exemption은 물체 주변 부피 삭제가 아니다

현재 관측되고 승인된 target surface sample만 선택적으로 제외한다. 이웃 표면, support, 애매한 경계, unknown space는 계속 보호한다.

### Exception은 모든 링크에 적용되지 않는다

접촉 링크 \(\mathcal L\)만 target-excluded field를 사용한다. 나머지 링크는 항상 full scene을 본다.

### Refine 성공은 실행 허가와 같지 않다

최적화 뒤에도 waypoint 사이를 포함한 비선형 거리 검사를 통과해야 한다. 실패하면 hold한다.

## 11. 모듈별 입력과 출력 요약

| 모듈 | 주요 입력 | 주요 출력 | 실패·불확실 시 행동 |
|---|---|---|---|
| ATS grounding | noun attention, depth, camera pose, robot model, support plane | 기본 목표 표면 \(\mathcal T_t^0\) | seed가 없으면 새 목표를 승인하지 않음 |
| ATS memory | 중심화된 VLA feature, 현재/과거 region | 선택 목표 \(\mathcal T_t\), identity, alert | 불일치 시 추적은 계속하되 새 exception 차단 |
| HSR phase | pooled hidden feature, gripper/거리/높이/크기 | confirmed phase \(s_t\) | 연속 관측 전에는 rule 변경 안 함 |
| HSR grasp | gripper 상태·명령, target–hand motion | \(z_t\): free/held/uncertain | uncertain이면 permissive transition 차단 |
| LTE | 목표 표면, validity, phase, grasp, link set | sphere별 거리장 \(\delta_{i,t}\), attached geometry | 조건이 부족하면 full scene field 사용 |
| CCR | VLA chunk, 거리장, 이전 계획, 현재 pose | 검증된 실행 prefix 또는 hold | 검사 실패 시 pose 유지 후 새 관측 대기 |

## 12. 한 문장 요약

**AG3S는 frozen VLA의 attention으로 목표를 가리키고, depth로 그 표면의 끝을 정하며, 중심화된 feature로 같은 물체인지 추적하고, phase와 물리적 grasp 증거로 접촉 권한의 시점을 정한 뒤, 링크별 거리장과 검증된 chunk refinement를 통해 필요한 접촉만 허용한다.**

