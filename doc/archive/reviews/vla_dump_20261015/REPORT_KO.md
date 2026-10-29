**VLA 전체 내부 텐서로 가능한 연구와 단일 프레임 탐색 결과**
후속 검토: 같은 record의 GT와 depth replay를 연결한 [3D grounding 추가 평가](DEPTH_GROUNDING_REVIEW_KO.md)가 있다. 아래 최초 분석의 “융합으로 map이 좁아짐”은 정확도 개선을 뜻하지 않으며, 후속 계산에서 seed 오염과 noun-seeded feature 확장의 차이를 확인했다.

작성: 2026-10-15 KST. 아래의 수치는 실제 덤프에서 계산한 관찰이고, 개선 방향은 아직 검증하지 않은 가설이다. 원본 데이터·모델·serving 코드는 수정하지 않았다.

**1. 연구 방향에 대한 판단**

가장 가치 있는 방향은 **언어로 지정된 물체, 행동 계산에 사용되는 표면, 물리적으로 접촉을 허용할 영역을 분리한 뒤 연결하는 것**이다. 저장소의 identity / extent / permission 연구 목표와 직접 맞는다. 단순히 더 선명한 attention 그림을 만드는 것은 충분한 성과가 아니다. 이 데이터는 “왜 이 물체인가”, “어느 표면까지 같은 물체인가”, “어떤 행동 구간에서 필요한가”를 나누어 검증하게 해준다.

우선순위는 다음과 같다.

| 순서 | 가설 | 필요한 연산·데이터 | 연구에 기여하는 지점 |
|---|---|---|---|
| 1 | action query마다 다른 역할이 있어 `last` 한 장으로 실제 실행을 대표하기 어렵다 | query별 attention·value·실제 실행 chunk 길이 | 즉시 실행 relevance와 미래 relevance 구분 |
| 2 | 명사와 행동의 공통 증거는 identity seed에, 시각 특징과 depth는 extent에 적합하다 | map 융합, feature 내적, graph Laplacian, depth | 물체 경계를 보존하고 주변 장애물의 과도한 제거 감소 |
| 3 | HSR의 높은 확률 중 일부는 state·trajectory 위치로 설명될 수 있다 | state-only baseline, modality ablation, episode split | learned phase의 실제 추가 정보 확인 |
| 4 | 행동에 민감한 feature 방향은 attention 크기와 다르다 | value 방향·output projection·gate·Jacobian·intervention | 출력과 연결되는 grounding 및 오류 원인 규명 |
| 5 | denoise 경로는 거의 직선이지만 map은 변할 수 있다 | flow 방향 내적·경로 길이·step ablation | 계산 축소 가능성과 grounding 안정성 분리 |
| 6 | multi-view feature 일치와 기하 일치의 결합으로 관측 표면의 신뢰도를 추정할 수 있다 | calibration, depth, covariance, surface normal | 가림 복구·target extent·neighbor 보존 |

**2. 실제로 확인한 데이터와 해석상의 조건**

입력 디렉터리:
`/mnt/dev/AG3S_VLA-storage/results/development/20261015T150325+0900_ddc7a00060c4_d857a08dcb63/vla_dump/`

실제 full dump는 `frame_022` 하나이고 원래 `t_step=176`이다. prompt는 `Place the orange into the basket.`이다. 공간 patch는 카메라마다 16×16, 총 768개다. layer/head 번호는 덤프의 **0-based 번호**다.

- text slot은 200개지만 유효 token은 72개이다. 명령문은 앞부분, `State:` 이후에는 숫자로 직렬화한 로봇 자세, 마지막에는 `Action:`이 들어 있다. 200개를 모두 자연어 feature로 평균하면 padding과 state를 섞게 된다.
- `noun_prefix_row=773`은 text index 5의 `orange`다. `basket`은 prefix row 776이다. 두 명사는 manipulated object와 destination이라는 다른 역할을 가진다.
- action expert hidden은 **1024차원**, prefix hidden은 **2048차원**이다. Q/K/V의 head 차원은 256이고, 이 모델은 8 query head에 1 KV head를 공유한다. 서로 다른 residual 공간을 직접 내적하면 안 된다.
- action 32차원 중 출력에는 첫 16차원이 사용된다. 현재 config/ALOHA output transform상 7 joint + gripper씩 양팔이다. joint delta를 정규화한 내부 좌표와 역변환한 절대 목표 joint를 구분해야 한다. 나머지 padding action 차원의 noise 제거가 전체 norm을 부풀릴 수 있어 분석은 32차원과 active 16차원을 따로 기록했다.
- prefix는 image/text 전체가 서로 볼 수 있는 양방향 attention이다. 저장된 `lm_next_token_logprob`는 다음 token을 가린 autoregressive 확률이 아니다. 미래 prompt 정보가 이미 보이므로 perplexity나 작업 확신도로 사용할 수 없다. LM-head 결과는 현 입력에서의 vocabulary 투영 진단 정도로 사용한다.
- 덤프 생성기의 `attn_logits`는 mask 적용 전 재계산 값이다. softmax를 재구성할 때 반드시 prefix/suffix mask와 RoPE, Q scaling을 복원해야 한다. FP16 저장 오차도 있다.
- SigLIP attention은 직접 캡처가 아니라 별도 float32 재계산이다. `sa` 원본 대비 최대 절대 차이가 camera별 약 0.499–0.574이다. 상대 오차를 재평가하기 전 정확한 원본 attention으로 인증할 수 없다.
- prefix 원본 대비 hidden/KV/probs 오차는 추출 로그상 0이다. suffix는 원본 대비 `v_t` 최대 차이 0.015625–0.046875, 최종 normalized x0 차이 0.00390625이다. 이 값은 original 경로와 replay 경로가 진행되면서 x_t까지 달라지는 비교이므로 순수한 동일 입력 layer 재계산 오차와 구분한다.
- 이번 replay action과 recorded serving action의 원소 단위 RMSE는 0.008607, maxabs는 0.043523이다. 단위가 섞인 요약치이고 동일 noise를 보장하지 않으므로 정책 오차나 mm 단위 차이로 해석하지 않는다.
- depth, calibration, 실제 contact/force, 독립 phase/segmentation GT, 모델 weight/Jacobian은 이 frame NPZ에 없다. “내부 데이터 전체”와 “물리 상태 및 인과 분석에 필요한 데이터 전체”는 다르다.

로컬 확인 근거: `third_party/openpi/src/openpi/models/pi0.py::embed_prefix`, `gemma.py::Attention`, `training/config.py::LeRobotAlohaDataConfig`, `policies/aloha_policy.py::AlohaOutputs`, 원본 `vla_full_dump.py` 및 `parity.json`.

**3. 직접 계산한 결과**

재현 코드: [analyze.py](analyze.py). 전체 수치: [metrics.json](metrics.json). [융합 그림](maps.png), [query 비교](queries.png), [denoise 그림](denoise.png).

map의 entropy·JS·camera 비중은 **image 768개 내부에서 합이 1이 되도록 정규화**해 계산했다. 원래 image mass도 별도로 보존했다. JS는 자연로그를 쓰므로 범위가 0–ln(2), 약 0–0.6931이다. 유효 patch 수는 `exp(entropy)`이며 실제 물체 면적이나 segmentation 품질이 아니다.

| 측정 | 실제 수치 | 의미와 한계 |
|---|---:|---|
| L8H2, denoise 0, last query의 image mass | 0.9010 | 전체 key attention 중 image에 할당된 질량 |
| 같은 map의 오른손 카메라 비중 | 98.27% | 세 카메라를 받더라도 이 head/query는 사실상 한 view에 집중 |
| orange noun map의 오른손 카메라 비중 | 81.73% | action map과 분포가 다름 |
| action–noun cosine / JS / top20 교집합 비율 | 0.5447 / 0.3295 / 35% | 두 map을 동일 신호로 취급할 수 없음 |
| action / noun / 곱셈 융합의 유효 patch 수 | 32.83 / 128.19 / 8.40 | 곱셈은 매우 좁은 seed를 만듦; extent coverage 개선 증거는 아님 |
| 곱셈 융합의 오른손 카메라 비중 | 99.89% | 단순 곱셈은 view 편중을 오히려 강화 |
| attention vs attention×value norm cosine | 0.9981 | 이 head에서는 norm 보정이 새 정보를 거의 추가하지 않음 |
| 같은 두 map의 top20 교집합 비율 | 95% | 가장 단순한 value 가중치 개선의 우선순위는 낮음 |
| first query vs last query image JS | 0.6470 | horizon 위치에 따른 map 차이가 매우 큼 |
| first query / first8 평균 / last의 원래 image mass | 0.7601 / 0.9275 / 0.9010 | 차이가 단지 image mass가 거의 0이라 생긴 현상은 아님 |
| 전체 query 평균 vs last JS | 0.2203 | query pooling 방식 역시 중요 |
| denoise 0 vs 9의 last map JS | 0.1084 | denoise 중 attention은 변함 |
| active16 인접 flow 방향 cosine | 0.99857–0.99996 | 한 noise 경로에서 방향이 거의 유지됨 |
| active16 경로 길이 / 시작–끝 거리 | 1.000306 | normalized action 공간에서 거의 직선인 경로 |
| active16 마지막 step 시작점에서 최종 x0까지 L2 거리 | 3.1941 | 방향 안정이 이미 최종 행동에 도달했다는 뜻은 아님 |
| weighted value covariance participation ratio | 2.96 | 이 분포가 가중한 256D value 변동은 소수 방향에 집중 |

그림에서 첫 query/first8 평균은 오른손 이미지의 상단·가장자리 쪽에 집중하고, last는 과일·gripper 주변에 집중한다. 이것은 **first query가 더 좋은 grounding이라는 증거가 아니다**. 위치성 sink, head의 query별 특화, 실제 미래 동작 의미를 구분해야 한다. first8은 비교용 임의 창이며 실제 controller가 8개를 실행한다고 가정하지 않았다.

value 방향도 직접 확인했다. 이미지 token 중 합성 head output 방향과 음의 내적을 가진 token은 46.35%였지만, 그 token들이 가진 image attention 질량은 **약 0.207%**에 불과했다. patch 개수만 보면 상쇄가 심해 보이지만 가중 기여는 작다. 전체 token Gram 행렬의 대각합은 7.765, 비대각합은 +55.109로 이 사례의 총 교차항은 강화 방향이다. `||Σc_i|| / Σ||c_i||=0.6601`은 벡터 합성의 비정렬 정도이며, 이것만으로 음의 상호작용이 지배적이라고 해석하면 안 된다.

HSR place 확률은 0.99865지만 독립 phase GT가 없어 정확도를 주장하지 않는다. 이 프레임은 기존 ep1839 기록에서 왔으며 로컬 평가 문서상 test 구간에 해당한다. **이 분석으로 head·threshold를 선택했다면 이 사례를 새 held-out 성능 검증으로 재사용하면 안 된다.**

**4. 내적: “어디를 보는가”에서 “어떤 계산 방향으로 기여하는가”로**

하나의 layer/head/query를 고정하면:

\[
a_{ji}=\operatorname{softmax}_i(q_j^\top k_i/\sqrt d+M_{ji}),\quad
c_{ji}=a_{ji}v_i,\quad y_j=\sum_i c_{ji}.
\]

여기서 RoPE를 적용한 Q/K를 쓰고, 이미 scaled인 Q에는 `1/sqrt(d)`를 다시 곱하지 않는다. QK 내적은 모델이 실제 학습한 호환도이고, raw image/text embedding cosine은 단지 탐색적 유사도다. projector 출력이 2048차원이라고 text embedding과 CLIP처럼 정렬되었다고 보장되지 않는다.

후보 점수는 다음과 같다.

- `a_ji ||v_i||`: 기여의 크기. 즉시 계산 가능하지만 이번 head에서는 거의 변화가 없었다.
- `c_ji^T y_j / ||y_j||`: head 결과 방향으로의 signed contribution. 정확한 국소 대수 분해이지만 최종 action 인과 효과는 아니다.
- `c_i^T c_k`: 두 token이 같은 계산 방향을 지지하는지 확인. `||Σc_i||²=Σ||c_i||²+2Σ_{i<k}c_i^T c_k`로 상쇄·강화를 구분한다.
- `δa_i ≈ J_(a←r_l) g_l W_O^h c_ji`: checkpoint의 output projection, adaRMS gate, 이후 layer 및 denoising 경로의 Jacobian까지 반영한 **국소 선형 근사**. finite intervention으로 검증해야 한다.

prefix의 2048D residual과 suffix의 1024D residual을 직접 비교하지 않는다. 서로 다른 head·layer의 value 좌표도 먼저 공통 output 공간으로 투영하거나 representation similarity로 비교한다. physical force 방향과 latent 방향은 다르다.

value norm을 함께 분석하는 발상 자체는 기존 연구가 있다. AG3S에서 입증할 것은 해당 점수가 실제 target–neighbor 분리와 실행 결과를 개선하는지다. [Kobayashi et al., Attention is Not Only a Weight](https://arxiv.org/abs/2004.10102). Attention weight만으로 설명을 확정할 수 없다는 실험적 선행 근거도 있다. [Jain & Wallace, Attention is not Explanation](https://arxiv.org/abs/1902.10186).

**5. 외적: covariance, 물체 관계, 행동 민감도**

고차원 feature의 “외적”은 두 개념을 구분한다.

**Outer product** `uv^T`는 임의 차원의 벡터를 관계 행렬로 만든다. **Cross product** `u×v`는 여기서는 3D 물리 벡터에 적용한다. 2048D embedding에서 임의의 세 좌표를 뽑아 cross product를 계산해도 물리 의미가 생기지 않는다.

(a) **물체 feature의 분포**

\[
\mu=\sum_i p_i f_i,\qquad
\Sigma_f=\sum_i p_i(f_i-\mu)(f_i-\mu)^\top.
\]

seed 주변의 feature가 하나의 일관된 군집인지, 이웃 장애물까지 섞였는지 보는 후보다. 이번 value covariance의 participation ratio `(tr Σ)²/tr(Σ²)=2.96`은 국소 저차원 구조를 보여준다. 그러나 patch들은 독립 표본이 아니고 attention이 몇 patch에 집중해도 저차원이 된다. 낮은 rank만으로 같은 물체라고 판정하지 않는다. uniform weighting, shuffled seed, 다른 물체 seed와 비교한다.

(b) **명령·시각·행동의 상호작용**

episode별 paired sample로 `C_fa=E[(f-μ_f)(a-μ_a)^T]`, `C_ft`를 추정하고 SVD/CCA/저랭크 bilinear probe로 어떤 feature 방향이 어느 action 채널과 연결되는지 본다. raw outer product 한 개는 항상 rank≤1이므로 발견이 아니다. static pose를 통제한 held-out 예측이 필요하다. 모든 2048² feature를 저장하기보다 저랭크 투영·sketch를 사용한다.

(c) **최적화에서 보존할 행동 방향**

물리 단위로 스케일을 맞춘 outcome `y`와 normalized action `a` 사이 `J=∂y/∂a`로 `G=J^T W J`를 만들고, TO의 원래 행동 보존 비용을 `δa^T(G+λI)δa`로 시험한다. 해당 행렬은 local task-sensitivity metric이며 자동으로 Fisher information이 되는 것은 아니다. task outcome의 미분 가능한 정의 또는 검증된 surrogate가 필요하다. 충돌 제약은 별도로 유지하며 task-sensitive 방향이라고 완화하지 않는다.

**6. 융합: identity seed와 물체 전체 범위를 분리**

가장 단순한 비교군은 다음이다.

\[
p_{\rm seed}(i)\propto(p_{\rm noun}(i)+\epsilon)^\alpha
(p_{\rm action}(i)+\epsilon)^\beta.
\]

이번 계산은 ε=0, α=β=1이다. sharpen되었지만 표면 전체를 잃을 수 있으므로 **seed 후보**로 쓰고 extent 추정은 따로 한다. 동일 모델에서 나온 두 map은 독립 증거가 아니므로 이 식을 Bayesian posterior라고 부르지 않는다. mixture, geometric mean, raw attention을 함께 비교한다.

extent 확장은 SigLIP/prefix feature와 depth를 묶은 graph에서 시험할 수 있다.

\[
w_{ij}=\mathbf1[\text{valid geometric neighbors}]
\exp(-\|\hat f_i-\hat f_j\|^2/\sigma_f^2
-\|p_i-p_j\|^2/\sigma_p^2)
\]

\[
\min_{0\le m\le1}\sum_i\lambda_i(m_i-s_i)^2
+\eta\sum_{ij}w_{ij}(m_i-m_j)^2.
\]

`s_i`는 positive seed와 reliable negative seed, `λ_i`는 그 신뢰도다. depth discontinuity·normal 불일치·확인된 neighbor 경계에서는 연결을 끊는다. 그러면 단순 ball 제거보다 물체 표면을 따라 membership을 넓히는 설계가 된다. 비슷한 색의 인접 물체나 접촉 경계에서 graph가 새는지 평가해야 한다. patch는 혼합 surface를 포함하고 SigLIP은 이미 global mixing을 거쳤으므로 pixel 수준 segmentation으로 그대로 간주하지 않는다.

언어 역할도 분리한다. `orange` map은 manipulated identity 후보, `basket` map은 destination 후보, action map은 움직임에 필요한 증거 후보다. 운반 중 destination attention이 증가했다고 held identity를 basket으로 교체하거나 basket 전체를 collision exemption으로 만들지 않는다. phase/grasp feedback과 identity memory를 연결한다.

**7. 물리: 3D 정합·표면 normal·접근 방향**

추가로 depth, 원본-to-224 전처리 변환, intrinsic/extrinsic, timestamp를 연결하면 patch evidence를 공통 좌표로 옮길 수 있다.

\[
p_w=T_{wc}[D(u)K^{-1}\tilde u;1].
\]

view별 동일 patch index를 평균하는 대신 같은 관측 surface/voxel에 투영한다. 가려짐·depth validity·view angle을 고려하고, 동일 이미지에서 나온 여러 head를 독립 센서 수로 세지 않는다.

3D 좌표의 weighted covariance로 extent의 주축과 관측 분산을 추정한다. 물체 자체의 공간적 크기와 센서 측정 오차 covariance를 구분한다. 두 개를 같은 uncertainty로 합치면 큰 물체를 불확실한 물체로 오인한다. 실제 반복 관측·calibration 오차로 측정 신뢰도를 추정한다.

표면 이웃의 실제 3D 벡터로는 cross product가 유용하다.

\[
n\propto(p_2-p_1)\times(p_3-p_1),\quad
v_n=n^T v_{ee},\quad v_{tan}=v_{ee}-v_n n.
\]

이를 통해 예측한 손 이동이 표면을 향하는지, 접선 방향으로 미끄러지는지 분해한다. normal 방향은 카메라/물체 기준으로 일관되게 정해야 하고, 정지 물체 가정이 안 맞으면 상대속도를 쓴다. 실제 contact force가 있어야 `τ=r×F` 같은 torque 해석이 가능하다. attention을 힘으로 치환하지 않는다.

action에서 FK/Jacobian으로 얻는 속도와 signed distance `h=d-r-margin`의 변화를 비교하면 어느 링크·어느 구간에 충돌 위험이 생기는지 볼 수 있다. CBF를 도입하려면 실제 시스템 `dot x=f(x)+g(x)u`와 입력 제약에서 `∇h^T(f+gu)+α(h)≥0`의 적용 조건을 검토해야 한다. 현재 discrete joint-position TO에 이 식을 적는 것만으로 안전 보장이 생기지는 않는다. [Ames et al., Control Barrier Function Based Quadratic Programs](https://arxiv.org/abs/1609.06408).

**8. denoising을 동역학으로 분석하되 로봇 시간과 구분**

현재 replay는 `x_(s+1)=x_s-0.1 v_s`다. 여기서 flow time은 noise→action 생성 시간이며 로봇의 실제 실행 시간이 아니다. `v_s`는 joint velocity가 아니고 `||v_s||²`는 kinetic energy가 아니다. flow matching은 생성 과정의 vector field를 모델링한다. [Lipman et al., Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).

이번 active16 경로는 거의 직선이지만 각 step에서 계속 큰 이동이 남아 있었다. 따라서 `x_s`를 그대로 반환하는 early stop과, `x_final≈x_s-t_s v_s`로 남은 구간을 적분·외삽하는 것은 다르다. 외삽은 velocity가 남은 구간에 충분히 일정하다는 가설에 의존한다.

후속 실험은 동일 observation·동일 initial noise에서 10-step baseline과 5/2/1-step 재추론, 또는 적응형 solver를 비교한다. 이미 10-step 경로에서 얻은 v를 골라 쓰는 것은 저비용 solver의 실제 성능 검증이 아니다. action RMSE 외에도 FK endpoint, executed-prefix clearance, contact, 성공률, 실제 latency를 측정한다.

단일 seed 경로의 안정성은 uncertainty 추정이 아니다. uncertainty는 여러 noise seed·관측 perturbation의 action 분산, view disagreement, phase calibration과 함께 검증한다. 이를 바로 collision margin으로 바꾸지 말고 실제 clearance error와 calibration을 확인한다.

**9. phase·표현·MLP의 정보 분석**

HSR은 기존 KV_L4 pooling에 state/text 정보가 섞일 수 있다. 다음 비교가 먼저다.

| 입력 | 검증 질문 |
|---|---|
| normalized state + gripper만 | 자세·gripper만으로 어디까지 예측 가능한가? |
| state + 이전 state 변화, 명시적 strong rule | 비학습적 물리 정보보다 나은가? |
| image-token KV만 / task-text KV만 / state-token KV만 | 어느 token 위치에서 phase가 잘 읽히는가? |
| image+state / 기존 KV_L4 / 여러 layer의 작은 probe | 추가 정보가 전환 지연·오전환 감소로 이어지는가? |

단, image-token KV도 양방향 prefix layer를 거친 뒤에는 text/state 정보를 담을 수 있다. **token 위치 pooling 비교는 입력 modality 제거 실험이 아니다.** modality 기여를 보려면 state 재토큰화, image mask/대체, 명령 교체를 입력에서 실시하고 prefix를 다시 계산해야 한다. 원래 hidden에서 일부 row를 빼는 것만으로 state 의존성이 없어지지 않는다.

물리 grasp/release event를 독립 label로 쓰고 episode·scene·object 기준 split을 한다. adjacent frame random split은 피한다. phase accuracy뿐 아니라 false target switch, grasp 전환 지연, release 후 stale exemption을 본다. feature+state가 state-only보다 낫다는 것이 곧 조건부 mutual information의 정확한 추정은 아니지만, 추가 정보의 실용적 검증이다.

CKA는 서로 다른 layer/폭의 표현이 같은 paired samples에 대해 비슷한 관계를 가지는지 확인하는 데 쓸 수 있다. 한 frame의 768 patch CKA는 공간 구조 비교이고 시간에 걸친 phase representation 증거가 아니다. [Kornblith et al., Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414).

MLP에서는 `GELU(gate)⊙up`의 활동도와 residual 증가량을 확인할 수 있다. suffix에서는 저장된 adaRMS gate를 적용한 실제 `resid_mid-resid_in`, `resid_out-resid_mid`를 우선 사용한다. 높은 norm의 layer가 원인이라는 결론은 안 되므로 layer/MLP patching으로 확인한다. attention rollout은 저비용 비교군이지만 residual·MLP·gate를 생략하는 근사다. [Abnar & Zuidema, Quantifying Attention Flow](https://arxiv.org/abs/2005.00928).

**10. 관찰을 실제 연구 결과로 바꾸는 실험**

A. **query 역할과 실행 grounding**: `last`, `first`, 전체 평균, 실제 실행 K개 평균을 비교한다. K는 controller trace에서 가져온다. 이후 action sensitivity 기반 가중치를 시험한다. 같은 head가 다른 query에서 의미를 유지하는지도 검사한다. 성공 여부뿐 아니라 독립 object/neighbor mask, 3D extent, target switch를 측정한다.

B. **융합과 extent**: 고정 identity에서 raw action / noun / product seed / product+feature graph / geometry baseline / oracle extent를 비교한다. target recall과 함께 neighbor false-exemption rate를 primary geometry 지표로 둔다. 더 날카로운 map이 target surface를 놓치지 않는지 확인한다.

C. **state 혼동 분리**: state-only / strong rule / KV probe / 입력 modality intervention을 같은 lifecycle에서 비교한다. calibration, transition delay, stale exemption, task outcome을 기록한다.

D. **인과 검증**: 같은 noise로 orange→다른 물체, basket→다른 목적지, camera occlusion의 paired inference를 만든다. 입력 변화를 넣은 뒤 clean/corrupt 사이의 image feature, KV, selected head/MLP를 선택적으로 교체하고 이후 계산을 다시 실행한다. action 변화와 원래 행동 복원 정도를 측정한다. equal-count random token/head, low-score token, norm-matched replacement를 대조군으로 둔다. 영벡터 ablation은 분포 밖 artifact일 수 있다. value 교체와 key 교체를 나누어 content와 routing 효과를 구분한다. activation intervention의 기존 방법론을 참고하되 VLA 결과는 새로 검증해야 한다. [Meng et al., Causal Tracing](https://rome.baulab.info/).

E. **denoise 축소**: 위 grounding·lifecycle 설계를 고정하고 solver만 바꾼다. attention 추출 지연까지 포함한 end-to-end runtime을 측정한다. 최종 action 수치만 비슷한지와 실제 폐루프 성능이 유지되는지는 별개다.

이 실험들은 train/dev/test를 episode 단위로 분리하고 head·layer·threshold를 dev에서 고정한 뒤 test에서 평가한다. head×layer×step×query를 대량 탐색하면 우연한 좋은 map이 생기므로 선택 비용을 숨기지 않는다. CI는 frame 수를 표본 수로 세지 않고 episode 수준 paired bootstrap 등을 사용한다. 실제 성능을 측정하기 전 novelty·안전 보장·개선 완료를 주장하지 않는다.

**11. 수집 전략과 재현**

모든 frame을 약 4 GB씩 저장하는 것보다 전환·실패 frame에는 full dump, 긴 episode에는 compact summary를 저장하는 편이 유용하다. 우선 보존할 것은 selected layer의 image feature/KV, query window별 map, phase/grasp state, 모든 step의 x_t/v_t, active16 action, depth/calibration, 실제 실행 action index와 독립 contact label이다. entropy·norm·camera mass 등은 온라인 요약이 가능하다. full logits·MLP activation은 원인 조사 구간에 한정한다.

고정해야 할 메타데이터: checkpoint/code hash, prompt 원문·token span·mask, noise seed/초기 noise, dtype, 카메라 순서·전처리 좌표 변환, norm stats, joint/action 정의, control timestep, head/query 선택, original/replay 경로 구분. 저장된 V는 후속 layer에서 context가 섞인 feature이므로 raw pixel attribution과 구분한다.

이번 계산은 CPU NumPy로 수행했고 모델 재추론·학습·robot 실행은 수행하지 않았다. 별도의 광범위한 선행연구 novelty 조사는 아니다. 구현의 수치 검증은 JSON finite 검사, Gram 합 항등식, 원본 derived map 재사용, Euler 저장 오차 확인 및 그림 시각 검토로 한정한다. Euler 재구성 최대 오차는 active16에서 약 0.001172로 저장/계산 정밀도의 한계가 남는다.

```bash
OPENBLAS_NUM_THREADS=1 .venv-ag3s/bin/python \
  doc/archive/reviews/vla_dump_20261015/analyze.py \
  /mnt/dev/AG3S_VLA-storage/results/development/20261015T150325+0900_ddc7a00060c4_d857a08dcb63/vla_dump \
  doc/reviews/vla_dump_20261015
```

가장 먼저 검증할 질문은 **“어떤 query와 내부 feature가, 실제로 실행하는 동작에 필요한 타깃 표면을 가리키는가?”**이다. 그 근거를 명사 identity, 다중 시점 geometry, 물리 grasp feedback과 결합하면 현재 AG3S의 핵심 문제에 직접 연결할 수 있다.
