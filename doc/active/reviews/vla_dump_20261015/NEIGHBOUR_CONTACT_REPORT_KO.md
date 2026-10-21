# 이웃 과일 접촉 문제 — 발견, 제안, 구현, 결과

작성: 2026-10-21 KST. 기간: 2026-10-20 – 10-21.

- 범위: VLA 단독으로 성공하던 장면이 AG3S를 붙이자 실패한 원인 중 **target 바로 옆의 이웃 과일(바나나)** 문제
- 결론
  - 이웃 과일과의 가벼운 손가락 접촉을 허용하자 10 장면 중 9 장면이 성공했다. VLA 단독은 10, 기존 target 분리 수정은 7이다
  - 영상으로 보면 손목 회전은 의미 있게 일어나지 않았다. 성공은 **접촉 허용** 덕분이다
  - 잡을 지점을 지키면서 피하는 궤적 수정은 **CCR(TrajOpt) 문제로 남기고** 다음 단계로 넘어간다(사용자 판정, 2026-10-21)
- 수치 출처: `AG3S/ag3s/docs/handoff/IFG-APPLE-*.verify.json`, 그림 `AG3S/ag3s/docs/figures/ifg-split/`. 세부 경과는 `WORKLOG_S0S3_KO.md`의 2026-10-20 – 10-21 항목

## 용어

| 용어 | 뜻 |
|---|---|
| ATS | Attention-based Target Segmentation. VLA attention으로 잡을 물체(target)를 고르고 분리한다 |
| HSR | Hidden-state Sub-task Recognition과 grasp latch. 지금 무엇을 하는 중인지(접근 · 닫기 · 쥠 · 놓기)를 판정한다 |
| LTE | Link-specific Target Exemption. target만 cuRobo TSDF/ESDF에서 빼고, 접촉이 허가된 link(손가락)만 그 거리장을 보게 한다 |
| CCR | Constrained Chunk Refinement. VLA의 50-step 관절 chunk를 SQP TrajOpt로 고쳐 충돌 제약을 만족시킨다 |
| chunk · 요청 | VLA가 한 번에 내는 50-step 관절 명령. 앞 8 step을 실행하고 다시 요청한다(15 Hz) |
| HOLD | 고친 chunk에도 충돌 위반이 남으면 실행하지 않고 그 자리를 유지하는 것 |
| 이웃 접촉 row | MuJoCo에서 로봇이 target이 아닌 과일과 닿은 control step 수(실제 접촉 지표) |
| latch 상태 | searching(대상 없음) → latched(대상 고정, 접근) → closing(닫는 중) → held(쥠) → placed(놓음) |

## 1. 어떻게 발견했나

### 1.1 장면 풀과 규칙 (사용자 판정, 2026-10-20)

- 모듈 시험 장면은 **VLA 단독으로 성공하는 장면**으로 한다. 왼손 · 오른손 사과 장면 10개이고, seed를 고정했다(56, 211, 1837, 320, 753, 954, 309, 250, 1668, 126)
- VLA 단독으로 성공하던 장면이 AG3S 때문에 실패하면, 원인과 **어떤 규칙이 작동해서** 실패했는지를 찾는다

### 1.2 발견 경위

1. **성공률 하락**: 기존 방식 + target 분리 수정(ATS)을 붙이자 10 중 7만 성공했다. 211 · 954 · 320이 깨졌다. 211 · 954는 사과 바로 옆에 바나나가 있는 장면이다
2. **영상 확인(사용자)**: 954 영상에서 로봇이 사과보다 앞쪽에서 그리퍼를 닫았다. 사용자 지적: target은 정지해 있으므로 잡을 지점이 바뀌면 안 된다. VLA 단독은 바나나에 닿으면서 잡는데, AG3S는 접촉을 피하느라 손을 옮겼다. 손을 옮긴 주체는 TrajOpt다
3. **작동한 규칙 추적** (`ifg_regression_trace.py`, 서버 chunk 기록)
   - 954: 손가락 link가 이웃 과일 voxel과 장애물 여유를 지켜야 하는 충돌 제약이 작동했다
     - 접근 12–16번째 요청에서 VLA 계획의 여유는 1–3 mm였다
     - TrajOpt는 손을 6–19 mm 평행 이동했고, 회전은 1.7° 이하였다(`ifg-954-pose-hesitation`)
   - 211: 같은 바나나에 대해 닫는 chunk의 위반(−8.8 mm)을 풀지 못했다. HOLD가 난 뒤 쥠이 철회됐다
4. **VLA 단독도 바나나에 닿는다**: MuJoCo 접촉을 세면 VLA 단독의 이웃 접촉 row는 954에서 17, 211에서 8이다. 사람이 보기에 정상인 성공 동작이 이미 이웃 과일을 건드린다
5. **잡을 지점이 밀린다** (`ifg_goal_drift.py`, 처음 닫는 순간 손끝 위치를 사과 초기 위치 기준으로 잼)

| 954 | x (앞, mm) | y (옆, mm) | z (위, mm) |
|---|---|---|---|
| VLA 단독 | −3.8 | 14.7 | 12.3 |
| target 분리 수정 | 12.9 | 38.1 | 22.9 |

   TrajOpt가 손을 민 뒤 VLA가 밀린 손에서 다시 계획한다. 그래서 VLA가 잡으려는 지점도 같이 밀려 빈손으로 닫았다

### 1.3 원인 정리

- 이웃 과일을 **일반 장애물과 같은 여유(10 mm, `--obstacle-margin 0.010`)**로 피하게 하는 규칙이 원인이다. 사과에 바짝 붙은 바나나에서는 이 여유를 지킬 수 있는 잡기 자세가 거의 없다
- TrajOpt의 추종 비용은 관절 공간 거리만 보고, 손이 어디서 닫아야 하는지는 모른다. 그래서 가장 싼 회피가 **손의 평행 이동**이 된다
- 정책(VLA)은 고친 결과를 다음 요청의 관측으로만 본다. 밀린 손을 기준으로 다시 계획하므로 잡을 지점도 따라 밀린다

## 2. 제안한 방법

| # | 방법 | 겨냥한 원인 | 출처 |
|---|---|---|---|
| 0 | look-ahead 끔 | 허들 회피용으로 미리 피하는 기능이 잡기 근처에서 손을 미는가 | 사용자 질문 |
| 1 | 꺾임 유지 항을 쥔 뒤에만 사용 (IFG-TOB) | 접근 중 회피 꺾임이 다음 chunk로 이어짐 | lead |
| 2 | 잡을 지점 고정 (IFG-TOA) + 안전 우선 대체 풀이 | 잡을 지점이 바뀌면 안 된다 | 사용자 |
| 3 | 교착 · 빈손 감지 후 후퇴 · 재시도 (IFG-REC) | 실패한 뒤 회복 | 사용자 · 문헌 조사 |
| 4 | 손목 관절 가중치 (IFG-TOW) | 위치 대신 방향으로 피하기 | 사용자: "관절 1–3은 위치, 4–6은 방향" |
| 5 | B: 이웃 규칙은 유지하고 손목 회전만으로 피하기 | 4의 극한 | 사용자: "B 해보고 안 되면 A" |
| 6 | A: 손가락의 가벼운 이웃 접촉 허용 (IFG-NBC) | 규칙 자체가 VLA 단독의 정상 동작보다 엄격함 | 사용자 판정 |

모든 flag는 기본 꺼짐이다. 끈 상태에서는 기존과 결과가 같다.

## 3. 구현

### 3.0 기준: CCR의 TrajOpt 문제

VLA chunk를 $R=[r_1,\dots,r_H]$, 고친 chunk를 $Q=[q_1,\dots,q_H]$($q_k\in\mathbb{R}^{14}$, 양팔 관절)라 하자.

$$
\min_{Q,\,s\ge 0}\;
\underbrace{w_{\text{track}}\sum_k \gamma_k \lVert q_k-r_k\rVert^2}_{\text{VLA 추종}}
+ J_{\text{smooth}} + J_{\text{cont}}
+ \underbrace{w_{\text{rate}}\sum_k \lVert D_k-D_{k-1}\rVert^2}_{\text{꺾임 유지}}
+ w_{\text{slack}}\sum s
$$

$$
\text{s.t.}\quad d\big(c_j(q_k)\big)-\rho_j-m_{k,j}\;\ge\;-s_{k,j},\qquad q_k\in[q_{\min},q_{\max}]
$$

- $D_k=q_k-r_k$는 TrajOpt가 VLA 계획에서 꺾은 양이다. $\gamma_k$는 뒤 step일수록 줄어드는 가중치다
- $c_j(q)$는 로봇 충돌 구 $j$의 중심, $\rho_j$는 그 반지름, $d(\cdot)$는 cuRobo ESDF 거리다. 손가락은 LTE로 target을 뺀 거리장을 쓴다
- $m_{k,j}$는 그 행의 여유다. 최근접 표면이 target도 지지면도 아닌 장애물이면 10 mm다
- slack $s$는 1차 비용($w_{\text{slack}}$ per m)이다. 제약은 "가능한 한 지키되, 못 지키면 위반량을 보고"하는 형태이고, 위반이 남으면 HOLD한다
- SQP: 매 반복 FK와 거리를 선형화해 OSQP로 QP를 푼다

### 3.1 꺾임 유지 항을 쥔 뒤에만 사용 (IFG-TOB, `--deflection-rate-after-grasp`)

$$
w_{\text{rate}}\leftarrow 0\quad\text{if latch}\in\{\text{searching},\text{latched},\text{closing}\}
$$

- 뜻: 접근 중 한 번 꺾은 회피를 다음 chunk의 첫 행까지 끌고 가지 않는다
- 위치: `safe_policy._deflection_rate_gate`, `refiner.py`(그 chunk에서만 가중치를 0으로 바꿨다가 복원)

### 3.2 잡을 지점 고정 (IFG-TOA, `--grasp-anchor W`)

손바닥 원점 $p_k(Q)$(FK)에 대해 두 항을 더한다.

$$
J_{\text{anchor}} = w_{\text{tcp}}\sum_k\lVert p_k(Q)-p_k(R)\rVert^2
+ w_{\text{goal}}\sum_{k=c}^{c+G-1}\lVert p_k(Q)-g\rVert^2
$$

- 첫 항: 손바닥을 VLA 경로에 붙인다. 평행 이동은 비싸고, 회전이나 null-space 움직임은 싸다
- 둘째 항: VLA가 처음 닫기를 계획한 행 $c$의 손바닥 위치를 목표 $g$로 고정하고, 닫는 $G=3$행 동안 그 자리에 둔다
- QP에는 손바닥 Jacobian $J_k$로 선형화해 넣는다. $c=p_k(X)-J_kX_k-\text{target}$이면 $P \mathrel{+}= 2wJ_k^\top J_k$, $q \mathrel{+}= 2wJ_k^\top c$다. merit(해 채택 판단)에는 정확한 값을 쓴다
- 안전 우선 대체 풀이(TOA3)
  - 고정 항을 넣은 해에 침범이 남으면 고정 항 없이 다시 푼다
  - 두 해 중 침범이 작은 쪽을 쓴다
- 켜지는 범위: latched · closing이고 손바닥이 목표 150 mm 안일 때. $W=2000\ \text{m}^{-2}$
- 위치: `AG3S/trajopt/grasp_anchor.py`, `sqp.py`, `safe_policy._grasp_anchor_gate`

### 3.3 교착 · 빈손 후퇴 · 재시도 (IFG-REC, `--recovery-retreat`)

- 발동 조건(쥐기 전에만)
  - 교착: chunk 전체가 충돌 때문에 HOLD된 요청이 3번 연속
  - 빈손: 닫는 중 손가락이 멈췄는데(settled) 아무것에도 막히지 않은(blocked 아님) 요청이 2번 연속
- 동작
  - 그리퍼를 연다
  - 손바닥이 target에서 120 mm 이상 떨어져 있던 가장 최근 관절 자세로, 행당 0.03 rad 이하로 돌아간다. 이 후퇴 chunk도 TrajOpt를 거친다
  - 도착하면 VLA에 다시 맡긴다. episode당 최대 2회
- 근거: Sheng et al. 2610.06921의 training-free retreat (`FAILURE_RECOVERY_SURVEY_KO.md`)
- 위치: `safe_policy._recovery_observe`, `_recovery_chunk`

### 3.4 손목 관절 가중치 (IFG-TOW, `--wrist-track-scale S`)

추종 항에 관절별 배율 $\sigma_j$를 넣는다.

$$
J_{\text{track}} = w_{\text{track}}\sum_k\gamma_k\sum_j \sigma_j\,(q_{k,j}-r_{k,j})^2,
\qquad
\sigma_j=\begin{cases}S & j\in\{\text{arm}_4,\text{arm}_5,\text{arm}_6\}\ \text{(쥐는 팔)}\\ 1 & \text{그 외}\end{cases}
$$

- 뜻: 관절 $j$를 $\delta$만큼 바꾸는 값이 $\sigma_j w\delta^2$가 된다. $S=0.1$이면 손목을 바꾸는 것이 10배 싸다
- 그래서 같은 충돌을 풀 때 위치를 크게 바꾸는 앞쪽 관절보다 방향을 바꾸는 손목을 먼저 쓴다. 앞쪽 관절은 잠그지 않는다
- 근거(954 관절별 분석, 관절 1°당 손끝 이동)

| 관절 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| mm/deg | 12.6 | 12.6 | 3.5–7 | 7.6 | 3.5 | 4.4 | 0.0 |

  target 분리 수정은 954에서 주로 0번과 3번 관절로 피했다
- 켜지는 범위: latched · closing이고 손바닥이 target 중심 150 mm 안일 때만이다. 허들 · 먼 접근 · 운반은 균일 가중치를 쓴다(사용자 우려 반영). QP · merit · look-ahead 경로에 똑같이 들어간다
- 위치: `problem.build_problem` · `objective`(`joint_track_scale`), `sqp.py`, `safe_policy._wrist_track_gate`

### 3.5 B: 손목 회전만으로 피할 수 있는가 (`ifg_roll_feasibility.py`, 실행 없이 기하 검사)

VLA 단독이 닫기 시작한 자세 $q^\ast$에서 마지막 손목 관절만 $\phi$만큼 돌린다. 이 관절은 손끝 위치를 바꾸지 않는다.

$$
h(\phi)=\min_{j\in\mathcal{F}}\Big[d\big(c_j(q^\ast\oplus\phi\,e_{\text{arm}_6})\big)-\rho_j\Big]
\;\overset{?}{\ge}\; m=10\ \text{mm},
\qquad \phi\in[-90^\circ,90^\circ]
$$

- $\mathcal{F}$는 쥐는 손의 손가락 · 그리퍼 구다. $d$는 AG3S run이 기록한 target 제외 거리장이다
- $h(\phi)\ge m$인 $\phi$가 있으면 "잡을 지점은 그대로 두고 방향만 바꿔 규칙을 지키는 자세"가 존재한다

### 3.6 A: 손가락의 가벼운 이웃 접촉 허용 (IFG-NBC, `--neighbour-contact-mm δ`)

조건을 만족하는 행만 여유를 음수로 바꾼다.

$$
m_{k,j}=\begin{cases}
-\delta & \text{class}_{k,j}=\text{obstacle}\ \wedge\ j\in\mathcal{F}_{\text{hand}}\ \wedge\ \lVert c_j(q_k)-c_{\text{target}}\rVert\le R_n\\
m_{k,j}\ (\text{그대로}) & \text{그 외}
\end{cases}
$$

$$
\Rightarrow\quad d\big(c_j(q_k)\big)-\rho_j\;\ge\;-\delta
$$

- 뜻: 쥐는 손의 손가락 구가 target 근처의 이웃 물체와 **최대 $\delta$까지 겹치는 것**을 허용한다. 가볍게 밀면서 잡는 VLA 단독의 동작을 받아들인다
- 행 분류(class)는 기존 규칙을 그대로 쓴다. 최근접 표면이 target이면 target 행, 지지면 평면이면 support 행, 쥔 물체면 held 행이고, 나머지가 obstacle 행이다. 탁자 · target · 손바닥 · 팔 · 쥔 물체 행과 운반 구간은 바뀌지 않는다
- 켜지는 범위: latched · closing이고 손바닥이 target 중심 150 mm 안일 때
- 값: $\delta=20$ mm, $R_n=120$ mm. 20 mm는 VLA 단독 잡기 자세에서 잰 손가락 최대 침범 14.1 mm(211, 닫기 15 step 뒤)보다 크게 잡은 값이다
- 위치: `SceneSnapshot.neighbour_contact`, `linearize.CollisionLinearizer`(margin 규칙), `safe_policy._neighbour_contact_gate`. 판정(HOLD)도 같은 scene으로 하므로 TrajOpt와 어긋나지 않는다

### 3.7 시험

- 새 단위 시험
  - `test_ifg_tob_deflection_rate_after_grasp.py` 4개
  - `test_ifg_toa_grasp_anchor.py` 8개
  - `test_ifg_rec_recovery.py` 6개
  - `test_ifg_tow_wrist_track.py` 4개
  - `test_ifg_nbc_neighbour_contact.py` 3개
- `tests/trajopt` 91개 파일 중 89개 통과, 2개 건너뜀
  - `test_t43t_replay.py`의 bit-identity 1개 실패는 수정 전 HEAD에서도 똑같이 실패한다. 옛 commit(682976e)의 기준 파일과 비교하는 기존 실패다

## 4. 실험 결과

### 4.1 설정

- 같은 10 장면과 seed, 장면당 1회. 변수는 한 번에 flag 하나다(TOW와 NBC는 누적)
- 성공은 "잠긴 접촉 없는 성공"이다(쥐고 들어 바구니 안에 놓음)
- 표본이 10이라 성공률 차이를 통계적으로 주장하지 않는다(예: target 분리 수정 대 A, McNemar p = 0.5). 실패 원인을 진단하기 위한 실험이다

### 4.2 성공 수와 실패 장면

| 구성 | 성공 / 10 | 실패 장면 | 비고 |
|---|---|---|---|
| VLA 단독 | 10 | — | 이웃 접촉 row: 954 17, 211 8 |
| 기존 방식 + target 분리 수정 | 7 | 211 · 320 · 954 | 비교 기준 |
| + look-ahead 끔 | 7 | 211 · 954 · 1837 | 211 · 954 그대로 |
| + 꺾임 유지 항 쥔 뒤에만 (TOB) | 7 | 211 · 954 · 1668 | 954 그대로 |
| + 잡을 지점 고정 (TOA2) | 7 | 211 · 954 · 1668 | 211 HOLD 63 chunk |
| + 잡을 지점 고정 + 안전 우선 대체 (TOA3) | 7 | 211 · 954 · 1668 | 211 HOLD 49 chunk |
| + 후퇴 · 재시도 (REC) | 7 | 211 · 320 · 954 | 성공 장면 오경보 0 |
| + 손목 관절 가중치 (TOW) | 8 | 211 · 954 | 320 회복 |
| + 손목 가중치 + 이웃 접촉 20 mm (NBC) | **9** | 211 | 954 회복 |

- 짝 비교(같은 장면)
  - NBC는 손목 가중치만 쓴 구성 대비 실패→성공 1(954), 성공→실패 0이다
  - target 분리 수정 대비 실패→성공 2(320 · 954), 성공→실패 0이다
- 그림: `ifg-to-variants.png`, `ifg-rec-recovery.png`, `ifg-nbc-result.png`

### 4.3 방법별 관찰

- **look-ahead 끔**: 954 · 211이 그대로 실패했다. 허들 회피 기능이 원인이 아니다
- **꺾임 유지 항**: 954의 잡을 지점 밀림이 그대로였다. 꺾임이 이어지는 것은 주원인이 아니다
- **잡을 지점 고정**
  - 954는 회전으로 피했지만, 약 2초 일찍 · 13 mm 높게 닫아 실패했다
  - 211은 고정 항이 바나나를 피하는 데 필요한 평행 이동까지 비싸게 만들어 HOLD에 갇혔다
  - 안전 우선 대체를 넣어도 211에서는 고정 항 없이 푼 해도 19–27 mm 침범이 남아(대체 풀이 51번 실행) 풀리지 않았다
- **후퇴 · 재시도**
  - 실패 3 장면 모두 빈손 조건으로 발동했다
  - 그러나 발동 때 손바닥이 이미 target에서 171–373 mm 떨어져 있었고, 후퇴 목표가 사실상 지금 자세였다
  - 재시도에서는 사과보다 50–170 mm 앞에서 닫았다. 설계 결함 두 가지(늦은 탐지, 후퇴 목표 선택)가 확인됐다
- **손목 가중치**
  - 320: 앞쪽 관절 수정(0번 9.7°, 1번 8.8°)이 1.2° 이하로 줄고 성공했다
  - 954: 손목 수정 합계가 29.3°에서 65.7°로 늘었지만, 앞쪽 수정은 29.9°에서 33.7°로 줄지 않았다. 닫는 위치가 (0.5, 37.2, 21.9) mm로 옆 23 mm · 위 10 mm 밀림이 남았다
- **B (손목 회전만)**

| 장면 | 회전 없음 | 가장 좋은 각도 | 닫기 15 step 뒤 | 판정 |
|---|---|---|---|---|
| 954 | −6.6 mm (바나나 침범) | −1.8 mm (−50°, 탁자 · 사과 아랫면 쪽) | — | 불가 |
| 211 | −4.1 mm | +0.7 mm (+10°) | 모든 각도 ≤ −6.5 mm | 불가 |

  요구 여유 10 mm를 만족하는 각도가 없다. 잡을 지점을 유지하면서 이웃 규칙을 지키는 자세가 존재하지 않으므로 closed-loop 실험 없이 A로 넘어갔다(그림 `ifg-roll-feasibility-ep954` · `-ep211`)
- **A (이웃 접촉 20 mm)**
  - 954: 닫는 위치가 (−7.3, 16.2, 11.4) mm로 VLA 단독(−3.8, 14.7, 12.3)과 거의 같아 성공했다. 실제 이웃 접촉은 8 row로 VLA 단독(17)보다 적다
  - 211은 실패했다. 실제 이웃 접촉은 14 row(VLA 단독 8)다
    - 손이 VLA 단독보다 약 2초 일찍 사과에 닿았다(t=143)
    - 사과를 약 70 mm 지나친 뒤 닫아 바나나를 쥐었다(개도 0.52에서 멈춤)
    - 쥠이 인정된 뒤 바로 철회됐다
    - 경로는 이웃 접촉 규칙이 켜지기 전(손바닥 163 mm, 12–13번째 요청의 HOLD)부터 갈라진다
  - 나머지 8 장면은 모든 구성에서 이웃 과일 접촉 0이었다
  - 영상 판정(사용자): 손목 회전은 의미 있게 일어나지 않았다. 954의 성공은 접촉 허용 덕분이다

## 5. 결론과 남은 문제

- **이웃 과일 규칙이 VLA 단독의 정상 잡기보다 엄격했다.** 사과에 붙은 바나나 장면에서는 그 규칙을 지키는 잡기 자세가 없다(B 불가). TrajOpt는 손을 옮기고, VLA는 밀린 잡을 지점을 따라간다
- **제한된 접촉 허용(A)이 실제로 효과가 있었다.** 954를 회복했고, 실제 접촉은 VLA 단독 이하였다. 다만 211이 남는다
- **CCR(TrajOpt) 문제로 남기는 것**(사용자 판정)
  - 잡을 지점을 지키며 방향으로 피하는 궤적 수정이 아직 안 된다. 손목 가중치만으로는 앞쪽 관절 수정이 줄지 않았다
  - 211에서 이웃 규칙이 켜지기 전부터 경로가 갈라져 일찍 도착하는 원인은 미진단이다
  - 사후(post-hoc) 보정의 한계: 정책은 chunk 안의 보정을 보지 못하고 밀린 손에서 다시 계획한다. 생성 중(denoising) 보정은 `AG3S/docs/AG3S_PHASE_DENOISING_PLAN.md` §4에 계획이 있다
- **논문 관련**(`POLICY_ACTION_MODIFICATION_SURVEY_KO.md`)
  - attention으로 target을 고르고 obstacle에서 빼는 앞부분은 Park et al. 2026(2606.09749)과 겹친다
  - post-hoc 보정의 약점은 OmniGuide(2603.10052)가 지적했다
  - 제한 접촉은 IMPACT(ICRA 2026) · Jain et al. 2013과 비교해 범위를 좁혀 주장한다
- **flag 상태**: 위 flag는 모두 기본 꺼짐이고 커밋 전이다. 다음 단계에서 기본 구성으로 무엇을 쓸지는 정하지 않았다

## 부록: 실행 기록

| 구성 | kit (`…/20261019T132719+0900_f15e587290f6_11408fd6d27f/verify/`) | snapshot tree | verify.json |
|---|---|---|---|
| look-ahead 끔 | `IFG-APPLE-LOOKAHEADOFF` | — | `IFG-APPLE-LOOKAHEADOFF.verify.json` |
| TOB | `IFG-APPLE-TOB` | `3b9b3a68` | `IFG-APPLE-TOB.verify.json` |
| TOA2 · TOA3 | `IFG-APPLE-TOA2` · `-TOA3` | `cba727f0` · `7da7599a` | `IFG-APPLE-TOA2.verify.json` · `IFG-APPLE-TOA3.verify.json` |
| REC | `IFG-APPLE-REC` | `d63c7292` | `IFG-APPLE-REC.verify.json` |
| TOW · B · NBC | `IFG-APPLE-TOW` · `-NBC` | `45bdd832` · `222a2f58` | `IFG-APPLE-NBC.verify.json` |
