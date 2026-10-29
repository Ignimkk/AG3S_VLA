# target 찾기 · 쥐기 상태 판단 — 채택한 것 (세션 B: identity · phase source 분석)

작성: 2026-10-29 00:25 KST · 세션 B (identity · phase source 분석, 2026-10-14 ~ 15 수행) · 기준 HEAD `b3b69311`.

- 이 문서는 세션 B 가 수행한 offline 분석 가운데 **사용자가 채택으로 정한 3 건**만 적는다 (사용자 지시 2026-10-28).
- 이 폴더의 다른 문서 ([ADOPTED_KO.md](ADOPTED_KO.md) · [NOT_ADOPTED_KO.md](NOT_ADOPTED_KO.md) · [OPEN_ISSUES_KO.md](OPEN_ISSUES_KO.md)) 와 주제가 다르다. 그 문서들은 충돌 구 모델 · TrajOpt 에 관한 것이고, 이 문서는 target 찾기 (ATS identity) 와 쥐기 상태 판단 (HSR · grasp latch) 에 관한 것이다.
- 수치는 출처 기록 (`AG3S/ag3s/docs/handoff/*.verify.json`) 의 값을 옮긴 것이다. 모든 실험은 MuJoCo 시뮬레이션 기록 위의 offline 분석이고, 새 closed-loop 은 돌리지 않았다.
- 이 문서에 넣지 않은 것 (판정 대기 · 재검토): 아래 항목은 이 문서의 결정 범위 밖이다.
  - 사전 고정 절차가 고른 구성을 서빙 기본으로 바꾸는 일
  - 명사 token 으로 target 찾기
  - 지시 물체 확인 gate
  - GT 없는 self-label head 선택

## 요약

| # | 채택 | 층 | 무엇이 바뀌나 | 상태 |
|---|---|---|---|---|
| D1 | **결과 전에 고정한 attention 선택 절차** | ATS (target identity) | attention head 를 관행 (L8H2) 이 아니라, 결과를 보기 전에 고정한 규칙으로 고르고 실행 중 고정한다 | 채택 (절차). 이 절차가 고른 구성의 서빙 교체는 closed-loop 뒤 판정 |
| D2 | **쥔 물체를 attach 하는 판정의 reach 기준점 결함을 수정 대상으로 확정** | HSR · grasp latch (attached object) | reach 검사가 손 안 물체가 아니라 grounding 이 고른 다른 물체의 centroid 를 쓰는 경로를 고친다 | 채택 (수정 대상). 수정 방식은 세션 A 가 Panda 의 같은 계열 결함과 묶어 판정 |
| D3 | **종료 구간 (놓기 완료 · 복귀 완료) 기준의 보조 평가** | 평가 규칙 | 이미 고정된 실험은 기존 판정 + 종료 구간 결과를 나란히 보고한다. 새 실험은 놓기 완료까지를 주 지표로 쓴다 | 채택 (평가 규칙) |
| D4 | **HSR 을 방법에 사용 (6 단계)** | HSR → LTE (단계별 허용 · 제약 · 목적함수) | VLA feature 로 approach · grasp · lift · transport · placement · retreat 를 판단해 LTE 가 단계마다 허용을 바꾸는 근거로 쓴다 | 채택 (사용자 2026-10-29). 사용 방식 (물리 evidence 와 결합) 판정 대기 |

---

## D1. 결과 전에 고정한 attention 선택 절차

### 무엇을 채택했나

VLA 의 어떤 attention 을 target 찾기 신호로 쓸지 정하는 절차다. 단계는 다음과 같다.

1. **후보:** 512 개. action token query (`last` · `mean` · `first`) 와 prompt 의 target 명사 token query 에, 각각 layer 2–17 × head 0–7 을 조합한다.
2. **label 없는 사전 걸러내기:** sink · 너무 넓은 map · 너무 약한 map 을 제외한다.
3. **2D 사전 선별:** query 마다 2D 점수 상위 6 개를 남긴다.
4. **3D 순위:** calibration 장면의 GT `identity_correct` (놓기 완료까지) 로 순위를 매긴다.
5. **조합 수 k:** leave-one-scene-out CV 와 one-SE 규칙으로 정한다.
6. **고정:** 고른 구성을 실행 중 바꾸지 않는다.

- 모든 규칙 · 창 · 후보 수를 **결과를 보기 전에** 고정했다 (`T61.task.md`, commit `95b6586c`).
- 선택 결과는 **새 data 를 열기 전에** sha256 과 함께 commit 했다 (`T61.selection.json` `c3f3b60f…`, commit `b49f5f6c`).

### 채택한 이유

- 기존 서빙 head (action 마지막 token, layer 8 · head 2 = "L8H2") 는 관행으로 쓰던 것이고, 고른 근거가 남아 있지 않다.
- 이전 선택 시도는 같은 질문에 다른 답을 냈다.
  - GT 로 고르면 `last:L12H4` 가 1 위였지만, 1 위 확률이 0.429 로 안정성 기준 (0.5) 을 넘지 못했다 (AG3S_T47_LOG "T51").
  - 평가 구간을 놓기 완료까지로 자르면 1 위가 바뀌었다 (AG3S_T47_LOG "T60").
- 그래서 "어느 head 가 최고인가" 가 아니라 **"어떤 절차로 고르는가"** 를 고정해야 재현 가능한 방법이 된다.
- 이 절차는 고른 구성이 L8H2 보다 낫지 않아도 그대로 결과로 남는다. 우월성을 전제하지 않는다.

### 수행한 실험

**(1) calibration 선택** — 장면 5 개 (`t2` · `ep1807` · `ep0` · `ep500` · `ep1200`), 평가 창 = 놓기 완료까지

| 항목 | 값 |
|---|---|
| 선택 구성 | **action token 평균 (`mean`), denoise 0, layer 10 · head 7, k = 1** (`mean:L10H7`) |
| CV 평균 (k = 1 · 2 · 3) | 0.692 · 0.676 · 0.685 |
| SE (k = 1) | 0.052 → one-SE 로 k = 1 |
| 보고만: scene bootstrap 1 위 확률 | 0.731 |
| 보고만: GT 없이 self label 로 고른 결과 | 같은 `mean:L10H7` |

**(2) 새 data 평가** — 선택 · 개발에 한 번도 쓰지 않은 새 16 episode

- 조건: 현재 서빙으로 rollout 한 기록에서 attention 을 다시 추출했다. calibration 과 같은 replay 설정을 썼고, 기존 L8H2 를 기준선으로 같은 frame 에서 비교했다.
- 지표: episode 평균 GT `identity_correct`.
- CI: episode bootstrap B 10000 · seed 6102.

| 평가 구간 | 선택 구성 | L8H2 | 차이 | episode 95 % CI | 오름 / 내림 episode | purity 차이 |
|---|---|---|---|---|---|---|
| **놓기 완료까지 (주)** | 69.98 % | 62.94 % | **+7.04 %p** | [−1.27, +18.36] | 5 / 6 | +12.10 %p |
| 전체 | 70.08 % | 55.17 % | +14.92 %p | [+3.08, +27.92] | 7 / 6 | +12.25 %p |
| 복귀 완료까지 | 63.00 % | 58.77 % | +4.23 %p | [−5.51, +16.44] | 4 / 7 | +10.95 %p |

- 그림: [t61-heldout-paired.png](../../../../AG3S/ag3s/docs/figures/t61/t61-heldout-paired.png). episode 별 차이와 요약 표를 담았다.

### 한계

- **주 지표의 CI 가 0 을 포함한다.** episode 방향도 5 / 6 으로 엇갈린다.
  - 그래서 "L8H2 보다 낫다" 고 쓰지 않는다. "유의한 차이가 없다" 를 동등성으로 쓰지도 않는다.
  - 채택한 것은 **절차**다. 고른 구성을 서빙 기본으로 바꾸는 일은 closed-loop 결과를 보고 따로 판정한다.
  - 후보 경로 (기본 off) 와 parity 는 세션 A 에 요청했다 (`T61.task.md` 끝 절, commit `d14fb398`).
- calibration 장면이 5 개뿐이라 CV SE 가 크다. 그중 3 개 (`ep0` · `ep500` · `ep1200`) 는 VLA 학습 episode 다.
- 새 data 에 (apple, 왼팔) 조합이 없다. 과일 × 팔 조합으로 일반화하지 않는다.
- 보고 전용 항목 (scene bootstrap) 은 사전 고정한 B 10000 · seed 6101 이 아니라 B 1000 · seed 5103 으로 돌았다. 선택 규칙에는 쓰이지 않는다.

### 출처

- 사전 고정 · 선택 고정 · 서빙 요청: [`T61.task.md`](../../../../AG3S/ag3s/docs/handoff/T61.task.md)
- 선택 결과: [`T61.selection.json`](../../../../AG3S/ag3s/docs/handoff/T61.selection.json)
- 수치: [`T61.verify.json`](../../../../AG3S/ag3s/docs/handoff/T61.verify.json) `heldout.numbers`
- 코드: `AG3S/ag3s/experiments/studies/t61_select.py` · `t61_fix_selection.py` · `t61_heldout.py` · `t61_heldout_verify.py`
- run: `results/development/20261014T183529+0900_53f3911df197_dd2fad8756a6` (선택 · rollout), `…/20261014T222232+0900_903ea16764c1_eb761f984d83/verify/T61_heldout` (평가)

---

## D2. 쥔 물체를 attach 하는 판정의 reach 기준점 결함 — 수정 대상으로 확정

### 무엇을 채택했나

grasp latch 는 "물체를 쥐었다" 고 판단하면 그 물체를 충돌 모델에 attach 한다. 이 판단의 evidence 중 하나가 **reach 검사**다. grasp link 원점에서 AG3S 가 고른 조작 대상의 centroid 까지 거리가 120 mm 이하여야 통과한다.

이 검사가 **손 안의 물체가 아니라 grounding 이 고른 다른 물체**로 거리를 재는 경로가 있다. 이 경로를 수정 대상으로 확정했다.

### 채택한 이유

- attach 가 빠지면 손에 쥔 물체가 충돌 모델에서 빠진다. 그 물체가 다른 장애물 · 로봇 자신과 부딪쳐도 TrajOpt 가 막지 못한다.
- 즉 논문의 attached-object protection 주장에 직접 영향을 주는 **정확성 결함**이다. 성능 비교가 아니라 결함이라서 고치는 것으로 정했다.

### 수행한 실험

- data: 기존 쥐기 상태 판정 비교 실험 (AG3S_T47_LOG "T55") 의 기록만 다시 읽었다. 새 실행은 없다.
- 조건: 새 16 episode 의 이미지와 로봇 상태는 그대로 두고, **prompt 의 과일 명사만 다른 과일로 바꿨다.** 서빙 grounding 은 다시 돌렸다.
- 정답: MuJoCo 접촉으로 판정한 "손이 실제로 물체를 쥠".
- 방법: 쥐었는데 attach (HELD) 가 안 된 frame 마다 세 가지를 읽었다.
  - latch 상태와 실패한 evidence 항목
  - AG3S 가 고른 조작 대상이 어느 과일인지
  - reach 거리
- 출처의 공개 수치 (333 · 117) 를 assert 로 재현한 뒤 분석했다.

| 항목 | 값 |
|---|---|
| 실제로 쥔 frame 중 attach 가 안 된 frame | **117 / 333** (원래 prompt 조건에서는 0 / 111) |
| 그 117 frame 의 조작 대상 | **117 / 117 이 바뀐 명사의 과일** (테이블 위, 손 안 아님) |
| latch 상태 · 실패 evidence | 닫는 중 · reach 만 실패 97, 닫는 중 · 손가락 닫힘 + reach 11, 대상 확정 단계 9 |
| latch 가 잰 reach 거리 | 162.5 – 741.8 mm (중앙값 529.7), 기준 120 mm |
| 실제 손 안 과일까지 거리 (MuJoCo 참값, 진단용) | 50.7 – 72.7 mm (중앙값 55.9) |
| trace 단위 (실제 쥐기가 있는 35 trace) | attach 됨 25. 그 순간 조작 대상은 손 안 과일 18 · 명사 과일 5 · 다른 과일 2. attach 안 됨 10, 전부 명사 과일 |

- 그림:
  - [t55b-scene-instruction.png](../../../../AG3S/ag3s/docs/figures/t55b/t55b-scene-instruction.png): 같은 이미지에 명사만 바꿨을 때 한쪽은 reach 실패, 다른 쪽은 attach
  - [t55b-held-fn-reach.png](../../../../AG3S/ag3s/docs/figures/t55b/t55b-held-fn-reach.png): 거리 분포 · attach 순간 조작 대상

### 한계

- 명사만 바꾼 합성 조건이다. 정책이 실제로 다른 물체를 집는 rollout 은 아니다.
- 명사 과일로 attach 된 5 trace 에서 attach 된 형상이 실제 손 안 물체와 같은지는 재지 않았다.
- 고친 latch 로 117 frame 이 회복되는지는 수정안이 정해진 뒤 같은 기록으로 다시 채점해 확인한다.
- Panda 의 같은 계열 결함 (가려진 target 의 오래된 centroid 로 reach 실패) 과는 **같은 검사를 지나지만 centroid 가 틀어지는 원인이 다르다.**
  - Panda 쪽은 centroid 가 오래된 것이고, 이쪽은 다른 물체의 것이다.
  - 그래서 "centroid 를 손과 함께 움직이기" 만으로는 이 결함이 고쳐지지 않는다. 손 안 물체의 identity 를 고르는 경로까지 다뤄야 한다.

### 출처

- 수치: [`T55b.verify.json`](../../../../AG3S/ag3s/docs/handoff/T55b.verify.json)
- 기록: AG3S_T47_LOG "T55b" 절
- 코드: `AG3S/ag3s/experiments/studies/t55b_held_fn.py`
- 수정안 판정: 세션 A (Panda 진단 T46g 와 묶어서)

---

## D3. 종료 구간 기준의 보조 평가

### 무엇을 채택했나

- 평가 구간을 둘로 자른다.
  - **놓기 완료:** target 이 crate 안 · 손과 접촉 없음 · 멈춤이 8 행 이어진 시점까지
  - **복귀 완료:** 놓은 뒤 팔이 준비자세로 돌아온 시점까지
- 도달하지 못한 run 은 전체 구간을 쓴다.
- 적용 방식:
  - 이미 기준을 고정한 실험은 기존 판정 (전체 구간) 을 유지하고, 두 구간 결과를 나란히 보고한다.
  - 새로 설계하는 실험은 놓기 완료까지를 주 지표로, 복귀 완료까지를 필수 보고로 한다.
- 정의는 세션 A · 이전 재검증 세션이 고정한 것이다 (AG3S_T47_LOG "T60"). 세션 B 는 진행 중이던 두 실험에 적용해 결과를 확인했다.

### 채택한 이유

- 작업이 끝난 뒤의 frame (놓은 뒤 · 복귀 뒤) 이 평가와 head 선택을 바꿀 수 있다는 것이 이전에 확인됐다 (AG3S_T47_LOG "T60").
- 그래서 이미 고정한 판정이 그 frame 때문에 생긴 것인지 확인해야 했다. 아래 두 실험에서 **기존 판정은 두 구간에서도 바뀌지 않았다.** 이 규칙을 계속 쓰면 판정이 작업 구간에서 나왔다는 것을 보일 수 있다.

### 수행한 실험

**(1) 쥐기 상태 판정 비교 (명사 token probe 와 HSR 을 strong rule 과 결합) 의 두 구간 재채점**
- data: 새 16 episode 재시뮬
- 실행: 세션 B 가 독립 재실행했고, 출력 5 파일이 1 차 실행과 byte 단위로 같다.

| 결합 | 구간 | "잘못 통과" 변화 | "올바른 보유 거부" 변화 | 사전 허용폭 |
|---|---|---|---|---|
| 명사 token probe | 전체 (주) | −4.46 %p [−6.44, −2.58] | +1.53 %p [0, +4.76] | 충족 |
| 명사 token probe | 놓기 완료까지 | −7.10 %p [−12.38, −3.48] | +1.53 %p | 충족 |
| 명사 token probe | 복귀 완료까지 | −6.34 %p [−9.87, −3.32] | +1.53 %p | 충족 |
| HSR | 전체 (주) | −1.19 %p [−2.22, −0.43] | +0.76 %p [0, +2.38] | 충족 |
| HSR | 놓기 · 복귀 완료까지 | −1.84 · −1.70 %p | +0.76 %p | 충족 |

- 놓기 · 복귀 구간에서 변화가 커진 것은 대부분 분모 (음성 frame 6616 → 4082 · 4656) 가 줄어든 효과다.

**(2) 명사 grounding 비교 (새 16 episode) 의 두 구간 재집계**
- 서빙 기록 (L8H2): success · `success_ct` 는 세 구간 모두 12/16 이다. HOLD chunk 는 150 → 놓기 완료까지 7 · 복귀 완료까지 16 이다.
- 명사 − L8H2 identity:
  - 현재 frame: +19.90 · +15.63 · +17.62 %p (전체 · 놓기 · 복귀)
  - 서빙처럼 상태를 유지하는 재생: 세 구간 모두 +0.84 %p
- 진입 판정은 세 구간에서 같다.
- 그림: [t54c-windows-ph.png](../../../../AG3S/ag3s/docs/figures/t54c/t54c-windows-ph.png)

### 한계

- 보조 분석이다. 쥐기 상태 판정 비교는 놓은 뒤 동작도 평가하므로, 놓기 완료 구간 결과가 전체 기능 평가를 대신하지 않는다.
- 두 실험 모두 전체 구간이 공개값을 재현하는 것을 assert 로 확인한 뒤 두 구간을 냈다. 재시뮬레이션 · 재추출은 하지 않았다.

### 출처

- 수치: [`T55_windows.verify.json`](../../../../AG3S/ag3s/docs/handoff/T55_windows.verify.json) · [`T54c_windows.verify.json`](../../../../AG3S/ag3s/docs/handoff/T54c_windows.verify.json)
- 기록: AG3S_T47_LOG "T55" §8 · "T54c" §10
- 코드: `t55_windows.py` · `t54c_windows.py`

---

## D4. HSR (VLA feature 로 작업 단계 판단) 을 방법에 사용 — 채택 (사용자 2026-10-29)

### 무엇을 채택했나

- VLA prefix feature (layer 4 KV, 서빙 추가 forward 없음) 위의 probe 로 작업 단계를 판단하는 HSR 을 AG3S 방법에 쓴다.
- 단계는 **6 개** 다: approach · grasp · lift · transport · placement · retreat.
- 목적: LTE 가 단계에 따라 접촉 허용 · 제약 · 목적함수를 바꾸는 근거로 쓴다.
- **사용 방식 (판정 대기):** HSR 단계를 단독으로 쓸지, grasp latch 의 물리 evidence 와 결합해 쓸지. 아래 실험 근거상 결합이 필요하다.

### 수행한 실험 (T64, 사전 고정 `T64.task.md`, 수치 `T64.verify.json`)

| 실험 | 결과 |
|---|---|
| 학습 데이터 (expert val 50 episode) | 6 단계 macro-F1 0.982 → 6 단계 유지 (기준 0.80) |
| closed-loop 기록 (새 16 episode, 물체 위치 정답) | HSR 0.730 대 같은 6 단계 rule 0.710, 차이 +0.020 [−0.043, +0.087]. grasp · approach · retreat 는 HSR, lift · transport 는 rule 이 낫다. **정답 정의를 사용자가 영상으로 확인하기 전이라 판정하지 않았다** |
| strong rule 이 놓친 attach 보완 (HSR 운반 단계 ∧ 손가락이 물체에 막힘) | 놓친 117 frame 중 107 회복 (91.5 %), 빈손 · 미끄러짐에서 잘못 세운 보유 +0, 원래 조건 오판 +0.18 %p → 사전 기준 3 개 충족 |

- outputs: `outputs/2026-10-29_HSR_6단계_단계구분_학습데이터_closedloop기록/` (영상 · 단계 띠 · 그래프 · 표)

### 한계

- **실패 episode 에서 HSR 은 물체가 없어도 팔 움직임을 따라 운반 · 놓기 단계를 낸다** (예: ep1849, 떨어뜨린 뒤에도 transport → placement). 그래서 "쥔 물체" 용 허용은 HSR 단독으로 주면 안 된다. 손가락 막힘 · latch 같은 물리 evidence 와 결합해야 한다.
- attach 보완은 지시만 바꾼 합성 조건에서 측정했다. closed-loop 확인 전이다.
- 학습 label 은 expert 수집기의 phase 이고, 실행은 VLA 정책이다. lift · transport 정확도는 실제 실행에서 크게 떨어졌다 (0.94 · 0.95 → 0.67 · 0.29).
