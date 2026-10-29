# AG3S-CORE-EVAL-HSR6 — 6 단계 HSR 을 서빙에 넣기 (기본 꺼짐 4 flag)

> writer: lead (세션 A) · 구현: ag3s-implementer (AG3S-CORE-EVAL-G1 합치기가 끝난 뒤 — 같은 파일을 고친다) · 작성: 2026-10-29 02:55 KST.
> 출처: 세션 B 의 [T64](T64.task.md) "closed-loop 후보" 절 (commit `461231a4`). 상위: [AG3S-CORE-EVALUATION](AG3S-CORE-EVALUATION.task.md).

## 0. 사용자 판정

- B 세션 (2026-10-29): "HSR 을 사용한다" · 사용 방식 3 가지 승인 → T64 끝 절.
- A 세션 (2026-10-29 02:40 KST):
  - 아래 4 사용처를 **모두** 첫 closed-loop 에 넣어 시험한다.
  - 보고는 "HSR 로 인해 기존과 결과가 어떻게 바뀌었는가".
  - 결합 시험에 나란히 넣는다: A8 + G1 대 A8 + G1 + HSR6, 같은 9 장면 × 2 = 36 run.

## 1. HSR6 (6 단계 단계 판정)

- probe: `/mnt/dev/AG3S_VLA-storage/results/development/20261029T013324+0900_b3b69311f948_7da63daefa49/verify/T64/partA/probe_6.npz`.
  - feature `kv_L4` (지금 서빙 HSR 과 같은 prefix KV layer 4, 추가 forward 없음).
  - StandardScaler + multinomial logistic C 0.03.
  - class 순서: approach · grasp · lift · transport · placement · retreat.
- 요청마다 6 단계 확률과 argmax 를 summary_json 에 기록한다 (`hsr6`). 지금 서빙의 pick / place / home HSR 은 그대로 둔다.
- **기준 경로 (B 답 2026-10-29):** 추가 전처리 없음.
  - feature: 지금 서빙 HSR (`SubtaskProbe.predict`) 에 들어가는 그 `kv_L4` (512 = [유효 text token 평균 | 이미지 768 token 평균], V · layer 4 · head 0).
    - 경로: `AG3S/trajopt/attention_policy.py` `AttentionPolicy._attention_and_feature` → `sampler.attention_and_subtask_feature`.
  - 적용: `AG3S/ag3s/stages/subtask_probe.py:110` `SubtaskProbe.proba` 와 같은 식.
    - p = softmax(((x − mean) / scale) @ coef.T + intercept), float64.
    - probe_6.npz 필드: mean · scale (512), coef (6×512), intercept (6), classes.
  - 판정: 요청마다 **raw argmax** (debounce · 지속 조건 없음). debounce 를 넣으려면 별도 flag. parity (H3) 는 raw 로 본다.
  - offline 기준 code: `AG3S/ag3s/experiments/studies/t64_partB.py` (`t50_core.LogReg.proba`).
  - SubtaskProbe 가 classes 를 npz 에서 읽는지 구현 때 확인한다.
- 물리 evidence: grasp latch 의 `closed` · `blocked` (손가락 막힘) · latch 보유 (HELD) — 지금 `summary.grasp.evidence` 에 있는 값.

## 2. 4 사용처 (각각 기본 꺼짐 flag, 결과 전 고정)

| # | flag (이름은 구현 때 정함, 예) | 규칙 |
|---|---|---|
| ① | `--hsr6-attach-assist` | 쥔 물체 attach = 지금 strong HELD **∨** (HSR6 ∈ {lift, transport, placement} ∧ `closed` ∧ `blocked`). T64 Part C 와 같은 식 |
| ② | `--hsr6-held-gate` | 쥔 물체 관련 허용 — held-body cover, 쥔 사과–탁자 · 사과–바구니 접촉 허용 — 을 HSR6 ∈ {lift, transport, placement} **∧** 물리 evidence (`blocked` ∨ latch HELD) 일 때만 준다. 지금보다 허용이 줄기만 한다 |
| ③ | `--hsr6-target-exempt` | 손가락 · 손바닥 (target 접촉 권한 링크) 이 target 을 지운 거리장을 보는 것을 HSR6 ∈ {approach, grasp} 일 때만 한다. retreat (놓은 뒤) 에는 놓은 사과도 장애물로 본다. lift · transport · placement 는 지금 규칙 (쥔 물체 처리) 그대로 |
| ④ | `--hsr6-objective` | 손가락 이웃 접촉 허용 (`--neighbour-contact-mm`, IFG-NBC) 과 닫는 행 목표 (A4 / G1) 를 지금 규칙 ∧ HSR6 = grasp 일 때만 켠다 |

- 4 flag 를 모두 켠 구성이 "HSR6 구성" 이다. 하나라도 꺼지면 그 부분은 지금과 bit 동일해야 한다.

## 3. 검증 (결과 전 고정)

| # | 검사 | 통과 조건 |
|---|---|---|
| H1 | 단위 test | `.venv-ag3s` 로 메인 checkout 에서 `tests/trajopt` + 새 test (4 규칙 각각의 참 · 거짓 경우) |
| H2 | 꺼짐 동등성 | 4 flag 를 끄면 핵심 평가 기록 요청 ≥ 20 개 재생에서 refined chunk bit 동일 (G1 합치기의 V2 와 같은 방식) |
| H3 | 단계 parity (offline) | 기록 frame 에서 서버가 낸 HSR6 단계 = B 의 `t64_partB` 단계, 요청 ≥ 50 개 중 ≥ 99 % 일치 |
| H4 | 발동 기록 | 4 규칙마다 "지금 규칙과 결정이 달라진 요청" 을 summary_json 에 표시한다 (lead 가 closed-loop 에서 센다) |

## 4. closed-loop (lead, H1–H3 통과 뒤)

- 구성:
  - (가) A8 + G1 (G1 · A4 대체 방식)
  - (나) (가) + ① – ④
  - 같은 9 장면 × 2 흔들림 = 36 run.
  - 비교 기준선: 핵심 평가 1차 (A8) 기록.
- 주 지표: 금지 접촉 없는 성공 (§0.1 정의).
- 보고: HSR 로 바뀐 결과를 짝 (같은 장면 · 흔들림) 으로 보인다.
  - 규칙별 발동 요청 수
  - 단계 띠 (정답 · 서빙 HSR · HSR6)
  - 바뀐 run 의 실제 씬
- 2 번 반복이라 구성 효과는 판정하지 않는다. 바꾸는 조건은 결과를 본 뒤가 아니라 이 문서에서 정한다:
  - 금지 접촉 run 이 늘지 않음.
  - 금지 접촉 없는 성공이 (가) 보다 낮지 않음.
  - 그러면 HSR6 을 최종 구성에 넣을지 사용자에게 묻는다.

## 5. 진행 (lead)

- **구현 완료 (2026-10-29 ~04:30 KST):** `AG3S-CORE-EVAL-HSR6.impl.md`.
  - flag: `--hsr6-attach-assist` · `--hsr6-held-gate` · `--hsr6-target-exempt` · `--hsr6-objective` (+ `--hsr6-probe`).
  - H1 통과: test 55/55, 전체 2,178 통과 (알려진 기존 실패 1).
  - H2 통과: 꺼짐 bit 동일 106/106.
  - H3 통과: 단계 parity 1,200/1,200 · 7,500/7,500.
  - H4 구현: `summary_json.hsr6.changed`.
- **결과 전에 적어 둔 위험 (implementer, 규칙은 바꾸지 않음):**
  - ②: 쥔 직후 (사과가 2 cm 오르기 전, HSR6 = grasp) 지지면 들어올림이 거두어져 쥔 몸통 구 – 탁자 위반 HOLD 가 날 수 있다.
  - ④: 닫기 전 HSR6 가 대부분 approach 라 G1 · A4 · NBC 가 거의 꺼질 수 있다.
  - closed-loop 에서 `hsr6.changed.*` 와 HOLD 사유를 같이 센다.
- **결합 closed-loop (2026-10-29 04:45 KST 대기열):**
  - snapshot `snapshot_final` (`$RUN/snapshot_final_head.txt`).
  - kit: `CB_{G1,HSR}_k{0,1}` · `OCC_CB_{G1,HSR}`, `cb_launch.sh`, 독점 lock.
- **T64 Part B 판정 (세션 B, 2026-10-29, commit `c7a09fdb`):** 사용자가 물리 정답 정의를 영상으로 확인했다.
  - rule 보다 나은 단계 구분: 미충족 (CI [−0.043, +0.087]).
  - 앞서 알아채는 가치: 미충족.
  - 보완 가치 (Part C, attach 보완): 충족.
  - B 결론: closed-loop 후보 (HSR6 + 물리 evidence) 는 그대로 유효하다. T65 파일은 B 가 commit 했다 (`28a7aacf`).
