# SL5i — 파지 인식 수정(latch 파지 확인) + closing 중 쥔 물체 면제

> writer: lead · 읽는 쪽: implementer (다음에 verifier) · 작성 2026-10-22 KST
> 사용자 판정 2026-10-22: "추천 방향에 맞게 수행" = ① latch가 실제 파지를 인정하도록 파지 확인을 고친다 + ② closing 중 쥔 물체에 대한 손의 제약을 뺀다.
> **이름 규칙 (사용자 2026-10-22):** 보고와 기록에서는 번호가 아니라 무엇을 바꿨는지로 부른다. 내부 profile key는 연속성을 위해 번호를 유지한다.

| 내부 key | 부르는 이름 | 내용 |
|---|---|---|
| `V11AB-pb6` | **파지 인식 수정판** | pb5(grasp zone 허용판) + ①(a)(b) |
| `V11AB-pb7` | **파지 인식 + closing 면제판** | pb6 + ② |

변경마다 따로 검증한다. 같은 dev 집합에서 같은 역변환(C, `lag_inverse_tracking`)으로 비교한다.

## 왜 (SL5g · SL5h · SL5h-G 숫자, lead 재확인 2026-10-22)

bowl을 실제로 들고 있는데 latch가 `closing`에 머무는 call이 LI 189개, LIBERO 10 37개다(SL5g pb5-C).

- 그 call에서 실패한 check: LI는 reach 179 · blocked 53 · dropped 42, LIBERO 10은 dropped 36 · reach 33.
- latch가 held로 넘어가지 않으면 다음 일이 이어진다:
  - AG3S의 manipulated centroid가 table 위의 옛 위치에 남는다.
  - TO가 손 근처의 쥔 bowl을 obstacle로 보고 손을 위로 민다(SL5h D4: 위 1431행, 아래 243행).
  - 역변환이 그 밀기를 실행해 over-lift가 생긴다.

### (a) reach가 나중 capture로 다시 재져 실패한다

- 기준: `reach-at-capture`(pb1)는 손-bowl 거리를 bowl의 **마지막 관측(capture)** 시점 자세로 잰다.
- **closing에 들어갈 때는 통과한다:**
  - LI t2e4 seq 6: reach 91.1 mm (limit 120), geometry frame 5
  - LIBERO t5e0 seq 10: 81.7 mm
- **그 뒤 다른 capture가 들어오면서 실패하고 그대로 고정된다:**
  - t2e4: capture frame 21에서 123 mm로 34 call 동안 고정
  - t5e0: seq 11부터 133 mm
- bowl은 쥐기 전에는 움직이지 않는다. 그러니 한 번의 닫힘 시도 안에서 "그 bowl에 손이 닿을 거리였다"는 사실은 바뀌지 않는다.

### (b) `dropped`가 늦은 latch 때문에 기준을 잃는다

- `dropped`는 `opening_before − opening ≥ min_drop`(0.05)인지 본다. `opening_before`는 LATCHED→CLOSING 전이 순간의 직전 측정 개도다(`grasp_latch.py` 553–555행).
- LIBERO t5e0에서는 latch가 손가락이 이미 거의 닫힌 뒤(seq 9, 개도 0.072)에 latched가 됐다.
  - 그래서 `opening_before`가 0.0719로 잡혔다.
  - 그 뒤 개도 0.055~0.072 범위에서 drop이 0.05에 못 미쳐 34 call 동안 `dropped=False`였다.
- 손가락은 실제로 열림(≈1.0)에서 닫혔다. 기준이 잘못 잡혔을 뿐이다.

### ② held로 넘어가기 전 구간

- closing 중 손가락이 닫혀 bowl을 쥔 구간에서도 같은 위쪽 밀기가 나온다.
- pb4의 held 면제(쥔 물체에 대한 손의 contact-link 행을 뺀다)는 HELD에서만 적용된다.

## 무엇을 바꾸나

각 변경은 flag 하나로 켜고, 기본값은 꺼짐이다. flag가 꺼져 있으면 기존 동작과 bit 단위로 같아야 한다.

### ①(a) `--grasp-reach-sticky` (pb6)

- 같은 닫힘 시도(LATCHED→CLOSING부터 reopen · detach · LATCHED 복귀 전까지) 안에서 reach check가 **한 번 통과했으면** 그 시도가 끝날 때까지 통과로 둔다.
  - 통과한 순간의 값과 capture frame은 record에 남긴다.
  - 그 뒤의 reach 값도 계속 측정해 record에 남긴다.
- **같은 판단을 AG3S의 `attach` reach gate(`pipeline.py` 736행 부근)에도 적용한다.** 그래야 latch는 held인데 attach가 거부되는 어긋남이 생기지 않는다.
- 다른 check(`closed` · `not_empty` · `blocked` · `settled` · `manipulated` · `dropped`)는 그대로 모두 참이어야 한다.

### ①(b) `--grasp-drop-from-open` (pb6)

- `dropped`의 기준 개도를 "LATCHED→CLOSING 순간의 개도" 대신 **"닫힘 명령이 시작되기 전 마지막으로 열려 있던 측정 개도"**로 바꾼다.
  - 예: 명령 ≥ `gripper_closed_below`이고 개도 ≥ `gripper_closed_below`였던 마지막 측정
  - SEARCHING 단계에서부터 추적한다.
- 그런 측정이 없으면 기존 값으로 돌아간다(fail-closed는 유지).

### ② `--closing-hand-exempt` (pb7)

- pb4 held 면제와 같은 규칙(grasping hand의 authorized link 행 중 nearest surface가 manipulated object인 행을 뺀다)을 다음 조건에서도 적용한다:
  - latch가 **CLOSING**이고
  - 그 call의 evidence에서 `closed` · `not_empty` · `blocked`가 모두 참일 때(손가락이 무언가를 쥐고 멈춘 상태)
- 다른 물체 · table · obstacle에 대한 행은 그대로 남긴다.
- record의 `grasp_contact`에 `closing_exempt_rows`를 추가한다.

### profile (`AG3S/adapters/panda/profile.py`)

- `V11AB-pb6` = pb5 + `--grasp-reach-sticky --grasp-drop-from-open`
- `V11AB-pb7` = pb6 + `--closing-hand-exempt`
- 주석에 위 숫자 근거를 짧게 남긴다.

## 구현 전후 확인 (CPU, implementer)

1. **기존 동작 보존:** flag가 꺼진 상태에서 기존 test가 통과하고 RB-Y1 golden이 bit 단위로 같은지 확인한다. 회귀 기준선은 `regression-baseline` skill로 판정한다.
2. **단위 test:**
   - sticky reach: 시도 안에서 유지, 시도가 끝나면 초기화
   - drop-from-open 기준
   - closing 면제: 조건이 참일 때만, manipulated object 행만 빠지는지
3. **offline latch replay:** 기록된 calls log의 latch 입력(실행 피드백 개도 · 명령 · closing_steps · manipulated_state · reach)을 GraspLatch에 다시 넣어, flag on일 때 **처음 held로 넘어가는 seq**를 구한다.
   - 대상: SL5g pb5-C의 LI 20 · LIBERO 10 episode
   - 결과를 원래 seq, MuJoCo 실제 lift 시점(`outputs/verify/SL5h/calls/*.json`의 `lifted`)과 나란히 표로 낸다.
   - **거짓 held 확인:** 빈손이거나 다른 물체를 쥔 episode가 held로 넘어가지 않는지 본다. 빈손 = MuJoCo에서 손가락이 target에 닿지 않음.
   - 첫 held 뒤에는 closed loop가 갈라지므로 첫 전이만 의미가 있다. 그 점을 적는다.

## 넘겨줄 것

- `AG3S/ag3s/docs/handoff/SL5i.impl.md`
  - 바뀐 file과 diff 범위
  - flag와 profile
  - test 결과
  - 기존 동작이 그대로라는 근거
  - offline replay 표
- replay 표와 script는 `outputs/verify/SL5i-pre/`에 둔다.
- commit은 하지 않는다(lead가 한다).

## 규칙

- GPU는 쓰지 않는다. CPU worker ≤ 2, RSS ≤ 3 GB. working set이 62 GiB를 넘으면 멈춘다.
- 다른 세션 process는 건드리지 않는다.
- 지금 진행 중인 benchmark run은 없다(2026-10-22 확인).
