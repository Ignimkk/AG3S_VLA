# SL10f — 진단: 기하 미인증 HOLD를 풀면 LIBERO-Safety 실패가 얼마나 돌아오나 (결과 전 고정, 채택 없음)

> writer: lead (benchmark checkout), 2026-11-05 06:5x KST, 자율 판정 시간(`doc/active/experiments/SL_AUTONOMOUS_DECISIONS_20261105_KO.md` E27).
> 근거: `outputs/2026-11-05_기준판_실패지도_3벤치/`
> - LIBERO-Safety 10에서 VLA 단독은 10/10 성공, 위반 0이다. 기준 판은 1/10이다.
> - 실패의 큰 유형 A는 "HOLD로 멈춤"(pc판 3, 표면 판 5)이다. 그 HOLD의 다수는 놓은 뒤 · 쥐는 중 · 쥔 중의 기하 미인증(`uncertified`: `degraded` · `no_target`)이다.
> 이름: 보고에서는 "미인증 HOLD 풀기 진단"이라고 부른다.

## 1. 성격

- **진단 run이다. 후보 판정을 하지 않는다.** "미인증 → HOLD"는 안전 규칙이다. 이것을 바꿀지는 사용자 판정 사항이다. 이 run은 그 판정에 쓸 숫자만 만든다.
- 바꾸는 것은 기존 서버 flag `--allow-uncertified` 하나다(`safety.require_certified_geometry=False`). 미인증 사유가 `uncertified_waived`(실행)로 바뀐다.
- 충돌 행 HOLD · TrajOpt 제약 · latch는 그대로다.

## 2. 변경

- 판 등록만 한다: `AG3S-final-panda-v3-e0to-pc-uncwaive` = pc판 + `--allow-uncertified`(Panda profile, 새 코드 경로 없음).
- 테스트:
  - 새 판 = 기준 + flag 하나
  - 기존 판 bit 동일(argv 비교)

## 3. 실행

- LIBERO-Safety 10 · SafeLIBERO 11 · LIBERO 9를 돈다. 순서도 이대로다.
- seed · conversion C는 SL9와 같다.
- 기준은 pc판 `76283ac9` 재실행(LIBERO-Safety · SafeLIBERO)과 SL10 기록 run(LIBERO)이다.

## 4. 기록할 숫자 (결과 전 고정)

- 벤치별 주 지표와 금지 접촉(위반 · 충돌)을 기준과 비교한다.
- 기준에서 유형 A였던 episode가 어떻게 끝났는지 적는다(성공 / 다른 유형 / 접촉).
- 풀린 미인증 call(`uncertified_waived`) 수, 그 call 뒤 실제 접촉이 났는지 적는다.
- **안전 쪽 경고:** 위반 · 충돌이 하나라도 늘면, 그 episode와 시점(풀린 미인증 call 뒤인지)을 따로 적는다.

## 5. 출력

- 원시 기록: storage `…_sl10f-*`
- 보기: `outputs/2026-11-05_SL10f_미인증HOLD풀기_진단/`
