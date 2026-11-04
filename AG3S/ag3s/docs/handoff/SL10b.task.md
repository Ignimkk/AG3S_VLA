# SL10b — 쥔 뒤 연속성 · 꺾임 변화율 항 끄기: closed loop 확인 (결과 전 고정)

> writer: lead (benchmark checkout), 2026-11-05 01:5x KST, 자율 판정 시간(`doc/active/experiments/SL_AUTONOMOUS_DECISIONS_20261105_KO.md` E12).
> 근거:
> - SL10 J2(`outputs/2026-11-05_SL10_제약기록_J0J1_계기근거/`): 쥔 뒤 유효 call 4개 모두, 이 두 항을 끄면 reference 편차 > 0.1 rad에서 벗어났다. 표본이 작다.
> - SL6-F: 이 항을 쥔 뒤에만 켜자 포화 step이 1194 → 1621로 늘었다. 그때 원인 확인 실험은 하지 않았다.
> - SL10 action 한계 제약(E8): 포화는 없앴지만 HOLD가 9배로 늘어 pc판에서 6/9 → 3/9가 됐다. 그래서 "포화를 막는 것"보다 "큰 수정을 요구하는 원인을 끄는 것"을 본다.
> 이름: 보고에서는 "쥔 뒤 변화율 항 끔"이라고 부른다.

## 1. 판 (기본 꺼짐 flag만 조합, 새 코드 없음)

| 판 | 정의 |
|---|---|
| `AG3S-final-panda-v3-e0to-pc-termsoff` | pc판에서 `--continuity-after-grasp --deflection-rate-after-grasp`를 빼고, 두 항을 전 구간 끈다(P0m-termsoff와 같은 방식) |
| `AG3S-final-panda-v3-e0to-pc-surface-termsoff` | pc판 + 표면에 같은 변경 |

- 기준 판과 서버 인자 차이는 이것뿐이어야 한다(test로 확인).
- 쥐기 전에는 두 기준 판도 이 항이 꺼져 있다. 그래서 행동 차이는 쥐는 중 · 쥔 뒤에서만 생겨야 한다. 쥐기 전 call은 bit 동일한지 확인한다.

## 2. 순서

1. LIBERO spatial 9(SL9 · SL10과 같은 장면 · seed 7 · conversion C), 두 판을 동시에 돈다(GPU 독점).
   - 기준: SL10 제약 기록 run(`62c0076f`, SL7 · SL9와 bit 동일).
2. 판마다 LIBERO 주 지표 ≥ 기준이면 그 판만 SafeLIBERO 11 · LIBERO-Safety 10을 돈다.
   - 기준: pc판 + 표면은 SL9 run(`914eb50f`)이다.
   - pc판은 같은 commit에서 기준을 다시 돈다. SafeLIBERO · LIBERO-Safety pc판 기준이 없기 때문이다.

## 3. 판정 규칙 (결과 전 고정)

- **기제 확인:** 쥔 뒤 구간(`grasp.state == held`)에서 두 값을 본다.
  - 실행 action 포화 성분이 기준의 절반 이하인가.
  - reference 편차 > 0.1 rad call 비율이 기준의 절반 이하인가.
  - 하나라도 아니면 "기제 미확인"으로 보고한다. 주 지표 판정은 그대로 적는다.
- **기준을 대신할 후보:** SL9 §5 그대로.
  - 세 벤치마크 모두 주 지표 ≥ 기준이다.
  - 금지 접촉 ≤ 기준이다.
  - 적어도 하나에서 주 지표 차이 > 2 episode다.
  - 2 episode 이하 차이는 "구분 못 함"이다.
- **안전 쪽 확인:** 쥔 뒤 실제 MuJoCo 접촉(쥔 물체 · 손 · 팔과 장애물)이 기준보다 늘면, 주 지표와 상관없이 "후보 아님"이다. 이 항이 맡던 일은 쥔 물체를 안정되게 옮기는 것이고, 그것이 깨지는지 보기 위해서다.
- HOLD 증가는 보호가 아니다. 실패마다 원인 층(ATS · HSR · LTE · CCR)을 적는다.

## 4. 실행 규칙

- 구현(판 등록 + test)은 worktree `wip/sl10-replay`에서 한다. main merge는 진행 중인 GPU 사슬이 끝난 뒤에 한다(D10 교훈).
- 출력: storage `results/benchmark/<날짜_commit>_sl10b-*`, 보기 `outputs/2026-11-05_SL10b_쥔뒤변화율항끔/`.
