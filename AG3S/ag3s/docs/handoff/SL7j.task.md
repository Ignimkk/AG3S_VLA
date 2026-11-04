# SL7j — LIBERO task 7 손실의 원인 가르기 (표면 외 두 차이)

> writer: lead (benchmark checkout), 2026-11-04 09:1x KST. **결과를 보기 전에 고정.** 판정 주체: lead 자율 판정 D12 (`doc/active/experiments/SL7_AUTONOMOUS_DECISIONS_20261104_KO.md`).
> 앞 단계: SL7i — `-pc-surface-fb`는 LIBERO 9 중 task 7만 pc판과 행동이 다르다(call 4부터). 첫 표면 채택은 call 7이다. 그래서 손실은 채택된 표면 때문이 아니다.

## 판 (진단용, pc판에서 하나씩 바꿈)

| 판 | 정의 |
|---|---|
| `AG3S-final-panda-v3-e0to-pc-l8h2explicit` | pc판 + `--attention-source action:last:L8H2` (기본 경로 대신 확장 경로) |
| `AG3S-final-panda-v3-e0to-pc-nonte` | pc판 − `--nontarget-evidence surface` |

장면은 libero_spatial task 7 × init 0, seed 7, conversion C(1 episode씩)이다. 대조는 pc판(성공)과 `-pc-surface-fb`(실패, call 4부터 다름)이다.

## 판정 규칙 (결과 전 고정)

각 판의 `actions_response`를 pc판과 call마다 비교한다.
1. 한 판만 pc판과 달라지고, 그 판의 첫 차이 call이 4이면 → 그 차이가 task 7 손실의 출발점이다.
2. 둘 다 pc판과 bit 동일하면 → 두 차이는 각각으로는 원인이 아니다(조합 또는 표면 경로 자체가 원인). 표면 경로가 기권 call에서도 무엇을 바꾸는지 다음에 본다.
3. 둘 다 달라지면 → 첫 차이 call과 성공 여부를 보고한다.
