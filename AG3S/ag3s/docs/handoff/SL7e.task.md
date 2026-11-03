# SL7e — 잘 되는 pc판 위에서 가르기: (C) pc판 + 표면 추출 (후보) · (A) pc판 − HSR6 (진단) — LIBERO 9

> writer: lead (benchmark checkout), 2026-11-04 05:1x KST. **결과를 보기 전에 고정.**
> 판정 주체: lead 자율 판정 (사용자 지시 2026-11-04 04:1x KST "11:30까지 스스로 판정, md로 기록") — 기록 `doc/active/experiments/SL7_AUTONOMOUS_DECISIONS_20261104_KO.md` D2.
> 앞 단계: SL7d — (가) P0m+L8H2 1/9 (포화 23%), (나) pc판+latch 6/9 (실행 행동이 pc판과 bit 단위로 같음 → latch label은 이 장면에서 효과 없음).

## 왜 이 두 판인가

P0m 쪽에서 pc판의 요소를 하나씩 더한 판은 모두 0–1/9였다: 이웃 허용(nbc) 0, Panda 보정(b) 1, 공 범위 + 비 target 증거 0, L8H2 1. latch label은 pc판에서 효과가 없었다. 아직 따로 보지 않은 단일 차이는 **HSR6 규칙**이고, 요소 여러 개가 함께 있어야 하는 경우(조합)도 남는다. 그래서 방향을 바꿔 **6/9인 pc판에서 출발**한다:
- (C) `AG3S-final-panda-v3-e0to-pc-surface` = pc판의 target 범위만 P0 표면 추출(hull 채우기 · 접촉 여유 10 mm 포함)로 바꾼다. 표면 seed 지도 = pc판 ATS cell `action:last:L8H2`(명시). config가 표면 추출과 함께 받지 않는 `--nontarget-evidence surface`는 뺀다. HSR6 · probe · Panda 보정 · 이웃 허용은 pc판 그대로. 표면 추출은 최종 구성에 필수이므로 **최종 Panda 후보**를 겸한다.
- (A) `AG3S-final-panda-v3-e0to-pc-nohsr6` = pc판 − HSR6 규칙 6개(attach assist, held gate, target exempt, objective + scope, placement contact), probe 없음. **진단 전용.**

## 장면 · 지표

SL7과 같은 libero_spatial task 0–4 · 6–9 × init 0, seed 7, conversion C. 주 지표 공식 성공, 진단 지표 0.5 넘게 바뀐 call · ±1 포화(episode 평균). 비교 pc판 6/9 · 포화 10%.

## 판정 규칙 (결과 전 고정)

1. (C) ≥ 5/9 이고 포화 ≤ 15% → "pc판 + 표면 추출"을 Panda 최종 **후보**로 SafeLIBERO 11 episode closed loop에 넘긴다(SL7 규약 §2 장면, 비교 pc판 SL6-H 기록). ≤ 3/9 → 잘 되는 판에서도 표면 추출이 성공을 깎는다 → 표면 seed 쪽 수정(카메라별 정규화 등)을 다음 단계로 정한다. 4/9 → 가르지 못함, (A)와 함께 본다.
2. (A) ≤ 3/9 → HSR6 규칙이 Panda 성공에 필요하다(P0m이 HSR6를 끈 것이 실패 원인 후보). ≥ 5/9 → HSR6는 원인이 아니다(조합 문제).
3. 결과는 `outputs/2026-11-04_SL7e_*`와 자율 판정 기록에 남긴다.
