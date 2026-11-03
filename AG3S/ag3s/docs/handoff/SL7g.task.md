# SL7g — 표면 추출 + 기권 시 공으로 물러남 (pc판 위, LIBERO 9)

> writer: lead (benchmark checkout), 2026-11-04 06:1x KST. **결과를 보기 전에 고정.** 판정 주체: lead 자율 판정 D5 (`doc/active/experiments/SL7_AUTONOMOUS_DECISIONS_20261104_KO.md`).
> 앞 단계: SL7f — pc판 + 표면 추출 + ATS gate 3/9 (gate 없는 판과 잃은 episode가 같음: task 1 · 6 · 7). 오프라인 SafeLIBERO 재생에서 gate는 채택 영역을 77 → 126 call로 늘렸지만 GT target 지움은 같았다.

## 가설

P0m의 표면 추출은 `--surface-keep-last`(영역이 처음부터 없으면 손 층에서 아무것도 지우지 않음, 공 · 점으로 물러나지 않음)와 `--surface-no-ball`(공을 만들지 않음)과 함께 쓴다. SafeLIBERO 재생에서 표면 영역이 채택된 call은 10–16%뿐이었다. 그 밖의 call에서는 target이 거리장에 남는다. 장애물이 없는 LIBERO에서 pc판(공)보다 3 episode를 잃는 것은 이 때문이라고 본다.

## 판

| 판 | 정의 |
|---|---|
| `AG3S-final-panda-v3-e0to-pc-surface-fb` | pc판 + `--target-extent attention-surface --surface-target-check overlap --attention-surface-device cuda`, seed 지도 `action:last:L8H2`. keep-last · grasp-hold · no-ball과 이들에 묶인 hull 채우기 · 접촉 여유는 **뺀다** → ATS와 겹치는 영역이 있으면 표면, 없으면(기권 · 기각) 그 call은 공(`estimated`) 범위 |
| `AG3S-final-panda-v3-e0to-pc-surface-fb-gate` | 위 + `--surface-seed-ats-gate` |

## 판정 규칙 (결과 전 고정)

장면 · 지표는 SL7e와 같다. 비교: pc판 6/9(포화 10%), pc판 + 표면 추출 3/9(19%).
1. 두 판 중 하나라도 ≥ 5/9 이고 포화 ≤ 15% → 성공이 더 높은 판(같으면 gate 없는 판 = 변경이 적은 쪽)을 **SafeLIBERO 11 episode** closed loop에 넘긴다(`experiments/segmentation/run_sl7_safelibero11.sh`). 비교 pc판 SL6-H(안전 성공 1 · 공식 성공 1 · 충돌 5).
2. 둘 다 ≤ 3/9 → 가설이 틀렸다. 표면 추출 자체(영역 품질)를 closed loop 기록으로 봐야 한다 → 서버 call 기록에 표면 상태를 남기는 기록 기능을 다음에 넣는다.
3. 4/9 → 가르지 못함.

이 판은 development 최종 구성(P0m: keep-last · no-ball)과 다르다. 채택은 사용자 피드백 사항이며, 여기서는 후보로만 올린다.
