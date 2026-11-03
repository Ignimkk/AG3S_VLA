# SL7b — Panda P0m의 행동 포화 원인 가르기 (LIBERO 9 episode) + 표면 영역 기각 원인 기록

> writer: lead (benchmark checkout), 2026-11-04 KST. **결과를 보기 전에 고정.**
> 사용자 판정 (2026-11-04 00:xx KST, development d2 세션 전달 — 이 checkout 사용자가 "development와 상의"로 넘긴 판정):
> 1. T92b: hull 채우기 · 접촉 여유 10 mm 포함이 최종. `p0m`이 최종 기준 판이고 `p0m-nohull`은 끈 판으로 유지한다(두 flag는 각각 독립).
> 2. Panda TrajOpt 정의: 아래 (c) · (b) · (d)를 같은 LIBERO 9 episode로 돌려 행동 포화 원인을 가르고, 성공이 비슷하면 보정이 적은 쪽을 고른다.
> 3. SafeLIBERO closed loop 전에 오프라인 재생의 "지운 voxel 0인 call 41%" 원인을 표면 상태 기록으로 확인한다.
> 앞 단계: `SL7.task.md`, 결과 `outputs/2026-11-04_SL7_LIBERO_pilot_P0m/` (p0m 0/9, p0m-nbc 0/9, pc판 6/9, VLA 단독 9/9).

## 1. 판

| 판 | 정의 | 무엇을 가르나 |
|---|---|---|
| (c) `AG3S-final-panda-p0m-termsoff` | P0m − `--continuity-after-grasp --deflection-rate-after-grasp` + `--w-continuity 0 --w-deflection-rate 0` (두 항을 모든 구간에서 끔, 3판과 같은 변경) | 포화가 연속성 · 꺾임 변화율 항 때문인가 |
| (b) `AG3S-final-panda-p0m-pandafix` | P0m + pc판의 Panda 보정 `--reach-at-capture --links arms --held-hand-exempt --latch-phase --grasp-zone-mm 10 --grasp-reach-sticky --grasp-drop-from-open --closing-hand-exempt` (V11AB-pb7 − pb2 gate). HSR6는 계속 끔 | 포화가 팔 link 모델 · 손 예외 · latch · reach 보정의 부재 때문인가 |
| (d) 위 둘 중 고른 판 + `--passthrough-when-clear` | VLA 행동에 위반이 없으면 TrajOpt를 거치지 않음(T88). **Panda 서버에 연결 · 테스트 후** 등록 | VLA 성공을 얼마나 되찾나 |

target 범위(표면 추출, hull 채우기 · 접촉 여유 포함)와 단계 판정(`--subtask-from-latch`)은 세 판 모두 P0m 그대로이고, D1(이웃 허용 끔)도 그대로다.

## 2. 장면 · 지표

- 장면: SL7과 같은 libero_spatial task 0–4, 6–9 × init 0, seed 7, conversion C (VLA 단독 9/9).
- 주 지표: 공식 성공(9개 중). 진단 지표: VLA 행동 대비 실행 행동 차이가 0.5 넘는 call 비율, 실행 행동 값의 ±1 포화 비율, HOLD 비율(보호로 세지 않음).

## 3. 판정 규칙 (결과 전 고정)

1. **(c)와 (b) 중 고르기:** 공식 성공이 더 높은 판. 차이가 **1 episode 이하이면 "비슷"으로 보고 보정이 적은 (c)**.
2. **(d)의 기반:** 위 1의 판(사용자 판정 문구: (c) · (b) 중 성공이 높은 쪽, 같으면 (c)).
3. **SafeLIBERO로 넘어가는 조건:** 1에서 고른 판(또는 (d)가 더 높으면 (d))이 LIBERO 9개 중 **5개 이상** 성공. 그 미만이면 SafeLIBERO 전에 보고하고 판정을 받는다.
4. **포화 원인 기록:** 고른 판의 포화 비율이 p0m(29%)보다 줄었는지와 pc판(10%) 수준인지를 적는다. 원인 층은 CCR(TrajOpt · 변환)로 기록한다.

## 4. 표면 영역 기각 원인 (오프라인, SafeLIBERO 20 episode)

- `experiments/segmentation/eval_real_ats_safelibero.py`가 call마다 `AG3S._surface_frame`의 스칼라(`applied`, `why`, region node 수, target check 수치, grasp hold 여부)를 `frames.jsonl`의 `surface` 키로 남긴다.
- P0m 판으로 같은 20 episode를 다시 재생하고, **지운 voxel 0인 call을 `why`로 나눈다**: `not_ats_target`(겹침 검사 기각) · `grasp_hold`(쥐는 중 고정) · 선택 자체가 영역을 못 만듦 · target이 안 보임(GT target voxel 0).
- 판정 규칙은 두지 않는다(원인 기록용). SafeLIBERO closed loop 전에 보고한다.

## 5. 순서 · GPU

오프라인 기각 원인 재생(shared, 약 11분) → (c) · (b) LIBERO closed loop(exclusive, 각 약 20분) → (d) 이식 · 테스트 → (d) LIBERO → 보고. 결과는 `outputs/2026-11-04_SL7b_*`와 development d2에 요약.
