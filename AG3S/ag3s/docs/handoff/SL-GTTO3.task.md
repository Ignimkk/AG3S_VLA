# SL-GTTO3 — GT 판에서 TrajOpt 쪽 두 변경(연속성 · 꺾임 변화율 항 끄기, 위반 없으면 통과) 가르기

> writer: lead, 2026-11-05 03:5x KST. 사용자 위임(2026-11-05 ~01:00 KST, "오전 11:30까지 스스로 판정하며 수행")에 따른 lead 자율 판정 G26.
> **결과를 보기 전에 고정.** 선행: [SL-GTTO2.task.md](SL-GTTO2.task.md) (F1 · F2), [SL7b.task.md](SL7b.task.md) (P0m 위 같은 두 변경).

## 0. 근거 — SL-GTTO2 smoke (2026-11-05 03:09–03:23, `AG3S-commonto-gt2`, commit `89b42459` + GTTO · GTTO2 patch)

smoke 키 4개는 모두 통과했다 (LIBERO t4e0 · SafeLIBERO t3e0 첫 call grounding `ok` HOLD 없음, LIBERO t0e0 · SafeLIBERO t1e1 attach 뒤 `attached_point` 행 0).
그러나 공식 성공은 SafeLIBERO t1e1 하나뿐이다 (t0e0 · t4e0 · t3e0 실패, 같은 기계 VLA 단독은 넷 다 성공).

| 관찰 (call 기록) | 수치 |
|---|---|
| TO 가 VLA 경로를 바꾼 정도(`to.reference_deviation`)별 변환 C 포화(`conversion.saturated_rows` ≥ 5/10) — v1 30 episode + gt2 smoke 4 episode, HOLD 아닌 call | dev 0: 0/289 · < 0.01: 1/172 · 0.01–0.05: 52/227 · 0.05–0.2: 269/386 · ≥ 0.2: 310/315. median `max_abs_correction` 0 · 0.05 · 1.46 · 3.91 · 12.33 |
| held 구간 | 모든 call `trajopt_status = violated` 는 `budget_only`(반복 상한 3, 최종 검사 통과) — 위반이 아니라 경로 수정. 그래도 dA(실행 action 변화) 1.5–1.9, 포화 20/20 (LIBERO t0e0 gt2) |
| F1 로 열린 latched 구간 | LIBERO t4e0: latched 42 call 중 포화 36, attach 없음. SafeLIBERO t3e0: latched 27 중 22, attach 없음 |
| 공용 TO 의 TrajOpt 설정 | `--continuity-after-grasp --deflection-rate-after-grasp` 켬 (V11AB 무게 0.5 / 20), `--passthrough-when-clear` 없음, minimal intervention gate 없음 |

SL7b 는 P0m 위에서 같은 두 변경 (c) · (d)를 시험했고 모두 1/9 였다. 그때 결론 "TrajOpt 는 원인 아님"은 P0m 인식(표면 영역 기각)이 먼저 막고 있었기 때문이다 (SL-GTTO.task §0).
GT 인식으로 그 층을 걷어 낸 지금 같은 두 변경을 다시 가른다. 사용자 지시("통과하지 못하면 그것은 TO(또는 Panda 변환 C · 쥐기 처리) 문제입니다. baseline 비교 전에 그 층을 먼저 고칩니다")의 TO 층이다.

## 1. 판 (profile 등록만, 새 code 없음 — 모든 flag 는 기존 serve_safe / Panda 서버 flag)

| 판 | 정의 |
|---|---|
| `AG3S-commonto-gt2` (기준) | SL-GTTO2 본 실행 그대로 |
| `AG3S-commonto-gt2-termsoff` | gt2 − `--continuity-after-grasp --deflection-rate-after-grasp` + `--w-continuity 0 --w-deflection-rate 0` (3판 SL6-E · SL7b (c) 와 같은 변경) |
| `AG3S-commonto-gt2-pass` | gt2 + `--passthrough-when-clear` (T88, SL7b (d)) |
| `AG3S-commonto-gt2-termsoff-pass` | 둘 다 |

test: `tests/benchmark/test_sl_gtto3_versions.py` (argv · config · 기록 읽기, gt2 argv 불변).

## 2. 장면 · 지표

- LIBERO spatial task 0–4, 6–9 × init 0, seed 7, conversion C (SL-GTTO 와 같은 9 장면, 같은 기계 VLA 단독 9/9).
- 주 지표: 공식 성공 (9 중). 진단: episode 별 포화 call 비율(`saturated_rows` ≥ 5 인 call / 전체 call), HOLD 비율, attach 여부.

## 3. 판정 (결과 전 고정)

1. **고르기:** gt2 · termsoff · pass · termsoff-pass 중 LIBERO 공식 성공이 가장 높은 판. 최고와 1 episode 이하 차이는 "비슷"으로 보고, 그중 episode 평균 포화 call 비율이 가장 낮은 판.
2. **넓히기:** 고른 판이 gt2 가 아니고 gt2 보다 **2 episode 이상** 높으면, 그 판으로 SafeLIBERO 11 · LIBERO-Safety 10 을 돌려 SL-GTTO §5 규칙 1–3 전체를 판정한다.
   그 미만이면 TrajOpt 두 변경은 GT 판의 실패를 설명하지 못한다고 기록하고, 남은 실패를 층별로 분류만 한다(새 수정은 사용자 판정 뒤).
3. SL-GTTO §5 규칙 그대로 (VLA 단독 기준은 같은 기계 run 재사용). 판 이름이 다르므로 공통 TO 정의를 바꾸는 것은 **사용자 판정 대상**으로 남긴다 — 이 단계의 결과는 "어느 TO 설정에서 GT 판이 §5 를 통과하나"의 근거다.

## 4. 순서 · GPU

queue `outputs/e1a/queue.txt`: gt2 본 실행(30 장면, SOLO) 바로 뒤에 LIBERO 9 × 3 판 (SOLO, 각 약 20 분). 규칙 2 가 성립하면 그 판의 SafeLIBERO · LIBERO-Safety 줄을 추가한다.

## 5. lead 예측 (2026-11-05 04:1x KST, 결과 전에 적음 — 판정 규칙이 아니라 사후 대조용)

gt2 smoke 기록을 더 읽은 결과 ④는 두 갈래다.

| 갈래 | 근거 | 예측 |
|---|---|---|
| ④a 쥐기 전(latched) 경로 수정 | 이 구간은 연속성 항이 이미 꺼져 있다 (`to.continuity_reason = disabled … before_grasp`). dev > 0 call은 HOLD가 아니면 `budget_only`, HOLD면 `collision` (`answer_excludes_target` 손 ↔ 서랍 · 캐비닛 −33 – −40 mm, `unauthorized_link` link6 coarse −75 mm). VLA reference가 target이 아닌 실제 기하(용기 벽 · 지지면)의 여유 안으로 들어가고, TO가 그것을 밀어낸다 | termsoff · pass 모두 LIBERO t4e0 · t9e0(서랍 · 캐비닛 그릇)를 바꾸지 못한다 |
| ④b 쥔 뒤(held · placed) 경로 수정 | held 구간에서 연속성 · 꺾임 변화율 항이 켜지고(`continuity_reason = feedback`), `budget_only`(위반 없음)인데 dev 0.07–1.3 · C 포화 | termsoff가 held 구간 포화를 줄이고 t0e0 · t2e0 · t7e0 같은 "들고 놓기 실패"를 일부 되찾는다. pass는 쥔 물체가 지지면 · 접시 여유 안에 있는 call에서는 효과가 없다 |
