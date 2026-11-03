# SL7d — Panda P0m의 남은 원인 후보 가르기: ATS · 표면 seed 신호 L8H2 / latch label 단계 판정 (LIBERO 9)

> writer: lead (benchmark checkout), 2026-11-04 04:1x KST. **결과를 보기 전에 고정.**
> 사용자 판정 (이 세션, 2026-11-04 04:1x KST, AskUserQuestion): "P0m+L8H2 와 pc판+latch 둘 다", 표면 seed 신호를 고칠 방향은 "다음 LIBERO 결과 본 뒤 결정".
> 이 판정부터 Panda 판정은 development 중계 없이 이 세션에서 받는다(development 전달, 2026-11-04).
> 앞 단계: `SL7c.task.md`, 결과 `outputs/2026-11-04_SL7c_공방식진단_표면어긋남원인/` (공 진단 판 0/9 → target 범위는 원인 아님. 남은 후보는 ATS target 신호 L8H6과 latch label 대 HSR6).

## 1. 판

| 판 | 정의 | 무엇을 가르나 |
|---|---|---|
| (가) `AG3S-final-panda-p0m-l8h2` | P0m에서 `--attention-source noun_target:L8H6`만 `action:last:L8H2`로 바꾼다. 나머지(표면 추출 · hull · 접촉 여유 · latch label · HSR6 끔 · E0 · D1)는 P0m 그대로다. P0m은 표면 seed 지도와 ATS가 같은 attention map을 쓰므로 **둘 다 L8H2**가 된다. Panda 서버가 표면 seed 지도로 `action:last:L8H2`를 받도록 `PANDA_SEED_MAP_LABELS`에 더한다 | ATS target 신호 L8H6이 원인인가. 표면 추출을 유지한 채라 원인 확인과 최종 후보를 겸한다 |
| (나) `AG3S-final-panda-v3-e0to-pc-latch` | pc판(현재 Panda 최종)에 `--subtask-from-latch`만 더한다(HSR6 · probe는 그대로, latch label이 3단계 gate의 label 출처) | latch label 단계 판정이 Panda에서 성공을 깎는가 |

주의: pc판은 `--attention-source`를 주지 않는다(기본 경로 = `action:last:L8H2`). (가)는 같은 cell을 **명시**하므로 T57-A extended 경로(별도 prefix pass)를 탄다. cell은 같지만 계산 경로가 다르다는 점을 결과와 함께 적는다.

## 2. 장면 · 지표

- 장면: SL7과 같은 libero_spatial task 0–4 · 6–9 × init 0, seed 7, conversion C (VLA 단독 9/9).
- 주 지표: 공식 성공(9개 중). 진단 지표: VLA 대비 0.5 넘게 바뀐 call 비율, 실행 행동 ±1 포화 비율(episode 평균, SL7b와 같은 계산).
- 비교: pc판 6/9(포화 10%), P0m 0/9(29%), 공 진단 판 0/9(29.5%).

## 3. 판정 규칙 (결과 전 고정)

1. **(가):** ≥ 5/9 이고 포화 ≤ 15% → 원인을 "ATS · 표면 seed 신호 L8H6"(ATS 층)으로 확정하고, (가)를 SafeLIBERO로 넘길 후보로 사용자께 올린다. ≤ 2/9 → attention 신호만으로는 설명되지 않는다고 적는다. 3–4/9 → 가르지 못함.
2. **(나):** ≤ 3/9 (pc판보다 3 이상 적음) → latch label 단계 판정이 Panda에서 성공을 깎는다고 적는다. ≥ 5/9 → latch label은 원인이 아니라고 적는다. 4/9 → 잡음 범위(규약 §4: 차이 1–2는 잡음).
3. 둘 다 결과가 나오면 SafeLIBERO 전에 보고하고, 표면 seed 신호 방향(카메라별 정규화 / 분리된 seed attention / ATS 점 seed)의 판정을 받는다.

## 4. 이식 · 테스트 (CPU)

- `AG3S/adapters/panda/attention.py` `PANDA_SEED_MAP_LABELS` += `"action:last:L8H2"` (Panda 서버만, RB-Y1 L10H7 규칙 그대로).
- `profile.py`: 두 판 등록. (나)는 HSR6 probe가 필요한 최종 판 목록(`FINAL_CONFIG_VERSIONS`)에 넣는다.
- 테스트 `tests/benchmark/test_sl7_panda_p0m.py`: flag 정의(P0m 대비 · pc판 대비 차이 정확히 하나), Panda 서버 CPU build(seed 지도 L8H2로 `attention_surface`, latch gate on), (나) build에 HSR6 probe와 latch gate가 함께 있음. 회귀: sl7 · merge · sl6a · sl5a golden.

## 5. 순서 · GPU

이식 + 테스트 → commit → (가) · (나) LIBERO closed loop 연속(exclusive, launcher가 lock을 잡음, 합 약 25분) → `outputs/2026-11-04_SL7d_*` → 사용자 보고.
