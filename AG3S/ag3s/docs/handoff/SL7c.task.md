# SL7c — Panda P0m의 원인 확인: ① 공 방식 target 범위 진단 판 (LIBERO 9) + ② 표면 영역 150 mm 어긋남 원인 가르기 (오프라인)

> writer: lead (benchmark checkout), 2026-11-04 03:2x KST. **결과를 보기 전에 고정.**
> 사용자 승인 (2026-11-04, development d2 세션 전달 "네"): ①②를 함께 진행. 앞 단계 `SL7b.task.md`,
> 결과 `outputs/2026-11-04_SL7b_P0m_TrajOpt대조_LIBERO_표면기각/` (P0m 0/9, (c) 1/9, (b) 1/9, (d) 1/9, pc판 6/9).

## ① 공 방식 진단 판 `AG3S-final-panda-p0m-ball` (closed loop)

**용도는 원인 확인뿐이다.** 표면 추출은 최종 구성에 반드시 들어가므로 이 판은 결과가 좋아도 **채택 후보가 아니다.**

| 항목 | 정의 |
|---|---|
| 판 | P0m − 표면 추출 flag(`--target-extent attention-surface --surface-target-check overlap --surface-keep-last --surface-grasp-hold --surface-no-ball --attention-surface-device cuda`) − hull 채우기 · 접촉 여유(`--surface-hull-fill --surface-contact-allowance-mm 10`, 둘 다 표면 추출 전용 — config가 다른 범위에서 거절) + `--nontarget-evidence surface`(1판의 비 target 증거). 곧 target 범위 = 1판과 같은 공(`--target-exclusion ball-owned --owned-max-dist-voxels 3` + 비 target 증거) |
| 그대로 둔 것 | P0m의 단계 판정(`--subtask-from-latch`), HSR6 끔, TrajOpt 여러 초기값 끔(E0), D1(이웃 허용 끔), 쥔 뒤 연속성 · 꺾임 변화율, Panda head `noun_target:L8H6` 등 나머지 전부 |
| 장면 | SL7과 같은 libero_spatial task 0–4 · 6–9 × init 0, seed 7, conversion C (VLA 단독 9/9) |
| 비교 | P0m 0/9, (c) 1/9, pc판 6/9 (기존 기록) |
| 주 지표 | 공식 성공(9개 중). 진단: VLA 대비 0.5 넘게 바뀐 call 비율, 실행 행동 ±1 포화 비율 |

**판정 (결과 전 고정):**
1. **≥ 5/9 이고 ±1 포화 ≤ 15%** → 원인을 "Panda에서 표면 추출이 target을 거리장에서 못 지움"(ATS 층)으로 확정한다.
2. **≤ 2/9** → target 범위가 아닌 차이(단계 · 쥠 처리: latch label 대 HSR6, 비 target 증거)를 다음으로 본다.
3. 그 사이(3–4/9, 또는 ≥ 5/9인데 포화 > 15%) → "가르지 못함"으로 적고 ②의 결과와 함께 보고한다.
4. 어느 경우든 **SafeLIBERO closed loop 전에 보고**한다. SafeLIBERO는 표면 추출을 고친 뒤에 시작한다.

## ② 표면 영역이 ATS target 점에서 ~150 mm 어긋나는 원인 (오프라인, GPU 불필요)

- 자료: SL7b 오프라인 재생 `20261104T0130_1f1e296c_sl7b-offline-safelibero-p0m-surface`의 `frames.jsonl`(`surface.why`)에서 `not_ats_target`로 기각된 call을 **서로 다른 episode 3–4개**(task가 겹치지 않게 가능한 만큼)에서 하나씩 고른다(episode 안에서는 기각이 처음 나온 call, 결과 보기 전 규칙).
- 같은 재생(같은 기록 · noun row · 그리퍼 신호)을 그 call까지 다시 돌려 표면 선택 내부 값을 꺼낸다(CPU).
- 그림: 카메라 영상 위 반투명 overlay — attention(L8H6 seed 지도)은 연속 colormap + colorbar, 고른 표면 영역 · ATS target 점은 표식, 정답 target 물체는 MuJoCo segmentation 윤곽선. 카메라 시점마다(head · wrist), 가짜 세 번째 view가 무엇을 내는지도 함께.
- 가를 요인:
  - (a) attention L8H6가 처음부터 target 밖을 보는가 — seed 지도 상위 patch가 정답 target mask 밖인가
  - (b) 가짜 세 번째 view가 seed · 점구름에 섞이는가 — dummy view가 seed나 점을 내는가
  - (c) seed는 target 위인데 표면 graph(5 mm node, jump 10 mm)가 탁자 등으로 번지는가 — seed node는 target mask 안, 고른 영역은 target 밖
  - (d) 해상도 (development 추가 요청, 결과 전 03:3x KST): Panda 렌더 256×256, policy 입력 224 — 같은 frame에서 target 위치의 attention patch 하나(16×16 격자)가 덮는 실제 크기(mm), depth 1 px이 덮는 실제 크기(mm), target 물체가 걸치는 attention 상위 patch 수를 적는다.
- 결과: call마다 원인 층(ATS · LTE 등)과 요인(a/b/c/d)을 표로 적는다. 판정 규칙은 두지 않는다(원인 기록용).

## 순서 · GPU

① 판 등록 + CPU 테스트 → ① LIBERO closed loop(exclusive, 약 20분, launcher가 lock을 잡음) 동안 ② 오프라인(CPU) → 결과를 `outputs/2026-11-04_SL7c_*`에 정리 → 사용자와 development d2에 보고. 채택이나 새 방향은 사용자 판정으로 올린다.
