# development → benchmark: ATS target · 장애물 분할 개선 제안 (SL8 근거) — 2026-11-04

> writer: development lead (분석 세션, ag3s-vla-development) · 작성: 2026-11-04 17:16 KST (`date -u` + 9 h).
> 받는 쪽: benchmark checkout 세션 (benchmark-17, `feature/panda-benchmark`).
> 사용자 지시 (benchmark-17 전달, 2026-11-04 15:1x KST): "development 세션이 방법론 몇 가지를 제안하고 benchmark 가 적용 후 테스트한다."
> **역할:** 이 문서는 제안과 자료 위치만 적는다. 적용 · Panda 값 · 사전 기준 · 판정은 benchmark 세션이 정하고 사용자에게 직접 보고한다. 모든 제안은 **기본 꺼짐 flag + Panda 판** 전제다.
> 근거: benchmark `outputs/2026-11-04_SL8_ATS_target장애물분할/README.md`, `SL8b.task.md` (commit 4957327d). development 수치는 각 항목의 출처 그대로다.

> **상태 (2026-11-04 17:5x KST, benchmark-17 전달):** benchmark 가 이 문서와 development-13 의 제안 (`BENCHMARK_SL8_ATS_PROPOSALS`) 을 합쳐 사용자 판정으로 순서를 정했다 — 0 진단 (eps 쓸기 · 일부 지움 원인 · task 3 admissibility) → 국소 받침면 (13 P1 = 이 문서 0) → 꺾임 끊기 (13 P2 = 이 문서 1) → 넓은 cluster 재절단 · target 없음 (13 P3 = 이 문서 1′) → T81 결합 (이 문서 2). 이 문서 §2 의 순서는 그 판정으로 **대체됐다**. 제안 4 는 development-13 의 A×B ablation 에서 본다. 구현은 benchmark worktree, closed-loop ablation 과 병행.

## 0. SL8 의 다섯 문제를 development 경험으로 읽기

| SL8 문제 (Panda, offline 20 episode) | RB-Y1 에서 같은 증상 | development 가 쓴 것 | RB-Y1 결과 | → 제안 |
|---|---|---|---|---|
| ① target 이 fixture · obstacle 과 한 DBSCAN cluster (접근 구간 44–46 %, oracle attention 에서도 같음). 합쳐진 cluster 는 다수가 다른 물체이거나 그리퍼 폭 (78 mm) 초과로 "쥘 수 없음" → 다른 물체가 고정됨 | 회귀 211 · 954 의 사과에 붙은 바나나. 공 방식 · attention 분할 (12 mm 재군집) 로 못 풀었음 | depth 연속 표면 graph (T78) + 오목 꺾임 끊기 (T78d) + 위로 솟은 돌출 보존 (T97) — **경계는 기하로만** | 맞닿은 장면 영역 정밀도 0.63 → 1.00, 떨어진 장면 덮음 손실 1.0 %p (영역 판정 통과, `DECISIONS_20261104_3a_KO.md` D30). 손 층 재생은 떨어진 장면 잔여 +2 %p 로 미통과 (D32) | **1** |
| ② 첫 고정 물체가 끝까지 안 바뀜 (A 10/20 episode 오고정, target 으로 switch 0) | 회귀 211 밀린 사과: target 이 옛 자리에 48 요청 잔류 (T75), 옮긴 사과 (T80) | "옛 위치 비어 있음" 규칙 `--target-vacancy-release` (T81) | 재선정까지 48 → 1 요청, 잘못 놓음 0 (6 run). 기능 후보 (기본 꺼짐, 사용자 판정 대기) | **2** (SL8b 보완) |
| ③ 이번 frame cluster 대신 저장 형상 사용 83–89 %, 쥐는 중 형상이 손을 안 따라감 (call 짝의 45–52 % 에서 0 mm) | IFG-211: anchor 가 요청 10 형상으로 고정돼 사과가 36 mm 밀려도 안 따라감 | `--kept-geometry-follow` (IFG-FOLLOW), 쥔 뒤 `--held-body-cover`, 옛 자리 잔상 `--roi-carve` (DA1) | 닫는 중 중심 오차 36 → 2 mm (재맞춤). roi-carve: 옛 자리 잔상 −54 %, 성공 14 → 15/18 | **3** |
| ④ 맞게 골라도 거리장 지움 recall 0.29–0.44 (cluster 가 target 5 mm voxel 의 11–19 %) | 공 방식 손 층 사과 잔여 13–24 % | P0m 표면 스택 (T78 · hull 채우기 · 접촉 여유) + 지운 seed 부호 수정 (T96e) | T92b: hull · 접촉 여유 포함 17/18 대 제외 11/18. T96e: 손 층 잔여 11.5 → 7.8 % (재생), 허들 12 run 9/12 (P0m 7/12) — 채택 대기 | **4** |
| ⑤ attention 자체 약함 (3D 질량 중 target 위: 행동 0.11 · 명사 0.24 · oracle 0.45) | T47 · T49 · T51 · T61 (head 선택 절차), 명사 `L10H7` 유지 | 결과 전 고정 선택 절차 (T61: dev split, leave-one-scene-out, 3D `identity_correct`) | 명사 유지 (T65 3-way) | **5** |
| (+) task 3 (cabinet 위 그릇) 선택 정답률 0 — seed 는 target 위인데 stove 위 다른 그릇 선택 | 허들 막대를 target 후보에서 거르려고 넣은 규칙이 있음 | `--supported-max-bottom-mm 60` (T43: 가장 낮은 점이 받침면 위 60 mm 넘으면 `not_supported`) + 받침면 RANSAC `max_planes = 2` | RB-Y1 탁자 장면에서만 검증됨 | **0** (가장 싼 점검) |

## 1. 제안 (점검 비용이 낮은 순)

### 제안 0 — 받침면 규칙 점검 (코드 변경 전, SL8 기록만으로)

- **가설:** cabinet 위 그릇은 가장 낮은 점이 탁자 평면에서 60 mm 넘게 떠 있어 `not_supported` 로 제외되고 (또는 RANSAC 2 평면이 cabinet 윗면을 못 잡아) stove 위 그릇이 대신 고정된다. SL8 그림 1-(4) 와 맞는다.
- **확인:** SL8 재생 기록에서 task 3 target cluster 의 admissibility `reason` (`not_supported` / `not_graspable` / `destination_overlap`) 과, 그 frame 의 받침면 평면 수 · 높이. 한 시간 안에 끝나는 점검이다.
- **맞으면 선택지:** (a) 받침면을 "cluster 발자국 아래 가장 가까운 수평면" 으로 (2 평면 RANSAC 결과 중 고르거나 `support_surface.max_planes` 를 늘림), (b) Panda 에서 `--supported-max-bottom-mm` 를 빼고 허들 막대 거르기는 다른 조건 (얇음 · 길이) 으로.
- **판정 제안:** task 3 선택 정답률 0 → > 0, 다른 task 정답률 비악화, obstacle 잘못 지움 ≤ 0.5 %.

### 제안 1 — identity 단계의 cluster 를 DBSCAN 대신 depth 연속 표면 성분으로 (+ 오목 꺾임 끊기 · 돌출 보존)

- **왜:** 병합은 attention 과 무관하다 (oracle 에서도 46 %). DBSCAN `eps = 30 mm` 는 RB-Y1 값이고 (`AG3S/ag3s/config.py` `ClusteringConfig.eps = 0.03`, `AG3S_RULES_REGISTRY` ATS-05), Panda 에서 target–obstacle 거리가 30 mm 미만인 call 이 55 % 다. 그리퍼 폭 규칙 (`stages/admissibility.py`: 가장 좁은 주축 폭 ≤ 최대 개도) 은 합쳐진 cluster 를 **버려서** 다른 물체를 고정시킨다 — 잘라야 할 것을 거부하고 있다.
- **development 에 있는 것 (모두 기본 꺼짐 flag):** `--target-extent attention-surface` (T78: 5 mm node, 이웃 pixel 의 depth 차 > 10 mm 면 edge 없음), `--surface-crease-cut --surface-crease-cut-voxel-mm 2.5 --surface-crease-cut-angle-deg 20` (T78d: 오목 20° 이상 edge 끊기), `--surface-crease-protrusion --surface-crease-protrusion-area 0.15` (T97: 선택 영역 꼭대기보다 위로 솟고 투영 면적 ≤ ρ 배이며 몸통 위에 얹힌 조각만 다시 붙임), `--surface-crease-hand-freeze-mm 80` (T97: 손이 가까우면 손이 멀 때 고른 영역 유지). 코드 `AG3S/ag3s/runtime/attention_surface.py` (T78 · T78d · T97 블록), `attention_surface_crease.py`.
- **지금의 한계:** 이것들은 **범위 (extent) 만** 바꾼다. 어느 물체가 target 인가 (identity) 는 여전히 DBSCAN cluster 로 정한다 (`doc/active/reviews/AG3S_decision/METHOD_STATUS_KO.md` §2.1 의 4–7 단계). 그래서 SL8 의 병합은 표면 스택을 켜도 identity 에서 먼저 막힌다 (`--surface-target-check overlap` 은 영역이 ATS cluster 와 겹쳐야 채택).
- **제안하는 구현 (새 flag, 예: `--identity-from-surface`):** 표면 graph 의 연결 성분 (꺾임 끊기 · 돌출 보존 뒤) 을 cluster 집합으로 admissibility · `TargetConfirm` 에 넘긴다. 점수 규칙 (attention 질량 · compactness) 은 그대로. 떨어진 물체 장면에서는 DBSCAN 결과와 같은 cluster 가 나와야 한다 (test 로 고정). 구현 주체는 benchmark 가 정한다 — development 가 기본 꺼짐으로 만들어 넘길 수도 있다.
- **더 싼 대안 (1′):** DBSCAN 은 그대로 두고, cluster 가 그리퍼 폭을 넘거나 다수 class 가 아닐 때 **거부 대신** 그 cluster 점들을 표면 graph 로 다시 잘라 seed 질량이 가장 큰 성분을 후보로 낸다.
- **Panda 값:** node 5 mm · jump 10 mm 는 depth 해상도 기준이라 그대로 시작. 꺾임 각 20° · ρ 0.15 는 RB-Y1 사과 (꼭지 · 잎) 로 고른 값 → dev split 에서 다시 고른다. 그리퍼 폭은 로봇 모델에서 계산된다 (숫자를 쓰지 않는다).
- **주의:** 그릇이 ramekin · cabinet 위에 얹힌 접점은 오목 꺾임이라 끊긴다. 돌출 보존은 "선택 영역보다 위로 솟은 **작은** 조각" 만 붙이므로, 선택 = 그릇이면 cabinet (큼) 은 안 붙는다. 반대로 seed 가 cabinet 쪽에 더 많으면 cabinet 이 선택된다 — 제안 0 · 5 와 같이 본다.
- **판정 제안 (offline SL8 재생, 같은 채점):** 접근 구간 "다른 물체 cluster 에 병합" 0.44 → ≤ 0.15 (oracle 조건에서도), 첫 닫힘 전 선택 정답률 0.33 → ≥ 0.5, obstacle 잘못 지움 ≤ 0.5 %, 떨어진 물체 장면의 cluster 변화 0.

### 제안 2 — 첫 고정 풀기: SL8b 에 "옛 위치 비어 있음" 기하 규칙을 더함

- SL8b (창 투표 교체 + 누적 attention 해제) 의 방향에 동의한다. development 에서 같은 자리에 쓴 것은 **`--target-vacancy-release`** (T81) 다: 쥐기 전, 마지막으로 보인 target 점들을 head · 손목 depth 에 투영해 "비어 있음 ≥ 0.45 ∧ 있음 ≤ 0.30" 이 연속 2 요청이면 target 을 놓고 옛 중심 200 mm 안에서 다시 고른다 (δ 20 mm, 판정 점 ≥ 20, 표본 ≤ 200 — 측정 뒤 · closed-loop 전 고정). RB-Y1 옮긴 사과 장면: 새 사과로 가기까지 48–49 → 1 요청, 잘못 놓음 0 (6 run, `outputs/2026-11-01_T81T82_r2_옛위치비어있음_HSR5단계_closedloop/`). 코드 `AG3S/ag3s/stages/target_grounding.py`.
- 두 규칙은 trigger 가 다르다 — SL8b 는 "attention 이 다른 곳을 본다", T81 은 "그 자리가 비었다". 둘 다 참일 때가 가장 확실한 해제 조건이고, 하나만 참일 때 (가림 · attention 흔들림) 는 유지하는 쪽이 안전하다. 쥐기 전에만, `--latched-identity-hold` 는 그대로.
- **주의 (SL8b §1 (b) 와 같음):** 병합이 풀리기 전에는 옮겨 갈 target cluster 가 없다. 그래서 제안 1 뒤에 다시 잰다.
- **판정 제안:** SL8b §3 규칙 그대로 두고, T81 을 더한 판을 arm 으로 추가 (회복 수 · 잘못된 교체 ≤ 1).

### 제안 3 — 저장 형상 (anchor) 사용을 묶기

- 접근 구간 `subset_kept_anchor` 74 %: RB-Y1 IFG-211 에서 anchor 가 요청 10 의 형상으로 고정돼 사과가 밀려도 안 따라갔다 → **`--kept-geometry-follow`** (고정 형상의 표면 구를 이번 frame 점에 반지름 고정 Gauss-Newton 으로 재맞춤; 점 ≥ 20 · 잔차 ≤ 4 mm · 이동 ≤ 30 mm). 닫는 중 중심 오차 36.01 → 2.05 mm (중앙). P0m 에 들어 있다. **benchmark pc 판에 이 flag 가 있는지 먼저 확인.**
- 쥐는 중 0 mm: closed-loop 에서는 latch `held` 뒤 `--held-body-cover` (쥔 물체가 손에 붙음) 와 `--roi-carve` (옛 자리의 보이는 빈 공간을 지움) 가 맡는다. **SL8 offline 재생에는 둘 다 없다** (README 한계) → 이 항목은 closed-loop 기록으로 다시 세어야 결론이 난다.
- **제안:** (a) anchor 유지 조건에 T81 의 vacancy 검사를 붙인다 — anchor 점 투영이 비어 있으면 anchor 를 버린다 (T81 코드 재사용). (b) `subset_kept_anchor` 판정에 재맞춤 잔차 조건 (잔차 > 4 mm 면 subset 아님).
- **판정 제안:** 접근 구간 "대체 형상 비율" 0.74 → ≤ 0.4, 선택 정답률 비악화.

### 제안 4 — 범위 · 거리장: P0m 표면 스택을 통째로 + 지운 seed 부호 수정

- **flag 묶음 (P0_FINAL_CONFIG §1.1):** `--target-extent attention-surface --surface-target-check overlap --surface-keep-last --surface-grasp-hold --surface-no-ball --attention-surface-device cuda --surface-hull-fill --surface-contact-allowance-mm 10` (`--nontarget-evidence surface` 는 같이 쓰지 않음). RB-Y1 T92b (seed 2 종): hull 채우기 · 접촉 여유 포함 17/18 대 제외 11/18 → 2026-11-04 확정.
- **+ `--surface-removed-sign-free`** (T96e, 기본 꺼짐, **채택 대기**): 지운 seed 가 TSDF 안쪽 음수로 남는 문제 (부호 교정 순도 판정 `|d| > 1.5 voxel`) 수정. 손 층 잔여 11.5 → 7.8 % (재생), 허들 12 run 9/12 (P0m 7/12, 금지 2 → 1), 9 run 9/9 (같음). 차 ≤ 2 run 이라 development 에서도 아직 확정 전이다 (`DECISIONS_20261104_3a_KO.md` D40–D43). 코드 `AG3S/ag3s/fields/curobo_builder.py` (T96e 블록).
- **의존:** `--surface-target-check overlap` 은 영역 pixel 가중의 50 % 이상이 ATS target 점 10 mm 안에 있어야 채택한다. identity (제안 1 · 0) 가 틀리면 표면 영역도 버려진다. **순서는 0 · 1 → 4.**
- **benchmark 주의 (D1):** `--surface-contact-allowance-mm 10` 은 target 표본 10 mm 안의 장애물 voxel 을 손 층에서 지운다. SafeLIBERO Level I (장애물이 target 옆) 에서는 충돌 위험이 있다 → 켠 판 · 끈 판을 둘 다 잰다. hull 채우기는 영역 convex hull 안만이라 위험이 덜하다. 쥐는 순간 이웃 허용 두 flag (`--neighbour-contact-mm 20` · `--neighbour-palm-contact-mm 10`) 는 D1 대로 benchmark 에서 끈다.
- **Panda 값:** z_floor guard (RANSAC 평면 + 미세 voxel 1 개 = 5 mm) · τ 5 mm · hull pad 5 mm 는 depth · voxel 기준이라 그대로 시작. 손바닥 창 (hull 80 mm · 이웃 허용 150 mm) 은 Panda 손 크기로 다시 정한다.
- **판정 제안:** 정답 call 의 거리장 지움 recall 0.29–0.44 → ≥ 0.7, obstacle 잘못 지움 ≤ 0.5 %, 쥐기 창 HOLD 비증가.

### 제안 5 — attention 원천은 0 · 1 뒤에 다시 고른다

- SL8 에서 명사 attention (B) 은 행동 attention (A) 보다 target 질량이 2 배 (0.24 대 0.11) 이고 "어느 cluster 에도 없음" 이 절반 (0.093 대 0.187) 인데, 선택 정답률은 같다 (0.199 대 0.198). 병목이 병합 · 받침면이라는 뜻이다. 제안 0 · 1 뒤 SL8 재생을 A · B 로 다시 돌리면 둘이 갈릴 가능성이 크다.
- 고를 때는 T61 의 결과 전 고정 절차 (dev split · 후보 격자 · leave-one-scene-out · 3D `identity_correct`, `AG3S/ag3s/docs/handoff/T61.task.md`) 를 Panda 기록에 그대로 쓰기를 권한다. hidden patch probe (T86 · T90) 는 Panda 재학습이 필요하고 development 에서도 보류 중이라 지금은 제안하지 않는다.

## 2. 시험 순서 제안 (benchmark 가 정한다)

1. offline SL8 재생으로 0 → 1 → 2 → 3 → 4 를 각각 **짝 비교** (pc 판 대 pc 판 + 기능 하나). 같은 채점 · 같은 20 episode. 사전 기준은 위 "판정 제안" 을 출발점으로 benchmark 가 결과 전에 고정한다.
2. 통과한 판만 closed-loop: SL7 과 같은 장면 (LIBERO 9 + SafeLIBERO 11), 같은 seed, 접촉은 벤치마크 정의 (Level I 은 target 옆 장애물 접촉 = 충돌).
3. 10 run 차 ≤ 2 는 "구분 못 함" (development 관례).

## 3. 자료 위치 (development checkout `/mnt/dev/AG3S_VLA-development`)

| 무엇 | 경로 |
|---|---|
| 표면 추출 · 꺾임 끊기 · 돌출 보존 | `AG3S/ag3s/runtime/attention_surface.py`, `attention_surface_gpu.py`, `attention_surface_crease.py`, `attention_surface_post.py`; task `handoff/T78.task.md` · `T78d.task.md` · `T97.task.md` · `T97.impl.md` |
| identity (DBSCAN · TargetConfirm · T81 vacancy) | `AG3S/ag3s/stages/target_grounding.py`, `stages/admissibility.py` (받침면 · 그리퍼 폭 규칙), `config.py` (`ClusteringConfig.eps`, `supported_max_bottom`, `support_surface.max_planes`); task `handoff/T81.task.md` |
| 손 층 · 부호 수정 · ROI carve | `AG3S/ag3s/fields/curobo_builder.py` (T96e), `runtime/pipeline.py`; task `handoff/T96.task.md` (T96e 절은 `T96.impl.md`), `AG3S-TASK-DYNAMIC-MAP.task.md` |
| 방법 설명 (identity 와 extent 가 갈리는 지점) | `doc/active/reviews/AG3S_decision/METHOD_STATUS_KO.md` §2.1 · §7 |
| 규칙 값과 그 근거 (RB-Y1 값 표시) | `AG3S/ag3s/docs/AG3S_RULES_REGISTRY.md` (ATS-05 eps, ATS-10 RANSAC, T43 supported 규칙, LTE-16 guard) |
| 결과 폴더 | `outputs/2026-11-04_T96_T97_target닫힌부피_맞닿은경계_재생/`, `outputs/2026-11-04_T96e_부호수정_closedloop/`, `outputs/2026-11-01_T81T82_r2_옛위치비어있음_HSR5단계_closedloop/`, `outputs/2026-11-03_T92b_P0m확정_seed반복_hull접촉여유짝비교/` |
| 판정 기록 | `handoff/DECISIONS_20261104_3a_KO.md` (D30 · D32 · D38–D43), `DECISIONS_20261103_KO.md`, `P0_FINAL_CONFIG_FOR_BENCHMARK_20261103_KO.md` |

- **코드 상태 (2026-11-04 갱신):** T78 · T78d · T81 · P0m 스택과 **T96 · T96c · T96d · T96e · T97 코드가 모두 commit 돼 있다** — development `d5a1142e` (코드) · `ff2edfec` (문서), main `4ecf8d61` (merge). benchmark 는 main `4ecf8d61` 을 merge 대상으로 쓴다. (처음 쓴 "미커밋" 문장은 이 commit 전의 상태였다.)
- **development 쪽 상태 요약:** T97 = 영역 판정 통과 · 손 층 미통과 (기본 꺼짐). T81 = 기능 후보 (기본 꺼짐). T96e = 채택 대기 (기본 꺼짐). P0m 표면 스택 = 확정 (2026-11-04).

## 4. 한계

- 모든 근거가 RB-Y1 · 사과 · 탁자 장면이다. 그릇 · cabinet · ramekin · stove 에서는 값 (꺾임 각 · ρ · 창) 을 다시 고른다.
- 제안 1 의 `--identity-from-surface` 는 **설계만 있고 구현이 없다.** 1′ 이 더 싸다.
- SL8 의 쥐는 중 · 놓은 뒤 수치는 offline 재생이라 latch · 쥔 물체 부착 · roi-carve 가 빠져 있다 (제안 3 은 closed-loop 로 다시 세어야 한다).
- 제안 0 의 가설은 SL8 기록으로 먼저 확인해야 한다. 틀리면 제안 0 은 버린다.
