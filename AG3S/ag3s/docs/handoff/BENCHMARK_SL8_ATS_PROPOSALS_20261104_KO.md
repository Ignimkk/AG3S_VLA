# benchmark 제안 — SL8 ATS 문제 (target 이 옆 물체와 한 cluster · 옛 형상 남음 · 일부만 지움) 에 대한 development 방법론

> writer: development lead (세션 ag3s-vla-development-13 = 3a) · 작성: 2026-11-04 18:0x KST.
> 사용자 지시 (2026-11-04 17:5x KST, development 세션): benchmark 의 SL8 결과에 대한 방법론 제안을 진행한다. benchmark 사용자 판정 (SL8b.task.md 머리말 2): "development 세션이 방법론 몇 가지를 제안한다. 그것을 적용한 뒤 테스트를 진행한다."
> 받는 쪽: benchmark checkout 세션 (ag3s-vla-benchmark-17). **제안일 뿐이다** — 적용 여부 · 사전 기준 · 판 이름은 benchmark 세션이 사용자 판정으로 정하고 결과 전에 고정한다. 모두 기본 꺼짐 flag 로.
> 근거 자료: benchmark `outputs/2026-11-04_SL8_ATS_target장애물분할/README.md` (이하 SL8), `AG3S/ag3s/docs/handoff/SL8b.task.md` (고정 풀기, 이 문서와 별개로 진행 중).

## 0. SL8 문제와 이 문서의 범위

| SL8 문제 (README "무엇이 문제인가") | 원인 층 | 이 문서 |
|---|---|---|
| 1. target 이 옆 물체와 한 cluster (접근 구간 44–46 %, oracle 에서도) | ATS 기하 | **P1 · P2 · P3** |
| 2. 첫 고정이 틀리면 안 바뀜 | ATS 식별 | SL8b (benchmark 진행 중) — P3 만 맞물림 |
| 3. 저장된 대체 형상 사용 (83–89 %), 쥔 뒤 옛 자리에 남음 | ATS 식별 | **P4** (쥐기 전 밀림 · 옮김만; 쥐는 중 · 놓은 뒤는 진단 먼저) |
| 4. 맞게 골라도 일부만 지움 (recall 0.29–0.44) | ATS 범위 · LTE | **P5** (진단 먼저) + 이미 보낸 ablation |
| 5. 행동 attention 자체가 약함 | attention | 이번 범위 밖 (SL8: 1–3 때문에 A · B 선택 정답률이 같음) |

## 1. 문제 1 의 원인 — development 코드에서 확인한 것

- `ground_target` (`AG3S/ag3s/stages/target_grounding.py:1841`) 은 seed 에서 `grow_region(eps)` 로 키운 뒤 DBSCAN (`clustering.eps = 0.03`, `min_points 20`) 으로 나눈다. 받침면 점은 `exclude_mask = support_mask` 로 연결에서 뺀다 (`runtime/pipeline.py:2550`).
- 그 받침면은 **장면 전체에서 큰 평면 최대 2 개** 다: `SupportSurfaceConfig.max_planes = 2`, 크기 문턱 `max(min_inliers 500, 5 % × 전체 점)` (`config.py:827–836`, `stages/support_surface.fit_support_surfaces`).
  - 그래서 탁자는 빠지지만, cabinet · stove 윗면 · ramekin 같은 **작은 받침** 은 문턱을 못 넘는다. 받침으로 빠지지 않고 연결에 남는다.
  - 그 결과 그 위에 앉은 그릇과 한 cluster 가 된다.
  - SL8 oracle 의 병합 상대 89 call 중 fixture (cabinet · stove) 가 52 call 이다.
- 맞닿은 obstacle (34 call) 은 받침이 아니라 옆에 붙은 물체다. eps 를 줄여도 맞닿음 (0 mm) 은 풀리지 않는다.
- cluster 가 그리퍼 개도 (78 mm) 보다 넓으면 쥘 수 없다고 판정되어 후보에서 빠진다. 그러면 다른 물체가 target 이 된다 (SL8 문제 1 마지막 줄).

## 2. 제안

### P1 — seed 아래 국소 받침면 (문제 1, fixture 위 물체) · 1 순위

- **원리:** 놓여 있는 물체는 받침면 위에 있다. 그래서 seed 아래에서 받침면을 **국소적으로** 찾고, 그 면과 그 아래 점을 seed 의 연결에서 뺀다. 전역 2 평면 규칙을 바꾸지 않고 더한다.
- **규칙 (제안값, 결과 전에 benchmark 가 고정):**
  1. 창 W: seed 점의 attention 가중 중심에서 수평 반경 0.10 m 안의 점 (로봇 mask · 전역 받침면 제외).
  2. W 안에서 seed attention 가중 중앙 높이 h_s 보다 **10 mm 이상 낮은** 점들에 수평 평면을 RANSAC 으로 맞춘다.
     - 수평 기준: `|n_z| > 0.9`, `_support_plane_z` 와 같은 기준.
     - 맞추는 함수: `stages.support_surface.fit_plane_ransac`. 거리 문턱 8 mm (serving 값). 크기 문턱 30 점 (T97 `PROTRUSION_PLANE_MIN_INLIERS` 와 같은 국소 문턱).
  3. 그 면이 있으면, 면 inlier 와 면 아래 점 (`n · p − d < threshold`) 을 이번 call 의 `exclude_mask` 에 더한다. 이 점들은 연결과 DBSCAN 에서만 빠지고, 받침면 후보로는 그대로 남는다 (지금 support 처리와 같음).
  4. 지키는 조건: seed attention 질량의 30 % 넘게가 면 inlier 위에 있으면 적용하지 않는다. target 자신이 평평한 물체 (접시 · 책) 일 수 있기 때문이다.
- **기록:** 국소 면 높이 · inlier 수 · 뺀 점 수 · 지키는 조건 발동 여부.
- **위험:**
  - 받침 가장자리에 걸친 물체는 면 아래로 내려간 부분이 빠진다.
  - 그릇 안 · 바구니 안 물체는 면이 그릇 바닥이 될 수 있다. 정답 기준으로 따로 센다.
- **RB-Y1 근거 없음.** 새 규칙이다. development 장면은 탁자 위라 전역 평면으로 충분했다.

### P2 — 오목 꺾임으로 cluster 연결 끊기 (문제 1, 맞닿은 obstacle) · 2 순위

- **원리:** 맞닿은 두 물체의 경계는 표면이 안쪽으로 꺾이는 **오목한 선** 이다. 한 물체의 겉면은 대부분 볼록하거나 매끄럽다. development T78d · T97 의 꺾임 끊기를 attention-surface 그래프가 아니라 `ground_target` 의 점 연결에 쓴다.
- **규칙 (development `runtime/attention_surface_crease.py` 의 함수를 그대로 다시 씀):**
  1. 법선: `normals_pca(pos, r = max(2 · voxel, 7.5 mm))`.
  2. 방향: `orient` 로 관측 카메라 쪽을 향하게.
  3. 끊는 연결: `cut_edges` 로 판정. 두 점 연결 (i, j) 이 오목하고 `(n_i − n_j) · (x_i − x_j) < 0`, 법선 각 ≥ 20° 이면 끊는다.
  4. 연결 범위: 길이 ≤ 10 mm 인 점 연결만 판정한다. eps 30 mm 연결은 꺾임을 건너뛰므로, P2 를 켜면 그 call 의 grow · DBSCAN 반경을 `min(eps, 10 mm)` 로 둔다. 이 반경 변경은 판 정의에 적는다.
  5. 끊은 뒤 연결 성분이 cluster 다.
  6. 손잡이 · 꼭지처럼 몸통에 오목하게 붙은 작은 조각은 T97 돌출 규칙 (`protrude`, ρ 0.15, 받침면 위로 솟은 것만) 으로 다시 붙인다.
- **RB-Y1 근거 (T97, 영역 판정, 쥐기 전 304 요청):**
  - 맞닿은 장면 target 정밀도 중앙: 0.63 → 1.00.
  - 떨어진 장면 덮음 손실: 1.0 %p.
  - 다른 물체를 붙인 node: 0 (돌출 판).
  - 손 층 지우기에서는 남은 voxel 이 늘어 보류됐다 (D32). 그것은 **범위 (지우기)** 쪽 문제이고, 여기서 쓰는 **식별 (어느 cluster 가 target)** 과는 다르다.
- **위험:**
  - LIBERO depth 해상도 · 잡음에서 법선이 흔들린다. 5 mm voxel 로 내려 받은 뒤 법선을 잰다.
  - 그릇 안쪽 면은 오목하다. 그래서 각 문턱 20° 아래의 매끄러운 오목은 끊지 않는다. 그릇이 쪼개지는 비율을 따로 센다 (§3 기준 2).

### P3 — 넓은 cluster 를 이유로 다른 물체로 넘기지 않음 (문제 1 → 2 의 연결)

- **지금:** seed 질량을 가진 cluster 가 너무 넓어 (병합) 빠지면 다음 후보 (다른 물체) 가 target 이 된다.
- **제안:** seed attention 질량 ≥ 50 % 를 가진 cluster 가 너비 때문에 빠지는 call 에서는 다른 cluster 로 넘기지 않는다. 대신 아래처럼 한다.
  - (a) P1 · P2 가 켜져 있으면 그 cluster 에만 다시 적용해 나눈다.
  - (b) 그래도 못 나누면 그 call 은 "target 없음 (merged)" 으로 둔다. 아무것도 지우지 않는다.
- SL8b 의 `attention_release_none` (seed 질량 50 % 조건) 과 같은 질량 규칙이다. benchmark 가 SL8b 구현에 합칠지 따로 둘지 정한다.
- 기대 효과: 틀린 물체 고정 (문제 2 의 출발점) 을 줄인다. 다만 target 을 바로 맞히지는 못한다 (나누기는 P1 · P2 의 일).

### P4 — 옛 자리가 비면 놓아주기 (문제 3) — development T81 을 Panda 로

- **기능:** development `--target-vacancy-release` (T81, 기본 꺼짐).
  - 쥐기 전 (탐색 · 다가감 · 닫는 중, 아무것도 부착 안 됨) 에만 작동한다. **놓은 뒤 (PLACED) 에는 작동하지 않는다** (`config.py` T81 주석). 유지 중인 형상 점을 카메라에 투영한다.
  - 측정 depth 로 비어 있음 / 가려짐 / 있음을 가른다.
  - 비어 있음이 이어지면 놓아주고 다시 고른다.
  - 지금 Panda 서버 UNPORTED 목록에 있다 (`target_vacancy_*`).
- **RB-Y1 근거 (T81 · T82 r2, 2026-11-01):**
  - 사과를 60 mm 옮긴 2 run: 새 사과로 가기까지 1 요청 (끔 48–49).
  - 제자리인데 잘못 놓은 경우: 0.
  - 회귀 · 가림 4 run 의 놓기: 0.
  - 사용자 판정: 기능 후보 (최종 미편입). development 우선순위 5 와 같은 기능이다.
- **Panda 에서 정할 것:**
  - 문턱 (δ 20 mm, p_empty · p_present · 연속 수) 을 Panda depth 로 다시 측정한다. T81 §4 처럼 측정 → 고정 → closed-loop 순서다.
  - 카메라: head · left_wrist 만. right_wrist dummy 는 빼야 한다.
  - 로봇 pixel mask: Panda FK 구.
- 다루는 경우: 접근 중 target 이 밀리거나 옮겨졌는데 옛 형상 (`unobserved` · `subset_kept_anchor`) 을 계속 쓰는 경우.
- 다루지 않는 경우:
  - **쥐는 중** "형상이 손을 안 따라감" 은 offline 재생에 쥔 물체 부착 (HSR6) 이 없어서일 수 있다 (SL8 한계). pc판은 이미 `--kept-geometry-follow` 를 쓴다. 그래서 closed-loop 기록에서 먼저 같은 지표를 재는 것을 제안한다.
  - **놓은 뒤** 의 낮은 정답률 (A 0.17) 은 T81 범위 밖이다. 원인부터 갈라야 한다 (놓은 물체를 목적지 위 cluster 로 다시 잡는지, 옛 형상인지).

### P5 — 일부만 지움 (문제 4) — 원인 가르기 먼저

- **pc판 범위는 공 방식** (`--target-exclusion ball-owned --owned-max-dist-voxels 3`) 이다. 이미 보낸 ablation 의 부호 수정 · 닫힌 물체는 표면 범위 (`--target-extent attention-surface`) 에서만 걸린다. 그래서 pc판 위에는 효과가 없다. `-pc-surface` · `-pc-surface-fb` 판에서만 읽는다.
- **제안 진단 (offline, SL8 재생 그대로):** 정답 call 에서 cluster 에 못 들어간 target 표면 5 mm voxel 을 다섯 가지로 나눈다. development T84 `causes()` 와 같은 방식이다.
  - (i) 전역 · 국소 받침면으로 빠짐
  - (ii) 다른 cluster 로 감
  - (iii) DBSCAN noise · min_points 미달
  - (iv) depth 점 없음 (안 보임)
  - (v) 로봇 mask
- 큰 원인이 (i) · (ii) 면 P1 · P2 로, (iv) 면 범위 규칙 (표면 범위 + 닫힌 물체 · 부호 수정) 으로 간다. 결과를 보고 고친다.

### 참고 진단 (방법 아님) — eps 쓸기

- `clustering.eps` 15 · 20 · 30 mm 를 SL8 재생으로 쓸어 둘을 잰다: 병합 비율과 target 이 둘 이상으로 쪼개지는 비율.
- 맞닿음 (0 mm) 은 eps 로 안 풀리므로 방법으로 채택하지 않는다. P2 의 10 mm 반경이 쪼개짐을 얼마나 만드는지 보는 기준선으로만 쓴다.

## 3. 시험 제안 (benchmark 가 결과 전에 고정)

- **offline:** SL8 과 같은 20 episode · 같은 채점 · 조건 A (행동 attention) 와 C (oracle).
  - 판: pc판 + P1, + P2, + P1 + P2, + P1 + P2 + P3.
  - 표면 판 (`-pc-surface`) 에도 P1 + P2 를 한 번.
- **기준 제안:**
  1. 접근 구간 "다른 물체 cluster 에 병합": oracle 0.461 → ≤ 0.20.
  2. target 쪼개짐 (target 점의 70 % 이상을 가진 cluster 가 없음): ≤ 0.10.
  3. obstacle 잘못 지움: ≤ 0.5 % (SL8b 기준 4 와 같음).
  4. 첫 닫힘 전 선택 정답률: SL8 A 0.33 이상.
- 통과 판을 SL8b 에서 고른 판과 합쳐 closed-loop 으로 간다 (SL8b §4: LIBERO 9 · SafeLIBERO 11).
- 순서 제안:
  1. P5 진단과 eps 쓸기 (코드 변경 없음, 가장 쌈)
  2. P1
  3. P2
  4. P3
  5. P4 (문턱 측정이 따로 필요)

## 4. development 쪽 자료

- 꺾임 끊기 코드 · test:
  - `AG3S/ag3s/runtime/attention_surface_crease.py`
  - `tests/ag3s/test_t97_crease_protrusion.py`
  - `tests/ag3s/test_t78d_crease_cut.py`
- 옛 자리 비어 있음 코드 · test:
  - `stages/target_vacancy.py`
  - `TargetConfirm._vacancy_step`
  - `AG3S/ag3s/docs/handoff/T81.task.md` · `T81.impl.md`
  - 측정 도구 `experiments/studies/t81_vacancy.py`
  - `tests/ag3s/test_t81_vacancy.py`
- 모두 main `4ecf8d61` 안에 있다 (benchmark 가 시험 merge 중인 것).
- development 는 Panda 판 구현을 하지 않는다. 질문 · RB-Y1 값이 필요하면 이 세션에 보낸다.
