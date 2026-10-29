# AG3S-DYNAMIC-FIELD — 장애물이 움직여도 거리장이 따라오게 하기 (동적 장애물 추종)

> writer: lead · 실행: lead 직접 (순차 작업)

작성: 2026-10-23 KST, lead. **결과를 보기 전에 고정한다.**
사용자 판정 2026-10-23: "동적 추종을 기여로 삼는다면 맞는 방향 3가지 좋다. AG3S에 적용해 보라."
이 판정은 같은 날 고정한 연구 범위(`doc/active/plans/AG3S_RESEARCH_SCOPE_20261023_KO.md`, 새 기능 중단)에
**별도 연구 축을 하나 더한 것**이다. 핵심 평가(`AG3S-CORE-EVALUATION.task.md`)의 세 구성 비교에는 넣지 않는다.

## 1. 왜 지금 이것인가 (측정된 사실)

- TSDF는 요청마다 카메라 3대 depth를 누적한다. **새 위치는 들어오지만 옛 위치가 잘 비워지지 않는다**
  - 실패 장면 TrajOpt 활성 제약의 44.1%: 시선이 열려 있는데(95%) depth가 그 voxel까지 닿지 않는 자리의 잔여(`IFG-UNKNOWN.verify.json`)
    - TSDF가 관측 표면 근처(truncation 띠)만 갱신하고, 그 앞의 보이는 빈 공간은 비우지 않는다
  - 44.5%: 지금은 없는 물체의 옛 기억이다(그 순간 보이는 비율 6%)
  - cuRobo 감쇠(`esdf.time_decay` · `frustum_decay`, `config.py:1037–1041`)는 기본 꺼짐이다. 감쇠를 켰을 때의 잔상 측정(F20)은 "측정 대기"로 남아 있다(`AG3S_T0T6_LOG.md:1215`)
  - cuRobo 감쇠는 depth를 보지 않는다(`wp_decay.py:170` "The current depth/range images are not inspected"). 카메라 시야 안의 block을 일괄 감쇠하므로, 시야 안에서 손에 가려진 실제 물체도 잊는다
- 쥔 물체 자리를 비우는 장치(`pipeline.py:_held_free_points`, T32 H3)는 쥔 뒤에만 동작한다

## 2. 구현 (기본 꺼짐, 꺼짐이면 출력 · 기록 키 그대로)

| # | 설정 | 무엇을 | 장면 전체인가 |
|---|---|---|---|
| D1 | `--free-space-carve` (`esdf.free_space_carve`) | **보이는 빈 공간 비우기.** 적분 뒤, TSDF의 점유 voxel(부호 음수 또는 표면 띠)을 카메라마다 투영한다. 그 화소의 관측 depth가 voxel 깊이보다 `carve_margin`(기본 = 그 TSDF의 truncation) 넘게 멀면, 광선이 그 voxel을 지나간 것이다. 그 voxel을 free로 만든다. 로봇 mask 화소와 depth 무효 화소는 판단하지 않는다(관측 없음 = 그대로) | 예 |
| D2 | `--robot-occupied-free` (`esdf.robot_occupied_free`) | **로봇 몸이 차지한 공간 비우기.** 이번 프레임 self-filter 로봇 구(전신)를 `robot_free_shrink`(기본 5 mm)만큼 줄인 부피 안의 TSDF voxel을 free로 만든다. 로봇이 지금 그 자리에 있으므로 다른 물체는 거기 없다 | 예 |
| D0 | (비교용) cuRobo 감쇠 flag | `--tsdf-time-decay` · `--tsdf-frustum-decay`(기존 config 연결만) | 예 (depth를 안 봄) |

- D2 안전 규칙(`_held_free_points`와 같다): 지지면 + pad 아래 점은 빼고(탁자 구멍 금지), 목적지(바구니) 점 2 pad 안은 뺀다
- 구 축소량은 손가락이 사과 · 이웃 과일에 닿을 때 그 표면을 지우지 않기 위한 것이다. 이번 기록의 이웃 과일 실제 침범 최대는 2.3 mm다
- 기록: `esdf_stats.dynamic = {carve: {n_candidates, n_freed, per_camera}, robot_free: {n_points, n_freed}}`(켰을 때만)

## 3. 평가 — 같은 입력에서 거리장 품질 (offline field replay, closed-loop 아님)

closed-loop은 1e-10 수준 차이로 궤적이 갈라진다(2026-10-23 WORKLOG). 그래서 **기록된 같은 관측**으로 거리장만 다시 만들어 GT와 비교한다.

- **도구(신규):** `AG3S/ag3s/experiments/studies/dynamic_field_replay.py`
  - 입력: policy record(요청별 depth 3대 · K · T · qpos)를 서버와 같은 AG3S 설정 · 로봇 mask로 builder에 차례로 넣는다
  - GT: 같은 요청의 MuJoCo 상태에서 로봇이 아닌 충돌 geom
- **데이터:** `IFG-APPLE-NBC-TARGETSURFACE` 10 장면 전 요청. 사과가 밀림(211), 이웃 과일이 밀림, 쥔 사과 운반이 포함된다. 정적 대조는 같은 장면의 탁자 · 바구니다
- **구성 5개:** 끔 / D1 / D2 / D1 + D2 / D0(time 0.99, frustum 0.8)
- **충실도 관문:** 끔 replay의 미세 · 거친 main 층이 서버 기록 `esdf_distance`와 같아야 한다(비교 범위와 허용오차는 실행 전 `definitions`에 적는다). 다르면 멈추고 보고한다

| 지표 | 정의 (요청마다, workspace 창 안) | 방향 |
|---|---|---|
| **잔상 voxel** | 거리장이 점유(≤ 0)인데 가장 가까운 GT 표면 · 로봇 구까지 15 mm 넘게 먼 voxel 수. 그중 "그 물체가 예전에 있던 자리"(이전 요청 GT 표면 15 mm 안) = 옛 기억 | 줄어야 한다 |
| **잊은 실제 장애물** | GT 표면 표본점(로봇 · 쥔 물체 제외) 중 거리장이 10 mm 넘게 멀다고 보는 점의 비율. 가려진 점과 보이는 점으로 나눈다 | 늘면 안 된다 |
| **비움 지연** | 10 mm 넘게 움직인 물체마다, 옛 자리 잔상이 0이 될 때까지의 요청 수 | 짧아야 한다 |
| 비용 | builder 시간 증가(ms, 중앙값 · p95) | 기록 |

**미리 정한 판정 (이번에 정한 연구 의사결정 기준이며 기존 측정이나 분야 표준이 아니다)**
- 후보가 끔 대비 **잔상 voxel을 50% 이상 줄이고**, **잊은 실제 장애물 비율이 1 percentage point 넘게 늘지 않으면** "거리장 수준의 동적 추종 개선"으로 본다. 가려진 점은 따로 본다
- 잊은 실제 장애물이 1 pp 넘게 늘면 그 구성은 채택하지 않는다. D0이 그렇게 될 것으로 예상하지만, 결과 전 예상일 뿐이다
- 이 단계에서 closed-loop 성공률은 주장하지 않는다

## 4. 그 다음 (이 문서의 범위 밖, 결과를 보고 사용자 판정)

- 장애물이 움직이는 장면(밀린 이웃 과일 · 옮겨지는 장애물)의 closed-loop 평가 설계. 성공 · 실제 접촉. 반복은 seed 여러 개로 한다(뒤집힘 문제)
- 산출물: `AG3S/ag3s/docs/handoff/AG3S-DYNAMIC-FIELD.verify.json`, 그림(실제 씬 · 잔상 시계열 · 표)
- 보기 쉬운 결과: `outputs/2026-10-23_동적장애물추종_거리장replay/`
