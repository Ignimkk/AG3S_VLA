# AG3S 최종 구성 (Panda / LIBERO 계열)

현재 시점의 최종 구성과 그 설명만 둔다. 구성이 바뀌면 이 문서를 고치고, 맨 아래 '교체 기록' 에 **무엇 → 무엇, 이유** 를 한 줄로 더한다.
판 정의 정본: `AG3S/adapters/panda/profile.py` (`AG3S_VERSIONS`). flag 정본: `AG3S/trajopt/serve_safe.py`. 판정 근거: `E1_GTTO_DECISIONS_20261104_KO.md`.

## 최종 구성 (2026-11-08 22:2x KST)

**판:** `AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345` — GT 없이 AG3S 자기 인식만 쓴다.
상태: 기제 확인 중 (평가 seed 의 gt24att 실패 task 만). 최종 평가 순서는 보정 seed 10 · 11 → 평가 LIBERO 27 → SafeLIBERO 11 → LIBERO-Safety 10.

참조 상한: 같은 뼈대에 GT 인식을 넣은 **B5** (`…-gt2-…-obst3-pts-nocarve-fh-t100`, 사용자 판정 기준 판) — LIBERO 27/27, SafeLIBERO 보정 11 ep (dev init) 9/11 (실제 충돌 1), LIBERO-Safety 9 (위반 0).

공식 규약 E1a (SafeLIBERO Spatial LI, test init 5–14, n = 40, `main_aegis`; dev-38 G108–G110, `outputs/2026-11-08_E1a_SafeLIBERO_B5/E1a_spatial_LI_표.md`) 의 SSR (안전 성공): B5 (GT) 13 · GT 없는 B5 (gt24att) 5 · π0.5 단독 6 · AEGIS 17 · KNOWS (GT) 21 · **CBF (GT) 23 = 넘어야 할 목표**. 지금 구성의 E1a 수치는 아직 없다.

### 1. 인식 (무엇이 target · 장애물 · 목적지인가)

| 역할 | 구성 | flag |
|---|---|---|
| target identity | VLA chunk 가 손을 닫는 지점 (gripper 명령이 latch 문턱 아래인 첫 row 의 손끝) 아래 물체. 그 점 주변 국소 받침면 (위 향한 법선 띠 중 가장 넓은 것) · 평평한 벽을 빼고 자른다. attach 전이면 즉시 채택, closing 부터 잠근다. 자른 점의 attention 은 손끝에 가까울수록 높다. 그 지점이 없는 call 은 attention grounding | `--close-seed-grounding` (+ `--attention-split-*`) |
| target 받침 판정 | 자른 물체가 서 있던 국소 받침면을 그 episode 동안 받침 test 에 더한다 (cabinet 위 · 서랍 안 그릇) | (위에 포함) |
| 장애물 물체 | 받침면을 뺀 점군의 15 mm 연결 성분 중 target · 그 받침 · 목적지가 아닌 것. 쥔 동안은 손 안 물체 근처 성분도 뺀다 | `--scene-obstacle-objects` · `--scene-obstacle-held-exclude` |
| 목적지 | 쥔 동안에만 인식으로 등록. 손 밖 attention 1 위 cluster 는 잡을 수 있는 크기여도 (접시) 후보 | `--destination-after-held` · `--destination-held-leader` |
| 로봇 · 쥔 물체 pixel | self-filter (GT mask 없음) | — |

### 2. TO (언제 VLA chunk 를 바꾸나)

| 규칙 | 값 | flag |
|---|---|---|
| VLA chunk 를 그대로 통과 | 위반이 없거나 아래 문턱 안이면 | `--passthrough-when-clear` |
| 접촉면 (target · 받침 · 탁자 · 목적지) | 100 mm 넘게 파고들 때만 개입 | `--violation-tolerance-mm 100` |
| 장애물 물체 | 여유 10 mm, chunk 10 step 전체를 본다. 장애물 물체 행 = 가장 가까운 표면이 장애물 물체이고 **기존 분류 (조작 대상 · target 부피 · 받침면 · 목적지 label · 쥔 물체 잔재) 도 장애물이라고 할 때** | `--passthrough-obstacle-tolerance-mm 10` · `--passthrough-obstacle-full-horizon` · `--passthrough-obstacle-class-tests` |
| 쥔 물체 모델 | 관측 점 위 20 mm 껍질 구 | `--held-point-model-mm 20` |
| 손 voxel 비우기 | 끔 (장애물 표면을 지움) | (`nocarve`) |
| 개입 방식 | 정책 action 공간 최소 변경 QP | `--ee-action-qp` |

### 3. 구현 위치 · test

| 무엇 | code | test |
|---|---|---|
| 손 닫는 지점 아래 물체 자르기 | `AG3S/ag3s/stages/close_seed.py` | `tests/ag3s/test_sl_pe1_close_seed.py` |
| 손 닫는 지점 계산 → AG3S | `AG3S/trajopt/safe_policy.py` `_close_seed_request` | `tests/ag3s/test_sl_pe_wiring.py` |
| grounding 입력 · 받침면 유지 · 쥔 물체 제외 | `AG3S/ag3s/runtime/pipeline.py` `_close_seed_grounding_input` · `_scene_obstacle_points(held_points=)` | 같은 파일 |
| 즉시 채택 · closing 잠금 · 목적지 등록 | `AG3S/ag3s/stages/target_grounding.py` `TargetConfirm` | 같은 파일 |
| 장애물 행 분류 | `AG3S/trajopt/linearize.py` `obstacle_object_rows(class_tests=)` | 같은 파일 |

실행: `experiments/segmentation/pe1c_queue.sh` (기제 확인), 평가는 `gtto16_queue.sh` 에 `V=<위 판>` · `EXTRA="--record-constraints full"`.

### 4. 지금까지의 수치 (기제 확인, LIBERO t3 · t4 · t7 · t9)

| | gt24att (교체 전) | **지금 구성 (pe12345)** |
|---|---|---|
| LIBERO t3 · 4 · 7 · 9 × seed 7 · 8 · 9 성공 | 2 / 12 | **8 / 12** |
| SafeLIBERO t3 (seed 7) 성공 · 안전 성공 | 1 · 0 / 4 | 2 · 1 / 4 |
| SafeLIBERO t3 실제 충돌 (새로 생긴 것) | — | **1 (e3, 와인병 296 mm)** |

e3 충돌: 손 닫는 지점이 그릇 옆 와인병에 찍혀 병을 identity 로 채택 → 장애물 집합에서 빠진 뒤 팔이 병을 가림 (G159). 대응 (attention 거름망) 은 검증 중이라 아직 구성에 넣지 않았다.

## 교체 기록 (무엇 → 무엇, 이유)

| 날짜 · 시각 (KST) | 교체 | 이유 (판정) |
|---|---|---|
| 2026-11-08 20:13 | target identity: attention grounding 만 → **VLA 가 손을 닫는 지점 아래 물체** (+ 목적지 등록은 쥔 동안만, 쥔 물체는 장애물에서 제외) | GT 없는 판 실패 16 중 12 가 첫 identity 오류 (그릇이 cabinet · 서랍과 한 성분). 손 닫는 지점 최근접 물체 = target 34 / 36 (G151–G152) |
| 2026-11-08 21:01 | 자른 물체 받침 판정: 탁자 plane 만 → **자른 물체의 국소 받침면 추가** | cabinet 위 그릇이 'not_supported' 로 거절돼 전환이 늦음 (G154) |
| 2026-11-08 21:02 | closing 중 identity: 손 닫는 지점마다 다시 고름 → **closing 부터 잠금** | closing 중 그릇 ↔ 서랍 조각을 오감 (G154) |
| 2026-11-08 22:19 | 장애물 물체 행: 장애물 점 집합만으로 판정 → **집합 AND 기존 분류** | 집합이 기존 분류 (쥔 물체 잔재 · 받침 · 목적지) 를 덮어써서 쥔 그릇이 장애물로 남음 (G157) |
| 2026-11-08 22:19 | 목적지 등록: 잡을 수 없는 크기만 → **쥔 동안 손 밖 attention 1 위면 크기 무관** | 접시 (10.7 × 135 × 135 mm) 가 '잡을 수 있음' 으로 분류돼 한 번도 등록 안 됨 → 내려놓을 때 장애물 (G157) |
| 2026-11-08 22:19 | 자른 점 attention: 모두 1 → **손 닫는 지점에 가까울수록 높게**, 국소 받침면: 그 call 만 → **episode 동안 유지** | 서랍 앞판 조각이 같은 점수로 1 위가 되어 identity 가 흔들림; 손 닫는 지점이 없는 call 에 그릇이 다시 거절됨 (G157) |
