# AG3S-TASK-DYNAMIC-MAP — 작업에 필요한 영역만 동적으로 유지하는 거리장

> writer: lead · 실행: lead 직접 (순차 작업 — 구현 → smoke → closed-loop → 기록 분석)

작성: 2026-10-23 KST, lead. **설계 · 지표 · 판정 기준 · 장면은 결과를 보기 전에 고정한다.**
STEP 이름 예약: 2026-10-23, 같은 pod 의 다른 세션 4개에 알림(`방법론 한번 굳히고 가자.` 세션이 확인 회신, 같은 code 파일을 만질 계획 없음).

## 0. 사용자 판정 (2026-10-23)

- "A 부터 진행" → target 전용 TSDF (아래 ①)
- "보이는 부분에서 이전 관측 위치에 물체가 없으면 지워지고, 다시 관측되면 추가되어야 한다. 멀리 있는 물체에는 관심 없고,
  target 과 그 근처, place 할 곳, 그리고 그 사이 경로에 무엇이 있는지가 중요하다" → 관심 영역 안 보이는 빈 공간 지우기 (②)
- "잡기 전에 target 이 옮겨지면 옛 위치도 갱신되어야 한다" → target TSDF 이동 감지 reset (①-b)
- 확인 질문 1–4 는 "권장 방향으로" → ① + ② 한 STEP · 4 구성 비교, 관심 영역 = 세 영역의 합(pad 0.10 m),
  아래 §5 판정 기준, STEP 이름 `AG3S-TASK-DYNAMIC-MAP`

이 STEP 은 `AG3S-DYNAMIC-FIELD` 의 후속 연구 축이다. `AG3S-CORE-EVALUATION` 의 세 구성 비교에는 넣지 않는다
(`doc/active/plans/AG3S_RESEARCH_SCOPE_20261023_KO.md` 의 "새 기능 중단" 은 핵심 평가 회차의 범위이고, 이 축은 사용자가 별도로 연 것이다).
develope_PLAN §0.10 연결: **경계 보존** 항목 — target 의 경계를 거리장 안에서 지우지 않고 **층으로 분리**해 보존한다
(identity = attention target, extent = target TSDF, contact = 링크별 층 선택). ablation 은 §4 의 4 구성이다.

## 1. 왜 지금 이것인가 (측정된 사실)

| 사실 | 출처 |
|---|---|
| cuRobo TSDF 는 이번 depth 표면 ± truncation 을 지나는 block 만 갱신한다. 그 앞의 보이는 빈 공간에 남은 옛 표면은 그대로다 | `third_party/curobo/.../builder_camera_integrate.py:154-159, 543-556` |
| cuRobo 의 동적 처리는 depth 를 보지 않는 시간 · 시야 감쇠뿐이다. TSDF voxel 에는 이름이 없어 attach 의 `disable_obstacle_names` 가 적용되지 않는다 | `wp_decay.py`, `attachment_manager.py:182` |
| 사과가 302 mm 떠난 뒤에도 그 자리가 −6.7 mm 로 점유, 8 frame 내내 자유가 되지 않았다 | F20 (`config.py:1030-1036`) |
| 잡기 전 211 에서 손바닥이 사과를 22.7 mm(닫기 전) · 36 mm(닫는 중) 밀었다. 기본 구성에서 형상은 옛 자리에 고정 | IFG-211-GRASP, commit `8da37119` |
| 장면 전체 "보이는 빈 공간 비우기" 는 main 층 옛 기억 잔상 −54.2 %, 처리 시간 20.3 → 122.5 ms | `AG3S-DYNAMIC-FIELD.verify.json` |

**앞 측정의 한계 (다른 세션 재집계, 2026-10-23):** `dynamic_field_replay.py` 는 `builder.update(target_points=None)` 를 바로 불러
서버의 쥔 물체 처리(depth mask · `free_held_traces` · target 제외 · held-body-cover)를 빠뜨린다. 끔 구성 옛 기억 잔상의 98.4 %
(106,454 / 108,136)가 닫힌 뒤 구간이다 — 서버가 이미 지우는 것일 수 있다. 서버 일치 확인도 첫 닫힘 전 167 / 364 요청뿐이다.
fine 층에서는 보이는 빈 공간 비우기가 잔상 −72 % 이지만 가려졌던 점 잊음이 +1.22 pp 다.
→ **이 STEP 은 그 replay 로 평가하지 않는다.** closed-loop 서버 기록의 실제 거리장을 GT 와 비교한다 (§4).

## 2. 선행 연구와의 관계

| 우리 구성 요소 | 가장 가까운 선행 연구 | 가져온 것 | 다른 점 |
|---|---|---|---|
| ① target 전용 TSDF | Panoptic Multi-TSDFs (Schmid et al., ICRA 2022) · TSDF++ (Grinvald et al., ICRA 2021) · nvblox 사람 layer (Millane et al., ICRA 2024) · open-vocabulary shared autonomy 의 target dynamic layer (arXiv 2607.17323) | **물체를 변화의 단위로** 둔다. 움직이는 것은 정적 지도에 넣지 않고 별도 층에 둔다 | 물체 identity 를 외부 panoptic segmenter 가 아니라 **VLA attention** 이 정한다. 층을 **링크별 권한**으로 고른다 (손 = 정적만, 그 밖 = 정적 ∪ target). 2607.17323 은 옛 자리 정리와 정량 평가가 없다 |
| ①-b 이동 감지 reset | Panoptic Multi-TSDFs 의 submap 변화 감지 · POCD (Qian et al., RSS 2022) 의 물체별 변화 판정 · Khronos (Schmid et al., RSS 2024) 의 short-term object fragment | 물체가 바뀌면 그 물체의 submap 만 다시 만든다 | 변화 판정이 확률 갱신이 아니라 `kept-geometry-follow` 의 고정 반지름 재맞춤(이동량)이다 |
| ② 관심 영역 안 보이는 빈 공간 지우기 | ReFusion (Palazzolo et al., IROS 2019) · Dynablox (Schmid et al., RA-L 2023) · DUFOMap (Duberg et al., RA-L 2024) · OctoMap/MoveIt ray clearing | depth 광선이 지나간 자리는 빈 공간이다 | 장면 전체가 아니라 **작업이 지나갈 영역**(target · 놓을 곳 · 손 경로)에만 건다. Dynablox 의 다중 frame 신뢰도는 이번에 넣지 않는다 (§6 되돌아올 지점) |
| (참고) 손에 가려진 자리 | Lost & Found (Behrens et al., RA-L 2025) · DSR-Net (Xu et al., CoRL 2020) · RoboEXP (Jiang et al., CoRL 2024) | 손–물체 상호작용 구간과 object permanence | 이번 범위 밖. 손에 가려 다시 보이지 않는 이웃 물체의 옛 자리는 ② 로도 안 지워진다 |
| (대조) 시간 감쇠 | cuRoboV2 (arXiv 2603.05493) · nvblox occupancy decay | — | 쓰지 않는다. 가려진 실제 장애물을 함께 잊는다 |

## 3. 구현 (모두 기본 꺼짐. 꺼짐이면 출력 · 기록 키 비트 동일, test 로 보인다)

### ① target 전용 TSDF — `esdf.target_layer` (`--target-layer`)

- **pixel 배정:** 카메라마다 robot mask 밖 유효 depth pixel 을 base frame 으로 역투영해, 이번 frame manipulated 점
  (`_build_esdf` 의 `target.points`)까지 `target_layer_pixel_tol`(기본 = 미세 voxel 1 칸) 안이고 지지면 + pad 위인 pixel 을 target pixel 로 둔다
  - target pixel → **target TSDF** 에만 적분 (target 주변만 덮는 별도 cuRobo `Mapper`, 미세 truncation)
  - 나머지 pixel → **정적 TSDF** (기존 main / fine TSDF). target pixel 은 robot mask 처럼 관측 없음으로 넘긴다
- **정적 TSDF 의 target 흔적:** target 확정 전 frame 에 들어간 target 표면을 지운다 — 매 frame 이번 manipulated 점의 tol 안 · 지지면 + pad 위의
  정적 voxel 을 free 로 쓴다 (`_free_tsdf_points` 와 같은 쓰기: `+truncation`, weight 유지)
- **①-b 이동 감지 reset:** target TSDF 를 reset 한 frame 의 manipulated 중심과 이번 중심이 `target_layer_reset_m`(기본 10 mm) 넘게 다르면
  target TSDF 를 reset 하고 이번 frame 의 target pixel 로 다시 적분한다. `clustering.kept_geometry_follow` 가 켜져 있어야 한다
  (꺼져 있으면 밀린 사과의 pixel 이 옛 고정 형상과 맞지 않아 정적 TSDF 로 들어간다 — 설정 검증에서 거부)
- **층 합성:** 본 계층(권한 없는 링크) = `min(정적, target)` (tier 마다 같은 격자). target 없는 계층(권한 있는 손) = 정적 tier 그대로 —
  공 · ownership 제외를 하지 않는다. 라벨: target tier 가 더 가까운 voxel 은 target 라벨
- **쥔 뒤 (attach):** target TSDF 를 reset 하고 쓰지 않는다. 쥔 물체는 기존 로봇 쪽 처리 (`free_held_traces` 그대로)
- 기록: `esdf_stats.target_layer = {n_target_px, n_static_freed, reset, reset_reason, centre_shift_mm, build_ms}` (켰을 때만)

### ② 관심 영역 안 보이는 빈 공간 지우기 — `esdf.roi_carve` (`--roi-carve`)

- 관심 영역 = AABB 합: manipulated 점 + `roi_pad` · 목적지 점 + `roi_pad` · 미세 창 상자(권한 링크의 실행 경로 swept 상자 ∪ manipulated 상자, T21).
  `roi_pad` 기본 0.10 m
- 영역 안의 점유 voxel 에만 기존 D1 판정(`_carve_seen_through`: 카메라마다 투영, 관측 depth 가 voxel 깊이보다 margin 넘게 멀면 free)을 건다.
  margin = 그 TSDF 의 truncation + voxel 1 칸 (D1 기본). robot mask · 무효 pixel 은 판정하지 않는다
- 정적 TSDF 와 (① 이 켜져 있으면) target TSDF 에 건다. 탁자 아래(지지면 + pad 아래)는 빼지 않는다 — 광선이 지나간 자리만 지우므로 보이는 탁자 면은 표면 자체라 지워지지 않는다
- 기록: `esdf_stats.roi_carve = {n_roi_occupied, n_freed, roi_boxes, ms}` (켰을 때만)

## 4. 평가 (closed-loop 서버 기록의 실제 거리장)

- **장면:** `IFG-APPLE-NBC-TARGETSURFACE` 와 같은 **사과 VLA 단독 성공 10 장면 · 같은 policy seed** (56, 126, 211, 250, 309, 320, 753, 954, 1668, 1837; 211 · 320 · 954 는 놓을 곳 장애물 포함). 결과를 보고 장면을 바꾸지 않는다
- **기준 구성 B:** NBCTS + `--kept-geometry-follow` (① 이 follow 를 요구하므로 네 구성 모두 같은 follow 위에서 비교). NBCTS 기존 기록은 follow 끈 참고로만 쓴다
- **4 구성:** B / B + ① / B + ② / B + ① + ② → 40 episode. client `--stop-after-place 2.5`, server `--record-constraints`
- 순서: CPU test → GPU smoke 1 episode (구성 B + ① + ②, 기록 키 · 시간 확인) → 40 episode → 기록 분석

| 지표 | 정의 (chunk 마다, 관심 영역 안) | 어느 층 |
|---|---|---|
| **주 지표 — 성공 · 금지 접촉** | task success, 금지 접촉(이웃 과일 포함) episode 수, 금지 접촉 없는 성공 수 (MuJoCo 실제 접촉) | — |
| **target 옛 자리 잔상** | 점유(≤ 0) voxel 중 현재 GT 표면 · 로봇 구에서 15 mm 넘게 멀고, target 이 그 episode 의 이전 chunk 에 있던 GT 표면 15 mm 안인 것 | 손이 보는 층 (`esdf_free_distance`) · 본 층 (`esdf_distance`) |
| **움직인 물체 옛 자리 잔상** | 같은 정의, target 이 아닌 movable body | 본 층 |
| **잊은 실제 장애물** | GT 표면 표본(로봇 · 쥔 물체 제외) 중 거리장이 10 mm 넘게 멀다고 보는 비율. 지금 보임 / 가려졌으나 전에 보임 | 본 층 · 손이 보는 층 |
| 비용 | builder 시간 (중앙값 · p95) | — |

GT 는 같은 chunk 의 client MuJoCo 상태(기록 traj / replay)에서 로봇이 아닌 충돌 geom 표면 표본. 15 mm · 10 mm 는 AG3S-DYNAMIC-FIELD 정의를 그대로 쓴다.

## 5. 미리 정한 판정 (이번에 정한 연구 의사결정 기준이며 기존 측정이나 분야 표준이 아니다)

구성마다 B 대비, 관심 영역 안 10 episode 합으로:
1. ① 이 든 구성: **target 옛 자리 잔상 90 % 이상 감소** (손이 보는 층과 본 층 각각 — §5a)
2. ② 가 든 구성: **움직인 물체 옛 자리 잔상 50 % 이상 감소** (본 층)
3. 모든 구성: **잊은 실제 장애물 +1 pp 이하** — 지금 보임 · 가려졌던 점 각각, 본 층과 손이 보는 층 각각
4. closed-loop: 성공 수가 B 보다 적지 않고 금지 접촉 episode 가 B 보다 많지 않을 것 (10 episode 이므로 우수성 주장은 하지 않는다)
- 1–3 을 만족하고 4 를 지키면 "작업 영역 동적 거리장" 으로 채택. 3 을 어기면 그 구성은 채택하지 않는다. 처리 시간은 판정에 넣지 않고 기록한다
- B 의 해당 잔상이 0 이면 그 기준은 "판정 불가" 로 적고 통과로 세지 않는다

### 5a. 결과 전 수정 (2026-10-24 04:2x KST, lead — 네 구성의 rollout 전)

평가 도구(`task_dynamic_map_eval.py`)를 **기존 기록** (IFG-APPLE-NBC-TARGETSURFACE NBCTS, 211, follow 꺼짐) 에 먼저 돌려 정의를 점검했다.
네 구성의 결과는 아직 없다.

| 무엇을 바꿨나 | 왜 |
|---|---|
| 잔상을 **표면 voxel** (기록값 \|d\| ≤ voxel 반 칸 = cuRobo seed) 로 센다 (§4 의 "점유 ≤ 0" 대신) | ≤ 0 은 탁자 판 · 물체 내부와 관측면 뒤 음수 띠를 세어 chunk 마다 약 1 만 voxel 이 "잔상" 으로 잡혔다 |
| "옛 자리" = 물체가 **10 mm 넘게 떠난** 이전 자세만 (물체별 중심 기준) | 바구니가 free joint 라 움직이지 않았는데도 기억에 들어가 옆의 그림자 seed 가 chunk 마다 13 ↔ 1,174 로 깜빡였다 |
| GT 장면에 client 와 같은 장애물 profile 을 배치 (`PickPlaceObstacleManager.activate`) | 211 · 320 · 954 의 놓을 곳 기둥이 GT 에 없으면 "잊은 실제 장애물" 에서 빠진다 |
| 관심 영역(평가용) = 사과 처음 GT 표면 ∪ 바구니 상자 ± 0.10 m, ∪ 사과 현재 GT 표면 ± 0.10 m | 구성마다 같은 영역이어야 한다 (알고리즘의 관심 영역과 무관하게 GT 로 고정) |
| **판정 1 을 두 층 각각으로** — target 옛 자리 잔상 90 % 이상 감소를 손이 보는 층과 본 층 **각각** | 기존 기록 211 에서 사과가 밀린 뒤의 옛 자리 잔상은 본 층 21,219 · 손이 보는 층 38 (75 chunk 합) — 공 제외가 손 층의 것은 이미 지우고, 남는 것은 손 외 몸이 보는 본 층이다 |

## 5b. 실행 준비 (2026-10-24 KST)

- run: `/mnt/dev/AG3S_VLA-storage/results/development/20261024T034606+0900_8da37119afd3_31d9b16e3c97`, snapshot `snapshot_tdm` (working tree = HEAD 8da37119 + 미 commit AG3S-DYNAMIC-FIELD · 이 STEP 코드)
- kit `verify/AG3S-TASK-DYNAMIC-MAP/p/` (IFG-APPLE-NBC-TARGETSURFACE kit 복사): 네 구성은 NBCTS guard 를 그대로 받고 (`base_arm`), `--kept-geometry-follow` 필수,
  `--target-layer` · `--roi-carve` 는 구성대로만, 다른 동적 거리장 flag 거부. 서버 chunk 통계는 `tdm_stats_check.py` 가 확인. smoke = B_TL_ROI 56 (`smoke_check_tdm.py`)
- 순서: smoke → A1 (B | B_TL, 서버 2) → A2 (B_ROI | B_TL_ROI) → 기존 접촉 분석 → §4 거리장 평가. 예상 GPU 약 2.5 h (episode 약 6 분, 기존 kit 실측)
- CPU 사전 점검: 네 구성의 서버 인자를 snapshot 코드로 파싱 → `esdf.target_layer` · `roi_carve` · `kept_geometry_follow` 가 구성대로 (B: F/F/T, B_TL: T/F/T, B_ROI: F/T/T, B_TL_ROI: T/T/T)
- test: `tests/ag3s/test_task_dynamic_map.py` 14 passed (GPU 4 포함). `tests/ag3s` + `tests/trajopt` CPU 3708 passed · 7 failed — 7 개 모두 이 STEP 이전부터:
  6 개는 미 commit WIP 의 `serve_safe.main` → `ag3s_from_args` 이동으로 깨진 소스 문자열 test, `test_t43t_replay::test_off_is_bit_identical…` 는 깨끗한 HEAD worktree 에서도 실패

## 6. 되돌아올 지점

- target pixel 배정이 이웃 과일 표면을 target 으로 가져가면 (잊은 실제 장애물 ↑) → tol 을 줄이거나 attention surface (`target_extent=attention_surface`) 의 pixel 판정으로 바꾼다
- ② 가 가려졌던 점을 잊으면 → Dynablox 식 다중 frame 확인 (연속 k frame 광선 통과) 을 더한다
- 손에 가려진 이웃 물체 옛 자리 → Lost & Found · POCD 식 상호작용 구간 표시 (별도 STEP)

## 6a. 1차 결과 (2026-10-24, verify.json) 와 사용자 판정

- 판정: 세 구성 모두 미채택 (B_TL: 기준 3 · 4, B_ROI: 기준 4 만, B_TL_ROI: 기준 1 · 3 · 4). 미리 정한 판정은 바꾸지 않는다
- 사용자 (2026-10-24): "사과 잔상을 제외하고는 거의 다 아슬아슬 — 나쁘지 않다" → 권장 방향으로 진행:
  ① 954 이웃 과일 접촉 진단 ② 가려졌던 장애물 잊음 증가의 원인 (물체별) ③ 판정 기준을 미리 고정하고 표본을 늘려 재비교

## 8. 추가 실험 E2 — 관심 영역 지우기, 표본 확대 (결과 전 고정, 2026-10-24 13:3x KST)

- **구성:** B (NBCTS + 추종) 와 B_ROI (+ `--roi-carve`) 두 개만. 코드 · snapshot (`snapshot_tdm`) · kit · 서버 인자는 1차와 같다
- **장면:** IFG-APPLE 풀 24 장면 중 VLA 단독 (E0) 성공 18 장면 **전부** = 1차 10 장면 + 새 8 장면
  (1345, 1596, 121, 119, 325, 901, 572, 850 — 901 · 572 · 850 은 놓을 곳 기둥). policy seed = 풀의 seed. 새 8 장면만 돌리고 1차 10 장면 기록과 합친다 (16 episode)
- **판정 (18 장면 합, B 대비, §5 · §5a 와 같은 정의 · 같은 한도):**
  2. 다른 물체 옛 자리 잔상 (본 층) 50 % 이상 감소
  3. 잊은 실제 장애물 +1 pp 이하 — 본 층 · 손 층 × 지금 보임 · 가려졌던 점 각각
  4. 성공 수 ≥ B, 금지 접촉 episode ≤ B
  - 넷 다 지키면 "관심 영역 지우기" 채택. target 옛 자리 잔상은 기록만 (B_ROI 는 ① 구성이 아니다)
  - 1차의 미달 판정은 그대로 남는다 — E2 는 별도 판정이며 1차를 덮지 않는다
- 진단 ① · ② 의 결과는 E2 의 설계를 바꾸지 않는다 (코드 수정 없이 같은 B_ROI 를 더 넓은 표본으로 잰다). 진단이 고칠 곳을 가리키면 별도 STEP 으로 제안한다

### 8a. E2 결과 (2026-10-24, verify.json `numbers.e2_18_scenes`)

- **B_ROI 채택** — 18 장면에서 기준 2 (다른 물체 옛 자리 잔상 1,277 → 65, −94.9 %) · 3 (최대 +0.951 pp, 손 층 가려졌던 점) · 4 (성공 16 ≥ 14, 금지 접촉 episode 3 ≤ 3) 모두 통과
- 채택은 이 STEP 의 판정이다. serving 기본값은 여전히 꺼짐 — 기본값 변경은 사용자 판정으로 한다
- 진단 ②: 잊은 장애물 증가는 놓을 곳 기둥 (반지름 15 mm) 에 몰렸다 → 가는 물체 가장자리 보존이 다음 수정 후보 (원인 층 LTE)

## 9. 추가 실험 E3 — 가는 물체 가장자리 보존 (결과 전 고정, 2026-10-24 KST)

사용자 판정 (2026-10-24 "수행하세요"): 권장안 — ② 의 판정을 voxel 의 투영 영역 전체로 바꾸는 기본 꺼짐 flag, 서버 기본값은 이 수정이
확인될 때까지 꺼짐, 로그는 scribe.

- **구현:** `--roi-carve-footprint` (`esdf.roi_carve_footprint`, `--roi-carve` 와 함께만). voxel 의 투영 영역 (반지름 ``ceil(0.5 · f · voxel / z) + 1`` px)
  의 모든 pixel 이 유효 (범위 안 · 로봇 아님 · 화면 안) 하고 voxel 보다 margin 넘게 뒤를 봤을 때만 지운다. 반지름 8 px 넘는 (아주 가까운) voxel 은 지우지 않는다
- 합성 장면 측정 (test `test_roi_carve_footprint_keeps_a_thin_post`): 30 mm 기둥, depth 를 0.5 px 중심으로 그리면 기존 판정은 기둥 옆면 표본의 10.7 % 를
  1 frame 부터 잃고, 영역 판정은 0 %. depth 를 정수 중심으로 그리면 둘 다 0 % — 실제 서버의 반 pixel 어긋남은 측정하지 않은 가설이다
- **구성:** B_ROI_FP = B_ROI + `--roi-carve-footprint`. 새 snapshot (`snapshot_tdm2`), kit `verify/AG3S-TASK-DYNAMIC-MAP-E3/` (E2 kit 복사)
- **장면:** E2 와 같은 18 장면 · 같은 policy seed. B · B_ROI 는 E2 기록을 그대로 쓴다 (18 episode 만 새로 돈다)
- **판정 (18 장면 합):**
  - (a) B 대비 §8 의 기준 2 · 3 · 4 를 모두 지킨다
  - (b) B_ROI 대비 놓을 곳 기둥의 잊은 표면 (본 층 · 가려졌던 점, 3 chunk 마다 표본, `diag_missed_by_body.py`) 50 % 이상 감소
  - (a) · (b) 를 지키면 관심 영역 지우기의 채택 형태를 B_ROI_FP 로 바꾸고 서버 기본값 켜기를 제안한다. (a) 만 지키면 B_ROI 를 유지한다
- 비용 (서버 거리장 단계 ms) 은 기록만

### 9a. E3 결과 (2026-10-24, verify.json `numbers.e3_18_scenes`)

- **가장자리 보존 미채택** — (b) 기둥 놓친 표면 (본 층 · 가려졌던 점) 17.64 % → 1.76 % (−90 %) 통과, (a) 기준 2 (다른 물체 옛 자리 잔상 1,277 → 1,145, −10.3 %) ·
  기준 4 (금지 접촉 episode 4 > 3) 미달. 기준 3 은 네 항목 모두 B 보다 낮다 (−0.08 ~ −0.39 pp). 성공 15 ≥ 14
- 관심 영역 지우기의 채택 형태는 B_ROI 유지 (§9 규칙)
- 관측: 이웃 물체가 옮겨진 1345 · 1668 에서 B_ROI 는 옛 자리 잔상 0 · 0, B_ROI_FP 는 327 · 783 (B 358 · 857). 영역 판정이 옛 자리도 지우지 않게 됐다 —
  기전 (손 근처 robot mask pixel 이 영역에 들어가 판정 보류 등) 은 측정하지 않은 가설
- 금지 접촉 episode: 211 (탁자) · 753 (탁자 78 row) · 954 (바나나) · 1345 (바나나)

## 10. 추가 실험 E4 — 가장자리 보존 완화 (결과 전 고정, 2026-10-24 KST)

사용자 판정 (2026-10-24 "1, 2 는 권장대로 · 로그는 새 문서"): (가) 가장자리 보존 완화를 검증하고, 통과하면 서버 기본값을 켜고 아니면 꺼짐 유지.
로그는 `AG3S/ag3s/docs/AG3S_TSDF_DYNAMIC_LOG.md` (scribe).

- **구현:** `--roi-carve-footprint-robot ignore` — 영역 안 robot mask pixel 을 판정 보류 사유에서 뺀다. 가장 가까운 pixel 자체는 robot 이 아니고 margin 넘게
  뒤를 봐야 한다 (손이 가린 자리는 여전히 지우지 않는다). `--roi-carve-footprint-pad-px 0` — 영역 반지름 = ``ceil(0.5 · f · voxel / z)`` px
- 합성 장면: 30 mm 기둥 옆면 손실 0 % (기존 판정 10.7 %, 0.5 px 중심), 치운 이웃 상자 옛 자리는 지워짐 (윗면 거리 32.5 mm). 합성 장면에 로봇이 없어
  E3 의 판정 보류 가설은 closed-loop 에서만 확인된다
- **구성:** B_ROI_FPR = B_ROI + `--roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore`. snapshot `snapshot_tdm3`,
  kit `verify/AG3S-TASK-DYNAMIC-MAP-E4/` (E3 kit 복사). 18 장면 · 같은 seed, B · B_ROI 는 E2 기록
- **판정 (§9 와 같은 정의 · 같은 한도):** (a) B 대비 기준 2 · 3 · 4 모두, (b) B_ROI 대비 놓을 곳 기둥 놓친 표면 (본 층 · 가려졌던 점) 50 % 이상 감소
  - (a) · (b) 모두 → 관심 영역 지우기의 채택 형태를 B_ROI_FPR 로 바꾸고 서버 기본값을 켠다. 켜기 전에 핵심 평가 (CORE-EVALUATION) 세션에 영향 (그 kit 은
    snapshot 이라 영향 없음, checkout 에서 새로 띄우는 실행은 바뀜) 을 알린다
  - 그 밖 → 서버 기본값 꺼짐 유지, 채택 형태는 B_ROI

### 10a. E4 결과 (2026-10-24, verify.json `numbers.e4_18_scenes`)

- **가장자리 보존 완화 채택** — (a) 기준 2: 다른 물체 옛 자리 잔상 1,277 → 584 (−54.3 %, 한도 −50 %) · 기준 3: 네 항목 모두 B 보다 낮음 (−0.007 ~ −0.344 pp) ·
  기준 4: 성공 15 ≥ 14, 금지 접촉 episode 1 ≤ 3 (1345 바나나 27 row). (b) 기둥 놓친 표면 (본 층 · 가려졌던 점) 17.64 % → 2.19 % (−87.6 %)
- 사과 옛 자리 잔상 (본 층) 50,377 → 6,508, 서버 거리장 단계 중앙값 303 ms (B 110 ms)
- **serving 기본값 켬** (2026-10-24, `serve_safe.ROI_CARVE_SERVING_DEFAULT`): `--esdf-backend curobo` 로 AG3S 를 띄우고 관심 영역 flag 를 하나도 안 주면
  `--roi-carve --roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore` 와 같다. `--no-roi-carve` = 이전 esdf 설정.
  `--no-safe` · `--no-perception` · legacy backend · 명시한 carve flag 는 이전 의미 그대로. `EsdfConfig` 기본값은 꺼짐 유지. 시작 로그 "AG3S roi_carve: …".
  핵심 평가 세션 (현재 `ag3s-vla-development-c4`) 에 켜기 전 알림 — 그 세션은 checkout 에서 kit 을 새로 만들 때 `--no-roi-carve` 를 manifest 에 명시한다고 했다

## 7. 산출물

- code: `AG3S/ag3s/fields/curobo_builder.py` · `AG3S/ag3s/runtime/pipeline.py` · `AG3S/ag3s/config.py` · `AG3S/trajopt/serve_safe.py`, test `tests/ag3s/test_task_dynamic_map.py`
- 분석: `AG3S/ag3s/experiments/studies/task_dynamic_map_eval.py` (서버 기록 거리장 × GT)
- `AG3S/ag3s/docs/handoff/AG3S-TASK-DYNAMIC-MAP.verify.json`, 그림(실제 씬 · 잔상 막대 · 표) `AG3S/ag3s/docs/figures/`
- 보기: `outputs/2026-10-23_작업영역_동적거리장/` (`python -m common.experiment_view`)
