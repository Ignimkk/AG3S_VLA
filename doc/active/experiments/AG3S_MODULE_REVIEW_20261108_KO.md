# AG3S 모듈별 리뷰 · 검증 (Panda/LIBERO, 2026-11-08 00:45 KST –)

> 읽는 사람: 사용자. 작성자: lead (session dev-49). 자율 판정은 **2026-11-08 13:00 KST 까지만** 하고, 그 뒤 판정은 사용자 피드백을 따른다.
> 이 문서가 모듈별 리뷰의 정본이다. 모듈마다 아래 순서로 쓴다:
> **무엇을 · 어떤 입력으로 · 어떤 방법으로 했는가 → 결과(표 · figure) → 판정(잘 됨 / 안 됨과 그 이유) → 한계**.
> figure 와 스크립트는 checkout `outputs/2026-11-08_*` (git 제외)에 있다. 공유 코드(`AG3S/`, `benchmark/runners/`)는 수정하지 않았다.
> 모든 검증은 CPU offline 이다. GPU 는 다른 세션(dev-ec)이 독점 lock 으로 쓰고 있어 쓰지 않았다.

## 요약 판정표 (자율 판정, 13:00 KST 전까지 갱신)

![overview](../../../outputs/2026-11-08_AG3S_review_overview/overview.png)

데이터 흐름 순서다. 초록 = 잘 됨, 노랑 = 부분적 / 알려진 구멍, 빨강 = 안 됨.

| § | 모듈 | 판정 | 핵심 근거 | 원인 · 제안 (구현하지 않음) |
|---|---|---|---|---|
| 1 | 로봇 마스크 (render) | Panda 수정판 잘 됨 / RB-Y1 그대로 안 됨 | Panda render recall 0.96–1.00, FP 0–91 px (3 episode). RB-Y1 방식 head recall 0.63–0.82, FP ~500 px | collision geom · site 를 그림 → group 1–2, site 끔. sphere mask 는 손 근처 target · **장애물** pixel 을 지움 (wrist 장애물 4 장 1210 px) |
| 3 | depth → point cloud | 잘 됨 (1 episode · 4 frame) | 광선 대조 오차 중앙 head 1.1 mm · wrist 0.3 mm. > 5 mm 의 99.1 % 는 깊이 경계 | — |
| 2 | attention 추출 · RGB 매핑 | 매핑 잘 됨 / head 마다 다른 약점 | L8H2: head target mass 균일의 18 배, 쥔 뒤 목적지로 감. noun L8H6: target 0.45 · 쥔 뒤에도 유지, 대신 distractor 0.24 (§2.5) | 쥔 뒤 identity 는 latch 로 유지해야 함 |
| 4 | cluster · grounding (GT 없는 판) | 안 됨 (부족) | 접근 정답 0.54, task 4 · 9 는 0. target 점의 ~26 % 가 support 로 빠짐. closed-loop smoke 에서 t4 실패 (no_target → HOLD 8) 로 예측과 일치 | 서랍 + 그릇이 한 cluster, distractor 선택, 두 번째 support 평면 |
| 5 | 표면 분할 (ATS surface, P0m) | 안 됨 (쥐기 전) | GT attention 으로도 쥐기 전 recall 0.22, abstain 45 %. 서버 후처리 포함 쥐기 전 target 표면의 평균 6 % (L8H2) · 11 % (P0m 본판 noun head) 만 지움, memory 의 35 % 는 distractor | 두 번째 RANSAC 평면 (1 평면 반사실 recall 0.88) + `not_ats_target` 기각 + `grasp_hold`. 탁자 위 target 에서만 크게 실패 (SafeLIBERO 의 높은 target 은 GT attention 으로 적용 0.99) |
| 16 | latch × grounding (GT 없는 판) | 안 됨 | latch 흉내 시 접근 정답 0.54 → 0.44. 틀린 첫 identity (10 / 40 episode) 가 고정됨. close 2 call 전 latch 반사실: grasp 0.60 (셋 중 최고), approach 0.53 (latch 없음 0.54 와 같은 수준) | LATCHED 가 너무 이름 → close 직전으로 늦추기 |
| 6–7 | 작업 단계 · grasp latch | attach 잘 됨 / reach 경계 실패 2 / 38 / detach 는 1 사례만 진실 대조 | LIBERO 26 / 26 · SafeLIBERO 10 / 10 attach 가 양손 접촉 상태. SafeLIBERO t3 e3 에서 놓기 3 frame 뒤 detach. 2 episode 는 reach 120.7 · 122.7 mm > 120 으로 끝까지 attach 못 함 | reach 문턱 (120 mm) 이 그릇 중심 기준. 세 판 모두 판당 4–6 / 48 episode 가 reach 로 attach 못 함 (§6.7). PLACED · home 은 도달하지 않음 |
| 13 | 쥔 물체 모델 · detach | F8 안 됨 / F12 는 열린 손 attach 0 / 최신 점 모델 부분 (들어 올린 뒤 아랫면 빠짐 8–45 mm, 반쯤 닫힌 attach 시 덮임 0.11) | F8 몸통 덮개 바닥 −30 – −61 mm, 열린 손 attach 뒤 detach 영구 불능. F12 detach 2 회 발생 (진실 대조는 SafeLIBERO 1 회) | 최신 점 모델은 미측정 |
| 8 | TSDF / ESDF | 보이는 면 잘 됨 / 안 보이는 면 없음 / 뒷면 그림자 | 표면 3 cm 안 오차 중앙 ±3 mm. 충돌 직전 6 시점 중 5 에서 보수적. 탁자 · cabinet 근처 41–46 % 못 봄. 물체 뒤 21–29 mm 점유 그림자 (truncation 30 mm 로 해석, kernel 미확인) | 관측 TSDF 의 성질 |
| 17 | 손 voxel 비우기 (HVC 20 mm) | 안 됨 (켜도 끄도 문제) | observe 7 / 7 · 최신 closed-loop 9 / 11 episode 에서 장애물 voxel 을 지움 (8215 · 7644, 지운 것의 37–38 %). t2 e1 은 실제 −1 – −13 mm 관통인데 거리장 +8 – +38 mm | HVC 끔 (B3): 장애물 놓침 23 → 4, 대신 쓰러진 병 ghost 로 HOLD deadlock (거짓 접촉 304 → 1225) → 근본은 동적 거리장 (§10) |
| 10 | 동적 거리장 | 안 됨 (일부 사례) | F8 observe: 장애물이 움직인 3 사례 중 2 에서 옛 자리 ghost 가 coarse 16–50 % 끝까지 남음. 최신 closed-loop: 장애물이 밀린 1 / 5 에서 0.43 → 0.19, target 옛 자리는 지워짐 | Panda 는 ROI carve 만 켜짐 (감쇠 · free carve 미이식) |
| 12 | target-free 층 | 부분 (GT 판) | target 표면 ~70 % 만 지움. 충돌 뒤 머그 voxel 까지 지움 (9 / 20 chunk) | GT 없는 판은 closed-loop 기록으로 §21 에서 측정 (성공 task 덮임 0.16–0.61, 실패 task 는 다른 물체를 지움) |
| 11 | 충돌 제약 (sphere clearance) | 검출 대체로 됨 / 놓침 있음 / 거짓 많음 | F8 observe 7 episode: 실제 접촉 132 vs 거리장 653 (놓침 45 · 거짓 566). 최신 closed-loop 11: 장애물 놓침 23 (수 mm, HVC episode), 장애물 거짓 8 | 거짓: target 근처 · 그림자 · ghost. 놓침: HVC · 움직인 물체 |
| 18 | 쥐기 영역 해제 | 코드상 구멍 / 이 기록 충돌 원인 아님 | LATCHED · CLOSING 내내 target 수평 100 mm 안 손 행을 표면 종류와 무관하게 해제. 장애물 20 mm 안 행의 14 % 해당 | target · support 행으로 제한 |
| 14 | TO 판정 (pass-through τ) | 안 됨 (F12) / 개선 (gt19pts) | F12: 충돌 8 / 8 이 직전 위반 6.6–20.7 mm < τ 25 mm 로 통과. gt19pts (장애물 행 τ 10): 충돌 5 / 11 — 쥔 물체 행 통과 1 · 장애물 τ 통과 1 · TO 개입 후 충돌 3 | 쥔 물체 ↔ 장애물 행에도 장애물 τ 적용 · 개입 실행 정확도 (§15) |
| 15 | EE → 관절 응답 모델 | 통과 잘 됨 / 개입 안 됨 | 5 step 오차 통과 5 mm vs 개입 33–39 mm (p95 72–82) | 개입 action 포화 (70–76 %) |
| 19 | 서버 support 평면 → TO 행 분류 | 평면 문제 있음 / 행 분류 영향 없음 (이 기록) | 두 번째 평면이 chunk 63–80 % 에 있음 (탁자 위 16–24 mm, ~2°). 장애물 근처 행 338 중 support 분류 0 | — |
| 21 | GT 없는 판 closed-loop (B5 att) | identity 맞으면 잘 됨 / 틀리면 실패 | LIBERO 27: 실패 10 중 8 이 identity (task 3 · 4 · 9 — 다른 물체를 잡아 그것을 지움), 2 는 운반 중 obstacle 거절 (task 7). SafeLIBERO 성공 4 / 11, t3 는 모두 다른 물체 | identity (grounding) 가 병목. offline (§4 · §16) 이 미리 맞힘 |
| 20 | self-collision 끔 | 문제 없음 (이 기록) | 1209 frame 에서 로봇 비인접 link · 쥔 그릇–팔 접촉 0 | — |

**측정하지 못한 것 (GPU 필요 · 기록 없음, 사용자 판정 때 고려):**

- GT 없는 판의 SafeLIBERO · LIBERO-Safety closed-loop — 13:00 까지 기록이 나오지 않았다. LIBERO seed 7 은 §21 에서 측정했다.
- 최신 쥔 물체 점 모델 (`--held-point-model-mm 20`) — constraint 기록 run 이 없다.
- 놓기 · PLACED — LIBERO 는 놓는 순간 episode 가 끝난다.
- cuRobo kernel 의 truncation · `depth_min` 처리 — 코드 인자로만 확인했다.

## 0. 공통 검증 방법

| 항목 | 내용 |
|---|---|
| 기록 | 같은 기계 run `results/benchmark/20261107T203215_ff21caa5042b_gtto-gt15loose2-libero9s7-…` (`libero_spatial`, seed 7). 모듈마다 필요한 다른 run 은 그 절에 적는다 |
| 입력 | `episodes/<stem>_geom.npz`: image instant 의 depth(uint16 mm) · K · T_base_cam · 9-D robot_state · `sim_state_image` · geom segmentation |
| 장면 재구성 | `export_scene.py` (`.venv-libero`): rollout 과 같은 env history (`env_history.episode_envs`) 로 LIBERO env 를 다시 만들고 MJCF 를 내보낸다. 재렌더한 RGB 가 기록과 **bit 단위로 같은지** 확인한 뒤에만 쓴다 |
| GT | 재구성 scene 을 `sim_state_image` 로 두고 다시 렌더한 geom segmentation. 관측 depth 와 가장 잘 맞는 설정(site 숨김, robosuite 처럼 geom group 0 숨김)을 측정으로 골랐다 |
| 절 번호 | 사용자 목록 번호를 따랐다 (1 · 2 · 3 … 11). 사용자 목록에 9 가 없어 §9 는 비어 있다. §12 이후는 목록에 없는 모듈이다 |
| 코드 상태 | branch `feature/panda-benchmark`, HEAD `ff21caa5` + **작업 트리** 의 미commit 수정이다. peer 세션 dev-ec 가 새 opt-in flag 를 넣으며 `pipeline.py` · `linearize.py` · `profile.py` 등을 수정 중이었다. offline replay (§4 · §5.5 · §16) 는 실행 시점의 작업 트리 코드를 import 했다. 기록 대조 (§6–§20) 는 기록을 만든 run 의 코드 (`ff21caa5` + patch) 를 따른다 |
| 채점 범위 | backprojection 유효 depth (0.05–3 m). Panda 서버의 TSDF 는 **cuRobo** backend 이고 (V11AB argv `--esdf-backend curobo`), `MapperCfg.depth_minimum_distance = esdf.depth_min = 0.05` 로 같은 범위만 적분한다 (`fields/curobo_builder.py:2172-2173`; legacy numpy 경로도 `fields/esdf.py:177` 에서 같다) |

---

## 1. 로봇 마스크 (self-filter + render mask) — 판정: **Panda 수정판 render mask 는 잘 됨, RB-Y1 방식 그대로는 안 됨**

### 1.1 무엇을 했나

- **질문:** RB-Y1 에서 쓰는 render mask (IFG R6, `runtime/robot_render_mask.py`, `esdf.robot_mask = "render"`) 를 Panda 에 그대로 적용하면 어떻게 되는가? Panda 에 맞게 고치면 어떻게 되는가?
- **현재 Panda 서버:** sphere self-filter 만 쓴다 (`build_panda_self_filter_model`, 4271 sphere). `--robot-mask` 는 `UNPORTED_FINAL_CONFIG_DESTS` 라 거절한다.
- **입력:** `libero_spatial_t0_e0` (bowl 을 집어 plate 에 놓기) obs 32 · 48 · 64 · 96. head (agentview) · wrist (eye_in_hand) 두 camera.
- **비교한 mask** (모두 같은 depth · K · T · robot_state):

| 이름 | 방법 |
|---|---|
| sphere (전) | 서버와 같은 `panda_depth_robot_mask` (V11AB `backproject` + `depth_robot_mask`, range_max 2.0) |
| RB-Y1 코드 그대로 | `RobotRenderMask(scene.xml)` 를 수정 없이 생성 |
| RB-Y1 방식 | 같은 알고리즘 (robot 만 렌더 → \|관측 depth − 렌더 depth\| ≤ 10 mm). Panda 이름만 넣었다: 9-D robot_state 로 자세, `robot0_base` 기준, `robot0_/gripper0_/mount0_` body. `MjvOption()` 기본값이라 geom group 0–2 와 site group 0–2 를 그린다 |
| Panda 수정 | RB-Y1 방식에서 geom group 1–2 만 (visual), site 는 그리지 않는다 |
| 후 | sphere ∪ render (서버의 TSDF 입력 규칙: 구 mask ∪ render mask) |

- 스크립트: `outputs/2026-11-08_panda_robot_render_mask/{export_scene.py, compare_masks.py}`

### 1.2 결과

**(a) RB-Y1 코드 그대로는 생성 단계에서 실패한다.**

- `ValueError: no joint ['torso_0', …, 'left_arm_6']`: `DEFAULT_RBY1_JOINTS` 20 개가 Panda 모델에 없다.
- 그 검사를 지나도 실패한다. `gripper_body_ids` 는 `KeyError` (`link_left_arm_6` 없음) 를 낸다. RB-Y1 body prefix (`base`, `link_`, …) 와 맞는 Panda body 는 0 개이고, `base` body 도 없다.

**(b) 전/후 이미지** (obs 64; 32 · 48 · 96 도 같은 폴더)

![obs 64](../../../outputs/2026-11-08_panda_robot_render_mask/panda_robot_mask_before_after_f064.png)

색: 파랑 = sphere mask, 주황 = render 가 찾은 robot pixel, 빨강 = robot 이 아닌데 지운 pixel (FP), 보라 = 남은 robot pixel (장애물이 된다), 회색 = depth < 5 cm (backprojection 제외).

**(c) 수치** (R = recall, FP / miss = pixel 수, 유효 depth 안)

| obs · cam | GT px | sphere R / FP | RB-Y1 render R / miss / FP | Panda render R / miss / FP | sphere∪RB-Y1 FP | sphere∪Panda FP |
|---|---|---|---|---|---|---|
| 32 head | 8905 | 0.992 / 62 | 0.816 / 1639 / 506 | **0.980 / 177 / 0** | 563 | 62 |
| 48 head | 10183 | 0.990 / 279 | 0.633 / 3739 / 523 | **0.971 / 294 / 0** | 786 | 279 |
| 64 head | 10250 | 0.991 / 273 | 0.645 / 3640 / 520 | **0.971 / 301 / 0** | 778 | 273 |
| 96 head | 9672 | 0.991 / 294 | 0.652 / 3367 / 531 | **0.970 / 295 / 3** | 799 | 294 |
| 32 wrist | 6463 | 1.000 / 48 | 0.999 / 9 / 40 | 0.997 / 22 / 0 | 52 | 48 |
| 48 wrist | 6464 | 0.999 / 52 | 0.998 / 16 / 85 | 0.996 / 28 / 0 | 98 | 52 |
| 64 wrist | 6012 | 1.000 / 50 | 0.996 / 22 / 70 | 1.000 / 0 / 0 | 80 | 50 |
| 96 wrist | 6014 | 1.000 / 481 | 0.996 / 26 / 158 | 1.000 / 2 / 19 | 526 | 481 |

**(d) wrist 에서 손 아래쪽 띠가 마스킹 안 된 것처럼 보이는 이유** (사용자 질문, 2026-11-08 00:5x)

![wrist](../../../outputs/2026-11-08_panda_robot_render_mask/wrist_depth_min_explained.png)

- wrist 화면의 robot pixel 은 11,900 개다. 그중 5,888 개 (49 %) 는 손 몸체 `gripper0_right_gripper` 인데, depth 가 4.2–5.0 cm 라 `depth_min` (0.05 m) 아래다.
- depth 제한 없이 보면 Panda render 는 11,900 / 11,900 을 덮는다.
- 이 띠는 backprojection 에서도, TSDF 적분에서도 버려진다. 그래서 마스킹 여부와 관계없이 ESDF 에 들어가지 않는다. TSDF 는 cuRobo `Mapper` 이고, `depth_minimum_distance` 0.05 를 받는다.
- sphere mask 는 backproject 된 점에만 적용되므로 이 띠를 원래 덮지 못한다.

### 1.3 판정과 이유

1. **RB-Y1 코드 그대로: 안 됨.** 이유는 RB-Y1 고정 이름 (20 joint, `link_*_arm_6`, body prefix, `base`) 이다.
2. **RB-Y1 방식 (이름만 이식): 안 됨.** 두 가지 이유가 측정으로 확인됐다.
   - **collision geom (group 0)**: Panda collision mesh 가 visual 표면 앞에 렌더되어 관측 depth 와 10 mm 넘게 어긋난다. 그래서 head render 단독 recall 이 0.63–0.82 이다 (그림의 보라색).
   - **site**: MJCF 에는 보이는 site (stove burner site, `gripper0_grip_site_cylinder`) 가 있다. robosuite 는 runtime 에 이 site 들의 alpha 를 음수로 바꿔 카메라에서 숨긴다. 그러나 MuJoCo `Renderer` 는 음수 alpha 여도 site 를 그린다. 이 때문에 FP 가 ~520 px 생긴다.
     - 확인 1: obs 64 head 에서 site 를 그리면 FP 506 · TP 9760, 끄면 FP 0 · TP 9949 다 (현재 GT 기준, `sites_check.json`). 처음 잰 "530 → 213" 은 GT 렌더에도 site 를 그린 상태의 값이었다.
     - 확인 2: 관측 depth 는 site 를 숨긴 렌더와 더 일치한다. |Δd| > 5 mm 비율이 head 1.2–1.5 % 대 1.7–1.9 % 다.
3. **Panda 수정판 (visual geom 만, site 없음): 잘 됨.**
   - head render 단독: recall 0.970–0.980, FP 0–3 px.
   - wrist: recall ≥ 0.996, FP 0–19 px.
4. **TSDF 입력 (sphere ∪ render) 에 대한 효과는 이 episode 에서 거의 없다.**
   - Panda sphere mask recall 이 이미 0.99 라서, Panda render 가 더하는 pixel 은 0–6 px 다.
   - RB-Y1 방식을 쓰면 FP 만 ~500 px 늘어난다.
   - 효과가 큰 곳은 render mask 만 쓰는 경로 (`attention_surface` graph, `esdf.robot_mask` 문서: "graph 는 render mask 만 쓴다") 다.
5. **sphere mask 의 남은 문제:** wrist obs 96 에서 sphere 가 쥔 bowl 의 rim 481 px 를 지운다 (FP). render 는 union 이라 이것을 줄이지 못한다.

### 1.4 한계

- episode 1 개 · frame 4 개다. 다른 장면 (SafeLIBERO, LIBERO-Safety) 은 측정하지 않았다.
- (02:0x 정정) Panda 서버의 TSDF 는 cuRobo backend 다. `depth_min` 0.05 가 `MapperCfg.depth_minimum_distance` 로 넘어가는 것은 코드로 확인했다. cuRobo 적분 kernel 이 그 값을 지키는지는 실행해서 확인하지 않았다 (GPU 를 쓰지 않음).
- `depth_min` 을 5 cm 아래로 낮추면 wrist 손 띠가 TSDF 에 들어오기 시작한다. 그때는 render mask 가 필요하다.
- 공유 코드에 Panda 이식은 하지 않았다. 이식할 때 바꿀 것 세 가지:
  1. joint · body 이름표
  2. geom group 1–2
  3. `sitegroup` 끄기

### 1.5 보충 — 다른 장면에서도 같은가 (03:0x KST)

- **입력:** SafeLIBERO observe run t3 e2 · e4 (VLA 충돌 episode) 의 obs 30 · 60 · 90 · 120, head · wrist (16 장).
- **방법:** 같은 세 mask 를 같은 방식으로 쟀다 (`mask_stats_more.py`, `outputs/2026-11-08_AG3S_review_esdf/robot_mask_more_<stem>.json`).

| mask | head recall | wrist recall | FP (px / 장) |
|---|---|---|---|
| sphere | 0.989–0.994 | 1.000 | **46–1254** (wrist 손 근처가 큼) |
| RB-Y1 방식 render | **0.64–0.76** | 0.995–0.999 | 77–827 |
| **Panda 수정 render** | 0.958–0.971 | 0.997–1.000 | **0–91** (16 장 중 9 장이 0) |

- 결론은 §1.3 과 같다.
- 하나가 더 보인다: sphere mask 는 손 근처의 실제 물체 pixel (쥔 그릇 · 목적지 등) 을 장당 최대 1254 px 지운다. 이 pixel 은 TSDF 에 들어가지 않는다.
- sphere ∪ render 규칙에서는 이 FP 가 그대로 남는다. render 단독이면 FP 가 0–91 px 로 준다 (TSDF 쪽 영향은 미측정).
- **sphere FP 의 물체별 내역** (같은 16 장 × 2 episode, `sphere_fp_objects_<stem>.json`):

  | episode | wrist: target 그릇 | wrist: 장애물 | head: 탁자 · 고정물 | head: target 그릇 | head: 장애물 |
  |---|---|---|---|---|---|
  | t3 e2 | 1626 | 46 | 845 | 342 | 82 |
  | t3 e4 | 848 | **1210** | 819 | 110 | 45 |

  - sphere mask 는 손 근처의 target 그릇뿐 아니라 **장애물 pixel 도 지운다.** t3 e4 wrist 에서 4 장 합 1210 px 다.
  - 지워진 pixel 은 TSDF 에 들어가지 않으므로, 손 바로 옆 장애물 표면이 거리장에서 빠진다. §17 의 HVC 와 같은 방향의 문제다.

---

## 2. Attention map 추출 (1) · target RGB 매핑 (2) — 판정: **매핑은 잘 됨. attention 은 접근 단계에서만 target 을 가리키고, 쥔 뒤에는 목적지를 가리킨다 (주의)**

### 2.1 무엇을 했나 (2026-11-08 01:0x–01:1x KST)

- **질문 1:** 서버가 쓰는 attention (`action:last:L8H2`) 이 실제 target 위에 있는가?
- **질문 2:** 16×16 grid 를 OpenCV 이미지로 옮기는 규칙 (`attention.unmirror_grid`: 열 뒤집기) 이 맞는가?
- **입력:** SL3 attention audit run `20261104T055314_…_nb1104-probe-attention-audit-dev-e1-4`.
  - pi0.5 단독 dev rollout `…_nb1104-probe-dev-pi05only-t0-9e1-4` 의 call 마다 V11AB attention 경로를 다시 돌린 기록이다.
  - 경로: 두 번째 checkpoint copy, denoise step 0, 집계 `last`, cell [8, 2].
  - request ↔ episode join 은 state · prompt · 두 이미지 모두 bit 단위로 검사됐다.
  - `libero_spatial` task 0–9 × init 1–4 = **40 episode · 839 call**.
- **GT:** `…_nb1104-hsr6-panda-labels-remade/phase/phase_<stem>.npz` 의 **image instant segmentation** `segP/<cam>`. 각 call frame 의 이미지와 같은 순간이다.
  - 역할: BDDL target, 같은 범주의 다른 instance (distractor, 예: 두 번째 검은 그릇), 목적지, 다른 물체, 로봇.
  - phase 는 `libero_phases.phase_labels` 를 쓴다: MuJoCo 손가락–target 접촉, lift z0 + 10 mm.
- **지표:** attention 을 camera block 합 1 로 정규화하고, GT pixel 을 16×16 patch 덮임률로 바꿔 역할별 **mass** 를 잰다.
  - 256 → 224 resize 는 정사각형이라 padding 이 없고, patch 하나 = OpenCV 16×16 px 이다.
  - 기준선 = 균일 attention 의 mass = 그 역할의 pixel 비율.
  - 대조 = mirror (열을 뒤집지 않음).
- **추가:** audit 의 `pooled_f16` (모든 layer 18 × head 8) 로 같은 지표를 잰다.
- 스크립트: `outputs/2026-11-08_AG3S_review_attention/{attention_localization.py, figures_attention.py}`

### 2.2 결과

![phase](../../../outputs/2026-11-08_AG3S_review_attention/attention_mass_by_phase.png)

| camera · 매핑 | approach target mass (균일) | approach argmax=target | approach distractor mass | transport target · 목적지 mass |
|---|---|---|---|---|
| head · **unmirror** | **0.210** (0.012) | 0.28 | 0.076 | 0.022 · **0.564** |
| head · mirror | 0.025 (0.012) | 0.02 | 0.001 | 0.001 · 0.000 |
| wrist · **unmirror** | **0.368** (0.068) | 0.37 | 0.003 | 0.023 · **0.618** |
| wrist · mirror | 0.205 (0.068) | 0.23 | 0.001 | 0.416 · 0.135 |

예 (task 0, 같은 범주 검은 그릇 두 개):

![t0](../../../outputs/2026-11-08_AG3S_review_attention/attention_example_t0e1.png)

task 별 (head, unmirror, approach + grasp):

| task | target mass / 균일 | argmax target | argmax distractor |
|---|---|---|---|
| 0 | 0.229 / 0.010 | 0.31 | **0.33** |
| 1 | 0.325 / 0.008 | 0.67 | 0.00 |
| 2 | 0.090 / 0.009 | 0.09 | 0.14 |
| 3 | 0.145 / 0.013 | 0.26 | 0.15 |
| 4 | 0.066 / 0.017 | 0.11 | 0.15 |
| 5 | 0.205 / 0.008 | 0.16 | 0.00 |
| 6 | 0.163 / 0.015 | 0.20 | 0.00 |
| 7 | **0.022** / 0.006 | 0.00 | 0.07 |
| 8 | 0.274 / 0.011 | 0.34 | 0.08 |
| 9 | 0.225 / 0.015 | 0.16 | 0.00 |

모든 [layer, head] (in-sample, 같은 40 episode):

![cells](../../../outputs/2026-11-08_AG3S_review_attention/attention_cells_target_mass.png)

| camera · phase | L8H2 target mass | L8H2 순위 / 144 | 최고 cell |
|---|---|---|---|
| head · approach | 0.212 | 2 | L12H3 0.441 |
| wrist · approach | 0.368 | 1 | — |
| head · transport | 0.022 | 81 | L4H4 0.159 |
| wrist · transport | 0.023 | 140 | L15H2 0.625 |

### 2.3 판정과 이유

1. **RGB 매핑 (unmirror): 잘 됨.**
   - head 에서 unmirror 는 target mass 가 균일의 18 배다. mirror 는 2 배로 우연 수준이다 (예: mirror 의 attention 은 cabinet 위).
   - wrist 에서도 unmirror 가 맞다. approach target (0.37 대 0.21) 과 transport 목적지 (0.62 대 0.14) 가 모두 unmirror 쪽에서 높다.
   - wrist transport 에서 mirror 의 target mass 가 높은 것 (0.42) 은 목적지 attention 이 좌우 반전되어 손 안의 그릇에 겹친 결과다. wrist 화면에서 그릇과 접시가 좌우로 붙어 있기 때문이다.
   - audit 의 join 검사 (policy 입력 이미지 = client 전처리, bit 동일) 와 합치면 policy grid → OpenCV grid 경로에 오류가 없다.
2. **Attention 추출 (L8H2): approach 단계에서는 잘 됨.**
   - head target mass 가 균일의 18 배, wrist 는 5 배다.
   - L8H2 는 head approach 2 위, wrist approach 1 위 cell 이다.
3. **안 되는 곳 1 — 쥔 뒤에는 target 을 보지 않는다.** grasp · transport 에서 L8H2 mass 는 목적지 (plate) 로 간다 (head 0.43 → 0.56, wrist 0.62).
   - 이것은 action query 의 attention 이 "지금 가야 할 곳" 을 보기 때문으로 해석한다.
   - 결과: 쥔 물체의 identity 를 attention 으로 계속 유지할 수 없다. latch · memory 가 필요하다. 다만 §16 에서 latch 가 틀린 첫 identity 를 고정하는 문제가 확인됐다.
4. **안 되는 곳 2 — 같은 범주 distractor.**
   - task 0 은 distractor argmax 0.33 이 target 0.31 을 넘는다.
   - task 2 · 4 · 7 은 target argmax ≤ 0.11 이고, task 7 은 target mass 0.022 로 사실상 못 찾는다.
   - G78 의 "noun cell 은 검은 그릇 둘을 못 가림" 과 같은 방향의 한계가 action cell 에도 있다.
5. **Float16 진단 grid 는 정밀 비교에 쓰지 않는다.** L8H2 의 f32 대 f16 target mass 차이가 최대 0.116 이다. 그래서 cell 순위는 경향만 본다.

### 2.4 한계

- head 선택이 in-sample 이다 (L8H2 는 같은 dev 40 episode 로 SL4b 에서 고른 cell).
- noun-row attention (`noun_target:L8H6`) 기록은 이 기계에 없어 비교하지 못했다.
- 측정은 pi0.5 단독 rollout 의 attention 이다. AG3S 가 개입한 rollout 의 attention 분포는 다를 수 있다.

### 2.5 보충 — P0m 의 noun head (`noun_target:L8H6`) (11:0x–11:2x KST, GPU)

- **방법:** dev-ec 의 GPU 작업이 끝난 뒤 (10:4x), 같은 dev 40 episode · 839 call 에서 noun-row attention 을 추출했다 (`attention_audit.py --noun-rows`, run `20261108T110245_ff21caa5042b_dev49-noun-rows-dev-e1-4`, layer 8–12, GPU lock 아래 약 20 분, peer 에 알림).
  - layer 8 · head 6 을 §2 와 같은 지표로 쟀다 (`rows_noun_L8H6.jsonl`).

| camera · phase | target mass (균일) | argmax = target | distractor mass | argmax = distractor | 목적지 mass |
|---|---|---|---|---|---|
| head · approach | **0.450** (0.012) | 0.59 | **0.239** | 0.31 | 0.006 |
| head · grasp | 0.326 | 0.51 | 0.161 | 0.23 | 0.042 |
| head · transport | 0.210 | 0.38 | 0.151 | 0.20 | 0.125 |
| wrist · approach | 0.731 (0.068) | 0.84 | 0.008 | 0.01 | 0.002 |
| wrist · transport | **0.898** (0.177) | 0.97 | 0.013 | 0.01 | 0.020 |

- **판정:** noun head 는 action L8H2 (head approach 0.21) 보다 target 을 두 배 강하게 가리킨다.
  - 쥔 뒤에도 목적지로 가지 않고 target 을 계속 본다 (wrist 0.90).
  - 대신 같은 범주 distractor 에도 0.24 가 간다 (argmax 0.31). G78 의 "noun cell 은 검은 그릇 둘을 못 가림" 이 측정으로 확인된다.
  - 즉 noun head 는 범주는 맞추고 instance 는 못 가리고, action head 는 instance 는 덜 헷갈리지만 쥔 뒤 목적지로 옮겨 간다.

## 3. Depth → point cloud backprojection (3) — 판정: **잘 됨**

### 3.1 무엇을 했나 (2026-11-08 01:1x KST)

- **질문:** 서버의 `stages.reconstruction.backproject` 가 만든 점이 실제 물체 표면 위에 있는가?
  - 이 검사로 depth 변환식 (robosuite z-depth, uint16 mm), K (주점 (W−1)/2 = 127.5), T_base_cam, pixel 규약이 함께 검증된다.
- **입력:** §1 과 같은 `libero_spatial_t0_e0` obs 32 · 48 · 64 · 96, head · wrist. scene 은 재구성본 (RGB bit 동일) 이다.
- **방법:**
  1. 서버 설정 (`range_max` 2.0, depth 0.05–3.0 m) 으로 모든 유효 pixel 을 back-project 한다.
  2. 그중 4000 pixel 을 무작위로 골라, camera 중심에서 pixel 중심 방향으로 MuJoCo `mj_ray` 를 쏜다. robosuite camera 가 그리는 geom group 1–2 만 맞고 site 는 제외한다.
  3. 그 hit 점 (base frame) 과의 거리를 잰다.
  4. 대조: 주점을 W/2 (robosuite `get_camera_intrinsic_matrix` 규약) 로 바꿨을 때.
- 스크립트: `outputs/2026-11-08_AG3S_review_backprojection/raycast_check.py`, 결과 `raycast_check.json`

### 3.2 결과

![bp](../../../outputs/2026-11-08_AG3S_review_overview/f3_backprojection_error.png)

| obs · cam | 점 수 | 오차 중앙 / p95 (mm) | > 5 mm 비율 | 그중 깊이 경계 | 경계 밖 중앙 (mm) | 주점 W/2 일 때 중앙 / p95 |
|---|---|---|---|---|---|---|
| 32 head | 51,219 | 1.07 / 2.38 | 1.3 % | 52 / 53 | 0.98 | 3.09 / 6.40 |
| 64 head | 50,342 | 1.10 / 2.44 | 1.9 % | 75 / 76 | 0.99 | 3.07 / 6.97 |
| 96 head | 50,252 | 1.13 / 2.53 | 1.9 % | 75 / 75 | 1.02 | 3.12 / 6.81 |
| 32 wrist | 60,160 | 0.31 / 0.67 | 0.25 % | 10 / 10 | 0.31 | 1.50 / 1.84 |
| 64 wrist | 60,160 | 0.31 / 0.68 | 0.55 % | 22 / 22 | 0.30 | 0.90 / 1.41 |
| 96 wrist | 60,160 | 0.31 / 0.77 | 0.35 % | 14 / 14 | 0.31 | 0.81 / 1.40 |

(obs 48 도 같은 수준: head 1.11 / 2.50, wrist 0.30 / 0.70.)

### 3.3 판정과 이유

- **잘 됨.** 오차 중앙값은 head 1.1 mm, wrist 0.3 mm 다. uint16 mm 양자화 (±0.5 mm) 와 head 거리 ~1.1 m 의 depth buffer 차이 수준이다.
- 5 mm 를 넘는 점의 **99.1 % (347 / 350, 8 view 합계, `boundary_check.json`) 는 깊이 불연속 경계** (3×3 안 깊이 차 > 10 mm) 에 있다. pixel 중심 광선이 경계 너머 표면을 맞는 sampling 차이다.
- **주점 규약:** 기록된 (W−1)/2 가 맞다. W/2 로 바꾸면 head 오차가 3 배가 된다 (3.1 mm). wire 규약 (`conventions_json.K`) 대로 쓰고 있다.
- head z-depth 에 +0.85 mm 의 작은 편향 (광선 거리 > 기록 depth) 이 있다. 5 mm voxel 보다 작아 영향이 없다고 판단한다.

### 3.4 한계

- episode 1 개, 서버 함수 (CPU numpy `backproject`) 만 검사했다.
- GPU 경로 (`stages.device_recon`, `--roi-carve-device cuda` 등) 의 back-projection 은 같은 식으로 검사하지 않았다.

## 4. Cluster · grounding (4) — GT 없는 최신 판 — 판정: **접근 단계 정답률 0.54 로 부족. task 4 · 9 는 0. 같은 범주 distractor 와 가림에 약하다**

### 4.1 무엇을 했나 (2026-11-08 01:3x–01:5x KST)

- **질문:** GT 없는 최신 판의 grounding 이 실제 target 을 고르는가? 대상 판은 `AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst-pts` (G78, `profile.py:968`) 다.
  - attention: 서버 기본 `action:last:L8H2`
  - target extent: `estimated` = ball-owned (owned-max-dist 3 voxel)
  - split: 12 mm raw · partition · supersedes-anchor, non-target evidence surface
- **방법:**
  - `profile.build_panda_ag3s` 로 그 판의 AG3S 를 만든다. `perception_replay.install_grounding_only` 로 ESDF 생성만 뺐다. cuRobo 는 GPU 가 필요하고, grounding 은 ESDF 결과를 읽지 않는다 (`tests/benchmark/test_panda_sl4b.py`).
  - 40 dev episode 의 모든 call 을 순서대로 `process_multi_debug(phase="approach", active_manipulators=["left"])` 에 넣는다. 서버 경로는 `wire_from_geom → strip_request → unpack_camera_observations` 이고, episode 마다 `reset()` 한다.
  - attention 은 audit 의 float32 L8H2 grid (unmirror) 다.
  - grasp 상태는 no-grasp 값이라 latch · identity hold 가 **돌지 않는다**. 그래서 쥔 뒤 결과는 서버 동작이 아니다.
- **채점 (GT 는 채점에만 쓴다):**
  - 두 camera 의 유효 pixel 을 back-project 하고, image instant GT 역할을 붙인다.
  - 예측 target 점마다 10 mm 안 최근접 GT 점의 역할을 붙인다.
  - `target_correct` = 다수결이 target 이고 중심 오차 ≤ 30 mm 이다 (SL4b 사전 등록 지표와 같은 정의).
- 스크립트: `outputs/2026-11-08_AG3S_review_grounding/{grounding_replay.py, summarize_grounding.py, figures_grounding.py}`

### 4.2 결과

![summary](../../../outputs/2026-11-08_AG3S_review_grounding/grounding_surface_summary.png)

| phase | call | target 냄 | target_correct | 점 precision 중앙 | 점 recall 중앙 | ball 반지름 중앙 | ball 안 GT target 비율 | ball 안 다른 물체 점 (중앙) |
|---|---|---|---|---|---|---|---|---|
| approach | 387 | 0.91 | **0.54** | 0.875 | 0.595 | 59 mm | 0.99 | 311 |
| grasp | 88 | 0.94 | 0.51 | 0.554 | 0.493 | 59 mm | 0.97 | 363 |
| transport (latch 없음 — 참고) | 364 | 0.69 | 0.02 | 0.0 | 0.0 | 61 mm | 0.0 | 758 |

- approach 다수결 역할: target 273 · distractor 47 · 목적지 16 · 다른 물체 16, target 없음 35 (`NO_ADMISSIBLE` 34).
- split 결정: 받아들임 184 · `partition_no_drop` 161 · `partition_too_small` 78 · `attention_not_discriminative` 24.

task 별 접근 + 쥐기 단계 target_correct:

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 정답률 | 0.73 | 0.64 | 0.68 | 0.72 | **0.00** | 0.32 | 0.80 | 0.61 | 0.79 | **0.00** |
| 주된 오답 | | | | | 다른 물체 · 목적지 | distractor | distractor | | distractor | **distractor 33/34** |

장면 예 (obs 25, approach):

![scene](../../../outputs/2026-11-08_AG3S_review_grounding/grounding_scene_examples.png)

- **t0:** 정답이다. target 점이 그릇 위에 있고 ball 이 그릇을 덮는다.
- **t9** ("cabinet 위의 그릇"): stove 위 distractor 를 고른다.
- **t4** ("cabinet 윗칸 안의 그릇"): grounding 은 target 을 내지 못하고, ATS surface 는 target 을 맞게 고른다.

### 4.3 판정과 이유

1. **접근 단계 target 정답률 0.54: 부족하다.** 틀린 call 178 개 중 distractor 다수결 47 · target 없음 35 · 목적지 · 다른 물체 32 이고, 64 개는 다수결은 target 이지만 중심 오차 > 30 mm 다.
2. **task 9 (0/44):** 주어진 L8H2 attention 이 처음부터 distractor 를 가리킨다. §2 의 task 9 distractor mass 는 낮았지만, grounding 의 percentile seed (95 %) + DBSCAN 다수 cluster 가 distractor cluster 를 고른다.
3. **task 4 (0/53):** target 이 cabinet 서랍 안에 있다. 그릇 점이 cabinet 점과 한 cluster 로 이어져 (eps 30 mm), 잡을 수 없는 크기 → `NO_ADMISSIBLE` 이 되거나 다수결이 cabinet 이 된다. 같은 frame 에서 ATS surface 는 그릇을 맞게 고른다 (그림 3 행).
4. **Ball extent:** target 을 99 % 덮는다. 대신 반지름 ~59 mm 라서 다른 물체 점도 중앙값 311 개를 같이 덮는다. 이 점들이 target-free 층에서 빠지면 권한 link 가 그 물체를 통과할 수 있다. ESDF 에서 실제로 무엇이 빠지는지는 `ball-owned` 가 owned seed 만 지우므로 다르다 → §8 에서 본다.
5. **쥔 뒤:** attention 이 목적지로 가서 (§2) latch 없이는 grounding 이 목적지를 고른다 (transport 0.02). 서버는 latch (`latched_identity_hold`) 로 이것을 막도록 되어 있다. 이 replay 는 latch 를 돌리지 않았다 → latch 를 흉내 낸 결과는 §16 (identity 고정 문제).

### 4.4 한계

- grasp state 는 no-grasp 값이고, AG3S 개입이 없는 pi0.5 단독 rollout 의 frame 이다.
- `NO_ADMISSIBLE` 을 target 없음으로 셌다.
- 정답 기준 (중심 ≤ 30 mm) 은 SL4b 정의를 따랐다.

### 4.5 보충 — 두 번째 support 평면이 grounding 에도 영향을 주는가 (02:4x KST)

- **질문:** §5 의 원인 (support 평면 2 개) 이 GT 없는 판의 grounding 에도 있는가?
  - grounding 도 같은 `fit_support_surfaces` (`max_planes` 2) 로 support 점을 연결 · seed 에서 뺀다.
- **방법:** 같은 40 episode 를 다시 replay 했다. 접근 · 쥐기 call 475 개에서 fused cloud 의 GT target 점 중 `support_mask` 비율을 쟀다 (`support_in_grounding.py`, `support_rows.jsonl`).

| | support 로 빠지는 비율 (중앙) |
|---|---|
| target 그릇 점 | **0.26** (p75 0.39; task 0 · 1 · 2 · 8 은 0.37–0.44) |
| 같은 범주 distractor 점 | 0.06 |
| 목적지 (접시) 점 | 0.98 |

| target 점 support 비율 | call | target_correct |
|---|---|---|
| 0–0.1 | 187 | 0.27 (대부분 task 4 · 9 · 5 · 7: 높은 곳 · 서랍 속 그릇이라 평면과 무관) |
| 0.1–0.3 | 68 | 0.78 |
| 0.3–0.5 | 203 | 0.69 |
| ≥ 0.5 | 17 | 0.65 |

- **판정:** grounding 에서도 탁자 위 그릇의 ~1/4–2/5 가 support 로 빠진다. target 점 recall 0.6 (§4.2) 의 주된 원인으로 본다 (상관이며, 평면 1 개 반사실로 grounding recall 을 다시 재지는 않았다). identity (어느 물체인가) 는 크게 망가뜨리지 않는다.
- 다만 ball extent 가 남은 위쪽 점으로 맞춰지므로, 그릇 아래쪽이 target-free 층에서 덜 지워질 수 있다 (§12 의 "target 표면 ~70 %" 와 같은 방향, 미분해).

### 4.6 closed-loop 확인 — GT 없는 판 smoke run (05:2x KST)

- **입력:** dev-ec 의 smoke run `20261108T032134_…gtto-smoke-att-obst4pts…` (`AG3S-commonto-att-…-obst4-pts`, LIBERO t0 e0 · t4 e0, 03:21) 이다. constraint 기록이 없어 call 기록만 봤다.

| episode | 결과 | call | HOLD | action 바뀐 call | `ag3s_status` | latch 흐름 | pass-through 사유 |
|---|---|---|---|---|---|---|---|
| t0 e0 | 성공 | 21 | 0 | 0 | ok 18 · no_target 3 | S 2 → L 6 → C 10 → H 3 | clear 21 |
| **t4 e0 (서랍)** | **실패 (max_steps)** | 44 | **8** | **12** | ok 36 · **no_target 8** | **S 10** → L 13 → C 7 → L 11 → C 3 (HELD 없음) | clear 32 · **geometry not certified 8** · obstacle 4 |

**판정:** offline replay 의 예측 (§4.2: t4 grounding 정답 0 / 53, 서랍 + 그릇이 한 cluster) 이 closed-loop 에서 그대로 나타났다.

- target 을 확정하지 못해 latch 가 10 call 동안 SEARCHING 이었다.
- `no_target` call 이 '인증 안 됨' 으로 HOLD 되었다.
- VLA 단독은 이 episode 를 성공한다 (F12 · VLA LIBERO 27 / 27).
- 표본은 2 episode 다. GT 없는 판의 본 평가는 dev-ec 의 queue (`gtto20_queue.sh`) 에 있다.

---

## 5. Target 표면 분할 (ATS surface, `--target-extent attention-surface`, P0m 판) — 판정: **안 됨 (쥐기 전). 원인은 두 번째 support 평면이 target 을 삼키는 것**

### 5.1 무엇을 했나 (2026-11-08 01:2x–01:4x KST)

- **질문:** P0m 의 target extent (`runtime.attention_surface.select_target_surface`, LOCK arm `top1-only-ns:N`) 가 target 표면을 맞게 고르는가?
  - 서버는 `device="cuda"` 로 돈다. 여기서는 같은 계산의 CPU chain 을 썼다 (module docstring).
  - 서버와 같은 입력: head · left_wrist + 서버 sphere robot mask + right_wrist dummy (`pad_missing_views`).
- **입력 조건:** 같은 40 episode · 839 call 에 두 attention 을 넣었다.
  - (a) **GT attention** — GT target 의 patch 덮임률이다. 특권 입력이며, 표면 알고리즘만 본다.
  - (b) **L8H2** — 기록된 attention 이고, end-to-end 결과다.
- **진단:**
  - `keep_debug` 로 GT target node 의 support 분류 비율과 해 m 을 쟀다 (126 frame).
  - 반사실 실험: graph 의 support 평면을 1 개 (`max_planes=1`) 로 바꿔 같은 전체를 다시 돌렸다. **진단 전용이고 서버 변경이 아니다.**
- 스크립트: `outputs/2026-11-08_AG3S_review_ats/{ats_surface_replay.py, ats_surface_replay_1plane.py, diagnose_surface.py, summarize_ats.py}`, 결과 `ats_summary.json`, `diagnose_surface.json`

### 5.2 결과

| 조건 | phase | 적용 (abstain 아님) | head IoU (abstain=0) | head precision / recall (적용 frame 중앙) | wrist IoU |
|---|---|---|---|---|---|
| GT attention · 평면 2 (서버) | approach | 0.55 | 0.23 | 0.99 / **0.22** | 0.22 |
| GT attention · 평면 2 (서버) | grasp | 0.83 | 0.30 | 1.00 / 0.18 | 0.37 |
| GT attention · 평면 2 (서버) | transport | 0.95 | 0.81 | 0.98 / 0.96 | 0.89 |
| GT attention · **평면 1 (진단)** | approach | **0.82** | **0.43** | 0.98 / **0.88** | 0.49 |
| GT attention · **평면 1 (진단)** | grasp | 1.00 | 0.59 | 0.98 / 0.87 | 0.73 |
| L8H2 · 평면 2 (서버) | approach | 0.46 | 0.15 | 0.14 / 0.02 | 0.18 |
| L8H2 · 평면 2 (서버) | transport | 0.66 | 0.53 | 0.98 / 0.96 | 0.57 |
| L8H2 · 평면 1 (진단) | approach | 0.68 | 0.26 | 0.87 / 0.13 | 0.30 |
| L8H2 · 평면 1 (진단) | transport | 0.33 | 0.27 | 0.98 / 0.97 | 0.29 |

abstain 사유는 모두 `no_region` 이다.

진단 (GT attention, approach · grasp 126 frame):

| | frame | GT target node 중 support 비율 (중앙) | seed 가 target 위 | target node 의 m 최대 (중앙) | target node 중 m ≥ 0.5 |
|---|---|---|---|---|---|
| `no_region` | 52 | **0.43** | 1.00 | **0.27** | 0.00 |
| 적용 | 74 | 0.08 | 1.00 | 1.00 | 0.37 |

두 번째 평면의 정체 (`no_region` 6 frame):

- 평면 1 은 탁자다 (z = −0.011 m, base).
- 평면 2 는 법선이 1–3° 기운 평면 (z ≈ +0.01 m) 이다. inlier 에 탁자 일부와 함께 **접시 369–825, target 그릇 213–279, 다른 그릇 89–140 node** 가 들어간다.
- support 로 분류된 target node 는 탁자 위 10–18 mm (최대 31 mm) 에 있다.

### 5.3 판정과 이유

1. **쥐기 전 표면 분할은 안 된다** (서버 설정, GT attention 으로도 IoU 0.23, recall 0.22, abstain 45 %).
   - target 이 탁자 위에 있는 동안은 정작 target 을 충돌 대상에서 빼야 하는 구간이다. 그런데 이 구간에서 target 의 일부만 빠진다.
2. **원인 (측정으로 확정):** surface graph 의 support fit (`fit_support_surfaces`, RANSAC, `max_planes` 2, inlier 8 mm, 크기 문턱 500 점 / 5 %) 이 LIBERO 의 낮은 물체 (접시 ~1 cm, 그릇 ~5 cm) 위에 두 번째 평면을 만든다.
   - 그 평면이 target node 의 25–43 % 를 support (unary s = 0, κ = 1) 로 만든다. 그래서 해 m 이 0.5 에 못 닿는다.
   - `support_surface.py` docstring 이 경고한 "물체 윗면을 가로지르는 sliver 평면" 이 크기 문턱을 통과한 경우다.
   - 평면 1 개 반사실에서 approach 적용 0.55 → 0.82, recall 0.22 → 0.88 로 회복한다.
3. **L8H2 를 넣으면 더 나쁘다** (approach precision 0.14). approach head 선택의 다수결이 다른 물체 37 · distractor 18 · target 87 (`ats_summary.json`) 이다. attention 이 target 밖으로 퍼지기 때문이다 (§2).
   - 평면 1 개는 approach 를 조금 낫게 하지만 transport 를 떨어뜨린다 (0.66 → 0.33 적용). 두 번째 평면이 접시를 support 로 만들어, 목적지로 간 attention 이 접시를 고르지 못하게 막던 우연한 효과가 사라지기 때문으로 보인다.
   - 그래서 `max_planes=1` 은 그대로 해결책이 아니다.
4. **쥔 뒤 (transport) 에는 잘 된다** (GT attention IoU 0.81–0.89). 들린 그릇은 support 와 떨어져 있기 때문이다.
5. **서버 판정 순서상의 의미:**
   - P0m 은 noun head (`noun_target:L8H6`) 를 쓴다. 처음에는 기록이 없어 재지 못했고, 11:2x 에 추출해 §5.7 에서 쟀다.
   - 다만 GT attention 으로도 쥐기 전 recall 0.22 이므로, 어떤 attention 이든 이 support 문제가 상한을 정한다.

### 5.4 한계

- 서버의 surface 후처리는 재현하지 않았다: `surface_target_check overlap` (grounding target 과 겹쳐야 받아들임), `surface_keep_last` (abstain 이면 지난 region), `surface_grasp_hold`, `surface_hull_fill`. per-frame 선택만 쟀다.
- GPU chain 과 CPU chain 의 동일성은 문서 주장에 기댔다.

### 5.5 보충 — 서버 후처리까지 포함한 P0m end-to-end (03:1x–03:2x KST)

- **질문:** §5.4 의 한계 (후처리 미재현) 를 메운다. 서버가 실제로 target-free 층에 쓰는 target extent 는 얼마나 맞는가?
- **입력:** profile 의 `AG3S-final-panda-p0m-l8h2` 판이다. P0m 과 같고 attention 만 L8H2 다. P0m 본판의 noun head 기록은 없다.
- **방법:**
  - `build_panda_ag3s` 그대로, grounding-only 로 ESDF 만 뺐다. surface 선택 · 후처리 · memory 는 ESDF 전에 돈다.
  - surface device 는 CPU 다 (serve_safe 만 cuda 로 바꾼다).
  - latch 는 §16 처럼 물리 진실로 흉내 냈다.
  - call 마다 **memory 의 target (code 1) sample** 을 image instant GT 와 비교했다. target-free 층은 이것으로 만들어진다.
- 스크립트: `outputs/2026-11-08_AG3S_review_grounding/p0m_replay.py`, 결과 `rows_p0m_part*.jsonl`, `p0m_summary.json`

| phase | 이번 frame 에 region 수용 | memory 에 target sample 있음 | target 표면 recall (중앙 / 평균, 없으면 0) | memory target sample 의 실제 역할 | abstain 사유 |
|---|---|---|---|---|---|
| approach (387) | **0.12** | 0.42 | 0.02 / **0.06** | target 52 % · **distractor 35 %** · 없음 8 % | `no_region` 192 · `not_ats_target` 117 · `grasp_hold` 31 |
| grasp (88) | 0.01 | 0.52 | 0.02 / 0.13 | target 54 % · 없음 20 % · distractor 20 % | `grasp_hold` 85 |
| transport (364, latch 흉내 · attach 없음) | 0 | 0.31 | 0 / 0.005 | 없음 46 % · distractor 35 % | `grasp_hold` 364 |

**판정:** 서버 경로에서 쥐기 전 target-free 층이 지우는 것은 target 표면의 평균 ~6 % 다. 나머지 ~94 % 는 손 · 손가락에게 장애물로 남는다.

- 원인은 셋이 겹친다:
  1. per-frame 선택 자체가 대부분 `no_region` (§5.2–5.3, 두 번째 support 평면)
  2. 선택돼도 `surface_target_check overlap` 이 grounding target 과 겹치지 않는다고 117 번 기각
  3. 접근 후반부터 `grasp_hold` 로 새 region 을 받지 않음
- memory 의 35 % 는 distractor 표면이다. distractor 를 손이 통과해도 되는 대상으로 지운다는 뜻이다.
- noun head (P0m 본판) 에서 attention 이 더 정확하면 2 · 3 은 줄 수 있다. 1 (support 평면) 은 attention 과 무관하다 (§5.2 GT attention 결과).

### 5.6 보충 — SafeLIBERO 장면에서는 (03:5x KST)

- **방법:** GT attention 조건은 attention 기록 없이 기하만으로 돌 수 있다. 그래서 F12 SafeLIBERO 11 episode 의 모든 call 에 같은 per-frame 선택을 돌렸다.
  - GT 는 기록의 return-time segmentation 이다.
  - phase 는 첫 양손가락 접촉 전 / 후 다.
  - 서버 설정 (평면 2) 과 1 평면 반사실을 비교했다.
  - `ats_surface_safelibero.py`, `ats_surface_safelibero.json`

| task (target 위치) | 접근 call | 평면 2: 적용 · head IoU | 평면 1: 적용 · head IoU |
|---|---|---|---|
| t1 (ramekin 위) | 16 | 0.88 · 0.71 | 1.00 · 0.86 |
| t2 (stove 위) | 50 | 1.00 · 0.66 | 1.00 · 0.88 |
| t3 (cabinet 위) | 84 | 1.00 · 0.63 | 1.00 · 0.68 |
| 전체 접근 | 150 | 0.99 · 0.65 (recall 0.96) | 1.00 · 0.76 |

**판정 보정:**

- §5 의 실패는 **target 이 탁자 위에 바로 놓인 경우** (LIBERO spatial 대부분) 에 해당한다.
  - SafeLIBERO 처럼 target 이 ramekin · stove · cabinet 위에 있으면 두 번째 평면 (탁자 위 1.6–2.4 cm) 을 피한다. 그래서 GT attention 으로 접근 단계 적용 0.99, recall 0.96 이 된다.
- 그래도 1 평면이 IoU 를 0.05–0.22 올린다. 두 번째 평면이 높은 target 의 일부도 먹는다.
- 즉 원인 (낮은 sliver 평면) 은 같고, 피해 크기는 target 높이에 달려 있다.

### 5.7 보충 — P0m 본판 head (noun L8H6) 로 서버 경로 다시 (11:2x KST, `p0m_noun_replay.py`, `p0m_noun_vs_l8h2.json`)

| attention | approach 에서 region 수용 | memory 에 target 있음 | target 표면 recall (평균, 없으면 0) | memory 중 target · distractor | abstain 사유 |
|---|---|---|---|---|---|
| action L8H2 (§5.5) | 0.12 | 0.42 | 0.06 | 52 % · 35 % | no_region 192 · not_ats_target 117 · grasp_hold 31 |
| **noun L8H6 (P0m 본판)** | **0.33** | 0.62 | **0.11** | 52 % · 35 % | no_region 99 · not_ats_target 129 · grasp_hold 31 |

**판정:** P0m 본판 head 를 넣어도 서버 경로가 쥐기 전에 지우는 target 표면은 평균 11 % 다. §5 의 판정 (안 됨) 이 P0m 본판에도 그대로 적용된다.

- attention 이 좋아지면 `no_region` 은 줄지만 (192 → 99), `not_ats_target` 기각과 두 번째 support 평면이 남는다.
- distractor 가 memory 의 35 % 인 것도 같다.

---

## 6–7. 작업 단계 (간소화판) · 쥐기 전 / 쥠 / 쥔 뒤 (grasp latch) — 판정: **attach 는 잘 됨 (열린 손 attach 0). LIBERO 1 / 27 · SafeLIBERO 1 / 11 은 reach 문턱 초과로 끝까지 attach 못 함. 놓기 → detach 는 SafeLIBERO 1 사례에서 맞게 동작 (§6.6). PLACED 는 0 회**

### 6.1 무엇이 검증 대상인가 (코드 확인)

| 개념 | 코드 | 최신 판 (F12 = `…-gt2-…-loose2`, GT 없는 att 판도 같음) |
|---|---|---|
| grasp latch 상태 | `trajopt/grasp_latch.py` SEARCHING → LATCHED → CLOSING → HELD → PLACED | `--grasp-opening-rule` (+ (c′) `--grasp-opening-rule-moved 0.05`, sticky reach, drop-from-open) |
| attach 조건 | `_observe_opening_rule` | 마지막 실행 명령 닫힘 (< 0.85) ∧ 개도 변화 < 0.02 (settled) ∧ 개도 ≥ 0.03 ∧ reach ≤ 120 mm ∧ (c′) 열림 기준 대비 ≥ 0.05 닫힘 |
| 작업 단계 라벨 (간소화 3 단계) | `--subtask-from-latch` → `LATCH_SUBTASK` | SEARCHING / LATCHED / CLOSING → pick, HELD → place, PLACED → home |
| contact phase (target margin) | `--latch-phase` | LATCHED → pre_grasp (5 mm), CLOSING · HELD → grasp (0), 나머지 approach (20 mm) |

G77 기록대로 "3 단계 판" 자체는 구현되지 않았다. 실제로 쓰이는 간소화 라벨은 위 표의 latch → pick / place / home 이다.

### 6.2 무엇을 했나 (2026-11-08 01:4x KST)

- **입력:** F12 LIBERO 27 run (`…gtto-gt15loose2-libero9s{7,8,9}…`, 9 task × seed 3) 의 call 별 서버 latch (`calls.jsonl` `grasp.state`).
- **물리 진실:** `phase_export.py --finger-contacts` 가 rollout env 이력을 재현하고 (call frame RGB bit 동일 확인), 매 frame 좌 · 우 손가락–target 접촉, target z 를 MuJoCo 로 계산한 것.
  - **truth 상태:** searching = close 명령 전, closing = close 후 양손가락 접촉 전, grasped = 양손가락 접촉, released = 들어 올린 뒤 마지막 양손 접촉 이후.
- 스크립트: `outputs/2026-11-08_AG3S_review_latch/{latch_vs_truth.py, figure_latch.py}`, 결과 `latch_vs_truth.json`, `latch_rows.jsonl`

### 6.3 결과

![latch](../../../outputs/2026-11-08_AG3S_review_latch/latch_timeline.png)

call 별 (594 call):

| truth \ latch | searching | latched | closing | held |
|---|---|---|---|---|
| searching | 54 | 205 | 0 | 0 |
| closing | 0 | 1 | **78** | 1 |
| grasped | 0 | 0 | 47 | **208** |
| released | – | – | – | – (LIBERO 는 성공하면 놓은 직후 episode 가 끝나 call 이 없다) |

| 항목 | 값 |
|---|---|
| attach 26 번 중 양손가락 접촉 상태 | **26 / 26** (열린 손 attach 0) |
| attach 지연 (첫 양손 접촉 → HELD) | 중앙 9 frame (≈ 2 call), 1–44 |
| attach 시점 − lift 시점 | 중앙 −3 frame (들기 전에 attach) |
| 쥐고 있는데 HELD 아님 | 47 call: `settled` 실패 36 (손가락이 아직 움직임) · `reach` 실패 11 |
| 접촉 없이 HELD | 1 call (s9 t1 obs 100, 순간 접촉 끊김) |
| 끝까지 attach 못 한 episode | **1 (s8 t4)**: obs 65 부터 양손으로 쥐고 들어 옮겼으나 reach 120.7 mm > 문턱 120 mm (0.7 mm 초과) |
| 단계 라벨 pick / place 일치 | truth pick → pick 375 · place 68 (쥔 뒤 들기 전), truth place → place 141 · pick 10 (attach 늦음), home 은 양쪽 다 없음 |
| contact phase | latched → pre_grasp 179, closing · held → grasp, 그러나 searching → latched 로 바뀌는 call (episode 당 1, 27 call) 은 approach. contact phase 가 latch 갱신보다 한 call 늦다 |

### 6.4 판정과 이유

1. **attach (쥠 판정): 잘 됨.**
   - 26 / 26 attach 가 양손가락 접촉 상태에서 일어났다. F8 의 열린 손 attach 문제 (G70) 가 (c′) 이후 사라진 것을 물리 진실로 확인했다.
   - attach 는 들기 전 (중앙 3 frame) 에 일어난다.
2. **안 되는 곳 — reach 문턱 경계:**
   - s8 t4 는 0.7 mm 차이로 HELD 가 되지 않았다. 그래서 쥔 그릇이 운반 내내 '쥔 물체' 가 아니라 target 으로 field 에 남았다.
   - 원인은 reach (`reach_limit_mm` 120) 가 쥔 물체 geometry 중심까지의 거리라는 점이다. 그릇 (반지름 ~60 mm) 을 서랍에서 꺼내는 자세에서는 손–중심 거리가 문턱 근처다.
   - LIBERO 는 TO 가 개입하지 않아 성공했지만 (F12 LIBERO 개입 0), 장애물이 있는 장면이면 HOLD · 개입 원인이 된다.
3. **attach 지연:** 쥐고 있는 47 call 은 CLOSING 이다 (settled · reach). 이 동안 contact phase 는 grasp (margin 0) 라 target 쪽 문제는 작다. 쥔 물체는 아직 robot 으로 접히지 않는다.
4. **놓기 · PLACED · home 라벨: 이 데이터로는 검증할 수 없다.**
   - LIBERO episode 는 놓는 순간 성공으로 끝난다.
   - 코드상 opening rule 의 detach 는 HELD → **LATCHED** 로 가서 PLACED 에 닿지 않는다 (`grasp_latch.py:849-862`, G77 "PLACED 0 회" 와 일치). 따라서 `home` 라벨과 `--placed-no-seed-certify` 는 이 판들에서 실제로 쓰이지 않는다 (코드 추론, 측정 아님).
5. **contact phase 한 call 지연 (27 call):** LATCHED 첫 call 에서 approach margin (20 mm) 이 남는다. 접근 중이므로 영향은 작다고 본다.
6. **(03:3x 기록 확인) 코드 추론 두 가지를 F12 전체 1516 call (LIBERO · SafeLIBERO · LIBERO-Safety) 기록으로 확인했다** (`exemption_counters.json`).
   - `--closing-hand-exempt` 는 사실상 꺼져 있다. CLOSING 358 call 에서 `to.grasp_contact.closing_exempt_rows` 가 모두 0 이다. opening rule 의 검사 키에 `closed · not_empty · blocked` 가 없기 때문이다.
   - PLACED 상태는 0 회다. 그래서 `--placed-no-seed-certify` · `home` 라벨도 실제로 쓰이지 않는다.

### 6.5 한계

- GT 판 (F12) 기록이다. latch 규칙은 GT 없는 판과 같다.
- 다만 GT 없는 판의 manipulated geometry 는 grounding ball (§4) 이라 reach 값이 달라질 수 있다 (미측정).
- SafeLIBERO · LIBERO-Safety 의 놓기 동작은 이 절에서 재지 않았다.

### 6.6 보충 — SafeLIBERO F12 11 episode (03:4x KST)

- **방법:** SafeLIBERO fork 를 활성화하는 wrapper (`phase_export_safelibero.py`) 로 같은 접촉 track 을 만들었다 (11 / 11 RGB bit 동일). 그 뒤 §6.2 와 같은 정의로 대조했다 (`latch_vs_truth_sl.py`, `latch_vs_truth_safelibero.json`).

| truth \ latch | searching | latched | closing | held |
|---|---|---|---|---|
| searching | 22 | 74 | 0 | 0 |
| closing | 0 | 6 | 78 | 3 |
| grasped | 0 | 2 | 33 | **171** |
| released | 0 | **14** | 0 | 0 |

![sl-latch](../../../outputs/2026-11-08_AG3S_review_overview/f66_latch_safelibero.png)

- **attach:** 10 / 10 이 양손가락 접촉 상태에서 일어났다. 손가락 접촉이 하나도 없는 HELD 는 0 이다. truth closing → latch held 3 call 은 한쪽 손가락만 닿은 순간이다.
- **놓기 (t3 e3):** 그릇이 놓인 지 (t = 227) 3 frame 뒤 detach (230) 됐고, 이후 14 call 이 LATCHED 다. 놓기 판정이 맞게 동작한 첫 사례다.
- **attach 못 함 1 / 11 (t3 e2):** 손가락이 자리를 잡은 순간부터 reach 가 122.7 mm (문턱 120) 라 끝까지 attach 하지 않았다. 그릇은 들려서 옮겨졌다.
  - LIBERO s8 t4 와 같은 **reach 경계 실패** 이고, 두 벤치 38 episode 중 2 개다.
- **attach 지연 (t3 e0 92 frame · t2 e4 64 · t2 e1 43):** VLA 가 쥐었다 놓았다를 반복해서 (개도 0.41 → 0.30 → 0.94 → 0.10) `settled` 가 기다린 것이다. 판정 자체는 맞다.

### 6.7 판을 넘어 본 reach 문턱 (07:2x KST, `reach_margin.json`)

- **입력:** F12 (gt15loose2) · gt19pts · B3 (gt21pts) 의 모든 call 기록 (각 LIBERO 27 + SafeLIBERO 11 + LIBERO-Safety 10) 이다.

| 판 | attach 수 | attach 순간 reach 중앙 / p90 / p99 / 최대 (mm) | > 110 mm | CLOSING 까지 가고 끝내 attach 못 한 episode |
|---|---|---|---|---|
| F12 | 44 | 73 / 105 / 118 / 118.9 | 4 | 4 (SafeLIBERO t3 e2 · LIBERO t4 e0 · LS affordance t0 e0 · e2) |
| gt19pts | 46 | 73 / 95 / 118 / 118.9 | 3 | 4 |
| B3 | 42 | 74 / 93 / 119 / 118.9 | 4 | 6 (+ LIBERO t6 e0 · LS human_safety t0 e0) |

- 끝내 attach 하지 못한 episode 에서 attach 를 막은 검사는 모두 **reach** (> 120 mm) 가 가장 많다. 손가락이 자리를 잡는 (`settled`) 순간 reach 가 문턱을 넘는다.
- LIBERO t4 e0 (서랍) 과 LIBERO-Safety affordance t0 e0 · e2 는 세 판 모두에서 같은 이유로 attach 하지 못했다. 장면 구조 (손 – 쥔 물체 중심 거리) 에 묶인 실패다.
- 성공한 attach 의 p99 도 118 mm 로 문턱에 붙어 있다. 120 mm 는 이 그릇들에 대해 여유가 거의 없는 값이다.
- **제안 (구현하지 않음):** reach 를 물체 중심이 아니라 손가락 사이 물체 표면 기준으로 재거나, 문턱에 그릇 반지름만큼 여유를 둔다.

### 6.8 빈손으로 닫힌 CLOSING 의 deadlock (B5 gt23t100 SafeLIBERO t3 e0, 08:0x KST)

- **현상:** B5 의 이 episode 는 obs 155–295 동안 HOLD 가 29 call 이어졌다. 사유는 모두 `geometry not certified` 다.
- **기록:**
  - latch 는 CLOSING 49 call 이다.
  - 검사 실패는 `opening_min · reach` 32 call 이다. 개도는 0.01 로 빈손으로 끝까지 닫혔고, reach 는 250–453 mm 로 그릇에서 멀다.
  - ag3s 상태는 obs 175 부터 `degraded` 다.
- **원인 (코드 · 기록):**
  - opening rule 에서 CLOSING → LATCHED 는 명령과 개도가 모두 ≥ 0.85 로 다시 열려야만 일어난다.
  - 쥐기를 놓쳐 빈손으로 닫힌 경우 VLA 가 다시 열지 않으면 CLOSING 에 갇힌다.
  - 그동안 target 이 인증되지 않아 HOLD 가 되고, HOLD 동안 로봇이 멈춰 다시 열 기회도 없다.
- **판정:** latch 상태 기계의 빠진 출구다. "빈손으로 닫힘 (개도 < `opening_min`, reach 큼) 이 지속되면 LATCHED 로 되돌린다" 같은 규칙이 없다 (제안, 구현하지 않음).
  - 세 판 중 이 판에서만 나타났지만 상태 기계 구조상 어느 판에서나 생길 수 있다.
- **(08:4x 추가) `degraded` 의 원인:** dev-ec 기록상 G3 `esdf_dead_camera` 다. obs 155 부터 left_wrist 카메라가 거리장에 voxel 을 하나도 보태지 않았다.
  - dev-ec 는 "손이 병을 눌러 손목 depth 가 범위 밖" 으로 설명했다.
  - 기록의 손목 depth 는 유효 (0.05–3 m) 비율 0.51–0.53, 중앙값 2.84–2.87 m 다. 즉 손목이 가려진 것이 아니라 먼 곳을 보고 있었다. 반경 `range_max` 2.0 m · 작업 공간 밖이라 voxel 이 0 이 된 것으로 보인다 (추정).
  - 어느 쪽이든 고리는 같다: 쥐기 실패 → 손목 카메라 기여 0 → 인증 실패 → HOLD. 그래서 CLOSING 출구만으로는 이 HOLD 가 풀리지 않는다.

---

## 8. TSDF / ESDF — 판정: **보이는 표면 근처는 잘 됨 (충돌 직전 장애물을 보수적으로 잡음). 안 보이는 면 (탁자 · cabinet) 은 거리장에 없다**

### 8.1 무엇을 했나 (2026-11-08 01:5x–02:2x KST)

- **질문:** 서버의 거리장이 실제 장면 기하와 맞는가? 특히 VLA 가 장애물에 부딪히기 직전, 로봇 sphere 위치에서 장애물을 보는가?
  - 서버 거리장은 cuRobo backend 다: TSDF 5 mm → fine ESDF 5 mm (128³ 창) + coarse 20 mm.
- **입력:**
  - SafeLIBERO observe-only F8 기록 `…gtto-gt12obs-safelibero11s10-t3-LI…` 의 e2 · e3 · e4. VLA action 이 그대로 실행되고 AG3S 는 관찰만 한다. 세 episode 모두 VLA 가 장애물에 충돌했다 (책 t = 38, 와인병 35, 머그 36).
  - 같은 방식으로 t2 e0 (moka pot, 충돌 t = 77) 도 넣었다.
  - `--record-constraints full` chunk 에는 fine · coarse · target-free ESDF 격자, constraint sphere 170 개 (TO 해의 0 단계 = 현재 자세), 계획 지평 clearance 가 있다.
  - t3 e0 · e1 은 LIBERO 배치 오류 (`RandomizationError`) 로 재구성하지 못했다.
- **GT:**
  - 재구성 장면 (SafeLIBERO fork 활성화 + rollout env 이력, RGB bit 동일) 의 로봇이 아닌 visual geom 표면을 ≤ 3 mm 간격으로 sampling 했다. latch HELD 동안은 쥔 물체도 뺐다.
  - KD-tree 무부호 거리 (정확도 ~1.5 mm) 를 쓴다.
- **비교:**
  - (a) fine 격자 voxel (|ESDF| ≤ 0.15 m 6000 개 + 균일 2000 개 / chunk)
  - (b) sphere 별 clearance: 기록값 = 서버 `cs.esdf.distance` − r. 내가 격자로 다시 계산한 값과 차이 1e-5 mm 로 같은 값인 것을 확인했다.
- 스크립트: `outputs/2026-11-08_AG3S_review_esdf/{export_scene.py, esdf_vs_truth.py, figures_esdf.py, slice_figure.py}`

### 8.2 결과

![esdf](../../../outputs/2026-11-08_AG3S_review_esdf/esdf_vs_truth_summary.png)

![slice](../../../outputs/2026-11-08_AG3S_review_esdf/slice_e4_obs35.png)

voxel (**t2 e0 한 episode**, `esdf_vs_truth_safelibero_spatial_t2_e0.json`. t3 세 episode 합은 `esdf_vs_truth_summary.json` `voxel_bins` 에 있고 같은 경향이다. 3–10 cm 못 봄 31 % · 부풂 11 %):

| 실제 거리 | ESDF − 실제 중앙 | ESDF 가 10 mm 넘게 큼 (못 봄) | ESDF 가 10 mm 넘게 작음 (부풂) |
|---|---|---|---|
| 0–1 cm | +2.6 mm | 41 % | 4 % (아래 3 에서 정정) |
| 1–3 cm | −2.6 mm | 32 % | 24 % (정정: free space 만 보면 6–7 %) |
| 3–10 cm | −1.6 mm | 35 % | 9 % |

(부풂 열은 물체 내부 voxel 이 섞인 값이다. free space 만 다시 센 값은 아래 §8.3 의 3 (t3 e2 · e4).)

| 물체 (표면 1 cm 안) | 못 보는 비율 |
|---|---|
| 탁자 | 0.46 |
| cabinet | 0.41–0.43 |
| 와인병 | 0.30 |
| target 그릇 | 0.17 |
| 머그 | 0.16 |
| 책 | 0.11 |
| stove | 0.02 |

충돌 직전 현재 자세 (장애물에 가장 가까운 sphere):

| episode (장애물, 충돌 t) | obs | 실제 여유 | 서버 거리장 여유 |
|---|---|---|---|
| e2 (책, 38) | 30 | 53.2 mm | 33.6 mm |
| e2 | 35 | 15.8 mm | 17.3 mm |
| e3 (와인병, 35) | 25 | 82.7 mm | 76.0 mm |
| e3 | 30 | 40.7 mm | 35.4 mm |
| e4 (머그, 36) | 30 | 32.8 mm | 27.4 mm |
| e4 | **35** | **−0.2 mm** | **−11.0 mm** |

### 8.3 판정과 이유

1. **보이는 표면 근처는 잘 된다.**
   - 표면 3 cm 안 오차 중앙값이 ±3 mm 다 (5 mm voxel 수준).
   - 충돌 직전 6 시점 중 5 시점에서 거리장이 실제보다 작거나 같게 (보수적으로) 장애물을 봤다.
   - 단면 그림에서 0 등고선이 실제 머그 테두리와 겹친다.
   - G71 의 "F8 이 충돌 전 40 step 안에 개입했을 것 10 / 10" 과 맞는 기하적 근거다.
2. **안 보이는 면은 거리장에 없다.** 탁자 (물체 밑 · 시야 밖) 와 cabinet (옆 · 뒷면) 근처 voxel 의 41–46 % 에서 거리장이 실제보다 10 mm 넘게 멀다고 본다.
   - 관측한 깊이만 적분하는 TSDF 의 성질이고 버그가 아니다.
   - 다만 cabinet 처럼 큰 고정물의 뒷면 · 옆면에 팔이 다가가면 거리장이 늦게 안다.
3. **(02:5x 정정) 부푼 쪽 (실제보다 가깝게 봄).**
   - 처음 표의 "1–3 cm 대 24 %" 는 내 GT 의 실수가 섞인 값이었다. 무부호 거리라 탁자 상판 · cabinet **내부** voxel 도 "표면에서 멀다" 로 셌다.
   - 모든 물체의 oriented bounding box (+5 mm) 밖, 즉 확실한 free space 만 다시 셌다 (`esdf_voxel_outside.py`, `esdf_outside_summary.json`, t3 e2 · e4):

   | 실제 거리 (free space) | ESDF − 실제 중앙 | 10 mm 넘게 큼 (못 봄) | 10 mm 넘게 작음 (부풂) |
   |---|---|---|---|
   | 0–1 cm | −0.4 / −1.0 mm | 33 / 37 % | 1 % |
   | 1–3 cm | −1.5 / −1.7 mm | 25 / 27 % | **6–7 %** |
   | 3–10 cm | −2.0 mm | 21 / 23 % | 15 / 17 % |
   | 10–30 cm | −4.6 / −7.4 mm | 7 % | 37 / 47 % |

   - **부풂의 정체:** free space 에서 ESDF = 0 (점유) 인 voxel 중 실제 표면에서 20 mm 넘게 떨어진 것이 첫 call (obs 20, 아무것도 움직이기 전) 에 이미 21–27 % 있다.
     - 모두 책 · 그릇 · cabinet · 머그 표면에서 **21–29 mm** 거리에 있고, 로봇 근처는 0 이다.
     - fine TSDF truncation 30 mm 가 관측 표면 **뒤쪽** 30 mm 까지를 점유로 칠해서, 얇거나 오목한 물체 (그릇 · 책) 의 뒷면 너머까지 2–3 cm 두께의 그림자가 생긴 것으로 해석한다.
     - 거리가 truncation 과 맞아서 이렇게 해석했다. cuRobo kernel 은 직접 확인하지 않았다.
   - 보수적인 쪽이라 충돌을 놓치게 만들지는 않지만, §11 의 target 그릇 주변 거짓 접촉의 원인이 된다.

### 8.4 한계

- episode 4 개다.
- fine 창 (0.64 m 정육면체) 밖의 sphere 9.8 % 는 coarse 층 값을 썼다 (서버와 같다).
- 무부호 GT 라서 물체 안쪽 깊이는 비교하지 않았다.

---

## 10. 동적 거리장 — 판정: **안 됨: 움직인 물체의 옛 자리가 남는다 (특히 coarse 층)**

### 10.1 코드 확인

- Panda 서버에서 켜진 동적 기능은 **ROI carve 하나** 다. 이 request 의 depth 광선이 지난 voxel 을 비우되, 쥔 물체 · 목적지 주변 상자와 권한 link 경로 상자 안에서만 비운다 (`roi_pad` 100 mm).
- 감쇠 (`tsdf_time_decay` · `tsdf_frustum_decay`), `free_space_carve`, `robot_occupied_free`, `target_layer` 는 `UNPORTED_FINAL_CONFIG_DESTS` 라 Panda 서버가 거절한다 (`server.py:611-637`).
- cuRobo TSDF 는 매 request 단조 가중치로 쌓인다.

### 10.2 무엇을 했나 (02:1x KST)

- **질문:** 물체가 움직인 뒤 옛 자리를 거리장이 비우는가?
- **방법:**
  - §8 과 같은 기록에서 첫 chunk 시점의 target · 장애물 표면 sample 을 '옛 자리' 로 잡는다.
  - 그중 지금은 모든 실제 표면에서 ≥ 3 cm 떨어진 (정말로 비워진) 점만 남긴다.
  - 이후 chunk 마다 fine · coarse ESDF 를 읽는다. 잊었으면 ≥ 3 cm, ghost 면 ≈ 0 이 나와야 한다.
- 스크립트: `dynamic_ghost.py`, 결과 `dynamic_ghost_<stem>.json`

### 10.3 결과

![ghost](../../../outputs/2026-11-08_AG3S_review_overview/f10_dynamic_ghost.png)

| episode | 움직인 물체 | 비워진 자리 중 ESDF < 10 mm (ghost) |
|---|---|---|
| t3 e2 | 책 (VLA 가 쳐서 이동) | fine 0.34 → 0.04 (차츰 지워짐), **coarse 0.40–0.50 그대로** |
| t3 e2 | target 그릇 (들어 옮김) | fine 0.22–0.40 → 0.09, **coarse 0.17–0.21 그대로** |
| t3 e4 | 머그 (이동) | **fine 0.18–0.19 · coarse 0.16–0.18 끝까지 남음** |
| t3 e4 | target 그릇 | fine 0.0 · coarse 0.018 (거의 지워짐) |
| t2 e2 (03:1x 추가) | 우유 (이동) | fine 0.12 → 0.005 · coarse 0.24 → 0.037 (대부분 지워짐) |
| t2 e1 (03:1x 추가) | 수납 상자 | 3 cm 넘게 비워진 자리 없음 (판정 불가) · target 그릇 0.0 |

§8 의 e3 (와인병이 넘어짐) 에서도 같은 현상이 보인다. 충돌 뒤 거리장이 접촉이라 하지만 실제로는 비어 있는 sphere 191 건이 생겼고, 반대로 병이 실제로 있는 곳 8 건은 놓쳤다. 넘어진 병의 옛 자리는 남고 새 자리는 아직 안 보였기 때문이다.

### 10.4 판정과 이유

- **안 됨 (일부 사례).** 장애물이 움직인 3 사례 중 2 (책 · 머그) 에서 ghost 가 끝까지 남았고, 1 (우유) 은 대부분 지워졌다. ROI carve 는 쥔 물체 · 목적지 주변만 비우므로, 지워지는지는 옛 자리가 그 상자 · 다시 보이는 시야 안에 드는지에 달린다 (추정).
  - coarse 층은 같은 광선 carving 을 받아도 20 mm voxel 이라 남는 비율이 더 높다 (측정, 원인은 추정).
- **결과:** 충돌 · 밀림 · 넘어짐 뒤에는 거짓 장애물 (불필요한 HOLD · 우회) 과 실제 장애물 누락이 함께 생긴다.
- **원인:** 감쇠 · free-space carve 계열이 Panda 에 이식되지 않았다 (RB-Y1 development 에는 있음).

### 10.4b 후속 — 최신 closed-loop 판 gt19pts 의 충돌 5 episode (04:5x KST, `dynamic_ghost_gt19_*.json`)

| episode | 장애물 옛 자리 | target 옛 자리 (coarse ghost 처음 → 끝) |
|---|---|---|
| t2 e0 | 3 cm 넘게 비워진 자리 없음 | 0.53 → **0.04** (fine 0.60 → 0.23) |
| t2 e2 | 없음 | 0 → 0 |
| t2 e4 | coarse 0.43 → 0.19 (fine 0.17 → 0.11) | 0.39 → **0** |
| t3 e0 | 없음 | 없음 |
| t3 e3 | 없음 | 1 chunk 만 (0.99) |

- 개입하는 판에서는 장애물이 3 cm 넘게 밀린 경우가 1 / 5 뿐이다. 그래서 §10.3 의 큰 장애물 ghost 상황 자체가 드물다.
- target 옛 자리는 대부분 지워졌다. ROI carve 가 쥔 물체 주변 상자를 비우기 때문이다.
- 장애물이 밀리면 (t2 e4) ghost 가 줄어들지만 남는다 (0.19).
- §10 의 판정 (ghost 가 남음) 은 장애물이 밀린 경우에 대한 것이고, 최신 판에서는 그런 경우가 줄었다.

### 10.5 한계

- 4 episode (장애물 이동 3 사례) 다.
- 충돌로 물체가 움직이는 상황 자체가 VLA observe-only 기록이라 생긴 것이다. 실제 개입판에서는 덜 일어날 수 있다.

---

## 11. 충돌 제약 (sphere clearance 수준) — 판정: **충돌 직전 검출은 대체로 됨 (HVC 가 장애물을 지운 경우 제외). 실제 접촉의 34 % 를 놓치고, 거짓 '접촉' 은 실제의 4.3 배**

### 11.1 무엇을 했나 (02:1x–02:2x KST)

- **질문:** TO 가 읽는 sphere 별 clearance (기록 `clearance[0]` = 서버 거리장 − r) 가 실제 접촉 / 여유와 맞는가?
- 실제 = 로봇이 아닌 실제 표면까지 거리 − r. latch HELD 면 쥔 물체를 뺀다.
- §8 과 같은 4 episode, sphere-step **18,870 개 (t3) + 4,080 개 (t2)**.

### 11.2 결과

| episode | 실제 접촉 (≤ 0) | 거리장 접촉 | 둘 다 | 놓침 | 거짓 접촉 | 거짓 접촉의 주된 물체 · latch |
|---|---|---|---|---|---|---|
| t2 e0 (moka pot, 충돌 t = 77) | 23 | 89 | 22 | 1 (0.6 mm 관통) | 67 | panda_hand · 실제 여유 중앙 4.9 mm |
| t3 e2 (책) | 8 | 96 | 4 | 4 | 92 | target 그릇 83 (closing · latched) |
| t3 e3 (와인병) | 23 | 243 | 6 | **17** | 237 | 와인병 191 · 그릇 46 (held · latched) |
| t3 e4 (머그) | 9 | 78 | 7 | 2 | 71 | target 그릇 65 (closing) |

(03:1x 추가) t2 e1 · e2 · e3 를 더해 7 episode, sphere-step 36,210 개 (`sphere_contact_summary.json`):

| episode | 실제 접촉 | 거리장 접촉 | 둘 다 | 놓침 | 거짓 |
|---|---|---|---|---|---|
| t2 e1 (수납 상자, 43) | 22 | 3 | 3 | **19** | 0 |
| t2 e2 (우유, 90) | 47 | 144 | 45 | 2 | 99 |
| t2 e3 (와인병, 85) | 0 | 0 | 0 | 0 | 0 (충돌은 쥔 그릇 쪽, sphere 밖) |
| **합계 7** | **132** | 653 | 87 | **45** | 566 |

t2 e1 의 놓침 19 는 §17 (손 voxel 비우기가 상자를 지움) 이 원인이다.

### 11.3 판정과 이유

1. **충돌 직전 검출은 대체로 된다.** 장애물에 다가가는 동안 거리장 clearance 가 실제보다 작거나 같다 (§8.2 표: 6 시점 중 5). e4 는 충돌 1 step 전에 −11 mm 로 접촉을 본다.
   - **예외:** t2 e1 은 손이 상자를 누르는 동안 HVC 가 상자 voxel 을 지워 실제 접촉 22 중 19 를 놓쳤다 (§17). 그래서 "검출됨" 은 HVC 가 그 장애물을 지우지 않은 경우에 한한다.
2. **놓침 45 / 132 (7 episode):**
   - 물체별 (6 episode, `sphere_contact_by_object.json`): 장애물 29 · target 그릇 15.
   - 장애물 놓침의 대부분은 t2 e1 의 HVC (19, §17) 와 t3 e3 의 넘어진 와인병 새 자리 (8, §10) 다.
   - 그릇 놓침은 손이 그릇에 닿는 순간을 거리장이 몇 mm 늦게 보는 경우다. 그릇은 권한 link 가 닿아도 되는 대상이다.
3. **거짓 접촉 566 건은 실제 접촉 132 건의 4.3 배다.**
   - 물체별 (6 episode, 499 건): target 그릇 268 (54 %) · 장애물 226 (45 %).
   - 그릇 쪽은 접근 · 쥐기 중 손이 그릇에 붙은 경우다. 이 기록의 clearance 는 주 층 기준이고, 권한 link 는 TO 에서 target-free 층과 contact phase margin 0 을 쓰므로 실제 제약에서는 대부분 풀린다.
   - 장애물 쪽은 충돌 뒤 남은 ghost (§10, 예: t3 e3 와인병 191) 와 물체 뒤 그림자 (§8) 다.
   - G67–G73 의 "LIBERO 거짓 양성" 과 같은 계열의 원인이 측정으로 확인된다.

### 11.4 후속 — 최신 closed-loop 판 gt19pts SafeLIBERO 11 (05:0x KST, `sphere_contact_summary_gt19.json`)

| 항목 | 값 |
|---|---|
| sphere-step | 71,910 |
| 실제 접촉 | 268 (target 그릇과의 접촉이 대부분 — 쥐기) |
| 거리장 접촉 | 438 (둘 다 134) |
| 놓침 134 | target 그릇 92 · **장애물 23** · 기타 19 |
| 거짓 접촉 304 | target 그릇 287 · 장애물 8 · 기타 9 |

- **거짓 접촉:** 장애물 쪽 거짓 접촉이 8 건으로 크게 줄었다 (F8 observe 226). 개입판에서는 장애물이 덜 밀려 ghost 가 적기 때문이다 (§10.4b). 나머지는 target 그릇 주변이고, 권한 link 는 target-free 층과 margin 0 을 쓰므로 실제 제약에서는 대부분 풀린다.
- **장애물 놓침 23 건:**
  - 모두 t2 e4 (obs 45) · t3 e0 (obs 45) · t3 e3 (obs 155–160) 이다. 손 · 팔이 장애물에 −0.1 – −4.5 mm 파고들 때 거리장은 +0.1 – +10 mm 로 봤다.
  - 세 episode 모두 HVC 가 장애물 voxel 을 지운 episode 다 (431 · 495 · 1749 개, §17.5).
  - 놓침이 HVC 탓인지 voxel 단위로는 확인하지 않았다. 크기는 수 mm 로, F8 의 t2 e1 (−13 mm 대 +38 mm) 보다 작다.

### 11.5 한계

- sphere 별 clearance 만 봤다. 행 분류 (`obstacle_object_rows`), pass-through τ, 인증, HOLD 사유 분류, QP 해의 정확성은 이 절에서 검증하지 않았다 (§12 에서 이어서).
- 권한 link 의 target-free 층 clearance 는 기록 (`esdf_free_*`) 이 있으나 아직 대조하지 않았다.

---

## 12. Target-free 층 (권한 link 가 묻는 거리장) — 판정: **target 은 ~70 % 만 지워짐. 충돌 뒤 장애물 일부도 같이 지워짐 (GT 판 기준)**

### 12.1 무엇을 했나 (02:2x–02:3x KST)

- **질문:** 손바닥 · 손가락 (권한 link) 이 묻는 target-free 층은 target 만 지우는가?
- **입력:** §8 기록의 `target_free_removed_ijk` (지운 voxel) 다. 이 기록 판은 GT 인식 F8 (extent `oracle_visible`, GT identity) 이다.
- **GT:** 각 voxel 중심을 1.5 voxel 안 최근접 실제 표면의 역할로 분류했다. 역할: target / 다른 물체 / 탁자 · 고정물 / 없음.
- **참고:** GT 없는 판의 ball-owned 층은 cuRobo (GPU) 가 필요해 재지 못했다. 대신 §4.2 의 ball 내용물 (다른 물체 점 중앙 311 개) 이 그 판의 위험을 보여준다.
- 스크립트: `outputs/2026-11-08_AG3S_review_esdf/target_free_check.py`, 결과 `target_free_<stem>.json`

### 12.2 결과

| episode | 지운 voxel (전체) | target | 탁자 · 고정물 | 다른 물체 | 없음 (현재 표면 7.5 mm 밖) | 쥐기 전 target 표면 덮임 (중앙) | 다른 물체를 지운 chunk |
|---|---|---|---|---|---|---|---|
| t3 e2 (책) | 31,401 | 0.46 | 0.13 | 0.00 (책 10) | 0.40 | 0.71 | 1 / 12 |
| t3 e4 (머그) | 52,501 | 0.40 | 0.17 | 0.007 (**머그 372**) | 0.43 | 0.72 | **9 / 20** |

### 12.3 판정과 이유

1. **target 을 다 지우지 못한다.** 쥐기 전 target 표면의 ~30 % 가 target-free 층에도 남는다. 손가락이 그릇 가장자리에 닿는 순간 거짓 접촉이 나는 원인 중 하나다 (§11 의 거짓 접촉 중 54 % 가 target 그릇).
2. **충돌 뒤에는 장애물도 지운다.** e4 에서 VLA 가 밀어 낸 머그가 target 근처로 오자 (obs 40–75), 머그 voxel 이 최대 96 개 / chunk 지워졌다. 손가락은 그 자리에서 머그를 못 본다.
   - GT extent 판에서도 이렇다. truncation 띠 · memory 가 옛 target 자리를 지우는데, 그 자리에 다른 물체가 들어온 경우로 추정한다 (추정이며 분해하지 않았다).
3. **탁자 · 고정물 13–17 %:** target 바닥 주변의 탁자 voxel 이다. support 행 규칙이 따로 있어 영향은 제한적이라고 판단한다 (미측정).

---

## 13. 쥔 물체 모델 · 놓기 (detach) — 판정: **F8 몸통 덮개는 물체와 크게 어긋남. 열린 손 attach 가 나면 detach 도 영원히 안 됨. 최신 판 (F12) 은 열린 손 attach 0, detach 가 실제로 일어남 (진실 대조 1 사례)**

### 13.1 무엇을 했나 (02:3x KST)

- **(a) 모델 크기 대조:** §8 기록 (F8, 몸통 덮개: fit sphere + 바닥 sphere) 의 HELD chunk 마다 attached query sphere 와 실제 쥔 그릇 표면을 비교했다.
  - 덮임 = 실제 표면 중 sphere 안에 든 비율
  - 여분 = sphere 부피 중 실제 표면에서 > 10 mm 떨어진 비율
  - 바닥 = sphere 최저점 − 실제 최저점
- **(b) detach 사례:** e3 에서 그릇이 손에 있는지를 MuJoCo 접촉 · 위치로 call 마다 대조했다.
- **(c) detach 수:** 최신 F12 SafeLIBERO 11 · LIBERO-Safety 10 의 detach · attach 수를 셌다.
- 스크립트: `held_model_check.py`, 결과 `held_model_check.json`

### 13.2 결과

| episode (F8) | HELD chunk | 덮임 중앙 | 여분 중앙 | 바닥 − 실제 (중앙) |
|---|---|---|---|---|
| t3 e2 | 11 | 1.00 | 0.60 | −30 mm |
| t3 e4 | 12 | 0.62 | 0.74 | **−61 mm** |
| t3 e3 | 47 | **0.00** | **1.00** | +80 mm |

t3 e3 의 시간 순서:

| obs | 일 |
|---|---|
| 65 | 손 개도 1.002 (열린 손) 로 attach → HELD |
| 80–140 | 실제로 쥐고 운반 (손가락–그릇 접촉 9–11) |
| ~150 | 그릇이 떨어짐 (z 0.91) |
| 155–290 | 손이 다시 열림 (0.97–0.99) 인데 episode 끝까지 **HELD** |

- detach 조건은 "개도 > attach 개도 + 0.02 = 1.022" 이고, 개도는 1 을 넘지 못한다. 그래서 detach 가 일어날 수 없다.
- 그동안 모델 sphere 는 빈손에 붙어 있다 (덮임 0).

최신 F12:

| 벤치 | attach | detach | 열린 손으로 HELD 인 채 끝난 episode |
|---|---|---|---|
| SafeLIBERO 11 | 10 | 1 (개도 0.14 → 0.36 에서 LATCHED 로; §6.6 에서 놓기 3 frame 뒤로 확인) | 0 |
| LIBERO-Safety 10 | 8 | 1 (같은 형태; 물리 진실 대조 안 함) | 0 |

### 13.3 판정과 이유

1. **F8 몸통 덮개: 안 됨.**
   - 모델이 그릇보다 30–61 mm 아래까지 내려간다. 탁자 · 목적지와의 거짓 위반 원인이 된다 (G70 의 "쥔 물체 구가 탁자 속 −84 mm" 와 같은 현상).
   - 부피의 60–74 % 가 여분이다.
   - 이것이 최신 판에서 20 mm 점 모델 (`--held-point-model-mm 20`, F14 · att 판) 로 바뀐 이유와 맞다. 점 모델 자체는 이 절에서 재지 않았다. peer 산출물 `outputs/2026-11-08_B_쥔물체_점모델/구모델_비교_s11_t4.png` 참조.
2. **열린 손 attach → detach 영구 불능 (F8):** attach 개도가 기준이 되기 때문이다.
   - 최신 판은 (c′) 로 열린 손 attach 가 0 이고 (§6), detach 가 실제로 일어난다 (F12 2 회). 그래서 이 연쇄는 막혀 있다.
   - 다만 detach 기준이 "attach 개도 + 0.02" 라는 구조는 그대로다. attach 가 덜 닫힌 상태에서 일어나면 같은 문제가 다시 생길 수 있다.

### 13.4 한계

- 몸통 덮개만 쟀다. 최신 점 모델은 constraint 기록 run 이 없어 (GPU) 재지 못했다.

### 13.5 후속 — 최신 점 모델 (`--held-point-model-mm 20`, gt19pts SafeLIBERO 11) (04:3x KST)

- **입력:** dev-ec 의 gt19pts run 이다 (`--record-constraints full`, closed-loop).
- **방법:** HELD chunk 152 개의 `attached_query_spheres` (점 모델 sphere 71–105 개, r 17.3 mm) 를 재구성 장면의 실제 그릇과 §13.1 과 같은 지표로 비교했다 (`held_model_check_gt19_t*.json`).

| episode | HELD chunk | 덮임 중앙 | 여분 중앙 | 바닥 − 실제 (처음 → 끝) |
|---|---|---|---|---|
| t1 e1 | 10 | 0.99 | 0.64 | −45 → **+38 mm** |
| t1 e4 | 12 | 0.98 | 0.69 | −50 → +33 |
| t2 e0 | 18 | 0.96 | 0.61 | −21 → +25 |
| t2 e1 | 16 | 0.96 | 0.48 | +1 → +32 |
| t2 e2 | 29 | 0.82 | 0.71 | −16 → +8 |
| t2 e3 | 17 | 0.63 | 0.79 | −33 → +20 |
| t2 e4 | 20 | 0.59 | 0.70 | −4 → +45 |
| t3 e2 | 12 | 0.98 | 0.41 | −3 → +24 |
| t3 e3 | 3 | 1.00 | 0.43 | −10 → +9 |
| t3 e4 | 15 | **0.11** | 0.95 | −12 → +17 |
| 전체 | 152 | 0.82 | 0.70 | 중앙 −14 (p10 −46 · p90 +20) |

**판정:** 몸통 덮개 (§13.2) 보다 덮임은 낫다 (대부분 ≥ 0.96). 그러나 두 가지가 안 된다.

1. **들어 올린 뒤 그릇 아랫부분이 모델에 없다.** 바닥이 실제보다 8–45 mm 위에 있다.
   - 점 모델이 관측된 점 (주로 윗면 · 옆면) 으로 만들어지기 때문이다.
   - 들기 전에는 받침면까지 채워서 반대로 45–50 mm 아래까지 내려간다.
   - 쥔 그릇의 아랫면이 장애물 · 접시에 닿는 충돌을 모델이 보지 못한다. §14.4 의 t2 e0 처럼 쥔 물체 행의 위반이 판정에 직접 들어가는 장면에서 위험하다.
2. **손이 덜 닫힌 채 attach 하면 모델이 어긋난다 (t3 e4).**
   - 개도 0.375 에서 attach 했다. (c′) moved 검사 (열림 기준 대비 ≥ 0.05 닫힘) 는 통과한다.
   - 그 뒤 손가락이 0.05 까지 더 닫히는 동안 그릇이 손 안에서 움직였다. attach 순간의 점으로 고정된 모델은 덮임이 0.39 → 0.06 으로 떨어졌다.

---

## 14. TO 판정 층 (pass-through τ · 개입) — 판정: **안 됨 (F12): 충돌 8 건 모두 직전 위반이 τ 25 mm 미만이라 통과됨**

### 14.1 무엇을 했나 (02:3x–02:4x KST)

- **질문:** F12 (LIBERO 동률판: `--violation-tolerance-mm 25`, pass-through when clear) 의 판정 층이 실제 충돌 앞에서 개입하는가?
- **입력:** F12 closed-loop SafeLIBERO 11 episode 의 call 기록 (`to.passthrough_clear`: applied · reason · reference_violation) 과 SafeLIBERO 공식 충돌 판정 (`collided`, `collide_t`).

### 14.2 결과

![tau](../../../outputs/2026-11-08_AG3S_review_overview/f14_tau_before_collision.png)

| episode | 장애물 | 충돌 t | 충돌 직전 call (obs: 판정, VLA 경로 위반) | TO 실행 call / 전체 |
|---|---|---|---|---|
| t2 e0 | moka pot | 77 | 70: 통과 6.6 mm · **75: 통과 20.4 mm** | 3 / 32 |
| t2 e1 | 수납 상자 | 42 | **40: 통과 15.0 mm** | 0 / 32 |
| t2 e2 | 우유 | 90 | 75: TO (28.2 mm) · **85: 통과 19.4 mm** | 5 / 56 |
| t2 e4 | 머그 | 44 | **40: 통과 11.3 mm** | 0 / 43 |
| t3 e0 | 와인병 | 46 | 35: TO (26.3 mm) · **45: 통과 13.8 mm** | 1 / 41 |
| t3 e2 | 책 | 38 | 30: 통과 14.8 mm · **35: 통과 16.5 mm** | 2 / 45 |
| t3 e3 | 와인병 | 36 | 30: 통과 6.6 mm · **35: 통과 20.7 mm** | 3 / 56 |
| t3 e4 | 머그 | 35 | **30: 통과 11.6 mm** | 1 / 31 |
| 충돌 없음 3 (t1 e1 · t1 e4 · t2 e3) | | | | 0 · 0 · 3 |

통과 사유 전체: `clear` 385, `collision` 18.

### 14.3 판정과 이유

- **안 됨.** 8 개 충돌 모두 직전 1–2 call 에서 거리장이 VLA 경로의 위반을 봤다 (6.6–20.7 mm).
  - 그런데 그 값이 τ 25 mm 미만이라 `clear` 로 통과했고, 다음 step 에 충돌했다.
  - τ 를 LIBERO 거짓 양성을 없애는 최소값으로 정했기 때문이다 (G69). 그 결과 장애물 행에도 같은 τ 가 적용됐다.
- 같은 장면의 F8 observe 기록 (§8, §11) 에서 거리장은 대부분의 충돌 직전에 장애물을 봤다. 그래서 F12 의 놓침은 주로 **판정 문턱** 에 있다.
  - 단 t2 e1 처럼 HVC 가 장애물을 지운 경우는 거리장 자체가 못 봤다 (§17).
  - F12 와 F8 은 다른 run 이라 충돌 t 가 조금 다르다 (예: t3 e3 F8 35 / F12 36).
- 이것이 G75 (F13: 장애물 물체 행에는 τ 0, `--passthrough-obstacle-tolerance-mm 0`) 가 고치려는 바로 그 지점이다.
  - F13 결과는 peer 세션이 측정 중이다. 이 문서에서는 판정하지 않는다.

### 14.4 후속 — 최신 GT 판 gt19pts (`…-loose2-obst3-pts`) SafeLIBERO 11 (04:1x KST)

- **대상 판:** dev-ec 가 03:17–03:53 에 돌린 run 이다.
  - 설정: `--passthrough-obstacle-tolerance-mm 10` + `--oracle-obstacle-objects` (GT 장애물 물체 점으로 장애물 행을 정함) + `--held-point-model-mm 20`
  - 일반 τ 는 25 mm 그대로다.
- **결과:** 충돌 5 / 11 (F12 8 / 11), 성공 7. 남은 5 충돌의 직전 call:

| episode (충돌 t) | 직전 call 판정 | 이유 |
|---|---|---|
| t2 e0 (77) | obs 75 **통과** (`clear`) | 가장 큰 위반 22.1 mm 가 **쥔 물체 점 모델 sphere** (`attached:panda_hand[74]`) 였고, 장애물 행 위반은 0 이었다. 그래서 일반 τ 25 mm 로 통과했다. dev-ec 는 같은 사건을 "그릇 옆 moka pot 이 쥔 그릇의 support 로 분류되어 장애물 행이 생기지 않음" 으로 진단했다 (gtto24 주석, G88 — 쥔 물체 행의 support 분류). 다음 판 B3 에서 support 규칙을 "under" 로 바꾼다 |
| t3 e3 (35) | obs 30 **통과** | 장애물 행 위반 1.9 mm < 장애물 τ 10 mm |
| t2 e2 (78) | obs 75 TO 실행 (`collision` 28.2 mm) | TO 가 고쳤는데도 충돌했다 (§15: 개입 경로 오차 33–39 mm) |
| t2 e4 (44) | obs 40 TO 실행 (`obstacle` 11.3 mm) | 같다 |
| t3 e0 (46) | obs 45 TO 실행 (`obstacle` 13.8 mm) | 같다 |

**판정:** 장애물 행 규칙이 충돌을 8 → 5 로 줄였다. 남은 충돌은 셋으로 나뉜다.

1. 쥔 물체 ↔ 장애물 행에 장애물 τ 가 적용되지 않는다 (1 건). G62 · G73 이 짚은 같은 구멍이다.
2. 장애물 τ 10 mm 가 접촉 직전 위반을 통과시킨다 (1 건).
3. TO 가 개입했지만 실제 경로가 예측과 달라 부딪힌다 (3 건, §15).

- 3 이 가장 많다. 판정 문턱보다 **개입의 실행 정확도 (포화 · 응답 모델)** 가 다음 병목이라는 뜻이다.
- 이 판정은 dev-ec 의 평가와 별개의 기록 대조이고, 채택 판단은 하지 않는다.

---

## 15. EE action → 관절 경로 응답 모델 (변환 C) — 판정: **통과 call 은 잘 맞음. TO 가 action 을 바꾼 call 은 실제 경로와 3–4 cm 어긋남 (포화)**

### 15.1 무엇을 했나 (02:4x KST)

- **질문:** TO 가 충돌을 검사 · 수정하는 경로 (EE delta action 을 응답 모델로 바꾼 예측 EE 경로 `predicted_ee_base`) 가 실제로 움직인 EE 와 맞는가?
- **방법:** call 마다 예측 경로의 j 번째 실행 step 을 시뮬레이터가 기록한 실제 EE (`obs_robot0_eef_pos` [t + j + 1], base frame) 와 비교했다.
- **입력:** F12 LIBERO 27 · SafeLIBERO 11 · LIBERO-Safety 10.
- 결과: `outputs/2026-11-08_AG3S_review_esdf/ee_prediction_error.json`

### 15.2 결과

![resp](../../../outputs/2026-11-08_AG3S_review_overview/f15_response_model_error.png)

| 벤치 | 판정 | call | 1 step 오차 중앙 / p95 | 5 step 오차 중앙 / p95 | action 포화 |
|---|---|---|---|---|---|
| LIBERO 27 | 통과 | 594 | 0.5 / 1.3 mm | 4.9 / 12.2 mm | – |
| SafeLIBERO 11 | 통과 | 385 | 0.6 / 3.0 mm | 5.4 / 24.8 mm | – |
| SafeLIBERO 11 | **TO 개입** | 18 | 8.8 / 16.2 mm | **32.7 / 82.3 mm** | 13 / 18 |
| LIBERO-Safety 10 | 통과 | 482 | 0.9 / 4.4 mm | 5.2 / 19.7 mm | – |
| LIBERO-Safety 10 | **TO 개입** | 37 | 6.1 / 21.1 mm | **39.1 / 71.6 mm** | 28 / 37 |

### 15.3 판정과 이유

1. **통과 call: 잘 됨.** 5 step 뒤 중앙 5 mm 로, 5 mm voxel 수준이다.
2. **TO 개입 call: 안 됨.**
   - TO 가 고친 action 이 |a| = 1 로 포화되면 (`conversion.saturated_rows`, 개입 call 의 70–76 %), 응답 모델이 실제 움직임을 맞히지 못한다.
   - TO 가 '안전하다' 고 인증한 경로와 실제 경로가 5 step 뒤 중앙 3–4 cm, p95 7–8 cm 어긋난다.
   - 개입이 필요한 바로 그 순간에 안전 인증이 실제 경로에 대해 성립하지 않는다는 뜻이다.
   - 기존 기록의 '층 ④ — TO 경로 수정이 변환 C 포화로' (SL-GTTO2) 와 같은 원인을 실제 EE 로 정량화한 것이다.

- **(03:4x 추가) 포화 여부로 나눔** (SafeLIBERO + LIBERO-Safety 개입 call, `ee_error_saturation.json`):

  | 개입 call | 개수 | 5 step 오차 중앙 | p90 |
  |---|---|---|---|
  | 포화 있음 | 41 | **44.5 mm** | 69.8 mm |
  | 포화 없음 | 14 | 14.0 mm | 27.7 mm |

  포화가 주원인이다. 포화 없는 개입도 통과 call (5 mm) 보다 3 배 크다. 응답 모델이 VLA 가 낸 action 분포 밖 (TO 가 고친 action) 에서 덜 맞는 것으로 해석한다 (추정).

### 15.4 한계

- 개입 call 은 55 개뿐이다.
- 오차 일부는 접촉 (쥐기 · 충돌) 으로 몸이 밀린 것이다. 충돌 ±10 step 안 call 은 5 step 오차 중앙 6.0 mm 로 다른 call (5.6 mm) 과 비슷하다 (`ee_error_near_collision.json`).

---

## 16. GT 없는 판에서 latch × grounding (identity 고정) — 판정: **안 됨: 초기에 걸리는 LATCHED 가 틀린 첫 identity 를 고정한다**

### 16.1 무엇을 했나 (02:4x–02:5x KST)

- **질문:** §2 에서 쥐기 단계 attention 은 이미 목적지로 간다. 그렇다면 GT 없는 판에서 latch (`latched_identity_hold`) 가 target identity 를 지켜 주는가?
- **방법:** §4 와 같은 replay 에 물리 진실을 따르는 latch 를 흉내 냈다. 형태는 F12 의 실제 latch timeline (§6.3 그림) 과 같다.

  | call | latch | 서버 입력 |
  |---|---|---|
  | 처음 2 call | SEARCHING | |
  | 그 뒤 첫 close 명령까지 | LATCHED | `set_grasp_latched` |
  | close 부터 lift 까지 | CLOSING | `set_grasp_active` + latched |
  | lift 부터 | HELD | active + `set_subtask` place |

  - attach (쥔 물체를 robot 으로 접기) 는 흉내 내지 않았다. 그래서 transport 수치는 판정에 쓰지 않는다.
- 스크립트: `outputs/2026-11-08_AG3S_review_grounding/grounding_latch_replay.py`, 결과 `rows_latch_part*.jsonl`, `grounding_latch_summary.json`

### 16.2 결과

| phase | latch 없음 (§4) target_correct | latch 흉내 target_correct | latch 흉내 다수결 |
|---|---|---|---|
| approach (387) | 0.54 | **0.44** | target 227 · **distractor 67** (§4: 47) · 목적지 29 · 없음 35 |
| grasp (88) | 0.51 | 0.43 | target 50 · distractor 12 |

episode 별 (접근 단계, call 2 에서 잠긴 identity 기준):

| 잠긴 identity | latch 없음이 나음 | 같음 | latch 가 나음 |
|---|---|---|---|
| target | 3 | 21 | 1 |
| **distractor** | **5** | 3 | 0 |
| **목적지** | **2** | 0 | 0 |
| 없음 · 다른 물체 | 0 | 5 | 0 |

### 16.3 판정과 이유

- **안 됨.** GT 없는 판에서는 LATCHED 가 grounding 의 첫 확정 identity 위에 걸린다. 확정 규칙은 같은 label 3 frame, score ratio 1.3 이다.
- 그 identity 가 틀렸으면 (distractor 8 / 40 episode, 목적지 2 / 40), `latched_identity_hold` 가 이후 바로잡을 기회를 막는다.
  - latch 없이 돌린 grounding 은 그중 7 episode 에서 접근 중에 target 으로 옮겨 갔다.
- GT 판 (F8 – F12) 에서는 identity 가 GT 라 이 문제가 드러나지 않았다. GT 없는 판 (G78) 을 돌리기 전에 알아야 할 위험이다.
- **제안 (구현하지 않음, 사용자 판정 사항):**
  - LATCHED 를 '접근 거리 / close 명령 직전' 까지 늦추거나,
  - LATCHED 중에도 attention 다수가 다른 물체를 가리키면 풀어 준다.

### 16.4 반사실 — LATCHED 를 늦추면 (03:4x KST, 진단 전용)

- **방법:** 같은 replay 에서 LATCHED 를 첫 close 명령 10 frame (2 call) 전부터만 걸었다. 그 전은 SEARCHING 이다 (`grounding_latch_late.py`, `latch_counterfactual.json`).

| latch 시작 | approach target_correct | grasp target_correct | approach distractor 다수결 |
|---|---|---|---|
| 없음 (§4) | 0.543 | 0.511 | 47 |
| 이름 (F12 형태, 현재) | 0.442 | 0.432 | 67 |
| **늦음 (close 2 call 전)** | **0.532** | **0.602** | 47 |

![latch-cf](../../../outputs/2026-11-08_AG3S_review_overview/f16_latch_counterfactual.png)

- 늦은 LATCHED 는 초반 잠금을 없애 approach 를 latch 없는 수준으로 되돌린다. 쥐는 동안에는 identity 를 지켜서 grasp 가 셋 중 가장 좋다.
- §16.3 의 제안 (LATCHED 늦추기) 을 지지하는 측정이다. 서버 구현 · closed-loop 확인은 하지 않았다.

### 16.5 한계

- latch 신호를 흉내 낸 것이다. GT 없는 판의 실제 latch 확정 시점은 grounding 결과에 달려 있어 다를 수 있다.
- attach 를 흉내 내지 않아 쥔 뒤 동작은 판정하지 않았다.

---

## 17. 손 voxel 비우기 (HVC, `--hand-voxel-carve current --hand-voxel-carve-mm 20`) — 판정: **안 됨: 손이 닿은 장애물 표면을 지워 충돌을 못 보게 만든다**

### 17.1 무엇을 했나 (03:1x KST)

- **발견 경위:** §11 을 7 episode 로 늘렸더니 t2 e1 에서만 실제 접촉 22 중 19 를 거리장이 놓쳤다.
  - 놓친 sphere 는 모두 `panda_hand` 대 흰 수납 상자 (obs 45–60) 다. 실제 −1 – −13 mm 관통인데 거리장은 +8 – +38 mm 로 봤다.
- **방법:** 그 chunk 들의 HVC 기록 (`hand_carve_freed_centres` = HVC 가 지운 fine TSDF voxel 중심) 을 재구성 장면의 실제 표면과 대조했다. voxel 중심에서 7.5 mm 안 최근접 물체로 분류했다.
- **배경 (코드):** HVC (F8 이후 모든 판, GT 없는 판 포함) 는 손 · 손가락 self-filter sphere 를 20 mm 키운 안쪽의 점유 voxel 을 매 request 비운다. 지지면 아래 · 목적지 근처 · target ball / 점은 제외한다 (`pipeline._hand_carve_spec`).

### 17.2 결과

![hvc](../../../outputs/2026-11-08_AG3S_review_overview/f17_hvc_freed.png)

| t_step (obs) | latch | HVC 가 지운 voxel | 그중 수납 상자 (장애물) |
|---|---|---|---|
| 4 (40) | latched | 182 | **182** |
| 5 (45) | latched | 281 | **265** |
| 7 (55) | held | 100 | **100** |
| 8 (60) | held | 69 | **66** |
| 10 (70) | held | 1 | 1 |
| 19 (115) | held | 93 | cabinet 40 · 없음 53 (§17.4 의 "다른 물체 38" 과의 차이 2 는 표면 sampling 난수 차이) |
| 합계 | | 726 | **장애물 614 (85 %)** |

SafeLIBERO 공식 판정: 이 episode 는 obs 43 에 수납 상자와 충돌했다.

### 17.3 판정과 이유

- **안 됨.** HVC 는 "손 근처 voxel 은 손 자신의 흔적" 이라는 가정으로 지운다. 하지만 손이 실제 장애물에 닿거나 파고들 때는 그 장애물 표면이 바로 손 근처에 있다.
  - 그래서 충돌이 일어나는 바로 그 순간에 장애물이 거리장에서 사라진다.
- G64 는 다른 장면 (t2 e4, 머그) 에서 "머그 3 cm 안 voxel 감소 0.15 %" 로 이 가설을 기각했다. 이 장면에서는 장애물 voxel 614 개가 지워졌으므로, 기각은 그 장면에만 해당한다.
- **제안 (구현하지 않음):**
  - HVC 를 self-filter 가 놓친 로봇 pixel (예: render mask 로 확인된 robot voxel) 에만 적용하거나,
  - 이번 request 의 관측 depth 가 비로봇 표면으로 확인한 voxel 은 지우지 않는다.

### 17.4 7 episode 전체 (03:2x KST, `hvc_check.py`, `hvc_check.json`)

| episode | HVC 가 지운 voxel 중 장애물 | 다른 물체 (탁자 · cabinet · 그릇 등) | target | 현재 표면에서 7.5 mm 밖 (흔적 · ghost) |
|---|---|---|---|---|
| t2 e0 | 808 | 143 | 0 | 1370 |
| t2 e1 | 614 | 38 | 0 | 74 |
| t2 e2 | 941 | 760 | 0 | 1013 |
| t2 e3 | 150 | 472 | 6 | 251 |
| t3 e2 | 1008 | 468 | 0 | 2749 |
| t3 e3 | 2323 | 682 | 6 | 2256 |
| t3 e4 | 2371 | 785 | 6 | 2192 |
| **합계** | **8215 (38 %)** | 3348 (16 %) | 18 | 9905 (46 %) |

- **7 / 7 episode 에서 HVC 가 실제 장애물 표면을 지웠다.**
- HVC 의 원래 목적은 손 주변에 남은 흔적 · ghost 를 비우는 것이고, 지운 voxel 의 46 % 가 그런 것이었다.
- 그러나 지운 것의 절반 이상은 실제로 지금 거기 있는 물체 표면이다.
- **한계:** SafeLIBERO observe 기록 (VLA 경로, F8 설정) 이다. 개입판에서는 손이 장애물에 덜 다가가므로 양은 다를 수 있다.

### 17.5 후속 — 최신 closed-loop 판 gt19pts SafeLIBERO 11 (04:2x KST, `hvc_check_gt19.json`)

| episode | 장애물 | 다른 물체 | target | 흔적 · ghost |
|---|---|---|---|---|
| t1_e1 | 0 | 0 | 0 | 1 |
| t1_e4 | 192 | 8 | 0 | 46 |
| t2_e0 | 790 | 362 | 0 | 723 |
| t2_e1 | 472 | 336 | 133 | 373 |
| t2_e2 | 868 | 701 | 228 | 669 |
| t2_e3 | 0 | 412 | 0 | 396 |
| t2_e4 | 431 | 617 | 204 | 327 |
| t3_e0 | 495 | 395 | 0 | 186 |
| t3_e2 | 1695 | 718 | 0 | 455 |
| t3_e3 | 1749 | 2782 | 0 | 1537 |
| t3_e4 | 952 | 789 | 113 | 463 |
| **합계** | **7644 (37 %)** | 7120 | 678 | 5176 |

- **개입하는 최신 판에서도 11 episode 중 9 개에서 HVC 가 장애물 표면을 지웠다.** 지운 비율 37 % 로 observe 기록 (38 %) 과 같다.
- 즉 이 문제는 VLA 경로 (observe) 만의 일이 아니다.
- dev-ec 는 GT 판에서는 `--oracle-robot-mask` 가 손 흔적을 처리하므로 HVC 를 뺀 판을 시험하겠다고 알려 왔다 (04:0x).
- **LIBERO (gt19pts seed 7, 9 episode, `hvc_check_gt19L.json`):** LIBERO 에는 장애물이 없다. 대신 HVC 가 고정물 · 다른 물체 표면을 지웠다.
  - 서랍 속 그릇을 꺼내는 t4: 1486
  - cabinet 위 그릇을 집는 t9: 1310
  - t6 · t7: ~500
  - 같은 기록의 sphere 판정 (`sphere_contact_summary_gt19L.json`) 에서 실제 cabinet 접촉 5 중 2 (t4) 를 거리장이 놓쳤다. 양은 작지만 HVC 가 많이 지운 바로 그 장면이다.
- dev-ec 는 이 리뷰를 근거로 HVC 를 끈 판 (B3, gtto24 · 26) 을 queue 에 넣었다. 그 기록으로 '끄면 놓침이 사라지는가' 를 확인할 예정이다 (§17.6).

### 17.6 HVC 를 끄면 (B3 = gt21pts `…-obst3-pts-nocarve`, SafeLIBERO 11, 06:3x–07:0x KST)

- **입력:** dev-ec 가 이 리뷰를 근거로 HVC 를 끄고 돌린 판 (06:04–06:2x, `--record-constraints full`) 이다.
- **방법:** 같은 11 장면을 재구성해 §11.4 와 같은 sphere 대조를 했다 (`esdf_vs_truth_gt21_*.json`, 비교 `hvc_off_comparison.json`).

| 판 | sphere-step | 장애물 실제 접촉 | **장애물 놓침** | 전체 놓침 | 거짓 접촉 | 거짓의 실제 여유 (중앙) |
|---|---|---|---|---|---|---|
| gt19pts (HVC 켬) | 71,910 | 30 | **23** | 134 | 304 | 7 mm |
| B3 gt21pts (HVC 끔) | 59,840 | 34 | **4** | 62 | **1225** | **81 mm** |

- **좋아진 것:** 장애물 놓침 23 → 4.
  - 남은 4 건은 −1.8 · −6.8 · 0 · −0.2 mm 관통에 거리장 +0.5 – +6.4 mm 다.
  - HVC 가 장애물 표면을 지우던 문제 (§17.2–17.5) 가 사라졌다.
- **나빠진 것:** 거짓 접촉 304 → 1225.
  - 그중 613 은 한 episode (t3 e3) 의 HELD 구간이다. 이 episode 는 obs 35 에 와인병을 쓰러뜨렸고, obs 80 부터 끝까지 (45 call) HOLD 로 멈췄다.
  - 거짓 접촉 sphere (r ~23 mm) 는 실제 표면에서 중앙 81 mm 떨어져 있다. 대신 **와인병이 쓰러지며 지나간 자리** (obs 20–95 의 병 표면) 와 −11 – −13 mm 로 겹친다.
  - 쓰러지는 동안 관측된 병이 감쇠 없는 TSDF 에 ghost 로 남아 손 앞을 막은 것으로 해석한다. 로봇의 과거 자리와도 겹치지만, 손이 원래 그 근처에 있었으므로 구분력은 없다.

**판정:**

- HVC 는 두 가지를 동시에 하고 있었다.
  - 해로운 쪽: 손이 닿은 실제 장애물 표면을 지움.
  - 이로운 쪽: 손 주변의 ghost (움직인 물체 · 흔적) 를 지움.
- 끄면 앞의 것이 사라지지만 뒤의 것이 드러나 HOLD deadlock 이 생긴다.
- 그래서 근본 해결은 HVC 를 켜고 끄는 것이 아니라 **동적 거리장** (§10: 감쇠 · free-space carve 미이식) 이다. 한 판을 고를 문제라면, 판정은 사용자에게 남긴다.
- B3 SafeLIBERO 결과 (참고): 성공 10 / 11, 충돌 6 / 11 (gt19pts 7 · 5).
- dev-ec 의 설명 (07:1x): A'' 이후 엄격한 장애물 행은 '가장 가까운 field 표면이 최근 3 request 에 관측된 GT 장애물 점 위에 있는 행' 뿐이다. 현재 관측에서 벗어난 ghost voxel 은 '접촉 표면' 류가 되고, B5 (gt23t100) 는 이를 τ 100 mm 로만 검사한다. 그래서 B5 가 HVC 없이 ghost HOLD 대부분을 흡수할 것이라는 예상이다. B5 기록이 나오면 확인한다 (§17.7).

### 17.7 세 판 비교 — gt19pts (HVC 켬) · B3 · B5 (HVC 끔) (08:3x KST, `hvc_versions_comparison.json`)

- **방법:** 세 판 모두 재구성에 성공한 공통 9 episode (SafeLIBERO t1 e1 · e4, t2 e1–e4, t3 e0 · e2 · e4) 로 같은 sphere 대조를 했다.

| 판 | 장애물 실제 접촉 | 장애물 놓침 | 전체 놓침 / 실제 접촉 | 거짓 접촉 | SafeLIBERO 11 충돌 · 안전 성공 (call 기록) | 최장 연속 HOLD |
|---|---|---|---|---|---|---|
| gt19pts (HVC 켬) | 16 | **16** | 107 / 212 | 229 | 5 · 5 | 1 |
| B3 (끔) | 34 | **4** | 61 / 212 | 412 | 6 · 5 | 44 (t3 e3, 병 ghost, §17.6) |
| B5 (끔, 접촉 표면 τ 100) | 11 | **10** | 34 / 182 | 431 | **2 · 8** | 29 (t3 e0, 빈손 CLOSING, §6.8) |

- **HVC 를 켠 gt19pts 는 장애물 접촉을 하나도 거리장으로 보지 못했다 (16 / 16).**
- **끈 B3 는 대부분 봤다 (30 / 34).** 거리장이 −2 – −40 mm 로 접촉을 잡는다.
- **B5 의 놓침 10 은 모두 t3 e0 obs 110 한 순간이다.** 빈손으로 닫힌 채 멈춘 손이 와인병을 −2 – −6 mm 누를 때 거리장은 0 – +8 mm 였다. 5 mm voxel 해상도 수준의 놓침이고, HVC 문제와는 다르다.
- **판정:** HVC 를 끄면 장애물 표면이 거리장에 남는다는 것이 세 판 비교로도 확인된다.
  - B5 의 결과 (충돌 2 · 안전 성공 8) 는 dev-ec 의 예상대로 ghost HOLD 를 접촉 표면 τ 로 흡수한 판정 층 효과다. 다만 다른 종류의 deadlock (§6.8) 이 하나 남았다.
  - 표본은 9 episode, 판당 1 run 이다.

---

## 18. 쥐기 영역 해제 (`--grasp-zone-release`, F8 이후 · GT 없는 판 포함) — 판정: **코드상 구멍이 있음. 이 기록의 충돌 원인은 아님**

### 18.1 코드 확인 (03:2x KST)

- `linearize._grasp_contact_margin` (`AG3S/trajopt/linearize.py:1196-1205`): latch 가 LATCHED · CLOSING 이면, 손 · 손가락 (권한 link) 행 중 manipulated 중심에서 **수평 100 mm 안** 의 행을 margin −1 m 로 푼다 (제약이 걸리지 않음).
- 주석에 "nearest surface 가 무엇이든 (target, support, wall, neighbour)" 이라고 되어 있다. 즉 장애물도 풀린다.
- §6 에서 LATCHED 는 접근 초반 (obs ~20) 부터 close 명령까지 이어진다. 그래서 접근 내내 target 수평 100 mm 안의 장애물은 손이 무시한다.

### 18.2 기록으로 잰 크기 (`outputs/2026-11-08_AG3S_review_overview/zone_release_check.json`)

- **입력:** §11 의 SafeLIBERO 6 episode (t2 e1 – e3, t3 e2 – e4).
- **방법:** 실제 장애물과의 거리를 쟀다. manipulated 중심은 chunk 의 `target_centroid` 를 썼다.

| | LATCHED · CLOSING 에서 이 규칙이 풀었을 행 | 풀리지 않음 |
|---|---|---|
| 실제 장애물 접촉 (≤ 0) 41 건 | 0 (모두 수평 109–198 mm) | 29 (latched · closing) + 12 (held) |
| 장애물 20 mm 안 338 건 | **46** (14 %) | 292 |

F12 전체 기록에서 이 규칙이 실제로 푼 행 수 (`to.grasp_contact.zone_release_rows` 합): LATCHED 340,896 · CLOSING 287,468 (1516 call 중 891 call). 접근 · 쥐기 동안 매우 넓게 작동한다.

### 18.3 판정

- 이 기록 (observe) 의 장애물 충돌은 target 에서 수평 11–20 cm 떨어진 곳에서 일어나, 이 규칙의 영향 밖이었다.
- 그러나 장애물 20 mm 안 행의 14 % 는 이 규칙이 켜진 판 (F8 이후, GT 없는 판 포함) 이었다면 제약이 풀렸다.
- target 바로 옆 (10 cm 안) 에 장애물이 있는 장면에서는 이 규칙이 손 · 손가락의 장애물 보호를 접근 내내 끈다.
- **제안 (구현하지 않음):** 해제를 target · support 분류 행으로 제한한다. 이것은 F4 이전의 `--grasp-zone-mm` 규칙과 같다.

---

## 19. 서버 support 평면과 TO 의 support 행 분류 — 판정: **두 번째 평면은 서버에도 있다. 이 기록에서는 장애물 행을 support 로 바꾸지 않았다**

### 19.1 무엇을 했나 (03:2x–03:3x KST)

- **질문 1:** §5 의 두 번째 RANSAC 평면이 서버 거리장 · TO 가 쓰는 support 평면 (`support_planes`, chunk 기록) 에도 있는가?
- **질문 2:** 있다면 TO 의 행 분류 (`linearize._obstacle_rows`: 어느 평면이든 |d − (n·p − o)| ≤ 2 voxel 이면 support) 에서 장애물 근처 행이 support 로 바뀌는가?
  - support 로 바뀐 행은 F13 장애물 행 τ 0 에서 빠지고, zone 규칙 (§18) 의 대상이 된다.

### 19.2 결과 (`outputs/2026-11-08_AG3S_review_overview/support_planes_check.json`)

| 기록 | chunk | 두 번째 평면 있음 | 기울기 (중앙) | 탁자 대비 높이 (중앙) |
|---|---|---|---|---|
| SafeLIBERO t2 (observe) | 158 | 100 (63 %) | 2.4° | +22 mm |
| SafeLIBERO t3 (observe) | 136 | 109 (80 %) | 1.8° | +16 mm |
| LIBERO s10 (observe, `…gtto-gt12obs-libero9s10…`) | 194 | 149 (77 %) | 2.0° | +24 mm |

- 장애물 20 mm 안 sphere 338 개 중 support 로 분류된 것은 **0** 이다 (평면 2 개 · 평면 1 개 모두).

### 19.3 판정

- **두 번째 평면은 서버에서도 대부분의 chunk 에 있다.** 탁자 위 1.6–2.4 cm, 약 2° 기울기로, §5 와 같은 sliver 평면이다.
- 장애물 (병 · 상자 · 머그 · 책) 은 키가 커서, 장애물 근처 로봇 sphere 가 이 평면 높이와 겹치지 않았다. 그래서 TO 행 분류에는 영향이 없었다.
- 낮은 물체 (접시 · 그릇 · 상자 바닥) 근처의 손 행은 support 로 분류될 수 있다 (미측정).

---

## 20. self-collision 끔 (`--no-self-collision`, 모든 Panda 판) — 판정: **이 기록에서는 문제 없음**

- AG3S 를 만들 때 "SELF-COLLISION IS OFF: 쥔 물체와 로봇 자신의 쌍이 전부 면제" 경고가 나온다.
- **방법:** 재구성한 SafeLIBERO 7 episode 의 모든 image instant (1209 frame) 에서 MuJoCo 접촉을 셌다 (`outputs/2026-11-08_AG3S_review_overview/self_contact_check.json`).
- **결과:**
  - 로봇 비인접 link 끼리의 접촉: 0
  - 쥔 target 그릇과 팔 link (`robot0_link*`) 의 접촉: 0
- **판정:** 이 장면 · 경로에서는 self-collision 을 끈 영향이 관찰되지 않았다. 긴 물체를 들고 팔을 접는 장면은 측정 범위 밖이다.


---

## 21. GT 없는 판 closed-loop 기록 검증 (B5 att = gt24att, LIBERO seed 7) — 판정: **offline 예측과 일치. 7 task 는 target 을 맞게 잡고 지움, task 4 · 9 는 다른 물체를 target 으로 잡아 실패**

### 21.1 무엇을 했나 (09:0x–09:2x KST)

- **입력:** dev-ec 의 GT 없는 판 `AG3S-commonto-att-…-obst4-pts-nocarve-fh-t100` 이다 (attention grounding + 장면 장애물 물체, HVC 끔, 접촉 표면 τ 100).
  - run: LIBERO seed 7, 08:11, `--record-constraints full`, 9 episode, 247 chunk.
  - GT 없는 판의 target-free 층은 cuRobo 전용이라 offline 으로 재지 못했었다 (§12). 이 기록으로 처음 대조한다.
- **방법:** 9 장면을 재구성했다 (RGB bit 동일). chunk 마다 실제 표면과 대조했다.
  - grounding 의 `target_points`: 10 mm 안 최근접 물체
  - target-free 층이 지운 voxel (`target_free_removed_ijk`): 1.5 voxel 안 최근접 물체, target 표면 덮임
  - `gtfree_closedloop.py`, `gtfree_closedloop_libero_s7.json`

### 21.2 결과

| task | 결과 | 쥐기 전 chunk | grounding 다수결 | target-free 가 지운 voxel 중 target / 다른 물체 / 고정물 | target 표면 덮임 (중앙) |
|---|---|---|---|---|---|
| 0 | 성공 | 18 | target 18 | 0.78 / 0.01 / 0.00 | 0.42 |
| 1 | 성공 | 15 | target 14 | 0.70 / 0.02 / 0.00 | 0.45 |
| 2 | 성공 | 18 | target 17 | 0.69 / 0.00 / 0.00 | 0.16 |
| 3 | 성공 | 12 | target 12 | 0.80 / 0.00 / 0.00 | 0.48 |
| **4 (서랍)** | **실패** | 44 | **다른 물체 29** · 없음 7 | **0.00 / 0.95** / 0.00 | **0.00** |
| 6 | 성공 | 14 | target 13 | 0.69 / 0.00 / 0.03 | 0.61 |
| 7 | 성공 | 12 | target 10 | 0.45 / 0.00 / 0.08 | 0.29 |
| 8 | 성공 | 15 | target 15 | 0.83 / 0.00 / 0.00 | 0.56 |
| **9 (cabinet 위)** | **실패** | 44 | **다른 물체 27** | **0.00 / 0.69** / 0.15 | **0.00** |

### 21.3 판정과 이유

1. **offline 판정 (§4) 이 closed-loop 에서 그대로 나타났다.**
   - offline replay 에서 정답률 0 이던 task 4 · 9 가 실패했다.
   - 두 episode 모두 grounding 이 target 이 아닌 물체를 target 으로 잡았다. target-free 층은 그 물체를 지우고 (95 % · 69 %) 진짜 target 은 하나도 지우지 않았다.
   - 손가락이 엉뚱한 물체를 통과해도 되도록 허락받고, 정작 쥐어야 할 그릇은 장애물로 남았다.
2. **성공한 7 task 에서는 target 을 맞게 잡았다.** 그러나 target-free 층이 덮는 target 표면은 중앙 0.16–0.61 로 부분적이다. GT 판 (§12, ~0.7) 보다 낮다. ball-owned extent 가 target 의 일부만 소유하기 때문이다 (§4.5 의 support 평면 영향 포함, 미분해).
3. **잘못 지운 다른 물체 · 고정물은 성공 task 에서 0–8 % 다.** §4.2 의 "ball 이 다른 물체 점 중앙 311 개를 덮음" 이라는 offline 우려는 ball-owned 의 소유 판정이 대부분 막았다. 이 판정은 기록이 없던 §12 의 빈칸을 채운다.
4. **다음 일 (제안):** GT 없는 판의 병목은 거리장 · TO 가 아니라 **identity (grounding)** 다 (§4 · §16).

### 21.4 확장 — LIBERO seed 8 · 9 와 SafeLIBERO (10:3x–11:2x KST)

- **입력:** 같은 판의 LIBERO seed 8 · 9 (각 9) 와 SafeLIBERO t1 · t2 · t3 (11 중 장면 재구성 9) 기록이다.
- **결과 파일:** `gtfree_closedloop_libero_s{8,9}.json`, `gtfree_closedloop_safelibero_t{1,2,3}.json`

LIBERO 27 episode (seed 7 · 8 · 9) — 실패 10 개의 원인:

| task | seed 7 | seed 8 | seed 9 | 쥐기 전 grounding = target | 지운 voxel 중 target / 다른 물체 |
|---|---|---|---|---|---|
| 3 ("cookies box 위") | 성공 | **실패** | **실패** | s8 0 / 34, s9 0 / 40 (처음부터 다른 물체, LATCHED 로 고정) | 0.00 / 0.75–0.89 |
| 4 (서랍) | **실패** | **실패** | **실패** | 0 / 36–41 | 0.00 / 0.94–0.96 |
| 7 (stove 위) | 성공 | **실패** | **실패** | 10 / 10 (맞음) | 0.73–0.86 / 0.00 |
| 9 (cabinet 위) | **실패** | **실패** | **실패** | 0 / 27–38 | 0.00 / 0.69–0.91 |
| 나머지 (0 · 1 · 2 · 6 · 8) | 성공 15 / 15 | | | 대부분 target (6 · 8 은 처음 2 call 다른 물체 → 바로잡힘) | 0.47–0.88 / 0.00–0.22 |

- **실패 10 중 8 개 (task 3 · 4 · 9) 는 identity 실패다.** grounding 이 처음부터 다른 물체를 target 으로 잡고 바로잡지 못했다.
  - §16 의 '이른 LATCHED 가 틀린 첫 identity 를 고정' 이 closed-loop 에서 그대로 보인다 (seed 8 task 3: obs 10 부터 다른 물체, obs 20 LATCHED).
- **task 7 실패 2 개는 identity 는 맞았다.** 정상 attach 뒤 운반 중 `obstacle` 거절이 반복되어 (seed 8: 19 call) 시간 초과했다.
  - GT 없는 장면 장애물 물체 (`--scene-obstacle-objects`) 가 운반 경로에 장애물 행을 만든 것으로 보인다 (원인 미분해).

SafeLIBERO (성공 4 / 11, 충돌 6, 안전 성공 2):

| task | episode (결과) | 쥐기 전 grounding 다수결 |
|---|---|---|
| t1 | e1 (실패) · e4 (성공) | e1: 다른 물체 24 / 24 · e4: target 6 / 10 |
| t2 | e0 · e1 · e3 (실패), e2 · e4 (성공) | e0: 다른 물체 · 없음, e1: 없음 50 / 56, e2: target 11 / 18, e3: 섞임 |
| t3 | e0 (실패, 재구성 못 함), e2 (성공) · e3 · e4 (실패) | e2 · e3 · e4: 다른 물체 30 / 33 · 56 / 56 · 54 / 56 |

- SafeLIBERO 에서도 실패의 대부분은 identity 다. 특히 t3 (cabinet 위 그릇) 은 LIBERO task 9 와 같은 장면 구조로 모두 다른 물체를 잡았다.
- 'none' (target 점이 실제 표면 10 mm 안에 없음) 이 많은 episode (t2 e1) 는 target 점이 ghost · 로봇 근처에 있는 경우로 보인다 (미분해).

**판정 (21 전체):**

- GT 없는 판에서 거리장 · target-free 층은 **잡은 identity 를 충실히 따른다.** identity 가 맞으면 target 을 지우고, 틀리면 엉뚱한 물체를 지운다.
- 그래서 GT 없는 판의 성패는 **grounding identity** 가 정한다. LIBERO 실패 10 중 8, SafeLIBERO 실패 대부분이 여기서 나왔다.
- offline 검증 (§4 정답률 task 4 · 9 = 0, §16 이른 latch 고정) 이 closed-loop 결과를 미리 맞혔다.

