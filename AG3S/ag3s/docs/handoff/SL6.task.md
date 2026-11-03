# SL6 — AG3S 최종 구성 Panda 이식 + 기하 grasp 보정 · 쥐기 목표 항 포함 / 제거 smoke

> writer: lead · 읽는 쪽: implementer 2 명 (SL6-A · SL6-B, 파일이 겹치지 않음) → lead (GPU) · 작성 2026-10-29 19:10 KST.
> 상위: [AG3S_FINAL_MERGE_REPORT.md](../../../../doc/active/migration/AG3S_FINAL_MERGE_REPORT.md) "다음 — Panda 이식 판정" · 최종 구성 spec [ADOPTED_KO.md](../../../../doc/active/reviews/AG3S_decision/ADOPTED_KO.md) §A8 "A8 최종 구성 확정" 6 · 서버 인자 [AG3S-CORE-EVAL-A6.impl.md](AG3S-CORE-EVAL-A6.impl.md) §3.
> 계획표의 **SL-6 phase / gate source** (미착수) 를 이 단계가 맡는다.

## 0. 사용자 판정 (결과 전, 이 문서가 고정한다)

| 날짜 | 판정 |
|---|---|
| 2026-10-29 | HSR6 probe 는 Panda 용으로 재학습. attention head 는 Panda VLA (`pi05_libero`) 에서 dev episode 로 다시 고른다. A6 손바닥 기준점 · 손바닥 껍질 구는 Panda 기하로 다시 정의하고 허용 거리 · 창 · 단계 · 금지 접촉 정의는 RB-Y1 값 유지 |
| 2026-10-29 | 기하 grasp 보정 (G1) · 쥐기 목표 항 (A4) 을 최종 구성에서 뺄지 둘지 **Panda closed-loop smoke 한 번** 으로 정한다. 이식을 마친 Panda 최종 구성에서 비교한다 |
| 2026-10-29 | HSR6 는 **retreat 없이** 학습해도 된다. **가장 빨리 실험할 수 있는 방법** 을 쓴다 → 새 rollout 없이 기존 dev 기록으로 학습 · 선택 (§3 · §4) |
| 2026-10-29 | Panda 최종 구성에 기존 Panda 개선판 (pb1–pb7) 은 **넣지 않는다**. smoke 에서 이전과 같은 이유 (SL5b reach 재측정, latch 가 closing 에 머묾, over-lift 등) 로 실패하면 그때 해당 pb 를 더하는 것을 제안한다 (자동으로 넣지 않는다) |
| 2026-10-29 | smoke 장면 · 판정 기준은 §6 그대로 고정 |
| 2026-10-29 (~20:40 KST) | **Panda 최종 구성은 물체를 쥐기 전에 연속성 항 · 꺾임 변화율 항을 끈다** (`--continuity-after-grasp --deflection-rate-after-grasp`, 단계 판정 = grasp latch: searching · latched · closing 에서 끔, held 부터 씀). RB-Y1 에서 측정한 최종 구성에는 없던 설정이다 (그때는 넓힌 목표 항 · 껍질 구와 묶여 한 번 시험되어 미채택, NOT_ADOPTED N13) → Panda 결과를 RB-Y1 결과와 나란히 쓸 때 이 차이를 적는다 |

| 2026-10-30 (~01:30 KST, §6 smoke 결과 뒤) | **G1 · A4 는 Panda 최종 구성에서 제거 확정** (§6 규칙: 작동판 1 vs 제거판 0). G1 은 581 call 중 0 회 작동해 시험되지 않았다는 사실과 함께 기록한다. G1 후보 실패 원인은 따로 조사하지 않는다 |
| 2026-10-30 (~01:30 KST) | §6 실패가 이전 pb 원인과 같다 (쥔 bowl 을 latch 가 인식 못함 · 가려진 bowl 의 reach 를 마지막 본 자리로 잼 · HELD 전 TrajOpt 가 손을 위로 밈) → **pb1 · pb6 · pb7 을 Panda 최종 구성에 더한다** (`--reach-at-capture`, `--grasp-reach-sticky --grasp-drop-from-open`, `--closing-hand-exempt`). pb2–pb5 는 넣지 않는다 |
| 2026-10-30 (~01:30 KST) | **target 신호를 action attention (`action:last:L8H2`, pb7 과 같은 기본 경로) 으로 되돌린다.** 명사 attention L8H6 은 §6 장면에서 같은 bowl 둘을 구분하지 못했다 (50 mm 안 3/11, 못 찾음 2 vs pb7 5/11 · 0) |
| 2026-10-30 (~07:00 KST, §8 결과 · task 1 충돌 진단 뒤) | §8 2판 미채택 (안전 성공 1 = pb7, 충돌 6 > 5). 진단 (`outputs/2026-10-30_task1충돌진단_Panda최종구성2판/`): t1e1 · t1e4 충돌은 TrajOpt 수정이 conversion C 에서 ±1 포화 action 이 된 직후였고, 2판 충돌 6 개 중 4 개는 팔 링크 (link6 · 7) 접촉이었다. **2판에는 pb7 누적판의 pb2–pb5 가 빠져 있었다** (lead 의 정의 실수: pb7 고유 flag 만 넣음) → **3판 = pb7 의 flag 9 개 전체를 넣는다** |
| 2026-10-30 (~07:00 KST) | **연속성 항과 꺾임 변화율 항을 전 구간에서 끈다** (`--w-continuity 0 --w-deflection-rate 0`, after-grasp flag 2 개는 뺀다). t1e1 의 20 mm 수정은 HELD 가 되어 두 항이 켜진 첫 call 에서 나왔다. 3판에 함께 넣고 한 번의 smoke 로 본다 (두 변경의 효과는 분리하지 않는다) |
| 2026-10-30 (~10:10 KST, §9 smoke 진행 중) | §9 다음에 **TrajOpt 를 E0 버전으로 한 smoke** 를 한다 (§10). "E0" = AG3S-PRE-CORE-TO 의 현재 TO 기준선 (E1–E3 이전). 정의 (사용자): 3판에서 **TO 설정 전체를 E0 때로** — GPU 여러 초기값 (A2) 과 최소 개입 gate (pb2) 를 빼고, **연속성 항 · 꺾임 변화율 항은 쥔 뒤에만** 쓴다 (`--continuity-after-grasp --deflection-rate-after-grasp`, 무게는 기본 0.5 · 20). 판정은 3판 대비 |
| 2026-10-30 (~12:00 KST, §10 결과 · 성공률 진단 뒤) | §10 판정대로 **E0 TrajOpt 채택** (E0판 안전 1 · 충돌 5 vs 3판 1 · 7). 성공률 진단 (`outputs/2026-10-30_작업성공률저하_원인진단_Panda/`): 실패 10 개 중 5 개는 놓기 단계에서 TrajOpt 가 하강을 막음, 4 개 (task 3) 는 target 을 다른 bowl · plate 로 고름. 사용자 판정: **(1) placement 단계 접촉 허용 = HSR6 placement ∧ 쥐고 있음 일 때 쥔 물체 행만 지지면 · 바로 아래 물체 (plate) 와의 접촉 허용, 손 · 팔 · 주변 장애물은 그대로** (§11). **(2) task 3 target 오류는 원인 분리 (attention 대 grounding) 를 먼저** 한다 (lead, CPU) |
| 2026-10-30 (~15:40 KST, task 3 원인 분리 뒤) | 원인 분리 (`outputs/2026-10-30_task3_target오류_원인분리/`): 첫 call (frame 20–30) 에는 진짜 bowl 이 손목 화면 밖이라 L8H2 가 plate · 다른 bowl 을 보고, frame 35–45 에 attention 90–99 % 가 진짜 bowl 로 옮겨 가도 target 이 바뀌지 않는다 (LATCHED 잠금 · grounding 유지). 사용자 판정: **target 교체를 넣는다** (§12). 사용자 질문 "입력 이미지 몇 장" → 2 장 (head · 손목, 세 번째 자리는 mask False) |
| 2026-10-30 (~17:15 KST, §11 smoke 뒤) | §11 pc판: 안전 1 · 공식 1 · 충돌 5 = E0판 → 규칙상 채택이지만 11 episode 결과가 E0판과 같다 (8 개 action bit 동일). 규칙이 3 episode · 10 call 에서만 행을 풀었다 (HSR6 가 plate 위를 transport 로 판정, latch 미인식, 쥔 물체 행 밖의 제약이 하강을 막음). `outputs/2026-10-30_Panda_placement접촉허용_SafeLIBERO-LI-dev11/`. 사용자 판정: **다음 단계 (제약 기록 진단 · §12 target 교체) 는 보류** |

## 1. Panda 최종 구성 (이식 대상) 의 정의

- = RB-Y1 최종 서버 argv (A6.impl §3) 를 Panda 기하로 옮긴 것 + 역변환 C (`lag_inverse_tracking`, adapter) + `--robot panda`.
- + `--continuity-after-grasp --deflection-rate-after-grasp` (사용자 판정 2026-10-29, 두 smoke 판 모두).
- pb1–pb7 flag 없음. `V11AB_PRE_MERGE_DEFAULTS` (`--no-roi-carve --blas-threads 0`) 도 붙이지 않는다 (최종 구성은 ROI carve 켬 · BLAS 1).
- Panda 가 이미 다르게 쓰는 값은 그대로: horizon H 10 · K 5 · 20 Hz · plan 10 (`--plan-horizon 16` 은 10 으로 대체), 관절 한계 = adapter URDF.
- **요청 수로 정한 값은 바꾸지 않는다** (`--hold-follow-to 3`, G1 `geo_close_timeout` · `geo_carry_requests`, HSR6 raw argmax). Panda 요청 간격은 0.25 s (RB-Y1 0.53 s) 라 시간으로는 절반이다 → 한계로 기록한다.
- 이름: 보고에서는 "Panda 최종 구성". profile key 는 `AG3S-final-panda` (G1 · A4 제거판), `AG3S-final-panda-g1a4` (작동판).

## 2. SL6-A — 서버 · 기하 이식 (implementer A, CPU)

**소유 파일:** `AG3S/adapters/panda/*`, `AG3S/trajopt/{safe_policy,gpu_seeds,multi_seed,sqp,geo_grasp,grasp_anchor,linearize,serve_safe}.py`, `AG3S/ag3s/grasp/*`, `AG3S/ag3s/robot_models/*`, `AG3S/ag3s/experiments/reports/grounding_report.py`, `benchmark/runners/launch_libero_paired.py` (version 등록), 그 test. **`AG3S/trajopt/attention_source.py` · `AG3S/trajopt/hsr6.py` · `AG3S/ag3s/stages/*` 는 SL6-B 소유 — 고치지 않고 호출만 한다.**

조사 근거: lead 의 read-only audit (2026-10-29, 아래 표의 file:line). 구현 때 다시 확인한다.

| # | 기능 | 해야 할 일 |
|---|---|---|
| A1 | launcher · wiring | profile 에 version `AG3S-final-panda` · `AG3S-final-panda-g1a4` 를 추가한다 (`V11AB_PRE_MERGE_DEFAULTS` 를 붙이지 않는 경로, `launch_libero_paired.py:71-75` · `profile.py:277`). `server.build_safe_server` 에 RB-Y1 `main` 과 같은 attention source (`serve_safe.py:3874-3903` 의 t57a kwargs) · `hsr6=` · `hsr6_probe` 전달 · GPU warm-up (`serve_safe.py:4097-4102`) 을 넣는다. flag 가 꺼지면 아무것도 넘기지 않는다 |
| A2 | target exclusion · ROI carve · evidence · follow | `profile.v11ab_config_dict` 에서 `target_exclusion` 거절을 풀고 `build_ag3s` 의 merge · 생존 검사 (`serve_safe.py:400-402`, `508-514`) 를 같게 옮긴다 |
| A3 | RB-Y1 기본값이 박힌 SafePolicy 상수 → 주입 (기본값 = 지금 RB-Y1 값) | 손목 관절 이름 (`safe_policy.py:2243` `{hand}_arm_{j}` → Panda `panda_joint5..7` 에 해당하는 RB-Y1 과 같은 손목 관절 집합), 손가락 이웃 접촉 (NBC) 손가락 쌍 (`safe_policy.py:2134` → `grasp_links` 쌍, Panda 는 `panda_leftfinger` · `panda_rightfinger` 둘 다), multi-seed 손바닥 link (`multi_seed.py:79` → `panda_hand`), A6 손바닥 기준점 (`safe_policy.py:78` `HOLD_LINK_SITE_OFFSET_M`) |
| A4 | GPU 초기값 (cuRobo) | `gpu_seeds.py` 의 RB-Y1 고정값 (URDF fallback `:126-133`, `CUROBO_BASE_LINK` `:65` 등, `TOOL_FRAMES` `:67`, `spec.py:79`) 을 robot 별로. Panda 는 `panda_libero.urdf`, base `panda_link0`, tool `panda_hand`. **build 실패 시 CPU 로 조용히 떨어지는 것 (`:658`) 은 Panda 서버 시작 log 에 경고 + config 기록** |
| A5 | 손바닥 껍질 구 (A3, `--palm-shell on --palm-shell-margin 5`) | **Panda 기하 정의 (고정):** RB-Y1 과 같은 생성 규칙 (`gripper_cover` 의 손바닥 껍질 = hold link 몸통 중 손가락 쪽 면을 덮는 구) 을 Panda 손 충돌 기하 (robosuite Panda hand mesh, 지금 `gripper0_right_gripper` 아래, `cover.py:53-54`) 에 적용한다. `contype=0` geom 을 건너뛰는 문제 (`cover.py:182`, `gripper_cover.py:149-150`) 는 껍질 생성에서만 mesh 를 읽도록 푼다. 구 개수 · 반지름 · 위치 (panda_hand frame) 를 impl.md 에 표로 적는다 |
| A6 | 이웃 접촉 허용 쥐기 (`--neighbour-palm-contact-mm 10`) | 기준점 = `panda_grip_site` (= `panda_hand` 원점 + (0, 0, 0.097) m, `panda_libero.urdf:121-126`). RB-Y1 은 hold link 앞 100 mm (`left_ee`). 허용 대상 구 = `panda_hand` 의 franka.yml 구 + A5 껍질 구. radius 0.12 · gate 0.06 · depth 10 mm · 쥐는 순간만 = RB-Y1 그대로 |
| A7 | 쥐기 목표 항 (A4) | 거절만 푼다 (`cost_overrides` 로 이미 전달, 닫는 행 = gripper 열 7) |
| A8 | 기하 grasp 보정 (G1) | `AG3S/ag3s/grasp/gripper.py` 에 Panda `ParallelGripperSpec` (MJCF 에서 측정: 손가락 pad, 최대 개도 0.08, 손바닥 box). frame 규약 차이 (RB-Y1: x = 닫는 방향, −z = 접근; Panda hand: y = 닫는 방향, +z = 접근) 를 spec 의 frame 변환으로 맞춘다. IK 관절 선택 prefix (`geo_grasp.py:117`, `:418`) 를 주입한다. **역변환이 gripper 열을 VLA 값 그대로 보내는 문제 (`action.py:257`)**: G1 이 닫기를 미룬 chunk 에서만 refined gripper 열을 보낸다 (G1 꺼짐 = bit 동일) |
| A9 | HSR ④ 적용 범위 flag (새, 기본 = 지금) | `--hsr6-objective-scope {all,nbc}`, 기본 `all` = 지금 규칙 (NBC · A4 · G1 모두 HSR6 = grasp 일 때만). `nbc` = ④ 를 손가락 이웃 접촉 허용에만 적용하고 A4 · G1 닫는 행 목표는 막지 않는다. RB-Y1 기본 동작 bit 동일 |
| A10 | 거절 목록 | 이식한 dest 만 `UNPORTED_FINAL_CONFIG_DESTS` · `UNPORTED_BUILD_KEYWORDS` 에서 뺀다. `--hsr6-probe` 는 Panda 에서 **필수** (기본값 = RB-Y1 probe → 거절) |
| A12 | 쥐기 전 연속성 · 꺾임 항 끔 | `continuity_after_grasp` · `deflection_rate_after_grasp` 를 거절 목록에서 빼고 두 Panda 최종 구성 version 에 넣는다 (cost-only, latch 단계 판정이라 Panda 기하 이식 없음). `--geo-continuity-off` 와 함께 parse · config 검사가 통과하는지 VA4 에서 확인 |
| A11 | 단계 label 없는 기능 | `placed_no_seed_certify` 는 3-class `home` label 이 필요해 Panda 에서 작동하지 않는다 (`types.py:1062-1075`). 고치지 않고 서버 시작 log 와 config 에 "inert on panda" 로 적는다 |

### 2.1 검증 (결과 전 고정)

| # | 검사 | 통과 조건 |
|---|---|---|
| VA1 | 단위 test | `.venv-ag3s` 로 `tests/trajopt` + `tests/benchmark` + 새 test (A3 주입 기본값 = RB-Y1, A5 구 수 · 위치, A6 기준점, A8 frame 변환: Panda 손이 닫는 방향 · 접근 방향을 spec 이 맞게 내는가, A9 두 값). 알려진 기존 실패 (`test_t43t_replay` numpy, `test_parallel_gpu_launcher` timing) 외 실패 0 |
| VA2 | RB-Y1 꺼짐 · 켜짐 동등성 | (a) SL5a golden (`tests/trajopt/_sl5a_golden.py`) 통과. (b) merge 검증 (2) 와 같은 RB-Y1 최종 구성 기록 재풀이 103 요청 refined bit 동일 (A3 주입의 기본값이 RB-Y1 동작을 바꾸지 않음) |
| VA3 | Panda V11AB 동등성 | 새 version flag 없이 `--robot panda` V11AB · V11AB-pb7 의 config dump 가 이전과 같다 (CPU). GPU 재확인은 lead (§5 G0) |
| VA4 | Panda 최종 구성 CPU 기동 | `AG3S-final-panda` · `-g1a4` argv 가 parse · 조합 검사 · config 생성을 통과 (checkpoint 미로딩, stub policy). config dump 에 A5 껍질 구 · A6 기준점 · gpu seed robot · ④ scope · inert 기능 목록이 있다 |

## 3. SL6-B — target 명사 · HSR6 · head 선택 도구 (implementer B, CPU)

**소유 파일:** `AG3S/trajopt/attention_source.py`, `AG3S/trajopt/hsr6.py`, `AG3S/ag3s/stages/*`, `AG3S/ag3s/experiments/studies/sl6_*.py` (새), `benchmark/runners/attention_audit.py`, `benchmark/adapters/libero_panda/*`, 그 test. SL6-A 파일은 고치지 않는다.

| # | 할 일 |
|---|---|
| B1 | **LIBERO target 명사 찾기.** 지금 `prompt_noun` 은 과일 4 단어 + `basket` 만 안다 (`attention_source.py:155-185`, `t49_noun_query_core.py:41-47`) → LIBERO 문장에서는 target 이 없어 AG3S 가 target 없이 돈다 (`attention_policy.py:226-231`). 규칙 (고정): `"pick up the <target phrase> ... and place it <prep> the <destination phrase>"` 에서 target phrase 의 마지막 명사 (head word) 를 target 명사로, 그 tokenizer piece 중 **마지막 piece** 를 query row 로 쓴다. 목적지 명사가 문장에 두 번 나와도 (예: plate) target 이 아니면 상관없다. 과일 · basket 문장은 지금과 bit 동일. 규칙에 맞지 않는 문장은 지금처럼 이유를 기록하고 target 없음 |
| B2 | **HSR6 class 부분집합.** `hsr6.py:80` 은 6 class 만 받는다 → npz `classes` 가 6 단계의 부분집합 (순서 유지) 이면 받는다. 없는 단계는 확률 0 으로 기록하고, 그 단계를 쓰는 규칙은 작동하지 않는다. RB-Y1 6 class probe 는 bit 동일 |
| B3 | **Panda 물리 단계 label** (CPU, 기존 기록에서). 입력: `results/benchmark/20261013T092537_714c891415d9_sl4b-dev-pi05only-t0-9e1-4/episodes/*_geom.npz` 의 `sim_state_image` (요청 시점 MuJoCo 상태) + BDDL target · 목적지. 규칙은 RB-Y1 T64 Part B (`t64_partB.py:51-82`) 를 옮긴다: **grasp** = gripper 닫기 명령이 시작된 뒤 두 손가락이 모두 target 에 닿기 전후 ~ lift 전, **lift** = 두 손가락 target 접촉 ∧ (target 2 cm 이상 상승 ∨ target 이 지지면과 접촉을 잃음 — libero_spatial t3 · t5 는 bowl 이 높은 곳에서 시작해 2 cm 를 못 오른다), **transport** = lift 지점에서 수평 5 cm 이상 이동, **placement** = target xy 가 목적지 영역 (목적지 geom 의 xy box 또는 region site) 안, **approach** = 그 밖 (실패 · 재시도 포함). retreat 없음 (성공 순간에 episode 가 끝남). 요청 단위 (kv feature 와 같은 call) 로 낸다 |
| B4 | **HSR6-Panda 학습.** feature = `results/benchmark/20261013T094739_714c891415d9_sl4b-attention-audit-dev-e1-4/attention/*_attn.npz` 의 `kv_feature` (서버 kv_L4 와 bit 동일이 SL4b 에서 확인됨, 40 episode · 약 873 요청). 모델 = RB-Y1 과 같음 (StandardScaler + multinomial logistic, `t64_phase6.py:54-143`), C 는 episode 단위 GroupKFold(5) 로 {0.003, 0.01, 0.03, 0.1, 0.3} 중 macro-F1 최대. 출력 `probe_6.npz` 형식 (mean · scale · coef · intercept · classes). 보고: episode 단위 5-fold macro-F1, 단계별 recall, grasp 단계 시작 지연 (요청 수) |
| B5 | **head 선택용 noun attention 추출 mode.** `attention_audit.py` 에 `--noun-rows` (B1 의 query row 에서 image token 으로 가는 attention, layer 2–17 × head 0–7 × 768 token per camera) 를 더한다. 입력은 위 pi05-only run 의 `policy_records` 로 요청을 다시 만든다 (GPU, lead 가 돌린다) |
| B6 | **head 선택 계산** (CPU). §4 의 기준 그대로 |

### 3.1 검증 (결과 전 고정)

| # | 검사 | 통과 조건 |
|---|---|---|
| VB1 | 단위 test | 기존 test + B1 (LIBERO spatial 10 문장 · SafeLIBERO 문장 · RB-Y1 과일 문장 bit 동일), B2 (6 class bit 동일, 5 class 부분집합), B3 (작은 합성 상태로 각 단계 경계) |
| VB2 | label 눈 확인 | lead 가 dev 4 episode 의 단계 띠 + 영상 frame 을 사용자에게 보인다 (RB-Y1 T64 와 같은 절차). 사용자 확인 전에 B4 결과로 서버를 띄우지 않는다 |
| VB3 | HSR6 사용 조건 | episode 단위 5-fold **macro-F1 ≥ 0.70** 이고 grasp 단계 recall ≥ 0.6 이면 서빙에 쓴다. 아니면 멈추고 보고한다 (④ 가 손가락 이웃 접촉 허용을 잘못 막을 수 있으므로) |
| VB4 | 서버 parity | Panda 서버가 기록 요청에서 낸 HSR6 단계 = offline B4 단계, 요청 ≥ 50 중 ≥ 99 % (RB-Y1 H3 와 같음) |

## 4. attention head 재선택 (결과 전 고정)

- 후보: noun query 의 layer 2–17 × head 0–7 = 128 cell (RB-Y1 T51b 와 같은 후보 집합). 참고 줄: 옮겨 온 `noun_target:L10H7`, Panda V11AB `action:last:L8H2`.
- 데이터: libero_spatial task 0–9 × init 1–4 (40 episode, dev) 의 **쥐기 전 요청** (B3 label 이 approach 인 요청), head 카메라. smoke 장면 (§6) 과 LIBERO task 0–9 init 0 은 쓰지 않는다.
- 정답: 요청 시점 head 카메라 segmentation (`segmentation_geom/head`) 에서 BDDL target 물체의 geom.
- **주 기준: 2D identity 정답률의 task 평균** = attention 질량 상위 1 % pixel 중 가장 많은 물체가 target 인 요청의 비율. 동점이면 target 질량 비율 (target pixel attention 합 / 전체), 다음 target 덮은 비율 (target pixel 중 상위 10 % 에 든 비율).
- 가장 높은 cell 을 쓴다. 최고 cell 과 L10H7 의 차가 2 %p 미만이면 L10H7 을 유지한다 (같은 값이면 RB-Y1 과 같은 cell 이 논리상 깔끔하다).
- 한계 (미리 적음): libero_spatial 은 같은 모양의 검은 bowl 이 둘이라 명사 하나로는 구분이 안 될 수 있다. RB-Y1 의 3D 기준 (grounding replay) 은 빠른 실험을 위해 생략한다.

## 5. GPU 단계 (lead, 순서대로)

| # | 내용 | lock |
|---|---|---|
| G0 | SL6-A 후 Panda V11AB-pb7 · 역변환 C, LIBERO task 0–9 init 0 10 episode → merge 검증 (3) 의 run 과 npz array bit 동일 | 독점 |
| G1 | B5 noun attention 추출 (873 요청) | 공유 |
| G2 | Panda 최종 구성 1 episode 기동 확인 (SafeLIBERO Level I dev t0e0 — smoke 장면이 아닌 것). gpu seed robot 이 Panda 로 지어졌는지, 요청 시간 | 독점 |
| G3 | §6 smoke 22 episode (2 판 × 11). **동시 1 job** (`run_parallel_gpu.sh --max-parallel 1`): 최종 구성은 TrajOpt 시간 마감 · GPU 초기값을 써서 GPU 를 나눠 쓰면 결과가 바뀔 수 있다 (V11AB 의 2 job bit 동일 근거를 옮기지 않는다) | 독점 |

## 6. G1 · A4 포함 / 제거 smoke (결과 전 고정, 사용자 판정 2026-10-29)

- **장면:** SafeLIBERO spatial Level I dev 중 VLA 단독이 공식 성공한 11 episode — t1e1, t1e4, t2e0, t2e1, t2e2, t2e3, t2e4, t3e0, t3e2, t3e3, t3e4 (SL5i-G verify `I1`). 그중 VLA 단독이 장애물과 충돌한 것 9 (safe success 는 t1e1 · t1e4 둘). 같은 init · policy seed (paired).
- **두 판** (둘 다 Panda 최종 구성 + `--hsr6-objective-scope nbc` + 쥐기 전 연속성 · 꺾임 항 끔, 차이는 A4 · G1 flag 뿐):
  - **제거판:** `--grasp-anchor` · `--geo-*` 없음.
  - **작동판:** + `--grasp-anchor 2000 --grasp-anchor-path-weight 0 --geo-grasp --geo-goal-mode correct --geo-track-scale 0.25 --geo-continuity-off` (대체 방식: 기하 후보가 있으면 G1, 없으면 A4).
- **주 지표:** safe success (공식 성공 ∧ 장애물 충돌 없음, SafeLIBERO 판정기).
- **판정:** 작동판 safe success ≥ 제거판 + 2 episode **이고** 충돌 episode 수가 늘지 않으면 **포함**. 아니면 **제거**.
  - 포함으로 나오면 Panda 구성이 RB-Y1 측정 구성 (④ 가 두 항을 막음) 과 달라진다 → 사용자에게 RB-Y1 쪽 처리를 묻는다.
- **함께 보고 (판정에는 쓰지 않음):**
  - 두 항이 목표를 맡은 요청 수 (G1 / A4 각각). 그중 HSR6 ≠ grasp 라 `scope all` 이면 막혔을 요청 수.
  - 공식 성공 · 충돌 · HOLD 요청 · 손바닥의 목표 오차.
  - VLA 단독 (SL5f run 재사용, VLA 경로 코드 무변경을 근거로) 과 파지 인식 + closing 면제판 (SL5i-G) 대비 전이표.
  - 실패 episode 마다 원인 층 (ATS · HSR · LTE · CCR) 과, 이전 pb 가 고친 원인과 같은지 (같으면 해당 pb 추가를 제안).
- 11 episode × 1 회라 효과 크기를 판정하지 않는다. 위 규칙만 적용한다.
- 결과 보기: `outputs/2026-10-29_Panda최종구성_G1A4포함제거_SafeLIBERO-LI-dev11/` (영상 `common.experiment_view`, 실제 씬, 그래프, 표, README).

### 6.1 결과 (2026-10-30 01:20 KST)

- `outputs/2026-10-29_Panda최종구성_G1A4포함제거_SafeLIBERO-LI-dev11/` (README · summary.json).
- 안전 성공: 제거판 0 · 작동판 1 → **제거**. 8/11 episode 는 두 판 bit 동일.
- 실패 원인 층:
  - ATS 8 (다른 물체 6 · 못 찾음 2).
  - HSR → CCR 5 (쥔 bowl 미인식 → TrajOpt 위로 밈).
  - 충돌 3 (원인 층 미진단).

## 8. SL6-C — Panda 최종 구성 2판 smoke (결과 전 고정, 사용자 판정 2026-10-30)

**2판의 정의:** profile key `AG3S-final-panda-v2`.
- = §1 Panda 최종 구성 (G1 · A4 없음)
- `--attention-source noun_target:L8H6` 를 뺀다 → 기본 `action:last:L8H2` (pb7 과 같은 경로).
- \+ `--reach-at-capture` (pb1) · `--grasp-reach-sticky --grasp-drop-from-open` (pb6) · `--closing-hand-exempt` (pb7).
- 나머지 (HSR6-Panda probe, scope `nbc`, 쥐기 전 연속성 · 꺾임 항 끔, 변환 C) 는 그대로.

**구현 · CPU 확인 (lead):**
- C1 — `profile.py` 에 2판을 등록한다. 기존 판의 argv 는 바꾸지 않는다.
  - attention flag 가 없는 판에 run 의 `--attention-source` 를 주면 덧붙인다.
- C2 — 단위 test 를 둔다. 확인 항목:
  - 2판 argv = §1 판 − attention 2 token + pb flag 4 개.
  - 기존 판 argv 는 그대로.
- C3 — CPU 기동 (VA4 방식): parse · 금지 조합 · 미이식 flag 검사 · config dump.
  - pb flag 가 SafePolicy 에 들어갔는지 확인한다 (`grasp contact rules` 기록, `reach_at_capture`).
- C4 — GPU 전에 V11AB-pb7 argv 가 바뀌지 않았음을 확인한다 (C2 test).

**smoke (lead, GPU 독점, 동시 1 job):**
- 장면 · seed: §6 과 같은 11 episode.
- 실행: 2판만 새로 돌린다. 비교는 §6 제거판 · pb7 (SL5i-G) · VLA 단독 (SL5f) 기존 run 으로 한다.
- 주 지표: safe success.
- **판정 (결과 전 고정):** 아래 둘을 모두 만족하면 2판을 Panda 최종 구성으로 삼고 다음 단계 (더 큰 평가) 를 제안한다.
  - 2판 safe success ≥ pb7 (1).
  - 2판 충돌 episode ≤ pb7 (5).
- 아니면 실패 원인 층을 다시 적고 사용자에게 묻는다.
- **함께 보고 (판정에 쓰지 않음):**
  - 공식 성공 · HOLD call.
  - target 선택 오차 (첫 기록 call, 50 mm 기준) · target 못 찾은 episode.
  - "쥔 bowl 미인식" episode 수 (MuJoCo 에서 target bowl 이 2 cm 이상 들리고 손과 9 cm 안인 frame > 20 ∧ 첫 닫기 뒤 latch CLOSING / LATCHED call > 5).
  - 그 구간의 Σ (실행 z − VLA z).
  - 실패 원인 층.
  - §6 제거판 · pb7 대비 전이표.
- 결과 보기: `outputs/2026-10-30_Panda최종구성2판_쥠인식pb_SafeLIBERO-LI-dev11/`.

### 8.1 결과 (2026-10-30 03:40 KST) · 진단 (06:50 KST)

- 2판: 안전 성공 1 · 공식 성공 2 · 충돌 6 · HOLD 38 / 574. **판정: 미채택** (충돌 6 > pb7 5). `outputs/2026-10-30_Panda최종구성2판_쥠인식pb_SafeLIBERO-LI-dev11/`.
- task 1 충돌 진단 (CPU, 기록 재생): `outputs/2026-10-30_task1충돌진단_Panda최종구성2판/`.
  - t1e1: HELD 전환 call 의 수정 20 mm → 포화 action → 들고 있던 bowl 이 storage box 에 닿음 (substep 재현).
  - t1e4: CLOSING 중 수정 24 mm → 포화 action → HOLD 뒤 관성 → 손 본체가 mug 에 닿음.
  - 2판 충돌 접촉 몸체: 팔 링크 4 · 손 1 · bowl 1. pb7: 팔 링크 2 · 손가락 1 · bowl 2.

## 9. SL6-E — Panda 최종 구성 3판 smoke (결과 전 고정, 사용자 판정 2026-10-30 ~07:00 KST)

**3판의 정의:** profile key `AG3S-final-panda-v3`.
- = 2판에서 `--continuity-after-grasp --deflection-rate-after-grasp` 를 뺀다.
- \+ V11AB-pb7 의 flag 9 개 전체: `--reach-at-capture --minimal-intervention-mm 10 --links arms --held-hand-exempt --latch-phase --grasp-zone-mm 10 --grasp-reach-sticky --grasp-drop-from-open --closing-hand-exempt` (pb1 · pb6 · pb7 의 4 개는 2판에 이미 있다; 같은 flag 를 두 번 넣지 않는다).
- \+ `--w-continuity 0 --w-deflection-rate 0` (연속성 · 꺾임 변화율 항 전 구간 끔).
- 나머지 (action attention L8H2, HSR6-Panda probe, scope `nbc`, 변환 C) 는 2판 그대로.

**구현 · CPU 확인 (lead):**
- E1 — `profile.py` 에 3판을 등록한다. 기존 판 (V11AB 계열 · 1판 · 작동판 · 2판) argv 는 바꾸지 않는다.
- E2 — 단위 test: 3판 argv 정의, 기존 판 불변, argparse 결과 (`links = arms`, `minimal_intervention_mm = 10`, `w_continuity = 0`, `w_deflection_rate = 0`, after-grasp 둘 꺼짐).
- E3 — CPU 기동: 최종 구성 flag 와 pb2–pb5 가 함께 받아들여지는지 (금지 조합 · 미이식 flag 검사), SafePolicy 에 pb 기능 · 무게 0 이 들어갔는지, 손바닥 껍질 구가 팔 링크 모델 위에서 붙는지.
- E4 — launcher dry-run 으로 서버 인자를 확인한다.

**smoke (lead, GPU 독점, 동시 1 job):**
- 장면 · seed: §6 · §8 과 같은 11 episode. 3판만 새로 돌린다.
- 주 지표: safe success.
- **판정 (결과 전 고정, §8 과 같은 기준):** 아래 둘을 모두 만족하면 3판을 Panda 최종 구성으로 삼고 다음 단계 (더 큰 평가) 를 제안한다.
  - 3판 safe success ≥ pb7 (1).
  - 3판 충돌 episode ≤ pb7 (5).
- 아니면 실패 원인 층을 다시 적고 사용자에게 묻는다.
- **함께 보고 (판정에 쓰지 않음):** 공식 성공 · HOLD call, 처음 닫는 시점 target 정답 수, "쥔 bowl 미인식" episode 수, **포화 action step 수**, **충돌마다 장애물에 닿은 몸체** (기록 재생), 2판 · pb7 대비 전이표, 실패 원인 층.
- 결과 보기: `outputs/2026-10-30_Panda최종구성3판_pb7전체_연속성꺾임끔_SafeLIBERO-LI-dev11/`.

## 10. SL6-F — 3판 + TrajOpt E0 smoke (결과 전 고정, 사용자 판정 2026-10-30 ~10:10 KST)

**E0판의 정의:** profile key `AG3S-final-panda-v3-e0to`.
- = 3판 (§9) 에서 TO 설정만 E0 때로 바꾼다.
  - 뺀다: `--to-multi-seed 16 --to-gpu-seeds curobo --to-gpu-seeds-check first-pass` (A2 GPU 여러 초기값), `--minimal-intervention-mm 10` (pb2), `--w-continuity 0 --w-deflection-rate 0`.
  - 더한다: `--continuity-after-grasp --deflection-rate-after-grasp` (두 항은 HELD 부터만; 무게는 V11AB 기본 연속성 0.5 · 꺾임 변화율 20).
- 그대로: 로봇 충돌 모델 (팔 link1–7 + 손 + 손바닥 껍질 구), 쥠 판단 · 접촉 규칙 (pb1 · pb4–pb7, HSR6), 거리장 · target 제외, action attention, 변환 C. 시간만 바꾸는 `--to-fast-eval` · `--blas-threads 1` 도 그대로 (명령 bit 동일).

**구현 · CPU 확인 (lead):** §9 E1–E4 와 같은 꼴 (등록, 기존 판 argv 불변 test, CPU 기동에서 multi-seed 0 · gpu seeds off · minimal intervention 없음 · after-grasp 두 항 켜짐 확인, launcher dry-run).

**smoke (lead, GPU 독점, 동시 1 job):** §6 · §8 · §9 와 같은 11 episode. E0판만 새로 돌린다.
- **판정 (결과 전 고정):** 아래 둘을 모두 만족하면 E0 TO 설정을 Panda 구성에 채택한다 (같으면 더 단순한 TO 를 고른다). 아니면 3판의 TO 설정을 유지한다.
  - E0판 safe success ≥ 3판.
  - E0판 충돌 episode ≤ 3판.
- **함께 보고 (판정에 쓰지 않음):** 공식 성공 · HOLD call, 포화 action step 수, 충돌마다 장애물에 닿은 몸체, TrajOpt 가 고치지 않은 call 수, 3판 대비 전이표, 실패 원인 층.
- 결과 보기: `outputs/2026-10-30_Panda최종구성3판_TO-E0_SafeLIBERO-LI-dev11/`.

## 11. SL6-H — placement 단계 쥔 물체 접촉 허용 (기본 꺼짐 flag, 사용자 판정 2026-10-30 ~12:00 KST)

**근거:** 진단 §0 마지막 행. E0판 t1e1 · t1e4 · t2e0 · t2e2 · t2e4 에서 bowl 이 plate 위 0.08–0.44 m (VLA 0.10–0.15 m) 에 머물고, 하강 명령이 위로 바뀐 row 14–66. pb7 t2e4: 쥔 bowl 모델 바닥 여유 34 → −10 mm 에서 VLA z −1.7 → 실행 +5. Panda 에는 목적지 (plate) 처리 기록이 없다.

**H1 — flag (implementer):** `--hsr6-placement-contact` (기본 꺼짐; 꺼지면 모든 판 bit 동일).
- 켜지는 call: HSR6 argmax = `placement` **∧** (latch HELD **∨** (latch CLOSING ∧ evidence `closed` ∧ `blocked`)).
- 허용 대상: **쥔 물체 (attached held object) 의 질의 구 행만.** 그 행이 **아래에 있는 표면** — 지지면 (탁자) 과 쥔 물체의 xy 발자국 아래에 있는 물체 표면 (plate) — 과 닿는 것을 막지 않는다.
- 그대로 막는 것: 손 · 손가락 · 팔 행 전부, 쥔 물체 행이라도 옆이나 위에 있는 표면 (발자국 밖의 장애물).
- 쥔 물체가 attach 되지 않은 call (CLOSING 미인식 등) 에서는 바꿀 행이 없다 — 아무것도 바꾸지 않고 그 사실을 기록한다.
- "아래 표면" 의 정확한 판정 (거리장 label, 최근접점이 구 중심 아래이고 발자국 반경 안, 지지면 band 등) 은 기존 구조 (held-body cover · support band · `hsr6_held_gate`) 를 읽고 implementer 가 정해 impl.md 에 쓴다. 판정이 장애물을 함께 풀 수 있는 경우가 있으면 그 경우를 적는다.

**H2 — 기록 (implementer):** call 마다 `hsr6` phase (argmax · 확률) 와 placement 접촉 허용의 on/off · 풀린 행 수를 Panda call 기록 (`_calls.jsonl`) 에 남긴다 (지금은 HSR6 phase 가 call 기록에 없다).

**H3 — profile (implementer):** `AG3S-final-panda-v3-e0to-pc` = E0판 (`AG3S-final-panda-v3-e0to`) + `--hsr6-placement-contact`. 기존 판 argv 불변.

**H4 — test (implementer, CPU):**
- 합성 장면: 쥔 구가 판 모양 물체 위로 내려갈 때 그 표면 때문에 생기던 제약 / 위반이 flag on + placement + HELD 에서 사라지고, flag off · phase ≠ placement · latch 미충족에서는 그대로.
- 같은 쥔 구 옆에 세운 장애물은 flag on 에서도 제약된다.
- 손 · 손가락 행은 flag on 에서도 바뀌지 않는다.
- flag off 회귀: 기존 판 (V11AB-pb7 · 2판 · 3판 · E0판) argv · CPU 기동 결과 불변, tests/benchmark 전체 통과.
- Panda 서버가 flag 를 probe 와 함께 받아들이고 (probe 없이는 거절), CPU 기동에서 켜진 것을 확인.

**smoke (lead, GPU 독점, 결과 전 고정):** 같은 11 episode, `AG3S-final-panda-v3-e0to-pc` 만 새로 돌린다.
- **판정:** pc판 safe success ≥ E0판 (1) **이고** 충돌 ≤ E0판 (5) 이면 placement 접촉 허용을 Panda 구성에 채택한다. 아니면 원인 층을 다시 적고 사용자에게 묻는다.
- **함께 보고:** 공식 성공, 도달 단계 (진단과 같은 정의), plate 위 높이, 하강이 뒤집힌 row, placement 허용이 켜진 call · 풀린 행, 충돌마다 닿은 몸체, E0판 대비 전이표.
- 결과 보기: `outputs/2026-10-30_Panda_placement접촉허용_SafeLIBERO-LI-dev11/`.

## 12. SL6-I — 쥐기 전 attention 이 옮겨 가면 target 교체 (기본 꺼짐 flag, 사용자 판정 2026-10-30 ~15:40 KST)

**I1 — flag (implementer, §11 구현이 끝난 뒤):** `--attention-target-switch` (기본 꺼짐; 꺼지면 모든 판 bit 동일). 부속 값은 결과 전에 고정한다: 연속 call 수 N = 2, 몫 기준 0.5.
- 대상 단계: latch 가 SEARCHING · LATCHED 일 때만. CLOSING · HELD · 놓은 뒤에는 바꾸지 않는다.
- 조건: 지금 target 이 아닌 후보 하나가 **연속 N call 동안** 그 call 의 target 후보 attention 질량 (grounding 이 후보를 고를 때 쓰는 정규화 질량) 의 **0.5 이상** 을 가져간다.
- 동작: target 을 그 후보로 바꾼다. LATCHED 였으면 latch 를 그 후보로 다시 잡는다 (`--latched-identity-hold` 의 잠금을 이 경우에만 푼다). 이전 target 은 다른 물체와 같이 장애물로 돌아간다.
- 기록: call 마다 후보별 질량 몫 (상위 3 개), 연속 count, 교체 여부 · 이전 / 새 target.
- grounding 이 SEARCHING 중에도 target 을 유지하는 규칙 (t3e2) 이 무엇인지 implementer 가 찾아 impl.md 에 쓰고, 이 flag 가 그 경우에도 작동하게 한다.

**I2 — profile · test:** 새 판 = 그때 채택된 판 (§11 판정 결과: 채택이면 `…-pc`, 아니면 E0판) + `--attention-target-switch`. 합성 test: 교체 (N 연속 · 0.5 이상) / 비교체 (1 call 만, 0.5 미만, CLOSING 이후), 잠금 해제, flag off 회귀 (tests/benchmark 전체).

**smoke (lead, GPU 독점, 결과 전 고정):** 같은 11 episode, 새 판만 새로 돌린다.
- **판정:** 새 판 safe success ≥ 기준판 **이고** 충돌 ≤ 기준판 이면 target 교체를 Panda 구성에 채택한다. 기준판 = 위 I2 의 바탕 판. 아니면 원인 층을 다시 적고 사용자에게 묻는다.
- **함께 보고:** task 3 의 처음 닫는 시점 target 정답 수 (기준판 0 / 4), 교체가 일어난 call · episode, task 1 · 2 에서 교체로 나빠진 episode, 도달 단계, 충돌마다 닿은 몸체.

## 7. 산출물

- implementer A · B: `SL6-A.impl.md` · `SL6-B.impl.md` (바꾼 파일 · md5, test, 위 검증 숫자, 서버 argv 두 줄, 예상 결과는 맨 끝 절에 따로).
- commit 은 lead 가 한다. implementer 는 commit 하지 않는다. storage 의 기존 run 은 읽기만 한다. 새 출력은 새 run (`source scripts/setup/env.sh`, `results/benchmark/<…>`) 아래에만 쓴다.
- CPU 만, 1 process, `nice -n 10`. 다른 세션 process 는 건드리지 않는다.
