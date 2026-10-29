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

## 1. Panda 최종 구성 (이식 대상) 의 정의

- = RB-Y1 최종 서버 argv (A6.impl §3) 를 Panda 기하로 옮긴 것 + 역변환 C (`lag_inverse_tracking`, adapter) + `--robot panda`.
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
- **두 판** (둘 다 Panda 최종 구성 + `--hsr6-objective-scope nbc`, 차이는 A4 · G1 flag 뿐):
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

## 7. 산출물

- implementer A · B: `SL6-A.impl.md` · `SL6-B.impl.md` (바꾼 파일 · md5, test, 위 검증 숫자, 서버 argv 두 줄, 예상 결과는 맨 끝 절에 따로).
- commit 은 lead 가 한다. implementer 는 commit 하지 않는다. storage 의 기존 run 은 읽기만 한다. 새 출력은 새 run (`source scripts/setup/env.sh`, `results/benchmark/<…>`) 아래에만 쓴다.
- CPU 만, 1 process, `nice -n 10`. 다른 세션 process 는 건드리지 않는다.
