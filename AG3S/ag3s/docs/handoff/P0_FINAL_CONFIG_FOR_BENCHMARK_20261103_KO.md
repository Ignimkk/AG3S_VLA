# P0 최종 구성 — benchmark checkout 으로 옮길 때 알아야 할 모든 것

> writer: lead (development 세션 ce) · 2026-11-03 KST · 읽는 쪽: benchmark checkout 세션 (Panda · SafeLIBERO · LIBERO-safety)
> 상태: **사용자가 P0 를 최종 구성으로 두는 데 찬성 (2026-11-03 오전).** **갱신 (2026-11-03 18:2x KST, 사용자 판정): 최종 구성 = P0m = P0 + `--subtask-from-latch` (T92).** 최종 확정은 T92b (seed 반복 + hull · 접촉 여유 짝 비교) 결과로 한다 — T92b 에서 hull 채우기 · 접촉 여유 10 mm 를 뺀 구성이 차이 ≤ 2 run 이면 그 구성이 최종이다 (`T92b.task.md`).
> 코드 기준: development HEAD `f964439d` (코드 `63e98cc8`), main `32a45fb8` (+ 이식 지시 문서 merge `a0b16d0a`). **모든 새 flag 는 기본 꺼짐.**
> 근거: `outputs/2026-11-03_T88_P0_HSR끔_G1끔_단순구성_closedloop/README.md` (RB-Y1 MuJoCo 같은 10 run: 옛 정의 7 / 10, 새 정의 9 / 10, 금지 접촉 0; 허들 211 제외 9 run 으로는 7 / 9 · 9 / 9).

## 0. 한 줄 요약

**target 범위는 공이 아니라 ATS 표면 추출 (depth · point 연속 표면, GPU) 로 정한다. 단계 규칙 (HSR6 ①–④) · 5 단계 분류기 · G1 · 쥐기 목표 항 · 여러 초기값 TrajOpt 는 쓰지 않는다. 쥠 상태와 3 단계 gate label 은 모두 grasp latch (그리퍼 측정 개도 · feedback) 로 정하고 (P0m, 학습 요소는 VLA 뿐), 쥐기 직전 · 쥐는 순간의 접촉 허용은 latch 와 손바닥–target 거리 창으로 켠다.**

## 1. 구성 요소별 (P0 에서 무엇을 쓰고 무엇을 안 쓰나)

### 1.1 ATS — target 찾기 · 범위

| 항목 | P0 | flag |
|---|---|---|
| attention 출처 | 명사 token 의 attention, RB-Y1 16D 에서 고른 head **L10H7** | `--attention-source noun_target:L10H7` |
| target cluster 분할 (attention, 3D) | 12 mm · raw 비율, 분할이 anchor 보다 우선 | `--attention-split-radius-mm 12 --attention-split-ratio-source raw --attention-split-partition --split-supersedes-anchor` |
| **target 범위 (최종)** | **표면 추출 (T78): depth 연속 graph (5 mm node, jump 10 mm) 위에서 attention seed 로 영역을 고름. 공 (ball) 은 끔** | `--target-extent attention-surface --surface-no-ball --attention-surface-device cuda` |
| 영역 검사 · 유지 | 영역이 ATS target 점과 10 mm 안에서 50 % 이상 겹쳐야 채택 · 영역이 없는 요청은 직전 채택 영역 유지 · 쥐는 동안 영역 고정 | `--surface-target-check overlap --surface-keep-last --surface-grasp-hold` |
| 영역 분할 (그리퍼 폭 · attention) | **쓰지 않음 (폐기)** | `--surface-split` 금지 |
| 쥐기 직전 hull 채우기 | latch latched · closing 또는 손바닥이 영역 중심 80 mm 안: 마지막 영역 convex hull + 5 mm 안 seed 를 손 층에서 지움 | `--surface-hull-fill` |
| 손 층 target 접촉 여유 | target 표본 10 mm 안 · 지지면 guard 위 seed 를 지움 (구간 제한 없음) | `--surface-contact-allowance-mm 10` |
| 지지면 guard (바닥 띠) | 사과 바닥 + 1 미세 voxel (5 mm) 아래는 **일부러 남김** (탁자를 지우지 않으려고). **주의: 이 guard 5 mm 와 잔여 voxel 측정에서 빼는 띠 (바닥 + 6 mm 이하) 는 다른 값이다** — 6 mm 는 평가 정의 (사용자 2026-11-02) | 코드 기본 (LTE-16) |
| 기존 ball-owned 인자 | 명령줄에 남아 있으나 **무효** — `--target-extent attention-surface` + `--surface-no-ball` 이면 target 범위는 표면 추출이 정하고 공은 쓰지 않는다 | `--target-exclusion ball-owned --owned-max-dist-voxels 3` (명령줄에만 남음) |
| 비 target 증거 (`--nontarget-evidence surface`) | **쓰지 않음** — T46c 에서 채택했던 것을 표면 추출이 대체 | **금지** |
| target 유지 기억 · 오목 꺾임 끊기 · corridor | 쓰지 않음 | `--surface-memory-target-sticky` · `--surface-crease-cut` · `--grasp-corridor-*` 금지 |

### 1.2 HSR — 단계 · 쥠 상태

| 항목 | P0 | flag |
|---|---|---|
| 단계 규칙 4 개 (HSR6 ① 쥠 인정 보조 · ② 쥔 물체 허용 gate · ③ 물러남 target 권한 거둠 · ④ grasp 단계만 이웃 허용) | **끔** | `--hsr6-*` 금지 |
| 5 단계 분류기 | **끔** | `--hsr5` 금지 |
| **3 단계 gate 의 label** | **P0 에서는 옛 3 단계 분류기 (`kv_L4_v1`, 서버 기본 on) 가 켜져 있었다** ("분류기 없음" 은 틀린 표현 — 정정 2026-11-03). **P0m 에서 grasp latch 로 대체** (searching · latched · closing → pick, held → place, placed → home). gate 동작 (place / home 이면 새 target 선택 막음, PLACED ∧ home 이면 target 해제) 은 같다. T92: 9 run 9 / 9, 놓기 전 막음 0 | **`--subtask-from-latch` (명시)** — `--hsr5` · `--hsr-gt-stage` · `--no-subtask-gate` · `--subtask-from-phase-machine` 와 같이 못 줌 |
| 손목 depth 단계 (T91c) | 기본 꺼짐 — 손목 카메라가 있는 benchmark 의 선택지 | `--subtask-from-phase-machine` |
| 쥠 상태 (grasp latch) | 찾기 → 정함 (같은 target 3 요청, 1 위 ≥ 2 위 × 1.3) → 닫는 중 (측정 개도 < 0.85) → 쥠 (reach ≤ 0.12 m · 개도 멈춤 · 줄어듦 ≥ 0.05 · 막힘 ≥ 0.05 · 빈손 아님) → 놓음 (다시 열림 · 닫힌 채 빠짐 2 요청). 거짓 쥠 4 chunk 안 회수 | `--held-slip-detach --latched-identity-hold --closing-keep-geometry --kept-geometry-follow --placed-no-seed-certify` |
| G1 기하 grasp 보정 · A4 쥐기 목표 항 | **끔** (사용자: 켜라고 할 때까지 끔) | `--geo-*` · `--grasp-anchor` 금지 |

### 1.3 LTE — 거리장 · 로봇 모델

| 항목 | P0 | flag |
|---|---|---|
| 거리장 | cuRobo, 거친 20 mm · 미세 5 mm · TSDF 5 mm | `--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --esdf-margin 0` |
| 제약 링크 | 양손 손바닥 + 손가락 4 개 (팔뚝 · 몸통 · 반대팔 제외) | `--links gripper --no-self-collision` |
| target 예외 | 쥐는 손 (권한 link) 행만 target 이 빠진 손 층을 봄 | `--target-field-policy exclude-authorized --target-permission-strict` |
| 구 모델 | capsule 반지름 × 0.05 · 간격 0.4 · 최대 32 / capsule · 손가락 덮개 · 손바닥 껍질 (5 mm) | `--capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --finger-cover both --palm-shell on --palm-shell-margin 5` |
| 쥔 물체 | 손의 일부 (쥔 구 반지름 ≤ 관측 반폭 + 10 mm) · 받침면 근처일 때 받침면 접촉 허용 | `--held-body-cover --held-cover-cap-mm 10 --supported-max-bottom-mm 60` |
| ROI carve | target · 목적지 · 쥐는 손 경로 상자 ± 0.10 m 의 빈 공간을 지움 (GPU) | `--roi-carve --roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore --roi-carve-device cuda` |

### 1.4 CCR — TrajOpt

| 항목 | P0 | flag |
|---|---|---|
| 풀이 | chunk 당 표준 SQP 하나 (E0) · 계획 16 행 · 장애물 가까울 때만 look-ahead · QP eps 1e-4 | `--plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --to-fast-eval --esdf-host-mirror --blas-threads 1` |
| 여러 초기값 (GPU) | **끔** | `--to-multi-seed` · `--to-gpu-seeds` 금지 |
| 여유 | 장애물 행 10 mm (target · 지지면 · 쥔 물체 행 0) · 손바닥 껍질 5 mm | `--obstacle-margin 0.010` |
| 쥐기 직전 손가락 이웃 허용 | 20 mm 겹침, 창 = latch latched · closing ∧ 손바닥 ≤ target 150 mm | `--neighbour-contact-mm 20` |
| 쥐는 순간 손바닥 이웃 허용 (A6) | 10 mm 겹침, 창 = 쥠 전 ∧ 손바닥 site ≤ target 60 mm | `--neighbour-palm-contact-mm 10` |
| 쥐기 전 손목 추종 약화 | target 150 mm 안에서 손목 관절 × 0.1 | `--wrist-track-scale 0.1` |
| 꺾임 변화율 · 연속성 항 | 쥔 뒤에만 켬 (w_deflection_rate 20) | `--w-deflection-rate 20 --deflection-rate-after-grasp --continuity-after-grasp` |
| HOLD 대신 TrajOpt 결과 | 최대 3 chunk | `--hold-follow-to 3` |
| 충돌 없으면 그대로 통과 (T88) | **끔 (미채택)** | `--passthrough-when-clear` 금지 |

### 1.5 client

`--safe-remote --safe-phase approach --safe-manipulators <쥐는 손> --safe-gate reasons --safe-hold-mode measured --safe-timeout 120 --obstacle-stop-distance 0 --stop-after-place 2.5`

## 2. P0 서버 명령줄 전체 (RB-Y1, 기록 그대로)

```
--config pi05_rby1_randomized_pick_place_16d_lora --checkpoint <RB-Y1 16D 30k> --model-xml <snapshot>/robots/rby1/description/models/rby1a/mujoco/model_transport.xml
--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized --esdf-margin 0
--no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --obstacle-margin 0.010 --held-body-cover
--closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3
--finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20 --to-fast-eval
--esdf-host-mirror --target-exclusion ball-owned --owned-max-dist-voxels 3 --attention-split-radius-mm 12 --attention-split-ratio-source raw
--target-permission-strict --attention-source noun_target:L10H7 --attention-split-partition --split-supersedes-anchor --wrist-track-scale 0.1
--neighbour-contact-mm 20 --placed-no-seed-certify --kept-geometry-follow --roi-carve --roi-carve-footprint --roi-carve-footprint-pad-px 0
--roi-carve-footprint-robot ignore --roi-carve-device cuda --blas-threads 1 --palm-shell on --palm-shell-margin 5 --neighbour-palm-contact-mm 10
--deflection-rate-after-grasp --continuity-after-grasp --target-extent attention-surface --surface-target-check overlap --surface-keep-last
--surface-grasp-hold --surface-no-ball --attention-surface-device cuda --surface-hull-fill --surface-contact-allowance-mm 10
```
(3 단계 gate 는 명령줄에 없지만 서버 기본 on.) 출처: `results/development/20261103T002419+0900_8ab3c7ef7757_82e2790bab11/verify/T88/CL/p0/k0/server_p0_*/server_cmdline.txt`.

## 3. benchmark checkout 의 지금 구성 (`AG3S/adapters/panda/profile.py` `FINAL_CONFIG_FLAGS`) 과 다른 점

| 항목 | benchmark `FINAL_CONFIG_FLAGS` (Panda 1 판, 2026-10-29 기준) | P0 |
|---|---|---|
| target 범위 | 공 (ball-owned) + `--nontarget-evidence surface` | **표면 추출** (`--target-extent attention-surface` + A2–A5 + hull 채우기 + 접촉 여유 10 mm), `--nontarget-evidence` 없음 |
| 단계 규칙 | `--hsr6-attach-assist --hsr6-held-gate --hsr6-target-exempt --hsr6-objective --hsr6-objective-scope nbc` (Panda HSR6 probe) | 없음 |
| TrajOpt 초기값 | `--to-multi-seed 16 --to-gpu-seeds curobo --to-gpu-seeds-check first-pass` | E0 (없음) |
| attention 출처 | `noun_target:L8H6` (2 판부터 기본 `action:last:L8H2`) | `noun_target:L10H7` (RB-Y1 값, Panda 는 다시 골라야 함) |
| G1 · A4 | 1 판에서 뺌 (G1A4_FLAGS 별도) | 끔 (같음) |
| 쥔 뒤 연속성 · 꺾임 변화율 | 켬 (1 판), 3 판은 모든 단계 끔 | 쥔 뒤에만 켬 |
| 나머지 (손가락 · 손바닥 이웃 허용, 손목 약화, ROI carve, 손바닥 껍질 등) | 같음 | 같음 |

benchmark 의 2 판 · 3 판 · E0 · pc 판 (SL6-C · E · H) 에 들어간 Panda 전용 보정 (pb1 · pb6 · pb7, placement 접촉 허용 등) 은 P0 에 없다. 그 판들의 근거는 benchmark 세션이 판단한다.

## 4. 코드 이식

- benchmark `serve_safe.py` 에는 `--target-extent` 만 있고 `--surface-target-check · --surface-keep-last · --surface-grasp-hold · --surface-no-ball · --attention-surface-device · --surface-hull-fill · --surface-contact-allowance-mm` 가 **없다.**
- 이 코드 (T78 · T78b · T78f, 그리고 미채택이지만 같은 파일에 있는 T78d · T78e · T85 · T88 의 기본 꺼짐 블록) 는 **development 작업 트리에 commit 되지 않은 상태** 다. 옮기려면 먼저 development 에 commit 해야 한다 (사용자 승인 필요).

## 5. 미결 (사용자 결정 대기)

1. ~~3 단계 gate~~ **해결 (2026-11-03 18:2x KST 사용자 판정):** latch 로 대체 (`--subtask-from-latch`, P0m). Panda 에도 이 길뿐이다 (`kv_L4_v1` 없음).
2. **손바닥 구 모델:** 서버 시작 로그가 "손바닥 몸체 구 6 mm, 덮으려면 33 mm" 라고 경고한다. T88 P1 의 금지 접촉 2 건이 이것 때문이었다. Panda 손에서도 같은 문제가 있는지 benchmark 쪽에서 확인 필요.

## 6. RB-Y1 에서 정한 값 — Panda 에서 다시 정해야 할 것 (규칙 목록 `AG3S_RULES_REGISTRY.md` 의 "새 환경 실패 위험")

- attention head (L10H7 · split 12 mm) · 3 단계 · 5 단계 분류기 (모두 체크포인트 · 시뮬레이터에 묶임).
- latch 문턱: 닫힘 0.85 · reach 0.12 m · 막힘 0.05 · 빈손 0.10 (RB-Y1 그리퍼 stroke 기준).
- 거리 창: 손바닥–target 60 · 80 · 150 mm, 손가락 행 120 mm (그리퍼 크기 기준).
- 접촉 허용: 손가락 20 mm · 손바닥 10 mm · target 접촉 여유 10 mm · hull pad 5 mm.
- 구 모델: 손가락 덮개 · 손바닥 껍질 (RB-Y1 손 mesh 기준).
- 평가 정의: MuJoCo 데모는 "쥐는 순간 쥐는 손–이웃 과일 접촉은 실패로 안 셈" (창 = 정답 쥠 전 ∧ 손바닥–사과 ≤ 60 mm). **이웃이 사과에 "붙은 (10 mm 안)" 조건은 D2 로 뺐다** (이웃–사과 거리와 상관없이 면제). 기둥 · 허들 접촉은 금지. **벤치마크 profile (사용자 확인 2026-11-03 18:2x KST, D1): `--neighbour-contact-mm` · `--neighbour-palm-contact-mm` 와 평가의 이웃 과일 면제를 모두 끄고, 접촉 허용은 target (쥘 때) · 목적지 (놓을 때) 만.** MuJoCo 데모 profile 과 다른 점으로 명시한다. 이 두 허용이 RB-Y1 쥐기를 7 → 9 / 10 으로 올렸으므로, LIBERO (장애물 없음) pilot 에서 끈 비용을 먼저 잰다.
