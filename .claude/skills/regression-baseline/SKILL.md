---
name: regression-baseline
description: AG3S 16D 회귀 기준선을 돌리고 판정한다. 코드를 고치기 전/후에 "이 숫자가 그대로인가" 를 확인할 때, 측정을 시작하기 전에 환경이 성한지 볼 때 쓴다.
---

# 회귀 기준선 (16D)

**이 숫자가 다르면 코드를 고치기 전에 원인부터 찾는다.**

```bash
MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -m \
  AG3S.trajopt.experiments.esdf_rollout --records run_16d_ep1800 \
  --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz \
  --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all \
  --support-surfaces field --constraint-links arms --out-json /tmp/base.json
```

## 판정 (2026-09-25 갱신 — 재현되는 것만 기준으로 삼는다)

**`feasible`/`violated`/`해소` 는 기준이 아니다.** `sqp.time_budget_ms` 가 벽시계 마감이라
같은 명령·같은 코드에서 `12/3/11` ↔ `14/1/13` 으로 흔들린다. 옛 기준값(`13/2/13`)이 그 폭
안에 들어 있어 기준 구실을 못 했다. 아래 셋은 **SQP 이전 값**이라 소수점까지 재현된다.

**왜 15/15 에서 14/15 로 내려갔나 — 결함이 제거된 것이지 기준선이 흔들린 것이 아니다.**
자기 필터가 로봇 밑동을 놓쳐 `base` 픽셀 5,758 개가 ESDF 에 **76~99 % 점유**로 들어가 있었다.
frame 0 에서 팔이 그 **유령 장애물**과 29 mm 겹쳤다고 보고했고, 그것이 옛 "시작 위반 15/15"
중 하나였다. `self_filter_inflation` 0.02 → 0.05 m 와 `UNCOVERED_LINKS` 보완(자기 필터 구
194 → 218) 뒤 `frame0 clearance_before` 가 **−29.031 → +0.157 mm** 로 올라왔다.


```
[legacy]  위반으로 시작 14/15 · has_target 15/15 · frame0 clearance_before −29.031048280806342 mm
[cuRobo]  위반으로 시작 10/15 · has_target 15/15 · frame0 clearance_before −9.171877401271193 mm
```

**2026-09-30 갱신 — 기준선이 둘이다** (사용자 판정 B, 근거 `ag3s/docs/handoff/T36.baseline.verify.json`).

- **legacy** = 위 명령 그대로 (`esdf_rollout` 기본 `--esdf-backend legacy`, numpy ESDF). 옛 값 13/15 · +0.185978 은
  self-filter inflation 0.05 가 **왼손 옆 실제 crate 점**(frame0 wrist_l 2,787 · wrist_r 569 px)을 지운 장면의 값이었다.
  T26 에서 inflation 0 이 되자 crate 가 남아 `link_left_arm_5` 구(r 81.2 mm) ↔ crate 가 +21.2 − margin 50 ≈ −29.03 mm.
  로봇 누수는 0 / 836,762 px (T30b · T32 H4). **옛 값으로 되돌리려고 inflation 을 올리지 마라** — 실제 물체를 지운다
  (T14 에서 사과를 지운 것과 같은 기전). T31diag · T32 H4 · T33 · T34 · T35 · T36 에서 소수점까지 재현.
- **cuRobo** = 위 명령 + `--esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005`, **`.venv-openpi-live`** 로
  (ag3s 쪽 python 에서는 curobo import 불가). 2026-09-30 단독 4 회 비트 단위 재현 (15 프레임 clearance_before sha1 동일,
  필드 tier 배열 sha1 도 15/15 동일). 서버와 같은 거리장 경로(분리된 fine TSDF · fine truncation 0.030 m)를 탄다.
- **cuRobo 기준선이 덮지 않는 것** (esdf_rollout 에 flag 가 없다): 제약 범위 gripper (rollout 은 arms) ·
  `--target-field-policy exclude-authorized` (target-free 층 0 개) · `--esdf-margin 0` (0.05 로 잰다) ·
  `--no-self-collision` · 구 옵션 · 쥔 물체(attached 0 · held freeing 0). **이 경로들은 폐루프 E3 실행으로만 검증된다.**
- 둘 중 하나라도 바뀌면 코드를 의심하기 전에 무엇이 같이 돌았는지부터 본다 (아래 "단독으로" 절).

**2026-09-28 갱신 — 14/15 · +0.157 은 `plan_horizon = 32` 시절 값이다.** `5c5291e` (T6f, TO 를
실행 창 8 스텝에만 걸기) 뒤의 HEAD 는 **13/15 · +0.18597810857728936 mm** 이고, T7a 가 그렇게
기록했다. T19 · T24 검증(2026-09-28)이 각각 단독으로 다시 재어 HEAD(`368d723`)와 소수점까지
같음을 확인했고, `git archive HEAD` 사본도 같은 값을 냈다 — 미커밋 변경(T18~T24)은 기준선을
바꾸지 않는다. **아래 문단들의 14/15 · +0.157 은 그 시절의 근거 설명이므로 그대로 둔다.**

**단독으로 돌린다. 다른 실행과 겹치면 숫자가 바뀐다.** `sqp.time_budget_ms` 는 벽시계
마감이라(`sqp.py:159`) 부하가 걸리면 같은 50 ms 안에 반복을 덜 돌고 더 나쁜 iterate 를
돌려준다. 2026-09-25 에 pytest 전체 · 다른 rollout 과 같이 돌렸더니 **`feasible 9 / violated 6`**
이 나왔다 — 뒤집힌 프레임 5·9·10 은 반복이 2 → 1 로 줄었고 clearance 가 +4.35 → −6.77 mm 까지
갔다. 단독으로 다시 돌리니 위 숫자가 한 자리도 안 틀리고 재현됐다.
**개수가 다르면 코드를 의심하기 전에 무엇이 같이 돌고 있었는지부터 본다.**

`feasible` 은 **clearance 만 세는 숫자가 아니다** (X3, 2026-09-25). `status` 는 clearance 와
**기하 인증**(`geometry_certified` — AG3S 가 그 프레임 씬을 빠짐없이 설명했다고 보증하는가)을
함께 본다. `sqp.py:287-293` 은 모든 제약을 통과한 궤적도 인증이 없으면 `VIOLATED` 로 내린다.

- **개수만 인용한다.** 프레임별 clearance 와 SQP 반복 수는 기준선이 아니다.
  `sqp.time_budget_ms` 는 벽시계 **마감**이라(`AG3S/trajopt/sqp.py:159`) 기계 부하에 따라
  반복이 하나씩 달라진다. 4 회 실행에서 위 개수는 전부 같았고, 프레임 13·14 의
  `clearance_after` 가 4.19 ↔ 4.97 mm 로 흔들렸다.
- **`--cameras all` 이 필수다.** head 한 대만 쓰면 에피소드 기반 씬에서 시작 위반이 0/15 라
  TO 가 고칠 것이 없어 기준선 구실을 못 한다. 세 대는 서빙 경로와 같은 진입점이기도 하다.

## 범위가 넓어졌다 (2026-09-25) — 이제 grounding 도 잰다

옛 주석은 *"이 기준선은 attention → target grounding 을 재지 않는다 (`has_target` 15/15 false).
차이는 오프라인이 고정된 attention 셀 선택을 쓰는 것"* 이었다. **그 원인 진단이 틀렸다.**

셀이 아니라 **배선**이었다. 오프라인은 attention 맵을 `zed_left` 한 대에만 붙이고 live 는
세 대에 각각 붙였다. 같은 셀·같은 기록에서 배선만 live 와 맞추니 **`has_target` 이
0/15 → 9/15** 로 섰고 **개수 넷은 그대로**였다 (회귀 없음, 2026-09-25 실측).

**2026-09-26 갱신 — `has_target` 은 9/15 가 아니라 `15/15` 다** (사용자 판정). `T5e` 가
`clustering.target_score_threshold` 기본값을 `0.25 → 0.0` 으로 내리고 target 을 **rank 1 +
`target_confirm_frames = 3`** 으로 고르게 바꿨다. 9/15 는 그 threshold 가 만들던 값이었다.
**나머지 둘(`위반으로 시작 14/15` · frame0 `clearance_before +0.15718632962849477 mm`)은
소수점까지 그대로다** — `T5f` 실측.

그래서 지금 기준선이 재는 범위는 `depth → 자기 필터 → attention 배선 → grounding → 필드
→ SQP → QP` 다. **`has_target` 15/15 도 함께 본다** — 거기서 떨어지면 배선이나 grounding 쪽이다.

전수 근거: 1296 조합(layer 18 × head 8 × denoise 3 × agg 3) 어느 것도 `cam_high` 에서
사과를 못 맞혔다. 셀 축에 답이 없다는 뜻이고, 답은 카메라 축에 있었다.

## 은퇴한 기준선

**14D(`run_0004` · 해소 14 / feasible 8, violated 7)는 2026-09-24 에 은퇴했다.**
`wire.ACTION_WIDTH` 가 16 이 되어 14 열 기록으로는 아예 돌지 않는다
(*"chunk has 14 columns but the layout needs at least 16"*).
`run_0004` 와 `attention_step1_run0004.npz` 는 14D 기록을 읽는 과거 실험 스크립트용으로 남긴다.

## 용어

- **violated** — 그 프레임의 궤적이 여유거리(clearance) 제약을 하나라도 어긴 상태.
- **feasible** — 최적화가 끝난 뒤 모든 제약을 만족하는 상태.
- **해소** — violated 로 시작해 feasible 로 끝난 프레임 수. **개선** — 어긋난 양이 줄어든 프레임 수.
