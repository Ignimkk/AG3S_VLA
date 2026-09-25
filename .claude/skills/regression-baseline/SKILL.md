---
name: regression-baseline
description: AG3S 16D 회귀 기준선을 돌리고 판정한다. 코드를 고치기 전/후에 "이 숫자가 그대로인가" 를 확인할 때, 측정을 시작하기 전에 환경이 성한지 볼 때 쓴다.
---

# 회귀 기준선 (16D)

**이 숫자가 다르면 코드를 고치기 전에 원인부터 찾는다.**

```bash
MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/work /mnt/dev/work/.venv-ag3s/bin/python -m \
  benchmark.trajopt.experiments.esdf_rollout --records run_16d_ep1800 \
  --attention benchmark/ag3s/asset/data/attention_16d_ep1800.npz \
  --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all \
  --support-surfaces field --constraint-links arms --out-json /tmp/base.json
```

## 판정

```
위반으로 시작 15/15 · 해소 13 · 개선 15 · feasible 13 · violated 2 · has_target 9/15
```

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
  `sqp.time_budget_ms` 는 벽시계 **마감**이라(`benchmark/trajopt/sqp.py:159`) 기계 부하에 따라
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

그래서 지금 기준선이 재는 범위는 `depth → 자기 필터 → attention 배선 → grounding → 필드
→ SQP → QP` 다. **`has_target` 9/15 도 함께 본다** — 거기서 떨어지면 배선이나 grounding 쪽이다.

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
