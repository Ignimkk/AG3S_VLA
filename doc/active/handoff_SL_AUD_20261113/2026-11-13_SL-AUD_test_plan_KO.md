# 시험 계획: action QP 와 놓는 구간의 수정 (SL-AUD) 과 최종 동결까지의 순서 (2026-11-13 밤)

받는 쪽: SDM 서버에서 `feature/panda-benchmark` 를 돌리는 세션 (sdm-f6). 작성: 로컬 PC 세션. 이 문서와 같이 인용하는 문서 (점검 문서, 전달 문서) 는 같은 폴더에 있다. 이 PC 에는 GPU 가 없어 closed loop 은 돌리지 못했다. code 와 unit test, offline 계산까지만 했다.

**용어.** 보정 크기 = action QP 가 action 한 성분을 바꾼 양의 최댓값 (action 범위는 ±1). row = 로봇 충돌 구 하나와 최근접 표면 사이의 거리 제약. 장애물 물체 row = 그 표면이 scene 장애물 집합에 속하는 row. 접촉 row = 그 밖의 row (target · 받침면 · 목적지 · 쥔 물체의 상). 잃은 ep = 판은 task 실패이고 같은 episode 의 π0.5 단독은 task 성공인 것.

---

## 1. 요청

1. branch `audit/sl-aud` 의 commit 을 가져와 unit test 를 돌려 주세요 (§3, CPU, 지금 가능).
2. **Tab. 1 AG3S arm 을 시작하기 전에** 거르기 두 개 (각 GPU 1 시간 20 분) 를 돌려 주세요: action QP 후보 (§5) 와 놓는 구간 후보 (§6).
3. 통과한 것이 있으면 gate 가 고른 판 위에 얹어 채택 gate 를 한 번 돌리고, 그 결과로 최종 동결을 정합니다 (§7).
4. 결과는 §8 의 형식으로 돌려 주세요.
5. **SDM 계획표를 고쳐 주세요.** 'gate 종료 후' 와 'Tab. 1 AG3S arm' 사이에 거르기 1 · 거르기 2 · 채택 gate 를 넣습니다 (§7 의 표 그대로). Tab. 1 AG3S arm 과 GTO arm 의 시작 시각은 그만큼 뒤로 갑니다. 계획서 (`AG3S_PLAN_SCHEDULE_20261110_KO.md`) 에는 §5 · §6.3 의 셋 · 판 · 기준을 결과 전에 적어 주세요.
6. commit 을 넣은 뒤 **동결 판이 그대로인지** 한 번 확인해 주세요 (§3 의 끝, GPU 약 5 분).

Tab. 1 AG3S arm 의 시작이 밀리는 시간은 약 2 시간 50 분 (두 거르기가 모두 닫힘) 에서 약 6 시간 30 분 (채택 gate 까지) 입니다 (§7).

---

## 2. 무엇을 고쳤고 왜 고쳤나

동결 판 `pkgFrs` 의 flag 가 켜는 code 를 점검했다 (`2026-11-13_frozen_config_code_audit_KO.md`). action QP (`sqp.py` `_solve_ee_action`) 에서 의도와 다른 곳 세 가지가 기록된 큰 보정과 맞았다. 근거는 init 19 의 raw 3 episode (209 call) 와 실제 Panda action map 으로 돌린 offline 계산이다.

| 문제 | 근거 | 고친 것 (flag, 모두 기본 꺼짐) | 판 segment |
|---|---|---|---|
| QP 가 chunk 의 step 0 부터 모든 row 를 지키려 한다. action 1 단위가 step 0 에서 움직이는 거리는 3.5 mm 라 5 mm 를 띄우려면 보정 1.42 가 필요하다. 풀리지 않으면 slack 벌점이 관계없는 축까지 ±1 로 보낸다 | 가장 나쁜 row 가 step 0–4 인 call 14 개 중 13 개가 보정 1 이상, step 5–9 는 58 개 중 11 개 | `--ee-action-qp-min-reach-mm X`: 고칠 수 있는 action 성분에 대한 gradient 가 X mm (action 1 단위당) 보다 작은 row 를 QP 에서 뺀다. 최종 검사는 그 row 도 본다 | `-mr8` · `-mr12` |
| 통과 판정은 접촉 row 를 −100 mm 까지 받아들이는데 QP 는 같은 row 를 +5 mm 로 올린다 | 여유 0.9 mm 부족 → 보정 1.04, 3.3 mm → 1.69. 계획서 §9.8 (11-11) 의 관측과 같다 | `--ee-action-qp-reference-floor`: 접촉 row 를 `min(5 mm, policy chunk 에서의 clearance)` 에 둔다. policy chunk 보다 나빠지지 못하고, 고치지도 않는다 | `-rf` |
| 관절 한계 하나로만 거절된 chunk 도 action QP 로 간다. 이 QP 에는 관절 한계 제약이 없다 | 장애물 위반 0 인 call 2 개에서 보정 0.90 · 1.61 | `--ee-action-qp-limit-passthrough`: 사유가 `limit` 이고 다음이 action QP 면 그대로 통과 | `-lp` |
| 재선형화의 조기 종료가 `full_violation` (양수 = 여유) 의 부호를 뒤집지 않고 비교한다. `_obstacle_margin_now` 가 이전 call · episode 의 값으로 남을 수 있다 | code | `--ee-action-qp-audit-fixes`: 둘을 바로잡는다 | `-af` |
| (전달 문서의 후보 B) 보정 크기 상한 | — | `--ee-action-qp-max-delta X`: 한 성분의 보정을 X 로 자른다 | `-md5` |

- **`-rf` 는 `-gr` (SL-SOTA4) 과 다르다.** `-gr` 은 접촉 row 를 −100 mm 까지 풀었고 11-11 에 탈락했다 (task 0: 5 · 6 · 4 대 9 · 8 · 7). offline 계산에서 `-gr` 은 손을 탁자 쪽으로 끝까지 민다 (행 0 의 z 보정 −1). `-rf` 는 풀지 않는다. 두 flag 는 같이 켤 수 없다.
- **`-rf` 와 `-mr8` 은 같이 있어야 한다.** 하나만 켜면 다른 쪽 원인으로 여전히 포화된다 (아래 표).
- **`-md5` 는 단독으로는 효과가 없을 것으로 본다.** 상한이 포화를 다음 행으로 옮길 뿐 방향은 그대로다 (아래 표).

**offline 계산** (`scripts_20261113/audit_qp_demo.py`). 합성 장면: 손이 벽 (장애물 물체 row) 과 나란히 움직이고 실제 여유가 14 mm (15 mm 에서 1 mm 부족), 손 아래 탁자는 3 mm. action map 은 실제 `ee_action_map`, 설정은 동결 판.

| 판 | 보정 크기 | ±1 (또는 상한) 에 붙은 성분 | 남은 slack (mm) | 5 step 뒤 EE 이동 (mm) | 행 0 의 보정 (x · y · z) |
|---|---:|---:|---:|---:|---|
| 동결 판 | 1.00 | 3 | 2.5 | 14.7 | −1.00 · +0.40 · +1.00 |
| `-gr` | 1.00 | 3 | 2.5 | 14.9 | −1.00 · +0.40 · −1.00 |
| `-rf` | 1.00 | 2 | 2.5 | 10.8 | −1.00 · +0.40 · 0.00 |
| `-mr8` | 0.42 | 0 | 0.0 | 9.3 | −0.42 · 0.00 · +0.15 |
| `-md5` | 0.50 | 2 | 4.2 | 15.5 | −0.50 · +0.40 · +0.50 |
| **`-rf` + `-mr8`** | 0.43 | 0 | 0.0 | 8.8 | −0.43 · 0.00 · 0.00 |

벽의 법선은 x 다. 동결 판은 y · z 까지 끝으로 보낸다. `-rf` + `-mr8` 에서 x 하나만 남는다. 손 아래에 30 mm 의 받아들여진 model 위반이 있는 경우에는 동결 판이 9 성분 포화 · EE 이동 37.9 mm, `-rf` + `-mr8` 은 위와 같다 (0.43 · 8.8 mm).

합성 장면이므로 크기는 closed loop 에서만 나온다.

---

## 3. 보낸 것 과 넣는 방법

branch `audit/sl-aud` 로 보냈다 (origin). `origin/feature/panda-benchmark` 의 `e5f0254d` 위에 code commit 둘과 문서 commit 이다.

| commit | 내용 |
|---|---|
| `282e22d9` | code + 동작 test. action QP: `AG3S/trajopt/sqp.py` · `config.py` · `serve_safe.py`. 놓는 구간: `AG3S/ag3s/runtime/pipeline.py` · `stages/target_grounding.py` · `stages/admissibility.py` · `ag3s/config.py`, `AG3S/trajopt/grasp_latch.py` · `safe_policy.py`. 공통: `AG3S/adapters/panda/server.py` (`PORTED_FINAL_CONFIG_DESTS`). 새 test `tests/trajopt/test_sl_aud_action_qp.py` · `tests/ag3s/test_sl_aud_place_phase.py` |
| `ac62c2bc` | 판 등록. `AG3S/adapters/panda/profile.py` (`SOTA_AUDIT_VERSIONS` 26 개), 새 test `tests/benchmark/test_sl_aud_versions.py`, 판 목록을 고정한 기존 test 세 줄 |
| 그 뒤의 commit | 문서만: `doc/active/handoff_SL_AUD_20261113/` (이 계획서, 점검 문서, 전달 문서, `scripts_20261113/audit_qp_demo.py`) |

`e5f0254d` 까지의 변경 (`-gh156a-sw`, `-a12h3`) 과는 이미 합쳐 두었다. SDM 에 아직 push 하지 않은 commit (FWDATT merge `656a98f6`, `-fwd` 판 등록) 과만 맞추면 된다.

```bash
# queue 경계에서 (queue 중 working tree 를 바꾸지 않는다). 먼저 worktree 에서 해 보려면 git worktree add ../aud-check HEAD
cd /mnt/dev/AG3S_VLA
git fetch origin audit/sl-aud:refs/remotes/origin/audit/sl-aud   # single-branch clone 에서도 origin/audit/sl-aud 가 생긴다
git cherry-pick 282e22d9            # code: 다른 block 과 겹치지 않는다
git cherry-pick ac62c2bc            # 판 등록: 겹치면 아래대로
git checkout origin/audit/sl-aud -- doc/active/handoff_SL_AUD_20261113   # 문서 폴더만 가져온다 (원하지 않으면 생략)
source scripts/setup/env.sh
.venv-ag3s/bin/python -m pytest -q tests/trajopt/test_sl_aud_action_qp.py tests/ag3s/test_sl_aud_place_phase.py \
    tests/benchmark/test_sl_aud_versions.py tests/ag3s/test_sl_pe_wiring.py tests/ag3s/test_admissibility.py \
    tests/ag3s/test_t47b_b3_sink_cli.py tests/benchmark/test_sl14_opening_rule.py \
    tests/trajopt/test_sl_sota1_obstacle_row_margin.py tests/trajopt/test_sl_gtto9_ee_action_qp.py \
    tests/trajopt/test_t88_passthrough_when_clear.py tests/trajopt/test_sl_sota11_opening_rule_reach_sticky.py \
    tests/benchmark/test_sota_clearance_versions.py tests/benchmark/test_sl_sota18_hand_sized.py \
    tests/trajopt/test_sl_gtto11_scaled_qp.py
# 그 뒤 판 목록을 세는 test 전체 (이 PC 에서는 일부가 fcntl 때문에 돌지 않았다)
.venv-ag3s/bin/python -m pytest -q tests/benchmark tests/trajopt -k "version or flag"
```

**`ac62c2bc` 가 충돌할 때.** `-fwd` 판을 등록하면서 같은 줄을 고쳤다면 아래 두 곳에서 겹친다. 둘 다 '양쪽 것을 모두 남기는' 것이 답이다.

- `profile.py`: `SOTA_AUDIT_VERSIONS` block 은 `GT15_VERSIONS = GT15_VERSIONS + SOTA_VERSIONS` 줄 바로 뒤에 통째로 들어간다. 다른 block 에 기대지 않는다. SDM 쪽 block 과 순서만 정하면 된다.
- 판 목록을 고정한 test 세 줄: `test_sl_sota11_opening_rule_reach_sticky.py` 의 `-rs-` 허용 목록, `test_sl_gtto11_scaled_qp.py` 의 `--ee-action-qp-scaled` 허용 목록에는 `+ pf.SOTA_AUDIT_VERSIONS` 를 더한다. `test_sl_sota18_hand_sized.py` 의 `--graspable-hand-sized-mm` 판 목록에는 `and v not in pf.SOTA_AUDIT_VERSIONS` 조건을 더한다.

**판 등록에서 SDM 이 더할 것.**

- base 는 `(FROZEN_20261112, <frozen>-gh156a-tro)` 이고 segment 는 13 개다 (판 26 개). **gate 가 고른 판의 `-fwd` 짝을 base 에 더해 주세요.** 그 이름이 이 clone 에는 없다. `-fwd` 판의 flag 를 가진 판 목록을 고정한 test 가 있으면 거기에도 `SOTA_AUDIT_VERSIONS` 를 더해야 한다.
- `SOTA_VERSIONS` 에는 넣지 않았다 (`test_sota_clearance_versions.py` 의 개수 고정을 건드리지 않으려고). `GT15_VERSIONS` 에만 넣었다.

**이 PC 에서 돌린 것.** Python 3.10, numpy 2.2.6 · scipy 1.15.3 · osqp 1.1.3 · casadi 3.8.2 · mujoco 3.17.0 (Windows).

| 묶음 | 결과 |
|---|---|
| 새 test 세 파일 | 30 통과 (action QP 17, 놓는 구간 12, 판 등록 1) |
| branch 에서 관련 test 21 파일 (SDM 의 새 test 포함) | 255 통과, 2 실패 (Windows 에 `fcntl` 이 없어서. 변경 전에도 같다) |
| 더 넓은 범위 | 변경으로 새로 실패하는 것 0 (§9) |

새 test 는 flag 가 꺼졌을 때 결과 · row · 기록 key 가 그대로인 것, 각 flag 의 동작, 실제 optimizer 에서의 `limit` 통과, 실제 linearizer 의 ESDF block 과 floor 의 shape 이 맞는 것, 동결 판의 config 가 그대로인 것을 고정한다. 놓는 구간 쪽은 장애물 집합 · 목적지 등록 · latch 를 각각 실제 함수로 돌린다 (`AG3S._scene_obstacle_points`, `TargetConfirm` + `ground_target`, `GraspLatch.observe_grasp`).

**동결 판이 그대로인지 확인 (commit 을 넣은 뒤, 거르기 전에).** flag 는 모두 기본 꺼짐이고 unit test 는 꺼졌을 때의 결과 · 기록 key 가 같다는 것을 고정한다. 다만 closed loop 에서 bit 단위로 같은지는 이 PC 에서 볼 수 없었다. 꺼진 상태에서도 지나가는 code 가 조금 바뀌었기 때문에 한 번 확인한다.

| 꺼진 상태에서도 바뀐 것 | 왜 결과가 같아야 하나 |
|---|---|
| `sqp.py` 의 조기 종료 조건을 두 줄로 나눔 | 같은 식이다. `full_violation` 은 앞의 두 조건이 참일 때만 불린다 (전과 같다) |
| `_finish_passthrough` 에 `note` 인자 | 기본값이면 전과 같은 note |
| `grasp_latch.py` 의 `_rule_released` 표시 | 값을 쓰기만 한다. 읽는 곳은 flag 가 켜졌을 때뿐이다 |
| `DestinationRegistry.observe` 의 `leader` 인자 | 기본값이면 index 0 (전과 같다) |
| `_scene_obstacle_points` 의 합집합 부분 | flag 가 꺼지면 앞 두 요청을 그대로 합친다 |

확인 방법: 기존 동결 판 run 하나의 **첫 episode** 를 같은 run 구성으로 다시 돌려 (예: Spatial L-I init 15–18 run 의 task 0 init 15) `*_calls.jsonl` 의 call 수, call 마다의 `max_abs_action_change_executed` · `to.passthrough_clear.reason` 이 기존 기록과 같은지 본다. 다르면 거르기를 시작하지 말고 알려 주세요. (joint SQP 로 간 call 이 있으면 wall-clock 마감 때문에 그 call 은 다를 수 있다. 그 경우 그 call 을 적어 주세요.)

---

## 4. 받고 싶은 답 (가설)

| 가설 | 맞으면 | 틀리면 |
|---|---|---|
| H1. 잃은 ep 의 큰 보정은 QP 식이 만든 것이고, 그것이 task 를 깨뜨린다 | 거르기 1 에서 큰 보정 call 이 줄고 잃은 ep 이 돌아온다 (SSR 9 → 12 이상) | 큰 보정 call 은 줄었는데 SSR 이 그대로다. 큰 보정은 원인이 아니라 결과다 |
| H2. 과한 밀어내기가 CAR 을 덤으로 올리고 있었다 | 거르기 1 에서 > 50 mm 충돌이 는다 | 충돌이 늘지 않는다 |
| H3. 팔이 펴지는 task (Goal L-I task 0 등) 의 실패는 IK 가 수렴하지 않아서다 (점검 문서 §6.1) | 거르기 1 에서도 Goal L-I task 0 의 2 ep 은 돌아오지 않고, 그 ep 의 `ik_iters_max == 60` call 수가 그대로다 | 돌아온다 |
| H4. 쥔 구간의 거절은 목적지가 늦게 빠져서 생긴다 (등록 지연 + 합집합의 2 요청) | 거르기 2 에서 쥔 구간의 거절 call 이 줄고 Goal 의 잃은 ep 이 돌아온다 | 거절 call 은 줄었는데 SSR 이 그대로다, 또는 거절 call 이 줄지 않는다 (목적지가 아예 등록되지 않는 것이 원인) |

제 예상: H1 은 일부만 맞는다. 112 ep 로 치면 TSR +3 – 8 ep, CAR −0 – 5 ep 쯤이고, 거르기 1 의 기준을 넘길 확률은 절반에 못 미친다. 최근 후보 네 개 (`-gr` · `-zp5` · `-dc` · `-hf30`) 가 모두 0 에서 −1 이었다. 그래도 재는 이유는, 못 넘기면 H1 이 닫혀서 남은 시간을 IK 와 인식 쪽에 쓸 수 있기 때문이다. 거르기 2 는 근거가 더 얇다 (code 를 읽어 찾은 것이고 기록으로 크기를 재지 못했다). 그래서 시간이 모자라면 거르기 2 를 먼저 뺀다.

---

## 5. 거르기 1 — action QP 후보 (결과 전 고정)

**셋.** 전달 문서의 S32 를 그대로 쓴다. episode 결과가 run 안의 위치에 따라 달라지므로 (commit `f1e662e8`) 기준 run 과 같은 구성으로 돌린다.

| 이름 | 구성 | 동결 판의 값 (TSR · CAR · SSR, > 50 mm) |
|---|---|---|
| S32 | Spatial L-I task 0–3 × init 15–18 (16 ep) + Goal L-I task 0–3 × init 17 · 18 (8 ep) + Goal L-II task 0–3 × init 17 · 18 (8 ep) | 13 · 23 · 9, 3 |

S32 에는 잃은 ep 13 개와 안전 성공 9 개가 들어 있다. mug task 는 없으므로 base 는 동결 판 그대로 쓴다 (`-gh156a` 의 효과와 섞이지 않는다).

**판과 기준.**

| 순서 | 판 | GPU | 통과 기준 |
|---|---|---|---|
| 1-1 | `<frozen>-rf-lp-mr8-af-tro` | 1 시간 20 분 | SSR ≥ 12, > 50 mm 충돌 ≤ 4, 동결 판의 안전 성공 9 개 중 잃는 것 ≤ 1 |

**1-1 뒤의 갈림 (결과를 보고 바꾸지 않는다).**

| 1-1 의 결과 | 다음 |
|---|---|
| 통과 | §7 의 채택 gate 에 넣는다. `-rf` · `-mr8` 을 따로 재는 것은 Tab. 1 뒤로 미룬다 |
| SSR 은 12 이상인데 > 50 mm 충돌이 5 이상 | 닫는다. `-mr12` 는 row 를 더 빼는 방향이라 재지 않는다 |
| SSR 이 11 이하이고 큰 보정 (1 이상) 이 5 call 이상인 ep 이 잃은 13 ep 중 7 개 이상 남음 | `<frozen>-md5-tro` 를 S32 로 한 번 (1 시간 20 분, 같은 기준). 통과하지 못하면 닫는다 |
| SSR 이 11 이하이고 큰 보정은 줄었음 | 닫는다. H1 이 틀린 것이다 |

'닫는다' = action QP 쪽 후보는 여기서 끝낸다. 거르기 2 는 이 결과와 관계없이 돌린다.

---

## 6. 거르기 2 — 놓는 구간 후보, 그리고 전달 문서 (final plan) 의 고친 안

### 6.1 전달 문서의 후보를 어떻게 고쳤나

`2026-11-13_SDM_handoff_final_plan_KO.md` 의 후보를 점검 결과에 맞춰 고친다.

| 후보 | 원래 안 | 고친 안 | 이유 |
|---|---|---|---|
| A `-gh156a` | M10 으로 확인 | **끝났다.** gate 1 단계 통과 (SSR 22 대 21), guard 진행 중. 더할 것 없음 | SDM 계획표 |
| B 보정 크기 상한 | 새 flag 를 구현해 S32 | **구현해서 보냈다** (`-md5`). 단독 실행은 거르기 1 이 SSR 에서 떨어지고 큰 보정이 남았을 때만 (§5) | offline 계산에서 상한은 포화를 다음 행으로 옮길 뿐이다 (EE 이동 15.5 대 14.7 mm). 원인을 고친 `-rf` + `-mr8` 이 먼저다 |
| C 놓는 지점의 목적지 등록 (open seed) | 새 rule 을 구현해 S32 | **새 rule 대신 같은 증상을 만드는 code 세 곳을 고쳤다** (§6.2, 구현해서 보냈다) | 점검에서 목적지 등록 지연의 원인이 code 에 있었다. rule 을 더하기 전에 있는 rule 이 의도대로 돌게 한다 |
| 6.1 잃은 29 ep 의 집계 | CPU | **항목을 더한다** (§8 의 표) | IK · `limit` · step 별 보정을 같이 세야 H1 · H3 이 갈린다 |
| 6.2 obstacle confidence 의 offline 실험 | CPU, 조건부로 후보 D | **그대로, 동결 뒤로 미룬다** | 최종 동결의 경로에 넣기에는 구현 · 거르기가 길다 |

### 6.2 고친 것 — 놓는 구간의 code 세 곳

| 문제 (점검 문서 §4) | 고친 것 (flag, 모두 기본 꺼짐) | 판 segment | code |
|---|---|---|---|
| 장애물 집합의 '최근 3 요청 합집합' 에서 앞 두 요청의 점은 지금의 제외 규칙으로 다시 거르지 않는다. 목적지가 등록돼도 2 요청 동안 접시가 장애물로 남는다 | `--scene-obstacle-history-refilter`: 이번 요청이 빼는 점 (조작 대상 · 그 받침 · 목적지 · 쥔 물체) 15 mm 안의 기억된 점도 뺀다 | `-ohr` | `pipeline.py` `_scene_obstacle_points` |
| 목적지 후보는 전체 점수 1 위 cluster 하나만 본다. 1 위가 손 60 mm 안의 쥔 물체 조각이면 연속 3 frame 조건이 0 으로 돌아간다 | `--destination-leader-outside-hand`: 손 60 mm 밖의 cluster 중 점수 1 위를 본다 | `-dlo` | `target_grounding.py`, `admissibility.py` `DestinationRegistry.observe` |
| opening rule 은 놓은 뒤 `LATCHED` 로 간다. target 이 옛 자리에 남고 진행 선호가 그쪽으로 끈다 (Goal L-I task 0 에서 놓은 뒤 24 call) | `--ee-action-qp-progress-release-off`: 놓은 뒤 다음 닫기 시도 전까지 진행 선호의 'target 쪽' 을 끈다 | `-pro` | `grasp_latch.py` (표시만), `safe_policy.py`, `sqp.py` `_progress_direction` |

- `-ohr` 은 지금 위치의 제외만 반영한다. 쥔 물체가 한 요청에 15 mm 넘게 움직이면 그 전 위치의 기억된 점은 최대 2 요청 남는다 (이 부분은 그대로다).
- `-ohr` 은 제외된 물체에 15 mm 안으로 붙은 실제 장애물의 기억된 점도 지운다. 그 장애물이 이번 요청에 보이면 집합에 그대로 있다. 한 frame 빠졌을 때 버텨 주는 것만 그 근처에서 사라진다.
- `-pro` 는 target 기하를 버리지 않는다. 놓은 뒤 옛 자리 100 mm 안에서 손 row 가 풀리는 것 (`--grasp-zone-release`) 은 그대로다. 영향이 커서 이번에는 넣지 않았다.
- 세 flag 모두 unit test 는 실제 함수로 돌렸다 (§3). closed loop 은 돌리지 않았다.

### 6.3 거르기 2 (결과 전 고정)

| 순서 | 판 | GPU | 통과 기준 (S32: 동결 판 13 · 23 · 9, > 50 mm 3) |
|---|---|---|---|
| 2-1 | `<frozen>-ohr-dlo-pro-tro` | 1 시간 20 분 | SSR ≥ 11, > 50 mm 충돌 ≤ 4, 동결 판의 안전 성공 9 개 중 잃는 것 ≤ 1 |

- 기준은 원래 후보 C 의 것이다. S32 의 Goal L-I · L-II 16 ep 은 '쥔 구간에서 목적지가 쥔 물체를 막는' episode 가 몰린 곳이다.
- 통과하면 §7 의 채택 gate 에 넣는다. 떨어지면 닫는다 (세 flag 를 따로 재지 않는다).
- 거르기 1 과 순서는 상관없다. 둘 다 동결 판 base 이고 서로의 결과에 기대지 않는다.

### 6.4 넣지 않는 것

- IK (점검 문서 §6.1). flag 밖이고 simulator 와 대조해야 한다. 이번에는 §8 의 집계로 크기만 잰다.
- 인식 쪽 나머지 (target cut 의 낡은 기하, 거리 가중 attention 과 partition, class test 의 층 불일치). 빈도를 모른다.
- 놓은 뒤 target 기하를 버리는 것. 위에 적은 대로다.

---

## 7. 일정과 최종 동결

T0 = guard 가 끝나는 시각 (계획표로는 약 00:00). X = gate 가 고른 판 (guard 를 통과하면 `-gh156a`, 아니면 동결 판).

| 시각 | 일 | GPU | 조건 |
|---|---|---|---|
| 지금 – T0 | commit 을 worktree 에 넣고 unit test (CPU). §8.4 의 집계 시작 (CPU, worker 2 이하) | 0 | queue 의 working tree 는 건드리지 않는다 |
| T0 | commit 을 본 checkout 에 넣는다 (queue 경계). 동결 판 1 ep 으로 '그대로' 확인 (§3) | 약 5 분 | 다르면 멈춤 |
| T0 – +1:20 | 거르기 1 (§5, `-rf-lp-mr8-af`) | 1 시간 20 분 | |
| +1:20 – +2:40 | 거르기 2 (§6.3, `-ohr-dlo-pro`) | 1 시간 20 분 | |
| (조건부) | `-md5` 의 S32 | 1 시간 20 분 | §5 의 셋째 줄일 때만 |
| 그 뒤 | **채택 gate**: X + (통과한 flag 전부) 를 한 판으로. `-gh156a` 에 쓴 것과 같은 단계 · 기준 (1 단계 E1a 40, 3 단계 guard; 계획서 §9.2) | 약 3 시간 30 분 | 통과한 후보가 하나라도 있을 때만 |
| 그 뒤 | Tab. 1 AG3S arm (그 판의 `-fwd` 짝) | 10 – 11 시간 | |

채택 gate 에 넣을 판은 등록해 두었다: 거르기 1 만 통과하면 `X-rf-lp-mr8-af`, 거르기 2 만 통과하면 `X-ohr-dlo-pro`, 둘 다면 `X-rf-lp-mr8-af-ohr-dlo-pro` (X 가 `-gh156a` 일 때와 동결 판일 때 모두).

**최종 동결의 정의.**

- 채택 gate 를 통과하면 X + (통과한 flag) 가 최종 동결 판이다.
- 채택 gate 에서 떨어지거나 통과한 후보가 없으면 X 가 최종 동결 판이다.
- 어느 쪽이든 Tab. 1 을 시작한 뒤에는 판을 바꾸지 않는다.

**Tab. 1 이 밀리는 시간.**

| 경우 | 밀리는 시간 |
|---|---|
| 두 거르기가 모두 닫힘 | 약 2 시간 50 분 |
| 하나 이상 통과 | 약 6 시간 30 분 |
| 거르기 1 뒤에 `-md5` 까지 돌림 | 위에 1 시간 20 분을 더함 |

채택 gate 는 한 번만 돈다. 시간이 모자라면 거르기 2 를 빼는 것이 먼저다 (거르기 1 이 기록과 더 직접 맞는다).

---

## 8. 돌려받고 싶은 것

### 8.1 GPU run 마다

| 항목 | 형식 |
|---|---|
| 판 이름, HEAD, run-id, 시작 · 끝 시각 | 한 줄 |
| TSR · CAR · SSR, > 50 mm 충돌 수 | cell 별 (Spatial L-I · Goal L-I · Goal L-II) 과 합 |
| episode 짝 | S32 의 32 ep 각각: 동결 판의 결과 → 이 판의 결과 (안전 성공 · 성공 + 충돌 · 실패 · 실패 + 충돌), 장애물 변위 mm |
| 기준 판정 | §5 또는 §6.3 의 세 조건 각각 통과 · 불통과 |

### 8.2 `calls.jsonl` 에서 센 값 (episode 마다, 동결 판의 같은 episode 와 나란히)

| 값 | key |
|---|---|
| action QP call 수, 보정 1 이상인 call 수 | `to.ee_action_qp.max_abs_delta_action` |
| ±1 에 붙은 성분의 합, slack 15 mm 초과 call 수 | `to.ee_action_qp.saturated_components`, `slack_max_m` |
| QP 에서 뺀 row 의 합 | `to.ee_action_qp.n_rows_unreachable` |
| 접촉 row 중 5 mm 아래였던 것의 합 | `to.ee_action_qp.reference_floor.n_rows_below_backoff` |
| 한계로 통과시킨 call 수 | `to.passthrough_clear.limit_passthrough` |
| 사유 `limit` 의 call 수 (동결 판) | `to.passthrough_clear.reason` |
| IK 가 한도에 걸린 call 수, 그때의 위치 오차 최대 | `conversion.ik_iters_max == 60`, `conversion.ik_pos_err_m_max` |
| OSQP 가 반복 한도에 걸린 call 수, QP 시간 중앙값 | `to.ee_action_qp.osqp_iter`, `to.solve_ms` |
| (거르기 2) 기억된 점에서 뺀 수의 합 | scene obstacle 기록의 `history_refilter.n_removed` |
| (거르기 2) 목적지가 등록된 call (attach 뒤 몇 번째 call 인가) | `chunk_summary.destination` 이 처음 생긴 call |
| (거르기 2) 진행 선호가 꺼진 call 수, 동결 판의 같은 구간에서 'target 쪽' 이었던 call 수 | `to.ee_action_qp.progress.mode` (`after the release` · `before the grasp: toward the target`) |
| (거르기 2) 쥔 구간에서 거절된 call 수 | `grasp.state == held` 이고 `to.passthrough_clear.applied == false` |

### 8.3 raw 기록 (다음 분석용)

거르기 1 · 2 의 run 각각에서 아래 6 ep 의 `*_calls.jsonl` · `chunk_summary.jsonl` · `*_geom.npz` 와, 동결 판의 같은 ep.

- Goal L-I task 0 × init 17 · 18 (팔이 펴지는 task, H3)
- Spatial L-I task 1 × init 15 · 16 (동결 판이 4 / 4 를 잃는 task)
- Goal L-II task 1 × init 17 · 18

### 8.4 CPU 집계 (GPU 와 무관, 지금 시작 가능)

동결 판의 144 ep (`ms-pkgFrssdm-*` 112 + `ms-tab10sdm-*` 32) 에 대해 §8.2 의 값 중 동결 판에 있는 것 (보정 크기 · 포화 · `limit` · IK · OSQP) 을 episode 마다 세고, 결과 (안전 성공 · 잃음 · 둘 다 실패 · 성공 + 충돌) 와 나란히 놓은 CSV. 전달 문서 §6.1 의 세 항목 (단계별 큰 보정, 막은 link 와 물체, 놓은 뒤의 close seed) 은 그대로 포함한다.

### 8.5 질문

- cherry-pick 이 깨끗이 들어갔는가. 손으로 고친 곳.
- unit test 의 출력 (통과 · 실패 수).
- 동결 판 1 ep 확인의 결과 (같음 · 다름, 다르면 어느 call).
- `-gh156a` 의 gate 1 단계에서 > 50 mm 충돌이 1 → 2 가 된 episode 는 어느 것인가.
- 계획표의 '동결 판 충돌 38 ep 의 원인' 네 줄. 받은 글에서 잘려 있었다.
- '충돌 검사에서 gripper 가 통째로 빠지는 경로' 를 보고 있다면, 점검에서 본 두 곳이 관련될 수 있다: `--grasp-zone-release` 는 `LATCHED` · `CLOSING` 동안 target 100 mm 안의 손 row 를 모두 푸는데, opening rule 은 놓은 뒤에도 `LATCHED` 이고 target 은 옛 자리에 남는다 (점검 문서 §4.1). 놓은 뒤 옛 자리 근처를 지날 때 손 row 가 풀린다.

---

## 9. 이 PC 의 test 결과

| 묶음 | 결과 |
|---|---|
| 새 test 세 파일 | 30 통과 |
| branch (`e5f0254d` 위) 에서 관련 test 21 파일 | 255 통과, 2 실패 |
| 판 목록을 세는 test 38 파일 + action QP · 통과 판정 test (460 개) | 423 통과, 34 skip, 실패 37 + 수집 오류 2 |
| `tests/ag3s` 전체 + latch · grasp · 놓기 관련 `tests/trajopt` · `tests/benchmark` (2,807 개) | 2,616 통과, 166 skip, 실패 25 + 수집 오류 2 |
| 위 실패를 변경 전 code 로 다시 돌림 | 두 묶음에서 각각 1 개만 변경 때문이었고 둘 다 고쳤다 (아래). 나머지는 변경 전에도 같다 (Windows: `fcntl` 없음, launcher · storage 경로) |
| `tests/trajopt` · `tests/benchmark` 전체 | 끝까지 돌리지 못했다 (이 PC 에서 1 시간 넘게 걸림). SDM 에서 돌려야 한다 |

- 고친 것 1: `test_sl_gtto11_scaled_qp.py::test_version_and_flag`. `--ee-action-qp-scaled` 를 가진 판의 허용 목록에 새 판 묶음이 없었다.
- 고친 것 2: `test_t47b_b3_sink_cli.py::test_audit_points_exist_in_code`. 이 test 는 `admissibility.py` 의 한 줄 (`if graspable[0] is False and …`) 을 글자 그대로 찾는다. 그 줄을 건드리지 않게 `DestinationRegistry.observe` 의 변경을 다시 썼다.
- **이 PC 에서 돌지 않는 test 안에도 판 목록을 도는 loop 가 있다** (`fcntl` 에서 먼저 멈춘다). 읽어서 확인한 바로는 모두 `GT15_VERSIONS` 로 허용하므로 새 판에 걸리지 않는다. SDM 의 실제 실행으로 확인해야 한다.
- closed loop 은 한 번도 돌리지 않았다.

---

## 10. 거르기 1 재실행 — `-rf` 를 고친 뒤 (2026-11-14 추가, 결과 전 고정)

**왜.** 거르기 1 (`aud1sdm`) 에서 `-rf` (SL-AUD1) 가 한 번도 적용되지 않았다. action-QP call 608 개가 모두 `reference_floor = {"why": "no ESDF row"}` 였다 (SDM 중간 보고 §A.4). 원인은 로컬 PC 쪽 code 의 잘못이다: ESDF block 은 `(H, n_query, 1)` 인데 2 차원으로 가정했다. commit `0f1d90b4` 로 고쳤다. 거르기 1 은 사실상 `-lp-mr8-af` 를 잰 것이므로 H1 은 아직 판정되지 않았다. 사용자 결정 (2026-11-14): 원래 계획한 판으로 다시 잰다.

**넣기.** `git fetch origin audit/sl-aud:refs/remotes/origin/audit/sl-aud` 뒤 `git cherry-pick 0f1d90b4`. 로컬 PC 가 더한 줄만 건드린다 (`sqp.py` 의 SL-AUD1 block, `tests/trajopt/test_sl_aud_action_qp.py`). unit test: `tests/trajopt/test_sl_aud_action_qp.py` (19 개. 실제 `EsdfField` 를 지나가는 것 둘 포함).

**먼저 확인 (GPU 약 5 분, 다르면 멈춤).** run 의 첫 episode 가 끝나면 그 `*_calls.jsonl` 에서 action-QP call 의 `to.ee_action_qp.reference_floor` 를 본다. `n_rows_below_backoff` 와 `min_floor_m` 이 있어야 한다. `why` · `error` · `shape_mismatch` 가 하나라도 있으면 run 을 멈추고 그 값 (`clearance_shape` · `centres_shape` 포함) 을 알려 주세요.

| 항목 | 내용 |
|---|---|
| 판 | `<frozen>-rf-lp-mr8-af-tro` (거르기 1 과 같은 이름. code 만 `0f1d90b4` 가 더해짐) |
| 셋 · run 구성 | S32, 거르기 1 과 같다 (§5). run-id 는 거르기 1 과 겹치지 않게 (예: TAG `aud1bsdm`) |
| GPU | 1 시간 20 분 |
| 통과 기준 | §5 그대로: SSR ≥ 12, > 50 mm 충돌 ≤ 4, 동결 판의 안전 성공 9 개 중 잃는 것 ≤ 1 |
| 통과하면 | §7 의 채택 gate 에 넣는다 |
| 떨어지면 | action QP 쪽 후보를 닫는다. `-rf` 를 따로 재지 않는다 |

**돌려받을 것.** §8.1 · §8.2 의 형식 그대로, 세 판을 나란히 (동결 판 → 거르기 1 → 재실행). 더할 값:

| 값 | key |
|---|---|
| `-rf` 가 적용된 action-QP call 수 / 전체 | `to.ee_action_qp.reference_floor` 에 `n_rows_below_backoff` 가 있는 call |
| 접촉 row 중 5 mm 아래였던 것의 합, call 당 중앙값 | `…reference_floor.n_rows_below_backoff` |
| 가장 깊은 floor 의 분포 (중앙값 · 최소) | `…reference_floor.min_floor_m` |
| 큰 보정 call 을 최악 row 의 step 0–4 · 5–9 로 나눈 수 | 거르기 1 의 표와 같은 방식 |

**읽는 법 (결과 전에 적어 둠).**

- 거르기 1 은 큰 보정을 318 → 167 로 줄이고도 SSR 이 9 → 10 이었다. 재실행에서 큰 보정이 더 줄었는데 SSR 이 그대로면 H1 (큰 보정이 실패의 원인) 은 틀린 것으로 닫는다.
- 112 ep 집계에서 잃은 ep 의 큰 보정은 최악 row 가 step 5–9 인 것이 247, step 0–4 가 77 이었다. `-mr8` 은 뒤쪽을 건드리지 않는다. `-rf` 가 step 5–9 쪽 큰 보정을 줄이는지가 이 재실행에서 볼 것이다.
- 거르기 1 에서 새로 생긴 충돌 둘 (Spatial L-I task 1 init 15 · 16) 중 init 16 은 통과 판정 (`clear`, 그 밖의 row 위반 29.7 mm) 에서 났다. `-rf` · `-mr8` 과 무관하므로 재실행에서도 날 수 있다.
