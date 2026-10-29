# AG3S-PRE-CORE-TO E1 — 구현 (목표 유지 TrajOpt: E1-C 비용 개선 · E1-B 목표 상자 제약)

> writer: ag3s-implementer · 읽는 쪽: lead, verifier, scribe
> 작성: 2026-10-25 21:30 KST (`date -u` 12:30 UTC + 9 h). 기준 코드: `feature/ag3s-development` HEAD `35b909cf` + 이 변경 (commit 안 함).
> 명세: [`AG3S-PRE-CORE-TO.task.md`](AG3S-PRE-CORE-TO.task.md) §0.1, §2 (IFG-TOA 줄), §6 E1.

## 무엇을 했나 (평이한 요약 먼저)

IFG-TOA 의 "잡을 지점 고정" 경로 위에 두 변형을 기본 꺼짐 flag 로 넣었다. **E1-C** 는 TOA 에서 손바닥 경로 추종 항만 뺀 것(목표 항만 남김, `--grasp-anchor W --grasp-anchor-path-weight 0`)이고, **E1-B** 는 손바닥 비용을 아예 두지 않고, 닫는 행부터 3 행의 손바닥 위치를 "목표 ± r mm" 상자 제약으로 QP 에 넣는 것(`--grasp-goal-box MM`)이다. 상자 제약의 slack 가격은 충돌 slack 의 1/10 이라 충돌 회피가 먼저다. 두 변형 모두 TOA 와 같은 gate · 같은 고정 목표 · 같은 안전 우선 대체 풀이(TOA3)를 쓴다.
기록된 closed-loop run 은 TOA 를 돌리지 않았으므로, offline 비교 도구(`tp_timing`)가 기록에서 **서버의 gate 함수 그대로** 목표를 다시 만들어 E0 · E1-C · E1-B 10 mm · E1-B 20 mm 를 같은 입력에서 푼다.
꺼져 있으면 궤적이 HEAD 와 bit 단위로 같다 (fixture 5 경우, 기록 replay 75/75 chunk).

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/trajopt/grasp_anchor.py:41` | `GOAL_BOX_SLACK_SCALE = 0.1` | E1-B slack 가격 = `w_slack × 0.1` per m (충돌 1 m = 상자 10 m) |
| `AG3S/trajopt/grasp_anchor.py:58–86` | `GraspAnchor.goal_box` (기본 0), `variant` (`"TOA"`/`"E1-C"`/`"E1-B"`), `goal_mask(H)` (가중치와 무관한 목표 행), `masks()` 의 목표 행 = `w_goal > 0` 또는 `goal_box > 0` | TOA 의 mask 는 그대로 (`w_goal > 0` 일 때 같은 행) |
| `AG3S/trajopt/grasp_anchor.py:166–224` | `goal_box_excess` (축별 상자 밖 초과량 = slack 의 정확한 값), `add_goal_box` (QP 에 행 3 블록 + 새 slack 열) | E1-B 의 QP 모델과 merit 항 |
| `AG3S/trajopt/grasp_anchor.py:227–` | `anchor_record` 에 `variant`, `goal_rows`, 축별 최대 목표 오차(`goal_err_axis_max_{reference,result}_mm`), E1-B 면 `goal_box_mm` · `goal_box_slack_sum_mm` · `goal_box_excess_max_mm` | 명세 "기록" 줄 |
| `AG3S/trajopt/problem.py:69–92` | `QpProblem.n_goal_slack` (기본 0), `n_var` 에 포함, `goal_slack(z)`, `kinematic_slack` 은 자기 구간만, `flatten` 이 0 을 채움 | 상자 slack 열을 맨 뒤에 두기 위해. 0 이면 전과 같은 배열 |
| `AG3S/trajopt/config.py:288, 315–320` | `CostConfig.grasp_goal_box` (m, 기본 0) + 검증 (≥ 0, TOA 가중치와 배타) | |
| `AG3S/trajopt/sqp.py:190–203` | `_objective`: 앵커 비용은 `cost_on` 일 때만, E1-B 면 `weight × excess.sum()` 를 merit 에 더함 | QP 와 같은 값을 step 수락이 보게 |
| `AG3S/trajopt/sqp.py:222–229, 315–332` | E1 변형인데 이 horizon 에 목표 행이 없으면 plain 풀이 + `{"applied": False, "reason": …}` 기록 (`_solve_plain_recorded`) | TOA 는 예전처럼 기록 없음 |
| `AG3S/trajopt/sqp.py:250–272` | E1 변형 기록에 `fallback_used` (bool) | 명세 "대체 풀이 여부". TOA 기록에는 안 붙임 |
| `AG3S/trajopt/sqp.py:281–313` | `_anchor_setup`: `goal_box > 0` 도 켜짐, 상태에 `cost_on` · `box` (행, 목표, r, 가격) | |
| `AG3S/trajopt/sqp.py:583–594` | QP 조립: `cost_on` 이면 TOA 항, `box` 면 `add_goal_box` (SQP 반복마다 선형화) | |
| `AG3S/trajopt/safe_policy.py:1789–1806, 1859–1861` | `_grasp_anchor_gate`: `cost.grasp_goal_box > 0` 도 gate 를 켜고 앵커에 `goal_box` 를 실음 | 같은 gate · 같은 목표 |
| `AG3S/trajopt/serve_safe.py:277–289` | `cost_overrides`: `--grasp-anchor-path-weight` → `w_grasp_tcp`, `--grasp-goal-box MM` → `grasp_goal_box` (m), 반경 flag 는 둘 중 하나와 | |
| `AG3S/trajopt/serve_safe.py:1724–1737` | 새 flag 두 개 (help 한국어) | |
| `AG3S/trajopt/serve_safe.py:2629–2646` | 검증: path-weight 는 `--grasp-anchor` 와 함께 · ≥ 0, goal-box > 0 · `--grasp-anchor` 와 배타 | |
| `AG3S/trajopt/serve_safe.py:3145–3156` | 시작 log 에 E1-C / E1-B 를 따로 알림 | |
| `AG3S/trajopt/experiments/e1_goal.py` (새 파일) | offline settings `E0` · `E1C` · `E1B10` · `E1B20` · `TOA`, `RecordedGraspGate` (서버 `_grasp_anchor_gate` 를 stub 위에서 그대로 호출), `rebuild_gates`, `goal_eval`, `summarize` | offline 비교 |
| `AG3S/trajopt/experiments/tp_timing.py:28–35, 307–321, 362–372, 392–396, 431–438, 472–474` | 켜진 setting 이 있으면 gate 를 기록에서 다시 만들어 `ctx["grasp_anchor"]`; 출력에 `grasp_goal` (chunk) · `grasp_anchor` · `goal_eval` (setting); `--replay-wrist-track` (기본 꺼짐) | 꺼진 setting 만이면 출력 그대로 |
| `tests/trajopt/test_e1_grasp_goal.py` (새 파일) | 15 test | 아래 |

`ta_offline.py` 는 고치지 않았다 (`VARIANTS` 에 넣으면 `--variants` 기본 목록이 바뀐다). offline 비교는 `tp_timing` 하나로 한다.

## flag 의미

| flag | config | 뜻 |
|---|---|---|
| (없음) | 모두 0 | 지금과 같다 |
| `--grasp-anchor W` | `w_grasp_tcp = w_grasp_goal = W` | IFG-TOA 그대로 (새 flag 를 안 주면 bit 동일, test 1) |
| `--grasp-anchor W --grasp-anchor-path-weight P` | `w_grasp_tcp = P`, `w_grasp_goal = W` | **E1-C** 는 `P = 0`: 경로 추종 항 없음, 목표 항(닫는 행 c 부터 `grasp_goal_rows`=3 행)만. 기록 `variant: "E1-C"` |
| `--grasp-goal-box MM` | `grasp_goal_box = MM/1000`, TOA 가중치 0 | **E1-B**: 손바닥 비용 없음. 목표 행 k, 축 a 마다 `g_a − r ≤ p_k(X) + J_k(Q_k − X_k) ≤ g_a + r` 를 slack `s ≥ 0` 하나로 (양쪽 공유) 완화, 가격 `w_slack/10` per m. merit 에 같은 `w_slack/10 × Σ max(0, |p−g|−r)`. 기록 `variant: "E1-B"` |
| `--grasp-anchor-radius-mm MM` | `grasp_anchor_radius` | `--grasp-anchor` 또는 `--grasp-goal-box` 와 함께 |

- **E1-B 와 `--grasp-anchor` 는 배타다** (CLI `ap.error`, config `TrajOptConfigError`). E1-B 에 경로 비용을 섞으면 명세의 "경로 추종 비용은 넣지 않는다" 와 어긋나서.
- gate (세 변형 공통, `safe_policy._grasp_anchor_gate`): latch `latched`/`closing`, 정책 chunk 가 처음 닫기를 계획한 요청에서 그 행의 손바닥(hold link 원점) 위치를 고정 (target 중심 120 mm 안일 때만), latch 가 벗어나거나 attach 취소 · target 변경이면 지움, 손바닥이 목표 150 mm 안일 때만 앵커를 넘김. 닫는 행 c 가 `0 < c < H` 가 아니면 목표 행이 없다 (c = 0 = 이미 닫힌 chunk, 들어 올리는 행).
- 대체 풀이 (TOA3, 공통): 앵커 풀이의 `max_violation > violation_tolerance` 면 앵커 없이 다시 풀고 침범이 더 작은 쪽 (같으면 앵커 쪽). 기록 `fallback {anchored_violation_mm, plain_violation_mm, used}` + E1 은 `fallback_used`.
- 기록 `to.metrics.grasp_anchor` (E1): `variant`, `goal_rows` (행 번호), `n_goal_rows`, `goal_err_result_mm` (Euclid 최대), `goal_err_axis_max_result_mm` (축별 최대), 정책 쪽 `_reference_` 같은 값, E1-B 는 `goal_box_mm` · `goal_box_slack_sum_mm` (반환 궤적에서 slack 이 가질 값의 합) · `goal_box_excess_max_mm`, `fallback_used`. 목표 행이 없는 chunk 는 `applied: False` + `reason`.

## offline 에서 목표를 다시 만드는 법 (`e1_goal.RecordedGraspGate`)

서버 함수 `SafePolicy._grasp_anchor_gate` 를 **복제하지 않고** stub(`object.__new__(SafePolicy)`, gate 가 읽는 속성만)에 기록 값을 넣어 그대로 부른다. run 의 모든 요청을 `t_step` 순서로 한 번 훑어 (setting 마다 stub 하나, 상태 유지) `{seq: (앵커, gate 기록)}` 를 만든다 — `--first-chunk/--max-chunks` 와 무관.

| gate 입력 | 기록에서 |
|---|---|
| latch 상태 (그 요청의 최종) | `summary_json.grasp.state` |
| attach 취소 | `summary_json.grasp.attach_revoked` |
| 손 · hold link · parent link | `grasp.hand` · `grasp.hold_link` · `grasp.parent_link` |
| 조작 물체 id · 중심 | `summary_json.manipulated.id` · `.centroid` (서버의 `manip.centroid` 와 같은 값) |
| 정책 chunk | `reference_chunk` |
| `q_now` | client planning row `qpos` (`ta_offline.Run.inputs` 와 같은 사상) |
| 닫힘 문턱 | `LatchConfig().gripper_closed_below` = 0.85 (서버에 이것을 바꾸는 flag 없음) |
| 120 mm · 150 mm | 각 setting 의 `cost.grasp_goal_max_from_target` · `cost.grasp_anchor_radius` (기본값) |

- planning row 가 없는 요청은 gate 가 보지 않는다 (tp_timing 도 풀지 않는다). ep211 에서는 75/75 가 다 있었다.
- `goal_eval` (모든 setting, E0 포함): 별도 probe gate (E1-B 꼴, 기본 반경) 가 앵커를 내는 chunk 에서, 반환 궤적과 정책 chunk 의 목표 행 손바닥 위치 – 목표 (Euclid 최대, 축별 최대, mm). E0 의 목표 오차도 같은 행 · 같은 목표로 잰다.
- `--replay-wrist-track`: 기록 서버는 `--wrist-track-scale 0.1` (IFG-TOW) 로 돌았는데 tp_timing 은 그 gate 를 재현하지 않았다. 이 flag 는 기록된 `to.metrics.joint_track_scale` 을 모든 setting 에 그대로 넘긴다. ep211 에서 E0 와 기록 refined chunk 의 차이 (실행 8 행, 관절 최대): 끄면 중앙값 0.0006° · p90 0.135° · 최대 1.29°, 켜면 0.0001° · 0.053° · 0.29°. **full 비교에서는 켜기를 권한다** (모든 setting 이 같은 replay 를 받으므로 비교는 공정하다).

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-development
# 새 test + TOA test
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=. \
  .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/trajopt/test_e1_grasp_goal.py tests/trajopt/test_ifg_toa_grasp_anchor.py
# trajopt 전체 + QpProblem 을 쓰는 ag3s test
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=. \
  .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/trajopt tests/ag3s/test_t46_target_free_residual.py
```

- `test_e1_grasp_goal.py` 15 passed + `test_ifg_toa_grasp_anchor.py` 8 passed = **23 passed** (모듈 `__file__` = 이 checkout 확인).
- trajopt 전체: __FULL__
- **`test_t43t_replay.py::test_off_is_bit_identical_to_the_pre_fix_commit` 는 HEAD 에서도 실패한다** (`AssertionError: to_T42_L1992_16`, `git archive HEAD` 사본 + 같은 venv 로 같은 key 실패 확인). 이 변경과 무관한 기존 실패다.

새 test 가 보이는 것:

| test | 내용 |
|---|---|
| `test_off_and_toa_are_bit_identical_to_head` | fixture (H=8, 침범 30 mm) 5 경우 — 앵커 없음 · TOA 2000 · TOA 20000 대체 풀이 · TOA 목표 항 · TOA 두 항 — 의 궤적 SHA-1 이 HEAD 사본에서 잰 값과 같다. 앵커 없음은 `grasp_anchor` 기록 키 없음 |
| `test_qp_problem_without_goal_box_is_unchanged` | `n_goal_slack = 0`, `n_var` · `flatten` · `kinematic_slack` 전과 같음 |
| `test_e1c_path_weight_zero_drops_the_path_term` | `w_tcp = 0` → track mask 없음, 경로 가중치 0, 목표 행 밖 QP 항 0, 목표 항은 TOA 의 목표 항과 같은 P · q |
| `test_e1c_solve_records_the_variant_and_no_path_rows` | 기록 `variant E1-C`, `n_track_rows 0`, 목표 오차 감소 |
| `test_goal_box_rows_hold_exactly_at_the_excess` | 선형 점 사상에서 slack = 축별 초과량일 때 행이 정확히 성립하고, 덜 주면 깨진다 |
| `test_e1b_without_collision_meets_the_box[10, 20]` | 충돌 없는 fixture (여유 50 mm), 정책은 목표에서 축별 162 mm → 결과 ≤ r + 0.5 mm (잰 값 10.01 / 20.01 mm), slack ≈ 0, 대체 풀이 없음 |
| `test_e1b_collision_wins_over_the_goal_and_the_slack_is_recorded` | 목표 = 깎이는 link 의 가장 깊은 침범 자세 (1 행, ±2 mm): 침범 0 으로 해소, 목표 오차 33.7 mm, slack 39.1 mm 기록, 대체 풀이 없이 |
| `test_e1b_anchored_violation_falls_back_to_the_plain_solve` | 상자 가격을 충돌보다 비싸게 (scale 100, monkeypatch) → 상자 풀이 침범 39.7 mm → 앵커 없는 풀이 0 mm 가 이김, `fallback_used True` |
| `test_e1b_without_goal_rows_is_the_plain_solve_with_a_record` | 닫는 행 없음 / 0 / horizon 밖 → 궤적이 앵커 없는 풀이와 같고 `applied False` 기록만 더해짐 |
| `test_safe_policy_gate_hands_out_the_box` | 서버 gate 가 `grasp_goal_box` 로 E1-B 앵커 (가중치 0), E1-C 설정으로 E1-C 앵커, 목표는 같은 값 |
| `test_recorded_gate_is_the_server_gate_on_a_record` | offline gate: searching 무시, 닫기 계획 전 없음, 처음 닫기 계획에서 고정, 정책의 닫는 점이 움직여도 목표 유지, 반경 밖 없음, held 에서 지움, 다시 고정, attach 취소로 지움; `goal_eval` 값 |
| `test_config_and_cli` | 기본 0, 음수 · 배타 거절, flag 없음 → override 키 없음, `--grasp-anchor 2000` = TOA 그대로, E1-C / E1-B override, 잘못된 조합 5 가지 `ap.error` |
| `test_offline_settings_are_registered_and_valid` · `test_summarize_counts_goal_chunks_and_all_chunks` | offline settings 와 요약 |

**offline 하네스 off 확인 (기록 replay):** HEAD 사본의 `tp_timing --variants V8` 와 이 checkout 의 `V8` · `E0` (replay flag 없음) 를 ep211 75 chunk 에서 비교 — `sha1` **75/75 같음**, 출력 JSON 의 최상위 · chunk · setting 키 같음. 다른 것은 기록된 config dict 의 `cost` 에 `grasp_goal_box: 0.0` 키가 하나 는 것뿐이다 (CostConfig 새 필드, IFG-TOA 때와 같은 방식).

## offline smoke (1 episode, 판정 아님)

```bash
RUN=/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581
O=$RUN/verify/AG3S-PRE-CORE-TO/E1/offline_smoke
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/AG3S_VLA-development OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/dev/AG3S_VLA-development/.venv-openpi-live/bin/python -m AG3S.trajopt.experiments.tp_timing \
  --index $RUN/verify/AG3S-GPU-STAGES/M2_cpu/e4_index.json --key E4/B_ROI_FPR_ep211_s2113 \
  --settings-module AG3S.trajopt.experiments.e1_goal --variants E0 E1C E1B10 E1B20 \
  --field cpu --mode server --replay-wrist-track --progress --out $O/B_ROI_FPR_ep211_s2113.json
.venv-ag3s/bin/python -m AG3S.trajopt.experiments.e1_goal summarize $O/B_ROI_FPR_ep211_s2113.json > $O/summary.json
```

- 산출: `$O/B_ROI_FPR_ep211_s2113.json` (chunk 75, 44.8 s), `$O/B_ROI_FPR_ep211_s2113.log`, `$O/summary.json`.
- 도구가 끝까지 돈다는 것만 본다. 숫자는 판정에 쓰지 않는다 (1 episode, 같은 시각 다른 pytest process 1 개가 돌아 시간도 오염).
- 그 run 에서 gate 가 앵커를 낸 요청: seq 11–22 · 62–67 (latched / closing). 목표 행이 실제로 있는 chunk (`goal_eval` 있음) 7 개.

## full 비교 명령 (lead 가 실행)

E4 18 장면 (S-REG). S-OCC 기록(e2 `OCC-APPLE-1345/round3/runs/` 등)은 같은 꼴의 index (`{"runs": {key: {"run_dir", "server_segment": {"first", "last", "n"}}}}`, `e4_index.json` 와 같음) 를 만들어 `IDX` 만 바꾼다.

```bash
RUN=/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581
IDX=$RUN/verify/AG3S-GPU-STAGES/M2_cpu/e4_index.json
O=$RUN/verify/AG3S-PRE-CORE-TO/E1/offline_e4
mkdir -p $O
for key in $(python3 -c "import json;print(' '.join(json.load(open('$IDX'))['runs']))"); do
  n=$(basename $key)
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/AG3S_VLA-development OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /mnt/dev/AG3S_VLA-development/.venv-openpi-live/bin/python -m AG3S.trajopt.experiments.tp_timing --index $IDX --key $key \
    --settings-module AG3S.trajopt.experiments.e1_goal --variants E0 E1C E1B10 E1B20 \
    --field cpu --mode server --replay-wrist-track --out $O/$n.json > $O/$n.log 2>&1 || echo "FAIL $key" >> $O/fail.txt
done
/mnt/dev/AG3S_VLA-development/.venv-ag3s/bin/python -m AG3S.trajopt.experiments.e1_goal summarize $O/*.json > $O/summary.json
```

- CPU 1 process, ep211 기준 episode 당 약 45 s × 18 ≈ 15 분. IFG-TOA 자체를 옆에 두려면 `--variants` 에 `TOA` 를 더한다.
- chunk 마다 setting 별로: `status`, `max_violation_mm`, `solve_ms`, `goal_eval.{result,reference}_err{,_axis_max}_mm`, `grasp_anchor` (변형 · 목표 행 · slack · `fallback_used`), chunk 의 `grasp_goal` (gate 기록: frozen, palm_to_goal_mm, reason …).

## verifier 가 알아야 할 것

- 새 flag: `--grasp-anchor-path-weight W` (E1-C, `--grasp-anchor` 와 함께), `--grasp-goal-box MM` (E1-B, `--grasp-anchor` 와 배타). 기본값 변경 없음. `tp_timing --replay-wrist-track` (기본 꺼짐).
- 새 config 키: `cost.grasp_goal_box` (기본 0.0) — `to_dict()` 에 키가 하나 는다. chunk 기록 (`summary_json`) 키는 꺼짐이면 같다.
- IFG-TOA (`--grasp-anchor` 만) 의 궤적은 같고, 그 `grasp_anchor` 기록에 `variant: "TOA"` · `goal_rows` · `goal_err_axis_max_*` 키가 는다 (꺼짐이 아니므로 허용 범위로 봤다).
- 재생산이 필요한 산출물: 없음.
- 옛 기록과 호환: 깨지지 않음 (`QpProblem` 새 필드 기본 0).

## 확인하지 못한 것 · 확실하지 않은 것

1. **look-ahead gate 가 닫힌 chunk 에서는 실행 창(8 행) 밖의 목표 행을 고치지 못한다.** gate 가 닫히면 창 optimizer(H = 8) 만 풀고 꼬리는 정책 그대로라, 닫는 행 c ≥ 8 이면 E1-B · E1-C 모두 그 chunk 에서는 효과가 없다 (TOA 도 같았다). smoke 의 seq 19 (c = 11, gate closed): E1-B 10 의 기록 slack 2.27 mm 는 고치지 않은 꼬리의 값이다. 실행되는 행이 아니므로 다음 chunk 에서 다시 본다는 설계로 두었지만, 이것이 문제인지는 lead 판단이다.
2. **시간:** 앵커가 켜지고 꺼질 때 QP 의 sparsity 가 바뀌어 OSQP 가 다시 setup 한다 (TOA 도 같음). smoke 에서 목표 행이 있는 chunk 의 OSQP 시간이 E0 의 약 2–4 배, 앵커 → plain 으로 바뀌는 chunk (seq 22) 에서 E1 쪽 OSQP 88–90 ms. 시간 판정은 thread 설정 (D1) 과 함께 봐야 한다.
3. **`goal_box_slack_sum_mm` 는 QP 의 slack 해가 아니라 반환 궤적에서 정확히 계산한 초과량이다** (QP slack 은 선형화 값이라 반환 궤적과 다르다). QP 해의 slack 은 `QpProblem.goal_slack(z)` 로 꺼낼 수 있지만 기록하지 않았다.
4. offline gate 는 서버 gate 함수를 그대로 부르지만, 입력은 기록에 있는 것뿐이다. latch 문턱(0.85)은 기본값을 썼고 (서버에 바꾸는 flag 가 없는 것을 grep 으로 확인), `manipulated.centroid` 가 서버가 그 순간 쓴 값과 같다고 가정했다 (같은 객체를 기록한 것).
5. `tp_timing --mode server` 에서는 setting 마다 optimizer · continuity 가 이어지므로, 한 chunk 의 작은 차이도 뒤 chunk 로 번진다 (ep211 E0 replay 켬/끔: TOW 가 처음 켜지는 seq 13 부터 거의 모든 chunk 의 SHA 가 다름). `--mode reset` 이면 chunk 마다 독립이다.
6. S-OCC 기록용 index 는 만들지 않았다 (위 꼴로 lead 가 만든다).
7. 그림은 만들지 않았다 (verifier 몫).

## 내가 기대하는 결과

> verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.

- E1-B 는 충돌과 목표가 겹치지 않는 chunk 에서 목표 오차를 r 안으로 넣고, 겹치면 침범을 먼저 해소하며 slack 을 남길 것이다. E1-C 는 211 · 1668 형 (평행 이동이 필요한) chunk 에서 TOA 보다 HOLD 가 적을 것으로 보지만, 목표 항만으로도 평행 이동을 비싸게 만들 수 있어 확실하지 않다.
