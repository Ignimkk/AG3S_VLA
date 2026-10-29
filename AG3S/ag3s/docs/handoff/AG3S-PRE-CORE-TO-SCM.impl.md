# AG3S-PRE-CORE-TO-SCM — 손바닥 껍질 구 행만의 여유 (`--palm-shell-margin MM`) 구현 · offline 확인

> writer: ag3s-implementer · 읽는 쪽: lead, verifier, scribe
> 작성 2026-10-26 20:56 KST (`date -u` 11:56 UTC + 9 h). Task: [`AG3S-PRE-CORE-TO.task.md`](AG3S-PRE-CORE-TO.task.md) §6 "CR · SCM · LTE" (사용자 판정 20:06 KST). 앞 단계: [`AG3S-PRE-CORE-TO-SC.impl.md`](AG3S-PRE-CORE-TO-SC.impl.md) (§6 what-if).
> 코드는 사본 `$RUN/snapshot_sc` 에서만 고쳤다 (main checkout 코드는 건드리지 않음). patch: `$RUN/verify/AG3S-PRE-CORE-TO/SCM/scm.patch` (1,505 줄, sha256 앞 16 `c38c65724835e4e4`, `snapshot_e3c` 대비 = SC + SCM, 코드 11 파일 + test 2 파일; main checkout 에 `git apply --check` 통과, sqp · serve_safe hunk 가 5–30 줄 밀려 붙음). `SC/sc.patch` 는 그대로 (`c00347b068469827`).
> `RUN = /mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581`. 원시 출력: `$RUN/verify/AG3S-PRE-CORE-TO/SCM/`.

## 0. 요약 (평이하게)

- **무엇을 했나:** TrajOpt 의 충돌 행 여유 규칙에서, 손바닥 껍질 구 (SC 의 `--palm-shell`, role `palm_shell`) 행만 다른 여유를 받게 하는 flag `--palm-shell-margin MM` 를 넣었다 (기본 = 안 줌 = 일반 `--obstacle-margin`). 최근접 표면이 장애물로 분류된 껍질 행만 `max(esdf_margin, 이 값)` 을 받고, 다른 모든 행 · 다른 분류 (target · 지지면 · 쥔 물체) 는 그대로다. `--palm-shell` 없이 또는 `--obstacle-margin 0` 으로 주면 서버 시작을 거절한다.
- **꺼짐은 그대로:** flag 를 안 주면 SC golden SHA (`dcc931c0ed740d85` · `e6f6986f5730bfc8`) 가 두 venv 에서 그대로이고, config dump · 기록 키 · client metadata 에 새 키가 없다. 일반 여유와 같은 값 (`--palm-shell-margin 10`) 을 주면 `--palm-shell on` 과 비트로 같다.
- **offline (실제 여유 경로, what-if 빼기 아님):** 성공 쥐기 20 run 의 닫기 1 s 창에서 껍질 새 위반 요청 **22 → 12** (run 12 → 7). SC what-if 와 요청 수 · run 수가 같다. 행 수는 what-if 613 대 실제 621 — target 으로 분류된 껍질 행 8 개는 obstacle 여유를 받지 않으므로 그대로 남는다 (ep250 한 요청). 막대 접촉 4 건은 5 mm 에서도 접촉 step 의 TrajOpt 행 값이 **−10.2 … −17.5 mm** 라 계속 위반으로 보이고, 접촉 3–16 step 전부터 보인다.
- **closed-loop 준비 (서버 없이):** E3 전체 명령줄 + `--palm-shell on --palm-shell-margin 5` 가 parse · 거절 검사 · `sqp_overrides` · config 를 통과하고, 구 **606** 개 (껍질 336), linearizer 껍질 열 336, 시작 log 줄 · client metadata 에 5.0 mm 가 찍힌다.
- **알아둘 차이 (gap):** cuRobo 초기값 (`--to-gpu-seeds curobo`, B0m world) 은 구마다 여유를 따로 둘 수 없어 껍질 행에도 10 mm 를 쓴다. 결과를 고르고 검사하는 CPU TrajOpt 는 5 mm 다. GPU 쪽은 이 pod (CPU) 에서 시험할 수 없어 고치지 않았다 (§4).

## 1. 설계 — 어느 층에 넣었나

`--obstacle-margin` 은 행마다 이렇게 붙는다 (`snapshot_sc/AG3S/trajopt/linearize.py:1009-1022`, `_esdf_clearance_core`):
행의 최근접 표면을 `_obstacle_rows` 가 obstacle · target · support · held 로 가르고 (거리 일치 검사), obstacle 행만 `margin = max(margin, obstacle_margin)` 을 받는다. 행 값 = `d − r − margin`.

SCM 은 같은 자리에서 obstacle 행의 그 값만 구별로 바꾼다:

| 껍질 행 (role `palm_shell`) | 다른 로봇 행 |
|---|---|
| obstacle 분류면 `max(margin, palm_shell_margin)` | `max(margin, obstacle_margin)` (그대로) |
| target · support · held 분류면 그대로 | 그대로 |

- 값의 출처: `serve_safe --palm-shell-margin MM` → `collision.palm_shell_margin` (m, `TrajOptConfig`) → `scene_from_constraint_set` 이 `SceneSnapshot.palm_shell_margin` 에 싣는다 (주었을 때만) → linearizer 가 자기 로봇 모델의 `sphere_roles` 로 껍질 열을 찾는다 (`_palm_shell_cols`).
- 분류 (`_obstacle_rows`) 는 여유와 무관하게 같다 — offline 59 요청 전부에서 분류 배열이 비트로 같았다.
- 여유 규칙은 TrajOpt (CCR) 층의 결정이다. 이번에는 사용자 판정 (20:06 KST) 이 이 층을 지정했다.
- IFG-NBC (이웃 접촉 허용) 는 `ee_finger_*` link 만 받으므로 껍질 행 (`ee_left` · `ee_right`) 에 겹치지 않는다.

## 2. 바뀐 파일 (줄 = `snapshot_sc` 기준)

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/trajopt/config.py:889-896` | `CollisionBackendConfig.palm_shell_margin: Optional[float] = None` (m) | 값의 자리. 기본 `None` = 껍질 행도 `obstacle_margin` |
| `AG3S/trajopt/config.py:1011-1020` | `validate`: `None` 이 아니면 유한 · ≥ 0 · bool 아님, `obstacle_margin > 0` 필요 | 효과 없는 설정을 거절 |
| `AG3S/trajopt/config.py:1171-1182` | `to_dict` 가 `None` 인 `palm_shell_margin` 을 빼고 씀 (`_OMIT_WHEN_NONE`) | config 를 통째로 기록하는 곳 (`--no-perception` TO-only recorder) 의 키를 이전과 같게. `from_dict` 는 기본값으로 되돌린다 |
| `AG3S/trajopt/linearize.py:232-236` | `SceneSnapshot.palm_shell_margin: Optional[float] = None` | scene 이 값을 나른다 |
| `AG3S/trajopt/linearize.py:1015-1022` | `_esdf_clearance_core`: 값이 있으면 껍질 열만 그 값으로 obstacle 여유 | §1 |
| `AG3S/trajopt/linearize.py:1137-1160` | `_palm_shell_cols(n_query)`: `robot_model.sphere_roles == "palm_shell"` 열 (쥔 물체 행 False). 껍질 구가 없는 모델이면 **ValueError** | 켰다고 믿었는데 아무 일도 없는 상태를 막는다 (서버 거절과 같은 규칙을 한 층 아래에서) |
| `AG3S/trajopt/linearize.py:1567-1572` | 최악 행 기록 (`max_violation_pair`) 에 `palm_shell_row` · `palm_shell_margin_m` (켰을 때만) | 기록에서 그 행이 어떤 여유를 받았는지 본다 |
| `AG3S/trajopt/linearize.py:2280-2283` | `scene_from_constraint_set`: `collision.palm_shell_margin` 이 있으면 scene 에 싣는다 | 배선 |
| `AG3S/trajopt/sqp.py:1375, 1517, 1521-1528` | `_scm_record()` → `to.metrics` 에 `palm_shell_margin_m` · `obstacle_margin_m` (켰을 때만, 두 metrics 경로 모두) | 요청마다 기록에 값이 남는다 |
| `AG3S/trajopt/safe_policy.py:387-389` | client metadata `palm_shell_margin_m` (켰을 때만) | client 기록에서 서버 구성을 본다 |
| `AG3S/trajopt/gpu_seeds.py:529-532` | GPU seed 기록에 `palm_shell_margin_gap` (켰을 때만: 우리 값 vs GPU 껍질 행 여유) | §4 gap 를 기록에 남긴다 |
| `AG3S/trajopt/serve_safe.py:727-746` | `palm_shell_margin_overrides(args)`: mm → m (`/1000`), `--palm-shell` 없음 · `--obstacle-margin 0` · 음수 · nan · inf → SystemExit | CLI → config |
| `AG3S/trajopt/serve_safe.py:747-756` | `announce_palm_shell_margin`: 서버 시작 WARNING 한 줄 (꺼짐이면 없음) | 시작 log 에 값 |
| `AG3S/trajopt/serve_safe.py:2156-2161` | CLI `--palm-shell-margin MM` (기본 None) | |
| `AG3S/trajopt/serve_safe.py:2908-2913` | `reject_bad_flag_combinations`: 체크포인트를 올리기 전에 거절 | |
| `AG3S/trajopt/serve_safe.py:3384` | `trajopt_config_from_args` 의 `collision` 에 `**palm_shell_margin_overrides(args)` | 안 주면 키 없음 |
| `AG3S/trajopt/serve_safe.py:3688` | `obstacle_margin` 시작 줄 다음에 `announce_palm_shell_margin` | |
| `AG3S/trajopt/experiments/ta_offline.py:156-158, 197-198` | variant `PSM5` (`--palm-shell on --palm-shell-margin 5` + config 5 mm) | lead 의 offline 다시 풀기용 |
| `AG3S/trajopt/experiments/ta_offline.py:585-588` | `Run.inputs` 가 직접 짓는 scene 에 `palm_shell_margin` (명령줄에 있을 때만) | offline 재구성이 서버와 같은 길 |
| `tests/trajopt/test_scm_palm_shell_margin.py` (새, 12 test) | §3 | |

CLI 단위: `--palm-shell-margin` 은 **mm** 다 (`--palm-shell-overflow` 와 같은 규약). `--obstacle-margin` 은 m (0.010) 이다.

## 3. 단위 검증

```bash
# cwd = $RUN/snapshot_sc, env = $RUN/verify/AG3S-PRE-CORE-TO/SC/env.sh (PYTHONPATH = snapshot_sc 먼저)
# AG3S.trajopt.sqp.__file__ = $RUN/snapshot_sc/AG3S/trajopt/sqp.py 확인 (두 venv)
.venv-openpi-live/bin/python -m pytest -q tests/trajopt/test_scm_palm_shell_margin.py               # 12 passed
bash $RUN/verify/AG3S-PRE-CORE-TO/SCM/tests/run_files.sh   # tests/trajopt + tests/ag3s, 파일마다 별도 프로세스, worker 2
#   209 파일 · 3,882 passed · 40 skipped · 4 failed  (SC 때 3,870 + 새 12; 4 failed 는 SC 와 같은 사전 실패)
.venv-ag3s/bin/python -m pytest -q tests/trajopt/test_scm_palm_shell_margin.py tests/trajopt/test_t41a_obstacle_margin.py  # 30 passed
.venv-ag3s/bin/python -m pytest -q tests/trajopt/test_sc_palm_shell.py                                                      # 26 passed
.venv-ag3s/bin/python -m pytest -q tests/trajopt/test_e1_grasp_goal.py                                                      # 15 passed
.venv-ag3s/bin/python -m pytest -q tests/trajopt/test_e2_multi_seed.py tests/trajopt/test_e3b_gpu_seeds.py                 # 19 passed, 4 skipped
```

- 4 failed (사전, 변경과 무관 — SC impl §4.2 와 같은 4 개): `test_t44_target_exclusion_owned::test_the_record_writes_the_trace_keys`, `test_e1_grasp_goal::test_off_and_toa_are_bit_identical_to_head`, `test_e2_multi_seed::test_off_is_bit_identical_to_head` · `test_not_triggered_when_the_standard_solve_resolves`. 뒤의 셋은 `.venv-ag3s` 에서 잰 golden 이라 `.venv-openpi-live` 에서만 실패하고, `.venv-ag3s` 에서는 위처럼 통과한다 (= E1 · E2 head 와 비트 동일). 원시 `SCM/tests/per_file/*.txt`, 합계 `SCM/tests/per_file_summary.json`, `.venv-ag3s` 결과 `SCM/tests/venv_ag3s/*.txt`.
- golden 다시 풀기 (`SC/golden_solve_sha.py`, 원시 `SCM/golden_*.txt`):

| 구성 | venv | seq 12 | seq 13 | 구 |
|---|---|---|---|---|
| flag 없음 | openpi-live · ag3s | `dcc931c0ed740d85` | `e6f6986f5730bfc8` | 270 |
| `--palm-shell on --palm-shell-margin 10` | openpi-live | `dcc931c0ed740d85` | `fa8d307e4d13cca6` (= SC `--palm-shell on` 값) | 606 |
| `--palm-shell on --palm-shell-margin 5` | openpi-live | `dcc931c0ed740d85` (껍질 행 비활성) | `98b8a19376884859` (바뀜) | 606 |

test 가 고정하는 것: (1) 안 주면 — CLI 키 없음 (E3 명령줄 포함), `to_dict` 키 없음 · round trip, golden SHA, `to.metrics` · scene 에 값 없음. (2) 10 mm (= 일반 여유) — 행 값 · 분류 · 다시 풀기 SHA 가 `--palm-shell on` 과 비트 동일. (3) 5 mm — 비껍질 열 비트 동일, 분류 동일, 껍질 열의 obstacle 행만 +5 mm (오차 1e-12), 그 밖의 껍질 행 0 변화, `to.metrics` 에 0.005 · 0.01. 최악 행 기록 키는 켰을 때만. client metadata 는 `palm_shell_margin_m` 하나만 다르다. 시작 log 는 켰을 때만. (4) 거절 — `--palm-shell` 없음 · `off` · `--obstacle-margin 0` · −1 · nan · inf, config 의 음수 · nan · inf · bool · 문자열 · `obstacle_margin 0`, 껍질 구 없는 모델에서 linearizer ValueError. (5) E3 전체 명령줄 + SCM 의 config 는 E3 와 `palm_shell_margin` 하나만 다르다.

## 4. verifier · lead 가 알아야 할 것

- **새 flag:** `--palm-shell-margin MM` (기본 None = 일반 여유). 기본값 변경 없음. 기록 키 (`to.metrics.palm_shell_margin_m` · `obstacle_margin_m`, `max_violation_pair.palm_shell_row` · `palm_shell_margin_m`, client metadata `palm_shell_margin_m`, GPU seed 기록 `palm_shell_margin_gap`) 는 켰을 때만 생긴다.
- **gap — cuRobo 초기값은 껍질 행에도 10 mm:** `gpu_seeds` 의 B0m world (`curobo_v2/composed_world.py:175, 297, 360-366`) 는 obstacle 여유가 스칼라 하나 (`m_obstacle`) 라 구마다 다른 값을 줄 수 없다. 그래서 GPU 가 내는 seed 는 껍질 행을 10 mm 로 밀고, 그 seed 를 검사 · 선택하는 CPU TrajOpt (`gpu_seeds.check` = `_finish`, first-pass 포함) 는 5 mm 로 본다. seed 가 더 보수적일 뿐 검사 기준이 바뀌지는 않는다. GPU 쪽을 구별 여유로 바꾸려면 `m_obstacle` 을 구 벡터로 바꿔야 하는데, 이 pod 에 CUDA 가 없어 시험할 수 없어서 넣지 않았다. closed-loop 전에 lead 가 이 gap 을 받아들일지 정한다.
- **`--obstacle-margin` 과 같은 의미의 `max`:** 껍질 obstacle 행 여유 = `max(esdf_margin, palm_shell_margin)`. E3 는 `--esdf-margin 0` 이라 5 mm 가 그대로 걸린다. `esdf_margin` 이 이 값보다 크면 효과가 없다 (`obstacle_margin` 과 같은 규칙).
- **target 분류 껍질 행은 바뀌지 않는다:** 쥐는 손의 껍질 구는 target-free 층에 묻는 권한 행이다 (SC §4.2 (4)). 그런 행이 target 으로 분류되면 obstacle 여유를 받지 않으므로 SCM 도 손대지 않는다 — SC what-if 가 모든 껍질 행에서 5 mm 를 뺀 것과 이 점이 다르다 (§5.2 의 8 행).
- 재생산할 산출물 없음. 옛 기록과의 호환 깨짐 없음 (키는 켰을 때만 생긴다).
- `ta_offline` 의 scene 재구성에는 `target_permission_strict` · 접촉 권한 행이 없다 (SC 와 같은 도구, 이전부터 그랬다). 아래 offline 수치는 PS · PSM5 모두 같은 재구성에서 쟀다.
- **TrajOpt 풀이 시간은 재지 않았다** (구 606 개, 독점 lock 필요). 행 수는 SC 와 같고, 여유 계산은 열 마스크 하나만 더한다.

## 5. offline 결과 — 실제 여유 경로 (CPU, worker 2)

스크립트 `SCM/scm_rows.py` (SC `sc_common.make_run` · `rows` 를 읽기만 함), job `SCM/a/jobs.json` · `SCM/b1/jobs.json` (SC b1 의 20 job 그대로, GT 귀속은 뺌), 실행 `SCM/run_offline.sh`, 원시 `SCM/a/a_rows.json` · `SCM/b1/b1_rows.json`, 요약 `SCM/summarize.py` → `SCM/summary_scm.json`.
모델 세 개를 기록된 서버 명령줄에서 짓는다: **old** (그대로, 270 구) · **PS** (`--palm-shell on`, 606) · **PSM5** (`--palm-shell on --palm-shell-margin 5`, 606; `trajopt_config_from_args` → scene `palm_shell_margin = 0.005` → `_esdf_clearance_core`). 행 값 = TrajOpt 의 `d − r − margin` (계획 궤적 `refined_chunk` 16 행). **what-if** = SC 와 같은 계산 (PS 껍질 행 < −5 mm).

### 5.1 ⓐ 막대 접촉 4 건 — 5 mm 에서도 TrajOpt 가 위반으로 본다

창 = 접촉 step `t_c` 의 16 step 전부터 `t_c` 까지, step 마다 그 step 을 실행한 요청의 행.

| 접촉 run | t_c (grasp 상태) | `t_c` 껍질 최저 행 값 PS (10 mm) | **PSM5 (5 mm)** | what-if | 껍질 위반 행 수 PS → PSM5 | 창 17 step 중 위반 step PS → PSM5 | PSM5 처음 위반 |
|---|---|---|---|---|---|---|---|
| E3 case3 L3 s13453 | 88 (latched) | −15.16 mm | **−10.16 mm** (obstacle) | −10.16 | 20 → 11 | 5 → 4 | t_c − 3 |
| E0 case3 L3 s13453 | 366 (latched) | −20.98 | **−15.98** (obstacle) | −15.98 | 45 → 35 | 14 → 12 | t_c − 16 |
| E0 case3 L3 s23453 | 324 (latched) | −18.00 | **−13.00** (obstacle) | −13.00 | 31 → 21 | 17 → 16 | t_c − 16 |
| E0 case3 L3 s33453 | 103 (latched) | −22.53 | **−17.53** (obstacle) | −17.53 | 49 → 39 | 7 → 5 | t_c − 16 |

- PS 값 = SC ⓐ 의 a2 (거리장 − r, 여유 없이 −5.16 · −10.98 · −8.00 · −12.53 mm) − 10 mm 와 맞는다. 5 mm 에서는 그 값 − 5 mm 다. 네 건 모두 `t_c` 의 가장 깊은 껍질 행은 obstacle 분류다.
- 비껍질 열은 모든 요청에서 PS 와 비트로 같다. E0 s23453 의 한 요청 (seq 41) 에서는 target 분류 껍질 행 1 개 때문에 what-if 와 행 집합이 1 개 다르다 (191 대 190).

### 5.2 ⓑ b1 — 성공 쥐기 20 run, 닫기 1 s 창 (59 요청)

| | 새 위반 행이 있는 요청 | 그런 run | 새 위반 행 수 | 가장 깊은 값 |
|---|---|---|---|---|
| PS (10 mm, SC 와 같은 계산) | 22 (SC 22) | 12 | 1,663 | −12.88 mm |
| what-if 5 mm (SC 계산, SC 12) | 12 | 7 | 613 | — |
| **PSM5 실제 경로** | **12** | **7** | **621** (obstacle 613 · target 8) | **−7.88 mm** |

- 요청 59 개 중 58 개에서 PSM5 위반 행 집합이 what-if 와 정확히 같다. 다른 하나 (ep250 seq 10, 오른손) 에서는 PS 위반 −5…0 mm 행 93 개 중 target 분류 8 개가 obstacle 여유를 받지 않아 PSM5 에서도 남는다 (그 요청의 위반 여부는 둘 다 "있음").
- 59/59 요청: 분류 배열 동일, 비껍질 열 비트 동일, 껍질 obstacle 행의 변화는 정확히 +5.000 mm, 다른 껍질 행 변화 0.
- 남는 7 run (PSM5 최저): ep119 −7.58 · ep250 −6.20 (오른손) · ep1668 −7.88 · case1 L1 s13453 −6.25 · L1 s33453 −5.44 · L3 s23453 −3.76 · L3 s33453 −0.32 mm. 사라지는 5 run: ep121 · ep126 · ep309 · ep325 · ep1837 (PS −3.1…−4.0 mm → PSM5 +1.0…+1.9 mm). 7 개 중 old 모델에도 위반 행이 있던 요청: 7 / 12.
- **판정 문장 없음** (이번 측정은 SC what-if 가 실제 경로와 맞는지의 확인). 결론: what-if 의 "22 → 약 12" 가 실제 여유 경로에서 **22 → 12** 로 확인되고, 막대 접촉은 5 mm 에서도 계속 보인다.

## 6. closed-loop 준비 확인 (서버 없이) — `SCM/readiness.py` → `SCM/readiness.json`

E3 kit 의 서버 명령줄 (`E3/closed_occ/p/launch_occ.sh` 의 BASE + SAFE, 96 인자, `--model-xml` 만 snapshot_sc) 과 + `--palm-shell on --palm-shell-margin 5` (100 인자):

| 항목 | E3 | E3 + SCM |
|---|---|---|
| parse · `reject_bad_flag_combinations` | 통과 | 통과 |
| `sqp_overrides` | `multi_seed 16 · gpu_seeds curobo · gpu_seeds_check first-pass` | 같음 |
| collision | obstacle 0.010 · esdf 0 · NBC 0.020 · strict | + `palm_shell_margin 0.005` |
| config dump 의 `palm_shell_margin` 키 | 없음 | 있음 |
| 제약 로봇 구 (`ta_offline.build_robot` = 서버의 `build_constraint_robot_model` 옵션) | 270 | **606** (껍질 336, `ee_left` · `ee_right`) |
| linearizer 껍질 열 | — | 336 |
| 시작 log 줄 | 없음 | `TO palm_shell_margin: 5.0 mm — … obstacle_margin 10.0 mm 그대로 …` |
| client metadata | `obstacle_margin_m 0.01` | + `palm_shell_margin_m 0.005`, `constraint_spheres 606` |

- 이 pod 에는 CUDA 가 없어 `--to-gpu-seeds curobo` 의 CUDA 확인 (`gpu_seeds.available` → `no CUDA device`) 을 "있음" 으로 바꿔 넣고 검사했다. 실제 확인은 서버 시작 때 돈다. `--roi-carve-device cuda` 는 parse 단계에서 막히지 않았다.
- AG3S 자체 (`ag3s_from_args` → `build_ag3s`) 는 짓지 않았다 (checkpoint · CUDA). 껍질 옵션이 그 길로 들어가는 것은 SC test (`test_server_main_path_passes_no_key_when_off`) 가 이미 고정한다.

## 7. 프로세스 · 캐시

- CPU 만, 동시 worker ≤ 2, `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu` (SC/env.sh). GPU · 서버 · closed-loop · git commit 없음. 내 프로세스는 모두 끝났다.
- storage run 안에 바깥을 가리키는 link 를 만들지 않았다 (`find $RUN/snapshot_sc -type l` = 기존 openpi 상대 link 2 개, `SCM/` 안 0 개). `E1/occ_index/make_links.sh` 를 쓰지 않았다 (`server_proc.txt` 를 직접 읽음). NFS 가 멈춘 옛 폴더는 열지 않았다.
- `python3 /mnt/dev/AG3S_VLA-development/scripts/setup/reclaim_cache.py $RUN $RUN/snapshot_sc` →
  `reclaim: ws 48.0->39.1 GiB | cache 8.9->0.0 GiB, 97600 files, 2 paths | skipped 22 open files (pid 903859,903902,903905,903924,906453,930724,930872,941304)` (건너뛴 open file 의 pid 는 내 프로세스가 아니다).

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다 -->
- flag 없음 · `--palm-shell off` 서버 · replay 는 이전과 비트 동일. `--palm-shell on --palm-shell-margin 10` 은 `--palm-shell on` 과 비트 동일.
- `--palm-shell on --palm-shell-margin 5` closed-loop: case3 L3 의 손바닥–막대 접촉 자리는 TrajOpt 에 계속 위반으로 보인다 (접촉 step −10…−18 mm). 쥐기 근처 껍질 위반 요청은 PS 보다 줄지만 (offline 22 → 12) 남는다 — 남는 7 run 은 palm cover · PS 처럼 닫기 직전에 팔이 수정될 수 있다. 풀이 시간은 PS (606 구) 와 같은 수준일 것.
