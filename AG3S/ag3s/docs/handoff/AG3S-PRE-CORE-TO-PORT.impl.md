# AG3S-PRE-CORE-TO-PORT — 실험 사본의 E3-b 1 차 검사 · 손바닥 껍질 구 · 껍질 행 여유를 main checkout 으로 옮김

> writer: ag3s-implementer · 읽는 쪽: lead, verifier, scribe · 2026-10-28 00:20 KST (`date -u` 15:20 UTC + 9 h) 작성 시작
> 명세: lead 지시 (2026-10-28) — 사용자 판정 2026-10-27 23:55 KST "껍질 구 (+ 여유 5 mm) 채택" 의 후속
> ([`AG3S-PRE-CORE-TO.task.md`](AG3S-PRE-CORE-TO.task.md) §10 2026-10-27 00:11 줄: "main checkout 에 E3 1 차 검사 · 껍질 구 코드 옮기기").
> 기준: `feature/ag3s-development` HEAD `b3b69311` + 작업 트리 (STEMCAP 포함). **commit 안 함.**
> `RUN = /mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581`

## 무엇을 했나 (평이한 요약 먼저)

지금 serving 구성 (E3 = E2 16 seed + cuRobo GPU seed + **1 차 검사** `--to-gpu-seeds-check first-pass`) 과 채택된 손바닥
껍질 구 (`--palm-shell` · `--palm-shell-overflow` · `--palm-shell-margin`) 는 실험 사본 (`$RUN/snapshot_e3c` · `snapshot_pt` ·
`snapshot_scpt`) 에만 있었고, 꼭지 덮개 (`--target-stem-cap`) 는 checkout 에만 있었다. 이제 checkout 하나에 셋이 모두 있다.

- **옮긴 코드는 사본과 글자 그대로 같다.** 사본 사이 관계를 먼저 확인했다: `snapshot_pt` = `snapshot_e3c` + client 흔들림 flag
  (`pi05_infer.py` · `tests/test_pt_first_chunk_perturb.py` 만 다름), `snapshot_scpt` = `snapshot_pt` + SC · SCM
  (= `snapshot_e3c` + `SCM/scm.patch`, 13 파일 모두 비트 동일 확인), `snapshot_sc` = `snapshot_scpt` (옮긴 15 파일 비트 동일).
- checkout 은 그 사본들을 뜬 뒤 **꼭지 덮개만** 더해졌다 (`serve_safe.py` 의 4 hunk + `ag3s/config.py` · `curobo_builder.py` ·
  `constraint_record.py` · test 1 개 — 이 넷은 사본이 건드리지 않은 파일). 그래서 `serve_safe.py` 를 뺀 14 파일은 checkout 이
  "사본 − 이번에 옮길 것" 과 정확히 같아 (아래 §확인 1) `snapshot_scpt` 파일로 바꾸었고, `serve_safe.py` 는 3-way merge
  (`git merge-file`, 충돌 0) 로 꼭지 덮개 4 hunk 를 지키며 14 hunk 를 더했다.
- 기본값은 하나도 바뀌지 않았다: 세 기능 모두 flag 를 안 주면 config · 기록 키 · 모델이 이전과 같다 (사본의 test 가 그것을 고정하고, 여기서 다시 돌렸다).

## 바뀐 파일 (줄 = 옮긴 뒤 checkout 기준) — 더한 hunk 전부

기능 표시: **FP** = E3-b 1 차 검사 (`snapshot_e3c`), **SC** = 손바닥 껍질 구, **SCM** = 껍질 행 여유 (`snapshot_scpt`). `+n −m` = 옮기기 전 checkout 대비 줄 수. `−` 줄은 모두 같은 줄을 늘린 것 (닫는 괄호 · 반환 줄 등, 기존 동작 삭제 없음).

| 파일 | 더한 hunk (줄) | 기능 | 무엇이 |
|---|---|---|---|
| `AG3S/trajopt/serve_safe.py` (+127 −0, 14 hunk) | 685–758 | SC · SCM | `PALM_SHELL_MODES`, `announce_palm_shell`, `palm_shell_options`, `palm_shell_margin_overrides`, `announce_palm_shell_margin` |
| | 1184–1185 | SC | `ag3s_from_args` → `build_ag3s(palm_shell=)` (off 면 키 없음) |
| | 1223 | SC | `build_ag3s(..., palm_shell=None)` 인자 |
| | 1295–1297 | SC | `build_constraint_robot_model` 에 `palm_shell` (켰을 때만) |
| | 1382 | SC | `announce_palm_shell(...)` 시작 log |
| | 1934–1938 | FP | CLI `--to-gpu-seeds-check {all,first-pass}` (기본 all) |
| | 2172–2185 | SC · SCM | CLI `--palm-shell {off,on,left,right}` · `--palm-shell-overflow MM` · `--palm-shell-margin MM` |
| | 2953–2964 | SC · SCM | `reject_bad_flag_combinations`: 껍질 + `--no-safe`/`--no-perception`/`--palm-cover`/`--gripper-cover` 거절, 여유 검사 |
| | 2999–3001 | FP | `--to-gpu-seeds-check first-pass` 인데 `--to-gpu-seeds off` → `ap.error` |
| | 3061–3062 | FP | `sqp_overrides`: `gpu_seeds_check` (all 이면 키 없음) |
| | 3375–3378 | FP | 시작 log (first-pass 일 때만) |
| | 3434–3435 | SCM | `trajopt_config_from_args` 의 `collision` 에 `**palm_shell_margin_overrides(args)` |
| | 3675–3677 | SC | client metadata `palm_shell` (켰을 때만) |
| | 3739 | SCM | `announce_palm_shell_margin(...)` 시작 log |
| `AG3S/trajopt/config.py` (+37 −1) | 718–724, 733–737 | FP | `SqpConfig.gpu_seeds_check = "all"` + 검증 |
| | 889–896, 1011–1020, 1173–1181 | SCM | `CollisionBackendConfig.palm_shell_margin = None` + 검증 + `to_dict` 가 `None` 이면 키를 뺌 (`_OMIT_WHEN_NONE`) |
| `AG3S/trajopt/sqp.py` (+44 −3) | 141–142, 471–505 | FP | `self.gpu_seeds_check`; `_run_seeds` 의 `order_key` · `resolves` → `GS.solve_seeds(first_pass=)`, 검사 안 한 후보는 `checked: False` 기록 |
| | 1374–1375, 1517, 1520–1528 | SCM | `_scm_record()` → `to.metrics` 에 `palm_shell_margin_m` · `obstacle_margin_m` (켰을 때만) |
| `AG3S/trajopt/gpu_seeds.py` (+44 −7) | 620–705 | FP | `solve_seeds(..., first_pass=None)`: clip 과 검사를 나누고 순서대로 검사 · 첫 해소에서 멈춤, `record["first_pass"]` |
| | 527–532 | SCM | GPU 기록 `palm_shell_margin_gap` (켰을 때만) |
| `AG3S/trajopt/linearize.py` (+47 −1) | 232–236, 1016–1022, 1137–1161, 1567–1572, 2280–2283 | SCM | `SceneSnapshot.palm_shell_margin`, 껍질 열만의 obstacle 여유, `_palm_shell_cols` (껍질 구 없으면 ValueError), 최악 행 기록 키, `scene_from_constraint_set` 배선 |
| `AG3S/trajopt/safe_policy.py` (+3) | 387–389 | SCM | client metadata `palm_shell_margin_m` (켰을 때만) |
| `AG3S/trajopt/experiments/ta_offline.py` (+15) | 154–158, 195–198, 270–272, 586–588 | SC · SCM | variant `PS` · `PSM5`, `build_robot` 이 `--palm-shell` 을 읽음, `Run.inputs` scene 에 여유 |
| `AG3S/trajopt/experiments/e3b_gpu_seeds.py` (+9 −3) | 10, 25, 44, 76–77, 84–85, 88–89 | FP | offline setting `E3b_16_fp`, summary `n_checked` |
| `AG3S/trajopt/curobo_v2/export_inputs.py` (+1 −1) | 60 | SC | `--palm-shell` 을 재구성 못 하는 flag 목록에 추가 |
| `AG3S/ag3s/robot_models/urdf_sphere_chain.py` (+8 −1) | 240–249 | SC | `PALM_SHELL_ROLE = "palm_shell"`, `EXACT_SPHERE_ROLES` 에 포함 |
| `AG3S/ag3s/experiments/sources/gripper_cover.py` (+100 −1) | 50, 323, 326, 1023–1119 | SC | role import, `grasp_limits` 가 껍질 구를 뺌, `palm_shell_parts` · `palm_shell_spheres` · `palm_shell_capsules` |
| `AG3S/ag3s/experiments/reports/grounding_report.py` (+51 −1) | 154–157, 235, 289–301, 421–454 | SC | `build_constraint_robot_model(palm_shell=None)`: 켜면 껍질 구를 맨 뒤에 붙여 다시 짓고 `palm_shell_report` |
| `tests/trajopt/test_e3b_gpu_seeds.py` (+81) | 20–22, 383–460 | FP | 사본의 1 차 검사 test 4 개 (GPU 2 개는 CUDA 없으면 skip) |
| `tests/trajopt/test_sc_palm_shell.py` (새, 사본 그대로) | 전부 | SC | 19 test 함수 (parametrize 포함 26 test) |
| `tests/trajopt/test_scm_palm_shell_margin.py` (새, 사본 그대로) | 전부 | SCM | 12 test |
| `tests/trajopt/test_pre_core_to_port.py` (새, 이 STEP) | 전부 | 통합 | 4 test (아래) |

**이미 있던 것 (옮기지 않음):**
- E3-b 본체 (`gpu_seeds.py`, `sqp.gpu_seeds`, `--to-gpu-seeds`, warm-up, `e3b_gpu_seeds.py`, `test_e3b_gpu_seeds.py` 의 6 test), E2 · E3-a · E1 — checkout 작업 트리에 이미 있었다 (사본 `snapshot_e3c` 와의 차이는 위 FP hunk 뿐).
- client `--first-chunk-perturb-rad` · `--perturb-seed`: checkout `integration/rby1/bringup/pi05_infer.py` 와 `tests/test_pt_first_chunk_perturb.py` 가 `snapshot_pt` · `snapshot_scpt` 와 **비트 동일** (`cmp`). 옮길 것 없음.
- 꼭지 덮개 (STEMCAP): checkout 에만 있다. `serve_safe.py` 의 4 hunk (`target_exclusion_section` 매핑, 시작 log, CLI 4 flag, 거절 검사) 와 `AG3S/ag3s/config.py` · `fields/curobo_builder.py` · `experiments/sources/constraint_record.py` · `tests/ag3s/test_pre_core_to_stemcap.py` 를 그대로 두었다 (사본은 이 파일들을 바꾸지 않았다).

## 확인 — 사본과 같은가

1. **기준 확인 (옮기기 전):** 옮긴 15 파일 중 `export_inputs` · `ta_offline` · `linearize` · `safe_policy` · `grounding_report` · `gripper_cover` · `urdf_sphere_chain` 7 개는 checkout = `snapshot_pt` = `snapshot_e3c` (md5). `config.py` · `sqp.py` · `gpu_seeds.py` · `e3b_gpu_seeds.py` · `test_e3b_gpu_seeds.py` 는 checkout → `snapshot_e3c` 차이가 **FP 줄을 더하는 것뿐** (`diff` 의 `<` 줄 = 바뀐 줄의 옛 모양뿐, 다른 내용 없음), 그리고 `snapshot_pt` = `snapshot_e3c`. `serve_safe.py` 의 checkout → `snapshot_e3c` 차이 = STEMCAP 4 hunk (checkout 에만) + FP 4 hunk (사본에만), 다른 것 없음.
2. **옮긴 뒤:** 14 파일 (`serve_safe.py` 제외 코드 11 + test 3) 이 `snapshot_scpt` 와 `cmp` 비트 동일.
3. **`serve_safe.py`:** `diff checkout snapshot_scpt` = STEMCAP 블록 4 개 (1019–1026, 1471–1476, 2044–2057, 2671–2693) 를 checkout 이 더 가진 것뿐 (`>` 줄 0). 즉 옮긴 뒤 checkout 에서 STEMCAP 블록을 빼면 `snapshot_scpt` 와 글자 그대로 같다. 의도한 통합 줄은 없다 (두 기능의 hunk 가 서로 다른 자리).
4. `snapshot_e3c` + `SCM/scm.patch` (`git apply`) = `snapshot_scpt` (patch 의 13 파일 비트 동일) — 옮긴 것이 SC · SCM 보고의 patch 와 같다.

방법 (재현): merge base = checkout 의 `serve_safe.py` 에서 STEMCAP 줄을 뺀 것 (= `snapshot_e3c` 에서 FP 줄을 뺀 것, `difflib` opcode 가 delete · insert 뿐임을 확인), `git merge-file -p checkout base snapshot_scpt` → 충돌 0. 옮기기 전 파일과 md5 는 session scratchpad `before/` 에 두었다.

## 단위 검증

`$RUN/verify/AG3S-PRE-CORE-TO/PT/E3G_REG_DONE` 이 생긴 뒤에만, checkout 에서 (worktree 아님), CPU 만 (`CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu`), `nice -n 19`, 파일마다 별도 프로세스 · 동시 2 개, `.venv-ag3s` 다음 `.venv-openpi-live`.

```bash
source scripts/setup/env.sh
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu MUJOCO_GL=osmesa OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nice -n 19 .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider <file> -rfE      # 파일마다, xargs -P 2
nice -n 19 .venv-openpi-live/bin/python -m AG3S.trajopt.serve_safe --help
```

실행: 00:40–01:05 KST (`.venv-ag3s`), 01:06–01:09 KST (`.venv-openpi-live`). `__file__` = checkout (worktree 아님). 원시 로그는 session scratchpad `tests/out_focus` · `out_all` · `out_live`.

| 실행 | 파일 | passed | skipped | failed |
|---|---|---|---|---|
| `.venv-ag3s` — `tests/trajopt` + `tests/ag3s` 전부 + `tests/test_pt_first_chunk_perturb.py` | 215 | **3,824** | 63 | **1** |
| `.venv-openpi-live` — `.venv-ag3s` 에서 모듈 전체가 skip 된 10 파일 (torch · cuRobo `importorskip`, 꼭지 덮개 test 포함) + 옮긴 4 파일 + `test_t43t_replay.py` | 15 | **170** | 11 | **1** |

옮긴 기능 test (두 venv 모두 통과): `test_sc_palm_shell.py` 26, `test_scm_palm_shell_margin.py` 12 (E4 1345 golden 다시 풀기 SHA `dcc931c0ed740d85` · `e6f6986f5730bfc8` 포함 — 꺼짐 = 사본 SC 전 코드와 비트 동일), `test_e3b_gpu_seeds.py` 5 passed · 4 skipped (GPU test 4 개: E3-b 2 + FP 2, CUDA 숨김), `test_pre_core_to_port.py` 4, `test_pre_core_to_stemcap.py` 10 (`.venv-openpi-live`), `test_pt_first_chunk_perturb.py` 11.

실패 2 개 — 둘 다 **옮기기 전 코드에서도 똑같이 실패** (scratchpad 에 옮기기 전 15 파일로 만든 overlay, `__file__` 이 overlay 인 것 확인 후 같은 venv 로 실행):

| test | venv | 옮긴 뒤 | 옮기기 전 overlay | 비고 |
|---|---|---|---|---|
| `tests/ag3s/test_t44_target_exclusion_owned.py::test_the_record_writes_the_trace_keys` | openpi-live | FAILED | FAILED | lead 가 알려 준 사전 실패 |
| `tests/trajopt/test_t43t_replay.py::test_off_is_bit_identical_to_the_pre_fix_commit` | ag3s | FAILED (`AssertionError: to_T42_L1992_16`, 마지막 bit 차이) | FAILED | `.venv-openpi-live` 에서는 통과 (9 passed). numpy 주 버전에 따른 golden — E3-b impl 이 같은 이유로 `--deselect` 했던 test |

`serve_safe --help` (`.venv-openpi-live`, rc 0): `--to-multi-seed` · `--to-gpu-seeds` · `--to-gpu-seeds-check` · `--palm-shell` · `--palm-shell-overflow` · `--palm-shell-margin` · `--target-stem-cap` · `--stem-cap-radius-mm` · `--stem-cap-top-mm` · `--stem-cap-bottom-fit-frac` 모두 있다 (test 로도 고정).

새 test `tests/trajopt/test_pre_core_to_port.py`:

| test | 내용 |
|---|---|
| `test_e3_first_pass_stem_cap_and_palm_shell_parse_together` | E3 closed-loop 명령줄 전체 (`--to-multi-seed 16 --to-gpu-seeds curobo --to-gpu-seeds-check first-pass`) + `--target-stem-cap in_ball` (+ 값 3 개) + `--palm-shell on --palm-shell-overflow 5 --palm-shell-margin 5` 가 parse · `reject_bad_flag_combinations` 를 통과하고, `sqp_overrides` · `target_exclusion_section` · `palm_shell_options` · `palm_shell_margin_overrides` · `trajopt_config_from_args` · `build_ag3s` 인자 (monkeypatch, 모델은 짓지 않음) 에 각 값이 있다 |
| `test_each_feature_moves_only_its_own_keys` | E3 (`all`) 에 세 기능을 하나씩 더하면 자기 자리 (sqp · target_exclusion · palm_shell · collision) 만 바뀐다; off 값을 명시해도 (`all` · `--palm-shell off` · `--target-stem-cap off`) 아무 키도 생기지 않는다 |
| `test_first_pass_and_shell_margin_still_refused_alone` | GPU seed 없는 1 차 검사, 껍질 없는 여유, 덮개 없는 덮개 값, 껍질 + palm cover 는 여전히 거절 |
| `test_help_lists_every_ported_flag` | `--help` 에 E3 · FP · SC · SCM · STEMCAP flag 10 개 |

`gpu_seeds.available` 은 CPU test 에서 CUDA 가 없어 `--to-gpu-seeds curobo` 를 거절하므로 test 안에서만 `(True, "")` 로 바꿨다 (SCM test 와 같은 방식). 실제 probe 는 서버 시작 때 돈다.

## verifier 가 알아야 할 것

- 새 flag (checkout 기준): `--to-gpu-seeds-check {all,first-pass}` (기본 all), `--palm-shell {off,on,left,right}` (기본 off), `--palm-shell-overflow MM` (기본 없음 = 5 mm), `--palm-shell-margin MM` (기본 없음 = `--obstacle-margin`). 기본값 변경 없음. config 키 `sqp.gpu_seeds_check` (`to_dict` 에 항상 — E2 · E3-b 의 `sqp` 키와 같은 방식), `collision.palm_shell_margin` (`None` 이면 `to_dict` 에서 빠짐).
- 사본에서 띄우던 구성은 이제 checkout 에서 같은 명령줄로 띄울 수 있다: E3 kit + `--palm-shell on --palm-shell-margin 5` (+ 필요하면 `--target-stem-cap ...`). 껍질 + 꼭지 덮개를 **함께 켠 closed-loop 는 아직 아무도 돌리지 않았다** (parse · 배선만 test).
- GPU 경로 (`--to-gpu-seeds curobo` 의 실제 풀이, FP 의 GPU test 2 개, SC/SCM 의 cuRobo world) 는 이 pod 의 GPU 가 독점 closed-loop 에 쓰이고 있어 **돌리지 않았다** (GPU smoke 없음, 지시대로). 그 test 들은 CUDA 가 없어 skip 됐다.
- SCM 의 알려진 gap 그대로: cuRobo 초기값 world 는 껍질 행에도 obstacle margin 10 mm, CPU TrajOpt 는 5 mm (기록 `gpu.palm_shell_margin_gap`).
- 재생산이 필요한 산출물: 없음. 옛 기록과 호환: 깨지지 않음 (켰을 때만 새 키).
- 문서 · 그림 · storage 의 실험 kit 은 건드리지 않았다. scribe 몫: 로그 반영.

## 옮기지 않은 것

- client (`pi05_infer.py`): 이미 비트 동일 — 옮길 것 없음.
- 사본의 그 밖의 차이: 없음. checkout ↔ 각 사본은 `AG3S/` · `tests/` · `integration/` · `common/` 을, `snapshot_pt` ↔ `snapshot_scpt` 는 거기에 `scripts/` 까지 비교했다. `AG3S/ag3s/asset/data/attention_16d_*.npz` 7 개는 사본을 뜰 때 일부러 뺀 link 라 checkout 에만 있다 — 그대로 둠.
- 실험 스크립트 (`$RUN/verify/AG3S-PRE-CORE-TO/SC/*.py`, `SCM/readiness.py`, `SC/golden_solve_sha.py`): storage 의 실험 kit 이라 옮기지 않았다. `test_sc_palm_shell.py` · `test_scm_palm_shell_margin.py` 의 golden 다시 풀기 test 는 E4 1345 기록 (`$RUN/verify/AG3S-GPU-STAGES/M2_cpu/e4_index.json`) 을 읽는다 (없으면 skip).

## 내가 기대하는 결과

> verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.

- checkout 의 `serve_safe` 를 사본 `snapshot_scpt` 와 같은 명령줄 (E3 kit + `--palm-shell on --palm-shell-margin 5`) 로 띄우면 같은 입력에서 refined chunk 가 비트로 같을 것이다 (꼭지 덮개 off 일 때). 꼭지 덮개를 같이 켜면 target-free 층만 바뀌고 구 모델 · TrajOpt 설정은 같을 것이다.
