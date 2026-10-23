# SL5j-recorder — 구현

> writer: ag3s-implementer · 읽는 쪽: lead, verifier, scribe · 작성 2026-10-23 01:54 KST · 기준 HEAD `32a4ca8f` (commit 안 함)
> 앞 단계: `SL5j-launcher.impl.md`. launcher 변경(`benchmark/runners/launch_libero_paired.py`, commit 전)은 건드리지 않았다.

## 무엇을 했나 (평이한 요약 먼저)

Panda server가 이제 `--record-constraints DIR`를 실제로 처리한다. flag가 있으면 RB-Y1 server와 같은 방식(`background=True`,
같은 `esdf_mode`, 같은 meta key)으로 `ConstraintRecordWriter`를 만들어 `SafePolicy(recorder=...)`에 넘긴다. server가 끝나면
`close()`를 반드시 부른다. 정상 종료, SIGINT(launcher `stop_own_process`), SIGTERM 세 경우 모두 해당하며, 그래서 큐에 남은
청크가 쓰이고 `meta.json`이 남는다. flag가 없으면 `recorder=None`이고, `SafePolicy` 인자와 `info`(`panda_server_config.json`)는
SL5j 전 `server.py`와 같다. 이것은 옛 파일을 git에서 불러와 새 파일과 나란히 돌리는 test로 고정했다.

CPU에서 Panda SafePolicy(legacy ESDF)를 녹화된 SL3 dev call 3개로 돌려 실제 청크를 확인했다. `sphere_centres`, `sphere_link_names`,
`clearance`, ESDF 주 계층이 들어 있었다. 기록기를 켜도 응답 `actions`는 sha1까지 같았다. cuRobo 다계층(`coarse`, `free`)은
GPU가 필요해서 확인하지 못했다. GPU smoke에서 확인해야 한다(아래 "verifier가 알아야 할 것" 2).

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/adapters/panda/server.py:29-34` | module docstring에 `--record-constraints` 설명 | |
| `AG3S/adapters/panda/server.py:205-208` | `PandaSafeServer.recorder` property(= `SafePolicy.recorder`) | `main`이 close할 대상을 찾기 위해 |
| `AG3S/adapters/panda/server.py:293-310` | `check_record_flags(args)`: flag가 없으면 아무것도 하지 않는다. `SUPPORTED_ESDF_BACKENDS` 밖의 backend는 `SystemExit`(RB-Y1 `serve_safe.py:2342-2350`과 같은 거절). Panda에서만 `--no-safe`와 함께 주면 `SystemExit` | RB-Y1은 `--no-safe` + flag를 조용히 무시한다. 그러면 빈 기록이 남으므로 Panda에서는 거절한다. flag를 줄 때만 해당하므로 기본 동작은 그대로다 |
| `AG3S/adapters/panda/server.py:313-354` | `recorder_meta(...)`: RB-Y1 meta key(`serve_safe.py:2296-2340`, 선택 key 규칙도 같다)에 Panda 값과 Panda key를 더한다(아래 표) | task 1항 |
| `AG3S/adapters/panda/server.py:357-376` | `constraint_recorder(...)`: flag가 없으면 `None`. 있으면 `ConstraintRecordWriter(DIR, esdf_mode=..., background=True, meta=...)`를 만든다. 시작 log와 cuRobo + `occupancy` 경고는 RB-Y1 문구와 같다 | |
| `AG3S/adapters/panda/server.py:379-391` | `close_recorder(served)`: recorder가 없으면 `None`. 있으면 `close()`를 부른다. 실패하면 `logging.exception`만 남기고 예외는 다시 던지지 않는다(종료 원인을 가리지 않으려고) | |
| `AG3S/adapters/panda/server.py:417, 420` | `recorder = constraint_recorder(...)` → `SafePolicy(..., recorder=recorder, ...)`. 예전에는 `recorder=None` 고정이었다 | |
| `AG3S/adapters/panda/server.py:436-442` | recorder가 있을 때만 `recorder.meta["grasp_contact"]`(적용된 pb4–pb7 값)과 `info["record_constraints"]`를 넣는다 | flag가 없으면 info key 집합이 그대로다 |
| `AG3S/adapters/panda/server.py:463` | `main` 첫 줄에서 `check_record_flags(args)`를 부른다. checkpoint를 올리기 **전**에 거절한다 | |
| `AG3S/adapters/panda/server.py:483-495` | recorder가 있으면 `ss.exit_cleanly_on_sigterm()`(SIGTERM → `SystemExit`)를 부른다. `serve_forever()`를 `try/finally`로 감싸 `close_recorder`를 부른다 | task 2항 |
| `AG3S/adapters/panda/profile.py:51-55` | "parse만 하고 쓰지 않는다" 문장을 바로잡았다. AG3S build에는 안 쓰고, 폐루프 server가 TO flag(SL-5)와 기록 flag(SL5j)를 적용한다 | task 4항 |
| `AG3S/adapters/panda/profile.py:546-548` | `manifest_entry()["not_ag3s_settings"]` 위에 주석을 달았다. **문자열 값은 byte 단위로 그대로 뒀다** | 모든 run manifest에 들어가는 값이다. 바꾸면 flag가 없을 때도 manifest가 달라진다. "offline"이라는 한정이 아직 맞다 |
| `AG3S/adapters/panda/profile.py:246-264, 560` | `version_flag_values(args)`, `match_ag3s_version(args)`, `__all__` | server argv에는 launcher의 `--ag3s-version` 이름이 없다. parse된 flag에서 이름을 되찾아 meta에 남긴다 |
| `tests/benchmark/test_panda_sl5j_recorder.py` (신규, 15 test) | 아래 | |

### meta(`run_*/meta.json`)

| key | 값 | RB-Y1과 비교 |
|---|---|---|
| `config`, `checkpoint`, `links` | argv 그대로 | 같다 |
| `shadow`, `exclude_links`, `self_collision`, `target_field_policy`, `collision`, `limits` | 기본이 아닐 때만 생긴다 | 같은 규칙. V11AB에서는 `self_collision: false` |
| `joint_limits`, `sqp` | `joint_limits_meta`/`sqp_meta`, **Panda의 실제 `to_config`**(`_replace_horizon_limits` 뒤) | 값이 실제로 쓴 값이다(`source: urdf`) |
| `plan_horizon` | 실제 `to_config.horizon.planned` = 10 | RB-Y1은 argv `--plan-horizon`(16)을 남긴다. Panda는 그 값을 쓰지 않는다 |
| `model_xml` | `None`, 대신 `robot_urdf`를 남긴다 | argv의 RB-Y1 MJCF는 Panda server가 읽지 않는다 |
| `sphere_options`, `gripper_cover`, `finger_cover`, `self_filter` | 준 경우에만 생긴다 | 같은 규칙. `v11ab_build_kwargs_not_applied: ["constraint_sphere_options","finger_cover"]`로 Panda model에 적용되지 않았다는 것을 함께 적는다(`info`와 같다) |
| `subtask` | `{"gate": <args>, "probe": null}` | Panda에는 probe가 없다 |
| **Panda 추가** | `robot: "panda"`, `robot_urdf`, `ag3s_version`(예 `V11AB-pb7`, 맞는 것이 없으면 `null`), `version_flags`(AG3S_VERSIONS가 쓰는 dest 9개의 값), `conversion: {mode, name}`, `esdf_backend`, `horizon`(PANDA_HORIZON), `constraint_links`, `n_constraint_spheres`, `esdf_bounds`, `grasp_contact`(적용된 값) | |
| writer가 넣는 것 | `esdf_mode`, `max_points`, `n_chunks`, `writer: {mode: background, n_failed, failed_paths}` | 같다 |

## 단위 검증

```bash
source scripts/setup/env.sh
export CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=$AG3S_PROJECT_ROOT OMP_NUM_THREADS=2
.venv-ag3s/bin/python -m pytest -q tests/benchmark/test_panda_sl5j_recorder.py         # 15 passed (3.2 s)
.venv-ag3s/bin/python -m pytest -q tests/benchmark                                      # 163 passed, 17 skipped, 4 failed + 2 errors
#   실패는 전부 test_panda_sl3.py(openpi_client 없음). SL5j-launcher 때와 같은 6개이고, 148 → 163은 새 test 15개다
.venv-ag3s/bin/python -m pytest -q tests/trajopt/test_sl5i_grasp_recognition.py tests/trajopt/test_pb45_grasp_contact.py  # 26 passed
.venv-ag3s/bin/python -m tests.trajopt._sl5a_golden --check tests/trajopt/data/sl5a_rby1_golden.npz                     # identical, 0 differences
.venv-ag3s/bin/python -m tests.trajopt._sl5a_golden --check-serve-safe tests/trajopt/data/sl5a_serve_safe_golden.json  # identical, 0 differences
```

| test | 무엇을 고정하나 |
|---|---|
| `test_without_flag_build_matches_pre_sl5j` × {V11AB, pb5, pb7} | `git show 32a4ca8f:AG3S/adapters/panda/server.py`를 따로 불러와 새 파일과 같은 args로 `build_safe_server`를 돌린다. `SafePolicy`를 감싸 생성자 인자를 받는다(객체는 type과 `to_dict`/`record`/`config`/dataclass 필드로 비교). 두 쪽 인자의 JSON이 같고, `recorder is None`이고, `info`가 byte 단위로 같고, `record_constraints` key가 없다 |
| `test_flag_makes_a_background_writer_and_passes_it` × {full, occupancy} | writer가 `SafePolicy`에 넘어간 바로 그 객체이고, `background`, `esdf_mode`, `run_dir == DIR/run_0000`이다. SafePolicy 인자 중 다른 것은 recorder뿐이고 info에는 `record_constraints`만 더해진다. meta key와 값(`model_xml None`, `plan_horizon 10`, `urdf`, `V11AB-pb7`, conversion `C`, `grasp_contact == info`)을 확인하고, `close_recorder` 뒤 `meta.json`(writer background, n_chunks 0)을 확인한다 |
| `test_every_version_is_read_back` | AG3S_VERSIONS 8개가 모두 자기 이름으로 읽힌다. 목록에 없는 조합은 `None`이다 |
| `test_record_flag_checks` | flag가 없으면 아무 일도 없다. `--no-safe`와 모르는 backend는 거절된다. cuRobo + `occupancy`이면 경고가 나온다(writer 생성에는 cuRobo import가 필요 없다) |
| `test_main_closes_the_recorder` × {return, KeyboardInterrupt, SystemExit} × {recorder 있음, 없음} | 가짜 openpi module로 `main`을 돌린다. recorder가 있으면 `close()`가 정확히 1번 불리고, 예외는 그대로 전파되고, SIGTERM handler가 설치된다. 없으면 handler를 설치하지 않는다(예전과 같다) |
| `test_signal_stopped_process_leaves_meta_json` × {SIGINT, SIGTERM} | 실제 subprocess를 띄운다. `asyncio.run` 모양의 가짜 server와 진짜 background writer로 청크 3개를 큐에 넣고 `READY` 뒤 신호를 보낸다. `meta.json`(`n_chunks 3`, `n_failed 0`)과 chunk 3개가 있고, process는 0이 아닌 code로 끝난다 |

**mutation 확인**(한 번씩 바꿔 돌린 뒤 `cmp`로 원복을 확인했다):

| 바꾼 것 | 실패한 test |
|---|---|
| `finally`의 `close_recorder` 제거 | 5개(main × 3, signal × 2) |
| `exit_cleanly_on_sigterm` 제거 | 4개(main recording × 3, SIGTERM subprocess) |
| `SafePolicy(recorder=None)`으로 되돌림 | 2개(flag test × 2) |
| `info`에 `record_constraints` key를 항상 넣음 | 5개(identity × 3, flag × 2) |

### CPU 한 call 기록(task 6항)

스크립트: `outputs/verify/SL5j-recorder-pre/scripts/cpu_record_smoke.py`. 입력은 `ccr_replay`의 녹화 입력과 같다. dev run은
`20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0`, attention run은 `20261012T202459_309e4c6da477_sl3-attention-audit-dev`,
episode는 `libero_spatial_t0_e0`의 call 0–2다. argv는 `ccr_replay.panda_server_args("legacy")`에 `V11AB-pb7` flag와
`--record-constraints <out>/server_constraints --record-constraints-esdf <mode>`를 붙였다. legacy backend를 썼고 GPU는 쓰지 않았다.
결과는 `outputs/verify/SL5j-recorder-pre/cpu_smoke/{off,full,occupancy}/`에 있다(report.json, stdout.log, 청크).

| mode | 청크 파일 | 청크 byte | `record_failures` | close 전 pending | `meta.json` |
|---|---|---|---|---|---|
| full | 3(`chunk_00000..2`) | 324,107 / 324,196 / 319,250 | 0 | 1 | 있음(`n_chunks 3`, background, `n_failed 0`) |
| occupancy | 3 | 254,261 / … | 0 | – | 있음 |

청크 안의 배열(full, call 0):

| 배열 | shape, dtype |
|---|---|
| `clearance` | (10, 59) float32. H 10 × 구 59(pb7 `--links arms`) |
| `sphere_centres` / `sphere_radii` | (59, 3) / (59,) float32 |
| `summary.sphere_link_names` | 59개, 집합 = `panda_hand`, `panda_leftfinger`, `panda_rightfinger`, `panda_link1..7` |
| `esdf_distance` | (76, 90, 86) float16. 주 계층 1개(legacy). `esdf_layers = [main]` |
| `esdf_occupancy` | (76, 90, 86) uint8(legacy는 3상태 점유가 있다) |
| 그 밖 | `raw_cloud`, `filtered_cloud`, `attention_cloud/values/raw`, `attention_head`, `attention_left_wrist`, `target_points/indices/centroid`, `seed_indices`, `support_mask`, `reference_chunk`, `refined_chunk` |

`summary.clearance_min_mm`은 148.3 / 144.0 / 138.5다. TO status `violated`, verdict HOLD는 SL5a CPU smoke와 같은 이유다.
legacy field에서 unknown fraction이 91%를 넘어 uncertified가 된다.

**기록기는 계산을 바꾸지 않는다(CPU legacy, 3 call):** 응답 `actions`의 sha1은 off, full, occupancy 세 mode에서 call마다 같다
(`181c860a…`, `63973891…`, `0357ffe8…`).

### RB-Y1 쪽이 그대로라는 근거

| 항목 | SL5i 기록 | 이번 |
|---|---|---|
| `sl5a_rby1_golden.npz`(SafePolicy 경로) | identical | **identical, 0 differences** |
| `sl5a_serve_safe_golden.json` | identical | **identical, 0 differences** |
| 회귀 기준선 legacy(`esdf_rollout … --records $AG3S_STORAGE_ROOT/recordings/run_16d_ep1800 --constraint-links arms`, CPU) | 위반으로 시작 14/15, has_target 15/15, frame0 −29.031048280806342 mm, clearance_before sha1 `40798fb0…` | **같다**(sha1까지). `outputs/verify/SL5j-recorder-pre/checks/base_legacy.json`, 비교 스크립트는 `outputs/verify/SL5i-pre/scripts/baseline_compare.py` |

`serve_safe.py`, `safe_policy.py`, `constraint_record.py`는 바꾸지 않았다. 기준선을 잴 때 host load average가 약 29였다
(다른 pod). 그래도 SQP 이전 값 세 개는 sha1까지 같았다.

## verifier가 알아야 할 것

1. **새 flag 동작(Panda server):** `--record-constraints DIR [--record-constraints-esdf none|occupancy|full]`(기본 `full`, serve_safe parser 그대로).
   `DIR/run_NNNN/chunk_*.npz`와 종료 시 `meta.json`을 쓴다. `--no-safe`와 함께 주면 이제 시작할 때 거절한다(Panda만).
   flag가 없으면 기존과 같다.
2. **CPU에서 못 본 것, GPU smoke에서 확인해야 할 것:**
   - cuRobo backend의 계층 key(`esdf_coarse_*`, `esdf_free_*`)와 `_DeferredGrid` D2H가 실제로 쓰이는지. legacy는 주 계층 하나뿐이다.
   - `occupancy` + cuRobo에서 `esdf_occupancy_available: false`가 남는지.
   - openpi `WebsocketPolicyServer`가 SIGINT를 받았을 때 `finally`까지 가는지. 가짜 `asyncio.run` server로는 SIGINT와 SIGTERM 모두 확인했다.
     실제 server의 websockets handler가 KeyboardInterrupt를 삼키는지는 CPU에서 확인할 수 없다. 확인 방법: launcher run 뒤
     `launcher_result.record_constraints.meta_json == ["run_0000"]`이고 server log에 `[ag3s] wrote N constraint records`가 있는지 본다.
   - J0(기록이 실행을 바꾸는가)은 GPU에서 따로 잰다. CPU legacy 3 call에서 `actions` sha1이 같다는 것은 근거 하나일 뿐이다.
     cuRobo에서는 기록이 GPU→host 복사와 thread를 더하므로 wall-clock `time_budget_ms`가 끼어들 수 있다. SL5i-G에서 `time_budget_hit`는 0회였다(launcher impl 4).
3. **close 시간:** `close()`는 큐(최대 8청크)를 비울 때까지 기다린다. full 청크는 동기 기준 약 150 ms이므로 최대 수 초다.
   `stop_own_process`는 SIGINT 뒤 60 s를 기다리므로 충분하다.
4. **RB-Y1 server의 같은 누락(기록만, 고치지 않음):** `serve_safe.main` RB-Y1 분기(`serve_safe.py:2449-2463`)는 recorder가 있으면
   `exit_cleanly_on_sigterm()`만 부르고 `close()`는 부르지 않는다. 그래서 atexit flush로 청크는 쓰이지만 `meta.json`이 생기지 않는다.
   T63b 기록에 `meta.json`이 없던 이유가 이것이다. 이번 범위 밖이다.
5. **manifest:** `profile.manifest_entry()` 값은 바꾸지 않았다(주석만 추가). flag가 없으면 run manifest와 `panda_server_config.json`은 그대로다.
   flag가 있으면 `panda_server_config.json`에 `record_constraints` key가 더해진다.
6. 재생산이 필요한 산출물: 없음. 옛 기록과의 호환: 깨지지 않는다. 기록 형식은 `constraint_record.py` 그대로이고, `load_constraint_run`으로 읽힌다.

## 내가 기대하는 결과

SL5j-G(V11AB-pb7, `--record-constraints full`)에서 각 server run마다 `server_constraints/run_0000/`에 call 수만큼 청크가 생기고
`meta.json`이 하나 남는다. `ag3s_version`은 `V11AB-pb7`이다. cuRobo 청크에는 거리장 3계층이 들어 있다. 응답은 기록이 없는 run과 같다(J0).
