# SL10d — 구현 (쥔 물체 비우기 = 충돌 구)

> writer: ag3s-implementer · 읽는 쪽: verifier, scribe, lead · worktree `wip/sl10d-hfq` · 2026-11-05

## 무엇을 했나 (평이한 요약 먼저)
쥔 동안 TSDF에서 비우는 구(`held_free`)가 solver가 충돌을 재는 구와 달랐다. 새 flag `--held-free-query-spheres`(기본 꺼짐)를 켜면
비우는 구에 "solver가 이 프레임에 질의하는 부착 구"를 합집합으로 더한다. 꺼짐이면 `_held_free_points`가 예전과 같은 배열을 낸다.
판 두 개(`-pc-hfq`, `-pc-surface-hfq`)를 등록했고, 기록에 추가된 구 수와 질의 구 부피의 비움 비율을 남긴다.

## 원인 확인 (코드 위치) — lead 추정과 일치
두 정의는 **같은 `held_spheres_of()` 를 쓰지만 서로 다른 attachment 객체**에 대해 부른다.

| 쪽 | 어디서 | 무엇의 구인가 |
|---|---|---|
| 비우기 | `AG3S/ag3s/runtime/pipeline.py` `_held_free_points` (`held = self._attached`, `held_spheres_of(held)`) | attach 때 만든 `self._attached` — `attach()`가 `_lift_off_support`(pipeline.py:1759, J1)로 받침면 위로 올린 **fit 구 1개**. 첫 2 프레임은 capture 구 추가 |
| 충돌 질의 | `pipeline.py` `_constraints` (:3208) `attached = self._frame_held_attached(robot_state, surfaces, execution_path)` → `constraint_set.attached` → `AG3S/trajopt/linearize.py:2470-2487` `held_spheres_of(attached)` → solver `attached_states` | `held_body_cover` 켜짐(pc판)이면 `_frame_held_attached`(:2131)가 `self._attached` 의 primitive 를 **body_cover segment**(`_attach_body_cover`:2053)로 바꾼 사본. 프레임마다 near_support 규칙으로 lift 한 것(`query["lifted"]`) 또는 lift 없는 것(`query["body"]`) 을 고른다 |

`held_body_cover` 가 꺼져 있으면 `_frame_held_attached` 는 `self._attached` 자체를 돌려주므로 두 정의가 같다 (그래서 새 flag 는 `--held-body-cover` 없이는 아무 일도 안 하고, 서버가 거절한다).
증거: lead 데이터 `held_free` 구 z 0.1347 vs 질의 구 z 0.087~0.088 — 약 47 mm 차이 = J1 lift. 질의 구는 2 개(segment), 비우는 구는 1 개.
`AG3S/trajopt/held_record.py` 는 기록 쪽이라 원인과 무관(`attached_query_spheres` 를 `linearizer.attached_states` 로 읽을 뿐).

추정과 다르지 않아 구현을 진행했다.

## 설계
- `AG3S.held_free_query_spheres: bool = False` (pipeline.py:376). `_held_free_points(..., execution_path=None)` 가 켜졌을 때 `self._frame_held_attached(robot_state, support_surfaces, execution_path)` 를 불러 같은 프레임의 질의 구를 얻고, 같은 `robot_state` 에서 `T_parent` 로 base 로 옮겨 `spheres` 뒤에 붙인다. 호출 인자는 `_constraints` 와 같은 값이다 (`surfaces`, `execution_path`, `robot_state` 가 `_build_esdf` / `_constraints` 에서 동일). `_frame_held_attached` 는 결정적이라 `_constraints` 에서 다시 불려도 같은 결과다 (`_held_record` 의 frame 키를 같은 값으로 덮어쓸 뿐).
- 합집합이다. 기존 구는 빼지 않는다. 바닥 규칙(받침면 + pad; z_plane 은 여전히 `spheres[0]` 기준), 목적지 제외, 첫 프레임 capture 구는 그대로 아래 코드를 지난다.
- 기록: stash(`held_free_record`)에 `query_spheres_added`(개수) 를 **켰을 때만** 추가. `first` 마스크는 capture 구만 표시(질의 구는 `first=False`). 꺼짐이면 `first` 식은 예전과 같은 값이다.
- `held_record.query_freed_fraction`: 질의 구(`attached_query_spheres`, q_now) 를 9^3 격자로 샘플해 비워진 구(`held_free_spheres`) 안에 든 비율. `summary_json.held_geometry_record.attached_query.freed = {fraction, fraction_above_floor, per_sphere, per_sphere_above_floor, n_sample_points}`. `held_free.query_spheres_added` 가 있을 때만 쓴다 (꺼짐이면 SL10c 기록 키 그대로). 목적지 제외는 모델링하지 않는다. `fraction_above_floor` 는 바닥 규칙 아래 점을 뺀 분모 — 판정에서 "기록상 비움 비율" 은 이 두 값을 함께 본다 (lead 의 SL10c 측정과 같은 정의는 `fraction`).
- 서버: `serve_safe.py` flag + `ag3s_from_args` → `build_ag3s(held_free_query_spheres=)` + Panda `profile.build_panda_ag3s` (`kw["held_free_query_spheres"]`, 켜졌을 때만 `info` 키). `--no-safe/--no-perception` 에서 거절, `--esdf-backend curobo` · `--held-body-cover` 없이 거절.
- 판 (`profile.py`): `AG3S-final-panda-v3-e0to-pc-hfq`, `AG3S-final-panda-v3-e0to-pc-surface-hfq` = 기준 flag 튜플 + `--held-free-query-spheres` 하나. `PC_LATCH_VERSIONS` 에 넣어 SL10b 판처럼 `--hsr6-probe` 를 받는다. `version_flag_values` 는 새 dest 를 켰을 때만 포함 (기존 판 기록 키 불변).

## 바뀐 파일
| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/ag3s/runtime/pipeline.py:376,3815,4206,4239,4278,4294` | flag 속성, `execution_path` 전달, 질의 구 합집합, `first` 마스크, `query_spheres_added` | 비우기 = 충돌 구 |
| `AG3S/trajopt/serve_safe.py:1385,1413,1802,3057,4231,4282` | CLI flag · build 전달 · 거절 | 서버 경로 |
| `AG3S/adapters/panda/profile.py:517,599,618,939,959` | 판 2개, `version_flag_values`, `build_panda_ag3s` | Panda 서버 경로 |
| `AG3S/trajopt/held_record.py:77,122` | `query_freed_fraction` + summary | 비움 비율 기록 |
| `tests/benchmark/test_sl10d_held_free_query.py` | 13 test | 아래 |

## 단위 검증
```bash
cd .runtime/worktrees/sl10d-hfq; export AG3S_STORAGE_ROOT=/mnt/dev/AG3S_VLA-storage AG3S_PROJECT_ROOT=$PWD
MUJOCO_GL=osmesa PYTHONPATH=$PWD /mnt/dev/AG3S_VLA-benchmark/.venv-ag3s/bin/python -m pytest \
  tests/benchmark/test_sl10d_held_free_query.py tests/benchmark/test_sl10*.py tests/benchmark/test_sl9_ablation_versions.py \
  tests/benchmark/test_launcher_record_constraints.py tests/ag3s/test_t43r2_held_body_cover.py \
  tests/ag3s/test_t43t_held_free_observed.py tests/benchmark/test_panda_sl4.py tests/benchmark/test_sl6a_panda_final.py -q -p no:cacheprovider
```
첫 실행: 192 passed, 3 skipped, 2 failed. 실패 이유는 둘 다 코드가 아니다:
- `test_sl10_panda_fixed_input.py::test_tool_does_not_modify_server_adapter_or_recorder` — guard 가 **commit 하지 않은 작업 트리**의 `profile.py` / `serve_safe.py` 변경을 본다. commit 후 통과해야 한다 (commit 뒤 재실행 결과는 아래 "commit 후").
- `test_launcher_record_constraints.py::test_parallel_script_passes_the_flag_through` — worktree 에 `.venv-ag3s` 가 없어 `run_parallel_gpu.sh` 가 job line 을 `$ROOT/.venv-ag3s/bin/python` 으로 파싱하지 못한다 (예고된 venv 경로 문제).

새 test 13개 요약: 꺼짐 = 독립 재유도한 예전 집합과 bit 동일(프레임 0-3, 첫 프레임 capture 포함) · 기록 키 불변; body cover 없이 켜면 no-op; 켬(손이 받침 위 / 60 mm 위) = 예전 집합이 prefix, 질의 구 부피 비움 ≥ 0.95 (꺼짐 < 0.95, 60 mm 위), 받침면+pad 아래 0, 목적지 유지, 기록 개수·`first`; 받침면을 올려 질의 구가 아래로 나가는 경우도 0; `query_freed_fraction` 단위 · 켜짐일 때만 기록; 판 = 기준 + flag 하나, probe 처리, 서버 build 에서 `ag3s.held_free_query_spheres` 전달, `match_ag3s_version`, 기존 판 기록 키 불변; 서버 거절 3 경우.

## verifier 가 알아야 할 것
- 새로 생긴 플래그: `--held-free-query-spheres` (기본 꺼짐). 기본값 변경은 없음.
- 재생산이 필요한 산출물: 없음 (이번 판정용 closed-loop 는 lead 가 `--record-constraints full` 로 돈다).
- 옛 기록과 호환: 깨지지 않음. 꺼짐이면 기록 키 동일; 켜졌을 때만 `held_free.query_spheres_added`, `attached_query.freed` 가 생긴다.
- 켜면 비우는 점이 늘어 `n_points_freed` 와 builder 비용이 조금 늘 수 있다 (측정 안 함, GPU 금지).
- 질의 구는 q_now(robot_state) 기준이다. 계획 경로를 따라 움직이는 질의 구(`attached_query_centres`) 전체를 비우지는 않는다 (task §1 범위 밖).
- 안전 쪽 기록(쥔 물체 아닌 geom 근처 지운 voxel 수)은 J1 geom 분류 도구가 `held_free_spheres` 로 계산할 수 있다 — 이 구현은 그 도구를 건드리지 않는다.

## 내가 기대하는 결과
(verifier 는 측정이 끝나기 전에 읽지 않는다.) 기록상 `attached_query.freed.fraction` 중앙값이 0.95 이상이고, 쥠 구간 `attached_point` 최악 행 비율이 줄어든다. 줄지 않으면 voxel 이 비워졌는데도 다른 원인(예: 질의 구 반지름 + guard 가 TSDF truncation 띠에 걸림)이 남은 것이다.
