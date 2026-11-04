# SL10c — 구현: 서버 제약 기록에 쥔 물체 기하와 지지면 평면을 더한다 (기록 전용)

> writer: ag3s-implementer · 읽는 쪽: lead, verifier, scribe · 2026-11-05 (KST) · worktree `wip/sl10c-rec`, base `a66efd3e`

## 무엇을 했나 (평이한 요약 먼저)

SL10 J1 은 "쥔 동안 서버가 HOLD 하는 주 원인이 쥔 bowl 의 부착 구(`attached:panda_hand[i]`, why=`attached_point`)가 bowl 자신의 점유 voxel 과
겹치는 것" 이라고 읽었다. 그런데 서버는 `held_free` 로 쥔 구의 부피를 비우고 있다. **비운 쪽이 틀린지, 질의한 쪽이 틀린지** 가르려면
call 마다 (1) 비운 구, (2) 질의에 쓴 구, (3) 쥔 구의 정의(parent frame), (4) 지지면 평면이 필요한데 기록에 없었다.

이제 `--record-constraints-esdf full` 일 때만 chunk npz 에 위 값들이 키로 더해진다. `none` / `occupancy` 에서는 키가 하나도 없고, 서버의 응답 ·
궤적 · 상태는 바뀌지 않는다 (기록 전용; 아래 "행동 불변"). 재풀이 도구(`panda_fixed_input_ablation.py`)는 새 키가 있으면 지지면 RANSAC 재맞춤 대신
기록된 평면을, 부착 구와 `held_capture_spheres` 도 기록값을 쓴다 (키가 없으면 이전 경로 그대로).

## 새 chunk npz 키 (모두 `esdf_mode == "full"` 에서만)

frame: **base** = 로봇 base frame (서버의 제약 · TSDF frame), **parent** = 쥔 구가 붙은 link frame. 단위는 모두 m (offset 도 m). 구는 `[cx, cy, cz, r]`.
dtype 은 float64 / bool / 0-d str. 정의의 코드: `AG3S/ag3s/experiments/sources/constraint_record.py` `HELD_RECORD_KEYS` · `AG3S/trajopt/held_record.py`.

| 키 | shape | frame | 무엇 | 있을 때 |
|---|---|---|---|---|
| `support_planes` | `(P, 4)` `[nx, ny, nz, offset]` | base | AG3S RANSAC 지지면 (`constraint_set.support_surfaces`), `n·p = offset`, 단위 법선(자유쪽). TrajOpt 가 `support_planes_of` 로 읽는 것과 같은 값 | **언제나** (`(0, 4)` = 서버에 평면이 없었다, 키 없음 = 기록 안 함) |
| `held_free_spheres` | `(N, 4)` | base | 그 프레임의 `_held_free_points` 가 TSDF 에서 비운 구: 앞 `n_held_spheres` 개는 `robot_state`(= 그 프레임 q_now)의 쥔 구(`T_parent @ local`), 뒤는 `_held_first_free` 의 구 (잡은 순간 자리 · 잡기 전 놓였던 자리 + guard pad) | 쥔 프레임에서 `_held_free_points` 가 돌았을 때 |
| `held_free_first_frame` | `(N,)` bool | — | `held_free_spheres[i]` 가 `_held_first_free` 에서 왔나. 쥔 뒤 처음 두 프레임에만 True 가 있다 | 위와 같음 |
| `held_local_spheres` | `(K, 4)` | **parent** | solver 가 본 쥔 구의 정의 = `SceneSnapshot.attached_points`(중심) + `attached_radii`. 반지름이 없는 (점만 있는) 부착은 r = 0 | 쥔 call |
| `attached_parent_link` | `()` str | — | `held_local_spheres` 가 붙은 link (Panda: `panda_hand`). 질의 이름은 `attached:<link>[i]`, i = `held_local_spheres` 의 행 | 쥔 call |
| `attached_query_spheres` | `(K, 4)` | base | solver 의 linearizer 가 쥔 구를 놓은 자리 **q_now 에서** (`held_free_spheres` 와 같은 상태라 겹쳐 볼 수 있다) + 질의 반지름 | 쥔 call, solver 의 `n_attached` 가 scene 과 같을 때 |
| `attached_query_centres` | `(H, K, 3)` | base | 같은 구의 중심을 **refined 계획**을 따라 (H = 계획 지평 = `clearance` 의 행 수). `max_violation_pair` 의 `step` · `query - S` 로 색인 | 위와 같음 |
| `held_capture_spheres` | `(C, 4)` | base | `AG3S.held_capture_spheres` — 쥐는 순간 물체가 있던 자리 (`_held_first_free`). 쥔 동안 계속 있다. TO 가 "물체 자기 잔상" 을 가르는 데 쓴다 | 쥔 call 이고 attach 가 armed 일 때 (prebuilt geometry 면 없음) |
| `finger_plan_opening` | `(n_hands, H)` | — | TO 창이 가정한 정규화 개도 (`SafePolicy._last_finger["plan"]["opening_norm"]`: 측정 · 명령 포락선). Panda 는 `n_hands = 1` | 손가락 경로가 걸렸을 때 |
| `finger_plan_hands` | `(n_hands,)` str | — | 위 행의 손 이름 (`left`) | 위와 같음 |

`summary_json["held_geometry_record"]` (같은 조건): `n_support_planes` · `held_free` (키 위 배열 외의 스칼라: `z_min_m` 지지면 바닥 + guard pad, `support_plane_z_m`,
`guard_pad_m`, `tsdf_voxel_m`, `sample_step_m`, `first_frames_left_before`, `n_held_spheres`, `n_sphere_points_after_floor`, `n_observed_points`, `observed_pad_m`,
`n_points_before_destination`, `destination_exclusion_m`, `n_points_freed`) · `attached` (`parent_link`, `n`, `radii_given`) · `attached_query` (`available`, 못 쟀으면 `why_not`).
오류가 나면 `{"error": …}` 만 쓰고 chunk 기록은 그대로 남는다.

J1 에서 가를 수 있는 것: `held_free_spheres` ∩ voxel (비운 부피) vs `attached_query_spheres[_centres]` ∩ voxel (질의). 비운 구의 `first` 가 False 인 행과 질의 구가 같은 parent
정의에서 나왔는지는 `held_local_spheres` 를 base 로 옮긴 것과 `held_free_spheres[:n_held_spheres]` 의 일치로 본다 (같은 q_now 라면 같아야 한다). `z_min_m` 아래의 구 부피는
비워지지 않는다 (지지면 규칙).

## 바뀐 파일

| 파일 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/ag3s/runtime/pipeline.py` | `record_held_free` (기본 False) · `_held_free_stash` · `held_free_record` property; `_held_free_points` 가 켜졌을 때 비운 구와 개수를 복사해 stash; `_build_esdf` 첫 줄에서 stash 비움 (켜졌을 때만) | (1) 비운 구 |
| `AG3S/trajopt/held_record.py` (신규) | `build_held_record(...)` — 위 키를 만드는 읽기 전용 함수 | (2)(3)(4) 와 개도 · capture |
| `AG3S/trajopt/safe_policy.py` | `full` recorder 가 있으면 `ag3s.record_held_free = True`; `_record` 가 `_held_record()` 로 만든 것을 `recorder.record(held_record=)` 로 넘김 (None 이면 인자 자체를 안 넘김) | 연결 |
| `AG3S/ag3s/experiments/sources/constraint_record.py` | `HELD_RECORD_KEYS`; `record(held_record=)` — `full` 에서만 키를 쓰고 정의 밖 키는 `KeyError` | 기록 |
| `AG3S/trajopt/experiments/panda_fixed_input_ablation.py` | `support_surfaces_recorded` · `attached_spec_from_record` · `held_capture_from_record`; `build_scene` 이 기록값을 우선 (지지면: 기록 → 없으면 RANSAC 재맞춤; 부착 구: `--attached-spheres-json` → 기록 → 없으면 `missing`; capture: 기록 → None) | 도구 연결 |
| `tests/benchmark/test_sl10c_record_held.py` (신규) | 19 개 | (a)(b)(c) |

## 행동 불변 (왜 bit 동일인가)

* 서버 flag 는 새로 없다. `full` recorder 가 없으면 `record_held_free` 는 False 이고 `_held_free_points` 는 `if self.record_held_free:` 한 줄을 더 평가할 뿐이다.
  켜져도 반환값(`pts`)은 그대로이고 stash 는 복사본이다 — `test_held_free_points_are_identical_with_the_record_on_or_off` (4 프레임, 목적지 제거 포함, 반환 배열과 `_held_first_free_frames` 가 같음).
* `_record` 는 응답 · 판정 **뒤**에 돈다 (`infer` 의 `_verdict` 다음). 새 계산은 solver linearizer 의 `attached_states` (순수 FK) 뿐이고, 거기서 만들어지는 probe 캐시는
  solver 가 첫 solve 에서 이미 만든 것이다. `test_building_the_record_does_not_change_what_the_solver_returns`: 같은 두 solver 중 하나만 solve 사이에 `build_held_record` 를 부르게 해서
  두 번째 solve 의 chunk · trajectory · max_violation 이 `np.array_equal` 로 같음.
* 기록 실패는 `held_geometry_record.error` 로만 남고 chunk 기록과 정책을 죽이지 않는다 (`test_a_failing_held_record_does_not_cost_the_chunk_its_record`).
* writer: `none` / `occupancy` 에서 `held_record` 를 줘도 npz 의 모든 배열 · summary 가 안 준 것과 바이트 단위로 같다; `full` 에서 안 주면 이전과 같다
  (`test_new_keys_exist_only_in_full_mode`).
* 기존 `full` 기록에는 키가 늘어난다 (기록 크기: 쥔 call 에서 수 KB, 지지면만 있는 call 에서 수십 byte).

## 도구가 새 키를 읽는 방법

* 지지면: `support_planes` 가 있으면 `SimpleNamespace(normal, offset)` 로 그대로 `scene_from_constraint_set` 에 (`scene_info.support_surfaces.source == "record support_planes"`). 없으면 `support_surfaces_from_record` (RANSAC 근사).
* 부착 구: `--attached-spheres-json` 이 있으면 그것(명시한 값이 이김), 없으면 `held_local_spheres` + `attached_parent_link` (`scene_info.attached.source == "record held_local_spheres"`), 둘 다 없으면 이전처럼 `missing`.
* `held_capture_spheres` 가 있으면 `scene_from_constraint_set(held_capture=)` 로 (그 함수가 서버의 `_held_capture_for_scene` 와 같은 조건으로만 쓴다).
* `finger_plan_opening` 은 기록만 하고 도구는 아직 쓰지 않는다 (도구는 명령에서 다시 계산하고 SL10 의 B 충실도에서 일치하는 것을 이미 확인했다). 쓰려면 `finger_state_for` 에서 `fmap.path` 의 openings 와 비교하면 된다.

## 단위 검증

```bash
cd <worktree>; export AG3S_STORAGE_ROOT=/mnt/dev/AG3S_VLA-storage AG3S_PROJECT_ROOT=$PWD
MUJOCO_GL=osmesa PYTHONPATH=$PWD /mnt/dev/AG3S_VLA-benchmark/.venv-ag3s/bin/python -m pytest \
    tests/benchmark/test_sl10c_record_held.py tests/benchmark/test_sl10_*.py tests/benchmark/test_sl10b_*.py \
    tests/benchmark/test_sl9_ablation_versions.py -q -p no:cacheprovider
```

결과는 최종 보고(lead 에게 간 답)에 적었다. 기록기 · 서버 주변 회귀(`test_panda_sl5j_recorder.py` · `test_record_and_hold_state.py` · `test_exec_feedback.py` · `test_t43t_held_free_observed.py` ·
`test_grasp_evidence.py` · `test_safe_policy.py` · `o4/test_record_background.py`)도 통과. 모듈 `__file__` 은 worktree. CPU 단일 process.

주의: `test_sl10_panda_fixed_input.py::test_tool_does_not_modify_server_adapter_or_recorder` 는 "도구 파일을 건드린 commit 이 보호 경로(`safe_policy.py` · `constraint_record.py` …)를 바꾸면 실패" 한다.
그래서 이 변경은 **두 commit 으로 나눴다**: 기록기 쪽(pipeline · safe_policy · constraint_record · held_record) 먼저, 도구 · test · 이 문서를 그 다음. 작업 트리에 미커밋 보호 경로 변경이 있는 동안에는 이 guard 가 실패한다(정상).

## verifier 가 알아야 할 것
- 새로 생긴 플래그·기본값 변경: 없음. 서버 argv 는 그대로다. `--record-constraints-esdf full` 의 npz 에 키가 늘었다 (아래 "옛 기록과 호환").
- 재생산이 필요한 산출물(npz 등): 새 키는 **새로 돌린 기록 run 에만** 있다. 기존 run(`…sl10-rec-…`)에는 없다 → 도구는 이전 경로로 돈다. 쥠 구간 J2/J1 재분석을 하려면 `full` 기록 run 을 다시 돌려야 한다 (GPU, 이 worktree 에서는 안 했다).
- 옛 기록과 호환이 깨지는가: 없음 (읽는 쪽은 키가 없으면 이전 경로; `load_constraint_run` 은 키를 가리지 않는다).
- 실제 서버 run 으로는 아직 돌려 보지 않았다 (GPU 금지). 합성 held 상태와 RBY1 URDF 로 시험했다. 첫 실제 run 에서 먼저 볼 것: `attached_query` 가 `available:true` 인가(solver 의 linearizer 가 scene 과 같은 n_attached 인지),
  `held_free_spheres[:n_held_spheres]` 가 `held_local_spheres` 를 q_now 에서 base 로 옮긴 것과 같은가, `z_min_m` 이 `support_planes` 의 테이블과 맞는가.
- 알려진 한계: `held_free_spheres` 는 `_held_free_points` 가 불린 프레임에서만 있다 (curobo backend ∧ `free_held_traces` ∧ 쥔 구가 있는 attach). 이 경로를 안 타는 판에서는 `held_free: null`.
  `held_free_observed_pad` (T43 T fix 2) 의 관측점 부피는 구가 아니라 점이라 개수(`n_observed_points`)만 남긴다.

## 내가 기대하는 결과
(기대는 측정 전에 적지 않는다. 합성 입력에서 키의 모양 · frame · 불변만 확인했다.)
