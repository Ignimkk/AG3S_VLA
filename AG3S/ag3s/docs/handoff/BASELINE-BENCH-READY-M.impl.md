# BASELINE-BENCH-READY-M — 구현 (privileged 변형 + benchmark job 생성 · 집계)

> writer: ag3s-implementer (M) · 읽는 쪽: verifier V6, lead, scribe · 작성 2026-10-25 (UTC 17:40, KST 2026-10-26 02:40)
> checkout `/mnt/dev/AG3S_VLA-benchmark`, branch `feature/panda-benchmark`, HEAD `3412861c` + working tree. commit 하지 않았다.
> GPU 를 쓰지 않았다. CPU worker 는 1 개다. task: `BASELINE-BENCH-READY.task.md` 의 "Implementer M" 절 (순서 2 · 3).
> 같은 시간에 L 이 `methods/**` 를 고쳤다. 그 파일은 건드리지 않았다.

## 무엇을 했나 (평이한 요약 먼저)

**privileged 변형.** client 에 flag `--emit-privileged-obstacles` 하나를 더했다. 기본은 off 이다.
- flag 가 켜지면 request 에 SafeLIBERO 장애물 ground truth 가 실린다. 물체 이름, active 여부, geom 별 type · 크기 · pose 이다.
  `--capture-segmentation` 이 함께 켜져 있으면 GT segmentation 도 실린다.
- flag 가 꺼져 있으면 request, `_calls.jsonl`, `_calls.npz` 가 수정 전 client 와 byte 단위로 같다.
  확인은 두 가지로 했다. fake env 를 쓴 py3.8 harness, 그리고 실제 SafeLIBERO CPU episode 이다.
- server 는 그 key 를 base frame 으로 바꿔 `SceneObservation.extras["privileged/..."]` 로 넘긴다.
  key 이름은 methods 가 offline 에서 이미 읽던 이름과 같다. 그래서 methods adapter 는 바꿀 필요가 없다 (L 의 계약과 같다).
- `--gt` arm 7 개를 등록했다. manifest 에는 `privileged: true`, privileged 입력 목록, 필요한 client flag 가 남는다.
- 실제 SafeLIBERO episode (CPU, recorded policy) 에서 `constrained_flow_matching:gt-spheres` 와 `knows:gt-seg` 를 끝까지 돌렸다.
  둘 다 56 call 에서 fallback 0 이고, 56 call 모두 active obstacle 로 `moka_pot_obstacle_1` 을 받았다.

**benchmark job.** `closed_loop/jobs.py` 를 새로 만들었다.
- arm × suite × level × task × split 로 `run_parallel_gpu.sh` 입력 (`jobs.txt`, `plan.json`) 을 만든다.
  대상은 LIBERO 4 suite 와 SafeLIBERO 4 suite × L I / L II 이다.
- split label, 선택 검사, suite 목록은 `suite_registry` 와 `eval_matrix.parse_suites` 를 그대로 쓴다.
- 줄마다 `launch_method` 의 parser 와 `check_selection` 을 통과시킨다.
- `aggregate` 는 suite × level × arm 마다 공식 지표, Wilson 구간, 같은 plan 의 `none` 과의 paired 비교를 낸다.
- `none` 은 plan 마다 다시 돌린다 (benchmark session 권고).
- LIBERO-Safety 는 붙일 자리 (`FAMILY_HOOKS`) 만 두었다.

**lead 추가 요청.** 모든 arm 의 `extras["suite"]` 를 server 가 넣는다 (`--suite`). 그래서 AEGIS 의 공식 workspace box 가 켜진다.
V5 동작은 `--no-suite-extras` 로 되돌린다. 이것과 L 의 `crop_mode="box"` · `action_box=False` 를 묶어 `<arm>@v5` preset 으로 만들었다.

## 구조

```
libero_safe_client.py --emit-privileged-obstacles [--capture-segmentation]      (.venv-libero, 기본 off)
  request += ag3s/privileged/frame = "world"
             ag3s/privileged/obstacles = [{name, active, pos, quat_wxyz, geoms:[{id,name,type,size,pos,rot(9),collision,group}]}]
             ag3s/privileged/monitor_obstacle
             ag3s/privileged/segmentation_geom/<cam>  (int16, OpenCV rows, depth 와 같은 frame; seg 일 때만)
        │  ag3s/* 는 모든 server 가 policy 앞에서 제거한다 (PandaVLAOnly, wire.strip_request) → policy 입력 불변
        v
serve_method --gt [SOURCE] --suite <suite>
  MethodServer._scene_obs : wire_obs.scene_observation + extras["suite"] (+ --gt: privileged.extras(request, T_world_base))
  filter   (sidecar)  names → ObstacleCloudSource(detector, detector_prompt="obstacle_names")
                      seg   → ObstacleCloudSource(gt_seg)
  guidance (server)   build_scene(obstacle_source="privileged")  ← privileged/obstacle_geoms (base frame)
                      target 는 OmniGuide (attractor) 만 detector, 나머지는 none (depth lift 없음)
launch_method --gt : client argv = lp.client_argv(...) + arm["client_flags"]; run tag <arm>-gt-<source>
jobs.py plan|smoke|aggregate (.venv-ag3s, CPU)
```

### privileged key (wire → extras)

| extras key | 내용 | 읽는 곳 |
|---|---|---|
| `privileged/obstacles` | wire 목록. pose 는 base frame (`inv(panda/T_world_base)`) 으로 바꾼다 | 기록 |
| `privileged/obstacle_names` | active obstacle 의 표시 이름. `moka_pot_obstacle_1` 은 `moka_pot_1` 이 되고, `phrase_from_object` 로 prompt `"moka pot"` 가 된다 | `ObstacleCloudSource(detector, "obstacle_names")` |
| `privileged/obstacle_geoms` | active obstacle 의 primitive (`Obstacles.from_primitives` 형식, base frame) | `guidance_common.scene.build_scene` |
| `privileged/segmentation_geom` `{cam: (H,W)}`, `privileged/instances_geom_ids` `{표시 이름: [geom ids]}` | instance 는 **active SafeLIBERO obstacle 만** 넣는다 | `ObstacleCloudSource(gt_seg)` |
| `privileged/target`, `privileged/destination` = `None`, `privileged/fixture_names` = `[]` | instance 에 장애물만 있다. 그래서 제외할 것이 없다 | `gt_seg` 의 `excluded_names` |

- L 의 계약에서는 `privileged/target` 이 BDDL target 이다. 그 계약은 장면 전체를 instance 로 둘 때의 것이다.
  여기서는 instance 가 SafeLIBERO obstacle 뿐이므로 target 이 들어가지 않는다. 결과는 같다. target 이 장애물이 되지 않는다.
- active 판정은 `CollisionMonitor` 규칙과 같다 (root body z > 0, |x|, |y| < 0.5).
  나머지 obstacle 은 SafeLIBERO 가 장면 밖 ±10 · 20 m 에 둔다. 실제 env 에서 확인했다.

### `--gt` source

| source | 기본 arm | 장애물 | client flag | 원 논문 · repo |
|---|---|---|---|---|
| `names` | `aegis` | GDINO prompt = GT obstacle 이름. depth crop 은 non-privileged arm 과 같다 (L 의 `crop_mode`) | `--emit-privileged-obstacles` | AEGIS: VLM 이 이름을 준다 |
| `seg` | `knows`, `multilink_cbf` (`aegis` 도 가능) | GT segmentation 의 active obstacle 점 | + `--capture-segmentation` | — |
| `spheres` | `constrained_flow_matching` | active obstacle 마다 collision geom 을 감싸는 구 1 개 | `--emit-privileged-obstacles` | CFM 논문 Sec. 5 "GT sphere" |
| `repo` | (CFM 선택) | 이름에 `obstacle` 이 든 geom 전부. mesh 는 `geom_size` box 로 근사 | 같음 | CFM repo `live_viewer.GroundTruthObstacleTracker` / `_serialize_obstacles` |
| `geoms` | `nudge`, `omniguide`, `lan_o3dp_guidance_adapted` | collision geom 을 primitive 로 (SafeLIBERO 물체는 box 들) | 같음 | — |

변형 이름은 `<arm>:gt-<source>` 이다. lead 예시와의 대응은 `aegis:gt_obstacles` = `aegis:gt-names`, `cfm:gt_spheres` = `constrained_flow_matching:gt-spheres` 이다.
`deviation_from_paper` 에는 `PRIVILEGED: …` 한 줄과 남는 차이가 들어간다. 남는 차이는 chunk 단위 filter, CPU callback, Lan 의 policy 차이 등이다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/runners/libero_safe_client.py:151-192` | `PRIVILEGED_PREFIX`, `_obstacle_tables` (joint 이름에 `obstacle` 이 든 물체 → root body, geom 목록), `privileged_obstacles` (env.sim 에서 읽기만 한다) | client 가 GT 를 읽는 곳 |
| `…libero_safe_client.py:207-212, 233-234` | `SafeClient(..., emit_privileged=False)`. observer 는 flag 가 켜졌을 때만 env 참조를 남긴다 | |
| `…libero_safe_client.py:263-278` | `_request` 끝: flag 가 켜졌을 때만 `_add_privileged` 를 부른다 | |
| `…libero_safe_client.py:384-386, 412-416, 433-434` | `--emit-privileged-obstacles` (store_true). on 이면 SafeLIBERO 만 허용하고 jsonl 에 `privileged_obstacles` 기록을 남긴다. `policy_for` 가 flag 를 넘긴다 | **client 수정은 이 flag 하나다** |
| `closed_loop/privileged.py` (신규) | `extras` · `obstacles_base` · `obstacle_geoms(shape ∈ collision/repo/sphere)` · `summary` · `display_name` | wire → extras |
| `closed_loop/arms.py:30-60, 101-134, 136-141, 170-182` | `GT_SOURCES` · `GT_DEFAULT` · `GT_DESCRIPTION` · `resolve_gt` · `resolve(..., gt=)` · `arm_spec` (`<arm>[:gt[-src]]`) | arm 기록 |
| `closed_loop/serve_method.py:157-167, 260-272` | `MethodServer(suite=)`. `_scene_obs` 에 `extras["suite"]` 를 넣고, `--gt` 이면 `privileged.extras` 와 `row["privileged"]` 를 더한다 | |
| `…serve_method.py:336-356` | guidance `--gt`: `build_scene(obstacle_source="privileged")`. target 은 OmniGuide 만 detector 이고, 나머지는 `target_points = (0,3)` 이다 (depth lift 없음) | L 의 지적 (`"points"` 고정) |
| `…serve_method.py:452-460` | filter `--gt`: names → `detector_prompt="obstacle_names"`, seg → `obstacle_source="gt_seg"` | |
| `…serve_method.py:502-505` | `--gt [SOURCE]`, `--suite` | |
| `closed_loop/launch_method.py:91-111, 126-128, 170, 224-228` | server argv 에 `--suite` (`--no-suite-extras` 이면 없음) 와 `--gt` 를 넣는다. `resolve_arm`, `method_client_argv` (non-gt 이면 `lp.client_argv` 그대로), run tag `-gt-<src>`, manifest `server_extras` | |
| `closed_loop/jobs.py` (신규) | `plan` · `smoke` · `aggregate`, `PRESETS["v5"]`, `FAMILY_HOOKS`, V5 pilot 기반 시간 모델 | benchmark job |
| `closed_loop/README.md` | privileged 변형 · jobs 절 | |
| `closed_loop/tests/client_harness.py` (신규, py3.8) | fake env · geometry · websocket 으로 `SafeClient` 를 구동하고 digest 를 낸다. digest 는 key 별 msgpack sha256, `_calls.jsonl` sha256, `_calls.npz` 배열 sha256 이다 | byte 동일성 |
| `closed_loop/tests/test_client_privileged.py` (신규, 13) · `tests/test_jobs.py` (신규, 6) | 아래 | |

수정하지 않은 파일: `launch_libero_paired.py`, `run_parallel_gpu.sh`, `eval_matrix.py`, `suites.json`, `suite_registry.py`, `methods/**`, `AG3S/**`, `third_party/**`. 앞의 두 파일은 benchmark session 이 고칠 예정이다. 나는 import 와 호출만 한다.

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/third_party/openpi/src \
  .venv-openpi-live/bin/python -m pytest -q -p no:cacheprovider -W ignore::DeprecationWarning \
  Baseline/paper_baselines/closed_loop/tests --ignore=Baseline/paper_baselines/closed_loop/tests/test_metrics_launcher.py \
  --ignore=Baseline/paper_baselines/closed_loop/tests/test_jobs.py
# -> 46 passed in 106 s   (기존 33 + 신규 test_client_privileged 13)
PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider \
  Baseline/paper_baselines/closed_loop/tests/test_jobs.py Baseline/paper_baselines/closed_loop/tests/test_metrics_launcher.py
# -> 12 passed
PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest -q tests/benchmark/test_safelibero_client.py \
  tests/benchmark/test_bm_setup_a_identity.py tests/benchmark/test_eval_matrix.py tests/benchmark/test_suite_registry.py \
  tests/benchmark/test_libero_safety_client.py tests/benchmark/test_panda_sl5a.py
# -> 67 passed, 3 skipped, 2 failed (아래 "알아야 할 것" 1: argparse 에 생긴 새 key 하나 때문이다)
```

| test | 무엇을 보나 |
|---|---|
| `test_client_privileged::test_flag_off_is_byte_identical_to_the_pre_change_client[vla_only-False / vla_v11ab-False / vla_v11ab-True]` | **off 동일성.** `3412861c` 의 client 와 worktree client 를 **py3.8** (`.venv-libero`) 에서 같은 fake 입력으로 구동한다. 6 request 의 key 순서가 같고, key 별 msgpack bytes 와 request 전체 bytes 가 같다. `save()` 기록, `_calls.jsonl` bytes, `_calls.npz` 의 모든 배열 (dtype · shape · bytes) 이 같다. SL5c argv 를 parse 한 결과도 같다. 다른 것은 `emit_privileged_obstacles=False` 하나뿐이다. V11AB mode (feedback 경로) 와 segmentation frame 이 있는 경우도 포함했다 |
| `::test_flag_on_only_adds_privileged_keys` | on 이면 기존 key 의 순서와 bytes 가 그대로다. 더해지는 것은 `ag3s/privileged/{frame, obstacles, monitor_obstacle, segmentation_geom/head, …/left_wrist}` 뿐이다. client 기록 파일은 off 와 같다 |
| `::test_privileged_keys_round_trip_to_the_extras` | client msgpack → `privileged.extras` 를 거친다. active 이름 · instance geom id · collision box 의 크기와 base frame pose 가 fake env 의 truth 와 같다 (≤1e-12). repo 규칙에는 visual mesh 가 더해진다. sphere 는 box 꼭짓점을 모두 감싼다. sidecar framing 왕복은 bit 동일이다. key 가 없으면 `KeyError("…--emit-privileged-obstacles")` 를 낸다 |
| `::test_policy_input_is_unchanged_by_the_privileged_keys` | `PandaVLAOnly` 뒤의 policy 입력 key 집합이 privileged key 유무와 상관없이 같다 |
| `::test_methods_read_the_privileged_extras` | `build_scene(obstacle_source="privileged")` 가 `privileged/obstacle_geoms` 를 읽고, 그 key 가 `privileged_inputs` 에 남는다. `ObstacleCloudSource("gt_seg")` 의 cloud 는 `moka_pot_1` 하나다. bowl geom 은 instance 가 아니다 |
| `::test_none_arm_serves_the_same_chunk_with_privileged_keys` | SL5c 녹화 request 에 privileged key 를 더해도 `none` 은 녹화 chunk 와 bytes 가 같다 |
| `::test_gt_guidance_arm_uses_the_privileged_obstacles[nudge / cfm / cfm-repo]` | 3 경우 모두 fallback 0, `guided` true 다. 장애물 출처는 `privileged/obstacle_geoms` 이고 sidecar 는 없다. active 는 moka 1 개, primitive 수는 1 / 1 / 2 다 |
| `::test_gt_arm_without_client_flag_falls_back_and_says_why` | flag 없이 `--gt` server 를 띄우면 그 call 은 unguided 로 fallback 한다. error 에 원인이 남는다 |
| `::test_gt_arm_records` | 7 arm 기본 source, `privileged: true`, 입력 목록, client flag (seg 만 segmentation), 잘못된 조합 거부, non-gt 는 모두 `privileged: false` |
| `test_jobs::test_every_in_scope_suite_and_level_and_the_split_rules` | 12 suite × level 이 나온다. 나머지 확인: dev episode = registry `[0..4]`, test = `[5..49]`. `--data-split` = `split_label`. SafeLIBERO 에만 `--safety-level` 이 붙는다. 모든 줄이 `launch_method` parser · `check_selection` 을 통과한다. run 이름은 겹치지 않는다. `--tasks-per-job` 분할 |
| `::test_selection_errors_follow_the_registry` | split 밖 episode, task 범위 밖, out-of-scope `libero_90`, `libero_safety` (NotImplementedError) 를 거부한다 |
| `::test_privileged_arms_only_on_safelibero_and_their_client_flags` | `:gt` 는 LIBERO 에서 skip 된다 (3 arm × 4 suite). gt client argv 에 flag 가 붙는다. `none` 의 client argv 는 `lp.client_argv` 와 같다 |
| `::test_none_is_always_rerun_and_presets` | `none` 이 없는 plan 은 거부한다. `@v5` preset 의 method kwargs 와 `--no-suite-extras` 가 맞게 들어간다. 기본 arm 의 server argv 에는 `--suite` 가 있다 |
| `::test_jobs_txt_is_run_parallel_gpu_input` | `jobs.txt` 줄을 `shlex.split` 하면 plan 의 argv 와 같다. 실행 명령은 `AG3S_PARALLEL_TEST_LAUNCHER` 이다 |
| `::test_aggregate_per_suite_level_arm_and_paired_vs_none` | 합성 run 에서 TSR · CAR · SS 와 `none` 대비 paired 표가 `eval_matrix.paired_metrics` 와 같다. 없는 run 은 `missing` 에 들어간다 |

### 실제 SafeLIBERO CPU episode (`--cpu-smoke --fake-policy recorded`, SL5c t0e0 chunk, spatial L I)

| run (`AG3S_STORAGE_ROOT/results/benchmark/`) | 무엇을 보나 | 결과 |
|---|---|---|
| `20261025T174524_3412861ca839_blbr-cpu-flagoff-LI-none` vs 수정 전 client 의 `20261024T152140_3c6775531841_clm-cpu-smoke-recorded-LI-none` | 실제 env 에서 flag off 동일성 | episode npz 14 / 14, `_calls.npz` 3 / 3, `_geom.npz` 33 / 33 배열이 bit 동일하다. `_calls.jsonl` 56 행은 timing key 를 빼면 같다. `safelibero` 기록 (충돌 t = 35, 0.25595 m) 도 같다. jsonl 에서 다른 것은 시각 · latency 와 `server_metadata` 뿐이다. `server_metadata` 차이는 K 의 fake policy 가 `kind` · `source_key` 를 더한 것이고, client 와 관계없다 |
| `20261025T174829_…_blbr-cpu-gt-LI-constrained_flow_matching-gt-spheres` | 실제 env 의 privileged 경로 (guidance) | 56 call, fallback 0. manifest `privileged: true`. 56 call 모두 active = `moka_pot_obstacle_1` 이다 (6 obstacle 물체 중 1 개, monitor 와 같다). base 중심 (0.608, 0.020, 0.094), scene = sphere 1 개, 개입 call 9 개, h_min −0.033 m |
| `20261025T175307_…_blbr-cpu-gt-LI-knows-gt-seg` | 실제 env 의 privileged 경로 (filter, sidecar, GT seg) | 56 call, fallback 0. `filter_perception = {source: gt_seg, privileged: true, n_points: {moka_pot_1: 4000}}`. ellipsoid c (0.609, 0.009, 0.105), 반축 (0.074, 0.129, 0.155). jsonl 에 `privileged_obstacles.segmentation: true` 가 남는다 |

recorded policy 는 open-loop chunk 다. 그래서 위 성공 · 충돌 수치는 연결 확인용이다. 성능 비교에 쓰지 않는다.

## verifier 가 알아야 할 것

1. **기존 test 2 개가 새 argparse key 때문에 실패한다.** 내 수정 범위 밖이라 고치지 않았다.
   - 실패하는 test: `tests/benchmark/test_bm_setup_a_identity.py::test_client_wiring_identical[libero_spatial, safelibero_spatial_II]`.
   - 원인: 이 test 는 `vars(args)` 를 비교하고 `safelibero_step_budget` 만 뺀다. 이제 `emit_privileged_obstacles` key 도 생긴다.
   - 나머지가 같음은 확인했다. scratchpad script 로 그 key 를 빼면 두 case 모두 `rest identical: True` 다.
   - 고칠 곳은 `_CLIENT_PROBE` 의 `k != "safelibero_step_budget"` 한 줄이다. `k not in ("safelibero_step_budget", "emit_privileged_obstacles")` 로 바꾼다. owner (lead / benchmark session) 가 고친다.
2. **VLA-only reuse.** `eval_matrix.VLA_ONLY_CODE_PATHS` 에 이 client 가 들어 있다. 그래서 옛 VLA-only run 을 `eval_matrix plan --vla-only reuse` 로 쓰려면 `--accept-diff benchmark/runners/libero_safe_client.py` 가 필요하다. 근거는 위 off 동일성 test 이다. jobs.py 는 reuse 를 하지 않고 `none` 을 매번 다시 돈다.
3. **새 flag · 기본값**
   - client `--emit-privileged-obstacles` (off)
   - server `--gt [SOURCE]` · `--suite` (기본 None)
   - launcher `--gt [SOURCE]` · `--no-suite-extras` (off)
   - **기본값 변화:** launcher 가 이제 server 에 `--suite <suite>` 를 넘긴다. 그래서 AEGIS 의 공식 workspace box 가 켜진다 (V5 는 `workspace=null`). L 의 `depth_trim` · CFM `action_box` 기본값 변화와 함께 V5 대비 aegis 결과가 바뀔 수 있다. V5 재현은 `<arm>@v5` (`--no-suite-extras --tag v5 --method-kwargs …`) 로 한다.
   - `none` arm 은 scene 을 만들지 않는다. 그래서 `--suite` 의 영향이 없다.
4. **재생산이 필요한 산출물:** 없다.
5. **옛 기록과의 호환:** V5 의 run 이름 · 지표 key 는 그대로다. `--gt` run 은 tag `-gt-<src>` 가 붙어 이름이 따로 생긴다. calls.jsonl 에 `privileged` key 가 더해진다 (gt 만).
6. **`run_parallel_gpu.sh` launcher.** 아직 `--launcher` 가 없다. 그래서 `AG3S_PARALLEL_TEST_LAUNCHER` override 를 쓴다 (`plan.json["meta"]["run"]`). benchmark session 이 `--launcher` 를 넣으면 그 명령으로 바꾼다.
7. **시간 모델.** `jobs.TIMING` 은 V5 pilot 의 job wall time 에서 server 시작 45 s 를 빼고 20 ep × 300 step 으로 나눈 값이다. 평균값이고 상한이 아니다. 다른 suite 와 `:gt` arm 은 재지 않았다. `seg` 는 카메라마다 render 를 1 번 더 한다.
8. **GPU 에서 처음 보는 것:** `aegis:gt-names` (GDINO 를 obstacle 이름으로 prompt). CPU 에서는 GDINO 비용 때문에 episode 로 돌리지 않았다. 대신 단위 test 와 같은 `detector_prompt="obstacle_names"` 경로를 쓴다 (methods 의 offline 경로). `multilink_cbf:gt-seg` · `omniguide:gt-geoms` · `lan:gt-geoms` 도 episode 로는 돌리지 않았다. 대신 단위 test (nudge / cfm 로 경로 공통) 로 확인했다.

## V6 명령

공통 준비 (GPU 규칙은 이전과 같다. 다른 session 에 알리고, compute process 가 없는지 확인한다):

```bash
cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
export OPENPI_DATA_HOME=$PWD/.cache/openpi XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer=
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader   # 비어 있어야 한다
L="$PWD/.venv-ag3s/bin/python $PWD/Baseline/paper_baselines/closed_loop/launch_method.py"
O=outputs/verify/BASELINE-BENCH-READY
```

1. **CPU 재확인** (위 "단위 검증" 의 세 명령).

2. **privileged 변형 smoke** — SafeLIBERO spatial L I, t0e0, 7 `:gt` arm + `none`, 각 1 ep. 예상 약 25 min (2 병렬).
   ```bash
   .venv-ag3s/bin/python -m Baseline.paper_baselines.closed_loop.jobs plan \
     --arms none,aegis:gt,knows:gt,multilink_cbf:gt,constrained_flow_matching:gt,constrained_flow_matching:gt-repo,nudge:gt,omniguide:gt,lan_o3dp_guidance_adapted:gt \
     --suites safelibero_spatial:I --split dev --tasks 0 --episodes 0 --run-prefix blbr-gt-smoke --step BASELINE-BENCH-READY-V6 \
     --out-dir $O/gt_smoke
   AG3S_PARALLEL_TEST_LAUNCHER="$L" benchmark/runners/run_parallel_gpu.sh --jobs $O/gt_smoke/jobs.txt --max-parallel 2 --est-gib-per-run 16
   .venv-ag3s/bin/python -m Baseline.paper_baselines.closed_loop.jobs aggregate --plan $O/gt_smoke/plan.json
   ```
   볼 것:
   - 각 run 의 `method_calls/calls.jsonl`: `fallback` 이 모두 null 인가, `privileged.active` 가 monitor obstacle 과 같은가.
   - guidance: `perception.scene.obstacle_source == "privileged/obstacle_geoms"` 인가.
   - filter: `filter_perception.privileged == true` 인가.
   - manifest: `settings.arm.privileged == true` 인가.
   - privileged vs non-privileged 장면 figure 의 입력: 같은 call 의 `privileged.centers` · `filter_obstacles` 를 non-gt run 과 비교한다.

3. **suite × level smoke** — 12 조합 × (`none`, `aegis`, `constrained_flow_matching`) t0e0. 36 job, 예상 약 1.1 h (2 병렬, V5 모델). 대표 arm 을 바꾸려면 `--arms` 를 쓴다.
   ```bash
   .venv-ag3s/bin/python -m Baseline.paper_baselines.closed_loop.jobs smoke --out-dir $O/suite_smoke
   AG3S_PARALLEL_TEST_LAUNCHER="$L" benchmark/runners/run_parallel_gpu.sh --jobs $O/suite_smoke/jobs.txt --max-parallel 2 --est-gib-per-run 16
   .venv-ag3s/bin/python -m Baseline.paper_baselines.closed_loop.jobs aggregate --plan $O/suite_smoke/plan.json   # aggregate.json / .md = smoke 표
   ```
   - LIBERO suite 는 `launch_method` 로 처음 돈다. client 는 같은 `libero_safe_client.py` 이고, SafeLIBERO 가 아니므로 충돌 지표 없이 SR 만 나온다.
   - 실패가 나오면 suite 별로 원인 (env · detector · sampler) 을 적는다.

4. **(선택) V5 재현 확인** — L 의 기본값 변화와 `extras["suite"]` 를 분리해서 본다.
   ```bash
   .venv-ag3s/bin/python -m Baseline.paper_baselines.closed_loop.jobs plan --arms none,aegis,aegis@v5,constrained_flow_matching,constrained_flow_matching@v5 \
     --suites safelibero_spatial:I --split dev --tasks 0 --episodes 0 --run-prefix blbr-v5chk --out-dir $O/v5_check
   ```

5. **전체 benchmark 의 형식 (실행은 하지 않는다. benchmark 단계에서 한다).**
   ```bash
   .venv-ag3s/bin/python -m Baseline.paper_baselines.closed_loop.jobs plan --arms none,aegis,knows,multilink_cbf,constrained_flow_matching,nudge,omniguide,lan_o3dp_guidance_adapted,aegis:gt,constrained_flow_matching:gt \
     --suites all --split test --tasks-per-job 2 --run-prefix blb-test --out-dir outputs/eval/baselines-test
   ```

## 확인하지 못한 것

- GPU 에서의 모든 것. 특히 `aegis:gt-names` 의 GDINO prompt 결과와 privileged arm 의 실제 성공률 · 충돌률이다.
- LIBERO 4 suite 를 `launch_method` 로 돌리는 것. CPU episode 는 SafeLIBERO spatial 로만 돌렸다.
- LIBERO-Safety 는 hook 자리만 두었다.
- CFM 논문의 "GT sphere" 반경 규칙은 논문에 식이 없다. 그래서 collision geom 을 감싸는 구로 정했다. repo 규칙은 `gt-repo` 로 따로 두었다.

## 내가 기대하는 결과

(verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.)

- privileged 변형은 non-privileged 보다 CAR 이 높을 것이다. 특히 filter (`seg`) 와 CFM (`spheres`) 이 그렇다. 장애물이 정확히 하나이고 배경 · target 이 섞이지 않기 때문이다.
- `aegis:gt-names` 는 GDINO 가 "moka pot" 같은 이름을 head 256 px 에서 놓치는 call 이 있을 것이다. 그래서 개선 폭은 seg 보다 작을 것이다.
- CFM `gt-spheres` 는 구가 커서 (moka pot 반경 약 0.13 m) 개입이 크다. TSR 은 depth cloud 판보다 높고 VLA-only 보다는 낮을 것이다.
