# SL3 — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 작성 2026-10-12 16:40 KST
> branch `feature/panda-benchmark`, 시작 HEAD `ee57ad31` (작업 중 scribe 가 `5a5a5f5f` 를 commit — 이 파일들과 겹치지 않음).
> commit 안 함. `git mv` 한 cover rename 1 건만 index 에 staged 상태다 (`AG3S_BENCHMARK_LOG.md` 는 건드리지 않았다).

## 무엇을 했나 (평이한 요약 먼저)

**SL2 후속 2건.** (1) Panda self-filter 기본 덮개를 **confined hand + mount** 로 바꿨다. 새 JSON
`libero_panda_self_filter_cover_confined_hand_mount.json`(1036 capsule → 4210 sphere, self-filter model 4271)은 repo 의
`cover.py --variant confined_hand_mount` 로 다시 만든 것이고 SL2 verifier 의 `raw/cover_confine_hand_mount.json` 과 capsule 이 완전히 같다.
옛 기본 JSON 은 내용을 바꾸지 않고 `..._unconfined.json` 으로 이름만 바꿨다 (SHA-256 `895b7dbf…` 그대로). manifest 의 `robot_model.self_filter_cover`
에 cover 이름·capsule 수·sphere 수·digest 가 들어간다. (2) `robot_geometry.py` CLI 가 기본 shell(env.sh)에서 동작한다 — 진입점이 launcher 와 **같은
함수**(`client_env.client_environment`, launcher 가 이제 이 함수에 위임)로 환경을 고쳐 자기 자신을 한 번 re-exec 한다.

**Panda attention binding.** `AG3S/adapters/panda/attention.py` 가 V11AB 의 `AttentionSampler`·`AGGREGATIONS`·cell 상수(step 0, `last`, L8, H2)를
**import 해서 그대로** 쓴다. pi05_libero token block 표(`base_0_rgb→head`, `left_wrist_0_rgb→left_wrist`, `right_wrist_0_rgb→padding`)는 가정으로
두지 않고 model input 에서 **읽어** 대조한다(`read_token_layout`: `embed_prefix` 순서, `PaliGemma.img` token 수, image mask, V11AB prefix pass 의
실제 prefix mask). 16×16 grid → OpenCV 256 pixel 대응은 "열 뒤집기(un-mirror) + V11AB 자신의 pixel-centre resampling" 이고, 반전을 풀지 않은 비교용
mapping 도 같은 함수에 `unmirror=False` 로 있다. **B:** LIBERO BDDL 에서 target(`goal` 의 첫 인자 ∈ `obj_of_interest`)을 읽고 segmentation geom id 와
잇는 `task_target.py`, recorded policy request 로 V11AB read-out 을 다시 계산해 **새 run** 에 저장하는 `benchmark/runners/attention_audit.py`.

GPU smoke(SL0 run 사본, 2 call): token layout 문제 0, V11AB `AttentionPolicy` 와 **bit 동일**(max |Δ| 0.0), padding block mass 0.0, JIT 후 67 ms.
AG3S core(`AG3S/ag3s/**`, `AG3S/trajopt/**`), OpenPI model 코드, submodule, storage input 은 한 줄도 바꾸지 않았다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `robots/panda/description/libero_panda_self_filter_cover_confined_hand_mount.json` (새, 생성물) | 기본 덮개 1036 capsule. provenance: `cover_name`, `capsule_digest` `0b6c68ea…`, `geometry_digest` `ea38872f…`(SL2 export 와 같음), `confined_bodies` = mount 4 + `gripper0_right_gripper`, fitter 상수, URDF·franka.yml SHA-256. file SHA-256 `b8ec2deb…` | 0.1 |
| `robots/panda/description/libero_panda_self_filter_cover_unconfined.json` (rename, 내용 불변) | SL2 기본 51 capsule | 0.1 비교용 보존 |
| `robots/panda/description/README.md:8-31` | 자산 표 2 variant, 재생성 명령(`--variant`), CLI 가 기본 shell 에서 됨 | 0.1, 0.2 |
| `AG3S/adapters/panda/cover.py:20-31, 54-66` | 문서; `HAND_BODY`, `COVER_VARIANTS = {confined_hand_mount: MOUNT_BODIES+(hand,), unconfined: ()}`, `DEFAULT_COVER`, `CONFINED_BODIES` 기본 = confined_hand_mount | 0.1 |
| `cover.py:254-259, 285-300` | `capsule_digest` (JSON 으로 쓰는 capsule 행의 SHA-256), `variant_of`, provenance 에 `cover_name`·`capsule_digest` | 0.1 digest |
| `cover.py:322-345` | CLI `--variant {confined_hand_mount,unconfined}` (기본 confined), `--confine BODY…|mount|none` 은 custom | 0.1 |
| `AG3S/adapters/panda/model.py:19-23, 59-66` | `PANDA_SELF_FILTER_COVERS`, `DEFAULT_SELF_FILTER_COVER="confined_hand_mount"`, `PANDA_SELF_FILTER_COVER` = 그 경로 | 0.1 builder 기본값 |
| `model.py:177-191, 232-245` | `self_filter_cover_path(name|path)`; `load_self_filter_cover`·`build_panda_self_filter_model` 이 variant 이름도 받음, `model.panda_cover_path` | 0.1 |
| `model.py:316-357` | `cover_manifest_entry` (name, path, sha256, count, confined_bodies, digests, `cover_spheres`, `self_filter_spheres`); `manifest_entry()["self_filter_cover"]` = 기본 variant, `["self_filter_cover_alternatives"]` = unconfined path·sha256 | 0.1 manifest |
| `benchmark/adapters/libero_panda/client_env.py` (새, py3.8, stdlib) | `sanitize_library_path`, `client_environment` (launcher 의 SL0 규칙을 옮김), `needs_reexec`, `reexec_with_client_environment` (sentinel `AG3S_LIBERO_CLIENT_ENV=1`, 1 회) | 0.2 |
| `benchmark/runners/launch_libero_pi05_only.py:33, 45-48, 219-233` | `CLIENT_THREADS`/`THREAD_LIMIT_KEYS`/`sanitize_library_path`/`client_environment` 가 `client_env` 에 위임 (이름·동작 동일) | 0.2 규칙 하나 |
| `benchmark/adapters/libero_panda/robot_geometry.py:23-25, 39-46` | `__main__` 일 때 numpy import 전에 `client_env.reexec_with_client_environment()` | 0.2 |
| `AG3S/adapters/panda/attention.py` (새) | 아래 "attention.py" 표 | A.3–A.5 |
| `benchmark/adapters/libero_panda/task_target.py` (새, py3.8) | BDDL parser(LIBERO `scan_tokens` 규칙: `;` 주석 제거, 소문자, 괄호 tree), `goal_roles`, `suite_task_names`(LIBERO `libero_task_map`·`task_orders[0]` 를 AST 로 읽음), `task_target`, `instance_body_ids`/`instance_geom_ids`/`instance_mask`/`scene_instances` | B.6 |
| `benchmark/runners/attention_audit.py` (새, `.venv-openpi-live`) | offline audit runner (아래) | B.7 |
| `tests/benchmark/test_panda_sl3.py` (새, 16 test), `tests/benchmark/sl3_tiny_pi05.py` (새, subprocess helper) | 아래 표 | B.8 |
| `tests/benchmark/libero_py38_checks.py:273-307` | 새 check `bddl_target_matches_libero_parser` | B.8 |
| `tests/benchmark/test_libero_sl1.py:203` | 위 check 등록 | |
| `tests/benchmark/test_panda_sl2.py:104-200, 328-341` | cover test 3 개를 variant 별로 parametrize; 덮개 포함 검사를 KD-tree 로(232 s → 18 s); manifest test 가 이름·1036·4210·4271; py3.8 문법 검사에 새 py3.8 파일 2 개 | 0.1 |

### attention.py

| 이름 | 하는 일 |
|---|---|
| `PI05_LIBERO_TOKEN_BLOCKS` | `TokenBlock(model_key, tokens, wire_camera, robosuite_camera, image_mask, v11ab_policy_camera)` × 3: `base_0_rgb (0,256) head agentview True cam_high` / `left_wrist_0_rgb (256,512) left_wrist robot0_eye_in_hand True cam_left_wrist` / `right_wrist_0_rgb (512,768) – – False cam_right_wrist` |
| `v11ab_cell()` | `attention_policy._DENOISE_STEP/_AGG/_LAYER/_HEAD` 를 읽어 돌려줌 (값을 다시 적지 않음) |
| `read_token_layout(model, obs, sampler=)` | 전처리된 observation 의 `obs.images` 순서(= `embed_prefix` 순서), 각 image 의 `PaliGemma.img` token 수(`jax.eval_shape`), `image_masks`, 그리고 V11AB `AttentionSampler._jit_prefix` 가 실제로 쓴 prefix mask 를 block 별로(`prefix_mask_all/any`) |
| `check_token_layout(layout)` | 위 표 + V11AB `CAMERA_BINDINGS` slice 위치와 대조, 차이 목록 (빈 list = 일치) |
| `resize_with_pad_geometry`, `check_square_no_padding` | openpi_client `_resize_with_pad_pil` 산술 재서술; 256→224 square 는 pad 0, scale 224/256 (test 가 실제 함수와 대조) |
| `unmirror_grid(g, unmirror=)` | `g[..., ::-1]` (False = 그대로, 비교용) |
| `patch_centre_uv(r, c, convention="index"|"edge", unmirror=)` | policy patch 중심의 OpenCV pixel. index 규약(adapter K `pixel_centre`) = `(16(15−c)+7.5, 16r+7.5)`; edge = V11AB `patch_to_uv(r, 15−c, 256, 256)` = index + 0.5 |
| `policy_patch_of_uv`, `policy_pixel_to_opencv_uv`, `patch_footprint` | 역방향 / policy pixel → OpenCV pixel / patch 가 덮는 16×16 OpenCV block |
| `attention_pixel_map(g, out_hw, unmirror=)` | `GridAttentionAdapter("bilinear").to_pixel_map(unmirror_grid(g), out_hw)` — V11AB `lift` 가 depth 에 쓰는 바로 그 resampling |
| `opencv_mask_to_policy_patches(mask, unmirror=)` | OpenCV mask 의 policy patch 별 덮인 비율 (resampling 없이 token-order grid 를 채점할 때) |
| `mapping_conventions()` | 위 규약의 JSON (run manifest `settings.mapping`) |
| `PandaAttentionReader(policy).read(request)` | V11AB `AttentionPolicy._attention_and_feature` 와 같은 단계: `policy._input_transform` → `Observation` → `sampler.attention_and_subtask_feature(obs, [0], noise_seed=0)` → `AGGREGATIONS["last"]` → `[8, 2]`. 반환: `cell_tokens (768,)`, `grids {head, left_wrist}` (policy 방향), `pooled (L,H,768)`, `feature (512,)`, `seconds` |
| `v11ab_attention_policy_grids(policy, request)` | V11AB `AttentionPolicy(policy, policy._model, subtask_probe=None)._attention` 자체 — 위 reader 의 기준 |

### mapping 정의 (평이하게)

client 는 `policy = resize_with_pad(raw[::-1, ::-1], 224)`, adapter 는 `opencv = raw[::-1]` 이다 (SL1: `policy256[i, j] = opencv[i, W−1−j]`).
pixel `i` 가 `[i, i+1)` 를 차지하는 연속 좌표에서: 256→224 는 pad 없는 균일 scale(PIL 은 centre-aligned) 이므로 `x_m = x_p·256/224`, 반전 해제는
`x_o = 256 − x_m`, 행은 그대로. 그래서 policy patch `(r, c)` (policy pixel `[14c, 14c+14)`) 는 **정확히** OpenCV 행 `[16r, 16r+16)`, 열
`[16(15−c), 16(16−c))` 이다 — un-mirror 는 `grid[:, ::-1]` 이고 224→256 scale 은 grid resampling 이 흡수한다. 반전 미해제 mapping 은 같은 grid 를
뒤집지 않고 쓰는 것.

### BDDL target 파싱

`task_target(suite, task_id)`: `suite_task_names` (LIBERO `libero_suite_task_map.py` 의 `libero_task_map[suite]` 를 `benchmark/__init__.py` 의
`task_orders[0]` 순서로 — 둘 다 AST 로 읽음) → `bddl_files/<suite>/<name>.bddl` → `parse_bddl` → `goal_roles`: goal predicate 중 첫 인자가
`obj_of_interest` 에 있는 첫 것; `On/In` 이면 (target, destination) = (arg1, arg2), unary(`Open` 등)면 (arg1, None). LIBERO Spatial 10 task 모두
`target = akita_black_bowl_1`, `destination = plate_1`, `obj_of_interest = [akita_black_bowl_1, plate_1]`, goal `[[on, akita_black_bowl_1, plate_1]]`.
segmentation 연결: object `X` = body `X_main` 과 `X_<part>` (`startswith(X + "_")` — `bowl_1` 이 `bowl_10` 을 잡지 않음), fixture `main_table` = body
`table`; geom id 는 `_geom.npz` 의 `model_geom_bodyid`/`model_body_names` 로. py3.8 check 가 LIBERO `robosuite_parse_problem`·
`get_task_bddl_file_path`·`get_task(i).name` 과 10 task 모두 같음, live env 의 `obj_of_interest`, 각 object `root_body == X_main` 을 확인.

### attention_audit.py (새 run 의 구조)

- source run 은 읽기만. `policy_records/step_<n>.npy` 는 server 가 받은 n 번째 request 다 (server 하나가 모든 episode 를 받는다) →
  `episodes.jsonl` 순서대로 episode 마다 `policy_calls` 개씩. 이 join 을 call 마다 **검사**: record state == episode `policy_state[k]`, prompt ==
  `task_description`, 두 image == client 전처리(`resize_with_pad(raw[::-1, ::-1], 224)`)를 `obs_index = chunk_t[k]` 의 raw 관측에 다시 한 결과 (bit
  동일). 하나라도 다르면 멈춘다. record 수 ≠ policy_calls 합이면 시작하지 않는다.
- target·segmentation instance 는 GPU 작업 전에 계산 (BDDL 연결이 안 되는 task 가 있으면 먼저 실패).
- `create_run` manifest `settings`: `step: SL3`, `attention_sources` (run id, path, commit, manifest·episodes.jsonl SHA-256, record 수, episode 목록),
  checkpoint identity (source run 의 checkpoint aggregate 와 다르면 거부), `v11ab_cell`, `token_blocks`, `mapping`, `attention`(num_denoise 10,
  noise_seed 0, subtask_feature on, full_grid on), interpreter·환경.
- 출력: `attention/<suite>_t<task>_e<ep>_attn.npz` (`cell_tokens (N,768) f32`, `cell_grid_policy`/`cell_grid_opencv (N,2,16,16) f32` [head, left_wrist],
  `block_mass_cell (N,3)`, `block_mass_all (N,L,H,3) f32`, `pooled_f16 (N,L,H,768)`, `kv_feature (N,512)`, `call_k`, `global_step`, `chunk_t`,
  `obs_index`, `record_sha256`, `seconds`), `attention_index.jsonl` (episode 행: source 경로·SHA, status, join 결과, `target` + `instances_geom_ids`,
  V11AB 대조), `token_layout.json`, `summary.json` (latency 첫 call 따로 + 나머지 p50/p95, `jax` device memory, V11AB 대조).
- smoke 크기: call 당 ~108 KB (압축). LIBERO Spatial 10 episode ≈ 20–40 call/episode 면 ~30 MB.

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
OMP_NUM_THREADS=4 .venv-ag3s/bin/python scripts/setup/run_without_legacy.py -m pytest -q -p no:cacheprovider -rs tests/benchmark/test_panda_sl3.py
# 15 passed, 1 skipped (tiny pi05 test: jax/flax 없음)
OMP_NUM_THREADS=4 .venv-openpi-live/bin/python scripts/setup/run_without_legacy.py -m pytest -q -p no:cacheprovider -rs tests/benchmark/test_panda_sl3.py
# 16 passed in 38 s (tiny pi05 subprocess 35 s, JAX_PLATFORMS=cpu)
OMP_NUM_THREADS=4 .venv-ag3s/bin/python scripts/setup/run_without_legacy.py -m pytest -q -p no:cacheprovider -rs tests/benchmark/test_panda_sl2.py
# 16 passed in ~62 s (confined refit 31 s, 덮개 포함 검사 18 s)
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 bash scripts/testing/run_cpu_regression.sh
# == A: 73 passed, 77 warnings in 41.46s                       (SL2 와 같은 73)
# == B: tests/benchmark (.venv-openpi-live): 78 passed in 165.81s  (SL2 58 + SL2 parametrize 3 + py38 check 1 + SL3 16)
# == C: tests/benchmark (.venv-ag3s): 76 passed, 2 skipped in 115.96s (SL2 57 + 3 + 1 + SL3 15; skip = serve_policy(flax 없음), tiny pi05(jax 없음))
# == cpu regression exit status: 0
```

| test (`test_panda_sl3.py`) | 무엇을 |
|---|---|
| `test_token_blocks_match_v11ab_bindings_and_the_camera_adapter` | block 위치 = V11AB `CAMERA_BINDINGS`, used block = adapter `CAMERA_TABLE`, `check_token_layout` 이 순서 뒤바뀜·mask·padding 누설을 잡음 |
| `test_v11ab_cell_is_read_from_attention_policy` | (0, last, 8, 2) |
| `test_resize_with_pad_geometry_is_openpi_clients` | 재서술 산술 = 실제 `image_tools.resize_with_pad` (4 크기, pad 있는 4:3 포함), 256→224 pad 0 |
| `test_unmirror_and_scale_invert_the_client_preprocessing` ×3 | OpenCV blob → raw → **실제 client 전처리** → policy centroid → `policy_pixel_to_opencv_uv` 가 원 위치를 0.06 px 안에서 복원; `unmirror=False` 는 `255−cx` 로 감; 그 patch 의 one-hot grid 를 `attention_pixel_map` 한 peak 가 footprint 안 |
| `test_patch_centres_footprints_and_masks_are_one_convention` | 256 patch 전부: index/edge 중심 = `patch_to_uv(r, 15−c)` (−0.5), footprint·역방향·mask→patch one-hot |
| `test_attention_pixel_map_is_v11ab_resampling_of_the_unmirrored_grid` | = `GridAttentionAdapter.to_pixel_map(g[:, ::-1])` bit 동일, = map 좌우반전 (1e-6), `unmirror=False` = V11AB 그대로 |
| `test_bddl_target_of_every_libero_spatial_task`, `test_bddl_parser_rules`, `test_segmentation_instances_by_body_name` | 10 task target, 주석·대소문자·typed list·unary goal·오류, `bowl_1`≠`bowl_10`, alias, mask |
| `test_launcher_and_client_env_are_one_rule` | launcher 함수 = `client_env` 함수, re-exec 판정, sentinel |
| `test_robot_geometry_cli_runs_from_the_env_sh_shell` | 외래 torch lib + checkout PYTHONPATH 환경에서 `fk --source libero` 성공; sentinel 로 re-exec 를 막으면 SL2 의 `libtorch_python.so` ImportError 가 재현됨 |
| `test_audit_runner_indexes_and_joins_policy_records`, `test_audit_runner_dry_run_settings` | 가짜 2-episode source run(임시 storage root): step→episode 배정, bit 동일 join, 다음 call frame 으로 join 하면 실패, record 수 불일치 거부, results 밖 경로 거부, dry-run settings |
| `test_attention_reader_on_a_tiny_pi05_libero_model` | 실제 `pi05_libero` Pi0 class + LIBERO input/model transform (norm stats 없음), test 안에서만 SigLIP `So400m/14→mu/14`, gemma dummy depth 10: layout 읽기 문제 0, V11AB `AttentionPolicy` 와 3 block bit 동일, padding block mass 0 (모든 L×H), feature 유무 무관 |

`libero_py38_checks.py` 새 check `bddl_target_matches_libero_parser`: ok (8 개 check 전부 ok).

**GPU smoke** (`AG3S_STORAGE_ROOT` = scratchpad 사본, SL0 run 의 `run_manifest.json`·`episodes.jsonl`·`episodes/`·`policy_records/` 만 복사; 실제 results 에는
쓰지 않음; `--limit-calls 2 --check-v11ab 2`):

| 항목 | 값 |
|---|---|
| model load | 56.2 s, device bytes_in_use 6.90 GB |
| token layout | `base 0–256 mask T / left_wrist 256–512 T / right_wrist 512–768 F`, prefix 968 (text 200, 유효 20), problems `[]` |
| V11AB `AttentionPolicy` 대조 | 2/2 call bit 동일 (max |Δ| 0.0) |
| padding block (512–768) mass | cell 0.0, 18×8 전체 max 0.0 |
| latency | 첫 call 5.25 s (JIT), 두 번째 66.9 ms; peak 7.14 GB |
| join | 2 call 모두 state·prompt·image·wrist bit 동일 (`chunk_t` 10, 15) |

target mass 등 #4–#6 숫자는 재지 않았다 (verifier 몫).

## verifier 가 알아야 할 것

- **실행 명령** (각 1 process. GPU 는 rollout server 1 개, 그 뒤 audit 1 개 — 동시에 돌리지 않는다):
  ```bash
  cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
  # 1. dev rollout: libero_spatial task 0–9 × init state 0 (episode_idx = init_state_index), geometry + segmentation, policy_records 포함
  .venv-ag3s/bin/python benchmark/runners/launch_libero_pi05_only.py --task-ids 0,1,2,3,4,5,6,7,8,9 --episode-ids 0 \
      --capture-geometry --capture-segmentation --run-id sl3-dev-pi05only-t0-9e0
  # 2. offline attention audit on that run (새 run 이 생김; source 는 읽기만)
  DEV=<1 이 출력한 run dir 이름>
  .venv-openpi-live/bin/python benchmark/runners/attention_audit.py --source-run $DEV --run-id sl3-attention-audit-dev
  #    (선택) 기존 SL0 episode 도 같이: --source-run 20261012T045129_6ba81024d243_sl0-pi05only-t0e0 를 한 번 더 (repeatable)
  # 3. SL2 후속 #1: 기본 shell 에서 export (re-exec 확인) + 기본 cover 로 SL2 self-filter 측정 재실행
  .venv-libero/bin/python benchmark/adapters/libero_panda/robot_geometry.py export --source libero --out outputs/verify/SL3/raw/geo_libero.npz
  bash scripts/testing/run_cpu_regression.sh
  ```
  dry-run 확인: 1 은 `--dry-run` 으로 manifest `geometry_capture.robot_model.self_filter_cover.name = confined_hand_mount` (1036 / 4210 / 4271),
  2 는 `--dry-run` 으로 call 수·settings.
- **새 플래그·기본값 변경**:
  - self-filter 기본 덮개 = `confined_hand_mount` (`model.DEFAULT_SELF_FILTER_COVER`, `build_panda_self_filter_model()` 무인자, `panda_depth_robot_mask`
    에 넘기는 model). 옛 덮개는 `build_panda_self_filter_model("unconfined")`.
  - `cover.py`: `--variant` (기본 `confined_hand_mount`); 무인자 `fit_self_filter_cover(...)`/`provenance_of(...)` 의 `confined` 기본도 confined 로 바뀜.
    `--confine` 는 custom 용, `none` 허용.
  - `robot_geometry.py` CLI: 환경이 client 환경과 다르면 한 번 re-exec (`AG3S_LIBERO_CLIENT_ENV=1`). 이전의 `env -u PYTHONPATH …` 명령도 그대로 된다.
  - launch_libero_pi05_only 는 동작 변경 없음 (함수 위임만). manifest `geometry_capture.robot_model` 에 `self_filter_cover.{name, cover_spheres,
    self_filter_spheres, confined_bodies, capsule_digest}`, `self_filter_cover_alternatives` 추가; `settings.step` 은 geometry on 이면 여전히 `"SL1"`.
  - `attention_audit.py` 기본: `--num-denoise 10 --noise-seed 0` (V11AB 와 같음), subtask feature 계산 on (V11AB server 가 probe 를 켜고 도는 경로 =
    `attention_and_subtask_feature`; 끄려면 `--no-feature`), full grid 저장 on (`--no-full-grid`), `--check-v11ab 2` (처음 2 call 을 V11AB
    `AttentionPolicy` 로 한 번 더 계산해 대조 — 그 2 call 은 latency 에 포함되지 않지만 pass 가 한 번 더 돈다).
- **덮개의 알려진 기하 사실 (0.1 측정 대상)**: confined 규칙은 hand·mount capsule 을 그 body 의 **collision** 표면 상자 안에 가두므로 collision 표면은 전부
  담지만, **visual** mesh 는 바깥으로 나온다 (4 mm 표본, 정확한 최대값): hand 0.880 mm, `mount0_torso` 34.448 mm, `mount0_pedestal` 93.215 mm.
  unconfined 는 visual 도 전부 담았다(SL2). SL2 verify 의 confined_hand_mount mount 제거율 0.987(unconfined 0.9993)과 같은 원인으로 보인다 — 측정은
  verifier. test 는 이 세 값을 상한으로 고정해 둔다 (`test_panda_sl2.py:VISUAL_EXCESS_M`).
- **재생산이 필요한 산출물**: 없음 (SL2 의 cover variant JSON 은 repo 기본값과 capsule 이 같다: `raw/cover_confine_hand_mount.json` == 새 기본 JSON 의
  `capsules`, `per_body`). self-filter model 의 sphere 수가 245 → 4271 이 되므로 self-filter 를 쓰는 분석은 느려진다.
- **옛 기록과 호환**: `PANDA_SELF_FILTER_COVER` 경로·내용이 바뀌었다 (옛 파일명 `libero_panda_self_filter_cover.json` 은 더 없음). SL2 run 의 manifest 는
  옛 이름을 가리킨다 — 그 내용은 `..._unconfined.json` 과 byte 동일. episode/geom npz 형식 변경 없음.
- **scoring 때 주의 (구현이 아니라 데이터 사실)**:
  - policy call `k` 의 frame 은 `obs_index = chunk_t[k]` 이고, 그 index 의 `_geom.npz` 행이 같은 관측이다 (`attention_index.jsonl` 의 `joins`).
  - segmentation 은 `env.step` return 시점 render (SL1 차이 #4, SL2 #1 에서 image 시점과 head 8686 px / 114 frame 차이). image 시점 mask 가 필요하면
    `sim_state_image` 로 다시 render.
  - `cell_grid_opencv` 는 `cell_grid_policy[..., ::-1]`. pixel map 은 `attention.attention_pixel_map(grid_policy, (256,256), unmirror=True|False)`,
    resampling 없는 채점은 `opencv_mask_to_policy_patches(mask, unmirror=…)` 와 token-order grid 의 곱. V11AB core 의 `policy_record.patch_coverage`
    (KNOWS Eq.2) 를 OpenCV label map 에 쓰면 OpenCV 방향 grid 와 짝이 된다.
  - pooled 배열은 float16 (V11AB `pi05_attention.py` `--store-dtype` 설명: 합한 뒤 ~0.2 % 상대오차). V11AB cell 은 `cell_tokens` float32 에 따로 있다.
  - phase(접근/grasp/이동)는 기록에 label 이 없다. `policy_state[:, 6:8]` (gripper qpos)·`action_chunks[..., 6]` (gripper 명령) 이 episode npz 에 있다.
    subtask probe(`kv_L4_v1`)는 RB-Y1 feature 로 학습된 것이라 Panda 에서 phase 로 쓰지 않았다 (SL-6). `kv_feature` 는 저장해 두었다.
- **자원**: GPU smoke 는 server 없이 1 process, 7.1 GB peak, 70 s. 다른 session process 는 건드리지 않았다. CPU test 는 `OMP_NUM_THREADS=4`.

### SL-4 가 필요로 할 AG3S core 주입 지점 (구현하지 않음)

| 위치 | RB-Y1 고정인 것 | Panda 에 필요한 것 |
|---|---|---|
| `AG3S/trajopt/serve_safe.py:1037-1058` `attention_extractor` (`ALIAS`) | `cam_high→zed_left`, `cam_left_wrist→wrist_cam_l`, `cam_right_wrist→wrist_cam_r` | `cam_high→head`, `cam_left_wrist→left_wrist`, `cam_right_wrist` 버림 (padding, mass 0) — 그리고 **grid 열 뒤집기** (`attention.unmirror_grid`) 를 `lift` 전에. core `GridAttentionAdapter`/`lift` 에는 flip 옵션이 없으므로 extractor(또는 wire adapter) 에서 해야 한다 |
| `AG3S/trajopt/attention_policy.py:126-129` | `POLICY_CAMERA_NAMES` 3 개 모두 dict 로 냄 | 그대로 써도 됨 (block 위치 동일, pi05_libero 확인). padding grid 를 extractor 가 버리면 됨 |
| `AG3S/ag3s/experiments/sources/pi05_attention.py:142` (`n_image_tokens = 16·16·len(CAMERA_BINDINGS)`) | 768 | pi05_libero 도 3 block = 768 → 변경 불필요 (test·smoke 확인) |
| `AG3S/ag3s/stages/subtask_probe.py` (`kv_L4_v1`) | RB-Y1 feature 로 학습된 probe | SL-6 audit (Panda `kv_feature` 는 이 run 에 있음) |
| `serve_safe.py` attention model 로드 (`load_attention_model(args.config, args.checkpoint)`) | config 이름 인자 | `pi05_libero` 로 그대로 동작 (smoke) |

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다 -->
- #1: confined 기본 덮개의 제거율·오제거는 SL2 verify `variants.confine_hand_mount` 와 정확히 같다 (같은 capsule). robot_geometry CLI 는 기본 shell 에서
  exit 0.
- #2: 성공률은 공식 pi05_libero 보고(Spatial ~98 %) 근처, 10 episode 중 실패 0–1. call 수는 episode 길이/5 (≈ 15–40).
- #3: smoke 와 같은 layout; padding block mass 는 모든 call·모든 L×H 에서 0.
- #4: un-mirror mapping 이 target mass 비율을 더 높게 줄 것 — 단, head 에서는 bowl 이 화면 중앙 근처라 차이가 작을 수 있고, wrist 에서 더 크게 날 것.
- #5: V11AB cell (L8, H2) 이 pi05_libero 에서 target 에 집중하는지는 모르겠다. RB-Y1 fine-tune 과 LIBERO fine-tune 은 다른 head 를 쓸 가능성이 있어,
  head 카메라에서 target mass 비율이 uniform(area 비율) 의 1–3 배 정도, argmax-in-target 은 접근 단계에서 낮고 grasp 근처에서 높아질 것으로 본다.
  wrist 는 접근 후반에 target 이 화면을 크게 차지하므로 area 비율 자체가 커서 지표 해석에 주의.
- #6: V11AB cell 이 전체 L×H 에서 상위권이 아닐 가능성이 있다 (KNOWS 가 LIBERO 에서 L12·H3 을 보고했다).
- #7: JIT 후 p50 60–80 ms (smoke 67 ms, 10 suffix pass + kv feature), 첫 call 5 s, device peak ~7.1 GB.
- #8: CPU regression A 73 / B 78 / C 76 + 2 skip (내 실행과 같음). legacy 16D 기준선 불변 (core 무변경). SL0 경로(off replay) bit 동일.
