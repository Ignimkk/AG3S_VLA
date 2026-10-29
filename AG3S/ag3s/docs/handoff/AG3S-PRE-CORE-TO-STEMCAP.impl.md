# AG3S-PRE-CORE-TO-STEMCAP — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 2026-10-27 23:48 KST
> 명세: `AG3S-PRE-CORE-TO.task.md` §6 "꼭지 덮개 (stem cap) 구현 — 실험용, 기본 꺼짐" (사용자 지시 2026-10-27 23:36 KST)
> commit 안 함 (작업 트리에 다른 작업의 미commit 변경이 있다 — 아래 표의 hunk 만 이 STEP 의 것)

## 무엇을 했나 (평이한 요약 먼저)

target 을 지운 미세 층 (`target_exclusion = ball_owned`) 을 지을 때, ball_owned 가 지우고 **남은** seed 중 공 위 좁은
원기둥 (꼭지 덮개) 안의 것을 target 으로 보고 추가로 지우는 기능을 기본 꺼짐 flag `--target-stem-cap {off,in_ball,full}` 로 넣었다.
off (기본) 이면 코드가 한 줄도 돌지 않아 값 · stats · 기록 키가 지금과 같다 (test 로 보였다). 켜면 지운 수와 parameter 가
summary `target_exclusion.stem_cap` 에, 지운 seed 의 ijk 가 새 npz 키 `target_free_stem_cap_ijk` 에 실린다 —
기존 `n_removed` · `target_free_removed_ijk` 는 ball_owned 몫 그대로이고 덮개 몫과 섞이지 않는다.

## 규칙 (what-if `whatif_cap.py` 와 같은 정의)

seed voxel 중심 p, 공 중심 c, 맞춘 반지름 fit_r = `ball.provenance["fit"]["radius_mm"]`, 공 반지름 R:

| 조건 | 식 | 기본값 (config `EsdfConfig`) |
|---|---|---|
| 수평 | ‖p_xy − c_xy‖ ≤ r_cap | `stem_cap_radius = 0.020` m (`--stem-cap-radius-mm`) |
| 아래 끝 | p_z ≥ c_z + f · fit_r | `stem_cap_bottom_fit_frac = 0.5` (`--stem-cap-bottom-fit-frac`) |
| 위 끝 | p_z ≤ c_z + fit_r + h | `stem_cap_top = 0.025` m (`--stem-cap-top-mm`) |
| 지원면 | p_z ≥ `ball.z_min` | (what-if 에 없던 안전장치 — 아래 끝이 c_z 위라 실제 장면에서는 걸리지 않는다) |
| `in_ball` | ‖p − c‖ ≤ R 도 | what-if `cap_in` |
| `full` | 공 제한 없음 | what-if `cap_all` |

- **seed 정의 · voxel 중심 규약 (명세가 물은 것):** builder 자기 정의를 썼다. seed = cuRobo `site_index ≥ 0` (seed 단계 직후, ball_owned 의
  `_clear_seeds` 가 지운 뒤 다시 읽음 — `_seed_mask`), voxel 중심 = `_origin_of` (첫 voxel **중심**) + ijk · vs — `_ball_candidates` ·
  `target_free_trace_origin` 과 같은 규약. what-if 는 기록 free 층의 `distance == 0` voxel 을 `esdf_free_origin + ijk · VS` 로 썼다.
  둘이 같은 집합 · 같은 좌표인지 (특히 `esdf_free_origin` 이 첫 voxel 중심인지, `target_free_trace_origin` 과 같은지) 는 verifier 재현에서 확인할 것.
- **순서:** ball_owned 분할 · 제거 → 덮개 (남은 seed 만) → propagate. what-if 가 기록 free 층 (= ball_owned 뒤) 의 seed 에 규칙을 건 것과 같은 순서다.
- **fit_r:** what-if 처럼 맞춤 방식 (`surface_fit` · `centroid` 물러남) 과 무관하게 `fit.radius_mm` 를 쓴다. 값이 없으면 덮개를 건너뛰고
  `stem_cap.applied = false, reason = "no_fit_radius"` 를 남긴다. `fit_method` 를 기록한다.
- **부호 교정:** 덮개가 지운 seed ijk 를 그 계층의 부호 교정 집합에 더한다 (ball_owned 의 `sign ⊇ removed` 와 같은 규칙,
  `_force_sign_on_pure_voxels` 의 순수 판정 |d| > 1.5 vs 는 그대로). 덮개 안 non-seed voxel 은 넣지 않는다 (이웃 내부 음수를 숨기지 않기 위해).
- **적용 범위:** 미세 target-free 층의 `ball_owned` 분할 (`_ball_exclusion` 의 owned 갈래, IFG-CL2 보호가 있을 때도 그 뒤) 에만. 공이 있는 요청만
  (pipeline 은 쥔 뒤 · target 없음이면 공을 만들지 않는다 — `pipeline.py` 무변경). 거친 target-free 층 · oracle 경로 · `ball` 방식은 그대로.
  공 AABB 가 미세 창 밖이라 ball cand 가 없는 frame 은 덮개도 돌지 않는다.

## 바뀐 파일 (이 STEP 이 더한 hunk 만, 현재 줄 번호)

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/ag3s/config.py:1225-1247` | `EsdfConfig.target_stem_cap="off"` · `TARGET_STEM_CAPS` · `stem_cap_radius=0.020` · `stem_cap_top=0.025` · `stem_cap_bottom_fit_frac=0.5` | config 기본값 (기본 꺼짐) |
| `AG3S/ag3s/config.py:1424-1443` | validate: mode 목록 · radius > 0 · top ≥ 0 · 유한; 켜짐이면 `target_exclusion == "ball_owned"` 필수, `target_extent ∈ {oracle_visible, attention_surface}` 와 거절 | 효과 없는 조합을 조용히 띄우지 않기 |
| `AG3S/ag3s/fields/curobo_builder.py:815-884` | `STEM_CAP_MODES` · `StemCapSpec` · `stem_cap_fit_radius` · `stem_cap_bounds` · `stem_cap_mask` (순수 numpy) | 규칙의 단일 정의 — 단위 test · verifier offline 재현이 같은 함수를 부른다 |
| `AG3S/ag3s/fields/curobo_builder.py:2393-2394` | `_exclusion_counts` 가 `stem_cap` 블록을 stats 로 옮김 (있을 때만) | summary `target_exclusion.stem_cap` |
| `AG3S/ag3s/fields/curobo_builder.py:2457-2462` | 첫 frame 로그 `!!! 꼭지 덮개 (stem cap) = … seed N 개를 더 지웠다 !!!` | 켜진 것을 크게 |
| `AG3S/ag3s/fields/curobo_builder.py:3053-3061` | `_ball_exclusion` (owned 갈래 끝): `_stem_cap_spec()` 이 None 이 아니면 덮개 적용 · `counts["stem_cap"]` · `trace["stem_cap_ijk"]` · 부호 집합에 추가 | off 면 이 블록은 if 하나만 보고 지나간다 |
| `AG3S/ag3s/fields/curobo_builder.py:3065-3130` | `_stem_cap_spec` (config → spec) · `_stem_cap_exclusion` (후보 · seed 읽기 · `_clear_seeds` · 수) | 구현 본체 |
| `AG3S/ag3s/experiments/sources/constraint_record.py:515-521` | `EXCLUSION_TRACE_KEYS` 에 `"stem_cap_ijk"` | npz `target_free_stem_cap_ijk` (int16 (N,3), trace 에 있을 때만) |
| `AG3S/ag3s/experiments/sources/constraint_record.py:546-548` | summary `target_exclusion` keep 목록에 `"stem_cap"` | summary 블록 |
| `AG3S/trajopt/serve_safe.py:945-952` | `target_exclusion_section`: `--target-stem-cap` (off · 없음이면 키 없음) · mm → m 변환 | 준 flag 만 config 로 (기존 패턴) |
| `AG3S/trajopt/serve_safe.py:1390-1395` | 서버 시작 로그 `!!! AG3S STEM CAP — target_stem_cap = … !!!` | 다른 target-free 선택지와 같은 꼴 |
| `AG3S/trajopt/serve_safe.py:1958-1971` | argparse `--target-stem-cap {off,in_ball,full}` (default None) · `--stem-cap-radius-mm` · `--stem-cap-top-mm` · `--stem-cap-bottom-fit-frac` | flag |
| `AG3S/trajopt/serve_safe.py:2571-2593` | 인자 검사: parameter flag 만 (덮개 off) · 덮개 + `ball` · 덮개 + oracle extent · 범위 밖 값 → `ap.error` | 서버 뜨기 전에 막기 |
| `tests/ag3s/test_pre_core_to_stemcap.py` (새 파일) | 10 test | 아래 |

serve_safe 의 다른 diff hunk (cost_overrides · sqp/qp · announce_ta · main 등) 는 이 STEP 이전의 미commit 작업이고 건드리지 않았다.
`AG3S/ag3s/runtime/pipeline.py` 는 이 STEP 에서 바꾸지 않았다 (builder 가 config 를 직접 읽는다).

## 기록 모양

summary `target_exclusion` (덮개 켜짐일 때만 생김; off 이면 키 자체가 없다):

```json
"stem_cap": {"mode": "full", "radius_mm": 20.0, "top_mm": 25.0, "bottom_fit_frac": 0.5,
             "fit_radius_mm": 30.0, "fit_method": "surface_fit", "ball_radius_mm": 35.0,
             "applied": true, "n_cap_voxels": 376, "n_cap_seeds": 97, "n_already_removed": 54,
             "n_removed": 43, "n_removed_outside_ball": 19, "z_bottom_m": 0.89, "z_top_m": 0.93, "ms": 0.94}
```

(합성 test 장면의 실제 값.) `n_cap_voxels` = 덮개 안 voxel 수, `n_cap_seeds` = 덮개 안 seed (ball_owned 전) = `n_already_removed`
(ball_owned 가 이미 지운 것) + `n_removed` (덮개가 더 지운 것). `n_removed_outside_ball` 은 `full` 에서 공 밖이었던 것 (what-if `cap_all` − `cap_in` 에 해당).
`target_exclusion.n_removed` 와 `target_free_removed_ijk` 는 ball_owned 몫 그대로 (덮개 몫을 더하지 않는다) — 두 ijk 집합은 겹치지 않는다.

npz: `target_free_stem_cap_ijk` int16 `(n_removed, 3)` — `target_free_trace_origin` + ijk · `target_free_trace_voxel_size` = voxel 중심 (기존 trace 와 같은 origin · voxel).

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-development
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=osmesa PYTHONPATH=/mnt/dev/AG3S_VLA-development nice -n 19 \
  .venv-openpi-live/bin/python -m pytest -q -p no:cacheprovider \
  tests/ag3s/test_pre_core_to_stemcap.py tests/ag3s/test_t44_target_exclusion_owned.py \
  tests/ag3s/test_t46_target_free_residual.py tests/ag3s/test_t46e_oracle_extent.py \
  tests/ag3s/test_ifg_cl1_attention_surface.py tests/ag3s/test_ifg_cl2_hybrid_extent.py \
  tests/ag3s/test_t44c_target_free_coarse.py tests/ag3s/test_t44c_a6_sign_floor.py \
  tests/ag3s/test_t44c_a7_default_plane.py tests/ag3s/test_t44e_a8_coarse_support.py \
  tests/ag3s/test_dynamic_field.py tests/ag3s/test_task_dynamic_map.py tests/ag3s/test_roi_carve_gpu.py \
  tests/ag3s/test_config.py tests/ag3s/test_config_roundtrip.py tests/trajopt/test_t46f_a14c_oracle_combination.py
```

결과: **296 passed · 17 skipped (GPU · storage 조건) · 1 failed** (152 s). 새 파일 `test_pre_core_to_stemcap.py` 10/10 통과.

실패 1 개는 **이 변경 전부터** 실패한다 (변경 전 같은 명령의 T44 · T46 · CL2 세 파일 실행에서도 같은 실패, 65 passed · 1 failed):
`test_t44_target_exclusion_owned.py::test_the_record_writes_the_trace_keys` — `KeyError: 'target_free_protected_ijk'`. 이 test 는
`EXCLUSION_TRACE_KEYS` 의 모든 키가 payload 에 있기를 요구하는데 IFG-CL2 가 보호가 걸린 frame 에만 쓰는 `protected_ijk` 를 그 tuple 에
넣은 뒤로 깨져 있다. `stem_cap_ijk` 도 같은 "켜졌을 때만" 키라 이 test 가 고쳐지면 그 키도 빼야 한다 (test 쪽 수정 — 이 STEP 범위 밖이라 손대지 않음).

| test | 무엇을 보이나 |
|---|---|
| `test_mask_geometry_matches_the_whatif_rule` | 축 위 · 수평 19 mm → 안; 21 mm · 아래 끝 밑 · 위 끝 위 · 공 아래 · 이웃 과일 꼭대기 → 밖; 공 밖 덮개 안은 `full` 만; parameter 가 경계를 옮김 |
| `test_off_is_bit_identical_to_the_default_and_adds_no_keys` | 기본 = 명시 off: 값 `torch.equal`, exclusion 수 동일, `stem_cap` 키 · `stem_cap_ijk` trace 없음, 기록 payload 키 · 배열 · summary 동일 |
| `test_full_removes_the_stem_and_keeps_neighbour_below_and_table` | 합성 사과 + 가는 꼭지 (non-target 근거로 잘못 들어감) + 1 mm 옆 이웃 과일 + 테이블: off 대비 사라진 seed = 정확히 `stem_cap_ijk`; 꼭지 seed 0 개 남음; 이웃 · 테이블 · 덮개 아래 seed 전부 유지 (이웃은 덮개 높이 띠 안에 있지만 수평 > 20 mm); ball_owned 수 · `removed_ijk` 불변; 지운 voxel 부호 |
| `test_in_ball_keeps_the_seeds_outside_the_ball` | `in_ball` ⊂ `full`, 공 밖 꼭지 seed 는 `in_ball` 에서 남음 |
| `test_deterministic_and_parameters_from_config` | 두 번 같은 값 · 같은 ijk; config parameter 가 기록 · 범위에 반영 |
| `test_no_fit_radius_skips_the_cap_and_says_why` | fit 반지름 없음 → `applied=false`, 값 = off |
| `test_record_has_the_stem_cap_block_and_ijk_key` | npz 키 int16, 수 = `stem_cap.n_removed`, `removed_ijk` 와 겹침 없음, JSON |
| `test_config_defaults_round_trip_and_validation` | 기본값 · 왕복 · 거절 (모드 오타 · `ball` 과 함께 · radius 0 · top < 0 · nan · oracle extent) |
| `test_cli_defaults_and_mapping` | flag 없음 / `off` → 키 없음; `in_ball` + mm flag → config 키 · `build_ag3s` 생존 검사와 같은 비교 |
| `test_cli_rejects_cap_without_ball_owned_and_params_without_cap` | `ap.error` (exit 2) 경우 |

GPU (실제 cuRobo) smoke 는 하지 않았다 (GPU 사용 금지 지시). 덮개 경로는 기존 ball_owned 와 같은 `_seed_mask` · `_clear_seeds` (torch 인덱싱) 만 쓴다.

## verifier 가 알아야 할 것

- 새로 생긴 플래그 · 기본값 변경: `--target-stem-cap {off,in_ball,full}` (기본 None → config `esdf.target_stem_cap = "off"`),
  `--stem-cap-radius-mm` (20) · `--stem-cap-top-mm` (25) · `--stem-cap-bottom-fit-frac` (0.5). `--target-exclusion ball-owned` 필수.
  serving 기본값 변경 없음.
- offline 재현 (§6 검증 항목 — what-if `cap_in` 894 · `cap_all` 400, 금지 class 0 · 2): `curobo_builder.stem_cap_mask(centres, ball, StemCapSpec(mode=…), fit_r)` 를
  기록 free 층 seed 중심에 부르면 builder 와 같은 규칙이다 (`fit_r` = `stem_cap_fit_radius(ball)[0]`). what-if 대비 차이는 `p_z ≥ z_min` 조건 하나 (실제 장면에서 효과 없을 것).
  seed 집합 · origin 규약이 기록 (`esdf_free_distance == 0` · `esdf_free_origin`) 과 builder (`site_index ≥ 0` · 첫 voxel 중심) 에서 같은지 확인할 것.
- 재생산이 필요한 산출물: 없음.
- 옛 기록과 호환이 깨지는가: 없음 (off 면 키가 늘지 않는다; 켜짐 기록에만 `stem_cap` · `target_free_stem_cap_ijk` 가 더해진다).
  `target_free_replay.py` · `cl2_hybrid_replay.py` 같은 offline 재구성 도구는 덮개를 모른다 — 켜짐 기록을 그 도구로 replay 하면 덮개 몫이 빠진다.

## 내가 기대하는 결과

(verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.)
같은 39 요청에서 what-if 숫자 (`cap_in` 새 위반 행 894 · `cap_all` 400, 금지 class 제거 0 · 2) 를 재현할 것으로 본다 — 규칙이 같고 적용 순서
(ball_owned 뒤 남은 seed) 도 같다. 차이가 나면 첫 후보는 seed 정의 (distance == 0 대 site_index ≥ 0) 또는 origin 규약이다.
