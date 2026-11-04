# SL9 — 구현 (수정안 A · B ablation 판 설정)

> writer: ag3s-implementer · 읽는 쪽: verifier, scribe, lead · 2026-11-04 KST · task `SL9.task.md` §1 · §2 · §4-2 · §4-3
> HEAD `94f46821` 위의 working tree (commit 안 함). GPU closed loop · 실험은 돌리지 않았다.

## 무엇을 했나 (평이한 요약 먼저)
development main `4ecf8d61` 의 새 서버 옵션 12개를 Panda 서버에서 분류했다. 수정안 A · B 의 7개는 "이식됨"
(`surface_seed_ats_gate` 와 같은 길로 AG3S esdf 설정에 실제로 들어간다), T97 3개와 `palm_cover_margin` 은 "미이식(거절)" 이다.
SL9 §2 의 새 판 7개를 `profile.py` 에 등록했고 (기준 판 flag 는 한 토큰도 바뀌지 않았다), 판마다 Panda 서버를 만들어
시작 거절이 없고 설정값이 AG3S 에 도달하는지 test 로 보였다. `panda/surface` 응답 키(SL7i)에는 A · B 가 켜졌을 때만
닫힌 물체 요약 · 부호 수정 요약 · 손 층 voxel 크기가 붙는다.

## 바뀐 파일
| 파일 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/adapters/panda/server.py` `UNPORTED_FINAL_CONFIG_DESTS` (뒤쪽 새 묶음) | `surface_crease_protrusion`, `surface_crease_protrusion_area`, `surface_crease_hand_freeze_mm`, `palm_cover_margin` | T97 은 이번에 안 잼, 손바닥 덮개는 Panda 에 해당 없음 (SL9 §4-1) |
| `AG3S/adapters/panda/server.py` `PORTED_FINAL_CONFIG_DESTS` | `surface_solid_owned`, `_protect_mm`, `_dilate_mm`, `_shadow_mm`, `_dn_ever_target`, `_dn_support`, `surface_removed_sign_free` | `serve_safe.target_exclusion_section` → esdf 키 → Panda `build_ag3s` 의 survival check 가 잃은 키를 `RuntimeError` 로 잡는다 |
| `AG3S/adapters/panda/server.py` `surface_call_record` + 새 `_solid_sign_free_record` | A · B 기록의 scalar 요약 (아래 "기록 키") | SL9 §4-3 |
| `AG3S/adapters/panda/profile.py` (`SOLID_FLAGS`, `SIGN_FREE_FLAGS`, `_SL9`, `P0M_SL9_VERSIONS`, `PC_SL9_VERSIONS`, `SL9_VERSIONS`) | 새 판 7개 | SL9 §2 |
| `AG3S/adapters/panda/profile.py` `is_final_version` · `version_server_flags` · `version_flag_values` | P0m 계열 SL9 판을 P0m 처럼(HSR6 probe 없음), pc-surface 계열은 `PC_LATCH_VERSIONS` 에 넣어 probe 를 받는다. SL9 flag 의 dest 는 **하나라도 켜졌을 때만** 기록 키에 합류 | 기준 판의 `version_flag_values` 키 집합을 그대로 둔다 |
| `tests/benchmark/test_sl9_ablation_versions.py` | 신규 18 test | |
| `tests/trajopt/data/ag3s_final_merge_surface.json` | merge `f223270a` 의 namespace key 11개 (parser default) 추가 | golden namespace test (`94f46821` 에서도 실패하던 것) |
| `tests/benchmark/test_sl6h_panda_placement_contact.py:59` | HEAD 비교 집합에서 `P0M_SL9_VERSIONS` 제외 | 새 판은 옛 commit 에 없다 |

새 판 flag (모두 기준 판 flag 뒤에 붙는다, A 는 먼저 `--surface-hull-fill`, `--surface-contact-allowance-mm 10` 을 뺀다):
A = `+ --surface-solid-owned --surface-solid-protect-mm 5 --surface-solid-dilate-mm 0 --surface-solid-dn-ever-target --surface-solid-shadow-mm 80`,
B = `+ --surface-removed-sign-free`.

| 판 | 기준 | A | B |
|---|---|:-:|:-:|
| `AG3S-final-panda-p0m-B` / `-A` / `-AB` | `P0M_FLAGS` | – ✓ ✓ | ✓ – ✓ |
| `AG3S-final-panda-v3-e0to-pc-surface-fb-B` | `PC_SURFACE_FALLBACK_FLAGS` | – | ✓ |
| `AG3S-final-panda-v3-e0to-pc-surface-B` / `-A` / `-AB` | `PC_SURFACE_FLAGS` | – ✓ ✓ | ✓ – ✓ |

### 기록 키 — 어디서 오나
`surface_call_record(safe)` 가 `safe.ag3s._last_esdf_field.stats["target_free"]` 를 읽는다 (읽기만; 매 field build 가 `_last_esdf_field` 를 갱신, `pipeline.py:2854`).
- A `solid` = `stats["target_free"]["attention_surface"]["solid"]` (`pipeline.py:3968-3975` 가 `_surface_solid` 기록(`pipeline.py:936`) + 빌더의 `oracle_split["solid"]` 를 `voxels` 로 붙인 것).
  scalar: `applied`/`why`, `volume_cm3`, `n_projected` (= R⊥ 점 수), `n_vertices`, `region_source`, `region_age_frames`, `protect_mm`, `dilate_mm`;
  `shadow` (`shadow_mm`, `applied`, `target_px_by_camera`), `dn_ever_target`, `voxels` (`n_removed_surface/band/solid/dilate/shadow/total`, `n_protected_tau*`, `n_shadow_hits`, `n_removed_shadow_by_camera` — 카메라별 기여, dummy right_wrist 는 키 자체가 없거나 0).
- B `removed_sign_free` = `stats["target_free"]["oracle_split"]["removed_sign_free"]` (`curobo_builder.py:2848-2850` 가 `counts["oracle"]` 에 넣고 `_exclusion_counts` 가 `oracle_split` 로 옮김): `applied`, `n_removed_listed`, `n_flipped`, `n_flipped_within_purity`, `n_zero_left`, `purity_radius_m`.
- `hand_voxel_m` = `stats["target_free"]["tiers"]` 의 `fine` tier `voxel_size` (손 층), `field_request_seq` = `geometry_version.request_seq` (이 field 가 어느 요청 것인지; 요청 사이에 field 를 다시 안 만들면 이전 값일 수 있어 둔다).
- 위 둘 다 없으면 (flag 꺼짐) `solid` · `removed_sign_free` · `hand_voxel_m` · `field_request_seq` 가 모두 빠져 SL7i 기록 그대로다 (test `test_record_is_unchanged_without_the_new_records`).
- `_surface_frame` 이 없는 호출은 `{"state": "none"}` 그대로 (A · B 키 없음).

## 단위 검증
```bash
source scripts/setup/env.sh
MUJOCO_GL=osmesa .venv-openpi-live/bin/python -m pytest tests/benchmark/test_sl9_ablation_versions.py -q   # 18 passed
```
전체 회귀 결과는 이 문서 맨 아래 "회귀" 절.

## verifier 가 알아야 할 것
- 새로 생긴 플래그·기본값 변경: 없음 (development flag 는 모두 기본 꺼짐). 새 **판 이름** 7개만 추가.
- 기준 판 `AG3S_VERSIONS` 35개의 flag 는 HEAD 와 토큰 단위로 같다 (HEAD 의 `profile.py` 를 불러 비교: 변경 0). 기준 판들의 `version_flag_values` 키 집합도 HEAD 와 같다.
- `surface_solid_dn_support` 는 어떤 판도 주지 않으므로 (기본 `exclude`) 기록 키에는 안 나온다. 분류는 PORTED.
- 옛 기록과 호환이 깨지는가: 없음. `panda/surface` 새 키는 A · B 판에서만 추가된다.
- 한계: `hand_voxel_m` 은 field 에 `fine` tier 가 있을 때만 (없으면 첫 tier).
- fb 기준에는 A 판이 없다 (config 가 hull fill 없이 solid 를 켜는 것은 허용하지만 `surface_keep_last` · `surface_no_ball` 이 필요한데 fb 는 둘 다 없다 — `EsdfConfig.validate`, `config.py:1815`).

## 내가 기대하는 결과
(verifier 는 측정이 끝나기 전에 읽지 않는다.) 새 판 1 episode smoke 에서 A: `solid.applied == true` 인 요청이 있고 `n_projected > 0`; B: `removed_sign_free.applied == true`; `hand_voxel_m` 가 찍힘.

## 회귀 (main checkout, worktree 아님)
- `tests/benchmark` (`.venv-openpi-live`, 변경 후 첫 전체 실행): 405 passed, 4 failed (1178 s). 4건의 원인과 처리:
  | test | 원인 | 처리 |
  |---|---|---|
  | `test_panda_sl5a.py::test_rby1_golden_identity` | development merge `f223270a` 가 serve_safe namespace 에 11 key 를 더했는데 `tests/trajopt/data/ag3s_final_merge_surface.json` 에 없었다. **`94f46821` 에서도 실패** (detached worktree 로 확인: `--check-serve-safe` identical false) | 11 key 를 parser default 로 `bare`/`v11ab` namespace 에 추가 (이전 merge 들과 같은 방식, `_doc` 에 기록). 통과 |
  | `test_sl6h_panda_placement_contact.py::test_every_head_version_keeps_its_argv` | 새 P0m 계열 SL9 판이 "옛 HEAD 에 없는 판" 제외 집합에 없었다 | 집합에 `P0M_SL9_VERSIONS` 추가. 통과 |
  | `test_panda_sl5j_recorder.py::test_signal_stopped_process_leaves_meta_json[2]` | subprocess 60 s timeout — trajopt 전체와 동시에 돌려 CPU 경합 | 단독 재실행 통과 |
  | `test_parallel_gpu_launcher.py::test_script_runs_at_most_two_jobs_under_one_held_lock` | 같은 경합 (시간 기반) | 단독 재실행 통과 |
  4건 재실행: `test_panda_sl5a::test_rby1_golden_identity` + sl5j + parallel launcher + sl6h = 37 passed. 이어서 SL9 · SL7i test 21 passed.
- `tests/trajopt` (`.venv-ag3s`): 2713 passed, 48 skipped, **1 failed** = 알려진 `test_t43t_replay::test_off_is_bit_identical_to_the_pre_fix_commit` (2411 s).
- 알려진 실패 중 `tests/ag3s` (`test_t55_*` 3, `test_geo_grasp`, `test_t86_patch_probe`) 는 이번 범위(`tests/benchmark` · `tests/trajopt`)가 아니라 돌리지 않았다.
