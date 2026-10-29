# IFG-CL2 — 구현 (hybrid target extent: ball_owned 범위 + attention-surface 보호)

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 2026-10-17 18:40 KST
> task: `AG3S/ag3s/docs/handoff/IFG-CL2.task.md` · branch `feature/ag3s-development` @ `f15e5872` (commit 안 함, CL1 미commit 변경 위에 얹음)
> CPU 만, test 는 `.venv-ag3s` 만 (torch 없음 — GPU 경로 import 불가). GPU · server · rollout 실행 없음. 잠금 `ifg_*.py` · 문서 · figure 수정 없음.

## 무엇을 했나 (평이한 요약 먼저)

새 `esdf.target_extent = "attention_surface_hybrid"` (CLI `--target-extent attention-surface-hybrid`) 를 넣었다. 지우는 범위는
C1aN 의 `ball_owned` 그대로이고, CL1 의 attention-surface 선택 (같은 LOCK 함수) 은 **보호에만** 쓴다: 공 안 seed 를 같은
`ball_owned` 규칙으로 한 번 더 가르는데, 이번 근거는 이번 frame 의 graph node 다 (target = 선택 영역 node, non-target = 선택 밖
· 비지지면 node). **두 분할 모두 "지워라" 한 seed 만 지운다** — 그래서 지우는 seed 는 C1aN 의 부분집합이다. 선택 영역이 manipulated
물체와 맞지 않으면 (identity guard) 그 frame 은 보호 없이 C1aN 과 같다. oracle memory 는 쓰지 않는다.

offline 재생 (IFG-CL1 closed-loop 기록 68 run, pre-grasp chunk 1,430 개, 기록만으로 재계산 — C1aN 기록 600/600 chunk 비트 일치)
결과: 같은 TSDF 위에서 hybrid 는 C1aN 규칙 대비 **이웃 과일 seed 제거를 62 % 줄인다** (chunk 평균 35.9 → 13.7, SURF 성공
run). 그러나 **정답 target seed 를 더 남긴다** (지지면 위, chunk 평균 55.5 → 65.9; 공 안 중앙값 0 → 35). 보호된 target seed 의 71 % 는
가장 가까운 "other" node 가 **선택되지 않은 target 자신의 표면**이다 (영역이 과일 일부만 덮음) — 수락 조건 "C1aN ≈ 1 수준" 은
**충족하지 못한다.** 그리고 task 의 진단 전제 (SURF 는 target 의 안 보인 쪽을 남긴다, 134 vs 1) 는 **공통 seed 집합으로 다시 세면
성립하지 않는다** (아래 §3) — lead 판단이 필요하다.

## 1. 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/ag3s/runtime/attention_surface_hybrid.py` (신규) | `HybridSamples` · `hybrid_samples(result)` (선택 node / 선택 밖 비지지 node / 전 node + 선택 여부) · `identity_guard` (`instance` = manipulated 기하 점의 ≥ ½ 이 fusion voxel 10 mm 안 최근접 node 가 선택 node — `OracleMemory.instance_of` 와 같은 규칙을 이번 frame node 에; `centre` = 선택 중심이 공 안; `both` 기본; 두 규칙 다 늘 기록) · `hybrid_evidence` (→ `SurfaceProtectEvidence` 또는 None + `why`: `attention_surface_abstained (<why>)` · `no_ball` · `identity_off (<why>)`, `ms`) | 설계 2 · 3 |
| `AG3S/ag3s/fields/curobo_builder.py:661-713` | `SurfaceProtectEvidence` (target / other node 점), `surface_protect_split` (순수 numpy): `est = _ball_owned_split(cloud 근거)`, `srf = _ball_owned_split(node 근거, 같은 tie · max_dist · unowned · ambiguous)`, removed = est ∧ srf, sign = est ∧ srf, `protected` = est.removed ∧ ¬srf.removed | 설계 1 · 2 (removed ⊆ C1aN) |
| `curobo_builder.py:1214` · `:1234-1240` | `update(surface_protect=)` — 없으면 예전 호출. 공 · `ball_owned` 없이 오거나 `oracle_evidence` 와 함께 오면 torch 전에 `ValueError` | fail loudly |
| `curobo_builder.py:1363-1366` · `_build_tiers` 시그니처 · `:1782-1784` | 미세 target-free 계층에만 `protect=` 를 `_ball_exclusion` 으로 (거친 계층 · 주 계층 무관) | |
| `curobo_builder.py:2140-2210` | `_ball_exclusion(protect=)`: protect 가 있으면 `surface_protect_split`. counts: `n_removed` (보호 후), `n_kept_nontarget` = cloud non-target ∪ protected, `surface_protect` {`n_protected`, `n_removed_estimated`, `n_protected_from_{target,unowned,ambiguous}`, `srf` 분류 수, `split_ms`}; trace: `nontarget_kept_ijk` ⊇ protected, 새 `protected_ijk`. protect 없으면 코드 경로 그대로 | 기록 · GT 대조 script 가 그대로 읽게 |
| `curobo_builder.py:1839-1840` | `_exclusion_counts` → `stats["target_free"]["surface_protect"]` (있을 때만) | |
| `AG3S/ag3s/config.py:1141-1152` · `:1312-1320` | `TARGET_EXTENTS` + `"attention_surface_hybrid"`; `surface_hybrid_identity = "both"` (`both`/`instance`/`centre`); hybrid 인데 `target_exclusion != "ball_owned"` 이면 `AG3SConfigError`. F1 · F2 는 hybrid 와 조합 허용 (보호하는 대상이 그 분할이다) | 설계 1 · 3 |
| `AG3S/ag3s/runtime/pipeline.py:213-215` · `:448` · `:1705` | `_surface_hybrid` (이번 frame samples) — init · reset · frame 시작에 None | 지난 frame 값이 남지 않게 |
| `pipeline.py:704-754` | `_attention_surface_hybrid_update` (CL1 `_attention_surface_update` 와 같은 view · robot mask 규칙, `select_target_surface(keep_debug=True)`, memory fold 없음; `_surface_frame` · `_surface_codes` · `_surface_hybrid`) · `_hybrid_protect` (radius = `timing.fusion_voxel_size`, rule = config) | 설계 2 · 4 |
| `pipeline.py:2011-2017` | ESDF 단계: hybrid 이면 위 update (profiler stage `attention_surface`, SURF 와 같은 이름) | 배선 |
| `pipeline.py:2227-2229` | `debug["attention_surface_codes"]` 를 hybrid 에도 (code 영상 기록 → offline 재생 가능) | 기록 |
| `pipeline.py:3018-3026` · `:3077-3079` · `:3087-3091` | `_build_esdf`: layer 없으면 `why: no_target_free_layer`, 있으면 `_hybrid_protect`; evidence 가 있을 때만 `surface_protect=` kwarg; `stats["target_free"]["attention_surface_hybrid"]` (보호 기록) + `["attention_surface"]` (선택 기록, CL1 과 같은 키 — 기존 집계 script 가 읽음) | 설계 3 · 기록 |
| `AG3S/ag3s/experiments/sources/constraint_record.py:514-518` · `:541-543` | npz `target_free_protected_ijk` (있을 때만), summary `target_exclusion.attention_surface_hybrid` · `.surface_protect` (있을 때만) | 기록 키 |
| `AG3S/trajopt/serve_safe.py:1689-1702` · `:1711-1717` | `--target-extent attention-surface-hybrid`, `--hybrid-identity {both,instance,centre}` | 새 flag |
| `serve_safe.py:917-918` · `:2038-2040` | esdf 키 `surface_hybrid_identity` (준 flag 만), record meta `target_extent` · `surface_hybrid_identity` (hybrid 일 때만) | |
| `serve_safe.py:2175-2199` | 시작 거절: hybrid 인데 `--target-exclusion ball-owned` 아님 · relax/none policy · `--no-safe`/`--no-perception` · `--attention-source` ≠ `noun_target:L10H7` · `--no-attention`; `--hybrid-identity` 단독; `--oracle-*` 는 hybrid 에서 안 읽히므로 기존 규칙대로 거절 | fail loudly |
| `serve_safe.py:1253-1261` | 시작 warning (`!!! AG3S IFG-CL2 …`) | |
| `AG3S/trajopt/safe_policy.py:389-391` | server metadata `target_extent` · `surface_hybrid_identity` (hybrid 일 때만, `ORACLE_SEG_REQUIRED` 없음) | |
| `AG3S/ag3s/experiments/studies/cl2_hybrid_replay.py` (신규, 잠금 아님) | offline 재생 도구 + `--summary` | 수락 조건 재생 |
| `tests/ag3s/test_ifg_cl2_hybrid_extent.py` (신규) | 16 test | 수락 조건 |

`AG3S/ag3s/runtime/attention_surface.py` (CL1) 는 **수정하지 않았다** (`keep_debug=True` 로 graph 를 받는다). 잠금 `ifg_*.py`, lead 의
`ifg_cl1_keptseeds.py` 는 import 만 한다.

## 2. 단위 검증 (전부 `.venv-ag3s`, CPU)

```bash
cd /mnt/dev/AG3S_VLA-development
MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m pytest -q \
    tests/ag3s/test_ifg_cl2_hybrid_extent.py tests/ag3s/test_ifg_cl1_attention_surface.py
#   → 39 passed (54 s)  [CL2 16 + CL1 23]
MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider \
    tests/ag3s tests/trajopt/test_safe_policy.py tests/trajopt/test_t44_target_permission.py tests/trajopt/test_t44c_a7_not_target_tiers.py \
    tests/trajopt/test_t44c_target_free_coarse_branch.py tests/trajopt/test_t44e_a9_nonauthorized_target_margin.py \
    tests/trajopt/test_t46c_placed_no_seed_certify.py tests/trajopt/test_t46f_a14_wrist_cover.py tests/trajopt/test_t46f_a14b_inset_pose.py \
    tests/trajopt/test_t46f_a14c_oracle_combination.py tests/trajopt/test_t46g_reach_hand_relative.py tests/trajopt/test_t57a_attention_source.py \
    tests/trajopt/test_record_and_hold_state.py
#   → 2045 passed, 18 skipped, 0 failed (18 m 42 s)
```

| 수락 조건 | test | 결과 |
|---|---|---|
| 기본 · 기존 모드 비트 동일 | `test_pipeline_default_never_calls_the_hybrid` (hybrid 함수를 터지게 바꿔도 기본 · `estimated` · `attention_surface` 에서 호출 · kwarg · stats 키 없음), `test_ball_exclusion_with_and_without_protection` (protect 없는 `_ball_exclusion` 은 `surface_protect` · `protected_ijk` 없음), `test_record_keys_only_when_present`, `test_cli_maps_to_esdf_keys_only_when_given`, `test_server_metadata_only_when_on` + 기존 `tests/ag3s` 전부 | 통과 |
| 보호 규칙 | `test_protection_keeps_the_missed_neighbour_and_never_widens_the_range` (cloud 가 놓친 이웃 seed ≥ 90 % 보호; removed ⊆ C1aN, sign ⊆; target 의 안 보인 아래쪽 (두 node 집합에서 max_dist 밖) 은 그대로 제거; selected node 쪽이 tie 이상 가까운 seed 는 보호 안 함; `est` 분류 키 불변), `test_far_other_nodes_change_nothing`, `test_ball_exclusion_with_and_without_protection` (counts · trace 정합, torch 없이 fake seed mask) | 통과 |
| identity 안전장치 | `test_identity_guard_rules` (이웃에 선 영역: instance 가 끈다 — centre 단독은 통과시킴을 기록; 공 밖 영역: centre; 반반: no_majority; 멀리: no_node_near; rule 선택), `test_hybrid_evidence_reasons`, `test_pipeline_hybrid_abstain_and_identity_off` | 통과 |
| 기록 키 | `test_record_keys_only_when_present`, `test_pipeline_hybrid_protects_the_ball_owned_split` (kwarg · `attention_surface_hybrid` · `attention_surface` · oracle memory 미사용 · reset) | 통과 |
| CLI 거절 | `test_cli_refusals` (ball-owned 없음 · source 없음/다름 · relax · `--hybrid-identity` 단독 · `--no-attention` · `--oracle-*`) | 통과 |
| 재생 정합 (storage) | `test_replay_reproduces_the_record` (C1aN 기록 trace 재현, SURF code 영상 pixel 일치, hybrid ⊆ est) | 통과 |

**돌리지 못한 것:** torch 가 필요한 builder test (`test_t44_target_exclusion_owned.py` 등 — `.venv-ag3s` 에서 skip, 18 skipped 중
일부). 내 builder 변경은 kwarg 가 없으면 같은 코드 경로지만 GPU 쪽 비트 동일은 **verifier 가 lock 을 잡고** 확인해야 한다.

## 3. offline 재생 (수락 조건 1)

- 도구: `AG3S/ag3s/experiments/studies/cl2_hybrid_replay.py`; 출력 `$AG3S_STORAGE_ROOT/results/development/20261017T172222+0900_f15e587290f6_70b986e85ef9/cl2_replay/<run>.json` · `summary.json`
- 입력 = `$CL1/analysis/<run>.json` 의 server segment chunk (k ↔ client step k), 전부 기록:
  seed = 기록된 미세 주 계층 거리 == 0 voxel (공 안 seed 수 · C1aN trace 와 **600/600 chunk 비트 일치**); `ball_owned` 근거 = 기록 target 점
  + `filtered_cloud − target_point_indices − support_mask` (또는 5 mm proximity); attention-surface = `select_target_surface` 를
  **client 기록의 depth/K/T** (서버가 받은 uint16 mm), chunk 의 float16 attention, robot mask = 기록 code 영상의 0 pixel 로 다시 실행 →
  기록 code 영상과 **pixel 일치 797/823 chunk** (나머지 26 chunk 최대 358 px 차 — float16 attention 의 동점 근처로 추정, 확인 안 함).
  code 가 없는 chunk (abstain) 는 서버와 같이 보호 없음. C1aN chunk 는 code 영상이 없어 hybrid 를 재생할 수 없다 (`recorded` 만).
- 범위: 68 run (C1aN 34 · SURF 34), grasp_t 전 chunk 중 공이 있는 chunk. GT = 재생 scene 의 convex hull ± 5 mm (`ifg_cl1_keptseeds` 와 같은 함수),
  배타 순서 target → 이웃 과일 → crate (box geom 별 hull) → 기타.
- **seed 집합을 공통으로 했다:** 미세 계층의 모든 seed. lead 의 `ifg_cl1_keptseeds` 는 trace 의 `nontarget_kept_ijk` 만 센다 — C1aN 은 공 안,
  SURF 는 oracle 후보 (공 밖까지) 라 두 arm 이 **다른 집합**을 센다. 그 수는 `keptseeds` 열로 따로 둔다.

chunk 당 값 (평균, 괄호 = 중앙값). "target 남김" = 정답 target 인데 지우지 않은 seed, "지지면 위" = 공의 `z_min` (지지면 + 5 mm) 위.

| 그룹 (chunk) | 변형 | target 남김 (전부) | target 남김 (지지면 위) | target 남김 (공 안) | 이웃 과일 지움 | crate 지움 | keptseeds (lead 방식) |
|---|---|---:|---:|---:|---:|---:|---:|
| C1aN 성공 34 run (600) | 기록 = C1aN 규칙 재생 | 126.5 (117) | 50.0 (3) | 47.3 (1) | 39.8 (0) | 0 | (0) |
| SURF 실패 6 run (352) | 기록 (SURF) | 140.4 (148) | **35.5 (24)** | 30.2 (22) | 39.0 (0) | **211.2 (0)** | (123.5) |
| | C1aN 규칙 (같은 TSDF) | 218.5 (188) | 113.6 (71) | 81.8 (56) | 36.7 (0) | 0 | |
| | **hybrid (both)** | 226.1 (196) | 121.1 (80) | 89.3 (59) | **14.0 (0)** | 0 | |
| | hybrid (instance / centre / guard 없음) | 226.3 / 230.2 / 233.0 | 121.4 / 125.3 / 128.1 | 89.6 / 93.4 / 96.3 | 14.0 / 9.7 / 9.7 | 0 | |
| SURF 성공 28 run (478) | 기록 (SURF) | 93.3 (80) | 16.8 (14) | 16.8 (14) | 2.1 (0) | 0 | (79) |
| | C1aN 규칙 (같은 TSDF) | 132.0 (108) | 55.5 (3) | 54.6 (0) | 35.9 (0) | 0 | |
| | **hybrid (both)** | 142.4 (123) | 65.9 (35) | 65.0 (35) | **13.7 (0)** | 0 | |
| | hybrid (instance / centre / guard 없음) | 142.4 / 143.0 / 153.9 | 65.9 / 66.5 / 77.4 | 65.0 / 65.7 / 76.5 | 13.7 / 10.4 / 10.4 | 0 | |

hybrid − C1aN 규칙 (같은 chunk 짝): target 남김 (지지면 위) 이 늘어난 chunk 27.6 % (실패) · 33.1 % (성공), 평균 +7.5 · +10.4, p90 +29 · +39,
최대 +150. 이웃 지움이 줄어든 chunk 9.1 % · 20.9 %, 평균 −22.7 · −22.1 (대부분 chunk 는 공 안에 이웃이 없어 0 → 0).

SURF 실패 6 run 별 (chunk 평균; 지지면 위 target 남김 기록/C1aN 규칙/hybrid · 이웃 지움 기록/C1aN 규칙/hybrid · SURF 의 crate 지움):

| run | 실패 | chunk | 보호 on | target 남김 | 이웃 지움 | crate 지움 (SURF) |
|---|---|---:|---:|---|---|---:|
| SURF_ep1769_s17691 | no_grasp | 73 | 0.73 | 4.0 / 44.9 / 46.2 | 0.0 / 5.1 / 4.9 | 431.3 |
| SURF_ep1893_s18931 | no_grasp | 55 | 0.82 | 28.9 / 80.0 / 86.0 | 2.6 / 0.1 / 0.1 | 311.0 |
| SURF_ep391_s3911 | no_grasp | 59 | 0.83 | 28.2 / 45.3 / 52.3 | 0.0 / 0.0 / 0.0 | 159.2 |
| SURF_ep485_s4851 | not_placed | 15 | 0.73 | 14.3 / 73.7 / 74.3 | 0.0 / 467.2 / 197.7 | 0.0 |
| SURF_ep623_s6231 | no_grasp | 75 | 0.71 | 47.8 / 130.1 / 149.3 | 180.8 / 73.7 / 21.4 | 19.3 |
| SURF_ep764_s7641 | no_grasp | 75 | 0.87 | 68.5 / 250.3 / 255.3 | 0.6 / 0.0 / 0.0 | 199.0 |

**보호가 꺼진 frame** (공이 있는 pre-grasp chunk 중, rule `both`):

| 그룹 | 보호 on | 꺼진 이유 (chunk 수) |
|---|---:|---|
| SURF 실패 (352) | 78.4 % | identity_off — instance:no_selected_node + centre:outside_ball 35 · instance:no_majority 15 · instance:no_node_near + centre 14 · centre 단독 5; attention_surface_abstained (no_region) 7 |
| SURF 성공 (478) | 96.9 % | identity_off — no_selected_node + centre 9 · no_majority 5 · no_node_near + centre 1 |
| C1aN (600) | — | code 영상이 기록되지 않아 재생 불가 (C1aN 서버는 attention-surface 를 계산하지 않았다) |

**보호된 target seed 는 어디서 오나** (진단, 4 run 짝수 chunk; scratch 계산 — 잠금 / 저장 산출물 아님): 보호된 정답 target seed
875 중 가장 가까운 "other" node 가 **정답 target 자신** 619 (71 %), 이웃 218, 기타 38; `srf` 분류는 ambiguous 451 · non-target 424.
그 node 의 대부분 (834/875) 이 manipulated cloud 점에서 15 mm 안이다 → **선택 영역이 과일의 일부만 덮을 때, 나머지 target 표면이
"other" 가 되어 target seed 를 보호한다** (CL1 impl 위험 6 과 같은 기전).

### 해석 (측정 기반, lead 판단 사항)

1. **수락 조건:** 이웃 지움 감소 — **충족** (같은 TSDF 에서 −62 %, C1aN 자체 run 평균 39.8 보다도 낮음). target 남김 "C1aN ≈ 1 수준" — **미충족**
   (공 안 중앙값 C1aN 규칙 0 → hybrid 35 on SURF 성공 chunk; 정의상 hybrid ≥ C1aN 규칙).
2. **task 의 진단 전제가 공통 집합에서 재현되지 않는다:** "SURF 는 target 의 안 보인 쪽 seed 를 남긴다 (chunk 중앙값 134 vs C1aN 1)" 의 134 는
   SURF 의 oracle 후보 (공 밖 · 지지면 guard 아래 포함) 를 세고 C1aN 은 공 안만 센 차이가 크다. 같은 seed 집합 · 같은 TSDF 에서 지지면 위
   target 남김은 SURF 기록 (실패 평균 35.5) < C1aN 규칙 (113.6) 이다 — SURF 는 오히려 target 을 **더** 지웠다. 공 밖 target seed 의 대부분은
   과일 밑 table seed (hull ± 5 mm 안, `z_min` 아래 — 모든 extent 가 일부러 남긴다).
3. **실패와 같이 가는 신호:** SURF 실패 6 중 5 run 이 **crate seed 를 수백 개씩 지웠다** (chunk 평균 159–431; SURF 성공 28 run 은 0).
   identity 오류로 선택 영역이 crate 쪽에 섰을 때 oracle 경로가 그 표면을 지운 것으로 보인다. hybrid · C1aN 규칙은 crate 를 지우지 않는다
   (공 밖 · guard). 원인이라고 확정하지는 않았다 (closed-loop 인과는 재생으로 못 본다).
4. 위 2 · 3 이 맞다면 hybrid 의 "ball 부피 제거" 부분은 SURF 실패의 원인을 고치는 것이 아니고, 실익은 이웃 보호 (2) 와 crate 를 지우지
   않는 것 (C1aN 과 같음) 이다.

**제안 (구현하지 않음 — 같은 held-out 기록으로 고르면 test 데이터에 맞추는 것이라서):** `other` 에서 manipulated cloud 점 (target_free_points)
의 fusion voxel 안 node 를 뺀다 (T46 F1 `surface` 가 cloud 근거에서 target 자신의 점을 뺀 것과 같은 방향). 위 진단상 보호된 target seed 의
대부분을 막을 것으로 보이나, 잴 거면 기준을 먼저 고정하고 다른 기록 (dev) 에서 재야 한다.

## 4. 1 요청당 추가 ms

이 machine (CPU, osmesa, load ≈ 19–20, replay 3 worker 동시) 의 재생 측정. 서버 실측 아님.

| 단계 | median | p90 | 비고 |
|---|---:|---:|---|
| `select_target_surface` (LOCK graph + solve + code 영상, `keep_debug=True`) | 697–710 ms | 756–759 ms | SURF 와 같은 계산. SURF 서버 실측 (closed-loop, GPU 서버 동시 부하) 은 attention_surface stage median 1,663 ms 였다 |
| `hybrid_samples` | 0.7 ms | 0.7 ms | |
| identity guard (`hybrid_evidence`) | 3.0 ms | 3.4 ms | KD-tree, manipulated 점 수십–수백 |
| `surface_protect_split` (두 분할) vs `_ball_owned_split` 하나 | 9.4–10.6 ms vs 3.4–3.8 ms | 17.8–21.4 ms vs 5.8–6.2 ms | **보호 분류 추가 ≈ +6–7 ms** (median) |
| oracle memory 갱신 | 0 | | SURF 는 50–130 ms (CL1 impl) — hybrid 는 하지 않는다 |
| **hybrid − C1aN 합계** | ≈ select + 10 ms | | = SURF 의 attention_surface − memory 갱신 + 보호 ≈ 10 ms. render mask 를 켜면 + render (CL1: 105–160 ms) |

서버에서 profiler stage 이름은 SURF 와 같은 `attention_surface` (보호 분류 ms 는 `stats.target_free.surface_protect.split_ms` 와
`attention_surface_hybrid.ms`, 선택 ms 는 `attention_surface.ms`).

## verifier 가 알아야 할 것

- 새 flag (기본 = 지금 그대로): `--target-extent attention-surface-hybrid` (esdf `target_extent = "attention_surface_hybrid"`;
  `--target-exclusion ball-owned` · `--attention-source noun_target:L10H7` · `--target-field-policy exclude-*` 필수), `--hybrid-identity
  {both,instance,centre}` (esdf `surface_hybrid_identity`, 기본 `both`), `--robot-mask` 는 CL1 그대로 독립 (render 이면 graph = render mask,
  TSDF = 구 ∪ render; spheres 이면 graph = 구 mask, arm 이름 `top1-only-ns:N`). `--owned-*` 는 두 분할 모두에 같이 걸린다.
- 새 기록: chunk npz `target_free_protected_ijk` (보호가 걸린 frame), `attention_surface_codes_<camera>` (hybrid 에서도); summary
  `target_exclusion.attention_surface_hybrid` (`applied`, `why`, `identity.{instance,centre}`, `n_target_points`, `n_other_points`,
  `n_support_points_excluded`, `ms`) · `target_exclusion.surface_protect` · `target_exclusion.attention_surface` (CL1 과 같은 선택 기록);
  record meta · server metadata `target_extent` · `surface_hybrid_identity` (hybrid 일 때만). `stats.target_free.mode` 는 `"ball"`,
  `exclusion_method` 는 `"ball_owned"` 로 찍힌다 (C1aN 과 같다) — 출처는 위 키로 구별.
- hybrid 의 `n_kept_nontarget` · `nontarget_kept_ijk` 에는 **보호된 seed 가 포함**된다 (`protected_ijk` 가 그 부분집합). `n_target_owned` ·
  `n_ambiguous` · `n_unowned_*` 의 분류는 C1aN 규칙 (`est`) 의 것 — `n_unowned_removed` 만 보호 후 값.
- **GPU 확인 필요:** builder 의 torch 경로 (`_build_tiers` · `_ball_exclusion` 실제 seed 지움) 는 `.venv-ag3s` 에서 못 돌렸다.
  `tests/ag3s/test_t44_target_exclusion_owned.py` 등 torch test 를 GPU lock 을 잡고 돌려 기본 비트 동일과 hybrid 지움을 확인해야 한다.
- 재생산이 필요한 산출물: 없음. 재생 재측정: `python -m AG3S.ag3s.experiments.studies.cl2_hybrid_replay --cl1 $CL1 --out <dir> --workers 2`,
  `--summary`.
- 옛 기록과 호환: 기존 모드의 기록 키 집합은 그대로 (test). CL1 `attention_surface` 경로는 손대지 않았다 (`debug` 조건 한 줄만 확장).
- 이 checkout 은 다른 세션 (hurdle campaign verifier) 이 snapshot 중이다. 내 변경은 위 표의 6 개 기존 파일 (`IFG-CL2` 주석으로 표시된
  블록) + 신규 3 개 (`runtime/attention_surface_hybrid.py`, `experiments/studies/cl2_hybrid_replay.py`, `tests/ag3s/test_ifg_cl2_hybrid_extent.py`).

## 남은 위험

1. **target 보호 과다 (측정됨):** 선택 영역이 과일 일부만 덮으면 나머지 target 표면 node 가 `other` 가 되어 target seed 를 보호한다 (§3 진단).
   closed loop 에서 손가락 앞 target 장애물이 C1aN 보다 늘 수 있다 (지지면 위 평균 +7.5–10.4 / chunk, 최대 +150).
2. **전제:** §3 해석 2 · 3 — SURF 실패의 원인이 "안 보인 쪽 seed" 가 아닐 수 있다 (crate 지움과 같이 감). 이 hybrid 가 SURF 실패를 고친다는
   근거는 재생에 없다.
3. identity guard `instance` 는 이번 frame node 만 본다 (oracle 경로는 episode memory). 손이 가려 target node 가 적은 grasp 직전 frame 에서
   `no_node_near` · `no_majority` 로 꺼지기 쉽다 (실패 run 21.6 % off) — 꺼지면 C1aN 과 같으므로 안전 쪽이다.
4. 재생의 code 영상 불일치 26 chunk (최대 358 px) — float16 attention 기록 탓으로 추정 (서버는 float32), 확인 안 함.
5. 지연: SURF 와 같은 LOCK graph 비용 (≈ 0.7 s 이 CPU, 서버 실측 1.66 s) 이 그대로 든다 — C1aN 대비 timing 차이가 closed-loop 비교를 흐릴 수 있다.
6. 재생 GT 의 crate 는 box geom 별 hull (±5 mm) 이다. crate 가 mesh 이거나 다른 형상이면 crate 지움이 덜 세진다.

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다 -->
closed loop 에서 hybrid 는 crate 를 지우지 않고 (C1aN 과 같음) 이웃 접촉 seed 를 C1aN 보다 덜 지울 것이다. 성공률은 C1aN 과 비슷하거나,
target 보호 과다 (위험 1) 때문에 grasp 직전 HOLD 가 C1aN 보다 늘 수 있다. SURF 대비 개선은 기대하지만, 그 이유가 "ball 부피 제거" 라는
근거는 재생에 없다 (§3 해석 2).

---

## CL2b — "other" 근거에서 manipulated 기하 근처 node 를 뺀다 (2026-10-17 20:55 KST, ag3s-implementer)

> task: `AG3S/ag3s/docs/handoff/IFG-CL2b.task.md` · branch `feature/ag3s-development` @ `f15e5872` (commit 안 함, CL1 · CL2 미commit 변경 위)
> CPU 만, test 는 `.venv-ag3s` 만. GPU · server · rollout 없음. 잠금 `ifg_*.py` · 문서 · figure 수정 없음. 값은 task 가 고정한 1.0 (와 재현용 0) 만 쟀다.

### 평이한 요약

hybrid 가 보호 근거로 쓰는 "other" node (선택 밖 · 비지지면) 중 **identity guard 가 읽는 manipulated 기하 점에서 1 × fusion voxel
(10 mm) 안** 에 있는 것을 근거에서 뺀다. 이 보완이 hybrid 의 기본 (`esdf.surface_hybrid_other_exclude_voxels = 1.0`) 이고, 0 이면
IFG-CL2 와 비트 동일 (재생 1,430 chunk 전부 IFG-CL2 재생 출력과 값 일치). manipulated 기하가 없으면 빼지 않는다.

재생 결과 (같은 TSDF, SURF arm chunk): 보완은 의도한 대로 **target 과다 보호를 거의 없앤다** (지지면 위 target 남김: hybrid 121.1 / 65.9 →
114.6 / 56.3, C1aN 규칙 113.6 / 55.5) — 그러나 **이웃 보호도 거의 같이 없어진다** (이웃 지움: hybrid 14.0 / 13.7 → 36.3 / 34.8, C1aN 규칙
36.7 / 35.9). 보호된 seed 합계는 −93 % (11,466 → 782 · 16,689 → 1,188). 즉 1.0 의 hybrid 는 재생상 **C1aN 규칙과 거의 같다.**
원인 (scratch 진단): 이웃 보호를 잃은 chunk 에서는 **manipulated 기하 점 자체의 35 % 가 이웃 과일 위** 에 있어 (8,564 중 2,969, GT hull ±5 mm),
빠진 node 가 거의 다 이웃 표면이다 (9,496 이웃 vs 145 target). target 보호를 잃은 chunk 에서는 빠진 node 가 target 표면이다 (197 target · 0 이웃).

### 1. 바뀐 파일 (CL2b 분, `IFG-CL2b` 주석)

| 파일:줄 | 무엇이 |
|---|---|
| `AG3S/ag3s/runtime/attention_surface_hybrid.py:31-38` | module docstring 에 CL2b 절 |
| `attention_surface_hybrid.py:79-94` | 신규 `other_near_geometry(other, geometry_points, radius)` — node 가 기하 점에서 `radius` 이하 (≤) 이면 True. radius ≤ 0 · node 없음 · 기하 없음 → 전부 False |
| `attention_surface_hybrid.py:152` · `:159-162` · `:187-202` | `hybrid_evidence(..., other_exclude_voxels=0.0)`: guard 통과 후, V > 0 이면 `other_near_geometry(other, geometry_points, V × radius)` 를 other 에서 뺀다 (radius = 기존 인자 = fusion voxel). 기록 `other_exclude` {`voxels`, `radius_mm`, `n_before`, `n_excluded`[, `why: no_geometry`]}, `n_other_points` = 뺀 뒤 수, evidence `info.other_exclude_voxels`. V = 0 이면 evidence · 기록 키 그대로 (함수 기본값 0 — 재생 도구 · 직접 호출은 IFG-CL2) |
| `attention_surface_hybrid.py:205-206` | `__all__` + `other_near_geometry` |
| `AG3S/ag3s/config.py:1153-1157` | `EsdfConfig.surface_hybrid_other_exclude_voxels: float = 1.0` (hybrid 에서만 읽음) |
| `config.py:1322-1325` | 검증: bool · 비수 · 비유한 · 음수 → `AG3SConfigError` |
| `AG3S/ag3s/runtime/pipeline.py:749-751` · `:757-759` | `_hybrid_protect` 가 config 값을 `other_exclude_voxels=` 로 넘긴다 (기하 점 = 기존 `target_free_points`, guard 와 같은 점) |
| `AG3S/trajopt/serve_safe.py:1720-1725` | 새 flag `--hybrid-other-exclude-voxels V` (기본 None → config 1.0) |
| `serve_safe.py:919-920` | esdf 키 `surface_hybrid_other_exclude_voxels` (준 flag 만) |
| `serve_safe.py:2050-2055` | record meta `surface_hybrid_other_exclude_voxels` = 실제 적용 값 (flag 없으면 `EsdfConfig` 기본 1.0), hybrid 일 때만 |
| `serve_safe.py:2194-2199` | 거절: hybrid 아닌데 flag · 음수 / nan |
| `serve_safe.py:1260-1264` | 시작 warning 에 값 |
| `AG3S/trajopt/safe_policy.py:392-393` | server metadata `surface_hybrid_other_exclude_voxels` (hybrid 일 때만) |
| `AG3S/ag3s/experiments/studies/cl2_hybrid_replay.py:13` · `:52-54` · `:291-296` · `:314-315` · `:329-330` · `:359-360` · `:378-388` | 재생 변형 `hybrid:both+x1.0` (값 1.0 상수 `CL2B_EXCLUDE_VOXELS`, 탐색 없음), 기록 `other_exclude`, summary `n_protected_when_on_both+x1.0` · `other_exclude_both+x1.0`; CL2b 이전 run 파일은 그 변형을 건너뜀 |
| `tests/ag3s/test_ifg_cl2_hybrid_extent.py:14` · `:143-144` · `:182` · `:458-603` | 표 9 행; 기존 2 test 의 기대값에 새 meta 키 (record meta · server metadata 에 값이 실리므로 — 동작 변경 아님); 신규 8 test |

`curobo_builder.py` 는 바꾸지 않았다 (`SurfaceProtectEvidence.info` 가 이미 `stats.target_free.surface_protect` 로 실린다 — V > 0 이면 `other_exclude_voxels` 키가 같이 보인다).

### 2. test (`.venv-ag3s`, CPU)

```bash
cd /mnt/dev/AG3S_VLA-development
MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider \
    tests/ag3s/test_ifg_cl2_hybrid_extent.py tests/ag3s/test_ifg_cl1_attention_surface.py
#   → 47 passed (55 s)  [CL2 16 + CL2b 8 + CL1 23]
# IFG-CL2 §2 와 같은 넓은 묶음 (tests/ag3s + trajopt 13 파일)
#   → 2053 passed, 18 skipped, 0 failed (18 m 25 s)   [CL2 때 2045 + CL2b 8]
```

| 수락 조건 | test |
|---|---|
| 보완 규칙 | `test_cl2b_other_near_geometry_rule` (≤ 경계 9.9 / 10.0 / 10.1 mm, 0 · 기하 없음 → 안 뺌), `test_cl2b_target_own_surface_stops_protecting` (선택이 target 윗부분만 덮은 장면: 빠진 node 수 = target 자신의 비선택 node 수 정확히; target seed 보호 ≤ 10 % 로; 20 mm 떨어진 이웃의 보호 seed 수 불변; removed 는 늘기만 하고 여전히 ⊆ `ball_owned`) |
| 0 = IFG-CL2 비트 동일 | `test_cl2b_zero_is_cl2_bit_identical` (kwarg 없음 vs 0: evidence · info · 기록 (ms 제외) · split `removed`/`sign`/`protected` 동일), `test_cl2b_pipeline_passes_the_config_value` (config 0 → builder 로 가는 other = samples.other 그대로, 기록에 `other_exclude` 없음; 기본 → 1.0 · 10 mm, n_excluded > 0) + **재생: 0 의 네 변형 (both · instance · centre · noguard) 이 IFG-CL2 재생 출력과 1,430/1,430 chunk 값 일치** (ms 제외, code parity 포함) |
| manipulated 기하 없음 | `test_cl2b_no_geometry_no_exclusion` (centre 규칙: 보호 적용 · other 그대로 · `why: no_geometry`; instance/both 는 IFG-CL2 대로 `identity_off (instance:no_points)`) |
| config · CLI · 기록 | `test_cl2b_config_and_cli` (기본 1.0, round trip, −1 · nan · inf · True · "1" 거절, flag 없으면 esdf 키 없음, 0 → 키 · meta 0.0), `test_cl2b_cli_refusals` (hybrid 없이 flag · attention-surface 와 flag · 음수 · nan), `test_cl2b_server_metadata_value` |

**돌리지 못한 것:** torch builder test (18 skipped 에 포함). builder 코드는 이번에 바꾸지 않았다.

### 3. offline 재생 (IFG-CL1 68 run, pre-grasp chunk 1,430)

- 도구: `cl2_hybrid_replay.py` (위 변경), 입력 · 규칙 · GT 는 §3 그대로. `--runs` = IFG-CL2 재생과 같은 68 run (C1aN 34 · SURF 34).
- 출력: `/mnt/dev/AG3S_VLA-storage/results/development/20261017T202707+0900_f15e587290f6_7024ee2643b6/cl2b_replay/<run>.json` · `summary.json`
  (log `../cl2b_replay.log`). 3 worker, 약 25 분.
- `hybrid (both)` 행 = 보완 0 (= IFG-CL2, 위 정합 확인으로 IFG-CL2 표와 같은 값). `hybrid (+exclude 1.0)` = 같은 `both` guard + 보완 1.0.
  C1aN chunk 는 code 영상이 없어 hybrid 재생 불가 (IFG-CL2 와 같음).

chunk 당 평균 (괄호 = 중앙값):

| 그룹 (chunk) | 변형 | target 남김 (전부) | target 남김 (지지면 위) | target 남김 (공 안) | 이웃 과일 지움 | crate 지움 | keptseeds (lead 방식) |
|---|---|---:|---:|---:|---:|---:|---:|
| C1aN 성공 34 run (600) | 기록 = C1aN 규칙 재생 | 126.5 (117) | 50.0 (3) | 47.3 (1) | 39.8 (0) | 0 | (0) |
| SURF 실패 6 run (352) | 기록 (SURF) | 140.4 (148) | 35.5 (24) | 30.2 (22) | 39.0 (0) | 211.2 (0) | (123.5) |
| | C1aN 규칙 (같은 TSDF) | 218.5 (188) | 113.6 (71) | 81.8 (56) | 36.7 (0) | 0 | |
| | hybrid (both) = exclude 0 | 226.1 (196) | 121.1 (80) | 89.3 (59) | 14.0 (0) | 0 | |
| | **hybrid (+exclude 1.0)** | **219.6 (190)** | **114.6 (71)** | **82.8 (58)** | **36.3 (0)** | 0 | |
| SURF 성공 28 run (478) | 기록 (SURF) | 93.3 (80) | 16.8 (14) | 16.8 (14) | 2.1 (0) | 0 | (79) |
| | C1aN 규칙 (같은 TSDF) | 132.0 (108) | 55.5 (3) | 54.6 (0) | 35.9 (0) | 0 | |
| | hybrid (both) = exclude 0 | 142.4 (123) | 65.9 (35) | 65.0 (35) | 13.7 (0) | 0 | |
| | **hybrid (+exclude 1.0)** | **132.8 (109)** | **56.3 (3)** | **55.4 (1)** | **34.8 (0)** | 0 | |

같은 chunk 짝 차이 (+exclude 1.0 − 비교 대상; 실패 / 성공):

| 지표 | − C1aN 규칙 | − hybrid (exclude 0) |
|---|---|---|
| target 남김 (지지면 위) | 평균 +1.0 / +0.8, 늘어난 chunk 17.3 % / 10.5 %, 최대 +30 / +38 | 평균 −6.5 / −9.6, 줄어든 chunk 15.9 % / 26.8 %, 최소 −150 / −143 (늘어난 chunk 0) |
| 이웃 과일 지움 | 평균 −0.4 / −1.0, 줄어든 chunk 5.4 % / 16.9 %, 최소 −26 / −28 | 평균 +22.2 / +21.1, 늘어난 chunk 6.2 % / 6.1 %, 최대 +401 / +443 (줄어든 chunk 0) |

| 그룹 | 보호 on chunk | 보호 seed 합 (0 → 1.0) | 보호 seed > 0 인 on chunk (0 → 1.0) | 뺀 node 가 있는 on chunk | 뺀 node (평균 / p90 / on chunk 당 other node 중앙값) |
|---|---:|---:|---:|---:|---|
| SURF 실패 | 276 (78.4 %, 꺼진 이유 IFG-CL2 표와 같음) | 11,466 → 782 (−93 %) | 42.0 % → 30.4 % | 29.0 % | 18.2 / 31 / 10,580 |
| SURF 성공 | 463 (96.9 %) | 16,689 → 1,188 (−93 %) | 45.8 % → 28.7 % | 31.3 % | 12.9 / 7 / 10,801 |

`no_geometry` (기하 없이 보호가 켜진 frame) 은 0 chunk (`both` 규칙은 기하가 없으면 guard 가 끈다).

SURF 실패 6 run 별 (chunk 평균; 지지면 위 target 남김 · 이웃 지움, 각각 기록 / C1aN 규칙 / hybrid 0 / **hybrid 1.0**):

| run | 실패 | chunk | 보호 on | target 남김 | 이웃 지움 |
|---|---|---:|---:|---|---|
| SURF_ep1769_s17691 | no_grasp | 73 | 0.73 | 4.0 / 44.9 / 46.2 / **45.3** | 0.0 / 5.1 / 4.9 / **4.9** |
| SURF_ep1893_s18931 | no_grasp | 55 | 0.82 | 28.9 / 80.0 / 86.0 / **81.3** | 2.6 / 0.1 / 0.1 / **0.1** |
| SURF_ep391_s3911 | no_grasp | 59 | 0.83 | 28.2 / 45.3 / 52.3 / **45.6** | 0.0 / 0.0 / 0.0 / **0.0** |
| SURF_ep485_s4851 | not_placed | 15 | 0.73 | 14.3 / 73.7 / 74.3 / **73.7** | 0.0 / 467.2 / 197.7 / **459.7** |
| SURF_ep623_s6231 | no_grasp | 75 | 0.71 | 47.8 / 130.1 / 149.3 / **132.7** | 180.8 / 73.7 / 21.4 / **73.4** |
| SURF_ep764_s7641 | no_grasp | 75 | 0.87 | 68.5 / 250.3 / 255.3 / **251.1** | 0.6 / 0.0 / 0.0 / **0.0** |

**왜 이웃 보호가 사라지나 (scratch 진단, 저장 산출물 아님 — `select_target_surface` 를 다시 돌려 뺀 node 를 GT hull ±5 mm 로 분류):**

| chunk 묶음 | chunk | 뺀 other node 의 GT | manipulated 기하 점의 GT |
|---|---:|---|---|
| 이웃 지움이 늘어난 chunk 전부 (SURF 5 run: ep485 · ep623 · ep1698 · ep1719 · ep603) | 51 | 이웃 9,496 · target 145 · crate 0 · 기타 0 | target 5,595 · **이웃 2,969 (35 %)** · 기타 0 |
| target 남김이 줄어든 chunk (k % 4 == 0 표본, 이웃 지움이 늘지 않은 것) | 29 | target 197 · 이웃 0 · 기타 12 | target 2,874 · 이웃 1 · 기타 190 |

→ 보완은 "기하 점 = target" 을 전제한다. manipulated 기하 (`target_free_points`) 가 이웃 과일까지 덮은 frame 에서는 그 이웃의 표면 node 가
빠져, IFG-CL2 가 지키던 이웃 seed 가 다시 지워진다 (= C1aN 규칙으로 돌아감). 기하가 target 에만 있는 frame 에서는 의도대로 target 자신의 node 만 빠진다.

### 4. 추가 ms (재생 측정, 이 CPU, 3 worker 동시)

identity guard + 보완 (`hybrid_evidence`, both): median 2.9 → 5.6–5.8 ms, p90 3.3 → 6.7 ms (≈ +2.7 ms — 기하 점 KD-tree 에 other node ≈ 10k 질의).
분할 (`surface_protect_split`) 은 같은 함수라 차이 없음 (both median 9.6–11.0 ms).

### 5. 허들 2 차 server flag (C1aN + hybrid; robot mask 두 줄)

IFG-CL1 C1aN 서버 (`.../verify/IFG-CL1/runs/server_8293_20261016T151539/server_proc.txt`) 의 flag 에 4 개를 더한 것. `--port` · `--run-dir` ·
`--record-constraints` 는 campaign 것으로 바꾼다. 아래 두 줄은 `build_parser` → `reject_bad_flag_combinations` → `target_exclusion_section` →
`AG3SConfig` 를 `.venv-ag3s` 에서 통과함을 확인했다 (서버는 띄우지 않음). 기본값을 일부러 적었다 (기록에서 바로 읽히게).

```
# spheres
... -m AG3S.trajopt.serve_safe --config pi05_rby1_randomized_pick_place_16d_lora --checkpoint <ckpt 29999> --model-xml <snapshot model_transport.xml> --port <P> --esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --obstacle-margin 0.010 --held-body-cover --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20 --to-fast-eval --esdf-host-mirror --target-exclusion ball-owned --owned-max-dist-voxels 3 --attention-split-radius-mm 12 --attention-split-ratio-source raw --target-permission-strict --attention-source noun_target:L10H7 --target-extent attention-surface-hybrid --hybrid-identity both --hybrid-other-exclude-voxels 1.0 --robot-mask spheres --run-dir <RUN> --record-constraints <RUN>/.../server_constraints
# render
(같은 줄) ... --target-extent attention-surface-hybrid --hybrid-identity both --hybrid-other-exclude-voxels 1.0 --robot-mask render --run-dir <RUN> --record-constraints <RUN>/.../server_constraints
```

서버 log guard: `!!! AG3S IFG-CL2 — target_extent = attention_surface_hybrid (identity both) … IFG-CL2b: manipulated 기하 점에서 1.00 fusion voxel 안의 node 는 보존 근거에서 뺀다`;
record meta · server metadata `surface_hybrid_other_exclude_voxels: 1.0`; chunk summary `target_exclusion.attention_surface_hybrid.other_exclude`.

### 남은 위험 (CL2b)

1. **재생상 1.0 의 hybrid ≈ C1aN 규칙** (target 남김 +0.8–1.0, 이웃 지움 −0.4–1.0 / chunk). closed loop 에서 C1aN 과 구별되는 효과가 작을 수 있다 —
   허들 2 차에서 C1aN 과 차이가 안 나는 것이 "hybrid 가 무효" 인지 "보완이 보호를 지웠다" 인지는 `other_exclude.n_excluded` 와 `surface_protect.n_protected` 로 나눠 봐야 한다.
2. **manipulated 기하가 이웃을 덮는 frame** (위 진단 51 chunk, 기하 점의 35 % 가 이웃 위) 에서 이웃 보호가 사라진다. 그 frame 에서 identity guard
   (`instance`: 기하 점의 ≥ ½ 이 선택 node 위) 는 통과했다 — 기하 점 다수는 target 이기 때문. 이 규칙은 기하 오염을 막지 않는다.
3. 이 재생은 IFG-CL1 (이제 dev) 기록 위의 측정이다. 값 1.0 은 재생 전에 고정됐고 재생으로 바꾸지 않았다; 0 / 1.0 외 값은 재지 않았다.
4. IFG-CL2 의 위험 3–6 (guard 가 grasp 직전 꺼짐, code parity 26 chunk, LOCK graph 지연, crate GT 근사) 은 그대로다.
