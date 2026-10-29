# IFG-SPHERE — 구현 (1단계: 남은 target voxel 원인 분해와 설계안)

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead
> 1단계는 분석 script 와 설계안만이다. 서빙 module (`AG3S/ag3s/**`, `AG3S/trajopt/**` 의 기존 파일) 은 고치지 않았다.

## 0. 분류 정의 (계산 전에 고정, 2026-10-22 02:40 KST)

이 절은 숫자를 보기 전에 썼다. 결과 절은 아래 정의를 바꾸지 않는다.

### 대상

- 입력: run `20261019T132719+0900_f15e587290f6_11408fd6d27f` 의 `verify/IFG-APPLE-SPLIT` (arm SPL, 24 장면).
  서버 flag 는 `runs/server_8293_20261020T015105/server_proc.txt`: `--target-exclusion ball-owned --owned-max-dist-voxels 3`,
  `--nontarget-evidence` 없음 (= `proximity`), `--owned-ambiguous` 없음 (= `keep`), `--target-extent` 없음 (= `estimated`,
  표면 보호 없음), 거친 target-free 층 없음 (chunk 기록 `esdf_stats.target_free.coarse.mode == "off"`).
- 요청: summary `grasp.state` 가 `latched` 또는 `closing` 이고 `ball_owned` 미세 target-free 층과 제거 trace 가 있는 요청
  (IFG-AUDIT-LTE 의 `rr1` 과 같은 집합).
- 재구성: `AG3S.trajopt.experiments.target_free_replay.rebuild_target_free` (서버의 `_nontarget_points` · `_ball_owned_split` ·
  `_ball_candidates` · `_force_sign_on_pure_voxels` 를 그대로 부른다). 요청마다 replay 의 trace 집합이 기록과 같은지
  (`recorded_split_equal`), 값 격자 차이 (> 0.6 mm voxel 수) 를 같이 보고한다. 분류의 split 값 (`d_t`, `d_o`, `sign`) 은
  같은 함수 `_ball_owned_split` 를 같은 입력으로 다시 불러 얻는다.
- 남은 GT target voxel (IFG-AUDIT-LTE 의 `free_residual_target_voxels` 와 같은 정의): GT target 중심 (target collision mesh
  AABB 중심) ±6 cm 상자 안의 미세 격자 voxel 중 target-free 값 `F ≤ vs/2` (2.5 mm) 이고 GT target 규칙
  (`d_target ≤ 5 mm` 이고 `d_target ≤ d_다른 모든 geom`) 을 만족하는 것. GT = run 의 MuJoCo model, planning-row qpos.
  수치가 LTE 의 중앙값 (latched 251, closing 516.5, stale 제외) 과 같은지 먼저 확인한다.
- seed = 주 미세 층 `esdf_distance == 0` 인 voxel (replay 와 같은 정의). 남은 seed = target-free 층 `F == 0`.

### 분류 (상호 배타, 위에서부터 첫 번째로 맞는 것)

요청 단위:

| # | 이름 | 조건 | 코드 경로 |
|---|---|---|---|
| S | `stale` | 공의 표면 맞춤 중심 (`ball.fit.centre_m`) 이 GT target 중심에서 30 mm 넘게 떨어진 요청 — 그 요청의 남은 voxel 전부 | `pipeline._target_ball` 이 옛 frame 기하 (anchor · occluded) 의 점으로 공을 만든다. 안쪽 분류는 따로 보고 |

voxel 단위 (`c` = voxel 중심, 공 = 기록의 `ball` 중심 · 반지름 `R` · `z_min`):

| # | 이름 | 조건 | 코드 경로 |
|---|---|---|---|
| 1 | `outside_ball_centre` | `|c − 공 중심| > R` 이고 `|c − GT 중심| ≤ R` (같은 반지름이라도 중심이 GT 면 들어온다) | `CuroboFieldBuilder._ball_candidates` (`curobo_builder.py:2078`) |
| 2 | `outside_ball_radius` | `|c − 공 중심| > R` 이고 `|c − GT 중심| > R` | 같은 곳 · `pipeline._target_ball` 반지름 (`pipeline.py:3588`) |
| 3 | `below_zmin` | 공 안, `c_z < z_min` | `_ball_candidates` 의 `centres[:, 2] >= ball.z_min` (`curobo_builder.py:2095`) |
| 4 | `kept_nontarget` | 공 안 seed, split 이 `nontarget` (`d_o < d_t`, tie 밖) | `_ball_owned_split` (`curobo_builder.py:417`) |
| 5 | `kept_ambiguous` | 공 안 seed, `|d_t − d_o| ≤ tie` (5 mm) | 같은 곳, `owned_ambiguous = keep` |
| 6 | `kept_unowned` | 공 안 seed, unowned 이고 `unowned = keep` | 같은 곳 (이 run 은 `remove` 라 0 이어야 한다) |
| 7 | `surface_protect` | `protected_ijk` 에 든 seed | `surface_protect_split` (`curobo_builder.py:688`) — 이 run 은 꺼짐, 0 이어야 한다 |
| 8 | `removed_sign_left` | 지운 seed 인데 `F ≤ vs/2` (음수로 남음) | `_force_sign_on_pure_voxels` (`curobo_builder.py:1978`) 의 순수 판정 `|d| > 1.5 vs` 실패 — 7.5 mm 안에 남은 seed 가 있다 |
| 9 | `interior_not_sign_set` | 공 안 non-seed, `F < 0`, split 의 sign 후보가 아님 (`d_t ≥ d_o − tie`) | `_ball_owned_split` 의 `sign = removed ∪ (non-seed ∧ d_t < d_o − tie)` |
| 10 | `interior_impure` | 공 안 non-seed, `F < 0`, sign 후보인데 `|F| ≤ 1.5 vs` | `_force_sign_on_pure_voxels` |
| 11 | `other` | 위 어느 것도 아님 (예: 남은 seed 인데 split 분류가 없는 것) | 0 이 아니면 재구성 오류로 본다 |

과제 문서의 예시 중 이 run 에서 꺼져 있는 경로는 분류에 넣되 0 으로 나와야 한다: `surface_protect` (7),
`nontarget_surface_evidence` 반지름 제한 (F1 꺼짐 — 분류 행 없음, 대신 sub-label 로 근거 점이 target 인지 본다),
거친 층 (꺼짐). 창 밖 · 창 경계는 voxel 분류와 따로 점 단위로 센다 (아래).

보조 라벨 (분류를 바꾸지 않고 원인을 한 단계 더 가른다):

- 4 · 5 · 9 의 **근거 점 class**: `d_o` 를 만든 가장 가까운 non-target 근거 점 (`_nontarget_points` 의 cloud 점) 의 GT class —
  그 점에서 10 mm 안의 가장 가까운 GT class (`target` · `neighbour` (가장 가까운 이웃 과일) · `fruit_other` · `table` ·
  `crate` · `obstacle` · `other`), 10 mm 안에 없으면 `none`. `target` 이면 "target 자신의 표면 점이 non-target 근거로 남아
  target seed 를 보존시켰다" (T46 F1 이 겨냥한 원인) 이다.
- 8 · 10 의 **가장 가까운 남은 seed 의 분류** (1–7 중 무엇 때문에 남은 seed 가 순수 판정을 막았나) 와 그 seed 가 GT target 인지.

창 · 합성 (점 단위, voxel 분류와 별도): GT target 표면점 48 방향 (IFG-AUDIT-LTE 와 같은 방법) 중 다른 물체에서 15 mm 넘게 떨어진 점에서
(a) 미세 target-free 격자 밖인 점 수, (b) 합성 `target_free_distance < 5 mm` 인데 미세 값 `F ≥ 5 mm` 인 점 수 (`window_edge` 가 답함).

### 표

- 단계별 (latched · closing) × 장면 유형 (neighbour = 가장 가까운 이웃 과일 collision 기하가 10 cm 안 / clear) 의 분류 합계와 비율,
  요청당 중앙값.
- IFG-PRECHECK-A raw_v2 의 SPLIT 9 장면 (실패 6 + 대조 3) 과 NBC ep211 의 `src == "target"` 이고 `layer == "target_free"` 인 행
  (`traj == "ref"` = active_ref, `traj == "out"` = binding_out 을 따로). 행이 "닿은" voxel: 행 위치 `p` 의 trilinear 8 꼭짓점 중
  `F < 0` 이 있으면 가중 기여 (`w·F`) 가 가장 음수인 꼭짓점, 없으면 `p` 에서 가장 가까운 남은 seed (`F == 0`). 그 voxel 의 분류 ·
  보조 라벨 · GT target 여부. 211 k11–14, 979, 320 은 따로 표로 낸다.

## 1. 무엇을 했나 (평이한 요약 먼저)

SPLIT 24 장면의 쥐기 전 요청 661 개를 서버와 같은 코드로 offline 재구성해서, target-free 층에 남은 GT 사과 voxel 하나하나가
어느 판단 때문에 남았는지 위 분류로 나눴다. 결론은 **공의 크기 · 중심이 아니라 "non-target 근거 점" 이 사과 자신의 표면 점이라는 것** 이다.
그 점 때문에 공 안 seed 가 ambiguous / non-target 으로 보존되고, 사과 내부 voxel 이 부호 교정 후보에서 빠지고, 보존된 seed 가
옆의 지운 seed 의 음수 부호까지 막는다. 이것을 겨냥한 옵션은 이미 있다 (T46 F1 `--nontarget-evidence surface`, T46c 에서 채택) —
**그런데 IFG-APPLE-SPLIT 서버는 그 flag 없이 돌았다** (`server_proc.txt`, `t46_stats_check.json` `n_evidence 0`).
F1 을 같은 기록에 offline 으로 걸면 남은 voxel 중앙값이 251 → 90 (latched), 520 → 136 (closing), target 출처 활성 행이 785 → 133 이다.
F1 뒤에 남는 것은 주로 지원면 근처 (공의 `z_min` 아래 bottom cap 과, 그 cap 의 seed 때문에 부호가 남은 띠) 다.

서빙 코드는 고치지 않았다. 새 파일은 분석 script 하나다.

## 2. 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/ag3s/experiments/studies/ifg_sphere_attrib.py` (새, 분석 전용) | `collect` (요청별 분류 · 행 분류 · offline 변형), `attrib` (변형이 남기는 것의 분류 + P2 support-aware sign 반사실), `report` · `report-attrib` (집계), `figure` (run dir 에만 쓴다) | IFG-SPHERE 1단계 |
| `AG3S/ag3s/docs/handoff/IFG-SPHERE.impl.md` (이 파일) | 분류 정의 (§0, 계산 전) · 결과 · 호출 경로 · 설계안 | |

서빙 module (`AG3S/ag3s/**`, `AG3S/trajopt/**` 의 기존 파일) 변경 없음. `AG3S/ag3s/docs/figures/**` 에 쓰지 않았다 (figure 는 run dir).

## 3. 단위 검증 (재구성 충실도)

```bash
cd /mnt/dev/AG3S_VLA-development
RUN=/mnt/dev/AG3S_VLA-storage/results/development/20261019T132719+0900_f15e587290f6_11408fd6d27f
PC=/mnt/dev/AG3S_VLA-storage/results/development/20261022T001543+0900_8e200ada6d53_f62e49bedd53/verify/IFG-PRECHECK-A/raw_v2
OUT=/mnt/dev/AG3S_VLA-storage/results/development/20261022T023623+0900_055f2ea2f00b_8e1a8f54bb5e/verify/IFG-SPHERE-ATTRIB
E="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src"
env $E .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_sphere_attrib collect --run $RUN --precheck $PC --out $OUT/raw \
    --variants F1,F2,F1F2,CORE0,CORE5,F1CORE5 [--eps ...]            # 2 worker 로 episode 를 나눠 돌림 (~11 분)
env $E .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_sphere_attrib collect --run $RUN --kit IFG-APPLE-NBC --arm SPLT --eps 211 \
    --precheck $PC --out $OUT/raw --variants F1,F2,F1F2,CORE0,CORE5,F1CORE5
env $E .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_sphere_attrib attrib --run $RUN --precheck $PC --out $OUT/raw --variant F1   # 와 OFF
env $E .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_sphere_attrib report --raw $OUT/raw --out $OUT/summary.json
env $E .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_sphere_attrib report-attrib --raw $OUT/raw --out $OUT/summary_attrib.json --variants OFF,F1
env $E .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_sphere_attrib figure --raw $OUT/raw --summary $OUT/summary.json --fig $OUT/fig
```

| 점검 | 결과 |
|---|---|
| 요청 | latched · closing 669 중 661 측정. 8 개 건너뜀: `target_points` 가 기록에 없음 (`has_target False`, ep1313 k18 · ep320 k22/53–55/63–65) — split 재구성 불가 |
| replay trace = 기록 trace (removed · ambiguous · nontarget_kept) | 660 / 661 (다른 1 개: ep211 k63 closing) |
| replay 값 격자 vs 기록, > 0.6 mm voxel | 6 요청 56 voxel (ep211 k63 45, ep526 5 요청 1–4), 최대 5.0 mm. 나머지 655 요청 0 (float16 기록 정밀도 0.24 mm) |
| 분류 `other` · `kept_unowned` · `surface_protect` | 0 · 0 · 0 (꺼진 경로가 0 으로 나옴 — 정의 §0 대로) |
| `d_o` 를 만든 근거 점 재구성 | 매 요청 `cKDTree` 최근접 거리 = `_ball_owned_split` 의 `d_o` (assert, 오차 1e-9) |
| LTE 와 같은 수치인가 (남은 voxel 중앙값, stale 제외) | latched **251** (LTE 251) · closing **520** (LTE 516.5; 집합 차이 = 위 8 요청) · recall 0.79 / 0.58 (LTE 0.788 / 0.585) |
| PRECHECK-A 행 재현 | 행 위치의 미세 target-free trilinear 값 − 기록 `d_mm` 중앙값 0.0 mm. target 출처 active_ref 행 1,178 (PRECHECK 1,188 − 건너뛴 요청 10) |

서빙 코드를 바꾸지 않아 기존 test 는 돌리지 않았다.

## 4. 결과 — 분해 표

모든 수치는 run `20261022T023623+0900_055f2ea2f00b_8e1a8f54bb5e/verify/IFG-SPHERE-ATTRIB/` 의 `summary.json` · `summary_attrib.json`.

### 4-1. 단계별 (record, F1 꺼짐 = 서버가 실제로 쓴 층)

| 분류 | latched (442 요청, 151,333 voxel) | closing (169 요청, 83,510 voxel) | 요청당 중앙값 latched / closing |
|---|---|---|---|
| S `stale` | 0.14 % | 1.75 % | — |
| 1 `outside_ball_centre` | 0 | 0.02 % | 0 / 0 |
| 2 `outside_ball_radius` | 0 | 0 | 0 / 0 |
| 3 `below_zmin` | 10.9 % | 11.0 % | 35 / 53 |
| 4 `kept_nontarget` | 4.2 % | 5.1 % | 0 / 8 |
| 5 `kept_ambiguous` | 18.9 % | 21.2 % | 34 / 103 |
| 8 `removed_sign_left` | 29.5 % | 25.5 % | 90 / 142 |
| 9 `interior_not_sign_set` | 27.9 % | 28.3 % | 43 / 170 |
| 10 `interior_impure` | 8.5 % | 7.2 % | 22 / 39 |
| 6 · 7 · 11 | 0 | 0 | |
| 남은 voxel 중앙값 / 요청 (stale 제외) | **251** | **520** | |

- **공 밖 (1 · 2) 은 사실상 0** 이다. 맞춤 중심 오차 중앙값 2.01 mm, 공 반지름 54.7 mm (맞춤 33.5 mm + 여유) 라 사과 전체가 공 안에 든다.
  "중심 · 반지름" 은 recall 0.59 의 원인이 아니다.
- **근거 점 class (4 · 5 · 9, stale 제외 두 단계 합):** 가장 가까운 non-target 근거 점이 GT target 인 것 **122,295 / 122,797 = 99.6 %**,
  neighbour 266, none 234, table 2. 즉 공 안에서 사과 voxel 을 지키는 "non-target 근거" 는 거의 전부 **사과 자신의 관측 표면 점** 이다.
  근거 출처: `proximity` 500 요청 · `indices` 161 요청 — 두 경로 모두 사과 표면 점을 근거에 남긴다 (fused cloud 10 mm 간격 vs proximity 5 mm).
- **공 안 보존 seed 전체 (잔여만이 아니라):** `kept_nontarget` 16,189 중 GT target 10,599 (65 %), `kept_ambiguous` 64,145 중 46,356 (72 %).
- **부호가 남은 voxel (8 · 10) 의 가장 가까운 남은 seed:** `kept_ambiguous` 53,515 · `below_zmin` 27,625 · `kept_nontarget` 3,506 · 공 밖 124.
  즉 8 · 10 도 2/3 는 같은 근거 오염의 2차 효과이고, 1/3 은 지원면 근처 (`z_min` 아래 seed) 효과다.
- **창 · 거친 층:** GT target 표면점 20,531 중 미세 격자 밖 0, 합성 < 5 mm 인 5,294 점은 전부 `target_free` tier 가 답했다
  (`window_edge` 0, 거친 target-free 층은 꺼짐). 창 · 거친 층은 원인이 아니다.
- **`z_min`:** GT 중심 − `z_min` 중앙값 23.4 mm. 사과 반지름 ≈ 32–33 mm 이므로 `z_min` 은 사과 바닥 위 약 9 mm — 그 cap (높이 ~9 mm) 이 3 이다.

### 4-2. 장면 유형별

| | neighbour | clear |
|---|---|---|
| latched | 442 요청, 위 표 그대로 | 13 요청, 남은 voxel 없음 (창 안에 GT target 없음) |
| closing | 169 요청, 위 표 그대로 | 37 요청, 같음 |

clear 50 요청은 **전부 ep320 k18–74, 사과가 탁자에서 떨어진 뒤** (GT 중심 z = 0.03 m) 다. 서버는 occluded 기억으로 옛 자리에 공을 계속
만든다 (맞춤 중심 − GT 638–830 mm, `stale`). 이 kit 의 나머지 요청은 모두 neighbour 유형 (이웃 과일 10 cm 안) 이라 "이웃 있음 / 없음" 의
비교는 이 자료로는 나오지 않는다. stale 요청: ep211 10 (closing 9 · latched 1, 쥐다 놓친 뒤), ep320 51 (k17 + 떨어진 뒤).

### 4-3. PRECHECK-A target 출처 활성 행 (target-free 층)

행이 닿은 voxel 의 분류 (`분류|근거 class 또는 그 voxel 을 막은 seed 분류`). GT target = 그 voxel 이 엄격한 GT target 규칙 (≤ 5 mm) 을 만족.

| 집합 | 행 | 분류 | GT target | F1 뒤 활성 |
|---|---|---|---|---|
| SPLIT active_ref 전체 (9 장면) | 1,178 | kept_ambiguous\|target 402 · **stale 393** (ep320 k17) · kept_nontarget\|target 132 · removed_sign_left\|kept_ambiguous 124 · interior_not_sign_set\|target 81 · interior_impure\|kept_ambiguous 33 · removed_sign_left\|below_zmin 8 · removed_sign_left\|kept_nontarget 5 | 789 | 526 (stale 393 + 133) |
| 그중 stale 아님 | 785 | 위에서 stale 뺀 것 — 근거 오염 계열 777 / 785 (99 %), 지원면 계열 8 | | **133** (−83 %) |
| SPLIT binding_out 전체 | 262 (h ≤ 1 mm 223) | kept_ambiguous\|target 88 · removed_sign_left\|kept_ambiguous 55 · below_zmin 32 · interior_not_sign_set 17 · removed_sign_left\|below_zmin 10 · … | 109 | 58 |
| **211 k11** (HOLD) | 24 | interior_not_sign_set\|target 14 · removed_sign_left\|kept_ambiguous 5 · kept_ambiguous 3 · kept_nontarget 2 | 23 | 8 |
| **211 k12** (HOLD) | 18 | kept_ambiguous 6 · kept_nontarget 5 · removed_sign_left\|kept_ambiguous 4 · interior_not_sign_set 3 | 18 | 5 |
| **211 k13** | 18 | interior_not_sign_set 7 · removed_sign_left\|kept_ambiguous 4 · kept_ambiguous 4 · kept_nontarget 3 | 18 | 6 |
| **211 k14** | 42 | kept_ambiguous 18 · interior_not_sign_set 18 · removed_sign_left\|kept_ambiguous 6 | 25 | 0 |
| 211 전체 | 332 | kept_nontarget 132 · kept_ambiguous 121 · interior_not_sign_set 47 · … (전부 근거 = target) | 218 | 123 (k58–59 regrasp 100, k11–13 19) |
| NBC 211 (k10–16 포함) | 112 | interior_not_sign_set 43 · kept_ambiguous 38 · removed_sign_left 19 · kept_nontarget 10 · impure 2 (k11–14 은 SPLIT 과 같은 행 · 같은 분류) | | 19 |
| **979** k11 · k12 | 25 · 33 | kept_ambiguous\|target 58 / 58 | 20 (k12 0: 닿은 seed 가 collision mesh 에서 5 mm 넘게 떨어짐) | 0 |
| **320** k16 | 42 | kept_ambiguous 25 · interior_impure 12 · removed_sign_left 5 | 41 | 0 |
| **320** k17 | 393 | **stale** — 맞춤 중심이 GT 에서 54.4 mm (closing 중 사과가 밀림, 다음 요청에 떨어짐). 행 341 은 공 밖 voxel, 52 는 공 안 | 301 | 393 (F1 무효) |
| 1313 k51 · 53 · 54 | 144 · 3 · 15 | kept_ambiguous 66 · removed_sign_left\|kept_ambiguous 49 · interior_not_sign_set 34 · … | 94 | 10 (전부 below_zmin 계열) |
| 126 (대조) | 119 | kept_ambiguous 67 · removed_sign_left 44 · impure 8 | 50 | 0 |

F1 뒤에 남는 non-stale active_ref 133 행: kept_nontarget\|target 65 + removed_sign_left\|kept_nontarget 19 (ep211, F1 의 이웃 보호 /
껍질 밖에 남은 사과 점) · below_zmin 23 + removed_sign_left\|below_zmin 7 · kept_ambiguous 17 · interior 2.
**HOLD 의 변화는 재지 않았다** (TO 재풀이 필요 — 2단계 offline 검증 항목). 위 "F1 뒤 활성" 은 행 값 `h` 에 층 값 차이만 더한 것이다
(같은 궤적, 재풀이 없음).

### 4-4. offline 변형 (서버의 split 함수를 같은 입력으로 다시 부름, 서빙 아님)

남은 voxel 중앙값 (stale 제외) · 이웃 과일 표면 voxel (±6 cm 상자, GT 이웃 표면 2.5 mm 안이고 기록에서 남아 있던 것) 의 보존 · 지운 seed 중
이웃 라벨 비율 (LTE composition 규칙):

| 변형 | latched 중앙값 | closing 중앙값 | recall latched / closing | 이웃 표면 voxel 잃음 latched / closing | 지운 seed 의 이웃 라벨 | 탁자 표면 voxel 잃음 | active_ref (non-stale 785) |
|---|---|---|---|---|---|---|---|
| record (OFF) | 251 | 520 | 0.79 / 0.58 | — (24,647 / 12,303) | 1.19 % (pooled) | — | 785 |
| **F1** (`--nontarget-evidence surface`) | **90** | **136** | 0.91 / 0.85 | **3 (0.01 %) / 7 (0.06 %)** | 1.25 % / 0.74 % | 0 | **133** |
| F2 (`--owned-ambiguous remove`) | 169 | 316 | 0.90 / 0.82 | 997 (4.0 %) / 60 (0.5 %) | 1.72 % / 0.85 % | 0 | 258 |
| F1F2 | 86 | 122 | 0.92 / 0.87 | 997 (4.0 %) / 60 (0.5 %) | 1.70 % / 0.83 % | 0 | 123 |
| CORE0 (맞춘 구 안 = 전부 target) | 152 | 277 | 0.85 / 0.72 | 0 / 0 | 1.28 % / 0.75 % | 0 | 652 |
| CORE5 (맞춘 반지름 − 5 mm) | 202 | 348 | 0.83 / 0.66 | 0 / 0 | | 0 | 751 |
| F1 + **P2** (support-aware sign, §6) | **40** | **75** | | 0 (seed 를 지우지 않음) | = F1 | 0 | 133 (binding_out 58 → 54) |
| OFF + P2 | 206 | 462 | | 0 | = record | 0 | 785 |

과제 문서의 "이웃 과일 오제거 1.9–2.3 %" 의 출처 정의는 찾지 못했다. 위 표의 같은 성격의 지표는 LTE composition 의 "지운 seed 중 이웃 라벨"
(record 1.19 %) 과 이웃 표면 voxel 보존이다.

### 4-5. F1 이 남기는 것 (`attrib --variant F1`, stale 제외)

| 분류 | latched (45,953 voxel, 중앙값 90) | closing (26,905, 중앙값 136) |
|---|---|---|
| `below_zmin` | 35.8 % | 34.0 % |
| `removed_sign_left` | 45.7 % (막은 seed: below_zmin 18,509 · kept_ambiguous 2,189 · kept_nontarget 242) | 41.7 % (below_zmin 8,109 · kept_ambiguous 2,885) |
| `interior_impure` | 10.9 % (below_zmin 4,599) | 10.0 % (below_zmin 1,895) |
| `kept_nontarget` · `kept_ambiguous` · `interior_not_sign_set` | 4.1 · 2.1 · 1.5 % | 4.0 · 8.1 · 2.2 % |

F1 뒤에는 **지원면 근처 (cap + 그 cap seed 로 부호가 남은 7.5 mm 띠) 가 latched 86 % · closing 71 %** 다.

## 5. 추적 · 기억 중심 (`manipulated.centroid`, 보이는 면 쪽 17.5 mm 치우침) 의 호출 경로

`ManipulatedIdentity.centroid` (`stages/target_grounding.py:365`, 마지막 관측 기하의 점 평균 — 보이는 면만 있으므로 치우친다;
`TargetGeometry.centroid` `stages/admissibility.py:196` = `points.mean`). 이 측정에서 non-stale 중앙값 17.45 mm (LTE 17.45 와 같음).

| 소비처 | 파일:줄 | 무엇에 쓰나 | 이 kit (SPLIT) 에서 켜짐? |
|---|---|---|---|
| target 제외 공 중심 | `runtime/pipeline.py:3588–3628` `_target_ball` | **쓰지 않는다** — `fit_surface_sphere` 중심 (661/661 `surface_fit`). 맞춤이 물러나면 그때만 centroid | 켜짐 (centroid 경로 0 회) |
| 미세 창 배치 | `fields/curobo_builder.py:798` `manipulated_box` | manipulated 점의 AABB (centroid 아님) | 켜짐 |
| self-filter guard 중심 | `runtime/pipeline.py:1805–1807` → `:1837` · `:1896` · `:2725` | 로봇 구 inflation 을 끄는 반경 (manip 반지름 + 1 voxel) 의 중심 — 치우치면 guard 가 보이는 면 쪽으로 이동 | 켜짐 (기본, radius 0 → 물체별) |
| attach reach (HSR) | `trajopt/safe_policy.py:1399` `parent_reach(..., manip.centroid)` → `trajopt/grasp_latch.py:714` (`reach_m ≤ 0.12 m`) | 손바닥 ↔ 중심 거리로 attach 승인 | 켜짐 |
| hand-relative reach (SL5b · IFG-HSR) | `trajopt/safe_policy.py:1336` `carried_point(...)`, `runtime/pipeline.py:1339–1368` pin | 가려진 동안 중심을 손과 함께 옮김 | 꺼짐 (`--latch-reach-hand-relative` 없음) |
| held 관측 점검 | `runtime/pipeline.py:1286` `held_observation_check(observed_centroid=…)` | 쥔 뒤 관측 중심 vs 쥔 구 | held 단계 |
| 손 구 예외 (NBC) | `trajopt/safe_policy.py:1731` `centre = manip.centroid` · gate `:1734` (`neighbour_contact_gate` 0.15 m) → `trajopt/linearize.py:1027` (손가락 구 중 이 중심에서 `neighbour_contact_radius` 0.12 m 안) | 이웃 접촉 margin −20 mm 를 받을 행 선택 | NBC kit 만 |
| wrist track | `trajopt/safe_policy.py:1777` | 손바닥 ↔ 중심 반경 gate | 꺼짐 (flag 없음) |
| grasp anchor (TOA) | `trajopt/safe_policy.py:1836` | 닫는 행 목표 ↔ 중심 거리 gate | 꺼짐 |
| recovery 기록 | `trajopt/safe_policy.py:1622` | `palm_to_target_m` 기록 | 기록만 |
| tip alignment (T43 T) | `trajopt/safe_policy.py:1545` | 측정 기록 | 기록만 |
| held 3D rule | `trajopt/held_3d.py:12` · `safe_policy.py:1259` | 직전 요청의 grounded target centroid (manipulated 아님) | S2 kit |
| TrajOpt cost · 제약 | — | `linearize.py:2225` `target_volume` 은 공 (`ball.centre_m`, 맞춤 중심) 을 쓴다 (`target_volume_exempt` 꺼짐). TO 의 target 목표점으로 centroid 를 쓰는 경로는 없다 | |

즉 17.5 mm 치우침은 **제외 공에는 들어가지 않는다.** 들어가는 곳은 attach reach · self-filter guard · NBC 행 선택 · (꺼진) gate 들이다.
NBC 가 켜진 kit 에서는 예외 반경 0.12 m 가 치우침보다 훨씬 커서 행 선택에 미치는 영향은 반경 경계의 행에 한정된다 (재지 않음).

## 6. 설계안

### P1 — 이미 있는 T46 F1 (`--nontarget-evidence surface`) 을 IFG 서빙 구성에 켠다 (코드 변경 없음)

- **바꿀 곳:** 서빙 launcher 의 flag 묶음. SPLIT 은 run dir 의 `verify/IFG-APPLE-SPLIT/p/env.sh:59`
  (`T44A="--target-exclusion ball-owned --owned-max-dist-voxels 3 …"`) 에 `--nontarget-evidence surface` 를 더한다 (`run_chain.sh:152` 의
  cmdline 확인 목록에도). 코드 경로는 이미 있다: `trajopt/serve_safe.py:1734` (flag, 기본 None = `proximity`) →
  `runtime/pipeline.py:3113–3128` → `fields/curobo_builder.py:470` `nontarget_surface_evidence` (shell 2 voxel, tie 1 voxel).
  기본값은 지금도 꺼짐이다. 새 flag 를 만들지 않는다.
- **왜 첫 번째인가:** 남은 voxel 의 근거 오염 계열 (4 · 5 · 9 + 그 2차 효과 8 · 10 의 2/3) 이 latched 의 약 75 % (4 · 5 · 9 = 51 %, 8 · 10 중 kept seed 가 막은 것 약 24 %), non-stale 활성 행의 99 % 다.
- **기대 효과 (§4-4, offline):** 남은 voxel 중앙값 251 → 90 / 520 → 136, recall 0.79 → 0.91 / 0.58 → 0.85,
  non-stale target 출처 active_ref 785 → 133 (211 k11–14: 102 → 19, 979: 58 → 0, 320 k16: 42 → 0, 1313: 162 → 10, 126: 119 → 0).
- **위험:** (a) 이웃 오제거 — 이웃 표면 voxel 잃음 3 / 24,647 · 7 / 12,303, 지운 seed 의 이웃 라벨 1.19 → 1.25 % (latched) — 작다.
  (b) 탁자 구멍 — 0 (`z_min` 은 그대로, F1 은 근거 점만 뺀다). (c) **T46 CL3 에서 F1 은 close 뒤 `uncertified` HOLD 를 37 → 130 으로
  늘렸다** (원인 T46b · T46c) — IFG 구성 (SPLIT · HSR · NBC) 에서 같은 일이 나는지는 closed-loop 로만 안다.
  (d) stale (ep320 k17, 393 행) 에는 효과 없음.

### P2 — support-aware sign: 지원면 seed 때문에 남은 음수 부호를 고친다 (새 flag, 기본 꺼짐)

- **무엇:** `_force_sign_on_pure_voxels` 는 지운 voxel 의 `|d| ≤ 1.5 vs` 이면 "다른 표면이 7.5 mm 안에 있다" 로 보고 부호를 남긴다.
  F1 뒤 남은 voxel 의 latched 50 % · closing 37 % 는 그 "다른 표면" 이 공의 `z_min` **아래** seed (탁자 상판 / 사과 bottom cap) 인 경우다. 그 voxel 은
  `z_min` 위에 있고 sign 후보 (target 쪽) 이므로, 가장 가까운 남은 seed (PBA site) 가 `z_min` 아래이면 `|d|` (양수) 를 준다.
  **seed 는 하나도 지우지 않는다** — 탁자 seed 는 그대로이고 그 거리도 그대로다 (부호만, `z_min` 위 voxel 만).
- **바꿀 곳:** `fields/curobo_builder.py:1799–1802` (`site_linear` 를 이미 만든 바로 뒤의 `_force_sign_on_pure_voxels` 호출에 `site_linear`,
  `ball.z_min`, flag 를 넘긴다), `fields/curobo_builder.py:1978` (`_force_sign_on_pure_voxels` 에 site 의 z < `z_min` 인 impure voxel 을
  양수로 하는 분기 — GPU tensor 경로와 numpy 경로 둘 다), `config.py:1158` 근처 `EsdfConfig.target_free_support_sign: bool = False`,
  `trajopt/serve_safe.py:1734` 근처 `--target-free-support-sign` (`--target-exclusion ball-owned` 와 함께만). 꺼짐이면 호출 인자가 같아 출력 불변.
- **기대 효과 (offline `support_sign`, nearest F==0 seed 를 PBA site 대신 씀):** F1 위에서 남은 voxel 중앙값 90 → 40 / 136 → 75,
  총량 45,953 → 22,845 / 26,905 → 16,900. **active_ref 행은 줄지 않는다** (133 → 133), binding_out 58 → 54. 즉 이 안은 남은 부피를 줄이지만
  이 기록의 HOLD 원인 행에는 거의 닿지 않는다. 우선순위는 P1 뒤, 필요할 때.
- **위험:** (a) 이웃 — 부호를 바꾼 voxel 56,000 중 133 개 (0.24 %) 가 GT 이웃 과일 **내부** 였다 (이웃 과일 바닥이 같은 지원면 위에 있을 때,
  그 내부 음수를 양수로 숨긴다 = 관통을 숨김). 완화: 바꾸는 voxel 을 공 안 · 맞춘 구 + shell 안으로 제한 (재지 않음).
  (b) 탁자 구멍 — seed 를 지우지 않으므로 거리 구멍은 없다. 다만 `z_min` 바로 위 voxel 이 "탁자 위 양수" 가 되어 탁자 쪽 clearance 는
  탁자 seed 까지의 거리로 그대로 남는다. (c) PBA site 와 offline 최근접 seed 가 다를 수 있다 (PBA 근사).

### 고르지 않은 것

- `z_min` 아래 cap (F1 뒤 34–36 %) 을 직접 지우는 것 (pad 5 mm → 0, 또는 맞춘 구로 cap 을 지움): cap 발자국 반경 ≈ √(2·33·9) ≈ 24 mm 안의
  탁자 seed 를 같이 지운다 = 탁자 구멍. 이 기록의 활성 행 중 cap 에 닿은 것은 F1 뒤 30 / 133 이다.
- CORE (맞춘 구 안 = 전부 target): F1 보다 약하다 (중앙값 152 / 277, 행 652). 근거 오염이 표면 seed 를 지키는 것을 못 고친다.
- F2 (ambiguous 제거): 이웃 표면 voxel 4.0 % 잃음 (T46c 에서 채택 안 한 이유와 같다).
- stale (ep320 k17 · 떨어진 뒤, ep211 closing 9): 기억 · 재관측 문제 (HSR / 추적) 이지 제외 영역 형상 문제가 아니다 — 이 STEP 범위 밖.

## 7. figure (run dir, `AG3S/ag3s/docs/figures` 에 쓰지 않음)

- `…/verify/IFG-SPHERE-ATTRIB/fig/ifg-sphere-attrib-shares.png` (+ `.json`) — 왼쪽: 분류 그룹 비율 (latched · closing × record · F1 replay),
  오른쪽: 변형별 남은 voxel 중앙값과 이웃 표면 voxel 손실.
- `…/fig/ifg-sphere-attrib-scene-ep211-k12.png` (+ `.json`) — ep211 k12 (HOLD) 의 남은 voxel 을 원인별 색으로 x-z · x-y **투영**
  (단면 아님), 공 · `z_min` · GT 중심 · 맞춤 중심 · target 점 · 활성 행이 닿은 voxel.
- verifier 가 docs figure 로 옮길 때 이름 충돌 없음 (`ifg-sphere-*` 없음).

## 8. verifier 가 알아야 할 것

- 새로 생긴 플래그 · 기본값 변경: 없음 (P1 · P2 는 제안).
- 재생산이 필요한 산출물: 없음. raw = `$AG3S_RUN_DIR/verify/IFG-SPHERE-ATTRIB/raw/` (episode 별 json + voxel npz, `*_attrib_{OFF,F1}.json`).
- 옛 기록과 호환: 입력 run 은 읽기만 했다.
- 끝난 뒤 process 0 개, `reclaim_cache.py`: `reclaim: ws 15.5->15.5 GiB | cache 1.2->0.0 GiB, 2975 files, 3 paths`.
- 측정하지 않은 것: HOLD 변화 (TO 재풀이), closed-loop, 이웃 과일 바닥과 지원면이 겹칠 때 P2 의 완화안, GT 엄격 규칙 (≤ 5 mm) 을 못 넘는 행
  voxel (SPLIT active_ref 1,178 중 389) 의 거리 분포.

## 9. 내가 기대하는 결과

(verifier 는 측정 전에 읽지 않는다) P1 을 켠 closed-loop 에서 211 · 979 류의 grasp 직전 target 출처 HOLD 는 줄고, 이웃 접촉은 늘지 않을 것이다.
close 뒤 `uncertified` HOLD 증가 (T46 CL3) 가 다시 나는지가 가장 불확실하다.
