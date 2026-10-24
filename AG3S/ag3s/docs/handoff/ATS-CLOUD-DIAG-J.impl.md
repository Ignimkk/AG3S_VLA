# ATS-CLOUD-DIAG — 구현 (cloud-labeling ATS 표현 옵션, Implementer J)

> writer: ag3s-implementer (J) · 읽는 쪽: verifier (V4), scribe, lead · 작성 2026-10-24 (UTC 06:12 = 15:12 KST)
> checkout `/mnt/dev/AG3S_VLA-benchmark` · branch `feature/panda-benchmark` · HEAD `3c677553` (dirty: 아래 두 파일) · **commit 없음**
> CPU worker 1개 · GPU 사용 안 함. `Baseline/paper_baselines/closed_loop/` 는 건드리지 않았다 (I 작업 중).

## 무엇을 했나 (평이한 요약 먼저)

`eval/cloud_labeling.py` 에 ATS 표현을 고르는 옵션 두 개를 추가했다. `--ats-repr {cloud_target,target_points,exclusion}`
(기본 `cloud_target`) 과 `--ats-tol` (기본 0.005 m) 이다. 기본값으로 돌리면 V1 run 과 `calls.jsonl` 이 byte 단위로 같다.
`exclusion` 은 H 의 `recompute_ats_arms.py` 와 같은 정의다. has_target call 에서는 `target_points` 를 쓰고,
has_target=False 이지만 `manipulated_state` 가 있는 call 에서는 episode 안의 직전 has_target call 의 `target_points` 를 이어 붙인다.
cloud 점 가운데 그 geometry 에서 tol 안에 있는 점이 ATS positive 다.
call 범주(fresh / anchor / occluded / excluded_no_target / no_admissible)는 모든 run 에서 `ats_calls.jsonl` 에 call 마다 남긴다.
기본값이 아닌 run 은 여기에 더해 `calls.jsonl` 의 모든 행에 `ats_call_cat` 를 넣는다.
기본 run 에 이 field 를 넣으면 byte 동일성이 깨지므로 넣지 않았다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `Baseline/paper_baselines/eval/cloud_labeling.py:18-33` | docstring: 세 표현과 범주 정의 | summary `definitions` 에 들어간다 |
| `cloud_labeling.py:79-87` | `ATS_CT_ARM = "ats_v11ab_cloud_target\|noun_attention"`, `ATS_REPRS`, `ATS_TOL_M = 0.005`, `ATS_CATEGORIES`, `CARRY_CENTROID_TOL_M = 1e-5`, `_ATS_GEOM_KEYS` | H 최소 변경안 |
| `cloud_labeling.py:173-182` `flags_near` | cKDTree, `d <= tol + 1e-9` | H 의 `flags_from` 과 같은 식 |
| `cloud_labeling.py:185-213` `ats_episode_geometry` | call 별 범주, `tp`, `excl`, `carried_from`, `carry_centroid_err_m` | 순수 함수이며 합성 test 대상이다 |
| `cloud_labeling.py:248-262` `summarize_by_category` | 범주별 `aggregate_arm` (ATS 행 · cloud_target 행) | 기본값이 아닌 run 의 summary `ats_by_category` |
| `cloud_labeling.py:276` · `:293-307` | `PerceptionEpisode` 가 `target_points`, `target_indices_offsets`, `manipulated_state`, `manipulated_centroid`, `grounding_status` 를 더 읽는다. `ats_geometry()` 를 추가했다. key 가 없으면 `LookupError` | 옛 perception run 에서는 기본 모드로만 돌고 범주는 비운다 |
| `cloud_labeling.py:367-382` | settings 에 `ats_repr`, `ats_tol_m` 를 넣는다. 값을 검증한다. 기본값이 아닌 run 은 output dir 을 만들기 전에 key 를 확인한다 | |
| `cloud_labeling.py:411-424` | 기본 run 의 `arm_flags` 는 그대로다. 기본값이 아닌 run 은 `ATS_ARM` = 선택한 표현, `ATS_CT_ARM` = 현행 `cloud_target`, seeds 다. `exclusion` 의 carry call 에서 centroid 오차가 None 이거나 1e-5 m 이상이면 `ValueError` | paired 기준은 `ATS_ARM` 그대로다. figure script 를 바꿀 필요가 없다 |
| `cloud_labeling.py:441-448` | `ats_calls.jsonl` (새 파일). call 마다 `ats_call_cat`, `grounding_status`, `manipulated_state`, `n_target_indices`, `n_target_points`, `n_cloud`, `n_cloud_target`, `n_ats_pred`, `carried_from`, `carry_centroid_err_m` | 범주 기록. 기본 run 에서도 쓴다 |
| `cloud_labeling.py:462-463` | 기본값이 아닌 run 의 summary 에 `ats_by_category` | V4 범주표의 입력 |
| `cloud_labeling.py:503-508` | `--ats-repr`, `--ats-tol` parser | |
| `Baseline/paper_baselines/tests/test_cloud_labeling.py:110-240` | test 5 개 추가 (parametrize 포함 8 case) | 아래 |

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=osmesa HF_HUB_OFFLINE=1 \
  PYTHONPATH=$PWD .venv-baselines/bin/python -m pytest -q -p no:cacheprovider Baseline/paper_baselines/tests/test_cloud_labeling.py
```

결과: **15 passed in 115 s** (`.venv-baselines`, 기존 7 + 새 8).
`.venv-ag3s` 에서도 `test_default_repr_byte_identical_to_v1` 과 `test_repr_equals_h_recompute[exclusion-0.005]` 이 통과한다 (2 passed, 42 s).

| test | 무엇을 | 결과 |
|---|---|---|
| `test_flags_near` | exact / 5 mm 경계, 빈 geometry | pass |
| `test_ats_episode_geometry_categories_and_carry_forward` | 합성 5 call (fresh→anchor→occluded→excluded→no_admissible): 범주, carry-forward, 표현별 flag 수 `[2,2,2,0,0] / [2,2,2,2,0] / exact [2,2,0,0,0]`, centroid 오차 보고, target 이전의 state-only call, `other` | pass |
| `test_parser_defaults_keep_f` | 기본 `cloud_target`, 0.005 | pass |
| `test_default_repr_byte_identical_to_v1` | t0·t4 (2 ep, 45 call × 14 arm): V1 `calls.jsonl` 의 해당 줄과 **bytes 동일**, `figure_dump` 4 개 배열 동일, `ats_calls.jsonl` `n_ats_pred == n_cloud_target` | pass |
| `test_repr_equals_h_recompute[exclusion-0.005 / 1e-6 / 0.010, target_points-0.005]` | t0·t4: `ATS_ARM` 행 = H `ats_ex_{5mm,exact,10mm}` / `ats_tp_5mm`, `ATS_CT_ARM` 행 = H `ats_ct` (모든 metric field), `ats_call_cat` = H `calls_diag.jsonl` `cat` (이름 대응). 다섯 범주가 모두 나온다 | pass |

recorded-data test 는 V1·SL4·SL3 run 이나 H 산출물(`outputs/ats_cloud_diag/`)이 없으면 skip 한다.

**10 episode 전체 동일성 확인** (회귀 측정이 아니다. 옵션이 정의대로 구현됐는지 보는 확인이다.
script `/tmp/claude-0/-mnt-dev-AG3S-VLA-benchmark/91b137ec-607f-41ff-9542-8a1b0025e74a/scratchpad/J/fullcheck.py`, `.venv-baselines`, 출력은 scratchpad):

| 확인 | 결과 |
|---|---|
| 기본 옵션 `calls.jsonl` vs V1 (3094 줄) | **bytes 동일** |
| 기본 옵션 summary `arms` · `paired_vs_ats` vs V1 | 동일 |
| `figure_dump` 20 개 vs V1 | 모든 배열 동일 |
| `ats_calls.jsonl` 범주 수 | fresh 45 · anchor 62 · occluded 72 · excluded_no_target 38 · no_admissible 4 (= H 범주표) |
| `exclusion` 5 mm, ATS 행 + cloud_target 행 442 개 vs H `calls_ats_arms.jsonl` | 불일치 0 |
| 같은 run 의 summary: ATS arm = H `ats_ex_5mm`, cloud_target arm = H F-ATS, segmenter 12 arm 의 `paired_vs_ats` = H `ref_ats_ex_5mm` | 모두 동일 |
| carry-forward centroid 오차 최대 (38 call) | 4.6e-9 m |

## V4 가 그대로 쓸 명령

V1 과 같은 환경(`.venv-baselines`)과 같은 입력이다. 네 run 은 각각 약 2 분 걸린다.
**CPU 1 worker 로 순서대로 돌린다.** 결과는 `create_run` 으로 `AG3S_VLA-storage/results/benchmark/<날짜_commit_run-id>/` 에 쌓인다.

```bash
cd /mnt/dev/AG3S_VLA-benchmark
source scripts/setup/env.sh; source Baseline/paper_baselines/setup/paper_baselines_env.sh
CL() { CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 MUJOCO_GL=osmesa \
  PYTHONPATH=$PWD .venv-baselines/bin/python -m Baseline.paper_baselines.eval.cloud_labeling \
  --ats-run 20261012T225820_46a08d6ff644_sl4-perception-dev \
  --grounding-run 20261023T134909_bf1403d54bc6_paper-baselines-2-sl3dev-notrack \
  --dump-figure-data --dump-per-episode 2 "$@"; }
# 주 표현: ESDF 가 받은 target geometry, 5 mm
CL --ats-repr exclusion --ats-tol 0.005 --run-id ats-cloud-exclusion-5mm-sl3dev
# sensitivity: exact (1e-6 m) / 10 mm
CL --ats-repr exclusion --ats-tol 1e-6  --run-id ats-cloud-exclusion-exact-sl3dev
CL --ats-repr exclusion --ats-tol 0.010 --run-id ats-cloud-exclusion-10mm-sl3dev
# fresh-only 진단 행 (= V1 과 byte 동일한 calls.jsonl + 범주 sidecar ats_calls.jsonl)
CL --ats-repr cloud_target --run-id ats-cloud-cloudtarget-sl3dev
# (선택) exclusion 의 carry 를 뺀 표현: CL --ats-repr target_points --ats-tol 0.005 --run-id ats-cloud-targetpoints-5mm-sl3dev
```

산출물 읽는 법:
- `calls.jsonl`: 기본값이 아닌 run 에서 `ats_v11ab|noun_attention` 은 **선택한 표현**이다 (`summary.settings.ats_repr` · `ats_tol_m` 로 확인한다).
  `ats_v11ab_cloud_target|noun_attention` 은 현행 `cloud_target` 이다. 모든 행에 `ats_call_cat` 가 있다.
- `summary.json`: `paired_vs_ats` 의 기준은 선택한 표현이다. segmenter arm 과의 paired bootstrap 이 이미 들어 있고,
  `ats_v11ab_cloud_target` 행도 그 대비로 나온다. `ats_by_category` 는 범주별 `aggregate_arm` (episode mean · pooled · found) 이다.
- `ats_calls.jsonl`: call 마다 범주와 carry 정보가 있다. cloud_target run 에서는 범주를 이 파일로 join 한다.
- stale geometry 의 GT 이동 거리는 이 도구가 계산하지 않는다. `ats_calls.jsonl` `carried_from` 과 perception npz `gt_target_position`,
  H `calls_diag.jsonl` `gt_target_shift_since_geom_mm` 을 쓴다. stand-in 의 geometry frame 은 frames.jsonl `manipulated.anchor.geometry_frame` 에 있다.
  `carried_from` 은 직전 has_target call 이며 geometry frame 과 다를 수 있다.

## verifier 가 알아야 할 것

- 새로 생긴 플래그·기본값 변경: `--ats-repr` (기본 `cloud_target`), `--ats-tol` (기본 0.005). **기본값은 V1 과 같고 V1 수치는 바뀌지 않는다.**
- 새 산출 파일: 모든 run 에 `ats_calls.jsonl` 이 생긴다. 기존 파일은 그대로다.
- 기본 run 의 `summary.json` 은 V1 과 비교해 `settings` (`argv`, 새 `ats_repr` · `ats_tol_m`) 와 `definitions` (docstring 이 늘었다) 만 다르다.
  `arms` · `paired_vs_ats` 는 같다.
- 범주 규칙(npz 만 사용): `fresh` = has_target ∧ `target_indices` 길이 > 0. `anchor` / `occluded` = has_target ∧ 길이 0 ∧
  `manipulated_state` visible / occluded. `excluded_no_target` = ¬has_target ∧ state ≠ "" ∧ episode 안에 이전 has_target call 이 있음.
  `no_admissible` = 나머지 ¬has_target (SL4 에서는 4 call 모두 grounding_status `no_admissible`). 그 밖은 `other`.
  H 는 `excluded_no_target` 를 frames.jsonl `exclusion.active` 로 나눴다. SL4 221 call 에서 두 규칙의 결과는 같다 (test 와 전체 확인).
- `exclusion` 은 carry centroid 검사(1e-5 m)가 실패하면 멈춘다. release split 이 있는 run 에서 이 불변식이 깨지면 그 call 에서 `ValueError` 가 난다.
- 재생산이 필요한 산출물: 없음 (perception 재실행 불필요).
- 옛 기록과 호환: 깨지지 않는다. 범주 key 가 없는 옛 perception run 은 기본 모드로만 돌고, `ats_calls.jsonl` 에는 범주 field 가 없다.

## 내가 기대하는 결과

(verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.)
exclusion 5 mm 의 ATS 행은 H 예비값과 같을 것이다: found 177/221, IoU episode mean 0.259. 위의 전체 동일성 확인에서 이미 같았다.
exact 와 10 mm 는 H 표의 (b+) exact (found 165, IoU 0.190) 와 10 mm (found 202, IoU 0.324) 일 것이다.
