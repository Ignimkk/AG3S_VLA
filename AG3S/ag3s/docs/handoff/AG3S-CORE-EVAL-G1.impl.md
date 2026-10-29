# AG3S-CORE-EVAL-G1 — 구현 (G1 기하 grasp 보정을 메인 작업 트리에 합치기)

> writer: ag3s-implementer · 읽는 쪽: verifier, scribe, lead · 작성 2026-10-29 03:10 KST (`date -u` 18:10 + 9 h)
> task: [AG3S-CORE-EVAL-G1.task.md](AG3S-CORE-EVAL-G1.task.md) · 결정: [AG3S_GEO_GRASP_DECISION_20261029_KO.md](../../../../doc/active/reviews/AG3S_decision/AG3S_GEO_GRASP_DECISION_20261029_KO.md) §2
> 원시 기록 · 도구: `$AG3S_STORAGE_ROOT/results/development/20261029T025059+0900_461231a432a2_772a1831d0a1/verify/AG3S-CORE-EVAL-G1/` (아래 `$G1`)

## 무엇을 했나 (평이한 요약 먼저)

- 메인 작업 트리에 **G1 "VLA 닫기 근처 보정"** 코드가 들어가 있다. 모두 기본 꺼짐이다.
  - G1 서버 flag: `--geo-grasp --geo-goal-mode correct --geo-track-scale 0.25 --geo-continuity-off`
  - 미채택 G7 (타겟 쪽 유도) 은 떼어 냈다.
  - G2–G6 은 G1 과 같은 controller · 후보 생성기 안의 flag 분기라서 남겼다. 모두 꺼져 있다.
- **사용자 판정 (a) 를 구현했다** (lead 전달, 2026-10-29). G1 과 쥐기 목표 항 (A4, `--grasp-anchor 2000 --grasp-anchor-path-weight 0`) 을 함께 켤 수 있다.
  - G1 이 기하 목표를 낸 chunk 에서는 G1 이 A4 를 **대체**한다.
  - 그 밖의 chunk 에서는 A4 가 그대로다.
  - 어느 항이 자리를 차지했는지 기록에 남는다.
- **꺼짐 동등성:** flag 를 주지 않으면 TrajOpt 결과가 평가 코드 (snapshot_core) 와 bit 단위로 같다.
  - core 평가 기록 106 요청을 재생해 106/106 이 같았다 (V2).
  - 같은 G1 flag 에서 worktree 코드와 합친 코드의 결과가 같다: 450/450 (V3).
- **A8 + G1 서버의 CPU 검사를 통과했다** (parse · flag 조합 · config · A8 로봇 모델 위 G1 controller 생성, V4). GPU 서버 실행은 lead 몫이다.

## 0. 시작할 때 발견한 것 — 메인 작업 트리는 이미 합쳐져 있었다

- task 는 "메인 작업 트리 = snapshot_core" 라고 적었다. 그러나 시작 시점 (2026-10-28 17:04 UTC) 의 메인 트리는 이미 달랐다.
  - geo-grasp 세션 (session `0cb12c2c`) 이 **15:57 UTC** 에 worktree 변경 **G1–G7 전부**를 3-way merge 해 두었다.
    - 6 파일 수정: `config.py` · `grasp_anchor.py` · `refiner.py` · `safe_policy.py` · `serve_safe.py` · `sqp.py`
    - 새 파일 복사: `AG3S/ag3s/grasp/` · `geo_grasp.py` · `approach_guide.py` · 실험 도구 · test
  - 그 세션의 merge 전 백업 `/tmp/claude-0/-mnt-dev-AG3S-VLA-development/0cb12c2c-edfa-4b14-a8e3-117eaeb89ad6/scratchpad/main_backup_1557/` 을 확인했다. 6 파일 모두 snapshot_core 와 `cmp` 동일이다. 즉 merge 전 메인 = 평가 코드.
- 그 merge 를 다시 계산해 대조했다. base 는 worktree 의 출발점인 `snapshot_e3c` (run `20261025T191319+0900_35b909cf096a_cc215238f581`) 이다.
  - worktree 수정 파일 중 6 파일은 snapshot_e3c 와 같다: multiview · pipeline · target_grounding · tp_timing · problem · qp.
    - 새 파일 일부도 같다: blas_threads · gpu_seeds · multi_seed · e1–e3 도구 · substage 등.
    - 이것들은 AG3S-PRE-CORE-TO 의 옛 판이며 G1 과 무관하다. 메인 쪽을 그대로 두었다 (task §1-1).
  - geo 변경이 있는 6 파일은 `git merge-file snapshot_core snapshot_e3c worktree` 결과와 메인이 같았다.
  - 충돌은 파일마다 2 곳이었다. 모두 양쪽을 다 남기는 union 으로 풀려 있었다.
    - `grasp_anchor.py`: dataclass 필드, `goal_mask` (explicit rows → GGW `widen` → close 순)
    - `serve_safe.py`: `_fmt_cost`, argparse 블록
- 그래서 이 STEP 은 세 가지 일이 되었다.
  1. 그 merge 를 G1 만 남도록 줄인다.
  2. 꺼짐 동등성을 증명한다.
  3. 판정 (a) 를 더한다.

## 1. 가져온 것 · 안 가져온 것

- 출처 판: worktree 의 geo 파일은 `snapshot_geo6` (run `20261026T204126+0900_b3b69311f948_8d603092427b`) 과 모두 `cmp` 동일하다.
- G7 은 `snapshot_geo5 → snapshot_geo6` 에서만 더해졌다 (`snapshot_geo6_head.txt`: "the approach guidance (task §3l)").

| 항목 | 결정 | 이유 |
|---|---|---|
| **G1** 보정 (`geo_goal_mode correct`, 6D 목표 항, 추종 0.25 · 연속성 끔, 보정 닫기 gate) | **가져옴** | 채택 |
| G2 접근 목표 (`geo_goal_mode redirect`, controller 기본값) · G3 닫기 gate (`--geo-close-gate valid-pose`) · G4 중심 통과 (`--geo-correct-max-offset`) | 가져옴 (`--geo-grasp` 없으면 꺼짐, G1 flag 에서도 안 쓰임) | `GeoGraspController.anchor` · `_correct_anchor` · `close_gate` 안의 분기다. 떼려면 controller 를 새로 써야 하고, 그러면 V3 (worktree 동등성) 의 근거가 사라진다 |
| G5 손목 · 팔 · 누적 장애물 (`--geo-wrist-check/-arm-check/-obstacle-memory`) · G6 거리장 조회 (`--geo-obstacle-source field`) · 생성 시간 상한 flag (`--geo-time-budget-ms`) | 가져옴 (기본 꺼짐) | G1 이 쓰는 후보 생성기 (`AG3S/ag3s/grasp/candidates.py` · `arm_shape.py` · `memory.py`) 와 `geo_grasp.py` 안의 분기다 |
| **G7** 타겟 쪽 유도 (`approach_guide.py`, `--approach-*`, `cost.approach_*`, `GraspAnchor.path_offset/track_from` · variant "DIR", refiner `reference_trajectory_override` · `approach_seeds`, sqp approach seed · 선택) | **빼냄** | 독립 모듈 + 5 파일 hook 이라 떼어 낼 수 있다. G1 평가 코드 (`snapshot_geo2`, 가림 18 전체 검증) 에도 없었다. `snapshot_geo6 → geo5` 역 patch 를 3-way 로 적용했다 (충돌 0). 적용 전 메인 6 파일 사본: `$G1/merge/backup_main_before_g7/` |
| G7 전용 새 파일 `AG3S/trajopt/approach_guide.py` · `experiments/approach_replay.py` · `tests/trajopt/test_approach_guide.py` | 메인에서 지움 | worktree 에 `cmp` 동일한 사본이 그대로 있다 (+ `$G1/merge/removed_g7_files/`) |
| 실험 도구 `geo_grasp_replay.py` (V3 에 씀) · `geo_gate_offline.py` (G3 offline) · `studies/geo_grasp_offline.py` (S1 후보 offline) | 남김 | 서버가 import 하지 않는 offline 도구. 기록 재현용 |
| 메인 쪽 다른 변경 (E1–E3 · BLAS · 손바닥 껍질 구 · 꼭지 덮개 · 팔 구 · 쥐기 목표 항 window · 렌더 mask 등) | **그대로** | 아래 6 파일에서 snapshot_core 대비 지운 줄은 geo 판으로 바뀐 줄뿐이다 (§2). 다른 파일은 손대지 않았다 |
| geo-grasp 세션이 메인에 복사해 둔 문서 (`docs/figures/geo-grasp/`, `AG3S-GEO-GRASP.task.md` · `.manifest.json`) | 손대지 않음 | scribe · verifier 소유 |

## 2. 바뀐 파일 (snapshot_core 대비) · md5

줄 수는 snapshot_core 대비 `diff` 결과다 (+ 더한 줄 / − 지운 줄). md5 는 이 STEP 끝의 메인 작업 트리 기준이다.

| 파일 | snapshot_core 대비 | md5 | 무엇이 · 왜 |
|---|---|---|---|
| `AG3S/trajopt/config.py` | +107 −1 | `57196351c1a28f96693dac3ad284631c` | `CostConfig.geo_*` 24 필드와 `validate` 의 geo 검사 (worktree 그대로, G7 `approach_*` 뺌). **이 STEP 에서 더함:** (1) `validate` `:383-395` — 판정 (a): `geo_goal_mode correct` + E1-C 목표 항 (`w_grasp_goal > 0`, `w_grasp_tcp 0`, goal box 0, window `close`) 조합만 허용. (2) `_OMIT_WHEN_NONE` · `_OMIT_WHEN_DEFAULT` 에 geo 필드 (`:1298-1313`) — 꺼짐이면 `to_dict()` (서버 `--record-constraints` meta `trajopt_config`) 키가 merge 전과 같다. GGW · SCM 과 같은 방식이다 |
| `AG3S/trajopt/grasp_anchor.py` | +115 −3 | `ddb18a344d5a666836ea9a00fc5cf027` | `GraspAnchor.goal_R / w_rot / explicit_rows / goal_kind / tol_*`, variant `"GEO"`, `goal_mask` 의 explicit rows, `LinkPoseFK` · `rotation_errors` · `rotation_cost` · `rotation_qp_terms`, `anchor_record` 의 회전 기록 (worktree 그대로, G7 뺌). **이 STEP:** `replaced_variant` 필드 (`:89`, 기본 None). `anchor_record` 는 그 값이 있을 때만 `replaced_by_geo` · `replaced_variant` 키를 낸다 (`:369-371`) |
| `AG3S/trajopt/refiner.py` | 변화 없음 | `d4609fce4b5addb7849dada69df2f268` (= snapshot_core) | G7 hook 2 개를 뺀 결과 |
| `AG3S/trajopt/safe_policy.py` | +68 −0 | `9d8511d759aa0a255937887701d9df16` | `GeoGraspController` 생성 (`cost.geo_grasp` 일 때만), reset, `close_gate` + HOLD 사유, `_geo_request` (worktree 그대로, G7 `_approach*` 뺌). **이 STEP:** `_geo_request` `:1094-1105` — 이미 A4 anchor 가 context 에 있으면 G1 anchor 에 `replaced_variant="E1-C"` 를 달고, 기록에 slot 을 남긴다 |
| `AG3S/trajopt/serve_safe.py` | +74 −1 | `d33243c8bc1741351c58ad3987fd39d3` | `--geo-*` flag 21 개 + `cost_overrides` 매핑, `_fmt_cost` docstring (G7 `--approach-*` 4 개 뺌) |
| `AG3S/trajopt/sqp.py` | +56 −14 | `e47b659e40230508f424a56732da2371` | 6D anchor: `_anchor_setup` (LinkPoseFK cache key), merit 의 `rotation_cost`, QP 의 `rotation_qp_terms`, `_anchor_pose`, GPU seed `gpu_goal_note` (G7 approach seed · 선택 뺌). −14 는 `anchor_record` 호출과 FK 줄을 6D 판으로 바꾼 줄이다. 6D 가 아닐 때의 계산은 같다 |
| `tests/trajopt/test_core_eval_g1_merge.py` (**새, 이 STEP**) | 7 test | `f0713d23478916237d88891eb202383c` | §3 V1 아래 |
| 새 파일 (worktree 와 `cmp` 동일) | | 아래 | `geo_grasp.py` (512 줄), `AG3S/ag3s/grasp/*.py` 9 개 (1,881 줄), 도구 3, test 3 |
| 지움: `approach_guide.py` · `experiments/approach_replay.py` · `tests/trajopt/test_approach_guide.py` | | | G7 전용 |

```text
1ef72b555f1e7f3d6b9cab61b498f0ae  AG3S/trajopt/geo_grasp.py
619340630c9798bb1f94c006804dc63a  AG3S/ag3s/grasp/__init__.py
d72f5809df8d357aa74fe5b1f785e6bc  AG3S/ag3s/grasp/arm_shape.py
47429021277f33a9d9a53d9f25fd679a  AG3S/ag3s/grasp/candidates.py
031613303039c4dc05106975b07dfe22  AG3S/ag3s/grasp/gripper.py
477a28e6831ee44ff3602567e8393333  AG3S/ag3s/grasp/ik.py
96511fe9546c759819053edc2c138d83  AG3S/ag3s/grasp/memory.py
5f2350989091d76db1adaa683ddd451c  AG3S/ag3s/grasp/planner.py
cadd6a3266cd3202bd2df2df8c0b45ef  AG3S/ag3s/grasp/surface.py
3efa685b172b1907e5804629f3d40a4b  AG3S/ag3s/grasp/task_state.py
ab77e13dd0a672f7db8fda026f955691  AG3S/trajopt/experiments/geo_grasp_replay.py
30887a660d5d425b74b16bd00d53b518  AG3S/trajopt/experiments/geo_gate_offline.py
3ee97f3f4c6eaff25147a7ccc0f54d68  AG3S/ag3s/experiments/studies/geo_grasp_offline.py
0d74555925be21bf7c9f38180ba006c2  tests/ag3s/test_geo_grasp.py
41598638ac388b6a3ffd94a53871a463  tests/ag3s/test_geo_grasp_obstacles.py
f913cf9c167f83981255c839929f4e97  tests/trajopt/test_geo_grasp_to.py
```

## 3. 검증 (task §2, 결과 전 고정)

모두 CPU · 1 process · `nice -n 10` 으로 돌렸다. 모듈 `__file__` 이 대상 트리를 가리키는 것을 로그 첫 줄 `trees:` 로 확인했다.

| # | 검사 | 결과 | 판정 |
|---|---|---|---|
| V1 | 단위 test (메인 checkout, `.venv-ag3s`) | 아래 표 | **통과** (실패 1 = 알려진 기존 실패) |
| V2 | 꺼짐 동등성: core 평가 서버 기록 재생 | 106 요청 (REG ep211 75 + OCC 121 31): 합친 코드 = snapshot_core **106/106 bit 동일**. 기록 비교 (시간 필드 제외) 106/106 같음 | **통과** (코드 대 코드). 기록 자체와의 bit 동일은 재생 도구 한계로 불가 — 아래 |
| V3 | 켜짐 동등성: worktree 대 합친 코드, G1 flag | E3 기록 121 · 122 의 150 요청 × 3 설정 = refine 450 회 **450/450 bit 동일**, 요청 기록 (시간 필드 제외) 150/150 같음, 보정이 켜진 요청 포함 | **통과** |
| V4 | 결합 서버 CPU 검사 | A8 · A8 + G1 (A4 포함) · A8 − A4 + G1: parse · flag 조합 검사 · config 모두 ok. A8 로봇 모델 (구 606, 껍질 구 5 mm) 위에 G1 controller 생성 ok (IK left · right). `.venv-ag3s` 와 서버 venv `.venv-openpi-live` 둘 다 | **통과** (GPU 기동은 lead) |

### V1 — 단위 test

```bash
cd /mnt/dev/AG3S_VLA-development
MUJOCO_GL=osmesa PYTHONPATH=$PWD OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 \
  .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/trajopt tests/ag3s/test_geo_grasp.py tests/ag3s/test_geo_grasp_obstacles.py
```

| 범위 | 결과 | 로그 |
|---|---|---|
| 전체 `tests/trajopt` + geo test 2 파일 (G7 제거 뒤, 판정 (a) 전 코드) | 2074 passed, 33 skipped, **1 failed = 알려진 기존 실패** `test_t43t_replay.py::test_off_is_bit_identical_to_the_pre_fix_commit` (`.venv-ag3s` numpy golden, PORT · GGW impl 에 같은 기록) — 24 분 | `$G1/V1/v1_pytest.log` |
| 판정 (a) + to_dict 키 정리 뒤 관련 18 파일 (새 merge test · geo 3 · E1 · E2 · E3b · TOA · TOW · GGW · PORT · refiner · safe_policy · SC · SCM · sqp · T31 2) | 227 passed, 4 skipped (GPU) | `$G1/V1/v1b_pytest.log` |
| **최종 코드 전체 재실행** (판정 (a) 포함, 새 test 포함) | **2079 passed, 35 skipped, 1 failed = 같은 알려진 실패** (`test_t43t_replay`) — 22 분 | `$G1/V1/v1_final_pytest.log` |

- **새 test `test_core_eval_g1_merge.py` (7) 가 고정하는 것:**
  1. 꺼짐 — `cost_overrides` · `to_dict()["cost"]` 에 geo 키가 없다 (A8 + A4 명령줄 포함). round trip.
  2. G1 flag 4 개 = cost 키 4 개. dump 에 그 4 개만 있다. 보정 반경 40 mm / 20°. 생성 상한은 기본 (150 ms).
  3. G7 없음 — module · cost 필드 · GraspAnchor 필드 · `--approach-guide` 거절 · refiner / sqp 원문.
  4. (a) 허용과 나머지 거절 — A8 + A4 + G1 은 config 가 만들어진다. IFG-TOA 경로 항 · E1-B 상자 · GGW widen · redirect 모드와 함께 주면 거절한다.
  5. 둘 다 켜짐, 보정 후보 있음 (parametrize) — context anchor = G1 목표 (variant GEO, `replaced_variant` E1-C). 기록은 `grasp.anchor.replaced_by_geo = True` · `grasp.geo.anchor.slot = "geo"` · `to.metrics.grasp_anchor.replaced_by_geo / replaced_variant`.
  6. 둘 다 켜짐, 보정 후보 없음 (parametrize) — context anchor = A4 anchor 객체 그대로 (목표 = A4 의 얼린 목표). 기록은 `replaced_by_geo = False` · `slot = "E1-C"`, `to.metrics` 에 새 키 없음.
  7. 한쪽만 켜짐 — G1 만 켜면 slot · replaced 키가 없다. A4 만 켜면 `anchor_record` 가 이전과 같다.
- **알아 둘 flaky test (기존, 이 STEP 과 무관):** `tests/ag3s/test_geo_grasp.py::test_state_keeps_grasp_follows_centre_and_invalidates` 는 단독으로 돌리면 실패한다. 다른 test 뒤에 돌면 통과한다.
  - 원인: 후보 생성의 시간 상한 150 ms (wall-clock). 첫 호출은 cold 상태라 pair 계산이 428–500 ms 걸려 상한에 걸리고, 후보가 0 개가 된다.
  - worktree 코드에서도 같다 (첫 호출 500 ms → 0 개, 두 번째부터 58 ms → 7 개).
  - 세 번의 전체 · 묶음 실행에서는 모두 통과했다.

### V2 — 꺼짐 동등성 (flag 없음)

- **도구:** 기존 `tp_timing.run_timing` 재생 경로다 (`ta_offline.Run` 재구성, server mode). `$G1/tools/v2_replay.py` 가 `TrajOptChunkRefiner.refine` 출력을 잡는다.
  - 기록된 서버 명령줄을 그대로 쓴다 (geo flag 없음). A4 anchor 는 `e1_goal` 로 기록에서 다시 만들고, IFG-TOW 배율은 기록값을 재생한다.
  - CPU 이므로 GPU seed 만 껐다 (`sqp.gpu_seeds off`, `gpu_seeds_check all`). CPU 여러 초기값 16 은 켜 두었다.
- **입력:** run `20261028T224856+0900_b3b69311f948_8784a1265958`
  - `REG_k0/runs/server_8292_*/server_constraints/run_0000` 첫 episode (ep211, 75 요청)
  - `OCC/k0/server_ag3s_8296_*/…/run_0000` 첫 episode (case1_L2_s13453, 31 요청)
  - OCC 서버 폴더에는 `server_proc.txt` 가 없다. 그래서 launch script (`occ/launch_occ_CORE.sh`) 의 BASE + SAFE 로 같은 줄을 다시 만들었다 (`$G1/V2/occ_server_proc_reconstructed.txt`, REG 와 flag 동일).
  - 기록 폴더는 symlink 로만 읽었다. 저장소에는 쓰지 않았다.
- **두 트리:**
  - snapshot_core 코드: `.py` 사본을 scratch 로 만들고 `PYTHONDONTWRITEBYTECODE=1`. snapshot 폴더는 읽기만 했다.
  - 합친 메인: 판정 (a) 코드 포함.

| episode | 요청 | 두 트리 refined bit 동일 | 기록 (시간 필드 제외) 동일 | A4 anchor 켜진 요청 | 여러 초기값 발동 | 기록과 bit 동일 (core / 합친) |
|---|---|---|---|---|---|---|
| REG ep211 | 75 | **75** | 75 (73 + 2 개는 `multi_seed.*_ms` 만 다름) | 27 | 2 | 0 / 0 |
| OCC 121 | 31 | **31** | 31 | 6 | 0 | 2 / 2 |

- **기록 자체와는 bit 동일이 아니다** (|합친 − 기록| 중앙값 3.2e-4 · 6.6e-5 rad). snapshot_core 코드도 같은 수만큼만 맞는다.
- 이유는 재생 도구가 서버 입력을 근사하기 때문이다.
  - 기록의 거리장이 float16 이다.
  - destination label 은 근사한다 (`ta_offline` docstring).
  - GPU seed 대신 CPU 로 푼다.
- 따라서 task 의 "기록과 bit 동일" 조건은 어느 코드로도 만족할 수 없다. 꺼짐 동등성은 **같은 재구성 입력에서 합친 코드 = 평가 코드** 로 판정했다.
- 결과: `$G1/V2/v2_compare.json`, 요청별 refined `core/*.npz` · `merged/*.npz`.

### V3 — 켜짐 동등성 (G1 flag)

- **도구:** worktree 의 `geo_grasp_replay.run` (goal mode `correct`; B0 = 기록 구성, B1 = + geo correct, B2 = B1 + 추종 0.25 + 연속성 끔 = **G1**). `$G1/tools/v3_replay.py` 가 refine 출력을 호출 순서대로 잡는다.
- **입력:** E3 closed-loop 기록 121 · 122 (geo-grasp 세션 S2e index, 각 75 요청).
  - 기록 명령줄에 `--palm-shell` · `--grasp-anchor` · 꼭지 덮개 · 팔 구가 없다. 그래서 메인 쪽 다른 기능이 꺼진 구성이다 (task 조건).
- **바꾼 것 하나:** 후보 생성의 150 ms wall-clock 상한이 재생을 부하에 따라 흔든다.
  - 그래서 두 트리 모두 `candidates._ms` 를 0 으로 바꿔 상한을 끈 채 비교했다 (= `geo_time_budget_ms 0`, 양쪽 같은 조작).

| 비교 | refine 출력 bit 동일 | 요청 기록 (시간 제외) 동일 | 보정이 켜진 요청 |
|---|---|---|---|
| worktree (`snapshot_geo6` 판) 대 합친 메인 | **450/450** | **150/150** | 121 seq 15 (B1 · B2, 가장 가까운 grasp 16.8 mm), 122 seq 14 (17.6 mm) — 양쪽 같음 |
| 참고: G1 전체 검증에 쓴 `snapshot_geo2` 대 합친 메인 | 450/450 | 147/150 (차이 3 = geo3 이후 더 자세해진 geo 기록 키) | 같음 |

- 결과: `$G1/V3/v3_compare_worktree_vs_merged.json`, `v3_compare_geo2_vs_merged.json`.
- 한계: V3 도구는 `GeoGraspController` 를 직접 부른다. 그래서 `SafePolicy._geo_request` 의 slot 처리 (판정 (a)) 는 V3 가 아니라 unit test 5–7 로 확인했다.

### V4 — 결합 서버 CPU 검사

`$G1/tools/v4_check.py`: `serve_safe.build_parser` → `reject_bad_flag_combinations` → `trajopt_config_from_args`.

- GPU 가용성 probe (`gpu_seeds.available`: torch + CUDA + cuRobo) 는 서버 host 의 일이라 이 검사에서는 통과로 바꿨다.
- 이어서 A8 로봇 모델 (`ta_offline.build_robot` = 서버의 `build_constraint_robot_model`) 위에서 `GeoGraspController` 를 서버와 같은 인자로 만든다.

| 구성 | parse | flag 조합 | config | 주요 값 |
|---|---|---|---|---|
| (ii) A8 단독 | ok | ok | ok | w_grasp_goal 2000, geo_grasp False |
| **A8 + G1 (A4 포함, 판정 (a))** | ok | ok | ok | w_grasp_goal 2000 · w_grasp_tcp 0 · geo correct · track 0.25 · continuity off |
| (i) A8 − A4 + G1 | ok | ok | ok | w_grasp_goal 0 · geo correct |
| G1 controller on A8 model | — | — | — | 구 606, 껍질 구 여유 5 mm, IK left · right, 생성 25 s (CasADi) |

- 로그: `$G1/V4/v4_check.log` (`.venv-ag3s`), `v4_check_openpi.log` (`.venv-openpi-live`, `CUDA_VISIBLE_DEVICES=` · `JAX_PLATFORMS=cpu`).
- `serve_safe --help` 에 `--geo-*` 줄이 있다.

## 4. 결합 — G1 항과 쥐기 목표 항 (A4) 이 같은 행에서 어떻게 만나는가

- 두 항은 같은 자리 `context["grasp_anchor"]` (`GraspAnchor` 하나) 를 쓴다.
- **판정 (a) 이후 동작:** G1 이 목표를 낸 chunk 에서는 G1 항이 A4 항을 대체한다. 두 위치 항이 같은 행에 동시에 걸리는 일은 없다.

| | 쥐기 목표 항 (A4, variant `E1-C`) | G1 보정 (variant `GEO`) |
|---|---|---|
| 만드는 곳 | `SafePolicy._grasp_anchor_gate` (`safe_policy.py:1879`, GraspAnchor `:1951`) | `GeoGraspController._correct_anchor` (`geo_grasp.py:324`, GraspAnchor `:379`), `SafePolicy._geo_request` (`safe_policy.py:1081`) 가 부름 |
| context 에 넣는 순서 | `safe_policy.py:1039-1041` 에서 먼저 | 그 다음 `:1052` → `:1094-1107`. G1 anchor 가 있으면 **덮어쓴다** |
| 켜지는 단계 | latch `latched` · `closing`, 손바닥이 목표 150 mm (`grasp_anchor_radius`) 안 | latch `latched` 만. 그 chunk 의 VLA 가 c 행 (0 < c < H) 에서 닫기를 계획할 때 |
| 목표 | **얼린** VLA 닫기 위치: 처음 닫기를 계획한 요청에서 그 행의 hold link (손바닥 원점) FK. target 중심 120 mm 안일 때만. 위치만 | 이번 chunk 의 VLA 닫기 자세에서 **40 mm / 20° 안의 가장 가까운 유효 기하 grasp** (추적 중인 것 또는 이번 후보). 위치 + 방향 |
| link · 행 | hold link (`ee_left/right`), 닫는 행 c 부터 `grasp_goal_rows` = 3 행 | 같은 hold link, 같은 c 부터 3 행 (`grasp_goal_rows` 를 함께 읽음) |
| 항 · 무게 | `w_goal ‖p_k − g‖²`, w = 2000 m⁻² (`anchor_qp_terms`; merit `anchor_cost`). 경로 항 없음 | `geo_w_pos ‖p_k − g‖²` 2000 m⁻² (같은 함수) + `geo_w_rot ‖Log(R_gᵀR_k)‖²` 5 rad⁻² (`rotation_qp_terms`, SQP 반복마다 선형화 `sqp.py:1035`; merit `rotation_cost`). 경로 항 없음 |
| 같은 chunk 의 다른 변화 | 없음 | 쥐는 팔 관절 추종 × 0.25. IFG-TOW 손목 배율 (A8 `--wrist-track-scale 0.1`, 관절 4–6) 과 **곱해진다** (`safe_policy.py:1115`): 손목 0.025, 나머지 팔 0.25. 연속성 항 끔. 보정한 닫기가 grasp 를 놓치면 (10 mm / 15° 밖) 닫기를 미루고, 8 번 넘으면 HOLD |
| 위반이 남을 때 | `sqp.py:314` 안전 우선 대체 풀이 (anchor 없이 다시 풀어 위반이 작은 쪽) | 같은 규칙 |
| GPU 여러 초기값 (A2) | 위치 항이 cuRobo seed 비용에 들어감 | 위치 항만 들어간다. 회전은 CPU merit 에서만 (`gpu_goal_note` 기록) |

### 세 선택지의 코드 차이 (lead 요청) — 구현한 것은 (a)

| | 바꿀 코드 | 영향 범위 | flag 꺼짐 (둘 중 하나라도 없음) 일 때 bit 동일한 것 |
|---|---|---|---|
| **(a) 대체 — 구현함** | `config.py:383-395` 검사 완화 (correct + E1-C 만). `safe_policy._geo_request` 에서 slot 표시. `GraspAnchor.replaced_variant` + `anchor_record` 키 2 개 | 둘 다 켰을 때만. 보정 chunk (G1 이 목표를 낸 chunk) 의 목적함수가 A4 위치 항 → G1 6D 항 + 추종 0.25 + 연속성 끔. 그 밖의 chunk 는 A4 그대로 | 궤적 · 기록 키 전부. A4 만이면 anchor · 기록이 이전과 같다 (`replaced_variant` None → 키 없음, test 7). G1 만이면 `prev` 가 None → slot 키 없음. V2 · V3 가 실제 기록으로 이를 보였다 |
| (b) 합산 (구현 안 함) | `context["grasp_anchor"]` 를 여러 anchor 로 바꾸고 다음을 모두 고쳐야 한다: refiner 전달 · `sqp._anchor_setup` (상태 2 개) · merit · QP 항 누적 · 안전 우선 대체 풀이 (어느 anchor 를 뺄지) · `_goal_probe` · `gpu_seeds.anchor_terms_for` (GPU 비용) · `anchor_record` 기록 구조 · E1 재생 도구 | 둘 다 켰을 때 같은 행에 두 위치 목표 → 그 사이 타협점으로 당긴다. 기록 구조가 바뀐다 | 한쪽만이면 같게 만들 수 있지만, 기록 구조 (`to.metrics.grasp_anchor` 가 list 가 되는지 등) 를 꺼짐에서도 지켜야 한다. 고칠 곳이 많아 회귀 위험이 크다 |
| (c) G1 단독 (A4 빼고 시험) | 코드 변경 없음 | 결합 시험에서 A4 효과 (가림 121–123 손바닥 1–3 mm) 가 빠진다 | 전부 (코드 그대로) |

## 5. 결합 서버 argv (한 줄)

- A8 flag (ADOPTED_KO.md §A8, 기록 `REG_k0/runs/server_8292_20261028T135513/server_proc.txt` 그대로) 뒤에 G1 flag 4 개를 붙였다.
- 판정 (a) 코드로 CPU 검사를 통과했다 (V4).
- `<…>` 는 lead 가 채운다: checkpoint · 새 snapshot · port · run dir · `--record-constraints`.

```text
python -m AG3S.trajopt.serve_safe --config pi05_rby1_randomized_pick_place_16d_lora --checkpoint <16D 30k ckpt> --model-xml <snapshot>/robots/rby1/description/models/rby1a/mujoco/model_transport.xml --esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --obstacle-margin 0.010 --held-body-cover --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20 --to-fast-eval --esdf-host-mirror --target-exclusion ball-owned --owned-max-dist-voxels 3 --attention-split-radius-mm 12 --attention-split-ratio-source raw --target-permission-strict --attention-source noun_target:L10H7 --attention-split-partition --split-supersedes-anchor --wrist-track-scale 0.1 --neighbour-contact-mm 20 --nontarget-evidence surface --placed-no-seed-certify --kept-geometry-follow --roi-carve --roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore --roi-carve-device cuda --blas-threads 1 --to-multi-seed 16 --to-gpu-seeds curobo --to-gpu-seeds-check first-pass --palm-shell on --palm-shell-margin 5 --grasp-anchor 2000 --grasp-anchor-path-weight 0 --geo-grasp --geo-goal-mode correct --geo-track-scale 0.25 --geo-continuity-off
```

- client flag 는 A8 과 같다. G1 은 client 를 바꾸지 않는다.
  - `--safe-remote --safe-timeout 120 --safe-phase approach --safe-manipulators <쥐는 손> --safe-gate reasons --safe-hold-mode measured --obstacle-stop-distance 0 --stop-after-place 2.5`
- **결합 시험에서 셀 기록:**
  - `summary_json.to.metrics.grasp_anchor.variant`: `GEO` = G1 이 자리를 차지했다, `E1-C` = A4.
  - 같은 위치의 `replaced_by_geo` · `replaced_variant` (G1 이 A4 를 대체한 chunk 에만 있다).
  - `grasp.anchor.replaced_by_geo` (A4 gate 가 켜진 chunk 마다 True / False).
  - `grasp.geo.anchor.slot` (`"geo"` / `"E1-C"`).
  - 보정 요청 수는 `grasp.geo.anchor.active` 와 `kind == "correct"` 로 센다.

## verifier 가 알아야 할 것

- **새 flag (기본 꺼짐):** `--geo-*` 21 개.
  - G1 은 그중 4 개를 쓴다 (`--geo-grasp --geo-goal-mode correct --geo-track-scale 0.25 --geo-continuity-off`).
  - `--geo-grasp` 없이 다른 `--geo-*` 를 주면 `cost_overrides` 가 조용히 무시한다 (worktree 동작 그대로).
- **새 조합 허용:** `--geo-grasp --geo-goal-mode correct` + `--grasp-anchor W --grasp-anchor-path-weight 0` (판정 (a)).
  - 다른 조합은 config 단계에서 거절한다: `--grasp-anchor` 만 (경로 항), `--grasp-goal-box`, `--grasp-goal-window widen`, redirect 모드.
- **기록 키:** 꺼짐이면 새 키가 없다 (`to_dict` · `to.metrics` · `grasp.*`). G1 이 켜지면 `grasp.geo`, `to.metrics.grasp_anchor` 의 `goal_kind` · `w_rot` · `goal_rot_err_*` · `tol_*` · `within_tolerance` 가 생긴다 (worktree 와 같음). 둘 다 켜지면 §5 의 slot 키가 더해진다.
- **재생산이 필요한 산출물:** 없음. 옛 기록과의 호환은 깨지지 않는다.
- **V2 의 한계:** 재생 도구로는 서버 기록과 bit 동일을 재현할 수 없다 (평가 코드도 106 중 2). 판정은 같은 입력에서 코드 대 코드로 했다.
- **V3 의 조작:** 생성 시간 상한을 양쪽 다 껐다. 실제 서버는 150 ms 상한으로 돈다. 상한에 걸리는 요청에서는 부하에 따라 결과가 갈릴 수 있다 (G1 의 원래 성질).
- **G1 이 E3 위에서만 검증됐다는 결정 문서 한계는 그대로다.** 껍질 구 (A3) · A4 와 함께 돈 closed-loop 은 아직 없다. 팔 덮개와 결합했을 때 보정 0 회였던 기록도 있다.
- **worktree 는 건드리지 않았다.** G7 코드는 worktree 에 남아 있다.
- **메인 작업 트리 상태:** commit 하지 않았다. 작업 중 다른 세션이 docs commit `461231a4` (T64.task.md 만) 를 올려 HEAD 가 바뀌었다. 코드 파일과는 무관하다.

## 내가 기대하는 결과

- 결합 closed-loop (A1–A4 + A7 + G1) 에서는 보정이 켜진 요청이 0 보다 클 것으로 본다.
  - 근거: 가림 121–122 처럼 VLA 닫기가 사과 40 mm 안이면 후보가 생긴다.
- 그 요청에서는 `variant = GEO` 이고 A4 는 대체될 것이다.
- 다만 껍질 구가 손바닥 주변 후보 검사에 영향을 주지 않으므로 (껍질 구는 TO 구 모델, 후보 생성은 손 상자 · 관측 점), 보정 0 회의 위험은 주로 팔 덮개 · 시간 상한 쪽에 있을 것이다.
