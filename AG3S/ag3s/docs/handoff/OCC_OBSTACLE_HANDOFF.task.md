# 가림 장애물 실험 (OCC-APPLE-1345) — 다른 세션 인계

> writer: lead (가림 장애물 세션, 2026-10-27 KST) · 읽는 쪽: 이 실험을 이어받는 새 세션의 lead

## 1. 무엇을 하는 실험인가

- 장면 1345 (사과 target, `--remove-objects orange` 로 오렌지 치움) 에 장애물을 두고, VLA 단독과 AG3S 를 같은 seed 로 돌려 비교한다
- 장애물은 head · 손목 camera 의 사과를 가리거나, VLA 단독 경로를 막는다
- 주 지표는 실제 MuJoCo 접촉과 성공이다. 실패마다 원인 층을 적는다: ATS (target 분할), HSR (sub-task 인식), LTE (링크별 target 면제), CCR (TrajOpt)
- 기록 정본: [AG3S_OCCLUSION_OBSTACLE_LOG.md](../AG3S_OCCLUSION_OBSTACLE_LOG.md)
  - 라운드 1–6, 공통 문제, 개선 방향 A–D 가 있다. 맨 위 "진행 현황" 표부터 읽는다
  - **T47_LOG 에는 쓰지 않는다** (사용자 지시 2026-10-25)
- 사람이 보는 결과: checkout `outputs/2026-10-2*_사과1345_*` 폴더. 각 폴더에 README 가 있다

## 2. 지금까지 알게 된 것 (자세한 숫자는 기록 정본)

| 라운드 | 장애물 | VLA 단독 | AG3S | AG3S 실패 원인 |
|---|---|---|---|---|
| 4 케이스 1 · 1–3 단계 | head 만 가리는 기둥 (가림 28 / 53 / 66 %) | 3/3 · 3/3 · 3/3 | 3/3 · **0/3** · 3/3 | 2 단계: 장애물 여유 10 mm 를 지키려다 쥐는 위치가 22 mm 밀림 (CCR) |
| 4 케이스 3 · 1–3 단계 | head + 손목 가림 + 경로 막는 가로 막대 | 0/3 (모두 접촉) | 0/3 | 1–2 단계: 막힌 접근을 돌아가지 못함 · 손목 camera 에서 사과를 놓침 (CCR). 3 단계: 손바닥 접촉 (충돌 구 범위) |
| 5 | 케이스 1 + 접근 경로 원기둥 (반지름 3 cm) | 0/3 | 0/3 | 팔 마지막 링크 `link_left_arm_6` 접촉. 충돌 구가 손에만 있음 |
| 6 | 케이스 1 + 사과 옆 절반 높이 기둥 (반지름 1.5 cm, 높이 10 cm) | 0/3 | 0/3 | 손바닥 `EE_BODY_L` 접촉. 충돌 구가 손바닥을 덮지 않음 |

- ATS 는 가려도 정확했다. 라운드 4 의 AG3S 18 run 에서 target 점 100 % 가 실제 사과 위에 있었다
- 이 장면의 왼손목 camera 는 사과를 거의 위에서 내려다본다. 세운 기둥으로는 손목 가림을 만들 수 없다 (라운드 6, 1085 자리 모두 0 %). 손목 가림이 필요하면 가로 막대가 필요하다 (라운드 4 케이스 3)
- 다른 세션 (핵심 평가 · TO 실험) 도 이 장면들을 기준선으로 쓴다. 해당 outputs 폴더: `outputs/2026-10-26_사과1345_가림장애물*`

## 3. 다음 후보 (사용자 판정 대기, 아직 안 함)

1. 충돌 구를 팔까지 넓혀 라운드 5 · 6 장면을 다시 돌린다 (`--links gripper_wrist` 또는 `all`)
2. 손바닥 구를 넣는다. 이전 시험 (AG3S-PALM-COVER-1345) 에서 쥐기를 잃었으므로, 쥐기 직전까지만 쓰거나 target 쪽 면을 뺀다
3. 기록 정본의 개선 방향 A–D:
   - A: 쥐기 목표 유지
   - B: 손목 시야 유지
   - C: 손바닥 덮기
   - D: 막힌 접근 감지 · 전역 계획

서빙 구성을 바꾸는 일은 실행 전에 사용자에게 묻는다.

## 4. 재사용할 장면 · 장애물 설정

모든 장면의 공통 조건: episode 1345, seed 13453 · 23453 · 33453, `--remove-objects orange`. 장애물 설정은 client `--obstacle-config <json> --obstacle-profile <name>` 로 준다.

| 장면 | config json | profile |
|---|---|---|
| 케이스 1 · 1 단계 (= 라운드 3 케이스 1) | `R3/placement3/ep1345_noorange_case1_head.json` | `ep1345_noorange_case1_head` |
| 케이스 1 · 2 – 3 단계 | `R3/placement4/ep1345_noorange_case1_L2.json`, `…_L3.json` | `ep1345_noorange_case1_L2`, `…_L3` |
| 케이스 3 · 1 단계 (= 라운드 3 케이스 3) | `R3/placement3/ep1345_noorange_case3_head_wrist_path.json` | `ep1345_noorange_case3_head_wrist_path` |
| 케이스 3 · 2 – 3 단계 | `R3/placement4/ep1345_noorange_case3_L2.json`, `…_L3.json` | `ep1345_noorange_case3_L2`, `…_L3` |
| 라운드 5 (두 장애물, 접근 경로) | `R5/placement/ep1345_noorange_case1_plus_approach.json` | `ep1345_noorange_case1_plus_approach` |
| 라운드 6 (두 장애물, 절반 높이 기둥) | `R6/placement/ep1345_noorange_case1_plus_halfpost.json` | `ep1345_noorange_case1_plus_halfpost` |

경로 약어 (storage `results/development/` 아래):
- `R3` = `20261025T174407+0900_7ddada87f7c1_5e63cdff154f/verify/OCC-APPLE-1345`. 코드 스냅샷은 `20261025T174407+0900_7ddada87f7c1_5e63cdff154f/snapshot_35b909cf_remove` 에 있다
- `R5` = `20261025T232739+0900_b3b69311f948_23b9bd9c6852/verify/OCC-APPLE-1345-R5`
- `R6` = `20261026T101924+0900_b3b69311f948_5d9ca9f0da51/verify/OCC-APPLE-1345-R6`

## 5. 실행 kit (복사해서 새 run 에서 쓴다)

- 실행
  - `R6/p/launch6.sh`: 서버 1 개 + client 여러 개. 사용법 `launch6.sh MODE SUB PORT ITEM…`, ITEM = `이름:seed:config:profile`
    - MODE `vla` = `serve_safe --no-safe`, MODE `ag3s` = 현재 서빙 구성 (`SAFE=` 줄)
    - `OSQP_ALGEBRA_BACKEND=builtin` 이 들어 있다
  - `R6/p/chain6.sh`: VLA 단독 3 개 다음 AG3S 3 개, GPU 독점 lock 하나로 돈다
  - 새로 쓸 때는 `RUN=` 줄과 `V=` 폴더 이름, item 이름만 바꾼다
- 배치 탐색 (CPU)
  - `R5/p/place_second.py`, `R6/p/place_wrist.py`, `R6/p/place6b.py`: 기준은 각 파일 docstring 에 실행 전에 고정해 두었다
  - 공용 함수: `R3/p/place_cases.py` (head · 손목 가림, 경로 거리) 와 `20261025T035831+0900_8da37119afd3_f082f20e9463/verify/AG3S-ROI-CARVE-GPU-SMOKE/p/place_geom.py`
- 집계
  - `R6/p/agg6.py`: 접촉 부위 이름, 정지 시각, TrajOpt 여유 (`to.metrics.clearance_m`), 충돌 구 링크
  - `R3/p3/split4.py`: 서버 하나의 chunk 기록을 client 순서대로 나눈다. 링크는 run 안에 만든다
- 그림
  - `R3/p3/fig_case_views.py`: 3인칭 · head · 손목 + 가림 mask
  - `R3/p3/fig_traj_3p.py`: 3인칭 궤적. VLA 단독 대 AG3S, 접촉 순간
  - `R3/p3/fig_keyframes.py`: 주요 순간
  - `R3/p3/fig_seg_check.py`: 실제 view 에서 target 분할 정답 대조
  - `R3/p3/fig_round4_summary.py`: 결과 격자
- 결과 보기: `common.experiment_view.build(name, notes, confs, scenes, root=outputs/<날짜>_<설명>)` 로 seed 별 하위 폴더를 만든다. 예시는 이 문서 §7 의 명령이다
- CPU 스크립트 공통 환경 (`cd $SN` 후 실행):

```bash
CO=/mnt/dev/AG3S_VLA-development
SN=/mnt/dev/AG3S_VLA-storage/results/development/20261025T174407+0900_7ddada87f7c1_5e63cdff154f/snapshot_35b909cf_remove
export OMP_NUM_THREADS=4 AG3S_STORAGE_ROOT=/mnt/dev/AG3S_VLA-storage CL_SNAP=$SN MUJOCO_GL=osmesa JAX_PLATFORMS=cpu \
  CUDA_VISIBLE_DEVICES= CL_EXTRA_ARGS="--remove-objects orange" \
  PYTHONPATH=$CO:$SN:$SN/integration/rby1/manipulation/src:$CO/third_party/openpi/src:$CO/third_party/openpi/packages/openpi-client/src
```

## 6. 함정 (실제로 겪은 것)

1. **NFS 멈춤**
   - 실행 중 죽은 process 가 쓰던 파일은 열거나 stat 만 해도 그 process 가 D 상태로 멈춘다
   - `R3/round4/server_ag3s_8294_20261025T122720`, `R3/round4/aborted` 는 열지 않는다
   - 서버 `--run-dir` 은 run 전체를 훑으므로 **새 실행은 항상 새 run** (`python -m common.run_output`) 에서 한다
   - 이 세션의 D 상태 process 4 개 (PID 4049618, 4072903, 4106919, 4113879) 는 종료되지 않는다. 건드리지 않는다
2. **run 밖을 가리키는 symlink**
   - run 폴더 안에 다른 run 을 가리키는 symlink 가 있으면 서버가 "Output must be inside run" 으로 바로 죽는다
   - 결과 보기용 kit 은 scratch 에 만든다
3. **GPU lock**
   - AG3S closed-loop 는 `python3 scripts/setup/gpu_lock.py exclusive --est-gib 30 --tag … -- bash chain.sh` 로 돌린다
   - 핵심 평가 세션도 독점 lock 을 자주 쓰므로 대기열에서 기다린다
4. **손목 가림 측정**
   - 장애물 픽셀은 collision geom 이 아니라 visual geom (`pick_obstacle_<slot>_post`) 으로 센다
   - target 가림은 사과 visual geom 픽셀 수의 비율이다
5. **slot 크기 범위**: `pick_place_obstacles.py` 의 `SLOT_PARAMS` 를 따른다
   - `crate_post_0`: 높이 0.05–0.40, 반지름 0.010–0.045
   - `fence_0`: 짧은 가로 막대, 높이 0.03–0.15, 길이 0.08–0.30
   - `overbar_0`: 길이 0.16 이상
   - `bollard_0`: 높이 고정 약 28 cm. 사과 근처에서는 시작 자세의 팔과 붙는다
6. **이름 규칙**
   - 접촉한 로봇 부위는 client 장면의 geom 번호로 기록된다 (`geom#1052` 등). 이름은 `agg6.py` 의 `gname` 으로 푼다
   - 로봇 모델 xml 만 읽으면 번호가 다르다

## 7. 결과 보기 명령 예시 (라운드 6)

```bash
K=<scratch>/view; mkdir -p $K/analysis
for a in VLA AG3S; do mkdir -p $K/runs/${a}_X; for s in 13453 23453 33453; do
  ln -sfn $V/roundN/runs/${a}_X_s$s $K/runs/${a}_X/${a}_X_ep1345_s$s
  echo '{"success": false, "failure_mode": "…"}' > $K/analysis/${a}_X_ep1345_s$s.json; done; done
.venv-openpi-live/bin/python -c "from common.experiment_view import build, CHECKOUT; …build(f'seed{s}', [설명], [('VLA단독', K, 'VLA_X'), ('AG3S', K, 'AG3S_X')], [f'1345_s{s}'], root=CHECKOUT/'outputs'/'<날짜>_<설명>')"
```
