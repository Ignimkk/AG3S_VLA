# AG3S-GPU-STAGES — 남은 느린 단계 세 개(장면 복원 · target 찾기 · TrajOpt 풀이)를 GPU로

> writer: lead · 실행: lead 직접 (순차 작업 — 원인 측정 → 구현 → 동등성 · 속도 확인 → serving 기본값 → 기록)

작성: 2026-10-25 19:10 KST (`date -u` 10:10 UTC + 9 h), lead. **설계 · 지표 · 판정 기준은 결과를 보기 전에 고정한다.**
앞선 STEP: `AG3S-ROI-CARVE-GPU` (관심 영역 지우기 GPU 판정, commit `7ddada87`). 다음 STEP: `AG3S-CORE-EVALUATION` (기준 commit은 이 STEP이 끝난 뒤 정한다, 그 task §0).
기준 코드: `feature/ag3s-development` HEAD `35b909cf` (AG3S · trajopt 코드에 commit 안 된 변경 없음).

> **2026-10-25 20:57 KST:** 이 STEP은 [`AG3S-PRE-CORE-TO.task.md`](AG3S-PRE-CORE-TO.task.md)의 P0가 되었다. 진행 상태 · 재개 방법은 그 문서 §1 표를 따른다. 측정 중간 결과(TrajOpt 시간의 큰 몫이 OpenBLAS thread 기본값이라는 offline 근거 등)는 그 문서 §2에 있다.

## 0. 사용자 판정 (2026-10-25)

| # | 판정 |
|---|---|
| U1 | 남은 느린 단계 세 개를 **핵심 평가 전에 모두** GPU로 옮긴다 |
| U2 | TrajOpt는 **두 경로 모두** 시험한다: (가) 충돌 계산(구 위치 FK · 거리장 조회 · 전체 해상도 검사 · linearize)만 GPU, OSQP는 CPU / (나) QP 풀이까지 GPU |
| U3 | serving 기본값으로 바꾸는 확인 방법: (가)는 **기록 재생에서 같은 명령**, (나)는 **18장면 시뮬레이션**에서 금지 접촉 없는 성공이 CPU 경로보다 나빠지지 않을 것 |

## 1. 이 STEP이 답하는 물음

장면 복원 · target 찾기 · TrajOpt 풀이를 GPU로 옮기면, 출력(또는 시뮬레이션 결과)을 지키면서 서버 한 chunk의 시간이 얼마나 줄어드는가?

현재 값(serving 기본값, 1345번 seed 13453, 31 chunk, run `20261025T035831+0900_8da37119afd3_f082f20e9463` `AG3S-ROI-CARVE-GPU-SMOKE`, lead 집계):

| 단계 (`profile_ms` / `to`) | 중앙값 ms | p95 ms |
|---|---|---|
| `target_grounding` | 79.3 | 163.2 |
| `scene_reconstruction` | 47.5 | 139.8 |
| `esdf` (관심 영역 지우기 GPU 포함) | 127.7 | 227.8 |
| `constraint_generation` (`esdf` 포함) | 170.9 | 274.0 |
| `to.solve_ms` | 141.1 | 668.4 |
| ─ 그중 `timing_ms.check` (전체 해상도 검사) | 31.6 | 299.8 |
| ─ `linearize` | 12.4 | 104.0 |
| ─ `assemble` | 10.4 | 91.4 |
| ─ `qp` (OSQP) | 6.9 | 155.5 |
| ─ 나머지 (`solve − Σtiming`) | 34.4 | 115.2 |

`sqp.time_budget_ms` = 50이지만 `min_iterations` = `max_iterations` = 3이라 예산이 걸린 chunk는 0개다(T31 반복 수 우선).
1 episode 값이라 아래 §2에서 18장면으로 다시 잰다.

## 2. 원인 측정 (구현 전)

- **M1 장면 복원 · target 찾기:** E4 kit(`20261024T034606+0900_8da37119afd3_31d9b16e3c97/verify/AG3S-TASK-DYNAMIC-MAP-E4`)의 18장면 policy record를
  `dynamic_field_replay.run_episode`처럼 E4 서버 argv의 AG3S builder에 차례로 넣는다. `fuse_observations` 안(카메라별 역투영 · 자기 필터 · 융합 · attention 올리기)과
  `ground_target` 안(seed 추출 · region growing · DBSCAN / connected components · 점수 · `TargetConfirm`)을 부분별로 잰다. 도구: `AG3S/ag3s/experiments/studies/stage_profile.py`(신규).
- **M2 TrajOpt:** 같은 18장면 서버 기록을 `AG3S/trajopt/experiments/tp_timing.py`(T43 TP, `fk` · `esdf` · `clearance` · `linearize` · `qp_build` · `osqp` · `objective` · `other`)로 다시 푼다.
  현재 argv로 그대로 도는지 먼저 확인하고, 안 되면 최소 수정만 한다. replay가 서버의 반환 trajectory를 재현하는지(SHA-1)를 함께 적는다.
- 측정 결과로 GPU로 옮길 부분을 고른다. 작은 부분(중앙값 5 ms 미만)은 옮기지 않는다.

## 3. 구현 (모두 기본 꺼짐 flag, 통과한 것만 serving 기본값)

| 경로 | flag (안) | 내용 |
|---|---|---|
| 장면 복원 | `--recon-device {cpu,cuda}` | M1이 가리키는 부분(역투영 · 자기 필터 · voxel 융합 후보)을 torch로. float64 유지 |
| target 찾기 | `--grounding-device {cpu,cuda}` | M1이 가리키는 부분(이웃 탐색 · 연결 성분 후보)을 GPU로. 판정 규칙은 그대로 |
| TrajOpt (가) | `--to-collision-device {cpu,cuda}` | 구 FK · Jacobian, 거리장 · gradient 조회, 전체 해상도 검사를 GPU에서 묶어 계산. OSQP는 CPU |
| TrajOpt (나) | `--qp-backend osqp-cuda` (안) | 첫 후보: OSQP 1.x의 CUDA algebra(같은 ADMM · 같은 설정; 이 checkout의 `.venv-openpi-live`에 CUDA backend 패키지 추가 — 현재 `osqp 1.1.3`, `algebras_available() = ['builtin']`). 설치가 안 되면 대안을 보고하고 판정을 받는다 |

## 4. 판정 (결과 전 고정)

| 기준 | 대상 | 내용 |
|---|---|---|
| G1 동등성 | 장면 복원 · target 찾기 | M1 replay의 모든 요청에서 CPU와 GPU의 융합 점 수 · `target_point_indices` · grounding status · manipulated id가 같다. 하나라도 다르면 기본값을 바꾸지 않고 차이의 위치를 보고한다 |
| G2 동등성 | TrajOpt (가) | M2 replay의 모든 chunk에서 status · 반환 후보 index가 같고 수정된 chunk의 최대 차이 ≤ 1e-6 rad. 어긋나면 기본값을 바꾸지 않고 어긋난 chunk를 보고한다 |
| G3 시뮬레이션 | TrajOpt (나) | E4와 같은 18장면 · seed closed-loop. 금지 접촉 없는 성공 수가 CPU 경로 이상이다. CPU 경로 값은 (가)까지 반영된 serving 기본값의 같은 18장면이다(동등성이 통과하면 E4 기록과 같은 명령이므로 E4 기록을 쓰고, 그 근거를 적는다). 작업 성공 · 금지 접촉 · HOLD · `solve_ms`를 나란히 보고한다 |
| G4 속도 | 옮긴 단계마다 | 같은 replay에서 그 단계 시간 중앙값이 CPU 대비 50 % 이상 줄고 p95도 준다 |
| G5 서버 경로 | 통과한 경로를 모두 켠 서버 | smoke 1 episode(1345, seed 13453)가 정상 종료하고 `profile_ms` 전 단계와 `to.solve_ms`를 §1 표와 나란히 적는다(성공 · 접촉은 판정에 쓰지 않는다) |

G1 · G2 · G4 · G5를 통과한 경로는 `serve_safe`의 serving 기본값에 넣는다. (나)는 G3 · G4 · G5를 통과할 때만 넣는다. 코드 기본값은 `cpu` / `osqp`로 둔다.
되돌리기: 각 flag의 `cpu` / `osqp`. 결과는 핵심 평가 세션의 실행 목록에 명시한다.

## 5. 범위 밖

- 판정 규칙 · 임계값 · 가중치 변경. GPU로 옮기며 알고리즘을 바꾸지 않는다.
- 거리장 단계(이미 GPU)와 VLA 추론.
- palm cover와 이웃 접촉 허용(F1 · F2) — 사용자 판정이 있을 때만.

## 6. 산출물

- 원시: `$AG3S_RUN_DIR/verify/AG3S-GPU-STAGES/`
- `AG3S/ag3s/docs/handoff/AG3S-GPU-STAGES.verify.json`, 그림 `AG3S/ag3s/docs/figures/gpu-stages/` (+ sidecar)
- 로그: `AG3S/ag3s/docs/AG3S_T47_LOG.md`에 이어 쓴다
- 보기: `outputs/2026-10-25_느린단계_GPU변환/`
- GPU 작업은 공용 lock(`/tmp/ag3s-vla-gpu-validation.lock`)을 잡고 `nvidia-smi`에서 다른 compute process가 없을 때만 한다. CPU replay(M2 등)는 lock 없이 할 수 있다.
