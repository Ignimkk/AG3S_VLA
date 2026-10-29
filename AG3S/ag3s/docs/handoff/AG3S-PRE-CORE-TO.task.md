# AG3S-PRE-CORE-TO — 핵심 평가 전 TrajOpt 결정: CPU / GPU, 그리고 잡을 목표를 지키는 궤적 수정

> writer: lead · 실행: lead 직접 (순차 작업 — 앞 실험의 결과가 다음 실험을 정한다). 긴 실행은 background shell.
> **중간에 끊겨도 이 문서만 보고 이어서 할 수 있게 쓴다.** 새 세션은 §1 "재개 방법"부터 읽는다.

작성: 2026-10-25 20:57 KST (`date -u` 11:57 UTC + 9 h), lead. **장면 · 지표 · 판정 기준은 결과를 보기 전에 고정한다** (§4 · §5 · §7). 고정 뒤 바꾸면 §10 변경 기록에 이유와 시각을 남긴다.
기준 코드: `feature/ag3s-development` HEAD `35b909cf` (+ commit 전 측정 코드, §1 표 P0 참고).
다음 STEP: [`AG3S-CORE-EVALUATION`](AG3S-CORE-EVALUATION.task.md). 그 task §0 판정 1("남은 느린 단계를 평가 전에 GPU로")은 이 문서의 P0와 E3로 이어진다.
포함하는 STEP: [`AG3S-GPU-STAGES`](AG3S-GPU-STAGES.task.md) (= 이 문서의 P0). 그 task의 판정 기준 G1–G5는 그대로 유효하다.

## ▶ 새 세션 시작점 (2026-10-26 사용자 지시: "구현이 끝나면 실험 · 결과는 새 세션에서")

**지금 상태 한 줄:** E0 · E1 · E2는 끝났고 E1 · E2 모두 미리 정한 기준 미통과. E3(GPU 경로) **구현은 2026-10-26 ~12:45 KST에 두 갈래 모두 완료**(E3a · E3b impl.md). 실험 · 결과 판정은 새 세션("AG3S - 핵심 평가 세션v2")이 한다.

새 세션은 이 순서로 한다.

1. **구현이 끝났는지 확인:** `AG3S/ag3s/docs/handoff/AG3S-PRE-CORE-TO-E3a.impl.md`(GPU QP, `--qp-backend osqp-cuda`)와 `AG3S-PRE-CORE-TO-E3b.impl.md`(cuRobo로 초기값을 한꺼번에, `--to-gpu-seeds curobo`)가 있고 test 수가 적혀 있는지 본다. 없으면 구현이 덜 된 것이다 — `git status`로 작업 트리를 보고, 새 implementer를 이 문서 §6 "E3 구현 명세"로 띄운다(이전 sub-agent를 재개하지 않는다).
2. **E3 offline 비교:** 두 impl.md의 "full offline 명령"을 그대로 쓴다. 입력은 E2와 같다 — S-OCC `$RUN/verify/AG3S-PRE-CORE-TO/E1/occ_index/occ_index.json`(실행 전 `make_links.sh`, 끝나면 반드시 `make_links.sh --remove`, `find $RUN -type l`에 snapshot의 openpi 링크만 남아야 서버가 뜬다), S-REG `$RUN/verify/AG3S-GPU-STAGES/M2_cpu/e4_index.json`. GPU 시간 측정은 독점 lock. 판정은 §6 E3 · §7 E3(발동 chunk 시간 중앙값 ≤ 533 ms 등).
3. **판정에 따라:** 통과한 갈래 하나만 closed-loop(E1 · E2와 같은 kit 방식: `E2/e2_closed_launch.sh`를 복사해 flag만 바꾼다). 둘 다 못 넘으면 숫자를 보고하고 멈춘다.
4. **결과 보기:** `outputs/<날짜>_<실험>/`을 E1 · E2 때와 같은 방법으로 만든다 — 회귀 장면은 `python -m common.experiment_view`에 kit를 바로, 가림 장면은 `$RUN/verify/AG3S-PRE-CORE-TO/view_occ`처럼 hard link 보기 kit를 만든 뒤(장면 코드 = 배치 2자리 + seed 순번), run별 궤적 그림은 `view_occ_figs/make_figs.sh` 방식(session 13의 `p3/fig_traj.py`, 서버 기록은 `E*/closed_occ/server_split/`로 run별 분할).
4b. **(필수, 사용자 재지시 2026-10-26)** 실험 단계(offline · closed-loop · 진단)가 하나 끝날 때마다 **보고 전에** outputs 를 만들고 보고에 경로를 적는다. E3 것: `outputs/2026-10-26_사과1345_가림장애물18_GPU여러초기값TO_closedloop/`(가림 영상 · offline 그림 · 결과표 · run별 궤적), 회귀 장면은 `outputs/2026-10-26_장애물없는회귀18장면_GPU여러초기값TO_closedloop/`(회귀 closed-loop이 끝나면). 그림 스크립트 `$RUN/verify/AG3S-PRE-CORE-TO/E3/figs/`, 궤적 `view_occ_figs/make_figs_e3.sh`, 보기 kit 항목 `E3/closed_occ/p/make_view.py`.
5. **열린 문제 (사용자 2026-10-26: "영상을 봐도 정확한 원인을 찾기 힘들다"):** 영상과 궤적 그림만으로는 실패 원인이 정해지지 않았다. 남은 실패는 (a) 케이스 1 단계 2 — VLA가 장애물 3–8 mm 옆에서 잡는데 TrajOpt 여유가 10 mm, (b) 케이스 3 단계 1 · 2 — 목표가 생기기 전 손목 camera가 사과를 놓침, (c) 케이스 3 단계 3 — E2가 계획상 침범을 풀었는데도 실제로 막대에 닿음(명령 대 실제 궤적 차이인지, 가려진 막대가 거리장에 빠졌는지 미확인). 원인을 정하려면 영상이 아니라 **숫자로 갈라 보는 진단**(같은 요청에서 계획 여유 · 실제 거리 · 거리장 안의 막대 voxel 비율 · 명령과 실제 관절 차이)이 필요하다 — 무엇을 할지는 새 세션이 사용자에게 묻는다.
6. 그 뒤 **DF**(§1 표): 핵심 평가에 넣을 TO 구성 · 기준 commit을 사용자 판정으로 정하고 [`AG3S-CORE-EVALUATION.task.md`](AG3S-CORE-EVALUATION.task.md) §0을 갱신한다.

**같이 읽을 것:** §1 표(줄별 상태 · 재개 명령), §2(이미 아는 것), `AG3S-PRE-CORE-TO.verify.json`의 `numbers`(P0d · E0_* · E1_* · E2_* · *_verdict), 사람이 보는 결과 2곳(§9 아래 "사람이 보는 결과"). **커밋하지 않은 코드가 많다**(E1 · E2 · `--blas-threads` · 부분별 시간 · E3 진행분) — commit은 사용자 지시가 있을 때만.
**환경 주의:** `osqp-cu12`는 공유 venv에 설치하지 않는다(설치하면 그 venv의 모든 OSQP 기본이 CUDA가 된다 — 2026-10-26 02:32:04–02:34:11 UTC에 한 번 있었고 제거함). GPU QP는 `.cache/osqp-cu12`에서만. NFS가 멈춘 옛 run 폴더(`20261025T174407…/verify/OCC-APPLE-1345/round4/server_ag3s_8294_20261025T122720`, `round4/aborted`)는 열지 않는다.

## 0. 사용자 지시 (2026-10-25 ~20:50 KST)

- GPU 우선순위를 "AG3S 사과 가림 장애물" 세션(`ag3s-vla-development-e2`, OCC-APPLE)에 넘겼다. 가림 상황에서 AG3S를 시험하기 위해서다. 그 세션이 GPU를 다 쓰면 **그 세션이 시험한 환경(장애물 장면)을 그대로 써서** 아래를 구현 · 검증한다.
- 핵심 평가 전에 "TO를 GPU / CPU 중 무엇으로 할지" 실험하기로 한 계획에 [`doc/idea/writing-block_TO.md`](../../../../doc/idea/writing-block_TO.md)(TrajOpt 충돌 회피 · Grasp Goal 유지 개선, 실험 E0–E5)를 더한다.
- 토큰 · 네트워크로 끊길 수 있으니 **재실행 가능한 핵심 평가 전 계획**으로 미리 써 둔다 (이 문서).
- 계획 문서의 최종 목표: **Collision-free 궤적이 아니라, Grasp Goal을 유지하면서 금지 접촉 없이 작업을 성공시키는 궤적.** 계획 문서의 원칙 4: E1에서 해결되면 planner 교체(E2–E4)는 보류한다.

### 0.1 사용자 판정 (2026-10-25 21:05 KST)

- **E1 방식: 둘 다 offline에서 비교한다.** (E1-B) 목표 상자 제약과 (E1-C) 비용 방식 개선(IFG-TOA에서 손바닥 경로 추종 항을 뺀 것)을 같은 고정 입력에서 비교하고, 나은 쪽만 closed-loop으로 돌린다.
- **지금 시작:** GPU를 기다리는 동안 CPU로 할 수 있는 E0 부분(이미 있는 가림 장면 기록의 실패 유형 분류, 목표 자세의 충돌 없는 IK 도달 가능성)과 E1 구현 · unit test를 먼저 한다.

## 1. 재개 방법 (새 세션은 여기부터)

1. `git rev-parse HEAD`, `git status --porcelain AG3S tests` 로 코드 상태를 본다. 아래 표의 "코드" 열과 다르면 그 차이부터 확인한다.
2. 아래 표에서 **상태가 `완료`가 아닌 첫 줄**을 이어서 한다. 각 줄의 "끝났다는 표시" 파일이 있으면 그 줄은 끝난 것이다 (표의 상태보다 파일이 우선).
3. GPU를 쓰는 줄은 `python3 scripts/setup/gpu_lock.py status` 로 lock을 보고, `python3 scripts/setup/gpu_lock.py exclusive --est-gib <N> --tag <tag> -- bash <launcher>` 로 줄을 선다. 시간을 재는 실행 · TrajOpt closed-loop은 **exclusive**다. 다른 세션 process는 건드리지 않는다.
4. 줄 하나를 끝내면 이 표의 상태 · 결과 요약 · 다음 판단을 고치고, 숫자는 `AG3S-PRE-CORE-TO.verify.json` 의 `numbers` 에만 쓴다.
5. 사용자 판정이 필요한 줄(`판정 대기`)에서는 멈추고 묻는다.

공통 경로: RUN = `/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581` (이 계획의 run). 결과는 `$RUN/verify/AG3S-PRE-CORE-TO/<줄 ID>/`, P0는 `$RUN/verify/AG3S-GPU-STAGES/`.

| 줄 | 내용 | 상태 (2026-10-25 20:57 KST) | 끝났다는 표시 | 코드 | 다음 판단 |
|---|---|---|---|---|---|
| P0-a | TrajOpt 부분별 시간 replay, CPU 거리장 (18장면 683 chunk) | 완료 | `AG3S-GPU-STAGES/M2_cpu/*.json` 18개 | HEAD | — |
| P0-b | 같은 replay, 서버와 같은 GPU 거리장 경로 | 완료 | `AG3S-GPU-STAGES/M2_cuda/*.json` 18개 · `GPU_DONE` | HEAD | — |
| P0-c | thread 원인 확인, CPU (1345) | 완료 | `AG3S-GPU-STAGES/M2T_cpu_threads_default/*.json` 3개 | HEAD | — |
| P0-d | 서버 부분별 측정 smoke (1345, thread 기본값) + 같은 smoke `OPENBLAS_NUM_THREADS=1` + GPU 거리장 replay thread 기본값 | **완료** 2026-10-25 23:07 KST. thread 기준 통과: 로봇 명령 32/32 bit 동일, 서버 TrajOpt 풀이 중앙값 135.8 → 36.3 ms (−73 %). replay 18장면 683/683 동일, 119.8 → 43.4 ms | `AG3S-GPU-STAGES/GPU_DONE2` | snapshot `$RUN/snapshot_m1` | verify `numbers.P0d`, 그림 `figures/pre-core-to/p0-openblas-threads` |
| D1 | thread 설정 · 남은 GPU 변환 범위 | **판정 완료** (사용자 2026-10-25 ~23:30 KST): (가) 서버 코드에서 BLAS thread를 기본 1로 — `--blas-threads N` (기본 1, 0 = library 기본), `AG3S/trajopt/blas_threads.py`, test `tests/trajopt/test_blas_threads.py` 2 통과. 함수 방식도 env와 같은 효과 확인(1345 replay 36.6 ms, SHA 동일). (나) **GPU 변환은 여기서 멈춤** — 남은 CPU 부분(점 융합 ~14 ms, target 찾기 ~12 ms)은 옮기지 않는다. TO를 GPU로 할지는 E1 → E3 결과로 정한다 | `blas_threads.py` 존재 | 작업 트리(commit 전) | 핵심 평가 실행 목록에 `--blas-threads 1` 명시 |
| S0 | 장면 목록 고정 (§4) | **완료** 2026-10-25 22:55 KST: S-OCC = OCC round 4 배치 6개(`placement4/case{1,3}_L{1,2,3}`) × seed 13453 · 23453 · 33453 = 18, 장애물 없는 기준 3, S-REG = E4 18 | `$RUN/verify/AG3S-PRE-CORE-TO/S0/scenes.json` + `.sha256` (cd3c0f62…) | — | round 4 AG3S 기록은 session `ag3s-vla-development-13`이 재개 중(11 episode 남음, 우리 P0-d 뒤 독점). 끝나면 그 기록이 E0 closed-loop 기준선 |
| E0 | 현재 TO 기준선 + 실패 유형 분석 + 목표 도달 가능성 | **기준선 closed-loop 확보** (session 13, OCC round 4, 2026-10-26 00:07 KST): 집계 `/mnt/dev/AG3S_VLA-storage/results/development/20261025T232739+0900_b3b69311f948_23b9bd9c6852/verify/OCC-APPLE-1345-R4B/analysis/round4_runs.json`. AG3S: 케이스 1 L1 3/3 · L2 0/3 (못 쥠 2 · 놓침 1, 접촉 0) · L3 3/3, 케이스 3 L1 · L2 0/6 (못 쥠, 접촉 0) · L3 0/3 (접촉 3). VLA 단독: 장애물 없음 3/3, 케이스 1 9/9, 케이스 3 9/9 접촉. 서버는 thread 기본값이었지만 명령은 thread와 무관(P0-d). 목표 도달 가능성 round 4 완료(2026-10-26 02:50 KST, `numbers.E0_goal_reach_round4`): 잡는 자세–장애물 거리 케이스 1 L2 5.2–7.6 mm(여유 10 mm 안) · L3 10.1–12.4 mm, 케이스 3 모든 단계 잡는 자세는 비충돌(6.0–20.8 mm)이나 VLA 경로가 20–38 step 침범 → 케이스 3은 경로 우회(E2) 문제. 실패 유형 표는 `numbers.E0_round4_baseline.taxonomy` | `.../E0/DONE` | — | **NFS 주의:** 옛 run `20261025T174407…/verify/OCC-APPLE-1345/round4/server_ag3s_8294_20261025T122720` 과 `round4/aborted` 는 열지 않는다 (D 상태로 멈춤, session 13 알림) |
| E1 | 목표 유지 TrajOpt: E1-B 상자 제약 · E1-C 비용 개선, offline 비교 → 나은 쪽 closed-loop | **완료 · 미리 정한 기준 미통과** (2026-10-26 02:15 KST). E1-C closed-loop: 가림 장면 금지 접촉 없는 성공 6 → 7 / 18(케이스 1 L2 s13453), 금지 접촉 run 3 → 3. 회귀 장면 금지 접촉 없는 성공 14 → 14 / 18, **금지 접촉 run 1 → 2** — 211(두 구성 모두 실패)에서 사과–왼쪽 바퀴 접촉 2 row(t 197–198, 11.8 mm), 접촉 전 사과 궤적은 미확인. 기준 "금지 접촉이 늘지 않는다"를 넘지 못함 → §7에 따라 E2. 그림 `figures/pre-core-to/e1c-closed-loop`, verify `numbers.E1_verdict` | `.../E1/DONE` | snapshot_e1 | 사용자 판정 대기: E2로 갈지 / 다른 순서 |
| E2 | Multi-seed (초기값 16, 제자리는 마지막 수단, E1-C 없이) | **완료 · 미리 정한 기준 미통과** (2026-10-26 11:25 KST). 가림 장면 금지 접촉 없는 성공 6 → 6 / 18, 금지 접촉 run 3 → 2. 케이스 3 L3에서 HOLD 요청 35 · 29 · 3 → 0이 됐고, s33453은 접촉이 사라졌지만 쥐지 못했고, s13453 · s23453은 계획상 침범을 풀었는데도 실제로 막대에 닿음(최대 0.5 mm). 회귀 장면 15 / 1 / 14 그대로(바뀐 장면 0). 그림 `figures/pre-core-to/e2-closed-loop`, verify `numbers.E2_verdict` | `.../E2/DONE` | snapshot_e2 | 사용자 판정 대기: §7대로 E3 / 케이스 3 L3 접촉 원인 진단 먼저 / DF |
| E3 | GPU 경로: E3-a OSQP CUDA · E3-b cuRobo batched seeds (VLA 추종 항 포함) | **E3-b + 검사 줄이기 closed-loop 완료 · 종합 미통과(E2와 같은 꼴), 사용자 판정 대기** (2026-10-26 15:25 KST, `numbers.E3_closed_occ` · `E3_closed_reg` · `E3_verdict`): 가림 금지 접촉 없는 성공 6(= E0 = E2), 금지 접촉 run 3 → 1, 회귀 14 / 14 / 14 · 금지 접촉 1 / 1 / 1. 기준: U3(CPU 경로 이상) 통과, "E0보다 +1" 미충족. outputs 2개(가림 · 회귀, §9). E3-a offline은 아직 측정 중(scratchpad link 재실행 + 회귀 key). 이전: **offline: E3-b + 검사 줄이기 통과 → closed-loop 실행 중** (14:10 KST, `E3/e3_closed_launch.sh`, code `snapshot_e3c`). offline(`numbers.E3_offline`): E3-b(검사 전부) 68/68 해소 · 544.5 ms → 미통과, E3-b + 검사 줄이기 68/68 + 회귀 2/2 · 302.3 ms → 통과, 궤적은 검사 전부와 967/967 · 683/683 동일. E3-a: 케이스 3 9 key 재실행 중(첫 시도는 검사 줄이기 스크립트가 replay link를 지워 FileNotFound, `E3/a/offline_occ/failed_attempt1/`). 이전 기록: **offline 비교 실행 중** (v2 세션, 2026-10-26 12:48 KST 시작, `$RUN/verify/AG3S-PRE-CORE-TO/E3/e3_offline_launch.sh` · `.out`; 순서 E3-b 가림 → E3-a 가림 → link 제거 → E3-b 회귀 → E3-a 회귀, 명령은 두 impl.md 그대로, 독점 lock). **E3-a 구현 완료** (`AG3S-PRE-CORE-TO-E3a.impl.md`, test 11, trajopt 전체 실질 0 실패; smoke 1 episode에서 발동 chunk당 54.6 s — CPU E2 3.5 s보다 느림, 초기값 풀이 GPU QP 26 %가 반복 한도). **E3-b 구현 완료** (`AG3S-PRE-CORE-TO-E3b.impl.md`, test 6, trajopt 전체 1966 통과 0 실패; smoke 1 episode 35/35 해소, seed 단계 중앙값 428 ms · chunk solve 중앙값 492 / p95 546 ms — 공유 lock 값이라 판정 아님). **다음: 새 세션이 독점 lock으로 E3 offline 비교(두 impl.md의 full 명령).** E3-a의 `osqp-cu12`는 공유 venv에 설치되면 OSQP 기본 algebra가 CUDA로 바뀌어, 02:32:04–02:34:11 UTC(pod 시계, 기록 `E3/a/install/`) 동안만 설치돼 있었고 제거 · `.cache/osqp-cu12`에 따로 설치(pip freeze 설치 전과 같음 확인). 그 창에 이 venv로 뜬 서버 없음(5d 세션 확인) | `handoff/AG3S-PRE-CORE-TO-E3a.impl.md` · `E3b.impl.md` → `.../E3/offline/DONE` → `.../E3/DONE` | 새 flag `--qp-backend osqp-cuda` · `--to-gpu-seeds curobo` (기본 꺼짐) | 사용자 판정 "계획대로 E3" |
| DX | 숫자로 원인 진단 (§6 DX) — **판정표는 잠정 가설** (사용자 16:00) | **완료** (2026-10-26 15:57 KST, `numbers.DX`, outputs `2026-10-26_사과1345_가림장애물_실패원인_숫자진단`): E3 실패 12 = 시야 상실 5(+311*) · 닫았지만 못 쥠 3(케이스 1 단계 2) · TrajOpt 밀어냄 1(312) · target 추정 1(333) · 명령 대 실제 1(331 접촉, 규칙 기준 5 mm가 평소 추종 지연 7 mm보다 작아 구 모델 공백과 구별 못 함; 접촉 4건 모두 EE_BODY_L mesh–막대). 사용자 판정 대기: 다음 층 | `.../DX/DONE` | — | 결과로 다음 층(TO · AG3S · 정책) 결정 |
| DX2 | 접촉 분해 · 케이스 1 단계 2 비교 · 가려짐 대 화면 밖 (§6 DX2) | **완료** (2026-10-26 16:30 KST, `numbers.DX2`, outputs `2026-10-26_사과1345_가림장애물_접촉분해_쥐기비교_시야구분`): ① 접촉 4건 형상 오차 10–20 mm(구 모델 밖 손 끝 몸체 EE_BODY_L), 거리장 · 추종 오차 ±5 mm 안 → DX의 "명령 대 실제" 기각 ③ 121–123 닫는 위치 AG3S 23–24 mm 대 VLA 11–15 mm, HOLD 0, 닫기 전 TrajOpt 수정 ≤ 1.1°, 정책 계획 닫는 위치가 x +15–22 · z +17–25 mm로 밀림, 초반 step 8 수정 2.6°(제약 없음) 뒤 경로가 갈라짐 ② 손목은 가려짐이 아니라 화면 밖 이탈(광축–사과 52–73°), head는 늘 화면 안 · 부분 가림 | `.../DX2/DONE` | — | E4 · DF · 시야 설계 보완을 사용자가 결정 |
| DX3 | 121–123 초반 수정: replay 재현 → 수정 전 제약 → 비용항 분해 → 최소 변경 closed-loop (§6 DX3) | **완료 — 성공 부분 통과, 닫는 거리 기준 미충족** (2026-10-26 17:07 KST, `numbers.DX3.step4_closed_loop`, outputs `2026-10-26_사과1345_케이스1단계2_연속성항끔_closedloop`): `--w-continuity 0`으로 121–123 금지 접촉 없는 성공 0 → 3 / 3, 닫는 순간 손바닥–사과 23–24 → 20–21 mm(기준 ≤ 16 미충족, VLA 단독 11–15). 요청 1–4 수정 0.00°. 회귀 · 다른 가림 장면 영향 미확인. 이전 기록: closed-loop 실행 중 (16:50 KST, `DX3/closed`). 1 재현 ✓(요청 2 · 3 = 0.000°) 2 수정 전 VLA chunk 16행 여유 31–33 mm · 한계 초과 0 3 요청 2 수정은 연속성 항 끔만 0.00°(3/3), 손목 가중은 비활성, 연속성 + 꺾임 둘 다 끄면 닫기 전 수정이 거의 사라짐 → 최소 변경 `--w-continuity 0` (`numbers.DX3`) | `.../DX3/DONE` | — | 최소 변경 closed-loop 판정 |
| SC | 손 끝 몸체 형상 보정 설계 + offline 검증 (§6 SC) | **완료 · 형상만으로는 부족** (19:19 KST, `numbers.SC`, outputs `2026-10-26_손바닥껍질구_형상보정_offline`): `--palm-shell` 껍질 구가 막대 접촉 4건을 덮지만(TrajOpt 여유 −5…−13 mm로 미리 봄) 성공 쥐기 20 중 12 run · 1345 오렌지에서 palm cover와 같은 방해. patch 미적용 | `handoff/AG3S-PRE-CORE-TO-SC.impl.md` | — | 사용자 판정 |
| VW | 손목 시야 유지 · 재관측 설계만 (§6 VW) | **설계안 작성** (16:49 KST, [`AG3S-PRE-CORE-TO-VW.design.md`](AG3S-PRE-CORE-TO-VW.design.md)) — 권장: 구현 전 DX3 최소 변경을 케이스 3 단계 1 · 2에 먼저 | 설계 문서 | — | 구현 여부 사용자 판정 |
| C0F | 연속성 항 끔 — 가림 18 · 회귀 18 (§6 C0F) | **완료 · 채택 제안 아님** (`numbers.C0F`): 가림 6 → 9(121–123 회복), 회귀 14 → 12, 금지 접촉 회귀 1 → 4, 장면별 회귀 119 · 250 · 320(쥐지 못함). 케이스 3 손목 이탈 남음(9/9). outputs 2개(가림 · 회귀 `..._연속성항끔_closedloop`) | `.../C0F/CLOSED_DONE` | — | 채택은 사용자 판정 |
| CR | 연속성 항 완화 분해 → 최소 변경 closed-loop (§6 CR) | **1 · 2단계 완료, 3단계는 PT로 갈음** (23:18 KST 사용자 판정): 1단계 — 회귀 119 · 250 · 320 갈라짐은 연속성 항과 무관(RP: 구성 간 갈라짐). 2단계(`numbers.CR_step2`) — 요청 2 수정은 무게에 비례해 0에서 사라지고, 요청 3 약 2°는 남음, 회귀 경계 점프 거의 같음 → 3단계 기준(요청 2–4 ≤ 0.5°)을 만족하는 값 없음 | `cr2-continuity-sweep.png`, outputs `2026-10-26_연속성항무게_0to0.5_offline재풀이/` | — | PT 결과 |
| SCM | 껍질 구 + 여유 축소 (§6 SCM) | **구현 · offline 완료** (21:07 KST, `numbers.SCM`): 성공 쥐기 방해 요청 22 → 12, 막대 접촉 계속 보임. closed-loop은 LTE 설계 판정 뒤 + GPU 초기값 여유 차이 결정 필요 | `handoff/AG3S-PRE-CORE-TO-SCM.impl.md` | — | closed-loop: PT 결과 뒤 PT 방식(사용자 판정 08:20 KST), 초기값 world 10 mm 대 SQP 5 mm 차이는 기록 후 유지 |
| LTE | target 없는 층의 사과 잔여 설계 (§6 LTE) | **설계 완료** 02:18 KST (`numbers.LTE`): 새 위반 행이 읽는 표면 = 사과 몸체 0 · 꼭지 · 잎(충돌 없는 visual) 49.8 % · 이웃 과일 31.2 % · 쥐는 손 19.0 %. what-if 상한 안 1 −18 % · 안 2 −49 %, 이웃 과일 · 장애물 seed 0개 지움 | `handoff/AG3S-PRE-CORE-TO-LTE.design.md`, `lte-apple-stem-residue.png`, outputs `2026-10-27_사과꼭지잎_target없는층_잔여_설계offline/` | — | 구현 안 함 — 알려진 한계 (08:20 KST 사용자 판정) |
| RP | 같은 장면 · seed 반복성 (§6 RP) | **완료** 22:56 KST: 같은 구성 반복은 9장면 · 3구성 모두 bit 단위로 같음(명령 차 0, 닫는 거리 폭 0). VLA 단독은 2026-10-19 기록과도 같음. 구성이 다르면 요청 0 chunk 차 ≤ 1.2e-7 → step 8 영상 15–66 pixel 차 → 요청 1 chunk 2e-3 – 4e-2 rad 차 → 다른 궤적(명령 차 > 0.01 rad 첫 step E3 9–17, 연속성 끔 27–132) | `numbers.RP`, `rp-repeatability.png`, outputs `2026-10-26_같은장면반복성_가림3_회귀6_closedloop/` | — | 장면당 1 run 비교를 어떻게 읽을지 · CR 2–3단계 (사용자 판정 대기) |
| PT | 작은 흔들림 반복 closed-loop (§6 PT) | **완료** 08:37 KST (`numbers.PT`): 9장면 × 3구성 × 5번. 가림 합계 VLA 15/15 · E3 4/15 · 연속성 끔 10/15, 회귀 합계 9/30 · 17/30 · 13/30. 차 ≥ 4 구성 효과 = 가림 121 · 123 E3 −4 (대 VLA), 회귀 250 · 320 E3 +5 (대 VLA, VLA는 테이블 · 바구니 접촉). E3 대 연속성 끔은 효과 없음(흔들림 안). 섞인 칸 13/27 | `pt-perturbation-repeats.png`, outputs `2026-10-27_작은흔들림5번_가림3_회귀6_closedloop/` | — | 다음 (사용자 판정) |
| E4 | Graph planner (E3 실패 사례만) | 조건부 | `.../E4/DONE` | — | E3 실패 사례가 있을 때만 |
| E5 | 가장 나은 planner + 단계별 접촉 제약 | 대기 | `.../E5/DONE` | — | §7 E5 |
| DF | 핵심 평가에 넣을 TO 구성 · 기준 commit 결정 | 판정 대기 | `AG3S-CORE-EVALUATION.task.md` §0 갱신 | — | 사용자 판정 |

## 2. 이미 아는 것 (다시 하지 않는다 — 근거는 링크)

| 사실 | 근거 | 이 계획에 미치는 영향 |
|---|---|---|
| 서버 TrajOpt 풀이 중앙값 138 ms 중 실제 SQP는 약 40 ms. 서버와 같은 GPU 거리장 replay는 43 ms(p95 77), CPU 거리장 replay와 683/683 bit 동일 | `AG3S-GPU-STAGES` M2_cpu · M2_cuda | 시간의 큰 몫은 계산량이 아니다 |
| 1345 replay에서 thread 제한이 없으면 107 ms (풀이 한 번 CPU 시간 2,363 ms), `OPENBLAS_NUM_THREADS=1`이면 36.7 ms. 결과 SHA 같음 | `AG3S-GPU-STAGES` M2T_cpu_threads_default | 서버는 thread env가 없다. 서버 안 확인이 P0-d |
| target 찾기 replay는 thread와 무관하게 14 ms, 서버 기록 62 ms | `AG3S-GPU-STAGES` M1_grounding · M1T | 서버 안 부분별 측정 필요 (P0-d) |
| 장면 복원은 이미 GPU (T38 B4, `RECON_DEVICE="auto"`, bit 동일) | `config.py:70–82`, `stages/device_recon.py` | 남은 48 ms는 GPU 밖 부분 |
| **목표 고정(비용 방식)은 이미 시험했다**: `--grasp-anchor 2000` (IFG-TOA · TOA2 · TOA3). 954에서 이동 대신 회전으로 피하게 했지만 성공 수 7 / 10 그대로. 211 · 1668은 고정 항이 평행 이동 회피를 막아 HOLD 교착, 안전 우선 대체 풀이도 211 교착을 못 풂 | [WORKLOG 10-21 02:40 – 05:30](../../../../doc/active/reviews/vla_dump_20261015/WORKLOG_S0S3_KO.md), `IFG-APPLE-TOA3.verify.json` | E1은 TOA를 반복하지 않는다. 다른 점을 §6 E1에 적는다 |
| TrajOpt 변형 다섯 가지(target 분리 · look-ahead 끔 · 꺾임 유지 항 끔 · 목표 고정 · 고정 + 대체)가 모두 7 / 10 | 같은 WORKLOG 05:30 | 실패 장면은 바뀌지만 수는 그대로 |
| 손목 가중치(IFG-TOW) 8 / 10, 이웃 접촉 20 mm 허용(IFG-NBC) 9 / 10. 954는 이웃 과일 여유를 지키면서 잡을 자세가 없음(손목 회전 −90°~+90° 전부 침범) | `IFG-APPLE-NBC.verify.json`, WORKLOG 10-21 20:00 | 일부 실패는 planner가 아니라 접촉 허용 규칙(E5)의 문제 |
| 211 교착: SQP 3회 반복 안에서 빠져나올 수 없는 자세(국소 최소). 고정 항이 없으면 54 mm 평행 이동으로 피했음 | WORKLOG 10-21 05:30 | E2(multi-seed)의 직접 동기 |
| cuRobo GPU TrajOpt(L-BFGS)를 서빙 자리에 넣으면 0 / 10 (모두 쥐지 못함). VLA 중간 경로 추종 항이 없다는 가설은 **시험 안 함**(사용자 판정으로 종료) | `T63b.verify.json`, `AG3S_T47_LOG.md` T63b §13 | E3는 T63b의 "되돌아올 지점 (a) · (b)"를 다시 여는 것이다. VLA 추종 항 없는 cuRobo를 다시 돌리지 않는다 |
| 빈 손 후퇴 · 재시도(IFG-REC): 발동했지만 성공 수 변화 0 (탐지가 늦고 후퇴 목표가 틀림) | `IFG-APPLE-REC.verify.json` | 회복은 이 계획의 범위 밖 |
| **가림 케이스 3** (1345, 사과 위 15 cm 가로 막대, head 22 % · 손목 최대 27 % 가림, VLA 경로 25 mm 침범): VLA 단독은 막대에 닿아 실패. AG3S는 접촉 0이지만 TrajOpt가 왼팔을 최대 8.6° 고친 뒤 손목 camera 광축–사과 각도가 41° → 53–63°(화각 ±45° 밖)로 벌어져 사과를 놓쳤고, 손이 막대 아래로 못 내려가 24 cm 떨어진 공중에서 닫음 | `outputs/2026-10-25_사과1345_오렌지제거_가림장애물2케이스/README.md` (e2 세션, 1 episode, 반사실 확인 안 함) | 계획 문서의 원인 4개에 없는 **"수정이 VLA의 시야를 빼앗는다"**를 E0 실패 유형에 넣는다 |
| **E0 측정 (2026-10-25 21:40):** 1345 케이스 3에서 장애물 없는 VLA 단독의 닫는 자세(기준 목표의 IK 해)는 막대와 6.0 mm 떨어져 충돌이 없다. 그러나 TO 장애물 여유 10 mm보다 작다. 같은 경로는 닫기 전 t 123–148(26 step)과 닫은 뒤 들어 올리는 t 167–178(12 step)에서 막대를 파고든다(최대 24.8 mm). 케이스 1은 경로 최소 46.8 mm, 침범 0 | `AG3S-PRE-CORE-TO.verify.json` `numbers.E0_goal_reach_round3` | 케이스 3의 목표는 **물리적으로는 도달 가능하지만 지금 TO 여유 규칙에서는 위반**이다. 상자 제약을 "충돌 우선"으로 두면 목표에서 밀려날 수 있다 → E1 offline에서 이 점을 따로 본다. 들어 올리는 경로도 막혀 있어 운반 우회(E2–E4 · E5 쥔 물체 회피)도 필요하다 |

## 3. 계획 문서와 이 계획의 대응

| 계획 문서 | 이 계획 | 바꾼 점과 이유 |
|---|---|---|
| E0 Baseline | E0 | 실패 유형에 "시야 상실"과 "목표 자체가 도달 불가"를 더한다 (§2 마지막 줄, 원칙 2) |
| E1 Goal Constraint | E1 | 비용이 아닌 **제약**(손바닥 목표 상자)으로 한다. 비용 방식은 IFG-TOA에서 이미 했다 |
| E2 Multi-Seed 16 / 32 / 64 | E2 | 먼저 offline 고정 입력에서 seed 수별 효과를 재고, closed-loop은 가장 나은 하나만 |
| E3 GPU MPPI + L-BFGS | E3 | OSQP CUDA(GPU-STAGES 사용자 판정 U2 (나))와 cuRobo(VLA 추종 · 목표 항 포함)를 같은 고정 입력에서 비교. **TO를 CPU / GPU 중 무엇으로 할지는 여기와 P0에서 정한다** |
| E4 Graph Planning | E4 | E3 실패 사례만, offline 먼저 |
| E5 Selective Contact | E5 | 단계별 · link별 접촉 허용은 대부분 이미 있다(`--target-field-policy exclude-authorized --target-permission-strict`, 쥔 물체 cover, 이웃 접촉 허용). E0에서 접촉 규칙 때문에 실패한 사례가 가리키는 부분만 바꾼다 |

## 4. 장면 (S0, 결과 전 고정)

- **S-OCC (주 장면):** e2 세션(OCC-APPLE)이 시험한 장애물 장면 전부. 이미 있는 것: 1345 · seed 13453, 오렌지 제거 + 케이스 1(head만 가림 기둥) · 케이스 3(head + 손목 가림 + 경로 방해 막대). 배치 기준과 설정 파일: `/mnt/dev/AG3S_VLA-storage/results/development/20261025T174407+0900_7ddada87f7c1_5e63cdff154f/verify/OCC-APPLE-1345/placement3/cases.json`. round 4 장면은 e2의 "끝" 메시지(장면 · seed · 장애물 설정 파일 경로와 sha256, run별 결과 집계 경로)로 받아 그대로 넣는다. **장면을 결과를 보고 고르지 않는다.** e2가 시험한 장애물 장면은 VLA 단독 성공 여부와 관계없이 모두 넣는다.
- **S-REG (회귀 장면):** E4 18장면(`AG3S-TASK-DYNAMIC-MAP-E4`, `seeds.json` sha는 kit의 `seeds.sha256`). 새 장애물은 없다. 바꾼 TO가 장애물 없는 장면을 망가뜨리지 않는지만 본다.
- 오렌지 제거 옵션(`--remove-objects`, `pi05_infer.py`)은 2026-10-25 현재 commit되지 않았다(e2 snapshot `snapshot_35b909cf_remove` + `PATCH_pi05_infer.diff`). S-OCC를 돌릴 때는 그 patch가 들어간 snapshot을 쓴다. commit 여부는 e2가 사용자에게 확인 중이다.
- 고정: `S0/scenes.json` (장면 · seed · 장애물 설정 경로 · sha256 · 오렌지 제거 여부 · 쓸 snapshot) 과 그 sha256.
- 한 구성의 closed-loop은 S-OCC 전부 + S-REG 18. 반복은 장면당 seed 1개(기록과 같은 seed)로 시작한다. 장면이 적어 성공률 판정이 어렵다는 점은 결과에 그대로 쓴다.

## 5. 지표 (결과 전 고정)

| 지표 | 정의 |
|---|---|
| **주 지표: 금지 접촉 없는 성공** | 핵심 평가 task §4와 같다: task 성공(목표 물체를 basket 안에 놓고 2.5 s 유지) ∧ 금지 접촉 없음 ∧ 접촉 기록 완전. 계획 문서의 "Safe Grasp Success"를 이것으로 둔다 |
| Grasp 성공 | 첫 들어 올림 뒤 사과가 두 손가락 사이에 1 s 이상 유지(기존 `robust_grasp` 판정, `p/grasp_robust.py`) |
| 금지 접촉 | 핵심 평가 task §4의 금지 쌍. MuJoCo physics 접촉 `dist < 0`. 장애물 · 이웃 과일 · 비허용 link–target 포함. palm cover 없는 구성이 손바닥 mesh 접촉을 제약에서 못 본다는 한계는 결과와 함께 적는다 |
| 목표 위치 오차 (cm) | 처음 닫는 순간의 손바닥(hold link 원점) 위치와 **기준 목표**의 거리. 기준 목표 = 같은 장면 · seed에서 장애물 없는 VLA 단독 실행이 처음 닫은 순간의 손바닥 위치(사과 기준 좌표로 옮긴 값). 평가에만 쓰고 제어에는 쓰지 않는다 |
| 목표 자세 오차 (deg) | 같은 두 순간의 손바닥 회전 차이(geodesic) |
| 시야 | 쥐기 전 구간에서 쥐는 손 손목 camera 광축과 사과 방향의 각도, 사과 pixel 수(케이스 3의 측정 방법, e2 `p3/wrist_view.py`) |
| 최적화 시간 | 서버 `to.solve_ms` 중앙값 · p95, chunk 왕복 중앙값 · p95 (thread 설정을 같이 적는다) |
| 궤적 길이 · 부드러움 | 실제 궤적(`measured_qpos`) 쥐는 팔 관절 경로 길이(rad), 관절 가속도 제곱합 |
| 원인 위치 | 실패마다 ATS · HSR · LTE · CCR 중 하나와 계획 문서의 원인(목표 이탈 · 국소 최소 · 작업 인식 부족 · 접촉 제약) 또는 "시야 상실" · "목표 도달 불가" · "기타" |

## 6. 실험 설계

### P0 — GPU-STAGES 마무리 (`AG3S-GPU-STAGES.task.md`)

- P0-d 실행 내용은 `gpu_launch2.sh`에 있다: ① 서버 부분별 측정 smoke(1345, E4 구성 + `--profile-substages`, thread 기본값) ② 같은 smoke를 서버 · client 모두 `OPENBLAS_NUM_THREADS=1` ③ GPU 거리장 replay 18장면 thread 기본값.
- 첫 시도는 client run-dir 경로 오류로 시작 못 했다(고침: `M1_server/p/client_args.py`). 실패 기록은 `M1_server/runs/failed_attempt1/`.
- 결과로 D1을 묻는다: (가) `OPENBLAS_NUM_THREADS=1`(또는 threadpool 제한)을 서버 기본으로 둘지, (나) target 찾기 · 장면 복원에서 GPU로 옮길 부분이 남는지.

### E0 — 현재 TO 기준선과 실패 분석

- 구성: 현재 serving 구성(E4 + GPU 관심 영역 판정) + D1 판정. 비교: VLA 단독(같은 장면 · seed). S-OCC 전부 + S-REG 18.
- 이미 있는 기록은 다시 돌리지 않는다: e2가 같은 구성으로 돌린 S-OCC run, E4 18장면 기록. D1으로 serving 구성이 바뀌면(thread 설정만 바뀌고 SHA가 같다면 다시 돌리지 않는다는 근거를 적는다) 다시 돌린다.
- 원칙 2 (목표 유효성): S-OCC 각 장면에서 기준 목표 손바닥 자세에 대해, 기록된 거리장에서 **충돌 없이 도달 가능한 IK 해**가 있는지 offline으로 확인한다(쥐는 팔 7 관절, 손가락 · 손바닥 구 10 mm 여유, 사과 접촉 허용 link만 사과 예외). 도달 불가 장면은 "목표 도달 불가"로 분류하고 E1–E4의 분모에서 빼지 않는다(실패로 남긴다).
- 산출: 장면별 실패 유형 표, 목표 위치 · 자세 오차, 시야 각도 timeline, 실제 궤적 그림.

### E1 — 목표 유지 TrajOpt: 상자 제약(E1-B)과 비용 개선(E1-C)을 offline 비교 (사용자 판정 §0.1)

**구현 명세 (implementer, `E1.impl.md`로 넘긴다).** 기존 IFG-TOA 경로(`AG3S/trajopt/grasp_anchor.py`, `sqp.py`의 `_anchor_state` · 안전 우선 대체 풀이, `safe_policy.py`의 `_grasp_anchor_gate`)를 재사용한다. 모두 기본 꺼짐이고, 꺼져 있으면 출력이 지금과 bit 단위로 같아야 한다.

| 변형 | flag (안) | 내용 |
|---|---|---|
| E1-C 비용 개선 | `--grasp-anchor W --grasp-anchor-path-weight 0` | `GraspAnchor.w_tcp = 0`: 손바닥 경로 추종 항을 빼고 목표 항(닫는 행부터 `grasp_goal_rows` 행)만 둔다. 나머지 gate · 목표 고정 · 대체 풀이는 TOA 그대로 |
| E1-B 상자 제약 | `--grasp-goal-box MM` | 같은 gate · 같은 목표. 목표 행의 손바닥 위치를 SQP 반복마다 선형화해 `g − r ≤ p_k + J_k Δq ≤ g + r`(축마다) 행으로 QP에 넣는다. 이 행에는 전용 slack을 두되 그 가중치를 충돌 slack(`w_slack`)의 1/10로 둬서 **충돌이 먼저**다. merit에도 같은 slack 위반을 넣는다. 풀이가 충돌 침범을 허용 오차 안으로 해소하지 못하면 TOA3의 안전 우선 대체(목표 없이 다시 풀어 침범이 작은 쪽)를 그대로 쓴다. 경로 추종 비용은 넣지 않는다 |

- 기록: `to.metrics.grasp_anchor`에 변형 이름, 목표 행, 목표 오차(손바닥–목표, mm, 축별 최대), 상자 slack, 대체 풀이 여부를 남긴다.
- **offline 비교 도구:** `AG3S/trajopt/experiments/tp_timing.py` / `ta_offline.py` 의 variant로 E0(현재) · E1-C · E1-B(r = 10, 20 mm)를 같은 기록 입력에서 푼다. 기록에 TOA 상태가 없으므로 목표는 기록된 정책 chunk(`reference_chunk`)에서 다시 만든다: 쥐기 전(기록 `summary_json.grasp`의 latch 상태)에서 정책 chunk가 처음 닫기를 계획한 요청의 닫는 행 손바닥 위치를 고정하고, target 중심 120 mm · 손바닥 150 mm 조건은 서버와 같게 적용한다. chunk마다 침범(최대 mm) · 목표 오차 · 상태 · 시간을 쓴다.
- unit test: 꺼짐 bit 동일, E1-C에서 `w_tcp = 0`이 경로 항을 0으로 만드는지, E1-B fixture에서 충돌이 없으면 목표 오차 ≤ r, 충돌과 목표가 충돌하면 침범 해소가 우선하고 slack이 기록되는지, 대체 풀이 경로.
- 실행 입력(먼저): e2 케이스 3 AG3S 기록(`OCC-APPLE-1345/round3/runs/`), E4 기록 중 실패 장면(211 · 954 · 1668 등, `AG3S-TASK-DYNAMIC-MAP-E4/analysis/*.json`의 `success`), round 4 기록은 받으면 추가.

**설계 원래 문장 (작성 20:57, 판정 전):** 아래는 판정 전 초안이며, E1-B가 그 내용이다.

### E1 (초안) — 목표 제약 TrajOpt (새 flag, 기본 꺼짐; 이름은 구현 때 `--grasp-goal-constraint MM`)

- 적용 조건: chunk의 정책 계획에 닫는 행이 있고(0 < 닫는 행 < 실행 창 + look-ahead), 쥐기 전(latched · closing)일 때만. 닫는 행이 없는 chunk에는 목표를 강제하지 않는다(계획 문서 방법 1의 마지막 줄).
- 목표: 그 chunk의 **정책 계획**(TrajOpt 전)이 닫는 행의 손바닥 위치(온라인에서 쓸 수 있는 값; 평가용 기준 목표와 다르다). target 중심에서 120 mm 밖이면 쓰지 않는다(IFG-TOA와 같은 안전 조건).
- **IFG-TOA와 다른 점:** TOA는 손바닥 경로 전체를 따르는 비용 + 목표 비용이었다. E1은 경로 비용을 넣지 않고, 닫는 행부터 3행의 손바닥 위치를 **목표 상자(±r mm) 제약**으로 QP에 넣는다. 충돌 제약과 함께 풀 수 없으면 목표 상자 slack을 충돌 slack보다 먼저 쓰게 하지 않는다(충돌 우선). 평행 이동 회피를 막는 경로 비용이 없으므로 211형 교착(TOA의 결함)이 다시 생기는지를 본다.
- 먼저 offline: E0 실패 chunk의 고정 입력(`tp_timing`과 같은 replay)에서 r = 10 / 20 mm로 목표 오차 · 침범을 잰다. 그다음 closed-loop은 offline에서 나은 r 하나로 S-OCC + S-REG.

### E2 — Multi-seed (조건부: E1이 §7 기준을 못 넘을 때)

- seed: ① 정책 계획(지금과 같다) ② 제자리(T43 Q `_stay_start`, 이미 있음) ③ 장애물 좌 · 우 · 위 우회(손바닥 경로를 장애물에서 먼 쪽으로 5 · 10 cm 옮긴 뒤 관절로 되돌림) ④ 정책 계획 + 관절 noise. 16 / 32 / 64개를 같은 SQP로 각각 풀고, 금지 침범 → 목표 오차 → 궤적 길이 → 부드러움 순으로 하나를 고른다.
- 먼저 offline 고정 입력에서 seed 수별로 "침범 해소 ∧ 목표 오차 ≤ r"인 chunk 비율과 시간을 잰다. CPU 병렬은 worker ≤ 4 규칙 안에서만. 시간이 실시간에 맞지 않으면 그 숫자가 E3의 동기다.

**E2 구현 명세 (사용자 판정 2026-10-26 02:20 KST "계획대로 E2", implementer → `AG3S-PRE-CORE-TO-E2.impl.md`).**

- flag `--to-multi-seed N` (기본 0 = 꺼짐, 꺼지면 bit 동일). **발동 조건:** 그 chunk의 표준 풀이(지금 SQP, E1-C가 켜져 있으면 E1-C 포함)가 침범을 `safety.violation_tolerance` 안으로 풀지 못했을 때만. 그 밖의 chunk는 표준 풀이 그대로다.
- seed (N개, 결정적 — chunk 번호로 정한 RNG): ① 정책 계획 ② 제자리(T43 Q `_stay_start`) ③ 우회: 가장 깊이 침범한 구에서 거리장 gradient로 구한 "장애물에서 멀어지는 방향"과 그 수직 두 방향(좌·우·위)으로 손바닥 경로를 5 · 10 cm 밀되, 행에 따라 매끄러운 bump(시작 · 끝 0)로 주고, 손바닥 Jacobian 의사역행렬로 관절에 되돌린 것 ④ 정책 계획 + 관절 noise(σ 0.02 · 0.05 rad, 시작 행 고정). N이 seed 종류 수보다 많으면 ③ · ④를 늘려 채운다.
- 각 seed를 같은 SQP(같은 설정 · 같은 반복 상한)로 풀고, **고르는 순서:** 최대 침범 → (목표가 있으면) 목표 오차 → 관절 경로 길이 → 가속도 제곱합. 표준 풀이도 후보에 넣는다(그래서 결과가 나빠질 수 없다: 침범이 같으면 표준 풀이 우선).
- 기록 `to.metrics.multi_seed`: 발동 여부, N, 후보별(종류 · 최대 침범 mm · 목표 오차 · 상태), 고른 후보, 전체 시간 ms.
- offline: `tp_timing` setting `E2_16 / E2_32 / E2_64`(E0 위)와 `E1C_E2_16 / 32 / 64`(E1-C 위). 같은 기록 입력(S-OCC `occ_index` — 실행 전 `make_links.sh`, 끝나면 `make_links.sh --remove` / S-REG `e4_index`)에서 chunk별 최대 침범 · 목표 오차 · 고른 후보 종류 · 시간.
- unit test: 꺼짐 bit 동일, 표준 풀이가 침범을 풀면 발동 안 함, fixture(표준 풀이가 국소 최소에 갇히는 장애물)에서 우회 seed가 침범을 푸는지, 고르는 순서, 결정성(같은 입력 같은 결과).
- **판정 (결과 전 고정, §7 E2 그대로):** offline에서 seed 수 하나 이상이 E0 침범 미해소 chunk의 "침범 해소 ∧ 목표 오차 ≤ 10 mm" 비율을 E1-C보다 높이면 그 seed 수 하나만 closed-loop(S-OCC + S-REG). closed-loop 기준은 E1과 같은 꼴(가림 +1 이상, 회귀 줄지 않음, 금지 접촉 늘지 않음).

### E3 — GPU 경로 (조건부: E2까지 미해결이거나 E2의 시간이 기준을 넘을 때)

- (가) OSQP CUDA: `GPU-STAGES` 판정 U2 (나). `.venv-openpi-live`의 OSQP는 1.1.3 `builtin`만 있다 → CUDA algebra 패키지 설치가 먼저다(이 checkout의 venv에만, 설치 전후 `pip freeze` 기록). 판정은 U3대로 시뮬레이션.
- (나) cuRobo MPPI + L-BFGS: T63의 cuRobo 경로에 **VLA 경로 추종 항**(T63b 되돌아올 지점 (a))과 E1 목표 항을 넣는다. VLA 추종 항이 없는 구성은 다시 돌리지 않는다. E2 seed를 MPPI 초기 분포로 쓴다.
- 같은 고정 입력에서 (가) · (나) · CPU SQP(E1 / E2 최선)의 침범 · 목표 오차 · 시간을 나란히 적고, 나은 GPU 후보 하나만 closed-loop.

**E3 구현 명세 (사용자 판정 2026-10-26 ~11:27 KST "계획대로 E3"). 두 갈래는 서로 독립이라 implementer 2개가 병렬로 한다.**

E2의 결과가 출발점이다: 표준 풀이가 침범을 못 푸는 chunk를 E2(초기값 16, `select_moving_first`)가 모두 풀었지만 발동 chunk 하나에 CPU로 약 4 s(실행 창 533 ms의 7배)가 걸린다. E3는 **같은 결과를 실행 창 안에서** 낼 수 있는지 본다.

| 갈래 | flag (기본 꺼짐) | 내용 | 인계 |
|---|---|---|---|
| E3-a GPU QP | `--qp-backend osqp-cuda` | `QpSolver`에 OSQP 1.1.3의 CUDA algebra(`osqp-cu12 1.1.3`, `.venv-openpi-live`에 설치 — 설치 전후 `pip freeze`를 남긴다)를 backend로 추가. 같은 ADMM 설정, 표준 풀이와 E2 seed 풀이 모두에 적용. 풀이가 실패하면 CPU OSQP로 다시 푼다(기록) | `AG3S-PRE-CORE-TO-E3a.impl.md` |
| E3-b GPU batched seeds | `--to-gpu-seeds curobo` (`--to-multi-seed N`과 함께, N개 seed를 GPU에서 한꺼번에) | E2의 seed(`multi_seed.seed_specs`와 같은 생성)를 cuRobo의 batched trajectory optimizer(L-BFGS; MPPI는 선택)에 초기값으로 넣고 한 번에 푼다. 비용 = **VLA 경로 추종(관절, 우리 `w_track`와 같은 꼴)** + smoothness + 충돌(우리 거리장을 cuRobo world로 — T63 `composed_world` 재사용, 여유 10 mm 반영). 목표 항은 앵커가 있을 때만. **VLA 추종 항 없는 cuRobo는 돌리지 않는다**(T63b 0/10). GPU 결과 N개는 우리 TrajOpt의 전체 해상도 검사로 다시 재고 `select_moving_first`로 고른다 — 표준 풀이는 계속 후보 0번 | `AG3S-PRE-CORE-TO-E3b.impl.md` |

- offline 비교(lead): E2와 같은 고정 입력(S-OCC 68 · S-REG 2 침범 chunk)에서 `E2_16`(CPU) · `E3a_16` · `E3b_16`의 해소 비율(움직이는 후보로 해소 ∧ 목표 ≤ 10 mm), 목표 오차, 발동 chunk 시간 중앙값 · p95. GPU 시간 측정은 독점 lock.
- 판정은 §7 E3 그대로: E2 최선보다 침범 · 목표 오차가 나쁘지 않고 **발동 chunk 시간 중앙값 ≤ 533 ms**. 넘는 갈래가 둘이면 시간이 짧은 쪽 하나만 closed-loop(S-OCC + S-REG, 기준은 E1 · E2와 같은 꼴 + GPU-STAGES U3 "금지 접촉 없는 성공이 CPU 경로 이상").
- 둘 다 시간 기준을 못 넘으면 그 숫자를 보고하고 멈춘다(E4로 자동 진행하지 않음).

### DX — 숫자로 원인 진단 (사용자 판정 2026-10-26 15:40 KST "숫자로 원인 진단 먼저", 결과 전 고정)

**왜:** TO 변형 다섯 가지(E0 · E1-C · E2 · E3)가 가림 장면에서 모두 금지 접촉 없는 성공 6–7 / 18이고, 남은 실패 대부분이 "쥐지 못함"이다. 영상으로는 원인이 갈리지 않는다(사용자 2026-10-26). 실패가 TrajOpt(CCR) 층인지, 관측 · 기하(ATS · LTE) 층인지, 명령 대 실제(서보) 차이인지, 정책 자체인지를 **같은 요청 단위 숫자**로 가른다.

**대상 (결과 전 고정):** E3 가림 closed-loop의 실패 12 run(케이스 1 단계 2 셋: 121 · 122 · 123, 케이스 3 아홉: 311–333) + 같은 장면의 지금 serving(E0, session 13 round 4) 12 run — TO 구성이 원인을 바꾸는지 보려고 둘 다. 비교 기준 목표는 §5 정의 그대로 같은 seed의 **장애물 없는** VLA 단독 실행이 처음 닫은 순간의 손바닥 위치(round 4 `VLA_noobst_s<seed>`; 15:45 KST 고침, 결과 전 — 처음 쓴 "장애물 있는 실행"은 §5와 어긋나고 케이스 3에서는 닫기 전에 접촉으로 멈춘다).

**지표 (요청 = 서버 chunk 하나, 실행 8 step):**

| ID | 이름 | 정의 |
|---|---|---|
| M1 | 계획 여유 | 서버 기록 `clearance`(행 × 구)의 실행 행 0–7 최소값(mm) — TrajOpt가 자기 거리장에서 본 여유 |
| M2 | 실제 거리 | 그 8 step의 `measured_qpos`로 MuJoCo를 다시 세워 로봇 충돌 geom–장애물 충돌 geom 최소 거리(mm, `mj_geomDistance`) |
| M3 | 장애물이 거리장에 있나 | 장애물 충돌 geom 표면 위 고른 점(축 · 둘레)에서 기록된 fine · coarse · free 거리장 값을 읽어, 값 ≤ 그 거리장 voxel 크기인 점의 비율(%) (origin = voxel 0 중심, 가장 가까운 voxel) |
| M4 | 명령 대 실제 | 같은 step의 `executed_actions` 대 `measured_qpos` 왼팔 7 관절 최대 차(rad)와, 두 자세의 왼손(hold link) 위치 차(mm) |
| M5 | 쥐기 쪽 | 처음 닫는 순간(t_close, agg4 정의): 손바닥–사과 실제 거리(mm), 직전 요청의 서버 target 중심–실제 사과 중심 거리(mm), 닫기 전 손목 camera에서 사과가 화면 밖인 요청 수(agg4), 닫기 전 16 step을 실행한 요청의 TrajOpt 수정량 최대(deg), 닫는 손바닥 위치의 VLA 단독 대비 오차(사과 기준 좌표, mm) |

**원인 판정 규칙 (결과 전 고정, 위에서부터 처음 맞는 것):**

- 접촉 run (그 접촉 step 기준): ① M3 fine 비율 < 50 % → "거리장에 장애물이 빠짐 (LTE · 관측)" ② M1 ≥ 0 이고 M2 < 0 이고 M4 손 위치 차 > 5 mm → "명령 대 실제 차이 (서보 추종)" ③ M1 < 0 → "TrajOpt가 침범을 못 풂 / HOLD 안 함 (CCR)" ④ 그 밖 → "원인 미상".
- 쥐지 못함 · 놓침 run: ① 서버 target 중심 오차 > 30 mm → "target 추정 (ATS)" ② 닫기 전 사과가 손목 화면 밖인 요청 ≥ 1 → "시야 상실" ③ 손바닥–사과 > 30 mm 이고 닫기 전 TrajOpt 수정량 > 2.9° (0.05 rad) 이고 VLA 단독 대비 닫는 위치 오차 > 30 mm → "TrajOpt가 잡을 위치에서 밀어냄 (CCR)" ④ 손바닥–사과 > 30 mm 이고 수정량 ≤ 2.9° → "정책 자체의 접근 실패 (VLA)" ⑤ 손바닥–사과 ≤ 30 mm → "닫았지만 못 쥠 (쥐기 · 접촉 모델)" ⑥ 그 밖 → "원인 미상".
- 문턱값(50 % · 5 mm · 30 mm · 0.05 rad)은 이 진단을 시작하며 정했고, 결과를 보고 바꾸지 않는다. 바꾸면 §10에 이유와 시각을 남긴다.

**산출:** `$RUN/verify/AG3S-PRE-CORE-TO/DX/`(스크립트 · run별 json · 요약), 그림(실제 씬: 접촉 / 닫는 순간의 장면 + 거리장 단면, 그래프: 요청별 M1 · M2 · M3 · M4 timeline, 표: run별 원인), outputs `outputs/2026-10-26_사과1345_가림장애물_실패원인_숫자진단/`. 숫자는 verify.json `numbers.DX`.

### DX2 — 접촉 분해 · 케이스 1 단계 2 비교 · 가려짐 대 화면 밖 (사용자 판정 2026-10-26 16:00 KST, 결과 전 고정)

사용자 판정: "① 구 모델 공백 확인과 ③ 케이스 1 단계 2 비교를 진행. DX 분류는 잠정 가설로 취급. 접촉은 형상 · 거리장 · 추종 오차를 분리. 쥐기 실패는 첫 회피 수정부터 닫기까지 목표 생성, 손 위치 · 방향, HOLD를 비교. ②는 가려짐과 화면 밖 이탈을 구분한 뒤 설계를 보완. E4와 DF는 이 결과를 보고 결정." → DX 판정표는 **잠정 가설**로 표시한다. 이번에는 판정 규칙 없이 분해된 숫자를 보고한다.

**DX2-A 접촉 분해 (접촉 4 run: E3 331, E0 331 · 332 · 333).** 접촉 step과 그 앞 16 step마다, 서버와 같은 구 모델(`ta_offline.build_robot`, 서버 기록 명령줄의 `--links` · 구 · cover 옵션)과 같은 거리장(`fixed_input_ablation.build_field`, 그 step을 실행한 요청의 기록):
- F_plan = 계획 자세(그 step의 `refined_chunk` 행)에서 구 중심의 **기록된 거리장** 값 − 반지름 (TrajOpt가 본 여유, margin 빼기 전)
- G_plan · G_cmd · G_act = 계획 · 명령(`executed_actions`) · 실제(`measured_qpos`) 자세에서 구 중심과 **실제 막대 geom**의 정확한 거리 − 반지름
- M_act = 실제 자세의 MuJoCo mesh 거리(client 함수)
- 분해: 거리장 오차 = F_plan − G_plan, 추종 오차 = G_plan − G_act (= 계획→명령 + 명령→실제), 형상 오차 = G_act − M_act. 모두 최소 구 / 최소 쌍 기준, 같은 step.
- 덧붙여: 접촉점(`mj_geomDistance` fromto의 로봇 쪽 점)에서 가장 가까운 구 표면까지 거리(> 0 = 접촉점이 구 모델 밖)와 그 구의 link.

**DX2-B 케이스 1 단계 2 비교 (121 · 122 · 123: AG3S(E3 = E0 같은 궤적) 대 VLA 단독(장애물 있음, 성공)).** 구간 = 첫 회피 수정(refined 대 VLA chunk 행 0–7 차 > 0.5° 이거나 HOLD인 첫 요청)부터 처음 닫기까지. step · 요청마다:
- 목표 생성: 정책 chunk가 계획한 닫는 행의 손바닥 위치(사과 기준), AG3S target 상태 · 중심 오차
- 손 위치 · 방향: 실제 왼손(left_ee) 위치와 회전, 같은 step의 VLA 단독과의 차(mm, deg), 닫는 순간 위치(사과 기준) · 방향 차, 닫는 시각 차
- HOLD: HOLD 요청 · 실행 HOLD 수 · 이유, TrajOpt 수정량 · 계획 여유

**DX2-C 가려짐 대 화면 밖 (시야 상실 6 run 311 · 313 · 321 · 322 · 323 · 332 + 같은 장면 VLA 단독).** 닫기 전 요청마다 손목 camera와 head camera에서 사과 segmentation pixel 수를 **장애물이 있을 때(보이는 pixel)**와 **장애물을 치웠을 때(화면 안 pixel)** 둘 다 센다 → 요청을 "보임" · "화면 안이지만 가려짐(치우면 보임)" · "화면 밖"으로 나눈다(보임 비율 < 10 % = 가려짐). 설계 보완은 이 결과를 보고 사용자와 정한다.

산출: `$RUN/verify/AG3S-PRE-CORE-TO/DX2/`, 그림(실제 씬 · 그래프 · 표), outputs `outputs/2026-10-26_사과1345_가림장애물_접촉분해_쥐기비교_시야구분/`, 숫자 `numbers.DX2`.

### DX3 · SC · VW — 사용자 판정 2026-10-26 16:35 KST (결과 전 고정). E4 · DF 보류

**DX3 (① 최우선) — 121–123의 초반 TrajOpt 수정 원인과 최소 변경 closed-loop.** 순서를 지킨다.
1. **replay 재현:** 기록 그대로의 설정(서버 명령줄)으로 121–123의 수정이 난 요청(step 8 · 16, 수정 2.5–2.7° · 2.0°)을 offline으로 다시 풀어, 기록된 `refined_chunk`와 같은지 본다(행 0–7 왼팔 최대 차 ≤ 0.01°를 "재현"으로 본다). 이전 chunk · 명령 이력 · 내부 상태(연속성 · 꺾임 이력 · 손목 추종 상태 · look-ahead · warm start)가 기록과 같은지 입력 단위로 확인한다. 재현이 안 되면 다음 단계로 가지 않고 그 차이부터 보고한다.
2. **"장애물 제약 없음" 확인:** 수정 **전** VLA chunk(`reference_chunk`)의 전체 최적화 구간(plan horizon 16 행, look-ahead 꼬리 포함)에서 모든 충돌 행(구 × 행)의 여유, 관절 위치 · 속도 · 가속도 한계, 시작 행 연속성(rest-start envelope)을 잰다. 하나라도 위반이면 그것이 수정 원인 후보다.
3. **비용항 분해:** 같은 입력에서 항을 하나씩 바꿔 풀고 수정량(행 0–7 왼팔 최대 차)을 잰다 — 손목 가중(`--wrist-track-scale 0.1` → 1.0 = 정상값 복원), 꺾임 유지(`w_deflection_rate` 20 → 0), 연속성 항, look-ahead 끔, 관절 한계 투영 끔, smoothness 항. **VLA 추종 항 끄기는 진단용 대조군**(개선 후보 아님).
4. **최소 변경 하나:** 3에서 수정의 주된 원인이 하나로 좁혀지면 그것만 바꾼 구성(기본 꺼짐 flag 또는 기존 flag 값)으로 121–123 closed-loop(seed 같음, GPU 독점). **판정(결과 전 고정):** 세 run 중 금지 접촉 없는 성공이 0 → 2 이상, 그리고 닫는 순간 손바닥–사과 거리가 VLA 단독 수준(≤ 16 mm). 원인이 하나로 좁혀지지 않으면 closed-loop 전에 보고한다. offline 수정량 감소만으로 끝내지 않는다(다음 VLA 입력이 바뀌므로 인과 확인은 closed-loop).

**SC (②) — 손 끝 몸체 형상 보정 설계와 offline 검증 (implementer, closed-loop 없음).** 접촉 4건의 접촉점이 구 모델 밖 17–28 mm(EE_BODY_L)이다. 이전 palm cover(손바닥 mesh 겉면 전체, 구 270 → 466)는 1345 오렌지에서 쥐기를 잃었다(AG3S-PALM-COVER-1345). 설계는 쥐는 쪽을 막지 않는 덮기를 찾는다. offline 검증 대상: ⓐ 접촉 4건(접촉점이 새 구 모델 안에 드는가, TrajOpt가 그 접촉을 미리 볼 수 있었는가) ⓑ 기존 성공 쥐기(회귀 14 · 가림 6의 닫는 자세에서 새 구가 target · 이웃 · 바닥 행을 위반해 쥐기를 막는가) ⓒ palm cover 실패 사례(1345 오렌지: 새 설계가 그 실패를 피하는가). 기본 꺼짐 flag, 꺼지면 bit 동일.

**VW (③) — 손목 시야 유지 · 재관측 설계만.** 화면 밖 이탈(광축–사과 52–73°)에 대해, 완화 가능한(강제 아닌) 시야 유지와 재관측 중심의 설계안 · 위험 · 시험 방법. 구현 여부는 DX3 결과를 보고 사용자가 정한다.

### C0F — 연속성 항 끔을 가림 18 · 회귀 18에 (사용자 판정 2026-10-26 17:40 KST, 결과 전 고정)

사용자 판정: "E3에서 `--w-continuity 0`만 바꾸고 가림 18 · 회귀 18을 비교. 안전한 작업 성공과 금지 접촉을 주 지표로, 장면별 회귀, chunk 경계 명령 변화, 속도 · 가속도 한계 및 추종 오차도 확인. 케이스 3의 시야 이탈도 같은 실행에서 측정. DX3의 ≤ 16 mm 기준 미충족 기록은 유지. 닫는 거리 분해는 이번 결과에서 쥐기 문제가 남으면. VW 구현과 SC 적용은 이번 비교에 섞지 않는다."

- 구성: E3 kit 그대로(`snapshot_e3c`, E2 + cuRobo 초기값 + 검사 줄이기 + BLAS 1) + `--w-continuity 0` 하나. 장면 · seed · 집계 도구는 E3와 같다(가림: session 13 `agg4.py`, 회귀: `tdm_agg.py` + MuJoCo replay 접촉).
- **주 지표 · 기준 (비교 대상 = E3):** ⓐ 가림 금지 접촉 없는 성공 ≥ 6 / 18 ⓑ 회귀 금지 접촉 없는 성공 ≥ 14 / 18 ⓒ 금지 접촉 run 수가 늘지 않음(가림 ≤ 1, 회귀 ≤ 1) ⓓ 장면별: E3에서 금지 접촉 없이 성공한 장면이 실패로 바뀐 수(회귀 장면 수)를 그대로 보고 — 1 이상이면 그 장면의 원인 위치를 적는다. 넷 다 만족해야 "채택 제안"이고, 채택은 사용자가 정한다.
- **함께 재는 것 (판정 없음, E3와 나란히):** ① chunk 경계 명령 변화 = 새 chunk 첫 실행 명령 − 직전 실행 명령(왼팔 7 관절, rad, 요청마다 최대값의 중앙값 · p95 · 최대) ② 실행 명령의 속도 · 가속도(`executed_actions` 차분 × 15 Hz)의 관절 한계 대비 초과 횟수와 최대 비율(한계 = 서버 `--model-xml`에서 읽는 `limits` 값) ③ 추종 오차 = `executed_actions` 대 `measured_qpos` 왼팔 최대 차(rad)와 왼손 위치 차(mm)의 중앙값 · p95 ④ 케이스 3 손목 시야: 닫기 전 손목 화면 밖 요청 수 · 광축–사과 최대 각(DX2-C 방법) ⑤ 닫는 순간 손바닥–사과 거리(쥐기가 있는 run).
- 산출: `$RUN/verify/AG3S-PRE-CORE-TO/C0F/`, outputs `outputs/2026-10-26_사과1345_가림18_연속성항끔_closedloop/` · `outputs/2026-10-26_장애물없는회귀18장면_연속성항끔_closedloop/`, 숫자 `numbers.C0F`.

### CR · SCM · LTE — 사용자 판정 2026-10-26 20:06 KST (결과 전 고정)

사용자 판정: "연속성 항 완화를 분해 · SC 껍질 + 여유 축소 closed-loop · SC target 없는 층의 사과 잔여 먼저". 순서: LTE 설계를 SCM closed-loop보다 먼저 보고한다(SCM closed-loop은 LTE 설계 판정 뒤). CR은 독립으로 진행.

**CR (lead) — 연속성 항 완화.**
1. C0F 회귀 장면 119 · 250 · 320(E3 성공 → C0F 쥐지 못함)을 E3와 step · 요청 단위로 비교(DX2-B 방법: 손바닥–사과 거리, 손 위치 · 방향 차, 닫는 순간 자세, chunk 경계 명령 변화, TrajOpt 수정량, HOLD).
2. offline 비용 sweep: `w_continuity` ∈ {0, 0.1, 0.25, 0.5}로 121–123(지금 serving 기록)과 119 · 250 · 320(E3 기록)을 다시 풀어(tp_timing, 서버 모드, 기록 입력) 요청별 수정량 · chunk 경계 연속성(새 chunk 0행 − 직전 chunk 실행 행)을 잰다.
3. 최소 변경 하나: 2에서 121–123의 초반 수정(요청 2–4)을 0.5° 이하로 만들면서 119 · 250 · 320의 chunk 경계 연속성을 E3에 가장 가깝게 지키는 값 하나를 고른다(그런 값이 없으면 closed-loop 전에 보고). closed-loop = 가림 18 + 회귀 18, **기준은 C0F와 같은 ⓐ–ⓓ(E3 대비)**.

**SCM (implementer → lead closed-loop) — 껍질 구 + 여유 축소.** `snapshot_sc`에 껍질 구 행만의 여유 flag(`--palm-shell-margin MM`, 기본 = 일반 여유, 그때 bit 동일)를 넣는다. closed-loop 구성(결과 전 고정) = E3 + `--palm-shell on --palm-shell-margin 5`(양손 — 회귀 장면 9개가 오른손), 가림 18 + 회귀 18, 기준 ⓐ–ⓓ(E3 대비) + 막대 접촉 4건이 사라지는가 + TrajOpt 풀이 시간(구 606개). 5 mm는 offline what-if(성공 쥐기 방해 요청 22 → 12)와 거리장 오차(±5 mm 안)를 보고 정했다. closed-loop은 LTE 설계 보고 뒤.

**LTE (설계만) — target 없는 거리장에 남는 사과 표면.** SC ⓑ의 새 위반 행 절반이 target 없는 층에 남은 사과 주변 표면을 가리킨다. 왜 남는지(ball-owned 반경 · ROI carve · 분할) 숫자로 확인하고, 줄이는 설계안 · 위험 · 시험 방법을 낸다. 구현 여부는 사용자 판정.

### RP — 같은 장면 · seed 반복성 (사용자 판정 2026-10-26 20:35 KST "반복성 먼저 측정", 결과 전 고정)

- 장면: 구성 사이 결과가 바뀐 9장면 = 가림 121 · 122 · 123 + 회귀 119 · 250 · 320 · 850 · 954 · 211 (seed 같음).
- 실행: E3 2번(rep1 · rep2) · C0F 2번 · VLA 단독 1번 추가(기존 1번과 합쳐 E3 · C0F 3번, VLA 단독 2번(회귀) / 2번(가림 = round 4 기록 + 1)). kit은 각 구성의 것 그대로, 출력만 rep 폴더.
- 재는 것(판정 없음): ① 같은 구성 반복 사이 결과(금지 접촉 없는 성공 / 실패 유형)가 같은 비율 ② 같은 구성 반복 사이 처음 갈라지는 step(실행 명령 차 > 1e-6 rad)과 손 위치 차 > 1 mm인 step ③ 요청 0(같은 초기 관측)의 VLA chunk가 반복 사이 같은가(최대 차) — 다르면 VLA 추론 비결정성 ④ 닫는 순간 손–사과 거리의 반복 사이 폭.
- 이 결과로 CR 2–3단계 진행 여부와, 앞선 비교(E1 · E2 · E3 · C0F)의 장면당 1 run 판정을 어떻게 읽을지 사용자와 정한다.

### PT · CR 2단계 — 사용자 판정 2026-10-26 23:02 KST (결과 전 고정)

사용자 판정: "작은 흔들림으로 여러 번 (권장)" · "CR offline sweep만 지금 진행 (권장)". 근거: RP — 같은 구성은 bit 단위로 재현되지만 구성이 조금만 달라도 요청 1부터 다른 궤적(`numbers.RP`).

**PT (implementer → lead closed-loop) — 작은 흔들림으로 여러 번.**
- 구현: client(`integration/rby1/bringup/pi05_infer.py`) 기본 꺼짐 flag `--first-chunk-perturb-rad R --perturb-seed K`. 첫 chunk의 실행 명령(step 0–7) 중 쥐는 팔 7관절에 seed K로 뽑은 고정 벡터(각 성분 균일 ±R)를 더한다(gripper · 다른 팔 · 몸통은 그대로). 꺼지면 bit 동일(unit test + VLA 단독 1 run으로 확인). 흔들림 값과 seed를 `traj.npz`에 기록한다.
- R = 1e-6 rad (RP에서 구성 간 실제 차이 ≤ 1.2e-7 rad가 이미 궤적을 갈랐으므로 같은 크기 수준. 손 위치로 ≈ 1 µm).
- 실행: 결과가 바뀐 9장면(가림 121 · 122 · 123 + 회귀 119 · 250 · 320 · 850 · 954 · 211) × 구성 3(VLA 단독 · E3 · E3 + 연속성 항 끔) × 흔들림 seed 1–4 = 108 run. 기존 흔들림 없는 run을 다섯 번째로 더해 장면 · 구성마다 5번.
- 재는 것: 장면 · 구성별 금지 접촉 없는 성공 수(k / 5), 실패 유형, 닫는 순간 손–사과 거리 분포. 9장면 합계의 구성별 성공 수와 95 % 구간(Wilson). 결과가 섞인 칸(0 < k < 5)의 비율 = 흔들림 크기.
- 읽는 규칙(고정): 장면 하나에서 두 구성의 성공 수 차가 4 이상(예: 5 / 5 대 ≤ 1 / 5)일 때만 그 장면의 "구성 효과"로 적는다. 그보다 작으면 "흔들림 안". 9장면 합계 비교는 구간과 함께 보고만 하고 판정은 사용자가 한다.
- 그 뒤: 결과를 보고 36장면(가림 18 + 회귀 18) 전체로 넓힐지 사용자에게 묻는다.

**CR 2단계 (lead, offline):** §6 CR 2단계 그대로 — `w_continuity` ∈ {0, 0.1, 0.25, 0.5}로 121–123(E3 serving 기록)과 119 · 250 · 320(E3 기록)을 다시 풀어 요청별 수정량(행 0–7, deg)과 chunk 경계 연속성(새 chunk 0행 − 직전 chunk 실행 행)을 잰다. 같은 입력의 기전 측정이라 RP와 무관하게 유효하다. 3단계 closed-loop은 PT 방식으로만 한다(값을 고른 뒤 사용자에게 묻는다).

### 껍질 구 5번 반복 (SCM-PT) · 회귀 성공 정의 — 사용자 판정 2026-10-27 08:50 KST (결과 전 고정)

사용자 판정: "SCM을 같은 9장면 × 5번" · "테이블 · 바구니 접촉 별도 표기". 근거: `numbers.PT` (E3 대 연속성 끔은 흔들림 안, 회귀 250 · 320 · 954에서 VLA 단독이 지금 금지 정의로 5번 모두 실패).

- **구성:** E3 + `--palm-shell on --palm-shell-margin 5` (양손), 코드 `snapshot_scpt` = snapshot_sc(껍질 구 · 여유 flag) + 흔들림 client flag. cuRobo 초기값 world는 껍질 행에 10 mm, SQP는 5 mm — 기록하고 그대로 둔다(08:20 판정).
- **실행:** PT와 같은 9장면 × 5번 = 흔들림 없음 1 + 흔들림 seed 1–4 (R = 1e-6 rad) = 45 run. 비교 대상 = PT의 VLA 단독 · E3 5번 결과(다시 돌리지 않음).
- **주 지표 (회귀 성공 정의 변경, 모든 구성 같은 정의로 PT도 다시 집계):** 금지 접촉 = 이웃 과일 · 기둥/막대 · 권한 없는 link–사과(body_target). 테이블 · 바구니 접촉은 보조 지표로 따로 센다(run 수 · 행 수). 가림 장면은 지금처럼 agg4 정의(장애물 접촉).
- **읽는 규칙 (PT와 같음):** 장면 하나에서 성공 수 차 ≥ 4 일 때만 "구성 효과", 합계는 Wilson 95 % 구간과 보고만. 판정은 사용자.
- **함께 재는 것:** 막대 · 기둥 실제 접촉 run 수(가림), 껍질 행 위반 요청 수, TrajOpt 풀이 시간(구 606개, 요청별 중앙값 · p95, 실행 창 533 ms 대비).

### 껍질 구 + 렌더 로봇 mask 5번 반복 — 사용자 지시 2026-10-27 10:43 KST (결과 전 고정)

사용자 지시: "현재 진행 중인 내용이 로봇 구 mask 관련해서 문제가 있다면, 이전에 로봇 구 대신 렌더링했던 방식을 사용해서 수행." (2026-10-19 render mask 보류 판정을 이 실험에 한해 풂.)

- **mask 관련 문제인지 (숫자로 본 것):** depth 에서 로봇을 지우는 단계(self-filter)는 지금 FK 구 mask 다. LTE 에서 껍질 구 새 위반 행 1,663 중 316 (19.0 %) 이 "쥐는 손 자신" 표면 — 손 depth pixel 이 구 mask 를 빠져나가 TSDF 에 들어간 것 (`numbers.LTE`). 이 몫은 mask 문제다. 막대 접촉 4건(손 끝 몸체가 TrajOpt 구 모델 밖)과 꼭지 · 잎(49.8 %) · 이웃 과일(31.2 %)은 mask 문제가 아니다 — 렌더 mask 로 바뀌지 않는다.
- **구성:** 지금 실행 중인 "껍질 구 + 여유 5 mm" 구성 그대로 + `--robot-mask render` (TSDF 에 구 mask ∪ 렌더 robot mask, 서버 MUJOCO_GL=osmesa). 코드 `snapshot_scpt` (렌더 mask 코드는 IFG-CL1 때 것 그대로 있음).
- **실행:** 같은 9장면 × 5번 (흔들림 없음 1 + 흔들림 seed 1–4) = 45 run, 지금 실행 끝난 뒤 독점 lock.
- **비교:** 주 비교 = "껍질 구 + 여유 5 mm" (렌더 mask 효과만 분리), 참고 = 지금 serving · VLA 단독 (PT). 주 지표 · 읽는 규칙은 08:50 고정과 같다 (금지 = 이웃 과일 · 기둥 · 권한 없는 link, 테이블 · 바구니 별도, 장면별 차 ≥ 4 만 구성 효과).
- **함께 재는 것:** 요청별 렌더 시간 · 서버 처리 시간 (실행 창 533 ms 대비; IFG 때 렌더 중앙값 약 166 ms), 막대 · 기둥 접촉 run 수.

### 연결된 애매함 지우기 offline what-if — 사용자 지시 2026-10-27 15:01 KST (결과 전 고정)

사용자 관찰: "잎과 몸통은 cloud 점으로 이어져 있지 않지만 voxel 에서는 이어져 있는데 왜 안 되나" → 지우기 (`ball_owned`) 는 voxel 연결을 보지 않고 voxel 마다 따로 투표한다. 사용자 지시: "수행해보세요".

- **규칙 (what-if, 구현 아님):** `conn_amb` = 공 안 "애매함 → 보존" voxel 중, 지운 target voxel 과 26-이웃으로 (애매함 voxel 을 거쳐) 이어진 것을 추가로 지운다. 대조: `all_amb` = 애매함 전부 지움 (T46c 류 상한), 기존 `oracle_stem` (GT 꼭지 · 잎 8 mm, 상한) · `cap_in` · `cap_all` 을 같은 실행에서 다시 잰다.
- **자료:** LTE what-if 와 같은 19 기록 · 56 요청 (지금 serving 기록, 껍질 구 행 재계산, `whatif_cap.py` 와 같은 방식: 새 값 = d_new − (d_old − 값_old)).
- **주 지표:** 껍질 구 새 위반 행 수 (기준 1,624). **안전 지표:** 규칙이 지운 voxel 중 GT 8 mm 안 가장 가까운 것이 이웃 과일 · 테이블 · 장애물 · 로봇인 수.
- **읽는 규칙:** 안전 지표 0 이고 새 위반 행이 1,624 보다 줄면 closed-loop 후보. 줄어든 크기는 GT 상한 (`oracle_stem`) 과 나란히 보고한다. 안전 지표 > 0 이면 그 class 와 위치를 실제 씬으로 보이고 후보에서 뺀다.
- 공 밖 꼭지 voxel (LTE 533 행) 은 이 규칙의 대상이 아니다 — 남는 몫으로 따로 센다.
- **범위 축소 (사용자 지시 2026-10-27 16:28 KST, "1시간 이내로"):** 19 기록 중 끝난 9 개 + 꼭지 행이 남았던 기록 4 개 (회귀 250 · 309 · 325, 가림 단계 3 s33453) 만 잰다. 꼭지 행이 없던 6 개 (회귀 320 · 572 · 1668, 가림 쪽 753 · 901 · 1596) 는 재지 않는다 — 그 기록의 이웃 과일 · 장애물 제거 수는 "측정 안 함" 으로 보고한다. 이미 본 부분 결과: 회귀 250 요청 8 에서 이웃 바나나 voxel 23 개 제거 (smoke).

### 렌더 로봇 mask 단독 — 공통 실패 장면 5번 반복 (사용자 지시 2026-10-27 16:31 KST, 결과 전 고정)

사용자 지시: "껍질 구 말고, 렌더 마스크만 사용해서 가림 3 · 회귀 6 장면 중 공통적으로 실패한 장면들에 대해서만 수행." + (16:32) "결과를 줄 때 실제 씬 성공률에 더해, 충돌 제약이 어떻게 변했는지 (TSDF/ESDF) · ATS 결과가 어떻게 변했는지도 보여줘야 한다."

- **구성:** 지금 serving (E3 kit, `snapshot_pt`) + `--robot-mask render` 하나 (서버 `MUJOCO_GL=osmesa`). 껍질 구 없음.
- **장면 (결과 전 고정):** "공통 실패" = 지금 serving 이 5 번 중 ≤ 2 번 성공한 장면 — 가림 121 (1/5) · 122 (2/5) · 123 (1/5), 회귀 850 (2/5) · 954 (0/5) · 211 (0/5). 6 장면 × 5 번 (흔들림 없음 + seed 1–4) = 30 run. **→ 사용자 지시 (16:35 KST, 실행 전): 장면마다 2 번 (흔들림 없음 + seed 1) = 12 run.** 2 번으로는 "장면별 차 ≥ 4" 규칙을 쓸 수 없으므로 구성 효과는 판정하지 않고, 같은 장면 · 같은 흔들림의 지금 serving run 과 짝지어 (궤적 동일 여부 · 결과가 바뀌었는지) 만 보고한다.
- **비교:** 같은 장면 · 흔들림의 지금 serving (PT 기록). 주 지표 · 읽는 규칙은 08:50 과 같다 (금지 = 이웃 과일 · 기둥 · 권한 없는 link, 테이블 · 바구니 따로, 장면별 차 ≥ 4 만 구성 효과).
- **함께 보이는 것 (결과 전 고정):** ① 쌍별 실행 궤적 동일 여부와 처음 갈라지는 step ② 서버 처리 시간 · 렌더 시간 ③ **충돌 제약 변화:** 갈라지는 첫 요청 (없으면 쥐기 직전 요청) 에서 두 구성의 TSDF/ESDF 표면 voxel 차이 (렌더 mask 로 사라진 · 생긴 voxel, 손 주변) 와 TrajOpt 가 본 최소 여유, 실제 씬 렌더 ④ **ATS 변화:** 같은 요청의 target 점 · target 지운 층 (지운 · 애매함 voxel) 차이, 실제 씬 렌더.
- **실행 순서 (worker ≤ 4):** 연결된 애매함 what-if 가 끝나면 독점 GPU lock. 가림 15 run 은 서버 1 개로 바로, 회귀 15 run 은 쥐기 자세 측정이 끝나면 두 번째 서버로 (아니면 가림 뒤에).

### 쥐기 목표 항 (기하 보정) 을 지금 serving 목적함수에 — 공통 실패 6 장면 × 2 번 + TrajOpt cost 분석 (사용자 지시 2026-10-27 22:47 KST, 결과 전 고정)

사용자 지시: "우선 목적함수에 기하보정을 넣고 실험 한번 더 해 보세요. 그리고 궤적최적화에서 나오는 cost 를 분석해 보세요. 왜 실패하는지 논리적으로 설명이 가능해야 한다."

- **구성:** 지금 serving (E3 kit, `snapshot_pt`) + 이미 있는 쥐기 목표 항 `--grasp-anchor 2000 --grasp-anchor-path-weight 0` (E1-C 와 같은 가중치). 이 항은 `w_goal · Σ_(닫는 행부터 3 행) ‖손바닥 − g‖²`, g = **VLA 가 처음 닫기를 계획했을 때의 손바닥 위치를 얼린 것** (`grasp_anchor.py`). 코드 변경 없음.
- **장면 · 반복:** 렌더 mask 단독과 같은 6 장면 (가림 121 · 122 · 123, 회귀 850 · 954 · 211) × 2 번 (흔들림 없음 + 흔들림 1) = 12 run. 같은 장면 · 흔들림의 지금 serving run 과 짝 비교 (2 번이라 구성 효과는 판정하지 않는다). 주 지표 08:50 정의 그대로.
- **함께 재는 것 (결과 전 고정):** ① 짝별 결과 · 궤적 동일 여부 ② 닫는 순간 손 중심에서 본 사과 중심 (MuJoCo 정답, 쥐기 자세 진단과 같은 정의) ③ 목표 g 와 실제 사과 중심 · AG3S 사과 중심의 거리, 목표 항이 켜진 요청 수 ④ **TrajOpt cost 분해:** 기록된 요청마다 VLA 원래 궤적 (reference) 과 TrajOpt 결과 (refined) 를 같은 가중치로 항별 재평가 — 추종 · 부드러움 · 연속성 · 꺾임 변화율 · 충돌 위반 벌점 · 쥐기 목표 — 어느 항이 손을 움직였는지. 지금 serving 기록과 이번 기록 둘 다.
- **실행:** 독점 GPU lock, 서버 2 개 (가림 · 회귀 동시), kit `PT/REG_E3G_k*` · `PT/occ/launch_occ_E3G.sh` · `PT/e3g_launch.sh`.

### 기둥 두 개 장면 (OCC-APPLE-1345 라운드 6 배치) — 허들 대신 (사용자 지시 2026-10-27 23:05 KST, 결과 전 고정)

사용자 지시: "grasp 장애물 실험할 때 타겟 위에 허들 놓는 것 하지 말고, 기둥 두 개 쓰는 실험으로 바꿔 보라 — 장면 1345 사과 (OCC-APPLE-1345) 를 활용", 질문 답 (23:05): "현재 배치되어 있는 상태로 수행. 분명 손바닥 부분이 작은 기둥에 걸릴 것", 비교 구성 "+ 쥐기 목표 항 구성".

- **장면:** 라운드 6 배치 그대로 (`OCC-APPLE-1345-R6/placement/ep1345_noorange_case1_plus_halfpost.json`) — 케이스 1 기둥 (r 1.5 cm, 높이 20 cm, head 가림) + 사과 표면에서 2.3 mm 떨어진 높이 10 cm 기둥 (r 1.5 cm). 오렌지 제거. seed 13453 · 23453 · 33453.
- **구성 3:** VLA 단독 · 지금 serving (E3 kit, `snapshot_pt`) · 지금 serving + 쥐기 목표 항 (`--grasp-anchor 2000 --grasp-anchor-path-weight 0`). 9 run, 흔들림 없음.
- **기록 · 지표 (결과 전 고정):** 성공 · 금지 접촉 (로봇 geom ↔ 어느 기둥, 라운드 6 집계 `agg6.py` 와 같은 정의), 최소 로봇–장애물 거리, 요청별 TrajOpt 여유. 라운드 6 (그때 serving · VLA 단독 0 / 3, 손바닥 `EE_BODY_L` ↔ 10 cm 기둥) 과 나란히 보고. 사용자 예상 "손바닥이 작은 기둥에 걸린다" 를 접촉 geom 으로 확인한다.
- **제약 · ATS 변화 (사용자 2026-10-27 16:32):** 접촉 직전 요청의 기록 거리장 (두 기둥이 들어 있는지) · 손 구 모델 여유 (손바닥 몸체가 구 모델 밖인지) · target 점을 실제 씬으로 그린다.
- **실행:** 쥐기 목표 항 실행 (E3G) 이 끝난 뒤 독점 GPU lock, 서버 1 개 (worker ≤ 4), `PT/hp_launch.sh` · `PT/occ/launch_occ_HP.sh`, 출력 `PT/OCC_HP/{vla,serving,goal}`.
- **실행 전 추가 (00:11 KST):** 껍질 구 채택 (사용자 판정 2026-10-27 23:55) 에 따라 넷째 구성 "지금 serving + 껍질 구 (`--palm-shell on --palm-shell-margin 5`, `snapshot_scpt`)" × seed 3 을 더했다 (출력 `PT/OCC_HP/shell`). 앞의 세 구성은 그대로.

### 꼭지 덮개 (stem cap) 구현 — 실험용, 기본 꺼짐 (사용자 지시 2026-10-27 23:36 KST)

사용자 지시: "꼭지 덮개를 구현해 놓고 실험용으로 사용합시다." 근거: offline what-if (`LTE/whatif_cap.py` · `whatif_conn.py`, 13 기록 · 39 요청) 에서 껍질 구 새 위반 행 1,189 → `cap_all` 400 · `cap_in` 894, 지운 voxel 중 이웃 과일 0 (`cap_all` 은 로봇 voxel 2, `cap_in` 은 0) — 연결된 애매함 지우기 (이웃 바나나 101) 와 달리 이웃을 지우지 않았다.

- **규칙 (what-if 와 같은 정의):** target 을 지운 층 (fine target-free, `target_exclusion = ball_owned`) 을 만들 때, 공 (`TargetBall`, 중심 c · 맞춘 반지름 fit_r · 반지름 R) 위쪽의 좁은 원기둥 안에 있는 TSDF seed (표면 voxel) 를 target 으로 보고 추가로 지운다: 수평 (base x · y) 거리 ‖p_xy − c_xy‖ ≤ 20 mm 이고 c_z + 0.5 · fit_r ≤ p_z ≤ c_z + fit_r + 25 mm. 두 방식: `in_ball` = 그중 ‖p − c‖ ≤ R 인 것만 (what-if `cap_in`), `full` = 공 제한 없음 (what-if `cap_all`).
- **flag:** `serve_safe --target-stem-cap {off,in_ball,full}` (기본 off = 지금과 bit 동일), 반경 · 높이 (20 mm · 25 mm · 0.5 fit_r) 는 config 기본값으로 두고 flag 로 바꿀 수 있게. target 이 있고 target-free 층을 짓는 요청에만 (쥔 뒤 · target 없음은 그대로).
- **기록:** summary `target_exclusion` 에 `stem_cap` 블록 (mode · 파라미터 · 지운 seed 수 · 지운 seed ijk 는 기존 `target_free_removed_ijk` 와 구분해 `target_free_stem_cap_ijk` 로).
- **test:** off bit 동일; 합성 사과 (공) + 꼭대기 위 가는 기둥 seed → 지워짐; 옆의 이웃 과일 seed · 공 아래 · 반경 밖 → 안 지워짐; `in_ball` 은 공 밖 seed 를 남김; `ball` (owned 아님) 방식과의 관계.
- **검증 (구현 뒤, verifier):** 같은 39 요청 기록에 구현 코드를 offline 으로 돌려 what-if 숫자 (`cap_in` 894 · `cap_all` 400, 금지 class 제거 0 · 2) 를 재현하는지.
- 구현만 하고 serving 은 바꾸지 않는다. closed-loop 사용은 사용자 판정 뒤.

### 쥐기 목표 항 켜는 조건 넓히기 + 연속성 · 꺾임 항은 운반 · 놓기 단계에만 (사용자 판정 2026-10-28 01:07 KST, 결과 전 고정)

사용자 판정: "연속성과 꺾임 항은 place 단계인 운반 단계에서 사용하고, 목표 항을 켜는 조건을 넓혀서 사용해 보자." 근거: cost 분해 (`numbers.COST`, `numbers.E3G_grasp_goal`) — 연속성 · 꺾임 항이 쥐기 전 충돌과 무관하게 손을 밀고 유지한다. 목표 항은 켜진 가림 4 run 에서 손바닥을 14–23 mm → 1–3 mm 로 되돌려 4/4 성공했지만, 요청 8–9 에 목표가 얼려진 뒤에도 (a) 닫는 행이 최적화 구간 밖 ("close row beyond this horizon", 닫기 직전 4–5 요청) 이거나 (b) chunk 가 닫힌 채 시작 ("starts closed (close row 0)") 이면 목표 행이 없어 작동하지 않았다 (가림 122 · 123 흔들림 없음은 한 번도 작동 안 함).

**구현 (implementer, 기본 꺼짐 flag, 코드 정리 작업 뒤):**
- **목표 행 넓히기** `serve_safe --grasp-goal-window {close,widen}` (기본 `close` = 지금 그대로). `widen`: 목표가 얼려져 있고 anchor 가 켜진 (쥐기 단계 latched · closing, 손바닥이 목표 반경 안) chunk 에서 — (a) 닫는 행 c ≥ H (구간 밖) 이면 목표 행 = 구간의 마지막 G 행, (b) c = 0 (닫힌 채 시작) 이면 목표 행 = 처음 G 행 (손가락이 닫히는 동안 손바닥을 목표에 둔다), (c) 0 < c < H 는 지금 그대로. G = `grasp_goal_rows` (3). 들어 올린 뒤 (held) 에는 지금처럼 꺼진다.
- **연속성 항 쥔 뒤에만** `serve_safe --continuity-after-grasp` (기본 꺼짐): 쥐기 단계가 searching · latched · closing 인 chunk 에서는 연속성 항을 빼고, held (들기 · 운반 · 놓기 · 돌아오기) 부터 쓴다. 이미 있는 `--deflection-rate-after-grasp` (IFG-TOB, 꺾임 항을 쥔 뒤에만) 와 같은 방식 · 같은 단계 판정.
- 기록: summary 의 grasp anchor 기록에 window 규칙 · 적용한 목표 행, `to.metrics` 에 연속성 항이 꺼진 이유 (`continuity_off_by`).

**실험 (구현 뒤, 결과 전 고정):**
- 구성: 지금 serving (E3) + 껍질 구 (채택) + 쥐기 목표 항 (`--grasp-anchor 2000 --grasp-anchor-path-weight 0 --grasp-goal-window widen`) + `--deflection-rate-after-grasp` + `--continuity-after-grasp`.
- 장면: 공통 실패 6 장면 (가림 121 · 122 · 123, 회귀 850 · 954 · 211) + 문제없던 회귀 3 장면 (119 · 250 · 320 — 119 는 껍질 구로 나빠졌던 장면) × 2 번 (흔들림 없음 + 1) = 18 run.
- 비교: 같은 장면 · 흔들림의 지금 serving · 껍질 구 (5 번 반복 기록) · 쥐기 목표 항 (E3G) — 짝 비교, 구성 효과 판정 없음. 주 지표 08:50 정의.
- 함께 보이는 것: 목표 항이 켜진 요청 수 · 목표 행 · 손바닥 되돌림 (cost 분해, 가림 장면), 연속성 · 꺾임 항이 꺼진 요청에서 첫 밀림 (요청 1) 이 사라지는지, 운반 단계의 가속 한계 초과 (연속성 항의 원래 역할).

### 충돌 고려 쥐기 자세 (grasp IK) offline 측정 — 사용자 판정 2026-10-27 15:50 KST (결과 전 고정)

사용자 판정: "grasp 할 때는 이웃 물체 · 장애물을 고려한 IK 를 찾고, 그것을 기반으로 TO 가 수행돼야 충돌을 피하고 쥐는 데 성공한다" → offline 측정 계획 고정. **충돌 없는 쥐기 자세가 없으면 "가벼운 이웃 접촉 허용".** 근거: 이웃 과일 접촉 장면 954 · 211 은 접촉이 쥐는 순간에 시작한다 (954: 손–이웃이 손–사과 첫 접촉과 같은 step, 211: 사과–이웃이 0–6 step 뒤; VLA 단독 · 지금 serving · 껍질 구 + 렌더 mask 모두 같은 꼴, `PT/REG_*/replay_ext`). 지금 TrajOpt 는 VLA 가 정한 쥐는 자세 근처만 고치므로 이 접촉을 고칠 수 없다.

**대상 run (70, 결과 전 고정):**
- 이웃 접촉: 회귀 954 · 211 × (VLA 단독 5 + 지금 serving 5, PT 기록) = 20
- 가림 케이스 1 단계 2: 121 · 122 · 123 × (VLA 단독 5 + 지금 serving 5, PT 기록) = 30
- 가림 케이스 3: 311–333 지금 serving (E3 closed_occ) 9
- 오렌지 1345 손바닥 접촉: AG3S-NBC-1345 기준 run (`20261025T035831+0900_8da37119afd3_f082f20e9463/verify/AG3S-NBC-1345`, 기준 구성) 1
- 대조 (금지 접촉 없이 성공): 회귀 250 · 320 × 지금 serving 5 = 10

**기준 쥐기 자세:** 그 run 에서 쥐는 팔 gripper 명령이 처음 0.9 아래로 내려간 step (t_close) 의 명령 관절 (`executed_actions` 쥐는 팔 7 관절). 닫지 않고 끝난 run 은 같은 seed 장애물 없는 VLA 단독 run 의 t_close 자세 (§5 기준 목표와 같은 정의). 장면 상태 (사과 · 이웃 · 장애물) 는 그 run 기록을 MuJoCo replay 로 t_close 까지 다시 세운다.

**후보 자세 (2,625):** 기준 자세의 손 (hold link) pose 를 사과 중심 기준으로 — 수직축 회전 −30 … +30° (10° 간격, 7) × 기울기 −10 · 0 · +10° (접근축을 수직축 쪽으로, 3) × 평행이동 x · y · z 각 −10 · −5 · 0 · +5 · +10 mm (125). 각 후보를 기준 관절에서 시작한 damped least squares IK (쥐는 팔 7 관절, 관절 한계 안, 수렴 = 위치 ≤ 2 mm · 자세 ≤ 2°) 로 풀고, 수렴 못 한 후보는 버린다.

**판정 세계 둘:**
- **GT (상한):** MuJoCo 충돌 geom. (i) 후보 자세 · 손가락 열림 상태에서 손 전체 (손바닥 몸체 · 두 손가락 subtree) – 이웃 과일 · 장애물 · 테이블 최소 거리. (ii) 그 팔 자세를 고정한 채 gripper 닫기를 물리로 9 control step (0.6 s) 돌린 뒤: 손–이웃 · 사과–이웃 접촉, 이웃 과일 최대 변위, 쥐기 (닫힌 손가락 열림 ≥ 0.6 이고 사과가 두 손가락 사이).
- **관측 (AG3S 가 볼 것):** t_close 직전 요청의 기록 거리장 (target 지운 층, 권한 link 는 target 예외) + 껍질 구 모델 (여유 5 mm) 에서 후보 손 구의 최소 여유. 닫는 동작은 관측 세계로 검사할 수 없다 (그렇게 기록한다).

**고르는 규칙 (벗어남 = 손 위치 차 mm + 회전 차 deg × 2 mm/deg):**
1. 충돌 없음: (i) 이웃 · 장애물 ≥ 5 mm, 테이블 ≥ 0 mm, (ii) 닫는 동안 손–이웃 · 사과–이웃 접촉 0 · 쥐기 성공 — 그중 벗어남 최소.
2. 없으면 **가벼운 이웃 접촉:** (ii) 에서 이웃 과일 변위 ≤ 5 mm, 장애물 · 테이블 접촉 0, 쥐기 성공 — 그중 벗어남 최소.
3. 둘 다 없으면 "쥐기 자세 없음".
- 관측 세계는 (ii) 없이 "관측 여유 ≥ 0 mm 중 벗어남 최소" 하나.

**TO 도달 (다):** 고른 GT 자세 (1 · 2 일 때) 를 E1-C 목표 항 (`--grasp-anchor`) 의 목표로 바꿔, t_close 전 마지막 3 요청 기록 입력에서 지금 serving 설정 (E3 kit) 으로 offline 풀이 → 닫는 행 손바닥 목표 오차 ≤ 10 mm 이고 침범 해소인 요청 비율.

**보고:** run 마다 기준 자세의 (i) · (ii) 값, GT · 관측 각각 고른 규칙 (1 / 2 / 3) 과 벗어남 (mm · deg), GT–관측 일치 (관측이 고른 자세가 GT 에서 1 또는 2 인가), TO 도달. 실제 씬 그림 (기준 자세 대 고른 자세, 이웃 · 장애물 거리) · 그래프 · 표 · outputs.

**closed-loop 으로 가는 기준 (결과 전 고정, 넷 다):** ⓐ 이웃 접촉 20 run 중 GT 규칙 1 또는 2 인 run ≥ 10 ⓑ 관측 세계가 고른 자세가 GT 에서도 1 또는 2 인 비율 ≥ 80 % ⓒ 대조 10 run 중 기준 자세를 그대로 고른 run (벗어남 0) ≥ 9 ⓓ TO 도달 ≥ 80 %. 넘으면 기본 꺼짐 flag 로 구현 → 같은 5번 반복 closed-loop (장면 · 구성은 그때 고정). 못 넘으면 어디서 막혔는지 보고한다.

**평가 정의와의 관계:** 주 지표의 금지 접촉 정의 (08:50, 이웃 과일 접촉 = 금지) 는 바꾸지 않는다. 가벼운 이웃 접촉 규칙으로 고른 run 은 따로 표시해 보고하고, 평가 정의를 바꿀지는 사용자가 정한다.

**자원:** CPU · MuJoCo 만. 연결된 애매함 what-if 와 껍질 구 GPU 실행이 끝나 worker 여유가 생긴 뒤 실행 (worker ≤ 4).

### 원인 분석 후속 세 가지 — 사용자 판정 2026-10-27 23:15 KST (결과 전 고정)

앞 단계: `outputs/2026-10-27_이웃과일_장애물_grasp실패_원인분석/`. 사용자: "네 수행해보세요". 모두 기존 기록 offline · CPU 1 worker · nice 19, 코드 변경 · 새 실행 없음. 결과는 한 outputs 폴더에 모은다.

**(가) 954 닫는 chunk 수정의 비용항 분해.** 지금 serving 954 5번 (E3 k0 + PT k1–4) 에서 쥠 상태가 latched → closing → held 로 가는 요청 (t 104–200) 마다, 서버와 같은 가중치로 목적함수 항 (VLA 추종 · 손목 추종 배율 · 부드러움 · 연속성 · 꺾임 변화율 · 충돌 벌점) 을 VLA 원래 chunk (a) 와 TO 결과 (b) 에서 재평가해 b − a 를 센다 (COST 의 `cost_terms.py` 와 같은 함수 호출, 재계산 merit 이 기록 merit 과 상대 오차 ≤ 1e-3 이어야 유효). 판정 규칙: 수정 ≥ 2° 인 요청에서 b − a 가 가장 큰 음수 (TO 가 줄인) 항을 그 수정의 원인 항으로 적는다. 충돌 벌점이 0 이고 연속성 또는 꺾임 변화율이 최대면 "충돌 무관 수정".

**(나) 기둥 단계 2 (121–123) 대 단계 3 (131–133).** 같은 COST 방법을 단계 3 의 지금 serving 기록 (E3 k0, 3 run) 에 적용하고, 단계 2 의 COST 결과 (k0) 와 짝으로 비교한다: ① 요청 1 의 연속성 수정 (°, 손바닥 mm) ② 기둥 쪽 충돌 벌점이 켜진 요청 수와 손바닥 최대 이동 ③ 닫는 순간 손바닥–사과 거리 (MuJoCo) 와 VLA 단독 대비 차 ④ 거리장이 기둥을 실제보다 두껍게 본 양 (기둥 축–사과 단면, 요청 2). 읽는 규칙: ①–④ 중 단계 2 와 3 이 run 범위가 겹치지 않는 항목만 "차이" 로 적는다.

**결과 (2026-10-27 23:27 KST, `outputs/2026-10-27_원인후속3가지_닫기수정_기둥단계2대3_거리장두께/`):** (가) 954 수정 ≥ 2° 17 요청 모두 유효, 원인 항 연속성 10 · 꺾임 변화율 6 · 충돌 1 → "충돌 무관 수정". (나) ①② · ④ 겹침, ③ 닫는 손–사과 단계 2 22.7–23.8 대 단계 3 15.2–16.1 mm 만 차이. (다) 그늘 · 띠 비율 0.066–0.181 → 기각; 기둥 윗모서리 덩어리는 head 광선이 지나간 자리의 남은 점유 (탐색).

**(다) 거리장 두께의 생성 원인.** 가설: fine TSDF truncation 30 mm → 한쪽에서만 본 표면 뒤 음수 띠 ≈ 20 mm (config.py `fine_truncation_m` 주석의 T32 측정) 가, 카메라 광선이 닿지 않는 그늘 쪽이라 비워지지 않는다. 시험 (기둥 단계 2 · 3, 막대 단계 2 · 3 각 요청 2): 실제 장애물 밖 점유 voxel 을 (i) 어느 카메라에서든 광선이 실제 장애물에 가려지는가 (그늘) (ii) 관측된 표면에서 그 광선 방향으로 30 mm 안인가 (띠) 로 나눠 센다. 판정: 실제 밖 점유 voxel 의 ≥ 80 % 가 (i)∧(ii) 이면 가설 지지, < 50 % 면 기각, 그 사이는 보고만.

### E4 — Graph planner (조건부: E3 실패 사례가 있을 때)

- E3가 우회하지 못한 chunk에만 PRM / RRT(cuRobo graph planner 또는 OMPL)로 손바닥 우회 경로를 만든 뒤 E3의 초기값으로 쓴다. offline만 먼저 하고, closed-loop은 offline에서 효과가 보일 때 사용자에게 묻는다.

### E5 — 가장 나은 planner + 단계별 접촉 제약

- 접근: target · 이웃 · 장애물 모두 피함 / 쥐기: 정해진 손가락–target 접촉만 허용 / 운반: 쥔 물체 형상 포함 회피 — 지금 구현(`exclude-authorized` · `--target-permission-strict` · `--held-body-cover`)이 이미 이 구분을 한다. E0에서 "접촉 제약" 원인으로 분류된 실패가 가리키는 부분만 바꾸고, 바꾼 것이 없으면 E5는 최선 planner의 최종 closed-loop으로 끝난다.
- 이웃 접촉 20 mm 허용(IFG-NBC)은 핵심 평가 범위에서 빠져 있다(핵심 평가 task §2.2). E5에서 다시 넣으려면 사용자 판정을 받는다.

## 7. 판정 기준 (결과 전 고정)

| 줄 | 넘었다고 보는 기준 | 못 넘으면 |
|---|---|---|
| P0 | `AG3S-GPU-STAGES` §4 G1–G5. thread 설정은 "smoke 두 run의 refined chunk가 chunk마다 같고(최대 차이 ≤ 1e-6 rad), 서버 `to.solve_ms` 중앙값이 50 % 이상 줄어듦" | D1에서 사용자에게 숫자를 보이고 묻는다 |
| E0 | 판정 없음(기준선). 실패마다 원인 위치 · 유형이 하나 이상 정해져야 끝 | 정하지 못한 실패는 "원인 미상"으로 남긴다 |
| E1 | S-OCC에서 금지 접촉 없는 성공이 E0보다 1개 이상 많고, S-REG에서 금지 접촉 없는 성공이 E0보다 적지 않으며, 금지 접촉이 늘지 않는다. **넘으면 E2–E4는 보류한다(계획 문서 원칙 4)** | E2로 간다 |
| E2 | offline: seed 수 하나 이상에서 E0 실패 chunk의 "침범 해소 ∧ 목표 오차 ≤ r" 비율이 E1보다 높다. closed-loop: E1 기준과 같은 꼴 | E3로 간다 |
| E3 | 같은 고정 입력에서 E2 최선보다 침범 · 목표 오차가 나쁘지 않고 시간 중앙값이 실행 창(8 step = 533 ms) 안. closed-loop은 GPU-STAGES U3: 금지 접촉 없는 성공이 CPU 경로 이상 | E3 실패 사례가 있으면 E4 |
| E5 | 최종 구성이 S-OCC · S-REG 모두에서 E0 이상, 금지 접촉이 늘지 않음 | 개선 없음으로 기록하고 E0 구성으로 핵심 평가 |
| DF | — | 사용자 판정: 핵심 평가 manifest에 넣을 TO(CPU SQP / GPU 경로 · 새 flag)와 기준 commit |

- 장면이 적어 우연을 배제하지 못한다. 위 기준은 다음 단계로 갈지 정하는 개발 판단이고, 성능 주장이 아니다. 성능은 핵심 평가에서 판정한다.
- 성공 수가 같아도 실패 장면이 바뀌면 그것을 적는다(§2: 변형 다섯 가지가 모두 7 / 10이었던 일).

## 8. 범위 밖

- 회복 · 재시도(IFG-REC 계열), 새 학습, palm cover 확장(F1 · F2 — 사용자 판정이 있을 때만), 핵심 평가 test 장면에서의 튜닝.
- 결과를 보고 장면을 바꾸거나 빼는 것.
- 다른 세션 process 종료, 공유 `.venv-ag3s`, 포트 8123.

## 9. 산출물과 기록

- 원시: `$RUN/verify/AG3S-PRE-CORE-TO/<줄 ID>/` (P0는 `$RUN/verify/AG3S-GPU-STAGES/`)
- 숫자: `AG3S/ag3s/docs/handoff/AG3S-PRE-CORE-TO.verify.json` (`numbers`), 그림 `AG3S/ag3s/docs/figures/pre-core-to/` (+ sidecar)
- 로그: `AG3S/ag3s/docs/AG3S_T47_LOG.md` (scribe)
- 보기: `outputs/<날짜>_<실험 설명>/` (`python -m common.experiment_view`)
- 줄마다 끝나면: 자기 process 종료, `python3 scripts/setup/reclaim_cache.py "$RUN" <읽은 큰 입력…>` 한 줄을 보고에 남긴다.

### 사람이 보는 결과 (사용자 지시 2026-10-26: 직접 보고 원인 분석)

- `outputs/2026-10-26_사과1345_가림장애물18_GPU여러초기값TO_closedloop/` — E3: 가림 18 run × (VLA단독 · 지금serving · CPU여러초기값 · GPU여러초기값) 영상, offline 풀이 시간 그림, 가림 결과표, `그림/run별_궤적/`(331–333 · 312, 지금serving · CPU 그림과 나란히).
- `outputs/2026-10-26_장애물없는회귀18장면_GPU여러초기값TO_closedloop/` — E3: 회귀 18장면 × 3 구성 영상, 결과표.

- `outputs/2026-10-26_사과1345_가림장애물18_목표유지TO_여러초기값TO_비교/` — 가림 18 run × (VLA단독 · 지금serving · 잡을목표유지 · 여러초기값) 영상, 결과 표, `그림/`(요약 그림 4장 + `run별_궤적/` 10장: 121 · 311 · 331–333, session 13의 `p3/fig_traj.py`). 보기 kit: `$RUN/verify/AG3S-PRE-CORE-TO/view_occ`(hard link), 서버 기록 run별 분할: `E{1,2}/closed_occ/server_split/`.
- `outputs/2026-10-26_장애물없는회귀18장면_목표유지TO_여러초기값TO_비교/` — 회귀 18장면 × (지금serving · 잡을목표유지 · 여러초기값) 영상, 결과 표, 결과표 그림.

## 10. 변경 기록

| 시각 (KST) | 변경 | 이유 |
|---|---|---|
| 2026-10-25 20:57 | 작성 | 사용자 지시 §0 |
| 2026-10-25 21:05 | §0.1 판정, E1을 E1-B / E1-C offline 비교로, 구현 명세 추가 | 사용자 판정 |
| 2026-10-26 17:10 | DX3 4단계 closed-loop 판정 기록, outputs 생성 | — |
| 2026-10-27 10:43 | 껍질 구 + 렌더 로봇 mask 5번 반복 계획 고정 (렌더 mask 보류를 이 실험에 한해 풂) | 사용자 지시 |
| 2026-10-27 13:56 | 껍질 구 + 렌더 mask 45 run 끝 (13:44) · 집계 (중간판). 가림 10/15, 회귀 14/30, 테이블 · 바구니 1/30. 회귀 119 에서 1/5 대 지금 serving 5/5 (차 4, 쥐지 못함 4 · 접촉 없음) — 원인 판정은 껍질 구만 회귀 결과 뒤. 렌더 중앙 84 ms, 서버 중앙 421 ms, 533 ms 초과 5.5 %. 껍질 구만 남은 33 run 은 13:44 부터 실행 중. outputs `outputs/2026-10-27_껍질구_렌더로봇mask_5번_가림3_회귀6_closedloop/` | lead |
| 2026-10-27 14:38 | LTE 정정: "꼭지가 target 점에 없음" 은 측정 방식 (anchor 기하와 1 mm 일치) 이 만든 틀린 결론. 다시 잼 (`LTE/stem_check2.json`): target 점 10–16 개가 꼭지 · 잎 위, 몸통 꼭대기 위 6–14 개 → ATS 는 꼭지를 target 에 넣는다. 안 지워지는 원인 = 공 경계 (533 행) + target 점 5 mm 판정에서 남는 꼭지 점 1–3 개 → ambiguous 보존 (264 행). design §3.3 에 정정 note | 사용자 질문 ("연결된 점은 하나의 객체 아니었나") |
| 2026-10-27 14:48 | 사용자 요청으로 꼭지 과정 실제 씬 시각화 (회귀 250 요청 9, 서버 기록을 MuJoCo 장면에 렌더). 재정정: target 기하는 이전 frame 이 아니라 같은 frame (13 요청 모두) — target 점이 기록 cloud 와 다른 점 집합 (`points_mismatch`) 이라 5 mm 판정을 쓴다. 이 요청: 꼭지 위 cloud 점 2 개가 5.03 · 5.92 mm / 44.18 · 44.33 mm 로 두 기준을 근소하게 넘어 꼭지 voxel 45 개가 애매함 → 보존. outputs `outputs/2026-10-27_사과꼭지_TSDF지우기_실제씬확인/`, 그림 `figures/pre-core-to/lte-stem-scene-ep250.png` | 사용자 요청 |
| 2026-10-27 15:50 | 충돌 고려 쥐기 자세 (grasp IK) offline 측정 계획 고정 (§6). 근거: 954 · 211 이웃 접촉은 쥐는 순간 시작 (모든 구성). 충돌 없는 자세 없으면 가벼운 이웃 접촉 허용 (사용자 판정). 평가 정의는 그대로 | 사용자 판정 |
| 2026-10-27 16:20 | 껍질 구 · 껍질 구 + 렌더 mask 5번 반복 집계 끝 (각 45 run). 가림 11/15 · 10/15 (지금 serving 4/15, VLA 단독 15/15), 회귀 14/30 · 14/30 (17/30, 21/30), 테이블 · 바구니 1/30. 고정 규칙 효과: 회귀 119 껍질 구 1 대 지금 serving 5 (두 구성 모두). 렌더 mask 는 45 쌍 중 31 쌍 궤적 bit 동일, 나머지도 step ≥ 104 에서 갈라짐 → 결과 영향 없음, 서버 중앙 253 → 421 ms. `numbers.SCM_SCMR_5draw`, outputs 최종판 | lead |
| 2026-10-27 16:28 | 연결된 애매함 what-if 범위 축소 (사용자 지시 "1시간 이내"): 끝난 9 + 꼭지 행 기록 4 (250 · 309 · 325 · 가림 단계 3 s33453), 6 기록 제외. 가림 쪽 753 진행분은 버리고 재시작 | 사용자 지시 |
| 2026-10-27 16:33 | 렌더 로봇 mask 단독 (지금 serving + render) 공통 실패 6 장면 × 5 번 계획 고정 (§6) · kit `PT/REG_E3R_k*` · `PT/occ/launch_occ_E3R.sh` · `PT/e3r_launch.sh`. 보고에 충돌 제약 (TSDF/ESDF) · ATS 변화 실제 씬 포함 (사용자 16:32) | 사용자 지시 |
| 2026-10-27 16:35 | 렌더 mask 단독: 장면마다 2 번 (흔들림 없음 + seed 1) = 12 run 으로 축소 (사용자 지시, 실행 전). 구성 효과 판정 없이 같은 흔들림 짝 비교만 | 사용자 지시 |
| 2026-10-27 17:08 | 연결된 애매함 지우기 what-if 끝 (13 기록 · 39 요청): 새 위반 행 1,189 → 904, 이웃 바나나 voxel 101 개 제거 → 안전 기준 미통과. 39/39 요청에서 애매함 voxel 이 전부 target 과 이어져 '애매함 전부 지우기' 와 같은 결과. 참고: 꼭지 덮개 1,189 → 400 · 이웃 0. `numbers.LTE_conn_amb`, outputs `2026-10-27_연결된애매함지우기_whatif_offline/`. 렌더 mask 단독 12 run 은 17:05 시작 | lead |
| 2026-10-27 22:02 | 렌더 mask 단독 12 run 끝 (18:35) · 집계 (세션 재시작 뒤). 짝 12: 성공 2 → 3 (122 두 번 좋아짐, 123 흔들림 1 나빠짐), 7 짝 궤적 bit 동일. 점구름 동일, 5 mm 층 표면 차 ≤ 4 voxel, 20 mm 층 일부 요청 347–471 voxel, ATS target · 지우기 동일, 명령 차 ≤ 0.96° → VLA 를 거쳐 갈라짐 (RP 와 같은 현상). `numbers.E3R_render_only`, outputs `2026-10-27_렌더로봇mask단독_공통실패6장면x2_closedloop/`. 쥐기 자세 측정은 verifier 재시작 (이전 세션 종료로 중단) | lead |
| 2026-10-27 22:11 | 충돌 고려 쥐기 자세 offline 측정 끝 (verifier, 70 run). ⓐ 이웃 접촉 20 run 중 정답 세계에서 쓸 자세 있음 20/20 통과 · ⓑ AG3S 가 보는 세계로 고른 자세가 정답에서도 맞음 22/44 = 50 % 미통과 · ⓒ 대조 기준 자세 그대로 10/10 통과 · ⓓ 계획대로 (GPU) 측정 안 함, CPU 변형 45/109 = 41.3 % (닫는 행이 계획 창 안인 요청만 45/46 — 결과 본 뒤 추가한 셈, `definitions.changed_after_results`). 케이스 3 9 run 중 8 은 쥐기 자세 없음 (손–사과 176–293 mm). 가벼운 이웃 접촉 규칙 4 run (211 serving). 기준 넷 다 넘지 못해 closed-loop 보류. `numbers.GIK`, outputs `2026-10-27_충돌고려쥐기자세_offline/` | lead |
| 2026-10-27 22:38 | 진단 (사용자 지시): 쥐기 자세 ⓑ 불일치 22 중 21 은 정적 충돌 거리는 정답 통과 · 닫는 단계 (쥐기 실패 22 · 사과–이웃 6 · 손–이웃 3) 에서 떨어짐, 손 기준점은 원인 아님. 쥐지 못하는 이유: 가림 121–123 에서 지금 serving 은 VLA 단독보다 18 step 일찍, 사과를 손 중심에서 31 mm (VLA 12.6) 비껴 닫음 (15 중 4 성공). 지금 serving 목적함수에 쥐기 목표 항 없음 (`--grasp-anchor` 없음), AG3S tip_alignment 는 측정만. `numbers.GIK_diag`, outputs `2026-10-27_쥐기자세_관측대정답_불일치와_쥐기실패원인_진단/` | lead |
| 2026-10-27 22:47 | 쥐기 목표 항 (E1-C 가중치) 을 지금 serving 에 — 공통 실패 6 장면 × 2 번 + cost 분해 계획 고정 (§6), 실행 시작 | 사용자 지시 |
| 2026-10-27 23:08 | 기둥 두 개 장면 (라운드 6 배치) × 구성 3 (VLA 단독 · 지금 serving · + 쥐기 목표 항) × seed 3 계획 고정 (§6), E3G 뒤 대기열 | 사용자 지시 |
| 2026-10-27 23:12 | cost 분해 1 단계 (verifier, 지금 serving 가림 121–123 × 5 = 15 run): 요청 1 (step 8) 수정은 15/15 충돌 위반 0 인데 연속성 항이 주도 (손바닥 6.3–6.7 mm), 이후 꺾임 변화율 항이 유지, 요청 6–10 에서 VLA 경로 위반 (52 요청) 을 충돌 벌점이 밀어냄 (손바닥 최대 11–19 mm), 손을 사과로 되돌리는 항 없음. `numbers.COST`, outputs `2026-10-27_TrajOpt_cost분해_지금serving_가림121-123/`. 2 단계는 E3G 실행 (GPU lock 대기) 뒤 | lead |
| 2026-10-27 23:36 | 꼭지 덮개 (stem cap) 구현 명세 고정 (§6, 기본 꺼짐 flag `--target-stem-cap`), implementer 시작 | 사용자 지시 |
| 2026-10-27 23:51 | 꼭지 덮개 구현 끝 (implementer, commit 안 함, `AG3S-PRE-CORE-TO-STEMCAP.impl.md`): `--target-stem-cap {off,in_ball,full}` 기본 꺼짐, 새 test 10 통과 (기존 실패 1 은 변경 전부터). lead 검증: 39 요청에서 구현 규칙이 what-if 와 같은 voxel 선택 (full 369 = 369, in_ball 362 = 362, 차이 0). GPU end-to-end 는 아직. `numbers.STEMCAP` | lead |
| 2026-10-27 00:11 | **사용자 판정 (23:55):** ① 렌더 mask 를 구 mask 와 합집합으로 쓰는 이유 질문 (답: 구 mask 가 쥔 물체 구 · target 보호 · 렌더가 모르는 여유를 함께 덮는다, `pipeline._apply_render_mask`) ② **껍질 구 (+ 여유 5 mm) 채택** ③ 꼭지 덮개는 실험 뒤 채택 결정 ④ 연속성 · 꺾임 유지 항은 놓기 단계에서만 쓰는 안 — 쥐기 목표 항 실험 · cost 2 단계 결과를 보고 다시 결정. 후속: 기둥 두 개 장면에 껍질 구 구성 추가 (실행 전), main checkout 에 E3 1 차 검사 · 껍질 구 코드 옮기기 (implementer) | 사용자 판정 |
| 2026-10-28 00:49 | 쥐기 목표 항 12 run 끝 (00:39) · 집계 · cost 2 단계: 짝 12 성공 2 → 5 (좋아짐 121×2 · 122 흔들림 1 · 211 흔들림 1, 나빠짐 850 흔들림 1). 목표 항은 닫는 요청 하나에서만 켜져 손바닥을 13–23 mm 당김 (목표 오차 14–23 → 1–3 mm); 122 흔들림 없음 (목표 행 구간 밖) · 123 흔들림 없음 (닫은 뒤 켜짐) 은 작동 안 함. `numbers.E3G_grasp_goal`, outputs `2026-10-28_쥐기목표항_…/` | lead |
| 2026-10-28 01:07 | **사용자 판정:** 연속성 · 꺾임 항은 운반 · 놓기 단계에만, 목표 항 켜는 조건 넓히기 — 명세 · 실험 계획 고정 (§6). 구현은 코드 정리 작업 뒤 | 사용자 판정 |
| 2026-10-28 01:12 | checkout 정리 끝 (implementer, commit 안 함, `AG3S-PRE-CORE-TO-PORT.impl.md`): E3-b 1 차 검사 · 껍질 구 · 껍질 여유를 checkout 으로 — 14 파일 `snapshot_scpt` 와 bit 동일, `serve_safe.py` 는 꼭지 덮개 4 블록만 더 있음. test: 옮긴 기능 SC 26 · SCM 12 · port 4 · stemcap 10 통과, 실패 2 는 옮기기 전과 같음 (t44 trace key, t43t numpy golden). GPU smoke 안 함. 다음: 목표 행 넓히기 · 연속성 항 쥔 뒤에만 구현 | lead |
| 2026-10-28 01:50 | 목표 행 넓히기 · 연속성 항 쥔 뒤에만 구현 끝 (implementer, `AG3S-PRE-CORE-TO-GGW.impl.md`): `--grasp-goal-window {close,widen}` · `--continuity-after-grasp` 기본 꺼짐, 새 test 16 통과, `tests/trajopt` 2025 통과 · 실패 1 (기존). GPU seed 경로도 같은 넓힌 목표 행을 쓴다. snapshot `snapshot_ggw` (checkout 과 핵심 파일 bit 동일) · kit `PT/REG_GGW_k*` · `PT/occ/launch_occ_GGW.sh` · `PT/ggw_launch.sh` (§6 결합 구성 18 run), 기둥 두 개 장면 뒤 대기열 | lead |
| 2026-10-28 09:20 | 기둥 두 개 장면 12 run 집계 (실행 01:22–02:43): VLA 단독 · serving · serving+목표 항 9 / 9 손바닥 `EE_BODY_L` ↔ 작은 기둥 `crate_post_0` 접촉 정지 (목표 항 = serving 과 bit 동일, 닫는 행이 구간 밖), 껍질 구 3 / 3 접촉 없음 · 쥐지 못함 (40 s). 그림 `figures/pre-core-to/hp-two-post-scene`, outputs `2026-10-28_기둥두개장면_손바닥접촉_4구성x3_closedloop` | §6 기둥 두 개 장면 |
| 2026-10-28 09:20 | 결합 구성 18 run 집계 (실행 04:28–05:59): 가림 6 / 6, 회귀 3 / 12 (serving 7, 껍질 구 6). 회귀 하락 = (1) 쥐기 실패 뒤 닫힌 채 재시도 중 얼린 목표를 100–178 mm · 15–27° 당김 (119 × 2, 850), (2) 명령 변화 ≤ 1.8° 인데 결과 갈림 (320 × 2, 250, 954) — 2 번이라 구성 효과 판정 안 함. 그림 `figures/pre-core-to/ggw-combined-shell-widegoal-carrycont`, outputs `2026-10-28_결합구성_껍질구+넓힌목표항+운반단계연속성_9장면x2_closedloop`. cost 분해 · 운반 가속도 집계 남음 | §6 결합 구성 |
| 2026-10-28 21:15 | **사용자 판정:** ① 렌더 로봇 mask 미채택 ② 쥐기 목표 항은 닫는 행만 버전 채택 (`--grasp-anchor 2000 --grasp-anchor-path-weight 0`), 넓힌 버전 미채택 ③ 꼭지 덮개 · 가벼운 이웃 접촉은 쥐기에서 허용 (채택, 꼭지 덮개 closed-loop 검증 전 · 이웃 접촉 구현 전). 결정 문서 `doc/active/reviews/AG3S_decision/{ADOPTED_KO,NOT_ADOPTED_KO §2,OPEN_ISSUES_KO}.md` | 사용자 판정 |
| 2026-10-28 21:42 | **사용자 판정:** 렌더 로봇 mask 는 켜지 않는다 (핵심 평가 포함). 충돌 모델 결정 문서 머리말의 "render 로 고정" 을 대체 표시, OPEN_ISSUES O8 해결 | 사용자 판정 |
| 2026-10-28 22:05 | **사용자 최종 판정:** 렌더 로봇 mask 는 끈다 (핵심 평가 포함). `ADOPTED_KO.md` A7 로 기록 | 사용자 판정 |
| 2026-10-27 08:50 | 껍질 구 5번 반복 계획 · 회귀 성공 정의(테이블 · 바구니 별도) 고정. E4 보류 이유 사용자에게 설명(DX: 우회 실패형은 312 하나) | 사용자 판정 |
| 2026-10-27 08:37 | PT 결과 기록. 회귀 kit 집계에 address_check.json이 없어 E3 kit의 것(모델 XML md5 b2959b30 같음)을 복사해 다시 집계 | — |
| 2026-10-27 08:20 | LTE: 사과 꼭지 · 잎은 "알려진 한계"(시뮬레이터에서 충돌 없는 visual)로 두고 구현하지 않음. SCM closed-loop은 PT 결과 뒤 PT 방식(장면마다 흔들림 5번)으로, cuRobo 초기값 world의 껍질 행 10 mm 대 SQP 5 mm 차이는 기록하고 그대로 둠 | 사용자 판정 |
| 2026-10-27 02:18 | LTE 설계 보고 기록(`numbers.LTE`, 설계 문서 what-if 자리 채움): 새 위반 행이 읽는 것은 사과 몸체 0 · 꼭지 · 잎 49.8 % · 이웃 과일 31.2 % · 쥐는 손 19.0 % | sub-agent 보고 |
| 2026-10-26 23:18 | CR 3단계는 PT 결과(E3 대 E3 + 연속성 항 끔)로 갈음, 새 무게 추가 없음. PT 구현 완료(implementer, snapshot_pt) | 사용자 판정 "PT 결과로 갈음" |
| 2026-10-26 23:14 | CR 2단계 결과: 요청 2 수정은 무게에 비례(0에서 0°), 요청 3 약 2°는 남음, 회귀 경계 점프 거의 같음 → 3단계 기준을 만족하는 값 없음, 사용자에게 보고 | — |
| 2026-10-26 23:02 | PT(작은 흔들림 반복) · CR 2단계 계획 고정 | 사용자 판정 |
| 2026-10-26 22:56 | RP 결과 기록: 같은 구성은 완전 재현, 구성 간 차이는 요청 1부터 영상 pixel 차로 갈라짐. VLA 단독 회귀 반복 kit 검사(동적 거리장 서버 flag 검사)가 bare-policy 서버에 걸려 두 번 막힘 → E0만 그 검사 건너뛰게 고쳐 재실행 | — |
| 2026-10-26 21:07 | SCM implementer 결과 기록, outputs 생성 | implementer 보고 |
| 2026-10-26 20:35 | RP(반복성) 계획 고정 | 사용자 판정 |
| 2026-10-26 20:12 | CR 1단계 결과: 회귀 실패는 연속성 항과 무관, run 흔들림 의심 → 2–3단계 보류하고 사용자에게 반복 측정 여부 질문 | — |
| 2026-10-26 20:06 | CR · SCM · LTE 계획 고정 | 사용자 판정 |
| 2026-10-26 19:19 | SC implementer 결과 기록(형상만으로는 부족), SC outputs · 그림 생성 | implementer 보고 |
| 2026-10-26 18:58 | C0F 판정 기록(채택 제안 아님), outputs 2개 생성. 회귀 장면 보조 지표는 쥐는 팔 기준으로 다시 잼(처음엔 왼팔만) | — |
| 2026-10-26 17:40 | C0F 계획 고정(기준 ⓐ–ⓓ, 보조 ①–⑤). DX3 ≤ 16 mm 미충족 기록 유지 | 사용자 판정 |
| 2026-10-26 17:12 | 시각 표기 정정: DX · DX2 · DX3 · VW 줄의 KST를 파일 수정 시각(`date -u` + 9 h)으로 고침 — 처음 적은 값이 30–90분 앞서 있었다 | — |
| 2026-10-26 16:50 | DX3 1–3단계 결과 기록, 최소 변경(`--w-continuity 0`) closed-loop 시작. 변형 TRACK0은 config가 0을 막아 w_track 1e-4로(진단용 대조군) | — |
| 2026-10-26 16:35 | DX3 · SC · VW 계획 고정, E4 · DF 보류 | 사용자 판정 |
| 2026-10-26 16:30 | DX2 완료 · 기록, outputs 생성 | — |
| 2026-10-26 16:05 | DX2 계획 고정(접촉 분해 · 케이스 1 단계 2 비교 · 가려짐 대 화면 밖), DX 판정표를 잠정 가설로 표시 | 사용자 판정 |
| 2026-10-26 15:57 | DX 완료 · 판정 기록, outputs 생성 | — |
| 2026-10-26 15:45 | DX(숫자 진단) 계획 고정: 대상 E3 · E0 실패 12 + 12 run, 지표 M1–M5, 원인 판정 규칙과 문턱값 | 사용자 판정 "숫자로 원인 진단 먼저" |
| 2026-10-26 15:25 | E3 closed-loop 판정 기록(가림 6/18 · 접촉 run 1, 회귀 14/18 · 접촉 1), outputs 2개 생성(사용자 지시 "수행한 것 outputs에 시각화 · 실제 씬"), 규칙 보강(`.claude/ag3s-rules*.md` · verifier · 메모리: 실험 단계마다 보고 전 outputs). 회귀 집계는 E2의 `agg/address_check.json` 복사(모델 XML md5 동일) | 사용자 지시 |
| 2026-10-26 14:13 | **replay link 충돌 두 번 (lead 실수, 결과 손실 없음):** ① 13:53 검사 줄이기 스크립트가 끝나며 `E1/occ_index` link를 지움 → lock 대기 중이던 E3-a 케이스 3 9 key가 FileNotFound(`E3/a/offline_occ/failed_attempt1/`). ② 그 재실행이 link를 run 안에 둔 채로 closed-loop가 시작 → 서버 8296이 "Output must be inside run"으로 거절(`E3/closed_failed_attempt1/`, episode 0개 실행). 조치: E3-a 재실행을 멈추고(2 key 완료분은 유지) 남은 7 key는 **scratchpad link**(`occ_index_e3a.json`)로 다시 줄 세움 — storage run에는 바깥 link 0. closed-loop 14:13 KST 재시작. **규칙: closed-loop가 이 run을 쓰는 동안 `make_links.sh`를 쓰지 않는다; offline replay link는 scratchpad에 만든다** | — |
| 2026-10-26 13:25 | E3-b 가림 offline 결과(독점 lock, `E3/b/offline_occ/summary.json`): 68 / 68 해소(E2와 같음), 발동 chunk `solve_ms` 중앙값 544.5 ms(기준 533 ms 미통과, 표준 124 + 초기값 단계 427, 그중 우리 검사 16번 260 ms). **사용자 판정 (13:10 KST):** ① 검사 줄이기를 기본 꺼짐 flag로 넣고 같은 입력으로 다시 잰다 — 판정 기준(≤ 533 ms)은 그대로 ② E3-a 측정은 끝까지. 구현: `sqp.gpu_seeds_check` / `--to-gpu-seeds-check {all,first-pass}`(기본 all), offline setting `E3b_16_fp`, test 4개(GPU 2). E3-a 측정이 작업 트리 코드를 import 중이라 **`$RUN/snapshot_e3c`에서만** 구현 · test · 측정(`E3/b_fp/`), E3-a가 끝나면 checkout에 옮긴다. 상태 표기(GPU 결과 `optimal`)는 코드가 `feasible`과 같게 다루므로(`types.py:47`, `client.py:383`) 그대로 둔다 | 사용자 판정 |
| 2026-10-26 12:48 | v2 세션: E3 offline 비교 시작(가림 · 회귀 × 두 갈래, `E3/e3_offline_launch.sh`). 끝났다는 표시 `E3/{a,b}/offline_{occ,e4}/DONE`, 전체 `E3/offline_DONE_RAW` → 판정 뒤 `E3/offline/DONE` | 시작점 절 2번 |
| 2026-10-26 12:45 | E3-a · E3-b 구현 완료 기록, 새 세션 v2에 인계 메시지 | — |
| 2026-10-26 11:54 | 맨 앞에 "새 세션 시작점" 절 추가 | 사용자 지시 "실험 · 결과는 새 세션에서" |
| 2026-10-26 11:35 | outputs 보기 2개 작성(사용자 지시), E3 구현 착수, osqp-cu12 공유 venv 문제 처리 | 사용자 지시 · implementer 보고 |
| 2026-10-26 11:28 | E3 명세 고정 | 사용자 판정 "계획대로 E3" |
| 2026-10-26 11:25 | E2 closed-loop 판정: 기준 미통과(가림 장면 6 → 6) | — |
| 2026-10-26 10:25 | E2 사용자 판정(제자리 마지막 수단 · 16개 · E1-C 없이) 반영, closed-loop 시작 | 사용자 판정 |
| 2026-10-26 03:40 | E2 offline 완료 · 규칙 재계산(`E2/rule_recompute.json`) | — |
| 2026-10-26 03:00 | E2 구현 완료, offline 시작, 고르는 규칙 결함 기록 | implementer 보고 |
| 2026-10-26 02:25 | E2 명세 고정 · 구현 착수 | 사용자 판정 "계획대로 E2" |
| 2026-10-26 02:15 | E1 판정: 기준 미통과(회귀 장면 금지 접촉 1 → 2). S-REG 집계는 E4의 address_check 표를 복사해 사용(모델 XML md5 동일) | — |
| 2026-10-26 01:40 | E1-C closed-loop S-OCC 결과 기록, S-REG 재실행(kit 검사 수정) | — |
| 2026-10-26 01:15 | E1 offline S-OCC 완료, E1-C 채택, closed-loop 시작(00:51 KST) | §6 E1 규칙 (offline에서 나은 쪽) |
| 2026-10-26 00:36 | D1 판정 반영 · `--blas-threads` 구현, E0 기준선(round 4) 기록, E1 S-OCC offline 시작 | 사용자 판정 · session 13 결과 |
| 2026-10-25 22:55 | 세션 재시작 후 재개: E1 구현 완료 확인, S0 고정, P0-d · E1 offline 실행 | 사용자 지시 "이어서 진행" |
| 2026-10-25 21:40 | E0 목표 도달 가능성은 기록된 거리장 대신 **MuJoCo 정답 기하**(client `_minimum_distance`)로 쟀다 | 목표 자체가 물리적으로 가능한지가 질문이라 정답 기하가 맞다. 거리장 기준 판정은 E1 offline이 한다 |
