# T52 – T57 세션 → 세션 B 인계

> writer: T52-T57 세션 lead · 2026-10-14 KST · 읽는 쪽: 새 세션 B (identity · phase source 분석), 참고: 세션 A (`ag3s-vla-development-d2`)
> 역할 정본: [`SESSIONS_A_B.md`](SESSIONS_A_B.md). 진행 이력: [`T52-T57.plan.task.md`](T52-T57.plan.task.md) (갱신 1 – 5).
> 기록 정본: [`../AG3S_T47_LOG.md`](../AG3S_T47_LOG.md). 공유 로그의 commit 은 A 가 한다.
> **A 도 새 세션으로 바뀐다** (2026-10-14, 이전 A = d2 의 인계: [`HANDOFF_A_main.md`](HANDOFF_A_main.md), `8127c0f2`).
> - 공유 로그의 미커밋분 (이 세션의 T54c · T55 · T56 절) 은 새 A 가 최종 diff OK 를 받은 뒤 commit 한다 (HANDOFF_A_main §2).
> - 이 세션의 "인계 완료" 보고는 새 A 의 이름이 알려지면 그쪽으로 보낸다. 끝까지 알려지지 않으면 이 문서 맨 아래 "인계 완료 기록" 이 그 보고를 대신한다.

## (a) STEP 별 상태 (T47 – T57)

| STEP | 내용 | 상태 | 근거 |
|---|---|---|---|
| T47 · T47b | label-free head 재선택 · 합산 · B3 | 완료 · 로그 기록 | 로그 "T47 · T47b" 절 |
| T48a · T48b | HSR counterfactual (prompt · state · image, 물체 유무) | 완료 · 로그 기록 | 로그 해당 절 |
| T49 · T49b | 명사 token query | 완료 · 로그 기록 | 로그 해당 절 |
| T50 | HSR instruction 둔감의 원인 분리 | 완료 · 로그 기록 | 로그 "T50" 절 |
| T51 · T51b | head 선택 protocol (P1 – P6) · 명사 행 | 완료 · 로그 기록 | 로그 "T51 · T51b" 절 |
| T52 | 장면 1800 실패 원인 분해 (+ follow-up) | 완료 · commit `8587083e` · 로그 `95b67f8c` | `T52.task.md` · `T52.verify.json` · `figures/t52/` |
| T53 | 새 test 16 episode rollout + 추출 | 완료 · commit `8587083e` · 로그 `95b67f8c` | `T53.*` · `figures/t53/` |
| T54 | L8H2 / N / A / B grounding 비교 | 완료 (A 기각) · commit · 로그 | `T54.*` · `figures/t54/` |
| T54b | `hold_attach` 재생 (기록 latch · attach 주입) | 완료 · commit · 로그 | `T54b.*` · `figures/t54b/` |
| T54c | N · B 새 16 episode 확인 (rollout + 추출 + 재생) | **완료 · 로그 기록 (미커밋 → 이번 인계 commit)** · N = T57 후보 | `T54c.task.md` · `T54c.impl.md` · `T54c.verify.json` · `figures/t54c/` |
| T55 | phase feature 대 strong rule (물리 predicate) | **완료 (test) · 로그 기록 (미커밋)** · 명사 probe = T57 단계 2 후보, HSR = 보조 | `T55.task.md` · `T55.impl.md` · `T55.verify.json` (dev + `test`) · `figures/t55/` |
| T55 P · H 보조 창 | T60 종료 구간 mask 로 T55 를 다시 채점 (add-on) | **구현 · 1 차 실행 완료** (전체 창 parity 정확 일치). 독립 재실행 (새 run, 약 59 s) · 해석 · 로그 기록은 **B** 가 한다. `t56_windows.py` 는 준비만 됨 (T56 `t56_summary.json` 이 생긴 뒤 실행) | `T55_windows.impl.md`, 출력 `results/development/20261014T193322+0900_284366d2bbec_e3469147913d/verify/T55_windows/` |
| T56 | self label 이 실제 품질 순위를 예측하는가 | dev 완료 · prereg 사용자 확정. **test 재실행 중 (2026-10-14 12:49:59 UTC 시작, CPU worker 4, B 로 인계)** — 끝나면 B 가 `T56.verify.json` 의 test 블록 · figure · 로그 T56 절 · `t56_windows` 를 마무리하고 판정을 묻는다 | `T56.task.md` · `T56.impl.md` · `T56.verify.json` (dev) · 진행 `…/60a2394b8ccc/verify/T56/test_status.txt` |
| T57 | closed-loop: 단계 1 N grounding, 단계 2 명사 probe task gate | task 작성 · **서빙 구현 A 위임 (A 사용자가 정리 뒤 착수, 대기)** | `T57.task.md` |

## (b) 받아 둔 사용자 판정 (날짜 · 위치)

| 날짜 | 판정 | 위치 |
|---|---|---|
| 2026-10-13 | 계열 전체: L8H2 유지 · HSR 보조, 장면 1800 원인 분석 먼저, grounding 비교 (중심 거리만으로 판정 안 함), 의도 · 보유 분리, self label 품질 예측, 미사용 data 검증 · block bootstrap, offline 통과 후보만 closed-loop | `T52-T57.plan.task.md` 맨 위 |
| 2026-10-13 | head 0 기록 `ep1800` · `long` 은 dev 에서 제외하고 과거 결과에 표시만 (재측정 안 함) | plan 갱신 1, 로그 재측정 대기 #32 |
| 2026-10-13 | T54 사전 고정 (A 주 · B 원인 확인, 현재 frame 주 · 유지 보조, 진입 기준 −3 %p · −5 %p, 명사 = L10H7 "GT 로 dev 에서 선택한 고정 head") | `T54.task.md` |
| 2026-10-13 | T54 결과 후: N · B 를 새 data 에서 확인 (T54c), 유지 재생에 attach 주입 (T54b) | plan 갱신 2, `T54b.task.md` · `T54c.task.md` |
| 2026-10-14 | T54c: 보수적 episode 목록 (1844 · 1903), rollout 승인 | `T54c.task.md` 끝 절 |
| 2026-10-14 | T54c 규칙 출력 → N 을 T57 후보 | 로그 "T54c" §5 |
| 2026-10-13 ~ 14 | **T55 사전 고정** (P_grasp · P_task, 재시뮬 + 자연 + instruction, HSR 고정 + 명사 probe, `task_ready = HELD ∧ feature`, dev 기준 = 비개입 재시뮬, perception = 재실행 grounding, 허용폭 +5 %p · +1 frame · 0 건). prereg: `…/20261013T171325+0900_cf38c9dbca7b_2ebfd8a5a43c/verify/T55/T55.prereg.json` (sha256 `8f62fc76…`) | `T55.task.md` 사용자 판정 절들 |
| 2026-10-14 | T55 prereg 확인: 합산 허용폭 유지 ("자연 조건 초과" 함께 보고), slip ghost · park 둘 다 유지 | `T55.task.md` 끝 절 |
| 2026-10-14 | T55 결과 후: 명사 probe gate 를 T57 **별도 단계 (단계 2)** 로, 서빙 구현은 A 위임 | `T57.task.md` 단계 2 절 |
| 2026-10-13 | **T56 사전 고정** (cell · arm 순위, S0 / S1 / S2, HELD event, regret 주 · τ-b 진단, "GT 로 개발하고 추론 시 GT 를 쓰지 않는 방법" 표기) | `T56.task.md` |
| 2026-10-14 | **T56 네 값 확정:** 동률 해상도 0.01 · regret 허용폭 0.03 · purity / coverage 허용 악화폭 0.05 (반올림 전 값으로 판정). prereg: `…/20261013T223613+0900_cf38c9dbca7b_60a2394b8ccc/verify/T56/T56.prereg.json` (sha256 `bbe98958…`) | `T56.task.md` 끝 절 |
| 2026-10-14 | **T57 단계 1:** T54c 16 episode 재사용 + N arm 만 추가, 채택 = P 까지 `success_ct` 비열등 (−1 run) + 비허용 접촉 run · HOLD chunk 비악화, H 필수 보고. 서빙 구현은 A | `T57.task.md` |
| 2026-10-14 | **T57 단계 2:** 명사 probe task gate, attach / detach 불변. **task_ready 연결 위치는 구현 전 판정 (대기)** | `T57.task.md` 끝 절 |
| 2026-10-14 | T60 P · H 창: 진행 중 실험 (T54c · T55 · T56) 은 보조 분석, 새 실험 (T57 부터) 은 P 주 지표 + H 필수 | plan 갱신 4, 로그 결정 행 |
| 2026-10-14 | SL5b reach 인계 (benchmark) 는 A 담당 | 로그 결정 행 |
| 2026-10-14 | sub agent 는 병렬 작업이 필요할 때만 | 메모리 `subagent-only-for-parallel` |

## (c) 판정 대기 · 아직 묻지 않은 질문

1. **T56 test 결과의 판정.** 사전 고정 기준 (regret 허용폭 안 · S0 대비 regret 감소 · τ-b 개선 · purity / coverage 악화 허용폭 안) 의 기계적 결과를 보고하고, 채택 · 유지 · 기각을 묻는다. 이 세션은 결과만 남긴다.
2. **T55 P · H 보조 창 결과의 해석** (add-on 결과가 나온 뒤). 주 판정은 바뀌지 않는다.
3. **T57 단계 2 의 task_ready 연결 위치** (예: 놓기 단계 인증 · 목적지 전환). A 가 구현하기 전에 사용자 판정이 필요하다. B 가 근거 (T55 결과) 를 정리해 묻는다.
4. **T54c 의 P · H 보조 창** 을 계산할지 (판정상 보조 분석 대상이지만 아직 계산하지 않았다).
5. **T55 재측정 대기 #48** (instruction 변경 시 strong rule 이 실제 보유를 HELD 로 인정하지 않음) 의 후속 여부. latch 쪽 원인이면 A 범위다.
6. **B 가 T56 · T61 을 하나의 head 선택 protocol 로 정리하는 방식** (SESSIONS_A_B 의 B 역할). 아직 사용자와 범위를 정하지 않았다.

## (d) 미커밋 · 실행 중 · port · run dir

**이번 인계 commit 에 넣는 자기 파일** (pathspec):
- handoff: `T54c.{task,impl}.md` · `T54c.verify.json`, `T55.{task,impl}.md` · `T55.verify.json`, `T55_windows.impl.md`, `T56.{task,impl}.md` · `T56.verify.json`, `T57.task.md`, `T52-T57.plan.task.md`, 이 문서
- figures: `figures/t54c/` · `figures/t55/` (· T56 figure 가 생기면 `figures/t56/`)
- code: `t54c_extract.py` · `t54c_rollout.py` · `t54c_records.py` (clone 수정) · `t55_*.py` · `t55_windows.py` · `t56_*.py` · `t56_windows.py`
- test: `tests/ag3s/test_t54c_records.py` (회귀 test 추가) · `test_t54c_rollout.py` · `test_t55_*.py` · `test_t55_windows.py` · `test_t56_*.py` · `test_t56_windows.py`

**넣지 않는 것:** `AG3S_T47_LOG.md` (T54c · T55 · T56 절 미커밋 → A 가 commit), 다른 세션 파일 (`t58` · `t59` · `t61_*` · `PPT_TO_V11AB` · `ppt-1013` · `paper/*` · `.vscode/settings.json`).

**실행 중 · 예정:**
- T56 test: `…/20261013T223613+0900_cf38c9dbca7b_60a2394b8ccc/verify/T56/run_test.sh`. driver 는 inputs → replay (worker 4) → held → score 순서다.
  - 첫 실행은 A14c 적용 때문에 08:59 UTC 에 중단했다 → `verify/T56/test_abandoned_run1/`.
  - 재실행은 HEAD `8df34591` 이후의 코드로 한다 (docs 만 다른 HEAD). 진행 상태는 `verify/T56/test_status.txt` 에 있다.
- T55 P · H add-on (`t55_windows.py`): 이 세션의 implementer 가 마무리 중이다. 출력은 새 run dir 의 `verify/T55_windows/` 이고, 경로는 `T55_windows.impl.md` 에 있다.
- worktree: 이 세션이 만든 것은 없다 (`.claude/worktrees/` 는 A 의 것).
- port: B 몫 8292 · 8293. 지금 e4 의 T61 rollout 이 쓰고 있다.

**run dir (results/development/…):**

| STEP | run |
|---|---|
| T52 | `20261013T094624+0900_693dcedeaadc_a2c23cf2a275` |
| T53 (rollout · 추출, T54 – T56 의 test data) | `20261013T100210+0900_4a540f1a18d7_cb037c425d2f` |
| T54 | `20261013T135232+0900_cf38c9dbca7b_611265305126` |
| T54b | `20261013T225350+0900_cf38c9dbca7b_f739ac556e8a` |
| T54c (rollout = T57 의 L8H2 기준 arm, 추출, 재생) | `20261014T133115+0900_cf38c9dbca7b_5697bbdb5190` |
| T55 (dev + test) | `20261013T171325+0900_cf38c9dbca7b_2ebfd8a5a43c` |
| T56 (dev + test) | `20261013T223613+0900_cf38c9dbca7b_60a2394b8ccc` (이전 run `…161631…_392250b77964` 는 abandoned) |

## (e) B 가 다음에 할 일 (추천 순서와 근거)

1. **T56 test 결과 판정 요청.** 사전 고정이 모두 끝났으므로 기계적 결과만 정리해 묻는다 (`T56.verify.json` 의 `criteria_mechanical`). 이것이 B 의 head 선택 protocol 정리 (T51 · T56 · T61) 의 입력이 된다.
2. **T55 · T56 의 P · H 보조 창** — `t55_windows` 를 새 run 에서 다시 돌려 (1 차 출력 경로는 (a) 표) 결과를 확인하고, T56 test 가 끝났으면 `t56_windows` 를 돌린 뒤 로그에 "기존 정의 · P · H 나란히" 로 기록한다 (사용자 판정 2026-10-14).
3. **T61 (e4 인계) 과 T56 을 묶어 identity source 선택 protocol 을 정리한다.**
   - 근거: T51 (GT 선택) · T56 (self label 선택) · T61 (사전 고정 head 선택 + 새 held-out) 이 같은 질문에 답한다.
   - T54c 에서 N (명사 L10H7) 이 현재 frame 에서는 크게 나았지만 (+19.9 %p), 서빙 유지 조건에서는 +0.8 %p 였다. 그래서 선택 기준에 시간 유지 조건을 넣을지 사용자와 정해야 한다.
4. **T57 지원.**
   - 단계 1 (N) 은 A 가 구현 · 실행한다. B 는 판정 근거 (T54c · T54b) 를 유지한다.
   - 단계 2 는 B 가 task_ready 연결 위치 질문 ((c) 3) 을 사용자에게 묻고, 그 답과 probe 정의 (`T55.prereg.json` · `noun_probe.npz`) 를 A 에 넘긴다.
5. **T54c P · H 보조 창** ((c) 4) — 판정을 받으면 `t60_termination.py` 정의로 계산한다.
6. **A 에 요청할 것:**
   - T57 서빙 구현 (이미 위임, 대기)
   - #48 의 latch 원인이 serving 쪽이면 그 진단
   - 새 rollout 이 필요하면 task 문서 "구현" 절로 요청

## (f) 알려진 함정

- **head 0 기록:** `ep1800` (`recordings/run_16d_ep1800`) 과 `long` (`20260924_long16d`) 은 X2 수정 전 녹화라 `head_1` = 0 이고 cam_high 가 벽을 본다. dev 에서 제외하고, 과거 결과 (T47 selection, T49b · T51 · T51b · T52 의 장면 1800) 는 표시만 한다 (재측정 대기 #32).
- **T53 은 N · B 에 대해 dev 다** (T54 결과를 보고 후보로 골랐다). T54c 16 episode 는 T57 의 closed-loop data 다.
- **유지 재생은 `hold_attach` 를 쓴다.** T54 의 `hold` 는 attach 를 재현하지 않아 grasp 이후 identity 가 0 이 된다. `hold_attach` 에서도 post · placed identity 는 서빙을 재생한 L8H2 까지 0 이다 (지표 정의 문제, 재측정 대기 #39).
- **dev 재시뮬은 기록과 같지 않다** (dev record 는 실행 동작이 float32 chunk 로만 저장됨). T53 · T54c 는 bit-exact 다.
- **`t54c_records` clone:** 원본 `t53_records` 를 먼저 import 해야 한다. 수정했고 새 process 회귀 test 가 있다.
- **추출 driver:** `t54c_extract` / `t53_extract` 의 모든 stage 에 `--episodes-root` 와 `--server-roots` 를 넘긴다 (layout 에만 넘기면 attention 에서 실패한다).
- **md5 고정과 다른 세션 commit:** 측정 중에 A 의 서빙 commit 이 들어오면 import 하는 측정이 오염된다. A 의 적용 전 알림을 받고, 측정 중이면 미루게 한다. T56 test 첫 실행은 이 때문에 버렸다.
- **공용 pod 메모리:** T56 replay (worker 4 ≈ 28 GB), T55 perception (worker 2 ≈ 14 GB), 다른 세션의 rollout (≈ 20 GB) 이 겹치면 64 GiB 를 넘는다 (실제로 70 GiB 순간 peak, OOM 은 없음). stage 마다 메모리 대기를 건다.
- **env.sh 는 메인으로 cd 한다.** worktree 에서 돌릴 때는 다시 cd 하고 import 경로를 확인한다.
- **strong rule 은 grounding 에 묶여 있다.** instruction 을 바꿔 재-grounding 하면 실제 보유를 HELD 로 인정하지 않는 경우가 생긴다 (#48). latch 의 reach 검사는 가려진 target 의 마지막 centroid 기준이라 Panda 에서 attach 를 막는다 (SL5b, A 담당).
- **"개선" 표현:** 진입 기준이나 허용폭을 통과한 것을 우월성 입증이라고 쓰지 않는다. 사용자 문구는 "offline 개선이 관측돼 closed-loop 확인 대상으로 선정했다" 다.

## 인계 완료 기록

- 2026-10-14 사용자 알림: "현재 당신의 역할을 다른 B 세션에 인계한 상황". 그래서 이 세션은 T56 test 를 끝까지 지켜보지 않고, 실행 중인 상태로 B 에 넘긴다.
- **T56 test (run2):**
  - driver: `…/20261013T223613+0900_cf38c9dbca7b_60a2394b8ccc/verify/T56/run_test.sh`. 순서는 inputs → replay (worker 4, 메모리 대기 ws + 28 ≤ 64) → held → score 다.
  - 상태는 `test_status.txt`, 로그는 `test_*.log`, 출력은 `test/` · `test/analysis/t56_summary.json` 이다.
  - 시작 HEAD 는 `8127c0f2` 이고, 코드는 `8df34591` 과 같다 (그 뒤 commit 은 docs 만).
  - **dev 대비 md5 차이는 둘이다:** `config.py` (A14c, 기본 off, bit-identical 주장 · A 검증) 와 새 `t56_windows.py` (add-on, test 경로 밖). 이 차이는 `md5_test_start_run2.txt` 에 있다.
  - 첫 실행은 `test_abandoned_run1/` 에 있다 (deviation).
- **B 가 할 일:** test 가 끝나면 `T56.verify.json` 의 test 블록 · figure · 로그 절을 마무리하고, `t56_windows` 를 실행한 뒤 (c) 1 의 판정을 묻는다. 중간에 멈추려면 `run_test.sh` 와 `t56_study` process (이 세션이 띄움) 를 끝낸다. 끝난 job 은 재실행 때 건너뛴다.
