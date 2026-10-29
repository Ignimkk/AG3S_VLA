## AG3S 검토 — 매 턴 자가 점검 (하네스가 주입한 것)

**적용 범위**: 아래 규칙은 AG3S 연구 구현·실행·검증 작업에 적용한다.
`paper/`의 논문 준비·작성 작업은 `.claude/rules/cvpr-workspace.md`와
`paper/SETUP_PLAN.md`를 따른다. 논문 작업만을 위해 연구 실행 로그를 수정하지 않는다.

**기록** (2026-10-12 사용자 지시로 개편 — 읽을 때는 최신 → 오래된 순):
- `AG3S/ag3s/docs/AG3S_T47_LOG.md` — **최신 정본.** STEP 번호 T47 이상은 여기에 **이어 쓴다.**
- `AG3S/ag3s/docs/AG3S_T0T6_LOG.md` — 이전 정본 (T0 ~ T46). T46 (V-T46 · CL3) 마무리만 쓰고, T47 이후는 쓰지 않는다.
T46 (V-T46 · CL3) 를 제외한 2026-10-12 이후의 모든 새 STEP — T46 의 후속 접미사 STEP (`T46b`, `T46c` …) 포함 — 은 `AG3S_T47_LOG.md` 에 쓴다. 새 STEP **번호** 는 쓰기 전에 다른 세션에 알려 예약한다 (번호 충돌 방지).
그 밖의 새 기록 파일은 만들지 않는다. 옛 로그 근거는 링크하고 옮겨 적지 않는다.

- **규칙 A · 시각화** — 구현/검증 단계마다 **실제 씬 + 그래프 + 표**를 되는 데까지 **다** 만든다.
  그림은 `AG3S/ag3s/docs/figures/` 에 저장하고 로그에서 링크한다.
  **숫자를 문장에 흩어 놓는 것은 시각화가 아니다.** 만든 그림은 SendUserFile 로 보낸다.
- **규칙 C · 용어** — 새 용어를 쓰면 **그 답변 안에서 풀어 쓰고** 해당 로그 (T47 이후는 `AG3S_T47_LOG.md`) 의 "용어" 절에 추가한다.
  `violated`/`feasible`, 거친 계층/미세 계층/2계층, 여유거리, ESDF, 청크 … 를 설명 없이 쓰지 않는다.
- **규칙 G · 발견 번호를 그냥 쓰지 않는다** — `E3`, `F11`, `G2`, `C5`, `X1` 같은 발견 ID 를
  말할 때는 **그 자리에서 무슨 내용인지 한 줄로 풀어 쓴다.** 번호만 던지면 읽는 쪽이 로그를
  뒤져야 한다. 예: "E3(쥔 물체가 optimizer 에 도달하지 않는다)". 스텝 번호도 같다.
- **규칙 D** — 모든 Step을 수행하거나 종료되면 위 규칙에 맞춰서 정리한다.
- **규칙 E** — `CONTAINER_SETUP.md` 는 폐기됐고 2026-09-21 에 삭제됐다. **찾지 않는다.**
- **규칙 F** — `AG3S_REVIEW_PLAN.md`에 수행한 내용에 대해서는 계획을 업데이트한다.
- **규칙 I** — 기록을 작성할 때나 나에게 설명할 때 자연스럽게 문장을 구사하기 위해서 기술 용어는 영어로 작성한다.

**병렬 세션 소유권** — 같은 파일을 둘이 쓰면 한쪽이 덮는다. 내 역할의 것만 쓴다.
| A0 lead | A1 구현 | A2 검증 | A3 기록 |
|---|---|---|---|
| `AG3S/ag3s/docs/handoff/*.task.md` · PLAN | `AG3S/**/*.py` · `Baseline/**/*.py` · `benchmark/**/*.py` · `integration/**/*.py` · `common/**/*.py` · `tests/**` | `AG3S/ag3s/docs/figures/**` · `*.verify.json` · `${AG3S_RUN_DIR}/verify/**` | `AG3S/ag3s/docs/*.md` · `references/legacy-papers/**` |

**로그 (`AG3S_T47_LOG.md` · `AG3S_T0T6_LOG.md`) 의 writer 는 A3 하나다.** 나머지는 `AG3S/ag3s/docs/handoff/` 로 넘긴다.
**A3 는 `verify.json` 의 `numbers` 에 없는 숫자를 쓰지 않는다** — 없으면 A2 에게 측정을 요청한다.

**sub-agent 사용 규칙** (2026-10-14 사용자 지시, 2026-10-19 보완, 2026-10-22 rules 에 명시)
- sub-agent 는 **서로 독립인 작업 둘 이상을 동시에 돌려 더 빨리 끝날 때만** 쓴다.
- 작업이 하나뿐이거나, 그 결과가 다음 판정 · 다음 단계를 정하는 작업(순차 작업)은 **lead 가 직접** 한다.
  긴 실행은 background shell 로 돌리고 끝나면 확인한다. closed-loop kit 구성 · 실행 · 집계, 원인 진단도 여기에 든다.
- 띄우기 전에 묻는다: "지금 동시에 진행해야 할 다른 독립 작업이 있는가?" 없으면 띄우지 않는다.
- 이유: 끊긴 sub-agent 를 다시 띄우는 토큰 낭비와 상태 혼동 (2026-10-14), 순차 작업을 sub-agent 에 맡긴 위반 (2026-10-22).
- 띄울 때도 위 소유권 · 동시 agent ≤ 3 · 실행 · 대기 · 단순 집계는 sonnet 규칙을 지킨다.

**실험 원칙 — 사용자 지적 모음, 역할별** (2026-10-23 memory 에서 옮김. 괄호 안은 지적한 날짜)
- **lead (설계 · 판정 · closed-loop 실행)**
  - 모든 STEP 을 `doc/active/plans/develope_PLAN.md` §0.10 방향에 연결한다. 방향은 경계 보존 방법 하나 + identity · extent · contact 분리 ablation + closed-loop 이다. task 문서에 어느 항목을 채우는지와 중단 기준을 적는다 (10-12)
  - 가설 · 주 지표 · 진행 / 중단 기준 · 장면 선정 규칙 · 비용 예산은 **결과 전에** task 문서에 고정한다. 결과를 보고 장면을 고르거나 기준을 바꾸지 않는다. FAIL 을 기준을 올려 PASS 로 바꾸지 않는다 (10-13 · 10-14)
  - closed-loop 은 **약 10 episode** 로 작게, 빠르게 반복한다. 24 · 40 장면은 사용자 요청이 있을 때만 (10-20)
  - 모듈 시험 장면 풀은 **VLA 단독 성공** 장면에서 결과 전에 고정한다. VLA 단독 성공 장면이 AG3S 로 실패하면, 그 원인과 발동한 규칙을 특정한다 (10-20)
  - 실패마다 원인 위치를 AG3S 구성 요소로 판별한다: ATS(target 분할 · 추적) · HSR(작업 단계 · 쥠 판단) · LTE(링크별 target 제외 · 거리장) · CCR(TrajOpt 궤적 수정 · HOLD). 증상 위치와 원인 위치가 다르면 둘 다 적는다 (10-20)
  - 수정은 원인 층에서 한다. 궤적 원인은 TrajOpt, 관측 · 기하 원인은 AG3S 핵심이다 (10-20)
  - 서빙 · closed-loop · 평가 판단은 다른 세션에 미루지 않는다. 직접 검토하고 위험 · 대안을 붙여 사용자에게 묻는다 (10-14)
  - "실험 계획" 질문에는 benchmark · baseline · ablation · 실물 수준으로 답한다. 개별 run 현황은 보조로 쓴다 (10-18)
  - **영상 · 검증 결과는 checkout `outputs/<YYYY-MM-DD>_<실험 설명>/` 에 만든다** (2026-10-23 사용자 지시). 원시 기록(서버 기록 · replay, 수 GB)은 storage run 에 둔다
    - **시점 (2026-10-26 사용자 재지시 — offline 비교 · closed-loop 결과를 outputs 없이 보고한 일이 있었다):** offline · closed-loop · 진단 등 실험 단계 하나가 끝날 때마다 **사용자에게 결과를 보고하기 전에** 만든다. lead 가 직접 돌린 실험도 같다. 여러 단계가 이어지는 실험(가림 → 회귀 등)은 끝난 단계부터 넣고 갱신한다
    - **내용:** 영상(closed-loop) + 실제 씬 그림(궤적 · 장면, 예: session 13 `p3/fig_traj.py`) + 그래프(분포 · 추이) + 표(구성별 결과) + README(목적 · 구성 · 결과 표 · 판정 · 원시 기록 경로). 그림 원본은 `AG3S/ag3s/docs/figures/` (+ sidecar) 에 두고 outputs `그림/` 에 복사한다
    - **보고에는 outputs 폴더 경로를 적는다. outputs 가 없으면 결과를 보고하지 않는다** (먼저 만든다)
    - closed-loop 이 끝나면 `python -m common.experiment_view --name … --about … --config "구성설명=<kit>:<구성>" …` 로 만든다
    - 장면별 폴더에 `<순번>_<구성 설명>_<결과>.mp4`(원본 hard link)와 README(무슨 실험 · 구성 · 결과 표 · 원시 기록 경로)가 생긴다
    - 그 실험의 verify.json · 그림 · 요약 표도 같은 폴더에 둔다. 사용자에게 알릴 때는 이 폴더 경로를 준다
  - kit 을 복사해 이름 · 경로를 바꾼 뒤에는 `grep` 으로 옛 이름 · 경로가 남지 않았는지 확인하고 띄운다 (10-23: `lists/` 이름이 안 바뀌어 smoke 에서 멈춤)
- **구현 (implementer)**
  - 새 기능은 기본 꺼짐 flag 로 넣는다. 꺼짐이면 출력 · 기록 키까지 그대로이고, test 로 보인다
  - worktree 에서 test 할 때 `env.sh` 가 main checkout 으로 `cd` 한다. 바꾼 module 의 `__file__` 이 worktree 를 가리키는지 확인한다 (10-14)
- **검증 (verifier)**
  - 주 지표는 실제 MuJoCo contact · 작업 성공이다. TrajOpt 근사 구의 판정은 보조 지표다 (10-13)
  - HOLD 증가로 줄어든 접촉은 보호 효과가 아니다. 평가 정의(허용 · 비허용 접촉)는 조건과 무관하게 같다. oracle 은 상한이 아니다 (10-13)
  - 허용오차가 있으면 "0" 대신 "허용오차 이내 + 원시 max + 허용오차를 정한 시점"으로 쓴다 (10-14)
  - 궤적 분석 · 실패 판정은 명령 궤적과 함께 **실제 궤적**(`measured_qpos`, replay)으로 한다. MuJoCo kp · damping 때문에 명령 ≠ 실제다 (10-20)
  - 판정 규칙 · 분류 정의는 계산 전에 `verify.json` `definitions` 에 적는다. 결과를 본 뒤 바꾸면 이유를 적고 이전 수치를 함께 남긴다
  - 출력 경로에 쓰는 변수가 비어 있지 않은지 확인한다 (10-21: 빈 변수로 `/w2.out` 생성)
  - CPU worker ≤ 4 는 pod 전체 기준이다. 다른 세션이 이미 더 띄운 것을 근거로 늘리지 않는다. 세는 단위는 process 가 아니라 실제 계산 worker 다 (10-14)
- **기록 · 보고 (scribe, 그리고 lead 의 사용자 보고)**
  - 보고 순서: 목적 → 변경(flag, 기본 꺼짐) → 실험 → 이전 대비(before → after). 개선이 아니면 그렇다고 쓴다 (10-12)
  - 구성 코드 · 자체 약어(SPL · F1 · W1 · P1 · arm · "A agent" 등)를 쓰지 않는다. "무슨 실험이고 무엇을 썼는지"로 쓴다. 줄인 이름이 꼭 필요하면 미리 알리고, 쓸 때마다 정의를 붙인다. ATS · HSR · LTE · CCR 은 메시지에서 처음 쓸 때 풀어 쓴다 (10-18 · 10-20 · 10-22)
  - 실패 표에는 "구성 요소" 열을 둔다 (10-20)
  - KST 시각은 `date -u` 에 9 시간을 더한다. 이 pod 에는 timezone 이 없어 `TZ=Asia/Seoul date` 가 UTC 를 그대로 보여 준다 (10-21)

**진행 방식**: `설명 → 사용자 질문(여기서 멈춤) → 공동 판정 → 확정된 것만 수정+검증 → 기록`.
**추측을 기록에 남기지 않는다** — 그 자리에서 실행해 수치로 확정한다.
체크포인트/MuJoCo 실행이 필요하면 먼저 사용자에게 요청한다.

**공용 pod 자원 — 끝난 작업의 메모리는 다른 세션이 쓸 수 있게 회수한다** (2026-10-15 사용자 지시)
- 한도: working set (`memory.current − inactive_file`, `/sys/fs/cgroup`) + 추정 ≤ 64 GiB (pod hard limit 80 GiB),
  worker ≤ 4, 서버 ≤ 2, GPU ≤ 130 GB. 시작 전에 확인한다.
- **page cache** 는 읽거나 쓴 파일 내용을 커널이 RAM 에 남겨 둔 것이다. process 가 끝나도 남고, 최근에 다시 읽힌
  `active_file` 은 working set 에 그대로 잡힌다 — 한 세션의 기록 · checkpoint 가 다른 세션의 실행을 막는다
  (2026-10-15 측정: process 메모리 6.5 GiB 인데 working set 56.6 GiB, 대부분 끝난 CL4 · T56 · T53 기록과 checkpoint 의 cache).
- **기록 · 테스트 · 추출 · rollout · replay 가 끝나면 (handoff 를 넘기기 전에) 반드시 회수한다:**
  1. 그 작업이 띄운 자기 process (server · worker · GPU process) 가 모두 끝났는지 확인하고, 남은 것은 종료한다.
  2. `python3 scripts/setup/reclaim_cache.py "$AG3S_RUN_DIR" <읽은 큰 입력 (기록 · attention / feature npz · dataset · checkpoint) …>`
     — 각 파일을 `fdatasync` 후 `posix_fadvise(DONTNEED)` 로 내린다. 파일은 디스크에 그대로 있고 다시 읽으면 NFS 에서 올라온다.
     어떤 process 든 **지금 열고 있는 파일은 건너뛴다** (`/proc/*/fd` 검사; 이 pod 에는 `fuser` 가 없다) — 공용 checkpoint 를
     서버가 들고 있거나 다른 세션이 run 을 쓰는 중이면 그 파일은 남는다. 다른 세션의 run 디렉터리는 인자로 넘기지 않는다.
  3. 출력 한 줄 (`reclaim: ws A->B GiB | cache C->D GiB, …`; working set 과 대상 `fincore` 합의 전후) 을 handoff
     (`impl.md` · `verify.json` 의 run 메모, 또는 사용자 보고) 에 그대로 남긴다. 확인만 할 때는 `--dry-run`.
- **쓰지 않는다 (사용자 승인 없이):** `/proc/sys/vm/drop_caches` (노드 전체), `/sys/fs/cgroup/memory.reclaim` 쓰기
  (pod 전체 — 다른 세션의 cache 까지 버린다). 다른 세션의 process 는 종료하지 않는다. VS Code 창이 닫혔는데 남은
  Pylance (`ppid 1`, `--clientProcessId` 의 process 없음) 처럼 주인이 없는 process 를 보면 사용자에게 보고한다.

**closed-loop 기록은 place 뒤 2–3 초에서 끝낸다** (2026-10-21 사용자 지시)
- place 를 작업 완료로 본다. place 이후 홈 복귀 구간(약 23 초 이후) 데이터는 저장하지 않는다.
- client `integration/rby1/bringup/pi05_infer.py --stop-after-place 2.5`
  - place 판정: target 과일 중심이 바구니 안(`rby1_manipulation.evaluation.transport.object_in_crate`, 분석 `analyze_t42.py` 와 같은 기준)이고,
    쓰는 팔의 그리퍼가 열린(≥ 0.5) 첫 control step
  - 그 2.5 초(38 step) 뒤에 episode 를 끝낸다. log `[place] stop after place`, `traj.npz` 에 `place_detect_t` · `place_stop_t` 를 남긴다
  - place 에 도달하지 못한 run 은 `--max-steps` 까지 그대로 기록한다(실패 분석용)
- 새 kit 을 만들 때 세 군데를 고친다.
  - `client_args.py` 가 위 flag 를 붙인다
  - `check_run.py` 는 `[place] stop after place` 로그가 있고 기록이 일관되면(`n_planning == ceil(n_control / 8)`, `executed_actions (n_control, 16)`)
    600 step 미만도 완료로 본다(`obstacle stop` 과 같은 방식)
  - `run_chain.sh` 의 client 인자 검사에 이 flag 를 넣는다
- 확인(2026-10-21, 기존 34 run 에 같은 규칙 적용): 27 run 이 place 에 도달했고, 평균 18.2 초에 끝난다(control step 55 % 감소). 최종 성공 판정이 바뀐 run 은 0
