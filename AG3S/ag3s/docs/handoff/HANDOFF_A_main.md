# 세션 A (메인 develope) 인계 — 2026-10-14 KST

> writer: lead (이전 A 세션 `ag3s-vla-development-d2`, 이전 이름 `-72`). 사용자 지시: 컨텍스트가 가득 차 새 A 세션으로 넘긴다.
> **새 A 는 먼저 [`SESSIONS_A_B.md`](SESSIONS_A_B.md) (역할 · 소유권 · 조율 규칙 정본) 를 읽는다.** 이 문서는 A 의 진행 상태 · 대기 · 함정만 담는다.

## 0. 먼저 읽을 것 (순서)
1. `CLAUDE.md` · `.claude/ag3s-rules.md` · `SESSIONS_A_B.md` (이 디렉터리).
2. `doc/active/plans/develope_PLAN.md` §0 · §6 (DX-00 · DX-01 · DX-04 · DX-09) — A 의 진행 정본.
3. `AG3S/ag3s/docs/AG3S_T47_LOG.md` 맨 위 표 + "T46e" · "T46f-A14 · V-T46f-A14" · "T46c-W1 · T46d · T46f" 절.
4. task: [`T46f.task.md`](T46f.task.md) (T46f 블록 전체 — 가설 · 접촉 정의 · 단계 1–5 · 모든 사용자 판정, **다음 작업의 정본**), [`T46e.task.md`](T46e.task.md), 위임받은 [`T57.task.md`](T57.task.md).
5. memory (`~/.claude/projects/-mnt-dev-AG3S-VLA-development/memory/`): 보고 형식 · 우수성 비전제 · worktree env.sh 함정 · sub-agent 는 병렬일 때만.

## 1. 지금 상태 (commit, push 안 함 — push 는 사용자)
| commit | 내용 |
|---|---|
| `cf38c9db` | T46e visible-target oracle extent (기본 off) 메인 적용 |
| `89d41c27` | V-T46f-A14 독립 검증 기록 · T46e verify 정정 |
| `95b67f8c` | 공유 로그 (T46e · T46f · T52–T54b · T60–T60c) |
| `53f3911d` | `develope_PLAN.md` (사용자 수정 + e4 문장 교체, 사용자 지시) |
| `8df34591` | **T46f A14 · A14b · A14c 손목 · 손바닥 형상 보정 (기본 off) 메인 적용** |
| `284366d2` · `18fbcaee` | T46f 메인 적용 기록 · 세션 2 개 체제 정본 |

- **T46e:** 완료 · 적용. oracle 은 "비대상 보존 중심의 관측 기반 비교 기준선" (상한이 아니다).
- **T46f 단계 1 (형상 보정):** 완료 · 적용. 상태 문구 (사용자 지정): **"형상 · 회귀 · 호환성 PASS. TO 상대 비용 기준 FAIL (median 2.65 > 2.0). 계산 비용 증가를 수용하여 실험용 적용 및 단계 2 진행 승인."** 최종 형상 = `--links gripper_wrist --palm-cover on` (1,349 구), **T46f 의 모든 조건 (① none · ② exclude-all · ③ exclude-authorized + strict + nonauthorized obstacle) 과 CL4 모든 arm 에 같은 형상 · TO 설정.** "동기식이라 해가 안 바뀐다" 는 쓰지 않는다.

## 2. 다음 작업 (사용자: 세션 정리 뒤 착수)
| 순서 (추천) | 작업 | 정본 | 메모 |
|---|---|---|---|
| 1 | **세션 통합 마무리** | `SESSIONS_A_B.md` §4 | T52-T57 · e4 가 인계 문서 (`HANDOFF_T52-T57_to_B.md` · `HANDOFF_e4_to_B.md`) 와 commit hash 를 A 에 보낸다 (**옛 A 이름 `ag3s-vla-development-d2` 로 보낼 수 있다 — 새 A 는 시작하자마자 두 세션에 새 이름을 알린다**). 이후 사용자가 B 세션을 연다 |
| 2 | **공유 로그 commit** | — | 지금 `AG3S_T47_LOG.md` 에 미커밋: T52-T57 의 **T54c 절 · T55 절** (재측정 대기 #47–#49), e4 의 **T60b 판정 단락 · T60 §9 ③ 갱신**, A 의 T46f §7 한 줄 (메인 적용 `8df34591`). T56 은 T52-T57 이 test 뒤 한 번 더 쓴다. **쓰는 세션들의 최종 diff OK 를 받은 뒤 A 가 이 파일만 pathspec commit.** 다음 재측정 대기 번호 #50 |
| 3 | **T46f 단계 2 — 진단 장면** | `T46f.task.md` §5 (S1–S4 사전 고정, dev / eval 분할) · §8 (중단 기준: eval 배치 < 10 이면 사용자 보고) | 후보: E0 rollout · 기존 기록에서 S1 (FT · arm–target 최소 거리 ≤ 50 mm 는 후보 선별 기준일 뿐) 을 만족하는 배치, 부족하면 규칙 기반 변형 배치. 평가용 허용 접촉 정의는 세 조건 공통 (§ "사용자 판정 A5") |
| 4 | T46f 단계 3 – 5 · CL4 (DT-02 extent block) | `T46f.task.md` §6 · §3 | 단계 4 전에 비교 조건 · 장면 수 · GPU 시간 · 진행/중단 기준을 정리해 **사용자 확인 후** 실행. oracle 은 T46e (`--target-extent oracle-visible` + client `--safe-oracle-seg`) |
| 5 | **T57 서빙 구현 (위임, 대기)** | `T57.task.md` "구현" 절 + 끝 절 | 단계 1: `--attention-source noun_L10H7` 류 (명사 attention 을 AG3S 입력으로, 기본 off, off bit-identical · on 은 T54c parity). 단계 2: task gate (`task_ready = HELD ∧ probe`, T55 probe v5 · C 0.1 · 임계 0.3, 기본 off, attach/detach 무관) — **task_ready 연결 위치는 구현 전 사용자 판정.** probe 정의 · 판정 근거는 B |
| 6 | **SL5b reach (benchmark 인계, 대기)** | `/mnt/dev/AG3S_VLA-benchmark/AG3S/ag3s/docs/handoff/SL5b-reach-to-development.md` (commit `6e82f4e9`, feature/panda-benchmark) | 가려진 들어 올린 target 의 마지막 centroid 로 reach > 120 mm → attach 거절 → held object 가 obstacle 로 남아 HOLD. benchmark 는 latch · reach 120 mm 를 바꾸지 않는다. 빈도: LIBERO 10 중 V11AB-B 실패 7 중 6 (benchmark 측 기록). 먼저 원인 확인 |
| 7 | T60 P · H 평가 정의 이관 | e4 의 `HANDOFF_e4_to_B.md` (A 로 넘길 것 절) | DX-00 소유 |

## 3. 재측정 대기 (A 몫, 로그 표에 등록됨)
#41 DX-01 조건 A (oracle identity, 미완료) · #42 T46e 오류 후보 5–8 mm 원인 · #43 oracle closed-loop · #44 최종 형상의 실물 실시간 · 비동기 예산 · #45 손목 / palm 비용 분리 (보류) · #46 T46f closed-loop.

## 4. 정리할 수 있는 worktree (모두 메인에 적용됨 — 지우기 전 내용 확인)
| worktree | 상태 |
|---|---|
| `.claude/worktrees/agent-a770cbc1ab3162e54` (T46e, base 15314ba3) | `cf38c9db` 로 적용됨 |
| `.claude/worktrees/t46e-apply` (detached 39dfb5dc) | 검증용, 적용됨 |
| `.claude/worktrees/agent-a48b7e7cad456d7a9` (A14 · A14b, base ab9b3fba) | `8df34591` 로 적용됨 (A14c 경유) |
| `.claude/worktrees/t46f-a14-apply` (detached cf38c9db) | A14c 검증용, 적용됨. `.runtime/legacy-*` fixture 복사본 있음 |

## 5. 다른 작업의 미커밋 (A 가 commit 하지 않는다)
`paper/**` (논문 작업), `.vscode/settings.json`, T52-T57 의 `t54c_records.py` · test · `T52-T57.plan.task.md` · `figures/t54c|t55|t58|t59` · `handoff/T54c|T55|T56|T57|T58|T59.*`, `figures/ppt-1013/` · `PPT_TO_V11AB.task.md`.

## 6. 조율 상대 (2026-10-14 기준, 이름은 바뀔 수 있다 — ListAgents)
- `T52-T57_ATS,HSR 개선작업` · `재검증 작업(T44~T56)` (e4): 인계 중, 끝나면 종료 → B 로 통합.
- `ag3s-vla-development-70`: T58 (장애물 closed-loop pilot, port 8294 – 8295). palm coverage 누락을 처음 관찰 (C3 계기).
- `ag3s-vla-development-a3`: 역할 미확인.
- benchmark (`ag3s-vla-benchmark-ab` · `-8f` 등): GPU lock 만 조율. SafeLIBERO Spatial dev paired (~1.5 h) 진행 중일 수 있다.
- e4 의 T61 rollout 은 `95b6586c` snapshot 으로 고정 (메인 변경과 무관). T52-T57 의 T56 test 는 `8df34591` 이후 HEAD 에서 T61 rollout 뒤 시작.

## 7. 함정 (이번 세션에서 실제로 났다)
- **`scripts/setup/env.sh` 는 메인으로 `cd` 한다 + `pyproject` 의 `pythonpath = ["."]`** → worktree 검증은 env.sh → `cd <worktree>` → import 경로 확인 (guard). 두 번 무효 실행이 났다 (로그 T46f §5).
- 공용 pod 메모리: `test_t46f_a14_wrist_cover.py` 는 peak RSS ≈ 32 GB. 전체 pytest 는 묶음 실행 + pod guard (`results/development/20261013T135703+0900_cf38c9dbca7b_e00b546bda0d/impl/T46f-A14c/` 의 방식). cgroup `oom_kill` 은 39 (누적).
- GPU lock 보유 process 는 다른 container 라 pid 가 안 보일 수 있다 — 세션에 물어본다.
- 이 container 에는 tzdata 가 없다: `TZ=Asia/Seoul date` 가 UTC 를 낸다. KST = `date -u -d '+9 hours'`.
- 세션 이름이 resume 마다 바뀐다 (`-72` → `-d2`). 메시지 주소는 ListAgents 로 확인.
- sub-agent 가 API 한도로 끊기면 worktree 에 반쯤 된 상태가 남는다 — 재개하지 말고 상태를 보고 직접 마무리하거나 task 로 새로 띄운다.

## 8. raw / run dir
- V-T46f-A14: `results/development/20261014T155631+0900_cf38c9dbca7b_04a1d797f62a/verify/T46f-A14/` (timing · audit · palm · pose · baseline, `p/` 에 driver · guard `sfcommon.py`).
- A14c pytest · patch: `results/development/20261013T135703+0900_cf38c9dbca7b_e00b546bda0d/impl/T46f-A14c/`.
- T46e-apply: `results/development/20261013T115928+0900_39dfb5dc2b38_a0fa7e4af1b6/verify/T46e-apply/`. T46e-D: `…/20261013T110034+0900_39dfb5dc2b38_7016ed1e8da5/verify/T46e-D/`.
