# development 세션 2 개 체제 (2026-10-14 사용자 판정) — 역할 정본

> writer: lead (세션 A). 사용자 판정 (2026-10-14): "3 개 세션 중 겹치거나 성격이 비슷한 세션은 하나로 합친다. develope 문서를 기반으로 개선 작업을 하는 본 세션이 메인이다. 2 개로 간다."
> 이전 3 세션: 본 세션 (T46~, `ag3s-vla-development-d2`) · `T52-T57_ATS,HSR 개선작업` · `재검증 작업(T44~T56)` (e4).
> 이 문서는 두 세션의 소유권 · 조율 규칙의 정본이다. 바뀌면 A 가 고치고 B 에 알린다.

## 1. 두 세션

| | **A — 메인 (develope)** | **B — identity · phase source 분석** |
|---|---|---|
| 세션 | 지금 본 세션 (`ag3s-vla-development-d2`, 이름이 바뀌면 ListAgents 로 확인) | 새 세션 (T52-T57 세션 + 재검증 세션 통합) |
| 기준 문서 | `doc/plans/develope_PLAN.md` (진행 정본) | develope_PLAN 의 DX-01 identity · DX-02 · DX-03 절 |
| 맡는 DX | DX-00 (평가 정의 · geometry, **T60 P · H 창 포함**), DX-01 **extent**, DX-04 LTE, DX-05, DX-06, DX-07, DX-09 | DX-01 **identity source** (attention head 선택 T51 · T56 · **T61** 통합, 명사 query), DX-02 HSR 대 strong rule (T55, T57 단계 2 probe 정의), DX-03 설계 |
| 실행 | **closed-loop 실행 · GPU 일정 조율 전부.** 서빙 코드 변경의 유일한 writer | offline 분석만. 새 rollout · 서빙 변경이 필요하면 task 문서로 A 에 요청 (A 가 실행) |
| 진행 중 · 대기 | T46f 단계 2 (진단 장면, 세션 정리 뒤), **T57 서빙 구현 (단계 1 명사 attention 입력 · 단계 2 task gate)**, SL5b reach (benchmark 인계), T60b 류 실행 중 막힘 진단 | T61 (사전 고정 head 선택, rollout · 추출 뒤의 선택 · held-out 분석), T56 (self label), T57 의 판정 근거 · probe 정의, T54c 후속 |
| STEP 번호 | `T46x` 접미사, 새 번호는 B 에 알리고 쓴다 | T5x · T6x 를 이어 쓴다 (T58 은 dev-70 사용 중). 새 번호는 A 에 알리고 쓴다 |

## 2. 파일 소유권

| 파일 | 쓰는 쪽 | 비고 |
|---|---|---|
| `doc/plans/develope_PLAN.md` | **A 만** (사용자 직접 수정 포함) | B 는 변경 제안을 A 에 보낸다 |
| 서빙 code — `AG3S/ag3s/{config.py,runtime/,fields/,constraints/,robot_models/}`, `AG3S/trajopt/` (experiments 제외), `integration/rby1/bringup/` | **A 만** | B 의 서빙 요구는 `handoff/<STEP>.task.md` "구현" 절로 |
| offline 분석 code — `AG3S/ag3s/experiments/studies/t5x_*` · `t6x_*` 와 그 test | B | 기존 파일 동작은 바꾸지 않는다 (새 파일) |
| `AG3S/ag3s/experiments/reports/` · `AG3S/trajopt/experiments/` | 만든 쪽. 상대 파일은 고치기 전 알림 | |
| `handoff/<STEP>.*` · `figures/<step>/` | 그 STEP 의 세션 | |
| `AG3S/ag3s/docs/AG3S_T47_LOG.md` | 둘 다 자기 절을 쓴다. **쓰기 전 · 후 상대에게 알림.** **commit 은 A** (B 의 최종 diff OK 뒤, 이 파일만 pathspec) | 수치는 verify.json `numbers` 만 |
| `.claude/worktrees/*` | 만든 쪽 | |

## 3. 조율 규칙

- **GPU:** `( flock -w N 9 || exit 13; nvidia-smi 확인; CMD ) 9>/tmp/ag3s-vla-gpu-validation.lock`. closed-loop · TO 시간처럼 GPU 단독이 필요한 측정은 A 가 일정을 잡고 benchmark 세션 (ab · 8f 등) 에도 알린다. B 의 offline GPU 추출은 stage 마다 lock 을 잡고 시작 · 끝을 A 에 알린다.
- **port:** A 8290 – 8291, B 8292 – 8293 (dev-70 은 8294 – 8295).
- **메인 tree 서빙 code 변경:** A 가 적용 전 B 에 알리고, B 의 측정이 메인 tree 를 import 중이면 끝난 뒤 적용, 시작 · 끝 · hash 를 알린다. rollout 은 HEAD 의 `git archive` snapshot 으로 도는 것을 권장 (T61 방식).
- **실행 경로 guard:** `scripts/setup/env.sh` 는 메인으로 `cd` 하고 `pyproject` 의 `pythonpath = ["."]` 는 cwd 코드를 import 한다. worktree 에서 돌릴 때는 env.sh → `cd <worktree>` → import 경로 확인, 다르면 중단.
- **공용 pod:** working set ≤ 64 GiB, worker ≤ 4, 서버 ≤ 2. 큰 pytest (손목 cover test ≈ 32 GB) 는 pod 부하를 보고 돌린다.
- **commit:** 자기 파일만 pathspec. push 는 사용자가 한다. Co-Authored-By 를 넣지 않는다.
- **진행 원칙 (공통):** 설명 → 사용자 질문 (멈춤) → 판정 → 확정된 것만 수정 · 검증 → 기록. 사전 고정 기준을 사후에 바꾸지 않는다. 제안 방법의 우수성을 전제하지 않는다. 실제 MuJoCo contact 가 주 지표, 정지 ≠ 보호. 보고는 목적 + 이전 (V11AB = C0 · C1a) 대비 before → after.

## 4. 통합 절차

1. A 가 이 문서를 만든다 (완료).
2. **T52-T57 세션:** 진행 중인 T56 test 를 끝내고 (8df34591 HEAD, T61 rollout 뒤), 결과를 verify.json 으로 남긴다. `handoff/HANDOFF_T52-T57_to_B.md` 를 쓰고 자기 파일을 pathspec commit, A 에 알린 뒤 종료.
3. **재검증 세션 (e4):** 진행 중인 T61 rollout · 추출 (95b6586c snapshot) 을 끝낸다. 선택 · held-out 분석은 B 로 넘긴다. `handoff/HANDOFF_e4_to_B.md` 를 쓰고 자기 파일을 pathspec commit, A 에 알린 뒤 종료.
4. A 가 두 세션의 로그 기록 (미커밋) 을 B · 두 세션의 diff OK 로 한 번에 commit 한다.
5. 사용자가 새 세션 B 를 열고 부록 C 의 프롬프트를 보낸다. B 는 두 인계 문서와 이 문서를 읽고 A 에 시작을 알린다.

---

## 부록 A — T52-T57 세션에 보낼 인계 프롬프트

```
[사용자 지시 · 세션 통합] development 세션을 2 개로 정리합니다. 역할 정본: AG3S/ag3s/docs/handoff/SESSIONS_A_B.md (먼저 읽으세요).
- 메인 A = ag3s-vla-development-d2 (develope_PLAN · 서빙 코드 · closed-loop · GPU 일정 · 공유 로그 commit).
- 새 세션 B = identity · phase source 분석 (이 세션과 재검증 세션 e4 를 합친 것). 이 세션은 인계 후 종료합니다.

이 세션에서 할 일 (순서대로, 새 STEP 은 시작하지 않습니다):
1. 진행 중인 T56 test 를 8df34591 HEAD 에서 끝까지 실행하고 T56.verify.json 을 마무리합니다 (T61 rollout 뒤 시작 — 기존 약속대로). 결과 해석 · 판정 요청은 하지 않고 B 로 넘깁니다. 이미 받아 둔 사용자 판정만 기록합니다.
2. AG3S_T47_LOG.md 에 아직 쓰지 않은 측정 완료분이 있으면 verify.json 수치로 씁니다 (쓰기 전 · 후 A 에 알림). commit 은 A 가 합니다.
3. AG3S/ag3s/docs/handoff/HANDOFF_T52-T57_to_B.md 를 씁니다. 포함할 것:
   (a) STEP 별 상태 표 T47–T57 (완료 · 진행 · 판정 대기 · 미착수, 근거 파일 경로)
   (b) 받아 둔 사용자 판정 목록 (날짜, 문서 위치) — 특히 T54c 판정, T55 사전 고정 (T55.prereg.json), T56 사전 고정, T57 단계 1 · 2 (task_ready 연결은 구현 전 판정 대기)
   (c) 판정 대기 질문 목록 (사용자에게 아직 묻지 않았거나 답을 못 받은 것)
   (d) 미커밋 파일 · worktree · 실행 중 process · 사용 port · run dir 경로
   (e) B 가 다음에 할 일 순서 추천과 그 근거, A 에 요청할 서빙 · closed-loop 작업 (T57 은 이미 A 에 위임됨)
   (f) 알려진 함정 (head 0 기록 ep1800 · long 제외 등)
4. 자기 파일 (handoff · figures · t5x code · test) 을 pathspec 으로 commit 합니다 (공유 로그 · 다른 세션 파일 제외). push 하지 않습니다.
5. A (ag3s-vla-development-d2) 에 "인계 완료 + commit hash + HANDOFF 경로" 를 보내고, 사용자에게 종료해도 된다고 보고합니다.
GPU 를 쓰면 공용 lock · nvidia-smi 확인 · 시작/끝 알림을 지킵니다.
```

## 부록 B — 재검증 세션 (e4) 에 보낼 인계 프롬프트

```
[사용자 지시 · 세션 통합] development 세션을 2 개로 정리합니다. 역할 정본: AG3S/ag3s/docs/handoff/SESSIONS_A_B.md (먼저 읽으세요).
- 메인 A = ag3s-vla-development-d2 (develope_PLAN · 서빙 코드 · closed-loop · GPU 일정 · 공유 로그 commit). T60 의 P · H 평가 정의 (종료 구간 술어) 는 A 가 이어받습니다.
- 새 세션 B = identity · phase source 분석 (이 세션과 T52-T57 세션을 합친 것). T61 의 선택 · held-out 분석은 B 가 이어받습니다. 이 세션은 인계 후 종료합니다.

이 세션에서 할 일 (순서대로, 새 STEP 은 시작하지 않습니다):
1. 진행 중인 T61 rollout (95b6586c snapshot) 과 추출을 끝까지 실행하고 결과를 verify.json (상태 partial 가능) 으로 남깁니다. 선택 · held-out 판정은 열지 않고 B 로 넘깁니다 (사전 고정은 T61.task.md 그대로).
2. AG3S_T47_LOG.md 에 쓸 측정 완료분이 있으면 verify.json 수치로 씁니다 (쓰기 전 · 후 A 에 알림). commit 은 A 가 합니다.
3. AG3S/ag3s/docs/handoff/HANDOFF_e4_to_B.md 를 씁니다. 포함할 것:
   (a) STEP 별 상태 표 T44–T56 재검증 · T60 · T60b · T60c · T61 (근거 파일 경로)
   (b) 받아 둔 사용자 판정 목록 (날짜, 문서 위치) — 특히 T60 P · H 창의 사용 범위, T61 사전 고정 (선택 규칙 · held-out registry)
   (c) 판정 대기 질문 목록
   (d) 미커밋 파일 · 실행 중 process · 사용 port · snapshot · run dir 경로
   (e) A 로 넘길 것 (T60 P · H 술어의 정의 · 코드 위치, T60b 복귀 미도달 진단의 후속) 과 B 로 넘길 것 (T61 분석 순서)
   (f) 알려진 함정
4. 자기 파일 (T61 code · test · handoff · figures) 을 pathspec 으로 commit 합니다 (공유 로그 · 다른 세션 파일 제외). push 하지 않습니다.
5. A (ag3s-vla-development-d2) 에 "인계 완료 + commit hash + HANDOFF 경로" 를 보내고, 사용자에게 종료해도 된다고 보고합니다.
GPU 를 쓰면 공용 lock · nvidia-smi 확인 · 시작/끝 알림을 지킵니다.
```

## 부록 C — 새 세션 B 를 열 때 보낼 프롬프트

```
이 세션은 development 의 "B — identity · phase source 분석" 세션입니다. 이전 두 세션 (T52-T57_ATS,HSR 개선작업 · 재검증 작업(T44~T56)) 을 합쳐 이어받습니다. 메인 세션 A 는 ag3s-vla-development-d2 입니다 (ListAgents 로 이름 확인).

먼저 읽을 것 (순서):
1. CLAUDE.md · .claude/ag3s-rules.md · AG3S/ag3s/docs/handoff/SESSIONS_A_B.md (역할 · 소유권 · 조율 규칙의 정본)
2. AG3S/ag3s/docs/handoff/HANDOFF_T52-T57_to_B.md · HANDOFF_e4_to_B.md (이어받을 STEP · 판정 · 대기 질문)
3. doc/plans/develope_PLAN.md 의 §0 과 DX-01 · DX-02 · DX-03 절
4. AG3S/ag3s/docs/AG3S_T47_LOG.md 맨 위 표 (용어 · 진행 현황 · 물려받은 결정 · 재측정 대기) 와 T51 · T54c · T55 · T56 · T60 · T61 관련 절

역할:
- DX-01 identity source (attention head 선택 T51 · T56 · T61 을 하나의 선택 protocol 로 정리, 명사 query), DX-02 HSR 대 strong rule (T55, T57 단계 2 probe 정의), DX-03 설계.
- offline 분석만 합니다. 서빙 코드 (AG3S/ag3s/{config.py,runtime/,fields/,constraints/,robot_models/}, AG3S/trajopt/ 의 experiments 외, integration/rby1/bringup/) 와 develope_PLAN.md 는 고치지 않습니다. 새 rollout · closed-loop · 서빙 변경이 필요하면 handoff/<STEP>.task.md 의 "구현" 절로 써서 A 에 요청합니다.
- STEP 번호는 T5x · T6x 를 이어 쓰고 (T58 은 dev-70), 새 번호는 A 에 알린 뒤 씁니다. port 8292 – 8293.
- AG3S_T47_LOG.md 는 자기 절만 쓰고 쓰기 전 · 후 A 에 알립니다. commit 은 A 가 합니다. 나머지 자기 파일은 pathspec commit, push 는 하지 않습니다.
- GPU offline 추출은 /tmp/ag3s-vla-gpu-validation.lock 을 stage 마다 잡고 nvidia-smi 확인, 시작 · 끝을 A 에 알립니다. 공용 pod 한도 (working set ≤ 64 GiB, worker ≤ 4) 를 지킵니다.
- worktree 에서 돌릴 때는 env.sh 를 source 한 뒤 다시 worktree 로 cd 하고 import 경로를 확인합니다 (env.sh 가 메인으로 cd 함).
- sub-agent 는 병렬 작업이 필요할 때만 씁니다.

진행 원칙: 설명 → 사용자 질문 (멈춤) → 판정 → 확정된 것만 수정 · 검증 → 기록. 사전 고정 기준을 사후에 바꾸지 않고, 제안 방법의 우수성을 전제하지 않습니다. 보고는 목적 + 이전 (V11AB = L8H2 서빙) 대비 before → after 로, develope_PLAN 의 어느 Figure · Table 에 쓰이는지 함께 적습니다.

첫 응답에서 할 일: 위 문서를 읽고 (1) 이어받은 STEP 상태 표, (2) 판정 대기 질문 목록, (3) 다음 작업 순서 추천 을 정리해 저에게 보고한 뒤 멈추세요. 그리고 A (ag3s-vla-development-d2) 에 "B 세션 시작" 을 알리세요. 아직 실행은 시작하지 않습니다.
```
