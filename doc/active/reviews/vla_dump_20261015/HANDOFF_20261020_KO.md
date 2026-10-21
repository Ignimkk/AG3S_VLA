# 세션 인계 (2026-10-20 19:35 KST) — 진행 중 · 할 일 · 기억할 것

새 세션은 이 문서 → [WORKLOG_S0S3_KO.md](WORKLOG_S0S3_KO.md) 마지막 다섯 절(2026-10-20 09:40 · 10:40 · 12:20 · 17:00 · 18:20 · 19:30) 순서로 읽는다.
공통 run: `/mnt/dev/AG3S_VLA-storage/results/development/20261019T132719+0900_f15e587290f6_11408fd6d27f` (아래 `$RUN`).

---

## 1. 지금 진행 중

### 10 episode 검증: 닫는 동안 유지한 형상을 관측으로 세지 않기 (HSR 쥠 인정 시점)

| 항목 | 내용 |
|---|---|
| 무엇 | 기존 방식 + target 분리 수정 + 손 기준 쥠 거리 + **이번 수정** (`--latch-reach-kept-unseen`). 직전 실험과 같은 10 장면에서 이 설정 하나만 더한 비교 |
| 기대 | 들어 올린 첫 순간부터 손 기준 쥠 거리 적용 → "들렸는데 쥠 인정 안 됨" chunk 1 → 0 (CL3 423 · 831 · 936). 423 · 936 결과 변화 여부 |
| run | `$RUN/verify/IFG-10EP-HSR-KEPTUNSEEN` (launcher 10:22 UTC 시작, benchmark 작업 600초 부재 → GPU lock → smoke → 9 장면 → 분석 · replay. 약 40분) |
| 상태 확인 | `grep -E "lock acquired|smoke|ROLLOUTS|ANALYSIS|STOPPED" $RUN/verify/IFG-10EP-HSR-KEPTUNSEEN/spl_launch.out` · `ls $RUN/verify/IFG-10EP-HSR-KEPTUNSEEN/runs/SPLK/*.done | wc -l` (10이면 완료) |
| 끝나면 할 일 | ① benchmark 세션(`uds:/tmp/cc-socks/3207780.sock`, 이름 ag3s-vla-benchmark-2d)에 GPU 반납 알림 ② 집계: 직전 kit 의 `agg3arm` 방식대로 복사본 폴더를 만들고 (`IFG-10EP-TARGETSPLIT-HANDREACH/agg3arm` 참고, **symlink 금지**) agg3.py 로 짝 비교 ③ `ifg_10ep_handreach.py` 를 이 arm(SPLK)까지 넣도록 고쳐 그림 생성 ④ verify.json · WORKLOG 기록 ⑤ `reclaim_cache.py` 한 줄 |
| 비교 기준 (이미 있는 기록) | 같은 10 장면의 기존 방식 · target 분리 수정 · target 분리 + 손 기준 쥠 거리 (4 / 6 / 7 성공). `AG3S/ag3s/docs/handoff/IFG-10EP-TARGETSPLIT-HANDREACH.verify.json` |

---

## 2. 진행해야 할 것 (우선순위 순)

1. **위 실험 결과 보고** (구성 요소별 진단 표 포함).
2. **LTE: 늦게 쥠 인정된 뒤 손에 든 물체가 거리장에 남는 문제 (CL3 423).** 쥠 인정 뒤에도 손가락 · 쥔 물체 구가 "쥔 물체 아닌 것" 과 26–31 mm 겹친다는 판정으로 공중 HOLD 35 chunk → 낙하. 쥔 물체를 거리장에서 지우는 위치가 들기 전 탁자 위로 잡힌 것으로 **추정**(미확인). 먼저 그 run 의 server 기록(ESDF · held_free)으로 확인 → 맞으면 손이 옮긴 위치 기준으로 지우는 수정.
3. **사용자 판정 대기 (세 가지)**:
   - 단계 정의: (1) 지금 갈래에 맞게 새로 정의 — 1단계 = 기존 방식의 ATS · HSR · LTE 결함 수정, 2단계 = 최종 실험 사슬의 장애물 장면 (허들 ③, 가림 · 경로 방해 ④) / (2) 문서(METHODOLOGY_EVOLUTION) 정의 유지 — 표면 영역 방식으로 돌아가 오늘 수정을 그 위에 적용
   - 계산 시간(현재 요청당 0.9–1.25 s, 예산 533 ms)을 1단계 완료 조건에 넣을지
   - 제안한 1단계 완료 조건 (아래 4절) 확정 여부
4. **방법론 B 중 closed-loop 미시험 항목** (논문 novelty 근거): feature 기억 identity 전환(한 번도 안 함), 안전 경보(기억 고착 수정 뒤 재시험 — 경보는 로봇만 지우는 mask 사용 + 쥔 동안 기억 갱신 중지), 3D 단계 판단, Grounded SAM 비교, 가림 장면 수집. 무엇을 할지는 사용자 판단.
5. **CCR 문제 (미수정)**: 사과 954 (이웃 과일과 5 mm 여유 → 손이 위로 밀림), 사과 211 (바나나가 잡는 경로 위). 이웃 과일 접촉 허용 정책 · 주 지표 정의와 함께 정해야 함.
6. **커밋 안 됨 (사용자 승인 필요)**: 아래 3절 코드 · test · study script · figure · handoff 문서. `git-commit` skill 규칙대로.
7. **전체 회귀 test 미실행**: attention 분할 · 기준 형상 대체 · 쥠 인정 수정 이후 `pytest tests/ag3s tests/trajopt` 전체를 아직 안 돌렸다 (관련 test 만 통과). 전체는 약 26분, 기존 실패 1건(`test_t43t_replay::test_off_is_bit_identical_to_the_pre_fix_commit`, HEAD 에서도 실패).
8. 오래된 할 일: run 폴더 이름에 설명 label 붙이기(`IFG-RUNLABEL.task.md`), 최종 실험 사슬 탐색 · 논문 figure (`FINAL_EXPERIMENT_AND_FIGURES_PLAN_KO.md`).

---

## 3. 기억해야 할 것

### 3.1 사용자 규칙 (2026-10-20, memory 에 저장됨)

- **줄임말 금지**: 설명 · 기록에서 내부 arm 코드(C1aN, SPL 등)를 쓰지 않고 풀어 쓴다. ATS · HSR · LTE · CCR 은 쓰되 처음에 뜻을 단다.
- **실패마다 원인 위치를 구성 요소로 판별**: ATS(Attention-based Target Segmentation, 목표 분할 · 추적) · HSR(Hidden-state Sub-task Recognition, 작업 단계 · 쥠 판단) · LTE(Link-specific Target Exemption, 링크별 거리장 · 쥔 물체) · CCR(Constrained Chunk Refinement, TrajOpt 수정 · 검증 · 멈춤). 정의: `paper/METHOD_GUIDE_KO.md`.
- **실험은 한 번에 10 episode 정도**로 빠르게. 장면은 결과 전에 규칙으로 고르고 "진단용, 성공률 표본 아님" 을 적는다.
- **고치는 위치는 원인 층에 맞춘다**: 궤적 원인 → TO(CCR), 관측 · 기하 원인 → AG3S 핵심.
- 진행 원칙: 설명 → 사용자 판정 → 수정 · 검증 → 기록. GPU 는 benchmark 세션에 알리고 lock 으로만 쓴다(동시 실행 금지). 다른 세션 process 는 종료하지 않는다.

### 3.2 내부 arm 코드 ↔ 풀어 쓴 이름 (기록 파일을 읽을 때만)

| 코드 | 뜻 |
|---|---|
| E0 | VLA 단독 (AG3S 없음) |
| C1aN | 기존 방식 (현재 serving + 명령 명사 attention 으로 target 선택) |
| SURF | 표면 영역 방식 (attention + depth 표면 graph 목표 영역 + 로봇 렌더 mask) — 문서의 1단계 |
| S2 | 기존 방식 + 안전 경보 + 3D 쥠 규칙 — 문서의 2단계 일부 (꺼짐) |
| SPL | 기존 방식 + target 분리 수정 (`--attention-split-partition --split-supersedes-anchor`) |
| SPLR | SPL + 손 기준 쥠 거리 (`--latch-reach-hand-relative`, 기존 T46g 기능) |
| SPLK | SPLR + 닫는 동안 유지한 형상을 관측으로 세지 않기 (`--latch-reach-kept-unseen`, 이번 수정) |

### 3.3 이번 세션의 결과 요약

| 실험 | 결과 | 기록 |
|---|---|---|
| 사과 24 장면 실패 분석 | 기존 방식만 실패한 장면 대부분이 사과 옆 과일(≤ 8.2 cm)과 관련: 목표 · 이웃 병합(ATS), 운반 중 쥠 해제(증상 HSR, 원인 ATS), 이웃 회피 밀림(CCR) | `IFG-APPLE-FAIL.verify.json` |
| 안전 경보 · 3D 쥠 규칙 원인 | 경보: 쥔 사과가 mask 에 지워져 기억이 다른 물체에 고착. 3D 쥠 규칙: 결정 변화 0 | `IFG-APPLE-S2-CAUSE.verify.json` |
| target 분리 수정 (사과 24) | 16 → 18 / 24, 실패→성공 2 · 성공→실패 0, 19 장면 실행 동일 | `IFG-SPLIT.verify.json` |
| target 분리 수정 (CL3 40, 독립) | 17 → 18 / 40, 실패→성공 2 · 성공→실패 1(936: 쥠 인정 실패 — HSR) → 미리 정한 규칙대로 기본 구성으로 채택 안 함 | `IFG-SPLIT.verify.json` `cl3_independent` |
| 진단 10 장면: + 손 기준 쥠 거리 | 4 → 6 → 7 / 10, 쥠 인정 회복, 831 성공, 퇴보 0. 남은 문제: 들린 첫 chunk 쥠 지연(HSR), 그 사이 손에 든 물체가 장애물(LTE → CCR 밀기 47–130 mm), 423 공중 HOLD(LTE) | `IFG-10EP-TARGETSPLIT-HANDREACH.verify.json` |
| 문서 방법론 B 시험 현황 | closed-loop 까지 간 것은 요소 1(표면 영역, 효과 미확정)뿐. 기억 identity 전환 · 3D 단계 판단 · Grounded SAM 비교 · 가림 장면은 closed-loop 미시험 | 이 문서 2절 4항 |

### 3.4 코드 변경 (커밋 안 됨, 모두 flag 기본 꺼짐 = 기존 출력 불변)

| 파일 | 내용 |
|---|---|
| `AG3S/ag3s/stages/attention_split.py` | `_partition` (attention 으로 붙은 덩어리 나누기) |
| `AG3S/ag3s/stages/target_grounding.py` | `SPLIT_SUPERSEDED_ANCHOR` (분할 조각이 merge 기준 형상 대체, 30 mm 이상 짧을 때만) |
| `AG3S/ag3s/config.py` | `attention_split_partition` · `attention_partition_min_drop` · `split_supersedes_anchor` · `latch_reach_kept_unseen` |
| `AG3S/ag3s/runtime/pipeline.py` | `manipulated_kept_this_frame`, kept frame 은 seen-state pin 을 옮기지 않음 |
| `AG3S/trajopt/safe_policy.py` | kept frame 에도 손 기준 쥠 거리 적용 |
| `AG3S/trajopt/serve_safe.py` | `--attention-split-partition` `--split-supersedes-anchor` `--latch-reach-kept-unseen` (뒤 것은 `--latch-reach-hand-relative` 필요) |
| tests | `tests/ag3s/test_ifg_split_partition.py`(15) · `tests/trajopt/test_ifg_hsr_kept_unseen.py`(4) 통과 |
| study scripts | `ifg_apple_s2_cause.py` `ifg_apple_fail.py` `ifg_apple_scene_bodies.py` `ifg_split_replay.py` `ifg_split_closedloop.py` `ifg_split_cl3.py` `ifg_10ep_handreach.py` |
| figures | `AG3S/ag3s/docs/figures/ifg-apple/ifg-apple-{s2-cause*,fail-*}` · `AG3S/ag3s/docs/figures/ifg-split/` |

### 3.5 함정 (반복하지 말 것)

1. **run 폴더 안에 run 밖을 가리키는 symlink 를 두면 서버가 시작을 거부한다** ("Output must be inside RUN"). 집계 폴더는 복사본으로 만든다.
2. **kit 을 복사할 때 바꿀 곳**: `env.sh`(V · O1 · arm flag · is_arm), `run_chain.sh`(arm 목록 · flag 검사), `check_run.py`, `analyze_cl.py` ARMS, `t46_stats_check.py`, `flag_check.py`, launcher 의 장면 수(n), `smoke_check_spl.py`(arm 경로 · 기준 run), **`run_replays_cl.sh` 의 runs/<ARM> 경로**, **`agg/address_check.json` 미리 복사**, `address_check.py` snapshot 경로, 코드 md5 기준(`code_md5.sh > code_md5_cl1.txt`). 새 코드면 snapshot 새로(`git read-tree HEAD; add -A AG3S tests; write-tree; git archive`).
3. **IFG-CL3 replay 의 서버 기록 정렬 오류**: 기존 방식 ep1096 · 1211 · 1529 · 1863(표면 영역 방식 9개)은 다른 run 의 chunk 를 가리킨다. plan 0 의 target 중심–기록 target 거리 > 5 cm 면 쓰지 않는다. 주 지표(client 기록)는 영향 없음.
4. **IFG-CL3 run 의 ep1034(기존 방식) · ep1135(표면 영역 방식) 폴더는 NFS 대기로 며칠째 멈춘 process 가 잡고 있다** — 읽지 말 것, 그 process 를 건드리지 말 것.
5. **대기 loop 에서 `pgrep -f <문자열>` 은 자기 명령줄에 걸린다** — 파일 존재 · 내용으로 기다린다.
6. 장면 재구성(`ifg_apple_scene_bodies.py`)은 cwd = snapshot, `AG3S_STORAGE_ROOT` 설정, `.venv-openpi-live` 로 돌린다. CL3 처럼 run 폴더를 읽으면 안 될 때는 `--seeds <seeds.json>` 모드.
7. GPU 쓰기 전 · 후 benchmark 세션에 SendMessage 로 알린다 (지금까지 매번 함).
