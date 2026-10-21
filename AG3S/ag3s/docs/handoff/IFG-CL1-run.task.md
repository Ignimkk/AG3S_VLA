# IFG-CL1-run — closed-loop 짝 비교 (C1a+noun vs attention-surface) + 실시간성 측정 (verifier)

작성: 2026-10-16 KST, lead. STEP 번호 없음 (IFG 기록은 `doc/active/reviews/vla_dump_20261015/WORKLOG_S0S3_KO.md`). 구현: `IFG-CL1.impl.md`.

## 사용자 판정 (2026-10-16)

| 항목 | 판정 |
|---|---|
| 비교군 | **C1aN** = C1a flag set + `--attention-source noun_target:L10H7` (대조군) vs **SURF** = C1aN + `--target-extent attention-surface --robot-mask render` (제안). 기존 C1a (L8H2) 는 collector 의 같은 seed 기록을 참고로만 쓴다 (새로 돌리지 않음) |
| TSDF robot 제거 | 구 ∪ render (현재 구현 그대로) |
| seed | held-out **성공** 주 평가 seed (collector 가 끝난 시점의 전부) |
| 주 지표 | 비허용 접촉 없는 작업 성공률 (task success ∧ 실제 MuJoCo 기준 비허용 link–target 접촉 0 ∧ 장애물 접촉 0). 정지 (HOLD) 수 · 완료 시간 함께 |
| 실시간성 | 반드시 측정 (아래 §4) |

## 1. 실행 틀 — collector 틀을 복사해 쓴다

참조: `/mnt/dev/AG3S_VLA-storage/results/development/20261016T000420+0900_30955b6ca8ef_2498252a873f/heldout/p/` (`hc_launch.sh`, `run_chain.sh`,
`start_server.sh`, `env.sh`, `analyze_cl.py`, `check_run.py`, `code_md5.sh`, `import_check.py`, `curobo_baseline.sh` …) 와 그 run 의
snapshot 생성 방식 (`snapshot_head.txt`, 상위 T54c `GENERATED.json`).

- 새 run: `.venv-openpi-live/bin/python -m common.run_output` → `$RUN`. 산출물은 `$RUN/verify/IFG-CL1/` 아래.
- **snapshot 은 commit 하지 않은 working tree 를 담아야 한다** (구현이 아직 commit 전). 브랜치 · index 를 건드리지 말고 임시 index 로 tree 를 만든다:
  `GIT_INDEX_FILE=$tmp git read-tree HEAD; GIT_INDEX_FILE=$tmp git add AG3S tests; tree=$(GIT_INDEX_FILE=$tmp git write-tree)` → `git archive $tree`.
  `AG3S/` · `tests/` 외 경로는 HEAD 그대로 (다른 세션의 paper · doc 변경을 넣지 않는다). tree SHA 와 `git diff HEAD --stat -- AG3S tests` 를
  `$RUN/snapshot_head.txt` 에 기록. robot mesh asset 등 collector snapshot 이 따로 챙긴 것이 있으면 같은 방식으로 챙긴다.
- arm 정의를 `env.sh` 의 `is_arm` / `start_server.sh` 에 추가: `C1aN`, `SURF`. 서버 flag 는 collector `C1a` 와 **완전히 같고** 위 추가 flag 만 다르다.
  SURF 서버는 `MUJOCO_GL=osmesa` (render mask; 구현이 osmesa 로 검증됨). 서버 log 검사 (`run_chain.sh` 의 refuse 목록) 는 C1a 것을 유지하고,
  arm 별로 추가 확인: C1aN/SURF = attention source 가 `noun_target:L10H7` 로 뜬 log 줄; SURF = attention_surface warning · render `probe()` 성공 줄.
- client 인자: collector 의 그 seed 기록 `manifest.json` `extra.argv` 와 같게 (episode, seed, warmup, obstacle config/profile, safe-* 옵션). 출력 경로만 새 run.
- seed 목록: **GPU lock 획득 직후** (= collector rollout · dump 종료 후) collector `heldout/state.json` 의 runs 중 `role ∈ {main, obstacle}` · `status = done` · `success = true` 전부. 목록을
  `$RUN/verify/IFG-CL1/seeds.json` 에 rollout 전에 쓴다. 짝 순서: seed 마다 C1aN · SURF 를 두 포트에 번갈아 배정 (포트 효과 상쇄).
- **GPU:** collector 가 `/tmp/ag3s-vla-gpu-validation.lock` 을 dump 끝까지 잡고 있다. launcher 는 그 lock 을 **blocking flock 으로 기다린다**
  (collector 를 멈추거나 그 process 를 건드리지 않는다). lock 획득 후 collector 처럼 foreign compute app 없음 · cuRobo baseline 일치를 확인.
  서버 ≤ 2, port 8292/8293, worker 한도 준수.

## 2. smoke gate (전체 전에 자동)

첫 seed 1 개 × 2 arm 을 먼저 돌리고 다음을 확인한 뒤에만 나머지를 진행. 하나라도 어긋나면 멈추고 `smoke_report.json` 에 이유.
- 두 run 모두 정상 종료, 완전성 (`check_run.py`) 통과.
- SURF 서버 기록에서 attention_surface 가 **실제로 적용된** frame 비율 (`target_free.attention_surface` 적용 · `identity_unresolved` / abstain fallback 제외)
  을 세어 기록. 적용 비율 < 30 % 면 멈춘다 (두 arm 이 사실상 같아지므로 lead 판단 필요).
- render mask 실패 (`robot_mask_render.why = render_failed`) 비율 기록.

## 3. 결과 (verify.json `AG3S/ag3s/docs/handoff/IFG-CL1.verify.json`, 해석 없이 수치)

seed · arm 별: task success, grasp, failure_mode (collector `analyze_cl.py`), 실제 MuJoCo 접촉 (비허용 link–target, 장애물–robot/물체, 이웃 과일),
HOLD chunk 수, 완료 시간. 짝 집계: 주 지표 짝 표 (both / only C1aN / only SURF / neither) + McNemar, stratum 별 (clear / adjacent / place).
SURF 진단: 적용 frame 비율, fallback 이유 분포, target-free 미세 계층이 지운 seed 수 (arm 별), code-1 pixel 수.
collector C1a (L8H2) 같은 seed 결과는 참고 열로.

## 4. 실시간성 (사용자 요청)

요청 (chunk) 마다 arm 별로:
- 서버 단계별 ms: VLA 추론, attention 추출, AG3S perception (점구름 · TSDF · ESDF), `robot_render_mask`, `attention_surface`, TrajOpt, 서버 총합.
  기록 위치 (record-constraints chunk · summary 의 profiler, 서버 log) 를 먼저 찾아 적는다. 없는 단계는 `not_measured`.
- client 왕복 (요청 → 응답) ms (`frames.jsonl` / client log).
- 요약: median · p90 · p95 · max, **chunk 예산 533 ms (8 step @ 15 Hz) 초과 비율**, SURF − C1aN 증가분 (짝 chunk 기준이 아니면 분포 비교).
- figure: arm 별 단계 누적 막대 (median) + 왕복 분포 (`AG3S/ag3s/docs/figures/ifg-cl1/`, + `.json`).

## 하지 말 것

- 코드 수정 금지 (`*.py` 변경 필요하면 멈추고 보고). 해석 · log 문서 작성 금지 (수치 · figure 만).
- collector 또는 다른 세션 process 종료 금지. lock 없이 GPU 사용 금지.
- 끝나면 자기 process 종료 + `python3 scripts/setup/reclaim_cache.py "$RUN"` 한 줄 보고.

## 넘길 것

launcher 를 background (nohup) 로 띄운 뒤에는 **기다리지 말고** 바로 보고: `$RUN`, launcher pid · log 경로, seeds.json, smoke gate 조건,
예상 소요. (결과 집계 · verify.json · figure 는 rollout 이 끝난 뒤 lead 가 새 verifier 로 맡긴다 — 집계 스크립트는 지금 만들어 두고 smoke 결과로 동작을 확인할 수 있으면 확인.)
