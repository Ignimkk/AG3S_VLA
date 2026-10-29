## AG3S 규칙 요약 (전문: `.claude/ag3s-rules.md` — 필요할 때만 읽는다)
- 로그: **T47 이후 STEP 은 `AG3S/ag3s/docs/AG3S_T47_LOG.md` (최신 정본)** 에 이어 쓴다. `AG3S_T0T6_LOG.md` 는 T46 까지 (이전 정본). T46 후속 접미사 STEP (`T46b` …) 도 새 로그. 새 STEP 번호는 다른 세션에 알려 예약한다. 읽을 때는 최신 → 오래된 순. writer 는 scribe 뿐이고 scribe 는 `verify.json` 의 `numbers` 에 있는 숫자만 쓴다.
- 단계마다 실제 씬 · 그래프 · 표 figure 를 `AG3S/ag3s/docs/figures/` 에 (+ `.json` sidecar). 새 용어 · 발견 ID 는 그 자리에서 풀어 쓴다. 기술 용어는 English.
- 소유권: lead = `AG3S/ag3s/docs/handoff/*.task.md` · PLAN, implementer = `*.py` · `tests/`, verifier = `AG3S/ag3s/docs/figures/` · `*.verify.json` · `${AG3S_RUN_DIR}/verify/`, scribe = `AG3S/ag3s/docs/*.md`.
- 진행: 설명 → 사용자 질문 (멈춤) → 판정 → 확정된 것만 수정 · 검증 → 기록. 추측을 기록하지 않는다.
- **sub-agent 는 서로 독립인 작업 둘 이상을 동시에 돌릴 때만 쓴다.** 작업이 하나이거나 결과가 다음 단계를 정하는 순차 작업(closed-loop 실행 · 집계, 원인 진단 포함)은 lead 가 직접, 긴 실행은 background shell 로 한다.
- **실험:** 기준 · 장면(VLA 단독 성공, 약 10 episode) · 주 지표는 결과 전에 고정한다. 주 지표는 실제 MuJoCo 접촉 · 성공(HOLD 증가 ≠ 보호)이고, 분석은 실제 궤적 기준이다. 실패마다 ATS · HSR · LTE · CCR 중 원인 위치를 적고 그 층에서 고친다(궤적 = TrajOpt, 관측 · 기하 = AG3S). 새 기능은 기본 꺼짐 flag 로 넣는다.
- **결과 보기 (필수, 사용자 재지시 2026-10-26):** offline · closed-loop · 진단 등 **실험 단계 하나가 끝날 때마다, 사용자에게 결과를 보고하기 전에** checkout `outputs/<날짜>_<실험 설명>/` 를 만든다 — 영상(closed-loop, `python -m common.experiment_view`) + 실제 씬 그림(궤적 · 장면) + 그래프 + 표 + README. 진행 중인 실험은 끝난 부분부터 넣고 갱신한다. **보고에는 그 폴더 경로를 적고, outputs 없이 숫자만 보고하지 않는다.** lead가 직접 돌린 실험도 같다. 원시 기록은 storage run 에 둔다.
- **보고:** 목적 → 변경 → 실험 → 이전 대비 순서로 쓴다. 구성 코드 · 약어(SPL · F1 · arm 등) 대신 무슨 실험이고 무엇을 썼는지로 쓴다. KST = `date -u` + 9 h. 역할별 상세는 전문의 "실험 원칙" 절.
- 토큰: 긴 sub-agent 를 재개하지 말고 task 문서로 새로 띄운다. verifier 기본은 Opus; 실행 · 대기 · 단순 집계만 맡길 때는 Agent 호출에 `model: sonnet` 을 지정하고, 결과 분석 · 원인 진단 · 구현은 Opus 로 따로 띄운다. 동시 agent ≤ 3. 그림은 평소 sidecar `.json` 으로 확인하되, 문제가 풀리지 않거나 꼭 필요할 때는 그림을 직접 연다.
- 공용 pod: working set (memory.current − inactive_file) + 추정 ≤ 64 GiB, worker ≤ 4, 서버 ≤ 2, GPU ≤ 130 GB, 다른 세션 프로세스 금지, 기존 workspace의 공유 `.venv-ag3s` · 포트 8123 금지; 새 checkout의 독립 `.venv-ag3s` 사용 가능.
- **closed-loop 기록은 place 뒤 2.5 초에서 끝낸다** (사용자 2026-10-21): client `--stop-after-place 2.5`, 새 kit 의 `client_args.py` · `check_run.py` · `run_chain.sh` 에 반영 (전문 참고).
- **메모리 회수 (필수):** 기록·테스트·추출·rollout 이 끝나면 handoff 전에 자기 process 를 종료하고 `python3 scripts/setup/reclaim_cache.py "$AG3S_RUN_DIR" <읽은 큰 입력…>` 출력 한 줄을 보고에 남긴다. `drop_caches`·`memory.reclaim`·다른 세션 process 종료는 승인 없이 금지.

- Shared input: `AG3S_STORAGE_ROOT`(checkout 옆 `AG3S_VLA-storage`); 이전 checkout을 input/code/env 경로로 사용하지 않는다. Interpreter는 `doc/active/SHARED_STORAGE_AND_PYTHON.md`를 따른다.
