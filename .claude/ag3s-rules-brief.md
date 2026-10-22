## AG3S 규칙 요약 (전문: `.claude/ag3s-rules.md` — 필요할 때만 읽는다)
- 로그: **T47 이후 STEP 은 `AG3S/ag3s/docs/AG3S_T47_LOG.md` (최신 정본)** 에 이어 쓴다. `AG3S_T0T6_LOG.md` 는 T46 까지 (이전 정본). T46 후속 접미사 STEP (`T46b` …) 도 새 로그. 새 STEP 번호는 다른 세션에 알려 예약한다. 읽을 때는 최신 → 오래된 순. writer 는 scribe 뿐이고 scribe 는 `verify.json` 의 `numbers` 에 있는 숫자만 쓴다.
- 단계마다 실제 씬 · 그래프 · 표 figure 를 `AG3S/ag3s/docs/figures/` 에 (+ `.json` sidecar). 새 용어 · 발견 ID 는 그 자리에서 풀어 쓴다. 기술 용어는 English.
- 소유권: lead = `AG3S/ag3s/docs/handoff/*.task.md` · PLAN, implementer = `*.py` · `tests/`, verifier = `AG3S/ag3s/docs/figures/` · `*.verify.json` · `${AG3S_RUN_DIR}/verify/`, scribe = `AG3S/ag3s/docs/*.md`.
- 진행: 설명 → 사용자 질문 (멈춤) → 판정 → 확정된 것만 수정 · 검증 → 기록. 추측을 기록하지 않는다.
- 토큰: 긴 sub-agent 를 재개하지 말고 task 문서로 새로 띄운다. verifier 기본은 Opus; 실행 · 대기 · 단순 집계만 맡길 때는 Agent 호출에 `model: sonnet` 을 지정하고, 결과 분석 · 원인 진단 · 구현은 Opus 로 따로 띄운다. 동시 agent ≤ 3. 그림은 평소 sidecar `.json` 으로 확인하되, 문제가 풀리지 않거나 꼭 필요할 때는 그림을 직접 연다.
- 공용 pod: working set (memory.current − inactive_file) + 추정 ≤ 64 GiB, worker ≤ 4, 서버 ≤ 2, GPU ≤ 130 GB, 다른 세션 프로세스 금지, 기존 workspace의 공유 `.venv-ag3s` · 포트 8123 금지; 새 checkout의 독립 `.venv-ag3s` 사용 가능.
- **closed-loop 기록은 place 뒤 2.5 초에서 끝낸다** (사용자 2026-10-21): client `--stop-after-place 2.5`, 새 kit 의 `client_args.py` · `check_run.py` · `run_chain.sh` 에 반영 (전문 참고).
- **메모리 회수 (필수):** 기록·테스트·추출·rollout 이 끝나면 handoff 전에 자기 process 를 종료하고 `python3 scripts/setup/reclaim_cache.py "$AG3S_RUN_DIR" <읽은 큰 입력…>` 출력 한 줄을 보고에 남긴다. `drop_caches`·`memory.reclaim`·다른 세션 process 종료는 승인 없이 금지.

- Shared input: `AG3S_STORAGE_ROOT`(checkout 옆 `AG3S_VLA-storage`); 이전 checkout을 input/code/env 경로로 사용하지 않는다. Interpreter는 `doc/active/SHARED_STORAGE_AND_PYTHON.md`를 따른다.
