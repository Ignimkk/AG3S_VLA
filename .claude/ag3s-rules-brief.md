## AG3S 규칙 요약 (전문: `.claude/ag3s-rules.md` — 필요할 때만 읽는다)
- 로그는 `AG3S/ag3s/docs/AG3S_T0T6_LOG.md` 에 이어 쓰고 writer 는 scribe 뿐. scribe 는 `verify.json` 의 `numbers` 에 있는 숫자만 쓴다.
- 단계마다 실제 씬 · 그래프 · 표 figure 를 `AG3S/ag3s/docs/figures/` 에 (+ `.json` sidecar). 새 용어 · 발견 ID 는 그 자리에서 풀어 쓴다. 기술 용어는 English.
- 소유권: lead = `AG3S/ag3s/docs/handoff/*.task.md` · PLAN, implementer = `*.py` · `tests/`, verifier = `AG3S/ag3s/docs/figures/` · `*.verify.json` · `outputs/verify/`, scribe = `AG3S/ag3s/docs/*.md`.
- 진행: 설명 → 사용자 질문 (멈춤) → 판정 → 확정된 것만 수정 · 검증 → 기록. 추측을 기록하지 않는다.
- 토큰: 긴 sub-agent 를 재개하지 말고 task 문서로 새로 띄운다. verifier 기본은 Opus; 실행 · 대기 · 단순 집계만 맡길 때는 Agent 호출에 `model: sonnet` 을 지정하고, 결과 분석 · 원인 진단 · 구현은 Opus 로 따로 띄운다. 동시 agent ≤ 3. 그림은 평소 sidecar `.json` 으로 확인하되, 문제가 풀리지 않거나 꼭 필요할 때는 그림을 직접 연다.
- 공용 pod: working set (memory.current − inactive_file) + 추정 ≤ 64 GiB, worker ≤ 4, 서버 ≤ 2, GPU ≤ 130 GB, 다른 세션 프로세스 금지, 기존 workspace의 공유 `.venv-ag3s` · 포트 8123 금지; 새 checkout의 독립 `.venv-ag3s` 사용 가능.
