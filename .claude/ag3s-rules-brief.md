## AG3S 규칙 요약 (전문: `.claude/ag3s-rules.md` — 필요할 때만 읽는다)
- 로그는 `benchmark/ag3s/docs/AG3S_T0T6_LOG.md` 에 이어 쓰고 writer 는 scribe 뿐. scribe 는 `verify.json` 의 `numbers` 에 있는 숫자만 쓴다.
- 단계마다 실제 씬 · 그래프 · 표 figure 를 `docs/figures/` 에 (+ `.json` sidecar). 새 용어 · 발견 ID 는 그 자리에서 풀어 쓴다. 기술 용어는 English.
- 소유권: lead = `handoff/*.task.md` · PLAN, implementer = `*.py` · `tests/`, verifier = `figures/` · `*.verify.json` · `outputs/verify/`, scribe = `docs/*.md`.
- 진행: 설명 → 사용자 질문 (멈춤) → 판정 → 확정된 것만 수정 · 검증 → 기록. 추측을 기록하지 않는다.
- 토큰: 긴 sub-agent 를 재개하지 말고 task 문서로 새로 띄운다. 반복 실행 · 집계 verifier 는 Sonnet. 동시 agent ≤ 3. 그림은 sidecar `.json` 으로 확인.
- 공용 pod: working set (memory.current − inactive_file) + 추정 ≤ 64 GiB, worker ≤ 4, 서버 ≤ 2, GPU ≤ 130 GB, 다른 세션 프로세스 금지, `.venv-ag3s` · 포트 8123 금지.
