# SL7i — call마다 표면 상태 기록 + 표면 추출 · 기권 시 공 판 LIBERO 9 재실행

> writer: lead (benchmark checkout), 2026-11-04 07:3x KST. **결과를 보기 전에 고정.** 판정 주체: lead 자율 판정 D9 (`doc/active/experiments/SL7_AUTONOMOUS_DECISIONS_20261104_KO.md`).
> 앞 단계: SL7g — `-pc-surface-fb` 5/9. 서버 log에는 episode마다 첫 거리장의 제외 방식만 한 줄 남는다. 그래서 call마다 표면이 쓰였는지 공으로 물러났는지는 셀 수 없었다.

## 변경 (기록 전용, 행동 불변)

- Panda 서버 `AG3S/adapters/panda/server.py` `surface_call_record`가 response key `panda/surface`를 만든다. 내용은 `AG3S._surface_frame` scalar(`applied`, `why`, …)와 `t78` region · target_check · hold · hull · seed_ats_gate scalar, 그리고 `state` = `surface` / `fallback` / `none`이다. 표면 범위(`target_extent == attention_surface`) 판에서만 key가 생긴다. V11AB · pb · pc판 response는 그대로다.
- client `benchmark/runners/libero_safe_client.py`는 key가 있을 때만 `*_calls.jsonl` row에 `surface`를 쓴다.
- 테스트는 `tests/benchmark/test_sl7i_surface_record.py`다. 개발은 worktree(`wip/sl7i-surface-record`)에서 했고, 실행 중인 SafeLIBERO closed loop(SL7h)이 끝난 뒤에 merge한다(실행 중 runner 수정 금지).

## 실험

`AG3S-final-panda-v3-e0to-pc-surface-fb`를 LIBERO 9(SL7g와 같은 장면 · seed · conversion C)로 다시 돈다.

## 판정 규칙 (결과 전 고정)

1. **행동 불변 확인:** 9 episode의 `actions_response`가 SL7g run과 bit 단위로 같아야 한다. 다르면 기록 변경이 행동을 바꾼 것이므로 merge를 되돌리고 보고한다. 단, 서버 GPU 비결정성 때문에 SL7g 안에서도 같은 판 반복이 다를 수 있다. 그 경우에는 공식 성공 · 포화가 같은지로 보고, "bit 동일 아님"을 그대로 적는다.
2. **표면 사용 비율:** 전체 call 중 `state == surface` 비율 s를 episode별 · 전체로 보고한다.
   - s < 10% → 이 판은 사실상 "pc판 + 드문 표면"이다. LIBERO 5/9가 표면 추출 덕이라고 말할 수 없다고 적는다.
   - s ≥ 10% → 성공 · 실패 episode의 s와 `why`(기권 · 기각 사유)를 비교해 보고한다.
3. task 7(pc판 성공, 이 판 실패)에서 표면이 쓰인 call과 그 직후 행동 변경을 따로 본다.
