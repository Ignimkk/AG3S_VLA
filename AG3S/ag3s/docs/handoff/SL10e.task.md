# SL10e — 들어 올림(lift) 단계에도 "쥔 물체 아래 받침면" 허용 (결과 전 고정)

> writer: lead (benchmark checkout), 2026-11-05 04:4x KST, 자율 판정 시간(`doc/active/experiments/SL_AUTONOMOUS_DECISIONS_20261105_KO.md` E24).
> 근거(SL10d 기록, `outputs/2026-11-05_SL10d_쥔물체비우기_충돌구맞춤/data/held_phase_placement.json` · `held_attached_clear.json`):
> - 쥔 뒤 큰 편차(> 0.1 rad) call은 대부분 단계 라벨이 `lift` · `transport`다(pc판 8/9, 표면 판 6/11).
> - 쥔 물체 구(반지름 약 64 mm, 납작한 bowl보다 크다)가 받침면 근처 · 아래에 있다.
> - `placement` 라벨 call은 HSR6 ⑤(내려놓기 접촉 허용)가 쥔 물체 행을 풀고, 대부분 조용하다(28/28, 23/28).
> 이름: 보고에서는 "들어 올림 때도 아래 받침면 허용"이라고 부른다.

## 1. 변경 (기본 꺼짐 flag)

- 새 flag `--hsr6-placement-contact-lift`(기본 꺼짐, `--hsr6-placement-contact`가 있어야 함, Panda만).
- 켜면 `placement_contact_ok`의 단계 조건을 `placement` → `placement ∨ lift`로 넓힌다. latch 조건(HELD, 또는 CLOSING ∧ closed ∧ blocked)은 그대로다.
- 풀리는 행의 기하 조건도 그대로다(`_placement_contact_margin`):
  - 쥔 물체 행만 해당한다.
  - 가장 가까운 면이 아래쪽 원뿔(cos ≥ `PLACEMENT_BENEATH_COS`) 안에 있어야 한다.
  - 그 면이 쥔 물체 xy footprint 안에 있어야 한다.
  - 로봇 행(손 · 손가락 · 팔)은 바뀌지 않는다.
- `transport`에는 켜지 않는다. 운반 중에 아래를 지나는 장애물을 허용하지 않기 위해서다.
- 기록: `hsr6.detail.placement_contact.phase_rule = "placement|lift"`.
- 판:
  - `AG3S-final-panda-v3-e0to-pc-plift` = pc판 + flag
  - `AG3S-final-panda-v3-e0to-pc-surface-plift` = pc판 + 표면 + flag
- 테스트:
  - 꺼짐이면 bit 동일
  - 켬이면 lift 단계에서 ⑤가 켜지고, transport에서는 꺼짐
  - 새 판 = 기준 + flag 하나
  - RB-Y1 서버는 거절

## 2. 순서

1. LIBERO 9 두 판(독점, `--record-constraints full`)을 돌린다. 기준은 SL10 제약 기록 run이다.
2. LIBERO ≥ 기준인 판만 SafeLIBERO 11 · LIBERO-Safety 10을 돈다.
   - LIBERO-Safety는 `AG3S_PARALLEL_TEST_LAUNCHER=…launch_libero_safety_paired.py`를 쓴다. 메모리 admission을 보고 병렬 1–2로 돈다.
   - 기준은 SL10d의 같은 commit 계열 run이다. pc판은 `76283ac9` 재실행, 표면 판은 SL9 `914eb50f`다.

## 3. 판정 규칙 (결과 전 고정)

- **기제 확인 (필수):** 쥔 뒤 `lift` 라벨 call에서 ⑤가 쥔 물체 행을 하나 이상 푼 call 비율이 ≥ 0.5여야 한다. 그리고 쥔 뒤 큰 편차(> 0.1 rad) call 비율 또는 쥔 뒤 실행 action 포화 성분 중 하나가 기준 대비 절반 이하여야 한다.
  - 첫째가 안 되면 "구현이 의도대로 동작하지 않음"으로 보고하고 멈춘다.
  - 둘째가 안 되면 "기제 미확인"으로 적고, 주 지표 판정은 그대로 한다.
- **기준을 대신할 후보:** SL9 §5 그대로.
  - 세 벤치마크 모두 주 지표 ≥ 기준이다.
  - 금지 접촉 ≤ 기준이다.
  - 적어도 하나에서 주 지표 차이 > 2 episode다.
  - 2 이하 차이는 "구분 못 함"이다.
- **안전 쪽:** 쥔 뒤 실제 MuJoCo 접촉(쥔 물체 · 손 · 팔과 장애물)이 기준보다 늘면 후보가 아니다. 들어 올릴 때 아래에 장애물이 있는 장면이 있을 수 있어서다.
- HOLD 증가는 보호가 아니다. 실패마다 원인 층을 적는다.

## 4. 실행 규칙

- 구현은 worktree `wip/sl10e-plift`에서 한다. main merge는 GPU가 비었을 때 한다.
- 출력:
  - 원시 기록: storage `…_sl10e-*`
  - 보기: `outputs/2026-11-05_SL10e_들어올림받침면허용/`
