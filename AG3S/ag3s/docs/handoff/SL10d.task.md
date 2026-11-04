# SL10d — 쥔 물체 비우기를 충돌 질의 구와 맞추기 (결과 전 고정)

> writer: lead (benchmark checkout), 2026-11-05 03:2x KST, 자율 판정 시간(`doc/active/experiments/SL_AUTONOMOUS_DECISIONS_20261105_KO.md` E19).
> 근거: SL10c 재기록(storage `20261105T0323*_f0bd74a53803_sl10c-rec-*`, 행동은 앞 run과 bit 동일).
> - 쥔 동안 `held_free`(`AG3S/ag3s/runtime/pipeline.py` `_held_free_points`)가 비우는 구는 `held_spheres_of(held)`다. 이것은 lift 보정이 들어간 **구 1개**로, 쥔 몸통보다 약 47 mm 위에 있다(첫 held 프레임에는 capture 구가 더해진다).
> - 충돌 질의에 쓰이는 부착 구는 body_cover segment **2개**이고, lift가 없다.
> - 질의 구 부피 중 비워지는 비율은 0.45–0.60이다(처음 1–2 프레임만 0.84–0.92).
> - pc판 + action 한계 t1e0에서 쥠 29 call 중 HOLD 27이다. 최악 행은 모두 `attached_point`이고, clearance는 −1.4 … −31 mm다.
> - J1 geom 분류: 그 행 근처 점유 voxel의 주인은 쥔 bowl 자신이다(27/28).
> 이름: 보고에서는 "쥔 물체 비우기 = 충돌 구"라고 부른다.

## 1. 변경 (기본 꺼짐 flag)

- 새 flag `--held-free-query-spheres`(기본 꺼짐, Panda · RB-Y1 공통 코드지만 이번 판정은 Panda만).
- 켜면 `held_free`가 비우는 구에 **solver가 충돌 질의에 쓰는 부착 구**를 더한다. 이 구는 지금 q_now에서 parent link로 옮긴 local 구이고, solver linearizer의 `attached_states`와 같은 정의다.
- 바닥 규칙(받침면 + guard 아래는 비우지 않음), 목적지 근처 제외, 첫 프레임 capture 구는 그대로 둔다.
- 기존 구는 빼지 않는다. 합집합이다.
- 기록: `held_geometry_record.held_free.query_spheres_added`, 질의 구 비움 비율.
- 판:
  - `AG3S-final-panda-v3-e0to-pc-hfq` = pc판 + flag
  - `AG3S-final-panda-v3-e0to-pc-surface-hfq` = pc판 + 표면 + flag
- 테스트:
  - 꺼짐이면 bit 동일
  - 켬이면 합성 held 상태에서 질의 구 부피 비움 비율 ≥ 0.95
  - 바닥 아래 비움 0
  - 새 판 = 기준 + flag 하나

## 2. 순서

1. LIBERO 9 두 판(독점, `--record-constraints full`)을 돌린다. 기준은 SL10 제약 기록 run이다.
2. LIBERO 주 지표 ≥ 기준인 판만 SafeLIBERO 11 · LIBERO-Safety 10을 돈다.
   - LIBERO-Safety는 `AG3S_PARALLEL_TEST_LAUNCHER=…launch_libero_safety_paired.py`를 쓴다.
   - pc판의 SafeLIBERO · LIBERO-Safety 기준은 같은 commit에서 다시 돈다.

## 3. 판정 규칙 (결과 전 고정)

- **기제 확인 (필수):** 쥠 구간 call 중 서버 verdict 최악 행이 `attached_point`인 call의 비율이 기준 대비 절반 이하이고, 기록상 질의 구 비움 비율의 중앙값이 ≥ 0.95여야 한다.
  - 아니면 "구현이 의도대로 동작하지 않음"으로 보고하고 멈춘다.
- **보조 지표 (판정 아님, 기록):** 쥔 뒤 실행 action 포화 성분, 쥔 뒤 편차 > 0.1 rad call 비율, 쥔 뒤 HOLD 수.
- **기준을 대신할 후보:** SL9 §5 그대로.
  - 세 벤치마크 모두 주 지표 ≥ 기준이다.
  - 금지 접촉 ≤ 기준이다.
  - 적어도 하나에서 주 지표 차이 > 2 episode다.
  - 2 이하 차이는 "구분 못 함"이다.
- **안전 쪽:** 쥔 뒤 쥔 물체 · 손이 장애물에 실제로 접촉한 수가 기준보다 늘면 후보가 아니다. 비우기를 넓히면 쥔 물체 주변의 진짜 장애물까지 지울 위험이 있기 때문이다. 그래서 쥔 뒤 지운 voxel 중 쥔 물체 아닌 geom 근처 voxel 수도 J1 geom 분류로 기록한다.
- HOLD 증가는 보호가 아니다. 실패마다 원인 층을 적는다.

## 4. 실행 규칙

- 구현은 worktree `wip/sl10d-hfq`에서 한다. main merge는 GPU가 비었을 때 한다.
- 출력:
  - 원시 기록: storage `results/benchmark/<날짜_commit>_sl10d-*`
  - 보기: `outputs/2026-11-05_SL10d_쥔물체비우기_충돌구맞춤/`
