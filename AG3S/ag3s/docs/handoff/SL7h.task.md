# SL7h — 표면 추출 + 기권 시 공 판의 SafeLIBERO 11 episode closed loop

> writer: lead (benchmark checkout), 2026-11-04 07:2x KST. **결과를 보기 전에 고정** (closed loop은 07:19 KST에 시작했고, 이 문서를 쓰는 시점에는 첫 launcher 호출이 돌고 있으며 결과는 아직 보지 않았다). 판정 주체: lead 자율 판정 (`doc/active/experiments/SL7_AUTONOMOUS_DECISIONS_20261104_KO.md` D7 · D8).
> 앞 단계: SL7g — `AG3S-final-panda-v3-e0to-pc-surface-fb` LIBERO 9 5/9, 포화 10.4% (pc판 6/9, 10.0%).

## 판 · 장면

- 판: `AG3S-final-panda-v3-e0to-pc-surface-fb` (SL7g 정의 그대로).
- 장면: SafeLIBERO spatial Level I dev 중 VLA 단독이 공식 성공한 11 episode (t1e1, t1e4, t2e0–e4, t3e0, t3e2–e4), seed 7, conversion C. 실행 스크립트는 `experiments/segmentation/run_sl7_safelibero11.sh`이다.
- 비교: pc판 SL6-H smoke (같은 장면: 안전 성공 1 · 공식 성공 1 · 충돌 5), VLA 단독 (안전 성공 2 · 공식 성공 11 · 충돌 9).

## 지표

protocol `doc/active/experiments/SL7_P0M_PANDA_3BENCH_PROTOCOL_20261103_KO.md` §3과 같다. 공식 성공, 충돌(활성 장애물 합계 변위 > 1 mm), 안전 성공(성공 ∧ 충돌 없음)은 `episodes.jsonl` `safelibero.*`에서 읽는다. 보조 지표는 포화 · 큰 변경 call이다. pc판과 episode별로 짝 비교한다(얻은 · 잃은 성공, 새 · 없앤 충돌).

## 판정 규칙 (결과 전 고정)

표본이 11 episode라 차이 1은 잡음 범위로 본다.
1. 충돌 ≤ 6 이고 공식 성공 ≥ 1 → 표면 추출이 장애물 쪽 결과를 pc판보다 눈에 띄게 나쁘게 하지 않는다. `-pc-surface-fb`를 **Panda 최종 구성 후보**로 올린다(채택은 사용자 피드백). 남은 시간에는 LIBERO-Safety 단계(protocol §5-5)의 준비 상태를 확인한다.
2. 충돌 ≥ 7 (pc판 + 2 이상) → 표면 추출이 장애물을 지우거나 공 범위가 커졌다고 본다. 같은 판의 SafeLIBERO 20 episode 오프라인 재생(SL7f와 같은 방법)으로 장애물 잘못 지움을 pc판과 비교한다.
3. 공식 성공 0 (충돌 ≤ 6) → 성공 손실을 LIBERO 결과와 함께 보고하고, 후보로 올리지 않는다.
