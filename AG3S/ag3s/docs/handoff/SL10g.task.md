# SL10g — 진단: LIBERO-Safety "쥐기 시도 반복"(유형 B)의 쥐기 직전 TrajOpt 수정의 주인 (결과 전 고정, 채택 없음)

> writer: lead (benchmark checkout), 2026-11-05 08:4x KST, 자율 판정 시간(`doc/active/experiments/SL_AUTONOMOUS_DECISIONS_20261105_KO.md` E30).
> 근거:
> - `outputs/2026-11-05_기준판_실패지도_3벤치/`: LIBERO-Safety에서 VLA 단독은 10/10이고, pc판은 1/10이다. 그 실패의 가장 큰 유형이 B(쥐기 시도 반복, 5 episode)다. LIBERO에서는 B episode의 쥐기 직전 실행 action 변화가 크다(중앙값 0.99 vs 성공 0.03). LIBERO-Safety에서는 구분되지 않았다.
> - `outputs/2026-11-05_SL10f_미인증HOLD풀기_진단/`: 미인증 HOLD를 풀어도 1 episode만 돌아온다.

## 1. 변경 (기록 경로만)

- LIBERO-Safety paired launcher(`benchmark/runners/launch_libero_safety_paired.py`)에 `--record-constraints {none,full,…}`을 연다. 기본은 `none`으로 지금과 같다.
- LIBERO launcher와 같은 `record_constraints_argv` · `record_constraints_summary` 경로를 쓴다. 서버 판 · flag는 바꾸지 않는다.
- 테스트:
  - 기본 `none`이면 server argv가 지금과 bit 동일하다.
  - `full`이면 LIBERO launcher와 같은 기록 argv가 붙는다.

## 2. 실행

- pc판(`AG3S-final-panda-v3-e0to-pc`) LIBERO-Safety 10, `--record-constraints full`, 독점, 병렬 1.
- **J0 (필수):** 이전 pc판 run(`76283ac9`)과 episode별 성공 · 실행 action이 bit 동일해야 한다. 아니면 그 사실을 적고, 분석은 이 run 기준으로만 한다.

## 3. 기록할 숫자 (결과 전 고정, 판정 아님)

쥐기 직전(latch `latched` · `closing`) call을 유형 B episode와 성공 episode로 나눠 센다.
- TrajOpt 편차(|refined − reference| > 0.1 rad) call 비율
- 위반 · 20 mm 안 행의 sphere 종류(손가락 · 손바닥 · 팔)
- 그 행 근처 ATS target 점(20 mm 안) 비율
- 기록 거리장에 target 점이 점유로 남은 비율(SL10 E7과 같은 방식)
- 기준: target 근처 비율 ≥ 0.5이면 "쥘 target이 거리장에 남아 손을 밀어냄(LTE)"으로 적는다. 아니면 다른 주인(받침 · 이웃 물체)을 적는다.

## 4. 출력

- 원시 기록: storage `…_sl10g-*`
- 보기: `outputs/2026-11-05_SL10g_LS쥐기반복_제약기록/`
