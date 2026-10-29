# PPT_TO_V11AB — V8 문제 그래프를 V11A · V11AB 기록으로 다시 그리기 (2026-10-13, lead)

목적: 발표 자료(`doc/archive/presentations/to_v8_v11ab/`)에서 V8 의 문제를 보인 그래프 두 종을 **같은 방법으로 V11A · V11AB 기록에 적용**해 개선 전후를 비교한다. 새 run 없음, **CPU 전용, 기록 재분석만**, GPU · 서버 · port 사용 금지. STEP 번호를 쓰지 않는 발표 지원 분석이다 (로그 기록은 사용자 판정 뒤).

## 입력 (읽기 전용)
- 기록: `$AG3S_STORAGE_ROOT/results/workspace/verify/T43/` — `V8/`, `V11/runs/{V11A,V11AB}/H/`, 기존 측정 `HX/` (`p/hx_measure.py` · `raw/`), `XM/` (`p/xm_cross.py` · `raw/`), `V11/xm_summary_{V11A,V11AB}.json`.
- 기존 V8 결과: `AG3S/ag3s/docs/figures/t43/t43hx-timeline.json` (V8 ep1807 s18073 carry apple, `P_geo_mm` · `M_geo_mm` · `S_geo_mm` · `hold`), `AG3S/ag3s/docs/handoff/T43XM.verify.json`.
- 코드 · 모델: **이 checkout** (`AG3S/trajopt` = 옛 `benchmark/trajopt`, `robots/rby1/description/models/rby1a/mujoco/model_transport.xml`). `/mnt/dev/work*` 옛 checkout · snapshot 은 input/code/env 경로로 쓰지 않는다 (CLAUDE.md). 옛 스크립트는 run dir 로 복사해 경로만 바꿔 쓴다. 꼭 필요한 것이 옛 snapshot 에만 있으면 **멈추고 lead 에 보고**.

## 할 일
0. **재현 확인 먼저.** 포팅한 측정으로 V8 ep1807 s18073 carry apple 의 `P_geo_mm` · `M_geo_mm` (row 153–182) 를 다시 계산해 `t43hx-timeline.json` 과 비교 (허용: 행별 |차| ≤ 0.5 mm). 안 맞으면 멈추고 보고.
1. **계획 vs 실제 timeline (그래프 1)**: V11A · V11AB 의 **같은 ep1807 s18073** H run 에서 사과를 든 허들 통과 구간의 행별 P (= 명령 C) · M · servo 예측 S 의 사과–허들 · 손가락–허들 기하 거리, HOLD 행. V8 과 같은 정의 (`hx_measure.py` docstring).
2. **집계 (같은 정의, H 14 run)**: V8 · V11A · V11AB 각각 — 허들 100 mm 안에서 실제(M) 거리가 10 mm 미만인 순간 수와 원인 분해 (HX 규칙), 운반 중 C − M 중앙 · p90, 허들 위 사과 바닥 – 막대 윗면 명령 · 실제 중앙.
3. **편향 profile (그래프 2)**:
   - (a) row k 별 손바닥 편향 |refined − reference| 중앙값 (carry_apple) — V8 · V11A · V11AB. `xm_summary_*.json` 의 값과 같은지 확인하고 출처를 적는다.
   - (b) **실제 연속 chunk trace**: ep1807 s18073 의 허들 통과 전후 연속 4–6 chunk 의 실행 row 별 손바닥 편향 (V8 · V11A · V11AB). 중앙값 반복이 아니라 실제 시계열.
4. figure (`make-figure` skill 스타일, 한글 라벨, `.json` sidecar):
   - `doc/archive/presentations/to_v8_v11ab/compare_figs/c_timeline.png` — V8 / V11A / V11AB 3 패널, 같은 축 범위 (그래프 1 비교).
   - `c_deflection_profile.png` (3a) · `c_deflection_trace.png` (3b) · `c_moments.png` (2 의 원인 분해 · C − M · 높이).
5. 산출: run dir 의 `verify/PPT_TO_V11AB.verify.json` (`numbers` 에 모든 수치 · 입력 경로 · 재현 확인 결과) + 위 figure. 해석 문장은 쓰지 않는다.

## 자원
- `source scripts/setup/env.sh`, `export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"`, CPU venv (`.venv-ag3s` 또는 mujoco 가 있는 venv).
- worker ≤ 4, working set ≤ 64 GiB, 다른 세션 프로세스 금지.
