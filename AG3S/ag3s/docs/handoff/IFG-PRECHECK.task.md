# IFG-PRECHECK — 관측 개선 착수 전 확인 2건 (CPU offline)

> writer: lead · 읽는 쪽: verifier

작성: 2026-10-22 KST, lead. 결과를 보기 전에 고정한다.
근거: WORKLOG_S0S3_KO.md 2026-10-21 23:30 절(관측 점검)과 23:57 절(사용자 판정: 순서 = 구 맞춤 → 기억 이동 → 거친 층),
`AG3S/ag3s/docs/figures/ifg-audit/README.md` "개선할 부분".

## 왜 지금 이것인가

- 순서 1(표면에 맞춘 구로 target 중심·제외 영역)의 기대 효과는 "손가락 거리장에 남은 사과 voxel(닫는 중 517개)이 TrajOpt 제약을 켜서 손을 밀거나 HOLD를 만든다"는 가정에 달려 있다. 앞의 954 분석은 미는 원인을 바나나로 봤다. 사과 voxel의 비중이 작으면 순서 1의 효과도 작다
- 순서 3(거친 층의 과대평가 +6.0 mm, 바구니 쪽)이 실제 접촉으로 이어졌다면 순서를 앞으로 올린다(실제 contact가 주 지표)

## 데이터

- run root: `/mnt/dev/AG3S_VLA-storage/results/development/20261019T132719+0900_f15e587290f6_11408fd6d27f/verify/`
- 주 데이터: `IFG-APPLE-SPLIT` (사과 24 장면, arm SPL, 요청 1,800개, 서버 constraint record 포함). 관측 점검(IFG-AUDIT-*)과 같은 데이터
- 보조: `IFG-APPLE-NBC` (이웃 접촉 20 mm 구성, 211 실패 유지)
- GT 재구성은 `AG3S/ag3s/experiments/studies/ifg_audit_lte.py` · `ifg_audit_ats.py` 방식(요청별 planning row qpos로 MuJoCo 장면 재구성, 충돌 geom 거리)을 재사용한다
- 기존 input은 수정하지 않는다. 출력은 새 run(`python -m common.run_output`)의 `verify/IFG-PRECHECK-{A,B}/`

## 측정 A — TrajOpt 제약을 켠 voxel의 출처 (→ verifier A)

| # | 무엇을 재나 | 어떤 숫자로 |
|---|---|---|
| A1 | SPLIT 실패 장면 전부(211 · 954 · 320 포함, 24 장면 중 실패 6) + 대조로 성공 장면 중 이웃 있는 것 3개(250 포함)에서, 쥐기 전(latched · closing) 요청마다 **활성 충돌 제약**(TrajOpt가 margin 안 / 침범으로 본 손 구-row)을 모은다 | 장면 · 단계별 활성 제약 수 |
| A2 | 각 활성 제약에서 거리장이 가리킨 장애물이 GT로 무엇인지 분류한다: target 사과 / 이웃 과일(바나나 등) / 바구니 · 기둥 / 탁자 / 기타 · 미상(GT 물체 모두에서 먼 거짓 voxel). 분류 규칙(예: 구 중심에서 거리장 값과 가장 가까운 GT 표면 거리를 갖는 물체, 허용 오차)은 결과 보기 전에 verify.json `definitions`에 적는다 | 출처별 비율, 장면별 |
| A3 | 같은 요청의 TrajOpt 결과: 수정량(손끝 이동 mm), HOLD 여부, 침범 mm를 출처별로 묶는다 | 출처별 수정량 중앙값 · HOLD 요청 수 |
| A4 | 211의 요청 10–16(손바닥 target 163 mm 근처 HOLD, "일찍 도착" 미분리)을 따로 표로 낸다. SPLIT와 NBC 둘 다 | 요청별 활성 제약 출처 · HOLD · 손바닥–target 거리 |

## 측정 B — 실제 접촉 (→ verifier B)

| # | 무엇을 재나 | 어떤 숫자로 |
|---|---|---|
| B1 | SPLIT 24 장면 전체 실행 기록(MuJoCo contact, 실제 qpos 기준)에서 로봇 링크(손 · 손가락 · 팔 전부)와 바구니 · 기둥 · 탁자 · 이웃 과일의 접촉 | 장면별 접촉 step 수, 접촉 상대, 단계 |
| B2 | 접촉 순간 근처(같은 요청)의 거친 층 / 고운 층 거리장 값 대 GT 거리 | 접촉별 field − GT (mm), 층 |
| B3 | 접촉이 충돌 검사 대상 링크(손 링크 6개)인지, 미검사 링크(팔 · 손목 · 팔꿈치)인지 | 링크 분류별 접촉 수 |

- 접촉 정의(penetration 임계, 의도된 target 사과 접촉 제외 규칙)는 결과 보기 전에 `definitions`에 적는다. 이웃 과일 접촉은 따로 센다

## 공통

- 수치는 `AG3S/ag3s/docs/handoff/IFG-PRECHECK-{A,B}.verify.json`의 `numbers`에, figure는 `AG3S/ag3s/docs/figures/ifg-precheck/`(실제 씬 1 · 그래프 1 · 표 1 이상, `.json` sidecar)
- script는 `AG3S/ag3s/experiments/studies/ifg_precheck_{a,b}.py`
- 해석은 쓰지 않는다. 숫자와 정의만
- CPU만 쓴다. worker ≤ 2 (pod 전체 ≤ 4), working set 확인. 끝나면 process를 종료하고 `python3 scripts/setup/reclaim_cache.py "$AG3S_RUN_DIR" <읽은 큰 입력…>` 출력 한 줄을 보고에 남긴다

## 이 STEP이 답하는 질문

실패 장면에서 TrajOpt를 움직인 제약 중 남은 target voxel이 차지하는 비중은 얼마이고, 거친 층 오차가 실제 접촉으로 이어진 사례가 있는가.

## 미리 정한 해석

- A에서 쥐기 전 활성 제약 중 target 사과 출처가 실패 장면의 수정 · HOLD를 주로 설명하면 → 순서 1 그대로 진행
- 이웃 과일 출처가 대부분이면 → 순서 1의 성공률 기대를 낮추고 사용자에게 다시 묻는다
- B에서 검사 대상 링크의 바구니 · 기둥 실제 접촉이 있고 그 순간 거친 층이 GT보다 멀게 봤으면 → 순서 3을 앞으로 올리는 안을 사용자에게 묻는다
