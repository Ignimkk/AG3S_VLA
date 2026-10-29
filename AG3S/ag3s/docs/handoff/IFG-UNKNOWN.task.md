# IFG-UNKNOWN — 출처 "미상" 활성 제약 voxel의 정체 (CPU offline)

> writer: lead · 읽는 쪽: verifier

작성: 2026-10-22 KST, lead. 결과를 보기 전에 고정한다. 근거: WORKLOG_S0S3 2026-10-22 00:36 절,
사용자 판정 2026-10-22 "미상 확인 + 순서 1 병행".

## 왜 지금 이것인가

사전 확인 A(`IFG-PRECHECK-A.verify.json`)에서 실패 6 장면의 쥐기 전 활성 제약 row 5,423개 중 47.5%가 어느 GT 물체와도
허용 오차(10 / 20 mm) 안에서 맞지 않았다(미상). 미상 row는 가장 가까운 GT 물체에서 중앙값 31 mm 떨어져 있고,
가장 가까운 물체는 탁자 1,171 · 바구니 1,154 · target 260 row다. 320(64%) · 1313(76%) · 526(56%)에서 절반을 넘는다.
대조 성공 3 장면은 10.7%다. 무엇이 거리장에 들어가 TrajOpt를 움직였는지 모르면 관측 수정의 대상을 정할 수 없다.

## 데이터

- `IFG-PRECHECK-A`의 raw: `/mnt/dev/AG3S_VLA-storage/results/development/20261022T001543+0900_8e200ada6d53_f62e49bedd53/verify/IFG-PRECHECK-A/raw_v2/`
  (row별 출처 분류) · script `AG3S/ag3s/experiments/studies/ifg_precheck_a.py`
- 원 기록: `.../20261019T132719+0900_f15e587290f6_11408fd6d27f/verify/IFG-APPLE-SPLIT` (서버 constraint record, ESDF full, 관측 depth · 점군)
- 기존 input 수정 금지. 출력: 새 run의 `verify/IFG-UNKNOWN/`

## 측정 (→ verifier)

| # | 무엇을 재나 | 어떤 숫자로 |
|---|---|---|
| U1 | 미상 row마다 거리장이 가리키는 표면 voxel(구 중심에서 거리장 기울기 방향으로 d만큼)을 찾고, 그 voxel을 만든 **관측 점**의 출처를 분류한다: 로봇 자기 몸(FK 구 mask 밖으로 남은 로봇 점 — GT 로봇 geom 거리로 판정) / 쥔 · 떨어진 · 밀린 target 사과(그 요청 GT 위치) / 이웃 과일의 GT 밖 부분 / 탁자 · 바구니 · 기둥 표면에서 GT보다 튀어나온 점(depth 오차) / 기억에서 온 형상(관측되지 않은 voxel, 예: kept · stale 형상) / 거리장 생성 artefact(관측 점 없음) / 기타 | 출처별 row · voxel 비율, 장면 · 단계별 |
| U2 | 출처마다 해당 voxel과 GT 표면의 거리 분포, 그리고 카메라에서 보이는 면인지 | 중앙값 · p95 (mm), 보임 비율 |
| U3 | 미상 row가 가장 크게 침범한 row인 요청(사전 확인 A의 미상 22 요청)의 TrajOpt 수정량과 HOLD | 요청별 표 |
| U4 | 대조 성공 3 장면의 미상 row(10.7%)도 같은 분류로 | 출처별 비율 |

- 분류 규칙과 거리 허용 오차는 결과 전에 verify.json `definitions`에 적는다. 한 voxel이 여러 출처에 맞으면 ambiguous로 따로 센다
- figure: 실제 씬 1(미상 voxel을 출처별 색으로, 320 또는 1313) · 그래프 1 · 표 1 이상, `AG3S/ag3s/docs/figures/ifg-unknown/` + `.json` sidecar
- 수치: `AG3S/ag3s/docs/handoff/IFG-UNKNOWN.verify.json` (`numbers` · `definitions` · `per_scene` · `scripts` · `figures` · `notes`)
- script: `AG3S/ag3s/experiments/studies/ifg_unknown_source.py`
- 해석은 쓰지 않는다. CPU만, worker ≤ 2. 끝나면 process 종료 + reclaim 한 줄

## 이 STEP이 답하는 질문

실패 장면의 TrajOpt 제약을 가장 많이 켠 "미상" voxel은 무엇에서 왔는가.
