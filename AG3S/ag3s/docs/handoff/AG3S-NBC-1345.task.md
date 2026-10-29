# AG3S-NBC-1345 — 1345 에서 왼손 몸체가 오렌지에 닿는 원인 분리: 이웃 접촉 허용 vs 구 모델이 손바닥을 덮지 못함

> writer: lead · 실행: lead 직접 (순차 작업 — 결과가 다음 수정의 층을 정한다)

작성: 2026-10-25 17:2x KST (`date -u` 08:2x UTC + 9 h), lead. **조건 · 지표 · 판정은 결과를 보기 전에 고정한다.**
앞선 STEP: `AG3S-ROI-CARVE-GPU` (R4 진단, `AG3S_TSDF_DYNAMIC_LOG.md` §13.6). 사용자 판정 2026-10-25: "우선 요인을 분리해봅시다".

## 0. 출발점 (측정된 사실)

| 사실 | 출처 |
|---|---|
| 1345 (seed 13453) 에서 왼손 몸체 mesh (EE_BODY_L geom 1052) 가 오렌지에 닿는다: 27 control row, 최대 겹침 0.57 mm, 요청 12–15 구간. VLA 단독 · 모든 AG3S 구성에서 닿는다 | `AG3S-ROI-CARVE-GPU.verify.json` R4 |
| 거리장은 오렌지를 갖고 있다 (요청 11–16 main 층 보이는 표면 놓침 0 %) → 거리장 층 (LTE) 이 원인 아님 | 같은 곳 |
| 그 요청들에서 이웃 접촉 허용 (`--neighbour-contact-mm 20`, 손가락 구가 이웃 장애물과 20 mm 까지 겹쳐도 됨) 이 켜져 있었고, 가장 가까운 손가락 구 (`ee_finger_l1`) 의 계획 여유가 음수였다 | 같은 곳 |
| **설계 확인 (offline, rollout 아님)** `sphere_coverage_check.py`: 지금 구 모델은 접촉 순간 손이 오렌지에서 10.2–13.3 mm 떨어져 있다고 본다 (실제 0.8–1.9 mm). 손바닥 link (`ee_left`) 의 capsule 배율을 1.0 (URDF 굵기, 구 반지름 33.5 mm) 으로 하면 −1.4 ~ −2.3 mm 로 접촉을 본다. 배율 0.35 는 바뀌지 않는다 (가장 가까운 구가 손가락 구 그대로) | `$RUN/verify/AG3S-NBC-1345/coverage_check.json` |

## 1. 조건 (1345 · seed 13453 · E4 serving 구성 + GPU 판정 위에서 하나씩 바꾼다)

| 이름 | 바꾼 것 | 무엇을 묻나 |
|---|---|---|
| 기준 (이미 있음) | — (`AG3S-ROI-CARVE-GPU-SMOKE` 의 B_ROI_FPR run, E4 와 bit 동일) | 27 row 접촉 |
| 허용 끔 (`NBC0`) | `--neighbour-contact-mm 20` 을 뺀다 | 허용만 없으면 TrajOpt 가 손을 오렌지에서 떼는가 |
| 손바닥 굵게 (`PALM`) | `--capsule-radius-scale-link ee_left=1.0 ee_right=1.0` | 구 모델이 손바닥을 덮기만 하면 떼는가 (허용은 손가락 구에만 걸리므로 손바닥 구에는 적용되지 않는다) |
| 둘 다 (`NBC0_PALM`) | 위 둘 | 둘이 함께 있어야만 떼는가 |

같은 snapshot (`snapshot_gpu`), 같은 client 설정 (`--stop-after-place 2.5`). 1 장면 진단이다 — 일반화 · 우수성 주장은 하지 않는다.

## 2. 지표와 판정 (결과 전 고정)

- 주 지표: MuJoCo replay 의 **손 – 이웃 과일 접촉 row 수** (오렌지), 그리고 모든 금지 접촉, 성공 (사과가 바구니에).
- 보조: 접촉 구간의 실제 궤적 (measured) 에서 손 – 오렌지 최소 거리.

| 결과 | 판정 (원인 층) |
|---|---|
| 허용 끔만으로 0 row, 손바닥 굵게만으로는 남음 | 원인 = 이웃 접촉 허용 (CCR, TrajOpt 설정) |
| 손바닥 굵게만으로 0 row, 허용 끔만으로는 남음 | 원인 = 구 모델이 손바닥을 덮지 못함 (제약 기하 — 로봇 구 모델) |
| 둘 다 있어야 0 row | 두 요인이 함께 필요 |
| 어느 하나만으로도 0 row | 어느 쪽도 충분 — 고칠 층은 부작용 (성공 · 다른 접촉) 으로 고른다 |
| 둘 다 해도 남음 | 다른 원인 (명령 ≠ 실제 궤적, chunk 경계 등) — 실제 궤적으로 다시 진단 |

성공을 잃거나 다른 금지 접촉이 생기면 그 사실을 함께 적는다 (판정을 바꾸지 않는다). 이 STEP 은 원인 분리까지이고, 고치는 것은 결과를 보고 사용자 판정으로 한다.

## 3. 산출물

- 원시: `$RUN/verify/AG3S-NBC-1345/` (run `20261025T035831+0900_8da37119afd3_f082f20e9463`)
- `AG3S/ag3s/docs/handoff/AG3S-NBC-1345.verify.json`, 그림 `AG3S/ag3s/docs/figures/nbc-1345/`
- 로그: `AG3S/ag3s/docs/AG3S_TSDF_DYNAMIC_LOG.md` (§13.6 의 후속)
- 보기: `outputs/2026-10-25_1345_오렌지접촉_원인분리/`
