# 관측 모듈 점검 (ATS · HSR · LTE) — 그림 모음

작성: 2026-10-21 KST

- **데이터**: `IFG-APPLE-SPLIT` 기록이다. 사과 24 장면, 지금 serving 구성(기존 방식 + target 분리)으로 돌렸고, 요청 1,800개를 전부 썼다
- **GT**: MuJoCo 상태에서 가져왔다
  - 각 요청의 planning row qpos로 장면을 다시 만들었다
  - 거리는 충돌 geom까지의 거리다. 상자는 해석식으로, mesh는 `mj_geomDistance`(convex hull)로 계산했다
- **수치 원본**
  - `AG3S/ag3s/docs/handoff/IFG-AUDIT-ATS.verify.json`
  - `IFG-AUDIT-HSR.verify.json`
  - `IFG-AUDIT-LTE.verify.json`
  - 그림마다 같은 이름의 `.json` sidecar가 있다
- **스크립트** (`AG3S/ag3s/experiments/studies/`): `ifg_audit_ats.py`, `ifg_audit_hsr.py`, `ifg_audit_lte.py`, `ifg_audit_gallery.py`

## 실제 씬 (3rd person, 녹화 카메라와 같은 시점)

| 파일 | 보이는 것 |
|---|---|
| `gallery-esdf-ep{250,954,211,56}.png` | **거리장 대 실제 씬.** 왼쪽은 사과 중심 높이(운반 중에는 바구니 벽 높이) 수평 단면의 ESDF를 같은 카메라로 본 것이다. 잡기 전에는 손가락이 보는 target 제외 층을 그렸다. 오른쪽은 같은 순간의 실제 씬이다. 선: 검정 = 거리장 0, 회색 점선 = 10 mm, 흰 점선 = GT 물체 경계 |
| `gallery-sequence-ep{250,954,211,56}.png` | **분할 · 기억 · 단계 · 링크를 시간 순서로 캡처.** 보라 = VLA attention(≥ 0.5), 초록 = 선택한 target 덩어리 · 추적 중심(x) · 예외 반경, 파랑 = 사과에 닿아도 되는 손 구, 회색 = 다른 손 구. 아래 그래프는 요청별 상위 attention 비율, 분할 precision, 단계 띠(GT / HSR latch / VLA probe)다 |
| `ifg-audit-ats-scene.png` | 분할 장면 하나(ep1345 요청 8)를 위와 옆에서 본 것. target 덩어리 · GT 윤곽 · attention |
| `ifg-audit-lte-section.png` | 거리장 단면(ep119 요청 15). main 층과 target 제외 층, GT 윤곽, 지운 voxel, 손 구 |
| `gallery-links-table.png` | 단계별 링크 처리 표 |

장면 선택:
- 250: 바나나 옆, 성공
- 954: 바나나 옆, 빈손 실패
- 211: 바나나 옆, 실패
- 56: 이웃 없음, 성공

## 그래프

| 파일 | 내용 |
|---|---|
| `ifg-audit-ats-phase.png` | 단계별 분할 precision · recall · IoU · identity |
| `ifg-audit-ats-attention-memory.png` | 요청별 물체별 attention 비율과 추적 target의 정답 여부(ep1345 · 1837 · 954) |
| `ifg_audit_hsr_timelines.png` | GT 단계 대 HSR latch 대 VLA probe(6 장면, 실패 3 포함) |
| `ifg_audit_hsr_confusion.png` | latch · probe confusion matrix |
| `ifg_audit_hsr_latency.png` | 단계 전환 지연 분포 |
| `ifg-audit-lte-error-bands.png` | 거리장 − GT 오차를 GT 거리 구간별 · 층별로 |
| `ifg-audit-lte-error-strata.png` | 같은 오차를 단계 · 장면 유형별로 |
| `ifg-audit-lte-exemption.png` | target 제외의 precision · recall · 남은 target voxel |
| `ifg-audit-lte-geometry.png` | target 크기 추정과 쥔 사과 위치 오차 |

## 핵심 수치

| 능력 | 지표 | 값 |
|---|---|---|
| **분할 (ATS)** | 접근 중 precision · IoU(중앙값) | 1.000 · 0.927 |
| | 닫는 중 IoU(중앙값): 전체 / 이웃 있음 / 이웃 없음 | 0.362 / 0.197 / 0.719 |
| | 보이는 요청(549) / 가려진 요청(444)의 precision | 1.000 / 0.000 |
| | target 중심 오차(접근 중) | 17.9 mm (보이는 표면 쪽으로 치우침) |
| | 가장 좁은 축 크기, 추정 대 GT | 50.1 대 61.0 mm |
| **작업 단계 (HSR latch)** | 단계 정확도(GT 5단계) | 0.965 |
| | 쥔 상태 recall · precision | 1.000 · 0.813 |
| | 쥔 동안 잘못 철회 / 다른 대상으로 바뀜 | 0 / 0 |
| | 놓음 판정 지연(중앙값) | 1 요청 (0.93 s) |
| **작업 단계 (VLA probe)** | label 정확도 | 0.843 |
| | 빈손 실패 5 run 중 "place"로 넘어간 run | 4 |
| | home 판정 지연(중앙값 / 최대) | 2.17 s / 5.67 s |
| **attention이 바뀌어도 기억** | 사과가 attention 1위가 아닌 요청 | 1,358 / 1,800 (1위는 대부분 바구니, 1,336) |
| | 그중 추적 target이 있는 568 요청: 맞음 / 옛 위치(stale) / 다른 물체로 바뀜 | 266 / 302 / 0 |
| | 가려진 동안 기억한 위치 | 마지막으로 본 world 좌표에 고정된다(운반 중 사과가 250.8 mm 이동) |
| **3D 기하 (LTE)** | target 제외 precision / recall: 접근 · 닫는 중 | 0.77 / 0.79 · 0.67 / 0.59 |
| | target 제외 층에 남은 사과 voxel(중앙값): 접근 · 닫는 중 | 251 · 517 |
| | 이웃 과일을 잘못 지운 비율 | 1.9–2.3% |
| | 쥔 사과 구 중심 오차(중앙값 / p95) | 9.5 / 18.2 mm |
| **거리장 대 GT** | 고운 층(5 mm), 5 mm 넘게 멀게 봄(위험) | 0.06% |
| | 거친 층(20 mm), 오차 중앙값 · 5 mm 넘게 멀게 봄 | +6.0 mm · 55% |
| | 손가락(target 제외 층), 10 mm 넘게 가깝게 봄(보수) | 34% |
| **링크 처리** | 충돌 모델 | 손 링크 6개의 구 270개뿐이다. 팔 · 손목 · 팔꿈치는 검사하지 않는다 |

## 두 "attention 비율"의 정의 차이

- `ifg-audit-ats-*`: 물체별 attention 합을 라벨이 붙은 물체들 사이에서 정규화한 값이다. 접근 중 사과의 비율 중앙값은 0.63이다
- `gallery-sequence-*`: attention ≥ 0.5인 점 중 사과 위(GT 구 + 10 mm)에 있는 점의 비율이다. 탁자 · 바나나 위의 강한 점까지 분모에 들어가므로 약 0.2로 낮다

## 주의

- 가려진 면: 카메라에 보이지 않는 면에서는 거리장이 물체를 가깝게(보수적으로) 본다
- 실행 중 추가된 장애물: 954의 바구니 기둥처럼 실행 중에 추가된 장애물은 gallery GT에 없다
- GT 정의: HSR의 GT 단계(쥠 = 1 cm 이상 들림 + 손 접촉 5 step 등)와 ATS · LTE의 stale 기준(30 mm)은 이번 점검에서 정한 것이다

## 개선할 부분 (관측 쪽, 2026-10-21 제안 — 사용자 판정 전)

| 순서 | 문제 (근거) | 구성 요소 | 제안 |
|---|---|---|---|
| 1 | 가려진 동안 target 기억이 마지막으로 본 world 위치에 멈춘다. 기억이 있는 568 요청 중 302 요청이 옛 위치이고, 닫는 중 요청의 절반(99 / 206)이 이 상태다. 사과가 떨어지거나 밀려도(ep320 · 211) 옛 자리를 지운다 | ATS · LTE | 손이 닿은 뒤에는 기억한 형상을 손 기준으로 옮긴다. 비어 있는 것이 관측된 옛 자리는 기억을 무효화한다 |
| 2 | target 중심이 보이는 면 쪽으로 17.9 mm 치우치고 좁은 축을 11 mm 작게 잡는다. 사과 제외 recall은 접근 0.79 · 닫는 중 0.59이고, 손가락 거리장에 사과 voxel이 251 → 517개 남는다 | ATS → LTE | 보이는 점의 평균 대신 표면에 맞춘 구(중심 오차 중앙값 2.0 mm)로 제외 영역과 기억을 잡는다. 1번과 원인이 같아 함께 진행을 권한다 |
| 3 | 20 mm 거친 층이 거리를 과대평가한다(중앙값 +6.0 mm, 55%가 5 mm 넘게 멀게 봄; 바구니 쪽). TrajOpt 여유가 0–10 mm라 위험한 방향이다 | LTE | 고운 창을 바구니까지 넓히거나, 거친 층 값에서 voxel 절반(10 mm)을 뺀다 |
| 4 | VLA probe가 빈손 실패 5 run 중 4 run에서 쥐지 않았는데 "place"로 넘어간다. latch에는 파지 실패 상태가 없다 | HSR | probe와 latch를 묶는다. "쥠 아님 + probe place" 모순을 빈손 신호로 써서 닫는 순간 감지한다 |
| 5 | 충돌 검사 대상이 손 링크 6개(구 270개)뿐이다. 팔 · 손목 · 팔꿈치는 검사하지 않는다 | 링크 설정 | 관측 범위 밖이라 기록만 한다 |
