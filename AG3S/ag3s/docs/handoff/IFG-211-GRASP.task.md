# IFG-211-GRASP — 211 장면: 사과 17 mm 앞에서 닫고도 쥐지 못한 원인 (CPU offline 진단)

> writer: lead · 읽는 쪽: verifier

작성: 2026-10-22 KST, lead. 결과를 보기 전에 고정한다. 사용자 판정 2026-10-22: "211 쥐기 실패 진단",
"새 구성을 비교 기준으로" (= 지금 최선 구성 + 장애물 근거에서 사과 표면 점 빼기 + 놓은 뒤 target을 못 찾아도 안전 인증).

## 무슨 실험인가

- **질문:** 새 비교 기준 구성에서 211은 쥐기 전 멈춤 없이 다가가 t = 233에 닫기 시작했다(손끝–사과 17.2 mm). 그 뒤 손바닥–사과 약 107 mm가 유지되고 쥠이 일어나지 않았다(실패 양상 no_grasp). 무엇 때문에 쥐지 못했나
- **데이터**
  - 새 구성: run root `20261019T132719+0900_f15e587290f6_11408fd6d27f`의 `verify/IFG-APPLE-NBC-TARGETSURFACE` (211 run, 서버 constraint record, client `traj.npz`의 measured_qpos · 명령 · 그리퍼)
  - 비교 1: 같은 root의 `verify/IFG-APPLE-NBC` 211 (이전 최선 구성, 실패: 75 mm 앞에서 닫음)
  - 비교 2: 같은 장면의 VLA 단독 성공 기록(`IFG-APPLE` E0 211, 닫는 t ≈ 184 근처)
- **재는 것은 실제 궤적 기준이다** (명령 ≠ 실제, MuJoCo kp · damping)

## 측정 (→ verifier)

| # | 무엇을 재나 | 어떤 숫자로 |
|---|---|---|
| G1 | 닫기 시작 전후 t = 200–300의 손끝(두 손가락) · 손바닥 위치를 사과 기준 좌표(앞 · 옆 · 위 mm)로, 세 기록 나란히 | 시계열 표 · 그래프 |
| G2 | 같은 구간의 MuJoCo 실제 접촉: 손가락–사과, 손가락–바나나, 손–탁자, 그 밖. 처음 닿은 t와 상대 | 접촉 timeline |
| G3 | 그리퍼 명령 대 실제 개도, 손가락이 멈춘 t와 그때 막힌 상대 | 시계열 |
| G4 | 사과 GT 위치 변화(밀림 · 굴러감 · 떨어짐) | mm, t별 |
| G5 | 서버 쪽: 요청별 VLA chunk 대 TrajOpt 출력의 손끝 수정량, HOLD, 쥠 판단 상태(latched · closing · held · 철회)와 그 근거 기록 | 요청별 표 |
| G6 | VLA 계획 자체가 닫는 위치를 어디로 계획했는가(TrajOpt 전 chunk의 닫는 행 손끝, 사과 기준) — 세 기록 비교 | 요청별 표 |

- 원인 후보는 이렇게 나누어 각각 숫자로 판정한다: (a) 손가락이 바나나 또는 탁자에 먼저 닿아 사과를 감싸지 못함 (b) 사과가 밀려 손가락 사이에서 빠짐 (c) VLA가 계획한 닫는 위치 자체가 어긋남 (d) TrajOpt가 닫는 동안 손을 옮김 (e) 쥠 판단이 쥔 상태를 인정하지 않아 이후 동작이 바뀜 (f) 그 밖. 판정 기준(예: 거리 · 접촉 · 수정량 임계)은 결과 전에 verify.json `definitions`에 적는다
- figure: 실제 씬 1(닫는 순간 세 기록 나란히, 녹화 카메라 시점) · 그래프 1(G1 · G3 시계열) · 표 1(G5 · G6), `AG3S/ag3s/docs/figures/ifg-split/ifg-211-grasp-*` + `.json` sidecar
- 수치: `AG3S/ag3s/docs/handoff/IFG-211-GRASP.verify.json`. script: `AG3S/ag3s/experiments/studies/ifg_211_grasp.py`
- 해석 문장은 쓰지 않는다. 원인 후보별 판정 값만 쓴다. CPU만, worker ≤ 2. 끝나면 process 종료 + reclaim 한 줄
- 보고와 그림에는 구성 기호 대신 설명 이름을 쓴다("새 비교 기준 구성", "이전 최선 구성", "VLA 단독")

## 이 STEP이 답하는 질문

새 비교 기준 구성에서 211이 쥐지 못한 직접 원인은 (a)–(f) 중 무엇이고, 그것은 ATS(target 분할) · HSR(작업 단계 · 쥠 판단) · LTE(3D 기하 · 거리장) · CCR(TrajOpt 궤적 수정) 중 어디에 있는가.
