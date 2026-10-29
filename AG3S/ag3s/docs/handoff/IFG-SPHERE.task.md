# IFG-SPHERE — 순서 1: target 제외 영역 · 기억의 형상 (1단계: 남은 target voxel의 원인 분해와 설계안)

> writer: lead · 읽는 쪽: implementer

작성: 2026-10-22 KST, lead. 근거: WORKLOG_S0S3 2026-10-21 23:30 · 23:57 절, 2026-10-22 00:36 절(IFG-PRECHECK),
사용자 판정 2026-10-22 "미상 확인 + 순서 1 병행".

## 왜 지금 이것인가

- 관측 점검: 닫는 중 target 제외 recall 0.59, 손가락 거리장(target 제외 층)에 사과 voxel 중앙값 517개 (접근 중 251).
  추적 · 기억 중심이 보이는 면 쪽으로 17.9 mm 치우치고 가장 좁은 축을 11 mm 작게 잡는다 (`IFG-AUDIT-ATS/LTE.verify.json`)
- 사전 확인 A: 실패 6 장면의 쥐기 전 활성 제약 중 target 출처 18.8%. 장면별 211 29% · 979 88% · 320 23%.
  **211 k11–14에서 가장 크게 침범한 row가 target이고 HOLD(k11 · k12)를 만든다. SPLIT · NBC 둘 다 같다** (`IFG-PRECHECK-A.verify.json`)
- 확인한 사실: 제외 공(`TargetBall`)의 중심은 이미 표면 구 맞춤(`centre_method == "surface_fit"`, `pipeline.py:3618`)이다.
  LTE 점검에서 맞춤 중심 오차는 stale 아닌 요청에서 중앙값 2.01 mm, 맞춤 반지름 33.5 mm, 공 반지름 54.7 mm.
  그러니 "평균 대신 구 맞춤" 만으로는 recall 0.59를 설명하지 못한다. 공 안에서 voxel을 **남기는** 다른 판단
  (T44 `ball_owned` 소유 판정 · ambiguous keep, `surface_protect_split`, `nontarget_surface_evidence` 의 반지름 제한,
  `z_min` 지원면 제한, 거친 층 `coarse_exclusion_params`, stale 기억으로 공이 옛 자리에 있음 등)이 원인일 수 있다

## 1단계 요청 (→ implementer, 코드 수정 없음 — 분석 script와 설계안만)

1. `IFG-APPLE-SPLIT` 기록(run `20261019T132719+0900_f15e587290f6_11408fd6d27f/verify/IFG-APPLE-SPLIT`, 24 장면, 서버 constraint record)을
   서버와 같은 flag로 offline 재구성해서(IFG-SPLIT · IFG-AUDIT-LTE의 replay 방식 재사용) target 제외 층에 **남은 GT target voxel 각각이
   어느 판단 때문에 남았는지** 분해한다. 분류는 코드 경로 기준으로 상호 배타적으로 정하고 결과 전에 적는다.
   예: 공 밖(반지름 · 중심) / 공 안이지만 `z_min` 아래 / 소유 판정 keep(ambiguous · non-target 쪽) / 표면 보호 / 반지름 제한 /
   stale 공(중심이 GT에서 30 mm 넘게 떨어짐) / 거친 층에서만 남음 / 기타
2. 단계별(latched · closing) · 장면 유형별(이웃 있음 / 없음)로 표를 낸다. 사전 확인 A의 **target 출처 활성 row**(특히 211 k11–14, 979, 320)에
   닿은 voxel은 따로 표로 낸다
3. 추적 · 기억 중심(17.9 mm 치우침)이 실제로 어디에 쓰이는지(제외 공, 손 구 예외 반경, HSR, TrajOpt target 등) 호출 경로를 정리한다
4. 위 결과로 **설계안 1–2개**를 낸다. 각 안마다 바꿀 파일:줄, flag 이름(기본 꺼짐), 기대 효과를 1단계 수치로 추정한 값,
   위험(이웃 과일 오제거 — 현재 1.9–2.3%, 탁자 구멍)을 적는다

- script: `AG3S/ag3s/experiments/studies/ifg_sphere_attrib.py`. 출력: 새 run(`python -m common.run_output`)의 `verify/IFG-SPHERE-ATTRIB/`
- 결과: `AG3S/ag3s/docs/handoff/IFG-SPHERE.impl.md` (분해 표 · 호출 경로 · 설계안)
- 손대지 않을 것: 서빙 코드(`AG3S/ag3s/**`, `AG3S/trajopt/**`의 기존 module) — 1단계는 분석만. 설계안은 사용자 판정 뒤 2단계에서 구현한다
- CPU만. worker ≤ 2. 끝나면 process 종료 + `python3 scripts/setup/reclaim_cache.py "$AG3S_RUN_DIR" <읽은 큰 입력…>` 한 줄

## 이 STEP이 답하는 질문 (1단계)

남은 target voxel은 주로 어느 판단 때문에 남는가, 그리고 그것을 고치는 가장 작은 수정은 무엇인가.

## 2단계 (사용자 판정 뒤, 미리 정한 검증)

- 구현은 flag 기본 꺼짐, 꺼짐이면 출력 불변(기존 test 통과)
- offline: SPLIT 24 장면에서 제외 precision · recall, 남은 target voxel, 이웃 과일 오제거, target 출처 활성 row 수와 HOLD (전 / 후)
- closed-loop: VLA 단독 성공 사과 10 장면, 기준 = 이웃 접촉 20 mm 구성(9 / 10)과 같은 seed 짝 비교
