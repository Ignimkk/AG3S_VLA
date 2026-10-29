# AG3S 핵심 연결 평가와 개발 종료 기준

작성: 2026-10-23 15:04 KST · writer: lead.
상태: **설계 결정 완료 / 새 구성 미실행 / 실행 manifest 미완성**.
기준 코드: `8da37119afd3b8ae1aaaa09315066c34c880857d`, `feature/ag3s-development`.
이 문서는 [한 페이지 연구 범위](../../../../doc/active/plans/AG3S_RESEARCH_SCOPE_20261023_KO.md)의 상세 실행 설계다.
기존 계획 §0.10의 identity·extent·contact 분리와 closed-loop 질문을 구체화한다.
사용자가 이번 세션에서 연구 범위와 평가 구성의 선정을 요청했다. 아래 예산은 설계상 상한이며 이번 문서 작성에서 GPU 실행·코드 수정·커밋은 하지 않았다.

## 0. 사용자 판정 (2026-10-25 19:00 KST) — 아래 본문보다 우선한다

동적 거리장 축 종료(`AG3S_TSDF_DYNAMIC_LOG.md` §14–15) 뒤 남은 결정에 대한 사용자 답이다. 본문과 다르면 이 절을 따른다. 본문 반영은 CPU 평가 준비 때 한다.

| # | 항목 | 판정 | 본문에 미치는 영향 |
|---|---|---|---|
| 1 | 작업 순서 | 남은 느린 단계 세 개(target 찾기 · 장면 복원 · TrajOpt 풀이)를 **평가 전에 모두** GPU로 옮긴다 | 기준 코드 `8da37119`는 낡았다. 새 기준 commit은 GPU 변환을 마친 뒤 정한다. TrajOpt 결과가 바뀌면 과거 9/10 · 7/10과 직접 비교하지 않는다 |
| 2 | 주 지표 | **금지 접촉 없는 성공 하나**를 주 지표로 쓴다 | §6.1의 세 조건 대신 `Δsafe` 하나로 판정한다(점추정 +0.05 이상, scene 단위 95% CI 하한 > 0). 작업 성공률과 금지 접촉률은 같은 표에 보조 지표로 싣는다 |
| 3 | 층 | 장면 결과를 **장애물 있음 / 없음**으로도 나눠 보고한다 | §5.1의 네 층은 유지하고, 그 위에 두 묶음을 더한다 |
| 4 | 같은 입력 진단 | §3.2의 진단(의미 · 경계 · 권한 · 정답 표면)을 **한 표**로 합친다 | §8 최종 산출물의 “같은 입력 분석표 1개”와 같다 |
| 5 | 9/10 구성 참고 줄 | **채택하지 않는다** | pilot은 세 구성 그대로다 |
| 6 | 평가 목적 | “지금까지 개선한 AG3S가 연구 목표에 비춰 어느 정도 성능을 내는가”를 평가한다. **모듈별(ATS · HSR · LTE · CCR) 성능을 각자의 지표로** 나타낸다 | 모듈별 지표 정의는 CPU 준비에서 제안하고 사용자 판정을 받는다 |
| 7 | 마감 | 사용자가 관리한다 | 이 문서에는 날짜 기준을 두지 않는다 |

### 0.1 모듈별 핵심 평가 1차 — 사용자 판정 2026-10-28 22:50 KST (결과 전 고정, 아래 본문 §2–§5보다 우선)

사용자 지시: "AG3S 핵심 모듈을 평가. 타겟 위 장애물은 하지 말 것. 평가할 때 각 모듈 · 각 기능별로 시각화 자료를 기록." 질문 답: ① 제외 = 타겟 위 막대(OCC 케이스 3 overbar)만, 장애물 없는 회귀 장면은 포함 ② 구성 = 채택 A1–A4 + A7 ③ 장면 = 기존 장면 + 기둥 1개만(기둥 두 개 장면 제외) ④ 모듈 평가 = 기록에서 offline 측정(추가 모듈 끔 run 없음).

| 항목 | 고정 내용 |
|---|---|
| 평가 구성 (AG3S) | [채택 문서](../../../../doc/active/reviews/AG3S_decision/ADOPTED_KO.md) A1–A4 + A7 = 지금 serving(E3 kit `B_ROI_FPR`: GPU 초기값 · BLAS 1 · 관심 영역 지우기 GPU) + `--palm-shell on --palm-shell-margin 5` + `--grasp-anchor 2000 --grasp-anchor-path-weight 0`(닫는 행만, `--grasp-goal-window close` 기본). 렌더 mask 없음(구 mask만). 꼭지 덮개 · 이웃 접촉 허용 쥐기 · 팔 구는 넣지 않는다(한계로 적는다) |
| 코드 | 새 run `20261028T224856+0900_b3b69311f948_8784a1265958`, snapshot `snapshot_core` = HEAD `b3b69311` + 작업 트리(AG3S · tests · common · scripts · robots · Baseline · integration), 차이 파일 md5 `snapshot_core_head.txt`. commit은 사용자 요청 때 |
| 장면 | 가림 기둥 1개: 장면 1345 오렌지 제거 + 케이스 1 L2 기둥(head 가림) × seed 13453 · 23453 · 33453. 장애물 없음(회귀): 119 · 250 · 320 · 850 · 954 · 211(seed = 장면×10+3, 기존 place 장애물 설정 그대로). 둘 다 이미 쓴 개발 장면이다 — 독립 test가 아니다 |
| 반복 | 장면마다 2번: 흔들림 없음 + 첫 chunk 1e-6 rad 흔들림 seed 1 (PT와 같은 방식) = AG3S 18 run |
| 비교 (다시 돌리지 않음) | 같은 장면 · 같은 흔들림의 기존 기록: VLA 단독(PT), 지금 serving(E3, A1–A2), 껍질 구(SCM, A1–A3) |
| 주 지표 | 금지 접촉 없는 성공 (PRE-CORE-TO 2026-10-27 08:50 정의: 금지 = 이웃 과일 · 기둥 · 권한 없는 link–사과, 테이블 · 바구니 접촉은 따로 센다). 장애물 있음 / 없음으로 나눠 보고 |
| 읽는 규칙 | 2번 반복이라 구성 효과를 판정하지 않는다. 장면 · 흔들림 짝으로 바뀐 결과와 합계(Wilson 95 %)만 보고. 실패마다 ATS · HSR · LTE · CCR 원인 위치 |

**모듈별 지표 (offline, AG3S 서버 기록 + MuJoCo 재생 정답, 요청 단위):**

| 모듈 · 기능 | 지표 | 시각화 (실제 씬 · 그래프 · 표) |
|---|---|---|
| ATS — target 찾기 (attention → 표면) | target 점 precision(정답 사과 표면 5 mm 안 비율), target 중심 오차(mm, 정답 사과 중심 대비), target 없음 요청 수 | head 영상 위 noun attention heatmap + target 점 + 정답 사과 윤곽; 3D 장면에 target 점 · 공 |
| ATS — 시간 추적 (identity 유지) | target이 다른 물체로 바뀐 요청 수, 쥐기 전 구간 중심 오차 시간축 | 시간축 그래프 |
| HSR — 작업 단계 (pick / place) | 정답 단계(사과가 손에 들려 탁자 위 2 cm 이상) 대비 일치율, 전환 지연(요청 수) | 시간축 띠 (정답 대 예측) + 전환 순간 실제 씬 |
| HSR — 쥠 상태 (free / held) | 정답 쥠(두 손가락 접촉 + 들림) 대비 일치율, held 인정 지연, 잘못된 held 수 | 같은 시간축 띠 |
| LTE — target 제외 거리장 | 쥐는 손 층에서 남은 사과 voxel 비율, 잘못 지운 voxel 수와 class(이웃 과일 · 장애물 · 탁자 · 로봇, 정답 8 mm 최근접) | 쥐기 직전 요청의 거리장 voxel을 class 색으로 칠한 실제 씬 + 지운 voxel |
| LTE — 링크별 거리 (구 모델) | 손바닥 껍질 행 · 손가락 행 · 손목 행별 최소 여유(mm), 위반 행 수 | 구를 여유 색으로 칠한 실제 씬, 링크 묶음별 여유 그래프 |
| CCR — chunk 수정 | 실제 MuJoCo 금지 접촉(run · 행 · 최대 겹침), VLA 대비 수정량(관절 rad · 손 위치 mm), HOLD 요청 비율, 쥐기 목표 항 켜진 요청 수 | VLA 원래 궤적 대 수정 궤적 실제 씬, 요청별 수정량 · 최소 여유 그래프 |
| CCR — 시간 | TrajOpt 풀이 · 요청 처리 시간 중앙값 · p95 (533 ms 대비) | 분포 그래프 |
| 전체 | 금지 접촉 없는 성공, 작업 성공, 쥐기 성공 | 영상(`common.experiment_view`) + 결과 표 |

**결과 (2026-10-29 00:55 KST, lead):** 18 run 완료. outputs `outputs/2026-10-28_핵심평가_모듈별_가림기둥3_회귀6_x2/README.md`, 수치 `AG3S-CORE-EVALUATION.verify.json`, 그림 `figures/core-eval/` (논문 후보 `figures/core-eval/paper/`). 금지 접촉 없는 성공: 가림 기둥 5/6 (VLA 6/6), 장애물 없음 7/12 (VLA 9/12, 탁자 · 바구니 접촉을 금지로 넣으면 VLA 4/12). 로봇–탁자 · 바구니 접촉 run VLA 5/12 → 0/12. 사용자 판정 대기: 이 결과로 방법론 채택 여부.

**사용자 판정 (2026-10-29 01:05 KST):**
1. 지금 AG3S (A1–A4 + 구 mask) 를 **논문 방법론으로 채택**.
2. B (f7) 요청의 target attention 신호 3 가지 비교를 **ATS ablation** 으로 돌린다.
   - 명사 `noun_target:L10H7` = 1차 기록 재사용. 새로 돌리는 것: 이전 기본 head `action:last:L8H2`, T61 선택 head `action:mean:L10H7`. 각 9 장면 × 2 = 36 run.
   - 바꾸는 기준 (결과 전 고정): 접촉 없는 성공 18 중 **+4 이상** ∧ 금지 접촉 run 이 늘지 않음 ∧ ATS 지표 (정확도 · 보이는 표면 덮은 비율 · 다른 물체 선택) 가 나빠지지 않음. 아니면 명사 유지.
   - kit: `verify/AG3S-CORE-EVAL/AS_{L8H2,T61}_k{0,1}`, `OCC_AS_{L8H2,T61}`, `as_launch.sh`. T57 단계 1 을 현 구성 안에서 대체한다 (장면 세트가 T57 의 16 episode 와 다르다).
   - **판정 규칙 교체 (사용자 판정 2026-10-29 01:50 KST, B 제안 + 장면 일관성, 결과 전):** 위 "+4 run" 규칙을 대체한다.
     - 주 지표: **고정 입력 ATS 지표.** 1차 기록 (명사 구성) 의 같은 frame 에서 attention 신호만 바꿔 재생한다.
       - target identity 정답률
       - target 점 정확도 · 보이는 표면 덮은 비율
       - 다른 물체 선택 비율
       - 거리장에서 잘못 지운 장애물 voxel · 남은 target voxel
     - 바꾸는 조건 (모두 만족):
       - identity · 정확도 · 다른 물체 선택에서 각각 **2 %p 이상 개선**, 나머지 지표 악화 없음.
       - 그 개선이 **9 장면 중 5 장면 이상에서 같은 방향**.
       - 안전 확인 통과: closed-loop 36 run 에서 접촉 없는 성공이 명사보다 4 run 이상 낮지 않고, 금지 접촉 run 이 늘지 않음.
     - 아니면 명사를 유지한다. 실패 run 마다 원인 층을 적고, 실패 시점에 ATS 지표가 틀렸을 때만 ATS 실패로 센다.
     - 고정 입력 재생은 **B (f7)** 가 한다 (GPU 공유 lock, 사전 고정 task 문서 먼저). A 는 기록 경로와 서버 argv 를 넘긴다.
   - **결과 (2026-10-29 ~05:00 KST): 명사 신호 유지.** B 의 T65 고정 입력 재생 (`T65.verify.json`, outputs `2026-10-29_ATS_attention_3way_fixed_input/`).
     - 두 대안 모두 규칙 1–3 미통과:
       - 9 장면 평균 정확도 0.853 / 0.802 / 0.844, 보이는 표면 덮은 비율 0.637 / 0.598 / 0.644 (명사 / L8H2 / L10H7).
       - L8H2 는 개선 없이 5 지표가 악화, L10H7 은 주 지표 개선 없음.
     - 7 / 9 장면은 세 신호의 결과가 같다. 차이는 211 · 250 에서만 난다.
     - 안전 확인 (closed-loop 36 run) 은 둘 다 통과했지만 바꾸는 근거가 아니다.
3. 다음 장면: **놓는 곳 근처 장애물** (VLA 단독이 실제로 부딪히는 장면). 배치 기준은 결과 전에 정해 사용자에게 묻는다.

### 0.2 놓는 곳 허들 장면 — 사용자 판정 2026-10-29 01:20 KST (결과 전 고정)

사용자 답: "이전 placement 과정에서 사용한 바구니 위 허들로 수행", 장면 = 이번 평가 7 장면.

| 항목 | 고정 내용 |
|---|---|
| 장애물 | T43 P2 의 바구니 위 허들 (overbar_0, 바구니 짧은 축을 가로지르는 막대 + 다리 2, 손잡이 안쪽, 사과 쪽 끝벽 가까이). 기존 바구니 옆 기둥 (`ep*_place`, 로봇에서 약 0.21 m) 은 이 허들로 바꾼다. 1345 는 가림 기둥 (case1 L2) 을 그대로 두고 허들을 더한다 |
| 배치 탐색 | T43 P2 를 snapshot_core 로 옮긴 `verify/AG3S-CORE-EVAL/hurdle/hurdle_search.py`. 격자 · 기준 · 순위는 T43 P2 그대로: a' 시작 자세 로봇 ≥ 30 mm, b 바구니 · 과일 · 사과와 안 겹침 + 다리 바구니 밖, **c VLA 단독 기준 궤적 2 개 (흔들림 없음 · 흔들림 1) 모두 운반 중 손 또는 사과가 허들에 닿음**, d 쥐기 전 경로 ≥ 30 mm, e 허들 위 여유 높이 ≥ 5 cm. 바뀐 점은 파일 머리말 CHANGES (기준 궤적 · 시작 상태 · 사과 바닥 계산 · 1345 기둥 · lift/place 시점) |
| 장면 | 회귀 119 · 250 · 320 · 850 · 954 · 211 + 1345 (seed 13453). 통과 후보가 없는 장면은 빼고 그 사실을 보고한다 |
| 실행 | 장면마다 VLA 단독 × 2 + AG3S 채택 구성 × 2 (흔들림 없음 · 흔들림 1) = 28 run |
| 주 지표 | 금지 접촉 없는 성공 (§0.1 정의, 허들 접촉은 장애물 접촉 = 금지). 보조: 허들 접촉 run 수 · 행 수 · 최대 겹침, 탁자 · 바구니 접촉, 실패 원인 층 |
| 읽는 규칙 | 2 번 반복이라 구성 효과 판정 안 함. 같은 장면 · 흔들림 짝으로 보고. 배치는 VLA 기준 궤적과 겹치게 골랐으므로 "VLA 가 부딪히는 조건에서의 보호" 시험이라고 쓴다 (성공률 일반화 아님) |

**결과 (2026-10-29 13:15 KST, lead):** `outputs/2026-10-29_바구니위허들7장면_VLA단독대AG3S최종구성_closedloop/`.
- 7 장면 모두 배치 후보가 있었다.
- AG3S 는 채택 구성 대신 사용자가 정한 최종 구성 (A8 + 기하 보정 + HSR6 ② 수정 + A6, `snapshot_final2`) 으로 돌렸다.
- 금지 접촉 없는 성공: VLA 단독 0 / 14 → AG3S 6 / 14.
- 허들 접촉 run: 12 → 2. AG3S 의 2 건은 겹침 0.1 · 0.3 mm 다. AG3S 는 손을 탁자 위 31–47 cm 로 올려 넘었다 (VLA 28–30 cm).
- AG3S 실패 8 건:
  - 허들 스침 2 건
  - 쥐지 못함 4 건 (954 × 2 · 1345 · 211, 이 중 211 은 손가락–바나나 접촉 겹침)
  - 들어 올린 뒤 놓침 2 건
  - VLA 단독은 954 · 1345 · 211 에서 사과를 들었다.
  - 쥐기 실패의 원인 층은 아직 진단하지 않았다.
- 이웃 과일 접촉은 client 기록에 없어 기록 frame 을 다시 재생해 셌다 (`hurdle/hurdle_outcomes.py`).

변경 기록 (정의 보완, 결과 본 뒤 — 이유 적음):
- ATS 중심 오차: 한쪽 면 관측의 중심은 구조상 약 r/2 앞에 놓인다 (17.9 mm). 그래서 "보이는 정답 표면 중심" 오차와 보이는 표면 덮은 비율 (5 mm) 을 더했다. 원래 정의 값도 그림 sidecar 에 남긴다.
- LTE 지운 voxel class: 사과 꼭지 · 잎 (visual mesh) 과 사과 반지름 안 (TSDF 띠) 을 따로 셌다.
- 쥐기 목표 항 작동 판정: 기록이 문자열이라 처음 0 으로 셌다. 고쳐서 다시 셌다 (16 run · 49 요청).

정답 정의 · 문턱은 위와 같이 결과 전에 고정한다. 계산 중 기록에 없는 값이 있으면 "측정 불가"로 적고 바꾼 정의는 §변경 기록에 이유와 시각을 남긴다. 결과 폴더: `outputs/2026-10-28_핵심평가_모듈별_가림기둥3_회귀6_x2/`.

**핵심 평가 전 단계 (사용자 지시 2026-10-25 ~20:50 KST):** 판정 1의 GPU 변환과 "TO를 CPU / GPU 중 무엇으로 할지", 그리고 grasp goal을 지키는 TrajOpt 개선(`doc/idea/writing-block_TO.md` E0–E5)을 [`AG3S-PRE-CORE-TO.task.md`](AG3S-PRE-CORE-TO.task.md)에서 먼저 한다. 그 문서의 DF 줄(사용자 판정)이 이 평가의 TO 구성과 기준 commit을 정한다. 재개는 그 문서 §1부터 한다.

인계(2026-10-25)에서 함께 반영할 것:

- 관심 영역 지우기는 이제 `--esdf-backend curobo`의 serving 기본값이다(`--roi-carve --roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore --roi-carve-device cuda`). **실행 목록의 모든 줄에 이 flag를 명시**하고, 서버 시작 로그의 `AG3S roi_carve:` 줄로 확인한다.
- 보고에 적을 한계(사용자 지시): “palm cover 없는 구성(`--links gripper` · `--links gripper_wrist` 모두)은 손바닥 mesh 접촉을 제약에서 보지 못한다.” 사례: 1345번(seed 13453)에서 왼손 몸체가 오렌지에 닿는다(27 row, 최대 겹침 0.57 mm). VLA 단독 실행에서도 있는 접촉이다. `--palm-cover on`을 켜면 이 접촉은 사라지지만 1 episode에서 사과를 쥐지 못했다. 근거: `AG3S_TSDF_DYNAMIC_LOG.md` §14–15, `AG3S-PALM-COVER-1345.task.md` §4.
- 분석 주의: chunk npz의 `clearance` 배열은 모든 구에 main 층을 쓴다. 사과 접촉 권한이 있는 손가락 · 손바닥 link의 여유는 `summary_json.to.metrics.max_violation_pair`로 본다.

## 1. 연구 질문과 이번 결정

**질문:** VLA 내부 task signal을 depth 표면에 연결해 링크별 접촉 제약으로 만들면, task completion을 유지하면서 불필요한 접촉을 줄이는가?

중심은 ATS(target 선택·표면 추정) → LTE(링크별 target 접촉 허용)의 연결이다. HSR(작업 단계 추정·쥠 상태 처리)와 CCR(제약에 따른 action chunk 수정)은 공통 기반이다. 네 모듈을 각각 새로운 기여로 증명하려고 범위를 늘리지 않는다.

이번 단계는 **사과 target / 현재 RB-Y1 simulation**에 대한 핵심 가설의 go/no-go다. 여러 과일·다른 checkpoint·SafeLIBERO·실물의 성능을 대신하지 않는다. 과거 공개 benchmark·실물 계획을 완료 처리하거나 삭제하지 않으며, 현재 후보를 benchmark branch에 자동 merge하지 않는다. 핵심이 성립하면 외적 검증을 별도 잠근 구성으로 진행한다.

## 2. 핵심 후보와 공통 기반

### 2.1 선택

핵심 후보는 **현재 구현된 noun-attention surface graph를 기존 접촉 제약과 연결한 최소 구성**이다. 기존 `ball-owned`보다 좋다는 결론은 아직 없다. 과거 surface 구성의 grasp 실패도 채택 판단에 불리한 기존 근거로 보존한다. 이번 선택은 최고 성공률을 선택한 것이 아니라 연구 가설을 가장 직접적으로 검증할 구성을 선택한 것이다.

| 구성 요소 | 이번 결정 | 이유와 한계 |
|---|---|---|
| Frozen VLA | 현재 16-D RB-Y1 checkpoint 고정 | 새 학습·checkpoint 선택 금지 |
| Target 의미 단서 | `noun_target:L10H7` 고정 | 이미 선택된 신호 재사용; 새 head sweep 금지 |
| 관측 표면 | 기존 5 mm depth-continuous graph, top-5 seed, 현재 잠근 파라미터 | feature edge/unary 추가는 기존 근거가 지지하지 않음 |
| Robot mask | 보정 구성 모두 `--robot-mask render` | extent와 mask를 동시에 바꾼 과거 비교의 혼동 방지 |
| Target 선택/anchor 공통 경로 | 기존 noun grounding + `--attention-split-partition --split-supersedes-anchor` | 두 보정 구성에 동일 적용; 분할 자체를 이번 새 기여로 계산하지 않음 |
| Field | 기존 cuRobo ESDF, 미세 5 mm/거친 20 mm | 미세 표면 경로와 거친 ball 경로가 같다고 서술하지 않음 |
| 링크별 권한 | `--target-field-policy exclude-authorized --target-permission-strict` | 권한이 optimizer·sweep·final check에서 일치하는지 확인 |
| 검사 geometry | **`--links gripper_wrist --finger-cover both`** | 손가락 외에 비허용 손목을 포함하는 최소 기존 집합; 전신 확장 아님 |
| Attached object | 기존 `--held-body-cover --held-slip-detach --held-cover-cap-mm 10` | 유지하되 observed coverage·실제 contact를 함께 보고 |
| Phase/grasp | 기존 probe와 latch를 두 보정 구성에서 동일하게 유지 | 새 probe 학습·새 grasp rule 없음. “HSR 제거”나 “시스템 무학습”이라고 쓰지 않음 |
| Action 수정 | 기존 SQP/OSQP, plan 16, 실행 8, 기존 tracking/smoothness/continuity/deflection | optimizer 비교·가중치 sweep을 재개하지 않음 |
| Gripper | 기존 passthrough/hold 처리 고정 | timing 보정이나 새로운 grasp 생성 없음 |
| Target 자기 표면의 obstacle 근거 오염 | `--nontarget-evidence surface` 공통 | 적용 경로가 ball-owned 미세 layer라는 점을 기록; surface 경로 효과로 혼동 금지 |
| 놓은 뒤 인증 | `--placed-no-seed-certify` 공통 | 이미 있는 수정 유지; metric 허용 접촉을 넓히지 않음 |

**구현 설명과 실제 경로의 차이:** `config.py`와 `runtime/attention_surface.py`에 따르면 surface extent는 미세 target-free layer를 바꾸며, 표면 추정 불가 시 `estimated` fallback이 있다. `robot-mask render`도 TSDF에는 sphere mask와의 합집합으로 적용되고 graph 입력에는 render mask가 쓰인다. 따라서 “모든 거리장이 순수 관측 표면이며 부피 삭제가 없다”라고 주장하지 않는다. 공통 render 설정은 같아도 기존 identity 경로의 mask까지 동일한지 CPU trace로 확인해야 한다.

### 2.2 본 평가에서 넣지 않을 것

| 제외/분리 | 이유 |
|---|---|
| `--neighbour-contact-mm 20` | 목표인 이웃 접촉 감소와 별개로 controller 허용 범위를 넓힌다. evaluator도 이를 허용으로 바꾸지 않음 |
| `--wrist-track-scale 0.1`, `--grasp-anchor`, `--deflection-rate-after-grasp` | 이번 표현 비교와 제어 전략 변경을 섞지 않음 |
| `--recovery-retreat` 및 새 retry | 성공 복구는 별도 능력 |
| `--confidence-gate on`, `--held-3d-rule on` | offline와 serving 간 차이가 확인됐고 downstream 이득 미확립 |
| Feature 기반 identity 재연결 | offline 추가 효과 불확실; 현재 surface serving의 동작과 원고 설명도 구분 필요 |
| `--kept-geometry-follow` | 구형 target anchor 추종의 후보. 이번 surface memory/field 전체를 고친 것으로 간주할 수 없어 별도 진단으로 보존 |
| 새 free-space memory 삭제 | 필요한 일반 원칙이나 아직 설계·검증 안 됨. 이번 평가 진입을 위해 새 mapping 연구를 시작하지 않음 |
| 생성 중 guidance, 새 grasp planner, 신규 학습 | 결과가 현재 질문을 부정할 때 별도 연구로 결정 |

고정 구형 추종을 제외한다고 static memory 오류를 허용해 성공 판정하지 않는다. 아래 계약 검사에서 오류가 핵심 비교를 무효화하면, 1회 수정 예산 안에서 원인 층의 일반적인 수정만 허용한다. 해결하지 못하면 `BLOCKED_IMPLEMENTATION`으로 이 버전의 평가 준비를 끝낸다. 오래된 기하를 조용히 무시하거나 오류 장면을 test에서 빼지 않는다.

### 2.3 실제 실행 설정을 만드는 방식

기준은 기존 run `20261019T132719+0900_f15e587290f6_11408fd6d27f/verify/IFG-APPLE-TARGETSURFACE-FOLLOW/p/{env,start_server}.sh`의 공통 설정이다. 스크립트 전체를 실행하거나 과거 결과 폴더를 복사해 재사용하지 않고, 새 manifest에 공통 argv를 펼쳐 저장한다.

공통으로 유지할 기존 설정:

```text
--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005
--target-field-policy exclude-authorized --target-permission-strict
--target-exclusion ball-owned --owned-max-dist-voxels 3
--attention-split-radius-mm 12 --attention-split-ratio-source raw
--attention-source noun_target:L10H7
--attention-split-partition --split-supersedes-anchor
--robot-mask render --links gripper_wrist --finger-cover both
--esdf-margin 0 --obstacle-margin 0.010 --no-self-collision
--capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32
--closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60
--held-body-cover --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3
--plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4
--w-deflection-rate 20 --to-fast-eval --esdf-host-mirror
--nontarget-evidence surface --placed-no-seed-certify
--confidence-gate off --held-3d-rule off
```

위는 **미실행 설계 argv**이며 바로 실행 가능한 launcher가 아니다. Source revision·checkpoint/probe/XML hash·기본값을 포함한 resolved config·접촉 허용 링크 목록·실제 sphere 목록·client args·seed 목록을 CPU 준비에서 작성한다. `--no-self-collision`의 실제 비활성 범위와 미검사 링크를 보고한다. 두 보정 구성 모두 렌더 backend 설정을 적용하고 과거 `SURF`라는 이름에만 적용되던 launcher 분기를 제거해야 한다.

## 3. 가설을 구분하는 비교

### 3.1 본 비교는 세 구성만

| 이름 | 설정 차이 | 답하는 질문 |
|---|---|---|
| VLA 단독 | 같은 checkpoint, `--no-safe`; AG3S 실행 개입 없음 | 전체 연결을 추가할 가치가 있는가? |
| 공 모양 제외 | §2.3 + `--target-extent estimated` | 같은 내부 신호와 controller에서 기존 기하 표현은 어떤가? |
| 관측 표면 제외(핵심 후보) | §2.3 + `--target-extent attention-surface` | 표면 경계를 보존하는 연결이 실제로 더 나은가? |

두 보정 구성의 actor·mask·검사 링크·phase/latch·solver budget·client stop 규칙은 같다. 실행 이후 관측이 달라지는 것은 closed-loop 효과이며, 같은 seed라고 이후 입력이 같다고 쓰지 않는다. 과거 9/10·7/10과 직접 비교해 “단순화 효과”라고 하지 않는다. 공 모양 제외는 내부 정보를 이미 사용하므로 **기하-only baseline이라고 부르지 않는다.**

### 3.2 원인 분리: 기존 기록의 같은 입력에서 먼저 확인

| 질문 | 고정하는 것 / 바꾸는 것 | 판정과 주장 한계 |
|---|---|---|
| 내부 신호가 작업을 구별하는가? | RGB-D·q·기하 고정, target instruction과 그에 대응한 noun attention 변경. 추가 음성 대조는 patch 순서 permutation, 합계 보존·seed 고정 | target identity와 잘못된 제외가 지시에 따라 달라지는지 측정. 성립하지 않으면 내부 task signal 기여 미확인. Policy의 인과적 reasoning이나 closed-loop 성공을 입증하지 않음 |
| 같은 identity에서 extent가 다른가? | 같은 recorded identity·pose·query와 raw depth를 두 extent 구현에 공급 | target 잔류, non-target 손실, 실제 field의 distance/gradient·권한 질의를 함께 비교. identity 정렬 실패는 별도 오류로 집계 |
| 링크 권한이 필요한가? | 같은 target geometry·q, `none` / `exclude-all` / `exclude-authorized` | 같은 target 근처 손가락·손목 query에서 다른 허용이 실제 발생해야 함. 구분되는 query가 없으면 LTE 이득 미검증으로 종료 |
| 표면 오차인가, 행동 보정 문제인가? | 동일 visible-target oracle extent를 두 표현 경로의 진단에 공급; optimizer는 그대로 | oracle도 실패하면 표현 이외의 병목 후보. oracle은 성능 상한이 아니며 이것만으로 CCR 원인을 확정하지 않음 |

사용 가능한 recorded input·adapter로 가능한 범위부터 한다. 동일 identity를 공급할 adapter가 없으면 필요한 최소 연결만 §7의 준비 예산에 넣는다. 없는데 이미 pure extent ablation이 가능하다고 쓰지 않는다. 처음부터 모든 조합의 closed-loop를 돌리지 않는다.

외부 segmenter baseline은 이번 내부 go/no-go의 선행 조건이 아니다. “외부 segmenter보다 우수/저비용”을 논문에서 주장하려면 동일 depth/controller를 사용하는 실제 비교를 추가해야 한다. 비교 없이 external-segmenter-free는 구성의 설명으로만 쓴다.

## 4. 평가 정의: controller와 독립

| 지표/규칙 | 이번 정의 |
|---|---|
| Task success | 목표 물체의 basket 내 release와 2.5초 유지. evaluator의 GT object/basket·measured gripper로 판정. 기존 성공 함수를 검토해 이 조건과 일치하는지 명시 |
| 금지 접촉 | robot–non-target fruit/obstacle/support, 비허용 robot link–target, 운반 target–장애물 접촉. 허용 쌍은 별도 고정 |
| 허용 target 접촉 | 작업에서 지정된 active hand의 기존 contact-link set–true target, 첫 유효 release 전. active palm 포함 여부를 구현에서 추출·고정하고 반드시 보고. Palm push와 target 이동량은 별도 진단 |
| 허용 support 접촉 | 초기 탁자 지지, 올바른 destination 내부 바닥에 내려놓는 접촉. Basket rim·바깥벽 접촉이나 robot–탁자를 일괄 면제하지 않음 |
| 초기 접촉 | 초기 fruit–fruit/static 지지 쌍은 별도 기록. 로봇 개입으로 새로 발생/증가한 물체 이동·접촉을 숨기지 않음; robot–이웃 과일은 항상 평가 |
| Contact-free success | 같은 run에서 task success ∧ 금지 접촉 없음 ∧ contact 기록 완전 |
| 평가 창 | 첫 명령부터 place 후 2.5초, place 미도달이면 600 control step. 모든 구성의 obstacle-stop=0; contact를 기록하며 같은 시간 예산 유지 |
| 실제 접촉 | MuJoCo physics substep contact 기준. `dist < 0`의 raw 침범을 주 판정, `dist <= 0` 접촉과 최대 침범·누적 접촉 시간도 보고. TO clearance 10 mm 위반을 물리 침범으로 세지 않음 |
| 검사 범위 | controller의 gripper_wrist·held object와 evaluator의 전체 robot contact를 각각 보고. 미검사 링크 접촉을 주 지표에서 삭제하지 않음 |
| 진행/비용 | HOLD 시간, timeout, 실패 단계, 완료시간, wall-clock end-to-end median/p95. 성공 run 완료시간만으로 속도 개선을 주장하지 않음 |

평가용 object/phase/permission은 predicted 값과 독립이다. Controller가 20 mm 접촉을 허용해도 evaluator의 금지 접촉이 허용으로 바뀌지 않는다. 자연 초기 배치·support 접촉 분류와 active contact-link set은 CPU 준비에서 event 예시로 검증하고 `definitions.json`에 고정한다. 그 파일의 hash가 없는 상태는 **본실험 실행 불가**다.

성공·접촉·시간의 분모는 예약한 모든 scene×seed다. 정책/solver 실패·timeout을 누락하지 않는다. 실행 전 인프라 실패만 같은 seed로 1회 재시도하며 원 시도를 보존한다. 접촉 기록이 없으면 접촉 0으로 바꾸지 않고 unknown으로 두고 낙관/비관 bounds를 보고한다. Unknown이 결론을 바꾸거나 paired run이 미완료이면 확정 성공 판정을 하지 않는다.

## 5. 장면·반복·예산

### 5.1 세 가지 데이터 용도

1. **기존 실패 자료:** 211·954·1668 및 이미 본 IFG/T43–T63 자료는 development/diagnosis다. 새로운 seed를 넣어도 새로운 scene test라고 부르지 않는다.
2. **개발 pilot:** metadata로 고른 10개 scene × 2 policy seeds × 3 구성 = 60 run. VLA 성공 여부로 걸러내지 않는다. 먼저 총량을 잠그고 실행한다.
3. **독립 확정 test:** 다른 80개 scene × 2 policy seeds × 3 구성 = 480 run을 기본 설계로 둔다. 이는 새 계획값이며 power 보장이 아니다. Pilot으로 비용·paired disagreement를 추정해 이 상한에서 의미 있는 판정이 가능한지 확인한다. 불가능하면 본실험을 시작하지 않고 “현재 예산으로 확정 불가”로 종료한다. 본실험 이후 유리한 seed나 scene을 추가하지 않는다.

독립 scene은 episode 번호만이 아니라 원본 layout/trajectory family를 기준으로 나눈다. 기존 개발·held-out·수정에 사용한 집합과 그 family를 제외하고 deterministic metadata hash 순서로 선정한다. Checkpoint 학습과의 독립성을 확인하지 못하면 **method tuning에 대한 held-out**이라고만 쓴다. 대상 pool이나 family metadata가 부족하면 독립성을 추정하지 말고 적용 범위를 줄여 보고한다.

Pilot strata는 clear 2 / target 근접·가림 3 / carry hurdle 3 / destination 장애물 2다. Test는 각각 20개로 균형화한다. Target 가림은 head camera 기준 약 30%(5–60% 범위) 또는 장애물 없는 기준 동작과의 기하 간섭으로 정의하며, 후보의 성공 여부로 배치를 고르지 않는다. 무장애물 reference를 새로 수집해야 하면 pilot/test 예산에 포함하고 그 결과로 성공 장면만 고르지 않는다. 장면·장애물 생성 규칙과 seed schedule은 test 결과 열람 전에 hash 고정한다.

### 5.2 재현성과 민감성은 보조 검사로 제한

- 기존 진단 scene 211·954·1668에서 공 모양 제외를 동일 설정·동일 seed로 두 번씩: 6 run.
- 같은 구성에서 첫 close 직전의 사전 정한 요청에, active arm의 미리 정한 관절 명령 하나에 ±1e-8 rad를 1 control row만 주입: 6 run. 주입 seed/시점/채널을 사전 기록한다.
- 이 12 run은 국소 민감성 진단일 뿐, 일반 변동률이나 필요 표본 수 전체를 추정하는 자료가 아니다. 정책 seed 변동과 floating-point perturbation을 같은 것으로 합치지 않는다.
- 본 비교는 여러 policy seeds로 수행한다. 동일 seed가 재현되지 않으면 최초 divergence·환경/seed reset을 조사하되 완전 bit determinism 달성을 별도 장기 과제로 만들지 않는다.

### 5.3 비용 상한과 중단

| 단계 | 상한(설계값) | 소진 시 행동 |
|---|---|---|
| CPU 평가 준비·계약 검사 | 2 working days, 일반 수정 묶음 1회 | 해결 못한 계약 위반을 기록하고 이 버전의 준비 종료 |
| Pilot + 재현성 + 원인 분리 | 60 + 12 + 추가 최대 18 = **90 run**, GPU 점유 wall time **18시간** 중 먼저 도달 | 실패별 새 설정을 만들지 않고 진행/중단 판정 |
| Locked test + reference 수집 | **최대 560 rollout**, GPU 점유 wall time **72시간** 중 먼저 도달 | 480 비교 run 미완료면 확정 결론 금지. 추가 자동 확대 없음 |

Test의 560 상한은 480 비교 run과 필요할 때 최대 80개의 무장애물 reference용이다. 재시도도 총 실행 예산에 포함한다. 실제 시간은 GPU 공유 대기와 분리하며, 과거 10 run의 약 58분은 현재 graph 비용 예측값이 아니다. 이 예산은 새로운 설계 제안이지 이미 예약한 GPU 시간이 아니다. 실행 전 자원 확인과 기존 lock 규칙을 적용한다.

## 6. 성능 목표와 결과 판정

### 6.1 이번에 정한 실용적 목표

5 percentage points(pp)는 “100회당 5회”에 해당하는 **이번 연구의 의사결정 여유**다. 과거 9/10에 맞춘 값도, 통계적으로 보장된 정확도도 아니다. Pilot 결과에 맞춰 완화하지 않는다.

각 baseline B(VLA 단독, 공 모양 제외)에 대해 후보 C의 다음 세 조건을 모두 만족할 때 해당 비교에서 개선을 인정한다.

1. `Δtask = task(C) − task(B)`의 양측 95% CI 하한이 **−0.05보다 큼**.
2. `Δcontact = forbidden(C) − forbidden(B)`의 점추정이 **−0.05 이하**, 95% CI 상한이 **0보다 작음**.
3. `Δsafe = contact_free_success(C) − contact_free_success(B)`의 점추정이 **+0.05 이상**, 95% CI 하한이 **0보다 큼**.

첫 번째 비교는 전체 시스템 대 VLA, 두 번째는 표현 방법 대 공 모양 제외다. 둘 다 통과해야 두 수준의 강한 개선 주장을 함께 한다. VLA 대비만 통과하면 geometric wrapper의 효과와 새 표면 방식의 추가 효과를 구분한다. 공 모양 제외의 접촉률이 이미 너무 낮으면 추가 접촉 감소를 증명할 수 없으며, 사후 다른 지표로 primary를 바꾸지 않는다.

**통계 단위:** scene별 seed 평균 후 scene을 동일 가중한다. 같은 scene의 seed·frame·contact row를 독립 scene으로 세지 않는다. Stratum을 보존한 paired scene bootstrap 10,000회와 모든 scene 결과를 보고한다. 확정 검정은 VLA 대비 조건을 먼저, 통과한 경우 공 모양 제외 대비 조건을 다음에 해석한다. 그 밖의 비교·stratum은 기술/탐색 결과로 표시한다.

**경계 사례:** discordant scene이 0이어서 bootstrap CI가 [0,0]이 되는 것을 동등성 증거로 사용하지 않는다. Binary endpoint는 scene 내 seed 중 불리한 discordance가 하나라도 있는 사건에 대한 exact binomial upper bound를 보수적으로 함께 사용한다(양측 95% 상한); 이를 만족하지 못하면 task 유지 미확인이다. 낮은 event 수·CI 방법의 유효성은 실행 전 CPU 분석 설계에서 확인하고 고정한다. 80개 scene도 검정력 보장이 아니며, CI가 넓다는 이유로 자동 표본 확대하지 않는다.

### 6.2 종료 결정표

| 결과 | 결정 | 다음 작업 |
|---|---|---|
| 두 baseline 대비 §6.1 충족 + same-input 경계/권한 효과 확인 | **핵심 가설 지지, 개발 종료** | 방법·파라미터 freeze, 정확한 Method/Results 작성, 외적 검증 |
| VLA 대비 충족, 공 모양 제외 대비 미충족 | **전체 연결은 유용하나 새 표현의 실행 우위 미확인** | 그 범위로 보고. 더 좋은 표면이 더 좋은 실행이라고 주장하지 않음 |
| 표면·query 오류 감소, contact/task 이득 미충족 | **기하 기여만 지지, 이 버전 개발 종료** | 기하 결과와 실패를 정리. 원래 실행 목표를 달성했다고 하지 않음 |
| Task 손실/접촉 악화의 명확한 근거 | **후보 기각** | 기존 구성을 보존, test를 다시 튜닝하지 않음 |
| CI 넓음, contact 드묾, 예산 소진, 데이터 결측 | **판정 보류로 평가 종료** | 한계 보고. 후속은 별도 질문·예산·새 test로만 시작 |
| 준비 계약 불일치가 1회 수정 후에도 남음 | **구현 준비 실패** | 통계 평가를 실행하지 않음. 시스템 통합 문제와 가설 반증을 구분 |

**연구 질문을 바꾸는 조건:** matched visible geometry에서 실제로 필요한 target 접촉이 가능하고, 보정 전후의 손–target 상대 자세와 gripper 사건을 비교했을 때 보정이 그 관계를 깨는 재현 가능한 증거가 있어야 한다. Oracle run도 실패했다는 사실만으로는 부족하다. 이 근거가 있을 때만 다음 질문을 “VLA의 접촉 조건을 보존하면서 행동을 수정할 수 있는가?”로 바꾼다. 생성 중 개입은 그 질문의 후보 구현 중 하나이며 자동 채택하지 않는다.

Recorded command를 바꾼 open-loop replay는 기전 진단으로 표시한다. 이것으로 새 closed-loop 방법의 성공을 예측하지 않는다. 필요하면 §5.3의 추가 18 run 예산 안에서 같은 개입의 closed-loop 확인을 하며, 그 예산이 없으면 원인 확정 대신 후보 원인으로 남긴다.

False exemption/잔류 field가 문제라면 질문은 “시간에 따라 유효한 contact geometry를 어떻게 유지하는가?”다. 이 역시 현재 판정을 마친 뒤 별도 방법 변경이다. 원인 미상은 미상으로 종료할 수 있다.

## 7. 개발을 허용하는 범위와 모듈 완료

| 계약 | 최소 완료 확인 | 허용하지 않는 범위 확대 |
|---|---|---|
| 의미→표면 | 입력 정합·GT 누수 없음·same-input 의미 대조·episode별 오류 | 모든 가림 해결, 새 feature 학습 |
| 표면→field | non-target 손실과 target 잔류 동시 측정, 실제 query에서 의도한 차이 확인, fallback/age/version 기록 | TSDF 전체 재설계, shape completion |
| 권한→실행 | 동일 snapshot·링크 권한을 최적화/검증이 공유, 손목 보호 구분, actual contact 기록 | 전신 모델 재구축 |
| 상태 처리 | 실제 grasp와 phase를 구별하고 인정 지연·오인·슬립을 측정 | 모든 파지 성공, 복구 정책 추가 |
| 시스템 | 잠근 실험과 결과표·불확실성·비용·실패 한계를 작성 | 211·954·1668 전부 성공할 때까지 반복 |

1회 수정은 좌표·단위·권한 불일치, seed 재설정 오류, 동일 geometry version 전달 실패처럼 실험의 의미를 무효화하는 문제에 쓴다. 성공을 얻기 위한 threshold/weight/scene 변경은 이 예산에 들어가지 않는다. 이미 알려진 쥠 인정 지연과 stale geometry도 무조건 먼저 고치는 것이 아니라 해당 계약을 깨는지 진단한다. 수정 전후와 모든 실패를 보존한다.

## 8. 바로 다음 작업과 산출물

**다음 작업은 CPU 평가 준비 하나다.**

1. 최근 형상 추종의 누락 `verify.json`을 기존 raw에서 정리하되 새 연구 성능으로 합산하지 않는다. `address_check`→analysis 순서를 새 launcher 원본에서 고치고 기존 실행 기록으로 확인한다.
2. §2.3의 세 구성 argv/resolved config를 생성한다. Mask·geometry·권한·fallback·HSR/latch의 차이를 표로 검토한다. 최근 9/10 구성은 historical reference로만 기록한다.
3. 기록된 장면으로 평가용 허용 쌍과 전체 robot contact 분류, success·place-stop, 실제 궤적 replay 정합을 확인한다. Geometry 그림은 같은 pose에서 손가락·손목·target·이웃 표면을 겹친다.
4. 기존 inspected scene/family 목록과 pilot/test 후보 pool을 작성한다. 80개 독립 test를 확보할 수 없으면 독립성 미확인으로 표시한다.
5. 최소 산출물: `config_manifest.json`, `definitions.json`, `split_manifest.json`, `preflight.verify.json`, field/query 그림 1장. 파일 hash·run 경로를 이 task에 연결한 뒤 pilot을 실행한다. 이번 문서는 이 파일들을 이미 생성했다고 주장하지 않는다.

최종 산출물은 main 비교표 1개(task/contact/contact-free success/HOLD/time/CI), 같은 입력의 의미·경계·권한 분석표 1개, 실제 scene→표면→field→실행 연결 그림 1개, 실패·runtime 표다. 성공 시연은 별도로 표시하고 정량 분모를 대신하지 않는다. 최종 판정은 이 task의 종료 표에 `SUPPORTED`, `REJECTED`, `INCONCLUSIVE`, `BLOCKED_IMPLEMENTATION` 중 하나와 근거를 남긴다.
