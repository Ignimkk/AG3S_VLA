# SL4b — Dev calibration 연구: Panda ATS용 attention cell (선택 규칙 사전 등록)

> writer: lead (A0) · 읽는 쪽: implementer, verifier · 작성 2026-10-13 KST
> branch `feature/panda-benchmark`, 시작 HEAD `c2ba3fea`. 계획: `doc/develope_PLAN_SUMMARY.md` §3.1, §6.0(dev init 0–4).
> 사용자 판정 2026-10-13: SL4 확정. ATS target 오선택(t4 plate, t5 bowl_2, t9 target 점 0)에 대해 dev calibration 연구 수행.
> Test(init 5–49)는 **사용하지 않는다.**

## 왜 지금 이것인가

SL4에서 V11AB attention cell(L8H2)로 만든 ATS가 접근 단계에서는 대체로 target을 잡았지만(101/105, centroid median 12 mm),
10개 task 중 3개에서 다른 물체(plate, 같은 모양의 bowl_2)를 target으로 골랐다. LIBERO Spatial은 같은 bowl 두 개를 위치 관계로 구분하는
과제라 이 오선택은 V11AB가 진짜 target을 장애물로 남겨 grasp를 방해하는 방향으로 작동할 수 있다. 원인이 **attention cell**인지
(다른 cell이 target을 더 잘 가리키는지), **grounding 기하/clustering**인지 분리하고, cell이 원인이면 dev에서만 고른 Panda cell을 제안한다.

## 사전 등록 (결과를 보기 전에 고정 — 바꾸지 않는다)

| 항목 | 정의 |
|---|---|
| Dev 데이터 | LIBERO Spatial task 0–9 × init state 0–4 (50 episodes). init 0은 SL3 run 재사용, init 1–4는 새 rollout(pi0.5-only, 같은 protocol) |
| Phase | SL3/SL4와 같은 정의(MuJoCo finger–target contact, lift z0+10 mm). **평가 phase = approach + grasp** (운반 단계는 attach가 없는 offline 한계로 제외) |
| 1차 지표 (`target_correct`) | call별: ATS가 target을 냈고(has_target) **ATS target 점의 최근접 instance 다수결 = 진짜 target** 이고 centroid 오차 ≤ 30 mm 이면 1. 평가 phase call 전체에 대한 비율 |
| 2차 지표 | pooled precision(10 mm), recall median, centroid 오차 median/p95, task별 `target_correct` |
| 후보 cell | V11AB L8H2 + **선별 지표**(dev 50 episode attention에서 approach+grasp call의 head+wrist target mass 평균, un-mirror)의 상위 16개 cell. 진단 grid는 denoise 0, agg `last`(V11AB와 같은 집계) |
| 채택 조건 | 후보 cell이 L8H2보다 1차 지표가 **≥ 10 %p 높고**, task별로 **10개 중 ≥ 6개 task에서 L8H2 이상**이면 "Panda calibrated cell" 후보로 사용자에게 제안. 여러 개면 1차 지표 최고, 동률이면 낮은 layer. 조건을 만족하는 cell이 없으면 L8H2 유지 |
| 원인 분리 | L8H2의 오선택 call마다: attention argmax(head·wrist)가 진짜 target 위에 있었는지 → attention 원인 / grounding 원인 비율 |
| 참고(후보 아님) | GT target oracle seed(segmentation으로 target pixel만 attention=1) — ATS 기하 상한. privileged input으로 표시, 선택에 사용 안 함 |

## 구현에 요청하는 것 (→ implementer)

AG3S core(`AG3S/ag3s/**`, `AG3S/trajopt/**`) 수정 금지(SL4와 같은 원칙).

1. `benchmark/runners/perception_replay.py`:
   - `--attention-cell L,H` : attention audit run에 저장된 진단 grid(`[L,H,768]`, float16)에서 해당 cell을 꺼내 사용(기본은 V11AB L8H2 — 기존과 bit 동일해야 함).
     float16 underflow(SL3 #6에서 일부 cell의 block 합이 0)는 그대로 두고 그 call을 `attention_invalid`로 표시.
   - `--grounding-only` : ESDF·constraint 단계를 건너뛰고 grounding까지만 기록(빠른 선별용). grounding 결과는 full 실행과 동일해야 함(test).
   - `--attention-source oracle` : GT target segmentation으로 attention을 만든 참고 실행(privileged 표시).
   - 여러 source run(init 0 run + init 1–4 run)을 한 번에 받는 것.
2. `benchmark/evaluators/ats_calibration.py`: 위 사전 등록 지표를 그대로 계산하는 평가기(phase 정의 재사용, 후보 선별, 채택 조건 판정, 원인 분리). 결과 JSON.
   채택 조건 문구를 코드 상수로 두고 manifest에 기록.
3. Test(CPU): `--attention-cell 8,2` = 기존 경로, grounding-only = full의 grounding, 평가기 지표 정의(작은 fixture).

## 측정에 요청하는 것 (→ verifier)

| # | 무엇을 | 명령/내용 |
|---|---|---|
| 1 | Dev rollout init 1–4 | `launch_libero_pi05_only.py --task-ids 0..9 --episode-ids 1,2,3,4 --capture-geometry --capture-segmentation` (GPU lock) |
| 2 | Attention audit | init 1–4 run (GPU lock). init 0은 SL3 audit 재사용 |
| 3 | Scene GT export | init 1–4 episode |
| 4 | 후보 선별 | 50 episode attention에서 선별 지표 → 상위 16 + L8H2 |
| 5 | Grounding-only replay | 후보 17개 + oracle 참고, 50 episode (GPU lock) |
| 6 | 평가 | 사전 등록 지표, 채택 조건 판정 결과, 원인 분리, task별 표 |
| 7 | 확인 실행 | 채택 후보가 있으면 그 cell로 **full**(ESDF 포함) replay 1회: SL4 #5·#6 지표를 L8H2와 나란히 |
| 8 | 회귀 | CPU regression, `--attention-cell 8,2` grounding = SL4 default run grounding |

- 판정 문장은 쓰지 않는다. 채택 조건 충족 여부는 평가기 출력(사실)으로만 기록.
- figure 3종(`sl4b-*`): (1) 실제 scene — L8H2 오선택 task(t4/t5/t9)에서 L8H2 vs 최고 후보 attention + ATS target, (2) graph — 후보별 1차 지표(task별 점), 원인 분리,
  (3) table — 사전 등록 규칙, 후보 순위, 채택 조건 결과.
- raw: `outputs/verify/SL4b/`. GPU는 실제로 lock을 잡는 형태(`( flock -w N 9 || exit 1; … ) 9>/tmp/ag3s-vla-gpu-validation.lock`), 시작·종료를 main에 알림.

## 이 STEP 이 답하는 질문

Panda/LIBERO에서 ATS 오선택의 원인이 attention cell인가, 그리고 dev에서만 고른 다른 cell이 사전 등록한 기준을 넘는가?

## 되돌아올 지점

- 원인이 grounding(기하/clustering)이면 → cell 교체로는 해결되지 않는다. V11AB grounding 규칙 변경은 method variant이므로 사용자 판정.
- 채택 조건을 넘는 cell이 없으면 → L8H2 유지, 오선택은 Panda 결과의 한계로 기록.
- 채택 후보를 main으로 쓸지, variant로 둘지는 사용자 판정(test 전 고정).
