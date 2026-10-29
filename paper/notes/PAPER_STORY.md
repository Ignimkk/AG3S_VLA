# Paper Story — Working Hypothesis

Updated: 2026-10-05 19:45 KST

## Status

`WORKING_HYPOTHESIS`. 원문의 desired outcome은 입증된 conclusion이 아니다.
결과와 수치의 측정·파생·추정·계획·placeholder 상태를 [RESULTS](../evidence/RESULTS.md) 또는 연결된 근거에 기록한다.

## Problem formulation

We study whether task-relevant representations inside a VLA policy can support
an explicit 3D representation for action refinement under task-dependent contact constraints.

## Research question

Can multi-view grounding of VLA internal attention improve physical interaction outcomes
without an unacceptable loss of manipulation performance?

## Proposed method narrative

VLA internal representation → multi-view 3D grounding → task target and obstacle reasoning
→ explicit 3D safety representation → constrained action refinement → receding-horizon execution.

This is the intended paper narrative. The fidelity of each stage and its causal contribution
require direct measurements and ablations. Attention should not be called a semantic explanation
or a reliable target detector solely because it is visualized.

## 현재 source에서 확인한 범위

T39 snapshot은 16-D VLA, AG3S, cuRobo ESDF, project SQP/OSQP와 gate/HOLD를 비교한
실행 자료를 갖고 있다. cuRobo MotionGen을 이용한 comparison으로 쓰지 않는다.
해당 flags는 gripper·left-arm approach 범위이며 whole-body safety를 증명하지 않는다.
Source: [T39 실행 설정](../../outputs/verify/T39/p/start_server.sh),
[research log](../../benchmark/ag3s/docs/AG3S_T0T6_LOG.md),
[BASELINES](../evidence/BASELINES.md).

## 평가로 연결할 주장

- C002: representation→3D grounding의 유효성 — P002·P003.
- C003·C004: target-aware contact와 refinement의 기여 — P001·P003.
- C005: 범위 밖 조건의 일반화 — P004와 split audit.
- C007: 실행 시간 예산 — P005의 end-to-end timing.

C006과 R001–R003은 현재의 descriptive observation이다. 이 결과만으로
"reduces collisions while preserving manipulation capability"를 abstract/conclusion에 쓰지 않는다.

## 논문 독자와 positioning

기본 독자는 robot motion planning을 알지만 AG3S 파이프라인은 처음 보는 연구자다.
내부 T/E 발견 번호 대신 현상·방법·조건을 설명한다. CFP scope 확인은
[guidelines source S01](../guidelines/SOURCES.md#s01)을 따른다.
VLA / embodied vision / 3D scene understanding 중심 positioning은 evidence를 확보한 뒤 조정한다.
