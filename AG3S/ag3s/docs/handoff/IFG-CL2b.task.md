# IFG-CL2b — 하이브리드 보호 규칙 보완: 잡을 물체 근처 node 를 "다른 물체" 근거에서 뺀다 (implementer, CPU)

작성: 2026-10-17 KST, lead. 선행: `IFG-CL2.task.md` · `IFG-CL2.impl.md` (하이브리드 구현, offline 재생 결과).

## 사용자 판정 (2026-10-17)

- 하이브리드에 보완 규칙을 **미리 정한 값으로** 추가한다. IFG-CL1 closed-loop 기록은 이제 **개발용 (dev)** 으로 쓰고, 허들 2 차를 독립 테스트로 쓴다.
- 동시에 GPU 로 render mask 분리 진단을 한다 (lead 가 별도로).

## 규칙 (값 고정 — 재생 결과를 보고 바꾸지 않는다)

`IFG-CL2.impl.md` 의 측정: 보호된 정답 target seed 의 71 % 는 가장 가까운 "other" node 가 선택되지 않은 target 자신의 표면이다
(영역이 과일 일부만 덮음). 보완: **"other" (선택 밖 · 비지지면) node 중 serving 의 manipulated 기하 점 (identity guard 가 쓰는 그 점들) 에서
`timing.fusion_voxel` (10 mm) 안에 있는 node 는 other 근거에서 뺀다.** manipulated 기하가 없으면 보완 없이 지금 규칙. 값 (1 × fusion voxel) 은
고정이며 flag 로 바꿀 수 있게 하되 기본 1.0 배. 이 보완을 켠 것이 hybrid 의 기본 동작이 되도록 하고 (`--hybrid-other-exclude-voxels`, 기본 1.0;
0 = IFG-CL2 그대로), IFG-CL2 동작은 0 으로 재현 가능해야 한다.

## 수락 / 보고 (CPU, `.venv-ag3s` 만)

- `cl2_hybrid_replay.py` 와 같은 offline 재생 (IFG-CL1 68 run, pre-grasp chunk): IFG-CL2 표와 같은 열에 `hybrid (+exclude 1.0)` 행 추가.
  보고만 한다 — 값을 바꾸거나 다른 값을 시험해 고르지 않는다 (0 과 1.0 두 행만).
- 새 test (보완 규칙 · 0 = IFG-CL2 비트 동일 · manipulated 기하 없을 때). 기존 test 통과.
- 허들 2 차에 쓸 server flag 한 줄 (C1aN + hybrid; robot mask 는 lead 가 진단 결과를 보고 정하므로 spheres / render 두 줄).

## 하지 말 것 · 넘길 것

GPU · server · rollout 금지, `.venv-ag3s` 외 venv 로 test 금지, 잠금 `ifg_*.py` · 문서 · figure 수정 금지, commit 금지.
`IFG-CL2.impl.md` 에 §"CL2b" 절을 덧붙인다 (바꾼 파일:줄, 재생 표, test, 남은 위험).
