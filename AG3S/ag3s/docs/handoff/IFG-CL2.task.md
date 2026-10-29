# IFG-CL2 — 하이브리드 target extent (ball 부피 제거 + attention-surface 로 이웃 보호) 구현 (implementer, CPU)

작성: 2026-10-17 KST, lead. 선행: `IFG-CL1.task.md` · `IFG-CL1.impl.md` (attention_surface / render mask 구현).

## 왜 (측정)

held-out closed-loop (C1aN vs SURF, 34 seed 짝): 성공 34 vs 28 (McNemar p 0.031), SURF 실패 = no_grasp 5 · not_placed 1. 진단
(`doc/active/reviews/vla_dump_20261015/WORKLOG_S0S3_KO.md` 2026-10-17 15:20 절):
- SURF 재실행 6/6 실패 재현. GT visible segmentation (oracle-visible, render mask 동일) 로도 6 중 4 실패 → **"보인 표면만 지우는" 경로**
  가 target 의 안 보인 부분 (TSDF seed) 을 손가락 앞에 장애물로 남긴다 (GT 대조: 정답 target 인데 남긴 seed 가 chunk 당 C1aN 1 · SURF 성공 79 · SURF 실패 134).
- 우리 attention 영역의 identity 오류 (다른 물체를 target 으로 지움) 가 실패를 더한다 (실패 6 중 5 run, 성공 28 중 1 run).

## 설계 (lead 제안 — 맞지 않으면 구현 전에 SendMessage 로 묻는다)

새 `esdf.target_extent = "attention_surface_hybrid"` (CLI `--target-extent attention-surface-hybrid`):
1. **지우는 범위 = 지금의 `estimated` (ball_owned) 그대로** — target 부피 (안 보인 쪽 포함) 를 지운다.
2. **attention-surface 는 보호에만 쓴다:** 공 안 seed 중, 이번 frame 의 attention-surface code 영상에서 **target 이 아닌 · support 가 아닌 관측 표면**
   (= 선택 영역 밖의 비지지면 graph node 의 pixel) 의 표본에 target 표본보다 가까운 seed 는 **지우지 않는다** (이웃 과일 · crate 표면 보존).
   ball_owned 의 non-target 근거 (`nontarget_evidence`) 를 이 표본으로 바꾸는 / 더하는 형태가 자연스러우면 그 방식을 제안해도 된다.
3. **identity 안전장치:** 선택 영역이 serving 의 manipulated 물체와 맞지 않으면 (기존 `identity_unresolved` 판정과 같은 규칙, 또는 선택 영역
   중심과 ball 중심 거리 > ball 반지름) 이번 frame 은 보호를 적용하지 않는다 (= C1aN 과 같은 `estimated`). 이유 기록.
4. robot mask 는 `--robot-mask` 를 따로 둔다 (spheres / render 독립).

## 수락 조건 (CPU, `.venv-ag3s` 만 — GPU 가 보이는 venv 로 test 금지)

- **offline 재생 비교:** IFG-CL1 closed-loop 기록 (`$CL1 = .../20261016T042508+0900_f15e587290f6_86dde89fb754/verify/IFG-CL1`, server chunk 의
  depth / attention / ball / seed 기록과 `analysis/<run>.json` 의 server_segment) 에서 SURF 실패 6 run 과 성공 run 몇 개의 pre-grasp chunk 를
  골라, 새 모드로 fine target-free 분할을 다시 계산한다. GT 대조는 lead 의 `AG3S/ag3s/experiments/studies/ifg_cl1_keptseeds.py` 와 같은
  방식 (target convex hull ±5 mm). 보고: chunk 당 **정답 target 인데 남긴 seed** (C1aN ≈ 1 수준이어야 함), **정답 이웃 물체인데 지운 seed** (C1aN 대비
  감소해야 함), 보호가 꺼진 frame 비율과 이유. 재생이 기록만으로 불가능한 항목은 그 이유를 적는다.
- 기본 · 기존 모드 비트 동일 (기존 test 통과), 새 test (보호 규칙 · identity 안전장치 · 기록 키).
- 1 요청당 추가 ms (attention_surface 계산은 그대로라 SURF 와 비슷할 것; 보호 분류 ms 를 따로).

## 하지 말 것 · 넘길 것

GPU · server · rollout 금지. 잠금 `ifg_*.py` 수정 금지. 문서 · figure 수정 금지. commit 금지.
`AG3S/ag3s/docs/handoff/IFG-CL2.impl.md`: 바꾼 파일, flag, offline 재생 수치 표, test 결과, 추가 ms, 남은 위험.
