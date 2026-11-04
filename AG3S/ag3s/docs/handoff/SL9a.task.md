# SL9a — Panda 표면 영역이 노드 1–2개로 줄어드는 원인 (진단만, offline)

> writer: lead (benchmark checkout), 2026-11-04 20:xx KST. 사용자 판정 "추천대로 진행"(SL9 §7).
> 진단만 한다. 코드 · parameter는 바꾸지 않는다. 고칠 방향은 결과 보고 뒤 사용자가 판정한다.

## 증상

- 근거:
  - SL9 smoke: P0m-A LIBERO t0e0, 채택 6 call 모두 영역 1–2 노드. graph 노드는 약 27,000개다.
  - SL7i(fb LIBERO 9): 채택 20개 중 6개가 1–2 노드였다. 선택 노드 수 중앙값은 7, p90은 2,405였다.
- 큰 영역(수백–수천 노드)은 대부분 ATS 점과 겹치지 않아 `not_ats_target`으로 기각됐다.
- 표면 선택: `AG3S/ag3s/runtime/attention_surface.py`. 방법은 `top1-only-ns:N`이다:
  - seed 자격: patch 안 비로봇 · 비받침 픽셀 ≥ 60
  - seed: attention 상위 patch
  - solve: graph 위에 값을 퍼뜨린다
  - 영역: `{m ≥ 0.5}` 중 seed 질량이 가장 큰 덩어리

## 무엇을 재나 (call마다, GT와 대조)

1. **seed patch:** 어느 patch가 seed인가, 그 patch 픽셀의 GT class 구성, 그 patch 안 graph 노드 수(target · 받침 · 로봇 · 기타).
2. **target의 graph 노드:** GT target 픽셀이 속한 노드 수. 그중 받침 노드로 표시된 수, 로봇 mask로 빠진 수.
   - target 노드들이 graph에서 몇 개의 연결 성분으로 나뉘는가(jump gate 10 mm 끊김).
3. **solve 값 m:** target 노드 위 m의 분포(중앙값 · 최대 · ≥ 0.5 비율), seed 노드의 m, 영역 노드 수.
   - 영역이 1–2 노드일 때 m이 0.5를 넘는 노드가 왜 그것뿐인지 본다: m이 seed에서 얼마나 빨리 떨어지는가, 정규화 문제인가.
4. **분류 (call마다 하나):**
   - (a) seed가 target 밖 / 가장자리
   - (b) target 노드가 받침 · 로봇으로 빠짐
   - (c) target 노드가 graph에서 조각남
   - (d) m이 0.5에 못 미침 (solve 문제)
   - (e) 기타

## 입력

- **LIBERO:** SL9 smoke 장면과 같다. task 0 init 0과 libero_spatial 9 장면의 VLA 단독 기록이 있으면 그것을 쓴다.
- **SafeLIBERO:** 20 episode 기록(SL8 · SL7b offline 재생과 같은 입력).
- 판: `AG3S-final-panda-p0m`(명사 L8H6 seed). SL7c ②의 도구(`experiments/segmentation/diag_sl7c_surface_offset.py`: 재생 → view 잡기 → CPU에서 `keep_debug=True`로 선택 재계산)를 확장해 쓴다.

## 출력

- 원시: storage `results/benchmark/<날짜_commit>_sl9a-*`
- 보기: `outputs/2026-11-04_SL9a_표면영역작음_원인/`
  - call별 overlay: seed patch, 영역, GT target, m 열지도
  - 분류 막대
  - 표 · `summary.json`
- GPU는 shared lock만 쓴다. exclusive closed loop(SL9 B)가 시작되기 전에 GPU 부분을 끝낸다. 나머지는 CPU에서 한다.
