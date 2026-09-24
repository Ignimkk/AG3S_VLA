---
paths:
  - "benchmark/ag3s/docs/figures/**"
  - "benchmark/ag3s/experiments/**"
  - "outputs/verify/**"
---

# 그림 규칙 (규칙 A)

**모든 구현·검증 스텝은 시각화를 남긴다.** 가장 좋은 것은 셋을 다 만드는 것이다.

1. **실제 씬** — MuJoCo 장면, depth, ESDF 단면, 로봇 구를 실제 좌표에 겹쳐 그린 것.
   *무엇이 어디서 일어나는가.* 단면은 **어디를 어느 방향에서 잘랐는지 배치도로 먼저 보인다.**
2. **그래프** — 프레임별 추이, 분포, 계층 간 차이. *얼마나, 어떤 추세로.*
3. **표·수식·도식** — 숫자를 나란히 놓은 표, 제약 식, 파이프라인 지도. *무엇과 무엇을 비교하나.*

## 기계적으로 지킬 것

- 스타일은 `benchmark/ag3s/experiments/common/figstyle.py` 하나에서 온다.
  색을 눈으로 다시 고르지 않는다: 정체(어느 물체)는 `CATEGORICAL` 을 **슬롯 순서대로**,
  크기(비율·거리)는 `SEQ_BLUE` 단일 램프.
- **한글은 `figstyle.use_korean()` 이 `False` 를 돌려주면 영어 라벨로 간다** — 네모 박스를
  그리지 않는다. Noto CJK 가 없으면 `fm.fontManager.addfont(...)` 를 먼저 부른다.
- **matplotlib 은 `**굵게**` 를 렌더하지 않는다.** 제목·라벨에 별표를 쓰지 않는다.
- 그림마다 같은 이름의 `.json` sidecar 에 그 그림이 쓴 숫자를 남긴다 (기존 관례:
  `a1-static-wiring.png` ↔ `a1-static-wiring.json`). 그림만 있고 숫자가 없으면 재현이 안 된다.
- 저장 위치는 `benchmark/ag3s/docs/figures/`. 만든 그림은 **`SendUserFile` 로 사용자에게 보낸다.**
