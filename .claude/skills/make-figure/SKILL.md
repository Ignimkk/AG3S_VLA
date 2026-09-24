---
name: make-figure
description: AG3S 규칙 A 의 figure 3종(실제 씬·그래프·표)을 프로젝트 스타일로 만든다. matplotlib 으로 그림을 그리거나 docs/figures 에 저장할 때 쓴다.
---

# 규칙 A — figure 3종

구현이든 검증이든 모든 스텝은 시각화를 남긴다. 되는 데까지 **셋 다** 만든다.

| | 무엇 | 무엇을 답하나 |
|---|---|---|
| 1 | **실제 씬** — MuJoCo 장면·depth·ESDF 단면·로봇 구를 실제 좌표에 겹쳐 그린 것 | 무엇이 **어디서** 일어나는가 |
| 2 | **그래프** — 프레임별 추이, 분포, 계층 간 차이 | **얼마나**, 어떤 추세로 |
| 3 | **표·수식·도식** — 나란히 놓은 숫자, 제약 식, 파이프라인 지도 | **무엇과 무엇을** 비교하나 |

**단면을 그릴 때는 어디를 어느 방향에서 잘랐는지 배치도로 먼저 보인다.**
숫자를 문장 안에 흩어 놓는 것은 시각화가 아니다.

## 스타일 — 눈으로 색을 다시 고르지 않는다

```python
import sys; sys.path.insert(0, "/mnt/dev/work")
from benchmark.ag3s.experiments.common import figstyle

figstyle.use_korean()          # False 면 CJK 폰트가 없다 → 영어 라벨로 간다
fig, axes = plt.subplots(1, 2)
figstyle.style_axes(fig, axes) # 표면색·눈금·spine 을 한 번에
```

- `figstyle.CATEGORICAL` — **정체**(어느 물체·어느 계층)는 슬롯 **순서대로** 배정한다.
  순서가 색각 이상 안전장치다. 차트마다 다시 섞지 않는다.
- `figstyle.SEQ_BLUE` / `figstyle.sequential_cmap()` — **크기**(거리·비율)는 단일 파랑 램프.
- `SURFACE` `INK` `INK_2` `GRID_INK` — 배경과 글자.

## 함정

- **matplotlib 은 `**굵게**` 를 렌더하지 않는다.** 제목·라벨에 별표를 쓰지 않는다.
- `use_korean()` 이 `False` 면 **네모 박스를 그리지 말고 영어 라벨로 간다.** 폰트가 있는데도
  못 찾으면 `matplotlib.font_manager.fontManager.addfont(<Noto CJK 경로>)` 를 먼저 부른다.
- CJK face 에는 제대로 된 마이너스 글리프가 없다 — `use_korean()` 이
  `axes.unicode_minus = False` 를 같이 세운다. 직접 폰트를 세울 때는 이것도 같이 한다.

## 저장과 전달

- 경로: `benchmark/ag3s/docs/figures/` (live-test 계열은 `figures/live-test/`).
- **같은 이름의 `.json` sidecar 에 그 그림이 쓴 숫자를 남긴다.**
  기존 관례: `a1-static-wiring.png` ↔ `a1-static-wiring.json`. 그림만 있고 숫자가 없으면 재현이 안 된다.
- 만든 그림은 **`SendUserFile` 로 사용자에게 보낸다.** 로그에서는 경로로 링크한다.
