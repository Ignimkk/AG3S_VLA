# 로봇 제거와 대상 단서의 분리 실험 — 개발 규약

**질문:** 이전 Object 개발 실험에서 분할이 낮았던 이유를 좁히기 위해, 대상 단서와 로봇 제거를 서로 독립적으로 바꾸면 target surface 복원이 어떻게 변하는가? 정답 robot mask 효과가 실제 관절 상태와 로봇 기하로 만든 마스크에서도 나타나는가?

이 규약은 결과 계산 전에 기록한다. 다만 같은 Object 개발 자료와 기존 결과를 이미 관찰했으므로 사전 등록한 독립 test는 아니다. 새로운 자료나 개선된 AG3S 전체 시스템의 성능을 주장하지 않는다.

- 기존 LIBERO Object task 0–9, 각 init 0, 10 episode의 281 policy-call frame을 모두 사용한다. 프레임 목록과 π0.5 noun attention L8H6을 고정한다.
- 두 단서 × 세 로봇 조건을 동일 프레임에서 교차한다: noun attention 또는 이미지 시점 GT target의 16×16 patch coverage; robot mask 없음, Panda 관절 상태와 기존 self-filter model에서 계산한 FK 구 마스크, 이미지 시점 GT robot mask. 두 GT 입력은 특권 진단이며 평가기 정답 이외에 모델로 공급되므로 실사용 성능이 아니다.
- FK 마스크는 관측 depth·카메라 기하·이미지 시점 9관절 상태를 이용하여 기존 Panda self-filter를 호출한다. 기본 PointCloudConfig, 구 inflation 0, guard 비활성, CPU KD-tree를 사용한다. 관절·기하·camera calibration이 제공된다는 가정이며 serving 통합 parity나 실제 로봇 성능은 주장하지 않는다.
- 기존 graph selector 설정을 그대로 호출하고 두 camera에 dummy right camera를 추가한다. 매 프레임 독립 평가, selector abstain은 빈 예측이다. 작업 행동, fallback, tracking은 실행하지 않는다.
- 모든 조건에서 이미지 시점 simulator target/robot GT를 평가에만 사용한다. 각 조건의 동일한 두 RGB-D view에 대해 5 mm soft-voxel IoU/recall, 이웃 FP, abstain 비율을 프레임→episode→task 평균과 함께 기록한다. 로봇 mask 자체의 pixel TP/FP/FN과 target pixel 삭제율도 관측 camera별로 기록한다. 로봇 마스크의 micro 합계와 task 평균을 구분한다.
- 동일 frame의 두 조건 차이를 task 단위 paired bootstrap(10,000회, seed 20261101)으로 기술한다. task당 episode가 1개이고 이미 본 개발 자료이므로 이 구간은 탐색적이며 확증 검정이 아니다. 특권 조건의 차이를 개별 모듈의 독립적인 인과 효과나 상한으로 단정하지 않는다.
- 프레임/카메라별 GT 정합, 더미 카메라 제외, noun+mask 없음 조건과 이전 기록의 일치, 예외 및 누락 유무를 검사한다. 결과 기반의 parameter 선택은 하지 않는다.
