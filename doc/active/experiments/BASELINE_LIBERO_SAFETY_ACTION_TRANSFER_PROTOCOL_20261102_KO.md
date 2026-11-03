# LIBERO-Safety 연속 동작 stress 실험 규약

## 질문과 고정 조건

LIBERO-Safety의 서로 다른 장면에서도 외부 분할 모델의 대상 식별과 연속 추적을 진단한다. 사용 가능한 해당 benchmark의 기록된 전용 policy rollout을 발견하지 못했으므로, 사용자에게 범위를 숨기지 않고 **다른 LIBERO task에서 실제 정책이 생성해 저장한 action을 그대로 옮겨 실행하는 탐색적 stress 시험**으로 한정한다. 이는 LIBERO-Safety task에 맞는 정책 rollout이나 benchmark 성공률 실험이 아니다.

결과를 보기 전에 Safety의 `affordance`, `human_safety`, `obstacle_avoidance` suite 각각 level 0 task 0 init 0, seed 7을 고정한다. 각각 기존 `knows_p0b/libero_spatial_task0/1/2_ep0.npz`의 모든 저장 action을 순서대로 실행한다. 각 action 직전 observation을 기록하고 환경이 done을 반환하면 해당 episode를 종료한다. 각 환경에서 10개 dummy step으로 초기 안정화한 뒤에 시작한다. action 범위를 바꾸거나 정답 객체 위치를 제어에 사용하지 않는다. 대상은 simulator BDDL goal에서 GT 전용으로 추출하고 각 모델 입력은 실제 지시문에서 추출한 phrase와 RGB다.

## 정답과 평가

매 step마다 동일 시뮬레이터 상태의 head RGB·visible target mask·target-only in-frame mask를 수집한다. 다른 geom을 일시적으로 숨기는 과정 전후 상태와 복구한 영상의 일치, visible이 target-only의 부분집합인지 확인한다. 이미지와 GT는 같은 좌표변환으로 224 해상도에 맞춘다. 평가 arm 및 IoU·경계·가림 구간·재등장 지표는 [연속 rollout 규약](BASELINE_CONTINUOUS_ROLLOUT_PROTOCOL_20261102_KO.md)과 [SafeLIBERO 가림 규약](BASELINE_SAFELIBERO_ROLLOUT_PROTOCOL_20261102_KO.md)을 따른다.

action의 출처와 전이 실험이라는 점, task 종료 및 실제 visibility 이벤트 수를 기록한다. 표본 세 개의 기술 통계이며 카메라·task·정책 분포의 대표 성능이나 실시간 성능을 주장하지 않는다. ATS는 사용하지 않는다.
